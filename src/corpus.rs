//! Frozen compatibility-corpus manifests, execution, and failure classification.
//!
//! The runner reads host files only while preparing a trusted test fixture. Simulated programs
//! receive those bytes through the VFS and retain no ambient host filesystem or network access.

use std::collections::BTreeMap;
use std::path::{Component, Path};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::profile::EnvironmentProfile;
use crate::{CommandTrust, Limits, RunOutcome, StopReason};

/// The only corpus manifest and report schema understood by this build.
pub const FORMAT_VERSION: u32 = 1;
const MAX_CASES: usize = 10_000;
const MAX_FIXTURE_BYTES: u64 = 64 * 1024 * 1024;

/// One frozen collection of compatibility workloads.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusManifest {
    pub version: u32,
    pub profile: EnvironmentProfile,
    pub cases: Vec<CorpusCase>,
}

/// Interpreter used for a corpus entrypoint.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProgramKind {
    Shell,
    Python,
}

/// Intended status of a frozen case.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedDisposition {
    Pass,
    Frontier,
    Skip,
}

/// A host file copied into the virtual filesystem before execution.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub source: String,
    pub destination: String,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default = "default_fixture_mode")]
    pub mode: u16,
}

const fn default_fixture_mode() -> u16 {
    0o644
}

/// Observable contract for one compatibility case.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseExpectation {
    pub disposition: ExpectedDisposition,
    #[serde(default)]
    pub exit_status: Option<i32>,
    #[serde(default)]
    pub stdout_base64: Option<String>,
    #[serde(default)]
    pub stderr_base64: Option<String>,
    #[serde(default)]
    pub skip_reason: Option<String>,
}

/// One program and its fully declared inputs.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusCase {
    pub id: String,
    #[serde(default)]
    pub source: Option<Provenance>,
    pub kind: ProgramKind,
    pub entrypoint: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub stdin_base64: String,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub fixtures: Vec<Fixture>,
    #[serde(default)]
    pub limits: Option<Limits>,
    pub expect: CaseExpectation,
}

/// Pinned origin of an imported workload.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub url: String,
    pub revision: String,
    pub path: String,
    pub sha256: String,
    pub license: String,
}

/// First decisive outcome of a compatibility workload.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ResultClass {
    Pass,
    Skipped,
    SetupFailure,
    ParseOrCompileFailure,
    UnknownCommand,
    UnsupportedFeature,
    MissingPythonModule,
    FixtureAssumption,
    SemanticMismatch,
    RuntimeFailure,
    ResourceExhaustion,
    HangOrDeadlock,
    CapabilityViolation,
}

/// Machine-readable result for one case.
#[derive(Debug, Serialize)]
pub struct CaseResult {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<Provenance>,
    pub expected: ExpectedDisposition,
    pub class: ResultClass,
    pub expectation_met: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<RunOutcome>,
    pub stdout_base64: String,
    pub stderr_base64: String,
    pub unsupported: Vec<String>,
    pub unsupported_commands: Vec<String>,
    pub workspace_changes: Vec<crate::harness::WorkspaceChange>,
}

/// Aggregate output of one manifest run.
#[derive(Debug, Serialize)]
pub struct CorpusReport {
    pub version: u32,
    pub profile: EnvironmentProfile,
    pub total: usize,
    pub expectation_failures: usize,
    pub classes: BTreeMap<ResultClass, usize>,
    pub cases: Vec<CaseResult>,
}

/// Parse and execute a manifest whose fixture paths are relative to `base`.
pub fn run_manifest_bytes(base: &Path, bytes: &[u8]) -> Result<CorpusReport, String> {
    let manifest: CorpusManifest =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid manifest: {error}"))?;
    run_manifest(base, manifest)
}

/// Execute a parsed corpus manifest under fresh isolated environments.
pub fn run_manifest(base: &Path, manifest: CorpusManifest) -> Result<CorpusReport, String> {
    if manifest.version != FORMAT_VERSION {
        return Err(format!(
            "unsupported corpus version {} (expected {FORMAT_VERSION})",
            manifest.version
        ));
    }
    if manifest.cases.len() > MAX_CASES {
        return Err(format!("corpus exceeds the {MAX_CASES}-case limit"));
    }

    let mut cases = Vec::with_capacity(manifest.cases.len());
    for case in manifest.cases {
        cases.push(run_case(base, manifest.profile, case));
    }
    let mut classes = BTreeMap::new();
    for case in &cases {
        *classes.entry(case.class).or_insert(0) += 1;
    }
    let expectation_failures = cases.iter().filter(|case| !case.expectation_met).count();
    Ok(CorpusReport {
        version: FORMAT_VERSION,
        profile: manifest.profile,
        total: cases.len(),
        expectation_failures,
        classes,
        cases,
    })
}

fn run_case(base: &Path, profile: EnvironmentProfile, case: CorpusCase) -> CaseResult {
    if case.expect.disposition == ExpectedDisposition::Skip {
        let valid = case
            .expect
            .skip_reason
            .as_ref()
            .is_some_and(|reason| !reason.is_empty());
        return CaseResult {
            id: case.id,
            source: case.source,
            expected: case.expect.disposition,
            class: ResultClass::Skipped,
            expectation_met: valid,
            detail: case
                .expect
                .skip_reason
                .or_else(|| Some("skipped cases require a non-empty skip_reason".to_string())),
            outcome: None,
            stdout_base64: String::new(),
            stderr_base64: String::new(),
            unsupported: Vec::new(),
            unsupported_commands: Vec::new(),
            workspace_changes: Vec::new(),
        };
    }

    let stdin = match STANDARD.decode(&case.stdin_base64) {
        Ok(stdin) => stdin,
        Err(error) => return setup_failure(&case, format!("invalid stdin_base64: {error}")),
    };
    let mut environment = match profile.create(case.limits.unwrap_or_default()) {
        Ok(environment) => environment,
        Err(error) => return setup_failure(&case, error),
    };
    for fixture in &case.fixtures {
        let source = match safe_fixture_path(base, &fixture.source) {
            Ok(path) => path,
            Err(error) => return setup_failure(&case, error),
        };
        let metadata = match std::fs::metadata(&source) {
            Ok(metadata) => metadata,
            Err(error) => {
                return setup_failure(
                    &case,
                    format!("cannot inspect {}: {error}", source.display()),
                )
            }
        };
        if metadata.len() > MAX_FIXTURE_BYTES {
            return setup_failure(
                &case,
                format!("fixture {} exceeds the byte limit", source.display()),
            );
        }
        let data = match std::fs::read(&source) {
            Ok(data) => data,
            Err(error) => {
                return setup_failure(&case, format!("cannot read {}: {error}", source.display()))
            }
        };
        if let Some(expected) = &fixture.sha256 {
            let actual = format!("{:x}", Sha256::digest(&data));
            if &actual != expected {
                return setup_failure(
                    &case,
                    format!(
                        "fixture {} has sha256 {actual}, expected {expected}",
                        source.display()
                    ),
                );
            }
        }
        if !fixture.destination.starts_with('/') {
            return setup_failure(&case, "fixture destination must be absolute".to_string());
        }
        let parent = Path::new(&fixture.destination)
            .parent()
            .and_then(Path::to_str)
            .unwrap_or("/");
        if let Err(error) = environment.vfs.mkdir_all("/", parent) {
            return setup_failure(
                &case,
                format!("cannot create fixture parent {parent}: {error:?}"),
            );
        }
        if let Err(error) =
            environment
                .vfs
                .put_file(&fixture.destination, data, u32::from(fixture.mode))
        {
            return setup_failure(
                &case,
                format!("cannot install {}: {error:?}", fixture.destination),
            );
        }
    }
    for (name, value) in &case.env {
        environment.set_var(name, value);
        environment.exported.insert(name.clone());
    }
    let workspace_before = environment.vfs.clone();
    let mut command = match case.kind {
        ProgramKind::Shell => format!("bash {}", quote_shell(&case.entrypoint)),
        ProgramKind::Python => format!("python3.14 {}", quote_shell(&case.entrypoint)),
    };
    for argument in &case.args {
        command.push(' ');
        command.push_str(&quote_shell(argument));
    }
    let (outcome, stdout, stderr) = environment.run_script_capture_with_stdin(&command, &stdin);
    let mut mismatches = Vec::new();
    let workspace_changes =
        match crate::harness::workspace_diff(&workspace_before, &environment.vfs) {
            Ok(changes) => changes,
            Err(error) => {
                mismatches.push(error);
                Vec::new()
            }
        };
    let expected_status = case.expect.exit_status.unwrap_or(0);
    if outcome.exit_status != expected_status {
        mismatches.push(format!(
            "expected exit status {expected_status}, got {}",
            outcome.exit_status
        ));
    }
    check_output(
        "stdout",
        case.expect.stdout_base64.as_deref(),
        &stdout,
        &mut mismatches,
    );
    check_output(
        "stderr",
        case.expect.stderr_base64.as_deref(),
        &stderr,
        &mut mismatches,
    );
    let unsupported = environment.unsupported.values();
    let unsupported_commands: Vec<String> = environment
        .invocations
        .events()
        .into_iter()
        .filter(|event| event.trust == CommandTrust::Unsupported)
        .filter_map(|event| event.argv.first().cloned())
        .collect();
    let class = classify(
        &outcome,
        &stderr,
        &unsupported,
        &unsupported_commands,
        &mismatches,
    );
    let expectation_met = match case.expect.disposition {
        ExpectedDisposition::Pass => mismatches.is_empty() && class == ResultClass::Pass,
        ExpectedDisposition::Frontier => class != ResultClass::Pass,
        ExpectedDisposition::Skip => unreachable!(),
    };
    CaseResult {
        id: case.id,
        source: case.source,
        expected: case.expect.disposition,
        class,
        expectation_met,
        detail: (!mismatches.is_empty()).then(|| mismatches.join("; ")),
        outcome: Some(outcome),
        stdout_base64: STANDARD.encode(stdout),
        stderr_base64: STANDARD.encode(stderr),
        unsupported,
        unsupported_commands,
        workspace_changes,
    }
}

fn safe_fixture_path(base: &Path, relative: &str) -> Result<std::path::PathBuf, String> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "fixture source must be a plain relative path: {relative}"
        ));
    }
    let base = base.canonicalize().map_err(|error| {
        format!(
            "cannot resolve corpus directory {}: {error}",
            base.display()
        )
    })?;
    let candidate = base.join(path).canonicalize().map_err(|error| {
        format!(
            "cannot resolve fixture source {}: {error}",
            base.join(path).display()
        )
    })?;
    if !candidate.starts_with(&base) {
        return Err(format!(
            "fixture source escapes the corpus directory: {relative}"
        ));
    }
    Ok(candidate)
}

fn quote_shell(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn check_output(label: &str, expected: Option<&str>, actual: &[u8], mismatches: &mut Vec<String>) {
    let Some(expected) = expected else {
        return;
    };
    match STANDARD.decode(expected) {
        Ok(expected) if expected == actual => {}
        Ok(_) => mismatches.push(format!("{label} differs from checked output")),
        Err(error) => mismatches.push(format!("invalid expected {label}: {error}")),
    }
}

fn classify(
    outcome: &RunOutcome,
    stderr: &[u8],
    unsupported: &[String],
    unsupported_commands: &[String],
    mismatches: &[String],
) -> ResultClass {
    if matches!(
        outcome.stop_reason,
        Some(
            StopReason::CpuExhausted
                | StopReason::MemoryExhausted
                | StopReason::OutputLimitExceeded
        )
    ) {
        return ResultClass::ResourceExhaustion;
    }
    if !unsupported_commands.is_empty() {
        return ResultClass::UnknownCommand;
    }
    let stderr = String::from_utf8_lossy(stderr);
    if stderr.contains("deadlock") || stderr.contains("no runnable tasks") {
        return ResultClass::HangOrDeadlock;
    }
    if !unsupported.is_empty() {
        return ResultClass::UnsupportedFeature;
    }
    if stderr.contains("ModuleNotFoundError") || stderr.contains("ImportError") {
        return ResultClass::MissingPythonModule;
    }
    if stderr.contains("SyntaxError") || stderr.contains("parse error") {
        return ResultClass::ParseOrCompileFailure;
    }
    if stderr.contains("No such file") || stderr.contains("not found") {
        return ResultClass::FixtureAssumption;
    }
    if !mismatches.is_empty() {
        return ResultClass::SemanticMismatch;
    }
    if outcome.exit_status != 0 {
        return ResultClass::RuntimeFailure;
    }
    ResultClass::Pass
}

fn setup_failure(case: &CorpusCase, detail: String) -> CaseResult {
    CaseResult {
        id: case.id.clone(),
        source: case.source.clone(),
        expected: case.expect.disposition,
        class: ResultClass::SetupFailure,
        expectation_met: false,
        detail: Some(detail),
        outcome: None,
        stdout_base64: String::new(),
        stderr_base64: String::new(),
        unsupported: Vec::new(),
        unsupported_commands: Vec::new(),
        workspace_changes: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_fixture_traversal() {
        assert!(safe_fixture_path(Path::new("/tmp/corpus"), "../secret").is_err());
        assert!(safe_fixture_path(Path::new("/tmp/corpus"), "/etc/passwd").is_err());
    }

    #[test]
    fn classifies_missing_modules_before_generic_mismatches() {
        let outcome = RunOutcome {
            exit_status: 1,
            stop_reason: None,
            limits: Limits::default(),
            usage: crate::Usage {
                cpu_used: 0,
                memory_current: 0,
                memory_peak: 0,
                disk_current: 0,
                disk_peak: 0,
                output_bytes: 0,
            },
            command_usage: Vec::new(),
            cost_model_version: crate::resources::COST_MODEL_VERSION,
        };
        assert_eq!(
            classify(
                &outcome,
                b"ModuleNotFoundError: missing",
                &[],
                &[],
                &["exit status".to_string()]
            ),
            ResultClass::MissingPythonModule
        );
    }
}
