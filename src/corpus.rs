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
    /// Execute `entrypoint` and `args` through modeled command dispatch.
    Command,
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
    /// Host fixture path relative to the manifest directory.
    #[serde(default)]
    pub source: Option<String>,
    /// Inline UTF-8 fixture contents for compact derived cases.
    #[serde(default)]
    pub contents: Option<String>,
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
    /// Exact UTF-8 stdout. Mutually exclusive with `stdout_base64`.
    #[serde(default)]
    pub stdout: Option<String>,
    #[serde(default)]
    pub stderr_base64: Option<String>,
    /// Exact UTF-8 stderr. Mutually exclusive with `stderr_base64`.
    #[serde(default)]
    pub stderr: Option<String>,
    #[serde(default)]
    pub skip_reason: Option<String>,
    /// Unsupported language features that are part of the checked behavior.
    #[serde(default)]
    pub unsupported: Vec<String>,
    /// Unsupported command names that are part of the checked behavior.
    #[serde(default)]
    pub unsupported_commands: Vec<String>,
    /// Required failure classification for a frontier case.
    #[serde(default)]
    pub class: Option<ResultClass>,
}

/// One program and its fully declared inputs.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusCase {
    pub id: String,
    #[serde(default)]
    pub source: Option<Provenance>,
    pub kind: ProgramKind,
    /// VFS path or modeled command name to execute.
    #[serde(default)]
    pub entrypoint: Option<String>,
    /// Inline shell or Python source. Mutually exclusive with `entrypoint`.
    #[serde(default)]
    pub code: Option<String>,
    /// Stable capability identifiers exercised by this case.
    #[serde(default)]
    pub covers: Vec<String>,
    #[serde(default)]
    pub args: Vec<String>,
    /// UTF-8 standard input. Mutually exclusive with `stdin_base64`.
    #[serde(default)]
    pub stdin: Option<String>,
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
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
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
    pub covers: Vec<String>,
}

/// Aggregate observations for one capability identifier.
#[derive(Debug, Default, Serialize)]
pub struct CoverageResult {
    pub passing: usize,
    pub frontiers: usize,
    pub skipped: usize,
    pub expectation_failures: usize,
}

/// Aggregate output of one manifest run.
#[derive(Debug, Serialize)]
pub struct CorpusReport {
    pub version: u32,
    pub profile: EnvironmentProfile,
    pub total: usize,
    pub expectation_failures: usize,
    pub classes: BTreeMap<ResultClass, usize>,
    pub coverage: BTreeMap<String, CoverageResult>,
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

    validate_manifest(&manifest)?;
    let mut cases = Vec::with_capacity(manifest.cases.len());
    for case in manifest.cases {
        cases.push(run_case(base, manifest.profile, case));
    }
    let mut classes = BTreeMap::new();
    for case in &cases {
        *classes.entry(case.class).or_insert(0) += 1;
    }
    let expectation_failures = cases.iter().filter(|case| !case.expectation_met).count();
    let mut coverage: BTreeMap<String, CoverageResult> = BTreeMap::new();
    for case in &cases {
        for requirement in &case.covers {
            let result = coverage.entry(requirement.clone()).or_default();
            if !case.expectation_met {
                result.expectation_failures += 1;
            } else {
                match case.class {
                    ResultClass::Pass => result.passing += 1,
                    ResultClass::Skipped => result.skipped += 1,
                    _ => result.frontiers += 1,
                }
            }
        }
    }
    Ok(CorpusReport {
        version: FORMAT_VERSION,
        profile: manifest.profile,
        total: cases.len(),
        expectation_failures,
        classes,
        coverage,
        cases,
    })
}

fn validate_manifest(manifest: &CorpusManifest) -> Result<(), String> {
    let mut ids = std::collections::BTreeSet::new();
    for case in &manifest.cases {
        if case.id.is_empty() {
            return Err("case id must not be empty".to_string());
        }
        if !ids.insert(&case.id) {
            return Err(format!("duplicate case id {:?}", case.id));
        }
        match (&case.entrypoint, &case.code) {
            (Some(_), None) | (None, Some(_)) => {}
            (Some(_), Some(_)) => {
                return Err(format!(
                    "case {:?} must not set both entrypoint and code",
                    case.id
                ))
            }
            (None, None) => {
                if case.expect.disposition != ExpectedDisposition::Skip {
                    return Err(format!(
                        "case {:?} must set exactly one of entrypoint or code",
                        case.id
                    ));
                }
            }
        }
        if case.entrypoint.as_ref().is_some_and(String::is_empty) {
            return Err(format!("case {:?} has an empty entrypoint", case.id));
        }
        if case.kind == ProgramKind::Command && case.code.is_some() {
            return Err(format!(
                "command case {:?} cannot execute inline code",
                case.id
            ));
        }
        if case.expect.disposition == ExpectedDisposition::Frontier && case.expect.class.is_none() {
            return Err(format!(
                "frontier case {:?} must declare an exact result class",
                case.id
            ));
        }
        if case.expect.disposition == ExpectedDisposition::Frontier {
            match case.expect.class {
                Some(ResultClass::UnsupportedFeature) if case.expect.unsupported.is_empty() => {
                    return Err(format!(
                        "unsupported-feature frontier {:?} must declare unsupported",
                        case.id
                    ));
                }
                Some(ResultClass::UnknownCommand)
                    if case.expect.unsupported_commands.is_empty() =>
                {
                    return Err(format!(
                        "unknown-command frontier {:?} must declare unsupported_commands",
                        case.id
                    ));
                }
                _ => {}
            }
        }
        if case.expect.stdout.is_some() && case.expect.stdout_base64.is_some() {
            return Err(format!(
                "case {:?} must not set both stdout and stdout_base64",
                case.id
            ));
        }
        if case.expect.stderr.is_some() && case.expect.stderr_base64.is_some() {
            return Err(format!(
                "case {:?} must not set both stderr and stderr_base64",
                case.id
            ));
        }
        if case.stdin.is_some() && !case.stdin_base64.is_empty() {
            return Err(format!(
                "case {:?} must not set both stdin and stdin_base64",
                case.id
            ));
        }
        let mut requirements = std::collections::BTreeSet::new();
        for requirement in &case.covers {
            if requirement.is_empty() || !requirements.insert(requirement) {
                return Err(format!(
                    "case {:?} has an empty or duplicate covers entry",
                    case.id
                ));
            }
        }
        for fixture in &case.fixtures {
            match (&fixture.source, &fixture.contents) {
                (Some(_), None) | (None, Some(_)) => {}
                _ => {
                    return Err(format!(
                        "fixture {:?} in case {:?} must set exactly one of source or contents",
                        fixture.destination, case.id
                    ))
                }
            }
            if fixture.sha256.is_some() && fixture.source.is_none() {
                return Err(format!(
                    "inline fixture {:?} in case {:?} cannot declare sha256",
                    fixture.destination, case.id
                ));
            }
            if fixture
                .contents
                .as_ref()
                .is_some_and(|contents| contents.len() as u64 > MAX_FIXTURE_BYTES)
            {
                return Err(format!(
                    "inline fixture {:?} in case {:?} exceeds the byte limit",
                    fixture.destination, case.id
                ));
            }
        }
    }
    Ok(())
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
            covers: case.covers,
        };
    }

    let stdin = if let Some(stdin) = &case.stdin {
        stdin.as_bytes().to_vec()
    } else {
        match STANDARD.decode(&case.stdin_base64) {
            Ok(stdin) => stdin,
            Err(error) => return setup_failure(&case, format!("invalid stdin_base64: {error}")),
        }
    };
    let mut environment = match profile.create(case.limits.unwrap_or_default()) {
        Ok(environment) => environment,
        Err(error) => return setup_failure(&case, error),
    };
    for fixture in &case.fixtures {
        let data = if let Some(relative) = &fixture.source {
            let source = match safe_fixture_path(base, relative) {
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
                    return setup_failure(
                        &case,
                        format!("cannot read {}: {error}", source.display()),
                    )
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
            data
        } else {
            fixture
                .contents
                .as_deref()
                .unwrap_or_default()
                .as_bytes()
                .to_vec()
        };
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
    let mut command = match (case.kind, &case.entrypoint, &case.code) {
        (ProgramKind::Command, Some(entrypoint), None) => quote_shell(entrypoint),
        (ProgramKind::Shell, Some(entrypoint), None) => {
            format!("bash {}", quote_shell(entrypoint))
        }
        (ProgramKind::Python, Some(entrypoint), None) => {
            format!("python3.14 {}", quote_shell(entrypoint))
        }
        (ProgramKind::Shell, None, Some(code)) => {
            format!("bash -c {} shellsim-corpus", quote_shell(code))
        }
        (ProgramKind::Python, None, Some(code)) => {
            format!("python3.14 -c {}", quote_shell(code))
        }
        _ => unreachable!("manifest validation enforces executable case shape"),
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
        expected_output(
            case.expect.stdout.as_deref(),
            case.expect.stdout_base64.as_deref(),
        )
        .as_deref(),
        &stdout,
        &mut mismatches,
    );
    check_output(
        "stderr",
        expected_output(
            case.expect.stderr.as_deref(),
            case.expect.stderr_base64.as_deref(),
        )
        .as_deref(),
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
    check_string_set(
        "unsupported features",
        &case.expect.unsupported,
        &unsupported,
        &mut mismatches,
    );
    check_string_set(
        "unsupported commands",
        &case.expect.unsupported_commands,
        &unsupported_commands,
        &mut mismatches,
    );
    let mut class = classify(
        &outcome,
        &stderr,
        &unsupported,
        &unsupported_commands,
        &mismatches,
    );
    if case.expect.disposition == ExpectedDisposition::Pass
        && mismatches.is_empty()
        && matches!(
            class,
            ResultClass::UnknownCommand | ResultClass::UnsupportedFeature
        )
    {
        class = ResultClass::Pass;
    }
    if let Some(expected) = case.expect.class {
        if expected != class {
            mismatches.push(format!(
                "expected result class {}, got {}",
                result_class_name(expected),
                result_class_name(class)
            ));
        }
    }
    let expectation_met = match case.expect.disposition {
        ExpectedDisposition::Pass => mismatches.is_empty() && class == ResultClass::Pass,
        ExpectedDisposition::Frontier => class != ResultClass::Pass && mismatches.is_empty(),
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
        covers: case.covers,
    }
}

fn expected_output(text: Option<&str>, encoded: Option<&str>) -> Option<String> {
    text.map(|value| STANDARD.encode(value.as_bytes()))
        .or_else(|| encoded.map(str::to_string))
}

fn result_class_name(class: ResultClass) -> &'static str {
    match class {
        ResultClass::Pass => "pass",
        ResultClass::Skipped => "skipped",
        ResultClass::SetupFailure => "setup_failure",
        ResultClass::ParseOrCompileFailure => "parse_or_compile_failure",
        ResultClass::UnknownCommand => "unknown_command",
        ResultClass::UnsupportedFeature => "unsupported_feature",
        ResultClass::MissingPythonModule => "missing_python_module",
        ResultClass::FixtureAssumption => "fixture_assumption",
        ResultClass::SemanticMismatch => "semantic_mismatch",
        ResultClass::RuntimeFailure => "runtime_failure",
        ResultClass::ResourceExhaustion => "resource_exhaustion",
        ResultClass::HangOrDeadlock => "hang_or_deadlock",
        ResultClass::CapabilityViolation => "capability_violation",
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

fn check_string_set(
    label: &str,
    expected: &[String],
    actual: &[String],
    mismatches: &mut Vec<String>,
) {
    let mut expected = expected.to_vec();
    let mut actual = actual.to_vec();
    expected.sort();
    actual.sort();
    if expected != actual {
        mismatches.push(format!("{label} differ from checked values"));
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
    // The manifest's checked status and output are the observable contract. Diagnostics may
    // intentionally contain words such as "not found", and a non-zero status can be expected.
    if mismatches.is_empty() {
        return ResultClass::Pass;
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
    if outcome.exit_status != 0 {
        return ResultClass::RuntimeFailure;
    }
    ResultClass::SemanticMismatch
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
        covers: case.covers.clone(),
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
