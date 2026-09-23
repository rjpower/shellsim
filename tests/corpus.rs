//! Integration coverage for frozen corpus manifests and the stock environment profile.

use std::path::PathBuf;
use std::process::Command;

use shellsim::corpus::{self, ResultClass};

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/stock-smoke")
}

fn corpus_root(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/corpus")
        .join(name)
}

fn run_frozen_corpus(name: &str) -> corpus::CorpusReport {
    let root = corpus_root(name);
    let manifest = std::fs::read(root.join("manifest.json")).expect("manifest fixture");
    corpus::run_manifest_bytes(&root, &manifest).expect("valid corpus")
}

#[test]
fn stock_manifest_runs_shell_python_and_explicit_skip_cases() {
    let root = fixture_root();
    let manifest = std::fs::read(root.join("manifest.json")).expect("manifest fixture");
    let report = corpus::run_manifest_bytes(&root, &manifest).expect("valid corpus");

    assert_eq!(report.total, 3);
    assert_eq!(report.expectation_failures, 0);
    assert_eq!(report.classes.get(&ResultClass::Pass), Some(&2));
    assert_eq!(report.classes.get(&ResultClass::Skipped), Some(&1));
}

#[test]
fn corpus_cli_emits_a_machine_readable_report() {
    let output = Command::new(env!("CARGO_BIN_EXE_shellsim"))
        .arg("corpus")
        .arg(fixture_root().join("manifest.json"))
        .output()
        .expect("run shellsim corpus");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).expect("JSON report");
    assert_eq!(report["version"], 1);
    assert_eq!(report["profile"], "stock_agent_v1");
    assert_eq!(report["total"], 3);
    assert_eq!(report["expectation_failures"], 0);
    assert_eq!(report["classes"]["pass"], 2);
    assert_eq!(report["classes"]["skipped"], 1);
}

#[test]
fn corpus_checks_generated_file_contents() {
    let manifest = br#"{
        "version": 1,
        "profile": "stock_agent_v1",
        "cases": [{
            "id": "generated-file",
            "kind": "shell",
            "code": "printf 'actual\\n' > /work/result",
            "expect": {"disposition": "pass", "files": {"/work/result": "expected\n"}}
        }]
    }"#;
    let report = corpus::run_manifest_bytes(&fixture_root(), manifest).expect("valid manifest");

    assert_eq!(report.expectation_failures, 1);
    assert_eq!(report.cases[0].class, ResultClass::SemanticMismatch);
    assert!(report.cases[0]
        .detail
        .as_deref()
        .is_some_and(|detail| detail.contains("/work/result")));
}

#[test]
fn setup_failures_are_reported_without_panicking() {
    let manifest = br#"{
        "version": 1,
        "profile": "stock_agent_v1",
        "cases": [{
            "id": "escape",
            "kind": "shell",
            "entrypoint": "/work/nope.sh",
            "fixtures": [{"source": "../nope", "destination": "/work/nope.sh"}],
            "expect": {"disposition": "pass"}
        }]
    }"#;
    let report = corpus::run_manifest_bytes(&fixture_root(), manifest).expect("valid manifest");

    assert_eq!(report.expectation_failures, 1);
    assert_eq!(report.cases[0].class, ResultClass::SetupFailure);
}

#[test]
fn a_profile_that_does_not_fit_is_a_setup_failure() {
    let manifest = br#"{
        "version": 1,
        "profile": "stock_agent_v1",
        "cases": [{
            "id": "tiny-disk",
            "kind": "shell",
            "entrypoint": "/work/nope.sh",
            "limits": {"cpu": 1000, "memory": 1048576, "disk": 0, "output": 1024},
            "expect": {"disposition": "pass"}
        }]
    }"#;
    let report = corpus::run_manifest_bytes(&fixture_root(), manifest).expect("valid manifest");

    assert_eq!(report.expectation_failures, 1);
    assert_eq!(report.cases[0].class, ResultClass::SetupFailure);
}

#[test]
fn imported_shell_and_python_corpora_match_checked_behavior() {
    for (name, minimum_cases) in [
        ("oils-spec", 50),
        ("micropython-basics", 50),
        ("python-derived", 30),
        ("posix-derived", 8),
        ("whole-programs", 19),
    ] {
        let report = run_frozen_corpus(name);
        assert_eq!(
            report.expectation_failures,
            0,
            "{name}: {:?}",
            report
                .cases
                .iter()
                .filter(|case| !case.expectation_met)
                .map(|case| (&case.id, case.class, &case.detail))
                .collect::<Vec<_>>()
        );
        assert!(
            report.total >= minimum_cases,
            "{name} should retain its semantic breadth"
        );
    }
}

#[test]
fn an_expected_unknown_command_can_be_part_of_a_pass_case() {
    let root = fixture_root();
    let manifest = br#"{
        "version": 1,
        "profile": "stock_agent_v1",
        "cases": [{
            "id": "checked-command-not-found",
            "kind": "shell",
            "entrypoint": "/work/missing.sh",
            "fixtures": [{"source": "missing.sh", "destination": "/work/missing.sh"}],
            "expect": {
                "disposition": "pass",
                "exit_status": 127,
                "unsupported": ["definitely-missing"],
                "unsupported_commands": ["definitely-missing"]
            }
        }]
    }"#;
    let report = corpus::run_manifest_bytes(&root, manifest).expect("valid manifest");

    assert_eq!(
        report.expectation_failures,
        0,
        "class={:?} detail={:?} unsupported={:?} commands={:?}",
        report.cases[0].class,
        report.cases[0].detail,
        report.cases[0].unsupported,
        report.cases[0].unsupported_commands
    );
    assert_eq!(report.cases[0].class, ResultClass::Pass);
}

#[test]
fn command_cases_can_use_a_modeled_module_entrypoint() {
    let root = fixture_root();
    let manifest = br#"{
        "version": 1,
        "profile": "stock_agent_v1",
        "cases": [{
            "id": "module-entrypoint",
            "kind": "command",
            "entrypoint": "python3.14",
            "args": ["-c", "print('module command')"],
            "expect": {
                "disposition": "pass",
                "stdout_base64": "bW9kdWxlIGNvbW1hbmQK",
                "stderr_base64": ""
            }
        }]
    }"#;
    let report = corpus::run_manifest_bytes(&root, manifest).expect("valid manifest");

    assert_eq!(report.expectation_failures, 0);
    assert_eq!(report.cases[0].class, ResultClass::Pass);
}

#[test]
fn derived_cases_support_inline_programs_fixtures_and_readable_io() {
    let root = fixture_root();
    let manifest = br#"{
        "version": 1,
        "profile": "stock_agent_v1",
        "cases": [{
            "id": "inline-python",
            "kind": "python",
            "code": "from helper import decorate\nprint(decorate(input()))",
            "covers": ["python.import.local", "python.io.stdio"],
            "stdin": "world\n",
            "fixtures": [{
                "contents": "def decorate(value):\n    return f'<{value}>'\n",
                "destination": "/work/helper.py"
            }],
            "expect": {
                "disposition": "pass",
                "stdout": "<world>\n",
                "stderr": ""
            }
        }]
    }"#;
    let report = corpus::run_manifest_bytes(&root, manifest).expect("valid manifest");

    assert_eq!(
        report.expectation_failures, 0,
        "{:?}",
        report.cases[0].detail
    );
    assert_eq!(report.coverage["python.import.local"].passing, 1);
    assert_eq!(report.coverage["python.io.stdio"].passing, 1);
}

#[test]
fn frontier_cases_require_an_exact_failure_class() {
    let manifest = br#"{
        "version": 1,
        "profile": "stock_agent_v1",
        "cases": [{
            "id": "ambiguous-frontier",
            "kind": "python",
            "code": "import unavailable",
            "expect": {"disposition": "frontier"}
        }]
    }"#;
    let error = corpus::run_manifest_bytes(&fixture_root(), manifest).unwrap_err();

    assert!(
        error.contains("must declare an exact result class"),
        "{error}"
    );
}

#[test]
fn corpus_rejects_ambiguous_inline_inputs() {
    let manifest = br#"{
        "version": 1,
        "profile": "stock_agent_v1",
        "cases": [{
            "id": "ambiguous-input",
            "kind": "shell",
            "entrypoint": "/work/script.sh",
            "code": "true",
            "expect": {"disposition": "pass"}
        }]
    }"#;
    let error = corpus::run_manifest_bytes(&fixture_root(), manifest).unwrap_err();

    assert!(error.contains("both entrypoint and code"), "{error}");
}

#[test]
fn inline_shell_arguments_start_at_one() {
    let manifest = br#"{
        "version": 1,
        "profile": "stock_agent_v1",
        "cases": [{
            "id": "inline-shell-arguments",
            "kind": "shell",
            "code": "printf '%s\\n' \"$1\"",
            "args": ["value"],
            "expect": {
                "disposition": "pass",
                "stdout": "value\n",
                "stderr": ""
            }
        }]
    }"#;
    let report = corpus::run_manifest_bytes(&fixture_root(), manifest).expect("valid manifest");

    assert_eq!(
        report.expectation_failures, 0,
        "{:?}",
        report.cases[0].detail
    );
}
