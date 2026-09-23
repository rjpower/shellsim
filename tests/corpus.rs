//! Integration coverage for frozen corpus manifests and the stock environment profile.

use std::path::PathBuf;
use std::process::Command;

use shellsim::corpus::{self, ResultClass};

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/stock-smoke")
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
