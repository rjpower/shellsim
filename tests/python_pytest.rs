use std::process::Command;

use shellsim::Environment;

fn run_pytest(source: &str, command: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file("/test_sample.py", source.as_bytes().to_vec(), 0o644)
        .expect("install test module");
    let (outcome, stdout, stderr) = environment.run_script_capture(command);
    (outcome.exit_status, stdout, stderr)
}

#[test]
fn collects_zero_argument_tests_in_definition_order() {
    let source = "def test_second():\n    assert 1 == 1\ndef helper():\n    pass\ndef test_first():\n    assert 2 == 2\n";
    let (status, stdout, stderr) = run_pytest(source, "pytest /test_sample.py");
    assert_eq!(status, 0, "{stderr:?}");
    assert_eq!(
        stdout,
        b"/test_sample.py::test_second PASSED\n/test_sample.py::test_first PASSED\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn assert_and_pytest_controls_have_expected_statuses() {
    let source = "import pytest\ndef test_assertion():\n    assert 1 == 2, 'nope'\ndef test_skip():\n    pytest.skip('later')\ndef test_raises():\n    with pytest.raises(ValueError):\n        raise ValueError('bad')\n";
    let (status, stdout, stderr) = run_pytest(source, "python3.14 -m pytest /test_sample.py");
    assert_eq!(status, 1, "{stderr:?}");
    assert_eq!(
        stdout,
        b"/test_sample.py::test_assertion FAILED nope\n/test_sample.py::test_skip SKIPPED\n/test_sample.py::test_raises PASSED\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn rejects_fixture_arguments_decorators_and_unknown_flags() {
    let with_fixture = "def test_needs_fixture(tmp_path):\n    pass\n";
    let (status, _stdout, stderr) = run_pytest(with_fixture, "pytest /test_sample.py");
    assert_eq!(status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("fixtures are unsupported"));

    let decorated = "@fixture\ndef test_decorated():\n    pass\n";
    let (status, _stdout, stderr) = run_pytest(decorated, "pytest /test_sample.py");
    assert_eq!(status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("decorators are unsupported"));

    let (status, _stdout, stderr) = run_pytest(
        "def test_ok():\n    pass\n",
        "pytest --bogus /test_sample.py",
    );
    assert_eq!(status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("pytest option --bogus"));
}

#[test]
fn no_tests_is_a_nonzero_collection_result() {
    let (status, stdout, stderr) =
        run_pytest("def helper():\n    pass\n", "pytest /test_sample.py");
    assert_eq!(status, 5);
    assert!(stdout.is_empty());
    assert_eq!(stderr, b"pytest: no tests collected\n");
}

#[test]
fn ctrf_option_writes_a_bounded_report_to_the_vfs() {
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file(
            "/test_sample.py",
            b"def test_ok():\n    pass\n".to_vec(),
            0o644,
        )
        .unwrap();
    let (outcome, stdout, stderr) =
        environment.run_script_capture("pytest --ctrf /logs/verifier/report.json /test_sample.py");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"/test_sample.py::test_ok PASSED\n");
    let report = environment
        .vfs
        .read_string("/", "/logs/verifier/report.json")
        .unwrap();
    assert!(report.contains("\"tests\":1"), "{report}");
    assert!(report.contains("\"passed\":1"), "{report}");
}

#[test]
fn assert_failure_shape_matches_cpython() {
    let source = "assert 1 == 2, 'nope'\n";
    let Ok(probe) = Command::new("python3.14")
        .args([
            "-c",
            "import sys; print(sys.version_info.major, sys.version_info.minor)",
        ])
        .output()
    else {
        return;
    };
    if !probe.status.success() || probe.stdout != b"3 14\n" {
        return;
    }
    let (status, _stdout, stderr) = run_pytest(source, "python3.14 /test_sample.py");
    let Ok(reference) = Command::new("python3.14").arg("-c").arg(source).output() else {
        return;
    };
    if !reference.status.success() && reference.status.code().is_none() {
        return;
    }
    assert_ne!(status, 0);
    assert_ne!(reference.status.code().unwrap_or(1), 0);
    assert!(String::from_utf8_lossy(&stderr).contains("AssertionError"));
    assert!(String::from_utf8_lossy(&reference.stderr).contains("AssertionError"));
}

#[test]
fn rejects_an_oversized_vfs_test_file_before_collection() {
    let mut environment = Environment::new();
    let source = "x = 1\n".repeat(50_000);
    environment
        .vfs
        .put_file("/too_large.py", source.into_bytes(), 0o644)
        .expect("install oversized test module");
    let (outcome, stdout, stderr) = environment.run_script_capture("pytest /too_large.py");
    assert_eq!(outcome.exit_status, 2);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("file too large"));
}

#[test]
fn caps_combined_vfs_source_before_parsing() {
    let mut environment = Environment::new();
    let source = "def test_ok():\n    pass\n".repeat(8_000);
    environment
        .vfs
        .put_file("/one.py", source.as_bytes().to_vec(), 0o644)
        .expect("install first test module");
    environment
        .vfs
        .put_file("/two.py", source.as_bytes().to_vec(), 0o644)
        .expect("install second test module");
    environment
        .vfs
        .put_file("/three.py", source.into_bytes(), 0o644)
        .expect("install third test module");
    let (outcome, stdout, stderr) =
        environment.run_script_capture("pytest /one.py /two.py /three.py");
    assert_eq!(outcome.exit_status, 2);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("combined source"));
}

#[test]
fn caps_generated_pytest_wrapper_before_execution() {
    let mut environment = Environment::new();
    let source = "def test_ok():\n    pass\n".repeat(7_000);
    environment
        .vfs
        .put_file("/many.py", source.into_bytes(), 0o644)
        .expect("install many-test module");
    let (outcome, stdout, stderr) = environment.run_script_capture("pytest /many.py");
    assert_eq!(outcome.exit_status, 2);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("generated wrapper"));
}
