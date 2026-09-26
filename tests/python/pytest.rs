//! Pytest entrypoint, collection, reporting, and explicit unsupported-frontier behavior.

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
fn accepts_common_presentation_only_flags() {
    let source = "def test_ok():\n    pass\n";
    let (status, stdout, stderr) = run_pytest(
        source,
        "pytest -q --disable-warnings --tb=short /test_sample.py",
    );
    assert_eq!(status, 0, "{stderr:?}");
    assert_eq!(stdout, b"/test_sample.py::test_ok PASSED\n");
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
fn runs_fixtures_parametrization_and_tmp_path() {
    let source = r#"import pytest
events = []

@pytest.fixture
def base():
    return 3

@pytest.fixture()
def doubled(base):
    yield base * 2
    events.append("closed")

@pytest.mark.parametrize("offset, expected", [(1, 7), (2, 8)])
def test_math(doubled, offset, expected, tmp_path):
    assert doubled + offset == expected
    assert tmp_path.exists()

def test_fixture_teardown():
    assert events == ["closed", "closed"]

@pytest.mark.skip(reason="later")
def test_skipped_marker():
    assert False
"#;
    let (status, stdout, stderr) = run_pytest(source, "pytest /test_sample.py");
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        stdout,
        b"/test_sample.py::test_math[0] PASSED\n/test_sample.py::test_math[1] PASSED\n/test_sample.py::test_fixture_teardown PASSED\n/test_sample.py::test_skipped_marker SKIPPED\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn rejects_unknown_flags() {
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

#[test]
fn raises_matches_subclasses_and_exposes_the_caught_error() {
    let source = r#"import pytest

class AppError(ValueError):
    pass

def test_subclass():
    with pytest.raises(ValueError) as info:
        raise AppError("detail")
    assert info.type is AppError
    assert str(info.value) == "detail"

def test_tuple_and_match():
    with pytest.raises((KeyError, TypeError)):
        raise TypeError("either")
    with pytest.raises(ValueError, match="de+tail"):
        raise ValueError("detail")

def test_other_types_propagate():
    try:
        with pytest.raises(KeyError):
            raise TypeError("x")
    except TypeError:
        return
    raise AssertionError("TypeError was swallowed")

def test_missing_exception_fails():
    with pytest.raises(KeyError):
        pass
"#;
    let (status, stdout, stderr) = run_pytest(source, "python3.14 -m pytest /test_sample.py");
    assert_eq!(status, 1, "{stderr:?}");
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "/test_sample.py::test_subclass PASSED\n/test_sample.py::test_tuple_and_match PASSED\n/test_sample.py::test_other_types_propagate PASSED\n/test_sample.py::test_missing_exception_fails FAILED DID NOT RAISE <class 'KeyError'>\n"
    );
}

#[test]
fn parametrize_rows_keep_signed_float_and_complex_literals() {
    let source = r#"import pytest

@pytest.mark.parametrize(("value", "expected"), [(-3, "int"), (2.0, "float"), (-1.5, "float"), (1e20, "float"), (1 + 2j, "complex"), (-2j, "complex")])
def test_type(value, expected):
    assert type(value).__name__ == expected

@pytest.mark.parametrize("value", [-3, 2.0, 1e20, 1 - 2j])
def test_round_trip(value):
    assert value in (-3, 2.0, 1e20, 1 - 2j)
"#;
    let (status, stdout, stderr) = run_pytest(source, "python3.14 -m pytest /test_sample.py");
    assert_eq!(status, 0, "{stderr:?} {stdout:?}");
}
