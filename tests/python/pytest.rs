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
fn accepts_swe_verifier_flags_and_applies_warning_filters() {
    let source =
        "import warnings\ndef test_warn():\n    warnings.warn('old', DeprecationWarning)\n";
    let command = "pytest --no-header -rA --tb=line --color=no -p no:cacheprovider -W ignore::DeprecationWarning --override-ini=addopts= --continue-on-collection-errors /test_sample.py";
    let (status, stdout, stderr) = run_pytest(source, command);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"/test_sample.py::test_warn PASSED\n");
    assert!(stderr.is_empty());
    let (status, stdout, _) = run_pytest(
        source,
        "pytest -W error::DeprecationWarning /test_sample.py",
    );
    assert_eq!(status, 1);
    assert!(String::from_utf8_lossy(&stdout).contains("FAILED"));
}

#[test]
fn timeouts_fail_only_the_slow_item_using_virtual_cpu_and_sleep() {
    for body in [
        "while True:\n        pass",
        "import time\n    time.sleep(1000)",
    ] {
        let source = format!("def test_slow():\n    {body}\ndef test_after():\n    assert True\n");
        let (status, stdout, stderr) = run_pytest(&source, "pytest --timeout=0.01 /test_sample.py");
        assert_eq!(status, 1, "{}", String::from_utf8_lossy(&stderr));
        let stdout = String::from_utf8_lossy(&stdout);
        assert!(stdout.contains("test_slow FAILED"), "{stdout}");
        assert!(stdout.contains("test_after PASSED"), "{stdout}");
    }
    let (status, _, stderr) = run_pytest(
        "def test_ok():\n    pass\n",
        "pytest --timeout=120 /test_sample.py",
    );
    assert_eq!(status, 0, "{stderr:?}");
    let source = "import time\ndef test_one():\n    time.sleep(0.008)\ndef test_two():\n    time.sleep(0.008)\n";
    let (status, _, stderr) = run_pytest(source, "pytest --timeout=0.01 /test_sample.py");
    assert_eq!(status, 0, "{stderr:?}");
    let (status, _, stderr) = run_pytest(source, "pytest --timeout=0 /test_sample.py");
    assert_eq!(status, 0, "{stderr:?}");
}

#[test]
fn pytest_timeout_cannot_override_the_environment_resource_limit() {
    let mut env = Environment::new();
    env.vfs
        .write(
            "/",
            "/test.py",
            b"def test_loop():\n    while True:\n        pass\n",
            0o644,
        )
        .unwrap();
    env.resources = shellsim::resources::Resources::new(shellsim::Limits {
        cpu: 100_000,
        ..Default::default()
    });
    let (outcome, _, _) = env.run_script_capture("pytest --timeout=120 /test.py");
    assert_eq!(
        outcome.stop_reason,
        Some(shellsim::StopReason::CpuExhausted)
    );
}

#[test]
fn collection_errors_continue_and_discovery_is_virtual() {
    let mut env = Environment::new();
    env.vfs.mkdir_all("/", "/work/tests").unwrap();
    for (name, source) in [
        ("test_bad.py", "def broken(:\n"),
        (
            "test_import.py",
            "raise ValueError('bad import')\ndef test_no():\n    pass\n",
        ),
        ("test_good.py", "def test_ok():\n    assert True\n"),
    ] {
        env.vfs
            .write(
                "/",
                &format!("/work/tests/{name}"),
                source.as_bytes(),
                0o644,
            )
            .unwrap();
    }
    let (outcome, stdout, stderr) =
        env.run_script_capture("cd /work; pytest --continue-on-collection-errors");
    assert_eq!(outcome.exit_status, 1, "{stderr:?}");
    let stdout = String::from_utf8_lossy(&stdout);
    assert!(stdout.contains("test_good.py::test_ok PASSED"), "{stdout}");
    assert!(stdout.contains("test_import.py ERROR"), "{stdout}");
    assert!(!stdout.contains("test_no PASSED"));
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

#[test]
fn parametrize_rows_keep_bytes_literals() {
    let source = r#"import pytest

@pytest.mark.parametrize("value, length", [(b"", 0), (b"a\x00\xff'\"", 5)])
def test_bytes(value, length):
    assert type(value) is bytes
    assert len(value) == length
    assert value in (b"", b"a\x00\xff'\"")
"#;
    let (status, stdout, stderr) = run_pytest(source, "python3.14 -m pytest /test_sample.py");
    assert_eq!(status, 0, "{stderr:?} {stdout:?}");
}

#[test]
fn parametrize_evaluates_arbitrary_expressions_when_the_module_runs() {
    let source = r#"import pytest
CASES = [(1, 2), (3, 4)]
seen = []

@pytest.mark.parametrize("left, right", CASES)
def test_named_cases(left, right):
    assert right - left == 1

@pytest.mark.parametrize("function, expected", [(lambda: 2 * 3, 6), (len, None)])
def test_callables(function, expected):
    assert expected is None or function() == expected

@pytest.mark.parametrize("x", [0, 1])
@pytest.mark.parametrize("y", [2, 3])
def test_stacked(x, y, tmp_path):
    assert tmp_path.exists()
    seen.append((x, y))

def test_stacked_order():
    assert seen == [(0, 2), (1, 2), (0, 3), (1, 3)]

@pytest.mark.parametrize("value", [])
def test_empty(value):
    assert False
"#;
    let (status, stdout, stderr) = run_pytest(source, "pytest /test_sample.py");
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "/test_sample.py::test_named_cases[0] PASSED\n/test_sample.py::test_named_cases[1] PASSED\n\
         /test_sample.py::test_callables[0] PASSED\n/test_sample.py::test_callables[1] PASSED\n\
         /test_sample.py::test_stacked[0] PASSED\n/test_sample.py::test_stacked[1] PASSED\n\
         /test_sample.py::test_stacked[2] PASSED\n/test_sample.py::test_stacked[3] PASSED\n\
         /test_sample.py::test_stacked_order PASSED\n"
    );
}

#[test]
fn parametrize_rows_must_match_their_names() {
    let source = r#"import pytest

@pytest.mark.parametrize("a, b", [(1, 2), (3,)])
def test_pairs(a, b):
    pass
"#;
    let (status, _stdout, stderr) = run_pytest(source, "pytest /test_sample.py");
    assert_ne!(status, 0);
    assert!(
        String::from_utf8_lossy(&stderr).contains(
            "in \"parametrize\" the number of names (2): ('a', 'b') must be equal to the number of values (1): (3,)"
        ),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
}

#[test]
fn warns_checks_category_and_message() {
    let source = r#"import warnings
import pytest

def test_matching_warning():
    with pytest.warns(UserWarning, match="^care") as record:
        warnings.warn("careful")
    assert [str(warning.message) for warning in record.list] == ["careful"]

def test_missing_warning():
    with pytest.warns(DeprecationWarning):
        warnings.warn("careful")
"#;
    let (status, stdout, stderr) = run_pytest(source, "pytest /test_sample.py");
    assert_eq!(status, 1, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "/test_sample.py::test_matching_warning PASSED\n/test_sample.py::test_missing_warning FAILED \
         DID NOT WARN. No warnings of type (<class 'DeprecationWarning'>,) were emitted.\n \
         Emitted warnings: [UserWarning('careful')].\n"
    );
}

#[test]
fn approx_compares_numbers_containers_and_arrays_within_tolerance() {
    let source = r#"import numpy as np
import pytest

def test_scalars():
    assert 0.1 + 0.2 == pytest.approx(0.3)
    assert 0.2 != pytest.approx(0.21)
    assert 1.0005 == pytest.approx(1.0, abs=1e-3)
    assert 1.0005 == pytest.approx(1.0, rel=1e-3, abs=0)
    assert float("nan") != pytest.approx(float("nan"))
    assert float("nan") == pytest.approx(float("nan"), nan_ok=True)
    assert 2.8 + 2.8j == pytest.approx(2.8000001 + 2.8j)
    assert None == pytest.approx(None)
    assert repr(pytest.approx(0.2)) == "0.2 ± 2.0e-07"
    assert repr(pytest.approx(float("nan"))) == "nan ± ???"

def test_containers():
    assert [1, 2] == pytest.approx((1, 2.0000001))
    assert {"a": 1.0} == pytest.approx({"a": 1.0000001})
    assert [1.0, 1.0] != pytest.approx(1.0)
    assert repr(pytest.approx([0.2, 1])) == "approx([0.2 ± 2.0e-07, 1 ± 1.0e-06])"

def test_arrays():
    assert np.array([1.0, 1.0000001]) == pytest.approx(1.0)
    assert np.array([1.0, 1.1]) != pytest.approx(1.0)
    assert np.array([1.0, 2.0]) == pytest.approx([1.0, 2.0])
    assert np.array([[1.0, 2.0]]) != pytest.approx(np.array([1.0, 2.0]))
    assert np.array(0.2) == pytest.approx(0.2)

def test_negative_tolerance():
    with pytest.raises(ValueError, match="absolute tolerance can't be negative: -1"):
        0.2 == pytest.approx(1.0, abs=-1)

def test_nested_sequences():
    with pytest.raises(TypeError, match="does not support nested data structures"):
        pytest.approx([[1.0]])
"#;
    let (status, stdout, stderr) = run_pytest(source, "pytest /test_sample.py");
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stdout));
    assert!(stderr.is_empty(), "{}", String::from_utf8_lossy(&stderr));
}
