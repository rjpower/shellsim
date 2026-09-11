//! Python-source compatibility suites exercised through shellsim's bounded pytest convention.
//!
//! Rust owns VFS installation and process-level assertions. Language semantics stay in `.py`
//! files that can also run under a real CPython pytest when that optional reference is available.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use shellsim::{Environment, Limits};

const OBJECT_MODEL_SUITE: &[u8] =
    include_bytes!("fixtures/python/object_model/test_object_model.py");
const EASY_WINS_SUITE: &[u8] = include_bytes!("fixtures/python/cpython_basics/test_easy_wins.py");
const COUNT_10_MILLION: &[u8] =
    include_bytes!("fixtures/python/performance/test_count_10_million.py");

#[test]
fn object_model_suite_runs_as_python_source() {
    let mut environment = Environment::new();
    let path = "/tests/test_object_model.py";
    environment
        .vfs
        .put_file(path, OBJECT_MODEL_SUITE.to_vec(), 0o644)
        .expect("install object-model suite");

    let (outcome, stdout, stderr) =
        environment.run_script_capture(&format!("python3.14 -m pytest {path}"));
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(
        stdout,
        b"/tests/test_object_model.py::test_scalar_storage_has_one_semantic_type PASSED\n\
/tests/test_object_model.py::test_binary_operations_dispatch_through_type_slots PASSED\n\
/tests/test_object_model.py::test_descriptors_and_super_share_the_mro PASSED\n\
/tests/test_object_model.py::test_metaclass_hooks_use_the_shared_allocator PASSED\n\
/tests/test_object_model.py::test_json_uses_ordinary_runtime_values PASSED\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn object_model_suite_passes_cpython_pytest_when_available() {
    let available = Command::new("python3.14")
        .args(["-c", "import pytest"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        return;
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/python/object_model/test_object_model.py");
    let output = Command::new("python3.14")
        .args(["-m", "pytest", "-q"])
        .arg(path)
        .output()
        .expect("run CPython pytest reference");
    assert!(
        output.status.success(),
        "CPython reference failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cpython_basic_easy_wins_run_as_python_source() {
    let mut environment = Environment::new();
    let path = "/tests/test_easy_wins.py";
    environment
        .vfs
        .put_file(path, EASY_WINS_SUITE.to_vec(), 0o644)
        .expect("install CPython-basic suite");

    let (outcome, stdout, stderr) =
        environment.run_script_capture(&format!("python3.14 -m pytest {path}"));
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(
        stdout,
        b"/tests/test_easy_wins.py::test_int_accepts_explicit_bases PASSED\n\
/tests/test_easy_wins.py::test_string_join_replace_and_format PASSED\n\
/tests/test_easy_wins.py::test_list_reverse_count_and_index PASSED\n\
/tests/test_easy_wins.py::test_dict_update_accepts_mappings_pairs_and_keywords PASSED\n\
/tests/test_easy_wins.py::test_dict_pop_supports_defaults PASSED\n\
/tests/test_easy_wins.py::test_set_union_accepts_multiple_iterables PASSED\n\
/tests/test_easy_wins.py::test_map_accepts_multiple_iterables PASSED\n\
/tests/test_easy_wins.py::test_filter_accepts_callables_and_none PASSED\n\
/tests/test_easy_wins.py::test_getattr_and_hasattr_share_attribute_lookup PASSED\n\
/tests/test_easy_wins.py::test_reversed_returns_an_iterator PASSED\n\
/tests/test_easy_wins.py::test_invalid_inputs_raise_python_exceptions PASSED\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn cpython_basic_easy_wins_pass_cpython_when_available() {
    let available = Command::new("python3.14")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        return;
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/python/cpython_basics/test_easy_wins.py");
    let output = Command::new("python3.14")
        .args([
            "-c",
            "import runpy, sys; ns = runpy.run_path(sys.argv[1]); [value() for name, value in ns.items() if name.startswith('test_')]",
        ])
        .arg(path)
        .output()
        .expect("run CPython reference");
    assert!(
        output.status.success(),
        "CPython pytest failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "manual release-mode throughput probe"]
fn count_to_ten_million_benchmark() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000_000,
        ..Limits::default()
    });
    let path = "/tests/test_count_10_million.py";
    environment
        .vfs
        .put_file(path, COUNT_10_MILLION.to_vec(), 0o644)
        .expect("install throughput probe");

    let started = Instant::now();
    let (outcome, stdout, stderr) = environment.run_script_capture(&format!("pytest {path}"));
    let elapsed = started.elapsed();

    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(
        stdout,
        b"/tests/test_count_10_million.py::test_count_to_ten_million PASSED\n"
    );
    eprintln!(
        "count-to-10m: {elapsed:.3?}, {} modeled CPU units",
        outcome.usage.cpu_used
    );
}
