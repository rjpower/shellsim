//! Python-source compatibility suites exercised through shellsim's bounded pytest convention.
//!
//! Rust owns VFS installation and process-level assertions. Language semantics stay in `.py`
//! files that can also run under a real CPython pytest when that optional reference is available.

use std::path::PathBuf;
use std::process::Command;

use shellsim::Environment;

const OBJECT_MODEL_SUITE: &[u8] =
    include_bytes!("fixtures/python/object_model/test_object_model.py");

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
        "CPython pytest failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
