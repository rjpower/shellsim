//! Python semantic suites executed unchanged by shellsim and an available CPython 3.14.
//!
//! Rust owns VFS installation and process-level assertions. Each source file owns its Python
//! assertions, so adding a case does not require updating a Rust stdout snapshot.

use std::path::PathBuf;
use std::process::Command;

use shellsim::Environment;

const BUILTINS: &[u8] = include_bytes!("test_builtins.py");
const COLLECTIONS: &[u8] = include_bytes!("test_collections.py");
const ASYNCIO: &[u8] = include_bytes!("test_asyncio.py");
const CONTEXTLIB: &[u8] = include_bytes!("test_contextlib.py");
const IMPORTLIB: &[u8] = include_bytes!("test_importlib.py");
const INSPECT: &[u8] = include_bytes!("test_inspect.py");
const LANGUAGE: &[u8] = include_bytes!("test_language.py");
const EXCEPTIONS: &[u8] = include_bytes!("test_exceptions.py");
const FRACTIONS: &[u8] = include_bytes!("test_fractions.py");
const OBJECT_MODEL: &[u8] = include_bytes!("test_object_model.py");
const TYPE_PROTOCOL: &[u8] = include_bytes!("test_type_protocol.py");
const OPERATOR: &[u8] = include_bytes!("test_operator.py");
const RANDOM: &[u8] = include_bytes!("test_random.py");
const STDLIB_SURFACE: &[u8] = include_bytes!("test_stdlib_surface.py");
const STDLIB_IO_PATHS: &[u8] = include_bytes!("test_stdlib_io_paths.py");
const WARNINGS: &[u8] = include_bytes!("test_warnings.py");

fn assert_source_suite(name: &str, source: &[u8]) {
    let mut environment = Environment::new();
    let simulated_path = format!("/tests/{name}");
    environment
        .vfs
        .put_file(&simulated_path, source.to_vec(), 0o644)
        .expect("install Python source suite");

    let (outcome, stdout, stderr) =
        environment.run_script_capture(&format!("python3.14 -m pytest {simulated_path}"));
    assert_eq!(
        outcome.exit_status,
        0,
        "shellsim suite {name} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr),
    );
    assert!(stderr.is_empty(), "{}", String::from_utf8_lossy(&stderr));

    let host_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/python")
        .join(name);
    let reference = Command::new("python3.14")
        .args([
            "-c",
            "import runpy, sys; ns = runpy.run_path(sys.argv[1]); [value() for name, value in ns.items() if name.startswith('test_')]",
        ])
        .arg(host_path)
        .output();
    match reference {
        Ok(output) => assert!(
            output.status.success(),
            "CPython reference {name} failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("could not run CPython reference {name}: {error}"),
    }
}

#[test]
fn builtins() {
    assert_source_suite("test_builtins.py", BUILTINS);
}

#[test]
fn asyncio() {
    assert_source_suite("test_asyncio.py", ASYNCIO);
}

#[test]
fn collections() {
    assert_source_suite("test_collections.py", COLLECTIONS);
}

#[test]
fn contextlib() {
    assert_source_suite("test_contextlib.py", CONTEXTLIB);
}

#[test]
fn importlib() {
    assert_source_suite("test_importlib.py", IMPORTLIB);
}

#[test]
fn inspect() {
    assert_source_suite("test_inspect.py", INSPECT);
}

#[test]
fn language() {
    assert_source_suite("test_language.py", LANGUAGE);
}

#[test]
fn exceptions() {
    assert_source_suite("test_exceptions.py", EXCEPTIONS);
}

#[test]
fn fractions() {
    assert_source_suite("test_fractions.py", FRACTIONS);
}

#[test]
fn object_model() {
    assert_source_suite("test_object_model.py", OBJECT_MODEL);
}

#[test]
fn type_protocol() {
    assert_source_suite("test_type_protocol.py", TYPE_PROTOCOL);
}

#[test]
fn operator() {
    assert_source_suite("test_operator.py", OPERATOR);
}

#[test]
fn random() {
    assert_source_suite("test_random.py", RANDOM);
}

#[test]
fn stdlib_surface() {
    assert_source_suite("test_stdlib_surface.py", STDLIB_SURFACE);
}

#[test]
fn stdlib_io_paths() {
    assert_source_suite("test_stdlib_io_paths.py", STDLIB_IO_PATHS);
}

#[test]
fn warnings() {
    assert_source_suite("test_warnings.py", WARNINGS);
}
