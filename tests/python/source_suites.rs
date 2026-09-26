//! Python semantic suites executed unchanged by shellsim and an available CPython 3.14.
//!
//! Rust owns VFS installation and process-level assertions. Each source file owns its Python
//! assertions, so adding a case does not require updating a Rust stdout snapshot.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use shellsim::{Environment, Limits};

const BUILTINS: &[u8] = include_bytes!("test_builtins.py");
const ASYNCIO: &[u8] = include_bytes!("test_asyncio.py");
const LANGUAGE: &[u8] = include_bytes!("test_language.py");
const EXCEPTIONS: &[u8] = include_bytes!("test_exceptions.py");
const OBJECT_MODEL: &[u8] = include_bytes!("test_object_model.py");
const COUNT_10_MILLION: &[u8] = include_bytes!("performance/test_count_10_million.py");

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
fn language() {
    assert_source_suite("test_language.py", LANGUAGE);
}

#[test]
fn exceptions() {
    assert_source_suite("test_exceptions.py", EXCEPTIONS);
}

#[test]
fn object_model() {
    assert_source_suite("test_object_model.py", OBJECT_MODEL);
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
