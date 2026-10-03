//! Python semantic suites executed unchanged by shellsim and an available CPython 3.14.
//!
//! Rust owns VFS installation and process-level assertions. Each source file owns its Python
//! assertions, so adding a case does not require updating a Rust stdout snapshot.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use shellsim::{Environment, Limits};

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

/// Manual lookup-cost probe. Each case has a fixed workload, so modeled CPU is comparable across
/// revisions; elapsed host time is informational and should be measured in a release build.
#[test]
#[ignore = "manual release-mode type-dispatch probe"]
fn type_dispatch_benchmark() {
    let cases = [
        (
            "exact sequence",
            "items = [1, 2, 3]\ntotal = 0\nfor _ in range(20_000):\n    total += len(items) + items[0]\n    for item in items:\n        total += item\nassert total == 200000\n",
        ),
        (
            "builtin subclass",
            "class Items(list):\n    pass\nitems = Items([1, 2, 3])\ntotal = 0\nfor _ in range(20_000):\n    total += len(items) + items[0]\nassert total == 80000\n",
        ),
        (
            "bound method",
            "class Counter:\n    def value(self):\n        return 3\nobj = Counter()\ntotal = 0\nfor _ in range(20_000):\n    total += obj.value()\nassert total == 60000\n",
        ),
        (
            "deep MRO",
            "class Root:\n    value = 3\nclass One(Root):\n    pass\nclass Two(One):\n    pass\nclass Three(Two):\n    pass\nobj = Three()\ntotal = 0\nfor _ in range(20_000):\n    total += obj.value\nassert total == 60000\n",
        ),
        (
            "dynamic miss",
            "class Holder:\n    pass\nobj = Holder()\nname = 'missing'\ntotal = 0\nfor _ in range(20_000):\n    total += getattr(obj, name, 0)\nassert total == 0\n",
        ),
        (
            "metaclass descriptor",
            "class Meta(type):\n    @property\n    def marker(cls):\n        return 7\nclass Subject(metaclass=Meta):\n    pass\ntotal = 0\nfor _ in range(20_000):\n    total += Subject.marker\nassert total == 140000\n",
        ),
        (
            "arithmetic",
            "value = 1\nfor _ in range(20_000):\n    value += 1\nassert value == 20001\n",
        ),
        (
            "ndarray in-place",
            "import numpy as np\nvalues = np.array([1, 2, 3])\nfor _ in range(2000):\n    values += 1\nassert values[0] == 2001\n",
        ),
    ];
    for (name, source) in cases {
        let mut environment = Environment::with_limits(Limits::unlimited());
        let path = "/tests/type_dispatch_benchmark.py";
        environment
            .vfs
            .put_file(path, source.as_bytes().to_vec(), 0o644)
            .expect("install type-dispatch probe");
        let started = Instant::now();
        let (outcome, _, stderr) = environment.run_script_capture(&format!("python3.14 {path}"));
        assert_eq!(
            outcome.exit_status,
            0,
            "{name}: {}",
            String::from_utf8_lossy(&stderr),
        );
        eprintln!(
            "{name}: {:.3} s, {} modeled CPU units",
            started.elapsed().as_secs_f64(),
            outcome.usage.cpu_used,
        );
    }
}
