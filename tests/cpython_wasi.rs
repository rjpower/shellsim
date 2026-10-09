// These opt-in tests import a trusted, separately built CPython WASI fixture into the VFS.
// Product execution then uses only virtual capabilities. Build with ports/python/cpython/build.py
// and run with CPYTHON_WASI_ROOT=<rootfs> cargo test --test cpython_wasi -- --ignored.
use std::path::Path;

use shellsim::{Environment, Limits};

fn import_tree(environment: &mut Environment, source: &Path, destination: &str) {
    environment.vfs.mkdir_all("/", destination).unwrap();
    let mut entries = std::fs::read_dir(source)
        .unwrap()
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = format!(
            "{}/{}",
            destination.trim_end_matches('/'),
            entry.file_name().to_str().unwrap()
        );
        let kind = entry.file_type().unwrap();
        if kind.is_dir() {
            import_tree(environment, &entry.path(), &path);
        } else {
            assert!(kind.is_file(), "fixture must contain only regular files");
            let bytes = std::fs::read(entry.path()).unwrap();
            let mode = if path.ends_with(".wasm") {
                0o755
            } else {
                0o644
            };
            environment.vfs.write("/", &path, &bytes, mode).unwrap();
        }
    }
}

fn environment() -> Environment {
    environment_with_limits(Limits {
        cpu: 1_000_000_000,
        memory: 256 * 1024 * 1024,
        disk: 256 * 1024 * 1024,
        ..Limits::default()
    })
}

fn environment_with_limits(limits: Limits) -> Environment {
    let root =
        std::env::var_os("CPYTHON_WASI_ROOT").expect("set CPYTHON_WASI_ROOT to the built rootfs");
    let mut environment = Environment::with_limits(limits);
    import_tree(&mut environment, Path::new(&root), "/");
    environment
}

fn python(environment: &mut Environment, source: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let source = source.replace('\'', "'\\''");
    let (outcome, stdout, stderr) =
        environment.run_script_capture(&format!("/usr/bin/python3.wasm -B -c '{source}'"));
    eprintln!("CPython usage: {:?}", outcome.usage);
    (outcome.exit_status, stdout, stderr)
}

#[test]
#[ignore = "requires a separately built CPython WASI rootfs"]
fn cpython_runs_source_and_imports_from_virtual_files() {
    let mut environment = environment();
    environment
        .vfs
        .write("/", "/work/local_module.py", b"answer = 42\n", 0o644)
        .unwrap();
    let (status, stdout, stderr) = python(&mut environment,
        "import sys, json, os; sys.path.insert(0, '/work'); import local_module; print(sys.implementation.name); print(json.dumps({'answer': local_module.answer}, sort_keys=True)); open('/work/result', 'w').write('virtual'); print(sorted(os.listdir('/work')))");
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        stdout,
        b"cpython\n{\"answer\": 42}\n['local_module.py', 'result']\n"
    );
    assert_eq!(
        environment.vfs.read("/", "/work/result").unwrap(),
        b"virtual"
    );
}

#[test]
#[ignore = "requires a separately built CPython WASI rootfs"]
fn cpython_uses_virtual_clocks_and_seeded_entropy() {
    let mut first = environment();
    let mut second = environment();
    let source = "import time, os; print(time.monotonic()); time.sleep(2); print(time.monotonic()); print(os.urandom(8).hex())";
    let result = python(&mut first, source);
    assert_eq!(result.0, 0, "{}", String::from_utf8_lossy(&result.2));
    assert_eq!(result, python(&mut second, source));
    assert!(result.1.starts_with(b"0.0\n2.0\n"));
    assert_eq!(first.clock.monotonic_ns(), 2_000_000_000);
}

#[test]
#[ignore = "requires a CPython WASI rootfs built with --with-pycosat"]
fn cpython_executes_static_native_extension() {
    let mut environment = environment();
    let (status, stdout, stderr) = python(&mut environment,
        "import pycosat; print(pycosat.__version__); print(pycosat.solve([[1], [-1]])); print(pycosat.solve([[1]]))");
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"0.6.6\nUNSAT\n[1]\n");
}

#[test]
#[ignore = "requires a separately built CPython WASI rootfs"]
fn cpython_guest_work_stops_at_the_machine_cpu_budget() {
    let mut environment = environment_with_limits(Limits {
        cpu: 100_000_000,
        memory: 256 * 1024 * 1024,
        disk: 256 * 1024 * 1024,
        ..Limits::default()
    });
    let (status, stdout, _) = python(
        &mut environment,
        "print('running', flush=True)\nwhile True: pass",
    );
    assert_eq!(status, 137);
    assert_eq!(stdout, b"running\n");
    assert_eq!(
        environment.resources.stop_reason(),
        Some(shellsim::resources::StopReason::CpuExhausted)
    );
}

#[test]
#[ignore = "requires a separately built CPython WASI rootfs"]
fn cpython_initial_memory_is_constrained_by_machine_limits() {
    let mut environment = environment_with_limits(Limits {
        cpu: 1_000_000_000,
        memory: 16 * 1024 * 1024,
        disk: 256 * 1024 * 1024,
        ..Limits::default()
    });
    let (status, stdout, stderr) = python(&mut environment, "print('unreachable')");
    assert_eq!(status, 126);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("exceeds memory limits"));
}
