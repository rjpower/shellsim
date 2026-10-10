// Real upstream interpreter proof uses separately built artifacts. Admission
// and lifecycle probes stay independent of host package/toolchain availability.
use shellsim::{Environment, Limits};
use std::path::Path;

#[test]
#[ignore = "requires the normal threaded CPython process-enabled bundle"]
fn upstream_cpython_subprocess_communicates_while_worker_thread_is_blocked() {
    let bundle = std::env::var("SHELLSIM_THREADED_BUNDLE").unwrap();
    let root = Path::new(&bundle).join("rootfs");
    let mut environment = Environment::with_limits(Limits {
        cpu: 50_000_000_000,
        memory: 2 * 1024 * 1024 * 1024,
        disk: 256 * 1024 * 1024,
        ..Limits::default()
    });
    mount_tree(&mut environment, &root, &root);
    let script = br#"import subprocess, sys, threading
ready = threading.Event()
release = threading.Event()
values = []
def worker():
    ready.set()
    assert release.wait(timeout=5)
    values.append(7)
thread = threading.Thread(target=worker)
thread.start()
assert ready.wait(timeout=5)
assert thread.is_alive()
child = subprocess.Popen([sys.executable, '-c', 'print(6 * 7)'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = child.communicate(timeout=2)
assert child.returncode == 0
assert int(stdout) == 42 and stderr == b''
assert thread.is_alive()
release.set()
thread.join(timeout=2)
assert not thread.is_alive() and values == [7]
print('combined-process-thread-ok')
"#;
    environment
        .vfs
        .write("/", "/combined.py", script, 0o644)
        .unwrap();
    let (outcome, stdout, stderr) =
        environment.run_script_capture("/usr/bin/python3.wasm /combined.py");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"combined-process-thread-ok\n");
}

#[test]
#[ignore = "requires the normal threaded CPython process-enabled bundle"]
fn threaded_cwd_interleaving_keeps_child_cwd_process_local() {
    let bundle = std::env::var("SHELLSIM_THREADED_BUNDLE").unwrap();
    let root = Path::new(&bundle).join("rootfs");
    let mut environment = Environment::with_limits(Limits {
        cpu: 50_000_000_000,
        memory: 2 * 1024 * 1024 * 1024,
        disk: 256 * 1024 * 1024,
        ..Limits::default()
    });
    mount_tree(&mut environment, &root, &root);
    for (directory, value) in [("/a", b"a"), ("/b", b"b")] {
        environment.vfs.mkdir_all("/", directory).unwrap();
        environment
            .vfs
            .write("/", &format!("{directory}/value"), value, 0o644)
            .unwrap();
    }
    let script = br#"import os, subprocess, sys, threading
first = threading.Event()
second = threading.Event()
observed = []
def worker():
    os.chdir('/a')
    assert os.getcwd() == '/a'
    first.set()
    assert second.wait(timeout=5)
    observed.append(os.getcwd())
    with open('value') as stream:
        observed.append(stream.read())
thread = threading.Thread(target=worker)
thread.start()
assert first.wait(timeout=5)
os.chdir('/b')
assert os.getcwd() == '/b'
second.set()
thread.join(timeout=5)
assert not thread.is_alive() and observed == ['/b', 'b']
child = subprocess.run([sys.executable, '-c', 'import os; print(os.getcwd())'],
                       cwd='/a', capture_output=True, timeout=2, check=True)
assert child.stdout.strip() == b'/a'
assert os.getcwd() == '/b'
with open('value') as stream:
    assert stream.read() == 'b'
print('threaded-cwd-ok')
"#;
    environment
        .vfs
        .write("/", "/cwd-proof.py", script, 0o644)
        .unwrap();
    let (outcome, stdout, stderr) =
        environment.run_script_capture("/usr/bin/python3.wasm /cwd-proof.py");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"threaded-cwd-ok\n");
}

fn mount_tree(environment: &mut Environment, root: &Path, directory: &Path) {
    let target = format!("/{}", directory.strip_prefix(root).unwrap().display());
    environment.vfs.mkdir_all("/", &target).unwrap();
    let mut entries = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            mount_tree(environment, root, &path);
        } else {
            let target = format!("/{}", path.strip_prefix(root).unwrap().display());
            environment
                .vfs
                .write(
                    "/",
                    &target,
                    &std::fs::read(path).unwrap(),
                    if target.ends_with(".wasm") {
                        0o755
                    } else {
                        0o644
                    },
                )
                .unwrap();
        }
    }
}

#[test]
#[ignore = "requires the pinned threaded CPython v3 artifact and rootfs"]
fn upstream_cpython_creates_threads_and_waits_against_virtual_time() {
    let executable = std::env::var("SHELLSIM_THREADED_CPYTHON").unwrap();
    let root = std::env::var("SHELLSIM_THREADED_ROOTFS").unwrap();
    let root = Path::new(&root);
    let mut environment = Environment::with_limits(Limits {
        cpu: 50_000_000_000,
        memory: 1024 * 1024 * 1024,
        disk: 256 * 1024 * 1024,
        ..Limits::default()
    });
    mount_tree(&mut environment, root, root);
    environment
        .vfs
        .write(
            "/",
            "/usr/bin/python3.wasm",
            &std::fs::read(executable).unwrap(),
            0o755,
        )
        .unwrap();
    let script = br#"import threading, time
values = []
lock = threading.Lock()
def work(value):
    for _ in range(3):
        with lock:
            values.append(value)
threads = [threading.Thread(target=work, args=(value,)) for value in range(2)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
assert sorted(values) == [0, 0, 0, 1, 1, 1]
condition = threading.Condition()
start = time.monotonic()
with condition:
    assert not condition.wait(timeout=0.02)
assert time.monotonic() - start >= 0.019
print('threaded-cpython-ok')
"#;
    environment
        .vfs
        .write("/", "/proof.py", script, 0o644)
        .unwrap();
    let (outcome, stdout, stderr) =
        environment.run_script_capture("/usr/bin/python3.wasm /proof.py");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"threaded-cpython-ok\n");
}

#[test]
#[ignore = "requires freshly linked deferred-init pthread and side fixtures"]
fn live_thread_calls_late_published_pointer_with_fresh_tls_and_shared_data() {
    let artifacts = std::env::var("SHELLSIM_DEFERRED_FIXTURES").unwrap();
    let artifacts = Path::new(&artifacts);
    let mut environment = Environment::with_limits(Limits {
        cpu: 5_000_000_000,
        memory: 512 * 1024 * 1024,
        ..Limits::default()
    });
    environment.vfs.mkdir_all("/", "/lib").unwrap();
    environment
        .vfs
        .write(
            "/",
            "/main",
            &std::fs::read(artifacts.join("main.wasm")).unwrap(),
            0o755,
        )
        .unwrap();
    environment
        .vfs
        .write(
            "/",
            "/lib/tls.so",
            &std::fs::read(artifacts.join("tls.so")).unwrap(),
            0o644,
        )
        .unwrap();
    let (outcome, stdout, stderr) = environment.run_script_capture("/main");
    assert_eq!(
        outcome.exit_status,
        0,
        "stdout={} stderr={}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(
        stdout,
        b"late load shared mutation fresh TLS constructors once\n"
    );
}

#[test]
#[ignore = "requires pinned A to B to C deferred-init side fixtures"]
fn local_dependency_graph_resolves_transitive_symbols_and_ctor_function_pointers() {
    let artifacts = std::env::var("SHELLSIM_DEFERRED_GRAPH").unwrap();
    let artifacts = Path::new(&artifacts);
    let mut environment = Environment::with_limits(Limits {
        cpu: 5_000_000_000,
        memory: 512 * 1024 * 1024,
        ..Limits::default()
    });
    environment.vfs.mkdir_all("/", "/lib").unwrap();
    environment
        .vfs
        .write(
            "/",
            "/main",
            &std::fs::read(artifacts.join("main.wasm")).unwrap(),
            0o755,
        )
        .unwrap();
    for name in ["liba.so", "libb.so", "libc.so"] {
        environment
            .vfs
            .write(
                "/",
                &format!("/lib/{name}"),
                &std::fs::read(artifacts.join(name)).unwrap(),
                0o644,
            )
            .unwrap();
    }
    let (outcome, stdout, stderr) = environment.run_script_capture("/main");
    assert_eq!(
        outcome.exit_status,
        0,
        "stdout={} stderr={}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(
        stdout,
        b"local transitive lookup and indirect constructor once\n"
    );
}

#[test]
#[ignore = "requires upstream threaded CPython and independently compiled late extension fixtures"]
fn upstream_cpython_imports_tls_extension_while_worker_is_fuel_suspended() {
    let directory = std::env::var("SHELLSIM_CPYTHON_LATE").unwrap();
    let directory = Path::new(&directory);
    let root = std::env::var("SHELLSIM_THREADED_ROOTFS").unwrap();
    let root = Path::new(&root);
    let mut environment = Environment::with_limits(Limits {
        cpu: 50_000_000_000,
        memory: 1024 * 1024 * 1024,
        disk: 256 * 1024 * 1024,
        ..Limits::default()
    });
    mount_tree(&mut environment, root, root);
    environment
        .vfs
        .write(
            "/",
            "/usr/bin/python3.wasm",
            &std::fs::read(directory.join("python3.wasm")).unwrap(),
            0o755,
        )
        .unwrap();
    environment.vfs.mkdir_all("/", "/site").unwrap();
    for name in ["thread_driver.so", "late_tls.so"] {
        environment
            .vfs
            .write(
                "/",
                &format!("/site/{name}"),
                &std::fs::read(directory.join(name)).unwrap(),
                0o644,
            )
            .unwrap();
    }
    let script = br#"import sys, threading
sys.path.insert(0, '/site')
import thread_driver
results = []
def work():
    results.append(thread_driver.wait_and_call())
worker = threading.Thread(target=work)
worker.start()
while not thread_driver.started():
    pass
import late_tls
assert late_tls.bump(5) == 16
thread_driver.attach(late_tls.capsule())
worker.join()
assert results == [14]
assert late_tls.state() == (16, 72, 1)
second = threading.Thread(target=work)
second.start()
second.join()
assert results == [14, 14]
assert late_tls.state() == (16, 73, 1)
assert late_tls.bump(1) == 17
print('threaded-cpython-late-tls-ok')
"#;
    environment
        .vfs
        .write("/", "/proof.py", script, 0o644)
        .unwrap();
    let (outcome, stdout, stderr) =
        environment.run_script_capture("/usr/bin/python3.wasm /proof.py");
    assert_eq!(
        outcome.exit_status,
        0,
        "stdout={} stderr={}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"threaded-cpython-late-tls-ok\n");
}

#[test]
#[ignore = "requires upstream threaded CPython with canonical SDK C++ EH runtime"]
fn upstream_cpython_catches_rethrows_and_destroys_across_cpp_modules_and_thread_stores() {
    let mut environment = cpp_environment(50_000_000_000);
    let script = br#"import sys, threading
sys.path.insert(0, '/site')
import cpp_source, cpp_catch
capsule = cpp_source.capsule()
assert cpp_catch.exercise(capsule) == 2
values = []
def work():
    values.append(cpp_catch.exercise(capsule))
workers = [threading.Thread(target=work) for _ in range(2)]
for worker in workers:
    worker.start()
for worker in workers:
    worker.join()
assert values == [2, 2]
assert cpp_catch.exercise(capsule) == 4
print('threaded-cpython-cpp-eh-ok')
"#;
    environment
        .vfs
        .write("/", "/proof.py", script, 0o644)
        .unwrap();
    let (outcome, stdout, stderr) =
        environment.run_script_capture("/usr/bin/python3.wasm /proof.py");
    assert_eq!(
        outcome.exit_status,
        0,
        "stdout={} stderr={}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"threaded-cpython-cpp-eh-ok\n");
    assert!(outcome.usage.memory_current < 1024 * 1024);
}

fn cpp_environment(cpu: u64) -> Environment {
    let directory = std::env::var("SHELLSIM_CPYTHON_CPP").unwrap();
    let directory = Path::new(&directory);
    let root = std::env::var("SHELLSIM_THREADED_ROOTFS").unwrap();
    let root = Path::new(&root);
    let mut environment = Environment::with_limits(Limits {
        cpu,
        memory: 1024 * 1024 * 1024,
        disk: 256 * 1024 * 1024,
        ..Limits::default()
    });
    mount_tree(&mut environment, root, root);
    environment
        .vfs
        .write(
            "/",
            "/usr/bin/python3.wasm",
            &std::fs::read(directory.join("python3.wasm")).unwrap(),
            0o755,
        )
        .unwrap();
    environment.vfs.mkdir_all("/", "/site").unwrap();
    for name in ["cpp_source.so", "cpp_catch.so"] {
        environment
            .vfs
            .write(
                "/",
                &format!("/site/{name}"),
                &std::fs::read(directory.join(name)).unwrap(),
                0o644,
            )
            .unwrap();
    }
    environment
}

fn install_cpp_workers(environment: &mut Environment, compute: bool) {
    let script = br#"import sys, threading
sys.path.insert(0, '/site')
import cpp_source, cpp_catch
capsule = cpp_source.capsule()
started = threading.Event()
def busy():
    assert cpp_catch.exercise(capsule) == 2
    started.set()
    while True:
        cpp_catch.exercise(capsule)
def blocked():
    threading.Event().wait(10)
    print('blocked-worker-resumed', flush=True)
threading.Thread(target=busy).start()
threading.Thread(target=blocked).start()
assert started.wait(1)
with open('/ready', 'w') as stream:
    stream.write('ready')
print('ready', flush=True)
threading.Event().wait(10)
print('main-resumed', flush=True)
"#;
    let script = if compute {
        script.to_vec()
    } else {
        String::from_utf8_lossy(script)
            .replace(
                "    while True:\n        cpp_catch.exercise(capsule)",
                "    threading.Event().wait(10)",
            )
            .into_bytes()
    };
    environment
        .vfs
        .write("/", "/proof.py", &script, 0o644)
        .unwrap();
}

#[test]
#[ignore = "set CPP and THREADED_ROOTFS for actual threaded CPython EH lifecycle"]
fn threaded_cpp_blocked_process_cancellation_releases_all_stores_and_heaps() {
    let mut environment = cpp_environment(50_000_000_000);
    install_cpp_workers(&mut environment, false);
    let (outcome, stdout, stderr) = environment.run_script_capture(
        "/usr/bin/python3.wasm /proof.py & pid=$!; while ! test -f /ready; do sleep 0.001; done; kill -TERM $pid; wait $pid");
    assert_eq!(
        outcome.exit_status,
        143,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"ready\n");
    assert!(outcome.usage.memory_current < 1024 * 1024);
    assert!(environment.clock.monotonic_ns() < 10_000_000_000);
}

#[test]
#[ignore = "set CPP and THREADED_ROOTFS for actual threaded CPython EH lifecycle"]
fn threaded_cpp_compute_exhaustion_releases_all_stores_and_heaps() {
    let mut environment = cpp_environment(5_000_000_000);
    install_cpp_workers(&mut environment, true);
    let (outcome, stdout, stderr) =
        environment.run_script_capture("/usr/bin/python3.wasm /proof.py");
    assert_eq!(
        outcome.exit_status,
        137,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"ready\n");
    assert!(outcome.usage.memory_current < 1024 * 1024);
}

#[test]
#[ignore = "set CPP and THREADED_ROOTFS for actual threaded CPython EH lifecycle"]
fn threaded_cpp_runnable_process_cancellation_releases_all_stores_and_heaps() {
    let mut environment = cpp_environment(5_000_000_000);
    install_cpp_workers(&mut environment, true);
    // This readiness loop is metered shell work. A timer cannot serve as the
    // handshake while another virtual process remains continuously runnable.
    let (outcome, stdout, stderr) = environment.run_script_capture(
        "/usr/bin/python3.wasm /proof.py & pid=$!; while ! test -f /ready; do :; done; kill -TERM $pid; wait $pid");
    assert_eq!(
        outcome.exit_status,
        143,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"ready\n");
    assert!(outcome.usage.memory_current < 1024 * 1024);
}

// LLVM emits weak import metadata for the env function, while its address is
// imported separately through GOT.func. Exercise both bindings in real Stores.
fn weak_side(weak: bool, direct: bool) -> Vec<u8> {
    let flags = if weak {
        r"\04\0f\01\03env\08optional\11"
    } else {
        ""
    };
    let call = if direct {
        "(call $optional)"
    } else {
        "(if (global.get $address) (then (call $optional)))"
    };
    wat::parse_str(format!(
        r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-threads-v3")
        (@custom "dylink.0" "\01\04\00\00\00\00\80\18\16shellsim.deferred-init\01{flags}")
        (import "env" "memory" (memory 1 4 shared))
        (import "env" "optional" (func $optional))
        (import "GOT.func" "optional" (global $address (mut i32)))
        (func (export "answer") (result i32) {call}
            (if (result i32) (global.get $address)
                (then (i32.load (i32.const 0))) (else (i32.const 7)))))"#
    ))
    .unwrap()
}

fn weak_environment(provider: bool, body: &str) -> Environment {
    let provider = if provider {
        r#"(func (export "optional") (i32.store (i32.const 0) (i32.const 42)))"#
    } else {
        ""
    };
    let main = wat::parse_str(format!(
        r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-threads-v3")
        (@custom "dylink.0" "\81\13\11shellsim.main-tls\01")
        (import "env" "memory" (memory 1 4 shared))
        (import "shellsim_threads_v2" "thread_ready" (func (param i32 i32)))
        (import "shellsim_dylink_v3" "open" (func $open (param i32 i32 i32) (result i32)))
        (import "shellsim_dylink_v3" "symbol" (func $symbol (param i32 i32 i32) (result i32)))
        (export "memory" (memory 0))
        (table (export "__indirect_function_table") 1 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (global (export "__stack_low") i32 (i32.const 32768))
        (global (export "__stack_high") i32 (i32.const 65536))
        (global $heap (mut i32) (i32.const 4096))
        (data $path "/side.so") (data $name "answer")
        (func (export "malloc") (param $size i32) (result i32) (local $base i32)
            (local.set $base (global.get $heap))
            (global.set $heap (i32.add (global.get $heap) (local.get $size))) (local.get $base))
        (func (export "wasi_thread_start") (param i32 i32))
        {provider}
        (func (export "_start") (local $handle i32)
            (memory.init $path (i32.const 32) (i32.const 0) (i32.const 8))
            (memory.init $name (i32.const 64) (i32.const 0) (i32.const 6))
            (data.drop $path) (data.drop $name)
            (local.set $handle (call $open (i32.const 32) (i32.const 8) (i32.const 2)))
            {body}))"#
    ))
    .unwrap();
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000_000,
        memory: 64 * 1024 * 1024,
        ..Limits::default()
    });
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    environment
}

#[test]
fn unresolved_weak_function_has_null_address_and_guarded_call_skips_it() {
    for (provider, expected) in [(false, 7), (true, 42)] {
        let body = format!(
            r#"
            (if (i32.eqz (local.get $handle)) (then unreachable))
            (if (i32.ne (call_indirect (result i32)
                (call $symbol (local.get $handle) (i32.const 64) (i32.const 6)))
                (i32.const {expected})) (then unreachable))"#
        );
        let mut environment = weak_environment(provider, &body);
        environment
            .vfs
            .write("/", "/side.so", &weak_side(true, false), 0o644)
            .unwrap();
        let (outcome, _, stderr) = environment.run_script_capture("/app");
        assert_eq!(
            outcome.exit_status,
            0,
            "{}",
            String::from_utf8_lossy(&stderr)
        );
    }
}

#[test]
fn calling_unresolved_weak_function_traps_instead_of_returning_success() {
    let mut environment = weak_environment(
        false,
        r#"
        (if (i32.eqz (local.get $handle)) (then unreachable))
        (drop (call_indirect (result i32)
            (call $symbol (local.get $handle) (i32.const 64) (i32.const 6))))"#,
    );
    environment
        .vfs
        .write("/", "/side.so", &weak_side(true, true), 0o644)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_ne!(outcome.exit_status, 0);
    assert!(String::from_utf8_lossy(&stderr).contains("unresolved weak function called: optional"));
}

#[test]
fn missing_strong_function_still_prevents_side_publication() {
    let mut environment = weak_environment(false, "(if (local.get $handle) (then unreachable))");
    environment
        .vfs
        .write("/", "/side.so", &weak_side(false, false), 0o644)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!(outcome.exit_status, 126);
    assert!(String::from_utf8_lossy(&stderr).contains("missing dynamic symbol: optional"));
}
