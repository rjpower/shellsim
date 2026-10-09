// Untrusted Wasm probes exercise publication across real thread Stores without
// relying on a compiler fixture. Upstream libffi/CPython acceptance follows the
// same boundary with separately pinned threaded artifacts.
use shellsim::{Environment, Limits};

fn module(main: &str, worker: &str) -> Vec<u8> {
    module_with_initializer(main, worker, "")
}

fn module_with_initializer(main: &str, worker: &str, initializer: &str) -> Vec<u8> {
    wat::parse_str(format!(r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-threads-v3")
        (@custom "dylink.0" "\81\13\11shellsim.main-tls\01")
        (type $callback (func (param i32) (result i32)))
        (import "env" "memory" (memory 1 4 shared))
        (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
        (import "wasi" "thread-spawn" (func $spawn (param i32) (result i32)))
        (import "shellsim_threads_v2" "thread_ready" (func $ready (param i32 i32)))
        (import "shellsim_threads_v2" "wait32" (func $wait (param i32 i32 i64) (result i32)))
        (import "shellsim_threads_v2" "notify" (func $notify (param i32 i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_alloc" (func $alloc (param i32 i32 i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_alloc_typed" (func $alloc_typed (param i32 i32 i32 i32 i32 i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_release" (func $release (param i32) (result i32)))
        (import "shellsim_ffi_v1" "invoke" (func $invoke (param i32 i32 i32 i32 i32 i32) (result i32)))
        (export "memory" (memory 0))
        (table (export "__indirect_function_table") 2 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (global (export "__stack_low") i32 (i32.const 4096))
        (global (export "__stack_high") i32 (i32.const 65536))
        (func $dispatch (param i32 i32) (result i32) (i32.add (local.get 0) (local.get 1)))
        (func $typed_dispatch (param $userdata i32) (param $arguments i32) (param $result i32) (result i32)
            (i64.store (local.get $result) (i64.add (i64.extend_i32_u (local.get $userdata)) (i64.load (local.get $arguments))))
            (i32.const 0))
        (elem (i32.const 0) $dispatch $typed_dispatch)
        (func $trace
            (i32.store8 (i32.const 384) (i32.const 88))
            (i32.store (i32.const 400) (i32.const 384))
            (i32.store (i32.const 404) (i32.const 1))
            (drop (call $write (i32.const 1) (i32.const 400) (i32.const 1) (i32.const 408))))
        (func $initialize {initializer})
        (start $initialize)
        (func (export "_start") {main})
        (func (export "wasi_thread_start") (param i32 i32)
            (global.set 0 (i32.const 49152))
            (call $ready (i32.const 32768) (i32.const 49152))
            {worker}))"#)).unwrap()
}

#[test]
fn callback_pointer_replays_in_worker_and_release_revokes_parent_slot() {
    let bytes = module(
        r#"
        (if (call $alloc (i32.const 0) (i32.const 10) (i32.const 256)) (then unreachable))
        (if (i32.lt_s (call $spawn (i32.const 0)) (i32.const 0)) (then unreachable))
        (drop (call $wait (i32.const 300) (i32.const 0) (i64.const -1)))
        (if (i32.ne (i32.load (i32.const 304)) (i32.const 17)) (then unreachable))
        (if (i32.ne (call $release (i32.load (i32.const 256))) (i32.const 28)) (then unreachable))
        (i32.store8 (i32.const 32) (i32.const 1))
        (i64.store (i32.const 64) (i64.const 7))
        (if (i32.ne (call $invoke (i32.load (i32.const 256)) (i32.const 32) (i32.const 64) (i32.const 1) (i32.const 1) (i32.const 128)) (i32.const 28)) (then unreachable))
        (if (call $alloc (i32.const 0) (i32.const 20) (i32.const 260)) (then unreachable))
        (if (i32.eq (i32.load (i32.const 256)) (i32.load (i32.const 260))) (then unreachable))
        (if (i32.ne (call_indirect (type $callback) (i32.const 7) (i32.load (i32.const 260))) (i32.const 27)) (then unreachable))
    "#,
        r#"
        (i32.store (i32.const 304) (call_indirect (type $callback) (i32.const 7) (i32.load (i32.const 256))))
        (if (call $release (i32.load (i32.const 256))) (then unreachable))
        (i32.atomic.store (i32.const 300) (i32.const 1))
        (drop (call $notify (i32.const 300) (i32.const 1)))
    "#,
    );
    let mut environment = Environment::with_limits(Limits {
        memory: 64 * 1024 * 1024,
        ..Limits::default()
    });
    environment
        .vfs
        .write("/", "/threaded-ffi", &bytes, 0o755)
        .unwrap();
    let (outcome, stdout, stderr) = environment.run_script_capture("/threaded-ffi");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
    assert_eq!(environment.resources.memory_mark(), 0);
    let (outcome, stdout, _) = environment.run_script_capture("printf recovered");
    assert_eq!(outcome.exit_status, 0);
    assert_eq!(stdout, b"recovered");
}

#[test]
fn callback_tombstones_bound_lifetime_slots_and_release_on_process_exit() {
    let bytes = module(
        r#"
        (local $count i32)
        (i32.store8 (i32.const 32) (i32.const 1))
        (if (i32.ne (call $alloc_typed (i32.const 0) (i32.const 10) (i32.const 32) (i32.const 1) (i32.const 1) (i32.const 256)) (i32.const 28)) (then unreachable))
        (i32.store8 (i32.const 32) (i32.const 255))
        (if (i32.ne (call $alloc_typed (i32.const 1) (i32.const 10) (i32.const 32) (i32.const 1) (i32.const 1) (i32.const 256)) (i32.const 28)) (then unreachable))
        (if (i32.ne (call $alloc (i32.const 0) (i32.const 10) (i32.const 65535)) (i32.const 21)) (then unreachable))
        (loop $slots
            (if (call $alloc (i32.const 0) (i32.const 10) (i32.const 256)) (then unreachable))
            (if (call $release (i32.load (i32.const 256))) (then unreachable))
            (local.set $count (i32.add (local.get $count) (i32.const 1)))
            (br_if $slots (i32.lt_u (local.get $count) (i32.const 64))))
        (if (i32.ne (call $alloc (i32.const 0) (i32.const 10) (i32.const 256)) (i32.const 51)) (then unreachable))
    "#,
        "",
    );
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/threaded-ffi", &bytes, 0o755)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("/threaded-ffi");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn callback_metadata_exhaustion_releases_all_process_and_store_charges() {
    let bytes = module(
        r#"
        (local $count i32)
        (loop $slots
            (if (call $alloc (i32.const 0) (i32.const 10) (i32.const 256)) (then unreachable))
            (call $trace)
            (local.set $count (i32.add (local.get $count) (i32.const 1)))
            (br_if $slots (i32.lt_u (local.get $count) (i32.const 64))))
    "#,
        "",
    );
    // Prepaid worker/table resources dominate this profile. Allow a measured
    // empty-process baseline plus less than the metadata for 64 callbacks.
    let mut baseline = Environment::new();
    baseline
        .vfs
        .write("/", "/threaded-ffi", &module("", ""), 0o755)
        .unwrap();
    let (control, _, stderr) = baseline.run_script_capture("/threaded-ffi");
    assert_eq!(
        control.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    let mut environment = Environment::with_limits(Limits {
        memory: control.usage.memory_peak + 128 * 1024,
        ..Limits::default()
    });
    environment
        .vfs
        .write("/", "/threaded-ffi", &bytes, 0o755)
        .unwrap();
    let (outcome, stdout, stderr) = environment.run_script_capture("/threaded-ffi");
    assert_eq!(
        outcome.exit_status,
        137,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(
        !stdout.is_empty(),
        "a callback must be published before exhaustion"
    );
    assert!(stdout.len() < 64);
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn cancelled_process_releases_callbacks_in_main_and_waiting_worker() {
    let bytes = module(
        r#"
        (if (call $alloc (i32.const 0) (i32.const 10) (i32.const 256)) (then unreachable))
        (if (i32.lt_s (call $spawn (i32.const 0)) (i32.const 0)) (then unreachable))
        (drop (call $wait (i32.const 300) (i32.const 0) (i64.const -1)))
    "#,
        r#"
        (if (i32.ne (call_indirect (type $callback) (i32.const 7) (i32.load (i32.const 256))) (i32.const 17)) (then unreachable))
        (drop (call $wait (i32.const 300) (i32.const 0) (i64.const -1)))
    "#,
    );
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/threaded-ffi", &bytes, 0o755)
        .unwrap();
    let (outcome, _, stderr) =
        environment.run_script_capture("/threaded-ffi & pid=$!; sleep 0.001; kill $pid; wait $pid");
    assert_eq!(
        outcome.exit_status,
        143,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    // Shell variables/process labels outlive the killed guest. Compare against
    // the same cancellation with two Stores and no callback allocations.
    let baseline = module(
        r#"
        (drop (call $spawn (i32.const 0)))
        (drop (call $wait (i32.const 300) (i32.const 0) (i64.const -1)))
    "#,
        "(drop (call $wait (i32.const 300) (i32.const 0) (i64.const -1)))",
    );
    let mut control = Environment::new();
    control
        .vfs
        .write("/", "/threaded-ffi", &baseline, 0o755)
        .unwrap();
    assert_eq!(
        control
            .run_script_capture("/threaded-ffi & pid=$!; sleep 0.001; kill $pid; wait $pid")
            .0
            .exit_status,
        143
    );
    assert_eq!(
        environment.resources.memory_mark(),
        control.resources.memory_mark()
    );
}

#[test]
fn typed_callback_uses_worker_stack_and_restores_its_pointer() {
    let bytes = module(
        r#"
        (i32.store8 (i32.const 32) (i32.const 1))
        (if (call $alloc_typed (i32.const 1) (i32.const 10) (i32.const 32) (i32.const 1) (i32.const 1) (i32.const 256)) (then unreachable))
        (drop (call $spawn (i32.const 0)))
        (drop (call $wait (i32.const 300) (i32.const 0) (i64.const -1)))
        (if (i32.ne (i32.load (i32.const 304)) (i32.const 17)) (then unreachable))
    "#,
        r#"
        (i32.store (i32.const 304) (call_indirect (type $callback) (i32.const 7) (i32.load (i32.const 256))))
        (if (i32.ne (global.get 0) (i32.const 49152)) (then unreachable))
        (if (call $release (i32.load (i32.const 256))) (then unreachable))
        (i32.atomic.store (i32.const 300) (i32.const 1))
        (drop (call $notify (i32.const 300) (i32.const 1)))
    "#,
    );
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/threaded-ffi", &bytes, 0o755)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("/threaded-ffi");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn initialization_rejects_callback_publication_before_mutation() {
    let bytes = module_with_initializer(
        "",
        "",
        "(drop (call $alloc (i32.const 0) (i32.const 10) (i32.const 256)))",
    );
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/threaded-ffi", &bytes, 0o755)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("/threaded-ffi");
    assert_ne!(outcome.exit_status, 0);
    assert!(String::from_utf8_lossy(&stderr)
        .contains("FFI during threaded loader initialization is unsupported"));
    assert_eq!(environment.resources.memory_mark(), 0);
    let (outcome, stdout, _) = environment.run_script_capture("printf recovered");
    assert_eq!(outcome.exit_status, 0);
    assert_eq!(stdout, b"recovered");
}

#[test]
fn static_thread_profile_does_not_admit_process_callback_publication() {
    let bytes = wat::parse_str(
        r#"(module
        (import "env" "memory" (memory 1 4 shared))
        (import "shellsim_threads_v1" "wait32" (func (param i32 i32 i64) (result i32)))
        (import "shellsim_ffi_v1" "closure_alloc" (func (param i32 i32 i32) (result i32)))
        (export "memory" (memory 0))
        (func (export "_start")))"#,
    )
    .unwrap();
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/threaded-ffi", &bytes, 0o755)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("/threaded-ffi");
    assert_eq!(outcome.exit_status, 126);
    assert!(String::from_utf8_lossy(&stderr)
        .contains("unsupported wasm import: shellsim_ffi_v1.closure_alloc"));
    assert_eq!(environment.resources.memory_mark(), 0);
}
