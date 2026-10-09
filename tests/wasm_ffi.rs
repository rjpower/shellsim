// Synthetic modules check the raw FFI boundary without requiring an SDK build.
use shellsim::{Environment, Limits};
use std::path::PathBuf;

fn run(wat: &str) -> (i32, Vec<u8>) {
    let mut environment = Environment::with_limits(Limits {
        cpu: 2_000_000,
        memory: 64 * 1024 * 1024,
        ..Limits::default()
    });
    let bytes = wat::parse_str(wat).unwrap();
    environment.vfs.write("/", "/app", &bytes, 0o755).unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    (outcome.exit_status, stderr)
}

#[test]
fn scalar_function_table_calls_preserve_integer_float_and_pointer_bits() {
    let wat = r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (import "shellsim_ffi_v1" "invoke" (func $invoke (param i32 i32 i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 3 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (func $add (param i32 i32) (result i32)
            (i32.add (local.get 0) (local.get 1)))
        (func $mix (param f64 i32) (result f64)
            (f64.add (local.get 0) (f64.convert_i32_s (local.get 1))))
        (func $identity (param i32) (result i32) (local.get 0))
        (elem (i32.const 0) $add $mix $identity)
        (func (export "_start")
            (i32.store8 (i32.const 32) (i32.const 1))
            (i32.store8 (i32.const 33) (i32.const 1))
            (i64.store (i32.const 64) (i64.const 17))
            (i64.store (i32.const 72) (i64.const 25))
            (if (i32.ne (call $invoke (i32.const 0) (i32.const 32) (i32.const 64)
                      (i32.const 2) (i32.const 1) (i32.const 128)) (i32.const 0))
                (then unreachable))
            (if (i64.ne (i64.load (i32.const 128)) (i64.const 42)) (then unreachable))
            (i32.store8 (i32.const 32) (i32.const 4))
            (i64.store (i32.const 64) (i64.const 4609434218613702656)) ;; 1.5
            (i64.store (i32.const 72) (i64.const 2))
            (if (i32.ne (call $invoke (i32.const 1) (i32.const 32) (i32.const 64)
                      (i32.const 2) (i32.const 4) (i32.const 128)) (i32.const 0))
                (then unreachable))
            (if (i64.ne (i64.load (i32.const 128)) (i64.const 4615063718147915776)) ;; 3.5
                (then unreachable))
            (i32.store8 (i32.const 32) (i32.const 1))
            (i64.store (i32.const 64) (i64.const 65280))
            (if (i32.ne (call $invoke (i32.const 2) (i32.const 32) (i32.const 64)
                      (i32.const 1) (i32.const 1) (i32.const 128)) (i32.const 0))
                (then unreachable))
            (if (i64.ne (i64.load (i32.const 128)) (i64.const 65280)) (then unreachable))))"#;
    assert_eq!(run(wat), (0, Vec::new()));
}

#[test]
fn invalid_signature_and_output_do_not_call_function() {
    let wat = r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (import "shellsim_ffi_v1" "invoke" (func $invoke (param i32 i32 i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 1 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (global $calls (mut i32) (i32.const 0))
        (func $target (param i32) (result i32)
            (global.set $calls (i32.add (global.get $calls) (i32.const 1)))
            (local.get 0))
        (elem (i32.const 0) $target)
        (func (export "_start")
            (i32.store8 (i32.const 32) (i32.const 4))
            (i64.store (i32.const 64) (i64.const 7))
            (if (i32.ne (call $invoke (i32.const 0) (i32.const 32) (i32.const 64)
                      (i32.const 1) (i32.const 1) (i32.const 128)) (i32.const 28))
                (then unreachable))
            (i32.store8 (i32.const 32) (i32.const 1))
            (if (i32.ne (call $invoke (i32.const 0) (i32.const 32) (i32.const 64)
                      (i32.const 1) (i32.const 1) (i32.const 65532)) (i32.const 21))
                (then unreachable))
            (if (i32.ne (global.get $calls) (i32.const 0)) (then unreachable))))"#;
    assert_eq!(run(wat), (0, Vec::new()));
}

#[test]
fn ffi_import_requires_exact_dynamic_profile() {
    let wat = r#"(module
        (import "shellsim_ffi_v1" "invoke" (func (param i32 i32 i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (func (export "_start")))"#;
    let (status, stderr) = run(wat);
    assert_eq!(status, 126);
    assert!(String::from_utf8(stderr)
        .unwrap()
        .contains("dynamic loading ABI mismatch"));
}

#[test]
fn callback_reenters_guest_and_release_tombstones_its_slot() {
    let wat = r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (type $callback (func (param i32) (result i32)))
        (import "shellsim_ffi_v1" "invoke" (func $invoke (param i32 i32 i32 i32 i32 i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_alloc" (func $alloc (param i32 i32 i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_release" (func $release (param i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 2 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (func $dispatch (param $userdata i32) (param $argument i32) (result i32)
            (i32.store8 (i32.const 32) (i32.const 1))
            (i32.store8 (i32.const 33) (i32.const 1))
            (i64.store (i32.const 64) (i64.extend_i32_u (local.get $userdata)))
            (i64.store (i32.const 72) (i64.extend_i32_u (local.get $argument)))
            (if (i32.ne (call $invoke (i32.const 1) (i32.const 32) (i32.const 64)
                      (i32.const 2) (i32.const 1) (i32.const 128)) (i32.const 0))
                (then unreachable))
            (i32.load (i32.const 128)))
        (func $add (param i32 i32) (result i32)
            (i32.add (local.get 0) (local.get 1)))
        (elem (i32.const 0) $dispatch $add)
        (func (export "_start") (local $slot i32)
            (if (i32.ne (call $alloc (i32.const 0) (i32.const 10) (i32.const 256)) (i32.const 0))
                (then unreachable))
            (local.set $slot (i32.load (i32.const 256)))
            (if (i32.ne (local.get $slot) (i32.const 2)) (then unreachable))
            (if (i32.ne (call_indirect (type $callback) (i32.const 7) (local.get $slot)) (i32.const 17))
                (then unreachable))
            (if (i32.ne (call $release (local.get $slot)) (i32.const 0)) (then unreachable))
            (if (i32.ne (call $release (local.get $slot)) (i32.const 28)) (then unreachable))
            (if (i32.eqz (ref.is_null (table.get (local.get $slot)))) (then unreachable))))"#;
    assert_eq!(run(wat), (0, Vec::new()));
}

#[test]
fn closure_preflight_capacity_and_retained_slots_are_bounded() {
    let wat = r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (import "shellsim_ffi_v1" "closure_alloc" (func $alloc (param i32 i32 i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_release" (func $release (param i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 2 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (func $dispatch (param i32 i32) (result i32) (i32.add (local.get 0) (local.get 1)))
        (func $wrong (param i32) (result i32) (local.get 0))
        (elem (i32.const 0) $dispatch $wrong)
        (func (export "_start") (local $i i32)
            (if (i32.ne (call $alloc (i32.const 0) (i32.const 0) (i32.const 65534)) (i32.const 21))
                (then unreachable))
            (if (i32.ne (call $alloc (i32.const 1) (i32.const 0) (i32.const 256)) (i32.const 28))
                (then unreachable))
            (if (i32.ne (table.size) (i32.const 2)) (then unreachable))
            (block $done (loop $again
                (br_if $done (i32.ge_u (local.get $i) (i32.const 64)))
                (if (i32.ne (call $alloc (i32.const 0) (local.get $i) (i32.const 256)) (i32.const 0))
                    (then unreachable))
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br $again)))
            (if (i32.ne (table.size) (i32.const 66)) (then unreachable))
            (if (i32.ne (call $release (i32.const 2)) (i32.const 0)) (then unreachable))
            (if (i32.ne (call $alloc (i32.const 0) (i32.const 0) (i32.const 256)) (i32.const 51))
                (then unreachable))
            (if (i32.ne (table.size) (i32.const 66)) (then unreachable))))"#;
    assert_eq!(run(wat), (0, Vec::new()));
}

fn callback_wait_guest(wait_ns: u64) -> Vec<u8> {
    wat::parse_str(format!(r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (type $callback (func (param i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_alloc" (func $alloc (param i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "poll_oneoff" (func $poll (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 1 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (func $dispatch (param i32 i32) (result i32)
            (i32.store (i32.const 16) (i32.const 1))
            (i64.store (i32.const 24) (i64.const {wait_ns}))
            (if (call $poll (i32.const 0) (i32.const 100) (i32.const 1) (i32.const 200))
                (then unreachable))
            (i32.add (local.get 0) (local.get 1)))
        (elem (i32.const 0) $dispatch)
        (func (export "_start")
            (if (call $alloc (i32.const 0) (i32.const 10) (i32.const 256)) (then unreachable))
            (if (i32.ne
                    (call_indirect (type $callback) (i32.const 7) (i32.load (i32.const 256)))
                    (i32.const 17))
                (then unreachable))))"#)).unwrap()
}

#[test]
fn callback_wait_resumes_on_virtual_clock() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 2_000_000,
        memory: 64 * 1024 * 1024,
        ..Limits::default()
    });
    environment
        .vfs
        .write("/", "/app", &callback_wait_guest(1_000_000_000), 0o755)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!((outcome.exit_status, stderr), (0, Vec::new()));
    assert_eq!(environment.clock.monotonic_ns(), 1_000_000_000);
}

#[test]
fn timeout_cancels_a_suspended_callback_and_releases_its_reservation() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 2_000_000,
        memory: 64 * 1024 * 1024,
        ..Limits::default()
    });
    environment
        .vfs
        .write("/", "/app", &callback_wait_guest(10_000_000_000), 0o755)
        .unwrap();
    let first = environment.run_script_capture("timeout 1 /app; echo $?");
    assert_eq!(
        (first.0.exit_status, first.1, first.2),
        (0, b"124\n".to_vec(), Vec::new())
    );
    let first_retained = environment.resources.memory_mark();
    let second = environment.run_script_capture("timeout 1 /app; echo $?");
    assert_eq!(
        (second.0.exit_status, second.1, second.2),
        (0, b"124\n".to_vec(), Vec::new())
    );
    assert_eq!(environment.resources.memory_mark(), first_retained);
    assert_eq!(environment.clock.monotonic_ns(), 2_000_000_000);
}

#[test]
fn callback_exception_can_be_caught_by_guest_caller() {
    let wat = r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (type $callback (func (param i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_alloc" (func $alloc (param i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 1 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (tag $error (param i32))
        (func $dispatch (param i32 i32) (result i32)
            (throw $error (i32.add (local.get 0) (local.get 1))))
        (elem (i32.const 0) $dispatch)
        (func (export "_start")
            (if (call $alloc (i32.const 0) (i32.const 10) (i32.const 256)) (then unreachable))
            (if (i32.ne
                    (block $caught (result i32)
                        (try_table (catch $error $caught)
                            (drop (call_indirect (type $callback)
                                (i32.const 7) (i32.load (i32.const 256)))))
                        (i32.const 0))
                    (i32.const 17))
                (then unreachable))))"#;
    assert_eq!(run(wat), (0, Vec::new()));
}

#[test]
fn typed_callbacks_preserve_double_and_mixed_scalar_bits() {
    let wat = r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (type $double (func (param f64) (result f64)))
        (type $mixed (func (param i32 f32 f64 i64) (result i64)))
        (import "shellsim_ffi_v1" "closure_alloc_typed"
            (func $alloc (param i32 i32 i32 i32 i32 i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_release" (func $release (param i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 2 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (global (export "__stack_low") i32 (i32.const 1024))
        (global (export "__stack_high") i32 (i32.const 65536))
        (func $double_dispatch (param i32 i32 i32) (result i32)
            (f64.store (local.get 2)
                (f64.add (f64.load (local.get 1)) (f64.convert_i32_s (local.get 0))))
            (i32.const 0))
        (func $mixed_dispatch (param i32 i32 i32) (result i32)
            (i64.store (local.get 2)
                (i64.add (i64.load offset=24 (local.get 1))
                    (i64.extend_i32_s
                        (i32.add (i32.load (local.get 1))
                            (i32.trunc_f32_s (f32.load offset=8 (local.get 1)))))))
            (i32.const 0))
        (elem (i32.const 0) $double_dispatch $mixed_dispatch)
        (func (export "_start") (local $double_slot i32) (local $mixed_slot i32)
            (i32.store8 (i32.const 32) (i32.const 4))
            (if (call $alloc (i32.const 0) (i32.const 2) (i32.const 32)
                    (i32.const 1) (i32.const 4) (i32.const 256)) (then unreachable))
            (local.set $double_slot (i32.load (i32.const 256)))
            (if (f64.ne
                    (call_indirect (type $double) (f64.const 1.5) (local.get $double_slot))
                    (f64.const 3.5)) (then unreachable))
            (i32.store8 (i32.const 32) (i32.const 1))
            (i32.store8 (i32.const 33) (i32.const 3))
            (i32.store8 (i32.const 34) (i32.const 4))
            (i32.store8 (i32.const 35) (i32.const 2))
            (if (call $alloc (i32.const 1) (i32.const 0) (i32.const 32)
                    (i32.const 4) (i32.const 2) (i32.const 260)) (then unreachable))
            (local.set $mixed_slot (i32.load (i32.const 260)))
            (if (i64.ne
                    (call_indirect (type $mixed)
                        (i32.const 3) (f32.const 4) (f64.const 9.5) (i64.const 20)
                        (local.get $mixed_slot))
                    (i64.const 27)) (then unreachable))
            (if (i32.ne (global.get 0) (i32.const 65536)) (then unreachable))
            (if (call $release (local.get $double_slot)) (then unreachable))
            (if (call $release (local.get $mixed_slot)) (then unreachable))
            (if (i32.eqz (ref.is_null (table.get (local.get $double_slot))))
                (then unreachable))))"#;
    assert_eq!(run(wat), (0, Vec::new()));
}

#[test]
fn typed_callback_rejects_bad_tags_signatures_and_output_without_allocating() {
    let wat = r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (import "shellsim_ffi_v1" "closure_alloc_typed"
            (func $alloc (param i32 i32 i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 1 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (global (export "__stack_low") i32 (i32.const 1024))
        (global (export "__stack_high") i32 (i32.const 65536))
        (func $wrong (param i32 i32) (result i32) (i32.const 0))
        (elem (i32.const 0) $wrong)
        (func (export "_start")
            (i32.store8 (i32.const 32) (i32.const 5))
            (if (i32.ne
                    (call $alloc (i32.const 0) (i32.const 0) (i32.const 32)
                        (i32.const 1) (i32.const 4) (i32.const 256))
                    (i32.const 28)) (then unreachable))
            (i32.store8 (i32.const 32) (i32.const 4))
            (if (i32.ne
                    (call $alloc (i32.const 0) (i32.const 0) (i32.const 32)
                        (i32.const 1) (i32.const 4) (i32.const 256))
                    (i32.const 28)) (then unreachable))
            (if (i32.ne
                    (call $alloc (i32.const 0) (i32.const 0) (i32.const 32)
                        (i32.const 1) (i32.const 4) (i32.const 65534))
                    (i32.const 21)) (then unreachable))
            (if (i32.ne (table.size) (i32.const 1)) (then unreachable))))"#;
    assert_eq!(run(wat), (0, Vec::new()));
}

#[test]
fn typed_callback_nesting_restores_c_stack_and_tombstones_release() {
    let wat = r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (type $callback (func (param i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_alloc_typed"
            (func $alloc (param i32 i32 i32 i32 i32 i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_release" (func $release (param i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 1 funcref)
        (global $sp (export "__stack_pointer") (mut i32) (i32.const 65536))
        (global (export "__stack_low") i32 (i32.const 1024))
        (global (export "__stack_high") i32 (i32.const 65536))
        (func $dispatch (param i32 i32 i32) (result i32)
            (if (i32.eqz (i32.load (local.get 1)))
                (then (i32.store (local.get 2) (i32.const 0)))
                (else (i32.store (local.get 2)
                    (i32.add (i32.const 1)
                        (call_indirect (type $callback)
                            (i32.sub (i32.load (local.get 1)) (i32.const 1))
                            (i32.load (i32.const 256)))))))
            (i32.const 0))
        (elem (i32.const 0) $dispatch)
        (func (export "_start") (local $slot i32)
            (i32.store8 (i32.const 32) (i32.const 1))
            (if (call $alloc (i32.const 0) (i32.const 0) (i32.const 32)
                    (i32.const 1) (i32.const 1) (i32.const 256)) (then unreachable))
            (local.set $slot (i32.load (i32.const 256)))
            (if (i32.ne (call_indirect (type $callback) (i32.const 4) (local.get $slot))
                    (i32.const 4)) (then unreachable))
            (if (i32.ne (global.get $sp) (i32.const 65536)) (then unreachable))
            (if (call $release (local.get $slot)) (then unreachable))
            (if (i32.eqz (ref.is_null (table.get (local.get $slot))))
                (then unreachable))))"#;
    assert_eq!(run(wat), (0, Vec::new()));
}

#[test]
fn typed_callback_restores_c_stack_after_guest_exception() {
    let wat = r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (type $callback (func (param f64) (result f64)))
        (import "shellsim_ffi_v1" "closure_alloc_typed"
            (func $alloc (param i32 i32 i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 1 funcref)
        (global $sp (export "__stack_pointer") (mut i32) (i32.const 65536))
        (global (export "__stack_low") i32 (i32.const 1024))
        (global (export "__stack_high") i32 (i32.const 65536))
        (tag $error (param i32))
        (func $dispatch (param i32 i32 i32) (result i32)
            (throw $error (i32.const 17)))
        (elem (i32.const 0) $dispatch)
        (func (export "_start")
            (i32.store8 (i32.const 32) (i32.const 4))
            (if (call $alloc (i32.const 0) (i32.const 0) (i32.const 32)
                    (i32.const 1) (i32.const 4) (i32.const 256)) (then unreachable))
            (if (i32.ne
                    (block $caught (result i32)
                        (try_table (catch $error $caught)
                            (drop (call_indirect (type $callback)
                                (f64.const 2) (i32.load (i32.const 256)))))
                        (i32.const 0))
                    (i32.const 17)) (then unreachable))
            (if (i32.ne (global.get $sp) (i32.const 65536)) (then unreachable))))"#;
    assert_eq!(run(wat), (0, Vec::new()));
}

#[test]
fn typed_callback_rejects_a_frame_below_the_static_stack_floor() {
    let wat = r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (type $callback (func (param i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_alloc_typed"
            (func $alloc (param i32 i32 i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 1 funcref)
        (global $sp (export "__stack_pointer") (mut i32) (i32.const 65536))
        (global (export "__stack_low") i32 (i32.const 1024))
        (global (export "__stack_high") i32 (i32.const 65536))
        (func $dispatch (param i32 i32 i32) (result i32)
            (i32.store (local.get 2) (i32.load (local.get 1)))
            (i32.const 0))
        (elem (i32.const 0) $dispatch)
        (func (export "_start")
            (i32.store8 (i32.const 32) (i32.const 1))
            (if (call $alloc (i32.const 0) (i32.const 0) (i32.const 32)
                    (i32.const 1) (i32.const 1) (i32.const 256)) (then unreachable))
            (global.set $sp (i32.const 1100))
            (drop (call_indirect (type $callback)
                (i32.const 5) (i32.load (i32.const 256))))))"#;
    let (status, stderr) = run(wat);
    assert_eq!(status, 126);
    assert!(String::from_utf8(stderr)
        .unwrap()
        .contains("FFI callback C stack exhausted"));
}

#[test]
fn typed_callback_rejects_a_stack_pointer_in_the_heap() {
    let wat = r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (type $callback (func (param i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_alloc_typed"
            (func $alloc (param i32 i32 i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 2)
        (table (export "__indirect_function_table") 1 funcref)
        (global $sp (export "__stack_pointer") (mut i32) (i32.const 65536))
        (global (export "__stack_low") i32 (i32.const 1024))
        (global (export "__stack_high") i32 (i32.const 65536))
        (func $dispatch (param i32 i32 i32) (result i32)
            (i32.store (local.get 2) (i32.load (local.get 1)))
            (i32.const 0))
        (elem (i32.const 0) $dispatch)
        (func (export "_start")
            (i32.store8 (i32.const 32) (i32.const 1))
            (if (call $alloc (i32.const 0) (i32.const 0) (i32.const 32)
                    (i32.const 1) (i32.const 1) (i32.const 256)) (then unreachable))
            (global.set $sp (i32.const 70000))
            (drop (call_indirect (type $callback)
                (i32.const 5) (i32.load (i32.const 256))))))"#;
    let (status, stderr) = run(wat);
    assert_eq!(status, 126);
    assert!(String::from_utf8(stderr)
        .unwrap()
        .contains("FFI callback C stack pointer is outside linker bounds"));
}

fn typed_callback_wait_guest(wait_ns: u64) -> Vec<u8> {
    wat::parse_str(format!(
        r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (type $callback (func (param f64) (result f64)))
        (import "shellsim_ffi_v1" "closure_alloc_typed"
            (func $alloc (param i32 i32 i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "poll_oneoff"
            (func $poll (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 1 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (global (export "__stack_low") i32 (i32.const 1024))
        (global (export "__stack_high") i32 (i32.const 65536))
        (func $dispatch (param i32 i32 i32) (result i32)
            (i32.store (i32.const 16) (i32.const 1))
            (i64.store (i32.const 24) (i64.const {wait_ns}))
            (if (call $poll (i32.const 0) (i32.const 100) (i32.const 1) (i32.const 200))
                (then unreachable))
            (f64.store (local.get 2)
                (f64.add (f64.load (local.get 1)) (f64.const 2)))
            (i32.const 0))
        (elem (i32.const 0) $dispatch)
        (func (export "_start")
            (i32.store8 (i32.const 32) (i32.const 4))
            (if (call $alloc (i32.const 0) (i32.const 0) (i32.const 32)
                    (i32.const 1) (i32.const 4) (i32.const 256)) (then unreachable))
            (if (f64.ne
                    (call_indirect (type $callback)
                        (f64.const 1.5) (i32.load (i32.const 256)))
                    (f64.const 3.5)) (then unreachable))))"#
    ))
    .unwrap()
}

#[test]
fn timeout_cancels_a_typed_callback_without_retaining_its_stack() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 2_000_000,
        memory: 64 * 1024 * 1024,
        ..Limits::default()
    });
    environment
        .vfs
        .write(
            "/",
            "/app",
            &typed_callback_wait_guest(10_000_000_000),
            0o755,
        )
        .unwrap();
    let first = environment.run_script_capture("timeout 1 /app; echo $?");
    assert_eq!(
        (first.0.exit_status, first.1, first.2),
        (0, b"124\n".to_vec(), Vec::new())
    );
    let first_retained = environment.resources.memory_mark();
    let second = environment.run_script_capture("timeout 1 /app; echo $?");
    assert_eq!(
        (second.0.exit_status, second.1, second.2),
        (0, b"124\n".to_vec(), Vec::new())
    );
    assert_eq!(environment.resources.memory_mark(), first_retained);
    assert_eq!(environment.clock.monotonic_ns(), 2_000_000_000);
}

fn recursive_invoke_guest(depth: u32) -> Vec<u8> {
    wat::parse_str(format!(r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (import "shellsim_ffi_v1" "invoke" (func $invoke (param i32 i32 i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 1 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (func $recurse (param $depth i32) (result i32)
            (if (i32.eqz (local.get $depth)) (then (return (i32.const 0))))
            (i64.store (i32.const 64) (i64.extend_i32_u
                (i32.sub (local.get $depth) (i32.const 1))))
            (if (call $invoke (i32.const 0) (i32.const 32) (i32.const 64)
                    (i32.const 1) (i32.const 1) (i32.const 128))
                (then unreachable))
            (i32.add (i32.load (i32.const 128)) (i32.const 1)))
        (elem (i32.const 0) $recurse)
        (func (export "_start")
            (i32.store8 (i32.const 32) (i32.const 1))
            (i64.store (i32.const 64) (i64.const {depth}))
            (if (call $invoke (i32.const 0) (i32.const 32) (i32.const 64)
                    (i32.const 1) (i32.const 1) (i32.const 128))
                (then unreachable))
            (if (i32.ne (i32.load (i32.const 128)) (i32.const {depth}))
                (then unreachable))))"#)).unwrap()
}

#[test]
fn nested_invoke_has_a_bounded_fiber_depth_and_releases_on_failure() {
    let peak_at_depth = |depth| {
        let mut environment = Environment::with_limits(Limits {
            cpu: 2_000_000,
            memory: 64 * 1024 * 1024,
            ..Limits::default()
        });
        environment
            .vfs
            .write("/", "/app", &recursive_invoke_guest(depth), 0o755)
            .unwrap();
        let (outcome, _, stderr) = environment.run_script_capture("/app");
        assert_eq!((outcome.exit_status, stderr), (0, Vec::new()));
        outcome.usage.memory_peak
    };
    assert_eq!(peak_at_depth(2) - peak_at_depth(1), 2 * 1024 * 1024);

    let mut environment = Environment::with_limits(Limits {
        cpu: 2_000_000,
        memory: 64 * 1024 * 1024,
        ..Limits::default()
    });
    environment
        .vfs
        .write("/", "/app", &recursive_invoke_guest(6), 0o755)
        .unwrap();
    let baseline = environment.resources.memory_mark();
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!((outcome.exit_status, stderr), (0, Vec::new()));
    assert_eq!(environment.resources.memory_mark(), baseline);

    environment
        .vfs
        .write("/", "/app", &recursive_invoke_guest(9), 0o755)
        .unwrap();
    let baseline = environment.resources.memory_mark();
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!(outcome.exit_status, 126);
    assert!(String::from_utf8(stderr)
        .unwrap()
        .contains("nested Wasm fiber limit exceeded"));
    let after_first = environment.resources.memory_mark();
    assert!(after_first.saturating_sub(baseline) < 1024);
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!(outcome.exit_status, 126);
    assert!(String::from_utf8(stderr)
        .unwrap()
        .contains("nested Wasm fiber limit exceeded"));
    // A failed shell command can retain a small binding; no 2 MiB fiber charge survives.
    assert!(
        environment
            .resources
            .memory_mark()
            .saturating_sub(after_first)
            < 1024
    );
}

#[test]
fn repeated_callbacks_reuse_one_nested_fiber_reservation() {
    let wat = r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-v2")
        (type $callback (func (param i32) (result i32)))
        (import "shellsim_ffi_v1" "closure_alloc" (func $alloc (param i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 1 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (func $dispatch (param i32 i32) (result i32) (i32.add (local.get 0) (local.get 1)))
        (elem (i32.const 0) $dispatch)
        (func (export "_start") (local $i i32)
            (if (call $alloc (i32.const 0) (i32.const 10) (i32.const 256)) (then unreachable))
            (block $done (loop $again
                (br_if $done (i32.ge_u (local.get $i) (i32.const 100)))
                (if (i32.ne (call_indirect (type $callback)
                        (local.get $i) (i32.load (i32.const 256)))
                        (i32.add (local.get $i) (i32.const 10)))
                    (then unreachable))
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br $again)))))"#;
    let mut environment = Environment::with_limits(Limits {
        cpu: 2_000_000,
        memory: 6 * 1024 * 1024,
        ..Limits::default()
    });
    environment
        .vfs
        .write("/", "/app", &wat::parse_str(wat).unwrap(), 0o755)
        .unwrap();
    let baseline = environment.resources.memory_mark();
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!((outcome.exit_status, stderr), (0, Vec::new()));
    assert_eq!(environment.resources.memory_mark(), baseline);
    assert!(outcome.usage.memory_peak < 6 * 1024 * 1024);
}

#[test]
#[ignore = "requires ports/ffi_proof/build.py SDK 34 artifacts"]
fn separately_compiled_sdk_provider_calls_and_callbacks() {
    let artifacts = PathBuf::from(
        std::env::var_os("SHELLSIM_FFI_PROOF_ARTIFACTS")
            .expect("set SHELLSIM_FFI_PROOF_ARTIFACTS to the proof artifact directory"),
    );
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000_000,
        memory: 256 * 1024 * 1024,
        ..Limits::default()
    });
    environment.vfs.mkdir_all("/", "/lib").unwrap();
    environment
        .vfs
        .write(
            "/",
            "/lib/libffi_proof.so",
            &std::fs::read(artifacts.join("libffi_proof.so")).unwrap(),
            0o644,
        )
        .unwrap();
    environment
        .vfs
        .write(
            "/",
            "/app",
            &std::fs::read(artifacts.join("ffi_proof.wasm")).unwrap(),
            0o755,
        )
        .unwrap();
    let (outcome, stdout, stderr) = environment.run_script_capture("/app");
    assert_eq!((outcome.exit_status, stderr), (0, Vec::new()), "{stdout:?}");
    assert_eq!(
        stdout,
        b"separate SDK provider, side FFI import and nested callback: ok\n"
    );
}
