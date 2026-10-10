// Small native Wasm fixtures isolate executable linkage from host compilers.
// Real C consumers are checked separately with pinned SDK artifacts.
use shellsim::{Environment, Limits};

fn main_module(body: &str, worker: bool, missing: &str) -> Vec<u8> {
    let worker_start = if worker {
        r#"(global.set $tls_base (i32.const 2048))
            (i32.store (i32.const 2048) (i32.const 77))
            (call $ready (i32.const 32768) (i32.const 65536))
            (if (i32.ne (call $main_tls_read) (i32.const 77)) (then unreachable))
            (if (i32.ne (call $tls_read) (i32.const 7)) (then unreachable))
            (call $tls_write (i32.const 9))
            (if (i32.ne (call $answer) (i32.const 42)) (then unreachable))
            (i32.atomic.store (i32.const 20) (i32.const 1))"#
    } else {
        ""
    };
    let worker_main = if worker {
        r#"(if (i32.lt_s (call $spawn (i32.const 0)) (i32.const 0)) (then unreachable))
            (loop $wait (br_if $wait (i32.eqz (i32.atomic.load (i32.const 20)))))
            (if (i32.ne (call $main_tls_read) (i32.const 33)) (then unreachable))
            (if (i32.ne (call $tls_read) (i32.const 7)) (then unreachable))
            (if (i32.ne (i32.load (i32.add (global.get $shared) (i32.const 8)))
                (i32.const 1)) (then unreachable))"#
    } else {
        ""
    };
    wat::parse_str(format!(r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-threads-v3")
        (@custom "dylink.0" "\81\13\11shellsim.main-tls\01\02\09\01\07root.so\03\0c\01\08main_tls\80\02")
        (import "env" "memory" (memory 1 4 shared))
        (import "env" "answer" (func $answer (result i32)))
        (import "env" "main_tls_read" (func $main_tls_read (result i32)))
        (import "env" "tls_read" (func $tls_read (result i32)))
        (import "env" "tls_write" (func $tls_write (param i32)))
        (import "GOT.mem" "shared" (global $shared (mut i32)))
        (import "GOT.func" "answer" (global $answer_slot (mut i32)))
        (import "shellsim_threads_v2" "thread_ready" (func $ready (param i32 i32)))
        (import "wasi" "thread-spawn" (func $spawn (param i32) (result i32)))
        {missing}
        (export "memory" (memory 0))
        (export "answer" (func $answer))
        (export "shared" (global $shared))
        (table (export "__indirect_function_table") 1 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (global (export "__stack_low") i32 (i32.const 32768))
        (global (export "__stack_high") i32 (i32.const 65536))
        (global $tls_base (export "__tls_base") (mut i32) (i32.const 1024))
        (global (export "main_tls") i32 (i32.const 0))
        (global $heap (mut i32) (i32.const 4096))
        (func (export "malloc") (param $size i32) (result i32) (local $base i32)
            (if (i32.ne (i32.load (i32.const 16)) (i32.const 12345)) (then unreachable))
            (local.set $base (global.get $heap))
            (global.set $heap (i32.add (global.get $heap) (local.get $size)))
            (local.get $base))
        (func (export "main_callback") (result i32)
            (i32.add (call $answer) (i32.load (i32.const 4))))
        (func (export "spawn_in_constructor")
            (drop (call $spawn (i32.const 0))))
        (func $init_memory (export "__wasm_init_memory")
            (if (i32.eqz (i32.load (i32.const 0))) (then
                (i32.store (i32.const 0) (i32.const 1))
                (i32.store (i32.const 4) (i32.const 10))
                (i32.store (i32.const 16) (i32.const 12345))
                (i32.store (i32.const 1024) (i32.const 33)))))
        (func $global_relocs (export "__wasm_apply_global_relocs")
            (i32.store (i32.const 8) (global.get $shared)))
        (func $initialize (call $global_relocs) (call $init_memory))
        (start $initialize)
        (func (export "wasi_thread_start") (param i32 i32) {worker_start})
        (func $main_ctors (export "__wasm_call_ctors")
            (if (i32.ne (i32.load (i32.add (global.get $shared) (i32.const 4)))
                (i32.const 52)) (then unreachable)))
        (func (export "_start")
            (if (i32.ne (call $answer) (i32.const 42)) (then unreachable))
            (if (i32.ne (call_indirect (result i32) (global.get $answer_slot))
                (i32.const 42)) (then unreachable))
            (if (i32.ne (i32.load (global.get $shared)) (i32.const 42)) (then unreachable))
            (if (i32.ne (i32.load (i32.const 8)) (global.get $shared)) (then unreachable))
            (if (i32.ne (call $main_tls_read) (i32.const 33)) (then unreachable))
            (call $main_ctors)
            {worker_main} {body}))"#)).unwrap()
}

fn side_module(needed: &str, answer: &str, constructor: &str) -> Vec<u8> {
    side_module_with_type(needed, answer, constructor, "i32")
}

fn side_module_with_type(needed: &str, answer: &str, constructor: &str, result: &str) -> Vec<u8> {
    wat::parse_str(format!(
        r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-threads-v3")
        (@custom "dylink.0" "\01\04\40\00\00\00\80\18\16shellsim.deferred-init\01{needed}")
        (import "env" "memory" (memory 1 4 shared))
        (import "env" "__memory_base" (global $base i32))
        (import "env" "main_callback" (func $callback (result i32)))
        (import "env" "spawn_in_constructor" (func $spawn))
        (import "GOT.mem" "main_tls" (global $main_tls (mut i32)))
        (global (export "shared") i32 (i32.const 0))
        (global (export "relocated_data") i32 (i32.const 12))
        (global (export "__tls_size") i32 (i32.const 16))
        (global (export "__tls_align") i32 (i32.const 4))
        (global $tls (mut i32) (i32.const 0))
        (func (export "__wasm_init_memory")
            (i32.store (global.get $base) (i32.const 42)))
        (func (export "__wasm_apply_data_relocs")
            (i32.store (i32.add (global.get $base) (i32.const 12)) (i32.const 314)))
        (func (export "__wasm_init_tls") (param i32)
            (global.set $tls (local.get 0))
            (i32.store (global.get $tls) (i32.const 7)))
        (func (export "answer") (result {result}) {answer})
        (func (export "main_tls_read") (result i32) (i32.load (global.get $main_tls)))
        (func (export "tls_read") (result i32) (i32.load (global.get $tls)))
        (func (export "tls_write") (param i32) (i32.store (global.get $tls) (local.get 0)))
        (func (export "__wasm_call_ctors")
            (if (i32.eqz (i32.load (i32.const 8))) (then unreachable))
            (i32.store (i32.add (global.get $base) (i32.const 4)) (call $callback))
            (i32.store (i32.add (global.get $base) (i32.const 8))
                (i32.add (i32.load (i32.add (global.get $base) (i32.const 8))) (i32.const 1)))
            {constructor}))"#
    ))
    .unwrap()
}

fn linked_environment(main: &[u8], side: &[u8]) -> Environment {
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000_000,
        memory: 64 * 1024 * 1024,
        ..Limits::default()
    });
    environment.vfs.mkdir_all("/", "/lib").unwrap();
    environment.vfs.write("/", "/app", main, 0o755).unwrap();
    environment
        .vfs
        .write("/", "/lib/root.so", side, 0o644)
        .unwrap();
    environment
}

#[test]
fn executable_binds_functions_data_and_constructor_callbacks_before_main() {
    let mut environment = linked_environment(
        &main_module("", false, ""),
        &side_module("", "(i32.load (global.get $base))", ""),
    );
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
}

#[test]
fn worker_rebinds_main_tls_and_keeps_shared_data_and_side_tls_separate() {
    let mut environment = linked_environment(
        &main_module("", true, ""),
        &side_module("", "(i32.load (global.get $base))", ""),
    );
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
}

#[test]
fn declared_root_precedes_its_dependency_when_both_export_a_function() {
    let extra = r#"(import "env" "call_collision" (func $collision_call (result i32)))"#;
    let check = "(if (i32.ne (call $collision_call) (i32.const 84)) (then unreachable))";
    let mut environment = linked_environment(
        &main_module(check, false, extra),
        &side_module(r"\02\08\01\06dep.so", "(i32.load (global.get $base))", ""),
    );
    let dependency = wat::parse_str(
        r#"(module
        (@custom "shellsim.abi" "shellsim-wasi-sdk34-cpython3137-threads-v3")
        (@custom "dylink.0" "\01\04\00\00\00\00\80\18\16shellsim.deferred-init\01")
        (import "env" "memory" (memory 1 4 shared))
        (import "env" "__indirect_function_table" (table 1 funcref))
        (import "GOT.func" "answer" (global $answer (mut i32)))
        (import "GOT.mem" "relocated_data" (global $root_data (mut i32)))
        (import "env" "answer" (func $preempted (result i32)))
        (func (export "answer") (result i32) (i32.const 99))
        (func (export "call_collision") (result i32)
            (i32.add (call $preempted)
                (call_indirect (result i32) (global.get $answer))))
        (func (export "__wasm_call_ctors")
            (if (i32.ne (i32.load (global.get $root_data)) (i32.const 314))
                (then unreachable))))"#,
    )
    .unwrap();
    environment
        .vfs
        .write("/", "/lib/dep.so", &dependency, 0o644)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
}

#[test]
fn strong_function_and_type_mismatch_fail_before_main_entry() {
    let missing = r#"(import "env" "absent" (func))"#;
    let mut environment = linked_environment(
        &main_module("", false, missing),
        &side_module("", "(i32.load (global.get $base))", ""),
    );
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!(outcome.exit_status, 126);
    assert!(String::from_utf8_lossy(&stderr).contains("missing executable function: absent"));
    let mut environment = linked_environment(
        &main_module("", false, ""),
        &side_module_with_type("", "(i64.const 42)", "", "i64"),
    );
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!(outcome.exit_status, 126);
    assert!(String::from_utf8_lossy(&stderr).contains("executable function type mismatch: answer"));
}

#[test]
fn strong_data_import_and_missing_dependency_fail_before_main_entry() {
    let missing = r#"(import "GOT.mem" "absent_data" (global (mut i32)))"#;
    let mut environment = linked_environment(
        &main_module("", false, missing),
        &side_module("", "(i32.load (global.get $base))", ""),
    );
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!(outcome.exit_status, 126);
    assert!(String::from_utf8_lossy(&stderr).contains("missing executable GOT symbol: absent_data"));
    let mut environment = linked_environment(
        &main_module("", false, ""),
        &side_module(r"\02\08\01\06dep.so", "(i32.load (global.get $base))", ""),
    );
    let (outcome, _, _) = environment.run_script_capture("/app");
    assert_eq!(outcome.exit_status, 126);
}

#[test]
fn eager_executable_still_enforces_resource_budgets_and_import_namespaces() {
    let main = main_module("", false, r#"(import "host" "execute" (func))"#);
    let mut environment =
        linked_environment(&main, &side_module("", "(i32.load (global.get $base))", ""));
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!(outcome.exit_status, 126);
    assert!(String::from_utf8_lossy(&stderr).contains("unsupported wasm import: host.execute"));
    let main = main_module("", false, "");
    let mut environment = Environment::with_limits(Limits {
        cpu: 1,
        ..Limits::default()
    });
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    let (outcome, _, _) = environment.run_script_capture("/app");
    assert_eq!(outcome.exit_status, 137);
}

#[test]
fn constructor_cannot_spawn_a_worker_during_initialization() {
    let mut environment = linked_environment(
        &main_module("", false, ""),
        &side_module("", "(i32.load (global.get $base))", "(call $spawn)"),
    );
    let (outcome, _, stderr) = environment.run_script_capture("/app");
    assert_eq!(outcome.exit_status, 126);
    assert!(String::from_utf8_lossy(&stderr)
        .contains("blocking during threaded dynamic initialization is unsupported"));
}

#[test]
#[ignore = "requires pinned native executable and shared-library artifacts"]
fn native_c_consumers_load_their_declared_shared_libraries() {
    let executables = std::env::var_os("SHELLSIM_SHARED_EXECUTABLES").unwrap();
    let libraries = std::env::var_os("SHELLSIM_SHARED_LIBRARIES").unwrap();
    for executable in std::env::split_paths(&executables) {
        let mut environment = Environment::with_limits(Limits {
            cpu: 100_000_000_000,
            memory: 2 * 1024 * 1024 * 1024,
            disk: 256 * 1024 * 1024,
            ..Limits::default()
        });
        environment.vfs.mkdir_all("/", "/lib").unwrap();
        for library in std::env::split_paths(&libraries) {
            let target = format!("/lib/{}", library.file_name().unwrap().to_str().unwrap());
            environment
                .vfs
                .write("/", &target, &std::fs::read(library).unwrap(), 0o644)
                .unwrap();
        }
        environment
            .vfs
            .write(
                "/",
                "/consumer",
                &std::fs::read(&executable).unwrap(),
                0o755,
            )
            .unwrap();
        let (outcome, stdout, stderr) = environment.run_script_capture("/consumer");
        assert_eq!(
            outcome.exit_status,
            0,
            "{}: stdout={} stderr={}",
            executable.display(),
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        );
        println!(
            "{}: {}",
            executable.display(),
            String::from_utf8_lossy(&stdout)
        );
    }
}
