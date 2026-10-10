// Independent guest images verify actual copied argv, PID continuity, process
// completion and resource admission without invoking a host or guest compiler.
use shellsim::{Environment, Limits};

#[derive(Clone, Copy)]
enum Launch {
    Spawn,
    Exec,
}

fn caller(launch: Launch, argc: u32, envc: u32, extra: &str) -> Vec<u8> {
    let operation = match launch {
        Launch::Spawn => {
            r#"
            (local.set $errno (call $spawn (i32.const 128) (i32.const 1024)
                (i32.const ARGC) (i32.const 20000) (i32.const ENVC)
                (i32.const 0) (i32.const 0) (i32.const 0) (i32.const 0) (i32.const 64)))
            (if (local.get $errno) (then (call $exit (local.get $errno))))
            (if (call $wait (i32.load (i32.const 64)) (i32.const 0)
                (i32.const 80) (i32.const 84)) (then unreachable))
            (if (i32.or (i32.load (i32.const 80))
                (i32.ne (i32.load (i32.const 64)) (i32.load (i32.const 84))))
                (then unreachable))"#
        }
        Launch::Exec => {
            r#"
            (call $exit (call $exec (i32.const 128) (i32.const 1024)
                (i32.const ARGC) (i32.const 20000) (i32.const ENVC) (i32.const 0)))"#
        }
    }
    .replace("ARGC", &argc.to_string())
    .replace("ENVC", &envc.to_string());
    wat::parse_str(format!(
        r#"(module
        (import "shellsim_posix_v1" "process_spawn" (func $spawn
            (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
        (import "shellsim_posix_v1" "process_exec" (func $exec
            (param i32 i32 i32 i32 i32 i32) (result i32)))
        (import "shellsim_posix_v1" "process_wait" (func $wait
            (param i32 i32 i32 i32) (result i32)))
        (import "shellsim_posix_v1" "process_identity" (func $pid (param i32) (result i32)))
        (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
        (import "wasi_snapshot_preview1" "fd_write" (func $write
            (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 3)
        (data (i32.const 128) "/child\00")
        (data (i32.const 256) "X=V\00")
        (data (i32.const 512) "decode\0a")
        (func (export "_start") (local $index i32) (local $digit i32)
            (local $pointer i32) (local $identity i32) (local $errno i32)
            (loop $arguments
                (local.set $pointer (i32.add (i32.const 32768)
                    (i32.mul (local.get $index) (i32.const 16))))
                (i32.store (i32.add (i32.const 1024)
                    (i32.mul (local.get $index) (i32.const 4))) (local.get $pointer))
                (i32.store8 (local.get $pointer) (i32.const 97))
                (local.set $digit (i32.const 0))
                (loop $digits
                    (i32.store8 (i32.add (local.get $pointer)
                        (i32.add (local.get $digit) (i32.const 1)))
                        (i32.add (i32.const 65) (i32.and (i32.const 15)
                            (i32.shr_u (local.get $index) (i32.mul (local.get $digit) (i32.const 4))))))
                    (local.set $digit (i32.add (local.get $digit) (i32.const 1)))
                    (br_if $digits (i32.lt_u (local.get $digit) (i32.const 4))))
                (local.set $index (i32.add (local.get $index) (i32.const 1)))
                (br_if $arguments (i32.lt_u (local.get $index) (i32.const {argc}))))
            (local.set $identity (call $pid (i32.const 0)))
            (i32.store8 (i32.const 32768) (i32.const 112))
            (local.set $digit (i32.const 0))
            (loop $identity_digits
                (i32.store8 (i32.add (i32.const 32769) (local.get $digit))
                    (i32.add (i32.const 65) (i32.and (i32.const 15)
                        (i32.shr_u (local.get $identity) (i32.mul (local.get $digit) (i32.const 4))))))
                (local.set $digit (i32.add (local.get $digit) (i32.const 1)))
                (br_if $identity_digits (i32.lt_u (local.get $digit) (i32.const 8))))
            (local.set $index (i32.const 0))
            (if (i32.const {envc}) (then (loop $environment
                (i32.store (i32.add (i32.const 20000)
                    (i32.mul (local.get $index) (i32.const 4))) (i32.const 256))
                (local.set $index (i32.add (local.get $index) (i32.const 1)))
                (br_if $environment (i32.lt_u (local.get $index) (i32.const {envc}))))))
            {extra}
            (i32.store (i32.const 32) (i32.const 512))
            (i32.store (i32.const 36) (i32.const 7))
            (if (call $write (i32.const 1) (i32.const 32) (i32.const 1) (i32.const 48))
                (then unreachable))
            {operation}))"#
    ))
    .unwrap()
}

fn child(check_pid: bool) -> Vec<u8> {
    let identity = if check_pid {
        r#"(if (i32.ne (local.get $identity) (call $pid (i32.const 0))) (then unreachable))"#
    } else {
        ""
    };
    wat::parse_str(format!(
        r#"(module
        (import "shellsim_posix_v1" "process_identity" (func $pid (param i32) (result i32)))
        (import "wasi_snapshot_preview1" "args_sizes_get" (func $sizes (param i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "args_get" (func $args (param i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_write" (func $write
            (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 3)
        (data (i32.const 512) "ordered\0a")
        (func (export "_start") (local $index i32) (local $digit i32)
            (local $pointer i32) (local $identity i32)
            (if (call $sizes (i32.const 16) (i32.const 20)) (then unreachable))
            (if (i32.ne (i32.load (i32.const 16)) (i32.const 1500)) (then unreachable))
            (if (call $args (i32.const 1024) (i32.const 32768)) (then unreachable))
            (local.set $pointer (i32.load (i32.const 1024)))
            (if (i32.ne (i32.load8_u (local.get $pointer)) (i32.const 112)) (then unreachable))
            (loop $identity_digits
                (local.set $identity (i32.or (local.get $identity)
                    (i32.shl (i32.sub (i32.load8_u (i32.add (local.get $pointer)
                        (i32.add (local.get $digit) (i32.const 1)))) (i32.const 65))
                        (i32.mul (local.get $digit) (i32.const 4)))))
                (local.set $digit (i32.add (local.get $digit) (i32.const 1)))
                (br_if $identity_digits (i32.lt_u (local.get $digit) (i32.const 8))))
            {identity}
            (if (i32.load8_u (i32.add (local.get $pointer) (i32.const 9))) (then unreachable))
            (local.set $index (i32.const 1))
            (loop $arguments
                (local.set $pointer (i32.load (i32.add (i32.const 1024)
                    (i32.mul (local.get $index) (i32.const 4)))))
                (if (i32.ne (i32.load8_u (local.get $pointer)) (i32.const 97)) (then unreachable))
                (local.set $digit (i32.const 0))
                (loop $digits
                    (if (i32.ne (i32.load8_u (i32.add (local.get $pointer)
                        (i32.add (local.get $digit) (i32.const 1))))
                        (i32.add (i32.const 65) (i32.and (i32.const 15)
                            (i32.shr_u (local.get $index) (i32.mul (local.get $digit) (i32.const 4))))))
                        (then unreachable))
                    (local.set $digit (i32.add (local.get $digit) (i32.const 1)))
                    (br_if $digits (i32.lt_u (local.get $digit) (i32.const 4))))
                (if (i32.load8_u (i32.add (local.get $pointer) (i32.const 5))) (then unreachable))
                (local.set $index (i32.add (local.get $index) (i32.const 1)))
                (br_if $arguments (i32.lt_u (local.get $index) (i32.const 1500))))
            (i32.store (i32.const 32) (i32.const 512))
            (i32.store (i32.const 36) (i32.const 8))
            (if (call $write (i32.const 1) (i32.const 32) (i32.const 1) (i32.const 48))
                (then unreachable))))"#
    ))
    .unwrap()
}

fn environment(launch: Launch, argc: u32, envc: u32, extra: &str, limits: Limits) -> Environment {
    let mut environment = Environment::with_limits(limits);
    environment
        .vfs
        .write("/", "/caller", &caller(launch, argc, envc, extra), 0o755)
        .unwrap();
    environment
        .vfs
        .write("/", "/child", &child(matches!(launch, Launch::Exec)), 0o755)
        .unwrap();
    environment
}

#[test]
fn spawn_and_exec_deliver_1500_ordered_arguments_and_release_launch_metadata() {
    for launch in [Launch::Spawn, Launch::Exec] {
        let mut environment = environment(launch, 1500, 1, "", Limits::default());
        for iteration in 0..4 {
            let before = environment.resources.memory_mark();
            let (outcome, stdout, stderr) = environment.run_script_capture("/caller");
            assert_eq!(
                outcome.exit_status,
                0,
                "{}",
                String::from_utf8_lossy(&stderr)
            );
            assert_eq!(stdout, b"decode\nordered\n");
            // The second shell entry charges surviving root-shell bindings.
            // Later runs must release all launch and replacement metadata.
            if iteration > 1 {
                assert_eq!(environment.resources.memory_mark(), before);
            }
        }
    }
}

#[test]
fn count_and_combined_string_bounds_reject_without_entering_the_child() {
    let oversized_strings = r#"
        (memory.fill (i32.const 100000) (i32.const 120) (i32.const 70000))
        (i32.store (i32.const 1024) (i32.const 100000))
        (i32.store (i32.const 1028) (i32.const 100000))"#;
    for launch in [Launch::Spawn, Launch::Exec] {
        for (argc, envc, extra) in [(4097, 0, ""), (1500, 257, ""), (1500, 0, oversized_strings)] {
            let mut environment = environment(launch, argc, envc, extra, Limits::default());
            let (outcome, stdout, stderr) = environment.run_script_capture("/caller");
            assert_eq!(
                outcome.exit_status,
                28,
                "{}",
                String::from_utf8_lossy(&stderr)
            );
            assert_eq!(stdout, b"decode\n");
        }
    }
}
