// Tiny independent Wasm images exercise virtual exec and descriptor atomicity directly.
use shellsim::Environment;

fn install(environment: &mut Environment, path: &str, source: &str) {
    let bytes = wat::parse_str(source).unwrap();
    environment.vfs.write("/", path, &bytes, 0o755).unwrap();
}

fn replacement(path: &str) -> String {
    format!(
        r#"(module
        (import "shellsim_posix_v1" "process_exec" (func $exec (param i32 i32 i32 i32 i32 i32) (result i32)))
        (import "shellsim_posix_v1" "process_identity" (func $pid (param i32) (result i32)))
        (memory (export "memory") 1)
        (data (i32.const 128) "{path}\00")
        (data (i32.const 256) "customzero\00")
        (func (export "_start") (local $index i32) (local $value i32)
            (i32.store (i32.const 32) (i32.const 256))
            (i32.store (i32.const 36) (i32.const 400))
            (local.set $value (call $pid (i32.const 0)))
            (loop $encode
                (i32.store8 (i32.add (i32.const 400) (local.get $index))
                    (i32.add (i32.and (i32.shr_u (local.get $value) (i32.mul (local.get $index) (i32.const 4))) (i32.const 15)) (i32.const 65)))
                (local.set $index (i32.add (local.get $index) (i32.const 1)))
                (br_if $encode (i32.lt_u (local.get $index) (i32.const 8))))
            (drop (call $exec (i32.const 128) (i32.const 32) (i32.const 2) (i32.const 0) (i32.const 0) (i32.const 0)))
            unreachable))"#
    )
}

#[test]
fn exec_transfers_to_independent_image_and_keeps_custom_argv_zero() {
    let mut environment = Environment::new();
    install(&mut environment, "/first", &replacement("/second"));
    install(
        &mut environment,
        "/second",
        r#"(module
        (import "shellsim_posix_v1" "process_identity" (func $pid (param i32) (result i32)))
        (import "wasi_snapshot_preview1" "args_get" (func $args (param i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (func (export "_start") (local $index i32) (local $value i32)
            (if (call $args (i32.const 16) (i32.const 128)) (then unreachable))
            (loop $decode
                (local.set $value (i32.or (local.get $value)
                    (i32.shl (i32.sub (i32.load8_u (i32.add (i32.load (i32.const 20)) (local.get $index))) (i32.const 65)) (i32.mul (local.get $index) (i32.const 4)))))
                (local.set $index (i32.add (local.get $index) (i32.const 1)))
                (br_if $decode (i32.lt_u (local.get $index) (i32.const 8))))
            (if (i32.ne (call $pid (i32.const 0)) (local.get $value)) (then unreachable))
            (i32.store (i32.const 32) (i32.load (i32.const 16)))
            (i32.store (i32.const 36) (i32.const 10))
            (if (call $write (i32.const 1) (i32.const 32) (i32.const 1) (i32.const 48)) (then unreachable))))"#,
    );
    let (outcome, stdout, stderr) = environment.run_script_capture("/first");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"customzero");
    // Re-entry charges surviving root-shell bindings independently from image metadata.
    assert_eq!(environment.run_script_capture("/first").0.exit_status, 0);
    let baseline = environment.resources.memory_mark();
    for _ in 0..8 {
        let (outcome, _, stderr) = environment.run_script_capture("/first");
        assert_eq!(
            outcome.exit_status,
            0,
            "{}",
            String::from_utf8_lossy(&stderr)
        );
        assert_eq!(environment.resources.memory_mark(), baseline);
    }
}

#[test]
fn failed_exec_returns_enoexec_and_old_guest_continues() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/broken", b"\0asm\x01", 0o755)
        .unwrap();
    install(
        &mut environment,
        "/first",
        r#"(module
        (import "shellsim_posix_v1" "process_exec" (func $exec (param i32 i32 i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
        (memory (export "memory") 1)
        (data (i32.const 128) "/broken\00")
        (func (export "_start")
            (i32.store (i32.const 32) (i32.const 128))
            (if (i32.ne (call $exec (i32.const 128) (i32.const 32) (i32.const 1) (i32.const 0) (i32.const 0) (i32.const 0)) (i32.const 45)) (then unreachable))
            (call $exit (i32.const 7))))"#,
    );
    assert_eq!(environment.run_script_capture("/first").0.exit_status, 7);
}

#[test]
fn atomic_open_sets_cloexec_and_exec_closes_only_flagged_alias() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/value", b"value", 0o644)
        .unwrap();
    install(
        &mut environment,
        "/first",
        r#"(module
        (import "shellsim_posix_v1" "descriptor_open" (func $open (param i32 i32 i32 i32) (result i32)))
        (import "shellsim_posix_v1" "descriptor_control" (func $control (param i32 i32 i32 i32) (result i32)))
        (import "shellsim_posix_v1" "process_exec" (func $exec (param i32 i32 i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (data (i32.const 128) "/value\00")
        (data (i32.const 256) "/second\00")
        (func (export "_start") (local $fd i32)
            (local.set $fd (call $open (i32.const -2) (i32.const 128) (i32.const 67633152) (i32.const 0)))
            (if (i32.ne (local.get $fd) (i32.const 5)) (then unreachable))
            (if (call $control (local.get $fd) (i32.const 1) (i32.const 0) (i32.const 48)) (then unreachable))
            (if (i32.ne (i32.load (i32.const 48)) (i32.const 1)) (then unreachable))
            (if (call $control (local.get $fd) (i32.const 3) (i32.const 6) (i32.const 48)) (then unreachable))
            (i32.store (i32.const 32) (i32.const 256))
            (drop (call $exec (i32.const 256) (i32.const 32) (i32.const 1) (i32.const 0) (i32.const 0) (i32.const 0)))
            unreachable))"#,
    );
    install(
        &mut environment,
        "/second",
        r#"(module
        (import "shellsim_posix_v1" "descriptor_control" (func $control (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (func (export "_start")
            (if (i32.ne (call $control (i32.const 5) (i32.const 1) (i32.const 0) (i32.const 48)) (i32.const 8)) (then unreachable))
            (if (call $control (i32.const 6) (i32.const 1) (i32.const 0) (i32.const 48)) (then unreachable))
            (if (i32.load (i32.const 48)) (then unreachable))))"#,
    );
    assert_eq!(environment.run_script_capture("/first").0.exit_status, 0);
}

#[test]
fn killed_replacement_releases_image_metadata_and_guest_memory() {
    let mut environment = Environment::new();
    install(&mut environment, "/first", &replacement("/second"));
    install(
        &mut environment,
        "/second",
        r#"(module
        (import "wasi_snapshot_preview1" "poll_oneoff" (func $poll (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (func (export "_start")
            (i32.store (i32.const 16) (i32.const 1))
            (i64.store (i32.const 24) (i64.const 60000000000))
            (drop (call $poll (i32.const 0) (i32.const 128) (i32.const 1) (i32.const 64)))))"#,
    );
    let baseline = environment.resources.memory_mark();
    for _ in 0..6 {
        let (outcome, _, stderr) =
            environment.run_script_capture("/first & pid=$!; sleep 0.001; kill $pid; wait $pid");
        assert_eq!(
            outcome.exit_status,
            143,
            "{}",
            String::from_utf8_lossy(&stderr)
        );
        assert!(environment.resources.memory_mark() < baseline + 4096);
    }
}

#[test]
fn creating_through_dangling_symlink_applies_mode_without_changing_existing_target() {
    let mut environment = Environment::new();
    environment.vfs.symlink("/", "/target", "/link").unwrap();
    install(
        &mut environment,
        "/first",
        r#"(module
        (import "shellsim_posix_v1" "descriptor_open" (func $open (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (data (i32.const 128) "/link\00")
        (func (export "_start")
            (if (i32.ne (call $open (i32.const -2) (i32.const 128) (i32.const 335564800) (i32.const 438)) (i32.const -20)) (then unreachable))
            (if (i32.lt_s (call $open (i32.const -2) (i32.const 128) (i32.const 335548416) (i32.const 438)) (i32.const 0)) (then unreachable))))"#,
    );
    assert_eq!(
        environment
            .run_script_capture("umask 027; /first")
            .0
            .exit_status,
        0
    );
    assert_eq!(
        environment.vfs.metadata("/", "/target", true).unwrap().mode,
        0o640
    );
    assert!(matches!(
        environment.vfs.metadata("/", "/link", false).unwrap().kind,
        shellsim::vfs::NodeKind::Symlink(_)
    ));
    environment
        .vfs
        .write("/", "/target", b"existing data", 0o600)
        .unwrap();
    environment.vfs.chmod("/", "/target", 0o600).unwrap();
    assert_eq!(environment.run_script_capture("/first").0.exit_status, 0);
    assert_eq!(
        environment.vfs.metadata("/", "/target", true).unwrap().mode,
        0o600
    );
    install(
        &mut environment,
        "/truncate",
        r#"(module
        (import "shellsim_posix_v1" "descriptor_open" (func $open (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (data (i32.const 128) "/link\00")
        (func (export "_start")
            (if (i32.lt_s (call $open (i32.const -2) (i32.const 128) (i32.const 335581184) (i32.const 438)) (i32.const 0)) (then unreachable))))"#,
    );
    assert_eq!(environment.run_script_capture("/truncate").0.exit_status, 0);
    assert!(environment.vfs.read("/", "/target").unwrap().is_empty());
    assert!(matches!(
        environment.vfs.metadata("/", "/link", false).unwrap().kind,
        shellsim::vfs::NodeKind::Symlink(_)
    ));
    assert_eq!(
        environment.vfs.metadata("/", "/target", true).unwrap().mode,
        0o600
    );
}
