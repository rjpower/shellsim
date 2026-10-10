// The fixtures exercise the WASI ABI from compiled Wasm, not a host process or filesystem.
use shellsim::{display::KeyEvent, Environment, Limits};
use std::sync::OnceLock;

fn guest_wc() -> &'static [u8] {
    static GUEST: OnceLock<Vec<u8>> = OnceLock::new();
    GUEST.get_or_init(|| {
        let root = env!("CARGO_MANIFEST_DIR");
        let output = std::env::temp_dir().join(format!("shellsim-wc-{}.wasm", std::process::id()));
        let result = std::process::Command::new("rustc")
            .current_dir(root)
            .args([
                "--edition=2021",
                "--target=wasm32-wasip1",
                "-C",
                "opt-level=z",
                "-C",
                "lto=fat",
                "-C",
                "strip=symbols",
                "guest/wc/src/main.rs",
                "-o",
            ])
            .arg(&output)
            .output()
            .expect("run rustc to build the WASI wc fixture");
        assert!(
            result.status.success(),
            "failed to build WASI wc fixture: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let bytes = std::fs::read(&output).expect("read compiled WASI wc fixture");
        std::fs::remove_file(&output).expect("remove temporary WASI wc fixture");
        bytes
    })
}

fn install(environment: &mut Environment, wat_source: &str) {
    let bytes = wat::parse_str(wat_source).unwrap();
    environment.vfs.write("/", "/app", &bytes, 0o755).unwrap();
}

#[test]
fn posix_dup_keeps_stream_alive_and_preserves_independent_flags() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"(module
        (import "shellsim_posix_v1" "descriptor_control" (func $control (param i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_close" (func $close (param i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (data (i32.const 128) "alias\n")
        (func (export "_start") (local $alias i32)
            (if (call $control (i32.const 1) (i32.const 4) (i32.const 5) (i32.const 32)) (then unreachable))
            (local.set $alias (i32.load (i32.const 32)))
            (if (call $control (local.get $alias) (i32.const 1) (i32.const 0) (i32.const 32)) (then unreachable))
            (if (i32.ne (i32.load (i32.const 32)) (i32.const 1)) (then unreachable))
            (if (call $control (i32.const 1) (i32.const 1) (i32.const 0) (i32.const 32)) (then unreachable))
            (if (i32.load (i32.const 32)) (then unreachable))
            (if (i32.ne (call $control (i32.const 999) (i32.const 3) (i32.const 0) (i32.const 32)) (i32.const 8)) (then unreachable))
            (if (i32.ne (call $control (local.get $alias) (i32.const 5) (i32.const 3) (i32.const 32)) (i32.const 28)) (then unreachable))
            (if (call $close (i32.const 1)) (then unreachable))
            (i32.store (i32.const 0) (i32.const 128))
            (i32.store (i32.const 4) (i32.const 6))
            (if (call $write (local.get $alias) (i32.const 0) (i32.const 1) (i32.const 8)) (then unreachable))))"#,
    );
    assert_eq!(
        run(&mut environment, "/app"),
        (0, b"alias\n".to_vec(), Vec::new())
    );
}

#[test]
fn posix_duped_file_shares_cursor_and_null_device_keeps_access_rights() {
    let mut environment = Environment::new();
    environment.vfs.write("/", "/value", b"ab", 0o644).unwrap();
    install(
        &mut environment,
        r#"(module
        (import "shellsim_posix_v1" "descriptor_control" (func $control (param i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_filestat_get" (func $stat (param i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_close" (func $close (param i32) (result i32)))
        (memory (export "memory") 1)
        (data (i32.const 256) "valuedev/null")
        (func (export "_start") (local $fd i32) (local $alias i32)
            (if (call $open (i32.const 4) (i32.const 0) (i32.const 256) (i32.const 5) (i32.const 0) (i64.const 2) (i64.const 0) (i32.const 0) (i32.const 32)) (then unreachable))
            (local.set $fd (i32.load (i32.const 32)))
            (if (call $control (local.get $fd) (i32.const 3) (i32.const 5) (i32.const 32)) (then unreachable))
            (local.set $alias (i32.load (i32.const 32)))
            (i32.store (i32.const 0) (i32.const 128))
            (i32.store (i32.const 4) (i32.const 1))
            (if (call $read (local.get $fd) (i32.const 0) (i32.const 1) (i32.const 8)) (then unreachable))
            (if (i32.ne (i32.load8_u (i32.const 128)) (i32.const 97)) (then unreachable))
            (if (call $close (local.get $fd)) (then unreachable))
            (if (call $read (local.get $alias) (i32.const 0) (i32.const 1) (i32.const 8)) (then unreachable))
            (if (i32.ne (i32.load8_u (i32.const 128)) (i32.const 98)) (then unreachable))
            (if (call $open (i32.const 4) (i32.const 0) (i32.const 261) (i32.const 8) (i32.const 0) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 32)) (then unreachable))
            (local.set $fd (i32.load (i32.const 32)))
            (if (call $stat (local.get $fd) (i32.const 64)) (then unreachable))
            (if (call $write (local.get $fd) (i32.const 0) (i32.const 1) (i32.const 8)) (then unreachable))
            (if (i32.ne (call $read (local.get $fd) (i32.const 0) (i32.const 1) (i32.const 8)) (i32.const 8)) (then unreachable))))"#,
    );
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
}

#[test]
fn inherited_regular_stdin_has_real_stat_and_seek_state() {
    let mut environment = Environment::new();
    environment.vfs.write("/", "/input", b"abc", 0o644).unwrap();
    install(
        &mut environment,
        r#"(module
        (import "wasi_snapshot_preview1" "fd_filestat_get" (func $stat (param i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_seek" (func $seek (param i32 i64 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_tell" (func $tell (param i32 i32) (result i32)))
        (memory (export "memory") 1)
        (func (export "_start")
            (if (call $stat (i32.const 0) (i32.const 64)) (then unreachable))
            (if (i32.ne (i32.load8_u (i32.const 80)) (i32.const 4)) (then unreachable))
            (if (i64.ne (i64.load (i32.const 96)) (i64.const 3)) (then unreachable))
            (if (call $seek (i32.const 0) (i64.const 2) (i32.const 0) (i32.const 8)) (then unreachable))
            (if (call $tell (i32.const 0) (i32.const 8)) (then unreachable))
            (if (i64.ne (i64.load (i32.const 8)) (i64.const 2)) (then unreachable))))"#,
    );
    assert_eq!(
        run(&mut environment, "/app < /input"),
        (0, Vec::new(), Vec::new())
    );
}

#[test]
fn positioned_reads_preserve_shared_cursor_and_read_unlinked_files() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/value", b"abcdef", 0o644)
        .unwrap();
    install(
        &mut environment,
        r#"(module
        (import "shellsim_posix_v1" "descriptor_control" (func $dup (param i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "path_unlink_file" (func $unlink (param i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_pread" (func $pread (param i32 i32 i32 i64 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_seek" (func $seek (param i32 i64 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_tell" (func $tell (param i32 i32) (result i32)))
        (memory (export "memory") 1)
        (data (i32.const 128) "value")
        (func (export "_start") (local $fd i32) (local $alias i32) (local $writer i32)
            (if (call $open (i32.const 4) (i32.const 0) (i32.const 128) (i32.const 5) (i32.const 0) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 32)) (then unreachable))
            (local.set $writer (i32.load (i32.const 32)))
            (if (call $open (i32.const 4) (i32.const 0) (i32.const 128) (i32.const 5) (i32.const 0) (i64.const 2) (i64.const 0) (i32.const 0) (i32.const 32)) (then unreachable))
            (local.set $fd (i32.load (i32.const 32)))
            (if (call $dup (local.get $fd) (i32.const 3) (i32.const 5) (i32.const 32)) (then unreachable))
            (local.set $alias (i32.load (i32.const 32)))
            (if (call $seek (local.get $fd) (i64.const 1) (i32.const 0) (i32.const 40)) (then unreachable))
            (i32.store (i32.const 0) (i32.const 256))
            (i32.store (i32.const 4) (i32.const 2))
            (i32.store (i32.const 8) (i32.const 258))
            (i32.store (i32.const 12) (i32.const 4))
            (if (call $pread (local.get $fd) (i32.const 0) (i32.const 2) (i64.const 2) (i32.const 32)) (then unreachable))
            (if (i32.ne (i32.load (i32.const 32)) (i32.const 4)) (then unreachable))
            (if (i32.ne (i32.load (i32.const 256)) (i32.const 0x66656463)) (then unreachable))
            (if (call $tell (local.get $alias) (i32.const 40)) (then unreachable))
            (if (i64.ne (i64.load (i32.const 40)) (i64.const 1)) (then unreachable))
            (i32.store (i32.const 4) (i32.const 1))
            (if (call $read (local.get $alias) (i32.const 0) (i32.const 1) (i32.const 32)) (then unreachable))
            (if (i32.ne (i32.load8_u (i32.const 256)) (i32.const 98)) (then unreachable))
            (if (call $unlink (i32.const 4) (i32.const 128) (i32.const 5)) (then unreachable))
            (i32.store (i32.const 4) (i32.const 4))
            (if (call $pread (local.get $fd) (i32.const 0) (i32.const 1) (i64.const 4) (i32.const 32)) (then unreachable))
            (if (i32.ne (i32.load (i32.const 32)) (i32.const 2)) (then unreachable))
            (if (i32.ne (i32.load16_u (i32.const 256)) (i32.const 0x6665)) (then unreachable))
            (if (call $tell (local.get $alias) (i32.const 40)) (then unreachable))
            (if (i64.ne (i64.load (i32.const 40)) (i64.const 2)) (then unreachable))
            (if (call $pread (local.get $fd) (i32.const 0) (i32.const 1) (i64.const 99) (i32.const 32)) (then unreachable))
            (if (i32.load (i32.const 32)) (then unreachable))
            (if (i32.ne (call $pread (local.get $fd) (i32.const 0) (i32.const 1) (i64.const 0) (i32.const 65534)) (i32.const 21)) (then unreachable))
            (if (i32.ne (call $pread (local.get $fd) (i32.const 0) (i32.const 1025) (i64.const 0) (i32.const 32)) (i32.const 28)) (then unreachable))
            (i32.store (i32.const 0) (i32.const 65535))
            (if (i32.ne (call $pread (local.get $fd) (i32.const 0) (i32.const 1) (i64.const 0) (i32.const 32)) (i32.const 21)) (then unreachable))
            (i32.store (i32.const 0) (i32.const 256))
            (if (i32.ne (call $pread (i32.const 999) (i32.const 0) (i32.const 1) (i64.const 0) (i32.const 32)) (i32.const 8)) (then unreachable))
            (if (i32.ne (call $pread (local.get $writer) (i32.const 0) (i32.const 1) (i64.const 0) (i32.const 32)) (i32.const 8)) (then unreachable))
            (if (i32.ne (call $pread (i32.const 0) (i32.const 0) (i32.const 1) (i64.const 0) (i32.const 32)) (i32.const 70)) (then unreachable))))"#,
    );
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
}

#[test]
fn positioned_reads_charge_requested_work_before_copying() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000,
        ..Limits::default()
    });
    environment.vfs.write("/", "/value", b"abc", 0o644).unwrap();
    install(
        &mut environment,
        r#"(module
        (import "wasi_snapshot_preview1" "fd_pread" (func $pread (param i32 i32 i32 i64 i32) (result i32)))
        (memory (export "memory") 2)
        (func (export "_start") (local $index i32)
            (loop $vectors
                (i32.store (local.get $index) (i32.const 4096))
                (i32.store offset=4 (local.get $index) (i32.const 4096))
                (local.set $index (i32.add (local.get $index) (i32.const 8)))
                (br_if $vectors (i32.lt_u (local.get $index) (i32.const 2048))))
            (drop (call $pread (i32.const 0) (i32.const 0) (i32.const 256) (i64.const 0) (i32.const 2048)))))"#,
    );
    let (outcome, _, _) = environment.run_script_capture("/app < /value");
    assert_eq!(
        outcome.stop_reason,
        Some(shellsim::StopReason::CpuExhausted)
    );
}

#[test]
fn posix_cwd_and_umask_use_only_virtual_process_state() {
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/work/dir").unwrap();
    install(
        &mut environment,
        r#"(module
        (import "shellsim_posix_v1" "cwd_get" (func $get (param i32 i32) (result i32)))
        (import "shellsim_posix_v1" "cwd_set" (func $set (param i32 i32 i32 i32) (result i32)))
        (import "shellsim_posix_v1" "umask" (func $mask (param i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (data (i32.const 256) "dirmissing")
        (func (export "_start")
            (if (i32.ne (call $mask (i32.const 63)) (i32.const 18)) (then unreachable))
            (if (i32.ne (call $mask (i32.const 18)) (i32.const 63)) (then unreachable))
            (if (i32.ne (call $set (i32.const 256) (i32.const 3) (i32.const 128) (i32.const 1)) (i32.const 68)) (then unreachable))
            (if (call $get (i32.const 128) (i32.const 128)) (then unreachable))
            (if (i32.ne (i32.load8_u (i32.const 133)) (i32.const 0)) (then unreachable))
            (if (i32.ne (call $set (i32.const 259) (i32.const 7) (i32.const 128) (i32.const 128)) (i32.const 44)) (then unreachable))
            (if (call $set (i32.const 256) (i32.const 3) (i32.const 128) (i32.const 128)) (then unreachable))
            (i32.store (i32.const 0) (i32.const 128))
            (i32.store (i32.const 4) (i32.const 9))
            (if (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)) (then unreachable))))"#,
    );
    assert_eq!(
        run(&mut environment, "cd /work; /app"),
        (0, b"/work/dir".to_vec(), Vec::new())
    );
}

fn run(environment: &mut Environment, source: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (outcome.exit_status, stdout, stderr)
}

#[test]
fn repeated_exec_reuses_code_but_not_guest_state_or_replaced_file() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"(module
            (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
            (memory (export "memory") 1)
            (global $count (mut i32) (i32.const 0))
            (data (i32.const 32) "0\n")
            (func (export "_start")
                (global.set $count (i32.add (global.get $count) (i32.const 1)))
                (i32.store8 (i32.const 32) (i32.add (i32.const 48) (global.get $count)))
                (i32.store (i32.const 0) (i32.const 32))
                (i32.store (i32.const 4) (i32.const 2))
                (drop (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))))"#,
    );
    assert_eq!(
        run(&mut environment, "/app"),
        (0, b"1\n".to_vec(), Vec::new())
    );
    assert_eq!(
        run(&mut environment, "/app"),
        (0, b"1\n".to_vec(), Vec::new())
    );

    install(
        &mut environment,
        r#"(module
            (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
            (memory (export "memory") 1)
            (data (i32.const 32) "B\n")
            (func (export "_start")
                (i32.store (i32.const 0) (i32.const 32))
                (i32.store (i32.const 4) (i32.const 2))
                (drop (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))))"#,
    );
    assert_eq!(
        run(&mut environment, "/app"),
        (0, b"B\n".to_vec(), Vec::new())
    );
}

#[test]
fn wasi_random_is_process_owned_and_faults_do_not_advance_it() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"(module
            (import "wasi_snapshot_preview1" "random_get" (func $random (param i32 i32) (result i32)))
            (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
            (memory (export "memory") 1)
            (func (export "_start")
                (if (i32.ne (call $random (i32.const 65535) (i32.const 4)) (i32.const 21)) (then unreachable))
                (if (i32.ne (call $random (i32.const 32) (i32.const 4)) (i32.const 0)) (then unreachable))
                (i32.store (i32.const 0) (i32.const 32))
                (i32.store (i32.const 4) (i32.const 4))
                (drop (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))))"#,
    );
    let mut snapshot = environment.clone();
    let (status, first, stderr) = run(&mut environment, "/app");
    assert_eq!((status, first.len(), stderr), (0, 4, Vec::new()));
    // Each run is a new process with its own entropy stream; a snapshot replays the same one.
    let (status, second, _) = run(&mut environment, "/app");
    assert_eq!(status, 0);
    assert_ne!(second, first);
    assert_eq!(run(&mut snapshot, "/app"), (0, first, Vec::new()));
}

#[test]
fn wasi_random_obeys_virtual_cpu_limit() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 20_000,
        ..Limits::default()
    });
    install(
        &mut environment,
        r#"(module
            (import "wasi_snapshot_preview1" "random_get" (func $random (param i32 i32) (result i32)))
            (memory (export "memory") 2)
            (func (export "_start")
                (drop (call $random (i32.const 0) (i32.const 100000)))))"#,
    );
    assert_eq!(run(&mut environment, "/app").0, 137);
}

#[test]
fn virtual_display_presents_frame_and_consumes_injected_key() {
    const GUEST: &str = r#"(module
        (import "shellsim" "display_open" (func $open (param i32 i32 i32) (result i32)))
        (import "shellsim" "display_present" (func $present (param i32 i32 i32 i32) (result i32)))
        (import "shellsim" "input_poll_key" (func $poll (param i32 i32) (result i32)))
        (import "shellsim" "display_close" (func $close (param i32) (result i32)))
        (memory (export "memory") 1)
        (func (export "_start") (local $handle i32)
            (local.set $handle (call $open (i32.const 2) (i32.const 1) (i32.const 1)))
            (if (i32.le_s (local.get $handle) (i32.const 0)) (then unreachable))
            (if (i32.ne (call $present (local.get $handle) (i32.const 32) (i32.const 8) (i32.const 8)) (i32.const 0)) (then unreachable))
            (if (i32.eq (call $poll (local.get $handle) (i32.const 16)) (i32.const 0))
                (then
                    (i32.store8 (i32.const 32) (i32.const 255))
                    (if (i32.ne (call $present (local.get $handle) (i32.const 32) (i32.const 8) (i32.const 8)) (i32.const 0)) (then unreachable))))
            (if (i32.ne (call $close (local.get $handle)) (i32.const 0)) (then unreachable))))"#;

    let mut idle = Environment::new();
    install(&mut idle, GUEST);
    assert_eq!(run(&mut idle, "/app"), (0, Vec::new(), Vec::new()));
    assert_eq!(idle.display.frame().unwrap().pixels, vec![0; 8]);

    let mut active = Environment::new();
    active
        .inject_key(KeyEvent {
            code: 32,
            pressed: true,
        })
        .unwrap();
    install(&mut active, GUEST);
    assert_eq!(run(&mut active, "/app"), (0, Vec::new(), Vec::new()));
    let frame = active.display.frame().unwrap();
    assert_eq!((frame.width, frame.height), (2, 1));
    assert_eq!(frame.pixels, [255, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn virtual_display_rejects_bad_geometry_and_unmetered_allocation() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"(module
            (import "shellsim" "display_open" (func $open (param i32 i32 i32) (result i32)))
            (memory (export "memory") 1)
            (func (export "_start")
                (if (i32.ne (call $open (i32.const 0) (i32.const 1) (i32.const 1)) (i32.const -28)) (then unreachable))
                (if (i32.ne (call $open (i32.const 1) (i32.const 1) (i32.const 2)) (i32.const -28)) (then unreachable))))"#,
    );
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    assert!(environment.display.frame().is_none());

    let mut constrained = Environment::with_limits(Limits {
        memory: 16 * 1024 * 1024,
        ..Limits::default()
    });
    install(
        &mut constrained,
        r#"(module
            (import "shellsim" "display_open" (func $open (param i32 i32 i32) (result i32)))
            (memory (export "memory") 1)
            (func (export "_start")
                (drop (call $open (i32.const 1024) (i32.const 1024) (i32.const 1)))))"#,
    );
    assert_eq!(run(&mut constrained, "/app").0, 137);
    assert!(constrained.display.frame().is_none());
}

#[test]
fn virtual_display_rejects_bad_pointers_without_losing_input() {
    let mut environment = Environment::new();
    environment
        .inject_key(KeyEvent {
            code: 27,
            pressed: true,
        })
        .unwrap();
    install(
        &mut environment,
        r#"(module
            (import "shellsim" "display_open" (func $open (param i32 i32 i32) (result i32)))
            (import "shellsim" "display_present" (func $present (param i32 i32 i32 i32) (result i32)))
            (import "shellsim" "input_poll_key" (func $poll (param i32 i32) (result i32)))
            (memory (export "memory") 1)
            (func (export "_start") (local $handle i32)
                (local.set $handle (call $open (i32.const 1) (i32.const 1) (i32.const 1)))
                (if (i32.ne (call $poll (local.get $handle) (i32.const 65532)) (i32.const 21)) (then unreachable))
                (if (i32.ne (call $poll (local.get $handle) (i32.const 16)) (i32.const 0)) (then unreachable))
                (if (i32.ne (i32.load (i32.const 16)) (i32.const 27)) (then unreachable))
                (if (i32.ne (call $poll (local.get $handle) (i32.const 16)) (i32.const 6)) (then unreachable))
                (if (i32.ne (call $present (i32.const 99) (i32.const 32) (i32.const 4) (i32.const 4)) (i32.const 8)) (then unreachable))
                (if (i32.ne (call $present (local.get $handle) (i32.const 65534) (i32.const 4) (i32.const 4)) (i32.const 21)) (then unreachable))))"#,
    );
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    assert_eq!(environment.display.frame().unwrap().pixels, [0, 0, 0, 0]);
}

#[test]
fn exception_handling_module_runs_without_extra_host_capabilities() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"(module
            (tag $error (param i32))
            (func (export "_start")
                (block $caught (result i32)
                    (try_table (catch $error $caught)
                        (i32.const 7)
                        (throw $error))
                    (i32.const 0))
                (drop)))"#,
    );
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
}

#[test]
fn wasi_stdio_close_invalidates_the_guest_descriptor() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"(module
            (import "wasi_snapshot_preview1" "fd_close" (func $close (param i32) (result i32)))
            (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
            (func (export "_start")
                (if (i32.ne (call $close (i32.const 1)) (i32.const 0)) (then unreachable))
                (if (i32.ne (call $close (i32.const 1)) (i32.const 8)) (then unreachable))
                (if (i32.ne (call $write (i32.const 1) (i32.const 0) (i32.const 0) (i32.const 0)) (i32.const 8)) (then unreachable))))"#,
    );
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
}

#[test]
fn compiled_wasi_wc_matches_native_wc_on_virtual_streams_and_files() {
    let guest = guest_wc();
    let cases = [
        "printf 'one two\\n' | COMMAND",
        "printf 'one two\\n' > /work/input; COMMAND -lc /work/input",
        "printf 'é one\\n' > /work/a; printf 'two\\n' > /work/b; COMMAND -m /work/a /work/b",
    ];
    for case in cases {
        let mut environment = Environment::new();
        environment
            .vfs
            .write("/", "/guest-wc", guest, 0o755)
            .unwrap();
        let native = run(&mut environment, &case.replace("COMMAND", "wc"));
        let guest = run(&mut environment, &case.replace("COMMAND", "/guest-wc"));
        assert_eq!(guest, native, "{case}");
    }
}

#[test]
fn compiled_wasi_wc_counts_a_pipe_larger_than_pipe_capacity() {
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/usr/bin").unwrap();
    environment
        .vfs
        .write("/", "/usr/bin/wc", guest_wc(), 0o755)
        .unwrap();
    environment
        .vfs
        .write("/", "/work/input", &vec![b'x'; 131_072], 0o644)
        .unwrap();

    let script = "cat /work/input | wc -c | cat";
    assert_eq!(
        run(&mut environment, script),
        (0, b"131072\n".to_vec(), Vec::new())
    );
}

#[test]
fn compiled_wasi_wc_rejects_invalid_option_without_host_execution() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/guest-wc", guest_wc(), 0o755)
        .unwrap();
    let (status, stdout, stderr) = run(&mut environment, "/guest-wc -Z");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("unimplemented option"));
}

#[test]
fn compiled_wasi_wc_reports_missing_virtual_file() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/guest-wc", guest_wc(), 0o755)
        .unwrap();
    let (status, stdout, stderr) = run(&mut environment, "/guest-wc /missing");
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("/missing"));
}

#[test]
fn compiled_wasi_wc_obeys_cpu_limit() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000,
        ..Limits::default()
    });
    environment
        .vfs
        .write("/", "/guest-wc", guest_wc(), 0o755)
        .unwrap();
    let (status, stdout, _) = run(&mut environment, "/guest-wc");
    assert_eq!(status, 137);
    assert!(stdout.is_empty());
}

#[test]
fn explicit_virtual_executable_overrides_standard_native_alias() {
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/usr/bin").unwrap();
    let guest = wat::parse_str(
        r#"
        (module
          (import "wasi_snapshot_preview1" "fd_write"
            (func $write (param i32 i32 i32 i32) (result i32)))
          (memory (export "memory") 1)
          (data (i32.const 32) "guest\n")
          (func (export "_start")
            (i32.store (i32.const 0) (i32.const 32))
            (i32.store (i32.const 4) (i32.const 6))
            (drop (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))))
    "#,
    )
    .unwrap();
    environment
        .vfs
        .write("/", "/usr/bin/wc", &guest, 0o755)
        .unwrap();
    assert_eq!(run(&mut environment, "/usr/bin/wc").1, b"guest\n");
    assert_eq!(run(&mut environment, "printf x | wc -c").1, b"guest\n");
    environment.vfs.chmod("/", "/usr/bin/wc", 0o644).unwrap();
    let (status, stdout, stderr) = run(&mut environment, "/usr/bin/wc");
    assert_eq!(status, 126);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("permission denied"));
}

#[test]
fn native_find_and_compiled_wasi_wc_coexist() {
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/usr/bin").unwrap();
    environment
        .vfs
        .write("/", "/usr/bin/wc", guest_wc(), 0o755)
        .unwrap();
    environment
        .vfs
        .write("/", "/work/one", b"one\n", 0o644)
        .unwrap();
    let native_find = run(&mut environment, "find /work -type f");
    assert_eq!(
        run(&mut environment, "/usr/bin/find /work -type f"),
        native_find
    );
    let native_wc = run(&mut environment, "find /work -type f | wc -l");
    assert_eq!(
        run(&mut environment, "find /work -type f | /usr/bin/wc -l"),
        native_wc
    );
}

#[test]
fn wasi_stdout_and_exit_code_use_virtual_command_streams() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"
        (module
          (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
          (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
          (memory (export "memory") 1)
          (data (i32.const 32) "hello\n")
          (func (export "_start")
            (i32.store (i32.const 0) (i32.const 32))
            (i32.store (i32.const 4) (i32.const 6))
            (drop (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))
            (call $exit (i32.const 7))))
    "#,
    );
    assert_eq!(
        run(&mut environment, "/app"),
        (7, b"hello\n".to_vec(), Vec::new())
    );
}

#[test]
fn wasi_reads_piped_stdin() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"
        (module
          (import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))
          (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
          (memory (export "memory") 1)
          (func (export "_start")
            (i32.store (i32.const 0) (i32.const 64))
            (i32.store (i32.const 4) (i32.const 16))
            (drop (call $read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 8)))
            (i32.store (i32.const 4) (i32.load (i32.const 8)))
            (drop (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 12)))))
    "#,
    );
    assert_eq!(
        run(&mut environment, "printf hello | /app"),
        (0, b"hello".to_vec(), Vec::new())
    );
}

#[test]
fn wasi_preopen_writes_only_to_virtual_filesystem() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"
        (module
          (import "wasi_snapshot_preview1" "path_open"
            (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
          (import "wasi_snapshot_preview1" "fd_write"
            (func $write (param i32 i32 i32 i32) (result i32)))
          (memory (export "memory") 1)
          (data (i32.const 64) "result.txt")
          (data (i32.const 96) "artifact")
          (func (export "_start")
            (if (i32.ne
                  (call $open (i32.const 3) (i32.const 0) (i32.const 64) (i32.const 10)
                    (i32.const 1) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 0))
                  (i32.const 0))
              (then unreachable))
            (i32.store (i32.const 8) (i32.const 96))
            (i32.store (i32.const 12) (i32.const 8))
            (if (i32.ne (call $write (i32.load (i32.const 0)) (i32.const 8)
                                (i32.const 1) (i32.const 16)) (i32.const 0))
              (then unreachable))))
    "#,
    );
    assert_eq!(
        run(&mut environment, "mkdir -p /work; cd /work; /app"),
        (0, Vec::new(), Vec::new())
    );
    assert_eq!(
        environment.vfs.read("/", "/work/result.txt").unwrap(),
        b"artifact"
    );
}

#[test]
fn wasi_rejects_unsupported_open_flags_explicitly() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"
        (module
          (import "wasi_snapshot_preview1" "path_open"
            (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
          (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
          (memory (export "memory") 1)
          (data (i32.const 64) "file")
          (func (export "_start")
            (call $exit
              (call $open (i32.const 3) (i32.const 0) (i32.const 64) (i32.const 4)
                (i32.const 0) (i64.const 2) (i64.const 0) (i32.const 2) (i32.const 0)))))
    "#,
    );
    assert_eq!(run(&mut environment, "/app").0, 28);
}

#[test]
fn unsupported_import_fails_explicitly_without_host_fallback() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"
        (module
          (import "wasi_snapshot_preview1" "sock_accept" (func $accept (param i32 i32 i32) (result i32)))
          (memory (export "memory") 1)
          (func (export "_start") (drop (call $accept (i32.const 0) (i32.const 0) (i32.const 0)))))
    "#,
    );
    let (status, stdout, stderr) = run(&mut environment, "/app");
    assert_eq!(status, 126);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("sock_accept"));
}

#[test]
fn unknown_import_namespace_is_rejected_even_when_unused() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"(module
            (import "host" "run" (func))
            (func (export "_start")))"#,
    );
    let (status, stdout, stderr) = run(&mut environment, "/app");
    assert_eq!(status, 126);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("host.run"));
}

#[test]
fn infinite_wasm_loop_is_metered() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000,
        ..Limits::default()
    });
    install(
        &mut environment,
        r#"(module (func (export "_start") (loop (br 0))))"#,
    );
    let (status, _, stderr) = run(&mut environment, "/app");
    assert_ne!(status, 0);
    assert!(String::from_utf8_lossy(&stderr).contains("wasm execution failed"));
}

#[test]
fn static_module_image_bound_and_compilation_cost_are_enforced() {
    // A large ignored custom section tests the image boundary without a numerical fixture.
    let mut bytes = wat::parse_str("(module (func (export \"_start\")))").unwrap();
    let payload_len = 17 * 1024 * 1024;
    bytes.push(0);
    let mut length = payload_len + 1;
    loop {
        let byte = (length & 127) as u8;
        length >>= 7;
        bytes.push(byte | if length != 0 { 128 } else { 0 });
        if length == 0 {
            break;
        }
    }
    bytes.push(0); // Empty custom-section name.
    bytes.resize(bytes.len() + payload_len, 0);
    let mut environment = Environment::with_limits(Limits {
        cpu: 1_000_000_000,
        disk: 256 * 1024 * 1024,
        memory: 2 * 1024 * 1024 * 1024,
        ..Limits::default()
    });
    environment.vfs.write("/", "/app", &bytes, 0o755).unwrap();
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));

    let mut constrained = Environment::with_limits(Limits {
        cpu: 100_000_000,
        ..Limits::default()
    });
    constrained.vfs.write("/", "/app", &bytes, 0o755).unwrap();
    let (status, _, stderr) = run(&mut constrained, "/app");
    assert_eq!(status, 137);
    assert!(String::from_utf8_lossy(&stderr).contains("wasm compilation budget exhausted"));

    let mut constrained_memory = Environment::with_limits(Limits {
        cpu: 1_000_000_000,
        ..Limits::default()
    });
    constrained_memory
        .vfs
        .write("/", "/app", &bytes, 0o755)
        .unwrap();
    let (status, _, stderr) = run(&mut constrained_memory, "/app");
    assert_eq!(status, 137);
    assert!(String::from_utf8_lossy(&stderr).contains("wasm compilation memory budget exhausted"));

    bytes.resize(128 * 1024 * 1024 + 1, 0);
    environment.vfs.write("/", "/app", &bytes, 0o755).unwrap();
    let (status, _, stderr) = run(&mut environment, "/app");
    assert_eq!(status, 126);
    assert!(String::from_utf8_lossy(&stderr).contains("wasm module exceeds size limit"));
}

#[test]
fn large_static_guest_tables_have_a_bounded_reservation() {
    let mut environment = Environment::with_limits(Limits {
        memory: 128 * 1024 * 1024,
        ..Limits::default()
    });
    install(&mut environment,
        "(module (memory (export \"memory\") 320) (table 10771 funcref) (func (export \"_start\")))");
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    assert!(environment.resources.outcome(0, 0, 0).usage.memory_peak >= 68 * 1024 * 1024);
    for source in [
        "(module (memory (export \"memory\") 320) (table 16385 funcref) (func (export \"_start\")))",
        "(module (memory (export \"memory\") 1) (table 10771 funcref) (func (export \"_start\")))",
    ] {
        install(&mut environment, source);
        let (status, _, stderr) = run(&mut environment, "/app");
        assert_eq!(status, 126);
        assert!(String::from_utf8_lossy(&stderr).contains("table minimum size"));
    }
}

#[test]
fn oversized_linear_memory_is_rejected() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"(module (memory (export "memory") 1025) (func (export "_start")))"#,
    );
    let (status, stdout, stderr) = run(&mut environment, "/app");
    assert_eq!(status, 126);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("wasm execution failed"));
}

#[test]
fn wasi_arguments_and_environment_are_process_local() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"
        (module
          (import "wasi_snapshot_preview1" "args_sizes_get" (func $args (param i32 i32) (result i32)))
          (import "wasi_snapshot_preview1" "environ_sizes_get" (func $env (param i32 i32) (result i32)))
          (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
          (memory (export "memory") 1)
          (func (export "_start")
            (drop (call $args (i32.const 32) (i32.const 36)))
            (drop (call $env (i32.const 40) (i32.const 44)))
            (i32.store8 (i32.const 64) (i32.add (i32.load (i32.const 32)) (i32.const 48)))
            (i32.store8 (i32.const 65) (i32.add (i32.load (i32.const 40)) (i32.const 48)))
            (i32.store (i32.const 0) (i32.const 64))
            (i32.store (i32.const 4) (i32.const 2))
            (drop (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))))
    "#,
    );
    let (status, stdout, stderr) = run(&mut environment, "export WASI_TEST=ok; /app one two");
    assert_eq!(status, 0);
    assert!(stderr.is_empty());
    assert_eq!(stdout[0], b'3');
    assert!(stdout[1] > b'0');
}

#[test]
fn wasi_clock_uses_virtual_time() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"
        (module
          (import "wasi_snapshot_preview1" "clock_time_get" (func $clock (param i32 i64 i32) (result i32)))
          (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
          (memory (export "memory") 1)
          (func (export "_start")
            (drop (call $clock (i32.const 1) (i64.const 1) (i32.const 32)))
            (i32.store8 (i32.const 64) (i32.add (i32.wrap_i64 (i64.load (i32.const 32))) (i32.const 48)))
            (i32.store (i32.const 0) (i32.const 64))
            (i32.store (i32.const 4) (i32.const 1))
            (drop (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))))
    "#,
    );
    assert_eq!(
        run(&mut environment, "/app"),
        (0, b"0".to_vec(), Vec::new())
    );
    assert_eq!(
        run(&mut environment, "sleep 0.000000001; /app"),
        (0, b"1".to_vec(), Vec::new())
    );
}

#[test]
fn invalid_iovec_pointer_returns_fault_without_host_panic() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"
        (module
          (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
          (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
          (memory (export "memory") 1)
          (func (export "_start")
            (call $exit (call $write (i32.const 1) (i32.const -2) (i32.const 1) (i32.const 8)))))
    "#,
    );
    assert_eq!(run(&mut environment, "/app"), (21, Vec::new(), Vec::new()));
}

/// Copies stdin to stdout in 4 KiB reads, retrying short writes; exits 2 or 3 on an errno.
const WAT_CAT: &str = r#"(module
    (import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))
    (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
    (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
    (memory (export "memory") 1)
    (func (export "_start")
        (loop $copy
            (i32.store (i32.const 0) (i32.const 1024))
            (i32.store (i32.const 4) (i32.const 4096))
            (if (call $read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 8))
                (then (call $exit (i32.const 2))))
            (if (i32.eqz (i32.load (i32.const 8))) (then return))
            (i32.store (i32.const 4) (i32.load (i32.const 8)))
            (loop $flush
                (if (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 12))
                    (then (call $exit (i32.const 3))))
                (i32.store (i32.const 0) (i32.add (i32.load (i32.const 0)) (i32.load (i32.const 12))))
                (i32.store (i32.const 4) (i32.sub (i32.load (i32.const 4)) (i32.load (i32.const 12))))
                (br_if $flush (i32.load (i32.const 4))))
            (br $copy))))"#;

/// Writes "y" forever and exits 3 if a write ever reports an errno.
const WAT_YES: &str = r#"(module
    (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
    (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
    (memory (export "memory") 1)
    (data (i32.const 1024) "yyyyyyyy")
    (func (export "_start")
        (i32.store (i32.const 0) (i32.const 1024))
        (i32.store (i32.const 4) (i32.const 8))
        (loop $forever
            (if (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 12))
                (then (call $exit (i32.const 3))))
            (br $forever))))"#;

/// Computes forever without any host call.
const WAT_SPIN: &str =
    r#"(module (memory (export "memory") 1) (func (export "_start") (loop (br 0))))"#;

fn install_at(environment: &mut Environment, path: &str, wat_source: &str) {
    let bytes = wat::parse_str(wat_source).unwrap();
    environment.vfs.write("/", path, &bytes, 0o755).unwrap();
}

#[test]
fn wasi_guest_waits_for_a_slow_pipe_writer() {
    let mut environment = Environment::new();
    install(&mut environment, WAT_CAT);
    assert_eq!(
        run(&mut environment, "{ printf a; sleep 1; printf b; } | /app"),
        (0, b"ab".to_vec(), Vec::new())
    );
}

#[test]
fn wasi_guests_stream_through_a_pipeline_larger_than_pipe_capacity() {
    let mut environment = Environment::new();
    install(&mut environment, WAT_CAT);
    let script = "set -o pipefail; head -c 300000 /dev/zero | /app | /app | wc -c; echo $?";
    assert_eq!(
        run(&mut environment, script),
        (0, b"300000\n0\n".to_vec(), Vec::new())
    );
}

#[test]
fn wasi_guest_sees_eof_when_the_writer_closes() {
    let mut environment = Environment::new();
    install(&mut environment, WAT_CAT);
    assert_eq!(
        run(
            &mut environment,
            "/app < /dev/null; echo $?; true | /app; echo $?"
        ),
        (0, b"0\n0\n".to_vec(), Vec::new())
    );
}

#[test]
fn wasi_writer_ends_with_sigpipe_status_when_the_reader_exits() {
    let mut environment = Environment::new();
    install(&mut environment, WAT_YES);
    assert_eq!(
        run(
            &mut environment,
            "set -o pipefail; /app | head -c 3; echo \" $?\""
        ),
        (0, b"yyy 141\n".to_vec(), Vec::new())
    );
}

#[test]
fn cpu_bound_guest_does_not_starve_the_shell_and_can_be_killed() {
    let mut environment = Environment::new();
    install(&mut environment, WAT_SPIN);
    let memory = environment.resources.memory_mark();
    let mut retained = Vec::new();
    for _ in 0..2 {
        assert_eq!(
            run(
                &mut environment,
                "/app & echo started; kill $!; wait $!; echo $?"
            ),
            (0, b"started\n143\n".to_vec(), Vec::new())
        );
        retained.push(environment.resources.memory_mark());
    }
    // Killing the guest returns its 64 KiB linear-memory reservation; only shell variables
    // such as `$!` stay charged, and running the guest again does not grow them.
    assert!(retained[0] < memory + 4096, "{retained:?}");
    assert_eq!(retained[1], retained[0]);
}

#[test]
fn timeout_stops_a_guest_blocked_on_input() {
    let mut environment = Environment::new();
    install(&mut environment, WAT_CAT);
    assert_eq!(
        run(&mut environment, "sleep 5 | timeout 1 /app; echo $?"),
        (0, b"124\n".to_vec(), Vec::new())
    );
}

#[test]
fn guests_in_one_pipeline_share_the_machine_cpu_budget() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 2_000_000,
        ..Limits::default()
    });
    install_at(&mut environment, "/spin", WAT_SPIN);
    let (status, _, _) = run(&mut environment, "/spin | /spin");
    assert_eq!(status, 137);
    // Each guest starts with fuel for the whole remaining budget; charging at fuel yields stops
    // the pair near the machine limit instead of letting each spend it separately.
    assert!(environment.resources.cpu_used() <= 2_000_000 + 100_000);
}

/// Polls two monotonic clock subscriptions (userdata 7 after `first_ns`, userdata 9 after five
/// seconds), prints "done", and exits with `10 * nevents + first userdata`, or with the errno.
/// `tag` sets the first subscription's type, so 1 requests fd readiness instead of a clock.
fn sleeper(tag: u8, first_ns: u64) -> String {
    format!(
        r#"(module
            (import "wasi_snapshot_preview1" "poll_oneoff" (func $poll (param i32 i32 i32 i32) (result i32)))
            (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
            (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
            (memory (export "memory") 1)
            (data (i32.const 400) "done\n")
            (func (export "_start") (local $errno i32)
                (i64.store (i32.const 0) (i64.const 7))
                (i32.store8 (i32.const 8) (i32.const {tag}))
                (i32.store (i32.const 16) (i32.const 1))
                (i64.store (i32.const 24) (i64.const {first_ns}))
                (i64.store (i32.const 48) (i64.const 9))
                (i32.store (i32.const 64) (i32.const 1))
                (i64.store (i32.const 72) (i64.const 5000000000))
                (local.set $errno (call $poll (i32.const 0) (i32.const 200) (i32.const 2) (i32.const 300)))
                (if (local.get $errno) (then (call $exit (local.get $errno))))
                (i32.store (i32.const 500) (i32.const 400))
                (i32.store (i32.const 504) (i32.const 5))
                (drop (call $write (i32.const 1) (i32.const 500) (i32.const 1) (i32.const 508)))
                (call $exit (i32.add
                    (i32.mul (i32.load (i32.const 300)) (i32.const 10))
                    (i32.load (i32.const 200))))))"#
    )
}

#[test]
fn guest_clock_wait_blocks_on_virtual_time_while_the_shell_runs() {
    let mut environment = Environment::new();
    install(&mut environment, &sleeper(0, 2_000_000_000));
    assert_eq!(
        run(
            &mut environment,
            "/app & sleep 1; echo mid; wait $!; echo $?"
        ),
        (0, b"mid\ndone\n17\n".to_vec(), Vec::new())
    );
    assert_eq!(environment.clock.monotonic_ns(), 2_000_000_000);
}

#[test]
fn timeout_interrupts_a_guest_clock_wait() {
    let mut environment = Environment::new();
    install(&mut environment, &sleeper(0, 10_000_000_000));
    assert_eq!(
        run(&mut environment, "timeout 1 /app; echo $?"),
        (0, b"124\n".to_vec(), Vec::new())
    );
    assert_eq!(environment.clock.monotonic_ns(), 1_000_000_000);
}

#[test]
fn guest_descriptor_poll_reports_access_errors_in_events() {
    let mut environment = Environment::new();
    let fixture = sleeper(1, 0).replace("(i32.store (i32.const 500)",
        "(if (i32.ne (i32.load16_u (i32.const 208)) (i32.const 8)) (then unreachable)) (i32.store (i32.const 500)");
    install(&mut environment, &fixture);
    assert_eq!(
        run(&mut environment, "/app"),
        (17, b"done\n".to_vec(), Vec::new())
    );
}

#[test]
fn guest_descriptor_poll_reports_eof_without_waiting_for_clock() {
    let mut environment = Environment::new();
    let fixture = sleeper(1, 0).replacen(
        "(i32.store (i32.const 16) (i32.const 1))",
        "(i32.store (i32.const 16) (i32.const 0))",
        1,
    );
    install(&mut environment, &fixture);
    assert_eq!(
        run(&mut environment, "/app"),
        (17, b"done\n".to_vec(), Vec::new())
    );
    assert_eq!(environment.clock.monotonic_ns(), 0);
}

#[test]
fn posix_spawn_pipe_poll_and_wait_use_virtual_child_readiness() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"(module
        (import "shellsim_posix_v1" "process_pipe" (func $pipe (param i32 i32 i32) (result i32)))
        (import "shellsim_posix_v1" "process_spawn" (func $spawn (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
        (import "shellsim_posix_v1" "process_wait" (func $wait (param i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_close" (func $close (param i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "poll_oneoff" (func $poll (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (data (i32.const 1000) "/bin/sh\00-c\00sleep 1; printf z\00")
        (func (export "_start")
            (if (call $pipe (i32.const 4) (i32.const 40) (i32.const 44)) (then unreachable))
            (i32.store (i32.const 64) (i32.const 1000))
            (i32.store (i32.const 68) (i32.const 1008))
            (i32.store (i32.const 72) (i32.const 1011))
            (i32.store (i32.const 200) (i32.const 2))
            (i32.store (i32.const 204) (i32.load (i32.const 44)))
            (i32.store (i32.const 208) (i32.const 1))
            (i32.store (i32.const 224) (i32.const 3))
            (i32.store (i32.const 228) (i32.const 3))
            (if (call $spawn (i32.const 1000) (i32.const 64) (i32.const 3) (i32.const 84) (i32.const 0)
                (i32.const 200) (i32.const 2) (i32.const 0) (i32.const 0) (i32.const 32)) (then unreachable))
            (if (call $close (i32.load (i32.const 44))) (then unreachable))
            (i64.store (i32.const 0) (i64.const 7))
            (i32.store8 (i32.const 8) (i32.const 1))
            (i32.store (i32.const 16) (i32.load (i32.const 40)))
            (i64.store (i32.const 48) (i64.const 9))
            (i32.store (i32.const 64) (i32.const 1))
            (i64.store (i32.const 72) (i64.const 2000000000))
            (if (call $poll (i32.const 0) (i32.const 120) (i32.const 2) (i32.const 96)) (then unreachable))
            (if (i64.ne (i64.load (i32.const 120)) (i64.const 7)) (then unreachable))
            (i32.store (i32.const 300) (i32.const 400))
            (i32.store (i32.const 304) (i32.const 1))
            (if (call $read (i32.load (i32.const 40)) (i32.const 300) (i32.const 1) (i32.const 308)) (then unreachable))
            (if (i32.ne (i32.load8_u (i32.const 400)) (i32.const 122)) (then unreachable))
            (if (call $wait (i32.load (i32.const 32)) (i32.const 0) (i32.const 312) (i32.const 316)) (then unreachable))
            (if (i32.load (i32.const 312)) (then unreachable))
            (if (call $close (i32.load (i32.const 40))) (then unreachable))))"#,
    );
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    assert_eq!(environment.clock.monotonic_ns(), 1_000_000_000);
}

#[test]
fn wasi_directory_handles_support_relative_stat_and_cookie_reads() {
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/work/data").unwrap();
    environment
        .vfs
        .write("/", "/work/data/a", b"abc", 0o644)
        .unwrap();
    environment.vfs.mkdir_all("/", "/work/data/b").unwrap();
    install(
        &mut environment,
        r#"(module
        (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "path_filestat_get" (func $stat (param i32 i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_readdir" (func $read (param i32 i32 i32 i64 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_close" (func $close (param i32) (result i32)))
        (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
        (memory (export "memory") 1)
        (data (i32.const 512) "work/data")
        (data (i32.const 528) "a")
        (func (export "_start") (local $fd i32)
            (if (call $open (i32.const 4) (i32.const 1) (i32.const 512) (i32.const 9)
                (i32.const 2) (i64.const -1) (i64.const -1) (i32.const 0) (i32.const 0)) (then unreachable))
            (local.set $fd (i32.load (i32.const 0)))
            (if (call $stat (local.get $fd) (i32.const 1) (i32.const 528) (i32.const 1) (i32.const 64)) (then unreachable))
            (if (i64.ne (i64.load (i32.const 96)) (i64.const 3)) (then unreachable))
            ;; One dirent is 24 bytes plus the one-byte filename. Cookie 1 starts at b.
            (if (call $read (local.get $fd) (i32.const 128) (i32.const 25) (i64.const 1) (i32.const 8)) (then unreachable))
            (if (i32.ne (i32.load (i32.const 8)) (i32.const 25)) (then unreachable))
            (if (i32.ne (i32.load8_u (i32.const 152)) (i32.const 98)) (then unreachable))
            (if (i32.ne (i32.load8_u (i32.const 148)) (i32.const 3)) (then unreachable))
            (if (call $close (local.get $fd)) (then unreachable))
            (call $exit (call $read (local.get $fd) (i32.const 128) (i32.const 25) (i64.const 0) (i32.const 8)))))"#,
    );
    assert_eq!(run(&mut environment, "/app"), (8, Vec::new(), Vec::new()));
}

#[test]
fn wasi_directory_handle_growth_is_bounded() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"(module
        (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
        (memory (export "memory") 1)
        (data (i32.const 512) ".")
        (func (export "_start") (local $errno i32)
            (loop $open_more
                (local.set $errno (call $open (i32.const 4) (i32.const 1) (i32.const 512) (i32.const 1)
                    (i32.const 2) (i64.const -1) (i64.const -1) (i32.const 0) (i32.const 0)))
                (br_if $open_more (i32.eqz (local.get $errno))))
            (call $exit (local.get $errno))))"#,
    );
    assert_eq!(run(&mut environment, "/app"), (51, Vec::new(), Vec::new()));
}

#[test]
fn wasi_standard_stream_stat_is_available_and_closed_descriptors_fail() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"(module
        (import "wasi_snapshot_preview1" "fd_filestat_get" (func $stat (param i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_close" (func $close (param i32) (result i32)))
        (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
        (memory (export "memory") 1)
        (func (export "_start")
            (if (call $stat (i32.const 0) (i32.const 32)) (then unreachable))
            (if (i32.ne (i32.load8_u (i32.const 48)) (i32.const 2)) (then unreachable))
            (if (call $close (i32.const 0)) (then unreachable))
            (call $exit (call $stat (i32.const 0) (i32.const 32)))))"#,
    );
    assert_eq!(run(&mut environment, "/app"), (8, Vec::new(), Vec::new()));
}

#[test]
fn wasi_directory_reads_truncate_headers_and_validate_guest_buffers() {
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/work/data").unwrap();
    environment
        .vfs
        .write("/", "/work/data/a", b"x", 0o644)
        .unwrap();
    install(
        &mut environment,
        r#"(module
        (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_readdir" (func $read (param i32 i32 i32 i64 i32) (result i32)))
        (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
        (memory (export "memory") 1)
        (data (i32.const 512) "work/data")
        (func (export "_start") (local $fd i32)
            ;; libc opendir asks for nonblocking directory access.
            (if (call $open (i32.const 4) (i32.const 1) (i32.const 512) (i32.const 9)
                (i32.const 2) (i64.const -1) (i64.const -1) (i32.const 4) (i32.const 0)) (then unreachable))
            (local.set $fd (i32.load (i32.const 0)))
            (if (call $read (local.get $fd) (i32.const 64) (i32.const 1) (i64.const 0) (i32.const 8)) (then unreachable))
            (if (i32.ne (i32.load (i32.const 8)) (i32.const 1)) (then unreachable))
            (if (i32.ne (i32.load8_u (i32.const 64)) (i32.const 1)) (then unreachable))
            (if (call $read (local.get $fd) (i32.const 64) (i32.const 25) (i64.const 1) (i32.const 8)) (then unreachable))
            (if (i32.load (i32.const 8)) (then unreachable))
            (call $exit (call $read (local.get $fd) (i32.const -1) (i32.const 25) (i64.const 0) (i32.const 8)))))"#,
    );
    assert_eq!(run(&mut environment, "/app"), (21, Vec::new(), Vec::new()));
}

#[test]
fn wasi_readlink_copies_only_a_bounded_target_prefix() {
    let mut environment = Environment::new();
    environment
        .vfs
        .symlink("/", &"target".repeat(200_000), "/work/link")
        .unwrap();
    install(
        &mut environment,
        r#"(module
        (import "wasi_snapshot_preview1" "path_readlink" (func $link (param i32 i32 i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
        (memory (export "memory") 1)
        (data (i32.const 512) "work/link")
        (func (export "_start")
            (if (call $link (i32.const 4) (i32.const 512) (i32.const 9) (i32.const 64) (i32.const 2) (i32.const 8)) (then unreachable))
            (if (i32.ne (i32.load (i32.const 8)) (i32.const 2)) (then unreachable))
            (if (i32.ne (i32.load16_u (i32.const 64)) (i32.const 24948)) (then unreachable))
            (if (i32.ne (call $link (i32.const 4) (i32.const 512) (i32.const 9) (i32.const 64) (i32.const 1048577) (i32.const 8)) (i32.const 28)) (then unreachable))
            (call $exit (call $link (i32.const 4) (i32.const 512) (i32.const 9) (i32.const -1) (i32.const 2) (i32.const 8)))))"#,
    );
    assert_eq!(run(&mut environment, "/app"), (21, Vec::new(), Vec::new()));
}
