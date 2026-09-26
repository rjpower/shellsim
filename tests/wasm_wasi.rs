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
                (drop (call $open (i32.const 2) (i32.const 2) (i32.const 1)))))"#,
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
                (i32.const 0) (i64.const 2) (i64.const 0) (i32.const 4) (i32.const 0)))))
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
fn oversized_linear_memory_is_rejected() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        r#"(module (memory (export "memory") 512) (func (export "_start")))"#,
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
    assert_eq!(
        run(
            &mut environment,
            "/app & echo started; kill $!; wait $!; echo $?"
        ),
        (0, b"started\n143\n".to_vec(), Vec::new())
    );
    // Killing the guest returns its linear-memory reservation.
    assert_eq!(environment.resources.memory_mark(), memory);
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
fn guest_descriptor_poll_is_explicitly_unsupported() {
    let mut environment = Environment::new();
    install(&mut environment, &sleeper(1, 0));
    // WASI errno 58 is ENOTSUP.
    assert_eq!(run(&mut environment, "/app"), (58, Vec::new(), Vec::new()));
}
