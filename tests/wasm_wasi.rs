// The fixtures exercise the WASI ABI from compiled Wasm, not a host process or filesystem.
use shellsim::{Environment, Limits};
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
