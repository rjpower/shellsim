//! Exact-byte cache reuse must preserve execution, admission, and resource acceptance.

use shellsim::{
    commands::{
        precompile_wasm, register_trusted_precompiled_wasm, PrecompiledWasmOutcome,
        WasmCompilationKind, WasmSession,
    },
    Environment, Limits,
};

fn execute(source: &[u8], limits: Limits) -> (shellsim::RunOutcome, Vec<u8>, Vec<u8>) {
    let mut environment = Environment::with_limits(limits);
    environment
        .vfs
        .write("/", "/cache-probe", source, 0o755)
        .unwrap();
    environment.run_script_capture("/cache-probe")
}

#[test]
fn warm_executable_preserves_usage_and_rejects_changed_or_forbidden_bytes() {
    let source = wat::parse_str(
        "(module (func (export \"_start\") (local i32) (local.set 0 (i32.const 379))))",
    )
    .unwrap();
    let (cold, stdout, stderr) = execute(&source, Limits::default());
    assert_eq!(cold.exit_status, 0);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
    let companion = precompile_wasm(&source, WasmCompilationKind::Executable).unwrap();
    // SAFETY: the host compiler returned this untouched companion.
    assert_eq!(
        unsafe {
            register_trusted_precompiled_wasm(&source, &companion, WasmCompilationKind::Executable)
        }
        .unwrap(),
        PrecompiledWasmOutcome::Registered
    );
    let (warm, stdout, stderr) = execute(&source, Limits::default());
    assert_eq!(warm.exit_status, 0);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
    assert_eq!(cold.usage.cpu_used, warm.usage.cpu_used);
    assert_eq!(cold.usage.memory_peak, warm.usage.memory_peak);
    assert_eq!(cold.usage.memory_current, warm.usage.memory_current);
    assert_eq!(cold.usage.disk_peak, warm.usage.disk_peak);
    let (limited, _, _) = execute(
        &source,
        Limits {
            cpu: 1,
            ..Limits::default()
        },
    );
    assert_eq!(limited.exit_status, 137);

    let changed = wat::parse_str("(module (func (export \"_start\") unreachable))").unwrap();
    // SAFETY: a genuine companion can safely be checked against different source bytes.
    assert_eq!(
        unsafe {
            register_trusted_precompiled_wasm(&changed, &companion, WasmCompilationKind::Executable)
        }
        .unwrap(),
        PrecompiledWasmOutcome::SourceMismatch
    );
    assert_eq!(execute(&changed, Limits::default()).0.exit_status, 126);
    let forbidden = wat::parse_str(
        "(module (memory 1 1 shared) (func (export \"_start\") (drop (memory.atomic.wait32 (i32.const 0) (i32.const 0) (i64.const -1)))))"
    ).unwrap();
    for _ in 0..2 {
        let (outcome, _, stderr) = execute(&forbidden, Limits::default());
        assert_eq!(outcome.exit_status, 126);
        assert!(String::from_utf8_lossy(&stderr)
            .contains("raw atomic wait/notify is outside the scheduler ABI"));
    }
}

#[test]
fn guest_replacement_cannot_select_native_companions_or_host_registration() {
    let source = wat::parse_str(
        r#"(module
            (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
            (func (export "_start") (call $exit (i32.const 23))))"#,
    )
    .unwrap();
    let companion = precompile_wasm(&source, WasmCompilationKind::Executable).unwrap();
    // SAFETY: the companion came directly from trusted host compilation.
    unsafe {
        register_trusted_precompiled_wasm(&source, &companion, WasmCompilationKind::Executable)
    }
    .unwrap();
    let mut environment = Environment::new();
    environment.vfs.write("/", "/app", &source, 0o755).unwrap();
    environment
        .vfs
        .write("/", "/app.cwasm", &companion, 0o755)
        .unwrap();
    assert_eq!(environment.run_script_capture("/app").0.exit_status, 23);
    let replacement = wat::parse_str(
        r#"(module
            (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
            (func (export "_start") (call $exit (i32.const 41))))"#,
    )
    .unwrap();
    environment
        .vfs
        .write("/", "/app", &replacement, 0o755)
        .unwrap();
    assert_eq!(environment.run_script_capture("/app").0.exit_status, 41);
    environment
        .vfs
        .write("/", "/app", &companion, 0o755)
        .unwrap();
    let error = WasmSession::start(environment, "/app", &[])
        .err()
        .expect("native companion in guest VFS must not be a Wasm executable");
    assert!(error.contains("not a Wasm executable"), "{error}");

    let guest = wat::parse_str(
        r#"(module
            (import "shellsim" "register_trusted_precompiled_wasm" (func $register))
            (func (export "_start") (call $register)))"#,
    )
    .unwrap();
    // Ordinary module compilation can retain unsupported imports, but guest admission must
    // still reject them after host registration.
    let companion = precompile_wasm(&guest, WasmCompilationKind::Module).unwrap();
    // SAFETY: the payload is untouched host compiler output.
    unsafe { register_trusted_precompiled_wasm(&guest, &companion, WasmCompilationKind::Module) }
        .unwrap();
    let (outcome, _, stderr) = execute(&guest, Limits::default());
    assert_eq!(outcome.exit_status, 126);
    assert!(String::from_utf8_lossy(&stderr)
        .contains("unsupported wasm import: shellsim.register_trusted_precompiled_wasm"));
}
