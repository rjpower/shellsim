// WAT covers the runtime gate; opt-in SDK probes cover LLVM lowering and cross-archive ABI.
use shellsim::{Environment, Limits};
use std::path::PathBuf;

#[test]
fn standard_wasm_exception_payload_is_supported() {
    let bytes = wat::parse_str(
        r#"(module
            (tag $error (param i32))
            (func (export "_start")
                (if (i32.ne
                    (block $caught (result i32)
                        (try_table (catch $error $caught)
                            (throw $error (i32.const 42)))
                        (i32.const 0))
                    (i32.const 42)) (then unreachable))))"#,
    )
    .unwrap();
    let mut environment = Environment::new();
    environment.vfs.write("/", "/probe", &bytes, 0o755).unwrap();
    let (outcome, stdout, stderr) = environment.run_script_capture("/probe");
    assert_eq!(outcome.exit_status, 0, "{stderr:?}");
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

fn install(environment: &mut Environment, name: &str) {
    let directory = std::env::var_os("SHELLSIM_EXCEPTION_ARTIFACTS")
        .expect("set SHELLSIM_EXCEPTION_ARTIFACTS to the verified SDK probe directory");
    let bytes = std::fs::read(PathBuf::from(directory).join(format!("{name}.wasm"))).unwrap();
    environment.vfs.write("/", "/probe", &bytes, 0o755).unwrap();
}

#[test]
#[ignore = "requires verified SDK C/C++ exception artifacts"]
fn compiler_setjmp_nested_cross_library_recovery_and_zero_normalization() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000_000,
        memory: 64 * 1024 * 1024,
        ..Limits::default()
    });
    install(&mut environment, "setjmp");
    let (outcome, stdout, stderr) = environment.run_script_capture("/probe");
    assert_eq!(outcome.exit_status, 0, "{stderr:?}");
    assert_eq!(
        stdout,
        b"setjmp: nested cross-library recovery and zero normalization passed\n"
    );
    assert!(stderr.is_empty());
}

#[test]
#[ignore = "requires verified SDK C/C++ exception artifacts"]
fn compiler_cpp_cross_library_catch_rethrow_and_destructors() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000_000,
        memory: 64 * 1024 * 1024,
        ..Limits::default()
    });
    install(&mut environment, "cpp");
    let (outcome, stdout, stderr) = environment.run_script_capture("/probe");
    assert_eq!(outcome.exit_status, 0, "{stderr:?}");
    assert_eq!(
        stdout,
        b"C++: cross-library catch, rethrow and destruction passed\n"
    );
    assert!(stderr.is_empty());
    let (outcome, stdout, _) = environment.run_script_capture("/probe uncaught");
    assert_ne!(outcome.exit_status, 0);
    assert!(stdout.is_empty());
}

#[test]
#[ignore = "requires verified SDK C/C++ exception artifacts"]
fn compiler_nonlocal_jumps_obey_cpu_budget() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 10_000_000,
        memory: 64 * 1024 * 1024,
        ..Limits::default()
    });
    install(&mut environment, "setjmp");
    let (outcome, _, _) = environment.run_script_capture("/probe loop");
    assert_ne!(outcome.exit_status, 0);
    assert!(outcome.usage.cpu_used > 0);
    assert_eq!(
        outcome.stop_reason,
        Some(shellsim::StopReason::CpuExhausted)
    );
}

#[test]
#[ignore = "requires verified SDK C/C++ exception artifacts"]
fn compiler_repeated_exceptions_reclaim_unreachable_objects() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000_000,
        memory: 64 * 1024 * 1024,
        ..Limits::default()
    });
    install(&mut environment, "setjmp");
    let (outcome, stdout, stderr) = environment.run_script_capture("/probe stress");
    assert_eq!(outcome.exit_status, 0, "{stderr:?}");
    assert_eq!(stdout, b"setjmp: one million recoveries passed\n");
    assert!(stderr.is_empty());
    assert!(outcome.usage.memory_peak <= 64 * 1024 * 1024);
}

#[test]
fn wasm_gc_syntax_remains_outside_the_guest_frontier() {
    let bytes = wat::parse_str(r#"(module (type $object (struct (field i32))) (func (export "_start") (drop (struct.new $object (i32.const 1)))))"#).unwrap();
    let mut environment = Environment::new();
    environment.vfs.write("/", "/probe", &bytes, 0o755).unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("/probe");
    assert_eq!(outcome.exit_status, 126);
    assert!(!stderr.is_empty());
}

#[test]
fn retained_exception_payloads_share_the_linear_memory_budget() {
    let params = " i32".repeat(128);
    let values = " (i32.const 7)".repeat(128);
    let bytes = wat::parse_str(format!(
        r#"(module
            (tag $error (param {params}))
            (memory 32)
            (table 4096 (ref null exn))
            (func (export "_start") (local $index i32)
                (loop $next
                    (table.set (local.get $index)
                        (block $caught (result (ref exn))
                            (try_table (catch_all_ref $caught) (throw $error {values}))
                            unreachable))
                    (local.set $index (i32.add (local.get $index) (i32.const 1)))
                    (br_if $next (i32.lt_u (local.get $index) (i32.const 4096))))))"#
    ))
    .unwrap();
    for (memory, expected) in [(4 * 1024 * 1024, 126), (16 * 1024 * 1024, 0)] {
        let mut environment = Environment::with_limits(Limits {
            cpu: 100_000_000,
            memory,
            ..Limits::default()
        });
        environment.vfs.write("/", "/probe", &bytes, 0o755).unwrap();
        let (outcome, stdout, stderr) = environment.run_script_capture("/probe");
        assert_eq!(outcome.exit_status, expected, "{stderr:?}");
        assert!(stdout.is_empty());
        assert!(outcome.usage.memory_peak <= memory);
    }
}
