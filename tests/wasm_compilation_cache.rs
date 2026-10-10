//! Exact-byte cache reuse must preserve execution, admission, and resource acceptance.

use shellsim::{Environment, Limits};

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
