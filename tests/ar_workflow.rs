//! Check that the virtual archiver rejects invalid objects and respects process fuel.

use shellsim::{Environment, Limits, StopReason};

#[test]
fn ar_rejects_non_object_without_creating_an_archive() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/plain", b"not wasm", 0o644)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("ar rc /work/lib.a /work/plain");
    assert_eq!(outcome.exit_status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("not a WebAssembly object"));
    assert!(!environment.vfs.lexists("/", "/work/lib.a"));
}

#[test]
fn ar_stops_before_parsing_an_object_that_exceeds_cpu_fuel() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 10_000,
        ..Limits::unlimited()
    });
    environment
        .vfs
        .write("/", "/work/large.o", &vec![0; 128 * 1024], 0o644)
        .unwrap();
    let (outcome, _, _) = environment.run_script_capture("ar rc /work/lib.a /work/large.o");
    assert_eq!(outcome.stop_reason, Some(StopReason::CpuExhausted));
    assert!(!environment.vfs.lexists("/", "/work/lib.a"));
}
