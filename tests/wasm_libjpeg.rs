// Real codec probes are opt-in because they require a locally built pinned native artifact.
use shellsim::{Environment, Limits};
use std::path::PathBuf;

#[test]
fn scalar_jpeg_round_trip_and_fatal_invalid_input() {
    let Some(directory) = std::env::var_os("SHELLSIM_LIBJPEG_ARTIFACTS") else {
        return;
    };
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000_000,
        memory: 64 * 1024 * 1024,
        disk: 8 * 1024 * 1024,
        ..Limits::default()
    });
    environment
        .vfs
        .write(
            "/",
            "/jpeg",
            &std::fs::read(PathBuf::from(directory).join("probe.wasm")).unwrap(),
            0o755,
        )
        .unwrap();
    let (outcome, stdout, stderr) = environment.run_script_capture("/jpeg");
    eprintln!("JPEG round trip usage: {:?}", outcome.usage);
    assert!(outcome.usage.cpu_used > 0);
    assert_eq!(outcome.exit_status, 0);
    assert_eq!(stdout, b"JPEG round trip: 2x1 RGB within tolerance 3\n");
    assert!(stderr.is_empty());
    let (outcome, stdout, stderr) = environment.run_script_capture("/jpeg invalid");
    eprintln!("JPEG invalid input usage: {:?}", outcome.usage);
    assert_eq!(outcome.exit_status, 1);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
}
