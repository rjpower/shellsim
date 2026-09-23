//! Compatibility tests for resumable byte and line streams.

use shellsim::interp::{Environment, Interp};

fn run(environment: &mut Interp, source: &str) -> (i32, Vec<u8>, String) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        stdout,
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn yes_streams_until_downstream_closes_the_pipe() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(&mut environment, "yes ready | head -n 3");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"ready\nready\nready\n");
}
