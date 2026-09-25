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

#[test]
fn tac_and_tail_read_pipes_to_eof_and_resolve_child_files() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/long", &vec![b'a'; 128 * 1024], 0o644)
        .unwrap();
    let (status, stdout, stderr) = run(&mut environment, "cat /work/long | tail -c 3");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"aaa");

    let (status, stdout, stderr) = run(
        &mut environment,
        "env -C /work tail -c 3 long; printf 'one\ntwo\n' | tac",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"aaatwo\none\n");
}

#[test]
fn tee_streams_through_an_early_closing_consumer() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(&mut environment, "yes ready | tee /work/copy | head -n 3");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"ready\nready\nready\n");
    let copy = environment.vfs.read("/", "/work/copy").unwrap();
    assert!(copy.starts_with(b"ready\nready\nready\n"));
    assert!(copy.len() < 64 * 1024);
}

#[test]
fn tee_appends_and_resolves_child_paths() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf first | env -C /work tee copy; printf second | env -C /work tee -a copy; cat /work/copy",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"firstsecondfirstsecond");
}
