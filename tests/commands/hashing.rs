//! Hash and encoding commands must read the active process's descriptors and VFS view.

use shellsim::interp::{Environment, Interp};
use shellsim::{Limits, StopReason};

fn run(environment: &mut Interp, source: &str) -> (i32, String, String) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn printf_and_hashes_are_process_images_with_child_cwd() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/word", b"abc", 0o644)
        .unwrap();
    let digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert_eq!(
        run(&mut environment, "/usr/bin/printf %s abc | sha256sum"),
        (0, format!("{digest}  -\n"), String::new())
    );
    assert_eq!(
        run(&mut environment, "env -C /work sha256sum word"),
        (0, format!("{digest}  word\n"), String::new())
    );
    for command in ["printf", "sha256sum"] {
        assert!(environment.invocations.events().iter().any(|event| {
            event.pid != 1_234 && event.argv.first().is_some_and(|arg| arg.ends_with(command))
        }));
    }
}

#[test]
fn hash_check_and_encoding_handle_files_pipes_and_failures() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/word", b"abc", 0o644)
        .unwrap();
    environment
        .vfs
        .write(
            "/",
            "/work/checks",
            b"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  /work/word\n",
            0o644,
        )
        .unwrap();
    assert_eq!(
        run(&mut environment, "sha256sum -c /work/checks"),
        (0, "/work/word: OK\n".into(), String::new())
    );
    assert_eq!(
        run(&mut environment, "cat /work/word | base64"),
        (0, "YWJj\n".into(), String::new())
    );
    assert_eq!(
        run(&mut environment, "printf YWJj | base64 -d"),
        (0, "abc".into(), String::new())
    );
    let missing = run(&mut environment, "sha256sum /work/missing");
    assert_eq!(missing.0, 1);
    assert!(missing.2.contains("/work/missing"), "{}", missing.2);
}

#[test]
fn hashing_waits_for_all_pipe_input() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/long", &vec![b'a'; 128 * 1024], 0o644)
        .unwrap();
    let direct = run(&mut environment, "sha256sum /work/long");
    let piped = run(&mut environment, "cat /work/long | sha256sum");
    assert_eq!(direct.0, 0, "{}", direct.2);
    assert_eq!(piped.0, 0, "{}", piped.2);
    assert_eq!(
        direct.1.split_whitespace().next(),
        piped.1.split_whitespace().next()
    );
}

#[test]
fn hashing_named_input_obeys_the_virtual_cpu_limit() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 10_000,
        ..Limits::unlimited()
    });
    environment
        .vfs
        .write("/", "/work/long", &vec![b'a'; 128 * 1024], 0o644)
        .unwrap();
    let (outcome, stdout, _) = environment.run_script_capture("sha256sum /work/long");
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::CpuExhausted));
    assert!(stdout.is_empty());
}
