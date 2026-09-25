//! Compatibility tests for deterministic line and column formatting.

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
fn formatting_commands_transform_columns_and_tabs() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(&mut environment, "printf 'a\\tb\\n' | expand -t 4; printf 'a   b\\n' | unexpand -a -t 4; printf 'a:long\\nb:x\\n' | column -t -s :; printf 'one two three four\\n' | fmt -w 9; printf 'abcdef\\n' | fold -w 3");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(
        stdout,
        b"a   b\na\tb\na  long\nb  x\none two\nthree\nfour\nabc\ndef\n"
    );
}

#[test]
fn formatting_commands_read_named_files_through_the_child_process() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/words", b"abcde\n", 0o644)
        .unwrap();
    let (status, stdout, stderr) = run(
        &mut environment,
        "env -C /work fold -w 3 words; expand missing",
    );
    assert_eq!(status, 1);
    assert_eq!(stdout, b"abc\nde\n");
    assert!(stderr.contains("expand: missing:"), "{stderr}");
    assert!(environment.invocations.events().iter().any(|event| {
        event.pid != 1_234 && event.argv.first().is_some_and(|arg| arg == "fold")
    }));
}

#[test]
fn formatting_commands_finish_after_pipe_input_spans_multiple_quanta() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/long", &vec![b'a'; 128 * 1024], 0o644)
        .unwrap();
    let direct = run(&mut environment, "fold -w 80 /work/long | wc -l");
    let piped = run(&mut environment, "cat /work/long | fold -w 80 | wc -l");
    assert_eq!(direct.0, 0, "{}", direct.2);
    assert_eq!(piped, direct);
}
