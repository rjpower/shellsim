//! Compatibility tests for bounded `sed` scripts, file mutation, and resource limits.

use shellsim::{Environment, Limits, StopReason};

fn run(source: &str) -> (i32, String, String) {
    let mut environment = Environment::new();
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn handles_delimiters_addresses_ranges_and_common_commands() {
    assert_eq!(
        run("printf 'a;b\\nstart\\nmid\\nend\\n' | sed -n 's/;/x/p; /start/,/end/p'"),
        (0, "axb\nstart\nmid\nend\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'a\\nb\\nc\\n' | sed '2i before; 2c changed; 3q'"),
        (0, "a\nbefore\nchanged\nc\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'abc\\n' | sed 'y/ac/AC/;='"),
        (0, "1\nAbC\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'a\\nb\\nc\\n' | sed '1,2c changed; /changed/!s/c/C/'"),
        (0, "changed\nC\n".into(), String::new())
    );

    let (status, _, stderr) = run("printf x | sed -n '0p'");
    assert_eq!(status, 2);
    assert!(stderr.contains("at least 1"), "{stderr}");
}

#[test]
fn script_files_and_in_place_writes_preserve_mode() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "script.sed", b"s/a/b/g\n", 0o644)
        .unwrap();
    environment.vfs.write("/", "tool", b"aa\n", 0o755).unwrap();
    let (outcome, stdout, stderr) =
        environment.run_script_capture("sed -i -f script.sed tool; cat tool");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"bb\n");
    assert_eq!(
        environment.vfs.metadata("/", "tool", false).unwrap().mode,
        0o755
    );
}

#[test]
fn rejects_invalid_script_files_and_occurrence_numbers() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "bad.sed", &[0xff], 0o644)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("sed -f bad.sed");
    assert_eq!(outcome.exit_status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("not valid UTF-8"));

    let (status, _, stderr) = run("printf a | sed 's/a/b/99999999999999999999'");
    assert_eq!(status, 2);
    assert!(
        stderr.contains("invalid substitution occurrence"),
        "{stderr}"
    );
}

#[test]
fn generated_text_obeys_memory_and_output_limits() {
    let mut environment = Environment::with_limits(Limits {
        memory: 128 * 1024,
        ..Limits::unlimited()
    });
    let source = "printf a | sed 's/a/&&&&&&&&&&/g; s/a/&&&&&&&&&&/g; s/a/&&&&&&&&&&/g; s/a/&&&&&&&&&&/g; s/a/&&&&&&&&&&/g'";
    let (outcome, _, _) = environment.run_script_capture(source);
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::MemoryExhausted));

    let mut environment = Environment::with_limits(Limits {
        output: 128,
        ..Limits::unlimited()
    });
    let (outcome, _, _) = environment.run_script_capture(
        "printf 'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz\\n' | sed -n 'p;p;p'",
    );
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));
}

#[test]
fn sed_reads_pipe_to_eof_and_resolves_child_cwd() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/long", &vec![b'a'; 128 * 1024], 0o644)
        .unwrap();
    let (outcome, stdout, stderr) = environment.run_script_capture("cat /work/long | sed -n '$=' ");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"1\n");

    let (outcome, stdout, stderr) = environment.run_script_capture("env -C /work sed -n '$=' long");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"1\n");
    assert!(environment
        .invocations
        .events()
        .iter()
        .any(|event| { event.pid != 1_234 && event.argv.first().is_some_and(|arg| arg == "sed") }));
}
