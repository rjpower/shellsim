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

#[test]
fn head_stops_infinite_input_and_reads_child_files() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(&mut environment, "yes ready | head -n 3");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"ready\nready\nready\n");

    environment
        .vfs
        .write("/", "/work/long", &vec![b'a'; 128 * 1024], 0o644)
        .unwrap();
    let (status, stdout, stderr) = run(&mut environment, "env -C /work head -c 3 long");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"aaa");

    environment
        .vfs
        .write("/", "/work/short", b"one\ntwo\n", 0o644)
        .unwrap();
    let (status, stdout, stderr) = run(&mut environment, "env -C /work head -n 1 short short");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"==> short <==\none\n\n==> short <==\none\n");
}

#[test]
fn stream_operand_boundaries_preserve_dash_prefixed_files_and_stdin() {
    for command in ["head", "tail"] {
        let mut environment = Environment::new();
        for file in ["-n", "-c", "--unknown", "--"] {
            environment
                .vfs
                .write("/", &format!("/work/{file}"), b"one\ntwo\n", 0o644)
                .unwrap();
            let (status, stdout, stderr) =
                run(&mut environment, &format!("cd /work; {command} -- {file}"));
            assert_eq!(status, 0, "{command} {file}: {stderr}");
            assert_eq!(stdout, b"one\ntwo\n", "{command} {file}");
            assert!(stderr.is_empty(), "{stderr}");
        }
        for operand in ["", " -"] {
            let (status, stdout, stderr) = run(
                &mut environment,
                &format!("printf 'one\\ntwo\\n' | {command} --{operand}"),
            );
            assert_eq!(status, 0, "{command}: {stderr}");
            assert_eq!(stdout, b"one\ntwo\n");
            assert!(stderr.is_empty(), "{stderr}");
        }
        let (status, stdout, stderr) = run(&mut environment, &format!("{command} --unknown"));
        assert_ne!(status, 0, "{command}");
        assert!(stdout.is_empty());
        assert!(!stderr.is_empty());
    }
}

#[test]
fn head_operand_boundary_preserves_early_pipe_termination() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(&mut environment, "yes ready | head -n 3 -- -");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"ready\nready\nready\n");
    assert!(stderr.is_empty(), "{stderr}");
}

#[test]
fn stream_counts_reject_overflow_and_unsupported_negative_head_counts() {
    let overflowing = "99999999999999999999999999999999999999999999999999";
    for command in ["head", "tail"] {
        let mut environment = Environment::new();
        for option in [format!("-{overflowing}"), format!("-n {overflowing}")] {
            let (status, stdout, stderr) = run(
                &mut environment,
                &format!("printf 'one\\ntwo\\n' | {command} {option}"),
            );
            assert_ne!(status, 0, "{command} {option}");
            assert!(stdout.is_empty());
            assert!(!stderr.is_empty());
        }
    }
    let mut environment = Environment::new();
    for option in ["-n -1", "-n-1", "-c -1", "-c-1"] {
        let (status, stdout, stderr) = run(
            &mut environment,
            &format!("printf 'one\\ntwo\\n' | head {option}"),
        );
        assert_ne!(status, 0, "head {option}");
        assert!(stdout.is_empty());
        assert!(!stderr.is_empty());
    }
}

#[test]
fn stream_count_options_select_the_last_mode() {
    for (command, options, expected) in [
        ("head", "-c 1 -n 1", b"one\n".as_slice()),
        ("head", "-n 1 -c 1", b"o".as_slice()),
        ("head", "-c 1 -n1", b"one\n".as_slice()),
        ("head", "-c 1 -1", b"one\n".as_slice()),
        ("tail", "-c 1 -n 1", b"two\n".as_slice()),
        ("tail", "-n 1 -c 1", b"\n".as_slice()),
        ("tail", "-c 1 -n1", b"two\n".as_slice()),
        ("tail", "-c 1 -1", b"two\n".as_slice()),
    ] {
        let mut environment = Environment::new();
        let (status, stdout, stderr) = run(
            &mut environment,
            &format!("printf 'one\\ntwo\\n' | {command} {options}"),
        );
        assert_eq!(status, 0, "{command} {options}: {stderr}");
        assert_eq!(stdout, expected, "{command} {options}");
        assert!(stderr.is_empty(), "{stderr}");
    }
}
