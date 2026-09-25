//! Compatibility tests for `awk` grammar, evaluation, and bounded execution.

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
fn supports_basic_control_flow_arrays_and_functions() {
    assert_eq!(
        run("awk 'BEGIN { for (i = 1; i <= 3; i++) { squares[i] = i * i } total = 0; for (key in squares) { if (key == 2) continue; total += squares[key] } print total, substr(\"hello\", 2, 3), index(\"abc\", \"b\") }'"),
        (0, "10 ell 2\n".into(), String::new())
    );
    assert_eq!(
        run("awk 'BEGIN { value = \"a,b,c\"; n = split(value, parts, \",\"); changed = gsub(/,/, \"-\", value); print n, parts[2], value, changed, match(\"abc\", /b/), RSTART, RLENGTH }'"),
        (0, "3 b a-b-c 2 2 2 1\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'a1\\na2\\n' > one; printf 'b1\\nb2\\n' > two; awk '{ print; nextfile }' one two"),
        (0, "a1\nb1\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'alpha\\ngem\\n' | awk '$0 ~ /^a/ { print } $0 !~ /a$/ { print \"no-a\" }'"),
        (0, "alpha\nno-a\n".into(), String::new())
    );
    assert_eq!(
        run("awk 'BEGIN { printf \"%.4f\\n\", log(8) / log(2) }'"),
        (0, "3.0000\n".into(), String::new())
    );
}

#[test]
fn awk_reads_pipe_to_eof_and_resolves_child_files() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/long", &vec![b'a'; 128 * 1024], 0o644)
        .unwrap();
    let (outcome, stdout, stderr) =
        environment.run_script_capture("cat /work/long | awk '{ print length($0) }'");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"131072\n");

    let (outcome, stdout, stderr) =
        environment.run_script_capture("env -C /work awk '{ print length($0) }' long");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"131072\n");
}

#[test]
fn validates_the_whole_program_before_execution() {
    for (source, expected) in [
        (
            "awk 'BEGIN { print \"must not print\"; unknown(1) }'",
            "unsupported function",
        ),
        (
            "awk 'BEGIN { printf \"%q\", 1 }'",
            "unsupported printf conversion",
        ),
        ("awk 'BEGIN { print 1 > \"out\" }'", "redirection"),
        (
            "printf 'one\\n' | awk -F '[' '{ print $1 }'",
            "invalid field separator",
        ),
        ("awk 'BEGIN { getline value }'", "unsupported keyword"),
        (
            "awk 'BEGIN { NR = 4 }'",
            "assignment to 'NR' is not supported",
        ),
        (
            "awk -v NF=4 'BEGIN { print 1 }'",
            "assignment to 'NF' is not supported",
        ),
    ] {
        let (status, stdout, stderr) = run(source);
        assert_eq!(status, 2, "{source}: {stderr}");
        assert!(stdout.is_empty(), "{source}");
        assert!(stderr.contains(expected), "{source}: {stderr}");
    }
}

#[test]
fn rejects_invalid_utf8_program_files() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "bad.awk", &[0xff], 0o644)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("awk -f bad.awk");
    assert_eq!(outcome.exit_status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("not valid UTF-8"));
}

#[test]
fn command_line_special_variables_affect_execution() {
    assert_eq!(
        run("printf 'a,b\\n' | awk -v FS=, -v OFS=: '{ print $1, $2 }'"),
        (0, "a:b\n".into(), String::new())
    );
}

#[test]
fn loops_consume_modeled_cpu_fuel() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 500,
        ..Limits::unlimited()
    });
    let (outcome, stdout, _) =
        environment.run_script_capture("awk 'BEGIN { while (1) { value++ } }'");
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::CpuExhausted));
    assert_eq!(outcome.usage.cpu_used, 500);
    assert!(stdout.is_empty());
}

#[test]
fn field_and_text_growth_obey_resource_limits() {
    for source in [
        "awk 'BEGIN { $1000000000 = 1 }'",
        "awk 'BEGIN { value = \"a\"; while (1) value = value value }'",
    ] {
        let mut environment = Environment::with_limits(Limits {
            memory: 128 * 1024,
            ..Limits::unlimited()
        });
        let (outcome, _, _) = environment.run_script_capture(source);
        assert_eq!(outcome.exit_status, 137, "{source}");
        assert_eq!(outcome.stop_reason, Some(StopReason::MemoryExhausted));
    }

    let mut environment = Environment::with_limits(Limits {
        output: 128,
        ..Limits::unlimited()
    });
    let (outcome, stdout, _) =
        environment.run_script_capture("awk 'BEGIN { printf \"%1000000000s\", \"x\" }'");
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));
    assert!(stdout.is_empty());
}
