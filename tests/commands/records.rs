//! Compatibility tests for buffered record-oriented text transforms.

use shellsim::interp::{Environment, Interp};
use shellsim::resources::Limits;

fn run(environment: &mut Interp, source: &str) -> (i32, Vec<u8>, String) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        stdout,
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn join_split_and_tsort_cover_common_build_script_usage() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(&mut environment, "printf '1 one\\n2 two\\n' > a; printf '1 uno\\n2 dos\\n' > b; join a b; printf 'a\\nb\\nc\\n' | split -l 2 - part; cat partaa partab; printf 'compile link\\nfetch compile\\n' | tsort");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(
        stdout,
        b"1 one uno\n2 two dos\na\nb\nc\nfetch\ncompile\nlink\n"
    );
}

#[test]
fn split_applies_the_current_umask_to_output_files() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "umask 077; printf 'one\\ntwo\\n' | split -l 1 - part; stat -c '%a' partaa partab",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"600\n600\n");
}

#[test]
fn shuf_is_a_reproducible_permutation() {
    let mut first = Environment::new();
    let mut second = Environment::new();
    let (_, first_stdout, _) = run(&mut first, "printf 'a\\nb\\nc\\nd\\n' | shuf");
    let (_, second_stdout, _) = run(&mut second, "printf 'a\\nb\\nc\\nd\\n' | shuf");
    assert_eq!(first_stdout, second_stdout);
    let mut lines = String::from_utf8(first_stdout)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines, ["a", "b", "c", "d"]);
}

#[test]
fn typed_text_commands_read_pipes_redirects_and_operands() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf 'one two\\n' > words; printf 'ab\\ncd\\n' | tr a-z A-Z | rev; wc -l < words; nl words; rev words; wc -c words; printf 'a:b\\n' | cut -d: -f2; paste words words; printf 'x\\nx\\ny\\n' | uniq -c",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(
        stdout,
        b"BA\nDC\n1\n     1\tone two\nowt eno\n8 words\nb\none two\tone two\n      2 x\n      1 y\n"
    );
}

#[test]
fn typed_text_command_with_file_operand_does_not_drain_stdin() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf 'file\\n' > words; printf 'pipe\\n' | rev words; cut -c1 words; paste words; printf 'after\\n'",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"elif\nf\nfile\nafter\n");
}

#[test]
fn typed_text_input_resumes_across_read_quanta_and_reports_file_errors() {
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file("/large", vec![b'a'; 10_000], 0o644)
        .unwrap();
    let (status, stdout, stderr) = run(&mut environment, "tr a b < large | wc -c");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"10000\n");

    let (status, stdout, stderr) = run(&mut environment, "rev absent");
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("rev: absent:"), "{stderr}");
}

#[test]
fn typed_text_input_obeys_the_memory_limit() {
    let mut environment = Environment::with_limits(Limits {
        memory: 6_000,
        ..Limits::default()
    });
    environment
        .vfs
        .put_file("/large", vec![b'a'; 10_000], 0o644)
        .unwrap();
    let (status, stdout, _) = run(&mut environment, "tr a b < large");
    assert_eq!(status, 137);
    assert!(stdout.is_empty());
}

#[test]
fn typed_text_commands_resolve_from_copied_executable_entries() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "cp /usr/bin/tr /tmp/upper; printf 'abc\\n' | /tmp/upper a-z A-Z",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"ABC\n");
}

#[test]
fn typed_text_command_reports_closed_stdin() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(&mut environment, "tr a b <&-");
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(
        stderr.contains("tr: cannot read standard input:"),
        "{stderr}"
    );
}

#[test]
fn seq_and_comm_run_as_native_children_with_visible_file_errors() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/left", b"a\nc\n", 0o644)
        .unwrap();
    environment
        .vfs
        .write("/", "/work/right", b"b\nc\n", 0o644)
        .unwrap();
    let (status, stdout, stderr) = run(&mut environment, "env -C /work comm left right; seq 1 3");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"a\n\tb\n\t\tc\n1\n2\n3\n");
    let (status, stdout, stderr) = run(&mut environment, "comm /work/missing /work/right");
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("comm: /work/missing:"), "{stderr}");
    for command in ["comm", "seq"] {
        assert!(environment.invocations.events().iter().any(|event| {
            event.pid != 1_234 && event.argv.first().is_some_and(|arg| arg == command)
        }));
    }
}

#[test]
fn buffered_record_tools_are_native_children_and_read_pipes_to_eof() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/long", &vec![b'a'; 128 * 1024], 0o644)
        .unwrap();
    let (status, stdout, stderr) = run(
        &mut environment,
        "cat /work/long | split -b 65536 - /work/part; wc -c /work/partaa /work/partab",
    );
    assert_eq!(status, 0, "{stderr}");
    assert!(String::from_utf8_lossy(&stdout).contains("65536 /work/partaa"));
    assert!(String::from_utf8_lossy(&stdout).contains("65536 /work/partab"));
    let (status, stdout, stderr) = run(&mut environment, "printf 'a 1\n' | join - /work/missing");
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("join: /work/missing:"), "{stderr}");
    let _ = run(&mut environment, "printf 'a\nb\n' | shuf");
    let _ = run(&mut environment, "printf 'a b\n' | tsort");
    for command in ["join", "split", "shuf", "tsort"] {
        assert!(environment.invocations.events().iter().any(|event| {
            event.pid != 1_234 && event.argv.first().is_some_and(|arg| arg == command)
        }));
    }
}

#[test]
fn sort_uses_child_cwd_and_reports_named_input_errors() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/unsorted", b"b\na\n", 0o644)
        .unwrap();
    let (status, stdout, stderr) = run(
        &mut environment,
        "env -C /work sort -o sorted unsorted; cat /work/sorted; printf 'd\nc\n' | sort",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"a\nb\nc\nd\n");
    let (status, stdout, stderr) = run(&mut environment, "sort /work/missing");
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("sort: /work/missing:"), "{stderr}");
    assert!(environment.invocations.events().iter().any(|event| {
        event.pid != 1_234 && event.argv.first().is_some_and(|arg| arg == "sort")
    }));
}
