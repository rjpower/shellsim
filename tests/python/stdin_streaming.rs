//! Streaming `sys.stdin` reads Python programs run through the shell scheduler.
//!
//! Test-file strategy: these tests dispatch through [`Environment::run_script_capture`] (the
//! shell scheduler), not the direct [`python::run_python`] embedding API, because only the
//! scheduler-owned dispatch path exercises the new incremental `read_fd` streaming added to
//! `sys.stdin`. A producer that never terminates (`yes`) is used to prove that Python no longer
//! waits for end-of-file before running: the old implementation buffered all of stdin up front and
//! would hang (and eventually get killed) on such a pipeline.

use shellsim::{Environment, Limits};

fn run_shell(source: &str) -> (i32, String, String) {
    let mut environment = Environment::new();
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        String::from_utf8(stdout).expect("stdout is UTF-8"),
        String::from_utf8(stderr).expect("stderr is UTF-8"),
    )
}

#[test]
fn readline_returns_the_first_line_of_an_infinite_producer() {
    let (status, stdout, stderr) =
        run_shell("yes | python3 -c 'import sys; print(sys.stdin.readline().strip())'");
    assert_eq!(status, 0, "stderr: {stderr}");
    assert_eq!(stdout, "y\n");
}

#[test]
fn for_loop_over_stdin_consumes_lines_as_a_slow_producer_emits_them() {
    let mut environment = Environment::new();
    let source = "{ echo a; sleep 1; echo b; } | python3 -c '\
import sys
for line in sys.stdin:
    print(line.strip())
'";
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    let stdout = String::from_utf8(stdout).expect("stdout is UTF-8");
    let stderr = String::from_utf8(stderr).expect("stderr is UTF-8");
    assert_eq!(outcome.exit_status, 0, "stderr: {stderr}");
    assert_eq!(stdout, "a\nb\n");
    assert_eq!(environment.clock.monotonic_ns(), 1_000_000_000);
}

#[test]
fn input_raises_eof_error_on_empty_stdin() {
    let (status, stdout, stderr) = run_shell(
        "printf '' | python3 -c '\
try:\n    input()\nexcept EOFError:\n    print(\"eof\")\n'",
    );
    assert_eq!(status, 0, "stderr: {stderr}");
    assert_eq!(stdout, "eof\n");
}

#[test]
fn sized_reads_stay_on_character_boundaries_for_multi_byte_utf8() {
    // "\u{e9}" (e-acute) is a two-byte UTF-8 sequence. `read(1)` must consume one *character*
    // (both bytes), not one byte, so the streaming reader must not split it even though it reads
    // fd 0 in raw byte quanta.
    let source = "printf '\u{e9}a\\nb\\n' | python3 -c '\
import sys
print(sys.stdin.read(1))
for line in sys.stdin:
    print(repr(line))
'";
    let (status, stdout, stderr) = run_shell(source);
    assert_eq!(status, 0, "stderr: {stderr}");
    assert_eq!(stdout, "\u{e9}\n'a\\n'\n'b\\n'\n");
}

#[test]
fn read_consumes_input_larger_than_one_pipe_buffer() {
    // DEFAULT_PIPE_CAPACITY is 64 KiB; ask for more than three buffers' worth so a single
    // pipe-sized read cannot satisfy the whole request and the streaming loop must retry.
    let source = "yes | head -c 200000 | python3 -c '\
import sys
data = sys.stdin.read()
print(len(data))
print(data[:1])
'";
    let (status, stdout, stderr) = run_shell(source);
    assert_eq!(status, 0, "stderr: {stderr}");
    let mut lines = stdout.lines();
    assert_eq!(lines.next(), Some("200000"));
    assert_eq!(lines.next(), Some("y"));
}

#[test]
fn buffer_read_returns_raw_bytes_without_text_decoding() {
    // \x01 is a non-printable, non-UTF-8-multibyte byte, so it round-trips as a single raw byte
    // through both the shell's `printf` escape handling and `sys.stdin.buffer`.
    let source = "printf 'a\\x01b' | python3 -c '\
import sys
data = sys.stdin.buffer.read()
print(len(data))
print(data == b\"a\\x01b\")
'";
    let (status, stdout, stderr) = run_shell(source);
    assert_eq!(status, 0, "stderr: {stderr}");
    assert_eq!(stdout, "3\nTrue\n");
}

#[test]
fn a_memory_limit_stops_an_unbounded_stdin_read() {
    let mut environment = Environment::with_limits(Limits {
        memory: 64 * 1024,
        ..Limits::unlimited()
    });
    let source = "yes | python3 -c 'import sys; sys.stdin.read()'";
    let (outcome, stdout, _stderr) = environment.run_script_capture(source);
    assert_eq!(outcome.exit_status, 137);
    assert!(stdout.is_empty());
}

#[test]
fn getpid_getppid_report_positive_ids_and_kill_rejects_a_missing_pid() {
    let source = r#"python3 -c "
import os, signal
pid = os.getpid()
ppid = os.getppid()
print(isinstance(pid, int) and pid > 0)
print(isinstance(ppid, int) and ppid >= 0)
try:
    os.kill(999999, signal.SIGTERM)
except ProcessLookupError:
    print('no-such-pid')
"
"#;
    let (status, stdout, stderr) = run_shell(source);
    assert_eq!(status, 0, "stderr: {stderr}");
    assert_eq!(stdout, "True\nTrue\nno-such-pid\n");
}

#[test]
fn kill_with_sigkill_terminates_the_running_process() {
    let (status, _stdout, _stderr) = run_shell(
        r#"python3 -c "
import os, signal
os.kill(os.getpid(), signal.SIGKILL)
print('unreachable')
"
"#,
    );
    assert_eq!(status, 137);
}

#[test]
fn comma_separated_import_of_os_and_signal_parses_and_runs() {
    let (status, stdout, stderr) = run_shell(
        r#"python3 -c "import os, signal
print(signal.SIGTERM > 0)
print(callable(os.getpid))
"
"#,
    );
    assert_eq!(status, 0, "stderr: {stderr}");
    assert_eq!(stdout, "True\nTrue\n");
}
