//! CLI boundary tests exercise host stdin routing without granting simulated code host access.

use std::io::Write;
use std::process::{Command, Output, Stdio};

fn run_with_stdin(args: &[&str], stdin: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_shellsim"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start shellsim CLI");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(stdin)
        .expect("write shellsim stdin");
    child.wait_with_output().expect("wait for shellsim CLI")
}

#[test]
fn command_action_receives_host_stdin_and_has_a_tmp_directory() {
    let output = run_with_stdin(
        &["-c", "cat > /tmp/input; printf prefix:; cat /tmp/input"],
        b"payload\n",
    );

    assert!(output.status.success());
    assert_eq!(output.stdout, b"prefix:payload\n");
    assert!(output.stderr.is_empty());
}

#[test]
fn action_console_collects_a_complete_heredoc() {
    let output = run_with_stdin(
        &["shell"],
        b"cat > sample <<'TEXT'\nhello\nTEXT\ncat sample\n",
    );

    assert!(output.status.success());
    assert_eq!(output.stdout, b"hello\n");
    assert!(output.stderr.is_empty());
}

#[test]
fn structured_evaluation_receives_action_stdin() {
    let output = run_with_stdin(&["eval", "-c", "cat"], b"payload\n");

    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("valid evaluation report");
    assert_eq!(report["stdout"], "payload\n");
    assert_eq!(report["outcome"]["exit_status"], 0);
}
