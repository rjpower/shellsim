//! CLI boundary tests exercise host stdin routing without granting simulated code host access.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "shellsim-serve-test-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

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

#[test]
fn persistent_protocol_replays_actions_and_workspace_changes() {
    let requests = [
        r#"{"id":1,"op":"write_file","path":"note","data_base64":"aGVsbG8="}"#,
        r#"{"id":2,"op":"checkpoint"}"#,
        r#"{"id":3,"op":"execute","source":"printf ' world' >> note; cat note"}"#,
        r#"{"id":4,"op":"workspace_diff"}"#,
        r#"{"id":5,"op":"reset_workspace"}"#,
        r#"{"id":6,"op":"read_file","path":"note"}"#,
        r#"{"id":7,"op":"inspect"}"#,
        r#"{"id":8,"op":"execute","source":"net route https://example.test 200 ok; curl https://example.test"}"#,
    ]
    .join("\n")
        + "\n";
    let output = run_with_stdin(&["serve"], requests.as_bytes());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(responses.len(), 8);
    assert!(responses.iter().all(|response| response["ok"] == true));
    assert_eq!(responses[2]["result"]["kind"], "execute");
    assert_eq!(responses[2]["result"]["stdout_base64"], "aGVsbG8gd29ybGQ=");
    assert_eq!(responses[3]["result"]["changes"][0]["path"], "/work/note");
    assert_eq!(responses[3]["result"]["changes"][0]["change"], "modified");
    assert_eq!(responses[5]["result"]["data_base64"], "aGVsbG8=");
    assert_eq!(responses[6]["result"]["kind"], "inspect");
    assert_eq!(responses[6]["result"]["cwd"], "/work");
    assert_eq!(responses[7]["result"]["stdout_base64"], "b2s=");
    assert_eq!(
        responses[7]["result"]["network_requests"][0]["method"],
        "GET"
    );
    assert_eq!(
        responses[7]["result"]["network_requests"][0]["matched"],
        true
    );
}

#[test]
fn persistent_protocol_reports_malformed_requests_and_continues() {
    let output = run_with_stdin(
        &["serve"],
        b"not json\n{\"id\":2,\"op\":\"read_file\",\"path\":\"/tmp/host\"}\n{\"id\":3,\"op\":\"list_paths\"}\n",
    );
    assert!(output.status.success());
    let responses = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(responses.len(), 3);
    assert_eq!(responses[0]["ok"], false);
    assert!(responses[0]["error"]
        .as_str()
        .unwrap()
        .contains("invalid request"));
    assert_eq!(responses[1]["ok"], false);
    assert_eq!(responses[1]["id"], 2);
    assert!(responses[1]["error"]
        .as_str()
        .unwrap()
        .contains("confined to /work"));
    assert_eq!(responses[2]["ok"], true);
    assert_eq!(responses[2]["id"], 3);
}

#[test]
fn persistent_protocol_reports_ordered_invocations_and_scheduler_waits() {
    let requests = [
        r#"{"id":1,"op":"execute","source":"sed 's/a/b/' /missing"}"#,
        r#"{"id":2,"op":"execute","source":"sed 's/a/b/' /missing; sleep 10 &"}"#,
        r#"{"id":3,"op":"execute","source":"true"}"#,
        r#"{"id":4,"op":"inspect"}"#,
    ]
    .join("\n")
        + "\n";
    let output = run_with_stdin(&["serve"], requests.as_bytes());
    assert!(output.status.success());
    let responses = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(responses[0]["result"]["partial_commands"][0], "sed");
    assert_eq!(responses[1]["result"]["partial_commands"][0], "sed");
    assert_eq!(responses[0]["result"]["invocations"][0]["trust"], "partial");
    assert_eq!(responses[0]["result"]["invocations"][0]["status"], 0);
    let processes = responses[3]["result"]["processes"].as_array().unwrap();
    let sleeping = processes
        .iter()
        .find(|process| process["command"] == "sleep 10")
        .expect("retained sleep process");
    assert_eq!(sleeping["status"]["state"], "blocked");
    assert_eq!(sleeping["status"]["reason"]["kind"], "timer");
    assert_eq!(responses[3]["result"]["pending_events"], 1);
    assert!(responses[3]["result"]["current_pid"].is_number());
}

#[test]
fn persistent_protocol_polls_retained_actions_and_streams_input_output() {
    let requests = [
        r#"{"id":1,"op":"start_execute","source":"printf one; sleep 2; printf two"}"#,
        r#"{"id":2,"op":"poll_action","action_id":0,"work_quanta":100,"advance_time":false}"#,
        r#"{"id":3,"op":"read_action_output","action_id":0}"#,
        r#"{"id":4,"op":"poll_action","action_id":0,"work_quanta":100,"advance_time":true}"#,
        r#"{"id":5,"op":"read_action_output","action_id":0}"#,
        r#"{"id":6,"op":"drop_action","action_id":0}"#,
        r#"{"id":7,"op":"start_execute","source":"cat","stdin_closed":false}"#,
        r#"{"id":8,"op":"poll_action","action_id":1,"work_quanta":100}"#,
        r#"{"id":9,"op":"write_stdin","action_id":1,"data_base64":"aGVsbG8="}"#,
        r#"{"id":10,"op":"close_stdin","action_id":1}"#,
        r#"{"id":11,"op":"poll_action","action_id":1,"work_quanta":100}"#,
        r#"{"id":12,"op":"read_action_output","action_id":1}"#,
        r#"{"id":13,"op":"drop_action","action_id":1}"#,
        r#"{"id":14,"op":"start_execute","source":"sleep 10"}"#,
        r#"{"id":15,"op":"poll_action","action_id":2,"work_quanta":100}"#,
        r#"{"id":16,"op":"signal_process","pid":1234,"signal":"BOGUS"}"#,
        r#"{"id":17,"op":"signal_process","pid":1234,"signal":"INT"}"#,
        r#"{"id":18,"op":"poll_action","action_id":2,"work_quanta":100}"#,
    ]
    .join("\n")
        + "\n";
    let output = run_with_stdin(&["serve"], requests.as_bytes());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();

    assert!(responses[..15]
        .iter()
        .all(|response| response["ok"] == true));
    assert_eq!(responses[1]["result"]["state"]["state"], "blocked");
    assert_eq!(responses[1]["result"]["state"]["reason"]["kind"], "timer");
    assert_eq!(responses[2]["result"]["stdout_base64"], "b25l");
    assert_eq!(responses[3]["result"]["state"]["state"], "complete");
    assert_eq!(responses[4]["result"]["stdout_base64"], "dHdv");
    assert_eq!(
        responses[7]["result"]["state"]["reason"]["kind"],
        "input_readable"
    );
    assert_eq!(responses[10]["result"]["state"]["state"], "complete");
    assert_eq!(responses[11]["result"]["stdout_base64"], "aGVsbG8=");
    assert_eq!(responses[11]["result"]["stdout_closed"], true);
    assert_eq!(responses[14]["result"]["state"]["state"], "blocked");
    assert_eq!(responses[15]["ok"], false);
    assert!(responses[15]["error"]
        .as_str()
        .unwrap()
        .contains("unsupported signal"));
    assert_eq!(responses[16]["ok"], true);
    assert_eq!(responses[17]["result"]["state"]["state"], "complete");
    assert_eq!(responses[17]["result"]["state"]["status"], 130);
    assert_eq!(responses[17]["result"]["invocations"][0]["status"], 130);
}

#[test]
fn persistent_protocol_routes_isolated_complete_state_forks() {
    let requests = [
        r#"{"id":1,"op":"execute","source":"X=parent; printf base > value"}"#,
        r#"{"id":2,"op":"fork_session","source":0}"#,
        r#"{"id":3,"session_id":1,"op":"execute","source":"X=child; printf branch > value"}"#,
        r#"{"id":4,"op":"execute","source":"printf '%s:' \"$X\"; cat value"}"#,
        r#"{"id":5,"session_id":1,"op":"execute","source":"printf '%s:' \"$X\"; cat value"}"#,
        r#"{"id":6,"op":"drop_session","target":1}"#,
        r#"{"id":7,"session_id":1,"op":"inspect"}"#,
    ]
    .join("\n")
        + "\n";
    let output = run_with_stdin(&["serve"], requests.as_bytes());
    assert!(output.status.success());
    let responses = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(responses[1]["result"]["kind"], "session");
    assert_eq!(responses[1]["result"]["session_id"], 1);
    assert_eq!(responses[3]["result"]["stdout_base64"], "cGFyZW50OmJhc2U=");
    assert_eq!(responses[4]["result"]["stdout_base64"], "Y2hpbGQ6YnJhbmNo");
    assert_eq!(responses[5]["ok"], true);
    assert_eq!(responses[6]["ok"], false);
    assert!(responses[6]["error"]
        .as_str()
        .unwrap()
        .contains("session 1 does not exist"));
}

#[test]
fn persistent_protocol_exposes_typed_workspace_operations() {
    let patch = "*** Begin Patch\n*** Update File: src/note\n@@\n-old\n+new\n*** End Patch\n";
    let requests = [
        r#"{"id":1,"op":"make_directory","path":"src","mode":488}"#.to_string(),
        r#"{"id":2,"op":"write_file","path":"src/note","data_base64":"b2xkCg==","mode":416}"#
            .to_string(),
        r#"{"id":3,"op":"create_symlink","path":"note-link","target":"src/note"}"#.to_string(),
        r#"{"id":4,"op":"stat_path","path":"note-link","follow_symlinks":false}"#.to_string(),
        serde_json::json!({"id": 5, "op": "apply_patch", "patch": patch, "strip": 1}).to_string(),
        r#"{"id":6,"op":"read_file","path":"src/note"}"#.to_string(),
    ]
    .join("\n")
        + "\n";
    let output = run_with_stdin(&["serve"], requests.as_bytes());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();

    assert!(responses.iter().all(|response| response["ok"] == true));
    assert_eq!(responses[3]["result"]["kind"], "path_metadata");
    assert_eq!(responses[3]["result"]["node_type"], "symlink");
    assert_eq!(responses[3]["result"]["mode"], 0o777);
    assert_eq!(responses[3]["result"]["symlink_target"], "src/note");
    assert_eq!(responses[5]["result"]["data_base64"], "bmV3Cg==");
}

#[test]
fn persistent_protocol_can_checkpoint_a_trusted_host_snapshot() {
    let project = TestDirectory::new();
    std::fs::write(project.path().join("input.txt"), b"snapshot\0bytes").unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_shellsim"))
        .args(["serve", "--root"])
        .arg(project.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(
            b"{\"id\":1,\"op\":\"read_file\",\"path\":\"input.txt\"}\n{\"id\":2,\"op\":\"workspace_diff\"}\n",
        )
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        responses[0]["result"]["data_base64"],
        "c25hcHNob3QAYnl0ZXM="
    );
    assert_eq!(responses[1]["result"]["changes"], serde_json::json!([]));
}

#[test]
fn scenario_replay_emits_paired_transcript_records() {
    let directory = TestDirectory::new();
    let scenario = directory.path().join("scenario.ndjson");
    std::fs::write(
        &scenario,
        concat!(
            "{\"id\":\"write\",\"op\":\"execute\",\"source\":\"printf saved > note\"}\n",
            "{\"id\":\"read\",\"op\":\"read_file\",\"path\":\"note\"}\n"
        ),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_shellsim"))
        .arg("replay")
        .arg(&scenario)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let records = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0]["sequence"], 0);
    assert_eq!(records[0]["request"]["id"], "write");
    assert_eq!(records[0]["response"]["ok"], true);
    assert_eq!(records[1]["request"]["id"], "read");
    assert_eq!(records[1]["response"]["result"]["data_base64"], "c2F2ZWQ=");
}

#[test]
fn scenario_replay_persists_complete_transcript_without_overwrite() {
    let directory = TestDirectory::new();
    let scenario = directory.path().join("scenario.ndjson");
    let transcript = directory.path().join("transcript.ndjson");
    std::fs::write(
        &scenario,
        b"{\"id\":\"run\",\"op\":\"execute\",\"source\":\"printf saved\"}\n",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_shellsim"))
        .arg("replay")
        .arg(&scenario)
        .arg("--transcript")
        .arg(&transcript)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(std::fs::read(&transcript).unwrap(), output.stdout);

    std::fs::write(&transcript, b"keep\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_shellsim"))
        .arg("replay")
        .arg(&scenario)
        .arg("--transcript")
        .arg(&transcript)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("destination already exists"));
    assert_eq!(std::fs::read(&transcript).unwrap(), b"keep\n");
}

#[test]
fn scenario_replay_rejects_invalid_actions() {
    let directory = TestDirectory::new();
    let scenario = directory.path().join("invalid.ndjson");
    let transcript = directory.path().join("transcript.ndjson");
    std::fs::write(&scenario, b"not json\n").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_shellsim"))
        .arg("replay")
        .arg(&scenario)
        .arg("--transcript")
        .arg(&transcript)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("action 1 is invalid"));
    assert!(output.stdout.is_empty());
    assert!(!transcript.exists());
}

#[test]
fn scenario_replay_checks_typed_expectations() {
    let directory = TestDirectory::new();
    let passing = directory.path().join("passing.ndjson");
    std::fs::write(
        &passing,
        concat!(
            "{\"request\":{\"id\":\"run\",\"op\":\"execute\",\"source\":\"printf ok > note; printf done\"},\"expect\":{\"ok\":true,\"exit_status\":0,\"stdout_base64\":\"ZG9uZQ==\",\"stderr_base64\":\"\",\"unsupported\":[],\"noop_commands\":[],\"partial_commands\":[]}}\n",
            "{\"request\":{\"id\":\"diff\",\"op\":\"workspace_diff\"},\"expect\":{\"workspace_change_count\":1}}\n"
        ),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_shellsim"))
        .arg("replay")
        .arg(&passing)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let records = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0]["assertion"]["passed"], true);
    assert_eq!(records[1]["assertion"]["passed"], true);

    let failing = directory.path().join("failing.ndjson");
    std::fs::write(
        &failing,
        b"{\"request\":{\"op\":\"execute\",\"source\":\"false\"},\"expect\":{\"exit_status\":0}}\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_shellsim"))
        .arg("replay")
        .arg(&failing)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("assertion failed"));
    let record: serde_json::Value =
        serde_json::from_slice(output.stdout.strip_suffix(b"\n").unwrap_or(&output.stdout))
            .unwrap();
    assert_eq!(record["assertion"]["passed"], false);
    assert!(record["assertion"]["failures"][0]
        .as_str()
        .unwrap()
        .contains("exit_status"));
}
