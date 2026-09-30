//! Exercise the typed host-tool exchange, cancellation, and ungranted virtual route.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{json, Value};
use shellsim::harness::{HarnessOperation, HarnessRequest, HarnessSession};
use shellsim::realtime::ClockMode;
use shellsim::Limits;

fn handle(session: &mut HarnessSession, operation: HarnessOperation) -> Value {
    let response = session.handle(HarnessRequest {
        id: None,
        session_id: None,
        operation,
    });
    assert!(response.ok, "harness error: {:?}", response.error);
    serde_json::to_value(response).unwrap()
}

fn start_call(session: &mut HarnessSession, command_prefix: &str) -> u64 {
    let guest = r#"import json
from urllib.request import Request, urlopen
request = Request("http://host.shellsim/tools", data=json.dumps({"tool": "workspace.read_file", "arguments": {"path": "/work/answer.txt"}}).encode(), headers={"Content-Type": "application/json"})
with urlopen(request) as response:
    print(json.loads(response.read().decode())["result"]["text"])
"#;
    let started = handle(
        session,
        HarnessOperation::StartExecute {
            source: format!("{command_prefix}python3.14 -c '{guest}'"),
            stdin_base64: String::new(),
            stdin_closed: true,
        },
    );
    started["result"]["action_id"].as_u64().unwrap()
}

fn poll_with_time(session: &mut HarnessSession, action_id: u64, advance_time: bool) -> Value {
    handle(
        session,
        HarnessOperation::PollAction {
            action_id,
            work_quanta: 100_000,
            advance_time,
        },
    )
}

fn poll(session: &mut HarnessSession, action_id: u64) -> Value {
    poll_with_time(session, action_id, false)
}

#[test]
fn python_guest_calls_host_tool_without_host_network() {
    let mut session =
        HarnessSession::with_clock_and_host_tools(Limits::default(), ClockMode::Virtual);
    let action_id = start_call(&mut session, "");
    let blocked = poll(&mut session, action_id);
    assert_eq!(blocked["result"]["state"]["state"], "blocked");
    let calls = blocked["result"]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
    assert_eq!(call["tool"], "workspace.read_file");
    assert_eq!(call["arguments"], json!({"path":"/work/answer.txt"}));
    assert_eq!(
        blocked["result"]["state"]["reason"]["request_id"],
        call["request_id"]
    );
    let request_id = call["request_id"].as_u64().unwrap();
    assert!(poll(&mut session, action_id)["result"]["tool_calls"]
        .as_array()
        .unwrap()
        .is_empty());
    handle(
        &mut session,
        HarnessOperation::RespondToolCall {
            request_id,
            result: Some(json!({"text":"42"})),
            error: None,
        },
    );
    let duplicate = session.handle(HarnessRequest {
        id: None,
        session_id: None,
        operation: HarnessOperation::RespondToolCall {
            request_id,
            result: Some(json!({"text":"wrong"})),
            error: None,
        },
    });
    assert!(!duplicate.ok);
    let finished = poll(&mut session, action_id);
    assert_eq!(
        finished["result"]["state"]["state"], "complete",
        "{finished}"
    );
    assert_eq!(finished["result"]["state"]["status"], 0);
    let output = handle(
        &mut session,
        HarnessOperation::ReadActionOutput { action_id },
    );
    assert_eq!(
        STANDARD
            .decode(output["result"]["stdout_base64"].as_str().unwrap())
            .unwrap(),
        b"42\n"
    );
}

#[test]
fn cancelling_guest_rejects_late_host_completion() {
    let mut session =
        HarnessSession::with_clock_and_host_tools(Limits::default(), ClockMode::Virtual);
    let action_id = start_call(&mut session, "");
    let blocked = poll(&mut session, action_id);
    let request_id = blocked["result"]["tool_calls"][0]["request_id"]
        .as_u64()
        .unwrap();
    handle(&mut session, HarnessOperation::CancelAction { action_id });
    let late = session.handle(HarnessRequest {
        id: None,
        session_id: None,
        operation: HarnessOperation::RespondToolCall {
            request_id,
            result: Some(json!({})),
            error: None,
        },
    });
    assert!(!late.ok);
}

#[test]
fn virtual_deadline_waits_for_host_reply() {
    let mut session =
        HarnessSession::with_clock_and_host_tools(Limits::default(), ClockMode::Virtual);
    let action_id = start_call(&mut session, "timeout 5 ");
    let blocked = poll_with_time(&mut session, action_id, true);
    assert_eq!(blocked["result"]["state"]["state"], "blocked");
    let request_id = blocked["result"]["tool_calls"][0]["request_id"]
        .as_u64()
        .unwrap();
    handle(
        &mut session,
        HarnessOperation::RespondToolCall {
            request_id,
            result: Some(json!({"text": "42"})),
            error: None,
        },
    );
    let finished = poll_with_time(&mut session, action_id, true);
    assert_eq!(finished["result"]["state"]["status"], 0, "{finished}");
}

#[test]
fn granted_session_rejects_synchronous_execution_and_fork() {
    let mut session =
        HarnessSession::with_clock_and_host_tools(Limits::default(), ClockMode::Virtual);
    let synchronous = session.handle(HarnessRequest {
        id: None,
        session_id: None,
        operation: HarnessOperation::Execute {
            source: "printf unreachable".into(),
            stdin_base64: String::new(),
        },
    });
    assert!(!synchronous.ok);
    assert!(session.fork().is_err());

    let invalid_reply = session.handle(HarnessRequest {
        id: None,
        session_id: None,
        operation: HarnessOperation::RespondToolCall {
            request_id: 0,
            result: Some(json!(42)),
            error: None,
        },
    });
    assert!(!invalid_reply.ok);
}

#[test]
fn ungranted_guest_cannot_call_host_tools() {
    let mut session = HarnessSession::new(Limits::default());
    let result = handle(
        &mut session,
        HarnessOperation::Execute {
            source: "python3.14 -c 'from urllib.request import Request, urlopen; from urllib.error import URLError;\ntry:\n    urlopen(Request(\"http://host.shellsim/tools\", data=b\"{}\"))\nexcept URLError:\n    print(\"ungranted\")'".into(),
            stdin_base64: String::new(),
        },
    );
    assert_eq!(result["result"]["outcome"]["exit_status"], 0);
    assert_eq!(
        STANDARD
            .decode(result["result"]["stdout_base64"].as_str().unwrap())
            .unwrap(),
        b"ungranted\n"
    );
}
