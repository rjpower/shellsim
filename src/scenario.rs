//! Versioned replay scenarios and their deterministic evaluation policy.
//!
//! This module keeps scenario assertions separate from execution. The harness reports modeled
//! behavior and trust; a scenario can elect to reject partial fidelity without changing command,
//! scheduler, or VM semantics. Assertions inspect only typed harness responses and never acquire
//! host capabilities.

use serde::{Deserialize, Serialize};

use crate::harness::{
    ActionState, HarnessOperation, HarnessRequest, HarnessResponse, HarnessResult,
    ProcessViewStatus,
};
use crate::harness_manager::HarnessManager;
use crate::CommandTrust;

/// The only scenario format understood by this build.
pub const FORMAT_VERSION: u32 = 1;

/// Optional assertion attached to one scenario action.
#[derive(Debug, Deserialize, Serialize)]
pub struct Expectation {
    #[serde(default)]
    pub ok: Option<bool>,
    #[serde(default)]
    pub exit_status: Option<i32>,
    #[serde(default)]
    pub stdout_base64: Option<String>,
    #[serde(default)]
    pub stderr_base64: Option<String>,
    #[serde(default)]
    pub unsupported: Option<Vec<String>>,
    #[serde(default)]
    pub noop_commands: Option<Vec<String>>,
    #[serde(default)]
    pub partial_commands: Option<Vec<String>>,
    #[serde(default)]
    pub workspace_change_count: Option<usize>,
    #[serde(default)]
    pub error_contains: Option<String>,
}

/// Result of checking an action or final-state expectation.
#[derive(Debug, Serialize)]
pub struct Assertion {
    pub passed: bool,
    pub failures: Vec<String>,
}

/// Version and policy selected by the first line of a versioned scenario.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub version: u32,
    #[serde(default)]
    pub strict: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_expectation: Option<FinalExpectation>,
}

/// Assertions evaluated against one session after every action has run.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FinalExpectation {
    #[serde(default)]
    pub session_id: u64,
    #[serde(default)]
    pub workspace_change_count: Option<usize>,
    #[serde(default)]
    pub exit_status: Option<i32>,
    #[serde(default)]
    pub active_action_count: Option<usize>,
    #[serde(default)]
    pub live_process_count: Option<usize>,
    #[serde(default)]
    pub cpu_used_at_most: Option<u64>,
    #[serde(default)]
    pub memory_peak_at_most: Option<u64>,
    #[serde(default)]
    pub disk_current_at_most: Option<u64>,
    #[serde(default)]
    pub output_bytes_at_most: Option<u64>,
}

/// Required outer shape of a scenario metadata line.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub scenario: Metadata,
}

/// One legacy request or a request paired with an expectation.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum Action {
    Request(HarnessRequest),
    Asserted {
        request: HarnessRequest,
        expect: Expectation,
    },
}

#[derive(Serialize)]
struct HeaderRecord<'a> {
    kind: &'static str,
    scenario: &'a Metadata,
}

#[derive(Serialize)]
struct ActionRecord<'a> {
    sequence: usize,
    request: &'a HarnessRequest,
    response: &'a HarnessResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    expectation: Option<&'a Expectation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    assertion: Option<&'a Assertion>,
}

#[derive(Serialize)]
struct FinalRecord<'a> {
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    expectation: Option<&'a FinalExpectation>,
    assertion: &'a Assertion,
}

/// Serialize the canonical transcript representation of a scenario header.
pub fn serialize_header(metadata: &Metadata) -> Vec<u8> {
    serde_json::to_vec(&HeaderRecord {
        kind: "scenario",
        scenario: metadata,
    })
    .expect("scenario metadata is serializable")
}

/// Serialize one canonical paired action record.
pub fn serialize_action(
    sequence: usize,
    request: &HarnessRequest,
    response: &HarnessResponse,
    expectation: Option<&Expectation>,
    assertion: Option<&Assertion>,
) -> Vec<u8> {
    serde_json::to_vec(&ActionRecord {
        sequence,
        request,
        response,
        expectation,
        assertion,
    })
    .expect("scenario action is serializable")
}

/// Serialize the canonical final assertion record.
pub fn serialize_final(expectation: Option<&FinalExpectation>, assertion: &Assertion) -> Vec<u8> {
    serde_json::to_vec(&FinalRecord {
        kind: "final",
        expectation,
        assertion,
    })
    .expect("scenario final assertion is serializable")
}

/// Check one explicit action expectation against its typed response.
pub fn check_expectation(expected: &Expectation, response: &HarnessResponse) -> Assertion {
    let mut failures = Vec::new();
    if let Some(ok) = expected.ok {
        if response.ok != ok {
            failures.push(format!("expected ok={ok}, got {}", response.ok));
        }
    }
    let execute = match response.result.as_ref() {
        Some(HarnessResult::Execute(result)) => Some(result),
        _ => None,
    };
    if let Some(status) = expected.exit_status {
        match execute {
            Some(result) if result.outcome.exit_status == status => {}
            Some(result) => failures.push(format!(
                "expected exit_status={status}, got {}",
                result.outcome.exit_status
            )),
            None => failures.push("exit_status requires an execute result".to_string()),
        }
    }
    for (name, expected_value, actual) in [
        (
            "stdout_base64",
            expected.stdout_base64.as_ref(),
            execute.map(|result| &result.stdout_base64),
        ),
        (
            "stderr_base64",
            expected.stderr_base64.as_ref(),
            execute.map(|result| &result.stderr_base64),
        ),
    ] {
        if let Some(expected_value) = expected_value {
            match actual {
                Some(actual) if actual == expected_value => {}
                Some(actual) => failures.push(format!(
                    "expected {name}={expected_value:?}, got {actual:?}"
                )),
                None => failures.push(format!("{name} requires an execute result")),
            }
        }
    }
    for (name, expected_values, actual) in [
        (
            "unsupported",
            expected.unsupported.as_ref(),
            execute.map(|result| &result.unsupported),
        ),
        (
            "noop_commands",
            expected.noop_commands.as_ref(),
            execute.map(|result| &result.noop_commands),
        ),
        (
            "partial_commands",
            expected.partial_commands.as_ref(),
            execute.map(|result| &result.partial_commands),
        ),
    ] {
        if let Some(expected_values) = expected_values {
            match actual {
                Some(actual) if actual == expected_values => {}
                Some(actual) => failures.push(format!(
                    "expected {name}={expected_values:?}, got {actual:?}"
                )),
                None => failures.push(format!("{name} requires an execute result")),
            }
        }
    }
    if let Some(expected_count) = expected.workspace_change_count {
        match response.result.as_ref() {
            Some(HarnessResult::WorkspaceDiff { changes }) if changes.len() == expected_count => {}
            Some(HarnessResult::WorkspaceDiff { changes }) => failures.push(format!(
                "expected workspace_change_count={expected_count}, got {}",
                changes.len()
            )),
            _ => {
                failures.push("workspace_change_count requires a workspace_diff result".to_string())
            }
        }
    }
    if let Some(fragment) = &expected.error_contains {
        match response.error.as_ref() {
            Some(error) if error.contains(fragment) => {}
            Some(error) => failures.push(format!(
                "expected error containing {fragment:?}, got {error:?}"
            )),
            None => failures.push(format!("expected error containing {fragment:?}, got none")),
        }
    }
    Assertion {
        passed: failures.is_empty(),
        failures,
    }
}

/// Return strict-policy failures observable in one action response.
pub fn strict_response_failures(sequence: usize, response: &HarnessResponse) -> Vec<String> {
    let prefix = format!("action {}", sequence + 1);
    let mut failures = Vec::new();
    if !response.ok {
        failures.push(format!("{prefix} returned an error"));
        return failures;
    }
    let (invocations, dropped_invocations, unsupported, network, dropped_network) =
        match response.result.as_ref() {
            Some(HarnessResult::Execute(result)) => (
                result.invocations.as_slice(),
                result.dropped_invocations,
                result.unsupported.as_slice(),
                result.network_requests.as_slice(),
                result.dropped_network_requests,
            ),
            Some(HarnessResult::Action(result)) => (
                result.invocations.as_slice(),
                result.dropped_invocations,
                result.unsupported.as_slice(),
                result.network_requests.as_slice(),
                result.dropped_network_requests,
            ),
            _ => return failures,
        };
    if !unsupported.is_empty() {
        failures.push(format!("{prefix} used unsupported behavior"));
    }
    if invocations
        .iter()
        .any(|event| event.trust != CommandTrust::Real)
    {
        failures.push(format!("{prefix} used a partial or no-op command"));
    }
    if network.iter().any(|request| !request.matched) {
        failures.push(format!("{prefix} made an unmatched network request"));
    }
    if dropped_invocations != 0 || dropped_network != 0 {
        failures.push(format!(
            "{prefix} exceeded an observability retention bound"
        ));
    }
    failures
}

/// Evaluate final state and resource assertions through the typed manager API.
pub fn check_final_expectation(
    manager: &mut HarnessManager,
    expected: &FinalExpectation,
    strict: bool,
    mut failures: Vec<String>,
) -> Assertion {
    let response = manager.handle(HarnessRequest {
        id: None,
        session_id: Some(expected.session_id),
        operation: HarnessOperation::Inspect,
    });
    let inspect = match response.result {
        Some(HarnessResult::Inspect(inspect)) => inspect,
        _ => {
            failures.push(format!(
                "cannot inspect final session {}: {}",
                expected.session_id,
                response
                    .error
                    .unwrap_or_else(|| "unexpected response".to_string())
            ));
            return Assertion {
                passed: false,
                failures,
            };
        }
    };

    if let Some(value) = expected.exit_status {
        if inspect.outcome.exit_status != value {
            failures.push(format!(
                "expected final exit_status={value}, got {}",
                inspect.outcome.exit_status
            ));
        }
    }
    let active_actions = inspect
        .actions
        .iter()
        .filter(|action| !matches!(action.state, ActionState::Complete { .. }))
        .count();
    if strict && active_actions != 0 {
        failures.push(format!(
            "strict scenario ended with {active_actions} active action(s)"
        ));
    }
    if let Some(value) = expected.active_action_count {
        if active_actions != value {
            failures.push(format!(
                "expected active_action_count={value}, got {active_actions}"
            ));
        }
    }
    let live_processes = inspect
        .processes
        .iter()
        .filter(|process| !matches!(process.status, ProcessViewStatus::Exited { .. }))
        .count();
    if let Some(value) = expected.live_process_count {
        if live_processes != value {
            failures.push(format!(
                "expected live_process_count={value}, got {live_processes}"
            ));
        }
    }
    for (name, actual, limit) in [
        (
            "cpu_used",
            inspect.outcome.usage.cpu_used,
            expected.cpu_used_at_most,
        ),
        (
            "memory_peak",
            inspect.outcome.usage.memory_peak,
            expected.memory_peak_at_most,
        ),
        (
            "disk_current",
            inspect.outcome.usage.disk_current,
            expected.disk_current_at_most,
        ),
        (
            "output_bytes",
            inspect.outcome.usage.output_bytes,
            expected.output_bytes_at_most,
        ),
    ] {
        if let Some(limit) = limit {
            if actual > limit {
                failures.push(format!("expected {name}<={limit}, got {actual}"));
            }
        }
    }
    if let Some(value) = expected.workspace_change_count {
        let workspace = manager.handle(HarnessRequest {
            id: None,
            session_id: Some(expected.session_id),
            operation: HarnessOperation::WorkspaceDiff,
        });
        match workspace.result {
            Some(HarnessResult::WorkspaceDiff { changes }) if changes.len() == value => {}
            Some(HarnessResult::WorkspaceDiff { changes }) => failures.push(format!(
                "expected final workspace_change_count={value}, got {}",
                changes.len()
            )),
            _ => failures.push("cannot inspect final workspace changes".to_string()),
        }
    }
    if strict {
        if inspect.dropped_invocations != 0 || inspect.dropped_network_requests != 0 {
            failures.push("strict scenario exceeded an observability retention bound".to_string());
        }
        if inspect
            .network_requests
            .iter()
            .any(|request| !request.matched)
        {
            failures.push("strict scenario made an unmatched network request".to_string());
        }
        if inspect.actions.iter().any(|action| {
            !action.unsupported.is_empty()
                || action.dropped_invocations != 0
                || action.dropped_network_requests != 0
                || action
                    .invocations
                    .iter()
                    .any(|event| event.trust != CommandTrust::Real)
                || action
                    .network_requests
                    .iter()
                    .any(|request| !request.matched)
        }) {
            failures.push("strict scenario contains an untrusted retained action".to_string());
        }
    }
    failures.sort();
    failures.dedup();
    Assertion {
        passed: failures.is_empty(),
        failures,
    }
}
