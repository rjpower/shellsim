//! Persistent, replayable host-side sessions for agent harnesses.
//!
//! This module is a trusted adapter around [`Environment`], not a simulated capability. It
//! accepts explicit bytes, exposes only typed VFS operations and shell actions, and reports
//! stable workspace changes and telemetry. No method makes host files, processes, environment,
//! network, or clocks visible to simulated programs.

use std::collections::BTreeSet;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::net::NetworkRequest;
use crate::process::{ProcessRecord, ProcessStatus};
use crate::scheduler::{TaskState, WaitReason};
use crate::telemetry::{CommandTrust, InvocationEvent};
use crate::vfs::{Node, NodeKind, Vfs};
use crate::{Environment, Limits, RunOutcome};

const MAX_TRANSFER_RAW_BYTES: usize = 6 * 1024 * 1024;
const MAX_SESSION_FORK_BYTES: u64 = 96 * 1024 * 1024;

/// One protocol request. `id` is echoed verbatim so clients can correlate responses.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HarnessRequest {
    #[serde(default)]
    pub id: Option<serde_json::Value>,
    #[serde(flatten)]
    pub operation: HarnessOperation,
}

/// Closed set of persistent harness operations.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum HarnessOperation {
    Execute {
        source: String,
        #[serde(default)]
        stdin_base64: String,
    },
    ReadFile {
        path: String,
    },
    WriteFile {
        path: String,
        data_base64: String,
        #[serde(default = "default_file_mode")]
        mode: u32,
    },
    RemovePath {
        path: String,
    },
    ListPaths {
        #[serde(default = "default_root")]
        root: String,
    },
    Checkpoint,
    WorkspaceDiff,
    ResetWorkspace,
    Inspect,
}

/// One response emitted for exactly one request.
#[derive(Debug, Serialize)]
pub struct HarnessResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<HarnessResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Typed successful response payloads.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HarnessResult {
    Execute(ExecuteResult),
    File(FileResult),
    Paths { paths: Vec<String> },
    WorkspaceDiff { changes: Vec<WorkspaceChange> },
    Inspect(InspectResult),
    Acknowledged,
}

/// Action output and telemetry. Stream bytes use explicit base64 without UTF-8 loss.
#[derive(Debug, Serialize)]
pub struct ExecuteResult {
    pub outcome: RunOutcome,
    pub stdout_base64: String,
    pub stderr_base64: String,
    pub commands: Vec<String>,
    pub unsupported: Vec<String>,
    pub noop_commands: Vec<String>,
    pub partial_commands: Vec<String>,
    pub invocations: Vec<InvocationEvent>,
    pub dropped_invocations: u64,
    pub network_requests: Vec<NetworkRequest>,
    pub dropped_network_requests: u64,
}

/// Exact VFS file bytes and metadata.
#[derive(Debug, Serialize)]
pub struct FileResult {
    pub path: String,
    pub data_base64: String,
    pub mode: u32,
}

/// Stable process/resource inspection for an active session.
#[derive(Debug, Serialize)]
pub struct InspectResult {
    pub cwd: String,
    pub outcome: RunOutcome,
    pub processes: Vec<ProcessView>,
    pub current_pid: Option<u32>,
    pub monotonic_ns: u64,
    pub wall_time_ns: i128,
    pub pending_events: usize,
    pub ready_events: usize,
    pub invocations: Vec<InvocationEvent>,
    pub dropped_invocations: u64,
    pub network_requests: Vec<NetworkRequest>,
    pub dropped_network_requests: u64,
}

#[derive(Debug, Serialize)]
pub struct ProcessView {
    pub pid: u32,
    pub ppid: u32,
    pub process_group: u32,
    pub command: String,
    pub cwd: String,
    pub status: ProcessViewStatus,
}

/// Typed lifecycle state in an inspection response.
#[derive(Debug, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ProcessViewStatus {
    Runnable,
    Running,
    Blocked { reason: WaitReasonView },
    Exited { status: i32 },
}

/// Serializable scheduler wait reason without exposing implementation strings.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WaitReasonView {
    Timer { deadline_ns: u64 },
    PipeReadable { pipe: u32 },
    PipeWritable { pipe: u32 },
    Child { pid: u32 },
    ChildActivity { pid: u32 },
    ChildDeadline { pid: u32, deadline_ns: u64 },
    ChildActivityDeadline { pid: u32, deadline_ns: u64 },
}

/// One stable path-level difference from the last checkpoint.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct WorkspaceChange {
    pub path: String,
    pub change: ChangeKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before: Option<WorkspaceNode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after: Option<WorkspaceNode>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    Added,
    Modified,
    Deleted,
}

/// Serializable content identity used by workspace diffs.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkspaceNode {
    File { mode: u32, data_base64: String },
    Directory { mode: u32 },
    Symlink { mode: u32, target: String },
}

/// One persistent simulated environment and its VFS checkpoint.
pub struct HarnessSession {
    pub environment: Environment,
    baseline: Vfs,
}

impl HarnessSession {
    /// Create an empty `/work` session and checkpoint its initial filesystem.
    pub fn new(limits: Limits) -> Self {
        let mut environment = Environment::with_limits(limits);
        let _ = environment.vfs.put_dir("/work", 0o755);
        environment.cwd = "/work".to_string();
        environment.set_var("PWD", "/work");
        let baseline = environment.vfs.clone();
        Self {
            environment,
            baseline,
        }
    }

    /// Apply one request without broadening simulated capabilities.
    pub fn handle(&mut self, request: HarnessRequest) -> HarnessResponse {
        let id = request.id;
        match self.apply(request.operation) {
            Ok(result) => HarnessResponse {
                id,
                ok: true,
                result: Some(result),
                error: None,
            },
            Err(error) => HarnessResponse {
                id,
                ok: false,
                result: None,
                error: Some(error),
            },
        }
    }

    /// Replace the diff/reset baseline with the current bounded VFS state.
    pub fn checkpoint_workspace(&mut self) -> Result<(), String> {
        if self.environment.vfs.disk_used() > MAX_TRANSFER_RAW_BYTES as u64 {
            return Err(format!(
                "workspace exceeds the {MAX_TRANSFER_RAW_BYTES}-byte checkpoint limit"
            ));
        }
        self.baseline = self.environment.vfs.clone();
        Ok(())
    }

    /// Clone the complete deterministic machine state for a branching evaluation.
    ///
    /// Unlike a workspace checkpoint, a fork includes shell and Python state, descriptors and
    /// pipe contents, logical processes, scheduler queues, timers, virtual-network fixtures,
    /// telemetry, and resource counters. The estimate is checked before the host allocation;
    /// each returned session subsequently owns and enforces an independent copy of its limits.
    pub fn fork(&self) -> Result<Self, String> {
        let current_disk = self.environment.vfs.disk_used();
        let baseline_disk = self.baseline.disk_used();
        if current_disk > MAX_TRANSFER_RAW_BYTES as u64
            || baseline_disk > MAX_TRANSFER_RAW_BYTES as u64
        {
            return Err(format!(
                "workspace exceeds the {MAX_TRANSFER_RAW_BYTES}-byte session fork limit"
            ));
        }
        let usage = self
            .environment
            .resources
            .outcome(0, current_disk, self.environment.vfs.disk_peak())
            .usage;
        let estimate = current_disk
            .saturating_add(baseline_disk)
            .saturating_add(usage.memory_current)
            .saturating_add(self.environment.pending_stdout.len() as u64)
            .saturating_add(self.environment.pending_stderr.len() as u64)
            .saturating_add(self.environment.invocations.modeled_bytes())
            .saturating_add(1024 * 1024);
        if estimate > MAX_SESSION_FORK_BYTES {
            return Err(format!(
                "session exceeds the {MAX_SESSION_FORK_BYTES}-byte fork limit"
            ));
        }
        Ok(Self {
            environment: self.environment.clone(),
            baseline: self.baseline.clone(),
        })
    }

    fn apply(&mut self, operation: HarnessOperation) -> Result<HarnessResult, String> {
        match operation {
            HarnessOperation::Execute {
                source,
                stdin_base64,
            } => {
                let (stdin, reserved) = self.decode_bytes(&stdin_base64)?;
                let result = self.execute(&source, &stdin);
                self.environment.resources.release_memory(reserved);
                result
            }
            HarnessOperation::ReadFile { path } => self.read_file(&path),
            HarnessOperation::WriteFile {
                path,
                data_base64,
                mode,
            } => {
                let path = absolute_workspace_path(&path)?;
                let (data, reserved) = self.decode_bytes(&data_base64)?;
                let result = self
                    .environment
                    .vfs
                    .put_file(&path, data, mode)
                    .map_err(|error| error.to_string());
                self.environment.resources.release_memory(reserved);
                result?;
                Ok(HarnessResult::Acknowledged)
            }
            HarnessOperation::RemovePath { path } => {
                let path = absolute_workspace_path(&path)?;
                self.environment
                    .vfs
                    .remove_all("/", &path)
                    .map_err(|error| error.to_string())?;
                Ok(HarnessResult::Acknowledged)
            }
            HarnessOperation::ListPaths { root } => {
                let root = absolute_workspace_path(&root)?;
                if !self.environment.vfs.is_dir("/", &root) {
                    return Err(format!("not a directory: {root}"));
                }
                Ok(HarnessResult::Paths {
                    paths: self.environment.vfs.walk(&root),
                })
            }
            HarnessOperation::Checkpoint => {
                self.checkpoint_workspace()?;
                Ok(HarnessResult::Acknowledged)
            }
            HarnessOperation::WorkspaceDiff => Ok(HarnessResult::WorkspaceDiff {
                changes: workspace_diff(&self.baseline, &self.environment.vfs)?,
            }),
            HarnessOperation::ResetWorkspace => {
                self.environment.vfs = self.baseline.clone();
                if !self.environment.vfs.is_dir("/", &self.environment.cwd) {
                    self.environment.cwd = "/work".to_string();
                    self.environment.set_var("PWD", "/work");
                }
                Ok(HarnessResult::Acknowledged)
            }
            HarnessOperation::Inspect => Ok(HarnessResult::Inspect(self.inspect())),
        }
    }

    fn execute(&mut self, source: &str, stdin: &[u8]) -> Result<HarnessResult, String> {
        let command_start = self.environment.cmd_trace.len();
        let unsupported_start = self.environment.unsupported.len();
        let invocation_start = self.environment.invocations.next_sequence();
        let dropped_invocations_start = self.environment.invocations.dropped();
        let network_start = self.environment.net.log.len();
        let dropped_network_start = self.environment.net.dropped_requests;
        let (outcome, stdout, stderr) = self
            .environment
            .run_script_capture_with_stdin(source, stdin);
        let invocations = self.environment.invocations.events_since(invocation_start);
        let noop_commands = invocation_names(&invocations, CommandTrust::NoOp);
        let partial_commands = invocation_names(&invocations, CommandTrust::Partial);
        Ok(HarnessResult::Execute(ExecuteResult {
            outcome,
            stdout_base64: STANDARD.encode(stdout),
            stderr_base64: STANDARD.encode(stderr),
            commands: self.environment.cmd_trace[command_start..].to_vec(),
            unsupported: self.environment.unsupported[unsupported_start..].to_vec(),
            noop_commands,
            partial_commands,
            invocations,
            dropped_invocations: self
                .environment
                .invocations
                .dropped()
                .saturating_sub(dropped_invocations_start),
            network_requests: self.environment.net.log[network_start..].to_vec(),
            dropped_network_requests: self
                .environment
                .net
                .dropped_requests
                .saturating_sub(dropped_network_start),
        }))
    }

    fn read_file(&self, path: &str) -> Result<HarnessResult, String> {
        let path = absolute_workspace_path(path)?;
        let length = self
            .environment
            .vfs
            .file_len("/", &path)
            .map_err(|error| error.to_string())?;
        if length > MAX_TRANSFER_RAW_BYTES {
            return Err(format!(
                "file exceeds the {MAX_TRANSFER_RAW_BYTES}-byte transfer limit"
            ));
        }
        let node = self
            .environment
            .vfs
            .metadata("/", &path, true)
            .map_err(|error| error.to_string())?;
        let NodeKind::File(data) = node.kind else {
            return Err(format!("not a regular file: {path}"));
        };
        Ok(HarnessResult::File(FileResult {
            path,
            data_base64: STANDARD.encode(data),
            mode: node.mode,
        }))
    }

    fn inspect(&self) -> InspectResult {
        let wall_time_ns = self
            .environment
            .clock
            .wall_time_ns()
            .unwrap_or(crate::clock::DEFAULT_EPOCH_UTC_NS);
        InspectResult {
            cwd: self.environment.cwd.clone(),
            outcome: self.environment.resources.outcome(
                self.environment.last_status,
                self.environment.vfs.disk_used(),
                self.environment.vfs.disk_peak(),
            ),
            processes: self
                .environment
                .processes
                .iter()
                .map(|record| process_view(record, self.environment.scheduler.state(record.pid)))
                .collect(),
            current_pid: self.environment.scheduler.current(),
            monotonic_ns: self.environment.clock.monotonic_ns(),
            wall_time_ns,
            pending_events: self.environment.clock.pending_len(),
            ready_events: self.environment.clock.ready_len(),
            invocations: self.environment.invocations.events(),
            dropped_invocations: self.environment.invocations.dropped(),
            network_requests: self.environment.net.log.clone(),
            dropped_network_requests: self.environment.net.dropped_requests,
        }
    }

    fn decode_bytes(&mut self, value: &str) -> Result<(Vec<u8>, u64), String> {
        let bound = value
            .len()
            .checked_div(4)
            .and_then(|groups| groups.checked_mul(3))
            .and_then(|bytes| bytes.checked_add(3))
            .ok_or_else(|| "base64 payload is too large".to_string())?;
        if bound > MAX_TRANSFER_RAW_BYTES {
            return Err(format!(
                "payload exceeds the {MAX_TRANSFER_RAW_BYTES}-byte transfer limit"
            ));
        }
        let reserved = u64::try_from(bound).map_err(|_| "payload is too large".to_string())?;
        if !self.environment.resources.reserve_memory(reserved) {
            return Err("memory limit exceeded while decoding request bytes".to_string());
        }
        match STANDARD.decode(value) {
            Ok(bytes) => Ok((bytes, reserved)),
            Err(error) => {
                self.environment.resources.release_memory(reserved);
                Err(format!("invalid base64 data: {error}"))
            }
        }
    }
}

fn process_view(record: &ProcessRecord, task_state: Option<TaskState>) -> ProcessView {
    ProcessView {
        pid: record.pid,
        ppid: record.ppid,
        process_group: record.process_group,
        command: record.command.clone(),
        cwd: record.cwd.clone(),
        status: match (record.status, task_state) {
            (ProcessStatus::Exited(status), _) | (_, Some(TaskState::Exited(status))) => {
                ProcessViewStatus::Exited { status }
            }
            (_, Some(TaskState::Runnable)) => ProcessViewStatus::Runnable,
            (_, Some(TaskState::Running)) => ProcessViewStatus::Running,
            (_, Some(TaskState::Blocked(reason))) => ProcessViewStatus::Blocked {
                reason: wait_reason_view(reason),
            },
            (ProcessStatus::Running, None) => ProcessViewStatus::Running,
        },
    }
}

fn wait_reason_view(reason: WaitReason) -> WaitReasonView {
    match reason {
        WaitReason::Timer(deadline_ns) => WaitReasonView::Timer { deadline_ns },
        WaitReason::PipeReadable(pipe) => WaitReasonView::PipeReadable { pipe },
        WaitReason::PipeWritable(pipe) => WaitReasonView::PipeWritable { pipe },
        WaitReason::Child(pid) => WaitReasonView::Child { pid },
        WaitReason::ChildActivity(pid) => WaitReasonView::ChildActivity { pid },
        WaitReason::ChildDeadline(pid, deadline_ns) => {
            WaitReasonView::ChildDeadline { pid, deadline_ns }
        }
        WaitReason::ChildActivityDeadline(pid, deadline_ns) => {
            WaitReasonView::ChildActivityDeadline { pid, deadline_ns }
        }
    }
}

fn invocation_names(events: &[InvocationEvent], trust: CommandTrust) -> Vec<String> {
    events
        .iter()
        .filter(|event| event.trust == trust)
        .filter_map(|event| event.argv.first().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn workspace_diff(before: &Vfs, after: &Vfs) -> Result<Vec<WorkspaceChange>, String> {
    let paths = before
        .all_paths()
        .map(|(path, _)| path)
        .chain(after.all_paths().map(|(path, _)| path))
        .filter(|path| path.as_str() == "/work" || path.starts_with("/work/"))
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut transfer = 0usize;
    let mut changes = Vec::new();
    for path in paths {
        let old = before.raw_get(&path);
        let new = after.raw_get(&path);
        if nodes_equal(old, new) {
            continue;
        }
        transfer = transfer
            .checked_add(old.map_or(0, node_transfer_bytes))
            .and_then(|size| size.checked_add(new.map_or(0, node_transfer_bytes)))
            .ok_or_else(|| "workspace diff is too large".to_string())?;
        if transfer > MAX_TRANSFER_RAW_BYTES {
            return Err(format!(
                "workspace diff exceeds the {MAX_TRANSFER_RAW_BYTES}-byte transfer limit"
            ));
        }
        let change = match (old, new) {
            (None, Some(_)) => ChangeKind::Added,
            (Some(_), None) => ChangeKind::Deleted,
            (Some(_), Some(_)) => ChangeKind::Modified,
            (None, None) => continue,
        };
        changes.push(WorkspaceChange {
            path,
            change,
            before: old.map(workspace_node),
            after: new.map(workspace_node),
        });
    }
    Ok(changes)
}

fn nodes_equal(left: Option<&Node>, right: Option<&Node>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) if left.mode == right.mode => match (&left.kind, &right.kind) {
            (NodeKind::File(left), NodeKind::File(right)) => left == right,
            (NodeKind::Dir, NodeKind::Dir) => true,
            (NodeKind::Symlink(left), NodeKind::Symlink(right)) => left == right,
            _ => false,
        },
        _ => false,
    }
}

fn node_transfer_bytes(node: &Node) -> usize {
    match &node.kind {
        NodeKind::File(data) => data.len(),
        NodeKind::Dir => 0,
        NodeKind::Symlink(target) => target.len(),
    }
}

fn workspace_node(node: &Node) -> WorkspaceNode {
    match &node.kind {
        NodeKind::File(data) => WorkspaceNode::File {
            mode: node.mode,
            data_base64: STANDARD.encode(data),
        },
        NodeKind::Dir => WorkspaceNode::Directory { mode: node.mode },
        NodeKind::Symlink(target) => WorkspaceNode::Symlink {
            mode: node.mode,
            target: target.clone(),
        },
    }
}

fn absolute_workspace_path(path: &str) -> Result<String, String> {
    if path.contains('\0') {
        return Err("workspace path contains NUL".to_string());
    }
    let path = if path.starts_with('/') {
        crate::vfs::normalize(path)
    } else {
        crate::vfs::resolve_against("/work", path)
    };
    if path == "/work" || path.starts_with("/work/") {
        Ok(path)
    } else {
        Err("harness file operations are confined to /work".to_string())
    }
}

fn default_file_mode() -> u32 {
    0o644
}

fn default_root() -> String {
    "/work".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_forks_preserve_and_isolate_complete_machine_state() {
        let mut original = HarnessSession::new(Limits::default());
        original
            .environment
            .run_script_capture("X=parent; printf base > /work/value; sleep 2 &");
        original.checkpoint_workspace().unwrap();
        let mut branch = original.fork().unwrap();

        let (_, original_out, original_err) = original
            .environment
            .run_script_capture("wait; printf '%s:' \"$X\"; cat /work/value; date +%s");
        let (_, branch_out, branch_err) = branch.environment.run_script_capture(
            "X=branch; printf child > /work/value; wait; printf '%s:' \"$X\"; cat /work/value; date +%s",
        );
        assert!(
            original_err.is_empty(),
            "{}",
            String::from_utf8_lossy(&original_err)
        );
        assert!(
            branch_err.is_empty(),
            "{}",
            String::from_utf8_lossy(&branch_err)
        );
        let original_out = String::from_utf8_lossy(&original_out);
        let branch_out = String::from_utf8_lossy(&branch_out);
        assert!(original_out.starts_with("parent:base"), "{original_out}");
        assert!(branch_out.starts_with("branch:child"), "{branch_out}");
        assert_eq!(
            original.environment.clock.monotonic_ns(),
            branch.environment.clock.monotonic_ns()
        );
        assert!(
            workspace_diff(&original.baseline, &original.environment.vfs)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            workspace_diff(&branch.baseline, &branch.environment.vfs)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn session_forks_deep_clone_python_heaps_and_reject_large_workspaces() {
        let mut original = HarnessSession::new(Limits::default());
        original.environment.run_script_capture("python");
        original.environment.run_script_capture("items = [1]");
        let mut branch = original.fork().unwrap();
        branch.environment.run_script_capture("items.append(2)");
        let (_, original_out, _) = original.environment.run_script_capture("print(items)");
        let (_, branch_out, _) = branch.environment.run_script_capture("print(items)");
        assert_eq!(String::from_utf8_lossy(&original_out), "[1]\n>>> ");
        assert_eq!(String::from_utf8_lossy(&branch_out), "[1, 2]\n>>> ");

        let mut oversized = HarnessSession::new(Limits::default());
        oversized
            .environment
            .vfs
            .put_file("/work/large", vec![0; MAX_TRANSFER_RAW_BYTES + 1], 0o644)
            .unwrap();
        let error = match oversized.fork() {
            Ok(_) => panic!("oversized session fork unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.contains("session fork limit"), "{error}");
    }

    #[test]
    fn execute_reports_each_partial_command_in_its_own_action() {
        let mut session = HarnessSession::new(Limits::default());
        for _ in 0..2 {
            let HarnessResult::Execute(result) = session
                .execute("sed 's/a/b/' /missing", &[])
                .expect("execute action")
            else {
                panic!("execute returned the wrong result kind");
            };
            assert_eq!(result.partial_commands, ["sed"]);
            assert_eq!(result.invocations.len(), 1);
            assert_eq!(result.invocations[0].trust, CommandTrust::Partial);
            assert!(result.invocations[0].status.is_some());
        }
    }

    #[test]
    fn inspect_uses_scheduler_state_and_typed_wait_reasons() {
        let mut session = HarnessSession::new(Limits::default());
        session.environment.run_script_capture("sleep 10 &");
        session.environment.run_script_capture("true");

        let inspect = session.inspect();
        assert_eq!(inspect.current_pid, Some(1_234));
        assert_eq!(inspect.pending_events, 1);
        assert!(inspect.processes.iter().any(|process| matches!(
            process.status,
            ProcessViewStatus::Blocked {
                reason: WaitReasonView::Timer {
                    deadline_ns: 10_000_000_000
                }
            }
        )));
        assert!(inspect
            .invocations
            .iter()
            .any(|event| event.argv == ["sleep", "10"] && event.status.is_none()));
    }
}
