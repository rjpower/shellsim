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
const MAX_RETAINED_ACTIONS: usize = 64;
const MAX_POLL_QUANTA: usize = 100_000;
const MAX_CANCEL_QUANTA: usize = 100_000;

/// One protocol request. `id` is echoed verbatim so clients can correlate responses.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HarnessRequest {
    #[serde(default)]
    pub id: Option<serde_json::Value>,
    /// Manager-owned session route. Omitted requests target compatibility session zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<u64>,
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
    StartExecute {
        source: String,
        #[serde(default)]
        stdin_base64: String,
        #[serde(default = "default_true")]
        stdin_closed: bool,
    },
    PollAction {
        action_id: u64,
        #[serde(default = "default_poll_quanta")]
        work_quanta: usize,
        #[serde(default)]
        advance_time: bool,
    },
    WriteStdin {
        action_id: u64,
        data_base64: String,
    },
    CloseStdin {
        action_id: u64,
    },
    ReadActionOutput {
        action_id: u64,
    },
    SignalProcess {
        pid: u32,
        signal: String,
        #[serde(default)]
        process_group: bool,
    },
    CancelAction {
        action_id: u64,
    },
    DropAction {
        action_id: u64,
    },
    ForkSession {
        source: u64,
    },
    DropSession {
        target: u64,
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
    StatPath {
        path: String,
        #[serde(default = "default_true")]
        follow_symlinks: bool,
    },
    MakeDirectory {
        path: String,
        #[serde(default = "default_directory_mode")]
        mode: u32,
        #[serde(default)]
        parents: bool,
    },
    CreateSymlink {
        path: String,
        target: String,
    },
    ApplyPatch {
        patch: String,
        #[serde(default = "default_patch_strip")]
        strip: usize,
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
    Action(ActionView),
    ActionOutput(ActionOutput),
    Session { session_id: u64 },
    File(FileResult),
    PathMetadata(PathMetadata),
    Paths { paths: Vec<String> },
    WorkspaceDiff { changes: Vec<WorkspaceChange> },
    Inspect(InspectResult),
    Acknowledged,
}

/// Stable lifecycle and telemetry for one retained foreground action.
#[derive(Debug, Serialize)]
pub struct ActionView {
    pub action_id: u64,
    pub root_pid: u32,
    pub state: ActionState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<RunOutcome>,
    pub invocations: Vec<InvocationEvent>,
    pub dropped_invocations: u64,
    pub unsupported: Vec<String>,
    pub network_requests: Vec<NetworkRequest>,
    pub dropped_network_requests: u64,
}

/// Retained action lifecycle. A blocked action can be resumed by input, a modeled event, or a
/// later poll that permits virtual-time advancement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ActionState {
    Running,
    Blocked { reason: Option<WaitReasonView> },
    Complete { status: i32 },
}

/// Newly available action stream bytes. Reads advance action-local delivery cursors.
#[derive(Debug, Serialize)]
pub struct ActionOutput {
    pub action_id: u64,
    pub stdout_base64: String,
    pub stderr_base64: String,
    pub stdout_closed: bool,
    pub stderr_closed: bool,
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

/// Exact VFS metadata for one workspace path.
#[derive(Debug, Serialize)]
pub struct PathMetadata {
    pub path: String,
    pub node_type: PathType,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub mtime_ms: u64,
    pub size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symlink_target: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PathType {
    File,
    Directory,
    Symlink,
}

/// Stable process/resource inspection for an active session.
#[derive(Debug, Serialize)]
pub struct InspectResult {
    pub cwd: String,
    pub outcome: RunOutcome,
    pub actions: Vec<ActionView>,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WaitReasonView {
    Timer { deadline_ns: u64 },
    InputReadable { description: u32 },
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
    actions: std::collections::BTreeMap<u64, RetainedAction>,
    active_action: Option<u64>,
    next_action_id: u64,
}

#[derive(Clone)]
struct RetainedAction {
    id: u64,
    root_pid: u32,
    execution: Option<crate::exec::ShellExecution>,
    state: ActionState,
    outcome: Option<RunOutcome>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    stdout_cursor: usize,
    stderr_cursor: usize,
    invocation_start: u64,
    dropped_invocations_start: u64,
    dropped_invocations: u64,
    invocations: Vec<InvocationEvent>,
    unsupported: Vec<String>,
    network_requests: Vec<NetworkRequest>,
    dropped_network_requests: u64,
    command_start: usize,
    unsupported_start: usize,
    network_start: usize,
    dropped_network_start: u64,
}

impl RetainedAction {
    fn new(id: u64, environment: &Environment) -> Self {
        Self {
            id,
            root_pid: environment.process.pid,
            execution: None,
            state: ActionState::Running,
            outcome: None,
            stdout: Vec::new(),
            stderr: Vec::new(),
            stdout_cursor: 0,
            stderr_cursor: 0,
            invocation_start: environment.invocations.next_sequence(),
            dropped_invocations_start: environment.invocations.dropped(),
            dropped_invocations: 0,
            invocations: Vec::new(),
            unsupported: Vec::new(),
            network_requests: Vec::new(),
            dropped_network_requests: 0,
            command_start: environment.cmd_trace.len(),
            unsupported_start: environment.unsupported.len(),
            network_start: environment.net.log.len(),
            dropped_network_start: environment.net.dropped_requests,
        }
    }

    fn complete(&mut self, environment: &Environment, status: i32) {
        let status = environment.termination_status().unwrap_or(status);
        self.state = ActionState::Complete { status };
        self.outcome = Some(environment.outcome(status));
        self.invocations = environment.invocations.events_since(self.invocation_start);
        self.dropped_invocations = environment
            .invocations
            .dropped()
            .saturating_sub(self.dropped_invocations_start);
        self.unsupported = environment.unsupported[self.unsupported_start..].to_vec();
        self.network_requests = environment.net.log[self.network_start..].to_vec();
        self.dropped_network_requests = environment
            .net
            .dropped_requests
            .saturating_sub(self.dropped_network_start);
    }

    fn view(&self, environment: &Environment) -> ActionView {
        ActionView {
            action_id: self.id,
            root_pid: self.root_pid,
            state: self.state,
            outcome: self.outcome.clone(),
            invocations: if self.execution.is_some() {
                environment.invocations.events_since(self.invocation_start)
            } else {
                self.invocations.clone()
            },
            dropped_invocations: if self.execution.is_some() {
                environment
                    .invocations
                    .dropped()
                    .saturating_sub(self.dropped_invocations_start)
            } else {
                self.dropped_invocations
            },
            unsupported: if self.execution.is_some() {
                environment.unsupported[self.unsupported_start..].to_vec()
            } else {
                self.unsupported.clone()
            },
            network_requests: if self.execution.is_some() {
                environment.net.log[self.network_start..].to_vec()
            } else {
                self.network_requests.clone()
            },
            dropped_network_requests: if self.execution.is_some() {
                environment
                    .net
                    .dropped_requests
                    .saturating_sub(self.dropped_network_start)
            } else {
                self.dropped_network_requests
            },
        }
    }
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
            actions: std::collections::BTreeMap::new(),
            active_action: None,
            next_action_id: 0,
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
            .saturating_add(self.actions.values().fold(0_u64, |bytes, action| {
                bytes
                    .saturating_add(action.stdout.len() as u64)
                    .saturating_add(action.stderr.len() as u64)
                    .saturating_add(512)
            }))
            .saturating_add(1024 * 1024);
        if estimate > MAX_SESSION_FORK_BYTES {
            return Err(format!(
                "session exceeds the {MAX_SESSION_FORK_BYTES}-byte fork limit"
            ));
        }
        Ok(Self {
            environment: self.environment.clone(),
            baseline: self.baseline.clone(),
            actions: self.actions.clone(),
            active_action: self.active_action,
            next_action_id: self.next_action_id,
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
            HarnessOperation::StartExecute {
                source,
                stdin_base64,
                stdin_closed,
            } => {
                let (stdin, reserved) = self.decode_bytes(&stdin_base64)?;
                let result = self.start_action(&source, &stdin, stdin_closed);
                self.environment.resources.release_memory(reserved);
                result.map(HarnessResult::Action)
            }
            HarnessOperation::PollAction {
                action_id,
                work_quanta,
                advance_time,
            } => self
                .poll_action(action_id, work_quanta, advance_time)
                .map(HarnessResult::Action),
            HarnessOperation::WriteStdin {
                action_id,
                data_base64,
            } => {
                let (bytes, reserved) = self.decode_bytes(&data_base64)?;
                let result = self.write_action_stdin(action_id, &bytes);
                self.environment.resources.release_memory(reserved);
                result?;
                Ok(HarnessResult::Acknowledged)
            }
            HarnessOperation::CloseStdin { action_id } => {
                self.close_action_stdin(action_id)?;
                Ok(HarnessResult::Acknowledged)
            }
            HarnessOperation::ReadActionOutput { action_id } => self
                .read_action_output(action_id)
                .map(HarnessResult::ActionOutput),
            HarnessOperation::SignalProcess {
                pid,
                signal,
                process_group,
            } => {
                let signal = crate::process::Signal::parse(&signal)
                    .ok_or_else(|| format!("unsupported signal '{signal}'"))?;
                if process_group {
                    self.environment.send_signal_group(pid, signal)?;
                } else {
                    self.environment.send_signal(pid, signal)?;
                }
                Ok(HarnessResult::Acknowledged)
            }
            HarnessOperation::CancelAction { action_id } => {
                self.cancel_action(action_id).map(HarnessResult::Action)
            }
            HarnessOperation::DropAction { action_id } => {
                self.drop_action(action_id)?;
                Ok(HarnessResult::Acknowledged)
            }
            HarnessOperation::ForkSession { .. } | HarnessOperation::DropSession { .. } => {
                Err("session lifecycle operations require a HarnessManager".to_string())
            }
            HarnessOperation::ReadFile { path } => self.read_file(&path),
            HarnessOperation::WriteFile {
                path,
                data_base64,
                mode,
            } => {
                validate_mode(mode)?;
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
            HarnessOperation::StatPath {
                path,
                follow_symlinks,
            } => self.stat_path(&path, follow_symlinks),
            HarnessOperation::MakeDirectory {
                path,
                mode,
                parents,
            } => {
                validate_mode(mode)?;
                let path = absolute_workspace_path(&path)?;
                self.environment.sync_vfs_time();
                if parents {
                    self.environment
                        .vfs
                        .put_dir(&path, mode)
                        .map_err(|error| error.to_string())?;
                } else {
                    self.environment
                        .vfs
                        .mkdir("/", &path)
                        .and_then(|_| self.environment.vfs.chmod("/", &path, mode))
                        .map_err(|error| error.to_string())?;
                }
                Ok(HarnessResult::Acknowledged)
            }
            HarnessOperation::CreateSymlink { path, target } => {
                if target.contains('\0') {
                    return Err("symlink target contains NUL".to_string());
                }
                if target.len() > MAX_TRANSFER_RAW_BYTES {
                    return Err("symlink target exceeds the transfer limit".to_string());
                }
                let path = absolute_workspace_path(&path)?;
                self.environment.sync_vfs_time();
                self.environment
                    .vfs
                    .symlink("/", &target, &path)
                    .map_err(|error| error.to_string())?;
                Ok(HarnessResult::Acknowledged)
            }
            HarnessOperation::ApplyPatch { patch, strip } => {
                crate::commands::patch::apply_harness_patch(&mut self.environment, &patch, strip)?;
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
        let action_id = self.start_action(source, stdin, true)?.action_id;
        while self.active_action == Some(action_id) {
            self.poll_action(action_id, MAX_POLL_QUANTA, true)?;
        }
        let action = self
            .actions
            .remove(&action_id)
            .ok_or_else(|| format!("action {action_id} disappeared before completion"))?;
        let outcome = action
            .outcome
            .clone()
            .ok_or_else(|| format!("action {action_id} completed without an outcome"))?;
        let noop_commands = invocation_names(&action.invocations, CommandTrust::NoOp);
        let partial_commands = invocation_names(&action.invocations, CommandTrust::Partial);
        Ok(HarnessResult::Execute(ExecuteResult {
            outcome,
            stdout_base64: STANDARD.encode(action.stdout),
            stderr_base64: STANDARD.encode(action.stderr),
            commands: self.environment.cmd_trace[action.command_start..].to_vec(),
            unsupported: action.unsupported,
            noop_commands,
            partial_commands,
            invocations: action.invocations,
            dropped_invocations: action.dropped_invocations,
            network_requests: action.network_requests,
            dropped_network_requests: action.dropped_network_requests,
        }))
    }

    fn start_action(
        &mut self,
        source: &str,
        stdin: &[u8],
        stdin_closed: bool,
    ) -> Result<ActionView, String> {
        if self.active_action.is_some() {
            return Err("session already has an active foreground action".to_string());
        }
        if self.actions.len() >= MAX_RETAINED_ACTIONS {
            return Err(format!(
                "session retained-action limit exceeded ({MAX_RETAINED_ACTIONS})"
            ));
        }
        let action_id = self.next_action_id;
        self.next_action_id = self
            .next_action_id
            .checked_add(1)
            .ok_or_else(|| "action identifier space exhausted".to_string())?;
        let mut action = RetainedAction::new(action_id, &self.environment);

        if let Some(status) = self.environment.termination_status() {
            action.complete(&self.environment, status);
        } else if self.environment.in_python_repl() {
            let (outcome, stdout, stderr) = self
                .environment
                .run_script_capture_with_stdin(source, stdin);
            action.stdout = stdout;
            action.stderr = stderr;
            action.complete(&self.environment, outcome.exit_status);
            action.outcome = Some(outcome);
        } else {
            match self.environment.parse_shell_action(source) {
                Ok(node) => {
                    let execution = crate::exec::ShellExecution::start(
                        &mut self.environment,
                        &node,
                        stdin,
                        stdin_closed,
                    )?;
                    action.root_pid = execution.target_pid();
                    action.execution = Some(execution);
                    self.active_action = Some(action_id);
                }
                Err((status, diagnostic)) => {
                    action.stderr = diagnostic;
                    action.complete(&self.environment, status);
                }
            }
        }
        let view = action.view(&self.environment);
        self.actions.insert(action_id, action);
        Ok(view)
    }

    fn poll_action(
        &mut self,
        action_id: u64,
        work_quanta: usize,
        advance_time: bool,
    ) -> Result<ActionView, String> {
        if work_quanta == 0 || work_quanta > MAX_POLL_QUANTA {
            return Err(format!(
                "work_quanta must be between 1 and {MAX_POLL_QUANTA}"
            ));
        }
        let mut action = self
            .actions
            .remove(&action_id)
            .ok_or_else(|| format!("action {action_id} does not exist"))?;
        let result = (|| {
            let Some(mut execution) = action.execution.take() else {
                return Ok(action.view(&self.environment));
            };
            action.state = ActionState::Running;
            let mut completed = None;
            for _ in 0..work_quanta {
                let poll = match execution.poll(&mut self.environment, advance_time) {
                    Ok(poll) => poll,
                    Err(error) => {
                        action.execution = Some(execution);
                        return Err(error);
                    }
                };
                match poll {
                    crate::exec::MachinePoll::Progress => {}
                    crate::exec::MachinePoll::Blocked => {
                        let reason = match self.environment.scheduler.state(action.root_pid) {
                            Some(TaskState::Blocked(reason)) => Some(wait_reason_view(reason)),
                            _ => None,
                        };
                        action.state = ActionState::Blocked { reason };
                        break;
                    }
                    crate::exec::MachinePoll::Ready(status) => {
                        completed = Some(status);
                        break;
                    }
                }
            }
            execution.drain_output(
                &mut self.environment,
                &mut action.stdout,
                &mut action.stderr,
            );
            if let Some(status) = completed {
                execution.restore(&mut self.environment);
                action.complete(&self.environment, status);
                self.active_action = None;
            } else {
                action.execution = Some(execution);
            }
            Ok(action.view(&self.environment))
        })();
        self.actions.insert(action_id, action);
        result
    }

    fn write_action_stdin(&mut self, action_id: u64, bytes: &[u8]) -> Result<(), String> {
        let mut action = self
            .actions
            .remove(&action_id)
            .ok_or_else(|| format!("action {action_id} does not exist"))?;
        let result = match action.execution.as_mut() {
            Some(execution) => execution.write_stdin(&mut self.environment, bytes),
            None => Err(format!("action {action_id} is complete")),
        };
        if result.is_ok() {
            action.state = ActionState::Running;
        }
        self.actions.insert(action_id, action);
        result
    }

    fn close_action_stdin(&mut self, action_id: u64) -> Result<(), String> {
        let mut action = self
            .actions
            .remove(&action_id)
            .ok_or_else(|| format!("action {action_id} does not exist"))?;
        let result = match action.execution.as_mut() {
            Some(execution) => execution.close_stdin(&mut self.environment),
            None => Err(format!("action {action_id} is complete")),
        };
        if result.is_ok() {
            action.state = ActionState::Running;
        }
        self.actions.insert(action_id, action);
        result
    }

    fn cancel_action(&mut self, action_id: u64) -> Result<ActionView, String> {
        let mut action = self
            .actions
            .remove(&action_id)
            .ok_or_else(|| format!("action {action_id} does not exist"))?;
        let Some(mut execution) = action.execution.take() else {
            self.actions.insert(action_id, action);
            return Err(format!("action {action_id} is complete"));
        };
        let result = cancel_execution(
            &mut self.environment,
            &mut execution,
            action_id,
            action.root_pid,
        );
        let result = match result {
            Ok(status) => {
                execution.drain_output(
                    &mut self.environment,
                    &mut action.stdout,
                    &mut action.stderr,
                );
                execution.restore(&mut self.environment);
                if self.environment.exiting == Some(status) {
                    self.environment.exiting = None;
                }
                self.environment.last_status = status;
                action.complete(&self.environment, status);
                self.active_action = None;
                Ok(action.view(&self.environment))
            }
            Err(error) => {
                action.execution = Some(execution);
                Err(error)
            }
        };
        self.actions.insert(action_id, action);
        result
    }

    fn read_action_output(&mut self, action_id: u64) -> Result<ActionOutput, String> {
        let mut action = self
            .actions
            .remove(&action_id)
            .ok_or_else(|| format!("action {action_id} does not exist"))?;
        if let Some(execution) = action.execution.as_mut() {
            execution.drain_output(
                &mut self.environment,
                &mut action.stdout,
                &mut action.stderr,
            );
        }
        let stdout = action.stdout[action.stdout_cursor..].to_vec();
        let stderr = action.stderr[action.stderr_cursor..].to_vec();
        action.stdout_cursor = action.stdout.len();
        action.stderr_cursor = action.stderr.len();
        let closed = action.execution.is_none();
        self.actions.insert(action_id, action);
        Ok(ActionOutput {
            action_id,
            stdout_base64: STANDARD.encode(stdout),
            stderr_base64: STANDARD.encode(stderr),
            stdout_closed: closed,
            stderr_closed: closed,
        })
    }

    fn drop_action(&mut self, action_id: u64) -> Result<(), String> {
        let action = self
            .actions
            .get(&action_id)
            .ok_or_else(|| format!("action {action_id} does not exist"))?;
        if action.execution.is_some() {
            return Err(format!("action {action_id} is still active"));
        }
        self.actions.remove(&action_id);
        Ok(())
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

    fn stat_path(&self, path: &str, follow_symlinks: bool) -> Result<HarnessResult, String> {
        let path = absolute_workspace_path(path)?;
        let node = self
            .environment
            .vfs
            .metadata("/", &path, follow_symlinks)
            .map_err(|error| error.to_string())?;
        let (node_type, size, symlink_target) = match node.kind {
            NodeKind::File(data) => (PathType::File, data.len() as u64, None),
            NodeKind::Dir => (PathType::Directory, 0, None),
            NodeKind::Symlink(target) => {
                let size = target.len() as u64;
                (PathType::Symlink, size, Some(target))
            }
        };
        Ok(HarnessResult::PathMetadata(PathMetadata {
            path,
            node_type,
            mode: node.mode,
            uid: node.uid,
            gid: node.gid,
            mtime_ms: node.mtime,
            size,
            symlink_target,
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
            actions: self
                .actions
                .values()
                .map(|action| action.view(&self.environment))
                .collect(),
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

fn cancel_execution(
    environment: &mut Environment,
    execution: &mut crate::exec::ShellExecution,
    action_id: u64,
    root_pid: u32,
) -> Result<i32, String> {
    let process_group = environment
        .processes
        .get(root_pid)
        .map(|record| record.process_group)
        .ok_or_else(|| format!("action {action_id} root process does not exist"))?;
    let _ = execution.close_stdin(environment);
    for pid in environment.processes.running_group(process_group) {
        if pid != root_pid {
            environment.send_signal(pid, crate::process::Signal::Kill)?;
        }
    }

    let mut root_signaled = false;
    for _ in 0..MAX_CANCEL_QUANTA {
        let live_group = environment.processes.running_group(process_group);
        if !root_signaled && live_group.iter().all(|pid| *pid == root_pid) {
            environment.send_signal(root_pid, crate::process::Signal::Kill)?;
            root_signaled = true;
        }
        match execution.poll(environment, false)? {
            crate::exec::MachinePoll::Progress => {}
            crate::exec::MachinePoll::Blocked => {
                return Err(format!(
                    "action {action_id} cancellation blocked without a modeled event"
                ));
            }
            crate::exec::MachinePoll::Ready(status) => return Ok(status),
        }
    }
    Err(format!(
        "action {action_id} cancellation exceeded its work bound"
    ))
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
        WaitReason::InputReadable(description) => WaitReasonView::InputReadable { description },
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

fn default_true() -> bool {
    true
}

fn default_poll_quanta() -> usize {
    1_024
}

fn default_directory_mode() -> u32 {
    0o755
}

fn default_patch_strip() -> usize {
    1
}

fn validate_mode(mode: u32) -> Result<(), String> {
    if mode & !0o7777 == 0 {
        Ok(())
    } else {
        Err(format!("invalid file mode {mode:#o}"))
    }
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

    #[test]
    fn retained_action_polls_time_and_output_incrementally() {
        let mut session = HarnessSession::new(Limits::default());
        let started = session
            .start_action("printf one; sleep 2; printf two", &[], true)
            .unwrap();
        let action_id = started.action_id;
        let blocked = session.poll_action(action_id, 100, false).unwrap();
        assert_eq!(
            blocked.state,
            ActionState::Blocked {
                reason: Some(WaitReasonView::Timer {
                    deadline_ns: 2_000_000_000
                })
            }
        );
        assert_eq!(session.environment.clock.monotonic_ns(), 0);
        let first = session.read_action_output(action_id).unwrap();
        assert_eq!(STANDARD.decode(first.stdout_base64).unwrap(), b"one");
        assert!(!first.stdout_closed);

        let mut branch = session.fork().unwrap();
        for machine in [&mut session, &mut branch] {
            let complete = machine.poll_action(action_id, 100, true).unwrap();
            assert_eq!(complete.state, ActionState::Complete { status: 0 });
            assert_eq!(machine.environment.clock.monotonic_ns(), 2_000_000_000);
            let final_output = machine.read_action_output(action_id).unwrap();
            assert_eq!(STANDARD.decode(final_output.stdout_base64).unwrap(), b"two");
            assert!(final_output.stdout_closed);
        }
    }

    #[test]
    fn retained_action_accepts_bounded_streaming_stdin() {
        let mut session = HarnessSession::new(Limits::default());
        let action_id = session.start_action("cat", &[], false).unwrap().action_id;
        let blocked = session.poll_action(action_id, 100, false).unwrap();
        assert!(matches!(
            blocked.state,
            ActionState::Blocked {
                reason: Some(WaitReasonView::InputReadable { .. })
            }
        ));
        session.write_action_stdin(action_id, b"hello").unwrap();
        let blocked = session.poll_action(action_id, 100, false).unwrap();
        assert!(matches!(blocked.state, ActionState::Blocked { .. }));
        session.close_action_stdin(action_id).unwrap();
        let complete = session.poll_action(action_id, 100, false).unwrap();
        assert_eq!(complete.state, ActionState::Complete { status: 0 });
        let output = session.read_action_output(action_id).unwrap();
        assert_eq!(STANDARD.decode(output.stdout_base64).unwrap(), b"hello");
        assert!(session.write_action_stdin(action_id, b"late").is_err());
        session.drop_action(action_id).unwrap();
    }

    #[test]
    fn typed_workspace_operations_share_atomic_vfs_foundations() {
        let mut session = HarnessSession::new(Limits::default());
        assert!(session
            .apply(HarnessOperation::MakeDirectory {
                path: "nested/leaf".into(),
                mode: 0o750,
                parents: false,
            })
            .is_err());
        session
            .apply(HarnessOperation::MakeDirectory {
                path: "nested/leaf".into(),
                mode: 0o750,
                parents: true,
            })
            .unwrap();
        session
            .apply(HarnessOperation::WriteFile {
                path: "nested/leaf/note".into(),
                data_base64: STANDARD.encode(b"old\n"),
                mode: 0o640,
            })
            .unwrap();
        session
            .apply(HarnessOperation::CreateSymlink {
                path: "link".into(),
                target: "nested/leaf/note".into(),
            })
            .unwrap();
        let HarnessResult::PathMetadata(link) = session
            .stat_path("link", false)
            .expect("lstat workspace link")
        else {
            panic!("stat returned the wrong result kind");
        };
        assert!(matches!(link.node_type, PathType::Symlink));
        assert_eq!(link.symlink_target.as_deref(), Some("nested/leaf/note"));
        let HarnessResult::PathMetadata(file) = session
            .stat_path("link", true)
            .expect("follow workspace link")
        else {
            panic!("stat returned the wrong result kind");
        };
        assert!(matches!(file.node_type, PathType::File));
        assert_eq!(file.mode, 0o640);
        assert_eq!(file.size, 4);

        session.environment.run_script_capture("cd /");
        session
            .apply(HarnessOperation::ApplyPatch {
                patch: "*** Begin Patch\n*** Update File: nested/leaf/note\n@@\n-old\n+new\n*** End Patch\n".into(),
                strip: 1,
            })
            .unwrap();
        assert_eq!(
            session
                .environment
                .vfs
                .read("/", "/work/nested/leaf/note")
                .unwrap(),
            b"new\n"
        );
        assert!(!session.environment.vfs.exists("/", "/nested/leaf/note"));
    }

    #[test]
    fn harness_patch_failure_is_atomic_and_bounded() {
        let mut session = HarnessSession::new(Limits::default());
        session
            .environment
            .vfs
            .put_file("/work/note", b"keep\n".to_vec(), 0o644)
            .unwrap();
        let invalid = HarnessOperation::ApplyPatch {
            patch:
                "*** Begin Patch\n*** Update File: note\n@@\n-missing\n+changed\n*** End Patch\n"
                    .into(),
            strip: 1,
        };
        assert!(session.apply(invalid).is_err());
        assert_eq!(
            session.environment.vfs.read("/", "/work/note").unwrap(),
            b"keep\n"
        );
        let oversized = HarnessOperation::ApplyPatch {
            patch: "x".repeat(8 * 1024 * 1024 + 1),
            strip: 1,
        };
        assert!(session.apply(oversized).unwrap_err().contains("8 MiB"));
    }

    #[test]
    fn cancel_action_terminates_foreground_group_and_reuses_session() {
        let mut session = HarnessSession::new(Limits::default());
        let action_id = session
            .start_action("sleep 10 | cat", &[], true)
            .unwrap()
            .action_id;
        let blocked = session.poll_action(action_id, 100, false).unwrap();
        assert!(matches!(blocked.state, ActionState::Blocked { .. }));
        assert!(session.environment.clock.pending_len() > 0);

        let cancelled = session.cancel_action(action_id).unwrap();
        assert!(matches!(
            cancelled.state,
            ActionState::Complete { status: 137 }
        ));
        assert_eq!(cancelled.outcome.unwrap().exit_status, 137);
        assert_eq!(session.environment.clock.pending_len(), 0);
        assert!(session.active_action.is_none());
        assert!(session
            .environment
            .processes
            .running_group(cancelled.root_pid)
            .iter()
            .all(|pid| *pid == cancelled.root_pid));

        let HarnessResult::Execute(reused) = session.execute("printf reused", &[]).unwrap() else {
            panic!("reused session returned the wrong result kind");
        };
        assert_eq!(reused.outcome.exit_status, 0);
        assert_eq!(STANDARD.decode(reused.stdout_base64).unwrap(), b"reused");
        assert!(session.cancel_action(action_id).is_err());
    }
}
