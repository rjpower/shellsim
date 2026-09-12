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
use crate::vfs::{Node, NodeKind, Vfs};
use crate::{Environment, Limits, RunOutcome};

const MAX_TRANSFER_RAW_BYTES: usize = 6 * 1024 * 1024;

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
    pub network_requests: Vec<NetworkRequest>,
    pub dropped_network_requests: u64,
}

#[derive(Debug, Serialize)]
pub struct ProcessView {
    pub pid: u32,
    pub ppid: u32,
    pub command: String,
    pub cwd: String,
    pub status: ProcessViewStatus,
}

/// Typed lifecycle state in an inspection response.
#[derive(Debug, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ProcessViewStatus {
    Running,
    Exited { status: i32 },
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
        let noop_before = self.environment.trust_noop.clone();
        let partial_before = self.environment.trust_partial.clone();
        let network_start = self.environment.net.log.len();
        let dropped_network_start = self.environment.net.dropped_requests;
        let (outcome, stdout, stderr) = self
            .environment
            .run_script_capture_with_stdin(source, stdin);
        Ok(HarnessResult::Execute(ExecuteResult {
            outcome,
            stdout_base64: STANDARD.encode(stdout),
            stderr_base64: STANDARD.encode(stderr),
            commands: self.environment.cmd_trace[command_start..].to_vec(),
            unsupported: self.environment.unsupported[unsupported_start..].to_vec(),
            noop_commands: self
                .environment
                .trust_noop
                .difference(&noop_before)
                .cloned()
                .collect(),
            partial_commands: self
                .environment
                .trust_partial
                .difference(&partial_before)
                .cloned()
                .collect(),
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
                .map(process_view)
                .collect(),
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

fn process_view(record: &ProcessRecord) -> ProcessView {
    ProcessView {
        pid: record.pid,
        ppid: record.ppid,
        command: record.command.clone(),
        cwd: record.cwd.clone(),
        status: match record.status {
            ProcessStatus::Running => ProcessViewStatus::Running,
            ProcessStatus::Exited(status) => ProcessViewStatus::Exited { status },
        },
    }
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
