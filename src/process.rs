//! Deterministic logical process identities and lifecycle state.
//!
//! Shellsim never creates host processes. This table gives simulated children stable PIDs and
//! parentage across cooperative scheduler activations. Running entries support process inspection;
//! exited background entries remain until `wait` reaps them. Fixed record and command-size bounds
//! prevent simulated input from growing unbounded host state.

use std::collections::BTreeMap;

use crate::descriptors::FdTable;

/// PID assigned to a simulated process.
pub type ProcessId = u32;

/// Maximum number of simultaneously retained logical process records.
pub const MAX_PROCESSES: usize = 1_024;
const MAX_COMMAND_BYTES: usize = 4 * 1024;

/// Standard signals modeled by shellsim's default-disposition process layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Signal {
    Hangup,
    Interrupt,
    Kill,
    Pipe,
    Terminate,
    Child,
}

impl Signal {
    /// Conventional Unix signal number used in shell exit statuses.
    pub const fn number(self) -> i32 {
        match self {
            Self::Hangup => 1,
            Self::Interrupt => 2,
            Self::Kill => 9,
            Self::Pipe => 13,
            Self::Terminate => 15,
            Self::Child => 17,
        }
    }

    /// Whether the currently modeled default disposition terminates the target.
    pub const fn terminates(self) -> bool {
        !matches!(self, Self::Child)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Hangup => "HUP",
            Self::Interrupt => "INT",
            Self::Kill => "KILL",
            Self::Pipe => "PIPE",
            Self::Terminate => "TERM",
            Self::Child => "CHLD",
        }
    }

    /// Parse a signal name with an optional `SIG` prefix, or a supported decimal number.
    pub fn parse(value: &str) -> Option<Self> {
        let upper = value.to_ascii_uppercase();
        match upper.strip_prefix("SIG").unwrap_or(&upper) {
            "1" | "HUP" => Some(Self::Hangup),
            "2" | "INT" => Some(Self::Interrupt),
            "9" | "KILL" => Some(Self::Kill),
            "13" | "PIPE" => Some(Self::Pipe),
            "15" | "TERM" => Some(Self::Terminate),
            "17" | "CHLD" => Some(Self::Child),
            _ => None,
        }
    }
}

/// Observable lifecycle state for a logical process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessStatus {
    /// The process is currently executing in the synchronous interpreter.
    Running,
    /// The process has completed and is waiting to be reaped.
    Exited(i32),
}

/// Read-only process metadata used by `ps`, job control, and synthetic `/proc` files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessRecord {
    pub pid: ProcessId,
    pub ppid: ProcessId,
    pub command: String,
    pub cwd: String,
    /// Exported environment captured at creation or the latest inspection point.
    pub environment: BTreeMap<String, String>,
    /// Snapshot of descriptor targets for generated `/proc/PID/fd` views.
    pub descriptors: BTreeMap<i32, String>,
    pub status: ProcessStatus,
}

/// Parent-owned descriptor endpoints and collected state for one live child handle.
///
/// The handle is machine state rather than Python heap state so native facades can refer to it by
/// PID without exposing descriptor arena identities to simulated code.
pub(crate) struct LiveChild {
    pub owner: ProcessId,
    pub endpoints: FdTable,
    pub status: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub communicated: bool,
    /// Input retained across timed-out `communicate` calls.
    pub communicate_input: Option<Vec<u8>>,
    /// First byte not yet accepted by the child stdin pipe.
    pub communicate_offset: usize,
    pub stdin_pipe: bool,
    pub stdout_pipe: bool,
    pub stderr_pipe: bool,
    pub stdout_inherit: bool,
    pub stderr_inherit: bool,
    pub terminating_signal: Option<Signal>,
}

/// Machine-wide allocator and process record store.
pub struct ProcessTable {
    next_pid: ProcessId,
    records: BTreeMap<ProcessId, ProcessRecord>,
}

impl ProcessTable {
    /// Create a process table containing the persistent top-level shell.
    pub fn new(root_pid: ProcessId, cwd: String, environment: BTreeMap<String, String>) -> Self {
        let mut records = BTreeMap::new();
        records.insert(
            root_pid,
            ProcessRecord {
                pid: root_pid,
                ppid: 0,
                command: "bash".to_string(),
                cwd,
                environment,
                descriptors: BTreeMap::new(),
                status: ProcessStatus::Running,
            },
        );
        Self {
            next_pid: root_pid.saturating_add(1),
            records,
        }
    }

    /// Allocate one bounded logical child. Returns `None` at the explicit process frontier.
    pub fn spawn(
        &mut self,
        ppid: ProcessId,
        command: &str,
        cwd: &str,
        environment: BTreeMap<String, String>,
    ) -> Option<ProcessId> {
        if self.records.len() >= MAX_PROCESSES || command.len() > MAX_COMMAND_BYTES {
            return None;
        }
        let pid = self.next_pid;
        self.next_pid = self.next_pid.checked_add(1)?;
        self.records.insert(
            pid,
            ProcessRecord {
                pid,
                ppid,
                command: command.to_string(),
                cwd: cwd.to_string(),
                environment,
                descriptors: BTreeMap::new(),
                status: ProcessStatus::Running,
            },
        );
        Some(pid)
    }

    /// Mark a logical process complete and capture its final working directory.
    pub fn exit(&mut self, pid: ProcessId, status: i32, cwd: &str) {
        if let Some(record) = self.records.get_mut(&pid) {
            record.cwd = cwd.to_string();
            record.status = ProcessStatus::Exited(status);
        }
    }

    /// Update the live metadata for the currently executing process.
    pub fn update_current(
        &mut self,
        pid: ProcessId,
        cwd: &str,
        environment: BTreeMap<String, String>,
    ) {
        if let Some(record) = self.records.get_mut(&pid) {
            record.cwd = cwd.to_string();
            record.environment = environment;
        }
    }

    /// Update the descriptor links visible for one retained process.
    pub fn update_descriptors(&mut self, pid: ProcessId, descriptors: BTreeMap<i32, String>) {
        if let Some(record) = self.records.get_mut(&pid) {
            record.descriptors = descriptors;
        }
    }

    /// Look up one retained record.
    pub fn get(&self, pid: ProcessId) -> Option<&ProcessRecord> {
        self.records.get(&pid)
    }

    /// Iterate records in stable PID order.
    pub fn iter(&self) -> impl Iterator<Item = &ProcessRecord> {
        self.records.values()
    }

    /// Reap an exited child. The root and running processes cannot be removed.
    pub fn reap(&mut self, pid: ProcessId) -> Option<ProcessRecord> {
        if matches!(self.records.get(&pid)?.status, ProcessStatus::Exited(_)) {
            self.records.remove(&pid)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pids_are_stable_and_exited_children_are_reapable() {
        let mut table = ProcessTable::new(1_000, "/".to_string(), BTreeMap::new());
        let child = table
            .spawn(1_000, "worker", "/work", BTreeMap::new())
            .unwrap();
        assert_eq!(child, 1_001);
        assert_eq!(table.get(child).unwrap().ppid, 1_000);
        table.exit(child, 7, "/tmp");
        assert_eq!(table.get(child).unwrap().status, ProcessStatus::Exited(7));
        assert_eq!(table.reap(child).unwrap().cwd, "/tmp");
        assert!(table.get(child).is_none());
    }

    #[test]
    fn process_capacity_fails_atomically_and_reaping_releases_a_slot() {
        let mut table = ProcessTable::new(1_000, "/".to_string(), BTreeMap::new());
        let mut last = 1_000;
        for _ in 1..MAX_PROCESSES {
            last = table.spawn(1_000, "worker", "/", BTreeMap::new()).unwrap();
        }
        assert!(table
            .spawn(1_000, "overflow", "/", BTreeMap::new())
            .is_none());
        table.exit(last, 0, "/");
        table.reap(last).unwrap();
        assert!(table
            .spawn(1_000, "replacement", "/", BTreeMap::new())
            .is_some());
    }
}
