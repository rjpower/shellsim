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
pub(crate) const MAX_COMMAND_BYTES: usize = 4 * 1024;

/// Standard signals modeled by shellsim's process and shell-disposition layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Signal {
    Hangup,
    Interrupt,
    Kill,
    User1,
    User2,
    Pipe,
    Terminate,
    Child,
    Continue,
    Stop,
}

impl Signal {
    /// Conventional Unix signal number used in shell exit statuses.
    pub const fn number(self) -> i32 {
        match self {
            Self::Hangup => 1,
            Self::Interrupt => 2,
            Self::Kill => 9,
            Self::User1 => 10,
            Self::User2 => 12,
            Self::Pipe => 13,
            Self::Terminate => 15,
            Self::Child => 17,
            Self::Continue => 18,
            Self::Stop => 19,
        }
    }

    /// Whether the currently modeled default disposition terminates the target.
    pub const fn terminates(self) -> bool {
        matches!(
            self,
            Self::Hangup
                | Self::Interrupt
                | Self::Kill
                | Self::User1
                | Self::User2
                | Self::Pipe
                | Self::Terminate
        )
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Hangup => "HUP",
            Self::Interrupt => "INT",
            Self::Kill => "KILL",
            Self::User1 => "USR1",
            Self::User2 => "USR2",
            Self::Pipe => "PIPE",
            Self::Terminate => "TERM",
            Self::Child => "CHLD",
            Self::Continue => "CONT",
            Self::Stop => "STOP",
        }
    }

    /// Parse a signal name with an optional `SIG` prefix, or a supported decimal number.
    pub fn parse(value: &str) -> Option<Self> {
        let upper = value.to_ascii_uppercase();
        match upper.strip_prefix("SIG").unwrap_or(&upper) {
            "1" | "HUP" => Some(Self::Hangup),
            "2" | "INT" => Some(Self::Interrupt),
            "9" | "KILL" => Some(Self::Kill),
            "10" | "USR1" => Some(Self::User1),
            "12" | "USR2" => Some(Self::User2),
            "13" | "PIPE" => Some(Self::Pipe),
            "15" | "TERM" => Some(Self::Terminate),
            "17" | "CHLD" => Some(Self::Child),
            "18" | "CONT" => Some(Self::Continue),
            "19" | "STOP" => Some(Self::Stop),
            _ => None,
        }
    }
}

/// Observable lifecycle state for a logical process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessStatus {
    /// The process is currently executing in the synchronous interpreter.
    Running,
    /// The process is suspended by an uncatchable job-control stop.
    Stopped(Signal),
    /// The process has completed and is waiting to be reaped.
    Exited(i32),
}

/// Read-only process metadata used by `ps`, job control, and synthetic `/proc` files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessRecord {
    pub pid: ProcessId,
    pub ppid: ProcessId,
    /// Modeled effective user identity inherited across logical forks.
    pub uid: u32,
    /// Process-group identity used for job-wide signal delivery.
    pub process_group: ProcessId,
    /// Session identity. A new session is also led by its first process group.
    pub session_id: ProcessId,
    pub command: String,
    pub cwd: String,
    /// Exported environment captured at creation or the latest inspection point.
    pub environment: BTreeMap<String, String>,
    /// Snapshot of descriptor targets for generated `/proc/PID/fd` views.
    pub descriptors: BTreeMap<i32, String>,
    pub status: ProcessStatus,
}

/// Placement of a new logical child relative to its parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChildPlacement {
    /// Inherit both session and process group.
    Inherit,
    /// Lead a new process group within the parent's session, as for a background job.
    NewProcessGroup,
    /// Lead a new session and process group, as for `setsid` or Python `start_new_session`.
    NewSession,
}

/// The single synthetic controlling terminal owned by the top-level shell session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllingTerminal {
    pub session_id: ProcessId,
    pub foreground_group: ProcessId,
}

impl ControllingTerminal {
    /// Create a terminal initially controlled by its session leader's process group.
    pub const fn new(session_id: ProcessId) -> Self {
        Self {
            session_id,
            foreground_group: session_id,
        }
    }

    /// Select a running group from the controlling session as foreground owner.
    pub fn set_foreground(
        &mut self,
        processes: &ProcessTable,
        process_group: ProcessId,
    ) -> Result<(), String> {
        let valid = processes.records.values().any(|record| {
            !matches!(record.status, ProcessStatus::Exited(_))
                && record.session_id == self.session_id
                && record.process_group == process_group
        });
        if !valid {
            return Err(format!(
                "process group {process_group} is not live in terminal session {}",
                self.session_id
            ));
        }
        self.foreground_group = process_group;
        Ok(())
    }
}

/// Parent-owned descriptor endpoints and collected state for one live child handle.
///
/// The handle is machine state rather than Python heap state so native facades can refer to it by
/// PID without exposing descriptor arena identities to simulated code.
#[derive(Clone)]
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
    /// Bytes retained while a direct Python stdin write is suspended by pipe backpressure.
    pub pending_stdin_write: Option<Vec<u8>>,
    /// First byte of `pending_stdin_write` not yet accepted by the pipe.
    pub pending_stdin_offset: usize,
    /// Virtual deadline retained across retries of `Popen.wait(timeout=...)`.
    pub wait_event: Option<crate::clock::EventId>,
    /// Virtual deadline retained across retries of `Popen.communicate(timeout=...)`.
    pub communicate_event: Option<crate::clock::EventId>,
    pub stdin_pipe: bool,
    pub stdout_pipe: bool,
    pub stderr_pipe: bool,
    pub stdout_inherit: bool,
    pub stderr_inherit: bool,
    pub terminating_signal: Option<Signal>,
}

/// Machine-wide allocator and process record store.
#[derive(Clone)]
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
                uid: 0,
                process_group: root_pid,
                session_id: root_pid,
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
        placement: ChildPlacement,
        command: &str,
        cwd: &str,
        environment: BTreeMap<String, String>,
    ) -> Option<ProcessId> {
        if self.records.len() >= MAX_PROCESSES || command.len() > MAX_COMMAND_BYTES {
            return None;
        }
        let pid = self.next_pid;
        self.next_pid = self.next_pid.checked_add(1)?;
        let parent = self.records.get(&ppid)?;
        let uid = parent.uid;
        let (process_group, session_id) = match placement {
            ChildPlacement::Inherit => (parent.process_group, parent.session_id),
            ChildPlacement::NewProcessGroup => (pid, parent.session_id),
            ChildPlacement::NewSession => (pid, pid),
        };
        self.records.insert(
            pid,
            ProcessRecord {
                pid,
                ppid,
                uid,
                process_group,
                session_id,
                command: command.to_string(),
                cwd: cwd.to_string(),
                environment,
                descriptors: BTreeMap::new(),
                status: ProcessStatus::Running,
            },
        );
        Some(pid)
    }

    /// Update the effective identity retained for one logical process.
    pub fn set_uid(&mut self, pid: ProcessId, uid: u32) {
        if let Some(record) = self.records.get_mut(&pid) {
            record.uid = uid;
        }
    }

    /// Mark a logical process complete and capture its final working directory.
    pub fn exit(&mut self, pid: ProcessId, status: i32, cwd: &str) {
        if let Some(record) = self.records.get_mut(&pid) {
            record.cwd = cwd.to_string();
            record.status = ProcessStatus::Exited(status);
        }
    }

    /// Mark a live process stopped while retaining its execution context.
    pub fn stop(&mut self, pid: ProcessId, signal: Signal) {
        if let Some(record) = self.records.get_mut(&pid) {
            if !matches!(record.status, ProcessStatus::Exited(_)) {
                record.status = ProcessStatus::Stopped(signal);
            }
        }
    }

    /// Mark a stopped process running again.
    pub fn continue_process(&mut self, pid: ProcessId) {
        if let Some(record) = self.records.get_mut(&pid) {
            if matches!(record.status, ProcessStatus::Stopped(_)) {
                record.status = ProcessStatus::Running;
            }
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

    /// Return running members of one process group in stable PID order.
    pub fn running_group(&self, process_group: ProcessId) -> Vec<ProcessId> {
        self.records
            .values()
            .filter(|record| {
                record.process_group == process_group && record.status == ProcessStatus::Running
            })
            .map(|record| record.pid)
            .collect()
    }

    /// Return non-exited members of one process group in stable PID order.
    pub fn live_group(&self, process_group: ProcessId) -> Vec<ProcessId> {
        self.records
            .values()
            .filter(|record| {
                record.process_group == process_group
                    && !matches!(record.status, ProcessStatus::Exited(_))
            })
            .map(|record| record.pid)
            .collect()
    }

    /// Return one process tree in stable PID order, including `root` when it is retained.
    pub(crate) fn process_tree(&self, root: ProcessId) -> Vec<ProcessId> {
        let mut selected = vec![root];
        let mut index = 0;
        while index < selected.len() {
            let parent = selected[index];
            for record in self.records.values() {
                if record.ppid == parent && !selected.contains(&record.pid) {
                    selected.push(record.pid);
                }
            }
            index += 1;
        }
        selected.retain(|pid| self.records.contains_key(pid));
        selected.sort_unstable();
        selected
    }

    /// Return the non-exited portion of [`Self::process_tree`].
    pub(crate) fn live_process_tree(&self, root: ProcessId) -> Vec<ProcessId> {
        self.process_tree(root)
            .into_iter()
            .filter(|pid| {
                self.records
                    .get(pid)
                    .is_some_and(|record| !matches!(record.status, ProcessStatus::Exited(_)))
            })
            .collect()
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
    fn user_signals_accept_names_prefixes_and_numbers() {
        assert_eq!(Signal::parse("USR1"), Some(Signal::User1));
        assert_eq!(Signal::parse("SIGUSR2"), Some(Signal::User2));
        assert_eq!(Signal::parse("10"), Some(Signal::User1));
        assert_eq!(Signal::parse("12"), Some(Signal::User2));
        assert!(Signal::User1.terminates());
    }

    #[test]
    fn pids_are_stable_and_exited_children_are_reapable() {
        let mut table = ProcessTable::new(1_000, "/".to_string(), BTreeMap::new());
        let child = table
            .spawn(
                1_000,
                ChildPlacement::Inherit,
                "worker",
                "/work",
                BTreeMap::new(),
            )
            .unwrap();
        assert_eq!(child, 1_001);
        assert_eq!(table.get(child).unwrap().ppid, 1_000);
        assert_eq!(table.get(child).unwrap().process_group, 1_000);
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
            last = table
                .spawn(
                    1_000,
                    ChildPlacement::Inherit,
                    "worker",
                    "/",
                    BTreeMap::new(),
                )
                .unwrap();
        }
        assert!(table
            .spawn(
                1_000,
                ChildPlacement::Inherit,
                "overflow",
                "/",
                BTreeMap::new()
            )
            .is_none());
        table.exit(last, 0, "/");
        table.reap(last).unwrap();
        assert!(table
            .spawn(
                1_000,
                ChildPlacement::NewProcessGroup,
                "replacement",
                "/",
                BTreeMap::new()
            )
            .is_some());
    }

    #[test]
    fn new_groups_use_the_child_pid_and_are_queryable() {
        let mut table = ProcessTable::new(1_000, "/".to_string(), BTreeMap::new());
        let leader = table
            .spawn(
                1_000,
                ChildPlacement::NewProcessGroup,
                "leader",
                "/",
                BTreeMap::new(),
            )
            .unwrap();
        let member = table
            .spawn(
                leader,
                ChildPlacement::Inherit,
                "member",
                "/",
                BTreeMap::new(),
            )
            .unwrap();
        assert_eq!(table.running_group(leader), vec![leader, member]);
        table.exit(leader, 0, "/");
        assert_eq!(table.running_group(leader), vec![member]);
    }

    #[test]
    fn groups_and_sessions_are_distinct_placements() {
        let mut table = ProcessTable::new(1_000, "/".to_string(), BTreeMap::new());
        let group = table
            .spawn(
                1_000,
                ChildPlacement::NewProcessGroup,
                "job",
                "/",
                BTreeMap::new(),
            )
            .unwrap();
        let session = table
            .spawn(
                1_000,
                ChildPlacement::NewSession,
                "session",
                "/",
                BTreeMap::new(),
            )
            .unwrap();
        assert_eq!(table.get(group).unwrap().process_group, group);
        assert_eq!(table.get(group).unwrap().session_id, 1_000);
        assert_eq!(table.get(session).unwrap().process_group, session);
        assert_eq!(table.get(session).unwrap().session_id, session);

        let mut terminal = ControllingTerminal::new(1_000);
        terminal.set_foreground(&table, group).unwrap();
        assert_eq!(terminal.foreground_group, group);
        assert!(terminal.set_foreground(&table, session).is_err());
    }
}
