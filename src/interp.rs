//! Interpreter state shared across the shell executor and all commands.

use std::collections::{BTreeMap, HashMap};
use std::ops::{Deref, DerefMut};

use crate::clock::{Clock, EventId};
use crate::descriptors::{
    DescriptionId, DescriptorArena, DescriptorError, Fd, FdTable, IoPoll, IoWait, MAX_CAPTURE_BYTES,
};
use crate::net::VirtualNet;
use crate::process::{ProcessId, ProcessTable, Signal};
use crate::resources::{Limits, Resources, RunOutcome};
use crate::scheduler::{Scheduler, WaitReason};
use crate::telemetry::InvocationLog;
use crate::vfs::Vfs;

/// A bash array value. Indexed arrays are sparse (`arr[5]=x` on an empty array is legal),
/// so unset slots are `None`. Associative arrays preserve sorted key order (bash uses an
/// unspecified hash order; sorted is deterministic and good enough for our checks).
#[derive(Clone, Debug)]
pub enum ArrayVal {
    Indexed(Vec<Option<String>>),
    Assoc(BTreeMap<String, String>),
}

/// A simulated background job started with `&`.
///
/// Its process continuation is owned separately by [`ProcessStates`]. This shell-facing record
/// supplies stable job IDs and retains exit status until `wait` reaps the process.
#[derive(Clone, Debug)]
pub struct Job {
    pub id: u32,
    pub pid: ProcessId,
    pub cmd: String,
    pub state: JobState,
}

/// Shell-visible lifecycle of a background job leader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobState {
    Running,
    Stopped(Signal),
    Done(i32),
}

/// Parsed simple-command alias retained as process-local shell state.
#[derive(Clone, Debug)]
pub(crate) struct AliasDefinition {
    pub source: String,
    pub words: Vec<String>,
}

/// Non-default shell behavior installed for one modeled signal.
#[derive(Clone, Debug)]
pub(crate) enum ShellSignalDisposition {
    Ignore,
    Handler {
        source: String,
        body: crate::shell::Node,
    },
}

/// One signal action selected for the active process at a scheduler boundary.
#[derive(Clone, Debug)]
pub(crate) enum SignalDelivery {
    Terminate(Signal),
    Stop(Signal),
    Handler(crate::shell::Node),
}

/// Machine-wide state and limits shared by cooperatively scheduled logical processes.
#[derive(Clone)]
pub struct Environment {
    pub vfs: Vfs,
    pub clock: Clock,
    pub net: VirtualNet,
    /// Deterministic CPU, transient-memory, and output accounting for this environment.
    pub resources: Resources,
    /// Machine-wide logical process identities and retained child statuses.
    pub processes: ProcessTable,
    /// Synthetic controlling terminal and its foreground process group.
    pub terminal: crate::process::ControllingTerminal,
    /// Deterministic runnable/blocked lifecycle for logical process execution.
    pub scheduler: Scheduler,
    /// Machine-owned open descriptions shared by forked process descriptor tables.
    pub descriptors: DescriptorArena,
    /// Complete process-local contexts keyed by logical PID, with one active context.
    pub process: ProcessStates,
    /// Parent-side endpoints retained for live subprocess handles.
    pub(crate) live_children: BTreeMap<ProcessId, crate::process::LiveChild>,
    next_temp_id: u64,
    /// Bounded trace of external command names executed.
    pub cmd_trace: crate::telemetry::BoundedTextLog,
    /// Bounded diagnostics for capabilities shellsim does not implement.
    pub unsupported: crate::telemetry::BoundedTextLog,
    /// Bounded ordered command occurrences, including suspended invocations.
    pub invocations: InvocationLog,
    /// Packages recorded by lightweight package-manager compatibility commands.
    pub packages: std::collections::BTreeSet<String>,
    /// Terminal bytes produced by detached children after their originating action returned.
    pub(crate) pending_stdout: Vec<u8>,
    pub(crate) pending_stderr: Vec<u8>,
}

/// Shell-local state for the single process currently executing in an [`Environment`].
#[derive(Clone)]
pub struct ProcessState {
    /// PID of the currently executing logical process.
    pub pid: ProcessId,
    /// PID of the logical parent process.
    pub ppid: ProcessId,
    /// Session inherited from the parent or established by `start_new_session`.
    pub session_id: ProcessId,
    /// Process group inherited across ordinary forks or established by a group leader.
    pub process_group: ProcessId,
    /// PID expanded by `$$`; preserved across Bash subshells.
    pub shell_pid: ProcessId,
    /// shell + environment variables (we don't distinguish exported vs not for simplicity,
    /// except that `env`/child python only sees exported ones, tracked in `exported`)
    pub vars: HashMap<String, String>,
    /// bash arrays (indexed + associative), keyed by variable name. A name present here is an
    /// array; it shadows any scalar `vars` entry of the same name for `${name[...]}` access.
    pub arrays: HashMap<String, ArrayVal>,
    pub exported: std::collections::BTreeSet<String>,
    pub cwd: String,
    /// Bash-style directory stack, stored oldest-to-newest beneath the current directory.
    pub directory_stack: Vec<String>,
    pub funcs: HashMap<String, crate::shell::Node>,
    /// Process-local simple-command aliases expanded before normal command dispatch.
    pub(crate) aliases: HashMap<String, AliasDefinition>,
    /// `$?`
    pub last_status: i32,
    /// positional parameters `$1 $2 ... $@`
    pub positional: Vec<String>,
    /// `set -e` / `set -u` / `set -x`
    pub opt_errexit: bool,
    pub opt_nounset: bool,
    pub opt_xtrace: bool,
    /// `set -o pipefail`
    pub opt_pipefail: bool,
    pub jobs: Vec<Job>,
    next_job_id: u32,
    /// Recursion and loop-control signaling.
    pub loop_break: u32,
    pub loop_continue: u32,
    pub returning: Option<i32>,
    pub exiting: Option<i32>,
    /// depth of "condition" contexts (if/while/&&/||/!) where `set -e` is suppressed
    pub cond_depth: u32,
    /// Deadline event currently unwinding the synchronous executor.  A future resumable
    /// scheduler will keep this on each task frame; today there is one shell process.
    pub(crate) deadline_interrupt: Option<EventId>,
    /// effective uid (for permission-ish checks / `id`)
    pub uid: u32,
    /// persistent input stream + cursor for `read` inside `while read…; done < file`
    pub input_stream: Vec<u8>,
    pub input_pos: usize,
    /// A foreground Python REPL, when `python` was invoked without a program. Keeping this in
    /// process state lets an agent enter Python in one shell action and continue it in later
    /// actions without giving the shim access to host stdin.
    pub python_repl: Option<crate::python::ReplState>,
    /// Unix-style descriptor map inherited by logical children.
    pub(crate) fds: FdTable,
    /// Resumable shell execution frames retained across scheduler activations.
    pub(crate) shell_continuation: Option<crate::exec::ShellContinuation>,
    /// Coalesced standard signals awaiting delivery at a scheduler boundary.
    pending_signals: std::collections::BTreeSet<Signal>,
    /// Non-default signal actions installed by the shell `trap` builtin.
    pub(crate) signal_dispositions: BTreeMap<Signal, ShellSignalDisposition>,
    /// Prevent ordinary caught signals from recursively interrupting their own handler.
    handling_signal: bool,
    /// Memory reserved for this forked context and released independently at exit.
    fork_allocation_bytes: u64,
    detached_output: bool,
}

/// Machine-owned process contexts.
///
/// Field access dereferences to the active context for compatibility with command code, while
/// inactive parents and runnable siblings remain stored by PID for scheduler activation.
#[derive(Clone)]
pub struct ProcessStates {
    active: ProcessId,
    states: BTreeMap<ProcessId, ProcessState>,
}

impl ProcessStates {
    fn new(root: ProcessState) -> Self {
        let active = root.pid;
        Self {
            active,
            states: BTreeMap::from([(active, root)]),
        }
    }

    pub(crate) fn activate(&mut self, pid: ProcessId) -> Result<(), String> {
        if !self.states.contains_key(&pid) {
            return Err(format!("process state does not exist for PID {pid}"));
        }
        self.active = pid;
        Ok(())
    }

    fn insert(&mut self, state: ProcessState) -> Result<(), String> {
        let pid = state.pid;
        if self.states.insert(pid, state).is_some() {
            return Err(format!("process state already exists for PID {pid}"));
        }
        Ok(())
    }

    fn remove(&mut self, pid: ProcessId) -> Option<ProcessState> {
        self.states.remove(&pid)
    }

    fn current(&self) -> &ProcessState {
        self.states
            .get(&self.active)
            .expect("active process state must exist")
    }

    fn current_mut(&mut self) -> &mut ProcessState {
        self.states
            .get_mut(&self.active)
            .expect("active process state must exist")
    }

    pub(crate) fn set_continuation(
        &mut self,
        pid: ProcessId,
        continuation: Option<crate::exec::ShellContinuation>,
    ) -> Result<(), String> {
        let state = self
            .states
            .get_mut(&pid)
            .ok_or_else(|| format!("process state does not exist for PID {pid}"))?;
        state.shell_continuation = continuation;
        Ok(())
    }

    pub(crate) fn set_deadline_interrupt(
        &mut self,
        pid: ProcessId,
        event: crate::clock::EventId,
    ) -> Result<(), String> {
        let state = self
            .states
            .get_mut(&pid)
            .ok_or_else(|| format!("process state does not exist for PID {pid}"))?;
        state.deadline_interrupt = Some(event);
        Ok(())
    }
}

impl Deref for ProcessStates {
    type Target = ProcessState;

    fn deref(&self) -> &Self::Target {
        self.current()
    }
}

impl DerefMut for ProcessStates {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.current_mut()
    }
}

const MAX_FORK_STATE_BYTES: u64 = 32 * 1024 * 1024;

impl ProcessState {
    fn fork_for_child(
        &self,
        identity: &crate::process::ProcessRecord,
        new_shell: bool,
        fds: FdTable,
        fork_allocation_bytes: u64,
        detached_output: bool,
    ) -> Self {
        Self {
            pid: identity.pid,
            ppid: self.pid,
            process_group: identity.process_group,
            session_id: identity.session_id,
            shell_pid: if new_shell {
                identity.pid
            } else {
                self.shell_pid
            },
            vars: self.vars.clone(),
            arrays: self.arrays.clone(),
            exported: self.exported.clone(),
            cwd: self.cwd.clone(),
            directory_stack: self.directory_stack.clone(),
            funcs: self.funcs.clone(),
            aliases: self.aliases.clone(),
            last_status: self.last_status,
            positional: self.positional.clone(),
            opt_errexit: self.opt_errexit,
            opt_nounset: self.opt_nounset,
            opt_xtrace: self.opt_xtrace,
            opt_pipefail: self.opt_pipefail,
            jobs: Vec::new(),
            next_job_id: 1,
            loop_break: 0,
            loop_continue: 0,
            returning: None,
            exiting: None,
            cond_depth: self.cond_depth,
            deadline_interrupt: self.deadline_interrupt,
            uid: self.uid,
            input_stream: self.input_stream.clone(),
            input_pos: self.input_pos,
            python_repl: None,
            fds,
            shell_continuation: None,
            pending_signals: std::collections::BTreeSet::new(),
            signal_dispositions: if new_shell {
                self.signal_dispositions
                    .iter()
                    .filter(|(_, disposition)| {
                        matches!(disposition, ShellSignalDisposition::Ignore)
                    })
                    .map(|(signal, disposition)| (*signal, disposition.clone()))
                    .collect()
            } else {
                self.signal_dispositions.clone()
            },
            handling_signal: false,
            fork_allocation_bytes,
            detached_output,
        }
    }

    fn fork_memory_bytes(&self) -> u64 {
        let string = |value: &String| (value.len() as u64).saturating_add(24);
        let mut bytes = self
            .vars
            .iter()
            .fold(0_u64, |total, (name, value)| {
                total
                    .saturating_add(string(name))
                    .saturating_add(string(value))
            })
            .saturating_add(
                self.exported
                    .iter()
                    .fold(0, |total, value| total.saturating_add(string(value))),
            )
            .saturating_add(string(&self.cwd))
            .saturating_add(
                self.directory_stack
                    .iter()
                    .fold(0, |total, value| total.saturating_add(string(value))),
            )
            .saturating_add(
                self.positional
                    .iter()
                    .fold(0, |total, value| total.saturating_add(string(value))),
            )
            .saturating_add(self.input_stream.len() as u64);
        bytes = bytes.saturating_add((self.fds.iter().count() as u64).saturating_mul(16));
        for (name, value) in &self.arrays {
            bytes = bytes.saturating_add(string(name));
            bytes = bytes.saturating_add(match value {
                ArrayVal::Indexed(values) => values.iter().fold(0, |total, value| {
                    total.saturating_add(value.as_ref().map_or(0, string))
                }),
                ArrayVal::Assoc(values) => values.iter().fold(0, |total, (name, value)| {
                    total
                        .saturating_add(string(name))
                        .saturating_add(string(value))
                }),
            });
        }
        for (name, body) in &self.funcs {
            bytes = bytes
                .saturating_add(string(name))
                .saturating_add(body.estimated_bytes());
        }
        for (name, alias) in &self.aliases {
            bytes = bytes
                .saturating_add(string(name))
                .saturating_add(string(&alias.source))
                .saturating_add(
                    alias
                        .words
                        .iter()
                        .fold(0, |total, word| total.saturating_add(string(word))),
                );
        }
        for disposition in self.signal_dispositions.values() {
            bytes = bytes.saturating_add(match disposition {
                ShellSignalDisposition::Ignore => 16,
                ShellSignalDisposition::Handler { source, body } => {
                    string(source).saturating_add(body.estimated_bytes())
                }
            });
        }
        bytes
    }
}

impl Deref for Environment {
    type Target = ProcessState;

    fn deref(&self) -> &Self::Target {
        self.process.current()
    }
}

impl DerefMut for Environment {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.process.current_mut()
    }
}

impl Environment {
    pub fn new() -> Self {
        Self::with_limits(Limits::default())
    }

    pub fn with_limits(limits: Limits) -> Self {
        let mut vars: HashMap<String, String> = HashMap::new();
        vars.insert("HOME".into(), "/root".into());
        vars.insert(
            "PATH".into(),
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".into(),
        );
        vars.insert("PWD".into(), "/".into());
        vars.insert("SHELL".into(), "/bin/bash".into());
        vars.insert("TERM".into(), "xterm-256color".into());
        vars.insert("USER".into(), "root".into());
        vars.insert("HOSTNAME".into(), "sandbox".into());
        vars.insert("LANG".into(), "C.UTF-8".into());
        vars.insert("IFS".into(), " \t\n".into());
        let mut exported = std::collections::BTreeSet::new();
        for k in [
            "HOME", "PATH", "PWD", "SHELL", "TERM", "USER", "HOSTNAME", "LANG",
        ] {
            exported.insert(k.to_string());
        }
        let clock = Clock::new();
        let mut descriptors = DescriptorArena::new();
        let mut fds = FdTable::new();
        for (fd, description) in [
            (
                0,
                descriptors
                    .open_input(Vec::new())
                    .expect("stdin descriptor"),
            ),
            (1, descriptors.open_capture().expect("stdout descriptor")),
            (2, descriptors.open_capture().expect("stderr descriptor")),
        ] {
            fds.install(fd, description, &mut descriptors)
                .expect("standard descriptor table");
        }
        let mut vfs = Vfs::with_disk_limit(limits.disk);
        vfs.set_mutation_time(clock.unix_ms());
        vfs.seed_dirs(["/root", "/tmp", "/work"]);
        const ROOT_PID: ProcessId = 1_234;
        let process_environment = exported
            .iter()
            .filter_map(|name| vars.get(name).map(|value| (name.clone(), value.clone())))
            .collect();
        let descriptor_snapshot = fds
            .iter()
            .filter_map(|(fd, id)| descriptors.label(id).ok().map(|label| (fd, label)))
            .collect();
        let mut processes = ProcessTable::new(ROOT_PID, "/".to_string(), process_environment);
        processes.update_descriptors(ROOT_PID, descriptor_snapshot);
        Environment {
            vfs,
            clock,
            net: VirtualNet::new(),
            resources: Resources::new(limits),
            processes,
            terminal: crate::process::ControllingTerminal::new(ROOT_PID),
            scheduler: Scheduler::new(ROOT_PID),
            descriptors,
            process: ProcessStates::new(ProcessState {
                pid: ROOT_PID,
                ppid: 0,
                process_group: ROOT_PID,
                session_id: ROOT_PID,
                shell_pid: ROOT_PID,
                vars,
                arrays: HashMap::new(),
                exported,
                cwd: "/".to_string(),
                directory_stack: Vec::new(),
                funcs: HashMap::new(),
                aliases: HashMap::new(),
                last_status: 0,
                positional: Vec::new(),
                opt_errexit: false,
                opt_nounset: false,
                opt_xtrace: false,
                opt_pipefail: false,
                jobs: Vec::new(),
                next_job_id: 1,
                loop_break: 0,
                loop_continue: 0,
                returning: None,
                exiting: None,
                cond_depth: 0,
                deadline_interrupt: None,
                uid: 0,
                input_stream: Vec::new(),
                input_pos: 0,
                python_repl: None,
                fds,
                shell_continuation: None,
                pending_signals: std::collections::BTreeSet::new(),
                signal_dispositions: BTreeMap::new(),
                handling_signal: false,
                fork_allocation_bytes: 0,
                detached_output: false,
            }),
            live_children: BTreeMap::new(),
            next_temp_id: 0,
            cmd_trace: crate::telemetry::BoundedTextLog::default(),
            unsupported: crate::telemetry::BoundedTextLog::default(),
            invocations: InvocationLog::default(),
            packages: std::collections::BTreeSet::new(),
            pending_stdout: Vec::new(),
            pending_stderr: Vec::new(),
        }
    }

    /// Enter a synchronous logical child while retaining the parent shell state.
    ///
    /// Machine capabilities remain on `Environment`; only process-local state is copied. The
    /// caller must pair a successful call with [`Environment::finish_child`].
    pub(crate) fn start_child(
        &mut self,
        command: &str,
        new_shell: bool,
    ) -> Result<ProcessId, String> {
        self.create_child(
            command,
            new_shell,
            true,
            false,
            crate::process::ChildPlacement::Inherit,
        )
    }

    /// Create a runnable child without blocking or switching away from the active parent.
    pub(crate) fn start_background_child(
        &mut self,
        command: &str,
        new_shell: bool,
    ) -> Result<ProcessId, String> {
        self.create_child(
            command,
            new_shell,
            false,
            true,
            crate::process::ChildPlacement::NewProcessGroup,
        )
    }

    /// Create a runnable pipeline stage whose descriptors are connected before dispatch.
    pub(crate) fn start_pipeline_child(
        &mut self,
        command: &str,
        new_shell: bool,
    ) -> Result<ProcessId, String> {
        self.create_child(
            command,
            new_shell,
            false,
            false,
            crate::process::ChildPlacement::Inherit,
        )
    }

    /// Create a parent-managed live child for a language-level process handle.
    pub(crate) fn start_live_child(
        &mut self,
        command: &str,
        new_shell: bool,
        new_session: bool,
    ) -> Result<ProcessId, String> {
        self.create_child(
            command,
            new_shell,
            false,
            false,
            if new_session {
                crate::process::ChildPlacement::NewSession
            } else {
                crate::process::ChildPlacement::Inherit
            },
        )
    }

    fn create_child(
        &mut self,
        command: &str,
        new_shell: bool,
        foreground: bool,
        detached_output: bool,
        placement: crate::process::ChildPlacement,
    ) -> Result<ProcessId, String> {
        let fork_bytes = self.process.fork_memory_bytes();
        if fork_bytes > MAX_FORK_STATE_BYTES {
            return Err("shell state exceeds the 32 MiB fork limit".to_string());
        }
        let allocation_bytes = fork_bytes.saturating_add(256);
        if !self.resources.reserve_memory(allocation_bytes) {
            return Err("memory limit exceeded while creating child process".to_string());
        }
        let environment: BTreeMap<String, String> = self.child_env().into_iter().collect();
        let mut child_fds = match self.process.fds.fork(&mut self.descriptors) {
            Ok(fds) => fds,
            Err(error) => {
                self.resources.release_memory(allocation_bytes);
                return Err(format!("unable to inherit descriptors: {error:?}"));
            }
        };
        self.processes
            .update_current(self.process.pid, &self.process.cwd, environment.clone());
        let Some(pid) = self.processes.spawn(
            self.process.pid,
            placement,
            command,
            &self.process.cwd,
            environment,
        ) else {
            child_fds.close_all(&mut self.descriptors);
            self.resources.release_memory(allocation_bytes);
            return Err("logical process limit exceeded".to_string());
        };
        if foreground {
            if let Err(error) = self.scheduler.block_current(WaitReason::Child(pid)) {
                child_fds.close_all(&mut self.descriptors);
                self.processes.exit(pid, 125, &self.process.cwd);
                self.processes.reap(pid);
                self.resources.release_memory(allocation_bytes);
                return Err(format!("unable to suspend parent process: {error:?}"));
            }
        }
        if let Err(error) = self.scheduler.spawn(pid) {
            child_fds.close_all(&mut self.descriptors);
            if foreground {
                let _ = self.scheduler.wake(self.process.pid);
                let _ = self.scheduler.dispatch();
            }
            self.processes.exit(pid, 125, &self.process.cwd);
            self.processes.reap(pid);
            self.resources.release_memory(allocation_bytes);
            return Err(format!("unable to schedule child process: {error:?}"));
        }
        let child_record = self
            .processes
            .get(pid)
            .expect("new process record must be retained")
            .clone();
        let child = self.process.fork_for_child(
            &child_record,
            new_shell,
            child_fds,
            allocation_bytes,
            detached_output,
        );
        self.process.insert(child)?;
        self.refresh_descriptor_snapshot(pid);
        if foreground {
            let scheduled = self
                .scheduler
                .dispatch()
                .map_err(|error| format!("unable to dispatch child process: {error:?}"))?
                .ok_or_else(|| "child process was not runnable after creation".to_string())?;
            self.process.activate(scheduled)?;
        }
        Ok(pid)
    }

    /// Replace one descriptor in a retained process before its continuation is dispatched.
    pub(crate) fn install_process_description(
        &mut self,
        pid: ProcessId,
        fd: Fd,
        description: DescriptionId,
    ) -> Result<(), DescriptorError> {
        let state = self
            .process
            .states
            .get_mut(&pid)
            .ok_or(DescriptorError::InvalidFd)?;
        state.fds.install(fd, description, &mut self.descriptors)?;
        self.refresh_descriptor_snapshot(pid);
        Ok(())
    }

    /// Configure cwd and exported environment for a retained child before its first dispatch.
    pub(crate) fn configure_process(
        &mut self,
        pid: ProcessId,
        cwd: Option<String>,
        environment: Option<BTreeMap<String, String>>,
    ) -> Result<(), String> {
        let state = self
            .process
            .states
            .get_mut(&pid)
            .ok_or_else(|| format!("process state does not exist for PID {pid}"))?;
        if let Some(cwd) = cwd {
            state.cwd = cwd;
            state.vars.insert("PWD".to_string(), state.cwd.clone());
            state.exported.insert("PWD".to_string());
        }
        if let Some(environment) = environment {
            state.vars.clear();
            state.arrays.clear();
            state.exported.clear();
            for (name, value) in &environment {
                state.vars.insert(name.clone(), value.clone());
                state.exported.insert(name.clone());
            }
            state.vars.insert("PWD".to_string(), state.cwd.clone());
            state.exported.insert("PWD".to_string());
        }
        let exported = state
            .exported
            .iter()
            .filter_map(|name| {
                state
                    .vars
                    .get(name)
                    .map(|value| (name.clone(), value.clone()))
            })
            .collect();
        self.processes.update_current(pid, &state.cwd, exported);
        Ok(())
    }

    /// Set positional parameters on a retained child without assuming it is currently active.
    pub(crate) fn set_process_positional(
        &mut self,
        pid: ProcessId,
        positional: Vec<String>,
    ) -> Result<(), String> {
        let state = self
            .process
            .states
            .get_mut(&pid)
            .ok_or_else(|| format!("process state does not exist for PID {pid}"))?;
        state.positional = positional;
        Ok(())
    }

    /// Resolve a descriptor belonging to a retained process without activating it.
    pub(crate) fn process_description(
        &self,
        pid: ProcessId,
        fd: Fd,
    ) -> Result<DescriptionId, DescriptorError> {
        self.process
            .states
            .get(&pid)
            .ok_or(DescriptorError::InvalidFd)?
            .fds
            .get(fd)
    }

    /// Restore the parent after a synchronous child and optionally retain the exited record.
    pub(crate) fn finish_child(&mut self, pid: ProcessId, status: i32) {
        let child_deadline_interrupt = self.process.deadline_interrupt;
        let fork_allocation_bytes = self.process.fork_allocation_bytes;
        let parent_pid = self.process.ppid;
        if self.process.detached_output {
            let stdout = self.process.fds.get(1).ok();
            let stderr = self.process.fds.get(2).ok();
            if let Some(stdout) = stdout {
                if let Ok(bytes) = self.descriptors.drain_capture(stdout) {
                    append_pending(&mut self.pending_stdout, &bytes);
                }
            }
            if let Some(stderr) = stderr.filter(|stderr| Some(*stderr) != stdout) {
                if let Ok(bytes) = self.descriptors.drain_capture(stderr) {
                    append_pending(&mut self.pending_stderr, &bytes);
                }
            }
        }
        let endpoints = self
            .process
            .fds
            .iter()
            .filter_map(|(_, description)| {
                self.descriptors.pipe_endpoint(description).ok().flatten()
            })
            .collect::<Vec<_>>();
        self.process.fds.close_all(&mut self.descriptors);
        for (pipe, reader) in endpoints {
            let reason = if reader {
                WaitReason::PipeWritable(pipe)
            } else {
                WaitReason::PipeReadable(pipe)
            };
            self.scheduler.wake_waiters(reason);
        }
        self.processes.update_descriptors(pid, BTreeMap::new());
        self.processes.exit(pid, status, &self.process.cwd);
        if self
            .processes
            .running_group(self.terminal.foreground_group)
            .is_empty()
        {
            self.terminal.foreground_group = self.terminal.session_id;
        }
        self.clock.cancel_task_events(u64::from(pid));
        let _ = self.scheduler.exit_current(status);
        self.scheduler.wake_child_waiters(pid);
        self.scheduler.wake_child_activity_waiters(pid);
        let scheduled = self.scheduler.dispatch().ok().flatten();
        let removed = self.process.remove(pid);
        debug_assert!(removed.is_some(), "finished child state must exist");
        let parent_retained = self.process.states.contains_key(&parent_pid);
        if let Some(parent) = self.process.states.get_mut(&parent_pid) {
            if let Some(job) = parent.jobs.iter_mut().find(|job| job.pid == pid) {
                job.state = JobState::Done(status);
            }
            if child_deadline_interrupt.is_some() {
                parent.deadline_interrupt = child_deadline_interrupt;
            }
        }
        if let Some(scheduled) = scheduled {
            self.process
                .activate(scheduled)
                .expect("scheduled process state must exist");
        }
        if parent_retained {
            let _ = self.send_signal(parent_pid, Signal::Child);
        }
        if !parent_retained && !self.live_children.contains_key(&pid) {
            self.processes.reap(pid);
            let _ = self.scheduler.reap(pid);
        }
        self.resources.release_memory(fork_allocation_bytes);
    }

    /// Queue a supported signal for delivery when the target next reaches a scheduler boundary.
    pub(crate) fn send_signal(&mut self, pid: ProcessId, signal: Signal) -> Result<(), String> {
        if !matches!(
            self.processes.get(pid).map(|record| record.status),
            Some(
                crate::process::ProcessStatus::Running | crate::process::ProcessStatus::Stopped(_)
            )
        ) {
            return Err(format!("process {pid} does not exist"));
        }
        if signal == Signal::Continue {
            self.continue_process(pid)?;
            return Ok(());
        }
        if signal == Signal::Stop && self.scheduler.current() != Some(pid) {
            self.stop_process(pid, signal)?;
            return Ok(());
        }
        if signal == Signal::Kill
            && matches!(
                self.processes.get(pid).map(|record| record.status),
                Some(crate::process::ProcessStatus::Stopped(_))
            )
        {
            self.continue_process(pid)?;
        }
        let target = self
            .process
            .states
            .get_mut(&pid)
            .ok_or_else(|| format!("process {pid} has no execution context"))?;
        target.pending_signals.insert(signal);
        let disposition = target.signal_dispositions.get(&signal);
        let interrupts = matches!(signal, Signal::Kill | Signal::Stop)
            || matches!(disposition, Some(ShellSignalDisposition::Handler { .. }))
            || (disposition.is_none() && signal.terminates());
        if interrupts
            && matches!(
                self.scheduler.state(pid),
                Some(crate::scheduler::TaskState::Blocked(_))
            )
        {
            self.scheduler
                .wake(pid)
                .map_err(|error| format!("unable to wake process {pid}: {error:?}"))?;
        }
        Ok(())
    }

    /// Queue a signal for every running member of one modeled process group.
    pub(crate) fn send_signal_group(
        &mut self,
        process_group: ProcessId,
        signal: Signal,
    ) -> Result<(), String> {
        let members = self.processes.live_group(process_group);
        if members.is_empty() {
            return Err(format!("process group {process_group} does not exist"));
        }
        for pid in members {
            self.send_signal(pid, signal)?;
        }
        Ok(())
    }

    /// Queue a terminal-generated signal for the current foreground process group.
    pub(crate) fn send_terminal_signal(&mut self, signal: Signal) -> Result<(), String> {
        self.send_signal_group(self.terminal.foreground_group, signal)
    }

    /// Transfer synthetic terminal foreground ownership to a live group in this session.
    pub(crate) fn set_terminal_foreground(
        &mut self,
        process_group: ProcessId,
    ) -> Result<(), String> {
        self.terminal.set_foreground(&self.processes, process_group)
    }

    fn stop_process(&mut self, pid: ProcessId, signal: Signal) -> Result<(), String> {
        self.scheduler
            .stop(pid)
            .map_err(|error| format!("unable to stop process {pid}: {error:?}"))?;
        self.processes.stop(pid, signal);
        for state in self.process.states.values_mut() {
            if let Some(job) = state.jobs.iter_mut().find(|job| job.pid == pid) {
                job.state = JobState::Stopped(signal);
            }
        }
        self.scheduler.wake_child_waiters(pid);
        self.scheduler.wake_child_activity_waiters(pid);
        if self
            .processes
            .running_group(self.terminal.foreground_group)
            .is_empty()
        {
            self.terminal.foreground_group = self.terminal.session_id;
        }
        Ok(())
    }

    fn continue_process(&mut self, pid: ProcessId) -> Result<(), String> {
        self.scheduler
            .continue_task(pid)
            .map_err(|error| format!("unable to continue process {pid}: {error:?}"))?;
        self.processes.continue_process(pid);
        let has_pending_signal = self
            .process
            .states
            .get(&pid)
            .is_some_and(|state| !state.pending_signals.is_empty());
        if has_pending_signal
            && matches!(
                self.scheduler.state(pid),
                Some(crate::scheduler::TaskState::Blocked(_))
            )
        {
            self.scheduler
                .wake(pid)
                .map_err(|error| format!("unable to wake continued process {pid}: {error:?}"))?;
        }
        for state in self.process.states.values_mut() {
            if let Some(job) = state.jobs.iter_mut().find(|job| job.pid == pid) {
                job.state = JobState::Running;
            }
        }
        Ok(())
    }

    /// Stop the active process at a scheduler boundary and select no replacement itself.
    pub(crate) fn stop_active_process(&mut self, signal: Signal) -> Result<(), String> {
        let pid = self.process.pid;
        self.stop_process(pid, signal)
    }

    /// Select the next non-ignored signal action for the active process.
    pub(crate) fn take_signal_delivery(&mut self) -> Option<SignalDelivery> {
        loop {
            let signal = if self.process.handling_signal {
                [Signal::Kill, Signal::Stop]
                    .into_iter()
                    .find(|signal| self.process.pending_signals.contains(signal))?
            } else {
                self.process.pending_signals.iter().next().copied()?
            };
            self.process.pending_signals.remove(&signal);
            if signal == Signal::Kill {
                self.clock.cancel_task_events(u64::from(self.process.pid));
                return Some(SignalDelivery::Terminate(signal));
            }
            if signal == Signal::Stop {
                return Some(SignalDelivery::Stop(signal));
            }
            match self.process.signal_dispositions.get(&signal).cloned() {
                Some(ShellSignalDisposition::Ignore) => continue,
                Some(ShellSignalDisposition::Handler { body, .. }) => {
                    self.process.handling_signal = true;
                    return Some(SignalDelivery::Handler(body));
                }
                None if signal.terminates() => {
                    self.clock.cancel_task_events(u64::from(self.process.pid));
                    return Some(SignalDelivery::Terminate(signal));
                }
                None => continue,
            }
        }
    }

    /// Mark the active shell's caught-signal handler complete.
    pub(crate) fn finish_signal_handler(&mut self) {
        self.process.handling_signal = false;
    }

    /// Roll back a background child whose process-local setup failed before dispatch.
    pub(crate) fn cancel_unstarted_child(&mut self, pid: ProcessId) {
        if let Some(mut child) = self.process.remove(pid) {
            child.fds.close_all(&mut self.descriptors);
            self.resources.release_memory(child.fork_allocation_bytes);
        }
        let _ = self.scheduler.discard_runnable(pid);
        self.processes.exit(pid, 125, &self.process.cwd);
        self.processes.reap(pid);
    }

    /// Synchronize the active descriptor table into generated process metadata.
    pub(crate) fn refresh_descriptor_snapshot(&mut self, pid: ProcessId) {
        let descriptors = self
            .process
            .states
            .get(&pid)
            .into_iter()
            .flat_map(|state| state.fds.iter())
            .filter_map(|(fd, id)| self.descriptors.label(id).ok().map(|label| (fd, label)))
            .collect();
        self.processes.update_descriptors(pid, descriptors);
    }

    /// Install a newly allocated open description, discarding it if the process table rejects
    /// the requested descriptor number.
    pub(crate) fn install_new_description(
        &mut self,
        fd: Fd,
        description: DescriptionId,
    ) -> Result<(), DescriptorError> {
        if let Err(error) = self
            .process
            .fds
            .install(fd, description, &mut self.descriptors)
        {
            let _ = self.descriptors.discard_unreferenced(description);
            return Err(error);
        }
        self.refresh_descriptor_snapshot(self.process.pid);
        Ok(())
    }

    /// Read from an active process descriptor without granting access to host handles.
    pub(crate) fn read_fd(&mut self, fd: Fd, maximum: usize) -> Result<IoPoll<Vec<u8>>, String> {
        let description = self.process.fds.get(fd).map_err(descriptor_message)?;
        if let Some(file) = self
            .descriptors
            .file_state(description)
            .map_err(descriptor_message)?
        {
            if !file.readable {
                return Err("descriptor is not open for reading".to_string());
            }
            let cursor = usize::try_from(file.cursor)
                .map_err(|_| "file cursor exceeds addressable memory".to_string())?;
            let data = self
                .fs_read_limited("/", &file.path, MAX_CAPTURE_BYTES)
                .map_err(|error| error.to_string())?;
            let end = cursor.saturating_add(maximum).min(data.len());
            let bytes = if cursor >= data.len() {
                Vec::new()
            } else {
                data[cursor..end].to_vec()
            };
            self.descriptors
                .advance_file(description, bytes.len())
                .map_err(descriptor_message)?;
            return Ok(IoPoll::Ready(bytes));
        }
        let maximum = if self
            .descriptors
            .is_generated_device(description)
            .map_err(descriptor_message)?
        {
            let maximum = maximum
                .min(crate::descriptors::DEVICE_READ_QUANTUM)
                .min(usize::try_from(self.resources.cpu_remaining() / 2).unwrap_or(usize::MAX));
            if maximum == 0 {
                let _ = self.resources.charge_cpu(1);
                return Err(self.resources.stop_reason().map_or_else(
                    || "device read limit exceeded".to_string(),
                    |reason| reason.to_string(),
                ));
            }
            if !self.resources.charge_cpu(maximum as u64) {
                return Err(self.resources.stop_reason().map_or_else(
                    || "device read limit exceeded".to_string(),
                    |reason| reason.to_string(),
                ));
            }
            maximum
        } else {
            maximum
        };
        let result = self
            .descriptors
            .read(description, maximum)
            .map_err(descriptor_message)?;
        if matches!(result, IoPoll::Ready(_)) {
            if let Some((pipe, true)) = self
                .descriptors
                .pipe_endpoint(description)
                .map_err(descriptor_message)?
            {
                self.scheduler.wake_waiters(WaitReason::PipeWritable(pipe));
                // A Python parent may be coordinating several child pipes through one
                // `communicate()` operation. Child activity is its retry signal; completion is
                // still checked by the process layer before the call returns.
                self.wake_child_activity_waiters(self.process.pid);
            }
        }
        Ok(result)
    }

    /// Write to an active process descriptor, routing file effects only through the VFS.
    pub(crate) fn write_fd(&mut self, fd: Fd, bytes: &[u8]) -> Result<IoPoll<usize>, String> {
        let description = self.process.fds.get(fd).map_err(descriptor_message)?;
        if let Some(file) = self
            .descriptors
            .file_state(description)
            .map_err(descriptor_message)?
        {
            if !file.writable {
                return Err("descriptor is not open for writing".to_string());
            }
            let cursor = usize::try_from(file.cursor)
                .map_err(|_| "file cursor exceeds addressable memory".to_string())?;
            self.sync_vfs_time();
            if let Err(error) = self.vfs.write_at("/", &file.path, cursor, bytes) {
                if file.remove_on_first_write_error && cursor == 0 {
                    let _ = self.vfs.remove_file("/", &file.path);
                }
                return Err(error.to_string());
            }
            self.descriptors
                .advance_file(description, bytes.len())
                .map_err(descriptor_message)?;
            return Ok(IoPoll::Ready(bytes.len()));
        }
        let result = self
            .descriptors
            .write(description, bytes)
            .map_err(descriptor_message)?;
        if matches!(result, IoPoll::Ready(_)) {
            if let Some((pipe, false)) = self
                .descriptors
                .pipe_endpoint(description)
                .map_err(descriptor_message)?
            {
                self.scheduler.wake_waiters(WaitReason::PipeReadable(pipe));
                self.wake_child_activity_waiters(self.process.pid);
            }
        }
        Ok(result)
    }

    /// Wake coordinators of this process or any enclosing modeled child boundary.
    ///
    /// A launched argv may itself enter a shell child before touching inherited pipes. Walking the
    /// bounded process ancestry ensures a Python `communicate()` waiting on the original handle
    /// observes descendant I/O without teaching the scheduler about descriptor ownership.
    fn wake_child_activity_waiters(&mut self, mut pid: ProcessId) {
        for _ in 0..crate::process::MAX_PROCESSES {
            self.scheduler.wake_child_activity_waiters(pid);
            let Some(parent) = self.processes.get(pid).map(|record| record.ppid) else {
                break;
            };
            if parent == pid {
                break;
            }
            pid = parent;
        }
    }

    /// Suspend the active task on the exact readiness condition returned by descriptor I/O.
    pub fn block_on_io(&mut self, wait: IoWait) -> Result<(), String> {
        let reason = match wait {
            IoWait::InputReadable(description) => WaitReason::InputReadable(description),
            IoWait::PipeReadable(pipe) => WaitReason::PipeReadable(pipe),
            IoWait::PipeWritable(pipe) => WaitReason::PipeWritable(pipe),
        };
        self.scheduler
            .block_current(reason)
            .map(|_| ())
            .map_err(|error| format!("unable to block process on descriptor: {error:?}"))
    }

    /// Record a package name installed by a compatibility command.
    pub fn install_package(&mut self, import_name: &str) {
        let name = import_name.trim();
        if name.is_empty() {
            return;
        }
        self.packages.insert(name.to_string());
        let deps: &[&str] = match name {
            "pandas" => &["numpy"],
            "scipy" => &["numpy"],
            "sklearn" => &["numpy", "scipy"],
            _ => &[],
        };
        for d in deps {
            self.packages.insert((*d).to_string());
        }
    }

    /// Make the wall-clock/VFS boundary explicit immediately before a filesystem effect.
    pub fn sync_vfs_time(&mut self) {
        self.vfs.set_mutation_time(self.clock.unix_ms());
    }

    /// Read through the generated pseudo-filesystem before consulting persistent VFS state.
    pub fn fs_read(&self, cwd: &str, path: &str) -> crate::vfs::Result<Vec<u8>> {
        crate::pseudo_fs::read(self, cwd, path).unwrap_or_else(|| self.vfs.read(cwd, path))
    }

    /// Return a generated or persistent file length without exposing its backing implementation.
    pub fn fs_file_len(&self, cwd: &str, path: &str) -> crate::vfs::Result<usize> {
        if let Some(result) = crate::pseudo_fs::read(self, cwd, path) {
            result.map(|data| data.len())
        } else {
            self.vfs.file_len(cwd, path)
        }
    }

    /// Read a finite generated or persistent file after enforcing a caller-provided bound.
    pub fn fs_read_limited(
        &self,
        cwd: &str,
        path: &str,
        limit: usize,
    ) -> crate::vfs::Result<Vec<u8>> {
        if let Some(result) = crate::pseudo_fs::read(self, cwd, path) {
            let data = result?;
            if data.len() > limit {
                return Err(crate::vfs::VfsError::TooLarge {
                    path: path.to_string(),
                    limit,
                });
            }
            Ok(data)
        } else {
            self.vfs.read_limited(cwd, path, limit)
        }
    }

    /// Inspect generated or persistent filesystem metadata through one capability boundary.
    pub fn fs_metadata(
        &self,
        cwd: &str,
        path: &str,
        follow: bool,
    ) -> crate::vfs::Result<crate::vfs::Node> {
        crate::pseudo_fs::metadata(self, cwd, path, follow)
            .unwrap_or_else(|| self.vfs.metadata(cwd, path, follow))
    }

    /// List a generated or persistent directory.
    pub fn fs_list_dir(&self, cwd: &str, path: &str) -> crate::vfs::Result<Vec<String>> {
        crate::pseudo_fs::list_dir(self, cwd, path).unwrap_or_else(|| self.vfs.list_dir(cwd, path))
    }

    /// Read a generated or persistent symbolic link.
    pub fn fs_read_link(&self, cwd: &str, path: &str) -> crate::vfs::Result<String> {
        crate::pseudo_fs::read_link(self, cwd, path)
            .unwrap_or_else(|| self.vfs.read_link(cwd, path))
    }

    pub fn get_var(&self, name: &str) -> Option<String> {
        match name {
            "?" => Some(self.last_status.to_string()),
            "$" => Some(self.shell_pid.to_string()),
            "PPID" => Some(self.ppid.to_string()),
            "BASHPID" => Some(self.pid.to_string()),
            "#" => Some(self.positional.len().to_string()),
            "PWD" => Some(self.cwd.clone()),
            "@" | "*" => Some(self.positional.join(" ")),
            _ => {
                if let Ok(n) = name.parse::<usize>() {
                    if n == 0 {
                        return Some("shellsim".to_string());
                    }
                    return self.positional.get(n - 1).cloned();
                }
                // A bare reference to an array name yields element 0 (`$arr` == `${arr[0]}`).
                if self.arrays.contains_key(name) {
                    return Some(self.array_get(name, "0").unwrap_or_default());
                }
                self.vars.get(name).cloned()
            }
        }
    }

    pub fn set_var(&mut self, name: &str, val: impl Into<String>) {
        let val = val.into();
        if name == "PWD" {
            self.cwd = val.clone();
        }
        // A plain scalar assignment to an array name in bash sets element 0; we instead treat it
        // as a fresh scalar (drop the array) — the common case in our scripts and lower-risk.
        self.arrays.remove(name);
        self.vars.insert(name.to_string(), val);
    }

    pub fn export(&mut self, name: &str) {
        self.exported.insert(name.to_string());
    }

    // ===================== arrays =====================

    pub fn is_array(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }

    /// Ensure `name` exists as an *indexed* array. If it was a plain scalar, bash promotes it
    /// so that the old scalar becomes element 0 (`x=1; x[2]=3` ⇒ x[0]=1).
    fn ensure_indexed(&mut self, name: &str) {
        if !self.arrays.contains_key(name) {
            let mut v: Vec<Option<String>> = Vec::new();
            if let Some(s) = self.vars.get(name).cloned() {
                v.push(Some(s));
            }
            self.arrays.insert(name.to_string(), ArrayVal::Indexed(v));
        }
    }

    /// Ensure `name` exists as an *associative* array (created empty if absent).
    pub fn declare_assoc(&mut self, name: &str) {
        if !matches!(self.arrays.get(name), Some(ArrayVal::Assoc(_))) {
            self.arrays
                .insert(name.to_string(), ArrayVal::Assoc(BTreeMap::new()));
        }
    }

    /// Ensure `name` exists as an indexed array (created empty if absent).
    pub fn declare_indexed(&mut self, name: &str) {
        self.ensure_indexed(name);
    }

    /// Replace the whole array at `name` with the given element list (indexed).
    pub fn set_array(&mut self, name: &str, elems: Vec<String>) {
        self.vars.remove(name);
        self.arrays.insert(
            name.to_string(),
            ArrayVal::Indexed(elems.into_iter().map(Some).collect()),
        );
    }

    /// Append elements to the end of an indexed array (creating/promoting as needed). For an
    /// associative array, callers should use `array_set` per key; this is indexed-only.
    pub fn array_append(&mut self, name: &str, elems: Vec<String>) {
        self.ensure_indexed(name);
        if let Some(ArrayVal::Indexed(v)) = self.arrays.get_mut(name) {
            for e in elems {
                v.push(Some(e));
            }
        }
    }

    /// Assign `value` to a subscript. For an associative array `key` is the literal key; for an
    /// indexed array `key` is parsed as an integer index (sparse — gaps become `None`).
    pub fn array_set(&mut self, name: &str, key: &str, value: String) {
        match self.arrays.get_mut(name) {
            Some(ArrayVal::Assoc(m)) => {
                m.insert(key.to_string(), value);
            }
            _ => {
                self.ensure_indexed(name);
                if let Some(ArrayVal::Indexed(v)) = self.arrays.get_mut(name) {
                    let idx = key.trim().parse::<usize>().unwrap_or(0);
                    if idx >= v.len() {
                        v.resize(idx + 1, None);
                    }
                    v[idx] = Some(value);
                }
            }
        }
    }

    /// Look up one subscript value.
    pub fn array_get(&self, name: &str, key: &str) -> Option<String> {
        match self.arrays.get(name) {
            Some(ArrayVal::Assoc(m)) => m.get(key).cloned(),
            Some(ArrayVal::Indexed(v)) => {
                let idx = key.trim().parse::<usize>().ok()?;
                v.get(idx).and_then(|o| o.clone())
            }
            None => None,
        }
    }

    /// All set values in order (`${arr[@]}` / `${arr[*]}`).
    pub fn array_all(&self, name: &str) -> Vec<String> {
        match self.arrays.get(name) {
            Some(ArrayVal::Assoc(m)) => m.values().cloned().collect(),
            Some(ArrayVal::Indexed(v)) => v.iter().filter_map(|o| o.clone()).collect(),
            None => Vec::new(),
        }
    }

    /// Keys/indices of set elements (`${!arr[@]}`).
    pub fn array_keys(&self, name: &str) -> Vec<String> {
        match self.arrays.get(name) {
            Some(ArrayVal::Assoc(m)) => m.keys().cloned().collect(),
            Some(ArrayVal::Indexed(v)) => v
                .iter()
                .enumerate()
                .filter_map(|(i, o)| o.as_ref().map(|_| i.to_string()))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Count of set elements (`${#arr[@]}`).
    pub fn array_len(&self, name: &str) -> usize {
        match self.arrays.get(name) {
            Some(ArrayVal::Assoc(m)) => m.len(),
            Some(ArrayVal::Indexed(v)) => v.iter().filter(|o| o.is_some()).count(),
            None => 0,
        }
    }

    /// Unset one subscript (an element); returns true if the name remained an array.
    pub fn array_unset_elem(&mut self, name: &str, key: &str) {
        match self.arrays.get_mut(name) {
            Some(ArrayVal::Assoc(m)) => {
                m.remove(key);
            }
            Some(ArrayVal::Indexed(v)) => {
                if let Ok(idx) = key.trim().parse::<usize>() {
                    if idx < v.len() {
                        v[idx] = None;
                    }
                }
            }
            None => {}
        }
    }

    /// Environment map visible to a child process (e.g. the python engine).
    pub fn child_env(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();
        for k in &self.exported {
            if let Some(v) = self.vars.get(k) {
                env.insert(k.clone(), v.clone());
            }
        }
        env.insert("PWD".to_string(), self.cwd.clone());
        env
    }

    pub fn new_job(&mut self, pid: ProcessId, cmd: String) -> Option<u32> {
        if self.jobs.len() >= crate::process::MAX_PROCESSES {
            return None;
        }
        let id = self.next_job_id;
        self.next_job_id = self.next_job_id.checked_add(1)?;
        self.jobs.push(Job {
            id,
            pid,
            cmd,
            state: JobState::Running,
        });
        Some(id)
    }

    pub(crate) fn next_temp_id(&mut self) -> Option<u64> {
        let id = self.next_temp_id;
        self.next_temp_id = self.next_temp_id.checked_add(1)?;
        Some(id)
    }

    pub fn note_unsupported(&mut self, what: &str) {
        self.unsupported.record(what);
    }

    pub fn outcome(&self, exit_status: i32) -> RunOutcome {
        self.resources
            .outcome(exit_status, self.vfs.disk_used(), self.vfs.disk_peak())
    }

    /// Whether this persistent shell can accept another action.
    pub fn is_terminated(&self) -> bool {
        self.resources.is_stopped() || self.exiting.is_some()
    }

    /// Whether subsequent session actions are currently interpreted by the minimal Python REPL.
    pub fn in_python_repl(&self) -> bool {
        self.python_repl.is_some()
    }

    /// Sticky terminal status after `exit`, `set -e`, or resource exhaustion.
    pub fn termination_status(&self) -> Option<i32> {
        self.resources
            .stop_reason()
            .map(crate::resources::StopReason::exit_status)
            .or(self.exiting)
    }
}

fn descriptor_message(error: DescriptorError) -> String {
    match error {
        DescriptorError::InvalidFd => "bad file descriptor",
        DescriptorError::WrongAccess => "descriptor is not open for that operation",
        DescriptorError::DescriptorLimit => "process descriptor limit exceeded",
        DescriptorError::DescriptionLimit => "open-description limit exceeded",
        DescriptorError::PipeLimit => "pipe limit exceeded",
        DescriptorError::InvalidPipeCapacity => "invalid pipe capacity",
        DescriptorError::BrokenPipe => "broken pipe",
        DescriptorError::OutputLimit => "descriptor output limit exceeded",
        DescriptorError::ReferenceOverflow => "descriptor reference limit exceeded",
    }
    .to_string()
}

fn append_pending(destination: &mut Vec<u8>, bytes: &[u8]) {
    let available = MAX_CAPTURE_BYTES.saturating_sub(destination.len());
    destination.extend_from_slice(&bytes[..bytes.len().min(available)]);
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

/// Compatibility name for callers written against the original shell simulator API.
pub type Interp = Environment;

#[cfg(test)]
mod process_state_tests {
    use super::*;

    #[test]
    fn inactive_parent_context_remains_machine_owned_while_child_runs() {
        let mut environment = Environment::new();
        let root = environment.pid;
        environment.set_var("scope", "parent");

        let child = environment.start_child("probe", false).unwrap();
        assert_eq!(environment.process.active, child);
        assert_eq!(environment.process.states.len(), 2);
        assert_eq!(
            environment.process.states[&root]
                .vars
                .get("scope")
                .map(String::as_str),
            Some("parent")
        );
        environment.set_var("scope", "child");

        environment.finish_child(child, 0);
        environment.processes.reap(child);
        environment.scheduler.reap(child).unwrap();
        assert_eq!(environment.process.active, root);
        assert_eq!(environment.process.states.len(), 1);
        assert_eq!(environment.get_var("scope").as_deref(), Some("parent"));
    }

    #[test]
    fn pipe_io_wakes_the_exact_blocked_descriptor_waiter() {
        let mut environment = Environment::new();
        let (reader, writer) = environment.descriptors.open_pipe(8).unwrap();
        environment.install_new_description(0, reader).unwrap();
        environment.install_new_description(1, writer).unwrap();

        let IoPoll::Blocked(wait) = environment.read_fd(0, 1).unwrap() else {
            panic!("empty pipe with a live writer must block");
        };
        environment.block_on_io(wait).unwrap();
        assert_eq!(
            environment.scheduler.state(environment.pid),
            Some(crate::scheduler::TaskState::Blocked(
                WaitReason::PipeReadable(1)
            ))
        );
        assert_eq!(environment.write_fd(1, b"x").unwrap(), IoPoll::Ready(1));
        assert_eq!(
            environment.scheduler.state(environment.pid),
            Some(crate::scheduler::TaskState::Runnable)
        );
    }
}
