//! Interpreter state shared across the shell executor and all commands.

use std::cell::Cell;
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

/// Cursor retained between calls to the `getopts` shell builtin.
#[derive(Clone, Debug)]
pub(crate) struct GetoptsState {
    pub(crate) optind: usize,
    pub(crate) offset: usize,
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

#[derive(Clone)]
struct LocalBinding {
    scalar: Option<String>,
    array: Option<ArrayVal>,
    exported: bool,
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

/// A failed shell expansion with the status and unwinding behavior required by its cause.
#[derive(Clone, Debug)]
pub(crate) struct ShellExpansionError {
    pub(crate) message: String,
    pub(crate) status: i32,
    pub(crate) abort_shell: bool,
}

impl ShellExpansionError {
    /// Parameter expansion errors abort a non-interactive shell with Bash-compatible status.
    pub(crate) fn parameter(message: String) -> Self {
        Self {
            message,
            status: 1,
            abort_shell: true,
        }
    }

    /// Assignment errors fail their simple command without unconditionally exiting the shell.
    pub(crate) fn assignment(message: String) -> Self {
        Self {
            message,
            status: 1,
            abort_shell: false,
        }
    }
}

/// Machine-wide state and limits shared by cooperatively scheduled logical processes.
#[derive(Clone)]
pub struct Environment {
    pub vfs: Vfs,
    /// Machine hostname, independent of process-local shell variables.
    pub(crate) hostname: String,
    pub clock: Clock,
    /// Virtual framebuffer and input queue; guests never receive host device handles.
    pub display: crate::display::VirtualDisplay,
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
    /// Variable names protected by the `readonly` builtin.
    pub readonly: std::collections::BTreeSet<String>,
    /// Process-local file creation mask. Creation sites apply this value explicitly.
    pub umask: u16,
    /// Compatibility options recorded by `shopt`; options with behavioral effects remain
    /// rejected until their corresponding parser or expansion support exists.
    pub shell_options: std::collections::BTreeSet<String>,
    /// Successful executable resolutions retained by the `hash` builtin.
    pub command_hash: BTreeMap<String, String>,
    pub cwd: String,
    /// Bash-style directory stack, stored oldest-to-newest beneath the current directory.
    pub directory_stack: Vec<String>,
    pub funcs: HashMap<String, crate::shell::Node>,
    /// Saved bindings for each active shell-function scope.
    local_scopes: Vec<HashMap<String, LocalBinding>>,
    /// Process-local simple-command aliases expanded before normal command dispatch.
    pub(crate) aliases: HashMap<String, AliasDefinition>,
    /// `$?`
    pub last_status: i32,
    /// Deterministic state for Bash's special `$RANDOM` parameter.
    random_state: Cell<u32>,
    /// Process-local entropy for guest and native virtual-kernel random calls.
    syscall_random_state: u64,
    /// `$0`: the shell or script name given when a shell image was loaded.
    pub(crate) arg0: String,
    /// positional parameters `$1 $2 ... $@`
    pub positional: Vec<String>,
    pub(crate) getopts: GetoptsState,
    /// `set -e` / `set -u` / `set -x`
    pub opt_errexit: bool,
    pub opt_nounset: bool,
    pub opt_xtrace: bool,
    /// Diagnostic and control-flow effect raised while expanding the current shell word.
    pub(crate) expansion_error: Option<ShellExpansionError>,
    /// `set -o pipefail`
    pub opt_pipefail: bool,
    pub jobs: Vec<Job>,
    next_job_id: u32,
    /// Recursion and loop-control signaling.
    pub loop_break: u32,
    pub loop_continue: u32,
    /// Number of active shell loops, used to reject loop-control builtins out of context.
    pub loop_depth: u32,
    pub returning: Option<i32>,
    /// Active sourced-script boundaries that may consume the `return` builtin.
    pub(crate) source_depth: u32,
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
    /// Resumable program image retained across scheduler activations.
    pub(crate) program: Option<crate::program::ProgramContinuation>,
    /// Coalesced standard signals awaiting delivery at a scheduler boundary.
    pending_signals: std::collections::BTreeSet<Signal>,
    /// Non-default signal actions installed by the shell `trap` builtin.
    pub(crate) signal_dispositions: BTreeMap<Signal, ShellSignalDisposition>,
    /// Process-local pseudo-signal action run once when the current shell script finishes.
    pub(crate) exit_disposition: Option<ShellSignalDisposition>,
    /// Prevent ordinary caught signals from recursively interrupting their own handler.
    handling_signal: bool,
    /// Memory reserved for this forked context and released independently at exit.
    fork_allocation_bytes: u64,
    detached_output: bool,
}

/// Program, parameters, and options of a shell loaded as a process image.
#[derive(Default)]
struct ShellImage {
    /// Parsed `-c` or script source; `None` reads the program from standard input.
    program: Option<crate::shell::Node>,
    arg0: String,
    positional: Vec<String>,
    errexit: bool,
    nounset: bool,
    xtrace: bool,
    pipefail: bool,
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
        self.set_program(
            pid,
            continuation.map(crate::program::ProgramContinuation::Shell),
        )
    }

    pub(crate) fn set_program(
        &mut self,
        pid: ProcessId,
        program: Option<crate::program::ProgramContinuation>,
    ) -> Result<(), String> {
        let state = self
            .states
            .get_mut(&pid)
            .ok_or_else(|| format!("process state does not exist for PID {pid}"))?;
        state.program = program;
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
    /// Fill a caller-metered buffer from deterministic process-local virtual entropy.
    pub(crate) fn fill_virtual_random(&mut self, bytes: &mut [u8]) {
        for byte in bytes {
            self.syscall_random_state = self
                .syscall_random_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            *byte = (self.syscall_random_state >> 32) as u8;
        }
    }

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
            readonly: self.readonly.clone(),
            umask: self.umask,
            shell_options: self.shell_options.clone(),
            command_hash: self.command_hash.clone(),
            cwd: self.cwd.clone(),
            directory_stack: self.directory_stack.clone(),
            funcs: self.funcs.clone(),
            local_scopes: self.local_scopes.clone(),
            aliases: self.aliases.clone(),
            last_status: self.last_status,
            random_state: Cell::new(self.random_state.get()),
            syscall_random_state: self.syscall_random_state,
            arg0: self.arg0.clone(),
            positional: self.positional.clone(),
            getopts: self.getopts.clone(),
            opt_errexit: self.opt_errexit,
            opt_nounset: self.opt_nounset,
            opt_xtrace: self.opt_xtrace,
            expansion_error: None,
            opt_pipefail: self.opt_pipefail,
            jobs: Vec::new(),
            next_job_id: 1,
            loop_break: 0,
            loop_continue: 0,
            loop_depth: if new_shell { 0 } else { self.loop_depth },
            returning: None,
            source_depth: 0,
            exiting: None,
            cond_depth: self.cond_depth,
            deadline_interrupt: self.deadline_interrupt,
            uid: self.uid,
            input_stream: self.input_stream.clone(),
            input_pos: self.input_pos,
            python_repl: None,
            fds,
            program: None,
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
            exit_disposition: if new_shell {
                None
            } else {
                self.exit_disposition.clone()
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
        for name in &self.readonly {
            bytes = bytes.saturating_add(string(name));
        }
        for name in &self.shell_options {
            bytes = bytes.saturating_add(string(name));
        }
        for (name, path) in &self.command_hash {
            bytes = bytes
                .saturating_add(string(name))
                .saturating_add(string(path));
        }
        for scope in &self.local_scopes {
            for (name, binding) in scope {
                bytes = bytes.saturating_add(string(name));
                bytes = bytes.saturating_add(binding.scalar.as_ref().map_or(0, string));
                if let Some(array) = &binding.array {
                    bytes = bytes.saturating_add(match array {
                        ArrayVal::Indexed(values) => values.iter().fold(0, |total, value| {
                            total.saturating_add(value.as_ref().map_or(0, string))
                        }),
                        ArrayVal::Assoc(values) => values.iter().fold(0, |total, (key, value)| {
                            total
                                .saturating_add(string(key))
                                .saturating_add(string(value))
                        }),
                    });
                }
            }
        }
        bytes = bytes.saturating_add(
            self.expansion_error
                .as_ref()
                .map_or(0, |error| string(&error.message)),
        );
        for disposition in self.signal_dispositions.values() {
            bytes = bytes.saturating_add(match disposition {
                ShellSignalDisposition::Ignore => 16,
                ShellSignalDisposition::Handler { source, body } => {
                    string(source).saturating_add(body.estimated_bytes())
                }
            });
        }
        if let Some(disposition) = &self.exit_disposition {
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

    /// Inject a key transition into the virtual display without consulting host input.
    pub fn inject_key(
        &mut self,
        event: crate::display::KeyEvent,
    ) -> Result<(), crate::display::DisplayError> {
        self.display.inject_key(&mut self.resources, event)
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
        vfs.seed_dirs(["/root", "/tmp", "/work", "/usr", "/usr/bin"]);
        for (name, program) in [
            ("true", crate::vfs::NativeProgram::True),
            ("false", crate::vfs::NativeProgram::False),
            ("pwd", crate::vfs::NativeProgram::Pwd),
            ("yes", crate::vfs::NativeProgram::Yes),
            ("cat", crate::vfs::NativeProgram::Cat),
            ("tee", crate::vfs::NativeProgram::Tee),
            ("head", crate::vfs::NativeProgram::Head),
            ("xargs", crate::vfs::NativeProgram::Xargs),
            ("find", crate::vfs::NativeProgram::Find),
            ("env", crate::vfs::NativeProgram::Env),
            ("sleep", crate::vfs::NativeProgram::Sleep),
            ("usleep", crate::vfs::NativeProgram::Usleep),
            ("timeout", crate::vfs::NativeProgram::Timeout),
            ("nohup", crate::vfs::NativeProgram::Nohup),
            ("nice", crate::vfs::NativeProgram::Nice),
            ("make", crate::vfs::NativeProgram::Make),
            // Shells parse and run their own program; see `Interp::shell_image`.
            ("sh", crate::vfs::NativeProgram::LegacyRegistered("sh")),
            ("bash", crate::vfs::NativeProgram::LegacyRegistered("bash")),
            ("dash", crate::vfs::NativeProgram::LegacyRegistered("dash")),
            ("zsh", crate::vfs::NativeProgram::LegacyRegistered("zsh")),
        ] {
            vfs.seed_native_executable(&format!("/usr/bin/{name}"), program);
        }
        for (path, image) in crate::commands::registered_executables() {
            if matches!(
                image,
                crate::vfs::NativeProgram::LegacyRegistered(name)
                    if crate::vfs::NativeProgram::from_name(name).is_some()
            ) {
                continue;
            }
            vfs.seed_native_executable(&path, image);
        }
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
            hostname: "sandbox".to_string(),
            clock,
            display: crate::display::VirtualDisplay::default(),
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
                readonly: std::collections::BTreeSet::new(),
                umask: 0o022,
                shell_options: std::collections::BTreeSet::new(),
                command_hash: BTreeMap::new(),
                cwd: "/".to_string(),
                directory_stack: Vec::new(),
                funcs: HashMap::new(),
                local_scopes: Vec::new(),
                aliases: HashMap::new(),
                last_status: 0,
                random_state: Cell::new(1),
                syscall_random_state: 0x5eed_5eed_5eed_5eed,
                arg0: "shellsim".to_string(),
                positional: Vec::new(),
                getopts: GetoptsState {
                    optind: 1,
                    offset: 1,
                },
                opt_errexit: false,
                opt_nounset: false,
                opt_xtrace: false,
                expansion_error: None,
                opt_pipefail: false,
                jobs: Vec::new(),
                next_job_id: 1,
                loop_break: 0,
                loop_continue: 0,
                loop_depth: 0,
                returning: None,
                source_depth: 0,
                exiting: None,
                cond_depth: 0,
                deadline_interrupt: None,
                uid: 0,
                input_stream: Vec::new(),
                input_pos: 0,
                python_repl: None,
                fds,
                program: None,
                pending_signals: std::collections::BTreeSet::new(),
                signal_dispositions: BTreeMap::new(),
                exit_disposition: None,
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

    /// Spawn and load one argv image using virtual process and descriptor state only.
    pub(crate) fn spawn_argv_child(
        &mut self,
        spec: crate::syscalls::SpawnSpec,
    ) -> Result<ProcessId, String> {
        if spec.argv.is_empty() {
            return Err("empty argv".into());
        }
        let input = spec
            .stdin
            .map(|bytes| self.descriptors.open_input(bytes))
            .transpose()
            .map_err(|error| format!("unable to prepare child input: {error:?}"))?;
        let display = crate::process::command_label(&spec.argv);
        let placement = if spec.new_process_group {
            crate::process::ChildPlacement::NewProcessGroup
        } else {
            crate::process::ChildPlacement::Inherit
        };
        let pid = match self.create_child(&display, true, !spec.detach, false, placement) {
            Ok(pid) => pid,
            Err(error) => {
                if let Some(input) = input {
                    let _ = self.descriptors.discard_unreferenced(input);
                }
                return Err(error);
            }
        };
        if let Some(input) = input {
            self.install_process_description(pid, 0, input)
                .expect("new child process must accept prepared standard input");
        }
        self.configure_process(pid, spec.cwd, spec.environment)
            .expect("new child process must accept its launch configuration");
        self.load_argv_program(pid, spec.argv)
            .expect("new child process must accept an argv continuation");
        Ok(pid)
    }

    /// Replace process `pid`'s program with an argv image, like `execve` after PATH lookup.
    /// The PID, descriptors, cwd, and environment are retained; the process-table label
    /// follows the new image.
    pub(crate) fn exec_argv_image(
        &mut self,
        pid: ProcessId,
        argv: Vec<String>,
    ) -> Result<(), String> {
        let label = crate::process::command_label(&argv);
        self.load_argv_program(pid, argv)?;
        self.processes.set_command(pid, label);
        Ok(())
    }

    /// Load an argv child through one image boundary. Migrated native commands execute as
    /// process-scoped Rust images; other argv still use the shell executor until ported.
    pub(crate) fn load_argv_program(
        &mut self,
        pid: ProcessId,
        mut argv: Vec<String>,
    ) -> Result<(), String> {
        let state = self
            .process
            .states
            .get(&pid)
            .ok_or_else(|| format!("process state does not exist for PID {pid}"))?;
        let lookup = crate::commands::util::resolve_executable_in(
            &self.vfs,
            &state.cwd,
            state.vars.get("PATH").map(String::as_str),
            &argv[0],
        );
        let native = match lookup {
            crate::commands::util::ExecutableLookup::Found(path) => {
                match self
                    .vfs
                    .metadata("/", &path, true)
                    .ok()
                    .map(|node| node.kind)
                {
                    Some(crate::vfs::NodeKind::NativeExecutable(
                        crate::vfs::NativeProgram::LegacyRegistered("sh" | "bash" | "dash" | "zsh"),
                    )) => match self.shell_image(pid, &argv[0], &argv[1..])? {
                        Ok(image) => {
                            self.load_shell_image(pid, image)?;
                            self.invocations.begin(
                                pid,
                                &argv,
                                crate::commands::Trust::Real,
                                None,
                                self.resources.cpu_used(),
                                self.vfs.disk_used(),
                            );
                            return Ok(());
                        }
                        Err((status, message)) => {
                            Some(crate::program::NativeProcess::failure(status, message))
                        }
                    },
                    Some(crate::vfs::NodeKind::NativeExecutable(
                        crate::vfs::NativeProgram::LegacyRegistered(_),
                    )) => {
                        argv[0] = path;
                        None
                    }
                    Some(crate::vfs::NodeKind::NativeExecutable(image)) => {
                        Some(crate::program::NativeProcess::from_image(image, &argv))
                    }
                    _ => {
                        argv[0] = path;
                        None
                    }
                }
            }
            crate::commands::util::ExecutableLookup::NotExecutable(path)
                if crate::vfs::NativeProgram::from_name(&argv[0]).is_some() =>
            {
                Some(crate::program::NativeProcess::failure(
                    126,
                    format!("{}: {path}: permission denied\n", argv[0]),
                ))
            }
            crate::commands::util::ExecutableLookup::NotFound
                if crate::vfs::NativeProgram::from_name(&argv[0]).is_some() =>
            {
                Some(crate::program::NativeProcess::failure(
                    127,
                    format!("{}: command not found\n", argv[0]),
                ))
            }
            _ => None,
        };
        if let Some(native) = native {
            let trust = native.trust();
            self.reset_for_exec(pid)?;
            self.process.set_program(
                pid,
                Some(crate::program::ProgramContinuation::Native(native)),
            )?;
            self.invocations.begin(
                pid,
                &argv,
                trust,
                None,
                self.resources.cpu_used(),
                self.vfs.disk_used(),
            );
            return Ok(());
        }
        self.process.set_continuation(
            pid,
            Some(crate::exec::ShellContinuation::new(
                &crate::shell::Node::ArgvCommand(argv),
            )),
        )
    }

    /// Caught shell traps reset to the default action when a process loads a new image; ignored
    /// signals stay ignored, as POSIX requires.
    fn reset_for_exec(&mut self, pid: ProcessId) -> Result<(), String> {
        let state = self
            .process
            .states
            .get_mut(&pid)
            .ok_or_else(|| format!("process state does not exist for PID {pid}"))?;
        state
            .signal_dispositions
            .retain(|_, disposition| matches!(disposition, ShellSignalDisposition::Ignore));
        state.exit_disposition = None;
        state.handling_signal = false;
        Ok(())
    }

    /// Parse a shell image's argv the way `bash`/`sh` do: short option clusters (`-c`, `-s`,
    /// `-e`, `-u`, `-x`, `-o NAME`), then `-c SOURCE [NAME [ARG...]]`, `SCRIPT [ARG...]`, or,
    /// with `-s` or no operand, a program read from standard input. A missing script or a
    /// syntax error in `-c`/script source becomes an exit status and diagnostic.
    fn shell_image(
        &mut self,
        pid: ProcessId,
        name: &str,
        args: &[String],
    ) -> Result<Result<ShellImage, (i32, String)>, String> {
        let mut image = ShellImage {
            arg0: name.to_string(),
            ..ShellImage::default()
        };
        let mut command = false;
        let mut from_stdin = false;
        let mut index = 0;
        while let Some(argument) = args.get(index) {
            if argument == "--" || argument == "-" {
                index += 1;
                break;
            }
            if argument.starts_with("--") {
                // Startup-file and mode switches such as --norc and --posix have no effect here.
                index += 1;
                continue;
            }
            let Some(flags) = argument
                .strip_prefix('-')
                .or_else(|| argument.strip_prefix('+'))
                .filter(|flags| !flags.is_empty())
            else {
                break;
            };
            let enable = argument.starts_with('-');
            index += 1;
            for flag in flags.chars() {
                match flag {
                    'c' => command = true,
                    's' => from_stdin = true,
                    'e' => image.errexit = enable,
                    'u' => image.nounset = enable,
                    'x' => image.xtrace = enable,
                    // Interactive and login shells are not modeled; the flags are accepted.
                    'i' | 'l' => {}
                    'o' => {
                        let Some(name) = args.get(index) else {
                            return Ok(Err((2, "bash: -o: option requires an argument\n".into())));
                        };
                        index += 1;
                        match name.as_str() {
                            "errexit" => image.errexit = enable,
                            "nounset" => image.nounset = enable,
                            "xtrace" => image.xtrace = enable,
                            "pipefail" => image.pipefail = enable,
                            other => {
                                return Ok(Err((
                                    2,
                                    format!("bash: {other}: invalid option name\n"),
                                )))
                            }
                        }
                    }
                    other => return Ok(Err((2, format!("bash: -{other}: invalid option\n")))),
                }
            }
        }
        let operands = &args[index..];
        let source = if command {
            let Some(source) = operands.first() else {
                return Ok(Err((2, "bash: -c: option requires an argument\n".into())));
            };
            if let Some(name) = operands.get(1) {
                image.arg0 = name.clone();
            }
            image.positional = operands.get(2..).unwrap_or_default().to_vec();
            source.clone()
        } else if from_stdin || operands.is_empty() {
            image.positional = operands.to_vec();
            return Ok(Ok(image));
        } else {
            let script = &operands[0];
            let cwd = self
                .process
                .states
                .get(&pid)
                .ok_or_else(|| format!("process state does not exist for PID {pid}"))?
                .cwd
                .clone();
            let Ok(source) = self.vfs.read_string(&cwd, script) else {
                return Ok(Err((
                    127,
                    format!("bash: {script}: No such file or directory\n"),
                )));
            };
            image.arg0 = script.clone();
            image.positional = operands[1..].to_vec();
            source
        };
        let mut error = Vec::new();
        Ok(
            match crate::commands::parse_shell_source(self, &source, &mut error) {
                Ok(program) => {
                    image.program = Some(program);
                    Ok(image)
                }
                Err(status) => Err((status, String::from_utf8_lossy(&error).into_owned())),
            },
        )
    }

    /// Install a parsed [`ShellImage`] as the program of process `pid`.
    fn load_shell_image(&mut self, pid: ProcessId, image: ShellImage) -> Result<(), String> {
        self.reset_for_exec(pid)?;
        let state = self
            .process
            .states
            .get_mut(&pid)
            .ok_or_else(|| format!("process state does not exist for PID {pid}"))?;
        // A new program image keeps only the environment: unexported variables, arrays,
        // functions, aliases, and other shell-local state belong to the replaced shell.
        let exported = std::mem::take(&mut state.exported);
        state.vars.retain(|name, _| exported.contains(name));
        state.exported = exported;
        state.arrays.clear();
        state.readonly.clear();
        state.funcs.clear();
        state.aliases.clear();
        state.command_hash.clear();
        state.directory_stack.clear();
        state.local_scopes.clear();
        state.arg0 = image.arg0;
        state.positional = image.positional;
        state.opt_errexit = image.errexit;
        state.opt_nounset = image.nounset;
        state.opt_xtrace = image.xtrace;
        state.opt_pipefail = image.pipefail;
        let continuation = match &image.program {
            Some(program) => crate::exec::ShellContinuation::subshell(program),
            None => crate::exec::ShellContinuation::standard_input_program(),
        };
        self.process.set_continuation(pid, Some(continuation))
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

    /// Create an idle shell in a new logical session over this environment's shared VFS.
    ///
    /// The new shell inherits the caller's current shell state once, then retains its own cwd,
    /// variables, descriptors, and parser state across later host actions. No host process is
    /// started. The default shell remains available through `run_script_capture`.
    pub fn spawn_shell_session(&mut self) -> Result<ProcessId, String> {
        let root = self.terminal.session_id;
        if self.process.pid != root
            || self.scheduler.current() != Some(root)
            || self.process.program.is_some()
            || self.exiting.is_some()
        {
            return Err("shell sessions can be created only from an idle default shell".into());
        }
        let pid = self.create_child(
            "bash",
            true,
            false,
            false,
            crate::process::ChildPlacement::NewSession,
        )?;
        self.scheduler
            .park_runnable(pid, WaitReason::ShellSession(pid))
            .map_err(|error| format!("unable to park shell session {pid}: {error:?}"))?;
        Ok(pid)
    }

    /// Run one action in a persistent shell session while other sessions keep their state.
    ///
    /// Host actions are serialized; modeled children still run cooperatively through the shared
    /// scheduler. The requested session must be idle, not a background child or exited process.
    pub fn run_shell_session_capture(
        &mut self,
        pid: ProcessId,
        source: &str,
    ) -> Result<(RunOutcome, Vec<u8>, Vec<u8>), String> {
        self.run_shell_session_capture_with_stdin(pid, source, &[])
    }

    /// Run one action with explicit input bytes in a persistent shell session.
    pub fn run_shell_session_capture_with_stdin(
        &mut self,
        pid: ProcessId,
        source: &str,
        stdin: &[u8],
    ) -> Result<(RunOutcome, Vec<u8>, Vec<u8>), String> {
        let root = self.terminal.session_id;
        if self.process.pid != root
            || self.scheduler.current() != Some(root)
            || self.process.program.is_some()
        {
            return Err("default shell is not idle".into());
        }
        if pid == root {
            return Ok(self.run_script_capture_with_stdin(source, stdin));
        }
        if !matches!(
            self.scheduler.state(pid),
            Some(crate::scheduler::TaskState::Blocked(WaitReason::ShellSession(waiting)))
                if waiting == pid
        ) {
            return Err(format!("shell session {pid} is not idle"));
        }
        self.scheduler
            .block_current(WaitReason::ShellSession(root))
            .map_err(|error| format!("unable to park default shell: {error:?}"))?;
        self.scheduler
            .wake(pid)
            .map_err(|error| format!("unable to wake shell session {pid}: {error:?}"))?;
        self.scheduler
            .dispatch_pid(pid)
            .map_err(|error| format!("unable to select shell session {pid}: {error:?}"))?;
        self.process.activate(pid)?;

        let result = self.run_script_capture_with_stdin(source, stdin);
        debug_assert_eq!(self.process.pid, pid);
        if let Some(status) = self.exiting {
            self.finish_child(pid, status);
        } else {
            self.scheduler
                .block_current(WaitReason::ShellSession(pid))
                .map_err(|error| format!("unable to park shell session {pid}: {error:?}"))?;
        }
        if self
            .scheduler
            .current()
            .is_some_and(|current| current != root)
        {
            self.scheduler
                .yield_current()
                .map_err(|error| format!("unable to yield to default shell: {error:?}"))?;
        }
        if self.scheduler.current().is_none() {
            self.scheduler
                .wake(root)
                .map_err(|error| format!("unable to wake default shell: {error:?}"))?;
            self.scheduler
                .dispatch_pid(root)
                .map_err(|error| format!("unable to select default shell: {error:?}"))?;
        }
        self.process.activate(root)?;
        Ok(result)
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
        self.vfs.retain_orphans(&self.descriptors.live_orphans());
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
            self.vfs.retain_orphans(&self.descriptors.live_orphans());
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
        self.read_fd_checked(fd, maximum)
            .map_err(|error| match error {
                crate::syscalls::SyscallError::Descriptor(error) => descriptor_message(error),
                crate::syscalls::SyscallError::File(error) => error.to_string(),
                crate::syscalls::SyscallError::Permission => {
                    "descriptor is not open for reading".to_string()
                }
                crate::syscalls::SyscallError::InvalidArgument => {
                    "file cursor exceeds addressable memory".to_string()
                }
                crate::syscalls::SyscallError::IsDirectory => "is a directory".to_string(),
                crate::syscalls::SyscallError::ResourceExhausted => {
                    self.resources.stop_reason().map_or_else(
                        || "device read limit exceeded".to_string(),
                        |reason| reason.to_string(),
                    )
                }
                crate::syscalls::SyscallError::NoSuchProcess => "no such process".to_string(),
                crate::syscalls::SyscallError::Process(error) => error,
            })
    }

    /// Typed descriptor read for native programs and guest ABI adapters.
    pub(crate) fn read_fd_checked(
        &mut self,
        fd: Fd,
        maximum: usize,
    ) -> Result<IoPoll<Vec<u8>>, crate::syscalls::SyscallError> {
        use crate::syscalls::SyscallError;

        let description = self.process.fds.get(fd)?;
        if let Some(file) = self.descriptors.file_state(description)? {
            if !file.readable {
                return Err(SyscallError::Permission);
            }
            let cursor = usize::try_from(file.cursor).map_err(|_| SyscallError::InvalidArgument)?;
            let bytes = match file.orphan {
                Some(id) => self
                    .vfs
                    .read_orphan_range(id, cursor, maximum.min(MAX_CAPTURE_BYTES)),
                None => {
                    self.vfs
                        .read_range("/", &file.path, cursor, maximum.min(MAX_CAPTURE_BYTES))
                }
            }?;
            self.descriptors.advance_file(description, bytes.len())?;
            return Ok(IoPoll::Ready(bytes));
        }
        let maximum = if self.descriptors.is_generated_device(description)? {
            let maximum = maximum
                .min(crate::descriptors::DEVICE_READ_QUANTUM)
                .min(usize::try_from(self.resources.cpu_remaining() / 2).unwrap_or(usize::MAX));
            if maximum == 0 {
                let _ = self.resources.charge_cpu(1);
                return Err(SyscallError::ResourceExhausted);
            }
            if !self.resources.charge_cpu(maximum as u64) {
                return Err(SyscallError::ResourceExhausted);
            }
            maximum
        } else {
            maximum
        };
        let result = self.descriptors.read(description, maximum)?;
        if matches!(result, IoPoll::Ready(_)) {
            if let Some((pipe, true)) = self.descriptors.pipe_endpoint(description)? {
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
        self.write_fd_checked(fd, bytes)
            .map_err(write_error_message)
    }

    /// Typed descriptor write used by native program images and guest syscall adapters.
    pub(crate) fn write_fd_checked(
        &mut self,
        fd: Fd,
        bytes: &[u8],
    ) -> Result<IoPoll<usize>, crate::syscalls::SyscallError> {
        use crate::syscalls::SyscallError;

        let description = self.process.fds.get(fd)?;
        if let Some(file) = self.descriptors.file_state(description)? {
            if !file.writable {
                return Err(SyscallError::Permission);
            }
            let cursor = usize::try_from(file.cursor).map_err(|_| SyscallError::InvalidArgument)?;
            self.sync_vfs_time();
            let result = match file.orphan {
                Some(id) => self.vfs.write_orphan_at(id, cursor, bytes),
                None => self.vfs.write_at("/", &file.path, cursor, bytes),
            };
            if let Err(error) = result {
                if file.remove_on_first_write_error && cursor == 0 {
                    let _ = self.vfs.remove_file("/", &file.path);
                }
                return Err(error.into());
            }
            self.descriptors.advance_file(description, bytes.len())?;
            return Ok(IoPoll::Ready(bytes.len()));
        }
        let result = self.descriptors.write(description, bytes)?;
        if matches!(result, IoPoll::Ready(_)) {
            if let Some((pipe, false)) = self.descriptors.pipe_endpoint(description)? {
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

    /// Resolve a path across both persistent and generated filesystem nodes.
    ///
    /// Symlink expansion is bounded and never consults the host filesystem. `follow_final`
    /// controls whether the last component is dereferenced, matching the metadata boundary.
    pub fn fs_realpath(
        &self,
        cwd: &str,
        path: &str,
        follow_final: bool,
    ) -> crate::vfs::Result<String> {
        let mut pending = crate::vfs::resolve_against(cwd, path);
        for _ in 0..40 {
            let components = pending
                .split('/')
                .filter(|component| !component.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>();
            let mut resolved = "/".to_string();
            let mut restarted = false;
            for (index, component) in components.iter().enumerate() {
                let candidate = if resolved == "/" {
                    format!("/{component}")
                } else {
                    format!("{resolved}/{component}")
                };
                let node = self.fs_metadata("/", &candidate, false)?;
                let is_final = index + 1 == components.len();
                if matches!(node.kind, crate::vfs::NodeKind::Symlink(_))
                    && (!is_final || follow_final)
                {
                    let target = self.fs_read_link("/", &candidate)?;
                    let parent = crate::vfs::parent_of(&candidate).unwrap_or_else(|| "/".into());
                    let mut replacement = crate::vfs::resolve_against(&parent, &target);
                    if !is_final {
                        replacement.push('/');
                        replacement.push_str(&components[index + 1..].join("/"));
                    }
                    pending = crate::vfs::normalize(&replacement);
                    restarted = true;
                    break;
                }
                resolved = candidate;
            }
            if !restarted {
                return Ok(resolved);
            }
        }
        Err(crate::vfs::VfsError::Loop(pending))
    }

    /// Walk a persistent or generated tree without following symbolic links.
    ///
    /// The VFS and process table are bounded, so the returned set is finite. Results use stable
    /// lexical order and include the starting path.
    pub fn fs_walk(&self, cwd: &str, path: &str) -> crate::vfs::Result<Vec<String>> {
        let start = crate::vfs::resolve_against(cwd, path);
        self.fs_metadata("/", &start, false)?;
        let mut pending = vec![start];
        let mut paths = Vec::new();
        while let Some(current) = pending.pop() {
            let node = self.fs_metadata("/", &current, false)?;
            let is_dir = matches!(node.kind, crate::vfs::NodeKind::Dir);
            paths.push(current.clone());
            if is_dir {
                let mut entries = self.fs_list_dir("/", &current)?;
                entries.sort();
                for entry in entries.into_iter().rev() {
                    pending.push(if current == "/" {
                        format!("/{entry}")
                    } else {
                        format!("{current}/{entry}")
                    });
                }
            }
        }
        Ok(paths)
    }

    pub fn get_var(&self, name: &str) -> Option<String> {
        match name {
            "?" => Some(self.last_status.to_string()),
            "$" => Some(self.shell_pid.to_string()),
            "PPID" => Some(self.ppid.to_string()),
            "BASHPID" => Some(self.pid.to_string()),
            "RANDOM" => {
                let state = self
                    .random_state
                    .get()
                    .wrapping_mul(1_103_515_245)
                    .wrapping_add(12_345);
                self.random_state.set(state);
                Some(((state >> 16) & 0x7fff).to_string())
            }
            "#" => Some(self.positional.len().to_string()),
            "PWD" => Some(self.cwd.clone()),
            "@" => Some(self.positional.join(" ")),
            // POSIX joins `$*` with the first IFS character: a space when IFS is unset and
            // nothing when IFS is empty.
            "*" => {
                let separator = match self.vars.get("IFS") {
                    Some(ifs) => ifs.chars().next().map(String::from).unwrap_or_default(),
                    None => " ".to_string(),
                };
                Some(self.positional.join(&separator))
            }
            _ => {
                if let Ok(n) = name.parse::<usize>() {
                    if n == 0 {
                        return Some(self.arg0.clone());
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
        if name == "RANDOM" {
            let seed = val.parse::<u32>().unwrap_or(0);
            self.random_state.set(seed);
            self.vars.remove(name);
            self.arrays.remove(name);
            return;
        }
        if name == "PWD" {
            self.cwd = val.clone();
        }
        if name == "OPTIND" {
            self.getopts.optind = val
                .parse::<usize>()
                .ok()
                .filter(|value| *value > 0)
                .unwrap_or(1);
            self.getopts.offset = 1;
        }
        // A plain scalar assignment to an array name in bash sets element 0; we instead treat it
        // as a fresh scalar (drop the array) — the common case in our scripts and lower-risk.
        self.arrays.remove(name);
        self.vars.insert(name.to_string(), val);
    }

    /// Enter a shell-function variable scope.
    pub(crate) fn enter_function_scope(&mut self) {
        self.local_scopes.push(HashMap::new());
    }

    /// Return whether `local` is valid in the current execution context.
    pub(crate) fn in_function_scope(&self) -> bool {
        !self.local_scopes.is_empty()
    }

    /// Save a binding once in the current function and reset it to an empty local value.
    pub(crate) fn declare_local(&mut self, name: &str) {
        let binding = LocalBinding {
            scalar: self.vars.get(name).cloned(),
            array: self.arrays.get(name).cloned(),
            exported: self.exported.contains(name),
        };
        let Some(scope) = self.local_scopes.last_mut() else {
            return;
        };
        if scope.contains_key(name) {
            return;
        }
        scope.insert(name.to_string(), binding);
        self.vars.remove(name);
        self.arrays.remove(name);
        self.exported.remove(name);
    }

    /// Restore bindings declared local by the function that just returned.
    pub(crate) fn leave_function_scope(&mut self) {
        let Some(scope) = self.local_scopes.pop() else {
            return;
        };
        for (name, binding) in scope {
            self.vars.remove(&name);
            self.arrays.remove(&name);
            self.exported.remove(&name);
            if let Some(value) = binding.scalar {
                self.vars.insert(name.clone(), value);
            }
            if let Some(value) = binding.array {
                self.arrays.insert(name.clone(), value);
            }
            if binding.exported {
                self.exported.insert(name);
            }
        }
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

/// Shell-facing text for a failed descriptor write.
pub(crate) fn write_error_message(error: crate::syscalls::SyscallError) -> String {
    match error {
        crate::syscalls::SyscallError::Descriptor(error) => descriptor_message(error),
        crate::syscalls::SyscallError::File(error) => error.to_string(),
        crate::syscalls::SyscallError::Permission => {
            "descriptor is not open for writing".to_string()
        }
        crate::syscalls::SyscallError::InvalidArgument => {
            "file cursor exceeds addressable memory".to_string()
        }
        crate::syscalls::SyscallError::IsDirectory => "is a directory".to_string(),
        crate::syscalls::SyscallError::ResourceExhausted => "resource limit exceeded".to_string(),
        crate::syscalls::SyscallError::NoSuchProcess => "no such process".to_string(),
        crate::syscalls::SyscallError::Process(error) => error,
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
