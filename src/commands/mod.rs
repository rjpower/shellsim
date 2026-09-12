//! Built-in commands and coreutils, implemented natively against the VFS.
//!
//! Ordinary commands are plain functions with the uniform signature
//! [`CmdFn`] = `fn(&mut CommandContext, &[String], &mut Io) -> i32`, where the slice is `argv[1..]`
//! and all I/O flows through the [`Io`] context. Commands that can block may additionally register
//! a resumable start function; native callers retain the synchronous entry point while shell
//! continuations receive a typed wait reason. Builtins that mutate environment state do so
//! through the context.
//!
//! Commands are looked up in a [`OnceLock`]-backed registry that records each command's
//! [`Trust`] level, so a run can report whether it stayed inside the faithfully-simulated
//! envelope. Unknown commands fall through to the VFS-script / shebang fallback.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::OnceLock;

use crate::interp::Interp;
use crate::scheduler::WaitReason;

mod archives;
mod awk;
mod builtins;
mod echo;
mod fs;
mod git;
mod hashing;
mod makecmd;
mod net;
mod patch;
pub(crate) mod pkg;
mod printf;
mod proc;
mod rgcmd;
mod sort;
mod system;
mod tarcmd;
mod text;
pub mod util;
mod zipcmd;

/// Bundled standard I/O for a command invocation.
pub struct Io<'a> {
    pub stdin: Vec<u8>,
    pub out: &'a mut Vec<u8>,
    pub err: &'a mut Vec<u8>,
}

/// Uniform command signature. `args` is `argv[1..]`.
pub type CmdFn = fn(&mut CommandContext<'_>, &[String], &mut Io) -> i32;

/// Result of starting a command from a resumable shell continuation.
pub(crate) enum CommandPoll {
    Ready(i32),
    /// A command consumed one bounded work quantum and remains runnable.
    Yielded(CommandResume),
    /// A resumed command completed with buffered output for normal descriptor flushing.
    ReadyOutput {
        command: String,
        status: i32,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
    Blocked(WaitReason, CommandResume),
    /// Starting the command changed the active scheduler process.
    Switched(CommandResume),
    /// Continue by executing shell syntax in the current process context.
    Inline(crate::shell::Node),
}

/// Command-owned state retained by the shell while a native command is suspended.
pub(crate) enum CommandResume {
    Status(i32),
    Child {
        pid: crate::process::ProcessId,
        reap: bool,
    },
    Wait {
        pids: Vec<crate::process::ProcessId>,
        status: i32,
        explicit: bool,
    },
    ChildSequence {
        pid: crate::process::ProcessId,
        remaining: std::collections::VecDeque<ChildCommand>,
        status: i32,
        stop_on_error: bool,
    },
    Timeout {
        pid: crate::process::ProcessId,
        deadline: Option<crate::clock::EventId>,
        preserve_status: bool,
        result_override: Option<i32>,
    },
    Python {
        command: String,
        continuation: Box<crate::python::PythonContinuation>,
    },
}

/// One scheduler-owned argv invocation requested by a modeled native command.
pub(crate) struct ChildCommand {
    pub argv: Vec<String>,
    pub stdin: Vec<u8>,
    pub cwd: Option<String>,
    pub environment: Option<std::collections::BTreeMap<String, String>>,
}

type ResumableCmdFn = fn(&mut CommandContext<'_>, &[String], &mut Io) -> CommandPoll;

/// The only environment handle handed to command implementations.
///
/// `Deref` keeps the initial port mechanical; quota-sensitive primitives are enforced by the
/// environment-owned VFS and resource meter even while older bodies use field syntax.
pub struct CommandContext<'a> {
    env: &'a mut Interp,
}

impl CommandContext<'_> {
    pub fn charge_cpu(&mut self, units: u64) -> bool {
        self.env.resources.charge_cpu(units)
    }

    pub fn reserve_memory(&mut self, bytes: u64) -> bool {
        self.env.resources.reserve_memory(bytes)
    }
}

impl Deref for CommandContext<'_> {
    type Target = Interp;

    fn deref(&self) -> &Self::Target {
        self.env
    }
}

impl DerefMut for CommandContext<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.env
    }
}

/// How faithfully a command is simulated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trust {
    /// Faithful implementation (coreutils / builtins we fully model).
    Real,
    /// A subset of the real behavior (jq, sed, grep, uv, …).
    Partial,
    /// Ignored / pretend-success (apt-get, pip, …).
    NoOp,
}

/// A registered command: its implementation and trust level.
pub struct CommandSpec {
    pub run: CmdFn,
    resume: Option<ResumableCmdFn>,
    resume_before_input: bool,
    pub trust: Trust,
    pub base_cpu: u64,
    pub base_memory: u64,
}

static REGISTRY: OnceLock<HashMap<&'static str, CommandSpec>> = OnceLock::new();

fn registry() -> &'static HashMap<&'static str, CommandSpec> {
    REGISTRY.get_or_init(build_registry)
}

/// Register `f` under every name in `names` with trust `t`.
fn reg(map: &mut HashMap<&'static str, CommandSpec>, names: &[&'static str], t: Trust, f: CmdFn) {
    reg_costed(map, names, t, 100, 10 * 1024, f);
}

/// Register a command with deterministic, deliberately-coarse base resource costs.
fn reg_costed(
    map: &mut HashMap<&'static str, CommandSpec>,
    names: &[&'static str],
    t: Trust,
    base_cpu: u64,
    base_memory: u64,
    f: CmdFn,
) {
    for &n in names {
        map.insert(
            n,
            CommandSpec {
                run: f,
                resume: None,
                resume_before_input: false,
                trust: t,
                base_cpu,
                base_memory,
            },
        );
    }
}

/// Register a command that can suspend when called by the shell while retaining a synchronous
/// compatibility entry point for nested native dispatchers.
fn reg_resumable(
    map: &mut HashMap<&'static str, CommandSpec>,
    names: &[&'static str],
    trust: Trust,
    run: CmdFn,
    resume: ResumableCmdFn,
) {
    for &name in names {
        map.insert(
            name,
            CommandSpec {
                run,
                resume: Some(resume),
                resume_before_input: true,
                trust,
                base_cpu: 100,
                base_memory: 10 * 1024,
            },
        );
    }
}

/// Register a resumable command whose continuation needs the command's complete bounded input.
fn reg_buffered_resumable(
    map: &mut HashMap<&'static str, CommandSpec>,
    names: &[&'static str],
    trust: Trust,
    run: CmdFn,
    resume: ResumableCmdFn,
) {
    reg_resumable(map, names, trust, run, resume);
    for name in names {
        map.get_mut(name)
            .expect("newly registered command must exist")
            .resume_before_input = false;
    }
}

fn build_registry() -> HashMap<&'static str, CommandSpec> {
    let mut m = HashMap::new();
    builtins::register(&mut m);
    archives::register(&mut m);
    awk::register(&mut m);
    echo::register(&mut m);
    printf::register(&mut m);
    sort::register(&mut m);
    system::register(&mut m);
    tarcmd::register(&mut m);
    text::register(&mut m);
    fs::register(&mut m);
    git::register(&mut m);
    hashing::register(&mut m);
    makecmd::register(&mut m);
    net::register(&mut m);
    patch::register(&mut m);
    proc::register(&mut m);
    rgcmd::register(&mut m);
    pkg::register(&mut m);
    zipcmd::register(&mut m);
    m
}

/// Dispatch entry point: look up `argv[0]`, record its trust, and run it.
///
/// Unknown commands fall through to the legacy fallback: try to execute a script that lives
/// in the VFS (shell or `#!`-python), otherwise record it as unsupported and return 127.
pub fn run(
    interp: &mut Interp,
    argv: &[String],
    stdin: Vec<u8>,
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
) -> i32 {
    match dispatch(interp, argv, stdin, out, err, false) {
        CommandPoll::Ready(status) => status,
        CommandPoll::ReadyOutput { .. } => {
            unreachable!("synchronous command output must use its borrowed buffers")
        }
        CommandPoll::Yielded(_) => {
            unreachable!("synchronous command dispatch cannot yield")
        }
        CommandPoll::Blocked(_, _) => {
            unreachable!("synchronous command dispatch cannot suspend")
        }
        CommandPoll::Switched(_) => {
            unreachable!("synchronous command dispatch cannot switch processes")
        }
        CommandPoll::Inline(_) => {
            unreachable!("synchronous command dispatch cannot inject shell frames")
        }
    }
}

/// Start a command from a shell continuation, allowing registered blocking commands to suspend.
pub(crate) fn poll(
    interp: &mut Interp,
    argv: &[String],
    stdin: Vec<u8>,
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
) -> CommandPoll {
    dispatch(interp, argv, stdin, out, err, true)
}

/// Whether the command has a continuation-aware entry point that does not consume standard
/// input before it can suspend.
pub(crate) fn starts_before_input(argv: &[String]) -> bool {
    if argv.len() == 1
        && argv
            .first()
            .is_some_and(|command| matches!(command.as_str(), "sh" | "bash" | "dash" | "zsh"))
    {
        return false;
    }
    argv.first().is_some_and(|requested| {
        let command = standard_utility_name(requested).unwrap_or(requested);
        registry()
            .get(command)
            .is_some_and(|spec| spec.resume.is_some() && spec.resume_before_input)
    })
}

/// Continue command-owned state after the scheduler wakes its process.
pub(crate) fn resume(interp: &mut Interp, continuation: CommandResume) -> CommandPoll {
    match continuation {
        CommandResume::Status(status) => CommandPoll::Ready(status),
        CommandResume::Child { pid, reap } => {
            match interp.processes.get(pid).map(|record| record.status) {
                Some(crate::process::ProcessStatus::Exited(status)) => {
                    if reap {
                        interp.processes.reap(pid);
                        let _ = interp.scheduler.reap(pid);
                    }
                    CommandPoll::Ready(status)
                }
                _ => {
                    CommandPoll::Blocked(WaitReason::Child(pid), CommandResume::Child { pid, reap })
                }
            }
        }
        CommandResume::Wait {
            mut pids,
            mut status,
            explicit,
        } => {
            while let Some(pid) = pids.first().copied() {
                let Some(position) = interp.jobs.iter().position(|job| job.pid == pid) else {
                    pids.remove(0);
                    continue;
                };
                if !interp.jobs[position].done {
                    return CommandPoll::Blocked(
                        WaitReason::Child(pid),
                        CommandResume::Wait {
                            pids,
                            status,
                            explicit,
                        },
                    );
                }
                let job = interp.jobs.remove(position);
                if explicit {
                    status = job.status;
                }
                interp.processes.reap(pid);
                let _ = interp.scheduler.reap(pid);
                pids.remove(0);
            }
            CommandPoll::Ready(status)
        }
        CommandResume::ChildSequence {
            pid,
            mut remaining,
            mut status,
            stop_on_error,
        } => match interp.processes.get(pid).map(|record| record.status) {
            Some(crate::process::ProcessStatus::Exited(child_status)) => {
                interp.processes.reap(pid);
                let _ = interp.scheduler.reap(pid);
                status = child_status;
                if stop_on_error && status != 0 {
                    CommandPoll::Ready(status)
                } else if let Some(command) = remaining.pop_front() {
                    start_child_sequence_item(interp, command, remaining, status, stop_on_error)
                } else {
                    CommandPoll::Ready(status)
                }
            }
            _ => CommandPoll::Blocked(
                WaitReason::Child(pid),
                CommandResume::ChildSequence {
                    pid,
                    remaining,
                    status,
                    stop_on_error,
                },
            ),
        },
        CommandResume::Timeout {
            pid,
            deadline,
            preserve_status,
            result_override,
        } => match interp.processes.get(pid).map(|record| record.status) {
            Some(crate::process::ProcessStatus::Exited(status)) => {
                let timed_out = deadline
                    .is_some_and(|event| interp.clock.monotonic_ns() >= event.deadline_ns());
                if let Some(deadline) = deadline {
                    interp.clock.cancel(deadline);
                }
                if timed_out {
                    for descendant in interp.processes.process_tree(pid).into_iter().rev() {
                        if descendant != pid {
                            interp.processes.reap(descendant);
                            let _ = interp.scheduler.reap(descendant);
                        }
                    }
                }
                interp.processes.reap(pid);
                let _ = interp.scheduler.reap(pid);
                CommandPoll::Ready(result_override.unwrap_or({
                    if timed_out && !preserve_status {
                        124
                    } else {
                        status
                    }
                }))
            }
            _ => CommandPoll::Blocked(
                WaitReason::Child(pid),
                CommandResume::Timeout {
                    pid,
                    deadline,
                    preserve_status,
                    result_override,
                },
            ),
        },
        CommandResume::Python {
            command,
            mut continuation,
        } => match continuation.poll(interp) {
            crate::python::PythonPoll::Ready(status) => {
                let (stdout, stderr) = (*continuation).into_output();
                CommandPoll::ReadyOutput {
                    command,
                    status,
                    stdout,
                    stderr,
                }
            }
            crate::python::PythonPoll::Runnable => CommandPoll::Yielded(CommandResume::Python {
                command,
                continuation,
            }),
            crate::python::PythonPoll::Blocked(reason) => CommandPoll::Blocked(
                reason,
                CommandResume::Python {
                    command,
                    continuation,
                },
            ),
        },
    }
}

/// Launch argv invocations sequentially as ordinary scheduler-owned logical children.
pub(crate) fn start_child_sequence(
    interp: &mut Interp,
    commands: Vec<ChildCommand>,
    stop_on_error: bool,
) -> CommandPoll {
    let mut commands = std::collections::VecDeque::from(commands);
    let Some(command) = commands.pop_front() else {
        return CommandPoll::Ready(0);
    };
    start_child_sequence_item(interp, command, commands, 0, stop_on_error)
}

fn start_child_sequence_item(
    interp: &mut Interp,
    command: ChildCommand,
    remaining: std::collections::VecDeque<ChildCommand>,
    status: i32,
    stop_on_error: bool,
) -> CommandPoll {
    if command.argv.is_empty() {
        return CommandPoll::Ready(status);
    }
    let pid = match start_child_command(interp, command) {
        Ok(pid) => pid,
        Err(status) => return CommandPoll::Ready(status),
    };
    CommandPoll::Switched(CommandResume::ChildSequence {
        pid,
        remaining,
        status,
        stop_on_error,
    })
}

/// Start one configured argv child and switch the scheduler to it.
pub(crate) fn start_child_command(
    interp: &mut Interp,
    command: ChildCommand,
) -> Result<crate::process::ProcessId, i32> {
    if command.argv.is_empty() {
        return Err(0);
    }
    let input = interp
        .descriptors
        .open_input(command.stdin)
        .map_err(|_| 125)?;
    let display = command.argv.join(" ");
    let pid = match interp.start_child(&display, true) {
        Ok(pid) => pid,
        Err(_) => {
            let _ = interp.descriptors.discard_unreferenced(input);
            return Err(125);
        }
    };
    interp
        .install_process_description(pid, 0, input)
        .expect("new child process must accept prepared standard input");
    interp
        .configure_process(pid, command.cwd, command.environment)
        .expect("new child process must accept its launch configuration");
    interp
        .process
        .set_continuation(
            pid,
            Some(crate::exec::ShellContinuation::new(
                &crate::shell::Node::ArgvCommand(command.argv),
            )),
        )
        .expect("new child process must accept an argv continuation");
    Ok(pid)
}

fn dispatch(
    interp: &mut Interp,
    argv: &[String],
    stdin: Vec<u8>,
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
    resumable: bool,
) -> CommandPoll {
    let requested = argv[0].as_str();
    // Agents frequently use explicit paths or `/usr/bin/env` shebangs. Standard utility paths
    // resolve to the same in-process command without pretending arbitrary host paths exist.
    let cmd = standard_utility_name(requested).unwrap_or(requested);
    let args = &argv[1..];
    if let Some(spec) = registry().get(cmd) {
        match spec.trust {
            Trust::NoOp => {
                // NoOp commands (package managers and native compilers) are recorded as unsupported,
                // preserving the legacy `note_unsupported(cmd)` behavior for that arm.
                interp.note_unsupported(cmd);
                interp.trust_noop.insert(cmd.to_string());
            }
            Trust::Partial => {
                interp.trust_partial.insert(cmd.to_string());
            }
            Trust::Real => {}
        }
        let cpu_before = interp.resources.cpu_used();
        let disk_before = interp.vfs.disk_used();
        let fs_read_before = interp.vfs.read_bytes();
        let input_bytes = stdin.len() as u64;
        let arg_bytes = args.iter().map(|a| a.len() as u64).sum::<u64>();
        let working_memory = spec.base_memory.saturating_add(input_bytes);
        if !interp
            .resources
            .charge_cpu(spec.base_cpu.saturating_add(arg_bytes))
        {
            return CommandPoll::Ready(
                interp
                    .resources
                    .stop_reason()
                    .map_or(137, |r| r.exit_status()),
            );
        }
        if !interp.resources.reserve_memory(working_memory) {
            return CommandPoll::Ready(
                interp
                    .resources
                    .stop_reason()
                    .map_or(137, |r| r.exit_status()),
            );
        }

        let out_before = out.len();
        let err_before = err.len();
        let output_meter_before = interp.resources.output_bytes();
        interp.sync_vfs_time();
        let mut io = Io { stdin, out, err };
        let mut context = CommandContext { env: interp };
        let mut result = match (resumable, spec.resume) {
            (true, Some(start)) => start(&mut context, args, &mut io),
            _ => CommandPoll::Ready((spec.run)(&mut context, args, &mut io)),
        };
        let out_bytes = io.out.len().saturating_sub(out_before);
        let err_bytes = io.err.len().saturating_sub(err_before);
        let output_bytes = out_bytes.saturating_add(err_bytes) as u64;
        // Nested dispatches already charged their own output. Only meter bytes produced directly
        // by this frame, otherwise `xargs`/`timeout` style commands would double-count them.
        let nested_output = interp
            .resources
            .output_bytes()
            .saturating_sub(output_meter_before);
        let unaccounted_output = output_bytes.saturating_sub(nested_output);
        let output_remaining = interp.resources.output_remaining();
        if unaccounted_output > output_remaining {
            let allowed_total = output_bytes.saturating_sub(unaccounted_output - output_remaining);
            let allowed_out = out_bytes.min(allowed_total as usize);
            io.out.truncate(out_before + allowed_out);
            let remaining = allowed_total.saturating_sub(allowed_out as u64) as usize;
            io.err.truncate(err_before + err_bytes.min(remaining));
        }
        let _ = interp.resources.charge_cpu(
            input_bytes
                .saturating_add(output_bytes)
                .saturating_add(interp.vfs.read_bytes().saturating_sub(fs_read_before)),
        );
        let _ = interp.resources.charge_output(unaccounted_output);
        interp.resources.release_memory(working_memory);
        interp
            .resources
            .record_command(cmd, cpu_before, disk_before, interp.vfs.disk_used());
        if !matches!(result, CommandPoll::Switched(_) | CommandPoll::Yielded(_)) {
            if let Some(reason) = interp.resources.stop_reason() {
                result = CommandPoll::Ready(reason.exit_status());
            }
        }
        return result;
    }

    // ---- fallback: resolve an executable script without exposing the host PATH/filesystem ----
    interp.sync_vfs_time();
    match util::resolve_executable(interp, requested) {
        util::ExecutableLookup::Found(path) => {
            if let Some(result) =
                util::try_exec_script(interp, &path, args, &stdin, out, err, resumable)
            {
                return result;
            }
        }
        util::ExecutableLookup::NotExecutable(path) => {
            util::ewln(err, &format!("{requested}: {path}: permission denied"));
            return CommandPoll::Ready(126);
        }
        util::ExecutableLookup::NotFound => {}
    }
    // An unknown command the task actually invoked (a missing tool, a compiled binary we can't
    // run, …) is a genuine simulation gap — record it so the trust verdict reflects it.
    interp.note_unsupported(requested);
    interp.trust_noop.insert(requested.to_string());
    util::ewln(err, &format!("{requested}: command not found"));
    CommandPoll::Ready(127)
}

/// Parse bounded shell source while charging the common parser resource model.
pub(crate) fn parse_shell_source(
    interp: &mut Interp,
    source: &str,
    err: &mut Vec<u8>,
) -> Result<crate::shell::Node, i32> {
    let parser_memory = 8 * 1024 + (source.len() as u64).saturating_mul(2);
    if !interp.resources.reserve_memory(parser_memory) {
        return Err(interp
            .resources
            .stop_reason()
            .map_or(137, |reason| reason.exit_status()));
    }
    let parsed = if interp.resources.charge_cpu(source.len() as u64) {
        crate::shell::parse(source).map_err(|error| error.to_string())
    } else {
        Err(String::new())
    };
    interp.resources.release_memory(parser_memory);
    match parsed {
        Ok(ast) => Ok(ast),
        Err(error) if error.is_empty() => Err(interp
            .resources
            .stop_reason()
            .map_or(137, |reason| reason.exit_status())),
        Err(error) => {
            util::ewln(err, &format!("shellsim: syntax error: {error}"));
            Err(2)
        }
    }
}

fn standard_utility_name(path: &str) -> Option<&str> {
    [
        "/bin/",
        "/sbin/",
        "/usr/bin/",
        "/usr/sbin/",
        "/usr/local/bin/",
    ]
    .into_iter()
    .find_map(|prefix| path.strip_prefix(prefix).filter(|name| !name.contains('/')))
}

/// Provide `run_script_into` for nested execution (source, eval, scripts).
impl Interp {
    pub fn run_script_into(&mut self, src: &str, out: &mut Vec<u8>, err: &mut Vec<u8>) -> i32 {
        let ast = match parse_shell_source(self, src, err) {
            Ok(ast) => ast,
            Err(status) => {
                self.last_status = status;
                return status;
            }
        };
        let r = self.returning.take();
        let code = crate::exec::exec(self, &ast, Vec::new(), out, err);
        if r.is_some() {
            self.returning = r;
        }
        code
    }
}
