//! The shell executor: walks the AST against the interpreter state.

use crate::expand::{expand_word, expand_words};
use crate::interp::Interp;
use crate::shell::{Node, RedirOp, Redirect};
use crate::syscalls::{ActiveSystem, OpenFile, System};
use crate::{descriptors::IoPoll, vfs::resolve_against};

/// Execute a node with finite input and capture its terminal output.
///
/// The byte-buffer API is the harness boundary. Internally, execution installs those streams as
/// process descriptors so redirection, duplication, and child inheritance use one model.
pub fn exec(
    interp: &mut Interp,
    node: &Node,
    stdin: Vec<u8>,
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
) -> i32 {
    let mut execution = match ShellExecution::start(interp, node, &stdin, true) {
        Ok(execution) => execution,
        Err(error) => {
            err.extend_from_slice(format!("shellsim: {error}\n").as_bytes());
            return 125;
        }
    };
    let status = loop {
        match execution.poll(interp, true) {
            Ok(MachinePoll::Progress) => {}
            Ok(MachinePoll::Blocked) => {
                err.extend_from_slice(b"shellsim: all processes are blocked without an event\n");
                break 125;
            }
            Ok(MachinePoll::Ready(status)) => break status,
            Err(error) => {
                err.extend_from_slice(format!("shellsim: scheduler: {error}\n").as_bytes());
                break 125;
            }
        }
    };
    execution.drain_output(interp, out, err);
    execution.restore(interp);
    status
}

/// Descriptor and continuation ownership for one foreground shell action.
#[derive(Clone)]
pub(crate) struct ShellExecution {
    target_pid: u32,
    saved: crate::descriptors::FdTable,
    stdin: crate::descriptors::DescriptionId,
    stdout: crate::descriptors::DescriptionId,
    stderr: crate::descriptors::DescriptionId,
    stdin_closed: bool,
}

impl ShellExecution {
    /// Install isolated action descriptors and a retained shell continuation.
    pub(crate) fn start(
        interp: &mut Interp,
        node: &Node,
        stdin: &[u8],
        stdin_closed: bool,
    ) -> Result<Self, String> {
        let saved = interp
            .process
            .fds
            .fork(&mut interp.descriptors)
            .map_err(|error| format!("unable to save descriptors: {error:?}"))?;
        let setup = (|| {
            let input = if stdin_closed {
                interp.descriptors.open_input(stdin.to_vec())?
            } else {
                let input = interp.descriptors.open_stream_input()?;
                if let Err(error) = interp.descriptors.append_input(input, stdin) {
                    let _ = interp.descriptors.discard_unreferenced(input);
                    return Err(error);
                }
                input
            };
            interp.install_new_description(0, input)?;
            let stdout = interp.descriptors.open_capture()?;
            interp.install_new_description(1, stdout)?;
            let stderr = interp.descriptors.open_capture()?;
            interp.install_new_description(2, stderr)?;
            Ok::<_, crate::descriptors::DescriptorError>((input, stdout, stderr))
        })();
        let (input, stdout, stderr) = match setup {
            Ok(descriptions) => descriptions,
            Err(error) => {
                restore_fds(interp, saved);
                return Err(format!("unable to install action descriptors: {error:?}"));
            }
        };
        let target_pid = match start_node(interp, node) {
            Ok(pid) => pid,
            Err(status) => {
                restore_fds(interp, saved);
                return Err(format!(
                    "unable to start shell continuation (status {status})"
                ));
            }
        };
        Ok(Self {
            target_pid,
            saved,
            stdin: input,
            stdout,
            stderr,
            stdin_closed,
        })
    }

    pub(crate) fn target_pid(&self) -> u32 {
        self.target_pid
    }

    pub(crate) fn poll(
        &mut self,
        interp: &mut Interp,
        advance_time: bool,
    ) -> Result<MachinePoll, String> {
        poll_machine(interp, self.target_pid, advance_time)
    }

    /// Append bounded action input and wake only tasks waiting on this description.
    pub(crate) fn write_stdin(&mut self, interp: &mut Interp, bytes: &[u8]) -> Result<(), String> {
        if self.stdin_closed {
            return Err("action stdin is closed".to_string());
        }
        interp
            .descriptors
            .append_input(self.stdin, bytes)
            .map_err(|error| format!("unable to append action stdin: {error:?}"))?;
        interp
            .scheduler
            .wake_waiters(crate::scheduler::WaitReason::InputReadable(self.stdin));
        Ok(())
    }

    /// Close action input and wake a drained reader so it can observe EOF.
    pub(crate) fn close_stdin(&mut self, interp: &mut Interp) -> Result<(), String> {
        if self.stdin_closed {
            return Ok(());
        }
        interp
            .descriptors
            .close_input(self.stdin)
            .map_err(|error| format!("unable to close action stdin: {error:?}"))?;
        self.stdin_closed = true;
        interp
            .scheduler
            .wake_waiters(crate::scheduler::WaitReason::InputReadable(self.stdin));
        Ok(())
    }

    /// Drain newly produced bytes without releasing action descriptors.
    pub(crate) fn drain_output(
        &mut self,
        interp: &mut Interp,
        stdout: &mut Vec<u8>,
        stderr: &mut Vec<u8>,
    ) {
        stdout.extend_from_slice(&std::mem::take(&mut interp.pending_stdout));
        stderr.extend_from_slice(&std::mem::take(&mut interp.pending_stderr));
        if let Ok(bytes) = interp.descriptors.drain_capture(self.stdout) {
            stdout.extend_from_slice(&bytes);
        }
        if let Ok(bytes) = interp.descriptors.drain_capture(self.stderr) {
            stderr.extend_from_slice(&bytes);
        }
    }

    /// Restore the persistent root descriptor table after action completion.
    pub(crate) fn restore(self, interp: &mut Interp) {
        restore_fds(interp, self.saved);
    }
}

fn restore_fds(interp: &mut Interp, saved: crate::descriptors::FdTable) {
    interp.process.fds.close_all(&mut interp.descriptors);
    interp.process.fds = saved;
    interp.refresh_descriptor_snapshot(interp.process.pid);
}

const MAX_SHELL_FRAMES: usize = 4_096;
const SHELL_POLL_QUANTUM: usize = 1;
/// Longest single command a shell reads from standard input before rejecting it.
const MAX_STDIN_PROGRAM_BYTES: usize = 256 * 1024;

/// Clock readings taken when a `time` pipeline starts.
#[derive(Clone, Copy)]
struct TimeStart {
    monotonic_ns: u64,
    cpu_ns: u64,
    posix: bool,
}

/// Bash's default `time` report (`TIMEFORMAT` is not consulted), or the POSIX `-p` form.
/// User time is the machine CPU consumed while the pipeline ran; system time is not modeled.
fn time_report(start: TimeStart, monotonic_ns: u64, cpu_ns: u64) -> String {
    let real = monotonic_ns.saturating_sub(start.monotonic_ns);
    let user = cpu_ns.saturating_sub(start.cpu_ns);
    if start.posix {
        let seconds = |ns: u64| {
            let centis = ns / 10_000_000;
            format!("{}.{:02}", centis / 100, centis % 100)
        };
        format!(
            "real {}\nuser {}\nsys {}\n",
            seconds(real),
            seconds(user),
            seconds(0)
        )
    } else {
        let clock = |ns: u64| {
            let millis = ns / 1_000_000;
            let seconds = millis / 1_000;
            format!("{}m{}.{:03}s", seconds / 60, seconds % 60, millis % 1_000)
        };
        format!(
            "\nreal\t{}\nuser\t{}\nsys\t{}\n",
            clock(real),
            clock(user),
            clock(0)
        )
    }
}

#[derive(Clone)]
enum ShellFrame {
    FinishTime(TimeStart),
    Eval(Node),
    /// Read the next complete command from fd 0, as a shell with no `-c` or script operand
    /// does. `pending` holds a partial command that needs more lines.
    ReadProgram {
        pending: Vec<u8>,
    },
    /// End the shell with the current status once an `exec`'d child finishes.
    ExitWithStatus,
    PrepareCommand(PreparedCommand),
    RunCommand {
        assigns: Vec<(String, String)>,
        words: Vec<String>,
        temporary_variables: Vec<(String, Option<String>)>,
        substitution_status: Option<i32>,
    },
    AwaitCommandSubstitution {
        command: PreparedCommand,
        pid: crate::process::ProcessId,
        capture: crate::descriptors::DescriptionId,
        variable: String,
        previous: Option<String>,
    },
    AwaitProcessSubstitution {
        command: PreparedCommand,
        pid: crate::process::ProcessId,
        capture: crate::descriptors::DescriptionId,
        location: ProcessSubstitutionLocation,
    },
    RunOutputSubstitutions {
        substitutions: Vec<OutputProcessSubstitution>,
        command_status: Option<i32>,
    },
    AwaitOutputProcessSubstitution {
        substitutions: Vec<OutputProcessSubstitution>,
        command_status: i32,
        pid: crate::process::ProcessId,
    },
    CleanupProcessSubstitutionFiles(Vec<String>),
    PrepareFor(PreparedFor),
    AwaitForSubstitution {
        state: PreparedFor,
        pid: crate::process::ProcessId,
        capture: crate::descriptors::DescriptionId,
        variable: String,
        previous: Option<String>,
    },
    PrepareCase(PreparedCase),
    AwaitCaseSubstitution {
        state: PreparedCase,
        pid: crate::process::ProcessId,
        capture: crate::descriptors::DescriptionId,
        variable: String,
        previous: Option<String>,
    },
    PrepareArithmetic(PreparedArithmetic),
    AwaitArithmeticSubstitution {
        state: PreparedArithmetic,
        pid: crate::process::ProcessId,
        capture: crate::descriptors::DescriptionId,
        start: usize,
        end: usize,
    },
    Sequence {
        nodes: Vec<Node>,
        next: usize,
    },
    Conditional {
        rhs: Node,
        run_on_success: bool,
    },
    Negate,
    IfNext {
        branches: Vec<(Node, Node)>,
        next: usize,
        els: Option<Node>,
    },
    IfAfter {
        branches: Vec<(Node, Node)>,
        next: usize,
        els: Option<Node>,
    },
    WhileCheck {
        cond: Node,
        body: Node,
        until: bool,
        body_status: i32,
    },
    WhileAfterCondition {
        cond: Node,
        body: Node,
        until: bool,
        body_status: i32,
    },
    WhileAfterBody {
        cond: Node,
        body: Node,
        until: bool,
    },
    ForNext {
        var: String,
        items: Vec<String>,
        body: Node,
        next: usize,
        body_status: i32,
    },
    ForAfterBody {
        var: String,
        items: Vec<String>,
        body: Node,
        next: usize,
    },
    CForCheck {
        cond: String,
        update: String,
        body: Node,
        body_status: i32,
    },
    CForAfterBody {
        cond: String,
        update: String,
        body: Node,
    },
    FinishLoop,
    RestoreRedirect(RedirectScope),
    FinishFunction {
        positional: Vec<String>,
        variables: Vec<(String, Option<String>)>,
    },
    FinishInlineCommand {
        variables: Vec<(String, Option<String>)>,
    },
    FinishSource {
        variables: Vec<(String, Option<String>)>,
    },
    FinishSignalHandler {
        previous_status: i32,
    },
    FinishExitTrap {
        previous_status: i32,
    },
    ResumeCommand {
        variables: Vec<(String, Option<String>)>,
        continuation: crate::commands::CommandResume,
    },
    ReadCommandInput {
        argv: Vec<String>,
        variables: Vec<(String, Option<String>)>,
        stdin: Vec<u8>,
        reserved: u64,
    },
    WriteCommandOutput {
        command: String,
        variables: Vec<(String, Option<String>)>,
        status: i32,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        stdout_offset: usize,
        stderr_offset: usize,
    },
    AwaitChild {
        pid: crate::process::ProcessId,
        reap: bool,
    },
    AwaitPipeline {
        pids: Vec<crate::process::ProcessId>,
        statuses: Vec<i32>,
        next: usize,
        pipefail: bool,
    },
}

#[derive(Clone)]
struct PreparedCommand {
    assigns: Vec<(String, String)>,
    words: Vec<String>,
    redirects: Vec<Redirect>,
    temporary_variables: Vec<(String, Option<String>)>,
    substitution_status: Option<i32>,
    output_substitutions: Vec<OutputProcessSubstitution>,
    process_substitution_files: Vec<String>,
}

#[derive(Clone)]
enum ProcessSubstitutionLocation {
    Redirect(usize),
    Word(usize),
}

#[derive(Clone)]
struct OutputProcessSubstitution {
    path: String,
    source: String,
}

#[derive(Clone)]
struct PreparedFor {
    var: String,
    words: Vec<String>,
    body: Node,
    temporary_variables: Vec<(String, Option<String>)>,
}

#[derive(Clone)]
struct PreparedCase {
    word: String,
    arms: Vec<(Vec<String>, Node)>,
    temporary_variables: Vec<(String, Option<String>)>,
}

#[derive(Clone)]
struct PreparedArithmetic {
    expression: String,
    continuation: ArithmeticContinuation,
}

#[derive(Clone)]
enum ArithmeticContinuation {
    Command,
    CForInit {
        cond: String,
        update: String,
        body: Node,
    },
    CForCondition {
        cond: String,
        update: String,
        body: Node,
        body_status: i32,
    },
    CForUpdate {
        cond: String,
        update: String,
        body: Node,
        body_status: i32,
    },
}

#[derive(Clone, Copy)]
enum ExpansionLocation {
    AssignmentKey(usize),
    AssignmentValue(usize),
    Word(usize),
    Redirect(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShellPoll {
    Pending,
    Blocked(crate::scheduler::WaitReason),
    Switched,
    /// The process image was replaced in place; the polled continuation must be discarded.
    Replaced,
    Ready(i32),
}

#[derive(Clone)]
pub(crate) struct ShellContinuation {
    frames: Vec<ShellFrame>,
    status: i32,
    /// Whether the current nonzero status came from an errexit-exempt AND-OR or negation context.
    errexit_exempt: bool,
    switched: bool,
    yielded: bool,
    blocked: Option<crate::scheduler::WaitReason>,
    exit_trap_started: bool,
    /// A subshell process exits after its program, so its final external command may replace
    /// the process image as Bash does, keeping `$!` and pipeline PIDs on the real program.
    exec_tail: bool,
    replaced: bool,
}

impl ShellContinuation {
    pub(crate) fn new(node: &Node) -> Self {
        Self {
            frames: vec![ShellFrame::Eval(node.clone())],
            status: 0,
            errexit_exempt: false,
            switched: false,
            yielded: false,
            blocked: None,
            exit_trap_started: false,
            exec_tail: false,
            replaced: false,
        }
    }

    /// Program for a forked subshell whose process ends with it: a background job, pipeline
    /// stage, or `sh -c` child.
    pub(crate) fn subshell(node: &Node) -> Self {
        Self {
            exec_tail: true,
            ..Self::new(node)
        }
    }

    /// Program for a shell that reads commands from standard input. It reads one line at a
    /// time and runs each complete command before reading more, so commands observe input
    /// that follows them and a streaming producer is consumed incrementally.
    pub(crate) fn standard_input_program() -> Self {
        Self {
            frames: vec![ShellFrame::ReadProgram {
                pending: Vec::new(),
            }],
            ..Self::new(&Node::Empty)
        }
    }

    pub(crate) fn poll(&mut self, interp: &mut Interp, budget: usize) -> ShellPoll {
        self.switched = false;
        self.yielded = false;
        self.blocked = None;
        for _ in 0..budget.max(1) {
            let Some(frame) = self.frames.pop() else {
                if !self.exit_trap_started {
                    self.exit_trap_started = true;
                    let previous_status = interp.exiting.take().unwrap_or(self.status);
                    match interp.exit_disposition.take() {
                        Some(crate::interp::ShellSignalDisposition::Handler { body, .. }) => {
                            self.status = previous_status;
                            self.frames
                                .push(ShellFrame::FinishExitTrap { previous_status });
                            self.frames.push(ShellFrame::Eval(body));
                            continue;
                        }
                        Some(crate::interp::ShellSignalDisposition::Ignore) | None => {}
                    }
                }
                return ShellPoll::Ready(self.status);
            };
            self.step(interp, frame);
            interp.last_status = self.status;
            if self.replaced {
                return ShellPoll::Replaced;
            }
            if self.switched {
                return ShellPoll::Switched;
            }
            if self.yielded {
                return ShellPoll::Pending;
            }
            if let Some(reason) = self.blocked.take() {
                return ShellPoll::Blocked(reason);
            }
        }
        if self.frames.is_empty() && (self.exit_trap_started || interp.exit_disposition.is_none()) {
            ShellPoll::Ready(self.status)
        } else {
            ShellPoll::Pending
        }
    }

    fn push(&mut self, interp: &mut Interp, frame: ShellFrame) -> bool {
        if self.frames.len() >= MAX_SHELL_FRAMES {
            write_diagnostic(
                interp,
                "shellsim: shell continuation frame limit exceeded\n",
            );
            self.status = 2;
            self.abort(interp);
            false
        } else {
            self.frames.push(frame);
            true
        }
    }

    fn ensure_capacity(&mut self, interp: &mut Interp, additional: usize) -> bool {
        if self.frames.len().saturating_add(additional) > MAX_SHELL_FRAMES {
            write_diagnostic(
                interp,
                "shellsim: shell continuation frame limit exceeded\n",
            );
            self.status = 2;
            self.abort(interp);
            false
        } else {
            true
        }
    }

    fn abort(&mut self, interp: &mut Interp) {
        for frame in std::mem::take(&mut self.frames).into_iter().rev() {
            match frame {
                ShellFrame::RestoreRedirect(scope) => end_redirects(interp, scope),
                ShellFrame::CleanupProcessSubstitutionFiles(paths) => {
                    for path in paths {
                        let _ = interp.vfs.remove_file("/", &path);
                    }
                }
                ShellFrame::RunOutputSubstitutions { substitutions, .. }
                | ShellFrame::AwaitOutputProcessSubstitution { substitutions, .. } => {
                    for substitution in substitutions {
                        let _ = interp.vfs.remove_file("/", &substitution.path);
                    }
                }
                ShellFrame::FinishFunction {
                    positional,
                    variables,
                } => {
                    interp.leave_function_scope();
                    interp.positional = positional;
                    restore_command_variables(interp, variables);
                }
                ShellFrame::FinishInlineCommand { variables } => {
                    restore_command_variables(interp, variables);
                }
                ShellFrame::FinishSource { variables } => {
                    interp.source_depth = interp.source_depth.saturating_sub(1);
                    interp.returning = None;
                    restore_command_variables(interp, variables);
                }
                ShellFrame::FinishSignalHandler { .. } => interp.finish_signal_handler(),
                ShellFrame::FinishExitTrap { .. } => {}
                ShellFrame::FinishLoop => {
                    interp.loop_depth = interp.loop_depth.saturating_sub(1);
                }
                ShellFrame::Conditional { .. }
                | ShellFrame::Negate
                | ShellFrame::IfAfter { .. }
                | ShellFrame::WhileAfterCondition { .. } => {
                    interp.cond_depth = interp.cond_depth.saturating_sub(1);
                }
                _ => {}
            }
        }
    }

    fn read_program(&mut self, interp: &mut Interp, mut pending: Vec<u8>) {
        if should_unwind(interp) {
            return;
        }
        // Read a byte at a time so input after the current command stays in the pipe or file
        // for the commands that run next, as a shell reading an unseekable stdin must.
        let eof = match interp.read_fd(0, 1) {
            Ok(IoPoll::Ready(bytes)) if bytes.is_empty() => true,
            Ok(IoPoll::Ready(bytes)) => {
                pending.extend_from_slice(&bytes);
                if pending.len() > MAX_STDIN_PROGRAM_BYTES {
                    write_diagnostic(interp, "shellsim: standard input command too long\n");
                    self.status = 2;
                    return;
                }
                if bytes != b"\n" {
                    self.frames.push(ShellFrame::ReadProgram { pending });
                    return;
                }
                false
            }
            Ok(IoPoll::Blocked(wait)) => {
                self.frames.push(ShellFrame::ReadProgram { pending });
                self.blocked = Some(io_wait_reason(wait));
                return;
            }
            // A closed or unreadable stdin ends the program as EOF does.
            Err(_) => true,
        };
        let source = String::from_utf8_lossy(&pending).into_owned();
        if !eof && crate::shell::needs_more_input(&source) {
            self.frames.push(ShellFrame::ReadProgram { pending });
            return;
        }
        if source.trim().is_empty() {
            if !eof {
                self.frames.push(ShellFrame::ReadProgram {
                    pending: Vec::new(),
                });
            }
            return;
        }
        let mut diagnostic = Vec::new();
        match crate::commands::parse_shell_source(interp, &source, &mut diagnostic) {
            Ok(program) => {
                if !eof {
                    self.frames.push(ShellFrame::ReadProgram {
                        pending: Vec::new(),
                    });
                }
                self.push(interp, ShellFrame::Eval(program));
            }
            Err(status) => {
                write_diagnostic(interp, &String::from_utf8_lossy(&diagnostic));
                self.status = status;
            }
        }
    }

    pub(crate) fn inject_signal_handler(&mut self, interp: &mut Interp, body: Node) {
        if !self.ensure_capacity(interp, 2) {
            interp.finish_signal_handler();
            return;
        }
        self.frames.push(ShellFrame::FinishSignalHandler {
            previous_status: self.status,
        });
        self.frames.push(ShellFrame::Eval(body));
    }

    fn retain_switched_command(
        &mut self,
        variables: Vec<(String, Option<String>)>,
        continuation: crate::commands::CommandResume,
    ) {
        self.frames.push(ShellFrame::ResumeCommand {
            variables,
            continuation,
        });
        self.switched = true;
    }

    fn enter_inline_command(
        &mut self,
        interp: &mut Interp,
        variables: Vec<(String, Option<String>)>,
        node: Node,
    ) {
        if !self.ensure_capacity(interp, 2) {
            restore_command_variables(interp, variables);
            return;
        }
        self.frames
            .push(ShellFrame::FinishInlineCommand { variables });
        self.frames.push(ShellFrame::Eval(node));
    }

    fn enter_source(
        &mut self,
        interp: &mut Interp,
        variables: Vec<(String, Option<String>)>,
        node: Node,
    ) {
        if !self.ensure_capacity(interp, 2) {
            restore_command_variables(interp, variables);
            return;
        }
        interp.source_depth = interp.source_depth.saturating_add(1);
        self.frames.push(ShellFrame::FinishSource { variables });
        self.frames.push(ShellFrame::Eval(node));
    }

    fn prepare_command(&mut self, interp: &mut Interp, mut command: PreparedCommand) {
        if let Some((word, source)) = next_output_process_substitution(&command) {
            if !self.ensure_capacity(interp, 1) {
                cleanup_prepared_process_substitutions(interp, &command);
                restore_command_variables(interp, command.temporary_variables);
                return;
            }
            let source = source.to_string();
            let path = match create_process_substitution_file(interp, &[]) {
                Ok(path) => path,
                Err(message) => {
                    write_diagnostic(interp, &message);
                    cleanup_prepared_process_substitutions(interp, &command);
                    restore_command_variables(interp, command.temporary_variables);
                    self.status = 125;
                    return;
                }
            };
            command.words[word] = path.clone();
            command
                .output_substitutions
                .push(OutputProcessSubstitution { path, source });
            self.push(interp, ShellFrame::PrepareCommand(command));
            return;
        }
        if let Some((location, source)) = next_process_substitution(&command) {
            let (pid, capture) = match start_command_substitution(interp, source) {
                Ok(child) => child,
                Err((status, message)) => {
                    if !message.is_empty() {
                        write_diagnostic(interp, &message);
                    }
                    cleanup_prepared_process_substitutions(interp, &command);
                    restore_command_variables(interp, command.temporary_variables);
                    self.status = status;
                    return;
                }
            };
            self.frames.push(ShellFrame::AwaitProcessSubstitution {
                command,
                pid,
                capture,
                location,
            });
            self.switched = true;
            return;
        }
        if let Some((location, substitution)) = next_command_substitution(&command) {
            if !self.ensure_capacity(interp, 1) {
                cleanup_prepared_process_substitutions(interp, &command);
                restore_command_variables(interp, command.temporary_variables);
                return;
            }
            let (variable, previous) = match new_substitution_variable(interp) {
                Ok(variable) => variable,
                Err(message) => {
                    write_diagnostic(interp, &message);
                    cleanup_prepared_process_substitutions(interp, &command);
                    restore_command_variables(interp, command.temporary_variables);
                    self.status = 125;
                    return;
                }
            };
            let replacement = format!("${{{variable}}}");
            replace_substitution(&mut command, location, &substitution, &replacement);
            let (pid, capture) = match start_command_substitution(interp, &substitution.source) {
                Ok(child) => child,
                Err((status, message)) => {
                    if !message.is_empty() {
                        write_diagnostic(interp, &message);
                    }
                    cleanup_prepared_process_substitutions(interp, &command);
                    restore_command_variables(interp, command.temporary_variables);
                    self.status = status;
                    return;
                }
            };
            self.frames.push(ShellFrame::AwaitCommandSubstitution {
                command,
                pid,
                capture,
                variable,
                previous,
            });
            self.switched = true;
            return;
        }

        let needed = 1
            + usize::from(!command.redirects.is_empty())
            + usize::from(!command.output_substitutions.is_empty())
            + usize::from(!command.process_substitution_files.is_empty());
        if !self.ensure_capacity(interp, needed) {
            cleanup_prepared_process_substitutions(interp, &command);
            restore_command_variables(interp, command.temporary_variables);
            return;
        }
        let descriptor_only_exec = matches!(command.words.as_slice(), [word] if word == "exec")
            || matches!(command.words.as_slice(), [word, separator] if word == "exec" && separator == "--");
        let redirect_scope = if command.redirects.is_empty() {
            None
        } else {
            match begin_redirects(interp, &command.redirects) {
                Ok(scope) if descriptor_only_exec => {
                    commit_redirects(interp, scope);
                    None
                }
                Ok(scope) => Some(scope),
                Err(error) => {
                    write_diagnostic(interp, &format!("shellsim: redirection: {error}\n"));
                    cleanup_prepared_process_substitutions(interp, &command);
                    restore_command_variables(interp, command.temporary_variables);
                    self.status = 1;
                    return;
                }
            }
        };
        if !command.output_substitutions.is_empty() {
            self.frames.push(ShellFrame::RunOutputSubstitutions {
                substitutions: command.output_substitutions,
                command_status: None,
            });
        }
        if !command.process_substitution_files.is_empty() {
            self.frames
                .push(ShellFrame::CleanupProcessSubstitutionFiles(
                    command.process_substitution_files,
                ));
        }
        if let Some(scope) = redirect_scope {
            self.frames.push(ShellFrame::RestoreRedirect(scope));
        }
        self.frames.push(ShellFrame::RunCommand {
            assigns: command.assigns,
            words: command.words,
            temporary_variables: command.temporary_variables,
            substitution_status: command.substitution_status,
        });
    }

    fn prepare_for(&mut self, interp: &mut Interp, mut state: PreparedFor) {
        if let Some((index, substitution)) =
            state.words.iter().enumerate().find_map(|(index, word)| {
                crate::expand::find_command_substitution(word)
                    .map(|substitution| (index, substitution))
            })
        {
            if !self.ensure_capacity(interp, 1) {
                restore_command_variables(interp, state.temporary_variables);
                return;
            }
            let (variable, previous) = match new_substitution_variable(interp) {
                Ok(variable) => variable,
                Err(message) => {
                    write_diagnostic(interp, &message);
                    restore_command_variables(interp, state.temporary_variables);
                    self.status = 125;
                    return;
                }
            };
            state.words[index].replace_range(
                substitution.start..substitution.end,
                &format!("${{{variable}}}"),
            );
            let (pid, capture) = match start_command_substitution(interp, &substitution.source) {
                Ok(child) => child,
                Err((status, message)) => {
                    if !message.is_empty() {
                        write_diagnostic(interp, &message);
                    }
                    restore_command_variables(interp, state.temporary_variables);
                    self.status = status;
                    return;
                }
            };
            self.frames.push(ShellFrame::AwaitForSubstitution {
                state,
                pid,
                capture,
                variable,
                previous,
            });
            self.switched = true;
            return;
        }
        if !self.ensure_capacity(interp, 1) {
            restore_command_variables(interp, state.temporary_variables);
            return;
        }
        let items = expand_words(interp, &state.words);
        if let Some(error) = interp.expansion_error.take() {
            write_diagnostic(interp, &error.message);
            restore_command_variables(interp, state.temporary_variables);
            self.status = error.status;
            if error.abort_shell {
                interp.exiting = Some(error.status);
            }
            return;
        }
        restore_command_variables(interp, state.temporary_variables);
        self.status = 0;
        self.frames.push(ShellFrame::ForNext {
            var: state.var,
            items,
            body: state.body,
            next: 0,
            body_status: 0,
        });
    }

    fn prepare_case(&mut self, interp: &mut Interp, mut state: PreparedCase) {
        let found = crate::expand::find_command_substitution(&state.word)
            .map(|substitution| (None, substitution))
            .or_else(|| {
                state
                    .arms
                    .iter()
                    .enumerate()
                    .find_map(|(arm, (patterns, _))| {
                        patterns.iter().enumerate().find_map(|(pattern, value)| {
                            crate::expand::find_command_substitution(value)
                                .map(|substitution| (Some((arm, pattern)), substitution))
                        })
                    })
            });
        if let Some((location, substitution)) = found {
            if !self.ensure_capacity(interp, 1) {
                restore_command_variables(interp, state.temporary_variables);
                return;
            }
            let (variable, previous) = match new_substitution_variable(interp) {
                Ok(variable) => variable,
                Err(message) => {
                    write_diagnostic(interp, &message);
                    restore_command_variables(interp, state.temporary_variables);
                    self.status = 125;
                    return;
                }
            };
            let target = match location {
                None => &mut state.word,
                Some((arm, pattern)) => &mut state.arms[arm].0[pattern],
            };
            target.replace_range(
                substitution.start..substitution.end,
                &format!("${{{variable}}}"),
            );
            let (pid, capture) = match start_command_substitution(interp, &substitution.source) {
                Ok(child) => child,
                Err((status, message)) => {
                    if !message.is_empty() {
                        write_diagnostic(interp, &message);
                    }
                    restore_command_variables(interp, state.temporary_variables);
                    self.status = status;
                    return;
                }
            };
            self.frames.push(ShellFrame::AwaitCaseSubstitution {
                state,
                pid,
                capture,
                variable,
                previous,
            });
            self.switched = true;
            return;
        }
        if !self.ensure_capacity(interp, 1) {
            restore_command_variables(interp, state.temporary_variables);
            return;
        }
        let subject = expand_word(interp, &state.word, false).join(" ");
        let arms = state.arms;
        let expanded_patterns = arms
            .iter()
            .map(|(patterns, _)| {
                patterns
                    .iter()
                    .map(|pattern| expand_word(interp, pattern, false).join(" "))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if let Some(error) = interp.expansion_error.take() {
            write_diagnostic(interp, &error.message);
            restore_command_variables(interp, state.temporary_variables);
            self.status = error.status;
            if error.abort_shell {
                interp.exiting = Some(error.status);
            }
            return;
        }
        restore_command_variables(interp, state.temporary_variables);
        self.status = 0;
        'arms: for ((_, body), patterns) in arms.into_iter().zip(expanded_patterns) {
            for pattern in patterns {
                if case_match(&pattern, &subject) {
                    self.frames.push(ShellFrame::Eval(body));
                    break 'arms;
                }
            }
        }
    }

    fn prepare_arithmetic(&mut self, interp: &mut Interp, state: PreparedArithmetic) {
        if let Some(substitution) = crate::expand::find_command_substitution(&state.expression) {
            if !self.ensure_capacity(interp, 1) {
                return;
            }
            let (pid, capture) = match start_command_substitution(interp, &substitution.source) {
                Ok(child) => child,
                Err((status, message)) => {
                    if !message.is_empty() {
                        write_diagnostic(interp, &message);
                    }
                    self.status = status;
                    return;
                }
            };
            self.frames.push(ShellFrame::AwaitArithmeticSubstitution {
                state,
                pid,
                capture,
                start: substitution.start,
                end: substitution.end,
            });
            self.switched = true;
            return;
        }

        let value = if matches!(
            &state.continuation,
            ArithmeticContinuation::CForCondition { .. }
        ) && state.expression.is_empty()
        {
            1
        } else {
            // Unlike `$(( ))`, a failed `(( ))` or arithmetic `for` clause only fails the
            // command; the enclosing loop stops with status 1.
            match crate::arith::evaluate(interp, &state.expression) {
                Ok(value) => value,
                Err(error) => {
                    write_diagnostic(
                        interp,
                        &format!("shellsim: ((: {}\n", error.describe(&state.expression)),
                    );
                    self.status = 1;
                    return;
                }
            }
        };
        match state.continuation {
            ArithmeticContinuation::Command => {
                self.status = i32::from(value == 0);
            }
            ArithmeticContinuation::CForInit { cond, update, body } => {
                self.status = 0;
                self.push(
                    interp,
                    ShellFrame::CForCheck {
                        cond,
                        update,
                        body,
                        body_status: 0,
                    },
                );
            }
            ArithmeticContinuation::CForCondition {
                cond,
                update,
                body,
                body_status,
            } => {
                if should_unwind(interp) || value == 0 {
                    self.status = body_status;
                } else if self.push(
                    interp,
                    ShellFrame::CForAfterBody {
                        cond,
                        update,
                        body: body.clone(),
                    },
                ) {
                    self.push(interp, ShellFrame::Eval(body));
                }
            }
            ArithmeticContinuation::CForUpdate {
                cond,
                update,
                body,
                body_status,
            } => {
                self.push(
                    interp,
                    ShellFrame::CForCheck {
                        cond,
                        update,
                        body,
                        body_status,
                    },
                );
            }
        }
    }

    fn step(&mut self, interp: &mut Interp, frame: ShellFrame) {
        match frame {
            ShellFrame::Eval(node) => self.eval(interp, node),
            ShellFrame::ReadProgram { pending } => self.read_program(interp, pending),
            ShellFrame::ExitWithStatus => interp.exiting = Some(self.status),
            ShellFrame::PrepareCommand(command) => self.prepare_command(interp, command),
            ShellFrame::RunCommand {
                assigns,
                words,
                temporary_variables,
                substitution_status,
            } => self.eval_command(
                interp,
                assigns,
                words,
                temporary_variables,
                substitution_status,
            ),
            ShellFrame::AwaitCommandSubstitution {
                mut command,
                pid,
                capture,
                variable,
                previous,
            } => {
                let (status, value) = finish_command_substitution(interp, pid, capture);
                if !self.ensure_capacity(interp, 1) {
                    cleanup_prepared_process_substitutions(interp, &command);
                    restore_command_variables(interp, command.temporary_variables);
                    return;
                }
                interp.set_var(&variable, value);
                command.temporary_variables.push((variable, previous));
                command.substitution_status = Some(status);
                self.push(interp, ShellFrame::PrepareCommand(command));
            }
            ShellFrame::AwaitProcessSubstitution {
                mut command,
                pid,
                capture,
                location,
            } => {
                let (_, bytes) = finish_substitution(interp, pid, capture);
                if !self.ensure_capacity(interp, 1) {
                    cleanup_prepared_process_substitutions(interp, &command);
                    restore_command_variables(interp, command.temporary_variables);
                    return;
                }
                match location {
                    ProcessSubstitutionLocation::Redirect(redirect) => {
                        command.redirects[redirect].op = RedirOp::HeredocRaw;
                        command.redirects[redirect].target =
                            String::from_utf8_lossy(&bytes).into_owned();
                    }
                    ProcessSubstitutionLocation::Word(word) => {
                        let path = match create_process_substitution_file(interp, &bytes) {
                            Ok(path) => path,
                            Err(message) => {
                                write_diagnostic(interp, &message);
                                cleanup_prepared_process_substitutions(interp, &command);
                                restore_command_variables(interp, command.temporary_variables);
                                self.status = 125;
                                return;
                            }
                        };
                        command.words[word] = path.clone();
                        command.process_substitution_files.push(path);
                    }
                }
                self.push(interp, ShellFrame::PrepareCommand(command));
            }
            ShellFrame::RunOutputSubstitutions {
                mut substitutions,
                command_status,
            } => {
                let command_status = command_status.unwrap_or(self.status);
                let Some(substitution) = substitutions.pop() else {
                    self.status = command_status;
                    return;
                };
                let bytes = interp.fs_read("/", &substitution.path).unwrap_or_default();
                let _ = interp.vfs.remove_file("/", &substitution.path);
                match start_process_substitution_consumer(interp, &substitution.source, bytes) {
                    Ok(pid) => {
                        self.frames
                            .push(ShellFrame::AwaitOutputProcessSubstitution {
                                substitutions,
                                command_status,
                                pid,
                            });
                        self.switched = true;
                    }
                    Err((status, message)) => {
                        if !message.is_empty() {
                            write_diagnostic(interp, &message);
                        }
                        for substitution in substitutions {
                            let _ = interp.vfs.remove_file("/", &substitution.path);
                        }
                        self.status = if command_status == 0 {
                            status
                        } else {
                            command_status
                        };
                    }
                }
            }
            ShellFrame::AwaitOutputProcessSubstitution {
                substitutions,
                command_status,
                pid,
            } => {
                interp.processes.reap(pid);
                let _ = interp.scheduler.reap(pid);
                self.status = command_status;
                self.push(
                    interp,
                    ShellFrame::RunOutputSubstitutions {
                        substitutions,
                        command_status: Some(command_status),
                    },
                );
            }
            ShellFrame::CleanupProcessSubstitutionFiles(paths) => {
                for path in paths {
                    let _ = interp.vfs.remove_file("/", &path);
                }
            }
            ShellFrame::PrepareFor(state) => self.prepare_for(interp, state),
            ShellFrame::AwaitForSubstitution {
                mut state,
                pid,
                capture,
                variable,
                previous,
            } => {
                let (_, value) = finish_command_substitution(interp, pid, capture);
                interp.set_var(&variable, value);
                state.temporary_variables.push((variable, previous));
                self.push(interp, ShellFrame::PrepareFor(state));
            }
            ShellFrame::PrepareCase(state) => self.prepare_case(interp, state),
            ShellFrame::AwaitCaseSubstitution {
                mut state,
                pid,
                capture,
                variable,
                previous,
            } => {
                let (_, value) = finish_command_substitution(interp, pid, capture);
                interp.set_var(&variable, value);
                state.temporary_variables.push((variable, previous));
                self.push(interp, ShellFrame::PrepareCase(state));
            }
            ShellFrame::PrepareArithmetic(state) => self.prepare_arithmetic(interp, state),
            ShellFrame::AwaitArithmeticSubstitution {
                mut state,
                pid,
                capture,
                start,
                end,
            } => {
                let (_, value) = finish_command_substitution(interp, pid, capture);
                let value = if value.trim().is_empty() { "0" } else { &value };
                state
                    .expression
                    .replace_range(start..end, &format!("({value})"));
                self.push(interp, ShellFrame::PrepareArithmetic(state));
            }
            ShellFrame::Sequence { nodes, next } => {
                if next > 0 && should_unwind(interp) {
                    if interp.deadline_interrupt.is_some() {
                        self.status = 124;
                    }
                    return;
                }
                if next > 0
                    && self.status != 0
                    && interp.opt_errexit
                    && interp.cond_depth == 0
                    && !self.errexit_exempt
                {
                    interp.exiting = Some(self.status);
                    return;
                }
                if let Some(node) = nodes.get(next).cloned() {
                    if self.push(
                        interp,
                        ShellFrame::Sequence {
                            nodes,
                            next: next + 1,
                        },
                    ) {
                        self.push(interp, ShellFrame::Eval(node));
                    }
                }
            }
            ShellFrame::Conditional {
                rhs,
                run_on_success,
            } => {
                interp.cond_depth = interp.cond_depth.saturating_sub(1);
                if !should_unwind(interp) && (self.status == 0) == run_on_success {
                    self.errexit_exempt = false;
                    self.push(interp, ShellFrame::Eval(rhs));
                } else if !should_unwind(interp) {
                    self.errexit_exempt = true;
                }
            }
            ShellFrame::FinishTime(start) => {
                let report = time_report(
                    start,
                    interp.clock.monotonic_ns(),
                    interp.resources.process_time_ns(),
                );
                write_diagnostic(interp, &report);
            }
            ShellFrame::Negate => {
                interp.cond_depth = interp.cond_depth.saturating_sub(1);
                self.status = i32::from(self.status == 0);
                self.errexit_exempt = true;
            }
            ShellFrame::IfNext {
                branches,
                next,
                els,
            } => {
                if let Some((condition, _)) = branches.get(next).cloned() {
                    interp.cond_depth = interp.cond_depth.saturating_add(1);
                    if self.push(
                        interp,
                        ShellFrame::IfAfter {
                            branches,
                            next,
                            els,
                        },
                    ) {
                        self.push(interp, ShellFrame::Eval(condition));
                    }
                } else if let Some(body) = els {
                    self.push(interp, ShellFrame::Eval(body));
                } else {
                    self.status = 0;
                }
            }
            ShellFrame::IfAfter {
                branches,
                next,
                els,
            } => {
                interp.cond_depth = interp.cond_depth.saturating_sub(1);
                if self.status == 0 && !should_unwind(interp) {
                    self.push(interp, ShellFrame::Eval(branches[next].1.clone()));
                } else if !should_unwind(interp) {
                    self.push(
                        interp,
                        ShellFrame::IfNext {
                            branches,
                            next: next + 1,
                            els,
                        },
                    );
                }
            }
            ShellFrame::WhileCheck {
                cond,
                body,
                until,
                body_status,
            } => {
                if should_unwind(interp) {
                    self.status = body_status;
                    return;
                }
                interp.cond_depth = interp.cond_depth.saturating_add(1);
                if self.push(
                    interp,
                    ShellFrame::WhileAfterCondition {
                        cond: cond.clone(),
                        body,
                        until,
                        body_status,
                    },
                ) {
                    self.push(interp, ShellFrame::Eval(cond));
                }
            }
            ShellFrame::WhileAfterCondition {
                cond,
                body,
                until,
                body_status,
            } => {
                interp.cond_depth = interp.cond_depth.saturating_sub(1);
                let enter = if until {
                    self.status != 0
                } else {
                    self.status == 0
                };
                if enter && !should_unwind(interp) {
                    if self.push(
                        interp,
                        ShellFrame::WhileAfterBody {
                            cond,
                            body: body.clone(),
                            until,
                        },
                    ) {
                        self.push(interp, ShellFrame::Eval(body));
                    }
                } else {
                    self.status = body_status;
                }
            }
            ShellFrame::WhileAfterBody { cond, body, until } => {
                if interp.loop_break > 0 {
                    interp.loop_break -= 1;
                    return;
                }
                if interp.loop_continue > 0 {
                    interp.loop_continue -= 1;
                }
                if !should_unwind(interp) {
                    self.push(
                        interp,
                        ShellFrame::WhileCheck {
                            cond,
                            body,
                            until,
                            body_status: self.status,
                        },
                    );
                }
            }
            ShellFrame::ForNext {
                var,
                items,
                body,
                next,
                body_status,
            } => {
                if should_unwind(interp) {
                    self.status = body_status;
                } else if let Some(item) = items.get(next).cloned() {
                    interp.set_var(&var, item);
                    if self.push(
                        interp,
                        ShellFrame::ForAfterBody {
                            var,
                            items,
                            body: body.clone(),
                            next: next + 1,
                        },
                    ) {
                        self.push(interp, ShellFrame::Eval(body));
                    }
                } else {
                    self.status = body_status;
                }
            }
            ShellFrame::ForAfterBody {
                var,
                items,
                body,
                next,
            } => {
                if interp.loop_break > 0 {
                    interp.loop_break -= 1;
                    return;
                }
                if interp.loop_continue > 0 {
                    interp.loop_continue -= 1;
                }
                if !should_unwind(interp) {
                    self.push(
                        interp,
                        ShellFrame::ForNext {
                            var,
                            items,
                            body,
                            next,
                            body_status: self.status,
                        },
                    );
                }
            }
            ShellFrame::CForCheck {
                cond,
                update,
                body,
                body_status,
            } => {
                if should_unwind(interp) {
                    self.status = body_status;
                } else {
                    self.push(
                        interp,
                        ShellFrame::PrepareArithmetic(PreparedArithmetic {
                            expression: cond.clone(),
                            continuation: ArithmeticContinuation::CForCondition {
                                cond,
                                update,
                                body,
                                body_status,
                            },
                        }),
                    );
                }
            }
            ShellFrame::CForAfterBody { cond, update, body } => {
                if interp.loop_break > 0 {
                    interp.loop_break -= 1;
                    return;
                }
                if interp.loop_continue > 0 {
                    interp.loop_continue -= 1;
                }
                if !should_unwind(interp) {
                    self.push(
                        interp,
                        ShellFrame::PrepareArithmetic(PreparedArithmetic {
                            expression: update.clone(),
                            continuation: ArithmeticContinuation::CForUpdate {
                                cond,
                                update,
                                body,
                                body_status: self.status,
                            },
                        }),
                    );
                }
            }
            ShellFrame::FinishLoop => {
                interp.loop_depth = interp.loop_depth.saturating_sub(1);
            }
            ShellFrame::RestoreRedirect(scope) => end_redirects(interp, scope),
            ShellFrame::FinishFunction {
                positional,
                variables,
            } => {
                interp.leave_function_scope();
                interp.positional = positional;
                self.status = interp.returning.take().unwrap_or(self.status);
                restore_command_variables(interp, variables);
            }
            ShellFrame::FinishInlineCommand { variables } => {
                restore_command_variables(interp, variables);
                interp.invocations.finish_latest(
                    interp.process.pid,
                    self.status,
                    interp.resources.cpu_used(),
                    interp.vfs.disk_used(),
                );
            }
            ShellFrame::FinishSource { variables } => {
                interp.source_depth = interp.source_depth.saturating_sub(1);
                self.status = interp.returning.take().unwrap_or(self.status);
                restore_command_variables(interp, variables);
                interp.invocations.finish_latest(
                    interp.process.pid,
                    self.status,
                    interp.resources.cpu_used(),
                    interp.vfs.disk_used(),
                );
            }
            ShellFrame::FinishSignalHandler { previous_status } => {
                interp.finish_signal_handler();
                self.status = previous_status;
            }
            ShellFrame::FinishExitTrap { previous_status } => {
                self.status = interp.exiting.take().unwrap_or(previous_status);
            }
            ShellFrame::ResumeCommand {
                variables,
                continuation,
            } => {
                let result = if interp.deadline_interrupt.is_some() {
                    interp.invocations.finish_latest(
                        interp.process.pid,
                        124,
                        interp.resources.cpu_used(),
                        interp.vfs.disk_used(),
                    );
                    crate::commands::CommandPoll::Ready(124)
                } else {
                    crate::commands::resume(interp, continuation)
                };
                match result {
                    crate::commands::CommandPoll::Ready(status) => {
                        self.status = status;
                        restore_command_variables(interp, variables);
                    }
                    crate::commands::CommandPoll::ReadyOutput {
                        command,
                        status,
                        stdout,
                        stderr,
                    } => {
                        self.frames.push(ShellFrame::WriteCommandOutput {
                            command,
                            variables,
                            status,
                            stdout,
                            stderr,
                            stdout_offset: 0,
                            stderr_offset: 0,
                        });
                    }
                    crate::commands::CommandPoll::Yielded(continuation) => {
                        self.frames.push(ShellFrame::ResumeCommand {
                            variables,
                            continuation,
                        });
                        self.yielded = true;
                    }
                    crate::commands::CommandPoll::Switched(continuation) => {
                        self.retain_switched_command(variables, continuation);
                    }
                    crate::commands::CommandPoll::Inline(node) => {
                        self.enter_inline_command(interp, variables, node);
                    }
                    crate::commands::CommandPoll::InlineSource(node) => {
                        self.enter_source(interp, variables, node);
                    }
                    crate::commands::CommandPoll::Blocked(reason, continuation) => {
                        self.frames.push(ShellFrame::ResumeCommand {
                            variables,
                            continuation,
                        });
                        self.blocked = Some(reason);
                    }
                }
            }
            ShellFrame::ReadCommandInput {
                argv,
                variables,
                mut stdin,
                mut reserved,
            } => {
                if crate::commands::buffers_standard_input(interp, &argv) {
                    match interp.read_fd(0, crate::descriptors::DEFAULT_PIPE_CAPACITY) {
                        Ok(IoPoll::Ready(bytes)) if !bytes.is_empty() => {
                            let bytes_len = bytes.len() as u64;
                            if !interp.resources.reserve_memory(bytes_len) {
                                interp.resources.release_memory(reserved);
                                restore_command_variables(interp, variables);
                                self.status = 137;
                                return;
                            }
                            reserved = reserved.saturating_add(bytes_len);
                            stdin.extend_from_slice(&bytes);
                            self.frames.push(ShellFrame::ReadCommandInput {
                                argv,
                                variables,
                                stdin,
                                reserved,
                            });
                            return;
                        }
                        Ok(IoPoll::Blocked(wait)) => {
                            self.frames.push(ShellFrame::ReadCommandInput {
                                argv,
                                variables,
                                stdin,
                                reserved,
                            });
                            self.blocked = Some(io_wait_reason(wait));
                            return;
                        }
                        Ok(IoPoll::Ready(_)) => {}
                        Err(error) => {
                            interp.resources.release_memory(reserved);
                            write_diagnostic(interp, &format!("shellsim: {}: {error}\n", argv[0]));
                            restore_command_variables(interp, variables);
                            self.status = 1;
                            return;
                        }
                    }
                }
                let mut stdout = Vec::new();
                let mut stderr = Vec::new();
                let result = crate::commands::poll(interp, &argv, stdin, &mut stdout, &mut stderr);
                interp.resources.release_memory(reserved);
                match result {
                    crate::commands::CommandPoll::Ready(status) => {
                        self.frames.push(ShellFrame::WriteCommandOutput {
                            command: argv[0].clone(),
                            variables,
                            status,
                            stdout,
                            stderr,
                            stdout_offset: 0,
                            stderr_offset: 0,
                        });
                    }
                    crate::commands::CommandPoll::ReadyOutput {
                        command,
                        status,
                        stdout,
                        stderr,
                    } => {
                        self.frames.push(ShellFrame::WriteCommandOutput {
                            command,
                            variables,
                            status,
                            stdout,
                            stderr,
                            stdout_offset: 0,
                            stderr_offset: 0,
                        });
                    }
                    crate::commands::CommandPoll::Yielded(continuation) => {
                        debug_assert!(stdout.is_empty() && stderr.is_empty());
                        self.frames.push(ShellFrame::ResumeCommand {
                            variables,
                            continuation,
                        });
                        self.yielded = true;
                    }
                    crate::commands::CommandPoll::Switched(continuation) => {
                        debug_assert!(stdout.is_empty() && stderr.is_empty());
                        self.retain_switched_command(variables, continuation);
                    }
                    crate::commands::CommandPoll::Inline(node) => {
                        debug_assert!(stdout.is_empty() && stderr.is_empty());
                        self.enter_inline_command(interp, variables, node);
                    }
                    crate::commands::CommandPoll::InlineSource(node) => {
                        debug_assert!(stdout.is_empty() && stderr.is_empty());
                        self.enter_source(interp, variables, node);
                    }
                    crate::commands::CommandPoll::Blocked(reason, continuation) => {
                        debug_assert!(stdout.is_empty() && stderr.is_empty());
                        self.frames.push(ShellFrame::ResumeCommand {
                            variables,
                            continuation,
                        });
                        self.blocked = Some(reason);
                    }
                }
            }
            ShellFrame::WriteCommandOutput {
                command,
                variables,
                mut status,
                stdout,
                mut stderr,
                mut stdout_offset,
                mut stderr_offset,
            } => {
                let (fd, bytes, offset) = if stdout_offset < stdout.len() {
                    (1, &stdout, &mut stdout_offset)
                } else if stderr_offset < stderr.len() {
                    (2, &stderr, &mut stderr_offset)
                } else {
                    self.status = status;
                    restore_command_variables(interp, variables);
                    return;
                };
                match interp.write_fd_checked(fd, &bytes[*offset..]) {
                    Ok(IoPoll::Ready(0)) => {
                        *offset = bytes.len();
                        status = 1;
                    }
                    Ok(IoPoll::Ready(written)) => *offset = offset.saturating_add(written),
                    Ok(IoPoll::Blocked(wait)) => self.blocked = Some(io_wait_reason(wait)),
                    // As with a kernel EPIPE, the writer receives SIGPIPE; its default action
                    // ends the process silently with status 141.
                    Err(crate::syscalls::SyscallError::Descriptor(
                        crate::descriptors::DescriptorError::BrokenPipe,
                    )) if !matches!(
                        interp
                            .signal_dispositions
                            .get(&crate::process::Signal::Pipe),
                        Some(crate::interp::ShellSignalDisposition::Ignore)
                    ) =>
                    {
                        stdout_offset = stdout.len();
                        stderr_offset = stderr.len();
                        status = 128 + crate::process::Signal::Pipe.number();
                        let pid = interp.process.pid;
                        if let Err(error) = interp.send_signal(pid, crate::process::Signal::Pipe) {
                            write_diagnostic(interp, &format!("shellsim: {error}\n"));
                        }
                    }
                    Err(error) => {
                        let error = crate::interp::write_error_message(error);
                        *offset = bytes.len();
                        status = 1;
                        if fd == 1 {
                            stderr.extend_from_slice(
                                format!("shellsim: {command}: {error}\n").as_bytes(),
                            );
                        }
                    }
                }
                self.frames.push(ShellFrame::WriteCommandOutput {
                    command,
                    variables,
                    status,
                    stdout,
                    stderr,
                    stdout_offset,
                    stderr_offset,
                });
            }
            ShellFrame::AwaitChild { pid, reap } => {
                match interp.processes.get(pid).map(|record| record.status) {
                    Some(crate::process::ProcessStatus::Exited(status)) => {
                        self.status = status;
                        if reap {
                            interp.processes.reap(pid);
                            let _ = interp.scheduler.reap(pid);
                        }
                    }
                    Some(
                        crate::process::ProcessStatus::Running
                        | crate::process::ProcessStatus::Stopped(_),
                    ) => {
                        self.frames.push(ShellFrame::AwaitChild { pid, reap });
                        self.blocked = Some(crate::scheduler::WaitReason::Child(pid));
                    }
                    None => {
                        write_diagnostic(interp, &format!("shellsim: child {pid} disappeared\n"));
                        self.status = 125;
                    }
                }
            }
            ShellFrame::AwaitPipeline {
                pids,
                mut statuses,
                mut next,
                pipefail,
            } => {
                while let Some(pid) = pids.get(next).copied() {
                    match interp.processes.get(pid).map(|record| record.status) {
                        Some(crate::process::ProcessStatus::Exited(status)) => {
                            statuses.push(status);
                            interp.processes.reap(pid);
                            let _ = interp.scheduler.reap(pid);
                            next += 1;
                        }
                        _ => {
                            self.frames.push(ShellFrame::AwaitPipeline {
                                pids,
                                statuses,
                                next,
                                pipefail,
                            });
                            self.blocked = Some(crate::scheduler::WaitReason::Child(pid));
                            return;
                        }
                    }
                }
                let last = statuses.last().copied().unwrap_or(0);
                self.status = if pipefail {
                    statuses
                        .into_iter()
                        .rev()
                        .find(|status| *status != 0)
                        .unwrap_or(last)
                } else {
                    last
                };
            }
        }
    }

    fn eval(&mut self, interp: &mut Interp, node: Node) {
        if !interp.resources.charge_cpu(10) {
            self.status = interp
                .resources
                .stop_reason()
                .map_or(137, |reason| reason.exit_status());
            return;
        }
        if should_unwind(interp) {
            self.status = if interp.deadline_interrupt.is_some() {
                124
            } else {
                interp.last_status
            };
            return;
        }
        match node {
            Node::Empty => self.status = 0,
            Node::Command {
                assigns,
                words,
                redirects,
            } => {
                self.errexit_exempt = false;
                self.push(
                    interp,
                    ShellFrame::PrepareCommand(PreparedCommand {
                        assigns,
                        words,
                        redirects,
                        temporary_variables: Vec::new(),
                        substitution_status: None,
                        output_substitutions: Vec::new(),
                        process_substitution_files: Vec::new(),
                    }),
                );
            }
            Node::ArgvCommand(argv) => {
                self.errexit_exempt = false;
                self.eval_external_argv(interp, argv, Vec::new(), true);
            }
            Node::Pipeline(stages) => {
                self.errexit_exempt = false;
                self.spawn_pipeline(interp, stages);
            }
            Node::And(lhs, rhs) => {
                interp.cond_depth = interp.cond_depth.saturating_add(1);
                if self.push(
                    interp,
                    ShellFrame::Conditional {
                        rhs: *rhs,
                        run_on_success: true,
                    },
                ) {
                    self.push(interp, ShellFrame::Eval(*lhs));
                }
            }
            Node::Or(lhs, rhs) => {
                interp.cond_depth = interp.cond_depth.saturating_add(1);
                if self.push(
                    interp,
                    ShellFrame::Conditional {
                        rhs: *rhs,
                        run_on_success: false,
                    },
                ) {
                    self.push(interp, ShellFrame::Eval(*lhs));
                }
            }
            Node::Timed { inner, posix } => {
                let start = TimeStart {
                    monotonic_ns: interp.clock.monotonic_ns(),
                    cpu_ns: interp.resources.process_time_ns(),
                    posix,
                };
                if self.push(interp, ShellFrame::FinishTime(start)) {
                    self.push(interp, ShellFrame::Eval(*inner));
                }
            }
            Node::Not(inner) => {
                interp.cond_depth = interp.cond_depth.saturating_add(1);
                if self.push(interp, ShellFrame::Negate) {
                    self.push(interp, ShellFrame::Eval(*inner));
                }
            }
            Node::Seq(nodes) => {
                self.status = 0;
                self.push(interp, ShellFrame::Sequence { nodes, next: 0 });
            }
            Node::Background(inner) => self.status = exec_background(interp, &inner),
            Node::Subshell(inner) => {
                self.errexit_exempt = false;
                self.spawn_child(interp, *inner, "(subshell)", true);
            }
            Node::Group(inner) => {
                self.push(interp, ShellFrame::Eval(*inner));
            }
            Node::Redirected(inner, redirects) => match begin_redirects(interp, &redirects) {
                Ok(scope) => {
                    if self.push(interp, ShellFrame::RestoreRedirect(scope)) {
                        self.push(interp, ShellFrame::Eval(*inner));
                    }
                }
                Err(error) => {
                    write_diagnostic(interp, &format!("shellsim: redirection: {error}\n"));
                    self.status = 1;
                }
            },
            Node::If {
                cond,
                then,
                elifs,
                els,
            } => {
                let mut branches = Vec::with_capacity(elifs.len() + 1);
                branches.push((*cond, *then));
                branches.extend(elifs);
                self.push(
                    interp,
                    ShellFrame::IfNext {
                        branches,
                        next: 0,
                        els: els.map(|node| *node),
                    },
                );
            }
            Node::While { cond, body, until } => {
                self.status = 0;
                if !self.ensure_capacity(interp, 2) {
                    return;
                }
                interp.loop_depth = interp.loop_depth.saturating_add(1);
                self.frames.push(ShellFrame::FinishLoop);
                self.frames.push(ShellFrame::WhileCheck {
                    cond: *cond,
                    body: *body,
                    until,
                    body_status: 0,
                });
            }
            Node::For { var, words, body } => {
                if !self.ensure_capacity(interp, 2) {
                    return;
                }
                interp.loop_depth = interp.loop_depth.saturating_add(1);
                self.frames.push(ShellFrame::FinishLoop);
                self.frames.push(ShellFrame::PrepareFor(PreparedFor {
                    var,
                    words,
                    body: *body,
                    temporary_variables: Vec::new(),
                }));
            }
            Node::CFor {
                init,
                cond,
                update,
                body,
            } => {
                if !self.ensure_capacity(interp, 2) {
                    return;
                }
                interp.loop_depth = interp.loop_depth.saturating_add(1);
                self.frames.push(ShellFrame::FinishLoop);
                self.frames
                    .push(ShellFrame::PrepareArithmetic(PreparedArithmetic {
                        expression: init,
                        continuation: ArithmeticContinuation::CForInit {
                            cond,
                            update,
                            body: *body,
                        },
                    }));
            }
            Node::Case { word, arms } => {
                self.push(
                    interp,
                    ShellFrame::PrepareCase(PreparedCase {
                        word,
                        arms,
                        temporary_variables: Vec::new(),
                    }),
                );
            }
            Node::FuncDef { name, body } => {
                interp.funcs.insert(name, *body);
                self.status = 0;
            }
            Node::Arithmetic(expression) => {
                self.push(
                    interp,
                    ShellFrame::PrepareArithmetic(PreparedArithmetic {
                        expression,
                        continuation: ArithmeticContinuation::Command,
                    }),
                );
            }
        }
    }

    fn eval_command(
        &mut self,
        interp: &mut Interp,
        assigns: Vec<(String, String)>,
        words: Vec<String>,
        temporary_variables: Vec<(String, Option<String>)>,
        substitution_status: Option<i32>,
    ) {
        let argv = expand_argv(interp, &words);
        if let Some(error) = interp.expansion_error.take() {
            write_diagnostic(interp, &error.message);
            restore_command_variables(interp, temporary_variables);
            self.status = error.status;
            if error.abort_shell {
                interp.exiting = Some(error.status);
            }
            return;
        }
        let argv = match expand_alias_argv(interp, argv) {
            Ok(argv) => argv,
            Err(message) => {
                write_diagnostic(interp, &format!("shellsim: alias: {message}\n"));
                restore_command_variables(interp, temporary_variables);
                self.status = 2;
                return;
            }
        };
        if argv.is_empty() {
            for (key, value) in &assigns {
                apply_assignment(interp, key, value);
                if let Some(error) = interp.expansion_error.take() {
                    write_diagnostic(interp, &error.message);
                    restore_command_variables(interp, temporary_variables);
                    self.status = error.status;
                    if error.abort_shell {
                        interp.exiting = Some(error.status);
                    }
                    return;
                }
            }
            restore_command_variables(interp, temporary_variables);
            self.status = substitution_status.unwrap_or(0);
            return;
        }
        let variables = install_command_variables(interp, &assigns);
        if let Some(error) = interp.expansion_error.take() {
            write_diagnostic(interp, &error.message);
            restore_command_variables(interp, variables);
            restore_command_variables(interp, temporary_variables);
            self.status = error.status;
            if error.abort_shell {
                interp.exiting = Some(error.status);
            }
            return;
        }
        restore_command_variables(interp, temporary_variables);
        interp.cmd_trace.record(&argv[0]);
        if argv[0] == "exec" && !interp.funcs.contains_key("exec") {
            self.exec_builtin(interp, argv, variables);
        } else if let Some(body) = interp.funcs.get(&argv[0]).cloned() {
            let positional = std::mem::replace(&mut interp.positional, argv[1..].to_vec());
            if !self.ensure_capacity(interp, 2) {
                interp.positional = positional;
                restore_command_variables(interp, variables);
                return;
            }
            interp.enter_function_scope();
            self.frames.push(ShellFrame::FinishFunction {
                positional,
                variables,
            });
            self.frames.push(ShellFrame::Eval(body));
        } else {
            self.eval_external_argv(interp, argv, variables, false);
        }
    }

    /// Run the `exec` builtin with a command operand. A native image replaces this process,
    /// keeping its PID and committing every pending redirection; the rest of the program and
    /// any EXIT trap are discarded, as with `execve`. Programs that cannot be loaded as an image
    /// (scripts and legacy bodies) run as a child, and the shell then exits with their status.
    /// `exec` with only redirections is handled when its command is prepared.
    fn exec_builtin(
        &mut self,
        interp: &mut Interp,
        mut argv: Vec<String>,
        variables: Vec<(String, Option<String>)>,
    ) {
        argv.remove(0);
        if argv.first().is_some_and(|argument| argument == "--") {
            argv.remove(0);
        }
        if let Some(option) = argv
            .first()
            .filter(|argument| argument.len() > 1 && argument.starts_with('-'))
        {
            let message = if matches!(option.as_str(), "-a" | "-c" | "-l") {
                format!("shellsim: exec: {option}: unsupported option\n")
            } else {
                format!("shellsim: exec: {option}: invalid option\n")
            };
            write_diagnostic(interp, &message);
            restore_command_variables(interp, variables);
            self.status = 2;
            return;
        }
        if argv.is_empty() {
            restore_command_variables(interp, variables);
            self.status = 0;
            return;
        }
        // `exec` bypasses builtins and functions: `exec printf` loads /usr/bin/printf.
        let image = match crate::commands::util::resolve_executable(interp, &argv[0]) {
            crate::commands::util::ExecutableLookup::Found(path) => path,
            // A failed exec leaves the shell in place, so its EXIT trap still runs.
            crate::commands::util::ExecutableLookup::NotExecutable(path) => {
                write_diagnostic(
                    interp,
                    &format!("shellsim: exec: {path}: Permission denied\n"),
                );
                restore_command_variables(interp, variables);
                self.status = 126;
                interp.exiting = Some(126);
                return;
            }
            crate::commands::util::ExecutableLookup::NotFound => {
                write_diagnostic(interp, &format!("shellsim: exec: {}: not found\n", argv[0]));
                restore_command_variables(interp, variables);
                self.status = 127;
                interp.exiting = Some(127);
                return;
            }
        };
        interp.exit_disposition = None;
        if !crate::commands::execs_native_image(interp, &image) {
            if !self.ensure_capacity(interp, 1) {
                restore_command_variables(interp, variables);
                return;
            }
            self.frames.push(ShellFrame::ExitWithStatus);
            self.eval_external_argv(interp, argv, variables, false);
            return;
        }
        let mut remaining = Vec::new();
        for frame in std::mem::take(&mut self.frames) {
            match frame {
                ShellFrame::RestoreRedirect(scope) => commit_redirects(interp, scope),
                frame => remaining.push(frame),
            }
        }
        self.frames = remaining;
        self.abort(interp);
        let pid = interp.process.pid;
        if let Err(error) = interp.exec_argv_image(pid, argv) {
            write_diagnostic(interp, &format!("shellsim: exec: {error}\n"));
            self.status = 126;
            interp.exiting = Some(126);
            return;
        }
        self.replaced = true;
    }

    /// Replace this subshell's image with `argv` when nothing else remains to run in it.
    /// Pending redirection restores are committed instead, since the process never returns to
    /// the shell. A pending EXIT trap keeps the shell image alive.
    fn exec_in_place(&mut self, interp: &mut Interp, argv: &[String]) -> bool {
        if !self.exec_tail
            || interp.exit_disposition.is_some()
            || !self
                .frames
                .iter()
                .all(|frame| matches!(frame, ShellFrame::RestoreRedirect(_)))
            || !crate::commands::execs_native_image(interp, &argv[0])
        {
            return false;
        }
        for frame in std::mem::take(&mut self.frames) {
            if let ShellFrame::RestoreRedirect(scope) = frame {
                commit_redirects(interp, scope);
            }
        }
        let pid = interp.process.pid;
        if let Err(error) = interp.exec_argv_image(pid, argv.to_vec()) {
            write_diagnostic(interp, &format!("shellsim: exec: {error}\n"));
            self.status = 126;
            return true;
        }
        self.replaced = true;
        true
    }

    fn eval_external_argv(
        &mut self,
        interp: &mut Interp,
        argv: Vec<String>,
        variables: Vec<(String, Option<String>)>,
        record_trace: bool,
    ) {
        if argv.is_empty() {
            self.status = 0;
            return;
        }
        if record_trace {
            interp.cmd_trace.record(&argv[0]);
        }
        if self.exec_in_place(interp, &argv) {
            return;
        }
        if crate::commands::starts_before_input(interp, &argv) {
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            match crate::commands::poll(interp, &argv, Vec::new(), &mut stdout, &mut stderr) {
                crate::commands::CommandPoll::Ready(status) => {
                    self.frames.push(ShellFrame::WriteCommandOutput {
                        command: argv[0].clone(),
                        variables,
                        status,
                        stdout,
                        stderr,
                        stdout_offset: 0,
                        stderr_offset: 0,
                    });
                }
                crate::commands::CommandPoll::ReadyOutput {
                    command,
                    status,
                    stdout,
                    stderr,
                } => {
                    self.frames.push(ShellFrame::WriteCommandOutput {
                        command,
                        variables,
                        status,
                        stdout,
                        stderr,
                        stdout_offset: 0,
                        stderr_offset: 0,
                    });
                }
                crate::commands::CommandPoll::Yielded(continuation) => {
                    debug_assert!(stdout.is_empty() && stderr.is_empty());
                    self.frames.push(ShellFrame::ResumeCommand {
                        variables,
                        continuation,
                    });
                    self.yielded = true;
                }
                crate::commands::CommandPoll::Switched(continuation) => {
                    debug_assert!(stdout.is_empty() && stderr.is_empty());
                    self.retain_switched_command(variables, continuation);
                }
                crate::commands::CommandPoll::Inline(node) => {
                    debug_assert!(stdout.is_empty() && stderr.is_empty());
                    self.enter_inline_command(interp, variables, node);
                }
                crate::commands::CommandPoll::InlineSource(node) => {
                    debug_assert!(stdout.is_empty() && stderr.is_empty());
                    self.enter_source(interp, variables, node);
                }
                crate::commands::CommandPoll::Blocked(reason, continuation) => {
                    self.frames.push(ShellFrame::ResumeCommand {
                        variables,
                        continuation,
                    });
                    self.blocked = Some(reason);
                }
            }
        } else {
            self.frames.push(ShellFrame::ReadCommandInput {
                argv,
                variables,
                stdin: Vec::new(),
                reserved: 0,
            });
        }
    }

    fn spawn_child(&mut self, interp: &mut Interp, node: Node, command: &str, reap: bool) {
        if !self.ensure_capacity(interp, 1) {
            return;
        }
        let parent_pid = interp.process.pid;
        match interp.start_child(command, false) {
            Ok(pid) => {
                self.frames.push(ShellFrame::AwaitChild { pid, reap });
                interp
                    .process
                    .set_continuation(pid, Some(ShellContinuation::new(&node)))
                    .expect("new child process state must exist");
                self.switched = true;
                debug_assert_ne!(parent_pid, interp.process.pid);
            }
            Err(error) => {
                write_diagnostic(interp, &format!("shellsim: {error}\n"));
                self.status = 125;
            }
        }
    }

    fn spawn_pipeline(&mut self, interp: &mut Interp, stages: Vec<Node>) {
        if stages.is_empty() {
            self.status = 0;
            return;
        }
        if !self.ensure_capacity(interp, 1) {
            return;
        }
        let mut pipes = Vec::with_capacity(stages.len().saturating_sub(1));
        for _ in 1..stages.len() {
            match interp
                .descriptors
                .open_pipe(crate::descriptors::DEFAULT_PIPE_CAPACITY)
            {
                Ok(pipe) => pipes.push(pipe),
                Err(error) => {
                    for (reader, writer) in pipes {
                        let _ = interp.descriptors.discard_unreferenced(reader);
                        let _ = interp.descriptors.discard_unreferenced(writer);
                    }
                    write_diagnostic(interp, &format!("shellsim: pipeline: {error:?}\n"));
                    self.status = 125;
                    return;
                }
            }
        }

        let mut pids = Vec::with_capacity(stages.len());
        for (index, stage) in stages.iter().enumerate() {
            let command = describe(stage);
            let pid = match interp.start_pipeline_child(&command, false) {
                Ok(pid) => pid,
                Err(error) => {
                    rollback_pipeline(interp, pids, pipes);
                    write_diagnostic(interp, &format!("shellsim: pipeline: {error}\n"));
                    self.status = 125;
                    return;
                }
            };
            let setup = (|| {
                if let Some((reader, _)) = index.checked_sub(1).and_then(|i| pipes.get(i)) {
                    interp
                        .install_process_description(pid, 0, *reader)
                        .map_err(|error| format!("{error:?}"))?;
                }
                if let Some((_, writer)) = pipes.get(index) {
                    interp
                        .install_process_description(pid, 1, *writer)
                        .map_err(|error| format!("{error:?}"))?;
                }
                interp
                    .process
                    .set_continuation(pid, Some(ShellContinuation::subshell(stage)))
            })();
            if let Err(error) = setup {
                interp.cancel_unstarted_child(pid);
                rollback_pipeline(interp, pids, pipes);
                write_diagnostic(interp, &format!("shellsim: pipeline: {error}\n"));
                self.status = 125;
                return;
            }
            pids.push(pid);
        }
        self.frames.push(ShellFrame::AwaitPipeline {
            pids: pids.clone(),
            statuses: Vec::with_capacity(pids.len()),
            next: 0,
            pipefail: interp.opt_pipefail,
        });
        self.blocked = Some(crate::scheduler::WaitReason::Child(pids[0]));
    }
}

fn next_command_substitution(
    command: &PreparedCommand,
) -> Option<(ExpansionLocation, crate::expand::CommandSubstitution)> {
    for (index, (key, value)) in command.assigns.iter().enumerate() {
        if let Some(substitution) = crate::expand::find_command_substitution(key) {
            return Some((ExpansionLocation::AssignmentKey(index), substitution));
        }
        if let Some(substitution) = crate::expand::find_command_substitution(value) {
            return Some((ExpansionLocation::AssignmentValue(index), substitution));
        }
    }
    for (index, word) in command.words.iter().enumerate() {
        if let Some(substitution) = crate::expand::find_command_substitution(word) {
            return Some((ExpansionLocation::Word(index), substitution));
        }
    }
    for (index, redirect) in command.redirects.iter().enumerate() {
        if redirect.op != RedirOp::HeredocRaw {
            if let Some(substitution) = crate::expand::find_command_substitution(&redirect.target) {
                return Some((ExpansionLocation::Redirect(index), substitution));
            }
        }
    }
    None
}

fn process_substitution_source(value: &str) -> Option<&str> {
    value.strip_prefix("<(")?.strip_suffix(')')
}

fn output_process_substitution_source(value: &str) -> Option<&str> {
    value.strip_prefix(">(")?.strip_suffix(')')
}

fn next_process_substitution(
    command: &PreparedCommand,
) -> Option<(ProcessSubstitutionLocation, &str)> {
    command
        .redirects
        .iter()
        .enumerate()
        .find_map(|(index, redirect)| {
            (redirect.op == RedirOp::Read)
                .then(|| process_substitution_source(&redirect.target))
                .flatten()
                .map(|source| (ProcessSubstitutionLocation::Redirect(index), source))
        })
        .or_else(|| {
            command.words.iter().enumerate().find_map(|(index, word)| {
                process_substitution_source(word)
                    .map(|source| (ProcessSubstitutionLocation::Word(index), source))
            })
        })
}

fn next_output_process_substitution(command: &PreparedCommand) -> Option<(usize, &str)> {
    command.words.iter().enumerate().find_map(|(index, word)| {
        output_process_substitution_source(word).map(|source| (index, source))
    })
}

fn create_process_substitution_file(interp: &mut Interp, bytes: &[u8]) -> Result<String, String> {
    let identifier = interp
        .next_temp_id()
        .ok_or_else(|| "shellsim: process substitution identity exhausted\n".to_string())?;
    let path = format!("/tmp/.shellsim-process-substitution-{identifier}");
    interp.sync_vfs_time();
    interp
        .vfs
        .write("/", &path, bytes, 0o600)
        .map_err(|error| format!("shellsim: process substitution: {error}\n"))?;
    Ok(path)
}

fn cleanup_prepared_process_substitutions(interp: &mut Interp, command: &PreparedCommand) {
    for substitution in &command.output_substitutions {
        let _ = interp.vfs.remove_file("/", &substitution.path);
    }
    for path in &command.process_substitution_files {
        let _ = interp.vfs.remove_file("/", path);
    }
}

fn replace_substitution(
    command: &mut PreparedCommand,
    location: ExpansionLocation,
    substitution: &crate::expand::CommandSubstitution,
    replacement: &str,
) {
    let target = match location {
        ExpansionLocation::AssignmentKey(index) => &mut command.assigns[index].0,
        ExpansionLocation::AssignmentValue(index) => &mut command.assigns[index].1,
        ExpansionLocation::Word(index) => &mut command.words[index],
        ExpansionLocation::Redirect(index) => &mut command.redirects[index].target,
    };
    target.replace_range(substitution.start..substitution.end, replacement);
}

fn new_substitution_variable(interp: &mut Interp) -> Result<(String, Option<String>), String> {
    loop {
        let identifier = interp
            .next_temp_id()
            .ok_or_else(|| "shellsim: command substitution identity exhausted\n".to_string())?;
        let variable = format!("__SHELLSIM_COMMAND_SUBSTITUTION_{identifier}");
        if interp.get_var(&variable).is_none() && !interp.exported.contains(&variable) {
            return Ok((variable, None));
        }
        if !interp.resources.charge_cpu(1) {
            return Err(
                "shellsim: resource limit exceeded during command substitution\n".to_string(),
            );
        }
    }
}

fn finish_command_substitution(
    interp: &mut Interp,
    pid: crate::process::ProcessId,
    capture: crate::descriptors::DescriptionId,
) -> (i32, String) {
    let (status, mut bytes) = finish_substitution(interp, pid, capture);
    while bytes.last() == Some(&b'\n') {
        bytes.pop();
    }
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn finish_substitution(
    interp: &mut Interp,
    pid: crate::process::ProcessId,
    capture: crate::descriptors::DescriptionId,
) -> (i32, Vec<u8>) {
    let status = match interp.processes.get(pid).map(|record| record.status) {
        Some(crate::process::ProcessStatus::Exited(status)) => status,
        _ => {
            write_diagnostic(
                interp,
                &format!("shellsim: command substitution child {pid} did not exit\n"),
            );
            125
        }
    };
    let bytes = interp
        .descriptors
        .drain_capture(capture)
        .unwrap_or_default();
    let _ = interp.descriptors.release_handle(capture);
    interp.processes.reap(pid);
    let _ = interp.scheduler.reap(pid);
    (status, bytes)
}

fn start_command_substitution(
    interp: &mut Interp,
    source: &str,
) -> Result<(crate::process::ProcessId, crate::descriptors::DescriptionId), (i32, String)> {
    let mut diagnostic = Vec::new();
    let ast = crate::commands::parse_shell_source(interp, source, &mut diagnostic)
        .map_err(|status| (status, String::from_utf8_lossy(&diagnostic).into_owned()))?;
    let input = interp
        .descriptors
        .open_input(Vec::new())
        .map_err(|error| (125, format!("shellsim: command substitution: {error:?}\n")))?;
    let capture = match interp.descriptors.open_capture() {
        Ok(capture) => capture,
        Err(error) => {
            let _ = interp.descriptors.discard_unreferenced(input);
            return Err((125, format!("shellsim: command substitution: {error:?}\n")));
        }
    };
    if let Err(error) = interp.descriptors.retain_handle(capture) {
        let _ = interp.descriptors.discard_unreferenced(input);
        let _ = interp.descriptors.discard_unreferenced(capture);
        return Err((125, format!("shellsim: command substitution: {error:?}\n")));
    }
    let pid = match interp.start_child("$(command substitution)", false) {
        Ok(pid) => pid,
        Err(error) => {
            let _ = interp.descriptors.discard_unreferenced(input);
            let _ = interp.descriptors.release_handle(capture);
            return Err((125, format!("shellsim: {error}\n")));
        }
    };
    interp
        .install_process_description(pid, 0, input)
        .expect("command substitution child must accept a valid input description");
    interp
        .install_process_description(pid, 1, capture)
        .expect("command substitution child must accept a valid capture description");
    interp
        .process
        .set_continuation(pid, Some(ShellContinuation::new(&ast)))
        .expect("command substitution child must accept a continuation");
    Ok((pid, capture))
}

fn start_process_substitution_consumer(
    interp: &mut Interp,
    source: &str,
    bytes: Vec<u8>,
) -> Result<crate::process::ProcessId, (i32, String)> {
    let mut diagnostic = Vec::new();
    let ast = crate::commands::parse_shell_source(interp, source, &mut diagnostic)
        .map_err(|status| (status, String::from_utf8_lossy(&diagnostic).into_owned()))?;
    let input = interp
        .descriptors
        .open_input(bytes)
        .map_err(|error| (125, format!("shellsim: process substitution: {error:?}\n")))?;
    let pid = match interp.start_child(">(process substitution)", false) {
        Ok(pid) => pid,
        Err(error) => {
            let _ = interp.descriptors.discard_unreferenced(input);
            return Err((125, format!("shellsim: {error}\n")));
        }
    };
    interp
        .install_process_description(pid, 0, input)
        .expect("process substitution child must accept a valid input description");
    interp
        .process
        .set_continuation(pid, Some(ShellContinuation::new(&ast)))
        .expect("process substitution child must accept a continuation");
    Ok(pid)
}

fn rollback_pipeline(
    interp: &mut Interp,
    pids: Vec<crate::process::ProcessId>,
    pipes: Vec<(
        crate::descriptors::DescriptionId,
        crate::descriptors::DescriptionId,
    )>,
) {
    for pid in pids {
        interp.cancel_unstarted_child(pid);
    }
    for (reader, writer) in pipes {
        let _ = interp.descriptors.discard_unreferenced(reader);
        let _ = interp.descriptors.discard_unreferenced(writer);
    }
}

fn should_unwind(interp: &Interp) -> bool {
    interp.resources.is_stopped()
        || interp.exiting.is_some()
        || interp.returning.is_some()
        || interp.loop_break > 0
        || interp.loop_continue > 0
        || interp.deadline_interrupt.is_some()
}

/// Result of one bounded machine scheduling quantum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MachinePoll {
    /// One process or scheduler transition made progress.
    Progress,
    /// Every retained process is blocked and time advancement was disabled or impossible.
    Blocked,
    /// The requested foreground continuation completed.
    Ready(i32),
}

/// Install a shell continuation without driving the scheduler.
pub(crate) fn start_node(interp: &mut Interp, node: &Node) -> Result<u32, i32> {
    if interp.process.program.is_some() {
        write_diagnostic(
            interp,
            "shellsim: attempted to replace an active shell continuation\n",
        );
        return Err(125);
    }
    let target_pid = interp.process.pid;
    interp.process.program = Some(crate::program::ProgramContinuation::Shell(
        ShellContinuation::new(node),
    ));
    if interp.scheduler.has_runnable() {
        interp
            .scheduler
            .yield_current()
            .expect("active process must own the running scheduler slot");
        let scheduled = interp
            .scheduler
            .dispatch()
            .expect("runnable queue must contain valid tasks")
            .expect("a runnable task was checked above");
        interp
            .process
            .activate(scheduled)
            .expect("scheduled task must own process state");
    }
    Ok(target_pid)
}

/// Poll one scheduler/process quantum for a retained foreground continuation.
///
/// When `advance_time` is false, an all-blocked machine is returned to its host-side driver
/// without moving virtual time. When true, the next modeled event may fire before one runnable
/// process is selected. No host clock or host process is consulted.
pub(crate) fn poll_machine(
    interp: &mut Interp,
    target_pid: u32,
    advance_time: bool,
) -> Result<MachinePoll, String> {
    if interp.scheduler.current().is_none() && !dispatch_available(interp, advance_time)? {
        return Ok(MachinePoll::Blocked);
    }
    let owner_pid = interp.process.pid;
    if let Some(delivery) = interp.take_signal_delivery() {
        match delivery {
            crate::interp::SignalDelivery::Terminate(signal) => {
                let status = 128 + signal.number();
                if let Some(mut program) = interp.process.program.take() {
                    program.release_owned_memory(interp);
                }
                interp.invocations.finish_process(
                    owner_pid,
                    status,
                    interp.resources.cpu_used(),
                    interp.vfs.disk_used(),
                );
                if owner_pid == target_pid {
                    interp.process.program = None;
                    interp.exiting = Some(status);
                    return Ok(MachinePoll::Ready(status));
                }
                interp.finish_child(owner_pid, status);
                return Ok(MachinePoll::Progress);
            }
            crate::interp::SignalDelivery::Stop(signal) => {
                interp.stop_active_process(signal)?;
                return if dispatch_available(interp, false)? {
                    Ok(MachinePoll::Progress)
                } else {
                    Ok(MachinePoll::Blocked)
                };
            }
            crate::interp::SignalDelivery::Handler(body) => {
                let mut continuation =
                    interp.process.program.take().ok_or_else(|| {
                        format!("active signaled process {owner_pid} has no program")
                    })?;
                continuation.inject_signal_handler(interp, body)?;
                interp.process.program = Some(continuation);
            }
        }
    }
    wake_due_events(interp)?;
    let owner_pid = interp.process.pid;
    let mut continuation = interp
        .process
        .program
        .take()
        .ok_or_else(|| format!("active process {owner_pid} has no program"))?;
    match continuation.poll(interp, SHELL_POLL_QUANTUM) {
        ShellPoll::Pending => {
            interp.process.set_program(owner_pid, Some(continuation))?;
            if interp.scheduler.has_runnable() {
                interp
                    .scheduler
                    .yield_current()
                    .map_err(|error| format!("{error:?}"))?;
                dispatch_available(interp, false)?;
            }
            Ok(MachinePoll::Progress)
        }
        ShellPoll::Switched => {
            interp.process.set_program(owner_pid, Some(continuation))?;
            Ok(MachinePoll::Progress)
        }
        ShellPoll::Replaced => {
            continuation.release_owned_memory(interp);
            Ok(MachinePoll::Progress)
        }
        ShellPoll::Blocked(reason) => {
            interp.process.set_program(owner_pid, Some(continuation))?;
            interp
                .scheduler
                .block_current(reason)
                .map_err(|error| format!("{error:?}"))?;
            if dispatch_available(interp, advance_time)? {
                Ok(MachinePoll::Progress)
            } else {
                Ok(MachinePoll::Blocked)
            }
        }
        ShellPoll::Ready(status) if owner_pid == target_pid => {
            interp.invocations.finish_process(
                owner_pid,
                status,
                interp.resources.cpu_used(),
                interp.vfs.disk_used(),
            );
            Ok(MachinePoll::Ready(status))
        }
        ShellPoll::Ready(status) => {
            interp.invocations.finish_process(
                owner_pid,
                status,
                interp.resources.cpu_used(),
                interp.vfs.disk_used(),
            );
            interp.finish_child(owner_pid, status);
            Ok(MachinePoll::Progress)
        }
    }
}

fn dispatch_available(interp: &mut Interp, advance_time: bool) -> Result<bool, String> {
    loop {
        if let Some(pid) = interp
            .scheduler
            .dispatch()
            .map_err(|error| format!("{error:?}"))?
        {
            interp.process.activate(pid)?;
            return Ok(true);
        }
        if !advance_time {
            return Ok(false);
        }
        let fired = interp
            .clock
            .advance_to_next()
            .map_err(|error| error.to_string())?;
        if fired.is_empty() {
            return Ok(false);
        }
        handle_ready_events(interp)?;
    }
}

/// Run at most one continuation quantum outside the active caller, then restore that caller.
///
/// Python live-process methods use this as a nested cooperative scheduling boundary. The Python
/// VM remains on the Rust stack while its logical process is temporarily blocked, so the caller
/// itself must never be polled by this function. Other runnable logical processes retain FIFO
/// ordering, and virtual time advances only up to `advance_until`.
pub(crate) fn drive_scheduler_step(
    interp: &mut Interp,
    target: crate::process::ProcessId,
    advance_until: Option<u64>,
) -> Result<bool, String> {
    if matches!(
        interp.processes.get(target).map(|record| record.status),
        Some(crate::process::ProcessStatus::Exited(_))
    ) {
        return Ok(true);
    }
    let caller = interp.process.pid;
    interp
        .scheduler
        .block_current(crate::scheduler::WaitReason::Child(target))
        .map_err(|error| format!("unable to suspend process {caller}: {error:?}"))?;

    let scheduled = match interp
        .scheduler
        .dispatch()
        .map_err(|error| format!("{error:?}"))?
    {
        Some(pid) => Some(pid),
        None => {
            let Some(next) = interp.clock.next_deadline_ns() else {
                restore_nested_caller(interp, caller)?;
                return Err("all child processes are blocked without a pending event".to_string());
            };
            if advance_until.is_some_and(|limit| next > limit) {
                interp
                    .clock
                    .advance_to(advance_until.expect("limit was checked"))
                    .map_err(|error| error.to_string())?;
                restore_nested_caller(interp, caller)?;
                return Ok(false);
            }
            interp
                .clock
                .advance_to_next()
                .map_err(|error| error.to_string())?;
            handle_ready_events(interp)?;
            interp
                .scheduler
                .dispatch()
                .map_err(|error| format!("{error:?}"))?
        }
    };

    if let Some(pid) = scheduled {
        interp.process.activate(pid)?;
        poll_active_nested_process(interp)?;
    }
    restore_nested_caller(interp, caller)?;
    Ok(matches!(
        interp.processes.get(target).map(|record| record.status),
        Some(crate::process::ProcessStatus::Exited(_))
    ))
}

fn poll_active_nested_process(interp: &mut Interp) -> Result<(), String> {
    wake_due_events(interp)?;
    let owner = interp.process.pid;
    if let Some(delivery) = interp.take_signal_delivery() {
        match delivery {
            crate::interp::SignalDelivery::Terminate(signal) => {
                let status = 128 + signal.number();
                interp.invocations.finish_process(
                    owner,
                    status,
                    interp.resources.cpu_used(),
                    interp.vfs.disk_used(),
                );
                interp.finish_child(owner, status);
                return Ok(());
            }
            crate::interp::SignalDelivery::Stop(signal) => {
                interp.stop_active_process(signal)?;
                return Ok(());
            }
            crate::interp::SignalDelivery::Handler(body) => {
                let mut continuation = interp
                    .process
                    .program
                    .take()
                    .ok_or_else(|| format!("scheduled process {owner} has no program"))?;
                continuation.inject_signal_handler(interp, body)?;
                interp.process.program = Some(continuation);
            }
        }
    }
    let mut continuation = interp
        .process
        .program
        .take()
        .ok_or_else(|| format!("scheduled process {owner} has no program"))?;
    match continuation.poll(interp, SHELL_POLL_QUANTUM) {
        ShellPoll::Pending => {
            interp.process.set_program(owner, Some(continuation))?;
            interp
                .scheduler
                .yield_current()
                .map_err(|error| format!("{error:?}"))?;
        }
        ShellPoll::Switched => {
            interp.process.set_program(owner, Some(continuation))?;
        }
        ShellPoll::Replaced => continuation.release_owned_memory(interp),
        ShellPoll::Blocked(reason) => {
            interp.process.set_program(owner, Some(continuation))?;
            interp
                .scheduler
                .block_current(reason)
                .map_err(|error| format!("{error:?}"))?;
        }
        ShellPoll::Ready(status) => {
            interp.invocations.finish_process(
                owner,
                status,
                interp.resources.cpu_used(),
                interp.vfs.disk_used(),
            );
            interp.finish_child(owner, status);
        }
    }
    Ok(())
}

fn restore_nested_caller(
    interp: &mut Interp,
    caller: crate::process::ProcessId,
) -> Result<(), String> {
    if interp.scheduler.current() == Some(caller) {
        interp.process.activate(caller)?;
        return Ok(());
    }
    if interp.scheduler.current().is_some() {
        interp
            .scheduler
            .yield_current()
            .map_err(|error| format!("{error:?}"))?;
    }
    if matches!(
        interp.scheduler.state(caller),
        Some(crate::scheduler::TaskState::Blocked(_))
    ) {
        interp
            .scheduler
            .wake(caller)
            .map_err(|error| format!("{error:?}"))?;
    }
    loop {
        let pid = interp
            .scheduler
            .dispatch()
            .map_err(|error| format!("{error:?}"))?
            .ok_or_else(|| "nested scheduler lost its caller".to_string())?;
        if pid == caller {
            interp.process.activate(caller)?;
            return Ok(());
        }
        interp
            .scheduler
            .yield_current()
            .map_err(|error| format!("{error:?}"))?;
    }
}

fn wake_due_events(interp: &mut Interp) -> Result<(), String> {
    interp
        .clock
        .advance_to(interp.clock.monotonic_ns())
        .map_err(|error| error.to_string())?;
    handle_ready_events(interp)
}

fn handle_ready_events(interp: &mut Interp) -> Result<(), String> {
    while let Some(event) = interp.clock.pop_ready() {
        match event.kind {
            crate::clock::EventKind::WakeTask { task } => {
                let Ok(pid) = crate::process::ProcessId::try_from(task) else {
                    continue;
                };
                let deadline = event.id.deadline_ns();
                let timed_wait = match interp.scheduler.state(pid) {
                    Some(crate::scheduler::TaskState::Blocked(reason))
                    | Some(crate::scheduler::TaskState::Stopped(
                        crate::scheduler::StoppedTask::Blocked(reason),
                    )) => reason.accepts_timer(deadline),
                    _ => false,
                };
                if timed_wait {
                    interp
                        .scheduler
                        .wake(pid)
                        .map_err(|error| format!("{error:?}"))?;
                }
            }
            crate::clock::EventKind::Deadline { task } => {
                let waiters = if task == crate::clock::MAIN_TASK_ID {
                    interp.scheduler.timer_waiters()
                } else {
                    crate::process::ProcessId::try_from(task)
                        .ok()
                        .into_iter()
                        .collect()
                };
                for pid in waiters {
                    interp.process.set_deadline_interrupt(pid, event.id)?;
                    interp
                        .scheduler
                        .wake(pid)
                        .map_err(|error| format!("{error:?}"))?;
                }
            }
            crate::clock::EventKind::SignalTask {
                task,
                signal,
                descendants,
            } => {
                if let Ok(pid) = crate::process::ProcessId::try_from(task) {
                    let targets = if descendants {
                        interp.processes.live_process_tree(pid)
                    } else {
                        vec![pid]
                    };
                    for target in targets.into_iter().rev() {
                        let _ = interp.send_signal(target, signal);
                    }
                }
            }
            crate::clock::EventKind::External { .. } => {}
        }
    }
    Ok(())
}

fn exec_background(interp: &mut Interp, node: &Node) -> i32 {
    let command = describe(node);
    let pid = match interp.start_background_child(&command, false) {
        Ok(pid) => pid,
        Err(error) => {
            write_diagnostic(interp, &format!("shellsim: {error}\n"));
            return 125;
        }
    };
    interp
        .process
        .set_continuation(pid, Some(ShellContinuation::subshell(node)))
        .expect("new background child state must exist");
    let Some(id) = interp.new_job(pid, command) else {
        interp.cancel_unstarted_child(pid);
        write_diagnostic(interp, "shellsim: job table limit exceeded\n");
        return 125;
    };
    debug_assert!(interp
        .jobs
        .iter()
        .any(|job| job.id == id && job.state == crate::interp::JobState::Running));
    interp.set_var("!", pid.to_string());
    0
}

fn describe(node: &Node) -> String {
    match node {
        Node::Command { words, .. } => words.join(" "),
        Node::ArgvCommand(argv) => argv.join(" "),
        _ => "<job>".to_string(),
    }
}

#[derive(Clone)]
struct RedirectScope {
    saved: crate::descriptors::FdTable,
    reserved_memory: u64,
}

fn begin_redirects(interp: &mut Interp, redirects: &[Redirect]) -> Result<RedirectScope, String> {
    let snapshot_bytes = interp.vfs.disk_used().saturating_add(4 * 1024);
    if !interp.resources.reserve_memory(snapshot_bytes) {
        return Err("memory limit exceeded while preparing redirections".to_string());
    }
    let vfs_before = interp.vfs.clone();
    let saved = match interp.process.fds.fork(&mut interp.descriptors) {
        Ok(saved) => saved,
        Err(error) => {
            interp.resources.release_memory(snapshot_bytes);
            return Err(format!("{error:?}"));
        }
    };
    if let Err(error) = apply_redirects(interp, redirects) {
        restore_fds(interp, saved);
        interp.vfs = vfs_before;
        interp.resources.release_memory(snapshot_bytes);
        return Err(error);
    }
    Ok(RedirectScope {
        saved,
        reserved_memory: snapshot_bytes,
    })
}

fn end_redirects(interp: &mut Interp, scope: RedirectScope) {
    restore_fds(interp, scope.saved);
    interp.resources.release_memory(scope.reserved_memory);
}

fn commit_redirects(interp: &mut Interp, mut scope: RedirectScope) {
    scope.saved.close_all(&mut interp.descriptors);
    interp.resources.release_memory(scope.reserved_memory);
}

/// Apply redirections from left to right. `dup` retains the open description selected at that
/// point, so `2>&1 >file` and `>file 2>&1` have distinct, Bash-compatible destinations.
fn apply_redirects(interp: &mut Interp, redirects: &[Redirect]) -> Result<(), String> {
    for redirect in redirects {
        match redirect.op {
            RedirOp::Read => {
                let path = redirect_path(interp, &redirect.target)?;
                if let Some(source) = device_fd(&path) {
                    ActiveSystem::new(interp)
                        .duplicate(source, redirect.fd)
                        .map_err(|error| format!("{path}: {error}"))?;
                } else {
                    open_redirect_file(interp, redirect.fd, &path, &redirect.op)?;
                }
            }
            RedirOp::ReadWrite => {
                let path = redirect_path(interp, &redirect.target)?;
                open_redirect_file(interp, redirect.fd, &path, &redirect.op)?;
            }
            RedirOp::Heredoc | RedirOp::HeredocRaw | RedirOp::HereString => {
                let mut bytes = match redirect.op {
                    RedirOp::Heredoc => expand_heredoc(interp, &redirect.target).into_bytes(),
                    RedirOp::HeredocRaw => redirect.target.clone().into_bytes(),
                    RedirOp::HereString => {
                        let mut bytes = expand_word(interp, &redirect.target, false)
                            .join(" ")
                            .into_bytes();
                        bytes.push(b'\n');
                        bytes
                    }
                    _ => unreachable!(),
                };
                if let Some(error) = interp.expansion_error.take() {
                    if error.abort_shell {
                        interp.exiting = Some(error.status);
                    }
                    return Err(error.message.trim().to_string());
                }
                let description = interp
                    .descriptors
                    .open_input(std::mem::take(&mut bytes))
                    .map_err(|error| format!("here document: {error:?}"))?;
                interp
                    .install_new_description(redirect.fd, description)
                    .map_err(|error| format!("here document: {error:?}"))?;
            }
            RedirOp::Write | RedirOp::Append => {
                let path = redirect_path(interp, &redirect.target)?;
                if let Some(source) = device_fd(&path) {
                    ActiveSystem::new(interp)
                        .duplicate(source, redirect.fd)
                        .map_err(|error| format!("{path}: {error}"))?;
                } else {
                    open_redirect_file(interp, redirect.fd, &path, &redirect.op)?;
                }
            }
            RedirOp::DupOut => {
                let source = redirect
                    .target
                    .trim_start_matches('&')
                    .parse::<i32>()
                    .map_err(|_| format!("bad file descriptor: {}", redirect.target))?;
                ActiveSystem::new(interp)
                    .duplicate(source, redirect.fd)
                    .map_err(|error| format!("{}: {error}", redirect.target))?;
            }
            RedirOp::Close => {
                ActiveSystem::new(interp)
                    .close(redirect.fd)
                    .map_err(|error| format!("{}: {error}", redirect.fd))?;
            }
        }
    }
    Ok(())
}

fn open_redirect_file(
    interp: &mut Interp,
    fd: i32,
    path: &str,
    op: &RedirOp,
) -> Result<(), String> {
    let (readable, writable, create, truncate, append) = match op {
        RedirOp::Read => (true, false, false, false, false),
        RedirOp::ReadWrite => (true, true, true, false, false),
        RedirOp::Write => (false, true, true, true, false),
        RedirOp::Append => (false, true, true, false, true),
        _ => unreachable!("only file-opening redirects reach this helper"),
    };
    ActiveSystem::new(interp)
        .open_file_at(
            fd,
            "/",
            path,
            OpenFile {
                readable,
                writable,
                create,
                exclusive: false,
                truncate,
                append,
            },
        )
        .map_err(|error| format!("{path}: {error}"))
}

fn redirect_path(interp: &mut Interp, word: &str) -> Result<String, String> {
    let fields = expand_word(interp, word, true);
    if let Some(error) = interp.expansion_error.take() {
        if error.abort_shell {
            interp.exiting = Some(error.status);
        }
        return Err(error.message.trim().to_string());
    }
    if fields.len() != 1 {
        return Err(format!("{word}: ambiguous redirect"));
    }
    Ok(resolve_against(&interp.cwd, &fields[0]))
}

fn device_fd(path: &str) -> Option<i32> {
    match path {
        "/dev/stdin" => Some(0),
        "/dev/stdout" => Some(1),
        "/dev/stderr" => Some(2),
        _ => path.strip_prefix("/dev/fd/")?.parse().ok(),
    }
}

fn write_all_fd(interp: &mut Interp, fd: i32, mut bytes: &[u8]) -> Result<(), String> {
    while !bytes.is_empty() {
        match interp.write_fd(fd, bytes)? {
            IoPoll::Ready(0) | IoPoll::Blocked(_) => {
                return Err("descriptor write would block".to_string());
            }
            IoPoll::Ready(written) => bytes = &bytes[written..],
        }
    }
    Ok(())
}

fn write_diagnostic(interp: &mut Interp, message: &str) {
    let _ = write_all_fd(interp, 2, message.as_bytes());
}

fn io_wait_reason(wait: crate::descriptors::IoWait) -> crate::scheduler::WaitReason {
    match wait {
        crate::descriptors::IoWait::InputReadable(description) => {
            crate::scheduler::WaitReason::InputReadable(description)
        }
        crate::descriptors::IoWait::PipeReadable(pipe) => {
            crate::scheduler::WaitReason::PipeReadable(pipe)
        }
        crate::descriptors::IoWait::PipeWritable(pipe) => {
            crate::scheduler::WaitReason::PipeWritable(pipe)
        }
    }
}

fn install_command_variables(
    interp: &mut Interp,
    assigns: &[(String, String)],
) -> Vec<(String, Option<String>)> {
    let expanded_assigns: Vec<(String, String)> = assigns
        .iter()
        .map(|(k, v)| (k.clone(), expand_word(interp, v, false).join(" ")))
        .collect();
    let saved: Vec<(String, Option<String>)> = expanded_assigns
        .iter()
        .map(|(k, _)| (k.clone(), interp.vars.get(k).cloned()))
        .collect();
    for (k, v) in &expanded_assigns {
        if interp.readonly.contains(k) {
            interp.expansion_error = Some(crate::interp::ShellExpansionError::assignment(format!(
                "shellsim: {k}: readonly variable\n"
            )));
            break;
        }
        interp.set_var(k, v.clone());
        interp.export(k); // exported to child for the command
    }
    saved
}

fn restore_command_variables(interp: &mut Interp, saved: Vec<(String, Option<String>)>) {
    for (key, value) in saved {
        match value {
            Some(value) => {
                interp.vars.insert(key, value);
            }
            None => {
                interp.vars.remove(&key);
            }
        }
    }
}

/// Expand a command's argv, but keep array-assignment literal words verbatim so the builtin
/// (`declare`/`local`/`typeset`/`readonly`) can parse them itself.
fn expand_argv(interp: &mut Interp, words: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let double_bracket = words.first().is_some_and(|word| word == "[[");
    for w in words {
        if is_array_assign_word(w) {
            out.push(w.clone());
        } else {
            out.extend(expand_word(interp, w, !double_bracket));
        }
    }
    out
}

fn expand_alias_argv(interp: &mut Interp, mut argv: Vec<String>) -> Result<Vec<String>, String> {
    let mut expanded = std::collections::BTreeSet::new();
    while let Some(command) = argv.first() {
        let Some(alias) = interp.aliases.get(command).cloned() else {
            break;
        };
        if !expanded.insert(command.clone()) || expanded.len() > 32 {
            return Err(format!("recursive alias involving {command:?}"));
        }
        let mut replacement = expand_argv(interp, &alias.words);
        if replacement.is_empty() {
            return Err(format!("alias {command:?} expands to an empty command"));
        }
        replacement.extend(argv.into_iter().skip(1));
        argv = replacement;
    }
    Ok(argv)
}

/// True if `w` is an array-assignment literal that must not be split/globbed:
/// `name=( … )`, `name+=( … )`, `name[sub]=…`, `name[sub]+=…`.
pub fn is_array_assign_word(w: &str) -> bool {
    let eq = match w.find('=') {
        Some(0) | None => return false,
        Some(e) => e,
    };
    let mut lhs = &w[..eq];
    if let Some(s) = lhs.strip_suffix('+') {
        lhs = s;
    }
    let has_subscript = lhs.contains('[') && lhs.ends_with(']');
    let is_array_literal = w[eq + 1..].trim_start().starts_with('(');
    if !has_subscript && !is_array_literal {
        return false;
    }
    // validate the name part
    let name = lhs.split('[').next().unwrap_or(lhs);
    !name.is_empty()
        && name
            .chars()
            .enumerate()
            .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
}

/// Apply one assignment word (`name=val`, `name+=val`, `name[sub]=val`, `name=( … )`, etc.).
/// `raw_key` retains any `[subscript]` and a trailing `+` (append); `raw_val` is unexpanded.
pub fn apply_assignment(interp: &mut Interp, raw_key: &str, raw_val: &str) {
    // Decode `+=` (append) and an optional `[subscript]`.
    let (key_body, append) = match raw_key.strip_suffix('+') {
        Some(b) => (b, true),
        None => (raw_key, false),
    };
    let (name, subscript) = match key_body.find('[') {
        Some(br) if key_body.ends_with(']') => {
            (&key_body[..br], Some(&key_body[br + 1..key_body.len() - 1]))
        }
        _ => (key_body, None),
    };
    if interp.readonly.contains(name) {
        interp.expansion_error = Some(crate::interp::ShellExpansionError::assignment(format!(
            "shellsim: {name}: readonly variable\n"
        )));
        return;
    }

    // Array literal value: `( … )`.
    let trimmed = raw_val.trim();
    if trimmed.starts_with('(') && trimmed.ends_with(')') {
        let inner = &trimmed[1..trimmed.len() - 1];
        // Associative literal? Detect `[key]=val` pairs.
        let assoc_existing = matches!(
            interp.arrays.get(name),
            Some(crate::interp::ArrayVal::Assoc(_))
        );
        if !append {
            // fresh array
            if assoc_existing {
                if let Some(crate::interp::ArrayVal::Assoc(m)) = interp.arrays.get_mut(name) {
                    m.clear();
                }
            } else {
                interp.declare_indexed(name);
                if let Some(crate::interp::ArrayVal::Indexed(v)) = interp.arrays.get_mut(name) {
                    v.clear();
                }
            }
        }
        for (subkey, val) in parse_array_elems(interp, inner, assoc_existing) {
            match subkey {
                Some(k) => {
                    if assoc_existing {
                        interp.array_set(name, &k, val);
                    } else {
                        // indexed array with explicit [i]=val
                        interp.array_set(name, &k, val);
                    }
                }
                None => interp.array_append(name, vec![val]),
            }
        }
        return;
    }

    // Subscripted scalar assignment: name[sub]=val (val expanded, no splitting).
    if let Some(sub) = subscript {
        let sub_key = expand_word(interp, sub, false).join(" ");
        let val = expand_word(interp, raw_val, false).join(" ");
        if append {
            let prev = interp.array_get(name, &sub_key).unwrap_or_default();
            interp.array_set(name, &sub_key, format!("{prev}{val}"));
        } else {
            interp.array_set(name, &sub_key, val);
        }
        return;
    }

    // Plain scalar (or scalar-append). If the name is already an array, += appends an element
    // (bash: `arr+=str` is `arr[0]+=str`, but `arr+=(x)` was handled above).
    let val = expand_word(interp, raw_val, false).join(" ");
    if append {
        if interp.is_array(name) {
            // `arr+=val` on an array appends to element 0
            let prev = interp.array_get(name, "0").unwrap_or_default();
            interp.array_set(name, "0", format!("{prev}{val}"));
        } else {
            let prev = interp.get_var(name).unwrap_or_default();
            interp.set_var(name, format!("{prev}{val}"));
        }
    } else {
        interp.set_var(name, val);
    }
}

/// Split an array-literal body into (optional explicit key, expanded value) pairs.
/// Each top-level word undergoes expansion + word-splitting (so `$(cmd)` splits on IFS and
/// `"$x"` stays one element). `[key]=val` forms yield an explicit key.
fn parse_array_elems(
    interp: &mut Interp,
    inner: &str,
    _assoc: bool,
) -> Vec<(Option<String>, String)> {
    let mut out = Vec::new();
    for tok in split_top_level_words(inner) {
        // explicit subscript form: [key]=value
        if let Some(rest) = tok.strip_prefix('[') {
            if let Some(close) = rest.find(']') {
                let key_raw = &rest[..close];
                let after = &rest[close + 1..];
                if let Some(val_raw) = after.strip_prefix('=') {
                    let key = expand_word(interp, key_raw, false).join(" ");
                    let val = expand_word(interp, val_raw, false).join(" ");
                    out.push((Some(key), val));
                    continue;
                }
            }
        }
        // ordinary element: expand with splitting+globbing
        for v in expand_word(interp, &tok, true) {
            out.push((None, v));
        }
    }
    out
}

/// Split a string into shell words at unquoted whitespace, preserving quotes/`$( )`/`${ }`.
fn split_top_level_words(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut started = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\n' => {
                if started {
                    words.push(std::mem::take(&mut cur));
                    started = false;
                }
                i += 1;
            }
            '\'' => {
                cur.push(c);
                started = true;
                i += 1;
                while i < chars.len() {
                    cur.push(chars[i]);
                    i += 1;
                    if chars[i - 1] == '\'' {
                        break;
                    }
                }
            }
            '"' => {
                cur.push(c);
                started = true;
                i += 1;
                while i < chars.len() {
                    let d = chars[i];
                    cur.push(d);
                    i += 1;
                    if d == '\\' && i < chars.len() {
                        cur.push(chars[i]);
                        i += 1;
                        continue;
                    }
                    if d == '"' {
                        break;
                    }
                }
            }
            '\\' => {
                cur.push(c);
                started = true;
                i += 1;
                if i < chars.len() {
                    cur.push(chars[i]);
                    i += 1;
                }
            }
            '$' if chars.get(i + 1) == Some(&'(') => {
                started = true;
                let mut depth = 0;
                cur.push(chars[i]);
                i += 1;
                while i < chars.len() {
                    let d = chars[i];
                    cur.push(d);
                    i += 1;
                    if d == '(' {
                        depth += 1;
                    } else if d == ')' {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                }
            }
            '`' => {
                started = true;
                cur.push(c);
                i += 1;
                while i < chars.len() {
                    cur.push(chars[i]);
                    i += 1;
                    if chars[i - 1] == '`' {
                        break;
                    }
                }
            }
            _ => {
                cur.push(c);
                started = true;
                i += 1;
            }
        }
    }
    if started {
        words.push(cur);
    }
    words
}

fn expand_heredoc(interp: &mut Interp, body: &str) -> String {
    // expand $VAR, ${...}, $(...), `...` in the heredoc body, line by line
    let mut out = String::new();
    for (i, line) in body.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        // reuse double-quote expansion semantics (no splitting/globbing)
        let parts = expand_word(interp, &double_wrap(line), false);
        out.push_str(&parts.join(" "));
    }
    out
}

/// Wrap a line so expand_word treats it as a double-quoted context (expansions, no split).
fn double_wrap(line: &str) -> String {
    // escape existing double quotes and backslashes minimally
    let escaped = line.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn case_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let re = format!("^{}$", glob_to_regex_body(pattern));
    regex::Regex::new(&re)
        .map(|r| r.is_match(text))
        .unwrap_or(pattern == text)
}

fn glob_to_regex_body(pat: &str) -> String {
    let mut re = String::new();
    let mut chars = pat.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' => re.push_str(".*"),
            '?' => re.push('.'),
            '[' => {
                re.push('[');
                while let Some(&n) = chars.peek() {
                    chars.next();
                    re.push(n);
                    if n == ']' {
                        break;
                    }
                }
            }
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '{' | '}' | '\\' => {
                re.push('\\');
                re.push(c);
            }
            _ => re.push(c),
        }
    }
    re
}

#[cfg(test)]
mod array_tests {
    use super::{ShellContinuation, ShellPoll};
    use crate::interp::Interp;
    use crate::shell::Node;

    /// Run a snippet and capture stdout as a String.
    fn run(src: &str) -> String {
        let mut i = Interp::new();
        let ast = crate::shell::parse(src).expect("array test source should parse");
        let mut out = Vec::new();
        let mut err = Vec::new();
        crate::exec::exec(&mut i, &ast, Vec::new(), &mut out, &mut err);
        String::from_utf8_lossy(&out).into_owned()
    }

    #[test]
    fn indexed_basics() {
        assert_eq!(
            run(r#"a=(x y z); echo "${a[1]} ${#a[@]} ${a[@]}""#),
            "y 3 x y z\n"
        );
    }

    #[test]
    fn append_and_count() {
        assert_eq!(
            run(r#"a=(x y z); a+=(w v); echo "${#a[@]} ${a[@]}""#),
            "5 x y z w v\n"
        );
    }

    #[test]
    fn sparse_indices_and_values() {
        assert_eq!(
            run(r#"a=(1 2 3); a[5]=six; echo "${!a[@]}"; echo "${a[@]}""#),
            "0 1 2 5\n1 2 3 six\n"
        );
    }

    #[test]
    fn shell_continuations_stop_at_the_requested_poll_quantum() {
        let mut interp = Interp::new();
        let node = Node::Seq(vec![Node::Empty; 1_000]);
        let mut continuation = ShellContinuation::new(&node);
        assert_eq!(continuation.poll(&mut interp, 1), ShellPoll::Pending);

        let mut polls = 1;
        loop {
            polls += 1;
            if let ShellPoll::Ready(status) = continuation.poll(&mut interp, 17) {
                assert_eq!(status, 0);
                break;
            }
        }
        assert!(polls > 1);
    }

    #[test]
    fn scalar_promotes_to_array() {
        assert_eq!(
            run(r#"x=1; x[2]=3; echo "${x[0]} ${x[2]} ${#x[@]}""#),
            "1 3 2\n"
        );
    }

    #[test]
    fn bare_ref_is_element_zero() {
        assert_eq!(run(r#"a=(p q r); echo "$a ${a}""#), "p p\n");
    }

    #[test]
    fn quoted_at_separate_words() {
        // each element stays a single word even with embedded spaces
        let out = run(r#"a=("one two" three); for x in "${a[@]}"; do echo "[$x]"; done"#);
        assert_eq!(out, "[one two]\n[three]\n");
    }

    #[test]
    fn empty_array_iterates_zero_times() {
        assert_eq!(
            run(r#"a=(); for x in "${a[@]}"; do echo "X$x"; done; echo done"#),
            "done\n"
        );
    }

    #[test]
    fn command_substitution_splits() {
        assert_eq!(
            run(r#"a=($(printf "f1\nf2\nf3\n")); echo "${#a[@]} ${a[1]}""#),
            "3 f2\n"
        );
    }

    #[test]
    fn associative_get_keys_count() {
        // sorted key order is deterministic in our impl
        assert_eq!(
            run(r#"declare -A m; m[foo]=1; m[bar]=2; echo "${m[foo]} ${!m[@]} ${#m[@]}""#),
            "1 bar foo 2\n"
        );
    }

    #[test]
    fn associative_literal_and_arith() {
        let out =
            run(r#"declare -A m=([a]=0 [b]=5); m[a]=$((${m[a]} + 1)); echo "${m[a]} ${m[b]}""#);
        assert_eq!(out, "1 5\n");
    }

    #[test]
    fn slice_and_last() {
        assert_eq!(
            run(r#"a=(a b c d e); echo "${a[@]:1:2}"; echo "${a[@]: -1}""#),
            "b c\ne\n"
        );
    }

    #[test]
    fn unset_element() {
        assert_eq!(
            run(r#"a=(1 2 3 4); unset "a[1]"; echo "${a[@]} ${!a[@]}""#),
            "1 3 4 0 2 3\n"
        );
    }
}
