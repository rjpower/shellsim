//! The shell executor: walks the AST against the interpreter state.

use crate::expand::{expand_word, expand_words};
use crate::interp::Interp;
use crate::shell::{Node, RedirOp, Redirect};
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
    let saved = match interp.process.fds.fork(&mut interp.descriptors) {
        Ok(saved) => saved,
        Err(error) => {
            err.extend_from_slice(
                format!("shellsim: unable to save descriptors: {error:?}\n").as_bytes(),
            );
            return 125;
        }
    };
    let input = match interp.descriptors.open_input(stdin) {
        Ok(description) => description,
        Err(error) => return restore_failed_setup(interp, saved, err, error),
    };
    if let Err(error) = interp.install_new_description(0, input) {
        return restore_failed_setup(interp, saved, err, error);
    }
    let stdout = match interp.descriptors.open_capture() {
        Ok(description) => description,
        Err(error) => return restore_failed_setup(interp, saved, err, error),
    };
    if let Err(error) = interp.install_new_description(1, stdout) {
        return restore_failed_setup(interp, saved, err, error);
    }
    let stderr = match interp.descriptors.open_capture() {
        Ok(description) => description,
        Err(error) => return restore_failed_setup(interp, saved, err, error),
    };
    if let Err(error) = interp.install_new_description(2, stderr) {
        return restore_failed_setup(interp, saved, err, error);
    }

    let status = exec_node(interp, node);
    out.extend_from_slice(&std::mem::take(&mut interp.pending_stdout));
    err.extend_from_slice(&std::mem::take(&mut interp.pending_stderr));
    if let Ok(bytes) = interp.descriptors.drain_capture(stdout) {
        out.extend_from_slice(&bytes);
    }
    if let Ok(bytes) = interp.descriptors.drain_capture(stderr) {
        err.extend_from_slice(&bytes);
    }
    restore_fds(interp, saved);
    status
}

fn restore_failed_setup(
    interp: &mut Interp,
    saved: crate::descriptors::FdTable,
    err: &mut Vec<u8>,
    error: crate::descriptors::DescriptorError,
) -> i32 {
    restore_fds(interp, saved);
    err.extend_from_slice(
        format!("shellsim: unable to install descriptors: {error:?}\n").as_bytes(),
    );
    125
}

fn restore_fds(interp: &mut Interp, saved: crate::descriptors::FdTable) {
    interp.process.fds.close_all(&mut interp.descriptors);
    interp.process.fds = saved;
    interp.refresh_descriptor_snapshot(interp.process.pid);
}

const MAX_SHELL_FRAMES: usize = 4_096;
const SHELL_POLL_QUANTUM: usize = 1;

enum ShellFrame {
    Eval(Node),
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
        iterations: usize,
        body_status: i32,
    },
    WhileAfterCondition {
        cond: Node,
        body: Node,
        until: bool,
        iterations: usize,
        body_status: i32,
    },
    WhileAfterBody {
        cond: Node,
        body: Node,
        until: bool,
        iterations: usize,
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
        iterations: usize,
        body_status: i32,
    },
    CForAfterBody {
        cond: String,
        update: String,
        body: Node,
        iterations: usize,
    },
    RestoreRedirect(RedirectScope),
    FinishFunction {
        positional: Vec<String>,
        variables: Vec<(String, Option<String>)>,
    },
    AwaitChild {
        pid: crate::process::ProcessId,
        reap: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShellPoll {
    Pending,
    Switched,
    Ready(i32),
}

pub(crate) struct ShellContinuation {
    frames: Vec<ShellFrame>,
    status: i32,
    switched: bool,
}

impl ShellContinuation {
    pub(crate) fn new(node: &Node) -> Self {
        Self {
            frames: vec![ShellFrame::Eval(node.clone())],
            status: 0,
            switched: false,
        }
    }

    pub(crate) fn poll(&mut self, interp: &mut Interp, budget: usize) -> ShellPoll {
        self.switched = false;
        for _ in 0..budget.max(1) {
            let Some(frame) = self.frames.pop() else {
                return ShellPoll::Ready(self.status);
            };
            self.step(interp, frame);
            interp.last_status = self.status;
            if self.switched {
                return ShellPoll::Switched;
            }
        }
        if self.frames.is_empty() {
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
                ShellFrame::FinishFunction {
                    positional,
                    variables,
                } => {
                    interp.positional = positional;
                    restore_command_variables(interp, variables);
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

    fn step(&mut self, interp: &mut Interp, frame: ShellFrame) {
        match frame {
            ShellFrame::Eval(node) => self.eval(interp, node),
            ShellFrame::Sequence { nodes, next } => {
                if next > 0 && should_unwind(interp) {
                    if interp.deadline_interrupt.is_some() {
                        self.status = 124;
                    }
                    return;
                }
                if next > 0 && self.status != 0 && interp.opt_errexit && interp.cond_depth == 0 {
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
                    self.push(interp, ShellFrame::Eval(rhs));
                }
            }
            ShellFrame::Negate => {
                interp.cond_depth = interp.cond_depth.saturating_sub(1);
                self.status = i32::from(self.status == 0);
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
                iterations,
                body_status,
            } => {
                if should_unwind(interp) || iterations >= 5_000 {
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
                        iterations,
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
                iterations,
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
                            iterations: iterations + 1,
                        },
                    ) {
                        self.push(interp, ShellFrame::Eval(body));
                    }
                } else {
                    self.status = body_status;
                }
            }
            ShellFrame::WhileAfterBody {
                cond,
                body,
                until,
                iterations,
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
                        ShellFrame::WhileCheck {
                            cond,
                            body,
                            until,
                            iterations,
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
                iterations,
                body_status,
            } => {
                if should_unwind(interp)
                    || iterations >= 5_000
                    || (!cond.is_empty() && crate::expand::eval_arith(interp, &cond) == 0)
                {
                    self.status = body_status;
                } else if self.push(
                    interp,
                    ShellFrame::CForAfterBody {
                        cond,
                        update,
                        body: body.clone(),
                        iterations: iterations + 1,
                    },
                ) {
                    self.push(interp, ShellFrame::Eval(body));
                }
            }
            ShellFrame::CForAfterBody {
                cond,
                update,
                body,
                iterations,
            } => {
                if interp.loop_break > 0 {
                    interp.loop_break -= 1;
                    return;
                }
                if interp.loop_continue > 0 {
                    interp.loop_continue -= 1;
                }
                if !should_unwind(interp) {
                    let _ = crate::expand::eval_arith(interp, &update);
                    self.push(
                        interp,
                        ShellFrame::CForCheck {
                            cond,
                            update,
                            body,
                            iterations,
                            body_status: self.status,
                        },
                    );
                }
            }
            ShellFrame::RestoreRedirect(scope) => end_redirects(interp, scope),
            ShellFrame::FinishFunction {
                positional,
                variables,
            } => {
                interp.positional = positional;
                self.status = interp.returning.take().unwrap_or(self.status);
                restore_command_variables(interp, variables);
            }
            ShellFrame::AwaitChild { pid, reap } => {
                self.status = match interp.processes.get(pid).map(|record| record.status) {
                    Some(crate::process::ProcessStatus::Exited(status)) => status,
                    _ => {
                        write_diagnostic(
                            interp,
                            &format!("shellsim: child {pid} resumed without exit status\n"),
                        );
                        125
                    }
                };
                if reap {
                    interp.processes.reap(pid);
                    let _ = interp.scheduler.reap(pid);
                }
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
            } => self.eval_command(interp, assigns, words, redirects),
            Node::Pipeline(stages) => self.status = exec_pipeline(interp, &stages),
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
            Node::Subshell(inner) => self.spawn_child(interp, *inner, "(subshell)", true),
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
                self.push(
                    interp,
                    ShellFrame::WhileCheck {
                        cond: *cond,
                        body: *body,
                        until,
                        iterations: 0,
                        body_status: 0,
                    },
                );
            }
            Node::For { var, words, body } => {
                let items = expand_words(interp, &words);
                self.status = 0;
                self.push(
                    interp,
                    ShellFrame::ForNext {
                        var,
                        items,
                        body: *body,
                        next: 0,
                        body_status: 0,
                    },
                );
            }
            Node::CFor {
                init,
                cond,
                update,
                body,
            } => {
                let _ = crate::expand::eval_arith(interp, &init);
                self.status = 0;
                self.push(
                    interp,
                    ShellFrame::CForCheck {
                        cond,
                        update,
                        body: *body,
                        iterations: 0,
                        body_status: 0,
                    },
                );
            }
            Node::Case { word, arms } => {
                let subject = expand_word(interp, &word, false).join(" ");
                self.status = 0;
                'arms: for (patterns, body) in arms {
                    for pattern in patterns {
                        let pattern = expand_word(interp, &pattern, false).join(" ");
                        if case_match(&pattern, &subject) {
                            self.push(interp, ShellFrame::Eval(body));
                            break 'arms;
                        }
                    }
                }
            }
            Node::FuncDef { name, body } => {
                interp.funcs.insert(name, *body);
                self.status = 0;
            }
            Node::Arithmetic(expression) => {
                self.status = i32::from(crate::expand::eval_arith(interp, &expression) == 0);
            }
        }
    }

    fn eval_command(
        &mut self,
        interp: &mut Interp,
        assigns: Vec<(String, String)>,
        words: Vec<String>,
        redirects: Vec<Redirect>,
    ) {
        if !redirects.is_empty() {
            match begin_redirects(interp, &redirects) {
                Ok(scope) => {
                    if self.push(interp, ShellFrame::RestoreRedirect(scope)) {
                        self.push(
                            interp,
                            ShellFrame::Eval(Node::Command {
                                assigns,
                                words,
                                redirects: Vec::new(),
                            }),
                        );
                    }
                }
                Err(error) => {
                    write_diagnostic(interp, &format!("shellsim: redirection: {error}\n"));
                    self.status = 1;
                }
            }
            return;
        }

        let argv = expand_argv(interp, &words);
        if argv.is_empty() {
            for (key, value) in &assigns {
                apply_assignment(interp, key, value);
            }
            self.status = 0;
            return;
        }
        let variables = install_command_variables(interp, &assigns);
        interp.cmd_trace.push(argv[0].clone());
        if let Some(body) = interp.funcs.get(&argv[0]).cloned() {
            let positional = std::mem::replace(&mut interp.positional, argv[1..].to_vec());
            if !self.ensure_capacity(interp, 2) {
                interp.positional = positional;
                restore_command_variables(interp, variables);
                return;
            }
            self.frames.push(ShellFrame::FinishFunction {
                positional,
                variables,
            });
            self.frames.push(ShellFrame::Eval(body));
        } else {
            self.status = run_external_command(interp, &argv);
            restore_command_variables(interp, variables);
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
                interp.process.shell_continuation = Some(ShellContinuation::new(&node));
                self.switched = true;
                debug_assert_ne!(parent_pid, interp.process.pid);
            }
            Err(error) => {
                write_diagnostic(interp, &format!("shellsim: {error}\n"));
                self.status = 125;
            }
        }
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

fn exec_node(interp: &mut Interp, node: &Node) -> i32 {
    if interp.process.shell_continuation.is_some() {
        write_diagnostic(
            interp,
            "shellsim: attempted to replace an active shell continuation\n",
        );
        return 125;
    }
    let target_pid = interp.process.pid;
    interp.process.shell_continuation = Some(ShellContinuation::new(node));
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
    loop {
        let owner_pid = interp.process.pid;
        let mut continuation = interp
            .process
            .shell_continuation
            .take()
            .expect("active process continuation was installed above");
        match continuation.poll(interp, SHELL_POLL_QUANTUM) {
            ShellPoll::Pending => {
                interp
                    .process
                    .set_continuation(owner_pid, Some(continuation))
                    .expect("polled process state must remain present");
                if interp.scheduler.has_runnable() {
                    interp
                        .scheduler
                        .yield_current()
                        .expect("polled process must be the running scheduler task");
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
            }
            ShellPoll::Switched => {
                interp
                    .process
                    .set_continuation(owner_pid, Some(continuation))
                    .expect("suspended parent process state must remain present");
            }
            ShellPoll::Ready(status) if owner_pid == target_pid => return status,
            ShellPoll::Ready(status) => interp.finish_child(owner_pid, status),
        }
    }
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
        .set_continuation(pid, Some(ShellContinuation::new(node)))
        .expect("new background child state must exist");
    let Some(id) = interp.new_job(pid, command) else {
        interp.cancel_unstarted_child(pid);
        write_diagnostic(interp, "shellsim: job table limit exceeded\n");
        return 125;
    };
    debug_assert!(interp.jobs.iter().any(|job| job.id == id && !job.done));
    interp.set_var("!", pid.to_string());
    0
}

fn describe(node: &Node) -> String {
    match node {
        Node::Command { words, .. } => words.join(" "),
        _ => "<job>".to_string(),
    }
}

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

/// Apply redirections from left to right. `dup` retains the open description selected at that
/// point, so `2>&1 >file` and `>file 2>&1` have distinct, Bash-compatible destinations.
fn apply_redirects(interp: &mut Interp, redirects: &[Redirect]) -> Result<(), String> {
    for redirect in redirects {
        match redirect.op {
            RedirOp::Read => {
                let path = redirect_path(interp, &redirect.target)?;
                if let Some(source) = device_fd(&path) {
                    interp
                        .process
                        .fds
                        .duplicate(source, redirect.fd, &mut interp.descriptors)
                        .map_err(|error| format!("{path}: {error:?}"))?;
                } else if path == "/dev/null" {
                    let description = interp
                        .descriptors
                        .open_null()
                        .map_err(|error| format!("{path}: {error:?}"))?;
                    interp
                        .install_new_description(redirect.fd, description)
                        .map_err(|error| format!("{path}: {error:?}"))?;
                } else {
                    interp
                        .fs_metadata("/", &path, true)
                        .map_err(|error| error.to_string())?;
                    let description = interp
                        .descriptors
                        .open_file(path.clone(), 0, true, false, false)
                        .map_err(|error| format!("{path}: {error:?}"))?;
                    interp
                        .install_new_description(redirect.fd, description)
                        .map_err(|error| format!("{path}: {error:?}"))?;
                }
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
                    interp
                        .process
                        .fds
                        .duplicate(source, redirect.fd, &mut interp.descriptors)
                        .map_err(|error| format!("{path}: {error:?}"))?;
                    continue;
                }
                if path == "/dev/null" {
                    let description = interp
                        .descriptors
                        .open_null()
                        .map_err(|error| format!("{path}: {error:?}"))?;
                    interp
                        .install_new_description(redirect.fd, description)
                        .map_err(|error| format!("{path}: {error:?}"))?;
                    continue;
                }
                let created_by_open = !interp.vfs.lexists("/", &path);
                interp.sync_vfs_time();
                let cursor = if redirect.op == RedirOp::Append {
                    interp
                        .vfs
                        .append("/", &path, &[], 0o644)
                        .map_err(|error| error.to_string())?;
                    interp
                        .vfs
                        .file_len("/", &path)
                        .map_err(|error| error.to_string())? as u64
                } else {
                    interp
                        .vfs
                        .write("/", &path, &[], 0o644)
                        .map_err(|error| error.to_string())?;
                    0
                };
                let description = interp
                    .descriptors
                    .open_file(path.clone(), cursor, false, true, created_by_open)
                    .map_err(|error| format!("{path}: {error:?}"))?;
                interp
                    .install_new_description(redirect.fd, description)
                    .map_err(|error| format!("{path}: {error:?}"))?;
            }
            RedirOp::DupOut => {
                let source = redirect
                    .target
                    .trim_start_matches('&')
                    .parse::<i32>()
                    .map_err(|_| format!("bad file descriptor: {}", redirect.target))?;
                interp
                    .process
                    .fds
                    .duplicate(source, redirect.fd, &mut interp.descriptors)
                    .map_err(|error| format!("{}: {error:?}", redirect.target))?;
            }
            RedirOp::Close => {
                interp
                    .process
                    .fds
                    .close(redirect.fd, &mut interp.descriptors)
                    .map_err(|error| format!("{}: {error:?}", redirect.fd))?;
            }
        }
        interp.refresh_descriptor_snapshot(interp.process.pid);
    }
    Ok(())
}

fn redirect_path(interp: &mut Interp, word: &str) -> Result<String, String> {
    let fields = expand_word(interp, word, true);
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

fn read_all_fd(interp: &mut Interp, fd: i32) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    loop {
        match interp.read_fd(fd, 64 * 1024)? {
            IoPoll::Ready(bytes) if bytes.is_empty() => return Ok(output),
            IoPoll::Ready(bytes) => {
                if output.len().saturating_add(bytes.len()) > crate::descriptors::MAX_CAPTURE_BYTES
                {
                    return Err("input exceeds descriptor capture limit".to_string());
                }
                output.extend_from_slice(&bytes);
            }
            IoPoll::Blocked(_) => return Err("descriptor read would block".to_string()),
        }
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

fn run_external_command(interp: &mut Interp, argv: &[String]) -> i32 {
    let mut local_out = Vec::new();
    let mut local_err = Vec::new();
    let cmd_stdin = if argv[0] == "read" {
        Vec::new()
    } else {
        match read_all_fd(interp, 0) {
            Ok(bytes) => bytes,
            Err(error) => {
                local_err.extend_from_slice(format!("shellsim: {}: {error}\n", argv[0]).as_bytes());
                Vec::new()
            }
        }
    };
    let mut status = crate::commands::run(interp, argv, cmd_stdin, &mut local_out, &mut local_err);

    let mut output_failed = false;
    if let Err(error) = write_all_fd(interp, 1, &local_out) {
        write_diagnostic(interp, &format!("shellsim: {}: {error}\n", argv[0]));
        output_failed = true;
    }
    if write_all_fd(interp, 2, &local_err).is_err() {
        output_failed = true;
    }
    if output_failed {
        status = 1;
    }
    status
}

/// Expand a command's argv, but keep array-assignment literal words verbatim so the builtin
/// (`declare`/`local`/`typeset`/`readonly`) can parse them itself.
fn expand_argv(interp: &mut Interp, words: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for w in words {
        if is_array_assign_word(w) {
            out.push(w.clone());
        } else {
            out.extend(expand_word(interp, w, true));
        }
    }
    out
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

fn exec_pipeline(interp: &mut Interp, stages: &[Node]) -> i32 {
    let mut input = match read_all_fd(interp, 0) {
        Ok(input) => input,
        Err(error) => {
            write_diagnostic(interp, &format!("shellsim: pipeline: {error}\n"));
            return 1;
        }
    };
    let mut last_status = 0;
    let mut statuses = Vec::new();
    for (idx, stage) in stages.iter().enumerate() {
        if interp.resources.is_stopped() {
            break;
        }
        let is_last = idx == stages.len() - 1;
        let command = describe(stage);
        let Some((status, stage_out)) = exec_child_capture_stdout(
            interp,
            stage,
            std::mem::take(&mut input),
            ChildExecution {
                command: &command,
                new_shell: false,
                retain: false,
            },
        ) else {
            write_diagnostic(interp, "shellsim: unable to create pipeline process\n");
            return 125;
        };
        last_status = status;
        statuses.push(last_status);
        if is_last {
            if let Err(error) = write_all_fd(interp, 1, &stage_out) {
                write_diagnostic(interp, &format!("shellsim: pipeline: {error}\n"));
                return 1;
            }
        } else {
            input = stage_out;
        }
    }
    if interp.opt_pipefail {
        statuses
            .into_iter()
            .rev()
            .find(|s| *s != 0)
            .unwrap_or(last_status)
    } else {
        last_status
    }
}

/// Lifecycle policy for one synchronous logical child execution.
pub(crate) struct ChildExecution<'a> {
    pub command: &'a str,
    pub new_shell: bool,
    pub retain: bool,
}

/// Execute one logical child against shared machine state and restore its parent shell state.
pub(crate) fn exec_child(
    interp: &mut Interp,
    node: &Node,
    child: ChildExecution<'_>,
) -> Option<(i32, crate::process::ProcessId)> {
    let pid = match interp.start_child(child.command, child.new_shell) {
        Ok(child) => child,
        Err(error) => {
            write_diagnostic(interp, &format!("shellsim: {error}\n"));
            return None;
        }
    };
    let status = exec_node(interp, node);
    interp.finish_child(pid, status);
    if !child.retain {
        interp.processes.reap(pid);
        let _ = interp.scheduler.reap(pid);
    }
    Some((status, pid))
}

pub(crate) fn exec_child_capture_stdout(
    interp: &mut Interp,
    node: &Node,
    stdin: Vec<u8>,
    child: ChildExecution<'_>,
) -> Option<(i32, Vec<u8>)> {
    let saved = interp.process.fds.fork(&mut interp.descriptors).ok()?;
    let input = interp.descriptors.open_input(stdin).ok()?;
    if interp.install_new_description(0, input).is_err() {
        restore_fds(interp, saved);
        return None;
    }
    let output = match interp.descriptors.open_capture() {
        Ok(output) => output,
        Err(_) => {
            restore_fds(interp, saved);
            return None;
        }
    };
    if interp.install_new_description(1, output).is_err() {
        restore_fds(interp, saved);
        return None;
    }
    let result = exec_child(interp, node, child);
    let bytes = interp
        .descriptors
        .capture(output)
        .map_or_else(|_| Vec::new(), <[u8]>::to_vec);
    restore_fds(interp, saved);
    result.map(|(status, _)| (status, bytes))
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
