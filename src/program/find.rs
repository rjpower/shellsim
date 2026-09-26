//! Resumable `find` image: walks the VFS in bounded quanta and spawns one real virtual child
//! process per `-exec`/`-execdir` invocation, instead of running it through synchronous nested
//! command dispatch the way [`crate::commands::find`]'s legacy body does.
//!
//! Parsing (the [`find::Parser`]/[`find::Expr`] types) and metadata-only predicate evaluation
//! ([`find::evaluate_metadata_leaf`]) are shared with `commands::find` so the native image and the
//! synchronous body kept for nested callers (such as `xargs` invoking `find` recursively) never
//! diverge on what `find` accepts. Only `-exec`'s side effects differ: this image never touches
//! `Interp` and drives everything through [`System`].
//!
//! ## Walk strategy
//! The walk keeps an explicit directory stack (mirroring [`crate::interp::Interp::fs_walk`]'s
//! pending-stack shape) instead of collecting the whole tree up front, so at most one pending
//! directory listing is held per stack frame at a time. Each visited entry charges one CPU unit,
//! and the image yields `ShellPoll::Pending` after a bounded number of entries per poll so other
//! processes stay scheduled during a large walk.
//!
//! ## Expression evaluation and suspension
//! A candidate's boolean expression is evaluated with a small explicit stack machine (`EvalState`)
//! instead of native recursion, because `-exec ... ;` must suspend the whole shell process while
//! its child runs: the machine walks down `Expr::And`/`Or`/`Not` pushing `Frame`s, evaluates leaves
//! synchronously, and short-circuits on the way back up exactly like the legacy recursive
//! evaluator. `-exec ... +` never suspends by itself (its predicate is always true; GNU find
//! defers the actual runs), except when appending a match would exceed `EXEC_BATCH_BYTES`, at
//! which point the accumulated batch is flushed as its own spawned child before appending.
//!
//! ## Output ordering
//! `-print`/`-print0` bytes are buffered and drained through [`super::poll_write`] at the top of
//! every `poll` iteration, before any subsequent spawn, so find's own output for one entry always
//! reaches fd 1 before a later `-exec` child's output can.

use std::collections::VecDeque;

use crate::commands::find::{self, Expr};
use crate::exec::ShellPoll;
use crate::process::ProcessId;
use crate::scheduler::WaitReason;
use crate::syscalls::{SpawnSpec, System};
use crate::vfs::resolve_against;

use super::poll_write;

/// Cap on accumulated `-exec ... +` argument bytes (each argument plus its NUL terminator, as
/// argv is actually laid out) before a batch is flushed early. GNU find packs matches into each
/// invocation up to a fraction of the host's `ARG_MAX`; 128 KiB is the same order of magnitude
/// and reproduces its "pack as many matches into as few children as possible" behavior. The
/// process table (`src/process.rs`) only truncates its diagnostic label past its own 4 KiB
/// ceiling rather than rejecting long commands, so a batch this size spawns without a separate
/// process-table limit coming into play first.
const EXEC_BATCH_BYTES: usize = 128 * 1024;

/// Bound on directory entries visited per `poll` call, so a large tree yields control instead of
/// running to completion in one scheduler turn.
const ENTRIES_PER_QUANTUM: usize = 512;

#[derive(Clone)]
pub(crate) struct FindProcess {
    parsed: Result<find::Parsed, String>,
    started: bool,
    status: i32,
    any_batch_failure: bool,
    stdout: Vec<u8>,
    stdout_offset: usize,
    stderr: Vec<u8>,
    stderr_offset: usize,
    done: bool,
    starts: VecDeque<StartPoint>,
    active: Option<ActiveWalk>,
    exec_batches: Vec<Vec<String>>,
    eval: Option<EvalState>,
    waiting: Option<Waiting>,
}

#[derive(Clone)]
struct StartPoint {
    token: String,
    absolute: String,
    /// Working directory at the time `find` started, used to render relative display paths.
    cwd: String,
}

#[derive(Clone)]
enum StackItem {
    /// Visit this path; if it is a directory, list and push its children before yielding it (or,
    /// in post-order/`-delete` mode, after).
    Enter(String, usize),
    /// A directory whose children have already been pushed and (in post-order mode) processed;
    /// yield it now.
    Leave(String, usize),
}

#[derive(Clone)]
struct ActiveWalk {
    start: StartPoint,
    post_order: bool,
    stack: Vec<StackItem>,
}

#[derive(Clone)]
struct EvalState {
    path: String,
    display: String,
    stack: Vec<Frame>,
    node: Option<Expr>,
    value: Option<bool>,
}

#[derive(Clone)]
enum Frame {
    Not,
    And(Expr),
    Or(Expr),
}

#[derive(Clone, Copy)]
enum Waiting {
    ExecOne { pid: ProcessId },
    Flush { pid: ProcessId },
}

/// One bounded unit of forward progress made by [`FindProcess::step`].
enum Step {
    /// Did some work; the caller should keep looping (subject to the quantum bound).
    Progress,
    /// Spawned a child and must block on it; the caller must return `ShellPoll::Switched`.
    Spawned,
}

impl FindProcess {
    pub(super) fn new(args: &[String]) -> Self {
        Self {
            parsed: find::Parser::parse(args),
            started: false,
            status: 0,
            any_batch_failure: false,
            stdout: Vec::new(),
            stdout_offset: 0,
            stderr: Vec::new(),
            stderr_offset: 0,
            done: false,
            starts: VecDeque::new(),
            active: None,
            exec_batches: Vec::new(),
            eval: None,
            waiting: None,
        }
    }

    pub(super) fn poll(&mut self, system: &mut impl System) -> ShellPoll {
        let mut quantum = 0usize;
        loop {
            if self.stderr_offset < self.stderr.len() {
                match poll_write(system, 2, &self.stderr, &mut self.stderr_offset, 0) {
                    ShellPoll::Ready(_) => {
                        self.stderr.clear();
                        self.stderr_offset = 0;
                    }
                    other => return other,
                }
                continue;
            }
            if self.stdout_offset < self.stdout.len() {
                match poll_write(system, 1, &self.stdout, &mut self.stdout_offset, 0) {
                    ShellPoll::Ready(_) => {
                        self.stdout.clear();
                        self.stdout_offset = 0;
                    }
                    other => return other,
                }
                continue;
            }
            if let Some(waiting) = self.waiting {
                match self.resolve_waiting(system, waiting) {
                    Ok(true) => continue,
                    Ok(false) => return ShellPoll::Blocked(WaitReason::Child(pid_of(waiting))),
                    Err(()) => continue,
                }
            }
            if self.done {
                return ShellPoll::Ready(self.status);
            }
            if !self.started {
                self.start(system);
                continue;
            }
            match self.step(system) {
                Step::Progress => {
                    quantum += 1;
                    if quantum >= ENTRIES_PER_QUANTUM {
                        return ShellPoll::Pending;
                    }
                }
                Step::Spawned => return ShellPoll::Switched,
            }
        }
    }

    /// Resolve a pending child wait. `Ok(true)` means the caller should keep looping (the wait is
    /// finished and state has been resumed); `Ok(false)` means still blocked; `Err(())` means a
    /// fatal error was recorded and the caller should re-check `self.done`.
    fn resolve_waiting(&mut self, system: &mut impl System, waiting: Waiting) -> Result<bool, ()> {
        let pid = pid_of(waiting);
        match system.child_status(pid) {
            Ok(None) => Ok(false),
            Ok(Some(status)) => {
                if let Err(error) = system.reap_child(pid) {
                    self.fail_fatal(125, format!("find: cannot reap child: {error}\n"));
                    self.waiting = None;
                    return Err(());
                }
                self.waiting = None;
                match waiting {
                    Waiting::ExecOne { .. } => self.resume_eval(status == 0),
                    Waiting::Flush { .. } => {
                        if status != 0 {
                            self.any_batch_failure = true;
                        }
                        self.resume_eval(true);
                    }
                }
                Ok(true)
            }
            Err(error) => {
                self.fail_fatal(125, format!("find: cannot wait for child: {error}\n"));
                self.waiting = None;
                Err(())
            }
        }
    }

    fn start(&mut self, system: &mut impl System) {
        self.started = true;
        let parsed = match &self.parsed {
            Err(message) => {
                self.stderr = format!("find: {message}\n").into_bytes();
                self.status = 2;
                self.done = true;
                return;
            }
            Ok(parsed) => parsed,
        };
        let cwd = system.cwd().to_string();
        self.exec_batches = vec![Vec::new(); parsed.exec_batches];
        for token in &parsed.paths {
            let absolute = resolve_against(&cwd, token);
            self.starts.push_back(StartPoint {
                token: token.clone(),
                absolute,
                cwd: cwd.clone(),
            });
        }
    }

    /// Perform one bounded unit of work: continue evaluating the current candidate's expression,
    /// or advance the walk to the next candidate, or finish once every start point and batch has
    /// been drained.
    fn step(&mut self, system: &mut impl System) -> Step {
        if self.eval.is_some() {
            return self.advance_eval(system);
        }
        match self.next_candidate(system) {
            Ok(Some((path, display))) => {
                let parsed = self.parsed();
                let root = if parsed.explicit_action {
                    parsed.expression.clone()
                } else {
                    // GNU find prints a match by default only when the expression has no other
                    // action; folding that into an implicit trailing `-print` reuses the same
                    // short-circuiting machinery as an explicit one.
                    Expr::And(
                        Box::new(parsed.expression.clone()),
                        Box::new(Expr::Print(false)),
                    )
                };
                self.eval = Some(EvalState {
                    path,
                    display,
                    stack: Vec::new(),
                    node: Some(root),
                    value: None,
                });
                Step::Progress
            }
            Ok(None) => {
                if self.active.is_none() && self.starts.is_empty() {
                    self.flush_remaining_batches(system)
                } else {
                    Step::Progress
                }
            }
            Err(Diagnostic::Operational(message)) => {
                self.push_stderr(format!("find: {message}\n"));
                self.status = self.status.max(1);
                Step::Progress
            }
            Err(Diagnostic::Resource) => {
                self.done = true;
                self.status = system.stop_status();
                Step::Progress
            }
        }
    }

    /// Advance the directory walk by one entry and return the next candidate's absolute path and
    /// display string, applying `-mindepth`/`-maxdepth` filtering. Directory listings are pruned
    /// once `depth == max_depth`, so entries deeper than the requested frontier are never listed
    /// or charged for.
    fn next_candidate(
        &mut self,
        system: &mut impl System,
    ) -> Result<Option<(String, String)>, Diagnostic> {
        loop {
            if self.active.is_none() {
                let Some(start) = self.starts.pop_front() else {
                    return Ok(None);
                };
                if let Err(error) = system.metadata("/", &start.absolute, false) {
                    self.status = self.status.max(1);
                    let reason = start_path_error_reason(&error);
                    self.push_stderr(format!("find: '{}': {reason}\n", start.token));
                    continue;
                }
                let post_order = self.parsed().delete;
                self.active = Some(ActiveWalk {
                    stack: vec![StackItem::Enter(start.absolute.clone(), 0)],
                    post_order,
                    start,
                });
            }
            let post_order = self
                .active
                .as_ref()
                .expect("active walk present")
                .post_order;
            let Some(item) = self
                .active
                .as_mut()
                .expect("active walk present")
                .stack
                .pop()
            else {
                self.active = None;
                continue;
            };
            if !system.charge_cpu(1) {
                return Err(Diagnostic::Resource);
            }
            let (abs, depth) = match item {
                StackItem::Leave(abs, depth) => (abs, depth),
                StackItem::Enter(abs, depth) => {
                    let info = system
                        .metadata("/", &abs, false)
                        .map_err(|error| Diagnostic::Operational(format!("{abs}: {error}")))?;
                    let is_dir = matches!(info.kind, crate::syscalls::FileKind::Directory);
                    if is_dir {
                        let max_depth = self.parsed().max_depth;
                        let within_depth = max_depth.is_none_or(|maximum| depth < maximum);
                        if within_depth {
                            let mut entries = system.list_dir("/", &abs).map_err(|error| {
                                Diagnostic::Operational(format!("{abs}: {error}"))
                            })?;
                            entries.sort();
                            let walk = self.active.as_mut().expect("active walk present");
                            if post_order {
                                walk.stack.push(StackItem::Leave(abs.clone(), depth));
                            }
                            for entry in entries.into_iter().rev() {
                                if !system.charge_cpu(1) {
                                    return Err(Diagnostic::Resource);
                                }
                                let child = if abs == "/" {
                                    format!("/{entry}")
                                } else {
                                    format!("{abs}/{entry}")
                                };
                                walk.stack
                                    .push(StackItem::Enter(child, depth.saturating_add(1)));
                            }
                        } else if post_order {
                            self.active
                                .as_mut()
                                .expect("active walk present")
                                .stack
                                .push(StackItem::Leave(abs.clone(), depth));
                        }
                        if post_order {
                            continue;
                        }
                    }
                    (abs, depth)
                }
            };
            // `depth` already counts levels below this start point (the start itself is 0), so
            // it is the relative depth `-mindepth`/`-maxdepth` compare against directly.
            let relative_depth = depth;
            let (min_depth, max_depth) = {
                let parsed = self.parsed();
                (parsed.min_depth, parsed.max_depth)
            };
            if relative_depth < min_depth
                || max_depth.is_some_and(|maximum| relative_depth > maximum)
            {
                continue;
            }
            let walk = self.active.as_ref().expect("active walk present");
            let display = find::display_path(
                &walk.start.cwd,
                &walk.start.token,
                &walk.start.absolute,
                &abs,
            );
            return Ok(Some((abs, display)));
        }
    }

    fn parsed(&self) -> &find::Parsed {
        self.parsed.as_ref().expect("parsed find expression")
    }

    /// Whether buffered `find`-owned output is still waiting to reach fd 1/2. Checked before
    /// every spawn so a match's own `-print` always reaches the descriptor before a later
    /// `-exec` child's output can, even though both are generated within the same `advance_eval`
    /// call: when this is true the caller must return to `poll`'s top-level drain loop and retry
    /// the same node afterward instead of spawning immediately.
    fn needs_output_drain(&self) -> bool {
        !self.stdout.is_empty() || !self.stderr.is_empty()
    }

    /// Continue the current candidate's boolean expression: evaluate leaves synchronously and
    /// unwind `Not`/`And`/`Or` short-circuits with an explicit stack, suspending (returning
    /// `Step::Spawned`) exactly when an `-exec` leaf or a `+`-batch flush needs a child process.
    fn advance_eval(&mut self, system: &mut impl System) -> Step {
        let Some(mut eval) = self.eval.take() else {
            return Step::Progress;
        };
        loop {
            if let Some(node) = eval.node.take() {
                if !system.charge_cpu(1) {
                    self.done = true;
                    self.status = system.stop_status();
                    return Step::Progress;
                }
                match node {
                    Expr::And(left, right) => {
                        eval.stack.push(Frame::And(*right));
                        eval.node = Some(*left);
                    }
                    Expr::Or(left, right) => {
                        eval.stack.push(Frame::Or(*right));
                        eval.node = Some(*left);
                    }
                    Expr::Not(inner) => {
                        eval.stack.push(Frame::Not);
                        eval.node = Some(*inner);
                    }
                    Expr::True => eval.value = Some(true),
                    Expr::Print(nul) => {
                        self.stdout.extend_from_slice(eval.display.as_bytes());
                        self.stdout.push(if nul { 0 } else { b'\n' });
                        eval.value = Some(true);
                    }
                    Expr::Exec {
                        argv,
                        batch: Some(id),
                        dir,
                    } => {
                        debug_assert!(!dir, "-execdir ... + is rejected while parsing");
                        if self.needs_output_drain() {
                            eval.node = Some(Expr::Exec {
                                argv,
                                batch: Some(id),
                                dir,
                            });
                            self.eval = Some(eval);
                            return Step::Progress;
                        }
                        if batch_would_overflow(&self.exec_batches[id], &eval.display) {
                            match self.spawn_flush(system, id) {
                                Ok(pid) => {
                                    eval.node = Some(Expr::Exec {
                                        argv,
                                        batch: Some(id),
                                        dir,
                                    });
                                    self.waiting = Some(Waiting::Flush { pid });
                                    self.eval = Some(eval);
                                    return Step::Spawned;
                                }
                                Err(message) => {
                                    self.fail_fatal(125, message);
                                    return Step::Progress;
                                }
                            }
                        }
                        self.exec_batches[id].push(eval.display.clone());
                        eval.value = Some(true);
                    }
                    Expr::Exec {
                        argv,
                        batch: None,
                        dir,
                    } => {
                        if self.needs_output_drain() {
                            eval.node = Some(Expr::Exec {
                                argv,
                                batch: None,
                                dir,
                            });
                            self.eval = Some(eval);
                            return Step::Progress;
                        }
                        let (cwd, substituted) = exec_target(&eval.path, &eval.display, dir);
                        let command: Vec<String> = argv
                            .iter()
                            .map(|argument| argument.replace("{}", &substituted))
                            .collect();
                        match system.spawn_argv(SpawnSpec {
                            argv: command,
                            cwd,
                            ..Default::default()
                        }) {
                            Ok(pid) => {
                                self.waiting = Some(Waiting::ExecOne { pid });
                                self.eval = Some(eval);
                                return Step::Spawned;
                            }
                            Err(error) => {
                                self.fail_fatal(
                                    125,
                                    format!("find: cannot spawn child: {error}\n"),
                                );
                                return Step::Progress;
                            }
                        }
                    }
                    leaf => {
                        match find::evaluate_metadata_leaf(system, &leaf, &eval.path, &eval.display)
                        {
                            Ok(value) => eval.value = Some(value),
                            Err(message) => {
                                self.push_stderr(format!("find: {message}\n"));
                                self.status = self.status.max(1);
                                eval.value = Some(false);
                            }
                        }
                    }
                }
                continue;
            }
            let value = eval.value.take().expect("eval has a node or a value");
            match eval.stack.pop() {
                None => return Step::Progress,
                Some(Frame::Not) => eval.value = Some(!value),
                Some(Frame::And(right)) => {
                    if value {
                        eval.node = Some(right);
                    } else {
                        eval.value = Some(false);
                    }
                }
                Some(Frame::Or(right)) => {
                    if value {
                        eval.value = Some(true);
                    } else {
                        eval.node = Some(right);
                    }
                }
            }
        }
    }

    /// Resume a suspended evaluation with the boolean result of the child that just exited.
    fn resume_eval(&mut self, value: bool) {
        if let Some(eval) = self.eval.as_mut() {
            eval.value = Some(value);
        }
    }

    /// Flush an accumulated `-exec ... +` batch as one child, clearing it so new matches start a
    /// fresh batch. Returns the spawned child's pid.
    fn spawn_flush(&mut self, system: &mut impl System, id: usize) -> Result<ProcessId, String> {
        let paths = std::mem::take(&mut self.exec_batches[id]);
        let Expr::Exec { argv, .. } = find_exec_expr(self.parsed(), id) else {
            unreachable!("batch id refers to an '-exec ... +' expression");
        };
        let mut command = argv.clone();
        command.extend(paths);
        system
            .spawn_argv(SpawnSpec {
                argv: command,
                ..Default::default()
            })
            .map_err(|error| format!("find: cannot spawn child: {error}\n"))
    }

    /// After the walk is exhausted, spawn one final child for every non-empty `+` batch that
    /// never reached the byte cap. Called repeatedly (once per resumed child) until every batch
    /// is empty.
    fn flush_remaining_batches(&mut self, system: &mut impl System) -> Step {
        for id in 0..self.exec_batches.len() {
            if self.exec_batches[id].is_empty() {
                continue;
            }
            return match self.spawn_flush(system, id) {
                Ok(pid) => {
                    self.waiting = Some(Waiting::Flush { pid });
                    Step::Spawned
                }
                Err(message) => {
                    self.fail_fatal(125, message);
                    Step::Progress
                }
            };
        }
        self.done = true;
        if self.any_batch_failure {
            self.status = self.status.max(1);
        }
        Step::Progress
    }

    fn push_stderr(&mut self, message: String) {
        self.stderr.extend_from_slice(message.as_bytes());
    }

    fn fail_fatal(&mut self, status: i32, message: String) {
        self.push_stderr(message);
        self.status = status;
        self.done = true;
    }
}

/// Locate the argv template for a `+`-batch by its id, by walking the parsed expression once.
/// Batch ids are assigned sequentially while parsing, so this always finds exactly one match.
fn find_exec_expr(parsed: &find::Parsed, id: usize) -> &Expr {
    fn search(expr: &Expr, id: usize) -> Option<&Expr> {
        match expr {
            Expr::Exec {
                batch: Some(found), ..
            } if *found == id => Some(expr),
            Expr::Not(inner) => search(inner, id),
            Expr::And(left, right) | Expr::Or(left, right) => {
                search(left, id).or_else(|| search(right, id))
            }
            _ => None,
        }
    }
    search(&parsed.expression, id).expect("exec batch id present in parsed expression")
}

/// Whether appending `display` to `batch` would exceed [`EXEC_BATCH_BYTES`] of argument bytes.
fn batch_would_overflow(batch: &[String], display: &str) -> bool {
    if batch.is_empty() {
        return false;
    }
    let current: usize = batch
        .iter()
        .map(|entry| entry.len().saturating_add(1))
        .sum();
    current.saturating_add(display.len()).saturating_add(1) > EXEC_BATCH_BYTES
}

/// Compute the child's `cwd` override and the `{}` substitution text for one `-exec`/`-execdir`
/// invocation. `-execdir` runs with the match's directory as `cwd` and refers to the match as
/// `./basename`; plain `-exec` keeps the caller's `cwd` and substitutes the display path.
fn exec_target(path: &str, display: &str, dir: bool) -> (Option<String>, String) {
    if dir {
        let (parent, base) = find::split_dir(path);
        (Some(parent), format!("./{base}"))
    } else {
        (None, display.to_string())
    }
}

/// Render a starting-point lookup failure the way GNU find does: the bare reason, without the
/// path repeated a second time. Every `VfsError`/`SyscallError` variant used here formats as
/// `"<reason>: <path>"`, so stripping the exact `absolute` path we looked up as a suffix recovers
/// the reason alone; an error that does not end that way (unexpected, but not fatal) is reported
/// unmodified rather than mangled.
fn start_path_error_reason(error: &crate::syscalls::SyscallError) -> String {
    match error {
        crate::syscalls::SyscallError::File(error) => error.reason().to_string(),
        other => other.to_string(),
    }
}

fn pid_of(waiting: Waiting) -> ProcessId {
    match waiting {
        Waiting::ExecOne { pid } | Waiting::Flush { pid } => pid,
    }
}

enum Diagnostic {
    Operational(String),
    Resource,
}
