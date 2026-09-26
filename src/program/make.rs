//! Resumable native `make` image.
//!
//! `make` runs as an ordinary process image against `System`, exactly like `xargs` or `timeout`;
//! it never touches `Interp` or host resources. Parsing and variable expansion live in
//! `crate::commands::makecmd`; this module owns graph traversal and process control.
//!
//! Rebuild decisions are made lazily, one target at a time, as its prerequisites finish: a
//! target's modification time is read with `System::metadata` only after all of its
//! prerequisites (including any recipes that rebuilt them) have completed, so a recipe that
//! writes a prerequisite file is visible to the target that depends on it. This rules out
//! precomputing the whole recipe list up front, which would race a still-pending prerequisite
//! recipe against the mtime check of whatever depends on it.
//!
//! Traversal is depth-first and modeled as an explicit frame stack rather than Rust recursion,
//! because a target's recipe can block the process (a child that sleeps or reads a pipe) across
//! many scheduler quanta; the frame stack is the state that survives a blocked poll.
//!
//! Each recipe line is one `sh -c LINE` child (shells started by argv run their `-c` source in
//! place, so this does not fork a nested interpreter chain). `-j` is accepted for compatibility
//! but recipes always run one at a time: parallel scheduling would make output interleaving and
//! resource accounting depend on host scheduling order, which conflicts with this simulator's
//! determinism goal.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::commands::makecmd::{self, Makefile, RecipeLine, Rule};
use crate::exec::ShellPoll;
use crate::process::ProcessId;
use crate::scheduler::WaitReason;
use crate::syscalls::{FileKind, SpawnSpec, System};

use super::poll_write;

/// Bound on prerequisite chain depth, independent of cycle detection (which catches a target
/// depending on itself transitively; this catches pathologically long acyclic chains).
const MAX_DEPTH: usize = 256;

#[derive(Clone, Copy)]
struct TargetResult {
    mtime: u64,
    rebuilt: bool,
}

/// A target whose prerequisites are being built, one at a time, left to right.
#[derive(Clone)]
struct PrereqFrame {
    target: String,
    top_level: bool,
    exists: bool,
    prerequisites: Vec<String>,
    results: Vec<TargetResult>,
}

/// A target whose recipe lines are being run, one at a time, in order.
#[derive(Clone)]
struct RecipeFrame {
    target: String,
    top_level: bool,
    lines: Vec<RecipeLine>,
    prerequisites: Vec<String>,
    index: usize,
}

#[derive(Clone)]
enum Frame {
    Prereqs(PrereqFrame),
    Recipes(RecipeFrame),
}

/// One recipe line in flight: echo (unless silenced), then spawn (unless `-n`), then wait.
#[derive(Clone)]
struct ActiveLine {
    target: String,
    line_no: usize,
    ignore_error: bool,
    spawn_after: bool,
    command: String,
    echo_bytes: Vec<u8>,
    echo_offset: usize,
    pid: Option<ProcessId>,
}

/// Persistent state for one `make` invocation once its options and Makefile are parsed.
#[derive(Clone)]
struct Run {
    cwd: String,
    makefile_name: String,
    vars: BTreeMap<String, String>,
    phony: BTreeSet<String>,
    rules: BTreeMap<String, Rule>,
    dry_run: bool,
    silent: bool,
    keep_going: bool,
    child_environment: BTreeMap<String, String>,
    pending_targets: VecDeque<String>,
    stack: Vec<Frame>,
    visited: BTreeMap<String, TargetResult>,
    visiting: BTreeSet<String>,
    recipe_count: usize,
    exit_status: i32,
    /// Set once a non-`-k` failure means the whole run must stop after unwinding in-flight work.
    fatal: bool,
    active: Option<ActiveLine>,
    /// Advisory messages (up-to-date, nothing-to-do, circular dependency) queued for output.
    /// Draining these through `poll_write` keeps them subject to the same output metering and
    /// backpressure as every other write, instead of a raw best-effort `System::write` loop.
    pending_stdout: Vec<u8>,
    pending_stdout_offset: usize,
    pending_stderr: Vec<u8>,
    pending_stderr_offset: usize,
}

#[derive(Clone)]
pub(crate) struct MakeProcess {
    args: Vec<String>,
    started: bool,
    diagnostic: Vec<u8>,
    diagnostic_offset: usize,
    status: i32,
    run: Option<Run>,
}

impl MakeProcess {
    pub(super) fn new(args: &[String]) -> Self {
        Self {
            args: args.to_vec(),
            started: false,
            diagnostic: Vec::new(),
            diagnostic_offset: 0,
            status: 0,
            run: None,
        }
    }

    pub(super) fn poll(&mut self, system: &mut impl System) -> ShellPoll {
        if !self.diagnostic.is_empty() {
            return poll_write(
                system,
                2,
                &self.diagnostic,
                &mut self.diagnostic_offset,
                self.status,
            );
        }
        if !self.started {
            self.started = true;
            match Run::start(system, &self.args) {
                Ok(run) => self.run = Some(run),
                Err((status, message)) => {
                    self.fail(status, message);
                    return ShellPoll::Pending;
                }
            }
        }
        let Some(run) = self.run.as_mut() else {
            return ShellPoll::Ready(self.status);
        };
        match run.poll(system) {
            RunOutcome::Progressed(poll) => poll,
            RunOutcome::Done(status) => ShellPoll::Ready(status),
            RunOutcome::Failed(status, message) => {
                self.run = None;
                self.fail(status, message);
                ShellPoll::Pending
            }
        }
    }

    fn fail(&mut self, status: i32, message: String) {
        self.status = status;
        self.diagnostic = message.into_bytes();
    }
}

/// What one call to `Run::poll` accomplished.
enum RunOutcome {
    /// Forward this scheduler outcome (a blocked wait, a switch to a spawned child, or pending
    /// backpressure on a write); the run has more work.
    Progressed(ShellPoll),
    /// The run finished; deliver this exit status once any diagnostic has been written.
    Done(i32),
    /// A fatal condition ended the run; write this diagnostic, then exit with this status.
    Failed(i32, String),
}

impl Run {
    fn start(system: &mut impl System, args: &[String]) -> Result<Self, (i32, String)> {
        let Some(options) = makecmd::parse_args(args) else {
            return Err((2, "make: unsupported option or malformed argument\n".into()));
        };
        let cwd = match options.directory.as_deref() {
            Some(directory) => {
                let resolved = crate::vfs::resolve_against(system.cwd(), directory);
                match system.metadata("/", &resolved, true) {
                    Ok(info) if info.kind == FileKind::Directory => {}
                    Ok(_) => {
                        return Err((2, format!("make: {resolved}: Not a directory\n")));
                    }
                    Err(_) => {
                        return Err((2, format!("make: {resolved}: No such directory\n")));
                    }
                }
                let canonical = system
                    .canonicalize("/", &resolved, true)
                    .unwrap_or(resolved);
                if let Err(error) = system.chdir(&canonical) {
                    return Err((2, format!("make: cannot chdir to {canonical}: {error}\n")));
                }
                canonical
            }
            None => system.cwd().to_string(),
        };
        let makefile_name = match options.file.as_deref() {
            Some("-") => {
                return Err((
                    2,
                    "make: reading makefiles from stdin is unsupported\n".into(),
                ));
            }
            Some(path) => path.to_string(),
            None if is_file(system, &cwd, "Makefile") => "Makefile".to_string(),
            None if is_file(system, &cwd, "makefile") => "makefile".to_string(),
            None => {
                return Err((
                    2,
                    "make: *** No targets specified and no makefile found.  Stop.\n".into(),
                ));
            }
        };
        let source = match read_makefile(system, &cwd, &makefile_name) {
            Ok(source) => source,
            Err(error) => return Err((2, format!("make: {error}\n"))),
        };
        let mut makefile: Makefile = makecmd::parse_makefile(system, &source)
            .map_err(|error| (2, format!("make: {error}\n")))?;
        let child_environment =
            build_child_environment(system.environment(), &makefile, &options.command_vars);
        for (name, value) in options.command_vars {
            makefile.vars.insert(name, value);
        }
        let targets: Vec<String> = if options.targets.is_empty() {
            match makefile.order.first().cloned() {
                Some(target) => vec![target],
                None => return Err((2, "make: *** No targets.  Stop.\n".into())),
            }
        } else {
            options.targets
        };
        if options.jobs.is_some() {
            // Accepted for compatibility; see the module docs for why execution stays serial.
            system.note_unsupported("make:-j (recipes run serially)");
        }
        Ok(Self {
            cwd,
            makefile_name,
            vars: makefile.vars,
            phony: makefile.phony,
            rules: makefile.rules,
            dry_run: options.dry_run,
            silent: options.silent,
            keep_going: options.keep_going,
            child_environment,
            pending_targets: targets.into(),
            stack: Vec::new(),
            visited: BTreeMap::new(),
            visiting: BTreeSet::new(),
            recipe_count: 0,
            exit_status: 0,
            fatal: false,
            active: None,
            pending_stdout: Vec::new(),
            pending_stdout_offset: 0,
            pending_stderr: Vec::new(),
            pending_stderr_offset: 0,
        })
    }

    fn poll(&mut self, system: &mut impl System) -> RunOutcome {
        loop {
            match poll_write(
                system,
                2,
                &self.pending_stderr,
                &mut self.pending_stderr_offset,
                0,
            ) {
                ShellPoll::Ready(0) => {}
                other => return RunOutcome::Progressed(other),
            }
            match poll_write(
                system,
                1,
                &self.pending_stdout,
                &mut self.pending_stdout_offset,
                0,
            ) {
                ShellPoll::Ready(0) => {}
                other => return RunOutcome::Progressed(other),
            }
            if let Some(active) = self.active.clone() {
                match self.poll_active(system, active) {
                    Some(outcome) => return outcome,
                    None => continue,
                }
            }
            if self.fatal {
                self.stack.clear();
                self.pending_targets.clear();
                return RunOutcome::Done(self.exit_status);
            }
            let Some(frame) = self.stack.pop() else {
                let Some(target) = self.pending_targets.pop_front() else {
                    return RunOutcome::Done(self.exit_status);
                };
                match self.enter_target(system, target, true) {
                    Ok(()) => continue,
                    Err(failure) => return failure,
                }
            };
            match frame {
                Frame::Prereqs(prereqs) => match self.advance_prereqs(system, prereqs) {
                    Ok(()) => continue,
                    Err(failure) => return failure,
                },
                Frame::Recipes(recipes) => match self.advance_recipes(system, recipes) {
                    Ok(()) => continue,
                    Err(failure) => return failure,
                },
            }
        }
    }

    /// Push a `Prereqs` frame for `target`, or resolve it immediately if it is a leaf (no rule)
    /// or already visited. Detects "no rule" failures and cycles.
    fn enter_target(
        &mut self,
        system: &mut impl System,
        target: String,
        top_level: bool,
    ) -> Result<(), RunOutcome> {
        if self.stack.len() >= MAX_DEPTH {
            return Err(RunOutcome::Failed(
                2,
                format!("make: prerequisite chain too deep for '{target}'\n"),
            ));
        }
        if let Some(result) = self.visited.get(&target) {
            self.deliver_result(system, target, *result, top_level)?;
            return Ok(());
        }
        if self.visiting.contains(&target) {
            let dependent = self
                .stack
                .iter()
                .rev()
                .map(|frame| match frame {
                    Frame::Prereqs(p) => p.target.clone(),
                    Frame::Recipes(r) => r.target.clone(),
                })
                .next()
                .unwrap_or_default();
            system.note_unsupported("make:circular dependency");
            self.pending_stderr.extend_from_slice(
                format!("make: Circular {target} <- {dependent} dependency dropped.\n").as_bytes(),
            );
            self.deliver_result(
                system,
                target,
                TargetResult {
                    mtime: 0,
                    rebuilt: false,
                },
                top_level,
            )?;
            return Ok(());
        }
        if !system.charge_cpu(target.len() as u64) {
            return Err(RunOutcome::Failed(
                2,
                system_stop_message("resource limit exceeded while planning targets"),
            ));
        }
        let rule = self.rules.get(&target).cloned();
        let exists = target_exists(system, &self.cwd, &target);
        let Some(rule) = rule else {
            if exists {
                let mtime = target_mtime(system, &self.cwd, &target);
                let result = TargetResult {
                    mtime,
                    rebuilt: false,
                };
                self.visited.insert(target.clone(), result);
                self.deliver_result(system, target, result, top_level)?;
                return Ok(());
            }
            return Err(RunOutcome::Failed(
                2,
                format!("make: *** No rule to make target '{target}'.  Stop.\n"),
            ));
        };
        let prerequisites = match self.expand_prerequisites(system, &target, &rule.prerequisites) {
            Ok(prerequisites) => prerequisites,
            Err(error) => return Err(RunOutcome::Failed(2, format!("make: {error}\n"))),
        };
        self.visiting.insert(target.clone());
        self.stack.push(Frame::Prereqs(PrereqFrame {
            target,
            top_level,
            exists,
            prerequisites,
            results: Vec::new(),
        }));
        Ok(())
    }

    /// Feed a completed prerequisite's result back to whatever frame is waiting on it, or, for a
    /// top-level target with no remaining stack, print the up-to-date/nothing-to-do message.
    fn deliver_result(
        &mut self,
        _system: &mut impl System,
        target: String,
        result: TargetResult,
        top_level: bool,
    ) -> Result<(), RunOutcome> {
        if let Some(Frame::Prereqs(frame)) = self.stack.last_mut() {
            frame.results.push(result);
            return Ok(());
        }
        if top_level && !result.rebuilt {
            let has_recipe = self
                .rules
                .get(&target)
                .is_some_and(|rule| !rule.recipes.is_empty());
            let message = if has_recipe {
                format!("make: '{target}' is up to date.\n")
            } else {
                format!("make: Nothing to be done for '{target}'.\n")
            };
            self.pending_stdout.extend_from_slice(message.as_bytes());
        }
        Ok(())
    }

    fn advance_prereqs(
        &mut self,
        system: &mut impl System,
        frame: PrereqFrame,
    ) -> Result<(), RunOutcome> {
        if frame.results.len() < frame.prerequisites.len() {
            let next = frame.prerequisites[frame.results.len()].clone();
            self.stack.push(Frame::Prereqs(frame));
            return self.enter_target(system, next, false);
        }
        self.visiting.remove(&frame.target);
        let target_mtime = target_mtime(system, &self.cwd, &frame.target);
        let needs_rebuild = self.phony.contains(&frame.target)
            || !frame.exists
            || frame
                .results
                .iter()
                .any(|result| result.rebuilt || result.mtime > target_mtime);
        let rule = self
            .rules
            .get(&frame.target)
            .cloned()
            .expect("a Prereqs frame always has a backing rule");
        if !needs_rebuild || rule.recipes.is_empty() {
            let result = TargetResult {
                mtime: target_mtime,
                rebuilt: false,
            };
            self.visited.insert(frame.target.clone(), result);
            return self.deliver_result(system, frame.target, result, frame.top_level);
        }
        self.stack.push(Frame::Recipes(RecipeFrame {
            target: frame.target,
            top_level: frame.top_level,
            lines: rule.recipes,
            prerequisites: frame.prerequisites,
            index: 0,
        }));
        Ok(())
    }

    fn advance_recipes(
        &mut self,
        system: &mut impl System,
        mut frame: RecipeFrame,
    ) -> Result<(), RunOutcome> {
        if frame.index >= frame.lines.len() {
            let mtime = target_mtime(system, &self.cwd, &frame.target);
            let result = TargetResult {
                mtime,
                rebuilt: true,
            };
            self.visited.insert(frame.target.clone(), result);
            return self.deliver_result(system, frame.target, result, frame.top_level);
        }
        let line = frame.lines[frame.index].clone();
        self.recipe_count = self.recipe_count.saturating_add(1);
        if self.recipe_count > makecmd::MAX_RECIPES {
            return Err(RunOutcome::Failed(
                2,
                format!("make: recipe limit exceeded ({})\n", makecmd::MAX_RECIPES),
            ));
        }
        if !system.charge_cpu(line.text.len() as u64) {
            return Err(RunOutcome::Failed(
                2,
                system_stop_message("resource limit exceeded while expanding recipe"),
            ));
        }
        let expanded = match makecmd::expand_vars(
            &line.text,
            &self.vars,
            &frame.target,
            &frame.prerequisites,
            makecmd::MAX_EXPANSION_DEPTH,
        ) {
            Ok(expanded) => expanded,
            Err(error) => return Err(RunOutcome::Failed(2, format!("make: {error}\n"))),
        };
        let (silent, ignore_error, command) = makecmd::recipe_prefixes(&expanded);
        if command.trim().is_empty() {
            frame.index += 1;
            self.stack.push(Frame::Recipes(frame));
            return Ok(());
        }
        let show = self.dry_run || !(silent || self.silent);
        let mut echo_bytes = Vec::new();
        if show {
            echo_bytes.extend_from_slice(command.as_bytes());
            echo_bytes.push(b'\n');
        }
        frame.index += 1;
        let target = frame.target.clone();
        let line_no = line.line;
        let spawn_after = !self.dry_run;
        let command = command.to_string();
        self.stack.push(Frame::Recipes(frame));
        self.active = Some(ActiveLine {
            target,
            line_no,
            ignore_error,
            spawn_after,
            command,
            echo_bytes,
            echo_offset: 0,
            pid: None,
        });
        Ok(())
    }

    /// Drive the currently active recipe line (echo, then spawn, then wait). Returns `Some` when
    /// the whole run must yield or stop; `None` when the caller's outer loop should keep going
    /// (the line finished synchronously, e.g. a `-n` echo, or its child just finished).
    fn poll_active(
        &mut self,
        system: &mut impl System,
        mut active: ActiveLine,
    ) -> Option<RunOutcome> {
        if active.pid.is_none() {
            match poll_write(system, 1, &active.echo_bytes, &mut active.echo_offset, 0) {
                ShellPoll::Ready(0) => {}
                ShellPoll::Ready(status) => {
                    self.active = None;
                    return Some(RunOutcome::Done(status));
                }
                other => {
                    self.active = Some(active);
                    return Some(RunOutcome::Progressed(other));
                }
            }
            if !active.spawn_after {
                self.active = None;
                return None;
            }
            match system.spawn_argv(SpawnSpec {
                argv: vec!["sh".to_string(), "-c".to_string(), active.command.clone()],
                // GNU make recipes inherit make's own stdin rather than an empty stream.
                stdin: None,
                cwd: None,
                environment: Some(self.child_environment.clone()),
                ..Default::default()
            }) {
                Ok(pid) => {
                    active.pid = Some(pid);
                    self.active = Some(active);
                    return Some(RunOutcome::Progressed(ShellPoll::Switched));
                }
                Err(error) => {
                    self.active = None;
                    return Some(RunOutcome::Failed(
                        2,
                        format!("make: cannot run recipe: {error}\n"),
                    ));
                }
            }
        }
        let pid = active.pid.expect("checked above");
        match system.child_status(pid) {
            Ok(None) => {
                self.active = Some(active);
                Some(RunOutcome::Progressed(ShellPoll::Blocked(
                    WaitReason::Child(pid),
                )))
            }
            Ok(Some(status)) => {
                if let Err(error) = system.reap_child(pid) {
                    self.active = None;
                    return Some(RunOutcome::Failed(
                        2,
                        format!("make: cannot reap recipe: {error}\n"),
                    ));
                }
                self.active = None;
                if status != 0 && !active.ignore_error {
                    self.pending_stderr.extend_from_slice(
                        format!(
                            "make: *** [{}:{}: {}] Error {}\n",
                            self.makefile_name, active.line_no, active.target, status
                        )
                        .as_bytes(),
                    );
                    self.exit_status = 2;
                    // Stop this target's own remaining recipe lines either way: a failed step
                    // means later steps for the same target are unlikely to be meaningful. Mark
                    // the target visited (so a diamond dependency does not retry the recipe) and,
                    // under `-k`, hand a result to whatever is waiting on it so traversal keeps
                    // moving; without `-k` the run stops on the next outer loop iteration.
                    let frame_still_active = matches!(
                        self.stack.last(),
                        Some(Frame::Recipes(frame)) if frame.target == active.target
                    );
                    if frame_still_active {
                        self.stack.pop();
                    }
                    let result = TargetResult {
                        mtime: target_mtime(system, &self.cwd, &active.target),
                        rebuilt: true,
                    };
                    self.visited.insert(active.target.clone(), result);
                    if !self.keep_going {
                        self.fatal = true;
                    } else if let Err(failure) =
                        self.deliver_result(system, active.target.clone(), result, false)
                    {
                        return Some(failure);
                    }
                }
                None
            }
            Err(error) => {
                self.active = None;
                Some(RunOutcome::Failed(
                    2,
                    format!("make: cannot wait for recipe: {error}\n"),
                ))
            }
        }
    }

    fn expand_prerequisites(
        &self,
        system: &mut impl System,
        target: &str,
        raw_prerequisites: &[String],
    ) -> Result<Vec<String>, String> {
        let source_bytes = raw_prerequisites
            .iter()
            .try_fold(0_u64, |total, value| total.checked_add(value.len() as u64))
            .ok_or_else(|| "prerequisite expansion is too large".to_string())?;
        if !system.charge_cpu(source_bytes) {
            return Err("resource limit exceeded while expanding prerequisites".to_string());
        }
        let mut expanded = Vec::new();
        for prerequisite in raw_prerequisites {
            let value = makecmd::expand_vars(
                prerequisite,
                &self.vars,
                target,
                &[],
                makecmd::MAX_EXPANSION_DEPTH,
            )?;
            for name in value.split_whitespace() {
                if expanded.len() >= makecmd::MAX_TARGETS {
                    return Err(format!(
                        "expanded prerequisite limit exceeded ({})",
                        makecmd::MAX_TARGETS
                    ));
                }
                makecmd::validate_name(name, 0)
                    .map_err(|_| format!("variable produced unsupported prerequisite '{name}'"))?;
                expanded.push(name.to_string());
            }
        }
        Ok(expanded)
    }
}

fn is_file(system: &mut impl System, cwd: &str, name: &str) -> bool {
    matches!(
        system.metadata(cwd, name, true),
        Ok(info) if info.kind == FileKind::File
    )
}

fn target_exists(system: &mut impl System, cwd: &str, target: &str) -> bool {
    system.metadata(cwd, target, true).is_ok()
}

fn target_mtime(system: &mut impl System, cwd: &str, target: &str) -> u64 {
    system
        .metadata(cwd, target, true)
        .map(|info| info.mtime_ms)
        .unwrap_or(0)
}

fn read_makefile(system: &mut impl System, cwd: &str, name: &str) -> Result<String, String> {
    let bytes = system
        .read_file_limited(cwd, name, makecmd::MAX_MAKEFILE_BYTES)
        .map_err(|error| error.to_string())?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn system_stop_message(context: &str) -> String {
    format!("make: {context}\n")
}

/// Compute the environment recipe children see, matching GNU make's export rules: a variable
/// that came from the process environment is exported automatically (using the makefile's value
/// if it overrides the variable), a makefile-only variable is exported only when named by an
/// explicit `export` directive, `unexport` removes a variable from the recipe environment even
/// if it came from the process environment, and command-line `make VAR=value` overrides are
/// always exported regardless of the above.
fn build_child_environment(
    ambient: BTreeMap<String, String>,
    makefile: &Makefile,
    command_vars: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut child = ambient.clone();
    let mut names: BTreeSet<String> = ambient.keys().cloned().collect();
    names.extend(makefile.vars.keys().cloned());
    names.extend(makefile.export_state.keys().cloned());
    for name in names {
        if command_vars.contains_key(&name) {
            continue;
        }
        let was_ambient = ambient.contains_key(&name);
        let exported = makefile
            .export_state
            .get(&name)
            .copied()
            .unwrap_or(was_ambient);
        if exported {
            if let Some(value) = makefile.vars.get(&name) {
                child.insert(name.clone(), value.clone());
            } else if !was_ambient {
                child.remove(&name);
            }
        } else {
            child.remove(&name);
        }
    }
    for (name, value) in command_vars {
        child.insert(name.clone(), value.clone());
    }
    child
}
