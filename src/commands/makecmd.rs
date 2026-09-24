//! A bounded, deterministic Makefile runner for shell-only recipes.
//!
//! This module intentionally implements a small graph evaluator rather than a build system:
//! rules, prerequisites, variables, timestamps, and tab-indented recipes are supported, while
//! pattern rules, includes, implicit search, parallelism, and host execution are rejected. Every
//! recipe is dispatched through shellsim's own interpreter against the VFS.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::commands::util::{ewln, wln};
use crate::commands::{ChildCommand, CommandContext, CommandPoll, CommandSpec, Io, Trust};
use crate::interp::Interp;

const MAX_MAKEFILE_BYTES: usize = 1024 * 1024;
const MAX_LINES: usize = 20_000;
const MAX_TARGETS: usize = 4_096;
const MAX_RECIPES: usize = 20_000;
const MAX_EXPANSION_DEPTH: usize = 16;
const MAX_EXPANDED_BYTES: usize = 4 * 1024 * 1024;

/// Register the modeled `make` command. The central command registry calls this function when
/// the module is enabled; it is kept public so the registry can remain a thin composition layer.
pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    super::reg_buffered_resumable(m, &["make"], Trust::Partial, cmd_make, start_make);
}

#[derive(Clone, Debug)]
struct Rule {
    prerequisites: Vec<String>,
    recipes: Vec<String>,
}

#[derive(Default)]
struct Makefile {
    vars: BTreeMap<String, String>,
    rules: BTreeMap<String, Rule>,
    order: Vec<String>,
    phony: BTreeSet<String>,
}

struct MakeRun<'a, 'b> {
    interp: &'a mut Interp,
    out: &'b mut Vec<u8>,
    err: &'b mut Vec<u8>,
    vars: BTreeMap<String, String>,
    phony: BTreeSet<String>,
    rules: BTreeMap<String, Rule>,
    visiting: BTreeSet<String>,
    visited: BTreeSet<String>,
    dry_run: bool,
    recipe_count: usize,
}

#[derive(Clone, Copy)]
struct PlannedTarget {
    mtime: u64,
    rebuilt: bool,
}

struct PlannedRecipe {
    source: String,
    display: String,
}

struct MakePlan<'a> {
    interp: &'a mut Interp,
    cwd: String,
    vars: BTreeMap<String, String>,
    phony: BTreeSet<String>,
    rules: BTreeMap<String, Rule>,
    visiting: BTreeSet<String>,
    visited: BTreeMap<String, PlannedTarget>,
    recipes: Vec<PlannedRecipe>,
    recipe_count: usize,
}

fn cmd_make(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(options) = parse_args(args) else {
        ewln(io.err, "make: unsupported option or malformed argument");
        return 2;
    };
    let Some(makefile_path) = options.file.clone() else {
        // GNU make's default is Makefile, then makefile. Keep lookup deterministic and VFS-only.
        let default = if interp.vfs.is_file(&interp.cwd, "Makefile") {
            "Makefile"
        } else {
            "makefile"
        };
        if !interp.vfs.is_file(&interp.cwd, default) {
            ewln(
                io.err,
                "make: *** No targets specified and no makefile found.  Stop.",
            );
            return 2;
        }
        return run_make(interp, io, options, default);
    };
    if makefile_path == "-" {
        ewln(io.err, "make: reading makefiles from stdin is unsupported");
        return 2;
    }
    run_make(interp, io, options, &makefile_path)
}

fn start_make(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let Some(options) = parse_args(args) else {
        ewln(io.err, "make: unsupported option or malformed argument");
        return CommandPoll::Ready(2);
    };
    let cwd = match options.directory.as_deref() {
        Some(directory) => {
            let resolved = crate::vfs::resolve_against(&interp.cwd, directory);
            if !interp.vfs.is_dir("/", &resolved) {
                ewln(io.err, &format!("make: {resolved}: No such directory"));
                return CommandPoll::Ready(2);
            }
            interp.vfs.realpath(&resolved, true).unwrap_or(resolved)
        }
        None => interp.cwd.clone(),
    };
    let makefile_path = match options.file.as_deref() {
        Some("-") => {
            ewln(io.err, "make: reading makefiles from stdin is unsupported");
            return CommandPoll::Ready(2);
        }
        Some(path) => path.to_string(),
        None if interp.vfs.is_file(&cwd, "Makefile") => "Makefile".to_string(),
        None if interp.vfs.is_file(&cwd, "makefile") => "makefile".to_string(),
        None => {
            ewln(
                io.err,
                "make: *** No targets specified and no makefile found.  Stop.",
            );
            return CommandPoll::Ready(2);
        }
    };
    let recipes = match plan_make(interp, options, &cwd, &makefile_path) {
        Ok(recipes) => recipes,
        Err(error) => {
            ewln(io.err, &format!("make: {error}"));
            return CommandPoll::Ready(2);
        }
    };
    if recipes.is_empty() {
        return CommandPoll::Ready(0);
    }
    let mut commands = Vec::with_capacity(recipes.len());
    for recipe in recipes {
        if recipe.source.is_empty() {
            wln(io.out, &recipe.display);
            continue;
        }
        let source = if recipe.display.is_empty() {
            format!("{} || exit 2", recipe.source)
        } else {
            format!(
                "printf '%s\\n' {}; {} || exit 2",
                shell_quote(&recipe.display),
                recipe.source
            )
        };
        commands.push(ChildCommand {
            argv: vec!["bash".to_string(), "-c".to_string(), source],
            stdin: Vec::new(),
            cwd: Some(cwd.clone()),
            environment: None,
        });
    }
    crate::commands::start_child_sequence(interp, commands, true)
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn recipe_prefixes(recipe: &str) -> (bool, bool, &str) {
    let mut command = recipe.trim_start();
    let mut silent = false;
    let mut ignore_error = false;
    loop {
        if let Some(rest) = command.strip_prefix('@') {
            silent = true;
            command = rest;
        } else if let Some(rest) = command.strip_prefix('-') {
            ignore_error = true;
            command = rest;
        } else {
            break;
        }
    }
    (silent, ignore_error, command.trim_start())
}

#[derive(Default)]
struct Options {
    file: Option<String>,
    directory: Option<String>,
    dry_run: bool,
    targets: Vec<String>,
    command_vars: BTreeMap<String, String>,
}

fn parse_args(args: &[String]) -> Option<Options> {
    let mut options = Options::default();
    let mut end_options = false;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if !end_options && arg == "--" {
            end_options = true;
            i += 1;
            continue;
        }
        if !end_options && arg == "-n" {
            options.dry_run = true;
            i += 1;
            continue;
        }
        if !end_options && (arg == "-f" || arg == "--file") {
            i += 1;
            options.file = Some(args.get(i)?.clone());
            i += 1;
            continue;
        }
        if !end_options && (arg == "-C" || arg == "--directory") {
            i += 1;
            options.directory = Some(args.get(i)?.clone());
            i += 1;
            continue;
        }
        if !end_options && arg.starts_with('-') {
            return None;
        }
        if let Some((name, value)) = parse_assignment(arg) {
            options.command_vars.insert(name, value);
        } else {
            if arg.contains('=') && !arg.starts_with('-') {
                return None;
            }
            options.targets.push(arg.clone());
        }
        i += 1;
    }
    Some(options)
}

fn run_make(
    interp: &mut CommandContext<'_>,
    io: &mut Io<'_>,
    options: Options,
    makefile_path: &str,
) -> i32 {
    let original_cwd = interp.cwd.clone();
    if let Some(directory) = options.directory.as_deref() {
        let directory = crate::vfs::resolve_against(&original_cwd, directory);
        if !interp.vfs.is_dir("/", &directory) {
            ewln(io.err, &format!("make: {directory}: No such directory"));
            return 2;
        }
        interp.cwd = interp.vfs.realpath(&directory, true).unwrap_or(directory);
        let cwd = interp.cwd.clone();
        interp.set_var("PWD", cwd);
    }

    let result = execute_make(interp, io, options, makefile_path);
    interp.cwd = original_cwd.clone();
    interp.set_var("PWD", original_cwd);
    match result {
        Ok(()) => 0,
        Err(error) => {
            ewln(io.err, &format!("make: {error}"));
            2
        }
    }
}

fn execute_make(
    interp: &mut CommandContext<'_>,
    io: &mut Io<'_>,
    options: Options,
    makefile_path: &str,
) -> Result<(), String> {
    let path = crate::vfs::resolve_against(&interp.cwd, makefile_path);
    let size = interp.vfs.file_len("/", &path).map_err(|e| e.to_string())?;
    if size > MAX_MAKEFILE_BYTES {
        return Err(format!(
            "makefile exceeds the {MAX_MAKEFILE_BYTES}-byte limit"
        ));
    }
    let source = interp
        .vfs
        .read_string_limited("/", &path, MAX_MAKEFILE_BYTES)
        .map_err(|e| e.to_string())?;
    let mut makefile = parse_makefile(interp, &source)?;
    for (name, value) in options.command_vars {
        makefile.vars.insert(name, value);
    }
    let target = options
        .targets
        .first()
        .cloned()
        .or_else(|| makefile.order.first().cloned())
        .ok_or_else(|| "no targets specified".to_string())?;
    let targets = if options.targets.is_empty() {
        vec![target]
    } else {
        options.targets
    };
    let mut runner = MakeRun {
        interp,
        out: io.out,
        err: io.err,
        vars: makefile.vars,
        phony: makefile.phony,
        rules: makefile.rules,
        visiting: BTreeSet::new(),
        visited: BTreeSet::new(),
        dry_run: options.dry_run,
        recipe_count: 0,
    };
    for target in targets {
        runner.build(&target, None)?;
    }
    Ok(())
}

fn plan_make(
    interp: &mut Interp,
    options: Options,
    cwd: &str,
    makefile_path: &str,
) -> Result<Vec<PlannedRecipe>, String> {
    let path = crate::vfs::resolve_against(cwd, makefile_path);
    let size = interp.vfs.file_len("/", &path).map_err(|e| e.to_string())?;
    if size > MAX_MAKEFILE_BYTES {
        return Err(format!(
            "makefile exceeds the {MAX_MAKEFILE_BYTES}-byte limit"
        ));
    }
    let source = interp
        .vfs
        .read_string_limited("/", &path, MAX_MAKEFILE_BYTES)
        .map_err(|e| e.to_string())?;
    let mut makefile = parse_makefile(interp, &source)?;
    for (name, value) in options.command_vars {
        makefile.vars.insert(name, value);
    }
    let targets = if options.targets.is_empty() {
        vec![makefile
            .order
            .first()
            .cloned()
            .ok_or_else(|| "no targets specified".to_string())?]
    } else {
        options.targets
    };
    let mut planner = MakePlan {
        interp,
        cwd: cwd.to_string(),
        vars: makefile.vars,
        phony: makefile.phony,
        rules: makefile.rules,
        visiting: BTreeSet::new(),
        visited: BTreeMap::new(),
        recipes: Vec::new(),
        recipe_count: 0,
    };
    for target in targets {
        planner.build(&target, None, options.dry_run)?;
    }
    Ok(planner.recipes)
}

impl MakePlan<'_> {
    fn build(
        &mut self,
        target: &str,
        parent: Option<&str>,
        dry_run: bool,
    ) -> Result<PlannedTarget, String> {
        if self.visiting.contains(target) {
            return Err(format!("circular dependency involving '{target}'"));
        }
        if let Some(result) = self.visited.get(target) {
            return Ok(*result);
        }
        self.visiting.insert(target.to_string());
        let rule = self.rules.get(target).cloned();
        let exists = self.interp.vfs.exists(&self.cwd, target);
        if rule.is_none() {
            self.visiting.remove(target);
            if exists {
                let result = PlannedTarget {
                    mtime: self.mtime(target),
                    rebuilt: false,
                };
                self.visited.insert(target.to_string(), result);
                return Ok(result);
            }
            return Err(format!("No rule to make target '{target}'"));
        }
        let rule = rule.expect("rule presence was checked");
        let prerequisites = self.expand_prerequisites(target, &rule.prerequisites)?;
        let mut prerequisite_results = Vec::with_capacity(prerequisites.len());
        for prerequisite in &prerequisites {
            prerequisite_results.push(self.build(prerequisite, Some(target), dry_run)?);
        }
        let target_mtime = self.mtime(target);
        let needs = self.phony.contains(target)
            || !exists
            || prerequisite_results
                .iter()
                .any(|result| result.rebuilt || result.mtime > target_mtime);
        if needs {
            for recipe in &rule.recipes {
                self.recipe_count = self.recipe_count.saturating_add(1);
                if self.recipe_count > MAX_RECIPES {
                    return Err(format!("recipe limit exceeded ({MAX_RECIPES})"));
                }
                let expanded = self.expand(recipe, target, parent, &prerequisites)?;
                if expanded.trim().is_empty() {
                    continue;
                }
                let (silent, ignore_error, command) = recipe_prefixes(&expanded);
                self.recipes.push(PlannedRecipe {
                    source: if dry_run {
                        String::new()
                    } else if ignore_error {
                        format!("( {command} ) || true")
                    } else {
                        command.to_string()
                    },
                    display: if silent && !dry_run {
                        String::new()
                    } else {
                        command.to_string()
                    },
                });
            }
        }
        self.visiting.remove(target);
        let result = PlannedTarget {
            mtime: target_mtime,
            rebuilt: needs,
        };
        self.visited.insert(target.to_string(), result);
        Ok(result)
    }

    fn mtime(&self, target: &str) -> u64 {
        self.interp
            .vfs
            .metadata(&self.cwd, target, true)
            .map(|node| node.mtime)
            .unwrap_or(0)
    }

    fn expand(
        &mut self,
        recipe: &str,
        target: &str,
        _parent: Option<&str>,
        prerequisites: &[String],
    ) -> Result<String, String> {
        if !self.interp.resources.charge_cpu(recipe.len() as u64) {
            return Err("resource limit exceeded while expanding recipe".to_string());
        }
        expand_vars(
            recipe,
            &self.vars,
            target,
            prerequisites,
            MAX_EXPANSION_DEPTH,
        )
    }

    fn expand_prerequisites(
        &mut self,
        target: &str,
        raw_prerequisites: &[String],
    ) -> Result<Vec<String>, String> {
        let source_bytes = raw_prerequisites
            .iter()
            .try_fold(0_u64, |total, value| total.checked_add(value.len() as u64))
            .ok_or_else(|| "prerequisite expansion is too large".to_string())?;
        if !self.interp.resources.charge_cpu(source_bytes) {
            return Err("resource limit exceeded while expanding prerequisites".to_string());
        }
        let mut expanded = Vec::new();
        for prerequisite in raw_prerequisites {
            let value = expand_vars(prerequisite, &self.vars, target, &[], MAX_EXPANSION_DEPTH)?;
            for name in value.split_whitespace() {
                if expanded.len() >= MAX_TARGETS {
                    return Err(format!(
                        "expanded prerequisite limit exceeded ({MAX_TARGETS})"
                    ));
                }
                validate_name(name, 0)
                    .map_err(|_| format!("variable produced unsupported prerequisite '{name}'"))?;
                expanded.push(name.to_string());
            }
        }
        Ok(expanded)
    }
}

fn parse_makefile(interp: &mut Interp, source: &str) -> Result<Makefile, String> {
    let bytes = source.len();
    if !interp.resources.charge_cpu(bytes as u64)
        || !interp
            .resources
            .reserve_memory((bytes as u64).saturating_mul(2))
    {
        return Err("resource limit exceeded while reading makefile".to_string());
    }
    let mut out = Makefile::default();
    let mut current_targets: Vec<String> = Vec::new();
    let mut current_has_prior_recipe = false;
    let mut recipe_count = 0usize;
    for (line_no, raw) in logical_make_lines(source)? {
        if let Some(recipe) = raw.strip_prefix('\t') {
            if current_targets.is_empty() {
                return Err(format!("recipe without a target at line {line_no}"));
            }
            if current_has_prior_recipe {
                return Err(format!(
                    "target has multiple recipe definitions at line {line_no}"
                ));
            }
            let recipe = recipe.to_string();
            recipe_count = recipe_count.saturating_add(1);
            if recipe_count > MAX_RECIPES {
                return Err(format!("recipe limit exceeded ({MAX_RECIPES})"));
            }
            for target in &current_targets {
                out.rules
                    .get_mut(target)
                    .ok_or_else(|| "internal rule state error".to_string())?
                    .recipes
                    .push(recipe.clone());
            }
            continue;
        }
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('.') {
            let Some(rest) = line.strip_prefix(".PHONY:") else {
                return Err(format!("unsupported directive at line {line_no}"));
            };
            for target in rest.split_whitespace() {
                validate_name(target, line_no)?;
                out.phony.insert(target.to_string());
            }
            current_targets.clear();
            current_has_prior_recipe = false;
            continue;
        }
        if let Some((name, value)) = parse_assignment(line) {
            out.vars.insert(name, value);
            current_targets.clear();
            current_has_prior_recipe = false;
            continue;
        }
        // Make expands target and prerequisite names while reading a rule, unlike recipes.
        let expanded = expand_vars(line, &out.vars, "", &[], MAX_EXPANSION_DEPTH)?;
        let Some((lhs, rhs)) = expanded.split_once(':') else {
            return Err(format!("missing ':' at line {line_no}"));
        };
        let targets = lhs.split_whitespace().collect::<Vec<_>>();
        if targets.is_empty() {
            return Err(format!("empty target at line {line_no}"));
        }
        let prerequisites = rhs
            .split_whitespace()
            .map(str::to_string)
            .collect::<Vec<_>>();
        for prerequisite in &prerequisites {
            if prerequisite == "|" {
                return Err(format!(
                    "order-only prerequisites are unsupported at line {}",
                    line_no
                ));
            }
            if !(prerequisite.starts_with("$(") && prerequisite.ends_with(')')) {
                validate_name(prerequisite, line_no)?;
            }
        }
        current_targets.clear();
        current_has_prior_recipe = false;
        for target in targets {
            validate_name(target, line_no)?;
            if out.rules.len() >= MAX_TARGETS && !out.rules.contains_key(target) {
                return Err(format!("target limit exceeded ({MAX_TARGETS})"));
            }
            let entry = out.rules.entry(target.to_string()).or_insert_with(|| {
                out.order.push(target.to_string());
                Rule {
                    prerequisites: Vec::new(),
                    recipes: Vec::new(),
                }
            });
            current_has_prior_recipe |= !entry.recipes.is_empty();
            entry.prerequisites.extend(prerequisites.iter().cloned());
            current_targets.push(target.to_string());
        }
    }
    Ok(out)
}

/// Keep one continued recipe in one shell invocation. Non-recipe continuation remains outside
/// this bounded Makefile subset, and a trailing backslash in a comment has no effect.
fn logical_make_lines(source: &str) -> Result<Vec<(usize, String)>, String> {
    let mut lines = Vec::new();
    let mut pending: Option<(usize, String)> = None;
    for (index, raw) in source.lines().enumerate() {
        let line_no = index + 1;
        if index >= MAX_LINES {
            return Err(format!("line limit exceeded ({MAX_LINES})"));
        }
        if let Some((_, content)) = &mut pending {
            content.push('\n');
            content.push_str(raw.strip_prefix('\t').unwrap_or(raw));
            if !raw.trim_end().ends_with('\\') {
                lines.push(pending.take().expect("pending recipe was checked"));
            }
            continue;
        }
        let meaningful = if raw.starts_with('\t') {
            raw
        } else {
            raw.split('#').next().unwrap_or("")
        };
        if meaningful.trim_end().ends_with('\\') {
            if !raw.starts_with('\t') {
                return Err(format!(
                    "line continuation is unsupported at line {line_no}"
                ));
            }
            pending = Some((line_no, raw.to_string()));
        } else {
            lines.push((line_no, raw.to_string()));
        }
    }
    if let Some((line_no, _)) = pending {
        return Err(format!(
            "unterminated recipe continuation at line {line_no}"
        ));
    }
    Ok(lines)
}

fn validate_name(name: &str, line: usize) -> Result<(), String> {
    if name.is_empty() || name.contains('%') || name.contains('$') {
        return Err(format!(
            "unsupported target or prerequisite '{name}' at line {line}"
        ));
    }
    Ok(())
}

fn parse_assignment(value: &str) -> Option<(String, String)> {
    let (name, value) = value.split_once('=')?;
    let name = name.trim();
    if name.is_empty()
        || !name
            .chars()
            .enumerate()
            .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
    {
        return None;
    }
    Some((name.to_string(), value.trim().to_string()))
}

impl MakeRun<'_, '_> {
    fn build(&mut self, target: &str, parent: Option<&str>) -> Result<u64, String> {
        if self.visiting.contains(target) {
            return Err(format!("circular dependency involving '{target}'"));
        }
        if self.visited.contains(target) {
            return Ok(self.mtime(target));
        }
        self.visiting.insert(target.to_string());
        let rule = self.rules.get(target).cloned();
        let exists = self.exists(target);
        if rule.is_none() {
            self.visiting.remove(target);
            if exists {
                self.visited.insert(target.to_string());
                return Ok(self.mtime(target));
            }
            return Err(format!("No rule to make target '{target}'"));
        }
        let rule = rule.unwrap();
        let prerequisites = self.expand_prerequisites(target, &rule.prerequisites)?;
        let mut prereq_mtimes = Vec::with_capacity(prerequisites.len());
        for prerequisite in &prerequisites {
            let mtime = self.build(prerequisite, Some(target))?;
            prereq_mtimes.push(mtime);
        }
        let target_mtime = self.mtime(target);
        let needs = self.phony.contains(target)
            || !exists
            || prereq_mtimes.iter().any(|mtime| *mtime > target_mtime);
        if needs {
            for recipe in &rule.recipes {
                self.recipe_count = self.recipe_count.saturating_add(1);
                let expanded = self.expand(recipe, target, parent, &prerequisites)?;
                if expanded.trim().is_empty() {
                    continue;
                }
                let (silent, ignore_error, command) = recipe_prefixes(&expanded);
                if self.dry_run || !silent {
                    wln(self.out, command);
                }
                if self.dry_run {
                    continue;
                }
                let saved_cwd = self.interp.cwd.clone();
                let mut command_out = Vec::new();
                let mut command_err = Vec::new();
                let status =
                    self.interp
                        .run_script_into(command, &mut command_out, &mut command_err);
                self.interp.cwd = saved_cwd.clone();
                self.interp.set_var("PWD", saved_cwd);
                self.out.extend_from_slice(&command_out);
                self.err.extend_from_slice(&command_err);
                if status != 0 && !ignore_error {
                    self.visiting.remove(target);
                    return Err(format!("recipe for '{target}' failed with status {status}"));
                }
            }
        }
        self.visiting.remove(target);
        self.visited.insert(target.to_string());
        Ok(self.mtime(target))
    }

    fn exists(&self, target: &str) -> bool {
        self.interp.vfs.exists(&self.interp.cwd, target)
    }

    fn mtime(&self, target: &str) -> u64 {
        self.interp
            .vfs
            .metadata(&self.interp.cwd, target, true)
            .map(|node| node.mtime)
            .unwrap_or(0)
    }

    fn expand(
        &mut self,
        recipe: &str,
        target: &str,
        _parent: Option<&str>,
        prerequisites: &[String],
    ) -> Result<String, String> {
        if !self.interp.resources.charge_cpu(recipe.len() as u64) {
            return Err("resource limit exceeded while expanding recipe".to_string());
        }
        expand_vars(
            recipe,
            &self.vars,
            target,
            prerequisites,
            MAX_EXPANSION_DEPTH,
        )
    }

    fn expand_prerequisites(
        &mut self,
        target: &str,
        raw_prerequisites: &[String],
    ) -> Result<Vec<String>, String> {
        let source_bytes = raw_prerequisites
            .iter()
            .try_fold(0_u64, |total, value| total.checked_add(value.len() as u64))
            .ok_or_else(|| "prerequisite expansion is too large".to_string())?;
        if !self.interp.resources.charge_cpu(source_bytes) {
            return Err("resource limit exceeded while expanding prerequisites".to_string());
        }
        let mut expanded = Vec::new();
        for prerequisite in raw_prerequisites {
            let value = expand_vars(prerequisite, &self.vars, target, &[], MAX_EXPANSION_DEPTH)?;
            for name in value.split_whitespace() {
                if expanded.len() >= MAX_TARGETS {
                    return Err(format!(
                        "expanded prerequisite limit exceeded ({MAX_TARGETS})"
                    ));
                }
                validate_name(name, 0)
                    .map_err(|_| format!("variable produced unsupported prerequisite '{name}'"))?;
                expanded.push(name.to_string());
            }
        }
        Ok(expanded)
    }
}

fn expand_vars(
    source: &str,
    vars: &BTreeMap<String, String>,
    target: &str,
    prerequisites: &[String],
    depth: usize,
) -> Result<String, String> {
    if depth == 0 {
        return Err("variable expansion is too recursive".to_string());
    }
    let mut output = String::with_capacity(source.len());
    let chars = source.chars().collect::<Vec<_>>();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '$' {
            output.push(chars[i]);
            ensure_expansion_size(&output)?;
            i += 1;
            continue;
        }
        let Some(next) = chars.get(i + 1).copied() else {
            output.push('$');
            break;
        };
        if next == '$' {
            output.push('$');
            ensure_expansion_size(&output)?;
            i += 2;
            continue;
        }
        if next == '@' {
            output.push_str(target);
            ensure_expansion_size(&output)?;
            i += 2;
            continue;
        }
        if next == '<' {
            output.push_str(prerequisites.first().map(String::as_str).unwrap_or(""));
            ensure_expansion_size(&output)?;
            i += 2;
            continue;
        }
        if next == '^' {
            output.push_str(&prerequisites.join(" "));
            ensure_expansion_size(&output)?;
            i += 2;
            continue;
        }
        if next != '(' {
            output.push('$');
            ensure_expansion_size(&output)?;
            i += 1;
            continue;
        }
        let mut end = i + 2;
        while end < chars.len() && chars[end] != ')' {
            end += 1;
        }
        if end == chars.len() {
            return Err("unterminated variable reference".to_string());
        }
        let name = chars[i + 2..end].iter().collect::<String>();
        if name.is_empty() || name.contains([' ', '\t', ':']) {
            return Err(format!("unsupported variable expression '$({name})'"));
        }
        let value = vars.get(&name).map(String::as_str).unwrap_or("");
        output.push_str(&expand_vars(value, vars, target, prerequisites, depth - 1)?);
        ensure_expansion_size(&output)?;
        i = end + 1;
    }
    Ok(output)
}

fn ensure_expansion_size(output: &str) -> Result<(), String> {
    if output.len() > MAX_EXPANDED_BYTES {
        Err("variable expansion is too large".to_string())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_assignments_rules_and_recipes() {
        let mut interp = Interp::new();
        let parsed = parse_makefile(
            &mut interp,
            "OUT = result.txt\n.PHONY: all\nall: $(OUT)\n\t@echo $(OUT)\n",
        )
        .unwrap();
        assert_eq!(parsed.vars["OUT"], "result.txt");
        assert!(parsed.phony.contains("all"));
        assert_eq!(parsed.rules["all"].prerequisites, vec!["result.txt"]);
    }

    #[test]
    fn expansion_supports_automatic_variables_and_dollar_escape() {
        let mut vars = BTreeMap::new();
        vars.insert("NAME".to_string(), "value".to_string());
        assert_eq!(
            expand_vars("$(NAME) $@ $< $$HOME", &vars, "out", &["in".into()], 4).unwrap(),
            "value out in $HOME"
        );
    }
}
