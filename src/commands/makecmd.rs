//! Parsing and variable expansion for the bounded, deterministic `make` command.
//!
//! This module owns only the text-level concerns: option parsing, Makefile parsing (variables,
//! rules, prerequisites, tab-indented recipes, `.PHONY`), and `$(VAR)`/automatic-variable
//! expansion. It never touches the virtual filesystem or spawns anything; the resumable process
//! image in `crate::program::make` drives execution against `System` and calls back into this
//! module for parsing. Pattern rules, includes, implicit search, order-only prerequisites, and
//! line continuations outside recipes are rejected explicitly rather than approximated.

use std::collections::{BTreeMap, BTreeSet};

use crate::syscalls::System;

pub(crate) const MAX_MAKEFILE_BYTES: usize = 1024 * 1024;
pub(crate) const MAX_LINES: usize = 20_000;
pub(crate) const MAX_TARGETS: usize = 4_096;
pub(crate) const MAX_RECIPES: usize = 20_000;
pub(crate) const MAX_EXPANSION_DEPTH: usize = 16;
pub(crate) const MAX_EXPANDED_BYTES: usize = 4 * 1024 * 1024;

/// One recipe line together with its source line number, needed for GNU-style
/// `Makefile:LINE: target` failure diagnostics.
#[derive(Clone, Debug)]
pub(crate) struct RecipeLine {
    pub(crate) line: usize,
    pub(crate) text: String,
}

#[derive(Clone, Debug)]
pub(crate) struct Rule {
    pub(crate) prerequisites: Vec<String>,
    pub(crate) recipes: Vec<RecipeLine>,
}

#[derive(Default, Debug)]
pub(crate) struct Makefile {
    pub(crate) vars: BTreeMap<String, String>,
    pub(crate) rules: BTreeMap<String, Rule>,
    pub(crate) order: Vec<String>,
    pub(crate) phony: BTreeSet<String>,
    /// Explicit `export`/`unexport` directives, keyed by variable name and keeping only the
    /// last directive seen for that name (later lines win, as in GNU make). `true` forces the
    /// variable into recipe environments; `false` forces it out even if it came from the
    /// process environment. Names never mentioned by an `export`/`unexport` line are exported
    /// only if they were already present in the process environment; see
    /// `crate::program::make` for how this combines with command-line `VAR=value` overrides,
    /// which are always exported regardless of this map.
    pub(crate) export_state: BTreeMap<String, bool>,
}

/// Parsed command-line options shared by the native `make` image.
#[derive(Default, Clone)]
pub(crate) struct Options {
    pub(crate) file: Option<String>,
    pub(crate) directory: Option<String>,
    pub(crate) dry_run: bool,
    pub(crate) silent: bool,
    pub(crate) keep_going: bool,
    /// `-j` is parsed and accepted for compatibility; recipes always run one at a time, so the
    /// requested job count has no observable effect beyond validation. See module docs on
    /// `crate::program::make` for the determinism rationale.
    pub(crate) jobs: Option<u32>,
    pub(crate) targets: Vec<String>,
    pub(crate) command_vars: BTreeMap<String, String>,
}

/// Parse `make`'s argv. Returns `None` for malformed or unrecognized options, which the caller
/// reports as `make: unsupported option or malformed argument`.
pub(crate) fn parse_args(args: &[String]) -> Option<Options> {
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
        if !end_options && (arg == "-j" || arg == "--jobs") {
            i += 1;
            let value = args.get(i)?;
            options.jobs = Some(value.parse().ok()?);
            i += 1;
            continue;
        }
        if !end_options {
            if let Some(value) = arg.strip_prefix("--jobs=") {
                options.jobs = Some(value.parse().ok()?);
                i += 1;
                continue;
            }
        }
        if !end_options && arg.starts_with("-j") {
            if let Some(value) = arg.strip_prefix("-j") {
                if value.is_empty() {
                    options.jobs = None;
                } else {
                    options.jobs = Some(value.parse().ok()?);
                }
                i += 1;
                continue;
            }
        }
        if !end_options && arg == "--dry-run" {
            options.dry_run = true;
            i += 1;
            continue;
        }
        if !end_options && arg == "--silent" {
            options.silent = true;
            i += 1;
            continue;
        }
        if !end_options && arg == "--keep-going" {
            options.keep_going = true;
            i += 1;
            continue;
        }
        if !end_options && arg.starts_with('-') && arg != "-" {
            // Cluster of single-character boolean flags, e.g. `-nk` or `-ns`.
            let rest = &arg[1..];
            if !rest.is_empty() && rest.chars().all(|c| matches!(c, 'n' | 's' | 'k')) {
                for c in rest.chars() {
                    match c {
                        'n' => options.dry_run = true,
                        's' => options.silent = true,
                        'k' => options.keep_going = true,
                        _ => unreachable!(),
                    }
                }
                i += 1;
                continue;
            }
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

pub(crate) fn recipe_prefixes(recipe: &str) -> (bool, bool, &str) {
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

pub(crate) fn parse_makefile(system: &mut impl System, source: &str) -> Result<Makefile, String> {
    let bytes = source.len();
    if !system.charge_cpu(bytes as u64) || !system.reserve_memory((bytes as u64).saturating_mul(2))
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
            recipe_count = recipe_count.saturating_add(1);
            if recipe_count > MAX_RECIPES {
                return Err(format!("recipe limit exceeded ({MAX_RECIPES})"));
            }
            let recipe_line = RecipeLine {
                line: line_no,
                text: recipe.to_string(),
            };
            for target in &current_targets {
                out.rules
                    .get_mut(target)
                    .ok_or_else(|| "internal rule state error".to_string())?
                    .recipes
                    .push(recipe_line.clone());
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
        if let Some(rest) = line.strip_prefix("export") {
            if rest.is_empty() || rest.starts_with(char::is_whitespace) {
                let rest = rest.trim_start();
                if rest.is_empty() {
                    // Bare `export` (export every variable from here on) is unsupported; this
                    // subset only tracks explicitly named variables.
                    return Err(format!("unsupported directive at line {line_no}"));
                }
                if let Some((name, value)) = parse_assignment(rest) {
                    out.vars.insert(name.clone(), value);
                    out.export_state.insert(name, true);
                } else {
                    for name in rest.split_whitespace() {
                        if !is_valid_var_name(name) {
                            return Err(format!(
                                "unsupported export target '{name}' at line {line_no}"
                            ));
                        }
                        out.export_state.insert(name.to_string(), true);
                    }
                }
                current_targets.clear();
                current_has_prior_recipe = false;
                continue;
            }
        }
        if let Some(rest) = line.strip_prefix("unexport") {
            if rest.is_empty() || rest.starts_with(char::is_whitespace) {
                let rest = rest.trim_start();
                if rest.is_empty() {
                    return Err(format!("unsupported directive at line {line_no}"));
                }
                for name in rest.split_whitespace() {
                    if !is_valid_var_name(name) {
                        return Err(format!(
                            "unsupported export target '{name}' at line {line_no}"
                        ));
                    }
                    out.export_state.insert(name.to_string(), false);
                }
                current_targets.clear();
                current_has_prior_recipe = false;
                continue;
            }
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

pub(crate) fn validate_name(name: &str, line: usize) -> Result<(), String> {
    if name.is_empty() || name.contains('%') || name.contains('$') {
        return Err(format!(
            "unsupported target or prerequisite '{name}' at line {line}"
        ));
    }
    Ok(())
}

/// Parse a `NAME=value` or `NAME:=value` assignment. This bounded subset does not distinguish
/// recursive (`=`) from simple (`:=`) expansion; both store the right-hand side immediately.
pub(crate) fn parse_assignment(value: &str) -> Option<(String, String)> {
    let (raw_name, value) = value.split_once('=')?;
    let name = raw_name.strip_suffix(':').unwrap_or(raw_name).trim();
    if !is_valid_var_name(name) {
        return None;
    }
    Some((name.to_string(), value.trim().to_string()))
}

/// A variable name make would accept: non-empty, `[A-Za-z_][A-Za-z0-9_]*`.
fn is_valid_var_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .enumerate()
            .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
}

pub(crate) fn expand_vars(
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
    use crate::interp::Interp;
    use crate::syscalls::ActiveSystem;

    #[test]
    fn parses_assignments_rules_and_recipes() {
        let mut interp = Interp::new();
        let mut system = ActiveSystem::new(&mut interp);
        let parsed = parse_makefile(
            &mut system,
            "OUT = result.txt\n.PHONY: all\nall: $(OUT)\n\t@echo $(OUT)\n",
        )
        .unwrap();
        assert_eq!(parsed.vars["OUT"], "result.txt");
        assert!(parsed.phony.contains("all"));
        assert_eq!(parsed.rules["all"].prerequisites, vec!["result.txt"]);
        assert_eq!(parsed.rules["all"].recipes[0].line, 4);
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

    #[test]
    fn parse_args_accepts_clustered_short_flags_and_jobs() {
        let options = parse_args(&["-nk".to_string(), "-j4".to_string(), "all".to_string()])
            .expect("valid options");
        assert!(options.dry_run);
        assert!(options.keep_going);
        assert_eq!(options.jobs, Some(4));
        assert_eq!(options.targets, vec!["all".to_string()]);
    }

    #[test]
    fn export_and_unexport_directives_are_recorded() {
        let mut interp = Interp::new();
        let mut system = ActiveSystem::new(&mut interp);
        let parsed = parse_makefile(
            &mut system,
            "A=1\nexport B=2\nexport C := 3\nexport A\nunexport D\nall:\n\t@true\n",
        )
        .unwrap();
        assert_eq!(parsed.vars["A"], "1");
        assert_eq!(parsed.vars["B"], "2");
        assert_eq!(parsed.vars["C"], "3");
        assert_eq!(parsed.export_state.get("A"), Some(&true));
        assert_eq!(parsed.export_state.get("B"), Some(&true));
        assert_eq!(parsed.export_state.get("C"), Some(&true));
        assert_eq!(parsed.export_state.get("D"), Some(&false));
        // A plain assignment does not export by itself.
        assert!(!parsed.export_state.contains_key("nonexistent"));
    }

    #[test]
    fn bare_export_directive_is_unsupported() {
        let mut interp = Interp::new();
        let mut system = ActiveSystem::new(&mut interp);
        let error = parse_makefile(&mut system, "export\nall:\n\t@true\n").unwrap_err();
        assert!(error.contains("unsupported"), "{error}");
    }
}
