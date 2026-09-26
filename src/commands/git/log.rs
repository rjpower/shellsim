//! `git log` and `git show`: choosing commits and printing them.
//!
//! Revision ranges, `--pretty` and `--format` templates, date and content filters, the ASCII
//! graph, and the commit header and diff that both commands render all live here.

use std::collections::BTreeSet;

use crate::commands::Io;
use crate::syscalls::System;

use super::compare::{self, Format, Options, RightSide};
use super::diff;
use super::repo::{self, Commit};
use super::{fatal, repo_error, usage, Arg, Flags};

/// How one commit's header is rendered.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Pretty {
    Medium,
    OneLine,
    /// `--oneline`, which abbreviates the commit id.
    AbbreviatedOneLine,
    Short,
    Full,
    Fuller,
    /// A `--format` string. `terminated` distinguishes `--format=` from `--pretty=format:`,
    /// which separates entries instead of terminating them.
    Custom {
        format: String,
        terminated: bool,
    },
}

/// Parse a `--pretty=` or `--format=` value, rejecting spellings the subset cannot render.
fn parse_pretty(value: &str, terminated: bool) -> Option<Pretty> {
    Some(match value {
        "oneline" => Pretty::OneLine,
        "short" => Pretty::Short,
        "medium" | "" => Pretty::Medium,
        "full" => Pretty::Full,
        "fuller" => Pretty::Fuller,
        other => {
            let format = other
                .strip_prefix("format:")
                .map(|format| (format, false))
                .or_else(|| other.strip_prefix("tformat:").map(|format| (format, true)));
            match format {
                Some((format, explicit)) => Pretty::Custom {
                    format: format.to_string(),
                    terminated: explicit || terminated,
                },
                // A bare word that is not a known preset is a format only when it uses a
                // placeholder; anything else would silently print itself.
                None if other.contains('%') => Pretty::Custom {
                    format: other.to_string(),
                    terminated,
                },
                None => return None,
            }
        }
    })
}

/// Names of the references that point directly at a commit, for `%d` and `--decorate`.
fn decorations(system: &mut dyn System, root: &str, id: &str) -> String {
    let mut names = Vec::new();
    let head_branch = repo::current_branch(system, root);
    for branch in repo::branch_names(system, root) {
        if repo::read_reference(system, root, &format!("refs/heads/{branch}")).as_deref()
            != Some(id)
        {
            continue;
        }
        if head_branch.as_deref() == Some(branch.as_str()) {
            names.insert(0, format!("HEAD -> {branch}"));
        } else {
            names.push(branch);
        }
    }
    for tag in repo::reference_names(system, root, "tags") {
        if repo::read_reference(system, root, &format!("refs/tags/{tag}")).as_deref() == Some(id) {
            names.push(format!("tag: {tag}"));
        }
    }
    if head_branch.is_none() && repo::head_commit(system, root).as_deref() == Some(id) {
        names.insert(0, "HEAD".to_string());
    }
    if names.is_empty() {
        String::new()
    } else {
        format!(" ({})", names.join(", "))
    }
}

fn render_commit_header(
    id: &str,
    commit: &Commit,
    pretty: &Pretty,
    decoration: &str,
    now: i64,
) -> String {
    let identity = format!("{} <{}>", commit.author_name, commit.author_email);
    // A commit with more than one parent names them, which is how a merge is recognised.
    let merge = if commit.parents.len() > 1 {
        let parents: Vec<&str> = commit.parents.iter().map(|id| repo::short(id)).collect();
        format!("Merge: {}\n", parents.join(" "))
    } else {
        String::new()
    };
    match pretty {
        Pretty::OneLine => format!("{id}{decoration} {}\n", commit.subject()),
        Pretty::AbbreviatedOneLine => {
            format!("{}{decoration} {}\n", repo::short(id), commit.subject())
        }
        Pretty::Short => format!(
            "commit {id}{decoration}\n{merge}Author: {identity}\n\n{}",
            indent(commit.subject())
        ),
        Pretty::Medium => format!(
            "commit {id}{decoration}\n{merge}Author: {identity}\nDate:   {}\n\n{}",
            repo::format_date(commit.timestamp),
            indent(&commit.message)
        ),
        Pretty::Full => format!(
            "commit {id}{decoration}\n{merge}Author: {identity}\nCommit: {identity}\n\n{}",
            indent(&commit.message)
        ),
        Pretty::Fuller => format!(
            "commit {id}{decoration}\n{merge}Author:     {identity}\nAuthorDate: {}\nCommit:     {identity}\nCommitDate: {}\n\n{}",
            repo::format_date(commit.timestamp),
            repo::format_date(commit.timestamp),
            indent(&commit.message)
        ),
        Pretty::Custom { format, .. } => expand_format(format, id, commit, decoration, now),
    }
}

fn indent(message: &str) -> String {
    message
        .lines()
        .map(|line| format!("    {line}\n"))
        .collect()
}

/// Render a timestamp in one of the formats Git's placeholders use.
pub(crate) fn stamp(timestamp: i64, format: &str) -> String {
    crate::commands::proc::format_date(i128::from(timestamp) * 1_000_000_000, format)
        .unwrap_or_else(|_| timestamp.to_string())
}

/// Expand the `--format` placeholders the subset supports.
///
/// Returns `None` for an unsupported placeholder so the caller can reject the format rather than
/// printing it literally.
fn expand_format_checked(
    format: &str,
    id: &str,
    commit: &Commit,
    decoration: &str,
    now: i64,
) -> Option<String> {
    let mut out = String::new();
    let mut characters = format.chars().peekable();
    while let Some(character) = characters.next() {
        if character != '%' {
            out.push(character);
            continue;
        }
        let first = characters.next()?;
        let name = match first {
            'a' | 'c' => format!("{first}{}", characters.next()?),
            // Colour placeholders expand to nothing: this subset never writes escape sequences.
            'C' => {
                if characters.peek() == Some(&'(') {
                    for character in characters.by_ref() {
                        if character == ')' {
                            break;
                        }
                    }
                } else {
                    while characters
                        .peek()
                        .is_some_and(|character| character.is_ascii_alphabetic())
                    {
                        characters.next();
                    }
                }
                continue;
            }
            'x' => {
                // `%xNN` inserts one byte written in hexadecimal.
                let digits: String = characters.by_ref().take(2).collect();
                let byte = u8::from_str_radix(&digits, 16).ok()?;
                out.push(byte as char);
                continue;
            }
            other => other.to_string(),
        };
        // Author and committer identities are the same in this subset.
        let text = match name.as_str() {
            "H" => id.to_string(),
            "h" => repo::short(id).to_string(),
            "s" => commit.subject().to_string(),
            "f" => commit
                .subject()
                .chars()
                .map(|character| {
                    if character.is_alphanumeric() {
                        character
                    } else {
                        '-'
                    }
                })
                .collect(),
            // The body drops the subject and its blank separator, and both keep a final newline.
            "b" => format!(
                "{}\n",
                commit
                    .message
                    .split_once("\n\n")
                    .map_or("", |rest| rest.1)
                    .trim_end()
            ),
            "B" => format!("{}\n", commit.message.trim_end()),
            "P" => commit.parents.join(" "),
            "p" => commit
                .parents
                .iter()
                .map(|parent| repo::short(parent).to_string())
                .collect::<Vec<_>>()
                .join(" "),
            "d" => decoration.to_string(),
            "D" => decoration
                .trim_start()
                .trim_start_matches('(')
                .trim_end_matches(')')
                .to_string(),
            "an" | "cn" | "aN" | "cN" => commit.author_name.clone(),
            "ae" | "ce" | "aE" | "cE" => commit.author_email.clone(),
            "ad" | "cd" => repo::format_date(commit.timestamp),
            "at" | "ct" => commit.timestamp.to_string(),
            "ai" | "ci" => stamp(commit.timestamp, "%Y-%m-%d %H:%M:%S +0000"),
            "aI" | "cI" => stamp(commit.timestamp, "%Y-%m-%dT%H:%M:%S+00:00"),
            "as" | "cs" => stamp(commit.timestamp, "%Y-%m-%d"),
            "aD" | "cD" => stamp(commit.timestamp, "%a, %-d %b %Y %H:%M:%S +0000"),
            "ar" | "cr" => relative_date(commit.timestamp, now),
            // Notes, boundary marks, and encodings have no counterpart in this subset.
            "N" | "m" | "e" => String::new(),
            "n" => "\n".to_string(),
            "%" => "%".to_string(),
            _ => return None,
        };
        out.push_str(&text);
    }
    Some(out)
}

fn expand_format(format: &str, id: &str, commit: &Commit, decoration: &str, now: i64) -> String {
    expand_format_checked(format, id, commit, decoration, now).unwrap_or_default()
}

/// Whether every placeholder in `format` is one this subset can render.
fn format_is_supported(format: &str) -> bool {
    expand_format_checked(format, &"0".repeat(40), &Commit::default(), "", 0).is_some()
}

/// Render an age the way `%ar` does.
fn relative_date(timestamp: i64, now: i64) -> String {
    let seconds = now.saturating_sub(timestamp).max(0);
    for (unit, size) in [
        ("year", 31_556_952),
        ("month", 2_629_746),
        ("week", 604_800),
        ("day", 86_400),
        ("hour", 3_600),
        ("minute", 60),
    ] {
        let count = seconds / size;
        if count > 0 {
            let plural = if count == 1 { "" } else { "s" };
            return format!("{count} {unit}{plural} ago");
        }
    }
    format!("{seconds} seconds ago")
}

/// Split `a..b` or `a...b` into its endpoints.
///
/// A path such as `../src` is never a range, so arguments that start with a path component are
/// left alone.
pub(crate) fn split_range(revision: &str) -> Option<(&str, &str, bool)> {
    if revision.starts_with('.') || revision.starts_with('/') {
        return None;
    }
    if let Some((left, right)) = revision.split_once("...") {
        return Some((left, right, true));
    }
    revision
        .split_once("..")
        .map(|(left, right)| (left, right, false))
}

/// One revision argument resolved into the commits it includes and the commits it excludes.
struct Selection {
    included: Vec<String>,
    excluded: BTreeSet<String>,
}

/// Resolve one revision argument, which may be a plain revision or an `a..b` range.
fn select_revision(system: &mut dyn System, root: &str, revision: &str) -> Option<Selection> {
    // `^rev` excludes everything reachable from `rev` and includes nothing.
    if let Some(excluded) = revision.strip_prefix('^') {
        let commit = repo::resolve_revision(system, root, excluded)?;
        return Some(Selection {
            included: Vec::new(),
            excluded: repo::ancestors(system, root, &commit),
        });
    }
    let Some((left, right, merge_base)) = split_range(revision) else {
        return Some(Selection {
            included: vec![repo::resolve_revision(system, root, revision)?],
            excluded: BTreeSet::new(),
        });
    };
    let right = if right.is_empty() { "HEAD" } else { right };
    let left = if left.is_empty() { "HEAD" } else { left };
    let (left_commit, right_commit) = (
        repo::resolve_revision(system, root, left)?,
        repo::resolve_revision(system, root, right)?,
    );
    let excluded = if merge_base {
        let base = repo::merge_base(system, root, &left_commit, &right_commit)?;
        repo::ancestors(system, root, &base)
    } else {
        repo::ancestors(system, root, &left_commit)
    };
    Some(Selection {
        included: vec![right_commit],
        excluded,
    })
}

/// The commits listed by a set of revision arguments, newest first.
pub(crate) fn history_for(
    system: &mut dyn System,
    root: &str,
    revisions: &[String],
    first_parent: bool,
) -> Option<Vec<(String, Commit)>> {
    let mut included = Vec::new();
    let mut excluded = BTreeSet::new();
    for revision in revisions {
        let selection = select_revision(system, root, revision)?;
        included.extend(selection.included);
        excluded.extend(selection.excluded);
    }
    let listed = if first_parent {
        included
            .first()
            .map(|start| repo::first_parent_history(system, root, start, 10_000))
            .unwrap_or_default()
    } else {
        repo::reachable_history(system, root, &included, 10_000)
    };
    Some(
        listed
            .into_iter()
            .filter(|(id, _)| !excluded.contains(id))
            .collect(),
    )
}

/// Every branch and tag tip, for `--all`.
pub(crate) fn all_reference_tips(system: &mut dyn System, root: &str) -> Vec<String> {
    let mut tips = Vec::new();
    for (kind, names) in [
        ("heads", repo::branch_names(system, root)),
        ("tags", repo::reference_names(system, root, "tags")),
    ] {
        for name in names {
            if let Some(commit) = repo::read_reference(system, root, &format!("refs/{kind}/{name}"))
            {
                tips.push(commit);
            }
        }
    }
    tips
}

pub(crate) fn git_log(system: &mut dyn System, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(system) else {
        return repo_error(io);
    };
    let mut pretty = Pretty::Medium;
    let mut limit = 10_000_usize;
    let mut skip = 0_usize;
    let mut revisions: Vec<String> = Vec::new();
    let mut all_references = false;
    let mut first_parent = false;
    let mut reverse = false;
    let mut decorate = false;
    let mut no_merges = false;
    let mut merges_only = false;
    let mut abbreviate = false;
    let mut author_filter: Option<String> = None;
    let mut message_filter: Option<String> = None;
    let mut since: Option<i64> = None;
    let mut until: Option<i64> = None;
    let mut ignore_case = false;
    // `-S` counts occurrences of a literal string; `-G` matches the changed lines as a regex.
    let mut pickaxe: Option<String> = None;
    let mut changed_lines: Option<String> = None;
    let mut patch = None;
    let mut stat = None;
    let mut graph: Option<Rail> = None;
    let mut paths: Vec<String> = Vec::new();
    let cwd = system.cwd().to_string();
    let mut flags = Flags::new(args).valued("nSG");
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) if flags.separated() => {
                paths.push(super::pathspec(&cwd, &root, &value));
                continue;
            }
            Arg::Operand(value) => {
                if select_revision(system, &root, &value).is_some() {
                    revisions.push(value);
                    continue;
                }
                if !super::names_a_path(system, &root, &value) {
                    return super::ambiguous_argument(io, &value);
                }
                paths.push(super::pathspec(&cwd, &root, &value));
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "--oneline" => pretty = Pretty::AbbreviatedOneLine,
            "--reverse" => reverse = true,
            // Decoration is either on or off here; `full` and `short` render the same names.
            "--decorate" => decorate = attached.as_deref() != Some("no"),
            "--no-decorate" | "--no-color" | "--color" => {}
            "--no-merges" => no_merges = true,
            "--merges" => merges_only = true,
            "--abbrev-commit" => abbreviate = true,
            "--first-parent" => first_parent = true,
            "--all" => all_references = true,
            "-n" | "--max-count" => {
                let Some(value) = flags.value(attached).and_then(|value| value.parse().ok()) else {
                    return usage(io, "log count must be a non-negative integer");
                };
                limit = value;
            }
            "--skip" => {
                let Some(value) = flags.value(attached).and_then(|value| value.parse().ok()) else {
                    return usage(io, "--skip requires a non-negative integer");
                };
                skip = value;
            }
            "-i" | "--regexp-ignore-case" => ignore_case = true,
            "--since" | "--after" | "--until" | "--before" => {
                let Some(value) = flags.value(attached) else {
                    return usage(io, &format!("{name} requires a date"));
                };
                let Some(seconds) = parse_date(&value, repo::now_seconds(system)) else {
                    return usage(io, &format!("unsupported date: {value}"));
                };
                if matches!(name.as_str(), "--since" | "--after") {
                    since = Some(seconds);
                } else {
                    until = Some(seconds);
                }
            }
            "--grep" => {
                let Some(value) = flags.value(attached) else {
                    return usage(io, "--grep requires a pattern");
                };
                message_filter = Some(value);
            }
            "--author" => {
                let Some(value) = flags.value(attached) else {
                    return usage(io, "--author requires a pattern");
                };
                author_filter = Some(value);
            }
            "-S" | "-G" => {
                let Some(value) = flags.value(attached) else {
                    return usage(io, &format!("{name} requires a string"));
                };
                if name == "-S" {
                    pickaxe = Some(value);
                } else {
                    changed_lines = Some(value);
                }
            }
            "--graph" => graph = Some(Rail::default()),
            "--no-graph" => graph = None,
            "--stat" => stat = Some(Format::Stat),
            "--name-only" => stat = Some(Format::NameOnly),
            "--name-status" => stat = Some(Format::NameStatus),
            "-p" | "-u" | "--patch" => patch = Some(Format::Patch),
            "--pretty" | "--format" => {
                let terminated = name == "--format";
                let Some(parsed) = parse_pretty(attached.as_deref().unwrap_or(""), terminated)
                    .filter(|parsed| match parsed {
                        Pretty::Custom { format, .. } => format_is_supported(format),
                        _ => true,
                    })
                else {
                    return fatal(io, &format!("unsupported log format: {name}"));
                };
                pretty = parsed;
            }
            // `git log -5` is the count written on its own.
            _ if super::plumbing::count_option(&name).is_some() => {
                limit = super::plumbing::count_option(&name).unwrap_or(usize::MAX);
            }
            _ => return usage(io, &format!("unsupported log option: {name}")),
        }
    }
    if all_references {
        revisions.extend(all_reference_tips(system, &root));
    }
    if revisions.is_empty() {
        revisions.push("HEAD".to_string());
    }
    let Some(mut history) = history_for(system, &root, &revisions, first_parent) else {
        if repo::head_commit(system, &root).is_none() {
            let branch = repo::current_branch(system, &root).unwrap_or_else(|| "HEAD".to_string());
            io.print_err(&format!(
                "fatal: your current branch '{branch}' does not have any commits yet\n"
            ));
            return 128;
        }
        return super::ambiguous_argument(io, &revisions.join(" "));
    };
    if no_merges {
        history.retain(|(_, commit)| commit.parents.len() < 2);
    }
    if merges_only {
        history.retain(|(_, commit)| commit.parents.len() > 1);
    }
    for (pattern, over_author) in [(&author_filter, true), (&message_filter, false)] {
        let Some(pattern) = pattern else { continue };
        let Ok(regex) = regex::RegexBuilder::new(pattern)
            .case_insensitive(ignore_case)
            .build()
        else {
            io.print_err(&format!("fatal: invalid pattern: {pattern}\n"));
            return 128;
        };
        history.retain(|(_, commit)| {
            if over_author {
                regex.is_match(&commit.author_name) || regex.is_match(&commit.author_email)
            } else {
                regex.is_match(&commit.message)
            }
        });
    }
    if let Some(seconds) = since {
        history.retain(|(_, commit)| commit.timestamp > seconds);
    }
    if let Some(seconds) = until {
        history.retain(|(_, commit)| commit.timestamp <= seconds);
    }
    if let Some(needle) = &pickaxe {
        history.retain(|(id, commit)| changes_occurrence_count(system, &root, id, commit, needle));
    }
    if let Some(pattern) = &changed_lines {
        let Ok(regex) = regex::RegexBuilder::new(pattern)
            .case_insensitive(ignore_case)
            .build()
        else {
            io.print_err(&format!("fatal: invalid pattern: {pattern}\n"));
            return 128;
        };
        history.retain(|(id, commit)| matches_changed_lines(system, &root, id, commit, &regex));
    }
    if !paths.is_empty() {
        history.retain(|(id, commit)| commit_touches(system, &root, id, commit, &paths));
    }
    if skip < history.len() {
        history.drain(..skip);
    } else {
        history.clear();
    }
    history.truncate(limit);
    if reverse {
        history.reverse();
    }
    let now = repo::now_seconds(system);
    let outer = io;
    // Multi-line formats are separated by a blank line; one-line formats are not.
    let separated = matches!(
        pretty,
        Pretty::Medium | Pretty::Short | Pretty::Full | Pretty::Fuller
    ) || matches!(
        pretty,
        Pretty::Custom {
            terminated: false,
            ..
        }
    );
    for (position, (id, commit)) in history.iter().enumerate() {
        // One commit's output is built up on its own so that `--graph` can prefix every line.
        let mut block = Vec::new();
        let io = &mut Io {
            stdin: Vec::new(),
            out: &mut block,
            err: outer.err,
        };
        let head = usize::from(separated && position != 0);
        if head == 1 {
            io.out.push(b'\n');
        }
        // `%d` and `%D` always expand, so decorations are computed whenever a format may use them.
        let decoration = if decorate || matches!(pretty, Pretty::Custom { .. }) {
            decorations(system, &root, id)
        } else {
            String::new()
        };
        let displayed = if abbreviate {
            repo::short(id)
        } else {
            id.as_str()
        };
        io.print(&render_commit_header(
            displayed,
            commit,
            &pretty,
            &decoration,
            now,
        ));
        if matches!(
            pretty,
            Pretty::Custom {
                terminated: true,
                ..
            }
        ) {
            io.out.push(b'\n');
        }
        for format in [stat, patch].into_iter().flatten() {
            if diff_needs_spacing(&pretty) {
                io.out.push(b'\n');
            }
            emit_commit_diff(system, &root, id, commit, format, &paths, io);
        }
        match graph.as_mut() {
            Some(rail) => {
                let rung = rail.advance(id, &commit.parents);
                draw_on_rail(&block, &rung, head, outer);
            }
            None => outer.out.extend_from_slice(&block),
        }
    }
    0
}

/// The rail `git log --graph` draws down the left of the output.
///
/// One column per commit whose line has not been drawn yet. A commit is marked in its own column
/// and its parents take that column's place, which is what makes a branch fan out and a merge fold
/// back in. The drawing is simpler than Git's: connectors always occupy a line of their own.
#[derive(Default)]
struct Rail {
    columns: Vec<String>,
}

/// Where one commit's lines sit on the rail.
struct Rung {
    /// The prefix for the commit's first line.
    first: String,
    /// The prefix for its remaining lines, and for the blank line between commits.
    rest: String,
    /// Connector lines drawn after the commit, each already terminated.
    connectors: String,
}

impl Rail {
    /// Draw `id` and move the rail on to its parents.
    fn advance(&mut self, id: &str, parents: &[String]) -> Rung {
        let column = match self.columns.iter().position(|open| open == id) {
            Some(column) => column,
            None => {
                self.columns.push(id.to_string());
                self.columns.len() - 1
            }
        };
        let width = self.columns.len();
        let mut first: String = (0..width)
            .map(|at| if at == column { "* " } else { "| " })
            .collect();
        // A merge widens the rail, and Git pads the commit line to the width that follows.
        let extra = parents.len().saturating_sub(1);
        first.push_str(&"  ".repeat(extra));
        let rest: String = "| ".repeat(width);
        let mut connectors = String::new();
        self.columns
            .splice(column..=column, parents.iter().cloned());
        if extra > 0 {
            connectors.push_str(&"| ".repeat(column));
            connectors.push_str("|\\\n");
        }
        // Two columns waiting for the same commit fold into the leftmost of them.
        while let Some((at, keep)) = self.duplicate() {
            self.columns.remove(at);
            connectors.push_str(&"| ".repeat(keep));
            connectors.push_str("|/\n");
        }
        Rung {
            first,
            rest,
            connectors,
        }
    }

    /// The first column that repeats an earlier one, as `(duplicate, original)`.
    fn duplicate(&self) -> Option<(usize, usize)> {
        for (at, open) in self.columns.iter().enumerate() {
            if let Some(first) = self.columns[..at].iter().position(|other| other == open) {
                return Some((at, first));
            }
        }
        None
    }
}

/// Prefix every line of one commit's output with its place on the rail.
///
/// `head` is the index of the line the commit itself starts on; anything before it is the blank
/// line that separates two commits, which rides the rail like a continuation line.
fn draw_on_rail(block: &[u8], rung: &Rung, head: usize, io: &mut Io) {
    let lines: Vec<&[u8]> = block.split(|byte| *byte == b'\n').collect();
    // `split` yields a trailing empty piece for the final newline, which is not a line.
    let lines = match lines.split_last() {
        Some((&[], rest)) => rest,
        _ => &lines[..],
    };
    for (position, line) in lines.iter().enumerate() {
        let prefix = if position == head {
            &rung.first
        } else {
            &rung.rest
        };
        io.print(prefix);
        io.out.extend_from_slice(line);
        io.out.push(b'\n');
    }
    io.print(&rung.connectors);
}

fn commit_touches(
    system: &mut dyn System,
    root: &str,
    id: &str,
    commit: &Commit,
    paths: &[String],
) -> bool {
    let tree = repo::commit_tree(system, root, id).unwrap_or_default();
    let parent = commit
        .parents
        .first()
        .and_then(|parent| repo::commit_tree(system, root, parent))
        .unwrap_or_default();
    let mut names: BTreeSet<String> = tree.keys().cloned().collect();
    names.extend(parent.keys().cloned());
    names
        .iter()
        .any(|path| tree.get(path) != parent.get(path) && compare::selected(paths, path))
}

/// The author date for a new commit, which `GIT_AUTHOR_DATE` may set as it does in Git.
pub(crate) fn author_date(system: &mut dyn System) -> i64 {
    let now = repo::now_seconds(system);
    system
        .environment()
        .get("GIT_AUTHOR_DATE")
        .and_then(|value| parse_date(value, now))
        .unwrap_or(now)
}

/// Parse the date forms `--since` and `--until` accept.
///
/// Absolute `YYYY-MM-DD[ HH:MM:SS]`, a bare epoch second count, and Git's relative `N units ago`
/// are understood; anything else is rejected rather than guessed at.
fn parse_date(value: &str, now: i64) -> Option<i64> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<i64>() {
        return Some(seconds);
    }
    if let Some(seconds) = parse_relative_date(value, now) {
        return Some(seconds);
    }
    let (date, time) = match value.split_once(['T', ' ']) {
        Some((date, time)) => (date, Some(time)),
        None => (value, None),
    };
    let mut parts = date.split('-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: i64 = parts.next()?.parse().ok()?;
    let day: i64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let mut seconds = days_from_civil(year, month, day) * 86_400;
    if let Some(time) = time {
        let mut fields = time.trim_end_matches('Z').split(':');
        let hours: i64 = fields.next()?.parse().ok()?;
        let minutes: i64 = fields.next().unwrap_or("0").parse().ok()?;
        let taken: i64 = fields.next().unwrap_or("0").parse().ok()?;
        seconds += hours * 3_600 + minutes * 60 + taken;
    }
    Some(seconds)
}

fn parse_relative_date(value: &str, now: i64) -> Option<i64> {
    let mut words = value.trim_end_matches(" ago").split_whitespace();
    let count: i64 = words.next()?.parse().ok()?;
    let unit = words.next()?.trim_end_matches('s');
    if words.next().is_some() {
        return None;
    }
    let size = match unit {
        "second" => 1,
        "minute" => 60,
        "hour" => 3_600,
        "day" => 86_400,
        "week" => 604_800,
        "month" => 2_629_746,
        "year" => 31_556_952,
        _ => return None,
    };
    Some(now - count * size)
}

/// Days since the Unix epoch for a civil date, by Howard Hinnant's algorithm.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The blob pair a commit changed for each path, against its first parent.
fn changed_blobs(
    system: &mut dyn System,
    root: &str,
    id: &str,
    commit: &Commit,
) -> Vec<(Option<String>, Option<String>)> {
    let tree = repo::commit_tree(system, root, id).unwrap_or_default();
    let parent = commit
        .parents
        .first()
        .and_then(|parent| repo::commit_tree(system, root, parent))
        .unwrap_or_default();
    let mut names: BTreeSet<String> = tree.keys().cloned().collect();
    names.extend(parent.keys().cloned());
    names
        .into_iter()
        .filter(|path| tree.get(path) != parent.get(path))
        .map(|path| {
            let hash = |tree: &repo::Tree| tree.get(&path).map(|entry| entry.hash.clone());
            (hash(&parent), hash(&tree))
        })
        .collect()
}

/// Whether a commit changed how many times `needle` appears, which is what `git log -S` selects.
fn changes_occurrence_count(
    system: &mut dyn System,
    root: &str,
    id: &str,
    commit: &Commit,
    needle: &str,
) -> bool {
    if needle.is_empty() {
        return false;
    }
    let blobs = changed_blobs(system, root, id, commit);
    blobs.into_iter().any(|(before, after)| {
        occurrence_count(system, root, before, needle)
            != occurrence_count(system, root, after, needle)
    })
}

fn occurrence_count(
    system: &mut dyn System,
    root: &str,
    hash: Option<String>,
    needle: &str,
) -> usize {
    let Some(data) = hash.and_then(|hash| repo::read_blob(system, root, &hash)) else {
        return 0;
    };
    String::from_utf8_lossy(&data).matches(needle).count()
}

/// Whether any line a commit added or removed matches `regex`, which is what `git log -G` selects.
fn matches_changed_lines(
    system: &mut dyn System,
    root: &str,
    id: &str,
    commit: &Commit,
    regex: &regex::Regex,
) -> bool {
    let blobs = changed_blobs(system, root, id, commit);
    blobs.into_iter().any(|(before, after)| {
        let before = read_blob_or_default(system, root, before);
        let after = read_blob_or_default(system, root, after);
        diff::edit_script(&diff::split_lines(&before), &diff::split_lines(&after))
            .iter()
            .any(|edit| !matches!(edit.op, diff::Op::Keep) && regex.is_match(edit.text))
    })
}

fn read_blob_or_default(system: &mut dyn System, root: &str, hash: Option<String>) -> Vec<u8> {
    hash.and_then(|hash| repo::read_blob(system, root, &hash))
        .unwrap_or_default()
}

/// Render one commit's difference against its first parent.
fn emit_commit_diff(
    system: &mut dyn System,
    root: &str,
    id: &str,
    commit: &Commit,
    format: Format,
    paths: &[String],
    io: &mut Io,
) {
    let tree = repo::commit_tree(system, root, id).unwrap_or_default();
    let parent = commit
        .parents
        .first()
        .and_then(|parent| repo::commit_tree(system, root, parent))
        .unwrap_or_default();
    let options = Options {
        format,
        right: RightSide::Stored,
        paths: paths.to_vec(),
        ..Options::default()
    };
    compare::emit(system, root, &parent, &tree, &options, io);
}

pub(crate) fn git_show(system: &mut dyn System, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(system) else {
        return repo_error(io);
    };
    let mut format = Some(Format::Patch);
    let mut pretty = Pretty::Medium;
    let mut revisions: Vec<String> = Vec::new();
    for argument in Flags::new(args) {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                revisions.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "-s" | "--no-patch" => format = None,
            "--stat" => format = Some(Format::Stat),
            "--name-only" => format = Some(Format::NameOnly),
            "--name-status" => format = Some(Format::NameStatus),
            "--oneline" => pretty = Pretty::AbbreviatedOneLine,
            "--no-color" | "--abbrev-commit" => {}
            "--pretty" | "--format" => {
                let Some(parsed) =
                    parse_pretty(attached.as_deref().unwrap_or(""), name == "--format").filter(
                        |parsed| match parsed {
                            Pretty::Custom { format, .. } => format_is_supported(format),
                            _ => true,
                        },
                    )
                else {
                    return usage(io, &format!("unsupported show format: {name}"));
                };
                pretty = parsed;
            }
            _ => return usage(io, &format!("unsupported show option: {name}")),
        }
    }
    if revisions.is_empty() {
        revisions.push("HEAD".to_string());
    }
    for revision in &revisions {
        // `REVISION:PATH` prints one file's contents at that revision.
        if revision.contains(':') {
            let prefix = revision.split(':').next().unwrap_or_default().to_string();
            let Some((tree, path)) = repo::tree_and_path(system, &root, revision) else {
                return super::ambiguous_argument(io, revision);
            };
            let Some(data) = tree
                .get(&path)
                .and_then(|entry| repo::read_blob(system, &root, &entry.hash))
            else {
                io.print_err(&format!(
                    "fatal: path '{path}' does not exist in '{prefix}'\n"
                ));
                return 128;
            };
            io.out.extend_from_slice(&data);
            continue;
        }
        let Some(id) = repo::resolve_revision(system, &root, revision) else {
            return super::ambiguous_argument(io, revision);
        };
        let Some(commit) = repo::load_commit(system, &root, &id) else {
            return super::ambiguous_argument(io, revision);
        };
        if let Some(annotation) = read_annotation(system, &root, revision) {
            io.print(&format!(
                "tag {revision}\nTagger: {} <{}>\nDate:   {}\n\n{}\n\n",
                annotation.author_name,
                annotation.author_email,
                repo::format_date(annotation.timestamp),
                annotation.message
            ));
        }
        let decoration = if matches!(pretty, Pretty::Custom { .. }) {
            decorations(system, &root, &id)
        } else {
            String::new()
        };
        let now = repo::now_seconds(system);
        io.print(&render_commit_header(
            &id,
            &commit,
            &pretty,
            &decoration,
            now,
        ));
        if matches!(
            pretty,
            Pretty::Custom {
                terminated: true,
                ..
            }
        ) {
            io.out.push(b'\n');
        }
        if let Some(format) = format {
            if diff_needs_spacing(&pretty) {
                io.out.push(b'\n');
            }
            emit_commit_diff(system, &root, &id, &commit, format, &[], io);
        }
    }
    0
}

/// Whether a diff that follows the header needs a blank line before it.
///
/// Git separates the message from the diff, but a one-line header has no message to separate.
fn diff_needs_spacing(pretty: &Pretty) -> bool {
    !matches!(pretty, Pretty::OneLine | Pretty::AbbreviatedOneLine)
}

/// Read an annotated tag's message record, if `name` names one.
fn read_annotation(system: &mut dyn System, root: &str, name: &str) -> Option<Commit> {
    repo::read_annotation(system, root, name)
}
