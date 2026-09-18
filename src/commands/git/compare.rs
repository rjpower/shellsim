//! Tree comparison and the `git diff` porcelain.
//!
//! One routine renders the difference between two trees in every output format the subset
//! supports, so `git diff`, `git show`, and `git log -p` agree on formatting. The right-hand side
//! may come from a stored tree or from the working tree, which is what distinguishes
//! `git diff --cached` from a plain `git diff`.

use std::collections::BTreeSet;

use crate::commands::{CommandContext, Io};

use super::diff::{self, DEFAULT_CONTEXT};
use super::repo::{self, Tree};
use super::usage;

/// How a tree comparison is presented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    Patch,
    NameOnly,
    NameStatus,
    Stat,
    NumStat,
    ShortStat,
    Check,
}

/// Where the right-hand contents of a comparison come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RightSide {
    /// Blobs named by the right-hand tree.
    Stored,
    /// Files as they currently are in the working tree.
    WorkingTree,
}

/// Everything that shapes one comparison's output.
#[derive(Clone, Debug)]
pub(crate) struct Options {
    pub format: Format,
    pub context: usize,
    pub right: RightSide,
    /// Repository-relative path prefixes to restrict the comparison to.
    pub paths: Vec<String>,
    /// Report only whether anything differs, printing nothing.
    pub quiet: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            format: Format::Patch,
            context: DEFAULT_CONTEXT,
            right: RightSide::Stored,
            paths: Vec::new(),
            quiet: false,
        }
    }
}

/// Whether a repository-relative path is covered by the given pathspec prefixes.
pub(crate) fn selected(paths: &[String], path: &str) -> bool {
    paths.is_empty()
        || paths.iter().any(|prefix| {
            prefix.is_empty() || path == prefix || path.starts_with(&format!("{prefix}/"))
        })
}

fn content(
    ctx: &CommandContext<'_>,
    root: &str,
    tree: &Tree,
    path: &str,
    side: RightSide,
) -> Option<Vec<u8>> {
    match side {
        RightSide::Stored => tree
            .get(path)
            .and_then(|hash| repo::read_blob(ctx, root, hash)),
        RightSide::WorkingTree => {
            tree.get(path)?;
            repo::read_work_file(ctx, root, path)
        }
    }
}

/// Render the difference between two trees. Returns true when a `--check` violation was found.
pub(crate) fn emit(
    ctx: &mut CommandContext<'_>,
    root: &str,
    old: &Tree,
    new: &Tree,
    options: &Options,
    io: &mut Io,
) -> bool {
    let mut names = BTreeSet::new();
    names.extend(old.keys().cloned());
    names.extend(new.keys().cloned());
    let changed: Vec<String> = names
        .into_iter()
        .filter(|path| selected(&options.paths, path) && old.get(path) != new.get(path))
        .collect();
    if changed.is_empty() {
        return false;
    }
    if options.format == Format::NameOnly {
        for path in &changed {
            io.out.extend_from_slice(format!("{path}\n").as_bytes());
        }
        return false;
    }
    if options.format == Format::NameStatus {
        for path in &changed {
            let status = match (old.contains_key(path), new.contains_key(path)) {
                (false, true) => 'A',
                (true, false) => 'D',
                _ => 'M',
            };
            io.out
                .extend_from_slice(format!("{status}\t{path}\n").as_bytes());
        }
        return false;
    }

    let mut check_failed = false;
    let mut stats: Vec<(String, usize, usize, bool)> = Vec::new();
    for path in &changed {
        let before = content(ctx, root, old, path, RightSide::Stored);
        let after = content(ctx, root, new, path, options.right);
        match options.format {
            Format::Check => {
                let report = diff::whitespace_errors(path, before.as_deref(), after.as_deref());
                check_failed |= !report.is_empty();
                io.out.extend_from_slice(report.as_bytes());
            }
            Format::Stat | Format::NumStat | Format::ShortStat => {
                let binary = before.as_deref().is_some_and(diff::is_binary)
                    || after.as_deref().is_some_and(diff::is_binary);
                let (insertions, deletions) = if binary {
                    (0, 0)
                } else {
                    diff::change_counts(before.as_deref(), after.as_deref())
                };
                stats.push((path.clone(), insertions, deletions, binary));
            }
            Format::Patch => {
                let patch =
                    diff::render(path, before.as_deref(), after.as_deref(), options.context);
                io.out.extend_from_slice(patch.as_bytes());
            }
            Format::NameOnly | Format::NameStatus => unreachable!("handled above"),
        }
    }
    if matches!(
        options.format,
        Format::Stat | Format::NumStat | Format::ShortStat
    ) {
        emit_stats(&stats, options.format, io);
    }
    check_failed
}

/// The trailing ` N files changed, ... ` line shared by `--stat`, `--shortstat`, and `git commit`.
pub(crate) fn summary_line(files: usize, insertions: usize, deletions: usize) -> String {
    let mut text = format!(" {files} file{} changed", plural(files));
    if insertions != 0 {
        text.push_str(&format!(
            ", {insertions} insertion{}(+)",
            plural(insertions)
        ));
    }
    if deletions != 0 {
        text.push_str(&format!(", {deletions} deletion{}(-)", plural(deletions)));
    }
    text.push('\n');
    text
}

fn plural(count: usize) -> &'static str {
    if count == 1 {
        ""
    } else {
        "s"
    }
}

fn emit_stats(stats: &[(String, usize, usize, bool)], format: Format, io: &mut Io) {
    let total_insertions: usize = stats.iter().map(|entry| entry.1).sum();
    let total_deletions: usize = stats.iter().map(|entry| entry.2).sum();
    if format == Format::NumStat {
        for (path, insertions, deletions, binary) in stats {
            if *binary {
                io.out
                    .extend_from_slice(format!("-\t-\t{path}\n").as_bytes());
            } else {
                io.out
                    .extend_from_slice(format!("{insertions}\t{deletions}\t{path}\n").as_bytes());
            }
        }
        return;
    }
    if format == Format::Stat {
        let name_width = stats.iter().map(|entry| entry.0.len()).max().unwrap_or(0);
        let count_width = stats
            .iter()
            .map(|entry| (entry.1 + entry.2).to_string().len())
            .max()
            .unwrap_or(1);
        for (path, insertions, deletions, binary) in stats {
            if *binary {
                io.out
                    .extend_from_slice(format!(" {path:name_width$} | Bin\n").as_bytes());
                continue;
            }
            let total = insertions + deletions;
            // Git scales the graph to at most 40 columns while keeping at least one mark per side.
            let (plus, minus) = scale_graph(*insertions, *deletions);
            io.out.extend_from_slice(
                format!(
                    " {path:name_width$} | {total:>count_width$} {}{}\n",
                    "+".repeat(plus),
                    "-".repeat(minus)
                )
                .as_bytes(),
            );
        }
    }
    io.out
        .extend_from_slice(summary_line(stats.len(), total_insertions, total_deletions).as_bytes());
}

fn scale_graph(insertions: usize, deletions: usize) -> (usize, usize) {
    let total = insertions + deletions;
    if total <= 40 {
        return (insertions, deletions);
    }
    let plus = (insertions * 40)
        .div_ceil(total)
        .max(usize::from(insertions > 0));
    let minus = (40_usize.saturating_sub(plus)).max(usize::from(deletions > 0));
    (plus, minus)
}

/// Parse the options `git diff` shares with `git log` and `git show`.
///
/// Returns `None` when the argument is not a recognized diff option.
pub(crate) fn apply_shared_option(options: &mut Options, argument: &str) -> Option<bool> {
    match argument {
        "--name-only" => options.format = Format::NameOnly,
        "--name-status" => options.format = Format::NameStatus,
        "--stat" => options.format = Format::Stat,
        "--numstat" => options.format = Format::NumStat,
        "--shortstat" => options.format = Format::ShortStat,
        "--check" => options.format = Format::Check,
        "-p" | "-u" | "--patch" => options.format = Format::Patch,
        "--no-color" | "--color=never" | "--no-ext-diff" | "--no-renames" => {}
        value => {
            let context = value
                .strip_prefix("-U")
                .or_else(|| value.strip_prefix("--unified="))?;
            options.context = context.parse().ok()?;
        }
    }
    Some(true)
}

/// Split a revision argument that may use range syntax into its two endpoints.
///
/// `a..b` compares the endpoints directly; `a...b` compares from their merge base.
fn split_range(revision: &str) -> Option<(&str, &str, bool)> {
    if let Some((left, right)) = revision.split_once("...") {
        return Some((left, right, true));
    }
    revision
        .split_once("..")
        .map(|(left, right)| (left, right, false))
}

pub(crate) fn git_diff(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return super::repo_error(io);
    };
    let mut options = Options::default();
    let mut cached = false;
    let mut exit_code = false;
    let mut revisions: Vec<String> = Vec::new();
    let mut operands_only = false;
    let cwd = ctx.cwd.clone();
    for argument in args {
        if operands_only {
            options.paths.push(super::pathspec(&cwd, &root, argument));
            continue;
        }
        match argument.as_str() {
            "--" => operands_only = true,
            "--cached" | "--staged" => cached = true,
            "--quiet" => options.quiet = true,
            "--exit-code" => exit_code = true,
            value if value.starts_with('-') => {
                if apply_shared_option(&mut options, value).is_none() {
                    return usage(io, &format!("unsupported diff option: {value}"));
                }
            }
            value if revisions.len() < 2 && repo::resolve_revision(ctx, &root, value).is_some() => {
                revisions.push(value.to_string());
            }
            value
                if revisions.is_empty()
                    && split_range(value).is_some_and(|(left, right, _)| {
                        repo::resolve_revision(ctx, &root, left).is_some()
                            && repo::resolve_revision(ctx, &root, right).is_some()
                    }) =>
            {
                revisions.push(value.to_string());
            }
            value => {
                if !super::names_a_path(ctx, &root, value) {
                    return super::ambiguous_argument(io, value);
                }
                options.paths.push(super::pathspec(&cwd, &root, value));
            }
        }
    }
    if exit_code && !options.quiet {
        // `--exit-code` still prints the patch; only `--quiet` suppresses it.
        options.quiet = false;
    }

    let index = repo::load_index(ctx, &root).unwrap_or_default();
    // Resolve the two sides. Ranges expand into both endpoints.
    if let Some(range) = revisions.first().and_then(|value| split_range(value)) {
        let (left, right, merge_base) = range;
        let (Some(mut left_commit), Some(right_commit)) = (
            repo::resolve_revision(ctx, &root, left),
            repo::resolve_revision(ctx, &root, right),
        ) else {
            return usage(io, "unknown revision range");
        };
        if merge_base {
            let Some(base) = repo::merge_base(ctx, &root, &left_commit, &right_commit) else {
                return usage(io, "the revisions have no common ancestor");
            };
            left_commit = base;
        }
        let old = repo::commit_tree(ctx, &root, &left_commit).unwrap_or_default();
        let new = repo::commit_tree(ctx, &root, &right_commit).unwrap_or_default();
        return finish(ctx, &root, &old, &new, &options, exit_code, io);
    }

    match revisions.len() {
        2 => {
            let old = tree_of(ctx, &root, &revisions[0]);
            let new = tree_of(ctx, &root, &revisions[1]);
            finish(ctx, &root, &old, &new, &options, exit_code, io)
        }
        1 if cached => {
            let old = tree_of(ctx, &root, &revisions[0]);
            finish(ctx, &root, &old, &index.clone(), &options, exit_code, io)
        }
        1 => {
            // `git diff REVISION` compares the revision against the working tree.
            let old = tree_of(ctx, &root, &revisions[0]);
            options.right = RightSide::WorkingTree;
            let Some(work) = tracked_working_tree(ctx, &root, &index) else {
                return repo::resource_error(ctx);
            };
            finish(ctx, &root, &old, &work, &options, exit_code, io)
        }
        _ if cached => {
            let head = repo::head_tree(ctx, &root);
            finish(ctx, &root, &head, &index.clone(), &options, exit_code, io)
        }
        _ => {
            options.right = RightSide::WorkingTree;
            let Some(work) = tracked_working_tree(ctx, &root, &index) else {
                return repo::resource_error(ctx);
            };
            finish(ctx, &root, &index, &work, &options, exit_code, io)
        }
    }
}

/// Hash the working tree, keeping only paths the index tracks.
///
/// `git diff` never reports untracked files, so limiting the snapshot here keeps build output and
/// ignored files out of every diff format.
pub(crate) fn tracked_working_tree(
    ctx: &mut CommandContext<'_>,
    root: &str,
    index: &Tree,
) -> Option<Tree> {
    let mut work = repo::collect_working_tree(ctx, root).ok()?.release(ctx);
    work.retain(|path, _| index.contains_key(path));
    Some(work)
}

fn tree_of(ctx: &mut CommandContext<'_>, root: &str, revision: &str) -> Tree {
    repo::resolve_revision(ctx, root, revision)
        .and_then(|commit| repo::commit_tree(ctx, root, &commit))
        .unwrap_or_default()
}

fn finish(
    ctx: &mut CommandContext<'_>,
    root: &str,
    old: &Tree,
    new: &Tree,
    options: &Options,
    exit_code: bool,
    io: &mut Io,
) -> i32 {
    let differs = old
        .keys()
        .chain(new.keys())
        .any(|path| selected(&options.paths, path) && old.get(path) != new.get(path));
    if options.quiet {
        return i32::from(differs);
    }
    // `git diff --check` reports whitespace problems with exit status 2.
    if emit(ctx, root, old, new, options, io) {
        return 2;
    }
    i32::from(exit_code && differs)
}
