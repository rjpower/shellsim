//! Tree comparison and the `git diff` porcelain.
//!
//! One routine renders the difference between two trees in every output format the subset
//! supports, so `git diff`, `git show`, and `git log -p` agree on formatting. The right-hand side
//! may come from a stored tree or from the working tree, which is what distinguishes
//! `git diff --cached` from a plain `git diff`.

use std::collections::BTreeSet;

use crate::commands::{CommandContext, Io};

use super::conflict;
use super::diff::{self, DEFAULT_CONTEXT};
use super::repo::{self, Tree};
use super::{fatal, usage, Arg, Flags};

/// How a tree comparison is presented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    Patch,
    NameOnly,
    NameStatus,
    Stat,
    NumStat,
    ShortStat,
    Summary,
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
    /// Whether patch paths carry the usual `a/` and `b/` prefixes.
    pub prefixes: bool,
    /// Swap the two sides, as `git diff -R` does.
    pub reverse: bool,
    /// Limit the report to these change letters, as `--diff-filter` does.
    pub filter: Option<String>,
    /// How whitespace differences are treated, as `-w` and `-b` set it.
    pub whitespace: diff::Whitespace,
    /// Paths a merge left unmerged, which are reported with `U` rather than `M`.
    pub unmerged: BTreeSet<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            format: Format::Patch,
            context: DEFAULT_CONTEXT,
            right: RightSide::Stored,
            paths: Vec::new(),
            quiet: false,
            prefixes: true,
            reverse: false,
            filter: None,
            whitespace: diff::Whitespace::Significant,
            unmerged: BTreeSet::new(),
        }
    }
}

/// Whether a repository-relative path is covered by the given pathspec prefixes.
pub(crate) fn selected(paths: &[String], path: &str) -> bool {
    paths.is_empty()
        || paths
            .iter()
            .any(|spec| super::ignore::matches_pathspec(spec, path))
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
            .and_then(|entry| repo::read_blob(ctx, root, &entry.hash)),
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
    let (old, new) = if options.reverse {
        (new, old)
    } else {
        (old, new)
    };
    let mut names = BTreeSet::new();
    names.extend(old.keys().cloned());
    names.extend(new.keys().cloned());
    let mut changed: Vec<String> = names
        .into_iter()
        .filter(|path| selected(&options.paths, path) && old.get(path) != new.get(path))
        .collect();
    let mut renames = detect_renames(old, new, &mut changed);
    if let Some(letters) = &options.filter {
        changed.retain(|path| {
            letters.contains(status_letter(
                old.contains_key(path),
                new.contains_key(path),
                options.unmerged.contains(path),
            ))
        });
        renames.retain(|_| letters.contains('R'));
    }
    if options.whitespace != diff::Whitespace::Significant {
        // Under `-w` or `-b` a file whose only difference is whitespace is not a change at all.
        let context = &*ctx;
        changed.retain(|path| {
            match (
                content(context, root, old, path, RightSide::Stored),
                content(context, root, new, path, options.right),
            ) {
                (Some(before), Some(after)) => {
                    diff::change_counts(Some(&before), Some(&after), options.whitespace) != (0, 0)
                }
                _ => true,
            }
        });
    }
    if changed.is_empty() && renames.is_empty() {
        return false;
    }
    if options.format == Format::NameOnly {
        for (_, to) in &renames {
            io.print(&format!("{to}\n"));
        }
        for path in &changed {
            io.print(&format!("{path}\n"));
        }
        return false;
    }
    if options.format == Format::NameStatus {
        for (from, to) in &renames {
            io.print(&format!("R100\t{from}\t{to}\n"));
        }
        for path in &changed {
            let status = status_letter(
                old.contains_key(path),
                new.contains_key(path),
                options.unmerged.contains(path),
            );
            io.print(&format!("{status}\t{path}\n"));
        }
        return false;
    }
    if options.format == Format::Summary {
        for (from, to) in &renames {
            io.print(&format!(" rename {from} => {to} (100%)\n"));
        }
        for path in &changed {
            match (old.get(path), new.get(path)) {
                (None, Some(entry)) => io.print(&format!(" create mode {} {path}\n", entry.mode())),
                (Some(entry), None) => io.print(&format!(" delete mode {} {path}\n", entry.mode())),
                (Some(before), Some(after)) if before.executable != after.executable => {
                    io.print(&format!(
                        " mode change {} => {} {path}\n",
                        before.mode(),
                        after.mode()
                    ))
                }
                _ => {}
            }
        }
        return false;
    }

    let mut check_failed = false;
    let mut stats: Vec<(String, usize, usize, bool)> = Vec::new();
    for (from, to) in &renames {
        match options.format {
            Format::Stat | Format::NumStat | Format::ShortStat => {
                stats.push((format!("{from} => {to}"), 0, 0, false));
            }
            Format::Patch => {
                io.print(&rename_header(from, to, options.prefixes));
            }
            _ => {}
        }
    }
    for path in &changed {
        let before = content(ctx, root, old, path, RightSide::Stored);
        let after = content(ctx, root, new, path, options.right);
        match options.format {
            Format::Check => {
                let report = diff::whitespace_errors(path, before.as_deref(), after.as_deref());
                check_failed |= !report.is_empty();
                io.print(&report);
            }
            Format::Stat | Format::NumStat | Format::ShortStat => {
                let binary = before.as_deref().is_some_and(diff::is_binary)
                    || after.as_deref().is_some_and(diff::is_binary);
                let (insertions, deletions) = if binary {
                    (0, 0)
                } else {
                    diff::change_counts(before.as_deref(), after.as_deref(), options.whitespace)
                };
                stats.push((path.clone(), insertions, deletions, binary));
            }
            Format::Patch => {
                let patch = diff::render(
                    path,
                    before.as_deref(),
                    after.as_deref(),
                    modes(old, new, path),
                    options.context,
                    options.prefixes,
                    options.whitespace,
                );
                io.print(&patch);
            }
            Format::NameOnly | Format::NameStatus | Format::Summary => {
                unreachable!("handled above")
            }
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

/// The modes a patch reports for one path, defaulting to the non-executable mode.
fn modes(old: &Tree, new: &Tree, path: &str) -> diff::Modes {
    let mode = |tree: &Tree| tree.get(path).map_or("100644", repo::Entry::mode);
    (mode(old), mode(new))
}

/// Pair a deletion with an addition of byte-identical content.
///
/// Similarity-based detection is outside this model, so only an exact move is a rename; the paired
/// paths are removed from `changed`.
fn detect_renames(old: &Tree, new: &Tree, changed: &mut Vec<String>) -> Vec<(String, String)> {
    let removed: Vec<String> = changed
        .iter()
        .filter(|path| !new.contains_key(*path))
        .cloned()
        .collect();
    let added: Vec<String> = changed
        .iter()
        .filter(|path| !old.contains_key(*path))
        .cloned()
        .collect();
    let mut pairs: Vec<(String, String)> = Vec::new();
    for source in removed {
        let Some(hash) = old.get(&source) else {
            continue;
        };
        let target = added.iter().find(|candidate| {
            new.get(*candidate) == Some(hash) && !pairs.iter().any(|(_, taken)| taken == *candidate)
        });
        if let Some(target) = target {
            pairs.push((source, target.clone()));
        }
    }
    changed.retain(|path| !pairs.iter().any(|(from, to)| path == from || path == to));
    pairs
}

/// The header Git prints for a rename, which carries no hunks when the content is unchanged.
fn rename_header(from: &str, to: &str, prefixes: bool) -> String {
    let (a, b) = if prefixes { ("a/", "b/") } else { ("", "") };
    format!(
        "diff --git {a}{from} {b}{to}\nsimilarity index 100%\nrename from {from}\nrename to {to}\n"
    )
}

/// The letter `--name-status` and `--diff-filter` use for one path.
fn status_letter(in_old: bool, in_new: bool, unmerged: bool) -> char {
    match (in_old, in_new) {
        _ if unmerged => 'U',
        (false, true) => 'A',
        (true, false) => 'D',
        _ => 'M',
    }
}

/// The trailing ` N files changed, ... ` line shared by `--stat`, `--shortstat`, and `git commit`.
pub(crate) fn summary_line(files: usize, insertions: usize, deletions: usize) -> String {
    let mut text = format!(" {files} file{} changed", plural(files));
    // A change with no line movement still reports both counts, as Git does for a pure rename.
    if insertions == 0 && deletions == 0 {
        text.push_str(", 0 insertions(+), 0 deletions(-)\n");
        return text;
    }
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
                io.print(&format!("-\t-\t{path}\n"));
            } else {
                io.print(&format!("{insertions}\t{deletions}\t{path}\n"));
            }
        }
        return;
    }
    if format == Format::Stat {
        let max_change = stats
            .iter()
            .map(|entry| entry.1 + entry.2)
            .max()
            .unwrap_or(0);
        let count_width = max_change.to_string().len();
        let (name_width, graph_width) = stat_widths(
            stats.iter().map(|entry| entry.0.len()).max().unwrap_or(0),
            max_change,
            count_width,
        );
        for (path, insertions, deletions, binary) in stats {
            let path = shorten_name(path, name_width);
            if *binary {
                io.print(&format!(" {path:name_width$} | Bin\n"));
                continue;
            }
            let total = insertions + deletions;
            let (plus, minus) = scale_graph(*insertions, *deletions, graph_width, max_change);
            let graph = format!("{}{}", "+".repeat(plus), "-".repeat(minus));
            let separator = if graph.is_empty() { "" } else { " " };
            io.print(&format!(
                " {path:name_width$} | {total:>count_width$}{separator}{graph}\n"
            ));
        }
    }
    io.print(&summary_line(
        stats.len(),
        total_insertions,
        total_deletions,
    ));
}

fn scale_graph(
    insertions: usize,
    deletions: usize,
    graph_width: usize,
    max_change: usize,
) -> (usize, usize) {
    if max_change <= graph_width {
        return (insertions, deletions);
    }
    // Git's linear scale, which always keeps one mark for a side that changed at all.
    let scale = |count: usize| {
        if count == 0 || max_change <= 1 {
            return usize::from(count > 0);
        }
        1 + (count - 1) * (graph_width - 1) / (max_change - 1)
    };
    (scale(insertions), scale(deletions))
}

/// The name and graph columns `--stat` gets, within Git's 80-column budget.
fn stat_widths(max_len: usize, max_change: usize, count_width: usize) -> (usize, usize) {
    const WIDTH: usize = 80;
    let mut name_width = max_len;
    let mut graph_width = max_change;
    if name_width + count_width + 6 + graph_width <= WIDTH {
        return (name_width, graph_width);
    }
    // The graph never takes more than three eighths of the line, and the name gives way first.
    let cap = (WIDTH * 3 / 8).saturating_sub(count_width + 6).max(6);
    graph_width = graph_width.min(cap);
    let available = WIDTH.saturating_sub(count_width + 6 + graph_width);
    if name_width > available {
        name_width = available;
    } else {
        graph_width = WIDTH - count_width - 6 - name_width;
    }
    (name_width, graph_width)
}

/// Shorten a path to `width`, cutting at a directory boundary the way Git does.
fn shorten_name(name: &str, width: usize) -> String {
    if name.len() <= width {
        return name.to_string();
    }
    let skip = name.len() - width + 3;
    let tail = match name[skip.min(name.len())..].find('/') {
        Some(offset) => &name[skip + offset..],
        None => &name[skip.min(name.len())..],
    };
    format!("...{tail}")
}

/// Parse the options `git diff` shares with `git log` and `git show`.
///
/// Returns `None` when the argument is not a recognized diff option.
pub(crate) fn apply_shared_option(
    options: &mut Options,
    flags: &mut Flags,
    name: &str,
    attached: Option<String>,
) -> Option<bool> {
    match name {
        "--name-only" => options.format = Format::NameOnly,
        "--name-status" => options.format = Format::NameStatus,
        "--stat" => options.format = Format::Stat,
        "--summary" => options.format = Format::Summary,
        "--no-prefix" => options.prefixes = false,
        "-R" => options.reverse = true,
        "--numstat" => options.format = Format::NumStat,
        "--shortstat" => options.format = Format::ShortStat,
        "--check" => options.format = Format::Check,
        "-p" | "-u" | "--patch" => options.format = Format::Patch,
        "--no-color" | "--color" | "--no-ext-diff" | "--no-renames" => {}
        // Exact renames are always detected here, so asking for them changes nothing, and the
        // similarity threshold a number would set has nothing to tune.
        "-M" | "--find-renames" | "--find-copies-harder" => {}
        "-w" | "--ignore-all-space" => options.whitespace = diff::Whitespace::IgnoreAll,
        "-b" | "--ignore-space-change" => options.whitespace = diff::Whitespace::IgnoreChange,
        "--diff-filter" => options.filter = Some(flags.value(attached)?.to_ascii_uppercase()),
        "-U" | "--unified" => options.context = flags.value(attached)?.parse().ok()?,
        _ => return None,
    }
    Some(true)
}

/// The short options `git diff` lets a value be written against, as `-U3` and `-M50%` are.
pub(crate) const DIFF_VALUED: &str = "UM";

/// Compare two files directly, as `git diff --no-index` does.
///
/// Exits 1 when the files differ, which is how the command is usually used as a plain differ.
fn diff_no_index(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut options = Options::default();
    let mut operands: Vec<String> = Vec::new();
    let mut flags = Flags::new(args).valued(DIFF_VALUED);
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                operands.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        if name == "--no-index" {
            continue;
        }
        if apply_shared_option(&mut options, &mut flags, &name, attached).is_none() {
            return super::usage(io, &format!("unsupported diff option: {name}"));
        }
    }
    let [left, right] = operands.as_slice() else {
        return super::usage(io, "usage: git diff --no-index PATH PATH");
    };
    let read = |path: &str| -> Option<Vec<u8>> {
        let absolute = crate::vfs::resolve_against(&ctx.cwd, path);
        ctx.fs_read_limited("/", &absolute, 16 * 1024 * 1024).ok()
    };
    let (Some(before), Some(after)) = (read(left), read(right)) else {
        io.print_err(&format!(
            "fatal: cannot read '{left}' or '{right}': No such file or directory\n"
        ));
        return 128;
    };
    if before == after {
        return 0;
    }
    // Git labels the two sides with the operands themselves rather than a repository path.
    let patch = diff::render(
        left,
        Some(&before),
        Some(&after),
        diff::PLAIN,
        options.context,
        options.prefixes,
        options.whitespace,
    );
    let patch = patch.replacen(
        &format!("diff --git a/{left} b/{left}"),
        &format!("diff --git a/{left} b/{right}"),
        1,
    );
    let patch = patch.replacen(&format!("+++ b/{left}"), &format!("+++ b/{right}"), 1);
    if patch.is_empty() {
        // `-w` or `-b` can make files that differ in bytes compare equal.
        return 0;
    }
    io.print(&patch);
    1
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
    // `--no-index` compares two files on their own and needs no repository.
    if args.iter().any(|argument| argument == "--no-index") {
        return diff_no_index(ctx, args, io);
    }
    let Some(root) = repo::find_repo_root(ctx) else {
        return super::repo_error(io);
    };
    let mut options = Options::default();
    let mut cached = false;
    let mut exit_code = false;
    let mut revisions: Vec<String> = Vec::new();
    let cwd = ctx.cwd.clone();
    let mut flags = Flags::new(args).valued(DIFF_VALUED);
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) if flags.separated() => {
                options.paths.push(super::pathspec(&cwd, &root, &value));
                continue;
            }
            Arg::Operand(value) => {
                let revision = revisions.len() < 2
                    && (repo::resolve_revision(ctx, &root, &value).is_some()
                        || (revisions.is_empty()
                            && split_range(&value).is_some_and(|(left, right, _)| {
                                repo::resolve_revision(ctx, &root, left).is_some()
                                    && repo::resolve_revision(ctx, &root, right).is_some()
                            })));
                if revision {
                    revisions.push(value);
                    continue;
                }
                if !super::names_a_path(ctx, &root, &value) {
                    return super::ambiguous_argument(io, &value);
                }
                options.paths.push(super::pathspec(&cwd, &root, &value));
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "--cached" | "--staged" => cached = true,
            "--quiet" => options.quiet = true,
            "--exit-code" => exit_code = true,
            _ => {
                if apply_shared_option(&mut options, &mut flags, &name, attached).is_none() {
                    return usage(io, &format!("unsupported diff option: {name}"));
                }
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
            return fatal(io, "unknown revision range");
        };
        if merge_base {
            let Some(base) = repo::merge_base(ctx, &root, &left_commit, &right_commit) else {
                return fatal(io, "the revisions have no common ancestor");
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
            // An unmerged path has no settled index entry to compare against, so the side that
            // was ours going into the merge stands in for it; otherwise the conflict markers the
            // merge wrote would compare equal and the tree would look clean.
            let mut index = index;
            for (path, entry) in conflict::load_stages(ctx, &root) {
                options.unmerged.insert(path.clone());
                match entry.ours {
                    Some(hash) => {
                        let recorded = index.get(&path).cloned().unwrap_or_default();
                        index.insert(path, repo::Entry { hash, ..recorded });
                    }
                    None => {
                        index.remove(&path);
                    }
                }
            }
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
    let mut work = repo::collect_working_tree(ctx, root).ok()?;
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
        .filter(|path| selected(&options.paths, path) && old.get(*path) != new.get(*path))
        .any(|path| {
            // Under `-w` or `-b` a whitespace-only difference does not count as a difference.
            if options.whitespace == diff::Whitespace::Significant {
                return true;
            }
            match (
                content(ctx, root, old, path, RightSide::Stored),
                content(ctx, root, new, path, options.right),
            ) {
                (Some(before), Some(after)) => {
                    diff::change_counts(Some(&before), Some(&after), options.whitespace) != (0, 0)
                }
                _ => true,
            }
        });
    if options.quiet {
        return i32::from(differs);
    }
    // `git diff --check` reports whitespace problems with exit status 2.
    if emit(ctx, root, old, new, options, io) {
        return 2;
    }
    i32::from(exit_code && differs)
}
