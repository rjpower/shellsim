//! Commit, history, reference, and branch-switching porcelain.
//!
//! Commits are linear or two-parent; the subset has no rebase, cherry-pick, or reflog. Merges are
//! fast-forward when possible and otherwise combine two trees file by file, refusing the merge
//! when the same file changed on both sides rather than writing conflict markers a simulated
//! resolver could not help with.

use std::collections::BTreeSet;

use crate::commands::{CommandContext, Io};

use super::compare::{self, Format, Options, RightSide};
use super::conflict::{self, Stages};
use super::diff;
use super::repo::{self, Commit, Tree};
use super::worktree;
use super::{repo_error, usage, Globals};

fn author_identity(
    ctx: &mut CommandContext<'_>,
    root: &str,
    globals: &Globals,
) -> (String, String) {
    super::config::identity(ctx, root, globals)
}

pub(crate) fn now_seconds(ctx: &CommandContext<'_>) -> i64 {
    ctx.clock
        .wall_time_seconds_floor()
        .ok()
        .and_then(|seconds| i64::try_from(seconds).ok())
        .unwrap_or_default()
}

/// Format a commit timestamp the way `git log` prints author dates.
fn format_date(timestamp: i64) -> String {
    crate::commands::proc::format_date(
        i128::from(timestamp) * 1_000_000_000,
        "%a %b %-d %H:%M:%S %Y +0000",
    )
    .unwrap_or_else(|_| timestamp.to_string())
}

// -- commit -----------------------------------------------------------------------------------

pub(crate) fn git_commit(
    ctx: &mut CommandContext<'_>,
    globals: &Globals,
    args: &[String],
    io: &mut Io,
) -> i32 {
    let mut messages: Vec<String> = Vec::new();
    let mut message_file = None;
    let mut stage_tracked = false;
    let mut amend = false;
    let mut allow_empty = false;
    let mut quiet = false;
    let mut dry_run = false;
    let mut signoff = false;
    let mut author = None;
    let mut paths: Vec<String> = Vec::new();
    let mut operands_only = false;
    let args = super::expand_clusters(args, "amqvns");
    let mut index = 0;
    while index < args.len() {
        let argument = args[index].as_str();
        if operands_only {
            paths.push(argument.to_string());
            index += 1;
            continue;
        }
        match argument {
            "--" => operands_only = true,
            "-m" | "--message" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return usage(io, "-m requires a message");
                };
                messages.push(value.clone());
            }
            "-F" | "--file" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return usage(io, "-F requires a file");
                };
                message_file = Some(value.clone());
            }
            "-a" | "--all" => stage_tracked = true,
            "--amend" => amend = true,
            "--allow-empty" | "--allow-empty-message" => allow_empty = true,
            "-q" | "--quiet" => quiet = true,
            "--dry-run" => dry_run = true,
            "-s" | "--signoff" => signoff = true,
            "-n" | "--no-verify" | "-v" | "--verbose" | "--no-edit" | "--no-gpg-sign" => {}
            "--author" => {
                index += 1;
                author = args.get(index).cloned();
            }
            value if value.starts_with("--message=") => {
                messages.push(value["--message=".len()..].to_string());
            }
            value if value.starts_with("--author=") => {
                author = Some(value["--author=".len()..].to_string());
            }
            value if value.starts_with("--file=") => {
                message_file = Some(value["--file=".len()..].to_string());
            }
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported commit option: {value}"))
            }
            value => paths.push(value.to_string()),
        }
        index += 1;
    }
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    if stage_tracked && !paths.is_empty() {
        return usage(io, "-a cannot be combined with paths");
    }
    if stage_tracked {
        if let Err(status) = worktree::stage_tracked_changes(ctx, &root, io) {
            return status;
        }
    }
    // `git commit MESSAGE PATH...` commits the working-tree state of those paths and leaves the
    // rest of the index alone, so the named paths are refreshed before the index is read.
    if !paths.is_empty() {
        let operands: Vec<String> = std::iter::once("--".to_string())
            .chain(paths.iter().cloned())
            .collect();
        let status = worktree::git_add(ctx, &operands, io);
        if status != 0 {
            return status;
        }
    }
    // A merge, cherry-pick, or revert waiting on the user supplies the message and the parents.
    let pending = pending_operation(ctx, &root);
    if !conflict::load_stages(ctx, &root).is_empty() {
        io.err.extend_from_slice(
            b"error: Committing is not possible because you have unmerged files.\n\
              hint: Fix them up in the work tree, and then use 'git add/rm <file>'\n\
              hint: as appropriate to mark resolution and make a commit.\n\
              fatal: Exiting because of an unresolved conflict.\n",
        );
        return 1;
    }
    let previous = repo::head_commit(ctx, &root)
        .and_then(|id| repo::load_commit(ctx, &root, &id).map(|commit| (id, commit)));
    if amend && previous.is_none() {
        io.err
            .extend_from_slice(b"fatal: You have nothing to amend.\n");
        return 128;
    }
    let mut message = messages.join("\n\n");
    if let Some(path) = &message_file {
        // `-F -` reads the message from standard input, as Git does.
        let bytes = if path == "-" {
            Ok(io.stdin.clone())
        } else {
            let absolute = crate::vfs::resolve_against(&ctx.cwd, path);
            ctx.fs_read_limited("/", &absolute, 1024 * 1024)
        };
        let Ok(bytes) = bytes else {
            io.err.extend_from_slice(
                format!("fatal: could not read log file '{path}': No such file or directory\n")
                    .as_bytes(),
            );
            return 128;
        };
        let text = String::from_utf8_lossy(&bytes).trim_end().to_string();
        message = if message.is_empty() {
            text
        } else {
            format!("{message}\n\n{text}")
        };
    }
    if message.is_empty() {
        if let (true, Some((_, commit))) = (amend, previous.as_ref()) {
            message = commit.message.clone();
        }
    }
    if message.is_empty() {
        if let Some(recorded) = pending
            .as_ref()
            .and_then(|_| conflict::pending_message(ctx, &root))
        {
            message = recorded;
        }
    }
    if dry_run {
        // Git reports what a commit would record, stops before asking for a message, and fails
        // when there is nothing staged to record.
        let staged =
            repo::load_index(ctx, &root).unwrap_or_default() != repo::head_tree(ctx, &root);
        let status = super::worktree::git_status(ctx, &[], io);
        return if status == 0 && !staged { 1 } else { status };
    }
    if message.is_empty() && !allow_empty {
        return usage(io, "a non-empty -m MESSAGE is required");
    }
    let Some(index_tree) = repo::load_index(ctx, &root) else {
        io.err
            .extend_from_slice(b"fatal: invalid simulated index\n");
        return 128;
    };
    // Amending replaces the previous commit, so the comparison tree is its parent's.
    let (parents, baseline) = match (amend, &previous) {
        (true, Some((_, commit))) => {
            let baseline = commit
                .parents
                .first()
                .and_then(|parent| repo::commit_tree(ctx, &root, parent))
                .unwrap_or_default();
            (commit.parents.clone(), baseline)
        }
        (_, Some((id, _))) => {
            let mut parents = vec![id.clone()];
            // Finishing a merge records the commit that was being merged as the second parent.
            if let Some((conflict::MERGE_HEAD, other)) = pending.as_ref().map(|(k, c)| (*k, c)) {
                parents.push(other.clone());
            }
            (
                parents,
                repo::commit_tree(ctx, &root, id).unwrap_or_default(),
            )
        }
        _ => (Vec::new(), Tree::new()),
    };
    // With pathspecs the commit records HEAD plus those paths; anything else stays staged.
    let index_tree = if paths.is_empty() {
        index_tree
    } else {
        let cwd = ctx.cwd.clone();
        let specs: Vec<String> = paths
            .iter()
            .map(|path| super::pathspec(&cwd, &root, path))
            .collect();
        let named = |path: &str| {
            specs
                .iter()
                .any(|spec| super::ignore::matches_pathspec(spec, path))
        };
        let mut tree = baseline.clone();
        tree.retain(|path, _| !named(path));
        for (path, hash) in &index_tree {
            if named(path) {
                tree.insert(path.clone(), hash.clone());
            }
        }
        tree
    };
    // Finishing a merge records a commit even when the merge changed nothing.
    if index_tree == baseline && !allow_empty && pending.is_none() {
        emit_nothing_to_commit(ctx, io);
        return 1;
    }
    if signoff {
        let (name, email) = author_identity(ctx, &root, globals);
        message.push_str(&format!("\n\nSigned-off-by: {name} <{email}>"));
    }
    let (default_name, default_email) = author_identity(ctx, &root, globals);
    let (author_name, author_email) = match author.as_deref().and_then(parse_author) {
        Some(parsed) => parsed,
        None => (default_name, default_email),
    };
    let commit = Commit {
        parents,
        author_name,
        author_email,
        timestamp: previous
            .as_ref()
            .filter(|_| amend)
            .map_or_else(|| author_date(ctx), |(_, commit)| commit.timestamp),
        message: message.clone(),
    };
    let id = match repo::store_commit(ctx, &root, &commit, &index_tree) {
        Ok(id) => id,
        Err(error) => {
            io.err
                .extend_from_slice(format!("git commit: {error}\n").as_bytes());
            return 1;
        }
    };
    if repo::update_head(ctx, &root, &id, &format!("commit: {}", commit.subject())).is_err() {
        return 1;
    }
    if pending.is_some() {
        conflict::clear(ctx, &root);
    }
    if quiet {
        return 0;
    }
    let label = repo::current_branch(ctx, &root)
        .unwrap_or_else(|| format!("detached HEAD {}", repo::short(&id)));
    // An amended root commit is not announced as one, since it is not a new commit.
    let root_commit = if commit.parents.is_empty() && !amend {
        "(root-commit) "
    } else {
        ""
    };
    io.out.extend_from_slice(
        format!(
            "[{label} {root_commit}{}] {}\n",
            repo::short(&id),
            commit.subject()
        )
        .as_bytes(),
    );
    if amend {
        // Amending keeps the original author date, which Git points out.
        io.out
            .extend_from_slice(format!(" Date: {}\n", format_date(commit.timestamp)).as_bytes());
    }
    emit_commit_summary(ctx, &root, &baseline, &index_tree, io);
    0
}

fn parse_author(value: &str) -> Option<(String, String)> {
    let (name, rest) = value.rsplit_once(" <")?;
    let email = rest.strip_suffix('>')?;
    Some((name.to_string(), email.to_string()))
}

/// Print the ` N files changed ...` line and the per-file mode lines Git shows after a commit.
fn emit_commit_summary(
    ctx: &mut CommandContext<'_>,
    root: &str,
    old: &Tree,
    new: &Tree,
    io: &mut Io,
) {
    let mut insertions = 0;
    let mut deletions = 0;
    let changed: BTreeSet<&String> = new
        .keys()
        .chain(old.keys())
        .filter(|path| old.get(*path) != new.get(*path))
        .collect();
    // An empty commit records nothing, so Git prints no summary at all.
    if changed.is_empty() {
        return;
    }
    for path in &changed {
        let before = old
            .get(*path)
            .and_then(|entry| repo::read_blob(ctx, root, &entry.hash));
        let after = new
            .get(*path)
            .and_then(|entry| repo::read_blob(ctx, root, &entry.hash));
        let (added, removed) = super::diff::change_counts(
            before.as_deref(),
            after.as_deref(),
            super::diff::Whitespace::Significant,
        );
        insertions += added;
        deletions += removed;
    }
    io.out
        .extend_from_slice(compare::summary_line(changed.len(), insertions, deletions).as_bytes());
    emit_mode_lines(old, new, io);
}

/// Print the ` create mode` and ` delete mode` lines that follow a change summary.
fn emit_mode_lines(old: &Tree, new: &Tree, io: &mut Io) {
    let changed: BTreeSet<&String> = new
        .keys()
        .chain(old.keys())
        .filter(|path| old.get(*path) != new.get(*path))
        .collect();
    for path in changed {
        match (old.get(path), new.get(path)) {
            (None, Some(entry)) => io
                .out
                .extend_from_slice(format!(" create mode {} {path}\n", entry.mode()).as_bytes()),
            (Some(entry), None) => io
                .out
                .extend_from_slice(format!(" delete mode {} {path}\n", entry.mode()).as_bytes()),
            (Some(before), Some(after)) if before.executable != after.executable => {
                io.out.extend_from_slice(
                    format!(
                        " mode change {} => {} {path}\n",
                        before.mode(),
                        after.mode()
                    )
                    .as_bytes(),
                )
            }
            _ => {}
        }
    }
}

/// Report that there is nothing staged, using the same wording as `git status`.
/// Report that nothing is staged, using the same wording as `git status`.
fn emit_nothing_to_commit(ctx: &mut CommandContext<'_>, io: &mut Io) {
    super::worktree::git_status(ctx, &[], io);
}

// -- log and show -----------------------------------------------------------------------------

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
fn decorations(ctx: &CommandContext<'_>, root: &str, id: &str) -> String {
    let mut names = Vec::new();
    let head_branch = repo::current_branch(ctx, root);
    for branch in repo::branch_names(ctx, root) {
        if repo::read_reference(ctx, root, &format!("refs/heads/{branch}")).as_deref() != Some(id) {
            continue;
        }
        if head_branch.as_deref() == Some(branch.as_str()) {
            names.insert(0, format!("HEAD -> {branch}"));
        } else {
            names.push(branch);
        }
    }
    for tag in repo::reference_names(ctx, root, "tags") {
        if repo::read_reference(ctx, root, &format!("refs/tags/{tag}")).as_deref() == Some(id) {
            names.push(format!("tag: {tag}"));
        }
    }
    if head_branch.is_none() && repo::head_commit(ctx, root).as_deref() == Some(id) {
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
            format_date(commit.timestamp),
            indent(&commit.message)
        ),
        Pretty::Full => format!(
            "commit {id}{decoration}\n{merge}Author: {identity}\nCommit: {identity}\n\n{}",
            indent(&commit.message)
        ),
        Pretty::Fuller => format!(
            "commit {id}{decoration}\n{merge}Author:     {identity}\nAuthorDate: {}\nCommit:     {identity}\nCommitDate: {}\n\n{}",
            format_date(commit.timestamp),
            format_date(commit.timestamp),
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
            "ad" | "cd" => format_date(commit.timestamp),
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
fn select_revision(ctx: &mut CommandContext<'_>, root: &str, revision: &str) -> Option<Selection> {
    // `^rev` excludes everything reachable from `rev` and includes nothing.
    if let Some(excluded) = revision.strip_prefix('^') {
        let commit = repo::resolve_revision(ctx, root, excluded)?;
        return Some(Selection {
            included: Vec::new(),
            excluded: repo::ancestors(ctx, root, &commit),
        });
    }
    let Some((left, right, merge_base)) = split_range(revision) else {
        return Some(Selection {
            included: vec![repo::resolve_revision(ctx, root, revision)?],
            excluded: BTreeSet::new(),
        });
    };
    let right = if right.is_empty() { "HEAD" } else { right };
    let left = if left.is_empty() { "HEAD" } else { left };
    let (left_commit, right_commit) = (
        repo::resolve_revision(ctx, root, left)?,
        repo::resolve_revision(ctx, root, right)?,
    );
    let excluded = if merge_base {
        let base = repo::merge_base(ctx, root, &left_commit, &right_commit)?;
        repo::ancestors(ctx, root, &base)
    } else {
        repo::ancestors(ctx, root, &left_commit)
    };
    Some(Selection {
        included: vec![right_commit],
        excluded,
    })
}

/// The commits listed by a set of revision arguments, newest first.
fn history_for(
    ctx: &mut CommandContext<'_>,
    root: &str,
    revisions: &[String],
    first_parent: bool,
) -> Option<Vec<(String, Commit)>> {
    let mut included = Vec::new();
    let mut excluded = BTreeSet::new();
    for revision in revisions {
        let selection = select_revision(ctx, root, revision)?;
        included.extend(selection.included);
        excluded.extend(selection.excluded);
    }
    let listed = if first_parent {
        included
            .first()
            .map(|start| repo::first_parent_history(ctx, root, start, 10_000))
            .unwrap_or_default()
    } else {
        repo::reachable_history(ctx, root, &included, 10_000)
    };
    Some(
        listed
            .into_iter()
            .filter(|(id, _)| !excluded.contains(id))
            .collect(),
    )
}

/// Every branch and tag tip, for `--all`.
fn all_reference_tips(ctx: &CommandContext<'_>, root: &str) -> Vec<String> {
    let mut tips = Vec::new();
    for (kind, names) in [
        ("heads", repo::branch_names(ctx, root)),
        ("tags", repo::reference_names(ctx, root, "tags")),
    ] {
        for name in names {
            if let Some(commit) = repo::read_reference(ctx, root, &format!("refs/{kind}/{name}")) {
                tips.push(commit);
            }
        }
    }
    tips
}

pub(crate) fn git_log(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
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
    let mut operands_only = false;
    let mut index = 0;
    let cwd = ctx.cwd.clone();
    while index < args.len() {
        let argument = args[index].as_str();
        if operands_only {
            paths.push(super::pathspec(&cwd, &root, argument));
            index += 1;
            continue;
        }
        match argument {
            "--" => operands_only = true,
            "--oneline" => pretty = Pretty::AbbreviatedOneLine,
            "--reverse" => reverse = true,
            "--decorate" | "--decorate=short" => decorate = true,
            "--no-decorate" | "--decorate=no" | "--no-color" => {}
            "--no-merges" => no_merges = true,
            "--merges" => merges_only = true,
            "--abbrev-commit" => abbreviate = true,
            "--first-parent" => first_parent = true,
            "--all" => all_references = true,
            "-n" | "--max-count" => {
                index += 1;
                let Some(value) = args.get(index).and_then(|value| value.parse().ok()) else {
                    return usage(io, "log count must be a non-negative integer");
                };
                limit = value;
            }
            "-i" | "--regexp-ignore-case" => ignore_case = true,
            "--since" | "--after" | "--until" | "--before" => {
                let after = matches!(argument, "--since" | "--after");
                index += 1;
                let Some(value) = args.get(index) else {
                    return usage(io, &format!("{argument} requires a date"));
                };
                let Some(seconds) = parse_date(value, now_seconds(ctx)) else {
                    return usage(io, &format!("unsupported date: {value}"));
                };
                if after {
                    since = Some(seconds);
                } else {
                    until = Some(seconds);
                }
            }
            "--grep" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return usage(io, "--grep requires a pattern");
                };
                message_filter = Some(value.clone());
            }
            "--author" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return usage(io, "--author requires a pattern");
                };
                author_filter = Some(value.clone());
            }
            "-S" | "-G" => {
                let option = argument.to_string();
                index += 1;
                let Some(value) = args.get(index) else {
                    return usage(io, &format!("{option} requires a string"));
                };
                if option == "-S" {
                    pickaxe = Some(value.clone());
                } else {
                    changed_lines = Some(value.clone());
                }
            }
            "--graph" => graph = Some(Rail::default()),
            "--no-graph" => graph = None,
            "--stat" => stat = Some(Format::Stat),
            "--name-only" => stat = Some(Format::NameOnly),
            "--name-status" => stat = Some(Format::NameStatus),
            "-p" | "-u" | "--patch" => patch = Some(Format::Patch),
            value if value.starts_with("--skip=") => {
                let Ok(value) = value["--skip=".len()..].parse() else {
                    return usage(io, "--skip requires a non-negative integer");
                };
                skip = value;
            }
            value if value.starts_with("--author=") => {
                author_filter = Some(value["--author=".len()..].to_string());
            }
            value if value.starts_with("--grep=") => {
                message_filter = Some(value["--grep=".len()..].to_string());
            }
            value
                if value.starts_with("--since=")
                    || value.starts_with("--after=")
                    || value.starts_with("--until=")
                    || value.starts_with("--before=") =>
            {
                let (name, date) = value.split_once('=').unwrap_or((value, ""));
                let Some(seconds) = parse_date(date, now_seconds(ctx)) else {
                    return usage(io, &format!("unsupported date: {date}"));
                };
                if matches!(name, "--since" | "--after") {
                    since = Some(seconds);
                } else {
                    until = Some(seconds);
                }
            }
            value if value.starts_with("-S") && value.len() > 2 => {
                pickaxe = Some(value[2..].to_string());
            }
            value if value.starts_with("-G") && value.len() > 2 => {
                changed_lines = Some(value[2..].to_string());
            }
            value if value.starts_with("--max-count=") => {
                let Ok(value) = value["--max-count=".len()..].parse() else {
                    return usage(io, "log count must be a non-negative integer");
                };
                limit = value;
            }
            value if value.starts_with("--pretty=") || value.starts_with("--format=") => {
                let terminated = value.starts_with("--format=");
                let Some(parsed) = parse_pretty(
                    value.split_once('=').map_or("", |parts| parts.1),
                    terminated,
                )
                .filter(|parsed| match parsed {
                    Pretty::Custom { format, .. } => format_is_supported(format),
                    _ => true,
                }) else {
                    return usage(io, &format!("unsupported log format: {value}"));
                };
                pretty = parsed;
            }
            value if value.starts_with("-n") && value.len() > 2 => {
                let Ok(value) = value[2..].parse() else {
                    return usage(io, "log count must be a non-negative integer");
                };
                limit = value;
            }
            value
                if value.len() > 1
                    && value.starts_with('-')
                    && value[1..].bytes().all(|byte| byte.is_ascii_digit()) =>
            {
                let Ok(value) = value[1..].parse() else {
                    return usage(io, "log count must be a non-negative integer");
                };
                limit = value;
            }
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported log option: {value}"))
            }
            value if select_revision(ctx, &root, value).is_some() => {
                revisions.push(value.to_string())
            }
            value => {
                if !super::names_a_path(ctx, &root, value) {
                    return super::ambiguous_argument(io, value);
                }
                paths.push(super::pathspec(&cwd, &root, value));
            }
        }
        index += 1;
    }
    if all_references {
        revisions.extend(all_reference_tips(ctx, &root));
    }
    if revisions.is_empty() {
        revisions.push("HEAD".to_string());
    }
    let Some(mut history) = history_for(ctx, &root, &revisions, first_parent) else {
        if repo::head_commit(ctx, &root).is_none() {
            let branch = repo::current_branch(ctx, &root).unwrap_or_else(|| "HEAD".to_string());
            io.err.extend_from_slice(
                format!("fatal: your current branch '{branch}' does not have any commits yet\n")
                    .as_bytes(),
            );
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
            io.err
                .extend_from_slice(format!("fatal: invalid pattern: {pattern}\n").as_bytes());
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
        history.retain(|(id, commit)| changes_occurrence_count(ctx, &root, id, commit, needle));
    }
    if let Some(pattern) = &changed_lines {
        let Ok(regex) = regex::RegexBuilder::new(pattern)
            .case_insensitive(ignore_case)
            .build()
        else {
            io.err
                .extend_from_slice(format!("fatal: invalid pattern: {pattern}\n").as_bytes());
            return 128;
        };
        history.retain(|(id, commit)| matches_changed_lines(ctx, &root, id, commit, &regex));
    }
    if !paths.is_empty() {
        history.retain(|(id, commit)| commit_touches(ctx, &root, id, commit, &paths));
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
    let now = now_seconds(ctx);
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
            decorations(ctx, &root, id)
        } else {
            String::new()
        };
        let displayed = if abbreviate {
            repo::short(id)
        } else {
            id.as_str()
        };
        io.out.extend_from_slice(
            render_commit_header(displayed, commit, &pretty, &decoration, now).as_bytes(),
        );
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
            io.out.push(b'\n');
            emit_commit_diff(ctx, &root, id, commit, format, &paths, io);
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
        io.out.extend_from_slice(prefix.as_bytes());
        io.out.extend_from_slice(line);
        io.out.push(b'\n');
    }
    io.out.extend_from_slice(rung.connectors.as_bytes());
}

fn commit_touches(
    ctx: &CommandContext<'_>,
    root: &str,
    id: &str,
    commit: &Commit,
    paths: &[String],
) -> bool {
    let tree = repo::commit_tree(ctx, root, id).unwrap_or_default();
    let parent = commit
        .parents
        .first()
        .and_then(|parent| repo::commit_tree(ctx, root, parent))
        .unwrap_or_default();
    let mut names: BTreeSet<String> = tree.keys().cloned().collect();
    names.extend(parent.keys().cloned());
    names
        .iter()
        .any(|path| tree.get(path) != parent.get(path) && compare::selected(paths, path))
}

/// The author date for a new commit, which `GIT_AUTHOR_DATE` may set as it does in Git.
fn author_date(ctx: &mut CommandContext<'_>) -> i64 {
    let now = now_seconds(ctx);
    ctx.get_var("GIT_AUTHOR_DATE")
        .and_then(|value| parse_date(&value, now))
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
    ctx: &CommandContext<'_>,
    root: &str,
    id: &str,
    commit: &Commit,
) -> Vec<(Option<String>, Option<String>)> {
    let tree = repo::commit_tree(ctx, root, id).unwrap_or_default();
    let parent = commit
        .parents
        .first()
        .and_then(|parent| repo::commit_tree(ctx, root, parent))
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
    ctx: &CommandContext<'_>,
    root: &str,
    id: &str,
    commit: &Commit,
    needle: &str,
) -> bool {
    let occurrences = |hash: Option<String>| -> usize {
        let Some(data) = hash.and_then(|hash| repo::read_blob(ctx, root, &hash)) else {
            return 0;
        };
        if needle.is_empty() {
            return 0;
        }
        String::from_utf8_lossy(&data).matches(needle).count()
    };
    changed_blobs(ctx, root, id, commit)
        .into_iter()
        .any(|(before, after)| occurrences(before) != occurrences(after))
}

/// Whether any line a commit added or removed matches `regex`, which is what `git log -G` selects.
fn matches_changed_lines(
    ctx: &CommandContext<'_>,
    root: &str,
    id: &str,
    commit: &Commit,
    regex: &regex::Regex,
) -> bool {
    changed_blobs(ctx, root, id, commit)
        .into_iter()
        .any(|(before, after)| {
            let read = |hash: Option<String>| {
                hash.and_then(|hash| repo::read_blob(ctx, root, &hash))
                    .unwrap_or_default()
            };
            let before = read(before);
            let after = read(after);
            diff::edit_script(&diff::split_lines(&before), &diff::split_lines(&after))
                .iter()
                .any(|edit| !matches!(edit.op, diff::Op::Keep) && regex.is_match(edit.text))
        })
}

/// Render one commit's difference against its first parent.
fn emit_commit_diff(
    ctx: &mut CommandContext<'_>,
    root: &str,
    id: &str,
    commit: &Commit,
    format: Format,
    paths: &[String],
    io: &mut Io,
) {
    let tree = repo::commit_tree(ctx, root, id).unwrap_or_default();
    let parent = commit
        .parents
        .first()
        .and_then(|parent| repo::commit_tree(ctx, root, parent))
        .unwrap_or_default();
    let options = Options {
        format,
        right: RightSide::Stored,
        paths: paths.to_vec(),
        ..Options::default()
    };
    compare::emit(ctx, root, &parent, &tree, &options, io);
}

pub(crate) fn git_show(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut format = Some(Format::Patch);
    let mut pretty = Pretty::Medium;
    let mut revisions: Vec<String> = Vec::new();
    for argument in args {
        match argument.as_str() {
            "-s" | "--no-patch" => format = None,
            "--stat" => format = Some(Format::Stat),
            "--name-only" => format = Some(Format::NameOnly),
            "--name-status" => format = Some(Format::NameStatus),
            "--oneline" => pretty = Pretty::AbbreviatedOneLine,
            "--no-color" | "--abbrev-commit" => {}
            value if value.starts_with("--pretty=") || value.starts_with("--format=") => {
                let Some(parsed) = parse_pretty(
                    value.split_once('=').map_or("", |parts| parts.1),
                    value.starts_with("--format="),
                )
                .filter(|parsed| match parsed {
                    Pretty::Custom { format, .. } => format_is_supported(format),
                    _ => true,
                }) else {
                    return usage(io, &format!("unsupported show format: {value}"));
                };
                pretty = parsed;
            }
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported show option: {value}"))
            }
            value => revisions.push(value.to_string()),
        }
    }
    if revisions.is_empty() {
        revisions.push("HEAD".to_string());
    }
    for revision in &revisions {
        // `REVISION:PATH` prints one file's contents at that revision.
        if let Some((prefix, path)) = revision.split_once(':') {
            let Some(commit) = repo::resolve_revision(ctx, &root, prefix) else {
                return super::ambiguous_argument(io, revision);
            };
            let tree = repo::commit_tree(ctx, &root, &commit).unwrap_or_default();
            let Some(data) = tree
                .get(path)
                .and_then(|entry| repo::read_blob(ctx, &root, &entry.hash))
            else {
                io.err.extend_from_slice(
                    format!("fatal: path '{path}' does not exist in '{prefix}'\n").as_bytes(),
                );
                return 128;
            };
            io.out.extend_from_slice(&data);
            continue;
        }
        let Some(id) = repo::resolve_revision(ctx, &root, revision) else {
            return super::ambiguous_argument(io, revision);
        };
        let Some(commit) = repo::load_commit(ctx, &root, &id) else {
            return super::ambiguous_argument(io, revision);
        };
        if let Some(annotation) = read_annotation(ctx, &root, revision) {
            io.out.extend_from_slice(
                format!(
                    "tag {revision}\nTagger: {} <{}>\nDate:   {}\n\n{}\n\n",
                    annotation.author_name,
                    annotation.author_email,
                    format_date(annotation.timestamp),
                    annotation.message
                )
                .as_bytes(),
            );
        }
        let decoration = if matches!(pretty, Pretty::Custom { .. }) {
            decorations(ctx, &root, &id)
        } else {
            String::new()
        };
        let now = now_seconds(ctx);
        io.out.extend_from_slice(
            render_commit_header(&id, &commit, &pretty, &decoration, now).as_bytes(),
        );
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
            io.out.push(b'\n');
            emit_commit_diff(ctx, &root, &id, &commit, format, &[], io);
        }
    }
    0
}

/// Read an annotated tag's message record, if `name` names one.
fn read_annotation(ctx: &CommandContext<'_>, root: &str, name: &str) -> Option<Commit> {
    repo::read_annotation(ctx, root, name)
}

// -- references -------------------------------------------------------------------------------

pub(crate) fn git_rev_parse(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut abbreviate: Option<usize> = None;
    let mut abbrev_ref = false;
    let mut full_name = false;
    let mut quiet = false;
    let mut revisions: Vec<String> = Vec::new();
    for argument in args {
        match argument.as_str() {
            "--show-toplevel" => {
                io.out.extend_from_slice(format!("{root}\n").as_bytes());
                return 0;
            }
            "--is-inside-work-tree" => {
                io.out.extend_from_slice(b"true\n");
                return 0;
            }
            "--is-inside-git-dir" => {
                let inside = repo::is_git_path(&root, &ctx.cwd);
                io.out
                    .extend_from_slice(if inside { b"true\n" } else { b"false\n" });
                return 0;
            }
            "--git-dir" => {
                // Git prints a path relative to the working directory when it can.
                let rendered = if ctx.cwd == root {
                    repo::GIT_DIR.to_string()
                } else {
                    repo::path_join(&root, repo::GIT_DIR)
                };
                io.out.extend_from_slice(format!("{rendered}\n").as_bytes());
                return 0;
            }
            "--absolute-git-dir" => {
                io.out.extend_from_slice(
                    format!("{}\n", repo::path_join(&root, repo::GIT_DIR)).as_bytes(),
                );
                return 0;
            }
            "--is-bare-repository" => {
                io.out.extend_from_slice(b"false\n");
                return 0;
            }
            "--show-cdup" => {
                let depth = repo::relative_path(&root, &ctx.cwd)
                    .map_or(0, |prefix| prefix.split('/').count());
                io.out
                    .extend_from_slice(format!("{}\n", "../".repeat(depth)).as_bytes());
                return 0;
            }
            "--show-prefix" => {
                let prefix = repo::relative_path(&root, &ctx.cwd).unwrap_or_default();
                if prefix.is_empty() {
                    io.out.push(b'\n');
                } else {
                    io.out.extend_from_slice(format!("{prefix}/\n").as_bytes());
                }
                return 0;
            }
            "--short" => abbreviate = Some(7),
            "--abbrev-ref" => abbrev_ref = true,
            "--symbolic-full-name" => full_name = true,
            "--verify" => {}
            "-q" | "--quiet" => quiet = true,
            value if value.starts_with("--short=") => {
                let Ok(length) = value["--short=".len()..].parse::<usize>() else {
                    return usage(io, "--short requires a length");
                };
                abbreviate = Some(length.clamp(4, 40));
            }
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported rev-parse option: {value}"))
            }
            value => revisions.push(value.to_string()),
        }
    }
    if revisions.is_empty() {
        revisions.push("HEAD".to_string());
    }
    for revision in &revisions {
        if full_name {
            let name = if revision == "HEAD" {
                repo::current_branch(ctx, &root).map_or_else(
                    || "HEAD".to_string(),
                    |branch| format!("refs/heads/{branch}"),
                )
            } else if revision.starts_with("refs/") {
                revision.clone()
            } else if repo::branch_names(ctx, &root).contains(revision) {
                format!("refs/heads/{revision}")
            } else if repo::reference_names(ctx, &root, "tags").contains(revision) {
                format!("refs/tags/{revision}")
            } else {
                String::new()
            };
            io.out.extend_from_slice(format!("{name}\n").as_bytes());
            continue;
        }
        if abbrev_ref {
            let name = if revision == "HEAD" {
                repo::current_branch(ctx, &root).unwrap_or_else(|| "HEAD".to_string())
            } else {
                revision.rsplit('/').next().unwrap_or(revision).to_string()
            };
            io.out.extend_from_slice(format!("{name}\n").as_bytes());
            continue;
        }
        let Some(commit) = rev_parse_object(ctx, &root, revision) else {
            if quiet {
                // `--quiet --verify` is the usual "does this reference exist?" probe.
                return 1;
            }
            return super::ambiguous_argument(io, revision);
        };
        let rendered = match abbreviate {
            Some(length) => commit[..length.min(commit.len())].to_string(),
            None => commit,
        };
        io.out.extend_from_slice(format!("{rendered}\n").as_bytes());
    }
    0
}

/// Resolve one `rev-parse` operand, which may name a commit, a tree, or a blob.
fn rev_parse_object(ctx: &mut CommandContext<'_>, root: &str, revision: &str) -> Option<String> {
    if let Some((base, path)) = revision.split_once(':') {
        let commit = repo::resolve_revision(ctx, root, base)?;
        let tree = repo::commit_tree(ctx, root, &commit)?;
        if path.is_empty() {
            return Some(repo::tree_hash(&tree));
        }
        return tree.get(path).map(|entry| entry.hash.clone());
    }
    if let Some(base) = revision.strip_suffix("^{tree}") {
        let commit = repo::resolve_revision(ctx, root, base)?;
        return Some(repo::tree_hash(&repo::commit_tree(ctx, root, &commit)?));
    }
    // `^{commit}` and `^{}` peel a tag to the commit it points at, which this subset already does.
    let revision = revision
        .strip_suffix("^{commit}")
        .or_else(|| revision.strip_suffix("^{}"))
        .unwrap_or(revision);
    repo::resolve_revision(ctx, root, revision)
}

pub(crate) fn git_rev_list(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut count = false;
    let mut limit = 10_000_usize;
    let mut revisions: Vec<String> = Vec::new();
    let mut all_references = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--count" => count = true,
            "--all" => all_references = true,
            "-n" | "--max-count" => {
                index += 1;
                let Some(value) = args.get(index).and_then(|value| value.parse().ok()) else {
                    return usage(io, "rev-list count must be a non-negative integer");
                };
                limit = value;
            }
            value if value.starts_with("--max-count=") => {
                let Ok(value) = value["--max-count=".len()..].parse() else {
                    return usage(io, "rev-list count must be a non-negative integer");
                };
                limit = value;
            }
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported rev-list option: {value}"))
            }
            value => revisions.push(value.to_string()),
        }
        index += 1;
    }
    if all_references {
        revisions.extend(all_reference_tips(ctx, &root));
    }
    if revisions.is_empty() {
        revisions.push("HEAD".to_string());
    }
    let Some(mut history) = history_for(ctx, &root, &revisions, false) else {
        return super::ambiguous_argument(io, &revisions.join(" "));
    };
    history.truncate(limit);
    if count {
        io.out
            .extend_from_slice(format!("{}\n", history.len()).as_bytes());
        return 0;
    }
    for (id, _) in history {
        io.out.extend_from_slice(format!("{id}\n").as_bytes());
    }
    0
}

fn valid_reference_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('-')
        && !name.starts_with('/')
        && !name.ends_with('/')
        && !name.contains("..")
        && !name.contains(char::is_whitespace)
        && !name.contains(['~', '^', ':', '?', '*', '[', '\\'])
}

pub(crate) fn git_branch(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut verbose = false;
    let mut delete = false;
    let mut rename = false;
    let mut force = false;
    let mut list = false;
    let mut contains: Option<(BranchFilter, String)> = None;
    let mut format: Option<String> = None;
    let mut operands: Vec<String> = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        index += 1;
        match argument.as_str() {
            "--show-current" => {
                if let Some(branch) = repo::current_branch(ctx, &root) {
                    io.out.extend_from_slice(format!("{branch}\n").as_bytes());
                }
                return 0;
            }
            "-v" | "-vv" | "--verbose" => verbose = true,
            "-d" | "--delete" => delete = true,
            "-D" => {
                delete = true;
                force = true;
            }
            "-m" | "--move" => rename = true,
            "-M" => {
                rename = true;
                force = true;
            }
            "-f" | "--force" => force = true,
            "-l" | "--list" => list = true,
            value
                if value.starts_with("--contains=")
                    || value.starts_with("--merged=")
                    || value.starts_with("--no-merged=") =>
            {
                let (name, revision) = value.split_once('=').unwrap_or((value, "HEAD"));
                contains = Some((branch_filter(name), revision.to_string()));
                list = true;
            }
            "--contains" | "--merged" | "--no-merged" => {
                // The revision is optional and defaults to HEAD.
                let revision = match args.get(index) {
                    Some(value) if !value.starts_with('-') => {
                        index += 1;
                        value.clone()
                    }
                    _ => "HEAD".to_string(),
                };
                contains = Some((branch_filter(argument), revision));
                list = true;
            }
            "-r" | "--remotes" => {
                // The simulation has no remotes, so there are no remote-tracking branches.
                return 0;
            }
            "-a" | "--all" | "--no-color" | "--no-column" => list = true,
            "-q" | "--quiet" => {}
            "--format" => {
                index += 1;
                format = args.get(index).cloned();
                list = true;
            }
            value if value.starts_with("--format=") => {
                format = Some(value["--format=".len()..].to_string());
                list = true;
            }
            value if value.starts_with("--sort=") => list = true,
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported branch option: {value}"))
            }
            value => operands.push(value.to_string()),
        }
    }
    if delete {
        if operands.is_empty() {
            return usage(io, "usage: git branch (-d|-D) NAME...");
        }
        return delete_branches(ctx, &root, &operands, force, io);
    }
    if rename {
        let (from, to) = match operands.as_slice() {
            [to] => (
                match repo::current_branch(ctx, &root) {
                    Some(branch) => branch,
                    None => return usage(io, "cannot rename a detached HEAD"),
                },
                to.clone(),
            ),
            [from, to] => (from.clone(), to.clone()),
            _ => return usage(io, "usage: git branch -m [OLD] NEW"),
        };
        return rename_branch(ctx, &root, &from, &to, force, io);
    }
    if operands.is_empty() || list {
        let pattern = operands.first().map(String::as_str);
        if let Some(format) = &format {
            return list_formatted_branches(ctx, &root, pattern, format, io);
        }
        return list_branches(ctx, &root, pattern, contains.as_ref(), verbose, io);
    }
    if operands.len() > 2 || !valid_reference_name(&operands[0]) {
        return usage(io, "usage: git branch NAME [START_POINT]");
    }
    let start = operands.get(1).map_or("HEAD", String::as_str);
    let Some(commit) = repo::resolve_revision(ctx, &root, start) else {
        io.err
            .extend_from_slice(format!("fatal: not a valid object name: '{start}'\n").as_bytes());
        return 128;
    };
    let reference = format!("refs/heads/{}", operands[0]);
    if !force && repo::read_reference(ctx, &root, &reference).is_some() {
        io.err.extend_from_slice(
            format!("fatal: a branch named '{}' already exists\n", operands[0]).as_bytes(),
        );
        return 128;
    }
    repo::write_reference(ctx, &root, &reference, &commit).map_or_else(
        |error| {
            io.err
                .extend_from_slice(format!("git branch: {error}\n").as_bytes());
            1
        },
        |()| 0,
    )
}

/// Which branches `--merged`, `--no-merged`, and `--contains` keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BranchFilter {
    /// The branch's tip is an ancestor of the named revision.
    Merged,
    /// It is not.
    NotMerged,
    /// The branch's own history includes the named revision, which is the other direction.
    Contains,
}

fn branch_filter(option: &str) -> BranchFilter {
    match option {
        "--contains" => BranchFilter::Contains,
        "--no-merged" => BranchFilter::NotMerged,
        _ => BranchFilter::Merged,
    }
}

/// Whether a branch survives `--merged`, `--no-merged`, or `--contains`.
fn branch_selected(
    ctx: &mut CommandContext<'_>,
    root: &str,
    branch: &str,
    filter: Option<&(BranchFilter, String)>,
) -> bool {
    let Some((filter, revision)) = filter else {
        return true;
    };
    let Some(target) = repo::resolve_revision(ctx, root, revision) else {
        return false;
    };
    let Some(tip) = repo::read_reference(ctx, root, &format!("refs/heads/{branch}")) else {
        return false;
    };
    match filter {
        BranchFilter::Merged => repo::ancestors(ctx, root, &target).contains(&tip),
        BranchFilter::NotMerged => !repo::ancestors(ctx, root, &target).contains(&tip),
        BranchFilter::Contains => repo::ancestors(ctx, root, &tip).contains(&target),
    }
}

fn list_branches(
    ctx: &mut CommandContext<'_>,
    root: &str,
    pattern: Option<&str>,
    contains: Option<&(BranchFilter, String)>,
    verbose: bool,
    io: &mut Io,
) -> i32 {
    let current = repo::current_branch(ctx, root);
    // A detached HEAD is listed first, as the checked-out "branch" it stands in for.
    let detached = match (&current, repo::head_commit(ctx, root)) {
        (None, Some(commit)) => Some(format!("(HEAD detached at {})", repo::short(&commit))),
        _ => None,
    };
    let branches = repo::branch_names(ctx, root);
    let width = branches
        .iter()
        .map(String::len)
        .chain(detached.iter().map(String::len))
        .max()
        .unwrap_or(0);
    if let Some(label) = &detached {
        if pattern.is_none() {
            if verbose {
                let commit = repo::head_commit(ctx, root).unwrap_or_default();
                let subject = repo::load_commit(ctx, root, &commit)
                    .map(|commit| commit.subject().to_string())
                    .unwrap_or_default();
                io.out.extend_from_slice(
                    format!("* {label:width$} {} {subject}\n", repo::short(&commit)).as_bytes(),
                );
            } else {
                io.out.extend_from_slice(format!("* {label}\n").as_bytes());
            }
        }
    }
    for branch in branches {
        if let Some(pattern) = pattern {
            if !crate::commands::util::glob_eq(pattern, &branch) {
                continue;
            }
        }
        if !branch_selected(ctx, root, &branch, contains) {
            continue;
        }
        let marker = if current.as_deref() == Some(branch.as_str()) {
            '*'
        } else {
            ' '
        };
        if !verbose {
            io.out
                .extend_from_slice(format!("{marker} {branch}\n").as_bytes());
            continue;
        }
        let summary = repo::read_reference(ctx, root, &format!("refs/heads/{branch}"))
            .map(|commit| {
                let subject = repo::load_commit(ctx, root, &commit)
                    .map(|commit| commit.subject().to_string())
                    .unwrap_or_default();
                format!("{} {subject}", repo::short(&commit))
            })
            .unwrap_or_default();
        io.out
            .extend_from_slice(format!("{marker} {branch:width$} {summary}\n").as_bytes());
    }
    0
}

/// List branches through a `--format` template.
fn list_formatted_branches(
    ctx: &mut CommandContext<'_>,
    root: &str,
    pattern: Option<&str>,
    format: &str,
    io: &mut Io,
) -> i32 {
    for branch in repo::branch_names(ctx, root) {
        if let Some(pattern) = pattern {
            if !crate::commands::util::glob_eq(pattern, &branch) {
                continue;
            }
        }
        let name = format!("refs/heads/{branch}");
        let Some(commit) = repo::read_reference(ctx, root, &name) else {
            continue;
        };
        let Some(line) = super::plumbing::expand_ref_format(format, &name, &commit) else {
            return usage(io, &format!("unsupported branch format: {format}"));
        };
        io.out.extend_from_slice(format!("{line}\n").as_bytes());
    }
    0
}

fn delete_branches(
    ctx: &mut CommandContext<'_>,
    root: &str,
    names: &[String],
    force: bool,
    io: &mut Io,
) -> i32 {
    let current = repo::current_branch(ctx, root);
    for name in names {
        if current.as_deref() == Some(name.as_str()) {
            io.err.extend_from_slice(
                format!("error: cannot delete branch '{name}' checked out\n").as_bytes(),
            );
            return 1;
        }
        let reference = format!("refs/heads/{name}");
        let Some(commit) = repo::read_reference(ctx, root, &reference) else {
            io.err
                .extend_from_slice(format!("error: branch '{name}' not found\n").as_bytes());
            return 1;
        };
        if !force {
            // Refuse to drop work that HEAD cannot reach, as Git's `-d` does.
            let reachable = repo::head_commit(ctx, root)
                .map(|head| repo::ancestors(ctx, root, &head))
                .unwrap_or_default();
            if !reachable.contains(&commit) {
                io.err.extend_from_slice(
                    format!(
                        "error: the branch '{name}' is not fully merged; use -D to delete it\n"
                    )
                    .as_bytes(),
                );
                return 1;
            }
        }
        if repo::delete_reference(ctx, root, &reference).is_err() {
            io.err
                .extend_from_slice(format!("error: branch '{name}' not found\n").as_bytes());
            return 1;
        }
        io.out.extend_from_slice(
            format!("Deleted branch {name} (was {}).\n", repo::short(&commit)).as_bytes(),
        );
    }
    0
}

fn rename_branch(
    ctx: &mut CommandContext<'_>,
    root: &str,
    from: &str,
    to: &str,
    force: bool,
    io: &mut Io,
) -> i32 {
    if !valid_reference_name(to) {
        return usage(io, &format!("invalid branch name: {to}"));
    }
    let Some(commit) = repo::read_reference(ctx, root, &format!("refs/heads/{from}")) else {
        io.err
            .extend_from_slice(format!("error: branch '{from}' not found\n").as_bytes());
        return 1;
    };
    if !force && repo::read_reference(ctx, root, &format!("refs/heads/{to}")).is_some() {
        io.err
            .extend_from_slice(format!("fatal: a branch named '{to}' already exists\n").as_bytes());
        return 128;
    }
    if repo::write_reference(ctx, root, &format!("refs/heads/{to}"), &commit).is_err()
        || repo::delete_reference(ctx, root, &format!("refs/heads/{from}")).is_err()
    {
        return 1;
    }
    if repo::current_branch(ctx, root).as_deref() == Some(from)
        && repo::set_head_to_branch(ctx, root, to, &format!("branch: renamed to {to}")).is_err()
    {
        return 1;
    }
    0
}

pub(crate) fn git_tag(
    ctx: &mut CommandContext<'_>,
    globals: &Globals,
    args: &[String],
    io: &mut Io,
) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut list = false;
    let mut delete = false;
    let mut force = false;
    let mut annotations = false;
    let mut message: Option<String> = None;
    let mut format: Option<String> = None;
    let mut points_at: Option<String> = None;
    let mut operands: Vec<String> = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-l" | "--list" => list = true,
            "-d" | "--delete" => delete = true,
            "-f" | "--force" => force = true,
            "-a" | "--annotate" => {}
            "-n" | "--list-annotations" => {
                list = true;
                annotations = true;
            }
            "-m" | "--message" => {
                index += 1;
                message = args.get(index).cloned();
            }
            value if value.starts_with("--message=") => {
                message = Some(value["--message=".len()..].to_string());
            }
            value if value.starts_with("-am") => {
                index += 1;
                message = args.get(index).cloned();
            }
            "-q" | "--quiet" => {}
            "--format" => {
                index += 1;
                format = args.get(index).cloned();
                list = true;
            }
            "--points-at" => {
                index += 1;
                points_at = args.get(index).cloned();
                list = true;
            }
            value if value.starts_with("--format=") => {
                format = Some(value["--format=".len()..].to_string());
                list = true;
            }
            value if value.starts_with("--points-at=") => {
                points_at = Some(value["--points-at=".len()..].to_string());
                list = true;
            }
            value if value.starts_with("--sort=") => list = true,
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported tag option: {value}"))
            }
            value => operands.push(value.to_string()),
        }
        index += 1;
    }
    if delete {
        for name in &operands {
            let was =
                repo::read_reference(ctx, &root, &format!("refs/tags/{name}")).unwrap_or_default();
            if repo::delete_reference(ctx, &root, &format!("refs/tags/{name}")).is_err() {
                io.err
                    .extend_from_slice(format!("error: tag '{name}' not found.\n").as_bytes());
                return 1;
            }
            io.out.extend_from_slice(
                format!("Deleted tag '{name}' (was {})\n", repo::short(&was)).as_bytes(),
            );
        }
        return 0;
    }
    if operands.is_empty() || list {
        // `--points-at` keeps only the tags on one commit.
        let target = match points_at {
            Some(revision) => match repo::resolve_revision(ctx, &root, &revision) {
                Some(commit) => Some(commit),
                None => return super::ambiguous_argument(io, &revision),
            },
            None => None,
        };
        for name in repo::reference_names(ctx, &root, "tags") {
            if let Some(pattern) = operands.first() {
                if !crate::commands::util::glob_eq(pattern, &name) {
                    continue;
                }
            }
            let reference = format!("refs/tags/{name}");
            let commit = repo::read_reference(ctx, &root, &reference).unwrap_or_default();
            if target.as_ref().is_some_and(|target| *target != commit) {
                continue;
            }
            if let Some(format) = &format {
                let Some(line) = super::plumbing::expand_ref_format(format, &reference, &commit)
                else {
                    return usage(io, &format!("unsupported tag format: {format}"));
                };
                io.out.extend_from_slice(format!("{line}\n").as_bytes());
                continue;
            }
            match repo::read_annotation(ctx, &root, &name).filter(|_| annotations) {
                Some(annotation) => io
                    .out
                    .extend_from_slice(format!("{name:<15} {}\n", annotation.subject()).as_bytes()),
                None => io.out.extend_from_slice(format!("{name}\n").as_bytes()),
            }
        }
        return 0;
    }
    let name = &operands[0];
    if !valid_reference_name(name) {
        return usage(io, &format!("invalid tag name: {name}"));
    }
    let start = operands.get(1).map_or("HEAD", String::as_str);
    let Some(commit) = repo::resolve_revision(ctx, &root, start) else {
        io.err
            .extend_from_slice(format!("fatal: not a valid object name: '{start}'\n").as_bytes());
        return 128;
    };
    let reference = format!("refs/tags/{name}");
    if !force && repo::read_reference(ctx, &root, &reference).is_some() {
        io.err
            .extend_from_slice(format!("fatal: tag '{name}' already exists\n").as_bytes());
        return 128;
    }
    if repo::write_reference(ctx, &root, &reference, &commit).is_err() {
        return 1;
    }
    if let Some(message) = message {
        // The annotation is stored beside the ref; the ref itself stays lightweight.
        let (author_name, author_email) = author_identity(ctx, &root, globals);
        let annotation = Commit {
            parents: Vec::new(),
            author_name,
            author_email,
            timestamp: now_seconds(ctx),
            message,
        };
        if repo::write_annotation(ctx, &root, name, &annotation).is_err() {
            return 1;
        }
    }
    0
}

// -- switching branches -----------------------------------------------------------------------

/// The tracked paths that differ from HEAD, which is what blocks a branch switch.
///
/// Untracked files never block a switch, so they are not considered here.
pub(crate) fn blocking_changes(
    ctx: &mut CommandContext<'_>,
    root: &str,
    target: &repo::Tree,
) -> Vec<String> {
    let index = repo::load_index(ctx, root).unwrap_or_default();
    let Ok(work) = repo::collect_working_tree(ctx, root) else {
        return vec!["<unreadable working tree>".to_string()];
    };
    let files = work.release(ctx);
    let head = repo::head_tree(ctx, root);
    let mut blocked: BTreeSet<String> = BTreeSet::new();
    for (path, hash) in &index {
        if files.get(path) != Some(hash) || head.get(path) != Some(hash) {
            blocked.insert(path.clone());
        }
    }
    for path in head.keys() {
        if !index.contains_key(path) {
            blocked.insert(path.clone());
        }
    }
    // Only the paths the move would actually rewrite can lose work.
    blocked
        .into_iter()
        .filter(|path| target.get(path) != head.get(path))
        .collect()
}

/// Move HEAD and the working tree to `commit`.
fn checkout_commit(
    ctx: &mut CommandContext<'_>,
    root: &str,
    commit: &str,
    io: &mut Io,
) -> Result<(), i32> {
    if let Some(previous) = repo::current_branch(ctx, root) {
        let _ = repo::write_vfs(
            ctx,
            &repo::git_path(root, "PREV_HEAD"),
            format!("{previous}\n").as_bytes(),
        );
    }
    let old = repo::head_tree(ctx, root);
    let Some(new) = repo::commit_tree(ctx, root, commit) else {
        return Err(usage(io, "target revision has an invalid commit"));
    };
    let blocked = blocking_changes(ctx, root, &new);
    if !blocked.is_empty() {
        io.err.extend_from_slice(
            b"error: Your local changes to the following files would be overwritten by checkout:\n",
        );
        for path in blocked {
            io.err.extend_from_slice(format!("\t{path}\n").as_bytes());
        }
        io.err.extend_from_slice(
            b"Please commit your changes or stash them before you switch branches.\nAborting\n",
        );
        return Err(1);
    }
    if let Err(error) = repo::update_work_tree(ctx, root, &old, &new) {
        io.err
            .extend_from_slice(format!("git switch: {error}\n").as_bytes());
        return Err(1);
    }
    if repo::store_index(ctx, root, &new).is_err() {
        return Err(1);
    }
    Ok(())
}

fn switch_to_branch(ctx: &mut CommandContext<'_>, root: &str, branch: &str, io: &mut Io) -> i32 {
    let Some(commit) = repo::read_reference(ctx, root, &format!("refs/heads/{branch}")) else {
        io.err
            .extend_from_slice(format!("fatal: invalid reference: {branch}\n").as_bytes());
        return 128;
    };
    if repo::current_branch(ctx, root).as_deref() == Some(branch) {
        io.err
            .extend_from_slice(format!("Already on '{branch}'\n").as_bytes());
        return 0;
    }
    if let Err(status) = checkout_commit(ctx, root, &commit, io) {
        return status;
    }
    let from = repo::current_branch(ctx, root).unwrap_or_else(|| "HEAD".to_string());
    let action = format!("checkout: moving from {from} to {branch}");
    if repo::set_head_to_branch(ctx, root, branch, &action).is_err() {
        return 1;
    }
    io.err
        .extend_from_slice(format!("Switched to branch '{branch}'\n").as_bytes());
    0
}

fn switch_detached(ctx: &mut CommandContext<'_>, root: &str, revision: &str, io: &mut Io) -> i32 {
    let Some(commit) = repo::resolve_revision(ctx, root, revision) else {
        io.err
            .extend_from_slice(format!("fatal: invalid reference: {revision}\n").as_bytes());
        return 128;
    };
    if let Err(status) = checkout_commit(ctx, root, &commit, io) {
        return status;
    }
    if repo::set_head_detached(
        ctx,
        root,
        &commit,
        &format!("checkout: moving to {revision}"),
    )
    .is_err()
    {
        return 1;
    }
    io.err.extend_from_slice(
        format!(
            "Note: switching to '{revision}'.\nHEAD is now at {} {}\n",
            repo::short(&commit),
            repo::load_commit(ctx, root, &commit)
                .map(|commit| commit.subject().to_string())
                .unwrap_or_default()
        )
        .as_bytes(),
    );
    0
}

/// Create `branch` at `start` and switch to it.
fn create_and_switch(
    ctx: &mut CommandContext<'_>,
    root: &str,
    branch: &str,
    start: Option<&str>,
    force: bool,
    io: &mut Io,
) -> i32 {
    let mut arguments = Vec::new();
    if force {
        arguments.push("-f".to_string());
    }
    arguments.push(branch.to_string());
    if let Some(start) = start {
        arguments.push(start.to_string());
    }
    let status = git_branch(ctx, &arguments, io);
    if status != 0 {
        return status;
    }
    let status = switch_to_branch(ctx, root, branch, io);
    if status == 0 {
        // Git reports creation rather than a plain switch.
        let switched = format!("Switched to branch '{branch}'\n");
        let created = format!("Switched to a new branch '{branch}'\n");
        replace_tail(io.err, &switched, &created);
    }
    status
}

fn replace_tail(buffer: &mut Vec<u8>, from: &str, to: &str) {
    if buffer.ends_with(from.as_bytes()) {
        buffer.truncate(buffer.len() - from.len());
        buffer.extend_from_slice(to.as_bytes());
    }
}

pub(crate) fn git_switch(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut create = false;
    let mut force = false;
    let mut detach = false;
    let mut operands: Vec<String> = Vec::new();
    let args = super::expand_clusters(args, "qcCdf");
    for argument in &args {
        match argument.as_str() {
            "-c" | "--create" => create = true,
            "-C" | "--force-create" => {
                create = true;
                force = true;
            }
            "-d" | "--detach" => detach = true,
            "-f" | "--force" | "--discard-changes" => force = true,
            "-q" | "--quiet" | "--no-guess" | "--no-track" => {}
            // A lone `-` names the previously checked-out branch.
            value if value.starts_with('-') && value != "-" => {
                return usage(io, &format!("unsupported switch option: {value}"))
            }
            value => operands.push(value.to_string()),
        }
    }
    match operands.as_slice() {
        [branch] if create => create_and_switch(ctx, &root, branch, None, force, io),
        [branch, start] if create => create_and_switch(ctx, &root, branch, Some(start), force, io),
        [revision] if detach => switch_detached(ctx, &root, revision, io),
        [branch] => {
            let branch = resolve_previous(ctx, &root, branch);
            match branch {
                Some(branch) => switch_to_branch(ctx, &root, &branch, io),
                None => {
                    io.err
                        .extend_from_slice(b"fatal: no previous branch to switch to\n");
                    128
                }
            }
        }
        _ => usage(io, "usage: git switch [-c] BRANCH [START_POINT]"),
    }
}

/// Expand `-` into the branch that was checked out before the current one.
fn resolve_previous(ctx: &CommandContext<'_>, root: &str, branch: &str) -> Option<String> {
    if branch != "-" {
        return Some(branch.to_string());
    }
    repo::previous_branch(ctx, root)
}

pub(crate) fn git_checkout(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut create = false;
    let mut force = false;
    let mut detach = false;
    let mut operands: Vec<String> = Vec::new();
    let mut paths: Vec<String> = Vec::new();
    let mut operands_only = false;
    let mut side = None;
    let args = super::expand_clusters(args, "qbBf");
    for argument in &args {
        if operands_only {
            paths.push(argument.clone());
            continue;
        }
        match argument.as_str() {
            "--" => operands_only = true,
            "-b" => create = true,
            "-B" => {
                create = true;
                force = true;
            }
            "--detach" => detach = true,
            "-f" | "--force" => force = true,
            "-q" | "--quiet" => {}
            "--ours" => side = Some(Side::Ours),
            "--theirs" => side = Some(Side::Theirs),
            value if value.starts_with('-') && value != "-" => {
                return usage(io, &format!("unsupported checkout option: {value}"))
            }
            value => operands.push(value.to_string()),
        }
    }
    if let Some(side) = side {
        let named = if paths.is_empty() { &operands } else { &paths };
        return checkout_side(ctx, &root, side, named, io);
    }
    if !paths.is_empty() {
        // `git checkout [REVISION] -- PATH...` restores files without moving HEAD.
        let source = operands.first().map(String::as_str);
        return super::worktree::git_restore(ctx, &restore_arguments(source, &paths), io);
    }
    match operands.as_slice() {
        [branch] if create => create_and_switch(ctx, &root, branch, None, force, io),
        [branch, start] if create => create_and_switch(ctx, &root, branch, Some(start), force, io),
        [revision] if detach => switch_detached(ctx, &root, revision, io),
        [target] => {
            if let Some(branch) = resolve_previous(ctx, &root, target) {
                if repo::read_reference(ctx, &root, &format!("refs/heads/{branch}")).is_some() {
                    return switch_to_branch(ctx, &root, &branch, io);
                }
            }
            if repo::resolve_revision(ctx, &root, target).is_some() {
                return switch_detached(ctx, &root, target, io);
            }
            if !super::names_a_path(ctx, &root, target) {
                io.err.extend_from_slice(
                    format!("error: pathspec '{target}' did not match any file(s) known to git\n")
                        .as_bytes(),
                );
                return 1;
            }
            // A bare path argument means "discard my changes to that path".
            super::worktree::git_restore(
                ctx,
                &restore_arguments(None, std::slice::from_ref(target)),
                io,
            )
        }
        _ => usage(
            io,
            "usage: git checkout [-b] BRANCH | [REVISION] -- PATH...",
        ),
    }
}

/// Take one side of a conflict for the named paths, as `git checkout --ours` does.
pub(crate) fn restore_side(
    ctx: &mut CommandContext<'_>,
    theirs: bool,
    named: &[String],
    io: &mut Io,
) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let side = if theirs { Side::Theirs } else { Side::Ours };
    checkout_side(ctx, &root, side, named, io)
}

/// Which recorded side of a conflict `git checkout --ours` and `--theirs` take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Ours,
    Theirs,
}

/// Resolve a conflict by taking one side's recorded content for the named paths.
///
/// As in Git the path stays unmerged until it is staged, so the user can look before committing.
fn checkout_side(
    ctx: &mut CommandContext<'_>,
    root: &str,
    side: Side,
    named: &[String],
    io: &mut Io,
) -> i32 {
    let stages = conflict::load_stages(ctx, root);
    if named.is_empty() {
        return usage(io, "usage: git checkout --ours|--theirs PATH...");
    }
    let cwd = ctx.cwd.clone();
    let mut chosen: Vec<(String, String)> = Vec::new();
    for operand in named {
        let path = super::pathspec(&cwd, root, operand);
        let Some(entry) = stages.get(&path) else {
            io.err.extend_from_slice(
                format!("error: path '{operand}' does not have their version\n").as_bytes(),
            );
            return 1;
        };
        let hash = match side {
            Side::Ours => entry.ours.clone(),
            Side::Theirs => entry.theirs.clone(),
        };
        // A side that deleted the file has nothing to check out, which Git reports the same way.
        let Some(hash) = hash else {
            io.err.extend_from_slice(
                format!("error: path '{operand}' does not have their version\n").as_bytes(),
            );
            return 1;
        };
        chosen.push((path, hash));
    }
    let count = chosen.len();
    for (path, hash) in chosen {
        let Some(data) = repo::read_blob(ctx, root, &hash) else {
            io.err
                .extend_from_slice(format!("error: missing blob for '{path}'\n").as_bytes());
            return 1;
        };
        if repo::write_vfs(ctx, &repo::path_join(root, &path), &data).is_err() {
            io.err
                .extend_from_slice(format!("error: cannot write '{path}'\n").as_bytes());
            return 1;
        }
    }
    let plural = if count == 1 { "" } else { "s" };
    io.out
        .extend_from_slice(format!("Updated {count} path{plural} from the index\n").as_bytes());
    0
}

/// Translate a `git checkout [REVISION] -- PATH...` invocation into `git restore` arguments.
///
/// Checking a path out of a named revision also updates the index, which is why the staged and
/// working-tree flags are both set in that case.
fn restore_arguments(source: Option<&str>, paths: &[String]) -> Vec<String> {
    let mut arguments = Vec::new();
    if let Some(source) = source {
        arguments.push("--source".to_string());
        arguments.push(source.to_string());
        arguments.push("--staged".to_string());
        arguments.push("--worktree".to_string());
    }
    arguments.push("--".to_string());
    arguments.extend(paths.iter().cloned());
    arguments
}

// -- merging ----------------------------------------------------------------------------------

pub(crate) fn git_merge(
    ctx: &mut CommandContext<'_>,
    globals: &Globals,
    args: &[String],
    io: &mut Io,
) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut no_fast_forward = false;
    let mut fast_forward_only = false;
    let mut message = None;
    let mut operands: Vec<String> = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--no-ff" => no_fast_forward = true,
            "--ff-only" => fast_forward_only = true,
            "--ff" | "--no-edit" | "-q" | "--quiet" => {}
            "--abort" => return abort_pending(ctx, &root, "merge", io),
            "--continue" => return continue_pending(ctx, &root, globals, "merge", io),
            "-m" => {
                index += 1;
                message = args.get(index).cloned();
            }
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported merge option: {value}"))
            }
            value => operands.push(value.to_string()),
        }
        index += 1;
    }
    let [target] = operands.as_slice() else {
        return usage(io, "usage: git merge [--no-ff|--ff-only] BRANCH");
    };
    let Some(other) = repo::resolve_revision(ctx, &root, target) else {
        io.err.extend_from_slice(
            format!("merge: {target} - not something we can merge\n").as_bytes(),
        );
        return 1;
    };
    let Some(head) = repo::head_commit(ctx, &root) else {
        io.err
            .extend_from_slice(b"fatal: no commit on the current branch to merge into\n");
        return 128;
    };
    if repo::ancestors(ctx, &root, &head).contains(&other) {
        io.out.extend_from_slice(b"Already up to date.\n");
        return 0;
    }
    let incoming = repo::commit_tree(ctx, &root, &other).unwrap_or_default();
    if !blocking_changes(ctx, &root, &incoming).is_empty() {
        io.err
            .extend_from_slice(b"error: Your local changes would be overwritten by merge.\n");
        return 1;
    }
    let base = repo::merge_base(ctx, &root, &head, &other);
    let fast_forward = base.as_deref() == Some(head.as_str());
    if fast_forward && !no_fast_forward {
        if let Err(status) = checkout_commit(ctx, &root, &other, io) {
            return status;
        }
        if repo::update_head(ctx, &root, &other, &format!("merge {target}: Fast-forward")).is_err()
        {
            return 1;
        }
        io.out.extend_from_slice(
            format!(
                "Updating {}..{}\nFast-forward\n",
                repo::short(&head),
                repo::short(&other)
            )
            .as_bytes(),
        );
        let before = repo::commit_tree(ctx, &root, &head).unwrap_or_default();
        let after = repo::commit_tree(ctx, &root, &other).unwrap_or_default();
        let options = Options {
            format: Format::Stat,
            ..Options::default()
        };
        compare::emit(ctx, &root, &before, &after, &options, io);
        emit_mode_lines(&before, &after, io);
        return 0;
    }
    if fast_forward_only {
        io.err
            .extend_from_slice(b"fatal: Not possible to fast-forward, aborting.\n");
        return 128;
    }
    let base_tree = base
        .as_deref()
        .and_then(|base| repo::commit_tree(ctx, &root, base))
        .unwrap_or_default();
    let head_tree = repo::commit_tree(ctx, &root, &head).unwrap_or_default();
    let other_tree = repo::commit_tree(ctx, &root, &other).unwrap_or_default();
    let subject = message.unwrap_or_else(|| {
        // Git names the branch merged into unless it is the repository's default.
        match repo::current_branch(ctx, &root).filter(|branch| branch != repo::DEFAULT_BRANCH) {
            Some(branch) => format!("Merge branch '{target}' into {branch}"),
            None => format!("Merge branch '{target}'"),
        }
    });
    let combined = conflict::combine(
        ctx,
        &root,
        &base_tree,
        &head_tree,
        &other_tree,
        "HEAD",
        target,
    );
    let merged = combined.tree;
    if let Err(error) = repo::update_work_tree(ctx, &root, &head_tree, &merged) {
        io.err
            .extend_from_slice(format!("git merge: {error}\n").as_bytes());
        return 1;
    }
    if repo::store_index(ctx, &root, &merged).is_err() {
        return 1;
    }
    if !combined.stages.is_empty() {
        return pause_for_conflicts(
            ctx,
            &root,
            conflict::MERGE_HEAD,
            &other,
            &subject,
            target,
            &combined.stages,
            "Automatic merge failed; fix conflicts and then commit the result.",
            io,
        );
    }
    let (author_name, author_email) = author_identity(ctx, &root, globals);
    let commit = Commit {
        parents: vec![head.clone(), other.clone()],
        author_name,
        author_email,
        timestamp: now_seconds(ctx),
        message: subject,
    };
    let Ok(id) = repo::store_commit(ctx, &root, &commit, &merged) else {
        return 1;
    };
    if repo::update_head(ctx, &root, &id, &format!("merge {target}")).is_err() {
        return 1;
    }
    io.out
        .extend_from_slice(b"Merge made by the 'ort' strategy.\n");
    // Git shows the per-file diffstat of the merge before the summary.
    let stat = compare::Options {
        format: Format::Stat,
        ..Default::default()
    };
    compare::emit(ctx, &root, &head_tree, &merged, &stat, io);
    emit_mode_lines(&head_tree, &merged, io);
    0
}

// -- replaying single commits -----------------------------------------------------------------

/// Apply one commit's change to the current branch, or undo it.
///
/// Both commands are the same three-way merge with different corners: a cherry-pick treats the
/// commit's parent as the base and the commit as the incoming side, and a revert swaps those two,
/// which turns replaying a change into undoing it.
pub(crate) fn git_replay(
    ctx: &mut CommandContext<'_>,
    globals: &Globals,
    revert: bool,
    args: &[String],
    io: &mut Io,
) -> i32 {
    let name = if revert { "revert" } else { "cherry-pick" };
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut no_commit = false;
    let mut operands: Vec<String> = Vec::new();
    for argument in super::expand_clusters(args, "ne") {
        match argument.as_str() {
            "-n" | "--no-commit" => no_commit = true,
            "-e" | "--edit" | "--no-edit" | "-q" | "--quiet" => {}
            "--abort" | "--quit" => return abort_pending(ctx, &root, name, io),
            "--continue" => return continue_pending(ctx, &root, globals, name, io),
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported {name} option: {value}"))
            }
            value => operands.push(value.to_string()),
        }
    }
    if operands.is_empty() {
        return usage(io, &format!("usage: git {name} [-n] COMMIT..."));
    }
    for revision in &operands {
        let status = replay_one(ctx, globals, &root, name, revert, no_commit, revision, io);
        if status != 0 {
            return status;
        }
    }
    0
}

#[allow(clippy::too_many_arguments)]
fn replay_one(
    ctx: &mut CommandContext<'_>,
    globals: &Globals,
    root: &str,
    name: &str,
    revert: bool,
    no_commit: bool,
    revision: &str,
    io: &mut Io,
) -> i32 {
    if pending_operation(ctx, root).is_some() {
        io.err.extend_from_slice(
            format!(
                "error: a {name} is already in progress\nhint: try \"git {name} --continue\" or \"git {name} --abort\"\n"
            )
            .as_bytes(),
        );
        return 128;
    }
    let Some(id) = repo::resolve_revision(ctx, root, revision) else {
        io.err
            .extend_from_slice(format!("fatal: bad revision '{revision}'\n").as_bytes());
        return 128;
    };
    let Some(commit) = repo::load_commit(ctx, root, &id) else {
        io.err
            .extend_from_slice(format!("fatal: bad object {revision}\n").as_bytes());
        return 128;
    };
    if commit.parents.len() > 1 {
        io.err.extend_from_slice(
            format!(
                "error: commit {id} is a merge but no -m option was given.\nfatal: {name} failed\n"
            )
            .as_bytes(),
        );
        return 128;
    }
    let Some(head) = repo::head_commit(ctx, root) else {
        io.err
            .extend_from_slice(format!("fatal: {name} needs a commit to apply onto\n").as_bytes());
        return 128;
    };
    let commit_tree = repo::commit_tree(ctx, root, &id).unwrap_or_default();
    let parent_tree = commit
        .parents
        .first()
        .and_then(|parent| repo::commit_tree(ctx, root, parent))
        .unwrap_or_default();
    let head_tree = repo::commit_tree(ctx, root, &head).unwrap_or_default();
    let (base, theirs) = if revert {
        (&commit_tree, &parent_tree)
    } else {
        (&parent_tree, &commit_tree)
    };
    // Committing the replay would fold anything already staged into it, which is why Git wants
    // a settled index first. `-n` leaves the commit to the user, so it can go ahead.
    let staged = repo::load_index(ctx, root).unwrap_or_default();
    let dirty = !no_commit && staged != head_tree;
    if dirty || !blocking_changes(ctx, root, theirs).is_empty() {
        io.err.extend_from_slice(
            format!(
                "error: your local changes would be overwritten by {name}.\nhint: commit your changes or stash them to proceed.\nfatal: {name} failed\n"
            )
            .as_bytes(),
        );
        return 128;
    }
    let message = if revert {
        format!(
            "Revert \"{}\"\n\nThis reverts commit {id}.",
            commit.subject()
        )
    } else {
        commit.message.clone()
    };
    let label = format!("{} ({})", repo::short(&id), commit.subject());
    let combined = conflict::combine(ctx, root, base, &head_tree, theirs, "HEAD", &label);
    let applied = combined.tree;
    if let Err(error) = repo::update_work_tree(ctx, root, &head_tree, &applied) {
        io.err
            .extend_from_slice(format!("git {name}: {error}\n").as_bytes());
        return 1;
    }
    // Only the paths the replay touched move in the index; anything else staged is left alone.
    let mut index = staged;
    for path in head_tree.keys().chain(applied.keys()) {
        if head_tree.get(path) == applied.get(path) {
            continue;
        }
        match applied.get(path) {
            Some(entry) => index.insert(path.clone(), entry.clone()),
            None => index.remove(path),
        };
    }
    if repo::store_index(ctx, root, &index).is_err() {
        return 1;
    }
    let kind = if revert {
        conflict::REVERT_HEAD
    } else {
        conflict::CHERRY_PICK_HEAD
    };
    if !combined.stages.is_empty() {
        let advice = format!(
            "error: could not apply {label}\n\
             hint: After resolving the conflicts, mark them with\n\
             hint: \"git add/rm <pathspec>\", then run \"git {name} --continue\"."
        );
        return pause_for_conflicts(
            ctx,
            root,
            kind,
            &id,
            &message,
            &label,
            &combined.stages,
            &advice,
            io,
        );
    }
    if no_commit {
        // `-n` leaves the change staged for the user to commit; Git records nothing in progress.
        return 0;
    }
    if applied == head_tree {
        io.err.extend_from_slice(
            format!("The previous cherry-pick is now empty, possibly due to conflict resolution.\nfatal: {name} failed\n")
                .as_bytes(),
        );
        return 1;
    }
    // A cherry-pick keeps the original author; a revert is the work of whoever ran it.
    let (author_name, author_email) = if revert {
        author_identity(ctx, root, globals)
    } else {
        (commit.author_name.clone(), commit.author_email.clone())
    };
    let replayed = Commit {
        parents: vec![head.clone()],
        author_name,
        author_email,
        timestamp: now_seconds(ctx),
        message,
    };
    let Ok(new_id) = repo::store_commit(ctx, root, &replayed, &applied) else {
        return 1;
    };
    if repo::update_head(
        ctx,
        root,
        &new_id,
        &format!("{name}: {}", replayed.subject()),
    )
    .is_err()
    {
        return 1;
    }
    let branch = repo::current_branch(ctx, root).unwrap_or_else(|| "detached HEAD".to_string());
    io.out.extend_from_slice(
        format!(
            "[{branch} {}] {}\n",
            repo::short(&new_id),
            replayed.subject()
        )
        .as_bytes(),
    );
    let stat = compare::Options {
        format: Format::Stat,
        ..Default::default()
    };
    compare::emit(ctx, root, &head_tree, &applied, &stat, io);
    0
}

/// Record an unfinished merge, cherry-pick, or revert and report the paths left to the user.
#[allow(clippy::too_many_arguments)]
fn pause_for_conflicts(
    ctx: &mut CommandContext<'_>,
    root: &str,
    kind: &str,
    commit: &str,
    message: &str,
    theirs_label: &str,
    stages: &Stages,
    advice: &str,
    io: &mut Io,
) -> i32 {
    for (path, entry) in stages {
        // A delete on one side needs the longer sentence, because the file left in the working
        // tree is not the one the user was last looking at.
        let line = match entry.porcelain() {
            "UD" => format!(
                "CONFLICT (modify/delete): {path} deleted in {theirs_label} and modified in HEAD.  Version HEAD of {path} left in tree.\n"
            ),
            "DU" => format!(
                "CONFLICT (modify/delete): {path} deleted in HEAD and modified in {theirs_label}.  Version {theirs_label} of {path} left in tree.\n"
            ),
            "AA" => format!("CONFLICT (add/add): Merge conflict in {path}\n"),
            _ => format!("CONFLICT (content): Merge conflict in {path}\n"),
        };
        io.err.extend_from_slice(line.as_bytes());
    }
    conflict::begin(ctx, root, kind, commit, message);
    if !conflict::store_stages(ctx, root, stages) {
        return 1;
    }
    io.err.extend_from_slice(format!("{advice}\n").as_bytes());
    1
}

/// The operation waiting to be finished, if any.
fn pending_operation(ctx: &CommandContext<'_>, root: &str) -> Option<(&'static str, String)> {
    for kind in [
        conflict::MERGE_HEAD,
        conflict::CHERRY_PICK_HEAD,
        conflict::REVERT_HEAD,
    ] {
        if let Some(commit) = conflict::in_progress(ctx, root, kind) {
            return Some((kind, commit));
        }
    }
    None
}

/// Throw away an unfinished merge, cherry-pick, or revert.
fn abort_pending(ctx: &mut CommandContext<'_>, root: &str, name: &str, io: &mut Io) -> i32 {
    if pending_operation(ctx, root).is_none() {
        io.err.extend_from_slice(
            format!("fatal: There is no {name} in progress ({name} --abort).\n").as_bytes(),
        );
        return 128;
    }
    let head = repo::head_tree(ctx, root);
    // Only paths the merge could have touched are restored; untracked files are left alone.
    let mut previous = repo::load_index(ctx, root).unwrap_or_default();
    for (path, hash) in &head {
        previous.entry(path.clone()).or_insert_with(|| hash.clone());
    }
    if let Err(error) = repo::replace_work_tree(ctx, root, &previous, &head) {
        io.err
            .extend_from_slice(format!("git {name}: {error}\n").as_bytes());
        return 1;
    }
    if repo::store_index(ctx, root, &head).is_err() {
        return 1;
    }
    conflict::clear(ctx, root);
    0
}

/// Finish an operation whose conflicts the user has resolved.
fn continue_pending(
    ctx: &mut CommandContext<'_>,
    root: &str,
    globals: &Globals,
    name: &str,
    io: &mut Io,
) -> i32 {
    if pending_operation(ctx, root).is_none() {
        io.err.extend_from_slice(
            format!("fatal: There is no {name} in progress ({name} --continue).\n").as_bytes(),
        );
        return 128;
    }
    if !conflict::load_stages(ctx, root).is_empty() {
        io.err.extend_from_slice(
            b"error: Committing is not possible because you have unmerged files.\n",
        );
        return 1;
    }
    git_commit(ctx, globals, &["--no-edit".to_string()], io)
}
