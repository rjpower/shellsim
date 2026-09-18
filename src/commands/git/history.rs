//! Commit, history, reference, and branch-switching porcelain.
//!
//! Commits are linear or two-parent; the subset has no rebase, cherry-pick, or reflog. Merges are
//! fast-forward when possible and otherwise combine two trees file by file, refusing the merge
//! when the same file changed on both sides rather than writing conflict markers a simulated
//! resolver could not help with.

use std::collections::BTreeSet;

use crate::commands::{CommandContext, Io};

use super::compare::{self, Format, Options, RightSide};
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

fn now_seconds(ctx: &CommandContext<'_>) -> i64 {
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
    let args = super::expand_clusters(args, "amqvns");
    let mut index = 0;
    while index < args.len() {
        let argument = args[index].as_str();
        match argument {
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
            _ => return usage(io, "committing a pathspec is not supported; stage it first"),
        }
        index += 1;
    }
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    if stage_tracked {
        if let Err(status) = worktree::stage_tracked_changes(ctx, &root, io) {
            return status;
        }
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
        (_, Some((id, _))) => (
            vec![id.clone()],
            repo::commit_tree(ctx, &root, id).unwrap_or_default(),
        ),
        _ => (Vec::new(), Tree::new()),
    };
    if index_tree == baseline && !allow_empty {
        emit_nothing_to_commit(ctx, io);
        return 1;
    }
    if signoff {
        let (name, email) = author_identity(ctx, &root, globals);
        message.push_str(&format!("\n\nSigned-off-by: {name} <{email}>"));
    }
    if dry_run {
        return super::worktree::git_status(ctx, &["--short".to_string()], io);
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
            .map_or_else(|| now_seconds(ctx), |(_, commit)| commit.timestamp),
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
    if repo::update_head(ctx, &root, &id).is_err() {
        return 1;
    }
    if quiet {
        return 0;
    }
    let label = repo::current_branch(ctx, &root)
        .unwrap_or_else(|| format!("detached HEAD {}", repo::short(&id)));
    let root_commit = if commit.parents.is_empty() {
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
    for path in &changed {
        let before = old
            .get(*path)
            .and_then(|hash| repo::read_blob(ctx, root, hash));
        let after = new
            .get(*path)
            .and_then(|hash| repo::read_blob(ctx, root, hash));
        let (added, removed) = super::diff::change_counts(before.as_deref(), after.as_deref());
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
        match (old.contains_key(path), new.contains_key(path)) {
            (false, true) => io
                .out
                .extend_from_slice(format!(" create mode 100644 {path}\n").as_bytes()),
            (true, false) => io
                .out
                .extend_from_slice(format!(" delete mode 100644 {path}\n").as_bytes()),
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

fn render_commit_header(id: &str, commit: &Commit, pretty: &Pretty, decoration: &str) -> String {
    let identity = format!("{} <{}>", commit.author_name, commit.author_email);
    match pretty {
        Pretty::OneLine => format!("{id}{decoration} {}\n", commit.subject()),
        Pretty::AbbreviatedOneLine => {
            format!("{}{decoration} {}\n", repo::short(id), commit.subject())
        }
        Pretty::Short => format!(
            "commit {id}{decoration}\nAuthor: {identity}\n\n{}",
            indent(commit.subject())
        ),
        Pretty::Medium => format!(
            "commit {id}{decoration}\nAuthor: {identity}\nDate:   {}\n\n{}",
            format_date(commit.timestamp),
            indent(&commit.message)
        ),
        Pretty::Full => format!(
            "commit {id}{decoration}\nAuthor: {identity}\nCommit: {identity}\n\n{}",
            indent(&commit.message)
        ),
        Pretty::Fuller => format!(
            "commit {id}{decoration}\nAuthor:     {identity}\nAuthorDate: {}\nCommit:     {identity}\nCommitDate: {}\n\n{}",
            format_date(commit.timestamp),
            format_date(commit.timestamp),
            indent(&commit.message)
        ),
        Pretty::Custom { format, .. } => expand_format(format, id, commit, decoration),
    }
}

fn indent(message: &str) -> String {
    message
        .lines()
        .map(|line| format!("    {line}\n"))
        .collect()
}

/// Render a timestamp in one of the formats Git's placeholders use.
fn stamp(timestamp: i64, format: &str) -> String {
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
) -> Option<String> {
    let mut out = String::new();
    let mut characters = format.chars();
    while let Some(character) = characters.next() {
        if character != '%' {
            out.push(character);
            continue;
        }
        let first = characters.next()?;
        let name = match first {
            'a' | 'c' => format!("{first}{}", characters.next()?),
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
            "b" => commit
                .message
                .split_once('\n')
                .map_or("", |rest| rest.1)
                .to_string(),
            "B" => commit.message.clone(),
            "P" => commit.parents.join(" "),
            "p" => commit
                .parents
                .iter()
                .map(|parent| repo::short(parent).to_string())
                .collect::<Vec<_>>()
                .join(" "),
            "d" => decoration.trim_start().to_string(),
            "D" => decoration
                .trim_start()
                .trim_start_matches('(')
                .trim_end_matches(')')
                .to_string(),
            "an" | "cn" => commit.author_name.clone(),
            "ae" | "ce" => commit.author_email.clone(),
            "ad" | "cd" => format_date(commit.timestamp),
            "at" | "ct" => commit.timestamp.to_string(),
            "ai" | "ci" => stamp(commit.timestamp, "%Y-%m-%d %H:%M:%S +0000"),
            "aI" | "cI" => stamp(commit.timestamp, "%Y-%m-%dT%H:%M:%S+00:00"),
            "n" => "\n".to_string(),
            "%" => "%".to_string(),
            _ => return None,
        };
        out.push_str(&text);
    }
    Some(out)
}

fn expand_format(format: &str, id: &str, commit: &Commit, decoration: &str) -> String {
    expand_format_checked(format, id, commit, decoration).unwrap_or_default()
}

/// Split `a..b` or `a...b` into its endpoints.
fn split_range(revision: &str) -> Option<(&str, &str, bool)> {
    if let Some((left, right)) = revision.split_once("...") {
        return Some((left, right, true));
    }
    revision
        .split_once("..")
        .map(|(left, right)| (left, right, false))
}

/// The commits reachable from `revision`, honoring `a..b` exclusion ranges.
fn history_for(
    ctx: &mut CommandContext<'_>,
    root: &str,
    revision: &str,
) -> Option<Vec<(String, Commit)>> {
    let Some((left, right, merge_base)) = split_range(revision) else {
        let start = repo::resolve_revision(ctx, root, revision)?;
        return Some(repo::first_parent_history(ctx, root, &start, 10_000));
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
    Some(
        repo::first_parent_history(ctx, root, &right_commit, 10_000)
            .into_iter()
            .filter(|(id, _)| !excluded.contains(id))
            .collect(),
    )
}

pub(crate) fn git_log(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut pretty = Pretty::Medium;
    let mut limit = 10_000_usize;
    let mut skip = 0_usize;
    let mut revision = "HEAD".to_string();
    let mut reverse = false;
    let mut decorate = false;
    let mut author_filter: Option<String> = None;
    let mut patch = None;
    let mut stat = None;
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
            "--no-decorate" | "--decorate=no" | "--no-merges" | "--first-parent" | "--no-color"
            | "--abbrev-commit" | "--all" => {}
            "-n" | "--max-count" => {
                index += 1;
                let Some(value) = args.get(index).and_then(|value| value.parse().ok()) else {
                    return usage(io, "log count must be a non-negative integer");
                };
                limit = value;
            }
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
                ) else {
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
            value if history_for(ctx, &root, value).is_some() => revision = value.to_string(),
            value => {
                if !super::names_a_path(ctx, &root, value) {
                    return super::ambiguous_argument(io, value);
                }
                paths.push(super::pathspec(&cwd, &root, value));
            }
        }
        index += 1;
    }
    let Some(mut history) = history_for(ctx, &root, &revision) else {
        if repo::head_commit(ctx, &root).is_none() {
            let branch = repo::current_branch(ctx, &root).unwrap_or_else(|| "HEAD".to_string());
            io.err.extend_from_slice(
                format!("fatal: your current branch '{branch}' does not have any commits yet\n")
                    .as_bytes(),
            );
            return 128;
        }
        return super::ambiguous_argument(io, &revision);
    };
    if let Some(author) = &author_filter {
        history.retain(|(_, commit)| {
            commit.author_name.contains(author) || commit.author_email.contains(author)
        });
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
        if separated && position != 0 {
            io.out.push(b'\n');
        }
        let decoration = if decorate {
            decorations(ctx, &root, id)
        } else {
            String::new()
        };
        io.out
            .extend_from_slice(render_commit_header(id, commit, &pretty, &decoration).as_bytes());
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
    }
    0
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
                ) else {
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
                .and_then(|hash| repo::read_blob(ctx, &root, hash))
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
        io.out
            .extend_from_slice(render_commit_header(&id, &commit, &pretty, "").as_bytes());
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
        if abbrev_ref {
            let name = if revision == "HEAD" {
                repo::current_branch(ctx, &root).unwrap_or_else(|| "HEAD".to_string())
            } else {
                revision.rsplit('/').next().unwrap_or(revision).to_string()
            };
            io.out.extend_from_slice(format!("{name}\n").as_bytes());
            continue;
        }
        let Some(commit) = repo::resolve_revision(ctx, &root, revision) else {
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

pub(crate) fn git_rev_list(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut count = false;
    let mut limit = 10_000_usize;
    let mut revision = "HEAD".to_string();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--count" => count = true,
            "--all" => {}
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
            value => revision = value.to_string(),
        }
        index += 1;
    }
    let Some(start) = repo::resolve_revision(ctx, &root, &revision) else {
        return usage(io, "unknown revision");
    };
    let mut history = repo::first_parent_history(ctx, &root, &start, 10_000);
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
    let mut contains: Option<String> = None;
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
            value if value.starts_with("--contains=") || value.starts_with("--merged=") => {
                contains = Some(
                    value
                        .split_once('=')
                        .map_or("", |parts| parts.1)
                        .to_string(),
                );
                list = true;
            }
            "--contains" | "--merged" => {
                // The revision is optional and defaults to HEAD.
                contains = Some(match args.get(index) {
                    Some(value) if !value.starts_with('-') => {
                        index += 1;
                        value.clone()
                    }
                    _ => "HEAD".to_string(),
                });
                list = true;
            }
            "-r" | "--remotes" => {
                // The simulation has no remotes, so there are no remote-tracking branches.
                return 0;
            }
            "-a" | "--all" | "--no-color" | "--no-column" => list = true,
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
        return list_branches(ctx, &root, pattern, contains.as_deref(), verbose, io);
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

fn list_branches(
    ctx: &mut CommandContext<'_>,
    root: &str,
    pattern: Option<&str>,
    contains: Option<&str>,
    verbose: bool,
    io: &mut Io,
) -> i32 {
    // `--contains`/`--merged` keep only branches whose tip is reachable from the named revision.
    let reachable = contains.and_then(|revision| {
        let commit = repo::resolve_revision(ctx, root, revision)?;
        Some(repo::ancestors(ctx, root, &commit))
    });
    let current = repo::current_branch(ctx, root);
    let branches = repo::branch_names(ctx, root);
    let width = branches.iter().map(String::len).max().unwrap_or(0);
    for branch in branches {
        if let Some(pattern) = pattern {
            if !crate::commands::util::glob_eq(pattern, &branch) {
                continue;
            }
        }
        if let Some(reachable) = &reachable {
            let tip = repo::read_reference(ctx, root, &format!("refs/heads/{branch}"));
            if !tip.is_some_and(|tip| reachable.contains(&tip)) {
                continue;
            }
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
        && repo::set_head_to_branch(ctx, root, to).is_err()
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
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported tag option: {value}"))
            }
            value => operands.push(value.to_string()),
        }
        index += 1;
    }
    if delete {
        for name in &operands {
            if repo::delete_reference(ctx, &root, &format!("refs/tags/{name}")).is_err() {
                io.err
                    .extend_from_slice(format!("error: tag '{name}' not found.\n").as_bytes());
                return 1;
            }
            io.out
                .extend_from_slice(format!("Deleted tag '{name}'\n").as_bytes());
        }
        return 0;
    }
    if operands.is_empty() || list {
        for name in repo::reference_names(ctx, &root, "tags") {
            if let Some(pattern) = operands.first() {
                if !crate::commands::util::glob_eq(pattern, &name) {
                    continue;
                }
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

fn working_tree_clean(ctx: &mut CommandContext<'_>, root: &str) -> bool {
    let index = repo::load_index(ctx, root).unwrap_or_default();
    let Ok(work) = repo::collect_working_tree(ctx, root) else {
        return false;
    };
    let files = work.release(ctx);
    let head = repo::head_tree(ctx, root);
    // Untracked files never block a switch.
    files
        .iter()
        .filter(|(path, _)| index.contains_key(*path))
        .all(|(path, hash)| index.get(path) == Some(hash))
        && index.keys().all(|path| files.contains_key(path))
        && index == head
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
    if !working_tree_clean(ctx, root) {
        io.err.extend_from_slice(
            b"error: local changes would be overwritten; commit or restore them first\n",
        );
        return Err(1);
    }
    let old = repo::head_tree(ctx, root);
    let Some(new) = repo::commit_tree(ctx, root, commit) else {
        return Err(usage(io, "target revision has an invalid commit"));
    };
    if let Err(error) = repo::replace_work_tree(ctx, root, &old, &new) {
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
    if repo::set_head_to_branch(ctx, root, branch).is_err() {
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
    if repo::set_head_detached(ctx, root, &commit).is_err() {
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
    for argument in args {
        match argument.as_str() {
            "-c" | "--create" => create = true,
            "-C" | "--force-create" => {
                create = true;
                force = true;
            }
            "-d" | "--detach" => detach = true,
            "-f" | "--force" | "--discard-changes" => force = true,
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
    let bytes = ctx.vfs.read("/", &repo::git_path(root, "PREV_HEAD")).ok()?;
    let previous = String::from_utf8_lossy(&bytes).trim().to_string();
    (!previous.is_empty()).then_some(previous)
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
    for argument in args {
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
            value if value.starts_with('-') && value != "-" => {
                return usage(io, &format!("unsupported checkout option: {value}"))
            }
            value => operands.push(value.to_string()),
        }
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
    if !working_tree_clean(ctx, &root) {
        io.err
            .extend_from_slice(b"error: your local changes would be overwritten by merge\n");
        return 1;
    }
    let base = repo::merge_base(ctx, &root, &head, &other);
    let fast_forward = base.as_deref() == Some(head.as_str());
    if fast_forward && !no_fast_forward {
        if let Err(status) = checkout_commit(ctx, &root, &other, io) {
            return status;
        }
        if repo::update_head(ctx, &root, &other).is_err() {
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
    let merged = match merge_trees(&base_tree, &head_tree, &other_tree) {
        Ok(tree) => tree,
        Err(conflicts) => {
            for path in &conflicts {
                io.err.extend_from_slice(
                    format!("CONFLICT (content): Merge conflict in {path}\n").as_bytes(),
                );
            }
            io.err.extend_from_slice(
                b"fatal: simulated git cannot resolve content conflicts; merge manually\n",
            );
            return 1;
        }
    };
    if let Err(error) = repo::replace_work_tree(ctx, &root, &head_tree, &merged) {
        io.err
            .extend_from_slice(format!("git merge: {error}\n").as_bytes());
        return 1;
    }
    if repo::store_index(ctx, &root, &merged).is_err() {
        return 1;
    }
    let (author_name, author_email) = author_identity(ctx, &root, globals);
    let commit = Commit {
        parents: vec![head.clone(), other.clone()],
        author_name,
        author_email,
        timestamp: now_seconds(ctx),
        message: message.unwrap_or_else(|| format!("Merge branch '{target}'")),
    };
    let Ok(id) = repo::store_commit(ctx, &root, &commit, &merged) else {
        return 1;
    };
    if repo::update_head(ctx, &root, &id).is_err() {
        return 1;
    }
    io.out
        .extend_from_slice(b"Merge made by the 'ort' strategy.\n");
    emit_commit_summary(ctx, &root, &head_tree, &merged, io);
    0
}

/// Combine two trees against their base, taking whichever side changed each path.
///
/// Returns the conflicting paths when both sides changed the same path differently.
fn merge_trees(base: &Tree, left: &Tree, right: &Tree) -> Result<Tree, Vec<String>> {
    let mut names: BTreeSet<&String> = base.keys().collect();
    names.extend(left.keys());
    names.extend(right.keys());
    let mut merged = Tree::new();
    let mut conflicts = Vec::new();
    for path in names {
        let original = base.get(path);
        let ours = left.get(path);
        let theirs = right.get(path);
        let resolved = match (ours == original, theirs == original) {
            (true, true) => ours,
            (false, true) => ours,
            (true, false) => theirs,
            (false, false) if ours == theirs => ours,
            (false, false) => {
                conflicts.push(path.clone());
                continue;
            }
        };
        if let Some(hash) = resolved {
            merged.insert(path.clone(), hash.clone());
        }
    }
    if conflicts.is_empty() {
        Ok(merged)
    } else {
        Err(conflicts)
    }
}

#[cfg(test)]
mod tests {
    use super::{merge_trees, Tree};

    fn tree(entries: &[(&str, &str)]) -> Tree {
        entries
            .iter()
            .map(|(path, hash)| ((*path).to_string(), (*hash).to_string()))
            .collect()
    }

    #[test]
    fn takes_the_changed_side_of_each_path() {
        let base = tree(&[("a", "1"), ("b", "1")]);
        let left = tree(&[("a", "2"), ("b", "1")]);
        let right = tree(&[("a", "1"), ("b", "3")]);
        assert_eq!(
            merge_trees(&base, &left, &right),
            Ok(tree(&[("a", "2"), ("b", "3")]))
        );
    }

    #[test]
    fn reports_paths_changed_on_both_sides() {
        let base = tree(&[("a", "1")]);
        let left = tree(&[("a", "2")]);
        let right = tree(&[("a", "3")]);
        assert_eq!(
            merge_trees(&base, &left, &right),
            Err(vec!["a".to_string()])
        );
    }

    #[test]
    fn honors_deletions_from_one_side() {
        let base = tree(&[("a", "1"), ("b", "1")]);
        let left = tree(&[("a", "1")]);
        let right = tree(&[("a", "1"), ("b", "1")]);
        assert_eq!(merge_trees(&base, &left, &right), Ok(tree(&[("a", "1")])));
    }
}
