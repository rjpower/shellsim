//! `git commit`: turning the staged tree into a new commit.
//!
//! The command reads the index, decides whether there is anything to record, writes the commit,
//! and prints the summary line and mode changes Git reports afterwards.

use std::collections::BTreeSet;

use crate::commands::{CommandContext, Io};

use super::compare::{self, Format, Options};
use super::conflict;
use super::log;
use super::merge;
use super::repo::{self, Commit, Tree};
use super::worktree;
use super::{repo_error, usage, Arg, Flags, Globals};

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
    let mut flags = Flags::new(args).clustered("amqvns").valued("mF");
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                paths.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "-m" | "--message" => {
                let Some(value) = flags.value(attached) else {
                    return usage(io, "-m requires a message");
                };
                messages.push(value);
            }
            "-F" | "--file" => {
                let Some(value) = flags.value(attached) else {
                    return usage(io, "-F requires a file");
                };
                message_file = Some(value);
            }
            "--author" => {
                let Some(value) = flags.value(attached) else {
                    return usage(io, "--author requires a name");
                };
                author = Some(value);
            }
            "-a" | "--all" => stage_tracked = true,
            "--amend" => amend = true,
            "--allow-empty" | "--allow-empty-message" => allow_empty = true,
            "-q" | "--quiet" => quiet = true,
            "--dry-run" => dry_run = true,
            "-s" | "--signoff" => signoff = true,
            "-n" | "--no-verify" | "-v" | "--verbose" | "--no-edit" | "--no-gpg-sign" => {}
            _ => return usage(io, &format!("unsupported commit option: {name}")),
        }
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
    let pending = merge::pending_operation(ctx, &root);
    if !conflict::load_stages(ctx, &root).is_empty() {
        io.print_err(
            "error: Committing is not possible because you have unmerged files.\n\
              hint: Fix them up in the work tree, and then use 'git add/rm <file>'\n\
              hint: as appropriate to mark resolution and make a commit.\n\
              fatal: Exiting because of an unresolved conflict.\n",
        );
        return 1;
    }
    let previous = repo::head_commit(ctx, &root)
        .and_then(|id| repo::load_commit(ctx, &root, &id).map(|commit| (id, commit)));
    if amend && previous.is_none() {
        io.print_err("fatal: You have nothing to amend.\n");
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
            io.print_err(&format!(
                "fatal: could not read log file '{path}': No such file or directory\n"
            ));
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
        let staged = match super::require_index(ctx, &root, io) {
            Ok(index) => index != repo::head_tree(ctx, &root),
            Err(status) => return status,
        };
        let status = super::worktree::git_status(ctx, &[], io);
        return if status == 0 && !staged { 1 } else { status };
    }
    if message.is_empty() && !allow_empty {
        return usage(io, "a non-empty -m MESSAGE is required");
    }
    let index_tree = match super::require_index(ctx, &root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    // Amending replaces the previous commit, so the comparison tree is its parent's.
    let (parents, baseline) = match (amend, &previous) {
        (true, Some((_, commit))) => {
            let baseline = match super::parent_tree(ctx, &root, commit.parents.first(), io) {
                Ok(tree) => tree,
                Err(status) => return status,
            };
            (commit.parents.clone(), baseline)
        }
        (_, Some((id, _))) => {
            let mut parents = vec![id.clone()];
            // Finishing a merge records the commit that was being merged as the second parent.
            if let Some((conflict::MERGE_HEAD, other)) = pending.as_ref().map(|(k, c)| (*k, c)) {
                parents.push(other.clone());
            }
            let baseline = match super::require_tree(ctx, &root, id, io) {
                Ok(tree) => tree,
                Err(status) => return status,
            };
            (parents, baseline)
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
        let (name, email) = repo::author_identity(ctx, &root, globals);
        message.push_str(&format!("\n\nSigned-off-by: {name} <{email}>"));
    }
    let (default_name, default_email) = repo::author_identity(ctx, &root, globals);
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
            .map_or_else(|| log::author_date(ctx), |(_, commit)| commit.timestamp),
        message: message.clone(),
    };
    let id = match repo::store_commit(ctx, &root, &commit, &index_tree) {
        Ok(id) => id,
        Err(error) => {
            io.print_err(&format!("git commit: {error}\n"));
            return 1;
        }
    };
    // Git words the reflog by what the commit was: the first on a branch, an amend, or neither.
    let what = match (commit.parents.is_empty(), amend) {
        (true, _) => "commit (initial)",
        (false, true) => "commit (amend)",
        (false, false) => "commit",
    };
    let action = format!("{what}: {}", commit.subject());
    if let Err(error) = repo::update_head(ctx, &root, &id, &action) {
        return super::cannot_write(io, "HEAD", &error);
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
    io.print(&format!(
        "[{label} {root_commit}{}] {}\n",
        repo::short(&id),
        commit.subject()
    ));
    if amend {
        // Amending keeps the original author date, which Git points out.
        io.print(&format!(" Date: {}\n", repo::format_date(commit.timestamp)));
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
pub(crate) fn emit_commit_summary(
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
    io.print(&compare::summary_line(changed.len(), insertions, deletions));
    emit_mode_lines(ctx, root, old, new, io);
}

/// Print the ` create mode`, ` delete mode` and ` rename` lines that follow a change summary.
pub(crate) fn emit_mode_lines(
    ctx: &mut CommandContext<'_>,
    root: &str,
    old: &Tree,
    new: &Tree,
    io: &mut Io,
) {
    let summary = Options {
        format: Format::Summary,
        ..Options::default()
    };
    compare::emit(ctx, root, old, new, &summary, io);
}

/// Report that nothing is staged, using the same wording as `git status`.
fn emit_nothing_to_commit(ctx: &mut CommandContext<'_>, io: &mut Io) {
    super::worktree::git_status(ctx, &[], io);
}
