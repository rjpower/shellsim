//! Setting work aside with `git stash`.
//!
//! A stash entry records two trees, the index and the working tree, plus the commit and branch
//! they were built on. Entries live in `.git/stash/` and are listed newest first in
//! `.git/stash-list`. Conflicting reapplication is refused rather than producing markers, which
//! matches how the merge subset behaves.

use crate::commands::{CommandContext, Io};

use super::repo::{self, Tree};
use super::{repo_error, usage};

/// One saved entry.
struct Entry {
    id: u64,
    base: String,
    branch: String,
    message: String,
}

fn list_path(root: &str) -> String {
    repo::git_path(root, "stash-list")
}

fn tree_path(root: &str, id: u64, kind: &str) -> String {
    repo::git_path(root, &format!("stash/{id}.{kind}"))
}

fn load_entries(ctx: &CommandContext<'_>, root: &str) -> Vec<Entry> {
    let Ok(bytes) = ctx.vfs.read("/", &list_path(root)) else {
        return Vec::new();
    };
    String::from_utf8_lossy(&bytes)
        .lines()
        .filter_map(|line| {
            let mut fields = line.splitn(4, '\t');
            Some(Entry {
                id: fields.next()?.parse().ok()?,
                base: fields.next()?.to_string(),
                branch: fields.next()?.to_string(),
                message: fields.next().unwrap_or_default().to_string(),
            })
        })
        .collect()
}

fn store_entries(ctx: &mut CommandContext<'_>, root: &str, entries: &[Entry]) -> bool {
    let text: String = entries
        .iter()
        .map(|entry| {
            format!(
                "{}\t{}\t{}\t{}\n",
                entry.id, entry.base, entry.branch, entry.message
            )
        })
        .collect();
    repo::write_vfs(ctx, &list_path(root), text.as_bytes()).is_ok()
}

/// Parse a `stash@{N}` selector, defaulting to the newest entry.
fn selector(operand: Option<&String>) -> Option<usize> {
    let Some(operand) = operand else {
        return Some(0);
    };
    operand
        .strip_prefix("stash@{")
        .and_then(|rest| rest.strip_suffix('}'))
        .unwrap_or(operand)
        .parse()
        .ok()
}

pub(crate) fn git_stash(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut include_untracked = false;
    let mut quiet = false;
    let mut message = None;
    let mut operands: Vec<String> = Vec::new();
    let mut index = 0;
    for argument in super::expand_clusters(args, "um") {
        match argument.as_str() {
            "-u" | "--include-untracked" => include_untracked = true,
            "-q" | "--quiet" => quiet = true,
            "--no-keep-index" => {}
            "-m" | "--message" => index = usize::MAX,
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported stash option: {value}"))
            }
            value if index == usize::MAX => {
                message = Some(value.to_string());
                index = 0;
            }
            value => operands.push(value.to_string()),
        }
    }
    let command = operands.first().map(String::as_str).unwrap_or("push");
    // `-q` only silences the progress report; diagnostics still reach standard error.
    let before = io.out.len();
    let status = match command {
        "push" | "save" => push(ctx, &root, message, include_untracked, io),
        "list" => list(ctx, &root, io),
        "pop" | "apply" => {
            let Some(position) = selector(operands.get(1)) else {
                return usage(io, "usage: git stash pop [stash@{N}]");
            };
            apply(ctx, &root, position, command == "pop", io)
        }
        "drop" => {
            let Some(position) = selector(operands.get(1)) else {
                return usage(io, "usage: git stash drop [stash@{N}]");
            };
            drop_entry(ctx, &root, position, io)
        }
        "clear" => {
            let entries: Vec<Entry> = Vec::new();
            i32::from(!store_entries(ctx, &root, &entries))
        }
        other => usage(io, &format!("unsupported stash command: {other}")),
    };
    if quiet {
        io.out.truncate(before);
    }
    status
}

fn push(
    ctx: &mut CommandContext<'_>,
    root: &str,
    message: Option<String>,
    include_untracked: bool,
    io: &mut Io,
) -> i32 {
    let Some(base) = repo::head_commit(ctx, root) else {
        io.err
            .extend_from_slice(b"fatal: You do not have the initial commit yet\n");
        return 128;
    };
    let index_tree = repo::load_index(ctx, root).unwrap_or_default();
    let head_tree = repo::head_tree(ctx, root);
    let mut work = match repo::collect_working_tree(ctx, root) {
        Ok(snapshot) => snapshot.release(ctx),
        Err(status) => return status,
    };
    if !include_untracked {
        work.retain(|path, _| index_tree.contains_key(path) || head_tree.contains_key(path));
    }
    if work == head_tree && index_tree == head_tree {
        io.out.extend_from_slice(b"No local changes to save\n");
        return 0;
    }
    // Blobs the stash refers to must outlive the reset, so store them now.
    for path in work.keys() {
        let Some(data) = repo::read_work_file(ctx, root, path) else {
            continue;
        };
        if repo::write_blob(ctx, root, &data).is_err() {
            return 1;
        }
    }
    let branch = repo::current_branch(ctx, root).unwrap_or_else(|| "HEAD".to_string());
    let subject = repo::load_commit(ctx, root, &base)
        .map(|commit| commit.subject().to_string())
        .unwrap_or_default();
    let mut entries = load_entries(ctx, root);
    let id = entries.iter().map(|entry| entry.id).max().unwrap_or(0) + 1;
    let message =
        message.unwrap_or_else(|| format!("WIP on {branch}: {} {subject}", repo::short(&base)));
    if repo::write_vfs(
        ctx,
        &tree_path(root, id, "work"),
        &repo::serialize_tree(&work),
    )
    .is_err()
        || repo::write_vfs(
            ctx,
            &tree_path(root, id, "index"),
            &repo::serialize_tree(&index_tree),
        )
        .is_err()
    {
        return 1;
    }
    entries.insert(
        0,
        Entry {
            id,
            base: base.clone(),
            branch: branch.clone(),
            message: message.clone(),
        },
    );
    if !store_entries(ctx, root, &entries) {
        return 1;
    }
    // Restore the working tree and index to HEAD, discarding what was just saved.
    let mut previous = head_tree.clone();
    previous.extend(index_tree);
    previous.extend(work);
    if let Err(error) = repo::replace_work_tree(ctx, root, &previous, &head_tree) {
        io.err
            .extend_from_slice(format!("git stash: {error}\n").as_bytes());
        return 1;
    }
    if repo::store_index(ctx, root, &head_tree).is_err() {
        return 1;
    }
    io.out.extend_from_slice(
        format!("Saved working directory and index state {message}\n").as_bytes(),
    );
    0
}

fn list(ctx: &mut CommandContext<'_>, root: &str, io: &mut Io) -> i32 {
    for (position, entry) in load_entries(ctx, root).iter().enumerate() {
        io.out
            .extend_from_slice(format!("stash@{{{position}}}: {}\n", entry.message).as_bytes());
    }
    0
}

fn read_tree(ctx: &CommandContext<'_>, path: &str) -> Option<Tree> {
    repo::parse_tree(&ctx.vfs.read("/", path).ok()?)
}

fn apply(
    ctx: &mut CommandContext<'_>,
    root: &str,
    position: usize,
    drop_after: bool,
    io: &mut Io,
) -> i32 {
    let entries = load_entries(ctx, root);
    let Some(entry) = entries.get(position) else {
        io.err.extend_from_slice(
            format!("fatal: stash@{{{position}}} is not a valid reference\n").as_bytes(),
        );
        return 128;
    };
    let (Some(work), Some(index_tree)) = (
        read_tree(ctx, &tree_path(root, entry.id, "work")),
        read_tree(ctx, &tree_path(root, entry.id, "index")),
    ) else {
        io.err
            .extend_from_slice(b"fatal: the stash entry is unreadable\n");
        return 128;
    };
    let current = repo::head_tree(ctx, root);
    // Reapplying overwrites whole files, so refuse when that would discard uncommitted work.
    let touched = super::history::blocking_changes(ctx, root, &work);
    if !touched.is_empty() {
        io.err.extend_from_slice(
            b"error: Your local changes to the following files would be overwritten by merge:\n",
        );
        for path in touched {
            io.err.extend_from_slice(format!("\t{path}\n").as_bytes());
        }
        io.err.extend_from_slice(
            b"Please commit your changes or stash them before you merge.\nAborting\n",
        );
        return 1;
    }
    if let Err(error) = repo::update_work_tree(ctx, root, &current, &work) {
        io.err
            .extend_from_slice(format!("git stash: {error}\n").as_bytes());
        return 1;
    }
    if repo::store_index(ctx, root, &index_tree).is_err() {
        return 1;
    }
    super::worktree::git_status(ctx, &[], io);
    if drop_after {
        return drop_entry(ctx, root, position, io);
    }
    0
}

fn drop_entry(ctx: &mut CommandContext<'_>, root: &str, position: usize, io: &mut Io) -> i32 {
    let mut entries = load_entries(ctx, root);
    if position >= entries.len() {
        io.err.extend_from_slice(
            format!("fatal: stash@{{{position}}} is not a valid reference\n").as_bytes(),
        );
        return 128;
    }
    let entry = entries.remove(position);
    // Git names the dropped stash commit; this subset has no such commit, so it derives a stable
    // identifier from the trees the entry saved.
    let identifier = repo::sha1(
        format!(
            "stash {} {} {}",
            entry.base,
            read_tree(ctx, &tree_path(root, entry.id, "work"))
                .as_ref()
                .map(repo::tree_hash)
                .unwrap_or_default(),
            read_tree(ctx, &tree_path(root, entry.id, "index"))
                .as_ref()
                .map(repo::tree_hash)
                .unwrap_or_default(),
        )
        .as_bytes(),
    );
    if !store_entries(ctx, root, &entries) {
        return 1;
    }
    io.out.extend_from_slice(
        format!("Dropped refs/stash@{{{position}}} ({identifier})\n").as_bytes(),
    );
    0
}
