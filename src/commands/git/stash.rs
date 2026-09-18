//! Setting work aside with `git stash`.
//!
//! A stash entry records two trees, the index and the working tree, plus the commit and branch
//! they were built on. Entries live in `.git/stash/` and are listed newest first in
//! `.git/stash-list`. Reapplying an entry is a three-way merge against the commit it was built
//! on, so work committed in the meantime survives; a conflict leaves markers and keeps the entry.

use crate::commands::{CommandContext, Io};

use super::conflict;
use super::repo::{self, Tree};
use super::{cannot_write, repo_error, require_index, usage, Arg, Flags};

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
    let mut patch = false;
    let mut message = None;
    let mut operands: Vec<String> = Vec::new();
    let mut flags = Flags::new(args).clustered("qum").valued("m");
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                operands.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "-u" | "--include-untracked" => include_untracked = true,
            "-q" | "--quiet" => quiet = true,
            "-p" | "--patch" => patch = true,
            "--no-keep-index" => {}
            "-m" | "--message" => {
                let Some(value) = flags.value(attached) else {
                    return usage(io, "-m requires a message");
                };
                message = Some(value);
            }
            _ => return usage(io, &format!("unsupported stash option: {name}")),
        }
    }
    let command = operands.first().map(String::as_str).unwrap_or("push");
    // `-q` only silences the progress report; diagnostics still reach standard error.
    let before = io.out.len();
    let status = match command {
        "push" | "save" => push(ctx, &root, message, include_untracked, io),
        "list" => list(ctx, &root, io),
        "show" => {
            let Some(position) = selector(operands.get(1).filter(|value| !value.starts_with('-')))
            else {
                return usage(io, "usage: git stash show [-p] [stash@{N}]");
            };
            show(ctx, &root, position, patch, io)
        }
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
        io.print_err("fatal: You do not have the initial commit yet\n");
        return 128;
    };
    // An unresolved conflict has three sides, and a stash entry records only one tree, so the
    // merge result would be silently thrown away. Git refuses for the same reason.
    let unmerged = conflict::load_stages(ctx, root);
    if !unmerged.is_empty() {
        for path in unmerged.keys() {
            io.print_err(&format!("{path}: needs merge\n"));
        }
        io.print_err("error: could not write index\n");
        return 1;
    }
    let index_tree = match require_index(ctx, root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    let head_tree = repo::head_tree(ctx, root);
    let mut work = match repo::collect_working_tree(ctx, root) {
        Ok(work) => work,
        Err(status) => return status,
    };
    if !include_untracked {
        work.retain(|path, _| index_tree.contains_key(path) || head_tree.contains_key(path));
    }
    if work == head_tree && index_tree == head_tree {
        io.print("No local changes to save\n");
        return 0;
    }
    // Blobs the stash refers to must outlive the reset, so store them now.
    for path in work.keys() {
        let Some(data) = repo::read_work_file(ctx, root, path) else {
            continue;
        };
        if let Err(error) = repo::write_blob(ctx, root, &data) {
            return cannot_write(io, "the object", &error);
        }
    }
    let branch = repo::current_branch(ctx, root).unwrap_or_else(|| "HEAD".to_string());
    let subject = repo::load_commit(ctx, root, &base)
        .map(|commit| commit.subject().to_string())
        .unwrap_or_default();
    let mut entries = load_entries(ctx, root);
    let id = entries.iter().map(|entry| entry.id).max().unwrap_or(0) + 1;
    // Git labels an explicit message `On <branch>:` and a default one `WIP on <branch>:`.
    let message = message.map_or_else(
        || format!("WIP on {branch}: {} {subject}", repo::short(&base)),
        |text| format!("On {branch}: {text}"),
    );
    for (kind, tree) in [("work", &work), ("index", &index_tree)] {
        if let Err(error) =
            repo::write_vfs(ctx, &tree_path(root, id, kind), &repo::serialize_tree(tree))
        {
            return cannot_write(io, "the stash entry", &error);
        }
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
        io.print_err(&format!("git stash: {error}\n"));
        return 1;
    }
    if let Err(error) = repo::store_index(ctx, root, &head_tree) {
        return cannot_write(io, "the index", &error);
    }
    io.print(&format!(
        "Saved working directory and index state {message}\n"
    ));
    0
}

fn list(ctx: &mut CommandContext<'_>, root: &str, io: &mut Io) -> i32 {
    for (position, entry) in load_entries(ctx, root).iter().enumerate() {
        io.print(&format!("stash@{{{position}}}: {}\n", entry.message));
    }
    0
}

/// Report what an entry would reapply, as `git stash show` does.
fn show(
    ctx: &mut CommandContext<'_>,
    root: &str,
    position: usize,
    patch: bool,
    io: &mut Io,
) -> i32 {
    let entries = load_entries(ctx, root);
    let Some(entry) = entries.get(position) else {
        io.print_err(&format!(
            "fatal: stash@{{{position}}} is not a valid reference\n"
        ));
        return 128;
    };
    let Some(work) = read_tree(ctx, &tree_path(root, entry.id, "work")) else {
        io.print_err("fatal: the stash entry is unreadable\n");
        return 128;
    };
    let base = repo::commit_tree(ctx, root, &entry.base).unwrap_or_default();
    let options = super::compare::Options {
        format: if patch {
            super::compare::Format::Patch
        } else {
            super::compare::Format::Stat
        },
        ..Default::default()
    };
    super::compare::emit(ctx, root, &base, &work, &options, io);
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
        io.print_err(&format!(
            "fatal: stash@{{{position}}} is not a valid reference\n"
        ));
        return 128;
    };
    let (Some(work), Some(index_tree)) = (
        read_tree(ctx, &tree_path(root, entry.id, "work")),
        read_tree(ctx, &tree_path(root, entry.id, "index")),
    ) else {
        io.print_err("fatal: the stash entry is unreadable\n");
        return 128;
    };
    // The entry is merged back in against the commit it was taken from, so anything committed or
    // edited since is kept rather than overwritten.
    let base = match super::require_tree(ctx, root, &entry.base, io) {
        Ok(tree) => tree,
        Err(status) => return status,
    };
    let mine = match repo::collect_working_tree(ctx, root) {
        Ok(work) => work,
        Err(status) => return status,
    };
    // The merge reads both sides from the object store, and a working-tree file that was never
    // staged has nothing there yet.
    for (path, recorded) in &mine {
        if base.get(path) == Some(recorded) || work.get(path) == Some(recorded) {
            continue;
        }
        let Some(data) = repo::read_work_file(ctx, root, path) else {
            continue;
        };
        if repo::write_blob(ctx, root, &data).is_err() {
            io.print_err(&format!("git stash: cannot record '{path}'\n"));
            return 1;
        }
    }
    // A file the entry would change that the working tree has already moved away from has two
    // unrecorded versions and room for one. Git refuses rather than choose, and so does this.
    let index = match require_index(ctx, root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    let doomed: Vec<String> = work
        .keys()
        .chain(base.keys())
        .filter(|path| work.get(*path) != base.get(*path) && mine.get(*path) != index.get(*path))
        .cloned()
        .collect::<std::collections::BTreeSet<String>>()
        .into_iter()
        .collect();
    if !doomed.is_empty() {
        io.print_err(
            "error: Your local changes to the following files would be overwritten by merge:\n",
        );
        for path in doomed {
            io.print_err(&format!("\t{path}\n"));
        }
        io.print_err(
            "Please commit your changes or stash them before you merge.\nAborting\n\
              The stash entry is kept in case you need it again.\n",
        );
        return 1;
    }
    let combined = conflict::combine(
        ctx,
        root,
        &base,
        &mine,
        &work,
        "Updated upstream",
        "Stashed changes",
    );
    let combined = match combined {
        Ok(combined) => combined,
        Err(failed) => return cannot_write(io, &failed.path, &failed.error),
    };
    if let Err(error) = repo::update_work_tree(ctx, root, &mine, &combined.tree) {
        io.print_err(&format!("git stash: {error}\n"));
        return 1;
    }
    // A plain apply restores the working tree only, leaving what was staged for the user to stage
    // again; that is what Git does without `--index`.
    // A file the entry brings back that nothing tracks yet cannot be left "unstaged", so Git
    // records it as a new file. Paths already tracked keep whatever the index says, and a file
    // that was untracked when the entry was pushed stays untracked.
    let mut index = match require_index(ctx, root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    let restored: Vec<(String, repo::Entry)> = combined
        .tree
        .iter()
        .filter(|(path, _)| !index.contains_key(*path) && index_tree.contains_key(*path))
        .map(|(path, entry)| (path.clone(), entry.clone()))
        .collect();
    if !restored.is_empty() {
        index.extend(restored);
        if let Err(error) = repo::store_index(ctx, root, &index) {
            return cannot_write(io, "the index", &error);
        }
    }
    if !combined.stages.is_empty() {
        for path in combined.stages.keys() {
            io.print_err(&format!("CONFLICT (content): Merge conflict in {path}\n"));
        }
        if !conflict::store_stages(ctx, root, &combined.stages) {
            io.print_err("fatal: unable to record the conflicted state\n");
            return 128;
        }
        // The entry stays on the list so the user can try again after settling the conflict.
        io.print_err("The stash entry is kept in case you need it again.\n");
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
        io.print_err(&format!(
            "fatal: stash@{{{position}}} is not a valid reference\n"
        ));
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
    io.print(&format!(
        "Dropped refs/stash@{{{position}}} ({identifier})\n"
    ));
    0
}
