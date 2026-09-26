//! Moving HEAD: `git switch` and `git checkout`.
//!
//! A move compares HEAD, the index, and the working tree against the target so that local work is
//! either carried across or the move refused, then replaces the files that differ. Checking out a
//! path, or one side of a conflict, is handled here too.

use std::collections::BTreeSet;

use crate::commands::Io;
use crate::syscalls::System;

use super::conflict;
use super::merge;
use super::rebase;
use super::refs;
use super::repo::{self, Tree};
use super::{repo_error, usage, Arg, Flags};

/// The tracked paths that differ from HEAD, which is what blocks a branch switch.
///
/// Untracked files never block a switch, so they are not considered here.
/// HEAD, the index and the working tree, read once.
///
/// Every question a move has to answer — what would be overwritten, what is only staged, which
/// untracked files are in the way — compares these three trees, and walking and re-hashing the
/// working tree for each question costs the same again.
pub(crate) struct Snapshot {
    pub head: Tree,
    pub index: Tree,
    pub work: Tree,
}

pub(crate) fn snapshot(system: &mut dyn System, root: &str, io: &mut Io) -> Result<Snapshot, i32> {
    let head = repo::head_tree(system, root);
    let index = super::require_index(system, root, io)?;
    let work = repo::collect_working_tree(system, root)?;
    Ok(Snapshot { head, index, work })
}

/// Tracked paths that moving to `target` would overwrite work in.
pub(crate) fn blocking_changes(snapshot: &Snapshot, target: &Tree) -> Vec<String> {
    // Only the paths the move would actually rewrite can lose work.
    dirty_paths(snapshot)
        .into_iter()
        .filter(|path| target.get(path) != snapshot.head.get(path))
        .collect()
}

/// Untracked working files that moving to `target` would write over.
///
/// An untracked file is recorded nowhere, so overwriting one destroys it for good. Git refuses to
/// move rather than do that, and names the files it would have replaced.
pub(crate) fn untracked_collisions(snapshot: &Snapshot, target: &Tree) -> Vec<String> {
    target
        .keys()
        .filter(|path| {
            snapshot.work.contains_key(*path)
                && !snapshot.index.contains_key(*path)
                && !snapshot.head.contains_key(*path)
        })
        .cloned()
        .collect()
}

/// Report the untracked files `target` would overwrite, the way Git reports them.
pub(crate) fn refuse_untracked_overwrite(
    snapshot: &Snapshot,
    target: &Tree,
    verb: &str,
    advice: &str,
    io: &mut Io,
) -> bool {
    let doomed = untracked_collisions(snapshot, target);
    if doomed.is_empty() {
        return false;
    }
    io.print_err(&format!(
        "error: The following untracked working tree files would be overwritten by {verb}:\n"
    ));
    for path in doomed {
        io.print_err(&format!("\t{path}\n"));
    }
    io.print_err(&format!(
        "Please move or remove them before you {advice}.\nAborting\n"
    ));
    true
}

/// Where a tracked path is not the same in HEAD, the index and the working tree.
///
/// A command that rewrites the whole tree, such as `git rebase`, has nowhere to put these, so it
/// refuses to start rather than overwrite them.
pub(crate) fn dirty_paths(snapshot: &Snapshot) -> Vec<String> {
    let mut blocked: BTreeSet<String> = BTreeSet::new();
    for (path, hash) in &snapshot.index {
        if snapshot.work.get(path) != Some(hash) || snapshot.head.get(path) != Some(hash) {
            blocked.insert(path.clone());
        }
    }
    for path in snapshot.head.keys() {
        if !snapshot.index.contains_key(path) {
            blocked.insert(path.clone());
        }
    }
    blocked.into_iter().collect()
}

/// Whether the difference is only staged, which Git words differently.
pub(crate) fn only_staged(snapshot: &Snapshot, paths: &[String]) -> bool {
    paths
        .iter()
        .all(|path| snapshot.work.get(path) == snapshot.index.get(path))
}

/// Move HEAD and the working tree to `commit`.
pub(crate) fn checkout_commit(
    system: &mut dyn System,
    root: &str,
    commit: &str,
    io: &mut Io,
) -> Result<(), i32> {
    if let Some(reason) = unfinished_operation(system, root) {
        io.print_err(&reason);
        return Err(if reason.starts_with("fatal") { 128 } else { 1 });
    }
    if let Some(previous) = repo::current_branch(system, root) {
        let _ = repo::write_vfs(
            system,
            &repo::git_path(root, "PREV_HEAD"),
            format!("{previous}\n").as_bytes(),
        );
    }
    let new = super::require_tree(system, root, commit, io)?;
    let snapshot = snapshot(system, root, io)?;
    let old = &snapshot.head;
    let blocked = blocking_changes(&snapshot, &new);
    if !blocked.is_empty() {
        io.print_err(
            "error: Your local changes to the following files would be overwritten by checkout:\n",
        );
        for path in blocked {
            io.print_err(&format!("\t{path}\n"));
        }
        io.print_err(
            "Please commit your changes or stash them before you switch branches.\nAborting\n",
        );
        return Err(1);
    }
    if refuse_untracked_overwrite(&snapshot, &new, "checkout", "switch branches", io) {
        return Err(1);
    }
    if let Err(error) = repo::update_work_tree(system, root, old, &new) {
        io.print_err(&format!("git switch: {error}\n"));
        return Err(1);
    }
    let index = carried_index(&snapshot, &new);
    if let Err(error) = repo::store_index(system, root, &index) {
        return Err(super::cannot_write(io, "the index", &error));
    }
    Ok(())
}

/// The index after moving to `target`, keeping whatever was staged and not committed.
///
/// A move is only allowed when every path that differs between HEAD and the index is the same in
/// both commits, so carrying those entries across cannot contradict the new commit. Git keeps
/// them; replacing the index with the target tree would silently unstage the lot.
fn carried_index(snapshot: &Snapshot, target: &Tree) -> Tree {
    let mut carried = target.clone();
    for (path, entry) in &snapshot.index {
        if snapshot.head.get(path) != Some(entry) {
            carried.insert(path.clone(), entry.clone());
        }
    }
    // A path staged for deletion stays deleted on the other side of the move.
    for path in snapshot.head.keys() {
        if !snapshot.index.contains_key(path) {
            carried.remove(path);
        }
    }
    carried
}

/// Why moving the working tree to another commit has to wait, if it does.
///
/// A rebase holds HEAD detached and a half-replayed history, and unresolved conflicts have
/// nowhere to go, so Git refuses to move in both cases rather than lose either.
fn unfinished_operation(system: &mut dyn System, root: &str) -> Option<String> {
    if rebase::in_progress(system, root) {
        return Some(
            "fatal: cannot switch branch while rebasing\nConsider \"git rebase --abort\" or \"git rebase --continue\".\n"
                .to_string(),
        );
    }
    let (kind, doing) = match merge::pending_operation(system, root)? {
        (conflict::REVERT_HEAD, _) => ("revert", "reverting"),
        (conflict::CHERRY_PICK_HEAD, _) => ("cherry-pick", "cherry-picking"),
        _ => ("merge", "merging"),
    };
    Some(format!(
        "fatal: cannot switch branch while {doing}\nConsider \"git {kind} --abort\" to give it up.\n"
    ))
}

/// Move HEAD to `branch`, saying so unless the caller is only passing through on its way
/// somewhere else, as `git rebase UPSTREAM BRANCH` does.
pub(crate) fn switch_to_branch(
    system: &mut dyn System,
    root: &str,
    branch: &str,
    announce: bool,
    io: &mut Io,
) -> i32 {
    let Some(commit) = repo::read_reference(system, root, &format!("refs/heads/{branch}")) else {
        io.print_err(&format!("fatal: invalid reference: {branch}\n"));
        return 128;
    };
    if repo::current_branch(system, root).as_deref() == Some(branch) {
        io.print_err(&format!("Already on '{branch}'\n"));
        return 0;
    }
    if let Err(status) = checkout_commit(system, root, &commit, io) {
        return status;
    }
    let from = repo::current_branch(system, root).unwrap_or_else(|| "HEAD".to_string());
    let action = format!("checkout: moving from {from} to {branch}");
    if let Err(error) = repo::set_head_to_branch(system, root, branch, &action) {
        return super::cannot_write(io, "HEAD", &error);
    }
    if announce {
        io.print_err(&format!("Switched to branch '{branch}'\n"));
    }
    0
}

fn switch_detached(system: &mut dyn System, root: &str, revision: &str, io: &mut Io) -> i32 {
    let Some(commit) = repo::resolve_revision(system, root, revision) else {
        io.print_err(&format!("fatal: invalid reference: {revision}\n"));
        return 128;
    };
    if let Err(status) = checkout_commit(system, root, &commit, io) {
        return status;
    }
    let action = format!("checkout: moving to {revision}");
    if let Err(error) = repo::set_head_detached(system, root, &commit, &action) {
        return super::cannot_write(io, "HEAD", &error);
    }
    io.print_err(&format!(
        "Note: switching to '{revision}'.\nHEAD is now at {} {}\n",
        repo::short(&commit),
        repo::load_commit(system, root, &commit)
            .map(|commit| commit.subject().to_string())
            .unwrap_or_default()
    ));
    0
}

/// Create `branch` at `start` and switch to it.
fn create_and_switch(
    system: &mut dyn System,
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
    let status = refs::git_branch(system, &arguments, io);
    if status != 0 {
        return status;
    }
    let status = switch_to_branch(system, root, branch, true, io);
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

pub(crate) fn git_switch(system: &mut dyn System, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(system) else {
        return repo_error(io);
    };
    let mut create = false;
    let mut force = false;
    let mut detach = false;
    let mut operands: Vec<String> = Vec::new();
    for argument in Flags::new(args).clustered("qcdf") {
        let name = match argument {
            Arg::Operand(value) => {
                operands.push(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "-c" | "--create" => create = true,
            "-C" | "--force-create" => {
                create = true;
                force = true;
            }
            "-d" | "--detach" => detach = true,
            "-f" | "--force" | "--discard-changes" => force = true,
            "-q" | "--quiet" | "--no-guess" | "--no-track" => {}
            _ => return usage(io, &format!("unsupported switch option: {name}")),
        }
    }
    match operands.as_slice() {
        [branch] if create => create_and_switch(system, &root, branch, None, force, io),
        [branch, start] if create => {
            create_and_switch(system, &root, branch, Some(start), force, io)
        }
        [revision] if detach => switch_detached(system, &root, revision, io),
        [branch] => {
            let branch = resolve_previous(system, &root, branch);
            match branch {
                Some(branch) => switch_to_branch(system, &root, &branch, true, io),
                None => {
                    io.print_err("fatal: no previous branch to switch to\n");
                    128
                }
            }
        }
        _ => usage(io, "usage: git switch [-c] BRANCH [START_POINT]"),
    }
}

/// Expand `-` into the branch that was checked out before the current one.
fn resolve_previous(system: &mut dyn System, root: &str, branch: &str) -> Option<String> {
    if branch != "-" {
        return Some(branch.to_string());
    }
    repo::previous_branch(system, root)
}

pub(crate) fn git_checkout(system: &mut dyn System, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(system) else {
        return repo_error(io);
    };
    let mut create = false;
    let mut force = false;
    let mut detach = false;
    let mut operands: Vec<String> = Vec::new();
    let mut paths: Vec<String> = Vec::new();
    let mut side = None;
    let mut flags = Flags::new(args).clustered("qbf");
    while let Some(argument) = flags.next() {
        let name = match argument {
            Arg::Operand(value) => {
                // Everything after `--` is a path; before it, a branch or revision.
                if flags.separated() {
                    paths.push(value);
                } else {
                    operands.push(value);
                }
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
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
            _ => return usage(io, &format!("unsupported checkout option: {name}")),
        }
    }
    if let Some(side) = side {
        let named = if paths.is_empty() { &operands } else { &paths };
        return checkout_side(system, &root, side, named, io);
    }
    if !paths.is_empty() {
        // `git checkout [REVISION] -- PATH...` restores files without moving HEAD.
        let source = operands.first().map(String::as_str);
        return super::worktree::git_restore(system, &restore_arguments(source, &paths), io);
    }
    match operands.as_slice() {
        [branch] if create => create_and_switch(system, &root, branch, None, force, io),
        [branch, start] if create => {
            create_and_switch(system, &root, branch, Some(start), force, io)
        }
        [revision] if detach => switch_detached(system, &root, revision, io),
        [target] => {
            if let Some(branch) = resolve_previous(system, &root, target) {
                if repo::read_reference(system, &root, &format!("refs/heads/{branch}")).is_some() {
                    return switch_to_branch(system, &root, &branch, true, io);
                }
            }
            if repo::resolve_revision(system, &root, target).is_some() {
                return switch_detached(system, &root, target, io);
            }
            if !super::names_a_path(system, &root, target) {
                io.print_err(&format!(
                    "error: pathspec '{target}' did not match any file(s) known to git\n"
                ));
                return 1;
            }
            // A bare path argument means "discard my changes to that path".
            super::worktree::git_restore(
                system,
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
    system: &mut dyn System,
    theirs: bool,
    named: &[String],
    io: &mut Io,
) -> i32 {
    let Some(root) = repo::find_repo_root(system) else {
        return repo_error(io);
    };
    let side = if theirs { Side::Theirs } else { Side::Ours };
    checkout_side(system, &root, side, named, io)
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
    system: &mut dyn System,
    root: &str,
    side: Side,
    named: &[String],
    io: &mut Io,
) -> i32 {
    let stages = conflict::load_stages(system, root);
    if named.is_empty() {
        return usage(io, "usage: git checkout --ours|--theirs PATH...");
    }
    let cwd = system.cwd().to_string();
    let mut chosen: Vec<(String, String)> = Vec::new();
    for operand in named {
        let path = super::pathspec(&cwd, root, operand);
        let Some(entry) = stages.get(&path) else {
            io.print_err(&format!(
                "error: path '{operand}' does not have their version\n"
            ));
            return 1;
        };
        let hash = match side {
            Side::Ours => entry.ours.clone(),
            Side::Theirs => entry.theirs.clone(),
        };
        // A side that deleted the file has nothing to check out, which Git reports the same way.
        let Some(hash) = hash else {
            io.print_err(&format!(
                "error: path '{operand}' does not have their version\n"
            ));
            return 1;
        };
        chosen.push((path, hash));
    }
    let count = chosen.len();
    for (path, hash) in chosen {
        let Some(data) = repo::read_blob(system, root, &hash) else {
            io.print_err(&format!("error: missing blob for '{path}'\n"));
            return 1;
        };
        if repo::write_vfs(system, &repo::path_join(root, &path), &data).is_err() {
            io.print_err(&format!("error: cannot write '{path}'\n"));
            return 1;
        }
    }
    let plural = if count == 1 { "" } else { "s" };
    io.print(&format!("Updated {count} path{plural} from the index\n"));
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
