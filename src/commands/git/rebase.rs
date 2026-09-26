//! Replaying a branch onto another commit with `git rebase`.
//!
//! A rebase is a run of cherry-picks: the commits the branch has and the upstream does not are
//! applied one at a time onto the new base, each with the three-way merge in [`super::conflict`].
//! What is left to do is written to `.git/REBASE_STATE`, so a rebase that stops at a conflict can
//! be continued, skipped past, or abandoned.

use crate::commands::Io;
use crate::syscalls::System;

use super::commit;
use super::conflict;
use super::repo::{self, Commit};
use super::switch;
use super::{cannot_write, repo_error, require_index, usage, Arg, Flags, Globals};

const STATE: &str = "REBASE_STATE";

/// What a rebase still has to do.
struct State {
    /// The branch being rebased, which moves as each commit lands.
    branch: String,
    /// Where that branch pointed before the rebase, for `--abort`.
    original: String,
    /// The commit the branch is being replayed onto, which `git status` names.
    onto: String,
    /// The commits still to apply, oldest first; the first is the one in progress.
    todo: Vec<String>,
}

fn load(system: &mut dyn System, root: &str) -> Option<State> {
    let bytes = repo::read_all(system, &repo::git_path(root, STATE))?;
    let text = String::from_utf8_lossy(&bytes).to_string();
    let mut branch = String::new();
    let mut original = String::new();
    let mut onto = String::new();
    let mut todo = Vec::new();
    for line in text.lines() {
        match line.split_once(' ') {
            Some(("branch", value)) => branch = value.to_string(),
            Some(("orig", value)) => original = value.to_string(),
            Some(("onto", value)) => onto = value.to_string(),
            Some(("todo", value)) => todo.push(value.to_string()),
            _ => {}
        }
    }
    (!branch.is_empty() && !original.is_empty()).then_some(State {
        branch,
        original,
        onto,
        todo,
    })
}

fn store(system: &mut dyn System, root: &str, state: &State) -> bool {
    let mut text = format!(
        "branch {}\norig {}\nonto {}\n",
        state.branch, state.original, state.onto
    );
    for id in &state.todo {
        text.push_str(&format!("todo {id}\n"));
    }
    repo::write_vfs(system, &repo::git_path(root, STATE), text.as_bytes()).is_ok()
}

/// Whether a rebase has started and not yet finished.
pub(crate) fn in_progress(system: &mut dyn System, root: &str) -> bool {
    load(system, root).is_some()
}

/// The branch a rebase is replaying and the commit it is replaying onto.
pub(crate) fn replaying(system: &mut dyn System, root: &str) -> Option<(String, String)> {
    load(system, root).map(|state| (state.branch, state.onto))
}

fn clear(system: &mut dyn System, root: &str) {
    let _ = system.unlink("/", &repo::git_path(root, STATE));
    conflict::clear(system, root);
}

pub(crate) fn git_rebase(
    system: &mut dyn System,
    globals: &Globals,
    args: &[String],
    io: &mut Io,
) -> i32 {
    let Some(root) = repo::find_repo_root(system) else {
        return repo_error(io);
    };
    let mut onto = None;
    let mut operands: Vec<String> = Vec::new();
    let mut flags = Flags::new(args);
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                operands.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "--continue" => return resume(system, &root, globals, io),
            "--abort" => return abort(system, &root, io),
            "--skip" => return skip(system, &root, globals, io),
            "-q" | "--quiet" | "--no-verify" | "--no-autostash" => {}
            "--onto" => {
                let Some(value) = flags.value(attached) else {
                    return usage(io, "--onto requires a revision");
                };
                onto = Some(value);
            }
            // Editing a rebase needs an editor, which the simulation does not have.
            "-i" | "--interactive" => {
                io.print_err("fatal: interactive rebase is not supported here\n");
                return 128;
            }
            _ => return usage(io, &format!("unsupported rebase option: {name}")),
        }
    }
    if load(system, &root).is_some() {
        io.print_err("fatal: a rebase is already in progress\nhint: try \"git rebase --continue\" or \"git rebase --abort\"\n");
        return 128;
    }
    // `git rebase UPSTREAM BRANCH` means "check out BRANCH first", which is how a rebase is
    // written when the branch to move is not the one currently checked out.
    let (upstream, wanted) = match operands.as_slice() {
        [upstream] => (upstream.clone(), None),
        [upstream, branch] => (upstream.clone(), Some(branch.clone())),
        _ => return usage(io, "usage: git rebase [--onto NEWBASE] UPSTREAM [BRANCH]"),
    };
    let upstream = &upstream;
    let Some(upstream_id) = repo::resolve_revision(system, &root, upstream) else {
        io.print_err(&format!("fatal: invalid upstream '{upstream}'\n"));
        return 128;
    };
    // Git names the new base in the reflog as the user wrote it, which is what makes it readable.
    let base_name = onto.clone().unwrap_or_else(|| upstream.clone());
    let onto_id = match onto {
        Some(revision) => match repo::resolve_revision(system, &root, &revision) {
            Some(id) => id,
            None => {
                io.print_err(&format!("fatal: invalid reference '{revision}'\n"));
                return 128;
            }
        },
        None => upstream_id.clone(),
    };
    // A rebase rewrites the whole tree, so anything not committed has nowhere to go.
    let snapshot = match switch::snapshot(system, &root, io) {
        Ok(snapshot) => snapshot,
        Err(status) => return status,
    };
    let dirty = switch::dirty_paths(&snapshot);
    if !dirty.is_empty() {
        let what = if switch::only_staged(&snapshot, &dirty) {
            "Your index contains uncommitted changes."
        } else {
            "You have unstaged changes."
        };
        io.print_err(&format!(
            "error: cannot rebase: {what}\nerror: Please commit or stash them.\n"
        ));
        return 1;
    }
    if let Some(wanted) = wanted {
        let status = switch::switch_to_branch(system, &root, &wanted, false, io);
        if status != 0 {
            return status;
        }
    }
    let Some(branch) = repo::current_branch(system, &root) else {
        io.print_err("fatal: rebasing a detached HEAD is not supported here\n");
        return 128;
    };
    let Some(head) = repo::head_commit(system, &root) else {
        io.print_err("fatal: no commit on the current branch to rebase\n");
        return 128;
    };
    // With the upstream already behind the branch there is nowhere new to put it, unless
    // `--onto` names somewhere else.
    let settled = onto_id == upstream_id
        && repo::merge_base(system, &root, &head, &upstream_id).as_deref()
            == Some(upstream_id.as_str());
    let todo = commits_to_replay(system, &root, &head, &upstream_id);
    if settled || (todo.is_empty() && head == onto_id) {
        io.print(&format!("Current branch {branch} is up to date.\n"));
        return 0;
    }
    repo::record_orig_head(system, &root);
    if todo.is_empty() {
        // The branch has nothing the new base lacks, so moving it there is all the rebase is.
        if let Err(status) = lay_down(system, &root, &onto_id, io) {
            return status;
        }
        let action = format!("rebase (finish): returning to refs/heads/{branch}");
        if let Err(error) = repo::update_head(system, &root, &onto_id, &action) {
            return cannot_write(io, "HEAD", &error);
        }
        io.print(&format!(
            "Successfully rebased and updated refs/heads/{branch}.\n"
        ));
        return 0;
    }
    // The branch moves to the new base first, and each commit is replayed on top of it.
    if let Err(status) = lay_down(system, &root, &onto_id, io) {
        return status;
    }
    // Git detaches HEAD for the duration, so the branch keeps naming its old tip until the
    // replay succeeds and nothing that reads the branch sees a half-finished history.
    let action = format!("rebase (start): checkout {base_name}");
    if let Err(error) = repo::set_head_detached(system, &root, &onto_id, &action) {
        return cannot_write(io, "HEAD", &error);
    }
    let state = State {
        branch,
        original: head,
        onto: onto_id.clone(),
        todo,
    };
    if !store(system, &root, &state) {
        return 1;
    }
    replay(system, &root, globals, state, io)
}

/// The commits `head` has that `upstream` does not, oldest first.
fn commits_to_replay(
    system: &mut dyn System,
    root: &str,
    head: &str,
    upstream: &str,
) -> Vec<String> {
    let already = repo::ancestors(system, root, upstream);
    let mut listed: Vec<String> = repo::first_parent_history(system, root, head, 10_000)
        .into_iter()
        .map(|(id, _)| id)
        .take_while(|id| !already.contains(id))
        .collect();
    listed.reverse();
    listed
}

/// Move the working tree and index to a commit, keeping nothing of what was there.
fn lay_down(system: &mut dyn System, root: &str, commit: &str, io: &mut Io) -> Result<(), i32> {
    let target = super::require_tree(system, root, commit, io)?;
    let snapshot = switch::snapshot(system, root, io)?;
    if switch::refuse_untracked_overwrite(&snapshot, &target, "checkout", "switch branches", io) {
        return Err(1);
    }
    if let Err(error) = repo::replace_work_tree(system, root, &snapshot.head, &target) {
        io.print_err(&format!("git rebase: {error}\n"));
        return Err(1);
    }
    if let Err(error) = repo::store_index(system, root, &target) {
        return Err(cannot_write(io, "the index", &error));
    }
    Ok(())
}

/// Apply the commits left in `state`, stopping at the first conflict.
fn replay(
    system: &mut dyn System,
    root: &str,
    globals: &Globals,
    mut state: State,
    io: &mut Io,
) -> i32 {
    while let Some(id) = state.todo.first().cloned() {
        let Some(original) = repo::load_commit(system, root, &id) else {
            io.print_err(&format!("fatal: bad object {id}\n"));
            return 128;
        };
        let Some(head) = repo::head_commit(system, root) else {
            return 1;
        };
        let base = match super::parent_tree(system, root, original.parents.first(), io) {
            Ok(tree) => tree,
            Err(status) => return status,
        };
        let (theirs, ours) = match (
            super::require_tree(system, root, &id, io),
            super::require_tree(system, root, &head, io),
        ) {
            (Ok(theirs), Ok(ours)) => (theirs, ours),
            (Err(status), _) | (_, Err(status)) => return status,
        };
        let label = format!("{} ({})", repo::short(&id), original.subject());
        let combined = match conflict::combine(system, root, &base, &ours, &theirs, "HEAD", &label)
        {
            Ok(combined) => combined,
            Err(failed) => return cannot_write(io, &failed.path, &failed.error),
        };
        if let Err(error) = repo::update_work_tree(system, root, &ours, &combined.tree) {
            io.print_err(&format!("git rebase: {error}\n"));
            return 1;
        }
        if let Err(error) = repo::store_index(system, root, &combined.tree) {
            return cannot_write(io, "the index", &error);
        }
        if !combined.stages.is_empty() {
            if !store(system, root, &state)
                || !conflict::store_stages(system, root, &combined.stages)
            {
                return 1;
            }
            for path in combined.stages.keys() {
                io.print_err(&format!("CONFLICT (content): Merge conflict in {path}\n"));
            }
            io.print_err(&format!(
                "error: could not apply {label}\n\
                     hint: Resolve all conflicts manually, mark them as resolved with\n\
                     hint: \"git add/rm <pathspec>\", then run \"git rebase --continue\".\n\
                     hint: You can instead skip this commit: run \"git rebase --skip\".\n\
                     hint: To abort and get back to the state before \"git rebase\", run\n\
                     hint: \"git rebase --abort\".\n"
            ));
            return 1;
        }
        // A commit whose change is already in the new base is dropped, as Git drops it.
        if combined.tree != ours {
            if let Err(status) = record(system, root, &original, &combined.tree, "pick", io) {
                return status;
            }
        }
        state.todo.remove(0);
        if !store(system, root, &state) {
            return 1;
        }
    }
    let _ = globals;
    if let Err(status) = land(system, root, &state, io) {
        return status;
    }
    clear(system, root);
    io.print(&format!(
        "Successfully rebased and updated refs/heads/{}.\n",
        state.branch
    ));
    0
}

/// Move the rebased branch to the commit the replay ended on and put HEAD back on it.
fn land(system: &mut dyn System, root: &str, state: &State, io: &mut Io) -> Result<(), i32> {
    let Some(tip) = repo::head_commit(system, root) else {
        return Err(1);
    };
    let reference = format!("refs/heads/{}", state.branch);
    if repo::write_reference(system, root, &reference, &tip).is_err() {
        io.print_err(&format!("git rebase: cannot update {reference}\n"));
        return Err(1);
    }
    let action = format!("rebase (finish): returning to {reference}");
    if let Err(error) = repo::set_head_to_branch(system, root, &state.branch, &action) {
        return Err(cannot_write(io, "HEAD", &error));
    }
    Ok(())
}

/// Write one replayed commit, keeping the author and message of the original.
fn record(
    system: &mut dyn System,
    root: &str,
    original: &Commit,
    tree: &repo::Tree,
    verb: &str,
    io: &mut Io,
) -> Result<String, i32> {
    let Some(head) = repo::head_commit(system, root) else {
        return Err(1);
    };
    let replayed = Commit {
        parents: vec![head],
        author_name: original.author_name.clone(),
        author_email: original.author_email.clone(),
        timestamp: original.timestamp,
        message: original.message.clone(),
    };
    let id = match repo::store_commit(system, root, &replayed, tree) {
        Ok(id) => id,
        Err(error) => return Err(cannot_write(io, "the commit", &error)),
    };
    let action = format!("rebase ({verb}): {}", replayed.subject());
    if let Err(error) = repo::update_head(system, root, &id, &action) {
        return Err(cannot_write(io, "HEAD", &error));
    }
    Ok(id)
}

/// Finish the commit the user has settled, then carry on.
fn resume(system: &mut dyn System, root: &str, globals: &Globals, io: &mut Io) -> i32 {
    let Some(mut state) = load(system, root) else {
        io.print_err("fatal: No rebase in progress?\n");
        return 128;
    };
    if !conflict::load_stages(system, root).is_empty() {
        io.print_err("error: you have unmerged paths\nhint: Mark them resolved with \"git add/rm <pathspec>\".\n");
        return 1;
    }
    let Some(id) = state.todo.first().cloned() else {
        clear(system, root);
        return 0;
    };
    let Some(original) = repo::load_commit(system, root, &id) else {
        return 128;
    };
    let staged = match require_index(system, root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    let head_tree = repo::head_tree(system, root);
    if staged != head_tree {
        // Git reports the commit the user just settled, which is the only one it names by hand.
        let id = match record(system, root, &original, &staged, "continue", io) {
            Ok(id) => id,
            Err(status) => return status,
        };
        io.print(&format!(
            "[detached HEAD {}] {}\n",
            repo::short(&id),
            original.subject()
        ));
        commit::emit_commit_summary(system, root, &head_tree, &staged, io);
    }
    state.todo.remove(0);
    conflict::clear(system, root);
    if !store(system, root, &state) {
        return 1;
    }
    replay(system, root, globals, state, io)
}

/// Drop the commit that stopped the rebase and carry on without it.
fn skip(system: &mut dyn System, root: &str, globals: &Globals, io: &mut Io) -> i32 {
    let Some(mut state) = load(system, root) else {
        io.print_err("fatal: No rebase in progress?\n");
        return 128;
    };
    let Some(head) = repo::head_commit(system, root) else {
        return 1;
    };
    if !state.todo.is_empty() {
        state.todo.remove(0);
    }
    conflict::clear(system, root);
    if let Err(status) = lay_down(system, root, &head, io) {
        return status;
    }
    if !store(system, root, &state) {
        return 1;
    }
    replay(system, root, globals, state, io)
}

/// Put the branch back where it was before the rebase started.
fn abort(system: &mut dyn System, root: &str, io: &mut Io) -> i32 {
    let Some(state) = load(system, root) else {
        io.print_err("fatal: No rebase in progress?\n");
        return 128;
    };
    if let Err(status) = lay_down(system, root, &state.original, io) {
        return status;
    }
    // HEAD is detached mid-rebase, so putting it back means naming the branch again.
    let reference = format!("refs/heads/{}", state.branch);
    if let Err(error) = repo::write_reference(system, root, &reference, &state.original) {
        return cannot_write(io, &reference, &error);
    }
    let action = format!("rebase (abort): returning to {reference}");
    if let Err(error) = repo::set_head_to_branch(system, root, &state.branch, &action) {
        return cannot_write(io, "HEAD", &error);
    }
    clear(system, root);
    0
}
