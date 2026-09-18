//! Replaying a branch onto another commit with `git rebase`.
//!
//! A rebase is a run of cherry-picks: the commits the branch has and the upstream does not are
//! applied one at a time onto the new base, each with the three-way merge in [`super::conflict`].
//! What is left to do is written to `.git/REBASE_STATE`, so a rebase that stops at a conflict can
//! be continued, skipped past, or abandoned.

use crate::commands::{CommandContext, Io};

use super::conflict;
use super::history;
use super::repo::{self, Commit};
use super::{repo_error, usage, Globals};

const STATE: &str = "REBASE_STATE";

/// What a rebase still has to do.
struct State {
    /// The branch being rebased, which moves as each commit lands.
    branch: String,
    /// Where that branch pointed before the rebase, for `--abort`.
    original: String,
    /// The commits still to apply, oldest first; the first is the one in progress.
    todo: Vec<String>,
}

fn load(interp: &crate::interp::Interp, root: &str) -> Option<State> {
    let bytes = interp.vfs.read("/", &repo::git_path(root, STATE)).ok()?;
    let text = String::from_utf8_lossy(&bytes).to_string();
    let mut branch = String::new();
    let mut original = String::new();
    let mut todo = Vec::new();
    for line in text.lines() {
        match line.split_once(' ') {
            Some(("branch", value)) => branch = value.to_string(),
            Some(("orig", value)) => original = value.to_string(),
            Some(("todo", value)) => todo.push(value.to_string()),
            _ => {}
        }
    }
    (!branch.is_empty() && !original.is_empty()).then_some(State {
        branch,
        original,
        todo,
    })
}

fn store(ctx: &mut CommandContext<'_>, root: &str, state: &State) -> bool {
    let mut text = format!("branch {}\norig {}\n", state.branch, state.original);
    for id in &state.todo {
        text.push_str(&format!("todo {id}\n"));
    }
    repo::write_vfs(ctx, &repo::git_path(root, STATE), text.as_bytes()).is_ok()
}

fn clear(ctx: &mut CommandContext<'_>, root: &str) {
    let _ = ctx.vfs.remove_file("/", &repo::git_path(root, STATE));
    conflict::clear(ctx, root);
}

pub(crate) fn git_rebase(
    ctx: &mut CommandContext<'_>,
    globals: &Globals,
    args: &[String],
    io: &mut Io,
) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut onto = None;
    let mut operands: Vec<String> = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--continue" => return resume(ctx, &root, globals, io),
            "--abort" => return abort(ctx, &root, io),
            "--skip" => return skip(ctx, &root, globals, io),
            "-q" | "--quiet" | "--no-verify" | "--no-autostash" => {}
            "--onto" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return usage(io, "--onto requires a revision");
                };
                onto = Some(value.clone());
            }
            // Editing a rebase needs an editor, which the simulation does not have.
            "-i" | "--interactive" => {
                io.err
                    .extend_from_slice(b"fatal: interactive rebase is not supported here\n");
                return 128;
            }
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported rebase option: {value}"))
            }
            value => operands.push(value.to_string()),
        }
        index += 1;
    }
    if load(ctx, &root).is_some() {
        io.err.extend_from_slice(
            b"fatal: a rebase is already in progress\nhint: try \"git rebase --continue\" or \"git rebase --abort\"\n",
        );
        return 128;
    }
    let [upstream] = operands.as_slice() else {
        return usage(io, "usage: git rebase [--onto NEWBASE] UPSTREAM");
    };
    let Some(upstream_id) = repo::resolve_revision(ctx, &root, upstream) else {
        io.err
            .extend_from_slice(format!("fatal: invalid upstream '{upstream}'\n").as_bytes());
        return 128;
    };
    let onto_id = match onto {
        Some(revision) => match repo::resolve_revision(ctx, &root, &revision) {
            Some(id) => id,
            None => {
                io.err.extend_from_slice(
                    format!("fatal: invalid reference '{revision}'\n").as_bytes(),
                );
                return 128;
            }
        },
        None => upstream_id.clone(),
    };
    let Some(branch) = repo::current_branch(ctx, &root) else {
        io.err
            .extend_from_slice(b"fatal: rebasing a detached HEAD is not supported here\n");
        return 128;
    };
    let Some(head) = repo::head_commit(ctx, &root) else {
        io.err
            .extend_from_slice(b"fatal: no commit on the current branch to rebase\n");
        return 128;
    };
    if !history::blocking_changes(ctx, &root, &repo::head_tree(ctx, &root)).is_empty() {
        io.err.extend_from_slice(
            b"error: cannot rebase: You have unstaged changes.\nerror: Please commit or stash them.\n",
        );
        return 1;
    }
    // With the upstream already behind the branch there is nowhere new to put it, unless
    // `--onto` names somewhere else.
    let settled = onto_id == upstream_id
        && repo::merge_base(ctx, &root, &head, &upstream_id).as_deref()
            == Some(upstream_id.as_str());
    let todo = commits_to_replay(ctx, &root, &head, &upstream_id);
    if settled || todo.is_empty() {
        io.out
            .extend_from_slice(format!("Current branch {branch} is up to date.\n").as_bytes());
        return 0;
    }
    // The branch moves to the new base first, and each commit is replayed on top of it.
    if let Err(status) = lay_down(ctx, &root, &onto_id, io) {
        return status;
    }
    if repo::update_head(ctx, &root, &onto_id, &format!("rebase: checkout {onto_id}")).is_err() {
        return 1;
    }
    let state = State {
        branch,
        original: head,
        todo,
    };
    if !store(ctx, &root, &state) {
        return 1;
    }
    replay(ctx, &root, globals, state, io)
}

/// The commits `head` has that `upstream` does not, oldest first.
fn commits_to_replay(
    ctx: &mut CommandContext<'_>,
    root: &str,
    head: &str,
    upstream: &str,
) -> Vec<String> {
    let already = repo::ancestors(ctx, root, upstream);
    let mut listed: Vec<String> = repo::first_parent_history(ctx, root, head, 10_000)
        .into_iter()
        .map(|(id, _)| id)
        .take_while(|id| !already.contains(id))
        .collect();
    listed.reverse();
    listed
}

/// Move the working tree and index to a commit, keeping nothing of what was there.
fn lay_down(
    ctx: &mut CommandContext<'_>,
    root: &str,
    commit: &str,
    io: &mut Io,
) -> Result<(), i32> {
    let current = repo::head_tree(ctx, root);
    let target = repo::commit_tree(ctx, root, commit).unwrap_or_default();
    if let Err(error) = repo::replace_work_tree(ctx, root, &current, &target) {
        io.err
            .extend_from_slice(format!("git rebase: {error}\n").as_bytes());
        return Err(1);
    }
    if repo::store_index(ctx, root, &target).is_err() {
        return Err(1);
    }
    Ok(())
}

/// Apply the commits left in `state`, stopping at the first conflict.
fn replay(
    ctx: &mut CommandContext<'_>,
    root: &str,
    globals: &Globals,
    mut state: State,
    io: &mut Io,
) -> i32 {
    while let Some(id) = state.todo.first().cloned() {
        let Some(original) = repo::load_commit(ctx, root, &id) else {
            io.err
                .extend_from_slice(format!("fatal: bad object {id}\n").as_bytes());
            return 128;
        };
        let Some(head) = repo::head_commit(ctx, root) else {
            return 1;
        };
        let base = original
            .parents
            .first()
            .and_then(|parent| repo::commit_tree(ctx, root, parent))
            .unwrap_or_default();
        let theirs = repo::commit_tree(ctx, root, &id).unwrap_or_default();
        let ours = repo::commit_tree(ctx, root, &head).unwrap_or_default();
        let label = format!("{} ({})", repo::short(&id), original.subject());
        let combined = conflict::combine(ctx, root, &base, &ours, &theirs, "HEAD", &label);
        if let Err(error) = repo::update_work_tree(ctx, root, &ours, &combined.tree) {
            io.err
                .extend_from_slice(format!("git rebase: {error}\n").as_bytes());
            return 1;
        }
        if repo::store_index(ctx, root, &combined.tree).is_err() {
            return 1;
        }
        if !combined.stages.is_empty() {
            if !store(ctx, root, &state) || !conflict::store_stages(ctx, root, &combined.stages) {
                return 1;
            }
            for path in combined.stages.keys() {
                io.err.extend_from_slice(
                    format!("CONFLICT (content): Merge conflict in {path}\n").as_bytes(),
                );
            }
            io.err.extend_from_slice(
                format!(
                    "error: could not apply {label}\n\
                     hint: Resolve all conflicts manually, mark them as resolved with\n\
                     hint: \"git add/rm <pathspec>\", then run \"git rebase --continue\".\n\
                     hint: You can instead skip this commit: run \"git rebase --skip\".\n\
                     hint: To abort and get back to the state before \"git rebase\", run\n\
                     hint: \"git rebase --abort\".\n"
                )
                .as_bytes(),
            );
            return 1;
        }
        // A commit whose change is already in the new base is dropped, as Git drops it.
        if combined.tree != ours {
            if let Err(status) = record(ctx, root, &original, &combined.tree, io) {
                return status;
            }
        }
        state.todo.remove(0);
        if !store(ctx, root, &state) {
            return 1;
        }
    }
    let _ = globals;
    clear(ctx, root);
    io.out.extend_from_slice(
        format!(
            "Successfully rebased and updated refs/heads/{}.\n",
            state.branch
        )
        .as_bytes(),
    );
    0
}

/// Write one replayed commit, keeping the author and message of the original.
fn record(
    ctx: &mut CommandContext<'_>,
    root: &str,
    original: &Commit,
    tree: &repo::Tree,
    io: &mut Io,
) -> Result<(), i32> {
    let Some(head) = repo::head_commit(ctx, root) else {
        return Err(1);
    };
    let replayed = Commit {
        parents: vec![head],
        author_name: original.author_name.clone(),
        author_email: original.author_email.clone(),
        timestamp: original.timestamp,
        message: original.message.clone(),
    };
    let Ok(id) = repo::store_commit(ctx, root, &replayed, tree) else {
        io.err
            .extend_from_slice(b"git rebase: cannot record commit\n");
        return Err(1);
    };
    let action = format!("rebase: {}", replayed.subject());
    if repo::update_head(ctx, root, &id, &action).is_err() {
        return Err(1);
    }
    Ok(())
}

/// Finish the commit the user has settled, then carry on.
fn resume(ctx: &mut CommandContext<'_>, root: &str, globals: &Globals, io: &mut Io) -> i32 {
    let Some(mut state) = load(ctx, root) else {
        io.err.extend_from_slice(b"fatal: No rebase in progress?\n");
        return 128;
    };
    if !conflict::load_stages(ctx, root).is_empty() {
        io.err.extend_from_slice(
            b"error: you have unmerged paths\nhint: Mark them resolved with \"git add/rm <pathspec>\".\n",
        );
        return 1;
    }
    let Some(id) = state.todo.first().cloned() else {
        clear(ctx, root);
        return 0;
    };
    let Some(original) = repo::load_commit(ctx, root, &id) else {
        return 128;
    };
    let staged = repo::load_index(ctx, root).unwrap_or_default();
    let head_tree = repo::head_tree(ctx, root);
    if staged != head_tree {
        if let Err(status) = record(ctx, root, &original, &staged, io) {
            return status;
        }
    }
    state.todo.remove(0);
    conflict::clear(ctx, root);
    if !store(ctx, root, &state) {
        return 1;
    }
    replay(ctx, root, globals, state, io)
}

/// Drop the commit that stopped the rebase and carry on without it.
fn skip(ctx: &mut CommandContext<'_>, root: &str, globals: &Globals, io: &mut Io) -> i32 {
    let Some(mut state) = load(ctx, root) else {
        io.err.extend_from_slice(b"fatal: No rebase in progress?\n");
        return 128;
    };
    let Some(head) = repo::head_commit(ctx, root) else {
        return 1;
    };
    if !state.todo.is_empty() {
        state.todo.remove(0);
    }
    conflict::clear(ctx, root);
    if let Err(status) = lay_down(ctx, root, &head, io) {
        return status;
    }
    if !store(ctx, root, &state) {
        return 1;
    }
    replay(ctx, root, globals, state, io)
}

/// Put the branch back where it was before the rebase started.
fn abort(ctx: &mut CommandContext<'_>, root: &str, io: &mut Io) -> i32 {
    let Some(state) = load(ctx, root) else {
        io.err.extend_from_slice(b"fatal: No rebase in progress?\n");
        return 128;
    };
    if let Err(status) = lay_down(ctx, root, &state.original, io) {
        return status;
    }
    if repo::update_head(ctx, root, &state.original, "rebase: aborted").is_err() {
        return 1;
    }
    clear(ctx, root);
    0
}
