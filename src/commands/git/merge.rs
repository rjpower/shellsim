//! `git merge`, `git cherry-pick`, and `git revert`.
//!
//! All three replay one tree onto another, so they share a sequencer that records the commits
//! still to apply, pauses on conflicts, and lets the operation be continued or aborted.

use crate::commands::Io;
use crate::syscalls::System;

use super::commit;
use super::compare::{self, Format, Options};
use super::conflict::{self, Stages};
use super::repo::{self, Commit};
use super::switch;
use super::{repo_error, usage, Arg, Flags, Globals};

pub(crate) fn git_merge(
    system: &mut dyn System,
    globals: &Globals,
    args: &[String],
    io: &mut Io,
) -> i32 {
    let Some(root) = repo::find_repo_root(system) else {
        return repo_error(io);
    };
    let mut no_fast_forward = false;
    let mut fast_forward_only = false;
    let mut message = None;
    let mut operands: Vec<String> = Vec::new();
    let mut flags = Flags::new(args).clustered("q").valued("m");
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                operands.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "--no-ff" => no_fast_forward = true,
            "--ff-only" => fast_forward_only = true,
            "--ff" | "--no-edit" | "-q" | "--quiet" => {}
            "--abort" => return abort_pending(system, &root, "merge", io),
            "--continue" => return continue_pending(system, &root, globals, "merge", io),
            "-m" | "--message" => message = flags.value(attached),
            _ => return usage(io, &format!("unsupported merge option: {name}")),
        }
    }
    let [target] = operands.as_slice() else {
        return usage(io, "usage: git merge [--no-ff|--ff-only] BRANCH");
    };
    let Some(other) = repo::resolve_revision(system, &root, target) else {
        io.print_err(&format!("merge: {target} - not something we can merge\n"));
        return 1;
    };
    let Some(head) = repo::head_commit(system, &root) else {
        io.print_err("fatal: no commit on the current branch to merge into\n");
        return 128;
    };
    if repo::ancestors(system, &root, &head).contains(&other) {
        io.print("Already up to date.\n");
        return 0;
    }
    let incoming = match super::require_tree(system, &root, &other, io) {
        Ok(tree) => tree,
        Err(status) => return status,
    };
    let snapshot = match switch::snapshot(system, &root, io) {
        Ok(snapshot) => snapshot,
        Err(status) => return status,
    };
    if !switch::blocking_changes(&snapshot, &incoming).is_empty() {
        io.print_err("error: Your local changes would be overwritten by merge.\n");
        return 1;
    }
    if switch::refuse_untracked_overwrite(&snapshot, &incoming, "merge", "merge", io) {
        return 1;
    }
    repo::record_orig_head(system, &root);
    let base = repo::merge_base(system, &root, &head, &other);
    let fast_forward = base.as_deref() == Some(head.as_str());
    if fast_forward && !no_fast_forward {
        if let Err(status) = switch::checkout_commit(system, &root, &other, io) {
            return status;
        }
        let action = format!("merge {target}: Fast-forward");
        if let Err(error) = repo::update_head(system, &root, &other, &action) {
            return super::cannot_write(io, "HEAD", &error);
        }
        io.print(&format!(
            "Updating {}..{}\nFast-forward\n",
            repo::short(&head),
            repo::short(&other)
        ));
        let before = repo::commit_tree(system, &root, &head).unwrap_or_default();
        let after = repo::commit_tree(system, &root, &other).unwrap_or_default();
        let options = Options {
            format: Format::Stat,
            ..Options::default()
        };
        compare::emit(system, &root, &before, &after, &options, io);
        commit::emit_mode_lines(system, &root, &before, &after, io);
        return 0;
    }
    if fast_forward_only {
        io.print_err("fatal: Not possible to fast-forward, aborting.\n");
        return 128;
    }
    let base_tree = match super::parent_tree(system, &root, base.as_ref(), io) {
        Ok(tree) => tree,
        Err(status) => return status,
    };
    let head_tree = match super::require_tree(system, &root, &head, io) {
        Ok(tree) => tree,
        Err(status) => return status,
    };
    let other_tree = incoming;
    let subject = message.unwrap_or_else(|| {
        // Git names the branch merged into unless it is the repository's default.
        match repo::current_branch(system, &root).filter(|branch| branch != repo::DEFAULT_BRANCH) {
            Some(branch) => format!("Merge branch '{target}' into {branch}"),
            None => format!("Merge branch '{target}'"),
        }
    });
    let combined = conflict::combine(
        system,
        &root,
        &base_tree,
        &head_tree,
        &other_tree,
        "HEAD",
        target,
    );
    let combined = match combined {
        Ok(combined) => combined,
        Err(failed) => return super::cannot_write(io, &failed.path, &failed.error),
    };
    let merged = combined.tree;
    if let Err(error) = repo::update_work_tree(system, &root, &head_tree, &merged) {
        io.print_err(&format!("git merge: {error}\n"));
        return 1;
    }
    if let Err(error) = repo::store_index(system, &root, &merged) {
        return super::cannot_write(io, "the index", &error);
    }
    if !combined.stages.is_empty() {
        return pause_for_conflicts(
            system,
            &root,
            &Paused {
                kind: conflict::MERGE_HEAD,
                commit: &other,
                message: &subject,
                theirs_label: target,
                advice: "Automatic merge failed; fix conflicts and then commit the result.",
            },
            &combined.stages,
            io,
        );
    }
    let (author_name, author_email) = repo::author_identity(system, &root, globals);
    let commit = Commit {
        parents: vec![head.clone(), other.clone()],
        author_name,
        author_email,
        timestamp: repo::now_seconds(system),
        message: subject,
    };
    let id = match repo::store_commit(system, &root, &commit, &merged) {
        Ok(id) => id,
        Err(error) => return super::cannot_write(io, "the merge commit", &error),
    };
    if let Err(error) = repo::update_head(system, &root, &id, &format!("merge {target}")) {
        return super::cannot_write(io, "HEAD", &error);
    }
    io.print("Merge made by the 'ort' strategy.\n");
    // Git shows the per-file diffstat of the merge before the summary.
    let stat = compare::Options {
        format: Format::Stat,
        ..Default::default()
    };
    compare::emit(system, &root, &head_tree, &merged, &stat, io);
    commit::emit_mode_lines(system, &root, &head_tree, &merged, io);
    0
}

// -- replaying single commits -----------------------------------------------------------------

/// Apply one commit's change to the current branch, or undo it.
///
/// Both commands are the same three-way merge with different corners: a cherry-pick treats the
/// commit's parent as the base and the commit as the incoming side, and a revert swaps those two,
/// which turns replaying a change into undoing it.
pub(crate) fn git_replay(
    system: &mut dyn System,
    globals: &Globals,
    revert: bool,
    args: &[String],
    io: &mut Io,
) -> i32 {
    let name = if revert { "revert" } else { "cherry-pick" };
    let Some(root) = repo::find_repo_root(system) else {
        return repo_error(io);
    };
    let mut no_commit = false;
    let mut mainline: Option<usize> = None;
    let mut operands: Vec<String> = Vec::new();
    let mut flags = Flags::new(args).clustered("neq").valued("m");
    while let Some(argument) = flags.next() {
        let (option, attached) = match argument {
            Arg::Operand(value) => {
                operands.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match option.as_str() {
            // Replaying a merge means saying which of its parents the change is measured against.
            "-m" | "--mainline" => {
                let parsed = flags
                    .value(attached)
                    .and_then(|value| value.parse::<usize>().ok());
                let Some(parent) = parsed.filter(|parent| *parent > 0) else {
                    return usage(io, &format!("{name} -m wants a parent number"));
                };
                mainline = Some(parent);
            }
            "-n" | "--no-commit" => no_commit = true,
            "-e" | "--edit" | "--no-edit" | "-q" | "--quiet" => {}
            "--abort" | "--quit" => return abort_sequence(system, &root, name, io),
            // Skipping abandons the commit in flight, then the rest of the list carries on.
            "--skip" => {
                let status = abort_pending(system, &root, name, io);
                return if status == 0 {
                    resume_sequence(system, globals, &root, io)
                } else {
                    status
                };
            }
            "--continue" => {
                let status = continue_pending(system, &root, globals, name, io);
                return if status == 0 {
                    resume_sequence(system, globals, &root, io)
                } else {
                    status
                };
            }
            _ => return usage(io, &format!("unsupported {name} option: {option}")),
        }
    }
    if operands.is_empty() {
        return usage(io, &format!("usage: git {name} [-n] [-m PARENT] COMMIT..."));
    }
    let Some(original) = repo::head_commit(system, &root) else {
        io.print_err(&format!("fatal: {name} needs a commit to apply onto\n"));
        return 128;
    };
    let mut todo = Vec::new();
    for revision in &operands {
        let Some(id) = repo::resolve_revision(system, &root, revision) else {
            io.print_err(&format!("fatal: bad revision '{revision}'\n"));
            return 128;
        };
        todo.push(id);
    }
    run_sequence(
        system,
        globals,
        &root,
        Sequence {
            revert,
            no_commit,
            mainline,
            original,
            todo,
        },
        io,
    )
}

/// What a `git cherry-pick A B C` still has to replay, so a conflict does not lose the rest.
struct Sequence {
    revert: bool,
    no_commit: bool,
    /// Which parent of a merge commit the change is measured against, for `-m`.
    mainline: Option<usize>,
    /// Where HEAD was before the first commit was replayed, which `--abort` returns to.
    original: String,
    /// The commits left to apply, oldest first.
    todo: Vec<String>,
}

/// How Git words its messages about a replay, which is also the command's name.
fn replay_name(revert: bool) -> &'static str {
    if revert {
        "revert"
    } else {
        "cherry-pick"
    }
}

const SEQUENCER: &str = "SEQUENCER";

fn load_sequence(system: &mut dyn System, root: &str) -> Option<Sequence> {
    let bytes = repo::read_all(system, &repo::git_path(root, SEQUENCER))?;
    let text = String::from_utf8_lossy(&bytes).to_string();
    let mut sequence = Sequence {
        revert: false,
        no_commit: false,
        mainline: None,
        original: String::new(),
        todo: Vec::new(),
    };
    for line in text.lines() {
        match line.split_once(' ') {
            Some(("revert", value)) => sequence.revert = value == "1",
            Some(("nocommit", value)) => sequence.no_commit = value == "1",
            // A zero means no `-m` was given; parent numbers start at one.
            Some(("mainline", value)) => {
                sequence.mainline = value.parse().ok().filter(|parent| *parent > 0)
            }
            Some(("orig", value)) => sequence.original = value.to_string(),
            Some(("todo", value)) => sequence.todo.push(value.to_string()),
            _ => {}
        }
    }
    (!sequence.original.is_empty()).then_some(sequence)
}

fn store_sequence(system: &mut dyn System, root: &str, sequence: &Sequence) -> bool {
    let mut text = format!(
        "revert {}\nnocommit {}\nmainline {}\norig {}\n",
        u8::from(sequence.revert),
        u8::from(sequence.no_commit),
        sequence.mainline.unwrap_or(0),
        sequence.original
    );
    for id in &sequence.todo {
        text.push_str(&format!("todo {id}\n"));
    }
    repo::write_vfs(system, &repo::git_path(root, SEQUENCER), text.as_bytes()).is_ok()
}

fn clear_sequence(system: &mut dyn System, root: &str) {
    let _ = system.unlink("/", &repo::git_path(root, SEQUENCER));
}

/// Replay each commit in turn, remembering what is left if one of them stops.
fn run_sequence(
    system: &mut dyn System,
    globals: &Globals,
    root: &str,
    mut sequence: Sequence,
    io: &mut Io,
) -> i32 {
    while !sequence.todo.is_empty() {
        let id = sequence.todo.remove(0);
        let status = replay_one(system, globals, root, &sequence, &id, io);
        if status != 0 {
            store_sequence(system, root, &sequence);
            return status;
        }
    }
    clear_sequence(system, root);
    0
}

/// Carry on with the commits a stopped cherry-pick or revert had left.
fn resume_sequence(system: &mut dyn System, globals: &Globals, root: &str, io: &mut Io) -> i32 {
    match load_sequence(system, root) {
        Some(sequence) => run_sequence(system, globals, root, sequence, io),
        None => 0,
    }
}

/// Abandon the whole sequence, putting the branch back where it started.
fn abort_sequence(system: &mut dyn System, root: &str, name: &str, io: &mut Io) -> i32 {
    let Some(sequence) = load_sequence(system, root) else {
        return abort_pending(system, root, name, io);
    };
    let head = repo::head_tree(system, root);
    let target = match super::require_tree(system, root, &sequence.original, io) {
        Ok(tree) => tree,
        Err(status) => return status,
    };
    let mut previous = match super::require_index(system, root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    for (path, entry) in &head {
        previous
            .entry(path.clone())
            .or_insert_with(|| entry.clone());
    }
    if let Err(error) = repo::replace_work_tree(system, root, &previous, &target) {
        io.print_err(&format!("git {name}: {error}\n"));
        return 1;
    }
    if let Err(error) = repo::store_index(system, root, &target) {
        return super::cannot_write(io, "the index", &error);
    }
    let action = format!("{name}: aborted");
    if let Err(error) = repo::update_head(system, root, &sequence.original, &action) {
        return super::cannot_write(io, "HEAD", &error);
    }
    conflict::clear(system, root);
    clear_sequence(system, root);
    0
}

fn replay_one(
    system: &mut dyn System,
    globals: &Globals,
    root: &str,
    sequence: &Sequence,
    revision: &str,
    io: &mut Io,
) -> i32 {
    let Sequence {
        revert,
        no_commit,
        mainline,
        ..
    } = *sequence;
    let name = replay_name(revert);
    if pending_operation(system, root).is_some() {
        io.print_err(&format!(
                "error: a {name} is already in progress\nhint: try \"git {name} --continue\" or \"git {name} --abort\"\n"
            ));
        return 128;
    }
    let Some(id) = repo::resolve_revision(system, root, revision) else {
        io.print_err(&format!("fatal: bad revision '{revision}'\n"));
        return 128;
    };
    let Some(commit) = repo::load_commit(system, root, &id) else {
        io.print_err(&format!("fatal: bad object {revision}\n"));
        return 128;
    };
    if commit.parents.len() > 1 && mainline.is_none() {
        io.print_err(&format!(
            "error: commit {id} is a merge but no -m option was given.\nfatal: {name} failed\n"
        ));
        return 128;
    }
    if let Some(parent) = mainline.filter(|_| commit.parents.len() < 2) {
        io.print_err(&format!(
            "error: mainline was specified but commit {id} is not a merge.\nfatal: {name} failed\n"
        ));
        let _ = parent;
        return 128;
    }
    let against = mainline.unwrap_or(1);
    // A root commit has no parent, and Git measures it against the empty tree rather than
    // refusing: replaying the first commit of a history is an ordinary thing to ask for.
    if against > commit.parents.len() && !commit.parents.is_empty() {
        io.print_err(&format!(
            "error: commit {id} does not have parent {against}\nfatal: {name} failed\n"
        ));
        return 128;
    }
    let Some(head) = repo::head_commit(system, root) else {
        io.print_err(&format!("fatal: {name} needs a commit to apply onto\n"));
        return 128;
    };
    let commit_tree = match super::require_tree(system, root, &id, io) {
        Ok(tree) => tree,
        Err(status) => return status,
    };
    let parent_tree = match super::parent_tree(system, root, commit.parents.get(against - 1), io) {
        Ok(tree) => tree,
        Err(status) => return status,
    };
    let head_tree = match super::require_tree(system, root, &head, io) {
        Ok(tree) => tree,
        Err(status) => return status,
    };
    let (base, theirs) = if revert {
        (&commit_tree, &parent_tree)
    } else {
        (&parent_tree, &commit_tree)
    };
    // Committing the replay would fold anything already staged into it, which is why Git wants
    // a settled index first. `-n` leaves the commit to the user, so it can go ahead.
    let snapshot = match switch::snapshot(system, root, io) {
        Ok(snapshot) => snapshot,
        Err(status) => return status,
    };
    let dirty = !no_commit && snapshot.index != head_tree;
    if dirty || !switch::blocking_changes(&snapshot, theirs).is_empty() {
        io.print_err(&format!(
                "error: your local changes would be overwritten by {name}.\nhint: commit your changes or stash them to proceed.\nfatal: {name} failed\n"
            ));
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
    let combined = conflict::combine(system, root, base, &head_tree, theirs, "HEAD", &label);
    let combined = match combined {
        Ok(combined) => combined,
        Err(failed) => return super::cannot_write(io, &failed.path, &failed.error),
    };
    let applied = combined.tree;
    if let Err(error) = repo::update_work_tree(system, root, &head_tree, &applied) {
        io.print_err(&format!("git {name}: {error}\n"));
        return 1;
    }
    // Only the paths the replay touched move in the index; anything else staged is left alone.
    let mut index = snapshot.index.clone();
    for path in head_tree.keys().chain(applied.keys()) {
        if head_tree.get(path) == applied.get(path) {
            continue;
        }
        match applied.get(path) {
            Some(entry) => index.insert(path.clone(), entry.clone()),
            None => index.remove(path),
        };
    }
    if let Err(error) = repo::store_index(system, root, &index) {
        return super::cannot_write(io, "the index", &error);
    }
    let kind = if revert {
        conflict::REVERT_HEAD
    } else {
        conflict::CHERRY_PICK_HEAD
    };
    if !combined.stages.is_empty() {
        // The escape hatches are named because an agent that reads only this line still needs
        // to know it can skip the commit or give the whole thing up.
        let advice = format!(
            "error: could not apply {label}\n\
             hint: After resolving the conflicts, mark them with\n\
             hint: \"git add/rm <pathspec>\", then run \"git {name} --continue\".\n\
             hint: You can instead skip this commit with \"git {name} --skip\".\n\
             hint: To abort and get back to the state before \"git {name}\",\n\
             hint: run \"git {name} --abort\"."
        );
        return pause_for_conflicts(
            system,
            root,
            &Paused {
                kind,
                commit: &id,
                message: &message,
                theirs_label: &label,
                advice: &advice,
            },
            &combined.stages,
            io,
        );
    }
    if no_commit {
        // `-n` leaves the change staged for the user to commit; Git records nothing in progress.
        return 0;
    }
    if applied == head_tree {
        io.print_err(&format!("The previous cherry-pick is now empty, possibly due to conflict resolution.\nfatal: {name} failed\n"));
        return 1;
    }
    // A cherry-pick keeps the original author; a revert is the work of whoever ran it.
    let (author_name, author_email) = if revert {
        repo::author_identity(system, root, globals)
    } else {
        (commit.author_name.clone(), commit.author_email.clone())
    };
    let replayed = Commit {
        parents: vec![head.clone()],
        author_name,
        author_email,
        timestamp: repo::now_seconds(system),
        message,
    };
    let new_id = match repo::store_commit(system, root, &replayed, &applied) {
        Ok(id) => id,
        Err(error) => return super::cannot_write(io, "the commit", &error),
    };
    let action = format!("{name}: {}", replayed.subject());
    if let Err(error) = repo::update_head(system, root, &new_id, &action) {
        return super::cannot_write(io, "HEAD", &error);
    }
    let branch = repo::current_branch(system, root).unwrap_or_else(|| "detached HEAD".to_string());
    io.print(&format!(
        "[{branch} {}] {}\n",
        repo::short(&new_id),
        replayed.subject()
    ));
    commit::emit_commit_summary(system, root, &head_tree, &applied, io);
    0
}

/// Record an unfinished merge, cherry-pick, or revert and report the paths left to the user.
/// The operation that stopped, as its messages and its record of what is in progress need it.
struct Paused<'a> {
    /// `MERGE_HEAD`, `CHERRY_PICK_HEAD` or `REVERT_HEAD`: the file that records the operation.
    kind: &'a str,
    /// The commit being brought in.
    commit: &'a str,
    /// The message the finished commit will carry.
    message: &'a str,
    /// What the incoming side is called in a conflict marker.
    theirs_label: &'a str,
    /// The closing line, which tells the user how to carry on.
    advice: &'a str,
}

fn pause_for_conflicts(
    system: &mut dyn System,
    root: &str,
    paused: &Paused<'_>,
    stages: &Stages,
    io: &mut Io,
) -> i32 {
    let Paused {
        kind,
        commit,
        message,
        theirs_label,
        advice,
    } = *paused;
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
        io.print_err(&line);
    }
    if !conflict::begin(system, root, kind, commit, message)
        || !conflict::store_stages(system, root, stages)
    {
        io.print_err("fatal: unable to record the conflicted state\n");
        return 128;
    }
    io.print_err(&format!("{advice}\n"));
    1
}

/// The operation waiting to be finished, if any.
pub(crate) fn pending_operation(
    system: &mut dyn System,
    root: &str,
) -> Option<(&'static str, String)> {
    for kind in [
        conflict::MERGE_HEAD,
        conflict::CHERRY_PICK_HEAD,
        conflict::REVERT_HEAD,
    ] {
        if let Some(commit) = conflict::in_progress(system, root, kind) {
            return Some((kind, commit));
        }
    }
    None
}

/// Throw away an unfinished merge, cherry-pick, or revert.
fn abort_pending(system: &mut dyn System, root: &str, name: &str, io: &mut Io) -> i32 {
    if pending_operation(system, root).is_none() {
        io.print_err(&format!(
            "fatal: There is no {name} in progress ({name} --abort).\n"
        ));
        return 128;
    }
    let head = repo::head_tree(system, root);
    // Only paths the merge could have touched are restored; untracked files are left alone.
    let mut previous = match super::require_index(system, root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    for (path, hash) in &head {
        previous.entry(path.clone()).or_insert_with(|| hash.clone());
    }
    if let Err(error) = repo::replace_work_tree(system, root, &previous, &head) {
        io.print_err(&format!("git {name}: {error}\n"));
        return 1;
    }
    if let Err(error) = repo::store_index(system, root, &head) {
        return super::cannot_write(io, "the index", &error);
    }
    conflict::clear(system, root);
    0
}

/// Finish an operation whose conflicts the user has resolved.
fn continue_pending(
    system: &mut dyn System,
    root: &str,
    globals: &Globals,
    name: &str,
    io: &mut Io,
) -> i32 {
    if pending_operation(system, root).is_none() {
        io.print_err(&format!(
            "fatal: There is no {name} in progress ({name} --continue).\n"
        ));
        return 128;
    }
    if !conflict::load_stages(system, root).is_empty() {
        io.print_err("error: Committing is not possible because you have unmerged files.\n");
        return 1;
    }
    commit::git_commit(system, globals, &["--no-edit".to_string()], io)
}
