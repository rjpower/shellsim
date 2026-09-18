//! Index and working-tree porcelain: `status`, `add`, `rm`, `mv`, `restore`, `reset`, `clean`,
//! and `ls-files`.
//!
//! These commands share one shape: read the HEAD tree, the index, and a hashed snapshot of the
//! working tree, compare them, then write back an index and possibly replace files. Snapshot
//! memory is reserved up front and released on every exit path.

use std::collections::BTreeSet;

use crate::commands::{CommandContext, Io};
use crate::vfs::resolve_against;

use super::compare;
use super::conflict;
use super::ignore;
use super::repo::{self, Tree};
use super::{cannot_write, fatal, repo_error, require_index, usage, Arg, Flags};

/// The state of one path relative to HEAD, the index, and the working tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Change {
    Added,
    Modified,
    Deleted,
    Renamed,
}

impl Change {
    fn porcelain(self) -> char {
        match self {
            Change::Added => 'A',
            Change::Modified => 'M',
            Change::Deleted => 'D',
            Change::Renamed => 'R',
        }
    }

    fn label(self) -> &'static str {
        match self {
            Change::Added => "new file",
            Change::Modified => "modified",
            Change::Deleted => "deleted",
            Change::Renamed => "renamed",
        }
    }
}

/// One path's staged and unstaged state, plus the source path when it was renamed.
struct Entry {
    path: String,
    origin: Option<String>,
    staged: Option<Change>,
    unstaged: Option<Change>,
}

impl Entry {
    /// The path as status prints it, including the `old -> new` form for a rename.
    fn display(&self) -> String {
        match &self.origin {
            Some(origin) => format!("{origin} -> {}", self.path),
            None => self.path.clone(),
        }
    }
}

/// Compare HEAD, the index, and the working tree, pairing exact renames.
fn classify(head: &Tree, index: &Tree, work: &Tree) -> Vec<Entry> {
    let mut paths = BTreeSet::new();
    paths.extend(head.keys().cloned());
    paths.extend(index.keys().cloned());
    paths.extend(work.keys().cloned());
    let mut entries = Vec::new();
    for path in paths {
        let (head_value, index_value, work_value) =
            (head.get(&path), index.get(&path), work.get(&path));
        let staged = match (head_value, index_value) {
            (None, Some(_)) => Some(Change::Added),
            (Some(old), Some(new)) if old != new => Some(Change::Modified),
            (Some(_), None) => Some(Change::Deleted),
            _ => None,
        };
        let unstaged = match (index_value, work_value) {
            // A path the index does not track is untracked, reported separately.
            (None, _) => None,
            (Some(_), None) => Some(Change::Deleted),
            (Some(old), Some(new)) if old != new => Some(Change::Modified),
            _ => None,
        };
        if staged.is_none() && unstaged.is_none() {
            continue;
        }
        entries.push(Entry {
            path,
            origin: None,
            staged,
            unstaged,
        });
    }
    pair_renames(&mut entries, head, index);
    entries
}

/// Fold a staged delete and a staged add of identical content into one rename entry.
fn pair_renames(entries: &mut Vec<Entry>, head: &Tree, index: &Tree) {
    let deleted: Vec<(usize, String)> = entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.staged == Some(Change::Deleted))
        .map(|(position, entry)| (position, entry.path.clone()))
        .collect();
    let mut removed = BTreeSet::new();
    for (position, source) in deleted {
        let Some(hash) = head.get(&source) else {
            continue;
        };
        let target = entries.iter().position(|entry| {
            entry.staged == Some(Change::Added)
                && entry.origin.is_none()
                && index.get(&entry.path) == Some(hash)
        });
        let Some(target) = target else {
            continue;
        };
        entries[target].staged = Some(Change::Renamed);
        entries[target].origin = Some(source);
        removed.insert(position);
    }
    let mut position = 0;
    entries.retain(|_| {
        let keep = !removed.contains(&position);
        position += 1;
        keep
    });
}

/// Which untracked files `git status` reports.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Untracked {
    Normal,
    All,
    No,
}

/// Collapse untracked paths into directories that contain nothing tracked, as Git does.
///
/// A directory with no index entry below it is reported as `dir/` instead of listing every file.
pub(crate) fn collapse_untracked(index: &Tree, untracked: &[String]) -> Vec<String> {
    let mut out = BTreeSet::new();
    for path in untracked {
        out.insert(collapsed_form(index, path));
    }
    out.into_iter().collect()
}

fn collapsed_form(index: &Tree, path: &str) -> String {
    let mut prefix = String::new();
    for component in path.split('/').take(path.split('/').count() - 1) {
        prefix.push_str(component);
        prefix.push('/');
        if !index.keys().any(|tracked| tracked.starts_with(&prefix)) {
            return prefix;
        }
    }
    path.to_string()
}

pub(crate) fn git_status(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut short = false;
    let mut version_two = false;
    let mut branch_header = false;
    let mut untracked = Untracked::Normal;
    let mut show_ignored = false;
    let mut nul = false;
    let mut paths: Vec<String> = Vec::new();
    for argument in Flags::new(args).clustered("sbz").valued("u") {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                paths.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "-s" | "--short" => short = true,
            // Git takes these values only when written against the option, so what follows a
            // bare `--porcelain` is a pathspec rather than its value.
            "--porcelain" => match attached.as_deref() {
                None | Some("v1") => short = true,
                Some("v2") => version_two = true,
                Some(other) => return usage(io, &format!("unsupported porcelain format: {other}")),
            },
            "-b" | "--branch" => branch_header = true,
            "-z" => {
                nul = true;
                short = true;
            }
            "-u" | "--untracked-files" => {
                untracked = match attached.as_deref() {
                    None | Some("normal") => Untracked::Normal,
                    Some("no") => Untracked::No,
                    Some("all") => Untracked::All,
                    Some(other) => {
                        return usage(io, &format!("unsupported untracked-files mode: {other}"))
                    }
                }
            }
            "--ignored" => show_ignored = true,
            "--long" => short = false,
            "--no-column" | "--no-renames" | "--no-color" | "--ahead-behind" => {}
            _ => return usage(io, &format!("unsupported status option: {name}")),
        }
    }
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let cwd = ctx.cwd.clone();
    let paths: Vec<String> = paths
        .iter()
        .map(|value| super::pathspec(&cwd, &root, value))
        .collect();
    let super::switch::Snapshot { head, index, work } =
        match super::switch::snapshot(ctx, &root, io) {
            Ok(snapshot) => snapshot,
            Err(status) => return status,
        };
    let rules = ignore::load(ctx, &root);
    // A path with recorded conflict stages is unmerged and is reported on its own.
    let unmerged = conflict::load_stages(ctx, &root);
    let entries: Vec<Entry> = classify(&head, &index, &work)
        .into_iter()
        .filter(|entry| !unmerged.contains_key(&entry.path))
        .filter(|entry| {
            compare::selected(&paths, &entry.path)
                || entry
                    .origin
                    .as_deref()
                    .is_some_and(|origin| compare::selected(&paths, origin))
        })
        .collect();
    let mut untracked_paths = Vec::new();
    let mut ignored_paths = Vec::new();
    if untracked != Untracked::No || show_ignored {
        for path in work.keys().filter(|path| !index.contains_key(*path)) {
            if !compare::selected(&paths, path) {
                continue;
            }
            if rules.is_ignored(path) {
                ignored_paths.push(path.clone());
            } else {
                untracked_paths.push(path.clone());
            }
        }
    }
    if untracked == Untracked::No {
        untracked_paths.clear();
    }
    if untracked != Untracked::All {
        untracked_paths = collapse_untracked(&index, &untracked_paths);
        ignored_paths = collapse_untracked(&index, &ignored_paths);
    }
    if !show_ignored {
        ignored_paths.clear();
    }
    // Git names paths relative to the working directory, reaching upwards with `../` when needed.
    let prefix = repo::relative_path(&root, &cwd).map_or_else(String::new, |p| format!("{p}/"));
    // The v2 format reports object ids, so the repository-relative names are kept alongside.
    let tracked: Vec<String> = entries.iter().map(|entry| entry.path.clone()).collect();
    let mut entries = entries;
    for entry in &mut entries {
        entry.path = displayed_path(&prefix, &entry.path);
        entry.origin = entry
            .origin
            .as_deref()
            .map(|origin| displayed_path(&prefix, origin));
    }
    for path in untracked_paths.iter_mut().chain(ignored_paths.iter_mut()) {
        *path = displayed_path(&prefix, path);
    }
    let unmerged: Vec<(String, conflict::Unmerged)> = unmerged
        .into_iter()
        .filter(|(path, _)| compare::selected(&paths, path))
        .map(|(path, entry)| (displayed_path(&prefix, &path), entry))
        .collect();
    let report = Report {
        branch: repo::current_branch(ctx, &root),
        head: repo::head_commit(ctx, &root),
        entries,
        tracked,
        untracked: untracked_paths,
        ignored: ignored_paths,
        unmerged,
        head_tree: head,
        index,
        work,
        hide_untracked: untracked == Untracked::No,
        branch_header,
        nul,
    };
    if version_two {
        emit_porcelain_v2(&report, io);
        return 0;
    }
    if short {
        emit_short_status(&report, io);
        return 0;
    }
    let pending = unfinished_operation(ctx, &root, !report.unmerged.is_empty());
    emit_long_status(&report, pending.as_deref(), io);
    0
}

/// Everything the three status formats print, worked out once.
///
/// The formats differ in layout rather than in what they report, so they read the same struct.
/// `head_tree`, `index` and `work` are the three trees a `v2` record reports object ids from, and
/// `tracked` holds each entry's repository-relative name, because the entry's own path has been
/// made relative to the working directory for display.
struct Report {
    branch: Option<String>,
    head: Option<String>,
    entries: Vec<Entry>,
    tracked: Vec<String>,
    untracked: Vec<String>,
    ignored: Vec<String>,
    unmerged: Vec<(String, conflict::Unmerged)>,
    head_tree: Tree,
    index: Tree,
    work: Tree,
    /// Set by `-u no`, which prints a line saying the list was left out.
    hide_untracked: bool,
    branch_header: bool,
    /// Set by `-z`, which separates records with NUL rather than a newline.
    nul: bool,
}

/// What Git prints above the file lists while a merge, replay or rebase is unfinished.
///
/// Naming the operation matters more than the wording: an agent that reads "you have unmerged
/// paths" mid-cherry-pick reaches for `git commit`, which refuses.
fn unfinished_operation(
    ctx: &mut CommandContext<'_>,
    root: &str,
    conflicted: bool,
) -> Option<String> {
    let next = |command: &str| {
        if conflicted {
            format!("  (fix conflicts and run \"git {command} --continue\")\n")
        } else {
            format!("  (all conflicts fixed: run \"git {command} --continue\")\n")
        }
    };
    if let Some((branch, onto)) = super::rebase::replaying(ctx, root) {
        return Some(format!(
            "You are currently rebasing branch '{branch}' on '{}'.\n{}  (use \"git rebase --skip\" to skip this patch)\n  (use \"git rebase --abort\" to check out the original branch)\n\n",
            repo::short(&onto),
            next("rebase"),
        ));
    }
    for (kind, command, cancel) in [
        (
            conflict::CHERRY_PICK_HEAD,
            "cherry-pick",
            "to cancel the cherry-pick operation",
        ),
        (
            conflict::REVERT_HEAD,
            "revert",
            "to cancel the revert operation",
        ),
    ] {
        if let Some(commit) = conflict::in_progress(ctx, root, kind) {
            let doing = if command == "revert" {
                "reverting"
            } else {
                "cherry-picking"
            };
            return Some(format!(
                "You are currently {doing} commit {}.\n{}  (use \"git {command} --skip\" to skip this patch)\n  (use \"git {command} --abort\" {cancel})\n\n",
                repo::short(&commit),
                next(command),
            ));
        }
    }
    conflicted.then(|| {
        "You have unmerged paths.\n  (fix conflicts and run \"git commit\")\n  (use \"git merge --abort\" to abort the merge)\n\n"
            .to_string()
    })
}

/// Render a repository-relative path relative to the working directory, as Git reports it.
fn displayed_path(prefix: &str, path: &str) -> String {
    if prefix.is_empty() {
        return path.to_string();
    }
    match path.strip_prefix(prefix) {
        Some(rest) => rest.to_string(),
        None => format!("{}{path}", "../".repeat(prefix.matches('/').count())),
    }
}

/// Report status in Git's `--porcelain=v2` format.
///
/// Every tracked entry is a `1` record; this subset never records a submodule or a file mode other
/// than `100644`, so those columns are constant.
fn emit_porcelain_v2(report: &Report, io: &mut Io) {
    const MISSING: &str = "0000000000000000000000000000000000000000";
    let Report {
        entries,
        tracked,
        untracked,
        ignored,
        unmerged,
        head_tree: head,
        index,
        work,
        ..
    } = report;
    if report.branch_header {
        let commit = report
            .head
            .clone()
            .unwrap_or_else(|| "(initial)".to_string());
        let branch = report
            .branch
            .clone()
            .unwrap_or_else(|| "(detached)".to_string());
        io.print(&format!("# branch.oid {commit}\n# branch.head {branch}\n"));
    }
    let mode = |recorded: Option<&repo::Entry>| recorded.map_or("000000", repo::Entry::mode);
    for (entry, path) in entries.iter().zip(tracked) {
        let x = entry.staged.map_or('.', Change::porcelain);
        let y = entry.unstaged.map_or('.', Change::porcelain);
        let staged = index.get(path);
        fn hash(recorded: Option<&repo::Entry>) -> &str {
            recorded.map_or(MISSING, |recorded| recorded.hash.as_str())
        }
        io.print(&format!(
            "1 {x}{y} N... {} {} {} {} {} {}\n",
            mode(head.get(path)),
            mode(staged),
            // A path the index no longer tracks has no working-tree mode to report.
            if staged.is_some() {
                mode(work.get(path))
            } else {
                "000000"
            },
            hash(head.get(path)),
            hash(staged),
            entry.display(),
        ));
    }
    for (path, entry) in unmerged {
        // Every conflict this subset produces comes from a plain two-parent merge.
        let stage = |hash: &Option<String>| match hash {
            Some(_) => "100644",
            None => "000000",
        };
        io.print(&format!(
            "u {} N... {} {} {} 100644 {} {} {} {path}\n",
            entry.porcelain(),
            stage(&entry.base),
            stage(&entry.ours),
            stage(&entry.theirs),
            entry.base.as_deref().unwrap_or(MISSING),
            entry.ours.as_deref().unwrap_or(MISSING),
            entry.theirs.as_deref().unwrap_or(MISSING),
        ));
    }
    for path in untracked {
        io.print(&format!("? {path}\n"));
    }
    for path in ignored {
        io.print(&format!("! {path}\n"));
    }
}

fn emit_short_status(report: &Report, io: &mut Io) {
    let Report {
        entries,
        untracked,
        ignored,
        unmerged,
        ..
    } = report;
    let terminator = if report.nul { 0 } else { b'\n' };
    if report.branch_header {
        let label = match (report.branch.as_deref(), report.head.as_deref()) {
            (Some(branch), None) => format!("No commits yet on {branch}"),
            (Some(branch), Some(_)) => branch.to_string(),
            (None, _) => "HEAD (no branch)".to_string(),
        };
        io.print(&format!("## {label}"));
        io.out.push(terminator);
    }
    // Tracked paths are listed in path order whether or not they are unmerged, as Git lists them.
    let mut tracked: Vec<(&str, String)> = unmerged
        .iter()
        .map(|(path, entry)| (path.as_str(), format!("{} {path}", entry.porcelain())))
        .collect();
    tracked.extend(entries.iter().map(|entry| {
        let x = entry.staged.map_or(' ', Change::porcelain);
        let y = entry.unstaged.map_or(' ', Change::porcelain);
        (entry.path.as_str(), format!("{x}{y} {}", entry.display()))
    }));
    tracked.sort_by(|left, right| left.0.cmp(right.0));
    for (_, line) in tracked {
        io.print(&line);
        io.out.push(terminator);
    }
    for path in untracked {
        io.print(&format!("?? {path}"));
        io.out.push(terminator);
    }
    for path in ignored {
        io.print(&format!("!! {path}"));
        io.out.push(terminator);
    }
}

fn emit_long_status(report: &Report, pending: Option<&str>, io: &mut Io) {
    let Report {
        entries,
        untracked,
        ignored,
        unmerged,
        hide_untracked,
        ..
    } = report;
    let head = report.head.as_deref();
    match report.branch.as_deref() {
        Some(branch) => io
            .out
            .extend_from_slice(format!("On branch {branch}\n").as_bytes()),
        None => {
            let id = head.map_or("unknown".to_string(), |id| repo::short(id).to_string());
            io.print(&format!("HEAD detached at {id}\n"));
        }
    }
    if head.is_none() {
        io.print("\nNo commits yet\n\n");
    }
    if let Some(pending) = pending {
        io.print(pending);
    }
    let staged: Vec<&Entry> = entries
        .iter()
        .filter(|entry| entry.staged.is_some())
        .collect();
    let unstaged: Vec<&Entry> = entries
        .iter()
        .filter(|entry| entry.unstaged.is_some())
        .collect();
    if !staged.is_empty() {
        let unstage_hint = if head.is_some() {
            "  (use \"git restore --staged <file>...\" to unstage)\n"
        } else {
            "  (use \"git rm --cached <file>...\" to unstage)\n"
        };
        io.print("Changes to be committed:\n");
        io.print(unstage_hint);
        for entry in &staged {
            emit_named_change(
                io,
                entry.staged.unwrap_or(Change::Modified),
                &entry.display(),
            );
        }
        io.out.push(b'\n');
    }
    if !unmerged.is_empty() {
        io.print("Unmerged paths:\n  (use \"git restore --staged <file>...\" to unstage)\n  (use \"git add <file>...\" to mark resolution)\n");
        for (path, entry) in unmerged {
            io.print(&format!(
                "\t{:<14}   {path}\n",
                format!("{}:", entry.label())
            ));
        }
        io.out.push(b'\n');
    }
    if !unstaged.is_empty() {
        io.print("Changes not staged for commit:\n  (use \"git add <file>...\" to update what will be committed)\n  (use \"git restore <file>...\" to discard changes in working directory)\n");
        for entry in &unstaged {
            emit_named_change(
                io,
                entry.unstaged.unwrap_or(Change::Modified),
                &entry.display(),
            );
        }
        io.out.push(b'\n');
    }
    if !untracked.is_empty() {
        io.print("Untracked files:\n  (use \"git add <file>...\" to include in what will be committed)\n");
        for path in untracked {
            io.print(&format!("\t{path}\n"));
        }
        io.out.push(b'\n');
    }
    if !ignored.is_empty() {
        io.print("Ignored files:\n  (use \"git add -f <file>...\" to include in what will be committed)\n");
        for path in ignored {
            io.print(&format!("\t{path}\n"));
        }
        io.out.push(b'\n');
    }
    if *hide_untracked {
        io.print("Untracked files not listed (use -u option to show untracked files)\n");
    }
    if !unmerged.is_empty() {
        io.print("no changes added to commit (use \"git add\" and/or \"git commit -a\")\n");
    } else if entries.is_empty() && untracked.is_empty() {
        io.print("nothing to commit, working tree clean\n");
    } else if staged.is_empty() && unstaged.is_empty() && !untracked.is_empty() {
        io.print(
            "nothing added to commit but untracked files present (use \"git add\" to track)\n",
        );
    } else if staged.is_empty() {
        io.print("no changes added to commit (use \"git add\" and/or \"git commit -a\")\n");
    }
}

fn emit_named_change(io: &mut Io, change: Change, path: &str) {
    io.print(&format!(
        "\t{:<11} {path}\n",
        format!("{}:", change.label())
    ));
}

/// Expand pathspecs into the set of repository-relative paths they name.
///
/// A path that is tracked but missing from the working tree still matches, so staging a deletion
/// works; a path known to neither the index nor the working tree is an error, as in Git.
fn selected_paths(
    cwd: &str,
    root: &str,
    args: &[String],
    index: &Tree,
    work: &Tree,
) -> Result<BTreeSet<String>, String> {
    let mut selected = BTreeSet::new();
    for argument in args {
        if argument.starts_with('-') {
            return Err(format!("unsupported pathspec: {argument}"));
        }
        let absolute = resolve_against(cwd, argument);
        if !repo::within(root, &absolute) {
            return Err(format!("pathspec '{argument}' is outside the repository"));
        }
        let relative = repo::relative_path(root, &absolute).unwrap_or_default();
        if relative.is_empty() {
            selected.extend(index.keys().cloned());
            selected.extend(work.keys().cloned());
            continue;
        }
        let mut found = false;
        for key in index.keys().chain(work.keys()) {
            if super::ignore::matches_pathspec(&relative, key) {
                selected.insert(key.clone());
                found = true;
            }
        }
        if !found {
            return Err(format!("pathspec '{argument}' did not match any files"));
        }
    }
    Ok(selected)
}

pub(crate) fn git_add(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut all = false;
    let mut update_only = false;
    let mut dry_run = false;
    let mut verbose = false;
    let mut force = false;
    let mut operands = Vec::new();
    for argument in Flags::new(args).clustered("Auvnf") {
        let name = match argument {
            Arg::Operand(value) => {
                operands.push(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "-A" | "--all" | "--no-ignore-removal" => all = true,
            "-u" | "--update" => update_only = true,
            "-n" | "--dry-run" => dry_run = true,
            "-v" | "--verbose" => verbose = true,
            "-f" | "--force" => force = true,
            _ => return usage(io, &format!("unsupported add option: {name}")),
        }
    }
    if !all && !update_only && operands.is_empty() {
        // Git treats this as a no-op with a hint rather than an error.
        io.print_err(
            "Nothing specified, nothing added.\nhint: Maybe you wanted to say 'git add .'?\n",
        );
        return 0;
    }
    if operands.len() > 256 {
        return fatal(io, "too many pathspecs");
    }
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut index = match require_index(ctx, &root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    let work = match repo::collect_working_tree(ctx, &root) {
        Ok(work) => work,
        Err(status) => return status,
    };
    let cwd = ctx.cwd.clone();
    let mut selected = if operands.is_empty() {
        let mut everything: BTreeSet<String> = index.keys().cloned().collect();
        everything.extend(work.keys().cloned());
        everything
    } else {
        let units = (operands.len() as u64)
            .saturating_mul((index.len() as u64).saturating_add(work.len() as u64));
        if !ctx.charge_cpu(units) {
            return repo::resource_error(ctx);
        }
        match selected_paths(&cwd, &root, &operands, &index, &work) {
            Ok(paths) => paths,
            Err(message) => {
                io.print_err(&format!("fatal: {message}\n"));
                return 128;
            }
        }
    };
    if update_only {
        selected.retain(|path| index.contains_key(path));
    }
    if !force {
        let rules = ignore::load(ctx, &root);
        // Naming an ignored file outright is an error; sweeping one up by directory or glob is not.
        let named: Vec<String> = operands
            .iter()
            .map(|operand| super::pathspec(&cwd, &root, operand))
            .filter(|path| {
                work.contains_key(path) && !index.contains_key(path) && rules.is_ignored(path)
            })
            .collect();
        if !named.is_empty() {
            io.print_err("The following paths are ignored by one of your .gitignore files:\n");
            for path in &named {
                io.print_err(&format!("{path}\n"));
            }
            io.print_err("hint: Use -f if you really want to add them.\n");
            io.print_err("fatal: no files added\n");
            return 1;
        }
        selected.retain(|path| index.contains_key(path) || !rules.is_ignored(path));
    }
    if selected.len() > 100_000 {
        return repo::resource_error(ctx);
    }
    let reserved = selected.len() as u64 * 64;
    if !ctx.reserve_memory(reserved) {
        return repo::resource_error(ctx);
    }
    let staging = match (dry_run, verbose) {
        (true, _) => Staging::DryRun,
        (false, true) => Staging::Verbose,
        (false, false) => Staging::Silent,
    };
    let status = stage_paths(ctx, &root, &mut index, &work, &selected, staging, io);
    ctx.resources.release_memory(reserved);
    // Staging a conflicted path is how the user says the conflict is settled.
    if status == 0 && !dry_run {
        conflict::resolve(ctx, &root, &selected);
    }
    status
}

/// What `git add` says about each path, and whether it stages it.
#[derive(Clone, Copy, PartialEq)]
enum Staging {
    /// The ordinary case: stage the path and say nothing.
    Silent,
    /// `-v`: stage the path and name it.
    Verbose,
    /// `-n`: name the path that would be staged, and stage nothing.
    DryRun,
}

fn stage_paths(
    ctx: &mut CommandContext<'_>,
    root: &str,
    index: &mut Tree,
    work: &Tree,
    selected: &BTreeSet<String>,
    staging: Staging,
    io: &mut Io,
) -> i32 {
    let dry_run = staging == Staging::DryRun;
    for path in selected {
        let action = if work.contains_key(path) {
            "add"
        } else {
            "remove"
        };
        if staging != Staging::Silent {
            io.print(&format!("{action} '{path}'\n"));
        }
        if dry_run {
            continue;
        }
        if let Some(recorded) = work.get(path) {
            let Some(data) = repo::read_work_file(ctx, root, path) else {
                io.print_err(&format!("git add: unable to read '{path}'\n"));
                return 1;
            };
            if repo::blob_hash(&data) != recorded.hash {
                io.print_err(&format!(
                    "git add: '{path}' changed while building the index\n"
                ));
                return 1;
            }
            if let Err(error) = repo::write_blob(ctx, root, &data) {
                io.print_err(&format!("git add: {error}\n"));
                return 1;
            }
            index.insert(path.clone(), recorded.clone());
        } else {
            index.remove(path);
        }
    }
    if dry_run {
        return 0;
    }
    if let Err(error) = repo::store_index(ctx, root, index) {
        io.print_err(&format!("git add: {error}\n"));
        return 1;
    }
    0
}

/// Resolve one operand to an absolute path and a repository-relative path.
fn repo_operand(cwd: &str, root: &str, operand: &str) -> Result<(String, String), String> {
    let absolute = resolve_against(cwd, operand);
    if !repo::within(root, &absolute) || repo::is_git_path(root, &absolute) {
        return Err(format!("pathspec '{operand}' is outside the working tree"));
    }
    let relative = repo::relative_path(root, &absolute)
        .ok_or_else(|| format!("pathspec '{operand}' does not name a file"))?;
    Ok((absolute, relative))
}

pub(crate) fn git_rm(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut cached = false;
    let mut force = false;
    let mut recursive = false;
    let mut ignore_unmatch = false;
    let mut quiet = false;
    let mut operands = Vec::new();
    for argument in Flags::new(args).clustered("rfq") {
        let name = match argument {
            Arg::Operand(value) => {
                operands.push(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "--cached" => cached = true,
            "-f" | "--force" => force = true,
            "-r" => recursive = true,
            "-q" | "--quiet" => quiet = true,
            "--ignore-unmatch" => ignore_unmatch = true,
            _ => return usage(io, &format!("unsupported rm option: {name}")),
        }
    }
    if operands.is_empty() || operands.len() > 256 {
        return usage(io, "git rm requires between 1 and 256 file paths");
    }
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut index = match require_index(ctx, &root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    let head = repo::head_tree(ctx, &root);
    // Removing an unmerged path is how a modify/delete conflict is settled, so it needs no force.
    let unmerged = conflict::load_stages(ctx, &root);
    let mut selected: Vec<(String, String)> = Vec::new();
    for operand in operands {
        let (absolute, relative) = match repo_operand(&ctx.cwd, &root, &operand) {
            Ok(paths) => paths,
            Err(message) => return fatal(io, &message),
        };
        let directory_prefix = format!("{relative}/");
        let tracked: Vec<String> = index
            .keys()
            .filter(|path| **path == relative || path.starts_with(&directory_prefix))
            .cloned()
            .collect();
        if tracked.is_empty() {
            if ignore_unmatch {
                continue;
            }
            io.print_err(&format!(
                "fatal: pathspec '{operand}' did not match any files\n"
            ));
            return 128;
        }
        if tracked.len() > 1 || tracked[0] != relative {
            if !recursive {
                io.print_err(&format!(
                    "fatal: not removing '{operand}' recursively without -r\n"
                ));
                return 128;
            }
        } else if !cached && !ctx.vfs.is_file("/", &absolute) {
            io.print_err(&format!("fatal: pathspec '{operand}' is missing\n"));
            return 128;
        }
        for path in tracked {
            let absolute = repo::path_join(&root, &path);
            if !force && !unmerged.contains_key(&path) {
                let index_hash = index
                    .get(&path)
                    .map(|entry| entry.hash.clone())
                    .unwrap_or_default();
                let work_hash = match repo::metered_file_hash(ctx, &absolute) {
                    Ok(hash) => hash,
                    Err(status) => return status,
                };
                let matches_work = work_hash.as_deref() == Some(index_hash.as_str());
                let matches_head =
                    head.get(&path).map(|entry| entry.hash.as_str()) == Some(index_hash.as_str());
                let safe = if cached {
                    matches_work || matches_head
                } else {
                    matches_work && matches_head
                };
                if !safe {
                    io.print_err(&format!(
                        "error: the following file has local modifications:\n    {path}\n\
                             (use --cached to keep the file, or -f to force removal)\n"
                    ));
                    return 1;
                }
            }
            selected.push((absolute, path));
        }
    }
    let reserved = ctx.vfs.disk_used().saturating_add(4 * 1024);
    if !ctx.reserve_memory(reserved) {
        return repo::resource_error(ctx);
    }
    let before = ctx.vfs.clone();
    let result = (|| -> Result<(), String> {
        for (absolute, relative) in &selected {
            if !cached && ctx.vfs.is_file("/", absolute) {
                ctx.vfs
                    .remove_file("/", absolute)
                    .map_err(|error| error.to_string())?;
                repo::prune_empty_parents(ctx, &root, absolute);
            }
            index.remove(relative);
        }
        repo::store_index(ctx, &root, &index).map_err(|error| error.to_string())
    })();
    ctx.resources.release_memory(reserved);
    if let Err(error) = result {
        ctx.vfs = before;
        io.print_err(&format!("git rm: {error}\n"));
        return 1;
    }
    conflict::resolve(ctx, &root, selected.iter().map(|(_, path)| path));
    if !quiet {
        for (_, relative) in &selected {
            io.print(&format!("rm '{relative}'\n"));
        }
    }
    0
}

pub(crate) fn git_mv(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut operands = Vec::new();
    let mut force = false;
    for argument in Flags::new(args).clustered("fkv") {
        let name = match argument {
            Arg::Operand(value) => {
                operands.push(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "-f" | "--force" => force = true,
            "-k" | "-v" | "--verbose" => {}
            _ => return usage(io, &format!("unsupported mv option: {name}")),
        }
    }
    let Some((destination, sources)) = operands.split_last() else {
        return usage(io, "usage: git mv SOURCE... DESTINATION");
    };
    if sources.is_empty() {
        return usage(io, "usage: git mv SOURCE... DESTINATION");
    }
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let (destination_absolute, _) = match repo_operand(&ctx.cwd, &root, destination) {
        Ok(paths) => paths,
        Err(message) => return fatal(io, &message),
    };
    let into_directory = ctx.vfs.is_dir("/", &destination_absolute);
    if sources.len() > 1 && !into_directory {
        return usage(
            io,
            "destination must be a directory when moving several sources",
        );
    }
    let mut index = match require_index(ctx, &root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    let mut moves: Vec<(String, String, String, String)> = Vec::new();
    for source in sources {
        let (source_absolute, source_relative) = match repo_operand(&ctx.cwd, &root, source) {
            Ok(paths) => paths,
            Err(message) => return fatal(io, &message),
        };
        // A directory source moves every tracked file below it.
        let tracked: Vec<String> = if ctx.vfs.is_dir("/", &source_absolute) {
            let prefix = format!("{source_relative}/");
            index
                .keys()
                .filter(|path| path.starts_with(&prefix))
                .cloned()
                .collect()
        } else {
            vec![source_relative.clone()]
        };
        if tracked.is_empty() || tracked.iter().any(|path| !index.contains_key(path)) {
            io.print_err(&format!("fatal: bad source, source={source}\n"));
            return 128;
        }
        for path in tracked {
            let suffix = path
                .strip_prefix(&format!("{source_relative}/"))
                .map(str::to_string);
            let target_absolute = match (&suffix, into_directory) {
                (Some(suffix), _) => format!(
                    "{destination_absolute}/{}",
                    if into_directory {
                        format!("{}/{suffix}", crate::vfs::basename(&source_absolute))
                    } else {
                        suffix.clone()
                    }
                ),
                (None, true) => format!(
                    "{destination_absolute}/{}",
                    crate::vfs::basename(&source_absolute)
                ),
                (None, false) => destination_absolute.clone(),
            };
            let Some(target_relative) = repo::relative_path(&root, &target_absolute) else {
                return fatal(io, "destination is outside the working tree");
            };
            if !force
                && (ctx.vfs.exists("/", &target_absolute) || index.contains_key(&target_relative))
            {
                io.print_err(&format!(
                    "fatal: destination exists, source={source}, destination={target_relative}\n"
                ));
                return 128;
            }
            moves.push((
                repo::path_join(&root, &path),
                path,
                target_absolute,
                target_relative,
            ));
        }
    }
    let reserved = ctx.vfs.disk_used().saturating_add(4 * 1024);
    if !ctx.reserve_memory(reserved) {
        return repo::resource_error(ctx);
    }
    let before = ctx.vfs.clone();
    let result = (|| -> Result<(), String> {
        for (source_absolute, source_relative, target_absolute, target_relative) in &moves {
            // Read the link itself rather than what it points at, so moving a symbolic link
            // stages its target text and not the contents of the file at the end of it.
            let data = repo::read_work_file(ctx, &root, source_relative)
                .ok_or_else(|| format!("cannot read '{source_relative}'"))?;
            if !ctx.charge_cpu(data.len() as u64) {
                return Err("cpu limit exceeded".to_string());
            }
            let hash = repo::write_blob(ctx, &root, &data).map_err(|error| error.to_string())?;
            // A move keeps the file's mode along with its content.
            let recorded = index.get(source_relative).cloned().unwrap_or_default();
            index.remove(source_relative);
            index.insert(target_relative.clone(), repo::Entry { hash, ..recorded });
            if ctx.vfs.exists("/", target_absolute) {
                ctx.vfs
                    .remove_file("/", target_absolute)
                    .map_err(|error| error.to_string())?;
            }
            if let Some(parent) = crate::vfs::parent_of(target_absolute) {
                ctx.vfs
                    .mkdir_all("/", &parent)
                    .map_err(|error| error.to_string())?;
            }
            ctx.sync_vfs_time();
            ctx.vfs
                .rename("/", source_absolute, target_absolute)
                .map_err(|error| error.to_string())?;
            repo::prune_empty_parents(ctx, &root, source_absolute);
        }
        repo::store_index(ctx, &root, &index).map_err(|error| error.to_string())
    })();
    ctx.resources.release_memory(reserved);
    if let Err(error) = result {
        ctx.vfs = before;
        io.print_err(&format!("git mv: {error}\n"));
        return 1;
    }
    0
}

pub(crate) fn git_restore(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut staged = false;
    let mut worktree = false;
    let mut source = None;
    let mut side = None;
    let mut paths = Vec::new();
    let mut flags = Flags::new(args).clustered("SWq").valued("s");
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                paths.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "--staged" | "-S" => staged = true,
            "--worktree" | "-W" => worktree = true,
            "--source" | "-s" => {
                let Some(value) = flags.value(attached) else {
                    return usage(io, "--source requires a revision");
                };
                source = Some(value);
            }
            "-q" | "--quiet" => {}
            "--ours" => side = Some(false),
            "--theirs" => side = Some(true),
            _ => return usage(io, &format!("unsupported restore option: {name}")),
        }
    }
    if let Some(theirs) = side {
        return super::switch::restore_side(ctx, theirs, &paths, io);
    }
    if paths.is_empty() {
        io.print_err("fatal: you must specify path(s) to restore\n");
        return 128;
    }
    if !staged {
        worktree = true;
    }
    restore_paths(ctx, &root, source.as_deref(), &paths, staged, worktree, io)
}

fn restore_paths(
    ctx: &mut CommandContext<'_>,
    root: &str,
    source: Option<&str>,
    paths: &[String],
    staged: bool,
    worktree: bool,
    io: &mut Io,
) -> i32 {
    let mut index_tree = match require_index(ctx, root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    let source_tree = match source {
        Some(revision) => {
            let Some(commit) = repo::resolve_revision(ctx, root, revision) else {
                return fatal(io, "unknown restore source");
            };
            match super::require_tree(ctx, root, &commit, io) {
                Ok(tree) => tree,
                Err(status) => return status,
            }
        }
        None if staged => repo::head_tree(ctx, root),
        None => index_tree.clone(),
    };
    if staged {
        let cwd = ctx.cwd.clone();
        let selected = match selected_paths(&cwd, root, paths, &index_tree, &source_tree) {
            Ok(selected) => selected,
            Err(message) => return fatal(io, &message),
        };
        for path in &selected {
            match source_tree.get(path) {
                Some(hash) => {
                    index_tree.insert(path.clone(), hash.clone());
                }
                None => {
                    index_tree.remove(path);
                }
            }
        }
        if let Err(error) = repo::store_index(ctx, root, &index_tree) {
            return cannot_write(io, "the index", &error);
        }
        // Putting a path back the way HEAD has it settles it: there is one side left, not three.
        conflict::resolve(ctx, root, &selected);
        if !worktree {
            return 0;
        }
    }
    let work = match repo::collect_working_tree(ctx, root) {
        Ok(work) => work,
        Err(status) => return status,
    };
    let cwd = ctx.cwd.clone();
    // When restoring the working tree after `--staged`, the index is the source of truth again.
    let source_tree = if staged { index_tree } else { source_tree };
    let selected = match selected_paths(&cwd, root, paths, &source_tree, &work) {
        Ok(selected) => selected,
        Err(message) => return fatal(io, &message),
    };
    let old: Tree = work
        .into_iter()
        .filter(|(path, _)| selected.contains(path))
        .collect();
    let new: Tree = source_tree
        .into_iter()
        .filter(|(path, _)| selected.contains(path))
        .collect();
    repo::replace_work_tree(ctx, root, &old, &new).map_or_else(
        |error| {
            io.print_err(&format!("git restore: {error}\n"));
            1
        },
        |()| 0,
    )
}

pub(crate) fn git_reset(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut mode = "--mixed".to_string();
    let mut revision = None;
    let mut paths = Vec::new();
    let mut flags = Flags::new(args).clustered("q");
    while let Some(argument) = flags.next() {
        let name = match argument {
            Arg::Operand(value) => {
                // Before `--`, a name that resolves is the revision to reset to; anything else
                // has to be a path, or Git calls the argument ambiguous.
                if revision.is_none()
                    && !flags.separated()
                    && repo::resolve_revision(ctx, &root, &value).is_some()
                {
                    revision = Some(value);
                    continue;
                }
                if !super::names_a_path(ctx, &root, &value) {
                    return super::ambiguous_argument(io, &value);
                }
                paths.push(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "--soft" | "--mixed" | "--hard" => mode = name,
            "-q" | "--quiet" => {}
            _ => return usage(io, &format!("unsupported reset option: {name}")),
        }
    }
    let revision = revision.unwrap_or_else(|| "HEAD".to_string());
    let commit = match repo::resolve_revision(ctx, &root, &revision) {
        Some(commit) => commit,
        // A repository with no commits has an empty HEAD tree; resetting to it is a no-op.
        None if revision == "HEAD" => {
            return i32::from(repo::store_index(ctx, &root, &Tree::new()).is_err())
        }
        None => return super::ambiguous_argument(io, &revision),
    };
    let tree = match super::require_tree(ctx, &root, &commit, io) {
        Ok(tree) => tree,
        Err(status) => return status,
    };
    if !paths.is_empty() {
        if mode != "--mixed" {
            return usage(io, "a pathspec cannot be combined with --soft or --hard");
        }
        // `git reset REVISION -- PATH` rewrites only those index entries.
        return restore_paths(ctx, &root, Some(&revision), &paths, true, false, io);
    }
    repo::record_orig_head(ctx, &root);
    // A whole-tree reset is how a merge, cherry-pick or revert is walked away from, so it drops
    // the recorded conflict sides along with the operation itself. Git clears them here too.
    conflict::clear(ctx, &root);
    if mode == "--hard" {
        // A hard reset removes paths known by either HEAD or the index while preserving untracked
        // files, matching the boundary agents rely on when discarding staged additions.
        let mut old = repo::head_tree(ctx, &root);
        match require_index(ctx, &root, io) {
            Ok(index) => old.extend(index),
            Err(status) => return status,
        }
        if let Err(error) = repo::replace_work_tree(ctx, &root, &old, &tree) {
            io.print_err(&format!("git reset: {error}\n"));
            return 1;
        }
    }
    if mode != "--soft" {
        if let Err(error) = repo::store_index(ctx, &root, &tree) {
            return cannot_write(io, "the index", &error);
        }
    }
    let target = revision.clone();
    let action = format!("reset: moving to {target}");
    if let Err(error) = repo::update_head(ctx, &root, &commit, &action) {
        return cannot_write(io, "HEAD", &error);
    }
    if mode == "--mixed" {
        emit_unstaged_after_reset(ctx, &root, &tree, io);
    }
    if mode == "--hard" {
        let subject = repo::load_commit(ctx, &root, &commit)
            .map(|commit| commit.subject().to_string())
            .unwrap_or_default();
        io.print(&format!(
            "HEAD is now at {} {subject}\n",
            repo::short(&commit)
        ));
    }
    0
}

/// List the paths that differ from the new index after a mixed reset, as Git does.
fn emit_unstaged_after_reset(ctx: &mut CommandContext<'_>, root: &str, tree: &Tree, io: &mut Io) {
    let Ok(work) = repo::collect_working_tree(ctx, root) else {
        return;
    };
    let changed: Vec<&String> = tree
        .keys()
        .filter(|path| work.get(*path) != tree.get(*path))
        .collect();
    if changed.is_empty() {
        return;
    }
    io.print("Unstaged changes after reset:\n");
    for path in changed {
        io.print(&format!("M\t{path}\n"));
    }
}

pub(crate) fn git_clean(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut force = false;
    let mut dry_run = false;
    let mut include_ignored = false;
    let mut directories = false;
    let mut paths = Vec::new();
    for argument in Flags::new(args).clustered("fdnxq") {
        let name = match argument {
            Arg::Operand(value) => {
                paths.push(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "-f" | "--force" => force = true,
            "-n" | "--dry-run" => dry_run = true,
            "-d" => directories = true,
            "-q" | "--quiet" => {}
            "-x" => include_ignored = true,
            _ => return usage(io, &format!("unsupported clean option: {name}")),
        }
    }
    if !force && !dry_run {
        io.print_err("fatal: clean.requireForce is true and -f not given: refusing to clean\n");
        return 128;
    }
    let index = match require_index(ctx, &root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    let work = match repo::collect_working_tree(ctx, &root) {
        Ok(work) => work,
        Err(status) => return status,
    };
    let rules = ignore::load(ctx, &root);
    let untracked: Vec<String> = work
        .keys()
        .filter(|path| !index.contains_key(*path))
        .filter(|path| include_ignored || !rules.is_ignored(path))
        .filter(|path| {
            paths.is_empty()
                || paths
                    .iter()
                    .any(|prefix| *path == prefix || path.starts_with(&format!("{prefix}/")))
        })
        .cloned()
        .collect();
    // Git names whole directories that contain nothing tracked, and only removes them with -d.
    let removable: Vec<String> = collapse_untracked(&index, &untracked)
        .into_iter()
        .filter(|entry| directories || !entry.ends_with('/'))
        .collect();
    for entry in &removable {
        if dry_run {
            io.print(&format!("Would remove {entry}\n"));
            continue;
        }
        let absolute = repo::path_join(&root, entry.trim_end_matches('/'));
        let removed = if entry.ends_with('/') {
            ctx.vfs.remove_all("/", &absolute).is_ok()
        } else {
            ctx.vfs.remove_file("/", &absolute).is_ok()
        };
        if !removed {
            io.print_err(&format!("warning: failed to remove {entry}\n"));
            continue;
        }
        io.print(&format!("Removing {entry}\n"));
    }
    0
}

pub(crate) fn git_ls_files(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut nul = false;
    let mut error_unmatch = false;
    let mut cached = false;
    let mut modified = false;
    let mut deleted = false;
    let mut others = false;
    let mut stage = false;
    let mut unmerged = false;
    let mut exclude_standard = false;
    let mut paths = Vec::new();
    for argument in Flags::new(args).clustered("cmdosuz") {
        let name = match argument {
            Arg::Operand(value) => {
                paths.push(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "-c" | "--cached" => cached = true,
            "-m" | "--modified" => modified = true,
            "-d" | "--deleted" => deleted = true,
            "-o" | "--others" => others = true,
            "-s" | "--stage" => stage = true,
            "-u" | "--unmerged" => unmerged = true,
            "--exclude-standard" => exclude_standard = true,
            "--error-unmatch" => error_unmatch = true,
            "--full-name" => {}
            "-z" => nul = true,
            _ => return usage(io, &format!("unsupported ls-files option: {name}")),
        }
    }
    if unmerged {
        // The three sides of each conflicted path, in the stage order Git prints.
        for (path, entry) in conflict::load_stages(ctx, &root) {
            if !paths.is_empty()
                && !paths
                    .iter()
                    .any(|selected| super::ignore::matches_pathspec(selected, &path))
            {
                continue;
            }
            for (stage, hash) in [(1, &entry.base), (2, &entry.ours), (3, &entry.theirs)] {
                if let Some(hash) = hash {
                    io.print(&format!("100644 {hash} {stage}\t{path}"));
                    io.out.push(if nul { 0 } else { b'\n' });
                }
            }
        }
        return 0;
    }
    if !cached && !modified && !deleted && !others {
        cached = true;
    }
    let index = repo::load_index(ctx, &root).unwrap_or_default();
    let needs_work = modified || deleted || others;
    let work = if needs_work {
        match repo::collect_working_tree(ctx, &root) {
            Ok(work) => work,
            Err(status) => return status,
        }
    } else {
        Tree::new()
    };
    let rules = if others && exclude_standard {
        ignore::load(ctx, &root)
    } else {
        ignore::IgnoreRules::default()
    };
    let mut listed: BTreeSet<String> = BTreeSet::new();
    if cached {
        listed.extend(index.keys().cloned());
    }
    if modified {
        listed.extend(
            index
                .iter()
                .filter(|(path, hash)| work.get(*path).is_some_and(|current| current != *hash))
                .map(|(path, _)| path.clone()),
        );
    }
    if deleted {
        listed.extend(
            index
                .keys()
                .filter(|path| !work.contains_key(*path))
                .cloned(),
        );
    }
    if others {
        listed.extend(
            work.keys()
                .filter(|path| !index.contains_key(*path))
                .filter(|path| !rules.is_ignored(path))
                .cloned(),
        );
    }
    let cwd_prefix = repo::relative_path(&root, &ctx.cwd).unwrap_or_default();
    let cwd_prefix = (!cwd_prefix.is_empty()).then(|| format!("{cwd_prefix}/"));
    let mut matched = false;
    for path in listed {
        let displayed = cwd_prefix
            .as_deref()
            .and_then(|prefix| path.strip_prefix(prefix))
            .unwrap_or(&path);
        if cwd_prefix.is_some() && displayed == path {
            continue;
        }
        if !paths.is_empty()
            && !paths
                .iter()
                .any(|selected| super::ignore::matches_pathspec(selected, displayed))
        {
            continue;
        }
        matched = true;
        if stage {
            let recorded = index.get(&path).cloned().unwrap_or_default();
            io.print(&format!("{} {} 0\t", recorded.mode(), recorded.hash));
        }
        io.print(displayed);
        io.out.push(if nul { 0 } else { b'\n' });
    }
    if error_unmatch && !matched {
        io.print_err("error: pathspec did not match any files\n");
        return 1;
    }
    0
}

/// Stage every tracked path that differs in the working tree, as `git commit -a` does.
pub(crate) fn stage_tracked_changes(
    ctx: &mut CommandContext<'_>,
    root: &str,
    io: &mut Io,
) -> Result<(), i32> {
    let mut index = require_index(ctx, root, io)?;
    let work = repo::collect_working_tree(ctx, root)?;
    let selected: BTreeSet<String> = index.keys().cloned().collect();
    let status = stage_paths(ctx, root, &mut index, &work, &selected, Staging::Silent, io);
    if status == 0 {
        Ok(())
    } else {
        Err(status)
    }
}
