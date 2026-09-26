//! Reference porcelain: `git rev-parse`, `git rev-list`, `git branch`, and `git tag`.
//!
//! These commands turn names into commit ids and list, create, delete, or rename the branch and
//! tag references kept under `.git/refs`.

use crate::commands::Io;
use crate::syscalls::System;

use super::log;
use super::repo::{self, Commit};
use super::{fatal, repo_error, usage, Arg, Flags, Globals};

pub(crate) fn git_rev_parse(system: &mut dyn System, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(system) else {
        return repo_error(io);
    };
    let mut abbreviate: Option<usize> = None;
    let mut abbrev_ref = false;
    let mut full_name = false;
    let mut quiet = false;
    let mut revisions: Vec<String> = Vec::new();
    for argument in Flags::new(args) {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                revisions.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "--show-toplevel" => {
                io.print(&format!("{root}\n"));
                return 0;
            }
            "--is-inside-work-tree" => {
                io.print("true\n");
                return 0;
            }
            "--is-inside-git-dir" => {
                let inside = repo::is_git_path(&root, system.cwd());
                io.print(if inside { "true\n" } else { "false\n" });
                return 0;
            }
            "--git-dir" => {
                // Git prints a path relative to the working directory when it can.
                let rendered = if system.cwd() == root {
                    repo::GIT_DIR.to_string()
                } else {
                    repo::path_join(&root, repo::GIT_DIR)
                };
                io.print(&format!("{rendered}\n"));
                return 0;
            }
            "--absolute-git-dir" => {
                io.print(&format!("{}\n", repo::path_join(&root, repo::GIT_DIR)));
                return 0;
            }
            "--is-bare-repository" => {
                io.print("false\n");
                return 0;
            }
            "--show-cdup" => {
                let depth = repo::relative_path(&root, system.cwd())
                    .map_or(0, |prefix| prefix.split('/').count());
                io.print(&format!("{}\n", "../".repeat(depth)));
                return 0;
            }
            "--show-prefix" => {
                let prefix = repo::relative_path(&root, system.cwd()).unwrap_or_default();
                if prefix.is_empty() {
                    io.out.push(b'\n');
                } else {
                    io.print(&format!("{prefix}/\n"));
                }
                return 0;
            }
            "--short" => {
                let Some(length) = attached
                    .map_or(Some(7), |value| value.parse::<usize>().ok())
                    .map(|length| length.clamp(4, 40))
                else {
                    return usage(io, "--short requires a length");
                };
                abbreviate = Some(length);
            }
            "--abbrev-ref" => abbrev_ref = true,
            "--symbolic-full-name" => full_name = true,
            "--verify" => {}
            "-q" | "--quiet" => quiet = true,
            _ => return usage(io, &format!("unsupported rev-parse option: {name}")),
        }
    }
    if revisions.is_empty() {
        revisions.push("HEAD".to_string());
    }
    for revision in &revisions {
        if full_name {
            let name = if revision == "HEAD" {
                repo::current_branch(system, &root).map_or_else(
                    || "HEAD".to_string(),
                    |branch| format!("refs/heads/{branch}"),
                )
            } else if revision.starts_with("refs/") {
                revision.clone()
            } else if repo::branch_names(system, &root).contains(revision) {
                format!("refs/heads/{revision}")
            } else if repo::reference_names(system, &root, "tags").contains(revision) {
                format!("refs/tags/{revision}")
            } else {
                String::new()
            };
            io.print(&format!("{name}\n"));
            continue;
        }
        if abbrev_ref {
            let name = if revision == "HEAD" {
                repo::current_branch(system, &root).unwrap_or_else(|| "HEAD".to_string())
            } else {
                revision.rsplit('/').next().unwrap_or(revision).to_string()
            };
            io.print(&format!("{name}\n"));
            continue;
        }
        let Some(commit) = rev_parse_object(system, &root, revision) else {
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
        io.print(&format!("{rendered}\n"));
    }
    0
}

/// Resolve one `rev-parse` operand, which may name a commit, a tree, or a blob.
fn rev_parse_object(system: &mut dyn System, root: &str, revision: &str) -> Option<String> {
    if revision.contains(':') {
        let (tree, path) = repo::tree_and_path(system, root, revision)?;
        if path.is_empty() {
            return Some(repo::tree_hash(&tree));
        }
        return tree.get(&path).map(|entry| entry.hash.clone());
    }
    if let Some(base) = revision.strip_suffix("^{tree}") {
        let commit = repo::resolve_revision(system, root, base)?;
        return Some(repo::tree_hash(&repo::commit_tree(system, root, &commit)?));
    }
    // `^{commit}` and `^{}` peel a tag to the commit it points at, which this subset already does.
    let revision = revision
        .strip_suffix("^{commit}")
        .or_else(|| revision.strip_suffix("^{}"))
        .unwrap_or(revision);
    repo::resolve_revision(system, root, revision)
}

pub(crate) fn git_rev_list(system: &mut dyn System, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(system) else {
        return repo_error(io);
    };
    let mut count = false;
    let mut limit = 10_000_usize;
    let mut revisions: Vec<String> = Vec::new();
    let mut all_references = false;
    let mut flags = Flags::new(args).valued("n");
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                revisions.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "--count" => count = true,
            "--all" => all_references = true,
            "-n" | "--max-count" => {
                let Some(value) = flags.value(attached).and_then(|value| value.parse().ok()) else {
                    return usage(io, "rev-list count must be a non-negative integer");
                };
                limit = value;
            }
            _ => match super::plumbing::count_option(&name) {
                Some(value) => limit = value,
                None => return usage(io, &format!("unsupported rev-list option: {name}")),
            },
        }
    }
    if all_references {
        revisions.extend(log::all_reference_tips(system, &root));
    }
    if revisions.is_empty() {
        revisions.push("HEAD".to_string());
    }
    let Some(mut history) = log::history_for(system, &root, &revisions, false) else {
        return super::ambiguous_argument(io, &revisions.join(" "));
    };
    history.truncate(limit);
    if count {
        io.print(&format!("{}\n", history.len()));
        return 0;
    }
    for (id, _) in history {
        io.print(&format!("{id}\n"));
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

pub(crate) fn git_branch(system: &mut dyn System, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(system) else {
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
    let mut flags = Flags::new(args).clustered("avdfqlr");
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                operands.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "--show-current" => {
                if let Some(branch) = repo::current_branch(system, &root) {
                    io.print(&format!("{branch}\n"));
                }
                return 0;
            }
            "-v" | "--verbose" => verbose = true,
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
            "--contains" | "--merged" | "--no-merged" => {
                // The revision is optional and defaults to HEAD.
                let revision = flags
                    .optional_value(attached)
                    .unwrap_or_else(|| "HEAD".to_string());
                contains = Some((branch_filter(&name), revision));
                list = true;
            }
            "-r" | "--remotes" => {
                // The simulation has no remotes, so there are no remote-tracking branches.
                return 0;
            }
            "-a" | "--all" | "--no-color" | "--no-column" => list = true,
            "-q" | "--quiet" => {}
            "--format" => {
                format = flags.value(attached);
                list = true;
            }
            "--sort" => {
                // The subset lists branches in name order, so the key is read and ignored.
                flags.value(attached);
                list = true;
            }
            _ => return usage(io, &format!("unsupported branch option: {name}")),
        }
    }
    if delete {
        if operands.is_empty() {
            return usage(io, "usage: git branch (-d|-D) NAME...");
        }
        return delete_branches(system, &root, &operands, force, io);
    }
    if rename {
        let (from, to) = match operands.as_slice() {
            [to] => (
                match repo::current_branch(system, &root) {
                    Some(branch) => branch,
                    None => return fatal(io, "cannot rename a detached HEAD"),
                },
                to.clone(),
            ),
            [from, to] => (from.clone(), to.clone()),
            _ => return usage(io, "usage: git branch -m [OLD] NEW"),
        };
        return rename_branch(system, &root, &from, &to, force, io);
    }
    if operands.is_empty() || list {
        let pattern = operands.first().map(String::as_str);
        if let Some(format) = &format {
            return list_formatted_branches(system, &root, pattern, format, io);
        }
        return list_branches(system, &root, pattern, contains.as_ref(), verbose, io);
    }
    if operands.len() > 2 || !valid_reference_name(&operands[0]) {
        return usage(io, "usage: git branch NAME [START_POINT]");
    }
    let start = operands.get(1).map_or("HEAD", String::as_str);
    let Some(commit) = repo::resolve_revision(system, &root, start) else {
        io.print_err(&format!("fatal: not a valid object name: '{start}'\n"));
        return 128;
    };
    let reference = format!("refs/heads/{}", operands[0]);
    if !force && repo::read_reference(system, &root, &reference).is_some() {
        io.print_err(&format!(
            "fatal: a branch named '{}' already exists\n",
            operands[0]
        ));
        return 128;
    }
    repo::write_reference(system, &root, &reference, &commit).map_or_else(
        |error| {
            io.print_err(&format!("git branch: {error}\n"));
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
    system: &mut dyn System,
    root: &str,
    branch: &str,
    filter: Option<&(BranchFilter, String)>,
) -> bool {
    let Some((filter, revision)) = filter else {
        return true;
    };
    let Some(target) = repo::resolve_revision(system, root, revision) else {
        return false;
    };
    let Some(tip) = repo::read_reference(system, root, &format!("refs/heads/{branch}")) else {
        return false;
    };
    match filter {
        BranchFilter::Merged => repo::ancestors(system, root, &target).contains(&tip),
        BranchFilter::NotMerged => !repo::ancestors(system, root, &target).contains(&tip),
        BranchFilter::Contains => repo::ancestors(system, root, &tip).contains(&target),
    }
}

fn list_branches(
    system: &mut dyn System,
    root: &str,
    pattern: Option<&str>,
    contains: Option<&(BranchFilter, String)>,
    verbose: bool,
    io: &mut Io,
) -> i32 {
    let current = repo::current_branch(system, root);
    // A detached HEAD is listed first, as the checked-out "branch" it stands in for.
    let detached = match (&current, repo::head_commit(system, root)) {
        (None, Some(commit)) => Some(format!("(HEAD detached at {})", repo::short(&commit))),
        _ => None,
    };
    let branches = repo::branch_names(system, root);
    let width = branches
        .iter()
        .map(String::len)
        .chain(detached.iter().map(String::len))
        .max()
        .unwrap_or(0);
    if let Some(label) = &detached {
        if pattern.is_none() {
            if verbose {
                let commit = repo::head_commit(system, root).unwrap_or_default();
                let subject = repo::load_commit(system, root, &commit)
                    .map(|commit| commit.subject().to_string())
                    .unwrap_or_default();
                io.print(&format!(
                    "* {label:width$} {} {subject}\n",
                    repo::short(&commit)
                ));
            } else {
                io.print(&format!("* {label}\n"));
            }
        }
    }
    for branch in branches {
        if let Some(pattern) = pattern {
            if !crate::commands::util::glob_eq(pattern, &branch) {
                continue;
            }
        }
        if !branch_selected(system, root, &branch, contains) {
            continue;
        }
        let marker = if current.as_deref() == Some(branch.as_str()) {
            '*'
        } else {
            ' '
        };
        if !verbose {
            io.print(&format!("{marker} {branch}\n"));
            continue;
        }
        let summary = repo::read_reference(system, root, &format!("refs/heads/{branch}"))
            .map(|commit| {
                let subject = repo::load_commit(system, root, &commit)
                    .map(|commit| commit.subject().to_string())
                    .unwrap_or_default();
                format!("{} {subject}", repo::short(&commit))
            })
            .unwrap_or_default();
        io.print(&format!("{marker} {branch:width$} {summary}\n"));
    }
    0
}

/// List branches through a `--format` template.
fn list_formatted_branches(
    system: &mut dyn System,
    root: &str,
    pattern: Option<&str>,
    format: &str,
    io: &mut Io,
) -> i32 {
    for branch in repo::branch_names(system, root) {
        if let Some(pattern) = pattern {
            if !crate::commands::util::glob_eq(pattern, &branch) {
                continue;
            }
        }
        let name = format!("refs/heads/{branch}");
        let Some(commit) = repo::read_reference(system, root, &name) else {
            continue;
        };
        let Some(line) = super::plumbing::expand_ref_format(format, &name, &commit) else {
            return usage(io, &format!("unsupported branch format: {format}"));
        };
        io.print(&format!("{line}\n"));
    }
    0
}

fn delete_branches(
    system: &mut dyn System,
    root: &str,
    names: &[String],
    force: bool,
    io: &mut Io,
) -> i32 {
    let current = repo::current_branch(system, root);
    for name in names {
        if current.as_deref() == Some(name.as_str()) {
            io.print_err(&format!(
                "error: cannot delete branch '{name}' checked out\n"
            ));
            return 1;
        }
        let reference = format!("refs/heads/{name}");
        let Some(commit) = repo::read_reference(system, root, &reference) else {
            io.print_err(&format!("error: branch '{name}' not found\n"));
            return 1;
        };
        if !force {
            // Refuse to drop work that HEAD cannot reach, as Git's `-d` does.
            let reachable = repo::head_commit(system, root)
                .map(|head| repo::ancestors(system, root, &head))
                .unwrap_or_default();
            if !reachable.contains(&commit) {
                io.print_err(&format!(
                    "error: the branch '{name}' is not fully merged; use -D to delete it\n"
                ));
                return 1;
            }
        }
        if repo::delete_reference(system, root, &reference).is_err() {
            io.print_err(&format!("error: branch '{name}' not found\n"));
            return 1;
        }
        io.print(&format!(
            "Deleted branch {name} (was {}).\n",
            repo::short(&commit)
        ));
    }
    0
}

fn rename_branch(
    system: &mut dyn System,
    root: &str,
    from: &str,
    to: &str,
    force: bool,
    io: &mut Io,
) -> i32 {
    if !valid_reference_name(to) {
        return fatal(io, &format!("invalid branch name: {to}"));
    }
    let Some(commit) = repo::read_reference(system, root, &format!("refs/heads/{from}")) else {
        io.print_err(&format!("error: branch '{from}' not found\n"));
        return 1;
    };
    if !force && repo::read_reference(system, root, &format!("refs/heads/{to}")).is_some() {
        io.print_err(&format!("fatal: a branch named '{to}' already exists\n"));
        return 128;
    }
    if let Err(error) = repo::write_reference(system, root, &format!("refs/heads/{to}"), &commit) {
        return super::cannot_write(io, &format!("refs/heads/{to}"), &error);
    }
    if let Err(error) = repo::delete_reference(system, root, &format!("refs/heads/{from}")) {
        return super::cannot_write(io, &format!("refs/heads/{from}"), &error);
    }
    if repo::current_branch(system, root).as_deref() == Some(from) {
        let action = format!("branch: renamed to {to}");
        if let Err(error) = repo::set_head_to_branch(system, root, to, &action) {
            return super::cannot_write(io, "HEAD", &error);
        }
    }
    0
}

pub(crate) fn git_tag(
    system: &mut dyn System,
    globals: &Globals,
    args: &[String],
    io: &mut Io,
) -> i32 {
    let Some(root) = repo::find_repo_root(system) else {
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
    let mut flags = Flags::new(args).clustered("adflqm").valued("m");
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                operands.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "-l" | "--list" => list = true,
            "-d" | "--delete" => delete = true,
            "-f" | "--force" => force = true,
            "-a" | "--annotate" => {}
            "-m" | "--message" => message = flags.value(attached),
            "-q" | "--quiet" => {}
            "--format" => {
                format = flags.value(attached);
                list = true;
            }
            "--points-at" => {
                points_at = flags.value(attached);
                list = true;
            }
            "--sort" => {
                // The subset lists tags in name order, so the key is read and ignored.
                flags.value(attached);
                list = true;
            }
            // `-n` and `-n<count>` both ask for the annotation; the subset prints all of it.
            "-n" | "--list-annotations" => {
                list = true;
                annotations = true;
            }
            _ => match super::plumbing::count_option(&name) {
                Some(_) => {
                    list = true;
                    annotations = true;
                }
                None => return usage(io, &format!("unsupported tag option: {name}")),
            },
        }
    }
    if delete {
        for name in &operands {
            let was = repo::read_reference(system, &root, &format!("refs/tags/{name}"))
                .unwrap_or_default();
            if repo::delete_reference(system, &root, &format!("refs/tags/{name}")).is_err() {
                io.print_err(&format!("error: tag '{name}' not found.\n"));
                return 1;
            }
            io.print(&format!(
                "Deleted tag '{name}' (was {})\n",
                repo::short(&was)
            ));
        }
        return 0;
    }
    if operands.is_empty() || list {
        // `--points-at` keeps only the tags on one commit.
        let target = match points_at {
            Some(revision) => match repo::resolve_revision(system, &root, &revision) {
                Some(commit) => Some(commit),
                None => return super::ambiguous_argument(io, &revision),
            },
            None => None,
        };
        for name in repo::reference_names(system, &root, "tags") {
            if let Some(pattern) = operands.first() {
                if !crate::commands::util::glob_eq(pattern, &name) {
                    continue;
                }
            }
            let reference = format!("refs/tags/{name}");
            let commit = repo::read_reference(system, &root, &reference).unwrap_or_default();
            if target.as_ref().is_some_and(|target| *target != commit) {
                continue;
            }
            if let Some(format) = &format {
                let Some(line) = super::plumbing::expand_ref_format(format, &reference, &commit)
                else {
                    return usage(io, &format!("unsupported tag format: {format}"));
                };
                io.print(&format!("{line}\n"));
                continue;
            }
            match repo::read_annotation(system, &root, &name).filter(|_| annotations) {
                Some(annotation) => io
                    .out
                    .extend_from_slice(format!("{name:<15} {}\n", annotation.subject()).as_bytes()),
                None => io.print(&format!("{name}\n")),
            }
        }
        return 0;
    }
    let name = &operands[0];
    if !valid_reference_name(name) {
        return fatal(io, &format!("invalid tag name: {name}"));
    }
    let start = operands.get(1).map_or("HEAD", String::as_str);
    let Some(commit) = repo::resolve_revision(system, &root, start) else {
        io.print_err(&format!("fatal: not a valid object name: '{start}'\n"));
        return 128;
    };
    let reference = format!("refs/tags/{name}");
    if !force && repo::read_reference(system, &root, &reference).is_some() {
        io.print_err(&format!("fatal: tag '{name}' already exists\n"));
        return 128;
    }
    if let Err(error) = repo::write_reference(system, &root, &reference, &commit) {
        return super::cannot_write(io, &reference, &error);
    }
    if let Some(message) = message {
        // The annotation is stored beside the ref; the ref itself stays lightweight.
        let (author_name, author_email) = repo::author_identity(system, &root, globals);
        let annotation = Commit {
            parents: Vec::new(),
            author_name,
            author_email,
            timestamp: repo::now_seconds(system),
            message,
        };
        if let Err(error) = repo::write_annotation(system, &root, name, &annotation) {
            return super::cannot_write(io, "the tag annotation", &error);
        }
    }
    0
}
