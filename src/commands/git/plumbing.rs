//! Inspection commands that read repository state without changing it.
//!
//! These are the plumbing and reporting commands agents reach for when orienting in a repository:
//! object inspection, tree listing, ignore checking, ancestry queries, and content search. They
//! share the storage layer in [`super::repo`] and never touch the host.

use std::collections::BTreeMap;

use crate::commands::{CommandContext, Io};
use crate::vfs::resolve_against;

use super::diff;
use super::ignore;
use super::repo;
use super::{cannot_write, fatal, repo_error, usage, Arg, Flags};

/// The most files `git grep` will search in one invocation.
const MAX_GREP_FILES: usize = 10_000;

pub(crate) fn git_cat_file(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut mode = None;
    let mut object = None;
    for argument in Flags::new(args) {
        let name = match argument {
            Arg::Operand(value) => {
                object = Some(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "-p" | "-t" | "-s" | "-e" => mode = Some(name),
            _ => return usage(io, &format!("unsupported cat-file option: {name}")),
        }
    }
    let (Some(mode), Some(object)) = (mode, object) else {
        return usage(io, "usage: git cat-file (-p|-t|-s|-e) OBJECT");
    };
    // An object is either a blob hash or a revision, optionally with a `:PATH` suffix.
    let blob = match object.contains(':') {
        true => repo::tree_and_path(ctx, &root, &object)
            .and_then(|(tree, path)| tree.get(&path).map(|entry| entry.hash.clone())),
        false => Some(object.clone()),
    };
    if let Some(data) = blob
        .as_deref()
        .and_then(|hash| repo::read_blob(ctx, &root, hash))
    {
        return match mode.as_str() {
            "-t" => {
                io.print("blob\n");
                0
            }
            "-s" => {
                io.print(&format!("{}\n", data.len()));
                0
            }
            "-e" => 0,
            _ => {
                io.out.extend_from_slice(&data);
                0
            }
        };
    }
    let Some(id) = repo::resolve_revision(ctx, &root, &object) else {
        io.print_err(&format!("fatal: Not a valid object name {object}\n"));
        return 128;
    };
    let Some(commit) = repo::load_commit(ctx, &root, &id) else {
        io.print_err(&format!("fatal: Not a valid object name {object}\n"));
        return 128;
    };
    let tree = repo::commit_tree(ctx, &root, &id).unwrap_or_default();
    let mut body = format!("tree {}\n", repo::tree_hash(&tree));
    for parent in &commit.parents {
        body.push_str(&format!("parent {parent}\n"));
    }
    let identity = format!(
        "{} <{}> {} +0000",
        commit.author_name, commit.author_email, commit.timestamp
    );
    body.push_str(&format!("author {identity}\ncommitter {identity}\n\n"));
    body.push_str(&commit.message);
    body.push('\n');
    // An annotated tag is its own object type, even though it resolves to a commit here.
    let annotated = object
        .strip_prefix("refs/tags/")
        .map(str::to_string)
        .or_else(|| Some(object.clone()))
        .filter(|name| repo::read_annotation(ctx, &root, name).is_some())
        .is_some();
    match mode.as_str() {
        "-t" if annotated => io.print("tag\n"),
        "-t" => io.print("commit\n"),
        "-e" => {}
        "-s" => io
            .out
            .extend_from_slice(format!("{}\n", body.len()).as_bytes()),
        _ => io.print(&body),
    }
    0
}

pub(crate) fn git_hash_object(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut write = false;
    let mut stdin = false;
    let mut files = Vec::new();
    let mut flags = Flags::new(args).valued("t");
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                files.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "-w" => write = true,
            "--stdin" => stdin = true,
            // Blobs are the only object this computes an id for.
            "-t" | "--type" => match flags.value(attached).as_deref() {
                Some("blob") => {}
                other => {
                    return usage(
                        io,
                        &format!("unsupported object type: {}", other.unwrap_or("")),
                    )
                }
            },
            _ => return usage(io, &format!("unsupported hash-object option: {name}")),
        }
    }
    let mut contents: Vec<Vec<u8>> = Vec::new();
    if stdin {
        contents.push(io.stdin.clone());
    }
    for file in &files {
        let absolute = resolve_against(&ctx.cwd, file);
        match ctx.fs_read_limited("/", &absolute, 16 * 1024 * 1024) {
            Ok(data) => contents.push(data),
            Err(_) => {
                io.print_err(&format!("fatal: could not open '{file}' for reading\n"));
                return 128;
            }
        }
    }
    if contents.is_empty() {
        return usage(io, "usage: git hash-object [-w] [--stdin] [FILE...]");
    }
    let root = repo::find_repo_root(ctx);
    for data in contents {
        let hash = repo::blob_hash(&data);
        if write {
            let Some(root) = root.as_deref() else {
                return repo_error(io);
            };
            if let Err(error) = repo::write_blob(ctx, root, &data) {
                return cannot_write(io, "the object", &error);
            }
        }
        io.print(&format!("{hash}\n"));
    }
    0
}

pub(crate) fn git_ls_tree(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut name_only = false;
    let mut recursive = false;
    let mut directories_only = false;
    let mut operands = Vec::new();
    for argument in Flags::new(args).clustered("rd") {
        let name = match argument {
            Arg::Operand(value) => {
                operands.push(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "--name-only" | "--name-status" => name_only = true,
            "-r" => recursive = true,
            "-d" => directories_only = true,
            "--full-name" => {}
            _ => return usage(io, &format!("unsupported ls-tree option: {name}")),
        }
    }
    let Some((revision, paths)) = operands.split_first() else {
        return usage(
            io,
            "usage: git ls-tree [-r] [--name-only] REVISION [PATH...]",
        );
    };
    let Some(commit) = repo::resolve_revision(ctx, &root, revision) else {
        return super::ambiguous_argument(io, revision);
    };
    let tree = repo::commit_tree(ctx, &root, &commit).unwrap_or_default();
    let prefix = paths
        .first()
        .map(|path| path.trim_end_matches('/').to_string())
        .unwrap_or_default();
    // Without -r, entries below the listed directory collapse into that directory.
    let mut listed: BTreeMap<String, Option<repo::Entry>> = BTreeMap::new();
    for (path, entry) in &tree {
        let relative = if prefix.is_empty() {
            Some(path.as_str())
        } else {
            path.strip_prefix(&format!("{prefix}/"))
        };
        let Some(relative) = relative else {
            continue;
        };
        match relative.split_once('/') {
            Some((directory, _)) if !recursive => {
                let name = if prefix.is_empty() {
                    directory.to_string()
                } else {
                    format!("{prefix}/{directory}")
                };
                listed.insert(name, None);
            }
            _ => {
                listed.insert(path.clone(), Some(entry.clone()));
            }
        }
    }
    for (name, entry) in listed {
        if directories_only && entry.is_some() {
            continue;
        }
        if name_only {
            io.print(&format!("{name}\n"));
            continue;
        }
        let line = match entry {
            Some(entry) => format!("{} blob {}\t{name}\n", entry.mode(), entry.hash),
            None => format!("040000 tree {}\t{name}\n", subtree_hash(&tree, &name)),
        };
        io.print(&line);
    }
    0
}

/// The identifier of the subtree rooted at `directory`.
fn subtree_hash(tree: &repo::Tree, directory: &str) -> String {
    let prefix = format!("{directory}/");
    let subtree: repo::Tree = tree
        .iter()
        .filter_map(|(path, entry)| Some((path.strip_prefix(&prefix)?.to_string(), entry.clone())))
        .collect();
    repo::tree_hash(&subtree)
}

pub(crate) fn git_check_ignore(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut paths = Vec::new();
    let mut verbose = false;
    let mut quiet = false;
    let mut non_matching = false;
    let mut from_stdin = false;
    for argument in Flags::new(args).clustered("vqn") {
        let name = match argument {
            Arg::Operand(value) => {
                paths.push(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "-v" | "--verbose" => verbose = true,
            "-q" | "--quiet" => quiet = true,
            "-n" | "--non-matching" => non_matching = true,
            "--stdin" => from_stdin = true,
            "--no-index" => {}
            _ => return usage(io, &format!("unsupported check-ignore option: {name}")),
        }
    }
    if from_stdin {
        paths.extend(
            String::from_utf8_lossy(&io.stdin)
                .lines()
                .map(str::to_string),
        );
    }
    if paths.is_empty() {
        return usage(io, "usage: git check-ignore PATH...");
    }
    if non_matching && !verbose {
        return usage(io, "check-ignore: -n requires -v");
    }
    let rules = ignore::load(ctx, &root);
    let cwd = ctx.cwd.clone();
    let mut any = false;
    for path in &paths {
        let relative = super::pathspec(&cwd, &root, path);
        let rule = rules.describe(&relative);
        any |= rule.is_some();
        if quiet {
            continue;
        }
        match (&rule, non_matching) {
            (Some(rule), _) if verbose => {
                io.print(&format!("{rule}\t{path}\n"));
            }
            (Some(_), _) => io.print(&format!("{path}\n")),
            // `-n` reports the paths no pattern covers, with empty source fields.
            (None, true) => io.print(&format!("::\t{path}\n")),
            (None, false) => {}
        }
    }
    i32::from(!any)
}

pub(crate) fn git_merge_base(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut is_ancestor = false;
    let mut revisions = Vec::new();
    for argument in Flags::new(args) {
        let name = match argument {
            Arg::Operand(value) => {
                revisions.push(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "--is-ancestor" => is_ancestor = true,
            _ => return usage(io, &format!("unsupported merge-base option: {name}")),
        }
    }
    let [left, right] = revisions.as_slice() else {
        return usage(
            io,
            "usage: git merge-base [--is-ancestor] REVISION REVISION",
        );
    };
    let (Some(left), Some(right)) = (
        repo::resolve_revision(ctx, &root, left),
        repo::resolve_revision(ctx, &root, right),
    ) else {
        return super::ambiguous_argument(io, &revisions.join(" "));
    };
    if is_ancestor {
        return i32::from(!repo::ancestors(ctx, &root, &right).contains(&left));
    }
    match repo::merge_base(ctx, &root, &left, &right) {
        Some(base) => {
            io.print(&format!("{base}\n"));
            0
        }
        None => 1,
    }
}

pub(crate) fn git_describe(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut always = false;
    let mut lightweight = false;
    let mut revision = "HEAD".to_string();
    for argument in Flags::new(args) {
        let name = match argument {
            Arg::Operand(value) => {
                revision = value;
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "--tags" => lightweight = true,
            "--abbrev" | "--long" => {}
            "--always" => always = true,
            _ => return usage(io, &format!("unsupported describe option: {name}")),
        }
    }
    let Some(start) = repo::resolve_revision(ctx, &root, &revision) else {
        return super::ambiguous_argument(io, &revision);
    };
    let tags: BTreeMap<String, String> = repo::reference_names(ctx, &root, "tags")
        .into_iter()
        .filter(|tag| lightweight || repo::read_annotation(ctx, &root, tag).is_some())
        .filter_map(|tag| {
            let commit = repo::read_reference(ctx, &root, &format!("refs/tags/{tag}"))?;
            Some((commit, tag))
        })
        .collect();
    for (distance, (id, _)) in repo::first_parent_history(ctx, &root, &start, 10_000)
        .into_iter()
        .enumerate()
    {
        let Some(tag) = tags.get(&id) else {
            continue;
        };
        let described = if distance == 0 {
            tag.clone()
        } else {
            format!("{tag}-{distance}-g{}", repo::short(&start))
        };
        io.print(&format!("{described}\n"));
        return 0;
    }
    if always {
        io.print(&format!("{}\n", repo::short(&start)));
        return 0;
    }
    io.print_err(&format!(
        "fatal: No annotated tags can describe '{start}'.\n"
    ));
    if !repo::reference_names(ctx, &root, "tags").is_empty() {
        io.print_err("However, there were unannotated tags: try --tags.\n");
    }
    128
}

pub(crate) fn git_shortlog(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut summary = false;
    let mut numbered = false;
    let mut with_email = false;
    let mut revisions: Vec<String> = Vec::new();
    for argument in Flags::new(args).clustered("sne") {
        let name = match argument {
            Arg::Operand(value) => {
                revisions.push(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "-s" | "--summary" => summary = true,
            "-n" | "--numbered" => numbered = true,
            "-e" | "--email" => with_email = true,
            "--no-merges" => {}
            _ => return usage(io, &format!("unsupported shortlog option: {name}")),
        }
    }
    let starts: Vec<String> = if revisions.is_empty() {
        repo::head_commit(ctx, &root).into_iter().collect()
    } else {
        let mut resolved = Vec::new();
        for revision in &revisions {
            let Some(commit) = repo::resolve_revision(ctx, &root, revision) else {
                return super::ambiguous_argument(io, revision);
            };
            resolved.push(commit);
        }
        resolved
    };
    if starts.is_empty() {
        return 0;
    }
    let mut counts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (_, commit) in repo::reachable_history(ctx, &root, &starts, 10_000) {
        let author = if with_email {
            format!("{} <{}>", commit.author_name, commit.author_email)
        } else {
            commit.author_name.clone()
        };
        counts
            .entry(author)
            .or_default()
            .push(commit.subject().to_string());
    }
    let mut authors: Vec<(String, Vec<String>)> = counts.into_iter().collect();
    if numbered {
        authors.sort_by(|left, right| right.1.len().cmp(&left.1.len()).then(left.0.cmp(&right.0)));
    }
    for (author, subjects) in authors {
        if summary {
            io.print(&format!("{:>6}\t{author}\n", subjects.len()));
            continue;
        }
        io.print(&format!("{author} ({}):\n", subjects.len()));
        for subject in subjects {
            io.print(&format!("      {subject}\n"));
        }
        io.out.push(b'\n');
    }
    0
}

/// Search tracked files, as `git grep` does.
pub(crate) fn git_grep(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut line_numbers = false;
    let mut ignore_case = false;
    let mut names_only = false;
    let mut count_only = false;
    let mut fixed = false;
    let mut invert = false;
    let mut word = false;
    let mut without_match = false;
    let mut pattern = None;
    let mut paths: Vec<String> = Vec::new();
    let mut before_separator = usize::MAX;
    let mut flags = Flags::new(args).clustered("niIlcEFv").valued("e");
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                if flags.separated() && before_separator == usize::MAX {
                    before_separator = paths.len();
                }
                if pattern.is_none() && !flags.separated() {
                    pattern = Some(value);
                } else {
                    paths.push(value);
                }
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "-n" | "--line-number" => line_numbers = true,
            "-i" | "--ignore-case" => ignore_case = true,
            "-l" | "--files-with-matches" | "--name-only" => names_only = true,
            "-c" | "--count" => count_only = true,
            "-F" | "--fixed-strings" => fixed = true,
            "-w" | "--word-regexp" => word = true,
            "-L" | "--files-without-match" => {
                names_only = true;
                without_match = true;
            }
            "-v" | "--invert-match" => invert = true,
            "-E" | "--extended-regexp" | "-I" | "--no-color" | "--cached" => {}
            "-e" => pattern = flags.value(attached),
            _ => return usage(io, &format!("unsupported grep option: {name}")),
        }
    }
    let Some(pattern) = pattern else {
        return usage(io, "usage: git grep [-n] [-i] [-l] PATTERN [-- PATH...]");
    };
    let mut expression = if fixed {
        regex::escape(&pattern)
    } else {
        pattern.clone()
    };
    if word {
        expression = format!(r"\b(?:{expression})\b");
    }
    let Ok(regex) = regex::RegexBuilder::new(&expression)
        .case_insensitive(ignore_case)
        .build()
    else {
        io.print_err(&format!("fatal: invalid pattern: {pattern}\n"));
        return 128;
    };
    let cwd = ctx.cwd.clone();
    // An operand before `--` that names no file but does resolve is a revision to search.
    let revision = paths
        .first()
        .filter(|_| before_separator.min(paths.len()) > 0)
        .filter(|candidate| !super::names_a_path(ctx, &root, candidate))
        .and_then(|candidate| {
            repo::resolve_revision(ctx, &root, candidate).map(|commit| (candidate.clone(), commit))
        });
    if revision.is_some() {
        paths.remove(0);
    }
    let paths: Vec<String> = paths
        .iter()
        .map(|path| super::pathspec(&cwd, &root, path))
        .collect();
    // Git searches only below the working directory and reports paths relative to it.
    let scope = repo::relative_path(&root, &cwd)
        .filter(|prefix| !prefix.is_empty())
        .map(|prefix| format!("{prefix}/"));
    let tree = match &revision {
        Some((_, commit)) => repo::commit_tree(ctx, &root, commit).unwrap_or_default(),
        None => repo::load_index(ctx, &root).unwrap_or_default(),
    };
    let mut found = false;
    for (searched, (path, blob)) in tree.iter().enumerate() {
        if searched >= MAX_GREP_FILES {
            break;
        }
        if !super::compare::selected(&paths, path) {
            continue;
        }
        let displayed = match &scope {
            Some(prefix) => match path.strip_prefix(prefix.as_str()) {
                Some(relative) => relative,
                None => continue,
            },
            None => path.as_str(),
        };
        let displayed = match &revision {
            Some((name, _)) => format!("{name}:{displayed}"),
            None => displayed.to_string(),
        };
        let data = match &revision {
            Some(_) => repo::read_blob(ctx, &root, &blob.hash),
            None => repo::read_work_file(ctx, &root, path),
        };
        let Some(data) = data else {
            continue;
        };
        if !ctx.charge_cpu(data.len() as u64) {
            return repo::resource_error(ctx);
        }
        if super::diff::is_binary(&data) {
            continue;
        }
        let text = String::from_utf8_lossy(&data);
        let mut matches = 0;
        for (number, line) in text.lines().enumerate() {
            if regex.is_match(line) == invert {
                continue;
            }
            matches += 1;
            found = true;
            if names_only || count_only {
                break;
            }
            let location = if line_numbers {
                format!("{displayed}:{}:", number + 1)
            } else {
                format!("{displayed}:")
            };
            io.print(&format!("{location}{line}\n"));
        }
        if names_only && (matches != 0) != without_match {
            found = true;
            io.print(&format!("{displayed}\n"));
        }
        if count_only {
            let total = if matches == 0 {
                0
            } else {
                text.lines()
                    .filter(|line| regex.is_match(line) != invert)
                    .count()
            };
            if total != 0 {
                io.print(&format!("{displayed}:{total}\n"));
            }
        }
    }
    i32::from(!found)
}

/// Every reference in the repository, as `refs/...` names paired with the commit they point at.
fn all_references(ctx: &CommandContext<'_>, root: &str) -> Vec<(String, String)> {
    let mut references = Vec::new();
    for kind in ["heads", "tags"] {
        for name in repo::reference_names(ctx, root, kind) {
            let full = format!("refs/{kind}/{name}");
            if let Some(commit) = repo::read_reference(ctx, root, &full) {
                references.push((full, commit));
            }
        }
    }
    references.sort();
    references
}

pub(crate) fn git_show_ref(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut heads_only = false;
    let mut tags_only = false;
    let mut verify = false;
    let mut patterns: Vec<String> = Vec::new();
    for argument in Flags::new(args) {
        let name = match argument {
            Arg::Operand(value) => {
                patterns.push(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "--heads" => heads_only = true,
            "--tags" => tags_only = true,
            "--verify" => verify = true,
            "-q" | "--quiet" | "--hash" | "-d" | "--dereference" => {}
            _ => return usage(io, &format!("unsupported show-ref option: {name}")),
        }
    }
    let mut matched = false;
    for (name, commit) in all_references(ctx, &root) {
        if heads_only && !name.starts_with("refs/heads/") {
            continue;
        }
        if tags_only && !name.starts_with("refs/tags/") {
            continue;
        }
        // `--verify` needs the full name; otherwise any trailing component may be given.
        let selected = patterns.is_empty()
            || patterns.iter().any(|pattern| {
                if verify {
                    name == *pattern
                } else {
                    name == *pattern || name.ends_with(&format!("/{pattern}"))
                }
            });
        if !selected {
            continue;
        }
        matched = true;
        io.print(&format!("{commit} {name}\n"));
    }
    i32::from(!matched)
}

pub(crate) fn git_symbolic_ref(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut short = false;
    let mut operands: Vec<String> = Vec::new();
    for argument in Flags::new(args) {
        let name = match argument {
            Arg::Operand(value) => {
                operands.push(value);
                continue;
            }
            Arg::Option { name, .. } => name,
        };
        match name.as_str() {
            "--short" => short = true,
            "-q" | "--quiet" => {}
            _ => return usage(io, &format!("unsupported symbolic-ref option: {name}")),
        }
    }
    match operands.as_slice() {
        [name] if name == repo::HEAD => match repo::head_reference(ctx, &root) {
            Some(reference) => {
                let rendered = if short {
                    reference
                        .rsplit('/')
                        .next()
                        .unwrap_or(&reference)
                        .to_string()
                } else {
                    reference
                };
                io.print(&format!("{rendered}\n"));
                0
            }
            None => {
                io.print_err("fatal: ref HEAD is not a symbolic ref\n");
                1
            }
        },
        [name, target] if name == repo::HEAD => {
            let Some(branch) = target.strip_prefix("refs/heads/") else {
                return fatal(io, "only refs/heads/* can be pointed at by HEAD");
            };
            repo::set_head_to_branch(ctx, &root, branch, "symbolic-ref: update").map_or(1, |()| 0)
        }
        _ => usage(io, "usage: git symbolic-ref [--short] HEAD [REF]"),
    }
}

pub(crate) fn git_for_each_ref(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut format = None;
    let mut prefixes: Vec<String> = Vec::new();
    let mut flags = Flags::new(args);
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                prefixes.push(value.trim_end_matches('*').to_string());
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "--format" => format = flags.value(attached),
            // Every reference is listed, in name order, so neither of these changes anything.
            "--count" | "--sort" => {
                flags.value(attached);
            }
            _ => return usage(io, &format!("unsupported for-each-ref option: {name}")),
        }
    }
    let format = format.unwrap_or_else(|| "%(objectname) %(objecttype)\t%(refname)".to_string());
    for (name, commit) in all_references(ctx, &root) {
        if !prefixes.is_empty() && !prefixes.iter().any(|prefix| name.starts_with(prefix)) {
            continue;
        }
        let Some(line) = expand_ref_format(&format, &name, &commit) else {
            return usage(io, &format!("unsupported for-each-ref format: {format}"));
        };
        io.print(&format!("{line}\n"));
    }
    0
}

/// Expand the `%(field)` placeholders `for-each-ref` and `--format` accept.
pub(crate) fn expand_ref_format(format: &str, name: &str, commit: &str) -> Option<String> {
    let short = name
        .strip_prefix("refs/heads/")
        .or_else(|| name.strip_prefix("refs/tags/"))
        .unwrap_or(name);
    let mut out = String::new();
    let mut rest = format;
    while let Some(start) = rest.find("%(") {
        out.push_str(&rest[..start]);
        let end = rest[start..].find(')')? + start;
        let value = match &rest[start + 2..end] {
            "refname" => name.to_string(),
            "refname:short" | "refname:lstrip=2" => short.to_string(),
            "objectname" => commit.to_string(),
            "objectname:short" => repo::short(commit).to_string(),
            "objecttype" => "commit".to_string(),
            _ => return None,
        };
        out.push_str(&value);
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    Some(out)
}

// -- blame ---------------------------------------------------------------------------------------

/// The most commits `git blame` will walk back through for one file.
const MAX_BLAME_COMMITS: usize = 2000;

/// Attribute each line of a file to the commit that last changed it.
///
/// The walk starts at the requested revision and follows first parents. At each step the parent's
/// version of the file is diffed against the child's: a line both versions keep carries on to the
/// parent, and a line only the child has belongs to the child. Whatever is left when the file runs
/// out of history belongs to the commit that introduced it.
pub(crate) fn git_blame(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut suppress = false;
    let mut long = false;
    let mut range: Option<String> = None;
    let mut operands: Vec<String> = Vec::new();
    // Where `--` fell, so `git blame REV -- PATH` names a path that the working tree lost.
    let mut after_separator = None;
    let mut flags = Flags::new(args).clustered("slwe").valued("L");
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(value) => {
                if flags.separated() && after_separator.is_none() {
                    after_separator = Some(operands.len());
                }
                operands.push(value);
                continue;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "-s" => suppress = true,
            "-l" => long = true,
            // Blame here has no similarity detection or whitespace modes to turn on.
            "-w" | "-e" | "--show-email" | "--root" => {}
            "-L" => {
                let Some(value) = flags.value(attached) else {
                    return usage(io, "-L requires a line range");
                };
                range = Some(value);
            }
            _ => return usage(io, &format!("unsupported blame option: {name}")),
        }
    }
    // The file is whatever followed `--`, or else the operand that names one; anything left over
    // is the revision to start from.
    let found = after_separator
        .filter(|position| *position < operands.len())
        .or_else(|| {
            operands
                .iter()
                .rposition(|operand| super::names_a_path(ctx, &root, operand))
        });
    let Some(position) = found else {
        return usage(io, "usage: git blame [-s] [-L RANGE] [REVISION] FILE");
    };
    let file = operands.remove(position);
    let revision = operands
        .first()
        .cloned()
        .unwrap_or_else(|| "HEAD".to_string());
    let Some(start) = repo::resolve_revision(ctx, &root, &revision) else {
        return super::ambiguous_argument(io, &revision);
    };
    let path = super::pathspec(&ctx.cwd, &root, &file);
    let tip = repo::commit_tree(ctx, &root, &start).unwrap_or_default();
    let Some(content) = tip
        .get(&path)
        .and_then(|entry| repo::read_blob(ctx, &root, &entry.hash))
    else {
        io.print_err(&format!("fatal: no such path {file} in {revision}\n"));
        return 128;
    };
    // Blaming the checked-out file, as Git does, means a local edit shows up as uncommitted work.
    let pending = (operands.is_empty() || revision == "HEAD")
        .then(|| repo::read_work_file(ctx, &root, &path))
        .flatten()
        .filter(|work| *work != content);
    let content = pending.clone().unwrap_or(content);
    let lines: Vec<String> = diff::split_lines(&content)
        .into_iter()
        .map(|line| line.trim_end_matches('\n').to_string())
        .collect();
    if !ctx.charge_cpu((lines.len() as u64).saturating_mul(64)) {
        return repo::resource_error(ctx);
    }
    let stored = pending.map(|_| tip[&path].hash.clone());
    let Some(origins) = trace_lines(ctx, &root, &start, &path, &content, stored, lines.len())
    else {
        return repo::resource_error(ctx);
    };
    let format = BlameFormat {
        range: range.as_deref(),
        suppress,
        long,
    };
    emit_blame(ctx, &root, &lines, &origins, &format, io)
}

/// The name blame gives lines that are only in the working tree.
const UNCOMMITTED: &str = "0000000000000000000000000000000000000000";

/// Carry each line of `new` back to `old`, claiming for `commit` the lines `old` does not have.
///
/// Returns the mapping for `old`: the line of the file as it is now that each of its lines became,
/// or `None` for a line that `new` dropped and so survives nowhere.
fn carry_lines(
    old: &[u8],
    new: &[u8],
    mapping: &[Option<usize>],
    commit: &str,
    origins: &mut [Option<Origin>],
) -> Vec<Option<usize>> {
    let old_lines = diff::split_lines(old);
    let new_lines = diff::split_lines(new);
    let mut carried = Vec::with_capacity(old_lines.len());
    let mut at = 0;
    for edit in diff::edit_script(&old_lines, &new_lines) {
        match edit.op {
            diff::Op::Keep => {
                carried.push(mapping[at]);
                at += 1;
            }
            diff::Op::Insert => {
                if let Some(line) = mapping[at] {
                    origins[line].get_or_insert(Origin {
                        commit: commit.to_string(),
                        boundary: false,
                    });
                }
                at += 1;
            }
            // `old` had this line and `new` does not, so it reaches nothing in the file today.
            diff::Op::Delete => carried.push(None),
        }
    }
    carried
}

/// The commit each line came from, and whether that commit is where the walk stopped.
struct Origin {
    commit: String,
    boundary: bool,
}

fn trace_lines(
    ctx: &mut CommandContext<'_>,
    root: &str,
    start: &str,
    path: &str,
    content: &[u8],
    // The blob the starting commit holds, when the working tree has moved on from it.
    stored: Option<String>,
    count: usize,
) -> Option<Vec<Origin>> {
    let mut origins: Vec<Option<Origin>> = (0..count).map(|_| None).collect();
    // Which line of the file as it is now each line of the version being examined became.
    let mut mapping: Vec<Option<usize>> = (0..count).map(Some).collect();
    let mut current = start.to_string();
    let mut current_content = content.to_vec();
    // The name the file had in the version being examined, which a rename moves.
    let mut path = path.to_string();
    if let Some(hash) = stored {
        let committed = repo::read_blob(ctx, root, &hash)?;
        mapping = carry_lines(
            &committed,
            &current_content,
            &mapping,
            UNCOMMITTED,
            &mut origins,
        );
        current_content = committed;
    }
    for _ in 0..MAX_BLAME_COMMITS {
        let commit = repo::load_commit(ctx, root, &current)?;
        let parent_tree = commit
            .parents
            .first()
            .and_then(|parent| repo::commit_tree(ctx, root, parent));
        let entry = parent_tree.as_ref().and_then(|parent| {
            parent.get(&path).cloned().or_else(|| {
                // The file was renamed here, so look for the same blob under its old name.
                let here = repo::commit_tree(ctx, root, &current)?;
                let moved = here.get(&path)?;
                let (was, before) = parent
                    .iter()
                    .find(|(name, entry)| entry.hash == moved.hash && !here.contains_key(*name))?;
                path = was.clone();
                Some(before.clone())
            })
        });
        let parent_content = entry.and_then(|entry| repo::read_blob(ctx, root, &entry.hash));
        let Some(parent_content) = parent_content else {
            // The file starts here, so every line still unclaimed is this commit's.
            for slot in mapping.into_iter().flatten() {
                origins[slot].get_or_insert(Origin {
                    commit: current.clone(),
                    boundary: true,
                });
            }
            break;
        };
        if !ctx.charge_cpu((current_content.len() + parent_content.len()) as u64) {
            return None;
        }
        let carried = carry_lines(
            &parent_content,
            &current_content,
            &mapping,
            &current,
            &mut origins,
        );
        // Nothing the parent holds reaches the file as it is now, so there is nothing left to ask.
        if carried.iter().all(Option::is_none) {
            break;
        }
        current = commit.parents.first().cloned()?;
        current_content = parent_content;
        mapping = carried;
    }
    // A walk that ran out of room still has to name something for the lines it did not reach.
    Some(
        origins
            .into_iter()
            .map(|origin| {
                origin.unwrap_or(Origin {
                    commit: current.clone(),
                    boundary: true,
                })
            })
            .collect(),
    )
}

/// How `git blame` was asked to print its lines.
struct BlameFormat<'a> {
    /// `-L`: the lines to show, out of the whole file.
    range: Option<&'a str>,
    /// `-s`: the hash and line number alone, with no author or date.
    suppress: bool,
    /// `-l`: the whole commit id rather than its abbreviation.
    long: bool,
}

fn emit_blame(
    ctx: &mut CommandContext<'_>,
    root: &str,
    lines: &[String],
    origins: &[Origin],
    format: &BlameFormat<'_>,
    io: &mut Io,
) -> i32 {
    let BlameFormat {
        range,
        suppress,
        long,
    } = *format;
    let (from, to) = match range {
        Some(range) => match parse_line_range(range, lines.len()) {
            Some(bounds) => bounds,
            None => return usage(io, &format!("unsupported line range: {range}")),
        },
        None => (1, lines.len()),
    };
    let width = lines.len().to_string().len();
    // Git lines the columns up by padding every author name to the widest one shown.
    let author_width = (from..=to)
        .filter_map(|number| origins.get(number - 1))
        .map(|origin| author_of(ctx, root, origin).len())
        .max()
        .unwrap_or(0);
    for number in from..=to {
        let Some(line) = lines.get(number - 1) else {
            break;
        };
        let origin = &origins[number - 1];
        // A boundary commit is marked with `^`, which takes the place of a hash digit.
        let digits = if long { origin.commit.len() } else { 8 };
        let name = if origin.boundary {
            format!("^{}", &origin.commit[..digits - 1])
        } else {
            origin.commit[..digits].to_string()
        };
        let described = if suppress {
            String::new()
        } else {
            let author = author_of(ctx, root, origin);
            let when = match repo::load_commit(ctx, root, &origin.commit) {
                Some(commit) => commit.timestamp,
                // Work that is not committed yet is dated now, as Git dates it.
                None => super::history::now_seconds(ctx),
            };
            format!(
                "({author:<author_width$} {} ",
                super::history::stamp(when, "%Y-%m-%d %H:%M:%S +0000")
            )
        };
        let close = if suppress { "" } else { ")" };
        io.print(&format!(
            "{name} {described}{number:>width$}{close}{}{line}\n",
            if suppress { ") " } else { " " }
        ));
    }
    0
}

/// The name shown for the commit a line came from.
fn author_of(ctx: &CommandContext<'_>, root: &str, origin: &Origin) -> String {
    match repo::load_commit(ctx, root, &origin.commit) {
        Some(commit) => commit.author_name,
        None => "Not Committed Yet".to_string(),
    }
}

/// Parse the `START,END` forms `-L` accepts, where either side may be left out.
fn parse_line_range(range: &str, total: usize) -> Option<(usize, usize)> {
    let (start, end) = range.split_once(',').unwrap_or((range, range));
    let start = if start.is_empty() {
        1
    } else {
        start.parse().ok()?
    };
    let end = if end.is_empty() {
        total
    } else {
        end.parse().ok()?
    };
    (start >= 1 && start <= end).then_some((start, end.min(total)))
}

// -- reflog --------------------------------------------------------------------------------------

/// The count in a bare `-5`, which Git accepts wherever it accepts `--max-count`.
pub(crate) fn count_option(name: &str) -> Option<usize> {
    let digits = name.strip_prefix('-')?;
    (!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

/// List where HEAD has been, which is what makes a bad reset recoverable.
pub(crate) fn git_reflog(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut limit = usize::MAX;
    let mut flags = Flags::new(args).valued("n");
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            // `show` is the only subcommand offered, and HEAD the only reference logged.
            Arg::Operand(value) if value == "show" || value == repo::HEAD => continue,
            Arg::Operand(value) => return usage(io, &format!("only HEAD is logged, not {value}")),
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "--oneline" | "--no-abbrev" => {}
            "-n" | "--max-count" => {
                let Some(value) = flags.value(attached).and_then(|value| value.parse().ok()) else {
                    return usage(io, "-n requires a count");
                };
                limit = value;
            }
            // `git reflog -5` is the count written on its own, as `git log -5` is.
            _ if count_option(&name).is_some() => limit = count_option(&name).unwrap_or(usize::MAX),
            _ => return usage(io, &format!("unsupported reflog option: {name}")),
        }
    }
    for (position, entry) in repo::read_head_log(ctx, &root)
        .iter()
        .enumerate()
        .take(limit)
    {
        io.print(&format!(
            "{} HEAD@{{{position}}}: {}\n",
            repo::short(&entry.after),
            entry.action
        ));
    }
    0
}
