//! Inspection commands that read repository state without changing it.
//!
//! These are the plumbing and reporting commands agents reach for when orienting in a repository:
//! object inspection, tree listing, ignore checking, ancestry queries, and content search. They
//! share the storage layer in [`super::repo`] and never touch the host.

use std::collections::BTreeMap;

use crate::commands::{CommandContext, Io};
use crate::vfs::resolve_against;

use super::ignore;
use super::repo;
use super::{repo_error, usage};

/// The most files `git grep` will search in one invocation.
const MAX_GREP_FILES: usize = 10_000;

pub(crate) fn git_cat_file(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut mode = None;
    let mut object = None;
    for argument in args {
        match argument.as_str() {
            "-p" | "-t" | "-s" | "-e" => mode = Some(argument.clone()),
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported cat-file option: {value}"))
            }
            value => object = Some(value.to_string()),
        }
    }
    let (Some(mode), Some(object)) = (mode, object) else {
        return usage(io, "usage: git cat-file (-p|-t|-s|-e) OBJECT");
    };
    // An object is either a blob hash or a revision, optionally with a `:PATH` suffix.
    let blob = match object.split_once(':') {
        Some((revision, path)) => repo::resolve_revision(ctx, &root, revision)
            .and_then(|commit| repo::commit_tree(ctx, &root, &commit))
            .and_then(|tree| tree.get(path).cloned()),
        None => Some(object.clone()),
    };
    if let Some(data) = blob
        .as_deref()
        .and_then(|hash| repo::read_blob(ctx, &root, hash))
    {
        return match mode.as_str() {
            "-t" => {
                io.out.extend_from_slice(b"blob\n");
                0
            }
            "-s" => {
                io.out
                    .extend_from_slice(format!("{}\n", data.len()).as_bytes());
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
        io.err
            .extend_from_slice(format!("fatal: Not a valid object name {object}\n").as_bytes());
        return 128;
    };
    let Some(commit) = repo::load_commit(ctx, &root, &id) else {
        io.err
            .extend_from_slice(format!("fatal: Not a valid object name {object}\n").as_bytes());
        return 128;
    };
    match mode.as_str() {
        "-t" => io.out.extend_from_slice(b"commit\n"),
        "-e" => {}
        "-s" => io
            .out
            .extend_from_slice(format!("{}\n", commit.message.len()).as_bytes()),
        _ => {
            for parent in &commit.parents {
                io.out
                    .extend_from_slice(format!("parent {parent}\n").as_bytes());
            }
            io.out.extend_from_slice(
                format!(
                    "author {} <{}> {} +0000\n\n{}\n",
                    commit.author_name, commit.author_email, commit.timestamp, commit.message
                )
                .as_bytes(),
            );
        }
    }
    0
}

pub(crate) fn git_hash_object(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut write = false;
    let mut stdin = false;
    let mut files = Vec::new();
    for argument in args {
        match argument.as_str() {
            "-w" => write = true,
            "--stdin" => stdin = true,
            "-t" => {}
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported hash-object option: {value}"))
            }
            value => files.push(value.to_string()),
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
                io.err.extend_from_slice(
                    format!("fatal: could not open '{file}' for reading\n").as_bytes(),
                );
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
            if repo::write_blob(ctx, root, &data).is_err() {
                return 1;
            }
        }
        io.out.extend_from_slice(format!("{hash}\n").as_bytes());
    }
    0
}

pub(crate) fn git_ls_tree(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut name_only = false;
    let mut recursive = false;
    let mut operands = Vec::new();
    for argument in super::expand_clusters(args, "rd") {
        match argument.as_str() {
            "--name-only" | "--name-status" => name_only = true,
            "-r" => recursive = true,
            "-d" | "--full-name" => {}
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported ls-tree option: {value}"))
            }
            value => operands.push(value.to_string()),
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
    let mut listed: BTreeMap<String, Option<String>> = BTreeMap::new();
    for (path, hash) in &tree {
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
                listed.insert(path.clone(), Some(hash.clone()));
            }
        }
    }
    for (name, hash) in listed {
        if name_only {
            io.out.extend_from_slice(format!("{name}\n").as_bytes());
            continue;
        }
        let line = match hash {
            Some(hash) => format!("100644 blob {hash}\t{name}\n"),
            None => format!("040000 tree {}\t{name}\n", "0".repeat(40)),
        };
        io.out.extend_from_slice(line.as_bytes());
    }
    0
}

pub(crate) fn git_check_ignore(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut paths = Vec::new();
    for argument in args {
        match argument.as_str() {
            "-v" | "--verbose" | "-q" | "--quiet" | "--no-index" => {}
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported check-ignore option: {value}"))
            }
            value => paths.push(value.to_string()),
        }
    }
    if paths.is_empty() {
        return usage(io, "usage: git check-ignore PATH...");
    }
    let rules = ignore::load(ctx, &root);
    let cwd = ctx.cwd.clone();
    let mut any = false;
    for path in &paths {
        let relative = super::pathspec(&cwd, &root, path);
        if rules.is_ignored(&relative) {
            any = true;
            io.out.extend_from_slice(format!("{path}\n").as_bytes());
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
    for argument in args {
        match argument.as_str() {
            "--is-ancestor" => is_ancestor = true,
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported merge-base option: {value}"))
            }
            value => revisions.push(value.to_string()),
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
            io.out.extend_from_slice(format!("{base}\n").as_bytes());
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
    for argument in args {
        match argument.as_str() {
            "--tags" => lightweight = true,
            "--abbrev=0" | "--long" => {}
            "--always" => always = true,
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported describe option: {value}"))
            }
            value => revision = value.to_string(),
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
        io.out
            .extend_from_slice(format!("{described}\n").as_bytes());
        return 0;
    }
    if always {
        io.out
            .extend_from_slice(format!("{}\n", repo::short(&start)).as_bytes());
        return 0;
    }
    io.err
        .extend_from_slice(b"fatal: No annotated tags can describe this commit; try --tags.\n");
    128
}

pub(crate) fn git_shortlog(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut summary = false;
    let mut numbered = false;
    for argument in super::expand_clusters(args, "sne") {
        match argument.as_str() {
            "-s" | "--summary" => summary = true,
            "-n" | "--numbered" => numbered = true,
            "-e" | "--email" => {}
            value => return usage(io, &format!("unsupported shortlog option: {value}")),
        }
    }
    let Some(head) = repo::head_commit(ctx, &root) else {
        return 0;
    };
    let mut counts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (_, commit) in repo::first_parent_history(ctx, &root, &head, 10_000) {
        counts
            .entry(commit.author_name.clone())
            .or_default()
            .push(commit.subject().to_string());
    }
    let mut authors: Vec<(String, Vec<String>)> = counts.into_iter().collect();
    if numbered {
        authors.sort_by(|left, right| right.1.len().cmp(&left.1.len()).then(left.0.cmp(&right.0)));
    }
    for (author, subjects) in authors {
        if summary {
            io.out
                .extend_from_slice(format!("{:>6}\t{author}\n", subjects.len()).as_bytes());
            continue;
        }
        io.out
            .extend_from_slice(format!("{author} ({}):\n", subjects.len()).as_bytes());
        for subject in subjects {
            io.out
                .extend_from_slice(format!("      {subject}\n").as_bytes());
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
    let mut pattern = None;
    let mut paths: Vec<String> = Vec::new();
    let mut operands_only = false;
    let mut index = 0;
    let args = super::expand_clusters(args, "niIlcEFv");
    while index < args.len() {
        let argument = args[index].as_str();
        if operands_only {
            paths.push(argument.to_string());
            index += 1;
            continue;
        }
        match argument {
            "--" => operands_only = true,
            "-n" | "--line-number" => line_numbers = true,
            "-i" | "--ignore-case" => ignore_case = true,
            "-l" | "--files-with-matches" | "--name-only" => names_only = true,
            "-c" | "--count" => count_only = true,
            "-F" | "--fixed-strings" => fixed = true,
            "-v" | "--invert-match" => invert = true,
            "-E" | "--extended-regexp" | "-I" | "--no-color" | "--cached" => {}
            "-e" => {
                index += 1;
                pattern = args.get(index).cloned();
            }
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported grep option: {value}"))
            }
            value if pattern.is_none() => pattern = Some(value.to_string()),
            value => paths.push(value.to_string()),
        }
        index += 1;
    }
    let Some(pattern) = pattern else {
        return usage(io, "usage: git grep [-n] [-i] [-l] PATTERN [-- PATH...]");
    };
    let expression = if fixed {
        regex::escape(&pattern)
    } else {
        pattern.clone()
    };
    let Ok(regex) = regex::RegexBuilder::new(&expression)
        .case_insensitive(ignore_case)
        .build()
    else {
        io.err
            .extend_from_slice(format!("fatal: invalid pattern: {pattern}\n").as_bytes());
        return 128;
    };
    let cwd = ctx.cwd.clone();
    let paths: Vec<String> = paths
        .iter()
        .map(|path| super::pathspec(&cwd, &root, path))
        .collect();
    let index_tree = repo::load_index(ctx, &root).unwrap_or_default();
    let mut found = false;
    for (searched, path) in index_tree.keys().enumerate() {
        if searched >= MAX_GREP_FILES {
            break;
        }
        if !super::compare::selected(&paths, path) {
            continue;
        }
        let Some(data) = repo::read_work_file(ctx, &root, path) else {
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
                format!("{path}:{}:", number + 1)
            } else {
                format!("{path}:")
            };
            io.out
                .extend_from_slice(format!("{location}{line}\n").as_bytes());
        }
        if names_only && matches != 0 {
            io.out.extend_from_slice(format!("{path}\n").as_bytes());
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
                io.out
                    .extend_from_slice(format!("{path}:{total}\n").as_bytes());
            }
        }
    }
    i32::from(!found)
}
