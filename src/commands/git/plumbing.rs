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
            .and_then(|tree| tree.get(path).map(|entry| entry.hash.clone())),
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
        "-t" if annotated => io.out.extend_from_slice(b"tag\n"),
        "-t" => io.out.extend_from_slice(b"commit\n"),
        "-e" => {}
        "-s" => io
            .out
            .extend_from_slice(format!("{}\n", body.len()).as_bytes()),
        _ => io.out.extend_from_slice(body.as_bytes()),
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
    let mut directories_only = false;
    let mut operands = Vec::new();
    for argument in super::expand_clusters(args, "rd") {
        match argument.as_str() {
            "--name-only" | "--name-status" => name_only = true,
            "-r" => recursive = true,
            "-d" => directories_only = true,
            "--full-name" => {}
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
            io.out.extend_from_slice(format!("{name}\n").as_bytes());
            continue;
        }
        let line = match entry {
            Some(entry) => format!("{} blob {}\t{name}\n", entry.mode(), entry.hash),
            None => format!("040000 tree {}\t{name}\n", subtree_hash(&tree, &name)),
        };
        io.out.extend_from_slice(line.as_bytes());
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
    let args = super::expand_clusters(args, "vqn");
    for argument in &args {
        match argument.as_str() {
            "-v" | "--verbose" => verbose = true,
            "-q" | "--quiet" => quiet = true,
            "-n" | "--non-matching" => non_matching = true,
            "--stdin" => from_stdin = true,
            "--no-index" => {}
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported check-ignore option: {value}"))
            }
            value => paths.push(value.to_string()),
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
                io.out
                    .extend_from_slice(format!("{rule}\t{path}\n").as_bytes());
            }
            (Some(_), _) => io.out.extend_from_slice(format!("{path}\n").as_bytes()),
            // `-n` reports the paths no pattern covers, with empty source fields.
            (None, true) => io.out.extend_from_slice(format!("::\t{path}\n").as_bytes()),
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
    io.err.extend_from_slice(
        format!("fatal: No annotated tags can describe '{start}'.\n").as_bytes(),
    );
    if !repo::reference_names(ctx, &root, "tags").is_empty() {
        io.err
            .extend_from_slice(b"However, there were unannotated tags: try --tags.\n");
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
    for argument in super::expand_clusters(args, "sne") {
        match argument.as_str() {
            "-s" | "--summary" => summary = true,
            "-n" | "--numbered" => numbered = true,
            "-e" | "--email" => with_email = true,
            "--no-merges" => {}
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported shortlog option: {value}"))
            }
            value => revisions.push(value.to_string()),
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
    let mut word = false;
    let mut without_match = false;
    let mut pattern = None;
    let mut paths: Vec<String> = Vec::new();
    let mut operands_only = false;
    let mut before_separator = usize::MAX;
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
            "--" => {
                before_separator = paths.len();
                operands_only = true;
            }
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
        io.err
            .extend_from_slice(format!("fatal: invalid pattern: {pattern}\n").as_bytes());
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
            io.out
                .extend_from_slice(format!("{location}{line}\n").as_bytes());
        }
        if names_only && (matches != 0) != without_match {
            found = true;
            io.out
                .extend_from_slice(format!("{displayed}\n").as_bytes());
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
                    .extend_from_slice(format!("{displayed}:{total}\n").as_bytes());
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
    for argument in args {
        match argument.as_str() {
            "--heads" => heads_only = true,
            "--tags" => tags_only = true,
            "--verify" => verify = true,
            "-q" | "--quiet" | "--hash" | "-d" | "--dereference" => {}
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported show-ref option: {value}"))
            }
            value => patterns.push(value.to_string()),
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
        io.out
            .extend_from_slice(format!("{commit} {name}\n").as_bytes());
    }
    i32::from(!matched)
}

pub(crate) fn git_symbolic_ref(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut short = false;
    let mut operands: Vec<String> = Vec::new();
    for argument in args {
        match argument.as_str() {
            "--short" => short = true,
            "-q" | "--quiet" => {}
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported symbolic-ref option: {value}"))
            }
            value => operands.push(value.to_string()),
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
                io.out.extend_from_slice(format!("{rendered}\n").as_bytes());
                0
            }
            None => {
                io.err
                    .extend_from_slice(b"fatal: ref HEAD is not a symbolic ref\n");
                1
            }
        },
        [name, target] if name == repo::HEAD => {
            let Some(branch) = target.strip_prefix("refs/heads/") else {
                return usage(io, "only refs/heads/* can be pointed at by HEAD");
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
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--format" => {
                index += 1;
                format = args.get(index).cloned();
            }
            value if value.starts_with("--format=") => {
                format = Some(value["--format=".len()..].to_string());
            }
            value if value.starts_with("--count=") || value.starts_with("--sort=") => {}
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported for-each-ref option: {value}"))
            }
            value => prefixes.push(value.trim_end_matches('*').to_string()),
        }
        index += 1;
    }
    let format = format.unwrap_or_else(|| "%(objectname) %(objecttype)\t%(refname)".to_string());
    for (name, commit) in all_references(ctx, &root) {
        if !prefixes.is_empty() && !prefixes.iter().any(|prefix| name.starts_with(prefix)) {
            continue;
        }
        let Some(line) = expand_ref_format(&format, &name, &commit) else {
            return usage(io, &format!("unsupported for-each-ref format: {format}"));
        };
        io.out.extend_from_slice(format!("{line}\n").as_bytes());
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
    let mut range: Option<String> = None;
    let mut operands: Vec<String> = Vec::new();
    let mut options = true;
    let expanded = super::expand_clusters(args, "slwe");
    let mut index = 0;
    while index < expanded.len() {
        let argument = expanded[index].as_str();
        match argument {
            "--" if options => options = false,
            "-s" if options => suppress = true,
            // Blame here has no similarity detection or whitespace modes to turn on.
            "-l" | "-w" | "-e" | "--show-email" | "--root" if options => {}
            "-L" if options => {
                index += 1;
                let Some(value) = expanded.get(index) else {
                    return usage(io, "-L requires a line range");
                };
                range = Some(value.clone());
            }
            value if options && value.starts_with("-L") && value.len() > 2 => {
                range = Some(value[2..].to_string());
            }
            value if options && value.starts_with('-') => {
                return usage(io, &format!("unsupported blame option: {value}"))
            }
            value => operands.push(value.to_string()),
        }
        index += 1;
    }
    // The file is the operand that names one; anything else is the revision to start from.
    let Some(position) = operands
        .iter()
        .rposition(|operand| super::names_a_path(ctx, &root, operand))
    else {
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
        io.err
            .extend_from_slice(format!("fatal: no such path {file} in {revision}\n").as_bytes());
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
    emit_blame(ctx, &root, &lines, &origins, range.as_deref(), suppress, io)
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
        let parent_content = commit
            .parents
            .first()
            .and_then(|parent| repo::commit_tree(ctx, root, parent))
            .and_then(|tree| tree.get(path).cloned())
            .and_then(|entry| repo::read_blob(ctx, root, &entry.hash));
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

fn emit_blame(
    ctx: &mut CommandContext<'_>,
    root: &str,
    lines: &[String],
    origins: &[Origin],
    range: Option<&str>,
    suppress: bool,
    io: &mut Io,
) -> i32 {
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
        let name = if origin.boundary {
            format!("^{}", &origin.commit[..7])
        } else {
            origin.commit[..8].to_string()
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
        io.out.extend_from_slice(
            format!(
                "{name} {described}{number:>width$}{close}{}{line}\n",
                if suppress { ") " } else { " " }
            )
            .as_bytes(),
        );
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

/// List where HEAD has been, which is what makes a bad reset recoverable.
pub(crate) fn git_reflog(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut limit = usize::MAX;
    let mut index = 0;
    while index < args.len() {
        let argument = args[index].as_str();
        match argument {
            // `show` is the only subcommand offered, and HEAD the only reference logged.
            "show" | "HEAD" | "--oneline" | "--no-abbrev" | "--" => {}
            "-n" | "--max-count" => {
                index += 1;
                let Some(value) = args.get(index).and_then(|value| value.parse().ok()) else {
                    return usage(io, "-n requires a count");
                };
                limit = value;
            }
            value if value.starts_with("--max-count=") => {
                let Ok(value) = value["--max-count=".len()..].parse() else {
                    return usage(io, "--max-count requires a count");
                };
                limit = value;
            }
            value if value.starts_with('-') && value[1..].chars().all(|c| c.is_ascii_digit()) => {
                limit = value[1..].parse().unwrap_or(usize::MAX);
            }
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported reflog option: {value}"))
            }
            value => return usage(io, &format!("only HEAD is logged, not {value}")),
        }
        index += 1;
    }
    for (position, entry) in repo::read_head_log(ctx, &root)
        .iter()
        .enumerate()
        .take(limit)
    {
        io.out.extend_from_slice(
            format!(
                "{} HEAD@{{{position}}}: {}\n",
                repo::short(&entry.after),
                entry.action
            )
            .as_bytes(),
        );
    }
    0
}
