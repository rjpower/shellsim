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
            repo::set_head_to_branch(ctx, &root, branch).map_or(1, |()| 0)
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
