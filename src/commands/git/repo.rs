//! Repository storage for the simulated Git porcelain.
//!
//! Everything lives below `.git` in the VFS; no host Git process or host filesystem is consulted.
//! The format is intentionally private and simple:
//!
//! * `.git/objects/<sha1>` holds blob content addressed by the SHA-1 of the bytes.
//! * `.git/commits/<id>.tree` holds a tree: a sorted map of repository-relative path to blob hash,
//!   with the path hex-encoded so that any byte sequence round-trips.
//! * `.git/commits/<id>.commit` holds commit metadata followed by a blank line and the message.
//! * `.git/index` holds the staged tree in the same format as a commit tree.
//! * `.git/HEAD`, `.git/refs/heads/*`, and `.git/refs/tags/*` hold symbolic and direct references.
//! * `.git/config` holds `key<TAB>value` lines for repository-local configuration.
//!
//! Real Git's zlib-compressed loose objects, packfiles, modes, and index extensions are
//! deliberately absent: nothing outside this module observes the layout.

use std::collections::{BTreeMap, BTreeSet};

use crate::commands::CommandContext;
use crate::interp::Interp;
use crate::vfs::{parent_of, NodeKind, Result as VfsResult};

pub(crate) const GIT_DIR: &str = ".git";
pub(crate) const HEAD: &str = "HEAD";
pub(crate) const INDEX: &str = "index";
pub(crate) const CONFIG: &str = "config";
pub(crate) const DEFAULT_BRANCH: &str = "main";

/// A staged or committed tree: repository-relative path to blob hash.
pub(crate) type Tree = BTreeMap<String, String>;

/// The most files one command will examine in a working tree.
const MAX_WORKING_FILES: u64 = 100_000;

// -- path helpers -------------------------------------------------------------------------------

pub(crate) fn path_join(root: &str, suffix: &str) -> String {
    if root == "/" {
        format!("/{suffix}")
    } else {
        format!("{root}/{suffix}")
    }
}

/// Absolute VFS path of `suffix` inside the repository's `.git` directory.
pub(crate) fn git_path(root: &str, suffix: &str) -> String {
    path_join(root, &format!("{GIT_DIR}/{suffix}"))
}

/// Find the nearest ancestor of the working directory that contains a `.git` directory.
pub(crate) fn find_repo_root(interp: &Interp) -> Option<String> {
    let mut current = interp.cwd.clone();
    loop {
        if interp.vfs.is_dir("/", &path_join(&current, GIT_DIR)) {
            return Some(current);
        }
        if current == "/" {
            return None;
        }
        current = parent_of(&current).unwrap_or_else(|| "/".to_string());
    }
}

pub(crate) fn within(root: &str, path: &str) -> bool {
    root == "/" || path == root || path.starts_with(&format!("{root}/"))
}

/// Convert an absolute VFS path into a repository-relative path, if it is inside the repository.
pub(crate) fn relative_path(root: &str, absolute: &str) -> Option<String> {
    if !within(root, absolute) {
        return None;
    }
    let relative = if root == "/" {
        absolute.trim_start_matches('/').to_string()
    } else {
        absolute
            .strip_prefix(root)?
            .trim_start_matches('/')
            .to_string()
    };
    (!relative.is_empty()).then_some(relative)
}

pub(crate) fn is_git_path(root: &str, path: &str) -> bool {
    let git = path_join(root, GIT_DIR);
    path == git || path.starts_with(&format!("{git}/"))
}

// -- hashing and encoding -----------------------------------------------------------------------

pub(crate) fn sha1(data: &[u8]) -> String {
    crate::hashes::sha1_hex(data)
}

/// Hash file content the way Git addresses a blob, so the ids printed in `index` lines and by
/// `git hash-object` match a real repository's.
pub(crate) fn blob_hash(data: &[u8]) -> String {
    let mut object = format!("blob {}\0", data.len()).into_bytes();
    object.extend_from_slice(data);
    sha1(&object)
}

/// The abbreviated hash Git prints in porcelain output.
pub(crate) fn short(hash: &str) -> &str {
    &hash[..hash.len().min(7)]
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

fn decode_hex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let high = (pair[0] as char).to_digit(16)? as u8;
        let low = (pair[1] as char).to_digit(16)? as u8;
        out.push((high << 4) | low);
    }
    Some(out)
}

// -- trees, blobs, and the index ----------------------------------------------------------------

pub(crate) fn serialize_tree(tree: &Tree) -> Vec<u8> {
    let mut out = Vec::new();
    for (path, hash) in tree {
        out.extend_from_slice(encode_hex(path.as_bytes()).as_bytes());
        out.push(b'\t');
        out.extend_from_slice(hash.as_bytes());
        out.push(b'\n');
    }
    out
}

pub(crate) fn parse_tree(bytes: &[u8]) -> Option<Tree> {
    let mut tree = Tree::new();
    for line in String::from_utf8_lossy(bytes).lines() {
        let (path_hex, hash) = line.split_once('\t')?;
        let path = String::from_utf8(decode_hex(path_hex)?).ok()?;
        if path.is_empty() || hash.len() != 40 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        tree.insert(path, hash.to_string());
    }
    Some(tree)
}

fn read_tree(interp: &Interp, path: &str) -> Option<Tree> {
    parse_tree(&interp.vfs.read("/", path).ok()?)
}

/// Write to the VFS with the simulated clock applied to the file's timestamps.
pub(crate) fn write_vfs(interp: &mut Interp, path: &str, bytes: &[u8]) -> VfsResult<()> {
    interp.sync_vfs_time();
    if let Some(parent) = parent_of(path) {
        interp.vfs.mkdir_all("/", &parent)?;
    }
    interp.vfs.write("/", path, bytes, 0o644)
}

pub(crate) fn load_index(interp: &Interp, root: &str) -> Option<Tree> {
    read_tree(interp, &git_path(root, INDEX))
}

pub(crate) fn store_index(ctx: &mut CommandContext<'_>, root: &str, index: &Tree) -> VfsResult<()> {
    write_vfs(ctx, &git_path(root, INDEX), &serialize_tree(index))
}

pub(crate) fn read_blob(interp: &Interp, root: &str, hash: &str) -> Option<Vec<u8>> {
    interp
        .vfs
        .read("/", &git_path(root, &format!("objects/{hash}")))
        .ok()
}

pub(crate) fn write_blob(
    ctx: &mut CommandContext<'_>,
    root: &str,
    data: &[u8],
) -> VfsResult<String> {
    let hash = blob_hash(data);
    write_vfs(ctx, &git_path(root, &format!("objects/{hash}")), data)?;
    Ok(hash)
}

/// Read a tracked file from the working tree.
pub(crate) fn read_work_file(interp: &Interp, root: &str, path: &str) -> Option<Vec<u8>> {
    interp.vfs.read("/", &path_join(root, path)).ok()
}

// -- commits ------------------------------------------------------------------------------------

/// One commit's metadata. The tree is stored separately and addressed by the commit id.
#[derive(Clone, Debug, Default)]
pub(crate) struct Commit {
    pub parents: Vec<String>,
    pub author_name: String,
    pub author_email: String,
    /// Author date in whole seconds since the Unix epoch, on the simulated clock.
    pub timestamp: i64,
    pub message: String,
}

impl Commit {
    /// The first line of the message, as Git's one-line formats print it.
    pub fn subject(&self) -> &str {
        self.message.lines().next().unwrap_or("")
    }

    fn serialize(&self) -> Vec<u8> {
        let mut text = String::new();
        for parent in &self.parents {
            text.push_str(&format!("parent {parent}\n"));
        }
        text.push_str(&format!(
            "author {} <{}> {} +0000\n\n",
            self.author_name, self.author_email, self.timestamp
        ));
        text.push_str(&self.message);
        text.into_bytes()
    }

    fn parse(bytes: &[u8]) -> Option<Self> {
        let text = String::from_utf8_lossy(bytes);
        let (header, message) = text.split_once("\n\n")?;
        let mut commit = Commit {
            message: message.to_string(),
            ..Commit::default()
        };
        for line in header.lines() {
            if let Some(parent) = line.strip_prefix("parent ") {
                commit.parents.push(parent.trim().to_string());
            } else if let Some(author) = line.strip_prefix("author ") {
                let (name, rest) = author.rsplit_once(" <")?;
                let (email, time) = rest.split_once("> ")?;
                commit.author_name = name.to_string();
                commit.author_email = email.to_string();
                commit.timestamp = time
                    .split_whitespace()
                    .next()
                    .and_then(|seconds| seconds.parse().ok())
                    .unwrap_or_default();
            }
        }
        Some(commit)
    }
}

fn commit_file(root: &str, id: &str) -> String {
    git_path(root, &format!("commits/{id}.commit"))
}

fn tree_file(root: &str, id: &str) -> String {
    git_path(root, &format!("commits/{id}.tree"))
}

pub(crate) fn load_commit(interp: &Interp, root: &str, id: &str) -> Option<Commit> {
    Commit::parse(&interp.vfs.read("/", &commit_file(root, id)).ok()?)
}

/// A stable identifier for a tree.
///
/// Real Git hashes a packed tree object; this subset hashes its own serialization, so the value
/// is self-consistent across `cat-file`, `ls-tree`, and `rev-parse` but differs from Git's.
pub(crate) fn tree_hash(tree: &Tree) -> String {
    let body = serialize_tree(tree);
    sha1(&[format!("tree {}\0", body.len()).as_bytes(), &body].concat())
}

pub(crate) fn commit_tree(interp: &Interp, root: &str, id: &str) -> Option<Tree> {
    read_tree(interp, &tree_file(root, id))
}

/// Store a commit and its tree, returning the new commit id.
///
/// The id is the SHA-1 of a Git-shaped header, so two commits differ whenever their tree,
/// parents, author, timestamp, or message differ.
pub(crate) fn store_commit(
    ctx: &mut CommandContext<'_>,
    root: &str,
    commit: &Commit,
    tree: &Tree,
) -> VfsResult<String> {
    let tree_bytes = serialize_tree(tree);
    let mut identity = format!("tree {}\n", sha1(&tree_bytes));
    identity.push_str(&String::from_utf8_lossy(&commit.serialize()));
    let id = sha1(identity.as_bytes());
    write_vfs(ctx, &tree_file(root, &id), &tree_bytes)?;
    write_vfs(ctx, &commit_file(root, &id), &commit.serialize())?;
    Ok(id)
}

/// Read an annotated tag's record, which reuses the commit header shape for tagger and message.
pub(crate) fn read_annotation(interp: &Interp, root: &str, name: &str) -> Option<Commit> {
    let path = git_path(root, &format!("tags/{name}.annotation"));
    Commit::parse(&interp.vfs.read("/", &path).ok()?)
}

pub(crate) fn write_annotation(
    ctx: &mut CommandContext<'_>,
    root: &str,
    name: &str,
    annotation: &Commit,
) -> VfsResult<()> {
    let path = git_path(root, &format!("tags/{name}.annotation"));
    write_vfs(ctx, &path, &annotation.serialize())
}

/// Walk first parents from `start`, most recent first, up to `limit` commits.
pub(crate) fn first_parent_history(
    interp: &Interp,
    root: &str,
    start: &str,
    limit: usize,
) -> Vec<(String, Commit)> {
    let mut out = Vec::new();
    let mut current = Some(start.to_string());
    let mut seen = BTreeSet::new();
    while let Some(id) = current {
        if out.len() >= limit || !seen.insert(id.clone()) {
            break;
        }
        let Some(commit) = load_commit(interp, root, &id) else {
            break;
        };
        current = commit.parents.first().cloned();
        out.push((id, commit));
    }
    out
}

/// Commits reachable from any of `starts`, newest first.
///
/// The order is topological: a commit is listed only after every commit that reaches it, with
/// ties broken by author timestamp and then by discovery order. Merge commits contribute both
/// parents, so nothing merged into the history disappears from `git log`.
pub(crate) fn reachable_history(
    interp: &Interp,
    root: &str,
    starts: &[String],
    limit: usize,
) -> Vec<(String, Commit)> {
    let mut commits: BTreeMap<String, (usize, Commit)> = BTreeMap::new();
    let mut queue: std::collections::VecDeque<String> = starts.iter().cloned().collect();
    let mut discovered = 0;
    while let Some(id) = queue.pop_front() {
        if commits.len() >= limit || commits.contains_key(&id) {
            continue;
        }
        let Some(commit) = load_commit(interp, root, &id) else {
            continue;
        };
        queue.extend(commit.parents.iter().cloned());
        commits.insert(id, (discovered, commit));
        discovered += 1;
    }
    // Count how many listed commits reach each commit directly.
    let mut pending: BTreeMap<&String, usize> = commits.keys().map(|id| (id, 0)).collect();
    for (_, commit) in commits.values() {
        for parent in &commit.parents {
            if let Some(count) = pending.get_mut(parent) {
                *count += 1;
            }
        }
    }
    let mut ready: std::collections::BinaryHeap<(i64, std::cmp::Reverse<usize>, String)> = commits
        .iter()
        .filter(|(id, _)| pending.get(id).copied() == Some(0))
        .map(|(id, (discovered, commit))| {
            (commit.timestamp, std::cmp::Reverse(*discovered), id.clone())
        })
        .collect();
    let mut out = Vec::with_capacity(commits.len());
    while let Some((_, _, id)) = ready.pop() {
        let Some((_, commit)) = commits.get(&id) else {
            continue;
        };
        for parent in commit.parents.clone() {
            let Some(count) = pending.get_mut(&parent) else {
                continue;
            };
            *count -= 1;
            if *count == 0 {
                if let Some((discovered, parent_commit)) = commits.get(&parent) {
                    ready.push((
                        parent_commit.timestamp,
                        std::cmp::Reverse(*discovered),
                        parent.clone(),
                    ));
                }
            }
        }
        out.push((id.clone(), commit.clone()));
    }
    out
}

/// Every ancestor of `commit`, including itself.
pub(crate) fn ancestors(interp: &Interp, root: &str, commit: &str) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut queue = vec![commit.to_string()];
    while let Some(id) = queue.pop() {
        if seen.len() >= 10_000 || !seen.insert(id.clone()) {
            continue;
        }
        if let Some(commit) = load_commit(interp, root, &id) {
            queue.extend(commit.parents);
        }
    }
    seen
}

/// The best common ancestor of two commits, preferring the one nearest `left`.
pub(crate) fn merge_base(interp: &Interp, root: &str, left: &str, right: &str) -> Option<String> {
    let right_ancestors = ancestors(interp, root, right);
    let mut queue = vec![left.to_string()];
    let mut seen = BTreeSet::new();
    while let Some(id) = queue.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        if right_ancestors.contains(&id) {
            return Some(id);
        }
        if let Some(commit) = load_commit(interp, root, &id) {
            queue.extend(commit.parents);
        }
    }
    None
}

// -- references ---------------------------------------------------------------------------------

/// Read the raw contents of `.git/HEAD`.
fn head_contents(interp: &Interp, root: &str) -> Option<String> {
    Some(
        String::from_utf8_lossy(&interp.vfs.read("/", &git_path(root, HEAD)).ok()?)
            .trim()
            .to_string(),
    )
}

/// The reference HEAD points at, or `None` when HEAD is detached.
pub(crate) fn head_reference(interp: &Interp, root: &str) -> Option<String> {
    head_contents(interp, root)?
        .strip_prefix("ref: ")
        .map(str::to_string)
}

pub(crate) fn head_commit(interp: &Interp, root: &str) -> Option<String> {
    let head = head_contents(interp, root)?;
    let commit = match head.strip_prefix("ref: ") {
        Some(reference) => read_reference(interp, root, reference)?,
        None => head,
    };
    (!commit.is_empty()).then_some(commit)
}

pub(crate) fn current_branch(interp: &Interp, root: &str) -> Option<String> {
    head_reference(interp, root)?
        .strip_prefix("refs/heads/")
        .map(str::to_string)
}

pub(crate) fn read_reference(interp: &Interp, root: &str, reference: &str) -> Option<String> {
    let value = String::from_utf8_lossy(&interp.vfs.read("/", &git_path(root, reference)).ok()?)
        .trim()
        .to_string();
    (!value.is_empty()).then_some(value)
}

pub(crate) fn write_reference(
    ctx: &mut CommandContext<'_>,
    root: &str,
    reference: &str,
    commit: &str,
) -> VfsResult<()> {
    write_vfs(
        ctx,
        &git_path(root, reference),
        format!("{commit}\n").as_bytes(),
    )
}

pub(crate) fn delete_reference(
    ctx: &mut CommandContext<'_>,
    root: &str,
    reference: &str,
) -> VfsResult<()> {
    ctx.vfs.remove_file("/", &git_path(root, reference))
}

/// Point HEAD at `commit`, following the current branch when HEAD is not detached.
pub(crate) fn update_head(ctx: &mut CommandContext<'_>, root: &str, commit: &str) -> VfsResult<()> {
    let destination = head_reference(ctx, root).unwrap_or_else(|| HEAD.to_string());
    write_reference(ctx, root, &destination, commit)
}

pub(crate) fn set_head_to_branch(
    ctx: &mut CommandContext<'_>,
    root: &str,
    branch: &str,
) -> VfsResult<()> {
    write_vfs(
        ctx,
        &git_path(root, HEAD),
        format!("ref: refs/heads/{branch}\n").as_bytes(),
    )
}

pub(crate) fn set_head_detached(
    ctx: &mut CommandContext<'_>,
    root: &str,
    commit: &str,
) -> VfsResult<()> {
    write_vfs(ctx, &git_path(root, HEAD), format!("{commit}\n").as_bytes())
}

/// Sorted names of the references below `.git/refs/<kind>`.
pub(crate) fn reference_names(interp: &Interp, root: &str, kind: &str) -> Vec<String> {
    let prefix = git_path(root, &format!("refs/{kind}"));
    let mut names = interp
        .vfs
        .walk(&prefix)
        .into_iter()
        .filter(|path| interp.vfs.is_file("/", path))
        .filter_map(|path| path.strip_prefix(&format!("{prefix}/")).map(str::to_string))
        .collect::<Vec<_>>();
    names.sort();
    names
}

pub(crate) fn branch_names(interp: &Interp, root: &str) -> Vec<String> {
    reference_names(interp, root, "heads")
}

pub(crate) fn head_tree(interp: &Interp, root: &str) -> Tree {
    head_commit(interp, root)
        .and_then(|commit| commit_tree(interp, root, &commit))
        .unwrap_or_default()
}

// -- revisions ----------------------------------------------------------------------------------

/// Resolve a revision expression to a commit id.
///
/// Supported forms are `HEAD`, `@`, a branch or tag name, a full ref path, a full or abbreviated
/// commit id, and the `~N` and `^N` ancestry suffixes. Ranges and reflog selectors are not
/// supported; callers split ranges before calling.
pub(crate) fn resolve_revision(interp: &Interp, root: &str, revision: &str) -> Option<String> {
    let boundary = revision.find(['~', '^']).unwrap_or(revision.len());
    let (base, suffix) = revision.split_at(boundary);
    let mut commit = resolve_base_revision(interp, root, base)?;
    let mut rest = suffix;
    while !rest.is_empty() {
        let (operator, remainder) = rest.split_at(1);
        let digits: String = remainder.chars().take_while(char::is_ascii_digit).collect();
        rest = &remainder[digits.len()..];
        match operator {
            "~" => {
                let steps: usize = if digits.is_empty() {
                    1
                } else {
                    digits.parse().ok()?
                };
                for _ in 0..steps {
                    commit = load_commit(interp, root, &commit)?.parents.first()?.clone();
                }
            }
            "^" => {
                let index: usize = if digits.is_empty() {
                    1
                } else {
                    digits.parse().ok()?
                };
                if index == 0 {
                    continue;
                }
                commit = load_commit(interp, root, &commit)?
                    .parents
                    .get(index - 1)?
                    .clone();
            }
            _ => return None,
        }
    }
    Some(commit)
}

fn resolve_base_revision(interp: &Interp, root: &str, revision: &str) -> Option<String> {
    if revision.is_empty() || revision == "HEAD" || revision == "@" {
        return head_commit(interp, root);
    }
    for candidate in [
        format!("refs/heads/{revision}"),
        format!("refs/tags/{revision}"),
        revision.to_string(),
    ] {
        if let Some(commit) = read_reference(interp, root, &candidate) {
            return Some(commit);
        }
    }
    if revision.len() < 4 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let mut matches = interp
        .vfs
        .walk(&git_path(root, "commits"))
        .into_iter()
        .filter_map(|path| {
            let id = crate::vfs::basename(&path)
                .strip_suffix(".commit")?
                .to_string();
            id.starts_with(revision).then_some(id)
        });
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

// -- configuration ------------------------------------------------------------------------------

/// Read one configuration file in Git's INI format.
///
/// Keys are normalized to the lowercase dotted form Git uses on the command line, so
/// `[remote "origin"] url = X` becomes `remote.origin.url`. Includes, conditional includes,
/// multi-valued keys, and value continuations are out of scope.
pub(crate) fn parse_config(text: &str) -> BTreeMap<String, String> {
    let mut config = BTreeMap::new();
    let mut prefix = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(header) = line
            .strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'))
        {
            prefix = match header.split_once(char::is_whitespace) {
                Some((section, subsection)) => format!(
                    "{}.{}.",
                    section.to_ascii_lowercase(),
                    subsection.trim().trim_matches('"')
                ),
                None => format!("{}.", header.to_ascii_lowercase()),
            };
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            // A bare key is Git's shorthand for a true boolean.
            config.insert(
                format!("{prefix}{}", line.to_ascii_lowercase()),
                "true".to_string(),
            );
            continue;
        };
        if config.len() >= 256 {
            break;
        }
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|rest| rest.strip_suffix('"'))
            .unwrap_or(value);
        config.insert(
            format!("{prefix}{}", key.trim().to_ascii_lowercase()),
            value.to_string(),
        );
    }
    config
}

/// Render configuration back to Git's INI format.
pub(crate) fn serialize_config(config: &BTreeMap<String, String>) -> Vec<u8> {
    let mut out = String::new();
    let mut current = String::new();
    for (key, value) in config {
        let (heading, name) = split_config_key(key);
        if heading != current {
            out.push_str(&format!("[{heading}]\n"));
            current = heading;
        }
        out.push_str(&format!("\t{name} = {value}\n"));
    }
    out.into_bytes()
}

/// Split `remote.origin.url` into the `remote "origin"` heading and the `url` name.
fn split_config_key(key: &str) -> (String, &str) {
    let parts: Vec<&str> = key.splitn(3, '.').collect();
    match parts.as_slice() {
        [section, subsection, name] => (format!("{section} \"{subsection}\""), *name),
        [section, name] => ((*section).to_string(), *name),
        _ => (key.to_string(), key),
    }
}

/// The path of the per-user configuration file inside the simulated filesystem.
pub(crate) fn global_config_path(interp: &Interp) -> String {
    let home = interp.get_var("HOME").unwrap_or_else(|| "/root".into());
    path_join(&home, ".gitconfig")
}

fn read_config_file(ctx: &mut CommandContext<'_>, path: &str) -> BTreeMap<String, String> {
    ctx.fs_read_limited("/", path, 256 * 1024)
        .ok()
        .map(|bytes| parse_config(&String::from_utf8_lossy(&bytes)))
        .unwrap_or_default()
}

/// Repository-local configuration.
pub(crate) fn load_config(ctx: &mut CommandContext<'_>, root: &str) -> BTreeMap<String, String> {
    read_config_file(ctx, &git_path(root, CONFIG))
}

/// Per-user configuration, which the repository's own settings override.
pub(crate) fn load_global_config(ctx: &mut CommandContext<'_>) -> BTreeMap<String, String> {
    let path = global_config_path(ctx);
    read_config_file(ctx, &path)
}

pub(crate) fn valid_config_key(key: &str) -> bool {
    key.contains('.')
        && !key.starts_with('.')
        && !key.ends_with('.')
        && !key.contains("..")
        && key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
}

pub(crate) fn write_config(
    ctx: &mut CommandContext<'_>,
    path: &str,
    config: &BTreeMap<String, String>,
) -> VfsResult<()> {
    write_vfs(ctx, path, &serialize_config(config))
}

// -- working tree -------------------------------------------------------------------------------

/// A hashed snapshot of every file in the working tree, with the memory it reserved.
pub(crate) struct WorkingTree {
    pub files: Tree,
    reserved_memory: u64,
}

impl WorkingTree {
    /// Release the memory the snapshot reserved. Call once the snapshot is no longer needed.
    pub fn release(self, ctx: &mut CommandContext<'_>) -> Tree {
        ctx.resources.release_memory(self.reserved_memory);
        self.files
    }
}

/// Hash every file in the working tree, metering the work against the caller's limits.
pub(crate) fn collect_working_tree(
    ctx: &mut CommandContext<'_>,
    root: &str,
) -> Result<WorkingTree, i32> {
    let mut file_count = 0_u64;
    let mut path_bytes = 0_u64;
    let mut content_bytes = 0_u64;
    for (path, node) in ctx.vfs.all_paths() {
        if is_git_path(root, path) || !within(root, path) {
            continue;
        }
        if let NodeKind::File(data) = &node.kind {
            let Some(relative) = relative_path(root, path) else {
                continue;
            };
            let Some(next_file_count) = file_count.checked_add(1) else {
                return Err(137);
            };
            let Some(next_path_bytes) = path_bytes.checked_add(relative.len() as u64) else {
                return Err(137);
            };
            let Some(next_content_bytes) = content_bytes.checked_add(data.len() as u64) else {
                return Err(137);
            };
            file_count = next_file_count;
            path_bytes = next_path_bytes;
            content_bytes = next_content_bytes;
        }
    }
    if file_count > MAX_WORKING_FILES {
        return Err(resource_error(ctx));
    }
    let reserved_memory = path_bytes.saturating_add(file_count.saturating_mul(64));
    if !ctx.reserve_memory(reserved_memory) {
        return Err(resource_error(ctx));
    }
    if !ctx.charge_cpu(path_bytes.saturating_add(content_bytes)) {
        ctx.resources.release_memory(reserved_memory);
        return Err(resource_error(ctx));
    }
    let mut files = Tree::new();
    for (path, node) in ctx.vfs.all_paths() {
        if is_git_path(root, path) || !within(root, path) {
            continue;
        }
        if let NodeKind::File(data) = &node.kind {
            if let Some(relative) = relative_path(root, path) {
                files.insert(relative, blob_hash(data));
            }
        }
    }
    Ok(WorkingTree {
        files,
        reserved_memory,
    })
}

/// Hash one working-tree file, charging the read against the caller's limits.
pub(crate) fn metered_file_hash(
    ctx: &mut CommandContext<'_>,
    path: &str,
) -> Result<Option<String>, i32> {
    if !ctx.vfs.is_file("/", path) {
        return Ok(None);
    }
    let length = ctx.fs_file_len("/", path).map_err(|_| 1)?;
    if !ctx.reserve_memory(length as u64) {
        return Err(resource_error(ctx));
    }
    if !ctx.charge_cpu(length as u64) {
        ctx.resources.release_memory(length as u64);
        return Err(resource_error(ctx));
    }
    let data = ctx.fs_read_limited("/", path, length);
    ctx.resources.release_memory(length as u64);
    data.map(|bytes| Some(blob_hash(&bytes))).map_err(|_| 1)
}

/// Exit status to report when a resource limit stopped the command.
pub(crate) fn resource_error(ctx: &CommandContext<'_>) -> i32 {
    ctx.resources
        .stop_reason()
        .map_or(137, |reason| reason.exit_status())
}

/// Replace the working-tree files named by `old` with the contents named by `new`.
///
/// The whole update is applied to a VFS copy so that a failure part-way through leaves the
/// working tree untouched.
/// Move the working tree from `old` to `new`, rewriting every path `new` names.
///
/// This discards uncommitted edits, which is what `git restore` and `git reset --hard` are for.
/// Remove the directories a deletion left empty, stopping at the repository root.
///
/// Git tracks files rather than directories, so removing the last file in a directory removes the
/// directory too.
pub(crate) fn prune_empty_parents(ctx: &mut CommandContext<'_>, root: &str, absolute: &str) {
    let mut current = parent_of(absolute);
    while let Some(directory) = current {
        if directory == root || !within(root, &directory) || is_git_path(root, &directory) {
            return;
        }
        match ctx.vfs.list_dir("/", &directory) {
            Ok(entries) if entries.is_empty() => {}
            _ => return,
        }
        if ctx.vfs.remove_all("/", &directory).is_err() {
            return;
        }
        current = parent_of(&directory);
    }
}

pub(crate) fn replace_work_tree(
    ctx: &mut CommandContext<'_>,
    root: &str,
    old: &Tree,
    new: &Tree,
) -> Result<(), String> {
    write_work_tree(ctx, root, old, new, true)
}

/// Move the working tree from `old` to `new`, leaving unchanged paths alone.
///
/// Checkout, merge, and stash reapplication go through here so that an uncommitted edit to a file
/// the move does not touch survives, as it does in Git.
pub(crate) fn update_work_tree(
    ctx: &mut CommandContext<'_>,
    root: &str,
    old: &Tree,
    new: &Tree,
) -> Result<(), String> {
    write_work_tree(ctx, root, old, new, false)
}

fn write_work_tree(
    ctx: &mut CommandContext<'_>,
    root: &str,
    old: &Tree,
    new: &Tree,
    force: bool,
) -> Result<(), String> {
    let reserved = ctx.vfs.disk_used().saturating_add(4 * 1024);
    if !ctx.reserve_memory(reserved) {
        return Err("memory limit exceeded".to_string());
    }
    let before = ctx.vfs.clone();
    let result = (|| {
        for path in old.keys().filter(|path| !new.contains_key(*path)) {
            let absolute = path_join(root, path);
            if ctx.vfs.is_file("/", &absolute) {
                ctx.vfs
                    .remove_file("/", &absolute)
                    .map_err(|error| error.to_string())?;
                prune_empty_parents(ctx, root, &absolute);
            }
        }
        for (path, hash) in new {
            // A path the move does not change keeps whatever the working tree holds, but a
            // missing file is still restored.
            if !force && old.get(path) == Some(hash) && ctx.vfs.is_file("/", &path_join(root, path))
            {
                continue;
            }
            let data = read_blob(ctx, root, hash)
                .ok_or_else(|| format!("missing blob {hash} for {path}"))?;
            write_vfs(ctx, &path_join(root, path), &data).map_err(|error| error.to_string())?;
        }
        Ok(())
    })();
    if result.is_err() {
        ctx.vfs = before;
    }
    ctx.resources.release_memory(reserved);
    result
}
