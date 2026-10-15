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
//!
//! Every function here takes `&mut dyn System`, the PID-scoped virtual-kernel interface, rather
//! than `Interp` or `CommandContext`: git must not gain ambient access to the owning process.

use std::collections::{BTreeMap, BTreeSet};

use crate::syscalls::{FileChange, FileKind, System};
use crate::vfs::{parent_of, Result as VfsResult, VfsError};

use super::Globals;

pub(crate) const GIT_DIR: &str = ".git";
pub(crate) const HEAD: &str = "HEAD";
pub(crate) const INDEX: &str = "index";
pub(crate) const CONFIG: &str = "config";
pub(crate) const DEFAULT_BRANCH: &str = "main";

/// One tracked file: the blob it holds and the only permission bit Git records.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Entry {
    pub hash: String,
    pub executable: bool,
    /// A symbolic link, whose blob holds the target rather than file content.
    pub symlink: bool,
}

impl Entry {
    /// The mode Git prints for this entry.
    pub fn mode(&self) -> &'static str {
        match (self.symlink, self.executable) {
            (true, _) => "120000",
            (_, true) => "100755",
            _ => "100644",
        }
    }
}

/// Whether a working-tree permission set makes the file executable, as Git decides it.
pub(crate) fn is_executable(mode: u32) -> bool {
    mode & 0o111 != 0
}

/// The permissions a checkout gives a file of this mode.
pub(crate) fn work_mode(executable: bool) -> u32 {
    if executable {
        0o755
    } else {
        0o644
    }
}

/// A staged or committed tree: repository-relative path to the file recorded there.
pub(crate) type Tree = BTreeMap<String, Entry>;

/// Configuration: dotted lowercase key to every value recorded for it, in file order.
pub(crate) type Config = BTreeMap<String, Vec<String>>;

/// The value a plain `git config <key>` read returns, which is the last one recorded.
pub(crate) fn config_value<'a>(config: &'a Config, key: &str) -> Option<&'a str> {
    config.get(key)?.last().map(String::as_str)
}

/// The most files one command will examine in a working tree.
const MAX_WORKING_FILES: u64 = 100_000;

// -- VFS helpers ----------------------------------------------------------------------------------
//
// `System` exposes only `metadata`, not Git's `is_dir`/`is_file`/`exists` convenience queries, so
// this module rebuilds the ones it needs on top of it.

fn node_kind(system: &mut dyn System, base: &str, path: &str, follow: bool) -> Option<FileKind> {
    system
        .metadata(base, path, follow)
        .ok()
        .map(|info| info.kind)
}

pub(crate) fn is_dir(system: &mut dyn System, base: &str, path: &str) -> bool {
    matches!(
        node_kind(system, base, path, true),
        Some(FileKind::Directory)
    )
}

pub(crate) fn is_file(system: &mut dyn System, base: &str, path: &str) -> bool {
    matches!(node_kind(system, base, path, true), Some(FileKind::File))
}

pub(crate) fn exists(system: &mut dyn System, base: &str, path: &str) -> bool {
    system.metadata(base, path, true).is_ok()
}

/// Remove a file, or a directory and everything below it.
///
/// `System` exposes `unlink` for one file and `rmdir` for one empty directory, so a whole subtree
/// is removed deepest-first: `walk` returns each path before its children, so reversing it visits
/// every descendant before the directory that holds it, which is what lets each `rmdir` see an
/// already-empty directory.
pub(crate) fn remove_tree(system: &mut dyn System, base: &str, path: &str) -> bool {
    let Ok(mut paths) = system.walk(base, path) else {
        return false;
    };
    paths.reverse();
    for entry in paths {
        let result = if is_dir(system, base, &entry) {
            system.rmdir(base, &entry)
        } else {
            system.unlink(base, &entry)
        };
        if result.is_err() {
            return false;
        }
    }
    true
}

/// Read a whole file, bounded by the caller's memory limit rather than an arbitrary constant.
///
/// Git objects can be as large as any file a user committed, so there is no small fixed bound
/// that is both safe and correct; the process memory limit is the same bound every other read of
/// an unbounded file in this codebase uses.
pub(crate) fn read_all(system: &mut dyn System, path: &str) -> Option<Vec<u8>> {
    let maximum = usize::try_from(system.limits().memory).unwrap_or(usize::MAX);
    system.read_file_limited("/", path, maximum).ok()
}

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
pub(crate) fn find_repo_root(system: &mut dyn System) -> Option<String> {
    let mut current = system.cwd().to_string();
    loop {
        if is_dir(system, "/", &path_join(&current, GIT_DIR)) {
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
    for (path, entry) in tree {
        out.extend_from_slice(encode_hex(path.as_bytes()).as_bytes());
        out.push(b'\t');
        out.extend_from_slice(entry.hash.as_bytes());
        // The mode is written only when it is not the default, so an ordinary tree serializes
        // exactly as it did before modes were recorded.
        if entry.executable || entry.symlink {
            out.push(b'\t');
            out.extend_from_slice(entry.mode().as_bytes());
        }
        out.push(b'\n');
    }
    out
}

pub(crate) fn parse_tree(bytes: &[u8]) -> Option<Tree> {
    let mut tree = Tree::new();
    for line in String::from_utf8_lossy(bytes).lines() {
        let (path_hex, rest) = line.split_once('\t')?;
        let path = String::from_utf8(decode_hex(path_hex)?).ok()?;
        let (hash, mode) = match rest.split_once('\t') {
            Some((hash, mode)) => (hash, mode),
            None => (rest, "100644"),
        };
        if path.is_empty() || hash.len() != 40 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        tree.insert(
            path,
            Entry {
                hash: hash.to_string(),
                executable: mode == "100755",
                symlink: mode == "120000",
            },
        );
    }
    Some(tree)
}

fn read_tree(system: &mut dyn System, path: &str) -> Option<Tree> {
    parse_tree(&read_all(system, path)?)
}

/// Write to the VFS, creating any missing parent directories.
///
/// `System::put_file_with_parents` stamps the write with the simulated clock and always applies
/// the given mode, matching what every caller here needs.
pub(crate) fn write_vfs(system: &mut dyn System, path: &str, bytes: &[u8]) -> VfsResult<()> {
    system
        .put_file_with_parents("/", path, bytes.to_vec(), 0o644)
        .map_err(to_vfs_error)
}

/// Translate a syscall failure back to the `VfsError` this module's callers expect.
///
/// Every caller of a `System` write already carries a `VfsError`-shaped result from before the
/// port; a resource limit has no `VfsError` counterpart; a full disk is the closest match, since
/// both mean "the write could not be made."
fn to_vfs_error(error: crate::syscalls::SyscallError) -> VfsError {
    match error {
        crate::syscalls::SyscallError::File(inner) => inner,
        crate::syscalls::SyscallError::ResourceExhausted => VfsError::NoSpace,
        other => VfsError::NotFound(other.to_string()),
    }
}

/// The staged tree, or `None` when the index cannot be read.
///
/// A repository with no index file has nothing staged, which is what Git makes of one; `None`
/// means the file is there and unreadable, which is worth saying out loud.
pub(crate) fn load_index(system: &mut dyn System, root: &str) -> Option<Tree> {
    match read_all(system, &git_path(root, INDEX)) {
        Some(bytes) => parse_tree(&bytes),
        None => Some(Tree::new()),
    }
}

pub(crate) fn store_index(system: &mut dyn System, root: &str, index: &Tree) -> VfsResult<()> {
    write_vfs(system, &git_path(root, INDEX), &serialize_tree(index))
}

/// The index write as a [`FileChange`], so a caller can commit it in the same
/// [`System::apply_file_batch`] transaction as the working-tree changes it goes with.
pub(crate) fn store_index_change(root: &str, index: &Tree) -> FileChange {
    FileChange::PutFile {
        path: git_path(root, INDEX),
        bytes: serialize_tree(index),
        mode: 0o644,
    }
}

pub(crate) fn read_blob(system: &mut dyn System, root: &str, hash: &str) -> Option<Vec<u8>> {
    read_all(system, &git_path(root, &format!("objects/{hash}")))
}

pub(crate) fn write_blob(system: &mut dyn System, root: &str, data: &[u8]) -> VfsResult<String> {
    let hash = blob_hash(data);
    write_vfs(system, &git_path(root, &format!("objects/{hash}")), data)?;
    Ok(hash)
}

/// A blob write as a [`FileChange`], so a caller can commit several object writes together with
/// the working-tree and index changes that reference them in one [`System::apply_file_batch`]
/// transaction, rather than writing objects immediately and possibly outliving a later failure.
pub(crate) fn blob_change(root: &str, data: Vec<u8>) -> (String, FileChange) {
    let hash = blob_hash(&data);
    let change = FileChange::PutFile {
        path: git_path(root, &format!("objects/{hash}")),
        bytes: data,
        mode: 0o644,
    };
    (hash, change)
}

/// The bytes Git records for one working-tree node, or `None` for something it does not track.
///
/// A symbolic link is stored as its target, which is how Git records one. A directory or a native
/// executable is not tracked at all.
fn tracked_content(system: &mut dyn System, base: &str, path: &str) -> Option<Vec<u8>> {
    let info = system.metadata(base, path, false).ok()?;
    if info.native_executable {
        return None;
    }
    match info.kind {
        FileKind::File => {
            let maximum = usize::try_from(system.limits().memory).unwrap_or(usize::MAX);
            system.read_file_limited(base, path, maximum).ok()
        }
        FileKind::Symlink => info.link_target.map(String::into_bytes),
        FileKind::Directory => None,
    }
}

/// Read a tracked file from the working tree.
pub(crate) fn read_work_file(system: &mut dyn System, root: &str, path: &str) -> Option<Vec<u8>> {
    let absolute = path_join(root, path);
    tracked_content(system, "/", &absolute)
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

pub(crate) fn load_commit(system: &mut dyn System, root: &str, id: &str) -> Option<Commit> {
    Commit::parse(&read_all(system, &commit_file(root, id))?)
}

/// A stable identifier for a tree.
///
/// Real Git hashes a packed tree object; this subset hashes its own serialization, so the value
/// is self-consistent across `cat-file`, `ls-tree`, and `rev-parse` but differs from Git's.
pub(crate) fn tree_hash(tree: &Tree) -> String {
    let body = serialize_tree(tree);
    sha1(&[format!("tree {}\0", body.len()).as_bytes(), &body].concat())
}

pub(crate) fn commit_tree(system: &mut dyn System, root: &str, id: &str) -> Option<Tree> {
    read_tree(system, &tree_file(root, id))
}

/// Store a commit and its tree, returning the new commit id.
///
/// The id is the SHA-1 of a Git-shaped header, so two commits differ whenever their tree,
/// parents, author, timestamp, or message differ.
pub(crate) fn store_commit(
    system: &mut dyn System,
    root: &str,
    commit: &Commit,
    tree: &Tree,
) -> VfsResult<String> {
    let tree_bytes = serialize_tree(tree);
    let mut identity = format!("tree {}\n", sha1(&tree_bytes));
    identity.push_str(&String::from_utf8_lossy(&commit.serialize()));
    let id = sha1(identity.as_bytes());
    write_vfs(system, &tree_file(root, &id), &tree_bytes)?;
    write_vfs(system, &commit_file(root, &id), &commit.serialize())?;
    Ok(id)
}

/// Read an annotated tag's record, which reuses the commit header shape for tagger and message.
pub(crate) fn read_annotation(system: &mut dyn System, root: &str, name: &str) -> Option<Commit> {
    let path = git_path(root, &format!("tags/{name}.annotation"));
    Commit::parse(&read_all(system, &path)?)
}

pub(crate) fn write_annotation(
    system: &mut dyn System,
    root: &str,
    name: &str,
    annotation: &Commit,
) -> VfsResult<()> {
    let path = git_path(root, &format!("tags/{name}.annotation"));
    write_vfs(system, &path, &annotation.serialize())
}

/// Walk first parents from `start`, most recent first, up to `limit` commits.
pub(crate) fn first_parent_history(
    system: &mut dyn System,
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
        let Some(commit) = load_commit(system, root, &id) else {
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
    system: &mut dyn System,
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
        let Some(commit) = load_commit(system, root, &id) else {
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
pub(crate) fn ancestors(system: &mut dyn System, root: &str, commit: &str) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut queue = vec![commit.to_string()];
    while let Some(id) = queue.pop() {
        if seen.len() >= 10_000 || !seen.insert(id.clone()) {
            continue;
        }
        if let Some(commit) = load_commit(system, root, &id) {
            queue.extend(commit.parents);
        }
    }
    seen
}

/// The best common ancestor of two commits, preferring the one nearest `left`.
pub(crate) fn merge_base(
    system: &mut dyn System,
    root: &str,
    left: &str,
    right: &str,
) -> Option<String> {
    let right_ancestors = ancestors(system, root, right);
    let mut queue = vec![left.to_string()];
    let mut seen = BTreeSet::new();
    while let Some(id) = queue.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        if right_ancestors.contains(&id) {
            return Some(id);
        }
        if let Some(commit) = load_commit(system, root, &id) {
            queue.extend(commit.parents);
        }
    }
    None
}

// -- commit identity and dates ------------------------------------------------------------------

pub(crate) fn author_identity(
    system: &mut dyn System,
    root: &str,
    globals: &Globals,
) -> (String, String) {
    super::config::identity(system, root, globals)
}

pub(crate) fn now_seconds(system: &dyn System) -> i64 {
    i64::try_from(system.wall_time_ms() / 1000).unwrap_or_default()
}

/// Format a commit timestamp the way `git log` prints author dates.
pub(crate) fn format_date(timestamp: i64) -> String {
    crate::commands::proc::format_date(
        i128::from(timestamp) * 1_000_000_000,
        "%a %b %-d %H:%M:%S %Y +0000",
    )
    .unwrap_or_else(|_| timestamp.to_string())
}

// -- references ---------------------------------------------------------------------------------

/// Read the raw contents of `.git/HEAD`.
fn head_contents(system: &mut dyn System, root: &str) -> Option<String> {
    Some(
        String::from_utf8_lossy(&read_all(system, &git_path(root, HEAD))?)
            .trim()
            .to_string(),
    )
}

/// The reference HEAD points at, or `None` when HEAD is detached.
pub(crate) fn head_reference(system: &mut dyn System, root: &str) -> Option<String> {
    head_contents(system, root)?
        .strip_prefix("ref: ")
        .map(str::to_string)
}

pub(crate) fn head_commit(system: &mut dyn System, root: &str) -> Option<String> {
    let head = head_contents(system, root)?;
    let commit = match head.strip_prefix("ref: ") {
        Some(reference) => read_reference(system, root, reference)?,
        None => head,
    };
    (!commit.is_empty()).then_some(commit)
}

pub(crate) fn current_branch(system: &mut dyn System, root: &str) -> Option<String> {
    head_reference(system, root)?
        .strip_prefix("refs/heads/")
        .map(str::to_string)
}

pub(crate) fn read_reference(
    system: &mut dyn System,
    root: &str,
    reference: &str,
) -> Option<String> {
    let value = String::from_utf8_lossy(&read_all(system, &git_path(root, reference))?)
        .trim()
        .to_string();
    (!value.is_empty()).then_some(value)
}

pub(crate) fn write_reference(
    system: &mut dyn System,
    root: &str,
    reference: &str,
    commit: &str,
) -> VfsResult<()> {
    write_vfs(
        system,
        &git_path(root, reference),
        format!("{commit}\n").as_bytes(),
    )
}

pub(crate) fn delete_reference(
    system: &mut dyn System,
    root: &str,
    reference: &str,
) -> VfsResult<()> {
    system
        .unlink("/", &git_path(root, reference))
        .map_err(to_vfs_error)
}

/// Point HEAD at `commit`, following the current branch when HEAD is not detached.
pub(crate) fn update_head(
    system: &mut dyn System,
    root: &str,
    commit: &str,
    action: &str,
) -> VfsResult<()> {
    let before = head_commit(system, root);
    let destination = head_reference(system, root).unwrap_or_else(|| HEAD.to_string());
    write_reference(system, root, &destination, commit)?;
    log_head_move(system, root, before.as_deref(), commit, action);
    Ok(())
}

/// Split a `REVISION:PATH` operand into the tree it names and the path within it.
///
/// An empty revision names the index, so `:file` and `:0:file` read what is staged rather than
/// what is committed, as they do in Git.
pub(crate) fn tree_and_path(
    system: &mut dyn System,
    root: &str,
    operand: &str,
) -> Option<(Tree, String)> {
    let (revision, path) = operand.split_once(':')?;
    if revision.is_empty() {
        let path = path.strip_prefix("0:").unwrap_or(path);
        return Some((
            load_index(system, root).unwrap_or_default(),
            path.to_string(),
        ));
    }
    let commit = resolve_revision(system, root, revision)?;
    Some((commit_tree(system, root, &commit)?, path.to_string()))
}

pub(crate) fn record_orig_head(system: &mut dyn System, root: &str) {
    if let Some(commit) = head_commit(system, root) {
        let _ = write_reference(system, root, "ORIG_HEAD", &commit);
    }
}

pub(crate) fn set_head_to_branch(
    system: &mut dyn System,
    root: &str,
    branch: &str,
    action: &str,
) -> VfsResult<()> {
    let before = head_commit(system, root);
    write_vfs(
        system,
        &git_path(root, HEAD),
        format!("ref: refs/heads/{branch}\n").as_bytes(),
    )?;
    if let Some(commit) = head_commit(system, root) {
        log_head_move(system, root, before.as_deref(), &commit, action);
    }
    Ok(())
}

pub(crate) fn set_head_detached(
    system: &mut dyn System,
    root: &str,
    commit: &str,
    action: &str,
) -> VfsResult<()> {
    let before = head_commit(system, root);
    write_vfs(
        system,
        &git_path(root, HEAD),
        format!("{commit}\n").as_bytes(),
    )?;
    log_head_move(system, root, before.as_deref(), commit, action);
    Ok(())
}

/// The most moves of HEAD the log keeps; older ones are forgotten.
const MAX_REFLOG_ENTRIES: usize = 1000;

/// One recorded move of HEAD.
pub(crate) struct HeadMove {
    pub before: String,
    pub after: String,
    pub action: String,
}

/// Record a move of HEAD, which is what `git reflog` reads and `HEAD@{N}` names.
fn log_head_move(
    system: &mut dyn System,
    root: &str,
    before: Option<&str>,
    after: &str,
    action: &str,
) {
    const MISSING: &str = "0000000000000000000000000000000000000000";
    // Git logs every move, including one that leaves HEAD where it was, so the numbering of
    // `HEAD@{N}` counts moves rather than distinct commits.
    let before = before.unwrap_or(MISSING);
    let mut moves = read_head_log(system, root);
    // Oldest first in the file, as Git writes it.
    moves.reverse();
    moves.push(HeadMove {
        before: before.to_string(),
        after: after.to_string(),
        action: action.replace(['\t', '\n'], " "),
    });
    if moves.len() > MAX_REFLOG_ENTRIES {
        moves.drain(..moves.len() - MAX_REFLOG_ENTRIES);
    }
    let text: String = moves
        .iter()
        .map(|entry| format!("{}\t{}\t{}\n", entry.before, entry.after, entry.action))
        .collect();
    let _ = write_vfs(system, &git_path(root, "logs/HEAD"), text.as_bytes());
}

/// Every recorded move of HEAD, newest first, which is the order `git reflog` prints.
pub(crate) fn read_head_log(system: &mut dyn System, root: &str) -> Vec<HeadMove> {
    let Some(bytes) = read_all(system, &git_path(root, "logs/HEAD")) else {
        return Vec::new();
    };
    let mut moves: Vec<HeadMove> = String::from_utf8_lossy(&bytes)
        .lines()
        .filter_map(|line| {
            let mut fields = line.splitn(3, '\t');
            Some(HeadMove {
                before: fields.next()?.to_string(),
                after: fields.next()?.to_string(),
                action: fields.next().unwrap_or_default().to_string(),
            })
        })
        .collect();
    moves.reverse();
    moves
}

/// Sorted names of the references below `.git/refs/<kind>`.
pub(crate) fn reference_names(system: &mut dyn System, root: &str, kind: &str) -> Vec<String> {
    let prefix = git_path(root, &format!("refs/{kind}"));
    let mut names = system
        .walk("/", &prefix)
        .unwrap_or_default()
        .into_iter()
        .filter(|path| is_file(system, "/", path))
        .filter_map(|path| path.strip_prefix(&format!("{prefix}/")).map(str::to_string))
        .collect::<Vec<_>>();
    names.sort();
    names
}

pub(crate) fn branch_names(system: &mut dyn System, root: &str) -> Vec<String> {
    reference_names(system, root, "heads")
}

pub(crate) fn head_tree(system: &mut dyn System, root: &str) -> Tree {
    head_commit(system, root)
        .and_then(|commit| commit_tree(system, root, &commit))
        .unwrap_or_default()
}

// -- revisions ----------------------------------------------------------------------------------

/// The branch the last branch switch moved away from, which `-` and `@{-1}` both name.
pub(crate) fn previous_branch(system: &mut dyn System, root: &str) -> Option<String> {
    let bytes = read_all(system, &git_path(root, "PREV_HEAD"))?;
    let previous = String::from_utf8_lossy(&bytes).trim().to_string();
    (!previous.is_empty()).then_some(previous)
}

/// Resolve a revision expression to a commit id.
///
/// Supported forms are `HEAD`, `@`, a branch or tag name, a full ref path, a full or abbreviated
/// commit id, and the `~N` and `^N` ancestry suffixes. Ranges and reflog selectors are not
/// supported; callers split ranges before calling.
pub(crate) fn resolve_revision(
    system: &mut dyn System,
    root: &str,
    revision: &str,
) -> Option<String> {
    let boundary = revision.find(['~', '^']).unwrap_or(revision.len());
    let (base, suffix) = revision.split_at(boundary);
    let mut commit = resolve_base_revision(system, root, base)?;
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
                    commit = load_commit(system, root, &commit)?.parents.first()?.clone();
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
                commit = load_commit(system, root, &commit)?
                    .parents
                    .get(index - 1)?
                    .clone();
            }
            _ => return None,
        }
    }
    Some(commit)
}

fn resolve_base_revision(system: &mut dyn System, root: &str, revision: &str) -> Option<String> {
    if revision.is_empty() || revision == "HEAD" || revision == "@" {
        return head_commit(system, root);
    }
    if let Some(position) = revision
        .strip_prefix("HEAD@{")
        .or_else(|| revision.strip_prefix("@{"))
        .and_then(|rest| rest.strip_suffix('}'))
        .and_then(|digits| digits.parse::<usize>().ok())
    {
        // `HEAD@{0}` is where HEAD is now; each step back is the state before one recorded move.
        let moves = read_head_log(system, root);
        return match position {
            0 => head_commit(system, root),
            _ => moves.get(position - 1).map(|entry| entry.before.clone()),
        };
    }
    if revision == "@{-1}" {
        let previous = previous_branch(system, root)?;
        return read_reference(system, root, &format!("refs/heads/{previous}"));
    }
    for candidate in [
        format!("refs/heads/{revision}"),
        format!("refs/tags/{revision}"),
        revision.to_string(),
    ] {
        if let Some(commit) = read_reference(system, root, &candidate) {
            return Some(commit);
        }
    }
    if revision.len() < 4 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let mut matches = system
        .walk("/", &git_path(root, "commits"))
        .unwrap_or_default()
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
/// `[remote "origin"] url = X` becomes `remote.origin.url`. A key may appear more than once, and
/// its values are kept in file order; the last one is what a plain read returns. Includes,
/// conditional includes, and value continuations are out of scope.
pub(crate) fn parse_config(text: &str) -> Config {
    let mut config = Config::new();
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
            config
                .entry(format!("{prefix}{}", line.to_ascii_lowercase()))
                .or_default()
                .push("true".to_string());
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
        config
            .entry(format!("{prefix}{}", key.trim().to_ascii_lowercase()))
            .or_default()
            .push(value.to_string());
    }
    config
}

/// Render configuration back to Git's INI format.
pub(crate) fn serialize_config(config: &Config) -> Vec<u8> {
    let mut out = String::new();
    let mut current = String::new();
    for (key, values) in config {
        let (heading, name) = split_config_key(key);
        if heading != current {
            out.push_str(&format!("[{heading}]\n"));
            current = heading;
        }
        for value in values {
            out.push_str(&format!("\t{name} = {value}\n"));
        }
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
///
/// `HOME` comes from the process's exported environment, the same as a real child process would
/// see: an unexported shell variable of that name is not visible here.
pub(crate) fn global_config_path(system: &dyn System) -> String {
    let home = system
        .environment()
        .get("HOME")
        .cloned()
        .unwrap_or_else(|| "/root".into());
    path_join(&home, ".gitconfig")
}

fn read_config_file(system: &mut dyn System, path: &str) -> Config {
    system
        .read_file_limited("/", path, 256 * 1024)
        .ok()
        .map(|bytes| parse_config(&String::from_utf8_lossy(&bytes)))
        .unwrap_or_default()
}

/// Repository-local configuration.
pub(crate) fn load_config(system: &mut dyn System, root: &str) -> Config {
    read_config_file(system, &git_path(root, CONFIG))
}

/// Per-user configuration, which the repository's own settings override.
pub(crate) fn load_global_config(system: &mut dyn System) -> Config {
    let path = global_config_path(system);
    read_config_file(system, &path)
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

pub(crate) fn write_config(system: &mut dyn System, path: &str, config: &Config) -> VfsResult<()> {
    write_vfs(system, path, &serialize_config(config))
}

// -- working tree -------------------------------------------------------------------------------

/// Hash every file in the working tree, metering the work against the caller's limits.
///
/// The memory the walk reserves is given back before returning, because every caller wants the
/// tree and nothing else.
pub(crate) fn collect_working_tree(system: &mut dyn System, root: &str) -> Result<Tree, i32> {
    let paths = system.walk("/", root).unwrap_or_default();
    let mut file_count = 0_u64;
    let mut path_bytes = 0_u64;
    let mut content_bytes = 0_u64;
    let mut tracked: Vec<(String, String, bool)> = Vec::new(); // (absolute, relative, symlink)
    for path in &paths {
        if is_git_path(root, path) || !within(root, path) {
            continue;
        }
        let Ok(info) = system.metadata("/", path, false) else {
            continue;
        };
        if info.native_executable || matches!(info.kind, FileKind::Directory) {
            continue;
        }
        let Some(relative) = relative_path(root, path) else {
            continue;
        };
        let Some(next_file_count) = file_count.checked_add(1) else {
            return Err(137);
        };
        let Some(next_path_bytes) = path_bytes.checked_add(relative.len() as u64) else {
            return Err(137);
        };
        let Some(next_content_bytes) = content_bytes.checked_add(info.size) else {
            return Err(137);
        };
        file_count = next_file_count;
        path_bytes = next_path_bytes;
        content_bytes = next_content_bytes;
        tracked.push((
            path.clone(),
            relative,
            matches!(info.kind, FileKind::Symlink),
        ));
    }
    if file_count > MAX_WORKING_FILES {
        return Err(resource_error(system));
    }
    let reserved_memory = path_bytes.saturating_add(file_count.saturating_mul(64));
    if !system.reserve_memory(reserved_memory) {
        return Err(resource_error(system));
    }
    if !system.charge_cpu(path_bytes.saturating_add(content_bytes)) {
        system.release_memory(reserved_memory);
        return Err(resource_error(system));
    }
    let mut files = Tree::new();
    for (absolute, relative, symlink) in tracked {
        if let Some(data) = tracked_content(system, "/", &absolute) {
            let mode = system
                .metadata("/", &absolute, false)
                .map(|info| info.mode)
                .unwrap_or(0o644);
            files.insert(
                relative,
                Entry {
                    hash: blob_hash(&data),
                    executable: !symlink && is_executable(mode),
                    symlink,
                },
            );
        }
    }
    system.release_memory(reserved_memory);
    Ok(files)
}

/// Hash one working-tree file, charging the read against the caller's limits.
pub(crate) fn metered_file_hash(
    system: &mut dyn System,
    path: &str,
) -> Result<Option<String>, i32> {
    // A symbolic link hashes to its target, so it must not be followed here.
    if let Ok(info) = system.metadata("/", path, false) {
        if info.kind == FileKind::Symlink {
            let target = info.link_target.unwrap_or_default();
            if !system.charge_cpu(target.len() as u64) {
                return Err(resource_error(system));
            }
            return Ok(Some(blob_hash(target.as_bytes())));
        }
    }
    if !is_file(system, "/", path) {
        return Ok(None);
    }
    let length = system
        .metadata("/", path, true)
        .map(|info| info.size)
        .map_err(|_| 1)?;
    if !system.reserve_memory(length) {
        return Err(resource_error(system));
    }
    if !system.charge_cpu(length) {
        system.release_memory(length);
        return Err(resource_error(system));
    }
    let data = system.read_file_limited("/", path, length as usize);
    system.release_memory(length);
    data.map(|bytes| Some(blob_hash(&bytes))).map_err(|_| 1)
}

/// Exit status to report when a resource limit stopped the command.
pub(crate) fn resource_error(system: &dyn System) -> i32 {
    system.stop_status()
}

/// Remove the directories a deletion left empty, stopping at the repository root.
///
/// Git tracks files rather than directories, so removing the last file in a directory removes the
/// directory too.
pub(crate) fn prune_empty_parents(system: &mut dyn System, root: &str, absolute: &str) {
    let mut current = parent_of(absolute);
    while let Some(directory) = current {
        if directory == root || !within(root, &directory) || is_git_path(root, &directory) {
            return;
        }
        match system.list_dir("/", &directory) {
            Ok(entries) if entries.is_empty() => {}
            _ => return,
        }
        if system.rmdir("/", &directory).is_err() {
            return;
        }
        current = parent_of(&directory);
    }
}

/// Replace the working-tree files named by `old` with the contents named by `new`.
///
/// The whole update is applied atomically: a failure part-way through (a missing blob, or a
/// resource limit) leaves the working tree untouched.
pub(crate) fn replace_work_tree(
    system: &mut dyn System,
    root: &str,
    old: &Tree,
    new: &Tree,
) -> Result<(), String> {
    write_work_tree(system, root, old, new, true)
}

/// Move the working tree from `old` to `new`, leaving unchanged paths alone.
///
/// Checkout, merge, and stash reapplication go through here so that an uncommitted edit to a file
/// the move does not touch survives, as it does in Git.
pub(crate) fn update_work_tree(
    system: &mut dyn System,
    root: &str,
    old: &Tree,
    new: &Tree,
) -> Result<(), String> {
    write_work_tree(system, root, old, new, false)
}

fn write_work_tree(
    system: &mut dyn System,
    root: &str,
    old: &Tree,
    new: &Tree,
    force: bool,
) -> Result<(), String> {
    let reserved = system.disk_used().saturating_add(4 * 1024);
    if !system.reserve_memory(reserved) {
        return Err("memory limit exceeded".to_string());
    }
    let result = write_work_tree_changes(system, root, old, new, force);
    system.release_memory(reserved);
    result
}

fn write_work_tree_changes(
    system: &mut dyn System,
    root: &str,
    old: &Tree,
    new: &Tree,
    force: bool,
) -> Result<(), String> {
    let mut changes = Vec::new();
    for path in old.keys().filter(|path| !new.contains_key(*path)) {
        let absolute = path_join(root, path);
        if is_file(system, "/", &absolute) {
            changes.push(FileChange::RemoveFile(absolute));
        }
    }
    for (path, entry) in new {
        // A path the move does not change keeps whatever the working tree holds, but a missing
        // file is still restored.
        let absolute = path_join(root, path);
        if !force && old.get(path) == Some(entry) && is_file(system, "/", &absolute) {
            continue;
        }
        let hash = &entry.hash;
        let data = read_blob(system, root, hash)
            .ok_or_else(|| format!("missing blob {hash} for {path}"))?;
        if entry.symlink {
            changes.push(FileChange::PutSymlink {
                link: absolute,
                target: String::from_utf8_lossy(&data).into_owned(),
            });
        } else {
            changes.push(FileChange::PutFile {
                path: absolute,
                bytes: data,
                mode: work_mode(entry.executable),
            });
        }
    }
    system
        .apply_file_batch("/", changes)
        .map_err(|error| error.to_string())?;
    // Removing a file can leave its directory empty; a batch does not prune those on its own.
    for path in old.keys().filter(|path| !new.contains_key(*path)) {
        prune_empty_parents(system, root, &path_join(root, path));
    }
    Ok(())
}
