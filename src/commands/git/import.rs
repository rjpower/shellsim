//! Trusted, bounded translation from host Git into Shellsim's private repository layout.
//!
//! The adapter invokes the host `git` executable before simulated execution begins. Git owns
//! repository discovery and loose/packed object decoding; this module reads raw objects through
//! `cat-file --batch`, validates and meters them, and immediately rewrites them into Shellsim's
//! private VFS format. The Git children retain host reads but run under a policy that blocks host
//! filesystem mutation, network access, and child processes. No process or host path is retained
//! after import.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Output, Stdio};

use crate::Environment;

use super::repo::{self, Commit, Entry, Tree};
use super::DEFAULT_CONFIG;

const MAX_COMMAND_OUTPUT: usize = 8 * 1024 * 1024;
const MAX_OBJECT_BYTES: usize = 16 * 1024 * 1024;
const MAX_COMMITS: usize = 512;
const MAX_COMMIT_PARENTS: usize = 64;
const MAX_TREE_DEPTH: usize = 64;
const MAX_TREE_ENTRIES: usize = 1_000_000;
const MAX_PATH_BYTES: usize = 4_096;
const MAX_BATCH_HEADER_BYTES: usize = 512;

/// Counts and identities from translating a bounded HEAD-reachable graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ImportReport {
    pub commits: usize,
    pub blobs: usize,
    pub tree_entries: usize,
    pub truncated_history: bool,
    pub source_head: String,
    pub imported_head: String,
}

struct SourceCommit {
    parents: Vec<String>,
    tree: String,
    author_name: String,
    author_email: String,
    timestamp: i64,
    message: String,
}

#[derive(Default)]
struct TranslationState {
    blobs: BTreeSet<String>,
    tree_entries: usize,
}

struct CommitSelection {
    order: Vec<String>,
    selected: BTreeSet<String>,
    truncated: bool,
}

/// Import recent history reachable from HEAD in `host_root` at `destination_root`.
///
/// The caller owns rollback of VFS writes. Host Git must be available, and this function must run
/// at the trusted ingestion boundary before a process sandbox denies native execution.
pub(crate) fn import_head_history(
    environment: &mut Environment,
    host_root: &Path,
    destination_root: &str,
) -> Result<ImportReport, String> {
    let host_root = host_root
        .canonicalize()
        .map_err(|error| format!("cannot resolve host root {}: {error}", host_root.display()))?;
    validate_repository(&host_root)?;

    let source_head = git_text(&host_root, &["rev-parse", "--verify", "HEAD^{commit}"])?;
    validate_oid(&source_head, "HEAD")?;
    let head_reference = git_optional_text(&host_root, &["symbolic-ref", "--quiet", "HEAD"])?;
    if let Some(reference) = &head_reference {
        validate_head_reference(reference)?;
    }

    let selection = select_commits(&host_root)?;
    if !selection.selected.contains(&source_head) {
        return Err("bounded revision walk did not include HEAD".to_string());
    }

    let mut objects = GitBatch::spawn(&host_root)?;
    initialize_private_repository(environment, destination_root)?;

    let mut translated = BTreeMap::new();
    let mut head_tree = None;
    let mut state = TranslationState::default();
    for source_id in &selection.order {
        let raw = objects.read_object(source_id, "commit")?;
        let source = parse_commit(&raw, source_id)?;
        let tree = import_tree(
            &mut objects,
            environment,
            destination_root,
            &source.tree,
            &mut state,
        )?;
        let parents = source
            .parents
            .iter()
            .filter(|parent| selection.selected.contains(*parent))
            .map(|parent| {
                translated.get(parent).cloned().ok_or_else(|| {
                    format!("selected parent {parent} was not imported before {source_id}")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let commit = Commit {
            parents,
            author_name: source.author_name,
            author_email: source.author_email,
            timestamp: source.timestamp,
            message: source.message,
        };
        let imported_id = repo::store_commit(environment, destination_root, &commit, &tree)
            .map_err(|error| format!("cannot store imported commit {source_id}: {error}"))?;
        if source_id == &source_head {
            head_tree = Some(tree);
        }
        translated.insert(source_id.clone(), imported_id);
    }

    let imported_head = translated
        .get(&source_head)
        .cloned()
        .ok_or_else(|| "HEAD was not imported".to_string())?;
    let head_tree = head_tree.ok_or_else(|| "HEAD tree was not imported".to_string())?;
    repo::store_index(environment, destination_root, &head_tree)
        .map_err(|error| format!("cannot store imported index: {error}"))?;
    match head_reference {
        Some(reference) => {
            repo::write_reference(environment, destination_root, &reference, &imported_head)
                .map_err(|error| format!("cannot store imported branch: {error}"))?;
            repo::write_vfs(
                environment,
                &repo::git_path(destination_root, repo::HEAD),
                format!("ref: {reference}\n").as_bytes(),
            )
            .map_err(|error| format!("cannot store imported HEAD: {error}"))?;
        }
        None => repo::write_vfs(
            environment,
            &repo::git_path(destination_root, repo::HEAD),
            format!("{imported_head}\n").as_bytes(),
        )
        .map_err(|error| format!("cannot store detached HEAD: {error}"))?,
    }

    Ok(ImportReport {
        commits: selection.order.len(),
        blobs: state.blobs.len(),
        tree_entries: state.tree_entries,
        truncated_history: selection.truncated,
        source_head,
        imported_head,
    })
}

fn validate_repository(host_root: &Path) -> Result<(), String> {
    if git_text(host_root, &["rev-parse", "--is-bare-repository"])? != "false" {
        return Err("bare Git repositories are unsupported by this spike".to_string());
    }
    if git_text(host_root, &["rev-parse", "--show-object-format"])? != "sha1" {
        return Err("SHA-256 Git repositories are unsupported by this spike".to_string());
    }
    let top = PathBuf::from(git_text(host_root, &["rev-parse", "--show-toplevel"])?);
    let top = top
        .canonicalize()
        .map_err(|error| format!("cannot resolve Git working tree {}: {error}", top.display()))?;
    if top != host_root {
        return Err("supplied directory is not the Git working-tree root".to_string());
    }
    let common = PathBuf::from(git_text(
        host_root,
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
    )?);
    if common.join("objects/info/alternates").exists() {
        return Err("Git object alternates are unsupported by this spike".to_string());
    }
    Ok(())
}

fn select_commits(host_root: &Path) -> Result<CommitSelection, String> {
    let limit = MAX_COMMITS
        .checked_add(1)
        .ok_or_else(|| "Git history limit overflow".to_string())?;
    let max_count = format!("--max-count={limit}");
    let output = git_text(
        host_root,
        &["rev-list", "--topo-order", "--reverse", &max_count, "HEAD"],
    )?;
    let mut order = output
        .lines()
        .map(|line| {
            validate_oid(line, "revision walk")?;
            Ok(line.to_string())
        })
        .collect::<Result<Vec<_>, String>>()?;
    if order.is_empty() {
        return Err("Git repository has no commits reachable from HEAD".to_string());
    }
    let truncated = order.len() > MAX_COMMITS;
    if truncated {
        order.remove(0);
    }
    let selected = order.iter().cloned().collect();
    Ok(CommitSelection {
        order,
        selected,
        truncated,
    })
}

fn parse_commit(bytes: &[u8], id: &str) -> Result<SourceCommit, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| format!("commit {id} contains non-UTF-8 metadata or message"))?;
    let (header, message) = text
        .split_once("\n\n")
        .ok_or_else(|| format!("commit {id} has no header terminator"))?;
    let mut tree = None;
    let mut parents = Vec::new();
    let mut author = None;
    for line in header.lines() {
        if let Some(value) = line.strip_prefix("tree ") {
            validate_oid(value, "commit tree")?;
            tree = Some(value.to_string());
        } else if let Some(value) = line.strip_prefix("parent ") {
            validate_oid(value, "commit parent")?;
            parents.push(value.to_string());
            if parents.len() > MAX_COMMIT_PARENTS {
                return Err(format!(
                    "commit {id} exceeds the {MAX_COMMIT_PARENTS}-parent spike limit"
                ));
            }
        } else if let Some(value) = line.strip_prefix("author ") {
            author = Some(parse_author(value, id)?);
        }
    }
    let (author_name, author_email, timestamp) =
        author.ok_or_else(|| format!("commit {id} has no author"))?;
    Ok(SourceCommit {
        parents,
        tree: tree.ok_or_else(|| format!("commit {id} has no tree"))?,
        author_name,
        author_email,
        timestamp,
        message: message.to_string(),
    })
}

fn parse_author(value: &str, id: &str) -> Result<(String, String, i64), String> {
    let (name, rest) = value
        .rsplit_once(" <")
        .ok_or_else(|| format!("commit {id} has an invalid author"))?;
    let (email, time) = rest
        .split_once("> ")
        .ok_or_else(|| format!("commit {id} has an invalid author email"))?;
    let timestamp = time
        .split_whitespace()
        .next()
        .ok_or_else(|| format!("commit {id} has no author timestamp"))?
        .parse::<i64>()
        .map_err(|_| format!("commit {id} has an invalid author timestamp"))?;
    Ok((name.to_string(), email.to_string(), timestamp))
}

fn import_tree(
    objects: &mut GitBatch,
    environment: &mut Environment,
    destination_root: &str,
    root: &str,
    state: &mut TranslationState,
) -> Result<Tree, String> {
    let mut imported = Tree::new();
    let mut pending = vec![(String::new(), root.to_string(), 0usize)];
    while let Some((prefix, tree_id, depth)) = pending.pop() {
        if depth > MAX_TREE_DEPTH {
            return Err(format!(
                "Git tree exceeds the {MAX_TREE_DEPTH}-level spike limit"
            ));
        }
        let tree = objects.read_object(&tree_id, "tree")?;
        for entry in parse_tree_entries(&tree, &tree_id)? {
            state.tree_entries = state
                .tree_entries
                .checked_add(1)
                .ok_or_else(|| "tree entry count overflow".to_string())?;
            if state.tree_entries > MAX_TREE_ENTRIES {
                return Err(format!(
                    "translated history exceeds the {MAX_TREE_ENTRIES}-tree-entry spike limit"
                ));
            }
            let name = std::str::from_utf8(&entry.name)
                .map_err(|_| "non-UTF-8 Git tree path is unsupported by this spike")?;
            if matches!(name, "" | "." | "..") || name.contains('/') {
                return Err(format!("unsafe Git tree path component: {name:?}"));
            }
            let path = if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}/{name}")
            };
            if path.len() > MAX_PATH_BYTES {
                return Err(format!(
                    "Git path exceeds the {MAX_PATH_BYTES}-byte spike limit"
                ));
            }
            match entry.mode.as_str() {
                "40000" | "040000" => pending.push((path, entry.id, depth + 1)),
                "100644" | "100755" | "120000" => {
                    if state.blobs.insert(entry.id.clone()) {
                        let blob = objects.read_object(&entry.id, "blob")?;
                        let imported_hash = repo::write_blob(environment, destination_root, &blob)
                            .map_err(|error| {
                                format!("cannot store imported blob {}: {error}", entry.id)
                            })?;
                        if imported_hash != entry.id {
                            return Err(format!(
                                "blob {} did not retain its SHA-1 during import",
                                entry.id
                            ));
                        }
                    }
                    imported.insert(
                        path,
                        Entry {
                            hash: entry.id,
                            executable: entry.mode == "100755",
                            symlink: entry.mode == "120000",
                        },
                    );
                }
                "160000" => {
                    return Err(format!("submodule at {path} is unsupported by this spike"));
                }
                mode => return Err(format!("unsupported Git mode {mode} at {path}")),
            }
        }
    }
    Ok(imported)
}

#[derive(Debug)]
struct RawTreeEntry {
    mode: String,
    name: Vec<u8>,
    id: String,
}

fn parse_tree_entries(bytes: &[u8], id: &str) -> Result<Vec<RawTreeEntry>, String> {
    let mut entries = Vec::new();
    let mut position = 0usize;
    while position < bytes.len() {
        let mode_end = bytes[position..]
            .iter()
            .position(|byte| *byte == b' ')
            .map(|offset| position + offset)
            .ok_or_else(|| format!("tree {id} has an unterminated mode"))?;
        let mode = std::str::from_utf8(&bytes[position..mode_end])
            .map_err(|_| format!("tree {id} has a non-ASCII mode"))?
            .to_string();
        position = mode_end + 1;
        let name_end = bytes[position..]
            .iter()
            .position(|byte| *byte == 0)
            .map(|offset| position + offset)
            .ok_or_else(|| format!("tree {id} has an unterminated name"))?;
        let name = bytes[position..name_end].to_vec();
        position = name_end + 1;
        let hash_end = position
            .checked_add(20)
            .ok_or_else(|| format!("tree {id} entry offset overflow"))?;
        let raw_hash = bytes
            .get(position..hash_end)
            .ok_or_else(|| format!("tree {id} has a truncated object id"))?;
        entries.push(RawTreeEntry {
            mode,
            name,
            id: hex(raw_hash),
        });
        position = hash_end;
    }
    Ok(entries)
}

fn initialize_private_repository(
    environment: &mut Environment,
    destination_root: &str,
) -> Result<(), String> {
    for suffix in ["", "refs/heads", "refs/tags", "commits", "objects"] {
        let path = if suffix.is_empty() {
            repo::path_join(destination_root, repo::GIT_DIR)
        } else {
            repo::git_path(destination_root, suffix)
        };
        environment
            .vfs
            .mkdir_all("/", &path)
            .map_err(|error| format!("cannot initialize imported repository: {error}"))?;
    }
    let config = DEFAULT_CONFIG
        .iter()
        .map(|(key, value)| ((*key).to_string(), vec![(*value).to_string()]))
        .collect();
    repo::write_vfs(
        environment,
        &repo::git_path(destination_root, repo::CONFIG),
        &repo::serialize_config(&config),
    )
    .map_err(|error| format!("cannot store imported Git config: {error}"))
}

struct GitBatch {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl GitBatch {
    fn spawn(host_root: &Path) -> Result<Self, String> {
        let mut command = git_command(host_root)?;
        let mut child = command
            .args(["cat-file", "--batch"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("cannot start host Git object reader: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "host Git object reader has no stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "host Git object reader has no stdout".to_string())?;
        Ok(Self {
            child,
            stdin: Some(stdin),
            stdout: BufReader::new(stdout),
        })
    }

    fn read_object(&mut self, id: &str, expected_kind: &str) -> Result<Vec<u8>, String> {
        validate_oid(id, "object request")?;
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| "host Git object reader is closed".to_string())?;
        writeln!(stdin, "{id}")
            .and_then(|_| stdin.flush())
            .map_err(|error| format!("cannot request Git object {id}: {error}"))?;

        let mut header = Vec::new();
        self.stdout
            .read_until(b'\n', &mut header)
            .map_err(|error| format!("cannot read Git object header for {id}: {error}"))?;
        if header.len() > MAX_BATCH_HEADER_BYTES {
            return Err(format!("Git object header for {id} is too large"));
        }
        let header = std::str::from_utf8(&header)
            .map_err(|_| format!("Git object header for {id} is not UTF-8"))?
            .trim_end_matches('\n');
        let fields = header.split_whitespace().collect::<Vec<_>>();
        if fields.len() == 2 && fields[1] == "missing" {
            return Err(format!("Git object {id} is missing"));
        }
        if fields.len() != 3 {
            return Err(format!("invalid Git object header for {id}: {header}"));
        }
        if fields[0] != id || fields[1] != expected_kind {
            return Err(format!(
                "Git object {id} resolved as {} {}, expected {expected_kind}",
                fields[0], fields[1]
            ));
        }
        let size = fields[2]
            .parse::<usize>()
            .map_err(|_| format!("Git object {id} has an invalid size"))?;
        if size > MAX_OBJECT_BYTES {
            return Err(format!(
                "Git object {id} exceeds the {MAX_OBJECT_BYTES}-byte spike limit"
            ));
        }
        let mut data = vec![0; size];
        self.stdout
            .read_exact(&mut data)
            .map_err(|error| format!("cannot read Git object {id}: {error}"))?;
        let mut newline = [0u8; 1];
        self.stdout
            .read_exact(&mut newline)
            .map_err(|error| format!("cannot finish Git object {id}: {error}"))?;
        if newline != *b"\n" {
            return Err(format!("Git object {id} has no record terminator"));
        }
        Ok(data)
    }
}

impl Drop for GitBatch {
    fn drop(&mut self) {
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn git_text(host_root: &Path, arguments: &[&str]) -> Result<String, String> {
    let output = run_git(host_root, arguments)?;
    if !output.status.success() {
        return Err(git_failure(arguments, &output));
    }
    if output.stdout.len() > MAX_COMMAND_OUTPUT {
        return Err(format!(
            "git {} output exceeds the {MAX_COMMAND_OUTPUT}-byte spike limit",
            arguments.join(" ")
        ));
    }
    std::str::from_utf8(&output.stdout)
        .map(|text| text.trim_end_matches(['\r', '\n']).to_string())
        .map_err(|_| format!("git {} returned non-UTF-8 output", arguments.join(" ")))
}

fn git_optional_text(host_root: &Path, arguments: &[&str]) -> Result<Option<String>, String> {
    let output = run_git(host_root, arguments)?;
    if output.status.success() {
        if output.stdout.len() > MAX_COMMAND_OUTPUT {
            return Err(format!(
                "git {} output exceeds the {MAX_COMMAND_OUTPUT}-byte spike limit",
                arguments.join(" ")
            ));
        }
        return std::str::from_utf8(&output.stdout)
            .map(|text| Some(text.trim_end_matches(['\r', '\n']).to_string()))
            .map_err(|_| format!("git {} returned non-UTF-8 output", arguments.join(" ")));
    }
    if output.status.code() == Some(1) {
        return Ok(None);
    }
    Err(git_failure(arguments, &output))
}

fn run_git(host_root: &Path, arguments: &[&str]) -> Result<Output, String> {
    git_command(host_root)?
        .args(arguments)
        .output()
        .map_err(|error| format!("cannot run host git {}: {error}", arguments.join(" ")))
}

fn git_command(host_root: &Path) -> Result<Command, String> {
    let mut command = Command::new("git");
    command
        .current_dir(host_root)
        .arg("--no-replace-objects")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_NO_LAZY_FETCH", "1")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_PAGER", "cat")
        .env("LC_ALL", "C")
        .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES")
        .env_remove("GIT_COMMON_DIR")
        .env_remove("GIT_CONFIG_COUNT")
        .env_remove("GIT_CONFIG_PARAMETERS")
        .env_remove("GIT_DIR")
        .env_remove("GIT_EXEC_PATH")
        .env_remove("GIT_INDEX_FILE")
        .env_remove("GIT_NAMESPACE")
        .env_remove("GIT_OBJECT_DIRECTORY")
        .env_remove("GIT_WORK_TREE");
    crate::sandbox::constrain_child_to_read_only(&mut command)?;
    Ok(command)
}

fn git_failure(arguments: &[&str], output: &Output) -> String {
    const MAX_DIAGNOSTIC_BYTES: usize = 8 * 1024;
    let stderr = &output.stderr[..output.stderr.len().min(MAX_DIAGNOSTIC_BYTES)];
    let diagnostic = String::from_utf8_lossy(stderr).trim().to_string();
    if diagnostic.is_empty() {
        format!(
            "host git {} exited with status {}",
            arguments.join(" "),
            output.status
        )
    } else {
        format!("host git {} failed: {diagnostic}", arguments.join(" "))
    }
}

fn validate_oid(value: &str, what: &str) -> Result<(), String> {
    if value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(format!("invalid SHA-1 for {what}: {value:?}"))
    }
}

fn validate_head_reference(reference: &str) -> Result<(), String> {
    let branch = reference
        .strip_prefix("refs/heads/")
        .ok_or_else(|| format!("unsupported symbolic HEAD target: {reference}"))?;
    if branch.is_empty()
        || branch
            .split('/')
            .any(|component| matches!(component, "" | "." | ".."))
    {
        return Err(format!("unsafe symbolic HEAD target: {reference}"));
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_tree_entries_preserve_modes_names_and_binary_object_ids() {
        let mut raw = b"100755 script.sh\0".to_vec();
        raw.extend([0x12; 20]);
        raw.extend_from_slice(b"40000 src\0");
        raw.extend([0xab; 20]);

        let entries = parse_tree_entries(&raw, "tree").unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].mode, "100755");
        assert_eq!(entries[0].name, b"script.sh");
        assert_eq!(entries[0].id, "12".repeat(20));
        assert_eq!(entries[1].mode, "40000");
        assert_eq!(entries[1].name, b"src");
        assert_eq!(entries[1].id, "ab".repeat(20));
    }

    #[test]
    fn raw_tree_entries_reject_truncated_object_ids() {
        let error = parse_tree_entries(b"100644 file\0short", "tree").unwrap_err();

        assert!(error.contains("truncated object id"), "{error}");
    }
}
