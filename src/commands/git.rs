//! A small, deterministic Git porcelain subset backed entirely by the simulated VFS.
//!
//! Repository metadata lives below `.git` in the VFS; no host Git process or host filesystem is
//! consulted.  The format is intentionally private and simple: the index is a sorted text map of
//! relative paths to SHA-1 content hashes, immutable blobs are stored by hash, and commits contain
//! a copy of the index map plus the message and parent. This supports common agent workflows
//! (`init`, `add`, `status`, `commit`, `diff`, and `rev-parse`) while rejecting unsupported Git
//! features explicitly.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::commands::{reg_costed, CommandContext, CommandSpec, Io, Trust};
use crate::interp::Interp;
use crate::vfs::{parent_of, resolve_against, NodeKind, Result as VfsResult};

const GIT_DIR: &str = ".git";
const HEAD: &str = "HEAD";
const INDEX: &str = "index";
const MAIN_REF: &str = "refs/heads/main";

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    reg_costed(m, &["git"], Trust::Partial, 150, 16 * 1024, cmd_git);
}

fn cmd_git(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(subcommand) = args.first().map(String::as_str) else {
        return usage(io, "missing command");
    };
    match subcommand {
        "init" => git_init(ctx, &args[1..], io),
        "status" => git_status(ctx, &args[1..], io),
        "add" => git_add(ctx, &args[1..], io),
        "commit" => git_commit(ctx, &args[1..], io),
        "diff" => git_diff(ctx, &args[1..], io),
        "rev-parse" => git_rev_parse(ctx, &args[1..], io),
        other => usage(io, &format!("unsupported subcommand: {other}")),
    }
}

fn usage(io: &mut Io, message: &str) -> i32 {
    io.err
        .extend_from_slice(format!("git: {message}\n").as_bytes());
    2
}

fn repo_error(io: &mut Io) -> i32 {
    io.err
        .extend_from_slice(b"fatal: not a git repository (or any parent up to mount point)\n");
    128
}

fn path_join(root: &str, suffix: &str) -> String {
    if root == "/" {
        format!("/{suffix}")
    } else {
        format!("{root}/{suffix}")
    }
}

fn git_path(root: &str, suffix: &str) -> String {
    path_join(root, &format!("{GIT_DIR}/{suffix}"))
}

fn find_repo_root(interp: &Interp) -> Option<String> {
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

fn within(root: &str, path: &str) -> bool {
    root == "/" || path == root || path.starts_with(&format!("{root}/"))
}

fn relative_path(root: &str, absolute: &str) -> Option<String> {
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

fn is_git_path(root: &str, path: &str) -> bool {
    let git = path_join(root, GIT_DIR);
    path == git || path.starts_with(&format!("{git}/"))
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

fn serialize_tree(tree: &BTreeMap<String, String>) -> Vec<u8> {
    let mut out = Vec::new();
    for (path, hash) in tree {
        out.extend_from_slice(encode_hex(path.as_bytes()).as_bytes());
        out.push(b'\t');
        out.extend_from_slice(hash.as_bytes());
        out.push(b'\n');
    }
    out
}

fn parse_tree(bytes: &[u8]) -> Option<BTreeMap<String, String>> {
    let mut tree = BTreeMap::new();
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

fn read_tree(interp: &Interp, path: &str) -> Option<BTreeMap<String, String>> {
    let bytes = interp.vfs.read("/", path).ok()?;
    parse_tree(&bytes)
}

fn write_vfs(interp: &mut Interp, path: &str, bytes: &[u8]) -> VfsResult<()> {
    interp.sync_vfs_time();
    interp.vfs.write("/", path, bytes, 0o644)
}

fn sha1(data: &[u8]) -> String {
    crate::hashes::sha1_hex(data)
}

fn collect_working_tree(
    ctx: &mut CommandContext<'_>,
    root: &str,
) -> Result<BTreeMap<String, String>, i32> {
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
    if file_count > 100_000 {
        return Err(resource_error(ctx));
    }
    if !ctx.reserve_memory(path_bytes.saturating_add(file_count.saturating_mul(64)))
        || !ctx.charge_cpu(path_bytes.saturating_add(content_bytes))
    {
        return Err(resource_error(ctx));
    }
    let mut tree = BTreeMap::new();
    for (path, node) in ctx.vfs.all_paths() {
        if is_git_path(root, path) || !within(root, path) {
            continue;
        }
        if let NodeKind::File(data) = &node.kind {
            if let Some(relative) = relative_path(root, path) {
                tree.insert(relative, sha1(data));
            }
        }
    }
    Ok(tree)
}

fn resource_error(ctx: &CommandContext<'_>) -> i32 {
    ctx.resources
        .stop_reason()
        .map_or(137, |reason| reason.exit_status())
}

fn git_init(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.iter().any(|arg| arg == "--bare") {
        return usage(io, "--bare is unsupported");
    }
    if args.len() > 1 || args.first().is_some_and(|arg| arg.starts_with('-')) {
        return usage(io, "usage: git init [DIRECTORY]");
    }
    let target = args
        .first()
        .map(|path| resolve_against(&ctx.cwd, path))
        .unwrap_or_else(|| ctx.cwd.clone());
    if !ctx.vfs.is_dir("/", &target) {
        if let Err(error) = ctx.vfs.mkdir_all("/", &target) {
            io.err
                .extend_from_slice(format!("git init: {error}\n").as_bytes());
            return 1;
        }
    }
    let git = path_join(&target, GIT_DIR);
    for directory in [
        git.clone(),
        path_join(&git, "refs"),
        path_join(&git, "refs/heads"),
        path_join(&git, "commits"),
        path_join(&git, "objects"),
    ] {
        if let Err(error) = ctx.vfs.mkdir_all("/", &directory) {
            io.err
                .extend_from_slice(format!("git init: {error}\n").as_bytes());
            return 1;
        }
    }
    let head = git_path(&target, HEAD);
    let index = git_path(&target, INDEX);
    if let Err(error) = write_vfs(ctx, &head, format!("ref: {MAIN_REF}\n").as_bytes()) {
        io.err
            .extend_from_slice(format!("git init: {error}\n").as_bytes());
        return 1;
    }
    if let Err(error) = write_vfs(ctx, &index, &[]) {
        io.err
            .extend_from_slice(format!("git init: {error}\n").as_bytes());
        return 1;
    }
    io.out
        .extend_from_slice(format!("Initialized empty Git repository in {git}/\n").as_bytes());
    0
}

fn load_index(interp: &Interp, root: &str) -> Option<BTreeMap<String, String>> {
    read_tree(interp, &git_path(root, INDEX))
}

fn head_commit(interp: &Interp, root: &str) -> Option<String> {
    let head = String::from_utf8_lossy(&interp.vfs.read("/", &git_path(root, HEAD)).ok()?)
        .trim()
        .to_string();
    let reference = head.strip_prefix("ref: ").unwrap_or(MAIN_REF);
    let commit = String::from_utf8_lossy(&interp.vfs.read("/", &git_path(root, reference)).ok()?)
        .trim()
        .to_string();
    (!commit.is_empty()).then_some(commit)
}

fn head_tree(interp: &Interp, root: &str) -> BTreeMap<String, String> {
    let Some(commit) = head_commit(interp, root) else {
        return BTreeMap::new();
    };
    read_tree(interp, &git_path(root, &format!("commits/{commit}.tree"))).unwrap_or_default()
}

fn git_status(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args
        .iter()
        .any(|arg| arg != "--short" && arg != "--porcelain")
    {
        return usage(io, "only --short/--porcelain are supported");
    }
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    let Some(index) = load_index(ctx, &root) else {
        io.err
            .extend_from_slice(b"fatal: invalid simulated index\n");
        return 128;
    };
    let work = match collect_working_tree(ctx, &root) {
        Ok(tree) => tree,
        Err(status) => return status,
    };
    let head = head_tree(ctx, &root);
    let mut paths = BTreeSet::new();
    paths.extend(head.keys().cloned());
    paths.extend(index.keys().cloned());
    paths.extend(work.keys().cloned());
    for path in paths {
        let head_value = head.get(&path);
        let index_value = index.get(&path);
        let work_value = work.get(&path);
        let staged = match (head_value, index_value) {
            (None, Some(_)) => Some('A'),
            (Some(old), Some(new)) if old != new => Some('M'),
            (Some(_), None) => Some('D'),
            _ => None,
        };
        let unstaged = match (index_value, work_value) {
            (None, Some(_)) if head_value.is_none() => Some('?'),
            (None, Some(_)) => None,
            (Some(_), None) => Some('D'),
            (Some(old), Some(new)) if old != new => Some('M'),
            _ => None,
        };
        if staged.is_none() && unstaged.is_none() {
            continue;
        }
        let (x, y) = match (staged, unstaged) {
            (None, Some('?')) => ('?', '?'),
            (a, b) => (a.unwrap_or(' '), b.unwrap_or(' ')),
        };
        io.out
            .extend_from_slice(format!("{x}{y} {path}\n").as_bytes());
    }
    0
}

fn selected_paths(
    cwd: &str,
    root: &str,
    args: &[String],
    index: &BTreeMap<String, String>,
    work: &BTreeMap<String, String>,
) -> Result<BTreeSet<String>, String> {
    let mut selected = BTreeSet::new();
    for argument in args {
        if argument.starts_with('-') {
            return Err(format!("unsupported add option: {argument}"));
        }
        let absolute = resolve_against(cwd, argument);
        if !within(root, &absolute) {
            return Err(format!("pathspec '{argument}' is outside the repository"));
        }
        let relative = relative_path(root, &absolute).unwrap_or_default();
        if relative.is_empty() {
            selected.extend(index.keys().cloned());
            continue;
        }
        let prefix = format!("{relative}/");
        let mut found = false;
        for key in index.keys() {
            if key == &relative || key.starts_with(&prefix) {
                selected.insert(key.clone());
                found = true;
            }
        }
        for work_path in work.keys() {
            if work_path == &relative || work_path.starts_with(&prefix) {
                selected.insert(work_path.clone());
                found = true;
            }
        }
        if !found {
            // Missing paths are allowed when removing a previously tracked file.
            selected.insert(relative);
        }
    }
    Ok(selected)
}

fn git_add(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let all = args.iter().any(|arg| arg == "-A" || arg == "--all");
    let operands: Vec<String> = args
        .iter()
        .filter(|arg| *arg != "-A" && *arg != "--all")
        .cloned()
        .collect();
    if all && !operands.is_empty() {
        return usage(io, "-A cannot be combined with paths in this subset");
    }
    if !all && operands.is_empty() {
        return usage(io, "nothing specified, nothing added");
    }
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut index = load_index(ctx, &root).unwrap_or_default();
    let work = match collect_working_tree(ctx, &root) {
        Ok(tree) => tree,
        Err(status) => return status,
    };
    let selected = if all {
        let mut all_paths = BTreeSet::new();
        all_paths.extend(index.keys().cloned());
        all_paths.extend(work.keys().cloned());
        all_paths
    } else {
        if operands.len() > 256 {
            return usage(io, "too many pathspecs");
        }
        let selection_units = (operands.len() as u64)
            .saturating_mul((index.len() as u64).saturating_add(work.len() as u64));
        if !ctx.charge_cpu(selection_units) {
            return resource_error(ctx);
        }
        let cwd = ctx.cwd.clone();
        match selected_paths(&cwd, &root, &operands, &index, &work) {
            Ok(paths) => paths,
            Err(message) => return usage(io, &message),
        }
    };
    if selected.len() > 100_000 {
        return resource_error(ctx);
    }
    if !ctx.reserve_memory(selected.len() as u64 * 64)
        || !ctx.charge_cpu(selected.len() as u64 * 64)
    {
        return resource_error(ctx);
    }
    for path in selected {
        if let Some(hash) = work.get(&path) {
            let Some(data) = read_work_file(ctx, &root, &path) else {
                io.err
                    .extend_from_slice(format!("git add: unable to read '{path}'\n").as_bytes());
                return 1;
            };
            if sha1(&data) != *hash {
                io.err.extend_from_slice(
                    format!("git add: '{path}' changed while building the index\n").as_bytes(),
                );
                return 1;
            }
            if let Err(error) = write_vfs(ctx, &git_path(&root, &format!("objects/{hash}")), &data)
            {
                io.err
                    .extend_from_slice(format!("git add: {error}\n").as_bytes());
                return 1;
            }
            index.insert(path, hash.clone());
        } else {
            index.remove(&path);
        }
    }
    let data = serialize_tree(&index);
    if let Err(error) = write_vfs(ctx, &git_path(&root, INDEX), &data) {
        io.err
            .extend_from_slice(format!("git add: {error}\n").as_bytes());
        return 1;
    }
    0
}

fn git_commit(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut message = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" => {
                i += 1;
                message = args.get(i).cloned();
            }
            arg if arg.starts_with("--message=") => message = Some(arg[10..].to_string()),
            arg if arg.starts_with('-') => {
                return usage(io, &format!("unsupported commit option: {arg}"))
            }
            _ => return usage(io, "usage: git commit -m MESSAGE"),
        }
        i += 1;
    }
    let Some(message) = message.filter(|value| !value.is_empty()) else {
        return usage(io, "a non-empty -m MESSAGE is required");
    };
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    let Some(index) = load_index(ctx, &root) else {
        io.err
            .extend_from_slice(b"fatal: invalid simulated index\n");
        return 128;
    };
    let parent = head_commit(ctx, &root);
    if index == head_tree(ctx, &root) {
        io.out
            .extend_from_slice(b"nothing to commit, working tree clean\n");
        return 1;
    }
    let tree = serialize_tree(&index);
    let id_input = format!(
        "tree\0{}\0{}\0{}",
        sha1(&tree),
        parent.as_deref().unwrap_or(""),
        message
    );
    let id = sha1(id_input.as_bytes());
    let commit_dir = git_path(&root, &format!("commits/{id}"));
    for (suffix, data) in [("tree", tree.as_slice()), ("message", message.as_bytes())] {
        if let Err(error) = write_vfs(ctx, &format!("{commit_dir}.{suffix}"), data) {
            io.err
                .extend_from_slice(format!("git commit: {error}\n").as_bytes());
            return 1;
        }
    }
    if let Some(parent) = &parent {
        if let Err(error) = write_vfs(ctx, &format!("{commit_dir}.parent"), parent.as_bytes()) {
            io.err
                .extend_from_slice(format!("git commit: {error}\n").as_bytes());
            return 1;
        }
    }
    let reference = git_path(&root, MAIN_REF);
    if let Err(error) = write_vfs(ctx, &reference, format!("{id}\n").as_bytes()) {
        io.err
            .extend_from_slice(format!("git commit: {error}\n").as_bytes());
        return 1;
    }
    let prefix = if parent.is_none() { "root-commit " } else { "" };
    io.out
        .extend_from_slice(format!("[main {prefix}{}] {message}\n", &id[..7]).as_bytes());
    0
}

fn read_work_file(ctx: &Interp, root: &str, path: &str) -> Option<Vec<u8>> {
    ctx.vfs.read("/", &path_join(root, path)).ok()
}

fn read_blob(ctx: &Interp, root: &str, hash: &str) -> Option<Vec<u8>> {
    ctx.vfs
        .read("/", &git_path(root, &format!("objects/{hash}")))
        .ok()
}

fn emit_diff(io: &mut Io, path: &str, old: Option<&[u8]>, new: Option<&[u8]>) {
    if old == new {
        return;
    }
    io.out
        .extend_from_slice(format!("diff --git a/{path} b/{path}\n").as_bytes());
    match old {
        Some(_) => io
            .out
            .extend_from_slice(format!("--- a/{path}\n").as_bytes()),
        None => io.out.extend_from_slice(b"--- /dev/null\n"),
    }
    match new {
        Some(_) => io
            .out
            .extend_from_slice(format!("+++ b/{path}\n").as_bytes()),
        None => io.out.extend_from_slice(b"+++ /dev/null\n"),
    }
    let old_lines = old
        .map(|data| {
            String::from_utf8_lossy(data)
                .lines()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let new_lines = new
        .map(|data| {
            String::from_utf8_lossy(data)
                .lines()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    io.out.extend_from_slice(
        format!(
            "@@ -1,{} +1,{} @@\n",
            old_lines.len().max(1),
            new_lines.len().max(1)
        )
        .as_bytes(),
    );
    for line in old_lines {
        io.out.extend_from_slice(format!("-{line}\n").as_bytes());
    }
    for line in new_lines {
        io.out.extend_from_slice(format!("+{line}\n").as_bytes());
    }
}

fn git_diff(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut cached = false;
    let mut paths = Vec::new();
    for arg in args {
        match arg.as_str() {
            "--cached" | "--staged" => cached = true,
            arg if arg.starts_with('-') => {
                return usage(io, &format!("unsupported diff option: {arg}"))
            }
            path => paths.push(path.to_string()),
        }
    }
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    let index = load_index(ctx, &root).unwrap_or_default();
    let work = if cached {
        BTreeMap::new()
    } else {
        match collect_working_tree(ctx, &root) {
            Ok(tree) => tree,
            Err(status) => return status,
        }
    };
    let left_tree = if cached {
        head_tree(ctx, &root)
    } else {
        index.clone()
    };
    let right_tree = if cached { index.clone() } else { work };
    let selected = |path: &str| {
        paths.is_empty()
            || paths
                .iter()
                .any(|prefix| path == prefix || path.starts_with(&format!("{prefix}/")))
    };
    let mut names = BTreeSet::new();
    names.extend(left_tree.keys().cloned());
    names.extend(right_tree.keys().cloned());
    for path in names {
        if !selected(&path) || left_tree.get(&path) == right_tree.get(&path) {
            continue;
        }
        let old = left_tree
            .get(&path)
            .and_then(|hash| read_blob(ctx, &root, hash));
        let new = if cached {
            right_tree
                .get(&path)
                .and_then(|hash| read_blob(ctx, &root, hash))
        } else {
            read_work_file(ctx, &root, &path)
        };
        emit_diff(io, &path, old.as_deref(), new.as_deref());
    }
    0
}

fn git_rev_parse(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    match args {
        [flag] if flag == "--show-toplevel" => {
            io.out.extend_from_slice(format!("{root}\n").as_bytes());
            0
        }
        [flag] if flag == "--is-inside-work-tree" => {
            io.out.extend_from_slice(b"true\n");
            0
        }
        [] => {
            let Some(commit) = head_commit(ctx, &root) else {
                io.err
                    .extend_from_slice(b"fatal: ambiguous argument 'HEAD'\n");
                return 128;
            };
            io.out.extend_from_slice(format!("{commit}\n").as_bytes());
            0
        }
        [single] if single == "HEAD" => {
            let Some(commit) = head_commit(ctx, &root) else {
                io.err
                    .extend_from_slice(b"fatal: ambiguous argument 'HEAD'\n");
                return 128;
            };
            io.out.extend_from_slice(format!("{commit}\n").as_bytes());
            0
        }
        _ => usage(io, "unsupported rev-parse form"),
    }
}
