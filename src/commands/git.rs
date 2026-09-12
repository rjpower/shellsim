//! A small, deterministic Git porcelain subset backed entirely by the simulated VFS.
//!
//! Repository metadata lives below `.git` in the VFS; no host Git process or host filesystem is
//! consulted.  The format is intentionally private and simple: the index is a sorted text map of
//! relative paths to SHA-1 content hashes, immutable blobs are stored by hash, and commits contain
//! a copy of the index map plus the message and parent. This supports common agent workflows
//! (`init`, staging, commits, refs, history, switching, restore/reset, diff, and revision lookup)
//! while rejecting unsupported Git features explicitly.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::commands::{reg_costed, CommandContext, CommandSpec, Io, Trust};
use crate::interp::Interp;
use crate::vfs::{parent_of, resolve_against, NodeKind, Result as VfsResult};

const GIT_DIR: &str = ".git";
const HEAD: &str = "HEAD";
const INDEX: &str = "index";
const CONFIG: &str = "config";
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
        "branch" => git_branch(ctx, &args[1..], io),
        "switch" => git_switch(ctx, &args[1..], io),
        "checkout" => git_checkout(ctx, &args[1..], io),
        "log" => git_log(ctx, &args[1..], io),
        "show" => git_show(ctx, &args[1..], io),
        "restore" => git_restore(ctx, &args[1..], io),
        "reset" => git_reset(ctx, &args[1..], io),
        "config" => git_config(ctx, &args[1..], io),
        "ls-files" => git_ls_files(ctx, &args[1..], io),
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

struct WorkingTreeSnapshot {
    files: BTreeMap<String, String>,
    reserved_memory: u64,
}

fn collect_working_tree(
    ctx: &mut CommandContext<'_>,
    root: &str,
) -> Result<WorkingTreeSnapshot, i32> {
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
    let reserved_memory = path_bytes.saturating_add(file_count.saturating_mul(64));
    if !ctx.reserve_memory(reserved_memory) {
        return Err(resource_error(ctx));
    }
    if !ctx.charge_cpu(path_bytes.saturating_add(content_bytes)) {
        ctx.resources.release_memory(reserved_memory);
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
    Ok(WorkingTreeSnapshot {
        files: tree,
        reserved_memory,
    })
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
    let config = git_path(&target, CONFIG);
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
    if !ctx.vfs.is_file("/", &config) {
        if let Err(error) = write_vfs(ctx, &config, &[]) {
            io.err
                .extend_from_slice(format!("git init: {error}\n").as_bytes());
            return 1;
        }
    }
    io.out
        .extend_from_slice(format!("Initialized empty Git repository in {git}/\n").as_bytes());
    0
}

fn load_index(interp: &Interp, root: &str) -> Option<BTreeMap<String, String>> {
    read_tree(interp, &git_path(root, INDEX))
}

fn load_config(ctx: &mut CommandContext<'_>, root: &str) -> Result<BTreeMap<String, String>, i32> {
    let bytes = ctx
        .fs_read_limited("/", &git_path(root, CONFIG), 256 * 1024)
        .map_err(|_| 128)?;
    let text = String::from_utf8(bytes).map_err(|_| 128)?;
    let mut config = BTreeMap::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('\t') else {
            return Err(128);
        };
        if config.len() == 256 || !valid_config_key(key) {
            return Err(128);
        }
        config.insert(key.to_string(), value.to_string());
    }
    Ok(config)
}

fn valid_config_key(key: &str) -> bool {
    key.contains('.')
        && !key.starts_with('.')
        && !key.ends_with('.')
        && !key.contains("..")
        && key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
}

fn write_config(
    ctx: &mut CommandContext<'_>,
    root: &str,
    config: &BTreeMap<String, String>,
) -> VfsResult<()> {
    let mut bytes = Vec::new();
    for (key, value) in config {
        bytes.extend_from_slice(key.as_bytes());
        bytes.push(b'\t');
        bytes.extend_from_slice(value.as_bytes());
        bytes.push(b'\n');
    }
    write_vfs(ctx, &git_path(root, CONFIG), &bytes)
}

fn git_config(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    let args = if args.first().is_some_and(|arg| arg == "--local") {
        &args[1..]
    } else {
        args
    };
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--global" | "--system"))
    {
        return usage(io, "only repository-local config is available");
    }
    let mut config = match load_config(ctx, &root) {
        Ok(config) => config,
        Err(status) => {
            io.err
                .extend_from_slice(b"fatal: invalid simulated config\n");
            return status;
        }
    };
    match args {
        [flag] if flag == "--list" => {
            for (key, value) in config {
                io.out
                    .extend_from_slice(format!("{key}={value}\n").as_bytes());
            }
            0
        }
        [flag, key] if flag == "--get" => {
            config.get(&key.to_ascii_lowercase()).map_or(1, |value| {
                io.out.extend_from_slice(format!("{value}\n").as_bytes());
                0
            })
        }
        [flag, key] if flag == "--unset" => {
            if config.remove(&key.to_ascii_lowercase()).is_none() {
                return 5;
            }
            write_config(ctx, &root, &config).map_or(1, |()| 0)
        }
        [key] if !key.starts_with('-') => {
            config.get(&key.to_ascii_lowercase()).map_or(1, |value| {
                io.out.extend_from_slice(format!("{value}\n").as_bytes());
                0
            })
        }
        [key, value] if !key.starts_with('-') => {
            let key = key.to_ascii_lowercase();
            if !valid_config_key(&key)
                || value.len() > 4096
                || value.contains(['\0', '\n', '\r', '\t'])
            {
                return usage(io, "invalid config key or value");
            }
            if config.len() == 256 && !config.contains_key(&key) {
                return usage(io, "too many config entries");
            }
            config.insert(key, value.clone());
            write_config(ctx, &root, &config).map_or(1, |()| 0)
        }
        _ => usage(
            io,
            "usage: git config [--local] [--get|--unset] NAME [VALUE]",
        ),
    }
}

fn git_ls_files(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut nul = false;
    let mut error_unmatch = false;
    let mut paths = Vec::new();
    let mut options = true;
    for arg in args {
        match arg.as_str() {
            "--" if options => options = false,
            "--cached" if options => {}
            "--error-unmatch" if options => error_unmatch = true,
            "-z" if options => nul = true,
            value if options && value.starts_with('-') => {
                return usage(io, &format!("unsupported ls-files option: {value}"))
            }
            value => paths.push(value.to_string()),
        }
    }
    let index = load_index(ctx, &root).unwrap_or_default();
    let cwd_prefix = relative_path(&root, &ctx.cwd).unwrap_or_default();
    let cwd_prefix = (!cwd_prefix.is_empty()).then(|| format!("{cwd_prefix}/"));
    let mut matched = false;
    for path in index.keys() {
        let displayed = cwd_prefix
            .as_deref()
            .and_then(|prefix| path.strip_prefix(prefix))
            .unwrap_or(path);
        if cwd_prefix.is_some() && displayed == path {
            continue;
        }
        if !paths.is_empty()
            && !paths.iter().any(|selected| {
                displayed == selected || displayed.starts_with(&format!("{selected}/"))
            })
        {
            continue;
        }
        matched = true;
        io.out.extend_from_slice(displayed.as_bytes());
        io.out.push(if nul { 0 } else { b'\n' });
    }
    if error_unmatch && !matched {
        io.err
            .extend_from_slice(b"error: pathspec did not match any files\n");
        1
    } else {
        0
    }
}

fn head_commit(interp: &Interp, root: &str) -> Option<String> {
    let head = String::from_utf8_lossy(&interp.vfs.read("/", &git_path(root, HEAD)).ok()?)
        .trim()
        .to_string();
    let commit = if let Some(reference) = head.strip_prefix("ref: ") {
        String::from_utf8_lossy(&interp.vfs.read("/", &git_path(root, reference)).ok()?)
            .trim()
            .to_string()
    } else {
        head
    };
    (!commit.is_empty()).then_some(commit)
}

fn head_reference(interp: &Interp, root: &str) -> Option<String> {
    let head = String::from_utf8_lossy(&interp.vfs.read("/", &git_path(root, HEAD)).ok()?)
        .trim()
        .to_string();
    head.strip_prefix("ref: ").map(str::to_string)
}

fn current_branch(interp: &Interp, root: &str) -> Option<String> {
    head_reference(interp, root)?
        .strip_prefix("refs/heads/")
        .map(str::to_string)
}

fn read_reference(interp: &Interp, root: &str, reference: &str) -> Option<String> {
    let value = String::from_utf8_lossy(&interp.vfs.read("/", &git_path(root, reference)).ok()?)
        .trim()
        .to_string();
    (!value.is_empty()).then_some(value)
}

fn resolve_revision(interp: &Interp, root: &str, revision: &str) -> Option<String> {
    if revision == "HEAD" {
        return head_commit(interp, root);
    }
    if let Some(commit) = read_reference(interp, root, &format!("refs/heads/{revision}")) {
        return Some(commit);
    }
    let commits = interp.vfs.walk(&git_path(root, "commits"));
    let mut matches = commits.into_iter().filter_map(|path| {
        let filename = crate::vfs::basename(&path);
        let id = filename.strip_suffix(".tree")?;
        id.starts_with(revision).then(|| id.to_string())
    });
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn commit_tree(interp: &Interp, root: &str, commit: &str) -> Option<BTreeMap<String, String>> {
    read_tree(interp, &git_path(root, &format!("commits/{commit}.tree")))
}

fn commit_parent(interp: &Interp, root: &str, commit: &str) -> Option<String> {
    let bytes = interp
        .vfs
        .read("/", &git_path(root, &format!("commits/{commit}.parent")))
        .ok()?;
    let parent = String::from_utf8_lossy(&bytes).trim().to_string();
    (!parent.is_empty()).then_some(parent)
}

fn commit_message(interp: &Interp, root: &str, commit: &str) -> Option<String> {
    interp
        .vfs
        .read("/", &git_path(root, &format!("commits/{commit}.message")))
        .ok()
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
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
        .any(|arg| arg != "--short" && arg != "--porcelain" && arg != "--porcelain=v1")
    {
        return usage(io, "only --short/--porcelain=v1 are supported");
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
        Ok(snapshot) => snapshot,
        Err(status) => return status,
    };
    let head = head_tree(ctx, &root);
    let mut paths = BTreeSet::new();
    paths.extend(head.keys().cloned());
    paths.extend(index.keys().cloned());
    paths.extend(work.files.keys().cloned());
    let mut entries = Vec::new();
    for path in paths {
        let head_value = head.get(&path);
        let index_value = index.get(&path);
        let work_value = work.files.get(&path);
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
        entries.push((x, y, path));
    }
    ctx.resources.release_memory(work.reserved_memory);
    if args.is_empty() {
        let branch = current_branch(ctx, &root).unwrap_or_else(|| "HEAD".to_string());
        io.out
            .extend_from_slice(format!("On branch {branch}\n\n").as_bytes());
        if entries.is_empty() {
            io.out
                .extend_from_slice(b"nothing to commit, working tree clean\n");
        } else {
            io.out.extend_from_slice(b"Changes:\n");
            for (x, y, path) in entries {
                io.out
                    .extend_from_slice(format!("  {x}{y} {path}\n").as_bytes());
            }
        }
    } else {
        for (x, y, path) in entries {
            io.out
                .extend_from_slice(format!("{x}{y} {path}\n").as_bytes());
        }
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
            selected.extend(work.keys().cloned());
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
        Ok(snapshot) => snapshot,
        Err(status) => return status,
    };
    let status = git_add_from_snapshot(ctx, &root, operands, all, &mut index, &work.files, io);
    ctx.resources.release_memory(work.reserved_memory);
    status
}

fn git_add_from_snapshot(
    ctx: &mut CommandContext<'_>,
    root: &str,
    operands: Vec<String>,
    all: bool,
    index: &mut BTreeMap<String, String>,
    work: &BTreeMap<String, String>,
    io: &mut Io,
) -> i32 {
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
        match selected_paths(&cwd, root, &operands, index, work) {
            Ok(paths) => paths,
            Err(message) => return usage(io, &message),
        }
    };
    if selected.len() > 100_000 {
        return resource_error(ctx);
    }
    let selected_reserved = selected.len() as u64 * 64;
    if !ctx.reserve_memory(selected_reserved) {
        return resource_error(ctx);
    }
    if !ctx.charge_cpu(selected_reserved) {
        ctx.resources.release_memory(selected_reserved);
        return resource_error(ctx);
    }
    let status = git_add_selected(ctx, root, index, work, selected, io);
    ctx.resources.release_memory(selected_reserved);
    status
}

fn git_add_selected(
    ctx: &mut CommandContext<'_>,
    root: &str,
    index: &mut BTreeMap<String, String>,
    work: &BTreeMap<String, String>,
    selected: BTreeSet<String>,
    io: &mut Io,
) -> i32 {
    for path in selected {
        if let Some(hash) = work.get(&path) {
            let Some(data) = read_work_file(ctx, root, &path) else {
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
            if let Err(error) = write_vfs(ctx, &git_path(root, &format!("objects/{hash}")), &data) {
                io.err
                    .extend_from_slice(format!("git add: {error}\n").as_bytes());
                return 1;
            }
            index.insert(path, hash.clone());
        } else {
            index.remove(&path);
        }
    }
    let data = serialize_tree(index);
    if let Err(error) = write_vfs(ctx, &git_path(root, INDEX), &data) {
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
    let head_reference = head_reference(ctx, &root);
    let reference = git_path(&root, head_reference.as_deref().unwrap_or(HEAD));
    if let Err(error) = write_vfs(ctx, &reference, format!("{id}\n").as_bytes()) {
        io.err
            .extend_from_slice(format!("git commit: {error}\n").as_bytes());
        return 1;
    }
    let branch = current_branch(ctx, &root).unwrap_or_else(|| "detached HEAD".to_string());
    let prefix = if parent.is_none() { "root-commit " } else { "" };
    io.out
        .extend_from_slice(format!("[{branch} {prefix}{}] {message}\n", &id[..7]).as_bytes());
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

fn emit_whitespace_errors(io: &mut Io, path: &str, old: Option<&[u8]>, new: Option<&[u8]>) -> bool {
    let old: Vec<String> = old
        .map(|bytes| {
            String::from_utf8_lossy(bytes)
                .lines()
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let new: Vec<String> = new
        .map(|bytes| {
            String::from_utf8_lossy(bytes)
                .lines()
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let prefix = old
        .iter()
        .zip(&new)
        .take_while(|(left, right)| left == right)
        .count();
    let suffix = old[prefix..]
        .iter()
        .rev()
        .zip(new[prefix..].iter().rev())
        .take_while(|(left, right)| left == right)
        .count();
    let changed_end = new.len().saturating_sub(suffix);
    let mut found = false;
    for (offset, line) in new[prefix..changed_end].iter().enumerate() {
        if line.ends_with([' ', '\t']) {
            found = true;
            io.out.extend_from_slice(
                format!(
                    "{path}:{}: trailing whitespace.\n+{line}\n",
                    prefix + offset + 1
                )
                .as_bytes(),
            );
        }
    }
    found
}

fn git_diff(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut cached = false;
    let mut name_only = false;
    let mut check = false;
    let mut paths = Vec::new();
    for arg in args {
        match arg.as_str() {
            "--cached" | "--staged" => cached = true,
            "--name-only" => name_only = true,
            "--check" => check = true,
            arg if arg.starts_with('-') => {
                return usage(io, &format!("unsupported diff option: {arg}"))
            }
            path => paths.push(path.to_string()),
        }
    }
    if check && name_only {
        return usage(io, "--check cannot be combined with --name-only");
    }
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    let index = load_index(ctx, &root).unwrap_or_default();
    let (work, work_reserved) = if cached {
        (BTreeMap::new(), 0)
    } else {
        match collect_working_tree(ctx, &root) {
            Ok(snapshot) => (snapshot.files, snapshot.reserved_memory),
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
    let mut whitespace_error = false;
    for path in names {
        if !selected(&path) || left_tree.get(&path) == right_tree.get(&path) {
            continue;
        }
        if name_only {
            io.out.extend_from_slice(format!("{path}\n").as_bytes());
        } else {
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
            if check {
                whitespace_error |=
                    emit_whitespace_errors(io, &path, old.as_deref(), new.as_deref());
            } else {
                emit_diff(io, &path, old.as_deref(), new.as_deref());
            }
        }
    }
    ctx.resources.release_memory(work_reserved);
    if whitespace_error {
        2
    } else {
        0
    }
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
        [flag] if flag == "--git-dir" => {
            io.out
                .extend_from_slice(format!("{}\n", path_join(&root, GIT_DIR)).as_bytes());
            0
        }
        [flag] if flag == "--show-prefix" => {
            let prefix = relative_path(&root, &ctx.cwd).unwrap_or_default();
            if prefix.is_empty() {
                io.out.push(b'\n');
            } else {
                io.out.extend_from_slice(format!("{prefix}/\n").as_bytes());
            }
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
        [flag, revision] if flag == "--abbrev-ref" && revision == "HEAD" => {
            let Some(branch) = current_branch(ctx, &root) else {
                io.out.extend_from_slice(b"HEAD\n");
                return 0;
            };
            io.out.extend_from_slice(format!("{branch}\n").as_bytes());
            0
        }
        [single] => {
            let Some(commit) = resolve_revision(ctx, &root, single) else {
                io.err.extend_from_slice(
                    format!("fatal: ambiguous argument '{single}'\n").as_bytes(),
                );
                return 128;
            };
            io.out.extend_from_slice(format!("{commit}\n").as_bytes());
            0
        }
        _ => usage(io, "unsupported rev-parse form"),
    }
}

fn branch_names(interp: &Interp, root: &str) -> Vec<String> {
    let prefix = git_path(root, "refs/heads");
    let mut branches = interp
        .vfs
        .walk(&prefix)
        .into_iter()
        .filter(|path| interp.vfs.is_file("/", path))
        .filter_map(|path| path.strip_prefix(&format!("{prefix}/")).map(str::to_string))
        .collect::<Vec<_>>();
    branches.sort();
    branches
}

fn valid_branch(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('-')
        && !name.starts_with('/')
        && !name.ends_with('/')
        && !name.contains("..")
        && !name.contains(char::is_whitespace)
        && !name.contains(['~', '^', ':', '?', '*', '[', '\\'])
}

fn git_branch(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    if args == ["--show-current"] {
        if let Some(branch) = current_branch(ctx, &root) {
            io.out.extend_from_slice(format!("{branch}\n").as_bytes());
        }
        return 0;
    }
    if args.is_empty() {
        let current = current_branch(ctx, &root);
        for branch in branch_names(ctx, &root) {
            let marker = if current.as_deref() == Some(branch.as_str()) {
                '*'
            } else {
                ' '
            };
            io.out
                .extend_from_slice(format!("{marker} {branch}\n").as_bytes());
        }
        return 0;
    }
    if matches!(args.first().map(String::as_str), Some("-d" | "-D")) {
        let [_, name] = args else {
            return usage(io, "usage: git branch (-d|-D) NAME");
        };
        if current_branch(ctx, &root).as_deref() == Some(name) {
            io.err
                .extend_from_slice(b"error: cannot delete the checked out branch\n");
            return 1;
        }
        let path = git_path(&root, &format!("refs/heads/{name}"));
        return match ctx.vfs.remove_file("/", &path) {
            Ok(()) => 0,
            Err(_) => {
                io.err
                    .extend_from_slice(format!("error: branch '{name}' not found\n").as_bytes());
                1
            }
        };
    }
    if args.len() > 2 || !valid_branch(&args[0]) {
        return usage(io, "usage: git branch NAME [START_POINT]");
    }
    let start = args.get(1).map_or("HEAD", String::as_str);
    let Some(commit) = resolve_revision(ctx, &root, start) else {
        io.err
            .extend_from_slice(format!("fatal: not a valid object name: '{start}'\n").as_bytes());
        return 128;
    };
    let reference = git_path(&root, &format!("refs/heads/{}", args[0]));
    if ctx.vfs.is_file("/", &reference) {
        io.err.extend_from_slice(
            format!("fatal: a branch named '{}' already exists\n", args[0]).as_bytes(),
        );
        return 128;
    }
    if let Some(parent) = parent_of(&reference) {
        if let Err(error) = ctx.vfs.mkdir_all("/", &parent) {
            io.err
                .extend_from_slice(format!("git branch: {error}\n").as_bytes());
            return 1;
        }
    }
    write_vfs(ctx, &reference, format!("{commit}\n").as_bytes()).map_or_else(
        |error| {
            io.err
                .extend_from_slice(format!("git branch: {error}\n").as_bytes());
            1
        },
        |()| 0,
    )
}

fn working_tree_clean(ctx: &mut CommandContext<'_>, root: &str) -> bool {
    let index = load_index(ctx, root).unwrap_or_default();
    let Ok(work) = collect_working_tree(ctx, root) else {
        return false;
    };
    let clean = work.files == index && index == head_tree(ctx, root);
    ctx.resources.release_memory(work.reserved_memory);
    clean
}

fn replace_work_tree(
    ctx: &mut CommandContext<'_>,
    root: &str,
    old: &BTreeMap<String, String>,
    new: &BTreeMap<String, String>,
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
                    .map_err(|e| e.to_string())?;
            }
        }
        for (path, hash) in new {
            let data = read_blob(ctx, root, hash)
                .ok_or_else(|| format!("missing blob {hash} for {path}"))?;
            let absolute = path_join(root, path);
            if let Some(parent) = parent_of(&absolute) {
                ctx.vfs.mkdir_all("/", &parent).map_err(|e| e.to_string())?;
            }
            ctx.sync_vfs_time();
            ctx.vfs
                .write("/", &absolute, &data, 0o644)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    })();
    if result.is_err() {
        ctx.vfs = before;
    }
    ctx.resources.release_memory(reserved);
    result
}

fn switch_to(ctx: &mut CommandContext<'_>, root: &str, branch: &str, io: &mut Io) -> i32 {
    let Some(commit) = read_reference(ctx, root, &format!("refs/heads/{branch}")) else {
        io.err
            .extend_from_slice(format!("fatal: invalid reference: {branch}\n").as_bytes());
        return 128;
    };
    if !working_tree_clean(ctx, root) {
        io.err.extend_from_slice(
            b"error: local changes would be overwritten by switch; commit or restore them first\n",
        );
        return 1;
    }
    let old = head_tree(ctx, root);
    let Some(new) = commit_tree(ctx, root, &commit) else {
        return usage(io, "target branch has an invalid commit");
    };
    if let Err(error) = replace_work_tree(ctx, root, &old, &new) {
        io.err
            .extend_from_slice(format!("git switch: {error}\n").as_bytes());
        return 1;
    }
    if write_vfs(ctx, &git_path(root, INDEX), &serialize_tree(&new)).is_err()
        || write_vfs(
            ctx,
            &git_path(root, HEAD),
            format!("ref: refs/heads/{branch}\n").as_bytes(),
        )
        .is_err()
    {
        return 1;
    }
    io.out
        .extend_from_slice(format!("Switched to branch '{branch}'\n").as_bytes());
    0
}

fn git_switch(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    let (create, branch) = match args {
        [branch] => (false, branch.as_str()),
        [flag, branch] if flag == "-c" || flag == "--create" => (true, branch.as_str()),
        _ => return usage(io, "usage: git switch [-c] BRANCH"),
    };
    if create {
        if !working_tree_clean(ctx, &root) {
            io.err.extend_from_slice(
                b"error: local changes would be overwritten by switch; commit or restore them first\n",
            );
            return 1;
        }
        let status = git_branch(ctx, &[branch.to_string()], io);
        if status != 0 {
            return status;
        }
    }
    switch_to(ctx, &root, branch, io)
}

fn git_checkout(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    match args {
        [flag, branch] if flag == "-b" => git_switch(ctx, &["-c".into(), branch.clone()], io),
        [branch] => git_switch(ctx, std::slice::from_ref(branch), io),
        _ => usage(io, "usage: git checkout [-b] BRANCH"),
    }
}

fn git_log(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut oneline = false;
    let mut limit = 100_usize;
    let mut revision = "HEAD";
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--oneline" => oneline = true,
            "-n" | "--max-count" => {
                index += 1;
                let Some(value) = args.get(index).and_then(|value| value.parse().ok()) else {
                    return usage(io, "log count must be a non-negative integer");
                };
                limit = value;
            }
            value if value.starts_with("-n") && value.len() > 2 => {
                let Ok(value) = value[2..].parse() else {
                    return usage(io, "log count must be a non-negative integer");
                };
                limit = value;
            }
            value
                if value.len() > 1
                    && value.starts_with('-')
                    && value[1..].bytes().all(|byte| byte.is_ascii_digit()) =>
            {
                let Ok(value) = value[1..].parse() else {
                    return usage(io, "log count must be a non-negative integer");
                };
                limit = value;
            }
            value if value.starts_with('-') => return usage(io, "unsupported log option"),
            value => revision = value,
        }
        index += 1;
    }
    let Some(mut commit) = resolve_revision(ctx, &root, revision) else {
        return usage(io, "unknown revision");
    };
    for _ in 0..limit.min(10_000) {
        let message = commit_message(ctx, &root, &commit).unwrap_or_default();
        if oneline {
            io.out.extend_from_slice(
                format!(
                    "{} {}\n",
                    &commit[..7],
                    message.lines().next().unwrap_or("")
                )
                .as_bytes(),
            );
        } else {
            io.out.extend_from_slice(
                format!(
                    "commit {commit}\n\n    {}\n\n",
                    message.replace('\n', "\n    ")
                )
                .as_bytes(),
            );
        }
        let Some(parent) = commit_parent(ctx, &root, &commit) else {
            break;
        };
        commit = parent;
    }
    0
}

fn git_show(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    let revision = args.first().map_or("HEAD", String::as_str);
    if args.len() > 1 || revision.starts_with('-') {
        return usage(io, "usage: git show [REVISION]");
    }
    let Some(commit) = resolve_revision(ctx, &root, revision) else {
        return usage(io, "unknown revision");
    };
    let message = commit_message(ctx, &root, &commit).unwrap_or_default();
    io.out
        .extend_from_slice(format!("commit {commit}\n\n    {message}\n\n").as_bytes());
    let tree = commit_tree(ctx, &root, &commit).unwrap_or_default();
    let parent = commit_parent(ctx, &root, &commit)
        .and_then(|parent| commit_tree(ctx, &root, &parent))
        .unwrap_or_default();
    emit_tree_diff(ctx, &root, &parent, &tree, io);
    0
}

fn emit_tree_diff(
    ctx: &Interp,
    root: &str,
    old: &BTreeMap<String, String>,
    new: &BTreeMap<String, String>,
    io: &mut Io,
) {
    let mut paths = BTreeSet::new();
    paths.extend(old.keys().cloned());
    paths.extend(new.keys().cloned());
    for path in paths {
        if old.get(&path) == new.get(&path) {
            continue;
        }
        let before = old.get(&path).and_then(|hash| read_blob(ctx, root, hash));
        let after = new.get(&path).and_then(|hash| read_blob(ctx, root, hash));
        emit_diff(io, &path, before.as_deref(), after.as_deref());
    }
}

fn git_restore(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut staged = false;
    let mut source = None;
    let mut paths = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--staged" => staged = true,
            "--source" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return usage(io, "--source requires a revision");
                };
                source = Some(value.clone());
            }
            "--" => {
                paths.extend_from_slice(&args[index + 1..]);
                break;
            }
            value if value.starts_with('-') => return usage(io, "unsupported restore option"),
            value => paths.push(value.to_string()),
        }
        index += 1;
    }
    if paths.is_empty() {
        return usage(io, "restore requires a pathspec");
    }
    let mut index_tree = load_index(ctx, &root).unwrap_or_default();
    let source_tree = if let Some(revision) = source.as_deref() {
        let Some(commit) = resolve_revision(ctx, &root, revision) else {
            return usage(io, "unknown restore source");
        };
        commit_tree(ctx, &root, &commit).unwrap_or_default()
    } else if staged {
        head_tree(ctx, &root)
    } else {
        index_tree.clone()
    };
    if staged {
        let cwd = ctx.cwd.clone();
        let selected = match selected_paths(&cwd, &root, &paths, &index_tree, &source_tree) {
            Ok(selected) => selected,
            Err(message) => return usage(io, &message),
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
        return write_vfs(ctx, &git_path(&root, INDEX), &serialize_tree(&index_tree))
            .map_or(1, |()| 0);
    }
    let Ok(work) = collect_working_tree(ctx, &root) else {
        return resource_error(ctx);
    };
    let cwd = ctx.cwd.clone();
    let selected = match selected_paths(&cwd, &root, &paths, &source_tree, &work.files) {
        Ok(selected) => selected,
        Err(message) => {
            ctx.resources.release_memory(work.reserved_memory);
            return usage(io, &message);
        }
    };
    let old = work
        .files
        .into_iter()
        .filter(|(path, _)| selected.contains(path))
        .collect();
    let new = source_tree
        .into_iter()
        .filter(|(path, _)| selected.contains(path))
        .collect();
    let status = replace_work_tree(ctx, &root, &old, &new).map_or_else(
        |error| {
            io.err
                .extend_from_slice(format!("git restore: {error}\n").as_bytes());
            1
        },
        |()| 0,
    );
    ctx.resources.release_memory(work.reserved_memory);
    status
}

fn update_head(ctx: &mut CommandContext<'_>, root: &str, commit: &str) -> VfsResult<()> {
    let destination = head_reference(ctx, root).unwrap_or_else(|| HEAD.to_string());
    write_vfs(
        ctx,
        &git_path(root, &destination),
        format!("{commit}\n").as_bytes(),
    )
}

fn git_reset(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(root) = find_repo_root(ctx) else {
        return repo_error(io);
    };
    let mut mode = "--mixed";
    let mut revision = "HEAD";
    let mut saw_revision = false;
    for argument in args {
        match argument.as_str() {
            "--soft" | "--mixed" | "--hard" => mode = argument,
            value if value.starts_with('-') => return usage(io, "unsupported reset option"),
            value if !saw_revision => {
                revision = value;
                saw_revision = true;
            }
            _ => return usage(io, "usage: git reset [--soft|--mixed|--hard] [REVISION]"),
        }
    }
    let Some(commit) = resolve_revision(ctx, &root, revision) else {
        return usage(io, "unknown reset revision");
    };
    let tree = commit_tree(ctx, &root, &commit).unwrap_or_default();
    if mode == "--hard" {
        // A hard reset removes paths known by either HEAD or the index while preserving untracked
        // files, matching the boundary agents rely on when discarding staged additions.
        let mut old = head_tree(ctx, &root);
        old.extend(load_index(ctx, &root).unwrap_or_default());
        if let Err(error) = replace_work_tree(ctx, &root, &old, &tree) {
            io.err
                .extend_from_slice(format!("git reset: {error}\n").as_bytes());
            return 1;
        }
    }
    if mode != "--soft" && write_vfs(ctx, &git_path(&root, INDEX), &serialize_tree(&tree)).is_err()
    {
        return 1;
    }
    if update_head(ctx, &root, &commit).is_err() {
        return 1;
    }
    0
}
