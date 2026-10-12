//! In-memory virtual filesystem.
//!
//! The VFS is the single source of truth for all simulated file state. It combines an
//! immutable installed tree and a private overlay keyed by clean absolute paths. Tombstones
//! hide deleted installed nodes, and shared file buffers copy only on content mutation.
//! Directory traversal merges both layers in path order.
//! Node identities are allocated monotonically and survive rename, open-unlink and snapshots;
//! copying or replacing a node allocates a new identity independently of its pathname.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub type Mode = u32;

/// Content work whose first mutation may detach an installed or snapshot-shared buffer.
pub(crate) enum FileMutation {
    Write,
    Resize(usize),
}

/// Native program image stored by identity rather than pretending to contain machine bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeProgram {
    True,
    False,
    Pwd,
    Yes,
    Cat,
    Tee,
    Head,
    Xargs,
    Find,
    Env,
    Sleep,
    Usleep,
    Timeout,
    Nohup,
    Nice,
    Make,
    /// Process-scoped Rust entrypoint identified by its registered virtual path.
    Registered(&'static str),
    /// Older command body still executed through the shell dispatcher by basename.
    LegacyRegistered(&'static str),
}

impl NativeProgram {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "true" => Some(Self::True),
            "false" => Some(Self::False),
            "pwd" => Some(Self::Pwd),
            "yes" => Some(Self::Yes),
            "cat" => Some(Self::Cat),
            "tee" => Some(Self::Tee),
            "head" => Some(Self::Head),
            "xargs" => Some(Self::Xargs),
            "find" => Some(Self::Find),
            "env" => Some(Self::Env),
            "sleep" => Some(Self::Sleep),
            "usleep" => Some(Self::Usleep),
            "timeout" => Some(Self::Timeout),
            "nohup" => Some(Self::Nohup),
            "nice" => Some(Self::Nice),
            "make" => Some(Self::Make),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::True => "true",
            Self::False => "false",
            Self::Pwd => "pwd",
            Self::Yes => "yes",
            Self::Cat => "cat",
            Self::Tee => "tee",
            Self::Head => "head",
            Self::Xargs => "xargs",
            Self::Find => "find",
            Self::Env => "env",
            Self::Sleep => "sleep",
            Self::Usleep => "usleep",
            Self::Timeout => "timeout",
            Self::Nohup => "nohup",
            Self::Nice => "nice",
            Self::Make => "make",
            Self::Registered(path) => path.rsplit('/').next().unwrap_or(path),
            Self::LegacyRegistered(name) => name,
        }
    }
}

#[derive(Clone, Debug)]
pub enum NodeKind {
    File(Arc<Vec<u8>>),
    Dir,
    Symlink(String),
    NativeExecutable(NativeProgram),
}

#[derive(Clone, Debug)]
pub struct Node {
    /// Stable identity allocated by the owning virtual filesystem; zero is reserved.
    pub inode: u64,
    pub kind: NodeKind,
    pub mode: Mode,
    pub uid: u32,
    pub gid: u32,
    /// virtual modification time in milliseconds (from the simulated clock)
    pub mtime: u64,
}

impl Node {
    fn dir(inode: u64, mode: Mode, mtime: u64) -> Self {
        Node {
            inode,
            kind: NodeKind::Dir,
            mode,
            uid: 0,
            gid: 0,
            mtime,
        }
    }
    fn file(inode: u64, data: Vec<u8>, mode: Mode, mtime: u64) -> Self {
        Node {
            inode,
            kind: NodeKind::File(Arc::new(data)),
            mode,
            uid: 0,
            gid: 0,
            mtime,
        }
    }
}

#[derive(Debug)]
pub enum VfsError {
    NotFound(String),
    NotADir(String),
    IsADir(String),
    NotEmpty(String),
    Exists(String),
    Loop(String),
    Invalid(String),
    /// A path component longer than [`NAME_MAX`] or a path longer than [`PATH_MAX`].
    NameTooLong(String),
    NoSpace,
    TooLarge {
        path: String,
        limit: usize,
    },
    ReadOnly(String),
}

impl VfsError {
    /// The `strerror`-style reason without the path, for diagnostics that name the operand
    /// themselves, such as `find: 'missing': No such file or directory`.
    pub fn reason(&self) -> &'static str {
        match self {
            VfsError::NotFound(_) => "No such file or directory",
            VfsError::NotADir(_) => "Not a directory",
            VfsError::IsADir(_) => "Is a directory",
            VfsError::NotEmpty(_) => "Directory not empty",
            VfsError::Exists(_) => "File exists",
            VfsError::Loop(_) => "Too many levels of symbolic links",
            VfsError::Invalid(_) => "Invalid argument",
            VfsError::NameTooLong(_) => "File name too long",
            VfsError::NoSpace => "No space left on device",
            VfsError::TooLarge { .. } => "File too large",
            VfsError::ReadOnly(_) => "Read-only filesystem",
        }
    }
}

impl std::fmt::Display for VfsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VfsError::NotFound(p) => write!(f, "No such file or directory: {p}"),
            VfsError::NotADir(p) => write!(f, "Not a directory: {p}"),
            VfsError::IsADir(p) => write!(f, "Is a directory: {p}"),
            VfsError::NotEmpty(p) => write!(f, "Directory not empty: {p}"),
            VfsError::Exists(p) => write!(f, "File exists: {p}"),
            VfsError::Loop(p) => write!(f, "Too many levels of symbolic links: {p}"),
            VfsError::Invalid(p) => write!(f, "Invalid argument: {p}"),
            VfsError::NameTooLong(p) => write!(f, "File name too long: {p}"),
            VfsError::NoSpace => write!(f, "No space left on device"),
            VfsError::TooLarge { path, limit } => {
                write!(f, "file too large: {path} (limit {limit} bytes)")
            }
            VfsError::ReadOnly(path) => write!(f, "Read-only filesystem: {path}"),
        }
    }
}

pub type Result<T> = std::result::Result<T, VfsError>;

/// Immutable installed filesystem shared by independent execution environments.
pub struct FrozenTree {
    nodes: BTreeMap<String, Node>,
    baseline_dirs: BTreeSet<String>,
    baseline_native: BTreeMap<String, NativeProgram>,
    next_inode: u64,
    disk_used: u64,
}

impl FrozenTree {
    /// Logical usage charged to every environment using this installation.
    pub fn disk_used(&self) -> u64 {
        self.disk_used
    }

    /// Conservative retained size for the installation cache, including node metadata.
    pub fn retained_bytes(&self) -> u64 {
        self.nodes.iter().fold(0u64, |total, (path, node)| {
            let payload = match &node.kind {
                NodeKind::File(data) => data.capacity(),
                NodeKind::Symlink(target) => target.capacity(),
                NodeKind::Dir | NodeKind::NativeExecutable(_) => 0,
            };
            total
                .saturating_add(NODE_OVERHEAD)
                .saturating_add(path.capacity() as u64)
                .saturating_add(payload as u64)
        })
    }
}

#[derive(Clone)]
enum OverlayEntry {
    Present(Node),
    Deleted,
}

#[derive(Clone)]
pub struct Vfs {
    base: Arc<FrozenTree>,
    overlay: BTreeMap<String, OverlayEntry>,
    /// Visible nodes, so callers can prepay directory scans before traversing the merged tree.
    node_count: usize,
    /// Unlinked files retained by open descriptions until their last close.
    orphaned: BTreeMap<u64, Node>,
    next_orphan: u64,
    next_inode: u64,
    /// Wall-clock timestamp assigned to subsequent content mutations.  The environment updates
    /// this immediately before an effect; the VFS never consults the host clock.
    mutation_time_ms: u64,
    disk_limit: u64,
    /// Directory nodes supplied by the base image rather than created by simulated actions.
    baseline_dirs: std::collections::BTreeSet<String>,
    /// Native images supplied by the base image do not consume writable quota.
    baseline_native: BTreeMap<String, NativeProgram>,
    disk_used: u64,
    disk_peak: u64,
    read_bytes: Cell<u64>,
}

impl Default for Vfs {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize an absolute path lexically (resolve "." and "..", collapse "//"),
/// WITHOUT resolving symlinks. Input must be absolute. Result has no trailing
/// slash except for the root "/".
pub fn normalize(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for comp in path.split('/') {
        match comp {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            c => out.push(c),
        }
    }
    if out.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", out.join("/"))
    }
}

/// Longest path a mutation may create, as Linux's `PATH_MAX` bounds it. Node keys are full
/// paths, so without this bound a deep directory chain costs the sum of all its prefixes.
pub const PATH_MAX: usize = 4096;
/// Longest single path component, as Linux's `NAME_MAX` bounds it.
pub const NAME_MAX: usize = 255;

/// Reject a path to be created when it or one of its components exceeds the Linux limits.
fn check_path_limits(abs: &str) -> Result<()> {
    if abs.len() >= PATH_MAX || abs.split('/').any(|component| component.len() > NAME_MAX) {
        return Err(VfsError::NameTooLong(abs.to_string()));
    }
    Ok(())
}

fn reject_pseudo_mutation(path: &str) -> Result<()> {
    if ["/proc", "/dev"]
        .iter()
        .any(|root| path == *root || path.starts_with(&format!("{root}/")))
    {
        Err(VfsError::ReadOnly(path.to_string()))
    } else {
        Ok(())
    }
}

/// Join a (possibly relative) path onto cwd and normalize lexically.
pub fn resolve_against(cwd: &str, path: &str) -> String {
    if path.starts_with('/') {
        normalize(path)
    } else {
        normalize(&format!("{cwd}/{path}"))
    }
}

pub fn parent_of(path: &str) -> Option<String> {
    if path == "/" {
        return None;
    }
    match path.rfind('/') {
        Some(0) => Some("/".to_string()),
        Some(i) => Some(path[..i].to_string()),
        None => None,
    }
}

pub fn basename(path: &str) -> &str {
    if path == "/" {
        return "/";
    }
    match path.rfind('/') {
        Some(i) => &path[i + 1..],
        None => path,
    }
}

impl Vfs {
    /// Bytes copied before a guest mutation, excluding growth already charged by its caller.
    /// Base nodes require copying even when their payload has only one Arc owner: copy-up
    /// introduces the second owner, and the installed tree must remain immutable.
    pub(crate) fn file_cow_bytes(
        &self,
        cwd: &str,
        path: &str,
        mutation: FileMutation,
    ) -> Result<u64> {
        let target = self.write_target(cwd, path)?;
        let Some(node) = self.raw_get(&target) else {
            return Ok(0);
        };
        let shared = !self.overlay.contains_key(&target);
        Ok(cow_copy_bytes(node, shared, mutation))
    }

    pub(crate) fn orphan_cow_bytes(&self, id: u64, mutation: FileMutation) -> Result<u64> {
        Ok(cow_copy_bytes(self.orphan_metadata(id)?, false, mutation))
    }
    /// Consume installed filesystem state, excluding open descriptions and execution counters.
    pub fn freeze(self) -> Arc<FrozenTree> {
        let disk_used = self
            .measured_usage()
            .saturating_sub(self.orphaned.values().fold(0u64, |total, node| {
                total
                    .saturating_add(NODE_OVERHEAD)
                    .saturating_add(node_payload_len(node))
            }));
        let nodes = self
            .all_paths()
            .map(|(path, node)| (path.clone(), node.clone()))
            .collect();
        Arc::new(FrozenTree {
            nodes,
            baseline_dirs: self.baseline_dirs,
            baseline_native: self.baseline_native,
            next_inode: self.next_inode,
            disk_used,
        })
    }

    /// Start a private writable view, charging the installed logical usage against its quota.
    pub fn from_base(base: Arc<FrozenTree>, disk_limit: u64) -> Result<Self> {
        if base.disk_used > disk_limit {
            return Err(VfsError::NoSpace);
        }
        Ok(Self {
            overlay: BTreeMap::new(),
            node_count: base.nodes.len(),
            orphaned: BTreeMap::new(),
            next_orphan: 1,
            next_inode: base.next_inode,
            mutation_time_ms: 0,
            disk_limit,
            baseline_dirs: base.baseline_dirs.clone(),
            baseline_native: base.baseline_native.clone(),
            disk_used: base.disk_used,
            disk_peak: base.disk_used,
            read_bytes: Cell::new(0),
            base,
        })
    }

    fn contains_node(&self, path: &str) -> bool {
        self.raw_get(path).is_some()
    }

    fn node_mut(&mut self, path: &str) -> Option<&mut Node> {
        if !self.overlay.contains_key(path) {
            let node = self.base.nodes.get(path)?.clone();
            self.overlay
                .insert(path.to_string(), OverlayEntry::Present(node));
        }
        match self.overlay.get_mut(path) {
            Some(OverlayEntry::Present(node)) => Some(node),
            _ => None,
        }
    }

    fn insert_node(&mut self, path: String, node: Node) {
        if !self.contains_node(&path) {
            self.node_count = self.node_count.saturating_add(1);
        }
        self.overlay.insert(path, OverlayEntry::Present(node));
    }

    fn remove_node(&mut self, path: &str) -> Option<Node> {
        let node = self.raw_get(path)?.clone();
        self.node_count = self.node_count.saturating_sub(1);
        if self.base.nodes.contains_key(path) {
            self.overlay.insert(path.to_string(), OverlayEntry::Deleted);
        } else {
            self.overlay.remove(path);
        }
        Some(node)
    }

    pub fn new() -> Self {
        Self::with_disk_limit(u64::MAX)
    }

    pub fn with_disk_limit(disk_limit: u64) -> Self {
        let mut nodes = BTreeMap::new();
        nodes.insert("/".to_string(), Node::dir(1, 0o755, 0));
        let disk_used = logical_usage(nodes.iter());
        Vfs {
            base: Arc::new(FrozenTree {
                nodes,
                baseline_dirs: BTreeSet::new(),
                baseline_native: BTreeMap::new(),
                next_inode: 2,
                disk_used,
            }),
            overlay: BTreeMap::new(),
            node_count: 1,
            orphaned: BTreeMap::new(),
            next_orphan: 1,
            next_inode: 2,
            mutation_time_ms: 0,
            disk_limit,
            baseline_dirs: std::collections::BTreeSet::new(),
            baseline_native: BTreeMap::new(),
            disk_used,
            disk_peak: disk_used,
            read_bytes: Cell::new(0),
        }
    }

    fn check_inode_capacity(&self, count: usize) -> Result<()> {
        self.next_inode
            .checked_add(count as u64)
            .ok_or(VfsError::NoSpace)?;
        Ok(())
    }

    fn allocate_inode(&mut self) -> Result<u64> {
        self.check_inode_capacity(1)?;
        let inode = self.next_inode;
        self.next_inode += 1;
        Ok(inode)
    }

    pub fn disk_used(&self) -> u64 {
        self.disk_used
    }

    /// The quota file contents count against; `u64::MAX` when no limit was configured.
    pub fn disk_limit(&self) -> u64 {
        self.disk_limit
    }

    pub fn disk_peak(&self) -> u64 {
        self.disk_peak
    }

    pub fn read_bytes(&self) -> u64 {
        self.read_bytes.get()
    }

    /// Set the timestamp source for future mutations.  This explicit handoff keeps VFS fixtures
    /// independently constructible while making the owning environment authoritative for time.
    pub fn set_mutation_time(&mut self, unix_ms: u64) {
        self.mutation_time_ms = unix_ms;
    }

    pub fn mutation_time(&self) -> u64 {
        self.mutation_time_ms
    }

    /// Add immutable image-layout directories before simulated execution begins.
    ///
    /// Base-image nodes do not consume the writable disk quota. Subsequent mutations account for
    /// all logical usage added beyond this initial layout.
    pub(crate) fn seed_dirs<const N: usize>(&mut self, paths: [&str; N]) {
        for path in paths {
            let path = normalize(path);
            let inode = self.allocate_inode().expect("base image inode capacity");
            self.insert_node(path.clone(), Node::dir(inode, 0o755, self.mutation_time_ms));
            self.baseline_dirs.insert(path);
        }
        self.refresh_usage();
        self.disk_peak = self.disk_peak.max(self.disk_used);
    }

    /// The usage after adding `added` bytes, or `NoSpace` when it would exceed the limit.
    /// Mutations check this before changing any node, so no rollback is needed.
    fn usage_after_adding(&self, added: u64) -> Result<u64> {
        let used = self.disk_used.saturating_add(added);
        if used > self.disk_limit {
            Err(VfsError::NoSpace)
        } else {
            Ok(used)
        }
    }

    /// The usage after a change of `delta` bytes, which may be negative.
    fn usage_after_delta(&self, delta: i128) -> Result<u64> {
        let used = i128::from(self.disk_used).saturating_add(delta);
        let used = u64::try_from(used.max(0)).unwrap_or(u64::MAX);
        if used > self.disk_limit {
            Err(VfsError::NoSpace)
        } else {
            Ok(used)
        }
    }

    fn refresh_usage(&mut self) {
        self.disk_used = self.measured_usage();
    }

    fn node_usage(&self, path: &str, node: &Node) -> u64 {
        if self.baseline_dirs.contains(path) && matches!(node.kind, NodeKind::Dir) {
            return 0;
        }
        if matches!(node.kind, NodeKind::NativeExecutable(program) if self.baseline_native.get(path) == Some(&program))
        {
            return 0;
        }
        NODE_OVERHEAD.saturating_add(node_payload_len(node))
    }

    fn projected_usage_after_replacement(&self, path: &str, replacement_usage: u64) -> Result<u64> {
        let previous = self
            .raw_get(path)
            .map_or(0, |node| self.node_usage(path, node));
        let used = self
            .disk_used
            .saturating_sub(previous)
            .saturating_add(replacement_usage);
        if used > self.disk_limit {
            Err(VfsError::NoSpace)
        } else {
            Ok(used)
        }
    }

    fn record_usage(&mut self, used: u64) {
        self.disk_used = used;
        self.disk_peak = self.disk_peak.max(used);
    }

    fn missing_directories(&self, abs: &str) -> Result<Vec<String>> {
        let components = abs.split('/').filter(|component| !component.is_empty());
        let mut planned = BTreeSet::new();
        let mut additions = Vec::new();
        let mut current = "/".to_string();
        for component in components {
            let next = if current == "/" {
                format!("/{component}")
            } else {
                format!("{current}/{component}")
            };
            let real = self.realpath(&next, true).unwrap_or(next);
            match self.raw_get(&real) {
                Some(Node {
                    kind: NodeKind::Dir,
                    ..
                }) => {}
                Some(_) => return Err(VfsError::NotADir(real)),
                None if planned.insert(real.clone()) => additions.push(real.clone()),
                None => {}
            }
            current = real;
        }
        Ok(additions)
    }

    fn directory_addition_usage(&self, additions: &[String]) -> u64 {
        additions
            .iter()
            .filter(|path| !self.baseline_dirs.contains(path.as_str()))
            .fold(0u64, |usage, _| usage.saturating_add(NODE_OVERHEAD))
    }

    fn insert_directories(&mut self, additions: Vec<String>) -> Result<()> {
        self.check_inode_capacity(additions.len())?;
        for path in additions {
            let inode = self.allocate_inode()?;
            self.insert_node(path, Node::dir(inode, 0o755, self.mutation_time_ms));
        }
        Ok(())
    }

    fn measured_usage(&self) -> u64 {
        let present_baseline_dirs = self
            .baseline_dirs
            .iter()
            .filter(|path| {
                matches!(
                    self.raw_get(path.as_str()),
                    Some(Node {
                        kind: NodeKind::Dir,
                        ..
                    })
                )
            })
            .count() as u64;
        let present_baseline_native = self
            .baseline_native
            .iter()
            .filter(|(path, program)| {
                matches!(
                    self.raw_get(path.as_str()),
                    Some(Node {
                        kind: NodeKind::NativeExecutable(actual),
                        ..
                    }) if actual == *program
                )
            })
            .count() as u64;
        logical_usage(self.all_paths())
            .saturating_sub(present_baseline_dirs.saturating_mul(NODE_OVERHEAD))
            .saturating_sub(present_baseline_native.saturating_mul(NODE_OVERHEAD))
            .saturating_add(self.orphaned.values().fold(0_u64, |total, node| {
                total.saturating_add(NODE_OVERHEAD.saturating_add(node_payload_len(node)))
            }))
    }

    // ---- low level ----

    pub fn raw_get(&self, abs: &str) -> Option<&Node> {
        match self.overlay.get(abs) {
            Some(OverlayEntry::Present(node)) => Some(node),
            Some(OverlayEntry::Deleted) => None,
            None => self.base.nodes.get(abs),
        }
    }

    /// Resolve symlinks along a normalized absolute path, returning the final real path.
    /// `follow_final` controls whether a trailing symlink is itself dereferenced.
    pub fn realpath(&self, abs: &str, follow_final: bool) -> Result<String> {
        self.realpath_inner(abs, follow_final, 0)
    }

    fn realpath_inner(&self, abs: &str, follow_final: bool, depth: usize) -> Result<String> {
        if depth > 40 {
            return Err(VfsError::Loop(abs.to_string()));
        }
        if abs == "/" {
            return Ok("/".to_string());
        }
        let parent = parent_of(abs).unwrap_or_else(|| "/".to_string());
        let real_parent = self.realpath_inner(&parent, true, depth + 1)?;
        let name = basename(abs);
        let candidate = if real_parent == "/" {
            format!("/{name}")
        } else {
            format!("{real_parent}/{name}")
        };
        match self.raw_get(&candidate) {
            Some(Node {
                kind: NodeKind::Symlink(target),
                ..
            }) if follow_final => {
                let next = resolve_against(&real_parent, target);
                self.realpath_inner(&next, true, depth + 1)
            }
            _ => Ok(candidate),
        }
    }

    pub fn exists(&self, cwd: &str, path: &str) -> bool {
        let abs = resolve_against(cwd, path);
        self.realpath(&abs, true)
            .map(|p| self.contains_node(&p))
            .unwrap_or(false)
    }

    pub fn lexists(&self, cwd: &str, path: &str) -> bool {
        let abs = resolve_against(cwd, path);
        self.realpath(&abs, false)
            .map(|p| self.contains_node(&p))
            .unwrap_or(false)
    }

    pub fn is_dir(&self, cwd: &str, path: &str) -> bool {
        let abs = resolve_against(cwd, path);
        match self.realpath(&abs, true) {
            Ok(p) => matches!(
                self.raw_get(&p),
                Some(Node {
                    kind: NodeKind::Dir,
                    ..
                })
            ),
            Err(_) => false,
        }
    }

    pub fn is_file(&self, cwd: &str, path: &str) -> bool {
        let abs = resolve_against(cwd, path);
        match self.realpath(&abs, true) {
            Ok(p) => matches!(
                self.raw_get(&p),
                Some(Node {
                    kind: NodeKind::File(_) | NodeKind::NativeExecutable(_),
                    ..
                })
            ),
            Err(_) => false,
        }
    }

    /// Return a regular file's byte length without cloning its contents.
    pub fn file_len(&self, cwd: &str, path: &str) -> Result<usize> {
        let abs = resolve_against(cwd, path);
        let real = self.realpath(&abs, true)?;
        match self.raw_get(&real) {
            Some(Node {
                kind: NodeKind::File(data),
                ..
            }) => Ok(data.len()),
            Some(Node {
                kind: NodeKind::NativeExecutable(_),
                ..
            }) => Ok(0),
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => Err(VfsError::IsADir(path.to_string())),
            _ => Err(VfsError::NotFound(path.to_string())),
        }
    }

    pub fn is_symlink(&self, cwd: &str, path: &str) -> bool {
        let abs = resolve_against(cwd, path);
        match self.realpath(&abs, false) {
            Ok(p) => matches!(
                self.raw_get(&p),
                Some(Node {
                    kind: NodeKind::Symlink(_),
                    ..
                })
            ),
            Err(_) => false,
        }
    }

    // ---- reads ----

    pub fn read(&self, cwd: &str, path: &str) -> Result<Vec<u8>> {
        let abs = resolve_against(cwd, path);
        let real = self.realpath(&abs, true)?;
        match self.raw_get(&real) {
            Some(Node {
                kind: NodeKind::File(d),
                ..
            }) => {
                self.read_bytes
                    .set(self.read_bytes.get().saturating_add(d.len() as u64));
                Ok(d.as_ref().clone())
            }
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => Err(VfsError::IsADir(path.to_string())),
            Some(Node {
                kind: NodeKind::NativeExecutable(_),
                ..
            }) => Err(VfsError::Invalid(format!(
                "opaque native executable: {path}"
            ))),
            _ => Err(VfsError::NotFound(path.to_string())),
        }
    }

    /// Read a file as exact bytes only when it is within `limit`.
    pub fn read_limited(&self, cwd: &str, path: &str, limit: usize) -> Result<Vec<u8>> {
        self.read_shared_limited(cwd, path, limit)
            .map(|data| data.as_ref().clone())
    }

    /// Retain immutable file bytes within `limit`, counting the same logical read as an owned
    /// read. Later VFS mutations detach this buffer, so consumers may safely cache its identity.
    pub fn read_shared_limited(&self, cwd: &str, path: &str, limit: usize) -> Result<Arc<Vec<u8>>> {
        let abs = resolve_against(cwd, path);
        let real = self.realpath(&abs, true)?;
        match self.raw_get(&real) {
            Some(Node {
                kind: NodeKind::File(data),
                ..
            }) => {
                if data.len() > limit {
                    return Err(VfsError::TooLarge {
                        path: path.to_string(),
                        limit,
                    });
                }
                self.read_bytes
                    .set(self.read_bytes.get().saturating_add(data.len() as u64));
                Ok(data.clone())
            }
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => Err(VfsError::IsADir(path.to_string())),
            Some(Node {
                kind: NodeKind::NativeExecutable(_),
                ..
            }) => Err(VfsError::Invalid(format!(
                "opaque native executable: {path}"
            ))),
            _ => Err(VfsError::NotFound(path.to_string())),
        }
    }

    pub fn read_string(&self, cwd: &str, path: &str) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.read(cwd, path)?).into_owned())
    }

    /// Read a bounded slice of a regular file, even when the whole file exceeds a capture limit.
    pub(crate) fn read_range(
        &self,
        cwd: &str,
        path: &str,
        offset: usize,
        maximum: usize,
    ) -> Result<Vec<u8>> {
        let abs = resolve_against(cwd, path);
        let real = self.realpath(&abs, true)?;
        match self.raw_get(&real) {
            Some(Node {
                kind: NodeKind::File(data),
                ..
            }) => {
                let start = offset.min(data.len());
                let end = start.saturating_add(maximum).min(data.len());
                self.read_bytes
                    .set(self.read_bytes.get().saturating_add((end - start) as u64));
                Ok(data[start..end].to_vec())
            }
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => Err(VfsError::IsADir(path.to_string())),
            Some(Node {
                kind: NodeKind::NativeExecutable(_),
                ..
            }) => Err(VfsError::Invalid(format!(
                "opaque native executable: {path}"
            ))),
            _ => Err(VfsError::NotFound(path.to_string())),
        }
    }

    /// Read a UTF-8-ish file only when it is within `limit` bytes.
    ///
    /// Unlike checking `metadata`, this avoids cloning a potentially enormous file merely to
    /// discover that a caller cannot safely process it.  The read counter is updated only for a
    /// read that is actually returned.
    pub fn read_string_limited(&self, cwd: &str, path: &str, limit: usize) -> Result<String> {
        let abs = resolve_against(cwd, path);
        let real = self.realpath(&abs, true)?;
        match self.raw_get(&real) {
            Some(Node {
                kind: NodeKind::File(data),
                ..
            }) => {
                if data.len() > limit {
                    return Err(VfsError::TooLarge {
                        path: path.to_string(),
                        limit,
                    });
                }
                self.read_bytes
                    .set(self.read_bytes.get().saturating_add(data.len() as u64));
                Ok(String::from_utf8_lossy(data).into_owned())
            }
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => Err(VfsError::IsADir(path.to_string())),
            Some(Node {
                kind: NodeKind::NativeExecutable(_),
                ..
            }) => Err(VfsError::Invalid(format!(
                "opaque native executable: {path}"
            ))),
            _ => Err(VfsError::NotFound(path.to_string())),
        }
    }

    pub fn metadata(&self, cwd: &str, path: &str, follow: bool) -> Result<Node> {
        self.metadata_ref(cwd, path, follow).cloned()
    }

    /// Check a package destination without following links or replacing foreign content.
    /// Existing directories retain their permissions; regular files require equal bytes and
    /// compatible read/execute permissions, preserving existing modes. Only an explicitly
    /// selected original base-image command may be replaced.
    pub(crate) fn package_destination_matches(
        &self,
        path: &str,
        contents: Option<&[u8]>,
        mode: Mode,
        replace_builtin: bool,
    ) -> Result<bool> {
        for parent in std::path::Path::new(path).ancestors().skip(1) {
            let parent = parent.to_str().expect("VFS paths are UTF-8");
            if let Some(node) = self.raw_get(parent) {
                if !matches!(node.kind, NodeKind::Dir) {
                    return Err(VfsError::Invalid(format!(
                        "package destination has a non-directory ancestor: {path}"
                    )));
                }
            }
        }
        if self.realpath(path, false)? != path {
            return Err(VfsError::Invalid(format!(
                "package destination follows a link: {path}"
            )));
        }
        let Some(node) = self.raw_get(path) else {
            return Ok(false);
        };
        match (&node.kind, contents) {
            (NodeKind::Dir, None) => Ok(true),
            (NodeKind::File(existing), Some(incoming))
                if existing.as_slice() == incoming && node.mode & mode & 0o555 == mode & 0o555 =>
            {
                Ok(true)
            }
            (NodeKind::NativeExecutable(program), Some(_))
                if replace_builtin && self.baseline_native.get(path) == Some(program) =>
            {
                Ok(false)
            }
            _ => Err(VfsError::Invalid(format!(
                "package import would overwrite an incompatible VFS node: {path}"
            ))),
        }
    }

    /// Inspect a resolved node without copying its file payload. Kernel metadata conversions
    /// borrow this view only for the duration of a syscall.
    pub(crate) fn metadata_ref(&self, cwd: &str, path: &str, follow: bool) -> Result<&Node> {
        let abs = resolve_against(cwd, path);
        let real = self.realpath(&abs, follow)?;
        self.raw_get(&real)
            .ok_or_else(|| VfsError::NotFound(path.to_string()))
    }

    /// List directory entry names (not including "." / "..").
    pub fn list_dir(&self, cwd: &str, path: &str) -> Result<Vec<String>> {
        let abs = resolve_against(cwd, path);
        let real = self.realpath(&abs, true)?;
        match self.raw_get(&real) {
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => {}
            Some(_) => return Err(VfsError::NotADir(path.to_string())),
            None => return Err(VfsError::NotFound(path.to_string())),
        }
        let prefix = if real == "/" {
            "/".to_string()
        } else {
            format!("{real}/")
        };
        let mut out = Vec::new();
        for key in self.all_paths().map(|(path, _)| path) {
            if let Some(rest) = key.strip_prefix(&prefix) {
                if !rest.is_empty() && !rest.contains('/') {
                    out.push(rest.to_string());
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// All paths under `dir` (recursive), including the dir itself, sorted.
    pub fn walk(&self, abs_dir: &str) -> Vec<String> {
        let prefix = if abs_dir == "/" {
            "/".to_string()
        } else {
            format!("{abs_dir}/")
        };
        let mut out = Vec::new();
        for key in self.all_paths().map(|(path, _)| path) {
            if key == abs_dir || key.starts_with(&prefix) {
                out.push(key.clone());
            }
        }
        out
    }

    pub fn read_link(&self, cwd: &str, path: &str) -> Result<String> {
        let abs = resolve_against(cwd, path);
        let real = self.realpath(&abs, false)?;
        match self.raw_get(&real) {
            Some(Node {
                kind: NodeKind::Symlink(t),
                ..
            }) => Ok(t.clone()),
            _ => Err(VfsError::Invalid(path.to_string())),
        }
    }

    /// Read at most `maximum` bytes of a symlink target without cloning the full target.
    /// The result is bytes because WASI may truncate a UTF-8 target between code points.
    pub(crate) fn read_link_prefix(
        &self,
        cwd: &str,
        path: &str,
        maximum: usize,
    ) -> Result<Vec<u8>> {
        match &self.metadata_ref(cwd, path, false)?.kind {
            NodeKind::Symlink(target) => {
                Ok(target.as_bytes()[..target.len().min(maximum)].to_vec())
            }
            _ => Err(VfsError::Invalid(path.to_string())),
        }
    }

    // ---- writes ----

    fn require_parent_dir(&self, abs: &str) -> Result<()> {
        let parent = parent_of(abs).unwrap_or_else(|| "/".to_string());
        let real_parent = self.realpath(&parent, true)?;
        match self.raw_get(&real_parent) {
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => Ok(()),
            Some(_) => Err(VfsError::NotADir(parent)),
            None => Err(VfsError::NotFound(parent)),
        }
    }

    /// Map a path to where the node should actually live (parent symlinks resolved).
    fn write_target(&self, cwd: &str, path: &str) -> Result<String> {
        let abs = resolve_against(cwd, path);
        reject_pseudo_mutation(&abs)?;
        check_path_limits(&abs)?;
        let parent = parent_of(&abs).unwrap_or_else(|| "/".to_string());
        let real_parent = self.realpath(&parent, true)?;
        let name = basename(&abs);
        Ok(if real_parent == "/" {
            format!("/{name}")
        } else {
            format!("{real_parent}/{name}")
        })
    }

    pub fn write(&mut self, cwd: &str, path: &str, data: &[u8], mode: Mode) -> Result<()> {
        let target = self.write_target(cwd, path)?;
        self.require_parent_dir(&target)?;
        if matches!(
            self.raw_get(&target),
            Some(Node {
                kind: NodeKind::Dir,
                ..
            })
        ) {
            return Err(VfsError::IsADir(path.to_string()));
        }
        let replacement_usage = NODE_OVERHEAD.saturating_add(data.len() as u64);
        let used = self.projected_usage_after_replacement(&target, replacement_usage)?;
        let mutation_time_ms = self.mutation_time_ms;
        match self.node_mut(&target) {
            Some(Node {
                kind: NodeKind::File(d),
                mtime,
                ..
            }) => {
                *d = Arc::new(data.to_vec());
                *mtime = mutation_time_ms;
            }
            _ => {
                let inode = self.allocate_inode()?;
                self.insert_node(
                    target,
                    Node::file(inode, data.to_vec(), mode, self.mutation_time_ms),
                );
            }
        }
        self.record_usage(used);
        Ok(())
    }

    pub fn append(&mut self, cwd: &str, path: &str, data: &[u8], mode: Mode) -> Result<()> {
        let target = self.write_target(cwd, path)?;
        self.require_parent_dir(&target)?;
        let added = match self.raw_get(&target) {
            Some(Node {
                kind: NodeKind::File(_),
                ..
            }) => data.len() as u64,
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => return Err(VfsError::IsADir(path.to_string())),
            Some(existing) => {
                let replacement = NODE_OVERHEAD.saturating_add(data.len() as u64);
                replacement.saturating_sub(self.node_usage(&target, existing))
            }
            None => NODE_OVERHEAD.saturating_add(data.len() as u64),
        };
        let used = self.usage_after_adding(added)?;
        let mutation_time_ms = self.mutation_time_ms;
        match self.node_mut(&target) {
            Some(Node {
                kind: NodeKind::File(d),
                mtime,
                ..
            }) => {
                Arc::make_mut(d).extend_from_slice(data);
                *mtime = mutation_time_ms;
            }
            _ => {
                let inode = self.allocate_inode()?;
                self.insert_node(
                    target,
                    Node::file(inode, data.to_vec(), mode, self.mutation_time_ms),
                );
            }
        }
        self.record_usage(used);
        Ok(())
    }

    /// Write bytes at a file offset, extending the file with zeroes when needed.
    ///
    /// Descriptor-backed writes use this operation so duplicated descriptors share a cursor
    /// without exposing the VFS's storage representation. The caller must create the file before
    /// writing; this keeps open/create policy at the descriptor boundary.
    pub(crate) fn write_at(
        &mut self,
        cwd: &str,
        path: &str,
        offset: usize,
        data: &[u8],
    ) -> Result<()> {
        let target = self.write_target(cwd, path)?;
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| VfsError::Invalid(path.to_string()))?;
        let existing_len = match self.raw_get(&target) {
            Some(Node {
                kind: NodeKind::File(bytes),
                ..
            }) => bytes.len(),
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => return Err(VfsError::IsADir(path.to_string())),
            Some(_) => return Err(VfsError::Invalid(path.to_string())),
            None => return Err(VfsError::NotFound(path.to_string())),
        };
        let growth = end.saturating_sub(existing_len) as u64;
        let used = self.usage_after_adding(growth)?;
        let mutation_time_ms = self.mutation_time_ms;
        let node = self.node_mut(&target).expect("file was checked above");
        let NodeKind::File(bytes) = &mut node.kind else {
            unreachable!("file was checked above")
        };
        if end > bytes.len() {
            Arc::make_mut(bytes).resize(end, 0);
        }
        Arc::make_mut(bytes)[offset..end].copy_from_slice(data);
        node.mtime = mutation_time_ms;
        self.record_usage(used);
        Ok(())
    }

    /// Resize an existing regular file under the disk budget, preserving its inode metadata.
    pub(crate) fn resize_file(&mut self, cwd: &str, path: &str, size: usize) -> Result<()> {
        let target = self.write_target(cwd, path)?;
        let old_size = self.file_len("/", &target)?;
        let used = self
            .usage_after_adding(size.saturating_sub(old_size) as u64)?
            .saturating_sub(old_size.saturating_sub(size) as u64);
        let mutation_time_ms = self.mutation_time_ms;
        let node = self.node_mut(&target).expect("existing file was checked");
        let NodeKind::File(bytes) = &mut node.kind else {
            unreachable!("regular file was checked");
        };
        resize_buffer(bytes, size);
        node.mtime = mutation_time_ms;
        self.record_usage(used);
        Ok(())
    }

    pub fn mkdir(&mut self, cwd: &str, path: &str) -> Result<()> {
        let target = self.write_target(cwd, path)?;
        if self.contains_node(&target) {
            return Err(VfsError::Exists(path.to_string()));
        }
        self.require_parent_dir(&target)?;
        let node = Node::dir(self.allocate_inode()?, 0o755, self.mutation_time_ms);
        let used = self.usage_after_adding(self.node_usage(&target, &node))?;
        self.insert_node(target, node);
        self.record_usage(used);
        Ok(())
    }

    pub fn mkdir_all(&mut self, cwd: &str, path: &str) -> Result<()> {
        let abs = resolve_against(cwd, path);
        reject_pseudo_mutation(&abs)?;
        check_path_limits(&abs)?;
        let additions = self.missing_directories(&abs)?;
        let added_usage = self.directory_addition_usage(&additions);
        let used = self.disk_used.saturating_add(added_usage);
        if used > self.disk_limit {
            return Err(VfsError::NoSpace);
        }
        self.insert_directories(additions)?;
        self.record_usage(used);
        Ok(())
    }

    pub fn remove_file(&mut self, cwd: &str, path: &str) -> Result<()> {
        let abs = resolve_against(cwd, path);
        reject_pseudo_mutation(&abs)?;
        let real = self.realpath(&abs, false)?;
        match self.raw_get(&real) {
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => Err(VfsError::IsADir(path.to_string())),
            Some(_) => {
                self.remove_node(&real);
                self.refresh_usage();
                Ok(())
            }
            None => Err(VfsError::NotFound(path.to_string())),
        }
    }

    /// Remove a pathname while retaining its node for descriptors that already have it open.
    /// The returned identity is not a path and is inaccessible through ordinary VFS lookup.
    pub(crate) fn unlink_open_file(&mut self, cwd: &str, path: &str) -> Result<u64> {
        let abs = resolve_against(cwd, path);
        reject_pseudo_mutation(&abs)?;
        let real = self.realpath(&abs, false)?;
        if !matches!(
            self.raw_get(&real).map(|node| &node.kind),
            Some(NodeKind::File(_))
        ) {
            return Err(VfsError::NotFound(path.to_string()));
        }
        let id = self.next_orphan;
        self.next_orphan = id.checked_add(1).ok_or(VfsError::NoSpace)?;
        let node = self.remove_node(&real).expect("file was checked");
        self.orphaned.insert(id, node);
        self.refresh_usage();
        Ok(id)
    }

    pub(crate) fn orphan_metadata(&self, id: u64) -> Result<&Node> {
        self.orphaned
            .get(&id)
            .ok_or_else(|| VfsError::NotFound(format!("unlinked file {id}")))
    }

    pub(crate) fn orphan_len(&self, id: u64) -> Result<usize> {
        let node = self.orphan_metadata(id)?;
        let NodeKind::File(bytes) = &node.kind else {
            unreachable!("orphan is always a file")
        };
        Ok(bytes.len())
    }

    pub(crate) fn read_orphan_range(
        &self,
        id: u64,
        offset: usize,
        maximum: usize,
    ) -> Result<Vec<u8>> {
        let node = self
            .orphaned
            .get(&id)
            .ok_or_else(|| VfsError::NotFound(format!("unlinked file {id}")))?;
        let NodeKind::File(bytes) = &node.kind else {
            unreachable!("orphan is always a file")
        };
        let start = offset.min(bytes.len());
        let end = start.saturating_add(maximum).min(bytes.len());
        self.read_bytes
            .set(self.read_bytes.get().saturating_add((end - start) as u64));
        Ok(bytes[start..end].to_vec())
    }

    pub(crate) fn write_orphan_at(&mut self, id: u64, offset: usize, data: &[u8]) -> Result<()> {
        let end = offset.checked_add(data.len()).ok_or(VfsError::NoSpace)?;
        let node = self
            .orphaned
            .get_mut(&id)
            .ok_or_else(|| VfsError::NotFound(format!("unlinked file {id}")))?;
        let NodeKind::File(bytes) = &mut node.kind else {
            unreachable!("orphan is always a file")
        };
        let growth = end.saturating_sub(bytes.len()) as u64;
        if self.disk_used.saturating_add(growth) > self.disk_limit {
            return Err(VfsError::NoSpace);
        }
        if end > bytes.len() {
            Arc::make_mut(bytes).resize(end, 0);
        }
        Arc::make_mut(bytes)[offset..end].copy_from_slice(data);
        node.mtime = self.mutation_time_ms;
        self.record_usage(self.disk_used.saturating_add(growth));
        Ok(())
    }

    /// Resize detached storage without recreating an unlinked pathname.
    pub(crate) fn resize_orphan(&mut self, id: u64, size: usize) -> Result<()> {
        let old_size = match &self.orphan_metadata(id)?.kind {
            NodeKind::File(bytes) => bytes.len(),
            _ => unreachable!("orphan is always a file"),
        };
        let used = self
            .usage_after_adding(size.saturating_sub(old_size) as u64)?
            .saturating_sub(old_size.saturating_sub(size) as u64);
        let node = self.orphaned.get_mut(&id).expect("orphan was checked");
        let NodeKind::File(bytes) = &mut node.kind else {
            unreachable!("orphan is always a file");
        };
        resize_buffer(bytes, size);
        node.mtime = self.mutation_time_ms;
        self.record_usage(used);
        Ok(())
    }

    /// Reclaim unlinked storage once no file description names its orphan identity.
    pub(crate) fn retain_orphans(&mut self, live: &BTreeSet<u64>) {
        // Every process exit and descriptor close reaches here, so the usage is adjusted by the
        // orphans dropped instead of recounting the whole tree.
        let mut freed = 0_u64;
        self.orphaned.retain(|id, node| {
            if live.contains(id) {
                return true;
            }
            freed = freed.saturating_add(NODE_OVERHEAD.saturating_add(node_payload_len(node)));
            false
        });
        if freed > 0 {
            self.record_usage(self.disk_used.saturating_sub(freed));
        }
    }

    pub fn remove_all(&mut self, cwd: &str, path: &str) -> Result<()> {
        let abs = resolve_against(cwd, path);
        reject_pseudo_mutation(&abs)?;
        let real = self.realpath(&abs, false)?;
        if !self.contains_node(&real) {
            return Err(VfsError::NotFound(path.to_string()));
        }
        for k in self.walk(&real) {
            self.remove_node(&k);
        }
        self.refresh_usage();
        Ok(())
    }

    pub fn rmdir(&mut self, cwd: &str, path: &str) -> Result<()> {
        let abs = resolve_against(cwd, path);
        reject_pseudo_mutation(&abs)?;
        let real = self.realpath(&abs, true)?;
        match self.raw_get(&real) {
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => {
                if !self.list_dir(cwd, path)?.is_empty() {
                    return Err(VfsError::NotEmpty(path.to_string()));
                }
                self.remove_node(&real);
                self.refresh_usage();
                Ok(())
            }
            Some(_) => Err(VfsError::NotADir(path.to_string())),
            None => Err(VfsError::NotFound(path.to_string())),
        }
    }

    pub fn rename(&mut self, cwd: &str, from: &str, to: &str) -> Result<()> {
        let from_abs = resolve_against(cwd, from);
        reject_pseudo_mutation(&from_abs)?;
        let from_real = self.realpath(&from_abs, false)?;
        if !self.contains_node(&from_real) {
            return Err(VfsError::NotFound(from.to_string()));
        }
        let mut to_target = self.write_target(cwd, to)?;
        // moving into an existing directory
        if matches!(
            self.raw_get(&to_target),
            Some(Node {
                kind: NodeKind::Dir,
                ..
            })
        ) {
            let name = basename(&from_real);
            to_target = if to_target == "/" {
                format!("/{name}")
            } else {
                format!("{to_target}/{name}")
            };
        }
        let subtree = self.walk(&from_real);
        // Baseline directories stop being free when renamed away and a replaced target stops
        // counting, so the usage change is the per-node difference.
        let mut delta: i128 = 0;
        for k in &subtree {
            let node = self.raw_get(k).unwrap();
            let newk = format!("{to_target}{}", &k[from_real.len()..]);
            check_path_limits(&newk)?;
            delta += i128::from(self.node_usage(&newk, node));
            delta -= i128::from(self.node_usage(k, node));
            if let Some(replaced) = self.raw_get(&newk) {
                if !subtree.contains(&newk) {
                    delta -= i128::from(self.node_usage(&newk, replaced));
                }
            }
        }
        let used = self.usage_after_delta(delta)?;
        for k in subtree {
            let node = self.remove_node(&k).unwrap();
            let suffix = &k[from_real.len()..];
            let newk = format!("{to_target}{suffix}");
            self.insert_node(newk, node);
        }
        self.record_usage(used);
        Ok(())
    }

    pub fn copy_file(&mut self, cwd: &str, from: &str, to: &str) -> Result<()> {
        reject_pseudo_mutation(&resolve_against(cwd, to))?;
        let source = self.metadata(cwd, from, true)?;
        if let NodeKind::NativeExecutable(program) = source.kind {
            let target = self.write_target(cwd, to)?;
            self.require_parent_dir(&target)?;
            let node = Node {
                inode: self.allocate_inode()?,
                kind: NodeKind::NativeExecutable(program),
                mode: source.mode,
                uid: source.uid,
                gid: source.gid,
                mtime: self.mutation_time_ms,
            };
            let used =
                self.projected_usage_after_replacement(&target, self.node_usage(&target, &node))?;
            self.insert_node(target, node);
            self.record_usage(used);
            return Ok(());
        }
        let NodeKind::File(data) = source.kind else {
            // Keep the existing public error behavior for non-regular operands.
            self.read(cwd, from)?;
            unreachable!("read only accepts regular files")
        };
        let target = self.write_target(cwd, to)?;
        self.require_parent_dir(&target)?;
        if matches!(
            self.raw_get(&target).map(|node| &node.kind),
            Some(NodeKind::Dir)
        ) {
            return Err(VfsError::IsADir(to.to_string()));
        }
        let used = self.projected_usage_after_replacement(
            &target,
            NODE_OVERHEAD.saturating_add(data.len() as u64),
        )?;
        self.read_bytes
            .set(self.read_bytes.get().saturating_add(data.len() as u64));
        let mutation_time_ms = self.mutation_time_ms;
        match self.node_mut(&target) {
            Some(Node {
                kind: NodeKind::File(contents),
                mtime,
                ..
            }) => {
                *contents = data;
                *mtime = mutation_time_ms;
            }
            _ => {
                let mut node = Node::file(
                    self.allocate_inode()?,
                    Vec::new(),
                    source.mode,
                    mutation_time_ms,
                );
                node.kind = NodeKind::File(data);
                self.insert_node(target, node);
            }
        }
        self.record_usage(used);
        Ok(())
    }

    /// Copy a subtree within the VFS, optionally retaining each node's virtual modification time.
    /// Permission bits and ownership are retained in either mode; ordinary copies receive the
    /// current virtual mutation time.
    pub fn copy_recursive(
        &mut self,
        cwd: &str,
        from: &str,
        to: &str,
        preserve: bool,
    ) -> Result<()> {
        let from_abs = resolve_against(cwd, from);
        let from_real = self.realpath(&from_abs, true)?;
        if !self.is_dir(cwd, from) {
            self.copy_file(cwd, from, to)?;
            if preserve {
                let source = self.metadata(cwd, from, true)?;
                self.chmod(cwd, to, source.mode)?;
                self.touch(cwd, to, source.mtime)?;
            }
            return Ok(());
        }
        let mut to_target = self.write_target(cwd, to)?;
        if matches!(
            self.raw_get(&to_target),
            Some(Node {
                kind: NodeKind::Dir,
                ..
            })
        ) {
            let name = basename(&from_real);
            to_target = if to_target == "/" {
                format!("/{name}")
            } else {
                format!("{to_target}/{name}")
            };
        }
        let subtree = self.walk(&from_real);
        let mut delta: i128 = 0;
        for k in &subtree {
            let node = self.raw_get(k).unwrap();
            let newk = format!("{to_target}{}", &k[from_real.len()..]);
            check_path_limits(&newk)?;
            delta += i128::from(self.node_usage(&newk, node));
            if let Some(replaced) = self.raw_get(&newk) {
                delta -= i128::from(self.node_usage(&newk, replaced));
            }
        }
        let used = self.usage_after_delta(delta)?;
        self.check_inode_capacity(subtree.len())?;
        for k in subtree {
            let mut node = self.raw_get(&k).unwrap().clone();
            node.inode = self.allocate_inode()?;
            if !preserve {
                node.mtime = self.mutation_time_ms;
            }
            let suffix = &k[from_real.len()..];
            let newk = format!("{to_target}{suffix}");
            self.insert_node(newk, node);
        }
        self.record_usage(used);
        Ok(())
    }

    pub fn symlink(&mut self, cwd: &str, target: &str, linkpath: &str) -> Result<()> {
        let link_target = self.write_target(cwd, linkpath)?;
        if self.contains_node(&link_target) {
            return Err(VfsError::Exists(linkpath.to_string()));
        }
        self.require_parent_dir(&link_target)?;
        let node = Node {
            inode: self.allocate_inode()?,
            kind: NodeKind::Symlink(target.to_string()),
            mode: 0o777,
            uid: 0,
            gid: 0,
            mtime: self.mutation_time_ms,
        };
        let used = self.usage_after_adding(self.node_usage(&link_target, &node))?;
        self.insert_node(link_target, node);
        self.record_usage(used);
        Ok(())
    }

    pub fn touch(&mut self, cwd: &str, path: &str, mtime: u64) -> Result<()> {
        let target = self.write_target(cwd, path)?;
        if let Some(node) = self.node_mut(&target) {
            node.mtime = mtime;
            return Ok(());
        }
        self.require_parent_dir(&target)?;
        let node = Node::file(self.allocate_inode()?, Vec::new(), 0o644, mtime);
        let used = self.usage_after_adding(self.node_usage(&target, &node))?;
        self.insert_node(target, node);
        self.record_usage(used);
        Ok(())
    }

    pub fn chmod(&mut self, cwd: &str, path: &str, mode: Mode) -> Result<()> {
        let abs = resolve_against(cwd, path);
        reject_pseudo_mutation(&abs)?;
        let real = self.realpath(&abs, true)?;
        self.node_mut(&real)
            .map(|n| n.mode = mode & 0o7777)
            .ok_or_else(|| VfsError::NotFound(path.to_string()))
    }

    pub fn chown(
        &mut self,
        cwd: &str,
        path: &str,
        uid: Option<u32>,
        gid: Option<u32>,
    ) -> Result<()> {
        let abs = resolve_against(cwd, path);
        reject_pseudo_mutation(&abs)?;
        let real = self.realpath(&abs, true)?;
        let n = self
            .node_mut(&real)
            .ok_or_else(|| VfsError::NotFound(path.to_string()))?;
        if let Some(u) = uid {
            n.uid = u;
        }
        if let Some(g) = gid {
            n.gid = g;
        }
        Ok(())
    }

    pub fn set_mtime(&mut self, abs_real: &str, mtime: u64) {
        if let Some(n) = self.node_mut(abs_real) {
            n.mtime = mtime;
        }
    }

    // ---- bulk load / dump (bridge to real dirs for task loading & snapshots) ----

    /// Insert a file directly at an absolute path, creating parent dirs.
    pub fn put_file(&mut self, abs: &str, data: Vec<u8>, mode: Mode) -> Result<()> {
        let norm = normalize(abs);
        reject_pseudo_mutation(&norm)?;
        check_path_limits(&norm)?;
        if matches!(
            self.raw_get(&norm).map(|node| &node.kind),
            Some(NodeKind::Dir)
        ) {
            return Err(VfsError::IsADir(norm));
        }
        let additions = match parent_of(&norm) {
            Some(parent) => self.missing_directories(&parent)?,
            None => Vec::new(),
        };
        let previous = self
            .raw_get(&norm)
            .map_or(0, |node| self.node_usage(&norm, node));
        let file_usage = NODE_OVERHEAD.saturating_add(data.len() as u64);
        let used = self
            .disk_used
            .saturating_sub(previous)
            .saturating_add(self.directory_addition_usage(&additions))
            .saturating_add(file_usage);
        if used > self.disk_limit {
            return Err(VfsError::NoSpace);
        }
        self.check_inode_capacity(additions.len().saturating_add(1))?;
        self.insert_directories(additions)?;
        let inode = self.allocate_inode()?;
        self.insert_node(norm, Node::file(inode, data, mode, self.mutation_time_ms));
        self.record_usage(used);
        Ok(())
    }

    /// Install a trusted native image at a virtual executable path in the base environment.
    pub(crate) fn seed_native_executable(&mut self, abs: &str, program: NativeProgram) {
        let norm = normalize(abs);
        let inode = self.allocate_inode().expect("base image inode capacity");
        self.insert_node(
            norm.clone(),
            Node {
                inode,
                kind: NodeKind::NativeExecutable(program),
                mode: 0o755,
                uid: 0,
                gid: 0,
                mtime: self.mutation_time_ms,
            },
        );
        self.baseline_native.insert(norm, program);
        self.refresh_usage();
    }

    pub fn put_dir(&mut self, abs: &str, mode: Mode) -> Result<()> {
        let norm = normalize(abs);
        reject_pseudo_mutation(&norm)?;
        self.mkdir_all("/", &norm)?;
        if let Some(n) = self.node_mut(&norm) {
            n.mode = mode;
        }
        Ok(())
    }

    pub fn all_paths(&self) -> impl Iterator<Item = (&String, &Node)> {
        let mut base = self.base.nodes.iter().peekable();
        let mut overlay = self.overlay.iter().peekable();
        std::iter::from_fn(move || loop {
            match (base.peek(), overlay.peek()) {
                (Some((base_path, _)), Some((overlay_path, _))) if base_path < overlay_path => {
                    return base.next();
                }
                (Some((base_path, _)), Some((overlay_path, _))) if base_path == overlay_path => {
                    base.next();
                }
                (Some(_), None) => return base.next(),
                (None, None) => return None,
                _ => {}
            }
            if let Some((path, OverlayEntry::Present(node))) = overlay.next() {
                return Some((path, node));
            }
        })
    }

    pub fn len(&self) -> usize {
        self.node_count
    }

    pub fn is_empty(&self) -> bool {
        self.node_count <= 1
    }
}

const NODE_OVERHEAD: u64 = 256;

fn cow_copy_bytes(node: &Node, installed: bool, mutation: FileMutation) -> u64 {
    let NodeKind::File(bytes) = &node.kind else {
        return 0;
    };
    if !installed && Arc::strong_count(bytes) == 1 {
        return 0;
    }
    match mutation {
        FileMutation::Write => bytes.len() as u64,
        FileMutation::Resize(size) if size == bytes.len() => 0,
        FileMutation::Resize(size) => size.min(bytes.len()) as u64,
    }
}

/// Truncation never clones bytes that the operation is about to discard.
fn resize_buffer(bytes: &mut Arc<Vec<u8>>, size: usize) {
    if size == bytes.len() {
        return;
    }
    if size == 0 {
        *bytes = Arc::new(Vec::new());
    } else if size < bytes.len() && Arc::strong_count(bytes) > 1 {
        *bytes = Arc::new(bytes[..size].to_vec());
    } else {
        Arc::make_mut(bytes).resize(size, 0);
    }
}

fn logical_usage<'a>(nodes: impl Iterator<Item = (&'a String, &'a Node)>) -> u64 {
    nodes
        .filter(|(path, _)| path.as_str() != "/")
        .fold(0u64, |total, (_, node)| {
            let payload = match &node.kind {
                NodeKind::File(data) => data.len() as u64,
                NodeKind::Symlink(target) => target.len() as u64,
                NodeKind::Dir | NodeKind::NativeExecutable(_) => 0,
            };
            total.saturating_add(NODE_OVERHEAD).saturating_add(payload)
        })
}

fn node_payload_len(node: &Node) -> u64 {
    match &node.kind {
        NodeKind::File(data) => data.len() as u64,
        NodeKind::Symlink(target) => target.len() as u64,
        NodeKind::Dir | NodeKind::NativeExecutable(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn frozen_overlays_isolate_content_metadata_and_recreated_subtrees() {
        let mut installation = Vfs::new();
        installation
            .put_file("/tree/old", b"base".to_vec(), 0o644)
            .unwrap();
        installation
            .put_file("/z", b"last".to_vec(), 0o644)
            .unwrap();
        let base = installation.freeze();
        let mut first = Vfs::from_base(base.clone(), u64::MAX).unwrap();
        let second = Vfs::from_base(base, u64::MAX).unwrap();
        first.copy_file("/", "/tree/old", "/copied").unwrap();
        first.write_at("/", "/copied", 0, b"X").unwrap();
        assert_eq!(first.read("/", "/tree/old").unwrap(), b"base");
        assert_eq!(first.read("/", "/copied").unwrap(), b"Xase");
        first.remove_file("/", "/copied").unwrap();
        first.append("/", "/tree/old", b" private", 0o644).unwrap();
        first.chmod("/", "/tree/old", 0o600).unwrap();
        assert_eq!(second.read("/", "/tree/old").unwrap(), b"base");
        assert_eq!(second.metadata("/", "/tree/old", true).unwrap().mode, 0o644);
        first.remove_all("/", "/tree").unwrap();
        first.put_file("/tree/new", b"new".to_vec(), 0o644).unwrap();
        first.put_file("/a", vec![], 0o644).unwrap();
        assert_eq!(first.list_dir("/", "/tree").unwrap(), vec!["new"]);
        assert_eq!(first.walk("/"), vec!["/", "/a", "/tree", "/tree/new", "/z"]);
        assert_eq!(
            first
                .all_paths()
                .map(|(path, _)| path.as_str())
                .collect::<Vec<_>>(),
            vec!["/", "/a", "/tree", "/tree/new", "/z"]
        );
        assert_eq!(second.list_dir("/", "/tree").unwrap(), vec!["old"]);
        let refrozen = Vfs::from_base(first.freeze(), u64::MAX).unwrap();
        assert!(!refrozen.exists("/", "/tree/old"));
        assert_eq!(refrozen.read("/", "/tree/new").unwrap(), b"new");
    }

    #[test]
    fn frozen_overlay_rename_and_open_unlink_preserve_quota_and_identity() {
        let mut installation = Vfs::new();
        installation
            .put_file("/tree/file", b"abc".to_vec(), 0o644)
            .unwrap();
        let inode = installation
            .metadata("/", "/tree/file", true)
            .unwrap()
            .inode;
        let base = installation.freeze();
        let quota = base.disk_used();
        assert!(matches!(
            Vfs::from_base(base.clone(), quota - 1),
            Err(VfsError::NoSpace)
        ));
        let mut first = Vfs::from_base(base.clone(), quota).unwrap();
        let second = Vfs::from_base(base, quota).unwrap();
        first.rename("/", "/tree", "/moved").unwrap();
        assert!(!first.exists("/", "/tree/file"));
        assert_eq!(
            first.metadata("/", "/moved/file", true).unwrap().inode,
            inode
        );
        let orphan = first.unlink_open_file("/", "/moved/file").unwrap();
        assert_eq!(first.disk_used(), quota);
        assert!(matches!(
            first.write_orphan_at(orphan, 3, b"d"),
            Err(VfsError::NoSpace)
        ));
        first.write_orphan_at(orphan, 0, b"X").unwrap();
        assert_eq!(first.read_orphan_range(orphan, 0, 10).unwrap(), b"Xbc");
        assert_eq!(second.read("/", "/tree/file").unwrap(), b"abc");
        first.retain_orphans(&BTreeSet::new());
        first.write("/", "/moved/file", b"new", 0o644).unwrap();
        assert_ne!(
            first.metadata("/", "/moved/file", true).unwrap().inode,
            inode
        );
        assert_eq!(first.disk_used(), quota);
    }

    #[test]
    fn inode_identity_tracks_nodes_across_mutation_clone_and_unlink() {
        let mut vfs = Vfs::new();
        vfs.put_file("/file", b"old".to_vec(), 0o644).unwrap();
        let original = vfs.metadata("/", "/file", true).unwrap().inode;
        assert_ne!(original, 0);
        vfs.write("/", "/file", b"new", 0o600).unwrap();
        vfs.rename("/", "/file", "/renamed").unwrap();
        assert_eq!(vfs.metadata("/", "/renamed", true).unwrap().inode, original);
        assert_eq!(
            vfs.clone().metadata("/", "/renamed", true).unwrap().inode,
            original
        );
        let orphan = vfs.unlink_open_file("/", "/renamed").unwrap();
        assert_eq!(vfs.orphan_metadata(orphan).unwrap().inode, original);
        vfs.put_file("/renamed", b"replacement".to_vec(), 0o644)
            .unwrap();
        assert_ne!(vfs.metadata("/", "/renamed", true).unwrap().inode, original);
        let replacement = vfs.metadata("/", "/renamed", true).unwrap().inode;
        vfs.put_file("/renamed", b"reinstalled".to_vec(), 0o644)
            .unwrap();
        assert_ne!(
            vfs.metadata("/", "/renamed", true).unwrap().inode,
            replacement
        );
        vfs.mkdir_all("/", "/tree/child").unwrap();
        let directory = vfs.metadata("/", "/tree", true).unwrap().inode;
        vfs.copy_recursive("/", "/tree", "/copy", true).unwrap();
        assert_ne!(vfs.metadata("/", "/copy", true).unwrap().inode, directory);
    }

    #[test]
    fn inode_exhaustion_rejects_bulk_creation_before_mutation() {
        let mut vfs = Vfs::new();
        vfs.next_inode = u64::MAX - 1;
        assert!(matches!(
            vfs.put_file("/a/b", vec![], 0o644),
            Err(VfsError::NoSpace)
        ));
        assert!(!vfs.exists("/", "/a"));
        assert_eq!(vfs.next_inode, u64::MAX - 1);
    }

    #[test]
    fn resize_preserves_data_and_unlinked_identity_and_rolls_back_disk_failure() {
        let mut vfs = super::Vfs::with_disk_limit(4096);
        vfs.write("/", "/file", b"abc", 0o640).unwrap();
        let before = vfs.disk_used();
        vfs.resize_file("/", "/file", 5).unwrap();
        assert_eq!(vfs.read("/", "/file").unwrap(), b"abc\0\0");
        assert_eq!(vfs.disk_used(), before + 2);
        assert!(matches!(
            vfs.resize_file("/", "/file", 5000),
            Err(super::VfsError::NoSpace)
        ));
        assert_eq!(vfs.read("/", "/file").unwrap(), b"abc\0\0");
        assert_eq!(vfs.disk_used(), before + 2);
        let orphan = vfs.unlink_open_file("/", "/file").unwrap();
        vfs.resize_orphan(orphan, 1).unwrap();
        assert_eq!(vfs.read_orphan_range(orphan, 0, 10).unwrap(), b"a");
        assert!(!vfs.exists("/", "/file"));
        assert!(matches!(
            vfs.resize_orphan(orphan, 5000),
            Err(super::VfsError::NoSpace)
        ));
        assert_eq!(vfs.read_orphan_range(orphan, 0, 10).unwrap(), b"a");
    }
    use super::*;

    #[test]
    fn normalize_paths() {
        assert_eq!(normalize("/a/b/../c"), "/a/c");
        assert_eq!(normalize("/a/./b/"), "/a/b");
        assert_eq!(normalize("/"), "/");
        assert_eq!(normalize("/../.."), "/");
        assert_eq!(resolve_against("/home/x", "y/z"), "/home/x/y/z");
        assert_eq!(resolve_against("/home/x", "/etc"), "/etc");
    }

    #[test]
    fn basic_rw() {
        let mut v = Vfs::new();
        v.mkdir_all("/", "/a/b").unwrap();
        v.write("/a/b", "f.txt", b"hello", 0o644).unwrap();
        assert_eq!(v.read_string("/a/b", "f.txt").unwrap(), "hello");
        assert_eq!(v.read_string("/", "/a/b/f.txt").unwrap(), "hello");
        v.append("/", "/a/b/f.txt", b" world", 0o644).unwrap();
        assert_eq!(v.read_string("/", "/a/b/f.txt").unwrap(), "hello world");
        assert!(v.is_dir("/", "/a"));
        assert!(v.is_file("/", "/a/b/f.txt"));
        assert_eq!(v.list_dir("/", "/a/b").unwrap(), vec!["f.txt"]);
    }

    #[test]
    fn rename_and_copy() {
        let mut v = Vfs::new();
        v.mkdir_all("/", "/src").unwrap();
        v.write("/", "/src/a.txt", b"A", 0o644).unwrap();
        v.rename("/", "/src/a.txt", "/src/b.txt").unwrap();
        assert!(!v.is_file("/", "/src/a.txt"));
        assert_eq!(v.read_string("/", "/src/b.txt").unwrap(), "A");
        v.mkdir_all("/", "/dst").unwrap();
        v.copy_recursive("/", "/src", "/dst", false).unwrap();
        assert_eq!(v.read_string("/", "/dst/src/b.txt").unwrap(), "A");
    }

    #[test]
    fn symlinks() {
        let mut v = Vfs::new();
        v.mkdir_all("/", "/real").unwrap();
        v.write("/", "/real/f", b"data", 0o644).unwrap();
        v.symlink("/", "/real", "/link").unwrap();
        assert!(v.is_symlink("/", "/link"));
        assert_eq!(v.read_string("/", "/link/f").unwrap(), "data");
    }

    #[test]
    fn disk_quota_is_atomic_and_delete_releases_space() {
        // The implicit root is free; the file node and three payload bytes are charged.
        let mut v = Vfs::with_disk_limit(NODE_OVERHEAD + 3);
        v.seed_dirs(["/builtin"]);
        v.seed_native_executable("/builtin/true", NativeProgram::True);
        let mut v = Vfs::from_base(v.freeze(), NODE_OVERHEAD + 3).unwrap();
        v.write("/", "/a", b"123", 0o644).unwrap();
        assert_eq!(v.disk_used(), NODE_OVERHEAD + 3);

        assert!(matches!(
            v.append("/", "/a", b"4", 0o644),
            Err(VfsError::NoSpace)
        ));
        assert_eq!(v.read_string("/", "/a").unwrap(), "123");
        assert_eq!(v.disk_used(), NODE_OVERHEAD + 3);

        v.remove_file("/", "/a").unwrap();
        assert_eq!(v.disk_used(), 0);
        v.write("/", "/b", b"123", 0o644).unwrap();
    }

    #[test]
    fn rename_and_copy_adjust_usage_without_recounting() {
        let mut v = Vfs::with_disk_limit(NODE_OVERHEAD * 2 + 7);
        v.write("/", "/a", b"12345", 0o644).unwrap();
        v.rename("/", "/a", "/b").unwrap();
        assert_eq!(v.disk_used(), NODE_OVERHEAD + 5);

        v.write("/", "/c", b"12", 0o644).unwrap();
        assert_eq!(v.disk_used(), NODE_OVERHEAD * 2 + 7);
        // Renaming over `/b` releases the replaced node.
        v.rename("/", "/c", "/b").unwrap();
        assert_eq!(v.disk_used(), NODE_OVERHEAD + 2);

        v.copy_file("/", "/b", "/d").unwrap();
        assert_eq!(v.disk_used(), NODE_OVERHEAD * 2 + 4);
        assert!(matches!(
            v.copy_file("/", "/b", "/e"),
            Err(VfsError::NoSpace)
        ));
        assert!(!v.exists("/", "/e"));
        assert_eq!(v.disk_used(), NODE_OVERHEAD * 2 + 4);
    }

    #[test]
    fn over_long_names_and_paths_are_rejected() {
        let mut v = Vfs::new();
        let name = "n".repeat(NAME_MAX + 1);
        assert!(matches!(
            v.write("/", &format!("/{name}"), b"", 0o644),
            Err(VfsError::NameTooLong(_))
        ));
        let deep = "/d".repeat(PATH_MAX / 2 + 1);
        assert!(matches!(
            v.mkdir_all("/", &deep),
            Err(VfsError::NameTooLong(_))
        ));
        assert!(!v.exists("/", "/d"));
        assert_eq!(v.disk_used(), 0);

        let just_fits = "n".repeat(NAME_MAX);
        v.write("/", &format!("/{just_fits}"), b"", 0o644).unwrap();
    }

    #[test]
    fn write_preflights_replacement_and_directory_usage() {
        let mut v = Vfs::with_disk_limit(NODE_OVERHEAD * 2 + 3);
        v.mkdir_all("/", "/dir").unwrap();
        v.write("/", "/dir/file", b"123", 0o644).unwrap();

        assert!(matches!(
            v.write("/", "/dir/file", b"1234", 0o644),
            Err(VfsError::NoSpace)
        ));
        assert_eq!(v.read_string("/", "/dir/file").unwrap(), "123");
        assert_eq!(v.disk_used(), NODE_OVERHEAD * 2 + 3);

        v.write("/", "/dir/file", b"1", 0o644).unwrap();
        assert_eq!(v.disk_used(), NODE_OVERHEAD * 2 + 1);
    }

    #[test]
    fn mkdir_all_rejects_the_complete_path_atomically() {
        let mut v = Vfs::with_disk_limit(NODE_OVERHEAD);

        assert!(matches!(
            v.mkdir_all("/", "/one/two"),
            Err(VfsError::NoSpace)
        ));
        assert!(!v.exists("/", "/one"));
        assert_eq!(v.disk_used(), 0);
    }

    #[test]
    fn put_file_preflights_parents_and_replaces_the_mode() {
        let mut bounded = Vfs::with_disk_limit(NODE_OVERHEAD * 2);
        assert!(matches!(
            bounded.put_file("/one/file", b"x".to_vec(), 0o755),
            Err(VfsError::NoSpace)
        ));
        assert!(!bounded.exists("/", "/one"));

        let mut v = Vfs::new();
        v.put_file("/file", b"one".to_vec(), 0o755).unwrap();
        v.put_file("/file", b"two".to_vec(), 0o640).unwrap();
        assert_eq!(v.metadata("/", "/file", true).unwrap().mode, 0o640);
        assert_eq!(v.read_string("/", "/file").unwrap(), "two");
    }

    #[test]
    fn put_file_rejects_directories_without_changing_them() {
        let mut v = Vfs::new();
        v.mkdir_all("/", "/dir/child").unwrap();
        let usage = v.disk_used();

        for path in ["/", "/dir", "/dir/child"] {
            assert!(matches!(
                v.put_file(path, b"data".to_vec(), 0o644),
                Err(VfsError::IsADir(_))
            ));
            assert!(v.is_dir("/", path));
        }
        assert_eq!(v.disk_used(), usage);
    }

    #[test]
    fn limited_read_rejects_before_returning_file_data() {
        let mut v = Vfs::new();
        v.write("/", "/large", b"12345", 0o644).unwrap();
        assert!(matches!(
            v.read_string_limited("/", "/large", 4),
            Err(VfsError::TooLarge { .. })
        ));
        assert_eq!(v.read_bytes(), 0);
        assert_eq!(v.read_string_limited("/", "/large", 5).unwrap(), "12345");
        assert_eq!(v.read_bytes(), 5);
        v.write("/", "/binary", b"\0\xff", 0o644).unwrap();
        assert!(matches!(
            v.read_limited("/", "/binary", 1),
            Err(VfsError::TooLarge { .. })
        ));
        assert_eq!(v.read_limited("/", "/binary", 2).unwrap(), b"\0\xff");
    }
}
