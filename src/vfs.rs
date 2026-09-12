//! In-memory virtual filesystem.
//!
//! The VFS is the single source of truth for all simulated file state. It is a flat
//! `BTreeMap<path, Node>` keyed by clean absolute paths ("/" is the root). This keeps
//! directory listing, rename, copy and snapshotting trivial at the scale of an RL task
//! (thousands of files), while still supporting unix permissions, ownership and symlinks.

use std::cell::Cell;
use std::collections::BTreeMap;

pub type Mode = u32;

#[derive(Clone, Debug)]
pub enum NodeKind {
    File(Vec<u8>),
    Dir,
    Symlink(String),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub kind: NodeKind,
    pub mode: Mode,
    pub uid: u32,
    pub gid: u32,
    /// virtual modification time in milliseconds (from the simulated clock)
    pub mtime: u64,
}

impl Node {
    fn dir(mode: Mode, mtime: u64) -> Self {
        Node {
            kind: NodeKind::Dir,
            mode,
            uid: 0,
            gid: 0,
            mtime,
        }
    }
    fn file(data: Vec<u8>, mode: Mode, mtime: u64) -> Self {
        Node {
            kind: NodeKind::File(data),
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
    NoSpace,
    TooLarge { path: String, limit: usize },
    ReadOnly(String),
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
            VfsError::NoSpace => write!(f, "No space left on device"),
            VfsError::TooLarge { path, limit } => {
                write!(f, "file too large: {path} (limit {limit} bytes)")
            }
            VfsError::ReadOnly(path) => write!(f, "Read-only filesystem: {path}"),
        }
    }
}

pub type Result<T> = std::result::Result<T, VfsError>;

#[derive(Clone)]
pub struct Vfs {
    nodes: BTreeMap<String, Node>,
    /// Wall-clock timestamp assigned to subsequent content mutations.  The environment updates
    /// this immediately before an effect; the VFS never consults the host clock.
    mutation_time_ms: u64,
    disk_limit: u64,
    /// Directory nodes supplied by the base image rather than created by simulated actions.
    baseline_dirs: std::collections::BTreeSet<String>,
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
    pub fn new() -> Self {
        Self::with_disk_limit(u64::MAX)
    }

    pub fn with_disk_limit(disk_limit: u64) -> Self {
        let mut nodes = BTreeMap::new();
        nodes.insert("/".to_string(), Node::dir(0o755, 0));
        let disk_used = logical_usage(&nodes);
        Vfs {
            nodes,
            mutation_time_ms: 0,
            disk_limit,
            baseline_dirs: std::collections::BTreeSet::new(),
            disk_used,
            disk_peak: disk_used,
            read_bytes: Cell::new(0),
        }
    }

    pub fn disk_used(&self) -> u64 {
        self.disk_used
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
            self.nodes
                .insert(path.clone(), Node::dir(0o755, self.mutation_time_ms));
            self.baseline_dirs.insert(path);
        }
        self.refresh_usage();
        self.disk_peak = self.disk_peak.max(self.disk_used);
    }

    fn finish_mutation(&mut self, before: BTreeMap<String, Node>) -> Result<()> {
        let used = self.measured_usage();
        if used > self.disk_limit {
            self.nodes = before;
            self.disk_used = self.measured_usage();
            Err(VfsError::NoSpace)
        } else {
            self.disk_used = used;
            self.disk_peak = self.disk_peak.max(used);
            Ok(())
        }
    }

    fn refresh_usage(&mut self) {
        self.disk_used = self.measured_usage();
    }

    fn measured_usage(&self) -> u64 {
        let present_baseline_dirs = self
            .baseline_dirs
            .iter()
            .filter(|path| {
                matches!(
                    self.nodes.get(path.as_str()),
                    Some(Node {
                        kind: NodeKind::Dir,
                        ..
                    })
                )
            })
            .count() as u64;
        logical_usage(&self.nodes)
            .saturating_sub(present_baseline_dirs.saturating_mul(NODE_OVERHEAD))
    }

    // ---- low level ----

    pub fn raw_get(&self, abs: &str) -> Option<&Node> {
        self.nodes.get(abs)
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
        match self.nodes.get(&candidate) {
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
            .map(|p| self.nodes.contains_key(&p))
            .unwrap_or(false)
    }

    pub fn lexists(&self, cwd: &str, path: &str) -> bool {
        let abs = resolve_against(cwd, path);
        self.realpath(&abs, false)
            .map(|p| self.nodes.contains_key(&p))
            .unwrap_or(false)
    }

    pub fn is_dir(&self, cwd: &str, path: &str) -> bool {
        let abs = resolve_against(cwd, path);
        match self.realpath(&abs, true) {
            Ok(p) => matches!(
                self.nodes.get(&p),
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
                self.nodes.get(&p),
                Some(Node {
                    kind: NodeKind::File(_),
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
        match self.nodes.get(&real) {
            Some(Node {
                kind: NodeKind::File(data),
                ..
            }) => Ok(data.len()),
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
                self.nodes.get(&p),
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
        match self.nodes.get(&real) {
            Some(Node {
                kind: NodeKind::File(d),
                ..
            }) => {
                self.read_bytes
                    .set(self.read_bytes.get().saturating_add(d.len() as u64));
                Ok(d.clone())
            }
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => Err(VfsError::IsADir(path.to_string())),
            _ => Err(VfsError::NotFound(path.to_string())),
        }
    }

    /// Read a file as exact bytes only when it is within `limit`.
    pub fn read_limited(&self, cwd: &str, path: &str, limit: usize) -> Result<Vec<u8>> {
        let abs = resolve_against(cwd, path);
        let real = self.realpath(&abs, true)?;
        match self.nodes.get(&real) {
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
            _ => Err(VfsError::NotFound(path.to_string())),
        }
    }

    pub fn read_string(&self, cwd: &str, path: &str) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.read(cwd, path)?).into_owned())
    }

    /// Read a UTF-8-ish file only when it is within `limit` bytes.
    ///
    /// Unlike checking `metadata`, this avoids cloning a potentially enormous file merely to
    /// discover that a caller cannot safely process it.  The read counter is updated only for a
    /// read that is actually returned.
    pub fn read_string_limited(&self, cwd: &str, path: &str, limit: usize) -> Result<String> {
        let abs = resolve_against(cwd, path);
        let real = self.realpath(&abs, true)?;
        match self.nodes.get(&real) {
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
            _ => Err(VfsError::NotFound(path.to_string())),
        }
    }

    pub fn metadata(&self, cwd: &str, path: &str, follow: bool) -> Result<Node> {
        let abs = resolve_against(cwd, path);
        let real = self.realpath(&abs, follow)?;
        self.nodes
            .get(&real)
            .cloned()
            .ok_or_else(|| VfsError::NotFound(path.to_string()))
    }

    /// List directory entry names (not including "." / "..").
    pub fn list_dir(&self, cwd: &str, path: &str) -> Result<Vec<String>> {
        let abs = resolve_against(cwd, path);
        let real = self.realpath(&abs, true)?;
        match self.nodes.get(&real) {
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
        for key in self.nodes.keys() {
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
        for key in self.nodes.keys() {
            if key == abs_dir || key.starts_with(&prefix) {
                out.push(key.clone());
            }
        }
        out
    }

    pub fn read_link(&self, cwd: &str, path: &str) -> Result<String> {
        let abs = resolve_against(cwd, path);
        let real = self.realpath(&abs, false)?;
        match self.nodes.get(&real) {
            Some(Node {
                kind: NodeKind::Symlink(t),
                ..
            }) => Ok(t.clone()),
            _ => Err(VfsError::Invalid(path.to_string())),
        }
    }

    // ---- writes ----

    fn require_parent_dir(&self, abs: &str) -> Result<()> {
        let parent = parent_of(abs).unwrap_or_else(|| "/".to_string());
        let real_parent = self.realpath(&parent, true)?;
        match self.nodes.get(&real_parent) {
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
        let before = self.nodes.clone();
        self.require_parent_dir(&target)?;
        match self.nodes.get_mut(&target) {
            Some(Node {
                kind: NodeKind::File(d),
                mtime,
                ..
            }) => {
                *d = data.to_vec();
                *mtime = self.mutation_time_ms;
            }
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => return Err(VfsError::IsADir(path.to_string())),
            _ => {
                self.nodes.insert(
                    target,
                    Node::file(data.to_vec(), mode, self.mutation_time_ms),
                );
            }
        }
        self.finish_mutation(before)
    }

    pub fn append(&mut self, cwd: &str, path: &str, data: &[u8], mode: Mode) -> Result<()> {
        let target = self.write_target(cwd, path)?;
        let before = self.nodes.clone();
        self.require_parent_dir(&target)?;
        match self.nodes.get_mut(&target) {
            Some(Node {
                kind: NodeKind::File(d),
                mtime,
                ..
            }) => {
                d.extend_from_slice(data);
                *mtime = self.mutation_time_ms;
            }
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => return Err(VfsError::IsADir(path.to_string())),
            _ => {
                self.nodes.insert(
                    target,
                    Node::file(data.to_vec(), mode, self.mutation_time_ms),
                );
            }
        }
        self.finish_mutation(before)
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
        let existing_len = match self.nodes.get(&target) {
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
        if self.disk_used.saturating_add(growth) > self.disk_limit {
            return Err(VfsError::NoSpace);
        }

        let before = self.nodes.clone();
        let node = self.nodes.get_mut(&target).expect("file was checked above");
        let NodeKind::File(bytes) = &mut node.kind else {
            unreachable!("file was checked above")
        };
        if end > bytes.len() {
            bytes.resize(end, 0);
        }
        bytes[offset..end].copy_from_slice(data);
        node.mtime = self.mutation_time_ms;
        self.finish_mutation(before)
    }

    pub fn mkdir(&mut self, cwd: &str, path: &str) -> Result<()> {
        let target = self.write_target(cwd, path)?;
        let before = self.nodes.clone();
        if self.nodes.contains_key(&target) {
            return Err(VfsError::Exists(path.to_string()));
        }
        self.require_parent_dir(&target)?;
        self.nodes
            .insert(target, Node::dir(0o755, self.mutation_time_ms));
        self.finish_mutation(before)
    }

    pub fn mkdir_all(&mut self, cwd: &str, path: &str) -> Result<()> {
        let abs = resolve_against(cwd, path);
        reject_pseudo_mutation(&abs)?;
        let before = self.nodes.clone();
        let comps: Vec<&str> = abs.split('/').filter(|c| !c.is_empty()).collect();
        // resolve symlinks progressively
        let mut cur = "/".to_string();
        for c in comps {
            let next = if cur == "/" {
                format!("/{c}")
            } else {
                format!("{cur}/{c}")
            };
            let real = self.realpath(&next, true).unwrap_or(next.clone());
            match self.nodes.get(&real) {
                Some(Node {
                    kind: NodeKind::Dir,
                    ..
                }) => {}
                Some(_) => {
                    self.nodes = before;
                    self.refresh_usage();
                    return Err(VfsError::NotADir(real));
                }
                None => {
                    self.nodes
                        .insert(real.clone(), Node::dir(0o755, self.mutation_time_ms));
                }
            }
            cur = real;
        }
        self.finish_mutation(before)
    }

    pub fn remove_file(&mut self, cwd: &str, path: &str) -> Result<()> {
        let abs = resolve_against(cwd, path);
        reject_pseudo_mutation(&abs)?;
        let real = self.realpath(&abs, false)?;
        match self.nodes.get(&real) {
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => Err(VfsError::IsADir(path.to_string())),
            Some(_) => {
                self.nodes.remove(&real);
                self.refresh_usage();
                Ok(())
            }
            None => Err(VfsError::NotFound(path.to_string())),
        }
    }

    pub fn remove_all(&mut self, cwd: &str, path: &str) -> Result<()> {
        let abs = resolve_against(cwd, path);
        reject_pseudo_mutation(&abs)?;
        let real = self.realpath(&abs, false)?;
        if !self.nodes.contains_key(&real) {
            return Err(VfsError::NotFound(path.to_string()));
        }
        for k in self.walk(&real) {
            self.nodes.remove(&k);
        }
        self.refresh_usage();
        Ok(())
    }

    pub fn rmdir(&mut self, cwd: &str, path: &str) -> Result<()> {
        let abs = resolve_against(cwd, path);
        reject_pseudo_mutation(&abs)?;
        let real = self.realpath(&abs, true)?;
        match self.nodes.get(&real) {
            Some(Node {
                kind: NodeKind::Dir,
                ..
            }) => {
                if !self.list_dir(cwd, path)?.is_empty() {
                    return Err(VfsError::NotEmpty(path.to_string()));
                }
                self.nodes.remove(&real);
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
        if !self.nodes.contains_key(&from_real) {
            return Err(VfsError::NotFound(from.to_string()));
        }
        let mut to_target = self.write_target(cwd, to)?;
        let before = self.nodes.clone();
        // moving into an existing directory
        if matches!(
            self.nodes.get(&to_target),
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
        for k in subtree {
            let node = self.nodes.remove(&k).unwrap();
            let suffix = &k[from_real.len()..];
            let newk = format!("{to_target}{suffix}");
            self.nodes.insert(newk, node);
        }
        self.finish_mutation(before)
    }

    pub fn copy_file(&mut self, cwd: &str, from: &str, to: &str) -> Result<()> {
        reject_pseudo_mutation(&resolve_against(cwd, to))?;
        let data = self.read(cwd, from)?;
        let mode = self.metadata(cwd, from, true)?.mode;
        self.write(cwd, to, &data, mode)
    }

    pub fn copy_recursive(&mut self, cwd: &str, from: &str, to: &str) -> Result<()> {
        let from_abs = resolve_against(cwd, from);
        let from_real = self.realpath(&from_abs, true)?;
        if !self.is_dir(cwd, from) {
            return self.copy_file(cwd, from, to);
        }
        let mut to_target = self.write_target(cwd, to)?;
        let before = self.nodes.clone();
        if matches!(
            self.nodes.get(&to_target),
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
        for k in self.walk(&from_real) {
            let mut node = self.nodes.get(&k).unwrap().clone();
            node.mtime = self.mutation_time_ms;
            let suffix = &k[from_real.len()..];
            let newk = format!("{to_target}{suffix}");
            self.nodes.insert(newk, node);
        }
        self.finish_mutation(before)
    }

    pub fn symlink(&mut self, cwd: &str, target: &str, linkpath: &str) -> Result<()> {
        let link_target = self.write_target(cwd, linkpath)?;
        let before = self.nodes.clone();
        if self.nodes.contains_key(&link_target) {
            return Err(VfsError::Exists(linkpath.to_string()));
        }
        self.require_parent_dir(&link_target)?;
        self.nodes.insert(
            link_target,
            Node {
                kind: NodeKind::Symlink(target.to_string()),
                mode: 0o777,
                uid: 0,
                gid: 0,
                mtime: self.mutation_time_ms,
            },
        );
        self.finish_mutation(before)
    }

    pub fn touch(&mut self, cwd: &str, path: &str, mtime: u64) -> Result<()> {
        let target = self.write_target(cwd, path)?;
        let before = self.nodes.clone();
        match self.nodes.get_mut(&target) {
            Some(n) => n.mtime = mtime,
            None => {
                self.require_parent_dir(&target)?;
                let mut n = Node::file(Vec::new(), 0o644, mtime);
                n.mtime = mtime;
                self.nodes.insert(target, n);
            }
        }
        self.finish_mutation(before)
    }

    pub fn chmod(&mut self, cwd: &str, path: &str, mode: Mode) -> Result<()> {
        let abs = resolve_against(cwd, path);
        reject_pseudo_mutation(&abs)?;
        let real = self.realpath(&abs, true)?;
        self.nodes
            .get_mut(&real)
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
            .nodes
            .get_mut(&real)
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
        if let Some(n) = self.nodes.get_mut(abs_real) {
            n.mtime = mtime;
        }
    }

    // ---- bulk load / dump (bridge to real dirs for task loading & snapshots) ----

    /// Insert a file directly at an absolute path, creating parent dirs.
    pub fn put_file(&mut self, abs: &str, data: Vec<u8>, mode: Mode) -> Result<()> {
        let norm = normalize(abs);
        reject_pseudo_mutation(&norm)?;
        let before = self.nodes.clone();
        if let Some(parent) = parent_of(&norm) {
            self.mkdir_all("/", &parent)?;
        }
        self.nodes
            .insert(norm, Node::file(data, mode, self.mutation_time_ms));
        self.finish_mutation(before)
    }

    pub fn put_dir(&mut self, abs: &str, mode: Mode) -> Result<()> {
        let norm = normalize(abs);
        reject_pseudo_mutation(&norm)?;
        let before = self.nodes.clone();
        self.mkdir_all("/", &norm)?;
        if let Some(n) = self.nodes.get_mut(&norm) {
            n.mode = mode;
        }
        self.finish_mutation(before)
    }

    pub fn all_paths(&self) -> impl Iterator<Item = (&String, &Node)> {
        self.nodes.iter()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 1
    }
}

const NODE_OVERHEAD: u64 = 256;

fn logical_usage(nodes: &BTreeMap<String, Node>) -> u64 {
    nodes
        .iter()
        .filter(|(path, _)| path.as_str() != "/")
        .fold(0u64, |total, (_, node)| {
            let payload = match &node.kind {
                NodeKind::File(data) => data.len() as u64,
                NodeKind::Symlink(target) => target.len() as u64,
                NodeKind::Dir => 0,
            };
            total.saturating_add(NODE_OVERHEAD).saturating_add(payload)
        })
}

#[cfg(test)]
mod tests {
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
        v.copy_recursive("/", "/src", "/dst").unwrap();
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
