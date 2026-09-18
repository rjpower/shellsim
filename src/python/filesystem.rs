//! Metered Python access to shellsim's virtual filesystem.
//!
//! This module is the capability boundary between Python compatibility modules and the VFS. It
//! owns path resolution, mutation-time synchronization, bounded reads, quota error translation,
//! and glob accounting. It never accesses the ambient host filesystem.

use std::path::Path;

use crate::interp::Interp;
use crate::vfs::{resolve_against, VfsError};

use super::native::{PyError, PyFileMetadata, PyFilesystem, PyResult};

const MAX_TEXT_FILE: usize = 4 * 1024 * 1024;
const MODELED_GLOB_RESULT_BYTES: usize = 64;

/// Preserve ordinary Python filesystem exception boundaries while keeping resource exhaustion
/// distinct for the shellsim harness.
fn map_vfs_error(error: VfsError) -> PyError {
    let message = error.to_string();
    match error {
        VfsError::NotFound(_) => PyError::exception("FileNotFoundError", message),
        VfsError::NotADir(_) => PyError::exception("NotADirectoryError", message),
        VfsError::IsADir(_) => PyError::exception("IsADirectoryError", message),
        VfsError::Exists(_) => PyError::exception("FileExistsError", message),
        VfsError::ReadOnly(_) => PyError::exception("PermissionError", message),
        VfsError::NoSpace | VfsError::TooLarge { .. } => PyError::resource_error(message),
        VfsError::NotEmpty(_) | VfsError::Loop(_) | VfsError::Invalid(_) => {
            PyError::exception("OSError", message)
        }
    }
}

/// Capability used by the VM to discover Python source without learning VFS layout policy.
pub(super) trait PyModuleLoader {
    fn load_module_source(
        &mut self,
        roots: &[String],
        module: &str,
    ) -> PyResult<Option<(String, String)>>;
}

fn charge_cpu(interp: &mut Interp, units: usize) -> PyResult<()> {
    let units = u64::try_from(units).unwrap_or(u64::MAX);
    interp
        .resources
        .charge_cpu(units)
        .then_some(())
        .ok_or_else(|| PyError::resource_error("Python CPU limit exceeded"))
}

fn reserve_memory(interp: &mut Interp, bytes: usize) -> PyResult<()> {
    let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
    interp
        .resources
        .reserve_memory(bytes)
        .then_some(())
        .ok_or_else(|| PyError::resource_error("Python memory limit exceeded"))
}

impl PyFilesystem for Interp {
    fn current_dir(&self) -> String {
        self.cwd.clone()
    }

    fn change_dir(&mut self, path: &str) -> PyResult<()> {
        let absolute = resolve_against(&self.cwd, path);
        let resolved = self.vfs.realpath(&absolute, true).map_err(map_vfs_error)?;
        let node = self
            .fs_metadata("/", &resolved, true)
            .map_err(map_vfs_error)?;
        if !matches!(node.kind, crate::vfs::NodeKind::Dir) {
            return Err(PyError::exception(
                "NotADirectoryError",
                format!("Not a directory: {path}"),
            ));
        }
        self.set_var("PWD", resolved);
        Ok(())
    }

    fn read_text(&mut self, path: &str) -> PyResult<String> {
        let length = self.fs_file_len(&self.cwd, path).map_err(map_vfs_error)?;
        if length > MAX_TEXT_FILE {
            return Err(PyError::resource_error("text file exceeds the 4 MiB limit"));
        }
        reserve_memory(self, length)?;
        charge_cpu(self, length)?;
        let bytes = self
            .fs_read_limited(&self.cwd, path, MAX_TEXT_FILE)
            .map_err(map_vfs_error)?;
        String::from_utf8(bytes)
            .map_err(|_| PyError::runtime_error(format!("file is not UTF-8: {path}")))
    }

    fn write_text(&mut self, path: &str, contents: &str) -> PyResult<()> {
        reserve_memory(self, contents.len())?;
        charge_cpu(self, contents.len())?;
        let path = resolve_against(&self.cwd, path);
        self.sync_vfs_time();
        self.vfs
            .put_file(&path, contents.as_bytes().to_vec(), 0o644)
            .map_err(map_vfs_error)
    }

    fn append_text(&mut self, path: &str, contents: &str) -> PyResult<usize> {
        self.append_python_file(path, contents.as_bytes())
    }

    fn read_bytes(&mut self, path: &str) -> PyResult<Vec<u8>> {
        let length = self.fs_file_len(&self.cwd, path).map_err(map_vfs_error)?;
        if length > MAX_TEXT_FILE {
            return Err(PyError::resource_error(
                "binary file exceeds the 4 MiB limit",
            ));
        }
        reserve_memory(self, length)?;
        charge_cpu(self, length)?;
        self.fs_read_limited(&self.cwd, path, MAX_TEXT_FILE)
            .map_err(map_vfs_error)
    }

    fn write_bytes(&mut self, path: &str, contents: &[u8]) -> PyResult<()> {
        reserve_memory(self, contents.len())?;
        charge_cpu(self, contents.len())?;
        let path = resolve_against(&self.cwd, path);
        self.sync_vfs_time();
        self.vfs
            .put_file(&path, contents.to_vec(), 0o644)
            .map_err(map_vfs_error)
    }

    fn append_bytes(&mut self, path: &str, contents: &[u8]) -> PyResult<usize> {
        self.append_python_file(path, contents)
    }

    fn remove_file(&mut self, path: &str) -> PyResult<()> {
        self.sync_vfs_time();
        let cwd = self.cwd.clone();
        self.vfs.remove_file(&cwd, path).map_err(map_vfs_error)
    }

    fn remove_tree(&mut self, path: &str) -> PyResult<()> {
        self.sync_vfs_time();
        let cwd = self.cwd.clone();
        self.vfs.remove_all(&cwd, path).map_err(map_vfs_error)
    }

    fn rename(&mut self, source: &str, destination: &str) -> PyResult<()> {
        self.sync_vfs_time();
        let cwd = self.cwd.clone();
        self.vfs
            .rename(&cwd, source, destination)
            .map_err(map_vfs_error)
    }

    fn exists(&self, path: &str) -> bool {
        self.vfs.exists(&self.cwd, path)
    }

    fn is_file(&self, path: &str) -> bool {
        self.vfs.is_file(&self.cwd, path)
    }

    fn is_dir(&self, path: &str) -> bool {
        self.vfs.is_dir(&self.cwd, path)
    }

    fn metadata(&self, path: &str) -> PyResult<PyFileMetadata> {
        let node = self
            .fs_metadata(&self.cwd, path, true)
            .map_err(map_vfs_error)?;
        let size = match &node.kind {
            crate::vfs::NodeKind::File(data) => data.len(),
            crate::vfs::NodeKind::Symlink(target) => target.len(),
            crate::vfs::NodeKind::Dir => 0,
        };
        Ok(PyFileMetadata {
            mode: node.mode,
            size,
        })
    }

    fn mkdir(&mut self, path: &str, parents: bool, exist_ok: bool) -> PyResult<()> {
        if exist_ok && self.vfs.is_dir(&self.cwd, path) {
            return Ok(());
        }
        self.sync_vfs_time();
        let cwd = self.cwd.clone();
        let result = if parents {
            self.vfs.mkdir_all(&cwd, path)
        } else {
            self.vfs.mkdir(&cwd, path)
        };
        result.map_err(map_vfs_error)
    }

    fn glob(&mut self, pattern: &str) -> PyResult<Vec<String>> {
        let relative = !pattern.starts_with('/');
        let absolute_pattern = resolve_against(&self.cwd, pattern);
        let pattern = glob::Pattern::new(&absolute_pattern)
            .map_err(|error| PyError::value_error(format!("invalid glob pattern: {error}")))?;
        let match_options = glob::MatchOptions {
            require_literal_separator: true,
            require_literal_leading_dot: true,
            ..glob::MatchOptions::new()
        };

        // Meter every candidate before matching. Reserving a conservative result bound before
        // collection also prevents a large modeled directory from driving unmetered host growth.
        let candidate_count = self.vfs.all_paths().count();
        charge_cpu(self, candidate_count)?;
        reserve_memory(
            self,
            candidate_count.saturating_mul(MODELED_GLOB_RESULT_BYTES),
        )?;
        let cwd_prefix = format!("{}/", self.cwd.trim_end_matches('/'));
        Ok(self
            .vfs
            .all_paths()
            .filter(|(path, _)| pattern.matches_path_with(Path::new(path), match_options))
            .map(|(path, _)| {
                if relative {
                    path.strip_prefix(&cwd_prefix).unwrap_or(path).to_string()
                } else {
                    path.clone()
                }
            })
            .collect())
    }
}

impl Interp {
    fn append_python_file(&mut self, path: &str, contents: &[u8]) -> PyResult<usize> {
        reserve_memory(self, contents.len())?;
        charge_cpu(self, contents.len())?;
        let path = resolve_against(&self.cwd, path);
        self.sync_vfs_time();
        self.vfs
            .append("/", &path, contents, 0o644)
            .map_err(map_vfs_error)?;
        self.vfs.file_len("/", &path).map_err(map_vfs_error)
    }
}

impl PyModuleLoader for Interp {
    fn load_module_source(
        &mut self,
        roots: &[String],
        module: &str,
    ) -> PyResult<Option<(String, String)>> {
        let attempts = roots.len().saturating_mul(2);
        let root_bytes = roots
            .iter()
            .fold(0usize, |total, root| total.saturating_add(root.len()));
        let candidate_bytes = root_bytes.saturating_mul(2).saturating_add(
            attempts.saturating_mul(module.len().saturating_add("/__init__.py".len())),
        );
        charge_cpu(self, candidate_bytes.saturating_add(attempts))?;
        reserve_memory(self, candidate_bytes)?;
        let relative = module.replace('.', "/");
        for root in roots {
            for suffix in [format!("{relative}.py"), format!("{relative}/__init__.py")] {
                let candidate = resolve_against(root, &suffix);
                if self.vfs.is_file("/", &candidate) {
                    let contents = self.read_text(&candidate)?;
                    return Ok(Some((candidate, contents)));
                }
            }
        }
        Ok(None)
    }
}
