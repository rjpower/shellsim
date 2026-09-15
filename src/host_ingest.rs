//! Trusted, bounded host-directory ingestion into a simulated VFS.
//!
//! CLI adapters may call this before simulated execution begins. The walk rejects symlinks and
//! non-UTF-8 names, skips common dependency/build directories, preserves basic permission bits,
//! obeys the destination VFS quota, and rolls back the complete import on error. Simulated code
//! has no reference to the host root or access to this adapter.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::Environment;

pub const MAX_HOST_FILES: usize = 10_000;
pub const DEFAULT_SKIPPED_DIRECTORIES: &[&str] = &[
    ".git",
    ".venv",
    "venv",
    "target",
    "node_modules",
    "__pycache__",
];

/// Observable result of importing one trusted host directory.
///
/// `skipped_directories` contains the names from [`DEFAULT_SKIPPED_DIRECTORIES`] that were
/// encountered during this import. The fixed set keeps reporting bounded even when a host tree
/// contains many skipped directories.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MountReport {
    pub files: usize,
    pub skipped_directories: Vec<String>,
}

/// Import a canonicalized host directory at one absolute VFS destination.
pub fn mount_host_tree(
    environment: &mut Environment,
    host_root: &Path,
    destination_root: &str,
) -> Result<usize, String> {
    mount_host_tree_report(environment, host_root, destination_root).map(|report| report.files)
}

/// Import a canonicalized host directory and report intentional directory omissions.
///
/// The host directory is trusted harness input. The complete VFS mutation is rolled back when
/// traversal, decoding, file-count enforcement, or disk accounting fails.
pub fn mount_host_tree_report(
    environment: &mut Environment,
    host_root: &Path,
    destination_root: &str,
) -> Result<MountReport, String> {
    let host_root = host_root
        .canonicalize()
        .map_err(|error| format!("cannot resolve host root {}: {error}", host_root.display()))?;
    if !host_root.is_dir() {
        return Err(format!(
            "host root is not a directory: {}",
            host_root.display()
        ));
    }
    if !destination_root.starts_with('/') || destination_root.contains('\0') {
        return Err("VFS destination root must be an absolute safe path".to_string());
    }
    let destination_root = crate::vfs::normalize(destination_root);
    let before = environment.vfs.clone();
    match mount_inner(environment, &host_root, &destination_root) {
        Ok(report) => Ok(report),
        Err(error) => {
            environment.vfs = before;
            Err(error)
        }
    }
}

fn mount_inner(
    environment: &mut Environment,
    host_root: &Path,
    destination_root: &str,
) -> Result<MountReport, String> {
    let mut pending = vec![(host_root.to_path_buf(), PathBuf::new())];
    let mut files = 0usize;
    let mut skipped_directories = BTreeSet::new();
    while let Some((host, relative)) = pending.pop() {
        let metadata = fs::symlink_metadata(&host)
            .map_err(|error| format!("cannot inspect {}: {error}", host.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("refusing host symlink {}", host.display()));
        }
        let destination = destination_path(destination_root, &relative)?;
        if metadata.is_dir() {
            if destination != destination_root {
                environment
                    .vfs
                    .put_dir(&destination, permission_mode(&metadata, true))
                    .map_err(|error| format!("cannot create {destination}: {error}"))?;
            }
            let mut entries = fs::read_dir(&host)
                .map_err(|error| format!("cannot read directory {}: {error}", host.display()))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("cannot read directory {}: {error}", host.display()))?;
            entries.sort_by_key(|entry| entry.file_name());
            for entry in entries.into_iter().rev() {
                let name = entry.file_name();
                let name = name
                    .to_str()
                    .ok_or_else(|| format!("non-UTF-8 host path below {}", host.display()))?;
                if DEFAULT_SKIPPED_DIRECTORIES.contains(&name)
                    && entry.file_type().is_ok_and(|kind| kind.is_dir())
                {
                    skipped_directories.insert(name.to_string());
                    continue;
                }
                pending.push((entry.path(), relative.join(name)));
            }
        } else if metadata.is_file() {
            files = files
                .checked_add(1)
                .ok_or_else(|| "host file count overflow".to_string())?;
            if files > MAX_HOST_FILES {
                return Err(format!(
                    "project exceeds the {MAX_HOST_FILES}-file ingestion limit"
                ));
            }
            if metadata.len() > environment.resources.limits().disk {
                return Err(format!(
                    "host file is larger than the configured VFS: {}",
                    host.display()
                ));
            }
            let contents = fs::read(&host)
                .map_err(|error| format!("cannot read {}: {error}", host.display()))?;
            environment
                .vfs
                .put_file(&destination, contents, permission_mode(&metadata, false))
                .map_err(|error| format!("cannot import {}: {error}", host.display()))?;
        } else {
            return Err(format!("unsupported host file type: {}", host.display()));
        }
    }
    Ok(MountReport {
        files,
        skipped_directories: skipped_directories.into_iter().collect(),
    })
}

fn destination_path(root: &str, relative: &Path) -> Result<String, String> {
    if relative.as_os_str().is_empty() {
        return Ok(root.to_string());
    }
    let relative = relative
        .to_str()
        .ok_or_else(|| "non-UTF-8 project path".to_string())?;
    Ok(format!(
        "{}/{}",
        root.trim_end_matches('/'),
        relative.replace('\\', "/")
    ))
}

#[cfg(unix)]
fn permission_mode(metadata: &fs::Metadata, _directory: bool) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode() & 0o7777
}

#[cfg(not(unix))]
fn permission_mode(_metadata: &fs::Metadata, directory: bool) -> u32 {
    if directory {
        0o755
    } else {
        0o644
    }
}
