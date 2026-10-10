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

/// Package conflicts are distinct from I/O and resource failures so callers can preserve
/// their existing resource exception contract.
#[derive(Debug)]
pub enum PackageImportError {
    Conflict(String),
    Other(String),
}

impl From<String> for PackageImportError {
    fn from(message: String) -> Self {
        Self::Other(message)
    }
}

impl std::fmt::Display for PackageImportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conflict(message) | Self::Other(message) => formatter.write_str(message),
        }
    }
}

/// Observable result of importing one trusted host directory.
///
/// `skipped_directories` contains the names from [`DEFAULT_SKIPPED_DIRECTORIES`] that were
/// encountered during this import. The fixed set keeps reporting bounded even when a host tree
/// contains many skipped directories.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MountReport {
    pub files: usize,
    pub skipped_directories: Vec<String>,
    pub git: Option<GitImportReport>,
}

/// Observable result of translating recent HEAD history from a host Git working tree.
///
/// `source_head` is the native Git object ID. `imported_head` differs because Shellsim rewrites
/// commits into its private format; blob IDs remain unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GitImportReport {
    pub commits: usize,
    pub blobs: usize,
    pub tree_entries: usize,
    pub truncated_history: bool,
    pub source_head: String,
    pub imported_head: String,
}

/// Import a canonicalized host directory at one absolute VFS destination.
pub fn mount_host_tree(
    environment: &mut Environment,
    host_root: &Path,
    destination_root: &str,
) -> Result<usize, String> {
    mount_host_tree_report(environment, host_root, destination_root).map(|report| report.files)
}

/// Import a canonicalized host directory and report intentional omissions and Git history.
///
/// The host directory is trusted harness input. The complete VFS mutation is rolled back when
/// traversal, decoding, Git translation, file-count enforcement, or disk accounting fails. A
/// root `.git` file or directory triggers bounded history translation automatically.
pub fn mount_host_tree_report(
    environment: &mut Environment,
    host_root: &Path,
    destination_root: &str,
) -> Result<MountReport, String> {
    mount_host_tree_report_with_policy(environment, host_root, destination_root, true, None)
        .map_err(|error| error.to_string())
}

/// Atomically import package files, rejecting conflicting content and links.
/// Declared tool destinations may replace only their original base-image native commands.
pub fn mount_package_tree(
    environment: &mut Environment,
    host_root: &Path,
    destination_root: &str,
    replace_builtin_tools: &[String],
) -> Result<MountReport, PackageImportError> {
    if replace_builtin_tools.len() > MAX_HOST_FILES {
        return Err(PackageImportError::Other(
            "package tool replacement list exceeds its bound".into(),
        ));
    }
    let tools = replace_builtin_tools.iter().cloned().collect();
    mount_host_tree_report_with_policy(
        environment,
        host_root,
        destination_root,
        false,
        Some(&tools),
    )
}

fn mount_host_tree_report_with_policy(
    environment: &mut Environment,
    host_root: &Path,
    destination_root: &str,
    skip_project_directories: bool,
    package_tools: Option<&BTreeSet<String>>,
) -> Result<MountReport, PackageImportError> {
    let host_root = host_root
        .canonicalize()
        .map_err(|error| format!("cannot resolve host root {}: {error}", host_root.display()))?;
    if !host_root.is_dir() {
        return Err(format!("host root is not a directory: {}", host_root.display()).into());
    }
    if !destination_root.starts_with('/') || destination_root.contains('\0') {
        return Err("VFS destination root must be an absolute safe path"
            .to_string()
            .into());
    }
    let destination_root = crate::vfs::normalize(destination_root);
    let import_git = has_git_metadata(&host_root)?;
    if import_git && !skip_project_directories {
        return Err("verified package staging tree contains Git metadata"
            .to_string()
            .into());
    }
    let before = environment.vfs.clone();
    let result = (|| {
        if package_tools.is_some() {
            environment
                .vfs
                .package_destination_matches(&destination_root, None, 0o755, false)
                .map_err(|error| PackageImportError::Conflict(error.to_string()))?;
        }
        if !environment.vfs.exists("/", &destination_root) {
            environment
                .vfs
                .put_dir(&destination_root, 0o755)
                .map_err(|error| format!("cannot create {destination_root}: {error}"))?;
        }
        let mut report = mount_inner(
            environment,
            &host_root,
            &destination_root,
            skip_project_directories,
            package_tools,
        )?;
        if !skip_project_directories && !report.skipped_directories.is_empty() {
            return Err("verified package staging tree contains Git metadata"
                .to_string()
                .into());
        }
        if import_git {
            let history = crate::commands::git::import::import_head_history(
                environment,
                &host_root,
                &destination_root,
            )?;
            report.git = Some(GitImportReport {
                commits: history.commits,
                blobs: history.blobs,
                tree_entries: history.tree_entries,
                truncated_history: history.truncated_history,
                source_head: history.source_head,
                imported_head: history.imported_head,
            });
        }
        Ok(report)
    })();
    match result {
        Ok(report) => Ok(report),
        Err(error) => {
            environment.vfs = before;
            Err(error)
        }
    }
}

fn has_git_metadata(host_root: &Path) -> Result<bool, String> {
    let marker = host_root.join(".git");
    match fs::symlink_metadata(&marker) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(format!(
            "refusing host symlink during Git discovery: {}",
            marker.display()
        )),
        Ok(metadata) if metadata.is_dir() || metadata.is_file() => Ok(true),
        Ok(_) => Err(format!(
            "Git metadata marker is not a file or directory: {}",
            marker.display()
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!(
            "cannot inspect Git metadata marker {}: {error}",
            marker.display()
        )),
    }
}

fn mount_inner(
    environment: &mut Environment,
    host_root: &Path,
    destination_root: &str,
    skip_project_directories: bool,
    package_tools: Option<&BTreeSet<String>>,
) -> Result<MountReport, PackageImportError> {
    let mut pending = vec![(host_root.to_path_buf(), PathBuf::new())];
    let mut files = 0usize;
    let mut skipped_directories = BTreeSet::new();
    while let Some((host, relative)) = pending.pop() {
        let metadata = fs::symlink_metadata(&host)
            .map_err(|error| format!("cannot inspect {}: {error}", host.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("refusing host symlink {}", host.display()).into());
        }
        let destination = destination_path(destination_root, &relative)?;
        if metadata.is_dir() {
            if destination != destination_root {
                let matches = if package_tools.is_some() {
                    environment
                        .vfs
                        .package_destination_matches(
                            &destination,
                            None,
                            permission_mode(&metadata, true),
                            false,
                        )
                        .map_err(|error| PackageImportError::Conflict(error.to_string()))?
                } else {
                    false
                };
                if !matches {
                    environment
                        .vfs
                        .put_dir(&destination, permission_mode(&metadata, true))
                        .map_err(|error| format!("cannot create {destination}: {error}"))?;
                }
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
                let is_directory = entry.file_type().is_ok_and(|kind| kind.is_dir());
                if name == ".git"
                    || (skip_project_directories
                        && DEFAULT_SKIPPED_DIRECTORIES.contains(&name)
                        && is_directory)
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
                return Err(
                    format!("project exceeds the {MAX_HOST_FILES}-file ingestion limit").into(),
                );
            }
            if metadata.len() > environment.resources.limits().disk {
                return Err(format!(
                    "host file is larger than the configured VFS: {}",
                    host.display()
                )
                .into());
            }
            let contents = fs::read(&host)
                .map_err(|error| format!("cannot read {}: {error}", host.display()))?;
            if let Some(tools) = package_tools {
                if environment
                    .vfs
                    .package_destination_matches(
                        &destination,
                        Some(&contents),
                        permission_mode(&metadata, false),
                        tools.contains(&destination),
                    )
                    .map_err(|error| PackageImportError::Conflict(error.to_string()))?
                {
                    continue;
                }
            }
            environment
                .vfs
                .put_file(&destination, contents, permission_mode(&metadata, false))
                .map_err(|error| format!("cannot import {}: {error}", host.display()))?;
        } else {
            return Err(format!("unsupported host file type: {}", host.display()).into());
        }
    }
    Ok(MountReport {
        files,
        skipped_directories: skipped_directories.into_iter().collect(),
        git: None,
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
