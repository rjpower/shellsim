//! Bounded tar export of committed regular files, without reading working-tree content.
//!
//! The shared tar encoder limits archive size and charges output allocation. Symbolic links and
//! alternate archive formats remain explicit frontiers because the extractor does not model them.

use crate::commands::tarcmd::{self, Entry, EntryKind};
use crate::commands::Io;
use crate::syscalls::System;

use super::{fatal, ignore, repo, repo_error, usage, Arg, Flags};

/// Export a commit's selected paths, retaining repository-relative names and executable modes.
pub(crate) fn git_archive(system: &mut dyn System, args: &[String], io: &mut Io) -> i32 {
    let mut operands = Vec::new();
    let mut flags = Flags::new(args);
    while let Some(argument) = flags.next() {
        match argument {
            Arg::Operand(value) => operands.push(value),
            Arg::Option { name, attached } if name == "--format" => {
                if flags.value(attached).as_deref() != Some("tar") {
                    return usage(io, "unsupported archive format; only tar is supported");
                }
            }
            Arg::Option { name, .. } => {
                return usage(io, &format!("unsupported archive option: {name}"));
            }
        }
    }
    let Some(revision) = operands.first() else {
        return usage(io, "usage: git archive [--format=tar] COMMIT [-- PATH...]");
    };
    let Some(root) = repo::find_repo_root(system) else {
        return repo_error(io);
    };
    let Some(id) = repo::resolve_revision(system, &root, revision) else {
        return fatal(io, &format!("not a valid object name: {revision}"));
    };
    let Some(tree) = repo::commit_tree(system, &root, &id) else {
        return fatal(io, "cannot read commit tree");
    };
    let prefix = repo::relative_path(&root, system.cwd()).unwrap_or_default();
    let paths: Vec<String> = operands[1..]
        .iter()
        .map(|path| {
            if prefix.is_empty() {
                path.clone()
            } else {
                format!("{prefix}/{path}")
            }
        })
        .collect();
    if !system
        .charge_cpu((tree.len() as u64).saturating_mul((paths.len() as u64).saturating_add(1)))
    {
        return repo::resource_error(system);
    }
    for path in &paths {
        if !tree.keys().any(|name| ignore::matches_pathspec(path, name)) {
            return fatal(io, &format!("pathspec '{path}' did not match any files"));
        }
    }
    let mark = system.memory_used();
    let result = encode_commit(system, &root, &tree, &paths, &prefix);
    system.release_memory(system.memory_used().saturating_sub(mark));
    match result {
        Ok(bytes) => {
            *io.out = bytes;
            0
        }
        Err(message) => {
            if message == "resource limit exceeded" || message == "memory limit exceeded" {
                return repo::resource_error(system);
            }
            fatal(io, &message)
        }
    }
}

/// Reserve each blob before reading it; never collect more than the tar container can hold.
fn encode_commit(
    system: &mut dyn System,
    root: &str,
    tree: &repo::Tree,
    paths: &[String],
    prefix: &str,
) -> Result<Vec<u8>, String> {
    let mut entries = Vec::new();
    let mut bytes = 0u64;
    let prefix = if prefix.is_empty() {
        String::new()
    } else {
        format!("{prefix}/")
    };
    for (name, entry) in tree {
        let Some(displayed) = name.strip_prefix(&prefix) else {
            continue;
        };
        if !paths.is_empty()
            && !paths
                .iter()
                .any(|path| ignore::matches_pathspec(path, name))
        {
            continue;
        }
        if entries.len() >= 4096 {
            return Err("archive exceeds the 4096-entry limit".into());
        }
        if entry.symlink {
            return Err("unsupported archive entry: symbolic link".into());
        }
        let blob = repo::git_path(root, &format!("objects/{}", entry.hash));
        let size = system
            .metadata("/", &blob, true)
            .map_err(|_| "cannot read archive blob")?
            .size;
        bytes = bytes.checked_add(size).ok_or("archive is too large")?;
        if bytes > 16 * 1024 * 1024 {
            return Err("archive exceeds the 16 MiB limit".into());
        }
        if !system.reserve_memory(size.saturating_add(name.len() as u64).saturating_add(128))
            || !system.charge_cpu(size.saturating_add(1))
        {
            return Err("resource limit exceeded".into());
        }
        let data = system
            .read_file_limited("/", &blob, size as usize)
            .map_err(|_| "cannot read archive blob")?;
        entries.push(Entry {
            name: displayed.to_string(),
            mode: repo::work_mode(entry.executable),
            kind: EntryKind::File(data),
        });
    }
    tarcmd::encode(system, &entries)
}
