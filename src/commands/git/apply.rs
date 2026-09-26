//! `git apply`, which reuses the `patch` command to work the hunks.
//!
//! Only the index handling lives here: `--index` records the result in the index as well as the
//! working tree, and `--cached` records it in the index alone. `--cached` never touches the
//! working tree: it patches staged blob content purely in memory through
//! [`crate::commands::patch::apply_patch_in_memory`], then commits the changed objects and index
//! in one [`crate::syscalls::System::apply_file_batch`] call.

use std::collections::BTreeSet;

use crate::commands::patch::{apply_patch_in_memory, PatchSource};
use crate::commands::Io;
use crate::syscalls::System;
use crate::vfs::resolve_against;

use super::repo::{self, Tree};
use super::{cannot_write, repo_error, require_index};

pub(crate) fn git_apply(system: &mut dyn System, args: &[String], io: &mut Io) -> i32 {
    let mut stage = false;
    let mut cached = false;
    let mut reverse = false;
    let mut report = None;
    let mut forwarded = Vec::new();
    for argument in args {
        match argument.as_str() {
            "--index" => stage = true,
            "--cached" => {
                stage = true;
                cached = true;
            }
            "-R" | "--reverse" => reverse = true,
            "--stat" | "--numstat" | "--summary" => report = Some(argument.clone()),
            value => forwarded.push(value.to_string()),
        }
    }
    // These read the patch and report on it without touching anything.
    if let Some(report) = report {
        let Some(text) = patch_text(system, &forwarded, io) else {
            return 128;
        };
        return describe(&text, &report, io);
    }
    if reverse {
        let Some(text) = patch_text(system, &forwarded, io) else {
            return 128;
        };
        io.stdin = reverse_patch(&text).into_bytes();
        forwarded.retain(|value| value.starts_with('-'));
    }
    if !stage {
        return crate::commands::patch::apply_unified_diff(system, &forwarded, io);
    }
    let Some(root) = repo::find_repo_root(system) else {
        return repo_error(io);
    };
    if cached {
        return apply_cached(system, &root, &forwarded, io);
    }
    // Plain `--index`: patch the real working tree first, exactly as `apply_unified_diff` does
    // for a bare `git apply` (one atomic `apply_file_batch` call, nothing written on failure),
    // then record what changed in the index.
    let before = match repo::collect_working_tree(system, &root) {
        Ok(work) => work,
        Err(status) => return status,
    };
    let status = crate::commands::patch::apply_unified_diff(system, &forwarded, io);
    if status != 0 {
        return status;
    }
    let after = match repo::collect_working_tree(system, &root) {
        Ok(work) => work,
        Err(status) => return status,
    };
    let mut names: BTreeSet<String> = before.keys().cloned().collect();
    names.extend(after.keys().cloned());
    let changed: Vec<(String, Option<Vec<u8>>)> = names
        .into_iter()
        .filter(|path| before.get(path) != after.get(path))
        .map(|path| {
            let content = after
                .get(&path)
                .and_then(|_| repo::read_work_file(system, &root, &path));
            (path, content)
        })
        .collect();
    let mut index = match require_index(system, &root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    for (path, content) in changed {
        match content {
            Some(data) => match repo::write_blob(system, &root, &data) {
                Ok(hash) => {
                    // A patch changes content, never the mode the index already records.
                    let recorded = index.get(&path).cloned().unwrap_or_default();
                    index.insert(path, repo::Entry { hash, ..recorded });
                }
                Err(error) => {
                    io.print_err(&format!("git apply: {error}\n"));
                    return 1;
                }
            },
            None => {
                index.remove(&path);
            }
        }
    }
    if let Err(error) = repo::store_index(system, &root, &index) {
        return cannot_write(io, "the index", &error);
    }
    0
}

/// `git apply --cached`: patch what is staged, never the working tree.
///
/// Each patched path's prior content comes straight from the object store through its staged
/// blob hash (via [`StagedSource`]), so nothing is written to, or read back from, the working
/// tree. The whole patch is computed and validated in memory by
/// [`crate::commands::patch::apply_patch_in_memory`] before anything is committed; the changed
/// objects and the new index are then written together in one [`System::apply_file_batch`] call,
/// so a failing hunk leaves the index exactly as it was.
fn apply_cached(system: &mut dyn System, root: &str, forwarded: &[String], io: &mut Io) -> i32 {
    let mut index = match require_index(system, root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    let Some(text) = patch_text(system, forwarded, io) else {
        return 128;
    };
    let strip = strip_count(forwarded);
    // Only tracked paths are patchable; an untracked path stays untracked.
    let mut tracked = repo::head_tree(system, root);
    tracked.extend(index.clone());
    let cwd = system.cwd().to_string();
    let patched = {
        let mut source = StagedSource {
            system,
            root,
            cwd: &cwd,
            tracked: &tracked,
        };
        apply_patch_in_memory(&text, strip, &mut source)
    };
    let patched = match patched {
        Ok(patched) => patched,
        Err(error) => {
            let message = error.strip_prefix("patch: ").unwrap_or(&error);
            io.print_err(&format!("error: {message}\n"));
            io.print_err("error: patch does not apply\n");
            return 1;
        }
    };
    let mut changes = Vec::new();
    for entry in patched {
        let absolute = resolve_against(&cwd, &entry.path);
        let Some(relative) = repo::relative_path(root, &absolute) else {
            io.print_err(&format!("error: {}: outside the repository\n", entry.path));
            return 1;
        };
        match entry.content {
            Some(data) => {
                let recorded = index.get(&relative).cloned().unwrap_or_default();
                let (hash, change) = repo::blob_change(root, data);
                changes.push(change);
                index.insert(relative, repo::Entry { hash, ..recorded });
            }
            None => {
                index.remove(&relative);
            }
        }
    }
    changes.push(repo::store_index_change(root, &index));
    match system.apply_file_batch("/", changes) {
        Ok(()) => 0,
        Err(error) => {
            io.print_err(&format!("git apply: {error}\n"));
            1
        }
    }
}

/// The `-p` strip count `git apply` uses when reading a patch's file headers, defaulting to `1`
/// as Git does (an explicit `patch` command defaults to `0` instead, which is why this is not
/// shared with it).
fn strip_count(forwarded: &[String]) -> usize {
    forwarded
        .iter()
        .find_map(|value| {
            value
                .strip_prefix("-p")
                .and_then(|number| number.parse().ok())
        })
        .unwrap_or(1)
}

/// Reads a staged path's prior content straight from its blob in the object store, so
/// `git apply --cached` never has to materialize it into the working tree.
struct StagedSource<'a> {
    system: &'a mut dyn System,
    root: &'a str,
    cwd: &'a str,
    tracked: &'a Tree,
}

impl PatchSource for StagedSource<'_> {
    fn read(&mut self, path: &str) -> Result<Vec<u8>, String> {
        let absolute = resolve_against(self.cwd, path);
        let relative = repo::relative_path(self.root, &absolute)
            .ok_or_else(|| format!("{path}: outside the repository"))?;
        let entry = self
            .tracked
            .get(&relative)
            .ok_or_else(|| format!("{path}: no such file"))?;
        repo::read_blob(self.system, self.root, &entry.hash)
            .ok_or_else(|| format!("{path}: object missing"))
    }
}

/// The patch text, from the first file operand or from standard input.
fn patch_text(system: &mut dyn System, args: &[String], io: &mut Io) -> Option<String> {
    let Some(file) = args
        .iter()
        .find(|value| !value.starts_with('-'))
        .filter(|value| *value != "-")
    else {
        return Some(String::from_utf8_lossy(&io.stdin).into_owned());
    };
    let absolute = crate::vfs::resolve_against(system.cwd(), file);
    match system.read_file_limited("/", &absolute, 16 * 1024 * 1024) {
        Ok(bytes) => Some(String::from_utf8_lossy(&bytes).into_owned()),
        Err(_) => {
            io.print_err(&format!(
                "error: can't open patch '{file}': No such file or directory\n"
            ));
            None
        }
    }
}

/// The name and line counts each file in a patch carries.
fn patch_counts(text: &str) -> Vec<(String, usize, usize)> {
    let mut files: Vec<(String, usize, usize)> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            let name = rest.split('\t').next().unwrap_or(rest);
            let name = name.split_once('/').map_or(name, |(_, tail)| tail);
            if name != "dev/null" {
                files.push((name.to_string(), 0, 0));
            }
            continue;
        }
        let Some(entry) = files.last_mut() else {
            continue;
        };
        if line.starts_with("+++") || line.starts_with("---") {
            continue;
        }
        if line.starts_with('+') {
            entry.1 += 1;
        } else if line.starts_with('-') {
            entry.2 += 1;
        }
    }
    files
}

/// Report a patch the way `--stat`, `--numstat`, and `--summary` do.
fn describe(text: &str, report: &str, io: &mut Io) -> i32 {
    let files = patch_counts(text);
    if report == "--numstat" {
        for (name, insertions, deletions) in &files {
            io.print(&format!("{insertions}\t{deletions}\t{name}\n"));
        }
        return 0;
    }
    if report == "--summary" {
        return 0;
    }
    let width = files.iter().map(|entry| entry.0.len()).max().unwrap_or(0);
    let mut insertions = 0;
    let mut deletions = 0;
    for (name, added, removed) in &files {
        insertions += added;
        deletions += removed;
        let total = added + removed;
        io.print(&format!(
            " {name:width$} | {total:>4} {}{}\n",
            "+".repeat(*added),
            "-".repeat(*removed)
        ));
    }
    io.print(&super::compare::summary_line(
        files.len(),
        insertions,
        deletions,
    ));
    0
}

/// Turn a unified diff around so that applying it undoes the original.
fn reverse_patch(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = String::new();
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        // The two file headers swap places as well as their markers.
        if let (Some(from), Some(to)) = (
            line.strip_prefix("--- "),
            lines
                .get(index + 1)
                .and_then(|next| next.strip_prefix("+++ ")),
        ) {
            out.push_str(&format!("--- {to}\n+++ {from}\n"));
            index += 2;
            continue;
        }
        let flipped = if let Some(rest) = line.strip_prefix("@@ ") {
            // `@@ -A,B +C,D @@` becomes `@@ -C,D +A,B @@`.
            match rest.split_once(" @@") {
                Some((ranges, tail)) => match ranges.split_once(' ') {
                    Some((old, new)) => format!(
                        "@@ -{} +{} @@{tail}",
                        new.trim_start_matches('+'),
                        old.trim_start_matches('-')
                    ),
                    None => line.to_string(),
                },
                None => line.to_string(),
            }
        } else if let Some(rest) = line.strip_prefix('+') {
            format!("-{rest}")
        } else if let Some(rest) = line.strip_prefix('-') {
            format!("+{rest}")
        } else {
            line.to_string()
        };
        out.push_str(&flipped);
        out.push('\n');
        index += 1;
    }
    out
}
