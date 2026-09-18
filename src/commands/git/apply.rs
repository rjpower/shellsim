//! `git apply`, which reuses the `patch` command to work the hunks.
//!
//! Only the index handling lives here: `--index` records the result in the index as well as the
//! working tree, and `--cached` records it in the index alone.

use std::collections::BTreeSet;

use crate::commands::{CommandContext, Io};

use super::repo;
use super::{cannot_write, repo_error, require_index};

pub(crate) fn git_apply(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
        let Some(text) = patch_text(ctx, &forwarded, io) else {
            return 128;
        };
        return describe(&text, &report, io);
    }
    if reverse {
        let Some(text) = patch_text(ctx, &forwarded, io) else {
            return 128;
        };
        io.stdin = reverse_patch(&text).into_bytes();
        forwarded.retain(|value| value.starts_with('-'));
    }
    if !stage {
        return crate::commands::patch::apply_unified_diff(ctx, &forwarded, io);
    }
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let work = match repo::collect_working_tree(ctx, &root) {
        Ok(work) => work,
        Err(status) => return status,
    };
    let unpatched = ctx.vfs.clone();
    // `--cached` patches what is staged, so the index content is laid down to be worked on and
    // the working tree is put back afterwards.
    let before = if cached {
        let staged = match require_index(ctx, &root, io) {
            Ok(index) => index,
            Err(status) => return status,
        };
        // Only tracked paths are laid down; an untracked file, the patch itself included, stays.
        let mut tracked = repo::head_tree(ctx, &root);
        tracked.extend(staged.clone());
        if let Err(error) = repo::replace_work_tree(ctx, &root, &tracked, &staged) {
            io.err
                .extend_from_slice(format!("git apply: {error}\n").as_bytes());
            return 1;
        }
        // What is on disk now: the staged content plus whatever was untracked.
        let mut laid_down = work;
        laid_down.retain(|path, _| !tracked.contains_key(path));
        laid_down.extend(staged);
        laid_down
    } else {
        work
    };
    let status = crate::commands::patch::apply_unified_diff(ctx, &forwarded, io);
    if status != 0 {
        ctx.vfs = unpatched;
        return status;
    }
    let after = match repo::collect_working_tree(ctx, &root) {
        Ok(work) => work,
        Err(status) => return status,
    };
    // The patched content has to be read before `--cached` puts the working tree back.
    let mut names: BTreeSet<String> = before.keys().cloned().collect();
    names.extend(after.keys().cloned());
    let changed: Vec<(String, Option<Vec<u8>>)> = names
        .into_iter()
        .filter(|path| before.get(path) != after.get(path))
        .map(|path| {
            let content = after
                .get(&path)
                .and_then(|_| repo::read_work_file(ctx, &root, &path));
            (path, content)
        })
        .collect();
    if cached {
        ctx.vfs = unpatched;
    }
    let mut index = match require_index(ctx, &root, io) {
        Ok(index) => index,
        Err(status) => return status,
    };
    for (path, content) in changed {
        match content {
            Some(data) => match repo::write_blob(ctx, &root, &data) {
                Ok(hash) => {
                    // A patch changes content, never the mode the index already records.
                    let recorded = index.get(&path).cloned().unwrap_or_default();
                    index.insert(path, repo::Entry { hash, ..recorded });
                }
                Err(error) => {
                    io.err
                        .extend_from_slice(format!("git apply: {error}\n").as_bytes());
                    return 1;
                }
            },
            None => {
                index.remove(&path);
            }
        }
    }
    if let Err(error) = repo::store_index(ctx, &root, &index) {
        return cannot_write(io, "the index", &error);
    }
    0
}

/// The patch text, from the first file operand or from standard input.
fn patch_text(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> Option<String> {
    let Some(file) = args
        .iter()
        .find(|value| !value.starts_with('-'))
        .filter(|value| *value != "-")
    else {
        return Some(String::from_utf8_lossy(&io.stdin).into_owned());
    };
    let absolute = crate::vfs::resolve_against(&ctx.cwd, file);
    match ctx.fs_read_limited("/", &absolute, 16 * 1024 * 1024) {
        Ok(bytes) => Some(String::from_utf8_lossy(&bytes).into_owned()),
        Err(_) => {
            io.err.extend_from_slice(
                format!("error: can't open patch '{file}': No such file or directory\n").as_bytes(),
            );
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
            io.out
                .extend_from_slice(format!("{insertions}\t{deletions}\t{name}\n").as_bytes());
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
        io.out.extend_from_slice(
            format!(
                " {name:width$} | {total:>4} {}{}\n",
                "+".repeat(*added),
                "-".repeat(*removed)
            )
            .as_bytes(),
        );
    }
    io.out.extend_from_slice(
        super::compare::summary_line(files.len(), insertions, deletions).as_bytes(),
    );
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
