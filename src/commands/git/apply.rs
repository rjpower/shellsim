//! `git apply`, which reuses the `patch` command to work the hunks.
//!
//! Only the index handling lives here: `--index` records the result in the index as well as the
//! working tree, and `--cached` records it in the index alone.

use std::collections::BTreeSet;

use crate::commands::{CommandContext, Io};

use super::repo;
use super::repo_error;

pub(crate) fn git_apply(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut stage = false;
    let mut cached = false;
    let mut forwarded = Vec::new();
    for argument in args {
        match argument.as_str() {
            "--index" => stage = true,
            "--cached" => {
                stage = true;
                cached = true;
            }
            value => forwarded.push(value.to_string()),
        }
    }
    if !stage {
        return crate::commands::patch::apply_unified_diff(ctx, &forwarded, io);
    }
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let work = match repo::collect_working_tree(ctx, &root) {
        Ok(snapshot) => snapshot.release(ctx),
        Err(status) => return status,
    };
    let unpatched = ctx.vfs.clone();
    // `--cached` patches what is staged, so the index content is laid down to be worked on and
    // the working tree is put back afterwards.
    let before = if cached {
        let staged = repo::load_index(ctx, &root).unwrap_or_default();
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
        Ok(snapshot) => snapshot.release(ctx),
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
    let mut index = repo::load_index(ctx, &root).unwrap_or_default();
    for (path, content) in changed {
        match content {
            Some(data) => match repo::write_blob(ctx, &root, &data) {
                Ok(hash) => {
                    index.insert(path, hash);
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
    if repo::store_index(ctx, &root, &index).is_err() {
        return 1;
    }
    0
}
