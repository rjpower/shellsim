//! Three-way merging and the unmerged state it leaves behind.
//!
//! A merge that cannot be resolved by taking one side writes conflict markers into the working
//! tree and records the three sides of each conflicted path in `.git/MERGE_STAGES`, alongside a
//! `.git/MERGE_HEAD` naming the commit being merged. That is what lets `git status` report
//! `UU`, `git add` mark a path resolved, and `git commit` finish the merge.

use std::collections::BTreeMap;

use crate::commands::CommandContext;
use crate::interp::Interp;

use super::diff::{self, Op};
use super::repo::{self, Tree};

/// Files under `.git` that record an operation waiting for the user to finish it.
pub(crate) const MERGE_HEAD: &str = "MERGE_HEAD";
pub(crate) const CHERRY_PICK_HEAD: &str = "CHERRY_PICK_HEAD";
pub(crate) const REVERT_HEAD: &str = "REVERT_HEAD";
const MERGE_STAGES: &str = "MERGE_STAGES";
const MERGE_MSG: &str = "MERGE_MSG";

/// The three sides recorded for one unmerged path.
#[derive(Clone, Debug, Default)]
pub(crate) struct Unmerged {
    pub base: Option<String>,
    pub ours: Option<String>,
    pub theirs: Option<String>,
}

impl Unmerged {
    /// The two-letter code `git status --short` prints for this path.
    pub fn porcelain(&self) -> &'static str {
        match (
            self.ours.is_some(),
            self.theirs.is_some(),
            self.base.is_some(),
        ) {
            (true, true, false) => "AA",
            (true, true, true) => "UU",
            (true, false, _) => "UD",
            (false, true, _) => "DU",
            _ => "DD",
        }
    }

    /// The phrase the long status format uses.
    pub fn label(&self) -> &'static str {
        match self.porcelain() {
            "AA" => "both added",
            "UD" => "deleted by them",
            "DU" => "deleted by us",
            "DD" => "both deleted",
            _ => "both modified",
        }
    }
}

pub(crate) type Stages = BTreeMap<String, Unmerged>;

pub(crate) fn load_stages(interp: &Interp, root: &str) -> Stages {
    let Ok(bytes) = interp.vfs.read("/", &repo::git_path(root, MERGE_STAGES)) else {
        return Stages::new();
    };
    let mut stages = Stages::new();
    for line in String::from_utf8_lossy(&bytes).lines() {
        let mut fields = line.splitn(3, '\t');
        let (Some(stage), Some(hash), Some(path)) = (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let entry = stages.entry(path.to_string()).or_default();
        let hash = Some(hash.to_string());
        match stage {
            "1" => entry.base = hash,
            "2" => entry.ours = hash,
            "3" => entry.theirs = hash,
            _ => {}
        }
    }
    stages
}

pub(crate) fn store_stages(ctx: &mut CommandContext<'_>, root: &str, stages: &Stages) -> bool {
    if stages.is_empty() {
        let _ = ctx
            .vfs
            .remove_file("/", &repo::git_path(root, MERGE_STAGES));
        return true;
    }
    let mut text = String::new();
    for (path, entry) in stages {
        for (stage, hash) in [(1, &entry.base), (2, &entry.ours), (3, &entry.theirs)] {
            if let Some(hash) = hash {
                text.push_str(&format!("{stage}\t{hash}\t{path}\n"));
            }
        }
    }
    repo::write_vfs(ctx, &repo::git_path(root, MERGE_STAGES), text.as_bytes()).is_ok()
}

/// The commit an unfinished merge, cherry-pick, or revert is bringing in.
pub(crate) fn in_progress(interp: &Interp, root: &str, kind: &str) -> Option<String> {
    let bytes = interp.vfs.read("/", &repo::git_path(root, kind)).ok()?;
    let text = String::from_utf8_lossy(&bytes).trim().to_string();
    (!text.is_empty()).then_some(text)
}

pub(crate) fn begin(
    ctx: &mut CommandContext<'_>,
    root: &str,
    kind: &str,
    commit: &str,
    message: &str,
) {
    let _ = repo::write_vfs(
        ctx,
        &repo::git_path(root, kind),
        format!("{commit}\n").as_bytes(),
    );
    let _ = repo::write_vfs(ctx, &repo::git_path(root, MERGE_MSG), message.as_bytes());
}

/// The message recorded for the operation in progress.
pub(crate) fn pending_message(interp: &Interp, root: &str) -> Option<String> {
    let bytes = interp
        .vfs
        .read("/", &repo::git_path(root, MERGE_MSG))
        .ok()?;
    Some(String::from_utf8_lossy(&bytes).trim_end().to_string())
}

/// Mark the named paths resolved, leaving any other conflict untouched.
pub(crate) fn resolve<'a>(
    ctx: &mut CommandContext<'_>,
    root: &str,
    paths: impl IntoIterator<Item = &'a String>,
) {
    let mut stages = load_stages(ctx, root);
    if stages.is_empty() {
        return;
    }
    for path in paths {
        stages.remove(path);
    }
    store_stages(ctx, root, &stages);
}

/// Forget every record of an operation in progress.
pub(crate) fn clear(ctx: &mut CommandContext<'_>, root: &str) {
    for name in [
        MERGE_HEAD,
        CHERRY_PICK_HEAD,
        REVERT_HEAD,
        MERGE_STAGES,
        MERGE_MSG,
    ] {
        let _ = ctx.vfs.remove_file("/", &repo::git_path(root, name));
    }
}

/// The outcome of combining two trees against their base.
pub(crate) struct Combined {
    /// The resolved tree, holding the conflict-marked content for conflicted paths.
    pub tree: Tree,
    /// The three sides of each path that could not be resolved.
    pub stages: Stages,
}

/// Combine `ours` and `theirs` against `base`, merging file content where both sides changed.
pub(crate) fn combine(
    ctx: &mut CommandContext<'_>,
    root: &str,
    base: &Tree,
    ours: &Tree,
    theirs: &Tree,
    ours_label: &str,
    theirs_label: &str,
) -> Combined {
    let mut names: Vec<String> = base.keys().cloned().collect();
    names.extend(ours.keys().cloned());
    names.extend(theirs.keys().cloned());
    names.sort();
    names.dedup();
    let mut tree = Tree::new();
    let mut stages = Stages::new();
    for path in names {
        let original = base.get(&path);
        let mine = ours.get(&path);
        let yours = theirs.get(&path);
        // A side that did not touch the path defers to the other, and two sides that made the
        // same change agree. The outer `None` means unresolved; an inner `None` resolves to a
        // deletion, which is why this is not a plain `Option<&String>`.
        let resolved = if mine == original {
            Some(yours)
        } else if yours == original || mine == yours {
            Some(mine)
        } else {
            None
        };
        if let Some(entry) = resolved {
            if let Some(entry) = entry {
                tree.insert(path, entry.clone());
            }
            continue;
        }
        if mine.is_none() || yours.is_none() {
            // One side deleted what the other changed; Git leaves the surviving content in place
            // and the decision to the user.
            if let Some(entry) = mine.or(yours) {
                tree.insert(path.clone(), entry.clone());
            }
            stages.insert(path, sides(original, mine, yours));
            continue;
        }
        let read = |entry: Option<&repo::Entry>| {
            entry
                .and_then(|entry| repo::read_blob(ctx, root, &entry.hash))
                .unwrap_or_default()
        };
        let (original_text, mine_text, yours_text) = (read(original), read(mine), read(yours));
        if [&original_text, &mine_text, &yours_text]
            .iter()
            .any(|data| diff::is_binary(data))
        {
            // Binary files cannot be merged line by line, so our side stays in the work tree.
            if let Some(entry) = mine.or(yours) {
                tree.insert(path.clone(), entry.clone());
            }
            stages.insert(path, sides(original, mine, yours));
            continue;
        }
        let (merged, conflicted) = merge_content(
            &original_text,
            &mine_text,
            &yours_text,
            ours_label,
            theirs_label,
        );
        let Ok(hash) = repo::write_blob(ctx, root, &merged) else {
            continue;
        };
        if conflicted {
            stages.insert(path.clone(), sides(original, mine, yours));
        }
        // A mode set on either side carries into the merged file.
        let executable = mine.is_some_and(|entry| entry.executable)
            || yours.is_some_and(|entry| entry.executable);
        tree.insert(
            path,
            repo::Entry {
                hash,
                executable,
                symlink: false,
            },
        );
    }
    Combined { tree, stages }
}

/// Record the three sides of a path the merge could not settle.
fn sides(
    base: Option<&repo::Entry>,
    ours: Option<&repo::Entry>,
    theirs: Option<&repo::Entry>,
) -> Unmerged {
    let hash = |entry: Option<&repo::Entry>| entry.map(|entry| entry.hash.clone());
    Unmerged {
        base: hash(base),
        ours: hash(ours),
        theirs: hash(theirs),
    }
}

/// Whether two replaced regions collide, so that neither side's change can simply be taken.
///
/// The regions are half-open, so a change ending where the other begins does not collide. Two
/// insertions at the very same point do, because there is no way to order them.
fn overlap(left: &Hunk, right: &Hunk) -> bool {
    let inserts_at_the_same_point =
        left.start == left.end && right.start == right.end && left.start == right.start;
    left.start.max(right.start) < left.end.min(right.end) || inserts_at_the_same_point
}

/// One region of the base that a side replaced.
struct Hunk {
    start: usize,
    end: usize,
    lines: Vec<String>,
}

/// The regions of `base` that `side` replaced, in order.
fn hunks(base: &[&str], side: &[&str]) -> Vec<Hunk> {
    let mut out: Vec<Hunk> = Vec::new();
    let mut current: Option<Hunk> = None;
    let mut position = 0;
    for edit in diff::edit_script(base, side) {
        match edit.op {
            Op::Keep => {
                if let Some(hunk) = current.take() {
                    out.push(hunk);
                }
                position += 1;
            }
            Op::Delete => {
                let hunk = current.get_or_insert(Hunk {
                    start: position,
                    end: position,
                    lines: Vec::new(),
                });
                position += 1;
                hunk.end = position;
            }
            Op::Insert => {
                current
                    .get_or_insert(Hunk {
                        start: position,
                        end: position,
                        lines: Vec::new(),
                    })
                    .lines
                    .push(edit.text.to_string());
            }
        }
    }
    if let Some(hunk) = current {
        out.push(hunk);
    }
    out
}

/// Apply `hunks` to `base[start..end]`, yielding one side's view of that region.
fn rebuild(base: &[&str], hunks: &[Hunk], start: usize, end: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut position = start;
    for hunk in hunks {
        out.extend(
            base[position..hunk.start]
                .iter()
                .map(|line| line.to_string()),
        );
        out.extend(hunk.lines.iter().cloned());
        position = hunk.end;
    }
    out.extend(base[position..end].iter().map(|line| line.to_string()));
    out
}

/// Merge two versions of a file against their common ancestor.
///
/// Returns the merged bytes and whether any region had to be written as a conflict. Regions the
/// two sides changed differently, or changed with no unchanged line between them, conflict, which
/// is how Git decides as well.
pub(crate) fn merge_content(
    base: &[u8],
    ours: &[u8],
    theirs: &[u8],
    ours_label: &str,
    theirs_label: &str,
) -> (Vec<u8>, bool) {
    let base_lines = diff::split_lines(base);
    let our_hunks = hunks(&base_lines, &diff::split_lines(ours));
    let their_hunks = hunks(&base_lines, &diff::split_lines(theirs));
    let mut out = String::new();
    let mut conflicted = false;
    let mut position = 0;
    let mut mine = 0;
    let mut yours = 0;
    while mine < our_hunks.len() || yours < their_hunks.len() {
        let next_mine = our_hunks.get(mine);
        let next_yours = their_hunks.get(yours);
        let touching = match (next_mine, next_yours) {
            (Some(left), Some(right)) => overlap(left, right),
            _ => false,
        };
        if !touching {
            // Only one side changed this region, or one side's change comes first.
            let take_mine = match (next_mine, next_yours) {
                (Some(left), Some(right)) => left.start <= right.start,
                (Some(_), None) => true,
                _ => false,
            };
            let (hunk, cursor) = if take_mine {
                (&our_hunks[mine], &mut mine)
            } else {
                (&their_hunks[yours], &mut yours)
            };
            out.extend(base_lines[position..hunk.start].iter().copied());
            out.extend(hunk.lines.iter().map(String::as_str));
            position = hunk.end;
            *cursor += 1;
            continue;
        }
        // Both sides changed an overlapping region; grow it until neither side reaches further.
        let mut start = our_hunks[mine].start.min(their_hunks[yours].start);
        let mut end = our_hunks[mine].end.max(their_hunks[yours].end);
        let (mut last_mine, mut last_yours) = (mine, yours);
        loop {
            let before = (last_mine, last_yours);
            while last_mine < our_hunks.len() && our_hunks[last_mine].start < end {
                start = start.min(our_hunks[last_mine].start);
                end = end.max(our_hunks[last_mine].end);
                last_mine += 1;
            }
            while last_yours < their_hunks.len() && their_hunks[last_yours].start < end {
                start = start.min(their_hunks[last_yours].start);
                end = end.max(their_hunks[last_yours].end);
                last_yours += 1;
            }
            if before == (last_mine, last_yours) {
                break;
            }
        }
        let ours_region = rebuild(&base_lines, &our_hunks[mine..last_mine], start, end);
        let theirs_region = rebuild(&base_lines, &their_hunks[yours..last_yours], start, end);
        out.extend(base_lines[position..start].iter().copied());
        if ours_region == theirs_region {
            out.extend(ours_region.iter().map(String::as_str));
        } else {
            conflicted = true;
            out.push_str(&format!("<<<<<<< {ours_label}\n"));
            push_block(&mut out, &ours_region);
            out.push_str("=======\n");
            push_block(&mut out, &theirs_region);
            out.push_str(&format!(">>>>>>> {theirs_label}\n"));
        }
        position = end;
        mine = last_mine;
        yours = last_yours;
    }
    out.extend(base_lines[position..].iter().copied());
    (out.into_bytes(), conflicted)
}

/// Append one side of a conflict, making sure a final line without a newline gets one so that the
/// marker that follows starts its own line.
fn push_block(out: &mut String, lines: &[String]) {
    for line in lines {
        out.push_str(line);
    }
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
}

#[cfg(test)]
mod tests {
    use super::merge_content;

    fn merged(base: &str, ours: &str, theirs: &str) -> (String, bool) {
        let (bytes, conflicted) = merge_content(
            base.as_bytes(),
            ours.as_bytes(),
            theirs.as_bytes(),
            "HEAD",
            "side",
        );
        (String::from_utf8(bytes).unwrap(), conflicted)
    }

    #[test]
    fn changes_to_different_regions_both_survive() {
        let (text, conflicted) = merged(
            "one\ntwo\nthree\nfour\nfive\n",
            "ONE\ntwo\nthree\nfour\nfive\n",
            "one\ntwo\nthree\nfour\nFIVE\n",
        );
        assert!(!conflicted);
        assert_eq!(text, "ONE\ntwo\nthree\nfour\nFIVE\n");
    }

    #[test]
    fn the_same_change_on_both_sides_is_not_a_conflict() {
        let (text, conflicted) = merged("a\nb\n", "a\nB\n", "a\nB\n");
        assert!(!conflicted);
        assert_eq!(text, "a\nB\n");
    }

    #[test]
    fn competing_changes_to_one_region_are_written_as_a_conflict() {
        let (text, conflicted) = merged("a\nb\nc\n", "a\nOURS\nc\n", "a\nTHEIRS\nc\n");
        assert!(conflicted);
        assert_eq!(
            text,
            "a\n<<<<<<< HEAD\nOURS\n=======\nTHEIRS\n>>>>>>> side\nc\n"
        );
    }

    #[test]
    fn a_missing_final_newline_still_separates_the_markers() {
        let (text, conflicted) = merged("a\n", "OURS", "THEIRS");
        assert!(conflicted);
        assert_eq!(text, "<<<<<<< HEAD\nOURS\n=======\nTHEIRS\n>>>>>>> side\n");
    }
}
