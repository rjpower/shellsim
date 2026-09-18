//! Line diffing and patch rendering for the simulated Git porcelain.
//!
//! Diffs are computed on whole lines with a longest-common-subsequence table, then grouped into
//! unified hunks with the usual three lines of context. The table is quadratic, so the common
//! prefix and suffix are trimmed first and an oversized remainder falls back to a single
//! replace-everything hunk rather than doing unbounded work. Rename and copy detection, word
//! diffs, and binary deltas are out of scope; binary files are reported as differing.
//!
//! Lines keep their terminating newline, so a file that ends without one compares unequal to the
//! same text that ends with one, exactly as Git reports it.

/// Lines of unchanged context Git shows around each hunk.
pub(crate) const DEFAULT_CONTEXT: usize = 3;

/// The largest changed region the quadratic matcher will align, per side.
const MAX_ALIGNED_LINES: usize = 1_000;

/// One step of an edit script.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Keep,
    Delete,
    Insert,
}

/// One edit-script entry: an operation and the line it applies to, newline included.
#[derive(Clone, Debug)]
pub(crate) struct Edit<'a> {
    pub op: Op,
    pub text: &'a str,
}

/// Split content into lines that keep their trailing newline.
pub(crate) fn split_lines(data: &[u8]) -> Vec<&str> {
    let text = std::str::from_utf8(data).unwrap_or("");
    text.split_inclusive('\n').collect()
}

/// Strip the trailing newline for display.
fn body(line: &str) -> &str {
    line.strip_suffix('\n').unwrap_or(line)
}

pub(crate) fn is_binary(data: &[u8]) -> bool {
    data.iter().take(8_000).any(|byte| *byte == 0) || std::str::from_utf8(data).is_err()
}

/// How whitespace is treated when two lines are compared.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) enum Whitespace {
    #[default]
    Significant,
    /// `-b`: runs of whitespace compare equal and trailing whitespace is ignored.
    IgnoreChange,
    /// `-w`: whitespace is ignored entirely.
    IgnoreAll,
}

/// The form of a line used for comparison under `whitespace`.
fn compare_key(line: &str, whitespace: Whitespace) -> String {
    match whitespace {
        Whitespace::Significant => line.to_string(),
        Whitespace::IgnoreAll => line.split_whitespace().collect(),
        Whitespace::IgnoreChange => line.split_whitespace().collect::<Vec<_>>().join(" "),
    }
}

/// Build an edit script that compares under `whitespace` but reports the original lines.
pub(crate) fn edit_script_ignoring<'a>(
    old: &[&'a str],
    new: &[&'a str],
    whitespace: Whitespace,
) -> Vec<Edit<'a>> {
    if whitespace == Whitespace::Significant {
        return edit_script(old, new);
    }
    let old_keys: Vec<String> = old
        .iter()
        .map(|line| compare_key(line, whitespace))
        .collect();
    let new_keys: Vec<String> = new
        .iter()
        .map(|line| compare_key(line, whitespace))
        .collect();
    let old_refs: Vec<&str> = old_keys.iter().map(String::as_str).collect();
    let new_refs: Vec<&str> = new_keys.iter().map(String::as_str).collect();
    let mut old_index = 0;
    let mut new_index = 0;
    edit_script(&old_refs, &new_refs)
        .into_iter()
        .map(|edit| {
            let text = match edit.op {
                Op::Insert => {
                    new_index += 1;
                    new[new_index - 1]
                }
                Op::Delete => {
                    old_index += 1;
                    old[old_index - 1]
                }
                Op::Keep => {
                    old_index += 1;
                    new_index += 1;
                    old[old_index - 1]
                }
            };
            Edit { op: edit.op, text }
        })
        .collect()
}

/// Build an edit script turning `old` into `new`.
pub(crate) fn edit_script<'a>(old: &[&'a str], new: &[&'a str]) -> Vec<Edit<'a>> {
    let prefix = old
        .iter()
        .zip(new)
        .take_while(|(left, right)| left == right)
        .count();
    let suffix = old[prefix..]
        .iter()
        .rev()
        .zip(new[prefix..].iter().rev())
        .take_while(|(left, right)| left == right)
        .count();
    let old_middle = &old[prefix..old.len() - suffix];
    let new_middle = &new[prefix..new.len() - suffix];

    let mut edits: Vec<Edit<'a>> = old[..prefix]
        .iter()
        .map(|text| Edit { op: Op::Keep, text })
        .collect();
    if old_middle.len() > MAX_ALIGNED_LINES || new_middle.len() > MAX_ALIGNED_LINES {
        edits.extend(old_middle.iter().map(|text| Edit {
            op: Op::Delete,
            text,
        }));
        edits.extend(new_middle.iter().map(|text| Edit {
            op: Op::Insert,
            text,
        }));
    } else {
        edits.extend(align(old_middle, new_middle));
    }
    edits.extend(
        old[old.len() - suffix..]
            .iter()
            .map(|text| Edit { op: Op::Keep, text }),
    );
    edits
}

/// Align two line sequences with a longest-common-subsequence table.
fn align<'a>(old: &[&'a str], new: &[&'a str]) -> Vec<Edit<'a>> {
    let rows = old.len() + 1;
    let columns = new.len() + 1;
    let mut table = vec![0_u32; rows * columns];
    for row in (0..old.len()).rev() {
        for column in (0..new.len()).rev() {
            table[row * columns + column] = if old[row] == new[column] {
                table[(row + 1) * columns + column + 1] + 1
            } else {
                table[(row + 1) * columns + column].max(table[row * columns + column + 1])
            };
        }
    }
    let mut edits = Vec::new();
    let (mut row, mut column) = (0, 0);
    while row < old.len() && column < new.len() {
        if old[row] == new[column] {
            edits.push(Edit {
                op: Op::Keep,
                text: old[row],
            });
            row += 1;
            column += 1;
        } else if table[(row + 1) * columns + column] >= table[row * columns + column + 1] {
            edits.push(Edit {
                op: Op::Delete,
                text: old[row],
            });
            row += 1;
        } else {
            edits.push(Edit {
                op: Op::Insert,
                text: new[column],
            });
            column += 1;
        }
    }
    edits.extend(old[row..].iter().map(|text| Edit {
        op: Op::Delete,
        text,
    }));
    edits.extend(new[column..].iter().map(|text| Edit {
        op: Op::Insert,
        text,
    }));
    edits
}

/// Inserted and deleted line counts, as `git diff --stat` reports them.
pub(crate) fn change_counts(
    old: Option<&[u8]>,
    new: Option<&[u8]>,
    whitespace: Whitespace,
) -> (usize, usize) {
    let old_lines = split_lines(old.unwrap_or_default());
    let new_lines = split_lines(new.unwrap_or_default());
    let edits = edit_script_ignoring(&old_lines, &new_lines, whitespace);
    (
        edits.iter().filter(|edit| edit.op == Op::Insert).count(),
        edits.iter().filter(|edit| edit.op == Op::Delete).count(),
    )
}

/// The `diff --git` header Git prints before each file's hunks.
fn header(
    path: &str,
    old: Option<&[u8]>,
    new: Option<&[u8]>,
    prefixes: bool,
    modes: Modes,
) -> String {
    let (left, right) = labels(prefixes);
    let blank = "0000000";
    let hash = |data: Option<&[u8]>| {
        data.map_or(blank.to_string(), |data| {
            super::repo::short(&super::repo::blob_hash(data)).to_string()
        })
    };
    let mut text = format!("diff --git {left}{path} {right}{path}\n");
    match (old.is_some(), new.is_some()) {
        (false, true) => text.push_str(&format!("new file mode {}\n", modes.1)),
        (true, false) => text.push_str(&format!("deleted file mode {}\n", modes.0)),
        _ if modes.0 != modes.1 => {
            text.push_str(&format!("old mode {}\nnew mode {}\n", modes.0, modes.1));
        }
        _ => {}
    }
    if old == new {
        // A change of mode alone has no content to index.
        return text;
    }
    text.push_str(&format!("index {}..{}", hash(old), hash(new)));
    if old.is_some() && new.is_some() {
        text.push_str(&format!(" {}", modes.1));
    }
    text.push('\n');
    text
}

/// The `a/` and `b/` path prefixes, which `git diff --no-prefix` drops.
fn labels(prefixes: bool) -> (&'static str, &'static str) {
    if prefixes {
        ("a/", "b/")
    } else {
        ("", "")
    }
}

/// Render one file's unified diff, including the `diff --git` header.
///
/// Returns an empty string when the contents are identical.
/// The `100644` or `100755` each side of a patch reports.
pub(crate) type Modes = (&'static str, &'static str);

/// The modes of a file neither side marks executable.
pub(crate) const PLAIN: Modes = ("100644", "100644");

pub(crate) fn render(
    path: &str,
    old: Option<&[u8]>,
    new: Option<&[u8]>,
    modes: Modes,
    context: usize,
    prefixes: bool,
    whitespace: Whitespace,
) -> String {
    if old == new && modes.0 == modes.1 {
        return String::new();
    }
    let (left, right) = labels(prefixes);
    let mut out = header(path, old, new, prefixes, modes);
    if old == new {
        // Only the mode changed, so the header says everything.
        return out;
    }
    if old.is_some_and(is_binary) || new.is_some_and(is_binary) {
        out.push_str(&format!(
            "Binary files {} and {} differ\n",
            old.map_or("/dev/null".to_string(), |_| format!("{left}{path}")),
            new.map_or("/dev/null".to_string(), |_| format!("{right}{path}"))
        ));
        return out;
    }
    out.push_str(&match old {
        Some(_) => format!("--- {left}{path}\n"),
        None => "--- /dev/null\n".to_string(),
    });
    out.push_str(&match new {
        Some(_) => format!("+++ {right}{path}\n"),
        None => "+++ /dev/null\n".to_string(),
    });
    let old_lines = split_lines(old.unwrap_or_default());
    let new_lines = split_lines(new.unwrap_or_default());
    let edits = edit_script_ignoring(&old_lines, &new_lines, whitespace);
    let hunks = render_hunks(&edits, context);
    if hunks.is_empty() {
        // Under `-w` or `-b` the files may compare equal, and Git then prints nothing at all.
        return String::new();
    }
    out.push_str(&hunks);
    out
}

/// Group an edit script into unified hunks.
fn render_hunks(edits: &[Edit<'_>], context: usize) -> String {
    let changed: Vec<usize> = edits
        .iter()
        .enumerate()
        .filter(|(_, edit)| edit.op != Op::Keep)
        .map(|(index, _)| index)
        .collect();
    if changed.is_empty() {
        return String::new();
    }
    // One-based line numbers of the line each edit refers to on each side.
    let mut old_line = Vec::with_capacity(edits.len());
    let mut new_line = Vec::with_capacity(edits.len());
    let (mut old_at, mut new_at) = (1_usize, 1_usize);
    for edit in edits {
        old_line.push(old_at);
        new_line.push(new_at);
        match edit.op {
            Op::Keep => {
                old_at += 1;
                new_at += 1;
            }
            Op::Delete => old_at += 1,
            Op::Insert => new_at += 1,
        }
    }

    let mut out = String::new();
    let mut index = 0;
    while index < changed.len() {
        let start = changed[index].saturating_sub(context);
        let mut end = changed[index];
        // Extend the hunk while the next change is close enough to share context.
        while index + 1 < changed.len() && changed[index + 1] <= end + 2 * context + 1 {
            index += 1;
            end = changed[index];
        }
        index += 1;
        let end = (end + context).min(edits.len() - 1);
        let slice = &edits[start..=end];
        let old_count = slice.iter().filter(|edit| edit.op != Op::Insert).count();
        let new_count = slice.iter().filter(|edit| edit.op != Op::Delete).count();
        let old_start = if old_count == 0 {
            old_line[start].saturating_sub(1)
        } else {
            old_line[start]
        };
        let new_start = if new_count == 0 {
            new_line[start].saturating_sub(1)
        } else {
            new_line[start]
        };
        let function = enclosing_definition(edits, start);
        out.push_str(&format!(
            "@@ -{} +{} @@{function}\n",
            range(old_start, old_count),
            range(new_start, new_count)
        ));
        for edit in slice {
            out.push(match edit.op {
                Op::Keep => ' ',
                Op::Delete => '-',
                Op::Insert => '+',
            });
            out.push_str(body(edit.text));
            out.push('\n');
            if !edit.text.ends_with('\n') {
                out.push_str("\\ No newline at end of file\n");
            }
        }
    }
    out
}

/// The nearest preceding definition-like line, which Git shows after the hunk header.
fn enclosing_definition(edits: &[Edit<'_>], start: usize) -> String {
    edits[..start]
        .iter()
        .rev()
        .filter(|edit| edit.op != Op::Insert)
        .map(|edit| body(edit.text))
        .find(|line| {
            line.starts_with(|character: char| {
                character.is_alphabetic() || character == '_' || character == '$'
            })
        })
        .map(|line| format!(" {}", line.trim_end()))
        .unwrap_or_default()
}

fn range(start: usize, count: usize) -> String {
    if count == 1 {
        start.to_string()
    } else {
        format!("{start},{count}")
    }
}

/// Report trailing whitespace introduced by the new side, as `git diff --check` does.
pub(crate) fn whitespace_errors(path: &str, old: Option<&[u8]>, new: Option<&[u8]>) -> String {
    let old_lines = split_lines(old.unwrap_or_default());
    let new_lines = split_lines(new.unwrap_or_default());
    let mut out = String::new();
    let mut line_number = 0;
    for edit in edit_script(&old_lines, &new_lines) {
        if edit.op == Op::Delete {
            continue;
        }
        line_number += 1;
        if edit.op == Op::Insert && body(edit.text).ends_with([' ', '\t']) {
            out.push_str(&format!(
                "{path}:{line_number}: trailing whitespace.\n+{}\n",
                body(edit.text)
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{change_counts, render, Whitespace, DEFAULT_CONTEXT, PLAIN};

    #[test]
    fn renders_a_hunk_with_context_rather_than_the_whole_file() {
        let old = b"one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten\n";
        let new = b"one\ntwo\nthree\nfour\nFIVE\nsix\nseven\neight\nnine\nten\n";
        let patch = render(
            "f.txt",
            Some(old),
            Some(new),
            PLAIN,
            DEFAULT_CONTEXT,
            true,
            Whitespace::Significant,
        );
        assert!(patch.contains("@@ -2,7 +2,7 @@ one\n"), "{patch}");
        assert!(patch.contains("-five\n+FIVE\n"), "{patch}");
        assert!(!patch.contains("-one"), "{patch}");
    }

    #[test]
    fn marks_new_and_deleted_files() {
        let created = render(
            "f.txt",
            None,
            Some(b"hello\n"),
            PLAIN,
            DEFAULT_CONTEXT,
            true,
            Whitespace::Significant,
        );
        assert!(created.contains("new file mode 100644\n"), "{created}");
        assert!(created.contains("@@ -0,0 +1 @@\n+hello\n"), "{created}");
        let removed = render(
            "f.txt",
            Some(b"hello\n"),
            None,
            PLAIN,
            DEFAULT_CONTEXT,
            true,
            Whitespace::Significant,
        );
        assert!(removed.contains("deleted file mode 100644\n"), "{removed}");
        assert!(removed.contains("@@ -1 +0,0 @@\n-hello\n"), "{removed}");
    }

    #[test]
    fn reports_a_missing_final_newline() {
        let patch = render(
            "f.txt",
            Some(b"a\n"),
            Some(b"a\nb"),
            PLAIN,
            DEFAULT_CONTEXT,
            true,
            Whitespace::Significant,
        );
        assert!(
            patch.contains("+b\n\\ No newline at end of file\n"),
            "{patch}"
        );
    }

    #[test]
    fn treats_removing_the_final_newline_as_a_change() {
        let patch = render(
            "f.txt",
            Some(b"a\nb\n"),
            Some(b"a\nb"),
            PLAIN,
            DEFAULT_CONTEXT,
            true,
            Whitespace::Significant,
        );
        assert!(patch.contains("@@ -1,2 +1,2 @@\n"), "{patch}");
        assert!(
            patch.contains("-b\n+b\n\\ No newline at end of file\n"),
            "{patch}"
        );
        assert_eq!(
            change_counts(Some(b"a\nb\n"), Some(b"a\nb"), Whitespace::Significant),
            (1, 1)
        );
    }

    #[test]
    fn counts_changed_lines_by_alignment() {
        assert_eq!(
            change_counts(
                Some(b"a\nb\nc\n"),
                Some(b"a\nx\nc\n"),
                Whitespace::Significant
            ),
            (1, 1)
        );
        assert_eq!(
            change_counts(None, Some(b"a\nb\n"), Whitespace::Significant),
            (2, 0)
        );
        assert_eq!(
            change_counts(Some(b"a\nb\n"), None, Whitespace::Significant),
            (0, 2)
        );
    }

    #[test]
    fn uses_git_blob_hashes_in_index_lines() {
        let patch = render(
            "f.txt",
            None,
            Some(b"a\nb\n"),
            PLAIN,
            DEFAULT_CONTEXT,
            true,
            Whitespace::Significant,
        );
        assert!(patch.contains("index 0000000..422c2b7\n"), "{patch}");
    }

    #[test]
    fn treats_content_with_nul_bytes_as_binary() {
        let patch = render(
            "f.bin",
            Some(b"\x00\x01"),
            Some(b"\x00\x02"),
            PLAIN,
            DEFAULT_CONTEXT,
            true,
            Whitespace::Significant,
        );
        assert!(
            patch.contains("Binary files a/f.bin and b/f.bin differ\n"),
            "{patch}"
        );
    }

    #[test]
    fn ignoring_whitespace_hides_reindentation_but_not_real_edits() {
        let old = b"fn main() {\nlet x = 1;\n}\n";
        let new = b"fn main() {\n    let x = 1;\n}\n";
        assert_eq!(
            render(
                "f.rs",
                Some(old),
                Some(new),
                PLAIN,
                DEFAULT_CONTEXT,
                true,
                Whitespace::IgnoreAll
            ),
            ""
        );
        assert_eq!(
            change_counts(Some(old), Some(new), Whitespace::IgnoreAll),
            (0, 0)
        );
        // A real change still shows, carrying the original text.
        let edited = b"fn main() {\n    let x = 2;\n}\n";
        let patch = render(
            "f.rs",
            Some(old),
            Some(edited),
            PLAIN,
            DEFAULT_CONTEXT,
            true,
            Whitespace::IgnoreAll,
        );
        assert!(patch.contains("-let x = 1;\n"), "{patch}");
        assert!(patch.contains("+    let x = 2;\n"), "{patch}");
    }
}
