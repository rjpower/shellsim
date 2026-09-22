//! File and directory comparison commands.
//!
//! Diff work is performed entirely against the virtual filesystem and meters the quadratic LCS
//! matrix before allocation.

use std::collections::HashMap;

use crate::commands::util::{ewln, split_flags, wln};
use crate::commands::{CommandContext, CommandSpec, Io, Trust};

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg;
    reg(m, &["diff"], Trust::Partial, cmd_diff);
    reg(m, &["cmp"], Trust::Real, cmd_cmp);
}

fn cmd_diff(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut unified = false;
    let mut recursive = false;
    let mut brief = false;
    let mut absent_empty = false;
    let mut whitespace = DiffWhitespace::Exact;
    let mut operands = Vec::new();
    for argument in args {
        match argument.as_str() {
            "-u" | "--unified" => unified = true,
            "-r" | "--recursive" => recursive = true,
            "-q" | "--brief" => brief = true,
            "-N" | "--new-file" => absent_empty = true,
            "-w" | "--ignore-all-space" => whitespace = DiffWhitespace::All,
            "-b" | "--ignore-space-change" => {
                if whitespace != DiffWhitespace::All {
                    whitespace = DiffWhitespace::Change;
                }
            }
            "--" => {}
            value
                if value.starts_with('-')
                    && value.len() > 2
                    && value[1..]
                        .chars()
                        .all(|flag| matches!(flag, 'u' | 'r' | 'q' | 'N' | 'w' | 'b')) =>
            {
                for flag in value[1..].chars() {
                    match flag {
                        'u' => unified = true,
                        'r' => recursive = true,
                        'q' => brief = true,
                        'N' => absent_empty = true,
                        'w' => whitespace = DiffWhitespace::All,
                        'b' if whitespace != DiffWhitespace::All => {
                            whitespace = DiffWhitespace::Change;
                        }
                        'b' => {}
                        _ => unreachable!("guarded option flag"),
                    }
                }
            }
            value if value.starts_with('-') => {
                ewln(io.err, &format!("diff: unsupported option '{value}'"));
                return 2;
            }
            _ => operands.push(argument),
        }
    }
    if operands.len() != 2 {
        ewln(io.err, "diff: missing operand");
        return 2;
    }
    let options = DiffOptions {
        unified,
        brief,
        absent_empty,
        whitespace,
    };
    let left_meta = interp.fs_metadata(&interp.cwd, operands[0], true);
    let right_meta = interp.fs_metadata(&interp.cwd, operands[1], true);
    let directories = matches!(
        left_meta.as_ref().map(|n| &n.kind),
        Ok(crate::vfs::NodeKind::Dir)
    ) || matches!(
        right_meta.as_ref().map(|n| &n.kind),
        Ok(crate::vfs::NodeKind::Dir)
    );
    if directories {
        if !recursive {
            ewln(io.err, "diff: directory comparison requires -r");
            return 2;
        }
        return diff_directories(interp, operands[0], operands[1], &options, io);
    }
    diff_files(interp, operands[0], operands[1], &options, io)
}

struct DiffOptions {
    unified: bool,
    brief: bool,
    absent_empty: bool,
    whitespace: DiffWhitespace,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DiffWhitespace {
    Exact,
    Change,
    All,
}

#[derive(Clone, Copy)]
enum DiffLine<'a> {
    Same(&'a str),
    Remove(&'a str),
    Add(&'a str),
}

fn diff_files(
    interp: &mut CommandContext<'_>,
    left: &str,
    right: &str,
    options: &DiffOptions,
    io: &mut Io,
) -> i32 {
    let read = |interp: &CommandContext<'_>, path: &str| interp.fs_read(&interp.cwd, path);
    let left_data = match read(interp, left) {
        Ok(v) => v,
        Err(_) if options.absent_empty => Vec::new(),
        Err(e) => {
            ewln(io.err, &format!("diff: {left}: {e}"));
            return 2;
        }
    };
    let right_data = match read(interp, right) {
        Ok(v) => v,
        Err(_) if options.absent_empty => Vec::new(),
        Err(e) => {
            ewln(io.err, &format!("diff: {right}: {e}"));
            return 2;
        }
    };
    if left_data == right_data {
        return 0;
    }
    let left_text = String::from_utf8_lossy(&left_data);
    let right_text = String::from_utf8_lossy(&right_data);
    let left_lines = left_text.lines().collect::<Vec<_>>();
    let right_lines = right_text.lines().collect::<Vec<_>>();
    let left_keys = left_lines
        .iter()
        .map(|line| diff_line_key(line, options.whitespace))
        .collect::<Vec<_>>();
    let right_keys = right_lines
        .iter()
        .map(|line| diff_line_key(line, options.whitespace))
        .collect::<Vec<_>>();
    if left_keys == right_keys {
        return 0;
    }
    if options.brief {
        wln(io.out, &format!("Files {left} and {right} differ"));
        return 1;
    }
    let cells = left_lines
        .len()
        .saturating_add(1)
        .saturating_mul(right_lines.len().saturating_add(1));
    if cells > 1_000_000 {
        ewln(io.err, "diff: inputs are too large for line comparison");
        return 2;
    }
    if !interp.charge_cpu(u64::try_from(cells).unwrap_or(u64::MAX)) {
        return 137;
    }
    let matrix_memory = u64::try_from(cells)
        .unwrap_or(u64::MAX)
        .saturating_mul(std::mem::size_of::<usize>() as u64);
    if !interp.reserve_memory(matrix_memory) {
        return 137;
    }
    let edits = lcs_edits(&left_lines, &right_lines, &left_keys, &right_keys);
    interp.resources.release_memory(matrix_memory);
    if options.unified {
        wln(io.out, &format!("--- {left}"));
        wln(io.out, &format!("+++ {right}"));
        wln(
            io.out,
            &format!("@@ -1,{} +1,{} @@", left_lines.len(), right_lines.len()),
        );
        for edit in edits {
            match edit {
                DiffLine::Same(line) => wln(io.out, &format!(" {line}")),
                DiffLine::Remove(line) => wln(io.out, &format!("-{line}")),
                DiffLine::Add(line) => wln(io.out, &format!("+{line}")),
            }
        }
    } else {
        for edit in edits {
            match edit {
                DiffLine::Same(_) => {}
                DiffLine::Remove(line) => wln(io.out, &format!("< {line}")),
                DiffLine::Add(line) => wln(io.out, &format!("> {line}")),
            }
        }
    }
    1
}

fn diff_line_key(line: &str, whitespace: DiffWhitespace) -> String {
    match whitespace {
        DiffWhitespace::Exact => line.to_string(),
        DiffWhitespace::All => line
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect(),
        DiffWhitespace::Change => {
            let mut key = String::with_capacity(line.len());
            let mut in_whitespace = false;
            for character in line.chars() {
                if character.is_whitespace() {
                    in_whitespace = true;
                } else {
                    if in_whitespace && !key.is_empty() {
                        key.push(' ');
                    }
                    key.push(character);
                    in_whitespace = false;
                }
            }
            key
        }
    }
}

fn lcs_edits<'a>(
    left: &[&'a str],
    right: &[&'a str],
    left_keys: &[String],
    right_keys: &[String],
) -> Vec<DiffLine<'a>> {
    let width = right.len() + 1;
    let mut lengths = vec![0usize; (left.len() + 1) * width];
    for i in (0..left.len()).rev() {
        for j in (0..right.len()).rev() {
            lengths[i * width + j] = if left_keys[i] == right_keys[j] {
                lengths[(i + 1) * width + j + 1] + 1
            } else {
                lengths[(i + 1) * width + j].max(lengths[i * width + j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut edits = Vec::new();
    while i < left.len() || j < right.len() {
        if i < left.len() && j < right.len() && left_keys[i] == right_keys[j] {
            edits.push(DiffLine::Same(left[i]));
            i += 1;
            j += 1;
        } else if j < right.len()
            && (i == left.len() || lengths[i * width + j + 1] > lengths[(i + 1) * width + j])
        {
            edits.push(DiffLine::Add(right[j]));
            j += 1;
        } else {
            edits.push(DiffLine::Remove(left[i]));
            i += 1;
        }
    }
    edits
}

fn diff_directories(
    interp: &mut CommandContext<'_>,
    left: &str,
    right: &str,
    options: &DiffOptions,
    io: &mut Io,
) -> i32 {
    let left_abs = crate::vfs::resolve_against(&interp.cwd, left);
    let right_abs = crate::vfs::resolve_against(&interp.cwd, right);
    let relative_entries =
        |interp: &CommandContext<'_>, root: &str| -> std::collections::BTreeMap<String, bool> {
            interp
                .fs_walk("/", root)
                .unwrap_or_default()
                .into_iter()
                .filter_map(|path| {
                    let relative = path.strip_prefix(root)?.trim_start_matches('/');
                    if relative.is_empty() {
                        return None;
                    }
                    let directory = matches!(
                        interp.fs_metadata("/", &path, true).map(|node| node.kind),
                        Ok(crate::vfs::NodeKind::Dir)
                    );
                    Some((relative.to_string(), directory))
                })
                .collect()
        };
    let left_entries = relative_entries(interp, &left_abs);
    let right_entries = relative_entries(interp, &right_abs);
    let names = left_entries
        .keys()
        .chain(right_entries.keys())
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let mut status = 0;
    let mut omitted_directories = Vec::<String>::new();
    for name in names {
        if omitted_directories
            .iter()
            .any(|directory| name.starts_with(&format!("{directory}/")))
        {
            continue;
        }
        let left_kind = left_entries.get(&name);
        let right_kind = right_entries.get(&name);
        if left_kind.is_none() || right_kind.is_none() {
            if options.absent_empty && (left_kind == Some(&false) || right_kind == Some(&false)) {
                // Missing regular files are compared with an empty file below.
            } else {
                let (side, directory) = if let Some(directory) = left_kind {
                    (left, *directory)
                } else {
                    (right, *right_kind.expect("entry exists on one side"))
                };
                print_only_in(io.out, side, &name);
                if directory {
                    omitted_directories.push(name);
                }
                status = 1;
                continue;
            }
        }
        if let (Some(left_directory), Some(right_directory)) = (left_kind, right_kind) {
            if left_directory != right_directory {
                wln(
                    io.out,
                    &format!(
                        "File {}/{} is a {} while file {}/{} is a {}",
                        left.trim_end_matches('/'),
                        name,
                        if *left_directory {
                            "directory"
                        } else {
                            "regular file"
                        },
                        right.trim_end_matches('/'),
                        name,
                        if *right_directory {
                            "directory"
                        } else {
                            "regular file"
                        }
                    ),
                );
                status = 1;
                continue;
            }
            if *left_directory {
                continue;
            }
        }
        if left_kind == Some(&true) || right_kind == Some(&true) {
            status = 1;
            continue;
        }
        let left_path = format!("{}/{name}", left.trim_end_matches('/'));
        let right_path = format!("{}/{name}", right.trim_end_matches('/'));
        let file_status = diff_files(interp, &left_path, &right_path, options, io);
        if file_status == 2 || file_status == 137 {
            return file_status;
        }
        status = status.max(file_status);
    }
    status
}

fn print_only_in(output: &mut Vec<u8>, root: &str, relative: &str) {
    let (parent, basename) = relative.rsplit_once('/').map_or(
        (root.trim_end_matches('/').to_string(), relative),
        |(parent, basename)| (format!("{}/{parent}", root.trim_end_matches('/')), basename),
    );
    wln(output, &format!("Only in {parent}: {basename}"));
}

fn cmd_cmp(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    if ops.len() < 2 {
        return 2;
    }
    let a = match interp.fs_read(&interp.cwd, ops[0]) {
        Ok(data) => data,
        Err(error) => {
            ewln(io.err, &format!("cmp: {}: {error}", ops[0]));
            return 2;
        }
    };
    let b = match interp.fs_read(&interp.cwd, ops[1]) {
        Ok(data) => data,
        Err(error) => {
            ewln(io.err, &format!("cmp: {}: {error}", ops[1]));
            return 2;
        }
    };
    if a == b {
        0
    } else {
        ewln(io.err, &format!("{} {} differ", ops[0], ops[1]));
        1
    }
}
