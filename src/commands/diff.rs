//! File and directory comparison commands.
//!
//! Diff work is performed entirely against the virtual filesystem and meters the quadratic LCS
//! matrix before allocation.

use std::collections::HashMap;

use crate::commands::util::{ewln, split_flags, wln};
use crate::commands::{CommandSpec, Io, Trust};
use crate::program::ProcessContext;
use crate::syscalls::{FileKind, System};

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    super::reg_system(m, "/usr/bin/diff", Trust::Partial, cmd_diff);
    super::reg_system(m, "/usr/bin/cmp", Trust::Real, cmd_cmp);
}

fn cmd_diff(context: &mut ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
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
    let cwd = system.cwd().to_string();
    let left_meta = system.metadata(&cwd, operands[0], true);
    let right_meta = system.metadata(&cwd, operands[1], true);
    let directories = matches!(left_meta.as_ref().map(|n| &n.kind), Ok(FileKind::Directory))
        || matches!(
            right_meta.as_ref().map(|n| &n.kind),
            Ok(FileKind::Directory)
        );
    if directories {
        if !recursive {
            ewln(io.err, "diff: directory comparison requires -r");
            return 2;
        }
        return diff_directories(system, operands[0], operands[1], &options, io);
    }
    diff_files(system, operands[0], operands[1], &options, io)
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
    system: &mut dyn System,
    left: &str,
    right: &str,
    options: &DiffOptions,
    io: &mut Io,
) -> i32 {
    let cwd = system.cwd().to_string();
    let maximum = usize::try_from(system.limits().memory).unwrap_or(usize::MAX);
    let left_data = match system.read_file_limited(&cwd, left, maximum) {
        Ok(v) => v,
        Err(_) if options.absent_empty => Vec::new(),
        Err(e) => {
            ewln(io.err, &format!("diff: {left}: {e}"));
            return 2;
        }
    };
    let right_data = match system.read_file_limited(&cwd, right, maximum) {
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
    if !system.charge_cpu(u64::try_from(cells).unwrap_or(u64::MAX)) {
        return system.stop_status();
    }
    let matrix_memory = u64::try_from(cells)
        .unwrap_or(u64::MAX)
        .saturating_mul(std::mem::size_of::<usize>() as u64);
    if !system.reserve_memory(matrix_memory) {
        return system.stop_status();
    }
    let edits = lcs_edits(&left_lines, &right_lines, &left_keys, &right_keys);
    system.release_memory(matrix_memory);
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
    system: &mut dyn System,
    left: &str,
    right: &str,
    options: &DiffOptions,
    io: &mut Io,
) -> i32 {
    let left_abs = crate::vfs::resolve_against(system.cwd(), left);
    let right_abs = crate::vfs::resolve_against(system.cwd(), right);
    let left_entries = match relative_entries(system, &left_abs) {
        Ok(entries) => entries,
        Err(error) => {
            ewln(io.err, &format!("diff: {left}: {error}"));
            return 2;
        }
    };
    let right_entries = match relative_entries(system, &right_abs) {
        Ok(entries) => entries,
        Err(error) => {
            ewln(io.err, &format!("diff: {right}: {error}"));
            return 2;
        }
    };
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
        let file_status = diff_files(system, &left_path, &right_path, options, io);
        if file_status == 2 || file_status == 137 {
            return file_status;
        }
        status = status.max(file_status);
    }
    status
}

fn relative_entries(
    system: &mut dyn System,
    root: &str,
) -> Result<std::collections::BTreeMap<String, bool>, crate::syscalls::SyscallError> {
    let mut entries = std::collections::BTreeMap::new();
    for path in system.walk("/", root)? {
        let Some(relative) = path.strip_prefix(root) else {
            continue;
        };
        let relative = relative.trim_start_matches('/');
        if relative.is_empty() {
            continue;
        }
        let directory = system.metadata("/", &path, true)?.kind == FileKind::Directory;
        entries.insert(relative.to_string(), directory);
    }
    Ok(entries)
}

fn print_only_in(output: &mut Vec<u8>, root: &str, relative: &str) {
    let (parent, basename) = relative.rsplit_once('/').map_or(
        (root.trim_end_matches('/').to_string(), relative),
        |(parent, basename)| (format!("{}/{parent}", root.trim_end_matches('/')), basename),
    );
    wln(output, &format!("Only in {parent}: {basename}"));
}

fn cmd_cmp(context: &mut ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let (_f, ops, _l) = split_flags(args);
    if ops.len() < 2 {
        return 2;
    }
    let cwd = system.cwd().to_string();
    let maximum = usize::try_from(system.limits().memory).unwrap_or(usize::MAX);
    let a = match system.read_file_limited(&cwd, ops[0], maximum) {
        Ok(data) => data,
        Err(error) => {
            ewln(io.err, &format!("cmp: {}: {error}", ops[0]));
            return 2;
        }
    };
    let b = match system.read_file_limited(&cwd, ops[1], maximum) {
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
