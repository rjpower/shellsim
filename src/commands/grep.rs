//! Bounded grep-family pattern matching against virtual files and buffered input.
//!
//! Recursive searches stay within the virtual filesystem. Pattern compilation, input scanning,
//! and retained pattern-file data are charged to the interpreter resource budget.

use std::collections::HashMap;

use crate::commands::options::{parse_options_or_report, OptionSpec};
use crate::commands::regex_compat::basic_regex_to_rust;
use crate::commands::util::{ewln, glob_eq, wln};
use crate::commands::{CommandContext, CommandSpec, Io, Trust};
use crate::interp::Interp;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg;
    reg(m, &["grep"], Trust::Partial, cmd_grep);
    reg(m, &["egrep"], Trust::Partial, cmd_egrep);
    reg(m, &["fgrep"], Trust::Partial, cmd_fgrep);
}

fn cmd_grep(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    grep_impl(interp, "grep", args, io)
}

fn cmd_egrep(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    grep_impl(interp, "egrep", args, io)
}

fn cmd_fgrep(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    grep_impl(interp, "fgrep", args, io)
}

/// grep with the command name available (egrep/fgrep change default regex flavor).
fn grep_impl(interp: &mut Interp, cmd: &str, args: &[String], io: &mut Io) -> i32 {
    let memory_mark = interp.resources.memory_mark();
    let status = grep_impl_inner(interp, cmd, args, io);
    interp.resources.restore_memory(memory_mark);
    status
}

fn grep_impl_inner(interp: &mut Interp, cmd: &str, args: &[String], io: &mut Io) -> i32 {
    #[derive(Clone, Copy, PartialEq)]
    enum Key {
        IgnoreCase,
        Invert,
        Count,
        LineNumber,
        FilesWith,
        FilesWithout,
        OnlyMatch,
        Recursive,
        Extended,
        Fixed,
        Word,
        Line,
        Quiet,
        NoFilename,
        WithFilename,
        SuppressErrors,
        Text,
        Regexp,
        PatternFile,
        MaxCount,
        Include,
        Exclude,
        ExcludeDir,
        BeforeContext,
        AfterContext,
        Context,
        Color,
        Help,
    }
    const OPTIONS: &[OptionSpec<Key>] = &[
        OptionSpec::flag(Key::IgnoreCase, Some('i'), Some("ignore-case")),
        OptionSpec::flag(Key::Invert, Some('v'), Some("invert-match")),
        OptionSpec::flag(Key::Count, Some('c'), Some("count")),
        OptionSpec::flag(Key::LineNumber, Some('n'), Some("line-number")),
        OptionSpec::flag(Key::FilesWith, Some('l'), Some("files-with-matches")),
        OptionSpec::flag(Key::FilesWithout, Some('L'), Some("files-without-match")),
        OptionSpec::flag(Key::OnlyMatch, Some('o'), Some("only-matching")),
        OptionSpec::flag(Key::Recursive, Some('r'), Some("recursive")),
        OptionSpec::flag(Key::Recursive, Some('R'), Some("dereference-recursive")),
        OptionSpec::flag(Key::Extended, Some('E'), Some("extended-regexp")),
        OptionSpec::flag(Key::Fixed, Some('F'), Some("fixed-strings")),
        OptionSpec::flag(Key::Word, Some('w'), Some("word-regexp")),
        OptionSpec::flag(Key::Line, Some('x'), Some("line-regexp")),
        OptionSpec::flag(Key::Quiet, Some('q'), Some("quiet")),
        OptionSpec::flag(Key::Quiet, None, Some("silent")),
        OptionSpec::flag(Key::NoFilename, Some('h'), Some("no-filename")),
        OptionSpec::flag(Key::WithFilename, Some('H'), Some("with-filename")),
        OptionSpec::flag(Key::SuppressErrors, Some('s'), Some("no-messages")),
        OptionSpec::flag(Key::Text, Some('a'), Some("text")),
        OptionSpec::required(Key::Regexp, Some('e'), Some("regexp")),
        OptionSpec::required(Key::PatternFile, Some('f'), Some("file")),
        OptionSpec::required(Key::MaxCount, Some('m'), Some("max-count")),
        OptionSpec::required(Key::Include, None, Some("include")),
        OptionSpec::required(Key::Exclude, None, Some("exclude")),
        OptionSpec::required(Key::ExcludeDir, None, Some("exclude-dir")),
        OptionSpec::required(Key::BeforeContext, Some('B'), Some("before-context")),
        OptionSpec::required(Key::AfterContext, Some('A'), Some("after-context")),
        OptionSpec::required(Key::Context, Some('C'), Some("context")),
        OptionSpec::optional_attached(Key::Color, None, Some("color")),
        OptionSpec::flag(Key::Help, None, Some("help")),
    ];
    let parsed = match parse_options_or_report(
        "grep",
        args,
        OPTIONS,
        (
            Key::Help,
            "usage: grep [OPTIONS] PATTERN [FILE...]\nsupported: -E -F -i -v -c -n -l -L -o -r -w -x -q -h -H -s -e -f -m -A -B -C\n",
        ),
        io.out,
        io.err,
    ) {
        Ok(parsed) => parsed,
        Err(status) => return status,
    };
    let mut ignore_case = false;
    let mut invert = false;
    let mut count = false;
    let mut line_num = false;
    let mut files_with = false;
    let mut files_without = false;
    let mut only_match = false;
    let mut recursive = false;
    let mut extended = cmd == "egrep";
    let mut fixed = cmd == "fgrep";
    let mut word = false;
    let mut line_regexp = false;
    let mut quiet = false;
    let mut suppress_errors = false;
    let mut text_mode = false;
    let mut filename_mode = None;
    let mut max_count = None;
    let mut before_context = 0;
    let mut after_context = 0;
    let mut includes = Vec::new();
    let mut excludes = Vec::new();
    let mut exclude_dirs = Vec::new();
    let mut pattern_files = Vec::new();
    let mut explicit_pattern = false;
    let mut patterns = Vec::new();
    for option in parsed.options {
        match option.key {
            Key::IgnoreCase => ignore_case = true,
            Key::Invert => invert = true,
            Key::Count => count = true,
            Key::LineNumber => line_num = true,
            Key::FilesWith => files_with = true,
            Key::FilesWithout => files_without = true,
            Key::OnlyMatch => only_match = true,
            Key::Recursive => recursive = true,
            Key::Extended => {
                extended = true;
                fixed = false;
            }
            Key::Fixed => fixed = true,
            Key::Word => word = true,
            Key::Line => line_regexp = true,
            Key::Quiet => quiet = true,
            Key::NoFilename => filename_mode = Some(false),
            Key::WithFilename => filename_mode = Some(true),
            Key::SuppressErrors => suppress_errors = true,
            Key::Text => text_mode = true,
            Key::Regexp => {
                explicit_pattern = true;
                patterns.push(option.value.expect("required option value"));
            }
            Key::PatternFile => {
                explicit_pattern = true;
                pattern_files.push(option.value.expect("required option value"));
            }
            Key::MaxCount => {
                let value = option.value.expect("required option value");
                max_count = match value.parse::<usize>() {
                    Ok(value) => Some(value),
                    Err(_) => {
                        ewln(io.err, &format!("grep: invalid max count: {value}"));
                        return 2;
                    }
                };
            }
            Key::Include => includes.push(option.value.expect("required option value")),
            Key::Exclude => excludes.push(option.value.expect("required option value")),
            Key::ExcludeDir => exclude_dirs.push(option.value.expect("required option value")),
            Key::BeforeContext | Key::AfterContext | Key::Context => {
                let value = option.value.expect("required option value");
                let count = match value.parse::<usize>() {
                    Ok(value) => value,
                    Err(_) => {
                        ewln(io.err, &format!("grep: invalid context length: {value}"));
                        return 2;
                    }
                };
                match option.key {
                    Key::BeforeContext => before_context = count,
                    Key::AfterContext => after_context = count,
                    Key::Context => {
                        before_context = count;
                        after_context = count;
                    }
                    _ => unreachable!(),
                }
            }
            Key::Color => match option.value.as_deref().unwrap_or("auto") {
                "never" => {}
                mode => {
                    ewln(io.err, &format!("grep: unsupported color mode '{mode}'"));
                    return 2;
                }
            },
            Key::Help => unreachable!("help is handled by the shared option parser"),
        }
    }
    for file in pattern_files {
        let bytes = match interp.vfs.read(&interp.cwd, &file) {
            Ok(bytes) => bytes,
            Err(error) => {
                ewln(io.err, &format!("grep: {file}: {error}"));
                return 2;
            }
        };
        let bytes_len = bytes.len() as u64;
        if !interp.resources.reserve_memory(bytes_len) || !interp.resources.charge_cpu(bytes_len) {
            return interp
                .resources
                .stop_reason()
                .map_or(137, |reason| reason.exit_status());
        }
        let content = match String::from_utf8(bytes) {
            Ok(content) => content,
            Err(_) => {
                ewln(
                    io.err,
                    &format!("grep: {file}: pattern file is not valid UTF-8"),
                );
                return 2;
            }
        };
        patterns.extend(content.lines().map(str::to_string));
    }
    if (before_context != 0 || after_context != 0)
        && (count || files_with || files_without || only_match || quiet)
    {
        ewln(
            io.err,
            "grep: context cannot be combined with count, file-list, only-match, or quiet modes",
        );
        return 2;
    }
    let mut operands = parsed.operands;
    let files = if !explicit_pattern && !operands.is_empty() {
        patterns.push(operands.remove(0));
        operands
    } else {
        operands
    };
    if patterns.is_empty() && !explicit_pattern {
        ewln(io.err, "grep: no pattern");
        return 2;
    }
    let mut pat_re = if patterns.is_empty() {
        r"[^\s\S]".to_string()
    } else if fixed {
        patterns
            .iter()
            .map(|pattern| regex::escape(pattern))
            .collect::<Vec<_>>()
            .join("|")
    } else {
        let patterns: Result<Vec<String>, String> = if extended {
            Ok(patterns.clone())
        } else {
            patterns
                .iter()
                .map(|pattern| basic_regex_to_rust(pattern))
                .collect::<Result<Vec<_>, _>>()
        };
        let patterns = match patterns {
            Ok(patterns) => patterns,
            Err(error) => {
                ewln(
                    io.err,
                    &format!("grep: unsupported regular expression: {error}"),
                );
                return 2;
            }
        };
        patterns
            .iter()
            .map(|pattern| format!("(?:{pattern})"))
            .collect::<Vec<_>>()
            .join("|")
    };
    if word {
        pat_re = format!(r"\b(?:{pat_re})\b");
    }
    if line_regexp {
        pat_re = format!(r"^(?:{pat_re})$");
    }
    let re = match regex::RegexBuilder::new(&pat_re)
        .case_insensitive(ignore_case)
        .build()
    {
        Ok(r) => r,
        Err(error) => {
            ewln(
                io.err,
                &format!("grep: invalid regular expression: {error}"),
            );
            return 2;
        }
    };

    // gather (label, data)
    let mut inputs: Vec<(String, Vec<u8>)> = Vec::new();
    let mut had_error = false;
    if files.is_empty() {
        inputs.push((String::new(), std::mem::take(&mut io.stdin)));
    } else if recursive {
        for f in &files {
            let abs = crate::vfs::resolve_against(&interp.cwd, f);
            let paths = match interp.fs_walk("/", &abs) {
                Ok(paths) => paths,
                Err(error) => {
                    if !suppress_errors {
                        ewln(io.err, &format!("grep: {error}"));
                    }
                    had_error = true;
                    continue;
                }
            };
            for p in paths {
                if matches!(
                    interp.fs_metadata("/", &p, false).map(|node| node.kind),
                    Ok(crate::vfs::NodeKind::File(_))
                ) {
                    if !grep_path_selected(&p, &includes, &excludes, &exclude_dirs) {
                        continue;
                    }
                    match interp.fs_read("/", &p) {
                        Ok(data) => inputs.push((p.clone(), data)),
                        Err(error) => {
                            if !suppress_errors {
                                ewln(io.err, &format!("grep: {error}"));
                            }
                            had_error = true;
                        }
                    }
                }
            }
        }
    } else {
        for f in &files {
            if !grep_path_selected(f, &includes, &excludes, &[]) {
                continue;
            }
            match interp.fs_read(&interp.cwd, f) {
                Ok(d) => inputs.push((f.clone(), d)),
                Err(error) => {
                    if !suppress_errors {
                        ewln(io.err, &format!("grep: {f}: {error}"));
                    }
                    had_error = true;
                }
            }
        }
    }
    let multi = filename_mode.unwrap_or(inputs.len() > 1 || recursive);
    let mut total_matches = 0;
    let mut selected_file = false;
    for (label, data) in &inputs {
        if !interp.resources.charge_cpu(data.len() as u64) {
            return 137;
        }
        if !text_mode && data.contains(&0) {
            if !suppress_errors {
                ewln(
                    io.err,
                    &format!("grep: {label}: binary input is not supported; use -a"),
                );
            }
            had_error = true;
            continue;
        }
        if parsed_text(data).is_err() && !suppress_errors {
            ewln(io.err, &format!("grep: {label}: input is not valid UTF-8"));
            had_error = true;
            continue;
        }
        let text = match parsed_text(data) {
            Ok(text) => text,
            Err(()) => {
                had_error = true;
                continue;
            }
        };
        if before_context != 0 || after_context != 0 {
            let matched = grep_context(
                &re,
                text,
                label,
                multi,
                line_num,
                invert,
                max_count,
                before_context,
                after_context,
                io.out,
            );
            total_matches += matched;
            selected_file |= matched != 0;
            continue;
        }
        let mut file_count = 0;
        let mut matched_file = false;
        for (lineno, line) in text.lines().enumerate() {
            if max_count == Some(0) {
                break;
            }
            let is_match = re.is_match(line) ^ invert;
            if is_match {
                matched_file = true;
                file_count += 1;
                total_matches += 1;
                if quiet {
                    return 0;
                }
                if count || files_with || files_without {
                    if files_with || max_count.is_some_and(|limit| file_count >= limit) {
                        break;
                    }
                    continue;
                }
                let mut prefix = String::new();
                if multi && !label.is_empty() {
                    prefix.push_str(label);
                    prefix.push(':');
                }
                if line_num {
                    prefix.push_str(&format!("{}:", lineno + 1));
                }
                if only_match && !invert {
                    for m in re.find_iter(line) {
                        wln(io.out, &format!("{prefix}{}", m.as_str()));
                    }
                } else {
                    wln(io.out, &format!("{prefix}{line}"));
                }
                if max_count.is_some_and(|limit| file_count >= limit) {
                    break;
                }
            }
        }
        if count {
            if multi && !label.is_empty() {
                wln(io.out, &format!("{label}:{file_count}"));
            } else {
                wln(io.out, &file_count.to_string());
            }
        }
        if files_with && matched_file {
            wln(io.out, label);
            selected_file = true;
        }
        if files_without && !matched_file {
            wln(io.out, label);
            selected_file = true;
        }
        if !files_with && !files_without {
            selected_file |= matched_file;
        }
    }
    if had_error {
        2
    } else if if files_without {
        selected_file
    } else {
        total_matches > 0 || selected_file
    } {
        0
    } else {
        1
    }
}

fn parsed_text(data: &[u8]) -> Result<&str, ()> {
    std::str::from_utf8(data).map_err(|_| ())
}

fn grep_path_selected(
    path: &str,
    includes: &[String],
    excludes: &[String],
    exclude_dirs: &[String],
) -> bool {
    let basename = crate::vfs::basename(path);
    if !includes.is_empty()
        && !includes
            .iter()
            .any(|pattern| glob_eq(pattern, path) || glob_eq(pattern, basename))
    {
        return false;
    }
    if excludes
        .iter()
        .any(|pattern| glob_eq(pattern, path) || glob_eq(pattern, basename))
    {
        return false;
    }
    !path.split('/').any(|component| {
        exclude_dirs
            .iter()
            .any(|pattern| glob_eq(pattern, component))
    })
}

#[allow(clippy::too_many_arguments)]
fn grep_context(
    regex: &regex::Regex,
    text: &str,
    label: &str,
    show_filename: bool,
    show_line_number: bool,
    invert: bool,
    max_count: Option<usize>,
    before: usize,
    after: usize,
    output: &mut Vec<u8>,
) -> usize {
    let lines = text.lines().collect::<Vec<_>>();
    let mut matches = Vec::with_capacity(lines.len());
    let mut matched_count = 0usize;
    for line in &lines {
        let matched = (regex.is_match(line) ^ invert)
            && max_count.is_none_or(|maximum| matched_count < maximum);
        matched_count = matched_count.saturating_add(usize::from(matched));
        matches.push(matched);
    }
    let mut last_emitted = None;
    for (matched_index, _) in matches.iter().enumerate().filter(|(_, matched)| **matched) {
        let start = matched_index.saturating_sub(before);
        let end = matched_index
            .saturating_add(after)
            .saturating_add(1)
            .min(lines.len());
        let first = last_emitted.map_or(start, |previous| start.max(previous + 1));
        if first >= end {
            continue;
        }
        if last_emitted.is_some_and(|previous| first > previous + 1) {
            wln(output, "--");
        }
        for index in first..end {
            let separator = if matches[index] { ':' } else { '-' };
            let mut prefix = String::new();
            if show_filename && !label.is_empty() {
                prefix.push_str(label);
                prefix.push(separator);
            }
            if show_line_number {
                prefix.push_str(&(index + 1).to_string());
                prefix.push(separator);
            }
            wln(output, &format!("{prefix}{}", lines[index]));
        }
        last_emitted = Some(end - 1);
    }
    matched_count
}
