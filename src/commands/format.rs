//! Line wrapping, tab expansion, and column-formatting commands.
//!
//! These transforms read process descriptors in bounded quanta and keep their input across
//! scheduler turns when a pipe is temporarily empty.

use std::collections::HashMap;

use crate::commands::util::{ewln, read_inputs_system, uses_standard_input, w};
use crate::commands::{CommandSpec, Io, Trust};
use crate::exec::ShellPoll;
use crate::program::ProcessContext;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg_system_poll;
    reg_system_poll(m, "/usr/bin/fold", Trust::Real, cmd_fold);
    reg_system_poll(m, "/usr/bin/fmt", Trust::Partial, cmd_fmt);
    reg_system_poll(m, "/usr/bin/expand", Trust::Real, cmd_expand);
    reg_system_poll(m, "/usr/bin/unexpand", Trust::Real, cmd_unexpand);
    reg_system_poll(m, "/usr/bin/column", Trust::Partial, cmd_column);
}

fn parse_width<'a>(
    command: &str,
    args: &'a [String],
    default: usize,
) -> Result<(usize, Vec<&'a String>), String> {
    let mut width = default;
    let mut files = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "-w" || argument == "--width" {
            index += 1;
            let value = args
                .get(index)
                .ok_or_else(|| format!("{command}: option requires an argument"))?;
            width = value
                .parse()
                .map_err(|_| format!("{command}: invalid width: {value}"))?;
        } else if let Some(value) = argument.strip_prefix("--width=") {
            width = value
                .parse()
                .map_err(|_| format!("{command}: invalid width: {value}"))?;
        } else if argument.starts_with('-') && argument != "-" {
            return Err(format!("{command}: unsupported option '{argument}'"));
        } else {
            files.push(argument);
        }
        index += 1;
    }
    if width == 0 {
        return Err(format!("{command}: width must be positive"));
    }
    Ok((width, files))
}

fn cmd_fold(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let (width, files) = match parse_width("fold", args, 80) {
        Ok(value) => value,
        Err(error) => {
            ewln(io.err, &error);
            return ShellPoll::Ready(1);
        }
    };
    if uses_standard_input(&files) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("fold: {error}"));
        return ShellPoll::Ready(1);
    }
    for line in String::from_utf8_lossy(&data).split_inclusive('\n') {
        let (line, terminated) = line
            .strip_suffix('\n')
            .map_or((line, false), |value| (value, true));
        let characters = line.chars().collect::<Vec<_>>();
        if characters.is_empty() && terminated {
            io.out.push(b'\n');
            continue;
        }
        for chunk in characters.chunks(width) {
            w(io.out, &chunk.iter().collect::<String>());
            io.out.push(b'\n');
        }
        if !terminated && !characters.is_empty() {
            io.out.pop();
        }
    }
    ShellPoll::Ready(0)
}

fn cmd_fmt(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let (width, files) = match parse_width("fmt", args, 75) {
        Ok(value) => value,
        Err(error) => {
            ewln(io.err, &error);
            return ShellPoll::Ready(1);
        }
    };
    if uses_standard_input(&files) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("fmt: {error}"));
        return ShellPoll::Ready(1);
    }
    let text = String::from_utf8_lossy(&data);
    let paragraphs = text.split("\n\n").collect::<Vec<_>>();
    for (paragraph_index, paragraph) in paragraphs.iter().enumerate() {
        let mut column = 0usize;
        for word in paragraph.split_whitespace() {
            if column == 0 {
                w(io.out, word);
                column = word.chars().count();
            } else if column.saturating_add(1 + word.chars().count()) <= width {
                io.out.push(b' ');
                w(io.out, word);
                column += 1 + word.chars().count();
            } else {
                io.out.push(b'\n');
                w(io.out, word);
                column = word.chars().count();
            }
        }
        io.out.push(b'\n');
        if paragraph_index + 1 < paragraphs.len() {
            io.out.push(b'\n');
        }
    }
    ShellPoll::Ready(0)
}

fn parse_tab_stop<'a>(
    command: &str,
    args: &'a [String],
) -> Result<(usize, bool, Vec<&'a String>), String> {
    let mut stop = 8usize;
    let mut all = false;
    let mut files = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-a" | "--all" => all = true,
            "-t" | "--tabs" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| format!("{command}: option requires an argument"))?;
                stop = value
                    .parse()
                    .map_err(|_| format!("{command}: invalid tab size: {value}"))?;
            }
            value if value.starts_with("--tabs=") => {
                stop = value[7..]
                    .parse()
                    .map_err(|_| format!("{command}: invalid tab size"))?
            }
            value if value.starts_with('-') && value != "-" => {
                return Err(format!("{command}: unsupported option '{value}'"))
            }
            _ => files.push(&args[index]),
        }
        index += 1;
    }
    if stop == 0 {
        return Err(format!("{command}: tab size must be positive"));
    }
    Ok((stop, all, files))
}

fn cmd_expand(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let (stop, _, files) = match parse_tab_stop("expand", args) {
        Ok(v) => v,
        Err(e) => {
            ewln(io.err, &e);
            return ShellPoll::Ready(1);
        }
    };
    if uses_standard_input(&files) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("expand: {error}"));
        return ShellPoll::Ready(1);
    }
    let mut column = 0usize;
    for character in String::from_utf8_lossy(&data).chars() {
        match character {
            '\t' => {
                let count = stop - column % stop;
                io.out.extend(std::iter::repeat_n(b' ', count));
                column += count;
            }
            '\n' => {
                io.out.push(b'\n');
                column = 0;
            }
            value => {
                w(io.out, &value.to_string());
                column += 1;
            }
        }
    }
    ShellPoll::Ready(0)
}

fn cmd_unexpand(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let (stop, all, files) = match parse_tab_stop("unexpand", args) {
        Ok(v) => v,
        Err(e) => {
            ewln(io.err, &e);
            return ShellPoll::Ready(1);
        }
    };
    if uses_standard_input(&files) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("unexpand: {error}"));
        return ShellPoll::Ready(1);
    }
    for line in String::from_utf8_lossy(&data).split_inclusive('\n') {
        let mut column = 0usize;
        let mut spaces = 0usize;
        let mut leading = true;
        for character in line.chars() {
            if character == ' ' && (all || leading) {
                spaces += 1;
                column += 1;
                if column.is_multiple_of(stop) {
                    io.out.push(b'\t');
                    spaces = 0;
                }
            } else {
                io.out.extend(std::iter::repeat_n(b' ', spaces));
                spaces = 0;
                w(io.out, &character.to_string());
                if character == '\n' {
                    column = 0;
                    leading = true;
                } else {
                    column += 1;
                    leading = false;
                }
            }
        }
        io.out.extend(std::iter::repeat_n(b' ', spaces));
    }
    ShellPoll::Ready(0)
}

fn cmd_column(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let mut separator = None;
    let mut table = false;
    let mut files = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-t" | "--table" => table = true,
            "-s" | "--separator" => {
                index += 1;
                separator = args.get(index).and_then(|v| v.chars().next());
                if separator.is_none() {
                    ewln(io.err, "column: missing separator");
                    return ShellPoll::Ready(1);
                }
            }
            value if value.starts_with('-') && value != "-" => {
                ewln(io.err, &format!("column: unsupported option '{value}'"));
                return ShellPoll::Ready(1);
            }
            _ => files.push(&args[index]),
        }
        index += 1;
    }
    if uses_standard_input(&files) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("column: {error}"));
        return ShellPoll::Ready(1);
    }
    if !table {
        io.out.extend_from_slice(&data);
        return ShellPoll::Ready(0);
    }
    let rows = String::from_utf8_lossy(&data)
        .lines()
        .map(|line| {
            separator.map_or_else(
                || line.split_whitespace().map(str::to_string).collect(),
                |sep| line.split(sep).map(str::to_string).collect(),
            )
        })
        .collect::<Vec<Vec<String>>>();
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    let widths = (0..columns)
        .map(|column| {
            rows.iter()
                .filter_map(|row| row.get(column))
                .map(|cell| cell.chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            w(io.out, cell);
            if index + 1 < row.len() {
                w(
                    io.out,
                    &" ".repeat(widths[index].saturating_sub(cell.chars().count()) + 2),
                );
            }
        }
        io.out.push(b'\n');
    }
    ShellPoll::Ready(0)
}
