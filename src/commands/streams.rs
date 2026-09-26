//! `tac` and `tail`, plus `head` option parsing shared with the native `head` image.
//!
//! Commands in this module read input through the typed `System` boundary and never access
//! host streams.

use std::collections::HashMap;

use crate::commands::util::{
    ewln, lines_of, read_inputs_system, split_flags, uses_standard_input, wln,
};
use crate::commands::{CommandSpec, Io, Trust};
use crate::exec::ShellPoll;
use crate::program::ProcessContext;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg_system_poll;
    reg_system_poll(m, "/usr/bin/tac", Trust::Real, cmd_tac);
    reg_system_poll(m, "/usr/bin/tail", Trust::Real, cmd_tail);
}

#[derive(Clone)]
pub(crate) struct HeadOptions {
    pub(crate) lines: usize,
    pub(crate) bytes: Option<usize>,
    pub(crate) files: Vec<String>,
}

pub(crate) fn parse_head_options(args: &[String]) -> Result<HeadOptions, String> {
    let mut lines = 10usize;
    let mut bytes = None;
    let mut files = Vec::new();
    let mut it = args.iter().peekable();
    while let Some(arg) = it.next() {
        if arg == "-n" {
            let value = it
                .next()
                .ok_or_else(|| "option requires an argument -- 'n'".to_string())?;
            lines = value
                .trim_start_matches('-')
                .parse()
                .map_err(|_| format!("invalid number of lines: {value}"))?;
        } else if let Some(value) = arg.strip_prefix("-n") {
            lines = value
                .trim_start_matches('-')
                .parse()
                .map_err(|_| format!("invalid number of lines: {value}"))?;
        } else if arg == "-c" {
            let value = it
                .next()
                .ok_or_else(|| "option requires an argument -- 'c'".to_string())?;
            bytes = Some(
                value
                    .parse()
                    .map_err(|_| format!("invalid number of bytes: {value}"))?,
            );
        } else if let Some(value) = arg.strip_prefix("-c") {
            bytes = Some(
                value
                    .parse()
                    .map_err(|_| format!("invalid number of bytes: {value}"))?,
            );
        } else if arg.starts_with('-')
            && arg.len() > 1
            && arg[1..].chars().all(|character| character.is_ascii_digit())
        {
            lines = arg[1..].parse().unwrap_or(10);
        } else if !arg.starts_with('-') || arg == "-" {
            files.push(arg.clone());
        } else {
            return Err(format!("unimplemented option '{arg}'"));
        }
    }
    Ok(HeadOptions {
        lines,
        bytes,
        files,
    })
}

fn cmd_tac(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let (_f, ops, _l) = split_flags(context.args);
    if uses_standard_input(&ops) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &ops, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("tac: {error}"));
        return ShellPoll::Ready(1);
    }
    let lines = lines_of(&data);
    for l in lines.iter().rev() {
        wln(io.out, l);
    }
    ShellPoll::Ready(0)
}

fn cmd_tail(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let mut n = 10usize;
    let mut bytes = false;
    let mut from_start = false;
    let mut files = Vec::new();
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        if a == "-n" {
            let Some(v) = it.next().cloned() else {
                ewln(io.err, "tail: option requires an argument -- 'n'");
                return ShellPoll::Ready(1);
            };
            from_start = v.starts_with('+');
            n = match v.trim_start_matches('+').trim_start_matches('-').parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of lines: {v}"));
                    return ShellPoll::Ready(1);
                }
            };
        } else if a == "-c" {
            let Some(v) = it.next().cloned() else {
                ewln(io.err, "tail: option requires an argument -- 'c'");
                return ShellPoll::Ready(1);
            };
            bytes = true;
            from_start = v.starts_with('+');
            n = match v.trim_start_matches(['+', '-']).parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of bytes: {v}"));
                    return ShellPoll::Ready(1);
                }
            };
        } else if let Some(v) = a.strip_prefix("-c") {
            bytes = true;
            from_start = v.starts_with('+');
            n = match v.trim_start_matches(['+', '-']).parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of bytes: {v}"));
                    return ShellPoll::Ready(1);
                }
            };
        } else if let Some(v) = a.strip_prefix('+').filter(|value| !value.is_empty()) {
            from_start = true;
            n = match v.parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of lines: {a}"));
                    return ShellPoll::Ready(1);
                }
            };
        } else if let Some(v) = a.strip_prefix("-n") {
            from_start = v.starts_with('+');
            n = match v.trim_start_matches('+').trim_start_matches('-').parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of lines: {v}"));
                    return ShellPoll::Ready(1);
                }
            };
        } else if let Some(v) = a.strip_prefix('-').filter(|value| {
            !value.is_empty() && value.chars().all(|character| character.is_ascii_digit())
        }) {
            n = v.parse().expect("validated decimal tail count");
        } else if a == "-f" || a == "-F" {
            ewln(io.err, "tail: unimplemented follow mode");
            return ShellPoll::Ready(2);
        } else if !a.starts_with('-') || a == "-" {
            files.push(a.clone());
        } else {
            ewln(io.err, &format!("tail: unimplemented option '{a}'"));
            return ShellPoll::Ready(2);
        }
    }
    if files.is_empty() || files.iter().any(|file| file == "-") {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let multiple = files.len() > 1;
    let files = if files.is_empty() {
        vec!["-".to_string()]
    } else {
        files
    };
    let mut status = 0;
    let mut emitted = false;
    for file in files {
        let data = if file == "-" {
            io.stdin.clone()
        } else {
            let cwd = context.system.cwd().to_string();
            let maximum = usize::try_from(context.system.limits().memory).unwrap_or(usize::MAX);
            match context.system.read_file_limited(&cwd, &file, maximum) {
                Ok(data) => data,
                Err(error) => {
                    ewln(io.err, &format!("tail: {file}: {error}"));
                    status = 1;
                    continue;
                }
            }
        };
        if multiple {
            if emitted {
                io.out.push(b'\n');
            }
            wln(io.out, &format!("==> {file} <=="));
        }
        if bytes {
            let start = if from_start {
                n.saturating_sub(1).min(data.len())
            } else {
                data.len().saturating_sub(n)
            };
            io.out.extend_from_slice(&data[start..]);
        } else {
            let lines = data
                .split_inclusive(|byte| *byte == b'\n')
                .collect::<Vec<_>>();
            let start = if from_start {
                n.saturating_sub(1).min(lines.len())
            } else {
                lines.len().saturating_sub(n)
            };
            for line in &lines[start..] {
                io.out.extend_from_slice(line);
            }
        }
        emitted = true;
    }
    ShellPoll::Ready(status)
}
