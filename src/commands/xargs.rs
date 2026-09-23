//! Argument batching and sequential child dispatch for xargs.
//!
//! Shell execution uses scheduler-owned logical children. Direct native dispatch retains the
//! synchronous entry point used by command composition outside a shell continuation.

use std::collections::HashMap;

use crate::commands::util::ewln;
use crate::commands::{ChildCommand, CommandContext, CommandPoll, CommandSpec, Io, Trust};

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg_buffered_resumable;
    reg_buffered_resumable(m, &["xargs"], Trust::Real, cmd_xargs, start_xargs);
}

fn cmd_xargs(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let commands = match xargs_commands(args, &io.stdin, io.err) {
        Ok(commands) => commands,
        Err(status) => return status,
    };
    let mut status = 0;
    for argv in commands {
        status = crate::commands::run(interp, &argv, Vec::new(), io.out, io.err);
    }
    status
}

fn start_xargs(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let commands = match xargs_commands(args, &io.stdin, io.err) {
        Ok(commands) => commands,
        Err(status) => return CommandPoll::Ready(status),
    };
    crate::commands::start_child_sequence(
        interp,
        commands
            .into_iter()
            .map(|argv| ChildCommand {
                argv,
                stdin: Vec::new(),
                cwd: None,
                environment: None,
            })
            .collect(),
        false,
    )
}

fn xargs_commands(
    args: &[String],
    stdin: &[u8],
    err: &mut Vec<u8>,
) -> Result<Vec<Vec<String>>, i32> {
    let mut i = 0;
    let mut replace: Option<String> = None;
    let mut nper: Option<usize> = None;
    let mut nul_delimited = false;
    let mut no_run_if_empty = false;
    while i < args.len() {
        match args[i].as_str() {
            "-I" => {
                let Some(value) = args.get(i + 1) else {
                    ewln(err, "xargs: option requires an argument -- 'I'");
                    return Err(1);
                };
                replace = Some(value.clone());
                i += 2;
            }
            option if option.starts_with("-I") && option.len() > 2 => {
                replace = Some(option[2..].to_string());
                i += 1;
            }
            "-n" => {
                let Some(value) = args.get(i + 1).and_then(|s| s.parse().ok()) else {
                    ewln(err, "xargs: invalid number for -n");
                    return Err(1);
                };
                if value == 0 {
                    ewln(err, "xargs: -n requires a positive number");
                    return Err(1);
                }
                nper = Some(value);
                i += 2;
            }
            option if option.starts_with("-n") && option.len() > 2 => {
                let Some(value) = option[2..].parse().ok() else {
                    ewln(err, "xargs: invalid number for -n");
                    return Err(1);
                };
                if value == 0 {
                    ewln(err, "xargs: -n requires a positive number");
                    return Err(1);
                }
                nper = Some(value);
                i += 1;
            }
            "-0" => {
                nul_delimited = true;
                i += 1;
            }
            "-r" | "--no-run-if-empty" => {
                no_run_if_empty = true;
                i += 1;
            }
            "--" => {
                i += 1;
                break;
            }
            option if option.starts_with('-') => {
                ewln(err, &format!("xargs: unsupported option '{option}'"));
                return Err(1);
            }
            _ => break,
        }
    }
    let mut cmd: Vec<String> = args[i..].to_vec();
    if cmd.is_empty() {
        cmd.push("echo".to_string());
    }
    let input = String::from_utf8_lossy(stdin);
    let tokens: Vec<String> = if nul_delimited {
        input
            .split('\0')
            .filter(|token| !token.is_empty())
            .map(str::to_string)
            .collect()
    } else {
        input.split_whitespace().map(str::to_string).collect()
    };
    if tokens.is_empty() {
        return Ok(if no_run_if_empty {
            Vec::new()
        } else {
            vec![cmd]
        });
    }
    let mut commands = Vec::new();
    if let Some(ph) = replace {
        for token in &tokens {
            let argv: Vec<String> = cmd.iter().map(|c| c.replace(&ph, token)).collect();
            commands.push(argv);
        }
    } else {
        let chunk = nper.unwrap_or(tokens.len().max(1));
        for batch in tokens.chunks(chunk.max(1)) {
            let mut argv = cmd.clone();
            argv.extend(batch.iter().cloned());
            commands.push(argv);
        }
    }
    Ok(commands)
}
