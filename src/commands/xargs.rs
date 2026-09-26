//! Argument batching and sequential child dispatch for xargs.
//!
//! Shell execution uses scheduler-owned logical children. Direct native dispatch retains the
//! synchronous entry point used by command composition outside a shell continuation.

use crate::commands::util::ewln;

const MAX_TOKENS: usize = 65_536;
const MAX_ARGUMENT_BYTES: usize = 16 * 1024 * 1024;

pub(crate) fn xargs_commands(
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
            .take(MAX_TOKENS + 1)
            .map(str::to_string)
            .collect()
    } else {
        input
            .split_whitespace()
            .take(MAX_TOKENS + 1)
            .map(str::to_string)
            .collect()
    };
    if tokens.len() > MAX_TOKENS {
        ewln(err, "xargs: too many input arguments");
        return Err(1);
    }
    let base_bytes = cmd.iter().map(String::len).sum::<usize>();
    if tokens.is_empty() {
        if base_bytes > MAX_ARGUMENT_BYTES {
            ewln(err, "xargs: arguments exceed 16 MiB");
            return Err(1);
        }
        return Ok(if no_run_if_empty {
            Vec::new()
        } else {
            vec![cmd]
        });
    }
    let mut commands = Vec::new();
    let mut argument_bytes = 0usize;
    if let Some(ph) = replace {
        if ph.is_empty() {
            ewln(err, "xargs: empty replacement marker");
            return Err(1);
        }
        for token in &tokens {
            let expanded_bytes = cmd.iter().try_fold(0usize, |total, arg| {
                let count = arg.matches(&ph).count();
                let removed = count.checked_mul(ph.len())?;
                let inserted = count.checked_mul(token.len())?;
                total.checked_add(arg.len().checked_sub(removed)?.checked_add(inserted)?)
            });
            argument_bytes = argument_bytes.saturating_add(expanded_bytes.unwrap_or(usize::MAX));
            if argument_bytes > MAX_ARGUMENT_BYTES {
                ewln(err, "xargs: arguments exceed 16 MiB");
                return Err(1);
            }
            let argv: Vec<String> = cmd.iter().map(|c| c.replace(&ph, token)).collect();
            commands.push(argv);
        }
    } else {
        let chunk = nper.unwrap_or(tokens.len().max(1));
        for batch in tokens.chunks(chunk.max(1)) {
            argument_bytes = argument_bytes.saturating_add(base_bytes);
            argument_bytes = argument_bytes.saturating_add(batch.iter().map(String::len).sum());
            if argument_bytes > MAX_ARGUMENT_BYTES {
                ewln(err, "xargs: arguments exceed 16 MiB");
                return Err(1);
            }
            let mut argv = cmd.clone();
            argv.extend(batch.iter().cloned());
            commands.push(argv);
        }
    }
    Ok(commands)
}
