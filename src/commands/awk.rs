//! Deterministic awk interpreter for common record-processing programs.
//!
//! Source is tokenized and parsed completely before simulated input is read. The direct evaluator
//! supports ordinary control flow, associative arrays, fields, expressions, and common scalar
//! functions. Unsupported grammar and functions fail explicitly; awk cannot access host state.

use std::collections::HashMap;

use crate::commands::options::{parse_options_or_report, OptionSpec};
use crate::commands::util::ewln;
use crate::commands::{CommandSpec, Io, Trust};
use crate::exec::ShellPoll;
use crate::program::ProcessContext;

mod ast;
mod eval;
mod lexer;
mod parser;

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    for name in ["awk", "gawk", "mawk", "nawk"] {
        let path = match name {
            "awk" => "/usr/bin/awk",
            "gawk" => "/usr/bin/gawk",
            "mawk" => "/usr/bin/mawk",
            _ => "/usr/bin/nawk",
        };
        super::reg_system_poll_costed(commands, path, Trust::Partial, 150, 24 * 1024, run);
    }
}

fn run(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    #[derive(Clone, Copy, PartialEq)]
    enum Key {
        FieldSeparator,
        Variable,
        ProgramFile,
        Help,
    }
    const OPTIONS: &[OptionSpec<Key>] = &[
        OptionSpec::required(Key::FieldSeparator, Some('F'), Some("field-separator")),
        OptionSpec::required(Key::Variable, Some('v'), Some("assign")),
        OptionSpec::required(Key::ProgramFile, Some('f'), Some("file")),
        OptionSpec::flag(Key::Help, None, Some("help")),
    ];
    let parsed = match parse_options_or_report(
        "awk",
        context.args,
        OPTIONS,
        (
            Key::Help,
            "usage: awk [-F SEP] [-v NAME=VALUE] [-f PROGRAM] [PROGRAM] [FILE...]\n",
        ),
        io.out,
        io.err,
    ) {
        Ok(parsed) => parsed,
        Err(status) => return ShellPoll::Ready(status),
    };

    let mut field_separator = " ".to_string();
    let mut variables = HashMap::new();
    let mut sources = Vec::new();
    for option in parsed.options {
        let value = option.value.expect("required option value");
        match option.key {
            Key::FieldSeparator => field_separator = value,
            Key::Variable => {
                let Some((name, value)) = assignment(&value) else {
                    ewln(
                        io.err,
                        &format!("awk: invalid variable assignment '{value}'"),
                    );
                    return ShellPoll::Ready(2);
                };
                variables.insert(name.to_string(), value.to_string());
            }
            Key::ProgramFile => {
                let cwd = context.system.cwd().to_string();
                let maximum = usize::try_from(context.system.limits().memory).unwrap_or(usize::MAX);
                match context.system.read_file_limited(&cwd, &value, maximum) {
                    Ok(bytes) => match String::from_utf8(bytes) {
                        Ok(source) => sources.push(source),
                        Err(_) => {
                            ewln(io.err, &format!("awk: {value}: program is not valid UTF-8"));
                            return ShellPoll::Ready(2);
                        }
                    },
                    Err(error) => {
                        ewln(io.err, &format!("awk: {value}: {error}"));
                        return ShellPoll::Ready(2);
                    }
                }
            }
            Key::Help => unreachable!("help is handled by the shared option parser"),
        }
    }

    let mut operands = parsed.operands.into_iter();
    if sources.is_empty() {
        let Some(source) = operands.next() else {
            ewln(io.err, "awk: missing program");
            return ShellPoll::Ready(2);
        };
        sources.push(source);
    }
    let source = sources.join("\n");
    if !context.system.charge_cpu(source.len() as u64) {
        return ShellPoll::Ready(context.system.stop_status());
    }
    let program = match parser::parse(&source).and_then(|program| {
        eval::validate(&program)?;
        Ok(program)
    }) {
        Ok(program) => program,
        Err(error) => {
            context.system.note_unsupported(&format!("awk:{error}"));
            ewln(io.err, &format!("awk: {error}"));
            return ShellPoll::Ready(2);
        }
    };

    let mut files = Vec::new();
    for operand in operands {
        if let Some((name, value)) = assignment(&operand) {
            variables.insert(name.to_string(), value.to_string());
        } else {
            files.push(operand);
        }
    }

    if let Some(name) = ["NR", "FNR", "NF", "FILENAME", "RS"]
        .into_iter()
        .find(|name| variables.contains_key(*name))
    {
        ewln(
            io.err,
            &format!("awk: assignment to '{name}' is not supported"),
        );
        return ShellPoll::Ready(2);
    }

    let effective_separator = variables
        .get("FS")
        .map_or(field_separator.as_str(), String::as_str);
    if let Err(error) = eval::validate_field_separator(effective_separator) {
        ewln(io.err, &format!("awk: {error}"));
        return ShellPoll::Ready(2);
    }

    if files.is_empty() || files.iter().any(|file| file == "-") {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }

    let inputs = if files.is_empty() {
        match String::from_utf8(std::mem::take(&mut io.stdin)) {
            Ok(input) => vec![("-".to_string(), input)],
            Err(_) => {
                ewln(io.err, "awk: input is not valid UTF-8");
                return ShellPoll::Ready(2);
            }
        }
    } else {
        let mut inputs = Vec::with_capacity(files.len());
        for file in files {
            let data = if file == "-" {
                std::mem::take(&mut io.stdin)
            } else {
                let cwd = context.system.cwd().to_string();
                let maximum = usize::try_from(context.system.limits().memory).unwrap_or(usize::MAX);
                match context.system.read_file_limited(&cwd, &file, maximum) {
                    Ok(data) => data,
                    Err(error) => {
                        ewln(io.err, &format!("awk: {file}: {error}"));
                        return ShellPoll::Ready(2);
                    }
                }
            };
            let text = match String::from_utf8(data) {
                Ok(text) => text,
                Err(_) => {
                    ewln(io.err, &format!("awk: {file}: input is not valid UTF-8"));
                    return ShellPoll::Ready(2);
                }
            };
            inputs.push((file, text));
        }
        inputs
    };

    let input_bytes = inputs
        .iter()
        .map(|(_, input)| input.len() as u64)
        .sum::<u64>();
    let scratch = input_bytes.saturating_mul(2);
    if !context.system.reserve_memory(scratch) {
        return ShellPoll::Ready(context.system.stop_status());
    }
    if !context.system.charge_cpu(input_bytes) {
        context.system.release_memory(scratch);
        return ShellPoll::Ready(context.system.stop_status());
    }
    let status = eval::execute(
        &program,
        context.system,
        io,
        variables,
        field_separator,
        inputs,
    );
    context.system.release_memory(scratch);
    ShellPoll::Ready(status)
}

fn assignment(value: &str) -> Option<(&str, &str)> {
    let (name, value) = value.split_once('=')?;
    valid_name(name).then_some((name, value))
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().enumerate().all(|(index, character)| {
            character == '_'
                || character.is_ascii_alphabetic()
                || (index > 0 && character.is_ascii_digit())
        })
}

#[cfg(test)]
mod tests {
    use crate::interp::Interp;

    #[test]
    fn fields_control_flow_arrays_and_functions_compose() {
        let mut environment = Interp::new();
        let (_, stdout, stderr) = environment.run_script_capture(
            "printf 'alice,2\\nbob,3\\nalice,4\\n' | awk -F, '\
             { if ($2 > 2 && !($1 in seen)) { seen[$1] = 1; total += $2 } } \
             END { for (name in seen) print toupper(substr(name, 1, 1)), total }'",
        );
        assert_eq!(String::from_utf8_lossy(&stdout), "A 7\nB 7\n");
        assert!(stderr.is_empty(), "{}", String::from_utf8_lossy(&stderr));
    }

    #[test]
    fn malformed_programs_and_arguments_fail_loudly() {
        let mut environment = Interp::new();
        for source in [
            "awk -v broken 'BEGIN { print 1 }'",
            "awk 'BEGIN { unknown(1) }'",
            "awk 'BEGIN { if (1) print 1'",
            "awk '{ print }' /missing",
        ] {
            let (outcome, _, stderr) = environment.run_script_capture(source);
            assert_eq!(
                outcome.exit_status,
                2,
                "{source}: {}",
                String::from_utf8_lossy(&stderr)
            );
            assert!(!stderr.is_empty(), "{source}");
        }
    }
}
