//! Deterministic byte-backed `sort` over simulated files and standard input.
//!
//! The implementation supports the common numeric, reverse, unique, fold-case, and field-key
//! modes. Other option surfaces fail explicitly instead of silently changing ordering semantics.

use std::collections::HashMap;

use super::options::{parse_options_or_report, OptionSpec};
use super::util::{ewln, read_inputs_system, uses_standard_input};
use super::{reg_system_poll, CommandSpec, Io, Trust};
use crate::exec::ShellPoll;
use crate::program::ProcessContext;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    reg_system_poll(m, "/usr/bin/sort", Trust::Real, run);
}

fn run(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    #[derive(Clone, Copy, PartialEq)]
    enum Key {
        Numeric,
        Reverse,
        Unique,
        FoldCase,
        Stable,
        Fields,
        Separator,
        Output,
        Help,
    }
    const OPTIONS: &[OptionSpec<Key>] = &[
        OptionSpec::flag(Key::Numeric, Some('n'), Some("numeric-sort")),
        OptionSpec::flag(Key::Reverse, Some('r'), Some("reverse")),
        OptionSpec::flag(Key::Unique, Some('u'), Some("unique")),
        OptionSpec::flag(Key::FoldCase, Some('f'), Some("ignore-case")),
        OptionSpec::flag(Key::Stable, Some('s'), Some("stable")),
        OptionSpec::required(Key::Fields, Some('k'), Some("key")),
        OptionSpec::required(Key::Separator, Some('t'), Some("field-separator")),
        OptionSpec::required(Key::Output, Some('o'), Some("output")),
        OptionSpec::flag(Key::Help, None, Some("help")),
    ];
    let parsed = match parse_options_or_report(
        "sort",
        args,
        OPTIONS,
        (
            Key::Help,
            "usage: sort [OPTIONS] [FILE...]\nsupported: -n -r -u -f -s -k KEY -t CHAR -o FILE\n",
        ),
        io.out,
        io.err,
    ) {
        Ok(parsed) => parsed,
        Err(status) => return ShellPoll::Ready(status),
    };
    let mut numeric = false;
    let mut reverse = false;
    let mut unique = false;
    let mut fold_case = false;
    let mut key = None;
    let mut separator = None;
    let mut output = None;
    for option in parsed.options {
        match option.key {
            Key::Numeric => numeric = true,
            Key::Reverse => reverse = true,
            Key::Unique => unique = true,
            Key::FoldCase => fold_case = true,
            Key::Stable => {}
            Key::Fields => {
                let value = option.value.expect("required option value");
                key = match parse_key(&value) {
                    Some(key) => Some(key),
                    None => {
                        ewln(
                            io.err,
                            &format!("sort: unsupported field specification '{value}'"),
                        );
                        return ShellPoll::Ready(2);
                    }
                };
            }
            Key::Separator => {
                let value = option.value.expect("required option value");
                let mut characters = value.chars();
                separator = match (characters.next(), characters.next()) {
                    (Some(separator), None) => Some(separator),
                    _ => {
                        ewln(io.err, "sort: field separator must be one character");
                        return ShellPoll::Ready(2);
                    }
                };
            }
            Key::Output => output = option.value,
            Key::Help => unreachable!("help is handled by the shared option parser"),
        }
    }
    let operands = parsed.operands.iter().collect::<Vec<_>>();
    if uses_standard_input(&operands) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &operands, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("sort: {error}"));
        return ShellPoll::Ready(1);
    }
    let line_count = data
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        .saturating_add(1) as u64;
    let scratch = 16_u64.saturating_mul(1024).saturating_add(
        (data.len() as u64)
            .saturating_mul(2)
            .saturating_add(line_count.saturating_mul(24)),
    );
    let comparisons = line_count.saturating_mul(line_count.max(1).ilog2() as u64);
    if !context.system.reserve_memory(scratch) {
        return ShellPoll::Ready(context.system.stop_status());
    }
    if !context
        .system
        .charge_cpu((data.len() as u64).saturating_add(comparisons))
    {
        context.system.release_memory(scratch);
        return ShellPoll::Ready(context.system.stop_status());
    }

    let mut lines = data
        .split_inclusive(|byte| *byte == b'\n')
        .map(<[u8]>::to_vec)
        .collect::<Vec<_>>();
    let compare = |a: &[u8], b: &[u8]| {
        let text = |line: &[u8]| {
            String::from_utf8_lossy(line)
                .trim_end_matches('\n')
                .to_string()
        };
        let field = |line: &[u8]| {
            let value = text(line);
            key.and_then(|key| {
                if let Some(separator) = separator {
                    value.split(separator).nth(key.field).map(str::to_string)
                } else {
                    value.split_whitespace().nth(key.field).map(str::to_string)
                }
            })
            .unwrap_or(value)
        };
        let ordering = if numeric || key.is_some_and(|key| key.numeric) {
            let number = |line: &[u8]| {
                field(line)
                    .split_whitespace()
                    .next()
                    .and_then(|word| word.parse::<f64>().ok())
                    .unwrap_or(0.0)
            };
            number(a)
                .partial_cmp(&number(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        } else if fold_case || key.is_some_and(|key| key.fold_case) {
            field(a).to_lowercase().cmp(&field(b).to_lowercase())
        } else {
            field(a).cmp(&field(b))
        };
        if key.is_some_and(|key| key.reverse) {
            ordering.reverse()
        } else {
            ordering
        }
    };
    lines.sort_by(|a, b| compare(a, b));
    if reverse {
        lines.reverse();
    }
    if unique {
        lines.dedup_by(|a, b| compare(a, b).is_eq());
    }
    let sorted = lines.into_iter().flatten().collect::<Vec<_>>();
    let status = if let Some(path) = output {
        let cwd = context.system.cwd().to_string();
        match context.system.write_file(&cwd, &path, &sorted, 0o644) {
            Ok(()) => 0,
            Err(error) => {
                ewln(io.err, &format!("sort: {path}: {error}"));
                1
            }
        }
    } else {
        io.out.extend_from_slice(&sorted);
        0
    };
    context.system.release_memory(scratch);
    ShellPoll::Ready(status)
}

#[derive(Clone, Copy)]
struct SortKey {
    field: usize,
    numeric: bool,
    reverse: bool,
    fold_case: bool,
}

fn parse_key_bound(value: &str) -> Option<SortKey> {
    let digits = value
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .count();
    if digits == 0 || value[digits..].contains('.') {
        return None;
    }
    let field = value[..digits].parse::<usize>().ok()?.checked_sub(1)?;
    let mut key = SortKey {
        field,
        numeric: false,
        reverse: false,
        fold_case: false,
    };
    for modifier in value[digits..].chars() {
        match modifier {
            'n' => key.numeric = true,
            'r' => key.reverse = true,
            'f' => key.fold_case = true,
            _ => return None,
        }
    }
    Some(key)
}

fn parse_key(value: &str) -> Option<SortKey> {
    let mut bounds = value.split(',');
    let mut start = parse_key_bound(bounds.next()?)?;
    if let Some(end) = bounds.next() {
        let end = parse_key_bound(end)?;
        if bounds.next().is_some() || end.field != start.field {
            return None;
        }
        start.numeric |= end.numeric;
        start.reverse |= end.reverse;
        start.fold_case |= end.fold_case;
    }
    Some(start)
}

#[cfg(test)]
mod tests {
    use super::parse_key;

    #[test]
    fn key_parser_accepts_common_modifiers_and_rejects_ranges() {
        let key = parse_key("2n").unwrap();
        assert_eq!(key.field, 1);
        assert!(key.numeric);

        let key = parse_key("3,3rf").unwrap();
        assert_eq!(key.field, 2);
        assert!(key.reverse);
        assert!(key.fold_case);

        assert!(parse_key("2,3").is_none());
        assert!(parse_key("2.1").is_none());
        assert!(parse_key("2M").is_none());
    }
}
