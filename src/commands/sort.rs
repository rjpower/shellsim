//! Deterministic byte-backed `sort` over simulated files and standard input.
//!
//! The implementation supports the common numeric, reverse, unique, fold-case, and field-key
//! modes. Other option surfaces fail explicitly instead of silently changing ordering semantics.

use std::collections::HashMap;

use super::util::{ewln, read_inputs};
use super::{reg_costed, CommandContext, CommandSpec, Io, Trust};

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    reg_costed(m, &["sort"], Trust::Real, 100, 16 * 1024, run);
}

fn run(env: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut numeric = false;
    let mut reverse = false;
    let mut unique = false;
    let mut fold_case = false;
    let mut key = None;
    let mut operands = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "-k" {
            let Some(value) = args.get(index + 1) else {
                ewln(io.err, "sort: option requires an argument -- 'k'");
                return 2;
            };
            key = parse_key(value);
            if key.is_none() {
                ewln(io.err, "sort: invalid field specification");
                return 2;
            }
            index += 2;
            continue;
        }
        if let Some(value) = argument
            .strip_prefix("-k")
            .filter(|value| !value.is_empty())
        {
            key = parse_key(value);
            if key.is_none() {
                ewln(io.err, "sort: invalid field specification");
                return 2;
            }
            index += 1;
            continue;
        }
        if argument.starts_with('-') && argument != "-" {
            for flag in argument[1..].chars() {
                match flag {
                    'n' => numeric = true,
                    'r' => reverse = true,
                    'u' => unique = true,
                    'f' => fold_case = true,
                    _ => {
                        ewln(io.err, &format!("sort: unimplemented option '-{flag}'"));
                        return 2;
                    }
                }
            }
        } else {
            operands.push(argument);
        }
        index += 1;
    }
    let (data, errors) = read_inputs(env, &operands, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("sort: {error}"));
        return 1;
    }
    let line_count = data
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        .saturating_add(1) as u64;
    let scratch = (data.len() as u64)
        .saturating_mul(2)
        .saturating_add(line_count.saturating_mul(24));
    let comparisons = line_count.saturating_mul(line_count.max(1).ilog2() as u64);
    if !env.reserve_memory(scratch) {
        return 137;
    }
    if !env.charge_cpu(data.len() as u64 + comparisons) {
        env.resources.release_memory(scratch);
        return 137;
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
            key.and_then(|position| value.split_whitespace().nth(position).map(str::to_string))
                .unwrap_or(value)
        };
        if numeric {
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
        } else if fold_case {
            field(a).to_lowercase().cmp(&field(b).to_lowercase())
        } else {
            field(a).cmp(&field(b))
        }
    };
    lines.sort_by(|a, b| compare(a, b));
    if reverse {
        lines.reverse();
    }
    if unique {
        lines.dedup_by(|a, b| compare(a, b).is_eq());
    }
    for line in lines {
        io.out.extend_from_slice(&line);
    }
    env.resources.release_memory(scratch);
    0
}

fn parse_key(value: &str) -> Option<usize> {
    let mut bounds = value.split(',');
    let start = bounds.next()?;
    let field = start.parse::<usize>().ok()?;
    if let Some(end) = bounds.next() {
        if bounds.next().is_some() || end.parse::<usize>().ok()? != field {
            return None;
        }
    }
    field.checked_sub(1)
}
