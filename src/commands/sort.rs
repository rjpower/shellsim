use std::collections::HashMap;

use super::util::{read_inputs, split_flags, wln};
use super::{reg_costed, CommandContext, CommandSpec, Io, Trust};

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    reg_costed(m, &["sort"], Trust::Real, 100, 16 * 1024, run);
}

fn run(env: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, operands, _) = split_flags(args);
    let (data, _) = read_inputs(env, &operands, &io.stdin);
    let line_count = data
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        .saturating_add(1) as u64;
    let scratch = (data.len() as u64)
        .saturating_mul(2)
        .saturating_add(line_count.saturating_mul(24));
    let comparisons = line_count.saturating_mul(line_count.max(1).ilog2() as u64);
    if !env.reserve_memory(scratch) || !env.charge_cpu(data.len() as u64 + comparisons) {
        return 137;
    }

    let mut lines: Vec<String> = String::from_utf8_lossy(&data)
        .lines()
        .map(str::to_string)
        .collect();
    if flags.contains(&'n') {
        lines.sort_by(|a, b| {
            let number = |line: &str| {
                line.split_whitespace()
                    .next()
                    .and_then(|word| word.parse::<f64>().ok())
                    .unwrap_or(0.0)
            };
            number(a)
                .partial_cmp(&number(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        lines.sort();
    }
    if flags.contains(&'r') {
        lines.reverse();
    }
    if flags.contains(&'u') {
        lines.dedup();
    }
    for line in lines {
        wln(io.out, &line);
    }
    0
}
