//! Buffered record transforms and small arithmetic helpers.
//!
//! Streaming I/O, comparison, formatting, pattern matching, and child batching live in focused
//! sibling modules.

use std::collections::HashMap;

use crate::commands::util::{
    ewln, lines_of, read_inputs_system, split_flags, uses_standard_input, w, wln,
};
use crate::commands::{CommandSpec, Io, Trust};
use crate::exec::ShellPoll;
use crate::program::ProcessContext;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::{reg_system, reg_system_poll, reg_unsupported};
    reg_system_poll(m, "/usr/bin/wc", Trust::Real, cmd_wc);
    reg_system_poll(m, "/usr/bin/uniq", Trust::Real, cmd_uniq);
    reg_system_poll(m, "/usr/bin/cut", Trust::Real, cmd_cut);
    reg_system_poll(m, "/usr/bin/tr", Trust::Real, cmd_tr);
    reg_system_poll(m, "/usr/bin/rev", Trust::Real, cmd_rev);
    reg_system_poll(m, "/usr/bin/nl", Trust::Real, cmd_nl);
    reg_system(m, "/usr/bin/seq", Trust::Real, cmd_seq);
    reg_system_poll(m, "/usr/bin/paste", Trust::Real, cmd_paste);
    reg_unsupported(m, &["pr"]);
    reg_system_poll(m, "/usr/bin/comm", Trust::Real, cmd_comm);
    reg_system_poll(m, "/usr/bin/join", Trust::Partial, cmd_join);
    reg_system_poll(m, "/usr/bin/split", Trust::Partial, cmd_split);
    reg_system_poll(m, "/usr/bin/shuf", Trust::Partial, cmd_shuf);
    reg_system_poll(m, "/usr/bin/tsort", Trust::Real, cmd_tsort);
    reg_unsupported(m, &["factor"]);
}

fn cmd_wc(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let (flags, ops, _l) = split_flags(args);
    if flags
        .iter()
        .any(|flag| !matches!(flag, 'l' | 'w' | 'c' | 'm'))
    {
        ewln(io.err, "wc: unimplemented option");
        return ShellPoll::Ready(2);
    }
    if uses_standard_input(&ops) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (cl, cw, cc, cm) = (
        flags.contains(&'l'),
        flags.contains(&'w'),
        flags.contains(&'c'),
        flags.contains(&'m'),
    );
    let none = !cl && !cw && !cc && !cm;
    let counts = |data: &[u8]| {
        let s = String::from_utf8_lossy(data);
        (
            s.matches('\n').count(),
            s.split_whitespace().count(),
            data.len(),
            s.chars().count(),
        )
    };
    let print_counts = |values: (usize, usize, usize, usize), out: &mut Vec<u8>, label: &str| {
        let (lines, words, bytes, chars) = values;
        let mut parts = Vec::new();
        if cl || none {
            parts.push(format!("{:>7}", lines));
        }
        if cw || none {
            parts.push(format!("{:>7}", words));
        }
        if cc || none {
            parts.push(format!("{:>7}", bytes));
        }
        if cm {
            parts.push(format!("{:>7}", chars));
        }
        let mut line = parts.join(" ");
        if !label.is_empty() {
            line.push(' ');
            line.push_str(label);
        }
        wln(out, line.trim_start());
    };
    if ops.is_empty() {
        print_counts(counts(&io.stdin), io.out, "");
    } else {
        let mut totals = (0usize, 0usize, 0usize, 0usize);
        let mut status = 0;
        for f in &ops {
            let (data, errors) = read_inputs_system(context.system, &[*f], &io.stdin);
            if errors.is_empty() {
                let values = counts(&data);
                print_counts(values, io.out, f);
                totals.0 = totals.0.saturating_add(values.0);
                totals.1 = totals.1.saturating_add(values.1);
                totals.2 = totals.2.saturating_add(values.2);
                totals.3 = totals.3.saturating_add(values.3);
            } else {
                ewln(io.err, &format!("wc: {}", errors[0]));
                status = 1;
            }
        }
        if ops.len() > 1 {
            print_counts(totals, io.out, "total");
        }
        return ShellPoll::Ready(status);
    }
    ShellPoll::Ready(0)
}

fn cmd_uniq(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let mut count = false;
    let mut only_dup = false;
    let mut only_uniq = false;
    let mut ignore_case = false;
    let mut skip_fields = 0usize;
    let mut skip_chars = 0usize;
    let mut operands = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        let parse_count = |value: &str, option: &str, io: &mut Io| {
            value.parse::<usize>().map_err(|_| {
                ewln(
                    io.err,
                    &format!("uniq: invalid number for {option}: {value}"),
                );
                1
            })
        };
        if argument == "-f" || argument == "-s" {
            let Some(value) = args.get(index + 1) else {
                ewln(io.err, &format!("uniq: {argument} requires an argument"));
                return ShellPoll::Ready(1);
            };
            let parsed = match parse_count(value, argument, io) {
                Ok(value) => value,
                Err(status) => return ShellPoll::Ready(status),
            };
            if argument == "-f" {
                skip_fields = parsed;
            } else {
                skip_chars = parsed;
            }
            index += 2;
            continue;
        }
        if let Some(value) = argument
            .strip_prefix("-f")
            .filter(|value| !value.is_empty())
        {
            skip_fields = match parse_count(value, "-f", io) {
                Ok(value) => value,
                Err(status) => return ShellPoll::Ready(status),
            };
        } else if let Some(value) = argument
            .strip_prefix("-s")
            .filter(|value| !value.is_empty())
        {
            skip_chars = match parse_count(value, "-s", io) {
                Ok(value) => value,
                Err(status) => return ShellPoll::Ready(status),
            };
        } else if argument.starts_with('-') && argument != "-" {
            for flag in argument[1..].chars() {
                match flag {
                    'c' => count = true,
                    'd' => only_dup = true,
                    'u' => only_uniq = true,
                    'i' => ignore_case = true,
                    _ => {
                        ewln(io.err, &format!("uniq: unimplemented option '-{flag}'"));
                        return ShellPoll::Ready(2);
                    }
                }
            }
        } else {
            operands.push(argument);
        }
        index += 1;
    }
    if operands.len() > 1 {
        ewln(io.err, "uniq: unimplemented output-file operand");
        return ShellPoll::Ready(2);
    }
    if uses_standard_input(&operands) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &operands, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("uniq: {error}"));
        return ShellPoll::Ready(1);
    }
    let lines = data
        .split_inclusive(|byte| *byte == b'\n')
        .map(<[u8]>::to_vec)
        .collect::<Vec<_>>();
    let key = |line: &[u8]| {
        let text = String::from_utf8_lossy(line);
        let mut offset = 0;
        for _ in 0..skip_fields {
            let rest = &text[offset..];
            let blanks = rest.len() - rest.trim_start_matches(char::is_whitespace).len();
            offset += blanks;
            let rest = &text[offset..];
            let field = rest.find(char::is_whitespace).unwrap_or(rest.len());
            offset += field;
        }
        let value = text[offset..].chars().skip(skip_chars).collect::<String>();
        if ignore_case {
            value.to_lowercase()
        } else {
            value
        }
    };
    let mut i = 0;
    while i < lines.len() {
        let mut j = i + 1;
        let current = key(&lines[i]);
        while j < lines.len() && key(&lines[j]) == current {
            j += 1;
        }
        let n = j - i;
        let show = (!only_dup && !only_uniq) || (only_dup && n > 1) || (only_uniq && n == 1);
        if show {
            if count {
                w(io.out, &format!("{:>7} ", n));
                io.out.extend_from_slice(&lines[i]);
            } else {
                io.out.extend_from_slice(&lines[i]);
            }
        }
        i = j;
    }
    ShellPoll::Ready(0)
}

fn cmd_cut(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let mut delim = '\t';
    let mut fields: Option<String> = None;
    let mut chars_spec: Option<String> = None;
    let mut files = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if let Some(d) = a.strip_prefix("-d") {
            delim = if d.is_empty() {
                it.next().and_then(|s| s.chars().next()).unwrap_or('\t')
            } else {
                d.chars().next().unwrap_or('\t')
            };
        } else if let Some(f) = a.strip_prefix("-f") {
            fields = Some(if f.is_empty() {
                it.next().cloned().unwrap_or_default()
            } else {
                f.to_string()
            });
        } else if let Some(c) = a.strip_prefix("-c") {
            chars_spec = Some(if c.is_empty() {
                it.next().cloned().unwrap_or_default()
            } else {
                c.to_string()
            });
        } else if !a.starts_with('-') || a == "-" {
            files.push(a);
        }
    }
    if uses_standard_input(&files) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("cut: {error}"));
        return ShellPoll::Ready(1);
    }
    let parse_ranges = |spec: &str, max: usize| -> Vec<usize> {
        let mut idx = Vec::new();
        for part in spec.split(',') {
            if let Some((a, b)) = part.split_once('-') {
                let lo: usize = a.parse().unwrap_or(1);
                let hi: usize = if b.is_empty() {
                    max
                } else {
                    b.parse().unwrap_or(max)
                };
                for k in lo..=hi.min(max) {
                    idx.push(k);
                }
            } else if let Ok(k) = part.parse() {
                idx.push(k);
            }
        }
        idx
    };
    for line in String::from_utf8_lossy(&data).lines() {
        if let Some(spec) = &fields {
            let parts: Vec<&str> = line.split(delim).collect();
            if !line.contains(delim) {
                wln(io.out, line);
                continue;
            }
            let idx = parse_ranges(spec, parts.len());
            let selected: Vec<&str> = idx
                .iter()
                .filter_map(|k| parts.get(k - 1).copied())
                .collect();
            wln(io.out, &selected.join(&delim.to_string()));
        } else if let Some(spec) = &chars_spec {
            let chars: Vec<char> = line.chars().collect();
            let idx = parse_ranges(spec, chars.len());
            let selected: String = idx.iter().filter_map(|k| chars.get(k - 1)).collect();
            wln(io.out, &selected);
        }
    }
    ShellPoll::Ready(0)
}

fn cmd_tr(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let (flags, ops, _l) = split_flags(args);
    let delete = flags.contains(&'d');
    let squeeze = flags.contains(&'s');
    let complement = flags.contains(&'c');
    if flags.iter().any(|flag| !matches!(flag, 'c' | 'd' | 's')) {
        ewln(io.err, "tr: unimplemented option");
        return ShellPoll::Ready(2);
    }
    let required = if delete || (squeeze && ops.len() == 1) {
        1
    } else {
        2
    };
    if ops.len() != required {
        ewln(io.err, "tr: missing or extra operand");
        return ShellPoll::Ready(1);
    }
    if let Err(poll) = context.read_standard_input(io) {
        return poll;
    }
    let set1 = ops.first().map(|s| expand_tr_set(s)).unwrap_or_default();
    let set2 = ops
        .get(1)
        .map(|s| expand_tr_set(s))
        .unwrap_or_else(|| set1.clone());
    let input = String::from_utf8_lossy(&io.stdin).into_owned();
    let mut result = String::new();
    if delete {
        for c in input.chars() {
            let in_set = set1.contains(&c);
            if in_set != complement {
                continue;
            }
            result.push(c);
        }
    } else {
        let mut last = None;
        for c in input.chars() {
            let mapped = if let Some(pos) = set1.iter().position(|x| *x == c) {
                set2.get(pos)
                    .copied()
                    .or_else(|| set2.last().copied())
                    .unwrap_or(c)
            } else {
                c
            };
            if squeeze && Some(mapped) == last && set2.contains(&mapped) {
                continue;
            }
            result.push(mapped);
            last = Some(mapped);
        }
    }
    w(io.out, &result);
    ShellPoll::Ready(0)
}

fn expand_tr_set(s: &str) -> Vec<char> {
    // handle ranges like a-z and classes [:digit:] minimally
    let mut out = Vec::new();
    let s = s
        .replace("[:digit:]", "0123456789")
        .replace("[:lower:]", "abcdefghijklmnopqrstuvwxyz")
        .replace("[:upper:]", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        .replace(
            "[:alpha:]",
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        )
        .replace("[:space:]", " \t\n\r");
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() {
            out.push(match chars[i + 1] {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '\\' => '\\',
                value => value,
            });
            i += 2;
        } else if i + 2 < chars.len() && chars[i + 1] == '-' {
            let (lo, hi) = (chars[i], chars[i + 2]);
            for c in lo..=hi {
                out.push(c);
            }
            i += 3;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn cmd_rev(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let (_f, ops, _l) = split_flags(args);
    if uses_standard_input(&ops) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &ops, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("rev: {error}"));
        return ShellPoll::Ready(1);
    }
    for line in data.split_inclusive(|byte| *byte == b'\n') {
        let newline = line.ends_with(b"\n");
        let content = if newline {
            &line[..line.len() - 1]
        } else {
            line
        };
        let reversed = String::from_utf8_lossy(content)
            .chars()
            .rev()
            .collect::<String>();
        w(io.out, &reversed);
        if newline {
            io.out.push(b'\n');
        }
    }
    ShellPoll::Ready(0)
}

fn cmd_nl(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let (_f, ops, _l) = split_flags(args);
    if uses_standard_input(&ops) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &ops, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("nl: {error}"));
        return ShellPoll::Ready(1);
    }
    let mut n = 1;
    for line in String::from_utf8_lossy(&data).lines() {
        if line.is_empty() {
            wln(io.out, "");
        } else {
            wln(io.out, &format!("{:>6}\t{}", n, line));
            n += 1;
        }
    }
    ShellPoll::Ready(0)
}

fn cmd_seq(context: &mut ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    if args
        .iter()
        .any(|argument| argument.starts_with('-') && argument.parse::<f64>().is_err())
    {
        ewln(io.err, "seq: unimplemented option");
        return 2;
    }
    let nums: Vec<f64> = match args.iter().map(|a| a.parse::<f64>()).collect() {
        Ok(values) => values,
        Err(_) => {
            ewln(io.err, "seq: invalid floating point argument");
            return 1;
        }
    };
    let (start, step, end) = match nums.len() {
        1 => (1.0, 1.0, nums[0]),
        2 => (nums[0], 1.0, nums[1]),
        3 => (nums[0], nums[1], nums[2]),
        _ => {
            ewln(io.err, "seq: missing or extra operand");
            return 1;
        }
    };
    let mut x = start;
    let int = start.fract() == 0.0 && step.fract() == 0.0 && end.fract() == 0.0;
    let mut emit = |value| -> Result<(), i32> {
        if !context.system.charge_cpu(1) {
            return Err(context.system.stop_status());
        }
        let line = fmt_num(value, int);
        let projected = io.out.len().saturating_add(line.len()).saturating_add(1);
        if u64::try_from(projected).unwrap_or(u64::MAX) > context.system.output_remaining() {
            let excess = context.system.output_remaining().saturating_add(1);
            let _ = context.system.charge_output(excess);
            return Err(context.system.stop_status());
        }
        wln(io.out, &line);
        Ok(())
    };
    if step > 0.0 {
        while x <= end + 1e-9 {
            if let Err(status) = emit(x) {
                return status;
            }
            x += step;
        }
    } else if step < 0.0 {
        while x >= end - 1e-9 {
            if let Err(status) = emit(x) {
                return status;
            }
            x += step;
        }
    } else {
        ewln(io.err, "seq: zero increment");
        return 1;
    }
    0
}

fn fmt_num(x: f64, int: bool) -> String {
    if int {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

fn cmd_paste(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let mut delim = '\t';
    let mut files = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "-d" {
            delim = it.next().and_then(|s| s.chars().next()).unwrap_or('\t');
        } else if let Some(d) = a.strip_prefix("-d") {
            delim = d.chars().next().unwrap_or('\t');
        } else if a.starts_with('-') && a != "-" {
            ewln(io.err, &format!("paste: unimplemented option '{a}'"));
            return ShellPoll::Ready(2);
        } else {
            files.push(a.clone());
        }
    }
    if files.is_empty() || files.iter().any(|file| file == "-") {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    if files.is_empty() {
        io.out.extend_from_slice(&io.stdin);
        return ShellPoll::Ready(0);
    }
    let mut columns = Vec::new();
    let mut retained_bytes = 0usize;
    for file in &files {
        if file == "-" {
            retained_bytes = retained_bytes.saturating_add(io.stdin.len());
            columns.push(lines_of(&io.stdin));
        } else {
            let cwd = context.system.cwd().to_string();
            let maximum = usize::try_from(context.system.limits().memory)
                .unwrap_or(usize::MAX)
                .saturating_sub(retained_bytes);
            match context.system.read_file_limited(&cwd, file, maximum) {
                Ok(data) => {
                    retained_bytes = retained_bytes.saturating_add(data.len());
                    columns.push(lines_of(&data));
                }
                Err(error) => {
                    ewln(io.err, &format!("paste: {file}: {error}"));
                    return ShellPoll::Ready(1);
                }
            }
        }
    }
    let max = columns.iter().map(|c| c.len()).max().unwrap_or(0);
    for i in 0..max {
        let row: Vec<String> = columns
            .iter()
            .map(|c| c.get(i).cloned().unwrap_or_default())
            .collect();
        wln(io.out, &row.join(&delim.to_string()));
    }
    ShellPoll::Ready(0)
}

fn cmd_comm(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let (flags, ops, _l) = split_flags(args);
    if ops.len() < 2 {
        ewln(io.err, "comm: missing operand");
        return ShellPoll::Ready(1);
    }
    if uses_standard_input(&ops[..2]) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let mut inputs = Vec::new();
    for operand in &ops[..2] {
        let (bytes, errors) = read_inputs_system(context.system, &[*operand], &io.stdin);
        if let Some(error) = errors.first() {
            ewln(io.err, &format!("comm: {error}"));
            return ShellPoll::Ready(1);
        }
        inputs.push(lines_of(&bytes));
    }
    let [a, b] = <[Vec<String>; 2]>::try_from(inputs).expect("two comm inputs");
    let (s1, s2, s3) = (
        !flags.contains(&'1'),
        !flags.contains(&'2'),
        !flags.contains(&'3'),
    );
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        if i < a.len() && (j >= b.len() || a[i] < b[j]) {
            if s1 {
                wln(io.out, &a[i]);
            }
            i += 1;
        } else if j < b.len() && (i >= a.len() || b[j] < a[i]) {
            if s2 {
                wln(io.out, &format!("\t{}", b[j]));
            }
            j += 1;
        } else {
            if s3 {
                wln(io.out, &format!("\t\t{}", a[i]));
            }
            i += 1;
            j += 1;
        }
    }
    ShellPoll::Ready(0)
}

fn cmd_join(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let mut separator = None;
    let mut field1 = 0usize;
    let mut field2 = 0usize;
    let mut include_unpaired = [false, false];
    let mut files = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-t" => {
                index += 1;
                separator = args.get(index).and_then(|value| value.chars().next());
                if separator.is_none() {
                    ewln(io.err, "join: missing delimiter");
                    return ShellPoll::Ready(1);
                }
            }
            "-1" | "-2" => {
                let side = usize::from(args[index] == "-2");
                index += 1;
                let Some(value) = args
                    .get(index)
                    .and_then(|v| v.parse::<usize>().ok())
                    .and_then(|v| v.checked_sub(1))
                else {
                    ewln(io.err, "join: invalid field number");
                    return ShellPoll::Ready(1);
                };
                if side == 0 {
                    field1 = value;
                } else {
                    field2 = value;
                }
            }
            "-a" => {
                index += 1;
                match args.get(index).map(String::as_str) {
                    Some("1") => include_unpaired[0] = true,
                    Some("2") => include_unpaired[1] = true,
                    _ => {
                        ewln(io.err, "join: invalid file number");
                        return ShellPoll::Ready(1);
                    }
                }
            }
            value if value.starts_with('-') && value != "-" => {
                ewln(io.err, &format!("join: unsupported option '{value}'"));
                return ShellPoll::Ready(1);
            }
            _ => files.push(&args[index]),
        }
        index += 1;
    }
    if files.len() != 2 {
        ewln(io.err, "join: expected two files");
        return ShellPoll::Ready(1);
    }
    if files.iter().any(|file| file.as_str() == "-") {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let mut inputs = Vec::new();
    for file in files {
        let (data, errors) = read_inputs_system(context.system, &[file], &io.stdin);
        if let Some(error) = errors.first() {
            ewln(io.err, &format!("join: {error}"));
            return ShellPoll::Ready(1);
        }
        inputs.push(
            lines_of(&data)
                .into_iter()
                .map(|line| {
                    separator.map_or_else(
                        || line.split_whitespace().map(str::to_string).collect(),
                        |sep| line.split(sep).map(str::to_string).collect(),
                    )
                })
                .collect::<Vec<Vec<String>>>(),
        );
    }
    let output_separator = separator.unwrap_or(' ').to_string();
    let comparison_work = inputs[0].len().saturating_mul(inputs[1].len());
    let retained_memory = inputs
        .iter()
        .flatten()
        .flatten()
        .map(String::len)
        .sum::<usize>()
        .saturating_add(inputs[1].len());
    if !context
        .system
        .charge_cpu(u64::try_from(comparison_work).unwrap_or(u64::MAX))
    {
        return ShellPoll::Ready(context.system.stop_status());
    }
    let retained_memory = u64::try_from(retained_memory).unwrap_or(u64::MAX);
    if !context.system.reserve_memory(retained_memory) {
        return ShellPoll::Ready(context.system.stop_status());
    }
    let mut matched2 = vec![false; inputs[1].len()];
    for row1 in &inputs[0] {
        let key = row1.get(field1);
        let mut matched = false;
        for (right_index, row2) in inputs[1].iter().enumerate() {
            if key.is_some() && key == row2.get(field2) {
                matched = true;
                matched2[right_index] = true;
                let mut output = vec![key.cloned().unwrap_or_default()];
                output.extend(
                    row1.iter()
                        .enumerate()
                        .filter(|(i, _)| *i != field1)
                        .map(|(_, v)| v.clone()),
                );
                output.extend(
                    row2.iter()
                        .enumerate()
                        .filter(|(i, _)| *i != field2)
                        .map(|(_, v)| v.clone()),
                );
                wln(io.out, &output.join(&output_separator));
            }
        }
        if !matched && include_unpaired[0] {
            wln(io.out, &row1.join(&output_separator));
        }
    }
    if include_unpaired[1] {
        for (index, row) in inputs[1].iter().enumerate() {
            if !matched2[index] {
                wln(io.out, &row.join(&output_separator));
            }
        }
    }
    context.system.release_memory(retained_memory);
    ShellPoll::Ready(0)
}

fn cmd_split(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let mut line_count = 1000usize;
    let mut byte_count = None;
    let mut operands = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-l" => {
                index += 1;
                line_count = match args.get(index).and_then(|v| v.parse().ok()) {
                    Some(0) | None => {
                        ewln(io.err, "split: invalid line count");
                        return ShellPoll::Ready(1);
                    }
                    Some(v) => v,
                };
            }
            "-b" => {
                index += 1;
                byte_count = match args.get(index).and_then(|v| v.parse::<usize>().ok()) {
                    Some(0) | None => {
                        ewln(io.err, "split: invalid byte count");
                        return ShellPoll::Ready(1);
                    }
                    value => value,
                };
            }
            value if value.starts_with('-') && value != "-" => {
                ewln(io.err, &format!("split: unsupported option '{value}'"));
                return ShellPoll::Ready(1);
            }
            _ => operands.push(&args[index]),
        }
        index += 1;
    }
    if operands.len() > 2 {
        ewln(io.err, "split: extra operand");
        return ShellPoll::Ready(1);
    }
    if operands.first().is_none_or(|value| value.as_str() == "-") {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let data = match operands.first().map(|v| v.as_str()) {
        None | Some("-") => io.stdin.clone(),
        Some(file) => {
            let cwd = context.system.cwd().to_string();
            let maximum = usize::try_from(context.system.limits().memory).unwrap_or(usize::MAX);
            match context.system.read_file_limited(&cwd, file, maximum) {
                Ok(v) => v,
                Err(e) => {
                    ewln(io.err, &format!("split: {file}: {e}"));
                    return ShellPoll::Ready(1);
                }
            }
        }
    };
    let prefix = operands.get(1).map_or("x", |value| value.as_str());
    let data_len = data.len();
    if !context
        .system
        .charge_cpu(u64::try_from(data_len).unwrap_or(u64::MAX))
    {
        return ShellPoll::Ready(context.system.stop_status());
    }
    let chunk_memory = data_len.saturating_mul(2).saturating_add(
        data_len
            .min(26 * 26)
            .saturating_mul(std::mem::size_of::<Vec<u8>>()),
    );
    let chunk_memory = u64::try_from(chunk_memory).unwrap_or(u64::MAX);
    if !context.system.reserve_memory(chunk_memory) {
        return ShellPoll::Ready(context.system.stop_status());
    }
    let chunks = if let Some(bytes) = byte_count {
        data.chunks(bytes).map(<[u8]>::to_vec).collect::<Vec<_>>()
    } else {
        let mut chunks = Vec::new();
        let mut current = Vec::new();
        let mut lines = 0;
        for byte in data {
            current.push(byte);
            if byte == b'\n' {
                lines += 1;
            }
            if lines == line_count {
                chunks.push(std::mem::take(&mut current));
                lines = 0;
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        chunks
    };
    for (number, chunk) in chunks.iter().enumerate() {
        if number >= 26 * 26 {
            ewln(io.err, "split: output file suffixes exhausted");
            context.system.release_memory(chunk_memory);
            return ShellPoll::Ready(1);
        }
        let name = format!(
            "{prefix}{}{}",
            char::from(b'a' + (number / 26) as u8),
            char::from(b'a' + (number % 26) as u8)
        );
        let cwd = context.system.cwd().to_string();
        let mode = 0o666 & !u32::from(context.system.umask());
        if let Err(error) = context.system.write_file(&cwd, &name, chunk, mode) {
            ewln(io.err, &format!("split: {name}: {error}"));
            context.system.release_memory(chunk_memory);
            return ShellPoll::Ready(1);
        }
    }
    context.system.release_memory(chunk_memory);
    ShellPoll::Ready(0)
}

fn cmd_shuf(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let mut count = None;
    let mut files = Vec::new();
    let mut index = 0;
    while index < args.len() {
        if args[index] == "-n" || args[index] == "--head-count" {
            index += 1;
            count = args.get(index).and_then(|v| v.parse::<usize>().ok());
            if count.is_none() {
                ewln(io.err, "shuf: invalid line count");
                return ShellPoll::Ready(1);
            }
        } else if args[index].starts_with('-') && args[index] != "-" {
            ewln(
                io.err,
                &format!("shuf: unsupported option '{}'", args[index]),
            );
            return ShellPoll::Ready(1);
        } else {
            files.push(&args[index]);
        }
        index += 1;
    }
    if files.len() > 1 {
        ewln(io.err, "shuf: extra operand");
        return ShellPoll::Ready(1);
    }
    if uses_standard_input(&files) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("shuf: {error}"));
        return ShellPoll::Ready(1);
    }
    if !context
        .system
        .charge_cpu(u64::try_from(data.len()).unwrap_or(u64::MAX))
    {
        return ShellPoll::Ready(context.system.stop_status());
    }
    let mut lines = lines_of(&data);
    // A fixed generator makes simulations reproducible while retaining permutation semantics.
    let mut state = 0x9e37_79b9_u32;
    for end in (1..lines.len()).rev() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let other = state as usize % (end + 1);
        lines.swap(end, other);
    }
    let take = count.unwrap_or(lines.len()).min(lines.len());
    for line in &lines[..take] {
        wln(io.out, line);
    }
    ShellPoll::Ready(0)
}

fn cmd_tsort(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    if args.len() > 1 {
        ewln(io.err, "tsort: extra operand");
        return ShellPoll::Ready(1);
    }
    let files = args.iter().collect::<Vec<_>>();
    if uses_standard_input(&files) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("tsort: {error}"));
        return ShellPoll::Ready(1);
    }
    if !context
        .system
        .charge_cpu(u64::try_from(data.len()).unwrap_or(u64::MAX))
    {
        return ShellPoll::Ready(context.system.stop_status());
    }
    let words = String::from_utf8_lossy(&data)
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if words.len() % 2 != 0 {
        ewln(io.err, "tsort: input contains an odd number of tokens");
        return ShellPoll::Ready(1);
    }
    let mut edges = std::collections::BTreeMap::<String, std::collections::BTreeSet<String>>::new();
    let mut indegree = std::collections::BTreeMap::<String, usize>::new();
    for pair in words.chunks(2) {
        indegree.entry(pair[0].clone()).or_default();
        indegree.entry(pair[1].clone()).or_default();
        if pair[0] != pair[1]
            && edges
                .entry(pair[0].clone())
                .or_default()
                .insert(pair[1].clone())
        {
            *indegree.entry(pair[1].clone()).or_default() += 1;
        }
    }
    let mut ready = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(node, _)| node.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let mut emitted = 0;
    while let Some(node) = ready.pop_first() {
        wln(io.out, &node);
        emitted += 1;
        if let Some(next) = edges.get(&node) {
            for target in next {
                let degree = indegree.get_mut(target).expect("known node");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(target.clone());
                }
            }
        }
    }
    if emitted != indegree.len() {
        ewln(io.err, "tsort: input contains a loop");
        return ShellPoll::Ready(1);
    }
    ShellPoll::Ready(0)
}
