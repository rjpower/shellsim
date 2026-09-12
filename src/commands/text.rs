//! Text processing: output (echo/printf/cat/tac/tee/yes), windowing (head/tail), counting
//! and reshaping (wc/sort/uniq/cut/tr/rev/nl/seq/paste/comm/diff/cmp), pattern tools
//! (grep/sed), the fold/fmt passthroughs, xargs, and the small arithmetic helpers expr/bc.

use std::collections::HashMap;

use crate::commands::util::{ewln, lines_of, read_inputs, split_flags, w, wln};
use crate::commands::{ChildCommand, CommandContext, CommandPoll, CommandSpec, Io, Trust};
use crate::interp::Interp;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::{reg, reg_buffered_resumable};
    reg(m, &["cat"], Trust::Real, cmd_cat);
    reg(m, &["tac"], Trust::Real, cmd_tac);
    reg(m, &["tee"], Trust::Real, cmd_tee);
    reg(m, &["yes"], Trust::Real, cmd_yes);
    reg(m, &["head"], Trust::Real, cmd_head);
    reg(m, &["tail"], Trust::Real, cmd_tail);
    reg(m, &["wc"], Trust::Real, cmd_wc);
    reg(m, &["uniq"], Trust::Real, cmd_uniq);
    reg(m, &["cut"], Trust::Real, cmd_cut);
    reg(m, &["tr"], Trust::Real, cmd_tr);
    reg(m, &["rev"], Trust::Real, cmd_rev);
    reg(m, &["grep"], Trust::Partial, cmd_grep);
    reg(m, &["egrep"], Trust::Partial, cmd_egrep);
    reg(m, &["fgrep"], Trust::Partial, cmd_fgrep);
    reg(m, &["sed"], Trust::Partial, cmd_sed);
    reg(m, &["nl"], Trust::Real, cmd_nl);
    reg(m, &["seq"], Trust::Real, cmd_seq);
    reg(m, &["paste"], Trust::Real, cmd_paste);
    reg(m, &["head_tail_placeholder"], Trust::Real, |_, _, _| 0);
    reg(
        m,
        &["fold", "fmt", "expand", "unexpand", "column", "pr"],
        Trust::Partial,
        cmd_passthrough,
    );
    reg_buffered_resumable(m, &["xargs"], Trust::Real, cmd_xargs, start_xargs);
    reg(m, &["comm"], Trust::Real, cmd_comm);
    reg(m, &["diff"], Trust::Partial, cmd_diff);
    reg(m, &["cmp"], Trust::Real, cmd_cmp);
    reg(m, &["expr"], Trust::Real, cmd_expr);
    reg(m, &["bc"], Trust::Real, cmd_bc);
    reg(m, &["factor"], Trust::Real, |_, _, _| 0);
}

fn cmd_cat(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _long) = split_flags(args);
    let number = flags.contains(&'n');
    let (data, errors) = read_inputs(interp, &ops, &io.stdin);
    if number {
        for (i, line) in String::from_utf8_lossy(&data).lines().enumerate() {
            wln(io.out, &format!("{:6}\t{}", i + 1, line));
        }
    } else {
        io.out.extend_from_slice(&data);
    }
    for e in &errors {
        ewln(io.err, &format!("cat: {e}"));
    }
    if errors.is_empty() {
        0
    } else {
        1
    }
}

fn cmd_tac(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    let (data, _e) = read_inputs(interp, &ops, &io.stdin);
    let lines = lines_of(&data);
    for l in lines.iter().rev() {
        wln(io.out, l);
    }
    0
}

fn cmd_tee(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    let append = flags.contains(&'a');
    let cwd = interp.cwd.clone();
    for f in &ops {
        if append {
            let _ = interp.vfs.append(&cwd, f, &io.stdin, 0o644);
        } else {
            let _ = interp.vfs.write(&cwd, f, &io.stdin, 0o644);
        }
    }
    io.out.extend_from_slice(&io.stdin);
    0
}

fn cmd_yes(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let s = if args.is_empty() {
        "y".to_string()
    } else {
        args.join(" ")
    };
    for _ in 0..1000 {
        wln(io.out, &s);
    }
    0
}

fn cmd_head(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut n = 10usize;
    let mut bytes: Option<usize> = None;
    let mut files = Vec::new();
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        if a == "-n" {
            n = it
                .next()
                .and_then(|s| s.trim_start_matches('-').parse().ok())
                .unwrap_or(10);
        } else if let Some(v) = a.strip_prefix("-n") {
            n = v.trim_start_matches('-').parse().unwrap_or(10);
        } else if a == "-c" {
            bytes = it.next().and_then(|s| s.parse().ok());
        } else if let Some(v) = a.strip_prefix("-c") {
            bytes = v.parse().ok();
        } else if a.starts_with('-') && a.len() > 1 && a[1..].chars().all(|c| c.is_ascii_digit()) {
            n = a[1..].parse().unwrap_or(10);
        } else if !a.starts_with('-') || a == "-" {
            files.push(a);
        }
    }
    let (data, _e) = read_inputs(interp, &files, &io.stdin);
    if let Some(b) = bytes {
        io.out.extend_from_slice(&data[..b.min(data.len())]);
    } else {
        for (i, line) in String::from_utf8_lossy(&data).lines().enumerate() {
            if i >= n {
                break;
            }
            wln(io.out, line);
        }
    }
    0
}

fn cmd_tail(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut n = 10usize;
    let mut from_start = false;
    let mut files = Vec::new();
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        if a == "-n" {
            let v = it.next().cloned().unwrap_or_default();
            from_start = v.starts_with('+');
            n = v
                .trim_start_matches('+')
                .trim_start_matches('-')
                .parse()
                .unwrap_or(10);
        } else if let Some(v) = a.strip_prefix("-n") {
            from_start = v.starts_with('+');
            n = v
                .trim_start_matches('+')
                .trim_start_matches('-')
                .parse()
                .unwrap_or(10);
        } else if a == "-f" || a == "-F" {
            // no follow in sim
        } else if !a.starts_with('-') || a == "-" {
            files.push(a);
        }
    }
    let (data, _e) = read_inputs(interp, &files, &io.stdin);
    let lines: Vec<&str> = String::from_utf8_lossy(&data)
        .lines()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .leak()
        .iter()
        .map(|s| s.as_str())
        .collect();
    if from_start {
        for l in lines.iter().skip(n.saturating_sub(1)) {
            wln(io.out, l);
        }
    } else {
        let start = lines.len().saturating_sub(n);
        for l in &lines[start..] {
            wln(io.out, l);
        }
    }
    0
}

fn cmd_wc(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    let (cl, cw, cc) = (
        flags.contains(&'l'),
        flags.contains(&'w'),
        flags.contains(&'c') || flags.contains(&'m'),
    );
    let none = !cl && !cw && !cc;
    let print_one = |data: &[u8], out: &mut Vec<u8>, label: &str| {
        let s = String::from_utf8_lossy(data);
        let lines = s.matches('\n').count();
        let words = s.split_whitespace().count();
        let chars = data.len();
        let mut parts = Vec::new();
        if cl || none {
            parts.push(format!("{:>7}", lines));
        }
        if cw || none {
            parts.push(format!("{:>7}", words));
        }
        if cc || none {
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
        print_one(&io.stdin, io.out, "");
    } else {
        for f in &ops {
            match interp.vfs.read(&interp.cwd, f) {
                Ok(d) => print_one(&d, io.out, f),
                Err(_) => return 1,
            }
        }
    }
    0
}

fn cmd_uniq(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    let count = flags.contains(&'c');
    let only_dup = flags.contains(&'d');
    let only_uniq = flags.contains(&'u');
    let (data, _e) = read_inputs(interp, &ops, &io.stdin);
    let lines: Vec<String> = String::from_utf8_lossy(&data)
        .lines()
        .map(|s| s.to_string())
        .collect();
    let mut i = 0;
    while i < lines.len() {
        let mut j = i + 1;
        while j < lines.len() && lines[j] == lines[i] {
            j += 1;
        }
        let n = j - i;
        let show = (!only_dup && !only_uniq) || (only_dup && n > 1) || (only_uniq && n == 1);
        if show {
            if count {
                wln(io.out, &format!("{:>7} {}", n, lines[i]));
            } else {
                wln(io.out, &lines[i]);
            }
        }
        i = j;
    }
    0
}

fn cmd_cut(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
        } else if !a.starts_with('-') {
            files.push(a);
        }
    }
    let (data, _e) = read_inputs(interp, &files, &io.stdin);
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
    0
}

fn cmd_tr(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    let delete = flags.contains(&'d');
    let squeeze = flags.contains(&'s');
    let complement = flags.contains(&'c');
    let set1 = ops.first().map(|s| expand_tr_set(s)).unwrap_or_default();
    let set2 = ops.get(1).map(|s| expand_tr_set(s)).unwrap_or_default();
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
    0
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
        if i + 2 < chars.len() && chars[i + 1] == '-' {
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

fn cmd_rev(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    let (data, _e) = read_inputs(interp, &ops, &io.stdin);
    for line in String::from_utf8_lossy(&data).lines() {
        wln(io.out, &line.chars().rev().collect::<String>());
    }
    0
}

fn cmd_nl(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    let (data, _e) = read_inputs(interp, &ops, &io.stdin);
    let mut n = 1;
    for line in String::from_utf8_lossy(&data).lines() {
        if line.is_empty() {
            wln(io.out, "");
        } else {
            wln(io.out, &format!("{:>6}\t{}", n, line));
            n += 1;
        }
    }
    0
}

fn cmd_seq(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let nums: Vec<f64> = args.iter().filter_map(|a| a.parse().ok()).collect();
    let (start, step, end) = match nums.len() {
        1 => (1.0, 1.0, nums[0]),
        2 => (nums[0], 1.0, nums[1]),
        3 => (nums[0], nums[1], nums[2]),
        _ => return 1,
    };
    let mut x = start;
    let int = start.fract() == 0.0 && step.fract() == 0.0 && end.fract() == 0.0;
    if step > 0.0 {
        while x <= end + 1e-9 {
            wln(io.out, &fmt_num(x, int));
            x += step;
        }
    } else if step < 0.0 {
        while x >= end - 1e-9 {
            wln(io.out, &fmt_num(x, int));
            x += step;
        }
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

fn cmd_paste(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut delim = '\t';
    let mut files = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "-d" {
            delim = it.next().and_then(|s| s.chars().next()).unwrap_or('\t');
        } else if let Some(d) = a.strip_prefix("-d") {
            delim = d.chars().next().unwrap_or('\t');
        } else {
            files.push(a.clone());
        }
    }
    let columns: Vec<Vec<String>> = files
        .iter()
        .map(|f| {
            if f == "-" {
                lines_of(&io.stdin)
            } else {
                interp
                    .vfs
                    .read(&interp.cwd, f)
                    .map(|d| lines_of(&d))
                    .unwrap_or_default()
            }
        })
        .collect();
    let max = columns.iter().map(|c| c.len()).max().unwrap_or(0);
    for i in 0..max {
        let row: Vec<String> = columns
            .iter()
            .map(|c| c.get(i).cloned().unwrap_or_default())
            .collect();
        wln(io.out, &row.join(&delim.to_string()));
    }
    0
}

fn cmd_passthrough(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    let (data, _e) = read_inputs(interp, &ops, &io.stdin);
    io.out.extend_from_slice(&data);
    0
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
            "-0" => {
                nul_delimited = true;
                i += 1;
            }
            "-r" | "--no-run-if-empty" => i += 1,
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
    let cmd: Vec<String> = args[i..].to_vec();
    if cmd.is_empty() {
        ewln(err, "xargs: missing command");
        return Err(1);
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

fn cmd_comm(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    if ops.len() < 2 {
        ewln(io.err, "comm: missing operand");
        return 1;
    }
    let a = interp
        .vfs
        .read(&interp.cwd, ops[0])
        .map(|d| lines_of(&d))
        .unwrap_or_default();
    let b = interp
        .vfs
        .read(&interp.cwd, ops[1])
        .map(|d| lines_of(&d))
        .unwrap_or_default();
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
    0
}

fn cmd_diff(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    if ops.len() < 2 {
        ewln(io.err, "diff: missing operand");
        return 2;
    }
    let a = interp
        .vfs
        .read_string(&interp.cwd, ops[0])
        .unwrap_or_default();
    let b = interp
        .vfs
        .read_string(&interp.cwd, ops[1])
        .unwrap_or_default();
    if a == b {
        0
    } else {
        // minimal unified-ish output (not a real LCS diff)
        let al: Vec<&str> = a.lines().collect();
        let bl: Vec<&str> = b.lines().collect();
        for (i, line) in al.iter().enumerate() {
            if bl.get(i) != Some(line) {
                wln(io.out, &format!("< {line}"));
            }
        }
        wln(io.out, "---");
        for (i, line) in bl.iter().enumerate() {
            if al.get(i) != Some(line) {
                wln(io.out, &format!("> {line}"));
            }
        }
        1
    }
}

fn cmd_cmp(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    if ops.len() < 2 {
        return 2;
    }
    let a = interp.vfs.read(&interp.cwd, ops[0]).unwrap_or_default();
    let b = interp.vfs.read(&interp.cwd, ops[1]).unwrap_or_default();
    if a == b {
        0
    } else {
        ewln(io.err, &format!("{} {} differ", ops[0], ops[1]));
        1
    }
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
    let mut ignore_case = false;
    let mut invert = false;
    let mut count = false;
    let mut line_num = false;
    let mut files_with = false;
    let mut only_match = false;
    let mut recursive = false;
    let mut extended = cmd == "egrep";
    let mut fixed = cmd == "fgrep";
    let mut word = false;
    let mut quiet = false;
    let mut after = 0usize;
    let mut pattern: Option<String> = None;
    let mut files = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a.starts_with('-') && a.len() > 1 && a != "-" {
            if let Some(p) = a.strip_prefix("-e") {
                if p.is_empty() {
                    pattern = it.next().cloned();
                } else {
                    pattern = Some(p.to_string());
                }
                continue;
            }
            if let Some(n) = a.strip_prefix("-A") {
                after = if n.is_empty() {
                    it.next().and_then(|s| s.parse().ok()).unwrap_or(0)
                } else {
                    n.parse().unwrap_or(0)
                };
                continue;
            }
            for c in a[1..].chars() {
                match c {
                    'i' => ignore_case = true,
                    'v' => invert = true,
                    'c' => count = true,
                    'n' => line_num = true,
                    'l' => files_with = true,
                    'o' => only_match = true,
                    'r' | 'R' => recursive = true,
                    'E' => extended = true,
                    'F' => fixed = true,
                    'w' => word = true,
                    'q' => quiet = true,
                    'h' | 's' | 'a' => {}
                    _ => {}
                }
            }
        } else if pattern.is_none() {
            pattern = Some(a.clone());
        } else {
            files.push(a.clone());
        }
    }
    let _ = extended;
    let Some(pat) = pattern else {
        ewln(io.err, "grep: no pattern");
        return 2;
    };
    let mut pat_re = if fixed {
        regex::escape(&pat)
    } else {
        pat.clone()
    };
    if word {
        pat_re = format!(r"\b(?:{pat_re})\b");
    }
    let re = match regex::RegexBuilder::new(&pat_re)
        .case_insensitive(ignore_case)
        .build()
    {
        Ok(r) => r,
        Err(_) => {
            // fall back to fixed-string
            regex::RegexBuilder::new(&regex::escape(&pat))
                .case_insensitive(ignore_case)
                .build()
                .unwrap()
        }
    };

    // gather (label, data)
    let mut inputs: Vec<(String, Vec<u8>)> = Vec::new();
    if files.is_empty() {
        inputs.push((String::new(), std::mem::take(&mut io.stdin)));
    } else if recursive {
        for f in &files {
            let abs = crate::vfs::resolve_against(&interp.cwd, f);
            for p in interp.vfs.walk(&abs) {
                if interp.vfs.is_file("/", &p) {
                    inputs.push((p.clone(), interp.vfs.read("/", &p).unwrap_or_default()));
                }
            }
        }
    } else {
        for f in &files {
            match interp.vfs.read(&interp.cwd, f) {
                Ok(d) => inputs.push((f.clone(), d)),
                Err(e) => ewln(io.err, &format!("grep: {e}")),
            }
        }
    }
    let multi = inputs.len() > 1 || recursive;
    let mut total_matches = 0;
    for (label, data) in &inputs {
        let mut file_count = 0;
        let mut matched_file = false;
        for (lineno, line) in String::from_utf8_lossy(data).lines().enumerate() {
            let is_match = re.is_match(line) ^ invert;
            if is_match {
                matched_file = true;
                file_count += 1;
                total_matches += 1;
                if quiet || count || files_with {
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
                let _ = after;
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
        }
    }
    if total_matches > 0 {
        0
    } else {
        1
    }
}

fn cmd_sed(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut in_place = false;
    let mut quiet = false;
    let mut scripts: Vec<String> = Vec::new();
    let mut extended = false;
    let mut files = Vec::new();
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        if a == "-i" || a.starts_with("-i") {
            in_place = true;
            if a.len() > 2 { /* suffix ignored */ }
        } else if a == "-n" {
            quiet = true;
        } else if a == "-r" || a == "-E" {
            extended = true;
        } else if a == "-e" {
            if let Some(s) = it.next() {
                scripts.push(s.clone());
            }
        } else if let Some(s) = a.strip_prefix("-e") {
            scripts.push(s.to_string());
        } else if a == "--" {
            continue;
        } else if scripts.is_empty() && !a.starts_with('-') {
            scripts.push(a.clone());
        } else {
            files.push(a.clone());
        }
    }
    let _ = extended;
    let commands: Vec<SedCmd> = scripts.iter().flat_map(|s| parse_sed_script(s)).collect();

    let process = |text: &str| -> String {
        let mut result = String::new();
        for line in text.split_inclusive('\n') {
            let had_nl = line.ends_with('\n');
            let mut content = line.trim_end_matches('\n').to_string();
            let mut deleted = false;
            let mut printed_extra = Vec::new();
            for cmd in &commands {
                match cmd {
                    SedCmd::Subst {
                        re,
                        rep,
                        global,
                        nth,
                        print,
                        ignore,
                    } => {
                        let _ = ignore;
                        content = sed_subst(re, rep, &content, *global, *nth);
                        if *print {
                            printed_extra.push(content.clone());
                        }
                    }
                    SedCmd::Delete => {
                        deleted = true;
                    }
                    SedCmd::Print => {
                        printed_extra.push(content.clone());
                    }
                }
            }
            if !quiet && !deleted {
                result.push_str(&content);
                if had_nl {
                    result.push('\n');
                }
            }
            for p in printed_extra {
                result.push_str(&p);
                result.push('\n');
            }
        }
        result
    };

    if in_place && !files.is_empty() {
        let cwd = interp.cwd.clone();
        for f in &files {
            match interp.vfs.read_string(&cwd, f) {
                Ok(text) => {
                    let new = process(&text);
                    let _ = interp.vfs.write(&cwd, f, new.as_bytes(), 0o644);
                }
                Err(e) => {
                    ewln(io.err, &format!("sed: can't read {f}: {e}"));
                    return 1;
                }
            }
        }
        0
    } else {
        let (data, _e) = read_inputs(interp, &files.iter().collect::<Vec<_>>(), &io.stdin);
        let text = String::from_utf8_lossy(&data);
        w(io.out, &process(&text));
        0
    }
}

enum SedCmd {
    Subst {
        re: regex::Regex,
        rep: String,
        global: bool,
        nth: usize,
        print: bool,
        ignore: bool,
    },
    Delete,
    Print,
}

fn parse_sed_script(s: &str) -> Vec<SedCmd> {
    let mut cmds = Vec::new();
    for part in s.split(';') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        // strip a leading line address like `3` or `/re/` (best-effort: ignore numeric/`$`)
        let body = part
            .trim_start_matches(|c: char| c.is_ascii_digit() || c == '$' || c == ',' || c == ' ');
        if let Some(rest) = body.strip_prefix('s') {
            if let Some(cmd) = parse_subst(rest) {
                cmds.push(cmd);
            }
        } else if body == "d" {
            cmds.push(SedCmd::Delete);
        } else if body == "p" {
            cmds.push(SedCmd::Print);
        }
    }
    cmds
}

fn parse_subst(rest: &str) -> Option<SedCmd> {
    let delim = rest.chars().next()?;
    let chars: Vec<char> = rest.chars().collect();
    let mut i = 1;
    let mut fields = [String::new(), String::new(), String::new()];
    let mut fi = 0;
    while i < chars.len() && fi < 3 {
        let c = chars[i];
        if c == '\\' && i + 1 < chars.len() {
            // keep escapes; but \<delim> becomes literal delim
            if chars[i + 1] == delim {
                fields[fi].push(delim);
            } else {
                fields[fi].push('\\');
                fields[fi].push(chars[i + 1]);
            }
            i += 2;
            continue;
        }
        if c == delim {
            fi += 1;
            i += 1;
            continue;
        }
        fields[fi].push(c);
        i += 1;
    }
    let (pat, rep, flags) = (&fields[0], &fields[1], &fields[2]);
    let global = flags.contains('g');
    let ignore = flags.contains('i') || flags.contains('I');
    let print = flags.contains('p');
    let nth: usize = flags
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap_or(0);
    let re = regex::RegexBuilder::new(pat)
        .case_insensitive(ignore)
        .build()
        .ok()?;
    // convert sed replacement backrefs \1 -> ${1}
    let rep = convert_sed_replacement(rep);
    Some(SedCmd::Subst {
        re,
        rep,
        global,
        nth,
        print,
        ignore,
    })
}

fn convert_sed_replacement(rep: &str) -> String {
    let mut out = String::new();
    let chars: Vec<char> = rep.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            out.push_str(&format!("${{{}}}", chars[i + 1]));
            i += 2;
        } else if chars[i] == '&' {
            out.push_str("${0}");
            i += 1;
        } else if chars[i] == '$' {
            out.push_str("$$");
            i += 1;
        } else if chars[i] == '\\' && i + 1 < chars.len() {
            match chars[i + 1] {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                c => out.push(c),
            }
            i += 2;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn sed_subst(re: &regex::Regex, rep: &str, text: &str, global: bool, nth: usize) -> String {
    if global && nth == 0 {
        re.replace_all(text, rep).into_owned()
    } else if nth > 0 {
        let mut count = 0;
        re.replace_all(text, |caps: &regex::Captures| {
            count += 1;
            if count == nth || (global && count >= nth) {
                expand_caps(rep, caps)
            } else {
                caps[0].to_string()
            }
        })
        .into_owned()
    } else {
        re.replace(text, rep).into_owned()
    }
}

fn expand_caps(rep: &str, caps: &regex::Captures) -> String {
    let mut out = String::new();
    caps.expand(rep, &mut out);
    out
}

fn cmd_expr(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    // minimal: arithmetic and string length
    if args.len() == 2 && args[0] == "length" {
        wln(io.out, &args[1].chars().count().to_string());
        return 0;
    }
    let joined = args.join(" ");
    // try arithmetic
    let mut i = Interp::new();
    let v = crate::expand::eval_arith(&mut i, &joined);
    wln(io.out, &v.to_string());
    if v == 0 {
        1
    } else {
        0
    }
}

fn cmd_bc(interp: &mut CommandContext<'_>, _args: &[String], io: &mut Io) -> i32 {
    for line in String::from_utf8_lossy(&io.stdin).lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v = crate::expand::eval_arith(interp, line);
        wln(io.out, &v.to_string());
    }
    0
}
