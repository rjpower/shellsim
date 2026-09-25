//! Argument-only `printf` formatting for shell builtins and native executable images.

use std::collections::HashMap;

use super::util::{unescape, w};
use super::{reg_system_costed, CommandSpec, Io, Trust};

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    reg_system_costed(m, "/usr/bin/printf", Trust::Real, 25, run);
}

fn run(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    if args.is_empty() {
        return 0;
    }
    w(io.out, &format_args(&args[0], &args[1..]));
    0
}

fn format_args(format: &str, args: &[String]) -> String {
    let format = unescape(format);
    let mut out = String::new();
    let mut argument = 0;
    let chars: Vec<char> = format.chars().collect();
    let mut i = 0;
    'formats: loop {
        let argument_at_start = argument;
        while i < chars.len() {
            if chars[i] != '%' {
                out.push(chars[i]);
                i += 1;
                continue;
            }
            if chars.get(i + 1) == Some(&'%') {
                out.push('%');
                i += 2;
                continue;
            }
            let spec_start = i;
            i += 1;
            while i < chars.len() && "-+ 0#".contains(chars[i]) {
                i += 1;
            }
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '*') {
                i += 1;
            }
            if i < chars.len() && chars[i] == '.' {
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let conv = chars.get(i).copied().unwrap_or('s');
            let spec: String = chars[spec_start..=i.min(chars.len() - 1)].iter().collect();
            i += 1;
            let arg = args.get(argument).cloned().unwrap_or_default();
            argument += 1;
            let (rendered, stop) = apply_conversion(&spec, conv, &arg);
            out.push_str(&rendered);
            if stop {
                break 'formats;
            }
        }
        if argument >= args.len() || argument == argument_at_start {
            break;
        }
        i = 0;
    }
    out
}

fn apply_conversion(spec: &str, conversion: char, arg: &str) -> (String, bool) {
    let options = spec.trim_start_matches('%');
    let width = options
        .trim_start_matches(['-', '+', ' ', '0', '#'])
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse::<usize>()
        .ok();
    let left = spec.contains('-');
    let zero = spec.starts_with("%0") || spec.starts_with("%-0");
    let positive_sign = if spec.contains('+') {
        "+"
    } else if spec.contains(' ') {
        " "
    } else {
        ""
    };
    let mut stop = false;
    let body = match conversion {
        'd' | 'i' => {
            let value = arg.trim().parse::<i64>().unwrap_or(0);
            if value >= 0 {
                format!("{positive_sign}{value}")
            } else {
                value.to_string()
            }
        }
        'x' => format!("{:x}", arg.trim().parse::<i64>().unwrap_or(0)),
        'X' => format!("{:X}", arg.trim().parse::<i64>().unwrap_or(0)),
        'o' => format!("{:o}", arg.trim().parse::<i64>().unwrap_or(0)),
        'f' | 'F' => {
            let precision = precision(spec).unwrap_or(6);
            format!("{:.*}", precision, arg.trim().parse::<f64>().unwrap_or(0.0))
        }
        's' => precision(spec)
            .map(|n| arg.chars().take(n).collect())
            .unwrap_or_else(|| arg.to_string()),
        'c' => arg
            .chars()
            .next()
            .map(|c| c.to_string())
            .unwrap_or_default(),
        'b' => {
            let (value, encountered_stop) = unescape_argument(arg);
            stop = encountered_stop;
            value
        }
        _ => arg.to_string(),
    };
    let rendered = match width.filter(|width| body.len() < *width) {
        Some(width) => {
            let pad = if zero && !left { "0" } else { " " }.repeat(width - body.len());
            if left {
                format!("{body}{pad}")
            } else if zero
                && (body.starts_with('-') || body.starts_with('+') || body.starts_with(' '))
            {
                format!("{}{pad}{}", &body[..1], &body[1..])
            } else {
                format!("{pad}{body}")
            }
        }
        None => body,
    };
    (rendered, stop)
}

fn unescape_argument(argument: &str) -> (String, bool) {
    let Some(index) = argument.find("\\c") else {
        return (unescape(argument), false);
    };
    (unescape(&argument[..index]), true)
}

fn precision(spec: &str) -> Option<usize> {
    spec.split('.')
        .nth(1)
        .and_then(|p| p.trim_end_matches(|c: char| c.is_alphabetic()).parse().ok())
}
