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
    let mut reserved = 0_u64;
    let system = &mut *context.system;
    let formatted = format_args(&args[0], &args[1..], &mut |bytes| {
        if system.reserve_memory(bytes) {
            reserved = reserved.saturating_add(bytes);
            true
        } else {
            false
        }
    });
    let status = match formatted {
        Some(text) => {
            w(io.out, &text);
            0
        }
        None => context.system.stop_status(),
    };
    context.system.release_memory(reserved);
    status
}

/// A parsed `%` conversion: flags, field width, precision, and conversion character.
struct Spec {
    left: bool,
    zero: bool,
    positive_sign: &'static str,
    width: Option<usize>,
    precision: Option<usize>,
    conversion: char,
}

/// Format `args` through `format`, reusing the format until the arguments run out.
///
/// `reserve` is asked for each conversion's output size before that output is built, since a
/// width or precision can make one conversion far larger than its inputs. Returns `None` when a
/// reservation fails.
fn format_args(
    format: &str,
    args: &[String],
    reserve: &mut dyn FnMut(u64) -> bool,
) -> Option<String> {
    let format = unescape(format);
    let mut out = String::new();
    let mut argument = 0;
    let chars: Vec<char> = format.chars().collect();
    let mut i = 0;
    let next_argument = |argument: &mut usize| {
        let value = args.get(*argument).cloned().unwrap_or_default();
        *argument += 1;
        value
    };
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
            i += 1;
            let mut spec = Spec {
                left: false,
                zero: false,
                positive_sign: "",
                width: None,
                precision: None,
                conversion: 's',
            };
            while i < chars.len() && "-+ 0#".contains(chars[i]) {
                match chars[i] {
                    '-' => spec.left = true,
                    '0' => spec.zero = true,
                    '+' => spec.positive_sign = "+",
                    ' ' if spec.positive_sign.is_empty() => spec.positive_sign = " ",
                    _ => {}
                }
                i += 1;
            }
            // `*` takes the width from the next argument; a negative width left-justifies.
            if chars.get(i) == Some(&'*') {
                i += 1;
                let width = next_argument(&mut argument)
                    .trim()
                    .parse::<i64>()
                    .unwrap_or(0);
                spec.left |= width < 0;
                spec.width = Some(width.unsigned_abs() as usize);
            } else {
                spec.width = digits(&chars, &mut i);
            }
            if chars.get(i) == Some(&'.') {
                i += 1;
                spec.precision = if chars.get(i) == Some(&'*') {
                    i += 1;
                    // A negative `*` precision is treated as omitted.
                    usize::try_from(
                        next_argument(&mut argument)
                            .trim()
                            .parse::<i64>()
                            .unwrap_or(0),
                    )
                    .ok()
                } else {
                    Some(digits(&chars, &mut i).unwrap_or(0))
                };
            }
            spec.conversion = chars.get(i).copied().unwrap_or('s');
            i += 1;
            let arg = next_argument(&mut argument);
            let (rendered, stop) = apply_conversion(&spec, &arg, reserve)?;
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
    Some(out)
}

/// Parse a run of decimal digits at `chars[*i]`, saturating rather than overflowing.
fn digits(chars: &[char], i: &mut usize) -> Option<usize> {
    let start = *i;
    let mut value = 0_usize;
    while let Some(digit) = chars.get(*i).and_then(|c| c.to_digit(10)) {
        value = value.saturating_mul(10).saturating_add(digit as usize);
        *i += 1;
    }
    (*i > start).then_some(value)
}

/// Upper bound on the digits before the point of a formatted `f64`.
const MAX_F64_INTEGER_DIGITS: u64 = 310;

fn apply_conversion(
    spec: &Spec,
    arg: &str,
    reserve: &mut dyn FnMut(u64) -> bool,
) -> Option<(String, bool)> {
    let mut stop = false;
    let body = match spec.conversion {
        'd' | 'i' => {
            let value = arg.trim().parse::<i64>().unwrap_or(0);
            if value >= 0 {
                format!("{}{value}", spec.positive_sign)
            } else {
                value.to_string()
            }
        }
        'x' => format!("{:x}", arg.trim().parse::<i64>().unwrap_or(0)),
        'X' => format!("{:X}", arg.trim().parse::<i64>().unwrap_or(0)),
        'o' => format!("{:o}", arg.trim().parse::<i64>().unwrap_or(0)),
        'f' | 'F' => {
            let precision = spec.precision.unwrap_or(6);
            if !reserve((precision as u64).saturating_add(MAX_F64_INTEGER_DIGITS)) {
                return None;
            }
            format!("{:.*}", precision, arg.trim().parse::<f64>().unwrap_or(0.0))
        }
        's' => spec
            .precision
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
    let rendered = match spec.width.filter(|width| body.len() < *width) {
        Some(width) => {
            if !reserve(width as u64) {
                return None;
            }
            let zero = spec.zero && !spec.left;
            let pad = if zero { "0" } else { " " }.repeat(width - body.len());
            if spec.left {
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
    Some((rendered, stop))
}

fn unescape_argument(argument: &str) -> (String, bool) {
    let Some(index) = argument.find("\\c") else {
        return (unescape(argument), false);
    };
    (unescape(&argument[..index]), true)
}
