//! A deliberately small, deterministic awk for the record-processing forms common in shell
//! tasks. It supports BEGIN/END rules, regex and comparison patterns, fields, variables,
//! assignments, print/printf, next, and a handful of scalar functions.

use std::collections::HashMap;

use crate::commands::util::{ewln, unescape};
use crate::commands::{reg_costed, CommandContext, CommandSpec, Io, Trust};

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    reg_costed(
        m,
        &["awk", "gawk", "mawk", "nawk"],
        Trust::Partial,
        150,
        24 * 1024,
        run,
    );
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Begin,
    Record,
    End,
}

struct Rule {
    phase: Phase,
    pattern: String,
    actions: String,
}

#[derive(Default)]
struct State {
    vars: HashMap<String, String>,
    record: String,
    fields: Vec<String>,
    nr: usize,
    fnr: usize,
    filename: String,
    fs: String,
    ofs: String,
    ors: String,
}

enum Control {
    Continue,
    Next,
    Exit(i32),
    Unsupported(String),
}

fn run(env: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut fs = " ".to_string();
    let mut vars = HashMap::new();
    let mut program = None;
    let mut files = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-F" => {
                i += 1;
                fs = args.get(i).cloned().unwrap_or_else(|| " ".to_string());
            }
            "-v" => {
                i += 1;
                if let Some((name, value)) = args.get(i).and_then(|arg| arg.split_once('=')) {
                    vars.insert(name.to_string(), value.to_string());
                }
            }
            "-f" => {
                i += 1;
                let Some(path) = args.get(i) else {
                    ewln(io.err, "awk: option -f requires an argument");
                    return 2;
                };
                match env.vfs.read(&env.cwd, path) {
                    Ok(data) => program = Some(String::from_utf8_lossy(&data).into_owned()),
                    Err(error) => {
                        ewln(io.err, &format!("awk: {path}: {error}"));
                        return 2;
                    }
                }
            }
            arg if arg.starts_with("-F") && arg.len() > 2 => fs = arg[2..].to_string(),
            arg if program.is_none() && !arg.starts_with('-') => program = Some(arg.to_string()),
            arg if program.is_some() => {
                if let Some((name, value)) = arg.split_once('=') {
                    if valid_name(name) {
                        vars.insert(name.to_string(), value.to_string());
                    } else {
                        files.push(arg.to_string());
                    }
                } else {
                    files.push(arg.to_string());
                }
            }
            arg => {
                ewln(io.err, &format!("awk: unsupported option: {arg}"));
                return 2;
            }
        }
        i += 1;
    }

    let Some(program) = program else {
        ewln(io.err, "awk: missing program");
        return 2;
    };
    let rules = parse_rules(&program);
    if rules.is_empty() {
        ewln(io.err, "awk: unsupported or empty program");
        return 2;
    }
    let mut state = State {
        vars,
        fs,
        ofs: " ".to_string(),
        ors: "\n".to_string(),
        ..State::default()
    };

    if let Some(status) = run_phase(&rules, Phase::Begin, &mut state, io) {
        return status;
    }

    let mut inputs: Vec<(String, Vec<u8>)> = Vec::new();
    if files.is_empty() {
        inputs.push(("-".to_string(), io.stdin.clone()));
    } else {
        for file in files {
            if file == "-" {
                inputs.push((file, io.stdin.clone()));
            } else {
                match env.vfs.read(&env.cwd, &file) {
                    Ok(data) => inputs.push((file, data)),
                    Err(error) => ewln(io.err, &format!("awk: {file}: {error}")),
                }
            }
        }
    }
    let input_bytes = inputs
        .iter()
        .map(|(_, data)| data.len() as u64)
        .sum::<u64>();
    if !env.reserve_memory(input_bytes.saturating_mul(2)) || !env.charge_cpu(input_bytes) {
        return 137;
    }

    let mut exit_status = None;
    'files: for (filename, data) in inputs {
        state.filename = filename;
        state.fnr = 0;
        for line in String::from_utf8_lossy(&data).lines() {
            state.nr += 1;
            state.fnr += 1;
            state.record = line.to_string();
            state.fields = split_fields(line, &state.fs);
            for rule in rules.iter().filter(|rule| rule.phase == Phase::Record) {
                if !eval_condition(&rule.pattern, &state) {
                    continue;
                }
                match run_actions(&rule.actions, &mut state, io) {
                    Control::Continue => {}
                    Control::Next => break,
                    Control::Exit(status) => {
                        exit_status = Some(status);
                        break 'files;
                    }
                    Control::Unsupported(action) => {
                        env.note_unsupported(&format!("awk:{action}"));
                        ewln(io.err, &format!("awk: unsupported action: {action}"));
                        return 2;
                    }
                }
            }
        }
    }

    if let Some(status) = run_phase(&rules, Phase::End, &mut state, io) {
        return status;
    }
    exit_status.unwrap_or(0)
}

fn run_phase(rules: &[Rule], phase: Phase, state: &mut State, io: &mut Io) -> Option<i32> {
    for rule in rules.iter().filter(|rule| rule.phase == phase) {
        match run_actions(&rule.actions, state, io) {
            Control::Continue | Control::Next => {}
            Control::Exit(status) => return Some(status),
            Control::Unsupported(action) => {
                ewln(io.err, &format!("awk: unsupported action: {action}"));
                return Some(2);
            }
        }
    }
    None
}

fn parse_rules(program: &str) -> Vec<Rule> {
    let mut rules = Vec::new();
    let mut cursor = 0usize;
    while cursor < program.len() {
        let Some(open_rel) = find_unquoted(&program[cursor..], '{') else {
            let pattern = program[cursor..].trim();
            if !pattern.is_empty() {
                rules.push(Rule {
                    phase: Phase::Record,
                    pattern: pattern.to_string(),
                    actions: "print".to_string(),
                });
            }
            break;
        };
        let open = cursor + open_rel;
        let Some(close) = matching_brace(program, open) else {
            break;
        };
        let raw_pattern = program[cursor..open].trim();
        let (phase, pattern) = match raw_pattern {
            "BEGIN" => (Phase::Begin, ""),
            "END" => (Phase::End, ""),
            _ => (Phase::Record, raw_pattern),
        };
        rules.push(Rule {
            phase,
            pattern: pattern.to_string(),
            actions: program[open + 1..close].to_string(),
        });
        cursor = close + 1;
    }
    rules
}

fn find_unquoted(source: &str, wanted: char) -> Option<usize> {
    let mut quote = None;
    let mut escaped = false;
    for (index, ch) in source.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
        } else if matches!(ch, '\'' | '"') {
            quote = if quote == Some(ch) {
                None
            } else if quote.is_none() {
                Some(ch)
            } else {
                quote
            };
        } else if ch == wanted && quote.is_none() {
            return Some(index);
        }
    }
    None
}

fn matching_brace(source: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut quote = None;
    let mut escaped = false;
    for (offset, ch) in source[open..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
        } else if matches!(ch, '\'' | '"') {
            quote = if quote == Some(ch) {
                None
            } else if quote.is_none() {
                Some(ch)
            } else {
                quote
            };
        } else if quote.is_none() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(open + offset);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

fn run_actions(actions: &str, state: &mut State, io: &mut Io) -> Control {
    for action in split_top_level(actions, &[';', '\n']) {
        let action = action.trim();
        if action.is_empty() {
            continue;
        }
        if action == "next" || action == "nextfile" {
            return Control::Next;
        }
        if let Some(rest) = action.strip_prefix("exit") {
            return Control::Exit(eval_value(rest.trim(), state).as_number() as i32);
        }
        if action == "print" {
            io.out.extend_from_slice(state.record.as_bytes());
            io.out.extend_from_slice(state.ors.as_bytes());
            continue;
        }
        if let Some(rest) = action.strip_prefix("print ") {
            let values = split_top_level(rest, &[','])
                .into_iter()
                .map(|expression| eval_value(expression.trim(), state).text)
                .collect::<Vec<_>>();
            io.out.extend_from_slice(values.join(&state.ofs).as_bytes());
            io.out.extend_from_slice(state.ors.as_bytes());
            continue;
        }
        if let Some(rest) = action.strip_prefix("printf ") {
            let args = split_top_level(rest, &[',']);
            let Some(format) = args.first() else { continue };
            let format = eval_value(format.trim(), state).text;
            let values = args
                .iter()
                .skip(1)
                .map(|value| eval_value(value.trim(), state))
                .collect::<Vec<_>>();
            io.out
                .extend_from_slice(format_printf(&format, &values).as_bytes());
            continue;
        }
        if let Some(name) = action.strip_suffix("++").map(str::trim) {
            if valid_name(name) {
                let value = variable(name, state).as_number() + 1.0;
                set_variable(name, number_text(value), state);
                continue;
            }
        }
        if let Some((name, op, expression)) = action_assignment(action) {
            let rhs = eval_value(expression, state);
            let value = match op {
                "=" => rhs.text,
                "+=" => number_text(variable(name, state).as_number() + rhs.as_number()),
                "-=" => number_text(variable(name, state).as_number() - rhs.as_number()),
                "*=" => number_text(variable(name, state).as_number() * rhs.as_number()),
                "/=" if rhs.as_number() != 0.0 => {
                    number_text(variable(name, state).as_number() / rhs.as_number())
                }
                _ => variable(name, state).text,
            };
            set_variable(name, value, state);
            continue;
        }
        return Control::Unsupported(action.to_string());
    }
    Control::Continue
}

#[derive(Clone)]
struct Scalar {
    text: String,
}

impl Scalar {
    fn as_number(&self) -> f64 {
        self.text.trim().parse().unwrap_or(0.0)
    }

    fn truthy(&self) -> bool {
        self.as_number() != 0.0 || (!self.text.is_empty() && self.text != "0")
    }
}

fn eval_value(expression: &str, state: &State) -> Scalar {
    let expression = expression.trim();
    if expression.starts_with('"') && expression.ends_with('"') && expression.len() >= 2 {
        return Scalar {
            text: unescape(&expression[1..expression.len() - 1]),
        };
    }
    if let Some(index) = expression.strip_prefix('$') {
        let index = if index == "NF" {
            state.fields.len()
        } else {
            index.parse().unwrap_or(0)
        };
        return Scalar {
            text: if index == 0 {
                state.record.clone()
            } else {
                state.fields.get(index - 1).cloned().unwrap_or_default()
            },
        };
    }
    for function in ["length", "tolower", "toupper", "int"] {
        if let Some(inner) = call_arg(expression, function) {
            let value = if inner.trim().is_empty() {
                Scalar {
                    text: state.record.clone(),
                }
            } else {
                eval_value(inner, state)
            };
            return Scalar {
                text: match function {
                    "length" => value.text.chars().count().to_string(),
                    "tolower" => value.text.to_lowercase(),
                    "toupper" => value.text.to_uppercase(),
                    "int" => (value.as_number() as i64).to_string(),
                    _ => String::new(),
                },
            };
        }
    }
    if let Some((left, op, right)) =
        split_binary(expression, &['+', '-']).or_else(|| split_binary(expression, &['*', '/', '%']))
    {
        let left = eval_value(left, state).as_number();
        let right = eval_value(right, state).as_number();
        let value = match op {
            '+' => left + right,
            '-' => left - right,
            '*' => left * right,
            '/' if right != 0.0 => left / right,
            '%' if right != 0.0 => left % right,
            _ => 0.0,
        };
        return Scalar {
            text: number_text(value),
        };
    }
    variable(expression, state)
}

fn variable(name: &str, state: &State) -> Scalar {
    let text = match name {
        "NR" => state.nr.to_string(),
        "FNR" => state.fnr.to_string(),
        "NF" => state.fields.len().to_string(),
        "FILENAME" => state.filename.clone(),
        "FS" => state.fs.clone(),
        "OFS" => state.ofs.clone(),
        "ORS" => state.ors.clone(),
        value if value.parse::<f64>().is_ok() => value.to_string(),
        value => state.vars.get(value).cloned().unwrap_or_default(),
    };
    Scalar { text }
}

fn set_variable(name: &str, value: String, state: &mut State) {
    match name {
        "FS" => state.fs = value,
        "OFS" => state.ofs = value,
        "ORS" => state.ors = value,
        _ => {
            state.vars.insert(name.to_string(), value);
        }
    }
}

fn eval_condition(expression: &str, state: &State) -> bool {
    let expression = expression.trim();
    if expression.is_empty() {
        return true;
    }
    if let Some((left, right)) = split_operator(expression, "||") {
        return eval_condition(left, state) || eval_condition(right, state);
    }
    if let Some((left, right)) = split_operator(expression, "&&") {
        return eval_condition(left, state) && eval_condition(right, state);
    }
    if let Some(rest) = expression.strip_prefix('!') {
        return !eval_condition(rest, state);
    }
    if expression.starts_with('/') && expression.ends_with('/') && expression.len() >= 2 {
        return regex::Regex::new(&expression[1..expression.len() - 1])
            .map(|regex| regex.is_match(&state.record))
            .unwrap_or(false);
    }
    for op in ["!~", "==", "!=", ">=", "<=", "~", ">", "<"] {
        if let Some((left, right)) = split_operator(expression, op) {
            let left = eval_value(left, state);
            let right = eval_value(right, state);
            let numeric_pair = left
                .text
                .trim()
                .parse::<f64>()
                .ok()
                .zip(right.text.trim().parse::<f64>().ok());
            return match op {
                "==" => numeric_pair
                    .map(|(left, right)| left == right)
                    .unwrap_or(left.text == right.text),
                "!=" => numeric_pair
                    .map(|(left, right)| left != right)
                    .unwrap_or(left.text != right.text),
                ">=" => left.as_number() >= right.as_number(),
                "<=" => left.as_number() <= right.as_number(),
                ">" => left.as_number() > right.as_number(),
                "<" => left.as_number() < right.as_number(),
                "~" | "!~" => {
                    let pattern = right.text.trim_matches('/');
                    let matched = regex::Regex::new(pattern)
                        .map(|regex| regex.is_match(&left.text))
                        .unwrap_or(false);
                    if op == "~" {
                        matched
                    } else {
                        !matched
                    }
                }
                _ => false,
            };
        }
    }
    eval_value(expression, state).truthy()
}

fn split_fields(record: &str, fs: &str) -> Vec<String> {
    if fs == " " || fs.is_empty() {
        return record.split_whitespace().map(str::to_string).collect();
    }
    regex::Regex::new(fs)
        .map(|regex| regex.split(record).map(str::to_string).collect())
        .unwrap_or_else(|_| record.split(fs).map(str::to_string).collect())
}

fn format_printf(format: &str, values: &[Scalar]) -> String {
    let mut output = String::new();
    let mut chars = format.chars().peekable();
    let mut value_index = 0usize;
    while let Some(ch) = chars.next() {
        if ch != '%' {
            output.push(ch);
            continue;
        }
        if chars.peek() == Some(&'%') {
            chars.next();
            output.push('%');
            continue;
        }
        let mut conversion = chars.next().unwrap_or('s');
        while !matches!(conversion, 's' | 'd' | 'i' | 'f' | 'g' | 'c') {
            conversion = chars.next().unwrap_or('s');
        }
        let value = values.get(value_index).cloned().unwrap_or(Scalar {
            text: String::new(),
        });
        value_index += 1;
        match conversion {
            'd' | 'i' => output.push_str(&(value.as_number() as i64).to_string()),
            'f' | 'g' => output.push_str(&number_text(value.as_number())),
            'c' => output.push(value.text.chars().next().unwrap_or('\0')),
            _ => output.push_str(&value.text),
        }
    }
    output
}

fn split_top_level<'a>(source: &'a str, separators: &[char]) -> Vec<&'a str> {
    let mut output = Vec::new();
    let mut quote = None;
    let mut depth = 0usize;
    let mut start = 0usize;
    for (index, ch) in source.char_indices() {
        if matches!(ch, '\'' | '"') {
            quote = if quote == Some(ch) {
                None
            } else if quote.is_none() {
                Some(ch)
            } else {
                quote
            };
        } else if ch == '(' && quote.is_none() {
            depth += 1;
        } else if ch == ')' && quote.is_none() {
            depth = depth.saturating_sub(1);
        } else if separators.contains(&ch) && quote.is_none() && depth == 0 {
            output.push(&source[start..index]);
            start = index + ch.len_utf8();
        }
    }
    output.push(&source[start..]);
    output
}

fn split_operator<'a>(source: &'a str, operator: &str) -> Option<(&'a str, &'a str)> {
    let mut quote = None;
    let mut depth = 0usize;
    for (index, ch) in source.char_indices() {
        if matches!(ch, '\'' | '"') {
            quote = if quote == Some(ch) {
                None
            } else if quote.is_none() {
                Some(ch)
            } else {
                quote
            };
        } else if ch == '(' && quote.is_none() {
            depth += 1;
        } else if ch == ')' && quote.is_none() {
            depth = depth.saturating_sub(1);
        } else if quote.is_none() && depth == 0 && source[index..].starts_with(operator) {
            return Some((&source[..index], &source[index + operator.len()..]));
        }
    }
    None
}

fn split_binary<'a>(expression: &'a str, operators: &[char]) -> Option<(&'a str, char, &'a str)> {
    for (index, ch) in expression.char_indices().rev() {
        if operators.contains(&ch) && index > 0 {
            return Some((
                &expression[..index],
                ch,
                &expression[index + ch.len_utf8()..],
            ));
        }
    }
    None
}

fn action_assignment(action: &str) -> Option<(&str, &str, &str)> {
    for op in ["+=", "-=", "*=", "/=", "="] {
        if let Some((name, expression)) = action.split_once(op) {
            let name = name.trim();
            if valid_name(name) {
                return Some((name, op, expression.trim()));
            }
        }
    }
    None
}

fn call_arg<'a>(expression: &'a str, function: &str) -> Option<&'a str> {
    expression
        .strip_prefix(function)?
        .trim()
        .strip_prefix('(')?
        .strip_suffix(')')
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().enumerate().all(|(index, ch)| {
            ch == '_' || ch.is_ascii_alphabetic() || (index > 0 && ch.is_ascii_digit())
        })
}

fn number_text(value: f64) -> String {
    if value.fract() == 0.0 {
        (value as i64).to_string()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use crate::interp::Interp;

    #[test]
    fn fields_patterns_and_totals() {
        let mut env = Interp::new();
        let (_, out, err) = env.run_script_capture(
            "printf 'a,2\\nb,3\\n' | awk -F, 'NR > 1 { total += $2; print $1 } END { print total }'",
        );
        assert_eq!(String::from_utf8_lossy(&out), "b\n3\n");
        assert!(err.is_empty());
    }
}
