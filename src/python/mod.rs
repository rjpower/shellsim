//! Minimal Python bootstrap and REPL compatibility.
//!
//! This is intentionally not a Python interpreter. It recognizes a small, explicit set of
//! one-liners, keeps simple REPL variables across shell actions, and fails loudly for everything
//! else. Host Python is never invoked.

use std::collections::HashMap;

use crate::interp::Interp;

type Out<'a> = &'a mut Vec<u8>;

#[derive(Clone, Debug)]
enum Value {
    String(String),
    Int(i64),
    Bool(bool),
    None,
}

impl Value {
    fn display(&self) -> String {
        match self {
            Self::String(value) => value.clone(),
            Self::Int(value) => value.to_string(),
            Self::Bool(value) => if *value { "True" } else { "False" }.to_string(),
            Self::None => "None".to_string(),
        }
    }

    fn repr(&self) -> String {
        match self {
            Self::String(value) => format!("'{value}'"),
            _ => self.display(),
        }
    }

    fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(value) => Some(*value),
            Self::Bool(value) => Some(i64::from(*value)),
            Self::String(value) => value.parse().ok(),
            Self::None => None,
        }
    }
}

/// Persistent locals for the deliberately-small foreground Python REPL.
#[derive(Default, Debug)]
pub struct ReplState {
    locals: HashMap<String, Value>,
}

enum ExecResult {
    Continue,
    Exit(i32),
    Unsupported(String),
}

pub fn run_python(interp: &mut Interp, argv: &[String], stdin: Vec<u8>, out: Out, err: Out) -> i32 {
    let args = argv.get(1..).unwrap_or_default();
    if args.iter().any(|arg| arg == "--version" || arg == "-V") {
        out.extend_from_slice(b"Python 3.12.0\n");
        return 0;
    }

    if args.first().map(String::as_str) == Some("-m") {
        return run_module(interp, &args[1..], err);
    }

    let (source, program_args) = if let Some(pos) = args.iter().position(|arg| arg == "-c") {
        (
            args.get(pos + 1).cloned().unwrap_or_default(),
            args.get(pos + 2..).unwrap_or_default(),
        )
    } else if args.first().map(String::as_str) == Some("-")
        || (args.is_empty() && !stdin.is_empty())
    {
        (
            String::from_utf8_lossy(&stdin).into_owned(),
            args.get(1..).unwrap_or_default(),
        )
    } else if args.is_empty() {
        interp.python_repl = Some(ReplState::default());
        out.extend_from_slice(
            b"Python 3.12.0 (shellsim minimal shim)\nType exit() or quit() to return to the shell.\n>>> ",
        );
        return 0;
    } else {
        return unsupported(interp, "script execution", err);
    };

    let scratch = 10 * 1024 + source.len() as u64;
    if !interp.resources.reserve_memory(scratch)
        || !interp.resources.charge_cpu(100 + source.len() as u64)
    {
        return 137;
    }

    let mut py_argv = vec!["-c".to_string()];
    py_argv.extend_from_slice(program_args);
    let mut state = ReplState::default();
    match execute_source(interp, &source, &py_argv, &mut state, false, out, err) {
        ExecResult::Continue => 0,
        ExecResult::Exit(status) => status,
        ExecResult::Unsupported(feature) => unsupported(interp, &feature, err),
    }
}

/// Execute one action while Python owns the foreground session. The caller temporarily removes
/// `state` from the environment to avoid aliasing it with `interp`.
pub fn run_repl_line(
    interp: &mut Interp,
    state: &mut ReplState,
    source: &str,
    out: Out,
    err: Out,
) -> (i32, bool) {
    let memory_mark = interp.resources.memory_mark();
    let scratch = 10 * 1024 + source.len() as u64;
    if !interp.resources.reserve_memory(scratch)
        || !interp.resources.charge_cpu(100 + source.len() as u64)
    {
        interp.resources.restore_memory(memory_mark);
        return (137, false);
    }

    let result = execute_source(
        interp,
        source,
        &["<stdin>".to_string()],
        state,
        true,
        out,
        err,
    );
    let (status, stay) = match result {
        ExecResult::Continue => (0, true),
        ExecResult::Exit(status) => (status, false),
        ExecResult::Unsupported(feature) => {
            interp.note_unsupported(&format!("python:{feature}"));
            err.extend_from_slice(
                format!("python: unsupported by minimal REPL: {feature}\n").as_bytes(),
            );
            (2, true)
        }
    };
    interp.resources.restore_memory(memory_mark);
    if stay {
        out.extend_from_slice(b">>> ");
    }
    (status, stay)
}

fn execute_source(
    interp: &Interp,
    source: &str,
    argv: &[String],
    state: &mut ReplState,
    interactive: bool,
    out: Out,
    err: Out,
) -> ExecResult {
    for statement in statements(source) {
        let statement = statement.trim();
        if statement.is_empty() || statement == "import sys" || statement == "import os" {
            continue;
        }
        if statement == "exit()" || statement == "quit()" {
            return ExecResult::Exit(0);
        }
        if let Some(inner) = call_argument(statement, "print") {
            let mut values = Vec::new();
            for expression in split_arguments(inner) {
                if expression.trim().is_empty() {
                    continue;
                }
                let Some(value) = eval_expr(interp, expression, argv, state) else {
                    return ExecResult::Unsupported(statement.to_string());
                };
                values.push(value.display());
            }
            out.extend_from_slice(values.join(" ").as_bytes());
            out.push(b'\n');
            continue;
        }
        if let Some(inner) = call_argument(statement, "sys.stdout.write") {
            let Some(value) = eval_expr(interp, inner, argv, state) else {
                return ExecResult::Unsupported(statement.to_string());
            };
            out.extend_from_slice(value.display().as_bytes());
            continue;
        }
        if let Some(inner) = call_argument(statement, "sys.stderr.write") {
            let Some(value) = eval_expr(interp, inner, argv, state) else {
                return ExecResult::Unsupported(statement.to_string());
            };
            err.extend_from_slice(value.display().as_bytes());
            continue;
        }
        if let Some(inner) = call_argument(statement, "sys.exit")
            .or_else(|| call_argument(statement, "raise SystemExit"))
        {
            let status = eval_expr(interp, inner, argv, state)
                .and_then(|value| value.as_int())
                .unwrap_or(0);
            return ExecResult::Exit(status as i32);
        }
        if let Some(name) = statement.strip_prefix("del ").map(str::trim) {
            state.locals.remove(name);
            continue;
        }
        if let Some((name, expression)) = assignment(statement) {
            let Some(value) = eval_expr(interp, expression, argv, state) else {
                return ExecResult::Unsupported(statement.to_string());
            };
            state.locals.insert(name.to_string(), value);
            continue;
        }
        if let Some(value) = eval_expr(interp, statement, argv, state) {
            if interactive {
                out.extend_from_slice(value.repr().as_bytes());
                out.push(b'\n');
            }
            continue;
        }
        return ExecResult::Unsupported(statement.to_string());
    }
    ExecResult::Continue
}

fn run_module(interp: &mut Interp, args: &[String], err: Out) -> i32 {
    match args.first().map(String::as_str) {
        Some("pip") => {
            if args.get(1).map(String::as_str) == Some("install") {
                crate::commands::pkg::register_install_args(interp, &args[1..]);
            }
            0
        }
        Some("venv") => {
            let Some(dir) = args.iter().skip(1).find(|arg| !arg.starts_with('-')) else {
                return 1;
            };
            let base = crate::vfs::resolve_against(&interp.cwd, dir);
            if interp.vfs.mkdir_all("/", &format!("{base}/bin")).is_err()
                || interp
                    .vfs
                    .put_file(
                        &format!("{base}/bin/python"),
                        b"#!shellsim-python\n".to_vec(),
                        0o755,
                    )
                    .is_err()
            {
                err.extend_from_slice(b"python: venv: No space left on device\n");
                return 1;
            }
            0
        }
        Some(module) => unsupported(interp, &format!("module {module}"), err),
        None => 1,
    }
}

/// Launchers such as `uv run pytest` use this loud stub so verifier-grade Python is never
/// mistaken for a supported capability.
pub fn run_pytest(interp: &mut Interp, _args: &[String], _out: Out, err: Out) -> i32 {
    unsupported(interp, "pytest", err)
}

fn unsupported(interp: &mut Interp, feature: &str, err: Out) -> i32 {
    interp.note_unsupported(&format!("python:{feature}"));
    err.extend_from_slice(format!("python: unsupported by minimal shim: {feature}\n").as_bytes());
    2
}

fn call_argument<'a>(statement: &'a str, name: &str) -> Option<&'a str> {
    statement
        .strip_prefix(name)?
        .trim()
        .strip_prefix('(')?
        .strip_suffix(')')
}

fn eval_expr(
    interp: &Interp,
    expression: &str,
    argv: &[String],
    state: &ReplState,
) -> Option<Value> {
    let expression = expression.trim();
    if let Some(value) = string_literal(expression) {
        return Some(Value::String(value));
    }
    if let Ok(value) = expression.parse::<i64>() {
        return Some(Value::Int(value));
    }
    match expression {
        "None" => return Some(Value::None),
        "True" => return Some(Value::Bool(true)),
        "False" => return Some(Value::Bool(false)),
        "sys.version" => return Some(Value::String("3.12.0 (shellsim minimal shim)".to_string())),
        "sys.version_info" => return Some(Value::String("(3, 12, 0, 'final', 0)".to_string())),
        "sys.executable" => return Some(Value::String("/usr/bin/python".to_string())),
        "sys.prefix" => return Some(Value::String("/usr".to_string())),
        _ => {}
    }
    if let Some(value) = state.locals.get(expression) {
        return Some(value.clone());
    }
    if let Some(index) = expression
        .strip_prefix("sys.argv[")
        .and_then(|rest| rest.strip_suffix(']'))
        .and_then(|index| index.parse::<usize>().ok())
    {
        return argv.get(index).cloned().map(Value::String);
    }
    for function in ["os.getenv", "os.environ.get"] {
        if let Some(arguments) = call_argument(expression, function) {
            let parts = split_arguments(arguments);
            let key = parts.first().and_then(|part| string_literal(part.trim()))?;
            let fallback = parts
                .get(1)
                .and_then(|part| string_literal(part.trim()))
                .unwrap_or_default();
            return Some(Value::String(interp.get_var(&key).unwrap_or(fallback)));
        }
    }
    for function in ["str", "repr", "int", "len"] {
        if let Some(inner) = call_argument(expression, function) {
            let value = eval_expr(interp, inner, argv, state)?;
            return match function {
                "str" => Some(Value::String(value.display())),
                "repr" => Some(Value::String(value.repr())),
                "int" => value.as_int().map(Value::Int),
                "len" => Some(Value::Int(value.display().chars().count() as i64)),
                _ => None,
            };
        }
    }
    if let Some((left, op, right)) =
        split_binary(expression, &['+', '-']).or_else(|| split_binary(expression, &['*', '/']))
    {
        let left = eval_expr(interp, left, argv, state)?;
        let right = eval_expr(interp, right, argv, state)?;
        return match op {
            '+' => match (left, right) {
                (Value::String(a), Value::String(b)) => Some(Value::String(a + &b)),
                (a, b) => Some(Value::Int(a.as_int()?.saturating_add(b.as_int()?))),
            },
            '-' => Some(Value::Int(left.as_int()?.saturating_sub(right.as_int()?))),
            '*' => Some(Value::Int(left.as_int()?.saturating_mul(right.as_int()?))),
            '/' if right.as_int()? != 0 => Some(Value::Int(left.as_int()? / right.as_int()?)),
            _ => None,
        };
    }
    None
}

fn assignment(statement: &str) -> Option<(&str, &str)> {
    let mut quote = None;
    let mut depth = 0usize;
    for (index, ch) in statement.char_indices() {
        match ch {
            '\'' | '"' => {
                quote = if quote == Some(ch) {
                    None
                } else if quote.is_none() {
                    Some(ch)
                } else {
                    quote
                };
            }
            '(' | '[' if quote.is_none() => depth += 1,
            ')' | ']' if quote.is_none() => depth = depth.saturating_sub(1),
            '=' if quote.is_none() && depth == 0 => {
                let before = statement[..index].trim();
                let next = statement[index + 1..].chars().next();
                if before.ends_with(['!', '<', '>', '=']) || next == Some('=') {
                    continue;
                }
                if !before.is_empty()
                    && before.chars().enumerate().all(|(i, c)| {
                        c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit())
                    })
                {
                    return Some((before, statement[index + 1..].trim()));
                }
            }
            _ => {}
        }
    }
    None
}

fn split_binary<'a>(expression: &'a str, operators: &[char]) -> Option<(&'a str, char, &'a str)> {
    let mut quote = None;
    let mut depth = 0usize;
    let chars: Vec<(usize, char)> = expression.char_indices().collect();
    for (position, (index, ch)) in chars.iter().enumerate().rev() {
        match *ch {
            '\'' | '"' => {
                quote = if quote == Some(*ch) {
                    None
                } else if quote.is_none() {
                    Some(*ch)
                } else {
                    quote
                };
            }
            ')' | ']' if quote.is_none() => depth += 1,
            '(' | '[' if quote.is_none() => depth = depth.saturating_sub(1),
            op if quote.is_none() && depth == 0 && operators.contains(&op) => {
                if position == 0 {
                    continue;
                }
                let right_index = index + op.len_utf8();
                return Some((&expression[..*index], op, &expression[right_index..]));
            }
            _ => {}
        }
    }
    None
}

fn string_literal(source: &str) -> Option<String> {
    let quote = source.chars().next()?;
    if !matches!(quote, '\'' | '"') || !source.ends_with(quote) || source.len() < 2 {
        return None;
    }
    let body = &source[1..source.len() - 1];
    Some(crate::commands::util::unescape(body))
}

fn statements(source: &str) -> Vec<&str> {
    split_quoted(source, &[';', '\n'])
}

fn split_arguments(source: &str) -> Vec<&str> {
    split_quoted(source, &[','])
}

fn split_quoted<'a>(source: &'a str, separators: &[char]) -> Vec<&'a str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut quote = None;
    let mut escaped = false;
    let mut depth = 0usize;
    for (index, ch) in source.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if matches!(ch, '\'' | '"') {
            quote = if quote == Some(ch) {
                None
            } else if quote.is_none() {
                Some(ch)
            } else {
                quote
            };
        } else if matches!(ch, '(' | '[') && quote.is_none() {
            depth += 1;
        } else if matches!(ch, ')' | ']') && quote.is_none() {
            depth = depth.saturating_sub(1);
        } else if separators.contains(&ch) && quote.is_none() && depth == 0 {
            result.push(&source[start..index]);
            start = index + ch.len_utf8();
        }
    }
    result.push(&source[start..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(source: &str) -> (i32, String, String) {
        let mut env = Interp::new();
        let mut out = Vec::new();
        let mut err = Vec::new();
        let status = run_python(
            &mut env,
            &["python".into(), "-c".into(), source.into()],
            Vec::new(),
            &mut out,
            &mut err,
        );
        (
            status,
            String::from_utf8_lossy(&out).into_owned(),
            String::from_utf8_lossy(&err).into_owned(),
        )
    }

    #[test]
    fn literal_print_and_write() {
        assert_eq!(
            run("print('hello'); sys.stdout.write(\"!\")"),
            (0, "hello\n!".into(), String::new())
        );
    }

    #[test]
    fn multiline_assignments_and_multiple_print_arguments() {
        assert_eq!(
            run("import sys\nx = 3\nprint('hello', x + 2)"),
            (0, "hello 5\n".into(), String::new())
        );
    }

    #[test]
    fn unsupported_python_fails_loudly() {
        let (status, _, err) = run("for x in range(3): print(x)");
        assert_eq!(status, 2);
        assert!(err.contains("unsupported"));
    }
}
