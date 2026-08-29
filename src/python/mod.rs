//! Minimal `python -c` compatibility shim.
//!
//! This is intentionally not a Python interpreter. It recognizes a small, explicit set of
//! bootstrap one-liners and fails loudly for everything else. Host Python is never invoked.

use crate::interp::Interp;

type Out<'a> = &'a mut Vec<u8>;

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
    for statement in statements(&source) {
        let statement = statement.trim();
        if statement.is_empty() || statement == "import sys" || statement == "import os" {
            continue;
        }
        if let Some(inner) = call_argument(statement, "print") {
            let mut values = Vec::new();
            for expression in split_arguments(inner) {
                if expression.trim().is_empty() {
                    continue;
                }
                let Some(value) = eval_expr(interp, expression, &py_argv) else {
                    return unsupported(interp, statement, err);
                };
                values.push(value);
            }
            out.extend_from_slice(values.join(" ").as_bytes());
            out.push(b'\n');
            continue;
        }
        if let Some(inner) = call_argument(statement, "sys.stdout.write") {
            let Some(value) = eval_expr(interp, inner, &py_argv) else {
                return unsupported(interp, statement, err);
            };
            out.extend_from_slice(value.as_bytes());
            continue;
        }
        if let Some(inner) = call_argument(statement, "sys.stderr.write") {
            let Some(value) = eval_expr(interp, inner, &py_argv) else {
                return unsupported(interp, statement, err);
            };
            err.extend_from_slice(value.as_bytes());
            continue;
        }
        if let Some(inner) = call_argument(statement, "sys.exit")
            .or_else(|| call_argument(statement, "raise SystemExit"))
        {
            return inner.trim().parse().unwrap_or(0);
        }
        return unsupported(interp, statement, err);
    }
    0
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

fn eval_expr(interp: &Interp, expression: &str, argv: &[String]) -> Option<String> {
    let expression = expression.trim();
    if let Some(value) = string_literal(expression) {
        return Some(value);
    }
    if expression.parse::<i64>().is_ok() {
        return Some(expression.to_string());
    }
    match expression {
        "None" => return Some("None".to_string()),
        "True" => return Some("True".to_string()),
        "False" => return Some("False".to_string()),
        "sys.version" => return Some("3.12.0 (shellsim minimal shim)".to_string()),
        "sys.version_info" => return Some("(3, 12, 0, 'final', 0)".to_string()),
        "sys.executable" => return Some("/usr/bin/python".to_string()),
        "sys.prefix" => return Some("/usr".to_string()),
        _ => {}
    }
    if let Some(index) = expression
        .strip_prefix("sys.argv[")
        .and_then(|rest| rest.strip_suffix(']'))
        .and_then(|index| index.parse::<usize>().ok())
    {
        return argv.get(index).cloned();
    }
    for function in ["os.getenv", "os.environ.get"] {
        if let Some(arguments) = call_argument(expression, function) {
            let parts = split_arguments(arguments);
            let key = parts.first().and_then(|part| string_literal(part.trim()))?;
            let fallback = parts
                .get(1)
                .and_then(|part| string_literal(part.trim()))
                .unwrap_or_default();
            return Some(interp.get_var(&key).unwrap_or(fallback));
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
        } else if separators.contains(&ch) && quote.is_none() {
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
    fn multiline_and_multiple_print_arguments() {
        assert_eq!(
            run("import sys\nprint('hello', 3)"),
            (0, "hello 3\n".into(), String::new())
        );
    }

    #[test]
    fn unsupported_python_fails_loudly() {
        let (status, _, err) = run("x = 1");
        assert_eq!(status, 2);
        assert!(err.contains("unsupported"));
    }
}
