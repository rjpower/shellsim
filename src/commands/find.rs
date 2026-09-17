//! Typed evaluator for ordinary `find` path predicates.
//!
//! Boolean expressions are parsed before the VFS walk. Supported predicates are deliberately
//! closed: type, name, path, depth bounds, and print actions. Unknown predicates fail rather than
//! being treated as paths or delegated to the host.

use std::collections::HashMap;

use super::util::{ewln, glob_eq};
use super::{reg_costed, CommandContext, CommandSpec, Io, Trust};
use crate::vfs::NodeKind;

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    reg_costed(commands, &["find"], Trust::Partial, 100, 16 * 1024, run);
}

#[derive(Clone, Debug)]
enum Expr {
    True,
    Type(char),
    Name(String),
    Path(String),
    Print(bool),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
}

#[derive(Debug)]
struct Parsed {
    paths: Vec<String>,
    expression: Expr,
    min_depth: usize,
    max_depth: Option<usize>,
    explicit_action: bool,
}

struct Parser<'a> {
    arguments: &'a [String],
    offset: usize,
    min_depth: usize,
    max_depth: Option<usize>,
    explicit_action: bool,
}

impl<'a> Parser<'a> {
    fn parse(arguments: &'a [String]) -> Result<Parsed, String> {
        let mut paths = Vec::new();
        let mut offset = 0;
        while arguments.get(offset).is_some_and(|argument| {
            !argument.starts_with('-') && !matches!(argument.as_str(), "!" | "(" | ")")
        }) {
            paths.push(arguments[offset].clone());
            offset += 1;
        }
        if paths.is_empty() {
            paths.push(".".to_string());
        }
        let mut parser = Self {
            arguments,
            offset,
            min_depth: 0,
            max_depth: None,
            explicit_action: false,
        };
        let expression = if offset == arguments.len() {
            Expr::True
        } else {
            parser.or()?
        };
        if let Some(argument) = parser.peek() {
            return Err(format!("unexpected argument '{argument}'"));
        }
        Ok(Parsed {
            paths,
            expression,
            min_depth: parser.min_depth,
            max_depth: parser.max_depth,
            explicit_action: parser.explicit_action,
        })
    }

    fn or(&mut self) -> Result<Expr, String> {
        let mut expression = self.and()?;
        while self.take_any(&["-o", "-or"]) {
            expression = Expr::Or(Box::new(expression), Box::new(self.and()?));
        }
        Ok(expression)
    }

    fn and(&mut self) -> Result<Expr, String> {
        let mut expression = self.not()?;
        loop {
            if self.take_any(&["-a", "-and"]) || self.peek().is_some_and(starts_primary) {
                expression = Expr::And(Box::new(expression), Box::new(self.not()?));
            } else {
                break;
            }
        }
        Ok(expression)
    }

    fn not(&mut self) -> Result<Expr, String> {
        if self.take_any(&["!", "-not"]) {
            Ok(Expr::Not(Box::new(self.not()?)))
        } else {
            self.primary()
        }
    }

    fn primary(&mut self) -> Result<Expr, String> {
        let Some(argument) = self.advance() else {
            return Err("expected predicate".to_string());
        };
        match argument {
            "(" => {
                let expression = self.or()?;
                if self.advance() != Some(")") {
                    return Err("missing ')'".to_string());
                }
                Ok(expression)
            }
            "-type" => {
                let value = self.required("-type")?;
                if value.len() == 1 && matches!(value, "f" | "d" | "l") {
                    Ok(Expr::Type(value.chars().next().expect("one character")))
                } else {
                    Err(format!("unsupported file type '{value}'"))
                }
            }
            "-name" => Ok(Expr::Name(self.required("-name")?.to_string())),
            "-path" | "-wholename" => Ok(Expr::Path(self.required(argument)?.to_string())),
            "-maxdepth" => {
                self.max_depth = Some(parse_depth(self.required("-maxdepth")?, "-maxdepth")?);
                Ok(Expr::True)
            }
            "-mindepth" => {
                self.min_depth = parse_depth(self.required("-mindepth")?, "-mindepth")?;
                Ok(Expr::True)
            }
            "-print" => {
                self.explicit_action = true;
                Ok(Expr::Print(false))
            }
            "-print0" => {
                self.explicit_action = true;
                Ok(Expr::Print(true))
            }
            value if value.starts_with('-') => Err(format!("unsupported predicate '{value}'")),
            value => Err(format!("unexpected path or expression '{value}'")),
        }
    }

    fn required(&mut self, predicate: &str) -> Result<&'a str, String> {
        self.advance()
            .ok_or_else(|| format!("missing argument to '{predicate}'"))
    }

    fn take_any(&mut self, expected: &[&str]) -> bool {
        if self.peek().is_some_and(|value| expected.contains(&value)) {
            self.offset += 1;
            true
        } else {
            false
        }
    }

    fn advance(&mut self) -> Option<&'a str> {
        let value = self.arguments.get(self.offset)?;
        self.offset += 1;
        Some(value)
    }

    fn peek(&self) -> Option<&str> {
        self.arguments.get(self.offset).map(String::as_str)
    }
}

fn starts_primary(argument: &str) -> bool {
    !matches!(argument, ")" | "-o" | "-or")
}

fn parse_depth(value: &str, predicate: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("invalid argument to '{predicate}': '{value}'"))
}

fn run(env: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let parsed = match Parser::parse(args) {
        Ok(parsed) => parsed,
        Err(error) => {
            ewln(io.err, &format!("find: {error}"));
            return 2;
        }
    };
    for start in &parsed.paths {
        let absolute = crate::vfs::resolve_against(&env.cwd, start);
        let base_depth = path_depth(&absolute);
        let mut paths = match env.fs_walk("/", &absolute) {
            Ok(paths) => paths,
            Err(error) => {
                ewln(io.err, &format!("find: {error}"));
                return 1;
            }
        };
        paths.sort();
        for path in paths {
            if !env.charge_cpu(1) {
                return 137;
            }
            let depth = path_depth(&path).saturating_sub(base_depth);
            if depth < parsed.min_depth || parsed.max_depth.is_some_and(|maximum| depth > maximum) {
                continue;
            }
            let display = display_path(&env.cwd, start, &absolute, &path);
            let selected = match evaluate(&parsed.expression, env, &path, &display, io) {
                Ok(selected) => selected,
                Err(()) => return 137,
            };
            if selected
                && !parsed.explicit_action
                && push_path(env, io.out, &display, false).is_err()
            {
                return 137;
            }
        }
    }
    0
}

fn evaluate(
    expression: &Expr,
    env: &mut CommandContext<'_>,
    path: &str,
    display: &str,
    io: &mut Io,
) -> Result<bool, ()> {
    if !env.charge_cpu(1) {
        return Err(());
    }
    match expression {
        Expr::True => Ok(true),
        Expr::Type(expected) => {
            let kind = env.fs_metadata("/", path, false).map_err(|_| ())?.kind;
            Ok(matches!(
                (expected, kind),
                ('f', NodeKind::File(_)) | ('d', NodeKind::Dir) | ('l', NodeKind::Symlink(_))
            ))
        }
        Expr::Name(pattern) => Ok(glob_eq(pattern, crate::vfs::basename(path))),
        Expr::Path(pattern) => Ok(glob_eq(pattern, display)),
        Expr::Print(nul) => push_path(env, io.out, display, *nul).map(|()| true),
        Expr::Not(value) => Ok(!evaluate(value, env, path, display, io)?),
        Expr::And(left, right) => {
            Ok(evaluate(left, env, path, display, io)? && evaluate(right, env, path, display, io)?)
        }
        Expr::Or(left, right) => {
            Ok(evaluate(left, env, path, display, io)? || evaluate(right, env, path, display, io)?)
        }
    }
}

fn push_path(
    env: &mut CommandContext<'_>,
    output: &mut Vec<u8>,
    display: &str,
    nul: bool,
) -> Result<(), ()> {
    let bytes = (display.len() as u64).saturating_add(1);
    if !env.resources.charge_output(bytes) {
        return Err(());
    }
    output.extend_from_slice(display.as_bytes());
    output.push(if nul { 0 } else { b'\n' });
    Ok(())
}

fn display_path(cwd: &str, start: &str, absolute: &str, path: &str) -> String {
    if start == "." {
        if path == absolute {
            ".".to_string()
        } else {
            let suffix = path
                .strip_prefix(absolute)
                .unwrap_or(path)
                .trim_start_matches('/');
            format!("./{suffix}")
        }
    } else if start.starts_with('/') {
        path.to_string()
    } else {
        let prefix = format!("{}/", cwd.trim_end_matches('/'));
        path.strip_prefix(&prefix).unwrap_or(path).to_string()
    }
}

fn path_depth(path: &str) -> usize {
    path.split('/').filter(|part| !part.is_empty()).count()
}

#[cfg(test)]
mod tests {
    use super::{Expr, Parser};

    #[test]
    fn parses_boolean_predicates_with_find_precedence() {
        let arguments = [
            ".".to_string(),
            "(".to_string(),
            "-name".to_string(),
            "*.rs".to_string(),
            "-o".to_string(),
            "-name".to_string(),
            "*.py".to_string(),
            ")".to_string(),
            "-type".to_string(),
            "f".to_string(),
        ];
        let parsed = Parser::parse(&arguments).unwrap();
        assert!(matches!(parsed.expression, Expr::And(_, _)));
    }
}
