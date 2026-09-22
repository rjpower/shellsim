//! Typed evaluator for ordinary `find` path predicates.
//!
//! Boolean expressions are parsed before the VFS walk. Supported predicates are deliberately
//! closed: metadata tests, boolean composition, printing, and deletion. Unknown predicates fail
//! rather than being treated as paths or delegated to the host.

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
    Empty,
    Newer(String),
    Size(SizeComparison),
    Perm(PermComparison),
    Age(AgeComparison),
    Print(bool),
    Delete,
    Exec {
        argv: Vec<String>,
        batch: Option<usize>,
    },
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug)]
struct SizeComparison {
    ordering: std::cmp::Ordering,
    units: u64,
    unit_bytes: u64,
}

#[derive(Clone, Debug)]
struct PermComparison {
    mode: u32,
    kind: PermKind,
}

#[derive(Clone, Debug)]
enum PermKind {
    Exact,
    All,
    Any,
}

#[derive(Clone, Debug)]
struct AgeComparison {
    ordering: std::cmp::Ordering,
    value: u64,
    unit_ms: u64,
}

#[derive(Debug)]
struct Parsed {
    paths: Vec<String>,
    expression: Expr,
    min_depth: usize,
    max_depth: Option<usize>,
    explicit_action: bool,
    delete: bool,
    exec_batches: usize,
}

struct Parser<'a> {
    arguments: &'a [String],
    offset: usize,
    min_depth: usize,
    max_depth: Option<usize>,
    explicit_action: bool,
    delete: bool,
    exec_batches: usize,
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
            delete: false,
            exec_batches: 0,
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
            delete: parser.delete,
            exec_batches: parser.exec_batches,
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
            "-empty" => Ok(Expr::Empty),
            "-newer" => Ok(Expr::Newer(self.required("-newer")?.to_string())),
            "-size" => Ok(Expr::Size(parse_size(self.required("-size")?)?)),
            "-perm" => Ok(Expr::Perm(parse_perm(self.required("-perm")?)?)),
            "-mtime" => Ok(Expr::Age(parse_age(self.required("-mtime")?, 86_400_000)?)),
            "-mmin" => Ok(Expr::Age(parse_age(self.required("-mmin")?, 60_000)?)),
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
            "-delete" => {
                self.explicit_action = true;
                self.delete = true;
                Ok(Expr::Delete)
            }
            "-exec" => self.parse_exec(),
            value if value.starts_with('-') => Err(format!("unsupported predicate '{value}'")),
            value => Err(format!("unexpected path or expression '{value}'")),
        }
    }

    fn parse_exec(&mut self) -> Result<Expr, String> {
        let mut argv = Vec::new();
        let batch = loop {
            let Some(argument) = self.advance() else {
                return Err("missing terminator for '-exec'".to_string());
            };
            match argument {
                ";" => break None,
                "+" => {
                    if argv.last().map(String::as_str) != Some("{}") {
                        return Err(
                            "'-exec ... +' requires '{}' immediately before '+'".to_string()
                        );
                    }
                    argv.pop();
                    let id = self.exec_batches;
                    self.exec_batches = self.exec_batches.saturating_add(1);
                    break Some(id);
                }
                value => argv.push(value.to_string()),
            }
        };
        if argv.is_empty() {
            return Err("missing command for '-exec'".to_string());
        }
        self.explicit_action = true;
        Ok(Expr::Exec { argv, batch })
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

fn parse_size(value: &str) -> Result<SizeComparison, String> {
    let (ordering, value) = match value.as_bytes().first() {
        Some(b'+') => (std::cmp::Ordering::Greater, &value[1..]),
        Some(b'-') => (std::cmp::Ordering::Less, &value[1..]),
        _ => (std::cmp::Ordering::Equal, value),
    };
    let (number, unit_bytes) = if let Some(number) = value.strip_suffix('c') {
        (number, 1)
    } else if let Some(number) = value.strip_suffix('k') {
        (number, 1024)
    } else if let Some(number) = value.strip_suffix('M') {
        (number, 1024 * 1024)
    } else {
        (value, 512)
    };
    let number = number
        .parse::<u64>()
        .map_err(|_| format!("invalid argument to '-size': '{value}'"))?;
    Ok(SizeComparison {
        ordering,
        units: number,
        unit_bytes,
    })
}

fn parse_perm(value: &str) -> Result<PermComparison, String> {
    let (kind, digits) = if let Some(value) = value.strip_prefix('-') {
        (PermKind::All, value)
    } else if let Some(value) = value.strip_prefix('/') {
        (PermKind::Any, value)
    } else {
        (PermKind::Exact, value)
    };
    let mode = u32::from_str_radix(digits, 8)
        .map_err(|_| format!("invalid mode to '-perm': '{value}'"))?;
    Ok(PermComparison { mode, kind })
}

fn parse_age(value: &str, unit_ms: u64) -> Result<AgeComparison, String> {
    let (ordering, digits) = match value.as_bytes().first() {
        Some(b'+') => (std::cmp::Ordering::Greater, &value[1..]),
        Some(b'-') => (std::cmp::Ordering::Less, &value[1..]),
        _ => (std::cmp::Ordering::Equal, value),
    };
    let value = digits
        .parse::<u64>()
        .map_err(|_| format!("invalid age '{value}'"))?;
    Ok(AgeComparison {
        ordering,
        value,
        unit_ms,
    })
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
    if let Err(error) = validate_references(&parsed.expression, env) {
        ewln(io.err, &format!("find: {error}"));
        return 1;
    }
    let mut exec_batches = vec![Vec::new(); parsed.exec_batches];
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
        if parsed.delete {
            paths.reverse();
        }
        for path in paths {
            if !env.charge_cpu(1) {
                return 137;
            }
            let depth = path_depth(&path).saturating_sub(base_depth);
            if depth < parsed.min_depth || parsed.max_depth.is_some_and(|maximum| depth > maximum) {
                continue;
            }
            let display = display_path(&env.cwd, start, &absolute, &path);
            let selected = match evaluate(
                &parsed.expression,
                env,
                &path,
                &display,
                io,
                &mut exec_batches,
            ) {
                Ok(selected) => selected,
                Err(EvalError::ResourceExhausted) => return 137,
                Err(EvalError::Operational(message)) => {
                    ewln(io.err, &format!("find: {message}"));
                    return 1;
                }
            };
            if selected
                && !parsed.explicit_action
                && push_path(env, io.out, &display, false).is_err()
            {
                return 137;
            }
        }
    }
    run_exec_batches(&parsed.expression, &mut exec_batches, env, io)
}

fn validate_references(expression: &Expr, env: &CommandContext<'_>) -> Result<(), String> {
    match expression {
        Expr::Newer(path) => env
            .fs_metadata(&env.cwd, path, true)
            .map(|_| ())
            .map_err(|error| format!("{path}: {error}")),
        Expr::Not(inner) => validate_references(inner, env),
        Expr::And(left, right) | Expr::Or(left, right) => {
            validate_references(left, env)?;
            validate_references(right, env)
        }
        _ => Ok(()),
    }
}

fn run_exec_batches(
    expression: &Expr,
    batches: &mut [Vec<String>],
    env: &mut CommandContext<'_>,
    io: &mut Io,
) -> i32 {
    match expression {
        Expr::Exec {
            argv,
            batch: Some(id),
        } => {
            let paths = std::mem::take(&mut batches[*id]);
            if paths.is_empty() {
                return 0;
            }
            let mut command = argv.clone();
            command.extend(paths);
            crate::commands::run(env, &command, Vec::new(), io.out, io.err)
        }
        Expr::Not(inner) => run_exec_batches(inner, batches, env, io),
        Expr::And(left, right) | Expr::Or(left, right) => {
            let left_status = run_exec_batches(left, batches, env, io);
            let right_status = run_exec_batches(right, batches, env, io);
            left_status.max(right_status)
        }
        _ => 0,
    }
}

enum EvalError {
    ResourceExhausted,
    Operational(String),
}

fn evaluate(
    expression: &Expr,
    env: &mut CommandContext<'_>,
    path: &str,
    display: &str,
    io: &mut Io,
    exec_batches: &mut [Vec<String>],
) -> Result<bool, EvalError> {
    if !env.charge_cpu(1) {
        return Err(EvalError::ResourceExhausted);
    }
    match expression {
        Expr::True => Ok(true),
        Expr::Type(expected) => {
            let kind = env
                .fs_metadata("/", path, false)
                .map_err(|error| EvalError::Operational(format!("{display}: {error}")))?
                .kind;
            Ok(matches!(
                (expected, kind),
                ('f', NodeKind::File(_)) | ('d', NodeKind::Dir) | ('l', NodeKind::Symlink(_))
            ))
        }
        Expr::Name(pattern) => Ok(glob_eq(pattern, crate::vfs::basename(path))),
        Expr::Path(pattern) => Ok(glob_eq(pattern, display)),
        Expr::Empty => {
            let metadata = env
                .fs_metadata("/", path, false)
                .map_err(|error| EvalError::Operational(format!("{display}: {error}")))?;
            Ok(match metadata.kind {
                NodeKind::File(data) => data.is_empty(),
                NodeKind::Dir => env
                    .vfs
                    .list_dir("/", path)
                    .map_err(|error| EvalError::Operational(format!("{display}: {error}")))?
                    .is_empty(),
                NodeKind::Symlink(_) => false,
            })
        }
        Expr::Newer(reference) => {
            let modified = env
                .fs_metadata("/", path, false)
                .map_err(|error| EvalError::Operational(format!("{display}: {error}")))?
                .mtime;
            let reference = env
                .fs_metadata(&env.cwd.clone(), reference, true)
                .map_err(|error| EvalError::Operational(format!("{reference}: {error}")))?
                .mtime;
            Ok(modified > reference)
        }
        Expr::Size(comparison) => {
            let size = match env
                .fs_metadata("/", path, false)
                .map_err(|error| EvalError::Operational(format!("{display}: {error}")))?
                .kind
            {
                NodeKind::File(data) => data.len() as u64,
                NodeKind::Symlink(target) => target.len() as u64,
                NodeKind::Dir => 0,
            };
            let units = size.saturating_add(comparison.unit_bytes.saturating_sub(1))
                / comparison.unit_bytes;
            Ok(units.cmp(&comparison.units) == comparison.ordering)
        }
        Expr::Perm(comparison) => {
            let mode = env
                .fs_metadata("/", path, false)
                .map_err(|error| EvalError::Operational(format!("{display}: {error}")))?
                .mode
                & 0o7777;
            Ok(match comparison.kind {
                PermKind::Exact => mode == comparison.mode,
                PermKind::All => mode & comparison.mode == comparison.mode,
                PermKind::Any => mode & comparison.mode != 0,
            })
        }
        Expr::Age(comparison) => {
            let modified = env
                .fs_metadata("/", path, false)
                .map_err(|error| EvalError::Operational(format!("{display}: {error}")))?
                .mtime;
            let age = env.clock.unix_ms().saturating_sub(modified) / comparison.unit_ms;
            Ok(age.cmp(&comparison.value) == comparison.ordering)
        }
        Expr::Print(nul) => push_path(env, io.out, display, *nul)
            .map(|()| true)
            .map_err(|()| EvalError::ResourceExhausted),
        Expr::Delete => {
            let metadata = env
                .fs_metadata("/", path, false)
                .map_err(|error| EvalError::Operational(format!("{display}: {error}")))?;
            match metadata.kind {
                NodeKind::Dir => env.vfs.rmdir("/", path),
                _ => env.vfs.remove_file("/", path),
            }
            .map_err(|error| {
                EvalError::Operational(format!("cannot delete '{display}': {error}"))
            })?;
            Ok(true)
        }
        Expr::Exec { argv, batch } => {
            if let Some(id) = batch {
                exec_batches[*id].push(display.to_string());
                Ok(true)
            } else {
                let command = argv
                    .iter()
                    .map(|argument| argument.replace("{}", display))
                    .collect::<Vec<_>>();
                Ok(crate::commands::run(env, &command, Vec::new(), io.out, io.err) == 0)
            }
        }
        Expr::Not(value) => Ok(!evaluate(value, env, path, display, io, exec_batches)?),
        Expr::And(left, right) => Ok(evaluate(left, env, path, display, io, exec_batches)?
            && evaluate(right, env, path, display, io, exec_batches)?),
        Expr::Or(left, right) => Ok(evaluate(left, env, path, display, io, exec_batches)?
            || evaluate(right, env, path, display, io, exec_batches)?),
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
