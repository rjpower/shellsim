//! Typed evaluator for ordinary `find` path predicates.
//!
//! Boolean expressions are parsed before the VFS walk. Supported predicates are deliberately
//! closed: metadata tests, boolean composition, printing, deletion, and `-exec`/`-execdir`.
//! Unknown predicates fail rather than being treated as paths or delegated to the host.
//!
//! [`Expr`] and [`Parser`] are shared with [`crate::program::find`], the resumable native process
//! image seeded at `/usr/bin/find`. This module keeps the original synchronous body (registered
//! under the bare name `find`) for nested callers that dispatch commands in-process, such as
//! `xargs` invoking `find` recursively; `-exec` there still runs through synchronous nested
//! dispatch rather than a real child process. The native image is the one real shell scripts
//! reach when they run `find` as an external command, and it spawns one virtual child per
//! `-exec`/`-execdir` invocation.

use super::util::glob_eq;

#[derive(Clone, Debug)]
pub(crate) enum Expr {
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
        /// `-execdir`: run with the file's directory as the child's cwd and substitute `{}`
        /// with `./basename` instead of the display path.
        dir: bool,
    },
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug)]
pub(crate) struct SizeComparison {
    ordering: std::cmp::Ordering,
    units: u64,
    unit_bytes: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct PermComparison {
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
pub(crate) struct AgeComparison {
    ordering: std::cmp::Ordering,
    value: u64,
    unit_ms: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct Parsed {
    pub(crate) paths: Vec<String>,
    pub(crate) expression: Expr,
    pub(crate) min_depth: usize,
    pub(crate) max_depth: Option<usize>,
    pub(crate) explicit_action: bool,
    pub(crate) delete: bool,
    pub(crate) exec_batches: usize,
}

pub(crate) struct Parser<'a> {
    arguments: &'a [String],
    offset: usize,
    min_depth: usize,
    max_depth: Option<usize>,
    explicit_action: bool,
    delete: bool,
    exec_batches: usize,
}

impl<'a> Parser<'a> {
    pub(crate) fn parse(arguments: &'a [String]) -> Result<Parsed, String> {
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
            "-exec" => self.parse_exec("-exec", false),
            "-execdir" => self.parse_exec("-execdir", true),
            "-ok" | "-okdir" => Err(format!(
                "'{argument}' is unsupported: it requires an interactive terminal, which the \
                 simulator does not provide"
            )),
            value if value.starts_with('-') => Err(format!("unsupported predicate '{value}'")),
            value => Err(format!("unexpected path or expression '{value}'")),
        }
    }

    fn parse_exec(&mut self, name: &str, dir: bool) -> Result<Expr, String> {
        let mut argv = Vec::new();
        let batch = loop {
            let Some(argument) = self.advance() else {
                return Err(format!("missing terminator for '{name}'"));
            };
            match argument {
                ";" => break None,
                "+" => {
                    if argv.last().map(String::as_str) != Some("{}") {
                        return Err(format!(
                            "'{name} ... +' requires '{{}}' immediately before '+'"
                        ));
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
            return Err(format!("missing command for '{name}'"));
        }
        if dir && batch.is_some() {
            return Err(
                "'-execdir ... +' is unsupported: batching would require grouping matches by \
                 directory"
                    .to_string(),
            );
        }
        self.explicit_action = true;
        Ok(Expr::Exec { argv, batch, dir })
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

/// Evaluate one metadata-only predicate leaf against the typed [`crate::syscalls::System`]
/// boundary, for the resumable native image in [`crate::program::find`].
///
/// `True`, `Print`, `Exec`, and the boolean combinators are handled by each caller instead: their
/// side effects (writing bytes, spawning children, short-circuit recursion) differ between the
/// synchronous [`evaluate`] above and the native image's suspend-capable evaluator, so only the
/// metadata comparisons that read the same [`crate::syscalls::FileInfo`] shape are shared here.
/// Semantics are kept bit-for-bit identical to `evaluate`'s handling of the same predicates,
/// including treating a native executable as a non-empty file for `-empty`.
pub(crate) fn evaluate_metadata_leaf(
    system: &mut dyn crate::syscalls::System,
    expr: &Expr,
    path: &str,
    display: &str,
) -> Result<bool, String> {
    use crate::syscalls::FileKind;
    match expr {
        Expr::Type(expected) => {
            let info = system
                .metadata("/", path, false)
                .map_err(|error| format!("{display}: {error}"))?;
            Ok(matches!(
                (expected, info.kind),
                ('f', FileKind::File) | ('d', FileKind::Directory) | ('l', FileKind::Symlink)
            ))
        }
        Expr::Name(pattern) => Ok(glob_eq(pattern, crate::vfs::basename(path))),
        Expr::Path(pattern) => Ok(glob_eq(pattern, display)),
        Expr::Empty => {
            let info = system
                .metadata("/", path, false)
                .map_err(|error| format!("{display}: {error}"))?;
            Ok(match info.kind {
                FileKind::File if info.native_executable => false,
                FileKind::File => info.size == 0,
                FileKind::Directory => system
                    .list_dir("/", path)
                    .map_err(|error| format!("{display}: {error}"))?
                    .is_empty(),
                FileKind::Symlink => false,
            })
        }
        Expr::Newer(reference) => {
            let modified = system
                .metadata("/", path, false)
                .map_err(|error| format!("{display}: {error}"))?
                .mtime_ms;
            let base = system.cwd().to_string();
            let reference_mtime = system
                .metadata(&base, reference, true)
                .map_err(|error| format!("{reference}: {error}"))?
                .mtime_ms;
            Ok(modified > reference_mtime)
        }
        Expr::Size(comparison) => {
            let info = system
                .metadata("/", path, false)
                .map_err(|error| format!("{display}: {error}"))?;
            let size = if matches!(info.kind, FileKind::Directory) {
                0
            } else {
                info.size
            };
            let units = size.saturating_add(comparison.unit_bytes.saturating_sub(1))
                / comparison.unit_bytes;
            Ok(units.cmp(&comparison.units) == comparison.ordering)
        }
        Expr::Perm(comparison) => {
            let mode = system
                .metadata("/", path, false)
                .map_err(|error| format!("{display}: {error}"))?
                .mode
                & 0o7777;
            Ok(match comparison.kind {
                PermKind::Exact => mode == comparison.mode,
                PermKind::All => mode & comparison.mode == comparison.mode,
                PermKind::Any => mode & comparison.mode != 0,
            })
        }
        Expr::Age(comparison) => {
            let modified = system
                .metadata("/", path, false)
                .map_err(|error| format!("{display}: {error}"))?
                .mtime_ms;
            let age = system.wall_time_ms().saturating_sub(modified) / comparison.unit_ms;
            Ok(age.cmp(&comparison.value) == comparison.ordering)
        }
        Expr::Delete => {
            let info = system
                .metadata("/", path, false)
                .map_err(|error| format!("{display}: {error}"))?;
            let result = if matches!(info.kind, FileKind::Directory) {
                system.rmdir("/", path)
            } else {
                system.unlink("/", path)
            };
            result.map_err(|error| format!("cannot delete '{display}': {error}"))?;
            Ok(true)
        }
        Expr::True
        | Expr::Print(_)
        | Expr::Exec { .. }
        | Expr::Not(_)
        | Expr::And(_, _)
        | Expr::Or(_, _) => {
            unreachable!("evaluate_metadata_leaf only handles metadata predicates")
        }
    }
}

/// Split an absolute path into its parent directory and basename, for `-execdir`'s `cwd`
/// substitution. The root path has itself as its own parent.
pub(crate) fn split_dir(path: &str) -> (String, String) {
    match path.rsplit_once('/') {
        Some(("", base)) => ("/".to_string(), base.to_string()),
        Some((parent, base)) => (parent.to_string(), base.to_string()),
        None => ("/".to_string(), path.to_string()),
    }
}

pub(crate) fn display_path(cwd: &str, start: &str, absolute: &str, path: &str) -> String {
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
