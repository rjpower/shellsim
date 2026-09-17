//! Shared helpers used across the command modules: output writers, flag splitting,
//! duration parsing, input gathering, and the VFS-script fallback executor.

use crate::interp::Interp;

/// Write a string followed by a newline.
pub fn wln(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(s.as_bytes());
    out.push(b'\n');
}

/// Write a string with no trailing newline.
pub fn w(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(s.as_bytes());
}

/// Write a string followed by a newline to stderr.
pub fn ewln(err: &mut Vec<u8>, s: &str) {
    err.extend_from_slice(s.as_bytes());
    err.push(b'\n');
}

/// Split argv into (single-char flags, positional operands, long/value flags).
pub fn split_flags(args: &[String]) -> (Vec<char>, Vec<&String>, Vec<(&str, String)>) {
    let mut flags = Vec::new();
    let mut ops = Vec::new();
    let mut long = Vec::new();
    let mut only_ops = false;
    for a in args {
        if only_ops {
            ops.push(a);
        } else if a == "--" {
            only_ops = true;
        } else if let Some(rest) = a.strip_prefix("--") {
            if let Some((k, v)) = rest.split_once('=') {
                long.push((k, v.to_string()));
            } else {
                long.push((rest, String::new()));
            }
        } else if a.len() > 1
            && a.starts_with('-')
            && !a[1..].chars().next().unwrap().is_ascii_digit()
        {
            for c in a[1..].chars() {
                flags.push(c);
            }
        } else {
            ops.push(a);
        }
    }
    (flags, ops, long)
}

/// Whether one command-line option accepts an argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionValue {
    None,
    Required,
    /// Accept a value only when attached to the option, as in `-i.bak` or `--in-place=.bak`.
    OptionalAttached,
}

/// One command-local option spelling mapped to a canonical key.
#[derive(Clone, Copy, Debug)]
pub struct OptionSpec {
    pub key: &'static str,
    pub short: Option<char>,
    pub long: Option<&'static str>,
    pub value: OptionValue,
}

impl OptionSpec {
    pub const fn flag(key: &'static str, short: Option<char>, long: Option<&'static str>) -> Self {
        Self {
            key,
            short,
            long,
            value: OptionValue::None,
        }
    }

    pub const fn required(
        key: &'static str,
        short: Option<char>,
        long: Option<&'static str>,
    ) -> Self {
        Self {
            key,
            short,
            long,
            value: OptionValue::Required,
        }
    }

    pub const fn optional_attached(
        key: &'static str,
        short: Option<char>,
        long: Option<&'static str>,
    ) -> Self {
        Self {
            key,
            short,
            long,
            value: OptionValue::OptionalAttached,
        }
    }
}

/// One parsed option occurrence. Repeated options remain repeated and ordered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedOption {
    pub key: &'static str,
    pub value: Option<String>,
}

/// Options and operands produced by [`parse_options`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedArgs {
    pub options: Vec<ParsedOption>,
    pub operands: Vec<String>,
}

/// Scan ordinary Unix utility options using a small command-local specification table.
///
/// The scanner handles `--`, short clusters, attached or separate required values, long options,
/// and `--name=value`. It deliberately does not attach command semantics to option names.
pub fn parse_options(args: &[String], specs: &[OptionSpec]) -> Result<ParsedArgs, String> {
    let mut parsed = ParsedArgs::default();
    let mut operands_only = false;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if operands_only || argument == "-" || !argument.starts_with('-') {
            parsed.operands.push(argument.clone());
            index += 1;
            continue;
        }
        if argument == "--" {
            operands_only = true;
            index += 1;
            continue;
        }
        if let Some(long) = argument.strip_prefix("--") {
            let (name, attached) = long
                .split_once('=')
                .map_or((long, None), |(name, value)| (name, Some(value)));
            let Some(spec) = specs.iter().find(|spec| spec.long == Some(name)) else {
                return Err(format!("unsupported option '--{name}'"));
            };
            let value = match spec.value {
                OptionValue::None if attached.is_some() => {
                    return Err(format!("option '--{name}' does not take an argument"));
                }
                OptionValue::None => None,
                OptionValue::OptionalAttached => attached.map(str::to_string),
                OptionValue::Required => match attached {
                    Some(value) => Some(value.to_string()),
                    None => {
                        index += 1;
                        Some(
                            args.get(index)
                                .ok_or_else(|| format!("option '--{name}' requires an argument"))?
                                .clone(),
                        )
                    }
                },
            };
            parsed.options.push(ParsedOption {
                key: spec.key,
                value,
            });
            index += 1;
            continue;
        }

        let cluster = &argument[1..];
        for (offset, short) in cluster.char_indices() {
            let Some(spec) = specs.iter().find(|spec| spec.short == Some(short)) else {
                return Err(format!("unsupported option '-{short}'"));
            };
            let remainder_start = offset + short.len_utf8();
            let remainder = &cluster[remainder_start..];
            let value = match spec.value {
                OptionValue::None => None,
                OptionValue::OptionalAttached => {
                    (!remainder.is_empty()).then(|| remainder.to_string())
                }
                OptionValue::Required if !remainder.is_empty() => Some(remainder.to_string()),
                OptionValue::Required => {
                    index += 1;
                    Some(
                        args.get(index)
                            .ok_or_else(|| format!("option '-{short}' requires an argument"))?
                            .clone(),
                    )
                }
            };
            parsed.options.push(ParsedOption {
                key: spec.key,
                value,
            });
            if spec.value != OptionValue::None {
                break;
            }
        }
        index += 1;
    }
    Ok(parsed)
}

/// Translate the commonly used POSIX basic regular-expression operators to Rust regex syntax.
///
/// BRE makes `+`, `?`, `|`, parentheses, and braces literal unless escaped. Rust's regex syntax
/// uses the extended spelling, so this small translation keeps grep and sed's default mode from
/// silently behaving like `-E`. Backreferences remain an explicit unsupported boundary because
/// Rust's linear-time regex engine does not implement them.
pub fn basic_regex_to_rust(pattern: &str) -> Result<String, String> {
    let mut translated = String::with_capacity(pattern.len());
    let mut characters = pattern.chars();
    let mut in_class = false;
    while let Some(character) = characters.next() {
        if character == '\\' {
            let Some(escaped) = characters.next() else {
                return Err("trailing backslash".to_string());
            };
            if !in_class && escaped.is_ascii_digit() {
                return Err("backreferences are not supported".to_string());
            }
            if !in_class && matches!(escaped, '+' | '?' | '|' | '(' | ')' | '{' | '}') {
                translated.push(escaped);
            } else if !in_class && matches!(escaped, '<' | '>') {
                translated.push_str(r"\b");
            } else {
                translated.push('\\');
                translated.push(escaped);
            }
            continue;
        }
        if character == '[' && !in_class {
            in_class = true;
        } else if character == ']' && in_class {
            in_class = false;
        }
        if !in_class && matches!(character, '+' | '?' | '|' | '(' | ')' | '{' | '}') {
            translated.push('\\');
        }
        translated.push(character);
    }
    Ok(translated)
}

/// Parse a non-negative decimal duration exactly into nanoseconds.
///
/// A missing suffix means seconds.  `ns`, `us`, `ms`, `s`, `m`, `h`, and `d` are accepted.
/// Fractional values are truncated below nanosecond precision, matching the simulator's clock
/// resolution.  Invalid, negative, non-finite, and overflowing values are errors rather than
/// silently becoming zero.
pub fn parse_duration_ns(input: &str) -> Result<u64, String> {
    use crate::clock::{NANOS_PER_MICROSECOND, NANOS_PER_MILLISECOND, NANOS_PER_SECOND};

    let input = input.trim();
    let (number, multiplier): (&str, u64) = if let Some(number) = input.strip_suffix("ns") {
        (number, 1)
    } else if let Some(number) = input.strip_suffix("us") {
        (number, NANOS_PER_MICROSECOND)
    } else if let Some(number) = input.strip_suffix("µs") {
        (number, NANOS_PER_MICROSECOND)
    } else if let Some(number) = input.strip_suffix("ms") {
        (number, NANOS_PER_MILLISECOND)
    } else if let Some(number) = input.strip_suffix('s') {
        (number, NANOS_PER_SECOND)
    } else if let Some(number) = input.strip_suffix('m') {
        (number, 60 * NANOS_PER_SECOND)
    } else if let Some(number) = input.strip_suffix('h') {
        (number, 60 * 60 * NANOS_PER_SECOND)
    } else if let Some(number) = input.strip_suffix('d') {
        (number, 24 * 60 * 60 * NANOS_PER_SECOND)
    } else {
        (input, NANOS_PER_SECOND)
    };
    let number = number.strip_prefix('+').unwrap_or(number);
    if number.is_empty() || number.starts_with('-') {
        return Err(format!("invalid duration {input:?}"));
    }
    let mut pieces = number.split('.');
    let whole = pieces.next().unwrap_or_default();
    let fraction = pieces.next();
    if pieces.next().is_some()
        || (!whole.is_empty() && !whole.bytes().all(|byte| byte.is_ascii_digit()))
        || fraction.is_some_and(|digits| !digits.bytes().all(|byte| byte.is_ascii_digit()))
        || (whole.is_empty() && fraction.is_none_or(str::is_empty))
    {
        return Err(format!("invalid duration {input:?}"));
    }
    let whole = if whole.is_empty() {
        0
    } else {
        whole
            .parse::<u128>()
            .map_err(|_| format!("duration is too large: {input:?}"))?
    };
    let mut nanos = whole
        .checked_mul(u128::from(multiplier))
        .ok_or_else(|| format!("duration is too large: {input:?}"))?;
    if let Some(digits) = fraction.filter(|digits| !digits.is_empty()) {
        // u128 can exactly hold 10^38.  More digits cannot affect nanosecond output for any
        // supported suffix, so discard them only after validating the full decimal above.
        let significant = &digits[..digits.len().min(38)];
        let numerator = significant
            .parse::<u128>()
            .map_err(|_| format!("invalid duration {input:?}"))?;
        let denominator = 10_u128.pow(significant.len() as u32);
        nanos = nanos
            .checked_add(
                numerator
                    .checked_mul(u128::from(multiplier))
                    .ok_or_else(|| format!("duration is too large: {input:?}"))?
                    / denominator,
            )
            .ok_or_else(|| format!("duration is too large: {input:?}"))?;
    }
    u64::try_from(nanos).map_err(|_| format!("duration is too large: {input:?}"))
}

/// Compatibility helper for older command implementations.
pub fn parse_duration(s: &str) -> u64 {
    parse_duration_ns(s)
        .map(|nanos| nanos / crate::clock::NANOS_PER_MILLISECOND)
        .unwrap_or_default()
}

/// Split bytes into owned lines (lossy UTF-8).
pub fn lines_of(s: &[u8]) -> Vec<String> {
    String::from_utf8_lossy(s)
        .lines()
        .map(|l| l.to_string())
        .collect()
}

/// Read each operand file (or stdin when "-") and return concatenated bytes plus any error.
pub fn read_inputs(interp: &Interp, files: &[&String], stdin: &[u8]) -> (Vec<u8>, Vec<String>) {
    let mut data = Vec::new();
    let mut errors = Vec::new();
    if files.is_empty() {
        data.extend_from_slice(stdin);
    } else {
        for f in files {
            if matches!(f.as_str(), "-" | "/dev/stdin") {
                data.extend_from_slice(stdin);
            } else {
                match interp.fs_read(&interp.cwd, f) {
                    Ok(d) => data.extend(d),
                    Err(e) => errors.push(format!("{f}: {e}")),
                }
            }
        }
    }
    (data, errors)
}

/// Interpret common C-style backslash escapes (used by `echo -e`, `printf`).
pub fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some(digit @ '0'..='7') => {
                    let mut octal = String::from(digit);
                    for _ in 1..3 {
                        if chars.peek().is_some_and(|next| matches!(next, '0'..='7')) {
                            octal.push(chars.next().expect("peeked octal digit"));
                        } else {
                            break;
                        }
                    }
                    if let Ok(value) = u8::from_str_radix(&octal, 8) {
                        out.push(char::from(value));
                    }
                }
                Some('a') => out.push('\u{7}'),
                Some('b') => out.push('\u{8}'),
                Some(o) => {
                    out.push('\\');
                    out.push(o);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Shell glob match (`*`, `?`, `[...]`-ish) used by `test =`, `find -name`, case patterns.
pub fn glob_eq(pattern: &str, text: &str) -> bool {
    if !pattern.contains('*') && !pattern.contains('?') && !pattern.contains('[') {
        return pattern == text;
    }
    let mut re = String::from("^");
    for c in pattern.chars() {
        match c {
            '*' => re.push_str(".*"),
            '?' => re.push('.'),
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '\\' => {
                re.push('\\');
                re.push(c);
            }
            _ => re.push(c),
        }
    }
    re.push('$');
    regex::Regex::new(&re)
        .map(|r| r.is_match(text))
        .unwrap_or(false)
}

/// Result of resolving an external command entirely within the modeled filesystem.
pub enum ExecutableLookup {
    Found(String),
    NotExecutable(String),
    NotFound,
}

/// Resolve a command path using the process `PATH`, without consulting the host filesystem.
pub fn resolve_executable(interp: &Interp, name: &str) -> ExecutableLookup {
    let candidates = if name.contains('/') {
        vec![crate::vfs::resolve_against(&interp.cwd, name)]
    } else {
        interp
            .get_var("PATH")
            .unwrap_or_default()
            .split(':')
            .map(|directory| {
                let directory = if directory.is_empty() {
                    interp.cwd.clone()
                } else {
                    crate::vfs::resolve_against(&interp.cwd, directory)
                };
                crate::vfs::resolve_against(&directory, name)
            })
            .collect()
    };
    let mut denied = None;
    for path in candidates {
        let Ok(metadata) = interp.fs_metadata("/", &path, true) else {
            continue;
        };
        if matches!(metadata.kind, crate::vfs::NodeKind::File(_)) && metadata.mode & 0o111 != 0 {
            return ExecutableLookup::Found(path);
        }
        denied.get_or_insert(path);
    }
    denied.map_or(ExecutableLookup::NotFound, ExecutableLookup::NotExecutable)
}

/// Run a previously resolved executable script from the VFS.
pub(crate) fn try_exec_script(
    interp: &mut Interp,
    path: &str,
    args: &[String],
    stdin: &[u8],
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
    resumable: bool,
) -> Option<crate::commands::CommandPoll> {
    let data = interp.vfs.read("/", path).ok()?;
    let text = String::from_utf8_lossy(&data);
    let first = text.lines().next().unwrap_or("");
    let code = if first.starts_with("#!") && first.contains("python") {
        let mut a = vec!["python3.14".to_string(), path.to_string()];
        a.extend(args.iter().cloned());
        if resumable {
            crate::commands::start_child_sequence(
                interp,
                vec![crate::commands::ChildCommand {
                    argv: a,
                    stdin: stdin.to_vec(),
                    cwd: None,
                    environment: None,
                }],
                true,
            )
        } else {
            crate::commands::CommandPoll::Ready(crate::python::run_python(
                interp,
                &a,
                stdin.to_vec(),
                out,
                err,
            ))
        }
    } else if !first.starts_with("#!")
        || first.split_whitespace().next().is_some_and(|interpreter| {
            matches!(interpreter, "#!/bin/sh" | "#!/bin/bash" | "#!/usr/bin/bash")
        })
    {
        if resumable {
            super::proc::start_shell_source(interp, &text, args.to_vec(), Some(stdin.to_vec()), err)
        } else {
            let saved_pos = std::mem::replace(&mut interp.positional, args.to_vec());
            let status = interp.run_script_into(&text, out, err);
            interp.positional = saved_pos;
            crate::commands::CommandPoll::Ready(status)
        }
    } else {
        ewln(err, &format!("{path}: unsupported script interpreter"));
        crate::commands::CommandPoll::Ready(126)
    };
    Some(code)
}

/// Names recognized by `which`/`type`/`command`.
pub const KNOWN_COMMANDS: &[&str] = &[
    "cat",
    "echo",
    "printf",
    "ls",
    "mkdir",
    "rmdir",
    "rm",
    "cp",
    "mv",
    "touch",
    "ln",
    "chmod",
    "chown",
    "head",
    "tail",
    "sort",
    "uniq",
    "cut",
    "tr",
    "grep",
    "rg",
    "sed",
    "awk",
    "gawk",
    "find",
    "seq",
    "wc",
    "cd",
    "pwd",
    "test",
    "true",
    "false",
    "basename",
    "dirname",
    "realpath",
    "readlink",
    "date",
    "sleep",
    "env",
    "printenv",
    "envsubst",
    "export",
    "python3",
    "python3.14",
    "python",
    "jq",
    "base64",
    "sha256sum",
    "curl",
    "tee",
    "xargs",
    "diff",
    "comm",
    "sort",
    "stat",
    "file",
    "mktemp",
    "uname",
    "hostname",
    "whoami",
    "id",
    "groups",
    "nproc",
    "getconf",
    "df",
    "free",
    "ps",
    "git",
    "make",
    "zip",
    "unzip",
];

#[cfg(test)]
mod tests {
    use super::{basic_regex_to_rust, parse_duration_ns, parse_options, OptionSpec, ParsedOption};

    #[test]
    fn durations_are_parsed_exactly_at_nanosecond_resolution() {
        assert_eq!(parse_duration_ns(".0015").unwrap(), 1_500_000);
        assert_eq!(parse_duration_ns("2us").unwrap(), 2_000);
        assert_eq!(parse_duration_ns("1.5m").unwrap(), 90_000_000_000);
        assert_eq!(parse_duration_ns("0.0000000009").unwrap(), 0);
    }

    #[test]
    fn invalid_durations_do_not_silently_become_zero() {
        for invalid in ["", "-1", "NaN", "inf", "1.2.3", "1fortnight"] {
            assert!(parse_duration_ns(invalid).is_err(), "accepted {invalid:?}");
        }
    }

    #[test]
    fn option_scanner_handles_clusters_values_and_operand_boundaries() {
        let specs = [
            OptionSpec::flag("ignore_case", Some('i'), Some("ignore-case")),
            OptionSpec::flag("verbose", Some('v'), Some("verbose")),
            OptionSpec::required("expression", Some('e'), Some("regexp")),
            OptionSpec::optional_attached("in_place", Some('I'), Some("in-place")),
        ];
        let args = [
            "-iv".to_string(),
            "-evalue".to_string(),
            "--in-place=.bak".to_string(),
            "first".to_string(),
            "--".to_string(),
            "-v".to_string(),
        ];
        let parsed = parse_options(&args, &specs).unwrap();
        assert_eq!(
            parsed.options,
            [
                ParsedOption {
                    key: "ignore_case",
                    value: None,
                },
                ParsedOption {
                    key: "verbose",
                    value: None,
                },
                ParsedOption {
                    key: "expression",
                    value: Some("value".into()),
                },
                ParsedOption {
                    key: "in_place",
                    value: Some(".bak".into()),
                },
            ]
        );
        assert_eq!(parsed.operands, ["first", "-v"]);
    }

    #[test]
    fn option_scanner_rejects_unknown_and_malformed_options() {
        let specs = [
            OptionSpec::flag("quiet", Some('q'), Some("quiet")),
            OptionSpec::required("expression", Some('e'), Some("regexp")),
        ];
        assert_eq!(
            parse_options(&["--colour".into()], &specs).unwrap_err(),
            "unsupported option '--colour'"
        );
        assert_eq!(
            parse_options(&["-e".into()], &specs).unwrap_err(),
            "option '-e' requires an argument"
        );
        assert_eq!(
            parse_options(&["--quiet=yes".into()], &specs).unwrap_err(),
            "option '--quiet' does not take an argument"
        );
    }

    #[test]
    fn basic_regex_translation_preserves_the_bre_boundary() {
        assert_eq!(basic_regex_to_rust("a+b?(c)").unwrap(), r"a\+b\?\(c\)");
        assert_eq!(basic_regex_to_rust(r"a\+b\{2,3\}").unwrap(), "a+b{2,3}");
        assert_eq!(basic_regex_to_rust(r"\<word\>").unwrap(), r"\bword\b");
        assert_eq!(
            basic_regex_to_rust(r"\(a\)\1").unwrap_err(),
            "backreferences are not supported"
        );
    }
}
