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

/// Parse a duration string into milliseconds; supports s/m/h/d suffix and fractional seconds.
pub fn parse_duration(s: &str) -> u64 {
    let s = s.trim();
    let (num, mult) = if let Some(n) = s.strip_suffix("ms") {
        (n, 1.0)
    } else if let Some(n) = s.strip_suffix('s') {
        (n, 1000.0)
    } else if let Some(n) = s.strip_suffix('m') {
        (n, 60_000.0)
    } else if let Some(n) = s.strip_suffix('h') {
        (n, 3_600_000.0)
    } else if let Some(n) = s.strip_suffix('d') {
        (n, 86_400_000.0)
    } else {
        (s, 1000.0)
    };
    (num.parse::<f64>().unwrap_or(0.0) * mult) as u64
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
            if f.as_str() == "-" {
                data.extend_from_slice(stdin);
            } else {
                match interp.vfs.read(&interp.cwd, f) {
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
                Some('0') => out.push('\0'),
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

/// Run a script file in the VFS if it exists and looks like a shell/python script.
pub fn try_exec_script(
    interp: &mut Interp,
    name: &str,
    args: &[String],
    stdin: &[u8],
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
) -> Option<i32> {
    let path = if name.contains('/') {
        crate::vfs::resolve_against(&interp.cwd, name)
    } else {
        return None;
    };
    let data = interp.vfs.read("/", &path).ok()?;
    let text = String::from_utf8_lossy(&data);
    let saved_pos = std::mem::replace(&mut interp.positional, args.to_vec());
    let first = text.lines().next().unwrap_or("");
    let code = if first.starts_with("#!") && (first.contains("python")) {
        // python script
        let mut a = vec![name.to_string()];
        a.extend(args.iter().cloned());
        crate::python::run_python(interp, &a, stdin.to_vec(), out, err)
    } else {
        interp.run_script_into(&text, out, err)
    };
    interp.positional = saved_pos;
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
    "sed",
    "awk",
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
    "export",
    "python3",
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
];
