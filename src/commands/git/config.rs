//! Configuration and remote bookkeeping.
//!
//! Two scopes exist: the repository's `.git/config` and a per-user `~/.gitconfig` inside the
//! simulated filesystem. Both use Git's INI format, so a configuration block appended by hand or
//! by another tool is read back correctly. There is no system scope and no network, so remotes
//! are recorded but never contacted.

use std::collections::BTreeMap;

use crate::commands::{CommandContext, Io};

use super::repo;
use super::{repo_error, usage, Globals};

/// Which configuration file a command reads and writes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    Local,
    Global,
}

/// Configuration as a command sees it: user settings, then repository settings, then `-c`
/// overrides, each layer overriding the one before.
pub(crate) fn effective_config(
    ctx: &mut CommandContext<'_>,
    root: &str,
    globals: &Globals,
) -> BTreeMap<String, String> {
    let mut config = repo::load_global_config(ctx);
    config.extend(repo::load_config(ctx, root));
    config.extend(
        globals
            .overrides
            .iter()
            .map(|(key, value)| (key.clone(), value.clone())),
    );
    config
}

/// Identity for new commits, preferring the environment as Git does.
pub(crate) fn identity(
    ctx: &mut CommandContext<'_>,
    root: &str,
    globals: &Globals,
) -> (String, String) {
    let config = effective_config(ctx, root, globals);
    let pick = |variable: &str, key: &str, fallback: &str| {
        ctx.get_var(variable)
            .filter(|value| !value.is_empty())
            .or_else(|| config.get(key).cloned())
            .unwrap_or_else(|| fallback.to_string())
    };
    (
        pick("GIT_AUTHOR_NAME", "user.name", "shellsim"),
        pick("GIT_AUTHOR_EMAIL", "user.email", "shellsim@localhost"),
    )
}

pub(crate) fn git_config(
    ctx: &mut CommandContext<'_>,
    globals: &Globals,
    args: &[String],
    io: &mut Io,
) -> i32 {
    let mut scope = Scope::Local;
    let mut explicit_scope = false;
    let mut operands: Vec<String> = Vec::new();
    for argument in args {
        match argument.as_str() {
            "--local" => explicit_scope = true,
            "--global" => {
                scope = Scope::Global;
                explicit_scope = true;
            }
            "--system" => return usage(io, "the system configuration scope is not available"),
            value => operands.push(value.to_string()),
        }
    }
    // Only the local scope needs a repository; `--global` works anywhere.
    let root = match repo::find_repo_root(ctx) {
        Some(root) => root,
        None if scope == Scope::Global => String::new(),
        None => return repo_error(io),
    };
    let file = match scope {
        Scope::Local => repo::git_path(&root, repo::CONFIG),
        Scope::Global => repo::global_config_path(ctx),
    };
    let mut stored = match scope {
        Scope::Local => repo::load_config(ctx, &root),
        Scope::Global => repo::load_global_config(ctx),
    };
    // An explicit scope reads only that file; otherwise every layer is visible.
    let readable = if explicit_scope || root.is_empty() {
        stored.clone()
    } else {
        effective_config(ctx, &root, globals)
    };
    let operands: Vec<&str> = operands.iter().map(String::as_str).collect();
    match operands.as_slice() {
        ["--list" | "-l"] => {
            for (key, value) in &readable {
                io.out
                    .extend_from_slice(format!("{key}={value}\n").as_bytes());
            }
            0
        }
        ["--get-regexp", pattern] => {
            let Ok(regex) = regex::Regex::new(pattern) else {
                return usage(io, "invalid --get-regexp pattern");
            };
            let mut matched = false;
            for (key, value) in &readable {
                if regex.is_match(key) {
                    matched = true;
                    io.out
                        .extend_from_slice(format!("{key} {value}\n").as_bytes());
                }
            }
            i32::from(!matched)
        }
        ["--get" | "--get-all", key] => emit_value(&readable, key, io),
        ["--unset", key] => {
            if stored.remove(&key.to_ascii_lowercase()).is_none() {
                return 5;
            }
            repo::write_config(ctx, &file, &stored).map_or(1, |()| 0)
        }
        [key] if !key.starts_with('-') => emit_value(&readable, key, io),
        ["--add", key, value] | [key, value] if !key.starts_with('-') => {
            let key = key.to_ascii_lowercase();
            if !repo::valid_config_key(&key)
                || value.len() > 4096
                || value.contains(['\0', '\n', '\r'])
            {
                return usage(io, "invalid config key or value");
            }
            if stored.len() == 256 && !stored.contains_key(&key) {
                return usage(io, "too many config entries");
            }
            stored.insert(key, (*value).to_string());
            repo::write_config(ctx, &file, &stored).map_or(1, |()| 0)
        }
        _ => usage(
            io,
            "usage: git config [--local|--global] [--get|--unset|--list] NAME [VALUE]",
        ),
    }
}

fn emit_value(config: &BTreeMap<String, String>, key: &str, io: &mut Io) -> i32 {
    config.get(&key.to_ascii_lowercase()).map_or(1, |value| {
        io.out.extend_from_slice(format!("{value}\n").as_bytes());
        0
    })
}

/// Record and report remotes. Nothing here contacts a network.
pub(crate) fn git_remote(
    ctx: &mut CommandContext<'_>,
    globals: &Globals,
    args: &[String],
    io: &mut Io,
) -> i32 {
    let Some(root) = repo::find_repo_root(ctx) else {
        return repo_error(io);
    };
    let verbose = args.iter().any(|argument| argument == "-v");
    let operands: Vec<&str> = args
        .iter()
        .filter(|argument| !argument.starts_with('-'))
        .map(String::as_str)
        .collect();
    let mut config = repo::load_config(ctx, &root);
    let file = repo::git_path(&root, repo::CONFIG);
    match operands.as_slice() {
        [] => {
            for (name, url) in remotes(&effective_config(ctx, &root, globals)) {
                if verbose {
                    io.out
                        .extend_from_slice(format!("{name}\t{url} (fetch)\n").as_bytes());
                    io.out
                        .extend_from_slice(format!("{name}\t{url} (push)\n").as_bytes());
                } else {
                    io.out.extend_from_slice(format!("{name}\n").as_bytes());
                }
            }
            0
        }
        ["add", name, url] => {
            let key = format!("remote.{}.url", name.to_ascii_lowercase());
            if !repo::valid_config_key(&key) {
                return usage(io, &format!("invalid remote name: {name}"));
            }
            config.insert(key, (*url).to_string());
            repo::write_config(ctx, &file, &config).map_or(1, |()| 0)
        }
        ["remove" | "rm", name] => {
            let key = format!("remote.{}.url", name.to_ascii_lowercase());
            if config.remove(&key).is_none() {
                io.err
                    .extend_from_slice(format!("error: No such remote: '{name}'\n").as_bytes());
                return 2;
            }
            repo::write_config(ctx, &file, &config).map_or(1, |()| 0)
        }
        ["get-url", name] => match config.get(&format!("remote.{}.url", name.to_ascii_lowercase()))
        {
            Some(url) => {
                io.out.extend_from_slice(format!("{url}\n").as_bytes());
                0
            }
            None => {
                io.err
                    .extend_from_slice(format!("error: No such remote '{name}'\n").as_bytes());
                2
            }
        },
        ["set-url", name, url] => {
            let key = format!("remote.{}.url", name.to_ascii_lowercase());
            if !config.contains_key(&key) {
                io.err
                    .extend_from_slice(format!("error: No such remote '{name}'\n").as_bytes());
                return 2;
            }
            config.insert(key, (*url).to_string());
            repo::write_config(ctx, &file, &config).map_or(1, |()| 0)
        }
        _ => usage(
            io,
            "usage: git remote [-v] [add|remove|get-url|set-url ...]",
        ),
    }
}

fn remotes(config: &BTreeMap<String, String>) -> Vec<(String, String)> {
    config
        .iter()
        .filter_map(|(key, value)| {
            let name = key.strip_prefix("remote.")?.strip_suffix(".url")?;
            Some((name.to_string(), value.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::commands::git::repo::{parse_config, serialize_config};

    #[test]
    fn reads_and_writes_git_ini_format() {
        let text = "[user]\n\tname = Ada\n\temail = ada@example.com\n[remote \"origin\"]\n\turl = https://example.com/r.git\n";
        let config = parse_config(text);
        assert_eq!(config.get("user.name").map(String::as_str), Some("Ada"));
        assert_eq!(
            config.get("remote.origin.url").map(String::as_str),
            Some("https://example.com/r.git")
        );
        // Serialization groups by section in sorted order, which keeps output deterministic.
        assert_eq!(
            String::from_utf8(serialize_config(&config)).unwrap(),
            "[remote \"origin\"]\n\turl = https://example.com/r.git\n[user]\n\temail = ada@example.com\n\tname = Ada\n"
        );
        assert_eq!(
            parse_config(&String::from_utf8(serialize_config(&config)).unwrap()),
            config
        );
    }

    #[test]
    fn accepts_comments_and_bare_boolean_keys() {
        let config = parse_config("# a comment\n[core]\n\tbare\n; another\n");
        assert_eq!(config.get("core.bare").map(String::as_str), Some("true"));
    }
}
