//! Configuration and remote bookkeeping.
//!
//! Two scopes exist: the repository's `.git/config` and a per-user `~/.gitconfig` inside the
//! simulated filesystem. Both use Git's INI format, so a configuration block appended by hand or
//! by another tool is read back correctly. There is no system scope and no network, so remotes
//! are recorded but never contacted.

use crate::commands::Io;
use crate::syscalls::System;

use super::repo::{self, Config};
use super::{fatal, repo_error, usage, Globals};

/// Which configuration file a command reads and writes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    Local,
    Global,
}

/// Configuration as a command sees it: user settings, then repository settings, then `-c`
/// overrides, each layer overriding the one before.
pub(crate) fn effective_config(system: &mut dyn System, root: &str, globals: &Globals) -> Config {
    let mut config = repo::load_global_config(system);
    config.extend(repo::load_config(system, root));
    config.extend(
        globals
            .overrides
            .iter()
            .map(|(key, value)| (key.clone(), vec![value.clone()])),
    );
    config
}

/// The single value a read of `key` yields, which is the last one recorded.
fn value_of(config: &Config, key: &str) -> Option<String> {
    repo::config_value(config, &key.to_ascii_lowercase()).map(str::to_string)
}

/// Identity for new commits, preferring the environment as Git does.
pub(crate) fn identity(system: &mut dyn System, root: &str, globals: &Globals) -> (String, String) {
    let config = effective_config(system, root, globals);
    let environment = system.environment();
    let pick = |variable: &str, key: &str, fallback: &str| {
        environment
            .get(variable)
            .filter(|value| !value.is_empty())
            .cloned()
            .or_else(|| value_of(&config, key))
            .unwrap_or_else(|| fallback.to_string())
    };
    (
        pick("GIT_AUTHOR_NAME", "user.name", "shellsim"),
        pick("GIT_AUTHOR_EMAIL", "user.email", "shellsim@localhost"),
    )
}

pub(crate) fn git_config(
    system: &mut dyn System,
    globals: &Globals,
    args: &[String],
    io: &mut Io,
) -> i32 {
    // `git config` writes its verbs as options, so these reach the match on operands below.
    const ACTIONS: &[&str] = &[
        "--list",
        "-l",
        "--get",
        "--get-all",
        "--get-regexp",
        "--unset",
        "--unset-all",
        "--add",
    ];
    let mut scope = Scope::Local;
    let mut explicit_scope = false;
    let mut show_origin = false;
    let mut as_boolean = false;
    let mut operands: Vec<String> = Vec::new();
    for argument in args {
        match argument.as_str() {
            "--local" => explicit_scope = true,
            "--global" => {
                scope = Scope::Global;
                explicit_scope = true;
            }
            "--system" => return usage(io, "the system configuration scope is not available"),
            "--show-origin" => show_origin = true,
            "--bool" | "--type=bool" => as_boolean = true,
            // The default type, so asking for it changes nothing.
            "--type=string" => {}
            // What is left is either one of the actions below, which read as operands because
            // `git config` spells them like options, or something this subset does not have.
            value if value.starts_with('-') && !ACTIONS.contains(&value) => {
                return usage(io, &format!("unsupported config option: {value}"))
            }
            value => operands.push(value.to_string()),
        }
    }
    // Only the local scope needs a repository; `--global` works anywhere.
    let root = match repo::find_repo_root(system) {
        Some(root) => root,
        None if scope == Scope::Global => String::new(),
        None => return repo_error(io),
    };
    let file = match scope {
        Scope::Local => repo::git_path(&root, repo::CONFIG),
        Scope::Global => repo::global_config_path(system),
    };
    let mut stored = match scope {
        Scope::Local => repo::load_config(system, &root),
        Scope::Global => repo::load_global_config(system),
    };
    // An explicit scope reads only that file; otherwise every layer is visible.
    let readable = if explicit_scope || root.is_empty() {
        stored.clone()
    } else {
        effective_config(system, &root, globals)
    };
    let operands: Vec<&str> = operands.iter().map(String::as_str).collect();
    // `--show-origin` prefixes each line with the file the value came from.
    let origin = if show_origin {
        format!("file:{file}\t")
    } else {
        String::new()
    };
    match operands.as_slice() {
        ["--list" | "-l"] => {
            for (key, values) in &readable {
                for value in values {
                    io.print(&format!("{origin}{key}={value}\n"));
                }
            }
            0
        }
        ["--get-regexp", pattern] => {
            let Ok(regex) = regex::Regex::new(pattern) else {
                return fatal(io, "invalid --get-regexp pattern");
            };
            let mut matched = false;
            for (key, values) in &readable {
                if !regex.is_match(key) {
                    continue;
                }
                matched = true;
                for value in values {
                    io.print(&format!("{origin}{key} {value}\n"));
                }
            }
            i32::from(!matched)
        }
        ["--get-all", key] => {
            let Some(values) = readable.get(&key.to_ascii_lowercase()) else {
                return 1;
            };
            for value in values {
                emit_one(&origin, value, as_boolean, io);
            }
            0
        }
        ["--get", key] | [key] if !key.starts_with('-') => {
            match repo::config_value(&readable, &key.to_ascii_lowercase()) {
                Some(value) => {
                    emit_one(&origin, value, as_boolean, io);
                    0
                }
                None => 1,
            }
        }
        ["--unset" | "--unset-all", key] => {
            if stored.remove(&key.to_ascii_lowercase()).is_none() {
                return 5;
            }
            repo::write_config(system, &file, &stored).map_or(1, |()| 0)
        }
        ["--add", key, value] | [key, value] if !key.starts_with('-') => {
            let adding = operands[0] == "--add";
            let key = key.to_ascii_lowercase();
            if !repo::valid_config_key(&key)
                || value.len() > 4096
                || value.contains(['\0', '\n', '\r'])
            {
                return fatal(io, "invalid config key or value");
            }
            if stored.len() == 256 && !stored.contains_key(&key) {
                return fatal(io, "too many config entries");
            }
            let entry = stored.entry(key).or_default();
            if !adding {
                entry.clear();
            }
            entry.push((*value).to_string());
            repo::write_config(system, &file, &stored).map_or(1, |()| 0)
        }
        _ => usage(
            io,
            "usage: git config [--local|--global] [--get|--unset|--list] NAME [VALUE]",
        ),
    }
}

fn emit_one(origin: &str, value: &str, as_boolean: bool, io: &mut Io) {
    let rendered = if as_boolean {
        // Git's boolean reading: these spellings are true, everything else is false.
        let truthy = matches!(
            value.to_ascii_lowercase().as_str(),
            "true" | "yes" | "on" | "1"
        ) || value.is_empty();
        truthy.to_string()
    } else {
        value.to_string()
    };
    io.print(&format!("{origin}{rendered}\n"));
}

/// The command an alias expands to, split into words.
pub(crate) fn alias(system: &mut dyn System, globals: &Globals, name: &str) -> Option<Vec<String>> {
    let root = repo::find_repo_root(system).unwrap_or_default();
    let config = effective_config(system, &root, globals);
    let expansion = repo::config_value(&config, &format!("alias.{}", name.to_ascii_lowercase()))?;
    // Shell aliases (`!cmd`) would need host execution, which this simulation does not provide.
    if expansion.starts_with('!') {
        return None;
    }
    Some(
        expansion
            .split_whitespace()
            .map(str::to_string)
            .collect::<Vec<_>>(),
    )
    .filter(|words: &Vec<String>| !words.is_empty())
}

/// Record and report remotes. Nothing here contacts a network.
pub(crate) fn git_remote(
    system: &mut dyn System,
    globals: &Globals,
    args: &[String],
    io: &mut Io,
) -> i32 {
    let Some(root) = repo::find_repo_root(system) else {
        return repo_error(io);
    };
    let verbose = args.iter().any(|argument| argument == "-v");
    let operands: Vec<&str> = args
        .iter()
        .filter(|argument| !argument.starts_with('-'))
        .map(String::as_str)
        .collect();
    let mut config = repo::load_config(system, &root);
    let file = repo::git_path(&root, repo::CONFIG);
    match operands.as_slice() {
        [] => {
            for (name, url) in remotes(&effective_config(system, &root, globals)) {
                if verbose {
                    io.print(&format!("{name}\t{url} (fetch)\n"));
                    io.print(&format!("{name}\t{url} (push)\n"));
                } else {
                    io.print(&format!("{name}\n"));
                }
            }
            0
        }
        ["add", name, url] => {
            let key = format!("remote.{}.url", name.to_ascii_lowercase());
            if !repo::valid_config_key(&key) {
                return fatal(io, &format!("invalid remote name: {name}"));
            }
            config.insert(key, vec![(*url).to_string()]);
            repo::write_config(system, &file, &config).map_or(1, |()| 0)
        }
        ["remove" | "rm", name] => {
            let key = format!("remote.{}.url", name.to_ascii_lowercase());
            if config.remove(&key).is_none() {
                io.print_err(&format!("error: No such remote: '{name}'\n"));
                return 2;
            }
            repo::write_config(system, &file, &config).map_or(1, |()| 0)
        }
        ["get-url", name] => match repo::config_value(
            &config,
            &format!("remote.{}.url", name.to_ascii_lowercase()),
        ) {
            Some(url) => {
                io.print(&format!("{url}\n"));
                0
            }
            None => {
                io.print_err(&format!("error: No such remote '{name}'\n"));
                2
            }
        },
        ["set-url", name, url] => {
            let key = format!("remote.{}.url", name.to_ascii_lowercase());
            if !config.contains_key(&key) {
                io.print_err(&format!("error: No such remote '{name}'\n"));
                return 2;
            }
            config.insert(key, vec![(*url).to_string()]);
            repo::write_config(system, &file, &config).map_or(1, |()| 0)
        }
        _ => usage(
            io,
            "usage: git remote [-v] [add|remove|get-url|set-url ...]",
        ),
    }
}

fn remotes(config: &Config) -> Vec<(String, String)> {
    config
        .iter()
        .filter_map(|(key, values)| {
            let name = key.strip_prefix("remote.")?.strip_suffix(".url")?;
            Some((name.to_string(), values.last()?.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::commands::git::repo::{config_value, parse_config, serialize_config};

    #[test]
    fn reads_and_writes_git_ini_format() {
        let text = "[user]\n\tname = Ada\n\temail = ada@example.com\n[remote \"origin\"]\n\turl = https://example.com/r.git\n";
        let config = parse_config(text);
        assert_eq!(config_value(&config, "user.name"), Some("Ada"));
        assert_eq!(
            config_value(&config, "remote.origin.url"),
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
        assert_eq!(config_value(&config, "core.bare"), Some("true"));
    }
}
