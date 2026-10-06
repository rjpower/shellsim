//! Closed pytest CLI subset used by dataset verifiers. Accepted presentation switches match
//! the runner's plain, header-free output; behavior switches are passed to the runner explicitly.

use crate::python::error::{PyError, PyResult};
#[derive(Default)]
pub(super) struct Options {
    pub paths: Vec<String>,
    pub ctrf: Option<String>,
    pub timeout: f64,
    pub continue_collection: bool,
    pub warning_filters: Vec<String>,
}

pub(super) fn parse(args: &[String]) -> PyResult<Options> {
    let mut options = Options::default();
    let mut index = 0;
    while let Some(arg) = args.get(index) {
        if arg == "--" {
            options.paths.extend_from_slice(&args[index + 1..]);
            break;
        }
        let (key, attached) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(k, v)| (k, Some(v)));
        match key {
            "-q" | "-v" | "-rA" | "--disable-warnings" | "--no-header" if attached.is_none() => {}
            "--continue-on-collection-errors" if attached.is_none() => {
                options.continue_collection = true
            }
            "--tb" if matches!(attached, Some("short" | "line")) => {}
            "--color" if attached == Some("no") => {}
            "--ctrf" | "--timeout" | "-W" | "-p" | "--override-ini" | "-o" => {
                let value = match attached {
                    Some(value) => value,
                    None => {
                        index += 1;
                        args.get(index)
                            .ok_or_else(|| format!("{key} requires a value"))?
                    }
                };
                match key {
                    "--ctrf" if !value.is_empty() => options.ctrf = Some(value.to_string()),
                    "--timeout" => {
                        options.timeout = value.parse().map_err(|_| "invalid timeout")?;
                        let ns = options.timeout * 1_000_000_000.0;
                        if !ns.is_finite() || ns < 0.0 || ns >= u64::MAX as f64 {
                            return Err("invalid timeout".into());
                        }
                    }
                    "-W" => options.warning_filters.push(warning_filter(value)?),
                    "-p" if value == "no:cacheprovider" => {}
                    "--override-ini" | "-o" if value == "addopts=" => {}
                    _ => return Err(format!("unsupported pytest option {key} {value}").into()),
                }
            }
            _ if arg.starts_with("-W") => options.warning_filters.push(warning_filter(&arg[2..])?),
            _ if arg.starts_with('-') => {
                return Err(format!("unsupported pytest option {arg}").into())
            }
            _ => options.paths.push(arg.clone()),
        }
        index += 1;
    }
    Ok(options)
}

/// Discover ordinary test modules with a bounded VFS walk, excluding hidden and environment trees.
pub(super) fn discover(
    interp: &mut crate::interp::Interp,
    requested: &[String],
) -> PyResult<Vec<String>> {
    let mut pending = if requested.is_empty() {
        vec![".".to_string()]
    } else {
        requested.iter().rev().cloned().collect()
    };
    let mut paths = Vec::new();
    let mut visited = 0usize;
    while let Some(path) = pending.pop() {
        visited += 1;
        if visited > 10_000 || pending.len() > 10_000 {
            return Err("pytest discovery exceeds the 10000-entry limit".into());
        }
        if !interp.resources.charge_cpu(1) {
            return Err("resource limit exceeded".into());
        }
        if interp.vfs.read_link(&interp.cwd, &path).is_ok() {
            return Err("pytest discovery does not follow symbolic links".into());
        }
        if interp.vfs.is_dir(&interp.cwd, &path) {
            let mut entries = interp
                .fs_list_dir(&interp.cwd, &path)
                .map_err(|error| PyError::from(error.to_string()))?;
            entries.sort();
            if entries
                .len()
                .saturating_add(visited)
                .saturating_add(pending.len())
                > 10_000
            {
                return Err("pytest discovery exceeds the 10000-entry limit".into());
            }
            for name in entries.into_iter().rev() {
                if name.starts_with('.')
                    || matches!(name.as_str(), "__pycache__" | "venv" | "node_modules")
                {
                    continue;
                }
                let child = format!("{}/{name}", path.trim_end_matches('/'));
                if !interp.resources.reserve_memory(child.len() as u64 + 32) {
                    return Err("resource limit exceeded".into());
                }
                if interp.vfs.read_link(&interp.cwd, &child).is_ok() {
                    continue;
                }
                if interp.vfs.is_dir(&interp.cwd, &child)
                    || (name.ends_with(".py")
                        && (name.starts_with("test_") || name.ends_with("_test.py")))
                {
                    pending.push(child);
                }
            }
        } else if interp.vfs.is_file(&interp.cwd, &path) {
            if paths.len() >= super::MAX_RUNNER_FILES {
                return Err("pytest file count limit exceeded".into());
            }
            paths.push(path);
        } else {
            return Err(format!("cannot read pytest input {path}").into());
        }
    }
    Ok(paths)
}

/// Build a filter call from validated fields; arbitrary CLI text never becomes Python code.
fn warning_filter(value: &str) -> PyResult<String> {
    let fields = value.split(':').collect::<Vec<_>>();
    if fields.len() > 5 {
        return Err("invalid warning filter".into());
    }
    let action = fields[0];
    if !matches!(
        action,
        "ignore" | "error" | "always" | "default" | "module" | "once"
    ) {
        return Err("unsupported warning action".into());
    }
    let category = fields
        .get(2)
        .filter(|v| !v.is_empty())
        .copied()
        .unwrap_or("Warning");
    if !matches!(
        category,
        "Warning"
            | "UserWarning"
            | "DeprecationWarning"
            | "PendingDeprecationWarning"
            | "RuntimeWarning"
            | "SyntaxWarning"
            | "FutureWarning"
            | "ImportWarning"
            | "UnicodeWarning"
            | "BytesWarning"
            | "ResourceWarning"
    ) {
        return Err("unsupported warning category".into());
    }
    let line = fields
        .get(4)
        .filter(|v| !v.is_empty())
        .map_or(Ok(0u32), |v| v.parse())
        .map_err(|_| "invalid warning line")?;
    let quote = |text: &str| serde_json::to_string(text).expect("serialize warning field");
    Ok(format!(
        "__shellsim_warnings.filterwarnings({}, {}, {category}, {}, {line})\n",
        quote(action),
        quote(fields.get(1).copied().unwrap_or("")),
        quote(fields.get(3).copied().unwrap_or(""))
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn option_values_and_boundaries_are_validated() {
        for args in [
            vec!["--timeout=nan"],
            vec!["--timeout=-1"],
            vec!["--timeout"],
            vec!["-p", "other"],
            vec!["--color=yes"],
            vec!["-W", "ignore::Unknown"],
        ] {
            assert!(parse(&args.into_iter().map(String::from).collect::<Vec<_>>()).is_err());
        }
        let parsed = parse(&["--".into(), "--test.py".into()]).unwrap();
        assert_eq!(parsed.paths, ["--test.py"]);
    }
}
