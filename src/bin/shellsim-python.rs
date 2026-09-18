//! Host-project adapter for experimenting with shellsim's Python interpreter.
//!
//! Host filesystem access ends at project ingestion. The selected tree is copied into a fresh
//! VFS before Python starts, and simulated code receives no path back to the host.

use std::fs;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::exit;

use serde::Serialize;
use shellsim::{python, Environment, Limits, RunOutcome};

const VFS_ROOT: &str = "/work";

#[derive(Default)]
struct Options {
    input: Option<PathBuf>,
    root: Option<PathBuf>,
    entry: Option<PathBuf>,
    pytest: bool,
    json: bool,
    arguments: Vec<String>,
    limits: Option<Limits>,
}

#[derive(Serialize)]
struct Report {
    mounted_root: String,
    mounted_files: usize,
    command: Vec<String>,
    outcome: RunOutcome,
    stdout: String,
    stderr: String,
    unsupported: Vec<String>,
    dropped_unsupported: u64,
}

fn main() {
    let options = parse_options(std::env::args().skip(1).collect());
    let status = run(options).unwrap_or_else(|error| {
        eprintln!("shellsim-python: {error}");
        2
    });
    exit(status);
}

fn run(options: Options) -> Result<i32, String> {
    let input = options
        .input
        .ok_or_else(|| usage("a file or directory is required"))?;
    let input = canonical(&input, "input path")?;
    let metadata = fs::metadata(&input)
        .map_err(|error| format!("cannot inspect {}: {error}", input.display()))?;
    let mount_root = match options.root {
        Some(root) => canonical(&root, "project root")?,
        None if metadata.is_dir() => input.clone(),
        None => input
            .parent()
            .ok_or_else(|| "input file has no parent directory".to_string())?
            .to_path_buf(),
    };
    if !input.starts_with(&mount_root) {
        return Err(format!(
            "input {} is outside project root {}",
            input.display(),
            mount_root.display()
        ));
    }

    let limits = options.limits.unwrap_or_default();
    let mut environment = Environment::with_limits(limits);
    environment.cwd = VFS_ROOT.to_string();
    environment.set_var("PWD", VFS_ROOT);
    let mounted_files =
        shellsim::host_ingest::mount_host_tree(&mut environment, &mount_root, VFS_ROOT)?;

    let pytest = options.pytest || (metadata.is_dir() && options.entry.is_none());
    let command = if pytest {
        let targets = if metadata.is_file() {
            vec![vfs_path(&mount_root, &input)?]
        } else {
            let tests = discover_tests(&input)?;
            if tests.is_empty() {
                return Err(format!(
                    "no test_*.py files found below {}",
                    input.display()
                ));
            }
            tests
                .iter()
                .map(|path| vfs_path(&mount_root, path))
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut command = vec!["python3.14".into(), "-m".into(), "pytest".into()];
        command.extend(targets);
        command
    } else {
        let target = match options.entry {
            Some(entry) if metadata.is_dir() => input.join(entry),
            Some(_) => return Err("--entry is only valid when PATH is a directory".into()),
            None => input,
        };
        let target = canonical(&target, "entry path")?;
        if !target.starts_with(&mount_root) || !target.is_file() {
            return Err("entry must be a file inside the mounted project root".into());
        }
        let mut command = vec!["python3.14".into(), vfs_path(&mount_root, &target)?];
        command.extend(options.arguments);
        command
    };

    let stdin = read_stdin();
    shellsim::sandbox::apply();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = python::run_python(&mut environment, &command, stdin, &mut stdout, &mut stderr);
    let outcome = environment.outcome(status);
    let exit_status = outcome.exit_status;
    if options.json {
        let report = Report {
            mounted_root: VFS_ROOT.into(),
            mounted_files,
            command,
            outcome,
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
            unsupported: environment.unsupported.values(),
            dropped_unsupported: environment.unsupported.dropped(),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|error| format!("could not serialize report: {error}"))?
        );
    } else {
        let _ = std::io::stdout().write_all(&stdout);
        let _ = std::io::stderr().write_all(&stderr);
    }
    Ok(exit_status)
}

fn parse_options(arguments: Vec<String>) -> Options {
    let mut options = Options {
        limits: Some(Limits::default()),
        ..Options::default()
    };
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "-h" | "--help" => {
                println!(
                    "{}",
                    usage("Run Python inside a fresh shellsim environment.")
                );
                exit(0);
            }
            "--pytest" => options.pytest = true,
            "--json" => options.json = true,
            "--root" => options.root = Some(path_value(&arguments, &mut index, "--root")),
            "--entry" => options.entry = Some(path_value(&arguments, &mut index, "--entry")),
            "--cpu" | "--memory" | "--disk" | "--output" => {
                let flag = arguments[index].clone();
                index += 1;
                let value = arguments
                    .get(index)
                    .and_then(|value| parse_quantity(value))
                    .unwrap_or_else(|| fail_usage(&format!("{flag} requires a size")));
                let limits = options.limits.as_mut().expect("default limits installed");
                match flag.as_str() {
                    "--cpu" => limits.cpu = value,
                    "--memory" => limits.memory = value,
                    "--disk" => limits.disk = value,
                    "--output" => limits.output = value,
                    _ => unreachable!(),
                }
            }
            "--" => {
                options.arguments.extend_from_slice(&arguments[index + 1..]);
                break;
            }
            value if value.starts_with('-') => fail_usage(&format!("unknown option {value}")),
            value if options.input.is_none() => options.input = Some(PathBuf::from(value)),
            value => options.arguments.push(value.to_string()),
        }
        index += 1;
    }
    options
}

fn path_value(arguments: &[String], index: &mut usize, flag: &str) -> PathBuf {
    *index += 1;
    arguments
        .get(*index)
        .map(PathBuf::from)
        .unwrap_or_else(|| fail_usage(&format!("{flag} requires a path")))
}

fn canonical(path: &Path, label: &str) -> Result<PathBuf, String> {
    path.canonicalize()
        .map_err(|error| format!("cannot resolve {label} {}: {error}", path.display()))
}

fn discover_tests(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut pending = vec![root.to_path_buf()];
    let mut tests = Vec::new();
    while let Some(path) = pending.pop() {
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("refusing host symlink {}", path.display()));
        }
        if metadata.is_dir() {
            let mut entries = fs::read_dir(&path)
                .map_err(|error| format!("cannot read directory {}: {error}", path.display()))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("cannot read directory {}: {error}", path.display()))?;
            entries.sort_by_key(|entry| entry.file_name());
            for entry in entries.into_iter().rev() {
                let name = entry.file_name();
                let name = name
                    .to_str()
                    .ok_or_else(|| format!("non-UTF-8 host path below {}", path.display()))?;
                if shellsim::host_ingest::DEFAULT_SKIPPED_DIRECTORIES.contains(&name)
                    && entry.file_type().is_ok_and(|kind| kind.is_dir())
                {
                    continue;
                }
                pending.push(entry.path());
            }
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("test_") && name.ends_with(".py"))
        {
            tests.push(path);
        }
    }
    tests.sort();
    Ok(tests)
}

fn destination_path(relative: &Path) -> Result<String, String> {
    if relative.as_os_str().is_empty() {
        return Ok(VFS_ROOT.into());
    }
    let relative = relative
        .to_str()
        .ok_or_else(|| "non-UTF-8 project path".to_string())?;
    Ok(format!("{VFS_ROOT}/{}", relative.replace('\\', "/")))
}

fn vfs_path(root: &Path, host: &Path) -> Result<String, String> {
    let relative = host
        .strip_prefix(root)
        .map_err(|_| format!("{} is outside {}", host.display(), root.display()))?;
    destination_path(relative)
}

fn parse_quantity(value: &str) -> Option<u64> {
    let (number, multiplier) = match value.as_bytes().last().copied() {
        Some(b'k' | b'K') => (&value[..value.len() - 1], 1024),
        Some(b'm' | b'M') => (&value[..value.len() - 1], 1024 * 1024),
        Some(b'g' | b'G') => (&value[..value.len() - 1], 1024 * 1024 * 1024),
        _ => (value, 1),
    };
    number.parse::<u64>().ok()?.checked_mul(multiplier)
}

fn read_stdin() -> Vec<u8> {
    if std::io::stdin().is_terminal() {
        return Vec::new();
    }
    let mut input = Vec::new();
    let _ = std::io::stdin().read_to_end(&mut input);
    input
}

fn usage(message: &str) -> String {
    format!(
        "{message}\nusage: shellsim-python [OPTIONS] PATH [-- ARGS...]\n\
         file PATH runs as a script; directory PATH discovers test_*.py files\n\
         options: --root DIR --entry FILE --pytest --json --cpu N --memory N --disk N --output N"
    )
}

fn fail_usage(message: &str) -> ! {
    eprintln!("shellsim-python: {}", usage(message));
    exit(2)
}
