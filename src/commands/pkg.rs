//! Package managers and build tools.
//!
//! pip and pip3 can activate the bundled numpy and pytest distributions without network access.
//! Other package managers, packages, native toolchains, and external runtimes are explicit
//! unsupported boundaries.

use std::collections::HashMap;

use crate::commands::options::{parse_options, OptionSpec};
use crate::commands::{CommandContext, CommandSpec, Io, Trust};
use crate::interp::Interp;

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    use super::{reg, reg_unsupported};
    reg(commands, &["pip", "pip3"], Trust::Partial, cmd_pip);
    reg_unsupported(
        commands,
        &[
            "conda",
            "pipx",
            "apt",
            "apt-get",
            "npm",
            "node",
            "cargo",
            "cmake",
            "gcc",
            "g++",
            "cc",
            "clang",
            "mvn",
            "gradle",
            "javac",
            "java",
            "docker",
            "systemctl",
            "service",
            "uvicorn",
            "gunicorn",
            "flask",
            "ld",
            "ar",
            "rustc",
            "go",
        ],
    );
}

/// Implement the bounded pip surface used to activate bundled packages.
fn cmd_pip(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut sub_index = 0;
    while args.get(sub_index).is_some_and(|argument| {
        matches!(
            argument.as_str(),
            "-q" | "--quiet" | "--disable-pip-version-check"
        )
    }) {
        sub_index += 1;
    }
    let subcommand = args.get(sub_index).map(String::as_str).unwrap_or("");
    match subcommand {
        "install" => match install_args(interp, &args[sub_index + 1..]) {
            Ok(()) => 0,
            Err(error) => {
                interp.note_unsupported(&format!("pip:{error}"));
                crate::commands::util::ewln(io.err, &format!("pip: {error}"));
                1
            }
        },
        "list" | "freeze" => {
            if args.len() != sub_index + 1 {
                crate::commands::util::ewln(io.err, "pip: unsupported option");
                return 2;
            }
            let mut names: Vec<&String> = interp.packages.iter().collect();
            names.sort();
            for name in names {
                if subcommand == "freeze" {
                    crate::commands::util::wln(io.out, &format!("{name}==0.0.0"));
                } else {
                    crate::commands::util::wln(io.out, &format!("{name} 0.0.0"));
                }
            }
            0
        }
        "show" => {
            let Some(name) = args
                .get(sub_index + 1)
                .filter(|_| args.len() == sub_index + 2)
            else {
                crate::commands::util::ewln(io.err, "pip: show requires one package name");
                return 2;
            };
            let key = normalize_package(name);
            if interp.packages.contains(&key) || interp.packages.contains(name.as_str()) {
                crate::commands::util::wln(io.out, &format!("Name: {name}"));
                crate::commands::util::wln(io.out, "Version: 0.0.0");
                0
            } else {
                crate::commands::util::ewln(
                    io.err,
                    &format!("WARNING: Package(s) not found: {name}"),
                );
                1
            }
        }
        "--help" | "-h" | "" => {
            crate::commands::util::wln(io.out, "usage: pip {install,list,freeze,show} [OPTIONS]");
            0
        }
        _ => {
            interp.note_unsupported(&format!("pip:{subcommand}"));
            crate::commands::util::ewln(
                io.err,
                &format!("pip: unsupported command '{subcommand}'"),
            );
            2
        }
    }
}

/// Validate package arguments completely, then activate every requested bundled package.
pub(crate) fn install_args(interp: &mut Interp, args: &[String]) -> Result<(), String> {
    let packages = resolve_install_args(interp, args)?;
    install_packages(interp, &packages);
    Ok(())
}

/// Resolve package arguments without changing interpreter state.
pub(crate) fn resolve_install_args(
    interp: &Interp,
    args: &[String],
) -> Result<Vec<String>, String> {
    const OPTIONS: &[OptionSpec] = &[
        OptionSpec::required("requirement", Some('r'), Some("requirement")),
        OptionSpec::flag("quiet", Some('q'), Some("quiet")),
        OptionSpec::flag("no_deps", None, Some("no-deps")),
        OptionSpec::flag(
            "disable_version_check",
            None,
            Some("disable-pip-version-check"),
        ),
    ];
    let parsed = parse_options(args, OPTIONS)?;
    let mut packages = resolve_package_specs(&parsed.operands)?;
    for option in parsed.options {
        match option.key {
            "requirement" => packages.extend(resolve_requirements_file(
                interp,
                &option.value.expect("required option value"),
            )?),
            "quiet" | "no_deps" | "disable_version_check" => {}
            _ => unreachable!("option keys come from OPTIONS"),
        }
    }
    if packages.is_empty() {
        return Err("no packages specified".to_string());
    }
    Ok(packages)
}

/// Resolve a requirements file without changing interpreter state.
pub(crate) fn resolve_requirements_file(
    interp: &Interp,
    file: &str,
) -> Result<Vec<String>, String> {
    let source = interp
        .vfs
        .read_string(&interp.cwd, file)
        .map_err(|error| format!("cannot read requirements file '{file}': {error}"))?;
    let mut packages = Vec::new();
    for line in source.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('-') {
            return Err(format!("unsupported requirement option '{line}'"));
        }
        packages.extend(resolve_package_specs(&[line.to_string()])?);
    }
    Ok(packages)
}

/// Resolve distribution specifications against the Python runtime's bundled distribution table.
pub(crate) fn resolve_package_specs(specs: &[String]) -> Result<Vec<String>, String> {
    let mut packages = Vec::with_capacity(specs.len());
    for spec in specs {
        if spec.starts_with("git+")
            || spec.contains("://")
            || spec.starts_with('.')
            || spec.ends_with(".whl")
            || spec.ends_with(".tar.gz")
        {
            return Err(format!("unsupported package source '{spec}'"));
        }
        let Some(name) = package_name_of(spec) else {
            return Err(format!("invalid package specification '{spec}'"));
        };
        if !crate::python::is_bundled_distribution(&name) {
            return Err(format!("package '{name}' is not bundled"));
        }
        packages.push(name);
    }
    Ok(packages)
}

/// Activate package names that have already passed bundled-distribution validation.
pub(crate) fn install_packages(interp: &mut Interp, packages: &[String]) {
    for name in packages {
        interp.install_package(name);
    }
}

/// Strip a requirement spec down to the import name registered by shellsim.
fn package_name_of(spec: &str) -> Option<String> {
    let spec = spec.trim().trim_matches('"').trim_matches('\'');
    if spec.is_empty() || spec.contains("://") {
        return None;
    }
    let end = spec
        .find(['=', '<', '>', '!', '~', '[', ' ', ';', '@'])
        .unwrap_or(spec.len());
    let distribution = &spec[..end];
    (!distribution.is_empty()).then(|| normalize_package(distribution))
}

fn normalize_package(distribution: &str) -> String {
    distribution.trim().to_lowercase().replace('-', "_")
}
