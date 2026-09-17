//! Package managers and build tools.
//!
//! `pip`/`pip3` can activate the bundled `numpy` and `pytest` distributions without network
//! access. Other package managers, packages, native toolchains, and external runtimes are
//! explicit unsupported boundaries.
//!
//! Build tools / compilers (`gcc`/`cargo`/…) fail visibly: see
//! `docs/implementation.md` for why shellsim deliberately does not simulate native compilation.

use std::collections::HashMap;

use crate::commands::{CommandContext, CommandSpec, Io, Trust};
use crate::interp::Interp;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::{reg, reg_unsupported};
    reg(m, &["pip", "pip3"], Trust::Partial, cmd_pip);
    reg_unsupported(
        m,
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

/// Implement the bounded `pip` surface used to activate bundled packages.
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
    let sub = args.get(sub_index).map(String::as_str).unwrap_or("");
    match sub {
        "install" => match register_install_args(interp, args) {
            Ok(_) => 0,
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
            for n in names {
                if sub == "freeze" {
                    crate::commands::util::wln(io.out, &format!("{n}==0.0.0"));
                } else {
                    crate::commands::util::wln(io.out, &format!("{n} 0.0.0"));
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
            let key = normalize_pkg(name);
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
            interp.note_unsupported(&format!("pip:{sub}"));
            crate::commands::util::ewln(io.err, &format!("pip: unsupported command '{sub}'"));
            2
        }
    }
}

/// Parse the token stream after `install` and register each concrete package. Handles version
/// specifiers (`pkg==1.2`, `"pkg>=1"`), extras (`pkg[all]`), `-r requirements.txt`, and the
/// common flags. The operation is validated completely before package state changes.
pub fn register_install_args(interp: &mut Interp, args: &[String]) -> Result<Vec<String>, String> {
    // flags that consume the following token as a value (and aren't packages)
    const VALUE_FLAGS: &[&str] = &[
        "-i",
        "--index-url",
        "--extra-index-url",
        "-f",
        "--find-links",
        "-c",
        "--constraint",
        "-t",
        "--target",
        "-p",
        "--python",
        "--prefix",
        "--root",
        "--platform",
        "--abi",
        "--implementation",
        "--cache-dir",
        "--no-binary",
        "--only-binary",
    ];
    const BOOLEAN_FLAGS: &[&str] = &[
        "-q",
        "--quiet",
        "-U",
        "--upgrade",
        "--no-deps",
        "--disable-pip-version-check",
    ];
    let mut i = 0;
    let mut seen_sub = false;
    let mut packages = Vec::new();
    while i < args.len() {
        let a = &args[i];
        if a == "install" || a == "add" {
            seen_sub = true;
            i += 1;
            continue;
        }
        if a == "-r" || a == "--requirement" {
            let file = args
                .get(i + 1)
                .ok_or_else(|| format!("option '{a}' requires an argument"))?;
            packages.extend(requirements_file_packages(interp, file)?);
            i += 2;
            continue;
        }
        if VALUE_FLAGS.contains(&a.as_str()) {
            if args.get(i + 1).is_none() {
                return Err(format!("option '{a}' requires an argument"));
            }
            i += 2;
            continue;
        }
        if BOOLEAN_FLAGS.contains(&a.as_str()) {
            i += 1;
            continue;
        }
        if a.starts_with('-') {
            return Err(format!("unsupported option '{a}'"));
        }
        if !seen_sub {
            // token before the subcommand (e.g. a `pip`-as-arg) — skip until we see install/add
            i += 1;
            continue;
        }
        // a package spec (or a local path / VCS URL we can't model — skip those)
        if a.starts_with("git+")
            || a.contains("://")
            || a.starts_with('.')
            || a.ends_with(".whl")
            || a.ends_with(".tar.gz")
        {
            return Err(format!("unsupported package source '{a}'"));
        }
        if let Some(name) = package_name_of(a) {
            if !is_bundled_package(&name) {
                return Err(format!("package '{name}' is not bundled"));
            }
            packages.push(name);
        }
        i += 1;
    }
    if !seen_sub {
        return Err("missing install command".to_string());
    }
    if packages.is_empty() {
        return Err("no packages specified".to_string());
    }
    for name in &packages {
        interp.install_package(name);
    }
    Ok(packages)
}

fn requirements_file_packages(interp: &Interp, file: &str) -> Result<Vec<String>, String> {
    let src = interp
        .vfs
        .read_string(&interp.cwd, file)
        .map_err(|error| format!("cannot read requirements file '{file}': {error}"))?;
    let mut packages = Vec::new();
    for line in src.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('-') {
            return Err(format!("unsupported requirement option '{line}'"));
        }
        if let Some(name) = package_name_of(line) {
            if !is_bundled_package(&name) {
                return Err(format!("package '{name}' is not bundled"));
            }
            packages.push(name);
        }
    }
    Ok(packages)
}

fn is_bundled_package(name: &str) -> bool {
    matches!(name, "numpy" | "pytest")
}

/// Strip a requirement spec down to the *import* name we register. Returns None for things we
/// can't map to an importable module (URLs, empties).
pub fn package_name_of(spec: &str) -> Option<String> {
    let s = spec.trim().trim_matches('"').trim_matches('\'');
    if s.is_empty() || s.contains("://") {
        return None;
    }
    // cut at the first version operator / extras bracket / whitespace / semicolon (markers)
    let end = s
        .find(['=', '<', '>', '!', '~', '[', ' ', ';', '@'])
        .unwrap_or(s.len());
    let dist = &s[..end];
    if dist.is_empty() {
        return None;
    }
    Some(normalize_pkg(dist))
}

/// Map a PyPI *distribution* name to the *import* name our libraries register under, lowercased.
fn normalize_pkg(dist: &str) -> String {
    let d = dist.trim().to_lowercase().replace('_', "-");
    match d.as_str() {
        "scikit-learn" => "sklearn",
        "pyyaml" => "yaml",
        "pillow" => "PIL",
        "beautifulsoup4" => "bs4",
        "opencv-python" | "opencv-python-headless" => "cv2",
        "python-dateutil" => "dateutil",
        "msgpack-python" => "msgpack",
        _ => return d.replace('-', "_"),
    }
    .to_string()
}
