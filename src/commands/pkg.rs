//! Package managers and build tools.
//!
//! Installers (`pip`/`pip3`/`conda`/`pipx`) don't fetch anything, but they DO record the
//! installed package names into [`Interp::packages`] so bootstrap probes such as `pip list`
//! remain deterministic. Package contents are absent, so installers remain [`Trust::NoOp`].
//!
//! Build tools / compilers (`gcc`/`make`/`cargo`/…) stay pure no-ops: see the
//! `COMPILERS.md` survey for why we deliberately don't simulate native compilation.

use std::collections::HashMap;

use crate::commands::{CommandContext, CommandSpec, Io, Trust};
use crate::interp::Interp;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg;
    // Real-ish: installers that update package state (still NoOp-trust: contents are absent).
    reg(m, &["pip", "pip3", "conda", "pipx"], Trust::NoOp, cmd_pip);
    // Pure no-ops: build tools, compilers, daemons, runtimes we don't simulate.
    reg(
        m,
        &[
            "apt",
            "apt-get",
            "npm",
            "node",
            "cargo",
            "make",
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
        Trust::NoOp,
        cmd_noop,
    );
}

/// Pretend success; the dispatcher already recorded the command as unsupported.
fn cmd_noop(_interp: &mut CommandContext<'_>, _args: &[String], _io: &mut Io) -> i32 {
    0
}

/// `pip install [flags] pkg[==ver] ...` / `pip install -r req.txt` / `conda install ...`.
/// Records package names so the embedded libraries become importable. Other subcommands
/// (`list`, `show`, `freeze`, `uninstall`, …) are handled enough to be plausible.
fn cmd_pip(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    // find the subcommand (first non-flag token)
    let sub = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .map(|s| s.as_str())
        .unwrap_or("");
    match sub {
        "install" => {
            register_install_args(interp, args);
            0
        }
        "list" | "freeze" => {
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
            // `pip show NAME` → minimal metadata if "installed"
            if let Some(name) = args
                .iter()
                .rfind(|a| !a.starts_with('-'))
                .filter(|a| a.as_str() != "show")
            {
                let key = normalize_pkg(name);
                if interp.packages.contains(&key) || interp.packages.contains(name.as_str()) {
                    crate::commands::util::wln(io.out, &format!("Name: {name}"));
                    crate::commands::util::wln(io.out, "Version: 0.0.0");
                }
            }
            0
        }
        _ => 0, // uninstall / download / config / wheel / cache → benign success
    }
}

/// Parse the token stream after `install` and register each concrete package. Handles version
/// specifiers (`pkg==1.2`, `"pkg>=1"`), extras (`pkg[all]`), `-r requirements.txt`, and the
/// common flags (skipping the values of value-taking ones). Shared by pip/conda/uv.
pub fn register_install_args(interp: &mut Interp, args: &[String]) {
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
    let mut i = 0;
    let mut seen_sub = false;
    while i < args.len() {
        let a = &args[i];
        if a == "install" || a == "add" {
            seen_sub = true;
            i += 1;
            continue;
        }
        if a == "-r" || a == "--requirement" {
            if let Some(file) = args.get(i + 1) {
                register_requirements_file(interp, file);
            }
            i += 2;
            continue;
        }
        if VALUE_FLAGS.contains(&a.as_str()) {
            i += 2;
            continue;
        }
        if a.starts_with('-') {
            i += 1;
            continue;
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
            i += 1;
            continue;
        }
        if let Some(name) = package_name_of(a) {
            interp.install_package(&name);
        }
        i += 1;
    }
}

/// Read a requirements file from the VFS and register each requirement line.
pub fn register_requirements_file(interp: &mut Interp, file: &str) {
    if let Ok(src) = interp.vfs.read_string(&interp.cwd, file) {
        for line in src.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() || line.starts_with('-') {
                continue;
            }
            if let Some(name) = package_name_of(line) {
                interp.install_package(&name);
            }
        }
    }
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
