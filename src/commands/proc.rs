//! "Process-ish" commands: virtual-clock time/scheduling (sleep/usleep/timeout/date/sync),
//! nested shells (sh/bash/dash/zsh), uv launchers, the Python shim, and jq.

use std::collections::HashMap;

use crate::commands::util::{parse_duration, wln};
use crate::commands::{CommandContext, CommandSpec, Io, Trust};
use crate::interp::Interp;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg;
    // time / scheduling (virtual clock, never blocks)
    reg(m, &["sleep"], Trust::Real, cmd_sleep);
    reg(m, &["usleep"], Trust::Real, cmd_usleep);
    reg(m, &["timeout"], Trust::Real, cmd_timeout);
    reg(m, &["date"], Trust::Real, cmd_date);
    reg(m, &["sync"], Trust::Real, |_, _, _| 0);

    // nested shells / uv-launched verifiers
    reg(m, &["sh", "bash", "dash", "zsh"], Trust::Real, cmd_sh);
    reg(m, &["uv", "uvx", "uvenv"], Trust::Partial, cmd_uv);

    // interpreters
    reg(m, &["python3"], Trust::Partial, cmd_python3);
    reg(m, &["python"], Trust::Partial, cmd_python);
    reg(m, &["python3.11"], Trust::Partial, cmd_python311);
    reg(m, &["python3.12"], Trust::Partial, cmd_python312);
    reg(m, &["python3.13"], Trust::Partial, cmd_python313);
    reg(m, &["jq"], Trust::Partial, cmd_jq);
}

fn cmd_sleep(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    let secs = args.first().map(|s| parse_duration(s)).unwrap_or(0);
    interp.clock.sleep_ms(secs);
    0
}

fn cmd_usleep(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    let us: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    interp.clock.sleep_ms(us / 1000);
    0
}

fn cmd_timeout(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    // timeout DURATION CMD ... : we never actually time out (virtual clock), just run cmd
    let rest: Vec<String> = args.iter().skip(1).cloned().collect();
    if rest.is_empty() {
        return 0;
    }
    let stdin = std::mem::take(&mut io.stdin);
    crate::commands::run(interp, &rest, stdin, io.out, io.err)
}

fn cmd_date(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let secs = interp.clock.unix_secs();
    // handle +FORMAT and %s
    if let Some(fmt) = args.iter().find(|a| a.starts_with('+')) {
        let f = &fmt[1..];
        if f.contains("%s") {
            wln(io.out, &f.replace("%s", &secs.to_string()));
        } else {
            wln(io.out, &format_date(secs, f));
        }
    } else {
        wln(io.out, &format_date(secs, "%a %b %e %H:%M:%S UTC %Y"));
    }
    0
}

fn format_date(secs: u64, fmt: &str) -> String {
    // convert unix secs to UTC fields (proleptic Gregorian)
    let days = (secs / 86400) as i64;
    let tod = secs % 86400;
    let (h, mi, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    let (y, mo, d) = civil_from_days(days);
    fmt.replace("%Y", &format!("{y:04}"))
        .replace("%m", &format!("{mo:02}"))
        .replace("%d", &format!("{d:02}"))
        .replace("%H", &format!("{h:02}"))
        .replace("%M", &format!("{mi:02}"))
        .replace("%S", &format!("{s:02}"))
        .replace("%e", &format!("{d:2}"))
        .replace("%a", "Mon")
        .replace("%b", "Jan")
        .replace("%s", &secs.to_string())
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `sh`/`bash -c "…"` or a script file — run it through our own interpreter.
fn cmd_sh(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-c" => {
                let src = args.get(i + 1).cloned().unwrap_or_default();
                // `sh -c SCRIPT [name [args…]]`: name is $0, the rest are $1+
                let extra = args.get(i + 3..).map(|s| s.to_vec()).unwrap_or_default();
                let saved = std::mem::replace(&mut interp.positional, extra);
                let code = interp.run_script_into(&src, io.out, io.err);
                interp.positional = saved;
                interp.exiting = None;
                return code;
            }
            "-o" => {
                i += 2;
            }
            s if s.starts_with('-') => {
                i += 1;
            }
            s => {
                if let Ok(src) = interp.vfs.read_string(&interp.cwd, s) {
                    let extra = args.get(i + 1..).map(|x| x.to_vec()).unwrap_or_default();
                    let saved = std::mem::replace(&mut interp.positional, extra);
                    let code = interp.run_script_into(&src, io.out, io.err);
                    interp.positional = saved;
                    interp.exiting = None;
                    return code;
                }
                crate::commands::util::ewln(
                    io.err,
                    &format!("{}: {}: No such file or directory", args[0], s),
                );
                return 127;
            }
        }
    }
    // no -c and no file → run stdin as a script
    let src = String::from_utf8_lossy(&io.stdin).into_owned();
    let code = interp.run_script_into(&src, io.out, io.err);
    interp.exiting = None;
    code
}

/// `uv` / `uvx` / `uv run` / `uv tool run`: package-management subcommands update the simulated
/// venv / installed-package state; `run`/`tool run`/`uvx` route the minimal Python shim
/// to our engines so verifiers launched via uv still run.
fn cmd_uv(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let has = |s: &str| args.iter().any(|a| a == s);
    // ---- package management (takes priority so `uv pip install pytest` installs, not runs) ----
    if has("add") {
        crate::commands::pkg::register_install_args(interp, args);
        update_pyproject(interp, &install_specs_after(args, "add"));
        ensure_venv(interp);
        return 0;
    }
    if has("remove") {
        return 0;
    }
    if has("sync") || has("lock") {
        uv_sync(interp);
        ensure_venv(interp);
        return 0;
    }
    if has("pip") && has("install") {
        crate::commands::pkg::register_install_args(interp, args);
        ensure_venv(interp);
        return 0;
    }
    if has("venv") || has("init") {
        ensure_venv(interp);
        return 0;
    }
    // ---- run / tool run / uvx: route the embedded interpreter ----
    if let Some(pos) = args
        .iter()
        .position(|a| a == "pytest" || a.ends_with("/pytest"))
    {
        return crate::python::run_pytest(interp, &args[pos + 1..], io.out, io.err);
    }
    if let Some(pos) = args.iter().position(|a| a == "python" || a == "python3") {
        let mut argv = vec![args[pos].clone()];
        argv.extend(args[pos + 1..].iter().cloned());
        let stdin = std::mem::take(&mut io.stdin);
        return crate::python::run_python(interp, &argv, stdin, io.out, io.err);
    }
    interp.note_unsupported(&args[0]);
    0
}

/// Collect the raw package specs following a `uv add` / `uv pip install` keyword (for pyproject).
fn install_specs_after(args: &[String], kw: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = false;
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == kw {
            seen = true;
            i += 1;
            continue;
        }
        if !seen {
            i += 1;
            continue;
        }
        if a == "--requirement" || a == "-r" {
            i += 2;
            continue;
        }
        if a.starts_with('-') {
            i += 1;
            continue;
        }
        out.push(a.clone());
        i += 1;
    }
    out
}

/// Create the marker files a real `uv`/`venv` would leave behind, so tasks that *inspect* the
/// environment (a `.venv`, a `uv.lock`) see plausible state.
fn ensure_venv(interp: &mut Interp) {
    let cwd = interp.cwd.clone();
    for d in [".venv", ".venv/bin"] {
        let p = crate::vfs::resolve_against(&cwd, d);
        let _ = interp.vfs.mkdir_all("/", &p);
    }
    let py = crate::vfs::resolve_against(&cwd, ".venv/bin/python");
    if !interp.vfs.is_file("/", &py) {
        let _ = interp
            .vfs
            .put_file(&py, b"#!shellsim-venv\n".to_vec(), 0o755);
    }
    let lock = crate::vfs::resolve_against(&cwd, "uv.lock");
    if !interp.vfs.is_file("/", &lock) {
        let _ = interp
            .vfs
            .put_file(&lock, b"# shellsim uv.lock\n".to_vec(), 0o644);
    }
}

/// `uv sync` / `uv lock`: register every dependency declared in `pyproject.toml`.
fn uv_sync(interp: &mut Interp) {
    let cwd = interp.cwd.clone();
    let path = crate::vfs::resolve_against(&cwd, "pyproject.toml");
    if let Ok(content) = interp.vfs.read_string("/", &path) {
        // pull each "name>=ver" / "name==ver" string out of the dependencies arrays
        for spec in extract_dep_specs(&content) {
            if let Some(name) = crate::commands::pkg::package_name_of(&spec) {
                interp.install_package(&name);
            }
        }
    }
    // a requirements.txt next to it, if present
    let req = crate::vfs::resolve_against(&cwd, "requirements.txt");
    if interp.vfs.is_file("/", &req) {
        crate::commands::pkg::register_requirements_file(interp, &req);
    }
}

/// Add specs to `pyproject.toml`'s `[project] dependencies` (creating the file/section if absent).
fn update_pyproject(interp: &mut Interp, specs: &[String]) {
    if specs.is_empty() {
        return;
    }
    let cwd = interp.cwd.clone();
    let path = crate::vfs::resolve_against(&cwd, "pyproject.toml");
    let mut content = interp.vfs.read_string("/", &path).unwrap_or_default();
    if !content.contains("[project]") {
        content.push_str("[project]\nname = \"app\"\nversion = \"0.1.0\"\ndependencies = []\n");
    }
    if !content.contains("dependencies") {
        content.push_str("dependencies = []\n");
    }
    for spec in specs {
        if content.contains(spec.as_str()) {
            continue;
        }
        if let Some(idx) = content.find("dependencies = [") {
            let at = idx + "dependencies = [".len();
            content.insert_str(at, &format!("\n    \"{spec}\","));
        }
    }
    let _ = interp.vfs.put_file(&path, content.into_bytes(), 0o644);
}

/// Extract `name[op ver]` dependency strings from a pyproject's dependencies arrays.
fn extract_dep_specs(toml_src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_deps = false;
    for line in toml_src.lines() {
        let t = line.trim();
        if t.starts_with("dependencies") && t.contains('[') {
            in_deps = true;
        }
        if in_deps {
            for part in t.split(['"', '\'']) {
                let p = part.trim().trim_end_matches(',');
                // a dependency spec starts with a letter (name); package_name_of strips any
                // version operator. Skips the `dependencies = [` token and bare punctuation.
                if !p.is_empty()
                    && p.chars()
                        .next()
                        .map(|c| c.is_ascii_alphabetic())
                        .unwrap_or(false)
                    && p != "dependencies"
                {
                    out.push(p.to_string());
                }
            }
            if t.contains(']') {
                in_deps = false;
            }
        }
    }
    out
}

fn cmd_python3(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    python_impl(interp, "python3", args, io)
}
fn cmd_python(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    python_impl(interp, "python", args, io)
}
fn cmd_python311(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    python_impl(interp, "python3.11", args, io)
}
fn cmd_python312(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    python_impl(interp, "python3.12", args, io)
}
fn cmd_python313(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    python_impl(interp, "python3.13", args, io)
}

fn python_impl(interp: &mut Interp, name: &str, args: &[String], io: &mut Io) -> i32 {
    // run_python expects argv[0] to be the program name (it skips it).
    let mut argv = vec![name.to_string()];
    argv.extend(args.iter().cloned());
    let stdin = std::mem::take(&mut io.stdin);
    crate::python::run_python(interp, &argv, stdin, io.out, io.err)
}

fn cmd_jq(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let stdin = std::mem::take(&mut io.stdin);
    crate::jqcmd::jq(interp, args, stdin, io.out, io.err)
}
