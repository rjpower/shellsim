//! "Process-ish" commands: virtual-clock time/scheduling (sleep/usleep/timeout/date/sync),
//! nested shells (sh/bash/dash/zsh), uv launchers, the Python shim, and jq.

use std::collections::HashMap;

use crate::clock::{
    BlockOutcome, EventKind, MAIN_TASK_ID, NANOS_PER_MICROSECOND, NANOS_PER_SECOND,
};
use crate::commands::util::{ewln, parse_duration_ns, wln};
use crate::commands::{CommandContext, CommandPoll, CommandResume, CommandSpec, Io, Trust};
use crate::interp::Interp;
use crate::scheduler::WaitReason;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::{reg, reg_resumable};
    // time / scheduling (virtual clock, never blocks)
    reg_resumable(m, &["sleep"], Trust::Real, cmd_sleep, start_sleep);
    reg_resumable(m, &["usleep"], Trust::Real, cmd_usleep, start_usleep);
    reg(m, &["timeout"], Trust::Partial, cmd_timeout);
    reg(m, &["date"], Trust::Real, cmd_date);
    reg(m, &["sync"], Trust::Real, |_, _, _| 0);

    // nested shells / uv-launched verifiers
    reg_resumable(
        m,
        &["sh", "bash", "dash", "zsh"],
        Trust::Real,
        cmd_sh,
        start_sh,
    );
    reg(m, &["uv", "uvx", "uvenv"], Trust::Partial, cmd_uv);

    // interpreters
    reg(m, &["python3"], Trust::Partial, cmd_python3);
    reg(m, &["python"], Trust::Partial, cmd_python);
    reg(m, &["python3.11"], Trust::Partial, cmd_python311);
    reg(m, &["python3.12"], Trust::Partial, cmd_python312);
    reg(m, &["python3.13"], Trust::Partial, cmd_python313);
    reg(m, &["python3.14"], Trust::Partial, cmd_python314);
    reg(m, &["pytest"], Trust::Partial, cmd_pytest);
    reg(m, &["jq"], Trust::Partial, cmd_jq);
}

fn start_sleep(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    if args.is_empty() {
        return CommandPoll::Ready(1);
    }
    let duration = args.iter().try_fold(0_u64, |total, argument| {
        parse_duration_ns(argument).and_then(|part| {
            total
                .checked_add(part)
                .ok_or_else(|| "duration is too large".to_string())
        })
    });
    start_timer(interp, duration, "sleep", io)
}

fn start_usleep(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let Some(argument) = args.first() else {
        ewln(io.err, "usleep: missing operand");
        return CommandPoll::Ready(1);
    };
    let duration = argument
        .parse::<u64>()
        .ok()
        .and_then(|micros| micros.checked_mul(NANOS_PER_MICROSECOND))
        .ok_or_else(|| format!("invalid time interval {argument:?}"));
    start_timer(interp, duration, "usleep", io)
}

fn start_timer(
    interp: &mut CommandContext<'_>,
    duration: Result<u64, String>,
    command: &str,
    io: &mut Io,
) -> CommandPoll {
    let duration = match duration {
        Ok(duration) => duration,
        Err(error) => {
            ewln(io.err, &format!("{command}: {error}"));
            return CommandPoll::Ready(1);
        }
    };
    let pid = interp.process.pid;
    match interp.clock.schedule_wake_after(u64::from(pid), duration) {
        Ok(event) => CommandPoll::Blocked(
            WaitReason::Timer(event.deadline_ns()),
            CommandResume::Status(0),
        ),
        Err(error) => {
            ewln(io.err, &format!("{command}: {error}"));
            CommandPoll::Ready(1)
        }
    }
}

fn cmd_sleep(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    if args.is_empty() {
        return 1;
    }
    let duration = args.iter().try_fold(0_u64, |total, argument| {
        parse_duration_ns(argument).and_then(|part| {
            total
                .checked_add(part)
                .ok_or_else(|| "duration is too large".to_string())
        })
    });
    let duration = match duration {
        Ok(duration) => duration,
        Err(error) => {
            ewln(_io.err, &format!("sleep: {error}"));
            return 1;
        }
    };
    match interp.clock.block_task(MAIN_TASK_ID, duration) {
        Ok(BlockOutcome::Completed) => 0,
        Ok(BlockOutcome::Interrupted(event)) => {
            interp.deadline_interrupt = Some(event.id);
            124
        }
        Err(error) => {
            ewln(_io.err, &format!("sleep: {error}"));
            1
        }
    }
}

fn cmd_usleep(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(argument) = args.first() else {
        ewln(io.err, "usleep: missing operand");
        return 1;
    };
    let duration = match argument
        .parse::<u64>()
        .ok()
        .and_then(|micros| micros.checked_mul(NANOS_PER_MICROSECOND))
    {
        Some(duration) => duration,
        None => {
            ewln(
                io.err,
                &format!("usleep: invalid time interval {argument:?}"),
            );
            return 1;
        }
    };
    match interp.clock.block_task(MAIN_TASK_ID, duration) {
        Ok(BlockOutcome::Completed) => 0,
        Ok(BlockOutcome::Interrupted(event)) => {
            interp.deadline_interrupt = Some(event.id);
            124
        }
        Err(error) => {
            ewln(io.err, &format!("usleep: {error}"));
            1
        }
    }
}

fn cmd_timeout(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    // Capability-free subset: deadline semantics are real; signal selection is accepted but the
    // synchronous executor reports GNU timeout's conventional 124 instead of modeling signals.
    let mut index = 0;
    let mut preserve_status = false;
    while index < args.len() {
        match args[index].as_str() {
            "--preserve-status" => {
                preserve_status = true;
                index += 1;
            }
            "--foreground" => index += 1,
            "-s" | "--signal" | "-k" | "--kill-after" => index += 2,
            option if option.starts_with("--signal=") || option.starts_with("--kill-after=") => {
                index += 1
            }
            _ => break,
        }
    }
    let Some(duration_argument) = args.get(index) else {
        ewln(io.err, "timeout: missing operand");
        return 125;
    };
    let duration = match parse_duration_ns(duration_argument) {
        Ok(duration) => duration,
        Err(error) => {
            ewln(io.err, &format!("timeout: {error}"));
            return 125;
        }
    };
    let rest: Vec<String> = args.iter().skip(index + 1).cloned().collect();
    if rest.is_empty() {
        ewln(io.err, "timeout: missing command");
        return 125;
    }
    // GNU timeout treats zero as disabling the timeout.
    let deadline = if duration == 0 {
        None
    } else {
        match interp
            .clock
            .schedule_after(duration, EventKind::Deadline { task: MAIN_TASK_ID })
        {
            Ok(event) => Some(event),
            Err(error) => {
                ewln(io.err, &format!("timeout: {error}"));
                return 125;
            }
        }
    };
    let stdin = std::mem::take(&mut io.stdin);
    let status = crate::commands::run(interp, &rest, stdin, io.out, io.err);
    let interrupted = interp.deadline_interrupt;
    if let Some(deadline) = deadline {
        interp.clock.cancel(deadline);
        if interrupted == Some(deadline) {
            interp.deadline_interrupt = None;
            return if preserve_status { status } else { 124 };
        }
    }
    // An enclosing timeout fired.  Preserve its interrupt so that its own command frame unwinds.
    if interrupted.is_some() {
        124
    } else {
        status
    }
}

fn cmd_date(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let unix_ns = match interp.clock.wall_time_ns() {
        Ok(value) => value,
        Err(error) => {
            ewln(io.err, &format!("date: {error}"));
            return 1;
        }
    };
    let rendered = if let Some(fmt) = args.iter().find(|a| a.starts_with('+')) {
        format_date(unix_ns, &fmt[1..])
    } else {
        format_date(unix_ns, "%a %b %e %H:%M:%S UTC %Y")
    };
    match rendered {
        Ok(rendered) => {
            wln(io.out, &rendered);
            0
        }
        Err(error) => {
            ewln(io.err, &format!("date: {error}"));
            1
        }
    }
}

fn format_date(unix_ns: i128, fmt: &str) -> Result<String, String> {
    // convert unix secs to UTC fields (proleptic Gregorian)
    let secs = unix_ns.div_euclid(i128::from(NANOS_PER_SECOND));
    let subsecond_ns = unix_ns.rem_euclid(i128::from(NANOS_PER_SECOND));
    let days = i64::try_from(secs.div_euclid(86_400))
        .map_err(|_| "timestamp is outside the supported calendar range".to_string())?;
    let tod = secs.rem_euclid(86_400) as u64;
    let (h, mi, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    let (y, mo, d) = civil_from_days(days);
    let weekdays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let weekday = weekdays[(days + 4).rem_euclid(7) as usize];
    let month = months[(mo - 1) as usize];
    let mut output = String::new();
    let mut chars = fmt.chars();
    while let Some(character) = chars.next() {
        if character != '%' {
            output.push(character);
            continue;
        }
        match chars.next() {
            Some('%') => output.push('%'),
            Some('Y') => output.push_str(&format!("{y:04}")),
            Some('m') => output.push_str(&format!("{mo:02}")),
            Some('d') => output.push_str(&format!("{d:02}")),
            Some('e') => output.push_str(&format!("{d:2}")),
            Some('H') => output.push_str(&format!("{h:02}")),
            Some('M') => output.push_str(&format!("{mi:02}")),
            Some('S') => output.push_str(&format!("{s:02}")),
            Some('N') => output.push_str(&format!("{subsecond_ns:09}")),
            Some('a') => output.push_str(weekday),
            Some('b') => output.push_str(month),
            Some('s') => output.push_str(&secs.to_string()),
            Some('z') => output.push_str("+0000"),
            Some('Z') => output.push_str("UTC"),
            Some('F') => output.push_str(&format!("{y:04}-{mo:02}-{d:02}")),
            Some('T') => output.push_str(&format!("{h:02}:{mi:02}:{s:02}")),
            Some(other) => {
                output.push('%');
                output.push(other);
            }
            None => output.push('%'),
        }
    }
    Ok(output)
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
    let Some((source, positional)) = shell_invocation(interp, args, io) else {
        return 127;
    };
    run_shell_child(interp, &source, positional, io)
}

/// Start a nested shell as an ordinary scheduled child instead of recursively executing it.
fn start_sh(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let Some((source, positional)) = shell_invocation(interp, args, io) else {
        return CommandPoll::Ready(127);
    };
    start_shell_source(interp, &source, positional, None, io.err)
}

/// Parse and launch one shell source as a scheduler-owned child.
pub(crate) fn start_shell_source(
    interp: &mut Interp,
    source: &str,
    positional: Vec<String>,
    stdin: Option<Vec<u8>>,
    err: &mut Vec<u8>,
) -> CommandPoll {
    let ast = match crate::commands::parse_shell_source(interp, source, err) {
        Ok(ast) => ast,
        Err(status) => return CommandPoll::Ready(status),
    };
    let input = match stdin {
        Some(stdin) => match interp.descriptors.open_input(stdin) {
            Ok(input) => Some(input),
            Err(error) => {
                ewln(err, &format!("bash: unable to prepare stdin: {error:?}"));
                return CommandPoll::Ready(125);
            }
        },
        None => None,
    };
    let pid = match interp.start_child("bash", true) {
        Ok(pid) => pid,
        Err(error) => {
            if let Some(input) = input {
                let _ = interp.descriptors.discard_unreferenced(input);
            }
            ewln(err, &format!("bash: {error}"));
            return CommandPoll::Ready(125);
        }
    };
    if let Some(input) = input {
        interp
            .install_process_description(pid, 0, input)
            .expect("new child process must accept a valid stdin description");
    }
    interp
        .set_process_positional(pid, positional)
        .expect("new child process state must retain positional arguments");
    interp
        .process
        .set_continuation(pid, Some(crate::exec::ShellContinuation::new(&ast)))
        .expect("new child process state must accept a continuation");
    CommandPoll::Switched(CommandResume::Child { pid, reap: true })
}

fn shell_invocation(
    interp: &mut CommandContext<'_>,
    args: &[String],
    io: &mut Io,
) -> Option<(String, Vec<String>)> {
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-c" => {
                let src = args.get(i + 1).cloned().unwrap_or_default();
                // `sh -c SCRIPT [name [args…]]`: name is $0, the rest are $1+
                let extra = args.get(i + 3..).map(|s| s.to_vec()).unwrap_or_default();
                return Some((src, extra));
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
                    return Some((src, extra));
                }
                crate::commands::util::ewln(
                    io.err,
                    &format!("bash: {s}: No such file or directory"),
                );
                return None;
            }
        }
    }
    // no -c and no file → run stdin as a script
    let src = String::from_utf8_lossy(&io.stdin).into_owned();
    Some((src, Vec::new()))
}

fn run_shell_child(
    interp: &mut CommandContext<'_>,
    source: &str,
    positional: Vec<String>,
    io: &mut Io,
) -> i32 {
    let pid = match interp.start_child("bash", true) {
        Ok(child) => child,
        Err(error) => {
            ewln(io.err, &format!("bash: {error}"));
            return 125;
        }
    };
    interp.positional = positional;
    let status = interp.run_script_into(source, io.out, io.err);
    interp.finish_child(pid, status);
    interp.processes.reap(pid);
    let _ = interp.scheduler.reap(pid);
    status
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
    if let Some(pos) = args
        .iter()
        .position(|a| matches!(a.as_str(), "python" | "python3" | "python3.14"))
    {
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
fn cmd_python314(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    python_impl(interp, "python3.14", args, io)
}

fn cmd_pytest(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    crate::python::run_pytest(interp, args, io.out, io.err)
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

#[cfg(test)]
mod time_tests {
    use super::format_date;
    use crate::clock::NANOS_PER_SECOND;
    use std::process::Command;

    #[test]
    fn utc_formatting_matches_coreutils_when_available() {
        let format = "%a %b %F %T %z %Z %s";
        for seconds in [0_i128, 951_827_696, 1_735_689_600, 1_772_368_496] {
            let expected = Command::new("date")
                .args(["-u", &format!("--date=@{seconds}"), &format!("+{format}")])
                .output();
            let Ok(expected) = expected else {
                return;
            };
            assert!(expected.status.success());
            assert_eq!(
                format_date(seconds * i128::from(NANOS_PER_SECOND), format).unwrap(),
                String::from_utf8_lossy(&expected.stdout).trim_end()
            );
        }
    }
}
