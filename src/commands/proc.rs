//! "Process-ish" commands: virtual-clock time/scheduling (sleep/usleep/timeout/date/sync),
//! nested shells (sh/bash/dash/zsh), uv launchers, the Python shim, and jq.

use std::collections::HashMap;

use crate::clock::{
    BlockOutcome, EventKind, MAIN_TASK_ID, NANOS_PER_MICROSECOND, NANOS_PER_SECOND,
};
use crate::commands::util::{ewln, parse_duration_ns, wln};
use crate::commands::{
    ChildCommand, CommandContext, CommandPoll, CommandResume, CommandSpec, Io, Trust,
};
use crate::interp::Interp;
use crate::scheduler::WaitReason;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::{reg, reg_buffered_resumable, reg_resumable, reg_unsupported};
    // time / scheduling (virtual clock, never blocks)
    reg_resumable(m, &["sleep"], Trust::Real, cmd_sleep, start_sleep);
    reg_resumable(m, &["usleep"], Trust::Real, cmd_usleep, start_usleep);
    reg_buffered_resumable(m, &["timeout"], Trust::Partial, cmd_timeout, start_timeout);
    super::reg_system(m, "/usr/bin/date", Trust::Real, cmd_date);
    super::reg_system(m, "/usr/bin/sync", Trust::Real, |_, _| 0);

    // nested shells / uv-launched verifiers
    reg_resumable(
        m,
        &["sh", "bash", "dash", "zsh"],
        Trust::Real,
        cmd_sh,
        start_sh,
    );
    reg(m, &["uv", "uvx"], Trust::Partial, cmd_uv);
    reg_unsupported(m, &["uvenv"]);

    // interpreters
    reg_buffered_resumable(m, &["python3"], Trust::Partial, cmd_python3, start_python3);
    reg_buffered_resumable(m, &["python"], Trust::Partial, cmd_python, start_python);
    reg_buffered_resumable(
        m,
        &["python3.11"],
        Trust::Partial,
        cmd_python311,
        start_python311,
    );
    reg_buffered_resumable(
        m,
        &["python3.12"],
        Trust::Partial,
        cmd_python312,
        start_python312,
    );
    reg_buffered_resumable(
        m,
        &["python3.13"],
        Trust::Partial,
        cmd_python313,
        start_python313,
    );
    reg_buffered_resumable(
        m,
        &["python3.14"],
        Trust::Partial,
        cmd_python314,
        start_python314,
    );
    reg(m, &["pytest"], Trust::Partial, cmd_pytest);
    super::reg_system_poll(m, "/usr/bin/jq", Trust::Partial, cmd_jq);
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
            CommandResume::Timer {
                deadline_ns: event.deadline_ns(),
                status: 0,
            },
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
    let invocation = match parse_timeout(args) {
        Ok(invocation) => invocation,
        Err(error) => {
            ewln(io.err, &format!("timeout: {error}"));
            return 125;
        }
    };
    // This entry remains for synchronous native callers. Ordinary shell execution uses
    // `start_timeout`, which delivers a signal to a scheduler-owned child.
    let deadline = if invocation.duration == 0 {
        None
    } else {
        match interp.clock.schedule_after(
            invocation.duration,
            EventKind::Deadline { task: MAIN_TASK_ID },
        ) {
            Ok(event) => Some(event),
            Err(error) => {
                ewln(io.err, &format!("timeout: {error}"));
                return 125;
            }
        }
    };
    let status = crate::commands::run(
        interp,
        &invocation.argv,
        std::mem::take(&mut io.stdin),
        io.out,
        io.err,
    );
    let interrupted = interp.deadline_interrupt;
    if let Some(deadline) = deadline {
        interp.clock.cancel(deadline);
        if interrupted == Some(deadline) {
            interp.deadline_interrupt = None;
            return if invocation.preserve_status {
                status
            } else {
                124
            };
        }
    }
    if interrupted.is_some() {
        124
    } else {
        status
    }
}

fn start_timeout(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let invocation = match parse_timeout(args) {
        Ok(invocation) => invocation,
        Err(error) => {
            ewln(io.err, &format!("timeout: {error}"));
            return CommandPoll::Ready(125);
        }
    };
    let pid = match crate::commands::start_child_command(
        interp,
        ChildCommand {
            argv: invocation.argv,
            stdin: Some(std::mem::take(&mut io.stdin)),
            cwd: None,
            environment: None,
        },
    ) {
        Ok(pid) => pid,
        Err(status) => return CommandPoll::Ready(status),
    };
    let (deadline, result_override) = if invocation.duration == 0 {
        (None, None)
    } else {
        match interp.clock.schedule_after(
            invocation.duration,
            EventKind::SignalTask {
                task: u64::from(pid),
                signal: invocation.signal,
                descendants: true,
            },
        ) {
            Ok(event) => (Some(event), None),
            Err(_) => {
                let _ = interp.send_signal(pid, crate::process::Signal::Kill);
                (None, Some(125))
            }
        }
    };
    CommandPoll::Switched(CommandResume::Timeout {
        pid,
        deadline,
        preserve_status: invocation.preserve_status,
        result_override,
    })
}

struct TimeoutInvocation {
    duration: u64,
    preserve_status: bool,
    signal: crate::process::Signal,
    argv: Vec<String>,
}

fn parse_timeout(args: &[String]) -> Result<TimeoutInvocation, String> {
    let mut index = 0;
    let mut preserve_status = false;
    let mut signal = crate::process::Signal::Terminate;
    while index < args.len() {
        match args[index].as_str() {
            "--preserve-status" => {
                preserve_status = true;
                index += 1;
            }
            "--foreground" => index += 1,
            "-s" | "--signal" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "option requires a signal".to_string())?;
                signal = crate::process::Signal::parse(value)
                    .filter(|signal| signal.terminates())
                    .ok_or_else(|| format!("unsupported signal '{value}'"))?;
                index += 2;
            }
            option if option.starts_with("--signal=") => {
                let value = option.trim_start_matches("--signal=");
                signal = crate::process::Signal::parse(value)
                    .filter(|signal| signal.terminates())
                    .ok_or_else(|| format!("unsupported signal '{value}'"))?;
                index += 1;
            }
            "-k" | "--kill-after" => {
                return Err("--kill-after is unsupported".to_string());
            }
            option if option.starts_with("--kill-after=") => {
                return Err("--kill-after is unsupported".to_string());
            }
            "--" => {
                index += 1;
                break;
            }
            option if option.starts_with('-') => {
                return Err(format!("unsupported option '{option}'"));
            }
            _ => break,
        }
    }
    let duration = parse_duration_ns(
        args.get(index)
            .ok_or_else(|| "missing operand".to_string())?,
    )?;
    let argv = args[index + 1..].to_vec();
    if argv.is_empty() {
        return Err("missing command".to_string());
    }
    Ok(TimeoutInvocation {
        duration,
        preserve_status,
        signal,
        argv,
    })
}

fn cmd_date(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let unix_ns = match context.system.wall_time_signed_ns() {
        Ok(value) => value,
        Err(error) => {
            ewln(io.err, &format!("date: {error}"));
            return 1;
        }
    };
    let rendered = if let Some(fmt) = context.args.iter().find(|a| a.starts_with('+')) {
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

pub(crate) fn format_date(unix_ns: i128, fmt: &str) -> Result<String, String> {
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
        // `%-X` suppresses the usual zero padding, as GNU date allows.
        let (unpadded, specifier) = match chars.next() {
            Some('-') => (true, chars.next()),
            other => (false, other),
        };
        if unpadded {
            match specifier {
                Some('Y') => output.push_str(&y.to_string()),
                Some('m') => output.push_str(&mo.to_string()),
                Some('d') | Some('e') => output.push_str(&d.to_string()),
                Some('H') => output.push_str(&h.to_string()),
                Some('M') => output.push_str(&mi.to_string()),
                Some('S') => output.push_str(&s.to_string()),
                Some(other) => {
                    output.push_str("%-");
                    output.push(other);
                }
                None => output.push_str("%-"),
            }
            continue;
        }
        match specifier {
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
    let first = args.first().map(String::as_str);
    if matches!(first, Some("--help" | "-h")) {
        wln(
            io.out,
            "usage: uv {add,pip install,sync,lock,venv,run,tool run} ...",
        );
        return 0;
    }
    // ---- package management (takes priority so `uv pip install pytest` installs, not runs) ----
    if first == Some("add") {
        let request = match crate::commands::pkg::resolve_install_args(interp, &args[1..]) {
            Ok(request) => request,
            Err(error) => return uv_failure(interp, io, &error, &error, 1),
        };
        crate::commands::pkg::install_packages(interp, &request.packages);
        update_pyproject(interp, &request.direct_specs);
        super::pkg::ensure_venv(interp, ".venv", true);
        return 0;
    }
    if first == Some("init") {
        if args.len() != 1 {
            return uv_failure(interp, io, "init-options", "unsupported init option", 2);
        }
        let path = crate::vfs::resolve_against(&interp.cwd, "pyproject.toml");
        if !interp.vfs.is_file("/", &path) {
            let source = b"[project]\nname = \"app\"\nversion = \"0.1.0\"\ndependencies = []\n";
            if let Err(error) = interp.vfs.put_file(&path, source.to_vec(), 0o644) {
                ewln(
                    io.err,
                    &format!("uv: cannot create pyproject.toml: {error}"),
                );
                return 1;
            }
        }
        return 0;
    }
    if first == Some("remove") {
        return uv_failure(interp, io, "remove", "unsupported command 'remove'", 2);
    }
    if matches!(first, Some("sync" | "lock")) {
        if args.len() != 1 {
            return uv_failure(
                interp,
                io,
                "sync-options",
                "unsupported sync or lock option",
                2,
            );
        }
        if let Err(error) = uv_sync(interp) {
            return uv_failure(interp, io, &error, &error, 1);
        }
        super::pkg::ensure_venv(interp, ".venv", true);
        return 0;
    }
    if first == Some("pip") && args.get(1).map(String::as_str) == Some("install") {
        let install_args = args[2..]
            .iter()
            .filter(|argument| argument.as_str() != "--system")
            .cloned()
            .collect::<Vec<_>>();
        if let Err(error) = crate::commands::pkg::install_args(interp, &install_args) {
            return uv_failure(interp, io, &error, &error, 1);
        }
        super::pkg::ensure_venv(interp, ".venv", true);
        return 0;
    }
    if first == Some("venv") {
        if args.len() > 2
            || args
                .get(1)
                .is_some_and(|argument| argument.starts_with('-'))
        {
            return uv_failure(interp, io, "venv-arguments", "unsupported venv argument", 2);
        }
        super::pkg::ensure_venv(interp, args.get(1).map_or(".venv", String::as_str), true);
        return 0;
    }
    // ---- run / tool run / uvx: route the embedded interpreter ----
    let launcher_args = if interp.command_name() == "uvx" {
        Some(args)
    } else if first == Some("run") {
        Some(&args[1..])
    } else if first == Some("tool") && args.get(1).map(String::as_str) == Some("run") {
        Some(&args[2..])
    } else {
        None
    };
    let launcher_args = launcher_args.map(|launcher_args| {
        if launcher_args.first().map(String::as_str) == Some("--") {
            &launcher_args[1..]
        } else {
            launcher_args
        }
    });
    let launcher_args = match launcher_args {
        Some(arguments) => match uv_launcher_args(interp, arguments) {
            Ok(arguments) => Some(arguments),
            Err(error) => return uv_failure(interp, io, &error, &error, 2),
        },
        None => None,
    };
    if let Some(launcher_args) = launcher_args.as_ref().filter(|args| {
        args.first()
            .is_some_and(|program| program == "pytest" || program.ends_with("/pytest"))
    }) {
        return crate::python::run_pytest(interp, &launcher_args[1..], io.out, io.err);
    }
    if let Some(launcher_args) = launcher_args.as_ref().filter(|args| {
        args.first()
            .is_some_and(|program| matches!(program.as_str(), "python" | "python3" | "python3.14"))
    }) {
        let mut argv = vec![launcher_args[0].clone()];
        argv.extend(launcher_args[1..].iter().cloned());
        let stdin = std::mem::take(&mut io.stdin);
        return crate::python::run_python(interp, &argv, stdin, io.out, io.err);
    }
    let invocation = args.join(" ");
    uv_failure(
        interp,
        io,
        &invocation,
        &format!("unsupported invocation '{invocation}'"),
        2,
    )
}

fn uv_launcher_args(interp: &mut Interp, args: &[String]) -> Result<Vec<String>, String> {
    let mut index = 0;
    let mut packages = Vec::new();
    while let Some(argument) = args.get(index) {
        let (option, attached) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(option, value)| {
                (option, Some(value))
            });
        let needs_value = matches!(option, "-p" | "--python" | "-w" | "--with");
        if !needs_value {
            if argument == "--" {
                index += 1;
                break;
            }
            if argument.starts_with('-') {
                return Err(format!("unsupported launcher option '{argument}'"));
            }
            break;
        }
        let value = if let Some(value) = attached {
            value
        } else {
            index += 1;
            args.get(index)
                .map(String::as_str)
                .ok_or_else(|| format!("option '{option}' requires a value"))?
        };
        if matches!(option, "-p" | "--python") {
            if !matches!(
                value,
                "3.11"
                    | "3.12"
                    | "3.13"
                    | "3.14"
                    | "python3.11"
                    | "python3.12"
                    | "python3.13"
                    | "python3.14"
            ) {
                return Err(format!("Python selector '{value}' is not modeled"));
            }
        } else {
            packages.push(value.to_string());
        }
        index += 1;
    }
    let packages = crate::commands::pkg::resolve_package_specs(&packages)?;
    crate::commands::pkg::install_packages(interp, &packages);
    Ok(args[index..].to_vec())
}

fn uv_failure(
    interp: &mut CommandContext<'_>,
    io: &mut Io,
    feature: &str,
    diagnostic: &str,
    status: i32,
) -> i32 {
    interp.note_unsupported(&format!("uv:{feature}"));
    ewln(io.err, &format!("uv: {diagnostic}"));
    status
}

/// Validate every declared project or requirements dependency, then activate all of them atomically.
fn uv_sync(interp: &mut Interp) -> Result<(), String> {
    let cwd = interp.cwd.clone();
    let mut packages = Vec::new();
    let path = crate::vfs::resolve_against(&cwd, "pyproject.toml");
    let req = crate::vfs::resolve_against(&cwd, "requirements.txt");
    let has_project = interp.vfs.is_file("/", &path);
    let has_requirements = interp.vfs.is_file("/", &req);
    if !has_project && !has_requirements {
        return Err("no pyproject.toml or requirements.txt found".to_string());
    }
    if has_project {
        let content = interp
            .vfs
            .read_string("/", &path)
            .map_err(|error| format!("cannot read pyproject.toml: {error}"))?;
        // Pull each "name>=ver" / "name==ver" string out of the dependencies arrays.
        packages.extend(crate::commands::pkg::resolve_package_specs(
            &extract_dep_specs(&content),
        )?);
    }
    // a requirements.txt next to it, if present
    if has_requirements {
        packages.extend(crate::commands::pkg::resolve_requirements_file(
            interp, &req,
        )?);
    }
    crate::commands::pkg::install_packages(interp, &packages);
    Ok(())
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
                // A dependency spec starts with a letter. Skip the dependencies token and bare
                // punctuation; package validation later separates the name from its version.
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
fn start_python3(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    start_python_impl(interp, "python3", args, io)
}
fn cmd_python(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    python_impl(interp, "python", args, io)
}
fn start_python(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    start_python_impl(interp, "python", args, io)
}
fn cmd_python311(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    python_impl(interp, "python3.11", args, io)
}
fn start_python311(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    start_python_impl(interp, "python3.11", args, io)
}
fn cmd_python312(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    python_impl(interp, "python3.12", args, io)
}
fn start_python312(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    start_python_impl(interp, "python3.12", args, io)
}
fn cmd_python313(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    python_impl(interp, "python3.13", args, io)
}
fn start_python313(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    start_python_impl(interp, "python3.13", args, io)
}
fn cmd_python314(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    python_impl(interp, "python3.14", args, io)
}
fn start_python314(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    start_python_impl(interp, "python3.14", args, io)
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

fn start_python_impl(interp: &mut Interp, name: &str, args: &[String], io: &mut Io) -> CommandPoll {
    let mut argv = vec![name.to_string()];
    argv.extend(args.iter().cloned());
    let stdin = std::mem::take(&mut io.stdin);
    match crate::python::start_python(interp, &argv, stdin, io.out, io.err) {
        crate::python::PythonCommandStart::Ready(status) => CommandPoll::Ready(status),
        crate::python::PythonCommandStart::Running(mut continuation) => {
            match continuation.poll(interp) {
                crate::python::PythonPoll::Ready(status) => {
                    let (stdout, stderr) = (*continuation).into_output();
                    io.out.extend_from_slice(&stdout);
                    io.err.extend_from_slice(&stderr);
                    CommandPoll::Ready(status)
                }
                crate::python::PythonPoll::Runnable => {
                    CommandPoll::Yielded(CommandResume::Python {
                        command: name.to_string(),
                        continuation,
                    })
                }
                crate::python::PythonPoll::Blocked(reason) => CommandPoll::Blocked(
                    reason,
                    CommandResume::Python {
                        command: name.to_string(),
                        continuation,
                    },
                ),
            }
        }
    }
}

fn cmd_jq(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> crate::exec::ShellPoll {
    if crate::jqcmd::reads_standard_input(context.args) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let stdin = std::mem::take(&mut io.stdin);
    crate::exec::ShellPoll::Ready(crate::jqcmd::jq(
        context.system,
        context.args,
        stdin,
        io.out,
        io.err,
    ))
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
