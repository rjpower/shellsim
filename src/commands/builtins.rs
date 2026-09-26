//! Shell builtins: state-mutating commands (cd, export, set, …), control-flow signals
//! (exit/return/break/continue/shift), `test`/`[`, `read`, `let`, `source`/`eval`, and the
//! small job-table queries. Unsupported process and interactive controls fail explicitly.

use std::collections::HashMap;

use crate::commands::util::{ewln, split_flags, wln};
use crate::commands::{CommandContext, CommandPoll, CommandResume, CommandSpec, Io, Trust};
use crate::interp::Interp;
use crate::syscalls::System;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::{reg, reg_resumable, reg_unsupported};
    reg(m, &["cd"], Trust::Real, cmd_cd);
    reg(m, &["pwd"], Trust::Real, cmd_pwd);
    reg(m, &["export"], Trust::Real, cmd_export);
    reg(m, &["unset"], Trust::Real, cmd_unset);
    reg(m, &["set"], Trust::Real, cmd_set);
    reg(m, &["declare", "typeset"], Trust::Real, cmd_declare);
    reg(m, &["local"], Trust::Real, cmd_local);
    reg(m, &["readonly"], Trust::Real, cmd_readonly);
    reg_resumable(m, &["source", "."], Trust::Real, start_source);
    reg_resumable(m, &["eval"], Trust::Real, start_eval);
    reg(m, &["exit"], Trust::Real, cmd_exit);
    reg(m, &["return"], Trust::Real, cmd_return);
    reg(m, &["break"], Trust::Real, cmd_break);
    reg(m, &["continue"], Trust::Real, cmd_continue);
    reg(m, &["shift"], Trust::Real, cmd_shift);
    reg(m, &["true", ":"], Trust::Real, cmd_true);
    reg(m, &["false"], Trust::Real, cmd_false);
    reg(m, &["test"], Trust::Real, cmd_test);
    reg(m, &["["], Trust::Real, cmd_bracket);
    reg(m, &["[["], Trust::Real, cmd_dbracket);
    reg(m, &["read"], Trust::Real, cmd_read);
    reg_resumable(m, &["wait"], Trust::Real, start_wait);
    reg(m, &["jobs"], Trust::Real, cmd_jobs);
    reg_resumable(m, &["fg"], Trust::Real, start_fg);
    reg(m, &["bg"], Trust::Real, cmd_bg);
    reg(m, &["trap"], Trust::Real, cmd_trap);
    reg(m, &["flock"], Trust::Partial, cmd_flock);
    reg(m, &["disown"], Trust::Real, cmd_disown);
    reg(m, &["umask"], Trust::Partial, cmd_umask);
    reg(m, &["ulimit"], Trust::Partial, cmd_ulimit);
    reg(m, &["hash"], Trust::Real, cmd_hash);
    reg(m, &["shopt"], Trust::Partial, cmd_shopt);
    reg(m, &["exec"], Trust::Partial, cmd_exec);
    reg_unsupported(m, &["complete", "bind", "history"]);
    reg(m, &["kill"], Trust::Real, cmd_kill);
    reg(m, &["which"], Trust::Real, cmd_which);
    reg(m, &["type"], Trust::Real, cmd_type);
    reg_resumable(m, &["command"], Trust::Real, start_command);
    reg(m, &["alias"], Trust::Partial, cmd_alias);
    reg(m, &["unalias"], Trust::Real, cmd_unalias);
    reg(m, &["getopts"], Trust::Real, cmd_getopts);
    reg(m, &["let"], Trust::Real, cmd_let);
    reg(m, &["mapfile", "readarray"], Trust::Real, cmd_mapfile);
    reg(m, &["pushd"], Trust::Real, cmd_pushd);
    reg(m, &["popd"], Trust::Real, cmd_popd);
    reg(m, &["dirs"], Trust::Real, cmd_dirs);
}

fn cmd_readonly(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.is_empty() || args == ["-p"] {
        for name in &interp.readonly {
            let value = interp.get_var(name).unwrap_or_default();
            wln(io.out, &format!("declare -r {name}=\"{value}\""));
        }
        return 0;
    }
    let mut status = 0;
    for argument in args.iter().filter(|argument| argument.as_str() != "--") {
        if argument.starts_with('-') {
            ewln(io.err, &format!("readonly: unsupported option {argument}"));
            status = 2;
            continue;
        }
        let name = declare_name_of(argument);
        if !shell_identifier(name) {
            ewln(io.err, &format!("readonly: {name}: not a valid identifier"));
            status = 1;
            continue;
        }
        if let Some((raw_key, raw_value)) = split_decl_assign(argument) {
            if interp.readonly.contains(name) {
                ewln(io.err, &format!("readonly: {name}: readonly variable"));
                status = 1;
                continue;
            }
            crate::exec::apply_assignment(interp, &raw_key, &raw_value);
        } else if interp.get_var(name).is_none() && !interp.arrays.contains_key(name) {
            interp.set_var(name, "");
        }
        interp.readonly.insert(name.to_string());
    }
    status
}

fn cmd_umask(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    match args {
        [] => {
            wln(io.out, &format!("{:04o}", interp.system().umask()));
            0
        }
        [flag] if flag == "-S" => {
            let allowed = 0o777 & !interp.system().umask();
            let triplet = |shift| {
                let bits = (allowed >> shift) & 7_u16;
                format!(
                    "{}{}{}",
                    if bits & 4_u16 != 0_u16 { "r" } else { "" },
                    if bits & 2_u16 != 0_u16 { "w" } else { "" },
                    if bits & 1_u16 != 0_u16 { "x" } else { "" }
                )
            };
            wln(
                io.out,
                &format!("u={},g={},o={}", triplet(6), triplet(3), triplet(0)),
            );
            0
        }
        [value] => match u16::from_str_radix(
            match value.trim_start_matches('0') {
                "" if !value.is_empty() => "0",
                digits => digits,
            },
            8,
        ) {
            Ok(mask) if mask <= 0o777 => {
                interp
                    .system()
                    .set_umask(mask)
                    .expect("validated octal mask");
                0
            }
            _ => {
                ewln(io.err, &format!("umask: {value}: invalid octal mask"));
                1
            }
        },
        _ => {
            ewln(io.err, "umask: usage: umask [-S] [octal-mask]");
            2
        }
    }
}

fn cmd_ulimit(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let limits = interp.system().limits();
    let selected = args.first().map(String::as_str).unwrap_or("-f");
    if args.len() > 1 || !matches!(selected, "-a" | "-f" | "-n" | "-t" | "-v") {
        ewln(
            io.err,
            "ulimit: setting limits and this option are not supported",
        );
        return 2;
    }
    let blocks = |bytes: u64| {
        if bytes == u64::MAX {
            "unlimited".into()
        } else {
            (bytes / 1024).to_string()
        }
    };
    match selected {
        "-a" => {
            wln(
                io.out,
                &format!(
                    "file size               (blocks, -f) {}",
                    blocks(limits.disk)
                ),
            );
            wln(io.out, "open files                      (-n) 1024");
            wln(
                io.out,
                &format!("cpu time               (seconds, -t) {}", limits.cpu),
            );
            wln(
                io.out,
                &format!(
                    "virtual memory           (kbytes, -v) {}",
                    blocks(limits.memory)
                ),
            );
        }
        "-f" => wln(io.out, &blocks(limits.disk)),
        "-n" => wln(io.out, "1024"),
        "-t" => wln(io.out, &limits.cpu.to_string()),
        "-v" => wln(io.out, &blocks(limits.memory)),
        _ => unreachable!(),
    }
    0
}

fn cmd_hash(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args == ["-r"] {
        interp.command_hash.clear();
        return 0;
    }
    if args.first().map(String::as_str) == Some("-t") {
        let mut status = 0;
        for name in &args[1..] {
            if let Some(path) = interp.command_hash.get(name) {
                wln(io.out, path);
            } else {
                ewln(io.err, &format!("hash: {name}: not found"));
                status = 1;
            }
        }
        return status;
    }
    if args.is_empty() {
        for (name, path) in &interp.command_hash {
            wln(io.out, &format!("{path}\t{name}"));
        }
        return 0;
    }
    let mut status = 0;
    for name in args {
        let path = if let crate::commands::util::ExecutableLookup::Found(path) =
            crate::commands::util::resolve_executable(interp, name)
        {
            Some(path)
        } else {
            None
        };
        if let Some(path) = path {
            interp.command_hash.insert(name.clone(), path);
        } else {
            ewln(io.err, &format!("hash: {name}: not found"));
            status = 1;
        }
    }
    status
}

const COMPAT_SHOPTS: &[&str] = &["expand_aliases", "sourcepath"];

fn cmd_shopt(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (mode, names) = match args.first().map(String::as_str) {
        Some("-s") => ('s', &args[1..]),
        Some("-u") => ('u', &args[1..]),
        Some("-q") => ('q', &args[1..]),
        Some(option) if option.starts_with('-') => {
            ewln(io.err, &format!("shopt: unsupported option {option}"));
            return 2;
        }
        _ => ('p', args),
    };
    let selected = if names.is_empty() {
        COMPAT_SHOPTS
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>()
    } else {
        names.to_vec()
    };
    let mut status = 0;
    for name in selected {
        if !COMPAT_SHOPTS.contains(&name.as_str()) {
            ewln(io.err, &format!("shopt: {name}: invalid shell option name"));
            status = 1;
            continue;
        }
        match mode {
            's' => {
                interp.shell_options.insert(name);
            }
            'u' => {
                interp.shell_options.remove(&name);
            }
            'q' => status |= i32::from(!interp.shell_options.contains(&name)),
            _ => wln(
                io.out,
                &format!(
                    "{}\t{}",
                    name,
                    if interp.shell_options.contains(&name) {
                        "on"
                    } else {
                        "off"
                    }
                ),
            ),
        }
    }
    status
}

fn cmd_disown(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args == ["-a"] {
        interp.jobs.clear();
        return 0;
    }
    let position = match job_position(interp, args, "disown") {
        Ok(position) => position,
        Err(error) => {
            ewln(io.err, &format!("disown: {error}"));
            return 1;
        }
    };
    interp.jobs.remove(position);
    0
}

/// `exec` with a command replaces the process in the shell executor (`exec_builtin`); this
/// body is reached only indirectly, as in `command exec cmd`, which is not modeled.
fn cmd_exec(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.is_empty() {
        return 0;
    }
    ewln(
        io.err,
        "exec: indirect invocation is not supported in shellsim",
    );
    2
}

const MAX_TRAP_STATE_BYTES: u64 = 1024 * 1024;

/// Validate the common descriptor-lock form. A shellsim process has no ambient competing host
/// processes, so acquiring or releasing an advisory lock on one of its own open descriptors is
/// deterministic and immediate. Path-and-command forms are a separate process-control surface.
fn cmd_flock(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut operand = None;
    for argument in args {
        match argument.as_str() {
            "-x" | "--exclusive" | "-s" | "--shared" | "-u" | "--unlock" | "-n" | "--nonblock" => {}
            "--" if operand.is_none() => {}
            value if operand.is_none() => operand = Some(value),
            _ => {
                ewln(
                    io.err,
                    "flock: path-and-command locking is not supported; use an open file descriptor",
                );
                return 2;
            }
        }
    }
    let Some(operand) = operand else {
        ewln(io.err, "flock: missing file descriptor");
        return 2;
    };
    let Ok(fd) = operand.parse::<i32>() else {
        ewln(
            io.err,
            "flock: path-and-command locking is not supported; use an open file descriptor",
        );
        return 2;
    };
    if interp.fds.get(fd).is_err() {
        ewln(io.err, &format!("flock: {fd}: bad file descriptor"));
        return 1;
    }
    0
}

fn cmd_trap(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.is_empty() {
        print_traps(interp, &[], true, io);
        return 0;
    }
    if args[0] == "-l" {
        if args.len() != 1 {
            ewln(io.err, "trap: -l does not accept operands");
            return 2;
        }
        return list_signal(None, io);
    }
    if args[0] == "-p" {
        let (include_exit, signals) = match parse_trap_targets(&args[1..]) {
            Ok(targets) => targets,
            Err(error) => {
                ewln(io.err, &format!("trap: {error}"));
                return 2;
            }
        };
        print_traps(interp, &signals, include_exit, io);
        return 0;
    }

    let action_index = usize::from(args[0] == "--");
    let Some(action) = args.get(action_index) else {
        ewln(io.err, "trap: missing action");
        return 2;
    };
    let signal_args = &args[action_index + 1..];
    if signal_args.is_empty() {
        ewln(io.err, "trap: missing signal operand");
        return 2;
    }
    let (has_exit, signals) = match parse_trap_targets(signal_args) {
        Ok(targets) => targets,
        Err(error) => {
            ewln(io.err, &format!("trap: {error}"));
            return 2;
        }
    };
    if signals.contains(&crate::process::Signal::Kill) {
        ewln(io.err, "trap: SIGKILL cannot be caught or ignored");
        return 1;
    }
    if signals.contains(&crate::process::Signal::Stop) {
        ewln(io.err, "trap: SIGSTOP cannot be caught or ignored");
        return 1;
    }
    if signals.contains(&crate::process::Signal::Continue) {
        ewln(io.err, "trap: SIGCONT handlers are not supported");
        return 2;
    }

    let disposition = if action == "-" {
        None
    } else if action.is_empty() {
        Some(crate::interp::ShellSignalDisposition::Ignore)
    } else {
        let body = match crate::shell::parse(action) {
            Ok(body) => body,
            Err(error) => {
                ewln(io.err, &format!("trap: invalid handler: {error}"));
                return 2;
            }
        };
        Some(crate::interp::ShellSignalDisposition::Handler {
            source: action.clone(),
            body,
        })
    };

    let mut updated = interp.signal_dispositions.clone();
    for signal in signals {
        if let Some(disposition) = &disposition {
            updated.insert(signal, disposition.clone());
        } else {
            updated.remove(&signal);
        }
    }
    let mut updated_exit = interp.exit_disposition.clone();
    if has_exit {
        updated_exit = disposition.clone();
    }
    let state_bytes = updated
        .values()
        .fold(0_u64, |total, disposition| {
            total.saturating_add(match disposition {
                crate::interp::ShellSignalDisposition::Ignore => 16,
                crate::interp::ShellSignalDisposition::Handler { source, body } => (source.len()
                    as u64)
                    .saturating_add(body.estimated_bytes())
                    .saturating_add(32),
            })
        })
        .saturating_add(updated_exit.as_ref().map_or(0, |disposition| {
            match disposition {
                crate::interp::ShellSignalDisposition::Ignore => 16,
                crate::interp::ShellSignalDisposition::Handler { source, body } => (source.len()
                    as u64)
                    .saturating_add(body.estimated_bytes())
                    .saturating_add(32),
            }
        }));
    if state_bytes > MAX_TRAP_STATE_BYTES {
        ewln(io.err, "trap: signal handler state exceeds the 1 MiB limit");
        return 2;
    }
    interp.signal_dispositions = updated;
    interp.exit_disposition = updated_exit;
    0
}

fn parse_trap_targets(values: &[String]) -> Result<(bool, Vec<crate::process::Signal>), String> {
    let mut has_exit = values.is_empty();
    let mut signals = Vec::with_capacity(values.len());
    for value in values {
        if value == "0" || value.eq_ignore_ascii_case("EXIT") {
            has_exit = true;
            continue;
        }
        let signal = crate::process::Signal::parse(value)
            .ok_or_else(|| format!("invalid signal: {value}"))?;
        if !signals.contains(&signal) {
            signals.push(signal);
        }
    }
    Ok((has_exit, signals))
}

fn print_traps(
    interp: &CommandContext<'_>,
    selected: &[crate::process::Signal],
    include_exit: bool,
    io: &mut Io,
) {
    if include_exit {
        if let Some(disposition) = &interp.exit_disposition {
            let action = match disposition {
                crate::interp::ShellSignalDisposition::Ignore => String::new(),
                crate::interp::ShellSignalDisposition::Handler { source, .. } => source.clone(),
            };
            wln(
                io.out,
                &format!("trap -- '{}' EXIT", action.replace('\'', "'\\''")),
            );
        }
    }
    for (signal, disposition) in &interp.signal_dispositions {
        if !selected.is_empty() && !selected.contains(signal) {
            continue;
        }
        let action = match disposition {
            crate::interp::ShellSignalDisposition::Ignore => String::new(),
            crate::interp::ShellSignalDisposition::Handler { source, .. } => source.clone(),
        };
        wln(
            io.out,
            &format!(
                "trap -- '{}' SIG{}",
                action.replace('\'', "'\\''"),
                signal.name()
            ),
        );
    }
}

fn cmd_jobs(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let print_pids = match args {
        [] => false,
        [flag] if flag == "-p" => true,
        _ => {
            ewln(io.err, "jobs: only -p is supported");
            return 2;
        }
    };
    for job in &interp.jobs {
        if print_pids {
            wln(io.out, &job.pid.to_string());
        } else {
            let state = match job.state {
                crate::interp::JobState::Running => "Running",
                crate::interp::JobState::Stopped(_) => "Stopped",
                crate::interp::JobState::Done(_) => "Done",
            };
            wln(io.out, &format!("[{}] {state} {}", job.id, job.cmd));
        }
    }
    0
}

fn start_fg(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let position = match foreground_job_position(interp, args) {
        Ok(position) => position,
        Err(error) => {
            ewln(io.err, &format!("fg: {error}"));
            return CommandPoll::Ready(1);
        }
    };
    if matches!(
        interp.jobs[position].state,
        crate::interp::JobState::Done(_)
    ) {
        return CommandPoll::Ready(reap_job(interp, position));
    }
    let pid = interp.jobs[position].pid;
    if let Err(error) = interp.set_terminal_foreground(pid) {
        ewln(io.err, &format!("fg: {error}"));
        return CommandPoll::Ready(1);
    }
    if matches!(
        interp.jobs[position].state,
        crate::interp::JobState::Stopped(_)
    ) {
        if let Err(error) = interp.send_signal_group(pid, crate::process::Signal::Continue) {
            interp.terminal.foreground_group = interp.terminal.session_id;
            ewln(io.err, &format!("fg: {error}"));
            return CommandPoll::Ready(1);
        }
    }
    crate::commands::resume(interp, CommandResume::Foreground { pid })
}

fn cmd_bg(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let position = match job_position(interp, args, "bg") {
        Ok(position) => position,
        Err(error) => {
            ewln(io.err, &format!("bg: {error}"));
            return 1;
        }
    };
    let job = &interp.jobs[position];
    if matches!(job.state, crate::interp::JobState::Done(_)) {
        ewln(io.err, "bg: job has terminated");
        return 1;
    }
    let (pid, id, command) = (job.pid, job.id, job.cmd.clone());
    if let Err(error) = interp.send_signal_group(pid, crate::process::Signal::Continue) {
        ewln(io.err, &format!("bg: {error}"));
        return 1;
    }
    wln(io.out, &format!("[{id}] {command} &"));
    0
}

fn foreground_job_position(interp: &CommandContext<'_>, args: &[String]) -> Result<usize, String> {
    job_position(interp, args, "fg")
}

fn job_position(
    interp: &CommandContext<'_>,
    args: &[String],
    command: &str,
) -> Result<usize, String> {
    let selected = match args {
        [] => interp.jobs.len().checked_sub(1),
        [selected] if matches!(selected.as_str(), "%+" | "%%") => interp.jobs.len().checked_sub(1),
        [selected] => {
            let id = selected
                .strip_prefix('%')
                .unwrap_or(selected)
                .parse::<u32>()
                .map_err(|_| format!("{selected}: invalid job"))?;
            interp
                .jobs
                .iter()
                .position(|job| job.id == id || job.pid == id)
        }
        _ => return Err(format!("usage: {command} [%JOB]")),
    };
    selected.ok_or_else(|| "no such job".to_string())
}

fn reap_job(interp: &mut CommandContext<'_>, position: usize) -> i32 {
    let job = interp.jobs.remove(position);
    interp.processes.reap(job.pid);
    let _ = interp.scheduler.reap(job.pid);
    match job.state {
        crate::interp::JobState::Done(status) => status,
        crate::interp::JobState::Running | crate::interp::JobState::Stopped(_) => 127,
    }
}

fn cmd_kill(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.first().is_some_and(|argument| argument == "-l") {
        return list_signal(args.get(1), io);
    }
    let mut signal = Some(crate::process::Signal::Terminate);
    let mut targets = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--" => {
                targets.extend_from_slice(&args[index + 1..]);
                break;
            }
            "-s" | "--signal" => {
                index += 1;
                let Some(name) = args.get(index) else {
                    ewln(io.err, "kill: -s requires a signal");
                    return 2;
                };
                signal = match parse_signal(name) {
                    Ok(signal) => signal,
                    Err(error) => {
                        ewln(io.err, &format!("kill: {error}"));
                        return 2;
                    }
                };
            }
            option if option.starts_with('-') && option.len() > 1 && targets.is_empty() => {
                signal = match parse_signal(&option[1..]) {
                    Ok(signal) => signal,
                    Err(error) => {
                        ewln(io.err, &format!("kill: {error}"));
                        return 2;
                    }
                };
            }
            target => targets.push(target.to_string()),
        }
        index += 1;
    }
    if targets.is_empty() {
        ewln(io.err, "kill: usage: kill [-s SIGNAL] PID|%JOB ...");
        return 2;
    }
    let mut status = 0;
    for target in targets {
        let Some(resolved) = resolve_kill_target(interp, &target) else {
            ewln(io.err, &format!("kill: {target}: no such process or job"));
            status = 1;
            continue;
        };
        if let Some(signal) = signal {
            let result = match resolved {
                KillTarget::Process(pid) => interp.send_signal(pid, signal),
                KillTarget::Group(process_group) => interp.send_signal_group(process_group, signal),
            };
            if let Err(error) = result {
                ewln(io.err, &format!("kill: {target}: {error}"));
                status = 1;
            }
        } else if !kill_target_exists(interp, resolved) {
            ewln(io.err, &format!("kill: {target}: no such process"));
            status = 1;
        }
    }
    status
}

fn parse_signal(value: &str) -> Result<Option<crate::process::Signal>, String> {
    if value == "0" {
        Ok(None)
    } else {
        crate::process::Signal::parse(value)
            .map(Some)
            .ok_or_else(|| format!("invalid signal: {value}"))
    }
}

#[derive(Clone, Copy)]
enum KillTarget {
    Process(u32),
    Group(u32),
}

fn resolve_kill_target(interp: &CommandContext<'_>, target: &str) -> Option<KillTarget> {
    if let Some(job) = target.strip_prefix('%') {
        let id = job.parse::<u32>().ok()?;
        interp
            .jobs
            .iter()
            .find(|job| job.id == id)
            .map(|job| KillTarget::Group(job.pid))
    } else if target == "0" {
        Some(KillTarget::Group(interp.process.process_group))
    } else if let Some(group) = target.strip_prefix('-') {
        let process_group = group.parse::<u32>().ok()?;
        (process_group != 0).then_some(KillTarget::Group(process_group))
    } else {
        let pid = target.parse::<u32>().ok()?;
        (pid != 0).then_some(KillTarget::Process(pid))
    }
}

fn kill_target_exists(interp: &CommandContext<'_>, target: KillTarget) -> bool {
    match target {
        KillTarget::Process(pid) => matches!(
            interp.processes.get(pid).map(|record| record.status),
            Some(
                crate::process::ProcessStatus::Running | crate::process::ProcessStatus::Stopped(_)
            )
        ),
        KillTarget::Group(process_group) => !interp.processes.live_group(process_group).is_empty(),
    }
}

fn list_signal(argument: Option<&String>, io: &mut Io) -> i32 {
    let Some(argument) = argument else {
        wln(io.out, "HUP INT KILL USR1 USR2 PIPE TERM CHLD CONT STOP");
        return 0;
    };
    let normalized = argument
        .parse::<i32>()
        .ok()
        .map(|number| if number > 128 { number - 128 } else { number })
        .map_or_else(|| argument.clone(), |number| number.to_string());
    let Some(signal) = crate::process::Signal::parse(&normalized) else {
        ewln(io.err, &format!("kill: unknown signal {argument}"));
        return 1;
    };
    wln(io.out, signal.name());
    0
}

fn start_wait(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let explicit = !args.is_empty();
    let pids = if args.is_empty() {
        interp.jobs.iter().map(|job| job.pid).collect()
    } else {
        let mut pids = Vec::with_capacity(args.len());
        for argument in args {
            let parsed = argument
                .strip_prefix('%')
                .unwrap_or(argument)
                .parse::<u32>();
            let Ok(identifier) = parsed else {
                ewln(io.err, &format!("wait: {argument}: invalid job id"));
                return CommandPoll::Ready(127);
            };
            let job = if argument.starts_with('%') {
                interp.jobs.iter().find(|job| job.id == identifier)
            } else {
                interp.jobs.iter().find(|job| job.pid == identifier)
            };
            let Some(job) = job else {
                ewln(io.err, &format!("wait: {argument}: no such job"));
                return CommandPoll::Ready(127);
            };
            pids.push(job.pid);
        }
        pids
    };
    crate::commands::resume(
        interp,
        CommandResume::Wait {
            pids,
            status: 0,
            explicit,
        },
    )
}

fn cmd_true(_interp: &mut CommandContext<'_>, _args: &[String], _io: &mut Io) -> i32 {
    0
}

fn cmd_false(_interp: &mut CommandContext<'_>, _args: &[String], _io: &mut Io) -> i32 {
    1
}

fn cmd_alias(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.is_empty() {
        let mut names = interp.aliases.keys().cloned().collect::<Vec<_>>();
        names.sort();
        for name in names {
            print_alias(interp, &name, io);
        }
        return 0;
    }
    let mut status = 0;
    for argument in args {
        let Some((name, source)) = argument.split_once('=') else {
            if interp.aliases.contains_key(argument) {
                print_alias(interp, argument, io);
            } else {
                ewln(io.err, &format!("alias: {argument}: not found"));
                status = 1;
            }
            continue;
        };
        if name.is_empty()
            || name
                .bytes()
                .any(|byte| byte.is_ascii_whitespace() || matches!(byte, b'/' | b'$' | b'`'))
        {
            ewln(io.err, &format!("alias: invalid alias name: {name}"));
            status = 1;
            continue;
        }
        let words = match crate::shell::parse(source) {
            Ok(crate::shell::Node::Command {
                assigns,
                words,
                redirects,
            }) if assigns.is_empty() && !words.is_empty() && redirects.is_empty() => words,
            _ => {
                ewln(
                    io.err,
                    &format!("alias: {name}: only simple-command aliases are supported"),
                );
                status = 2;
                continue;
            }
        };
        if interp.aliases.len() >= 256 && !interp.aliases.contains_key(name) {
            ewln(io.err, "alias: alias limit exceeded");
            return 1;
        }
        interp.aliases.insert(
            name.to_string(),
            crate::interp::AliasDefinition {
                source: source.to_string(),
                words,
            },
        );
    }
    status
}

fn print_alias(interp: &CommandContext<'_>, name: &str, io: &mut Io) {
    if let Some(alias) = interp.aliases.get(name) {
        let quoted = alias.source.replace('\'', "'\\''");
        wln(io.out, &format!("alias {name}='{quoted}'"));
    }
}

fn cmd_unalias(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args == ["-a"] {
        interp.aliases.clear();
        return 0;
    }
    if args.is_empty() {
        ewln(io.err, "unalias: usage: unalias [-a] name ...");
        return 2;
    }
    let mut status = 0;
    for name in args {
        if interp.aliases.remove(name).is_none() {
            ewln(io.err, &format!("unalias: {name}: not found"));
            status = 1;
        }
    }
    status
}

fn cmd_pwd(interp: &mut CommandContext<'_>, _args: &[String], io: &mut Io) -> i32 {
    wln(io.out, interp.system().cwd());
    0
}

fn cmd_cd(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.len() > 1 {
        ewln(io.err, "cd: too many arguments");
        return 1;
    }
    let print = args.first().is_some_and(|argument| argument == "-");
    let target = match args.first().map(|s| s.as_str()) {
        None | Some("~") => interp.get_var("HOME").unwrap_or_else(|| "/".into()),
        Some("-") => interp
            .get_var("OLDPWD")
            .unwrap_or_else(|| interp.cwd.clone()),
        Some(p) => p.to_string(),
    };
    match change_directory(interp, &target) {
        Ok(()) => {
            if print {
                wln(io.out, &interp.cwd);
            }
            0
        }
        Err(()) => {
            ewln(io.err, &format!("cd: {target}: No such file or directory"));
            1
        }
    }
}

fn change_directory(interp: &mut CommandContext<'_>, target: &str) -> Result<(), ()> {
    let old = interp.cwd.clone();
    interp.system().chdir(target).map_err(|_| ())?;
    interp.set_var("OLDPWD", old);
    let cwd = interp.cwd.clone();
    interp.set_var("PWD", cwd);
    Ok(())
}

const MAX_DIRECTORY_STACK: usize = 256;

fn cmd_pushd(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.len() > 1
        || args
            .first()
            .is_some_and(|argument| argument.starts_with('-'))
    {
        ewln(io.err, "pushd: usage: pushd [DIR]");
        return 2;
    }
    if interp.directory_stack.len() >= MAX_DIRECTORY_STACK {
        ewln(io.err, "pushd: directory stack limit exceeded");
        return 1;
    }
    let previous = interp.cwd.clone();
    let target = match args.first() {
        Some(target) => target.clone(),
        None => match interp.directory_stack.pop() {
            Some(target) => target,
            None => {
                ewln(io.err, "pushd: no other directory");
                return 1;
            }
        },
    };
    if change_directory(interp, &target).is_err() {
        if args.is_empty() {
            interp.directory_stack.push(target.clone());
        }
        ewln(
            io.err,
            &format!("pushd: {target}: No such file or directory"),
        );
        return 1;
    }
    interp.directory_stack.push(previous);
    print_directory_stack(interp, false, io);
    0
}

fn cmd_popd(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if !args.is_empty() {
        ewln(io.err, "popd: arguments are not supported");
        return 2;
    }
    let Some(target) = interp.directory_stack.pop() else {
        ewln(io.err, "popd: directory stack empty");
        return 1;
    };
    if change_directory(interp, &target).is_err() {
        interp.directory_stack.push(target.clone());
        ewln(
            io.err,
            &format!("popd: {target}: No such file or directory"),
        );
        return 1;
    }
    print_directory_stack(interp, false, io);
    0
}

fn cmd_dirs(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    match args {
        [] => print_directory_stack(interp, false, io),
        [flag] if flag == "-p" => print_directory_stack(interp, true, io),
        [flag] if flag == "-c" => interp.directory_stack.clear(),
        _ => {
            ewln(io.err, "dirs: only -c and -p are supported");
            return 2;
        }
    }
    0
}

fn print_directory_stack(interp: &CommandContext<'_>, one_per_line: bool, io: &mut Io) {
    let entries = std::iter::once(&interp.cwd)
        .chain(interp.directory_stack.iter().rev())
        .cloned()
        .collect::<Vec<_>>();
    if one_per_line {
        for entry in entries {
            wln(io.out, &entry);
        }
    } else {
        wln(io.out, &entries.join(" "));
    }
}

fn cmd_export(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut status = 0;
    for a in args {
        if let Some((k, v)) = a.split_once('=') {
            if interp.readonly.contains(k) {
                ewln(io.err, &format!("export: {k}: readonly variable"));
                status = 1;
                continue;
            }
            interp.set_var(k, v);
            interp.export(k);
        } else {
            interp.export(a);
        }
    }
    status
}

fn cmd_unset(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut status = 0;
    for a in args {
        if a == "-v" || a == "-f" {
            continue;
        }
        // unset arr[i] / unset arr[key] removes one element; unset name removes the whole var.
        if let Some(br) = a.find('[') {
            if a.ends_with(']') {
                let name = &a[..br];
                if interp.readonly.contains(name) {
                    ewln(io.err, &format!("unset: {name}: readonly variable"));
                    status = 1;
                    continue;
                }
                let key = &a[br + 1..a.len() - 1];
                let key = key.trim_matches('"').trim_matches('\'');
                interp.array_unset_elem(name, key);
                continue;
            }
        }
        if interp.readonly.contains(a) {
            ewln(io.err, &format!("unset: {a}: readonly variable"));
            status = 1;
            continue;
        }
        interp.vars.remove(a);
        interp.arrays.remove(a);
        interp.exported.remove(a);
        interp.funcs.remove(a);
    }
    status
}

fn cmd_set(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "-e" => interp.opt_errexit = true,
            "+e" => interp.opt_errexit = false,
            "-u" => interp.opt_nounset = true,
            "+u" => interp.opt_nounset = false,
            "-x" | "+x" => {
                ewln(io.err, "set: xtrace is unimplemented");
                return 2;
            }
            "-o" => {
                if let Some(opt) = args.get(i + 1) {
                    match opt.as_str() {
                        "pipefail" => interp.opt_pipefail = true,
                        "errexit" => interp.opt_errexit = true,
                        "nounset" => interp.opt_nounset = true,
                        _ => {
                            ewln(io.err, &format!("set: unimplemented option '{opt}'"));
                            return 2;
                        }
                    }
                    i += 1;
                }
            }
            "+o" => {
                if let Some(opt) = args.get(i + 1) {
                    match opt.as_str() {
                        "pipefail" => interp.opt_pipefail = false,
                        "errexit" => interp.opt_errexit = false,
                        "nounset" => interp.opt_nounset = false,
                        _ => {
                            ewln(io.err, &format!("set: unimplemented option '{opt}'"));
                            return 2;
                        }
                    }
                    i += 1;
                }
            }
            "--" => {
                interp.positional = args[i + 1..].to_vec();
                break;
            }
            s if s.starts_with('-') && s.len() > 1 => {
                for option in s[1..].chars() {
                    match option {
                        'e' => interp.opt_errexit = true,
                        'u' => interp.opt_nounset = true,
                        'x' => {
                            ewln(io.err, "set: xtrace is unimplemented");
                            return 2;
                        }
                        'o' if args.get(i + 1).map(String::as_str) == Some("pipefail") => {
                            interp.opt_pipefail = true;
                            i += 1;
                        }
                        'o' => {
                            ewln(io.err, "set: option name required after -o");
                            return 2;
                        }
                        other => {
                            ewln(io.err, &format!("set: unimplemented option '-{other}'"));
                            return 2;
                        }
                    }
                }
            }
            s if !s.starts_with('-') && !s.starts_with('+') => {
                // set positional params
                interp.positional = args[i..].to_vec();
                break;
            }
            _ => {}
        }
        i += 1;
    }
    0
}

fn cmd_declare(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut assoc = false;
    let mut indexed = false;
    let mut print = false;
    let mut status = 0;
    for a in args {
        if a == "--" {
            continue;
        }
        if a.starts_with('-') && a.len() > 1 {
            assoc |= a.contains('A');
            indexed |= a.contains('a');
            print |= a.contains('p');
            continue;
        }
        // operand: NAME, NAME=value, NAME=( … ), NAME[sub]=value
        let name = declare_name_of(a);
        if interp.readonly.contains(name) && split_decl_assign(a).is_some() {
            ewln(io.err, &format!("declare: {name}: readonly variable"));
            status = 1;
            continue;
        }
        // Establish the array kind first so a following literal lands in the right store.
        if assoc {
            interp.declare_assoc(name);
        } else if indexed {
            interp.declare_indexed(name);
        }
        if print {
            print_declared(interp, name, io);
            continue;
        }
        if let Some((raw_key, raw_val)) = split_decl_assign(a) {
            // Array literals / subscripts arrive verbatim (unexpanded) and are parsed by
            // apply_assignment; plain scalars already had argv expansion, so just store them.
            if crate::exec::is_array_assign_word(a) {
                crate::exec::apply_assignment(interp, &raw_key, &raw_val);
            } else {
                let v = raw_val.trim_matches('"').trim_matches('\'');
                interp.set_var(&raw_key, v);
            }
        }
    }
    status
}

fn cmd_local(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if !interp.in_function_scope() {
        ewln(io.err, "local: can only be used in a function");
        return 1;
    }
    for argument in args.iter().filter(|argument| !argument.starts_with('-')) {
        interp.declare_local(declare_name_of(argument));
    }
    cmd_declare(interp, args, io)
}

/// The bare variable name of a declare operand (`NAME`, `NAME=…`, `NAME[i]=…`, `NAME+=…`).
fn declare_name_of(a: &str) -> &str {
    let lhs = a.split('=').next().unwrap_or(a);
    let lhs = lhs.strip_suffix('+').unwrap_or(lhs);
    match lhs.find('[') {
        Some(i) => &lhs[..i],
        None => lhs,
    }
}

/// Split a declare operand into (raw_key, raw_val) preserving `[sub]` / `+` markers, or None
/// when there's no `=`.
fn split_decl_assign(a: &str) -> Option<(String, String)> {
    let eq = a.find('=')?;
    Some((a[..eq].to_string(), a[eq + 1..].to_string()))
}

fn print_declared(interp: &Interp, name: &str, io: &mut Io) {
    use crate::interp::ArrayVal;
    match interp.arrays.get(name) {
        Some(ArrayVal::Indexed(_)) => {
            let mut s = String::from("declare -a ");
            s.push_str(name);
            s.push_str("=(");
            let parts: Vec<String> = interp
                .array_keys(name)
                .into_iter()
                .map(|k| {
                    let v = interp.array_get(name, &k).unwrap_or_default();
                    format!("[{k}]=\"{v}\"")
                })
                .collect();
            s.push_str(&parts.join(" "));
            s.push(')');
            wln(io.out, &s);
        }
        Some(ArrayVal::Assoc(_)) => {
            let mut s = String::from("declare -A ");
            s.push_str(name);
            s.push_str("=(");
            let parts: Vec<String> = interp
                .array_keys(name)
                .into_iter()
                .map(|k| {
                    let v = interp.array_get(name, &k).unwrap_or_default();
                    format!("[{k}]=\"{v}\"")
                })
                .collect();
            s.push_str(&parts.join(" "));
            s.push(')');
            wln(io.out, &s);
        }
        None => {
            if let Some(v) = interp.get_var(name) {
                wln(io.out, &format!("declare -- {name}=\"{v}\""));
            }
        }
    }
}

fn start_source(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let Some(path) = args.first() else {
        ewln(io.err, "source: filename argument required");
        return CommandPoll::Ready(2);
    };
    let source = match interp.fs_read(&interp.cwd, path) {
        Ok(source) => String::from_utf8_lossy(&source).into_owned(),
        Err(_) => {
            ewln(
                io.err,
                &format!("source: {path}: No such file or directory"),
            );
            return CommandPoll::Ready(1);
        }
    };
    match crate::commands::parse_shell_source(interp, &source, io.err) {
        Ok(ast) => CommandPoll::InlineSource(ast),
        Err(status) => CommandPoll::Ready(status),
    }
}

fn start_eval(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let source = args.join(" ");
    match crate::commands::parse_shell_source(interp, &source, io.err) {
        Ok(ast) => CommandPoll::Inline(ast),
        Err(status) => CommandPoll::Ready(status),
    }
}

fn cmd_exit(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.len() > 1 {
        ewln(io.err, "exit: too many arguments");
        return 1;
    }
    let code = match args.first() {
        Some(value) => match value.parse::<i32>() {
            Ok(value) => value.rem_euclid(256),
            Err(_) => {
                ewln(io.err, &format!("exit: {value}: numeric argument required"));
                interp.exiting = Some(2);
                return 2;
            }
        },
        None => interp.last_status,
    };
    interp.exiting = Some(code);
    code
}

fn cmd_return(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if !interp.in_function_scope() && interp.source_depth == 0 {
        ewln(io.err, "return: can only be used in a function");
        return 1;
    }
    let code = match parse_control_status("return", args, interp.last_status, io) {
        Ok(code) => code,
        Err(status) => return status,
    };
    interp.returning = Some(code);
    code
}

fn cmd_break(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if interp.loop_depth == 0 {
        ewln(io.err, "break: only meaningful in a loop");
        return 1;
    }
    let count = match parse_loop_count("break", args, io) {
        Ok(count) => count,
        Err(status) => return status,
    };
    interp.loop_break = count;
    0
}

fn cmd_continue(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if interp.loop_depth == 0 {
        ewln(io.err, "continue: only meaningful in a loop");
        return 1;
    }
    let count = match parse_loop_count("continue", args, io) {
        Ok(count) => count,
        Err(status) => return status,
    };
    interp.loop_continue = count;
    0
}

fn parse_control_status(
    command: &str,
    args: &[String],
    default: i32,
    io: &mut Io,
) -> Result<i32, i32> {
    if args.len() > 1 {
        ewln(io.err, &format!("{command}: too many arguments"));
        return Err(1);
    }
    match args.first() {
        Some(value) => value
            .parse::<i32>()
            .map(|value| value.rem_euclid(256))
            .map_err(|_| {
                ewln(
                    io.err,
                    &format!("{command}: {value}: numeric argument required"),
                );
                2
            }),
        None => Ok(default),
    }
}

fn parse_loop_count(command: &str, args: &[String], io: &mut Io) -> Result<u32, i32> {
    if args.len() > 1 {
        ewln(io.err, &format!("{command}: too many arguments"));
        return Err(1);
    }
    match args.first().map(|value| value.parse::<u32>()) {
        Some(Ok(0) | Err(_)) => {
            ewln(
                io.err,
                &format!("{command}: loop count must be a positive integer"),
            );
            Err(1)
        }
        Some(Ok(value)) => Ok(value),
        None => Ok(1),
    }
}

fn cmd_shift(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    let n: usize = match args.first() {
        Some(value) => match value.parse() {
            Ok(value) => value,
            Err(_) => return 1,
        },
        None => 1,
    };
    if n > interp.positional.len() {
        return 1;
    }
    for _ in 0..n {
        if interp.positional.is_empty() {
            break;
        }
        interp.positional.remove(0);
    }
    0
}

fn cmd_getopts(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.len() < 2 {
        ewln(io.err, "getopts: usage: getopts optstring name [arg ...]");
        return 2;
    }
    let optstring = &args[0];
    let name = &args[1];
    let operands = if args.len() > 2 {
        args[2..].to_vec()
    } else {
        interp.positional.clone()
    };
    let visible_optind = interp
        .get_var("OPTIND")
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1);
    if visible_optind != interp.getopts.optind {
        interp.getopts.optind = visible_optind;
        interp.getopts.offset = 1;
    }

    let argument_index = interp.getopts.optind - 1;
    let Some(argument) = operands.get(argument_index) else {
        set_getopts_optind(interp);
        return 1;
    };
    if argument == "--" {
        interp.getopts.optind += 1;
        interp.getopts.offset = 1;
        set_getopts_optind(interp);
        return 1;
    }
    let option_characters = argument.chars().collect::<Vec<_>>();
    if !argument.starts_with('-')
        || argument == "-"
        || interp.getopts.offset >= option_characters.len()
    {
        set_getopts_optind(interp);
        return 1;
    }

    let option = option_characters[interp.getopts.offset];
    let silent = optstring.starts_with(':');
    let specification = optstring
        .trim_start_matches(':')
        .chars()
        .collect::<Vec<_>>();
    let Some(specification_index) = specification
        .iter()
        .position(|candidate| *candidate == option)
    else {
        advance_getopts(interp, option_characters.len());
        if silent {
            interp.set_var("OPTARG", option.to_string());
        } else {
            interp.vars.remove("OPTARG");
        }
        interp.set_var(name, "?");
        set_getopts_optind(interp);
        if !silent {
            ewln(io.err, &format!("getopts: illegal option -- {option}"));
        }
        return 0;
    };
    let requires_argument = specification.get(specification_index + 1) == Some(&':');
    if !requires_argument {
        advance_getopts(interp, option_characters.len());
        interp.vars.remove("OPTARG");
        interp.set_var(name, option.to_string());
        set_getopts_optind(interp);
        return 0;
    }

    let inline = option_characters[interp.getopts.offset + 1..]
        .iter()
        .collect::<String>();
    let option_argument = if !inline.is_empty() {
        interp.getopts.optind += 1;
        inline
    } else if let Some(value) = operands.get(argument_index + 1) {
        interp.getopts.optind += 2;
        value.clone()
    } else {
        interp.getopts.optind += 1;
        interp.getopts.offset = 1;
        set_getopts_optind(interp);
        interp.set_var(name, if silent { ":" } else { "?" });
        if silent {
            interp.set_var("OPTARG", option.to_string());
        } else {
            interp.vars.remove("OPTARG");
            ewln(
                io.err,
                &format!("getopts: option requires an argument -- {option}"),
            );
        }
        return 0;
    };
    interp.getopts.offset = 1;
    interp.set_var("OPTARG", option_argument);
    interp.set_var(name, option.to_string());
    set_getopts_optind(interp);
    0
}

fn advance_getopts(interp: &mut CommandContext<'_>, argument_length: usize) {
    interp.getopts.offset += 1;
    if interp.getopts.offset >= argument_length {
        interp.getopts.optind += 1;
        interp.getopts.offset = 1;
    }
}

fn set_getopts_optind(interp: &mut CommandContext<'_>) {
    let value = interp.getopts.optind.to_string();
    interp.vars.insert("OPTIND".to_string(), value);
}

fn cmd_test(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    eval_test_cmd(interp, "test", args, io)
}

fn cmd_bracket(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    eval_test_cmd(interp, "[", args, io)
}

fn cmd_dbracket(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    eval_test_cmd(interp, "[[", args, io)
}

fn eval_test_cmd(interp: &mut Interp, cmd: &str, args: &[String], io: &mut Io) -> i32 {
    let mut a: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let expected = if cmd == "[[" { "]]" } else { "]" };
    if a.last() == Some(&expected) {
        a.pop();
    } else if cmd == "[" || cmd == "[[" {
        ewln(io.err, &format!("{cmd}: missing '{expected}'"));
        return 2;
    }
    if a.len() == 3
        && matches!(a[1], "-eq" | "-ne" | "-lt" | "-le" | "-gt" | "-ge")
        && (a[0].trim().parse::<i64>().is_err() || a[2].trim().parse::<i64>().is_err())
    {
        ewln(io.err, &format!("{cmd}: integer expression expected"));
        return 2;
    }
    let r = eval_test(interp, &a);
    if r {
        0
    } else {
        1
    }
}

fn eval_test(interp: &mut Interp, a: &[&str]) -> bool {
    let a = strip_test_parens(a);
    if let Some(pos) = top_level_test_operator(a, &["-o", "||"]) {
        return eval_test(interp, &a[..pos]) || eval_test(interp, &a[pos + 1..]);
    }
    if let Some(pos) = top_level_test_operator(a, &["-a", "&&"]) {
        return eval_test(interp, &a[..pos]) && eval_test(interp, &a[pos + 1..]);
    }
    if a.first() == Some(&"!") {
        return !eval_test(interp, &a[1..]);
    }
    match a.len() {
        0 => false,
        1 => !a[0].is_empty(),
        2 => {
            let (op, x) = (a[0], a[1]);
            match op {
                "-z" => x.is_empty(),
                "-n" => !x.is_empty(),
                "-e" | "-a" => interp.fs_metadata(&interp.cwd, x, true).is_ok(),
                "-f" => matches!(
                    interp
                        .fs_metadata(&interp.cwd, x, true)
                        .map(|node| node.kind),
                    Ok(crate::vfs::NodeKind::File(_))
                ),
                "-d" => matches!(
                    interp
                        .fs_metadata(&interp.cwd, x, true)
                        .map(|node| node.kind),
                    Ok(crate::vfs::NodeKind::Dir)
                ),
                "-s" => interp
                    .fs_file_len(&interp.cwd, x)
                    .is_ok_and(|size| size > 0),
                "-r" => interp
                    .fs_metadata(&interp.cwd, x, true)
                    .is_ok_and(|node| node.mode & 0o444 != 0),
                "-w" => interp
                    .fs_metadata(&interp.cwd, x, true)
                    .is_ok_and(|node| node.mode & 0o222 != 0),
                "-x" => interp
                    .fs_metadata(&interp.cwd, x, true)
                    .is_ok_and(|node| node.mode & 0o111 != 0),
                "-L" | "-h" => matches!(
                    interp
                        .fs_metadata(&interp.cwd, x, false)
                        .map(|node| node.kind),
                    Ok(crate::vfs::NodeKind::Symlink(_))
                ),
                "-v" => interp.get_var(x).is_some() || interp.arrays.contains_key(x),
                "!" => !eval_test(interp, &a[1..]),
                _ => !op.is_empty(),
            }
        }
        3 => {
            let (x, op, y) = (a[0], a[1], a[2]);
            match op {
                "=" | "==" => crate::commands::util::glob_eq(y, x),
                "!=" => !crate::commands::util::glob_eq(y, x),
                "-eq" => num(x) == num(y),
                "-ne" => num(x) != num(y),
                "-lt" => num(x) < num(y),
                "-le" => num(x) <= num(y),
                "-gt" => num(x) > num(y),
                "-ge" => num(x) >= num(y),
                "<" => x < y,
                ">" => x > y,
                "=~" => regex::Regex::new(y).is_ok_and(|regex| {
                    let Some(captures) = regex.captures(x) else {
                        interp.set_array("BASH_REMATCH", Vec::new());
                        return false;
                    };
                    interp.set_array(
                        "BASH_REMATCH",
                        captures
                            .iter()
                            .map(|capture| capture.map_or("", |value| value.as_str()).to_string())
                            .collect(),
                    );
                    true
                }),
                "-nt" => {
                    let left = interp.fs_metadata(&interp.cwd, x, true).ok();
                    let right = interp.fs_metadata(&interp.cwd, y, true).ok();
                    left.as_ref().is_some_and(|left| {
                        right.as_ref().is_none_or(|right| left.mtime > right.mtime)
                    })
                }
                "-ot" => {
                    let left = interp.fs_metadata(&interp.cwd, x, true).ok();
                    let right = interp.fs_metadata(&interp.cwd, y, true).ok();
                    right.as_ref().is_some_and(|right| {
                        left.as_ref().is_none_or(|left| left.mtime < right.mtime)
                    })
                }
                _ => false,
            }
        }
        _ => false,
    }
}

fn strip_test_parens<'a>(mut args: &'a [&'a str]) -> &'a [&'a str] {
    while args.first() == Some(&"(") && args.last() == Some(&")") {
        let mut depth = 0_i32;
        let wraps_all = args.iter().enumerate().all(|(index, token)| {
            match *token {
                "(" => depth += 1,
                ")" => depth -= 1,
                _ => {}
            }
            depth > 0 || index + 1 == args.len()
        });
        if !wraps_all {
            break;
        }
        args = &args[1..args.len() - 1];
    }
    args
}

fn top_level_test_operator(args: &[&str], operators: &[&str]) -> Option<usize> {
    let mut depth = 0_i32;
    for (index, token) in args.iter().enumerate() {
        match *token {
            "(" => depth += 1,
            ")" => depth -= 1,
            _ if depth == 0 && operators.contains(token) => return Some(index),
            _ => {}
        }
    }
    None
}

fn num(s: &str) -> i64 {
    s.trim().parse().unwrap_or(0)
}

fn cmd_read(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, long) = split_flags(args);
    if !long.is_empty() || flags.iter().any(|flag| !matches!(flag, 'a' | 'r')) {
        ewln(io.err, "read: supported options are -a and -r");
        return 2;
    }
    // `read -a arr`: split the line into an indexed array (the name follows `-a`).
    if flags.contains(&'a') {
        let line = read_one_line(interp, io);
        let Some(line) = line else { return 1 };
        let line = if flags.contains(&'r') {
            line
        } else {
            decode_read_backslashes(&line)
        };
        let arr = ops.first().map(|s| s.as_str()).unwrap_or("REPLY");
        let ifs = interp.get_var("IFS").unwrap_or_else(|| " \t\n".to_string());
        let elems: Vec<String> = if ifs.is_empty() {
            vec![line]
        } else {
            split_read_fields(&line, &ifs)
        };
        interp.set_array(arr, elems);
        return 0;
    }
    // Obtain a line: from explicit stdin (pipe) if present, else from the persistent input
    // cursor (set up by a `< file` redirect on an enclosing loop).
    let Some(line) = read_one_line(interp, io) else {
        return 1;
    };
    let line = if flags.contains(&'r') {
        line
    } else {
        decode_read_backslashes(&line)
    };

    let ifs = interp.get_var("IFS").unwrap_or_else(|| " \t\n".to_string());
    if ops.is_empty() {
        interp.set_var("REPLY", line);
    } else if ifs.is_empty() {
        interp.set_var(ops[0], line);
        for v in &ops[1..] {
            interp.set_var(v, "");
        }
    } else {
        let parts = split_read_fields(&line, &ifs);
        let separator = ifs.chars().next().unwrap_or(' ').to_string();
        for (i, var) in ops.iter().enumerate() {
            if i == ops.len() - 1 {
                interp.set_var(var, parts.get(i..).unwrap_or_default().join(&separator));
            } else {
                interp.set_var(var, parts.get(i).map_or("", String::as_str));
            }
        }
    }
    0
}

/// Split a `read` record according to POSIX IFS classes.
///
/// IFS whitespace collapses and trims. Every non-whitespace IFS character is a delimiter in its
/// own right, so adjacent delimiters retain an empty field for assignment to the final variable.
fn split_read_fields(line: &str, ifs: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut after_ifs_whitespace = false;
    for character in line.chars() {
        if ifs.contains(character) && matches!(character, ' ' | '\t' | '\n') {
            if !field.is_empty() {
                fields.push(std::mem::take(&mut field));
            }
            after_ifs_whitespace = true;
        } else if ifs.contains(character) {
            if !field.is_empty() {
                fields.push(std::mem::take(&mut field));
            } else if !after_ifs_whitespace {
                fields.push(String::new());
            }
            after_ifs_whitespace = false;
        } else {
            field.push(character);
            after_ifs_whitespace = false;
        }
    }
    if !field.is_empty() {
        fields.push(field);
    }
    fields
}

fn decode_read_backslashes(line: &str) -> String {
    let mut decoded = String::with_capacity(line.len());
    let mut characters = line.chars();
    while let Some(character) = characters.next() {
        if character == '\\' {
            if let Some(escaped) = characters.next() {
                decoded.push(escaped);
            }
        } else {
            decoded.push(character);
        }
    }
    decoded
}

const MAX_MAPFILE_RECORDS: usize = 100_000;

fn cmd_mapfile(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut trim_delimiter = false;
    let mut maximum = None;
    let mut skip = 0usize;
    let mut origin = None;
    let mut delimiter = b'\n';
    let mut name = "MAPFILE".to_string();
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--" => {
                index += 1;
                if index < args.len() {
                    name = args[index].clone();
                    index += 1;
                }
                break;
            }
            "-t" => trim_delimiter = true,
            "-n" | "-s" | "-O" | "-d" => {
                let option = args[index].clone();
                index += 1;
                let Some(value) = args.get(index) else {
                    ewln(io.err, &format!("mapfile: {option} requires a value"));
                    return 2;
                };
                match option.as_str() {
                    "-n" => {
                        maximum = match parse_mapfile_count(value, "-n", io) {
                            Some(0) => None,
                            Some(value) => Some(value),
                            None => return 2,
                        }
                    }
                    "-s" => {
                        let Some(value) = parse_mapfile_count(value, "-s", io) else {
                            return 2;
                        };
                        skip = value;
                    }
                    "-O" => {
                        let Some(value) = parse_mapfile_count(value, "-O", io) else {
                            return 2;
                        };
                        origin = Some(value);
                    }
                    "-d" => {
                        delimiter = if value.is_empty() {
                            0
                        } else {
                            let bytes = value.as_bytes();
                            if bytes.len() != 1 {
                                ewln(io.err, "mapfile: -d requires one byte or an empty value");
                                return 2;
                            }
                            bytes[0]
                        };
                    }
                    _ => unreachable!(),
                }
            }
            option if option.starts_with('-') => {
                ewln(io.err, &format!("mapfile: unsupported option {option}"));
                return 2;
            }
            value => {
                name = value.to_string();
                index += 1;
                break;
            }
        }
        index += 1;
    }
    if index != args.len() || !shell_identifier(&name) {
        ewln(io.err, "mapfile: expected one valid array name");
        return 2;
    }
    let mut output_index = origin.unwrap_or(0);
    if origin.is_none() {
        interp.set_array(&name, Vec::new());
    } else {
        interp.declare_indexed(&name);
    }
    let mut cursor = 0usize;
    let mut seen = 0usize;
    let mut stored = 0usize;
    while cursor < io.stdin.len() {
        let relative_end = io.stdin[cursor..]
            .iter()
            .position(|byte| *byte == delimiter);
        let end = relative_end.map_or(io.stdin.len(), |offset| cursor + offset + 1);
        if seen >= skip {
            if maximum.is_some_and(|limit| stored >= limit) {
                break;
            }
            if stored >= MAX_MAPFILE_RECORDS || output_index >= MAX_MAPFILE_RECORDS {
                ewln(io.err, "mapfile: record limit exceeded");
                return 1;
            }
            let mut record = &io.stdin[cursor..end];
            // Bash variables cannot contain NUL, so `mapfile -d ''` necessarily drops it even
            // without `-t`.
            if (trim_delimiter || delimiter == 0) && record.last() == Some(&delimiter) {
                record = &record[..record.len() - 1];
            }
            interp.array_set(
                &name,
                &output_index.to_string(),
                String::from_utf8_lossy(record).into_owned(),
            );
            output_index = output_index.saturating_add(1);
            stored = stored.saturating_add(1);
        }
        seen = seen.saturating_add(1);
        cursor = end;
    }
    0
}

fn parse_mapfile_count(value: &str, option: &str, io: &mut Io) -> Option<usize> {
    match value.parse::<usize>() {
        Ok(value) => Some(value),
        Err(_) => {
            ewln(
                io.err,
                &format!("mapfile: {option} requires a non-negative integer"),
            );
            None
        }
    }
}

fn shell_identifier(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte == b'_' || byte.is_ascii_alphabetic())
        && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
}

/// Read one line for `read`: from an explicit stdin pipe if present, else the persistent input
/// cursor (a `< file` redirect on an enclosing loop). Returns None at EOF.
fn read_one_line(interp: &mut Interp, io: &Io) -> Option<String> {
    if !io.stdin.is_empty() {
        return Some(
            String::from_utf8_lossy(&io.stdin)
                .lines()
                .next()
                .unwrap_or("")
                .to_string(),
        );
    }
    if interp.input_pos < interp.input_stream.len() {
        let rest = &interp.input_stream[interp.input_pos..];
        let nl = rest.iter().position(|&b| b == b'\n');
        let (line_bytes, adv) = match nl {
            Some(i) => (&rest[..i], i + 1),
            None => (rest, rest.len()),
        };
        let l = String::from_utf8_lossy(line_bytes).into_owned();
        interp.input_pos += adv;
        return Some(l);
    }
    let mut line = Vec::new();
    let mut consumed = false;
    while line.len() < crate::descriptors::MAX_CAPTURE_BYTES {
        match interp.read_fd(0, 1).ok()? {
            crate::descriptors::IoPoll::Ready(bytes) if bytes.is_empty() => break,
            crate::descriptors::IoPoll::Ready(bytes) => {
                consumed = true;
                if bytes[0] == b'\n' {
                    break;
                }
                line.push(bytes[0]);
            }
            crate::descriptors::IoPoll::Blocked(_) => return None,
        }
    }
    if !consumed {
        None
    } else {
        Some(String::from_utf8_lossy(&line).into_owned())
    }
}

fn cmd_which(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    command_lookup(interp, args, io, false, false)
}

fn cmd_type(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    command_lookup(interp, args, io, true, true)
}

fn command_lookup(
    interp: &mut CommandContext<'_>,
    args: &[String],
    io: &mut Io,
    describe: bool,
    include_builtins: bool,
) -> i32 {
    let mut ok = true;
    for a in args.iter().filter(|arg| !arg.starts_with('-')) {
        if interp.funcs.contains_key(a) {
            wln(io.out, &format!("{a} is a function"));
        } else if include_builtins && is_shell_builtin_name(a) {
            if describe {
                wln(io.out, &format!("{a} is a shell builtin"));
            } else {
                wln(io.out, a);
            }
        } else if let crate::commands::util::ExecutableLookup::Found(path) =
            crate::commands::util::resolve_executable(interp, a)
        {
            if describe {
                wln(io.out, &format!("{a} is {path}"));
            } else {
                wln(io.out, &path);
            }
        } else {
            ok = false;
        }
    }
    if ok {
        0
    } else {
        1
    }
}

pub(crate) fn is_shell_builtin_name(name: &str) -> bool {
    matches!(
        name,
        "." | ":"
            | "["
            | "[["
            | "alias"
            | "bg"
            | "break"
            | "cd"
            | "command"
            | "continue"
            | "declare"
            | "disown"
            | "dirs"
            | "echo"
            | "eval"
            | "exec"
            | "exit"
            | "export"
            | "false"
            | "fg"
            | "getopts"
            | "hash"
            | "jobs"
            | "kill"
            | "let"
            | "local"
            | "mapfile"
            | "popd"
            | "printf"
            | "pushd"
            | "pwd"
            | "read"
            | "readarray"
            | "readonly"
            | "return"
            | "set"
            | "shift"
            | "shopt"
            | "source"
            | "test"
            | "trap"
            | "true"
            | "type"
            | "typeset"
            | "ulimit"
            | "umask"
            | "unalias"
            | "unset"
            | "wait"
    )
}

fn start_command(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    if args.is_empty() {
        return CommandPoll::Ready(0);
    }
    if matches!(args.first().map(String::as_str), Some("-v" | "-V")) {
        return CommandPoll::Ready(command_lookup(
            interp,
            &args[1..],
            io,
            args[0] == "-V",
            true,
        ));
    }
    let argv = if args.first().map(String::as_str) == Some("--") {
        &args[1..]
    } else {
        args
    };
    if argv.is_empty() {
        CommandPoll::Ready(0)
    } else {
        CommandPoll::Inline(crate::shell::Node::ArgvCommand(argv.to_vec()))
    }
}

fn cmd_let(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.is_empty() {
        ewln(io.err, "let: expression expected");
        return 1;
    }
    let mut last = 0i64;
    for expression in args {
        match crate::arith::evaluate(&mut **interp, expression) {
            Ok(value) => last = value,
            Err(error) => {
                ewln(io.err, &format!("let: {}", error.describe(expression)));
                return 1;
            }
        }
    }
    i32::from(last == 0)
}
