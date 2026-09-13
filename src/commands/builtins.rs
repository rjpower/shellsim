//! Shell builtins: state-mutating commands (cd, export, set, …), control-flow signals
//! (exit/return/break/continue/shift), `test`/`[`, `read`, `let`, `source`/`eval`, and the
//! small job-table queries. Unsupported process and interactive controls fail explicitly.

use std::collections::HashMap;

use crate::commands::util::{ewln, split_flags, wln, KNOWN_COMMANDS};
use crate::commands::{CommandContext, CommandPoll, CommandResume, CommandSpec, Io, Trust};
use crate::interp::Interp;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::{reg, reg_resumable};
    reg(m, &["cd"], Trust::Real, cmd_cd);
    reg(m, &["pwd"], Trust::Real, cmd_pwd);
    reg(m, &["export"], Trust::Real, cmd_export);
    reg(m, &["unset"], Trust::Real, cmd_unset);
    reg(m, &["set"], Trust::Real, cmd_set);
    reg(
        m,
        &["declare", "typeset", "local", "readonly"],
        Trust::Real,
        cmd_declare,
    );
    reg_resumable(m, &["source", "."], Trust::Real, cmd_source, start_source);
    reg_resumable(m, &["eval"], Trust::Real, cmd_eval, start_eval);
    reg(m, &["exit"], Trust::Real, cmd_exit);
    reg(m, &["return"], Trust::Real, cmd_return);
    reg(m, &["break"], Trust::Real, cmd_break);
    reg(m, &["continue"], Trust::Real, cmd_continue);
    reg(m, &["shift"], Trust::Real, cmd_shift);
    reg(m, &["true", ":"], Trust::Real, cmd_true);
    reg(m, &["false"], Trust::Real, cmd_false);
    reg(m, &["test", "["], Trust::Real, cmd_test);
    reg(m, &["[["], Trust::Real, cmd_dbracket);
    reg(m, &["read"], Trust::Real, cmd_read);
    reg_resumable(m, &["wait"], Trust::Real, cmd_wait, start_wait);
    reg(m, &["jobs"], Trust::Real, cmd_jobs);
    reg(m, &["trap"], Trust::Real, cmd_trap);
    reg(
        m,
        &[
            "disown", "umask", "ulimit", "hash", "complete", "shopt", "bind", "history", "exec",
        ],
        Trust::NoOp,
        cmd_unsupported,
    );
    reg(m, &["kill"], Trust::Real, cmd_kill);
    reg(m, &["killall", "pkill"], Trust::NoOp, cmd_unsupported);
    reg(m, &["type", "which"], Trust::Real, cmd_which);
    reg_resumable(m, &["command"], Trust::Real, cmd_command, start_command);
    reg(m, &["alias"], Trust::Partial, cmd_alias);
    reg(m, &["unalias"], Trust::Real, cmd_unalias);
    reg(m, &["getopts"], Trust::Real, cmd_getopts);
    reg(m, &["let"], Trust::Real, cmd_let);
    reg(m, &["mapfile", "readarray"], Trust::Real, cmd_mapfile);
    reg(m, &["pushd"], Trust::Real, cmd_pushd);
    reg(m, &["popd"], Trust::Real, cmd_popd);
    reg(m, &["dirs"], Trust::Real, cmd_dirs);
}

fn cmd_unsupported(_interp: &mut CommandContext<'_>, _args: &[String], io: &mut Io) -> i32 {
    ewln(io.err, "shellsim: builtin is not supported");
    2
}

const MAX_TRAP_STATE_BYTES: u64 = 1024 * 1024;

fn cmd_trap(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.is_empty() {
        print_traps(interp, &[], io);
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
        let signals = match parse_trap_signals(&args[1..]) {
            Ok(signals) => signals,
            Err(error) => {
                ewln(io.err, &format!("trap: {error}"));
                return 2;
            }
        };
        print_traps(interp, &signals, io);
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
    let signals = match parse_trap_signals(signal_args) {
        Ok(signals) => signals,
        Err(error) => {
            ewln(io.err, &format!("trap: {error}"));
            return 2;
        }
    };
    if signals.contains(&crate::process::Signal::Kill) {
        ewln(io.err, "trap: SIGKILL cannot be caught or ignored");
        return 1;
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
    let state_bytes = updated.values().fold(0_u64, |total, disposition| {
        total.saturating_add(match disposition {
            crate::interp::ShellSignalDisposition::Ignore => 16,
            crate::interp::ShellSignalDisposition::Handler { source, body } => (source.len()
                as u64)
                .saturating_add(body.estimated_bytes())
                .saturating_add(32),
        })
    });
    if state_bytes > MAX_TRAP_STATE_BYTES {
        ewln(io.err, "trap: signal handler state exceeds the 1 MiB limit");
        return 2;
    }
    interp.signal_dispositions = updated;
    0
}

fn parse_trap_signals(values: &[String]) -> Result<Vec<crate::process::Signal>, String> {
    let mut signals = Vec::with_capacity(values.len());
    for value in values {
        if value == "0" || value.eq_ignore_ascii_case("EXIT") {
            return Err("EXIT traps are not supported".to_string());
        }
        let signal = crate::process::Signal::parse(value)
            .ok_or_else(|| format!("invalid signal: {value}"))?;
        if !signals.contains(&signal) {
            signals.push(signal);
        }
    }
    Ok(signals)
}

fn print_traps(interp: &CommandContext<'_>, selected: &[crate::process::Signal], io: &mut Io) {
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
            let state = if job.done { "Done" } else { "Running" };
            wln(io.out, &format!("[{}] {state} {}", job.id, job.cmd));
        }
    }
    0
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
            Some(crate::process::ProcessStatus::Running)
        ),
        KillTarget::Group(process_group) => {
            !interp.processes.running_group(process_group).is_empty()
        }
    }
}

fn list_signal(argument: Option<&String>, io: &mut Io) -> i32 {
    let Some(argument) = argument else {
        wln(io.out, "HUP INT KILL PIPE TERM CHLD");
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

fn cmd_wait(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.is_empty() {
        let pids = interp.jobs.iter().map(|job| job.pid).collect::<Vec<_>>();
        interp.jobs.clear();
        for pid in pids {
            interp.processes.reap(pid);
            let _ = interp.scheduler.reap(pid);
        }
        return 0;
    }
    let mut status = 0;
    for argument in args {
        let parsed = argument
            .strip_prefix('%')
            .unwrap_or(argument)
            .parse::<u32>();
        let Ok(identifier) = parsed else {
            ewln(io.err, &format!("wait: {argument}: invalid job id"));
            status = 127;
            continue;
        };
        let position = if argument.starts_with('%') {
            interp.jobs.iter().position(|job| job.id == identifier)
        } else {
            interp.jobs.iter().position(|job| job.pid == identifier)
        };
        match position {
            Some(position) if interp.jobs[position].done => {
                let job = interp.jobs.remove(position);
                status = job.status;
                interp.processes.reap(job.pid);
                let _ = interp.scheduler.reap(job.pid);
            }
            Some(_) => {
                ewln(io.err, &format!("wait: {argument}: job is not complete"));
                status = 127;
            }
            None => {
                ewln(io.err, &format!("wait: {argument}: no such job"));
                status = 127;
            }
        }
    }
    status
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

fn cmd_getopts(_interp: &mut CommandContext<'_>, _args: &[String], _io: &mut Io) -> i32 {
    1 // signal "no more options" — scripts usually guard on this
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
    wln(io.out, &interp.cwd);
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
    let absolute = crate::vfs::resolve_against(&interp.cwd, target);
    if !interp.vfs.is_dir("/", &absolute) {
        return Err(());
    }
    let old = interp.cwd.clone();
    interp.set_var("OLDPWD", old);
    interp.cwd = interp.vfs.realpath(&absolute, true).unwrap_or(absolute);
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

fn cmd_export(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    for a in args {
        if let Some((k, v)) = a.split_once('=') {
            interp.set_var(k, v);
            interp.export(k);
        } else {
            interp.export(a);
        }
    }
    0
}

fn cmd_unset(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    for a in args {
        if a == "-v" || a == "-f" {
            continue;
        }
        // unset arr[i] / unset arr[key] removes one element; unset name removes the whole var.
        if let Some(br) = a.find('[') {
            if a.ends_with(']') {
                let name = &a[..br];
                let key = &a[br + 1..a.len() - 1];
                let key = key.trim_matches('"').trim_matches('\'');
                interp.array_unset_elem(name, key);
                continue;
            }
        }
        interp.vars.remove(a);
        interp.arrays.remove(a);
        interp.exported.remove(a);
        interp.funcs.remove(a);
    }
    0
}

fn cmd_set(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "-e" => interp.opt_errexit = true,
            "+e" => interp.opt_errexit = false,
            "-u" => interp.opt_nounset = true,
            "+u" => interp.opt_nounset = false,
            "-x" => interp.opt_xtrace = true,
            "+x" => interp.opt_xtrace = false,
            "-o" => {
                if let Some(opt) = args.get(i + 1) {
                    match opt.as_str() {
                        "pipefail" => interp.opt_pipefail = true,
                        "errexit" => interp.opt_errexit = true,
                        "nounset" => interp.opt_nounset = true,
                        _ => {}
                    }
                    i += 1;
                }
            }
            "+o" => {
                if let Some(opt) = args.get(i + 1) {
                    if opt == "pipefail" {
                        interp.opt_pipefail = false;
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
                        'x' => interp.opt_xtrace = true,
                        'o' if args.get(i + 1).map(String::as_str) == Some("pipefail") => {
                            interp.opt_pipefail = true;
                            i += 1;
                        }
                        'o' => {}
                        _ => {}
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
    0
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

fn cmd_source(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let Some(path) = args.first() else { return 0 };
    match interp.vfs.read_string(&interp.cwd, path) {
        Ok(src) => interp.run_script_into(&src, io.out, io.err),
        Err(_) => {
            ewln(
                io.err,
                &format!("source: {path}: No such file or directory"),
            );
            1
        }
    }
}

fn start_source(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let Some(path) = args.first() else {
        return CommandPoll::Ready(0);
    };
    let source = match interp.vfs.read_string(&interp.cwd, path) {
        Ok(source) => source,
        Err(_) => {
            ewln(
                io.err,
                &format!("source: {path}: No such file or directory"),
            );
            return CommandPoll::Ready(1);
        }
    };
    match crate::commands::parse_shell_source(interp, &source, io.err) {
        Ok(ast) => CommandPoll::Inline(ast),
        Err(status) => CommandPoll::Ready(status),
    }
}

fn cmd_eval(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let src = args.join(" ");
    interp.run_script_into(&src, io.out, io.err)
}

fn start_eval(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let source = args.join(" ");
    match crate::commands::parse_shell_source(interp, &source, io.err) {
        Ok(ast) => CommandPoll::Inline(ast),
        Err(status) => CommandPoll::Ready(status),
    }
}

fn cmd_exit(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    let code = args
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(interp.last_status);
    interp.exiting = Some(code);
    code
}

fn cmd_return(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    let code = args
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(interp.last_status);
    interp.returning = Some(code);
    code
}

fn cmd_break(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    interp.loop_break = args.first().and_then(|s| s.parse().ok()).unwrap_or(1);
    0
}

fn cmd_continue(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    interp.loop_continue = args.first().and_then(|s| s.parse().ok()).unwrap_or(1);
    0
}

fn cmd_shift(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    let n: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(1);
    for _ in 0..n {
        if interp.positional.is_empty() {
            break;
        }
        interp.positional.remove(0);
    }
    0
}

fn cmd_test(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    eval_test_cmd(interp, "[", args)
}

fn cmd_dbracket(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    eval_test_cmd(interp, "[[", args)
}

fn eval_test_cmd(interp: &mut Interp, cmd: &str, args: &[String]) -> i32 {
    let mut a: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    if (cmd == "[" || cmd == "[[") && a.last() == Some(&"]").or(Some(&"]]")) {
        a.pop();
    }
    if a.last() == Some(&"]") || a.last() == Some(&"]]") {
        a.pop();
    }
    let r = eval_test(interp, &a);
    if r {
        0
    } else {
        1
    }
}

fn eval_test(interp: &Interp, a: &[&str]) -> bool {
    match a.len() {
        0 => false,
        1 => !a[0].is_empty(),
        2 => {
            let (op, x) = (a[0], a[1]);
            match op {
                "-z" => x.is_empty(),
                "-n" => !x.is_empty(),
                "-e" | "-a" => interp.vfs.lexists(&interp.cwd, x),
                "-f" => interp.vfs.is_file(&interp.cwd, x),
                "-d" => interp.vfs.is_dir(&interp.cwd, x),
                "-s" => interp
                    .vfs
                    .read(&interp.cwd, x)
                    .map(|d| !d.is_empty())
                    .unwrap_or(false),
                "-r" | "-w" | "-x" => interp.vfs.lexists(&interp.cwd, x),
                "-L" | "-h" => interp.vfs.is_symlink(&interp.cwd, x),
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
                "=~" => regex::Regex::new(y)
                    .map(|regex| regex.is_match(x))
                    .unwrap_or(false),
                "-nt" => true,
                "-ot" => false,
                _ => false,
            }
        }
        _ => {
            // handle && || and ! and parens minimally: split on -a/-o
            if let Some(pos) = a.iter().position(|s| *s == "-a" || *s == "&&") {
                return eval_test(interp, &a[..pos]) && eval_test(interp, &a[pos + 1..]);
            }
            if let Some(pos) = a.iter().position(|s| *s == "-o" || *s == "||") {
                return eval_test(interp, &a[..pos]) || eval_test(interp, &a[pos + 1..]);
            }
            if a[0] == "!" {
                return !eval_test(interp, &a[1..]);
            }
            false
        }
    }
}

fn num(s: &str) -> i64 {
    s.trim().parse().unwrap_or(0)
}

fn cmd_read(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _long) = split_flags(args);
    // `read -a arr`: split the line into an indexed array (the name follows `-a`).
    if flags.contains(&'a') {
        let line = read_one_line(interp, io);
        let Some(line) = line else { return 1 };
        let arr = ops.first().map(|s| s.as_str()).unwrap_or("REPLY");
        let ifs = interp.get_var("IFS").unwrap_or_else(|| " \t\n".to_string());
        let elems: Vec<String> = if ifs.is_empty() {
            vec![line]
        } else {
            line.split(|c| ifs.contains(c))
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        };
        interp.set_array(arr, elems);
        return 0;
    }
    // Obtain a line: from explicit stdin (pipe) if present, else from the persistent input
    // cursor (set up by a `< file` redirect on an enclosing loop).
    let Some(line) = read_one_line(interp, io) else {
        return 1;
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
        let parts: Vec<&str> = line
            .split(|c| ifs.contains(c))
            .filter(|s| !s.is_empty())
            .collect();
        for (i, var) in ops.iter().enumerate() {
            if i == ops.len() - 1 {
                interp.set_var(var, parts[i..].join(" "));
            } else {
                interp.set_var(var, parts.get(i).copied().unwrap_or(""));
            }
        }
    }
    0
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
            if trim_delimiter && record.last() == Some(&delimiter) {
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
    let known = KNOWN_COMMANDS;
    let mut ok = true;
    for a in args.iter().filter(|arg| !arg.starts_with('-')) {
        if interp.funcs.contains_key(a) {
            wln(io.out, &format!("{a} is a function"));
        } else if known.contains(&a.as_str()) {
            wln(io.out, &format!("/usr/bin/{a}"));
        } else if let crate::commands::util::ExecutableLookup::Found(path) =
            crate::commands::util::resolve_executable(interp, a)
        {
            wln(io.out, &path);
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

fn cmd_command(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.is_empty() {
        return 0;
    }
    if matches!(args.first().map(String::as_str), Some("-v" | "-V")) {
        return cmd_which(interp, &args[1..], io);
    }
    let argv = if args.first().map(String::as_str) == Some("--") {
        &args[1..]
    } else {
        args
    };
    if argv.is_empty() {
        return 0;
    }
    crate::commands::run(interp, argv, io.stdin.clone(), io.out, io.err)
}

fn start_command(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    if args.is_empty() {
        return CommandPoll::Ready(0);
    }
    if matches!(args.first().map(String::as_str), Some("-v" | "-V")) {
        return CommandPoll::Ready(cmd_which(interp, &args[1..], io));
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

fn cmd_let(interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> i32 {
    let mut last = 0i64;
    for a in args {
        if let Some((name, expr)) = a.split_once('=') {
            let v = crate::expand::eval_arith(interp, expr);
            interp.set_var(name, v.to_string());
            last = v;
        } else {
            last = crate::expand::eval_arith(interp, a);
        }
    }
    if last == 0 {
        1
    } else {
        0
    }
}
