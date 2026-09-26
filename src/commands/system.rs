//! Small system-introspection commands. Their answers describe the simulated environment rather
//! than the host running shellsim.

use std::collections::{BTreeMap, HashMap};

use crate::commands::util::{ewln, wln};
use crate::commands::{reg_system, CommandSpec, Io, Trust};
use crate::process::ProcessStatus;
use crate::syscalls::{FileKind, System};

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    reg_system(m, "/usr/bin/printenv", Trust::Real, run_printenv);
    super::reg_system_poll(m, "/usr/bin/envsubst", Trust::Partial, cmd_envsubst);
    reg_system(m, "/usr/bin/uname", Trust::Real, run_uname);
    reg_system(m, "/usr/bin/arch", Trust::Real, run_arch);
    reg_system(m, "/usr/bin/hostname", Trust::Real, cmd_hostname);
    reg_system(m, "/usr/bin/whoami", Trust::Real, run_whoami);
    reg_system(m, "/usr/bin/logname", Trust::Real, run_whoami);
    reg_system(m, "/usr/bin/id", Trust::Real, run_id);
    reg_system(m, "/usr/bin/groups", Trust::Real, run_groups);
    reg_system(m, "/usr/bin/nproc", Trust::Real, run_nproc);
    reg_system(m, "/usr/bin/getconf", Trust::Partial, run_getconf);
    reg_system(m, "/usr/bin/df", Trust::Real, run_df);
    reg_system(m, "/usr/bin/free", Trust::Real, run_free);
    reg_system(m, "/usr/bin/ps", Trust::Partial, cmd_ps);
    reg_system(m, "/usr/bin/pgrep", Trust::Partial, cmd_pgrep);
    reg_system(m, "/usr/bin/pkill", Trust::Partial, cmd_pkill);
    reg_system(m, "/usr/bin/killall", Trust::Partial, cmd_killall);
    reg_system(m, "/usr/bin/lsof", Trust::Partial, cmd_lsof);
    reg_system(m, "/usr/bin/ss", Trust::Partial, cmd_sockets);
    reg_system(m, "/usr/bin/netstat", Trust::Partial, cmd_sockets);
}

pub(crate) struct EnvAction {
    pub(crate) environment: BTreeMap<String, String>,
    pub(crate) cwd: Option<String>,
    pub(crate) argv: Vec<String>,
}

pub(crate) fn parse_env_action(
    system: &mut dyn System,
    args: &[String],
) -> Result<EnvAction, String> {
    let mut environment = system.environment();
    let mut cwd = None;
    let mut index = 0;
    let mut options = true;
    while options && index < args.len() {
        match args[index].as_str() {
            "--" => {
                options = false;
                index += 1;
            }
            "-i" | "--ignore-environment" => {
                environment.clear();
                index += 1;
            }
            "-u" | "--unset" => {
                let name = args
                    .get(index + 1)
                    .ok_or_else(|| "option requires an argument -- 'u'".to_string())?;
                environment.remove(name);
                index += 2;
            }
            "-C" | "--chdir" => {
                let directory = args
                    .get(index + 1)
                    .ok_or_else(|| "option requires an argument -- 'C'".to_string())?;
                let current = system.cwd().to_string();
                if !matches!(
                    system.metadata(&current, directory, true),
                    Ok(info) if info.kind == FileKind::Directory
                ) {
                    return Err(format!("cannot change directory to '{directory}'"));
                }
                cwd = Some(
                    system
                        .canonicalize(&current, directory, true)
                        .map_err(|_| format!("cannot change directory to '{directory}'"))?,
                );
                index += 2;
            }
            option if option.starts_with('-') => {
                return Err(format!("unrecognized option '{option}'"));
            }
            _ => options = false,
        }
    }
    while let Some(argument) = args.get(index) {
        let Some((name, value)) = argument.split_once('=') else {
            break;
        };
        if name.is_empty() || name.contains('=') {
            break;
        }
        environment.insert(name.to_string(), value.to_string());
        index += 1;
    }
    Ok(EnvAction {
        environment,
        cwd,
        argv: args[index..].to_vec(),
    })
}

fn run_printenv(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let env = context.environment;
    if args.is_empty() {
        for (name, value) in env {
            wln(io.out, &format!("{name}={value}"));
        }
        return 0;
    }
    let mut status = 0;
    for name in args {
        if let Some(value) = env.get(name) {
            wln(io.out, value);
        } else {
            status = 1;
        }
    }
    status
}

fn cmd_envsubst(
    context: &mut crate::program::ProcessContext<'_>,
    io: &mut Io,
) -> crate::exec::ShellPoll {
    if !context.args.is_empty() {
        ewln(io.err, "envsubst: unsupported operand");
        return crate::exec::ShellPoll::Ready(2);
    }
    if let Err(poll) = context.read_standard_input(io) {
        return poll;
    }
    let source = String::from_utf8_lossy(&io.stdin);
    let chars = source.chars().collect::<Vec<_>>();
    let mut output = String::new();
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] != '$' {
            output.push(chars[i]);
            i += 1;
            continue;
        }
        i += 1;
        let braced = chars.get(i) == Some(&'{');
        if braced {
            i += 1;
        }
        let start = i;
        while i < chars.len() && (chars[i] == '_' || chars[i].is_ascii_alphanumeric()) {
            i += 1;
        }
        let name = chars[start..i].iter().collect::<String>();
        if braced && chars.get(i) == Some(&'}') {
            i += 1;
        }
        if name.is_empty() {
            output.push('$');
        } else {
            output.push_str(context.environment.get(&name).map_or("", String::as_str));
        }
    }
    io.print(&output);
    crate::exec::ShellPoll::Ready(0)
}

fn run_uname(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let mut flags = Vec::new();
    for argument in args {
        if argument == "--all" {
            flags.extend(['s', 'n', 'r', 'v', 'm', 'o']);
            continue;
        }
        let Some(value) = argument.strip_prefix('-') else {
            ewln(io.err, "uname: extra operand");
            return 1;
        };
        if value.is_empty()
            || value
                .chars()
                .any(|flag| !matches!(flag, 'a' | 's' | 'n' | 'r' | 'v' | 'm' | 'o'))
        {
            ewln(io.err, &format!("uname: unimplemented option '{argument}'"));
            return 2;
        }
        for flag in value.chars() {
            if flag == 'a' {
                flags.extend(['s', 'n', 'r', 'v', 'm', 'o']);
            } else {
                flags.push(flag);
            }
        }
    }
    if flags.is_empty() {
        flags.push('s');
    }
    let mut values = Vec::new();
    for flag in ['s', 'n', 'r', 'v', 'm', 'o'] {
        if flags.contains(&flag) {
            values.push(match flag {
                's' => "Linux",
                'n' => context.system.hostname(),
                'r' => "6.6.0-shellsim",
                'v' => "#1 SMP",
                'm' => "x86_64",
                'o' => "GNU/Linux",
                _ => unreachable!(),
            });
        }
    }
    wln(io.out, &values.join(" "));
    0
}

fn run_arch(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    if !args.is_empty() {
        ewln(io.err, "arch: unimplemented option or operand");
        return 2;
    }
    wln(io.out, "x86_64");
    0
}

fn cmd_hostname(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    if args.iter().any(|arg| arg.starts_with('-') && arg != "-f") {
        ewln(io.err, "hostname: unimplemented option");
        return 2;
    }
    if let Some(name) = args.iter().find(|arg| !arg.starts_with('-')) {
        if let Err(error) = context.system.set_hostname(name) {
            ewln(io.err, &format!("hostname: {error}"));
            return 1;
        }
    } else {
        wln(io.out, context.system.hostname());
    }
    0
}

fn run_whoami(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    if !args.is_empty() {
        ewln(io.err, "whoami: unimplemented option or operand");
        return 2;
    }
    wln(
        io.out,
        if context.system.uid() == 0 {
            "root"
        } else {
            "user"
        },
    );
    0
}

fn run_id(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let mut user = false;
    let mut group = false;
    let mut name_output = false;
    for argument in args {
        let Some(options) = argument.strip_prefix('-').filter(|value| !value.is_empty()) else {
            ewln(io.err, "id: unimplemented user operand");
            return 2;
        };
        for option in options.chars() {
            match option {
                'u' => user = true,
                'g' => group = true,
                'n' => name_output = true,
                _ => {
                    ewln(io.err, &format!("id: unimplemented option '-{option}'"));
                    return 2;
                }
            }
        }
    }
    if user && group {
        ewln(io.err, "id: cannot print only user and only group");
        return 1;
    }
    if name_output && !user && !group {
        ewln(io.err, "id: option '-n' requires '-u' or '-g'");
        return 1;
    }
    let uid = context.system.uid();
    let name = if uid == 0 { "root" } else { "user" };
    if user || group {
        let value = if name_output {
            name.to_string()
        } else {
            uid.to_string()
        };
        wln(io.out, &value);
    } else {
        wln(
            io.out,
            &format!(
                "uid={}({name}) gid={}({name}) groups={}({name})",
                uid, uid, uid
            ),
        );
    }
    0
}

fn run_groups(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    if !args.is_empty() {
        ewln(io.err, "groups: unimplemented user operand");
        return 2;
    }
    wln(
        io.out,
        if context.system.uid() == 0 {
            "root"
        } else {
            "user"
        },
    );
    0
}

fn run_nproc(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    if !args.is_empty() {
        ewln(io.err, "nproc: unimplemented option");
        return 2;
    }
    wln(io.out, "1");
    0
}

fn run_getconf(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let Some(name) = args.first() else { return 1 };
    let value = match name.as_str() {
        "_NPROCESSORS_ONLN" | "NPROCESSORS_ONLN" => "1",
        "PAGESIZE" | "PAGE_SIZE" => "4096",
        "ARG_MAX" => "2097152",
        "PATH_MAX" => "4096",
        "OPEN_MAX" => "256",
        _ => {
            ewln(io.err, &format!("getconf: Unrecognized variable `{name}'"));
            return 2;
        }
    };
    wln(io.out, value);
    0
}

fn run_df(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    if args
        .iter()
        .any(|arg| arg.starts_with('-') && arg != "-h" && arg != "-k" && arg != "-P")
    {
        ewln(io.err, "df: unimplemented option");
        return 2;
    }
    let human = args.iter().any(|arg| arg.contains('h'));
    let cwd = system.cwd().to_string();
    for path in args.iter().filter(|argument| !argument.starts_with('-')) {
        if system.metadata(&cwd, path, true).is_err() {
            ewln(io.err, &format!("df: {path}: No such file or directory"));
            return 1;
        }
    }
    let limit = system.limits().disk;
    let used = system.disk_used();
    let available = limit.saturating_sub(used);
    wln(io.out, "Filesystem      Size  Used Avail Use% Mounted on");
    let percent = used.saturating_mul(100).checked_div(limit).unwrap_or(100);
    if human {
        wln(
            io.out,
            &format!(
                "shellsim       {}  {}  {}  {percent}% /",
                human_size(limit),
                human_size(used),
                human_size(available)
            ),
        );
    } else {
        wln(
            io.out,
            &format!(
                "shellsim       {}  {}  {}  {percent}% /",
                limit / 1024,
                used / 1024,
                available / 1024
            ),
        );
    }
    0
}

fn run_free(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    if args.iter().any(|arg| arg != "-b") {
        ewln(io.err, "free: unimplemented option or operand");
        return 2;
    }
    let bytes = args.iter().any(|arg| arg == "-b");
    let divisor = if bytes { 1 } else { 1024 };
    let limit = system.limits().memory;
    let used = system.memory_used();
    wln(
        io.out,
        "              total        used        free      shared  buff/cache   available",
    );
    wln(
        io.out,
        &format!(
            "Mem:     {:>10}  {:>10}  {:>10}           0           0  {:>10}",
            limit / divisor,
            used / divisor,
            limit.saturating_sub(used) / divisor,
            limit.saturating_sub(used) / divisor
        ),
    );
    wln(io.out, "Swap:             0           0           0");
    0
}

fn cmd_ps(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let mut aux = false;
    let mut full = false;
    let mut columns: Option<Vec<&str>> = None;
    let mut selected_pids: Option<Vec<u32>> = None;
    let mut index = 0;
    while let Some(argument) = args.get(index) {
        match argument.as_str() {
            "aux" => aux = true,
            "-e" | "-A" => {}
            "-f" | "-ef" => full = true,
            "-o" => {
                index += 1;
                let Some(specification) = args.get(index) else {
                    ewln(io.err, "ps: -o requires a column list");
                    return 2;
                };
                let parsed = specification
                    .split(',')
                    .map(|column| column.split('=').next().unwrap_or(column))
                    .collect::<Vec<_>>();
                if parsed.iter().any(|column| {
                    !matches!(
                        *column,
                        "pid"
                            | "ppid"
                            | "pgid"
                            | "sid"
                            | "stat"
                            | "comm"
                            | "cmd"
                            | "args"
                            | "user"
                            | "uid"
                    )
                }) {
                    ewln(io.err, "ps: unsupported output column");
                    return 2;
                }
                columns = Some(parsed);
            }
            "-p" | "--pid" => {
                index += 1;
                let Some(list) = args.get(index) else {
                    ewln(io.err, "ps: -p requires a PID list");
                    return 2;
                };
                let parsed = list
                    .split(',')
                    .map(str::parse::<u32>)
                    .collect::<Result<Vec<_>, _>>();
                let Ok(parsed) = parsed else {
                    ewln(io.err, "ps: invalid PID list");
                    return 2;
                };
                selected_pids = Some(parsed);
            }
            pid if pid.bytes().all(|byte| byte.is_ascii_digit()) => {
                let Ok(pid) = pid.parse::<u32>() else {
                    ewln(io.err, "ps: invalid PID");
                    return 2;
                };
                selected_pids.get_or_insert_with(Vec::new).push(pid);
            }
            option => {
                ewln(io.err, &format!("ps: unsupported option {option}"));
                return 2;
            }
        }
        index += 1;
    }
    let processes = context.system.process_snapshot();
    if let Some(columns) = &columns {
        wln(
            io.out,
            &columns
                .iter()
                .map(|column| ps_heading(column))
                .collect::<Vec<_>>()
                .join(" "),
        );
    } else if aux {
        wln(
            io.out,
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND",
        );
    } else if full {
        wln(
            io.out,
            "UID          PID    PPID  C STIME TTY          TIME CMD",
        );
    } else {
        wln(io.out, "    PID TTY          TIME CMD");
    }
    for process in &processes {
        if selected_pids
            .as_ref()
            .is_some_and(|pids| !pids.contains(&process.pid))
        {
            continue;
        }
        let state = match process.status {
            ProcessStatus::Running => "R",
            ProcessStatus::Stopped(_) => "T",
            ProcessStatus::Exited(_) => "Z",
        };
        if let Some(columns) = &columns {
            wln(
                io.out,
                &columns
                    .iter()
                    .map(|column| ps_value(column, process, state))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        } else if aux {
            let user = process
                .environment
                .get("USER")
                .map(String::as_str)
                .unwrap_or("user");
            wln(
                io.out,
                &format!(
                    "{user:<8}  {:>5}  0.0  0.0      0     0 ?        {state:<4} 00:00   0:00 {}",
                    process.pid, process.command,
                ),
            );
        } else if full {
            wln(
                io.out,
                &format!(
                    "{:<10} {:>5} {:>7}  0 00:00 ?        00:00:00 {}",
                    process.uid, process.pid, process.ppid, process.command
                ),
            );
        } else {
            wln(
                io.out,
                &format!("{:>7} ?        00:00:00 {}", process.pid, process.command),
            );
        }
    }
    0
}

fn ps_heading(column: &str) -> &'static str {
    match column {
        "pid" => "PID",
        "ppid" => "PPID",
        "pgid" => "PGID",
        "sid" => "SID",
        "stat" => "STAT",
        "comm" => "COMMAND",
        "cmd" | "args" => "CMD",
        "user" => "USER",
        "uid" => "UID",
        _ => unreachable!(),
    }
}

fn ps_value(column: &str, process: &crate::process::ProcessRecord, state: &str) -> String {
    match column {
        "pid" => process.pid.to_string(),
        "ppid" => process.ppid.to_string(),
        "pgid" => process.process_group.to_string(),
        "sid" => process.session_id.to_string(),
        "stat" => state.to_string(),
        "comm" => process
            .command
            .split_whitespace()
            .next()
            .unwrap_or("")
            .rsplit('/')
            .next()
            .unwrap_or("")
            .to_string(),
        "cmd" | "args" => process.command.clone(),
        "user" => process
            .environment
            .get("USER")
            .cloned()
            .unwrap_or_else(|| "user".into()),
        "uid" => process.uid.to_string(),
        _ => unreachable!(),
    }
}

/// Short process name: the basename of the first word of the process-table label.
fn process_name(command: &str) -> &str {
    command
        .split_whitespace()
        .next()
        .unwrap_or("")
        .rsplit('/')
        .next()
        .unwrap_or("")
}

/// Live processes other than the caller whose name (or full label with `full`) matches
/// `pattern`, as `pgrep` and `pkill` select them.
fn matching_processes(
    system: &mut dyn System,
    pattern: &str,
    full: bool,
    exact: bool,
) -> Result<Vec<crate::process::ProcessRecord>, String> {
    let regex = regex::Regex::new(pattern).map_err(|error| format!("invalid pattern: {error}"))?;
    let caller = system.pid();
    Ok(system
        .process_snapshot()
        .into_iter()
        .filter(|process| {
            process.pid != caller && !matches!(process.status, ProcessStatus::Exited(_))
        })
        .filter(|process| {
            let candidate = if full {
                process.command.as_str()
            } else {
                process_name(&process.command)
            };
            if exact {
                regex
                    .find(candidate)
                    .is_some_and(|found| found.as_str() == candidate)
            } else {
                regex.is_match(candidate)
            }
        })
        .collect())
}

/// Parse a signal name or number as `kill -SIGNAL` and `pkill -SIGNAL` accept it.
fn parse_signal_operand(value: &str) -> Option<crate::process::Signal> {
    crate::process::Signal::parse(value)
}

fn cmd_pgrep(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let mut full = false;
    let mut exact = false;
    let mut list_name = false;
    let mut list_full = false;
    let mut index = 0;
    while let Some(argument) = args.get(index) {
        match argument.as_str() {
            "-f" => full = true,
            "-x" => exact = true,
            "-l" => list_name = true,
            "-a" => list_full = true,
            value if value.starts_with('-') => {
                ewln(io.err, &format!("pgrep: unsupported option {value}"));
                return 2;
            }
            _ => break,
        }
        index += 1;
    }
    let Some(pattern) = args.get(index) else {
        ewln(io.err, "pgrep: pattern required");
        return 2;
    };
    if index + 1 != args.len() {
        ewln(io.err, "pgrep: too many patterns");
        return 2;
    }
    let processes = match matching_processes(context.system, pattern, full, exact) {
        Ok(processes) => processes,
        Err(error) => {
            ewln(io.err, &format!("pgrep: {error}"));
            return 2;
        }
    };
    for process in &processes {
        if list_full {
            wln(io.out, &format!("{} {}", process.pid, process.command));
        } else if list_name {
            wln(
                io.out,
                &format!("{} {}", process.pid, process_name(&process.command)),
            );
        } else {
            wln(io.out, &process.pid.to_string());
        }
    }
    i32::from(processes.is_empty())
}

fn cmd_pkill(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let mut signal = crate::process::Signal::Terminate;
    let mut full = false;
    let mut exact = false;
    let mut index = 0;
    while let Some(argument) = args.get(index) {
        match argument.as_str() {
            "-f" => full = true,
            "-x" => exact = true,
            "-s" | "--signal" => {
                index += 1;
                let Some(parsed) = args
                    .get(index)
                    .and_then(|value| parse_signal_operand(value))
                else {
                    ewln(io.err, "pkill: invalid or missing signal");
                    return 2;
                };
                signal = parsed;
            }
            value if value.starts_with('-') && value.len() > 1 => {
                let Some(parsed) = parse_signal_operand(&value[1..]) else {
                    ewln(io.err, &format!("pkill: unsupported option {value}"));
                    return 2;
                };
                signal = parsed;
            }
            _ => break,
        }
        index += 1;
    }
    let Some(pattern) = args.get(index) else {
        ewln(io.err, "pkill: pattern required");
        return 2;
    };
    if index + 1 != args.len() {
        ewln(io.err, "pkill: too many patterns");
        return 2;
    }
    let processes = match matching_processes(context.system, pattern, full, exact) {
        Ok(processes) => processes,
        Err(error) => {
            ewln(io.err, &format!("pkill: {error}"));
            return 2;
        }
    };
    let mut signalled = false;
    for process in &processes {
        // A process can exit between the snapshot and the signal; pkill skips it silently.
        signalled |= context
            .system
            .kill(crate::syscalls::SignalTarget::Process(process.pid), signal)
            .is_ok();
    }
    i32::from(!signalled)
}

fn cmd_killall(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let mut signal = crate::process::Signal::Terminate;
    let mut names = Vec::new();
    let mut index = 0;
    while let Some(argument) = args.get(index) {
        match argument.as_str() {
            "-s" | "--signal" => {
                index += 1;
                let Some(parsed) = args
                    .get(index)
                    .and_then(|value| parse_signal_operand(value))
                else {
                    ewln(io.err, "killall: invalid or missing signal");
                    return 2;
                };
                signal = parsed;
            }
            value if value.starts_with('-') && value.len() > 1 => {
                let Some(parsed) = parse_signal_operand(&value[1..]) else {
                    ewln(io.err, &format!("killall: unsupported option {value}"));
                    return 2;
                };
                signal = parsed;
            }
            name => names.push(name),
        }
        index += 1;
    }
    if names.is_empty() {
        ewln(io.err, "killall: process name required");
        return 2;
    }
    let caller = context.system.pid();
    let processes = context.system.process_snapshot();
    let mut status = 0;
    for name in names {
        let mut signalled = false;
        for process in processes.iter().filter(|process| {
            process.pid != caller
                && !matches!(process.status, ProcessStatus::Exited(_))
                && process_name(&process.command) == name
        }) {
            signalled |= context
                .system
                .kill(crate::syscalls::SignalTarget::Process(process.pid), signal)
                .is_ok();
        }
        if !signalled {
            ewln(io.err, &format!("{name}: no process found"));
            status = 1;
        }
    }
    status
}

fn cmd_lsof(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let mut selected_pid = None;
    let mut selected_name = None;
    let mut index = 0;
    while let Some(argument) = args.get(index) {
        match argument.as_str() {
            "-p" => {
                index += 1;
                selected_pid = args.get(index).and_then(|value| value.parse::<u32>().ok());
                if selected_pid.is_none() {
                    ewln(io.err, "lsof: -p requires a PID");
                    return 2;
                }
            }
            value if value.starts_with('-') => {
                ewln(io.err, &format!("lsof: unsupported option {value}"));
                return 2;
            }
            value => selected_name = Some(value.to_string()),
        }
        index += 1;
    }
    wln(io.out, "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME");
    let mut found = false;
    for process in context.system.process_snapshot() {
        if selected_pid.is_some_and(|pid| process.pid != pid) {
            continue;
        }
        let command = process
            .command
            .split_whitespace()
            .next()
            .unwrap_or("?")
            .rsplit('/')
            .next()
            .unwrap_or("?");
        for (fd, name) in &process.descriptors {
            if selected_name
                .as_ref()
                .is_some_and(|selected| selected != name)
            {
                continue;
            }
            found = true;
            wln(
                io.out,
                &format!("{command} {} root {fd}u REG 0,0 0 0 {name}", process.pid),
            );
        }
    }
    i32::from(!found && (selected_pid.is_some() || selected_name.is_some()))
}

fn cmd_sockets(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    if args.iter().any(|argument| !argument.starts_with('-')) {
        ewln(io.err, "socket listing: operands are not supported");
        return 2;
    }
    if args
        .iter()
        .flat_map(|argument| argument.trim_start_matches('-').chars())
        .any(|flag| !matches!(flag, 'l' | 'n' | 't' | 'u' | 'a' | 'p'))
    {
        ewln(io.err, "socket listing: unsupported option");
        return 2;
    }
    wln(
        io.out,
        "State  Recv-Q Send-Q Local Address:Port Peer Address:Port Process",
    );
    let mut listeners = context.system.listener_snapshot();
    listeners.sort();
    for address in listeners {
        wln(
            io.out,
            &format!("LISTEN 0      128    {address:<18} 0.0.0.0:*         -"),
        );
    }
    0
}

fn human_size(bytes: u64) -> String {
    for (unit, size) in [("G", 1024u64.pow(3)), ("M", 1024u64.pow(2)), ("K", 1024)] {
        if bytes >= size {
            return format!("{}{}", bytes / size, unit);
        }
    }
    format!("{bytes}B")
}
