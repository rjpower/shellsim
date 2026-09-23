//! Small system-introspection commands. Their answers describe the simulated environment rather
//! than the host running shellsim.

use std::collections::{BTreeMap, HashMap};

use crate::commands::util::{ewln, wln};
use crate::commands::{
    reg, reg_buffered_resumable, ChildCommand, CommandContext, CommandPoll, CommandSpec, Io, Trust,
};
use crate::interp::Interp;
use crate::process::ProcessStatus;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    reg_buffered_resumable(m, &["env"], Trust::Real, cmd_env, start_env);
    reg(m, &["printenv"], Trust::Real, cmd_printenv);
    reg(m, &["envsubst"], Trust::Partial, cmd_envsubst);
    reg(m, &["uname"], Trust::Real, cmd_uname);
    reg(m, &["arch"], Trust::Real, cmd_arch);
    reg(m, &["hostname"], Trust::Real, cmd_hostname);
    reg(m, &["whoami", "logname"], Trust::Real, cmd_whoami);
    reg(m, &["id"], Trust::Real, cmd_id);
    reg(m, &["groups"], Trust::Real, cmd_groups);
    reg(m, &["nproc"], Trust::Real, cmd_nproc);
    reg(m, &["getconf"], Trust::Partial, cmd_getconf);
    reg(m, &["df"], Trust::Real, cmd_df);
    reg(m, &["free"], Trust::Real, cmd_free);
    reg(m, &["ps"], Trust::Partial, cmd_ps);
    reg(m, &["pgrep"], Trust::Partial, cmd_pgrep);
    reg(m, &["lsof"], Trust::Partial, cmd_lsof);
    reg(m, &["ss", "netstat"], Trust::Partial, cmd_sockets);
    reg(m, &["nohup"], Trust::Partial, cmd_nohup);
}

fn cmd_env(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let action = match parse_env_action(interp, args) {
        Ok(action) => action,
        Err(error) => {
            ewln(io.err, &format!("env: {error}"));
            return 125;
        }
    };
    let saved_vars = interp.vars.clone();
    let saved_arrays = interp.arrays.clone();
    let saved_exported = interp.exported.clone();
    let saved_cwd = interp.cwd.clone();
    interp.vars = action.environment.clone().into_iter().collect();
    interp.arrays.clear();
    interp.exported = action.environment.keys().cloned().collect();
    if let Some(cwd) = &action.cwd {
        interp.cwd = cwd.clone();
    }

    let status = if action.argv.is_empty() {
        for (name, value) in &action.environment {
            wln(io.out, &format!("{name}={value}"));
        }
        0
    } else {
        crate::commands::run(interp, &action.argv, io.stdin.clone(), io.out, io.err)
    };
    interp.vars = saved_vars;
    interp.arrays = saved_arrays;
    interp.exported = saved_exported;
    interp.cwd = saved_cwd;
    status
}

fn start_env(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let action = match parse_env_action(interp, args) {
        Ok(action) => action,
        Err(error) => {
            ewln(io.err, &format!("env: {error}"));
            return CommandPoll::Ready(125);
        }
    };
    if action.argv.is_empty() {
        for (name, value) in action.environment {
            wln(io.out, &format!("{name}={value}"));
        }
        return CommandPoll::Ready(0);
    }
    crate::commands::start_child_sequence(
        interp,
        vec![ChildCommand {
            argv: action.argv,
            stdin: std::mem::take(&mut io.stdin),
            cwd: action.cwd,
            environment: Some(action.environment),
        }],
        true,
    )
}

struct EnvAction {
    environment: BTreeMap<String, String>,
    cwd: Option<String>,
    argv: Vec<String>,
}

fn parse_env_action(interp: &Interp, args: &[String]) -> Result<EnvAction, String> {
    let mut environment = interp.child_env().into_iter().collect::<BTreeMap<_, _>>();
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
                let resolved = crate::vfs::resolve_against(&interp.cwd, directory);
                if !interp.vfs.is_dir("/", &resolved) {
                    return Err(format!("cannot change directory to '{directory}'"));
                }
                cwd = Some(interp.vfs.realpath(&resolved, true).unwrap_or(resolved));
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

fn cmd_printenv(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let env = interp.child_env();
    if args.is_empty() {
        let mut entries = env.into_iter().collect::<Vec<_>>();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, value) in entries {
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

fn cmd_envsubst(interp: &mut CommandContext<'_>, _args: &[String], io: &mut Io) -> i32 {
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
            output.push_str(&interp.get_var(&name).unwrap_or_default());
        }
    }
    io.print(&output);
    0
}

fn cmd_uname(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
                'n' => "sandbox",
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

fn cmd_arch(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if !args.is_empty() {
        ewln(io.err, "arch: unimplemented option or operand");
        return 2;
    }
    wln(io.out, "x86_64");
    0
}

fn cmd_hostname(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.iter().any(|arg| arg.starts_with('-') && arg != "-f") {
        ewln(io.err, "hostname: unimplemented option");
        return 2;
    }
    if let Some(name) = args.iter().find(|arg| !arg.starts_with('-')) {
        interp.set_var("HOSTNAME", name);
        interp.export("HOSTNAME");
    } else {
        wln(
            io.out,
            &interp
                .get_var("HOSTNAME")
                .unwrap_or_else(|| "sandbox".to_string()),
        );
    }
    0
}

fn cmd_whoami(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if !args.is_empty() {
        ewln(io.err, "whoami: unimplemented option or operand");
        return 2;
    }
    wln(io.out, if interp.uid == 0 { "root" } else { "user" });
    0
}

fn cmd_id(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
    let name = if interp.uid == 0 { "root" } else { "user" };
    if user || group {
        let value = if name_output {
            name.to_string()
        } else {
            interp.uid.to_string()
        };
        wln(io.out, &value);
    } else {
        wln(
            io.out,
            &format!(
                "uid={}({name}) gid={}({name}) groups={}({name})",
                interp.uid, interp.uid, interp.uid
            ),
        );
    }
    0
}

fn cmd_groups(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if !args.is_empty() {
        ewln(io.err, "groups: unimplemented user operand");
        return 2;
    }
    wln(io.out, if interp.uid == 0 { "root" } else { "user" });
    0
}

fn cmd_nproc(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if !args.is_empty() {
        ewln(io.err, "nproc: unimplemented option");
        return 2;
    }
    wln(io.out, "1");
    0
}

fn cmd_getconf(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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

fn cmd_df(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args
        .iter()
        .any(|arg| arg.starts_with('-') && arg != "-h" && arg != "-k" && arg != "-P")
    {
        ewln(io.err, "df: unimplemented option");
        return 2;
    }
    let human = args.iter().any(|arg| arg.contains('h'));
    for path in args.iter().filter(|argument| !argument.starts_with('-')) {
        if interp.fs_metadata(&interp.cwd, path, true).is_err() {
            ewln(io.err, &format!("df: {path}: No such file or directory"));
            return 1;
        }
    }
    let limit = interp.resources.limits().disk;
    let used = interp.vfs.disk_used();
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

fn cmd_free(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.iter().any(|arg| arg != "-b") {
        ewln(io.err, "free: unimplemented option or operand");
        return 2;
    }
    let bytes = args.iter().any(|arg| arg == "-b");
    let divisor = if bytes { 1 } else { 1024 };
    let limit = interp.resources.limits().memory;
    let used = interp.resources.memory_mark();
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

fn cmd_ps(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
    let pid = interp.pid;
    let cwd = interp.cwd.clone();
    let environment = interp.child_env().into_iter().collect();
    interp.processes.update_current(pid, &cwd, environment);
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
    for process in interp.processes.iter() {
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

fn cmd_pgrep(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
    let regex = match regex::Regex::new(pattern) {
        Ok(regex) => regex,
        Err(error) => {
            ewln(io.err, &format!("pgrep: invalid pattern: {error}"));
            return 2;
        }
    };
    let mut found = false;
    for process in interp
        .processes
        .iter()
        .filter(|process| !matches!(process.status, ProcessStatus::Exited(_)))
    {
        let short = process
            .command
            .split_whitespace()
            .next()
            .unwrap_or("")
            .rsplit('/')
            .next()
            .unwrap_or("");
        let candidate = if full {
            process.command.as_str()
        } else {
            short
        };
        let matched = if exact {
            regex
                .find(candidate)
                .is_some_and(|found| found.as_str() == candidate)
        } else {
            regex.is_match(candidate)
        };
        if matched {
            found = true;
            if list_full {
                wln(io.out, &format!("{} {}", process.pid, process.command));
            } else if list_name {
                wln(io.out, &format!("{} {short}", process.pid));
            } else {
                wln(io.out, &process.pid.to_string());
            }
        }
    }
    i32::from(!found)
}

fn cmd_lsof(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
    for process in interp.processes.iter() {
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

fn cmd_sockets(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
    let mut listeners = interp
        .net
        .listening
        .iter()
        .filter(|(_, active)| **active)
        .map(|(address, _)| address)
        .collect::<Vec<_>>();
    listeners.sort();
    for address in listeners {
        wln(
            io.out,
            &format!("LISTEN 0      128    {address:<18} 0.0.0.0:*         -"),
        );
    }
    0
}

fn cmd_nohup(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let argv = if args.first().map(String::as_str) == Some("--") {
        &args[1..]
    } else {
        args
    };
    if argv.is_empty() {
        ewln(io.err, "nohup: missing operand");
        return 125;
    }
    // Captured shellsim streams are not terminals, so GNU nohup's nohup.out redirection does not
    // apply. Signal disposition is process-local; the delegated command runs synchronously here.
    crate::commands::run(interp, argv, std::mem::take(&mut io.stdin), io.out, io.err)
}

fn human_size(bytes: u64) -> String {
    for (unit, size) in [("G", 1024u64.pow(3)), ("M", 1024u64.pow(2)), ("K", 1024)] {
        if bytes >= size {
            return format!("{}{}", bytes / size, unit);
        }
    }
    format!("{bytes}B")
}
