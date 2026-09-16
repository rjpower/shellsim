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
    io.out.extend_from_slice(output.as_bytes());
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
    let aux = args == ["aux"];
    let flags_valid = args.iter().all(|argument| {
        argument.starts_with('-') && argument[1..].chars().all(|flag| matches!(flag, 'e' | 'f'))
    });
    if !args.is_empty() && !aux && !flags_valid {
        ewln(
            io.err,
            "ps: only ps, ps -e/-f/-ef, and ps aux are supported",
        );
        return 2;
    }
    let full = !aux && args.iter().any(|argument| argument.contains('f'));
    let pid = interp.pid;
    let cwd = interp.cwd.clone();
    let environment = interp.child_env().into_iter().collect();
    interp.processes.update_current(pid, &cwd, environment);
    if aux {
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
        let state = match process.status {
            ProcessStatus::Running => "R",
            ProcessStatus::Stopped(_) => "T",
            ProcessStatus::Exited(_) => "Z",
        };
        if aux {
            wln(
                io.out,
                &format!(
                    "root      {:>5}  0.0  0.0      0     0 ?        {state:<4} 00:00   0:00 {}",
                    process.pid, process.command
                ),
            );
        } else if full {
            wln(
                io.out,
                &format!(
                    "root       {:>5} {:>7}  0 00:00 ?        00:00:00 {}",
                    process.pid, process.ppid, process.command
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

fn human_size(bytes: u64) -> String {
    for (unit, size) in [("G", 1024u64.pow(3)), ("M", 1024u64.pow(2)), ("K", 1024)] {
        if bytes >= size {
            return format!("{}{}", bytes / size, unit);
        }
    }
    format!("{bytes}B")
}
