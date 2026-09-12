//! Small system-introspection commands. Their answers describe the simulated environment rather
//! than the host running shellsim.

use std::collections::HashMap;

use crate::commands::util::{ewln, wln};
use crate::commands::{reg, CommandContext, CommandSpec, Io, Trust};
use crate::process::ProcessStatus;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    reg(m, &["env"], Trust::Real, cmd_env);
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
    let saved_vars = interp.vars.clone();
    let saved_arrays = interp.arrays.clone();
    let saved_exported = interp.exported.clone();
    let saved_cwd = interp.cwd.clone();
    let mut clear = false;
    let mut unset = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-i" | "--ignore-environment" => clear = true,
            "-u" | "--unset" => {
                i += 1;
                if let Some(name) = args.get(i) {
                    unset.push(name.clone());
                }
            }
            arg if arg.starts_with('-') => {}
            _ => break,
        }
        i += 1;
    }
    if clear {
        interp.vars.clear();
        interp.exported.clear();
    }
    for name in unset {
        interp.vars.remove(&name);
        interp.exported.remove(&name);
    }
    while let Some(arg) = args.get(i) {
        let Some((name, value)) = arg.split_once('=') else {
            break;
        };
        interp.set_var(name, value);
        interp.export(name);
        i += 1;
    }

    let status = if i == args.len() {
        let mut entries = interp.child_env().into_iter().collect::<Vec<_>>();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, value) in entries {
            wln(io.out, &format!("{name}={value}"));
        }
        0
    } else {
        let argv = args[i..].to_vec();
        crate::commands::run(interp, &argv, io.stdin.clone(), io.out, io.err)
    };
    interp.vars = saved_vars;
    interp.arrays = saved_arrays;
    interp.exported = saved_exported;
    interp.cwd = saved_cwd;
    status
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
    let all = args.iter().any(|arg| arg == "-a" || arg == "--all");
    if all {
        wln(
            io.out,
            "Linux sandbox 6.6.0-shellsim #1 SMP x86_64 GNU/Linux",
        );
    } else if args.iter().any(|arg| arg.contains('m')) {
        wln(io.out, "x86_64");
    } else if args.iter().any(|arg| arg.contains('n')) {
        wln(io.out, "sandbox");
    } else if args.iter().any(|arg| arg.contains('r')) {
        wln(io.out, "6.6.0-shellsim");
    } else {
        wln(io.out, "Linux");
    }
    0
}

fn cmd_arch(_interp: &mut CommandContext<'_>, _args: &[String], io: &mut Io) -> i32 {
    wln(io.out, "x86_64");
    0
}

fn cmd_hostname(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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

fn cmd_whoami(interp: &mut CommandContext<'_>, _args: &[String], io: &mut Io) -> i32 {
    wln(io.out, if interp.uid == 0 { "root" } else { "user" });
    0
}

fn cmd_id(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let name = if interp.uid == 0 { "root" } else { "user" };
    if args.iter().any(|arg| arg == "-u" || arg == "-g") {
        wln(io.out, &interp.uid.to_string());
    } else if args
        .iter()
        .any(|arg| arg == "-un" || arg == "-nu" || arg == "-gn" || arg == "-ng")
    {
        wln(io.out, name);
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

fn cmd_groups(interp: &mut CommandContext<'_>, _args: &[String], io: &mut Io) -> i32 {
    wln(io.out, if interp.uid == 0 { "root" } else { "user" });
    0
}

fn cmd_nproc(_interp: &mut CommandContext<'_>, _args: &[String], io: &mut Io) -> i32 {
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
    let human = args.iter().any(|arg| arg.contains('h'));
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
