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
    reg(m, &["source", "."], Trust::Real, cmd_source);
    reg(m, &["eval"], Trust::Real, cmd_eval);
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
    reg(
        m,
        &[
            "trap", "disown", "umask", "ulimit", "hash", "complete", "shopt", "bind", "history",
            "exec",
        ],
        Trust::NoOp,
        cmd_unsupported,
    );
    reg(
        m,
        &["kill", "killall", "pkill"],
        Trust::NoOp,
        cmd_unsupported,
    );
    reg(m, &["type", "which"], Trust::Real, cmd_which);
    reg(m, &["command"], Trust::Real, cmd_command);
    reg(m, &["alias", "unalias"], Trust::NoOp, cmd_unsupported);
    reg(m, &["getopts"], Trust::Real, cmd_getopts);
    reg(m, &["let"], Trust::Real, cmd_let);
    reg(m, &["mapfile", "readarray"], Trust::NoOp, cmd_unsupported);
    reg(m, &["pushd", "popd", "dirs"], Trust::NoOp, cmd_unsupported);
}

fn cmd_unsupported(_interp: &mut CommandContext<'_>, _args: &[String], io: &mut Io) -> i32 {
    ewln(io.err, "shellsim: builtin is not supported");
    2
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

fn cmd_pwd(interp: &mut CommandContext<'_>, _args: &[String], io: &mut Io) -> i32 {
    wln(io.out, &interp.cwd);
    0
}

fn cmd_cd(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let target = match args.first().map(|s| s.as_str()) {
        None | Some("~") => interp.get_var("HOME").unwrap_or_else(|| "/".into()),
        Some("-") => interp
            .get_var("OLDPWD")
            .unwrap_or_else(|| interp.cwd.clone()),
        Some(p) => p.to_string(),
    };
    let abs = crate::vfs::resolve_against(&interp.cwd, &target);
    if interp.vfs.is_dir("/", &abs) {
        let old = interp.cwd.clone();
        interp.set_var("OLDPWD", old);
        interp.cwd = interp.vfs.realpath(&abs, true).unwrap_or(abs);
        let cwd = interp.cwd.clone();
        interp.set_var("PWD", cwd);
        0
    } else {
        ewln(io.err, &format!("cd: {target}: No such file or directory"));
        1
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

fn cmd_eval(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let src = args.join(" ");
    interp.run_script_into(&src, io.out, io.err)
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
