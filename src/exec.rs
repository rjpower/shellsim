//! The shell executor: walks the AST against the interpreter state.

use crate::expand::{expand_word, expand_words};
use crate::interp::Interp;
use crate::shell::{Node, RedirOp, Redirect};
use crate::{descriptors::IoPoll, vfs::resolve_against};

/// Execute a node with finite input and capture its terminal output.
///
/// The byte-buffer API is the harness boundary. Internally, execution installs those streams as
/// process descriptors so redirection, duplication, and child inheritance use one model.
pub fn exec(
    interp: &mut Interp,
    node: &Node,
    stdin: Vec<u8>,
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
) -> i32 {
    let saved = match interp.process.fds.fork(&mut interp.descriptors) {
        Ok(saved) => saved,
        Err(error) => {
            err.extend_from_slice(
                format!("shellsim: unable to save descriptors: {error:?}\n").as_bytes(),
            );
            return 125;
        }
    };
    let input = match interp.descriptors.open_input(stdin) {
        Ok(description) => description,
        Err(error) => return restore_failed_setup(interp, saved, err, error),
    };
    if let Err(error) = interp.install_new_description(0, input) {
        return restore_failed_setup(interp, saved, err, error);
    }
    let stdout = match interp.descriptors.open_capture() {
        Ok(description) => description,
        Err(error) => return restore_failed_setup(interp, saved, err, error),
    };
    if let Err(error) = interp.install_new_description(1, stdout) {
        return restore_failed_setup(interp, saved, err, error);
    }
    let stderr = match interp.descriptors.open_capture() {
        Ok(description) => description,
        Err(error) => return restore_failed_setup(interp, saved, err, error),
    };
    if let Err(error) = interp.install_new_description(2, stderr) {
        return restore_failed_setup(interp, saved, err, error);
    }

    let status = exec_node(interp, node);
    if let Ok(bytes) = interp.descriptors.capture(stdout) {
        out.extend_from_slice(bytes);
    }
    if let Ok(bytes) = interp.descriptors.capture(stderr) {
        err.extend_from_slice(bytes);
    }
    restore_fds(interp, saved);
    status
}

fn restore_failed_setup(
    interp: &mut Interp,
    saved: crate::descriptors::FdTable,
    err: &mut Vec<u8>,
    error: crate::descriptors::DescriptorError,
) -> i32 {
    restore_fds(interp, saved);
    err.extend_from_slice(
        format!("shellsim: unable to install descriptors: {error:?}\n").as_bytes(),
    );
    125
}

fn restore_fds(interp: &mut Interp, saved: crate::descriptors::FdTable) {
    interp.process.fds.close_all(&mut interp.descriptors);
    interp.process.fds = saved;
    interp.refresh_descriptor_snapshot(interp.process.pid);
}

fn exec_node(interp: &mut Interp, node: &Node) -> i32 {
    if !interp.resources.charge_cpu(10) {
        return interp
            .resources
            .stop_reason()
            .map_or(137, |r| r.exit_status());
    }
    if interp.exiting.is_some()
        || interp.returning.is_some()
        || interp.loop_break > 0
        || interp.loop_continue > 0
        || interp.deadline_interrupt.is_some()
    {
        return if interp.deadline_interrupt.is_some() {
            124
        } else {
            interp.last_status
        };
    }
    let status = match node {
        Node::Empty => 0,
        Node::Command { .. } => exec_command(interp, node),
        Node::Pipeline(stages) => exec_pipeline(interp, stages),
        Node::And(a, b) => {
            let sa = exec_cond(interp, a);
            if sa == 0 && interp.exiting.is_none() {
                exec_node(interp, b)
            } else {
                sa
            }
        }
        Node::Or(a, b) => {
            let sa = exec_cond(interp, a);
            if sa != 0 && interp.exiting.is_none() {
                exec_node(interp, b)
            } else {
                sa
            }
        }
        Node::Not(a) => {
            let sa = exec_cond(interp, a);
            if sa == 0 {
                1
            } else {
                0
            }
        }
        Node::Seq(nodes) => {
            let mut s = 0;
            for n in nodes {
                if interp.resources.is_stopped() {
                    break;
                }
                s = exec_node(interp, n);
                if interp.exiting.is_some() || interp.returning.is_some() {
                    break;
                }
                if interp.deadline_interrupt.is_some() {
                    s = 124;
                    break;
                }
                if interp.loop_break > 0 || interp.loop_continue > 0 {
                    break;
                }
                if s != 0 && interp.opt_errexit && interp.cond_depth == 0 {
                    interp.exiting = Some(s);
                    break;
                }
            }
            s
        }
        Node::Background(a) => {
            let cmd = describe(a);
            let Some((s, pid)) = exec_child(
                interp,
                a,
                ChildExecution {
                    command: &cmd,
                    new_shell: false,
                    retain: true,
                },
            ) else {
                write_diagnostic(interp, "shellsim: unable to create background process\n");
                return 125;
            };
            let Some(id) = interp.new_job(pid, cmd) else {
                interp.processes.reap(pid);
                let _ = interp.scheduler.reap(pid);
                write_diagnostic(interp, "shellsim: job table limit exceeded\n");
                return 125;
            };
            if let Some(job) = interp.jobs.iter_mut().find(|job| job.id == id) {
                job.done = true;
                job.status = s;
            }
            interp.set_var("!", pid.to_string());
            0
        }
        Node::Subshell(a) => exec_child(
            interp,
            a,
            ChildExecution {
                command: "(subshell)",
                new_shell: false,
                retain: false,
            },
        )
        .map_or(125, |(status, _)| status),
        Node::Group(a) => exec_node(interp, a),
        Node::Redirected(inner, redirs) => {
            with_redirects(interp, redirs, |interp| exec_node(interp, inner))
        }
        Node::If {
            cond,
            then,
            elifs,
            els,
        } => {
            if exec_cond(interp, cond) == 0 {
                exec_node(interp, then)
            } else {
                for (c, b) in elifs {
                    if exec_cond(interp, c) == 0 {
                        return exec_node(interp, b);
                    }
                }
                if let Some(e) = els {
                    exec_node(interp, e)
                } else {
                    0
                }
            }
        }
        Node::While { cond, body, until } => {
            let mut s = 0;
            let mut guard = 0;
            loop {
                if interp.resources.is_stopped() {
                    break;
                }
                guard += 1;
                // bound runaway poll loops (e.g. `until curl ...; do sleep; done` against a
                // service we don't simulate). Real task loops never need this many iterations.
                if guard > 5_000 {
                    break;
                }
                let c = exec_cond(interp, cond);
                let go = if *until { c != 0 } else { c == 0 };
                if !go || interp.exiting.is_some() {
                    break;
                }
                s = exec_node(interp, body);
                if interp.loop_break > 0 {
                    interp.loop_break -= 1;
                    break;
                }
                if interp.loop_continue > 0 {
                    interp.loop_continue -= 1;
                    continue;
                }
                if interp.exiting.is_some() || interp.returning.is_some() {
                    break;
                }
            }
            s
        }
        Node::For { var, words, body } => {
            let items = expand_words(interp, words);
            let mut s = 0;
            for item in items {
                if interp.resources.is_stopped() {
                    break;
                }
                interp.set_var(var, item);
                s = exec_node(interp, body);
                if interp.loop_break > 0 {
                    interp.loop_break -= 1;
                    break;
                }
                if interp.loop_continue > 0 {
                    interp.loop_continue -= 1;
                    continue;
                }
                if interp.exiting.is_some() || interp.returning.is_some() {
                    break;
                }
            }
            s
        }
        Node::CFor {
            init,
            cond,
            update,
            body,
        } => {
            let _ = crate::expand::eval_arith(interp, init);
            let mut status = 0;
            let mut guard = 0usize;
            while cond.is_empty() || crate::expand::eval_arith(interp, cond) != 0 {
                if interp.resources.is_stopped() || guard >= 5_000 {
                    break;
                }
                guard += 1;
                status = exec_node(interp, body);
                if interp.loop_break > 0 {
                    interp.loop_break -= 1;
                    break;
                }
                if interp.loop_continue > 0 {
                    interp.loop_continue -= 1;
                }
                if interp.exiting.is_some() || interp.returning.is_some() {
                    break;
                }
                let _ = crate::expand::eval_arith(interp, update);
            }
            status
        }
        Node::Case { word, arms } => {
            let subject = expand_word(interp, word, false).join(" ");
            for (pats, body) in arms {
                for pat in pats {
                    let p = expand_word(interp, pat, false).join(" ");
                    if case_match(&p, &subject) {
                        return exec_node(interp, body);
                    }
                }
            }
            0
        }
        Node::FuncDef { name, body } => {
            interp.funcs.insert(name.clone(), (**body).clone());
            0
        }
        Node::Arithmetic(expression) => {
            if crate::expand::eval_arith(interp, expression) == 0 {
                1
            } else {
                0
            }
        }
    };
    interp.last_status = status;
    status
}

/// Execute as a condition: errexit is suppressed inside.
fn exec_cond(interp: &mut Interp, node: &Node) -> i32 {
    interp.cond_depth += 1;
    let s = exec_node(interp, node);
    interp.cond_depth -= 1;
    s
}

fn describe(node: &Node) -> String {
    match node {
        Node::Command { words, .. } => words.join(" "),
        _ => "<job>".to_string(),
    }
}

fn with_redirects(
    interp: &mut Interp,
    redirects: &[Redirect],
    run: impl FnOnce(&mut Interp) -> i32,
) -> i32 {
    let memory_mark = interp.resources.memory_mark();
    let snapshot_bytes = interp.vfs.disk_used().saturating_add(4 * 1024);
    if !interp.resources.reserve_memory(snapshot_bytes) {
        write_diagnostic(
            interp,
            "shellsim: redirection: memory limit exceeded while preparing redirections\n",
        );
        return 1;
    }
    let vfs_before = interp.vfs.clone();
    let saved = match interp.process.fds.fork(&mut interp.descriptors) {
        Ok(saved) => saved,
        Err(error) => {
            write_diagnostic(interp, &format!("shellsim: redirection: {error:?}\n"));
            interp.resources.restore_memory(memory_mark);
            return 1;
        }
    };
    if let Err(error) = apply_redirects(interp, redirects) {
        restore_fds(interp, saved);
        interp.vfs = vfs_before;
        interp.resources.restore_memory(memory_mark);
        write_diagnostic(interp, &format!("shellsim: redirection: {error}\n"));
        return 1;
    }
    let status = run(interp);
    restore_fds(interp, saved);
    interp.resources.restore_memory(memory_mark);
    status
}

/// Apply redirections from left to right. `dup` retains the open description selected at that
/// point, so `2>&1 >file` and `>file 2>&1` have distinct, Bash-compatible destinations.
fn apply_redirects(interp: &mut Interp, redirects: &[Redirect]) -> Result<(), String> {
    for redirect in redirects {
        match redirect.op {
            RedirOp::Read => {
                let path = redirect_path(interp, &redirect.target)?;
                if let Some(source) = device_fd(&path) {
                    interp
                        .process
                        .fds
                        .duplicate(source, redirect.fd, &mut interp.descriptors)
                        .map_err(|error| format!("{path}: {error:?}"))?;
                } else if path == "/dev/null" {
                    let description = interp
                        .descriptors
                        .open_null()
                        .map_err(|error| format!("{path}: {error:?}"))?;
                    interp
                        .install_new_description(redirect.fd, description)
                        .map_err(|error| format!("{path}: {error:?}"))?;
                } else {
                    interp
                        .fs_metadata("/", &path, true)
                        .map_err(|error| error.to_string())?;
                    let description = interp
                        .descriptors
                        .open_file(path.clone(), 0, true, false, false)
                        .map_err(|error| format!("{path}: {error:?}"))?;
                    interp
                        .install_new_description(redirect.fd, description)
                        .map_err(|error| format!("{path}: {error:?}"))?;
                }
            }
            RedirOp::Heredoc | RedirOp::HeredocRaw | RedirOp::HereString => {
                let mut bytes = match redirect.op {
                    RedirOp::Heredoc => expand_heredoc(interp, &redirect.target).into_bytes(),
                    RedirOp::HeredocRaw => redirect.target.clone().into_bytes(),
                    RedirOp::HereString => {
                        let mut bytes = expand_word(interp, &redirect.target, false)
                            .join(" ")
                            .into_bytes();
                        bytes.push(b'\n');
                        bytes
                    }
                    _ => unreachable!(),
                };
                let description = interp
                    .descriptors
                    .open_input(std::mem::take(&mut bytes))
                    .map_err(|error| format!("here document: {error:?}"))?;
                interp
                    .install_new_description(redirect.fd, description)
                    .map_err(|error| format!("here document: {error:?}"))?;
            }
            RedirOp::Write | RedirOp::Append => {
                let path = redirect_path(interp, &redirect.target)?;
                if let Some(source) = device_fd(&path) {
                    interp
                        .process
                        .fds
                        .duplicate(source, redirect.fd, &mut interp.descriptors)
                        .map_err(|error| format!("{path}: {error:?}"))?;
                    continue;
                }
                if path == "/dev/null" {
                    let description = interp
                        .descriptors
                        .open_null()
                        .map_err(|error| format!("{path}: {error:?}"))?;
                    interp
                        .install_new_description(redirect.fd, description)
                        .map_err(|error| format!("{path}: {error:?}"))?;
                    continue;
                }
                let created_by_open = !interp.vfs.lexists("/", &path);
                interp.sync_vfs_time();
                let cursor = if redirect.op == RedirOp::Append {
                    interp
                        .vfs
                        .append("/", &path, &[], 0o644)
                        .map_err(|error| error.to_string())?;
                    interp
                        .vfs
                        .file_len("/", &path)
                        .map_err(|error| error.to_string())? as u64
                } else {
                    interp
                        .vfs
                        .write("/", &path, &[], 0o644)
                        .map_err(|error| error.to_string())?;
                    0
                };
                let description = interp
                    .descriptors
                    .open_file(path.clone(), cursor, false, true, created_by_open)
                    .map_err(|error| format!("{path}: {error:?}"))?;
                interp
                    .install_new_description(redirect.fd, description)
                    .map_err(|error| format!("{path}: {error:?}"))?;
            }
            RedirOp::DupOut => {
                let source = redirect
                    .target
                    .trim_start_matches('&')
                    .parse::<i32>()
                    .map_err(|_| format!("bad file descriptor: {}", redirect.target))?;
                interp
                    .process
                    .fds
                    .duplicate(source, redirect.fd, &mut interp.descriptors)
                    .map_err(|error| format!("{}: {error:?}", redirect.target))?;
            }
            RedirOp::Close => {
                interp
                    .process
                    .fds
                    .close(redirect.fd, &mut interp.descriptors)
                    .map_err(|error| format!("{}: {error:?}", redirect.fd))?;
            }
        }
        interp.refresh_descriptor_snapshot(interp.process.pid);
    }
    Ok(())
}

fn redirect_path(interp: &mut Interp, word: &str) -> Result<String, String> {
    let fields = expand_word(interp, word, true);
    if fields.len() != 1 {
        return Err(format!("{word}: ambiguous redirect"));
    }
    Ok(resolve_against(&interp.cwd, &fields[0]))
}

fn device_fd(path: &str) -> Option<i32> {
    match path {
        "/dev/stdin" => Some(0),
        "/dev/stdout" => Some(1),
        "/dev/stderr" => Some(2),
        _ => path.strip_prefix("/dev/fd/")?.parse().ok(),
    }
}

fn read_all_fd(interp: &mut Interp, fd: i32) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    loop {
        match interp.read_fd(fd, 64 * 1024)? {
            IoPoll::Ready(bytes) if bytes.is_empty() => return Ok(output),
            IoPoll::Ready(bytes) => {
                if output.len().saturating_add(bytes.len()) > crate::descriptors::MAX_CAPTURE_BYTES
                {
                    return Err("input exceeds descriptor capture limit".to_string());
                }
                output.extend_from_slice(&bytes);
            }
            IoPoll::Blocked(_) => return Err("descriptor read would block".to_string()),
        }
    }
}

fn write_all_fd(interp: &mut Interp, fd: i32, mut bytes: &[u8]) -> Result<(), String> {
    while !bytes.is_empty() {
        match interp.write_fd(fd, bytes)? {
            IoPoll::Ready(0) | IoPoll::Blocked(_) => {
                return Err("descriptor write would block".to_string());
            }
            IoPoll::Ready(written) => bytes = &bytes[written..],
        }
    }
    Ok(())
}

fn write_diagnostic(interp: &mut Interp, message: &str) {
    let _ = write_all_fd(interp, 2, message.as_bytes());
}

fn exec_command(interp: &mut Interp, node: &Node) -> i32 {
    let (assigns, words, redirects) = match node {
        Node::Command {
            assigns,
            words,
            redirects,
        } => (assigns, words, redirects),
        _ => unreachable!(),
    };
    if !redirects.is_empty() {
        return with_redirects(interp, redirects, |interp| {
            exec_command_parts(interp, assigns, words)
        });
    }
    exec_command_parts(interp, assigns, words)
}

fn exec_command_parts(interp: &mut Interp, assigns: &[(String, String)], words: &[String]) -> i32 {
    // No command words → assignments are persistent (including array assignments).
    // Words that look like array-assignment literals (`name=( … )`, `name[i]=v`) are kept
    // verbatim — they must not be word-split/globbed — so `declare`/`local` can parse them.
    let argv = expand_argv(interp, words);
    if argv.is_empty() {
        for (k, v) in assigns {
            apply_assignment(interp, k, v);
        }
        return 0;
    }

    // With command words, the leading assignments are temporary (scalar only here; array
    // command-prefixes don't occur in our corpus). Expand scalar values for set/restore.
    let expanded_assigns: Vec<(String, String)> = assigns
        .iter()
        .map(|(k, v)| (k.clone(), expand_word(interp, v, false).join(" ")))
        .collect();

    // Temporary assignments apply only for the duration of this command (we set then restore).
    let saved: Vec<(String, Option<String>)> = expanded_assigns
        .iter()
        .map(|(k, _)| (k.clone(), interp.vars.get(k).cloned()))
        .collect();
    for (k, v) in &expanded_assigns {
        interp.set_var(k, v.clone());
        interp.export(k); // exported to child for the command
    }

    let mut local_out = Vec::new();
    let mut local_err = Vec::new();

    interp.cmd_trace.push(argv[0].clone());

    let mut status = if let Some(body) = interp.funcs.get(&argv[0]).cloned() {
        // function call: set positional params
        let saved_pos = std::mem::replace(&mut interp.positional, argv[1..].to_vec());
        let s = exec_node(interp, &body);
        interp.positional = saved_pos;
        interp.returning.take().unwrap_or(s)
    } else {
        let cmd_stdin = if argv[0] == "read" {
            Vec::new()
        } else {
            match read_all_fd(interp, 0) {
                Ok(bytes) => bytes,
                Err(error) => {
                    local_err
                        .extend_from_slice(format!("shellsim: {}: {error}\n", argv[0]).as_bytes());
                    Vec::new()
                }
            }
        };
        crate::commands::run(interp, &argv, cmd_stdin, &mut local_out, &mut local_err)
    };

    // restore temporary assignments
    for (k, v) in saved {
        match v {
            Some(val) => {
                interp.vars.insert(k, val);
            }
            None => {
                interp.vars.remove(&k);
            }
        }
    }

    let mut output_failed = false;
    if let Err(error) = write_all_fd(interp, 1, &local_out) {
        write_diagnostic(interp, &format!("shellsim: {}: {error}\n", argv[0]));
        output_failed = true;
    }
    if write_all_fd(interp, 2, &local_err).is_err() {
        output_failed = true;
    }
    if output_failed {
        status = 1;
    }
    status
}

/// Expand a command's argv, but keep array-assignment literal words verbatim so the builtin
/// (`declare`/`local`/`typeset`/`readonly`) can parse them itself.
fn expand_argv(interp: &mut Interp, words: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for w in words {
        if is_array_assign_word(w) {
            out.push(w.clone());
        } else {
            out.extend(expand_word(interp, w, true));
        }
    }
    out
}

/// True if `w` is an array-assignment literal that must not be split/globbed:
/// `name=( … )`, `name+=( … )`, `name[sub]=…`, `name[sub]+=…`.
pub fn is_array_assign_word(w: &str) -> bool {
    let eq = match w.find('=') {
        Some(0) | None => return false,
        Some(e) => e,
    };
    let mut lhs = &w[..eq];
    if let Some(s) = lhs.strip_suffix('+') {
        lhs = s;
    }
    let has_subscript = lhs.contains('[') && lhs.ends_with(']');
    let is_array_literal = w[eq + 1..].trim_start().starts_with('(');
    if !has_subscript && !is_array_literal {
        return false;
    }
    // validate the name part
    let name = lhs.split('[').next().unwrap_or(lhs);
    !name.is_empty()
        && name
            .chars()
            .enumerate()
            .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
}

/// Apply one assignment word (`name=val`, `name+=val`, `name[sub]=val`, `name=( … )`, etc.).
/// `raw_key` retains any `[subscript]` and a trailing `+` (append); `raw_val` is unexpanded.
pub fn apply_assignment(interp: &mut Interp, raw_key: &str, raw_val: &str) {
    // Decode `+=` (append) and an optional `[subscript]`.
    let (key_body, append) = match raw_key.strip_suffix('+') {
        Some(b) => (b, true),
        None => (raw_key, false),
    };
    let (name, subscript) = match key_body.find('[') {
        Some(br) if key_body.ends_with(']') => {
            (&key_body[..br], Some(&key_body[br + 1..key_body.len() - 1]))
        }
        _ => (key_body, None),
    };

    // Array literal value: `( … )`.
    let trimmed = raw_val.trim();
    if trimmed.starts_with('(') && trimmed.ends_with(')') {
        let inner = &trimmed[1..trimmed.len() - 1];
        // Associative literal? Detect `[key]=val` pairs.
        let assoc_existing = matches!(
            interp.arrays.get(name),
            Some(crate::interp::ArrayVal::Assoc(_))
        );
        if !append {
            // fresh array
            if assoc_existing {
                if let Some(crate::interp::ArrayVal::Assoc(m)) = interp.arrays.get_mut(name) {
                    m.clear();
                }
            } else {
                interp.declare_indexed(name);
                if let Some(crate::interp::ArrayVal::Indexed(v)) = interp.arrays.get_mut(name) {
                    v.clear();
                }
            }
        }
        for (subkey, val) in parse_array_elems(interp, inner, assoc_existing) {
            match subkey {
                Some(k) => {
                    if assoc_existing {
                        interp.array_set(name, &k, val);
                    } else {
                        // indexed array with explicit [i]=val
                        interp.array_set(name, &k, val);
                    }
                }
                None => interp.array_append(name, vec![val]),
            }
        }
        return;
    }

    // Subscripted scalar assignment: name[sub]=val (val expanded, no splitting).
    if let Some(sub) = subscript {
        let sub_key = expand_word(interp, sub, false).join(" ");
        let val = expand_word(interp, raw_val, false).join(" ");
        if append {
            let prev = interp.array_get(name, &sub_key).unwrap_or_default();
            interp.array_set(name, &sub_key, format!("{prev}{val}"));
        } else {
            interp.array_set(name, &sub_key, val);
        }
        return;
    }

    // Plain scalar (or scalar-append). If the name is already an array, += appends an element
    // (bash: `arr+=str` is `arr[0]+=str`, but `arr+=(x)` was handled above).
    let val = expand_word(interp, raw_val, false).join(" ");
    if append {
        if interp.is_array(name) {
            // `arr+=val` on an array appends to element 0
            let prev = interp.array_get(name, "0").unwrap_or_default();
            interp.array_set(name, "0", format!("{prev}{val}"));
        } else {
            let prev = interp.get_var(name).unwrap_or_default();
            interp.set_var(name, format!("{prev}{val}"));
        }
    } else {
        interp.set_var(name, val);
    }
}

/// Split an array-literal body into (optional explicit key, expanded value) pairs.
/// Each top-level word undergoes expansion + word-splitting (so `$(cmd)` splits on IFS and
/// `"$x"` stays one element). `[key]=val` forms yield an explicit key.
fn parse_array_elems(
    interp: &mut Interp,
    inner: &str,
    _assoc: bool,
) -> Vec<(Option<String>, String)> {
    let mut out = Vec::new();
    for tok in split_top_level_words(inner) {
        // explicit subscript form: [key]=value
        if let Some(rest) = tok.strip_prefix('[') {
            if let Some(close) = rest.find(']') {
                let key_raw = &rest[..close];
                let after = &rest[close + 1..];
                if let Some(val_raw) = after.strip_prefix('=') {
                    let key = expand_word(interp, key_raw, false).join(" ");
                    let val = expand_word(interp, val_raw, false).join(" ");
                    out.push((Some(key), val));
                    continue;
                }
            }
        }
        // ordinary element: expand with splitting+globbing
        for v in expand_word(interp, &tok, true) {
            out.push((None, v));
        }
    }
    out
}

/// Split a string into shell words at unquoted whitespace, preserving quotes/`$( )`/`${ }`.
fn split_top_level_words(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut started = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\n' => {
                if started {
                    words.push(std::mem::take(&mut cur));
                    started = false;
                }
                i += 1;
            }
            '\'' => {
                cur.push(c);
                started = true;
                i += 1;
                while i < chars.len() {
                    cur.push(chars[i]);
                    i += 1;
                    if chars[i - 1] == '\'' {
                        break;
                    }
                }
            }
            '"' => {
                cur.push(c);
                started = true;
                i += 1;
                while i < chars.len() {
                    let d = chars[i];
                    cur.push(d);
                    i += 1;
                    if d == '\\' && i < chars.len() {
                        cur.push(chars[i]);
                        i += 1;
                        continue;
                    }
                    if d == '"' {
                        break;
                    }
                }
            }
            '\\' => {
                cur.push(c);
                started = true;
                i += 1;
                if i < chars.len() {
                    cur.push(chars[i]);
                    i += 1;
                }
            }
            '$' if chars.get(i + 1) == Some(&'(') => {
                started = true;
                let mut depth = 0;
                cur.push(chars[i]);
                i += 1;
                while i < chars.len() {
                    let d = chars[i];
                    cur.push(d);
                    i += 1;
                    if d == '(' {
                        depth += 1;
                    } else if d == ')' {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                }
            }
            '`' => {
                started = true;
                cur.push(c);
                i += 1;
                while i < chars.len() {
                    cur.push(chars[i]);
                    i += 1;
                    if chars[i - 1] == '`' {
                        break;
                    }
                }
            }
            _ => {
                cur.push(c);
                started = true;
                i += 1;
            }
        }
    }
    if started {
        words.push(cur);
    }
    words
}

fn exec_pipeline(interp: &mut Interp, stages: &[Node]) -> i32 {
    let mut input = match read_all_fd(interp, 0) {
        Ok(input) => input,
        Err(error) => {
            write_diagnostic(interp, &format!("shellsim: pipeline: {error}\n"));
            return 1;
        }
    };
    let mut last_status = 0;
    let mut statuses = Vec::new();
    for (idx, stage) in stages.iter().enumerate() {
        if interp.resources.is_stopped() {
            break;
        }
        let is_last = idx == stages.len() - 1;
        let command = describe(stage);
        let Some((status, stage_out)) = exec_child_capture_stdout(
            interp,
            stage,
            std::mem::take(&mut input),
            ChildExecution {
                command: &command,
                new_shell: false,
                retain: false,
            },
        ) else {
            write_diagnostic(interp, "shellsim: unable to create pipeline process\n");
            return 125;
        };
        last_status = status;
        statuses.push(last_status);
        if is_last {
            if let Err(error) = write_all_fd(interp, 1, &stage_out) {
                write_diagnostic(interp, &format!("shellsim: pipeline: {error}\n"));
                return 1;
            }
        } else {
            input = stage_out;
        }
    }
    if interp.opt_pipefail {
        statuses
            .into_iter()
            .rev()
            .find(|s| *s != 0)
            .unwrap_or(last_status)
    } else {
        last_status
    }
}

/// Lifecycle policy for one synchronous logical child execution.
pub(crate) struct ChildExecution<'a> {
    pub command: &'a str,
    pub new_shell: bool,
    pub retain: bool,
}

/// Execute one logical child against shared machine state and restore its parent shell state.
pub(crate) fn exec_child(
    interp: &mut Interp,
    node: &Node,
    child: ChildExecution<'_>,
) -> Option<(i32, crate::process::ProcessId)> {
    let (pid, parent) = match interp.start_child(child.command, child.new_shell) {
        Ok(child) => child,
        Err(error) => {
            write_diagnostic(interp, &format!("shellsim: {error}\n"));
            return None;
        }
    };
    let status = exec_node(interp, node);
    interp.finish_child(pid, parent, status, child.retain);
    Some((status, pid))
}

pub(crate) fn exec_child_capture_stdout(
    interp: &mut Interp,
    node: &Node,
    stdin: Vec<u8>,
    child: ChildExecution<'_>,
) -> Option<(i32, Vec<u8>)> {
    let saved = interp.process.fds.fork(&mut interp.descriptors).ok()?;
    let input = interp.descriptors.open_input(stdin).ok()?;
    if interp.install_new_description(0, input).is_err() {
        restore_fds(interp, saved);
        return None;
    }
    let output = match interp.descriptors.open_capture() {
        Ok(output) => output,
        Err(_) => {
            restore_fds(interp, saved);
            return None;
        }
    };
    if interp.install_new_description(1, output).is_err() {
        restore_fds(interp, saved);
        return None;
    }
    let result = exec_child(interp, node, child);
    let bytes = interp
        .descriptors
        .capture(output)
        .map_or_else(|_| Vec::new(), <[u8]>::to_vec);
    restore_fds(interp, saved);
    result.map(|(status, _)| (status, bytes))
}

fn expand_heredoc(interp: &mut Interp, body: &str) -> String {
    // expand $VAR, ${...}, $(...), `...` in the heredoc body, line by line
    let mut out = String::new();
    for (i, line) in body.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        // reuse double-quote expansion semantics (no splitting/globbing)
        let parts = expand_word(interp, &double_wrap(line), false);
        out.push_str(&parts.join(" "));
    }
    out
}

/// Wrap a line so expand_word treats it as a double-quoted context (expansions, no split).
fn double_wrap(line: &str) -> String {
    // escape existing double quotes and backslashes minimally
    let escaped = line.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn case_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let re = format!("^{}$", glob_to_regex_body(pattern));
    regex::Regex::new(&re)
        .map(|r| r.is_match(text))
        .unwrap_or(pattern == text)
}

fn glob_to_regex_body(pat: &str) -> String {
    let mut re = String::new();
    let mut chars = pat.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' => re.push_str(".*"),
            '?' => re.push('.'),
            '[' => {
                re.push('[');
                while let Some(&n) = chars.peek() {
                    chars.next();
                    re.push(n);
                    if n == ']' {
                        break;
                    }
                }
            }
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '{' | '}' | '\\' => {
                re.push('\\');
                re.push(c);
            }
            _ => re.push(c),
        }
    }
    re
}

#[cfg(test)]
mod array_tests {
    use crate::interp::Interp;

    /// Run a snippet and capture stdout as a String.
    fn run(src: &str) -> String {
        let mut i = Interp::new();
        let ast = crate::shell::parse(src).expect("array test source should parse");
        let mut out = Vec::new();
        let mut err = Vec::new();
        crate::exec::exec(&mut i, &ast, Vec::new(), &mut out, &mut err);
        String::from_utf8_lossy(&out).into_owned()
    }

    #[test]
    fn indexed_basics() {
        assert_eq!(
            run(r#"a=(x y z); echo "${a[1]} ${#a[@]} ${a[@]}""#),
            "y 3 x y z\n"
        );
    }

    #[test]
    fn append_and_count() {
        assert_eq!(
            run(r#"a=(x y z); a+=(w v); echo "${#a[@]} ${a[@]}""#),
            "5 x y z w v\n"
        );
    }

    #[test]
    fn sparse_indices_and_values() {
        assert_eq!(
            run(r#"a=(1 2 3); a[5]=six; echo "${!a[@]}"; echo "${a[@]}""#),
            "0 1 2 5\n1 2 3 six\n"
        );
    }

    #[test]
    fn scalar_promotes_to_array() {
        assert_eq!(
            run(r#"x=1; x[2]=3; echo "${x[0]} ${x[2]} ${#x[@]}""#),
            "1 3 2\n"
        );
    }

    #[test]
    fn bare_ref_is_element_zero() {
        assert_eq!(run(r#"a=(p q r); echo "$a ${a}""#), "p p\n");
    }

    #[test]
    fn quoted_at_separate_words() {
        // each element stays a single word even with embedded spaces
        let out = run(r#"a=("one two" three); for x in "${a[@]}"; do echo "[$x]"; done"#);
        assert_eq!(out, "[one two]\n[three]\n");
    }

    #[test]
    fn empty_array_iterates_zero_times() {
        assert_eq!(
            run(r#"a=(); for x in "${a[@]}"; do echo "X$x"; done; echo done"#),
            "done\n"
        );
    }

    #[test]
    fn command_substitution_splits() {
        assert_eq!(
            run(r#"a=($(printf "f1\nf2\nf3\n")); echo "${#a[@]} ${a[1]}""#),
            "3 f2\n"
        );
    }

    #[test]
    fn associative_get_keys_count() {
        // sorted key order is deterministic in our impl
        assert_eq!(
            run(r#"declare -A m; m[foo]=1; m[bar]=2; echo "${m[foo]} ${!m[@]} ${#m[@]}""#),
            "1 bar foo 2\n"
        );
    }

    #[test]
    fn associative_literal_and_arith() {
        let out =
            run(r#"declare -A m=([a]=0 [b]=5); m[a]=$((${m[a]} + 1)); echo "${m[a]} ${m[b]}""#);
        assert_eq!(out, "1 5\n");
    }

    #[test]
    fn slice_and_last() {
        assert_eq!(
            run(r#"a=(a b c d e); echo "${a[@]:1:2}"; echo "${a[@]: -1}""#),
            "b c\ne\n"
        );
    }

    #[test]
    fn unset_element() {
        assert_eq!(
            run(r#"a=(1 2 3 4); unset "a[1]"; echo "${a[@]} ${!a[@]}""#),
            "1 3 4 0 2 3\n"
        );
    }
}
