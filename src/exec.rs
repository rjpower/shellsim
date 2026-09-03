//! The shell executor: walks the AST against the interpreter state.

use crate::expand::{expand_word, expand_words};
use crate::interp::Interp;
use crate::shell::{Node, RedirOp, Redirect};

/// Execute a node. `stdin` is the input byte stream (from a pipe or empty). Command output
/// is appended to `out`/`err` unless redirected. Returns the exit status.
pub fn exec(
    interp: &mut Interp,
    node: &Node,
    stdin: Vec<u8>,
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
) -> i32 {
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
        Node::Command { .. } => exec_command(interp, node, stdin, out, err),
        Node::Pipeline(stages) => exec_pipeline(interp, stages, stdin, out, err),
        Node::And(a, b) => {
            let sa = exec_cond(interp, a, stdin.clone(), out, err);
            if sa == 0 && interp.exiting.is_none() {
                exec(interp, b, stdin, out, err)
            } else {
                sa
            }
        }
        Node::Or(a, b) => {
            let sa = exec_cond(interp, a, stdin.clone(), out, err);
            if sa != 0 && interp.exiting.is_none() {
                exec(interp, b, stdin, out, err)
            } else {
                sa
            }
        }
        Node::Not(a) => {
            let sa = exec_cond(interp, a, stdin, out, err);
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
                s = exec(interp, n, stdin.clone(), out, err);
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
            let id = interp.new_job(cmd);
            // Run synchronously (deterministic); file effects land immediately. `wait` is a no-op.
            let mut bout = Vec::new();
            let mut berr = Vec::new();
            let s = exec(interp, a, Vec::new(), &mut bout, &mut berr);
            if let Some(j) = interp.jobs.iter_mut().find(|j| j.id == id) {
                j.done = true;
                j.status = s;
            }
            out.extend(bout);
            err.extend(berr);
            interp.set_var("!", id.to_string());
            0
        }
        Node::Subshell(a) | Node::Group(a) => {
            // a real subshell would isolate env; we approximate Group exactly and Subshell
            // closely enough for file-state tasks (cwd is saved/restored for subshells).
            let saved_cwd = interp.cwd.clone();
            let s = exec(interp, a, stdin, out, err);
            if matches!(node, Node::Subshell(_)) {
                interp.cwd = saved_cwd;
            }
            s
        }
        Node::Redirected(inner, redirs) => {
            let mut local_out = Vec::new();
            let mut local_err = Vec::new();
            let plan = plan_redirects(interp, redirs);
            // Make a `< file` input available to `read` across loop iterations via a cursor.
            let has_stdin = redirs.iter().any(|r| r.fd == 0);
            let saved = if has_stdin {
                let s = (std::mem::take(&mut interp.input_stream), interp.input_pos);
                interp.input_stream = plan.stdin.clone();
                interp.input_pos = 0;
                Some(s)
            } else {
                None
            };
            let mut s = exec(
                interp,
                inner,
                plan.stdin.clone(),
                &mut local_out,
                &mut local_err,
            );
            if let Some((stream, pos)) = saved {
                interp.input_stream = stream;
                interp.input_pos = pos;
            }
            if !apply_outputs(interp, &plan, &local_out, &local_err, out, err) {
                s = 1;
            }
            s
        }
        Node::If {
            cond,
            then,
            elifs,
            els,
        } => {
            if exec_cond(interp, cond, stdin.clone(), out, err) == 0 {
                exec(interp, then, stdin, out, err)
            } else {
                for (c, b) in elifs {
                    if exec_cond(interp, c, stdin.clone(), out, err) == 0 {
                        return exec(interp, b, stdin, out, err);
                    }
                }
                if let Some(e) = els {
                    exec(interp, e, stdin, out, err)
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
                let c = exec_cond(interp, cond, Vec::new(), out, err);
                let go = if *until { c != 0 } else { c == 0 };
                if !go || interp.exiting.is_some() {
                    break;
                }
                s = exec(interp, body, Vec::new(), out, err);
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
                s = exec(interp, body, Vec::new(), out, err);
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
                status = exec(interp, body, Vec::new(), out, err);
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
                        return exec(interp, body, stdin, out, err);
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
fn exec_cond(
    interp: &mut Interp,
    node: &Node,
    stdin: Vec<u8>,
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
) -> i32 {
    interp.cond_depth += 1;
    let s = exec(interp, node, stdin, out, err);
    interp.cond_depth -= 1;
    s
}

fn describe(node: &Node) -> String {
    match node {
        Node::Command { words, .. } => words.join(" "),
        _ => "<job>".to_string(),
    }
}

struct RedirPlan {
    stdin: Vec<u8>,
    /// final destination for fd1 and fd2: None=parent buffer, Some((path,append))=file
    out_dest: OutDest,
    err_dest: OutDest,
}

#[derive(Clone)]
enum OutDest {
    Parent,
    File(String, bool), // (path, append)
}

fn plan_redirects(interp: &mut Interp, redirs: &[Redirect]) -> RedirPlan {
    let mut stdin = Vec::new();
    let mut out_dest = OutDest::Parent;
    let mut err_dest = OutDest::Parent;
    for r in redirs {
        match r.op {
            RedirOp::Read => {
                let path = expand_word(interp, &r.target, true).join(" ");
                stdin = interp.vfs.read(&interp.cwd, &path).unwrap_or_default();
            }
            RedirOp::Heredoc => {
                // expand $, command subs in body
                let expanded = expand_heredoc(interp, &r.target);
                stdin = expanded.into_bytes();
            }
            RedirOp::HeredocRaw => {
                stdin = r.target.clone().into_bytes();
            }
            RedirOp::HereString => {
                stdin = expand_word(interp, &r.target, false).join(" ").into_bytes();
                stdin.push(b'\n');
            }
            RedirOp::Write | RedirOp::Append => {
                let path = expand_word(interp, &r.target, true).join(" ");
                let dest = OutDest::File(path, r.op == RedirOp::Append);
                if r.fd == 2 {
                    err_dest = dest;
                } else {
                    out_dest = dest;
                }
            }
            RedirOp::DupOut => {
                // &N : duplicate target fd's destination
                let tgt = r.target.trim_start_matches('&');
                let src = tgt.parse::<i32>().unwrap_or(1);
                let dup = if src == 1 {
                    out_dest.clone()
                } else {
                    err_dest.clone()
                };
                if r.fd == 2 {
                    err_dest = dup;
                } else {
                    out_dest = dup;
                }
            }
        }
    }
    RedirPlan {
        stdin,
        out_dest,
        err_dest,
    }
}

fn apply_outputs(
    interp: &mut Interp,
    plan: &RedirPlan,
    local_out: &[u8],
    local_err: &[u8],
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
) -> bool {
    let mut ok = true;
    match &plan.out_dest {
        OutDest::Parent => out.extend_from_slice(local_out),
        OutDest::File(p, append) => {
            if let Err(error) = write_to(interp, p, local_out, *append) {
                err.extend_from_slice(format!("shellsim: {p}: {error}\n").as_bytes());
                ok = false;
            }
        }
    }
    match &plan.err_dest {
        OutDest::Parent => err.extend_from_slice(local_err),
        OutDest::File(p, append) => {
            if let Err(error) = write_to(interp, p, local_err, *append) {
                err.extend_from_slice(format!("shellsim: {p}: {error}\n").as_bytes());
                ok = false;
            }
        }
    }
    ok
}

fn write_to(interp: &mut Interp, path: &str, data: &[u8], append: bool) -> crate::vfs::Result<()> {
    if path == "/dev/null" {
        return Ok(());
    }
    if path == "/dev/stdout" {
        return Ok(());
    }
    interp.sync_vfs_time();
    let cwd = interp.cwd.clone();
    if append {
        interp.vfs.append(&cwd, path, data, 0o644)
    } else {
        interp.vfs.write(&cwd, path, data, 0o644)
    }
}

fn exec_command(
    interp: &mut Interp,
    node: &Node,
    stdin: Vec<u8>,
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
) -> i32 {
    let (assigns, words, redirects) = match node {
        Node::Command {
            assigns,
            words,
            redirects,
        } => (assigns, words, redirects),
        _ => unreachable!(),
    };

    // No command words → assignments are persistent (including array assignments).
    // Words that look like array-assignment literals (`name=( … )`, `name[i]=v`) are kept
    // verbatim — they must not be word-split/globbed — so `declare`/`local` can parse them.
    let argv = expand_argv(interp, words);
    if argv.is_empty() {
        for (k, v) in assigns {
            apply_assignment(interp, k, v);
        }
        // a bare redirection like `> file` still truncates
        if !redirects.is_empty() {
            let plan = plan_redirects(interp, redirects);
            if !apply_outputs(interp, &plan, &[], &[], out, err) {
                return 1;
            }
        }
        return 0;
    }

    // With command words, the leading assignments are temporary (scalar only here; array
    // command-prefixes don't occur in our corpus). Expand scalar values for set/restore.
    let expanded_assigns: Vec<(String, String)> = assigns
        .iter()
        .map(|(k, v)| (k.clone(), expand_word(interp, v, false).join(" ")))
        .collect();

    // Set up redirects.
    let plan = plan_redirects(interp, redirects);
    let cmd_stdin = if redirects.iter().any(|r| r.fd == 0) {
        plan.stdin.clone()
    } else {
        stdin
    };

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
        let s = exec(interp, &body, cmd_stdin, &mut local_out, &mut local_err);
        interp.positional = saved_pos;
        interp.returning.take().unwrap_or(s)
    } else {
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

    if !apply_outputs(interp, &plan, &local_out, &local_err, out, err) {
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

fn exec_pipeline(
    interp: &mut Interp,
    stages: &[Node],
    stdin: Vec<u8>,
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
) -> i32 {
    let mut input = stdin;
    let mut last_status = 0;
    let mut statuses = Vec::new();
    for (idx, stage) in stages.iter().enumerate() {
        if interp.resources.is_stopped() {
            break;
        }
        let is_last = idx == stages.len() - 1;
        let mut stage_out = Vec::new();
        // stderr of all stages flows to the shared err
        last_status = exec(
            interp,
            stage,
            std::mem::take(&mut input),
            &mut stage_out,
            err,
        );
        statuses.push(last_status);
        if is_last {
            out.extend_from_slice(&stage_out);
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
        let ast = crate::shell::parse(src);
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
