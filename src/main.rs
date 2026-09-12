//! Resource-constrained shell simulator CLI.
//!
//! Usage:
//!   shellsim -c '<command>'
//!   shellsim run <script.sh> [args...]
//!   shellsim shell [limits]
//!   shellsim eval [--cpu N] [--memory N] [--disk N] [--output N] -c '<command>'

use std::process::exit;

use shellsim::{Environment, Limits, RunOutcome};

fn main() {
    shellsim::sandbox::apply();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        run_script(&read_stdin(), &[]);
    }
    match args[1].as_str() {
        "-c" => run_script_with_stdin(
            args.get(2).map(String::as_str).unwrap_or_default(),
            &args[3..],
            &read_stdin_bytes(),
        ),
        "run" => {
            let Some(path) = args.get(2) else {
                usage_error("run requires a script path");
            };
            let source = std::fs::read_to_string(path).unwrap_or_else(|error| {
                eprintln!("shellsim: cannot read {path}: {error}");
                exit(2);
            });
            run_script_with_stdin(&source, &args[3..], &read_stdin_bytes());
        }
        "shell" => interactive_shell(&args[2..]),
        "eval" => evaluate(&args[2..]),
        command => usage_error(&format!("unknown command: {command}")),
    }
}

fn fresh_environment(limits: Limits, positional: &[String]) -> Environment {
    let mut env = Environment::with_limits(limits);
    env.positional = positional.to_vec();
    if env.vfs.put_dir("/work", 0o755).is_ok() {
        env.cwd = "/work".to_string();
        env.set_var("PWD", "/work");
    }
    env
}

fn run_script(source: &str, positional: &[String]) -> ! {
    run_script_with_stdin(source, positional, &[])
}

fn run_script_with_stdin(source: &str, positional: &[String], stdin: &[u8]) -> ! {
    let mut env = fresh_environment(Limits::default(), positional);
    let (outcome, out, err) = env.run_script_capture_with_stdin(source, stdin);
    use std::io::Write;
    let _ = std::io::stdout().write_all(&out);
    let _ = std::io::stderr().write_all(&err);
    exit(outcome.exit_status);
}

#[derive(serde::Serialize)]
struct EvalReport {
    outcome: RunOutcome,
    stdout: String,
    stderr: String,
    unsupported: Vec<String>,
    commands: Vec<String>,
    noop_commands: Vec<String>,
    partial_commands: Vec<String>,
}

fn evaluate(args: &[String]) -> ! {
    let mut limits = Limits::default();
    let mut source = None;
    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--cpu" => limits.cpu = limit_value(args, &mut i, "--cpu"),
            "--memory" => limits.memory = limit_value(args, &mut i, "--memory"),
            "--disk" => limits.disk = limit_value(args, &mut i, "--disk"),
            "--output" => limits.output = limit_value(args, &mut i, "--output"),
            "-c" => {
                i += 1;
                source = Some(
                    args.get(i)
                        .cloned()
                        .unwrap_or_else(|| usage_error("-c requires source")),
                );
            }
            "--" => {
                positional.extend_from_slice(&args[i + 1..]);
                break;
            }
            value => usage_error(&format!("unexpected eval argument: {value}")),
        }
        i += 1;
    }

    let (source, stdin) = match source {
        Some(source) => (source, read_stdin_bytes()),
        None => (read_stdin(), Vec::new()),
    };
    let mut env = fresh_environment(limits, &positional);
    let (outcome, stdout, stderr) = env.run_script_capture_with_stdin(&source, &stdin);
    let status = outcome.exit_status;
    let report = EvalReport {
        outcome,
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        unsupported: std::mem::take(&mut env.unsupported),
        commands: std::mem::take(&mut env.cmd_trace),
        noop_commands: std::mem::take(&mut env.trust_noop).into_iter().collect(),
        partial_commands: std::mem::take(&mut env.trust_partial).into_iter().collect(),
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    exit(status);
}

fn interactive_shell(args: &[String]) -> ! {
    use std::io::{BufRead, IsTerminal, Write};

    let mut limits = Limits::default();
    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--cpu" => limits.cpu = limit_value(args, &mut i, "--cpu"),
            "--memory" => limits.memory = limit_value(args, &mut i, "--memory"),
            "--disk" => limits.disk = limit_value(args, &mut i, "--disk"),
            "--output" => limits.output = limit_value(args, &mut i, "--output"),
            "--" => {
                positional.extend_from_slice(&args[i + 1..]);
                break;
            }
            value => usage_error(&format!("unexpected shell argument: {value}")),
        }
        i += 1;
    }

    let mut env = fresh_environment(limits, &positional);
    let stdin = std::io::stdin();
    let show_prompt = stdin.is_terminal();
    let mut input = stdin.lock();
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    let mut line = String::new();

    loop {
        // Python prints its own trailing `>>> ` prompt as part of each foreground action.
        if show_prompt && !env.in_python_repl() {
            let _ = write!(stdout, "shellsim:{}$ ", env.cwd);
            let _ = stdout.flush();
        }
        line.clear();
        match input.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(error) => {
                let _ = writeln!(stderr, "shellsim: input error: {error}");
                exit(1);
            }
        }

        let mut action = line.clone();
        while !shellsim::shell::heredocs_complete(&action) {
            if show_prompt {
                let _ = write!(stdout, "> ");
                let _ = stdout.flush();
            }
            line.clear();
            match input.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => action.push_str(&line),
                Err(error) => {
                    let _ = writeln!(stderr, "shellsim: input error: {error}");
                    exit(1);
                }
            }
        }

        let (outcome, out, err) = env.run_script_capture(&action);
        let _ = stdout.write_all(&out);
        let _ = stderr.write_all(&err);
        let _ = stdout.flush();
        let _ = stderr.flush();
        if env.is_terminated() {
            if let Some(reason) = outcome.stop_reason {
                let _ = writeln!(stderr, "shellsim: {reason}");
            }
            exit(outcome.exit_status);
        }
    }
    exit(env.last_status);
}

fn limit_value(args: &[String], index: &mut usize, flag: &str) -> u64 {
    *index += 1;
    args.get(*index)
        .and_then(|value| parse_quantity(value))
        .unwrap_or_else(|| {
            usage_error(&format!(
                "{flag} requires a count, optionally suffixed with k, m, or g"
            ))
        })
}

fn parse_quantity(value: &str) -> Option<u64> {
    let (number, multiplier) = match value.as_bytes().last().copied() {
        Some(b'k' | b'K') => (&value[..value.len() - 1], 1024),
        Some(b'm' | b'M') => (&value[..value.len() - 1], 1024 * 1024),
        Some(b'g' | b'G') => (&value[..value.len() - 1], 1024 * 1024 * 1024),
        _ => (value, 1),
    };
    number.parse::<u64>().ok()?.checked_mul(multiplier)
}

fn read_stdin() -> String {
    use std::io::Read;
    let mut source = String::new();
    let _ = std::io::stdin().read_to_string(&mut source);
    source
}

fn read_stdin_bytes() -> Vec<u8> {
    use std::io::{IsTerminal, Read};
    if std::io::stdin().is_terminal() {
        return Vec::new();
    }
    let mut input = Vec::new();
    let _ = std::io::stdin().read_to_end(&mut input);
    input
}

fn usage_error(message: &str) -> ! {
    eprintln!("shellsim: {message}");
    eprintln!(
        "usage: shellsim -c SOURCE | run SCRIPT [ARGS...] | shell [LIMITS] | eval [LIMITS] -c SOURCE"
    );
    exit(2);
}
