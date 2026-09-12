//! Resource-constrained shell simulator CLI.
//!
//! Usage:
//!   shellsim -c '<command>'
//!   shellsim run <script.sh> [args...]
//!   shellsim shell [limits]
//!   shellsim eval [--cpu N] [--memory N] [--disk N] [--output N] -c '<command>'
//!   shellsim serve [--cpu N] [--memory N] [--disk N] [--output N]
//!   shellsim replay SCENARIO.ndjson [--root PATH] [limits]

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
        "serve" => serve(&args[2..]),
        "replay" => replay(&args[2..]),
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

const MAX_PROTOCOL_REQUEST_BYTES: usize = 20 * 1024 * 1024;
const MAX_SCENARIO_BYTES: usize = 64 * 1024 * 1024;
const MAX_SCENARIO_ACTIONS: usize = 4_096;

struct HarnessOptions {
    limits: Limits,
    host_root: Option<String>,
}

fn harness_options(args: &[String], command: &str) -> HarnessOptions {
    let mut limits = Limits::default();
    let mut host_root = None;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--cpu" => limits.cpu = limit_value(args, &mut index, "--cpu"),
            "--memory" => limits.memory = limit_value(args, &mut index, "--memory"),
            "--disk" => limits.disk = limit_value(args, &mut index, "--disk"),
            "--output" => limits.output = limit_value(args, &mut index, "--output"),
            "--root" => {
                index += 1;
                host_root = Some(
                    args.get(index)
                        .cloned()
                        .unwrap_or_else(|| usage_error("--root requires a path")),
                );
            }
            value => usage_error(&format!("unexpected {command} argument: {value}")),
        }
        index += 1;
    }
    HarnessOptions { limits, host_root }
}

fn harness_session(options: HarnessOptions, command: &str) -> shellsim::harness::HarnessSession {
    let mut session = shellsim::harness::HarnessSession::new(options.limits);
    if let Some(host_root) = options.host_root {
        shellsim::host_ingest::mount_host_tree(
            &mut session.environment,
            std::path::Path::new(&host_root),
            "/work",
        )
        .and_then(|_| session.checkpoint_workspace())
        .unwrap_or_else(|error| {
            eprintln!("shellsim {command}: {error}");
            exit(2);
        });
    }
    session
}

fn serve(args: &[String]) -> ! {
    use std::io::Write;

    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut output = std::io::stdout().lock();
    let mut session = harness_session(harness_options(args, "serve"), "serve");
    while let Some(line) = read_bounded_protocol_line(&mut input).unwrap_or_else(|error| {
        eprintln!("shellsim serve: input error: {error}");
        exit(1);
    }) {
        let response = match line {
            Ok(line) => match serde_json::from_slice::<shellsim::harness::HarnessRequest>(&line) {
                Ok(request) => session.handle(request),
                Err(error) => protocol_error(format!("invalid request: {error}")),
            },
            Err(error) => protocol_error(error),
        };
        if serde_json::to_writer(&mut output, &response).is_err()
            || output.write_all(b"\n").is_err()
            || output.flush().is_err()
        {
            exit(1);
        }
    }
    exit(0);
}

#[derive(serde::Serialize)]
struct TranscriptRecord<'a> {
    sequence: usize,
    request: &'a shellsim::harness::HarnessRequest,
    response: &'a shellsim::harness::HarnessResponse,
}

fn replay(args: &[String]) -> ! {
    use std::io::{BufReader, Write};

    let Some(path) = args.first() else {
        usage_error("replay requires a scenario path");
    };
    let file = std::fs::File::open(path).unwrap_or_else(|error| {
        eprintln!("shellsim replay: cannot read {path}: {error}");
        exit(2);
    });
    let mut input = BufReader::new(file);
    let mut output = std::io::stdout().lock();
    let mut session = harness_session(harness_options(&args[1..], "replay"), "replay");
    let mut input_bytes = 0usize;
    let mut transcript_bytes = 0usize;
    for sequence in 0..MAX_SCENARIO_ACTIONS {
        let Some(line) = read_bounded_protocol_line(&mut input).unwrap_or_else(|error| {
            eprintln!("shellsim replay: input error: {error}");
            exit(1);
        }) else {
            exit(0);
        };
        let line = line.unwrap_or_else(|error| {
            eprintln!("shellsim replay: action {}: {error}", sequence + 1);
            exit(2);
        });
        input_bytes = input_bytes.saturating_add(line.len().saturating_add(1));
        if input_bytes > MAX_SCENARIO_BYTES {
            eprintln!("shellsim replay: scenario exceeds the {MAX_SCENARIO_BYTES}-byte limit");
            exit(2);
        }
        let request = serde_json::from_slice::<shellsim::harness::HarnessRequest>(&line)
            .unwrap_or_else(|error| {
                eprintln!(
                    "shellsim replay: action {} is invalid: {error}",
                    sequence + 1
                );
                exit(2);
            });
        let response = session.handle(request.clone());
        let record = serde_json::to_vec(&TranscriptRecord {
            sequence,
            request: &request,
            response: &response,
        })
        .unwrap();
        transcript_bytes = transcript_bytes.saturating_add(record.len() + 1);
        if transcript_bytes > MAX_SCENARIO_BYTES {
            eprintln!("shellsim replay: transcript exceeds the {MAX_SCENARIO_BYTES}-byte limit");
            exit(2);
        }
        if output.write_all(&record).is_err() || output.write_all(b"\n").is_err() {
            exit(1);
        }
    }
    if read_bounded_protocol_line(&mut input)
        .unwrap_or_else(|error| {
            eprintln!("shellsim replay: input error: {error}");
            exit(1);
        })
        .is_some()
    {
        eprintln!("shellsim replay: scenario exceeds the {MAX_SCENARIO_ACTIONS}-action limit");
        exit(2);
    }
    exit(0);
}

fn protocol_error(message: String) -> shellsim::harness::HarnessResponse {
    shellsim::harness::HarnessResponse {
        id: None,
        ok: false,
        result: None,
        error: Some(message),
    }
}

fn read_bounded_protocol_line(
    input: &mut impl std::io::BufRead,
) -> std::io::Result<Option<Result<Vec<u8>, String>>> {
    read_bounded_line(input, MAX_PROTOCOL_REQUEST_BYTES)
}

fn read_bounded_line(
    input: &mut impl std::io::BufRead,
    maximum: usize,
) -> std::io::Result<Option<Result<Vec<u8>, String>>> {
    let mut line = Vec::new();
    let mut too_large = false;
    let mut saw_input = false;
    loop {
        let available = input.fill_buf()?;
        if available.is_empty() {
            return if saw_input {
                Ok(Some(if too_large {
                    Err(format!("request exceeds the {maximum}-byte limit"))
                } else {
                    Ok(line)
                }))
            } else {
                Ok(None)
            };
        }
        saw_input = true;
        let consumed = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        if !too_large {
            let remaining = maximum.saturating_add(1).saturating_sub(line.len());
            line.extend_from_slice(&available[..consumed.min(remaining)]);
            too_large = line.len() > maximum;
        }
        let complete = available[..consumed].ends_with(b"\n");
        input.consume(consumed);
        if complete {
            if !too_large {
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
            }
            return Ok(Some(if too_large {
                Err(format!("request exceeds the {maximum}-byte limit"))
            } else {
                Ok(line)
            }));
        }
    }
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
        "usage: shellsim -c SOURCE | run SCRIPT [ARGS...] | shell [LIMITS] | eval [LIMITS] -c SOURCE | serve [LIMITS] | replay SCENARIO.ndjson [LIMITS]"
    );
    exit(2);
}

#[cfg(test)]
mod protocol_tests {
    use super::read_bounded_line;
    use std::io::Cursor;

    #[test]
    fn oversized_protocol_lines_are_discarded_without_losing_the_next_request() {
        let mut input = Cursor::new(b"123456\n{}\n".to_vec());
        let first = read_bounded_line(&mut input, 4).unwrap().unwrap();
        assert_eq!(first.unwrap_err(), "request exceeds the 4-byte limit");
        let second = read_bounded_line(&mut input, 4).unwrap().unwrap().unwrap();
        assert_eq!(second, b"{}");
    }
}
