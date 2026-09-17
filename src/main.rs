//! Resource-constrained shell simulator CLI.
//!
//! Usage:
//!   shellsim -c '<command>'
//!   shellsim run <script.sh> [args...]
//!   shellsim shell [limits]
//!   shellsim eval [--cpu N] [--memory N] [--disk N] [--output N] -c '<command>'
//!   shellsim serve [--cpu N] [--memory N] [--disk N] [--output N]
//!   shellsim mcp [--root PATH] [limits]
//!   shellsim replay SCENARIO.ndjson [--root PATH] [--transcript PATH] [limits]

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
        "mcp" => mcp(&args[2..]),
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
    dropped_unsupported: u64,
    commands: Vec<String>,
    dropped_commands: u64,
    unsupported_commands: Vec<String>,
    partial_commands: Vec<String>,
    invocations: Vec<shellsim::InvocationEvent>,
    dropped_invocations: u64,
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
    let invocations = env.invocations.events();
    let unsupported_commands = invocation_names(&invocations, shellsim::CommandTrust::Unsupported);
    let partial_commands = invocation_names(&invocations, shellsim::CommandTrust::Partial);
    let report = EvalReport {
        outcome,
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        unsupported: env.unsupported.values(),
        dropped_unsupported: env.unsupported.dropped(),
        commands: env.cmd_trace.values(),
        dropped_commands: env.cmd_trace.dropped(),
        unsupported_commands,
        partial_commands,
        invocations,
        dropped_invocations: env.invocations.dropped(),
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    exit(status);
}

fn invocation_names(
    events: &[shellsim::InvocationEvent],
    trust: shellsim::CommandTrust,
) -> Vec<String> {
    events
        .iter()
        .filter(|event| event.trust == trust)
        .filter_map(|event| event.argv.first().cloned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
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
    let session = harness_session(harness_options(args, "serve"), "serve");
    let mut manager = shellsim::harness_manager::HarnessManager::new(session);
    while let Some(line) = read_bounded_protocol_line(&mut input).unwrap_or_else(|error| {
        eprintln!("shellsim serve: input error: {error}");
        exit(1);
    }) {
        let response = match line {
            Ok(line) => match serde_json::from_slice::<shellsim::harness::HarnessRequest>(&line) {
                Ok(request) => manager.handle(request),
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

fn mcp(args: &[String]) -> ! {
    use std::io::BufReader;

    let stdin = std::io::stdin();
    let mut input = BufReader::new(stdin.lock());
    let mut output = std::io::stdout().lock();
    let session = harness_session(harness_options(args, "mcp"), "mcp");
    let mut manager = shellsim::harness_manager::HarnessManager::new(session);
    if let Err(error) = shellsim::mcp::serve(&mut input, &mut output, &mut manager) {
        eprintln!("shellsim mcp: {error}");
        exit(1);
    }
    exit(0);
}

fn emit_transcript_record(
    output: &mut impl std::io::Write,
    retained: &mut Option<Vec<u8>>,
    transcript_bytes: &mut usize,
    record: &[u8],
) -> Result<(), String> {
    *transcript_bytes = transcript_bytes.saturating_add(record.len().saturating_add(1));
    if *transcript_bytes > MAX_SCENARIO_BYTES {
        return Err(format!(
            "transcript exceeds the {MAX_SCENARIO_BYTES}-byte limit"
        ));
    }
    output
        .write_all(record)
        .map_err(|error| error.to_string())?;
    output.write_all(b"\n").map_err(|error| error.to_string())?;
    if let Some(transcript) = retained.as_mut() {
        transcript
            .try_reserve_exact(record.len().saturating_add(1))
            .map_err(|_| "cannot allocate bounded transcript buffer".to_string())?;
        transcript.extend_from_slice(record);
        transcript.push(b'\n');
    }
    Ok(())
}

fn replay(args: &[String]) -> ! {
    use std::io::BufReader;

    use shellsim::scenario::{self, Action, FinalExpectation, Header};

    let Some(path) = args.first() else {
        usage_error("replay requires a scenario path");
    };
    let (transcript_path, harness_args) = replay_options(&args[1..]);
    let file = std::fs::File::open(path).unwrap_or_else(|error| {
        eprintln!("shellsim replay: cannot read {path}: {error}");
        exit(2);
    });
    let mut input = BufReader::new(file);
    let mut input_bytes = 0usize;
    let first_line = read_bounded_protocol_line(&mut input).unwrap_or_else(|error| {
        eprintln!("shellsim replay: input error: {error}");
        exit(1);
    });
    let (metadata, mut pending_action) = match first_line {
        Some(Ok(line)) => {
            input_bytes = line.len().saturating_add(1);
            let is_header = serde_json::from_slice::<serde_json::Value>(&line)
                .ok()
                .is_some_and(|value| value.get("scenario").is_some());
            if is_header {
                let header = serde_json::from_slice::<Header>(&line).unwrap_or_else(|error| {
                    eprintln!("shellsim replay: invalid scenario header: {error}");
                    exit(2);
                });
                if header.scenario.version != scenario::FORMAT_VERSION {
                    eprintln!(
                        "shellsim replay: unsupported scenario version {} (expected {})",
                        header.scenario.version,
                        scenario::FORMAT_VERSION
                    );
                    exit(2);
                }
                (Some(header.scenario), None)
            } else {
                (None, Some(line))
            }
        }
        Some(Err(error)) => {
            eprintln!("shellsim replay: action 1: {error}");
            exit(2);
        }
        None => (None, None),
    };
    let mut output = std::io::stdout().lock();
    let session = harness_session(harness_options(&harness_args, "replay"), "replay");
    let mut manager = shellsim::harness_manager::HarnessManager::new(session);
    let mut transcript_bytes = 0usize;
    let mut retained_transcript = transcript_path.as_ref().map(|_| Vec::new());
    let mut sequence = 0usize;
    let mut assertion_failed = false;
    let mut strict_failures = Vec::new();
    if let Some(metadata) = metadata.as_ref() {
        let record = scenario::serialize_header(metadata);
        emit_transcript_record(
            &mut output,
            &mut retained_transcript,
            &mut transcript_bytes,
            &record,
        )
        .unwrap_or_else(|error| {
            eprintln!("shellsim replay: {error}");
            exit(1);
        });
    }
    while sequence < MAX_SCENARIO_ACTIONS {
        let line = if let Some(line) = pending_action.take() {
            line
        } else {
            let Some(line) = read_bounded_protocol_line(&mut input).unwrap_or_else(|error| {
                eprintln!("shellsim replay: input error: {error}");
                exit(1);
            }) else {
                break;
            };
            let line = line.unwrap_or_else(|error| {
                eprintln!("shellsim replay: action {}: {error}", sequence + 1);
                exit(2);
            });
            input_bytes = input_bytes.saturating_add(line.len().saturating_add(1));
            line
        };
        if input_bytes > MAX_SCENARIO_BYTES {
            eprintln!("shellsim replay: scenario exceeds the {MAX_SCENARIO_BYTES}-byte limit");
            exit(2);
        }
        let action = serde_json::from_slice::<Action>(&line).unwrap_or_else(|error| {
            eprintln!(
                "shellsim replay: action {} is invalid: {error}",
                sequence + 1
            );
            exit(2);
        });
        let (request, expectation) = match action {
            Action::Request(request) => (request, None),
            Action::Asserted { request, expect } => (request, Some(expect)),
        };
        let response = manager.handle(request.clone());
        let assertion = expectation
            .as_ref()
            .map(|expected| scenario::check_expectation(expected, &response));
        if assertion.as_ref().is_some_and(|result| !result.passed) {
            assertion_failed = true;
            eprintln!("shellsim replay: action {} assertion failed", sequence + 1);
        }
        if metadata.as_ref().is_some_and(|metadata| metadata.strict) {
            strict_failures.extend(scenario::strict_response_failures(sequence, &response));
        }
        let record = scenario::serialize_action(
            sequence,
            &request,
            &response,
            expectation.as_ref(),
            assertion.as_ref(),
        );
        emit_transcript_record(
            &mut output,
            &mut retained_transcript,
            &mut transcript_bytes,
            &record,
        )
        .unwrap_or_else(|error| {
            eprintln!("shellsim replay: {error}");
            exit(1);
        });
        sequence += 1;
    }
    if sequence == MAX_SCENARIO_ACTIONS
        && read_bounded_protocol_line(&mut input)
            .unwrap_or_else(|error| {
                eprintln!("shellsim replay: input error: {error}");
                exit(1);
            })
            .is_some()
    {
        eprintln!("shellsim replay: scenario exceeds the {MAX_SCENARIO_ACTIONS}-action limit");
        exit(2);
    }
    if let Some(metadata) = metadata.as_ref() {
        strict_failures.sort();
        strict_failures.dedup();
        let default_final = FinalExpectation::default();
        let expected = metadata
            .final_expectation
            .as_ref()
            .unwrap_or(&default_final);
        let final_assertion = scenario::check_final_expectation(
            &mut manager,
            expected,
            metadata.strict,
            strict_failures,
        );
        if !final_assertion.passed {
            assertion_failed = true;
            eprintln!("shellsim replay: final assertion failed");
        }
        let record =
            scenario::serialize_final(metadata.final_expectation.as_ref(), &final_assertion);
        emit_transcript_record(
            &mut output,
            &mut retained_transcript,
            &mut transcript_bytes,
            &record,
        )
        .unwrap_or_else(|error| {
            eprintln!("shellsim replay: {error}");
            exit(1);
        });
    }
    if let (Some(path), Some(transcript)) = (transcript_path, retained_transcript) {
        persist_transcript(std::path::Path::new(&path), &transcript).unwrap_or_else(|error| {
            eprintln!("shellsim replay: cannot persist transcript to {path}: {error}");
            exit(2);
        });
    }
    exit(i32::from(assertion_failed));
}

fn replay_options(args: &[String]) -> (Option<String>, Vec<String>) {
    let mut transcript = None;
    let mut harness_args = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        if args[index] == "--transcript" {
            index += 1;
            let path = args
                .get(index)
                .cloned()
                .unwrap_or_else(|| usage_error("--transcript requires a path"));
            if transcript.replace(path).is_some() {
                usage_error("--transcript may only be specified once");
            }
        } else {
            harness_args.push(args[index].clone());
        }
        index += 1;
    }
    (transcript, harness_args)
}

/// Install a completed transcript without exposing a partial file or replacing an existing path.
fn persist_transcript(path: &std::path::Path, contents: &[u8]) -> Result<(), String> {
    use std::io::Write;

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut temporary = None;
    for attempt in 0..128u32 {
        let candidate = parent.join(format!(
            ".shellsim-transcript-{}-{attempt}.tmp",
            std::process::id()
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    let Some((temporary_path, mut file)) = temporary else {
        return Err("could not allocate a temporary file after 128 attempts".to_string());
    };
    let result = (|| -> std::io::Result<()> {
        file.write_all(contents)?;
        file.sync_all()?;
        drop(file);
        std::fs::hard_link(&temporary_path, path)?;
        std::fs::remove_file(&temporary_path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary_path);
    }
    result.map_err(|error| {
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            "destination already exists".to_string()
        } else {
            error.to_string()
        }
    })
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
        "usage: shellsim -c SOURCE | run SCRIPT [ARGS...] | shell [LIMITS] | eval [LIMITS] -c SOURCE | serve [LIMITS] | replay SCENARIO.ndjson [--transcript PATH] [LIMITS]"
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
