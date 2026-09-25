//! Streaming byte and line commands with bounded scheduler continuations.
//!
//! Commands in this module preserve pipe backpressure and never access host streams.

use std::collections::{HashMap, VecDeque};

use crate::commands::util::{
    ewln, lines_of, read_inputs, read_inputs_system, split_flags, uses_standard_input, w, wln,
};
use crate::commands::{CommandContext, CommandPoll, CommandSpec, Io, Trust};
use crate::descriptors::{DeviceStream, IoPoll, IoWait, DEVICE_READ_QUANTUM};
use crate::exec::ShellPoll;
use crate::interp::Interp;
use crate::program::ProcessContext;
use crate::scheduler::WaitReason;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::{reg_resumable, reg_system_poll};
    reg_resumable(m, &["cat"], Trust::Real, cmd_cat, start_cat);
    reg_system_poll(m, "/usr/bin/tac", Trust::Real, cmd_tac);
    reg_resumable(m, &["yes"], Trust::Real, cmd_yes, start_yes);
    reg_resumable(m, &["head"], Trust::Real, cmd_head, start_head);
    reg_system_poll(m, "/usr/bin/tail", Trust::Real, cmd_tail);
}

#[derive(Clone)]
enum StreamSource {
    Descriptor,
    Device(DeviceStream),
}

#[derive(Clone)]
pub(crate) struct StreamState {
    sources: VecDeque<StreamSource>,
    pending: Vec<u8>,
    offset: usize,
}

#[derive(Clone)]
pub(crate) struct HeadStream {
    stream: StreamState,
    bytes_remaining: Option<usize>,
    lines_remaining: usize,
    done: bool,
}

/// Continuation state for text commands that must preserve pipe backpressure.
#[derive(Clone)]
pub(crate) enum TextStream {
    Cat(StreamState),
    Head(HeadStream),
    Yes(YesStream),
}

#[derive(Clone)]
pub(crate) struct YesStream {
    stream: StreamState,
    line: Vec<u8>,
}

#[derive(Clone)]
struct HeadOptions {
    lines: usize,
    bytes: Option<usize>,
    files: Vec<String>,
}

fn parse_head_options(args: &[String]) -> Result<HeadOptions, String> {
    let mut lines = 10usize;
    let mut bytes = None;
    let mut files = Vec::new();
    let mut it = args.iter().peekable();
    while let Some(arg) = it.next() {
        if arg == "-n" {
            let value = it
                .next()
                .ok_or_else(|| "option requires an argument -- 'n'".to_string())?;
            lines = value
                .trim_start_matches('-')
                .parse()
                .map_err(|_| format!("invalid number of lines: {value}"))?;
        } else if let Some(value) = arg.strip_prefix("-n") {
            lines = value
                .trim_start_matches('-')
                .parse()
                .map_err(|_| format!("invalid number of lines: {value}"))?;
        } else if arg == "-c" {
            let value = it
                .next()
                .ok_or_else(|| "option requires an argument -- 'c'".to_string())?;
            bytes = Some(
                value
                    .parse()
                    .map_err(|_| format!("invalid number of bytes: {value}"))?,
            );
        } else if let Some(value) = arg.strip_prefix("-c") {
            bytes = Some(
                value
                    .parse()
                    .map_err(|_| format!("invalid number of bytes: {value}"))?,
            );
        } else if arg.starts_with('-')
            && arg.len() > 1
            && arg[1..].chars().all(|character| character.is_ascii_digit())
        {
            lines = arg[1..].parse().unwrap_or(10);
        } else if !arg.starts_with('-') || arg == "-" {
            files.push(arg.clone());
        } else {
            return Err(format!("unimplemented option '{arg}'"));
        }
    }
    Ok(HeadOptions {
        lines,
        bytes,
        files,
    })
}

fn stream_sources(cwd: &str, files: &[String]) -> Option<VecDeque<StreamSource>> {
    if files.is_empty() {
        return Some(VecDeque::from([StreamSource::Descriptor]));
    }
    files
        .iter()
        .map(|file| {
            if file == "-" || file == "/dev/stdin" {
                Some(StreamSource::Descriptor)
            } else {
                crate::pseudo_fs::device_kind(cwd, file)
                    .map(DeviceStream::new)
                    .map(StreamSource::Device)
            }
        })
        .collect()
}

/// Return true when a command needs direct descriptor streaming instead of eager input capture.
pub(super) fn streams_before_input(command: &str, args: &[String]) -> bool {
    match command {
        "cat" => {
            if args
                .iter()
                .any(|arg| arg.starts_with('-') && arg != "-" && arg != "--")
            {
                return false;
            }
            let files = args
                .iter()
                .filter(|arg| arg.as_str() != "--")
                .cloned()
                .collect::<Vec<_>>();
            stream_sources("/", &files).is_some()
        }
        "head" => parse_head_options(args)
            .ok()
            .is_some_and(|options| stream_sources("/", &options.files).is_some()),
        _ => false,
    }
}

fn start_cat(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let files = args
        .iter()
        .filter(|arg| arg.as_str() != "--")
        .cloned()
        .collect::<Vec<_>>();
    let Some(sources) = stream_sources(&interp.cwd, &files) else {
        return CommandPoll::Ready(cmd_cat(interp, args, io));
    };
    CommandPoll::Yielded(crate::commands::CommandResume::TextStream(TextStream::Cat(
        StreamState {
            sources,
            pending: Vec::new(),
            offset: 0,
        },
    )))
}

fn start_head(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let options = match parse_head_options(args) {
        Ok(options) => options,
        Err(error) => {
            ewln(io.err, &format!("head: {error}"));
            return CommandPoll::Ready(1);
        }
    };
    let Some(sources) = stream_sources(&interp.cwd, &options.files) else {
        return CommandPoll::Ready(cmd_head(interp, args, io));
    };
    CommandPoll::Yielded(crate::commands::CommandResume::TextStream(
        TextStream::Head(HeadStream {
            stream: StreamState {
                sources,
                pending: Vec::new(),
                offset: 0,
            },
            bytes_remaining: options.bytes,
            lines_remaining: options.lines,
            done: options.bytes == Some(0) || (options.bytes.is_none() && options.lines == 0),
        }),
    ))
}

fn start_yes(_interp: &mut CommandContext<'_>, args: &[String], _io: &mut Io) -> CommandPoll {
    let line = format!(
        "{}\n",
        if args.is_empty() {
            "y".to_string()
        } else {
            args.join(" ")
        }
    )
    .into_bytes();
    CommandPoll::Yielded(crate::commands::CommandResume::TextStream(TextStream::Yes(
        YesStream {
            stream: StreamState {
                sources: VecDeque::new(),
                pending: line.clone(),
                offset: 0,
            },
            line,
        },
    )))
}

fn wait_reason(wait: IoWait) -> WaitReason {
    match wait {
        IoWait::InputReadable(description) => WaitReason::InputReadable(description),
        IoWait::PipeReadable(pipe) => WaitReason::PipeReadable(pipe),
        IoWait::PipeWritable(pipe) => WaitReason::PipeWritable(pipe),
    }
}

enum FlushPoll {
    Complete,
    Yielded,
    Blocked(WaitReason),
    Failed(i32),
}

fn resource_status(interp: &Interp) -> i32 {
    interp
        .resources
        .stop_reason()
        .map_or(137, |reason| reason.exit_status())
}

fn flush_pending(interp: &mut Interp, state: &mut StreamState) -> FlushPoll {
    if state.offset >= state.pending.len() {
        state.pending.clear();
        state.offset = 0;
        return FlushPoll::Complete;
    }
    let output_remaining =
        usize::try_from(interp.resources.output_remaining()).unwrap_or(usize::MAX);
    if output_remaining == 0 {
        let _ = interp.resources.charge_output(1);
        return FlushPoll::Failed(resource_status(interp));
    }
    let cpu_remaining = usize::try_from(interp.resources.cpu_remaining()).unwrap_or(usize::MAX);
    if cpu_remaining == 0 {
        let _ = interp.resources.charge_cpu(1);
        return FlushPoll::Failed(resource_status(interp));
    }
    let end = state
        .offset
        .saturating_add(output_remaining.min(cpu_remaining))
        .min(state.pending.len());
    match interp.write_fd(1, &state.pending[state.offset..end]) {
        Ok(IoPoll::Ready(0)) => FlushPoll::Failed(1),
        Ok(IoPoll::Ready(written)) => {
            state.offset = state.offset.saturating_add(written);
            if !interp.resources.charge_cpu(written as u64)
                || !interp.resources.charge_output(written as u64)
            {
                return FlushPoll::Failed(resource_status(interp));
            }
            if state.offset == state.pending.len() {
                state.pending.clear();
                state.offset = 0;
            }
            FlushPoll::Yielded
        }
        Ok(IoPoll::Blocked(wait)) => FlushPoll::Blocked(wait_reason(wait)),
        Err(error) if error.contains("BrokenPipe") => FlushPoll::Failed(141),
        Err(_) => FlushPoll::Failed(1),
    }
}

fn read_stream(
    interp: &mut Interp,
    sources: &mut VecDeque<StreamSource>,
    maximum: usize,
) -> Result<IoPoll<Vec<u8>>, i32> {
    loop {
        let Some(source) = sources.front_mut() else {
            return Ok(IoPoll::Ready(Vec::new()));
        };
        let result = match source {
            StreamSource::Descriptor => interp
                .read_fd(0, maximum)
                .map_err(|_| resource_status(interp))?,
            StreamSource::Device(stream) => {
                let maximum = maximum.min(DEVICE_READ_QUANTUM).min(
                    usize::try_from(interp.resources.cpu_remaining() / 2).unwrap_or(usize::MAX),
                );
                if maximum == 0 {
                    let _ = interp.resources.charge_cpu(1);
                    return Err(resource_status(interp));
                }
                if !interp.resources.charge_cpu(maximum as u64) {
                    return Err(resource_status(interp));
                }
                IoPoll::Ready(stream.read(maximum))
            }
        };
        if matches!(&result, IoPoll::Ready(bytes) if bytes.is_empty()) {
            sources.pop_front();
            continue;
        }
        return Ok(result);
    }
}

fn poll_cat(interp: &mut Interp, mut state: StreamState) -> CommandPoll {
    if !state.pending.is_empty() {
        return match flush_pending(interp, &mut state) {
            FlushPoll::Complete | FlushPoll::Yielded => CommandPoll::Yielded(
                crate::commands::CommandResume::TextStream(TextStream::Cat(state)),
            ),
            FlushPoll::Blocked(reason) => CommandPoll::Blocked(
                reason,
                crate::commands::CommandResume::TextStream(TextStream::Cat(state)),
            ),
            FlushPoll::Failed(status) => CommandPoll::Ready(status),
        };
    }
    match read_stream(interp, &mut state.sources, DEVICE_READ_QUANTUM) {
        Ok(IoPoll::Ready(bytes)) if bytes.is_empty() => CommandPoll::Ready(0),
        Ok(IoPoll::Ready(bytes)) => {
            state.pending = bytes;
            CommandPoll::Yielded(crate::commands::CommandResume::TextStream(TextStream::Cat(
                state,
            )))
        }
        Ok(IoPoll::Blocked(wait)) => CommandPoll::Blocked(
            wait_reason(wait),
            crate::commands::CommandResume::TextStream(TextStream::Cat(state)),
        ),
        Err(status) => CommandPoll::Ready(status),
    }
}

fn poll_head(interp: &mut Interp, mut state: HeadStream) -> CommandPoll {
    if !state.stream.pending.is_empty() {
        return match flush_pending(interp, &mut state.stream) {
            FlushPoll::Complete | FlushPoll::Yielded => CommandPoll::Yielded(
                crate::commands::CommandResume::TextStream(TextStream::Head(state)),
            ),
            FlushPoll::Blocked(reason) => CommandPoll::Blocked(
                reason,
                crate::commands::CommandResume::TextStream(TextStream::Head(state)),
            ),
            FlushPoll::Failed(status) => CommandPoll::Ready(status),
        };
    }
    if state.done {
        return CommandPoll::Ready(0);
    }
    let maximum = state
        .bytes_remaining
        .unwrap_or(DEVICE_READ_QUANTUM)
        .min(DEVICE_READ_QUANTUM);
    match read_stream(interp, &mut state.stream.sources, maximum) {
        Ok(IoPoll::Ready(bytes)) if bytes.is_empty() => CommandPoll::Ready(0),
        Ok(IoPoll::Ready(bytes)) => {
            let take = if let Some(remaining) = &mut state.bytes_remaining {
                let take = (*remaining).min(bytes.len());
                *remaining = remaining.saturating_sub(take);
                state.done = *remaining == 0;
                take
            } else {
                let mut take = bytes.len();
                for (index, byte) in bytes.iter().enumerate() {
                    if *byte == b'\n' {
                        state.lines_remaining = state.lines_remaining.saturating_sub(1);
                        if state.lines_remaining == 0 {
                            take = index + 1;
                            state.done = true;
                            break;
                        }
                    }
                }
                take
            };
            state.stream.pending.extend_from_slice(&bytes[..take]);
            CommandPoll::Yielded(crate::commands::CommandResume::TextStream(
                TextStream::Head(state),
            ))
        }
        Ok(IoPoll::Blocked(wait)) => CommandPoll::Blocked(
            wait_reason(wait),
            crate::commands::CommandResume::TextStream(TextStream::Head(state)),
        ),
        Err(status) => CommandPoll::Ready(status),
    }
}

fn poll_yes(interp: &mut Interp, mut state: YesStream) -> CommandPoll {
    match flush_pending(interp, &mut state.stream) {
        FlushPoll::Complete => {
            state.stream.pending.clone_from(&state.line);
            CommandPoll::Yielded(crate::commands::CommandResume::TextStream(TextStream::Yes(
                state,
            )))
        }
        FlushPoll::Yielded => CommandPoll::Yielded(crate::commands::CommandResume::TextStream(
            TextStream::Yes(state),
        )),
        FlushPoll::Blocked(reason) => CommandPoll::Blocked(
            reason,
            crate::commands::CommandResume::TextStream(TextStream::Yes(state)),
        ),
        FlushPoll::Failed(status) => CommandPoll::Ready(status),
    }
}

/// Resume one bounded streaming text-command quantum.
pub(crate) fn resume_stream(interp: &mut Interp, continuation: TextStream) -> CommandPoll {
    match continuation {
        TextStream::Cat(state) => poll_cat(interp, state),
        TextStream::Head(state) => poll_head(interp, state),
        TextStream::Yes(state) => poll_yes(interp, state),
    }
}

fn cmd_cat(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, long) = split_flags(args);
    if let Some(option) = long.first() {
        ewln(
            io.err,
            &format!("cat: unimplemented option '--{}'", option.0),
        );
        return 2;
    }
    if let Some(flag) = flags.iter().find(|flag| **flag != 'n') {
        ewln(io.err, &format!("cat: unimplemented option '-{flag}'"));
        return 2;
    }
    let number = flags.contains(&'n');
    let (data, errors) = read_inputs(interp, &ops, &io.stdin);
    if number {
        for (i, line) in data.split_inclusive(|byte| *byte == b'\n').enumerate() {
            w(io.out, &format!("{:6}\t", i + 1));
            io.out.extend_from_slice(line);
        }
    } else {
        io.out.extend_from_slice(&data);
    }
    for e in &errors {
        ewln(io.err, &format!("cat: {e}"));
    }
    if errors.is_empty() {
        0
    } else {
        1
    }
}

fn cmd_tac(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let (_f, ops, _l) = split_flags(context.args);
    if uses_standard_input(&ops) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &ops, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("tac: {error}"));
        return ShellPoll::Ready(1);
    }
    let lines = lines_of(&data);
    for l in lines.iter().rev() {
        wln(io.out, l);
    }
    ShellPoll::Ready(0)
}

fn cmd_head(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let options = match parse_head_options(args) {
        Ok(options) => options,
        Err(error) => {
            ewln(io.err, &format!("head: {error}"));
            return 1;
        }
    };
    let mut status = 0;
    let multiple = options.files.len() > 1;
    let files = if options.files.is_empty() {
        vec!["-".to_string()]
    } else {
        options.files
    };
    let mut emitted = false;
    for file in files {
        let data = if file == "-" {
            io.stdin.clone()
        } else {
            match interp.fs_read(&interp.cwd, &file) {
                Ok(data) => data,
                Err(error) => {
                    ewln(io.err, &format!("head: {file}: {error}"));
                    status = 1;
                    continue;
                }
            }
        };
        if multiple {
            if emitted {
                io.out.push(b'\n');
            }
            wln(io.out, &format!("==> {file} <=="));
        }
        if let Some(bytes) = options.bytes {
            io.out.extend_from_slice(&data[..bytes.min(data.len())]);
        } else {
            for line in data
                .split_inclusive(|byte| *byte == b'\n')
                .take(options.lines)
            {
                io.out.extend_from_slice(line);
            }
        }
        emitted = true;
    }
    status
}

fn cmd_tail(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let mut n = 10usize;
    let mut bytes = false;
    let mut from_start = false;
    let mut files = Vec::new();
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        if a == "-n" {
            let Some(v) = it.next().cloned() else {
                ewln(io.err, "tail: option requires an argument -- 'n'");
                return ShellPoll::Ready(1);
            };
            from_start = v.starts_with('+');
            n = match v.trim_start_matches('+').trim_start_matches('-').parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of lines: {v}"));
                    return ShellPoll::Ready(1);
                }
            };
        } else if a == "-c" {
            let Some(v) = it.next().cloned() else {
                ewln(io.err, "tail: option requires an argument -- 'c'");
                return ShellPoll::Ready(1);
            };
            bytes = true;
            from_start = v.starts_with('+');
            n = match v.trim_start_matches(['+', '-']).parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of bytes: {v}"));
                    return ShellPoll::Ready(1);
                }
            };
        } else if let Some(v) = a.strip_prefix("-c") {
            bytes = true;
            from_start = v.starts_with('+');
            n = match v.trim_start_matches(['+', '-']).parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of bytes: {v}"));
                    return ShellPoll::Ready(1);
                }
            };
        } else if let Some(v) = a.strip_prefix('+').filter(|value| !value.is_empty()) {
            from_start = true;
            n = match v.parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of lines: {a}"));
                    return ShellPoll::Ready(1);
                }
            };
        } else if let Some(v) = a.strip_prefix("-n") {
            from_start = v.starts_with('+');
            n = match v.trim_start_matches('+').trim_start_matches('-').parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of lines: {v}"));
                    return ShellPoll::Ready(1);
                }
            };
        } else if let Some(v) = a.strip_prefix('-').filter(|value| {
            !value.is_empty() && value.chars().all(|character| character.is_ascii_digit())
        }) {
            n = v.parse().expect("validated decimal tail count");
        } else if a == "-f" || a == "-F" {
            ewln(io.err, "tail: unimplemented follow mode");
            return ShellPoll::Ready(2);
        } else if !a.starts_with('-') || a == "-" {
            files.push(a.clone());
        } else {
            ewln(io.err, &format!("tail: unimplemented option '{a}'"));
            return ShellPoll::Ready(2);
        }
    }
    if files.is_empty() || files.iter().any(|file| file == "-") {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let multiple = files.len() > 1;
    let files = if files.is_empty() {
        vec!["-".to_string()]
    } else {
        files
    };
    let mut status = 0;
    let mut emitted = false;
    for file in files {
        let data = if file == "-" {
            io.stdin.clone()
        } else {
            let cwd = context.system.cwd().to_string();
            let maximum = usize::try_from(context.system.limits().memory).unwrap_or(usize::MAX);
            match context.system.read_file_limited(&cwd, &file, maximum) {
                Ok(data) => data,
                Err(error) => {
                    ewln(io.err, &format!("tail: {file}: {error}"));
                    status = 1;
                    continue;
                }
            }
        };
        if multiple {
            if emitted {
                io.out.push(b'\n');
            }
            wln(io.out, &format!("==> {file} <=="));
        }
        if bytes {
            let start = if from_start {
                n.saturating_sub(1).min(data.len())
            } else {
                data.len().saturating_sub(n)
            };
            io.out.extend_from_slice(&data[start..]);
        } else {
            let lines = data
                .split_inclusive(|byte| *byte == b'\n')
                .collect::<Vec<_>>();
            let start = if from_start {
                n.saturating_sub(1).min(lines.len())
            } else {
                lines.len().saturating_sub(n)
            };
            for line in &lines[start..] {
                io.out.extend_from_slice(line);
            }
        }
        emitted = true;
    }
    ShellPoll::Ready(status)
}

fn cmd_yes(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let line = format!(
        "{}\n",
        if args.is_empty() {
            "y".to_string()
        } else {
            args.join(" ")
        }
    );
    let remaining = usize::try_from(interp.resources.output_remaining()).unwrap_or(usize::MAX);
    if line.is_empty() || remaining == usize::MAX {
        ewln(io.err, "yes: a finite output limit is required");
        return 1;
    }
    let repetitions = remaining / line.len() + 1;
    let allocation = u64::try_from(repetitions.saturating_mul(line.len())).unwrap_or(u64::MAX);
    if !interp.reserve_memory(allocation) {
        return 137;
    }
    let chunk_repetitions = repetitions.min((64 * 1024 / line.len()).max(1));
    let chunk = line.repeat(chunk_repetitions);
    let mut remaining_repetitions = repetitions;
    while remaining_repetitions >= chunk_repetitions {
        io.out.extend_from_slice(chunk.as_bytes());
        remaining_repetitions -= chunk_repetitions;
    }
    for _ in 0..remaining_repetitions {
        io.out.extend_from_slice(line.as_bytes());
    }
    interp.resources.release_memory(allocation);
    0
}
