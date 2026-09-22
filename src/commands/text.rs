//! Text processing: output (cat/tac/tee), windowing (head/tail), counting and reshaping
//! (wc/uniq/cut/tr/rev/nl/seq/paste/comm/diff/cmp), table and formatting utilities, pattern
//! tools (grep/sed), xargs, and the small arithmetic helpers expr/bc.

use std::collections::{HashMap, VecDeque};

use crate::commands::options::{parse_options_or_report, OptionSpec};
use crate::commands::regex_compat::basic_regex_to_rust;
use crate::commands::util::{ewln, glob_eq, lines_of, read_inputs, split_flags, w, wln};
use crate::commands::{ChildCommand, CommandContext, CommandPoll, CommandSpec, Io, Trust};
use crate::descriptors::{DeviceStream, IoPoll, IoWait, DEVICE_READ_QUANTUM};
use crate::interp::Interp;
use crate::scheduler::WaitReason;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::{reg, reg_buffered_resumable, reg_resumable, reg_unsupported};
    reg_resumable(m, &["cat"], Trust::Real, cmd_cat, start_cat);
    reg(m, &["tac"], Trust::Real, cmd_tac);
    reg(m, &["tee"], Trust::Real, cmd_tee);
    reg_resumable(m, &["yes"], Trust::Real, cmd_yes, start_yes);
    reg_resumable(m, &["head"], Trust::Real, cmd_head, start_head);
    reg(m, &["tail"], Trust::Real, cmd_tail);
    reg(m, &["wc"], Trust::Real, cmd_wc);
    reg(m, &["uniq"], Trust::Real, cmd_uniq);
    reg(m, &["cut"], Trust::Real, cmd_cut);
    reg(m, &["tr"], Trust::Real, cmd_tr);
    reg(m, &["rev"], Trust::Real, cmd_rev);
    reg(m, &["grep"], Trust::Partial, cmd_grep);
    reg(m, &["egrep"], Trust::Partial, cmd_egrep);
    reg(m, &["fgrep"], Trust::Partial, cmd_fgrep);
    reg(m, &["nl"], Trust::Real, cmd_nl);
    reg(m, &["seq"], Trust::Real, cmd_seq);
    reg(m, &["paste"], Trust::Real, cmd_paste);
    reg_buffered_resumable(m, &["fold"], Trust::Real, cmd_fold, start_buffered_text);
    reg_buffered_resumable(m, &["fmt"], Trust::Partial, cmd_fmt, start_buffered_text);
    reg_buffered_resumable(m, &["expand"], Trust::Real, cmd_expand, start_buffered_text);
    reg_buffered_resumable(
        m,
        &["unexpand"],
        Trust::Real,
        cmd_unexpand,
        start_buffered_text,
    );
    reg_buffered_resumable(
        m,
        &["column"],
        Trust::Partial,
        cmd_column,
        start_buffered_text,
    );
    reg_unsupported(m, &["pr"]);
    reg_buffered_resumable(m, &["xargs"], Trust::Real, cmd_xargs, start_xargs);
    reg(m, &["comm"], Trust::Real, cmd_comm);
    reg(m, &["diff"], Trust::Partial, cmd_diff);
    reg(m, &["cmp"], Trust::Real, cmd_cmp);
    reg_buffered_resumable(m, &["join"], Trust::Partial, cmd_join, start_buffered_text);
    reg_buffered_resumable(
        m,
        &["split"],
        Trust::Partial,
        cmd_split,
        start_buffered_text,
    );
    reg_buffered_resumable(m, &["shuf"], Trust::Partial, cmd_shuf, start_buffered_text);
    reg_buffered_resumable(m, &["tsort"], Trust::Real, cmd_tsort, start_buffered_text);
    reg(m, &["expr"], Trust::Real, cmd_expr);
    reg(m, &["bc"], Trust::Real, cmd_bc);
    reg_unsupported(m, &["factor"]);
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
    let (flags, ops, _long) = split_flags(args);
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

fn cmd_tac(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    let (data, errors) = read_inputs(interp, &ops, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("tac: {error}"));
        return 1;
    }
    let lines = lines_of(&data);
    for l in lines.iter().rev() {
        wln(io.out, l);
    }
    0
}

fn cmd_tee(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    let append = flags.contains(&'a');
    let cwd = interp.cwd.clone();
    let mut status = 0;
    for f in &ops {
        let result = if append {
            interp.vfs.append(&cwd, f, &io.stdin, 0o644)
        } else {
            interp.vfs.write(&cwd, f, &io.stdin, 0o644)
        };
        if let Err(error) = result {
            ewln(io.err, &format!("tee: {f}: {error}"));
            status = 1;
        }
    }
    io.out.extend_from_slice(&io.stdin);
    status
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

fn cmd_tail(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut n = 10usize;
    let mut bytes = false;
    let mut from_start = false;
    let mut files = Vec::new();
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        if a == "-n" {
            let Some(v) = it.next().cloned() else {
                ewln(io.err, "tail: option requires an argument -- 'n'");
                return 1;
            };
            from_start = v.starts_with('+');
            n = match v.trim_start_matches('+').trim_start_matches('-').parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of lines: {v}"));
                    return 1;
                }
            };
        } else if a == "-c" {
            let Some(v) = it.next().cloned() else {
                ewln(io.err, "tail: option requires an argument -- 'c'");
                return 1;
            };
            bytes = true;
            from_start = v.starts_with('+');
            n = match v.trim_start_matches(['+', '-']).parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of bytes: {v}"));
                    return 1;
                }
            };
        } else if let Some(v) = a.strip_prefix("-c") {
            bytes = true;
            from_start = v.starts_with('+');
            n = match v.trim_start_matches(['+', '-']).parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of bytes: {v}"));
                    return 1;
                }
            };
        } else if let Some(v) = a.strip_prefix('+').filter(|value| !value.is_empty()) {
            from_start = true;
            n = match v.parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of lines: {a}"));
                    return 1;
                }
            };
        } else if let Some(v) = a.strip_prefix("-n") {
            from_start = v.starts_with('+');
            n = match v.trim_start_matches('+').trim_start_matches('-').parse() {
                Ok(value) => value,
                Err(_) => {
                    ewln(io.err, &format!("tail: invalid number of lines: {v}"));
                    return 1;
                }
            };
        } else if let Some(v) = a.strip_prefix('-').filter(|value| {
            !value.is_empty() && value.chars().all(|character| character.is_ascii_digit())
        }) {
            n = v.parse().expect("validated decimal tail count");
        } else if a == "-f" || a == "-F" {
            ewln(io.err, "tail: unimplemented follow mode");
            return 2;
        } else if !a.starts_with('-') || a == "-" {
            files.push(a.clone());
        } else {
            ewln(io.err, &format!("tail: unimplemented option '{a}'"));
            return 2;
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
            match interp.fs_read(&interp.cwd, &file) {
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
    status
}

fn cmd_wc(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    if flags
        .iter()
        .any(|flag| !matches!(flag, 'l' | 'w' | 'c' | 'm'))
    {
        ewln(io.err, "wc: unimplemented option");
        return 2;
    }
    let (cl, cw, cc, cm) = (
        flags.contains(&'l'),
        flags.contains(&'w'),
        flags.contains(&'c'),
        flags.contains(&'m'),
    );
    let none = !cl && !cw && !cc && !cm;
    let counts = |data: &[u8]| {
        let s = String::from_utf8_lossy(data);
        (
            s.matches('\n').count(),
            s.split_whitespace().count(),
            data.len(),
            s.chars().count(),
        )
    };
    let print_counts = |values: (usize, usize, usize, usize), out: &mut Vec<u8>, label: &str| {
        let (lines, words, bytes, chars) = values;
        let mut parts = Vec::new();
        if cl || none {
            parts.push(format!("{:>7}", lines));
        }
        if cw || none {
            parts.push(format!("{:>7}", words));
        }
        if cc || none {
            parts.push(format!("{:>7}", bytes));
        }
        if cm {
            parts.push(format!("{:>7}", chars));
        }
        let mut line = parts.join(" ");
        if !label.is_empty() {
            line.push(' ');
            line.push_str(label);
        }
        wln(out, line.trim_start());
    };
    if ops.is_empty() {
        print_counts(counts(&io.stdin), io.out, "");
    } else {
        let mut totals = (0usize, 0usize, 0usize, 0usize);
        let mut status = 0;
        for f in &ops {
            match interp.fs_read(&interp.cwd, f) {
                Ok(data) => {
                    let values = counts(&data);
                    print_counts(values, io.out, f);
                    totals.0 = totals.0.saturating_add(values.0);
                    totals.1 = totals.1.saturating_add(values.1);
                    totals.2 = totals.2.saturating_add(values.2);
                    totals.3 = totals.3.saturating_add(values.3);
                }
                Err(error) => {
                    ewln(io.err, &format!("wc: {f}: {error}"));
                    status = 1;
                }
            }
        }
        if ops.len() > 1 {
            print_counts(totals, io.out, "total");
        }
        return status;
    }
    0
}

fn cmd_uniq(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut count = false;
    let mut only_dup = false;
    let mut only_uniq = false;
    let mut ignore_case = false;
    let mut skip_fields = 0usize;
    let mut skip_chars = 0usize;
    let mut operands = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        let parse_count = |value: &str, option: &str, io: &mut Io| {
            value.parse::<usize>().map_err(|_| {
                ewln(
                    io.err,
                    &format!("uniq: invalid number for {option}: {value}"),
                );
                1
            })
        };
        if argument == "-f" || argument == "-s" {
            let Some(value) = args.get(index + 1) else {
                ewln(io.err, &format!("uniq: {argument} requires an argument"));
                return 1;
            };
            let parsed = match parse_count(value, argument, io) {
                Ok(value) => value,
                Err(status) => return status,
            };
            if argument == "-f" {
                skip_fields = parsed;
            } else {
                skip_chars = parsed;
            }
            index += 2;
            continue;
        }
        if let Some(value) = argument
            .strip_prefix("-f")
            .filter(|value| !value.is_empty())
        {
            skip_fields = match parse_count(value, "-f", io) {
                Ok(value) => value,
                Err(status) => return status,
            };
        } else if let Some(value) = argument
            .strip_prefix("-s")
            .filter(|value| !value.is_empty())
        {
            skip_chars = match parse_count(value, "-s", io) {
                Ok(value) => value,
                Err(status) => return status,
            };
        } else if argument.starts_with('-') && argument != "-" {
            for flag in argument[1..].chars() {
                match flag {
                    'c' => count = true,
                    'd' => only_dup = true,
                    'u' => only_uniq = true,
                    'i' => ignore_case = true,
                    _ => {
                        ewln(io.err, &format!("uniq: unimplemented option '-{flag}'"));
                        return 2;
                    }
                }
            }
        } else {
            operands.push(argument);
        }
        index += 1;
    }
    if operands.len() > 1 {
        ewln(io.err, "uniq: unimplemented output-file operand");
        return 2;
    }
    let (data, errors) = read_inputs(interp, &operands, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("uniq: {error}"));
        return 1;
    }
    let lines = data
        .split_inclusive(|byte| *byte == b'\n')
        .map(<[u8]>::to_vec)
        .collect::<Vec<_>>();
    let key = |line: &[u8]| {
        let text = String::from_utf8_lossy(line);
        let mut offset = 0;
        for _ in 0..skip_fields {
            let rest = &text[offset..];
            let blanks = rest.len() - rest.trim_start_matches(char::is_whitespace).len();
            offset += blanks;
            let rest = &text[offset..];
            let field = rest.find(char::is_whitespace).unwrap_or(rest.len());
            offset += field;
        }
        let value = text[offset..].chars().skip(skip_chars).collect::<String>();
        if ignore_case {
            value.to_lowercase()
        } else {
            value
        }
    };
    let mut i = 0;
    while i < lines.len() {
        let mut j = i + 1;
        let current = key(&lines[i]);
        while j < lines.len() && key(&lines[j]) == current {
            j += 1;
        }
        let n = j - i;
        let show = (!only_dup && !only_uniq) || (only_dup && n > 1) || (only_uniq && n == 1);
        if show {
            if count {
                w(io.out, &format!("{:>7} ", n));
                io.out.extend_from_slice(&lines[i]);
            } else {
                io.out.extend_from_slice(&lines[i]);
            }
        }
        i = j;
    }
    0
}

fn cmd_cut(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut delim = '\t';
    let mut fields: Option<String> = None;
    let mut chars_spec: Option<String> = None;
    let mut files = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if let Some(d) = a.strip_prefix("-d") {
            delim = if d.is_empty() {
                it.next().and_then(|s| s.chars().next()).unwrap_or('\t')
            } else {
                d.chars().next().unwrap_or('\t')
            };
        } else if let Some(f) = a.strip_prefix("-f") {
            fields = Some(if f.is_empty() {
                it.next().cloned().unwrap_or_default()
            } else {
                f.to_string()
            });
        } else if let Some(c) = a.strip_prefix("-c") {
            chars_spec = Some(if c.is_empty() {
                it.next().cloned().unwrap_or_default()
            } else {
                c.to_string()
            });
        } else if !a.starts_with('-') {
            files.push(a);
        }
    }
    let (data, errors) = read_inputs(interp, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("cut: {error}"));
        return 1;
    }
    let parse_ranges = |spec: &str, max: usize| -> Vec<usize> {
        let mut idx = Vec::new();
        for part in spec.split(',') {
            if let Some((a, b)) = part.split_once('-') {
                let lo: usize = a.parse().unwrap_or(1);
                let hi: usize = if b.is_empty() {
                    max
                } else {
                    b.parse().unwrap_or(max)
                };
                for k in lo..=hi.min(max) {
                    idx.push(k);
                }
            } else if let Ok(k) = part.parse() {
                idx.push(k);
            }
        }
        idx
    };
    for line in String::from_utf8_lossy(&data).lines() {
        if let Some(spec) = &fields {
            let parts: Vec<&str> = line.split(delim).collect();
            if !line.contains(delim) {
                wln(io.out, line);
                continue;
            }
            let idx = parse_ranges(spec, parts.len());
            let selected: Vec<&str> = idx
                .iter()
                .filter_map(|k| parts.get(k - 1).copied())
                .collect();
            wln(io.out, &selected.join(&delim.to_string()));
        } else if let Some(spec) = &chars_spec {
            let chars: Vec<char> = line.chars().collect();
            let idx = parse_ranges(spec, chars.len());
            let selected: String = idx.iter().filter_map(|k| chars.get(k - 1)).collect();
            wln(io.out, &selected);
        }
    }
    0
}

fn cmd_tr(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    let delete = flags.contains(&'d');
    let squeeze = flags.contains(&'s');
    let complement = flags.contains(&'c');
    if flags.iter().any(|flag| !matches!(flag, 'c' | 'd' | 's')) {
        ewln(io.err, "tr: unimplemented option");
        return 2;
    }
    let required = if delete || (squeeze && ops.len() == 1) {
        1
    } else {
        2
    };
    if ops.len() != required {
        ewln(io.err, "tr: missing or extra operand");
        return 1;
    }
    let set1 = ops.first().map(|s| expand_tr_set(s)).unwrap_or_default();
    let set2 = ops
        .get(1)
        .map(|s| expand_tr_set(s))
        .unwrap_or_else(|| set1.clone());
    let input = String::from_utf8_lossy(&io.stdin).into_owned();
    let mut result = String::new();
    if delete {
        for c in input.chars() {
            let in_set = set1.contains(&c);
            if in_set != complement {
                continue;
            }
            result.push(c);
        }
    } else {
        let mut last = None;
        for c in input.chars() {
            let mapped = if let Some(pos) = set1.iter().position(|x| *x == c) {
                set2.get(pos)
                    .copied()
                    .or_else(|| set2.last().copied())
                    .unwrap_or(c)
            } else {
                c
            };
            if squeeze && Some(mapped) == last && set2.contains(&mapped) {
                continue;
            }
            result.push(mapped);
            last = Some(mapped);
        }
    }
    w(io.out, &result);
    0
}

fn expand_tr_set(s: &str) -> Vec<char> {
    // handle ranges like a-z and classes [:digit:] minimally
    let mut out = Vec::new();
    let s = s
        .replace("[:digit:]", "0123456789")
        .replace("[:lower:]", "abcdefghijklmnopqrstuvwxyz")
        .replace("[:upper:]", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        .replace(
            "[:alpha:]",
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        )
        .replace("[:space:]", " \t\n\r");
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() {
            out.push(match chars[i + 1] {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '\\' => '\\',
                value => value,
            });
            i += 2;
        } else if i + 2 < chars.len() && chars[i + 1] == '-' {
            let (lo, hi) = (chars[i], chars[i + 2]);
            for c in lo..=hi {
                out.push(c);
            }
            i += 3;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn cmd_rev(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    let (data, errors) = read_inputs(interp, &ops, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("rev: {error}"));
        return 1;
    }
    for line in data.split_inclusive(|byte| *byte == b'\n') {
        let newline = line.ends_with(b"\n");
        let content = if newline {
            &line[..line.len() - 1]
        } else {
            line
        };
        let reversed = String::from_utf8_lossy(content)
            .chars()
            .rev()
            .collect::<String>();
        w(io.out, &reversed);
        if newline {
            io.out.push(b'\n');
        }
    }
    0
}

fn cmd_nl(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    let (data, errors) = read_inputs(interp, &ops, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("nl: {error}"));
        return 1;
    }
    let mut n = 1;
    for line in String::from_utf8_lossy(&data).lines() {
        if line.is_empty() {
            wln(io.out, "");
        } else {
            wln(io.out, &format!("{:>6}\t{}", n, line));
            n += 1;
        }
    }
    0
}

fn cmd_seq(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args
        .iter()
        .any(|argument| argument.starts_with('-') && argument.parse::<f64>().is_err())
    {
        ewln(io.err, "seq: unimplemented option");
        return 2;
    }
    let nums: Vec<f64> = match args.iter().map(|a| a.parse::<f64>()).collect() {
        Ok(values) => values,
        Err(_) => {
            ewln(io.err, "seq: invalid floating point argument");
            return 1;
        }
    };
    let (start, step, end) = match nums.len() {
        1 => (1.0, 1.0, nums[0]),
        2 => (nums[0], 1.0, nums[1]),
        3 => (nums[0], nums[1], nums[2]),
        _ => {
            ewln(io.err, "seq: missing or extra operand");
            return 1;
        }
    };
    let mut x = start;
    let int = start.fract() == 0.0 && step.fract() == 0.0 && end.fract() == 0.0;
    if step > 0.0 {
        while x <= end + 1e-9 {
            wln(io.out, &fmt_num(x, int));
            x += step;
        }
    } else if step < 0.0 {
        while x >= end - 1e-9 {
            wln(io.out, &fmt_num(x, int));
            x += step;
        }
    } else {
        ewln(io.err, "seq: zero increment");
        return 1;
    }
    0
}

fn fmt_num(x: f64, int: bool) -> String {
    if int {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

fn cmd_paste(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut delim = '\t';
    let mut files = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "-d" {
            delim = it.next().and_then(|s| s.chars().next()).unwrap_or('\t');
        } else if let Some(d) = a.strip_prefix("-d") {
            delim = d.chars().next().unwrap_or('\t');
        } else if a.starts_with('-') && a != "-" {
            ewln(io.err, &format!("paste: unimplemented option '{a}'"));
            return 2;
        } else {
            files.push(a.clone());
        }
    }
    if files.is_empty() {
        io.out.extend_from_slice(&io.stdin);
        return 0;
    }
    let mut columns = Vec::new();
    for file in &files {
        if file == "-" {
            columns.push(lines_of(&io.stdin));
        } else {
            match interp.fs_read(&interp.cwd, file) {
                Ok(data) => columns.push(lines_of(&data)),
                Err(error) => {
                    ewln(io.err, &format!("paste: {file}: {error}"));
                    return 1;
                }
            }
        }
    }
    let max = columns.iter().map(|c| c.len()).max().unwrap_or(0);
    for i in 0..max {
        let row: Vec<String> = columns
            .iter()
            .map(|c| c.get(i).cloned().unwrap_or_default())
            .collect();
        wln(io.out, &row.join(&delim.to_string()));
    }
    0
}

fn cmd_xargs(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let commands = match xargs_commands(args, &io.stdin, io.err) {
        Ok(commands) => commands,
        Err(status) => return status,
    };
    let mut status = 0;
    for argv in commands {
        status = crate::commands::run(interp, &argv, Vec::new(), io.out, io.err);
    }
    status
}

fn start_xargs(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> CommandPoll {
    let commands = match xargs_commands(args, &io.stdin, io.err) {
        Ok(commands) => commands,
        Err(status) => return CommandPoll::Ready(status),
    };
    crate::commands::start_child_sequence(
        interp,
        commands
            .into_iter()
            .map(|argv| ChildCommand {
                argv,
                stdin: Vec::new(),
                cwd: None,
                environment: None,
            })
            .collect(),
        false,
    )
}

fn xargs_commands(
    args: &[String],
    stdin: &[u8],
    err: &mut Vec<u8>,
) -> Result<Vec<Vec<String>>, i32> {
    let mut i = 0;
    let mut replace: Option<String> = None;
    let mut nper: Option<usize> = None;
    let mut nul_delimited = false;
    let mut no_run_if_empty = false;
    while i < args.len() {
        match args[i].as_str() {
            "-I" => {
                let Some(value) = args.get(i + 1) else {
                    ewln(err, "xargs: option requires an argument -- 'I'");
                    return Err(1);
                };
                replace = Some(value.clone());
                i += 2;
            }
            option if option.starts_with("-I") && option.len() > 2 => {
                replace = Some(option[2..].to_string());
                i += 1;
            }
            "-n" => {
                let Some(value) = args.get(i + 1).and_then(|s| s.parse().ok()) else {
                    ewln(err, "xargs: invalid number for -n");
                    return Err(1);
                };
                if value == 0 {
                    ewln(err, "xargs: -n requires a positive number");
                    return Err(1);
                }
                nper = Some(value);
                i += 2;
            }
            option if option.starts_with("-n") && option.len() > 2 => {
                let Some(value) = option[2..].parse().ok() else {
                    ewln(err, "xargs: invalid number for -n");
                    return Err(1);
                };
                if value == 0 {
                    ewln(err, "xargs: -n requires a positive number");
                    return Err(1);
                }
                nper = Some(value);
                i += 1;
            }
            "-0" => {
                nul_delimited = true;
                i += 1;
            }
            "-r" | "--no-run-if-empty" => {
                no_run_if_empty = true;
                i += 1;
            }
            "--" => {
                i += 1;
                break;
            }
            option if option.starts_with('-') => {
                ewln(err, &format!("xargs: unsupported option '{option}'"));
                return Err(1);
            }
            _ => break,
        }
    }
    let mut cmd: Vec<String> = args[i..].to_vec();
    if cmd.is_empty() {
        cmd.push("echo".to_string());
    }
    let input = String::from_utf8_lossy(stdin);
    let tokens: Vec<String> = if nul_delimited {
        input
            .split('\0')
            .filter(|token| !token.is_empty())
            .map(str::to_string)
            .collect()
    } else {
        input.split_whitespace().map(str::to_string).collect()
    };
    if tokens.is_empty() {
        return Ok(if no_run_if_empty {
            Vec::new()
        } else {
            vec![cmd]
        });
    }
    let mut commands = Vec::new();
    if let Some(ph) = replace {
        for token in &tokens {
            let argv: Vec<String> = cmd.iter().map(|c| c.replace(&ph, token)).collect();
            commands.push(argv);
        }
    } else {
        let chunk = nper.unwrap_or(tokens.len().max(1));
        for batch in tokens.chunks(chunk.max(1)) {
            let mut argv = cmd.clone();
            argv.extend(batch.iter().cloned());
            commands.push(argv);
        }
    }
    Ok(commands)
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

fn start_buffered_text(
    interp: &mut CommandContext<'_>,
    args: &[String],
    io: &mut Io,
) -> CommandPoll {
    let command = interp.command_name().to_string();
    let status = match command.as_str() {
        "fold" => cmd_fold(interp, args, io),
        "fmt" => cmd_fmt(interp, args, io),
        "expand" => cmd_expand(interp, args, io),
        "unexpand" => cmd_unexpand(interp, args, io),
        "column" => cmd_column(interp, args, io),
        "join" => cmd_join(interp, args, io),
        "split" => cmd_split(interp, args, io),
        "shuf" => cmd_shuf(interp, args, io),
        "tsort" => cmd_tsort(interp, args, io),
        _ => unreachable!("registered buffered text command"),
    };
    CommandPoll::Ready(status)
}

fn parse_width<'a>(
    command: &str,
    args: &'a [String],
    default: usize,
) -> Result<(usize, Vec<&'a String>), String> {
    let mut width = default;
    let mut files = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "-w" || argument == "--width" {
            index += 1;
            let value = args
                .get(index)
                .ok_or_else(|| format!("{command}: option requires an argument"))?;
            width = value
                .parse()
                .map_err(|_| format!("{command}: invalid width: {value}"))?;
        } else if let Some(value) = argument.strip_prefix("--width=") {
            width = value
                .parse()
                .map_err(|_| format!("{command}: invalid width: {value}"))?;
        } else if argument.starts_with('-') && argument != "-" {
            return Err(format!("{command}: unsupported option '{argument}'"));
        } else {
            files.push(argument);
        }
        index += 1;
    }
    if width == 0 {
        return Err(format!("{command}: width must be positive"));
    }
    Ok((width, files))
}

fn cmd_fold(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (width, files) = match parse_width("fold", args, 80) {
        Ok(value) => value,
        Err(error) => {
            ewln(io.err, &error);
            return 1;
        }
    };
    let (data, errors) = read_inputs(interp, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("fold: {error}"));
        return 1;
    }
    for line in String::from_utf8_lossy(&data).split_inclusive('\n') {
        let (line, terminated) = line
            .strip_suffix('\n')
            .map_or((line, false), |value| (value, true));
        let characters = line.chars().collect::<Vec<_>>();
        if characters.is_empty() && terminated {
            io.out.push(b'\n');
            continue;
        }
        for chunk in characters.chunks(width) {
            w(io.out, &chunk.iter().collect::<String>());
            io.out.push(b'\n');
        }
        if !terminated && !characters.is_empty() {
            io.out.pop();
        }
    }
    0
}

fn cmd_fmt(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (width, files) = match parse_width("fmt", args, 75) {
        Ok(value) => value,
        Err(error) => {
            ewln(io.err, &error);
            return 1;
        }
    };
    let (data, errors) = read_inputs(interp, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("fmt: {error}"));
        return 1;
    }
    let text = String::from_utf8_lossy(&data);
    let paragraphs = text.split("\n\n").collect::<Vec<_>>();
    for (paragraph_index, paragraph) in paragraphs.iter().enumerate() {
        let mut column = 0usize;
        for word in paragraph.split_whitespace() {
            if column == 0 {
                w(io.out, word);
                column = word.chars().count();
            } else if column.saturating_add(1 + word.chars().count()) <= width {
                io.out.push(b' ');
                w(io.out, word);
                column += 1 + word.chars().count();
            } else {
                io.out.push(b'\n');
                w(io.out, word);
                column = word.chars().count();
            }
        }
        io.out.push(b'\n');
        if paragraph_index + 1 < paragraphs.len() {
            io.out.push(b'\n');
        }
    }
    0
}

fn parse_tab_stop<'a>(
    command: &str,
    args: &'a [String],
) -> Result<(usize, bool, Vec<&'a String>), String> {
    let mut stop = 8usize;
    let mut all = false;
    let mut files = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-a" | "--all" => all = true,
            "-t" | "--tabs" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| format!("{command}: option requires an argument"))?;
                stop = value
                    .parse()
                    .map_err(|_| format!("{command}: invalid tab size: {value}"))?;
            }
            value if value.starts_with("--tabs=") => {
                stop = value[7..]
                    .parse()
                    .map_err(|_| format!("{command}: invalid tab size"))?
            }
            value if value.starts_with('-') && value != "-" => {
                return Err(format!("{command}: unsupported option '{value}'"))
            }
            _ => files.push(&args[index]),
        }
        index += 1;
    }
    if stop == 0 {
        return Err(format!("{command}: tab size must be positive"));
    }
    Ok((stop, all, files))
}

fn cmd_expand(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (stop, _, files) = match parse_tab_stop("expand", args) {
        Ok(v) => v,
        Err(e) => {
            ewln(io.err, &e);
            return 1;
        }
    };
    let (data, errors) = read_inputs(interp, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("expand: {error}"));
        return 1;
    }
    let mut column = 0usize;
    for character in String::from_utf8_lossy(&data).chars() {
        match character {
            '\t' => {
                let count = stop - column % stop;
                io.out.extend(std::iter::repeat_n(b' ', count));
                column += count;
            }
            '\n' => {
                io.out.push(b'\n');
                column = 0;
            }
            value => {
                w(io.out, &value.to_string());
                column += 1;
            }
        }
    }
    0
}

fn cmd_unexpand(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (stop, all, files) = match parse_tab_stop("unexpand", args) {
        Ok(v) => v,
        Err(e) => {
            ewln(io.err, &e);
            return 1;
        }
    };
    let (data, errors) = read_inputs(interp, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("unexpand: {error}"));
        return 1;
    }
    for line in String::from_utf8_lossy(&data).split_inclusive('\n') {
        let mut column = 0usize;
        let mut spaces = 0usize;
        let mut leading = true;
        for character in line.chars() {
            if character == ' ' && (all || leading) {
                spaces += 1;
                column += 1;
                if column.is_multiple_of(stop) {
                    io.out.push(b'\t');
                    spaces = 0;
                }
            } else {
                io.out.extend(std::iter::repeat_n(b' ', spaces));
                spaces = 0;
                w(io.out, &character.to_string());
                if character == '\n' {
                    column = 0;
                    leading = true;
                } else {
                    column += 1;
                    leading = false;
                }
            }
        }
        io.out.extend(std::iter::repeat_n(b' ', spaces));
    }
    0
}

fn cmd_column(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut separator = None;
    let mut table = false;
    let mut files = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-t" | "--table" => table = true,
            "-s" | "--separator" => {
                index += 1;
                separator = args.get(index).and_then(|v| v.chars().next());
                if separator.is_none() {
                    ewln(io.err, "column: missing separator");
                    return 1;
                }
            }
            value if value.starts_with('-') && value != "-" => {
                ewln(io.err, &format!("column: unsupported option '{value}'"));
                return 1;
            }
            _ => files.push(&args[index]),
        }
        index += 1;
    }
    let (data, errors) = read_inputs(interp, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("column: {error}"));
        return 1;
    }
    if !table {
        io.out.extend_from_slice(&data);
        return 0;
    }
    let rows = String::from_utf8_lossy(&data)
        .lines()
        .map(|line| {
            separator.map_or_else(
                || line.split_whitespace().map(str::to_string).collect(),
                |sep| line.split(sep).map(str::to_string).collect(),
            )
        })
        .collect::<Vec<Vec<String>>>();
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    let widths = (0..columns)
        .map(|column| {
            rows.iter()
                .filter_map(|row| row.get(column))
                .map(|cell| cell.chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            w(io.out, cell);
            if index + 1 < row.len() {
                w(
                    io.out,
                    &" ".repeat(widths[index].saturating_sub(cell.chars().count()) + 2),
                );
            }
        }
        io.out.push(b'\n');
    }
    0
}

fn cmd_comm(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    if ops.len() < 2 {
        ewln(io.err, "comm: missing operand");
        return 1;
    }
    let a = interp
        .vfs
        .read(&interp.cwd, ops[0])
        .map(|d| lines_of(&d))
        .unwrap_or_default();
    let b = interp
        .vfs
        .read(&interp.cwd, ops[1])
        .map(|d| lines_of(&d))
        .unwrap_or_default();
    let (s1, s2, s3) = (
        !flags.contains(&'1'),
        !flags.contains(&'2'),
        !flags.contains(&'3'),
    );
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        if i < a.len() && (j >= b.len() || a[i] < b[j]) {
            if s1 {
                wln(io.out, &a[i]);
            }
            i += 1;
        } else if j < b.len() && (i >= a.len() || b[j] < a[i]) {
            if s2 {
                wln(io.out, &format!("\t{}", b[j]));
            }
            j += 1;
        } else {
            if s3 {
                wln(io.out, &format!("\t\t{}", a[i]));
            }
            i += 1;
            j += 1;
        }
    }
    0
}

fn cmd_join(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut separator = None;
    let mut field1 = 0usize;
    let mut field2 = 0usize;
    let mut include_unpaired = [false, false];
    let mut files = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-t" => {
                index += 1;
                separator = args.get(index).and_then(|value| value.chars().next());
                if separator.is_none() {
                    ewln(io.err, "join: missing delimiter");
                    return 1;
                }
            }
            "-1" | "-2" => {
                let side = usize::from(args[index] == "-2");
                index += 1;
                let Some(value) = args
                    .get(index)
                    .and_then(|v| v.parse::<usize>().ok())
                    .and_then(|v| v.checked_sub(1))
                else {
                    ewln(io.err, "join: invalid field number");
                    return 1;
                };
                if side == 0 {
                    field1 = value;
                } else {
                    field2 = value;
                }
            }
            "-a" => {
                index += 1;
                match args.get(index).map(String::as_str) {
                    Some("1") => include_unpaired[0] = true,
                    Some("2") => include_unpaired[1] = true,
                    _ => {
                        ewln(io.err, "join: invalid file number");
                        return 1;
                    }
                }
            }
            value if value.starts_with('-') && value != "-" => {
                ewln(io.err, &format!("join: unsupported option '{value}'"));
                return 1;
            }
            _ => files.push(&args[index]),
        }
        index += 1;
    }
    if files.len() != 2 {
        ewln(io.err, "join: expected two files");
        return 1;
    }
    let mut inputs = Vec::new();
    for file in files {
        let data = if file == "-" {
            io.stdin.clone()
        } else {
            match interp.fs_read(&interp.cwd, file) {
                Ok(v) => v,
                Err(e) => {
                    ewln(io.err, &format!("join: {file}: {e}"));
                    return 1;
                }
            }
        };
        inputs.push(
            lines_of(&data)
                .into_iter()
                .map(|line| {
                    separator.map_or_else(
                        || line.split_whitespace().map(str::to_string).collect(),
                        |sep| line.split(sep).map(str::to_string).collect(),
                    )
                })
                .collect::<Vec<Vec<String>>>(),
        );
    }
    let output_separator = separator.unwrap_or(' ').to_string();
    let comparison_work = inputs[0].len().saturating_mul(inputs[1].len());
    let retained_memory = inputs
        .iter()
        .flatten()
        .flatten()
        .map(String::len)
        .sum::<usize>()
        .saturating_add(inputs[1].len());
    if !interp.charge_cpu(u64::try_from(comparison_work).unwrap_or(u64::MAX)) {
        return 137;
    }
    let retained_memory = u64::try_from(retained_memory).unwrap_or(u64::MAX);
    if !interp.reserve_memory(retained_memory) {
        return 137;
    }
    let mut matched2 = vec![false; inputs[1].len()];
    for row1 in &inputs[0] {
        let key = row1.get(field1);
        let mut matched = false;
        for (right_index, row2) in inputs[1].iter().enumerate() {
            if key.is_some() && key == row2.get(field2) {
                matched = true;
                matched2[right_index] = true;
                let mut output = vec![key.cloned().unwrap_or_default()];
                output.extend(
                    row1.iter()
                        .enumerate()
                        .filter(|(i, _)| *i != field1)
                        .map(|(_, v)| v.clone()),
                );
                output.extend(
                    row2.iter()
                        .enumerate()
                        .filter(|(i, _)| *i != field2)
                        .map(|(_, v)| v.clone()),
                );
                wln(io.out, &output.join(&output_separator));
            }
        }
        if !matched && include_unpaired[0] {
            wln(io.out, &row1.join(&output_separator));
        }
    }
    if include_unpaired[1] {
        for (index, row) in inputs[1].iter().enumerate() {
            if !matched2[index] {
                wln(io.out, &row.join(&output_separator));
            }
        }
    }
    interp.resources.release_memory(retained_memory);
    0
}

fn cmd_split(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut line_count = 1000usize;
    let mut byte_count = None;
    let mut operands = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-l" => {
                index += 1;
                line_count = match args.get(index).and_then(|v| v.parse().ok()) {
                    Some(0) | None => {
                        ewln(io.err, "split: invalid line count");
                        return 1;
                    }
                    Some(v) => v,
                };
            }
            "-b" => {
                index += 1;
                byte_count = match args.get(index).and_then(|v| v.parse::<usize>().ok()) {
                    Some(0) | None => {
                        ewln(io.err, "split: invalid byte count");
                        return 1;
                    }
                    value => value,
                };
            }
            value if value.starts_with('-') && value != "-" => {
                ewln(io.err, &format!("split: unsupported option '{value}'"));
                return 1;
            }
            _ => operands.push(&args[index]),
        }
        index += 1;
    }
    if operands.len() > 2 {
        ewln(io.err, "split: extra operand");
        return 1;
    }
    let data = match operands.first().map(|v| v.as_str()) {
        None | Some("-") => io.stdin.clone(),
        Some(file) => match interp.fs_read(&interp.cwd, file) {
            Ok(v) => v,
            Err(e) => {
                ewln(io.err, &format!("split: {file}: {e}"));
                return 1;
            }
        },
    };
    let prefix = operands.get(1).map_or("x", |value| value.as_str());
    let data_len = data.len();
    if !interp.charge_cpu(u64::try_from(data_len).unwrap_or(u64::MAX)) {
        return 137;
    }
    let chunk_memory = data_len.saturating_mul(2).saturating_add(
        data_len
            .min(26 * 26)
            .saturating_mul(std::mem::size_of::<Vec<u8>>()),
    );
    let chunk_memory = u64::try_from(chunk_memory).unwrap_or(u64::MAX);
    if !interp.reserve_memory(chunk_memory) {
        return 137;
    }
    let chunks = if let Some(bytes) = byte_count {
        data.chunks(bytes).map(<[u8]>::to_vec).collect::<Vec<_>>()
    } else {
        let mut chunks = Vec::new();
        let mut current = Vec::new();
        let mut lines = 0;
        for byte in data {
            current.push(byte);
            if byte == b'\n' {
                lines += 1;
            }
            if lines == line_count {
                chunks.push(std::mem::take(&mut current));
                lines = 0;
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        chunks
    };
    for (number, chunk) in chunks.iter().enumerate() {
        if number >= 26 * 26 {
            ewln(io.err, "split: output file suffixes exhausted");
            interp.resources.release_memory(chunk_memory);
            return 1;
        }
        let name = format!(
            "{prefix}{}{}",
            char::from(b'a' + (number / 26) as u8),
            char::from(b'a' + (number % 26) as u8)
        );
        let cwd = interp.cwd.clone();
        let mode = 0o666 & !u32::from(interp.umask);
        if let Err(error) = interp.vfs.write(&cwd, &name, chunk, mode) {
            ewln(io.err, &format!("split: {name}: {error}"));
            interp.resources.release_memory(chunk_memory);
            return 1;
        }
    }
    interp.resources.release_memory(chunk_memory);
    0
}

fn cmd_shuf(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut count = None;
    let mut files = Vec::new();
    let mut index = 0;
    while index < args.len() {
        if args[index] == "-n" || args[index] == "--head-count" {
            index += 1;
            count = args.get(index).and_then(|v| v.parse::<usize>().ok());
            if count.is_none() {
                ewln(io.err, "shuf: invalid line count");
                return 1;
            }
        } else if args[index].starts_with('-') && args[index] != "-" {
            ewln(
                io.err,
                &format!("shuf: unsupported option '{}'", args[index]),
            );
            return 1;
        } else {
            files.push(&args[index]);
        }
        index += 1;
    }
    if files.len() > 1 {
        ewln(io.err, "shuf: extra operand");
        return 1;
    }
    let (data, errors) = read_inputs(interp, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("shuf: {error}"));
        return 1;
    }
    let mut lines = lines_of(&data);
    // A fixed generator makes simulations reproducible while retaining permutation semantics.
    let mut state = 0x9e37_79b9_u32;
    for end in (1..lines.len()).rev() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let other = state as usize % (end + 1);
        lines.swap(end, other);
    }
    let take = count.unwrap_or(lines.len()).min(lines.len());
    for line in &lines[..take] {
        wln(io.out, line);
    }
    0
}

fn cmd_tsort(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.len() > 1 {
        ewln(io.err, "tsort: extra operand");
        return 1;
    }
    let files = args.iter().collect::<Vec<_>>();
    let (data, errors) = read_inputs(interp, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("tsort: {error}"));
        return 1;
    }
    let words = String::from_utf8_lossy(&data)
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if words.len() % 2 != 0 {
        ewln(io.err, "tsort: input contains an odd number of tokens");
        return 1;
    }
    let mut edges = std::collections::BTreeMap::<String, std::collections::BTreeSet<String>>::new();
    let mut indegree = std::collections::BTreeMap::<String, usize>::new();
    for pair in words.chunks(2) {
        indegree.entry(pair[0].clone()).or_default();
        indegree.entry(pair[1].clone()).or_default();
        if pair[0] != pair[1]
            && edges
                .entry(pair[0].clone())
                .or_default()
                .insert(pair[1].clone())
        {
            *indegree.entry(pair[1].clone()).or_default() += 1;
        }
    }
    let mut ready = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(node, _)| node.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let mut emitted = 0;
    while let Some(node) = ready.pop_first() {
        wln(io.out, &node);
        emitted += 1;
        if let Some(next) = edges.get(&node) {
            for target in next {
                let degree = indegree.get_mut(target).expect("known node");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(target.clone());
                }
            }
        }
    }
    if emitted != indegree.len() {
        ewln(io.err, "tsort: input contains a loop");
        return 1;
    }
    0
}

fn cmd_diff(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut unified = false;
    let mut recursive = false;
    let mut brief = false;
    let mut absent_empty = false;
    let mut whitespace = DiffWhitespace::Exact;
    let mut operands = Vec::new();
    for argument in args {
        match argument.as_str() {
            "-u" | "--unified" => unified = true,
            "-r" | "--recursive" => recursive = true,
            "-q" | "--brief" => brief = true,
            "-N" | "--new-file" => absent_empty = true,
            "-w" | "--ignore-all-space" => whitespace = DiffWhitespace::All,
            "-b" | "--ignore-space-change" => {
                if whitespace != DiffWhitespace::All {
                    whitespace = DiffWhitespace::Change;
                }
            }
            "--" => {}
            value
                if value.starts_with('-')
                    && value.len() > 2
                    && value[1..]
                        .chars()
                        .all(|flag| matches!(flag, 'u' | 'r' | 'q' | 'N' | 'w' | 'b')) =>
            {
                for flag in value[1..].chars() {
                    match flag {
                        'u' => unified = true,
                        'r' => recursive = true,
                        'q' => brief = true,
                        'N' => absent_empty = true,
                        'w' => whitespace = DiffWhitespace::All,
                        'b' if whitespace != DiffWhitespace::All => {
                            whitespace = DiffWhitespace::Change;
                        }
                        'b' => {}
                        _ => unreachable!("guarded option flag"),
                    }
                }
            }
            value if value.starts_with('-') => {
                ewln(io.err, &format!("diff: unsupported option '{value}'"));
                return 2;
            }
            _ => operands.push(argument),
        }
    }
    if operands.len() != 2 {
        ewln(io.err, "diff: missing operand");
        return 2;
    }
    let options = DiffOptions {
        unified,
        brief,
        absent_empty,
        whitespace,
    };
    let left_meta = interp.fs_metadata(&interp.cwd, operands[0], true);
    let right_meta = interp.fs_metadata(&interp.cwd, operands[1], true);
    let directories = matches!(
        left_meta.as_ref().map(|n| &n.kind),
        Ok(crate::vfs::NodeKind::Dir)
    ) || matches!(
        right_meta.as_ref().map(|n| &n.kind),
        Ok(crate::vfs::NodeKind::Dir)
    );
    if directories {
        if !recursive {
            ewln(io.err, "diff: directory comparison requires -r");
            return 2;
        }
        return diff_directories(interp, operands[0], operands[1], &options, io);
    }
    diff_files(interp, operands[0], operands[1], &options, io)
}

struct DiffOptions {
    unified: bool,
    brief: bool,
    absent_empty: bool,
    whitespace: DiffWhitespace,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DiffWhitespace {
    Exact,
    Change,
    All,
}

#[derive(Clone, Copy)]
enum DiffLine<'a> {
    Same(&'a str),
    Remove(&'a str),
    Add(&'a str),
}

fn diff_files(
    interp: &mut CommandContext<'_>,
    left: &str,
    right: &str,
    options: &DiffOptions,
    io: &mut Io,
) -> i32 {
    let read = |interp: &CommandContext<'_>, path: &str| interp.fs_read(&interp.cwd, path);
    let left_data = match read(interp, left) {
        Ok(v) => v,
        Err(_) if options.absent_empty => Vec::new(),
        Err(e) => {
            ewln(io.err, &format!("diff: {left}: {e}"));
            return 2;
        }
    };
    let right_data = match read(interp, right) {
        Ok(v) => v,
        Err(_) if options.absent_empty => Vec::new(),
        Err(e) => {
            ewln(io.err, &format!("diff: {right}: {e}"));
            return 2;
        }
    };
    if left_data == right_data {
        return 0;
    }
    let left_text = String::from_utf8_lossy(&left_data);
    let right_text = String::from_utf8_lossy(&right_data);
    let left_lines = left_text.lines().collect::<Vec<_>>();
    let right_lines = right_text.lines().collect::<Vec<_>>();
    let left_keys = left_lines
        .iter()
        .map(|line| diff_line_key(line, options.whitespace))
        .collect::<Vec<_>>();
    let right_keys = right_lines
        .iter()
        .map(|line| diff_line_key(line, options.whitespace))
        .collect::<Vec<_>>();
    if left_keys == right_keys {
        return 0;
    }
    if options.brief {
        wln(io.out, &format!("Files {left} and {right} differ"));
        return 1;
    }
    let cells = left_lines
        .len()
        .saturating_add(1)
        .saturating_mul(right_lines.len().saturating_add(1));
    if cells > 1_000_000 {
        ewln(io.err, "diff: inputs are too large for line comparison");
        return 2;
    }
    if !interp.charge_cpu(u64::try_from(cells).unwrap_or(u64::MAX)) {
        return 137;
    }
    let matrix_memory = u64::try_from(cells)
        .unwrap_or(u64::MAX)
        .saturating_mul(std::mem::size_of::<usize>() as u64);
    if !interp.reserve_memory(matrix_memory) {
        return 137;
    }
    let edits = lcs_edits(&left_lines, &right_lines, &left_keys, &right_keys);
    interp.resources.release_memory(matrix_memory);
    if options.unified {
        wln(io.out, &format!("--- {left}"));
        wln(io.out, &format!("+++ {right}"));
        wln(
            io.out,
            &format!("@@ -1,{} +1,{} @@", left_lines.len(), right_lines.len()),
        );
        for edit in edits {
            match edit {
                DiffLine::Same(line) => wln(io.out, &format!(" {line}")),
                DiffLine::Remove(line) => wln(io.out, &format!("-{line}")),
                DiffLine::Add(line) => wln(io.out, &format!("+{line}")),
            }
        }
    } else {
        for edit in edits {
            match edit {
                DiffLine::Same(_) => {}
                DiffLine::Remove(line) => wln(io.out, &format!("< {line}")),
                DiffLine::Add(line) => wln(io.out, &format!("> {line}")),
            }
        }
    }
    1
}

fn diff_line_key(line: &str, whitespace: DiffWhitespace) -> String {
    match whitespace {
        DiffWhitespace::Exact => line.to_string(),
        DiffWhitespace::All => line
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect(),
        DiffWhitespace::Change => {
            let mut key = String::with_capacity(line.len());
            let mut in_whitespace = false;
            for character in line.chars() {
                if character.is_whitespace() {
                    in_whitespace = true;
                } else {
                    if in_whitespace && !key.is_empty() {
                        key.push(' ');
                    }
                    key.push(character);
                    in_whitespace = false;
                }
            }
            key
        }
    }
}

fn lcs_edits<'a>(
    left: &[&'a str],
    right: &[&'a str],
    left_keys: &[String],
    right_keys: &[String],
) -> Vec<DiffLine<'a>> {
    let width = right.len() + 1;
    let mut lengths = vec![0usize; (left.len() + 1) * width];
    for i in (0..left.len()).rev() {
        for j in (0..right.len()).rev() {
            lengths[i * width + j] = if left_keys[i] == right_keys[j] {
                lengths[(i + 1) * width + j + 1] + 1
            } else {
                lengths[(i + 1) * width + j].max(lengths[i * width + j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut edits = Vec::new();
    while i < left.len() || j < right.len() {
        if i < left.len() && j < right.len() && left_keys[i] == right_keys[j] {
            edits.push(DiffLine::Same(left[i]));
            i += 1;
            j += 1;
        } else if j < right.len()
            && (i == left.len() || lengths[i * width + j + 1] > lengths[(i + 1) * width + j])
        {
            edits.push(DiffLine::Add(right[j]));
            j += 1;
        } else {
            edits.push(DiffLine::Remove(left[i]));
            i += 1;
        }
    }
    edits
}

fn diff_directories(
    interp: &mut CommandContext<'_>,
    left: &str,
    right: &str,
    options: &DiffOptions,
    io: &mut Io,
) -> i32 {
    let left_abs = crate::vfs::resolve_against(&interp.cwd, left);
    let right_abs = crate::vfs::resolve_against(&interp.cwd, right);
    let relative_entries =
        |interp: &CommandContext<'_>, root: &str| -> std::collections::BTreeMap<String, bool> {
            interp
                .fs_walk("/", root)
                .unwrap_or_default()
                .into_iter()
                .filter_map(|path| {
                    let relative = path.strip_prefix(root)?.trim_start_matches('/');
                    if relative.is_empty() {
                        return None;
                    }
                    let directory = matches!(
                        interp.fs_metadata("/", &path, true).map(|node| node.kind),
                        Ok(crate::vfs::NodeKind::Dir)
                    );
                    Some((relative.to_string(), directory))
                })
                .collect()
        };
    let left_entries = relative_entries(interp, &left_abs);
    let right_entries = relative_entries(interp, &right_abs);
    let names = left_entries
        .keys()
        .chain(right_entries.keys())
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let mut status = 0;
    let mut omitted_directories = Vec::<String>::new();
    for name in names {
        if omitted_directories
            .iter()
            .any(|directory| name.starts_with(&format!("{directory}/")))
        {
            continue;
        }
        let left_kind = left_entries.get(&name);
        let right_kind = right_entries.get(&name);
        if left_kind.is_none() || right_kind.is_none() {
            if options.absent_empty && (left_kind == Some(&false) || right_kind == Some(&false)) {
                // Missing regular files are compared with an empty file below.
            } else {
                let (side, directory) = if let Some(directory) = left_kind {
                    (left, *directory)
                } else {
                    (right, *right_kind.expect("entry exists on one side"))
                };
                print_only_in(io.out, side, &name);
                if directory {
                    omitted_directories.push(name);
                }
                status = 1;
                continue;
            }
        }
        if let (Some(left_directory), Some(right_directory)) = (left_kind, right_kind) {
            if left_directory != right_directory {
                wln(
                    io.out,
                    &format!(
                        "File {}/{} is a {} while file {}/{} is a {}",
                        left.trim_end_matches('/'),
                        name,
                        if *left_directory {
                            "directory"
                        } else {
                            "regular file"
                        },
                        right.trim_end_matches('/'),
                        name,
                        if *right_directory {
                            "directory"
                        } else {
                            "regular file"
                        }
                    ),
                );
                status = 1;
                continue;
            }
            if *left_directory {
                continue;
            }
        }
        if left_kind == Some(&true) || right_kind == Some(&true) {
            status = 1;
            continue;
        }
        let left_path = format!("{}/{name}", left.trim_end_matches('/'));
        let right_path = format!("{}/{name}", right.trim_end_matches('/'));
        let file_status = diff_files(interp, &left_path, &right_path, options, io);
        if file_status == 2 || file_status == 137 {
            return file_status;
        }
        status = status.max(file_status);
    }
    status
}

fn print_only_in(output: &mut Vec<u8>, root: &str, relative: &str) {
    let (parent, basename) = relative.rsplit_once('/').map_or(
        (root.trim_end_matches('/').to_string(), relative),
        |(parent, basename)| (format!("{}/{parent}", root.trim_end_matches('/')), basename),
    );
    wln(output, &format!("Only in {parent}: {basename}"));
}

fn cmd_cmp(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    if ops.len() < 2 {
        return 2;
    }
    let a = match interp.fs_read(&interp.cwd, ops[0]) {
        Ok(data) => data,
        Err(error) => {
            ewln(io.err, &format!("cmp: {}: {error}", ops[0]));
            return 2;
        }
    };
    let b = match interp.fs_read(&interp.cwd, ops[1]) {
        Ok(data) => data,
        Err(error) => {
            ewln(io.err, &format!("cmp: {}: {error}", ops[1]));
            return 2;
        }
    };
    if a == b {
        0
    } else {
        ewln(io.err, &format!("{} {} differ", ops[0], ops[1]));
        1
    }
}

fn cmd_grep(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    grep_impl(interp, "grep", args, io)
}

fn cmd_egrep(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    grep_impl(interp, "egrep", args, io)
}

fn cmd_fgrep(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    grep_impl(interp, "fgrep", args, io)
}

/// grep with the command name available (egrep/fgrep change default regex flavor).
fn grep_impl(interp: &mut Interp, cmd: &str, args: &[String], io: &mut Io) -> i32 {
    let memory_mark = interp.resources.memory_mark();
    let status = grep_impl_inner(interp, cmd, args, io);
    interp.resources.restore_memory(memory_mark);
    status
}

fn grep_impl_inner(interp: &mut Interp, cmd: &str, args: &[String], io: &mut Io) -> i32 {
    #[derive(Clone, Copy, PartialEq)]
    enum Key {
        IgnoreCase,
        Invert,
        Count,
        LineNumber,
        FilesWith,
        FilesWithout,
        OnlyMatch,
        Recursive,
        Extended,
        Fixed,
        Word,
        Line,
        Quiet,
        NoFilename,
        WithFilename,
        SuppressErrors,
        Text,
        Regexp,
        PatternFile,
        MaxCount,
        Include,
        Exclude,
        ExcludeDir,
        BeforeContext,
        AfterContext,
        Context,
        Color,
        Help,
    }
    const OPTIONS: &[OptionSpec<Key>] = &[
        OptionSpec::flag(Key::IgnoreCase, Some('i'), Some("ignore-case")),
        OptionSpec::flag(Key::Invert, Some('v'), Some("invert-match")),
        OptionSpec::flag(Key::Count, Some('c'), Some("count")),
        OptionSpec::flag(Key::LineNumber, Some('n'), Some("line-number")),
        OptionSpec::flag(Key::FilesWith, Some('l'), Some("files-with-matches")),
        OptionSpec::flag(Key::FilesWithout, Some('L'), Some("files-without-match")),
        OptionSpec::flag(Key::OnlyMatch, Some('o'), Some("only-matching")),
        OptionSpec::flag(Key::Recursive, Some('r'), Some("recursive")),
        OptionSpec::flag(Key::Recursive, Some('R'), Some("dereference-recursive")),
        OptionSpec::flag(Key::Extended, Some('E'), Some("extended-regexp")),
        OptionSpec::flag(Key::Fixed, Some('F'), Some("fixed-strings")),
        OptionSpec::flag(Key::Word, Some('w'), Some("word-regexp")),
        OptionSpec::flag(Key::Line, Some('x'), Some("line-regexp")),
        OptionSpec::flag(Key::Quiet, Some('q'), Some("quiet")),
        OptionSpec::flag(Key::Quiet, None, Some("silent")),
        OptionSpec::flag(Key::NoFilename, Some('h'), Some("no-filename")),
        OptionSpec::flag(Key::WithFilename, Some('H'), Some("with-filename")),
        OptionSpec::flag(Key::SuppressErrors, Some('s'), Some("no-messages")),
        OptionSpec::flag(Key::Text, Some('a'), Some("text")),
        OptionSpec::required(Key::Regexp, Some('e'), Some("regexp")),
        OptionSpec::required(Key::PatternFile, Some('f'), Some("file")),
        OptionSpec::required(Key::MaxCount, Some('m'), Some("max-count")),
        OptionSpec::required(Key::Include, None, Some("include")),
        OptionSpec::required(Key::Exclude, None, Some("exclude")),
        OptionSpec::required(Key::ExcludeDir, None, Some("exclude-dir")),
        OptionSpec::required(Key::BeforeContext, Some('B'), Some("before-context")),
        OptionSpec::required(Key::AfterContext, Some('A'), Some("after-context")),
        OptionSpec::required(Key::Context, Some('C'), Some("context")),
        OptionSpec::optional_attached(Key::Color, None, Some("color")),
        OptionSpec::flag(Key::Help, None, Some("help")),
    ];
    let parsed = match parse_options_or_report(
        "grep",
        args,
        OPTIONS,
        (
            Key::Help,
            "usage: grep [OPTIONS] PATTERN [FILE...]\nsupported: -E -F -i -v -c -n -l -L -o -r -w -x -q -h -H -s -e -f -m -A -B -C\n",
        ),
        io.out,
        io.err,
    ) {
        Ok(parsed) => parsed,
        Err(status) => return status,
    };
    let mut ignore_case = false;
    let mut invert = false;
    let mut count = false;
    let mut line_num = false;
    let mut files_with = false;
    let mut files_without = false;
    let mut only_match = false;
    let mut recursive = false;
    let mut extended = cmd == "egrep";
    let mut fixed = cmd == "fgrep";
    let mut word = false;
    let mut line_regexp = false;
    let mut quiet = false;
    let mut suppress_errors = false;
    let mut text_mode = false;
    let mut filename_mode = None;
    let mut max_count = None;
    let mut before_context = 0;
    let mut after_context = 0;
    let mut includes = Vec::new();
    let mut excludes = Vec::new();
    let mut exclude_dirs = Vec::new();
    let mut pattern_files = Vec::new();
    let mut explicit_pattern = false;
    let mut patterns = Vec::new();
    for option in parsed.options {
        match option.key {
            Key::IgnoreCase => ignore_case = true,
            Key::Invert => invert = true,
            Key::Count => count = true,
            Key::LineNumber => line_num = true,
            Key::FilesWith => files_with = true,
            Key::FilesWithout => files_without = true,
            Key::OnlyMatch => only_match = true,
            Key::Recursive => recursive = true,
            Key::Extended => {
                extended = true;
                fixed = false;
            }
            Key::Fixed => fixed = true,
            Key::Word => word = true,
            Key::Line => line_regexp = true,
            Key::Quiet => quiet = true,
            Key::NoFilename => filename_mode = Some(false),
            Key::WithFilename => filename_mode = Some(true),
            Key::SuppressErrors => suppress_errors = true,
            Key::Text => text_mode = true,
            Key::Regexp => {
                explicit_pattern = true;
                patterns.push(option.value.expect("required option value"));
            }
            Key::PatternFile => {
                explicit_pattern = true;
                pattern_files.push(option.value.expect("required option value"));
            }
            Key::MaxCount => {
                let value = option.value.expect("required option value");
                max_count = match value.parse::<usize>() {
                    Ok(value) => Some(value),
                    Err(_) => {
                        ewln(io.err, &format!("grep: invalid max count: {value}"));
                        return 2;
                    }
                };
            }
            Key::Include => includes.push(option.value.expect("required option value")),
            Key::Exclude => excludes.push(option.value.expect("required option value")),
            Key::ExcludeDir => exclude_dirs.push(option.value.expect("required option value")),
            Key::BeforeContext | Key::AfterContext | Key::Context => {
                let value = option.value.expect("required option value");
                let count = match value.parse::<usize>() {
                    Ok(value) => value,
                    Err(_) => {
                        ewln(io.err, &format!("grep: invalid context length: {value}"));
                        return 2;
                    }
                };
                match option.key {
                    Key::BeforeContext => before_context = count,
                    Key::AfterContext => after_context = count,
                    Key::Context => {
                        before_context = count;
                        after_context = count;
                    }
                    _ => unreachable!(),
                }
            }
            Key::Color => match option.value.as_deref().unwrap_or("auto") {
                "never" => {}
                mode => {
                    ewln(io.err, &format!("grep: unsupported color mode '{mode}'"));
                    return 2;
                }
            },
            Key::Help => unreachable!("help is handled by the shared option parser"),
        }
    }
    for file in pattern_files {
        let bytes = match interp.vfs.read(&interp.cwd, &file) {
            Ok(bytes) => bytes,
            Err(error) => {
                ewln(io.err, &format!("grep: {file}: {error}"));
                return 2;
            }
        };
        let bytes_len = bytes.len() as u64;
        if !interp.resources.reserve_memory(bytes_len) || !interp.resources.charge_cpu(bytes_len) {
            return interp
                .resources
                .stop_reason()
                .map_or(137, |reason| reason.exit_status());
        }
        let content = match String::from_utf8(bytes) {
            Ok(content) => content,
            Err(_) => {
                ewln(
                    io.err,
                    &format!("grep: {file}: pattern file is not valid UTF-8"),
                );
                return 2;
            }
        };
        patterns.extend(content.lines().map(str::to_string));
    }
    if (before_context != 0 || after_context != 0)
        && (count || files_with || files_without || only_match || quiet)
    {
        ewln(
            io.err,
            "grep: context cannot be combined with count, file-list, only-match, or quiet modes",
        );
        return 2;
    }
    let mut operands = parsed.operands;
    let files = if !explicit_pattern && !operands.is_empty() {
        patterns.push(operands.remove(0));
        operands
    } else {
        operands
    };
    if patterns.is_empty() && !explicit_pattern {
        ewln(io.err, "grep: no pattern");
        return 2;
    }
    let mut pat_re = if patterns.is_empty() {
        r"[^\s\S]".to_string()
    } else if fixed {
        patterns
            .iter()
            .map(|pattern| regex::escape(pattern))
            .collect::<Vec<_>>()
            .join("|")
    } else {
        let patterns: Result<Vec<String>, String> = if extended {
            Ok(patterns.clone())
        } else {
            patterns
                .iter()
                .map(|pattern| basic_regex_to_rust(pattern))
                .collect::<Result<Vec<_>, _>>()
        };
        let patterns = match patterns {
            Ok(patterns) => patterns,
            Err(error) => {
                ewln(
                    io.err,
                    &format!("grep: unsupported regular expression: {error}"),
                );
                return 2;
            }
        };
        patterns
            .iter()
            .map(|pattern| format!("(?:{pattern})"))
            .collect::<Vec<_>>()
            .join("|")
    };
    if word {
        pat_re = format!(r"\b(?:{pat_re})\b");
    }
    if line_regexp {
        pat_re = format!(r"^(?:{pat_re})$");
    }
    let re = match regex::RegexBuilder::new(&pat_re)
        .case_insensitive(ignore_case)
        .build()
    {
        Ok(r) => r,
        Err(error) => {
            ewln(
                io.err,
                &format!("grep: invalid regular expression: {error}"),
            );
            return 2;
        }
    };

    // gather (label, data)
    let mut inputs: Vec<(String, Vec<u8>)> = Vec::new();
    let mut had_error = false;
    if files.is_empty() {
        inputs.push((String::new(), std::mem::take(&mut io.stdin)));
    } else if recursive {
        for f in &files {
            let abs = crate::vfs::resolve_against(&interp.cwd, f);
            let paths = match interp.fs_walk("/", &abs) {
                Ok(paths) => paths,
                Err(error) => {
                    if !suppress_errors {
                        ewln(io.err, &format!("grep: {error}"));
                    }
                    had_error = true;
                    continue;
                }
            };
            for p in paths {
                if matches!(
                    interp.fs_metadata("/", &p, false).map(|node| node.kind),
                    Ok(crate::vfs::NodeKind::File(_))
                ) {
                    if !grep_path_selected(&p, &includes, &excludes, &exclude_dirs) {
                        continue;
                    }
                    match interp.fs_read("/", &p) {
                        Ok(data) => inputs.push((p.clone(), data)),
                        Err(error) => {
                            if !suppress_errors {
                                ewln(io.err, &format!("grep: {error}"));
                            }
                            had_error = true;
                        }
                    }
                }
            }
        }
    } else {
        for f in &files {
            if !grep_path_selected(f, &includes, &excludes, &[]) {
                continue;
            }
            match interp.fs_read(&interp.cwd, f) {
                Ok(d) => inputs.push((f.clone(), d)),
                Err(error) => {
                    if !suppress_errors {
                        ewln(io.err, &format!("grep: {f}: {error}"));
                    }
                    had_error = true;
                }
            }
        }
    }
    let multi = filename_mode.unwrap_or(inputs.len() > 1 || recursive);
    let mut total_matches = 0;
    let mut selected_file = false;
    for (label, data) in &inputs {
        if !interp.resources.charge_cpu(data.len() as u64) {
            return 137;
        }
        if !text_mode && data.contains(&0) {
            if !suppress_errors {
                ewln(
                    io.err,
                    &format!("grep: {label}: binary input is not supported; use -a"),
                );
            }
            had_error = true;
            continue;
        }
        if parsed_text(data).is_err() && !suppress_errors {
            ewln(io.err, &format!("grep: {label}: input is not valid UTF-8"));
            had_error = true;
            continue;
        }
        let text = match parsed_text(data) {
            Ok(text) => text,
            Err(()) => {
                had_error = true;
                continue;
            }
        };
        if before_context != 0 || after_context != 0 {
            let matched = grep_context(
                &re,
                text,
                label,
                multi,
                line_num,
                invert,
                max_count,
                before_context,
                after_context,
                io.out,
            );
            total_matches += matched;
            selected_file |= matched != 0;
            continue;
        }
        let mut file_count = 0;
        let mut matched_file = false;
        for (lineno, line) in text.lines().enumerate() {
            if max_count == Some(0) {
                break;
            }
            let is_match = re.is_match(line) ^ invert;
            if is_match {
                matched_file = true;
                file_count += 1;
                total_matches += 1;
                if quiet {
                    return 0;
                }
                if count || files_with || files_without {
                    if files_with || max_count.is_some_and(|limit| file_count >= limit) {
                        break;
                    }
                    continue;
                }
                let mut prefix = String::new();
                if multi && !label.is_empty() {
                    prefix.push_str(label);
                    prefix.push(':');
                }
                if line_num {
                    prefix.push_str(&format!("{}:", lineno + 1));
                }
                if only_match && !invert {
                    for m in re.find_iter(line) {
                        wln(io.out, &format!("{prefix}{}", m.as_str()));
                    }
                } else {
                    wln(io.out, &format!("{prefix}{line}"));
                }
                if max_count.is_some_and(|limit| file_count >= limit) {
                    break;
                }
            }
        }
        if count {
            if multi && !label.is_empty() {
                wln(io.out, &format!("{label}:{file_count}"));
            } else {
                wln(io.out, &file_count.to_string());
            }
        }
        if files_with && matched_file {
            wln(io.out, label);
            selected_file = true;
        }
        if files_without && !matched_file {
            wln(io.out, label);
            selected_file = true;
        }
        if !files_with && !files_without {
            selected_file |= matched_file;
        }
    }
    if had_error {
        2
    } else if if files_without {
        selected_file
    } else {
        total_matches > 0 || selected_file
    } {
        0
    } else {
        1
    }
}

fn parsed_text(data: &[u8]) -> Result<&str, ()> {
    std::str::from_utf8(data).map_err(|_| ())
}

fn grep_path_selected(
    path: &str,
    includes: &[String],
    excludes: &[String],
    exclude_dirs: &[String],
) -> bool {
    let basename = crate::vfs::basename(path);
    if !includes.is_empty()
        && !includes
            .iter()
            .any(|pattern| glob_eq(pattern, path) || glob_eq(pattern, basename))
    {
        return false;
    }
    if excludes
        .iter()
        .any(|pattern| glob_eq(pattern, path) || glob_eq(pattern, basename))
    {
        return false;
    }
    !path.split('/').any(|component| {
        exclude_dirs
            .iter()
            .any(|pattern| glob_eq(pattern, component))
    })
}

#[allow(clippy::too_many_arguments)]
fn grep_context(
    regex: &regex::Regex,
    text: &str,
    label: &str,
    show_filename: bool,
    show_line_number: bool,
    invert: bool,
    max_count: Option<usize>,
    before: usize,
    after: usize,
    output: &mut Vec<u8>,
) -> usize {
    let lines = text.lines().collect::<Vec<_>>();
    let mut matches = Vec::with_capacity(lines.len());
    let mut matched_count = 0usize;
    for line in &lines {
        let matched = (regex.is_match(line) ^ invert)
            && max_count.is_none_or(|maximum| matched_count < maximum);
        matched_count = matched_count.saturating_add(usize::from(matched));
        matches.push(matched);
    }
    let mut last_emitted = None;
    for (matched_index, _) in matches.iter().enumerate().filter(|(_, matched)| **matched) {
        let start = matched_index.saturating_sub(before);
        let end = matched_index
            .saturating_add(after)
            .saturating_add(1)
            .min(lines.len());
        let first = last_emitted.map_or(start, |previous| start.max(previous + 1));
        if first >= end {
            continue;
        }
        if last_emitted.is_some_and(|previous| first > previous + 1) {
            wln(output, "--");
        }
        for index in first..end {
            let separator = if matches[index] { ':' } else { '-' };
            let mut prefix = String::new();
            if show_filename && !label.is_empty() {
                prefix.push_str(label);
                prefix.push(separator);
            }
            if show_line_number {
                prefix.push_str(&(index + 1).to_string());
                prefix.push(separator);
            }
            wln(output, &format!("{prefix}{}", lines[index]));
        }
        last_emitted = Some(end - 1);
    }
    matched_count
}

fn cmd_expr(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    // minimal: arithmetic and string length
    if args.len() == 2 && args[0] == "length" {
        wln(io.out, &args[1].chars().count().to_string());
        return 0;
    }
    let joined = args.join(" ");
    // try arithmetic
    let mut i = Interp::new();
    let v = crate::expand::eval_arith(&mut i, &joined);
    wln(io.out, &v.to_string());
    if v == 0 {
        1
    } else {
        0
    }
}

fn cmd_bc(interp: &mut CommandContext<'_>, _args: &[String], io: &mut Io) -> i32 {
    for line in String::from_utf8_lossy(&io.stdin).lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v = crate::expand::eval_arith(interp, line);
        wln(io.out, &v.to_string());
    }
    0
}
