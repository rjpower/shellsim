//! Text processing: output (echo/printf/cat/tac/tee/yes), windowing (head/tail), counting
//! and reshaping (wc/sort/uniq/cut/tr/rev/nl/seq/paste/comm/diff/cmp), pattern tools
//! (grep/sed), the fold/fmt passthroughs, xargs, and the small arithmetic helpers expr/bc.

use std::collections::{HashMap, VecDeque};

use crate::commands::util::{ewln, lines_of, read_inputs, split_flags, w, wln};
use crate::commands::{ChildCommand, CommandContext, CommandPoll, CommandSpec, Io, Trust};
use crate::descriptors::{DeviceStream, IoPoll, IoWait, DEVICE_READ_QUANTUM};
use crate::interp::Interp;
use crate::scheduler::WaitReason;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::{reg, reg_buffered_resumable, reg_resumable};
    reg_resumable(m, &["cat"], Trust::Real, cmd_cat, start_cat);
    reg(m, &["tac"], Trust::Real, cmd_tac);
    reg(m, &["tee"], Trust::Real, cmd_tee);
    reg(m, &["yes"], Trust::Real, cmd_yes);
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
    reg(m, &["sed"], Trust::Partial, cmd_sed);
    reg(m, &["nl"], Trust::Real, cmd_nl);
    reg(m, &["seq"], Trust::Real, cmd_seq);
    reg(m, &["paste"], Trust::Real, cmd_paste);
    reg(
        m,
        &["fold", "fmt", "expand", "unexpand", "column", "pr"],
        Trust::Partial,
        cmd_passthrough,
    );
    reg_buffered_resumable(m, &["xargs"], Trust::Real, cmd_xargs, start_xargs);
    reg(m, &["comm"], Trust::Real, cmd_comm);
    reg(m, &["diff"], Trust::Partial, cmd_diff);
    reg(m, &["cmp"], Trust::Real, cmd_cmp);
    reg(m, &["expr"], Trust::Real, cmd_expr);
    reg(m, &["bc"], Trust::Real, cmd_bc);
    reg(m, &["factor"], Trust::Real, |_, _, io| {
        ewln(io.err, "factor: unimplemented");
        2
    });
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

/// Resume one bounded streaming text-command quantum.
pub(crate) fn resume_stream(interp: &mut Interp, continuation: TextStream) -> CommandPoll {
    match continuation {
        TextStream::Cat(state) => poll_cat(interp, state),
        TextStream::Head(state) => poll_head(interp, state),
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

fn cmd_yes(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let _ = args;
    ewln(io.err, "yes: unimplemented streaming output");
    2
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

fn cmd_passthrough(_interp: &mut CommandContext<'_>, _args: &[String], io: &mut Io) -> i32 {
    ewln(io.err, "unimplemented");
    2
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

fn cmd_diff(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    if ops.len() < 2 {
        ewln(io.err, "diff: missing operand");
        return 2;
    }
    let a = match interp.fs_read(&interp.cwd, ops[0]) {
        Ok(data) => String::from_utf8_lossy(&data).into_owned(),
        Err(error) => {
            ewln(io.err, &format!("diff: {}: {error}", ops[0]));
            return 2;
        }
    };
    let b = match interp.fs_read(&interp.cwd, ops[1]) {
        Ok(data) => String::from_utf8_lossy(&data).into_owned(),
        Err(error) => {
            ewln(io.err, &format!("diff: {}: {error}", ops[1]));
            return 2;
        }
    };
    if a == b {
        0
    } else {
        // minimal unified-ish output (not a real LCS diff)
        let al: Vec<&str> = a.lines().collect();
        let bl: Vec<&str> = b.lines().collect();
        for (i, line) in al.iter().enumerate() {
            if bl.get(i) != Some(line) {
                wln(io.out, &format!("< {line}"));
            }
        }
        wln(io.out, "---");
        for (i, line) in bl.iter().enumerate() {
            if al.get(i) != Some(line) {
                wln(io.out, &format!("> {line}"));
            }
        }
        1
    }
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
    let mut ignore_case = false;
    let mut invert = false;
    let mut count = false;
    let mut line_num = false;
    let mut files_with = false;
    let mut only_match = false;
    let mut recursive = false;
    let mut extended = cmd == "egrep";
    let mut fixed = cmd == "fgrep";
    let mut word = false;
    let mut quiet = false;
    let mut suppress_errors = false;
    let mut suppress_filename = false;
    let mut patterns = Vec::new();
    let mut files = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a.starts_with('-') && a.len() > 1 && a != "-" {
            if let Some(p) = a.strip_prefix("-e") {
                if p.is_empty() {
                    let Some(pattern) = it.next() else {
                        ewln(io.err, "grep: option requires an argument -- 'e'");
                        return 2;
                    };
                    patterns.push(pattern.clone());
                } else {
                    patterns.push(p.to_string());
                }
                continue;
            }
            if a.starts_with("-A") || a.starts_with("-B") || a.starts_with("-C") {
                ewln(io.err, &format!("grep: unimplemented context option '{a}'"));
                return 2;
            }
            for c in a[1..].chars() {
                match c {
                    'i' => ignore_case = true,
                    'v' => invert = true,
                    'c' => count = true,
                    'n' => line_num = true,
                    'l' => files_with = true,
                    'o' => only_match = true,
                    'r' | 'R' => recursive = true,
                    'E' => extended = true,
                    'F' => fixed = true,
                    'w' => word = true,
                    'q' => quiet = true,
                    'h' => suppress_filename = true,
                    's' => suppress_errors = true,
                    'a' => {}
                    _ => {
                        ewln(io.err, &format!("grep: unimplemented option '-{c}'"));
                        return 2;
                    }
                }
            }
        } else if patterns.is_empty() {
            patterns.push(a.clone());
        } else {
            files.push(a.clone());
        }
    }
    let _ = extended;
    if patterns.is_empty() {
        ewln(io.err, "grep: no pattern");
        return 2;
    }
    let mut pat_re = if fixed {
        patterns
            .iter()
            .map(|pattern| regex::escape(pattern))
            .collect::<Vec<_>>()
            .join("|")
    } else {
        patterns
            .iter()
            .map(|pattern| format!("(?:{pattern})"))
            .collect::<Vec<_>>()
            .join("|")
    };
    if word {
        pat_re = format!(r"\b(?:{pat_re})\b");
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
    let multi = inputs.len() > 1 || recursive;
    let mut total_matches = 0;
    for (label, data) in &inputs {
        let mut file_count = 0;
        let mut matched_file = false;
        for (lineno, line) in String::from_utf8_lossy(data).lines().enumerate() {
            let is_match = re.is_match(line) ^ invert;
            if is_match {
                matched_file = true;
                file_count += 1;
                total_matches += 1;
                if quiet || count || files_with {
                    continue;
                }
                let mut prefix = String::new();
                if multi && !suppress_filename && !label.is_empty() {
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
            }
        }
        if count {
            if multi && !suppress_filename && !label.is_empty() {
                wln(io.out, &format!("{label}:{file_count}"));
            } else {
                wln(io.out, &file_count.to_string());
            }
        }
        if files_with && matched_file {
            wln(io.out, label);
        }
    }
    if had_error {
        2
    } else if total_matches > 0 {
        0
    } else {
        1
    }
}

fn cmd_sed(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut in_place = false;
    let mut quiet = false;
    let mut scripts: Vec<String> = Vec::new();
    let mut extended = false;
    let mut files = Vec::new();
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        if a == "-i" || a.starts_with("-i") {
            in_place = true;
            if a.len() > 2 {
                ewln(io.err, "sed: unimplemented in-place backup suffix");
                return 2;
            }
        } else if a == "-n" {
            quiet = true;
        } else if a == "-r" || a == "-E" {
            extended = true;
        } else if a == "-e" {
            if let Some(s) = it.next() {
                scripts.push(s.clone());
            }
        } else if let Some(s) = a.strip_prefix("-e") {
            scripts.push(s.to_string());
        } else if a == "--" {
            continue;
        } else if a.starts_with('-') {
            ewln(io.err, &format!("sed: unimplemented option '{a}'"));
            return 2;
        } else if scripts.is_empty() && !a.starts_with('-') {
            scripts.push(a.clone());
        } else {
            files.push(a.clone());
        }
    }
    let _ = extended;
    if scripts.is_empty() {
        ewln(io.err, "sed: missing command");
        return 1;
    }
    let mut commands = Vec::new();
    for script in &scripts {
        match parse_sed_script(script) {
            Ok(parsed) => commands.extend(parsed),
            Err(error) => {
                ewln(io.err, &format!("sed: {error}"));
                return 2;
            }
        }
    }

    let process = |text: &str| -> String {
        let mut result = String::new();
        let lines = text.split_inclusive('\n').collect::<Vec<_>>();
        for (index, line) in lines.iter().enumerate() {
            let had_nl = line.ends_with('\n');
            let mut content = line.trim_end_matches('\n').to_string();
            let mut deleted = false;
            let mut printed_extra = Vec::new();
            for operation in &commands {
                if !operation.address.matches(index + 1, lines.len()) {
                    continue;
                }
                match &operation.command {
                    SedCmd::Subst {
                        re,
                        rep,
                        global,
                        nth,
                        print,
                        ignore,
                    } => {
                        let _ = ignore;
                        content = sed_subst(re, rep, &content, *global, *nth);
                        if *print {
                            printed_extra.push(content.clone());
                        }
                    }
                    SedCmd::Delete => {
                        deleted = true;
                    }
                    SedCmd::Print => {
                        printed_extra.push(content.clone());
                    }
                }
            }
            if !quiet && !deleted {
                result.push_str(&content);
                if had_nl {
                    result.push('\n');
                }
            }
            for p in printed_extra {
                result.push_str(&p);
                result.push('\n');
            }
        }
        result
    };

    if in_place && !files.is_empty() {
        let cwd = interp.cwd.clone();
        for f in &files {
            match interp.fs_read(&cwd, f) {
                Ok(data) => {
                    let text = String::from_utf8_lossy(&data);
                    let new = process(&text);
                    if let Err(error) = interp.vfs.write(&cwd, f, new.as_bytes(), 0o644) {
                        ewln(io.err, &format!("sed: can't write {f}: {error}"));
                        return 1;
                    }
                }
                Err(e) => {
                    ewln(io.err, &format!("sed: can't read {f}: {e}"));
                    return 1;
                }
            }
        }
        0
    } else {
        let (data, errors) = read_inputs(interp, &files.iter().collect::<Vec<_>>(), &io.stdin);
        if let Some(error) = errors.first() {
            ewln(io.err, &format!("sed: {error}"));
            return 1;
        }
        let text = String::from_utf8_lossy(&data);
        w(io.out, &process(&text));
        0
    }
}

enum SedAddress {
    Every,
    Line(usize),
    Last,
}

impl SedAddress {
    fn matches(&self, line: usize, total: usize) -> bool {
        match self {
            Self::Every => true,
            Self::Line(expected) => line == *expected,
            Self::Last => line == total,
        }
    }
}

struct SedOperation {
    address: SedAddress,
    command: SedCmd,
}

enum SedCmd {
    Subst {
        re: regex::Regex,
        rep: String,
        global: bool,
        nth: usize,
        print: bool,
        ignore: bool,
    },
    Delete,
    Print,
}

fn parse_sed_script(s: &str) -> Result<Vec<SedOperation>, String> {
    let mut cmds = Vec::new();
    for part in s.split(';') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let digit_count = part
            .chars()
            .take_while(|character| character.is_ascii_digit())
            .count();
        let (address, body) = if digit_count > 0 {
            let line = part[..digit_count]
                .parse()
                .map_err(|_| "invalid line address".to_string())?;
            (SedAddress::Line(line), part[digit_count..].trim_start())
        } else if let Some(body) = part.strip_prefix('$') {
            (SedAddress::Last, body.trim_start())
        } else if part.starts_with('/') || part.contains(',') {
            return Err(format!("unimplemented address in '{part}'"));
        } else {
            (SedAddress::Every, part)
        };
        if let Some(rest) = body.strip_prefix('s') {
            if let Some(cmd) = parse_subst(rest) {
                cmds.push(SedOperation {
                    address,
                    command: cmd,
                });
            } else {
                return Err(format!("invalid substitution '{body}'"));
            }
        } else if body == "d" {
            cmds.push(SedOperation {
                address,
                command: SedCmd::Delete,
            });
        } else if body == "p" {
            cmds.push(SedOperation {
                address,
                command: SedCmd::Print,
            });
        } else {
            return Err(format!("unimplemented command '{body}'"));
        }
    }
    Ok(cmds)
}

fn parse_subst(rest: &str) -> Option<SedCmd> {
    let delim = rest.chars().next()?;
    let chars: Vec<char> = rest.chars().collect();
    let mut i = 1;
    let mut fields = [String::new(), String::new(), String::new()];
    let mut fi = 0;
    while i < chars.len() && fi < 3 {
        let c = chars[i];
        if c == '\\' && i + 1 < chars.len() {
            // keep escapes; but \<delim> becomes literal delim
            if chars[i + 1] == delim {
                fields[fi].push(delim);
            } else {
                fields[fi].push('\\');
                fields[fi].push(chars[i + 1]);
            }
            i += 2;
            continue;
        }
        if c == delim {
            fi += 1;
            i += 1;
            continue;
        }
        fields[fi].push(c);
        i += 1;
    }
    let (pat, rep, flags) = (&fields[0], &fields[1], &fields[2]);
    let global = flags.contains('g');
    let ignore = flags.contains('i') || flags.contains('I');
    let print = flags.contains('p');
    let nth: usize = flags
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap_or(0);
    let re = regex::RegexBuilder::new(pat)
        .case_insensitive(ignore)
        .build()
        .ok()?;
    // convert sed replacement backrefs \1 -> ${1}
    let rep = convert_sed_replacement(rep);
    Some(SedCmd::Subst {
        re,
        rep,
        global,
        nth,
        print,
        ignore,
    })
}

fn convert_sed_replacement(rep: &str) -> String {
    let mut out = String::new();
    let chars: Vec<char> = rep.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            out.push_str(&format!("${{{}}}", chars[i + 1]));
            i += 2;
        } else if chars[i] == '&' {
            out.push_str("${0}");
            i += 1;
        } else if chars[i] == '$' {
            out.push_str("$$");
            i += 1;
        } else if chars[i] == '\\' && i + 1 < chars.len() {
            match chars[i + 1] {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                c => out.push(c),
            }
            i += 2;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn sed_subst(re: &regex::Regex, rep: &str, text: &str, global: bool, nth: usize) -> String {
    if global && nth == 0 {
        re.replace_all(text, rep).into_owned()
    } else if nth > 0 {
        let mut count = 0;
        re.replace_all(text, |caps: &regex::Captures| {
            count += 1;
            if count == nth || (global && count >= nth) {
                expand_caps(rep, caps)
            } else {
                caps[0].to_string()
            }
        })
        .into_owned()
    } else {
        re.replace(text, rep).into_owned()
    }
}

fn expand_caps(rep: &str, caps: &regex::Captures) -> String {
    let mut out = String::new();
    caps.expand(rep, &mut out);
    out
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
