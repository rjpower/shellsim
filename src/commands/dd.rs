//! Bounded byte copying for the common `dd` operand interface.
//!
//! The implementation deliberately materializes the selected input and output in memory. This is
//! simple and deterministic, and the environment's memory and disk limits bound both buffers.
//! Supported conversions stop at `conv=notrunc`; device-specific flags and encoding conversions
//! fail explicitly instead of changing data approximately.

use std::collections::HashMap;

use crate::commands::util::ewln;
use crate::commands::{reg, CommandContext, CommandSpec, Io, Trust};
use crate::vfs::NodeKind;

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    reg(commands, &["dd"], Trust::Partial, cmd_dd);
}

#[derive(Debug)]
struct Options {
    input: Option<String>,
    output: Option<String>,
    input_block: usize,
    output_block: usize,
    count: Option<usize>,
    skip: usize,
    seek: usize,
    notrunc: bool,
    status: Status,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Default,
    None,
    NoXfer,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            input: None,
            output: None,
            input_block: 512,
            output_block: 512,
            count: None,
            skip: 0,
            seek: 0,
            notrunc: false,
            status: Status::Default,
        }
    }
}

fn cmd_dd(environment: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let memory_mark = environment.resources.memory_mark();
    let status = run_dd(environment, args, io);
    environment.resources.restore_memory(memory_mark);
    status
}

fn run_dd(environment: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let options = match parse_options(args) {
        Ok(options) => options,
        Err(error) => return fail(io, &error),
    };
    let offset = match options.skip.checked_mul(options.input_block) {
        Some(offset) => offset,
        None => return fail(io, "input offset is too large"),
    };
    let maximum = match options.count {
        Some(count) => match count.checked_mul(options.input_block) {
            Some(maximum) => Some(maximum),
            None => return fail(io, "input size is too large"),
        },
        None => None,
    };

    let input = match read_input(environment, io, &options, maximum) {
        Ok(input) => input,
        Err(error) => return fail(io, &error),
    };
    let selected = if options.input.as_deref() == Some("/dev/zero") {
        input.as_slice()
    } else {
        let start = offset.min(input.len());
        let end = maximum
            .and_then(|length| start.checked_add(length))
            .unwrap_or(input.len())
            .min(input.len());
        &input[start..end]
    };
    if !environment.charge_cpu(u64::try_from(selected.len()).unwrap_or(u64::MAX)) {
        return 137;
    }

    let status = if let Some(path) = &options.output {
        write_output(environment, path, selected, &options, io)
    } else if options.seek != 0 {
        fail(io, "seek requires an output file")
    } else {
        if !environment
            .resources
            .charge_output(u64::try_from(selected.len()).unwrap_or(u64::MAX))
        {
            return 137;
        }
        io.out.extend_from_slice(selected);
        0
    };
    if status != 0 {
        return status;
    }
    if options.status != Status::None {
        let full = selected.len() / options.input_block;
        let partial = usize::from(selected.len() % options.input_block != 0);
        ewln(io.err, &format!("{full}+{partial} records in"));
        let full = selected.len() / options.output_block;
        let partial = usize::from(selected.len() % options.output_block != 0);
        ewln(io.err, &format!("{full}+{partial} records out"));
        if options.status == Status::Default {
            ewln(io.err, &format!("{} bytes copied", selected.len()));
        }
    }
    0
}

fn read_input(
    environment: &mut CommandContext<'_>,
    io: &Io,
    options: &Options,
    maximum: Option<usize>,
) -> Result<Vec<u8>, String> {
    match options.input.as_deref() {
        None | Some("-") => {
            reserve_bytes(environment, io.stdin.len())?;
            Ok(io.stdin.clone())
        }
        Some("/dev/zero") => {
            let length = maximum.ok_or_else(|| {
                "reading /dev/zero requires count= to keep the operation bounded".to_string()
            })?;
            reserve_bytes(environment, length)?;
            Ok(vec![0; length])
        }
        Some(path) => {
            let length = environment
                .fs_file_len(&environment.cwd, path)
                .map_err(|error| format!("failed to open {path}: {error}"))?;
            reserve_bytes(environment, length)?;
            environment
                .fs_read(&environment.cwd, path)
                .map_err(|error| format!("failed to open {path}: {error}"))
        }
    }
}

fn write_output(
    environment: &mut CommandContext<'_>,
    path: &str,
    selected: &[u8],
    options: &Options,
    io: &mut Io,
) -> i32 {
    if path == "/dev/null" {
        return 0;
    }
    let offset = match options.seek.checked_mul(options.output_block) {
        Some(offset) => offset,
        None => return fail(io, "output offset is too large"),
    };
    let end = match offset.checked_add(selected.len()) {
        Some(end) => end,
        None => return fail(io, "output size is too large"),
    };
    let absolute = crate::vfs::resolve_against(&environment.cwd, path);
    let real = environment
        .vfs
        .realpath(&absolute, true)
        .unwrap_or(absolute);
    let existing = environment.vfs.raw_get(&real).map(|node| match &node.kind {
        NodeKind::File(bytes) => Ok((node.mode, bytes.len())),
        _ => Err(()),
    });
    let (mode, existing_length) = match existing {
        Some(Ok(metadata)) => metadata,
        Some(Err(())) => return fail(io, &format!("failed to open {path}: Is a directory")),
        None => (0o666, 0),
    };
    let working_size = if options.notrunc {
        existing_length.max(end)
    } else {
        end
    };
    if reserve_bytes(environment, working_size).is_err() {
        return 137;
    }
    let mut output = if options.notrunc {
        if existing.is_some() {
            match environment.vfs.read(&environment.cwd, path) {
                Ok(bytes) => bytes,
                Err(error) => return fail(io, &format!("failed to open {path}: {error}")),
            }
        } else {
            Vec::new()
        }
    } else {
        Vec::with_capacity(end)
    };
    if output.len() < end {
        output.resize(end, 0);
    }
    output[offset..end].copy_from_slice(selected);
    environment.sync_vfs_time();
    let cwd = environment.cwd.clone();
    if let Err(error) = environment.vfs.write(&cwd, path, &output, mode) {
        return fail(io, &format!("failed to open {path}: {error}"));
    }
    0
}

fn reserve_bytes(environment: &mut CommandContext<'_>, length: usize) -> Result<(), String> {
    if environment.reserve_memory(u64::try_from(length).unwrap_or(u64::MAX)) {
        Ok(())
    } else {
        Err("memory limit exceeded".into())
    }
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    for argument in args {
        if argument == "--help" || argument == "--version" || argument.starts_with('-') {
            return Err(format!("unsupported option {argument:?}"));
        }
        let Some((name, value)) = argument.split_once('=') else {
            return Err(format!("unrecognized operand {argument:?}"));
        };
        match name {
            "if" => options.input = Some(value.to_string()),
            "of" => options.output = Some(value.to_string()),
            "bs" => {
                let size = parse_size(value)?;
                options.input_block = size;
                options.output_block = size;
            }
            "ibs" => options.input_block = parse_size(value)?,
            "obs" => options.output_block = parse_size(value)?,
            "count" => options.count = Some(parse_number(value)?),
            "skip" => options.skip = parse_number(value)?,
            "seek" => options.seek = parse_number(value)?,
            "conv" if value.split(',').all(|part| part == "notrunc") => {
                options.notrunc = true;
            }
            "conv" => return Err(format!("unsupported conversion {value:?}")),
            "status" => {
                options.status = match value {
                    "none" => Status::None,
                    "noxfer" => Status::NoXfer,
                    _ => return Err(format!("unsupported status {value:?}")),
                };
            }
            _ => return Err(format!("unrecognized operand {argument:?}")),
        }
    }
    Ok(options)
}

fn parse_size(value: &str) -> Result<usize, String> {
    let (number, multiplier) = match value.as_bytes().last().copied() {
        Some(b'c') => (&value[..value.len() - 1], 1usize),
        Some(b'b') => (&value[..value.len() - 1], 512usize),
        Some(b'K') | Some(b'k') => (&value[..value.len() - 1], 1024usize),
        Some(b'M') => (&value[..value.len() - 1], 1024usize * 1024),
        Some(b'G') => (&value[..value.len() - 1], 1024usize * 1024 * 1024),
        _ => (value, 1usize),
    };
    let size = parse_number(number)?
        .checked_mul(multiplier)
        .ok_or_else(|| format!("invalid number {value:?}"))?;
    if size == 0 {
        return Err("block size must be greater than zero".into());
    }
    Ok(size)
}

fn parse_number(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid number {value:?}"))
}

fn fail(io: &mut Io, message: &str) -> i32 {
    ewln(io.err, &format!("dd: {message}"));
    1
}

#[cfg(test)]
mod tests {
    use super::{parse_options, parse_size};

    #[test]
    fn parses_block_suffixes_and_rejects_unknown_conversions() {
        assert_eq!(parse_size("2K").unwrap(), 2048);
        assert_eq!(parse_size("2b").unwrap(), 1024);
        let error = parse_options(&["conv=sync".into()]).unwrap_err();
        assert!(error.contains("unsupported conversion"));
    }
}
