//! Bounded archive and compression commands over simulated files and descriptors.
//!
//! The first slice models deterministic gzip streams. Decompression is capped before host-side
//! materialization, all named inputs and outputs stay in the VFS, and unsupported command-line
//! forms fail explicitly instead of reaching a host utility.

use std::collections::HashMap;
use std::io::{Read, Write};

use flate2::read::MultiGzDecoder;
use flate2::{Compression, GzBuilder};

use crate::commands::util::ewln;
use crate::commands::{CommandContext, CommandSpec, Io, Trust};
use crate::exec::ShellPoll;
use crate::program::ProcessContext;
use crate::syscalls::{FileKind, System};

const MAX_DECOMPRESSED_BYTES: usize = 16 * 1024 * 1024;

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    super::reg_system_poll(commands, "/usr/bin/gzip", Trust::Partial, cmd_gzip);
    super::reg_system_poll(commands, "/usr/bin/gunzip", Trust::Partial, cmd_gunzip);
    super::reg_system_poll(commands, "/usr/bin/zcat", Trust::Partial, cmd_zcat);
}

#[derive(Clone, Copy)]
enum Operation {
    Compress,
    Decompress,
}

struct Options {
    operation: Operation,
    stdout: bool,
    keep: bool,
    force: bool,
    level: u32,
    files: Vec<String>,
}

fn cmd_gzip(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    run(context, io, Operation::Compress, false)
}

fn cmd_gunzip(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    run(context, io, Operation::Decompress, false)
}

fn cmd_zcat(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    run(context, io, Operation::Decompress, true)
}

fn run(
    context: &mut ProcessContext<'_>,
    io: &mut Io,
    default_operation: Operation,
    force_stdout: bool,
) -> ShellPoll {
    let options = match parse_options(context.args, default_operation, force_stdout) {
        Ok(options) => options,
        Err(message) => {
            ewln(io.err, &message);
            return ShellPoll::Ready(1);
        }
    };
    if options.files.is_empty() || options.files.iter().any(|file| file == "-") {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let mark = context.system.memory_used();
    let status = run_ready(context.system, io, &options);
    context
        .system
        .release_memory(context.system.memory_used().saturating_sub(mark));
    ShellPoll::Ready(status)
}

fn run_ready(system: &mut dyn System, io: &mut Io, options: &Options) -> i32 {
    if options.files.is_empty() {
        return transform_to_stdout(system, options, &io.stdin, io.out, io.err);
    }

    let mut status = 0;
    for path in &options.files {
        if path == "-" {
            if transform_to_stdout(system, options, &io.stdin, io.out, io.err) != 0 {
                status = 1;
            }
            continue;
        }
        let cwd = system.cwd().to_string();
        let input_len = match system.metadata(&cwd, path, true) {
            Ok(info) if info.kind == FileKind::File => info.size,
            Ok(_) => {
                ewln(io.err, &format!("gzip: {path}: not a regular file"));
                status = 1;
                continue;
            }
            Err(error) => {
                ewln(io.err, &format!("gzip: {path}: {error}"));
                status = 1;
                continue;
            }
        };
        if !system.reserve_memory(input_len) {
            ewln(io.err, &format!("gzip: {path}: memory limit exceeded"));
            status = 1;
            continue;
        }
        let input = match system.read_file_limited(
            &cwd,
            path,
            usize::try_from(system.limits().memory).unwrap_or(usize::MAX),
        ) {
            Ok(input) => input,
            Err(error) => {
                ewln(io.err, &format!("gzip: {path}: {error}"));
                status = 1;
                continue;
            }
        };
        if options.stdout {
            if transform_to_stdout(system, options, &input, io.out, io.err) != 0 {
                status = 1;
            }
            continue;
        }
        let destination = match output_name(path, options.operation) {
            Ok(path) => path,
            Err(message) => {
                ewln(io.err, &message);
                status = 1;
                continue;
            }
        };
        if !options.force && system.metadata(&cwd, &destination, false).is_ok() {
            ewln(io.err, &format!("gzip: {destination} already exists"));
            status = 1;
            continue;
        }
        let output = match transform(system, options.operation, options.level, &input) {
            Ok(output) => output,
            Err(message) => {
                ewln(io.err, &format!("gzip: {path}: {message}"));
                status = 1;
                continue;
            }
        };
        if let Err(error) = system.write_file(&cwd, &destination, &output, 0o644) {
            ewln(io.err, &format!("gzip: {destination}: {error}"));
            status = 1;
            continue;
        }
        if !options.keep {
            if let Err(error) = system.unlink(&cwd, path) {
                ewln(io.err, &format!("gzip: {path}: {error}"));
                status = 1;
            }
        }
    }
    status
}

fn transform_to_stdout(
    system: &mut dyn System,
    options: &Options,
    input: &[u8],
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
) -> i32 {
    match transform(system, options.operation, options.level, input) {
        Ok(output) => {
            out.extend_from_slice(&output);
            0
        }
        Err(message) => {
            ewln(err, &format!("gzip: {message}"));
            1
        }
    }
}

fn transform(
    system: &mut dyn System,
    operation: Operation,
    level: u32,
    input: &[u8],
) -> Result<Vec<u8>, String> {
    if !system.charge_cpu(u64::try_from(input.len()).unwrap_or(u64::MAX)) {
        return Err("resource limit exceeded".to_string());
    }
    match operation {
        Operation::Compress => {
            let bound = input
                .len()
                .checked_add(input.len() / 8)
                .and_then(|size| size.checked_add(128))
                .ok_or_else(|| "compressed output is too large".to_string())?;
            if !system.reserve_memory(u64::try_from(bound).unwrap_or(u64::MAX)) {
                return Err("memory limit exceeded".to_string());
            }
            let encoder = GzBuilder::new()
                .mtime(0)
                .write(Vec::with_capacity(bound), Compression::new(level));
            let mut encoder = encoder;
            encoder
                .write_all(input)
                .map_err(|error| error.to_string())?;
            encoder.finish().map_err(|error| error.to_string())
        }
        Operation::Decompress => {
            if !system.reserve_memory(MAX_DECOMPRESSED_BYTES as u64) {
                return Err("memory limit exceeded".to_string());
            }
            let decoder = MultiGzDecoder::new(input);
            let mut output = Vec::new();
            decoder
                .take((MAX_DECOMPRESSED_BYTES + 1) as u64)
                .read_to_end(&mut output)
                .map_err(|error| error.to_string())?;
            if output.len() > MAX_DECOMPRESSED_BYTES {
                return Err("decompressed output exceeds the 16 MiB limit".to_string());
            }
            if !system.charge_cpu(u64::try_from(output.len()).unwrap_or(u64::MAX)) {
                return Err("resource limit exceeded".to_string());
            }
            Ok(output)
        }
    }
}

/// Compress a tar byte stream with shellsim's deterministic gzip envelope.
pub(super) fn gzip_bytes(interp: &mut CommandContext<'_>, input: &[u8]) -> Result<Vec<u8>, String> {
    transform(&mut interp.system(), Operation::Compress, 6, input)
}

/// Decompress a bounded gzip byte stream for another modeled archive command.
pub(super) fn gunzip_bytes(
    interp: &mut CommandContext<'_>,
    input: &[u8],
) -> Result<Vec<u8>, String> {
    transform(&mut interp.system(), Operation::Decompress, 6, input)
}

fn output_name(path: &str, operation: Operation) -> Result<String, String> {
    match operation {
        Operation::Compress => {
            if path.ends_with(".gz") {
                Err(format!("gzip: {path} already has .gz suffix"))
            } else {
                Ok(format!("{path}.gz"))
            }
        }
        Operation::Decompress => path
            .strip_suffix(".gz")
            .map(str::to_string)
            .or_else(|| path.strip_suffix(".tgz").map(|stem| format!("{stem}.tar")))
            .ok_or_else(|| format!("gzip: {path}: unknown suffix")),
    }
}

fn parse_options(
    args: &[String],
    default_operation: Operation,
    force_stdout: bool,
) -> Result<Options, String> {
    let mut options = Options {
        operation: default_operation,
        stdout: force_stdout,
        keep: false,
        force: false,
        level: 6,
        files: Vec::new(),
    };
    let mut operands = false;
    for argument in args {
        if operands || !argument.starts_with('-') || argument == "-" {
            options.files.push(argument.clone());
            continue;
        }
        if argument == "--" {
            operands = true;
            continue;
        }
        match argument.as_str() {
            "--stdout" => options.stdout = true,
            "--decompress" => options.operation = Operation::Decompress,
            "--keep" => options.keep = true,
            "--force" => options.force = true,
            _ if argument.starts_with("--") => {
                return Err(format!("gzip: unsupported option {argument}"));
            }
            _ => {
                for flag in argument[1..].chars() {
                    match flag {
                        'c' => options.stdout = true,
                        'd' => options.operation = Operation::Decompress,
                        'k' => options.keep = true,
                        'f' => options.force = true,
                        '1'..='9' => options.level = flag.to_digit(10).expect("digit was matched"),
                        _ => return Err(format!("gzip: unsupported option -{flag}")),
                    }
                }
            }
        }
    }
    Ok(options)
}
