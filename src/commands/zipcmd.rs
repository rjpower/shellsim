//! Deterministic, bounded ZIP containers over the simulated filesystem.
//!
//! Creation emits portable stored entries. Listing and extraction also accept raw-deflate
//! entries, validate CRCs and sizes, reject special entries and unsafe paths, and extract into a
//! cloned VFS so malformed archives cannot leave partial changes.

use std::collections::{BTreeSet, HashMap};
use std::io::Read;

use crc32fast::hash;
use flate2::read::DeflateDecoder;

use crate::commands::util::{ewln, wln};
use crate::commands::{CommandSpec, Io, Trust};
use crate::program::ProcessContext;
use crate::syscalls::{FileChange, FileKind, System};
use crate::vfs::{parent_of, resolve_against};

const MAX_ARCHIVE_BYTES: usize = 16 * 1024 * 1024;
const MAX_ENTRIES: usize = 4_096;
const MAX_NAME_BYTES: usize = 4_096;

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    super::reg_system(commands, "/usr/bin/zip", Trust::Partial, cmd_zip);
    super::reg_system(commands, "/usr/bin/unzip", Trust::Partial, cmd_unzip);
}

#[derive(Clone)]
struct Entry {
    name: String,
    mode: u32,
    data: Option<Vec<u8>>,
}

fn cmd_zip(context: &mut ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let mut recursive = false;
    let mut operands = Vec::new();
    let mut options_done = false;
    for argument in args {
        match argument.as_str() {
            "--" if !options_done => options_done = true,
            "-r" | "--recurse-paths" if !options_done => recursive = true,
            "-q" | "--quiet" if !options_done => {}
            option
                if !options_done
                    && option.len() > 1
                    && option.starts_with('-')
                    && option[1..].chars().all(|flag| matches!(flag, 'q' | 'r')) =>
            {
                recursive |= option.contains('r');
            }
            option if !options_done && option.starts_with('-') => {
                ewln(io.err, &format!("zip: unsupported option: {option}"));
                return 2;
            }
            _ => operands.push(argument.clone()),
        }
    }
    let Some(archive_path) = operands.first() else {
        ewln(io.err, "zip: missing archive path");
        return 2;
    };
    if operands.len() == 1 {
        ewln(io.err, "zip: nothing to do");
        return 12;
    }
    let entries = match collect(system, &operands[1..], recursive) {
        Ok(entries) => entries,
        Err(error) => {
            ewln(io.err, &format!("zip: {error}"));
            return 2;
        }
    };
    let archive = match encode(system, &entries) {
        Ok(archive) => archive,
        Err(error) => {
            ewln(io.err, &format!("zip: {error}"));
            return 2;
        }
    };
    let cwd = system.cwd().to_string();
    if let Err(error) = system.write_file(&cwd, archive_path, &archive, 0o644) {
        ewln(io.err, &format!("zip: {archive_path}: {error}"));
        return 2;
    }
    0
}

fn collect(
    system: &mut dyn System,
    operands: &[String],
    recursive: bool,
) -> Result<Vec<Entry>, String> {
    let base = system.cwd().to_string();
    let mut paths = BTreeSet::new();
    for operand in operands {
        let absolute = resolve_against(&base, operand);
        let metadata = system
            .metadata("/", &absolute, false)
            .map_err(|error| format!("{operand}: {error}"))?;
        match metadata.kind {
            FileKind::File if !metadata.native_executable => {
                paths.insert(absolute);
            }
            FileKind::Directory if recursive => paths.extend(
                system
                    .walk("/", &absolute)
                    .map_err(|error| error.to_string())?,
            ),
            FileKind::Directory => return Err(format!("{operand}: is a directory (use -r)")),
            FileKind::Symlink => {
                return Err(format!("{operand}: symbolic links are not supported"));
            }
            FileKind::File => {
                return Err(format!("{operand}: native executables cannot be archived"));
            }
        }
    }
    if paths.len() > MAX_ENTRIES {
        return Err(format!("archive exceeds the {MAX_ENTRIES}-entry limit"));
    }
    paths
        .into_iter()
        .map(|path| {
            let metadata = system
                .metadata("/", &path, false)
                .map_err(|error| error.to_string())?;
            let mut name = if base == "/" {
                path.trim_start_matches('/').to_string()
            } else {
                path.strip_prefix(&format!("{base}/"))
                    .unwrap_or_else(|| path.trim_start_matches('/'))
                    .to_string()
            };
            let data = match metadata.kind {
                FileKind::File if !metadata.native_executable => Some(
                    system
                        .read_file_limited("/", &path, MAX_ARCHIVE_BYTES)
                        .map_err(|error| error.to_string())?,
                ),
                FileKind::Directory => {
                    name.push('/');
                    None
                }
                FileKind::Symlink => return Err(format!("{path}: unsupported symbolic link")),
                FileKind::File => {
                    return Err(format!("{path}: native executables cannot be archived"));
                }
            };
            validate_name(&name)?;
            Ok(Entry {
                name,
                mode: metadata.mode,
                data,
            })
        })
        .collect()
}

fn encode(system: &mut dyn System, entries: &[Entry]) -> Result<Vec<u8>, String> {
    let estimated = entries.iter().try_fold(22usize, |total, entry| {
        total
            .checked_add(30 + entry.name.len() + entry.data.as_ref().map_or(0, Vec::len))?
            .checked_add(46 + entry.name.len())
    });
    let estimated = estimated.ok_or_else(|| "archive size overflow".to_string())?;
    if estimated > MAX_ARCHIVE_BYTES || entries.len() > usize::from(u16::MAX) {
        return Err("archive exceeds the supported ZIP32 limits".to_string());
    }
    if !system.reserve_memory(estimated as u64) {
        return Err("memory limit exceeded".to_string());
    }
    if !system.charge_cpu(estimated as u64) {
        system.release_memory(estimated as u64);
        return Err("resource limit exceeded".to_string());
    }
    let mut output = Vec::with_capacity(estimated);
    let mut central = Vec::with_capacity(entries.len());
    for entry in entries {
        let offset = u32::try_from(output.len()).map_err(|_| "archive is too large")?;
        let data = entry.data.as_deref().unwrap_or_default();
        let crc = hash(data);
        output.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        output.extend_from_slice(&20u16.to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        output.extend_from_slice(&[0; 4]);
        output.extend_from_slice(&crc.to_le_bytes());
        output.extend_from_slice(&(data.len() as u32).to_le_bytes());
        output.extend_from_slice(&(data.len() as u32).to_le_bytes());
        output.extend_from_slice(&(entry.name.len() as u16).to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        output.extend_from_slice(entry.name.as_bytes());
        output.extend_from_slice(data);
        central.push((entry, offset, crc));
    }
    let central_offset = output.len();
    for (entry, offset, crc) in central {
        let data_len = entry.data.as_ref().map_or(0, Vec::len) as u32;
        let kind = if entry.data.is_some() {
            0o100000
        } else {
            0o040000
        };
        let unix_mode = kind | (entry.mode & 0o7777);
        output.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        output.extend_from_slice(&0x031eu16.to_le_bytes());
        output.extend_from_slice(&20u16.to_le_bytes());
        output.extend_from_slice(&[0; 8]);
        output.extend_from_slice(&crc.to_le_bytes());
        output.extend_from_slice(&data_len.to_le_bytes());
        output.extend_from_slice(&data_len.to_le_bytes());
        output.extend_from_slice(&(entry.name.len() as u16).to_le_bytes());
        output.extend_from_slice(&[0; 8]);
        output.extend_from_slice(&(unix_mode << 16).to_le_bytes());
        output.extend_from_slice(&offset.to_le_bytes());
        output.extend_from_slice(entry.name.as_bytes());
    }
    let central_size = output.len() - central_offset;
    output.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    output.extend_from_slice(&[0; 4]);
    output.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    output.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    output.extend_from_slice(&(central_size as u32).to_le_bytes());
    output.extend_from_slice(&(central_offset as u32).to_le_bytes());
    output.extend_from_slice(&0u16.to_le_bytes());
    system.release_memory(estimated as u64);
    Ok(output)
}

fn cmd_unzip(context: &mut ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let mut list = false;
    let mut names_only = false;
    let mut directory = None;
    let mut archive = None;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-l" => list = true,
            "-Z1" => names_only = true,
            // Extraction already replaces existing VFS files atomically, so `-o` only makes the
            // non-interactive intent explicit.
            "-o" | "-q" => {}
            "-d" => {
                index += 1;
                directory = args.get(index).cloned();
                if directory.is_none() {
                    ewln(io.err, "unzip: -d requires a directory");
                    return 2;
                }
            }
            option
                if option.len() > 2
                    && option.starts_with('-')
                    && option[1..]
                        .chars()
                        .all(|value| matches!(value, 'l' | 'o' | 'q')) =>
            {
                list |= option[1..].contains('l');
            }
            option if option.starts_with('-') => {
                ewln(io.err, &format!("unzip: unsupported option: {option}"));
                return 2;
            }
            value if archive.is_none() => archive = Some(value.to_string()),
            value => {
                ewln(
                    io.err,
                    &format!("unzip: member selection is not supported: {value}"),
                );
                return 2;
            }
        }
        index += 1;
    }
    let Some(archive) = archive else {
        ewln(io.err, "unzip: missing archive path");
        return 2;
    };
    let cwd = system.cwd().to_string();
    let bytes = match system.read_file_limited(&cwd, &archive, MAX_ARCHIVE_BYTES) {
        Ok(bytes) => bytes,
        Err(error) => {
            ewln(io.err, &format!("unzip: {archive}: {error}"));
            return 2;
        }
    };
    let entries = match decode(system, &bytes) {
        Ok(entries) => entries,
        Err(error) => {
            ewln(io.err, &format!("unzip: {archive}: {error}"));
            return 2;
        }
    };
    if list || names_only {
        for entry in entries {
            if names_only {
                wln(io.out, &entry.name);
            } else {
                wln(
                    io.out,
                    &format!(
                        "{:>9}  {}",
                        entry.data.as_ref().map_or(0, Vec::len),
                        entry.name
                    ),
                );
            }
        }
        return 0;
    }
    let base = directory.as_deref().map_or_else(
        || system.cwd().to_string(),
        |path| resolve_against(system.cwd(), path),
    );
    if let Err(error) = extract(system, &base, &entries) {
        ewln(io.err, &format!("unzip: {error}"));
        return 2;
    }
    0
}

type CentralRecord = (String, u16, u32, usize, usize, u32, usize);

fn decode(system: &mut dyn System, archive: &[u8]) -> Result<Vec<Entry>, String> {
    let eocd = find_eocd(archive)?;
    if read_u16(archive, eocd + 4)? != 0 || read_u16(archive, eocd + 6)? != 0 {
        return Err("multi-disk archives are not supported".to_string());
    }
    let count = usize::from(read_u16(archive, eocd + 10)?);
    if usize::from(read_u16(archive, eocd + 8)?) != count {
        return Err("multi-disk archives are not supported".to_string());
    }
    if count > MAX_ENTRIES {
        return Err(format!("archive exceeds the {MAX_ENTRIES}-entry limit"));
    }
    let central_size = read_u32(archive, eocd + 12)? as usize;
    let central_offset = read_u32(archive, eocd + 16)? as usize;
    let central_end = central_offset
        .checked_add(central_size)
        .filter(|end| *end <= eocd)
        .ok_or_else(|| "invalid central directory bounds".to_string())?;
    let mut cursor = central_offset;
    let mut records: Vec<CentralRecord> = Vec::with_capacity(count);
    let mut total_size = 0usize;
    for _ in 0..count {
        if read_u32(archive, cursor)? != 0x0201_4b50 {
            return Err("invalid central directory entry".to_string());
        }
        let flags = read_u16(archive, cursor + 8)?;
        if flags & 1 != 0 {
            return Err("encrypted entries are not supported".to_string());
        }
        let method = read_u16(archive, cursor + 10)?;
        let crc = read_u32(archive, cursor + 16)?;
        let compressed = read_u32(archive, cursor + 20)? as usize;
        let uncompressed = read_u32(archive, cursor + 24)? as usize;
        total_size = total_size
            .checked_add(uncompressed)
            .ok_or_else(|| "uncompressed size overflow".to_string())?;
        if total_size > MAX_ARCHIVE_BYTES {
            return Err("uncompressed contents exceed the 16 MiB limit".to_string());
        }
        let name_len = usize::from(read_u16(archive, cursor + 28)?);
        let extra_len = usize::from(read_u16(archive, cursor + 30)?);
        let comment_len = usize::from(read_u16(archive, cursor + 32)?);
        let external = read_u32(archive, cursor + 38)?;
        let local_offset = read_u32(archive, cursor + 42)? as usize;
        if name_len == 0 || name_len > MAX_NAME_BYTES {
            return Err("invalid member name length".to_string());
        }
        let name_start = cursor.saturating_add(46);
        let name_end = name_start
            .checked_add(name_len)
            .filter(|end| *end <= central_end)
            .ok_or_else(|| "truncated member name".to_string())?;
        let name = std::str::from_utf8(&archive[name_start..name_end])
            .map_err(|_| "member name is not UTF-8".to_string())?
            .to_string();
        validate_name(&name)?;
        records.push((
            name,
            method,
            crc,
            compressed,
            uncompressed,
            external,
            local_offset,
        ));
        cursor = name_end
            .checked_add(extra_len)
            .and_then(|next| next.checked_add(comment_len))
            .filter(|next| *next <= central_end)
            .ok_or_else(|| "truncated central directory".to_string())?;
    }
    if !system.reserve_memory(total_size as u64) {
        return Err("memory limit exceeded".to_string());
    }
    if !system.charge_cpu(archive.len() as u64 + total_size as u64) {
        system.release_memory(total_size as u64);
        return Err("resource limit exceeded".to_string());
    }
    let result = records
        .into_iter()
        .map(|record| decode_entry(archive, record))
        .collect();
    system.release_memory(total_size as u64);
    result
}

fn decode_entry(archive: &[u8], record: CentralRecord) -> Result<Entry, String> {
    let (name, method, crc, compressed, uncompressed, external, local_offset) = record;
    if read_u32(archive, local_offset)? != 0x0403_4b50 {
        return Err(format!("{name}: invalid local header"));
    }
    if read_u16(archive, local_offset + 8)? != method {
        return Err(format!("{name}: local compression method mismatch"));
    }
    let local_name = usize::from(read_u16(archive, local_offset + 26)?);
    let local_extra = usize::from(read_u16(archive, local_offset + 28)?);
    let data_start = local_offset
        .checked_add(30 + local_name + local_extra)
        .ok_or_else(|| format!("{name}: local header overflow"))?;
    let local_name_end = local_offset
        .checked_add(30 + local_name)
        .filter(|end| *end <= archive.len())
        .ok_or_else(|| format!("{name}: truncated local name"))?;
    if archive.get(local_offset + 30..local_name_end) != Some(name.as_bytes()) {
        return Err(format!("{name}: local name mismatch"));
    }
    let data_end = data_start
        .checked_add(compressed)
        .filter(|end| *end <= archive.len())
        .ok_or_else(|| format!("{name}: truncated contents"))?;
    let directory = name.ends_with('/');
    let data = if directory {
        if uncompressed != 0 || compressed != 0 || crc != 0 {
            return Err(format!("{name}: directory has contents"));
        }
        None
    } else {
        let data = match method {
            0 => archive[data_start..data_end].to_vec(),
            8 => {
                let mut output = Vec::with_capacity(uncompressed);
                DeflateDecoder::new(&archive[data_start..data_end])
                    .take((uncompressed as u64).saturating_add(1))
                    .read_to_end(&mut output)
                    .map_err(|error| format!("{name}: {error}"))?;
                output
            }
            _ => return Err(format!("{name}: unsupported compression method {method}")),
        };
        if data.len() != uncompressed || hash(&data) != crc {
            return Err(format!("{name}: size or CRC mismatch"));
        }
        Some(data)
    };
    let file_type = (external >> 16) & 0o170000;
    if file_type != 0 && file_type != 0o100000 && file_type != 0o040000 {
        return Err(format!("{name}: special entries are not supported"));
    }
    let mode = (external >> 16) & 0o7777;
    Ok(Entry {
        name,
        mode: if mode == 0 {
            if data.is_some() {
                0o644
            } else {
                0o755
            }
        } else {
            mode
        },
        data,
    })
}

fn extract(system: &mut dyn System, base: &str, entries: &[Entry]) -> Result<(), String> {
    let mut changes = vec![FileChange::MkdirAll(base.to_string())];
    for entry in entries {
        let destination = resolve_against(base, entry.name.trim_end_matches('/'));
        validate_destination(system, base, &destination)?;
        if let Some(data) = &entry.data {
            if let Some(parent) = parent_of(&destination) {
                changes.push(FileChange::MkdirAll(parent));
            }
            changes.push(FileChange::PutFile {
                path: destination,
                bytes: data.clone(),
                mode: entry.mode,
            });
        } else {
            changes.push(FileChange::MkdirAll(destination));
        }
    }
    system
        .apply_file_batch("/", changes)
        .map_err(|error| error.to_string())
}

fn validate_name(name: &str) -> Result<(), String> {
    let trimmed = name.trim_end_matches('/');
    if name.len() > MAX_NAME_BYTES
        || name.starts_with('/')
        || name.contains('\\')
        || trimmed
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(format!("unsafe archive path: {name}"));
    }
    Ok(())
}

fn validate_destination(
    system: &mut dyn System,
    base: &str,
    destination: &str,
) -> Result<(), String> {
    let base_prefix = format!("{}/", base.trim_end_matches('/'));
    if destination != base && !destination.starts_with(&base_prefix) {
        return Err(format!("unsafe archive destination: {destination}"));
    }
    if matches!(
        system
            .metadata("/", destination, false)
            .map(|info| info.kind),
        Ok(FileKind::Symlink)
    ) {
        return Err(format!(
            "symbolic-link destination is not allowed: {destination}"
        ));
    }
    let mut current = parent_of(destination);
    while let Some(path) = current {
        if path.len() < base.len() {
            break;
        }
        if matches!(
            system.metadata("/", &path, false).map(|info| info.kind),
            Ok(FileKind::Symlink)
        ) {
            return Err(format!("symbolic-link parent is not allowed: {path}"));
        }
        if path == base {
            break;
        }
        current = parent_of(&path);
    }
    Ok(())
}

fn find_eocd(input: &[u8]) -> Result<usize, String> {
    let start = input.len().saturating_sub(65_557);
    for offset in (start..input.len().saturating_sub(3)).rev() {
        if input.get(offset..offset + 4) != Some(b"PK\x05\x06") {
            continue;
        }
        let Ok(comment_len) = read_u16(input, offset + 20) else {
            continue;
        };
        if offset
            .checked_add(22 + usize::from(comment_len))
            .is_some_and(|end| end == input.len())
        {
            return Ok(offset);
        }
    }
    Err("missing end-of-central-directory record".to_string())
}

fn read_u16(input: &[u8], offset: usize) -> Result<u16, String> {
    let bytes = input
        .get(offset..offset.saturating_add(2))
        .ok_or_else(|| "truncated ZIP structure".to_string())?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(input: &[u8], offset: usize) -> Result<u32, String> {
    let bytes = input
        .get(offset..offset.saturating_add(4))
        .ok_or_else(|| "truncated ZIP structure".to_string())?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}
