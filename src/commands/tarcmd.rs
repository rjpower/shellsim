//! Deterministic, bounded tar container support over the VFS.
//!
//! Creation, listing, and extraction support regular files and directories plus optional gzip
//! composition. Extraction validates every header and destination before mutating a cloned VFS,
//! rejects traversal and special-file entries, and rolls the entire operation back on failure.

use std::collections::{BTreeSet, HashMap};

use crate::commands::util::{ewln, wln};
use crate::commands::{CommandContext, CommandSpec, Io, Trust};
use crate::vfs::{parent_of, resolve_against, NodeKind};

const BLOCK: usize = 512;
const MAX_ARCHIVE_BYTES: usize = 16 * 1024 * 1024;
const MAX_ENTRIES: usize = 4_096;

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg;
    reg(commands, &["tar"], Trust::Partial, cmd_tar);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Create,
    List,
    Extract,
}

struct Options {
    mode: Mode,
    archive: Option<String>,
    directory: Option<String>,
    gzip: bool,
    verbose: bool,
    files: Vec<String>,
}

#[derive(Clone)]
struct Entry {
    name: String,
    mode: u32,
    kind: EntryKind,
}

#[derive(Clone)]
enum EntryKind {
    File(Vec<u8>),
    Directory,
}

fn cmd_tar(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let options = match parse_options(args) {
        Ok(options) => options,
        Err(message) => {
            ewln(io.err, &format!("tar: {message}"));
            return 2;
        }
    };
    let base = options.directory.as_deref().map_or_else(
        || interp.cwd.clone(),
        |path| resolve_against(&interp.cwd, path),
    );
    if !interp.vfs.is_dir("/", &base) {
        ewln(io.err, &format!("tar: {base}: not a directory"));
        return 2;
    }
    match options.mode {
        Mode::Create => create(interp, &options, &base, io),
        Mode::List | Mode::Extract => read_archive(interp, &options, &base, io),
    }
}

fn create(interp: &mut CommandContext<'_>, options: &Options, base: &str, io: &mut Io) -> i32 {
    if options.files.is_empty() {
        ewln(io.err, "tar: refusing to create an empty archive");
        return 2;
    }
    let entries = match collect_entries(interp, base, &options.files) {
        Ok(entries) => entries,
        Err(message) => {
            ewln(io.err, &format!("tar: {message}"));
            return 2;
        }
    };
    let mut archive = match encode(interp, &entries) {
        Ok(archive) => archive,
        Err(message) => {
            ewln(io.err, &format!("tar: {message}"));
            return 2;
        }
    };
    if options.gzip {
        archive = match super::archives::gzip_bytes(interp, &archive) {
            Ok(archive) => archive,
            Err(message) => {
                ewln(io.err, &format!("tar: {message}"));
                return 2;
            }
        };
    }
    for entry in &entries {
        if options.verbose {
            wln(io.out, &entry.name);
        }
    }
    match options.archive.as_deref() {
        None | Some("-") => io.out.extend_from_slice(&archive),
        Some(path) => {
            let cwd = interp.cwd.clone();
            if let Err(error) = interp.vfs.write(&cwd, path, &archive, 0o644) {
                ewln(io.err, &format!("tar: {path}: {error}"));
                return 2;
            }
        }
    }
    0
}

fn collect_entries(
    interp: &mut CommandContext<'_>,
    base: &str,
    operands: &[String],
) -> Result<Vec<Entry>, String> {
    let mut paths = BTreeSet::new();
    for operand in operands {
        let absolute = resolve_against(base, operand);
        let metadata = interp
            .vfs
            .metadata("/", &absolute, false)
            .map_err(|error| format!("{operand}: {error}"))?;
        match metadata.kind {
            NodeKind::Dir => paths.extend(interp.vfs.walk(&absolute)),
            NodeKind::File(_) => {
                paths.insert(absolute);
            }
            NodeKind::Symlink(_) => {
                return Err(format!("{operand}: symbolic links are not supported"));
            }
        }
    }
    if paths.len() > MAX_ENTRIES {
        return Err(format!("archive exceeds the {MAX_ENTRIES}-entry limit"));
    }
    let mut entries = Vec::with_capacity(paths.len());
    for path in paths {
        let metadata = interp
            .vfs
            .metadata("/", &path, false)
            .map_err(|error| error.to_string())?;
        let name = archive_name(base, &path)?;
        let kind = match metadata.kind {
            NodeKind::Dir => EntryKind::Directory,
            NodeKind::File(data) => EntryKind::File(data),
            NodeKind::Symlink(_) => {
                return Err(format!("{path}: symbolic links are not supported"));
            }
        };
        entries.push(Entry {
            name,
            mode: metadata.mode,
            kind,
        });
    }
    Ok(entries)
}

fn archive_name(base: &str, path: &str) -> Result<String, String> {
    let relative = if base == "/" {
        path.trim_start_matches('/')
    } else {
        path.strip_prefix(&format!("{base}/"))
            .or_else(|| (path == base).then_some("."))
            .unwrap_or_else(|| path.trim_start_matches('/'))
    };
    validate_name(relative)?;
    Ok(relative.to_string())
}

fn encode(interp: &mut CommandContext<'_>, entries: &[Entry]) -> Result<Vec<u8>, String> {
    let estimated = entries.iter().try_fold(BLOCK * 2, |total, entry| {
        let data = match &entry.kind {
            EntryKind::File(data) => data.len(),
            EntryKind::Directory => 0,
        };
        total.checked_add(BLOCK)?.checked_add(round_block(data))
    });
    let estimated = estimated.ok_or_else(|| "archive is too large".to_string())?;
    if estimated > MAX_ARCHIVE_BYTES {
        return Err("archive exceeds the 16 MiB limit".to_string());
    }
    if !interp.reserve_memory(estimated as u64) {
        return Err("memory limit exceeded".to_string());
    }
    if !interp.charge_cpu(estimated as u64) {
        return Err("resource limit exceeded".to_string());
    }
    let mut output = Vec::with_capacity(estimated);
    for entry in entries {
        let data = match &entry.kind {
            EntryKind::File(data) => data.as_slice(),
            EntryKind::Directory => &[],
        };
        output.extend_from_slice(&header(entry, data.len())?);
        output.extend_from_slice(data);
        output.resize(output.len() + padding(data.len()), 0);
    }
    output.resize(output.len() + BLOCK * 2, 0);
    Ok(output)
}

fn header(entry: &Entry, size: usize) -> Result<[u8; BLOCK], String> {
    let mut header = [0u8; BLOCK];
    let mut name = entry.name.clone();
    if matches!(entry.kind, EntryKind::Directory) && !name.ends_with('/') {
        name.push('/');
    }
    write_bytes(&mut header[0..100], name.as_bytes(), "path")?;
    write_octal(&mut header[100..108], u64::from(entry.mode & 0o7777))?;
    write_octal(&mut header[108..116], 0)?;
    write_octal(&mut header[116..124], 0)?;
    write_octal(&mut header[124..136], size as u64)?;
    write_octal(&mut header[136..148], 0)?;
    header[148..156].fill(b' ');
    header[156] = if matches!(entry.kind, EntryKind::Directory) {
        b'5'
    } else {
        b'0'
    };
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    let checksum = header.iter().map(|byte| u64::from(*byte)).sum();
    write_checksum(&mut header[148..156], checksum)?;
    Ok(header)
}

fn read_archive(
    interp: &mut CommandContext<'_>,
    options: &Options,
    base: &str,
    io: &mut Io,
) -> i32 {
    let mut archive = match options.archive.as_deref() {
        None | Some("-") => io.stdin.clone(),
        Some(path) => match interp
            .vfs
            .read_limited(&interp.cwd, path, MAX_ARCHIVE_BYTES)
        {
            Ok(archive) => archive,
            Err(error) => {
                ewln(io.err, &format!("tar: {path}: {error}"));
                return 2;
            }
        },
    };
    if options.gzip {
        archive = match super::archives::gunzip_bytes(interp, &archive) {
            Ok(archive) => archive,
            Err(message) => {
                ewln(io.err, &format!("tar: {message}"));
                return 2;
            }
        };
    }
    let entries = match decode(interp, &archive) {
        Ok(entries) => entries,
        Err(message) => {
            ewln(io.err, &format!("tar: {message}"));
            return 2;
        }
    };
    if options.mode == Mode::List {
        for entry in entries {
            wln(io.out, &entry.name);
        }
        return 0;
    }
    let before = interp.vfs.clone();
    for entry in &entries {
        if options.verbose {
            wln(io.out, &entry.name);
        }
        let destination = resolve_against(base, &entry.name);
        if let Err(message) = validate_destination(interp, base, &destination) {
            interp.vfs = before;
            ewln(io.err, &format!("tar: {message}"));
            return 2;
        }
        let result = match &entry.kind {
            EntryKind::Directory => interp.vfs.mkdir_all("/", &destination),
            EntryKind::File(data) => {
                if let Some(parent) = parent_of(&destination) {
                    if let Err(error) = interp.vfs.mkdir_all("/", &parent) {
                        Err(error)
                    } else {
                        interp.vfs.put_file(&destination, data.clone(), entry.mode)
                    }
                } else {
                    interp.vfs.put_file(&destination, data.clone(), entry.mode)
                }
            }
        };
        if let Err(error) = result {
            interp.vfs = before;
            ewln(io.err, &format!("tar: {}: {error}", entry.name));
            return 2;
        }
    }
    0
}

fn validate_destination(
    interp: &CommandContext<'_>,
    base: &str,
    destination: &str,
) -> Result<(), String> {
    let parent = parent_of(destination).unwrap_or_else(|| "/".to_string());
    let real_parent = interp
        .vfs
        .realpath(&parent, true)
        .map_err(|error| error.to_string())?;
    if real_parent != parent {
        return Err(format!(
            "refusing to extract through symbolic-link parent: {destination}"
        ));
    }
    if (base == "/" && destination.starts_with('/'))
        || destination == base
        || destination.starts_with(&format!("{base}/"))
    {
        Ok(())
    } else {
        Err(format!("unsafe extraction destination: {destination}"))
    }
}

fn decode(interp: &mut CommandContext<'_>, archive: &[u8]) -> Result<Vec<Entry>, String> {
    if archive.len() > MAX_ARCHIVE_BYTES {
        return Err("archive exceeds the 16 MiB limit".to_string());
    }
    if !interp.reserve_memory(archive.len() as u64) || !interp.charge_cpu(archive.len() as u64) {
        return Err("resource limit exceeded".to_string());
    }
    let mut entries = Vec::new();
    let mut offset = 0usize;
    while offset
        .checked_add(BLOCK)
        .is_some_and(|end| end <= archive.len())
    {
        let header = &archive[offset..offset + BLOCK];
        if header.iter().all(|byte| *byte == 0) {
            return Ok(entries);
        }
        if entries.len() >= MAX_ENTRIES {
            return Err(format!("archive exceeds the {MAX_ENTRIES}-entry limit"));
        }
        validate_checksum(header)?;
        let name = field_string(&header[0..100])?;
        let prefix = field_string(&header[345..500])?;
        let name = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        let name = name.trim_end_matches('/').to_string();
        validate_name(&name)?;
        let size = parse_octal(&header[124..136])?;
        let size = usize::try_from(size).map_err(|_| "entry is too large".to_string())?;
        let start = offset + BLOCK;
        let end = start
            .checked_add(size)
            .ok_or_else(|| "entry size overflow".to_string())?;
        if end > archive.len() {
            return Err("truncated archive entry".to_string());
        }
        let mode = u32::try_from(parse_octal(&header[100..108])?)
            .map_err(|_| "invalid entry mode".to_string())?;
        match header[156] {
            0 | b'0' => entries.push(Entry {
                name,
                mode,
                kind: EntryKind::File(archive[start..end].to_vec()),
            }),
            b'5' if size == 0 => entries.push(Entry {
                name,
                mode,
                kind: EntryKind::Directory,
            }),
            b'g' => parse_pax_global(&archive[start..end])?,
            _ => return Err(format!("unsupported entry type for {name}")),
        }
        offset = start
            .checked_add(round_block(size))
            .ok_or_else(|| "archive offset overflow".to_string())?;
    }
    Err("archive is truncated or lacks an end marker".to_string())
}

/// Accept global metadata that cannot change extraction semantics. Path, size, or per-file
/// overrides need explicit handling rather than being ignored and extracting the wrong entry.
fn parse_pax_global(data: &[u8]) -> Result<(), String> {
    let mut offset = 0usize;
    while offset < data.len() {
        let remainder = &data[offset..];
        let separator = remainder
            .iter()
            .position(|byte| *byte == b' ')
            .ok_or_else(|| "malformed PAX record".to_string())?;
        let length = std::str::from_utf8(&remainder[..separator])
            .ok()
            .and_then(|text| text.parse::<usize>().ok())
            .ok_or_else(|| "malformed PAX record length".to_string())?;
        if length <= separator + 1 || length > remainder.len() {
            return Err("malformed PAX record length".to_string());
        }
        let record = &remainder[separator + 1..length];
        if record.last() != Some(&b'\n') {
            return Err("malformed PAX record".to_string());
        }
        let text = std::str::from_utf8(&record[..record.len() - 1])
            .map_err(|_| "PAX record is not UTF-8".to_string())?;
        let (key, _) = text
            .split_once('=')
            .ok_or_else(|| "malformed PAX record".to_string())?;
        if key != "comment" {
            return Err(format!("unsupported PAX global key: {key}"));
        }
        offset += length;
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.starts_with('/') || name.contains('\0') {
        return Err("archive contains an invalid path".to_string());
    }
    if name
        .split('/')
        .any(|component| component == ".." || component.is_empty())
    {
        return Err(format!("unsafe archive path: {name}"));
    }
    Ok(())
}

fn field_string(field: &[u8]) -> Result<String, String> {
    let end = field
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(field.len());
    std::str::from_utf8(&field[..end])
        .map(str::to_string)
        .map_err(|_| "archive path is not UTF-8".to_string())
}

fn parse_octal(field: &[u8]) -> Result<u64, String> {
    let text = std::str::from_utf8(field)
        .map_err(|_| "invalid tar numeric field".to_string())?
        .trim_matches(['\0', ' ']);
    if text.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(text, 8).map_err(|_| "invalid tar numeric field".to_string())
}

fn validate_checksum(header: &[u8]) -> Result<(), String> {
    let stored = parse_octal(&header[148..156])?;
    let actual = header
        .iter()
        .enumerate()
        .map(|(index, byte)| {
            if (148..156).contains(&index) {
                u64::from(b' ')
            } else {
                u64::from(*byte)
            }
        })
        .sum::<u64>();
    if stored == actual {
        Ok(())
    } else {
        Err("invalid header checksum".to_string())
    }
}

fn write_bytes(field: &mut [u8], value: &[u8], label: &str) -> Result<(), String> {
    if value.len() >= field.len() {
        return Err(format!("{label} is too long for the tar format"));
    }
    field[..value.len()].copy_from_slice(value);
    Ok(())
}

fn write_octal(field: &mut [u8], value: u64) -> Result<(), String> {
    let text = format!("{:0width$o}\0", value, width = field.len() - 1);
    if text.len() != field.len() {
        return Err("numeric value is too large for the tar format".to_string());
    }
    field.copy_from_slice(text.as_bytes());
    Ok(())
}

fn write_checksum(field: &mut [u8], value: u64) -> Result<(), String> {
    let text = format!("{:06o}\0 ", value);
    if text.len() != field.len() {
        return Err("checksum is too large for the tar format".to_string());
    }
    field.copy_from_slice(text.as_bytes());
    Ok(())
}

fn round_block(size: usize) -> usize {
    size.saturating_add(padding(size))
}

fn padding(size: usize) -> usize {
    (BLOCK - size % BLOCK) % BLOCK
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut mode = None;
    let mut archive = None;
    let mut directory = None;
    let mut gzip = false;
    let mut verbose = false;
    let mut files = Vec::new();
    let mut index = 0usize;
    let mut options = true;
    while index < args.len() {
        let argument = &args[index];
        if options && argument == "--" {
            options = false;
            index += 1;
            continue;
        }
        if options && matches!(argument.as_str(), "-f" | "--file" | "-C" | "--directory") {
            let value = args
                .get(index + 1)
                .ok_or_else(|| format!("{argument} requires an argument"))?
                .clone();
            if argument == "-f" || argument == "--file" {
                archive = Some(value);
            } else {
                directory = Some(value);
            }
            index += 2;
            continue;
        }
        if options && argument.starts_with("--") {
            match argument.as_str() {
                "--create" => set_mode(&mut mode, Mode::Create)?,
                "--extract" | "--get" => set_mode(&mut mode, Mode::Extract)?,
                "--list" => set_mode(&mut mode, Mode::List)?,
                "--gzip" => gzip = true,
                "--verbose" => verbose = true,
                _ => return Err(format!("unsupported option {argument}")),
            }
            index += 1;
            continue;
        }
        let cluster = if options && argument.starts_with('-') {
            Some(&argument[1..])
        } else if index == 0 && argument.chars().all(|flag| "cxtzvf".contains(flag)) {
            Some(argument.as_str())
        } else {
            None
        };
        if let Some(cluster) = cluster {
            let chars = cluster.chars().collect::<Vec<_>>();
            let mut position = 0usize;
            while position < chars.len() {
                match chars[position] {
                    'c' => set_mode(&mut mode, Mode::Create)?,
                    'x' => set_mode(&mut mode, Mode::Extract)?,
                    't' => set_mode(&mut mode, Mode::List)?,
                    'z' => gzip = true,
                    'v' => verbose = true,
                    'f' => {
                        if position + 1 != chars.len() {
                            return Err("archive name must follow f as a separate argument".into());
                        }
                        index += 1;
                        archive = Some(
                            args.get(index)
                                .ok_or_else(|| "f requires an archive name".to_string())?
                                .clone(),
                        );
                    }
                    flag => return Err(format!("unsupported option -{flag}")),
                }
                position += 1;
            }
            index += 1;
            continue;
        }
        files.push(argument.clone());
        options = false;
        index += 1;
    }
    Ok(Options {
        mode: mode.ok_or_else(|| "one of -c, -t, or -x is required".to_string())?,
        archive,
        directory,
        gzip,
        verbose,
        files,
    })
}

fn set_mode(current: &mut Option<Mode>, next: Mode) -> Result<(), String> {
    if current.is_some_and(|mode| mode != next) {
        Err("only one of -c, -t, or -x may be used".to_string())
    } else {
        *current = Some(next);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::parse_pax_global;

    #[test]
    fn pax_global_comments_are_accepted_but_semantic_overrides_are_not() {
        assert!(parse_pax_global(b"20 comment=upstream\n").is_ok());
        assert_eq!(
            parse_pax_global(b"14 path=wrong\n"),
            Err("unsupported PAX global key: path".to_string())
        );
        assert_eq!(
            parse_pax_global(b"99 comment=truncated\n"),
            Err("malformed PAX record length".to_string())
        );
    }
}
