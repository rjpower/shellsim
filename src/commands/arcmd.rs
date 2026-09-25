//! Bounded Unix archive construction for WebAssembly and TinyCC object files.
//!
//! The archive index names only defined, externally visible linking symbols. Parsing stays
//! inside the VFS and rejects object variants or archive names this small tool does not support.

use std::collections::HashMap;

use crate::commands::util::ewln;
use crate::commands::{CommandSpec, Io, Trust};
use crate::program::ProcessContext;
use crate::syscalls::{FileKind, System};

const MAX_ARCHIVE_BYTES: usize = 32 * 1024 * 1024;
const MAX_MEMBERS: usize = 4_096;
const MAGIC: &[u8] = b"!<arch>\n";

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    super::reg_system(commands, "/usr/bin/ar", Trust::Partial, cmd_ar);
    super::reg_system(commands, "/usr/bin/ranlib", Trust::Partial, cmd_ranlib);
}

struct Member {
    name: String,
    data: Vec<u8>,
    symbols: Vec<String>,
}

fn cmd_ar(context: &mut ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let Some((flags, rest)) = args.split_first() else {
        ewln(io.err, "ar: expected operation and archive");
        return 2;
    };
    let flags = flags.strip_prefix('-').unwrap_or(flags);
    let Some((archive, paths)) = rest.split_first() else {
        ewln(io.err, "ar: expected archive path");
        return 2;
    };
    let result = match flags {
        "s" => reindex(context.system, archive),
        "rc" | "cr" | "rcs" | "crs" => replace(context.system, archive, paths),
        _ => Err(format!("unsupported operation '{flags}'")),
    };
    match result {
        Ok(()) => 0,
        Err(error) => {
            ewln(io.err, &format!("ar: {error}"));
            2
        }
    }
}

fn cmd_ranlib(context: &mut ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let [archive] = args else {
        ewln(io.err, "ranlib: expected one archive path");
        return 2;
    };
    match reindex(context.system, archive) {
        Ok(()) => 0,
        Err(error) => {
            ewln(io.err, &format!("ranlib: {error}"));
            2
        }
    }
}

fn replace(system: &mut dyn System, archive: &str, paths: &[String]) -> Result<(), String> {
    if paths.is_empty() {
        return Err("expected at least one object".into());
    }
    let cwd = system.cwd().to_string();
    let mut members = if matches!(
        system.metadata(&cwd, archive, true).map(|info| info.kind),
        Ok(FileKind::File)
    ) {
        let bytes = read_input(system, archive)?;
        decode_archive(&bytes)?
    } else {
        Vec::new()
    };
    for path in paths {
        let name = path.rsplit('/').next().unwrap_or("");
        if name.is_empty() || name.len() > 15 || !name.is_ascii() {
            return Err(format!("unsupported member name '{name}'"));
        }
        let data = read_input(system, path)?;
        let symbols = object_symbols(&data)?;
        let member = Member {
            name: name.into(),
            data,
            symbols,
        };
        if let Some(index) = members.iter().position(|old| old.name == name) {
            members[index] = member;
        } else {
            members.push(member);
        }
        if members.len() > MAX_MEMBERS {
            return Err("too many archive members".into());
        }
    }
    save(system, archive, &members)
}

fn reindex(system: &mut dyn System, archive: &str) -> Result<(), String> {
    let bytes = read_input(system, archive)?;
    let members = decode_archive(&bytes)?;
    save(system, archive, &members)
}

fn save(system: &mut dyn System, archive: &str, members: &[Member]) -> Result<(), String> {
    let work = members.iter().try_fold(0usize, |total, member| {
        total
            .checked_add(member.data.len())
            .ok_or("archive size overflow")
    })?;
    if work > MAX_ARCHIVE_BYTES || !system.charge_cpu(work as u64) {
        return Err("archive resource limit exceeded".into());
    }
    let bytes = encode_archive(members)?;
    let cwd = system.cwd().to_string();
    system
        .write_file(&cwd, archive, &bytes, 0o644)
        .map_err(|error| error.to_string())
}

fn read_input(system: &mut dyn System, path: &str) -> Result<Vec<u8>, String> {
    let cwd = system.cwd().to_string();
    let size = system
        .metadata(&cwd, path, true)
        .map(|info| info.size)
        .map_err(|error| error.to_string())?;
    if size > MAX_ARCHIVE_BYTES as u64 || !system.charge_cpu(size) {
        return Err("archive resource limit exceeded".into());
    }
    system
        .read_file_limited(&cwd, path, MAX_ARCHIVE_BYTES)
        .map_err(|error| error.to_string())
}

fn checked_end(start: usize, len: usize, limit: usize) -> Result<usize, String> {
    let end = start.checked_add(len).ok_or("archive size overflow")?;
    if end > limit {
        return Err("archive exceeds size limit".into());
    }
    Ok(end)
}

fn encode_archive(members: &[Member]) -> Result<Vec<u8>, String> {
    let count = members.iter().try_fold(0usize, |total, member| {
        total
            .checked_add(member.symbols.len())
            .ok_or("too many symbols".to_string())
    })?;
    let names_len =
        members
            .iter()
            .flat_map(|member| &member.symbols)
            .try_fold(0usize, |total, name| {
                total
                    .checked_add(name.len() + 1)
                    .ok_or("symbol table too large".to_string())
            })?;
    let index_len = 4usize
        .checked_add(count.checked_mul(4).ok_or("symbol table too large")?)
        .and_then(|size| size.checked_add(names_len))
        .ok_or("symbol table too large")?;
    let mut offset = checked_end(
        MAGIC.len() + 60,
        index_len + index_len % 2,
        MAX_ARCHIVE_BYTES,
    )?;
    let mut offsets = Vec::with_capacity(members.len());
    for member in members {
        offsets.push(u32::try_from(offset).map_err(|_| "archive offset overflow")?);
        offset = checked_end(
            offset + 60,
            member.data.len() + member.data.len() % 2,
            MAX_ARCHIVE_BYTES,
        )?;
    }
    let mut out = Vec::with_capacity(offset);
    out.extend_from_slice(MAGIC);
    append_header(&mut out, "/", index_len)?;
    out.extend_from_slice(
        &u32::try_from(count)
            .map_err(|_| "too many symbols")?
            .to_be_bytes(),
    );
    for (member, offset) in members.iter().zip(&offsets) {
        for _ in &member.symbols {
            out.extend_from_slice(&offset.to_be_bytes());
        }
    }
    for member in members {
        for name in &member.symbols {
            out.extend_from_slice(name.as_bytes());
            out.push(0);
        }
    }
    if out.len() % 2 != 0 {
        out.push(b'\n');
    }
    for member in members {
        append_header(&mut out, &format!("{}/", member.name), member.data.len())?;
        out.extend_from_slice(&member.data);
        if out.len() % 2 != 0 {
            out.push(b'\n');
        }
    }
    Ok(out)
}

fn append_header(out: &mut Vec<u8>, name: &str, size: usize) -> Result<(), String> {
    if name.len() > 16 {
        return Err("member name too long".into());
    }
    let fields = [(name, 16), ("0", 12), ("0", 6), ("0", 6), ("100644", 8)];
    for (value, width) in fields {
        out.extend_from_slice(value.as_bytes());
        out.extend(std::iter::repeat_n(b' ', width - value.len()));
    }
    let size = size.to_string();
    if size.len() > 10 {
        return Err("archive member too large".into());
    }
    out.extend_from_slice(size.as_bytes());
    out.extend(std::iter::repeat_n(b' ', 10 - size.len()));
    out.extend_from_slice(b"`\n");
    Ok(())
}

fn decode_archive(bytes: &[u8]) -> Result<Vec<Member>, String> {
    if bytes.len() > MAX_ARCHIVE_BYTES || !bytes.starts_with(MAGIC) {
        return Err("unsupported or oversized archive".into());
    }
    let mut pos = MAGIC.len();
    let mut members = Vec::new();
    while pos < bytes.len() {
        let header_end = checked_end(pos, 60, bytes.len())?;
        let header = &bytes[pos..header_end];
        if &header[58..60] != b"`\n" {
            return Err("malformed archive header".into());
        }
        let size = std::str::from_utf8(&header[48..58])
            .map_err(|_| "invalid member size")?
            .trim()
            .parse::<usize>()
            .map_err(|_| "invalid member size")?;
        let data_end = checked_end(header_end, size, bytes.len())?;
        let name = std::str::from_utf8(&header[..16])
            .map_err(|_| "invalid member name")?
            .trim_end()
            .trim_end_matches('/');
        if !name.is_empty() {
            if name.len() > 15 || name == "/" {
                return Err("unsupported archive member name".into());
            }
            let data = bytes[header_end..data_end].to_vec();
            let symbols = object_symbols(&data)?;
            members.push(Member {
                name: name.into(),
                data,
                symbols,
            });
            if members.len() > MAX_MEMBERS {
                return Err("too many archive members".into());
            }
        }
        pos = checked_end(data_end, size % 2, bytes.len())?;
    }
    Ok(members)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(len).ok_or("object offset overflow")?;
        let bytes = self
            .bytes
            .get(self.pos..end)
            .ok_or("truncated Wasm object")?;
        self.pos = end;
        Ok(bytes)
    }

    fn byte(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn uleb(&mut self) -> Result<usize, String> {
        let mut value = 0usize;
        for shift in (0..35).step_by(7) {
            let byte = self.byte()?;
            value |= usize::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("invalid Wasm LEB128".into())
    }

    fn string(&mut self) -> Result<&'a str, String> {
        let len = self.uleb()?;
        std::str::from_utf8(self.take(len)?).map_err(|_| "invalid Wasm symbol name".into())
    }

    fn rest(&mut self, len: usize) -> Result<Cursor<'a>, String> {
        Ok(Cursor::new(self.take(len)?))
    }
}

fn object_symbols(bytes: &[u8]) -> Result<Vec<String>, String> {
    if bytes.starts_with(b"\x7fELF") {
        return elf_symbols(bytes);
    }
    if !bytes.starts_with(b"\0asm\x01\0\0\0") {
        return Err("member is not a supported WebAssembly object".into());
    }
    let mut module = Cursor::new(&bytes[8..]);
    while module.pos < module.bytes.len() {
        let kind = module.byte()?;
        let len = module.uleb()?;
        let mut section = module.rest(len)?;
        if kind == 0 && section.string()? == "linking" {
            let _version = section.uleb()?;
            while section.pos < section.bytes.len() {
                let kind = section.byte()?;
                let len = section.uleb()?;
                let mut subsection = section.rest(len)?;
                if kind == 8 {
                    return parse_symbols(&mut subsection);
                }
            }
        }
    }
    Err("Wasm object has no linking symbol table".into())
}

fn elf_symbols(bytes: &[u8]) -> Result<Vec<String>, String> {
    // TinyCC's wasm32 target emits ELF32 relocatable objects with EM_WEBASSEMBLY.
    if bytes.len() < 52
        || bytes[4..7] != [1, 1, 1]
        || read_u16(bytes, 16)? != 1
        || read_u16(bytes, 18)? != 0x4157
    {
        return Err("unsupported ELF object".into());
    }
    let section_offset = read_u32(bytes, 32)? as usize;
    let section_size = usize::from(read_u16(bytes, 46)?);
    let section_count = usize::from(read_u16(bytes, 48)?);
    if section_size < 40 || section_count > 4_096 {
        return Err("unsupported ELF sections".into());
    }
    let sections_len = section_size
        .checked_mul(section_count)
        .ok_or("ELF section table overflow")?;
    checked_slice(bytes, section_offset, sections_len)?;
    let mut names = Vec::new();
    for index in 0..section_count {
        let section = checked_slice(bytes, section_offset + index * section_size, section_size)?;
        if read_u32(section, 4)? != 2 {
            continue;
        }
        let entries = checked_slice(
            bytes,
            read_u32(section, 16)? as usize,
            read_u32(section, 20)? as usize,
        )?;
        let entry_size = read_u32(section, 36)? as usize;
        if entry_size < 16
            || entries.len() % entry_size != 0
            || entries.len() / entry_size > 100_000
        {
            return Err("invalid ELF symbol table".into());
        }
        let strings_index = read_u32(section, 24)? as usize;
        if strings_index >= section_count {
            return Err("invalid ELF string table index".into());
        }
        let strings_section = checked_slice(
            bytes,
            section_offset + strings_index * section_size,
            section_size,
        )?;
        if read_u32(strings_section, 4)? != 3 {
            return Err("invalid ELF string table".into());
        }
        let strings = checked_slice(
            bytes,
            read_u32(strings_section, 16)? as usize,
            read_u32(strings_section, 20)? as usize,
        )?;
        for entry in entries.chunks_exact(entry_size) {
            let binding = entry[12] >> 4;
            let defined = read_u16(entry, 14)? != 0;
            if !defined || (binding != 1 && binding != 2) {
                continue;
            }
            let offset = read_u32(entry, 0)? as usize;
            let name = strings.get(offset..).ok_or("invalid ELF symbol name")?;
            let end = name
                .iter()
                .position(|byte| *byte == 0)
                .ok_or("unterminated ELF symbol name")?;
            if end > 0 {
                names.push(
                    std::str::from_utf8(&name[..end])
                        .map_err(|_| "non-UTF-8 ELF symbol name")?
                        .to_owned(),
                );
            }
        }
    }
    Ok(names)
}

fn checked_slice(bytes: &[u8], start: usize, len: usize) -> Result<&[u8], String> {
    let end = start.checked_add(len).ok_or("ELF object offset overflow")?;
    bytes
        .get(start..end)
        .ok_or_else(|| "truncated ELF object".into())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, String> {
    Ok(u16::from_le_bytes(
        checked_slice(bytes, offset, 2)?.try_into().unwrap(),
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    Ok(u32::from_le_bytes(
        checked_slice(bytes, offset, 4)?.try_into().unwrap(),
    ))
}

fn parse_symbols(table: &mut Cursor<'_>) -> Result<Vec<String>, String> {
    let count = table.uleb()?;
    if count > 100_000 {
        return Err("too many Wasm symbols".into());
    }
    let mut names = Vec::new();
    for _ in 0..count {
        let kind = table.byte()?;
        let flags = table.uleb()?;
        let defined = flags & 0x10 == 0;
        let global = flags & 0x3 != 2;
        let name = match kind {
            0 | 2 | 4 | 5 => {
                let _index = table.uleb()?;
                if defined || flags & 0x40 != 0 {
                    Some(table.string()?)
                } else {
                    None
                }
            }
            1 => {
                let name = table.string()?;
                if defined {
                    let _segment = table.uleb()?;
                    let _offset = table.uleb()?;
                    let _size = table.uleb()?;
                }
                Some(name)
            }
            3 => {
                let _index = table.uleb()?;
                None
            }
            _ => return Err("unsupported Wasm symbol kind".into()),
        };
        if defined && global {
            if let Some(name) = name {
                names.push(name.to_string());
            }
        }
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OBJECT: &[u8] = b"\0asm\x01\0\0\0\0\x13\x07linking\x02\x08\x08\x01\0\0\0\x03foo";

    #[test]
    fn indexed_archive_round_trip() {
        let member = Member {
            name: "foo.o".into(),
            data: OBJECT.to_vec(),
            symbols: object_symbols(OBJECT).unwrap(),
        };
        assert_eq!(member.symbols, ["foo"]);
        let archive = encode_archive(&[member]).unwrap();
        assert!(archive.starts_with(b"!<arch>\n/"));
        let decoded = decode_archive(&archive).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].name, "foo.o");
        assert_eq!(decoded[0].symbols, ["foo"]);
    }

    #[test]
    fn malformed_object_is_rejected() {
        assert!(object_symbols(b"not wasm").is_err());
        assert!(object_symbols(&OBJECT[..OBJECT.len() - 1]).is_err());
        assert!(object_symbols(b"\x7fELF").is_err());
    }

    #[test]
    fn tinycc_elf32_symbols_are_indexed() {
        let mut object = vec![0; 52 + 3 * 40 + 2 * 16 + 5];
        object[..7].copy_from_slice(b"\x7fELF\x01\x01\x01");
        object[16..18].copy_from_slice(&1_u16.to_le_bytes());
        object[18..20].copy_from_slice(&0x4157_u16.to_le_bytes());
        object[32..36].copy_from_slice(&52_u32.to_le_bytes());
        object[46..48].copy_from_slice(&40_u16.to_le_bytes());
        object[48..50].copy_from_slice(&3_u16.to_le_bytes());
        let symbols = 52 + 40;
        object[symbols + 4..symbols + 8].copy_from_slice(&2_u32.to_le_bytes());
        object[symbols + 16..symbols + 20].copy_from_slice(&172_u32.to_le_bytes());
        object[symbols + 20..symbols + 24].copy_from_slice(&32_u32.to_le_bytes());
        object[symbols + 24..symbols + 28].copy_from_slice(&2_u32.to_le_bytes());
        object[symbols + 36..symbols + 40].copy_from_slice(&16_u32.to_le_bytes());
        let strings = 52 + 80;
        object[strings + 4..strings + 8].copy_from_slice(&3_u32.to_le_bytes());
        object[strings + 16..strings + 20].copy_from_slice(&204_u32.to_le_bytes());
        object[strings + 20..strings + 24].copy_from_slice(&5_u32.to_le_bytes());
        object[188..192].copy_from_slice(&1_u32.to_le_bytes());
        object[200] = 0x10;
        object[202..204].copy_from_slice(&1_u16.to_le_bytes());
        object[204..].copy_from_slice(b"\0foo\0");

        assert_eq!(object_symbols(&object).unwrap(), ["foo"]);
        assert!(object_symbols(&object[..object.len() - 1]).is_err());
    }
}
