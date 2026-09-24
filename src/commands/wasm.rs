//! Bounded WASI preview1 execution against shellsim's virtual command boundary.
//!
//! The host functions expose buffered streams, process metadata, the virtual clock, and
//! bounded regular-file access. Unknown imports fail instantiation rather than acquiring
//! ambient host capabilities. Live pipe suspension remains outside this first slice.

use std::collections::BTreeSet;
use std::fmt;
use wasmtime::{
    Caller, Config, Engine, Error, Extern, Linker, Memory, Module, Store, StoreLimits,
    StoreLimitsBuilder,
};

use crate::descriptors::DescriptorError;
use crate::interp::Interp;
use crate::syscalls::{self, OpenFile, SyscallError};
use crate::vfs::{resolve_against, NodeKind, VfsError};

use super::{util::ewln, CommandPoll};

const MAX_WASM_BYTES: usize = 8 * 1024 * 1024;
const MAX_WASM_MEMORY: usize = 16 * 1024 * 1024;
const MAX_IO_BYTES: usize = 1024 * 1024;
// Wasm instructions are cheaper than a modeled CPU unit. This keeps a compiled byte-oriented
// utility usable on ordinary input without relaxing the host's execution bound.
const WASM_FUEL_PER_CPU_UNIT: u64 = 10;
const ERRNO_SUCCESS: i32 = 0;
const ERRNO_BADF: i32 = 8;
const ERRNO_FAULT: i32 = 21;
const ERRNO_INVAL: i32 = 28;
const ERRNO_EXIST: i32 = 20;
const ERRNO_ISDIR: i32 = 31;
const ERRNO_NOENT: i32 = 44;
const ERRNO_NOTDIR: i32 = 54;
const ERRNO_NOTEMPTY: i32 = 55;
const ERRNO_PERM: i32 = 63;

struct Host {
    interp: Interp,
    cwd: String,
    args: Vec<Vec<u8>>,
    environment: Vec<Vec<u8>>,
    stdin: Vec<u8>,
    stdin_offset: usize,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    open_files: BTreeSet<i32>,
    append_files: BTreeSet<i32>,
    random_state: u64,
    limits: StoreLimits,
}

#[derive(Debug)]
struct GuestExit(i32);

impl fmt::Display for GuestExit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "guest exited with status {}", self.0)
    }
}

impl std::error::Error for GuestExit {}

fn vfs_errno(error: &VfsError) -> i32 {
    match error {
        VfsError::NotFound(_) => ERRNO_NOENT,
        VfsError::Exists(_) => ERRNO_EXIST,
        VfsError::IsADir(_) => ERRNO_ISDIR,
        VfsError::NotADir(_) => ERRNO_NOTDIR,
        VfsError::NotEmpty(_) => ERRNO_NOTEMPTY,
        VfsError::ReadOnly(_) => ERRNO_PERM,
        _ => ERRNO_INVAL,
    }
}

fn syscall_errno(error: &SyscallError) -> i32 {
    match error {
        SyscallError::File(error) => vfs_errno(error),
        SyscallError::Descriptor(DescriptorError::InvalidFd | DescriptorError::WrongAccess) => {
            ERRNO_BADF
        }
        SyscallError::Descriptor(_) => ERRNO_INVAL,
        SyscallError::InvalidArgument => ERRNO_INVAL,
        SyscallError::IsDirectory => ERRNO_ISDIR,
        SyscallError::Permission => ERRNO_PERM,
    }
}

fn guest_file(caller: &Caller<'_, Host>, fd: i32) -> Result<crate::descriptors::FileState, i32> {
    if !caller.data().open_files.contains(&fd) {
        return Err(ERRNO_BADF);
    }
    syscalls::file_state(&caller.data().interp, fd).map_err(|error| syscall_errno(&error))
}

fn memory(caller: &mut Caller<'_, Host>) -> Option<Memory> {
    caller.get_export("memory").and_then(Extern::into_memory)
}

fn read_u32(caller: &mut Caller<'_, Host>, address: u32) -> Option<u32> {
    let mut bytes = [0; 4];
    memory(caller)?
        .read(caller, address as usize, &mut bytes)
        .ok()?;
    Some(u32::from_le_bytes(bytes))
}

fn write_u32(caller: &mut Caller<'_, Host>, address: u32, value: u32) -> bool {
    memory(caller).is_some_and(|memory| {
        memory
            .write(caller, address as usize, &value.to_le_bytes())
            .is_ok()
    })
}

fn write_u64(caller: &mut Caller<'_, Host>, address: u32, value: u64) -> bool {
    memory(caller).is_some_and(|memory| {
        memory
            .write(caller, address as usize, &value.to_le_bytes())
            .is_ok()
    })
}

fn strings_get(caller: &mut Caller<'_, Host>, pointers: u32, buffer: u32, arguments: bool) -> i32 {
    let values = if arguments {
        &caller.data().args
    } else {
        &caller.data().environment
    };
    let values = values.clone();
    let Some(memory) = memory(caller) else {
        return ERRNO_FAULT;
    };
    let mut offset = buffer as usize;
    for (index, value) in values.iter().enumerate() {
        let Some(pointer_offset) = (pointers as usize).checked_add(index.saturating_mul(4)) else {
            return ERRNO_FAULT;
        };
        let Ok(address) = u32::try_from(offset) else {
            return ERRNO_FAULT;
        };
        if memory
            .write(&mut *caller, pointer_offset, &address.to_le_bytes())
            .is_err()
            || memory.write(&mut *caller, offset, value).is_err()
            || memory
                .write(&mut *caller, offset + value.len(), &[0])
                .is_err()
        {
            return ERRNO_FAULT;
        }
        offset = offset.saturating_add(value.len()).saturating_add(1);
    }
    ERRNO_SUCCESS
}

fn strings_sizes(caller: &mut Caller<'_, Host>, count: u32, size: u32, arguments: bool) -> i32 {
    let values = if arguments {
        &caller.data().args
    } else {
        &caller.data().environment
    };
    let Ok(number) = u32::try_from(values.len()) else {
        return ERRNO_INVAL;
    };
    let Ok(bytes) = u32::try_from(
        values
            .iter()
            .map(|v| v.len().saturating_add(1))
            .sum::<usize>(),
    ) else {
        return ERRNO_INVAL;
    };
    if write_u32(caller, count, number) && write_u32(caller, size, bytes) {
        ERRNO_SUCCESS
    } else {
        ERRNO_FAULT
    }
}

struct PathOpen {
    directory: u32,
    path_pointer: u32,
    path_length: u32,
    oflags: u32,
    rights_base: u64,
    fdflags: u32,
    result: u32,
}

fn read_path(caller: &mut Caller<'_, Host>, pointer: u32, length: u32) -> Result<String, i32> {
    if length == 0 || length as usize > 4096 {
        return Err(ERRNO_INVAL);
    }
    let memory = memory(caller).ok_or(ERRNO_FAULT)?;
    let mut bytes = vec![0; length as usize];
    memory
        .read(caller, pointer as usize, &mut bytes)
        .map_err(|_| ERRNO_FAULT)?;
    let path = std::str::from_utf8(&bytes).map_err(|_| ERRNO_INVAL)?;
    if path.contains('\0') {
        return Err(ERRNO_INVAL);
    }
    Ok(path.to_string())
}

fn preopen_base(caller: &Caller<'_, Host>, fd: u32) -> Result<String, i32> {
    match fd {
        3 => Ok(caller.data().cwd.clone()),
        4 => Ok("/".to_string()),
        _ => Err(ERRNO_BADF),
    }
}

fn path_open(mut caller: Caller<'_, Host>, request: PathOpen) -> i32 {
    let cwd = match preopen_base(&caller, request.directory) {
        Ok(cwd) => cwd,
        Err(error) => return error,
    };
    if request.oflags & !0b1111 != 0 || request.fdflags & !1 != 0 {
        return ERRNO_INVAL;
    }
    if request.oflags & 2 != 0 {
        return ERRNO_NOTDIR;
    }
    let path = match read_path(&mut caller, request.path_pointer, request.path_length) {
        Ok(path) => path,
        Err(error) => return error,
    };
    let readable = request.rights_base & 2 != 0;
    let writable = request.rights_base & 64 != 0;
    let options = OpenFile {
        readable,
        writable,
        create: request.oflags & 1 != 0,
        exclusive: request.oflags & 4 != 0,
        truncate: request.oflags & 8 != 0,
        append: request.fdflags & 1 != 0,
    };
    let fd = match syscalls::open_file(&mut caller.data_mut().interp, &cwd, &path, options) {
        Ok(fd) => fd,
        Err(error) => return syscall_errno(&error),
    };
    if !write_u32(&mut caller, request.result, fd as u32) {
        let _ = syscalls::close(&mut caller.data_mut().interp, fd);
        return ERRNO_FAULT;
    }
    let host = caller.data_mut();
    host.open_files.insert(fd);
    if options.append {
        host.append_files.insert(fd);
    }
    ERRNO_SUCCESS
}

// WASI Preview 1 has no permission-changing operation. Toolchains use this bounded extension to
// mark a linked module executable without receiving access to host filesystem metadata.
fn path_chmod(mut caller: Caller<'_, Host>, pointer: u32, length: u32, mode: u32) -> i32 {
    let path = match read_path(&mut caller, pointer, length) {
        Ok(path) => path,
        Err(error) => return error,
    };
    let cwd = caller.data().cwd.clone();
    match syscalls::chmod(&mut caller.data_mut().interp, &cwd, &path, mode) {
        Ok(()) => ERRNO_SUCCESS,
        Err(error) => syscall_errno(&error),
    }
}

fn fd_prestat_get(mut caller: Caller<'_, Host>, fd: u32, pointer: u32) -> i32 {
    if !matches!(fd, 3 | 4) {
        return ERRNO_BADF;
    }
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    let mut value = [0; 8];
    value[4..8].copy_from_slice(&1_u32.to_le_bytes());
    if memory.write(&mut caller, pointer as usize, &value).is_ok() {
        ERRNO_SUCCESS
    } else {
        ERRNO_FAULT
    }
}

fn fd_prestat_dir_name(mut caller: Caller<'_, Host>, fd: u32, pointer: u32, length: u32) -> i32 {
    if !matches!(fd, 3 | 4) {
        return ERRNO_BADF;
    }
    if length < 1 {
        return ERRNO_INVAL;
    }
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    let name = if fd == 3 { b"." } else { b"/" };
    if memory.write(&mut caller, pointer as usize, name).is_ok() {
        ERRNO_SUCCESS
    } else {
        ERRNO_FAULT
    }
}

fn fd_fdstat_get(mut caller: Caller<'_, Host>, fd: u32, pointer: u32) -> i32 {
    let mut value = [0; 24];
    let rights = match fd {
        0 => {
            value[0] = 2;
            2_u64
        }
        1 | 2 => {
            value[0] = 2;
            64_u64
        }
        3 | 4 => {
            value[0] = 3;
            u64::MAX
        }
        _ => match guest_file(&caller, fd as i32) {
            Ok(file) => {
                value[0] = 4;
                (if file.readable { 2 } else { 0 }) | (if file.writable { 64 } else { 0 })
            }
            Err(error) => return error,
        },
    };
    value[8..16].copy_from_slice(&rights.to_le_bytes());
    value[16..24].copy_from_slice(&rights.to_le_bytes());
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    if memory.write(&mut caller, pointer as usize, &value).is_ok() {
        ERRNO_SUCCESS
    } else {
        ERRNO_FAULT
    }
}

fn fd_close(mut caller: Caller<'_, Host>, fd: u32) -> i32 {
    let fd = fd as i32;
    if !caller.data_mut().open_files.remove(&fd) {
        return ERRNO_BADF;
    }
    caller.data_mut().append_files.remove(&fd);
    match syscalls::close(&mut caller.data_mut().interp, fd) {
        Ok(()) => ERRNO_SUCCESS,
        Err(error) => syscall_errno(&error),
    }
}

fn fd_seek(mut caller: Caller<'_, Host>, fd: u32, delta: i64, whence: u32, result: u32) -> i32 {
    if let Err(error) = guest_file(&caller, fd as i32) {
        return error;
    }
    let position = match syscalls::seek(&mut caller.data_mut().interp, fd as i32, delta, whence) {
        Ok(position) => position,
        Err(error) => return syscall_errno(&error),
    };
    if !write_u64(&mut caller, result, position) {
        return ERRNO_FAULT;
    }
    ERRNO_SUCCESS
}

fn fd_tell(mut caller: Caller<'_, Host>, fd: u32, result: u32) -> i32 {
    let file = match guest_file(&caller, fd as i32) {
        Ok(file) => file,
        Err(error) => return error,
    };
    let position = file.cursor;
    if write_u64(&mut caller, result, position) {
        ERRNO_SUCCESS
    } else {
        ERRNO_FAULT
    }
}

fn filestat(node: &crate::vfs::Node) -> [u8; 64] {
    let mut value = [0; 64];
    value[16] = match &node.kind {
        NodeKind::Dir => 3,
        NodeKind::File(_) | NodeKind::NativeExecutable(_) => 4,
        NodeKind::Symlink(_) => 7,
    };
    value[24..32].copy_from_slice(&1_u64.to_le_bytes());
    let size = match &node.kind {
        NodeKind::File(bytes) => bytes.len() as u64,
        _ => 0,
    };
    value[32..40].copy_from_slice(&size.to_le_bytes());
    let mtime = node.mtime.saturating_mul(1_000_000);
    value[40..48].copy_from_slice(&mtime.to_le_bytes());
    value[48..56].copy_from_slice(&mtime.to_le_bytes());
    value[56..64].copy_from_slice(&mtime.to_le_bytes());
    value
}

fn fd_filestat_get(mut caller: Caller<'_, Host>, fd: u32, result: u32) -> i32 {
    let file = if fd >= 5 {
        match guest_file(&caller, fd as i32) {
            Ok(file) => Some(file),
            Err(error) => return error,
        }
    } else {
        None
    };
    let node = match file.as_ref().and_then(|file| file.orphan) {
        Some(id) => caller.data().interp.vfs.orphan_metadata(id),
        None => {
            let path = match (fd, file) {
                (3, _) => caller.data().cwd.clone(),
                (4, _) => "/".to_string(),
                (_, Some(file)) => file.path,
                _ => return ERRNO_BADF,
            };
            caller.data().interp.vfs.metadata("/", &path, true)
        }
    };
    let node = match node {
        Ok(node) => node,
        Err(error) => return vfs_errno(&error),
    };
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    if memory
        .write(&mut caller, result as usize, &filestat(&node))
        .is_ok()
    {
        ERRNO_SUCCESS
    } else {
        ERRNO_FAULT
    }
}

fn path_filestat_get(
    mut caller: Caller<'_, Host>,
    fd: u32,
    flags: u32,
    pointer: u32,
    length: u32,
    result: u32,
) -> i32 {
    let cwd = match preopen_base(&caller, fd) {
        Ok(cwd) => cwd,
        Err(error) => return error,
    };
    let path = match read_path(&mut caller, pointer, length) {
        Ok(path) => path,
        Err(error) => return error,
    };
    let path = resolve_against(&cwd, &path);
    let node = match caller
        .data()
        .interp
        .vfs
        .metadata("/", &path, flags & 1 != 0)
    {
        Ok(node) => node,
        Err(error) => return vfs_errno(&error),
    };
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    if memory
        .write(&mut caller, result as usize, &filestat(&node))
        .is_ok()
    {
        ERRNO_SUCCESS
    } else {
        ERRNO_FAULT
    }
}

fn path_unlink_file(mut caller: Caller<'_, Host>, fd: u32, pointer: u32, length: u32) -> i32 {
    let cwd = match preopen_base(&caller, fd) {
        Ok(cwd) => cwd,
        Err(error) => return error,
    };
    let path = match read_path(&mut caller, pointer, length) {
        Ok(path) => path,
        Err(error) => return error,
    };
    match syscalls::unlink(&mut caller.data_mut().interp, &cwd, &path) {
        Ok(()) => ERRNO_SUCCESS,
        Err(error) => syscall_errno(&error),
    }
}

fn path_create_directory(mut caller: Caller<'_, Host>, fd: u32, pointer: u32, length: u32) -> i32 {
    let cwd = match preopen_base(&caller, fd) {
        Ok(cwd) => cwd,
        Err(error) => return error,
    };
    let path = match read_path(&mut caller, pointer, length) {
        Ok(path) => path,
        Err(error) => return error,
    };
    match syscalls::mkdir(&mut caller.data_mut().interp, &cwd, &path) {
        Ok(()) => ERRNO_SUCCESS,
        Err(error) => syscall_errno(&error),
    }
}

fn path_remove_directory(mut caller: Caller<'_, Host>, fd: u32, pointer: u32, length: u32) -> i32 {
    let cwd = match preopen_base(&caller, fd) {
        Ok(cwd) => cwd,
        Err(error) => return error,
    };
    let path = match read_path(&mut caller, pointer, length) {
        Ok(path) => path,
        Err(error) => return error,
    };
    match syscalls::rmdir(&mut caller.data_mut().interp, &cwd, &path) {
        Ok(()) => ERRNO_SUCCESS,
        Err(error) => syscall_errno(&error),
    }
}

fn path_rename(
    mut caller: Caller<'_, Host>,
    old_fd: u32,
    old_pointer: u32,
    old_length: u32,
    new_fd: u32,
    new_pointer: u32,
    new_length: u32,
) -> i32 {
    if !matches!(old_fd, 3 | 4) || !matches!(new_fd, 3 | 4) {
        return ERRNO_BADF;
    }
    let old_base = preopen_base(&caller, old_fd).expect("preopen fd was checked");
    let new_base = preopen_base(&caller, new_fd).expect("preopen fd was checked");
    let old_path = match read_path(&mut caller, old_pointer, old_length) {
        Ok(path) => path,
        Err(error) => return error,
    };
    let new_path = match read_path(&mut caller, new_pointer, new_length) {
        Ok(path) => path,
        Err(error) => return error,
    };
    let old_path = resolve_against(&old_base, &old_path);
    let new_path = resolve_against(&new_base, &new_path);
    match syscalls::rename(&mut caller.data_mut().interp, "/", &old_path, &new_path) {
        Ok(()) => ERRNO_SUCCESS,
        Err(error) => syscall_errno(&error),
    }
}

fn random_get(mut caller: Caller<'_, Host>, pointer: u32, length: u32) -> i32 {
    if length as usize > MAX_IO_BYTES {
        return ERRNO_INVAL;
    }
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    let mut bytes = vec![0; length as usize];
    for byte in &mut bytes {
        let state = caller
            .data()
            .random_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        caller.data_mut().random_state = state;
        *byte = (state >> 32) as u8;
    }
    if memory.write(&mut caller, pointer as usize, &bytes).is_ok() {
        ERRNO_SUCCESS
    } else {
        ERRNO_FAULT
    }
}

fn fd_write(mut caller: Caller<'_, Host>, fd: i32, iovs: u32, count: u32, written: u32) -> i32 {
    if fd != 1 && fd != 2 {
        match guest_file(&caller, fd) {
            Ok(file) if file.writable => {}
            Ok(_) => return ERRNO_BADF,
            Err(error) => return error,
        }
    }
    if count > 1024 {
        return ERRNO_INVAL;
    }
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    let mut chunks = Vec::new();
    let mut total = 0usize;
    for index in 0..count {
        let Some(base) = iovs.checked_add(index.saturating_mul(8)) else {
            return ERRNO_FAULT;
        };
        let Some(length_address) = base.checked_add(4) else {
            return ERRNO_FAULT;
        };
        let (Some(pointer), Some(length)) = (
            read_u32(&mut caller, base),
            read_u32(&mut caller, length_address),
        ) else {
            return ERRNO_FAULT;
        };
        let Some(next_total) = total.checked_add(length as usize) else {
            return ERRNO_INVAL;
        };
        if next_total > MAX_IO_BYTES
            || (fd == 1 || fd == 2)
                && next_total as u64 > caller.data().interp.resources.output_remaining()
        {
            return ERRNO_INVAL;
        }
        let mut chunk = vec![0; length as usize];
        if memory.read(&caller, pointer as usize, &mut chunk).is_err() {
            return ERRNO_FAULT;
        }
        chunks.push(chunk);
        total = next_total;
    }
    if !write_u32(&mut caller, written, total as u32) {
        return ERRNO_FAULT;
    }
    if !caller.data_mut().interp.resources.charge_cpu(total as u64) {
        return ERRNO_INVAL;
    }
    if fd == 1 || fd == 2 {
        if !caller
            .data_mut()
            .interp
            .resources
            .charge_output(total as u64)
        {
            return ERRNO_INVAL;
        }
        let destination = if fd == 1 {
            &mut caller.data_mut().stdout
        } else {
            &mut caller.data_mut().stderr
        };
        for chunk in chunks {
            destination.extend_from_slice(&chunk);
        }
    } else {
        if caller.data().append_files.contains(&fd) {
            if let Err(error) = syscalls::seek(&mut caller.data_mut().interp, fd, 0, 2) {
                return syscall_errno(&error);
            }
        }
        let mut bytes = Vec::with_capacity(total);
        for chunk in chunks {
            bytes.extend_from_slice(&chunk);
        }
        match caller.data_mut().interp.write_fd(fd, &bytes) {
            Ok(crate::descriptors::IoPoll::Ready(_)) => {}
            Ok(crate::descriptors::IoPoll::Blocked(_)) | Err(_) => return ERRNO_INVAL,
        }
    }
    ERRNO_SUCCESS
}

fn fd_read(mut caller: Caller<'_, Host>, fd: i32, iovs: u32, count: u32, read: u32) -> i32 {
    if fd != 0 {
        match guest_file(&caller, fd) {
            Ok(file) if file.readable => {}
            Ok(_) => return ERRNO_BADF,
            Err(error) => return error,
        }
    }
    if count > 1024 {
        return ERRNO_INVAL;
    }
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    let mut vectors = Vec::new();
    let mut total = 0usize;
    for index in 0..count {
        let Some(base) = iovs.checked_add(index.saturating_mul(8)) else {
            return ERRNO_FAULT;
        };
        let Some(length_address) = base.checked_add(4) else {
            return ERRNO_FAULT;
        };
        let (Some(pointer), Some(length)) = (
            read_u32(&mut caller, base),
            read_u32(&mut caller, length_address),
        ) else {
            return ERRNO_FAULT;
        };
        let Some(end) = (pointer as usize).checked_add(length as usize) else {
            return ERRNO_FAULT;
        };
        if end > memory.data(&caller).len() {
            return ERRNO_FAULT;
        }
        total = total.saturating_add(length as usize);
        if total > MAX_IO_BYTES {
            return ERRNO_INVAL;
        }
        vectors.push((pointer, length as usize));
    }
    let input = if fd == 0 {
        let offset = caller.data().stdin_offset;
        let end = offset.saturating_add(total).min(caller.data().stdin.len());
        caller.data().stdin[offset..end].to_vec()
    } else {
        match caller.data_mut().interp.read_fd(fd, total) {
            Ok(crate::descriptors::IoPoll::Ready(bytes)) => bytes,
            Ok(crate::descriptors::IoPoll::Blocked(_)) | Err(_) => return ERRNO_INVAL,
        }
    };
    let mut copied = 0usize;
    for (pointer, length) in vectors {
        let take = length.min(input.len().saturating_sub(copied));
        if memory
            .write(&mut caller, pointer as usize, &input[copied..copied + take])
            .is_err()
        {
            return ERRNO_FAULT;
        }
        copied += take;
        if take < length {
            break;
        }
    }
    if !write_u32(&mut caller, read, copied as u32) {
        return ERRNO_FAULT;
    }
    if !caller.data_mut().interp.resources.charge_cpu(copied as u64) {
        return ERRNO_INVAL;
    }
    if fd == 0 {
        caller.data_mut().stdin_offset += copied;
    }
    ERRNO_SUCCESS
}

/// Execute one validated Wasm command with a restricted WASI preview1 import set.
pub(super) fn run(
    interp: &mut Interp,
    path: &str,
    wasm: &[u8],
    args: &[String],
    stdin: &[u8],
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
) -> CommandPoll {
    if wasm.len() > MAX_WASM_BYTES {
        ewln(err, &format!("{path}: wasm module exceeds size limit"));
        return CommandPoll::Ready(126);
    }
    if !interp
        .resources
        .charge_cpu((wasm.len() as u64).saturating_mul(10))
    {
        ewln(err, &format!("{path}: wasm compilation budget exhausted"));
        return CommandPoll::Ready(137);
    }
    let mut config = Config::default();
    config.consume_fuel(true);
    // The strict profile rejects legitimate compiler modules with over 1,000 small data
    // segments. Module bytes, guest memory, and execution fuel remain independently bounded.
    config.wasm_exceptions(true);
    let engine = Engine::new(&config).expect("valid Wasmtime configuration");
    let module = match Module::new(&engine, wasm) {
        Ok(module) => module,
        Err(error) => {
            ewln(err, &format!("{path}: invalid wasm module: {error}"));
            return CommandPoll::Ready(126);
        }
    };
    let mut linker = Linker::<Host>::new(&engine);
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "proc_exit",
            |_caller: Caller<'_, Host>, code: i32| -> Result<(), Error> {
                Err(Error::new(GuestExit(code)))
            },
        )
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "args_sizes_get",
            |mut caller: Caller<'_, Host>, count: u32, size: u32| {
                strings_sizes(&mut caller, count, size, true)
            },
        )
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "args_get",
            |mut caller: Caller<'_, Host>, pointers: u32, buffer: u32| {
                strings_get(&mut caller, pointers, buffer, true)
            },
        )
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "environ_sizes_get",
            |mut caller: Caller<'_, Host>, count: u32, size: u32| {
                strings_sizes(&mut caller, count, size, false)
            },
        )
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "environ_get",
            |mut caller: Caller<'_, Host>, pointers: u32, buffer: u32| {
                strings_get(&mut caller, pointers, buffer, false)
            },
        )
        .expect("unique WASI import");
    linker
        .func_wrap("wasi_snapshot_preview1", "fd_write", fd_write)
        .expect("unique WASI import");
    linker
        .func_wrap("wasi_snapshot_preview1", "fd_read", fd_read)
        .expect("unique WASI import");
    linker
        .func_wrap("wasi_snapshot_preview1", "fd_close", fd_close)
        .expect("unique WASI import");
    linker
        .func_wrap("wasi_snapshot_preview1", "fd_seek", fd_seek)
        .expect("unique WASI import");
    linker
        .func_wrap("wasi_snapshot_preview1", "fd_tell", fd_tell)
        .expect("unique WASI import");
    linker
        .func_wrap("wasi_snapshot_preview1", "fd_fdstat_get", fd_fdstat_get)
        .expect("unique WASI import");
    linker
        .func_wrap("wasi_snapshot_preview1", "fd_filestat_get", fd_filestat_get)
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "path_filestat_get",
            path_filestat_get,
        )
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "path_unlink_file",
            path_unlink_file,
        )
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "path_create_directory",
            path_create_directory,
        )
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "path_remove_directory",
            path_remove_directory,
        )
        .expect("unique WASI import");
    linker
        .func_wrap("wasi_snapshot_preview1", "path_rename", path_rename)
        .expect("unique WASI import");
    linker
        .func_wrap("shellsim", "path_chmod", path_chmod)
        .expect("unique shellsim import");
    linker
        .func_wrap("wasi_snapshot_preview1", "fd_prestat_get", fd_prestat_get)
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "fd_prestat_dir_name",
            fd_prestat_dir_name,
        )
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "path_open",
            |caller: Caller<'_, Host>,
             directory: u32,
             _dirflags: u32,
             path_pointer: u32,
             path_length: u32,
             oflags: u32,
             rights_base: u64,
             _rights_inheriting: u64,
             fdflags: u32,
             result: u32| {
                path_open(
                    caller,
                    PathOpen {
                        directory,
                        path_pointer,
                        path_length,
                        oflags,
                        rights_base,
                        fdflags,
                        result,
                    },
                )
            },
        )
        .expect("unique WASI import");
    linker
        .func_wrap("wasi_snapshot_preview1", "random_get", random_get)
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "fd_sync",
            |caller: Caller<'_, Host>, fd: u32| {
                if fd <= 4 || caller.data().open_files.contains(&(fd as i32)) {
                    ERRNO_SUCCESS
                } else {
                    ERRNO_BADF
                }
            },
        )
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "fd_datasync",
            |caller: Caller<'_, Host>, fd: u32| {
                if fd <= 4 || caller.data().open_files.contains(&(fd as i32)) {
                    ERRNO_SUCCESS
                } else {
                    ERRNO_BADF
                }
            },
        )
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "clock_time_get",
            |mut caller: Caller<'_, Host>, clock: i32, _precision: i64, address: u32| {
                let now = match clock {
                    0 => caller
                        .data()
                        .interp
                        .clock
                        .wall_time_ns()
                        .ok()
                        .and_then(|value| u64::try_from(value).ok()),
                    1 => Some(caller.data().interp.clock.monotonic_ns()),
                    _ => None,
                };
                match now {
                    Some(value) if write_u64(&mut caller, address, value) => ERRNO_SUCCESS,
                    Some(_) => ERRNO_FAULT,
                    None => ERRNO_INVAL,
                }
            },
        )
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "sched_yield",
            |_caller: Caller<'_, Host>| ERRNO_SUCCESS,
        )
        .expect("unique WASI import");
    let arguments = std::iter::once(path.as_bytes().to_vec())
        .chain(args.iter().map(|arg| arg.as_bytes().to_vec()))
        .collect();
    let environment = interp
        .process
        .exported
        .iter()
        .filter_map(|name| {
            interp
                .process
                .vars
                .get(name)
                .map(|value| format!("{name}={value}").into_bytes())
        })
        .collect();
    let cwd = interp.cwd.clone();
    let memory_limit = interp
        .resources
        .memory_remaining()
        .min(MAX_WASM_MEMORY as u64) as usize;
    if !interp.resources.reserve_memory(memory_limit as u64) {
        ewln(err, &format!("{path}: wasm memory budget exhausted"));
        return CommandPoll::Ready(137);
    }
    let host = Host {
        interp: std::mem::take(interp),
        cwd,
        args: arguments,
        environment,
        stdin: stdin.to_vec(),
        stdin_offset: 0,
        stdout: std::mem::take(out),
        stderr: std::mem::take(err),
        open_files: BTreeSet::new(),
        append_files: BTreeSet::new(),
        random_state: 0x5eed_5eed_5eed_5eed,
        limits: StoreLimitsBuilder::new()
            .memory_size(memory_limit)
            .table_elements(10_000)
            .memories(1)
            .tables(16)
            .build(),
    };
    let mut store = Store::new(&engine, host);
    store.limiter(|host| &mut host.limits);
    let fuel = store
        .data()
        .interp
        .resources
        .cpu_remaining()
        .saturating_mul(WASM_FUEL_PER_CPU_UNIT);
    if store.set_fuel(fuel).is_err() {
        store
            .data_mut()
            .interp
            .resources
            .release_memory(memory_limit as u64);
        ewln(
            &mut store.data_mut().stderr,
            &format!("{path}: wasm fuel unavailable"),
        );
        let mut host = store.into_data();
        *interp = std::mem::take(&mut host.interp);
        *out = host.stdout;
        *err = host.stderr;
        return CommandPoll::Ready(137);
    }
    let result = linker
        .instantiate(&mut store, &module)
        .and_then(|instance| {
            let start = instance.get_typed_func::<(), ()>(&mut store, "_start")?;
            start.call(&mut store, ())
        });
    let consumed = fuel.saturating_sub(store.get_fuel().unwrap_or(0));
    let cpu_cost = consumed.saturating_add(WASM_FUEL_PER_CPU_UNIT - 1) / WASM_FUEL_PER_CPU_UNIT;
    let _ = store.data_mut().interp.resources.charge_cpu(cpu_cost);
    let open_files = std::mem::take(&mut store.data_mut().open_files);
    for fd in open_files {
        let _ = syscalls::close(&mut store.data_mut().interp, fd);
    }
    store
        .data_mut()
        .interp
        .resources
        .release_memory(memory_limit as u64);
    let stopped = store.data().interp.resources.stop_reason();
    let status = match result {
        Ok(()) => CommandPoll::Ready(0),
        Err(error) if error.downcast_ref::<GuestExit>().is_some() => {
            CommandPoll::Ready(error.downcast_ref::<GuestExit>().expect("checked exit").0)
        }
        Err(error) => {
            ewln(
                &mut store.data_mut().stderr,
                &format!("{path}: wasm execution failed: {error}"),
            );
            CommandPoll::Ready(126)
        }
    };
    let mut host = store.into_data();
    *interp = std::mem::take(&mut host.interp);
    *out = host.stdout;
    *err = host.stderr;
    stopped.map_or(status, |reason| CommandPoll::Ready(reason.exit_status()))
}
