//! Bounded WASI preview1 execution against shellsim's virtual command boundary.
//!
//! The host functions expose buffered streams, process metadata, the virtual clock, and
//! bounded regular-file access. Unavailable WASI calls trap if reached; other namespaces fail
//! instantiation. Neither path grants ambient host capabilities. Live pipe suspension remains
//! outside this first slice.

use std::collections::{BTreeSet, VecDeque};
use std::fmt;
use std::future::poll_fn;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll, Waker};
use wasmtime::{
    Caller, Config, Engine, Error, Extern, Linker, Memory, Module, Store, StoreLimits,
    StoreLimitsBuilder,
};

use crate::descriptors::DescriptorError;
use crate::display::DisplayError;
use crate::interp::Interp;
use crate::syscalls::{ActiveSystem, ClockId, FileInfo, FileKind, OpenFile, SyscallError, System};
use crate::vfs::{resolve_against, VfsError};

use super::{util::ewln, CommandPoll};

const MAX_WASM_BYTES: usize = 8 * 1024 * 1024;
const MAX_WASM_MEMORY: usize = 16 * 1024 * 1024;
const MAX_IO_BYTES: usize = 1024 * 1024;
const MAX_CACHED_MODULES: usize = 4;
const MAX_CACHED_MODULE_BYTES: usize = 128 * 1024 * 1024;
// Wasm instructions are cheaper than a modeled CPU unit. This keeps a compiled byte-oriented
// utility usable on ordinary input without relaxing the host's execution bound.
const WASM_FUEL_PER_CPU_UNIT: u64 = 10;
const ERRNO_SUCCESS: i32 = 0;
const ERRNO_BADF: i32 = 8;
const ERRNO_AGAIN: i32 = 6;
const ERRNO_BUSY: i32 = 10;
const ERRNO_FAULT: i32 = 21;
const ERRNO_INVAL: i32 = 28;
const ERRNO_EXIST: i32 = 20;
const ERRNO_ISDIR: i32 = 31;
const ERRNO_NOENT: i32 = 44;
const ERRNO_NOTDIR: i32 = 54;
const ERRNO_NOTEMPTY: i32 = 55;
const ERRNO_PERM: i32 = 63;
const ERRNO_NOSPC: i32 = 51;

struct CachedModule {
    source: Vec<u8>,
    module: Module,
    bytes: usize,
}

#[derive(Default)]
struct ModuleCache {
    entries: VecDeque<CachedModule>,
    bytes: usize,
}

impl ModuleCache {
    fn get(&mut self, source: &[u8]) -> Option<Module> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.source == source)?;
        let entry = self.entries.remove(index).expect("index from cache");
        let module = entry.module.clone();
        self.entries.push_back(entry);
        Some(module)
    }

    fn insert(&mut self, source: &[u8], module: Module) {
        let image = module.image_range();
        let bytes = source
            .len()
            .saturating_add((image.end as usize).saturating_sub(image.start as usize));
        if bytes > MAX_CACHED_MODULE_BYTES {
            return;
        }
        while self.entries.len() >= MAX_CACHED_MODULES
            || self.bytes.saturating_add(bytes) > MAX_CACHED_MODULE_BYTES
        {
            let old = self
                .entries
                .pop_front()
                .expect("cache has an entry to evict");
            self.bytes -= old.bytes;
        }
        self.entries.push_back(CachedModule {
            source: source.to_vec(),
            module,
            bytes,
        });
        self.bytes += bytes;
    }
}

fn command_engine() -> &'static Engine {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    ENGINE.get_or_init(|| {
        let mut config = Config::default();
        config.consume_fuel(true).wasm_exceptions(true);
        Engine::new(&config).expect("valid Wasmtime configuration")
    })
}

fn compiled_command_module(source: &[u8]) -> Result<Module, Error> {
    static CACHE: OnceLock<Mutex<ModuleCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(ModuleCache::default()));
    if let Some(module) = cache.lock().expect("module cache lock").get(source) {
        return Ok(module);
    }
    let module = Module::new(command_engine(), source)?;
    let mut cache = cache.lock().expect("module cache lock");
    // A concurrent miss may have populated the cache while compilation ran.
    if let Some(existing) = cache.get(source) {
        return Ok(existing);
    }
    cache.insert(source, module.clone());
    Ok(module)
}

struct Host {
    interp: Interp,
    cwd: String,
    args: Vec<Vec<u8>>,
    environment: Vec<Vec<u8>>,
    stdin: Vec<u8>,
    stdin_offset: usize,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    closed_stdio: BTreeSet<i32>,
    open_files: BTreeSet<i32>,
    append_files: BTreeSet<i32>,
    limits: StoreLimits,
    interaction: Option<Arc<Mutex<Interaction>>>,
}

#[derive(Default)]
struct Interaction {
    frame: Option<crate::display::DisplayFrame>,
    generation: u64,
    keys: VecDeque<crate::display::KeyEvent>,
    stop_requested: bool,
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
        SyscallError::ResourceExhausted => ERRNO_INVAL,
        SyscallError::NoSuchProcess | SyscallError::Process(_) => ERRNO_INVAL,
    }
}

fn display_errno(error: DisplayError) -> i32 {
    match error {
        DisplayError::InvalidArgument => ERRNO_INVAL,
        DisplayError::InvalidHandle => ERRNO_BADF,
        DisplayError::Busy => ERRNO_BUSY,
        DisplayError::QueueFull | DisplayError::ResourceExhausted => ERRNO_NOSPC,
    }
}

fn guest_file(
    caller: &mut Caller<'_, Host>,
    fd: i32,
) -> Result<crate::descriptors::FileState, i32> {
    if !caller.data().open_files.contains(&fd) {
        return Err(ERRNO_BADF);
    }
    ActiveSystem::new(&mut caller.data_mut().interp)
        .file_state(fd)
        .map_err(|error| syscall_errno(&error))
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
    let fd = match ActiveSystem::new(&mut caller.data_mut().interp).open_file(&cwd, &path, options)
    {
        Ok(fd) => fd,
        Err(error) => return syscall_errno(&error),
    };
    if !write_u32(&mut caller, request.result, fd as u32) {
        let _ = ActiveSystem::new(&mut caller.data_mut().interp).close(fd);
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
    match ActiveSystem::new(&mut caller.data_mut().interp).chmod(&cwd, &path, mode) {
        Ok(()) => ERRNO_SUCCESS,
        Err(error) => syscall_errno(&error),
    }
}

// These calls copy pixels and key events through guest memory. No guest pointer or host device
// escapes the active virtual process.
fn display_open(mut caller: Caller<'_, Host>, width: u32, height: u32, format: u32) -> i32 {
    match ActiveSystem::new(&mut caller.data_mut().interp).display_open(width, height, format) {
        Ok(handle) => handle as i32,
        Err(error) => -display_errno(error),
    }
}

fn display_present(
    mut caller: Caller<'_, Host>,
    handle: u32,
    pointer: u32,
    length: u32,
    stride: u32,
) -> i32 {
    let Some(frame) = caller.data().interp.display.frame() else {
        return ERRNO_BADF;
    };
    if usize::try_from(length).ok() != Some(frame.pixels.len()) {
        return ERRNO_INVAL;
    }
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    if !caller
        .data_mut()
        .interp
        .resources
        .reserve_memory(u64::from(length))
    {
        return ERRNO_NOSPC;
    }
    let mut pixels = vec![0; length as usize];
    let read = memory.read(&caller, pointer as usize, &mut pixels);
    let result = if read.is_ok() {
        ActiveSystem::new(&mut caller.data_mut().interp)
            .display_present(handle, &pixels, stride)
            .map_or_else(display_errno, |_| ERRNO_SUCCESS)
    } else {
        ERRNO_FAULT
    };
    caller
        .data_mut()
        .interp
        .resources
        .release_memory(u64::from(length));
    if result == ERRNO_SUCCESS {
        if let Some(interaction) = &caller.data().interaction {
            let mut interaction = interaction.lock().expect("interactive state lock");
            interaction.frame = caller.data().interp.display.frame().cloned();
            interaction.generation = interaction.generation.saturating_add(1);
        }
    }
    result
}

fn input_poll_key(mut caller: Caller<'_, Host>, handle: u32, result: u32) -> i32 {
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    let mut slot = [0; 8];
    if memory.read(&caller, result as usize, &mut slot).is_err() {
        return ERRNO_FAULT;
    }
    if let Some(interaction) = &caller.data().interaction {
        let event = interaction
            .lock()
            .expect("interactive state lock")
            .keys
            .pop_front();
        if let Some(event) = event {
            if let Err(error) = caller.data_mut().interp.inject_key(event) {
                return display_errno(error);
            }
        }
    }
    match ActiveSystem::new(&mut caller.data_mut().interp).input_poll_key(handle) {
        Ok(Some(event)) => {
            slot[..4].copy_from_slice(&event.code.to_le_bytes());
            slot[4..].copy_from_slice(&u32::from(event.pressed).to_le_bytes());
            if memory.write(&mut caller, result as usize, &slot).is_ok() {
                ERRNO_SUCCESS
            } else {
                ERRNO_FAULT
            }
        }
        Ok(None) => ERRNO_AGAIN,
        Err(error) => display_errno(error),
    }
}

fn display_close(mut caller: Caller<'_, Host>, handle: u32) -> i32 {
    ActiveSystem::new(&mut caller.data_mut().interp)
        .display_close(handle)
        .map_or_else(display_errno, |_| ERRNO_SUCCESS)
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
    if caller.data().closed_stdio.contains(&(fd as i32)) {
        return ERRNO_BADF;
    }
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
        _ => match guest_file(&mut caller, fd as i32) {
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
    if (0..=2).contains(&fd) {
        return if caller.data_mut().closed_stdio.insert(fd) {
            ERRNO_SUCCESS
        } else {
            ERRNO_BADF
        };
    }
    if !caller.data_mut().open_files.remove(&fd) {
        return ERRNO_BADF;
    }
    caller.data_mut().append_files.remove(&fd);
    match ActiveSystem::new(&mut caller.data_mut().interp).close(fd) {
        Ok(()) => ERRNO_SUCCESS,
        Err(error) => syscall_errno(&error),
    }
}

fn fd_seek(mut caller: Caller<'_, Host>, fd: u32, delta: i64, whence: u32, result: u32) -> i32 {
    if let Err(error) = guest_file(&mut caller, fd as i32) {
        return error;
    }
    let position =
        match ActiveSystem::new(&mut caller.data_mut().interp).seek(fd as i32, delta, whence) {
            Ok(position) => position,
            Err(error) => return syscall_errno(&error),
        };
    if !write_u64(&mut caller, result, position) {
        return ERRNO_FAULT;
    }
    ERRNO_SUCCESS
}

fn fd_tell(mut caller: Caller<'_, Host>, fd: u32, result: u32) -> i32 {
    let file = match guest_file(&mut caller, fd as i32) {
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

fn filestat(info: &FileInfo) -> [u8; 64] {
    let mut value = [0; 64];
    value[16] = match info.kind {
        FileKind::Directory => 3,
        FileKind::File => 4,
        FileKind::Symlink => 7,
    };
    value[24..32].copy_from_slice(&1_u64.to_le_bytes());
    let size = if info.kind == FileKind::File {
        info.size
    } else {
        0
    };
    value[32..40].copy_from_slice(&size.to_le_bytes());
    let mtime = info.mtime_ms.saturating_mul(1_000_000);
    value[40..48].copy_from_slice(&mtime.to_le_bytes());
    value[48..56].copy_from_slice(&mtime.to_le_bytes());
    value[56..64].copy_from_slice(&mtime.to_le_bytes());
    value
}

fn fd_filestat_get(mut caller: Caller<'_, Host>, fd: u32, result: u32) -> i32 {
    let cwd = caller.data().cwd.clone();
    let system = &mut ActiveSystem::new(&mut caller.data_mut().interp);
    let info = match fd {
        3 => system.metadata("/", &cwd, true),
        4 => system.metadata("/", "/", true),
        5.. => system.metadata_fd(fd as i32),
        _ => return ERRNO_BADF,
    };
    let info = match info {
        Ok(info) => info,
        Err(error) => return syscall_errno(&error),
    };
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    if memory
        .write(&mut caller, result as usize, &filestat(&info))
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
    let info = match ActiveSystem::new(&mut caller.data_mut().interp).metadata(
        &cwd,
        &path,
        flags & 1 != 0,
    ) {
        Ok(info) => info,
        Err(error) => return syscall_errno(&error),
    };
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    if memory
        .write(&mut caller, result as usize, &filestat(&info))
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
    match ActiveSystem::new(&mut caller.data_mut().interp).unlink(&cwd, &path) {
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
    match ActiveSystem::new(&mut caller.data_mut().interp).mkdir(&cwd, &path) {
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
    match ActiveSystem::new(&mut caller.data_mut().interp).rmdir(&cwd, &path) {
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
    match ActiveSystem::new(&mut caller.data_mut().interp).rename("/", &old_path, &new_path) {
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
    let Some(end) = (pointer as usize).checked_add(length as usize) else {
        return ERRNO_FAULT;
    };
    if end > memory.data(&caller).len() {
        return ERRNO_FAULT;
    }
    let reservation = u64::from(length);
    if !ActiveSystem::new(&mut caller.data_mut().interp).reserve_memory(reservation) {
        return ERRNO_INVAL;
    }
    let mut bytes = vec![0; length as usize];
    if let Err(error) = ActiveSystem::new(&mut caller.data_mut().interp).random_fill(&mut bytes) {
        ActiveSystem::new(&mut caller.data_mut().interp).release_memory(reservation);
        return syscall_errno(&error);
    }
    let result = if memory.write(&mut caller, pointer as usize, &bytes).is_ok() {
        ERRNO_SUCCESS
    } else {
        ERRNO_FAULT
    };
    ActiveSystem::new(&mut caller.data_mut().interp).release_memory(reservation);
    result
}

fn fd_write(mut caller: Caller<'_, Host>, fd: i32, iovs: u32, count: u32, written: u32) -> i32 {
    if caller.data().closed_stdio.contains(&fd) {
        return ERRNO_BADF;
    }
    if fd != 1 && fd != 2 {
        match guest_file(&mut caller, fd) {
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
    let output_remaining = if fd == 1 || fd == 2 {
        ActiveSystem::new(&mut caller.data_mut().interp).output_remaining()
    } else {
        u64::MAX
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
        if next_total > MAX_IO_BYTES || next_total as u64 > output_remaining {
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
    if !ActiveSystem::new(&mut caller.data_mut().interp).charge_cpu(total as u64) {
        return ERRNO_INVAL;
    }
    if fd == 1 || fd == 2 {
        if !ActiveSystem::new(&mut caller.data_mut().interp).charge_output(total as u64) {
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
            if let Err(error) = ActiveSystem::new(&mut caller.data_mut().interp).seek(fd, 0, 2) {
                return syscall_errno(&error);
            }
        }
        let mut bytes = Vec::with_capacity(total);
        for chunk in chunks {
            bytes.extend_from_slice(&chunk);
        }
        match ActiveSystem::new(&mut caller.data_mut().interp).write(fd, &bytes) {
            Ok(crate::descriptors::IoPoll::Ready(_)) => {}
            Ok(crate::descriptors::IoPoll::Blocked(_)) | Err(_) => return ERRNO_INVAL,
        }
    }
    ERRNO_SUCCESS
}

fn fd_read(mut caller: Caller<'_, Host>, fd: i32, iovs: u32, count: u32, read: u32) -> i32 {
    if caller.data().closed_stdio.contains(&fd) {
        return ERRNO_BADF;
    }
    if fd != 0 {
        match guest_file(&mut caller, fd) {
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
        match ActiveSystem::new(&mut caller.data_mut().interp).read(fd, total) {
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
    if !ActiveSystem::new(&mut caller.data_mut().interp).charge_cpu(copied as u64) {
        return ERRNO_INVAL;
    }
    if fd == 0 {
        caller.data_mut().stdin_offset += copied;
    }
    ERRNO_SUCCESS
}

fn build_linker(engine: &Engine) -> Linker<Host> {
    let mut linker = Linker::<Host>::new(engine);
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
        .func_wrap("shellsim", "display_open", display_open)
        .expect("unique shellsim import");
    linker
        .func_wrap("shellsim", "display_present", display_present)
        .expect("unique shellsim import");
    linker
        .func_wrap("shellsim", "input_poll_key", input_poll_key)
        .expect("unique shellsim import");
    linker
        .func_wrap("shellsim", "display_close", display_close)
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
                let clock = match clock {
                    0 => ClockId::Realtime,
                    1 => ClockId::Monotonic,
                    _ => return ERRNO_INVAL,
                };
                match ActiveSystem::new(&mut caller.data_mut().interp).clock_time_ns(clock) {
                    Ok(value) if write_u64(&mut caller, address, value) => ERRNO_SUCCESS,
                    Ok(_) => ERRNO_FAULT,
                    Err(error) => syscall_errno(&error),
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
    linker
}

/// Host-visible progress from one bounded async Wasm execution quantum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionPoll {
    /// Guest is still running; poll again after serving host-side work.
    Running,
    /// A complete frame with this generation number is available from `frame()`.
    Frame(u64),
    /// Guest exited or exhausted its virtual resource budget.
    Ready(i32),
}

/// Completed guest output and the virtual environment returned to its caller.
pub struct SessionResult {
    /// Machine state returned after the guest's Wasmtime stack has stopped.
    pub environment: Interp,
    /// Guest exit status, or 126/137 for a trap or resource exhaustion.
    pub status: i32,
    /// Captured guest standard output.
    pub stdout: Vec<u8>,
    /// Captured guest standard error and trap diagnostics.
    pub stderr: Vec<u8>,
}

/// A single guest execution that the host can poll between frames without cloning live Wasm.
///
/// The session owns its environment until completion. It does not grant the guest host I/O or
/// make an active Wasmtime stack part of `Environment`'s cloneable snapshot.
pub struct WasmSession {
    execution: Pin<Box<dyn Future<Output = SessionResult> + Send>>,
    interaction: Arc<Mutex<Interaction>>,
    generation: u64,
    result: Option<SessionResult>,
}

impl WasmSession {
    /// Start an executable Wasm file already present in the virtual filesystem.
    ///
    /// The environment is moved into the session so its active Wasmtime stack cannot be cloned.
    /// Startup errors return a diagnostic and discard that moved environment.
    pub fn start(mut environment: Interp, path: &str, args: &[String]) -> Result<Self, String> {
        let node = environment
            .vfs
            .metadata("/", path, true)
            .map_err(|error| error.to_string())?;
        if node.mode & 0o111 == 0 {
            return Err(format!("{path}: permission denied"));
        }
        let wasm = environment
            .vfs
            .read_limited("/", path, MAX_WASM_BYTES)
            .map_err(|error| error.to_string())?;
        if !wasm.starts_with(b"\0asm") {
            return Err(format!("{path}: not a Wasm executable"));
        }
        if !environment
            .resources
            .charge_cpu((wasm.len() as u64).saturating_mul(10))
        {
            return Err(format!("{path}: wasm compilation budget exhausted"));
        }
        let mut config = Config::default();
        config.consume_fuel(true).wasm_exceptions(true);
        let engine = Engine::new(&config).map_err(|error| error.to_string())?;
        let module = Module::new(&engine, &wasm).map_err(|error| error.to_string())?;
        for import in module.imports() {
            if import.module() != "wasi_snapshot_preview1"
                && !(import.module() == "shellsim"
                    && matches!(
                        import.name(),
                        "path_chmod"
                            | "display_open"
                            | "display_present"
                            | "input_poll_key"
                            | "display_close"
                    ))
            {
                return Err(format!(
                    "{path}: unsupported wasm import: {}.{}",
                    import.module(),
                    import.name()
                ));
            }
        }
        let mut linker = build_linker(&engine);
        linker.allow_shadowing(true);
        linker
            .func_wrap_async(
                "shellsim",
                "display_present",
                |caller: Caller<'_, Host>,
                 (handle, pointer, length, stride): (u32, u32, u32, u32)| {
                    Box::new(async move {
                        let interaction =
                            caller.data().interaction.clone().expect("interactive host");
                        let result = display_present(caller, handle, pointer, length, stride);
                        if result == ERRNO_SUCCESS {
                            let mut yielded = false;
                            poll_fn(|context| {
                                if yielded {
                                    Poll::Ready(())
                                } else {
                                    yielded = true;
                                    context.waker().wake_by_ref();
                                    Poll::Pending
                                }
                            })
                            .await;
                        }
                        if interaction
                            .lock()
                            .expect("interactive state lock")
                            .stop_requested
                        {
                            Err(Error::new(GuestExit(130)))
                        } else {
                            Ok(result)
                        }
                    })
                },
            )
            .map_err(|error| error.to_string())?;
        linker
            .define_unknown_imports_as_traps(&module)
            .map_err(|error| error.to_string())?;

        let memory_limit = environment
            .resources
            .memory_remaining()
            .min(MAX_WASM_MEMORY as u64) as usize;
        if !environment.resources.reserve_memory(memory_limit as u64) {
            return Err(format!("{path}: wasm memory budget exhausted"));
        }
        let interaction = Arc::new(Mutex::new(Interaction::default()));
        let host = Host {
            cwd: ActiveSystem::new(&mut environment).cwd().to_string(),
            args: std::iter::once(path.as_bytes().to_vec())
                .chain(args.iter().map(|arg| arg.as_bytes().to_vec()))
                .collect(),
            environment: ActiveSystem::new(&mut environment)
                .environment()
                .iter()
                .map(|(name, value)| format!("{name}={value}").into_bytes())
                .collect(),
            interp: environment,
            stdin: Vec::new(),
            stdin_offset: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
            closed_stdio: BTreeSet::new(),
            open_files: BTreeSet::new(),
            append_files: BTreeSet::new(),
            limits: StoreLimitsBuilder::new()
                .memory_size(memory_limit)
                .table_elements(10_000)
                .memories(1)
                .tables(16)
                .build(),
            interaction: Some(interaction.clone()),
        };
        let path = path.to_string();
        let execution = Box::pin(async move {
            let mut store = Store::new(&engine, host);
            store.limiter(|host| &mut host.limits);
            let fuel = store
                .data()
                .interp
                .resources
                .cpu_remaining()
                .saturating_mul(WASM_FUEL_PER_CPU_UNIT);
            store.set_fuel(fuel).expect("fuel configured");
            store
                .fuel_async_yield_interval(Some(1_000_000))
                .expect("fuel configured");
            let result = match linker.instantiate_async(&mut store, &module).await {
                Ok(instance) => match instance.get_typed_func::<(), ()>(&mut store, "_start") {
                    Ok(start) => start.call_async(&mut store, ()).await,
                    Err(error) => Err(error),
                },
                Err(error) => Err(error),
            };
            let consumed = fuel.saturating_sub(store.get_fuel().unwrap_or(0));
            let cpu_cost =
                consumed.saturating_add(WASM_FUEL_PER_CPU_UNIT - 1) / WASM_FUEL_PER_CPU_UNIT;
            let _ = store.data_mut().interp.resources.charge_cpu(cpu_cost);
            let open_files = std::mem::take(&mut store.data_mut().open_files);
            for fd in open_files {
                let _ = ActiveSystem::new(&mut store.data_mut().interp).close(fd);
            }
            let pid = store.data().interp.process.pid;
            store.data_mut().interp.display.close_owner(pid);
            store
                .data_mut()
                .interp
                .resources
                .release_memory(memory_limit as u64);
            let status = match result {
                Ok(()) => 0,
                Err(error) if error.downcast_ref::<GuestExit>().is_some() => {
                    error.downcast_ref::<GuestExit>().expect("checked exit").0
                }
                Err(error) => {
                    ewln(
                        &mut store.data_mut().stderr,
                        &format!("{path}: wasm execution failed: {error:#}"),
                    );
                    if store.get_fuel().unwrap_or(0) == 0 {
                        137
                    } else {
                        126
                    }
                }
            };
            let mut host = store.into_data();
            SessionResult {
                status: host
                    .interp
                    .resources
                    .stop_reason()
                    .map_or(status, |reason| reason.exit_status()),
                environment: std::mem::take(&mut host.interp),
                stdout: host.stdout,
                stderr: host.stderr,
            }
        });
        Ok(Self {
            execution,
            interaction,
            generation: 0,
            result: None,
        })
    }

    /// Advance the guest by one Wasmtime async poll and report a newly presented frame.
    pub fn poll(&mut self) -> SessionPoll {
        if let Some(result) = &self.result {
            return SessionPoll::Ready(result.status);
        }
        let outcome = self
            .execution
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()));
        if let Poll::Ready(result) = outcome {
            let status = result.status;
            self.result = Some(result);
            return SessionPoll::Ready(status);
        }
        let generation = self
            .interaction
            .lock()
            .expect("interactive state lock")
            .generation;
        if generation > self.generation {
            self.generation = generation;
            SessionPoll::Frame(generation)
        } else {
            SessionPoll::Running
        }
    }

    /// Queue one bounded key transition for the guest's next input poll.
    pub fn inject_key(&mut self, event: crate::display::KeyEvent) -> Result<(), DisplayError> {
        if event.code == 0 || event.code > 0xffff {
            return Err(DisplayError::InvalidArgument);
        }
        let mut interaction = self.interaction.lock().expect("interactive state lock");
        if interaction.keys.len() >= 256 {
            return Err(DisplayError::QueueFull);
        }
        interaction.keys.push_back(event);
        Ok(())
    }

    /// Copy the last completed frame without exposing guest memory or a host window.
    pub fn frame(&self) -> Option<crate::display::DisplayFrame> {
        self.interaction
            .lock()
            .expect("interactive state lock")
            .frame
            .clone()
    }

    /// Stop at the next frame boundary and return the environment with status 130.
    pub fn request_stop(&mut self) {
        self.interaction
            .lock()
            .expect("interactive state lock")
            .stop_requested = true;
    }

    /// Return the environment and output after execution has completed, or `None` if still live.
    pub fn into_result(self) -> Option<SessionResult> {
        self.result
    }
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
    // Charge the same virtual cost on a cache hit. Host cache state must not change whether a
    // simulated process passes its resource limit.
    let engine = command_engine();
    let module = match compiled_command_module(wasm) {
        Ok(module) => module,
        Err(error) => {
            ewln(err, &format!("{path}: invalid wasm module: {error}"));
            return CommandPoll::Ready(126);
        }
    };
    let mut linker = build_linker(engine);
    // Toolchains import a broad libc surface even when a particular compile does not call it.
    // An unavailable WASI operation must trap if reached, never touch the host or return fake
    // success. Non-WASI namespaces must still be rejected at instantiation.
    for import in module.imports() {
        if import.module() != "wasi_snapshot_preview1"
            && !(import.module() == "shellsim"
                && matches!(
                    import.name(),
                    "path_chmod"
                        | "display_open"
                        | "display_present"
                        | "input_poll_key"
                        | "display_close"
                ))
        {
            ewln(
                err,
                &format!(
                    "{path}: unsupported wasm import: {}.{}",
                    import.module(),
                    import.name()
                ),
            );
            return CommandPoll::Ready(126);
        }
    }
    if let Err(error) = linker.define_unknown_imports_as_traps(&module) {
        ewln(err, &format!("{path}: invalid wasm imports: {error}"));
        return CommandPoll::Ready(126);
    }
    let arguments = std::iter::once(path.as_bytes().to_vec())
        .chain(args.iter().map(|arg| arg.as_bytes().to_vec()))
        .collect();
    let environment = ActiveSystem::new(interp)
        .environment()
        .iter()
        .map(|(name, value)| format!("{name}={value}").into_bytes())
        .collect();
    let cwd = ActiveSystem::new(interp).cwd().to_string();
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
        closed_stdio: BTreeSet::new(),
        open_files: BTreeSet::new(),
        append_files: BTreeSet::new(),
        limits: StoreLimitsBuilder::new()
            .memory_size(memory_limit)
            .table_elements(10_000)
            .memories(1)
            .tables(16)
            .build(),
        interaction: None,
    };
    let mut store = Store::new(engine, host);
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
        let _ = ActiveSystem::new(&mut store.data_mut().interp).close(fd);
    }
    let pid = store.data().interp.process.pid;
    store.data_mut().interp.display.close_owner(pid);
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
                &format!("{path}: wasm execution failed: {error:#}"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_reuses_exact_module_bytes_and_evicts_oldest_entry() {
        let mut cache = ModuleCache::default();
        let sources = (0..=MAX_CACHED_MODULES)
            .map(|number| {
                wat::parse_str(format!(
                    "(module (func (export \"_start\") (drop (i32.const {number}))))"
                ))
                .unwrap()
            })
            .collect::<Vec<_>>();
        let first = Module::new(command_engine(), &sources[0]).unwrap();
        cache.insert(&sources[0], first.clone());
        assert!(Module::same(&first, &cache.get(&sources[0]).unwrap()));
        assert!(cache.get(&sources[1]).is_none());

        for source in sources.iter().skip(1) {
            cache.insert(source, Module::new(command_engine(), source).unwrap());
        }
        assert_eq!(cache.entries.len(), MAX_CACHED_MODULES);
        assert!(cache.get(&sources[0]).is_none());
        assert!(cache.get(&sources[1]).is_some());
        assert!(cache.bytes <= MAX_CACHED_MODULE_BYTES);
    }
}
