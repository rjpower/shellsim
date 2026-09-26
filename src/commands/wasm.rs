//! Bounded WASI preview1 execution against shellsim's virtual command boundary.
//!
//! A Wasm executable runs as a scheduled process image ([`WasmProcess`]) on the process's
//! virtual descriptors. The guest runs on a Wasmtime async stack: a stream call that would block
//! suspends that stack and reports the exact wait reason, and fuel yields bound each scheduler
//! quantum, so a guest can sit in a pipeline or wait at a prompt without stalling other
//! processes. Consumed fuel is charged to the machine's CPU budget before each host call and at
//! each yield. A live guest stack is never cloned: a machine snapshot taken mid-run gets a copy
//! that fails explicitly.
//!
//! The host functions expose standard streams, process metadata, the virtual clock, and bounded
//! regular-file access. Unavailable WASI calls trap if reached; other namespaces fail
//! instantiation. Neither path grants ambient host capabilities. [`WasmSession`] runs one guest
//! against buffered standard streams for display-driven embedding.

use std::collections::{BTreeSet, VecDeque};
use std::fmt;
use std::future::poll_fn;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll, Waker};
use wasmtime::{
    AsContextMut, CallHook, Caller, Config, Engine, Error, Extern, Linker, Memory, Module, Store,
    StoreContextMut, StoreLimits, StoreLimitsBuilder,
};

use crate::descriptors::{DescriptorError, IoPoll};
use crate::display::DisplayError;
use crate::exec::ShellPoll;
use crate::interp::Interp;
use crate::program::{poll_write, wait_reason};
use crate::scheduler::WaitReason;
use crate::syscalls::{ActiveSystem, ClockId, FileInfo, FileKind, OpenFile, SyscallError, System};
use crate::vfs::{resolve_against, VfsError};

use super::util::ewln;

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
/// Fuel a guest may consume between cooperative yields to the scheduler.
const FUEL_YIELD_INTERVAL: u64 = 100_000;
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

/// Access to the virtual machine for host calls.
///
/// A guest's Wasmtime store lives across many scheduler turns, so it cannot hold a borrow of
/// the machine. The poller publishes the machine for exactly the duration of one poll of the
/// guest's future ([`MachineAccess::enter`]). Host calls run synchronously inside that poll and
/// reach the machine only through `&mut Host`, so at most one `&mut Interp` derived from the
/// published pointer exists at a time, and none outlives the poll. The poller does not touch
/// the machine while the guest runs.
#[derive(Clone, Default)]
pub(crate) struct MachineAccess(Arc<MachineShared>);

#[derive(Default)]
struct MachineShared {
    machine: AtomicPtr<Interp>,
    signals: Mutex<Signals>,
}

/// State passed between host calls and the poller across a guest suspension.
#[derive(Default)]
struct Signals {
    /// Why a host call suspended the guest. `None` after a pending poll means a fuel yield.
    suspension: Option<Suspension>,
    /// Guest fuel already charged to the machine's CPU budget.
    charged_fuel: u64,
    /// Fuel yields observed so far. Wasmtime refills exactly one interval per yield.
    fuel_yields: u64,
}

enum Suspension {
    /// A stream call would block on this virtual resource.
    Blocked(WaitReason),
    /// The guest or a display frame voluntarily gave up the rest of its quantum.
    Yielded,
}

impl MachineAccess {
    /// Run `poll` with the machine published to host calls.
    fn enter<R>(&self, interp: &mut Interp, poll: impl FnOnce() -> R) -> R {
        struct Withdraw<'a>(&'a AtomicPtr<Interp>);
        impl Drop for Withdraw<'_> {
            fn drop(&mut self) {
                self.0.store(std::ptr::null_mut(), Ordering::Release);
            }
        }
        let previous = self.0.machine.swap(interp, Ordering::AcqRel);
        assert!(previous.is_null(), "a wasm guest is already being polled");
        let _withdraw = Withdraw(&self.0.machine);
        poll()
    }

    fn get(&mut self) -> &mut Interp {
        let machine = self.0.machine.load(Ordering::Acquire);
        assert!(!machine.is_null(), "wasm host call outside a guest poll");
        // SAFETY: the pointer comes from the exclusive borrow held by `enter` for the duration
        // of this poll, which the poller does not use meanwhile. The returned borrow is tied to
        // `&mut self`, and every `MachineAccess` reached by host calls is the one inside the
        // exclusively borrowed `Host`, so no second `&mut Interp` can coexist with it.
        unsafe { &mut *machine }
    }

    fn get_ref(&self) -> &Interp {
        let machine = self.0.machine.load(Ordering::Acquire);
        assert!(!machine.is_null(), "wasm host call outside a guest poll");
        // SAFETY: as for `get`; a shared borrow of `Host` excludes the mutable path.
        unsafe { &*machine }
    }

    fn signals(&self) -> std::sync::MutexGuard<'_, Signals> {
        self.0.signals.lock().expect("wasm signal lock")
    }

    /// Return `Pending` once so the poller can report `suspension` to the scheduler.
    async fn suspend(&self, suspension: Suspension) {
        self.signals().suspension = Some(suspension);
        let mut suspended = false;
        poll_fn(|context| {
            if suspended {
                Poll::Ready(())
            } else {
                suspended = true;
                context.waker().wake_by_ref();
                Poll::Pending
            }
        })
        .await;
    }

    /// Record that the guest has consumed at least `consumed` fuel and return the CPU units not
    /// yet charged. Charging is monotonic, so a conservative estimate is never charged twice.
    fn account_fuel(&self, consumed: u64) -> u64 {
        let mut signals = self.signals();
        let charged = signals.charged_fuel;
        if consumed <= charged {
            return 0;
        }
        signals.charged_fuel = consumed;
        fuel_cpu_units(consumed) - fuel_cpu_units(charged)
    }
}

fn fuel_cpu_units(fuel: u64) -> u64 {
    fuel.div_ceil(WASM_FUEL_PER_CPU_UNIT)
}

/// Where the guest's standard descriptors go.
enum Stdio {
    /// A display session buffers standard streams and returns them with the result.
    Buffered {
        stdin: Vec<u8>,
        offset: usize,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
    /// A scheduled process uses its virtual descriptors 0, 1, and 2 directly.
    Descriptors,
}

struct Host {
    machine: MachineAccess,
    cwd: String,
    args: Vec<Vec<u8>>,
    environment: Vec<Vec<u8>>,
    stdio: Stdio,
    /// Execution failure reported after the guest stops.
    diagnostic: Vec<u8>,
    /// Fuel granted at startup; the difference from the store's remaining fuel is consumption.
    initial_fuel: u64,
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
    ActiveSystem::new(caller.data_mut().machine.get())
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
    let fd =
        match ActiveSystem::new(caller.data_mut().machine.get()).open_file(&cwd, &path, options) {
            Ok(fd) => fd,
            Err(error) => return syscall_errno(&error),
        };
    if !write_u32(&mut caller, request.result, fd as u32) {
        let _ = ActiveSystem::new(caller.data_mut().machine.get()).close(fd);
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
    match ActiveSystem::new(caller.data_mut().machine.get()).chmod(&cwd, &path, mode) {
        Ok(()) => ERRNO_SUCCESS,
        Err(error) => syscall_errno(&error),
    }
}

// These calls copy pixels and key events through guest memory. No guest pointer or host device
// escapes the active virtual process.
fn display_open(mut caller: Caller<'_, Host>, width: u32, height: u32, format: u32) -> i32 {
    match ActiveSystem::new(caller.data_mut().machine.get()).display_open(width, height, format) {
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
    let Some(frame) = caller.data().machine.get_ref().display.frame() else {
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
        .machine
        .get()
        .resources
        .reserve_memory(u64::from(length))
    {
        return ERRNO_NOSPC;
    }
    let mut pixels = vec![0; length as usize];
    let read = memory.read(&caller, pointer as usize, &mut pixels);
    let result = if read.is_ok() {
        ActiveSystem::new(caller.data_mut().machine.get())
            .display_present(handle, &pixels, stride)
            .map_or_else(display_errno, |_| ERRNO_SUCCESS)
    } else {
        ERRNO_FAULT
    };
    caller
        .data_mut()
        .machine
        .get()
        .resources
        .release_memory(u64::from(length));
    if result == ERRNO_SUCCESS {
        if let Some(interaction) = &caller.data().interaction {
            let mut interaction = interaction.lock().expect("interactive state lock");
            interaction.frame = caller.data().machine.get_ref().display.frame().cloned();
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
            if let Err(error) = caller.data_mut().machine.get().inject_key(event) {
                return display_errno(error);
            }
        }
    }
    match ActiveSystem::new(caller.data_mut().machine.get()).input_poll_key(handle) {
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
    ActiveSystem::new(caller.data_mut().machine.get())
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
        let host = caller.data_mut();
        if !host.closed_stdio.insert(fd) {
            return ERRNO_BADF;
        }
        // Closing a real standard descriptor lets a pipe reader see EOF before the guest exits.
        if matches!(host.stdio, Stdio::Descriptors) {
            let _ = ActiveSystem::new(host.machine.get()).close(fd);
        }
        return ERRNO_SUCCESS;
    }
    if !caller.data_mut().open_files.remove(&fd) {
        return ERRNO_BADF;
    }
    caller.data_mut().append_files.remove(&fd);
    match ActiveSystem::new(caller.data_mut().machine.get()).close(fd) {
        Ok(()) => ERRNO_SUCCESS,
        Err(error) => syscall_errno(&error),
    }
}

fn fd_seek(mut caller: Caller<'_, Host>, fd: u32, delta: i64, whence: u32, result: u32) -> i32 {
    if let Err(error) = guest_file(&mut caller, fd as i32) {
        return error;
    }
    let position =
        match ActiveSystem::new(caller.data_mut().machine.get()).seek(fd as i32, delta, whence) {
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
    let system = &mut ActiveSystem::new(caller.data_mut().machine.get());
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
    let info = match ActiveSystem::new(caller.data_mut().machine.get()).metadata(
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
    match ActiveSystem::new(caller.data_mut().machine.get()).unlink(&cwd, &path) {
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
    match ActiveSystem::new(caller.data_mut().machine.get()).mkdir(&cwd, &path) {
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
    match ActiveSystem::new(caller.data_mut().machine.get()).rmdir(&cwd, &path) {
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
    match ActiveSystem::new(caller.data_mut().machine.get()).rename("/", &old_path, &new_path) {
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
    if !ActiveSystem::new(caller.data_mut().machine.get()).reserve_memory(reservation) {
        return ERRNO_INVAL;
    }
    let mut bytes = vec![0; length as usize];
    if let Err(error) = ActiveSystem::new(caller.data_mut().machine.get()).random_fill(&mut bytes) {
        ActiveSystem::new(caller.data_mut().machine.get()).release_memory(reservation);
        return syscall_errno(&error);
    }
    let result = if memory.write(&mut caller, pointer as usize, &bytes).is_ok() {
        ERRNO_SUCCESS
    } else {
        ERRNO_FAULT
    };
    ActiveSystem::new(caller.data_mut().machine.get()).release_memory(reservation);
    result
}

/// Result of one attempt at a WASI stream call that may need to wait for a pipe or terminal.
enum StreamCall {
    Done(i32),
    Blocked(WaitReason),
}

fn exhausted() -> Error {
    Error::msg("virtual resource budget exhausted")
}

/// Read a guest iovec array, keeping at most `MAX_IO_BYTES` in total. WASI permits short
/// reads and writes, so a larger request transfers a prefix instead of failing.
fn iovecs(
    caller: &mut Caller<'_, Host>,
    memory: &Memory,
    iovs: u32,
    count: u32,
) -> Result<Vec<(usize, usize)>, i32> {
    if count > 1024 {
        return Err(ERRNO_INVAL);
    }
    let mut vectors = Vec::new();
    let mut total = 0usize;
    for index in 0..count {
        let base = iovs
            .checked_add(index.saturating_mul(8))
            .ok_or(ERRNO_FAULT)?;
        let length_address = base.checked_add(4).ok_or(ERRNO_FAULT)?;
        let (Some(pointer), Some(length)) =
            (read_u32(caller, base), read_u32(caller, length_address))
        else {
            return Err(ERRNO_FAULT);
        };
        let end = (pointer as usize)
            .checked_add(length as usize)
            .ok_or(ERRNO_FAULT)?;
        if end > memory.data(&*caller).len() {
            return Err(ERRNO_FAULT);
        }
        let length = (length as usize).min(MAX_IO_BYTES - total);
        total += length;
        vectors.push((pointer as usize, length));
    }
    Ok(vectors)
}

fn fd_write(
    caller: &mut Caller<'_, Host>,
    fd: i32,
    iovs: u32,
    count: u32,
    written: u32,
) -> Result<StreamCall, Error> {
    if caller.data().closed_stdio.contains(&fd) {
        return Ok(StreamCall::Done(ERRNO_BADF));
    }
    let standard = fd == 1 || fd == 2;
    if !standard {
        match guest_file(caller, fd) {
            Ok(file) if file.writable => {}
            Ok(_) => return Ok(StreamCall::Done(ERRNO_BADF)),
            Err(error) => return Ok(StreamCall::Done(error)),
        }
    }
    let Some(memory) = memory(caller) else {
        return Ok(StreamCall::Done(ERRNO_FAULT));
    };
    let vectors = match iovecs(caller, &memory, iovs, count) {
        Ok(vectors) => vectors,
        Err(error) => return Ok(StreamCall::Done(error)),
    };
    let mut bytes = Vec::new();
    for (pointer, length) in vectors {
        bytes.extend_from_slice(&memory.data(&*caller)[pointer..pointer + length]);
    }
    let host = caller.data_mut();
    if standard {
        let remaining = ActiveSystem::new(host.machine.get()).output_remaining();
        if remaining == 0 && !bytes.is_empty() {
            let _ = ActiveSystem::new(host.machine.get()).charge_output(1);
            return Err(exhausted());
        }
        bytes.truncate(remaining.min(bytes.len() as u64) as usize);
    } else if host.append_files.contains(&fd) {
        if let Err(error) = ActiveSystem::new(host.machine.get()).seek(fd, 0, 2) {
            return Ok(StreamCall::Done(syscall_errno(&error)));
        }
    }
    let count = match &mut host.stdio {
        Stdio::Buffered { stdout, stderr, .. } if standard => {
            let destination = if fd == 1 { stdout } else { stderr };
            destination.extend_from_slice(&bytes);
            bytes.len()
        }
        _ => match ActiveSystem::new(host.machine.get()).write(fd, &bytes) {
            Ok(IoPoll::Ready(count)) => count,
            Ok(IoPoll::Blocked(wait)) => return Ok(StreamCall::Blocked(wait_reason(wait))),
            // WASI has no signals. Writing to a pipe without readers ends the guest as the
            // default SIGPIPE disposition would, so a guest that ignores EPIPE cannot spin.
            Err(SyscallError::Descriptor(DescriptorError::BrokenPipe)) => {
                return Err(Error::new(GuestExit(141)));
            }
            Err(error) => return Ok(StreamCall::Done(syscall_errno(&error))),
        },
    };
    let mut system = ActiveSystem::new(host.machine.get());
    if !system.charge_cpu(count as u64) || (standard && !system.charge_output(count as u64)) {
        return Err(exhausted());
    }
    Ok(StreamCall::Done(
        if write_u32(caller, written, count as u32) {
            ERRNO_SUCCESS
        } else {
            ERRNO_FAULT
        },
    ))
}

fn fd_read(
    caller: &mut Caller<'_, Host>,
    fd: i32,
    iovs: u32,
    count: u32,
    read: u32,
) -> Result<StreamCall, Error> {
    if caller.data().closed_stdio.contains(&fd) {
        return Ok(StreamCall::Done(ERRNO_BADF));
    }
    if fd != 0 {
        match guest_file(caller, fd) {
            Ok(file) if file.readable => {}
            Ok(_) => return Ok(StreamCall::Done(ERRNO_BADF)),
            Err(error) => return Ok(StreamCall::Done(error)),
        }
    }
    let Some(memory) = memory(caller) else {
        return Ok(StreamCall::Done(ERRNO_FAULT));
    };
    let vectors = match iovecs(caller, &memory, iovs, count) {
        Ok(vectors) => vectors,
        Err(error) => return Ok(StreamCall::Done(error)),
    };
    let total = vectors.iter().map(|(_, length)| length).sum::<usize>();
    let host = caller.data_mut();
    let input = match &mut host.stdio {
        Stdio::Buffered { stdin, offset, .. } if fd == 0 => {
            let end = offset.saturating_add(total).min(stdin.len());
            let bytes = stdin[*offset..end].to_vec();
            *offset = end;
            bytes
        }
        _ => match ActiveSystem::new(host.machine.get()).read(fd, total) {
            Ok(IoPoll::Ready(bytes)) => bytes,
            Ok(IoPoll::Blocked(wait)) => return Ok(StreamCall::Blocked(wait_reason(wait))),
            Err(error) => return Ok(StreamCall::Done(syscall_errno(&error))),
        },
    };
    if !ActiveSystem::new(host.machine.get()).charge_cpu(input.len() as u64) {
        return Err(exhausted());
    }
    let mut copied = 0usize;
    for (pointer, length) in vectors {
        let take = length.min(input.len() - copied);
        if memory
            .write(&mut *caller, pointer, &input[copied..copied + take])
            .is_err()
        {
            return Ok(StreamCall::Done(ERRNO_FAULT));
        }
        copied += take;
        if take < length {
            break;
        }
    }
    Ok(StreamCall::Done(
        if write_u32(caller, read, copied as u32) {
            ERRNO_SUCCESS
        } else {
            ERRNO_FAULT
        },
    ))
}

/// `fd_read` or `fd_write`: descriptor, iovec array, iovec count, and result address.
type StreamFn = fn(&mut Caller<'_, Host>, i32, u32, u32, u32) -> Result<StreamCall, Error>;

/// Register a stream call that suspends the guest's Wasmtime stack while its descriptor would
/// block, then retries once the scheduler resumes the process.
fn wrap_stream_call(linker: &mut Linker<Host>, name: &str, call: StreamFn) {
    linker
        .func_wrap_async(
            "wasi_snapshot_preview1",
            name,
            move |mut caller: Caller<'_, Host>, (fd, iovs, count, result): (i32, u32, u32, u32)| {
                Box::new(async move {
                    loop {
                        match call(&mut caller, fd, iovs, count, result)? {
                            StreamCall::Done(errno) => return Ok(errno),
                            StreamCall::Blocked(reason) => {
                                let machine = caller.data().machine.clone();
                                machine.suspend(Suspension::Blocked(reason)).await;
                            }
                        }
                    }
                })
            },
        )
        .expect("unique WASI import");
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
    wrap_stream_call(&mut linker, "fd_write", fd_write);
    wrap_stream_call(&mut linker, "fd_read", fd_read);
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
                match ActiveSystem::new(caller.data_mut().machine.get()).clock_time_ns(clock) {
                    Ok(value) if write_u64(&mut caller, address, value) => ERRNO_SUCCESS,
                    Ok(_) => ERRNO_FAULT,
                    Err(error) => syscall_errno(&error),
                }
            },
        )
        .expect("unique WASI import");
    linker
        .func_wrap_async(
            "wasi_snapshot_preview1",
            "sched_yield",
            |caller: Caller<'_, Host>, (): ()| {
                let machine = caller.data().machine.clone();
                Box::new(async move {
                    machine.suspend(Suspension::Yielded).await;
                    Ok(ERRNO_SUCCESS)
                })
            },
        )
        .expect("unique WASI import");
    linker
}

/// A started guest, advanced one poll at a time inside [`MachineAccess::enter`].
struct Guest {
    execution: Pin<Box<dyn Future<Output = GuestOutcome> + Send>>,
    machine: MachineAccess,
    /// Memory reserved for the guest's store until the owner releases it.
    reserved: u64,
}

/// A stopped guest and the host state it leaves behind.
struct GuestOutcome {
    status: i32,
    host: Host,
}

enum GuestPoll {
    Pending,
    Blocked(WaitReason),
    /// The machine's CPU budget ran out at a fuel yield; the owner drops the guest.
    Exhausted,
    Ready(Box<GuestOutcome>),
}

/// What a caller supplies to start one guest.
struct Launch<'a> {
    path: &'a str,
    argv: Vec<Vec<u8>>,
    stdio: Stdio,
    interaction: Option<Arc<Mutex<Interaction>>>,
}

impl Guest {
    fn poll(&mut self, interp: &mut Interp) -> GuestPoll {
        let execution = &mut self.execution;
        let outcome = self.machine.enter(interp, || {
            execution
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
        });
        if let Poll::Ready(outcome) = outcome {
            return GuestPoll::Ready(Box::new(outcome));
        }
        let mut signals = self.machine.signals();
        match signals.suspension.take() {
            Some(Suspension::Blocked(reason)) => GuestPoll::Blocked(reason),
            Some(Suspension::Yielded) => GuestPoll::Pending,
            None => {
                // Wasmtime refills one interval per yield, so at least this much fuel has run.
                signals.fuel_yields += 1;
                let consumed = signals.fuel_yields.saturating_mul(FUEL_YIELD_INTERVAL);
                drop(signals);
                let cost = self.machine.account_fuel(consumed);
                if interp.resources.charge_cpu(cost) {
                    GuestPoll::Pending
                } else {
                    GuestPoll::Exhausted
                }
            }
        }
    }
}

/// Validate and compile a Wasm command and prepare its execution without running guest code.
///
/// Failures return an exit status and a diagnostic without a trailing newline.
fn start_guest(interp: &mut Interp, launch: Launch<'_>) -> Result<Guest, (i32, String)> {
    let path = launch.path;
    let wasm = match interp.vfs.read_limited("/", path, MAX_WASM_BYTES) {
        Ok(wasm) => wasm,
        Err(VfsError::TooLarge { .. }) => {
            return Err((126, format!("{path}: wasm module exceeds size limit")));
        }
        Err(error) => return Err((126, format!("{path}: {error}"))),
    };
    if !wasm.starts_with(b"\0asm") {
        return Err((126, format!("{path}: not a Wasm executable")));
    }
    // Charge the same virtual cost on a cache hit. Host cache state must not change whether a
    // simulated process passes its resource limit.
    if !interp
        .resources
        .charge_cpu((wasm.len() as u64).saturating_mul(10))
    {
        return Err((137, format!("{path}: wasm compilation budget exhausted")));
    }
    let module = compiled_command_module(&wasm)
        .map_err(|error| (126, format!("{path}: invalid wasm module: {error}")))?;
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
            return Err((
                126,
                format!(
                    "{path}: unsupported wasm import: {}.{}",
                    import.module(),
                    import.name()
                ),
            ));
        }
    }
    let mut linker = build_linker(command_engine());
    if launch.interaction.is_some() {
        register_frame_yield(&mut linker);
    }
    linker
        .define_unknown_imports_as_traps(&module)
        .map_err(|error| (126, format!("{path}: invalid wasm imports: {error}")))?;
    let memory_limit = interp
        .resources
        .memory_remaining()
        .min(MAX_WASM_MEMORY as u64);
    if !interp.resources.reserve_memory(memory_limit) {
        return Err((137, format!("{path}: wasm memory budget exhausted")));
    }
    let machine = MachineAccess::default();
    let initial_fuel = interp
        .resources
        .cpu_remaining()
        .saturating_mul(WASM_FUEL_PER_CPU_UNIT);
    let system = ActiveSystem::new(interp);
    let host = Host {
        machine: machine.clone(),
        cwd: system.cwd().to_string(),
        args: launch.argv,
        environment: system
            .environment()
            .iter()
            .map(|(name, value)| format!("{name}={value}").into_bytes())
            .collect(),
        stdio: launch.stdio,
        diagnostic: Vec::new(),
        initial_fuel,
        closed_stdio: BTreeSet::new(),
        open_files: BTreeSet::new(),
        append_files: BTreeSet::new(),
        limits: StoreLimitsBuilder::new()
            .memory_size(memory_limit as usize)
            .table_elements(10_000)
            .memories(1)
            .tables(16)
            .build(),
        interaction: launch.interaction,
    };
    Ok(Guest {
        execution: Box::pin(execute(host, linker, module, path.to_string())),
        machine,
        reserved: memory_limit,
    })
}

/// Present a frame, then yield so a display session can show it before the guest continues.
fn register_frame_yield(linker: &mut Linker<Host>) {
    linker.allow_shadowing(true);
    linker
        .func_wrap_async(
            "shellsim",
            "display_present",
            |caller: Caller<'_, Host>, (handle, pointer, length, stride): (u32, u32, u32, u32)| {
                Box::new(async move {
                    let interaction = caller.data().interaction.clone().expect("interactive host");
                    let machine = caller.data().machine.clone();
                    let result = display_present(caller, handle, pointer, length, stride);
                    if result == ERRNO_SUCCESS {
                        machine.suspend(Suspension::Yielded).await;
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
        .expect("display_present override");
}

/// Charge fuel consumed since the last charge to the machine's CPU budget.
fn charge_consumed_fuel(mut store: StoreContextMut<'_, Host>) -> Result<(), Error> {
    let consumed = store.data().initial_fuel.saturating_sub(store.get_fuel()?);
    let host = store.data_mut();
    let cost = host.machine.account_fuel(consumed);
    if host.machine.get().resources.charge_cpu(cost) {
        Ok(())
    } else {
        Err(exhausted())
    }
}

async fn execute(host: Host, linker: Linker<Host>, module: Module, path: String) -> GuestOutcome {
    let initial_fuel = host.initial_fuel;
    let mut store = Store::new(command_engine(), host);
    store.limiter(|host| &mut host.limits);
    store.set_fuel(initial_fuel).expect("fuel configured");
    store
        .fuel_async_yield_interval(Some(FUEL_YIELD_INTERVAL))
        .expect("fuel configured");
    // Concurrent processes share one CPU budget. Charging before each host call keeps a guest
    // from acting after that budget is spent; fuel yields charge compute-only stretches.
    store.call_hook(|store, hook| {
        if matches!(hook, CallHook::CallingHost) {
            charge_consumed_fuel(store)
        } else {
            Ok(())
        }
    });
    let result = match linker.instantiate_async(&mut store, &module).await {
        Ok(instance) => match instance.get_typed_func::<(), ()>(&mut store, "_start") {
            Ok(start) => start.call_async(&mut store, ()).await,
            Err(error) => Err(error),
        },
        Err(error) => Err(error),
    };
    let _ = charge_consumed_fuel(store.as_context_mut());
    let out_of_fuel = store.get_fuel().unwrap_or(0) == 0;
    let host = store.data_mut();
    for fd in std::mem::take(&mut host.open_files) {
        let _ = ActiveSystem::new(host.machine.get()).close(fd);
    }
    let status = match result {
        Ok(()) => 0,
        Err(error) => match error.downcast_ref::<GuestExit>() {
            Some(exit) => exit.0,
            None => {
                ewln(
                    &mut host.diagnostic,
                    &format!("{path}: wasm execution failed: {error:#}"),
                );
                if out_of_fuel {
                    137
                } else {
                    126
                }
            }
        },
    };
    let status = host
        .machine
        .get()
        .resources
        .stop_reason()
        .map_or(status, |reason| reason.exit_status());
    GuestOutcome {
        status,
        host: store.into_data(),
    }
}

/// Return the guest's memory reservation and display surfaces to the machine.
fn release_guest_resources(interp: &mut Interp, reserved: &mut u64) {
    interp.resources.release_memory(std::mem::take(reserved));
    let pid = interp.process.pid;
    interp.display.close_owner(pid);
}

/// Whether the regular file at absolute `path` starts with the Wasm binary magic.
pub(crate) fn is_wasm_executable(vfs: &crate::vfs::Vfs, path: &str) -> bool {
    vfs.read_range("/", path, 0, 4).ok().as_deref() == Some(b"\0asm")
}

/// A Wasm command loaded as a scheduled process image.
///
/// Standard streams are the process's virtual descriptors 0, 1, and 2. A stream call that would
/// block suspends the guest's Wasmtime stack and reports the exact wait reason, so pipelines and
/// terminal prompts behave as they do for native images.
pub(crate) struct WasmProcess {
    path: String,
    state: WasmState,
    /// Memory reserved by a guest that this image, or the snapshot it was cloned from, started.
    reserved: u64,
}

enum WasmState {
    Starting(Vec<String>),
    Running(Guest),
    Exiting {
        status: i32,
        message: Vec<u8>,
        offset: usize,
    },
}

impl Clone for WasmProcess {
    /// A live Wasmtime stack cannot be copied into a machine snapshot. The copy fails with a
    /// diagnostic the next time it is scheduled, instead of restarting the guest or sharing its
    /// store with the original.
    fn clone(&self) -> Self {
        let state = match &self.state {
            WasmState::Starting(argv) => WasmState::Starting(argv.clone()),
            WasmState::Running(_) => WasmState::Exiting {
                status: 126,
                message: format!("{}: cannot snapshot a running wasm process\n", self.path)
                    .into_bytes(),
                offset: 0,
            },
            WasmState::Exiting {
                status,
                message,
                offset,
            } => WasmState::Exiting {
                status: *status,
                message: message.clone(),
                offset: *offset,
            },
        };
        Self {
            path: self.path.clone(),
            state,
            reserved: self.reserved,
        }
    }
}

impl WasmProcess {
    /// Load the Wasm executable at the resolved `path`; `argv` becomes the guest's arguments.
    pub(crate) fn new(path: String, argv: Vec<String>) -> Self {
        Self {
            path,
            state: WasmState::Starting(argv),
            reserved: 0,
        }
    }

    /// Release the guest's reservations when the process ends, including when it is killed.
    pub(crate) fn release_owned_memory(&mut self, interp: &mut Interp) {
        if self.reserved > 0 {
            release_guest_resources(interp, &mut self.reserved);
        }
    }

    pub(crate) fn poll(&mut self, interp: &mut Interp) -> ShellPoll {
        if let WasmState::Starting(argv) = &mut self.state {
            let argv = std::mem::take(argv)
                .into_iter()
                .map(String::into_bytes)
                .collect();
            let launch = Launch {
                path: &self.path,
                argv,
                stdio: Stdio::Descriptors,
                interaction: None,
            };
            self.state = match start_guest(interp, launch) {
                Ok(guest) => {
                    self.reserved = guest.reserved;
                    WasmState::Running(guest)
                }
                Err((status, message)) => WasmState::Exiting {
                    status,
                    message: format!("{message}\n").into_bytes(),
                    offset: 0,
                },
            };
        }
        if let WasmState::Running(guest) = &mut self.state {
            let (status, message) = match guest.poll(interp) {
                GuestPoll::Pending => return ShellPoll::Pending,
                GuestPoll::Blocked(reason) => return ShellPoll::Blocked(reason),
                GuestPoll::Exhausted => (ActiveSystem::new(interp).stop_status(), Vec::new()),
                GuestPoll::Ready(outcome) => (outcome.status, outcome.host.diagnostic),
            };
            self.state = WasmState::Exiting {
                status,
                message,
                offset: 0,
            };
        }
        self.release_owned_memory(interp);
        let WasmState::Exiting {
            status,
            message,
            offset,
        } = &mut self.state
        else {
            unreachable!("wasm process advanced past startup and execution");
        };
        poll_write(&mut ActiveSystem::new(interp), 2, message, offset, *status)
    }
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
    running: Option<(Interp, Guest)>,
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
        let interaction = Arc::new(Mutex::new(Interaction::default()));
        let launch = Launch {
            path,
            argv: std::iter::once(path.as_bytes().to_vec())
                .chain(args.iter().map(|arg| arg.as_bytes().to_vec()))
                .collect(),
            stdio: Stdio::Buffered {
                stdin: Vec::new(),
                offset: 0,
                stdout: Vec::new(),
                stderr: Vec::new(),
            },
            interaction: Some(interaction.clone()),
        };
        let guest = start_guest(&mut environment, launch).map_err(|(_, message)| message)?;
        Ok(Self {
            running: Some((environment, guest)),
            interaction,
            generation: 0,
            result: None,
        })
    }

    /// Advance the guest by one Wasmtime async poll and report a newly presented frame.
    pub fn poll(&mut self) -> SessionPoll {
        let Some((environment, guest)) = &mut self.running else {
            return SessionPoll::Ready(self.result.as_ref().expect("finished session").status);
        };
        let outcome = match guest.poll(environment) {
            GuestPoll::Ready(outcome) => Some(outcome),
            GuestPoll::Exhausted => None,
            GuestPoll::Pending | GuestPoll::Blocked(_) => {
                let generation = self
                    .interaction
                    .lock()
                    .expect("interactive state lock")
                    .generation;
                if generation > self.generation {
                    self.generation = generation;
                    return SessionPoll::Frame(generation);
                }
                return SessionPoll::Running;
            }
        };
        let (mut environment, mut guest) = self.running.take().expect("running session");
        release_guest_resources(&mut environment, &mut guest.reserved);
        let (status, stdout, stderr) = match outcome {
            Some(outcome) => {
                let GuestOutcome { status, host } = *outcome;
                let Stdio::Buffered {
                    stdout, mut stderr, ..
                } = host.stdio
                else {
                    unreachable!("display sessions buffer standard streams");
                };
                stderr.extend_from_slice(&host.diagnostic);
                (status, stdout, stderr)
            }
            None => (
                ActiveSystem::new(&mut environment).stop_status(),
                Vec::new(),
                Vec::new(),
            ),
        };
        self.result = Some(SessionResult {
            environment,
            status,
            stdout,
            stderr,
        });
        SessionPoll::Ready(status)
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
