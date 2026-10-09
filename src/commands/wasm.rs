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

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::future::poll_fn;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll, Waker};
use wasmtime::{
    AsContextMut, CallHook, Caller, Collector, Config, Engine, Error, Extern, Linker, Module,
    Store, StoreContextMut, StoreLimitsBuilder,
};

use crate::descriptors::{
    DescriptorError, DescriptorKind, DescriptorState, IoPoll, MAX_FDS_PER_PROCESS,
};
use crate::display::DisplayError;
use crate::exec::ShellPoll;
use crate::interp::Interp;
use crate::program::{poll_write, wait_reason};
use crate::scheduler::WaitReason;
use crate::syscalls::{ActiveSystem, ClockId, FileInfo, FileKind, OpenFile, SyscallError, System};
use crate::vfs::{resolve_against, VfsError};

use super::util::ewln;

mod dynamic;
mod guest_memory;
mod threads;

use guest_memory::GuestMemory;
mod limits;
mod posix_exec;
mod posix_open;
mod posix_process;

// The static CPython + NumPy + Pillow image is 15.2 MB. Keep compilation input bounded.
const MAX_WASM_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_WASM_MEMORY: usize = 16 * 1024 * 1024;
const DEFAULT_TABLE_ELEMENTS: usize = 10_000;
const LARGE_TABLE_ELEMENTS: usize = 16_384;
// Bound all sixteen possible tables with a conservative per-element host allocation.
const LARGE_TABLE_MEMORY: u64 = 16 * LARGE_TABLE_ELEMENTS as u64 * 16;
// CPython's static WASI image needs 20 MiB before allocating its interpreter heap.
const MAX_WASM_MEMORY: usize = 64 * 1024 * 1024;
const MAX_IO_BYTES: usize = 1024 * 1024;
const MAX_CACHED_MODULES: usize = 4;
const MAX_DIRECTORY_HANDLES: u32 = 64;
// Reserve path storage and conservative map overhead before allocating directory handles.
const DIRECTORY_MEMORY: u64 = (MAX_DIRECTORY_HANDLES as u64) * (4096 + 128);
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
const ERRNO_NOTSUP: i32 = 58;
const MAX_POLL_SUBSCRIPTIONS: u32 = 64;
const SUBSCRIPTION_BYTES: u32 = 48;
const EVENT_BYTES: u32 = 32;
/// Fuel a guest may consume between cooperative yields to the scheduler.
const FUEL_YIELD_INTERVAL: u64 = 100_000;
const ASYNC_STACK_BYTES: usize = 2 * 1024 * 1024;
const ERRNO_NOSPC: i32 = 51;
const ERRNO_RANGE: i32 = 68;

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
        config
            .consume_fuel(true)
            .async_stack_size(ASYNC_STACK_BYTES)
            .wasm_exceptions(true)
            .wasm_threads(true)
            .shared_memory(true)
            .wasm_gc(false)
            .collector(Collector::DeferredReferenceCounting);
        Engine::new(&config).expect("valid Wasmtime configuration")
    })
}

fn compiled_command_module(source: &[u8]) -> Result<Module, Error> {
    threads::reject_raw_waits(source)?;
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
/// the machine. For exactly the duration of one poll of the guest's future
/// ([`MachineAccess::enter`]), the poller moves the machine into this shared slot and leaves a
/// spare in its place, then moves it back. Host calls run synchronously inside that poll and
/// borrow the machine from the slot; a second simultaneous borrow is a bug and panics rather
/// than deadlocking.
#[derive(Clone, Default)]
pub(crate) struct MachineAccess(Arc<MachineShared>, Arc<Mutex<Signals>>);

#[derive(Default)]
struct MachineShared {
    machine: Mutex<Option<Interp>>,
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

thread_local! {
    /// A machine value to leave behind while the real one is lent to a guest. Building one is
    /// comparatively slow, so each thread keeps one for reuse; it is never observed.
    static SPARE_MACHINE: std::cell::RefCell<Option<Interp>> =
        const { std::cell::RefCell::new(None) };
}

/// The machine borrowed by one host call.
pub(crate) struct MachineGuard<'a>(std::sync::MutexGuard<'a, Option<Interp>>);

impl std::ops::Deref for MachineGuard<'_> {
    type Target = Interp;

    fn deref(&self) -> &Interp {
        self.0
            .as_ref()
            .expect("wasm host call outside a guest poll")
    }
}

impl std::ops::DerefMut for MachineGuard<'_> {
    fn deref_mut(&mut self) -> &mut Interp {
        self.0
            .as_mut()
            .expect("wasm host call outside a guest poll")
    }
}

/// Return a lent machine to the poller's borrow, even if the guest poll panics.
struct Restore<'a> {
    slot: &'a Mutex<Option<Interp>>,
    interp: &'a mut Interp,
}

impl Drop for Restore<'_> {
    fn drop(&mut self) {
        let lent = self
            .slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(machine) = lent {
            let spare = std::mem::replace(self.interp, machine);
            SPARE_MACHINE.with(|cell| *cell.borrow_mut() = Some(spare));
        }
    }
}

impl MachineAccess {
    /// Run `poll` with the machine lent to host calls.
    fn enter<R>(&self, interp: &mut Interp, poll: impl FnOnce() -> R) -> R {
        let spare = SPARE_MACHINE
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default();
        let machine = std::mem::replace(interp, spare);
        {
            let mut slot = self.slot();
            assert!(slot.is_none(), "a wasm guest is already being polled");
            *slot = Some(machine);
        }
        let _restore = Restore {
            slot: &self.0.machine,
            interp,
        };
        poll()
    }

    fn slot(&self) -> std::sync::MutexGuard<'_, Option<Interp>> {
        match self.0.machine.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                panic!("the wasm machine is already borrowed by this host call")
            }
        }
    }

    fn get(&self) -> MachineGuard<'_> {
        MachineGuard(self.slot())
    }

    fn get_ref(&self) -> MachineGuard<'_> {
        MachineGuard(self.slot())
    }

    fn fork_thread(&self) -> Self {
        Self(self.0.clone(), Arc::default())
    }

    fn signals(&self) -> std::sync::MutexGuard<'_, Signals> {
        self.1.lock().expect("wasm signal lock")
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
    /// Buffered aliases retain the original stream when dup2 redirects a standard fd.
    buffered_streams: BTreeMap<i32, i32>,
    limits: limits::GuestLimits,
    interaction: Option<Arc<Mutex<Interaction>>>,
    dynamic: dynamic::Dynamic,
    thread: Option<threads::ThreadContext>,
    retained: Arc<AtomicU64>,
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
        SyscallError::ExecutableFormat => 45,
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
    guest_descriptor(caller, fd)?;
    ActiveSystem::new(&mut caller.data_mut().machine.get())
        .file_state(fd)
        .map_err(|error| syscall_errno(&error))
}

fn guest_descriptor(caller: &mut Caller<'_, Host>, fd: i32) -> Result<DescriptorState, i32> {
    if caller.data().closed_stdio.contains(&fd)
        || (fd > 2 && !caller.data().open_files.contains(&fd))
    {
        return Err(ERRNO_BADF);
    }
    ActiveSystem::new(&mut caller.data_mut().machine.get())
        .descriptor_state(fd)
        .map_err(|error| syscall_errno(&error))
}

/// POSIX descriptor operations use virtual aliases, never host handles. Operations are
/// 1=get CLOEXEC, 2=set CLOEXEC, 3=dup above minimum, 4=dup with CLOEXEC,
/// 5=dup2, 6=dup3 with CLOEXEC.
fn descriptor_control(
    mut caller: Caller<'_, Host>,
    fd: i32,
    operation: u32,
    argument: i32,
    result: u32,
) -> Result<i32, Error> {
    if !write_u32(&mut caller, result, 0) {
        return Ok(ERRNO_FAULT);
    }
    if let Err(error) = guest_descriptor(&mut caller, fd) {
        return Ok(error);
    }
    if !ActiveSystem::new(&mut caller.data_mut().machine.get())
        .charge_cpu(MAX_FDS_PER_PROCESS as u64)
    {
        return Err(exhausted());
    }
    if operation == 1 || operation == 2 {
        if operation == 2 && !(0..=1).contains(&argument) {
            return Ok(ERRNO_INVAL);
        }
        let flags = {
            let mut machine = caller.data_mut().machine.get();
            if operation == 2 {
                machine
                    .process
                    .fds
                    .set_close_on_exec(fd, argument != 0)
                    .map(|()| 0)
            } else {
                machine.process.fds.close_on_exec(fd).map(i32::from)
            }
        };
        return Ok(match flags {
            Ok(flags) if write_u32(&mut caller, result, flags as u32) => ERRNO_SUCCESS,
            Ok(_) => ERRNO_FAULT,
            Err(_) => ERRNO_BADF,
        });
    }
    let upper = MAX_FDS_PER_PROCESS as i32 + 5;
    if !(3..=6).contains(&operation)
        || !(0..upper).contains(&argument)
        || (operation >= 5 && (argument == 3 || argument == 4))
        || (operation == 6 && argument == fd)
    {
        return Ok(ERRNO_INVAL);
    }
    let destination = if operation >= 5 {
        argument
    } else {
        let host = caller.data_mut();
        let machine = host.machine.get();
        let Some(destination) = (argument..upper).find(|candidate| {
            *candidate != 3 && *candidate != 4 && machine.process.fds.get(*candidate).is_err()
        }) else {
            return Ok(ERRNO_NOSPC);
        };
        destination
    };
    let buffered = caller.data().buffered_streams.get(&fd).copied();
    if let Err(error) =
        ActiveSystem::new(&mut caller.data_mut().machine.get()).duplicate(fd, destination)
    {
        return Ok(syscall_errno(&error));
    }
    if operation == 4 || operation == 6 {
        caller
            .data_mut()
            .machine
            .get()
            .process
            .fds
            .set_close_on_exec(destination, true)
            .expect("new descriptor is installed");
    }
    let host = caller.data_mut();
    host.open_files.insert(destination);
    host.closed_stdio.remove(&destination);
    host.buffered_streams.remove(&destination);
    if let Some(stream) = buffered {
        host.buffered_streams.insert(destination, stream);
    }
    Ok(if write_u32(&mut caller, result, destination as u32) {
        ERRNO_SUCCESS
    } else {
        ERRNO_FAULT
    })
}

fn memory(caller: &mut Caller<'_, Host>) -> Option<GuestMemory> {
    if let Some(thread) = &caller.data().thread {
        return Some(GuestMemory::Shared(thread.memory()));
    }
    caller
        .data()
        .dynamic
        .shared_memory
        .map(GuestMemory::Ordinary)
        .or_else(|| match caller.get_export("memory")? {
            Extern::Memory(memory) => Some(GuestMemory::Ordinary(memory)),
            Extern::SharedMemory(memory) => Some(GuestMemory::Shared(memory)),
            _ => None,
        })
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
        _ => ActiveSystem::new(&mut caller.data().machine.get())
            .directory_path(fd as i32)
            .map_err(|error| syscall_errno(&error)),
    }
}

fn path_open(mut caller: Caller<'_, Host>, request: PathOpen) -> i32 {
    let cwd = match preopen_base(&caller, request.directory) {
        Ok(cwd) => cwd,
        Err(error) => return error,
    };
    if request.oflags & !0b1111 != 0 || request.fdflags & !5 != 0 {
        return ERRNO_INVAL;
    }
    let path = match read_path(&mut caller, request.path_pointer, request.path_length) {
        Ok(path) => path,
        Err(error) => return error,
    };
    let info = ActiveSystem::new(&mut caller.data_mut().machine.get()).metadata(&cwd, &path, true);
    if matches!(&info, Ok(info) if info.kind == FileKind::Directory) {
        if request.oflags & !2 != 0 || request.fdflags & !4 != 0 {
            return ERRNO_INVAL;
        }
        let directory_count = {
            let machine = caller.data().machine.get();
            machine
                .process
                .fds
                .iter()
                .filter(|(_, id)| {
                    machine
                        .descriptors
                        .state(*id)
                        .is_ok_and(|state| state.kind == DescriptorKind::Directory)
                })
                .count()
        };
        if directory_count >= MAX_DIRECTORY_HANDLES as usize {
            return ERRNO_NOSPC;
        }
        let path = resolve_against(&cwd, &path);
        if path.len() > 4096 {
            return ERRNO_INVAL;
        }
        let fd = match ActiveSystem::new(&mut caller.data_mut().machine.get()).open_file(
            "/",
            &path,
            OpenFile {
                readable: true,
                writable: false,
                create: false,
                exclusive: false,
                truncate: false,
                append: false,
            },
        ) {
            Ok(fd) => fd,
            Err(error) => return syscall_errno(&error),
        };
        if request.fdflags & 4 != 0 {
            let mut machine = caller.data_mut().machine.get();
            let description = machine
                .process
                .fds
                .get(fd)
                .expect("new directory descriptor");
            machine
                .descriptors
                .set_nonblocking(description, true)
                .expect("new directory description");
        }
        if !write_u32(&mut caller, request.result, fd as u32) {
            let _ = ActiveSystem::new(&mut caller.data_mut().machine.get()).close(fd);
            return ERRNO_FAULT;
        }
        caller.data_mut().open_files.insert(fd);
        return ERRNO_SUCCESS;
    }
    if request.oflags & 2 != 0 {
        return info.map_or_else(|error| syscall_errno(&error), |_| ERRNO_NOTDIR);
    }
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
    let fd = match ActiveSystem::new(&mut caller.data_mut().machine.get())
        .open_file(&cwd, &path, options)
    {
        Ok(fd) => fd,
        Err(error) => return syscall_errno(&error),
    };
    if !write_u32(&mut caller, request.result, fd as u32) {
        let _ = ActiveSystem::new(&mut caller.data_mut().machine.get()).close(fd);
        return ERRNO_FAULT;
    }
    let host = caller.data_mut();
    host.open_files.insert(fd);
    if request.fdflags & 4 != 0 {
        let mut machine = host.machine.get();
        let description = machine.process.fds.get(fd).expect("new file descriptor");
        machine
            .descriptors
            .set_nonblocking(description, true)
            .expect("new open description");
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
    match ActiveSystem::new(&mut caller.data_mut().machine.get()).chmod(&cwd, &path, mode) {
        Ok(()) => ERRNO_SUCCESS,
        Err(error) => syscall_errno(&error),
    }
}

// These calls copy pixels and key events through guest memory. No guest pointer or host device
// escapes the active virtual process.
fn display_open(mut caller: Caller<'_, Host>, width: u32, height: u32, format: u32) -> i32 {
    match ActiveSystem::new(&mut caller.data_mut().machine.get())
        .display_open(width, height, format)
    {
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
    let Some(frame_length) = caller
        .data()
        .machine
        .get_ref()
        .display
        .frame()
        .map(|frame| frame.pixels.len())
    else {
        return ERRNO_BADF;
    };
    if usize::try_from(length).ok() != Some(frame_length) {
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
        ActiveSystem::new(&mut caller.data_mut().machine.get())
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
    let polled = ActiveSystem::new(&mut caller.data_mut().machine.get()).input_poll_key(handle);
    match polled {
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
    ActiveSystem::new(&mut caller.data_mut().machine.get())
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
        3 | 4 => {
            value[0] = 3;
            u64::MAX
        }
        _ => match guest_descriptor(&mut caller, fd as i32) {
            Ok(state) => {
                value[0] = match state.kind {
                    DescriptorKind::File => 4,
                    DescriptorKind::Directory => 3,
                    _ => 2,
                };
                if state.kind == DescriptorKind::Directory {
                    u64::MAX
                } else {
                    (if state.readable { 2 } else { 0 }) | (if state.writable { 64 } else { 0 })
                }
            }
            Err(error) => return error,
        },
    };
    value[8..16].copy_from_slice(&rights.to_le_bytes());
    value[16..24].copy_from_slice(&rights.to_le_bytes());
    if guest_file(&mut caller, fd as i32).is_ok_and(|file| file.append) {
        value[2] = 1;
    }
    if guest_descriptor(&mut caller, fd as i32).is_ok_and(|state| state.nonblocking) {
        value[2] |= 4;
    }
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    if memory.write(&mut caller, pointer as usize, &value).is_ok() {
        ERRNO_SUCCESS
    } else {
        ERRNO_FAULT
    }
}

fn fd_filestat_set_size(mut caller: Caller<'_, Host>, fd: i32, size: u64) -> Result<i32, Error> {
    match guest_descriptor(&mut caller, fd) {
        Ok(state) if state.writable && state.kind == DescriptorKind::File => {}
        Ok(_) => return Ok(ERRNO_BADF),
        Err(error) => return Ok(error),
    }
    let Ok(size) = usize::try_from(size) else {
        return Ok(ERRNO_NOSPC);
    };
    let mut machine = caller.data_mut().machine.get();
    let system = &mut ActiveSystem::new(&mut machine);
    let old_size = match system.metadata_fd(fd) {
        Ok(info) => info.size,
        Err(error) => return Ok(syscall_errno(&error)),
    };
    let growth = (size as u64).saturating_sub(old_size);
    if !system.charge_cpu(growth.saturating_add(1)) {
        return Err(exhausted());
    }
    if !system.reserve_memory(growth) {
        return Err(exhausted());
    }
    let resized = system.resize_file(fd, size);
    system.release_memory(growth);
    Ok(match resized {
        Ok(()) => ERRNO_SUCCESS,
        Err(error) => syscall_errno(&error),
    })
}

fn fd_fdstat_set_flags(mut caller: Caller<'_, Host>, fd: i32, flags: u32) -> Result<i32, Error> {
    if flags & !5 != 0 {
        return Ok(ERRNO_INVAL);
    }
    let state = match guest_descriptor(&mut caller, fd) {
        Ok(state) => state,
        Err(error) => return Ok(error),
    };
    if flags & 1 != 0 && state.kind != DescriptorKind::File {
        return Ok(ERRNO_INVAL);
    }
    if !ActiveSystem::new(&mut caller.data_mut().machine.get())
        .charge_cpu(MAX_FDS_PER_PROCESS as u64)
    {
        return Err(exhausted());
    }
    let mut machine = caller.data_mut().machine.get();
    let description = machine.process.fds.get(fd).expect("validated descriptor");
    if state.kind == DescriptorKind::File {
        machine
            .descriptors
            .set_append(description, flags & 1 != 0)
            .expect("validated regular file");
    }
    machine
        .descriptors
        .set_nonblocking(description, flags & 4 != 0)
        .expect("validated descriptor");
    Ok(ERRNO_SUCCESS)
}

fn fd_close(mut caller: Caller<'_, Host>, fd: u32) -> i32 {
    let fd = fd as i32;
    caller.data_mut().buffered_streams.remove(&fd);
    if (0..=2).contains(&fd) {
        let host = caller.data_mut();
        if !host.closed_stdio.insert(fd) {
            return ERRNO_BADF;
        }
        // Closing a real standard descriptor lets a pipe reader see EOF before the guest exits.
        let _ = ActiveSystem::new(&mut host.machine.get()).close(fd);
        return ERRNO_SUCCESS;
    }
    if !caller.data_mut().open_files.remove(&fd) {
        return ERRNO_BADF;
    }
    match ActiveSystem::new(&mut caller.data_mut().machine.get()).close(fd) {
        Ok(()) => ERRNO_SUCCESS,
        Err(error) => syscall_errno(&error),
    }
}

fn fd_seek(mut caller: Caller<'_, Host>, fd: u32, delta: i64, whence: u32, result: u32) -> i32 {
    if let Err(error) = guest_file(&mut caller, fd as i32) {
        return error;
    }
    let position = match ActiveSystem::new(&mut caller.data_mut().machine.get())
        .seek(fd as i32, delta, whence)
    {
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
    let descriptor = guest_descriptor(&mut caller, fd as i32);
    if descriptor.as_ref().is_ok_and(|state| {
        state.kind != DescriptorKind::File && state.kind != DescriptorKind::Directory
    }) {
        // Standard streams can be pipes, captures, or terminals; they have no regular-file size.
        let mut value = [0; 64];
        value[16] = 2;
        value[24..32].copy_from_slice(&1_u64.to_le_bytes());
        let Some(memory) = memory(&mut caller) else {
            return ERRNO_FAULT;
        };
        return if memory.write(&mut caller, result as usize, &value).is_ok() {
            ERRNO_SUCCESS
        } else {
            ERRNO_FAULT
        };
    }
    let directory = preopen_base(&caller, fd).ok();
    let regular_file = descriptor.is_ok_and(|state| state.kind == DescriptorKind::File);
    let info = {
        let mut machine = caller.data_mut().machine.get();
        let system = &mut ActiveSystem::new(&mut machine);
        match directory {
            Some(path) => system.metadata("/", &path, true),
            None if regular_file => system.metadata_fd(fd as i32),
            None => return ERRNO_BADF,
        }
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
    let info = match ActiveSystem::new(&mut caller.data_mut().machine.get()).metadata(
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

/// Encode stable sorted VFS directory entries using preview1 cookies. A short buffer receives
/// a prefix of the final dirent, as required by libc's readdir retry protocol.
fn fd_readdir(
    mut caller: Caller<'_, Host>,
    fd: u32,
    pointer: u32,
    length: u32,
    cookie: u64,
    result: u32,
) -> i32 {
    let base = match preopen_base(&caller, fd) {
        Ok(base) => base,
        Err(error) => return error,
    };
    if length as usize > MAX_IO_BYTES {
        return ERRNO_INVAL;
    }
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    let end = (pointer as usize).saturating_add(length as usize);
    if end > memory.data_size(&caller) {
        return ERRNO_FAULT;
    }
    let reservation = {
        let mut machine = caller.data_mut().machine.get();
        // list_dir scans the VFS and may generate finite /proc or /dev entries. Charge the
        // full scan before it runs, and reserve all possible names plus the output buffer.
        let nodes = machine.vfs.len() as u64;
        let processes = machine.processes.iter().count() as u64;
        if !machine
            .resources
            .charge_cpu(nodes.saturating_add(processes).saturating_mul(16))
        {
            return ERRNO_NOSPC;
        }
        let (path_bytes, link_bytes, longest_link) = machine.vfs.all_paths().fold(
            (0_u64, 0_u64, 0_u64),
            |(paths, links, longest), (path, node)| {
                let length = match &node.kind {
                    crate::vfs::NodeKind::Symlink(target) => target.len() as u64,
                    _ => 0,
                };
                (
                    paths.saturating_add(path.len() as u64),
                    links.saturating_add(length),
                    longest.max(length),
                )
            },
        );
        // FileInfo retains a symlink target for native callers. Only one such result is live
        // during enumeration, but its copy still needs both CPU and temporary memory.
        if !machine.resources.charge_cpu(link_bytes) {
            return ERRNO_NOSPC;
        }
        let names = path_bytes
            .saturating_add(nodes.saturating_mul(64))
            .saturating_add(processes.saturating_mul(64))
            .saturating_add(16 * 1024);
        let reservation = names
            .saturating_mul(2)
            .saturating_add(u64::from(length))
            .saturating_add(longest_link);
        if !machine.resources.reserve_memory(reservation) {
            return ERRNO_NOSPC;
        }
        reservation
    };
    let errno = (|| {
        let entries =
            match ActiveSystem::new(&mut caller.data_mut().machine.get()).list_dir("/", &base) {
                Ok(entries) => entries,
                Err(error) => return syscall_errno(&error),
            };
        let mut bytes = Vec::with_capacity(length as usize);
        for (index, name) in entries
            .iter()
            .enumerate()
            .skip(usize::try_from(cookie).unwrap_or(usize::MAX))
        {
            if bytes.len() >= length as usize {
                break;
            }
            let info = match ActiveSystem::new(&mut caller.data_mut().machine.get())
                .metadata(&base, name, false)
            {
                Ok(info) => info,
                Err(error) => return syscall_errno(&error),
            };
            let mut entry = [0; 24];
            entry[..8].copy_from_slice(&(index as u64 + 1).to_le_bytes());
            entry[16..20].copy_from_slice(&(name.len() as u32).to_le_bytes());
            entry[20] = match info.kind {
                FileKind::Directory => 3,
                FileKind::File => 4,
                FileKind::Symlink => 7,
            };
            for part in [&entry[..], name.as_bytes()] {
                let remaining = length as usize - bytes.len();
                bytes.extend_from_slice(&part[..part.len().min(remaining)]);
            }
        }
        if memory.write(&mut caller, pointer as usize, &bytes).is_ok()
            && write_u32(&mut caller, result, bytes.len() as u32)
        {
            ERRNO_SUCCESS
        } else {
            ERRNO_FAULT
        }
    })();
    caller
        .data_mut()
        .machine
        .get()
        .resources
        .release_memory(reservation);
    errno
}

fn path_readlink(
    mut caller: Caller<'_, Host>,
    fd: u32,
    pointer: u32,
    length: u32,
    buffer: u32,
    buffer_length: u32,
    result: u32,
) -> i32 {
    let base = match preopen_base(&caller, fd) {
        Ok(base) => base,
        Err(error) => return error,
    };
    let path = match read_path(&mut caller, pointer, length) {
        Ok(path) => path,
        Err(error) => return error,
    };
    if buffer_length as usize > MAX_IO_BYTES {
        return ERRNO_INVAL;
    }
    let Some(memory) = memory(&mut caller) else {
        return ERRNO_FAULT;
    };
    if (buffer as usize).saturating_add(buffer_length as usize) > memory.data_size(&caller) {
        return ERRNO_FAULT;
    }
    let reservation = u64::from(buffer_length);
    {
        let mut machine = caller.data_mut().machine.get();
        if !machine.resources.charge_cpu(reservation)
            || !machine.resources.reserve_memory(reservation)
        {
            return ERRNO_NOSPC;
        }
    }
    let errno =
        (|| {
            let bytes = match ActiveSystem::new(&mut caller.data_mut().machine.get())
                .read_link_prefix(&base, &path, buffer_length as usize)
            {
                Ok(bytes) => bytes,
                Err(error) => return syscall_errno(&error),
            };
            if memory.write(&mut caller, buffer as usize, &bytes).is_ok()
                && write_u32(&mut caller, result, bytes.len() as u32)
            {
                ERRNO_SUCCESS
            } else {
                ERRNO_FAULT
            }
        })();
    caller
        .data_mut()
        .machine
        .get()
        .resources
        .release_memory(reservation);
    errno
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
    match ActiveSystem::new(&mut caller.data_mut().machine.get()).unlink(&cwd, &path) {
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
    match ActiveSystem::new(&mut caller.data_mut().machine.get()).mkdir(&cwd, &path) {
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
    match ActiveSystem::new(&mut caller.data_mut().machine.get()).rmdir(&cwd, &path) {
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
    let old_base = match preopen_base(&caller, old_fd) {
        Ok(base) => base,
        Err(error) => return error,
    };
    let new_base = match preopen_base(&caller, new_fd) {
        Ok(base) => base,
        Err(error) => return error,
    };
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
    match ActiveSystem::new(&mut caller.data_mut().machine.get()).rename("/", &old_path, &new_path)
    {
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
    if end > memory.data_size(&caller) {
        return ERRNO_FAULT;
    }
    let reservation = u64::from(length);
    if !ActiveSystem::new(&mut caller.data_mut().machine.get()).reserve_memory(reservation) {
        return ERRNO_INVAL;
    }
    let mut bytes = vec![0; length as usize];
    let filled = ActiveSystem::new(&mut caller.data_mut().machine.get()).random_fill(&mut bytes);
    if let Err(error) = filled {
        ActiveSystem::new(&mut caller.data_mut().machine.get()).release_memory(reservation);
        return syscall_errno(&error);
    }
    let result = if memory.write(&mut caller, pointer as usize, &bytes).is_ok() {
        ERRNO_SUCCESS
    } else {
        ERRNO_FAULT
    };
    ActiveSystem::new(&mut caller.data_mut().machine.get()).release_memory(reservation);
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
    memory: &GuestMemory,
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
        if end > memory.data_size(&*caller) {
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
    let state = match guest_descriptor(caller, fd) {
        Ok(state) if state.writable => state,
        Ok(_) => return Ok(StreamCall::Done(ERRNO_BADF)),
        Err(error) => return Ok(StreamCall::Done(error)),
    };
    let standard = fd == 1 || fd == 2 || state.kind == DescriptorKind::Stream;
    let Some(memory) = memory(caller) else {
        return Ok(StreamCall::Done(ERRNO_FAULT));
    };
    let vectors = match iovecs(caller, &memory, iovs, count) {
        Ok(vectors) => vectors,
        Err(error) => return Ok(StreamCall::Done(error)),
    };
    let mut bytes = Vec::new();
    for (pointer, length) in vectors {
        bytes.extend_from_slice(&memory.bytes(&*caller, pointer, length)?);
    }
    let host = caller.data_mut();
    let buffered = host.buffered_streams.get(&fd).copied();
    if standard {
        let remaining = ActiveSystem::new(&mut host.machine.get()).output_remaining();
        if remaining == 0 && !bytes.is_empty() {
            let _ = ActiveSystem::new(&mut host.machine.get()).charge_output(1);
            return Err(exhausted());
        }
        bytes.truncate(remaining.min(bytes.len() as u64) as usize);
    }
    let count = match &mut host.stdio {
        Stdio::Buffered { stdout, stderr, .. } if matches!(buffered, Some(1 | 2)) => {
            let destination = if buffered == Some(1) { stdout } else { stderr };
            destination.extend_from_slice(&bytes);
            bytes.len()
        }
        _ => {
            let outcome = {
                let mut machine = host.machine.get();
                ActiveSystem::new(&mut machine).write(fd, &bytes)
            };
            match outcome {
                Ok(IoPoll::Ready(count)) => count,
                Ok(IoPoll::Blocked(_)) if state.nonblocking => {
                    return Ok(StreamCall::Done(ERRNO_AGAIN));
                }
                Ok(IoPoll::Blocked(wait)) => return Ok(StreamCall::Blocked(wait_reason(wait))),
                Err(SyscallError::Descriptor(DescriptorError::BrokenPipe)) => {
                    let mut machine = host.machine.get();
                    if matches!(
                        machine
                            .process
                            .signal_dispositions
                            .get(&crate::process::Signal::Pipe),
                        Some(crate::interp::ShellSignalDisposition::Ignore)
                    ) {
                        return Ok(StreamCall::Done(64)); // WASI EPIPE.
                    }
                    let pid = machine.process.pid;
                    machine
                        .processes
                        .mark_signal_termination(pid, crate::process::Signal::Pipe);
                    return Err(Error::new(GuestExit(141)));
                }
                Err(error) => return Ok(StreamCall::Done(syscall_errno(&error))),
            }
        }
    };
    let charged = {
        let mut machine = host.machine.get();
        let mut system = ActiveSystem::new(&mut machine);
        system.charge_cpu(count as u64) && (!standard || system.charge_output(count as u64))
    };
    if !charged {
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
    let state = match guest_descriptor(caller, fd) {
        Ok(state) if state.readable => state,
        Ok(_) => return Ok(StreamCall::Done(ERRNO_BADF)),
        Err(error) => return Ok(StreamCall::Done(error)),
    };
    let Some(memory) = memory(caller) else {
        return Ok(StreamCall::Done(ERRNO_FAULT));
    };
    let vectors = match iovecs(caller, &memory, iovs, count) {
        Ok(vectors) => vectors,
        Err(error) => return Ok(StreamCall::Done(error)),
    };
    let total = vectors.iter().map(|(_, length)| length).sum::<usize>();
    let host = caller.data_mut();
    let buffered = host.buffered_streams.get(&fd).copied();
    let input = match &mut host.stdio {
        Stdio::Buffered { stdin, offset, .. } if buffered == Some(0) => {
            let end = offset.saturating_add(total).min(stdin.len());
            let bytes = stdin[*offset..end].to_vec();
            *offset = end;
            bytes
        }
        _ => match ActiveSystem::new(&mut host.machine.get()).read(fd, total) {
            Ok(IoPoll::Ready(bytes)) => bytes,
            Ok(IoPoll::Blocked(_)) if state.nonblocking => {
                return Ok(StreamCall::Done(ERRNO_AGAIN));
            }
            Ok(IoPoll::Blocked(wait)) => return Ok(StreamCall::Blocked(wait_reason(wait))),
            Err(error) => return Ok(StreamCall::Done(syscall_errno(&error))),
        },
    };
    if !ActiveSystem::new(&mut host.machine.get()).charge_cpu(input.len() as u64) {
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

/// One bounded WASI subscription over virtual time or a process descriptor.
struct PollWait {
    userdata: u64,
    kind: PollKind,
}

enum PollKind {
    Clock(u64),
    Descriptor { fd: i32, writing: bool },
}

/// Decode bounded subscriptions before any scheduler or descriptor side effects.
fn poll_waits(caller: &mut Caller<'_, Host>, input: u32, count: u32) -> Result<Vec<PollWait>, i32> {
    if count == 0 || count > MAX_POLL_SUBSCRIPTIONS {
        return Err(ERRNO_INVAL);
    }
    let memory = memory(caller).ok_or(ERRNO_FAULT)?;
    let mut bytes = vec![0; (count * SUBSCRIPTION_BYTES) as usize];
    memory
        .read(&*caller, input as usize, &mut bytes)
        .map_err(|_| ERRNO_FAULT)?;
    let mut machine = caller.data_mut().machine.get();
    let system = ActiveSystem::new(&mut machine);
    let now = system
        .clock_time_ns(ClockId::Monotonic)
        .map_err(|error| syscall_errno(&error))?;
    let field = |record: &[u8], offset: usize, width: usize| {
        let mut value = [0; 8];
        value[..width].copy_from_slice(&record[offset..offset + width]);
        u64::from_le_bytes(value)
    };
    bytes
        .chunks_exact(SUBSCRIPTION_BYTES as usize)
        .map(|record| {
            match record[8] {
                0 => {}
                1 | 2 => {
                    return Ok(PollWait {
                        userdata: field(record, 0, 8),
                        kind: PollKind::Descriptor {
                            fd: field(record, 16, 4) as i32,
                            writing: record[8] == 2,
                        },
                    });
                }
                _ => return Err(ERRNO_INVAL),
            }
            let timeout = field(record, 24, 8);
            let absolute = field(record, 40, 2) & 1 != 0;
            let deadline = match (field(record, 16, 4), absolute) {
                (0 | 1, false) => now.saturating_add(timeout),
                (1, true) => timeout,
                (0, true) => {
                    let realtime = system
                        .clock_time_ns(ClockId::Realtime)
                        .map_err(|error| syscall_errno(&error))?;
                    now.saturating_add(timeout.saturating_sub(realtime))
                }
                _ => return Err(ERRNO_INVAL),
            };
            Ok(PollWait {
                userdata: field(record, 0, 8),
                kind: PollKind::Clock(deadline),
            })
        })
        .collect()
}

/// Wait for virtual descriptor readiness or the earliest clock subscription.
/// Output ranges are validated before scheduling; readiness checks never perform I/O.
async fn poll_oneoff(
    mut caller: Caller<'_, Host>,
    (input, output, count, result): (u32, u32, u32, u32),
) -> Result<i32, Error> {
    let waits = match poll_waits(&mut caller, input, count) {
        Ok(waits) => waits,
        Err(error) => return Ok(error),
    };
    let Some(memory) = memory(&mut caller) else {
        return Ok(ERRNO_FAULT);
    };
    let length = memory.data_size(&caller);
    if (output as usize)
        .checked_add(count as usize * EVENT_BYTES as usize)
        .is_none_or(|end| end > length)
        || (result as usize)
            .checked_add(4)
            .is_none_or(|end| end > length)
    {
        return Ok(ERRNO_FAULT);
    }
    let mut scheduled = None;
    loop {
        let mut events = Vec::new();
        let mut reasons = Vec::new();
        let now = caller.data_mut().machine.get().clock.monotonic_ns();
        let mut earliest = None::<u64>;
        for wait in &waits {
            let mut available = 0;
            let mut hangup = false;
            let (kind, status) = match wait.kind {
                PollKind::Clock(deadline) => {
                    earliest = Some(earliest.map_or(deadline, |old| old.min(deadline)));
                    (
                        0u8,
                        if now >= deadline {
                            Some(ERRNO_SUCCESS)
                        } else {
                            None
                        },
                    )
                }
                PollKind::Descriptor { fd, writing } => {
                    let status = match guest_descriptor(&mut caller, fd) {
                        Err(error) => Some(error),
                        Ok(_) => {
                            let ready = ActiveSystem::new(&mut caller.data_mut().machine.get())
                                .descriptor_readiness(fd, writing);
                            match ready {
                                Ok(IoPoll::Ready(ready)) => {
                                    available = ready.bytes;
                                    hangup = ready.hangup;
                                    Some(ERRNO_SUCCESS)
                                }
                                Ok(IoPoll::Blocked(reason)) => {
                                    reasons.push(wait_reason(reason));
                                    None
                                }
                                Err(error) => Some(syscall_errno(&error)),
                            }
                        }
                    };
                    (if writing { 2 } else { 1 }, status)
                }
            };
            if let Some(errno) = status {
                let mut event = [0u8; EVENT_BYTES as usize];
                event[..8].copy_from_slice(&wait.userdata.to_le_bytes());
                event[8..10].copy_from_slice(&(errno as u16).to_le_bytes());
                event[10] = kind;
                if kind != 0 && errno == ERRNO_SUCCESS {
                    event[16..24].copy_from_slice(&available.to_le_bytes());
                    event[24..26].copy_from_slice(&u16::from(hangup).to_le_bytes());
                }
                events.push(event);
            }
        }
        if !ActiveSystem::new(&mut caller.data_mut().machine.get()).charge_cpu(waits.len() as u64) {
            return Err(exhausted());
        }
        if !events.is_empty() {
            if let Some(event) = scheduled {
                caller.data_mut().machine.get().clock.cancel(event);
            }
            for (index, event) in events.iter().enumerate() {
                memory.write(
                    &mut caller,
                    output as usize + index * EVENT_BYTES as usize,
                    event,
                )?;
            }
            return Ok(if write_u32(&mut caller, result, events.len() as u32) {
                ERRNO_SUCCESS
            } else {
                ERRNO_FAULT
            });
        }
        let host = caller.data_mut();
        if let Some(deadline) = earliest {
            let buffered = matches!(host.stdio, Stdio::Buffered { .. });
            let mut interp = host.machine.get();
            if buffered && reasons.is_empty() && interp.real_time.is_none() {
                if interp.clock.advance_to(deadline).is_err() {
                    return Ok(ERRNO_INVAL);
                }
                continue;
            }
            if !buffered && scheduled.is_none() {
                let pid = interp.process.pid;
                scheduled = match interp
                    .clock
                    .schedule_wake_after(u64::from(pid), deadline.saturating_sub(now))
                {
                    Ok(event) => Some(event),
                    Err(_) => return Ok(ERRNO_INVAL),
                };
            }
            reasons.push(WaitReason::Timer(deadline));
        }
        let reason = if reasons.len() == 1 {
            reasons.pop().expect("one wait reason")
        } else {
            WaitReason::Any(reasons)
        };
        host.machine
            .clone()
            .suspend(Suspension::Blocked(reason))
            .await;
    }
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

fn cwd_get(mut caller: Caller<'_, Host>, pointer: u32, capacity: u32) -> Result<i32, Error> {
    let path = caller.data().cwd.as_bytes();
    if path.len() >= capacity as usize || capacity > 4096 {
        return Ok(ERRNO_RANGE);
    }
    let length = path.len();
    if !ActiveSystem::new(&mut caller.data_mut().machine.get()).charge_cpu(length as u64 + 1) {
        return Err(exhausted());
    }
    let mut bytes = caller.data().cwd.as_bytes().to_vec();
    bytes.push(0);
    let Some(memory) = memory(&mut caller) else {
        return Ok(ERRNO_FAULT);
    };
    Ok(
        if memory.write(&mut caller, pointer as usize, &bytes).is_ok() {
            ERRNO_SUCCESS
        } else {
            ERRNO_FAULT
        },
    )
}

fn cwd_set(
    mut caller: Caller<'_, Host>,
    pointer: u32,
    length: u32,
    output: u32,
    capacity: u32,
) -> Result<i32, Error> {
    if capacity > 4096 {
        return Ok(ERRNO_RANGE);
    }
    let Some(memory) = memory(&mut caller) else {
        return Ok(ERRNO_FAULT);
    };
    if (output as usize)
        .checked_add(capacity as usize)
        .is_none_or(|end| end > memory.data_size(&caller))
    {
        return Ok(ERRNO_FAULT);
    }
    let path = match read_path(&mut caller, pointer, length) {
        Ok(path) => path,
        Err(error) => return Ok(error),
    };
    if path.is_empty() {
        return Ok(ERRNO_NOENT);
    }
    let base = caller.data().cwd.clone();
    if !ActiveSystem::new(&mut caller.data_mut().machine.get())
        .charge_cpu((base.len() + path.len() + 1) as u64)
    {
        return Err(exhausted());
    }
    let resolved = {
        let mut machine = caller.data_mut().machine.get();
        let system = &mut ActiveSystem::new(&mut machine);
        let resolved = match system.canonicalize(&base, &path, true) {
            Ok(path) => path,
            Err(error) => return Ok(syscall_errno(&error)),
        };
        if resolved.len() >= capacity as usize {
            return Ok(ERRNO_RANGE);
        }
        if let Err(error) = system.chdir(&resolved) {
            return Ok(syscall_errno(&error));
        }
        resolved
    };
    let mut bytes = resolved.as_bytes().to_vec();
    bytes.push(0);
    caller.data_mut().cwd = resolved;
    Ok(
        if memory.write(&mut caller, output as usize, &bytes).is_ok() {
            ERRNO_SUCCESS
        } else {
            ERRNO_FAULT
        },
    )
}

fn build_linker(engine: &Engine) -> Linker<Host> {
    let mut linker = Linker::<Host>::new(engine);
    posix_process::register(&mut linker);
    linker
        .func_wrap("shellsim_posix_v1", "cwd_get", cwd_get)
        .expect("unique POSIX import");
    linker
        .func_wrap("shellsim_posix_v1", "cwd_set", cwd_set)
        .expect("unique POSIX import");
    linker
        .func_wrap(
            "shellsim_posix_v1",
            "umask",
            |mut caller: Caller<'_, Host>, mask: u32| -> Result<u32, Error> {
                let mut machine = caller.data_mut().machine.get();
                let system = &mut ActiveSystem::new(&mut machine);
                if !system.charge_cpu(1) {
                    return Err(exhausted());
                }
                let previous = system.umask();
                system
                    .set_umask((mask & 0o777) as u16)
                    .expect("permission mask is bounded");
                Ok(u32::from(previous))
            },
        )
        .expect("unique POSIX import");
    linker
        .func_wrap(
            "shellsim_posix_v1",
            "descriptor_control",
            descriptor_control,
        )
        .expect("unique POSIX import");
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
        .func_wrap(
            "wasi_snapshot_preview1",
            "fd_fdstat_set_flags",
            fd_fdstat_set_flags,
        )
        .expect("unique WASI import");
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "fd_filestat_set_size",
            fd_filestat_set_size,
        )
        .expect("unique WASI import");
    linker
        .func_wrap("wasi_snapshot_preview1", "fd_readdir", fd_readdir)
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
        .func_wrap("wasi_snapshot_preview1", "path_readlink", path_readlink)
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
            "clock_res_get",
            |mut caller: Caller<'_, Host>, clock: u32, result: u32| {
                if clock > 1 {
                    return ERRNO_INVAL;
                }
                if write_u64(&mut caller, result, 1) {
                    ERRNO_SUCCESS
                } else {
                    ERRNO_FAULT
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
                let time =
                    ActiveSystem::new(&mut caller.data_mut().machine.get()).clock_time_ns(clock);
                match time {
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
        .func_wrap_async("wasi_snapshot_preview1", "poll_oneoff", |caller, params| {
            Box::new(poll_oneoff(caller, params))
        })
        .expect("unique WASI import");
    linker
}

/// A started guest, advanced one poll at a time inside [`MachineAccess::enter`].
struct Guest {
    execution: Execution,
    machine: MachineAccess,
    /// Memory reserved for the guest's store until the owner releases it.
    reserved: GuestReservation,
}

enum Execution {
    Single(Pin<Box<dyn Future<Output = GuestOutcome> + Send>>),
    Threads(Box<threads::ThreadGroup>),
}

/// Retained images and v2 heap growth share an accounting counter with the guest owner, so cancellation
/// can release them even when the suspended Store cannot return its host state.
#[derive(Clone, Default)]
struct GuestReservation {
    fixed: u64,
    dynamic: Arc<AtomicU64>,
}

impl GuestReservation {
    fn snapshot(&self) -> Self {
        Self {
            fixed: self.fixed,
            dynamic: Arc::new(AtomicU64::new(self.dynamic.load(Ordering::Relaxed))),
        }
    }
}

/// A stopped guest and the host state it leaves behind.
enum GuestOutcome {
    Exit {
        status: i32,
        host: Host,
    },
    ThreadReturn {
        host: Host,
    },
    Exec {
        spec: posix_exec::ExecSpec,
        host: Host,
    },
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
        if let Execution::Threads(group) = &mut self.execution {
            return group.poll(interp);
        }
        let Execution::Single(execution) = &mut self.execution else {
            unreachable!()
        };
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

/// Apply the same namespace boundary before initial launch and exec preflight.
/// Broad WASI imports may remain unused; unavailable operations trap on use.
pub(super) fn validate_imports(module: &Module, threaded: bool) -> Result<(), Error> {
    for import in module.imports() {
        let allowed = import.module() == "wasi_snapshot_preview1"
            || (threaded
                && ((import.module() == "env" && import.name() == "memory")
                    || (import.module() == "wasi" && import.name() == "thread-spawn")
                    || (import.module() == threads::NAMESPACE
                        && matches!(import.name(), "wait32" | "notify"))))
            || (import.module() == "shellsim_posix_v1"
                && matches!(
                    import.name(),
                    "descriptor_control"
                        | "descriptor_open"
                        | "umask"
                        | "cwd_get"
                        | "cwd_set"
                        | "process_pipe"
                        | "process_spawn"
                        | "process_wait"
                        | "process_kill"
                        | "process_signal_disposition"
                        | "process_identity"
                        | "process_exec"
                ))
            || (dynamic::Abi::from_namespace(import.module()).is_some()
                && matches!(import.name(), "open" | "symbol" | "error"))
            || (import.module() == "shellsim"
                && matches!(
                    import.name(),
                    "path_chmod"
                        | "display_open"
                        | "display_present"
                        | "input_poll_key"
                        | "display_close"
                ));
        if !allowed {
            return Err(Error::msg(format!(
                "unsupported wasm import: {}.{}",
                import.module(),
                import.name()
            )));
        }
    }
    Ok(())
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
    threads::reject_raw_waits(&wasm).map_err(|error| (126, format!("{path}: {error}")))?;
    let thread_profile =
        threads::profile(&wasm).map_err(|error| (126, format!("{path}: {error}")))?;
    let scratch = if thread_profile.is_some() {
        (wasm.len() as u64).saturating_mul(65).saturating_add(4096)
    } else {
        0
    };
    if !interp.resources.reserve_memory(scratch) {
        return Err((
            137,
            format!("{path}: thread compilation memory budget exhausted"),
        ));
    }
    let compiled = compiled_command_module(&wasm);
    interp.resources.release_memory(scratch);
    let module =
        compiled.map_err(|error| (126, format!("{path}: invalid wasm module: {error}")))?;
    validate_imports(&module, thread_profile.is_some())
        .map_err(|error| (126, format!("{path}: {error}")))?;
    let mut linker = build_linker(command_engine());
    let mut dynamic_abi = None;
    for import in module.imports() {
        if let Some(abi) = dynamic::Abi::from_namespace(import.module()) {
            if dynamic_abi.is_some_and(|old| old != abi) {
                return Err((
                    126,
                    format!("{path}: mixed dynamic loading ABIs are unsupported"),
                ));
            }
            dynamic_abi = Some(abi);
        }
    }
    if let Some(abi) = dynamic_abi {
        dynamic::compatible_main(&wasm, abi).map_err(|error| (126, format!("{path}: {error}")))?;
    }
    dynamic::register(&mut linker);
    posix_exec::register(&mut linker);
    posix_open::register(&mut linker);
    register_frame_yield(&mut linker);
    if thread_profile.is_none() {
        linker
            .define_unknown_imports_as_traps(&module)
            .map_err(|error| (126, format!("{path}: invalid wasm imports: {error}")))?;
    }
    let minimum_memory = module
        .exports()
        .filter_map(|export| match export.ty() {
            wasmtime::ExternType::Memory(memory) => Some(memory.minimum().saturating_mul(65_536)),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    // Small utilities retain their existing reservation, allowing several in one pipeline.
    // Larger static interpreter images receive a larger bounded heap reservation.
    let large_image = minimum_memory > DEFAULT_WASM_MEMORY as u64;
    let table_memory = if dynamic_abi == Some(dynamic::Abi::V2) {
        0
    } else if large_image {
        LARGE_TABLE_MEMORY
    } else {
        0
    };
    let loader_memory = dynamic_abi.map_or(0, dynamic::Abi::fixed_reservation);
    let memory_cap = if thread_profile.is_some() {
        MAX_WASM_MEMORY as u64
    } else if dynamic_abi == Some(dynamic::Abi::V2) {
        256 * 1024 * 1024
    } else if large_image {
        MAX_WASM_MEMORY as u64
    } else {
        DEFAULT_WASM_MEMORY as u64
    };
    let memory_limit = interp
        .resources
        .memory_remaining()
        .saturating_sub(DIRECTORY_MEMORY)
        .saturating_sub(table_memory)
        .saturating_sub(loader_memory)
        // Only v1 prepays its Store reservation. V2 heap and host scratch compete for the
        // same remaining environment budget when they actually allocate.
        .saturating_sub(if dynamic_abi == Some(dynamic::Abi::V2) {
            0
        } else {
            MAX_IO_BYTES as u64
        })
        .min(memory_cap);
    let linear_reservation = if let Some(profile) = &thread_profile {
        let maximum = profile.maximum_pages.saturating_mul(65_536);
        if !interp.resources.charge_cpu(maximum) {
            return Err((
                137,
                format!("{path}: thread shared memory initialization budget exhausted"),
            ));
        }
        if maximum > memory_limit {
            return Err((
                137,
                format!("{path}: thread shared memory budget exhausted"),
            ));
        }
        maximum
    } else if dynamic_abi == Some(dynamic::Abi::V2) {
        0
    } else {
        memory_limit
    };
    let reserved = linear_reservation
        .saturating_add(DIRECTORY_MEMORY)
        .saturating_add(table_memory);
    let thread_memory = if thread_profile.is_some() {
        let mut metadata = (interp.process.cwd.len() as u64).saturating_add(96);
        for value in &launch.argv {
            metadata = metadata
                .saturating_add(value.len() as u64)
                .saturating_add(96);
        }
        for name in &interp.exported {
            if let Some(value) = interp.vars.get(name) {
                metadata = metadata
                    .saturating_add(name.len() as u64)
                    .saturating_add(value.len() as u64)
                    .saturating_add(96);
            }
        }
        let metadata = metadata.saturating_mul(threads::MAX_THREADS as u64 + 2);
        if !interp.resources.charge_cpu(metadata) {
            return Err((137, format!("{path}: thread metadata budget exhausted")));
        }
        let image = module.image_range();
        threads::THREAD_HOST_BYTES * threads::MAX_THREADS as u64
            + metadata
            + (image.end as usize).saturating_sub(image.start as usize) as u64
            + (wasm.len() as u64).saturating_mul(2)
            + 4096
    } else {
        0
    };
    let reserved = reserved
        .saturating_add(loader_memory)
        .saturating_add(thread_memory);
    if !interp.resources.reserve_memory(reserved) {
        return Err((137, format!("{path}: wasm memory budget exhausted")));
    }
    let reserved = GuestReservation {
        fixed: reserved,
        dynamic: Arc::default(),
    };
    let machine = MachineAccess::default();
    let initial_fuel = interp
        .resources
        .cpu_remaining()
        .saturating_mul(WASM_FUEL_PER_CPU_UNIT);
    let inherited_files = interp
        .process
        .fds
        .iter()
        .filter_map(|(fd, _)| (fd > 4).then_some(fd))
        .collect();
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
        diagnostic: Vec::new(),
        initial_fuel,
        closed_stdio: BTreeSet::new(),
        open_files: inherited_files,
        buffered_streams: if matches!(launch.stdio, Stdio::Buffered { .. }) {
            BTreeMap::from([(0, 0), (1, 1), (2, 2)])
        } else {
            BTreeMap::new()
        },
        stdio: launch.stdio,
        limits: {
            let store = StoreLimitsBuilder::new()
                .memory_size(memory_limit as usize)
                .table_elements(if large_image {
                    LARGE_TABLE_ELEMENTS
                } else {
                    DEFAULT_TABLE_ELEMENTS
                })
                .memories(1)
                .tables(16)
                .instances(dynamic::MAX_LOADS + 1)
                .build();
            if dynamic_abi == Some(dynamic::Abi::V2) {
                limits::GuestLimits::incremental(
                    store,
                    memory_limit as usize,
                    machine.clone(),
                    reserved.dynamic.clone(),
                )
            } else {
                limits::GuestLimits::new(store, memory_limit as usize)
            }
        },
        interaction: launch.interaction,
        dynamic: dynamic::Dynamic::new(dynamic_abi, reserved.dynamic.clone()),
        thread: None,
        retained: reserved.dynamic.clone(),
    };
    if let Some(profile) = thread_profile {
        return threads::start(host, module, path.to_string(), profile, reserved.clone()).map_err(
            |error| {
                interp.resources.release_memory(reserved.fixed);
                (126, format!("{path}: {error}"))
            },
        );
    }
    Ok(Guest {
        execution: Execution::Single(Box::pin(execute(host, linker, module, path.to_string()))),
        machine,
        reserved,
    })
}

/// Yield after presenting a frame so an action host can inspect it before execution resumes.
fn register_frame_yield(linker: &mut Linker<Host>) {
    linker.allow_shadowing(true);
    linker
        .func_wrap_async(
            "shellsim",
            "display_present",
            |caller: Caller<'_, Host>, (handle, pointer, length, stride): (u32, u32, u32, u32)| {
                Box::new(async move {
                    let interaction = caller.data().interaction.clone();
                    let machine = caller.data().machine.clone();
                    let result = display_present(caller, handle, pointer, length, stride);
                    if result == ERRNO_SUCCESS {
                        machine.suspend(Suspension::Yielded).await;
                    }
                    if interaction.is_some_and(|state| {
                        state.lock().expect("interactive state lock").stop_requested
                    }) {
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
    store.call_hook(|mut store, hook| {
        if matches!(hook, CallHook::CallingHost) {
            charge_consumed_fuel(store)
        } else if matches!(hook, CallHook::ReturningFromHost) {
            let mut machine = store.data_mut().machine.get();
            // Cancellation can unwind a suspended host call after the poller has reclaimed
            // the machine. Signal delivery belongs only to an active guest poll.
            if let Some(signal) = machine
                .0
                .as_mut()
                .and_then(Interp::take_default_termination)
            {
                return Err(Error::new(GuestExit(128 + signal.number())));
            }
            Ok(())
        } else {
            Ok(())
        }
    });
    let result = match linker.instantiate_async(&mut store, &module).await {
        Ok(instance) => {
            store.data_mut().dynamic.main = Some(instance);
            let shared_memory = instance.get_memory(&mut store, "memory");
            store.data_mut().dynamic.shared_memory = shared_memory;
            match instance.get_typed_func::<(), ()>(&mut store, "_start") {
                Ok(start) => start.call_async(&mut store, ()).await,
                Err(error) => Err(error),
            }
        }
        Err(error) => Err(error),
    };
    let _ = charge_consumed_fuel(store.as_context_mut());
    let out_of_fuel = store.get_fuel().unwrap_or(0) == 0;
    let host = store.data_mut();
    if !result
        .as_ref()
        .is_err_and(|error| error.is::<posix_exec::GuestExec>())
    {
        for fd in std::mem::take(&mut host.open_files) {
            let _ = ActiveSystem::new(&mut host.machine.get()).close(fd);
        }
    }
    finish_guest(result, store.into_data(), out_of_fuel, &path, false)
}

/// Preserve process-wide termination and exec separately from a worker return.
fn finish_guest(
    result: Result<(), Error>,
    mut host: Host,
    out_of_fuel: bool,
    path: &str,
    worker: bool,
) -> GuestOutcome {
    let returned = result.is_ok();
    let status = match result {
        Ok(()) => 0,
        Err(error) if error.is::<posix_exec::GuestExec>() => {
            let posix_exec::GuestExec(spec) = error
                .downcast::<posix_exec::GuestExec>()
                .expect("checked exec transfer");
            return GuestOutcome::Exec { spec, host };
        }
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
    if worker && returned && status == 0 {
        GuestOutcome::ThreadReturn { host }
    } else {
        GuestOutcome::Exit { status, host }
    }
}

/// Return the guest's memory reservation and display surfaces to the machine.
fn release_guest_resources(interp: &mut Interp, reserved: &mut GuestReservation) {
    let total = std::mem::take(&mut reserved.fixed)
        .saturating_add(reserved.dynamic.swap(0, Ordering::Relaxed));
    interp.resources.release_memory(total);
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
    reserved: GuestReservation,
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
            reserved: self.reserved.snapshot(),
        }
    }
}

impl WasmProcess {
    /// Load the Wasm executable at the resolved `path`; `argv` becomes the guest's arguments.
    pub(crate) fn new(path: String, argv: Vec<String>) -> Self {
        Self {
            path,
            state: WasmState::Starting(argv),
            reserved: GuestReservation::default(),
        }
    }

    /// Release the guest's reservations when the process ends, including when it is killed.
    pub(crate) fn release_owned_memory(&mut self, interp: &mut Interp) {
        if let WasmState::Running(Guest {
            execution: Execution::Threads(group),
            ..
        }) = &mut self.state
        {
            group.cancel_timer(interp);
        }
        if self.reserved.fixed > 0 || self.reserved.dynamic.load(Ordering::Relaxed) > 0 {
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
                    self.reserved = guest.reserved.clone();
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
                GuestPoll::Ready(outcome) => match *outcome {
                    GuestOutcome::Exit { status, host } => (status, host.diagnostic),
                    GuestOutcome::ThreadReturn { host } => (0, host.diagnostic),
                    GuestOutcome::Exec { spec, host } => {
                        self.reserved
                            .dynamic
                            .fetch_sub(spec.reserved_bytes, Ordering::Relaxed);
                        if let Execution::Threads(group) = &mut guest.execution {
                            group.cancel_timer(interp);
                        }
                        drop(host);
                        self.state = WasmState::Exiting {
                            status: 0,
                            message: Vec::new(),
                            offset: 0,
                        };
                        self.release_owned_memory(interp);
                        match posix_exec::apply(spec, interp) {
                            Ok(()) => return ShellPoll::Replaced,
                            Err(error) => {
                                self.state = WasmState::Exiting {
                                    status: 126,
                                    message: format!("{}: exec failed: {error}\n", self.path)
                                        .into_bytes(),
                                    offset: 0,
                                };
                                return self.poll(interp);
                            }
                        }
                    }
                },
            };
            if let Execution::Threads(group) = &mut guest.execution {
                group.cancel_timer(interp);
            }
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
    /// A real-time session's guest is sleeping. Poll again after this host duration; polling
    /// earlier is harmless. Virtual-time sessions never report this.
    Sleeping(std::time::Duration),
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
        // Advancing to physical time cannot rewind, and the clock horizon is centuries away.
        let _ = environment.sync_host_time();
        let outcome = match guest.poll(environment) {
            GuestPoll::Ready(outcome) => Some(outcome),
            GuestPoll::Exhausted => None,
            GuestPoll::Blocked(WaitReason::Timer(deadline)) => {
                return SessionPoll::Sleeping(
                    environment
                        .host_time_until(deadline)
                        .unwrap_or(std::time::Duration::ZERO),
                );
            }
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
                let (status, host) = match *outcome {
                    GuestOutcome::Exit { status, host } => (status, host),
                    GuestOutcome::ThreadReturn { host } => (0, host),
                    GuestOutcome::Exec { .. } => {
                        unreachable!("buffered exec is rejected before transfer")
                    }
                };
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
    fn cancelled_process_snapshots_release_independent_dynamic_reservations() {
        let mut original = Interp::new();
        assert!(original.resources.reserve_memory(300));
        let mut process = WasmProcess::new("/fixture".into(), Vec::new());
        process.reserved = GuestReservation {
            fixed: 100,
            dynamic: Arc::new(AtomicU64::new(200)),
        };
        let guest_owner = process.reserved.clone();
        let mut snapshot = original.clone();
        let mut snapshot_process = process.clone();
        process.release_owned_memory(&mut original);
        assert_eq!(original.resources.memory_mark(), 0);
        assert_eq!(guest_owner.dynamic.load(Ordering::Relaxed), 0);
        assert_eq!(snapshot.resources.memory_mark(), 300);
        snapshot_process.release_owned_memory(&mut snapshot);
        assert_eq!(snapshot.resources.memory_mark(), 0);
        process.release_owned_memory(&mut original);
        snapshot_process.release_owned_memory(&mut snapshot);
        assert_eq!(original.resources.memory_mark(), 0);
        assert_eq!(snapshot.resources.memory_mark(), 0);
    }

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
