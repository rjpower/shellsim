//! Standalone deterministic pthread proof; it does not change shellsim's runtime.
//!
//! Each thread owns a live Wasmtime continuation. Only FIFO polling runs guest
//! code. Shared memory is bounded and charged once, and raw waits are rejected.

use std::collections::{BTreeMap, VecDeque};
use std::future::{poll_fn, Future};
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use wasmtime::{
    Caller, Config, Engine, Error, ExternType, Linker, MemoryType, Module, SharedMemory, Store,
    StoreLimits, StoreLimitsBuilder,
};

const MEMORY_BYTES: usize = 16 * 1024 * 1024;
const BUDGET_BYTES: usize = 128 * 1024 * 1024;
const THREAD_BYTES: usize = 512 * 1024 + 128 * 1024;
const MAX_THREADS: u32 = 16;
const MAX_TURNS: usize = 10_000;
const OUTPUT_BYTES: usize = 64 * 1024;

type Execution = Pin<Box<dyn Future<Output = Result<(), Error>> + Send>>;

#[derive(Clone, Copy)]
struct Waiter {
    address: u32,
    sequence: u64,
    deadline: Option<u64>,
    result: Option<i32>,
}

struct Process {
    memory: SharedMemory,
    waits: BTreeMap<u32, Waiter>,
    spawns: Vec<(u32, u32)>,
    runnable: VecDeque<u32>,
    next_tid: u32,
    now: u64,
    output: Vec<u8>,
    init_owner: Option<u32>,
    init_failed: bool,
    reserved: usize,
    waits_started: u64,
    notifications: u64,
    timed_out: u64,
}

impl Process {
    fn reserve(&mut self, bytes: usize) -> Result<(), Error> {
        self.reserved = self
            .reserved
            .checked_add(bytes)
            .ok_or_else(|| Error::msg("resource overflow"))?;
        if self.reserved > BUDGET_BYTES {
            return Err(Error::msg("thread process memory budget exhausted"));
        }
        Ok(())
    }

    fn range(&self, address: u32, length: usize) -> Result<usize, Error> {
        let start = address as usize;
        let end = start
            .checked_add(length)
            .ok_or_else(|| Error::msg("guest address overflow"))?;
        if end > self.memory.data_size() {
            return Err(Error::msg("guest address out of bounds"));
        }
        Ok(start)
    }

    fn read_u32(&self, address: u32) -> Result<u32, Error> {
        let start = self.range(address, 4)?;
        if start % 4 != 0 {
            return Err(Error::msg("unaligned guest u32"));
        }
        // SharedMemory is page aligned. All accesses use atomics, and this
        // runner polls one guest at a time without any host execution threads.
        let value = unsafe {
            &*(self
                .memory
                .data()
                .as_ptr()
                .cast::<u8>()
                .add(start)
                .cast::<AtomicU32>())
        };
        Ok(value.load(Ordering::SeqCst))
    }

    fn write(&self, address: u32, data: &[u8]) -> Result<(), Error> {
        let start = self.range(address, data.len())?;
        for (offset, byte) in data.iter().enumerate() {
            let cell = unsafe {
                &*(self
                    .memory
                    .data()
                    .as_ptr()
                    .cast::<u8>()
                    .add(start + offset)
                    .cast::<AtomicU8>())
            };
            cell.store(*byte, Ordering::SeqCst);
        }
        Ok(())
    }

    fn bytes(&self, address: u32, length: usize) -> Result<Vec<u8>, Error> {
        let start = self.range(address, length)?;
        Ok((0..length)
            .map(|offset| {
                let cell = unsafe {
                    &*(self
                        .memory
                        .data()
                        .as_ptr()
                        .cast::<u8>()
                        .add(start + offset)
                        .cast::<AtomicU8>())
                };
                cell.load(Ordering::SeqCst)
            })
            .collect())
    }
}

struct Host {
    process: Arc<Mutex<Process>>,
    tid: u32,
    limits: StoreLimits,
}

#[derive(Debug)]
struct Exit(i32);
impl std::fmt::Display for Exit {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(output, "virtual exit {}", self.0)
    }
}
impl std::error::Error for Exit {}

fn audit(bytes: &[u8]) -> Result<(), Error> {
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload? {
            for operation in body.get_operators_reader()? {
                match operation? {
                    wasmparser::Operator::MemoryAtomicWait32 { .. }
                    | wasmparser::Operator::MemoryAtomicWait64 { .. }
                    | wasmparser::Operator::MemoryAtomicNotify { .. } => {
                        return Err(Error::msg(
                            "raw atomic wait/notify is outside the scheduler ABI",
                        ));
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

fn linker(
    engine: &Engine,
    module: &Module,
    store: &mut Store<Host>,
) -> Result<Linker<Host>, Error> {
    let mut linker = Linker::new(engine);
    linker.define(
        &*store,
        "env",
        "memory",
        store.data().process.lock().unwrap().memory.clone(),
    )?;
    linker.func_wrap(
        "wasi",
        "thread-spawn",
        |caller: Caller<'_, Host>, argument: u32| -> Result<i32, Error> {
            let mut process = caller.data().process.lock().unwrap();
            if process.next_tid >= MAX_THREADS {
                return Ok(-1);
            }
            process.reserve(THREAD_BYTES)?;
            let tid = process.next_tid;
            process.next_tid += 1;
            process.spawns.push((tid, argument));
            Ok(tid as i32)
        },
    )?;
    linker.func_wrap_async(
        "shellsim_threads_v1",
        "wait32",
        |caller: Caller<'_, Host>, (address, expected, timeout): (u32, u32, i64)| {
            Box::new(async move {
                let state = caller.data().process.clone();
                let tid = caller.data().tid;
                {
                    let mut process = state.lock().unwrap();
                    if process.read_u32(address)? != expected {
                        return Ok::<i32, Error>(1);
                    }
                    if timeout == 0 {
                        return Ok(2);
                    }
                    let deadline = if timeout < 0 {
                        None
                    } else {
                        Some(
                            process
                                .now
                                .checked_add(timeout as u64)
                                .ok_or_else(|| Error::msg("virtual deadline overflow"))?,
                        )
                    };
                    process.waits_started += 1;
                    let sequence = process.waits_started;
                    process.waits.insert(
                        tid,
                        Waiter {
                            address,
                            sequence,
                            deadline,
                            result: None,
                        },
                    );
                }
                poll_fn(move |_| {
                    let mut process = state.lock().unwrap();
                    let waiter = process.waits.get(&tid).expect("registered waiter");
                    if let Some(result) = waiter.result {
                        process.waits.remove(&tid);
                        Poll::Ready(Ok(result))
                    } else {
                        Poll::Pending
                    }
                })
                .await
            })
        },
    )?;
    linker.func_wrap(
        "shellsim_threads_v1",
        "notify",
        |caller: Caller<'_, Host>, address: u32, count: u32| -> Result<i32, Error> {
            let mut process = caller.data().process.lock().unwrap();
            let _ = process.read_u32(address)?;
            let mut waiting: Vec<(u64, u32)> = process
                .waits
                .iter()
                .filter(|(_, waiter)| waiter.address == address && waiter.result.is_none())
                .map(|(tid, waiter)| (waiter.sequence, *tid))
                .collect();
            waiting.sort_unstable();
            let mut woken = 0u32;
            for (_, tid) in waiting.into_iter().take(count as usize) {
                process.waits.get_mut(&tid).unwrap().result = Some(0);
                process.runnable.push_back(tid);
                woken += 1;
            }
            process.notifications += woken as u64;
            Ok(woken as i32)
        },
    )?;
    linker.func_wrap(
        "wasi_snapshot_preview1",
        "clock_time_get",
        |caller: Caller<'_, Host>,
         clock: u32,
         _precision: u64,
         target: u32|
         -> Result<i32, Error> {
            if clock > 1 {
                return Ok(28);
            }
            let process = caller.data().process.lock().unwrap();
            process.write(target, &process.now.to_le_bytes())?;
            Ok(0)
        },
    )?;
    linker.func_wrap(
        "wasi_snapshot_preview1",
        "fd_write",
        |caller: Caller<'_, Host>,
         fd: u32,
         vectors: u32,
         count: u32,
         written: u32|
         -> Result<i32, Error> {
            if !(fd == 1 || fd == 2) || count > 32 {
                return Ok(8);
            }
            let mut process = caller.data().process.lock().unwrap();
            let mut total = 0u32;
            for index in 0..count {
                let vector = vectors
                    .checked_add(index * 8)
                    .ok_or_else(|| Error::msg("iovec overflow"))?;
                let address = process.read_u32(vector)?;
                let length = process.read_u32(vector + 4)?;
                if process
                    .output
                    .len()
                    .checked_add(length as usize)
                    .is_none_or(|size| size > OUTPUT_BYTES)
                {
                    return Err(Error::msg("bounded guest output exhausted"));
                }
                let data = process.bytes(address, length as usize)?;
                process.output.extend(data);
                total = total
                    .checked_add(length)
                    .ok_or_else(|| Error::msg("output overflow"))?;
            }
            process.write(written, &total.to_le_bytes())?;
            Ok(0)
        },
    )?;
    linker.func_wrap(
        "wasi_snapshot_preview1",
        "fd_fdstat_get",
        |caller: Caller<'_, Host>, fd: u32, target: u32| -> Result<i32, Error> {
            if fd > 2 {
                return Ok(8);
            }
            let process = caller.data().process.lock().unwrap();
            let mut data = [0u8; 24];
            data[0] = 2;
            process.write(target, &data)?;
            Ok(0)
        },
    )?;
    linker.func_wrap(
        "wasi_snapshot_preview1",
        "proc_exit",
        |_caller: Caller<'_, Host>, status: i32| -> Result<(), Error> {
            Err(Error::new(Exit(status)))
        },
    )?;
    for import in module.imports() {
        if linker
            .get(&mut *store, import.module(), import.name())
            .is_ok()
        {
            continue;
        }
        if import.module() != "wasi_snapshot_preview1" {
            return Err(Error::msg(format!(
                "unsupported import {}.{}",
                import.module(),
                import.name()
            )));
        }
        let ExternType::Func(ty) = import.ty() else {
            return Err(Error::msg("unsupported import type"));
        };
        let name = import.name().to_owned();
        linker.func_new(import.module(), import.name(), ty, move |_caller, _, _| {
            Err(Error::msg(format!("unsupported WASI operation {name}")))
        })?;
    }
    Ok(linker)
}

fn execution(
    engine: Engine,
    module: Module,
    process: Arc<Mutex<Process>>,
    tid: u32,
    argument: u32,
) -> Execution {
    Box::pin(async move {
        let mut store = Store::new(
            &engine,
            Host {
                process: process.clone(),
                tid,
                limits: StoreLimitsBuilder::new()
                    .table_elements(4096)
                    .instances(1)
                    .tables(1)
                    .build(),
            },
        );
        store.limiter(|host| &mut host.limits);
        store.set_fuel(1_000_000_000)?;
        store.fuel_async_yield_interval(Some(10_000))?;
        let linker = linker(&engine, &module, &mut store)?;
        poll_fn(|_| {
            let mut process = process.lock().unwrap();
            if process.init_failed {
                return Poll::Ready(Err(Error::msg("prior process initialization failed")));
            }
            if process.init_owner.is_none() {
                process.init_owner = Some(tid);
                Poll::Ready(Ok(()))
            } else {
                Poll::Pending
            }
        })
        .await?;
        let result = linker.instantiate_async(&mut store, &module).await;
        {
            let mut process = process.lock().unwrap();
            assert_eq!(process.init_owner, Some(tid));
            process.init_owner = None;
            process.init_failed |= result.is_err();
        }
        let instance = result?;
        let result = if tid == 0 {
            instance
                .get_typed_func::<(), ()>(&mut store, "_start")?
                .call_async(&mut store, ())
                .await
        } else {
            instance
                .get_typed_func::<(u32, u32), ()>(&mut store, "wasi_thread_start")?
                .call_async(&mut store, (tid, argument))
                .await
        };
        match result {
            Err(error)
                if tid == 0
                    && error
                        .downcast_ref::<Exit>()
                        .is_some_and(|status| status.0 == 0) =>
            {
                Ok(())
            }
            result => result,
        }
    })
}

fn main() -> Result<(), Error> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| Error::msg("supply explicit pthread fixture path"))?;
    let bytes = std::fs::read(path)?;
    if bytes.len() > 1024 * 1024 {
        return Err(Error::msg("fixture input limit"));
    }
    audit(&bytes)?;
    let mut config = Config::new();
    config
        .consume_fuel(true)
        .wasm_threads(true)
        .shared_memory(true)
        .max_wasm_stack(128 * 1024)
        .async_stack_size(512 * 1024);
    let engine = Engine::new(&config)?;
    let scratch = bytes
        .len()
        .checked_mul(65)
        .and_then(|size| size.checked_add(4096))
        .ok_or_else(|| Error::msg("compiler budget overflow"))?;
    if scratch + MEMORY_BYTES + THREAD_BYTES + OUTPUT_BYTES > BUDGET_BYTES {
        return Err(Error::msg("compiler memory reservation exhausted"));
    }
    let module = Module::new(&engine, &bytes)?;
    let image = module.image_range();
    let compiled = image.end as usize - image.start as usize;
    let retained = compiled
        .checked_add(bytes.len() * 2 + 4096)
        .ok_or_else(|| Error::msg("compiled image accounting overflow"))?;
    // Reserve the full shared maximum before constructing SharedMemory.
    if retained + MEMORY_BYTES + THREAD_BYTES + OUTPUT_BYTES > BUDGET_BYTES {
        return Err(Error::msg("retained process reservation exhausted"));
    }
    let memory = SharedMemory::new(&engine, MemoryType::shared(256, 256))?;
    let process = Arc::new(Mutex::new(Process {
        memory,
        waits: BTreeMap::new(),
        spawns: Vec::new(),
        runnable: VecDeque::from([0]),
        next_tid: 1,
        now: 0,
        output: Vec::new(),
        init_owner: None,
        init_failed: false,
        reserved: retained + MEMORY_BYTES + THREAD_BYTES + OUTPUT_BYTES,
        waits_started: 0,
        notifications: 0,
        timed_out: 0,
    }));
    let mut threads: BTreeMap<u32, Execution> = BTreeMap::from([(
        0,
        execution(engine.clone(), module.clone(), process.clone(), 0, 0),
    )]);
    let mut turns = 0;
    while !threads.is_empty() {
        if turns >= MAX_TURNS {
            return Err(Error::msg("aggregate scheduler CPU limit"));
        }
        let next = process.lock().unwrap().runnable.pop_front();
        if let Some(tid) = next {
            turns += 1;
            let result = threads
                .get_mut(&tid)
                .unwrap()
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()));
            match result {
                Poll::Ready(result) => {
                    result?;
                    threads.remove(&tid);
                }
                Poll::Pending => {
                    let mut process = process.lock().unwrap();
                    if !process
                        .waits
                        .get(&tid)
                        .is_some_and(|waiter| waiter.result.is_none())
                    {
                        process.runnable.push_back(tid);
                    }
                }
            }
            let spawns = std::mem::take(&mut process.lock().unwrap().spawns);
            for (child, argument) in spawns {
                threads.insert(
                    child,
                    execution(
                        engine.clone(),
                        module.clone(),
                        process.clone(),
                        child,
                        argument,
                    ),
                );
                process.lock().unwrap().runnable.push_back(child);
            }
        } else {
            let mut process = process.lock().unwrap();
            let deadline = process
                .waits
                .values()
                .filter_map(|waiter| waiter.deadline)
                .min()
                .ok_or_else(|| Error::msg("virtual pthread deadlock"))?;
            process.now = deadline;
            let mut expired: Vec<(u64, u32)> = process
                .waits
                .iter()
                .filter(|(_, waiter)| {
                    waiter.deadline.is_some_and(|value| value <= deadline)
                        && waiter.result.is_none()
                })
                .map(|(tid, waiter)| (waiter.sequence, *tid))
                .collect();
            expired.sort_unstable();
            for (_, tid) in expired {
                process.waits.get_mut(&tid).unwrap().result = Some(2);
                process.runnable.push_back(tid);
                process.timed_out += 1;
            }
        }
    }
    let process = process.lock().unwrap();
    let expected = b"two pthreads: join, mutex, condition, TLS, virtual timeout passed\n";
    if process.output != expected
        || process.now != 5_000_000
        || process.next_tid != 3
        || process.timed_out != 1
    {
        return Err(Error::msg(format!(
            "pthread proof mismatch: output={:?}, time={}, threads={}, timeouts={}",
            process.output,
            process.now,
            process.next_tid - 1,
            process.timed_out
        )));
    }
    println!("{}", String::from_utf8_lossy(&process.output).trim_end());
    println!(
        "virtual_ns={} spawned={} waits={} notifications={} turns={} reserved_bytes={}",
        process.now,
        process.next_tid - 1,
        process.waits_started,
        process.notifications,
        turns,
        process.reserved
    );
    Ok(())
}
