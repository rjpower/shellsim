//! Scheduler state for the versioned, deterministic pthread ABI.
//!
//! Wait comparison and registration happen within one serial guest poll. No
//! operation parks a host thread. Deadlines use the virtual monotonic clock;
//! the process owner schedules their wake events and drops this state on exit.

use std::collections::BTreeMap;

pub(super) const NAMESPACE: &str = "shellsim_threads_v1";
pub(super) const MAX_THREADS: usize = 16;
// The pinned wasi-libc pthread_create checks the proposal's 29-bit positive TIDs.
const MAX_THREAD_ID: u32 = 0x1fff_ffff;

/// Raw Wasmtime waits use host blocking and time. Reject them before compiling
/// either a main module or a side module, including on a compilation cache hit.
pub(super) fn reject_raw_waits(bytes: &[u8]) -> Result<(), wasmtime::Error> {
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload? {
            for operation in body.get_operators_reader()? {
                if matches!(
                    operation?,
                    wasmparser::Operator::MemoryAtomicWait32 { .. }
                        | wasmparser::Operator::MemoryAtomicWait64 { .. }
                        | wasmparser::Operator::MemoryAtomicNotify { .. }
                ) {
                    return Err(wasmtime::Error::msg(
                        "raw atomic wait/notify is outside the scheduler ABI",
                    ));
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WaitResult {
    Notified = 0,
    Unequal = 1,
    TimedOut = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WaitError {
    AlreadyWaiting,
    Limit,
    DeadlineOverflow,
    SequenceOverflow,
}

#[derive(Clone, Copy)]
struct Waiter {
    address: u32,
    sequence: u64,
    deadline: Option<u64>,
    result: Option<WaitResult>,
}

/// At most one live wait per virtual thread. Completed waits remain registered
/// until their continuation consumes the result, preventing duplicate wakes.
#[derive(Default)]
pub(super) struct Waits {
    entries: BTreeMap<u32, Waiter>,
    sequence: u64,
}

impl Waits {
    /// Caller must load `observed` atomically and register without yielding in
    /// between. The process poll gate makes that compare/enqueue indivisible.
    pub(super) fn begin(
        &mut self,
        tid: u32,
        address: u32,
        observed: u32,
        expected: u32,
        timeout_ns: i64,
        now: u64,
    ) -> Result<Option<WaitResult>, WaitError> {
        if self.entries.contains_key(&tid) {
            return Err(WaitError::AlreadyWaiting);
        }
        if observed != expected {
            return Ok(Some(WaitResult::Unequal));
        }
        if timeout_ns == 0 {
            return Ok(Some(WaitResult::TimedOut));
        }
        if self.entries.len() >= MAX_THREADS {
            return Err(WaitError::Limit);
        }
        let deadline = if timeout_ns < 0 {
            None
        } else {
            Some(
                now.checked_add(timeout_ns as u64)
                    .ok_or(WaitError::DeadlineOverflow)?,
            )
        };
        let sequence = self
            .sequence
            .checked_add(1)
            .ok_or(WaitError::SequenceOverflow)?;
        self.sequence = sequence;
        self.entries.insert(
            tid,
            Waiter {
                address,
                sequence,
                deadline,
                result: None,
            },
        );
        Ok(None)
    }

    /// Return newly notified thread IDs in registration order. The bounded
    /// process queue schedules their live continuations after this host call.
    pub(super) fn notify(&mut self, address: u32, count: u32) -> Vec<u32> {
        let mut waiting: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, waiter)| waiter.address == address && waiter.result.is_none())
            .map(|(tid, waiter)| (waiter.sequence, *tid))
            .collect();
        waiting.sort_unstable();
        waiting
            .into_iter()
            .take(count as usize)
            .map(|(_, tid)| {
                self.entries
                    .get_mut(&tid)
                    .expect("registered waiter")
                    .result = Some(WaitResult::Notified);
                tid
            })
            .collect()
    }

    pub(super) fn expire(&mut self, now: u64) -> Vec<u32> {
        let mut waiting: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, waiter)| {
                waiter.result.is_none() && waiter.deadline.is_some_and(|deadline| deadline <= now)
            })
            .map(|(tid, waiter)| (waiter.sequence, *tid))
            .collect();
        waiting.sort_unstable();
        waiting
            .into_iter()
            .map(|(_, tid)| {
                self.entries
                    .get_mut(&tid)
                    .expect("registered waiter")
                    .result = Some(WaitResult::TimedOut);
                tid
            })
            .collect()
    }

    pub(super) fn earliest_deadline(&self) -> Option<u64> {
        self.entries
            .values()
            .filter(|waiter| waiter.result.is_none())
            .filter_map(|waiter| waiter.deadline)
            .min()
    }

    pub(super) fn blocked(&self, tid: u32) -> bool {
        self.entries
            .get(&tid)
            .is_some_and(|waiter| waiter.result.is_none())
    }

    pub(super) fn take_result(&mut self, tid: u32) -> Option<WaitResult> {
        let result = self.entries.get(&tid)?.result?;
        self.entries.remove(&tid);
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notification_is_fifo_and_cannot_wake_twice() {
        let mut waits = Waits::default();
        waits.begin(9, 4, 7, 7, -1, 0).unwrap();
        waits.begin(2, 4, 7, 7, -1, 0).unwrap();
        assert_eq!(waits.notify(4, 1), vec![9]);
        assert!(!waits.blocked(9));
        assert!(waits.blocked(2));
        assert_eq!(waits.notify(4, u32::MAX), vec![2]);
        assert!(waits.notify(4, 1).is_empty());
        assert_eq!(waits.take_result(9), Some(WaitResult::Notified));
        assert_eq!(waits.take_result(9), None);
    }

    #[test]
    fn unequal_value_avoids_lost_wake_and_deadlines_are_virtual() {
        let mut waits = Waits::default();
        assert_eq!(
            waits.begin(1, 4, 8, 7, -1, 0),
            Ok(Some(WaitResult::Unequal))
        );
        waits.begin(1, 4, 7, 7, 5, 10).unwrap();
        assert_eq!(waits.earliest_deadline(), Some(15));
        assert!(waits.expire(14).is_empty());
        assert_eq!(waits.expire(15), vec![1]);
        assert_eq!(waits.earliest_deadline(), None);
        assert!(waits.notify(4, 1).is_empty());
        assert_eq!(waits.take_result(1), Some(WaitResult::TimedOut));
        assert_eq!(
            waits.begin(1, 4, 7, 7, 1, u64::MAX),
            Err(WaitError::DeadlineOverflow)
        );
        assert_eq!(
            waits.begin(1, 4, 7, 7, 0, 0),
            Ok(Some(WaitResult::TimedOut))
        );
    }

    #[test]
    fn waits_are_bounded_and_duplicate_registration_rejected() {
        let mut waits = Waits::default();
        for tid in 0..MAX_THREADS as u32 {
            waits.begin(tid, 4, 0, 0, -1, 0).unwrap();
        }
        assert_eq!(
            waits.begin(0, 4, 0, 0, -1, 0),
            Err(WaitError::AlreadyWaiting)
        );
        assert_eq!(
            waits.begin(MAX_THREADS as u32, 4, 0, 0, -1, 0),
            Err(WaitError::Limit)
        );
    }
}

/// Static threaded ABI memory must be imported, bounded and wasm32. Dynamic
/// graph/TLS replay needs a separate coherent ABI before sharing these Stores.
pub(super) struct Profile {
    pub(super) minimum_pages: u64,
    pub(super) maximum_pages: u64,
}

pub(super) fn profile(bytes: &[u8]) -> Result<Option<Profile>, wasmtime::Error> {
    let mut shared = None;
    let mut scheduler = false;
    let mut dynamic = false;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        match payload? {
            wasmparser::Payload::ImportSection(imports) => {
                for import in imports.into_imports() {
                    let import = import?;
                    scheduler |= import.module == NAMESPACE;
                    dynamic |= super::dynamic::Abi::from_namespace(import.module).is_some();
                    if let wasmparser::TypeRef::Memory(memory) = import.ty {
                        if memory.shared {
                            let maximum = memory.maximum.ok_or_else(|| {
                                wasmtime::Error::msg("thread memory requires a fixed maximum")
                            })?;
                            if import.module != "env"
                                || import.name != "memory"
                                || memory.memory64
                                || shared.is_some()
                                || maximum > 1024
                            {
                                return Err(wasmtime::Error::msg(
                                    "unsupported thread shared-memory ABI",
                                ));
                            }
                            shared = Some(Profile {
                                minimum_pages: memory.initial,
                                maximum_pages: maximum,
                            });
                        }
                    }
                }
            }
            wasmparser::Payload::CodeSectionEntry(body) if shared.is_some() => {
                for operation in body.get_operators_reader()? {
                    if matches!(
                        operation?,
                        wasmparser::Operator::TableSet { .. }
                            | wasmparser::Operator::TableGrow { .. }
                            | wasmparser::Operator::TableCopy { .. }
                            | wasmparser::Operator::TableInit { .. }
                            | wasmparser::Operator::TableFill { .. }
                            | wasmparser::Operator::ElemDrop { .. }
                    ) {
                        return Err(wasmtime::Error::msg(
                            "thread table mutation requires coherent process-wide replay",
                        ));
                    }
                }
            }
            wasmparser::Payload::DataSection(data) if shared.is_some() => {
                for segment in data {
                    if matches!(segment?.kind, wasmparser::DataKind::Active { .. }) {
                        return Err(wasmtime::Error::msg(
                            "thread data must use guarded passive initialization",
                        ));
                    }
                }
            }
            wasmparser::Payload::MemorySection(memories) => {
                for memory in memories {
                    if memory?.shared {
                        return Err(wasmtime::Error::msg(
                            "thread memory must be process-owned and imported",
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    if shared.is_some() && (!scheduler || dynamic) {
        return Err(wasmtime::Error::msg(
            "shared memory requires the static shellsim_threads_v1 ABI",
        ));
    }
    if shared.is_none() && scheduler {
        return Err(wasmtime::Error::msg(
            "thread scheduler imports require shared memory",
        ));
    }
    Ok(shared)
}

use super::*;
use wasmtime::{MemoryType, SharedMemory};

// Prepay sixteen reusable Store/stack/table slots for the process lifetime.
// Completed workers free their physical Store; later distinct IDs reuse a slot.
pub(super) const THREAD_HOST_BYTES: u64 = ASYNC_STACK_BYTES as u64 + 128 * 1024 + 4096 * 16;

struct Control {
    memory: SharedMemory,
    waits: Waits,
    ready: VecDeque<u32>,
    spawns: Vec<(u32, u32)>,
    next_tid: u32,
    live: usize,
    init_owner: Option<u32>,
    init_failed: bool,
}

#[derive(Clone)]
pub(super) struct ThreadContext {
    control: Arc<Mutex<Control>>,
    tid: u32,
}

impl ThreadContext {
    pub(super) fn memory(&self) -> SharedMemory {
        self.control.lock().expect("thread control").memory.clone()
    }
}

pub(super) struct ThreadGroup {
    control: Arc<Mutex<Control>>,
    guests: BTreeMap<u32, Guest>,
    blocked: BTreeMap<u32, WaitReason>,
    template: Template,
    module: Module,
    path: String,
    scheduled: Option<crate::clock::EventId>,
}

struct Template {
    machine: MachineAccess,
    cwd: String,
    args: Vec<Vec<u8>>,
    environment: Vec<Vec<u8>>,
    initial_fuel: u64,
    retained: Arc<AtomicU64>,
}

pub(super) fn start(
    host: Host,
    module: Module,
    path: String,
    profile: Profile,
    reserved: GuestReservation,
) -> Result<Guest, Error> {
    if !matches!(host.stdio, Stdio::Descriptors) {
        return Err(Error::msg(
            "threaded buffered display sessions require shared stream state",
        ));
    }
    let memory = SharedMemory::new(
        command_engine(),
        MemoryType::shared(profile.minimum_pages as u32, profile.maximum_pages as u32),
    )?;
    let control = Arc::new(Mutex::new(Control {
        memory,
        waits: Waits::default(),
        ready: VecDeque::from([0]),
        spawns: Vec::new(),
        next_tid: 1,
        live: 1,
        init_owner: None,
        init_failed: false,
    }));
    let machine = host.machine.clone();
    let template = Template {
        machine: machine.clone(),
        cwd: host.cwd.clone(),
        args: host.args.clone(),
        environment: host.environment.clone(),
        initial_fuel: host.initial_fuel,
        retained: host.retained.clone(),
    };
    let mut group = ThreadGroup {
        control,
        guests: BTreeMap::new(),
        blocked: BTreeMap::new(),
        template,
        module,
        path,
        scheduled: None,
    };
    group.insert(0, 0, Some(host));
    Ok(Guest {
        execution: Execution::Threads(Box::new(group)),
        machine,
        reserved,
    })
}

impl ThreadGroup {
    fn insert(&mut self, tid: u32, argument: u32, main: Option<Host>) {
        let machine = self.template.machine.fork_thread();
        let mut host = main.unwrap_or_else(|| Host {
            machine: machine.clone(),
            cwd: self.template.cwd.clone(),
            args: self.template.args.clone(),
            environment: self.template.environment.clone(),
            stdio: Stdio::Descriptors,
            diagnostic: Vec::new(),
            initial_fuel: self.template.initial_fuel,
            closed_stdio: BTreeSet::new(),
            open_files: BTreeSet::new(),
            buffered_streams: BTreeMap::new(),
            limits: limits::GuestLimits::new(
                StoreLimitsBuilder::new()
                    .memory_size(0)
                    .table_elements(4096)
                    .memories(1)
                    .tables(1)
                    .instances(1)
                    .build(),
                0,
            ),
            interaction: None,
            dynamic: dynamic::Dynamic::new(None, self.template.retained.clone()),
            ffi: super::ffi::State::default(),
            fibers: super::fibers::Budget::default(),
            thread: None,
            retained: self.template.retained.clone(),
        });
        host.limits = limits::GuestLimits::new(
            StoreLimitsBuilder::new()
                .memory_size(0)
                .table_elements(4096)
                .memories(1)
                .tables(1)
                .instances(1)
                .build(),
            0,
        );
        host.machine = machine.clone();
        host.thread = Some(ThreadContext {
            control: self.control.clone(),
            tid,
        });
        self.guests.insert(
            tid,
            Guest {
                execution: Execution::Single(Box::pin(execute_thread(
                    host,
                    self.module.clone(),
                    self.path.clone(),
                    argument,
                ))),
                machine,
                reserved: GuestReservation::default(),
            },
        );
    }

    pub(super) fn cancel_timer(&mut self, interp: &mut Interp) {
        if let Some(event) = self.scheduled.take() {
            interp.clock.cancel(event);
        }
    }

    pub(super) fn poll(&mut self, interp: &mut Interp) -> GuestPoll {
        self.cancel_timer(interp);
        let now = interp.clock.monotonic_ns();
        {
            let mut control = self.control.lock().expect("thread control");
            let expired = control.waits.expire(now);
            control.ready.extend(expired);
        }
        // The process was rescheduled by its aggregate descriptor/timer reason.
        // Retrying bounded nonblocking host operations is safe; it cannot perform
        // duplicate I/O because suspended host calls retain their continuation.
        for tid in std::mem::take(&mut self.blocked).into_keys() {
            self.control
                .lock()
                .expect("thread control")
                .ready
                .push_back(tid);
        }
        let next = self
            .control
            .lock()
            .expect("thread control")
            .ready
            .pop_front();
        if let Some(tid) = next {
            let result = self.guests.get_mut(&tid).expect("live thread").poll(interp);
            match result {
                GuestPoll::Ready(outcome) => match *outcome {
                    GuestOutcome::ThreadReturn { .. } if tid != 0 => {
                        self.guests.remove(&tid);
                        self.control.lock().expect("thread control").live -= 1;
                    }
                    other => {
                        self.cancel_timer(interp);
                        return GuestPoll::Ready(Box::new(other));
                    }
                },
                GuestPoll::Exhausted => return GuestPoll::Exhausted,
                GuestPoll::Blocked(reason) => {
                    self.blocked.insert(tid, reason);
                }
                GuestPoll::Pending => {
                    let mut control = self.control.lock().expect("thread control");
                    if !control.waits.blocked(tid) {
                        control.ready.push_back(tid);
                    }
                }
            }
            let spawns = std::mem::take(&mut self.control.lock().expect("thread control").spawns);
            for (tid, argument) in spawns {
                self.insert(tid, argument, None);
                self.control
                    .lock()
                    .expect("thread control")
                    .ready
                    .push_back(tid);
            }
        }
        let control = self.control.lock().expect("thread control");
        if !control.ready.is_empty() {
            return GuestPoll::Pending;
        }
        let deadline = control.waits.earliest_deadline();
        drop(control);
        let mut reasons: Vec<_> = self.blocked.values().cloned().collect();
        if let Some(deadline) = deadline {
            if self.scheduled.is_none() {
                self.scheduled = interp
                    .clock
                    .schedule_wake_after(
                        u64::from(interp.process.pid),
                        deadline.saturating_sub(now),
                    )
                    .ok();
                if self.scheduled.is_none() {
                    return GuestPoll::Exhausted;
                }
            }
            reasons.push(WaitReason::Timer(deadline));
        }
        if reasons.is_empty() {
            // An indefinite futex wait has no external wake source, but remains
            // cancellable through the owning process's normal lifecycle.
            GuestPoll::Blocked(WaitReason::Any(Vec::new()))
        } else if reasons.len() == 1 {
            GuestPoll::Blocked(reasons.pop().expect("one reason"))
        } else {
            GuestPoll::Blocked(WaitReason::Any(reasons))
        }
    }
}

fn register(linker: &mut Linker<Host>) {
    linker
        .func_wrap(
            "wasi",
            "thread-spawn",
            |caller: Caller<'_, Host>, argument: u32| -> Result<i32, Error> {
                if !caller
                    .data()
                    .machine
                    .get()
                    .resources
                    .charge_cpu(MAX_THREADS as u64)
                {
                    return Err(exhausted());
                }
                let thread = caller.data().thread.as_ref().expect("thread host");
                let mut control = thread.control.lock().expect("thread control");
                if control.init_owner.is_some() || control.init_failed {
                    return Err(Error::msg(
                        "pthread spawn during process initialization is unsupported",
                    ));
                }
                if control.live >= MAX_THREADS || control.next_tid > MAX_THREAD_ID {
                    return Ok(-6);
                }
                if !caller.data().machine.get().resources.charge_cpu(4096) {
                    return Err(exhausted());
                }
                let tid = control.next_tid;
                control.next_tid = control
                    .next_tid
                    .checked_add(1)
                    .ok_or_else(|| Error::msg("thread ID overflow"))?;
                control.live += 1;
                control.spawns.push((tid, argument));
                Ok(tid as i32)
            },
        )
        .expect("thread spawn import");
    linker
        .func_wrap(
            NAMESPACE,
            "notify",
            |caller: Caller<'_, Host>, address: u32, count: u32| -> Result<i32, Error> {
                if !caller
                    .data()
                    .machine
                    .get()
                    .resources
                    .charge_cpu(MAX_THREADS as u64)
                {
                    return Err(exhausted());
                }
                let thread = caller.data().thread.as_ref().expect("thread host");
                let mut control = thread.control.lock().expect("thread control");
                atomic_word(&control.memory, address)?;
                let ready = control.waits.notify(address, count);
                let count = ready.len() as i32;
                control.ready.extend(ready);
                Ok(count)
            },
        )
        .expect("thread notify import");
    linker
        .func_wrap_async(
            NAMESPACE,
            "wait32",
            |caller: Caller<'_, Host>, (address, expected, timeout): (u32, u32, i64)| {
                Box::new(async move {
                    let thread = caller.data().thread.as_ref().expect("thread host").clone();
                    let machine = caller.data().machine.clone();
                    let now = {
                        let mut interp = machine.get();
                        if !interp.resources.charge_cpu(MAX_THREADS as u64) {
                            return Err(exhausted());
                        }
                        interp.clock.monotonic_ns()
                    };
                    {
                        let mut control = thread.control.lock().expect("thread control");
                        let observed =
                            atomic_word(&control.memory, address)?.load(Ordering::SeqCst);
                        if let Some(result) = control
                            .waits
                            .begin(thread.tid, address, observed, expected, timeout, now)
                            .map_err(|error| Error::msg(format!("thread wait failed: {error:?}")))?
                        {
                            return Ok(result as i32);
                        }
                    }
                    poll_fn(|_| {
                        let mut control = thread.control.lock().expect("thread control");
                        match control.waits.take_result(thread.tid) {
                            Some(result) => Poll::Ready(Ok(result as i32)),
                            None => {
                                machine.signals().suspension = Some(Suspension::Yielded);
                                Poll::Pending
                            }
                        }
                    })
                    .await
                })
            },
        )
        .expect("thread wait import");
}

#[allow(unsafe_code)]
fn atomic_word(
    memory: &SharedMemory,
    address: u32,
) -> Result<&std::sync::atomic::AtomicU32, Error> {
    let offset = address as usize;
    if !offset.is_multiple_of(4)
        || offset
            .checked_add(4)
            .is_none_or(|end| end > memory.data_size())
    {
        return Err(Error::msg("unaligned or out-of-range thread wait address"));
    }
    // SharedMemory is page aligned with stable backing storage. Only one guest
    // continuation runs at a time; host word/byte accesses cannot overlap it.
    Ok(unsafe {
        &*memory
            .data()
            .as_ptr()
            .cast::<u8>()
            .add(offset)
            .cast::<std::sync::atomic::AtomicU32>()
    })
}

async fn execute_thread(host: Host, module: Module, path: String, argument: u32) -> GuestOutcome {
    let initial_fuel = host.initial_fuel;
    let thread = host.thread.as_ref().expect("thread host").clone();
    let machine = host.machine.clone();
    let mut store = Store::new(command_engine(), host);
    store.limiter(|host| &mut host.limits);
    store.set_fuel(initial_fuel).expect("fuel configured");
    store
        .fuel_async_yield_interval(Some(FUEL_YIELD_INTERVAL))
        .expect("fuel configured");
    store.call_hook(|mut store, hook| {
        if matches!(hook, CallHook::CallingHost) {
            charge_consumed_fuel(store.as_context_mut())?;
            let host = store.data_mut();
            let interp = host.machine.get();
            host.cwd = interp.process.cwd.clone();
            host.open_files = interp
                .process
                .fds
                .iter()
                .filter_map(|(fd, _)| (fd > 4).then_some(fd))
                .collect();
            host.closed_stdio = (0..3)
                .filter(|fd| interp.process.fds.get(*fd).is_err())
                .collect();
        } else if matches!(hook, CallHook::ReturningFromHost) {
            let mut interp = store.data_mut().machine.get();
            if let Some(signal) = interp.0.as_mut().and_then(Interp::take_default_termination) {
                return Err(Error::new(GuestExit(128 + signal.number())));
            }
        }
        Ok(())
    });
    let mut linker = build_linker(command_engine());
    register(&mut linker);
    posix_exec::register(&mut linker);
    posix_open::register(&mut linker);
    let result = async {
        linker.define(&store, "env", "memory", thread.memory())?;
        linker.define_unknown_imports_as_traps(&module)?;
        poll_fn(|_| {
            let mut control = thread.control.lock().expect("thread control");
            if control.init_failed {
                return Poll::Ready(Err(Error::msg("prior process initialization failed")));
            }
            if control.init_owner.is_none() {
                control.init_owner = Some(thread.tid);
                Poll::Ready(Ok(()))
            } else {
                machine.signals().suspension = Some(Suspension::Yielded);
                Poll::Pending
            }
        })
        .await?;
        let initialized = linker.instantiate_async(&mut store, &module).await;
        {
            let mut control = thread.control.lock().expect("thread control");
            assert_eq!(control.init_owner, Some(thread.tid));
            control.init_owner = None;
            control.init_failed |= initialized.is_err();
        }
        let instance = initialized?;
        if thread.tid == 0 {
            instance
                .get_typed_func::<(), ()>(&mut store, "_start")?
                .call_async(&mut store, ())
                .await
        } else {
            instance
                .get_typed_func::<(u32, u32), ()>(&mut store, "wasi_thread_start")?
                .call_async(&mut store, (thread.tid, argument))
                .await
        }
    }
    .await;
    let _ = charge_consumed_fuel(store.as_context_mut());
    let out_of_fuel = store.get_fuel().unwrap_or(0) == 0;
    finish_guest(
        result,
        store.into_data(),
        out_of_fuel,
        &path,
        thread.tid != 0,
    )
}
