//! Process-owned publication state for the SDK34 threaded dynamic cohort.
//!
//! Compiled modules and allocation addresses may cross thread Stores. Instances,
//! functions, tags and globals may not. Publication occurs only after process
//! initialization finishes; a suspended Store reconstructs its own handles at an
//! exclusive async resume checkpoint. TLS storage is prepaid for reusable thread
//! slots so replay never enters an allocator interrupted with its lock held.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use wasmtime::{AsContextMut, Error, Instance, Module, StoreContextMut};

pub(super) mod callbacks;
pub(super) mod executable;
pub(super) mod layout;
mod loading;
mod replay;

pub(super) const NAMESPACE: &str = "shellsim_dylink_v3";
pub(super) const MARKER: &[u8] = b"shellsim-wasi-sdk34-cpython3137-threads-v3";
pub(super) const MAX_MODULES: usize = 256;
pub(super) const MAX_TABLE_ELEMENTS: usize = 65_536;
pub(super) const MAX_MEMORY_PAGES: u64 = 4096;

#[derive(Default)]
pub(super) struct StoreState {
    pub(super) main: Option<Instance>,
    pub(super) ready: bool,
    pub(super) stack_bounds: Option<(u32, u32)>,
    generation: u64,
    owned_bytes: u64,
    main_symbols: BTreeMap<String, u32>,
    libraries: Vec<Loaded>,
    initializing: bool,
    loading: bool,
    error: Option<String>,
    strings_reserved: bool,
    callbacks: callbacks::StoreState,
    executable: executable::Bindings,
}

struct Loaded {
    instance: Instance,
    record: Arc<Record>,
    imports: Vec<replay::DeferredImport>,
    functions: BTreeMap<String, wasmtime::Func>,
    deferred: bool,
}

fn reserve_store(store: &mut StoreContextMut<'_, super::Host>, bytes: u64) -> Result<(), Error> {
    let total = store
        .data()
        .threaded_dynamic
        .owned_bytes
        .checked_add(bytes)
        .ok_or_else(|| Error::msg("threaded Store accounting overflow"))?;
    if !store.data().machine.get().resources.reserve_memory(bytes) {
        return Err(super::exhausted());
    }
    if store
        .data()
        .retained
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |owned| owned.checked_add(bytes),
        )
        .is_err()
    {
        store.data().machine.get().resources.release_memory(bytes);
        return Err(Error::msg("threaded process accounting overflow"));
    }
    store.data_mut().threaded_dynamic.owned_bytes = total;
    Ok(())
}

/// Called after a worker Store has been consumed and all async calls are gone.
pub(super) fn release_store(host: &mut super::Host) {
    let bytes = std::mem::take(&mut host.threaded_dynamic.owned_bytes);
    if bytes != 0 {
        host.machine.get().resources.release_memory(bytes);
        host.retained
            .fetch_sub(bytes, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Give every exported main function a deterministic process table address.
/// This runs separately in each Store; no function handle crosses Stores.
pub(super) fn prepare_main(
    mut store: StoreContextMut<'_, super::Host>,
    module: &Module,
    instance: Instance,
) -> Result<(), Error> {
    let table = instance
        .get_table(&mut store, "__indirect_function_table")
        .ok_or_else(|| Error::msg("threaded dynamic main requires exported function table"))?;
    let tls = store
        .data()
        .thread
        .as_ref()
        .expect("thread host")
        .main_tls
        .clone();
    for name in tls.iter() {
        if !store
            .data()
            .machine
            .get()
            .resources
            .charge_cpu(name.len() as u64 + 64)
        {
            return Err(super::exhausted());
        }
        let global = instance
            .get_global(&mut store, name)
            .ok_or_else(|| Error::msg("main TLS metadata requires a global export"))?;
        if !wasmtime::ValType::eq(global.ty(&store).content(), &wasmtime::ValType::I32) {
            return Err(Error::msg("main TLS export requires an i32 offset"));
        }
    }
    let exports = module
        .exports()
        .filter(|export| matches!(export.ty(), wasmtime::ExternType::Func(_)))
        .count();
    let name_bytes = module
        .exports()
        .filter(|export| matches!(export.ty(), wasmtime::ExternType::Func(_)))
        .fold(0u64, |total, export| {
            total.saturating_add(export.name().len() as u64)
        });
    let table_entries = table.size(&store);
    if !store.data().machine.get().resources.charge_cpu(
        table_entries
            .saturating_mul(128)
            .saturating_add((exports as u64).saturating_mul(128))
            .saturating_add(name_bytes),
    ) {
        return Err(super::exhausted());
    }
    // Include the scan map and both Store/process copies of variable-length
    // names. This conservative high-water charge lasts until Store teardown.
    let metadata = table_entries
        .saturating_mul(64)
        .saturating_add((exports as u64).saturating_mul(192))
        .saturating_add(name_bytes.saturating_mul(2))
        .saturating_add(4096);
    reserve_store(&mut store, metadata)?;
    let mut addresses = BTreeMap::new();
    for index in 0..table.size(&store) {
        if let Some(wasmtime::Ref::Func(Some(function))) = table.get(&mut store, index) {
            addresses
                .entry(function.to_raw(&mut store) as usize)
                .or_insert(index as u32);
        }
    }
    let mut symbols = BTreeMap::new();
    for export in module.exports() {
        if !matches!(export.ty(), wasmtime::ExternType::Func(_))
            || export.name() == layout::START_EXPORT
            || store
                .data()
                .thread
                .as_ref()
                .expect("thread host")
                .executable
                .forwarded_exports
                .contains(export.name())
        {
            continue;
        }
        let function = instance
            .get_func(&mut store, export.name())
            .expect("main function export");
        let key = function.to_raw(&mut store) as usize;
        let index = if let Some(index) = addresses.get(&key) {
            *index
        } else {
            let index = table.grow(&mut store, 1, wasmtime::Ref::Func(Some(function)))? as u32;
            addresses.insert(key, index);
            index
        };
        symbols.insert(export.name().to_owned(), index);
    }
    let thread = store.data().thread.as_ref().expect("thread host");
    thread
        .dynamic_process()
        .prepare_main(symbols.clone(), table.size(&store) as u32)?;
    store.data_mut().threaded_dynamic.main_symbols = symbols;
    Ok(())
}

/// Called while Wasmtime grants exclusive access to a suspended Store. Waiting
/// for the initialization owner yields to the virtual process scheduler.
pub(super) async fn checkpoint(mut store: StoreContextMut<'_, super::Host>) -> Result<(), Error> {
    // Nested guest initialization calls can enter call hooks when replay was
    // requested by thread_ready rather than by a hook. The owner already holds
    // the gate; reentering replay would reconstruct its partially installed record.
    if store.data().threaded_dynamic.initializing {
        return Ok(());
    }
    let thread = store.data().thread.as_ref().expect("thread host").clone();
    let process = thread.dynamic_process();
    let machine = store.data().machine.clone();
    std::future::poll_fn(|_| match process.resume_allowed(thread.id()) {
        Ok(true) => std::task::Poll::Ready(Ok(())),
        Ok(false) => {
            if !machine.get().resources.charge_cpu(32) {
                return std::task::Poll::Ready(Err(super::exhausted()));
            }
            machine.signals().suspension = Some(super::Suspension::Yielded);
            std::task::Poll::Pending
        }
        Err(error) => std::task::Poll::Ready(Err(error)),
    })
    .await?;
    // A new pthread must honor process initialization before libc establishes
    // TLS. Reconstruction itself begins only after thread_ready.
    if !store.data().threaded_dynamic.ready {
        return Ok(());
    }
    let (generation, count) = process.generation();
    if generation != store.data().threaded_dynamic.generation {
        if !store
            .data()
            .machine
            .get()
            .resources
            .charge_cpu((count as u64).saturating_mul(128).saturating_add(16))
        {
            return Err(super::exhausted());
        }
        reserve_store(&mut store, (count as u64).saturating_mul(32))?;
        let mut initialization = process.begin_replay(thread.id())?;
        initialization.started();
        let (_, records) = process.snapshot();
        let installed = store.data().threaded_dynamic.libraries.len();
        for record in records.into_iter().skip(installed) {
            if let Err(error) = replay::install(store.as_context_mut(), record, false, false).await
            {
                process.fail(&error);
                return Err(error);
            }
        }
        if let Err(error) = callbacks::replay(store.as_context_mut(), &process) {
            process.fail(&error);
            return Err(error);
        }
        store.data_mut().threaded_dynamic.generation = generation;
        initialization.finish_replay();
    }
    Ok(())
}

/// Published, immutable process allocations. Every TLS address belongs to a
/// reusable host slot, rather than a monotonically increasing pthread ID.
pub(super) struct Record {
    pub(super) path: String,
    pub(super) sha256: [u8; 32],
    pub(super) module: Module,
    pub(super) memory_base: u32,
    pub(super) table_base: u32,
    pub(super) tls: [u32; super::threads::MAX_THREADS],
    pub(super) dependencies: Vec<u32>,
    pub(super) global: std::sync::atomic::AtomicBool,
    layout: layout::Layout,
    function_slots: BTreeMap<String, u32>,
    store_cost: u64,
}

#[derive(Default)]
struct State {
    records: Vec<Arc<Record>>,
    paths: BTreeMap<String, u32>,
    generation: u64,
    attempts: usize,
    owner: Option<u32>,
    preparation_owner: Option<u32>,
    paused: BTreeSet<u32>,
    wake: BTreeSet<u32>,
    failed: bool,
    failure: Option<String>,
    main_symbols: Option<BTreeMap<String, u32>>,
    table_next: u32,
    callbacks: callbacks::ProcessState,
    startup_roots: Vec<u32>,
    startup_modules: usize,
}

#[derive(Clone, Default)]
pub(super) struct Process(Arc<Mutex<State>>);

/// Cancellation never publishes partial state. Once initialization can have
/// changed process memory, abandonment poisons further initialization.
pub(super) struct Initialization {
    process: Process,
    tid: u32,
    started: bool,
    finished: bool,
    admitted: usize,
}

impl Drop for Initialization {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let mut state = self.process.0.lock().expect("threaded dynamic registry");
        assert_eq!(state.owner, Some(self.tid));
        state.owner = None;
        state.failed |= self.started;
        let paused = std::mem::take(&mut state.paused);
        state.wake.extend(paused);
    }
}

impl Process {
    fn allocate_table(&self, size: u32, align: u32) -> Result<u32, Error> {
        let mut state = self.0.lock().expect("threaded dynamic registry");
        let mask = (1u32
            .checked_shl(align)
            .ok_or_else(|| Error::msg("invalid table alignment"))?)
        .saturating_sub(1);
        let base = state
            .table_next
            .checked_add(mask)
            .ok_or_else(|| Error::msg("threaded table overflow"))?
            & !mask;
        let end = base
            .checked_add(size)
            .ok_or_else(|| Error::msg("threaded table overflow"))?;
        if end as usize > MAX_TABLE_ELEMENTS {
            return Err(Error::msg("threaded dynamic table limit exceeded"));
        }
        state.table_next = end;
        Ok(base)
    }
    fn prepare_main(&self, symbols: BTreeMap<String, u32>, table_next: u32) -> Result<(), Error> {
        let mut state = self.0.lock().expect("threaded dynamic registry");
        if let Some(expected) = &state.main_symbols {
            if expected != &symbols {
                return Err(Error::msg("threaded main table reconstruction mismatch"));
            }
        } else {
            state.main_symbols = Some(symbols);
            state.table_next = table_next;
        }
        Ok(())
    }
    /// Admission counts failed attempts too, bounding repeated invalid loads.
    /// Caller prepays metadata/compiled images before retaining a record.
    #[cfg(test)]
    pub(super) fn begin(&self, tid: u32, modules: usize) -> Result<Initialization, Error> {
        self.admit(modules)?;
        self.begin_admitted(tid, modules)
    }

    // Replay executes guest relocation/TLS thunks, so it owns the same gate
    // as publication until all Store-local state is coherent.
    fn begin_replay(&self, tid: u32) -> Result<Initialization, Error> {
        let mut state = self.0.lock().expect("threaded dynamic registry");
        if state.failed || state.owner.is_some() {
            return Err(Error::msg(
                "threaded replay initialization gate unavailable",
            ));
        }
        state.owner = Some(tid);
        Ok(Initialization {
            process: self.clone(),
            tid,
            started: false,
            finished: false,
            admitted: 0,
        })
    }

    fn begin_admitted(&self, tid: u32, modules: usize) -> Result<Initialization, Error> {
        let mut state = self.0.lock().expect("threaded dynamic registry");
        if state.failed {
            return Err(Error::msg(state.failure.clone().unwrap_or_else(|| {
                "prior threaded dynamic initialization failed".to_owned()
            })));
        }
        if state.owner.is_some() {
            return Err(Error::msg(
                "nested threaded dynamic initialization is unsupported",
            ));
        }
        if modules == 0 || modules > MAX_MODULES {
            return Err(Error::msg("threaded dynamic module limit exceeded"));
        }
        state.owner = Some(tid);
        Ok(Initialization {
            process: self.clone(),
            tid,
            started: false,
            finished: false,
            admitted: modules,
        })
    }

    /// A different initialization owner blocks guest continuation across fuel
    /// yields. Failed initialization traps every subsequent checkpoint.
    pub(super) fn resume_allowed(&self, tid: u32) -> Result<bool, Error> {
        let mut state = self.0.lock().expect("threaded dynamic registry");
        if state.failed {
            return Err(Error::msg(state.failure.clone().unwrap_or_else(|| {
                "prior threaded dynamic initialization failed".to_owned()
            })));
        }
        let allowed = state.owner.is_none_or(|owner| owner == tid);
        if !allowed {
            state.paused.insert(tid);
        }
        Ok(allowed)
    }

    pub(super) fn existing(&self, path: &str, sha256: [u8; 32]) -> Result<Option<u32>, Error> {
        let state = self.0.lock().expect("threaded dynamic registry");
        let Some(handle) = state.paths.get(path).copied() else {
            return Ok(None);
        };
        if state.records[handle as usize - 1].sha256 != sha256 {
            return Err(Error::msg("threaded dynamic source identity changed"));
        }
        Ok(Some(handle))
    }

    pub(super) fn generation(&self) -> (u64, usize) {
        let state = self.0.lock().expect("threaded dynamic registry");
        (state.generation, state.records.len())
    }

    /// A bounded snapshot contains no Store-owned Wasmtime handles. The caller
    /// charges proportional replay work before cloning/installing these records.
    pub(super) fn snapshot(&self) -> (u64, Vec<Arc<Record>>) {
        let state = self.0.lock().expect("threaded dynamic registry");
        (state.generation, state.records.clone())
    }
}

impl Initialization {
    pub(super) fn started(&mut self) {
        self.started = true;
    }

    fn finish_replay(mut self) {
        let mut state = self.process.0.lock().expect("threaded dynamic registry");
        assert_eq!(state.owner, Some(self.tid));
        assert_eq!(self.admitted, 0);
        state.owner = None;
        let paused = std::mem::take(&mut state.paused);
        state.wake.extend(paused);
        self.finished = true;
    }

    /// Commit only after relocation/constructors and loading-Store TLS succeed.
    pub(super) fn publish(mut self, records: Vec<Arc<Record>>) -> Result<Vec<u32>, Error> {
        let mut state = self.process.0.lock().expect("threaded dynamic registry");
        assert_eq!(state.owner, Some(self.tid));
        let mut paths = std::collections::BTreeSet::new();
        if records.is_empty()
            || records.len() > self.admitted
            || records.len() > MAX_MODULES.saturating_sub(state.records.len())
            || records
                .iter()
                .any(|record| state.paths.contains_key(&record.path) || !paths.insert(&record.path))
        {
            return Err(Error::msg("invalid threaded dynamic publication"));
        }
        let generation = state
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::msg("threaded dynamic generation overflow"))?;
        let mut handles = Vec::with_capacity(records.len());
        for record in records {
            let handle = state.records.len() as u32 + 1;
            state.paths.insert(record.path.clone(), handle);
            state.records.push(record);
            handles.push(handle);
        }
        state.generation = generation;
        state.owner = None;
        let paused = std::mem::take(&mut state.paused);
        state.wake.extend(paused);
        self.finished = true;
        Ok(handles)
    }
}

/// Initialization is a nonblocking phase. Explicit host checks remain necessary
/// because Wasmtime temporarily removes the async call hook during its callback.
pub(super) fn can_block(host: &super::Host) -> Result<(), Error> {
    if host.threaded_dynamic.initializing {
        return Err(Error::msg(
            "blocking during threaded dynamic initialization is unsupported",
        ));
    }
    Ok(())
}

struct Preparation {
    process: Process,
    tid: u32,
}

impl Drop for Preparation {
    fn drop(&mut self) {
        let mut state = self.process.0.lock().expect("threaded dynamic registry");
        assert_eq!(state.preparation_owner, Some(self.tid));
        state.preparation_owner = None;
        let paused = std::mem::take(&mut state.paused);
        state.wake.extend(paused);
    }
}

impl Process {
    fn admit(&self, modules: usize) -> Result<(), Error> {
        let mut state = self.0.lock().expect("threaded dynamic registry");
        if state.failed || modules == 0 || modules > MAX_MODULES.saturating_sub(state.attempts) {
            return Err(Error::msg("threaded dynamic admission exhausted or failed"));
        }
        state.attempts += modules;
        Ok(())
    }

    fn try_prepare(&self, tid: u32) -> Result<bool, Error> {
        let mut state = self.0.lock().expect("threaded dynamic registry");
        if state.failed {
            return Err(Error::msg(state.failure.clone().unwrap_or_else(|| {
                "prior threaded dynamic initialization failed".to_owned()
            })));
        }
        if let Some(owner) = state.preparation_owner {
            if owner == tid {
                return Err(Error::msg("nested threaded loading is unsupported"));
            }
            state.paused.insert(tid);
            return Ok(false);
        }
        state.preparation_owner = Some(tid);
        Ok(true)
    }

    pub(super) fn paused(&self, tid: u32) -> bool {
        self.0
            .lock()
            .expect("threaded dynamic registry")
            .paused
            .contains(&tid)
    }

    pub(super) fn take_ready(&self) -> BTreeSet<u32> {
        let mut state = self.0.lock().expect("threaded dynamic registry");
        std::mem::take(&mut state.wake)
    }

    fn fail(&self, error: &Error) {
        let mut state = self.0.lock().expect("threaded dynamic registry");
        state.failed = true;
        if state.failure.is_none() {
            state.failure = Some(error.to_string().chars().take(1024).collect());
        }
        let paused = std::mem::take(&mut state.paused);
        state.wake.extend(paused);
    }
}

impl Process {
    fn known(&self, path: &str) -> Option<(u32, [u8; 32])> {
        let state = self.0.lock().expect("threaded dynamic registry");
        let handle = *state.paths.get(path)?;
        Some((handle, state.records[handle as usize - 1].sha256))
    }

    fn promote(&self, handle: u32) -> Result<(), Error> {
        let state = self.0.lock().expect("threaded dynamic registry");
        let mut pending = vec![handle];
        let mut visited = BTreeSet::new();
        while let Some(handle) = pending.pop() {
            if !visited.insert(handle) {
                continue;
            }
            let record = state
                .records
                .get(handle as usize - 1)
                .ok_or_else(|| Error::msg("invalid threaded dynamic handle"))?;
            record
                .global
                .store(true, std::sync::atomic::Ordering::Relaxed);
            pending.extend(record.dependencies.iter().copied());
        }
        Ok(())
    }
}

fn strings(mut store: StoreContextMut<'_, super::Host>) -> Result<(), Error> {
    if !store.data().threaded_dynamic.strings_reserved {
        reserve_store(&mut store, 32 * 1024)?;
        store.data_mut().threaded_dynamic.strings_reserved = true;
    }
    Ok(())
}

fn guest_string(
    caller: &mut wasmtime::Caller<'_, super::Host>,
    pointer: u32,
    length: u32,
) -> Result<String, Error> {
    use wasmtime::AsContextMut;
    if length == 0 || length > 4096 {
        return Err(Error::msg("invalid threaded loader string length"));
    }
    strings(caller.as_context_mut())?;
    if !caller
        .data()
        .machine
        .get()
        .resources
        .charge_cpu(u64::from(length))
    {
        return Err(super::exhausted());
    }
    let mut bytes = vec![0; length as usize];
    super::memory(caller)
        .ok_or_else(|| Error::msg("threaded memory unavailable"))?
        .read(caller, pointer as usize, &mut bytes)?;
    if bytes.contains(&0) {
        return Err(Error::msg("threaded loader string contains NUL"));
    }
    Ok(String::from_utf8(bytes)?)
}

/// The namespace is admitted only with the exact shared-memory v3 main marker.
/// Side imports reuse this registration and the same process publication owner.
pub(super) fn register(linker: &mut wasmtime::Linker<super::Host>) {
    use wasmtime::{AsContextMut, Caller};
    linker
        .func_wrap_async(
            NAMESPACE,
            "open",
            |mut caller: Caller<'_, super::Host>, (pointer, length, flags): (u32, u32, u32)| {
                Box::new(async move {
                    strings(caller.as_context_mut())?;
                    if pointer == 0 && length == 0 {
                        if flags & !(1 | 2 | 8 | 256 | 4096) == 0
                            && flags & 3 != 0
                            && flags & 3 != 3
                            && flags & (8 | 256) != (8 | 256)
                        {
                            return Ok(u32::MAX);
                        }
                        caller.data_mut().threaded_dynamic.error =
                            Some("unsupported dlopen flags".to_owned());
                        return Ok(0);
                    }
                    let path = guest_string(&mut caller, pointer, length)?;
                    if caller.data().threaded_dynamic.loading
                        || caller.data().threaded_dynamic.initializing
                    {
                        return Err(Error::msg("nested threaded loading is unsupported"));
                    }
                    caller.data_mut().threaded_dynamic.loading = true;
                    let result = loading::load(&mut caller.as_context_mut(), path, flags).await;
                    caller.data_mut().threaded_dynamic.loading = false;
                    match result {
                        Ok(handle) => Ok(handle),
                        Err(error) => {
                            let process = caller
                                .data()
                                .thread
                                .as_ref()
                                .expect("thread host")
                                .dynamic_process();
                            let failed =
                                process.0.lock().expect("threaded dynamic registry").failed;
                            // A dropped initialization guard poisons the process. Retain
                            // its cause before the next checkpoint observes that poison.
                            if failed {
                                process.fail(&error);
                            }
                            if failed
                                || error.is::<super::GuestExit>()
                                || caller
                                    .data()
                                    .machine
                                    .get()
                                    .resources
                                    .stop_reason()
                                    .is_some()
                                || caller.get_fuel()? == 0
                            {
                                return Err(error);
                            }
                            caller.data_mut().threaded_dynamic.error =
                                Some(format!("{error:#}").chars().take(2047).collect());
                            Ok(0)
                        }
                    }
                })
            },
        )
        .expect("threaded loader open signature");
    linker
        .func_wrap(
            NAMESPACE,
            "symbol",
            |mut caller: Caller<'_, super::Host>,
             handle: u32,
             pointer: u32,
             length: u32|
             -> Result<u32, Error> {
                let name = guest_string(&mut caller, pointer, length)?;
                match replay::symbol(caller.as_context_mut(), handle, &name) {
                    Ok(address) => Ok(address),
                    Err(error) => {
                        if caller
                            .data()
                            .machine
                            .get()
                            .resources
                            .stop_reason()
                            .is_some()
                        {
                            return Err(error);
                        }
                        caller.data_mut().threaded_dynamic.error =
                            Some(format!("{error:#}").chars().take(2047).collect());
                        Ok(0)
                    }
                }
            },
        )
        .expect("threaded loader symbol signature");
    linker
        .func_wrap(
            NAMESPACE,
            "error",
            |mut caller: Caller<'_, super::Host>,
             pointer: u32,
             capacity: u32|
             -> Result<u32, Error> {
                if capacity == 0 || capacity > 4096 {
                    return Err(Error::msg("invalid threaded loader error buffer"));
                }
                let Some(error) = caller.data_mut().threaded_dynamic.error.take() else {
                    return Ok(0);
                };
                let size = error.len().min(capacity as usize - 1);
                if !caller
                    .data()
                    .machine
                    .get()
                    .resources
                    .charge_cpu(size as u64 + 1)
                {
                    return Err(super::exhausted());
                }
                let memory = super::memory(&mut caller)
                    .ok_or_else(|| Error::msg("threaded memory unavailable"))?;
                memory.write(&mut caller, pointer as usize, &error.as_bytes()[..size])?;
                memory.write(&mut caller, pointer as usize + size, &[0])?;
                Ok(size as u32)
            },
        )
        .expect("threaded loader error signature");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialization_blocks_other_threads_and_cancellation_poison_is_explicit() {
        let process = Process::default();
        let mut initialization = process.begin(7, 1).unwrap();
        assert!(process.resume_allowed(7).unwrap());
        assert!(!process.resume_allowed(9).unwrap());
        assert!(process.begin(7, 1).is_err());
        initialization.started();
        drop(initialization);
        assert!(process.resume_allowed(7).is_err());
        assert!(process.begin(9, 1).is_err());
        assert_eq!(process.snapshot().0, 0);
        process.fail(&Error::msg("missing native initializer"));
        assert_eq!(
            process.resume_allowed(7).unwrap_err().to_string(),
            "missing native initializer"
        );
        process.fail(&Error::msg("later checkpoint failure"));
        assert_eq!(
            process.try_prepare(9).unwrap_err().to_string(),
            "missing native initializer"
        );
    }

    #[test]
    fn replay_holds_gate_until_all_store_local_bindings_are_installed() {
        let process = Process::default();
        let mut replay = process.begin_replay(1).unwrap();
        replay.started();
        assert!(!process.resume_allowed(2).unwrap());
        assert!(process.begin_admitted(2, 1).is_err());
        replay.finish_replay();
        assert!(process.resume_allowed(2).unwrap());
        assert!(process.take_ready().contains(&2));
        assert_eq!(process.generation(), (0, 0));
    }

    #[test]
    fn preflight_cancellation_rolls_back_gate_but_attempts_stay_bounded() {
        let process = Process::default();
        for _ in 0..MAX_MODULES {
            drop(process.begin(1, 1).unwrap());
            assert!(process.resume_allowed(2).unwrap());
        }
        assert!(process.begin(1, 1).is_err());
        assert_eq!(process.snapshot().0, 0);
    }
}
