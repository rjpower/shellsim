//! Publish stable callback addresses with a separate local Func in every Store.
//!
//! Slots and retired metadata remain owned by the process until teardown. The
//! shared active flag also guards an already obtained Func after another thread
//! releases its slot; clearing a table cell alone cannot revoke that handle.

use super::{checkpoint, loading, reserve_store, Process};
use crate::commands::wasm::{ffi, Host};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use wasmtime::{AsContextMut, Caller, Error, Ref, StoreContextMut, Table};

struct Entry {
    index: u32,
    active: Arc<AtomicBool>,
    defined: bool,
}

#[derive(Clone)]
enum Event {
    Reserve(u32),
    Define(u32, ffi::CallbackSpec, Arc<AtomicBool>),
    Release(u32),
}

#[derive(Default)]
pub(super) struct ProcessState {
    entries: Vec<Entry>,
    events: Vec<Event>,
}

#[derive(Default)]
pub(super) struct StoreState {
    cursor: usize,
    charged: Vec<u32>,
}

fn table(mut store: StoreContextMut<'_, Host>) -> Result<Table, Error> {
    let main = store
        .data()
        .threaded_dynamic
        .main
        .ok_or_else(|| Error::msg("threaded FFI runtime unavailable"))?;
    main.get_table(&mut store, "__indirect_function_table")
        .ok_or_else(|| Error::msg("threaded FFI table unavailable"))
}

fn install(mut store: StoreContextMut<'_, Host>, event: &Event) -> Result<(), Error> {
    let table = table(store.as_context_mut())?;
    match event {
        Event::Reserve(index) => {
            if !store
                .data()
                .threaded_dynamic
                .callbacks
                .charged
                .contains(index)
            {
                reserve_store(&mut store, ffi::CLOSURE_METADATA_BYTES)?;
                store
                    .data_mut()
                    .threaded_dynamic
                    .callbacks
                    .charged
                    .push(*index);
            }
            let end = u64::from(*index) + 1;
            if table.size(&store) < end {
                let growth = end - table.size(&store);
                table.grow(&mut store, growth, Ref::Func(None))?;
            }
            table.set(&mut store, u64::from(*index), Ref::Func(None))?;
        }
        Event::Define(index, spec, active) => {
            let function = ffi::make_callback(store.as_context_mut(), spec, active.clone());
            table.set(&mut store, u64::from(*index), Ref::Func(Some(function)))?;
        }
        Event::Release(index) => table.set(&mut store, u64::from(*index), Ref::Func(None))?,
    }
    Ok(())
}

pub(super) fn replay(mut store: StoreContextMut<'_, Host>, process: &Process) -> Result<(), Error> {
    let cursor = store.data().threaded_dynamic.callbacks.cursor;
    let count = process
        .0
        .lock()
        .expect("threaded dynamic registry")
        .callbacks
        .events
        .len()
        - cursor;
    if !store
        .data()
        .machine
        .get()
        .resources
        .charge_cpu(count as u64 * 32 + 8)
    {
        return Err(super::super::exhausted());
    }
    let events = {
        let state = process.0.lock().expect("threaded dynamic registry");
        state.callbacks.events[cursor..].to_vec()
    };
    for event in &events {
        install(store.as_context_mut(), event)?;
        store.data_mut().threaded_dynamic.callbacks.cursor += 1;
    }
    Ok(())
}

/// Reject initialization before waiting on its gate or changing any table cell.
async fn prepare(caller: &mut Caller<'_, Host>) -> Result<Process, Error> {
    if caller.data().threaded_dynamic.initializing || !caller.data().threaded_dynamic.ready {
        return Err(Error::msg(
            "FFI during threaded loader initialization is unsupported",
        ));
    }
    if !caller.data().machine.get().resources.charge_cpu(16) {
        return Err(super::super::exhausted());
    }
    checkpoint(caller.as_context_mut()).await?;
    Ok(caller
        .data()
        .thread
        .as_ref()
        .expect("thread host")
        .dynamic_process())
}

// After prepare awaits, publication performs no await or guest call. The
// cooperative scheduler therefore cannot advance another Store between reading
// the next generation and acquiring this gate. Keep table installation and
// publication in that same non-yielding interval.
fn publish(caller: &mut Caller<'_, Host>, process: &Process, event: Event, generation: u64) {
    let mut state = process.0.lock().expect("threaded dynamic registry");
    state.callbacks.events.push(event);
    state.generation = generation;
    caller.data_mut().threaded_dynamic.callbacks.cursor = state.callbacks.events.len();
    caller.data_mut().threaded_dynamic.generation = generation;
}

/// Allocate an address without a callable value. Failed mutations poison the
/// gate; consumed addresses and metadata are retained rather than reused.
pub(in crate::commands::wasm) async fn reserve(
    caller: &mut Caller<'_, Host>,
) -> Result<Option<u32>, Error> {
    let process = prepare(caller).await?;
    let generation = {
        let state = process.0.lock().expect("threaded dynamic registry");
        if state.callbacks.entries.len() == ffi::MAX_CLOSURES {
            return Ok(None);
        }
        state
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::msg("threaded publication generation overflow"))?
    };
    loading::reserve_process(caller, ffi::CLOSURE_METADATA_BYTES)?;
    // This charge and the tombstone survive any later failure, as does the slot.
    let mut gate =
        process.begin_replay(caller.data().thread.as_ref().expect("thread host").id())?;
    let index = process.allocate_table(1, 0)?;
    let event = Event::Reserve(index);
    gate.started();
    {
        let mut state = process.0.lock().expect("threaded dynamic registry");
        state.callbacks.entries.push(Entry {
            index,
            active: Arc::new(AtomicBool::new(true)),
            defined: false,
        });
    }
    install(caller.as_context_mut(), &event)?;
    publish(caller, &process, event, generation);
    gate.finish_replay();
    Ok(Some(index))
}

/// Publish a prevalidated signature and dispatcher after local installation.
/// The FFI boundary checks scalar tags, dispatcher type and stack bounds first.
pub(in crate::commands::wasm) async fn define(
    caller: &mut Caller<'_, Host>,
    index: u32,
    spec: ffi::CallbackSpec,
) -> Result<bool, Error> {
    let process = prepare(caller).await?;
    let (active, generation) = {
        let state = process.0.lock().expect("threaded dynamic registry");
        let Some(entry) = state.callbacks.entries.iter().find(|entry| {
            entry.index == index && !entry.defined && entry.active.load(Ordering::Acquire)
        }) else {
            return Ok(false);
        };
        (
            entry.active.clone(),
            state
                .generation
                .checked_add(1)
                .ok_or_else(|| Error::msg("threaded publication generation overflow"))?,
        )
    };
    let mut gate =
        process.begin_replay(caller.data().thread.as_ref().expect("thread host").id())?;
    let event = Event::Define(index, spec, active);
    gate.started();
    install(caller.as_context_mut(), &event)?;
    process
        .0
        .lock()
        .expect("threaded dynamic registry")
        .callbacks
        .entries
        .iter_mut()
        .find(|entry| entry.index == index)
        .expect("reserved entry")
        .defined = true;
    publish(caller, &process, event, generation);
    gate.finish_replay();
    Ok(true)
}

/// Revoke the address process-wide before publishing the local table clear.
/// Old Funcs observe the same active flag even before their next replay.
pub(in crate::commands::wasm) async fn release(
    caller: &mut Caller<'_, Host>,
    index: u32,
) -> Result<bool, Error> {
    let process = prepare(caller).await?;
    let (active, generation) = {
        let state = process.0.lock().expect("threaded dynamic registry");
        let Some(entry) = state
            .callbacks
            .entries
            .iter()
            .find(|entry| entry.index == index && entry.active.load(Ordering::Acquire))
        else {
            return Ok(false);
        };
        (
            entry.active.clone(),
            state
                .generation
                .checked_add(1)
                .ok_or_else(|| Error::msg("threaded publication generation overflow"))?,
        )
    };
    let mut gate =
        process.begin_replay(caller.data().thread.as_ref().expect("thread host").id())?;
    gate.started();
    active.store(false, Ordering::Release);
    let event = Event::Release(index);
    install(caller.as_context_mut(), &event)?;
    publish(caller, &process, event, generation);
    gate.finish_replay();
    Ok(true)
}

/// Wait for prior publications before resolving a local dispatcher or raw target.
pub(in crate::commands::wasm) async fn synchronize(
    caller: &mut Caller<'_, Host>,
) -> Result<(), Error> {
    prepare(caller).await.map(|_| ())
}
