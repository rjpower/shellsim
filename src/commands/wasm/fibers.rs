//! Account for nested Wasmtime async fiber stacks owned by one Store.
//!
//! The outer Store stack is prepaid at launch. Each guest-to-host-to-guest call
//! needs another 2 MiB stack, and Wasmtime retains its last completed stack for
//! reuse. We retain a bounded high-water reservation until Store teardown so a
//! cancelled future never needs to borrow the virtual machine from `Drop`.

use super::{Host, ASYNC_STACK_BYTES};
use std::sync::{atomic::Ordering, Arc, Mutex};
use wasmtime::{AsContextMut, Error};

const MAX_NESTED: usize = 8;

#[derive(Default)]
struct State {
    active: usize,
    reserved: usize,
}

#[derive(Default)]
pub(super) struct Budget(Arc<Mutex<State>>);

pub(super) struct Guard(Arc<Mutex<State>>);

impl Drop for Guard {
    fn drop(&mut self) {
        let mut state = self.0.lock().expect("fiber budget");
        state.active -= 1;
    }
}

/// Prepay one more concurrent fiber before a host import enters guest code.
pub(super) fn begin(mut store: impl AsContextMut<Data = Host>) -> Result<Guard, Error> {
    let store = store.as_context_mut();
    let budget = store.data().fibers.0.clone();
    let mut state = budget.lock().expect("fiber budget");
    if state.active >= MAX_NESTED {
        return Err(Error::msg("nested Wasm fiber limit exceeded"));
    }
    if state.active == state.reserved {
        let bytes = ASYNC_STACK_BYTES as u64;
        if !store.data().machine.get().resources.reserve_memory(bytes) {
            return Err(super::exhausted());
        }
        store.data().retained.fetch_add(bytes, Ordering::Relaxed);
        state.reserved += 1;
    }
    state.active += 1;
    drop(state);
    Ok(Guard(budget))
}

/// Return one finished worker Store's nested stack charges to the process.
///
/// The caller invokes this only after every future in that Store has completed.
/// The outer stack remains covered by the thread-slot prepayment.
pub(super) fn release_store(host: &mut Host) {
    let mut state = host.fibers.0.lock().expect("fiber budget");
    assert_eq!(state.active, 0, "finished Store has no live nested fibers");
    let bytes = (state.reserved as u64).saturating_mul(ASYNC_STACK_BYTES as u64);
    state.reserved = 0;
    if bytes != 0 {
        host.machine.get().resources.release_memory(bytes);
        host.retained.fetch_sub(bytes, Ordering::Relaxed);
    }
}
