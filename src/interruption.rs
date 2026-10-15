//! Caller-owned cancellation requests for one host-driven action.
//!
//! The request is independent of the machine lock and never resets resource usage. A token
//! binds once, so a delayed request cannot spill into a later action using another token.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// A single-use, thread-safe interruption request. Clones address the same action.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<State>);

#[derive(Debug, Default)]
struct State {
    requested: AtomicBool,
    bound: AtomicBool,
    finished: AtomicBool,
    wake: Condvar,
    waiting: Mutex<()>,
}

/// A host action's binding is deliberately absent from a machine snapshot.
#[derive(Debug, Default)]
pub(crate) struct ActionCancellation {
    pub(crate) token: Option<CancellationToken>,
    pub(crate) interrupted: Cell<bool>,
    pub(crate) generation: Option<u64>,
}

impl Clone for ActionCancellation {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl CancellationToken {
    /// Create a request that may be passed to one cancellable run.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request interruption without acquiring the environment's execution lock.
    pub fn cancel(&self) {
        let _waiting = self.0.waiting.lock().expect("cancellation wait lock");
        if !self.0.finished.load(Ordering::Acquire) {
            self.0.requested.store(true, Ordering::Release);
        }
        self.0.wake.notify_all();
    }

    /// Whether the caller has requested cancellation, including before execution starts.
    pub fn is_cancelled(&self) -> bool {
        self.0.requested.load(Ordering::Acquire)
    }

    pub(crate) fn bind(&self) -> Result<(), String> {
        self.0
            .bound
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| "cancellation token has already been used".to_string())
    }

    pub(crate) fn finish(&self) {
        let _waiting = self.0.waiting.lock().expect("cancellation wait lock");
        self.0.finished.store(true, Ordering::Release);
    }

    /// A real-time driver may sleep, but a host cancellation must wake it immediately.
    pub(crate) fn wait(&self, duration: Duration) {
        let waiting = self.0.waiting.lock().expect("cancellation wait lock");
        drop(
            self.0
                .wake
                .wait_timeout_while(waiting, duration, |_| !self.is_cancelled())
                .expect("cancellation wait lock"),
        );
    }
}
