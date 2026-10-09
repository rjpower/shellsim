//! Bound linear memory and the exception collector's heap within one guest reservation.
//!
//! Wasmtime routes both heaps through the memory limiter. A per-memory limit alone
//! would permit each heap to consume the whole reservation. Growth callbacks are
//! sequential; a failed approved growth is rolled back before another request.

use super::MachineAccess;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use wasmtime::{Error, ResourceLimiter, StoreLimits};

struct Incremental {
    machine: MachineAccess,
    retained: Arc<AtomicU64>,
    charge_tables: bool,
}

pub(super) struct GuestLimits {
    store: StoreLimits,
    memory_cap: usize,
    memory_used: usize,
    pending_growth: usize,
    pending_table_growth: u64,
    incremental: Option<Incremental>,
}

impl GuestLimits {
    pub(super) fn new(store: StoreLimits, memory_cap: usize) -> Self {
        Self {
            store,
            memory_cap,
            memory_used: 0,
            pending_growth: 0,
            pending_table_growth: 0,
            incremental: None,
        }
    }

    /// V2 charges actual linear and exception heap growth to the process's shared owner.
    pub(super) fn incremental(
        store: StoreLimits,
        memory_cap: usize,
        machine: MachineAccess,
        retained: Arc<AtomicU64>,
    ) -> Self {
        let mut limits = Self::new(store, memory_cap);
        limits.incremental = Some(Incremental {
            machine,
            retained,
            charge_tables: true,
        });
        limits
    }
    /// Shared linear memory and table capacity are prepaid by the thread group.
    /// Only the Store-local exception collector can request private heap growth.
    pub(super) fn threaded_exception_heap(
        store: StoreLimits,
        memory_cap: usize,
        machine: MachineAccess,
        retained: Arc<AtomicU64>,
    ) -> Self {
        let mut limits = Self::incremental(store, memory_cap, machine, retained);
        limits
            .incremental
            .as_mut()
            .expect("incremental heap")
            .charge_tables = false;
        limits
    }

    /// The worker Store has already been consumed, so its exception heap is gone.
    /// Cancelled active Stores leave this charge with the shared process owner.
    pub(super) fn release_thread_heap(&mut self) {
        let Some(incremental) = &self.incremental else {
            return;
        };
        if incremental.charge_tables {
            return;
        }
        incremental
            .machine
            .get()
            .resources
            .release_memory(self.memory_used as u64);
        incremental
            .retained
            .fetch_sub(self.memory_used as u64, Ordering::Relaxed);
        self.memory_used = 0;
        self.pending_growth = 0;
    }
}

impl ResourceLimiter for GuestLimits {
    fn memory_growing(
        &mut self,
        current: usize,
        desired: usize,
        maximum: Option<usize>,
    ) -> Result<bool, Error> {
        self.pending_growth = 0;
        if !self.store.memory_growing(current, desired, maximum)? {
            return Ok(false);
        }
        let growth = desired.saturating_sub(current);
        let Some(total) = self.memory_used.checked_add(growth) else {
            return Ok(false);
        };
        if total > self.memory_cap {
            return Ok(false);
        }
        if let Some(incremental) = &self.incremental {
            if !incremental
                .machine
                .get()
                .resources
                .reserve_memory(growth as u64)
            {
                return Ok(false);
            }
            if incremental
                .retained
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |owned| {
                    owned.checked_add(growth as u64)
                })
                .is_err()
            {
                incremental
                    .machine
                    .get()
                    .resources
                    .release_memory(growth as u64);
                return Err(Error::msg("Wasm retained heap accounting overflow"));
            }
        }
        self.memory_used = total;
        self.pending_growth = growth;
        Ok(true)
    }

    fn memory_grow_failed(&mut self, error: Error) -> Result<(), Error> {
        if let Some(incremental) = &self.incremental {
            incremental
                .machine
                .get()
                .resources
                .release_memory(self.pending_growth as u64);
            incremental
                .retained
                .fetch_sub(self.pending_growth as u64, Ordering::Relaxed);
        }
        self.memory_used = self.memory_used.saturating_sub(self.pending_growth);
        self.pending_growth = 0;
        self.store.memory_grow_failed(error)
    }

    fn table_growing(
        &mut self,
        current: usize,
        desired: usize,
        maximum: Option<usize>,
    ) -> Result<bool, Error> {
        self.pending_table_growth = 0;
        if !self.store.table_growing(current, desired, maximum)? {
            return Ok(false);
        }
        if let Some(incremental) = &self.incremental {
            if !incremental.charge_tables {
                return Ok(true);
            }
            let growth = desired.saturating_sub(current) as u64;
            let Some(bytes) = growth.checked_mul(16) else {
                return Ok(false);
            };
            let mut machine = incremental.machine.get();
            if !machine.resources.charge_cpu(growth) || !machine.resources.reserve_memory(bytes) {
                return Ok(false);
            }
            if incremental
                .retained
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |owned| {
                    owned.checked_add(bytes)
                })
                .is_err()
            {
                machine.resources.release_memory(bytes);
                return Err(Error::msg("Wasm retained table accounting overflow"));
            }
            self.pending_table_growth = bytes;
        }
        Ok(true)
    }

    fn table_grow_failed(&mut self, error: Error) -> Result<(), Error> {
        if let Some(incremental) = &self.incremental {
            incremental
                .machine
                .get()
                .resources
                .release_memory(self.pending_table_growth);
            incremental
                .retained
                .fetch_sub(self.pending_table_growth, Ordering::Relaxed);
        }
        self.pending_table_growth = 0;
        self.store.table_grow_failed(error)
    }
    fn instances(&self) -> usize {
        self.store.instances()
    }
    fn tables(&self) -> usize {
        self.store.tables()
    }
    fn memories(&self) -> usize {
        self.store.memories()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasmtime::StoreLimitsBuilder;

    #[test]
    fn incremental_failed_growth_returns_environment_and_owner_charges() {
        let machine = MachineAccess::default();
        let retained = Arc::new(AtomicU64::new(0));
        let mut environment = crate::interp::Interp::with_limits(crate::resources::Limits {
            memory: 200,
            ..crate::resources::Limits::default()
        });
        let mut limits = GuestLimits::incremental(
            StoreLimitsBuilder::new().memory_size(200).build(),
            200,
            machine.clone(),
            retained.clone(),
        );
        machine.enter(&mut environment, || {
            assert!(limits.memory_growing(0, 120, None).unwrap());
            assert!(limits.memory_growing(0, 60, None).unwrap());
            limits
                .memory_grow_failed(Error::msg("allocation failed"))
                .unwrap();
            assert_eq!(retained.load(Ordering::Relaxed), 120);
            assert_eq!(machine.get().resources.memory_mark(), 120);
            assert!(limits.memory_growing(0, 80, None).unwrap());
            assert_eq!(retained.load(Ordering::Relaxed), 200);
            assert_eq!(machine.get().resources.memory_mark(), 200);
        });
    }

    #[test]
    fn threaded_heap_growth_refunds_failure_and_finished_worker_without_recharging_table() {
        let machine = MachineAccess::default();
        let retained = Arc::new(AtomicU64::new(0));
        let mut environment = crate::interp::Interp::with_limits(crate::resources::Limits {
            memory: 200,
            ..crate::resources::Limits::default()
        });
        let mut limits = GuestLimits::threaded_exception_heap(
            StoreLimitsBuilder::new()
                .memory_size(200)
                .table_elements(16)
                .build(),
            200,
            machine.clone(),
            retained.clone(),
        );
        machine.enter(&mut environment, || {
            assert!(limits.table_growing(0, 4, Some(16)).unwrap());
            assert_eq!(retained.load(Ordering::Relaxed), 0);
            assert!(limits.memory_growing(0, 120, None).unwrap());
            assert!(limits.memory_growing(0, 60, None).unwrap());
            limits
                .memory_grow_failed(Error::msg("allocation failed"))
                .unwrap();
            assert_eq!(retained.load(Ordering::Relaxed), 120);
            assert!(!limits.memory_growing(120, 210, None).unwrap());
            limits.release_thread_heap();
            assert_eq!(retained.load(Ordering::Relaxed), 0);
            assert_eq!(machine.get().resources.memory_mark(), 0);
            assert!(limits.memory_growing(0, 200, None).unwrap());
            limits.release_thread_heap();
            assert_eq!(machine.get().resources.memory_mark(), 0);
        });
    }

    #[test]
    fn two_heaps_share_one_memory_bound() {
        let mut limits = GuestLimits::new(StoreLimitsBuilder::new().memory_size(200).build(), 200);
        assert!(limits.memory_growing(0, 120, None).unwrap());
        assert!(limits.memory_growing(0, 60, None).unwrap());
        assert!(!limits.memory_growing(60, 90, None).unwrap());
        assert!(limits.memory_growing(60, 80, None).unwrap());
        assert_eq!(limits.memory_used, 200);
    }

    #[test]
    fn failed_growth_is_rolled_back_once() {
        let mut limits = GuestLimits::new(StoreLimitsBuilder::new().memory_size(200).build(), 200);
        assert!(limits.memory_growing(0, 120, None).unwrap());
        assert!(limits.memory_growing(0, 60, None).unwrap());
        limits
            .memory_grow_failed(Error::msg("allocation failed"))
            .unwrap();
        assert_eq!(limits.memory_used, 120);
        assert!(limits.memory_growing(0, 80, None).unwrap());
        assert_eq!(limits.memory_used, 200);
    }

    #[test]
    fn incremental_failed_table_growth_returns_the_environment_charge_once() {
        let machine = MachineAccess::default();
        let retained = Arc::new(AtomicU64::new(0));
        let mut environment = crate::interp::Interp::with_limits(crate::resources::Limits {
            memory: 250,
            ..crate::resources::Limits::default()
        });
        let mut limits = GuestLimits::incremental(
            StoreLimitsBuilder::new()
                .memory_size(200)
                .table_elements(100)
                .build(),
            200,
            machine.clone(),
            retained.clone(),
        );
        machine.enter(&mut environment, || {
            assert!(limits.table_growing(0, 5, None).unwrap());
            assert!(limits.memory_growing(0, 100, None).unwrap());
            limits
                .table_grow_failed(Error::msg("allocation failed"))
                .unwrap();
            limits
                .table_grow_failed(Error::msg("duplicate notification"))
                .unwrap();
            assert_eq!(retained.load(Ordering::Relaxed), 100);
            assert_eq!(machine.get().resources.memory_mark(), 100);
            assert!(limits.table_growing(0, 9, None).unwrap());
            assert_eq!(retained.load(Ordering::Relaxed), 244);
            assert_eq!(machine.get().resources.memory_mark(), 244);
        });
    }
    #[test]
    fn actual_wasm_exception_heap_is_metered_and_released_after_store_drop() {
        let module = wasmtime::Module::new(
            super::super::command_engine(),
            wat::parse_str(
                r#"(module (tag (param i32)) (func (export "raise") i32.const 7 throw 0))"#,
            )
            .unwrap(),
        )
        .unwrap();
        for (cap, memory_budget) in [
            (0, 16 * 1024 * 1024),
            (8 * 1024 * 1024, 16 * 1024 * 1024),
            (8 * 1024 * 1024, 32 * 1024),
        ] {
            let machine = MachineAccess::default();
            let retained = Arc::new(AtomicU64::new(0));
            let mut environment = crate::interp::Interp::with_limits(crate::resources::Limits {
                memory: memory_budget,
                ..crate::resources::Limits::default()
            });
            machine.enter(&mut environment, || {
                let limits = GuestLimits::threaded_exception_heap(
                    StoreLimitsBuilder::new().memory_size(cap).build(),
                    cap,
                    machine.clone(),
                    retained.clone(),
                );
                let mut store = wasmtime::Store::new(super::super::command_engine(), limits);
                store.limiter(|limits| limits);
                store.set_fuel(100_000).unwrap();
                let result = {
                    let mut call = std::pin::pin!(async {
                        let instance =
                            wasmtime::Instance::new_async(&mut store, &module, &[]).await?;
                        instance
                            .get_typed_func::<(), ()>(&mut store, "raise")?
                            .call_async(&mut store, ())
                            .await
                    });
                    let result = {
                        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
                        let mut completed = None;
                        for _ in 0..16 {
                            if let std::task::Poll::Ready(result) =
                                std::future::Future::poll(call.as_mut(), &mut context)
                            {
                                completed = Some(result);
                                break;
                            }
                        }
                        completed.expect("bounded throw did not complete")
                    };
                    result
                };
                eprintln!("EH heap cap {cap}: {:#}", result.as_ref().unwrap_err());
                assert!(result.is_err());
                if cap == 0 || memory_budget == 32 * 1024 {
                    assert_eq!(retained.load(Ordering::Relaxed), 0);
                } else {
                    assert!(retained.load(Ordering::Relaxed) > 0);
                }
                // No VM objects can use the private heap after Store consumption.
                let mut limits = store.into_data();
                limits.release_thread_heap();
                assert_eq!(retained.load(Ordering::Relaxed), 0);
                assert_eq!(machine.get().resources.memory_mark(), 0);
            });
        }
    }
}
