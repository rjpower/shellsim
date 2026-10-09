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
        limits.incremental = Some(Incremental { machine, retained });
        limits
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
            incremental
                .retained
                .fetch_add(growth as u64, Ordering::Relaxed);
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
            let growth = desired.saturating_sub(current) as u64;
            let Some(bytes) = growth.checked_mul(16) else {
                return Ok(false);
            };
            let mut machine = incremental.machine.get();
            if !machine.resources.charge_cpu(growth) || !machine.resources.reserve_memory(bytes) {
                return Ok(false);
            }
            incremental.retained.fetch_add(bytes, Ordering::Relaxed);
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
}
