//! Bound linear memory and the exception collector's heap within one guest reservation.
//!
//! Wasmtime routes both heaps through the memory limiter. A per-memory limit alone
//! would permit each heap to consume the whole reservation. Growth callbacks are
//! sequential; a failed approved growth is rolled back before another request.

use wasmtime::{Error, ResourceLimiter, StoreLimits};

pub(super) struct GuestLimits {
    store: StoreLimits,
    memory_cap: usize,
    memory_used: usize,
    pending_growth: usize,
}

impl GuestLimits {
    pub(super) fn new(store: StoreLimits, memory_cap: usize) -> Self {
        Self {
            store,
            memory_cap,
            memory_used: 0,
            pending_growth: 0,
        }
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
        self.memory_used = total;
        self.pending_growth = growth;
        Ok(true)
    }

    fn memory_grow_failed(&mut self, error: Error) -> Result<(), Error> {
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
        self.store.table_growing(current, desired, maximum)
    }

    fn table_grow_failed(&mut self, error: Error) -> Result<(), Error> {
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
}
