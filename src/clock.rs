//! Virtual clock.
//!
//! There is NO real wall-clock blocking anywhere in the simulator. `sleep`, timeouts and
//! background-job scheduling all advance this logical clock instead. This makes episodes
//! fully deterministic and lets a rollout that "sleeps 30s" complete in microseconds.

#[derive(Clone, Debug)]
pub struct Clock {
    /// logical time in milliseconds since the (virtual) epoch
    now_ms: u64,
    /// total simulated time slept — useful telemetry for RL reward shaping
    pub slept_ms: u64,
    /// epoch offset so `date` can render a plausible absolute time
    epoch_base_ms: u64,
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock {
    pub fn new() -> Self {
        // default virtual epoch: 2025-01-01T00:00:00Z, deterministic across runs
        Clock {
            now_ms: 0,
            slept_ms: 0,
            epoch_base_ms: 1_735_689_600_000,
        }
    }

    pub fn now_ms(&self) -> u64 {
        self.now_ms
    }

    /// Absolute unix time (ms) as seen by simulated programs.
    pub fn unix_ms(&self) -> u64 {
        self.epoch_base_ms + self.now_ms
    }

    pub fn unix_secs(&self) -> u64 {
        self.unix_ms() / 1000
    }

    /// Advance the logical clock. This is the ONLY effect of `sleep` — no real blocking.
    pub fn sleep_ms(&mut self, ms: u64) {
        self.now_ms += ms;
        self.slept_ms += ms;
    }

    /// Advance by a number of abstract "ticks" (1 tick = 1ms by default). Lets a driver
    /// step background work without committing to a unit.
    pub fn tick(&mut self, ticks: u64) {
        self.now_ms += ticks;
    }
}
