//! Opt-in physical-time clock mode for interactive demos.
//!
//! By default a machine runs on purely virtual time: when every process waits on a timer, the
//! scheduler jumps the clock to the next deadline, so sleeps cost no host time and runs are
//! reproducible. A machine booted with [`ClockMode::RealTime`] instead keeps its virtual
//! monotonic clock at or after the physical time elapsed since boot. Guests read time and sleep
//! through the same virtual clock in both modes; only the embedder chooses the mode, so no
//! simulated program can request host time.
//!
//! Polling never blocks in either mode. In real-time mode the clock cannot jump, so an idle
//! machine reports that it is blocked, and [`Environment::host_wait_until`] tells a driver how
//! long to wait before polling again. Run-to-completion drivers and Python's synchronous nested
//! waits are the only code that sleeps the host thread. A snapshot of a real-time machine keeps
//! following physical time and is therefore not reproducible.

use std::time::{Duration, Instant};

use crate::interp::Environment;

/// How a machine's virtual clock relates to physical time, fixed when the machine boots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClockMode {
    /// Time advances only when every process is waiting on it. Deterministic.
    #[default]
    Virtual,
    /// Time follows the host's monotonic clock from boot. For interactive demos only.
    RealTime,
}

/// The host instant and virtual time at which a real-time machine booted.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostAnchor {
    host: Instant,
    virtual_ns: u64,
}

impl HostAnchor {
    pub(crate) fn now(virtual_ns: u64) -> Self {
        Self {
            host: Instant::now(),
            virtual_ns,
        }
    }

    fn elapsed_virtual_ns(&self) -> u64 {
        let elapsed = u64::try_from(self.host.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.virtual_ns.saturating_add(elapsed)
    }
}

impl Environment {
    /// The mode this machine booted with.
    pub fn clock_mode(&self) -> ClockMode {
        if self.real_time.is_some() {
            ClockMode::RealTime
        } else {
            ClockMode::Virtual
        }
    }

    /// Move a real-time machine's clock up to physical time. Events that become due are left
    /// ready for the caller to handle. Virtual machines are unchanged.
    pub(crate) fn sync_host_time(&mut self) -> Result<(), String> {
        let Some(anchor) = self.real_time else {
            return Ok(());
        };
        let target = anchor.elapsed_virtual_ns();
        if target > self.clock.monotonic_ns() {
            self.clock
                .advance_to(target)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    /// How long a driver should wait before polling an idle machine again: until the next timer
    /// in real-time mode. `None` in virtual mode, where polling with time advance moves the
    /// clock itself, and when no timer is pending.
    pub fn host_wait_until(&self) -> Option<Duration> {
        let anchor = self.real_time?;
        let deadline = self.clock.next_deadline_ns()?;
        Some(Duration::from_nanos(
            deadline.saturating_sub(anchor.elapsed_virtual_ns()),
        ))
    }

    /// Host time until a real-time machine's clock reaches `deadline`; `None` on virtual time.
    pub(crate) fn host_time_until(&self, deadline: u64) -> Option<Duration> {
        let anchor = self.real_time?;
        Some(Duration::from_nanos(
            deadline.saturating_sub(anchor.elapsed_virtual_ns()),
        ))
    }

    /// Wait for virtual time to reach `deadline` from code that cannot yield to the scheduler.
    /// Virtual machines jump the clock; real-time machines sleep the host thread first.
    pub(crate) fn wait_for_time(&mut self, deadline: u64) -> Result<(), String> {
        if let Some(anchor) = self.real_time {
            let now = anchor.elapsed_virtual_ns();
            if deadline > now {
                std::thread::sleep(Duration::from_nanos(deadline - now));
            }
            self.sync_host_time()?;
        }
        if deadline > self.clock.monotonic_ns() {
            self.clock
                .advance_to(deadline)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::{MachinePoll, ShellExecution};

    #[test]
    fn idle_real_time_poll_returns_blocked_with_the_host_wait() {
        let mut environment = Environment::with_limits_and_clock(
            crate::resources::Limits::default(),
            ClockMode::RealTime,
        );
        let node = crate::shell::parse("sleep 10").unwrap();
        let mut execution = ShellExecution::start(&mut environment, &node, b"", true).unwrap();
        let started = Instant::now();
        let blocked = (0..1_000).any(|_| {
            matches!(
                execution.poll(&mut environment, true).unwrap(),
                MachinePoll::Blocked
            )
        });
        // Asking to advance time must not jump a real-time clock or sleep the host.
        assert!(blocked);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(environment.clock.monotonic_ns() < 5_000_000_000);
        let wait = environment.host_wait_until().unwrap();
        assert!(wait > Duration::from_secs(5) && wait <= Duration::from_secs(10));
    }
}
