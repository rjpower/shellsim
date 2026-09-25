//! Resumable `sleep` and `usleep` images on the virtual monotonic clock.
//!
//! The process schedules one wake event and blocks on it, so a signal that terminates the
//! process also cancels the pending wake. No host time is observed.

use crate::clock::NANOS_PER_MICROSECOND;
use crate::exec::ShellPoll;
use crate::scheduler::WaitReason;
use crate::syscalls::{ClockId, System};

use super::poll_write;

#[derive(Clone)]
pub(crate) struct SleepProcess {
    duration: Result<u64, String>,
    deadline: Option<u64>,
    diagnostic: Vec<u8>,
    diagnostic_offset: usize,
}

impl SleepProcess {
    /// GNU `sleep`: the sum of one or more durations with optional `s`/`m`/`h`/`d` suffixes.
    pub(super) fn sleep(args: &[String]) -> Self {
        let duration = if args.is_empty() {
            Err("sleep: missing operand".to_string())
        } else {
            args.iter()
                .try_fold(0_u64, |total, argument| {
                    let part = crate::commands::util::parse_duration_ns(argument)
                        .map_err(|_| format!("invalid time interval '{argument}'"))?;
                    total
                        .checked_add(part)
                        .ok_or_else(|| format!("invalid time interval '{argument}'"))
                })
                .map_err(|error| format!("sleep: {error}"))
        };
        Self::new(duration)
    }

    /// `usleep`: one whole number of microseconds.
    pub(super) fn usleep(args: &[String]) -> Self {
        let duration = match args.first() {
            None => Err("usleep: missing operand".to_string()),
            Some(argument) => argument
                .parse::<u64>()
                .ok()
                .and_then(|micros| micros.checked_mul(NANOS_PER_MICROSECOND))
                .ok_or_else(|| format!("usleep: invalid time interval '{argument}'")),
        };
        Self::new(duration)
    }

    fn new(duration: Result<u64, String>) -> Self {
        Self {
            duration,
            deadline: None,
            diagnostic: Vec::new(),
            diagnostic_offset: 0,
        }
    }

    pub(super) fn poll(&mut self, system: &mut impl System) -> ShellPoll {
        if !self.diagnostic.is_empty() {
            return poll_write(system, 2, &self.diagnostic, &mut self.diagnostic_offset, 1);
        }
        let Some(deadline) = self.deadline else {
            let duration = match &self.duration {
                Ok(0) => return ShellPoll::Ready(0),
                Ok(duration) => *duration,
                Err(error) => {
                    self.diagnostic = format!("{error}\n").into_bytes();
                    return ShellPoll::Pending;
                }
            };
            return match system.schedule_wake(duration) {
                Ok(deadline) => {
                    self.deadline = Some(deadline);
                    ShellPoll::Blocked(WaitReason::Timer(deadline))
                }
                Err(error) => {
                    self.diagnostic = format!("sleep: {error}\n").into_bytes();
                    ShellPoll::Pending
                }
            };
        };
        match system.clock_time_ns(ClockId::Monotonic) {
            Ok(now) if now >= deadline => ShellPoll::Ready(0),
            _ => ShellPoll::Blocked(WaitReason::Timer(deadline)),
        }
    }
}
