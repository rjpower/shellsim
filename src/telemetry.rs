//! Bounded, ordered telemetry for simulated command invocations.
//!
//! The log records occurrences rather than sets of command names. An event is installed before
//! dispatch and completed in place after the command returns, including across cooperative
//! suspension. Storage is bounded by both event count and modeled bytes so simulated input cannot
//! cause unbounded host allocation. Oldest events are discarded deterministically.

use std::collections::VecDeque;

use serde::Serialize;

use crate::process::ProcessId;

/// Maximum retained command occurrences in one environment.
pub const MAX_INVOCATION_EVENTS: usize = 4_096;
/// Maximum modeled heap bytes retained by command telemetry.
pub const MAX_INVOCATION_BYTES: u64 = 4 * 1024 * 1024;

/// How faithfully one command implementation models its namesake.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandTrust {
    /// The supported command surface has real modeled effects.
    Real,
    /// The implementation deliberately supports only a documented subset.
    Partial,
    /// The command is a compatibility no-op or is unavailable.
    NoOp,
}

/// One ordered command occurrence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct InvocationEvent {
    pub sequence: u64,
    pub pid: ProcessId,
    pub argv: Vec<String>,
    pub trust: CommandTrust,
    /// `None` means the invocation is still suspended or otherwise active.
    pub status: Option<i32>,
    /// Inclusive deterministic CPU consumed while this invocation was active.
    pub cpu: Option<u64>,
    /// Inclusive VFS disk change while this invocation was active.
    pub disk_delta: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unsupported_reason: Option<String>,
}

#[derive(Clone, Debug)]
struct RetainedInvocation {
    event: InvocationEvent,
    cpu_before: u64,
    disk_before: u64,
    modeled_bytes: u64,
}

/// Cloneable bounded occurrence log owned by an [`Environment`](crate::Environment).
#[derive(Clone, Debug, Default)]
pub struct InvocationLog {
    events: VecDeque<RetainedInvocation>,
    modeled_bytes: u64,
    next_sequence: u64,
    dropped: u64,
}

impl InvocationLog {
    /// Begin an invocation. Completion is LIFO per process, matching nested native dispatch.
    pub fn begin(
        &mut self,
        pid: ProcessId,
        argv: &[String],
        trust: CommandTrust,
        unsupported_reason: Option<String>,
        cpu_before: u64,
        disk_before: u64,
    ) {
        let Some(sequence) = self.next_sequence.checked_add(1) else {
            self.dropped = self.dropped.saturating_add(1);
            return;
        };
        let current_sequence = self.next_sequence;
        self.next_sequence = sequence;
        let modeled_bytes = argv
            .iter()
            .fold(128_u64, |size, value| {
                size.saturating_add(value.len() as u64)
            })
            .saturating_add(
                unsupported_reason
                    .as_ref()
                    .map_or(0, |reason| reason.len() as u64),
            );
        if modeled_bytes > MAX_INVOCATION_BYTES {
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        while self.events.len() >= MAX_INVOCATION_EVENTS
            || self.modeled_bytes.saturating_add(modeled_bytes) > MAX_INVOCATION_BYTES
        {
            let Some(event) = self.events.pop_front() else {
                break;
            };
            self.modeled_bytes = self.modeled_bytes.saturating_sub(event.modeled_bytes);
            self.dropped = self.dropped.saturating_add(1);
        }
        self.modeled_bytes = self.modeled_bytes.saturating_add(modeled_bytes);
        self.events.push_back(RetainedInvocation {
            event: InvocationEvent {
                sequence: current_sequence,
                pid,
                argv: argv.to_vec(),
                trust,
                status: None,
                cpu: None,
                disk_delta: None,
                unsupported_reason,
            },
            cpu_before,
            disk_before,
            modeled_bytes,
        });
    }

    /// Complete the innermost active invocation for `pid`.
    pub fn finish_latest(&mut self, pid: ProcessId, status: i32, cpu_after: u64, disk_after: u64) {
        let Some(invocation) =
            self.events.iter_mut().rev().find(|invocation| {
                invocation.event.pid == pid && invocation.event.status.is_none()
            })
        else {
            return;
        };
        invocation.event.status = Some(status);
        invocation.event.cpu = Some(cpu_after.saturating_sub(invocation.cpu_before));
        invocation.event.disk_delta = Some(signed_delta(disk_after, invocation.disk_before));
    }

    /// Sequence marker suitable for a later [`InvocationLog::events_since`] query.
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Return retained events begun at or after `sequence`.
    pub fn events_since(&self, sequence: u64) -> Vec<InvocationEvent> {
        self.events
            .iter()
            .filter(|invocation| invocation.event.sequence >= sequence)
            .map(|invocation| invocation.event.clone())
            .collect()
    }

    /// Return the complete retained window in occurrence order.
    pub fn events(&self) -> Vec<InvocationEvent> {
        self.events
            .iter()
            .map(|invocation| invocation.event.clone())
            .collect()
    }

    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    pub fn modeled_bytes(&self) -> u64 {
        self.modeled_bytes
    }
}

fn signed_delta(after: u64, before: u64) -> i64 {
    let delta = i128::from(after).saturating_sub(i128::from(before));
    delta.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_invocations_complete_in_lifo_order() {
        let mut log = InvocationLog::default();
        log.begin(7, &["outer".into()], CommandTrust::Partial, None, 10, 20);
        log.begin(7, &["inner".into()], CommandTrust::Real, None, 12, 20);
        log.finish_latest(7, 0, 15, 21);
        log.finish_latest(7, 3, 18, 19);

        let events = log.events();
        assert_eq!(events[0].status, Some(3));
        assert_eq!(events[0].cpu, Some(8));
        assert_eq!(events[0].disk_delta, Some(-1));
        assert_eq!(events[1].status, Some(0));
    }

    #[test]
    fn retention_is_bounded_and_reports_drops() {
        let mut log = InvocationLog::default();
        for index in 0..=MAX_INVOCATION_EVENTS {
            log.begin(
                1,
                &[format!("command-{index}")],
                CommandTrust::Real,
                None,
                0,
                0,
            );
            log.finish_latest(1, 0, 0, 0);
        }
        assert_eq!(log.events().len(), MAX_INVOCATION_EVENTS);
        assert_eq!(log.dropped(), 1);
        assert_eq!(log.events()[0].sequence, 1);
    }
}
