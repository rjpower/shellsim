//! Deterministic virtual time and event ordering.
//!
//! Shellsim never reads the host clock and never blocks a host thread.  A [`Timeline`] owns
//! three deliberately separate notions of time:
//!
//! * monotonic time orders effects and deadlines;
//! * wall time is a presentation offset from monotonic time and may be adjusted;
//! * CPU time is accounted by [`crate::resources::Resources`], not this module.
//!
//! Events at the same instant are ordered by an insertion sequence.  Consequently a cloned
//! timeline contains all state required to replay the same future ordering.

use std::collections::{BTreeMap, VecDeque};

pub const NANOS_PER_MICROSECOND: u64 = 1_000;
pub const NANOS_PER_MILLISECOND: u64 = 1_000_000;
pub const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// 2025-01-01T00:00:00Z.  A fixed default makes independent simulations reproducible.
pub const DEFAULT_EPOCH_UTC_NS: i128 = 1_735_689_600_000_000_000;

/// The initial executor has one runnable shell task.  Giving it an explicit identity keeps the
/// event contract ready for resumable jobs without baking "the current process" into events.
pub const MAIN_TASK_ID: u64 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventId {
    deadline_ns: u64,
    sequence: u64,
}

impl EventId {
    pub fn deadline_ns(self) -> u64 {
        self.deadline_ns
    }

    pub fn sequence(self) -> u64 {
        self.sequence
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EventKind {
    WakeTask {
        task: u64,
    },
    Deadline {
        task: u64,
    },
    /// Deliver a modeled signal to one logical process at a virtual monotonic instant.
    SignalTask {
        task: u64,
        signal: crate::process::Signal,
        descendants: bool,
    },
    /// An effect injected by a simulator driver.  Both fields are opaque deterministic data;
    /// they never grant access to the host environment.
    External {
        source: String,
        payload: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledEvent {
    pub id: EventId,
    pub kind: EventKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimelineLimits {
    pub max_pending_events: usize,
    /// Maximum distance into the future at which one event may be scheduled.
    pub max_event_horizon_ns: u64,
}

impl Default for TimelineLimits {
    fn default() -> Self {
        Self {
            max_pending_events: 16_384,
            // A century is ample for task simulation while rejecting accidental infinities.
            max_event_horizon_ns: 100 * 365 * 24 * 60 * 60 * NANOS_PER_SECOND,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TimelineError {
    Overflow,
    WouldRewind { now_ns: u64, requested_ns: u64 },
    EventLimit { limit: usize },
    EventBeyondHorizon { horizon_ns: u64 },
}

impl std::fmt::Display for TimelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow => f.write_str("virtual time overflow"),
            Self::WouldRewind {
                now_ns,
                requested_ns,
            } => write!(
                f,
                "virtual monotonic clock cannot rewind from {now_ns}ns to {requested_ns}ns"
            ),
            Self::EventLimit { limit } => {
                write!(f, "virtual timeline event limit exceeded ({limit})")
            }
            Self::EventBeyondHorizon { horizon_ns } => write!(
                f,
                "virtual event exceeds the configured {horizon_ns}ns scheduling horizon"
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockOutcome {
    Completed,
    Interrupted(ScheduledEvent),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Timeline {
    monotonic_ns: u64,
    epoch_utc_ns: i128,
    wall_adjustment_ns: i128,
    next_sequence: u64,
    pending: BTreeMap<EventId, EventKind>,
    ready: VecDeque<ScheduledEvent>,
    limits: TimelineLimits,
    /// Sum of requested sleep durations.  This is telemetry, not a clock domain.
    slept_ns: u64,
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Timeline {
    pub fn new() -> Self {
        Self::with_epoch_and_limits(DEFAULT_EPOCH_UTC_NS, TimelineLimits::default())
    }

    pub fn with_epoch(epoch_utc_ns: i128) -> Self {
        Self::with_epoch_and_limits(epoch_utc_ns, TimelineLimits::default())
    }

    pub fn with_epoch_and_limits(epoch_utc_ns: i128, limits: TimelineLimits) -> Self {
        Self {
            monotonic_ns: 0,
            epoch_utc_ns,
            wall_adjustment_ns: 0,
            next_sequence: 0,
            pending: BTreeMap::new(),
            ready: VecDeque::new(),
            limits,
            slept_ns: 0,
        }
    }

    pub fn monotonic_ns(&self) -> u64 {
        self.monotonic_ns
    }

    pub fn monotonic_seconds(&self) -> f64 {
        self.monotonic_ns as f64 / NANOS_PER_SECOND as f64
    }

    pub fn wall_time_ns(&self) -> Result<i128, TimelineError> {
        self.epoch_utc_ns
            .checked_add(i128::from(self.monotonic_ns))
            .and_then(|value| value.checked_add(self.wall_adjustment_ns))
            .ok_or(TimelineError::Overflow)
    }

    pub fn wall_time_seconds(&self) -> Result<f64, TimelineError> {
        Ok(self.wall_time_ns()? as f64 / NANOS_PER_SECOND as f64)
    }

    pub fn wall_time_millis(&self) -> Result<i128, TimelineError> {
        Ok(self
            .wall_time_ns()?
            .div_euclid(i128::from(NANOS_PER_MILLISECOND)))
    }

    pub fn wall_time_seconds_floor(&self) -> Result<i128, TimelineError> {
        Ok(self
            .wall_time_ns()?
            .div_euclid(i128::from(NANOS_PER_SECOND)))
    }

    /// Change the wall-clock presentation without affecting monotonic deadlines.
    pub fn adjust_wall_time(&mut self, delta_ns: i128) -> Result<(), TimelineError> {
        let adjustment = self
            .wall_adjustment_ns
            .checked_add(delta_ns)
            .ok_or(TimelineError::Overflow)?;
        self.epoch_utc_ns
            .checked_add(i128::from(self.monotonic_ns))
            .and_then(|value| value.checked_add(adjustment))
            .ok_or(TimelineError::Overflow)?;
        self.wall_adjustment_ns = adjustment;
        Ok(())
    }

    pub fn slept_ns(&self) -> u64 {
        self.slept_ns
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn ready_len(&self) -> usize {
        self.ready.len()
    }

    pub fn next_deadline_ns(&self) -> Option<u64> {
        self.pending.first_key_value().map(|(id, _)| id.deadline_ns)
    }

    pub fn schedule_after(
        &mut self,
        delay_ns: u64,
        kind: EventKind,
    ) -> Result<EventId, TimelineError> {
        let deadline_ns = self
            .monotonic_ns
            .checked_add(delay_ns)
            .ok_or(TimelineError::Overflow)?;
        self.schedule_at(deadline_ns, kind)
    }

    /// Schedule a cooperative task wake and account the requested duration as sleep telemetry.
    pub fn schedule_wake_after(
        &mut self,
        task: u64,
        duration_ns: u64,
    ) -> Result<EventId, TimelineError> {
        let event = self.schedule_after(duration_ns, EventKind::WakeTask { task })?;
        self.slept_ns = self.slept_ns.saturating_add(duration_ns);
        Ok(event)
    }

    pub fn schedule_at(
        &mut self,
        deadline_ns: u64,
        kind: EventKind,
    ) -> Result<EventId, TimelineError> {
        if deadline_ns < self.monotonic_ns {
            return Err(TimelineError::WouldRewind {
                now_ns: self.monotonic_ns,
                requested_ns: deadline_ns,
            });
        }
        if self.pending.len().saturating_add(self.ready.len()) >= self.limits.max_pending_events {
            return Err(TimelineError::EventLimit {
                limit: self.limits.max_pending_events,
            });
        }
        if deadline_ns - self.monotonic_ns > self.limits.max_event_horizon_ns {
            return Err(TimelineError::EventBeyondHorizon {
                horizon_ns: self.limits.max_event_horizon_ns,
            });
        }
        let id = EventId {
            deadline_ns,
            sequence: self.next_sequence,
        };
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(TimelineError::Overflow)?;
        self.pending.insert(id, kind);
        Ok(id)
    }

    /// Cancel an event that has not fired.  Returns whether it was still pending.
    pub fn cancel(&mut self, id: EventId) -> bool {
        self.pending.remove(&id).is_some()
    }

    /// Cancel future wake/deadline events owned by one logical task.
    ///
    /// Process exit uses this to prevent an abandoned sleep from advancing virtual time later.
    pub fn cancel_task_events(&mut self, task: u64) -> usize {
        let ids = self
            .pending
            .iter()
            .filter_map(|(id, kind)| match kind {
                EventKind::WakeTask { task: owner }
                | EventKind::Deadline { task: owner }
                | EventKind::SignalTask { task: owner, .. }
                    if *owner == task =>
                {
                    Some(*id)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        for id in &ids {
            self.pending.remove(id);
        }
        self.ready.retain(|event| {
            !matches!(
                event.kind,
                EventKind::WakeTask { task: owner }
                | EventKind::Deadline { task: owner }
                | EventKind::SignalTask { task: owner, .. }
                    if owner == task
            )
        });
        ids.len()
    }

    /// Advance to an exact monotonic instant and make every due event ready in stable order.
    pub fn advance_to(&mut self, target_ns: u64) -> Result<Vec<ScheduledEvent>, TimelineError> {
        if target_ns < self.monotonic_ns {
            return Err(TimelineError::WouldRewind {
                now_ns: self.monotonic_ns,
                requested_ns: target_ns,
            });
        }
        self.monotonic_ns = target_ns;
        let due_ids: Vec<EventId> = self
            .pending
            .range(
                ..=EventId {
                    deadline_ns: target_ns,
                    sequence: u64::MAX,
                },
            )
            .map(|(id, _)| *id)
            .collect();
        let mut fired = Vec::with_capacity(due_ids.len());
        for id in due_ids {
            let kind = self
                .pending
                .remove(&id)
                .expect("event key was collected from pending map");
            let event = ScheduledEvent { id, kind };
            self.ready.push_back(event.clone());
            fired.push(event);
        }
        Ok(fired)
    }

    pub fn advance_by(&mut self, duration_ns: u64) -> Result<Vec<ScheduledEvent>, TimelineError> {
        let target = self
            .monotonic_ns
            .checked_add(duration_ns)
            .ok_or(TimelineError::Overflow)?;
        self.advance_to(target)
    }

    /// Fast-forward to the next event, if any.  Events sharing its deadline fire as one batch.
    pub fn advance_to_next(&mut self) -> Result<Vec<ScheduledEvent>, TimelineError> {
        match self.next_deadline_ns() {
            Some(deadline) => self.advance_to(deadline),
            None => Ok(Vec::new()),
        }
    }

    pub fn pop_ready(&mut self) -> Option<ScheduledEvent> {
        self.ready.pop_front()
    }

    fn remove_ready(&mut self, id: EventId) {
        self.ready.retain(|event| event.id != id);
    }

    /// Block one virtual task.  Other due events are left in the ready queue for the scheduler.
    /// The clock jumps only when no runnable work exists (the current synchronous executor calls
    /// this precisely at that point).
    pub fn block_task(
        &mut self,
        task: u64,
        duration_ns: u64,
    ) -> Result<BlockOutcome, TimelineError> {
        let wake = self.schedule_wake_after(task, duration_ns)?;
        loop {
            let fired = self.advance_to_next()?;
            for event in fired {
                match event.kind {
                    EventKind::WakeTask { task: owner } if event.id == wake && owner == task => {
                        self.remove_ready(event.id);
                        return Ok(BlockOutcome::Completed);
                    }
                    EventKind::Deadline { task: owner } if owner == task => {
                        self.cancel(wake);
                        self.remove_ready(wake);
                        self.remove_ready(event.id);
                        return Ok(BlockOutcome::Interrupted(event));
                    }
                    _ => {}
                }
            }
        }
    }

    // Compatibility accessors for existing embedders.  New code should name the clock domain.
    pub fn now_ms(&self) -> u64 {
        self.monotonic_ns / NANOS_PER_MILLISECOND
    }

    pub fn unix_ms(&self) -> u64 {
        self.wall_time_millis()
            .ok()
            .and_then(|value| u64::try_from(value).ok())
            .unwrap_or_default()
    }

    pub fn unix_secs(&self) -> u64 {
        self.wall_time_seconds_floor()
            .ok()
            .and_then(|value| u64::try_from(value).ok())
            .unwrap_or_default()
    }

    pub fn sleep_ms(&mut self, millis: u64) {
        let duration = millis.saturating_mul(NANOS_PER_MILLISECOND);
        let _ = self.block_task(MAIN_TASK_ID, duration);
    }

    pub fn tick(&mut self, ticks: u64) {
        let duration = ticks.saturating_mul(NANOS_PER_MILLISECOND);
        let _ = self.advance_by(duration);
    }
}

/// Source-compatible name retained for callers of the original API.
pub type Clock = Timeline;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_deadlines_fire_in_insertion_order() {
        let mut timeline = Timeline::new();
        let first = timeline
            .schedule_after(10, EventKind::WakeTask { task: 2 })
            .unwrap();
        let second = timeline
            .schedule_after(10, EventKind::WakeTask { task: 1 })
            .unwrap();

        let fired = timeline.advance_to_next().unwrap();
        assert_eq!(
            fired.iter().map(|event| event.id).collect::<Vec<_>>(),
            vec![first, second]
        );
        assert_eq!(timeline.monotonic_ns(), 10);
    }

    #[test]
    fn wall_adjustment_does_not_change_monotonic_deadlines() {
        let mut timeline = Timeline::new();
        let event = timeline
            .schedule_after(NANOS_PER_SECOND, EventKind::Deadline { task: 0 })
            .unwrap();
        timeline
            .adjust_wall_time(-3_600 * i128::from(NANOS_PER_SECOND))
            .unwrap();

        assert_eq!(timeline.next_deadline_ns(), Some(event.deadline_ns()));
        assert_eq!(timeline.monotonic_ns(), 0);
        assert_eq!(
            timeline.wall_time_ns().unwrap(),
            DEFAULT_EPOCH_UTC_NS - 3_600_000_000_000
        );
    }

    #[test]
    fn deadline_interrupts_block_before_its_wake_event() {
        let mut timeline = Timeline::new();
        let deadline = timeline
            .schedule_after(2 * NANOS_PER_SECOND, EventKind::Deadline { task: 7 })
            .unwrap();

        assert_eq!(
            timeline.block_task(7, 10 * NANOS_PER_SECOND).unwrap(),
            BlockOutcome::Interrupted(ScheduledEvent {
                id: deadline,
                kind: EventKind::Deadline { task: 7 },
            })
        );
        assert_eq!(timeline.monotonic_ns(), 2 * NANOS_PER_SECOND);
        assert_eq!(timeline.pending_len(), 0);
    }

    #[test]
    fn clone_is_a_complete_replay_snapshot() {
        let mut timeline = Timeline::new();
        timeline
            .schedule_after(
                12,
                EventKind::External {
                    source: "test".into(),
                    payload: "ready".into(),
                },
            )
            .unwrap();
        let mut snapshot = timeline.clone();

        assert_eq!(
            timeline.advance_to_next().unwrap(),
            snapshot.advance_to_next().unwrap()
        );
        assert_eq!(timeline, snapshot);
    }

    #[test]
    fn event_count_and_horizon_are_bounded() {
        let limits = TimelineLimits {
            max_pending_events: 1,
            max_event_horizon_ns: 5,
        };
        let mut timeline = Timeline::with_epoch_and_limits(0, limits);
        assert!(matches!(
            timeline.schedule_after(6, EventKind::WakeTask { task: 0 }),
            Err(TimelineError::EventBeyondHorizon { .. })
        ));
        timeline
            .schedule_after(5, EventKind::WakeTask { task: 0 })
            .unwrap();
        assert!(matches!(
            timeline.schedule_after(1, EventKind::WakeTask { task: 1 }),
            Err(TimelineError::EventLimit { limit: 1 })
        ));
    }

    #[test]
    fn canceling_task_events_removes_pending_and_ready_work() {
        let mut timeline = Timeline::new();
        timeline
            .schedule_after(1, EventKind::WakeTask { task: 7 })
            .unwrap();
        timeline
            .schedule_after(2, EventKind::Deadline { task: 7 })
            .unwrap();
        timeline
            .schedule_after(3, EventKind::WakeTask { task: 8 })
            .unwrap();
        timeline.advance_to(1).unwrap();

        assert_eq!(timeline.cancel_task_events(7), 1);
        assert_eq!(timeline.ready_len(), 0);
        assert_eq!(timeline.pending_len(), 1);
        assert_eq!(timeline.next_deadline_ns(), Some(3));
    }
}
