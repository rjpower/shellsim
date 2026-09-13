//! Deterministic cooperative task scheduling primitives.
//!
//! The scheduler owns only runnable/blocked lifecycle and stable dispatch order. Interpreter
//! continuations remain typed process payloads outside this module. It never creates host threads
//! or waits on host time; callers advance the virtual timeline and wake tasks explicitly.

use std::collections::{BTreeMap, VecDeque};

use crate::process::ProcessId;

/// Maximum number of tasks retained by one scheduler.
pub const MAX_TASKS: usize = crate::process::MAX_PROCESSES;

/// Resource condition on which a cooperative task is suspended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitReason {
    Timer(u64),
    InputReadable(u32),
    PipeReadable(u32),
    PipeWritable(u32),
    Child(ProcessId),
    /// Any descriptor progress or exit within a modeled child's process subtree.
    ChildActivity(ProcessId),
    /// Child completion, bounded by an absolute virtual monotonic deadline.
    ChildDeadline(ProcessId, u64),
    /// Child-tree descriptor progress or completion, bounded by a virtual deadline.
    ChildActivityDeadline(ProcessId, u64),
}

/// Scheduler-owned lifecycle for one logical task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskState {
    Runnable,
    Running,
    Blocked(WaitReason),
    /// Suspended by job control while retaining the state used by `SIGCONT`.
    Stopped(StoppedTask),
    Exited(i32),
}

/// Scheduler state restored when a stopped task receives `SIGCONT`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoppedTask {
    Runnable,
    Blocked(WaitReason),
}

/// Errors reject invalid transitions without partially changing scheduler state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulerError {
    Capacity,
    DuplicateTask,
    UnknownTask,
    NoRunningTask,
    TaskAlreadyRunning,
    InvalidTransition,
}

/// Single-threaded FIFO scheduler with deterministic wake ordering.
#[derive(Clone)]
pub struct Scheduler {
    states: BTreeMap<ProcessId, TaskState>,
    runnable: VecDeque<ProcessId>,
    blocked: VecDeque<ProcessId>,
    current: Option<ProcessId>,
}

impl Scheduler {
    /// Create a scheduler with one currently running root process.
    pub fn new(root: ProcessId) -> Self {
        Self {
            states: BTreeMap::from([(root, TaskState::Running)]),
            runnable: VecDeque::new(),
            blocked: VecDeque::new(),
            current: Some(root),
        }
    }

    /// Register a new runnable task at the back of the stable FIFO queue.
    pub fn spawn(&mut self, pid: ProcessId) -> Result<(), SchedulerError> {
        if self.states.contains_key(&pid) {
            return Err(SchedulerError::DuplicateTask);
        }
        if self.states.len() >= MAX_TASKS {
            return Err(SchedulerError::Capacity);
        }
        self.states.insert(pid, TaskState::Runnable);
        self.runnable.push_back(pid);
        Ok(())
    }

    /// Yield the current task and place it at the end of the runnable queue.
    pub fn yield_current(&mut self) -> Result<ProcessId, SchedulerError> {
        let pid = self.current.take().ok_or(SchedulerError::NoRunningTask)?;
        self.states.insert(pid, TaskState::Runnable);
        self.runnable.push_back(pid);
        Ok(pid)
    }

    /// Suspend the current task on an explicit modeled resource.
    pub fn block_current(&mut self, reason: WaitReason) -> Result<ProcessId, SchedulerError> {
        let pid = self.current.take().ok_or(SchedulerError::NoRunningTask)?;
        self.states.insert(pid, TaskState::Blocked(reason));
        self.blocked.push_back(pid);
        Ok(pid)
    }

    /// Mark the current task exited. The record remains until its owner reaps it.
    pub fn exit_current(&mut self, status: i32) -> Result<ProcessId, SchedulerError> {
        let pid = self.current.take().ok_or(SchedulerError::NoRunningTask)?;
        self.states.insert(pid, TaskState::Exited(status));
        Ok(pid)
    }

    /// Wake one blocked task. Repeated wakes are idempotent for an already runnable task.
    pub fn wake(&mut self, pid: ProcessId) -> Result<(), SchedulerError> {
        match self.states.get(&pid).copied() {
            Some(TaskState::Blocked(_)) => {
                self.states.insert(pid, TaskState::Runnable);
                self.blocked.retain(|blocked| *blocked != pid);
                self.runnable.push_back(pid);
                Ok(())
            }
            Some(TaskState::Runnable) => Ok(()),
            Some(TaskState::Stopped(StoppedTask::Blocked(_))) => {
                self.blocked.retain(|blocked| *blocked != pid);
                self.states
                    .insert(pid, TaskState::Stopped(StoppedTask::Runnable));
                Ok(())
            }
            Some(TaskState::Stopped(StoppedTask::Runnable)) => Ok(()),
            Some(TaskState::Running | TaskState::Exited(_)) => {
                Err(SchedulerError::InvalidTransition)
            }
            None => Err(SchedulerError::UnknownTask),
        }
    }

    /// Stop a live task without discarding the resource condition it was waiting on.
    ///
    /// Waking a stopped blocked task changes its saved resume state to runnable while leaving it
    /// stopped. This prevents elapsed timers and descriptor readiness from being lost.
    pub fn stop(&mut self, pid: ProcessId) -> Result<(), SchedulerError> {
        let stopped = match self.states.get(&pid).copied() {
            Some(TaskState::Runnable) => {
                self.runnable.retain(|queued| *queued != pid);
                StoppedTask::Runnable
            }
            Some(TaskState::Running) if self.current == Some(pid) => {
                self.current = None;
                StoppedTask::Runnable
            }
            Some(TaskState::Blocked(reason)) => StoppedTask::Blocked(reason),
            Some(TaskState::Stopped(_)) => return Ok(()),
            Some(TaskState::Running | TaskState::Exited(_)) => {
                return Err(SchedulerError::InvalidTransition)
            }
            None => return Err(SchedulerError::UnknownTask),
        };
        self.states.insert(pid, TaskState::Stopped(stopped));
        Ok(())
    }

    /// Continue a stopped task, restoring its runnable or blocked scheduler state.
    pub fn continue_task(&mut self, pid: ProcessId) -> Result<(), SchedulerError> {
        match self.states.get(&pid).copied() {
            Some(TaskState::Stopped(StoppedTask::Runnable)) => {
                self.states.insert(pid, TaskState::Runnable);
                self.runnable.push_back(pid);
                Ok(())
            }
            Some(TaskState::Stopped(StoppedTask::Blocked(reason))) => {
                self.states.insert(pid, TaskState::Blocked(reason));
                Ok(())
            }
            Some(TaskState::Runnable | TaskState::Running | TaskState::Blocked(_)) => Ok(()),
            Some(TaskState::Exited(_)) => Err(SchedulerError::InvalidTransition),
            None => Err(SchedulerError::UnknownTask),
        }
    }

    /// Wake all tasks waiting on one modeled resource in stable blocking order.
    pub fn wake_waiters(&mut self, reason: WaitReason) -> usize {
        let waiting: Vec<ProcessId> = self
            .blocked
            .iter()
            .copied()
            .filter(|pid| {
                matches!(
                    self.state(*pid),
                    Some(TaskState::Blocked(blocked))
                        | Some(TaskState::Stopped(StoppedTask::Blocked(blocked)))
                        if blocked == reason
                )
            })
            .collect();
        for pid in &waiting {
            self.wake(*pid)
                .expect("blocked queue contains a known blocked task");
        }
        waiting.len()
    }

    /// Wake tasks whose next child-status check may now complete.
    pub fn wake_child_waiters(&mut self, child: ProcessId) -> usize {
        self.wake_matching(|reason| {
            matches!(
                reason,
                WaitReason::Child(pid) | WaitReason::ChildDeadline(pid, _) if pid == child
            )
        })
    }

    /// Wake tasks coordinating pipe activity anywhere beneath one child handle.
    pub fn wake_child_activity_waiters(&mut self, child: ProcessId) -> usize {
        self.wake_matching(|reason| {
            matches!(
                reason,
                WaitReason::ChildActivity(pid) | WaitReason::ChildActivityDeadline(pid, _)
                    if pid == child
            )
        })
    }

    fn wake_matching(&mut self, predicate: impl Fn(WaitReason) -> bool) -> usize {
        let waiting: Vec<ProcessId> = self
            .blocked
            .iter()
            .copied()
            .filter(|pid| {
                matches!(
                    self.state(*pid),
                    Some(TaskState::Blocked(reason))
                        | Some(TaskState::Stopped(StoppedTask::Blocked(reason)))
                        if predicate(reason)
                )
            })
            .collect();
        for pid in &waiting {
            self.wake(*pid)
                .expect("blocked queue contains a known blocked task");
        }
        waiting.len()
    }

    /// Dispatch the next runnable task, or `None` when every retained task is blocked/exited.
    pub fn dispatch(&mut self) -> Result<Option<ProcessId>, SchedulerError> {
        if self.current.is_some() {
            return Err(SchedulerError::TaskAlreadyRunning);
        }
        let Some(pid) = self.runnable.pop_front() else {
            return Ok(None);
        };
        let state = self
            .states
            .get_mut(&pid)
            .ok_or(SchedulerError::UnknownTask)?;
        if *state != TaskState::Runnable {
            return Err(SchedulerError::InvalidTransition);
        }
        *state = TaskState::Running;
        self.current = Some(pid);
        Ok(Some(pid))
    }

    pub fn current(&self) -> Option<ProcessId> {
        self.current
    }

    /// Whether another task is queued to run after the current task yields or exits.
    pub fn has_runnable(&self) -> bool {
        !self.runnable.is_empty()
    }

    pub fn state(&self, pid: ProcessId) -> Option<TaskState> {
        self.states.get(&pid).copied()
    }

    /// Timer-blocked tasks in stable blocking order, used for legacy enclosing deadlines that do
    /// not yet carry a concrete process ID.
    pub(crate) fn timer_waiters(&self) -> Vec<ProcessId> {
        self.blocked
            .iter()
            .copied()
            .filter(|pid| {
                matches!(
                    self.state(*pid),
                    Some(TaskState::Blocked(WaitReason::Timer(_)))
                        | Some(TaskState::Stopped(StoppedTask::Blocked(WaitReason::Timer(
                            _
                        ))))
                )
            })
            .collect()
    }

    /// Remove an exited task after its parent has collected the status.
    pub fn reap(&mut self, pid: ProcessId) -> Result<i32, SchedulerError> {
        match self.states.get(&pid).copied() {
            Some(TaskState::Exited(status)) => {
                self.states.remove(&pid);
                Ok(status)
            }
            Some(_) => Err(SchedulerError::InvalidTransition),
            None => Err(SchedulerError::UnknownTask),
        }
    }

    /// Remove a task that failed setup before it was ever dispatched.
    pub(crate) fn discard_runnable(&mut self, pid: ProcessId) -> Result<(), SchedulerError> {
        if self.states.get(&pid) != Some(&TaskState::Runnable) {
            return Err(SchedulerError::InvalidTransition);
        }
        self.states.remove(&pid);
        self.runnable.retain(|queued| *queued != pid);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_and_wake_order_is_stable() {
        let mut scheduler = Scheduler::new(1);
        scheduler.spawn(2).unwrap();
        scheduler.spawn(3).unwrap();
        scheduler.block_current(WaitReason::Child(2)).unwrap();
        assert_eq!(scheduler.dispatch().unwrap(), Some(2));
        scheduler.block_current(WaitReason::Timer(10)).unwrap();
        assert_eq!(scheduler.dispatch().unwrap(), Some(3));
        scheduler.yield_current().unwrap();
        scheduler.wake(1).unwrap();
        scheduler.wake(2).unwrap();
        assert_eq!(scheduler.dispatch().unwrap(), Some(3));
        scheduler.exit_current(0).unwrap();
        assert_eq!(scheduler.dispatch().unwrap(), Some(1));
    }

    #[test]
    fn invalid_transitions_are_atomic() {
        let mut scheduler = Scheduler::new(1);
        assert_eq!(
            scheduler.dispatch(),
            Err(SchedulerError::TaskAlreadyRunning)
        );
        assert_eq!(scheduler.wake(99), Err(SchedulerError::UnknownTask));
        assert_eq!(scheduler.current(), Some(1));
        assert_eq!(scheduler.state(1), Some(TaskState::Running));
    }

    #[test]
    fn resource_wakes_preserve_the_order_tasks_blocked() {
        let mut scheduler = Scheduler::new(1);
        scheduler.spawn(2).unwrap();
        scheduler.spawn(3).unwrap();
        scheduler
            .block_current(WaitReason::PipeReadable(7))
            .unwrap();
        assert_eq!(scheduler.dispatch().unwrap(), Some(2));
        scheduler
            .block_current(WaitReason::PipeReadable(7))
            .unwrap();
        assert_eq!(scheduler.dispatch().unwrap(), Some(3));
        scheduler
            .block_current(WaitReason::PipeWritable(7))
            .unwrap();

        assert_eq!(scheduler.wake_waiters(WaitReason::PipeReadable(7)), 2);
        assert_eq!(scheduler.dispatch().unwrap(), Some(1));
        scheduler.yield_current().unwrap();
        assert_eq!(scheduler.dispatch().unwrap(), Some(2));
        assert_eq!(
            scheduler.state(3),
            Some(TaskState::Blocked(WaitReason::PipeWritable(7)))
        );
    }

    #[test]
    fn child_wakes_include_deadlines_but_keep_activity_distinct() {
        let mut scheduler = Scheduler::new(1);
        scheduler.spawn(2).unwrap();
        scheduler.spawn(3).unwrap();
        scheduler
            .block_current(WaitReason::ChildDeadline(9, 100))
            .unwrap();
        assert_eq!(scheduler.dispatch().unwrap(), Some(2));
        scheduler
            .block_current(WaitReason::ChildActivityDeadline(9, 100))
            .unwrap();
        assert_eq!(scheduler.dispatch().unwrap(), Some(3));
        scheduler.block_current(WaitReason::Child(8)).unwrap();

        assert_eq!(scheduler.wake_child_waiters(9), 1);
        assert_eq!(scheduler.state(1), Some(TaskState::Runnable));
        assert_eq!(
            scheduler.state(2),
            Some(TaskState::Blocked(WaitReason::ChildActivityDeadline(
                9, 100
            )))
        );
        assert_eq!(scheduler.wake_child_activity_waiters(9), 1);
        assert_eq!(scheduler.state(2), Some(TaskState::Runnable));
        assert_eq!(
            scheduler.state(3),
            Some(TaskState::Blocked(WaitReason::Child(8)))
        );
    }

    #[test]
    fn stopped_tasks_retain_waits_and_record_wakes_until_continued() {
        let mut scheduler = Scheduler::new(1);
        scheduler.spawn(2).unwrap();
        scheduler
            .block_current(WaitReason::PipeReadable(7))
            .unwrap();
        scheduler.stop(1).unwrap();
        assert_eq!(
            scheduler.state(1),
            Some(TaskState::Stopped(StoppedTask::Blocked(
                WaitReason::PipeReadable(7)
            )))
        );
        assert_eq!(scheduler.wake_waiters(WaitReason::PipeReadable(7)), 1);
        assert_eq!(
            scheduler.state(1),
            Some(TaskState::Stopped(StoppedTask::Runnable))
        );
        scheduler.continue_task(1).unwrap();
        assert_eq!(scheduler.state(1), Some(TaskState::Runnable));
    }
}
