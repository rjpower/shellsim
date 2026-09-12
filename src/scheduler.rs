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
    PipeReadable(u32),
    PipeWritable(u32),
    Child(ProcessId),
    /// Any descriptor progress or exit within a modeled child's process subtree.
    ChildActivity(ProcessId),
}

/// Scheduler-owned lifecycle for one logical task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskState {
    Runnable,
    Running,
    Blocked(WaitReason),
    Exited(i32),
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
            Some(TaskState::Running | TaskState::Exited(_)) => {
                Err(SchedulerError::InvalidTransition)
            }
            None => Err(SchedulerError::UnknownTask),
        }
    }

    /// Wake all tasks waiting on one modeled resource in stable blocking order.
    pub fn wake_waiters(&mut self, reason: WaitReason) -> usize {
        let waiting: Vec<ProcessId> = self
            .blocked
            .iter()
            .copied()
            .filter(|pid| self.state(*pid) == Some(TaskState::Blocked(reason)))
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
}
