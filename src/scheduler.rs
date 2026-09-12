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
    current: Option<ProcessId>,
}

impl Scheduler {
    /// Create a scheduler with one currently running root process.
    pub fn new(root: ProcessId) -> Self {
        Self {
            states: BTreeMap::from([(root, TaskState::Running)]),
            runnable: VecDeque::new(),
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

    pub fn state(&self, pid: ProcessId) -> Option<TaskState> {
        self.states.get(&pid).copied()
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
}
