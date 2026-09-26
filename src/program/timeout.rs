//! Resumable `timeout` image: run a command and signal it when a virtual deadline passes.
//!
//! As in GNU timeout, the command leads its own process group unless `--foreground` is given,
//! so the signal reaches the command and everything it started. The wait is a child wait
//! bounded by the deadline; no host time or host process is involved.

use crate::exec::ShellPoll;
use crate::process::{ProcessId, Signal};
use crate::scheduler::WaitReason;
use crate::syscalls::{ClockId, SignalTarget, SpawnSpec, System};

use super::poll_write;

/// Status when the command timed out and `--preserve-status` was not given.
const TIMED_OUT: i32 = 124;
/// Status when timeout itself fails.
const TIMEOUT_FAILED: i32 = 125;

#[derive(Clone)]
pub(crate) struct TimeoutProcess {
    args: Vec<String>,
    started: bool,
    preserve_status: bool,
    foreground: bool,
    signal: Signal,
    kill_after: Option<u64>,
    child: Option<ProcessId>,
    deadline: Option<u64>,
    timed_out: bool,
    killed: bool,
    diagnostic: Vec<u8>,
    diagnostic_offset: usize,
    status: i32,
}

impl TimeoutProcess {
    pub(super) fn new(args: &[String]) -> Self {
        Self {
            args: args.to_vec(),
            started: false,
            preserve_status: false,
            foreground: false,
            signal: Signal::Terminate,
            kill_after: None,
            child: None,
            deadline: None,
            timed_out: false,
            killed: false,
            diagnostic: Vec::new(),
            diagnostic_offset: 0,
            status: 0,
        }
    }

    pub(super) fn poll(&mut self, system: &mut impl System) -> ShellPoll {
        if !self.diagnostic.is_empty() {
            return poll_write(
                system,
                2,
                &self.diagnostic,
                &mut self.diagnostic_offset,
                self.status,
            );
        }
        if !self.started {
            self.started = true;
            return self.start(system);
        }
        let Some(child) = self.child else {
            return ShellPoll::Ready(self.status);
        };
        match system.child_status(child) {
            Ok(Some(_)) => self.finish(system, child),
            Ok(None) => {
                let now = system.clock_time_ns(ClockId::Monotonic).unwrap_or(0);
                if self.deadline.is_some_and(|deadline| now >= deadline) {
                    self.expire(system, child);
                }
                ShellPoll::Blocked(self.wait_reason(child))
            }
            Err(error) => self.fail(format!("timeout: cannot wait for child: {error}")),
        }
    }

    fn start(&mut self, system: &mut impl System) -> ShellPoll {
        let invocation = match crate::commands::proc::parse_timeout(&self.args) {
            Ok(invocation) => invocation,
            Err(error) => return self.fail(format!("timeout: {error}")),
        };
        self.preserve_status = invocation.preserve_status;
        self.foreground = invocation.foreground;
        self.signal = invocation.signal;
        self.kill_after = invocation.kill_after;
        let child = match system.spawn_argv(SpawnSpec {
            argv: invocation.argv,
            detach: true,
            new_process_group: !invocation.foreground,
            ..Default::default()
        }) {
            Ok(child) => child,
            Err(error) => return self.fail(format!("timeout: cannot run command: {error}")),
        };
        self.child = Some(child);
        if invocation.duration > 0 {
            match system.schedule_wake(invocation.duration) {
                Ok(deadline) => self.deadline = Some(deadline),
                Err(error) => {
                    let _ = system.kill(SignalTarget::Process(child), Signal::Kill);
                    self.status = TIMEOUT_FAILED;
                    self.diagnostic = format!("timeout: {error}\n").into_bytes();
                    return ShellPoll::Blocked(WaitReason::Child(child));
                }
            }
        }
        ShellPoll::Blocked(self.wait_reason(child))
    }

    /// Send the configured signal at the first deadline and KILL at the `-k` deadline.
    fn expire(&mut self, system: &mut impl System, child: ProcessId) {
        let target = if self.foreground {
            SignalTarget::Process(child)
        } else {
            SignalTarget::Group(child)
        };
        self.deadline = None;
        if !self.timed_out {
            self.timed_out = true;
            let _ = system.kill(target, self.signal);
            if let Some(kill_after) = self.kill_after {
                self.deadline = system.schedule_wake(kill_after).ok();
            }
        } else if !self.killed {
            self.killed = true;
            let _ = system.kill(target, Signal::Kill);
        }
    }

    fn finish(&mut self, system: &mut impl System, child: ProcessId) -> ShellPoll {
        self.child = None;
        let status = match system.reap_child(child) {
            Ok(status) => status,
            Err(error) => return self.fail(format!("timeout: cannot reap child: {error}")),
        };
        if self.status == TIMEOUT_FAILED {
            return ShellPoll::Pending;
        }
        ShellPoll::Ready(if !self.timed_out || self.preserve_status {
            status
        } else if self.killed {
            128 + Signal::Kill.number()
        } else {
            TIMED_OUT
        })
    }

    fn wait_reason(&self, child: ProcessId) -> WaitReason {
        match self.deadline {
            Some(deadline) => WaitReason::ChildDeadline(child, deadline),
            None => WaitReason::Child(child),
        }
    }

    fn fail(&mut self, message: String) -> ShellPoll {
        self.status = TIMEOUT_FAILED;
        self.diagnostic = format!("{message}\n").into_bytes();
        ShellPoll::Pending
    }
}
