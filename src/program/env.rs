//! Resumable `env` image with process-local overrides and typed child execution.

use crate::exec::ShellPoll;
use crate::process::ProcessId;
use crate::scheduler::WaitReason;
use crate::syscalls::{SpawnSpec, System};

use super::poll_write;

#[derive(Clone)]
pub(crate) struct EnvProcess {
    args: Vec<String>,
    started: bool,
    waiting: Option<ProcessId>,
    output: Vec<u8>,
    output_offset: usize,
    output_fd: i32,
    reserved_output: u64,
    status: i32,
}

impl EnvProcess {
    pub(super) fn new(args: &[String]) -> Self {
        Self {
            args: args.to_vec(),
            started: false,
            waiting: None,
            output: Vec::new(),
            output_offset: 0,
            output_fd: 1,
            reserved_output: 0,
            status: 0,
        }
    }

    pub(super) fn take_reserved_output(&mut self) -> u64 {
        std::mem::take(&mut self.reserved_output)
    }

    pub(super) fn poll(&mut self, system: &mut impl System) -> ShellPoll {
        if !self.started {
            self.started = true;
            match crate::commands::system::parse_env_action(system, &self.args) {
                Ok(action) if action.argv.is_empty() => {
                    let size = action
                        .environment
                        .iter()
                        .fold(0_u64, |total, (name, value)| {
                            total.saturating_add(name.len() as u64 + value.len() as u64 + 2)
                        });
                    if !system.reserve_memory(size) {
                        return ShellPoll::Ready(system.stop_status());
                    }
                    self.reserved_output = size;
                    for (name, value) in action.environment {
                        self.output.extend_from_slice(name.as_bytes());
                        self.output.push(b'=');
                        self.output.extend_from_slice(value.as_bytes());
                        self.output.push(b'\n');
                    }
                }
                Ok(action) => {
                    return match system.spawn_argv(SpawnSpec {
                        argv: action.argv,
                        stdin: None,
                        cwd: action.cwd,
                        environment: Some(action.environment),
                    }) {
                        Ok(pid) => {
                            self.waiting = Some(pid);
                            ShellPoll::Switched
                        }
                        Err(error) => {
                            self.status = 125;
                            self.output_fd = 2;
                            self.output =
                                format!("env: cannot spawn child: {error}\n").into_bytes();
                            ShellPoll::Pending
                        }
                    };
                }
                Err(error) => {
                    self.status = 125;
                    self.output_fd = 2;
                    self.output = format!("env: {error}\n").into_bytes();
                }
            }
        }
        if let Some(pid) = self.waiting {
            match system.child_status(pid) {
                Ok(None) => return ShellPoll::Blocked(WaitReason::Child(pid)),
                Ok(Some(_)) => match system.reap_child(pid) {
                    Ok(status) => {
                        self.status = status;
                        self.waiting = None;
                    }
                    Err(error) => {
                        self.status = 125;
                        self.waiting = None;
                        self.output_fd = 2;
                        self.output = format!("env: cannot reap child: {error}\n").into_bytes();
                    }
                },
                Err(error) => {
                    self.status = 125;
                    self.waiting = None;
                    self.output_fd = 2;
                    self.output = format!("env: cannot wait for child: {error}\n").into_bytes();
                }
            }
        }
        let outcome = poll_write(
            system,
            self.output_fd,
            &self.output,
            &mut self.output_offset,
            self.status,
        );
        if matches!(outcome, ShellPoll::Ready(_)) {
            system.release_memory(self.take_reserved_output());
        }
        outcome
    }
}
