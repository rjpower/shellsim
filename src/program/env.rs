//! Resumable `env` image: print the environment, or exec a command with a modified one.
//!
//! Like `execvp`, running a command replaces this process, so the command keeps env's PID,
//! descriptors, and ignored signals, and `-C` and `KEY=value` apply only to it.

use crate::exec::ShellPoll;
use crate::syscalls::System;

use super::poll_write;

/// Status when env cannot run its command, as GNU env reports.
const ENV_FAILED: i32 = 125;

#[derive(Clone)]
pub(crate) struct EnvProcess {
    args: Vec<String>,
    started: bool,
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
                    let result = match &action.cwd {
                        Some(cwd) => system.chdir(cwd),
                        None => Ok(()),
                    }
                    .and_then(|()| system.exec_argv(action.argv, Some(action.environment)));
                    match result {
                        Ok(()) => return ShellPoll::Replaced,
                        Err(error) => self.fail(format!("env: cannot run command: {error}")),
                    }
                }
                Err(error) => self.fail(format!("env: {error}")),
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

    fn fail(&mut self, message: String) {
        self.status = ENV_FAILED;
        self.output_fd = 2;
        self.output = format!("{message}\n").into_bytes();
    }
}
