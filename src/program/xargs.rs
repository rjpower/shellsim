//! Resumable xargs image that reads fd 0 and launches virtual argv children in sequence.

use std::collections::VecDeque;

use crate::descriptors::IoPoll;
use crate::exec::ShellPoll;
use crate::process::ProcessId;
use crate::scheduler::WaitReason;
use crate::syscalls::{SpawnSpec, System};

use super::{poll_write, wait_reason};

const MAX_INPUT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone)]
pub(crate) struct XargsProcess {
    args: Vec<String>,
    input: Vec<u8>,
    reserved_input: u64,
    commands: Option<VecDeque<Vec<String>>>,
    waiting: Option<ProcessId>,
    status: i32,
    diagnostic: Vec<u8>,
    diagnostic_offset: usize,
}

impl XargsProcess {
    pub(super) fn new(args: &[String]) -> Self {
        Self {
            args: args.to_vec(),
            input: Vec::new(),
            reserved_input: 0,
            commands: None,
            waiting: None,
            status: 0,
            diagnostic: Vec::new(),
            diagnostic_offset: 0,
        }
    }

    pub(super) fn take_reserved_input(&mut self) -> u64 {
        std::mem::take(&mut self.reserved_input)
    }

    pub(super) fn poll(&mut self, system: &mut impl System) -> ShellPoll {
        if !self.diagnostic.is_empty() {
            let outcome = poll_write(
                system,
                2,
                &self.diagnostic,
                &mut self.diagnostic_offset,
                self.status,
            );
            if matches!(outcome, ShellPoll::Ready(_)) {
                system.release_memory(self.take_reserved_input());
            }
            return outcome;
        }
        if self.commands.is_none() {
            match system.read(0, 4096) {
                Ok(IoPoll::Ready(bytes)) if bytes.is_empty() => {
                    let mut error = Vec::new();
                    match crate::commands::xargs::xargs_commands(
                        &self.args,
                        &self.input,
                        &mut error,
                    ) {
                        Ok(commands) => {
                            let argument_bytes = commands
                                .iter()
                                .flatten()
                                .fold(0_u64, |total, arg| total.saturating_add(arg.len() as u64));
                            if !system.reserve_memory(argument_bytes) {
                                system.release_memory(self.take_reserved_input());
                                return ShellPoll::Ready(system.stop_status());
                            }
                            self.reserved_input =
                                self.reserved_input.saturating_add(argument_bytes);
                            if !system
                                .charge_cpu(argument_bytes.saturating_add(self.input.len() as u64))
                            {
                                system.release_memory(self.take_reserved_input());
                                return ShellPoll::Ready(system.stop_status());
                            }
                            self.commands = Some(commands.into());
                        }
                        Err(status) => {
                            self.status = status;
                            self.diagnostic = error;
                        }
                    }
                    self.input.clear();
                    return ShellPoll::Pending;
                }
                Ok(IoPoll::Ready(bytes)) => {
                    if self.input.len().saturating_add(bytes.len()) > MAX_INPUT_BYTES {
                        self.status = 1;
                        self.diagnostic = b"xargs: input exceeds 4 MiB\n".to_vec();
                        return ShellPoll::Pending;
                    }
                    if !system.reserve_memory(bytes.len() as u64) {
                        return ShellPoll::Ready(system.stop_status());
                    }
                    self.reserved_input = self.reserved_input.saturating_add(bytes.len() as u64);
                    self.input.extend_from_slice(&bytes);
                    return ShellPoll::Pending;
                }
                Ok(IoPoll::Blocked(wait)) => return ShellPoll::Blocked(wait_reason(wait)),
                Err(error) => {
                    self.status = 1;
                    self.diagnostic = format!("xargs: cannot read input: {error}\n").into_bytes();
                    return ShellPoll::Pending;
                }
            }
        }
        if let Some(pid) = self.waiting {
            match system.child_status(pid) {
                Ok(None) => return ShellPoll::Blocked(WaitReason::Child(pid)),
                Ok(Some(status)) => {
                    if let Err(error) = system.reap_child(pid) {
                        self.status = 125;
                        self.diagnostic =
                            format!("xargs: cannot reap child: {error}\n").into_bytes();
                        return ShellPoll::Pending;
                    }
                    self.status = status;
                    self.waiting = None;
                }
                Err(error) => {
                    self.status = 125;
                    self.diagnostic =
                        format!("xargs: cannot wait for child: {error}\n").into_bytes();
                    return ShellPoll::Pending;
                }
            }
        }
        let Some(argv) = self.commands.as_mut().and_then(VecDeque::pop_front) else {
            system.release_memory(self.take_reserved_input());
            return ShellPoll::Ready(self.status);
        };
        match system.spawn_argv(SpawnSpec {
            argv,
            stdin: Some(Vec::new()),
            cwd: None,
            environment: None,
        }) {
            Ok(pid) => {
                self.waiting = Some(pid);
                ShellPoll::Switched
            }
            Err(error) => {
                self.status = 125;
                self.diagnostic = format!("xargs: cannot spawn child: {error}\n").into_bytes();
                ShellPoll::Pending
            }
        }
    }
}
