//! `nohup` image: ignore SIGHUP, then exec the command in the same process.
//!
//! Shellsim descriptors are never terminals, so, as GNU nohup does for non-terminal output,
//! the command's standard streams are left unchanged and no `nohup.out` is created.

use crate::exec::ShellPoll;
use crate::process::Signal;
use crate::syscalls::System;

use super::poll_write;

/// GNU nohup's status when it cannot run the command.
const NOHUP_FAILED: i32 = 125;

#[derive(Clone)]
pub(crate) struct NohupProcess {
    argv: Vec<String>,
    diagnostic: Vec<u8>,
    diagnostic_offset: usize,
}

impl NohupProcess {
    pub(super) fn new(args: &[String]) -> Self {
        let argv = match args.first().map(String::as_str) {
            Some("--") => args[1..].to_vec(),
            _ => args.to_vec(),
        };
        Self {
            argv,
            diagnostic: Vec::new(),
            diagnostic_offset: 0,
        }
    }

    pub(super) fn poll(&mut self, system: &mut impl System) -> ShellPoll {
        if !self.diagnostic.is_empty() {
            return poll_write(
                system,
                2,
                &self.diagnostic,
                &mut self.diagnostic_offset,
                NOHUP_FAILED,
            );
        }
        if self.argv.is_empty() {
            self.diagnostic = b"nohup: missing operand\n".to_vec();
            return ShellPoll::Pending;
        }
        if let Some(option) = self.argv.first().filter(|arg| arg.starts_with('-')) {
            self.diagnostic = format!("nohup: unrecognized option '{option}'\n").into_bytes();
            return ShellPoll::Pending;
        }
        let result = system
            .ignore_signal(Signal::Hangup)
            .and_then(|()| system.exec_argv(std::mem::take(&mut self.argv), None));
        match result {
            Ok(()) => ShellPoll::Replaced,
            Err(error) => {
                self.diagnostic = format!("nohup: cannot run command: {error}\n").into_bytes();
                ShellPoll::Pending
            }
        }
    }
}
