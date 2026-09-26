//! `nice` image: validate a niceness adjustment, then exec the command in the same process.
//!
//! The scheduler is a deterministic FIFO over runnable processes and has no priorities, so the
//! adjustment is accepted but has no scheduling effect. Every process reports niceness 0.

use crate::exec::ShellPoll;
use crate::syscalls::System;

use super::poll_write;

/// GNU nice's status when it cannot run the command or rejects its own options.
const NICE_FAILED: i32 = 125;

#[derive(Clone)]
pub(crate) struct NiceProcess {
    args: Vec<String>,
    output: Vec<u8>,
    offset: usize,
    fd: i32,
    status: i32,
    started: bool,
}

impl NiceProcess {
    pub(super) fn new(args: &[String]) -> Self {
        Self {
            args: args.to_vec(),
            output: Vec::new(),
            offset: 0,
            fd: 1,
            status: 0,
            started: false,
        }
    }

    pub(super) fn poll(&mut self, system: &mut impl System) -> ShellPoll {
        if !self.started {
            self.started = true;
            match parse_nice(&self.args) {
                Ok(argv) if argv.is_empty() => self.output = b"0\n".to_vec(),
                Ok(argv) => match system.exec_argv(argv, None) {
                    Ok(()) => return ShellPoll::Replaced,
                    Err(error) => self.fail(format!("nice: cannot run command: {error}")),
                },
                Err(error) => self.fail(format!("nice: {error}")),
            }
        }
        poll_write(system, self.fd, &self.output, &mut self.offset, self.status)
    }

    fn fail(&mut self, message: String) {
        self.status = NICE_FAILED;
        self.fd = 2;
        self.output = format!("{message}\n").into_bytes();
    }
}

/// Accept `-n N`, `-nN`, `--adjustment=N`, and the historical `-N`, returning the command argv.
fn parse_nice(args: &[String]) -> Result<Vec<String>, String> {
    let mut index = 0;
    let adjustment = match args.first().map(String::as_str) {
        Some("-n" | "--adjustment") => {
            index = 2;
            args.get(1)
                .map(String::as_str)
                .ok_or_else(|| "option requires an argument -- 'n'".to_string())?
        }
        Some(option) if option.starts_with("--adjustment=") => {
            index = 1;
            &option["--adjustment=".len()..]
        }
        Some(option) if option.starts_with("-n") => {
            index = 1;
            &option[2..]
        }
        Some(option) if option.len() > 1 && option.starts_with('-') && option != "--" => {
            index = 1;
            &option[1..]
        }
        _ => "0",
    };
    adjustment
        .parse::<i64>()
        .map_err(|_| format!("invalid adjustment '{adjustment}'"))?;
    if args.get(index).map(String::as_str) == Some("--") {
        index += 1;
    }
    Ok(args[index..].to_vec())
}
