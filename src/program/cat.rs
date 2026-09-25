//! Bounded, resumable `cat` image using only the active process's descriptors.
//!
//! The continuation retains source names and a small pending output chunk, never a kernel
//! handle. Each poll does one open, read, or write step so pipe readiness can reschedule it.

use std::collections::VecDeque;

use crate::descriptors::IoPoll;
use crate::exec::ShellPoll;
use crate::syscalls::{OpenFile, System};

use super::{poll_write, wait_reason};

#[derive(Clone)]
pub(crate) struct CatProcess {
    sources: VecDeque<String>,
    current: Option<(i32, bool)>,
    numbered: bool,
    next_line: u64,
    at_line_start: bool,
    pending: Vec<u8>,
    offset: usize,
    diagnostic: Vec<u8>,
    diagnostic_offset: usize,
    status: i32,
    stop_after_diagnostic: bool,
}

impl CatProcess {
    pub(super) fn new(args: &[String]) -> Self {
        let mut sources = VecDeque::new();
        let mut numbered = false;
        let mut options = true;
        let mut diagnostic = Vec::new();
        for arg in args {
            if options && arg == "--" {
                options = false;
            } else if options && arg.starts_with("--") {
                diagnostic = format!("cat: unimplemented option '{arg}'\n").into_bytes();
                break;
            } else if options && arg.starts_with('-') && arg != "-" {
                for flag in arg[1..].chars() {
                    if flag == 'n' {
                        numbered = true;
                    } else {
                        diagnostic = format!("cat: unimplemented option '-{flag}'\n").into_bytes();
                        break;
                    }
                }
                if !diagnostic.is_empty() {
                    break;
                }
            } else {
                sources.push_back(arg.clone());
            }
        }
        if sources.is_empty() {
            sources.push_back("-".into());
        }
        let stop_after_diagnostic = !diagnostic.is_empty();
        Self {
            sources,
            current: None,
            numbered,
            next_line: 1,
            at_line_start: true,
            pending: Vec::new(),
            offset: 0,
            diagnostic,
            diagnostic_offset: 0,
            status: if stop_after_diagnostic { 2 } else { 0 },
            stop_after_diagnostic,
        }
    }

    pub(super) fn poll(&mut self, system: &mut impl System) -> ShellPoll {
        if !self.pending.is_empty() {
            return match poll_write(system, 1, &self.pending, &mut self.offset, 0) {
                ShellPoll::Ready(0) => {
                    self.pending.clear();
                    self.offset = 0;
                    ShellPoll::Pending
                }
                other => other,
            };
        }
        if !self.diagnostic.is_empty() {
            let done_status = if self.stop_after_diagnostic {
                self.status
            } else {
                0
            };
            return match poll_write(
                system,
                2,
                &self.diagnostic,
                &mut self.diagnostic_offset,
                done_status,
            ) {
                ShellPoll::Ready(0) if !self.stop_after_diagnostic => {
                    self.diagnostic.clear();
                    self.diagnostic_offset = 0;
                    ShellPoll::Pending
                }
                other => other,
            };
        }
        if self.current.is_none() {
            let Some(source) = self.sources.pop_front() else {
                return ShellPoll::Ready(self.status);
            };
            if source == "-" || source == "/dev/stdin" {
                self.current = Some((0, false));
            } else {
                let base = system.cwd().to_string();
                match system.open_file(
                    &base,
                    &source,
                    OpenFile {
                        readable: true,
                        writable: false,
                        create: false,
                        exclusive: false,
                        truncate: false,
                        append: false,
                    },
                ) {
                    Ok(fd) => self.current = Some((fd, true)),
                    Err(error) => {
                        self.diagnostic = format!("cat: {source}: {error}\n").into_bytes();
                        self.status = 1;
                    }
                }
            }
            return ShellPoll::Pending;
        }
        let (fd, owned) = self.current.expect("source was selected");
        let maximum = if self.numbered { 512 } else { 4096 };
        match system.read(fd, maximum) {
            Ok(IoPoll::Ready(bytes)) if bytes.is_empty() => {
                if owned && system.close(fd).is_err() {
                    self.status = 1;
                }
                self.current = None;
                ShellPoll::Pending
            }
            Ok(IoPoll::Ready(bytes)) => {
                if !system.charge_cpu(bytes.len() as u64) {
                    return ShellPoll::Ready(system.stop_status());
                }
                if self.numbered {
                    for byte in bytes {
                        if self.at_line_start {
                            self.pending
                                .extend_from_slice(format!("{:6}\t", self.next_line).as_bytes());
                            self.next_line = self.next_line.saturating_add(1);
                        }
                        self.pending.push(byte);
                        self.at_line_start = byte == b'\n';
                    }
                } else {
                    self.pending = bytes;
                }
                ShellPoll::Pending
            }
            Ok(IoPoll::Blocked(wait)) => ShellPoll::Blocked(wait_reason(wait)),
            Err(error) => {
                if owned {
                    let _ = system.close(fd);
                }
                self.current = None;
                self.diagnostic = format!("cat: {error}\n").into_bytes();
                self.status = 1;
                ShellPoll::Pending
            }
        }
    }
}
