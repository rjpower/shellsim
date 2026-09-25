//! Resumable `head` image using the process descriptor table.
//!
//! It stops reading as soon as its line or byte limit is reached, so an infinite producer can
//! finish when the consumer exits. File operands use the same open/read calls as Wasm guests.

use std::collections::VecDeque;

use crate::commands::streams::parse_head_options;
use crate::descriptors::IoPoll;
use crate::exec::ShellPoll;
use crate::syscalls::{OpenFile, System};

use super::{poll_write, wait_reason};

#[derive(Clone)]
pub(crate) struct HeadProcess {
    sources: VecDeque<String>,
    multiple: bool,
    current: Option<(i32, bool)>,
    byte_limit: Option<usize>,
    line_limit: usize,
    bytes_remaining: Option<usize>,
    lines_remaining: usize,
    finished_source: bool,
    emitted: bool,
    pending: Vec<u8>,
    offset: usize,
    diagnostic: Vec<u8>,
    diagnostic_offset: usize,
    status: i32,
    invalid: bool,
}

impl HeadProcess {
    pub(super) fn new(args: &[String]) -> Self {
        let parsed = parse_head_options(args);
        let (sources, byte_limit, line_limit, diagnostic, invalid) = match parsed {
            Ok(options) => {
                let sources = if options.files.is_empty() {
                    VecDeque::from(["-".to_string()])
                } else {
                    options.files.into()
                };
                (sources, options.bytes, options.lines, Vec::new(), false)
            }
            Err(error) => (
                VecDeque::new(),
                None,
                0,
                format!("head: {error}\n").into_bytes(),
                true,
            ),
        };
        let multiple = sources.len() > 1;
        Self {
            sources,
            multiple,
            current: None,
            byte_limit,
            line_limit,
            bytes_remaining: byte_limit,
            lines_remaining: line_limit,
            finished_source: false,
            emitted: false,
            pending: Vec::new(),
            offset: 0,
            diagnostic,
            diagnostic_offset: 0,
            status: if invalid { 1 } else { 0 },
            invalid,
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
            return match poll_write(system, 2, &self.diagnostic, &mut self.diagnostic_offset, 0) {
                ShellPoll::Ready(0) => {
                    self.diagnostic.clear();
                    self.diagnostic_offset = 0;
                    if self.invalid {
                        ShellPoll::Ready(self.status)
                    } else {
                        ShellPoll::Pending
                    }
                }
                other => other,
            };
        }
        if self.finished_source {
            if let Some((fd, owned)) = self.current.take() {
                if owned && system.close(fd).is_err() {
                    self.status = 1;
                }
            }
            self.finished_source = false;
            return ShellPoll::Pending;
        }
        if self.current.is_none() {
            let Some(source) = self.sources.pop_front() else {
                return ShellPoll::Ready(self.status);
            };
            let current = if source == "-" || source == "/dev/stdin" {
                Some((0, false))
            } else {
                let cwd = system.cwd().to_string();
                match system.open_file(
                    &cwd,
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
                    Ok(fd) => Some((fd, true)),
                    Err(error) => {
                        self.diagnostic = format!("head: {source}: {error}\n").into_bytes();
                        self.status = 1;
                        None
                    }
                }
            };
            if let Some(current) = current {
                self.current = Some(current);
                self.bytes_remaining = self.byte_limit;
                self.lines_remaining = self.line_limit;
                if self.multiple {
                    if self.emitted {
                        self.pending.push(b'\n');
                    }
                    self.pending
                        .extend_from_slice(format!("==> {source} <==\n").as_bytes());
                }
                self.emitted = true;
            }
            return ShellPoll::Pending;
        }
        if self.bytes_remaining == Some(0)
            || (self.bytes_remaining.is_none() && self.lines_remaining == 0)
        {
            self.finished_source = true;
            return ShellPoll::Pending;
        }
        let (fd, _) = self.current.expect("selected source");
        let maximum = self.bytes_remaining.unwrap_or(4096).min(4096);
        match system.read(fd, maximum) {
            Ok(IoPoll::Ready(bytes)) if bytes.is_empty() => {
                self.finished_source = true;
                ShellPoll::Pending
            }
            Ok(IoPoll::Ready(bytes)) => {
                if !system.charge_cpu(bytes.len() as u64) {
                    return ShellPoll::Ready(system.stop_status());
                }
                let take = if let Some(remaining) = &mut self.bytes_remaining {
                    let take = (*remaining).min(bytes.len());
                    *remaining -= take;
                    take
                } else {
                    let mut take = bytes.len();
                    for (index, byte) in bytes.iter().enumerate() {
                        if *byte == b'\n' {
                            self.lines_remaining -= 1;
                            if self.lines_remaining == 0 {
                                take = index + 1;
                                break;
                            }
                        }
                    }
                    take
                };
                self.pending.extend_from_slice(&bytes[..take]);
                if self.bytes_remaining == Some(0)
                    || (self.bytes_remaining.is_none() && self.lines_remaining == 0)
                {
                    self.finished_source = true;
                }
                ShellPoll::Pending
            }
            Ok(IoPoll::Blocked(wait)) => ShellPoll::Blocked(wait_reason(wait)),
            Err(error) => {
                self.diagnostic = format!("head: {error}\n").into_bytes();
                self.status = 1;
                self.finished_source = true;
                ShellPoll::Pending
            }
        }
    }
}
