//! Resumable `tee` image over process descriptors.
//!
//! Input is copied one bounded chunk at a time. The continuation retains only that chunk and
//! output cursors, so an infinite upstream and an early-closing downstream cannot deadlock.

use crate::descriptors::IoPoll;
use crate::exec::ShellPoll;
use crate::syscalls::{OpenFile, System};

use super::{poll_write, wait_reason};

#[derive(Clone)]
pub(crate) struct TeeProcess {
    paths: Vec<String>,
    append: bool,
    files: Vec<(String, i32)>,
    opened: bool,
    pending: Vec<u8>,
    sink: usize,
    offset: usize,
    diagnostic: Vec<u8>,
    diagnostic_offset: usize,
    status: i32,
    invalid: bool,
}

impl TeeProcess {
    pub(super) fn new(args: &[String]) -> Self {
        let mut paths = Vec::new();
        let mut append = false;
        let mut diagnostic = Vec::new();
        let mut options = true;
        for argument in args {
            if options && argument == "--" {
                options = false;
            } else if options && matches!(argument.as_str(), "-a" | "--append") {
                append = true;
            } else if options && argument.starts_with('-') && argument != "-" {
                diagnostic = format!("tee: unsupported option '{argument}'\n").into_bytes();
                break;
            } else {
                paths.push(argument.clone());
            }
        }
        let invalid = !diagnostic.is_empty();
        Self {
            paths,
            append,
            files: Vec::new(),
            opened: false,
            pending: Vec::new(),
            sink: 0,
            offset: 0,
            diagnostic,
            diagnostic_offset: 0,
            status: if invalid { 2 } else { 0 },
            invalid,
        }
    }

    pub(super) fn poll(&mut self, system: &mut impl System) -> ShellPoll {
        if !self.opened {
            self.opened = true;
            if !self.invalid {
                let cwd = system.cwd().to_string();
                for path in &self.paths {
                    match system.open_file(
                        &cwd,
                        path,
                        OpenFile {
                            readable: false,
                            writable: true,
                            create: true,
                            exclusive: false,
                            truncate: !self.append,
                            append: self.append,
                        },
                    ) {
                        Ok(fd) => self.files.push((path.clone(), fd)),
                        Err(error) => {
                            self.diagnostic
                                .extend_from_slice(format!("tee: {path}: {error}\n").as_bytes());
                            self.status = 1;
                        }
                    }
                }
            }
            return ShellPoll::Pending;
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
        if !self.pending.is_empty() {
            if self.sink < self.files.len() {
                let fd = self.files[self.sink].1;
                return match system.write(fd, &self.pending[self.offset..]) {
                    Ok(IoPoll::Ready(0)) => ShellPoll::Ready(1),
                    Ok(IoPoll::Ready(written)) => {
                        if !system.charge_cpu(written as u64) {
                            return ShellPoll::Ready(system.stop_status());
                        }
                        self.offset += written;
                        if self.offset == self.pending.len() {
                            self.sink += 1;
                            self.offset = 0;
                        }
                        ShellPoll::Pending
                    }
                    Ok(IoPoll::Blocked(wait)) => ShellPoll::Blocked(wait_reason(wait)),
                    Err(error) => {
                        self.diagnostic.extend_from_slice(
                            format!("tee: {}: {error}\n", self.files[self.sink].0).as_bytes(),
                        );
                        self.status = 1;
                        let _ = system.close(fd);
                        self.files.remove(self.sink);
                        self.offset = 0;
                        ShellPoll::Pending
                    }
                };
            }
            return match poll_write(system, 1, &self.pending, &mut self.offset, 0) {
                ShellPoll::Ready(0) => {
                    self.pending.clear();
                    self.sink = 0;
                    self.offset = 0;
                    ShellPoll::Pending
                }
                other => other,
            };
        }
        match system.read(0, 4096) {
            Ok(IoPoll::Ready(bytes)) if bytes.is_empty() => {
                for (_, fd) in self.files.drain(..) {
                    if system.close(fd).is_err() {
                        self.status = 1;
                    }
                }
                ShellPoll::Ready(self.status)
            }
            Ok(IoPoll::Ready(bytes)) => {
                if !system.charge_cpu(bytes.len() as u64) {
                    return ShellPoll::Ready(system.stop_status());
                }
                self.pending = bytes;
                ShellPoll::Pending
            }
            Ok(IoPoll::Blocked(wait)) => ShellPoll::Blocked(wait_reason(wait)),
            Err(error) => {
                self.diagnostic =
                    format!("tee: cannot read standard input: {error}\n").into_bytes();
                self.status = 1;
                self.invalid = true;
                ShellPoll::Pending
            }
        }
    }
}
