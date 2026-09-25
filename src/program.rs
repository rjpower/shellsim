//! Runnable program images retained by the logical process scheduler.
//!
//! The shell is one image, not a privileged execution path. Native images call a scoped
//! process syscall interface directly; guest ABI adapters can implement the same operations
//! without granting either image access to host resources.

use crate::descriptors::{IoPoll, IoWait};
use crate::exec::{ShellContinuation, ShellPoll};
use crate::interp::Interp;
use crate::scheduler::WaitReason;
use crate::syscalls::{ActiveProcessSyscalls, NativeSyscalls, SyscallError};

/// Owned state needed to resume one process after a scheduler turn.
#[derive(Clone)]
pub(crate) enum ProgramContinuation {
    Shell(ShellContinuation),
    Native(NativeProcess),
}

impl ProgramContinuation {
    pub(crate) fn is_native(&self) -> bool {
        matches!(self, Self::Native(_))
    }

    pub(crate) fn poll(&mut self, interp: &mut Interp, budget: usize) -> ShellPoll {
        match self {
            Self::Shell(shell) => shell.poll(interp, budget),
            Self::Native(native) => native.poll(&mut ActiveProcessSyscalls::new(interp)),
        }
    }

    pub(crate) fn inject_signal_handler(
        &mut self,
        interp: &mut Interp,
        body: crate::shell::Node,
    ) -> Result<(), String> {
        match self {
            Self::Shell(shell) => {
                shell.inject_signal_handler(interp, body);
                Ok(())
            }
            Self::Native(_) => Err("native process cannot run a shell signal handler".into()),
        }
    }
}

/// First native images migrated to the process loader. The enum keeps state cloneable for
/// deterministic machine snapshots; native bodies never retain a kernel reference.
#[derive(Clone)]
pub(crate) enum NativeProcess {
    Status(i32),
    Pwd {
        output: Option<Vec<u8>>,
        offset: usize,
    },
    Yes {
        output: Vec<u8>,
        offset: usize,
    },
    Failure {
        status: i32,
        message: Vec<u8>,
        offset: usize,
    },
}

impl NativeProcess {
    /// Construct an owned continuation from one VFS native program identity.
    pub(crate) fn from_image(image: crate::vfs::NativeProgram, argv: &[String]) -> Self {
        match image {
            crate::vfs::NativeProgram::True => Self::Status(0),
            crate::vfs::NativeProgram::False => Self::Status(1),
            crate::vfs::NativeProgram::Pwd => Self::Pwd {
                output: None,
                offset: 0,
            },
            crate::vfs::NativeProgram::Yes => {
                let args = &argv[1..];
                let line_bytes = args
                    .iter()
                    .fold(args.len().saturating_sub(1), |total, arg| {
                        total.saturating_add(arg.len())
                    })
                    .saturating_add(1);
                if line_bytes > 4096 {
                    return Self::failure(1, "yes: arguments exceed 4096 bytes\n".into());
                }
                let line = if args.is_empty() {
                    b"y\n".to_vec()
                } else {
                    format!("{}\n", args.join(" ")).into_bytes()
                };
                let output = line.repeat((4096 / line.len()).max(1));
                Self::Yes { output, offset: 0 }
            }
        }
    }

    pub(crate) fn failure(status: i32, message: String) -> Self {
        Self::Failure {
            status,
            message: message.into_bytes(),
            offset: 0,
        }
    }

    fn poll(&mut self, syscalls: &mut impl NativeSyscalls) -> ShellPoll {
        if !syscalls.charge_cpu(1) {
            return ShellPoll::Ready(syscalls.stop_status());
        }
        match self {
            Self::Status(status) => ShellPoll::Ready(*status),
            Self::Pwd { output, offset } => {
                let bytes = output.get_or_insert_with(|| {
                    let mut bytes = syscalls.cwd().as_bytes().to_vec();
                    bytes.push(b'\n');
                    bytes
                });
                poll_write(syscalls, 1, bytes, offset, 0)
            }
            Self::Yes { output, offset } => match poll_write(syscalls, 1, output, offset, 0) {
                ShellPoll::Ready(0) => {
                    *offset = 0;
                    ShellPoll::Pending
                }
                other => other,
            },
            Self::Failure {
                status,
                message,
                offset,
            } => poll_write(syscalls, 2, message, offset, *status),
        }
    }
}

fn poll_write(
    syscalls: &mut impl NativeSyscalls,
    fd: i32,
    bytes: &[u8],
    offset: &mut usize,
    status: i32,
) -> ShellPoll {
    let mut quantum = 4096_usize;
    while *offset < bytes.len() {
        if quantum == 0 {
            return ShellPoll::Pending;
        }
        let remaining = syscalls.output_remaining().min(quantum as u64) as usize;
        if remaining == 0 {
            let _ = syscalls.charge_output(1);
            return ShellPoll::Ready(syscalls.stop_status());
        }
        let end = offset.saturating_add(remaining).min(bytes.len());
        match syscalls.write(fd, &bytes[*offset..end]) {
            Ok(IoPoll::Ready(0)) => return ShellPoll::Ready(1),
            Ok(IoPoll::Ready(written)) => {
                *offset += written;
                quantum -= written;
                if !syscalls.charge_cpu(written as u64) || !syscalls.charge_output(written as u64) {
                    return ShellPoll::Ready(syscalls.stop_status());
                }
            }
            Ok(IoPoll::Blocked(wait)) => return ShellPoll::Blocked(wait_reason(wait)),
            Err(SyscallError::Descriptor(crate::descriptors::DescriptorError::BrokenPipe)) => {
                return ShellPoll::Ready(141);
            }
            Err(_) => return ShellPoll::Ready(1),
        }
    }
    ShellPoll::Ready(status)
}

fn wait_reason(wait: IoWait) -> WaitReason {
    match wait {
        IoWait::InputReadable(description) => WaitReason::InputReadable(description),
        IoWait::PipeReadable(pipe) => WaitReason::PipeReadable(pipe),
        IoWait::PipeWritable(pipe) => WaitReason::PipeWritable(pipe),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PartialWriter {
        calls: usize,
        bytes: Vec<u8>,
        remaining: u64,
        stopped: bool,
    }

    impl NativeSyscalls for PartialWriter {
        fn cwd(&self) -> &str {
            "/work"
        }

        fn write(&mut self, fd: i32, bytes: &[u8]) -> Result<IoPoll<usize>, SyscallError> {
            assert_eq!(fd, 1);
            self.calls += 1;
            if self.calls == 1 {
                return Ok(IoPoll::Blocked(IoWait::PipeWritable(7)));
            }
            let written = bytes.len().min(2);
            self.bytes.extend_from_slice(&bytes[..written]);
            Ok(IoPoll::Ready(written))
        }

        fn charge_cpu(&mut self, _units: u64) -> bool {
            true
        }

        fn output_remaining(&self) -> u64 {
            self.remaining
        }

        fn charge_output(&mut self, bytes: u64) -> bool {
            if bytes > self.remaining {
                self.stopped = true;
                return false;
            }
            self.remaining -= bytes;
            true
        }

        fn stop_status(&self) -> i32 {
            137
        }
    }

    #[test]
    fn native_process_retains_output_across_blocked_and_partial_writes() {
        let mut process =
            NativeProcess::from_image(crate::vfs::NativeProgram::Pwd, &["pwd".into()]);
        let mut syscalls = PartialWriter {
            calls: 0,
            bytes: Vec::new(),
            remaining: 1_000,
            stopped: false,
        };
        assert_eq!(
            process.poll(&mut syscalls),
            ShellPoll::Blocked(WaitReason::PipeWritable(7))
        );
        assert_eq!(process.poll(&mut syscalls), ShellPoll::Ready(0));
        assert_eq!(syscalls.bytes, b"/work\n");
    }

    #[test]
    fn native_process_stops_at_the_output_limit() {
        let mut process =
            NativeProcess::from_image(crate::vfs::NativeProgram::Pwd, &["pwd".into()]);
        let mut syscalls = PartialWriter {
            calls: 1,
            bytes: Vec::new(),
            remaining: 3,
            stopped: false,
        };
        assert_eq!(process.poll(&mut syscalls), ShellPoll::Ready(137));
        assert_eq!(syscalls.bytes, b"/wo");
        assert!(syscalls.stopped);
    }
}
