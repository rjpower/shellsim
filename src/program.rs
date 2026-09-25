//! Runnable program images retained by the logical process scheduler.
//!
//! The shell is one image, not a privileged execution path. Native images call a scoped
//! process syscall interface directly; guest ABI adapters can implement the same operations
//! without granting either image access to host resources.

use crate::descriptors::{IoPoll, IoWait};
use crate::exec::{ShellContinuation, ShellPoll};
use crate::interp::Interp;
use crate::scheduler::WaitReason;
use crate::syscalls::{ActiveSystem, SyscallError, System};

mod cat;

/// Invocation view borrowed by a native command for one scheduler quantum.
/// Only the owned command continuation survives a blocked operation.
pub(crate) struct ProcessContext<'a> {
    pub(crate) system: &'a mut dyn System,
    pub(crate) command_name: &'a str,
    pub(crate) args: &'a [String],
}

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
            Self::Native(native) => native.poll(&mut ActiveSystem::new(interp)),
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
    SystemCommand(SystemCommandProcess),
    Cat(cat::CatProcess),
    Failure {
        status: i32,
        message: Vec<u8>,
        offset: usize,
    },
}

/// Adapter for existing bounded commands whose bodies already use only `System`.
/// Output is retained across descriptor backpressure; commands with live input use an owned
/// continuation instead.
#[derive(Clone)]
pub(crate) struct SystemCommandProcess {
    name: &'static str,
    args: Vec<String>,
    run: crate::commands::SystemCmdFn,
    result: Option<SystemCommandOutput>,
}

#[derive(Clone)]
struct SystemCommandOutput {
    status: i32,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    stdout_offset: usize,
    stderr_offset: usize,
}

impl SystemCommandProcess {
    fn new(name: &'static str, run: crate::commands::SystemCmdFn, args: &[String]) -> Self {
        Self {
            name,
            args: args.to_vec(),
            run,
            result: None,
        }
    }

    fn poll(&mut self, system: &mut impl System) -> ShellPoll {
        let result = self.result.get_or_insert_with(|| {
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            let argument_bytes = self
                .args
                .iter()
                .fold(0_u64, |total, arg| total.saturating_add(arg.len() as u64));
            let status = if !system.charge_cpu(100_u64.saturating_add(argument_bytes)) {
                system.stop_status()
            } else {
                let mut io = crate::commands::Io {
                    stdin: Vec::new(),
                    out: &mut stdout,
                    err: &mut stderr,
                };
                let mut context = ProcessContext {
                    system,
                    command_name: self.name,
                    args: &self.args,
                };
                (self.run)(&mut context, &mut io)
            };
            SystemCommandOutput {
                status,
                stdout,
                stderr,
                stdout_offset: 0,
                stderr_offset: 0,
            }
        });
        match poll_write(system, 1, &result.stdout, &mut result.stdout_offset, 0) {
            ShellPoll::Ready(0) => {}
            other => return other,
        }
        poll_write(
            system,
            2,
            &result.stderr,
            &mut result.stderr_offset,
            result.status,
        )
    }
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
            crate::vfs::NativeProgram::Cat => Self::Cat(cat::CatProcess::new(&argv[1..])),
            crate::vfs::NativeProgram::Registered(name) => {
                let Some(run) = crate::commands::system_command(name) else {
                    return Self::failure(
                        125,
                        "registered program needs a command continuation\n".into(),
                    );
                };
                Self::SystemCommand(SystemCommandProcess::new(name, run, &argv[1..]))
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

    fn poll(&mut self, syscalls: &mut impl System) -> ShellPoll {
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
            Self::SystemCommand(command) => command.poll(syscalls),
            Self::Cat(cat) => cat.poll(syscalls),
            Self::Failure {
                status,
                message,
                offset,
            } => poll_write(syscalls, 2, message, offset, *status),
        }
    }
}

fn poll_write(
    syscalls: &mut impl System,
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

    impl System for PartialWriter {
        fn cwd(&self) -> &str {
            "/work"
        }

        fn chdir(&mut self, _path: &str) -> Result<(), SyscallError> {
            unreachable!("native writer does not change directory")
        }

        fn umask(&self) -> u16 {
            unreachable!("native writer does not inspect umask")
        }

        fn set_umask(&mut self, _mask: u16) -> Result<(), SyscallError> {
            unreachable!("native writer does not change umask")
        }

        fn limits(&self) -> crate::resources::Limits {
            unreachable!("native writer does not inspect limits")
        }

        fn metadata(
            &mut self,
            _base: &str,
            _path: &str,
            _follow: bool,
        ) -> Result<crate::syscalls::FileInfo, SyscallError> {
            unreachable!("native writer does not inspect files")
        }

        fn metadata_fd(&mut self, _fd: i32) -> Result<crate::syscalls::FileInfo, SyscallError> {
            unreachable!("native writer does not inspect descriptors")
        }

        fn list_dir(&mut self, _base: &str, _path: &str) -> Result<Vec<String>, SyscallError> {
            unreachable!("native writer does not list directories")
        }

        fn walk(&mut self, _base: &str, _path: &str) -> Result<Vec<String>, SyscallError> {
            unreachable!("native writer does not walk directories")
        }

        fn read(&mut self, _fd: i32, _maximum: usize) -> Result<IoPoll<Vec<u8>>, SyscallError> {
            unreachable!("native writer does not read")
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

        fn open_file(
            &mut self,
            _base: &str,
            _path: &str,
            _options: crate::syscalls::OpenFile,
        ) -> Result<i32, SyscallError> {
            unreachable!("native writer does not open files")
        }

        fn file_state(&self, _fd: i32) -> Result<crate::descriptors::FileState, SyscallError> {
            unreachable!("native writer does not inspect files")
        }

        fn close(&mut self, _fd: i32) -> Result<(), SyscallError> {
            unreachable!("native writer does not close files")
        }

        fn seek(&mut self, _fd: i32, _delta: i64, _whence: u32) -> Result<u64, SyscallError> {
            unreachable!("native writer does not seek")
        }

        fn chmod(&mut self, _base: &str, _path: &str, _mode: u32) -> Result<(), SyscallError> {
            unreachable!("native writer does not change file modes")
        }

        fn unlink(&mut self, _base: &str, _path: &str) -> Result<(), SyscallError> {
            unreachable!("native writer does not unlink files")
        }

        fn mkdir(&mut self, _base: &str, _path: &str) -> Result<(), SyscallError> {
            unreachable!("native writer does not create directories")
        }

        fn mkdir_all(&mut self, _base: &str, _path: &str) -> Result<(), SyscallError> {
            unreachable!("native writer does not create directories")
        }

        fn rmdir(&mut self, _base: &str, _path: &str) -> Result<(), SyscallError> {
            unreachable!("native writer does not remove directories")
        }

        fn rename(&mut self, _base: &str, _from: &str, _to: &str) -> Result<(), SyscallError> {
            unreachable!("native writer does not rename files")
        }

        fn symlink(&mut self, _base: &str, _target: &str, _link: &str) -> Result<(), SyscallError> {
            unreachable!("native writer does not create links")
        }

        fn chown(
            &mut self,
            _base: &str,
            _path: &str,
            _uid: Option<u32>,
            _gid: Option<u32>,
        ) -> Result<(), SyscallError> {
            unreachable!("native writer does not change ownership")
        }

        fn touch(&mut self, _base: &str, _path: &str, _mtime_ms: u64) -> Result<(), SyscallError> {
            unreachable!("native writer does not touch files")
        }

        fn read_link(&mut self, _base: &str, _path: &str) -> Result<String, SyscallError> {
            unreachable!("native writer does not read links")
        }

        fn canonicalize(
            &mut self,
            _base: &str,
            _path: &str,
            _strict: bool,
        ) -> Result<String, SyscallError> {
            unreachable!("native writer does not resolve paths")
        }

        fn wall_time_ms(&self) -> u64 {
            unreachable!("native writer does not inspect time")
        }

        fn display_open(
            &mut self,
            _width: u32,
            _height: u32,
            _format: u32,
        ) -> Result<u32, crate::display::DisplayError> {
            unreachable!("native writer does not open displays")
        }

        fn display_present(
            &mut self,
            _handle: u32,
            _pixels: &[u8],
            _stride: u32,
        ) -> Result<(), crate::display::DisplayError> {
            unreachable!("native writer does not present displays")
        }

        fn input_poll_key(
            &mut self,
            _handle: u32,
        ) -> Result<Option<crate::display::KeyEvent>, crate::display::DisplayError> {
            unreachable!("native writer does not read keys")
        }

        fn display_close(&mut self, _handle: u32) -> Result<(), crate::display::DisplayError> {
            unreachable!("native writer does not close displays")
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
