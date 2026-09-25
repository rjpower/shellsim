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
mod env;
mod head;
mod sleep;
mod tee;
mod xargs;

/// Invocation view borrowed by a native command for one scheduler quantum.
/// Only the owned command continuation survives a blocked operation.
pub(crate) struct ProcessContext<'a> {
    pub(crate) system: &'a mut dyn System,
    pub(crate) command_name: &'a str,
    pub(crate) args: &'a [String],
    pub(crate) environment: &'a std::collections::BTreeMap<String, String>,
    pub(crate) stdin_state: Option<(&'a mut bool, &'a mut u64)>,
}

impl ProcessContext<'_> {
    /// Read fd 0 to EOF in bounded quanta, retaining partial input across scheduler turns.
    /// Nested synchronous dispatch has already materialized `Io::stdin`.
    pub(crate) fn read_standard_input(
        &mut self,
        io: &mut crate::commands::Io<'_>,
    ) -> Result<(), ShellPoll> {
        let Some((complete, reserved)) = self.stdin_state.as_mut() else {
            return Ok(());
        };
        if **complete {
            return Ok(());
        }
        match self.system.read(0, 4096) {
            Ok(IoPoll::Ready(bytes)) if bytes.is_empty() => {
                **complete = true;
                Ok(())
            }
            Ok(IoPoll::Ready(bytes)) => {
                let size = bytes.len() as u64;
                if !self.system.reserve_memory(size) {
                    return Err(ShellPoll::Ready(self.system.stop_status()));
                }
                **reserved = reserved.saturating_add(size);
                io.stdin.extend_from_slice(&bytes);
                Err(ShellPoll::Pending)
            }
            Ok(IoPoll::Blocked(wait)) => Err(ShellPoll::Blocked(wait_reason(wait))),
            Err(error) => {
                io.print_err(&format!(
                    "{}: cannot read standard input: {error}\n",
                    self.command_name
                ));
                Err(ShellPoll::Ready(1))
            }
        }
    }
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

    /// Return outstanding input reservations when a process exits before consuming EOF.
    pub(crate) fn release_owned_memory(&mut self, interp: &mut Interp) {
        if let Self::Native(NativeProcess::SystemCommand(command)) = self {
            interp.resources.release_memory(command.reserved_input);
            command.reserved_input = 0;
            interp.resources.release_memory(command.base_reserved);
            command.base_reserved = 0;
        }
        if let Self::Native(NativeProcess::Xargs(command)) = self {
            interp
                .resources
                .release_memory(command.take_reserved_input());
        }
        if let Self::Native(NativeProcess::Env(command)) = self {
            interp
                .resources
                .release_memory(command.take_reserved_output());
        }
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
    Tee(tee::TeeProcess),
    Head(head::HeadProcess),
    Xargs(xargs::XargsProcess),
    Env(env::EnvProcess),
    Sleep(sleep::SleepProcess),
    Failure {
        status: i32,
        message: Vec<u8>,
        offset: usize,
    },
}

/// Adapter for bounded commands whose bodies use only `System`.
/// Input and output remain owned across descriptor backpressure.
#[derive(Clone)]
pub(crate) struct SystemCommandProcess {
    name: &'static str,
    args: Vec<String>,
    run: crate::commands::SystemRun,
    base_cpu: u64,
    base_memory: u64,
    trust: crate::telemetry::CommandTrust,
    started: bool,
    base_reserved: u64,
    cpu_before: u64,
    disk_before: u64,
    usage_recorded: bool,
    stdin: Vec<u8>,
    stdin_complete: bool,
    reserved_input: u64,
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
    fn new(name: &'static str, command: crate::commands::SystemCommand, args: &[String]) -> Self {
        Self {
            name,
            args: args.to_vec(),
            run: command.run,
            base_cpu: command.base_cpu,
            base_memory: command.base_memory,
            trust: command.trust,
            started: false,
            base_reserved: 0,
            cpu_before: 0,
            disk_before: 0,
            usage_recorded: false,
            stdin: Vec::new(),
            stdin_complete: false,
            reserved_input: 0,
            result: None,
        }
    }

    fn poll(&mut self, system: &mut impl System) -> ShellPoll {
        if self.result.is_none() {
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            let argument_bytes = self
                .args
                .iter()
                .fold(0_u64, |total, arg| total.saturating_add(arg.len() as u64));
            let start_status = if self.started {
                None
            } else {
                self.started = true;
                self.cpu_before = system.cpu_used();
                self.disk_before = system.disk_used();
                if !system.reserve_memory(self.base_memory) {
                    Some(system.stop_status())
                } else {
                    self.base_reserved = self.base_memory;
                    (!system.charge_cpu(self.base_cpu.saturating_add(argument_bytes)))
                        .then(|| system.stop_status())
                }
            };
            let outcome = if let Some(status) = start_status {
                ShellPoll::Ready(status)
            } else {
                let environment = system.environment();
                let mut io = crate::commands::Io {
                    stdin: std::mem::take(&mut self.stdin),
                    out: &mut stdout,
                    err: &mut stderr,
                };
                let mut context = ProcessContext {
                    system,
                    command_name: self.name,
                    args: &self.args,
                    environment: &environment,
                    stdin_state: Some((&mut self.stdin_complete, &mut self.reserved_input)),
                };
                let outcome = match self.run {
                    crate::commands::SystemRun::Once(run) => {
                        ShellPoll::Ready(run(&mut context, &mut io))
                    }
                    crate::commands::SystemRun::Poll(run) => run(&mut context, &mut io),
                };
                if !matches!(outcome, ShellPoll::Ready(_)) {
                    self.stdin = io.stdin;
                }
                outcome
            };
            let status = match outcome {
                ShellPoll::Ready(status) => status,
                other => {
                    if !stdout.is_empty() || !stderr.is_empty() {
                        self.result = Some(SystemCommandOutput {
                            status: 125,
                            stdout: Vec::new(),
                            stderr: format!(
                                "{}: command suspended after producing output\n",
                                self.name
                            )
                            .into_bytes(),
                            stdout_offset: 0,
                            stderr_offset: 0,
                        });
                    } else {
                        return other;
                    }
                    125
                }
            };
            system.release_memory(self.reserved_input);
            self.reserved_input = 0;
            system.release_memory(self.base_reserved);
            self.base_reserved = 0;
            if self.result.is_none() {
                self.result = Some(SystemCommandOutput {
                    status,
                    stdout,
                    stderr,
                    stdout_offset: 0,
                    stderr_offset: 0,
                });
            }
        }
        let result = self.result.as_mut().expect("command result was produced");
        match poll_write(system, 1, &result.stdout, &mut result.stdout_offset, 0) {
            ShellPoll::Ready(0) => {}
            other => return other,
        }
        let completion = poll_write(
            system,
            2,
            &result.stderr,
            &mut result.stderr_offset,
            result.status,
        );
        if matches!(completion, ShellPoll::Ready(_)) && !self.usage_recorded {
            system.record_command_usage(self.name, self.cpu_before, self.disk_before);
            self.usage_recorded = true;
        }
        completion
    }
}

impl NativeProcess {
    pub(crate) fn trust(&self) -> crate::telemetry::CommandTrust {
        match self {
            Self::SystemCommand(command) => command.trust,
            _ => crate::telemetry::CommandTrust::Real,
        }
    }

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
            crate::vfs::NativeProgram::Tee => Self::Tee(tee::TeeProcess::new(&argv[1..])),
            crate::vfs::NativeProgram::Head => Self::Head(head::HeadProcess::new(&argv[1..])),
            crate::vfs::NativeProgram::Xargs => Self::Xargs(xargs::XargsProcess::new(&argv[1..])),
            crate::vfs::NativeProgram::Env => Self::Env(env::EnvProcess::new(&argv[1..])),
            crate::vfs::NativeProgram::Sleep => Self::Sleep(sleep::SleepProcess::sleep(&argv[1..])),
            crate::vfs::NativeProgram::Usleep => {
                Self::Sleep(sleep::SleepProcess::usleep(&argv[1..]))
            }
            crate::vfs::NativeProgram::Registered(path) => {
                let Some(command) = crate::commands::system_command(path) else {
                    return Self::failure(
                        125,
                        "registered program needs a command continuation\n".into(),
                    );
                };
                let name = path.rsplit('/').next().unwrap_or(path);
                Self::SystemCommand(SystemCommandProcess::new(name, command, &argv[1..]))
            }
            crate::vfs::NativeProgram::LegacyRegistered(_) => {
                Self::failure(125, "legacy command has no native continuation\n".into())
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
            Self::Tee(tee) => tee.poll(syscalls),
            Self::Head(head) => head.poll(syscalls),
            Self::Xargs(xargs) => xargs.poll(syscalls),
            Self::Env(env) => env.poll(syscalls),
            Self::Sleep(sleep) => sleep.poll(syscalls),
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
        fn environment(&self) -> std::collections::BTreeMap<String, String> {
            std::collections::BTreeMap::new()
        }

        fn hostname(&self) -> &str {
            unreachable!("native writer does not inspect hostname")
        }

        fn set_hostname(&mut self, _name: &str) -> Result<(), SyscallError> {
            unreachable!("native writer does not set hostname")
        }

        fn uid(&self) -> u32 {
            unreachable!("native writer does not inspect identity")
        }

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

        fn cpu_used(&self) -> u64 {
            0
        }

        fn disk_used(&self) -> u64 {
            unreachable!("native writer does not inspect disk usage")
        }

        fn memory_used(&self) -> u64 {
            unreachable!("native writer does not inspect memory usage")
        }

        fn record_command_usage(&mut self, _name: &str, _cpu_before: u64, _disk_before: u64) {}

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

        fn read_file_limited(
            &mut self,
            _base: &str,
            _path: &str,
            _maximum: usize,
        ) -> Result<Vec<u8>, SyscallError> {
            unreachable!("native writer does not read files")
        }

        fn write_file(
            &mut self,
            _base: &str,
            _path: &str,
            _bytes: &[u8],
            _mode: u32,
        ) -> Result<(), SyscallError> {
            unreachable!("native writer does not write files")
        }

        fn put_file_with_parents(
            &mut self,
            _base: &str,
            _path: &str,
            _bytes: Vec<u8>,
            _mode: u32,
        ) -> Result<(), SyscallError> {
            unreachable!("native writer does not write files")
        }

        fn apply_file_batch(
            &mut self,
            _base: &str,
            _changes: Vec<crate::syscalls::FileChange>,
        ) -> Result<(), SyscallError> {
            unreachable!("native writer does not change files")
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

        fn open_file_at(
            &mut self,
            _fd: i32,
            _base: &str,
            _path: &str,
            _options: crate::syscalls::OpenFile,
        ) -> Result<(), SyscallError> {
            unreachable!("native writer does not open files")
        }

        fn duplicate(&mut self, _source: i32, _destination: i32) -> Result<(), SyscallError> {
            unreachable!("native writer does not duplicate descriptors")
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

        fn copy_file(&mut self, _base: &str, _from: &str, _to: &str) -> Result<(), SyscallError> {
            unreachable!("native writer does not copy files")
        }

        fn copy_recursive(
            &mut self,
            _base: &str,
            _from: &str,
            _to: &str,
            _preserve: bool,
        ) -> Result<(), SyscallError> {
            unreachable!("native writer does not copy trees")
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

        fn wall_time_signed_ns(&self) -> Result<i128, SyscallError> {
            unreachable!("native writer does not inspect time")
        }

        fn clock_time_ns(&self, _clock: crate::syscalls::ClockId) -> Result<u64, SyscallError> {
            unreachable!("native writer does not inspect time")
        }

        fn random_fill(&mut self, _bytes: &mut [u8]) -> Result<(), SyscallError> {
            unreachable!("native writer does not request entropy")
        }

        fn allocate_temp_id(&mut self) -> Option<u64> {
            unreachable!("native writer does not allocate temporary names")
        }

        fn http_request(
            &mut self,
            _request: crate::net::HttpRequest,
        ) -> Result<crate::net::HttpResponse, crate::net::RequestError> {
            unreachable!("native writer does not make HTTP requests")
        }

        fn http_route_static(
            &mut self,
            _pattern: &str,
            _status: u16,
            _body: Vec<u8>,
        ) -> Result<(), crate::net::RouteError> {
            unreachable!("native writer does not register HTTP routes")
        }

        fn http_route_file(
            &mut self,
            _pattern: &str,
            _path: &str,
        ) -> Result<(), crate::net::RouteError> {
            unreachable!("native writer does not register HTTP routes")
        }

        fn network_listen(&mut self, _host_port: &str) {
            unreachable!("native writer does not listen on virtual network")
        }

        fn network_request_count(&self) -> usize {
            unreachable!("native writer does not inspect network requests")
        }

        fn network_request_at(&self, _index: usize) -> Option<crate::net::NetworkRequest> {
            unreachable!("native writer does not inspect network requests")
        }

        fn process_snapshot(&mut self) -> Vec<crate::process::ProcessRecord> {
            unreachable!("native writer does not inspect processes")
        }

        fn listener_snapshot(&self) -> Vec<String> {
            unreachable!("native writer does not inspect listeners")
        }

        fn schedule_wake(&mut self, _duration_ns: u64) -> Result<u64, SyscallError> {
            unreachable!("native writer does not sleep")
        }

        fn spawn_argv(
            &mut self,
            _spec: crate::syscalls::SpawnSpec,
        ) -> Result<crate::process::ProcessId, SyscallError> {
            unreachable!("native writer does not spawn processes")
        }

        fn child_status(
            &self,
            _pid: crate::process::ProcessId,
        ) -> Result<Option<i32>, SyscallError> {
            unreachable!("native writer does not inspect child processes")
        }

        fn reap_child(&mut self, _pid: crate::process::ProcessId) -> Result<i32, SyscallError> {
            unreachable!("native writer does not reap child processes")
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

        fn reserve_memory(&mut self, _bytes: u64) -> bool {
            true
        }

        fn release_memory(&mut self, _bytes: u64) {}

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

        fn note_unsupported(&mut self, _feature: &str) {
            unreachable!("native writer does not reject features")
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
