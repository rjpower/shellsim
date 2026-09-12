//! Python-facing execution over shellsim's bounded logical process layer.
//!
//! This is the capability boundary for `subprocess`: argv is dispatched directly to registered
//! commands or VFS scripts, never parsed by a host shell and never passed to a host process. The
//! each invocation receives a real modeled PID, resumable continuation, and isolated process
//! state.

use crate::descriptors::{DescriptorError, FdTable, IoPoll, IoWait, DEFAULT_PIPE_CAPACITY};
use crate::interp::Interp;
use crate::process::{LiveChild, ProcessStatus, Signal};
use crate::vfs::resolve_against;

use super::native::{
    PyError, PyProcessHandle, PyProcessOutput, PyProcessStartRequest, PyResult, PyStdio,
};

/// Start a modeled argv child and retain its parent-side pipe endpoints by logical PID.
pub(super) fn start(
    interp: &mut Interp,
    request: PyProcessStartRequest,
) -> PyResult<PyProcessHandle> {
    validate_argv(&request.argv)?;
    let cwd = validated_cwd(interp, request.cwd.as_deref())?;
    if request.stdin == PyStdio::MergeStdout || request.stdout == PyStdio::MergeStdout {
        return Err(PyError::value_error(
            "invalid subprocess stream disposition",
        ));
    }
    let owner = interp.process.pid;
    let command = request.argv.join(" ");
    let pid = interp
        .start_live_child(&command, true)
        .map_err(PyError::resource_error)?;
    let mut endpoints = FdTable::new();
    let setup = (|| -> Result<(), String> {
        interp.configure_process(pid, cwd, request.environment)?;
        setup_stdin(interp, pid, request.stdin, &mut endpoints)?;
        setup_output(interp, pid, 1, request.stdout, &mut endpoints)?;
        if request.stderr == PyStdio::MergeStdout {
            let stdout = interp
                .process_description(pid, 1)
                .map_err(descriptor_message)?;
            interp
                .install_process_description(pid, 2, stdout)
                .map_err(descriptor_message)?;
        } else {
            setup_output(interp, pid, 2, request.stderr, &mut endpoints)?;
        }
        interp
            .process
            .set_continuation(
                pid,
                Some(crate::exec::ShellContinuation::new(
                    &crate::shell::Node::ArgvCommand(request.argv),
                )),
            )
            .map_err(|error| error.to_string())?;
        Ok(())
    })();
    if let Err(error) = setup {
        endpoints.close_all(&mut interp.descriptors);
        interp.cancel_unstarted_child(pid);
        return Err(PyError::runtime_error(format!(
            "unable to start subprocess: {error}"
        )));
    }
    interp.live_children.insert(
        pid,
        LiveChild {
            owner,
            endpoints,
            status: None,
            stdout: Vec::new(),
            stderr: Vec::new(),
            communicated: false,
            communicate_input: None,
            communicate_offset: 0,
            stdin_pipe: request.stdin == PyStdio::Pipe,
            stdout_pipe: request.stdout == PyStdio::Pipe,
            stderr_pipe: request.stderr == PyStdio::Pipe,
            stdout_inherit: request.stdout == PyStdio::Inherit,
            stderr_inherit: request.stderr == PyStdio::Inherit,
            terminating_signal: None,
        },
    );
    Ok(PyProcessHandle { pid })
}

pub(super) fn poll(interp: &mut Interp, handle: PyProcessHandle) -> PyResult<Option<i32>> {
    let owner = checked_owner(interp, handle)?;
    if live_status(interp, handle.pid).is_none() {
        crate::exec::drive_scheduler_step(interp, handle.pid, Some(interp.clock.monotonic_ns()))
            .map_err(PyError::runtime_error)?;
    }
    debug_assert_eq!(interp.process.pid, owner);
    record_status(interp, handle.pid)
}

pub(super) fn wait(
    interp: &mut Interp,
    handle: PyProcessHandle,
    timeout_ns: Option<u64>,
) -> PyResult<PyProcessOutput> {
    checked_owner(interp, handle)?;
    let deadline = deadline(interp, timeout_ns)?;
    let mut child = interp
        .live_children
        .remove(&handle.pid)
        .ok_or_else(|| PyError::runtime_error("unknown subprocess handle"))?;
    let result = wait_inner(interp, handle.pid, deadline, &mut child);
    interp.live_children.insert(handle.pid, child);
    result
}

fn wait_inner(
    interp: &mut Interp,
    pid: u32,
    deadline: Option<u64>,
    handle: &mut LiveChild,
) -> PyResult<PyProcessOutput> {
    loop {
        drain_output(interp, handle, 1)?;
        drain_output(interp, handle, 2)?;
        if let Some(status) = process_status(interp, pid) {
            handle.status = Some(python_returncode(handle.terminating_signal, status));
            reap_process(interp, pid);
            return Ok(output_from_handle(handle, false));
        }
        let exited = crate::exec::drive_scheduler_step(interp, pid, deadline)
            .map_err(PyError::runtime_error)?;
        if !exited && deadline.is_some_and(|limit| interp.clock.monotonic_ns() >= limit) {
            return Ok(output_from_handle(handle, true));
        }
    }
}

pub(super) fn communicate(
    interp: &mut Interp,
    process: PyProcessHandle,
    input: Vec<u8>,
    timeout_ns: Option<u64>,
) -> PyResult<PyProcessOutput> {
    checked_owner(interp, process)?;
    let deadline = deadline(interp, timeout_ns)?;
    let mut handle = interp
        .live_children
        .remove(&process.pid)
        .ok_or_else(|| PyError::runtime_error("unknown subprocess handle"))?;
    let result = communicate_inner(interp, process.pid, input, deadline, &mut handle);
    interp.live_children.insert(process.pid, handle);
    result
}

fn communicate_inner(
    interp: &mut Interp,
    pid: u32,
    input: Vec<u8>,
    deadline: Option<u64>,
    handle: &mut LiveChild,
) -> PyResult<PyProcessOutput> {
    if handle.communicated {
        return Ok(output_from_handle(handle, false));
    }
    if !input.is_empty() && !handle.stdin_pipe {
        return Err(PyError::value_error(
            "communicate input requires stdin=PIPE",
        ));
    }
    match &handle.communicate_input {
        None => handle.communicate_input = Some(input),
        Some(previous) if input.is_empty() || previous == &input => {}
        Some(_) => {
            return Err(PyError::value_error(
                "communicate input differs from the first call",
            ))
        }
    }
    loop {
        write_communicate_input(interp, handle)?;
        drain_output(interp, handle, 1)?;
        drain_output(interp, handle, 2)?;
        if process_status(interp, pid).is_some() || handle.status.is_some() {
            drain_output(interp, handle, 1)?;
            drain_output(interp, handle, 2)?;
            break;
        }
        let exited = crate::exec::drive_scheduler_step(interp, pid, deadline)
            .map_err(PyError::runtime_error)?;
        if !exited && deadline.is_some_and(|limit| interp.clock.monotonic_ns() >= limit) {
            return Ok(output_from_handle(handle, true));
        }
    }
    handle.status = process_status(interp, pid)
        .map(|status| python_returncode(handle.terminating_signal, status));
    reap_process(interp, pid);
    handle.communicated = true;
    Ok(output_from_handle(handle, false))
}

pub(super) fn send_signal(
    interp: &mut Interp,
    handle: PyProcessHandle,
    signal: Signal,
) -> PyResult<()> {
    checked_owner(interp, handle)?;
    if live_status(interp, handle.pid).is_some() {
        return Ok(());
    }
    interp
        .send_signal(handle.pid, signal)
        .map_err(PyError::runtime_error)?;
    if signal.terminates() {
        if let Some(child) = interp.live_children.get_mut(&handle.pid) {
            child.terminating_signal = Some(signal);
        }
    }
    Ok(())
}

/// Read captured child output while allowing the producer to run whenever its pipe is empty.
pub(super) fn read_pipe(
    interp: &mut Interp,
    process: PyProcessHandle,
    fd: i32,
    amount: Option<usize>,
) -> PyResult<Vec<u8>> {
    checked_owner(interp, process)?;
    if !matches!(fd, 1 | 2) {
        return Err(PyError::value_error("only stdout and stderr are readable"));
    }
    let mut handle = interp
        .live_children
        .remove(&process.pid)
        .ok_or_else(|| PyError::runtime_error("unknown subprocess handle"))?;
    let result = read_pipe_inner(interp, process.pid, fd, amount, &mut handle);
    interp.live_children.insert(process.pid, handle);
    result
}

fn read_pipe_inner(
    interp: &mut Interp,
    pid: u32,
    fd: i32,
    amount: Option<usize>,
    handle: &mut LiveChild,
) -> PyResult<Vec<u8>> {
    let captured = if fd == 1 {
        handle.stdout_pipe
    } else {
        handle.stderr_pipe
    };
    if !captured {
        return Err(PyError::value_error("stream was not opened with PIPE"));
    }
    if amount == Some(0) {
        return Ok(Vec::new());
    }
    loop {
        drain_output(interp, handle, fd)?;
        let available = if fd == 1 {
            handle.stdout.len()
        } else {
            handle.stderr.len()
        };
        let endpoint_closed = handle.endpoints.get(fd).is_err();
        if amount.is_some_and(|limit| available >= limit) || endpoint_closed {
            let take = amount.map_or(available, |limit| limit.min(available));
            let buffer = if fd == 1 {
                &mut handle.stdout
            } else {
                &mut handle.stderr
            };
            return Ok(buffer.drain(..take).collect());
        }
        crate::exec::drive_scheduler_step(interp, pid, None).map_err(PyError::runtime_error)?;
    }
}

/// Write all bytes to a piped child stdin, yielding whenever pipe capacity is exhausted.
pub(super) fn write_pipe(
    interp: &mut Interp,
    process: PyProcessHandle,
    input: Vec<u8>,
) -> PyResult<usize> {
    checked_owner(interp, process)?;
    let input_len = input.len();
    let mut handle = interp
        .live_children
        .remove(&process.pid)
        .ok_or_else(|| PyError::runtime_error("unknown subprocess handle"))?;
    let result = write_pipe_inner(interp, process.pid, &input, &mut handle);
    interp.live_children.insert(process.pid, handle);
    result.map(|()| input_len)
}

fn write_pipe_inner(
    interp: &mut Interp,
    pid: u32,
    input: &[u8],
    handle: &mut LiveChild,
) -> PyResult<()> {
    if !handle.stdin_pipe {
        return Err(PyError::value_error("stdin was not opened with PIPE"));
    }
    let mut offset = 0usize;
    while offset < input.len() {
        let description = handle
            .endpoints
            .get(0)
            .map_err(|_| PyError::runtime_error("write to closed subprocess stdin"))?;
        match interp
            .descriptors
            .write(description, &input[offset..])
            .map_err(|error| PyError::runtime_error(descriptor_message(error)))?
        {
            IoPoll::Ready(written) => {
                offset = offset.saturating_add(written);
                wake_io(interp, IoWait::PipeReadable(pipe_id(interp, description)?));
            }
            IoPoll::Blocked(_) => {
                if process_status(interp, pid).is_some() {
                    return Err(PyError::runtime_error("broken subprocess stdin pipe"));
                }
                crate::exec::drive_scheduler_step(interp, pid, None)
                    .map_err(PyError::runtime_error)?;
            }
        }
    }
    Ok(())
}

/// Close a parent-side pipe and wake a child waiting for EOF or broken-pipe delivery.
pub(super) fn close_pipe(interp: &mut Interp, process: PyProcessHandle, fd: i32) -> PyResult<()> {
    checked_owner(interp, process)?;
    if !matches!(fd, 0..=2) {
        return Err(PyError::value_error("invalid subprocess stream"));
    }
    let mut handle = interp
        .live_children
        .remove(&process.pid)
        .ok_or_else(|| PyError::runtime_error("unknown subprocess handle"))?;
    let result = if handle.endpoints.get(fd).is_ok() {
        close_endpoint(interp, &mut handle, fd)
    } else {
        Ok(())
    };
    interp.live_children.insert(process.pid, handle);
    result
}

fn validate_argv(argv: &[String]) -> PyResult<()> {
    if argv.is_empty() || argv[0].is_empty() {
        Err(PyError::value_error("subprocess argv must not be empty"))
    } else {
        Ok(())
    }
}

fn validated_cwd(interp: &Interp, cwd: Option<&str>) -> PyResult<Option<String>> {
    let cwd = cwd.map(|path| resolve_against(&interp.cwd, path));
    if let Some(path) = &cwd {
        if !interp.vfs.is_dir("/", path) {
            return Err(PyError::runtime_error(format!(
                "subprocess cwd is not a directory: {path}"
            )));
        }
    }
    Ok(cwd)
}

fn setup_stdin(
    interp: &mut Interp,
    pid: u32,
    mode: PyStdio,
    endpoints: &mut FdTable,
) -> Result<(), String> {
    match mode {
        PyStdio::Inherit => Ok(()),
        PyStdio::Pipe => install_pipe(interp, pid, 0, 0, endpoints, true),
        PyStdio::DevNull => install_null(interp, pid, 0),
        PyStdio::MergeStdout => Err("stdin cannot merge stdout".to_string()),
    }
}

fn setup_output(
    interp: &mut Interp,
    pid: u32,
    fd: i32,
    mode: PyStdio,
    endpoints: &mut FdTable,
) -> Result<(), String> {
    match mode {
        PyStdio::Inherit => install_pipe(interp, pid, fd, fd, endpoints, false),
        PyStdio::Pipe => install_pipe(interp, pid, fd, fd, endpoints, false),
        PyStdio::DevNull => install_null(interp, pid, fd),
        PyStdio::MergeStdout => Err("only stderr may merge stdout".to_string()),
    }
}

fn install_pipe(
    interp: &mut Interp,
    pid: u32,
    child_fd: i32,
    parent_fd: i32,
    endpoints: &mut FdTable,
    child_reads: bool,
) -> Result<(), String> {
    let (reader, writer) = interp
        .descriptors
        .open_pipe(DEFAULT_PIPE_CAPACITY)
        .map_err(descriptor_message)?;
    let (child, parent) = if child_reads {
        (reader, writer)
    } else {
        (writer, reader)
    };
    if let Err(error) = endpoints.install(parent_fd, parent, &mut interp.descriptors) {
        let _ = interp.descriptors.discard_unreferenced(reader);
        let _ = interp.descriptors.discard_unreferenced(writer);
        return Err(descriptor_message(error));
    }
    if let Err(error) = interp.install_process_description(pid, child_fd, child) {
        let _ = endpoints.close(parent_fd, &mut interp.descriptors);
        let _ = interp.descriptors.discard_unreferenced(child);
        return Err(descriptor_message(error));
    }
    Ok(())
}

fn install_null(interp: &mut Interp, pid: u32, fd: i32) -> Result<(), String> {
    let null = interp.descriptors.open_null().map_err(descriptor_message)?;
    if let Err(error) = interp.install_process_description(pid, fd, null) {
        let _ = interp.descriptors.discard_unreferenced(null);
        return Err(descriptor_message(error));
    }
    Ok(())
}

fn checked_owner(interp: &Interp, handle: PyProcessHandle) -> PyResult<u32> {
    let child = interp
        .live_children
        .get(&handle.pid)
        .ok_or_else(|| PyError::runtime_error("unknown subprocess handle"))?;
    if child.owner != interp.process.pid {
        return Err(PyError::runtime_error(
            "subprocess handle belongs to another logical process",
        ));
    }
    Ok(child.owner)
}

fn process_status(interp: &Interp, pid: u32) -> Option<i32> {
    match interp.processes.get(pid).map(|record| record.status) {
        Some(ProcessStatus::Exited(status)) => Some(status),
        _ => None,
    }
}

fn live_status(interp: &Interp, pid: u32) -> Option<i32> {
    interp
        .live_children
        .get(&pid)
        .and_then(|handle| handle.status)
        .or_else(|| process_status(interp, pid))
}

fn record_status(interp: &mut Interp, pid: u32) -> PyResult<Option<i32>> {
    let status = live_status(interp, pid);
    if let Some(status) = status {
        let handle = interp
            .live_children
            .get_mut(&pid)
            .ok_or_else(|| PyError::runtime_error("unknown subprocess handle"))?;
        handle.status = Some(python_returncode(handle.terminating_signal, status));
        reap_process(interp, pid);
    }
    Ok(interp
        .live_children
        .get(&pid)
        .and_then(|handle| handle.status))
}

fn reap_process(interp: &mut Interp, pid: u32) {
    if matches!(
        interp.processes.get(pid).map(|record| record.status),
        Some(ProcessStatus::Exited(_))
    ) {
        interp.processes.reap(pid);
        let _ = interp.scheduler.reap(pid);
    }
}

fn deadline(interp: &Interp, timeout_ns: Option<u64>) -> PyResult<Option<u64>> {
    timeout_ns
        .map(|duration| {
            interp
                .clock
                .monotonic_ns()
                .checked_add(duration)
                .ok_or_else(|| PyError::value_error("subprocess timeout is too large"))
        })
        .transpose()
}

fn write_communicate_input(interp: &mut Interp, handle: &mut LiveChild) -> PyResult<()> {
    let Ok(description) = handle.endpoints.get(0) else {
        return Ok(());
    };
    let input = handle.communicate_input.as_deref().unwrap_or_default();
    if handle.communicate_offset < input.len() {
        let write = interp
            .descriptors
            .write(description, &input[handle.communicate_offset..]);
        match write {
            Ok(IoPoll::Ready(written)) => {
                handle.communicate_offset = handle.communicate_offset.saturating_add(written);
                wake_io(interp, IoWait::PipeReadable(pipe_id(interp, description)?));
            }
            Ok(IoPoll::Blocked(_)) => return Ok(()),
            Err(DescriptorError::BrokenPipe) => {
                // `communicate` deliberately suppresses broken-pipe errors when a child exits
                // before consuming all input, matching the public subprocess contract.
                handle.communicate_offset = input.len();
            }
            Err(error) => return Err(PyError::runtime_error(descriptor_message(error))),
        }
    }
    if handle.communicate_offset == input.len() {
        close_endpoint(interp, handle, 0)?;
    }
    Ok(())
}

fn drain_output(interp: &mut Interp, handle: &mut LiveChild, fd: i32) -> PyResult<()> {
    let Ok(description) = handle.endpoints.get(fd) else {
        return Ok(());
    };
    loop {
        match interp
            .descriptors
            .read(description, 16 * 1024)
            .map_err(|error| PyError::runtime_error(descriptor_message(error)))?
        {
            IoPoll::Ready(bytes) if bytes.is_empty() => {
                close_endpoint(interp, handle, fd)?;
                return Ok(());
            }
            IoPoll::Ready(bytes) => {
                if fd == 1 {
                    handle.stdout.extend_from_slice(&bytes);
                } else {
                    handle.stderr.extend_from_slice(&bytes);
                }
                wake_io(interp, IoWait::PipeWritable(pipe_id(interp, description)?));
            }
            IoPoll::Blocked(_) => return Ok(()),
        }
    }
}

fn close_endpoint(interp: &mut Interp, handle: &mut LiveChild, fd: i32) -> PyResult<()> {
    let description = handle
        .endpoints
        .get(fd)
        .map_err(|error| PyError::runtime_error(descriptor_message(error)))?;
    let peer = interp.descriptors.pipe_endpoint(description).ok().flatten();
    handle
        .endpoints
        .close(fd, &mut interp.descriptors)
        .map_err(|error| PyError::runtime_error(descriptor_message(error)))?;
    if let Some((pipe, reader)) = peer {
        let reason = if reader {
            crate::scheduler::WaitReason::PipeWritable(pipe)
        } else {
            crate::scheduler::WaitReason::PipeReadable(pipe)
        };
        interp.scheduler.wake_waiters(reason);
    }
    Ok(())
}

fn pipe_id(interp: &Interp, description: u32) -> PyResult<u32> {
    interp
        .descriptors
        .pipe_endpoint(description)
        .map_err(|error| PyError::runtime_error(descriptor_message(error)))?
        .map(|(pipe, _)| pipe)
        .ok_or_else(|| PyError::runtime_error("subprocess endpoint is not a pipe"))
}

fn wake_io(interp: &mut Interp, wait: IoWait) {
    let reason = match wait {
        IoWait::PipeReadable(pipe) => crate::scheduler::WaitReason::PipeReadable(pipe),
        IoWait::PipeWritable(pipe) => crate::scheduler::WaitReason::PipeWritable(pipe),
    };
    interp.scheduler.wake_waiters(reason);
}

fn output_from_handle(handle: &mut LiveChild, timed_out: bool) -> PyProcessOutput {
    PyProcessOutput {
        status: handle.status.unwrap_or_default(),
        stdout: handle.stdout_pipe.then(|| handle.stdout.clone()),
        stderr: handle.stderr_pipe.then(|| handle.stderr.clone()),
        timed_out,
        inherited_stdout: if handle.stdout_inherit {
            std::mem::take(&mut handle.stdout)
        } else {
            Vec::new()
        },
        inherited_stderr: if handle.stderr_inherit {
            std::mem::take(&mut handle.stderr)
        } else {
            Vec::new()
        },
    }
}

fn descriptor_message(error: DescriptorError) -> String {
    format!("descriptor error: {error:?}")
}

fn python_returncode(signal: Option<Signal>, status: i32) -> i32 {
    signal
        .filter(|signal| status == 128 + signal.number())
        .map_or(status, |signal| -signal.number())
}
