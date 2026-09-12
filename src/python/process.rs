//! Python-facing execution over shellsim's bounded logical process layer.
//!
//! This is the capability boundary for `subprocess`: argv is dispatched directly to registered
//! commands or VFS scripts, never parsed by a host shell and never passed to a host process. The
//! initial runner is synchronous, but each invocation still receives a real modeled PID and an
//! isolated process state.

use crate::clock::{EventKind, MAIN_TASK_ID};
use crate::interp::Interp;
use crate::vfs::resolve_against;

use super::native::{PyError, PyProcessOutput, PyProcessRequest, PyResult};

/// Run one modeled child to completion while preserving machine-wide VFS, clock, and resource
/// effects. Process-local cwd and environment changes are discarded when the parent is restored.
pub(super) fn run(interp: &mut Interp, request: PyProcessRequest) -> PyResult<PyProcessOutput> {
    if request.argv.is_empty() || request.argv[0].is_empty() {
        return Err(PyError::value_error("subprocess argv must not be empty"));
    }

    let child_cwd = request
        .cwd
        .as_deref()
        .map(|path| resolve_against(&interp.cwd, path));
    if let Some(path) = &child_cwd {
        if !interp.vfs.is_dir("/", path) {
            return Err(PyError::runtime_error(format!(
                "subprocess cwd is not a directory: {path}"
            )));
        }
    }

    let deadline = request
        .timeout_ns
        .map(|duration| {
            interp
                .clock
                .schedule_after(duration, EventKind::Deadline { task: MAIN_TASK_ID })
                .map_err(|error| {
                    PyError::value_error(format!("invalid subprocess timeout: {error}"))
                })
        })
        .transpose()?;

    let command = request.argv.join(" ");
    let pid = match interp.start_child(&command, true) {
        Ok(child) => child,
        Err(error) => {
            if let Some(deadline) = deadline {
                interp.clock.cancel(deadline);
            }
            return Err(PyError::resource_error(error));
        }
    };

    if let Some(environment) = request.environment {
        interp.vars.clear();
        interp.arrays.clear();
        interp.exported.clear();
        for (name, value) in environment {
            interp.set_var(&name, value);
            interp.export(&name);
        }
    }
    if let Some(path) = child_cwd {
        interp.set_var("PWD", path);
        interp.export("PWD");
    }
    let process_environment = interp.child_env().into_iter().collect();
    let process_cwd = interp.cwd.clone();
    interp
        .processes
        .update_current(pid, &process_cwd, process_environment);

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut status = crate::commands::run(
        interp,
        &request.argv,
        request.stdin,
        &mut stdout,
        &mut stderr,
    );
    if let Some(signal) = interp.take_terminating_signal() {
        status = 128 + signal.number();
    }
    let interrupted = interp.deadline_interrupt;
    let timed_out = deadline.is_some_and(|event| interrupted == Some(event));
    if let Some(deadline) = deadline {
        interp.clock.cancel(deadline);
        if timed_out {
            interp.deadline_interrupt = None;
        }
    }
    interp.finish_child(pid, status);
    interp.processes.reap(pid);
    let _ = interp.scheduler.reap(pid);

    Ok(PyProcessOutput {
        status,
        stdout: Some(stdout),
        stderr: Some(stderr),
        timed_out,
    })
}
