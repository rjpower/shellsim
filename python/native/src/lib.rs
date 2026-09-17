//! PyO3 adapter for embedding shellsim in a host Python process.
//!
//! The adapter never installs shellsim's process-wide seccomp filter. Simulated execution stays
//! capability-free through the Rust library boundary, while the embedding Python process retains
//! its ordinary capabilities. Potentially deep simulator work runs on a sized scoped thread with
//! the Python interpreter detached, and every adapter operation contains Rust panics.

use std::any::Any;
use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Mutex;
use std::thread;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Serialize;
use shellsim::{CommandTrust, InvocationEvent, Limits, RunOutcome};

const ACTION_STACK_BYTES: usize = 8 * 1024 * 1024;

create_exception!(
    shellsim._native,
    SimulationError,
    PyException,
    "An adapter or simulator operation could not be completed."
);

#[derive(Serialize)]
struct RunMetadata {
    outcome: RunOutcome,
    unsupported: Vec<String>,
    dropped_unsupported: u64,
    commands: Vec<String>,
    dropped_commands: u64,
    unsupported_commands: Vec<String>,
    partial_commands: Vec<String>,
    invocations: Vec<InvocationEvent>,
    dropped_invocations: u64,
}

#[derive(Serialize)]
struct MountMetadata {
    files: usize,
    skipped_directories: Vec<String>,
}

/// One persistent simulated machine owned by Python.
#[pyclass(module = "shellsim._native")]
struct NativeEnvironment {
    environment: Mutex<shellsim::Environment>,
}

#[pymethods]
impl NativeEnvironment {
    #[new]
    fn new(cpu: u64, memory: u64, disk: u64, output: u64) -> Self {
        Self {
            environment: Mutex::new(shellsim::Environment::with_limits(Limits {
                cpu,
                memory,
                disk,
                output,
            })),
        }
    }

    /// Execute one shell action and return JSON metadata plus byte-preserving output streams.
    fn run(
        &self,
        py: Python<'_>,
        source: String,
        stdin: Vec<u8>,
    ) -> PyResult<(String, Py<PyBytes>, Py<PyBytes>)> {
        let (metadata, stdout, stderr) = py
            .detach(|| {
                let mut environment = self.lock_environment()?;
                on_worker(&mut environment, move |environment| {
                    Ok(run_action(environment, &source, &stdin))
                })
            })
            .map_err(SimulationError::new_err)?;
        let metadata = serde_json::to_string(&metadata)
            .map_err(|error| SimulationError::new_err(error.to_string()))?;
        Ok((
            metadata,
            PyBytes::new(py, &stdout).unbind(),
            PyBytes::new(py, &stderr).unbind(),
        ))
    }

    /// Write one byte string into the simulated filesystem.
    fn write_file(&self, py: Python<'_>, path: String, data: Vec<u8>, mode: u32) -> PyResult<()> {
        py.detach(|| {
            let mut environment = self.lock_environment()?;
            on_worker(&mut environment, move |environment| {
                let cwd = environment.cwd.clone();
                environment
                    .vfs
                    .write(&cwd, &path, &data, mode)
                    .map_err(|error| error.to_string())
            })
        })
        .map_err(SimulationError::new_err)
    }

    /// Read one simulated file as exact bytes.
    fn read_file(&self, py: Python<'_>, path: String) -> PyResult<Py<PyBytes>> {
        let bytes = py
            .detach(|| {
                let mut environment = self.lock_environment()?;
                on_worker(&mut environment, move |environment| {
                    environment
                        .vfs
                        .read(&environment.cwd, &path)
                        .map_err(|error| error.to_string())
                })
            })
            .map_err(SimulationError::new_err)?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    /// Create a directory in the simulated filesystem.
    fn mkdir(&self, py: Python<'_>, path: String, parents: bool) -> PyResult<()> {
        py.detach(|| {
            let mut environment = self.lock_environment()?;
            on_worker(&mut environment, move |environment| {
                let cwd = environment.cwd.clone();
                let result = if parents {
                    environment.vfs.mkdir_all(&cwd, &path)
                } else {
                    environment.vfs.mkdir(&cwd, &path)
                };
                result.map_err(|error| error.to_string())
            })
        })
        .map_err(SimulationError::new_err)
    }

    /// Import one explicitly selected trusted host tree into the simulated filesystem.
    fn mount(
        &self,
        py: Python<'_>,
        host_root: String,
        destination_root: String,
    ) -> PyResult<String> {
        let report = py
            .detach(|| {
                let mut environment = self.lock_environment()?;
                on_worker(&mut environment, move |environment| {
                    shellsim::host_ingest::mount_host_tree_report(
                        environment,
                        Path::new(&host_root),
                        &destination_root,
                    )
                })
            })
            .map_err(SimulationError::new_err)?;
        serde_json::to_string(&MountMetadata {
            files: report.files,
            skipped_directories: report.skipped_directories,
        })
        .map_err(|error| SimulationError::new_err(error.to_string()))
    }

    #[getter]
    fn terminated(&self) -> PyResult<bool> {
        Ok(self
            .lock_environment()
            .map_err(SimulationError::new_err)?
            .termination_status()
            .is_some())
    }
}

impl NativeEnvironment {
    fn lock_environment(&self) -> Result<std::sync::MutexGuard<'_, shellsim::Environment>, String> {
        self.environment
            .lock()
            .map_err(|_| "shellsim environment lock is poisoned".to_string())
    }
}

fn run_action(
    environment: &mut shellsim::Environment,
    source: &str,
    stdin: &[u8],
) -> (RunMetadata, Vec<u8>, Vec<u8>) {
    let command_start = environment.cmd_trace.len();
    let dropped_commands_start = environment.cmd_trace.dropped();
    let unsupported_start = environment.unsupported.len();
    let dropped_unsupported_start = environment.unsupported.dropped();
    let invocation_start = environment.invocations.next_sequence();
    let dropped_invocations_start = environment.invocations.dropped();
    let (outcome, stdout, stderr) = environment.run_script_capture_with_stdin(source, stdin);
    let invocations = environment.invocations.events_since(invocation_start);
    let metadata = RunMetadata {
        outcome,
        unsupported: environment.unsupported.values_since(unsupported_start),
        dropped_unsupported: environment
            .unsupported
            .dropped()
            .saturating_sub(dropped_unsupported_start),
        commands: environment.cmd_trace.values_since(command_start),
        dropped_commands: environment
            .cmd_trace
            .dropped()
            .saturating_sub(dropped_commands_start),
        unsupported_commands: invocation_names(&invocations, CommandTrust::Unsupported),
        partial_commands: invocation_names(&invocations, CommandTrust::Partial),
        invocations,
        dropped_invocations: environment
            .invocations
            .dropped()
            .saturating_sub(dropped_invocations_start),
    };
    (metadata, stdout, stderr)
}

fn invocation_names(events: &[InvocationEvent], trust: CommandTrust) -> Vec<String> {
    events
        .iter()
        .filter(|event| event.trust == trust)
        .filter_map(|event| event.argv.first().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Run an environment operation with a predictable native stack and contain all Rust unwinds.
fn on_worker<T, F>(environment: &mut shellsim::Environment, operation: F) -> Result<T, String>
where
    T: Send,
    F: FnOnce(&mut shellsim::Environment) -> Result<T, String> + Send,
{
    thread::scope(|scope| {
        let worker = thread::Builder::new()
            .name("shellsim-python".to_string())
            .stack_size(ACTION_STACK_BYTES)
            .spawn_scoped(scope, move || {
                catch_unwind(AssertUnwindSafe(|| operation(environment)))
            })
            .map_err(|error| format!("could not start shellsim worker: {error}"))?;
        match worker.join() {
            Ok(Ok(result)) => result,
            Ok(Err(payload)) | Err(payload) => Err(format!(
                "shellsim operation panicked{}",
                panic_suffix(payload.as_ref())
            )),
        }
    })
}

fn panic_suffix(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        format!(": {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!(": {message}")
    } else {
        String::new()
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeEnvironment>()?;
    module.add("SimulationError", module.py().get_type::<SimulationError>())?;
    Ok(())
}
