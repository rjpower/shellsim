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

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Serialize;
use shellsim::harness::{HarnessOperation, HarnessRequest, HarnessSession};
use shellsim::net::NetworkRequest;
use shellsim::realtime::ClockMode;
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
    network_requests: Vec<NetworkRequest>,
    dropped_network_requests: u64,
}

#[derive(Serialize)]
struct MountMetadata {
    files: usize,
    skipped_directories: Vec<String>,
}

struct MetadataStart {
    command: usize,
    dropped_commands: u64,
    unsupported: usize,
    dropped_unsupported: u64,
    invocation: u64,
    dropped_invocations: u64,
    network: usize,
    dropped_network: u64,
}

/// One persistent simulated machine owned by Python.
#[pyclass(module = "shellsim._native")]
struct NativeEnvironment {
    environment: Mutex<shellsim::Environment>,
}

/// In-process host control for a machine granted the virtual tool endpoint.
#[pyclass(module = "shellsim._native")]
struct NativeContainer {
    session: Mutex<HarnessSession>,
}

#[pymethods]
impl NativeContainer {
    #[new]
    fn new(cpu: u64, memory: u64, disk: u64, output: u64) -> Self {
        Self {
            session: Mutex::new(HarnessSession::with_clock_and_host_tools(
                Limits {
                    cpu,
                    memory,
                    disk,
                    output,
                },
                ClockMode::Virtual,
            )),
        }
    }

    fn write_file(&self, py: Python<'_>, path: String, data: Vec<u8>, mode: u32) -> PyResult<()> {
        self.apply(
            py,
            HarnessOperation::WriteFile {
                path,
                data_base64: STANDARD.encode(data),
                mode,
            },
        )?;
        Ok(())
    }

    fn read_file(&self, py: Python<'_>, path: String) -> PyResult<Py<PyBytes>> {
        let result = self.apply(py, HarnessOperation::ReadFile { path })?;
        let encoded = result["data_base64"]
            .as_str()
            .ok_or_else(|| SimulationError::new_err("missing file bytes"))?;
        let bytes = STANDARD
            .decode(encoded)
            .map_err(|error| SimulationError::new_err(error.to_string()))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    fn start_execute(&self, py: Python<'_>, source: String, stdin: Vec<u8>) -> PyResult<String> {
        self.apply(
            py,
            HarnessOperation::StartExecute {
                source,
                stdin_base64: STANDARD.encode(stdin),
                stdin_closed: true,
            },
        )
        .and_then(serialize_result)
    }

    fn poll_action(&self, py: Python<'_>, action_id: u64) -> PyResult<String> {
        self.apply(
            py,
            HarnessOperation::PollAction {
                action_id,
                work_quanta: 100_000,
                advance_time: true,
            },
        )
        .and_then(serialize_result)
    }

    fn respond_tool_call(
        &self,
        py: Python<'_>,
        request_id: u64,
        result_json: Option<String>,
        error: Option<String>,
    ) -> PyResult<()> {
        let result = result_json
            .map(|json| serde_json::from_str(&json))
            .transpose()
            .map_err(|error| SimulationError::new_err(error.to_string()))?;
        self.apply(
            py,
            HarnessOperation::RespondToolCall {
                request_id,
                result,
                error,
            },
        )?;
        Ok(())
    }

    fn read_action_output(&self, py: Python<'_>, action_id: u64) -> PyResult<String> {
        self.apply(py, HarnessOperation::ReadActionOutput { action_id })
            .and_then(serialize_result)
    }

    fn cancel_action(&self, py: Python<'_>, action_id: u64) -> PyResult<()> {
        self.apply(py, HarnessOperation::CancelAction { action_id })?;
        Ok(())
    }

    fn drop_action(&self, py: Python<'_>, action_id: u64) -> PyResult<()> {
        self.apply(py, HarnessOperation::DropAction { action_id })?;
        Ok(())
    }
}

impl NativeContainer {
    fn apply(&self, py: Python<'_>, operation: HarnessOperation) -> PyResult<serde_json::Value> {
        py.detach(|| {
            let mut session = self
                .session
                .lock()
                .map_err(|_| "shellsim container lock is poisoned".to_string())?;
            on_worker(&mut session, move |session| {
                let response = session.handle(HarnessRequest {
                    id: None,
                    session_id: None,
                    operation,
                });
                if !response.ok {
                    return Err(response
                        .error
                        .unwrap_or_else(|| "container operation failed".into()));
                }
                serde_json::to_value(response.result).map_err(|error| error.to_string())
            })
        })
        .map_err(SimulationError::new_err)
    }
}

fn serialize_result(result: serde_json::Value) -> PyResult<String> {
    serde_json::to_string(&result).map_err(|error| SimulationError::new_err(error.to_string()))
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

    /// Create an idle shell with its own process state over the shared virtual machine.
    fn create_shell(&self, py: Python<'_>) -> PyResult<u32> {
        py.detach(|| {
            let mut environment = self.lock_environment()?;
            on_worker(&mut environment, |environment| {
                environment.spawn_shell_session()
            })
        })
        .map_err(SimulationError::new_err)
    }

    /// Execute one action in a selected persistent shell session.
    fn run_shell(
        &self,
        py: Python<'_>,
        pid: u32,
        source: String,
        stdin: Vec<u8>,
    ) -> PyResult<(String, Py<PyBytes>, Py<PyBytes>)> {
        let (metadata, stdout, stderr) = py
            .detach(|| {
                let mut environment = self.lock_environment()?;
                on_worker(&mut environment, move |environment| {
                    let start = metadata_start(environment);
                    let (outcome, stdout, stderr) =
                        environment.run_shell_session_capture_with_stdin(pid, &source, &stdin)?;
                    Ok((metadata_finish(environment, outcome, start), stdout, stderr))
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

    /// Execute Python source directly without passing it through the shell parser.
    fn run_python(
        &self,
        py: Python<'_>,
        source: String,
        argv: Vec<String>,
        stdin: Vec<u8>,
    ) -> PyResult<(String, Py<PyBytes>, Py<PyBytes>)> {
        let (metadata, stdout, stderr) = py
            .detach(|| {
                let mut environment = self.lock_environment()?;
                on_worker(&mut environment, move |environment| {
                    Ok(run_python_action(environment, source, argv, stdin))
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

    /// Register one static response in the simulated HTTP broker.
    fn route_http(
        &self,
        py: Python<'_>,
        pattern: String,
        method: Option<String>,
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    ) -> PyResult<()> {
        py.detach(|| {
            let mut environment = self.lock_environment()?;
            on_worker(&mut environment, move |environment| {
                environment
                    .net
                    .route(
                        &pattern,
                        method.as_deref(),
                        shellsim::net::HttpResponse {
                            status,
                            headers,
                            body,
                        },
                    )
                    .map_err(|error| error.to_string())
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
    let start = metadata_start(environment);
    let (outcome, stdout, stderr) = environment.run_script_capture_with_stdin(source, stdin);
    (metadata_finish(environment, outcome, start), stdout, stderr)
}

fn run_python_action(
    environment: &mut shellsim::Environment,
    source: String,
    arguments: Vec<String>,
    stdin: Vec<u8>,
) -> (RunMetadata, Vec<u8>, Vec<u8>) {
    let start = metadata_start(environment);
    let mut argv = vec!["python3.14".to_string(), "-c".to_string(), source];
    argv.extend(arguments);
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::exec::exec(
        environment,
        &shellsim::shell::Node::ArgvCommand(argv),
        stdin,
        &mut stdout,
        &mut stderr,
    );
    let outcome = environment.outcome(status);
    (metadata_finish(environment, outcome, start), stdout, stderr)
}

fn metadata_start(environment: &shellsim::Environment) -> MetadataStart {
    MetadataStart {
        command: environment.cmd_trace.len(),
        dropped_commands: environment.cmd_trace.dropped(),
        unsupported: environment.unsupported.len(),
        dropped_unsupported: environment.unsupported.dropped(),
        invocation: environment.invocations.next_sequence(),
        dropped_invocations: environment.invocations.dropped(),
        network: environment.net.log.len(),
        dropped_network: environment.net.dropped_requests,
    }
}

fn metadata_finish(
    environment: &shellsim::Environment,
    outcome: RunOutcome,
    start: MetadataStart,
) -> RunMetadata {
    let invocations = environment.invocations.events_since(start.invocation);
    RunMetadata {
        outcome,
        unsupported: environment.unsupported.values_since(start.unsupported),
        dropped_unsupported: environment
            .unsupported
            .dropped()
            .saturating_sub(start.dropped_unsupported),
        commands: environment.cmd_trace.values_since(start.command),
        dropped_commands: environment
            .cmd_trace
            .dropped()
            .saturating_sub(start.dropped_commands),
        unsupported_commands: invocation_names(&invocations, CommandTrust::Unsupported),
        partial_commands: invocation_names(&invocations, CommandTrust::Partial),
        invocations,
        dropped_invocations: environment
            .invocations
            .dropped()
            .saturating_sub(start.dropped_invocations),
        network_requests: environment.net.log[start.network..].to_vec(),
        dropped_network_requests: environment
            .net
            .dropped_requests
            .saturating_sub(start.dropped_network),
    }
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
fn on_worker<State, T, F>(
    guard: &mut std::sync::MutexGuard<'_, State>,
    operation: F,
) -> Result<T, String>
where
    State: Send,
    T: Send,
    F: FnOnce(&mut State) -> Result<T, String> + Send,
{
    let environment = &mut **guard;
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
    module.add_class::<NativeContainer>()?;
    module.add("SimulationError", module.py().get_type::<SimulationError>())?;
    Ok(())
}
