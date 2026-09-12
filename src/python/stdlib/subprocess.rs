//! Native process primitive for the frozen `subprocess` facade.
//!
//! This module performs only checked conversion between erased Python values and the explicit
//! logical-process capability. Public API policy and result objects live in `source/subprocess.py`.

use std::collections::BTreeMap;

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyBytes, PyDict, PyError, PyKind, PyList, PyProcessRequest,
    PyResult, PyRuntime, PyStdio, PyString, PyTuple, PyValue, PyValueCast, ValueDef,
};
use super::super::number::PyNumber;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_shellsim_subprocess",
    functions: &[FunctionDef {
        module: "_shellsim_subprocess",
        name: "run",
        call: run,
    }],
    values: &[
        ValueDef::Factory {
            name: "CalledProcessError",
            get: called_process_error,
        },
        ValueDef::Factory {
            name: "TimeoutExpired",
            get: timeout_expired,
        },
    ],
};

fn called_process_error(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.exception_type("CalledProcessError"))
}

fn timeout_expired(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.exception_type("TimeoutExpired"))
}

fn run(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_subprocess.run", 7, 7)?;
    args.reject_keywords("_shellsim_subprocess.run")?;
    let values = args.positional();
    let argv = string_sequence(runtime, values[0])?;
    let stdin = optional_input(runtime, values[1])?;
    let cwd = optional_string(runtime, values[2], "cwd")?;
    let environment = optional_environment(runtime, values[3])?;
    let timeout_ns = optional_timeout(runtime, values[4])?;
    let stdout = stdio(runtime, values[5], false)?;
    let stderr = stdio(runtime, values[6], true)?;

    let output = runtime.processes().run(PyProcessRequest {
        argv,
        stdin,
        cwd,
        environment,
        timeout_ns,
        stdout,
        stderr,
    })?;
    let stdout = output
        .stdout
        .map(|bytes| runtime.new_bytes(bytes))
        .transpose()?
        .unwrap_or(PyValue::None);
    let stderr = output
        .stderr
        .map(|bytes| runtime.new_bytes(bytes))
        .transpose()?
        .unwrap_or(PyValue::None);
    runtime.new_namespace(vec![
        ("returncode".into(), PyValue::Int(i64::from(output.status))),
        ("stdout".into(), stdout),
        ("stderr".into(), stderr),
        ("timed_out".into(), PyValue::Bool(output.timed_out)),
    ])
}

fn string_sequence(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<String>> {
    let items = match runtime.kind(&value)? {
        PyKind::List => value.cast::<PyList>(runtime)?.items(runtime)?,
        PyKind::Tuple => value.cast::<PyTuple>(runtime)?.items(runtime)?,
        _ => {
            return Err(PyError::type_error(
                "subprocess args must be a list or tuple",
            ))
        }
    };
    if items.is_empty() {
        return Err(PyError::value_error("subprocess args must not be empty"));
    }
    let argv = items
        .into_iter()
        .map(|item| item.cast::<PyString>(runtime).map(|item| item.0))
        .collect::<PyResult<Vec<_>>>()?;
    if argv.iter().any(|argument| argument.contains('\0')) {
        return Err(PyError::value_error(
            "subprocess arguments may not contain NUL",
        ));
    }
    Ok(argv)
}

fn optional_input(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<u8>> {
    match runtime.kind(&value)? {
        PyKind::None => Ok(Vec::new()),
        PyKind::Bytes | PyKind::ByteArray => value.cast::<PyBytes>(runtime).map(|value| value.0),
        _ => Err(PyError::type_error("subprocess input must be bytes")),
    }
}

fn optional_string(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    name: &str,
) -> PyResult<Option<String>> {
    if runtime.kind(&value)? == PyKind::None {
        Ok(None)
    } else {
        value
            .cast::<PyString>(runtime)
            .map(|value| Some(value.0))
            .map_err(|_| PyError::type_error(format!("subprocess {name} must be a string")))
    }
}

fn optional_environment(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<BTreeMap<String, String>>> {
    if runtime.kind(&value)? == PyKind::None {
        return Ok(None);
    }
    let items = value.cast::<PyDict>(runtime)?.items(runtime)?;
    let mut environment = BTreeMap::new();
    for (key, value) in items {
        let PyString(key) = key.cast(runtime)?;
        let PyString(value) = value.cast(runtime)?;
        if key.is_empty() || key.contains(['=', '\0']) || value.contains('\0') {
            return Err(PyError::value_error("invalid subprocess environment entry"));
        }
        environment.insert(key, value);
    }
    Ok(Some(environment))
}

fn optional_timeout(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Option<u64>> {
    if runtime.kind(&value)? == PyKind::None {
        return Ok(None);
    }
    let seconds = value.cast::<PyNumber>(runtime)?.into_f64()?;
    if !seconds.is_finite() || seconds < 0.0 || seconds > u64::MAX as f64 / 1_000_000_000.0 {
        return Err(PyError::value_error(
            "subprocess timeout must be a finite non-negative number",
        ));
    }
    Ok(Some((seconds * 1_000_000_000.0).ceil() as u64))
}

fn stdio(runtime: &mut dyn PyRuntime, value: PyValue, allow_merge: bool) -> PyResult<PyStdio> {
    if runtime.kind(&value)? == PyKind::None {
        return Ok(PyStdio::Inherit);
    }
    match runtime.int_value(&value) {
        Some(-1) => Ok(PyStdio::Pipe),
        Some(-2) if allow_merge => Ok(PyStdio::MergeStdout),
        Some(-3) => Ok(PyStdio::DevNull),
        _ => Err(PyError::value_error(
            "subprocess stdio must be PIPE, STDOUT, DEVNULL, or None",
        )),
    }
}
