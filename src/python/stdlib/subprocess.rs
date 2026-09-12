//! Native process primitive for the frozen `subprocess` facade.
//!
//! This module performs only checked conversion between erased Python values and the explicit
//! logical-process capability. Public API policy and result objects live in `source/subprocess.py`.

use std::collections::BTreeMap;

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyBytes, PyDict, PyError, PyKind, PyList, PyProcessHandle,
    PyProcessOutput, PyProcessStartRequest, PyResult, PyRuntime, PyStdio, PyString, PyTuple,
    PyValue, PyValueCast, ValueDef,
};
use super::super::number::PyNumber;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_shellsim_subprocess",
    functions: &[
        FunctionDef {
            module: "_shellsim_subprocess",
            name: "start",
            call: start,
        },
        FunctionDef {
            module: "_shellsim_subprocess",
            name: "poll",
            call: poll,
        },
        FunctionDef {
            module: "_shellsim_subprocess",
            name: "wait",
            call: wait,
        },
        FunctionDef {
            module: "_shellsim_subprocess",
            name: "communicate",
            call: communicate,
        },
        FunctionDef {
            module: "_shellsim_subprocess",
            name: "read_pipe",
            call: read_pipe,
        },
        FunctionDef {
            module: "_shellsim_subprocess",
            name: "write_pipe",
            call: write_pipe,
        },
        FunctionDef {
            module: "_shellsim_subprocess",
            name: "close_pipe",
            call: close_pipe,
        },
        FunctionDef {
            module: "_shellsim_subprocess",
            name: "send_signal",
            call: send_signal,
        },
    ],
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

fn start(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_subprocess.start", 7, 7)?;
    args.reject_keywords("_shellsim_subprocess.start")?;
    let values = args.positional();
    let request = PyProcessStartRequest {
        argv: string_sequence(runtime, values[0])?,
        cwd: optional_string(runtime, values[1], "cwd")?,
        environment: optional_environment(runtime, values[2])?,
        stdin: stdio(runtime, values[3], false)?,
        stdout: stdio(runtime, values[4], false)?,
        stderr: stdio(runtime, values[5], true)?,
        start_new_session: values[6]
            .bool_value()
            .ok_or_else(|| PyError::type_error("start_new_session must be a bool"))?,
    };
    let handle = runtime.processes().start(request)?;
    Ok(PyValue::Int(i64::from(handle.pid)))
}

fn poll(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_subprocess.poll", 1, 1)?;
    args.reject_keywords("_shellsim_subprocess.poll")?;
    let handle = process_handle(runtime, args.positional()[0])?;
    Ok(runtime
        .processes()
        .poll(handle)?
        .map_or(PyValue::None, |status| PyValue::Int(i64::from(status))))
}

fn wait(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_subprocess.wait", 2, 2)?;
    args.reject_keywords("_shellsim_subprocess.wait")?;
    let values = args.positional();
    let handle = process_handle(runtime, values[0])?;
    let timeout = optional_timeout(runtime, values[1])?;
    let output = runtime.processes().wait(handle, timeout)?;
    process_output(runtime, output)
}

fn communicate(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_subprocess.communicate", 3, 3)?;
    args.reject_keywords("_shellsim_subprocess.communicate")?;
    let values = args.positional();
    let handle = process_handle(runtime, values[0])?;
    let input = optional_input(runtime, values[1])?;
    let timeout = optional_timeout(runtime, values[2])?;
    let output = runtime.processes().communicate(handle, input, timeout)?;
    process_output(runtime, output)
}

fn read_pipe(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_subprocess.read_pipe", 3, 3)?;
    args.reject_keywords("_shellsim_subprocess.read_pipe")?;
    let values = args.positional();
    let handle = process_handle(runtime, values[0])?;
    let fd = stream_fd(runtime, values[1])?;
    let amount = runtime
        .int_value(&values[2])
        .ok_or_else(|| PyError::type_error("read size must be an integer"))?;
    let amount = if amount < 0 {
        None
    } else {
        Some(
            usize::try_from(amount)
                .map_err(|_| PyError::overflow_error("read size is too large"))?,
        )
    };
    let bytes = runtime.processes().read_pipe(handle, fd, amount)?;
    runtime.new_bytes(bytes)
}

fn write_pipe(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_subprocess.write_pipe", 2, 2)?;
    args.reject_keywords("_shellsim_subprocess.write_pipe")?;
    let values = args.positional();
    let handle = process_handle(runtime, values[0])?;
    let input = optional_input(runtime, values[1])?;
    let written = runtime.processes().write_pipe(handle, input)?;
    let written =
        i64::try_from(written).map_err(|_| PyError::overflow_error("write result is too large"))?;
    Ok(PyValue::Int(written))
}

fn close_pipe(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_subprocess.close_pipe", 2, 2)?;
    args.reject_keywords("_shellsim_subprocess.close_pipe")?;
    let values = args.positional();
    let handle = process_handle(runtime, values[0])?;
    let fd = stream_fd(runtime, values[1])?;
    runtime.processes().close_pipe(handle, fd)?;
    Ok(PyValue::None)
}

fn send_signal(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_subprocess.send_signal", 2, 2)?;
    args.reject_keywords("_shellsim_subprocess.send_signal")?;
    let values = args.positional();
    let handle = process_handle(runtime, values[0])?;
    let number = runtime
        .int_value(&values[1])
        .ok_or_else(|| PyError::type_error("signal must be an integer"))?;
    let signal = crate::process::Signal::parse(&number.to_string())
        .ok_or_else(|| PyError::value_error("unsupported signal"))?;
    runtime.processes().send_signal(handle, signal)?;
    Ok(PyValue::None)
}

fn process_handle(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<PyProcessHandle> {
    let pid = runtime
        .int_value(&value)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| PyError::value_error("invalid subprocess handle"))?;
    Ok(PyProcessHandle { pid })
}

fn stream_fd(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<i32> {
    let fd = runtime
        .int_value(&value)
        .and_then(|value| i32::try_from(value).ok())
        .ok_or_else(|| PyError::value_error("invalid subprocess stream"))?;
    if matches!(fd, 0..=2) {
        Ok(fd)
    } else {
        Err(PyError::value_error("invalid subprocess stream"))
    }
}

fn process_output(runtime: &mut dyn PyRuntime, output: PyProcessOutput) -> PyResult {
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
