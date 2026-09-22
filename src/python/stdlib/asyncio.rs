//! VM primitives used by the deterministic `asyncio` compatibility module.
//!
//! Scheduling policy stays in frozen Python. This module exposes bounded coroutine stepping and
//! checked waits on virtual time and modeled process resources. It cannot acquire host threads,
//! clocks, processes, files, or network access.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyIterator, PyKind, PyList, PyProcessHandle,
    PyProcessPoll, PyResult, PyRuntime, PyTuple, PyValueCast,
};
use super::super::Value;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_asyncio",
    functions: &[
        FunctionDef {
            module: "_asyncio",
            name: "_step",
            call: step,
        },
        FunctionDef {
            module: "_asyncio",
            name: "_is_coroutine",
            call: is_coroutine,
        },
        FunctionDef {
            module: "_asyncio",
            name: "_monotonic_ns",
            call: monotonic_ns,
        },
        FunctionDef {
            module: "_asyncio",
            name: "_wait_resources",
            call: wait_resources,
        },
        FunctionDef {
            module: "_asyncio",
            name: "_process_poll",
            call: process_poll,
        },
        FunctionDef {
            module: "_asyncio",
            name: "_process_read",
            call: process_read,
        },
        FunctionDef {
            module: "_asyncio",
            name: "_process_write",
            call: process_write,
        },
    ],
    values: &[],
};

fn step(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_asyncio._step", 2, 2)?;
    args.reject_keywords("_asyncio._step")?;
    let coroutine = args.positional()[0];
    if runtime.kind(&coroutine)? != PyKind::Generator {
        return Err(PyError::type_error("expected a coroutine"));
    }
    let iterator = coroutine.cast::<PyIterator>(runtime)?;
    let sent = args.positional()[1];
    let (status, value) = runtime.coroutine_step(iterator, sent)?;
    runtime.new_tuple(vec![Value::Int(i64::from(status)), value])
}

fn is_coroutine(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_asyncio._is_coroutine", 1, 1)?;
    args.reject_keywords("_asyncio._is_coroutine")?;
    Ok(Value::Bool(
        runtime.kind(&args.positional()[0])? == PyKind::Generator,
    ))
}

fn monotonic_ns(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_asyncio._monotonic_ns", 0, 0)?;
    args.reject_keywords("_asyncio._monotonic_ns")?;
    runtime.clock().monotonic_ns().map(Value::Int)
}

fn wait_resources(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_asyncio._wait_resources", 1, 1)?;
    args.reject_keywords("_asyncio._wait_resources")?;
    let tokens = args.positional()[0].cast::<PyList>(runtime)?;
    let tokens = runtime.list_items(tokens)?;
    if tokens.len() > crate::scheduler::MAX_TASKS {
        return Err(PyError::resource_error("too many asyncio resource waits"));
    }
    let mut reasons = Vec::with_capacity(tokens.len());
    for token in tokens {
        runtime.charge_cpu(1)?;
        reasons.push(parse_wait_token(runtime, token)?);
    }
    runtime.wait_on(reasons)?;
    Ok(Value::None)
}

fn process_poll(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_asyncio._process_poll", 1, 1)?;
    args.reject_keywords("_asyncio._process_poll")?;
    let handle = process_handle(runtime, args.positional()[0])?;
    Ok(runtime
        .processes()
        .poll(handle)?
        .map_or(Value::None, |status| Value::Int(i64::from(status))))
}

fn process_read(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_asyncio._process_read", 3, 3)?;
    args.reject_keywords("_asyncio._process_read")?;
    let values = args.positional();
    let handle = process_handle(runtime, values[0])?;
    let fd = integer(runtime, values[1], "stream descriptor")?;
    let fd = i32::try_from(fd).map_err(|_| PyError::value_error("invalid stream descriptor"))?;
    let amount = integer(runtime, values[2], "read size")?;
    let amount = if amount < 0 {
        None
    } else {
        Some(
            usize::try_from(amount)
                .map_err(|_| PyError::overflow_error("read size is too large"))?,
        )
    };
    match runtime.processes().try_read_pipe(handle, fd, amount)? {
        PyProcessPoll::Ready(bytes) => {
            let bytes = runtime.new_bytes(bytes)?;
            poll_tuple(runtime, true, bytes, Value::None)
        }
        PyProcessPoll::Blocked(reason) => {
            let token = wait_token(runtime, reason)?;
            poll_tuple(runtime, false, Value::None, token)
        }
    }
}

fn process_write(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_asyncio._process_write", 2, 2)?;
    args.reject_keywords("_asyncio._process_write")?;
    let values = args.positional();
    let handle = process_handle(runtime, values[0])?;
    let input = runtime
        .bytes_value(&values[1])?
        .ok_or_else(|| PyError::type_error("process input must be bytes"))?;
    match runtime.processes().try_write_pipe(handle, input)? {
        PyProcessPoll::Ready(written) => {
            let written = i64::try_from(written)
                .map_err(|_| PyError::overflow_error("write result is too large"))?;
            poll_tuple(runtime, true, Value::Int(written), Value::None)
        }
        PyProcessPoll::Blocked(reason) => {
            let token = wait_token(runtime, reason)?;
            poll_tuple(runtime, false, Value::None, token)
        }
    }
}

fn poll_tuple(runtime: &mut dyn PyRuntime, ready: bool, value: Value, token: Value) -> PyResult {
    runtime.new_tuple(vec![Value::Bool(ready), value, token])
}

fn process_handle(runtime: &dyn PyRuntime, value: Value) -> PyResult<PyProcessHandle> {
    let pid = integer(runtime, value, "subprocess handle")?;
    let pid = u32::try_from(pid).map_err(|_| PyError::value_error("invalid subprocess handle"))?;
    Ok(PyProcessHandle { pid })
}

fn integer(runtime: &dyn PyRuntime, value: Value, name: &str) -> PyResult<i64> {
    runtime
        .int_value(&value)
        .ok_or_else(|| PyError::type_error(format!("{name} must be an integer")))
}

fn parse_wait_token(
    runtime: &mut dyn PyRuntime,
    token: Value,
) -> PyResult<crate::scheduler::WaitReason> {
    let token = token.cast::<PyTuple>(runtime)?;
    let parts = runtime.tuple_items(token)?;
    if parts.len() != 2 {
        return Err(PyError::value_error("invalid asyncio resource token"));
    }
    let kind = runtime
        .string_value(&parts[0])?
        .ok_or_else(|| PyError::type_error("asyncio resource kind must be a string"))?;
    let value = integer(runtime, parts[1], "asyncio resource identifier")?;
    match kind.as_str() {
        "timer" => u64::try_from(value)
            .map(crate::scheduler::WaitReason::Timer)
            .map_err(|_| PyError::value_error("invalid timer deadline")),
        "input_readable" => u32::try_from(value)
            .map(crate::scheduler::WaitReason::InputReadable)
            .map_err(|_| PyError::value_error("invalid input descriptor")),
        "pipe_readable" => u32::try_from(value)
            .map(crate::scheduler::WaitReason::PipeReadable)
            .map_err(|_| PyError::value_error("invalid pipe identifier")),
        "pipe_writable" => u32::try_from(value)
            .map(crate::scheduler::WaitReason::PipeWritable)
            .map_err(|_| PyError::value_error("invalid pipe identifier")),
        "child" => u32::try_from(value)
            .map(crate::scheduler::WaitReason::Child)
            .map_err(|_| PyError::value_error("invalid child identifier")),
        "child_activity" => u32::try_from(value)
            .map(crate::scheduler::WaitReason::ChildActivity)
            .map_err(|_| PyError::value_error("invalid child identifier")),
        _ => Err(PyError::value_error("unsupported asyncio resource kind")),
    }
}

fn wait_token(runtime: &mut dyn PyRuntime, reason: crate::scheduler::WaitReason) -> PyResult {
    let (kind, value) = match reason {
        crate::scheduler::WaitReason::Timer(value) => ("timer", u64_to_i64(value)?),
        crate::scheduler::WaitReason::InputReadable(value) => ("input_readable", i64::from(value)),
        crate::scheduler::WaitReason::PipeReadable(value) => ("pipe_readable", i64::from(value)),
        crate::scheduler::WaitReason::PipeWritable(value) => ("pipe_writable", i64::from(value)),
        crate::scheduler::WaitReason::Child(value) => ("child", i64::from(value)),
        crate::scheduler::WaitReason::ChildActivity(value) => ("child_activity", i64::from(value)),
        crate::scheduler::WaitReason::ChildDeadline(_, _)
        | crate::scheduler::WaitReason::ChildActivityDeadline(_, _)
        | crate::scheduler::WaitReason::Any(_) => {
            return Err(PyError::runtime_error(
                "unsupported nested asyncio resource wait",
            ))
        }
    };
    let kind = runtime.new_string(kind.to_string())?;
    runtime.new_tuple(vec![kind, Value::Int(value)])
}

fn u64_to_i64(value: u64) -> PyResult<i64> {
    i64::try_from(value).map_err(|_| PyError::overflow_error("timer deadline is too large"))
}
