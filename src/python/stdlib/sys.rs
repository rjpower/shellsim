//! Static interpreter metadata and invocation values for the modeled Python process.

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyConstant, PyError, PyMarker,
    PyResult, PyRuntime, ValueDef,
};

pub(crate) static STREAM_TYPE: NativeTypeDef = NativeTypeDef {
    name: "shellsim.stream",
    methods: &[
        MethodDef {
            type_name: "shellsim.stream",
            name: "write",
            call: write,
        },
        MethodDef {
            type_name: "shellsim.stream",
            name: "read",
            call: read,
        },
        MethodDef {
            type_name: "shellsim.stream",
            name: "readline",
            call: readline,
        },
        MethodDef {
            type_name: "shellsim.stream",
            name: "__iter__",
            call: iter,
        },
        MethodDef {
            type_name: "shellsim.stream",
            name: "__next__",
            call: next,
        },
    ],
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "sys",
    functions: &[FunctionDef {
        module: "sys",
        name: "exit",
        call: exit,
    }],
    values: &[
        ValueDef::Constant {
            name: "version",
            value: PyConstant::String("3.14.0 (shellsim)"),
        },
        ValueDef::Constant {
            name: "version_info",
            value: PyConstant::String("(3, 14, 0, 'final', 0)"),
        },
        ValueDef::Constant {
            name: "executable",
            value: PyConstant::String("/usr/bin/python3.14"),
        },
        ValueDef::Constant {
            name: "prefix",
            value: PyConstant::String("/usr"),
        },
        ValueDef::Factory {
            name: "argv",
            get: argv,
        },
        ValueDef::Factory {
            name: "stdin",
            get: stdin,
        },
        ValueDef::Factory {
            name: "stdout",
            get: stdout,
        },
        ValueDef::Factory {
            name: "stderr",
            get: stderr,
        },
    ],
};

fn exit(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("sys.exit", 0, 1)?;
    args.reject_keywords("sys.exit")?;
    let status = match args.positional().first() {
        None => 0,
        Some(value) if *value == Value::None => 0,
        Some(value) => runtime
            .int_value(value)
            .and_then(|status| i32::try_from(status).ok())
            .unwrap_or(1),
    };
    Err(PyError::exit(status))
}

fn argv(runtime: &mut dyn PyRuntime) -> PyResult {
    runtime.new_argv()
}

fn stdin(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.marker(PyMarker::Stdin))
}

fn stdout(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.marker(PyMarker::Stdout))
}

fn stderr(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.marker(PyMarker::Stderr))
}

fn write(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("stream.write", 1, 1)?;
    args.reject_keywords("stream.write")?;
    let text = runtime.display(&args.positional()[0])?;
    let written = runtime.write_stream(&receiver, &text)?;
    Ok(Value::Int(i64::try_from(written).unwrap_or(i64::MAX)))
}

fn read(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    read_inner(runtime, receiver, args, false)
}

fn readline(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    read_inner(runtime, receiver, args, true)
}

fn iter(_runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("stream.__iter__", 0, 0)?;
    args.reject_keywords("stream.__iter__")?;
    Ok(slot_iter(_runtime, receiver)?.expect("stream iteration always returns itself"))
}

fn next(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("stream.__next__", 0, 0)?;
    args.reject_keywords("stream.__next__")?;
    slot_next(runtime, receiver)?.ok_or_else(|| PyError::exception("StopIteration", ""))
}

pub(crate) fn slot_iter(_runtime: &mut dyn PyRuntime, receiver: Value) -> PyResult<Option<Value>> {
    Ok(Some(receiver))
}

pub(crate) fn slot_next(runtime: &mut dyn PyRuntime, receiver: Value) -> PyResult<Option<Value>> {
    let text = runtime.read_stream(&receiver, None, true)?;
    if text.is_empty() {
        Err(PyError::exception("StopIteration", ""))
    } else {
        runtime.new_string(text).map(Some)
    }
}

fn read_inner(
    runtime: &mut dyn PyRuntime,
    receiver: Value,
    args: CallArgs,
    line: bool,
) -> PyResult {
    args.expect_positional("stream.read", 0, 1)?;
    args.reject_keywords("stream.read")?;
    let size = match args.positional().first() {
        None => None,
        Some(value) => match runtime.int_value(value) {
            Some(value) if value < 0 => None,
            Some(value) => Some(
                usize::try_from(value)
                    .map_err(|_| PyError::value_error("stream size is too large"))?,
            ),
            None => return Err(PyError::type_error("stream size must be an integer")),
        },
    };
    let text = runtime.read_stream(&receiver, size, line)?;
    runtime.new_string(text)
}
