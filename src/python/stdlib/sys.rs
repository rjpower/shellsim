//! Static interpreter metadata and invocation values for the modeled Python process.

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, GetterDef, MethodDef, ModuleDef, NativeTypeDef, PyConstant, PyError,
    PyMarker, PyResult, PyRuntime, PyStreamRead, ValueDef,
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
            name: "readlines",
            call: readlines,
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
    getters: &[GetterDef {
        owner: "shellsim.stream",
        name: "buffer",
        get: buffer,
    }],
};

/// `sys.stdin.buffer` reads the same descriptor as raw bytes instead of decoded text. Binary
/// views of the output streams are not modeled.
fn buffer<'s>(runtime: &mut dyn PyRuntime<'s>, stream: Value<'s>) -> PyResult<'s> {
    let stdin = runtime.marker(PyMarker::Stdin);
    if runtime.identical(&stream, &stdin) {
        return Ok(runtime.marker(PyMarker::StdinBuffer));
    }
    Err(PyError::exception(
        "AttributeError",
        "only sys.stdin models a binary buffer",
    ))
}

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "sys",
    functions: &[
        FunctionDef {
            module: "sys",
            name: "exit",
            call: exit,
        },
        FunctionDef {
            module: "sys",
            name: "_getframemodulename",
            call: getframemodulename,
        },
    ],
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
        // The modeled interpreter is a 64-bit build.
        ValueDef::Constant {
            name: "maxsize",
            value: PyConstant::Int(i64::MAX),
        },
        ValueDef::Factory {
            name: "argv",
            get: argv,
        },
        ValueDef::Factory {
            name: "path",
            get: path,
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

fn exit<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("sys.exit", 0, 1)?;
    args.reject_keywords("sys.exit")?;
    let status = match args.positional().first() {
        None => 0,
        Some(value) if value.is_none() => 0,
        Some(value) => runtime
            .int_value(value)
            .and_then(|status| i32::try_from(status).ok())
            .unwrap_or(1),
    };
    Err(PyError::exit(status))
}

/// `sys._getframemodulename(depth=0)`: the module name of the caller `depth` frames up, which
/// `collections.namedtuple` uses to set `__module__` on the classes it creates.
fn getframemodulename<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    const NAME: &str = "_getframemodulename";
    if let Some((name, _)) = args.keywords().iter().find(|(name, _)| name != "depth") {
        return Err(PyError::type_error(format!(
            "{NAME}() got an unexpected keyword argument '{name}'"
        )));
    }
    let depth = match (args.positional(), args.keyword(NAME, "depth")?) {
        ([], None) => 0,
        ([value], None) | ([], Some(value)) => match runtime.int_value(value) {
            Some(depth) => depth,
            None => {
                return Err(PyError::type_error(format!(
                    "'{}' object cannot be interpreted as an integer",
                    runtime.type_name(value)?
                )))
            }
        },
        (positional, keyword) => {
            let given = positional.len() + usize::from(keyword.is_some());
            return Err(PyError::type_error(format!(
                "{NAME}() takes at most 1 argument ({given} given)"
            )));
        }
    };
    // CPython reads a negative depth as the innermost frame.
    let depth = usize::try_from(depth.max(0)).unwrap_or(usize::MAX);
    Ok(runtime.frame_module_name(depth)?.unwrap_or(Value::None))
}

fn argv<'s>(runtime: &mut dyn PyRuntime<'s>) -> PyResult<'s> {
    runtime.new_argv()
}

fn path<'s>(runtime: &mut dyn PyRuntime<'s>) -> PyResult<'s> {
    runtime.new_import_path()
}

fn stdin<'s>(runtime: &mut dyn PyRuntime<'s>) -> PyResult<'s> {
    Ok(runtime.marker(PyMarker::Stdin))
}

fn stdout<'s>(runtime: &mut dyn PyRuntime<'s>) -> PyResult<'s> {
    Ok(runtime.marker(PyMarker::Stdout))
}

fn stderr<'s>(runtime: &mut dyn PyRuntime<'s>) -> PyResult<'s> {
    Ok(runtime.marker(PyMarker::Stderr))
}

fn write<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("stream.write", 1, 1)?;
    args.reject_keywords("stream.write")?;
    let text = runtime.display(&args.positional()[0])?;
    let written = runtime.write_stream(&receiver, &text)?;
    Ok(Value::Int(i64::try_from(written).unwrap_or(i64::MAX)))
}

fn read<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    read_inner(runtime, receiver, args, false)
}

fn readline<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    read_inner(runtime, receiver, args, true)
}

/// Read every remaining line eagerly. Bounded by the same memory accounting as `read()`, since a
/// caller that wants line-by-line backpressure should iterate the stream instead.
fn readlines<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("stream.readlines", 0, 1)?;
    args.reject_keywords("stream.readlines")?;
    let mut lines = Vec::new();
    loop {
        let read = runtime.read_stream(&receiver, None, true)?;
        if read.is_empty() {
            break;
        }
        lines.push(package_read(runtime, read)?);
    }
    runtime.new_list(lines)
}

fn iter<'s>(
    _runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("stream.__iter__", 0, 0)?;
    args.reject_keywords("stream.__iter__")?;
    Ok(slot_iter(_runtime, receiver)?.expect("stream iteration always returns itself"))
}

fn next<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("stream.__next__", 0, 0)?;
    args.reject_keywords("stream.__next__")?;
    slot_next(runtime, receiver)?.ok_or_else(|| PyError::exception("StopIteration", ""))
}

pub(crate) fn slot_iter<'s>(
    _runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
) -> PyResult<'s, Option<Value<'s>>> {
    Ok(Some(receiver))
}

pub(crate) fn slot_next<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
) -> PyResult<'s, Option<Value<'s>>> {
    let read = runtime.read_stream(&receiver, None, true)?;
    if read.is_empty() {
        Err(PyError::exception("StopIteration", ""))
    } else {
        package_read(runtime, read).map(Some)
    }
}

fn read_inner<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
    line: bool,
) -> PyResult<'s> {
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
    let read = runtime.read_stream(&receiver, size, line)?;
    package_read(runtime, read)
}

/// Package one stream read into its Python representation: `str` for text, `bytes` for `.buffer`.
fn package_read<'s>(runtime: &mut dyn PyRuntime<'s>, read: PyStreamRead) -> PyResult<'s> {
    match read {
        PyStreamRead::Text(text) => runtime.new_string(text),
        PyStreamRead::Bytes(bytes) => runtime.new_bytes(bytes),
    }
}
