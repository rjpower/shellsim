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
        MethodDef {
            type_name: "shellsim.stream",
            name: "flush",
            call: flush,
        },
        MethodDef {
            type_name: "shellsim.stream",
            name: "fileno",
            call: fileno,
        },
        MethodDef {
            type_name: "shellsim.stream",
            name: "isatty",
            call: isatty,
        },
        MethodDef {
            type_name: "shellsim.stream",
            name: "readable",
            call: readable,
        },
        MethodDef {
            type_name: "shellsim.stream",
            name: "writable",
            call: writable,
        },
        MethodDef {
            type_name: "shellsim.stream",
            name: "seekable",
            call: seekable,
        },
    ],
    getters: &[
        GetterDef {
            owner: "shellsim.stream",
            name: "buffer",
            get: buffer,
        },
        GetterDef {
            owner: "shellsim.stream",
            name: "encoding",
            get: encoding,
        },
        GetterDef {
            owner: "shellsim.stream",
            name: "errors",
            get: errors,
        },
        GetterDef {
            owner: "shellsim.stream",
            name: "closed",
            get: closed,
        },
        GetterDef {
            owner: "shellsim.stream",
            name: "name",
            get: name,
        },
        GetterDef {
            owner: "shellsim.stream",
            name: "mode",
            get: mode,
        },
    ],
};

/// The descriptor number of a standard stream: 0 for stdin, 1 for stdout, 2 for stderr.
fn descriptor<'s>(runtime: &mut dyn PyRuntime<'s>, stream: &Value<'s>) -> i64 {
    for (marker, number) in [
        (PyMarker::Stdin, 0),
        (PyMarker::StdinBuffer, 0),
        (PyMarker::Stdout, 1),
        (PyMarker::Stderr, 2),
    ] {
        let candidate = runtime.marker(marker);
        if runtime.identical(stream, &candidate) {
            return number;
        }
    }
    -1
}

/// Output is written through as it is produced, so `flush` has nothing left to do.
fn flush<'s>(runtime: &mut dyn PyRuntime<'s>, _: Value<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("stream.flush", 0, 0)?;
    args.reject_keywords("stream.flush")?;
    let _ = runtime;
    Ok(Value::None)
}

fn fileno<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("stream.fileno", 0, 0)?;
    args.reject_keywords("stream.fileno")?;
    Ok(Value::Int(descriptor(runtime, &receiver)))
}

/// The modeled streams are pipes, never terminals.
fn isatty<'s>(runtime: &mut dyn PyRuntime<'s>, _: Value<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("stream.isatty", 0, 0)?;
    args.reject_keywords("stream.isatty")?;
    let _ = runtime;
    Ok(Value::Bool(false))
}

fn readable<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("stream.readable", 0, 0)?;
    args.reject_keywords("stream.readable")?;
    Ok(Value::Bool(descriptor(runtime, &receiver) == 0))
}

fn writable<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("stream.writable", 0, 0)?;
    args.reject_keywords("stream.writable")?;
    Ok(Value::Bool(descriptor(runtime, &receiver) > 0))
}

fn seekable<'s>(runtime: &mut dyn PyRuntime<'s>, _: Value<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("stream.seekable", 0, 0)?;
    args.reject_keywords("stream.seekable")?;
    let _ = runtime;
    Ok(Value::Bool(false))
}

fn encoding<'s>(runtime: &mut dyn PyRuntime<'s>, _: Value<'s>) -> PyResult<'s> {
    runtime.new_string("utf-8".to_string())
}

fn errors<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: Value<'s>) -> PyResult<'s> {
    // CPython's stderr uses backslashreplace so diagnostics never fail to encode.
    let errors = if descriptor(runtime, &receiver) == 2 {
        "backslashreplace"
    } else {
        "strict"
    };
    runtime.new_string(errors.to_string())
}

fn closed<'s>(_: &mut dyn PyRuntime<'s>, _: Value<'s>) -> PyResult<'s> {
    Ok(Value::Bool(false))
}

fn name<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: Value<'s>) -> PyResult<'s> {
    let name = match descriptor(runtime, &receiver) {
        0 => "<stdin>",
        1 => "<stdout>",
        _ => "<stderr>",
    };
    runtime.new_string(name.to_string())
}

fn mode<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: Value<'s>) -> PyResult<'s> {
    let mode = if descriptor(runtime, &receiver) == 0 {
        "r"
    } else {
        "w"
    };
    runtime.new_string(mode.to_string())
}

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
    name: "_sys",
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
        FunctionDef {
            module: "sys",
            name: "active_exception",
            call: active_exception,
        },
        FunctionDef {
            module: "sys",
            name: "modules",
            call: modules,
        },
        FunctionDef {
            module: "sys",
            name: "set_module",
            call: set_module,
        },
        FunctionDef {
            module: "sys",
            name: "remove_module",
            call: remove_module,
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
/// `_sys.active_exception()`: the exception being handled, or None outside an `except` block.
fn active_exception<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("sys.exception", 0, 0)?;
    args.reject_keywords("sys.exception")?;
    Ok(runtime.active_exception().unwrap_or(Value::None))
}

/// `_sys.set_module(name, module)`: register `module` so `import name` returns it.
fn set_module<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("sys.modules.__setitem__", 2, 2)?;
    args.reject_keywords("sys.modules.__setitem__")?;
    let name = runtime
        .string_value(&args.positional()[0])?
        .ok_or_else(|| PyError::type_error("module names must be strings"))?;
    runtime.register_module(&name, args.positional()[1])?;
    Ok(Value::None)
}

/// `_sys.remove_module(name)`: forget a registered module; returns whether one was registered.
fn remove_module<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("sys.modules.__delitem__", 1, 1)?;
    args.reject_keywords("sys.modules.__delitem__")?;
    let name = runtime
        .string_value(&args.positional()[0])?
        .ok_or_else(|| PyError::type_error("module names must be strings"))?;
    Ok(Value::Bool(runtime.unregister_module(&name)))
}

/// `_sys.modules()`: a dict snapshot of the modules loaded so far.
fn modules<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("sys.modules", 0, 0)?;
    args.reject_keywords("sys.modules")?;
    runtime.loaded_modules()
}

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
