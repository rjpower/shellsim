//! Static interpreter metadata and invocation values for the modeled Python process.

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, MethodDef, ModuleDef, NativeTypeDef, PyConstant, PyMarker, PyResult, PyRuntime,
    ValueDef,
};

pub(crate) static STREAM_TYPE: NativeTypeDef = NativeTypeDef {
    name: "shellsim.stream",
    methods: &[MethodDef {
        type_name: "shellsim.stream",
        name: "write",
        call: write,
    }],
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "sys",
    functions: &[],
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
            name: "stdout",
            get: stdout,
        },
        ValueDef::Factory {
            name: "stderr",
            get: stderr,
        },
    ],
};

fn argv(runtime: &mut dyn PyRuntime) -> PyResult {
    runtime.new_argv()
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
