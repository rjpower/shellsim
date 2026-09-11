//! Environment access backed only by shellsim's modeled process-environment capability.

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyMarker, PyResult, PyRuntime,
    PyString, PyValueCast, ValueDef,
};

pub(crate) static ENVIRONMENT_TYPE: NativeTypeDef = NativeTypeDef {
    name: "shellsim.environment",
    methods: &[MethodDef {
        type_name: "shellsim.environment",
        name: "get",
        call: environment_get,
    }],
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "os",
    functions: &[FunctionDef {
        module: "os",
        name: "getenv",
        call: getenv,
    }],
    values: &[ValueDef::Factory {
        name: "environ",
        get: environ,
    }],
};

fn getenv(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("os.getenv", 1, 2)?;
    args.reject_keywords("os.getenv")?;
    let PyString(name) = args.positional()[0].cast(runtime)?;
    if let Some(value) = runtime.environment().get(&name) {
        runtime.new_string(value)
    } else {
        Ok(args.positional().get(1).copied().unwrap_or(Value::None))
    }
}

fn environ(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.marker(PyMarker::Environment))
}

fn environment_get(runtime: &mut dyn PyRuntime, _receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("environ.get", 1, 2)?;
    args.reject_keywords("environ.get")?;
    let PyString(name) = args.positional()[0].cast(runtime)?;
    if let Some(value) = runtime.environment().get(&name) {
        runtime.new_string(value)
    } else {
        Ok(args.positional().get(1).copied().unwrap_or(Value::None))
    }
}
