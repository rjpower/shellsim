//! Environment access backed only by shellsim's modeled process-environment capability.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyMarker, PyResult, PyRuntime, PyString, PyValueCast,
    ValueDef,
};
use super::super::Value;

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
    let PyString(name) = args.positional()[0].clone().cast(runtime)?;
    Ok(runtime
        .environment()
        .get(&name)
        .map(Value::String)
        .unwrap_or_else(|| args.positional().get(1).cloned().unwrap_or(Value::None)))
}

fn environ(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.marker(PyMarker::Environment))
}
