//! Class decorators for the bounded :mod:`dataclasses` compatibility surface.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyClass, PyError, PyResult, PyRuntime, PyValueCast,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "dataclasses",
    functions: &[
        FunctionDef {
            module: "dataclasses",
            name: "dataclass",
            call: dataclass,
        },
        FunctionDef {
            module: "dataclasses",
            name: "fields",
            call: fields,
        },
    ],
    values: &[],
};

fn dataclass(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("dataclass", 1, 1)?;
    args.reject_keywords("dataclass")?;
    let value = args.positional()[0].clone();
    let class = value.clone().cast::<PyClass>(runtime)?;
    runtime.mark_dataclass(class)?;
    Ok(value)
}

fn fields(_runtime: &mut dyn PyRuntime, _args: CallArgs) -> PyResult {
    Err(PyError::runtime_error(
        "dataclasses.fields is not implemented",
    ))
}
