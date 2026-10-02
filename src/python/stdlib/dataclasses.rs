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

fn dataclass<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("dataclass", 1, 1)?;
    args.reject_keywords("dataclass")?;
    let value = args.positional()[0];
    let class = value.cast::<PyClass>(runtime)?;
    runtime.mark_dataclass(class)?;
    Ok(value)
}

fn fields<'s>(_runtime: &mut dyn PyRuntime<'s>, _args: CallArgs<'s>) -> PyResult<'s> {
    Err(PyError::runtime_error(
        "dataclasses.fields is not implemented",
    ))
}
