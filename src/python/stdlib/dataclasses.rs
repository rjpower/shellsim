//! Native class marker used by the source-level :mod:`dataclasses` compatibility surface.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyClass, PyResult, PyRuntime, PyValueCast,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_dataclasses",
    functions: &[FunctionDef {
        module: "_dataclasses",
        name: "mark_dataclass",
        call: mark_dataclass,
    }],
    values: &[],
};

fn mark_dataclass(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("mark_dataclass", 1, 1)?;
    args.reject_keywords("mark_dataclass")?;
    let value = args.positional()[0];
    let class = value.cast::<PyClass>(runtime)?;
    runtime.mark_dataclass(class)?;
    Ok(value)
}
