//! Native capability boundary for VFS-only dynamic Python module loading.
//!
//! Public ``importlib.util`` objects live in the frozen Python facade. These two primitives only
//! allocate an interpreter module and execute a bounded VFS source file in its namespace.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyModule, PyResult, PyRuntime, PyString, PyValueCast,
};
use super::super::Value;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_importlib",
    functions: &[
        FunctionDef {
            module: "_importlib",
            name: "new_module",
            call: new_module,
        },
        FunctionDef {
            module: "_importlib",
            name: "exec_module",
            call: exec_module,
        },
    ],
    values: &[],
};

fn new_module(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_importlib.new_module", 4, 4)?;
    args.reject_keywords("_importlib.new_module")?;
    let PyString(name) = args.positional()[0].cast(runtime)?;
    let PyString(path) = args.positional()[1].cast(runtime)?;
    runtime.new_module(name, path, args.positional()[2], args.positional()[3])
}

fn exec_module(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_importlib.exec_module", 2, 2)?;
    args.reject_keywords("_importlib.exec_module")?;
    let module = args.positional()[0].cast::<PyModule>(runtime)?;
    let PyString(path) = args.positional()[1].cast(runtime)?;
    runtime.exec_module(module, &path)?;
    Ok(Value::None)
}
