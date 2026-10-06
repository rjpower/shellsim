//! Native capability boundary for VFS-only dynamic Python module loading.
//!
//! Public ``importlib`` objects live in the frozen Python facade. These primitives import a module
//! by its absolute name, allocate an interpreter module, and execute a bounded VFS source file in
//! its namespace.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, OwnedPyString, PyModule, PyResult, PyRuntime, PyValueCast,
};
use super::super::Value;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_importlib",
    functions: &[
        FunctionDef {
            module: "_importlib",
            name: "import_module",
            call: import_module,
        },
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

/// `_importlib.import_module(name)`: import the module with absolute dotted `name` through the
/// ordinary importer and return that module itself, not its top-level package.
fn import_module(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_importlib.import_module", 1, 1)?;
    args.reject_keywords("_importlib.import_module")?;
    let OwnedPyString(name) = args.positional()[0].cast(runtime)?;
    runtime.import_module(&name)
}

fn new_module(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_importlib.new_module", 4, 4)?;
    args.reject_keywords("_importlib.new_module")?;
    let OwnedPyString(name) = args.positional()[0].cast(runtime)?;
    let OwnedPyString(path) = args.positional()[1].cast(runtime)?;
    runtime.new_module(name, path, args.positional()[2], args.positional()[3])
}

fn exec_module(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_importlib.exec_module", 2, 2)?;
    args.reject_keywords("_importlib.exec_module")?;
    let module = args.positional()[0].cast::<PyModule>(runtime)?;
    let OwnedPyString(path) = args.positional()[1].cast(runtime)?;
    runtime.exec_module(module, &path)?;
    Ok(Value::None)
}
