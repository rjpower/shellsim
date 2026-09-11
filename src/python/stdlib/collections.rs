//! Native primitives used by the frozen :mod:`collections` implementation.
//!
//! Stateful mapping behavior remains a runtime object protocol; this module only validates the
//! public constructor and requests a capability-free object from [`PyRuntime`].

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyCallable, PyResult, PyRuntime, PyValueCast,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_collections",
    functions: &[FunctionDef {
        module: "_collections",
        name: "defaultdict",
        call: defaultdict,
    }],
    values: &[],
};

fn defaultdict(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("defaultdict", 1, 1)?;
    args.reject_keywords("defaultdict")?;
    let factory = args.positional()[0].cast::<PyCallable>(runtime)?;
    runtime.new_default_dict(factory)
}
