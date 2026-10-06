//! Private call-stack lookup for the frozen `warnings` module.
//!
//! `warnings.warn(stacklevel=n)` attributes a warning to the line executing `n` Python frames
//! above its own. Frames expose only that line and the script path, which is also what tracebacks
//! report; no frame object or local variable crosses this boundary.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyIndex, PyResult, PyRuntime, PyValueCast,
};
use super::super::Value;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_shellsim_warnings",
    functions: &[FunctionDef {
        module: "_shellsim_warnings",
        name: "caller",
        call: caller,
    }],
    values: &[],
};

/// `caller(stacklevel)`: the `(filename, lineno)` executing `stacklevel` frames above the Python
/// function that calls this helper, or `None` when the stack is not that deep.
fn caller(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("caller", 1, 1)?;
    args.reject_keywords("caller")?;
    let PyIndex(stacklevel) = args.positional()[0].cast(runtime)?;
    let depth = usize::try_from(stacklevel.max(1)).unwrap_or(usize::MAX);
    let Some((filename, line)) = runtime.caller_location(depth) else {
        return Ok(Value::None);
    };
    let filename = runtime.new_string(filename)?;
    let line = Value::Int(i64::from(line));
    runtime.new_tuple(vec![filename, line])
}
