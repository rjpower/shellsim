//! Native helpers for the frozen `operator` module, mirroring CPython's `_operator`.
//!
//! `operator.index` needs the interpreter's own `__index__` protocol and type names: CPython
//! names extension types by their dotted `tp_name` (`'numpy.float64' object cannot be
//! interpreted as an integer`), which Python-level `type(a).__name__` cannot reproduce.

use super::super::native::{CallArgs, FunctionDef, ModuleDef, PyError, PyResult, PyRuntime};
use super::super::number::NumberRef;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_operator",
    functions: &[FunctionDef {
        module: "_operator",
        name: "index",
        call: index,
    }],
    values: &[],
};

/// `operator.index(a)`: `a` as an exact `int`, through `__index__` for other types.
fn index<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("index", 1, 1)?;
    args.reject_keywords("index")?;
    let value = args.positional()[0];
    if let Some(integer) = runtime.int_value(&value) {
        return Ok(super::super::Value::Int(integer));
    }
    if matches!(runtime.number(&value), Some(NumberRef::BigInt(_))) {
        return Ok(value);
    }
    let Some(method) = runtime.get_attribute(value, "__index__")? else {
        let name = runtime.type_name(&value)?;
        return Err(PyError::type_error(format!(
            "'{name}' object cannot be interpreted as an integer"
        )));
    };
    let result = runtime.call_value(method, CallArgs::new(Vec::new(), Vec::new()))?;
    if runtime.int_value(&result).is_some()
        || matches!(runtime.number(&result), Some(NumberRef::BigInt(_)))
    {
        return Ok(result);
    }
    let name = runtime.type_name(&result)?;
    Err(PyError::type_error(format!(
        "__index__ returned non-int (type {name})"
    )))
}
