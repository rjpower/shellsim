//! VM primitives used by the capability-free `asyncio` compatibility module.
//!
//! Scheduling policy stays in frozen Python. This module exposes only bounded coroutine frame
//! stepping so the facade cannot acquire host threads, clocks, or I/O capabilities.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyIterator, PyKind, PyResult, PyRuntime, PyValueCast,
};
use super::super::Value;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_asyncio",
    functions: &[
        FunctionDef {
            module: "_asyncio",
            name: "_step",
            call: step,
        },
        FunctionDef {
            module: "_asyncio",
            name: "_is_coroutine",
            call: is_coroutine,
        },
    ],
    values: &[],
};

fn step(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_asyncio._step", 2, 2)?;
    args.reject_keywords("_asyncio._step")?;
    let coroutine = args.positional()[0];
    if runtime.kind(&coroutine)? != PyKind::Generator {
        return Err(PyError::type_error("expected a coroutine"));
    }
    let iterator = coroutine.cast::<PyIterator>(runtime)?;
    let sent = args.positional()[1];
    let (status, value) = runtime.coroutine_step(iterator, sent)?;
    runtime.new_tuple(vec![Value::Int(i64::from(status)), value])
}

fn is_coroutine(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_asyncio._is_coroutine", 1, 1)?;
    args.reject_keywords("_asyncio._is_coroutine")?;
    Ok(Value::Bool(
        runtime.kind(&args.positional()[0])? == PyKind::Generator,
    ))
}
