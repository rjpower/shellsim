//! Minimal pytest control functions implemented over structured Python exceptions.
//!
//! `pytest.raises` lives in the frozen `pytest` module so it can use `issubclass` and expose the
//! caught exception.

use super::super::native::{CallArgs, FunctionDef, ModuleDef, PyError, PyResult, PyRuntime};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_pytest",
    functions: &[
        FunctionDef {
            module: "_pytest",
            name: "fail",
            call: fail,
        },
        FunctionDef {
            module: "_pytest",
            name: "skip",
            call: skip,
        },
    ],
    values: &[],
};

fn fail(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    control_error(runtime, args, "pytest.fail", "Failed")
}

fn skip(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    control_error(runtime, args, "pytest.skip", "Skipped")
}

fn control_error(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    function: &str,
    kind: &'static str,
) -> PyResult {
    args.expect_positional(function, 0, 1)?;
    args.reject_keywords(function)?;
    let message = args
        .positional()
        .first()
        .map(|value| runtime.display(value))
        .transpose()?
        .unwrap_or_default();
    Err(PyError::exception(kind, message))
}
