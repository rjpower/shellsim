//! Minimal pytest control functions implemented over structured Python exceptions.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyExceptionType, PyResult, PyRuntime, PyValueCast,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "pytest",
    functions: &[
        FunctionDef {
            module: "pytest",
            name: "fail",
            call: fail,
        },
        FunctionDef {
            module: "pytest",
            name: "skip",
            call: skip,
        },
        FunctionDef {
            module: "pytest",
            name: "raises",
            call: raises,
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

fn raises(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("pytest.raises", 1, 1)?;
    args.reject_keywords("pytest.raises")?;
    let PyExceptionType(expected) = args.positional()[0].clone().cast(runtime)?;
    runtime.new_raises_context(expected.to_string())
}
