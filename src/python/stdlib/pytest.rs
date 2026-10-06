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
            name: "_set_timeout",
            call: set_timeout,
        },
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

fn set_timeout(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    use super::super::native::PyValueCast;
    use super::super::number::PyNumber;
    args.expect_positional("_pytest._set_timeout", 1, 1)?;
    args.reject_keywords("_pytest._set_timeout")?;
    let seconds = args.positional()[0].cast::<PyNumber>(runtime)?.into_f64()?;
    let nanos = seconds * 1_000_000_000.0;
    if !nanos.is_finite() || nanos < 0.0 || nanos >= u64::MAX as f64 {
        return Err(PyError::value_error(
            "timeout must be finite and non-negative",
        ));
    }
    runtime.set_test_timeout((nanos > 0.0).then_some((nanos as u64).max(1)));
    Ok(super::super::Value::None)
}

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
