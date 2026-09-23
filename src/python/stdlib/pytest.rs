//! Minimal pytest control functions implemented over structured Python exceptions.

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyError, PyExceptionType,
    PyRaisesContext, PyResult, PyRuntime, PyValueCast,
};

pub(crate) static RAISES_CONTEXT_TYPE: NativeTypeDef = NativeTypeDef {
    name: "pytest.raises",
    methods: &[
        MethodDef {
            type_name: "pytest.raises",
            name: "__enter__",
            call: raises_enter,
        },
        MethodDef {
            type_name: "pytest.raises",
            name: "__exit__",
            call: raises_exit,
        },
    ],
};

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
        FunctionDef {
            module: "_pytest",
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
    let PyExceptionType(expected) = args.positional()[0].cast(runtime)?;
    runtime.new_raises_context(expected.to_string())
}

fn raises_enter(_runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("pytest.raises.__enter__", 0, 0)?;
    args.reject_keywords("pytest.raises.__enter__")?;
    Ok(receiver)
}

fn raises_exit(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("pytest.raises.__exit__", 3, 3)?;
    args.reject_keywords("pytest.raises.__exit__")?;
    let context = receiver.cast::<PyRaisesContext>(runtime)?;
    let expected = runtime.raises_expected(context)?;
    if args.positional()[0].is_none() {
        return Err(PyError::exception("Failed", "DID NOT RAISE"));
    }
    let kind = runtime
        .exception_type_name(&args.positional()[0])
        .ok_or_else(|| PyError::type_error("invalid exception context"))?;
    Ok(Value::Bool(
        expected == "Exception" || expected == "BaseException" || kind == expected,
    ))
}
