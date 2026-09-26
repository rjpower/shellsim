//! Bounded assertion methods for the capability-free :mod:`unittest` compatibility surface.

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, MethodDef, ModuleDef, NativeTypeDef, PyError, PyExceptionType, PyMarker, PyResult,
    PyRuntime, PyValueCast, ValueDef,
};

pub(crate) static TEST_CASE_TYPE: NativeTypeDef = NativeTypeDef {
    name: "unittest.TestCase",
    methods: &[
        MethodDef {
            type_name: "unittest.TestCase",
            name: "assertEqual",
            call: assert_equal,
        },
        MethodDef {
            type_name: "unittest.TestCase",
            name: "assertTrue",
            call: assert_true,
        },
        MethodDef {
            type_name: "unittest.TestCase",
            name: "assertFalse",
            call: assert_false,
        },
        MethodDef {
            type_name: "unittest.TestCase",
            name: "assertIsNone",
            call: assert_is_none,
        },
        MethodDef {
            type_name: "unittest.TestCase",
            name: "assertRaises",
            call: assert_raises,
        },
    ],
    getters: &[],
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "unittest",
    functions: &[],
    values: &[ValueDef::Factory {
        name: "TestCase",
        get: test_case,
    }],
};

fn test_case(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.marker(PyMarker::UnitTestBase))
}

fn assert_equal(runtime: &mut dyn PyRuntime, _receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("assertEqual", 2, 3)?;
    args.reject_keywords("assertEqual")?;
    if runtime.equals(&args.positional()[0], &args.positional()[1])? {
        return Ok(Value::None);
    }
    let message = if let Some(message) = args.positional().get(2) {
        runtime.display(message)?
    } else {
        format!(
            "{} != {}",
            runtime.repr(&args.positional()[0])?,
            runtime.repr(&args.positional()[1])?
        )
    };
    Err(PyError::exception("AssertionError", message))
}

fn assert_true(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    assert_truth(runtime, receiver, args, true)
}

fn assert_false(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    assert_truth(runtime, receiver, args, false)
}

fn assert_truth(
    runtime: &mut dyn PyRuntime,
    _receiver: Value,
    args: CallArgs,
    expected: bool,
) -> PyResult {
    let name = if expected {
        "assertTrue"
    } else {
        "assertFalse"
    };
    args.expect_positional(name, 1, 2)?;
    args.reject_keywords(name)?;
    if runtime.truth(&args.positional()[0])? == expected {
        return Ok(Value::None);
    }
    let message = args
        .positional()
        .get(1)
        .map(|value| runtime.display(value))
        .transpose()?
        .unwrap_or_else(|| {
            if expected {
                "False is not true".into()
            } else {
                "True is not false".into()
            }
        });
    Err(PyError::exception("AssertionError", message))
}

fn assert_is_none(runtime: &mut dyn PyRuntime, _receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("assertIsNone", 1, 2)?;
    args.reject_keywords("assertIsNone")?;
    if matches!(args.positional()[0], Value::None) {
        return Ok(Value::None);
    }
    let message = args
        .positional()
        .get(1)
        .map(|value| runtime.display(value))
        .transpose()?
        .unwrap_or_else(|| "value is not None".into());
    Err(PyError::exception("AssertionError", message))
}

fn assert_raises(runtime: &mut dyn PyRuntime, _receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("assertRaises", 1, 1)?;
    args.reject_keywords("assertRaises")?;
    let PyExceptionType(expected) = args.positional()[0].cast(runtime)?;
    runtime.new_raises_context(expected.to_string())
}
