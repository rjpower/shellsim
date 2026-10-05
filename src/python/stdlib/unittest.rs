//! Bounded assertion methods for the capability-free :mod:`unittest` compatibility surface.

use super::super::exception_types;
use super::super::heap::{NativeObject, Ref};
use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, MethodDef, ModuleDef, NativeTypeDef, PyError, PyExceptionType, PyMarker,
    PyRaisesContext, PyResult, PyRuntime, PyValueCast, ValueDef,
};
use super::super::object_model::{BuiltinType, TypeId};

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

fn test_case<'s>(runtime: &mut dyn PyRuntime<'s>) -> PyResult<'s> {
    Ok(runtime.marker(PyMarker::TestCaseType))
}

fn assert_equal<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    _receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
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

fn assert_true<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    assert_truth(runtime, receiver, args, true)
}

fn assert_false<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    assert_truth(runtime, receiver, args, false)
}

fn assert_truth<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    _receiver: Value<'s>,
    args: CallArgs<'s>,
    expected: bool,
) -> PyResult<'s> {
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

fn assert_is_none<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    _receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("assertIsNone", 1, 2)?;
    args.reject_keywords("assertIsNone")?;
    if args.positional()[0].is_none() {
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

fn assert_raises<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    _receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("assertRaises", 1, 1)?;
    args.reject_keywords("assertRaises")?;
    let PyExceptionType(expected) = args.positional()[0].cast(runtime)?;
    runtime.new_raises_context(expected.to_string())
}

/// A `pytest.raises(...)` context manager as the heap stores it: the expected exception name.
#[derive(Debug)]
pub(crate) struct RaisesContextObject {
    pub expected: String,
}

impl NativeObject for RaisesContextObject {
    fn python_type(&self) -> TypeId {
        BuiltinType::RaisesContext.id()
    }

    fn modeled_bytes(&self) -> Result<u64, String> {
        u64::try_from(self.expected.len()).map_err(|_| "modeled object size overflow".into())
    }

    fn dup(&self) -> Box<dyn NativeObject> {
        Box::new(Self {
            expected: self.expected.clone(),
        })
    }

    fn repr(&self, _: &mut dyn FnMut(&Ref) -> Result<String, String>) -> Result<String, String> {
        Ok("<pytest.raises>".into())
    }
}

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
    getters: &[],
};

fn raises_enter<'s>(
    _runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("pytest.raises.__enter__", 0, 0)?;
    args.reject_keywords("pytest.raises.__enter__")?;
    Ok(receiver)
}

fn raises_exit<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
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
    Ok(Value::Bool(exception_types::exception_is_subclass(
        kind, &expected,
    )))
}
