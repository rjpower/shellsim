//! Static markers for the deliberately small :mod:`typing` compatibility surface.
//!
//! These values affect only Python syntax and representation. They do not consult host typing
//! state or perform runtime type checking.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyConstant, PyMarker, PyResult, PyRuntime, ValueDef,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "typing",
    functions: &[
        FunctionDef {
            module: "typing",
            name: "TypeVar",
            call: type_var,
        },
        FunctionDef {
            module: "typing",
            name: "cast",
            call: cast,
        },
        FunctionDef {
            module: "typing",
            name: "overload",
            call: identity_decorator,
        },
        FunctionDef {
            module: "typing",
            name: "runtime_checkable",
            call: identity_decorator,
        },
        FunctionDef {
            module: "typing",
            name: "get_origin",
            call: get_origin,
        },
        FunctionDef {
            module: "typing",
            name: "get_args",
            call: get_args,
        },
    ],
    values: &[
        ValueDef::Factory {
            name: "List",
            get: list,
        },
        ValueDef::Constant {
            name: "Any",
            value: PyConstant::String("typing.Any"),
        },
        ValueDef::Constant {
            name: "Optional",
            value: PyConstant::String("typing.Optional"),
        },
        ValueDef::Constant {
            name: "Dict",
            value: PyConstant::String("typing.Dict"),
        },
        ValueDef::Constant {
            name: "Tuple",
            value: PyConstant::String("typing.Tuple"),
        },
        ValueDef::Constant {
            name: "Set",
            value: PyConstant::String("typing.Set"),
        },
        ValueDef::Constant {
            name: "Callable",
            value: PyConstant::String("typing.Callable"),
        },
        typing_value("Annotated"),
        typing_value("ClassVar"),
        typing_value("Final"),
        typing_value("Generic"),
        typing_value("Generator"),
        typing_value("Iterable"),
        typing_value("Iterator"),
        typing_value("Literal"),
        typing_value("Mapping"),
        typing_value("MutableMapping"),
        typing_value("Never"),
        typing_value("Protocol"),
        typing_value("Self"),
        typing_value("Sequence"),
        typing_value("Type"),
        typing_value("TypeAlias"),
        typing_value("Union"),
    ],
};

const fn typing_value(name: &'static str) -> ValueDef {
    ValueDef::Constant {
        name,
        value: PyConstant::String(name),
    }
}

fn list(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.marker(PyMarker::TypingList))
}

fn type_var(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("typing.TypeVar", 1, usize::MAX)?;
    let name = runtime
        .string_value(&args.positional()[0])?
        .ok_or_else(|| {
            super::super::native::PyError::type_error("TypeVar name must be a string")
        })?;
    runtime.new_string(format!("~{name}"))
}

fn cast(_runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("typing.cast", 2, 2)?;
    args.reject_keywords("typing.cast")?;
    Ok(args.positional()[1])
}

fn identity_decorator(_runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("typing decorator", 1, 1)?;
    args.reject_keywords("typing decorator")?;
    Ok(args.positional()[0])
}

fn get_origin(_runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("typing.get_origin", 1, 1)?;
    args.reject_keywords("typing.get_origin")?;
    Ok(super::super::native::PyValue::None)
}

fn get_args(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("typing.get_args", 1, 1)?;
    args.reject_keywords("typing.get_args")?;
    runtime.new_tuple(Vec::new())
}
