//! Static markers for the deliberately small :mod:`typing` compatibility surface.
//!
//! These values affect only Python syntax and representation. They do not consult host typing
//! state or perform runtime type checking.

use super::super::native::{ModuleDef, PyConstant, PyMarker, PyResult, PyRuntime, ValueDef};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "typing",
    functions: &[],
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
    ],
};

fn list(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.marker(PyMarker::TypingList))
}
