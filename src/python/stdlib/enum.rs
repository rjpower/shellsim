//! The registered Enum class for the bounded :mod:`enum` compatibility surface.

use super::super::native::{ModuleDef, PyMarker, PyResult, PyRuntime, ValueDef};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "enum",
    functions: &[],
    values: &[ValueDef::Factory {
        name: "Enum",
        get: enum_base,
    }],
};

fn enum_base<'s>(runtime: &mut dyn PyRuntime<'s>) -> PyResult<'s> {
    Ok(runtime.marker(PyMarker::EnumType))
}
