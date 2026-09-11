//! Type markers for the bounded :mod:`enum` compatibility surface.

use super::super::native::{ModuleDef, PyMarker, PyResult, PyRuntime, ValueDef};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "enum",
    functions: &[],
    values: &[ValueDef::Factory {
        name: "Enum",
        get: enum_base,
    }],
};

fn enum_base(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.marker(PyMarker::EnumBase))
}
