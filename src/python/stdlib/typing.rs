//! Static markers for the deliberately small :mod:`typing` compatibility surface.
//!
//! These values affect only Python syntax and representation. They do not consult host typing
//! state or perform runtime type checking.

use super::super::native::{ModuleDef, PyMarker, PyResult, PyRuntime, ValueDef};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "typing",
    functions: &[],
    values: &[ValueDef::Factory {
        name: "List",
        get: list,
    }],
};

fn list(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.marker(PyMarker::TypingList))
}
