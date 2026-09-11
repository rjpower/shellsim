//! Type markers for the bounded :mod:`unittest` compatibility surface.

use super::super::native::{ModuleDef, PyMarker, PyResult, PyRuntime, ValueDef};

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
