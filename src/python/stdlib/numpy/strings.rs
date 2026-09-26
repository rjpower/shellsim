//! String operations behind `numpy.strings` and `numpy.char`.
//!
//! Functions are exported through the native module `_numpy_strings`, which the frozen `numpy`
//! package re-exports.

use super::super::super::native::{FunctionDef, ModuleDef};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_strings",
    functions: FUNCTIONS,
    values: &[],
};

static FUNCTIONS: &[FunctionDef] = &[];
