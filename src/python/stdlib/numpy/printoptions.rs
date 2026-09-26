//! Print options and formatting functions: `set_printoptions`, `get_printoptions`,
//! `array2string`, `array_repr`, and `array_str`.
//!
//! Functions are exported through the native module `_numpy_print`, which the frozen `numpy`
//! package re-exports.

use super::super::super::native::{FunctionDef, ModuleDef};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_print",
    functions: FUNCTIONS,
    values: &[],
};

static FUNCTIONS: &[FunctionDef] = &[];
