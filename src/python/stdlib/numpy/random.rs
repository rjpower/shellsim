//! Bit generators and distributions behind `numpy.random`.
//!
//! Functions are exported through the native module `_numpy_random`, which the frozen `numpy`
//! package re-exports.

use super::super::super::native::{FunctionDef, ModuleDef};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_random",
    functions: FUNCTIONS,
    values: &[],
};

static FUNCTIONS: &[FunctionDef] = &[];
