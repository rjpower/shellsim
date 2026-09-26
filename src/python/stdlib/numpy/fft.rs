//! Kernels behind `numpy.fft`.
//!
//! Functions are exported through the native module `_numpy_fft`, which the frozen `numpy`
//! package re-exports.

use super::super::super::native::{FunctionDef, ModuleDef};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_fft",
    functions: FUNCTIONS,
    values: &[],
};

static FUNCTIONS: &[FunctionDef] = &[];
