//! Kernels behind `numpy.linalg`: norms, solves, inverses, determinants, decompositions, and
//! eigenvalues.
//!
//! Functions are exported through the native module `_numpy_linalg`, which the frozen `numpy`
//! package re-exports.

use super::super::super::native::{FunctionDef, ModuleDef};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_linalg",
    functions: FUNCTIONS,
    values: &[],
};

static FUNCTIONS: &[FunctionDef] = &[];
