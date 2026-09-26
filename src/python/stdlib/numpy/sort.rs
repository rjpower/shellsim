//! Sorting, searching, and set operations: `sort`, `argsort`, `unique`, `searchsorted`,
//! `where`, `nonzero`, set membership, `bincount`, `histogram`, and `digitize`.
//!
//! Functions are exported through the native module `_numpy_sort`, which the frozen `numpy`
//! package re-exports.

use super::super::super::native::{FunctionDef, ModuleDef, NativeTypeDef};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_sort",
    functions: FUNCTIONS,
    values: &[],
};

static FUNCTIONS: &[FunctionDef] = &[];

/// Methods this area installs on `numpy.ndarray`.
pub(in crate::python) static ARRAY_METHODS: NativeTypeDef = NativeTypeDef {
    name: "numpy.ndarray",
    methods: &[],
    getters: &[],
};
