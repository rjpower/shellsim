//! `.npy`/`.npz` persistence and text I/O through the virtual filesystem.
//!
//! Functions are exported through the native module `_numpy_io`, which the frozen `numpy`
//! package re-exports.

use super::super::super::native::{FunctionDef, ModuleDef, NativeTypeDef};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_io",
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
