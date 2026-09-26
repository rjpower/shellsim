//! Elementwise helpers built on ufuncs: `isclose`, `allclose`, `array_equal`, `round`, `clip`,
//! `real`/`imag`/`angle`, `nan_to_num`, `diff`, `gradient`, `interp`, polynomials, `convolve`,
//! `cov`, and `corrcoef`.
//!
//! Functions are exported through the native module `_numpy_math`, which the frozen `numpy`
//! package re-exports.

use super::super::super::native::{FunctionDef, ModuleDef, NativeTypeDef};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_math",
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
