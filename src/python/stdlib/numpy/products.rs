//! Array products in the top-level namespace: `dot`, `vdot`, `inner`, `outer`, `matmul`,
//! `tensordot`, `kron`, and `cross`.
//!
//! Functions are exported through the native module `_numpy_products`, which the frozen `numpy`
//! package re-exports.

use super::super::super::native::{
    FunctionDef, ModuleDef, NativeTypeDef, PyError, PyResult, PyRuntime, PyValue,
};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_products",
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

/// `a @ b` and `np.matmul(a, b)`.
pub(in crate::python) fn matmul(
    _runtime: &mut dyn PyRuntime,
    _left: PyValue,
    _right: PyValue,
) -> PyResult {
    Err(PyError::unsupported(
        "matrix multiplication is not implemented yet",
    ))
}
