//! Reductions and accumulations: `sum`, `prod`, `min`/`max`, `mean`, `var`/`std`, `median`,
//! `percentile`/`quantile`, `argmin`/`argmax`, `any`/`all`, `cumsum`/`cumprod`, `count_nonzero`,
//! the `nan*` variants, and `ufunc.reduce`/`ufunc.accumulate`.
//!
//! Functions are exported through the native module `_numpy_reduce`, which the frozen `numpy`
//! package re-exports.

use super::super::super::native::{
    CallArgs, FunctionDef, ModuleDef, NativeTypeDef, PyError, PyResult, PyRuntime,
};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_reduce",
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


/// `ufunc.reduce(array, axis=0, dtype=None, out=None, keepdims=False, initial, where)`.
pub(in crate::python) fn ufunc_reduce(_runtime: &mut dyn PyRuntime, index: usize, _args: CallArgs) -> PyResult {
    Err(PyError::unsupported(format!(
        "{}.reduce is not implemented yet",
        super::ufunc::UFUNCS[index].name
    )))
}

/// `ufunc.accumulate(array, axis=0, dtype=None, out=None)`.
pub(in crate::python) fn ufunc_accumulate(_runtime: &mut dyn PyRuntime, index: usize, _args: CallArgs) -> PyResult {
    Err(PyError::unsupported(format!(
        "{}.accumulate is not implemented yet",
        super::ufunc::UFUNCS[index].name
    )))
}
