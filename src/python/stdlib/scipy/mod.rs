//! shellsim's SciPy: native kernels behind the frozen `scipy` package.
//!
//! The Python-visible package is frozen source (`source/scipy/`). Its submodules star-import
//! the native modules registered in [`native_module`] and add the parts SciPy writes in Python.
//! Arrays, dtypes and ufunc dispatch come from shellsim's NumPy, so SciPy's ufuncs are ordinary
//! `numpy.ufunc` values.
//!
//! Like the NumPy core, this module reaches no host capability, and every kernel charges CPU
//! through the ufunc loop that runs it.

pub(in crate::python) mod special;

use super::super::native::ModuleDef;

/// The native module `name`, for names starting with `_scipy`.
///
/// `scipy.linalg` has no native module of its own: it is built entirely in Python over
/// `_numpy_linalg`'s primitives (see `source/scipy/linalg.py`), so there is nothing to register
/// here for it.
pub(in crate::python) fn native_module(name: &str) -> Option<&'static ModuleDef> {
    Some(match name {
        "_scipy_special" => &special::MODULE,
        _ => return None,
    })
}
