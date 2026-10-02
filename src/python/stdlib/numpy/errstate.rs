//! Floating-point error reporting for ufunc loops.
//!
//! Loops record IEEE-style flags in [`FpFlags`] without calling back into Python. After a loop
//! finishes, [`report`] hands raised flags to `numpy._errors._report`, which applies the modes
//! set by `np.errstate`/`np.seterr`: ignore, warn with `RuntimeWarning`, or raise
//! `FloatingPointError`. Keeping the state in Python means `errstate` nests and restores like
//! any context manager, and native code pays nothing when no flag is raised.

use super::super::super::native::{CallArgs, PyError, PyResult, PyRuntime};
use super::super::super::Value;
use super::ops::FpFlags;

/// Report the flags raised by ufunc `name`. `name` is the text NumPy prints after
/// "encountered in", such as `divide` or `scalar add`.
pub(in crate::python) fn report<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    name: &str,
    flags: FpFlags,
) -> PyResult<'s, ()> {
    if !flags.any() {
        return Ok(());
    }
    let module = runtime.import_module("numpy._errors")?;
    let report = runtime
        .get_attribute(module, "_report")?
        .ok_or_else(|| PyError::runtime_error("numpy._errors._report is missing"))?;
    let name = runtime.new_string(name.to_string())?;
    runtime.call_value(
        report,
        CallArgs::new(
            vec![
                name,
                Value::Bool(flags.divide),
                Value::Bool(flags.overflow),
                Value::Bool(flags.underflow),
                Value::Bool(flags.invalid),
            ],
            Vec::new(),
        ),
    )?;
    Ok(())
}

/// Warn that a complex value lost its imaginary part in a conversion to a real type, as NumPy
/// does with `ComplexWarning` for `float(np.complex128(1+2j))`.
pub(in crate::python) fn warn_complex_discard<'s>(
    runtime: &mut dyn PyRuntime<'s>,
) -> PyResult<'s, ()> {
    warn(runtime, "_warn_complex_discard")
}

/// Warn that a ufunc called with `where=` and no `out=` leaves unselected elements
/// uninitialized, as NumPy 2 does with a `UserWarning`.
pub(in crate::python) fn warn_where_without_out<'s>(
    runtime: &mut dyn PyRuntime<'s>,
) -> PyResult<'s, ()> {
    warn(runtime, "_warn_where_without_out")
}

fn warn<'s>(runtime: &mut dyn PyRuntime<'s>, name: &str) -> PyResult<'s, ()> {
    call_warning(runtime, "numpy._errors", name)
}

/// Call the no-argument warning function `name` of the frozen module `module`, which issues
/// the warning from Python so filters and `stacklevel` apply.
pub(in crate::python) fn call_warning<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    module: &str,
    name: &str,
) -> PyResult<'s, ()> {
    let module_value = runtime.import_module(module)?;
    let warn = runtime
        .get_attribute(module_value, name)?
        .ok_or_else(|| PyError::runtime_error(format!("{module}.{name} is missing")))?;
    runtime.call_value(warn, CallArgs::new(Vec::new(), Vec::new()))?;
    Ok(())
}
