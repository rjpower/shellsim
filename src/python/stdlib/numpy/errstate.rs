//! Floating-point error reporting for ufunc loops.
//!
//! Loops record IEEE-style flags in [`FpFlags`] without calling back into Python. After a loop
//! finishes, [`report`] hands raised flags to `numpy._errstate._report`, which applies the modes
//! set by `np.errstate`/`np.seterr`: ignore, warn with `RuntimeWarning`, or raise
//! `FloatingPointError`. Keeping the state in Python means `errstate` nests and restores like
//! any context manager, and native code pays nothing when no flag is raised.

use super::super::super::native::{CallArgs, PyError, PyResult, PyRuntime};
use super::super::super::Value;
use super::ops::FpFlags;

/// Report the flags raised by ufunc `name`. `name` is the text NumPy prints after
/// "encountered in", such as `divide` or `scalar add`.
pub(in crate::python) fn report(
    runtime: &mut dyn PyRuntime,
    name: &str,
    flags: FpFlags,
) -> PyResult<()> {
    if !flags.any() {
        return Ok(());
    }
    let module = runtime.import_module("numpy._errstate")?;
    let report = runtime
        .get_attribute(module, "_report")?
        .ok_or_else(|| PyError::runtime_error("numpy._errstate._report is missing"))?;
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
pub(in crate::python) fn warn_complex_discard(runtime: &mut dyn PyRuntime) -> PyResult<()> {
    warn(runtime, "_warn_complex_discard")
}

/// Warn that a ufunc called with `where=` and no `out=` leaves unselected elements
/// uninitialized, as NumPy 2 does with a `UserWarning`.
pub(in crate::python) fn warn_where_without_out(runtime: &mut dyn PyRuntime) -> PyResult<()> {
    warn(runtime, "_warn_where_without_out")
}

/// Call the warning function `name` of `numpy._errstate`, which issues the warning from
/// Python so filters and `stacklevel` apply.
fn warn(runtime: &mut dyn PyRuntime, name: &str) -> PyResult<()> {
    let module = runtime.import_module("numpy._errstate")?;
    let warn = runtime
        .get_attribute(module, name)?
        .ok_or_else(|| PyError::runtime_error(format!("numpy._errstate.{name} is missing")))?;
    runtime.call_value(warn, CallArgs::new(Vec::new(), Vec::new()))?;
    Ok(())
}
