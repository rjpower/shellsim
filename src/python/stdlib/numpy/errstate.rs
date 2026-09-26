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
pub(in crate::python) fn report(runtime: &mut dyn PyRuntime, name: &str, flags: FpFlags) -> PyResult<()> {
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
