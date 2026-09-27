//! `diagonal` and `trace`, following NumPy's `PyArray_Diagonal` and `PyArray_Trace`.
//!
//! A diagonal is a read-only view: the two chosen axes are removed and one axis stepping by
//! the sum of their strides is appended. `trace` sums that view over its last axis through
//! `ndarray.sum`, so its dtype rules and `out=` handling are the reduction's.

use super::super::super::super::native::{CallArgs, PyError, PyResult, PyRuntime, PyValue};
use super::super::super::super::Value;
use super::super::args::{self, Signature};
use super::super::array::{self, Array};
use super::super::convert;
use super::axis_index;

/// The diagonal view of `array` between `axis1` and `axis2`, `offset` above the main diagonal.
fn diagonal(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    offset: i64,
    axis1: i64,
    axis2: i64,
) -> PyResult<Array> {
    let ndim = array.ndim();
    if ndim < 2 {
        return Err(PyError::value_error(
            "diag requires an array of at least two dimensions",
        ));
    }
    let first = axis_index(axis1, ndim, Some("axis1"))?;
    let second = axis_index(axis2, ndim, Some("axis2"))?;
    if first == second {
        return Err(PyError::value_error("axis1 and axis2 cannot be the same"));
    }
    let (rows, columns) = (array.shape()[first], array.shape()[second]);
    let (row_stride, column_stride) = (array.strides()[first], array.strides()[second]);
    let (skip_rows, skip_columns) = if offset >= 0 {
        (0, offset.unsigned_abs())
    } else {
        (offset.unsigned_abs(), 0)
    };
    let length = (rows as u64)
        .saturating_sub(skip_rows)
        .min((columns as u64).saturating_sub(skip_columns)) as usize;
    let mut start = array.view.offset as isize;
    if length > 0 {
        start += skip_rows as isize * row_stride + skip_columns as isize * column_stride;
    }
    let mut shape = Vec::with_capacity(ndim - 1);
    let mut strides = Vec::with_capacity(ndim - 1);
    for axis in (0..ndim).filter(|axis| *axis != first && *axis != second) {
        shape.push(array.shape()[axis]);
        strides.push(array.strides()[axis]);
    }
    shape.push(length);
    strides.push(row_stride + column_stride);
    let view = array::new_view(runtime, array, array.dtype, shape, strides, start as usize)?;
    runtime.set_array_writeable(view.handle, false)?;
    Ok(view)
}

/// The `offset`, `axis1`, and `axis2` arguments, defaulting to `0, 0, 1`.
fn diagonal_arguments(
    runtime: &mut dyn PyRuntime,
    bound: &args::Bound,
) -> PyResult<(i64, i64, i64)> {
    Ok((
        args::optional_int(runtime, bound.value("offset"))?.unwrap_or(0),
        args::optional_int(runtime, bound.value("axis1"))?.unwrap_or(0),
        args::optional_int(runtime, bound.value("axis2"))?.unwrap_or(1),
    ))
}

/// `np.diagonal(a, offset=0, axis1=0, axis2=1)`.
pub(super) fn module_diagonal(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("diagonal", &["a", "offset", "axis1", "axis2"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let (offset, axis1, axis2) = diagonal_arguments(runtime, &bound)?;
    Ok(diagonal(runtime, &array, offset, axis1, axis2)?.value())
}

/// `ndarray.diagonal(offset=0, axis1=0, axis2=1)`.
pub(super) fn method_diagonal(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    static SIGNATURE: Signature = Signature::new("diagonal", &["offset", "axis1", "axis2"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    let (offset, axis1, axis2) = diagonal_arguments(runtime, &bound)?;
    Ok(diagonal(runtime, &array, offset, axis1, axis2)?.value())
}

/// Sum the diagonal over its last axis with `dtype` and `out`.
fn trace(runtime: &mut dyn PyRuntime, array: &Array, bound: &args::Bound) -> PyResult {
    let (offset, axis1, axis2) = diagonal_arguments(runtime, bound)?;
    let view = diagonal(runtime, array, offset, axis1, axis2)?;
    let sum = runtime
        .get_attribute(view.value(), "sum")?
        .ok_or_else(|| PyError::runtime_error("ndarray.sum is missing"))?;
    let mut keywords = vec![("axis".to_string(), Value::Int(-1))];
    for name in ["dtype", "out"] {
        if let Some(value) = bound.value(name) {
            keywords.push((name.to_string(), value));
        }
    }
    runtime.call_value(sum, CallArgs::new(Vec::new(), keywords))
}

/// `np.trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None)`.
pub(super) fn module_trace(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "trace",
        &["a", "offset", "axis1", "axis2", "dtype", "out"],
        1,
    );
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    trace(runtime, &array, &bound)
}

/// `ndarray.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)`.
pub(super) fn method_trace(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("trace", &["offset", "axis1", "axis2", "dtype", "out"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    trace(runtime, &array, &bound)
}
