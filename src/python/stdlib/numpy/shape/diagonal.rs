//! `np.diagonal` and `np.trace`: read-only diagonal views and their sum.
//!
//! A diagonal is a view, never a copy: stepping `offset` elements into the plane of `axis1` and
//! `axis2` and then walking both axes together (stride `strides[axis1] + strides[axis2]`) reads
//! every diagonal element without moving any storage. NumPy makes the view read-only because a
//! write through it would touch two logical axes from one store; this module marks it read-only
//! for the same reason. `trace` reuses the view: it is exactly [`super::super::reduce::reduce`]
//! ("add") of the diagonal's last axis, which is where the view puts it.

use super::super::super::super::native::{CallArgs, PyError, PyResult, PyRuntime, PyValue};
use super::super::args::{self, Axes, Signature};
use super::super::array::{self, Array};
use super::super::convert;
use super::super::reduce::{self, ReduceOptions};
use super::super::ufunc;

/// A read-only view of the `axis1`/`axis2` diagonal of `array` at `offset`, with `axis1` and
/// `axis2` removed and the diagonal appended as the last axis — NumPy's own placement,
/// confirmed black-box: `np.diagonal(np.arange(24).reshape(2,3,4))` (the default `axis1=0,
/// axis2=1`) has shape `(4, 2)`, the untouched axis first and the length-`min(2,3)` diagonal
/// last.
///
/// The diagonal's length follows NumPy's clamp, letting `offset` run past either axis down to
/// an empty (not an error) result: `min(d1, d2 - offset)` for `offset >= 0`, or `min(d1 +
/// offset, d2)` for `offset < 0`, floored at `0`.
fn diagonal_view(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    offset: i64,
    axis1: i64,
    axis2: i64,
) -> PyResult<Array> {
    if array.ndim() < 2 {
        return Err(PyError::value_error(
            "diag requires an array of at least two dimensions",
        ));
    }
    let axis1 = super::axis_index(axis1, array.ndim(), Some("axis1"))?;
    let axis2 = super::axis_index(axis2, array.ndim(), Some("axis2"))?;
    if axis1 == axis2 {
        return Err(PyError::value_error("axis1 and axis2 cannot be the same"));
    }
    let (d1, d2) = (array.shape()[axis1], array.shape()[axis2]);
    let (row_start, col_start, length) = if offset >= 0 {
        let offset = offset as usize;
        (0usize, offset, d1.min(d2.saturating_sub(offset)))
    } else {
        let offset = offset.unsigned_abs() as usize;
        (offset, 0usize, d1.saturating_sub(offset).min(d2))
    };

    let mut shape = Vec::with_capacity(array.ndim() - 1);
    let mut strides = Vec::with_capacity(array.ndim() - 1);
    for axis in 0..array.ndim() {
        if axis != axis1 && axis != axis2 {
            shape.push(array.shape()[axis]);
            strides.push(array.strides()[axis]);
        }
    }
    shape.push(length);
    strides.push(array.strides()[axis1] + array.strides()[axis2]);

    let base_offset = (array.view.offset as isize
        + row_start as isize * array.strides()[axis1]
        + col_start as isize * array.strides()[axis2]) as usize;
    let view = array::new_view(runtime, array, array.dtype, shape, strides, base_offset)?;
    runtime.set_array_writeable(view.handle, false)?;
    Ok(view)
}

pub(super) fn module_diagonal(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("diagonal", &["a", "offset", "axis1", "axis2"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let offset = args::optional_int(runtime, bound.get("offset"))?.unwrap_or(0);
    let axis1 = args::optional_int(runtime, bound.get("axis1"))?.unwrap_or(0);
    let axis2 = args::optional_int(runtime, bound.get("axis2"))?.unwrap_or(1);
    Ok(diagonal_view(runtime, &array, offset, axis1, axis2)?.value())
}

pub(super) fn method_diagonal(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    static SIGNATURE: Signature = Signature::new("diagonal", &["offset", "axis1", "axis2"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    let offset = args::optional_int(runtime, bound.get("offset"))?.unwrap_or(0);
    let axis1 = args::optional_int(runtime, bound.get("axis1"))?.unwrap_or(0);
    let axis2 = args::optional_int(runtime, bound.get("axis2"))?.unwrap_or(1);
    Ok(diagonal_view(runtime, &array, offset, axis1, axis2)?.value())
}

/// `np.trace`: the sum of one diagonal. Since [`diagonal_view`] always appends the diagonal as
/// the last axis, this is [`reduce::reduce`] of that one axis with the `add` ufunc — the same
/// engine `np.sum` uses, so `dtype=`/`out=` and the empty-diagonal-sums-to-zero identity behave
/// exactly as they do for `sum`.
fn trace_array(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    offset: i64,
    axis1: i64,
    axis2: i64,
    dtype: Option<super::super::dtype::DType>,
    out: Option<Array>,
) -> PyResult {
    let diagonal = diagonal_view(runtime, array, offset, axis1, axis2)?;
    let axis = diagonal.ndim() - 1;
    reduce::reduce(
        runtime,
        ufunc::named("add"),
        &diagonal,
        &ReduceOptions {
            axes: Axes::Some(vec![axis]),
            dtype,
            out,
            keepdims: false,
            initial: None,
        },
    )
}

/// Accept `out=array` or an omitted/`None` `out=`.
fn out_argument(runtime: &mut dyn PyRuntime, value: Option<PyValue>) -> PyResult<Option<Array>> {
    value
        .map(|value| {
            Array::from_value(runtime, value)
                .map_err(|_| PyError::type_error("output must be an array"))
        })
        .transpose()
}

pub(super) fn module_trace(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "trace",
        &["a", "offset", "axis1", "axis2", "dtype", "out"],
        1,
    );
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let offset = args::optional_int(runtime, bound.get("offset"))?.unwrap_or(0);
    let axis1 = args::optional_int(runtime, bound.get("axis1"))?.unwrap_or(0);
    let axis2 = args::optional_int(runtime, bound.get("axis2"))?.unwrap_or(1);
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?;
    let out = out_argument(runtime, bound.value("out"))?;
    trace_array(runtime, &array, offset, axis1, axis2, dtype, out)
}

pub(super) fn method_trace(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("trace", &["offset", "axis1", "axis2", "dtype", "out"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    let offset = args::optional_int(runtime, bound.get("offset"))?.unwrap_or(0);
    let axis1 = args::optional_int(runtime, bound.get("axis1"))?.unwrap_or(0);
    let axis2 = args::optional_int(runtime, bound.get("axis2"))?.unwrap_or(1);
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?;
    let out = out_argument(runtime, bound.value("out"))?;
    trace_array(runtime, &array, offset, axis1, axis2, dtype, out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn diagonal_length_clamps_to_zero_past_either_axis() {
        // offset >= 0: min(d1, d2 - offset), floored at 0.
        assert_eq!(clamp_length(2, 3, 0), 2);
        assert_eq!(clamp_length(2, 3, 1), 2);
        assert_eq!(clamp_length(2, 3, 10), 0);
        // offset < 0: min(d1 + offset, d2), floored at 0.
        assert_eq!(clamp_length(2, 3, -1), 1);
        assert_eq!(clamp_length(2, 3, -10), 0);
    }

    /// Test-only mirror of [`diagonal_view`]'s length clamp, isolated from array plumbing.
    fn clamp_length(d1: usize, d2: usize, offset: i64) -> usize {
        if offset >= 0 {
            let offset = offset as usize;
            d1.min(d2.saturating_sub(offset))
        } else {
            let offset = offset.unsigned_abs() as usize;
            d1.saturating_sub(offset).min(d2)
        }
    }
}
