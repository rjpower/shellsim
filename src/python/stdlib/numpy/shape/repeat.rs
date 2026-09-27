//! `np.repeat` and `ndarray.repeat`: each element along one axis copied a given number of times.
//!
//! This follows NumPy's `PyArray_Repeat`. `repeats` converts to `intp` with unsafe casting, so
//! `2.5` repeats twice. A scalar or single count applies to every element, and any other count
//! array must match the axis length. Without an axis the array is flattened first.

use super::super::super::super::native::{CallArgs, PyError, PyResult, PyRuntime, PyValue};
use super::super::args::{self, Signature};
use super::super::array::{self, Array};
use super::super::convert;
use super::super::dtype::DType;
use super::super::index;

/// `np.repeat(a, repeats, axis=None)`.
pub(super) fn module_repeat(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("repeat", &["a", "repeats", "axis"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    Ok(repeat(
        runtime,
        &array,
        bound.required("repeats"),
        bound.value("axis"),
    )?
    .value())
}

/// `ndarray.repeat(repeats, axis=None)`.
pub(super) fn method_repeat(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    static SIGNATURE: Signature = Signature::new("repeat", &["repeats", "axis"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    Ok(repeat(
        runtime,
        &array,
        bound.required("repeats"),
        bound.value("axis"),
    )?
    .value())
}

fn repeat(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    repeats: PyValue,
    axis: Option<PyValue>,
) -> PyResult<Array> {
    let counts = convert::array_from_python(runtime, repeats, Some(DType::INT64), false)?;
    if counts.ndim() > 1 {
        return Err(PyError::value_error(
            "setting an array element with a sequence. The requested array would exceed the \
             maximum number of dimension of 1.",
        ));
    }
    let counts = array::read_elements::<i64>(runtime, &counts)?;
    let (source, axis) = match args::axis(runtime, axis, array.ndim().max(1))? {
        Some(axis) if array.ndim() > 0 => (array.clone(), axis),
        _ => (array::ravel(runtime, array)?, 0),
    };
    let length = source.shape()[axis];
    let per_element = match counts.as_slice() {
        [count] => vec![*count; length],
        counts if counts.len() == length => {
            if counts.iter().any(|count| *count < 0) {
                return Err(PyError::value_error(
                    "repeats may not contain negative values.",
                ));
            }
            counts.to_vec()
        }
        counts => {
            return Err(PyError::value_error(format!(
                "operands could not be broadcast together with shape ({length},) ({},)",
                counts.len()
            )))
        }
    };
    let total = per_element
        .iter()
        .try_fold(0i64, |total, count| total.checked_add(*count))
        .ok_or_else(|| PyError::value_error("array is too big"))?;
    // A negative scalar count gives a negative length, which NumPy rejects as a dimension.
    let total = usize::try_from(total)
        .map_err(|_| PyError::value_error("negative dimensions are not allowed"))?;
    let mut shape = source.shape().to_vec();
    shape[axis] = total;
    let count = array::element_count(&shape)?;
    runtime.reserve_memory(count.saturating_mul(8))?;
    runtime.charge_cpu(count as u64 + length as u64 + 1)?;
    let stride = source.strides()[axis];
    let outer = index::relative_offsets(&source.shape()[..axis], &source.strides()[..axis]);
    let inner = index::relative_offsets(&source.shape()[axis + 1..], &source.strides()[axis + 1..]);
    let mut offsets = Vec::with_capacity(count);
    // With no elements, a huge count must not drive an empty loop.
    for outer in outer.iter().filter(|_| count > 0) {
        for (position, times) in per_element.iter().enumerate() {
            let base = source.view.offset as isize + outer + position as isize * stride;
            for _ in 0..*times {
                offsets.extend(inner.iter().map(|inner| (base + inner) as usize));
            }
        }
    }
    index::gather(runtime, &source, &offsets, shape)
}
