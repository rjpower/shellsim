//! `np.repeat`: repeat each element of an array a fixed or per-element number of times along
//! one axis, or along the flattened array when no axis is given.
//!
//! Repetition is a gather: every output element copies exactly one source element, so the
//! kernel computes the source byte offset of each output position (grouped by the axes before
//! the repeat axis, the repeated source position, and the axes after it) and reads them through
//! [`index::gather`], which reserves memory and charges CPU for the copy. This module charges
//! again first, in proportion to the output element count, before it builds the offset list —
//! the same order [`index`]'s own `gather_plan` (advanced indexing) uses.

use super::super::super::super::native::{CallArgs, PyError, PyResult, PyRuntime, PyValue};
use super::super::args::{self, Signature};
use super::super::array::{self, format_shape, Array};
use super::super::convert;
use super::super::index;

/// `repeats=` broadcast against an axis of length `n`: one value for every position (a scalar,
/// or a size-1 array), or one value per position (a size-`n` array). NumPy's error text differs
/// by which case applies, confirmed black-box against the reference interpreter: a scalar's (or
/// size-1 array's) negative value reuses the "negative dimensions" message `reshape` and
/// `empty` use, while a genuine per-position array both requires an exact length match (with a
/// broadcast-shaped mismatch error) and has its own message for a negative entry.
fn resolve_repeats(counts: Vec<i64>, n: usize) -> PyResult<Vec<usize>> {
    if let [count] = counts[..] {
        if count < 0 {
            return Err(PyError::value_error("negative dimensions are not allowed"));
        }
        return Ok(vec![count as usize; n]);
    }
    if counts.len() != n {
        return Err(PyError::value_error(format!(
            "operands could not be broadcast together with shape {} {}",
            format_shape(&[n]),
            format_shape(&[counts.len()]),
        )));
    }
    counts
        .into_iter()
        .map(|count| {
            usize::try_from(count)
                .map_err(|_| PyError::value_error("repeats may not contain negative values."))
        })
        .collect()
}

/// `np.repeat(a, repeats, axis=None)` and `a.repeat(repeats, axis=None)`. `axis=None` repeats
/// the flattened array; otherwise the source position at `axis` index `i` becomes `repeats[i]`
/// (or every position becomes the one scalar `repeats`) consecutive copies, keeping every other
/// axis's order.
fn repeat_array(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    repeats: &PyValue,
    axis: Option<PyValue>,
) -> PyResult<Array> {
    let (source, axis) = match args::axis(runtime, axis, array.ndim())? {
        Some(axis) => (array.clone(), axis),
        None => (array::ravel(runtime, array)?, 0),
    };
    let n = source.shape()[axis];
    let counts = super::int_list(runtime, repeats)?;
    let counts = resolve_repeats(counts, n)?;

    let (pre_shape, rest_shape) = source.shape().split_at(axis);
    let post_shape = &rest_shape[1..];
    let (pre_strides, rest_strides) = source.strides().split_at(axis);
    let (axis_stride, post_strides) = (rest_strides[0], &rest_strides[1..]);

    let pre_offsets = index::relative_offsets(pre_shape, pre_strides);
    let post_offsets = index::relative_offsets(post_shape, post_strides);

    let output_axis_len: usize = counts.iter().sum();
    let mut shape = source.shape().to_vec();
    shape[axis] = output_axis_len;
    let total = array::element_count(&shape)?;
    array::reserve_elements(runtime, source.dtype, total)?;
    runtime.charge_cpu(total as u64 + 1)?;

    let mut offsets = Vec::with_capacity(total);
    for pre in &pre_offsets {
        for (position, count) in counts.iter().enumerate() {
            let base = source.view.offset as isize + pre + position as isize * axis_stride;
            for _ in 0..*count {
                for post in &post_offsets {
                    offsets.push((base + post) as usize);
                }
            }
        }
    }
    index::gather(runtime, &source, &offsets, shape)
}

pub(super) fn module_repeat(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("repeat", &["a", "repeats", "axis"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let repeats = bound.required("repeats");
    Ok(repeat_array(runtime, &array, &repeats, bound.value("axis"))?.value())
}

pub(super) fn method_repeat(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    static SIGNATURE: Signature = Signature::new("repeat", &["repeats", "axis"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    let repeats = bound.required("repeats");
    Ok(repeat_array(runtime, &array, &repeats, bound.value("axis"))?.value())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_repeats_broadcasts_a_single_value_and_validates_a_list() {
        assert_eq!(resolve_repeats(vec![2], 3).unwrap(), [2, 2, 2]);
        assert_eq!(resolve_repeats(vec![1, 0, 2], 3).unwrap(), [1, 0, 2]);
        assert_eq!(
            resolve_repeats(vec![-1], 3).unwrap_err().message,
            "negative dimensions are not allowed"
        );
        assert_eq!(
            resolve_repeats(vec![1, -1, 2], 3).unwrap_err().message,
            "repeats may not contain negative values."
        );
        assert_eq!(
            resolve_repeats(vec![1, 2], 3).unwrap_err().message,
            "operands could not be broadcast together with shape (3,) (2,)"
        );
    }
}
