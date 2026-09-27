//! Broadcasting helpers: `broadcast_to`, `broadcast_arrays`, and `broadcast_shapes`.
//!
//! A broadcast is a view: axes of length one (and missing leading axes) repeat the same
//! element through a zero stride, so no element storage is allocated. `broadcast_to` marks its
//! result read-only as NumPy does, because writing through a zero stride would change many
//! logical elements at once.

use super::super::super::super::native::{CallArgs, PyError, PyResult, PyRuntime, PyValue};
use super::super::args::Signature;
use super::super::array::{self, format_shape, Array};
use super::super::convert;
use super::super::ndarray::int_tuple;

/// A view of `array` with `shape`, repeating length-one and missing leading axes. Errors use
/// the text of NumPy's `broadcast_to`.
pub(super) fn broadcast_view(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    shape: &[usize],
) -> PyResult<Array> {
    if shape.is_empty() && array.ndim() > 0 {
        return Err(PyError::value_error(
            "cannot broadcast a non-scalar to a scalar array",
        ));
    }
    let Some(extra) = shape.len().checked_sub(array.ndim()) else {
        return Err(PyError::value_error(
            "input operand has more dimensions than allowed by the axis remapping",
        ));
    };
    let mut strides = vec![0isize; shape.len()];
    for (axis, (dimension, stride)) in array.shape().iter().zip(array.strides()).enumerate() {
        let target = shape[extra + axis];
        if *dimension == target {
            strides[extra + axis] = *stride;
        } else if *dimension != 1 {
            return Err(PyError::value_error(format!(
                "operands could not be broadcast together with remapped shapes \
                 [original->remapped]: {}  and requested shape {}",
                format_shape(array.shape()),
                format_shape(shape)
            )));
        }
    }
    array::new_view(
        runtime,
        array,
        array.dtype,
        shape.to_vec(),
        strides,
        array.view.offset,
    )
}

/// The common shape of `shapes`, with the error text of NumPy's `np.broadcast`, which names
/// the first argument that fixed a mismatched axis and the argument that disagreed.
pub(super) fn common_shape(shapes: &[Vec<usize>]) -> PyResult<Vec<usize>> {
    let rank = shapes.iter().map(Vec::len).max().unwrap_or(0);
    let mut result = vec![1usize; rank];
    for (axis, target) in result.iter_mut().enumerate() {
        let mut source = 0;
        for (position, shape) in shapes.iter().enumerate() {
            let Some(index) = (axis + shape.len()).checked_sub(rank) else {
                continue;
            };
            let dimension = shape[index];
            if dimension == 1 {
                continue;
            }
            if *target == 1 {
                *target = dimension;
                source = position;
            } else if *target != dimension {
                return Err(PyError::value_error(format!(
                    "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch \
                     is between arg {source} with shape {} and arg {position} with shape {}.",
                    format_shape(&shapes[source]),
                    format_shape(shape)
                )));
            }
        }
    }
    Ok(result)
}

/// `np.broadcast_to(array, shape, subok=False)`: a read-only broadcast view.
pub(super) fn module_broadcast_to(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("broadcast_to", &["array", "shape", "subok"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("array"))?;
    let requested = super::int_list(runtime, &bound.required("shape"))?;
    if requested.iter().any(|dimension| *dimension < 0) {
        return Err(PyError::value_error(
            "all elements of broadcast shape must be non-negative",
        ));
    }
    let shape = requested
        .into_iter()
        .map(|dimension| dimension as usize)
        .collect::<Vec<_>>();
    array::element_count(&shape)?;
    let view = broadcast_view(runtime, &array, &shape)?;
    runtime.set_array_writeable(view.handle, false)?;
    Ok(view.value())
}

/// `np.broadcast_arrays(*args, subok=False)`: each argument as an array of the common shape.
/// Arrays that already have it are returned unchanged; the others become broadcast views.
/// NumPy marks those views writeable-with-a-warning; shellsim keeps the source's writeability
/// and does not warn.
pub(super) fn module_broadcast_arrays(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.reject_unknown_keywords("broadcast_arrays", &["subok"])?;
    let arrays = args
        .positional()
        .iter()
        .map(|value| convert::as_array(runtime, *value))
        .collect::<PyResult<Vec<_>>>()?;
    let shapes = arrays
        .iter()
        .map(|array| array.shape().to_vec())
        .collect::<Vec<_>>();
    let shape = common_shape(&shapes)?;
    let mut results = Vec::with_capacity(arrays.len());
    for array in &arrays {
        results.push(if array.shape() == shape.as_slice() {
            array.value()
        } else {
            broadcast_view(runtime, array, &shape)?.value()
        });
    }
    runtime.new_tuple(results)
}

/// One `broadcast_shapes` argument: an int or a sequence of ints.
fn shape_argument(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<Vec<usize>> {
    let dimensions = super::int_list(runtime, value)?;
    let shape = dimensions
        .into_iter()
        .map(|dimension| {
            usize::try_from(dimension)
                .map_err(|_| PyError::value_error("negative dimensions are not allowed"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    array::element_count(&shape)?;
    Ok(shape)
}

/// `np.broadcast_shapes(*args)`: the broadcast shape as a tuple.
pub(super) fn module_broadcast_shapes(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.reject_keywords("broadcast_shapes")?;
    let shapes = args
        .positional()
        .iter()
        .map(|value| shape_argument(runtime, value))
        .collect::<PyResult<Vec<_>>>()?;
    let shape = common_shape(&shapes)?;
    int_tuple(runtime, shape.into_iter().map(|dimension| dimension as i64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatch_names_the_argument_that_fixed_the_axis() {
        let shapes = [vec![2, 1], vec![3], vec![4]];
        let error = common_shape(&shapes).unwrap_err();
        assert_eq!(
            error.message,
            "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is \
             between arg 1 with shape (3,) and arg 2 with shape (4,)."
        );
        assert_eq!(
            common_shape(&[vec![2, 1], vec![3], vec![]]).unwrap(),
            [2, 3]
        );
        assert_eq!(common_shape(&[]).unwrap(), Vec::<usize>::new());
    }
}
