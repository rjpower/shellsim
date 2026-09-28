//! `broadcast_to`: the one broadcast primitive that touches storage layout.
//!
//! A broadcast is a view: axes of length one (and missing leading axes) repeat the same
//! element through a zero stride, so no element storage is allocated. `broadcast_to` marks its
//! result read-only as NumPy does, because writing through a zero stride would change many
//! logical elements at once. `broadcast_arrays` and `broadcast_shapes` are frozen Python in
//! `numpy._shapes`, built on this view plus the common-shape computation there.

use super::super::super::super::native::{CallArgs, PyError, PyResult, PyRuntime};
use super::super::args::Signature;
use super::super::array::{self, format_shape, Array};
use super::super::convert;

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
