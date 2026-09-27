//! `np.concatenate`: the joining kernel behind `stack`, `hstack`, `vstack`, `dstack`,
//! `column_stack`, `append`, and `resize`.
//!
//! The result dtype is the promotion of the inputs, where strings absorb numbers at their
//! printed width as in NumPy's array construction. Each input is checked against the
//! `casting` rule, converted, and written into a view of the output at its position along the
//! joined axis, so every result is a new array. With `axis=None` the inputs are flattened first.

use super::super::super::super::native::{
    CallArgs, PyError, PyNativeKind, PyResult, PyRuntime, PyValue,
};
use super::super::args::{self, Signature};
use super::super::array::{self, Array};
use super::super::convert;
use super::super::dtype::{self, Casting, DType};

/// Where the joined result goes: a new array of an optional dtype, or an existing array.
enum Target {
    New(Option<DType>),
    Out(Array),
}

/// Join `arrays` along `axis` (flattening first when `None`), checking every input against
/// `casting`.
fn concatenate(
    runtime: &mut dyn PyRuntime,
    arrays: Vec<Array>,
    axis: Option<i64>,
    target: Target,
    casting: Casting,
) -> PyResult<Array> {
    if arrays.is_empty() {
        return Err(PyError::value_error(
            "need at least one array to concatenate",
        ));
    }
    let (arrays, axis) = match axis {
        None => (
            arrays
                .iter()
                .map(|array| array::ravel(runtime, array))
                .collect::<PyResult<Vec<_>>>()?,
            0,
        ),
        Some(axis) => {
            let ndim = arrays[0].ndim();
            if ndim == 0 {
                return Err(PyError::value_error(
                    "zero-dimensional arrays cannot be concatenated",
                ));
            }
            (arrays, array::normalize_axis(axis, ndim)?)
        }
    };
    let shape = joined_shape(&arrays, axis)?;
    let dtype = match &target {
        Target::Out(out) => out.dtype,
        Target::New(Some(dtype)) => *dtype,
        Target::New(None) => {
            let mut dtype = arrays[0].dtype;
            for array in &arrays[1..] {
                dtype = convert::infer_promote(dtype, array.dtype)?;
            }
            dtype
        }
    };
    for array in &arrays {
        if !dtype::can_cast(array.dtype, dtype, casting) {
            return Err(PyError::type_error(format!(
                "Cannot cast array data from {} to {} according to the rule '{}'",
                array.dtype.repr(),
                dtype.repr(),
                casting.name()
            )));
        }
    }
    let output = match target {
        Target::Out(out) => {
            if out.ndim() != shape.len() {
                return Err(PyError::value_error(
                    "Output array has wrong dimensionality",
                ));
            }
            if out.shape() != shape.as_slice() {
                return Err(PyError::value_error("Output array is the wrong shape"));
            }
            out
        }
        Target::New(_) => {
            let count = array::element_count(&shape)?;
            let buffer = array::zeroed_buffer(runtime, dtype, count)?;
            array::new_array(runtime, buffer, dtype, shape)?
        }
    };
    let stride = output.strides()[axis];
    let mut start = 0usize;
    for array in &arrays {
        let offset = output.view.offset as isize + start as isize * stride;
        let block = array::new_view(
            runtime,
            &output,
            output.dtype,
            array.shape().to_vec(),
            output.strides().to_vec(),
            offset as usize,
        )?;
        array::assign(runtime, &block, array)?;
        start += array.shape()[axis];
    }
    Ok(output)
}

/// The shape of the joined result, with NumPy's errors for mismatched inputs.
fn joined_shape(arrays: &[Array], axis: usize) -> PyResult<Vec<usize>> {
    let first = arrays[0].shape();
    let mut shape = first.to_vec();
    for (position, array) in arrays.iter().enumerate().skip(1) {
        if array.ndim() != first.len() {
            return Err(PyError::value_error(format!(
                "all the input arrays must have same number of dimensions, but the array at \
                 index 0 has {} dimension(s) and the array at index {position} has {} \
                 dimension(s)",
                first.len(),
                array.ndim()
            )));
        }
        for (dimension, (expected, actual)) in first.iter().zip(array.shape()).enumerate() {
            if dimension != axis && expected != actual {
                return Err(PyError::value_error(format!(
                    "all the input array dimensions except for the concatenation axis must \
                     match exactly, but along dimension {dimension}, the array at index 0 has \
                     size {expected} and the array at index {position} has size {actual}"
                )));
            }
        }
        // A saturated length fails the allocation check that follows.
        shape[axis] = shape[axis].saturating_add(array.shape()[axis]);
    }
    Ok(shape)
}

/// The arrays of `concatenate`'s first argument: a list, a tuple, or the rows of an ndarray.
fn input_arrays(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<Array>> {
    let items = if runtime.native_kind(&value)? == Some(PyNativeKind::Array) {
        let array = Array::from_value(runtime, value)?;
        if array.ndim() == 0 {
            return Err(PyError::type_error(
                "The first input argument needs to be a sequence",
            ));
        }
        super::super::ndarray::rows(runtime, &array)?
    } else {
        super::sequence_items(runtime, &value)?
            .ok_or_else(|| PyError::type_error("The first input argument needs to be a sequence"))?
    };
    items
        .into_iter()
        .map(|item| convert::as_array(runtime, item))
        .collect()
}

/// NumPy's `casting=` parsing for joining functions.
fn casting_arg(runtime: &mut dyn PyRuntime, value: Option<PyValue>) -> PyResult<Casting> {
    let Some(value) = value else {
        return Ok(Casting::SameKind);
    };
    let text = runtime.string_value(&value)?.unwrap_or_default();
    Casting::parse(&text).map_err(|_| {
        PyError::value_error(format!(
            "casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '{text}')"
        ))
    })
}

/// `np.concatenate(arrays, axis=0, out=None, *, dtype=None, casting="same_kind")`.
pub(super) fn module_concatenate(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("concatenate", &["arrays", "axis", "out"], 1)
        .keyword_only(&["dtype", "casting"]);
    let bound = SIGNATURE.bind(&args)?;
    let arrays = input_arrays(runtime, bound.required("arrays"))?;
    let axis = match bound.get("axis") {
        None => Some(0),
        Some(value) if value.is_none() => None,
        Some(value) => Some(args::index_int(runtime, &value)?),
    };
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?;
    let out = bound.value("out");
    let target = match (out, dtype) {
        (Some(_), Some(_)) => {
            return Err(PyError::type_error(
                "concatenate() only takes `out` or `dtype` as an argument, but both were \
                 provided.",
            ))
        }
        (Some(out), None) => Target::Out(Array::from_value(runtime, out)?),
        (None, dtype) => Target::New(dtype),
    };
    let casting = casting_arg(runtime, bound.value("casting"))?;
    Ok(concatenate(runtime, arrays, axis, target, casting)?.value())
}
