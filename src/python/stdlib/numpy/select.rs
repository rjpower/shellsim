//! Selection by truth value: `nonzero`, `where`, and `copyto`.
//!
//! These follow NumPy's C implementations (`PyArray_Nonzero`, `PyArray_Where`, and
//! `array_copyto`). An element is true when it is nonzero: NaN counts as true, a complex value
//! when either part is nonzero, a string when it is non-empty, and an object by Python truth.
//!
//! `where(condition, x, y)` promotes `x` and `y` under NEP 50, so Python scalars are weak and
//! `where(mask, int8_array, 5)` stays `int8`. The result always has the broadcast shape of all
//! three operands and is an array, 0-d included.

use super::super::super::native::{
    CallArgs, MethodDef, NativeTypeDef, PyArrayBuffer, PyError, PyNativeKind, PyResult, PyRuntime,
    PyValue,
};
use super::args::Signature;
use super::array::{self, broadcast_buffer, broadcast_shapes, Array};
use super::convert;
use super::dtype::{self, Casting, DType};
use super::index;

/// Methods this area installs on `numpy.ndarray`.
pub(in crate::python) static ARRAY_METHODS: NativeTypeDef = NativeTypeDef {
    name: "numpy.ndarray",
    methods: &[MethodDef {
        type_name: "numpy.ndarray",
        name: "nonzero",
        call: method_nonzero,
    }],
    getters: &[],
};

const ZERO_DIMENSIONAL: &str =
    "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.";

/// The indices of true elements as a tuple of `int64` arrays, one per axis.
fn nonzero(runtime: &mut dyn PyRuntime, array: &Array, zero_dimensional: &str) -> PyResult {
    if array.ndim() == 0 {
        return Err(PyError::value_error(zero_dimensional));
    }
    let coordinates = index::nonzero(runtime, array)?;
    let arrays = coordinates
        .into_iter()
        .map(|values| {
            let length = values.len();
            Ok(array::array_from_elements(runtime, DType::INT64, vec![length], &values)?.value())
        })
        .collect::<PyResult<Vec<_>>>()?;
    runtime.new_tuple(arrays)
}

/// `np.nonzero(a)`.
pub(in crate::python) fn module_nonzero(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("nonzero", &["a"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    nonzero(runtime, &array, ZERO_DIMENSIONAL)
}

fn method_nonzero(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("nonzero", 0, 0)?;
    args.reject_keywords("nonzero")?;
    let array = Array::from_value(runtime, receiver)?;
    nonzero(runtime, &array, ZERO_DIMENSIONAL)
}

/// An operand of `where` or `copyto` converted to `target`. Python scalars convert as weak
/// values, so one that does not fit an integer target raises NumPy's `OverflowError`.
fn operand(runtime: &mut dyn PyRuntime, value: PyValue, target: DType) -> PyResult<Array> {
    if let Some((_, leaf)) = convert::weak_scalar(runtime, &value)? {
        return convert::weak_array(runtime, value, &leaf, target);
    }
    convert::as_array(runtime, value)
}

/// The dtype `where` gives its result: `x` and `y` promoted with weak Python scalars.
fn where_dtype(runtime: &mut dyn PyRuntime, values: [PyValue; 2]) -> PyResult<DType> {
    let mut strong = Vec::new();
    let mut weak = Vec::new();
    for value in values {
        match convert::weak_scalar(runtime, &value)? {
            Some((kind, _)) => weak.push(kind),
            None => strong.push(convert::as_array(runtime, value)?.dtype),
        }
    }
    if strong.is_empty() {
        // Two Python scalars promote as their default dtypes, as `np.where(c, 1, 2.5)` does.
        for value in values {
            strong.push(convert::as_array(runtime, value)?.dtype);
        }
        weak.clear();
    }
    dtype::result_type(&strong, &weak)
}

/// `np.where(condition, [x, y])`.
pub(in crate::python) fn module_where(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("where", &["condition", "x", "y"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let condition = convert::as_array(runtime, bound.required("condition"))?;
    let (x, y) =
        match (bound.get("x"), bound.get("y")) {
            (None, None) => return nonzero(
                runtime,
                &condition,
                "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() \
                 instead. If the context of this error is of the form `arr[nonzero(cond)]`, just \
                 use `arr[cond]`.",
            ),
            (Some(x), Some(y)) => (x, y),
            _ => {
                return Err(PyError::value_error(
                    "either both or neither of x and y should be given",
                ))
            }
        };
    let dtype = where_dtype(runtime, [x, y])?;
    let x = operand(runtime, x, dtype)?;
    let y = operand(runtime, y, dtype)?;
    let shape = broadcast_shapes(&[condition.shape(), x.shape(), y.shape()])?;
    let truth = broadcast_truth(runtime, &condition, &shape)?;
    let chosen = broadcast_buffer(runtime, &x, dtype, &shape)?;
    let other = broadcast_buffer(runtime, &y, dtype, &shape)?;
    let buffer = select(chosen, other, &truth, dtype.itemsize());
    Ok(array::new_array(runtime, buffer, dtype, shape)?.value())
}

/// The truth of each element of `condition` broadcast to `shape`, in C order.
pub(in crate::python) fn broadcast_truth(
    runtime: &mut dyn PyRuntime,
    condition: &Array,
    shape: &[usize],
) -> PyResult<Vec<bool>> {
    let strides = array::broadcast_strides(&condition.view, shape)?;
    let view = array::new_view(
        runtime,
        condition,
        condition.dtype,
        shape.to_vec(),
        strides,
        condition.view.offset,
    )?;
    index::truth_values(runtime, &view)
}

/// Elements of `chosen` where `truth` holds and of `other` elsewhere. Both buffers hold one
/// element per truth value, with the same storage kind.
fn select(
    chosen: PyArrayBuffer,
    other: PyArrayBuffer,
    truth: &[bool],
    itemsize: usize,
) -> PyArrayBuffer {
    match (chosen, other) {
        (PyArrayBuffer::Bytes(chosen), PyArrayBuffer::Bytes(mut other)) => {
            for (index, _) in truth.iter().enumerate().filter(|(_, truth)| **truth) {
                let range = index * itemsize..(index + 1) * itemsize;
                other[range.clone()].copy_from_slice(&chosen[range]);
            }
            PyArrayBuffer::Bytes(other)
        }
        (PyArrayBuffer::Values(chosen), PyArrayBuffer::Values(mut other)) => {
            for (index, _) in truth.iter().enumerate().filter(|(_, truth)| **truth) {
                other[index] = chosen[index];
            }
            PyArrayBuffer::Values(other)
        }
        _ => unreachable!("where selects between buffers of one dtype"),
    }
}

/// `np.copyto(dst, src, casting='same_kind', where=True)`.
pub(in crate::python) fn module_copyto(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("copyto", &["dst", "src", "casting", "where"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let destination = bound.required("dst");
    if runtime.native_kind(&destination)? != Some(PyNativeKind::Array) {
        return Err(PyError::type_error(format!(
            "copyto() argument 1 must be a numpy.ndarray, not {}",
            runtime.type_name(&destination)?
        )));
    }
    let destination = Array::from_value(runtime, destination)?;
    let casting = match bound.value("casting") {
        None => Casting::SameKind,
        Some(value) => {
            let text = runtime.string_value(&value)?.unwrap_or_default();
            Casting::parse(&text).map_err(|_| {
                PyError::value_error(format!(
                    "casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got \
                     '{text}')"
                ))
            })?
        }
    };
    // A Python scalar takes the destination's dtype when it fits the weak promotion rules, and
    // otherwise its own default dtype, which the casting rule then judges.
    let source_value = bound.required("src");
    let source = match convert::weak_scalar(runtime, &source_value)? {
        Some((kind, leaf)) => {
            let target = dtype::result_type(&[destination.dtype], &[kind])?;
            convert::weak_array(runtime, source_value, &leaf, target)?
        }
        None => convert::as_array(runtime, source_value)?,
    };
    if !dtype::can_cast(source.dtype, destination.dtype, casting) {
        return Err(PyError::type_error(format!(
            "Cannot cast {} from {} to {} according to the rule '{}'",
            if source.ndim() == 0 {
                "scalar"
            } else {
                "array data"
            },
            source.dtype.repr(),
            destination.dtype.repr(),
            casting.name()
        )));
    }
    let shape = destination.shape().to_vec();
    let values = broadcast_buffer(runtime, &source, destination.dtype, &shape)?;
    let mask = match bound.value("where") {
        None => None,
        Some(mask) => {
            let mask = convert::as_array(runtime, mask)?;
            Some(broadcast_truth(runtime, &mask, &shape)?)
        }
    };
    let offsets = destination.offsets().collect::<Vec<_>>();
    match mask {
        None => array::scatter(runtime, &destination, &offsets, &values)?,
        Some(mask) => {
            let (offsets, values) = masked(offsets, values, &mask, destination.dtype);
            array::scatter(runtime, &destination, &offsets, &values)?;
        }
    }
    Ok(PyValue::None)
}

/// The destination offsets and source elements where `mask` holds.
fn masked(
    offsets: Vec<usize>,
    values: PyArrayBuffer,
    mask: &[bool],
    dtype: DType,
) -> (Vec<usize>, PyArrayBuffer) {
    let kept = offsets
        .into_iter()
        .zip(mask)
        .filter(|(_, keep)| **keep)
        .map(|(offset, _)| offset)
        .collect();
    let values = match values {
        PyArrayBuffer::Bytes(bytes) => {
            let itemsize = dtype.itemsize();
            PyArrayBuffer::Bytes(
                bytes
                    .chunks_exact(itemsize.max(1))
                    .zip(mask)
                    .filter(|(_, keep)| **keep)
                    .flat_map(|(chunk, _)| chunk.iter().copied())
                    .collect(),
            )
        }
        PyArrayBuffer::Values(values) => PyArrayBuffer::Values(
            values
                .into_iter()
                .zip(mask)
                .filter(|(_, keep)| **keep)
                .map(|(value, _)| value)
                .collect(),
        ),
    };
    (kept, values)
}
