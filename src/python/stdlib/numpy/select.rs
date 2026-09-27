//! Selection by truth: `nonzero`, `where`, and `copyto`.
//!
//! `nonzero` and `where` share NumPy's per-dtype truth rule (a numeric zero, an empty string, a
//! falsy object, or a `False` bool is false; everything else is true), applied through
//! [`broadcast_truth`], which casts a condition array to `bool` and reads it broadcast to a
//! target shape. `where(condition, x, y)` promotes `x` and `y` under NEP 50 the same way a
//! binary ufunc does (see [`super::ufunc::operand`] and [`super::ufunc::common_dtype`]), then
//! selects each output element's bytes (or object reference) from the matching broadcast
//! operand. `copyto` reuses the same truth broadcast to mask an in-place assignment.
//!
//! Costs are charged once per selected/broadcast element before the copy or comparison loop
//! that does the work.

use super::super::super::native::{
    CallArgs, MethodDef, NativeTypeDef, PyArrayBuffer, PyArrayData, PyError, PyResult, PyRuntime,
    PyValue,
};
use super::super::super::Value;
use super::args::Signature;
use super::array::{self, Array, Offsets};
use super::convert;
use super::dtype::{self, Casting, DType};
use super::index;
use super::ufunc;

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

fn receiver(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Array> {
    Array::from_value(runtime, value)
}

/// The truth of `mask`, broadcast to `shape`, in C order. `mask` need not already be boolean:
/// it converts with NumPy's per-dtype truth rule, the same as an `if` condition or `bool(x)`.
/// Shared by `where`, `copyto`, and masked ufunc calls (`np.add(a, b, where=mask)`).
pub(in crate::python) fn broadcast_truth(
    runtime: &mut dyn PyRuntime,
    mask: &Array,
    shape: &[usize],
) -> PyResult<Vec<bool>> {
    let mask = if mask.dtype == DType::BOOL {
        mask.clone()
    } else {
        convert::cast_array(runtime, mask, DType::BOOL, false)?
    };
    let strides = array::broadcast_strides(&mask.view, shape)?;
    let count = array::element_count(shape)?;
    runtime.reserve_memory(count)?;
    runtime.charge_cpu(count as u64 / 8 + 1)?;
    let mut result = Vec::with_capacity(count);
    runtime.read_arrays(&[mask.handle], &mut |arrays| {
        let PyArrayData::Bytes(bytes) = arrays[0].data else {
            return Err(PyError::runtime_error("bool array has object storage"));
        };
        let offsets = Offsets::new(shape, &strides, mask.view.offset);
        result.extend(offsets.map(|offset| bytes[offset] != 0));
        Ok(())
    })?;
    Ok(result)
}

/// `np.nonzero(a)`/`a.nonzero()`: one int64 index array per axis, C order. Unlike the internal
/// boolean-mask path ([`index::nonzero`]), which treats a 0-d array as shape `(1,)` for `a[mask]`
/// indexing, the top-level function rejects 0-d input outright.
fn nonzero(runtime: &mut dyn PyRuntime, array: &Array) -> PyResult<PyValue> {
    if array.ndim() == 0 {
        return Err(PyError::value_error(
            "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() \
             instead.",
        ));
    }
    let coordinates = index::nonzero(runtime, array)?;
    let mut outputs = Vec::with_capacity(coordinates.len());
    for values in coordinates {
        let length = values.len();
        outputs.push(
            array::array_from_elements(runtime, DType::INT64, vec![length], &values)?.value(),
        );
    }
    runtime.new_tuple(outputs)
}

pub(in crate::python) fn module_nonzero(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("nonzero", &["a"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    nonzero(runtime, &array)
}

fn method_nonzero(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("nonzero", 0, 0)?;
    args.reject_keywords("nonzero")?;
    let array = receiver(runtime, receiver_value)?;
    nonzero(runtime, &array)
}

/// One NEP 50 operand (array or weak Python scalar) cast to `dtype`.
fn operand_array(
    runtime: &mut dyn PyRuntime,
    operand: ufunc::Operand,
    dtype: DType,
) -> PyResult<Array> {
    match operand {
        ufunc::Operand::Array(array) => convert::cast_array(runtime, &array, dtype, false),
        ufunc::Operand::Weak { value, leaf, .. } => convert::weak_array(runtime, value, &leaf, dtype),
    }
}

/// Combine two same-dtype, same-shape broadcast buffers by `truth`, element by element.
fn select_buffer(truth: &[bool], itemsize: usize, x: &PyArrayBuffer, y: &PyArrayBuffer) -> PyArrayBuffer {
    match (x, y) {
        (PyArrayBuffer::Bytes(x), PyArrayBuffer::Bytes(y)) => {
            let mut out = vec![0u8; truth.len() * itemsize];
            for (index, &pick_x) in truth.iter().enumerate() {
                let source = if pick_x { x } else { y };
                out[index * itemsize..(index + 1) * itemsize]
                    .copy_from_slice(&source[index * itemsize..(index + 1) * itemsize]);
            }
            PyArrayBuffer::Bytes(out)
        }
        (PyArrayBuffer::Values(x), PyArrayBuffer::Values(y)) => PyArrayBuffer::Values(
            truth
                .iter()
                .enumerate()
                .map(|(index, &pick_x)| if pick_x { x[index] } else { y[index] })
                .collect(),
        ),
        _ => unreachable!("x and y share dtype storage"),
    }
}

/// `np.where(condition, x, y)`: `x` where `condition` is true, `y` elsewhere, promoted and
/// broadcast together like a binary ufunc.
fn where_select(runtime: &mut dyn PyRuntime, condition: PyValue, x: PyValue, y: PyValue) -> PyResult {
    let condition = convert::array_from_python(runtime, condition, Some(DType::BOOL), false)?;
    let operands = [ufunc::operand(runtime, x)?.0, ufunc::operand(runtime, y)?.0];
    let dtype = ufunc::common_dtype(&operands)?;
    let [x_operand, y_operand] = operands;
    let x_array = operand_array(runtime, x_operand, dtype)?;
    let y_array = operand_array(runtime, y_operand, dtype)?;
    let shape = array::broadcast_shapes(&[condition.shape(), x_array.shape(), y_array.shape()])?;
    let truth = broadcast_truth(runtime, &condition, &shape)?;
    let x_buffer = array::broadcast_buffer(runtime, &x_array, dtype, &shape)?;
    let y_buffer = array::broadcast_buffer(runtime, &y_array, dtype, &shape)?;
    let count = array::element_count(&shape)?;
    array::reserve_elements(runtime, dtype, count)?;
    runtime.charge_cpu(count as u64 + 1)?;
    let buffer = select_buffer(&truth, dtype.itemsize(), &x_buffer, &y_buffer);
    Ok(array::new_array(runtime, buffer, dtype, shape)?.value())
}

pub(in crate::python) fn module_where(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("where", &["condition", "x", "y"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let condition = bound.required("condition");
    match (bound.get("x"), bound.get("y")) {
        (None, None) => {
            let array = convert::as_array(runtime, condition)?;
            nonzero(runtime, &array)
        }
        (Some(x), Some(y)) => where_select(runtime, condition, x, y),
        _ => Err(PyError::value_error(
            "either both or neither of x and y should be given",
        )),
    }
}

/// Keep only the offsets and matching buffer entries selected by `truth`.
fn filter_selected(
    truth: &[bool],
    itemsize: usize,
    offsets: &[usize],
    buffer: &PyArrayBuffer,
) -> (Vec<usize>, PyArrayBuffer) {
    let selected_offsets = offsets
        .iter()
        .zip(truth)
        .filter_map(|(&offset, &selected)| selected.then_some(offset))
        .collect();
    let selected_buffer = match buffer {
        PyArrayBuffer::Bytes(bytes) => PyArrayBuffer::Bytes(
            truth
                .iter()
                .enumerate()
                .filter(|(_, &selected)| selected)
                .flat_map(|(index, _)| bytes[index * itemsize..(index + 1) * itemsize].to_vec())
                .collect(),
        ),
        PyArrayBuffer::Values(values) => PyArrayBuffer::Values(
            truth
                .iter()
                .enumerate()
                .filter(|(_, &selected)| selected)
                .map(|(index, _)| values[index])
                .collect(),
        ),
    };
    (selected_offsets, selected_buffer)
}

/// `np.copyto(dst, src, casting='same_kind', where=True)`: `dst[...] = src`, masked by `where`.
/// Always returns `None`.
pub(in crate::python) fn module_copyto(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("copyto", &["dst", "src"], 2).keyword_only(&["casting", "where"]);
    let bound = SIGNATURE.bind(&args)?;
    let dst = Array::from_value(runtime, bound.required("dst"))?;
    if !dst.view.writeable {
        return Err(PyError::value_error("assignment destination is read-only"));
    }
    let casting = match bound.value("casting") {
        Some(value) => {
            let text = runtime.string_value(&value)?.unwrap_or_default();
            Casting::parse(&text)?
        }
        None => Casting::SameKind,
    };
    let src = convert::as_array(runtime, bound.required("src"))?;
    if !dtype::can_cast(src.dtype, dst.dtype, casting) {
        return Err(PyError::type_error(format!(
            "Cannot cast array data from {} to {} according to the rule '{}'",
            src.dtype.repr(),
            dst.dtype.repr(),
            casting.name()
        )));
    }
    match bound.value("where") {
        None => array::assign(runtime, &dst, &src)?,
        Some(where_value) => {
            let mask = convert::array_from_python(runtime, where_value, None, false)?;
            let truth = broadcast_truth(runtime, &mask, dst.shape())?;
            let buffer = array::broadcast_buffer(runtime, &src, dst.dtype, dst.shape())?;
            let offsets = dst.offsets().collect::<Vec<_>>();
            runtime.charge_cpu(offsets.len() as u64 + 1)?;
            let (offsets, buffer) = filter_selected(&truth, dst.itemsize(), &offsets, &buffer);
            array::scatter(runtime, &dst, &offsets, &buffer)?;
        }
    }
    Ok(Value::None)
}
