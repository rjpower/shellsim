//! The `numpy.ndarray` type: core attributes, element access methods, and protocol slots.
//!
//! Arithmetic, comparison, and in-place operators all run through the ufunc table, so an
//! operator and its ufunc share dtype resolution, casting, and error reporting. Area modules
//! (reductions, shape, sorting, linear algebra) install further methods on the same type
//! through their own `ARRAY_METHODS` tables; see `mod.rs`.
//!
//! `flags` and `flat` return small Python helper objects defined in the frozen `numpy`
//! package, because they carry a reference to the array and support attribute or item
//! assignment.

use super::super::super::native::{
    CallArgs, GetterDef, MethodDef, NativeTypeDef, PyArrayBuffer, PyError, PyResult, PyRuntime,
    PyValue,
};
use super::super::super::Value;
use super::args::{self, Signature};
use super::array::{self, Array};
use super::convert;
use super::dtype::{Category, DType, Kind};
use super::index;
use super::layout::{self, Order};
use super::ufunc;

pub(in crate::python) static ARRAY_TYPE: NativeTypeDef = NativeTypeDef {
    name: "numpy.ndarray",
    methods: &[
        method("tolist", method_tolist),
        method("item", method_item),
        method("__float__", method_float),
        method("__int__", method_int),
        method("__complex__", method_complex),
        method("__index__", method_index),
        method("copy", method_copy),
        method("__copy__", method_copy),
        method("__deepcopy__", method_deepcopy),
        method("astype", method_astype),
        method("fill", method_fill),
        method("view", method_view),
        method("take", method_take),
        method("put", method_put),
        method("setflags", method_setflags),
        method("conjugate", method_conjugate),
        method("conj", method_conjugate),
        method("__iadd__", method_iadd),
        method("__isub__", method_isub),
        method("__imul__", method_imul),
        method("__itruediv__", method_itruediv),
        method("__ifloordiv__", method_ifloordiv),
        method("__imod__", method_imod),
        method("__ipow__", method_ipow),
        method("__iand__", method_iand),
        method("__ior__", method_ior),
        method("__ixor__", method_ixor),
        method("__ilshift__", method_ilshift),
        method("__irshift__", method_irshift),
    ],
    getters: &[
        getter("shape", get_shape),
        getter("ndim", get_ndim),
        getter("size", get_size),
        getter("dtype", get_dtype),
        getter("itemsize", get_itemsize),
        getter("nbytes", get_nbytes),
        getter("strides", get_strides),
        getter("T", get_transposed),
        getter("mT", get_matrix_transposed),
        getter("real", get_real),
        getter("imag", get_imag),
        getter("base", get_base),
        getter("flags", get_flags),
        getter("flat", get_flat),
    ],
};

const fn method(
    name: &'static str,
    call: fn(&mut dyn PyRuntime, PyValue, CallArgs) -> PyResult,
) -> MethodDef {
    MethodDef {
        type_name: "numpy.ndarray",
        name,
        call,
    }
}

const fn getter(name: &'static str, get: fn(&mut dyn PyRuntime, PyValue) -> PyResult) -> GetterDef {
    GetterDef {
        owner: "numpy.ndarray",
        name,
        get,
    }
}

fn receiver(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Array> {
    Array::from_value(runtime, value)
}

/// A tuple of Python ints.
pub(in crate::python) fn int_tuple(
    runtime: &mut dyn PyRuntime,
    values: impl IntoIterator<Item = i64>,
) -> PyResult {
    let items = values.into_iter().map(Value::Int).collect();
    runtime.new_tuple(items)
}

fn get_shape(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let array = receiver(runtime, value)?;
    int_tuple(
        runtime,
        array.shape().iter().map(|dimension| *dimension as i64),
    )
}

fn get_ndim(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(Value::Int(receiver(runtime, value)?.ndim() as i64))
}

fn get_size(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(Value::Int(receiver(runtime, value)?.size() as i64))
}

fn get_dtype(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let array = receiver(runtime, value)?;
    super::dtype_object::new(runtime, array.dtype)
}

fn get_itemsize(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(Value::Int(receiver(runtime, value)?.itemsize() as i64))
}

fn get_nbytes(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let array = receiver(runtime, value)?;
    Ok(Value::Int((array.size() * array.itemsize()) as i64))
}

fn get_strides(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let array = receiver(runtime, value)?;
    int_tuple(runtime, array.strides().iter().map(|stride| *stride as i64))
}

/// `a.T`: the view with axes reversed.
fn get_transposed(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let array = receiver(runtime, value)?;
    let shape = array.shape().iter().rev().copied().collect();
    let strides = array.strides().iter().rev().copied().collect();
    let view = array::new_view(
        runtime,
        &array,
        array.dtype,
        shape,
        strides,
        array.view.offset,
    )?;
    Ok(view.value())
}

/// `a.mT`: a view with the last two axes swapped, which NumPy requires at least two of.
fn get_matrix_transposed(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let array = receiver(runtime, value)?;
    let ndim = array.ndim();
    if ndim < 2 {
        return Err(PyError::value_error(
            "matrix transpose with ndim < 2 is undefined",
        ));
    }
    let mut shape = array.shape().to_vec();
    let mut strides = array.strides().to_vec();
    shape.swap(ndim - 2, ndim - 1);
    strides.swap(ndim - 2, ndim - 1);
    let view = array::new_view(
        runtime,
        &array,
        array.dtype,
        shape,
        strides,
        array.view.offset,
    )?;
    Ok(view.value())
}

/// `a.real`: a view of the real parts of a complex array, or the array's own view otherwise.
fn get_real(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let array = receiver(runtime, value)?;
    let dtype = if array.dtype.category() == Category::Complex {
        array.dtype.real_part()
    } else {
        array.dtype
    };
    let view = array::new_view(
        runtime,
        &array,
        dtype,
        array.shape().to_vec(),
        array.strides().to_vec(),
        array.view.offset,
    )?;
    Ok(view.value())
}

/// `a.imag`: a view of the imaginary parts of a complex array. Real arrays have a read-only
/// array of zeros, as in NumPy.
fn get_imag(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let array = receiver(runtime, value)?;
    if array.dtype.category() == Category::Complex {
        let dtype = array.dtype.real_part();
        let view = array::new_view(
            runtime,
            &array,
            dtype,
            array.shape().to_vec(),
            array.strides().to_vec(),
            array.view.offset + dtype.itemsize(),
        )?;
        return Ok(view.value());
    }
    if array.dtype.kind() == Kind::Object {
        return Err(PyError::not_implemented_error(
            "ndarray.imag of object arrays is not supported",
        ));
    }
    let buffer = array::zeroed_buffer(runtime, array.dtype, array.size())?;
    let zeros = array::new_array(runtime, buffer, array.dtype, array.shape().to_vec())?;
    runtime.set_array_writeable(zeros.handle, false)?;
    Ok(zeros.value())
}

fn get_base(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let array = receiver(runtime, value)?;
    Ok(runtime.array_base(array.handle)?.unwrap_or(Value::None))
}

/// Call a helper defined in the frozen `numpy` package with `array`.
pub(super) fn python_helper(runtime: &mut dyn PyRuntime, name: &str, array: PyValue) -> PyResult {
    let module = runtime.import_module("numpy")?;
    let helper = runtime
        .get_attribute(module, name)?
        .ok_or_else(|| PyError::runtime_error(format!("numpy.{name} is missing")))?;
    runtime.call_value(helper, CallArgs::new(vec![array], Vec::new()))
}

fn get_flags(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    python_helper(runtime, "_flagsobj", value)
}

fn get_flat(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    python_helper(runtime, "flatiter", value)
}

/// `a.setflags(write=None, align=None, uic=None)`. Only `write` is modeled.
fn method_setflags(
    runtime: &mut dyn PyRuntime,
    receiver_value: PyValue,
    args: CallArgs,
) -> PyResult {
    static SIGNATURE: Signature = Signature::new("setflags", &["write", "align", "uic"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let array = receiver(runtime, receiver_value)?;
    if let Some(write) = bound.value("write") {
        let write = runtime.truth(&write)?;
        if write && !array.view.writeable {
            // A view of read-only storage cannot become writeable; an owner can.
            if runtime.array_base(array.handle)?.is_some() {
                return Err(PyError::value_error(
                    "cannot set WRITEABLE flag to True of this array",
                ));
            }
        }
        runtime.set_array_writeable(array.handle, write)?;
    }
    Ok(Value::None)
}

/// Nested lists of Python scalars, or one scalar for a 0-d array.
pub(in crate::python) fn to_list(runtime: &mut dyn PyRuntime, array: &Array) -> PyResult {
    if array.ndim() == 0 {
        return convert::element_to_python(runtime, array, array.view.offset);
    }
    runtime.charge_cpu(array.size() as u64 + 1)?;
    runtime.reserve_memory(array.size().saturating_mul(24))?;
    list_level(runtime, array, 0, array.view.offset)
}

fn list_level(runtime: &mut dyn PyRuntime, array: &Array, axis: usize, offset: usize) -> PyResult {
    let length = array.shape()[axis];
    let stride = array.strides()[axis];
    let mut items = Vec::with_capacity(length);
    for position in 0..length {
        let offset = (offset as isize + position as isize * stride) as usize;
        items.push(if axis + 1 == array.ndim() {
            convert::element_to_python(runtime, array, offset)?
        } else {
            list_level(runtime, array, axis + 1, offset)?
        });
    }
    runtime.new_list(items)
}

fn method_tolist(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("tolist", 0, 0)?;
    args.reject_keywords("tolist")?;
    let array = receiver(runtime, receiver_value)?;
    to_list(runtime, &array)
}

/// Convert the element of a 0-d array with the builtin `type_name`, as NumPy's
/// `array_float`, `array_int`, and `array_complex` convert `a.item()`.
fn convert_item(runtime: &mut dyn PyRuntime, receiver_value: PyValue, type_name: &str) -> PyResult {
    let array = receiver(runtime, receiver_value)?;
    if array.ndim() != 0 {
        return Err(PyError::type_error(
            "only 0-dimensional arrays can be converted to Python scalars",
        ));
    }
    if type_name == "complex" && array.dtype.kind() == Kind::Str {
        return Err(PyError::type_error(format!(
            "Unable to convert {} to complex",
            array.dtype.repr()
        )));
    }
    let item = convert::element_to_python(runtime, &array, array.view.offset)?;
    let class = runtime
        .builtin_type(type_name)
        .ok_or_else(|| PyError::runtime_error(format!("builtin {type_name} is missing")))?;
    runtime.call_value(class, CallArgs::new(vec![item], Vec::new()))
}

fn method_float(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("__float__", 0, 0)?;
    convert_item(runtime, receiver_value, "float")
}

fn method_int(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("__int__", 0, 0)?;
    convert_item(runtime, receiver_value, "int")
}

fn method_complex(
    runtime: &mut dyn PyRuntime,
    receiver_value: PyValue,
    args: CallArgs,
) -> PyResult {
    args.expect_positional("__complex__", 0, 0)?;
    convert_item(runtime, receiver_value, "complex")
}

/// `operator.index(a)`: only 0-d integer arrays are indices.
fn method_index(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("__index__", 0, 0)?;
    let array = receiver(runtime, receiver_value)?;
    if array.ndim() != 0 || !array.dtype.is_integer() {
        return Err(PyError::type_error(
            "only integer scalar arrays can be converted to a scalar index",
        ));
    }
    convert::element_to_python(runtime, &array, array.view.offset)
}

/// `a.item(*index)`: one element as a Python scalar, by flat position or full index.
fn method_item(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    args.reject_keywords("item")?;
    let array = receiver(runtime, receiver_value)?;
    let positional = args.positional().to_vec();
    let offset = match positional.as_slice() {
        [] => {
            if array.size() != 1 {
                return Err(PyError::value_error(
                    "can only convert an array of size 1 to a Python scalar",
                ));
            }
            array.offsets().next().expect("one element")
        }
        [single] if runtime.kind(single)? != super::super::super::native::PyKind::Tuple => {
            let position = args::index_int(runtime, single)?;
            flat_offset(&array, position)?
        }
        values => {
            let values = if let [tuple] = values {
                let tuple = super::super::super::native::PyValueCast::cast(*tuple, runtime)?;
                runtime.tuple_items(tuple)?
            } else {
                values.to_vec()
            };
            if values.len() != array.ndim() {
                return Err(PyError::value_error(
                    "incorrect number of indices for array",
                ));
            }
            let mut index = Vec::with_capacity(values.len());
            for (axis, value) in values.iter().enumerate() {
                let position = args::index_int(runtime, value)?;
                let length = array.shape()[axis] as i64;
                let normalized = if position < 0 {
                    position + length
                } else {
                    position
                };
                if !(0..length).contains(&normalized) {
                    return Err(PyError::exception(
                        "IndexError",
                        format!(
                            "index {position} is out of bounds for axis {axis} with size {length}"
                        ),
                    ));
                }
                index.push(normalized as usize);
            }
            array.offset_of(&index)
        }
    };
    convert::element_to_python(runtime, &array, offset)
}

/// Byte offset of the element at C-order flat `position`.
pub(in crate::python) fn flat_offset(array: &Array, position: i64) -> PyResult<usize> {
    let size = array.size() as i64;
    let normalized = if position < 0 {
        position + size
    } else {
        position
    };
    if !(0..size).contains(&normalized) {
        return Err(PyError::exception(
            "IndexError",
            format!("index {position} is out of bounds for size {size}"),
        ));
    }
    let mut remaining = normalized as usize;
    let mut index = vec![0usize; array.ndim()];
    for axis in (0..array.ndim()).rev() {
        index[axis] = remaining % array.shape()[axis];
        remaining /= array.shape()[axis];
    }
    Ok(array.offset_of(&index))
}

/// `a.copy(order='C')`.
fn method_copy(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("copy", &["order"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let array = receiver(runtime, receiver_value)?;
    let order = Order::parse(runtime, bound.value("order"), Order::C)?;
    let axes = layout::axes_like(&array, order, array.ndim());
    Ok(layout::copy(runtime, &array, &axes)?.value())
}

/// `copy.deepcopy(a)`: object elements are deep-copied through the `copy` module.
fn method_deepcopy(
    runtime: &mut dyn PyRuntime,
    receiver_value: PyValue,
    args: CallArgs,
) -> PyResult {
    args.expect_positional("__deepcopy__", 1, 1)?;
    let array = receiver(runtime, receiver_value)?;
    let copy = array::copy_array(runtime, &array)?;
    if array.dtype.kind() != Kind::Object {
        return Ok(copy.value());
    }
    let module = runtime.import_module("copy")?;
    let deepcopy = runtime
        .get_attribute(module, "deepcopy")?
        .ok_or_else(|| PyError::runtime_error("copy.deepcopy is missing"))?;
    let memo = args.positional()[0];
    let values = array::read_objects(runtime, &copy)?;
    let mut copied = Vec::with_capacity(values.len());
    for value in values {
        copied.push(runtime.call_value(deepcopy, CallArgs::new(vec![value, memo], Vec::new()))?);
    }
    let offsets = copy.offsets().collect::<Vec<_>>();
    array::scatter(runtime, &copy, &offsets, &PyArrayBuffer::Values(copied))?;
    Ok(copy.value())
}

/// `a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)`.
fn method_astype(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("astype", &["dtype", "order", "casting", "subok", "copy"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let array = receiver(runtime, receiver_value)?;
    let target = args::dtype(runtime, bound.required("dtype"))?;
    if let Some(casting) = bound.value("casting") {
        let text = runtime.string_value(&casting)?.unwrap_or_default();
        let casting = super::dtype::Casting::parse(&text)?;
        if !super::dtype::can_cast(array.dtype, target, casting) {
            // NumPy names a 0-d source a scalar.
            let source = if array.ndim() == 0 {
                "scalar"
            } else {
                "array data"
            };
            return Err(PyError::type_error(format!(
                "Cannot cast {source} from {} to {} according to the rule '{}'",
                array.dtype.repr(),
                target.repr(),
                casting.name()
            )));
        }
    }
    let order = Order::parse(runtime, bound.value("order"), Order::K)?;
    let copy = args::flag(runtime, bound.get("copy"), true)?;
    let target = convert::cast_target(runtime, &array, target)?;
    // As in NumPy, `A` keeps any contiguous array rather than only a Fortran one.
    let layout_kept = match order {
        Order::A => array.is_c_contiguous() || layout::is_f_contiguous(&array),
        order => layout::satisfies(&array, order),
    };
    if !copy && target == array.dtype && layout_kept {
        return Ok(receiver_value);
    }
    let axes = layout::axes_like(&array, order, array.ndim());
    Ok(convert::cast_array_in(runtime, &array, target, &axes)?.value())
}

/// `a.fill(value)`.
fn method_fill(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("fill", 1, 1)?;
    args.reject_keywords("fill")?;
    let array = receiver(runtime, receiver_value)?;
    let value = args.positional()[0];
    let buffer = if array.dtype.kind() == Kind::Object {
        PyArrayBuffer::Values(vec![value])
    } else {
        convert::value_to_buffer(runtime, value, array.dtype)?
    };
    let element = array::new_array(runtime, buffer, array.dtype, Vec::new())?;
    array::assign(runtime, &array, &element)?;
    Ok(Value::None)
}

/// `a.view()` or `a.view(dtype)` for a dtype of the same item size.
fn method_view(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("view", &["dtype", "type"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let array = receiver(runtime, receiver_value)?;
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?.unwrap_or(array.dtype);
    if dtype != array.dtype {
        let objects = dtype.kind() == Kind::Object || array.dtype.kind() == Kind::Object;
        if objects {
            return Err(PyError::type_error(
                "Cannot change data-type for array of references.",
            ));
        }
        if dtype.itemsize() != array.itemsize() {
            return Err(PyError::not_implemented_error(
                "ndarray.view with a dtype of a different item size is not supported",
            ));
        }
        // Storage is little-endian whatever the dtype says, so reinterpreting it is faithful
        // only between dtypes of one byte order, and not when big-endian complex parts would
        // be split or joined.
        let complex =
            dtype.category() == Category::Complex || array.dtype.category() == Category::Complex;
        let big_endian = !dtype.is_native() || !array.dtype.is_native();
        if big_endian && (dtype.is_native() != array.dtype.is_native() || complex) {
            return Err(PyError::not_implemented_error(
                "ndarray.view between byte orders is not supported",
            ));
        }
    }
    let view = array::new_view(
        runtime,
        &array,
        dtype,
        array.shape().to_vec(),
        array.strides().to_vec(),
        array.view.offset,
    )?;
    Ok(view.value())
}

fn method_conjugate(
    runtime: &mut dyn PyRuntime,
    receiver_value: PyValue,
    args: CallArgs,
) -> PyResult {
    args.expect_positional("conjugate", 0, 0)?;
    let index = ufunc::find("conjugate").expect("conjugate is a ufunc");
    if receiver(runtime, receiver_value)?.dtype.category() != Category::Complex {
        return Ok(receiver_value);
    }
    ufunc::apply(
        runtime,
        index,
        &[receiver_value],
        &ufunc::Options::default(),
    )
}

/// `take`'s options: `out=` and `mode=`.
pub(in crate::python) struct TakeOptions {
    pub out: Option<Array>,
    pub mode: index::ClipMode,
}

impl TakeOptions {
    pub(in crate::python) fn parse(
        runtime: &mut dyn PyRuntime,
        bound: &args::Bound,
    ) -> PyResult<Self> {
        let out = bound
            .value("out")
            .map(|out| Array::from_value(runtime, out))
            .transpose()?;
        let mode = index::ClipMode::parse(runtime, bound.get("mode"))?;
        Ok(Self { out, mode })
    }
}

/// `np.take(a, indices, axis=None, out=None, mode='raise')` and `a.take(...)`, following
/// `PyArray_TakeFrom`.
pub(in crate::python) fn take(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    indices: PyValue,
    axis: Option<PyValue>,
    options: TakeOptions,
) -> PyResult {
    let positions = index::index_array(runtime, indices)?;
    let axis = args::axis(runtime, axis, array.ndim())?;
    let length = match axis {
        Some(axis) => array.shape()[axis],
        None => array.size(),
    };
    if length == 0 && positions.size() != 0 {
        return Err(PyError::exception(
            "IndexError",
            "cannot do a non-empty take from an empty axes.",
        ));
    }
    let result = match axis {
        None => {
            let flat = index::flat_positions(runtime, &positions, length, None, options.mode)?;
            let all = array.offsets().collect::<Vec<_>>();
            let offsets = flat
                .iter()
                .map(|position| all[*position])
                .collect::<Vec<_>>();
            index::gather(runtime, array, &offsets, positions.shape().to_vec())?
        }
        Some(axis) => {
            let chosen =
                index::flat_positions(runtime, &positions, length, Some(axis), options.mode)?;
            take_along(runtime, array, axis, &chosen, positions.shape())?
        }
    };
    if let Some(out) = options.out {
        if out.shape() != result.shape() {
            return Err(PyError::value_error(
                "output array does not match result of ndarray.take",
            ));
        }
        array::assign(runtime, &out, &result)?;
        return Ok(out.value());
    }
    // Like `PyArray_Return`, a 0-d result becomes a scalar.
    if result.ndim() == 0 {
        return convert::element_to_scalar(runtime, &result, result.view.offset);
    }
    Ok(result.value())
}

/// The elements at positions `chosen` along `axis`, which take the place of that axis with
/// `index_shape`.
fn take_along(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    axis: usize,
    chosen: &[usize],
    index_shape: &[usize],
) -> PyResult<Array> {
    let stride = array.strides()[axis];
    let before = &array.shape()[..axis];
    let after = &array.shape()[axis + 1..];
    let outer = index::relative_offsets(before, &array.strides()[..axis]);
    let inner = index::relative_offsets(after, &array.strides()[axis + 1..]);
    let mut shape = before.to_vec();
    shape.extend_from_slice(index_shape);
    shape.extend_from_slice(after);
    let count = array::element_count(&shape)?;
    runtime.reserve_memory(count.saturating_mul(8))?;
    let mut offsets = Vec::with_capacity(count);
    for outer in &outer {
        for position in chosen {
            for inner in &inner {
                offsets.push(
                    (array.view.offset as isize + outer + *position as isize * stride + inner)
                        as usize,
                );
            }
        }
    }
    index::gather(runtime, array, &offsets, shape)
}

fn method_take(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("take", &["indices", "axis", "out", "mode"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let options = TakeOptions::parse(runtime, &bound)?;
    let array = receiver(runtime, receiver_value)?;
    take(
        runtime,
        &array,
        bound.required("indices"),
        bound.value("axis"),
        options,
    )
}

/// `np.put(a, indices, values)`: store `values`, cycled, at C-order flat positions.
pub(in crate::python) fn put(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    indices: PyValue,
    values: PyValue,
    mode: index::ClipMode,
) -> PyResult {
    let positions = index::index_array(runtime, indices)?;
    let flat = index::flat_positions(runtime, &positions, array.size(), None, mode)?;
    let source = if array.dtype.kind() == Kind::Object {
        convert::array_from_python(runtime, values, Some(DType::OBJECT), false)?
    } else {
        convert::array_from_python(runtime, values, Some(array.dtype), false)?
    };
    let source = array::copy_array(runtime, &source)?;
    if flat.is_empty() {
        return Ok(Value::None);
    }
    if source.size() == 0 {
        return Err(PyError::value_error(
            "cannot put with an empty values array",
        ));
    }
    let buffer = array::contiguous_buffer(runtime, &source)?;
    let all = array.offsets().collect::<Vec<_>>();
    let offsets = flat
        .iter()
        .map(|position| all[*position])
        .collect::<Vec<_>>();
    let cycled = cycle_buffer(buffer, source.itemsize(), source.size(), offsets.len());
    array::scatter(runtime, array, &offsets, &cycled)?;
    Ok(Value::None)
}

/// Repeat a buffer of `count` elements cyclically to `length` elements.
fn cycle_buffer(
    buffer: PyArrayBuffer,
    itemsize: usize,
    count: usize,
    length: usize,
) -> PyArrayBuffer {
    match buffer {
        PyArrayBuffer::Bytes(bytes) => PyArrayBuffer::Bytes(
            (0..length)
                .flat_map(|index| {
                    let start = (index % count) * itemsize;
                    bytes[start..start + itemsize].to_vec()
                })
                .collect(),
        ),
        PyArrayBuffer::Values(values) => {
            PyArrayBuffer::Values((0..length).map(|index| values[index % count]).collect())
        }
    }
}

fn method_put(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("put", &["indices", "values", "mode"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let mode = index::ClipMode::parse(runtime, bound.get("mode"))?;
    let array = receiver(runtime, receiver_value)?;
    put(
        runtime,
        &array,
        bound.required("indices"),
        bound.required("values"),
        mode,
    )
}

macro_rules! inplace_methods {
    ($($method:ident => $ufunc:literal;)*) => {
        $(
            fn $method(
                runtime: &mut dyn PyRuntime,
                receiver_value: PyValue,
                args: CallArgs,
            ) -> PyResult {
                args.expect_positional(stringify!($method), 1, 1)?;
                ufunc::inplace(runtime, $ufunc, receiver_value, args.positional()[0])
            }
        )*
    };
}

inplace_methods! {
    method_iadd => "add";
    method_isub => "subtract";
    method_imul => "multiply";
    method_itruediv => "divide";
    method_ifloordiv => "floor_divide";
    method_imod => "remainder";
    method_ipow => "power";
    method_iand => "bitwise_and";
    method_ior => "bitwise_or";
    method_ixor => "bitwise_xor";
    method_ilshift => "left_shift";
    method_irshift => "right_shift";
}

/// `repr(array)` or `str(array)` through `numpy._printing.<function>`, which holds the layout
/// rules (bracket nesting, column alignment, summarization, `dtype=`/`shape=` suffixes).
fn array_text(runtime: &mut dyn PyRuntime, array: PyValue, function: &str) -> PyResult<PyValue> {
    let module = runtime.import_module("numpy._printing")?;
    let implementation = runtime
        .get_attribute(module, function)?
        .ok_or_else(|| PyError::runtime_error(format!("numpy._printing.{function} is missing")))?;
    runtime.call_value(implementation, CallArgs::new(vec![array], Vec::new()))
}

pub(in crate::python) fn slot_repr(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    array_text(runtime, value, "_array_repr_implementation").map(Some)
}

pub(in crate::python) fn slot_str(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    array_text(runtime, value, "_array_str_implementation").map(Some)
}

pub(in crate::python) fn slot_bool(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = receiver(runtime, value)?;
    match array.size() {
        1 => {
            let truth = index::truth_values(runtime, &array)?;
            Ok(Some(Value::Bool(truth[0])))
        }
        0 => Err(PyError::value_error(
            "The truth value of an empty array is ambiguous. Use `array.size > 0` to check that \
             an array is not empty.",
        )),
        _ => Err(PyError::value_error(
            "The truth value of an array with more than one element is ambiguous. Use a.any() or \
             a.all()",
        )),
    }
}

pub(in crate::python) fn slot_length(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = receiver(runtime, value)?;
    match array.shape().first() {
        Some(length) => Ok(Some(Value::Int(*length as i64))),
        None => Err(PyError::type_error("len() of unsized object")),
    }
}

/// Rows along the first axis: views for n-d arrays, scalars for 1-d arrays.
pub(in crate::python) fn rows(
    runtime: &mut dyn PyRuntime,
    array: &Array,
) -> PyResult<Vec<PyValue>> {
    let length = array.shape()[0];
    runtime.charge_cpu(length as u64 + 1)?;
    let mut rows = Vec::with_capacity(length);
    for position in 0..length {
        let offset = (array.view.offset as isize + position as isize * array.strides()[0]) as usize;
        rows.push(if array.ndim() == 1 {
            convert::element_to_scalar(runtime, array, offset)?
        } else {
            array::new_view(
                runtime,
                array,
                array.dtype,
                array.shape()[1..].to_vec(),
                array.strides()[1..].to_vec(),
                offset,
            )?
            .value()
        });
    }
    Ok(rows)
}

pub(in crate::python) fn slot_iter(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = receiver(runtime, value)?;
    if array.ndim() == 0 {
        return Err(PyError::type_error("iteration over a 0-d array"));
    }
    let rows = rows(runtime, &array)?;
    runtime.new_iterator(rows).map(Some)
}

pub(in crate::python) fn slot_get_item(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    index: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = receiver(runtime, value)?;
    index::get_item(runtime, &array, index).map(Some)
}

pub(in crate::python) fn slot_set_item(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    index: PyValue,
    item: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = receiver(runtime, value)?;
    index::set_item(runtime, &array, index, item)?;
    Ok(Some(Value::None))
}

pub(in crate::python) fn slot_matrix_multiply(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    if super::ufunc::defers_to(runtime, right)? {
        return Ok(None);
    }
    super::products::matmul(runtime, left, right).map(Some)
}

pub(in crate::python) fn slot_reflected_matrix_multiply(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    if super::ufunc::defers_to(runtime, right)? {
        return Ok(None);
    }
    super::products::matmul(runtime, right, left).map(Some)
}

/// `_numpy._c_contiguous(a)`, backing `a.flags.c_contiguous`.
pub(in crate::python) fn c_contiguous(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_c_contiguous", 1, 1)?;
    let array = receiver(runtime, args.positional()[0])?;
    Ok(Value::Bool(array.is_c_contiguous()))
}

/// `_numpy._writeable(a)`, backing `a.flags.writeable`.
pub(in crate::python) fn writeable(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_writeable", 1, 1)?;
    let array = receiver(runtime, args.positional()[0])?;
    Ok(Value::Bool(array.view.writeable))
}
