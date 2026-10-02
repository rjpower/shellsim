//! Array constructors and dtype queries: `np.array`, `zeros`, `full`, `arange`, `fromiter`,
//! `result_type`, `can_cast`, `issubdtype`, and friends.
//!
//! Constructors reserve memory for the whole result before filling it and charge CPU per
//! element. Values are computed in `f64` or `i64` and written with the target dtype's cast
//! rules, which matches NumPy for every dtype these constructors produce. `linspace`, `logspace`,
//! `geomspace`, `eye`, `identity`, `diag`, `meshgrid`, and the `*_like` family are frozen Python
//! in `numpy._creation`, built on `arange`/`zeros`/`empty`/`full` plus reshape and slicing.

use super::super::super::native::{
    CallArgs, KindBase, PyError, PyKind, PyNativeKind, PyResult, PyRuntime, PyTypeObject, PyValue,
    ValueKindDef,
};
use super::super::super::Value;
use super::args::{self, Signature};
use super::array::{self, Array};
use super::convert::{self, Leaf};
use super::dtype::{self, Casting, DType, Kind, Weak};
use super::element::{self, Number};
use super::layout::{self, Order};

/// `np.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)`.
pub(in crate::python) fn array<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    static SIGNATURE: Signature = Signature::new("array", &["object", "dtype"], 1)
        .keyword_only(&["copy", "order", "subok", "ndmin", "like"]);
    let bound = SIGNATURE.bind(&args)?;
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?;
    let copy = match bound.get("copy") {
        None => Copy::Always,
        Some(value) => copy_mode(runtime, value)?,
    };
    let order = Order::parse(runtime, bound.value("order"), Order::K)?;
    let ndmin = args::optional_int(runtime, bound.value("ndmin"))?.unwrap_or(0);
    let ndmin = usize::try_from(ndmin).unwrap_or(0);
    Ok(from_object(runtime, bound.required("object"), dtype, copy, order, ndmin)?.value())
}

/// NumPy's `copy` argument to array constructors.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Copy {
    /// `True`: always copy.
    Always,
    /// `None`: copy only when the dtype or layout requires it.
    IfNeeded,
    /// `False`: raise instead of copying.
    Never,
}

fn copy_mode<'s>(runtime: &mut dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Copy> {
    Ok(if value.is_none() {
        Copy::IfNeeded
    } else if runtime.truth(&value)? {
        Copy::Always
    } else {
        Copy::Never
    })
}

/// NumPy's `_array_fromobject_generic`: the array behind `np.array`, `np.asarray`, and their
/// contiguous variants. An existing array is returned unchanged when neither its dtype nor its
/// layout needs to change and `copy` allows it; otherwise it is copied in `order`, resolved
/// against the source. Nested sequences are built in C order, or Fortran order when asked.
fn from_object<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    object: PyValue<'s>,
    dtype: Option<DType>,
    copy: Copy,
    order: Order,
    ndmin: usize,
) -> PyResult<'s, Array<'s>> {
    let no_copy =
        || PyError::value_error("Unable to avoid copy while creating an array as requested.");
    let result = if runtime.native_kind(&object)? == Some(PyNativeKind::Array) {
        let source = Array::from_value(runtime, object)?;
        let target = match dtype {
            Some(dtype) => convert::cast_target(runtime, &source, dtype)?,
            None => source.dtype,
        };
        if target == source.dtype && copy != Copy::Always && layout::satisfies(&source, order) {
            source
        } else if copy == Copy::Never {
            return Err(no_copy());
        } else {
            // A converting copy keeps a Fortran source's order unless C order is requested,
            // and otherwise follows the source's strides even for `A`.
            let order = match order {
                Order::A if target != source.dtype => Order::K,
                order => order,
            };
            let axes = layout::axes_like(&source, order, source.ndim());
            convert::cast_array_in(runtime, &source, target, &axes)?
        }
    } else {
        if copy == Copy::Never {
            return Err(no_copy());
        }
        let array = convert::array_from_python(runtime, object, dtype, false)?;
        if order == Order::F && !layout::is_f_contiguous(&array) {
            layout::copy(runtime, &array, &layout::axes(Order::F, array.ndim()))?
        } else {
            array
        }
    };
    with_ndmin(runtime, result, ndmin, order)
}

/// Prepend length-one axes until the array has `ndmin` dimensions, as NumPy's `_prepend_ones`
/// does: the new axes take the item size as their stride for Fortran order, and the extent of
/// the outermost axis otherwise.
fn with_ndmin<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: Array<'s>,
    ndmin: usize,
    order: Order,
) -> PyResult<'s, Array<'s>> {
    let missing = ndmin.saturating_sub(array.ndim());
    if missing == 0 {
        return Ok(array);
    }
    let stride = if order == Order::F || layout::is_fortran(&array) || array.ndim() == 0 {
        array.itemsize() as isize
    } else {
        array.strides()[0].saturating_mul(array.shape()[0] as isize)
    };
    let mut shape = vec![1; missing];
    shape.extend_from_slice(array.shape());
    let mut strides = vec![stride; missing];
    strides.extend_from_slice(array.strides());
    array::new_view(
        runtime,
        &array,
        array.dtype,
        shape,
        strides,
        array.view.offset,
    )
}

/// `np.asarray(a, dtype=None, order=None, *, copy=None)`, which `np.asanyarray` shares.
pub(in crate::python) fn asarray<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    static SIGNATURE: Signature = Signature::new("asarray", &["a", "dtype", "order"], 1)
        .keyword_only(&["copy", "like", "device"]);
    let bound = SIGNATURE.bind(&args)?;
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?;
    let order = Order::parse(runtime, bound.value("order"), Order::K)?;
    let copy = match bound.value("copy") {
        None => Copy::IfNeeded,
        Some(value) => copy_mode(runtime, value)?,
    };
    Ok(from_object(runtime, bound.required("a"), dtype, copy, order, 0)?.value())
}

/// `np.ascontiguousarray(a, dtype=None)`: a C-contiguous array of at least one dimension,
/// copying only when needed.
pub(in crate::python) fn ascontiguousarray<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    contiguous(runtime, &args, "ascontiguousarray", Order::C)
}

/// `np.asfortranarray(a, dtype=None)`: a Fortran-contiguous array of at least one dimension,
/// copying only when needed.
pub(in crate::python) fn asfortranarray<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    contiguous(runtime, &args, "asfortranarray", Order::F)
}

fn contiguous<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: &CallArgs<'s>,
    name: &'static str,
    order: Order,
) -> PyResult<'s> {
    static C: Signature =
        Signature::new("ascontiguousarray", &["a", "dtype"], 1).keyword_only(&["like"]);
    static F: Signature =
        Signature::new("asfortranarray", &["a", "dtype"], 1).keyword_only(&["like"]);
    let bound = if name == "asfortranarray" { &F } else { &C }.bind(args)?;
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?;
    let object = bound.required("a");
    Ok(from_object(runtime, object, dtype, Copy::IfNeeded, order, 1)?.value())
}

/// `np.copy(a, order='K', subok=False)`.
pub(in crate::python) fn copy<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    static SIGNATURE: Signature = Signature::new("copy", &["a", "order", "subok"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let order = Order::parse(runtime, bound.value("order"), Order::K)?;
    Ok(from_object(runtime, bound.required("a"), None, Copy::Always, order, 0)?.value())
}

/// What a new array's elements start as.
#[derive(Clone, Copy)]
enum Fill<'s> {
    /// Zero bytes, which `object` arrays read as `None`, as `np.empty` leaves them.
    Empty,
    /// Zero, which `object` arrays hold as the int `0`, as `np.zeros` fills them.
    Zero,
    Value(PyValue<'s>),
}

/// An array of `shape`, laid out in `axes` order, whose every element is `fill`, already
/// converted to `dtype` storage.
fn filled<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    shape: Vec<usize>,
    axes: &[usize],
    dtype: DType,
    fill: Fill<'s>,
) -> PyResult<'s, Array<'s>> {
    let count = array::element_count(&shape)?;
    let fill = match fill {
        Fill::Zero if dtype.kind() == Kind::Object => Value::Int(0),
        Fill::Empty | Fill::Zero => {
            let buffer = array::zeroed_buffer(runtime, dtype, count)?;
            return layout::new_array(runtime, buffer, dtype, shape, axes);
        }
        Fill::Value(fill) => fill,
    };
    let element = if dtype.kind() == Kind::Object {
        let buffer = super::super::super::native::PyArrayBuffer::Values(vec![fill]);
        array::new_array(runtime, buffer, dtype, Vec::new())?
    } else {
        let source = convert::array_from_python(runtime, fill, None, false)?;
        convert::cast_array(runtime, &source, dtype, false)?
    };
    // Every element is the same, so the buffer is already in any memory order.
    let buffer = array::broadcast_buffer(runtime, &element, dtype, &shape)?;
    layout::new_array(runtime, buffer, dtype, shape, axes)
}

/// `np.zeros`, `np.ones`, and `np.empty` share one signature; `empty` is zero-filled.
fn shaped<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: &CallArgs<'s>,
    name: &'static str,
    fill: Fill<'s>,
) -> PyResult<'s> {
    static ZEROS: Signature =
        Signature::new("zeros", &["shape", "dtype", "order"], 1).keyword_only(&["like", "device"]);
    static ONES: Signature =
        Signature::new("ones", &["shape", "dtype", "order"], 1).keyword_only(&["like", "device"]);
    static EMPTY: Signature =
        Signature::new("empty", &["shape", "dtype", "order"], 1).keyword_only(&["like", "device"]);
    let signature = match name {
        "zeros" => &ZEROS,
        "ones" => &ONES,
        _ => &EMPTY,
    };
    let bound = signature.bind(args)?;
    let shape = args::shape(runtime, bound.required("shape"))?;
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?.unwrap_or(DType::FLOAT64);
    let axes = layout::axes(
        Order::parse_new(runtime, bound.value("order"))?,
        shape.len(),
    );
    Ok(filled(runtime, shape, &axes, dtype, fill)?.value())
}

pub(in crate::python) fn zeros<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    shaped(runtime, &args, "zeros", Fill::Zero)
}

pub(in crate::python) fn empty<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    shaped(runtime, &args, "empty", Fill::Empty)
}

pub(in crate::python) fn ones<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    shaped(runtime, &args, "ones", Fill::Value(Value::Int(1)))
}

/// `np.full(shape, fill_value, dtype=None, order='C')`: the dtype defaults to the fill value's.
pub(in crate::python) fn full<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    static SIGNATURE: Signature =
        Signature::new("full", &["shape", "fill_value", "dtype", "order"], 2)
            .keyword_only(&["like", "device"]);
    let bound = SIGNATURE.bind(&args)?;
    let shape = args::shape(runtime, bound.required("shape"))?;
    let fill = bound.required("fill_value");
    let dtype = match args::optional_dtype(runtime, bound.value("dtype"))? {
        Some(dtype) => dtype,
        None => convert::array_from_python(runtime, fill, None, false)?.dtype,
    };
    let axes = layout::axes(
        Order::parse_new(runtime, bound.value("order"))?,
        shape.len(),
    );
    Ok(filled(runtime, shape, &axes, dtype, Fill::Value(fill))?.value())
}

/// A float or int argument of a range constructor. A 0-d array stands for its element.
fn range_number<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    value: &PyValue<'s>,
) -> PyResult<'s, (Number, bool)> {
    if runtime.native_kind(value)? == Some(PyNativeKind::Array) {
        let array = Array::from_value(runtime, *value)?;
        if array.view.shape.is_empty() {
            let element = convert::element_to_scalar(runtime, &array, array.view.offset)?;
            return range_number(runtime, &element);
        }
        // NumPy compares the bounds before converting them, so an array with no or several
        // elements fails on its truth value, and a one-element array on the conversion.
        if array.view.shape.iter().product::<usize>() != 1 {
            runtime.truth(value)?;
        }
        return Err(PyError::type_error(
            "only 0-dimensional arrays can be converted to Python scalars",
        ));
    }
    match convert::leaf(runtime, value)? {
        Leaf::Bool(value) => Ok((Number::Int(i64::from(value)), false)),
        Leaf::Int(value) => i64::try_from(value)
            .map(|value| (Number::Int(value), false))
            .map_err(|_| PyError::overflow_error("Python int too large to convert to C long")),
        Leaf::Float(value) => Ok((Number::Float(value), true)),
        Leaf::NumPy(dtype, number) => Ok((number, dtype.is_inexact())),
        Leaf::Complex(..) => Err(PyError::type_error(
            "arange does not support complex arguments",
        )),
        _ => Err(PyError::type_error(format!(
            "unsupported operand type for arange: '{}'",
            runtime.type_name(value)?
        ))),
    }
}

/// `np.arange([start, ]stop[, step], dtype=None)`.
pub(in crate::python) fn arange<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    static SIGNATURE: Signature = Signature::new("arange", &["start", "stop", "step", "dtype"], 0)
        .keyword_only(&["like", "device"]);
    let bound = SIGNATURE.bind(&args)?;
    let (start, stop) = match (bound.value("start"), bound.value("stop")) {
        (Some(start), Some(stop)) => (Some(start), stop),
        (Some(stop), None) | (None, Some(stop)) => (None, stop),
        (None, None) => {
            return Err(PyError::type_error(
                "arange() requires stop to be specified.",
            ))
        }
    };
    let (start, start_float) = match start {
        Some(start) => range_number(runtime, &start)?,
        None => (Number::Int(0), false),
    };
    let (stop, stop_float) = range_number(runtime, &stop)?;
    let (step, step_float) = match bound.value("step") {
        Some(step) => range_number(runtime, &step)?,
        None => (Number::Int(1), false),
    };
    let inferred = if start_float || stop_float || step_float {
        DType::FLOAT64
    } else {
        DType::INT64
    };
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?.unwrap_or(inferred);
    let (start, stop, step) = (start.as_f64(), stop.as_f64(), step.as_f64());
    if step == 0.0 {
        return Err(PyError::exception("ZeroDivisionError", "division by zero"));
    }
    let length = ((stop - start) / step).ceil();
    let length = if length.is_nan() || length <= 0.0 {
        0
    } else if length > usize::MAX as f64 {
        return Err(PyError::value_error("Maximum allowed size exceeded"));
    } else {
        length as usize
    };
    array::reserve_elements(runtime, dtype, length)?;
    runtime.charge_cpu(length as u64 + 1)?;
    let integral = inferred == DType::INT64 && (dtype.is_integer() || dtype.kind() == Kind::Object);
    // NumPy fills a float range from its first two stored elements, so element `i` is
    // `start + i * ((start + step) - start)` in the result's precision, which can differ from
    // `start + i * step` in the last bit.
    let narrow = |value: f64| {
        if dtype == DType::FLOAT32 {
            f64::from(value as f32)
        } else {
            value
        }
    };
    let first = narrow(start);
    let delta = narrow(narrow(start + step) - first);
    let values = (0..length).map(|index| {
        if integral {
            Number::Int(start as i64 + index as i64 * step as i64)
        } else {
            Number::Float(first + index as f64 * delta)
        }
    });
    Ok(numbers_array(runtime, dtype, vec![length], values)?.value())
}

/// A new array from numbers, cast into `dtype`. An `object` array holds them as Python
/// numbers, as `np.arange(3, dtype=object)` holds ints.
pub(in crate::python) fn numbers_array<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    dtype: DType,
    shape: Vec<usize>,
    values: impl Iterator<Item = Number>,
) -> PyResult<'s, Array<'s>> {
    if dtype.kind() == Kind::Object {
        let count = array::element_count(&shape)?;
        array::reserve_elements(runtime, dtype, count)?;
        let mut objects = Vec::with_capacity(count);
        for value in values.take(count) {
            objects.push(super::scalar::number_to_python(runtime, value)?);
        }
        let buffer = super::super::super::native::PyArrayBuffer::Values(objects);
        return array::new_array(runtime, buffer, dtype, shape);
    }
    if !dtype.is_numeric() {
        return Err(PyError::not_implemented_error(format!(
            "{} arrays cannot be built from numbers here",
            dtype.name()
        )));
    }
    let count = array::element_count(&shape)?;
    array::reserve_elements(runtime, dtype, count)?;
    let itemsize = dtype.itemsize();
    let mut bytes = vec![0u8; count * itemsize];
    for (value, chunk) in values.zip(bytes.chunks_exact_mut(itemsize)) {
        element::write_number(dtype.kind(), value, chunk);
    }
    array::new_array(
        runtime,
        super::super::super::native::PyArrayBuffer::Bytes(bytes),
        dtype,
        shape,
    )
}

/// `np.fromiter(iter, dtype, count=-1)`.
pub(in crate::python) fn fromiter<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    static SIGNATURE: Signature =
        Signature::new("fromiter", &["iter", "dtype", "count"], 2).keyword_only(&["like"]);
    let bound = SIGNATURE.bind(&args)?;
    let dtype = args::dtype(runtime, bound.required("dtype"))?;
    let count = args::optional_int(runtime, bound.value("count"))?.unwrap_or(-1);
    let iterator = runtime.iterator(bound.required("iter"))?;
    let mut values = Vec::new();
    while count < 0 || (values.len() as i64) < count {
        let Some(value) = runtime.iterator_next(iterator)? else {
            break;
        };
        runtime.charge_cpu(1)?;
        runtime.reserve_memory(16)?;
        values.push(value);
    }
    if count >= 0 && (values.len() as i64) < count {
        return Err(PyError::value_error(format!(
            "iterator too short: Expected {count} but iterator had only {} items.",
            values.len()
        )));
    }
    let length = values.len();
    let list = runtime.new_list(values)?;
    let result = convert::array_from_python(runtime, list, Some(dtype), false)?;
    if result.ndim() != 1 || result.size() != length {
        return Err(PyError::value_error(
            "setting an array element with a sequence.",
        ));
    }
    Ok(result.value())
}

/// Whether two arrays can share memory, and whether they do. `exact` compares the bytes each
/// element covers; otherwise overlapping extents are enough.
fn memory_overlap<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
    name: &str,
    exact: bool,
) -> PyResult<'s> {
    args.expect_positional(name, 2, 2)?;
    let values = args.positional();
    let arrays = values
        .iter()
        .map(|value| {
            if runtime.native_kind(value)? == Some(PyNativeKind::Array) {
                Array::from_value(runtime, *value).map(Some)
            } else {
                Ok(None)
            }
        })
        .collect::<PyResult<'s, Vec<_>>>()?;
    let (Some(left), Some(right)) = (&arrays[0], &arrays[1]) else {
        return Ok(Value::Bool(false));
    };
    if runtime.array_storage(left.handle)? != runtime.array_storage(right.handle)?
        || left.size() == 0
        || right.size() == 0
    {
        return Ok(Value::Bool(false));
    }
    let extent = |array: &Array| {
        let offsets = array.offsets();
        let (low, high) = offsets.fold((usize::MAX, 0), |(low, high), offset| {
            (low.min(offset), high.max(offset + array.itemsize()))
        });
        (low, high)
    };
    runtime.charge_cpu((left.size() + right.size()) as u64 + 1)?;
    let ((left_low, left_high), (right_low, right_high)) = (extent(left), extent(right));
    if left_high <= right_low || right_high <= left_low {
        return Ok(Value::Bool(false));
    }
    if !exact {
        return Ok(Value::Bool(true));
    }
    runtime.reserve_memory((left.size() + right.size()).saturating_mul(16))?;
    let mut ranges = left
        .offsets()
        .map(|offset| (offset, offset + left.itemsize(), 0u8))
        .chain(
            right
                .offsets()
                .map(|offset| (offset, offset + right.itemsize(), 1u8)),
        )
        .collect::<Vec<_>>();
    ranges.sort_unstable();
    // Track the furthest end reached by each side; any start before the other side's end
    // is shared memory.
    let mut ends = [0usize; 2];
    for (start, end, side) in ranges {
        let other = usize::from(1 - side);
        if start < ends[other] {
            return Ok(Value::Bool(true));
        }
        ends[usize::from(side)] = ends[usize::from(side)].max(end);
    }
    Ok(Value::Bool(false))
}

pub(in crate::python) fn shares_memory<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let (positional, _) = args.into_parts();
    memory_overlap(
        runtime,
        CallArgs::new(positional, Vec::new()),
        "shares_memory",
        true,
    )
}

pub(in crate::python) fn may_share_memory<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let (positional, _) = args.into_parts();
    memory_overlap(
        runtime,
        CallArgs::new(positional, Vec::new()),
        "may_share_memory",
        false,
    )
}

/// `np.ndim(a)`, `np.shape(a)`, and `np.size(a)` accept any array-like.
pub(in crate::python) fn ndim<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("ndim", 1, 1)?;
    let array = convert::as_array(runtime, args.positional()[0])?;
    Ok(Value::Int(array.ndim() as i64))
}

pub(in crate::python) fn shape<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("shape", 1, 1)?;
    let array = convert::as_array(runtime, args.positional()[0])?;
    super::ndarray::int_tuple(
        runtime,
        array.shape().iter().map(|dimension| *dimension as i64),
    )
}

pub(in crate::python) fn size<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    static SIGNATURE: Signature = Signature::new("size", &["a", "axis"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    Ok(Value::Int(
        match args::axis(runtime, bound.value("axis"), array.ndim())? {
            Some(axis) => array.shape()[axis],
            None => array.size(),
        } as i64,
    ))
}

/// `np.take(a, indices, axis=None, out=None, mode='raise')`.
pub(in crate::python) fn take<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    static SIGNATURE: Signature =
        Signature::new("take", &["a", "indices", "axis", "out", "mode"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let options = super::ndarray::TakeOptions::parse(runtime, &bound)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    super::ndarray::take(
        runtime,
        &array,
        bound.required("indices"),
        bound.value("axis"),
        options,
    )
}

/// `np.put(a, ind, v, mode='raise')`.
pub(in crate::python) fn put<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    static SIGNATURE: Signature = Signature::new("put", &["a", "ind", "v", "mode"], 3);
    let bound = SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, bound.required("a"))
        .map_err(|_| PyError::type_error("argument 1 must be numpy.ndarray"))?;
    let mode = super::index::ClipMode::parse(runtime, bound.get("mode"))?;
    super::ndarray::put(
        runtime,
        &array,
        bound.required("ind"),
        bound.required("v"),
        mode,
    )
}

/// One operand of `result_type`: a strong dtype, or a weak Python scalar.
enum TypeOperand {
    Strong(DType),
    Weak(Weak),
}

fn type_operand<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    value: PyValue<'s>,
) -> PyResult<'s, TypeOperand> {
    if let Some((weak, _)) = convert::weak_scalar(runtime, &value)? {
        return Ok(TypeOperand::Weak(weak));
    }
    if let Some((dtype, _)) = super::scalar::unbox(runtime, &value) {
        return Ok(TypeOperand::Strong(dtype));
    }
    if runtime.native_kind(&value)? == Some(PyNativeKind::Array) {
        return Ok(TypeOperand::Strong(
            Array::from_value(runtime, value)?.dtype,
        ));
    }
    args::dtype(runtime, value).map(TypeOperand::Strong)
}

/// `np.result_type(*arrays_and_dtypes)` under NEP 50.
pub(in crate::python) fn result_type<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.reject_keywords("result_type")?;
    if args.positional().is_empty() {
        return Err(PyError::value_error(
            "at least one array or dtype is required",
        ));
    }
    let mut strong = Vec::new();
    let mut weak = Vec::new();
    for value in args.positional() {
        match type_operand(runtime, *value)? {
            TypeOperand::Strong(dtype) => strong.push(dtype),
            TypeOperand::Weak(kind) => weak.push(kind),
        }
    }
    let dtype = dtype::result_type(&strong, &weak)?;
    super::dtype_object::new(runtime, dtype)
}

/// `np.promote_types(type1, type2)`.
pub(in crate::python) fn promote_types<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("promote_types", 2, 2)?;
    let left = args::dtype(runtime, args.positional()[0])?;
    let right = args::dtype(runtime, args.positional()[1])?;
    let dtype = dtype::promote(left, right)?;
    super::dtype_object::new(runtime, dtype)
}

/// `np.can_cast(from_, to, casting='safe')`.
pub(in crate::python) fn can_cast<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    static SIGNATURE: Signature = Signature::new("can_cast", &["from_", "to", "casting"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let from =
        match type_operand(runtime, bound.required("from_"))? {
            TypeOperand::Strong(dtype) => dtype,
            TypeOperand::Weak(_) => return Err(PyError::type_error(
                "can_cast() does not support Python ints, floats, and complex because the result \
                 used to depend on the value.\nThis change was part of adopting NEP 50, we may \
                 explicitly allow them again in the future.",
            )),
        };
    let to = args::dtype(runtime, bound.required("to"))?;
    let casting = match bound.value("casting") {
        Some(casting) => Casting::parse(&runtime.string_value(&casting)?.unwrap_or_default())?,
        None => Casting::Safe,
    };
    Ok(Value::Bool(dtype::can_cast(from, to, casting)))
}

/// The scalar type an `issubdtype` argument names: a NumPy type, or the scalar type of a
/// dtype specification. `None` stands for the `str` and `object` dtypes, which box to builtin
/// values and sit directly below `generic`.
fn issubdtype_kind<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    value: PyValue<'s>,
) -> PyResult<'s, Option<&'static ValueKindDef>> {
    if let Some(PyTypeObject::Kind(kind)) = runtime.type_object(&value) {
        return Ok(Some(kind));
    }
    let dtype = args::dtype(runtime, value)?;
    Ok(super::scalar::scalar_kind(dtype.kind()))
}

fn is_subkind(kind: &'static ValueKindDef, ancestor: &'static ValueKindDef) -> bool {
    std::ptr::eq(kind, ancestor)
        || kind.bases.iter().any(|base| match base {
            KindBase::Kind(base) => is_subkind(base, ancestor),
            KindBase::Float | KindBase::Complex => false,
        })
}

/// `np.issubdtype(arg1, arg2)`.
pub(in crate::python) fn issubdtype<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("issubdtype", 2, 2)?;
    let child = issubdtype_kind(runtime, args.positional()[0])?;
    let parent = issubdtype_kind(runtime, args.positional()[1])?;
    let generic = &super::scalar::GENERIC;
    Ok(Value::Bool(match (child, parent) {
        (Some(child), Some(parent)) => is_subkind(child, parent),
        (None, Some(parent)) => std::ptr::eq(parent, generic),
        (Some(_), None) => false,
        (None, None) => {
            let left = args::dtype(runtime, args.positional()[0])?;
            let right = args::dtype(runtime, args.positional()[1])?;
            left.kind() == right.kind()
        }
    }))
}

/// `np.isscalar(element)`: NumPy scalars and Python numbers, strings, and bytes.
pub(in crate::python) fn isscalar<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("isscalar", 1, 1)?;
    let value = args.positional()[0];
    if super::scalar::unbox(runtime, &value).is_some() {
        return Ok(Value::Bool(true));
    }
    Ok(Value::Bool(matches!(
        runtime.kind(&value)?,
        PyKind::Bool
            | PyKind::Int
            | PyKind::Float
            | PyKind::Complex
            | PyKind::String
            | PyKind::Bytes
    )))
}
