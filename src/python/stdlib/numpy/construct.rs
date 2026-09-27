//! Array constructors and dtype queries: `np.array`, `zeros`, `arange`, `linspace`, `eye`,
//! `diag`, `meshgrid`, `fromiter`, `result_type`, `can_cast`, `issubdtype`, and friends.
//!
//! Constructors reserve memory for the whole result before filling it and charge CPU per
//! element. Values are computed in `f64` or `i64` and written with the target dtype's cast
//! rules, which matches NumPy for every dtype these constructors produce.

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

/// `np.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)`.
pub(in crate::python) fn array(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("array", &["object", "dtype"], 1)
        .keyword_only(&["copy", "order", "subok", "ndmin", "like"]);
    let bound = SIGNATURE.bind(&args)?;
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?;
    let object = bound.required("object");
    // copy=True always copies; copy=None copies only when needed; copy=False never copies.
    let copy = match bound.get("copy") {
        None => Some(true),
        Some(value) if value.is_none() => None,
        Some(value) => Some(runtime.truth(&value)?),
    };
    let ndmin = args::optional_int(runtime, bound.value("ndmin"))?.unwrap_or(0);
    let result = if runtime.native_kind(&object)? == Some(PyNativeKind::Array) {
        let source = Array::from_value(runtime, object)?;
        let target = dtype.unwrap_or(source.dtype);
        let converted = convert::cast_array(runtime, &source, target, copy == Some(true))?;
        if copy == Some(false) && converted.handle != source.handle {
            return Err(PyError::value_error(
                "Unable to avoid copy while creating an array as requested.",
            ));
        }
        converted
    } else {
        if copy == Some(false) {
            return Err(PyError::value_error(
                "Unable to avoid copy while creating an array as requested.",
            ));
        }
        convert::array_from_python(runtime, object, dtype, false)?
    };
    Ok(with_ndmin(runtime, result, ndmin)?.value())
}

/// Prepend length-one axes until the array has `ndmin` dimensions.
fn with_ndmin(runtime: &mut dyn PyRuntime, array: Array, ndmin: i64) -> PyResult<Array> {
    let missing = usize::try_from(ndmin)
        .unwrap_or(0)
        .saturating_sub(array.ndim());
    if missing == 0 {
        return Ok(array);
    }
    let mut shape = vec![1; missing];
    shape.extend_from_slice(array.shape());
    let mut strides = vec![0; missing];
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

/// `np.asarray(a, dtype=None, order=None, *, copy=None)`.
pub(in crate::python) fn asarray(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("asarray", &["a", "dtype", "order"], 1)
        .keyword_only(&["copy", "like", "device"]);
    let bound = SIGNATURE.bind(&args)?;
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?;
    let copy = args::flag(runtime, bound.value("copy"), false)?;
    Ok(convert::array_from_python(runtime, bound.required("a"), dtype, copy)?.value())
}

/// `np.ascontiguousarray(a, dtype=None)`: a C-contiguous array, copying only when needed.
pub(in crate::python) fn ascontiguousarray(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("ascontiguousarray", &["a", "dtype"], 1).keyword_only(&["like"]);
    let bound = SIGNATURE.bind(&args)?;
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?;
    let array = convert::array_from_python(runtime, bound.required("a"), dtype, false)?;
    let array = if array.is_c_contiguous() {
        array
    } else {
        array::copy_array(runtime, &array)?
    };
    Ok(with_ndmin(runtime, array, 1)?.value())
}

/// `np.copy(a)`.
pub(in crate::python) fn copy(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("copy", &["a", "order", "subok"], 1);
    let bound = SIGNATURE.bind(&args)?;
    Ok(convert::array_from_python(runtime, bound.required("a"), None, true)?.value())
}

/// An array of `shape` whose every element is `fill`, already converted to `dtype` storage.
fn filled(
    runtime: &mut dyn PyRuntime,
    shape: Vec<usize>,
    dtype: DType,
    fill: Option<PyValue>,
) -> PyResult<Array> {
    let count = array::element_count(&shape)?;
    let Some(fill) = fill else {
        let buffer = array::zeroed_buffer(runtime, dtype, count)?;
        return array::new_array(runtime, buffer, dtype, shape);
    };
    let element = if dtype.kind() == Kind::Object {
        let buffer = super::super::super::native::PyArrayBuffer::Values(vec![fill]);
        array::new_array(runtime, buffer, dtype, Vec::new())?
    } else {
        let source = convert::array_from_python(runtime, fill, None, false)?;
        convert::cast_array(runtime, &source, dtype, false)?
    };
    let buffer = array::broadcast_buffer(runtime, &element, dtype, &shape)?;
    array::new_array(runtime, buffer, dtype, shape)
}

/// `np.zeros`, `np.ones`, and `np.empty` share one signature; `empty` is zero-filled.
fn shaped(
    runtime: &mut dyn PyRuntime,
    args: &CallArgs,
    name: &'static str,
    fill: Option<PyValue>,
) -> PyResult {
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
    Ok(filled(runtime, shape, dtype, fill)?.value())
}

pub(in crate::python) fn zeros(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    shaped(runtime, &args, "zeros", None)
}

pub(in crate::python) fn empty(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    shaped(runtime, &args, "empty", None)
}

pub(in crate::python) fn ones(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    shaped(runtime, &args, "ones", Some(Value::Int(1)))
}

/// `np.full(shape, fill_value, dtype=None)`: the dtype defaults to the fill value's.
pub(in crate::python) fn full(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
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
    Ok(filled(runtime, shape, dtype, Some(fill))?.value())
}

/// The `*_like` constructors: dtype and shape come from the prototype unless overridden.
fn like(runtime: &mut dyn PyRuntime, bound: &args::Bound, fill: Option<PyValue>) -> PyResult {
    let prototype = convert::as_array(runtime, bound.required("a"))?;
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?.unwrap_or(prototype.dtype);
    let shape = match bound.value("shape") {
        Some(shape) => args::shape(runtime, shape)?,
        None => prototype.shape().to_vec(),
    };
    Ok(filled(runtime, shape, dtype, fill)?.value())
}

pub(in crate::python) fn zeros_like(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("zeros_like", &["a", "dtype", "order", "subok", "shape"], 1)
            .keyword_only(&["device"]);
    let bound = SIGNATURE.bind(&args)?;
    like(runtime, &bound, None)
}

pub(in crate::python) fn empty_like(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "empty_like",
        &["prototype", "dtype", "order", "subok", "shape"],
        1,
    )
    .keyword_only(&["device"]);
    let bound = SIGNATURE.bind(&args)?;
    let prototype = convert::as_array(runtime, bound.required("prototype"))?;
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?.unwrap_or(prototype.dtype);
    let shape = match bound.value("shape") {
        Some(shape) => args::shape(runtime, shape)?,
        None => prototype.shape().to_vec(),
    };
    Ok(filled(runtime, shape, dtype, None)?.value())
}

pub(in crate::python) fn ones_like(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("ones_like", &["a", "dtype", "order", "subok", "shape"], 1)
            .keyword_only(&["device"]);
    let bound = SIGNATURE.bind(&args)?;
    like(runtime, &bound, Some(Value::Int(1)))
}

pub(in crate::python) fn full_like(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "full_like",
        &["a", "fill_value", "dtype", "order", "subok", "shape"],
        2,
    )
    .keyword_only(&["device"]);
    let bound = SIGNATURE.bind(&args)?;
    let fill = bound.required("fill_value");
    like(runtime, &bound, Some(fill))
}

/// A float or int argument of a range constructor.
fn range_number(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<(Number, bool)> {
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
pub(in crate::python) fn arange(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
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
    let integral = inferred == DType::INT64 && dtype.is_integer();
    let values = (0..length).map(|index| {
        if integral {
            Number::Int(start as i64 + index as i64 * step as i64)
        } else {
            Number::Float(start + index as f64 * step)
        }
    });
    Ok(numbers_array(runtime, dtype, vec![length], values)?.value())
}

/// A new array from numbers, cast into `dtype`.
pub(in crate::python) fn numbers_array(
    runtime: &mut dyn PyRuntime,
    dtype: DType,
    shape: Vec<usize>,
    values: impl Iterator<Item = Number>,
) -> PyResult<Array> {
    if !dtype.is_numeric() {
        return Err(PyError::unsupported(format!(
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

/// The float values of `linspace`, computed as NumPy does: `start + i * step`, with the last
/// sample pinned to `stop` when it is included.
fn linspace_values(start: f64, stop: f64, num: usize, endpoint: bool) -> (Vec<f64>, f64) {
    let divisions = if endpoint { num.saturating_sub(1) } else { num };
    let delta = stop - start;
    let step = if divisions > 0 {
        delta / divisions as f64
    } else {
        f64::NAN
    };
    let mut values = (0..num)
        .map(|index| {
            if divisions == 0 {
                start
            } else if step == 0.0 {
                start + index as f64 / divisions as f64 * delta
            } else {
                start + index as f64 * step
            }
        })
        .collect::<Vec<_>>();
    if endpoint && num > 1 {
        values[num - 1] = stop;
    }
    (values, step)
}

fn linspace_count(runtime: &mut dyn PyRuntime, value: Option<PyValue>) -> PyResult<usize> {
    let num = args::optional_int(runtime, value)?.unwrap_or(50);
    usize::try_from(num).map_err(|_| {
        PyError::value_error(format!("Number of samples, {num}, must be non-negative."))
    })
}

/// Store floats into `dtype`; integer targets round toward negative infinity, as NumPy 2 does.
fn float_array(runtime: &mut dyn PyRuntime, dtype: DType, values: &[f64]) -> PyResult<Array> {
    let floor = dtype.is_integer();
    numbers_array(
        runtime,
        dtype,
        vec![values.len()],
        values
            .iter()
            .map(|value| Number::Float(if floor { value.floor() } else { *value })),
    )
}

/// `np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)`.
pub(in crate::python) fn linspace(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "linspace",
        &[
            "start", "stop", "num", "endpoint", "retstep", "dtype", "axis",
        ],
        2,
    )
    .keyword_only(&["device"]);
    let bound = SIGNATURE.bind(&args)?;
    if bound
        .value("axis")
        .is_some_and(|axis| axis != Value::Int(0))
    {
        return Err(PyError::unsupported(
            "linspace() with axis= is not supported",
        ));
    }
    let start = args::float_arg(runtime, &bound.required("start"))?;
    let stop = args::float_arg(runtime, &bound.required("stop"))?;
    let num = linspace_count(runtime, bound.value("num"))?;
    let endpoint = args::flag(runtime, bound.get("endpoint"), true)?;
    let retstep = args::flag(runtime, bound.get("retstep"), false)?;
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?.unwrap_or(DType::FLOAT64);
    runtime.charge_cpu(num as u64 + 1)?;
    let (values, step) = linspace_values(start, stop, num, endpoint);
    let result = float_array(runtime, dtype, &values)?.value();
    if !retstep {
        return Ok(result);
    }
    let step = super::scalar::box_number(runtime, DType::FLOAT64, Number::Float(step))?;
    runtime.new_tuple(vec![result, step])
}

/// `np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)`.
pub(in crate::python) fn logspace(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "logspace",
        &["start", "stop", "num", "endpoint", "base", "dtype", "axis"],
        2,
    );
    let bound = SIGNATURE.bind(&args)?;
    let start = args::float_arg(runtime, &bound.required("start"))?;
    let stop = args::float_arg(runtime, &bound.required("stop"))?;
    let num = linspace_count(runtime, bound.value("num"))?;
    let endpoint = args::flag(runtime, bound.get("endpoint"), true)?;
    let base = match bound.value("base") {
        Some(base) => args::float_arg(runtime, &base)?,
        None => 10.0,
    };
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?.unwrap_or(DType::FLOAT64);
    runtime.charge_cpu(2 * num as u64 + 1)?;
    let (exponents, _) = linspace_values(start, stop, num, endpoint);
    let values = exponents
        .iter()
        .map(|exponent| base.powf(*exponent))
        .collect::<Vec<_>>();
    Ok(float_array(runtime, dtype, &values)?.value())
}

/// `np.geomspace(start, stop, num=50, endpoint=True, dtype=None)` for positive real bounds.
pub(in crate::python) fn geomspace(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "geomspace",
        &["start", "stop", "num", "endpoint", "dtype", "axis"],
        2,
    );
    let bound = SIGNATURE.bind(&args)?;
    let start = args::float_arg(runtime, &bound.required("start"))?;
    let stop = args::float_arg(runtime, &bound.required("stop"))?;
    if start == 0.0 || stop == 0.0 {
        return Err(PyError::value_error(
            "Geometric sequence cannot include zero",
        ));
    }
    if (start < 0.0) != (stop < 0.0) {
        return Err(PyError::unsupported(
            "geomspace() between bounds of different signs is not supported",
        ));
    }
    let num = linspace_count(runtime, bound.value("num"))?;
    let endpoint = args::flag(runtime, bound.get("endpoint"), true)?;
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?.unwrap_or(DType::FLOAT64);
    runtime.charge_cpu(2 * num as u64 + 1)?;
    let sign = start.signum();
    let (exponents, _) = linspace_values(start.abs().log10(), stop.abs().log10(), num, endpoint);
    let mut values = exponents
        .iter()
        .map(|exponent| sign * 10f64.powf(*exponent))
        .collect::<Vec<_>>();
    if let Some(first) = values.first_mut() {
        *first = start;
    }
    if endpoint && num > 1 {
        values[num - 1] = stop;
    }
    Ok(float_array(runtime, dtype, &values)?.value())
}

fn dimension(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<usize> {
    let value = args::index_int(runtime, value)?;
    usize::try_from(value).map_err(|_| PyError::value_error("negative dimensions are not allowed"))
}

/// `np.eye(N, M=None, k=0, dtype=float)`.
pub(in crate::python) fn eye(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("eye", &["N", "M", "k", "dtype", "order"], 1)
        .keyword_only(&["like", "device"]);
    let bound = SIGNATURE.bind(&args)?;
    let rows = dimension(runtime, &bound.required("N"))?;
    let columns = match bound.value("M") {
        Some(columns) => dimension(runtime, &columns)?,
        None => rows,
    };
    let k = args::optional_int(runtime, bound.value("k"))?.unwrap_or(0);
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?.unwrap_or(DType::FLOAT64);
    Ok(eye_array(runtime, rows, columns, k, dtype)?.value())
}

fn eye_array(
    runtime: &mut dyn PyRuntime,
    rows: usize,
    columns: usize,
    k: i64,
    dtype: DType,
) -> PyResult<Array> {
    let count = array::element_count(&[rows, columns])?;
    runtime.charge_cpu(count as u64 + 1)?;
    let values = (0..count).map(|flat| {
        let (row, column) = (flat / columns.max(1), flat % columns.max(1));
        Number::Bool(column as i64 - row as i64 == k)
    });
    numbers_array(runtime, dtype, vec![rows, columns], values)
}

/// `np.identity(n, dtype=float)`.
pub(in crate::python) fn identity(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("identity", &["n", "dtype"], 1).keyword_only(&["like"]);
    let bound = SIGNATURE.bind(&args)?;
    let size = dimension(runtime, &bound.required("n"))?;
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?.unwrap_or(DType::FLOAT64);
    Ok(eye_array(runtime, size, size, 0, dtype)?.value())
}

/// `np.diag(v, k=0)`: a matrix with `v` on diagonal `k`, or a read-only view of diagonal `k`
/// of a matrix.
pub(in crate::python) fn diag(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("diag", &["v", "k"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let source = convert::as_array(runtime, bound.required("v"))?;
    let k = args::optional_int(runtime, bound.value("k"))?.unwrap_or(0);
    match source.ndim() {
        1 => {
            let length = source.shape()[0];
            let size = length
                .checked_add(k.unsigned_abs() as usize)
                .ok_or_else(|| PyError::value_error("diagonal offset is too large"))?;
            let result = filled(runtime, vec![size, size], source.dtype, None)?;
            let (row, column) = if k >= 0 {
                (0, k as usize)
            } else {
                ((-k) as usize, 0)
            };
            let offset = result.offset_of(&[row, column]);
            let stride = result.strides()[0] + result.strides()[1];
            let diagonal = array::new_view(
                runtime,
                &result,
                result.dtype,
                vec![length],
                vec![stride],
                offset,
            )?;
            array::assign(runtime, &diagonal, &source)?;
            Ok(result.value())
        }
        2 => Ok(diagonal_view(runtime, &source, k)?.value()),
        _ => Err(PyError::value_error("Input must be 1- or 2-d.")),
    }
}

/// A read-only view of diagonal `k` of a matrix.
pub(in crate::python) fn diagonal_view(
    runtime: &mut dyn PyRuntime,
    matrix: &Array,
    k: i64,
) -> PyResult<Array> {
    let (rows, columns) = (matrix.shape()[0] as i64, matrix.shape()[1] as i64);
    let (row, column) = if k >= 0 { (0, k) } else { (-k, 0) };
    let length = (rows - row).min(columns - column).max(0) as usize;
    let offset = if length > 0 {
        matrix.offset_of(&[row as usize, column as usize])
    } else {
        matrix.view.offset
    };
    let stride = matrix.strides()[0] + matrix.strides()[1];
    let view = array::new_view(
        runtime,
        matrix,
        matrix.dtype,
        vec![length],
        vec![stride],
        offset,
    )?;
    runtime.set_array_writeable(view.handle, false)?;
    Ok(view)
}

/// `np.meshgrid(*xi, indexing='xy', sparse=False, copy=True)`.
pub(in crate::python) fn meshgrid(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let mut indexing = "xy".to_string();
    let mut copy = true;
    for (name, value) in args.keywords() {
        match name.as_str() {
            "indexing" => indexing = runtime.string_value(value)?.unwrap_or_default(),
            "copy" => copy = runtime.truth(value)?,
            "sparse" if !runtime.truth(value)? => {}
            "sparse" => {
                return Err(PyError::unsupported(
                    "meshgrid() with sparse=True is not supported",
                ))
            }
            _ => {
                return Err(PyError::type_error(format!(
                    "meshgrid() got an unexpected keyword argument '{name}'"
                )))
            }
        }
    }
    if indexing != "xy" && indexing != "ij" {
        return Err(PyError::value_error(
            "Valid values for `indexing` are 'xy' and 'ij'.",
        ));
    }
    let inputs = args
        .positional()
        .iter()
        .map(|value| {
            let array = convert::as_array(runtime, *value)?;
            array::ravel(runtime, &array)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut shape = inputs.iter().map(|input| input.size()).collect::<Vec<_>>();
    let swap = indexing == "xy" && inputs.len() >= 2;
    if swap {
        shape.swap(0, 1);
    }
    let mut grids = Vec::with_capacity(inputs.len());
    for (position, input) in inputs.iter().enumerate() {
        let axis = match position {
            0 if swap => 1,
            1 if swap => 0,
            other => other,
        };
        let mut strides = vec![0isize; shape.len()];
        strides[axis] = input.strides()[0];
        let view = array::new_view(
            runtime,
            input,
            input.dtype,
            shape.clone(),
            strides,
            input.view.offset,
        )?;
        let grid = if copy {
            array::copy_array(runtime, &view)?
        } else {
            runtime.set_array_writeable(view.handle, false)?;
            view
        };
        grids.push(grid.value());
    }
    runtime.new_tuple(grids)
}

/// `np.fromiter(iter, dtype, count=-1)`.
pub(in crate::python) fn fromiter(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
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
fn memory_overlap(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    name: &str,
    exact: bool,
) -> PyResult {
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
        .collect::<PyResult<Vec<_>>>()?;
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

pub(in crate::python) fn shares_memory(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let (positional, _) = args.into_parts();
    memory_overlap(
        runtime,
        CallArgs::new(positional, Vec::new()),
        "shares_memory",
        true,
    )
}

pub(in crate::python) fn may_share_memory(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let (positional, _) = args.into_parts();
    memory_overlap(
        runtime,
        CallArgs::new(positional, Vec::new()),
        "may_share_memory",
        false,
    )
}

/// `np.ndim(a)`, `np.shape(a)`, and `np.size(a)` accept any array-like.
pub(in crate::python) fn ndim(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("ndim", 1, 1)?;
    let array = convert::as_array(runtime, args.positional()[0])?;
    Ok(Value::Int(array.ndim() as i64))
}

pub(in crate::python) fn shape(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("shape", 1, 1)?;
    let array = convert::as_array(runtime, args.positional()[0])?;
    super::ndarray::int_tuple(
        runtime,
        array.shape().iter().map(|dimension| *dimension as i64),
    )
}

pub(in crate::python) fn size(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
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
pub(in crate::python) fn take(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
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
pub(in crate::python) fn put(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
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

fn type_operand(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<TypeOperand> {
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
pub(in crate::python) fn result_type(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
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
pub(in crate::python) fn promote_types(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("promote_types", 2, 2)?;
    let left = args::dtype(runtime, args.positional()[0])?;
    let right = args::dtype(runtime, args.positional()[1])?;
    let dtype = dtype::promote(left, right)?;
    super::dtype_object::new(runtime, dtype)
}

/// `np.can_cast(from_, to, casting='safe')`.
pub(in crate::python) fn can_cast(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
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
fn issubdtype_kind(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<&'static ValueKindDef>> {
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
pub(in crate::python) fn issubdtype(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
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
pub(in crate::python) fn isscalar(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
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
