//! NumPy scalar types as registered value kinds.
//!
//! A NumPy scalar is one element plus its dtype. Kinds whose element fits in eight bytes carry
//! the little-endian element bytes inline; `complex128` uses a 16-byte heap payload. Boxing
//! happens only at the edge: indexing, iteration, and 0-d results.
//!
//! The abstract hierarchy (`generic` → `number` → `integer` → `signedinteger`, …) is registered
//! first so `isinstance(np.int8(1), np.integer)` works. `float64` and `complex128` also derive
//! from Python's `float` and `complex`, as in NumPy. `str` and `object` elements box to plain
//! Python values rather than `np.str_` and `np.object_` instances.

use super::super::super::native::{
    CallArgs, GetterDef, KindBase, KindNumber, MethodDef, PyError, PyKind, PyResult, PyRuntime,
    PyValue, PyValueCast, ValueKindDef, ValueKindSlots,
};
use super::super::super::Value;
use super::dtype::{Category, DType, Kind};
use super::element::{Number, C128};
use super::format::{complex_repr, float_repr, number_str, Precision};

pub(in crate::python) const NO_SLOTS: ValueKindSlots = ValueKindSlots {
    repr: None,
    str_: None,
    bool_: None,
    get_item: None,
    positive: None,
    negative: None,
    invert: None,
    absolute: None,
    add: None,
    reflected_add: None,
    subtract: None,
    reflected_subtract: None,
    multiply: None,
    reflected_multiply: None,
    divide: None,
    reflected_divide: None,
    floor_divide: None,
    reflected_floor_divide: None,
    remainder: None,
    reflected_remainder: None,
    divmod: None,
    reflected_divmod: None,
    power: None,
    reflected_power: None,
    left_shift: None,
    reflected_left_shift: None,
    right_shift: None,
    reflected_right_shift: None,
    bitwise_and: None,
    reflected_bitwise_and: None,
    bitwise_xor: None,
    reflected_bitwise_xor: None,
    bitwise_or: None,
    reflected_bitwise_or: None,
    equal: None,
    not_equal: None,
    less_than: None,
    less_equal: None,
    greater_than: None,
    greater_equal: None,
};

const fn abstract_kind(name: &'static str, bases: &'static [KindBase]) -> ValueKindDef {
    ValueKindDef {
        name,
        construct: construct_abstract,
        slots: NO_SLOTS,
        methods: &[],
        getters: &[],
        bases,
        call: None,
        numeric: None,
    }
}

fn construct_abstract(_runtime: &mut dyn PyRuntime, _args: CallArgs) -> PyResult {
    Err(PyError::type_error(
        "cannot create instances of an abstract NumPy scalar type",
    ))
}

pub(in crate::python) static GENERIC: ValueKindDef = ValueKindDef {
    name: "numpy.generic",
    construct: construct_abstract,
    slots: NO_SLOTS,
    methods: GENERIC_METHODS,
    getters: GENERIC_GETTERS,
    bases: &[],
    call: None,
    numeric: None,
};
pub(in crate::python) static NUMBER: ValueKindDef =
    abstract_kind("numpy.number", &[KindBase::Kind(&GENERIC)]);
pub(in crate::python) static INTEGER: ValueKindDef = ValueKindDef {
    methods: &[method("__index__", method_index)],
    ..abstract_kind("numpy.integer", &[KindBase::Kind(&NUMBER)])
};
pub(in crate::python) static SIGNED_INTEGER: ValueKindDef =
    abstract_kind("numpy.signedinteger", &[KindBase::Kind(&INTEGER)]);
pub(in crate::python) static UNSIGNED_INTEGER: ValueKindDef =
    abstract_kind("numpy.unsignedinteger", &[KindBase::Kind(&INTEGER)]);
pub(in crate::python) static INEXACT: ValueKindDef =
    abstract_kind("numpy.inexact", &[KindBase::Kind(&NUMBER)]);
pub(in crate::python) static FLOATING: ValueKindDef =
    abstract_kind("numpy.floating", &[KindBase::Kind(&INEXACT)]);
pub(in crate::python) static COMPLEX_FLOATING: ValueKindDef =
    abstract_kind("numpy.complexfloating", &[KindBase::Kind(&INEXACT)]);

/// Abstract kinds in registration order.
pub(in crate::python) static ABSTRACT: [&ValueKindDef; 8] = [
    &GENERIC,
    &NUMBER,
    &INTEGER,
    &SIGNED_INTEGER,
    &UNSIGNED_INTEGER,
    &INEXACT,
    &FLOATING,
    &COMPLEX_FLOATING,
];

macro_rules! scalar_kind {
    ($name:literal, $construct:ident, $dtype:expr, [$($base:expr),*]) => {
        ValueKindDef {
            name: $name,
            construct: $construct,
            slots: SCALAR_SLOTS,
            methods: &[],
            getters: &[],
            bases: &[$($base),*],
            call: None,
            numeric: Some(scalar_numeric),
        }
    };
}

macro_rules! scalar_constructors {
    ($($function:ident => $dtype:expr;)*) => {
        $(
            fn $function(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
                construct(runtime, args, $dtype)
            }
        )*
    };
}

scalar_constructors! {
    construct_bool => DType::BOOL;
    construct_int8 => DType::INT8;
    construct_int16 => DType::INT16;
    construct_int32 => DType::INT32;
    construct_int64 => DType::INT64;
    construct_uint8 => DType::UINT8;
    construct_uint16 => DType::UINT16;
    construct_uint32 => DType::UINT32;
    construct_uint64 => DType::UINT64;
    construct_float16 => DType::FLOAT16;
    construct_float32 => DType::FLOAT32;
    construct_float64 => DType::FLOAT64;
    construct_complex64 => DType::COMPLEX64;
    construct_complex128 => DType::COMPLEX128;
}

/// Concrete scalar kinds indexed by numeric [`Kind`] discriminant.
pub(in crate::python) static SCALARS: [ValueKindDef; 14] = [
    scalar_kind!(
        "numpy.bool",
        construct_bool,
        DType::BOOL,
        [KindBase::Kind(&GENERIC)]
    ),
    scalar_kind!(
        "numpy.int8",
        construct_int8,
        DType::INT8,
        [KindBase::Kind(&SIGNED_INTEGER)]
    ),
    scalar_kind!(
        "numpy.int16",
        construct_int16,
        DType::INT16,
        [KindBase::Kind(&SIGNED_INTEGER)]
    ),
    scalar_kind!(
        "numpy.int32",
        construct_int32,
        DType::INT32,
        [KindBase::Kind(&SIGNED_INTEGER)]
    ),
    scalar_kind!(
        "numpy.int64",
        construct_int64,
        DType::INT64,
        [KindBase::Kind(&SIGNED_INTEGER)]
    ),
    scalar_kind!(
        "numpy.uint8",
        construct_uint8,
        DType::UINT8,
        [KindBase::Kind(&UNSIGNED_INTEGER)]
    ),
    scalar_kind!(
        "numpy.uint16",
        construct_uint16,
        DType::UINT16,
        [KindBase::Kind(&UNSIGNED_INTEGER)]
    ),
    scalar_kind!(
        "numpy.uint32",
        construct_uint32,
        DType::UINT32,
        [KindBase::Kind(&UNSIGNED_INTEGER)]
    ),
    scalar_kind!(
        "numpy.uint64",
        construct_uint64,
        DType::UINT64,
        [KindBase::Kind(&UNSIGNED_INTEGER)]
    ),
    scalar_kind!(
        "numpy.float16",
        construct_float16,
        DType::FLOAT16,
        [KindBase::Kind(&FLOATING)]
    ),
    scalar_kind!(
        "numpy.float32",
        construct_float32,
        DType::FLOAT32,
        [KindBase::Kind(&FLOATING)]
    ),
    scalar_kind!(
        "numpy.float64",
        construct_float64,
        DType::FLOAT64,
        [KindBase::Kind(&FLOATING), KindBase::Float]
    ),
    scalar_kind!(
        "numpy.complex64",
        construct_complex64,
        DType::COMPLEX64,
        [KindBase::Kind(&COMPLEX_FLOATING)]
    ),
    scalar_kind!(
        "numpy.complex128",
        construct_complex128,
        DType::COMPLEX128,
        [KindBase::Kind(&COMPLEX_FLOATING), KindBase::Complex]
    ),
];

/// Protocol slots shared by every concrete scalar kind. Arithmetic and comparisons run through
/// the same ufunc loops as arrays, so scalars keep dtype semantics such as int8 wrapping.
const SCALAR_SLOTS: ValueKindSlots = ValueKindSlots {
    repr: Some(slot_repr),
    str_: Some(slot_str),
    bool_: Some(slot_bool),
    get_item: Some(slot_get_item),
    positive: Some(super::ufunc::slot_positive),
    negative: Some(super::ufunc::slot_negative),
    invert: Some(super::ufunc::slot_invert),
    absolute: Some(super::ufunc::slot_absolute),
    add: Some(super::ufunc::slot_add),
    reflected_add: Some(super::ufunc::slot_reflected_add),
    subtract: Some(super::ufunc::slot_subtract),
    reflected_subtract: Some(super::ufunc::slot_reflected_subtract),
    multiply: Some(super::ufunc::slot_multiply),
    reflected_multiply: Some(super::ufunc::slot_reflected_multiply),
    divide: Some(super::ufunc::slot_divide),
    reflected_divide: Some(super::ufunc::slot_reflected_divide),
    floor_divide: Some(super::ufunc::slot_floor_divide),
    reflected_floor_divide: Some(super::ufunc::slot_reflected_floor_divide),
    remainder: Some(super::ufunc::slot_remainder),
    reflected_remainder: Some(super::ufunc::slot_reflected_remainder),
    divmod: Some(super::ufunc::slot_divmod),
    reflected_divmod: Some(super::ufunc::slot_reflected_divmod),
    power: Some(super::ufunc::slot_power),
    reflected_power: Some(super::ufunc::slot_reflected_power),
    left_shift: Some(super::ufunc::slot_left_shift),
    reflected_left_shift: Some(super::ufunc::slot_reflected_left_shift),
    right_shift: Some(super::ufunc::slot_right_shift),
    reflected_right_shift: Some(super::ufunc::slot_reflected_right_shift),
    bitwise_and: Some(super::ufunc::slot_bitwise_and),
    reflected_bitwise_and: Some(super::ufunc::slot_reflected_bitwise_and),
    bitwise_xor: Some(super::ufunc::slot_bitwise_xor),
    reflected_bitwise_xor: Some(super::ufunc::slot_reflected_bitwise_xor),
    bitwise_or: Some(super::ufunc::slot_bitwise_or),
    reflected_bitwise_or: Some(super::ufunc::slot_reflected_bitwise_or),
    equal: Some(super::ufunc::slot_equal),
    not_equal: Some(super::ufunc::slot_not_equal),
    less_than: Some(super::ufunc::slot_less_than),
    less_equal: Some(super::ufunc::slot_less_equal),
    greater_than: Some(super::ufunc::slot_greater_than),
    greater_equal: Some(super::ufunc::slot_greater_equal),
};

static GENERIC_METHODS: &[MethodDef] = &[
    method("item", method_item),
    method("tolist", method_item),
    method("conjugate", method_conjugate),
    method("conj", method_conjugate),
    method("__int__", method_int),
    method("__float__", method_float),
    method("__complex__", method_complex),
    method("__round__", method_round),
];

static GENERIC_GETTERS: &[GetterDef] = &[
    getter("dtype", get_dtype),
    getter("real", get_real),
    getter("imag", get_imag),
    getter("itemsize", get_itemsize),
    getter("nbytes", get_itemsize),
    getter("ndim", get_ndim),
    getter("shape", get_shape),
    getter("size", get_size),
];

const fn method(
    name: &'static str,
    call: fn(&mut dyn PyRuntime, PyValue, CallArgs) -> PyResult,
) -> MethodDef {
    MethodDef {
        type_name: "numpy.generic",
        name,
        call,
    }
}

const fn getter(name: &'static str, get: fn(&mut dyn PyRuntime, PyValue) -> PyResult) -> GetterDef {
    GetterDef {
        owner: "numpy.generic",
        name,
        get,
    }
}

/// The concrete scalar kind for a numeric dtype.
pub(in crate::python) fn scalar_kind(kind: Kind) -> Option<&'static ValueKindDef> {
    SCALARS.get(kind as usize)
}

/// The numeric dtype of a concrete scalar kind, found by position in [`SCALARS`].
pub(in crate::python) fn kind_dtype(kind: &'static ValueKindDef) -> Option<DType> {
    let index = SCALARS
        .iter()
        .position(|candidate| std::ptr::eq(candidate, kind))?;
    Some(DType::of(super::dtype::KINDS[index]))
}

/// Box one element of a numeric dtype as a NumPy scalar.
pub(in crate::python) fn box_bytes(
    runtime: &mut dyn PyRuntime,
    dtype: DType,
    bytes: &[u8],
) -> PyResult<PyValue> {
    let kind = scalar_kind(dtype.kind())
        .ok_or_else(|| PyError::runtime_error("only numeric elements box as NumPy scalars"))?;
    let itemsize = dtype.itemsize();
    let mut padded = [0u8; 16];
    padded[..itemsize].copy_from_slice(&bytes[..itemsize]);
    if itemsize > 8 {
        let low = u64::from_le_bytes(padded[..8].try_into().expect("eight bytes"));
        let high = u64::from_le_bytes(padded[8..].try_into().expect("eight bytes"));
        runtime.new_wide_value_kind(kind, [low, high])
    } else {
        let payload = u64::from_le_bytes(padded[..8].try_into().expect("eight bytes"));
        runtime.new_value_kind(kind, payload)
    }
}

/// Box a number as a NumPy scalar of `dtype`, with unsafe-cast semantics.
pub(in crate::python) fn box_number(
    runtime: &mut dyn PyRuntime,
    dtype: DType,
    value: Number,
) -> PyResult<PyValue> {
    let mut bytes = [0u8; 16];
    super::element::write_number(dtype.kind(), value, &mut bytes);
    box_bytes(runtime, dtype, &bytes)
}

/// The dtype and element bytes of a NumPy scalar.
pub(in crate::python) fn unbox(
    runtime: &dyn PyRuntime,
    value: &PyValue,
) -> Option<(DType, [u8; 16])> {
    let kind = runtime.value_kind_of(value)?;
    let dtype = kind_dtype(kind)?;
    let mut bytes = [0u8; 16];
    if dtype.itemsize() > 8 {
        let [low, high] = runtime.wide_value_kind_payload(value, kind)?;
        bytes[..8].copy_from_slice(&low.to_le_bytes());
        bytes[8..].copy_from_slice(&high.to_le_bytes());
    } else {
        let payload = runtime.value_kind_payload(value, kind)?;
        bytes[..8].copy_from_slice(&payload.to_le_bytes());
    }
    Some((dtype, bytes))
}

/// The dtype and numeric value of a NumPy scalar.
pub(in crate::python) fn unbox_number(
    runtime: &dyn PyRuntime,
    value: &PyValue,
) -> Option<(DType, Number)> {
    let (dtype, bytes) = unbox(runtime, value)?;
    Some((dtype, super::element::read_number(dtype.kind(), &bytes)))
}

/// The Python number a scalar stands for; see [`ValueKindDef::numeric`].
fn scalar_numeric(kind: &'static ValueKindDef, payload: [u64; 2]) -> Option<KindNumber> {
    let dtype = kind_dtype(kind)?;
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&payload[0].to_le_bytes());
    bytes[8..].copy_from_slice(&payload[1].to_le_bytes());
    Some(match super::element::read_number(dtype.kind(), &bytes) {
        Number::Bool(value) => KindNumber::Bool(value),
        Number::Int(value) => KindNumber::Int(value),
        Number::UInt(value) => KindNumber::UInt(value),
        Number::Float(value) => KindNumber::Float(value),
        Number::Complex(real, imag) => KindNumber::Complex(real, imag),
    })
}

/// `scalar[()]` is the scalar itself and `scalar[...]` a 0-d array holding it; NumPy rejects
/// every other index.
fn slot_get_item(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    index: PyValue,
) -> PyResult<Option<PyValue>> {
    if runtime.is_ellipsis(&index) {
        return super::convert::as_array(runtime, receiver).map(|array| Some(array.value()));
    }
    if runtime.kind(&index)? == PyKind::Tuple {
        let tuple = index.cast(runtime)?;
        if runtime.tuple_items(tuple)?.is_empty() {
            return Ok(Some(receiver));
        }
    }
    Err(PyError::exception(
        "IndexError",
        "invalid index to scalar variable.",
    ))
}

/// Printing precision of a float or complex dtype.
pub(in crate::python) fn precision(dtype: DType) -> Precision {
    match dtype.kind() {
        Kind::Float16 => Precision::Half,
        Kind::Float32 | Kind::Complex64 => Precision::Single,
        _ => Precision::Double,
    }
}

/// `str()` of a NumPy scalar.
pub(in crate::python) fn scalar_str(dtype: DType, value: Number) -> String {
    number_str(value, precision(dtype))
}

/// `repr()` of a NumPy scalar, e.g. `np.float64(1.5)` or `np.True_`.
pub(in crate::python) fn scalar_repr(dtype: DType, value: Number) -> String {
    match value {
        Number::Bool(true) => "np.True_".to_string(),
        Number::Bool(false) => "np.False_".to_string(),
        Number::Complex(real, imag) => format!(
            "np.{}({})",
            dtype.kind().name(),
            complex_repr(real, imag, precision(dtype))
                .trim_start_matches('(')
                .trim_end_matches(')')
        ),
        Number::Float(value) => format!(
            "np.{}({})",
            dtype.kind().name(),
            float_repr(value, precision(dtype))
        ),
        other => format!(
            "np.{}({})",
            dtype.kind().name(),
            number_str(other, precision(dtype))
        ),
    }
}

fn receiver_number(runtime: &dyn PyRuntime, receiver: &PyValue) -> PyResult<(DType, Number)> {
    unbox_number(runtime, receiver).ok_or_else(|| PyError::type_error("expected a NumPy scalar"))
}

fn slot_repr(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Option<PyValue>> {
    let (dtype, number) = receiver_number(runtime, &value)?;
    runtime.new_string(scalar_repr(dtype, number)).map(Some)
}

fn slot_bool(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Option<PyValue>> {
    let (_, number) = receiver_number(runtime, &value)?;
    Ok(Some(Value::Bool(number.is_true())))
}

fn slot_str(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult<Option<PyValue>> {
    let (dtype, number) = receiver_number(runtime, &receiver)?;
    runtime.new_string(scalar_str(dtype, number)).map(Some)
}

/// The real part of a scalar for a conversion to a real Python type, warning as NumPy does when
/// an imaginary part is dropped.
fn real_value(runtime: &mut dyn PyRuntime, receiver: &PyValue) -> PyResult<Number> {
    let (_, number) = receiver_number(runtime, receiver)?;
    Ok(match number {
        Number::Complex(real, _) => {
            super::errstate::warn_complex_discard(runtime)?;
            Number::Float(real)
        }
        number => number,
    })
}

/// `int(scalar)`: exact for integers; floats truncate toward zero.
fn method_int(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("__int__", 0, 0)?;
    match real_value(runtime, &receiver)? {
        Number::Float(value) => python_int_from_float(runtime, value),
        number => number_to_python_int(runtime, number),
    }
}

fn method_index(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("__index__", 0, 0)?;
    let (_, number) = receiver_number(runtime, &receiver)?;
    number_to_python_int(runtime, number)
}

fn method_float(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("__float__", 0, 0)?;
    Ok(Value::Float(real_value(runtime, &receiver)?.as_f64()))
}

fn method_complex(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("__complex__", 0, 0)?;
    let (_, number) = receiver_number(runtime, &receiver)?;
    let (real, imag) = number.as_complex();
    runtime.new_complex(real, imag)
}

/// `round(scalar)` gives a Python `int`, rounding half to even; `round(scalar, n)` keeps the
/// scalar's dtype and follows `np.round`.
fn method_round(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("__round__", 0, 1)?;
    args.reject_unknown_keywords("__round__", &["ndigits"])?;
    let ndigits = match args
        .positional()
        .first()
        .or(args.keyword("__round__", "ndigits")?)
    {
        Some(value) if !value.is_none() => Some(super::args::index_int(runtime, value)?),
        _ => None,
    };
    let (dtype, number) = receiver_number(runtime, &receiver)?;
    if dtype.category() == Category::Complex {
        return Err(PyError::type_error(format!(
            "type numpy.{} doesn't define __round__ method",
            dtype.kind().name()
        )));
    }
    match ndigits {
        Some(decimals) => {
            let rounded = super::math::round_number(dtype, number, decimals);
            box_number(runtime, dtype, rounded)
        }
        None => match number {
            Number::Float(value) => python_int_from_float(runtime, value.round_ties_even()),
            number => number_to_python_int(runtime, number),
        },
    }
}

/// A Python `int` from an integral NumPy number.
fn number_to_python_int(runtime: &mut dyn PyRuntime, number: Number) -> PyResult {
    match number {
        Number::Bool(value) => Ok(Value::Int(i64::from(value))),
        Number::Int(value) => Ok(Value::Int(value)),
        Number::UInt(value) => runtime.new_integer(&value.to_string()),
        Number::Float(value) => python_int_from_float(runtime, value),
        Number::Complex(real, _) => python_int_from_float(runtime, real),
    }
}

/// A Python `int` truncated from a float, with CPython's errors for NaN and infinities.
fn python_int_from_float(runtime: &mut dyn PyRuntime, value: f64) -> PyResult {
    if value.is_nan() {
        return Err(PyError::value_error("cannot convert float NaN to integer"));
    }
    if value.is_infinite() {
        return Err(PyError::overflow_error(
            "cannot convert float infinity to integer",
        ));
    }
    runtime.new_integer(&format!("{:.0}", value.trunc()))
}

/// The Python value `item()` returns for a number of `dtype`.
pub(in crate::python) fn number_to_python(
    runtime: &mut dyn PyRuntime,
    value: Number,
) -> PyResult<PyValue> {
    Ok(match value {
        Number::Bool(value) => Value::Bool(value),
        Number::Int(value) => Value::Int(value),
        Number::UInt(value) => match i64::try_from(value) {
            Ok(value) => Value::Int(value),
            Err(_) => runtime.new_integer(&value.to_string())?,
        },
        Number::Float(value) => Value::Float(value),
        Number::Complex(real, imag) => runtime.new_complex(real, imag)?,
    })
}

fn method_item(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let (_, number) = receiver_number(runtime, &receiver)?;
    if !args.positional().is_empty() {
        let index = args.positional()[0];
        let empty = runtime.kind(&index)? == PyKind::Tuple
            && runtime
                .new_tuple(Vec::new())
                .is_ok_and(|empty| runtime.equals(&index, &empty).unwrap_or(false));
        let zero = runtime
            .int_value(&index)
            .is_some_and(|value| value == 0 || value == -1);
        if !(empty || zero) || args.positional().len() > 1 {
            return Err(PyError::exception(
                "IndexError",
                "index 1 is out of bounds for size 1",
            ));
        }
    }
    number_to_python(runtime, number)
}

fn method_conjugate(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("conjugate", 0, 0)?;
    let (dtype, number) = receiver_number(runtime, &receiver)?;
    match number {
        Number::Complex(real, imag) => box_number(runtime, dtype, Number::Complex(real, -imag)),
        _ => Ok(receiver),
    }
}

fn get_dtype(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    let (dtype, _) = receiver_number(runtime, &receiver)?;
    super::dtype_object::new(runtime, dtype)
}

fn get_real(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    let (dtype, number) = receiver_number(runtime, &receiver)?;
    match number {
        Number::Complex(real, _) => box_number(runtime, dtype.real_part(), Number::Float(real)),
        _ => Ok(receiver),
    }
}

fn get_imag(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    let (dtype, number) = receiver_number(runtime, &receiver)?;
    match number {
        Number::Complex(_, imag) => box_number(runtime, dtype.real_part(), Number::Float(imag)),
        _ => box_number(runtime, dtype, Number::Int(0)),
    }
}

fn get_itemsize(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    let (dtype, _) = receiver_number(runtime, &receiver)?;
    Ok(Value::Int(dtype.itemsize() as i64))
}

fn get_ndim(_runtime: &mut dyn PyRuntime, _receiver: PyValue) -> PyResult {
    Ok(Value::Int(0))
}

fn get_shape(runtime: &mut dyn PyRuntime, _receiver: PyValue) -> PyResult {
    runtime.new_tuple(Vec::new())
}

fn get_size(_runtime: &mut dyn PyRuntime, _receiver: PyValue) -> PyResult {
    Ok(Value::Int(1))
}

/// `np.float32(x)` and friends: convert one Python or NumPy value, or cast an array-like.
fn construct(runtime: &mut dyn PyRuntime, args: CallArgs, dtype: DType) -> PyResult {
    let positional = args.positional();
    if !args.keywords().is_empty() || positional.len() > 1 {
        return Err(PyError::type_error(format!(
            "{}() takes at most 1 argument",
            dtype.kind().name()
        )));
    }
    let Some(value) = positional.first().copied() else {
        return box_number(runtime, dtype, Number::Int(0));
    };
    let array = super::convert::array_from_python(runtime, value, Some(dtype), false)?;
    if array.ndim() > 0 {
        return Ok(array.value());
    }
    super::convert::element_to_scalar(runtime, &array, array.view.offset)
}

/// Whether `dtype`'s values support `__index__`.
pub(in crate::python) fn is_index_dtype(dtype: DType) -> bool {
    matches!(
        dtype.category(),
        Category::Bool | Category::Signed | Category::Unsigned
    )
}

/// A complex128 value read from scalar bytes.
pub(in crate::python) fn complex_value(bytes: &[u8; 16]) -> C128 {
    use super::element::Element;
    C128::read(bytes)
}
