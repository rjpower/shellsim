//! Central numeric views and coercions for the bounded Python runtime.
//!
//! Immediate integers, heap-backed arbitrary-precision integers, and IEEE-754 doubles all cross
//! the native-module boundary through this owned, representation-independent view.

use num_bigint::BigInt;
use num_traits::{FromPrimitive, Signed, ToPrimitive, Zero};

use super::ast::{BinaryOperator, ComparisonOperator};
use super::heap::{Heap, InstancePayload, Object};
use super::native::{
    CallArgs, FromPyValue, GetterDef, KindNumber, MethodDef, NativeTypeDef, PyError, PyKind,
    PyResult, PyRuntime, PyValue, ValueKindDef,
};
use super::ValueTag;

/// Borrowed numeric payload used by VM protocols without exposing physical value tags.
#[derive(Clone, Copy, Debug)]
pub(super) enum NumberRef<'a> {
    Int(i64),
    BigInt(&'a BigInt),
    /// An unsigned integer above `i64::MAX`, such as a large NumPy `uint64`.
    UInt(u64),
    Float(f64),
    /// Real and imaginary components of a builtin `complex`.
    Complex(f64, f64),
}

impl NumberRef<'_> {
    /// The exact integer this number holds, if it is an integer.
    pub(super) fn to_bigint(self) -> Option<BigInt> {
        match self {
            Self::Int(value) => Some(BigInt::from(value)),
            Self::BigInt(value) => Some(value.clone()),
            Self::UInt(value) => Some(BigInt::from(value)),
            Self::Float(_) | Self::Complex(..) => None,
        }
    }
}

/// Python's `==` between two numbers of any representation. An integer equals a float only when
/// the float is integral with the same value, so the comparison stays exact beyond 2**53.
pub(super) fn numbers_equal(left: NumberRef<'_>, right: NumberRef<'_>) -> bool {
    let split = |number| match number {
        NumberRef::Complex(real, imag) => (NumberRef::Float(real), imag),
        number => (number, 0.0),
    };
    let ((left, left_imag), (right, right_imag)) = (split(left), split(right));
    if left_imag != right_imag {
        return false;
    }
    match (left, right) {
        (NumberRef::Float(left), NumberRef::Float(right)) => left == right,
        (NumberRef::Float(float), integer) | (integer, NumberRef::Float(float)) => {
            float.is_finite()
                && float.fract() == 0.0
                && BigInt::from_f64(float) == integer.to_bigint()
        }
        (left, right) => left.to_bigint() == right.to_bigint(),
    }
}

/// The kind and number of a registered value with a numeric view, such as a NumPy scalar.
pub(super) fn registered_number(
    heap: &Heap,
    value: &PyValue,
) -> Option<(&'static ValueKindDef, KindNumber)> {
    let (index, payload) = match value.registered_parts() {
        Some((index, payload)) => (index, [payload, 0]),
        None => match heap.get(value.object_id()?).ok()? {
            Object::WideValue { kind, payload, .. } => (*kind, *payload),
            _ => return None,
        },
    };
    let kind = super::stdlib::value_kind(index)?;
    Some((kind, (kind.numeric?)(kind, payload)?))
}

/// Resolve Python numeric storage into one semantic numeric view.
///
/// Registered values with a numeric view take part as the Python number they stand for, so a
/// NumPy `int64` indexes a list and a NumPy `float64` formats like a float.
pub(super) fn view<'a>(heap: &'a Heap, value: &PyValue) -> Option<NumberRef<'a>> {
    if let Some((_, number)) = registered_number(heap, value) {
        return Some(match number {
            KindNumber::Bool(value) => NumberRef::Int(i64::from(value)),
            KindNumber::Int(value) => NumberRef::Int(value),
            KindNumber::UInt(value) => match i64::try_from(value) {
                Ok(value) => NumberRef::Int(value),
                Err(_) => NumberRef::UInt(value),
            },
            KindNumber::Float(value) => NumberRef::Float(value),
            KindNumber::Complex(real, imag) => NumberRef::Complex(real, imag),
        });
    }
    if let Some(value) = value.float_value() {
        return Some(NumberRef::Float(value));
    }
    if let Some(value) = value.bool_value() {
        return Some(NumberRef::Int(i64::from(value)));
    }
    if matches!(value.tag(), super::ValueTag::Int) {
        return Some(NumberRef::Int(value.immediate_int().expect("tag checked")));
    }
    match value.object_id().and_then(|id| heap.get(id).ok())? {
        Object::BigInt(value) => Some(NumberRef::BigInt(value)),
        Object::Complex { real, imag } => Some(NumberRef::Complex(*real, *imag)),
        Object::Instance {
            payload: InstancePayload::Builtin(value),
            ..
        } => view(heap, value),
        _ => None,
    }
}

/// Numeric-tower attributes shared by one builtin real number type.
///
/// `bool` inherits the integer namespace through its MRO. Results follow CPython:
/// `True.real` is the integer `1`, `(3).imag` is `0`, and `(1.5).imag` is `0.0`.
macro_rules! real_number_type {
    ($name:literal, rational) => {
        NativeTypeDef {
            name: $name,
            methods: &[
                MethodDef {
                    type_name: $name,
                    name: "conjugate",
                    call: real_conjugate,
                },
                MethodDef {
                    type_name: $name,
                    name: "bit_length",
                    call: int_bit_length,
                },
                MethodDef {
                    type_name: $name,
                    name: "bit_count",
                    call: int_bit_count,
                },
                MethodDef {
                    type_name: $name,
                    name: "as_integer_ratio",
                    call: int_as_integer_ratio,
                },
                MethodDef {
                    type_name: $name,
                    name: "is_integer",
                    call: int_is_integer,
                },
                MethodDef {
                    type_name: $name,
                    name: "to_bytes",
                    call: int_to_bytes,
                },
            ],
            getters: &[
                GetterDef {
                    owner: $name,
                    name: "real",
                    get: real_part,
                },
                GetterDef {
                    owner: $name,
                    name: "imag",
                    get: real_imaginary_part,
                },
                GetterDef {
                    owner: $name,
                    name: "numerator",
                    get: real_part,
                },
                GetterDef {
                    owner: $name,
                    name: "denominator",
                    get: integer_denominator,
                },
            ],
        }
    };
    ($name:literal) => {
        NativeTypeDef {
            name: $name,
            methods: &[MethodDef {
                type_name: $name,
                name: "conjugate",
                call: real_conjugate,
            }],
            getters: &[
                GetterDef {
                    owner: $name,
                    name: "real",
                    get: real_part,
                },
                GetterDef {
                    owner: $name,
                    name: "imag",
                    get: real_imaginary_part,
                },
            ],
        }
    };
}

pub(super) static INT_TYPE: NativeTypeDef = real_number_type!("int", rational);

/// `int.__new__(cls, value=0)`, installed on `int` alone since `bool` cannot be subclassed.
pub(super) static INT_CONSTRUCTOR: NativeTypeDef = NativeTypeDef {
    name: "int",
    methods: &[MethodDef {
        type_name: "int",
        name: "__new__",
        call: int_new,
    }],
    getters: &[],
};

fn int_new(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    runtime.new_builtin_instance(super::object_model::BuiltinType::Int, receiver, args)
}

/// `int.from_bytes`, which receives the class so that `bool.from_bytes` returns a `bool`.
pub(super) static INT_CLASS_METHODS: &[MethodDef] = &[MethodDef {
    type_name: "int",
    name: "from_bytes",
    call: int_from_bytes,
}];
pub(super) static FLOAT_TYPE: NativeTypeDef = NativeTypeDef {
    methods: &[
        MethodDef {
            type_name: "float",
            name: "conjugate",
            call: real_conjugate,
        },
        MethodDef {
            type_name: "float",
            name: "is_integer",
            call: float_is_integer,
        },
        MethodDef {
            type_name: "float",
            name: "as_integer_ratio",
            call: float_as_integer_ratio,
        },
        MethodDef {
            type_name: "float",
            name: "hex",
            call: float_hex,
        },
    ],
    ..real_number_type!("float")
};

/// `float.as_integer_ratio()`: the exact fraction in lowest terms with a positive denominator.
fn float_as_integer_ratio(runtime: &mut dyn PyRuntime, value: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("float.as_integer_ratio", 0, 0)?;
    let Some(NumberRef::Float(value)) = runtime.number(&value) else {
        return Err(PyError::type_error(
            "descriptor 'as_integer_ratio' requires a 'float' object",
        ));
    };
    if value.is_nan() {
        return Err(PyError::value_error("cannot convert NaN to integer ratio"));
    }
    if value.is_infinite() {
        return Err(PyError::overflow_error(
            "cannot convert Infinity to integer ratio",
        ));
    }
    // A finite double is `mantissa * 2**exponent` with an integer mantissa below 2**53.
    let bits = value.to_bits();
    let biased = i32::try_from((bits >> 52) & 0x7ff).expect("eleven-bit exponent");
    let fraction = bits & ((1u64 << 52) - 1);
    let (mut mantissa, mut exponent) = if biased == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1u64 << 52), biased - 1075)
    };
    if mantissa == 0 {
        exponent = 0;
    }
    while mantissa != 0 && mantissa % 2 == 0 && exponent < 0 {
        mantissa /= 2;
        exponent += 1;
    }
    let mut numerator = BigInt::from(mantissa);
    let mut denominator = BigInt::from(1u8);
    if exponent >= 0 {
        numerator <<= usize::try_from(exponent).expect("non-negative exponent");
    } else {
        denominator <<= usize::try_from(-exponent).expect("positive exponent");
    }
    if value.is_sign_negative() {
        numerator = -numerator;
    }
    let numerator = runtime.new_integer(&numerator.to_string())?;
    let denominator = runtime.new_integer(&denominator.to_string())?;
    runtime.new_tuple(vec![numerator, denominator])
}

/// `float.hex()`: the exact value in C99 hexadecimal notation, with all 13 fraction digits.
///
/// ```text
/// (0.1).hex() == '0x1.999999999999ap-4'
/// (5e-324).hex() == '0x0.0000000000001p-1022'
/// ```
fn float_hex(runtime: &mut dyn PyRuntime, value: PyValue, args: CallArgs) -> PyResult {
    args.reject_keywords("float.hex")?;
    if !args.positional().is_empty() {
        return Err(PyError::type_error(format!(
            "float.hex() takes no arguments ({} given)",
            args.positional().len()
        )));
    }
    let Some(NumberRef::Float(value)) = runtime.number(&value) else {
        return Err(PyError::type_error(
            "descriptor 'hex' requires a 'float' object",
        ));
    };
    runtime.new_string(float_hex_text(value))
}

fn float_hex_text(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if value.is_infinite() {
        return format!("{sign}inf");
    }
    if value == 0.0 {
        return format!("{sign}0x0.0p+0");
    }
    let bits = value.to_bits();
    let biased = (bits >> 52) & 0x7ff;
    let fraction = bits & ((1u64 << 52) - 1);
    // Subnormals keep the minimum exponent with a leading 0 digit.
    let (leading, exponent) = if biased == 0 {
        (0, -1022)
    } else {
        (1, biased as i64 - 1023)
    };
    format!("{sign}0x{leading}.{fraction:013x}p{exponent:+}")
}

/// `float.is_integer()`: whether a finite float has no fractional part.
fn float_is_integer(runtime: &mut dyn PyRuntime, value: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("float.is_integer", 0, 0)?;
    let Some(NumberRef::Float(value)) = runtime.number(&value) else {
        return Err(PyError::type_error(
            "descriptor 'is_integer' requires a 'float' object",
        ));
    };
    Ok(PyValue::Bool(value.is_finite() && value.fract() == 0.0))
}

/// Return a real number as itself, normalizing `bool` and `int` subclasses to the equal `int`.
fn real_part(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(match runtime.number(&value) {
        Some(NumberRef::Int(value)) => PyValue::Int(value),
        _ => value,
    })
}

/// Return the zero imaginary component with the receiver's int-or-float result type.
fn real_imaginary_part(_runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(if value.float_value().is_some() {
        PyValue::Float(0.0)
    } else {
        PyValue::Int(0)
    })
}

fn integer_denominator(_runtime: &mut dyn PyRuntime, _value: PyValue) -> PyResult {
    Ok(PyValue::Int(1))
}

fn real_conjugate(runtime: &mut dyn PyRuntime, value: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("conjugate", 0, 0)?;
    args.reject_keywords("conjugate")?;
    real_part(runtime, value)
}

/// The receiver of an `int` method as an exact integer; `bool` receivers are 0 or 1.
fn integer_receiver(runtime: &dyn PyRuntime, value: &PyValue, method: &str) -> PyResult<BigInt> {
    runtime
        .number(value)
        .and_then(NumberRef::to_bigint)
        .ok_or_else(|| {
            PyError::type_error(format!("descriptor '{method}' requires a 'int' object"))
        })
}

fn expect_no_arguments(method: &str, args: &CallArgs) -> PyResult<()> {
    if !args.keywords().is_empty() {
        return Err(PyError::type_error(format!(
            "{method}() takes no keyword arguments"
        )));
    }
    match args.positional().len() {
        0 => Ok(()),
        given => Err(PyError::type_error(format!(
            "{method}() takes no arguments ({given} given)"
        ))),
    }
}

/// `int.bit_length()`: the number of bits in the absolute value, so `(-255).bit_length() == 8`.
fn int_bit_length(runtime: &mut dyn PyRuntime, value: PyValue, args: CallArgs) -> PyResult {
    expect_no_arguments("int.bit_length", &args)?;
    let value = integer_receiver(runtime, &value, "bit_length")?;
    let bits = i64::try_from(value.bits())
        .map_err(|_| PyError::overflow_error("int too large to count bits"))?;
    Ok(PyValue::Int(bits))
}

/// `int.bit_count()`: the number of one bits in the absolute value.
fn int_bit_count(runtime: &mut dyn PyRuntime, value: PyValue, args: CallArgs) -> PyResult {
    expect_no_arguments("int.bit_count", &args)?;
    let value = integer_receiver(runtime, &value, "bit_count")?;
    let ones = i64::try_from(value.magnitude().count_ones())
        .map_err(|_| PyError::overflow_error("int too large to count bits"))?;
    Ok(PyValue::Int(ones))
}

/// `int.as_integer_ratio()`: the pair `(int(self), 1)`.
fn int_as_integer_ratio(runtime: &mut dyn PyRuntime, value: PyValue, args: CallArgs) -> PyResult {
    expect_no_arguments("int.as_integer_ratio", &args)?;
    integer_receiver(runtime, &value, "as_integer_ratio")?;
    let numerator = real_part(runtime, value)?;
    runtime.new_tuple(vec![numerator, PyValue::Int(1)])
}

/// `int.is_integer()`: always true, for duck-typing compatibility with `float.is_integer`.
fn int_is_integer(runtime: &mut dyn PyRuntime, value: PyValue, args: CallArgs) -> PyResult {
    expect_no_arguments("int.is_integer", &args)?;
    integer_receiver(runtime, &value, "is_integer")?;
    Ok(PyValue::Bool(true))
}

/// Bind the `(first, byteorder='big', *, signed=False)` parameters shared by `int.to_bytes` and
/// `int.from_bytes`, positionally or by name. The result holds the value bound to each of the
/// three parameters, in order.
fn bind_byte_conversion(
    function: &str,
    first: &str,
    args: &CallArgs,
) -> PyResult<[Option<PyValue>; 3]> {
    let positional = args.positional();
    let given = positional.len().saturating_add(args.keywords().len());
    if given > 3 {
        return Err(PyError::type_error(format!(
            "{function}() takes at most 3 arguments ({given} given)"
        )));
    }
    if positional.len() > 2 {
        return Err(PyError::type_error(format!(
            "{function}() takes at most 2 positional arguments ({} given)",
            positional.len()
        )));
    }
    let mut bound = [
        positional.first().copied(),
        positional.get(1).copied(),
        None,
    ];
    for (name, value) in args.keywords() {
        let slot = match name.as_str() {
            "byteorder" => 1,
            "signed" => 2,
            name if name == first => 0,
            _ => {
                return Err(PyError::type_error(format!(
                    "{function}() got an unexpected keyword argument '{name}'"
                )))
            }
        };
        if bound[slot].is_some() {
            return Err(PyError::type_error(format!(
                "argument for {function}() given by name ('{name}') and position ({})",
                slot + 1
            )));
        }
        bound[slot] = Some(*value);
    }
    Ok(bound)
}

/// Whether a `byteorder` argument selects little-endian order; the default is `'big'`.
fn little_endian(
    runtime: &dyn PyRuntime,
    function: &str,
    byteorder: Option<PyValue>,
) -> PyResult<bool> {
    let Some(byteorder) = byteorder else {
        return Ok(false);
    };
    match runtime.string_value(&byteorder)?.as_deref() {
        Some("little") => Ok(true),
        Some("big") => Ok(false),
        Some(_) => Err(PyError::value_error(
            "byteorder must be either 'little' or 'big'",
        )),
        None => Err(PyError::type_error(format!(
            "{function}() argument 'byteorder' must be str, not {}",
            runtime.type_name(&byteorder)?
        ))),
    }
}

/// `int.to_bytes(length=1, byteorder='big', *, signed=False)`: the integer in exactly `length`
/// bytes, as two's complement when `signed` is true.
///
/// ```text
/// (1024).to_bytes(2) == b'\x04\x00'
/// (-1).to_bytes(2, 'little', signed=True) == b'\xff\xff'
/// (256).to_bytes(1) -> OverflowError: int too big to convert
/// ```
fn int_to_bytes(runtime: &mut dyn PyRuntime, value: PyValue, args: CallArgs) -> PyResult {
    let [length, byteorder, signed] = bind_byte_conversion("to_bytes", "length", &args)?;
    let length = length.map_or(Ok(1), |length| index_argument(runtime, &length))?;
    let little = little_endian(runtime, "to_bytes", byteorder)?;
    let signed = signed.map_or(Ok(false), |signed| runtime.truth(&signed))?;
    let length = usize::try_from(length)
        .map_err(|_| PyError::value_error("length argument must be non-negative"))?;
    let value = integer_receiver(runtime, &value, "to_bytes")?;
    if value.is_negative() && !signed {
        return Err(PyError::overflow_error(
            "can't convert negative int to unsigned",
        ));
    }
    // The shortest little-endian encoding; zero needs no bytes at all.
    let mut bytes = if value.is_zero() {
        Vec::new()
    } else if signed {
        value.to_signed_bytes_le()
    } else {
        value.magnitude().to_bytes_le()
    };
    if bytes.len() > length {
        return Err(PyError::overflow_error("int too big to convert"));
    }
    runtime.reserve_memory(length)?;
    runtime.charge_cpu(u64::try_from(length).unwrap_or(u64::MAX))?;
    bytes.resize(length, if value.is_negative() { 0xff } else { 0 });
    if !little {
        bytes.reverse();
    }
    runtime.new_bytes(bytes)
}

/// `int.from_bytes(bytes, byteorder='big', *, signed=False)`: the integer encoded by a bytes-like
/// object or an iterable of byte values.
///
/// The receiver is the class. As in CPython, a subclass such as `bool` converts the integer by
/// calling the class, so `bool.from_bytes(b'\x01')` is `True`.
fn int_from_bytes(runtime: &mut dyn PyRuntime, class: PyValue, args: CallArgs) -> PyResult {
    let [data, byteorder, signed] = bind_byte_conversion("from_bytes", "bytes", &args)?;
    let Some(data) = data else {
        return Err(PyError::type_error(
            "from_bytes() missing required argument 'bytes' (pos 1)",
        ));
    };
    let little = little_endian(runtime, "from_bytes", byteorder)?;
    let signed = signed.map_or(Ok(false), |signed| runtime.truth(&signed))?;
    let mut bytes = byte_values(runtime, data)?;
    runtime.charge_cpu(u64::try_from(bytes.len()).unwrap_or(u64::MAX))?;
    if !little {
        bytes.reverse();
    }
    let value = if signed {
        BigInt::from_signed_bytes_le(&bytes)
    } else {
        BigInt::from_bytes_le(num_bigint::Sign::Plus, &bytes)
    };
    let value = runtime.new_integer(&value.to_string())?;
    runtime.call_value(class, CallArgs::new(vec![value], Vec::new()))
}

/// The bytes `int.from_bytes` decodes: a `bytes` or `bytearray` as is, or any other iterable of
/// byte values converted as `bytes(value)` does. Integers and strings, which `bytes()` would
/// treat as a length or text, are rejected as in CPython.
fn byte_values(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<u8>> {
    if let Some(bytes) = runtime.bytes_value(&value)? {
        return Ok(bytes);
    }
    if matches!(
        runtime.kind(&value)?,
        PyKind::Bool | PyKind::Int | PyKind::Float | PyKind::String | PyKind::Complex
    ) {
        return Err(PyError::type_error(format!(
            "cannot convert '{}' object to bytes",
            runtime.type_name(&value)?
        )));
    }
    let bytes_type = runtime
        .builtin_type("bytes")
        .expect("bytes is a builtin type");
    let converted = runtime.call_value(bytes_type, CallArgs::new(vec![value], Vec::new()))?;
    Ok(runtime
        .bytes_value(&converted)?
        .expect("bytes() returns bytes"))
}

/// Convert an index argument as CPython's `__index__` protocol does for builtin methods.
pub(super) fn index_argument(runtime: &dyn PyRuntime, value: &PyValue) -> PyResult<i64> {
    if let Some(index) = runtime.int_value(value) {
        return Ok(index);
    }
    if runtime.kind(value)? == PyKind::Int {
        return Err(PyError::overflow_error(
            "Python int too large to convert to C ssize_t",
        ));
    }
    let actual = runtime.type_name(value)?;
    Err(PyError::type_error(format!(
        "'{actual}' object cannot be interpreted as an integer"
    )))
}

/// Return the exact index value accepted by sequence protocols. A registered boolean, like
/// NumPy's `bool`, converts with `int()` but is not an index.
pub(super) fn index<'a>(heap: &'a Heap, value: &PyValue) -> Option<NumberRef<'a>> {
    if let Some((_, KindNumber::Bool(_))) = registered_number(heap, value) {
        return None;
    }
    match view(heap, value)? {
        value @ (NumberRef::Int(_) | NumberRef::BigInt(_) | NumberRef::UInt(_)) => Some(value),
        NumberRef::Float(_) | NumberRef::Complex(..) => None,
    }
}

/// The number a builtin `complex` operator accepts as its other operand.
///
/// Like CPython's `complex` slots, it accepts registered `float` and `complex` subclasses such
/// as NumPy's `float64`, and declines other registered numbers so that their own reflected
/// operators run.
pub(super) fn complex_operand<'a>(
    runtime: &'a dyn PyRuntime,
    value: &PyValue,
) -> Option<NumberRef<'a>> {
    if let Some(kind) = runtime.value_kind_of(value) {
        if !(kind.is_float_subclass() || kind.is_complex_subclass()) {
            return None;
        }
    }
    runtime.number(value)
}

/// Coerce a real numeric value to `f64`, rejecting complex and non-numeric storage.
pub(super) fn as_f64(heap: &Heap, value: &PyValue) -> Option<f64> {
    match view(heap, value)? {
        NumberRef::Int(value) => Some(value as f64),
        NumberRef::BigInt(value) => num_traits::ToPrimitive::to_f64(value),
        NumberRef::UInt(value) => Some(value as f64),
        NumberRef::Float(value) => Some(value),
        NumberRef::Complex(..) => None,
    }
}

/// Whether a value is a builtin `complex`, for real-only paths that reject it explicitly.
pub(super) fn is_complex(heap: &Heap, value: &PyValue) -> bool {
    registered_number(heap, value).is_none()
        && matches!(view(heap, value), Some(NumberRef::Complex(..)))
}

/// Allocate a builtin complex value, e.g. for an imaginary literal or a negative base raised
/// to a fractional power.
pub(super) fn create_complex(
    runtime: &mut dyn PyRuntime,
    real: f64,
    imaginary: f64,
) -> PyResult<PyValue> {
    runtime.new_complex(real, imaginary)
}

/// Parse the textual forms accepted by the bounded `int` constructor.
///
/// The result remains decimal text so allocation and immediate-versus-bigint selection continue
/// through [`PyRuntime::new_integer`]. Bases use Python's `0` autodetection or the range 2..=36.
pub(super) fn parse_integer_text(text: &str, requested_base: i64) -> PyResult<String> {
    if requested_base != 0 && !(2..=36).contains(&requested_base) {
        return Err(PyError::value_error(
            "int() base must be >= 2 and <= 36, or 0",
        ));
    }
    let invalid = || {
        PyError::value_error(format!(
            "invalid literal for int() with base {requested_base}: {}",
            super::protocol::quote_string(text)
        ))
    };
    let mut text = text.trim();
    let negative = text.starts_with('-');
    if text.starts_with(['-', '+']) {
        text = &text[1..];
    }
    if text.is_empty() {
        return Err(invalid());
    }

    let prefixed = text.len() >= 2 && text.as_bytes()[0] == b'0';
    let prefix_base = if prefixed {
        match text.as_bytes()[1].to_ascii_lowercase() {
            b'x' => Some(16),
            b'o' => Some(8),
            b'b' => Some(2),
            _ => None,
        }
    } else {
        None
    };
    let base = if requested_base == 0 {
        prefix_base.unwrap_or(10)
    } else {
        u32::try_from(requested_base).expect("validated positive base")
    };
    let had_prefix = prefix_base == Some(base);
    if had_prefix {
        text = &text[2..];
    }
    if text.is_empty()
        || text.ends_with('_')
        || text.contains("__")
        || (text.starts_with('_') && !had_prefix)
    {
        return Err(invalid());
    }
    let digits = text.strip_prefix('_').unwrap_or(text).replace('_', "");
    if digits.is_empty() || !digits.chars().all(|character| character.is_digit(base)) {
        return Err(invalid());
    }
    let mut value = BigInt::parse_bytes(digits.as_bytes(), base).ok_or_else(invalid)?;
    if negative {
        value = -value;
    }
    Ok(value.to_string())
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum PyNumber {
    Int(i64),
    BigInt(String),
    Float(f64),
}

/// Any real number, including registered numbers such as NumPy scalars, as functions like
/// `math.sqrt` accept through `__float__` and `__index__`.
impl FromPyValue for PyNumber {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        if let Some(number) = runtime.number(&value).and_then(real_number) {
            return Ok(number);
        }
        let actual = runtime.type_name(&value)?;
        Err(PyError::type_error(format!(
            "must be real number, not {actual}"
        )))
    }
}

impl PyNumber {
    pub fn into_f64(self) -> PyResult<f64> {
        match self {
            Self::Int(value) => Ok(value as f64),
            Self::BigInt(value) => {
                let value = value
                    .parse::<f64>()
                    .map_err(|_| PyError::overflow_error("int too large to convert to float"))?;
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(PyError::overflow_error("int too large to convert to float"))
                }
            }
            Self::Float(value) => Ok(value),
        }
    }
}

pub(super) fn slot_positive(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_unary(runtime, value, UnaryNumericOperation::Positive)
}

pub(super) fn slot_negative(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_unary(runtime, value, UnaryNumericOperation::Negative)
}

pub(super) fn slot_invert(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_unary(runtime, value, UnaryNumericOperation::Invert)
}

pub(super) fn slot_absolute(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_unary(runtime, value, UnaryNumericOperation::Absolute)
}

pub(super) fn slot_add(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::Add)
}

pub(super) fn slot_subtract(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::Subtract)
}

pub(super) fn slot_reflected_subtract(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::Subtract)
}

pub(super) fn slot_multiply(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::Multiply)
}

pub(super) fn slot_power(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::Power)
}

pub(super) fn slot_reflected_power(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::Power)
}

pub(super) fn slot_divide(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::Divide)
}

pub(super) fn slot_reflected_divide(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::Divide)
}

pub(super) fn slot_floor_divide(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::FloorDivide)
}

pub(super) fn slot_reflected_floor_divide(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::FloorDivide)
}

pub(super) fn slot_remainder(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::Remainder)
}

pub(super) fn slot_reflected_remainder(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::Remainder)
}

/// `int.__divmod__` and `float.__divmod__`: the pair `(left // right, left % right)`.
pub(super) fn slot_divmod(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    divmod_numbers(runtime, left, right)
}

pub(super) fn slot_reflected_divmod(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    divmod_numbers(runtime, left, right)
}

fn divmod_numbers(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let (Some(left), Some(right)) = (try_number(runtime, left)?, try_number(runtime, right)?)
    else {
        return Ok(None);
    };
    let quotient = binary_numbers(
        runtime,
        left.clone(),
        right.clone(),
        BinaryOperator::FloorDivide,
    )?;
    let remainder = binary_numbers(runtime, left, right, BinaryOperator::Remainder)?;
    match (quotient, remainder) {
        (Some(quotient), Some(remainder)) => {
            Ok(Some(runtime.new_tuple(vec![quotient, remainder])?))
        }
        _ => Ok(None),
    }
}

pub(super) fn slot_bitwise_and(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::BitwiseAnd)
}

pub(super) fn slot_left_shift(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::LeftShift)
}

pub(super) fn slot_right_shift(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::RightShift)
}

pub(super) fn slot_bitwise_xor(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::BitwiseXor)
}

pub(super) fn slot_bitwise_or(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, BinaryOperator::BitwiseOr)
}

/// Run a builtin numeric operation without consulting Python type slots.
///
/// Only values whose physical representation proves they are exact builtin integers, booleans,
/// or floats enter this path. Heap-backed subclasses and registered value kinds return `None` and
/// retain normal reflected-operation dispatch.
#[inline]
pub(super) fn exact_binary(
    runtime: &mut dyn PyRuntime,
    operation: BinaryOperator,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    if matches!(operation, BinaryOperator::MatrixMultiply) {
        return Ok(None);
    }
    if let (Some(left), Some(right)) = (exact_integer(left), exact_integer(right)) {
        if let Some(result) = immediate_integer_binary(operation, left, right) {
            return Ok(Some(PyValue::Int(result)));
        }
    }
    let Some(left) = exact_number(left) else {
        return Ok(None);
    };
    let Some(right) = exact_number(right) else {
        return Ok(None);
    };
    binary_numbers(runtime, left, right, operation)
}

/// Compare exact immediate integers and booleans without invoking their builtin slots.
///
/// Float and heap-backed integer comparison stays on the general protocol path, which owns NaN
/// and arbitrary-precision ordering semantics.
#[inline]
pub(super) fn exact_integer_comparison(
    operation: ComparisonOperator,
    left: PyValue,
    right: PyValue,
) -> Option<bool> {
    let left = exact_integer(left)?;
    let right = exact_integer(right)?;
    Some(match operation {
        ComparisonOperator::Equal => left == right,
        ComparisonOperator::NotEqual => left != right,
        ComparisonOperator::Less => left < right,
        ComparisonOperator::LessEqual => left <= right,
        ComparisonOperator::Greater => left > right,
        ComparisonOperator::GreaterEqual => left >= right,
        ComparisonOperator::In
        | ComparisonOperator::NotIn
        | ComparisonOperator::Is
        | ComparisonOperator::IsNot => return None,
    })
}

fn slot_binary(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    operation: BinaryOperator,
) -> PyResult<Option<PyValue>> {
    let Some(left) = try_number(runtime, left)? else {
        return Ok(None);
    };
    let Some(right) = try_number(runtime, right)? else {
        return Ok(None);
    };
    binary_numbers(runtime, left, right, operation)
}

/// Python's float floor division and remainder for a nonzero divisor. The remainder takes the
/// divisor's sign, including a signed zero, and `left == quotient * right + remainder` up to
/// rounding. A finite dividend over an infinite divisor with the opposite sign has quotient
/// `-1` and an infinite remainder; an infinite or NaN dividend gives NaN for both.
fn float_divmod(left: f64, right: f64) -> (f64, f64) {
    // Rust's `%` is C `fmod`: exact, with the dividend's sign.
    let mut remainder = left % right;
    let mut quotient = (left - remainder) / right;
    if remainder != 0.0 && (remainder < 0.0) != (right < 0.0) {
        remainder += right;
        quotient -= 1.0;
    }
    if remainder == 0.0 {
        remainder = 0.0_f64.copysign(right);
    }
    if quotient == 0.0 {
        // Keep true division's sign, so `0.0 // -1.0` and `1.0 // -inf` are `-0.0`.
        return (0.0_f64.copysign(left / right), remainder);
    }
    // `quotient` is integral in exact arithmetic; rounding removes division error.
    (quotient.round(), remainder)
}

fn binary_numbers(
    runtime: &mut dyn PyRuntime,
    left: PyNumber,
    right: PyNumber,
    operation: BinaryOperator,
) -> PyResult<Option<PyValue>> {
    if matches!(left, PyNumber::Float(_)) || matches!(right, PyNumber::Float(_)) {
        if matches!(
            operation,
            BinaryOperator::BitwiseAnd
                | BinaryOperator::BitwiseXor
                | BinaryOperator::BitwiseOr
                | BinaryOperator::LeftShift
                | BinaryOperator::RightShift
                | BinaryOperator::MatrixMultiply
        ) {
            return Ok(None);
        }
        let left = left.into_f64()?;
        let right = right.into_f64()?;
        if matches!(
            operation,
            BinaryOperator::Divide | BinaryOperator::FloorDivide | BinaryOperator::Remainder
        ) && right == 0.0
        {
            return Err(PyError::zero_division_error("division by zero"));
        }
        return Ok(Some(PyValue::Float(match operation {
            BinaryOperator::Add => left + right,
            BinaryOperator::Subtract => left - right,
            BinaryOperator::Multiply => left * right,
            BinaryOperator::Power => {
                // Infinite and NaN operands follow IEEE 754 `pow`, as in CPython. Only finite
                // operands raise, and a negative base with a fractional exponent gives the
                // principal complex root.
                if left == 0.0 && right < 0.0 && right.is_finite() {
                    return Err(PyError::zero_division_error("zero to a negative power"));
                }
                if left < 0.0 && left.is_finite() && right.is_finite() && right.fract() != 0.0 {
                    let magnitude = (-left).powf(right);
                    let angle = std::f64::consts::PI * right;
                    return create_complex(
                        runtime,
                        magnitude * angle.cos(),
                        magnitude * angle.sin(),
                    )
                    .map(Some);
                }
                let value = left.powf(right);
                if value.is_infinite() && left.is_finite() && right.is_finite() {
                    return Err(PyError::overflow_error(
                        "(34, 'Numerical result out of range')",
                    ));
                }
                value
            }
            BinaryOperator::Divide => left / right,
            BinaryOperator::FloorDivide => float_divmod(left, right).0,
            BinaryOperator::Remainder => float_divmod(left, right).1,
            BinaryOperator::BitwiseAnd
            | BinaryOperator::BitwiseXor
            | BinaryOperator::BitwiseOr
            | BinaryOperator::LeftShift
            | BinaryOperator::RightShift
            | BinaryOperator::MatrixMultiply => {
                unreachable!("integer-only operation rejected above")
            }
        })));
    }

    if matches!(operation, BinaryOperator::Divide) {
        let left = left.into_f64()?;
        let right = right.into_f64()?;
        if right == 0.0 {
            return Err(PyError::zero_division_error("division by zero"));
        }
        return Ok(Some(PyValue::Float(left / right)));
    }
    if matches!(operation, BinaryOperator::Power) {
        let exponent = integer_decimal(right)
            .parse::<BigInt>()
            .map_err(|_| PyError::runtime_error("invalid internal integer representation"))?;
        if exponent.is_negative() {
            let left = left.into_f64()?;
            let right = exponent
                .to_f64()
                .ok_or_else(|| PyError::overflow_error("power exponent is too large"))?;
            if left == 0.0 {
                return Err(PyError::zero_division_error("zero to a negative power"));
            }
            return Ok(Some(PyValue::Float(left.powf(right))));
        }
        let exponent = exponent
            .to_u32()
            .ok_or_else(|| PyError::resource_error("power exponent is too large"))?;
        let left = integer_decimal(left);
        let result_bound = left
            .len()
            .checked_mul(exponent as usize)
            .and_then(|bytes| bytes.checked_add(2))
            .ok_or_else(|| PyError::resource_error("power result is too large"))?;
        runtime.charge_cpu(u64::try_from(result_bound.max(1)).unwrap_or(u64::MAX))?;
        runtime.reserve_memory(result_bound)?;
        let left = left
            .parse::<BigInt>()
            .map_err(|_| PyError::runtime_error("invalid internal integer representation"))?;
        return runtime
            .new_integer(&left.pow(exponent).to_string())
            .map(Some);
    }
    if matches!(
        operation,
        BinaryOperator::LeftShift | BinaryOperator::RightShift
    ) {
        let shift = integer_decimal(right)
            .parse::<BigInt>()
            .map_err(|_| PyError::runtime_error("invalid internal integer representation"))?;
        if shift.is_negative() {
            return Err(PyError::value_error("negative shift count"));
        }
        let shift = shift
            .to_usize()
            .ok_or_else(|| PyError::resource_error("shift count is too large"))?;
        let left = integer_decimal(left);
        let result_bound = left
            .len()
            .checked_add(shift.saturating_add(2) / 3)
            .and_then(|bytes| bytes.checked_add(2))
            .ok_or_else(|| PyError::resource_error("shift result is too large"))?;
        runtime.charge_cpu(u64::try_from(result_bound.max(1)).unwrap_or(u64::MAX))?;
        runtime.reserve_memory(result_bound)?;
        let left = left
            .parse::<BigInt>()
            .map_err(|_| PyError::runtime_error("invalid internal integer representation"))?;
        let result = if matches!(operation, BinaryOperator::LeftShift) {
            left << shift
        } else {
            left >> shift
        };
        return runtime.new_integer(&result.to_string()).map(Some);
    }
    if let (PyNumber::Int(left), PyNumber::Int(right)) = (&left, &right) {
        if let Some(result) = immediate_integer_binary(operation, *left, *right) {
            return Ok(Some(PyValue::Int(result)));
        }
    }

    let left = integer_decimal(left);
    let right = integer_decimal(right);
    let work = left
        .len()
        .checked_add(right.len())
        .ok_or_else(|| PyError::resource_error("integer operation is too large"))?;
    runtime.charge_cpu(u64::try_from(work).unwrap_or(u64::MAX))?;
    let result_bound = match operation {
        BinaryOperator::Add
        | BinaryOperator::Subtract
        | BinaryOperator::BitwiseAnd
        | BinaryOperator::BitwiseXor
        | BinaryOperator::BitwiseOr => left.len().max(right.len()).saturating_add(2),
        BinaryOperator::Multiply => work.saturating_add(1),
        BinaryOperator::Power => unreachable!("power returned above"),
        BinaryOperator::LeftShift | BinaryOperator::RightShift => {
            unreachable!("shifts returned above")
        }
        BinaryOperator::FloorDivide | BinaryOperator::Remainder => {
            left.len().max(right.len()).saturating_add(2)
        }
        BinaryOperator::Divide => unreachable!("division returned above"),
        BinaryOperator::MatrixMultiply => return Ok(None),
    };
    runtime.reserve_memory(result_bound)?;
    let left = left
        .parse::<BigInt>()
        .map_err(|_| PyError::runtime_error("invalid internal integer representation"))?;
    let right = right
        .parse::<BigInt>()
        .map_err(|_| PyError::runtime_error("invalid internal integer representation"))?;
    if right.is_zero()
        && matches!(
            operation,
            BinaryOperator::FloorDivide | BinaryOperator::Remainder
        )
    {
        return Err(PyError::zero_division_error("division by zero"));
    }
    let result = match operation {
        BinaryOperator::Add => left + right,
        BinaryOperator::Subtract => left - right,
        BinaryOperator::Multiply => left * right,
        BinaryOperator::Power => unreachable!("power returned above"),
        BinaryOperator::LeftShift | BinaryOperator::RightShift => {
            unreachable!("shifts returned above")
        }
        BinaryOperator::Divide => unreachable!("division returned above"),
        BinaryOperator::FloorDivide => bigint_floor_div(&left, &right),
        BinaryOperator::Remainder => {
            let quotient = bigint_floor_div(&left, &right);
            left - quotient * right
        }
        BinaryOperator::BitwiseAnd => left & right,
        BinaryOperator::BitwiseXor => left ^ right,
        BinaryOperator::BitwiseOr => left | right,
        BinaryOperator::MatrixMultiply => unreachable!("matrix multiplication returned above"),
    };
    runtime.new_integer(&result.to_string()).map(Some)
}

#[inline]
fn exact_number(value: PyValue) -> Option<PyNumber> {
    match value.tag() {
        ValueTag::Int | ValueTag::Bool => Some(PyNumber::Int(
            value.immediate_int().expect("integer tag checked"),
        )),
        ValueTag::Float => Some(PyNumber::Float(
            value.float_value().expect("float tag checked"),
        )),
        _ => None,
    }
}

#[inline]
fn exact_integer(value: PyValue) -> Option<i64> {
    match value.tag() {
        ValueTag::Int | ValueTag::Bool => value.immediate_int(),
        _ => None,
    }
}

#[inline]
fn immediate_integer_binary(operation: BinaryOperator, left: i64, right: i64) -> Option<i64> {
    match operation {
        BinaryOperator::Add => left.checked_add(right),
        BinaryOperator::Subtract => left.checked_sub(right),
        BinaryOperator::Multiply => left.checked_mul(right),
        BinaryOperator::BitwiseAnd => Some(left & right),
        BinaryOperator::BitwiseXor => Some(left ^ right),
        BinaryOperator::BitwiseOr => Some(left | right),
        BinaryOperator::FloorDivide => python_floor_div(left, right),
        BinaryOperator::Remainder => python_remainder(left, right),
        BinaryOperator::MatrixMultiply
        | BinaryOperator::Power
        | BinaryOperator::Divide
        | BinaryOperator::LeftShift
        | BinaryOperator::RightShift => None,
    }
}

fn python_floor_div(left: i64, right: i64) -> Option<i64> {
    let quotient = left.checked_div(right)?;
    let remainder = left.checked_rem(right)?;
    Some(if remainder != 0 && (remainder < 0) != (right < 0) {
        quotient - 1
    } else {
        quotient
    })
}

fn python_remainder(left: i64, right: i64) -> Option<i64> {
    if right == 0 {
        return None;
    }
    if left == i64::MIN && right == -1 {
        return Some(0);
    }
    let remainder = left % right;
    Some(if remainder != 0 && (remainder < 0) != (right < 0) {
        remainder + right
    } else {
        remainder
    })
}

#[derive(Clone, Copy)]
enum UnaryNumericOperation {
    Positive,
    Negative,
    Invert,
    Absolute,
}

fn slot_unary(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    operation: UnaryNumericOperation,
) -> PyResult<Option<PyValue>> {
    let Some(value) = try_number(runtime, value)? else {
        return Ok(None);
    };
    match value {
        PyNumber::Float(value) => Ok(match operation {
            UnaryNumericOperation::Positive => Some(PyValue::Float(value)),
            UnaryNumericOperation::Negative => Some(PyValue::Float(-value)),
            UnaryNumericOperation::Invert => None,
            UnaryNumericOperation::Absolute => Some(PyValue::Float(value.abs())),
        }),
        PyNumber::Int(value) => {
            let immediate = match operation {
                UnaryNumericOperation::Positive => Some(value),
                UnaryNumericOperation::Negative => value.checked_neg(),
                UnaryNumericOperation::Invert => Some(!value),
                UnaryNumericOperation::Absolute => value.checked_abs(),
            };
            if let Some(value) = immediate {
                return Ok(Some(PyValue::Int(value)));
            }
            let result = match operation {
                UnaryNumericOperation::Negative => -BigInt::from(value),
                UnaryNumericOperation::Absolute => BigInt::from(value).abs(),
                UnaryNumericOperation::Positive | UnaryNumericOperation::Invert => {
                    unreachable!("these immediate operations cannot overflow")
                }
            };
            runtime.new_integer(&result.to_string()).map(Some)
        }
        PyNumber::BigInt(value) => {
            runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
            runtime.reserve_memory(value.len().saturating_add(2))?;
            let value = value
                .parse::<BigInt>()
                .map_err(|_| PyError::runtime_error("invalid internal integer representation"))?;
            let result = match operation {
                UnaryNumericOperation::Positive => value,
                UnaryNumericOperation::Negative => -value,
                UnaryNumericOperation::Invert => !value,
                UnaryNumericOperation::Absolute => value.abs(),
            };
            runtime.new_integer(&result.to_string()).map(Some)
        }
    }
}

fn bigint_floor_div(left: &BigInt, right: &BigInt) -> BigInt {
    let mut quotient = left / right;
    let remainder = left % right;
    if !remainder.is_zero() && remainder.is_negative() != right.is_negative() {
        quotient -= 1;
    }
    quotient
}

/// The operand of a builtin `int` or `float` operator. Registered numbers decline, as CPython's
/// `int.__add__` declines anything but `int`, so `1 + np.int8(127)` reaches NumPy's reflected
/// operator and keeps int8 wrapping.
fn try_number(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Option<PyNumber>> {
    if runtime.value_kind_of(&value).is_some() {
        return Ok(None);
    }
    Ok(runtime.number(&value).and_then(real_number))
}

/// A real number as the owned [`PyNumber`] builtin arithmetic computes with.
fn real_number(number: NumberRef<'_>) -> Option<PyNumber> {
    match number {
        NumberRef::Int(value) => Some(PyNumber::Int(value)),
        NumberRef::Float(value) => Some(PyNumber::Float(value)),
        NumberRef::BigInt(_) | NumberRef::UInt(_) => number
            .to_bigint()
            .map(|value| PyNumber::BigInt(value.to_string())),
        NumberRef::Complex(..) => None,
    }
}

fn integer_decimal(value: PyNumber) -> String {
    match value {
        PyNumber::Int(value) => value.to_string(),
        PyNumber::BigInt(value) => value,
        PyNumber::Float(_) => unreachable!("float arithmetic returned above"),
    }
}

/// Resolve the Python integer protocol to a bounded repetition count at the erased ABI boundary.
pub(super) fn runtime_repeat_count(
    runtime: &mut dyn PyRuntime,
    value: &PyValue,
) -> PyResult<Option<usize>> {
    if let Some(value) = runtime.int_value(value) {
        return usize::try_from(value.max(0))
            .map(Some)
            .map_err(|_| PyError::overflow_error("sequence repeat is too large"));
    }
    let Some(value) = runtime.integer_text(value)? else {
        return Ok(None);
    };
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let value = value
        .parse::<BigInt>()
        .map_err(|_| PyError::runtime_error("invalid internal integer representation"))?;
    if value.is_negative() {
        Ok(Some(0))
    } else {
        value
            .to_usize()
            .map(Some)
            .ok_or_else(|| PyError::overflow_error("sequence repeat is too large"))
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_integer_text, python_floor_div, python_remainder};

    #[test]
    fn integer_text_parsing_handles_bases_signs_and_separators() {
        assert_eq!(parse_integer_text("ff", 16).unwrap(), "255");
        assert_eq!(parse_integer_text(" -0b1_010 ", 0).unwrap(), "-10");
        assert_eq!(parse_integer_text("0x_ff", 16).unwrap(), "255");
        assert!(parse_integer_text("10", 1).is_err());
        assert!(parse_integer_text("_10", 10).is_err());
        assert!(parse_integer_text("1__0", 10).is_err());
        assert!(parse_integer_text("2", 2).is_err());
    }

    #[test]
    fn immediate_floor_division_and_remainder_follow_python_signs() {
        for (left, right, quotient, remainder) in [
            (7, 3, 2, 1),
            (-7, 3, -3, 2),
            (7, -3, -3, -2),
            (-7, -3, 2, -1),
        ] {
            assert_eq!(python_floor_div(left, right), Some(quotient));
            assert_eq!(python_remainder(left, right), Some(remainder));
        }
        assert_eq!(python_floor_div(i64::MIN, -1), None);
        assert_eq!(python_remainder(i64::MIN, -1), Some(0));
        assert_eq!(python_floor_div(1, 0), None);
        assert_eq!(python_remainder(1, 0), None);
    }
}
