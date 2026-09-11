//! Central numeric views and coercions for the bounded Python runtime.
//!
//! Immediate integers, heap-backed arbitrary-precision integers, and IEEE-754 doubles all cross
//! the native-module boundary through this owned, representation-independent view.

use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};

use super::heap::{Heap, InstancePayload, Object};
use super::native::{FromPyValue, PyError, PyResult, PyRuntime, PyValue};

/// Borrowed numeric payload used by VM protocols without exposing physical value tags.
#[derive(Clone, Copy, Debug)]
pub(super) enum NumberRef<'a> {
    Int(i64),
    BigInt(&'a BigInt),
    Float(f64),
}

/// Resolve Python numeric storage into one semantic numeric view.
pub(super) fn view<'a>(heap: &'a Heap, value: &PyValue) -> Option<NumberRef<'a>> {
    if let Some(value) = value.float_value() {
        return Some(NumberRef::Float(value));
    }
    if let Some(value) = value.bool_value() {
        return Some(NumberRef::Int(i64::from(value)));
    }
    if value.tag() == super::ValueTag::Int {
        return Some(NumberRef::Int(value.immediate_int().expect("tag checked")));
    }
    match value.object_id().and_then(|id| heap.get(id).ok())? {
        Object::BigInt(value) => Some(NumberRef::BigInt(value)),
        Object::Instance {
            payload: InstancePayload::Int(value),
            ..
        } => Some(NumberRef::Int(*value)),
        _ => None,
    }
}

/// Return the exact index value accepted by sequence protocols.
pub(super) fn index<'a>(heap: &'a Heap, value: &PyValue) -> Option<NumberRef<'a>> {
    match view(heap, value)? {
        value @ (NumberRef::Int(_) | NumberRef::BigInt(_)) => Some(value),
        NumberRef::Float(_) => None,
    }
}

/// Coerce a real numeric value to `f64`, rejecting non-numeric storage.
pub(super) fn as_f64(heap: &Heap, value: &PyValue) -> Option<f64> {
    match view(heap, value)? {
        NumberRef::Int(value) => Some(value as f64),
        NumberRef::BigInt(value) => num_traits::ToPrimitive::to_f64(value),
        NumberRef::Float(value) => Some(value),
    }
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
    let mut text = text.trim();
    let negative = text.starts_with('-');
    if text.starts_with(['-', '+']) {
        text = &text[1..];
    }
    if text.is_empty() {
        return Err(PyError::value_error("invalid literal for int()"));
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
        return Err(PyError::value_error("invalid literal for int()"));
    }
    let digits = text.strip_prefix('_').unwrap_or(text).replace('_', "");
    if digits.is_empty() || !digits.chars().all(|character| character.is_digit(base)) {
        return Err(PyError::value_error("invalid literal for int()"));
    }
    let mut value = BigInt::parse_bytes(digits.as_bytes(), base)
        .ok_or_else(|| PyError::value_error("invalid literal for int()"))?;
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

impl FromPyValue for PyNumber {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        if let Some(value) = runtime.int_value(&value) {
            return Ok(Self::Int(value));
        }
        if let Some(value) = runtime.integer_text(&value)? {
            return Ok(Self::BigInt(value));
        }
        if let Some(value) = value.float_value() {
            Ok(Self::Float(value))
        } else {
            let actual = runtime.type_name(&value)?;
            Err(PyError::type_error(format!(
                "expected a real number, got {actual}"
            )))
        }
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

#[derive(Clone, Copy)]
enum NumericOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    FloorDivide,
    Remainder,
    BitwiseAnd,
    BitwiseXor,
    BitwiseOr,
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
    slot_binary(runtime, left, right, NumericOperation::Add)
}

pub(super) fn slot_subtract(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, NumericOperation::Subtract)
}

pub(super) fn slot_reflected_subtract(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, NumericOperation::Subtract)
}

pub(super) fn slot_multiply(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, NumericOperation::Multiply)
}

pub(super) fn slot_divide(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, NumericOperation::Divide)
}

pub(super) fn slot_reflected_divide(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, NumericOperation::Divide)
}

pub(super) fn slot_floor_divide(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, NumericOperation::FloorDivide)
}

pub(super) fn slot_reflected_floor_divide(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, NumericOperation::FloorDivide)
}

pub(super) fn slot_remainder(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, NumericOperation::Remainder)
}

pub(super) fn slot_reflected_remainder(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, NumericOperation::Remainder)
}

pub(super) fn slot_bitwise_and(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, NumericOperation::BitwiseAnd)
}

pub(super) fn slot_bitwise_xor(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, NumericOperation::BitwiseXor)
}

pub(super) fn slot_bitwise_or(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_binary(runtime, left, right, NumericOperation::BitwiseOr)
}

fn slot_binary(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    operation: NumericOperation,
) -> PyResult<Option<PyValue>> {
    let Some(left) = try_number(runtime, left)? else {
        return Ok(None);
    };
    let Some(right) = try_number(runtime, right)? else {
        return Ok(None);
    };
    if matches!(left, PyNumber::Float(_)) || matches!(right, PyNumber::Float(_)) {
        if matches!(
            operation,
            NumericOperation::BitwiseAnd
                | NumericOperation::BitwiseXor
                | NumericOperation::BitwiseOr
        ) {
            return Ok(None);
        }
        let left = left.into_f64()?;
        let right = right.into_f64()?;
        if matches!(
            operation,
            NumericOperation::Divide | NumericOperation::FloorDivide | NumericOperation::Remainder
        ) && right == 0.0
        {
            return Err(PyError::zero_division_error(
                if matches!(operation, NumericOperation::Divide) {
                    "division by zero"
                } else {
                    "float division or modulo by zero"
                },
            ));
        }
        return Ok(Some(PyValue::Float(match operation {
            NumericOperation::Add => left + right,
            NumericOperation::Subtract => left - right,
            NumericOperation::Multiply => left * right,
            NumericOperation::Divide => left / right,
            NumericOperation::FloorDivide => (left / right).floor(),
            NumericOperation::Remainder => left - (left / right).floor() * right,
            NumericOperation::BitwiseAnd
            | NumericOperation::BitwiseXor
            | NumericOperation::BitwiseOr => unreachable!("bitwise float rejected above"),
        })));
    }

    if matches!(operation, NumericOperation::Divide) {
        let left = left.into_f64()?;
        let right = right.into_f64()?;
        if right == 0.0 {
            return Err(PyError::zero_division_error("division by zero"));
        }
        return Ok(Some(PyValue::Float(left / right)));
    }
    if let (PyNumber::Int(left), PyNumber::Int(right)) = (&left, &right) {
        let result = match operation {
            NumericOperation::Add => left.checked_add(*right),
            NumericOperation::Subtract => left.checked_sub(*right),
            NumericOperation::Multiply => left.checked_mul(*right),
            NumericOperation::Divide => unreachable!("division returned above"),
            NumericOperation::FloorDivide | NumericOperation::Remainder => None,
            NumericOperation::BitwiseAnd => Some(*left & *right),
            NumericOperation::BitwiseXor => Some(*left ^ *right),
            NumericOperation::BitwiseOr => Some(*left | *right),
        };
        if let Some(result) = result {
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
        NumericOperation::Add
        | NumericOperation::Subtract
        | NumericOperation::BitwiseAnd
        | NumericOperation::BitwiseXor
        | NumericOperation::BitwiseOr => left.len().max(right.len()).saturating_add(2),
        NumericOperation::Multiply => work.saturating_add(1),
        NumericOperation::FloorDivide | NumericOperation::Remainder => {
            left.len().max(right.len()).saturating_add(2)
        }
        NumericOperation::Divide => unreachable!("division returned above"),
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
            NumericOperation::FloorDivide | NumericOperation::Remainder
        )
    {
        return Err(PyError::zero_division_error(
            "integer division or modulo by zero",
        ));
    }
    let result = match operation {
        NumericOperation::Add => left + right,
        NumericOperation::Subtract => left - right,
        NumericOperation::Multiply => left * right,
        NumericOperation::Divide => unreachable!("division returned above"),
        NumericOperation::FloorDivide => bigint_floor_div(&left, &right),
        NumericOperation::Remainder => {
            let quotient = bigint_floor_div(&left, &right);
            left - quotient * right
        }
        NumericOperation::BitwiseAnd => left & right,
        NumericOperation::BitwiseXor => left ^ right,
        NumericOperation::BitwiseOr => left | right,
    };
    runtime.new_integer(&result.to_string()).map(Some)
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

fn try_number(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Option<PyNumber>> {
    if let Some(value) = runtime.int_value(&value) {
        return Ok(Some(PyNumber::Int(value)));
    }
    if let Some(value) = runtime.integer_text(&value)? {
        return Ok(Some(PyNumber::BigInt(value)));
    }
    Ok(value.float_value().map(PyNumber::Float))
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
    use super::parse_integer_text;

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
}
