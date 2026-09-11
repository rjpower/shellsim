//! Central numeric views and coercions for the bounded Python runtime.
//!
//! Immediate integers, heap-backed arbitrary-precision integers, and IEEE-754 doubles all cross
//! the native-module boundary through this owned, representation-independent view.

use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive};

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
        let left = left.into_f64()?;
        let right = right.into_f64()?;
        return Ok(Some(PyValue::Float(match operation {
            NumericOperation::Add => left + right,
            NumericOperation::Subtract => left - right,
            NumericOperation::Multiply => left * right,
        })));
    }
    if let (PyNumber::Int(left), PyNumber::Int(right)) = (&left, &right) {
        let result = match operation {
            NumericOperation::Add => left.checked_add(*right),
            NumericOperation::Subtract => left.checked_sub(*right),
            NumericOperation::Multiply => left.checked_mul(*right),
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
        NumericOperation::Add | NumericOperation::Subtract => left.len().max(right.len()) + 2,
        NumericOperation::Multiply => work.saturating_add(1),
    };
    runtime.reserve_memory(result_bound)?;
    let left = left
        .parse::<BigInt>()
        .map_err(|_| PyError::runtime_error("invalid internal integer representation"))?;
    let right = right
        .parse::<BigInt>()
        .map_err(|_| PyError::runtime_error("invalid internal integer representation"))?;
    let result = match operation {
        NumericOperation::Add => left + right,
        NumericOperation::Subtract => left - right,
        NumericOperation::Multiply => left * right,
    };
    runtime.new_integer(&result.to_string()).map(Some)
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
