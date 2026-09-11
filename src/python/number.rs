//! Central numeric views and coercions for the bounded Python runtime.
//!
//! The runtime currently stores signed 64-bit integers and IEEE-754 doubles. Keeping promotion
//! and conversion behind `PyNumber` gives a single future extension point for arena-backed big
//! integers without exposing their representation to modules.

use super::native::{FromPyValue, PyError, PyResult, PyRuntime, PyValue};
use super::Value;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum PyNumber {
    Int(i64),
    Float(f64),
}

impl FromPyValue for PyNumber {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        if let Some(value) = runtime.int_value(&value) {
            return Ok(Self::Int(value));
        }
        match value {
            Value::Float(value) => Ok(Self::Float(value)),
            other => {
                let actual = runtime.type_name(&other)?;
                Err(PyError::type_error(format!(
                    "expected a real number, got {actual}"
                )))
            }
        }
    }
}

impl PyNumber {
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Int(value) => value as f64,
            Self::Float(value) => value,
        }
    }
}
