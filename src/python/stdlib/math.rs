//! A small, deterministic subset of Python's :mod:`math` module.
//!
//! This module contains no shellsim or host capabilities.  The VM can translate its `MathValue`
//! and `MathError` values into Python objects/exceptions when it wires the module into imports.
//! Functions intentionally accept `f64`: [`PyNumber`] performs checked coercion from immediate
//! and arbitrary-precision integers without exposing either VM representation here.

use std::fmt;

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyConstant, PyError, PyResult, PyRuntime, PyValueCast,
    ValueDef,
};
use super::super::number::PyNumber;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "math",
    functions: &[
        FunctionDef {
            module: "math",
            name: "acos",
            call: native_acos,
        },
        FunctionDef {
            module: "math",
            name: "asin",
            call: native_asin,
        },
        FunctionDef {
            module: "math",
            name: "atan",
            call: native_atan,
        },
        FunctionDef {
            module: "math",
            name: "atan2",
            call: native_atan2,
        },
        FunctionDef {
            module: "math",
            name: "ceil",
            call: native_ceil,
        },
        FunctionDef {
            module: "math",
            name: "cos",
            call: native_cos,
        },
        FunctionDef {
            module: "math",
            name: "degrees",
            call: native_degrees,
        },
        FunctionDef {
            module: "math",
            name: "exp",
            call: native_exp,
        },
        FunctionDef {
            module: "math",
            name: "hypot",
            call: native_hypot,
        },
        FunctionDef {
            module: "math",
            name: "isinf",
            call: native_isinf,
        },
        FunctionDef {
            module: "math",
            name: "isnan",
            call: native_isnan,
        },
        FunctionDef {
            module: "math",
            name: "log",
            call: native_log,
        },
        FunctionDef {
            module: "math",
            name: "log10",
            call: native_log10,
        },
        FunctionDef {
            module: "math",
            name: "pow",
            call: native_pow,
        },
        FunctionDef {
            module: "math",
            name: "radians",
            call: native_radians,
        },
        FunctionDef {
            module: "math",
            name: "sin",
            call: native_sin,
        },
        FunctionDef {
            module: "math",
            name: "sqrt",
            call: native_sqrt,
        },
        FunctionDef {
            module: "math",
            name: "tan",
            call: native_tan,
        },
    ],
    values: &[
        ValueDef::Constant {
            name: "e",
            value: PyConstant::Float(std::f64::consts::E),
        },
        ValueDef::Constant {
            name: "pi",
            value: PyConstant::Float(std::f64::consts::PI),
        },
        ValueDef::Constant {
            name: "tau",
            value: PyConstant::Float(std::f64::consts::TAU),
        },
        ValueDef::Constant {
            name: "inf",
            value: PyConstant::Float(f64::INFINITY),
        },
        ValueDef::Constant {
            name: "nan",
            value: PyConstant::Float(f64::NAN),
        },
    ],
};

fn native_acos(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "acos")
}

fn native_asin(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "asin")
}

fn native_atan(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "atan")
}

fn native_atan2(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "atan2")
}

fn native_ceil(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "ceil")
}

fn native_cos(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "cos")
}

fn native_degrees(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "degrees")
}

fn native_exp(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "exp")
}

fn native_hypot(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "hypot")
}

fn native_isinf(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "isinf")
}

fn native_isnan(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "isnan")
}

fn native_log(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "log")
}

fn native_log10(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "log10")
}

fn native_pow(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "pow")
}

fn native_radians(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "radians")
}

fn native_sin(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "sin")
}

fn native_sqrt(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "sqrt")
}

fn native_tan(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "tan")
}

fn native_call(runtime: &mut dyn PyRuntime, args: CallArgs, name: &'static str) -> PyResult {
    args.reject_keywords(&format!("math.{name}"))?;
    let values = args
        .positional()
        .iter()
        .cloned()
        .map(|value| value.cast::<PyNumber>(runtime).and_then(PyNumber::into_f64))
        .collect::<PyResult<Vec<_>>>()?;
    runtime.charge_cpu(u64::try_from(values.len()).unwrap_or(u64::MAX))?;
    let value = call(name, &values).map_err(math_error)?;
    Ok(match value {
        MathValue::Float(value) => Value::Float(value),
        MathValue::Int(value) => Value::Int(value),
        MathValue::Bool(value) => Value::Bool(value),
    })
}

fn math_error(error: MathError) -> PyError {
    let message = error.to_string();
    match error {
        MathError::ValueError(_) => PyError::value_error(message),
        MathError::OverflowError(_) => PyError::overflow_error(message),
        MathError::ZeroDivisionError(_) => PyError::zero_division_error(message),
        MathError::Arity { .. } => PyError::type_error(message),
        MathError::UnknownFunction(_) => PyError::runtime_error(message),
    }
}

/// Values returned by the math functions in this shim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MathValue {
    Float(f64),
    Int(i64),
    Bool(bool),
}

/// Python-facing failure classes used by the dispatcher.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MathError {
    /// The requested function is not part of this intentionally small surface.
    UnknownFunction(String),
    /// The number of positional arguments was not accepted by the function.
    Arity {
        function: &'static str,
        expected: &'static str,
        actual: usize,
    },
    /// Python's `ValueError` for a domain or conversion failure.
    ValueError(&'static str),
    /// Python's `OverflowError` for an unrepresentable result.
    OverflowError(&'static str),
    /// `math.log(x, 1)` follows Python's division-by-zero behavior.
    ZeroDivisionError(&'static str),
}

impl fmt::Display for MathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownFunction(name) => write!(formatter, "unknown math function {name:?}"),
            Self::Arity {
                function,
                expected,
                actual,
            } => write!(
                formatter,
                "math.{function}() takes {expected} arguments ({actual} given)"
            ),
            Self::ValueError(message) => formatter.write_str(message),
            Self::OverflowError(message) => formatter.write_str(message),
            Self::ZeroDivisionError(message) => formatter.write_str(message),
        }
    }
}

pub type MathResult = Result<MathValue, MathError>;

const I64_EXCLUSIVE_UPPER: f64 = 9_223_372_036_854_775_808.0; // 2**63
const I64_INCLUSIVE_LOWER: f64 = -9_223_372_036_854_775_808.0; // -2**63

/// Return a named numeric constant from the supported `math` surface.
///
/// `inf` and `nan` are useful for deterministic wrappers and mirror Python's module constants.
#[cfg(test)]
pub fn constant(name: &str) -> Option<MathValue> {
    match name {
        "e" => Some(MathValue::Float(std::f64::consts::E)),
        "pi" => Some(MathValue::Float(std::f64::consts::PI)),
        "tau" => Some(MathValue::Float(std::f64::consts::TAU)),
        "inf" => Some(MathValue::Float(f64::INFINITY)),
        "nan" => Some(MathValue::Float(f64::NAN)),
        _ => None,
    }
}

/// Dispatch one of the observed/reviewer-requested `math` functions.
pub fn call(name: &str, args: &[f64]) -> MathResult {
    match name {
        "acos" => unary("acos", args, acos),
        "asin" => unary("asin", args, asin),
        "atan" => unary("atan", args, |value| Ok(MathValue::Float(value.atan()))),
        "atan2" => binary("atan2", args, |y, x| Ok(MathValue::Float(y.atan2(x)))),
        "ceil" => unary("ceil", args, ceil),
        "cos" => unary("cos", args, cos),
        "degrees" => unary("degrees", args, |value| {
            Ok(MathValue::Float(value.to_degrees()))
        }),
        "exp" => unary("exp", args, exp),
        "hypot" => Ok(MathValue::Float(
            args.iter().copied().fold(0.0_f64, f64::hypot),
        )),
        "isinf" => unary("isinf", args, |value| Ok(MathValue::Bool(isinf(value)))),
        "isnan" => unary("isnan", args, |value| Ok(MathValue::Bool(isnan(value)))),
        "log" => match args {
            [value] => log(*value, None),
            [value, base] => log(*value, Some(*base)),
            _ => Err(MathError::Arity {
                function: "log",
                expected: "1 or 2",
                actual: args.len(),
            }),
        },
        "log10" => unary("log10", args, log10),
        "pow" => binary("pow", args, pow),
        "radians" => unary("radians", args, |value| {
            Ok(MathValue::Float(value.to_radians()))
        }),
        "sin" => unary("sin", args, sin),
        "sqrt" => unary("sqrt", args, sqrt),
        "tan" => unary("tan", args, tan),
        _ => Err(MathError::UnknownFunction(name.to_string())),
    }
}

fn binary(
    function: &'static str,
    args: &[f64],
    operation: impl FnOnce(f64, f64) -> MathResult,
) -> MathResult {
    let [left, right] = args else {
        return Err(MathError::Arity {
            function,
            expected: "exactly 2",
            actual: args.len(),
        });
    };
    operation(*left, *right)
}

fn unary(
    function: &'static str,
    args: &[f64],
    operation: impl FnOnce(f64) -> MathResult,
) -> MathResult {
    let [value] = args else {
        return Err(MathError::Arity {
            function,
            expected: "exactly 1",
            actual: args.len(),
        });
    };
    operation(*value)
}

/// Return the ceiling as a signed 64-bit integer.
///
/// Python integers are arbitrary precision.  This low-level shim reports `OverflowError`
/// instead of silently saturating when a result cannot fit in the VM's current integer type.
pub fn ceil(value: f64) -> MathResult {
    if value.is_nan() {
        return Err(MathError::ValueError("cannot convert float NaN to integer"));
    }
    if !value.is_finite() {
        return Err(MathError::OverflowError(
            "cannot convert float infinity to integer",
        ));
    }
    let rounded = value.ceil();
    if !(I64_INCLUSIVE_LOWER..I64_EXCLUSIVE_UPPER).contains(&rounded) {
        return Err(MathError::OverflowError("cannot convert float to integer"));
    }
    Ok(MathValue::Int(rounded as i64))
}

/// Return `e**value`, reporting Python's `OverflowError` for an infinite finite-input result.
pub fn exp(value: f64) -> MathResult {
    let result = value.exp();
    if result.is_infinite() && value.is_finite() {
        return Err(MathError::OverflowError("math range error"));
    }
    if value.is_infinite() && value.is_sign_positive() {
        return Err(MathError::OverflowError("math range error"));
    }
    Ok(MathValue::Float(result))
}

/// Return the inverse sine in radians.
pub fn asin(value: f64) -> MathResult {
    if !(-1.0..=1.0).contains(&value) {
        return Err(MathError::ValueError("math domain error"));
    }
    Ok(MathValue::Float(value.asin()))
}

/// Return the inverse cosine in radians.
pub fn acos(value: f64) -> MathResult {
    if !(-1.0..=1.0).contains(&value) {
        return Err(MathError::ValueError("math domain error"));
    }
    Ok(MathValue::Float(value.acos()))
}

/// Return the base-ten logarithm of a positive value.
pub fn log10(value: f64) -> MathResult {
    if value <= 0.0 {
        return Err(MathError::ValueError("math domain error"));
    }
    Ok(MathValue::Float(value.log10()))
}

/// Return `left**right` with Python's real-number domain and overflow errors.
pub fn pow(left: f64, right: f64) -> MathResult {
    if left == 0.0 && right < 0.0 {
        return Err(MathError::ValueError("math domain error"));
    }
    let value = left.powf(right);
    if value.is_nan() && !(left.is_nan() || right.is_nan()) {
        return Err(MathError::ValueError("math domain error"));
    }
    if value.is_infinite() && left.is_finite() && right.is_finite() {
        return Err(MathError::OverflowError("math range error"));
    }
    Ok(MathValue::Float(value))
}

/// Return whether `value` is positive or negative infinity.
pub const fn isinf(value: f64) -> bool {
    value.is_infinite()
}

/// Return whether `value` is a NaN.
pub const fn isnan(value: f64) -> bool {
    value.is_nan()
}

/// Return the natural logarithm, optionally with a positive base other than one.
pub fn log(value: f64, base: Option<f64>) -> MathResult {
    // Comparisons intentionally leave NaN alone (Python returns NaN for log(NaN)) while rejecting
    // both zero signs as not-positive inputs.
    if value <= 0.0 {
        return Err(MathError::ValueError("expected a positive input"));
    }
    let Some(base) = base else {
        return Ok(MathValue::Float(value.ln()));
    };
    if base == 1.0 {
        return Err(MathError::ZeroDivisionError("float division by zero"));
    }
    if base <= 0.0 {
        return Err(MathError::ValueError("expected a positive input"));
    }
    Ok(MathValue::Float(value.log(base)))
}

/// Return the sine, rejecting infinite inputs as Python does.
pub fn sin(value: f64) -> MathResult {
    if value.is_infinite() {
        return Err(MathError::ValueError("expected a finite input"));
    }
    Ok(MathValue::Float(value.sin()))
}

/// Return the cosine, rejecting infinite inputs as Python does.
pub fn cos(value: f64) -> MathResult {
    if value.is_infinite() {
        return Err(MathError::ValueError("expected a finite input"));
    }
    Ok(MathValue::Float(value.cos()))
}

/// Return the tangent, rejecting infinite inputs as Python does.
pub fn tan(value: f64) -> MathResult {
    if value.is_infinite() {
        return Err(MathError::ValueError("expected a finite input"));
    }
    Ok(MathValue::Float(value.tan()))
}

/// Return the non-negative square root.
pub fn sqrt(value: f64) -> MathResult {
    if value < 0.0 {
        return Err(MathError::ValueError("expected a nonnegative input"));
    }
    Ok(MathValue::Float(value.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::{call, ceil, constant, exp, isinf, isnan, log, sin, sqrt, MathError, MathValue};

    fn float(result: MathValue) -> f64 {
        match result {
            MathValue::Float(value) => value,
            other => panic!("expected float, got {other:?}"),
        }
    }

    #[test]
    fn constants_are_capability_free_and_exactly_named() {
        assert_eq!(float(constant("pi").unwrap()), std::f64::consts::PI);
        assert!(float(constant("inf").unwrap()).is_infinite());
        assert!(float(constant("nan").unwrap()).is_nan());
        assert!(constant("host_path").is_none());
    }

    #[test]
    fn observed_functions_match_basic_values() {
        assert_eq!(ceil(1.01), Ok(MathValue::Int(2)));
        assert!((float(exp(1.0).unwrap()) - std::f64::consts::E).abs() < 1e-15);
        assert!(isinf(f64::INFINITY));
        assert!(isnan(f64::NAN));
        assert!((float(log(std::f64::consts::E, None).unwrap()) - 1.0).abs() < 1e-15);
        assert!((float(log(8.0, Some(2.0)).unwrap()) - 3.0).abs() < 1e-15);
        assert!((float(sin(0.0).unwrap())).abs() < f64::EPSILON);
        assert_eq!(float(sqrt(9.0).unwrap()), 3.0);
    }

    #[test]
    fn dispatcher_checks_names_and_arities() {
        assert_eq!(call("sqrt", &[16.0]), Ok(MathValue::Float(4.0)));
        assert_eq!(
            call("sqrt", &[]),
            Err(MathError::Arity {
                function: "sqrt",
                expected: "exactly 1",
                actual: 0
            })
        );
        assert_eq!(
            call("not_a_function", &[1.0]),
            Err(MathError::UnknownFunction("not_a_function".to_string()))
        );
    }

    #[test]
    fn domain_and_overflow_errors_are_explicit() {
        assert_eq!(
            sqrt(-1.0),
            Err(MathError::ValueError("expected a nonnegative input"))
        );
        assert_eq!(float(sqrt(-0.0).unwrap()), -0.0);
        assert_eq!(
            log(0.0, None),
            Err(MathError::ValueError("expected a positive input"))
        );
        assert_eq!(
            log(-0.0, None),
            Err(MathError::ValueError("expected a positive input"))
        );
        assert_eq!(
            log(2.0, Some(1.0)),
            Err(MathError::ZeroDivisionError("float division by zero"))
        );
        assert_eq!(
            sin(f64::INFINITY),
            Err(MathError::ValueError("expected a finite input"))
        );
        assert_eq!(
            exp(1000.0),
            Err(MathError::OverflowError("math range error"))
        );
        assert_eq!(
            ceil(f64::INFINITY),
            Err(MathError::OverflowError(
                "cannot convert float infinity to integer"
            ))
        );
        assert_eq!(
            ceil(f64::NAN),
            Err(MathError::ValueError("cannot convert float NaN to integer"))
        );
    }
}
