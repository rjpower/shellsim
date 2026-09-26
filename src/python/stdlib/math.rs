//! A small, deterministic subset of Python's :mod:`math` module.
//!
//! This module contains no shellsim or host capabilities.  The VM can translate its `MathValue`
//! and `MathError` values into Python objects/exceptions when it wires the module into imports.
//! Functions intentionally accept `f64`: [`PyNumber`] performs checked coercion from immediate
//! and arbitrary-precision integers without exposing either VM representation here.

use std::fmt;

use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};

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
            name: "atanh",
            call: native_atanh,
        },
        FunctionDef {
            module: "math",
            name: "acosh",
            call: native_acosh,
        },
        FunctionDef {
            module: "math",
            name: "asinh",
            call: native_asinh,
        },
        FunctionDef {
            module: "math",
            name: "tanh",
            call: native_tanh,
        },
        FunctionDef {
            module: "math",
            name: "cosh",
            call: native_cosh,
        },
        FunctionDef {
            module: "math",
            name: "sinh",
            call: native_sinh,
        },
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
            name: "comb",
            call: native_comb,
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
            name: "fabs",
            call: native_fabs,
        },
        FunctionDef {
            module: "math",
            name: "factorial",
            call: native_factorial,
        },
        FunctionDef {
            module: "math",
            name: "floor",
            call: native_floor,
        },
        FunctionDef {
            module: "math",
            name: "gcd",
            call: native_gcd,
        },
        FunctionDef {
            module: "math",
            name: "hypot",
            call: native_hypot,
        },
        FunctionDef {
            module: "math",
            name: "isfinite",
            call: native_isfinite,
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
            name: "lcm",
            call: native_lcm,
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
            name: "log2",
            call: native_log2,
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
            name: "trunc",
            call: native_trunc,
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

fn native_sinh(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "sinh")
}

fn native_cosh(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "cosh")
}

fn native_tanh(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "tanh")
}

fn native_asinh(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "asinh")
}

fn native_acosh(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "acosh")
}

fn native_atanh(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "atanh")
}

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

/// Compute exact combinations with work and result storage bounded before multiplication.
fn native_comb(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("math.comb", 2, 2)?;
    args.reject_keywords("math.comb")?;
    let n = integer_argument(runtime, &args.positional()[0], "comb")?;
    let k = integer_argument(runtime, &args.positional()[1], "comb")?;
    if n.is_negative() {
        return Err(PyError::value_error("n must be a non-negative integer"));
    }
    if k.is_negative() {
        return Err(PyError::value_error("k must be a non-negative integer"));
    }
    if k > n {
        return runtime.new_integer("0");
    }
    let complement = &n - &k;
    let selected = if k <= complement { k } else { complement };
    let count = selected
        .to_usize()
        .ok_or_else(|| PyError::resource_error("combination length is too large"))?;
    if count > 100_000 {
        return Err(PyError::resource_error("combination length is too large"));
    }
    if count == 0 {
        return runtime.new_integer("1");
    }
    let bytes = bigint_bytes(&n)?
        .checked_mul(count.saturating_add(1))
        .ok_or_else(|| PyError::resource_error("combination result is too large"))?;
    runtime.reserve_memory(bytes)?;
    let mut result = BigInt::from(1_u8);
    for index in 1..=count {
        runtime.charge_cpu(result.bits().div_ceil(64).saturating_add(1))?;
        result = result * (&n - &selected + index) / index;
    }
    runtime.new_integer(&result.to_string())
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

fn native_fabs(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "fabs")
}

fn native_factorial(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("math.factorial", 1, 1)?;
    args.reject_keywords("math.factorial")?;
    let value = integer_argument(runtime, &args.positional()[0], "factorial")?;
    if value.is_negative() {
        return Err(PyError::value_error(
            "factorial() not defined for negative values",
        ));
    }
    let value = value
        .to_u64()
        .ok_or_else(|| PyError::resource_error("factorial argument is too large"))?;
    if value > 100_000 {
        return Err(PyError::resource_error("factorial argument is too large"));
    }
    let result_bound = usize::try_from(value)
        .unwrap_or(usize::MAX)
        .checked_mul(value.to_string().len())
        .ok_or_else(|| PyError::resource_error("factorial result is too large"))?;
    runtime.reserve_memory(result_bound)?;
    let mut result = BigInt::from(1_u8);
    for factor in 2..=value {
        runtime.charge_cpu(1)?;
        result *= factor;
    }
    runtime.new_integer(&result.to_string())
}

fn native_floor(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_round_direction(runtime, args, true)
}

fn native_trunc(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_round_direction(runtime, args, false)
}

fn native_round_direction(runtime: &mut dyn PyRuntime, args: CallArgs, floor: bool) -> PyResult {
    args.expect_positional("math integer conversion", 1, 1)?;
    args.reject_keywords("math integer conversion")?;
    if let Some(integer) = runtime.integer_text(&args.positional()[0])? {
        return runtime.new_integer(&integer);
    }
    let value = args.positional()[0].cast::<PyNumber>(runtime)?.into_f64()?;
    if value.is_nan() {
        return Err(PyError::value_error("cannot convert float NaN to integer"));
    }
    if value.is_infinite() {
        return Err(PyError::overflow_error(
            "cannot convert float infinity to integer",
        ));
    }
    let value = if floor { value.floor() } else { value.trunc() };
    runtime.new_integer(&format!("{value:.0}"))
}

fn native_gcd(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    integer_fold(runtime, args, false)
}

fn native_lcm(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    integer_fold(runtime, args, true)
}

fn integer_fold(runtime: &mut dyn PyRuntime, args: CallArgs, lcm: bool) -> PyResult {
    args.reject_keywords(if lcm { "math.lcm" } else { "math.gcd" })?;
    let mut result = if lcm {
        BigInt::from(1_u8)
    } else {
        BigInt::zero()
    };
    for value in args.positional() {
        runtime.charge_cpu(1)?;
        let value = integer_argument(runtime, value, if lcm { "lcm" } else { "gcd" })?.abs();
        if lcm {
            if result.is_zero() || value.is_zero() {
                result = BigInt::zero();
            } else {
                let divisor = bigint_gcd(runtime, result.clone(), value.clone())?;
                let product_bytes = bigint_bytes(&result)?
                    .checked_add(bigint_bytes(&value)?)
                    .ok_or_else(|| PyError::resource_error("lcm result is too large"))?;
                runtime.reserve_memory(product_bytes)?;
                result = (result / divisor) * value;
            }
        } else {
            result = bigint_gcd(runtime, result, value)?;
        }
    }
    runtime.new_integer(&result.to_string())
}

fn integer_argument(runtime: &dyn PyRuntime, value: &Value, name: &str) -> PyResult<BigInt> {
    runtime
        .integer_text(value)?
        .ok_or_else(|| PyError::type_error(format!("{name}() only accepts integral values")))?
        .parse::<BigInt>()
        .map_err(|_| PyError::runtime_error("invalid internal integer representation"))
}

fn bigint_gcd(
    runtime: &mut dyn PyRuntime,
    mut left: BigInt,
    mut right: BigInt,
) -> PyResult<BigInt> {
    runtime.reserve_memory(bigint_bytes(&left)?.max(bigint_bytes(&right)?))?;
    while !right.is_zero() {
        runtime.charge_cpu(1)?;
        let remainder = left % &right;
        left = right;
        right = remainder;
    }
    Ok(left.abs())
}

fn bigint_bytes(value: &BigInt) -> PyResult<usize> {
    usize::try_from(value.bits().div_ceil(8).max(1))
        .map_err(|_| PyError::resource_error("integer result is too large"))
}

fn native_hypot(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "hypot")
}

fn native_isinf(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "isinf")
}

fn native_isfinite(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "isfinite")
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

fn native_log2(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_call(runtime, args, "log2")
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
        "sinh" => unary("sinh", args, sinh),
        "cosh" => unary("cosh", args, cosh),
        "tanh" => unary("tanh", args, tanh),
        "asinh" => unary("asinh", args, asinh),
        "acosh" => unary("acosh", args, acosh),
        "atanh" => unary("atanh", args, atanh),
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
        "fabs" => unary("fabs", args, |value| Ok(MathValue::Float(value.abs()))),
        "hypot" => Ok(MathValue::Float(
            args.iter().copied().fold(0.0_f64, f64::hypot),
        )),
        "isfinite" => unary("isfinite", args, |value| {
            Ok(MathValue::Bool(value.is_finite()))
        }),
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
        "log2" => unary("log2", args, log2),
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

/// Hyperbolic functions preserve IEEE infinities and NaNs; finite overflow is a Python error.
fn hyperbolic_result(input: f64, result: f64) -> MathResult {
    if input.is_finite() && result.is_infinite() {
        return Err(MathError::OverflowError("math range error"));
    }
    Ok(MathValue::Float(result))
}
fn sinh(value: f64) -> MathResult {
    hyperbolic_result(value, value.sinh())
}
fn cosh(value: f64) -> MathResult {
    hyperbolic_result(value, value.cosh())
}
fn tanh(value: f64) -> MathResult {
    Ok(MathValue::Float(value.tanh()))
}
fn asinh(value: f64) -> MathResult {
    Ok(MathValue::Float(value.asinh()))
}
/// The real inverse hyperbolic cosine has domain [1, infinity); NaNs propagate.
fn acosh(value: f64) -> MathResult {
    if value < 1.0 {
        return Err(MathError::ValueError("math domain error"));
    }
    Ok(MathValue::Float(value.acosh()))
}
/// The real inverse hyperbolic tangent has open domain (-1, 1); NaNs propagate.
fn atanh(value: f64) -> MathResult {
    if value.abs() >= 1.0 {
        return Err(MathError::ValueError("math domain error"));
    }
    Ok(MathValue::Float(value.atanh()))
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

/// Return the base-two logarithm of a positive value.
pub fn log2(value: f64) -> MathResult {
    if value <= 0.0 {
        return Err(MathError::ValueError("math domain error"));
    }
    Ok(MathValue::Float(value.log2()))
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
    fn hyperbolic_domains_and_finite_overflow_match_python() {
        for name in ["sinh", "cosh", "tanh", "asinh", "acosh", "atanh"] {
            assert!(float(call(name, &[f64::NAN]).unwrap()).is_nan());
        }
        assert_eq!(
            call("acosh", &[0.0]),
            Err(MathError::ValueError("math domain error"))
        );
        assert_eq!(
            call("atanh", &[1.0]),
            Err(MathError::ValueError("math domain error"))
        );
        assert_eq!(
            call("sinh", &[1000.0]),
            Err(MathError::OverflowError("math range error"))
        );
        assert_eq!(
            float(call("sinh", &[f64::INFINITY]).unwrap()),
            f64::INFINITY
        );
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
        assert_eq!(call("log2", &[8.0]), Ok(MathValue::Float(3.0)));
        assert_eq!(call("log10", &[100.0]), Ok(MathValue::Float(2.0)));
        assert_eq!(
            call("log2", &[]),
            Err(MathError::Arity {
                function: "log2",
                expected: "exactly 1",
                actual: 0
            })
        );
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
        for name in ["log2", "log10"] {
            for value in [0.0, -0.0, -1.0] {
                assert_eq!(
                    call(name, &[value]),
                    Err(MathError::ValueError("math domain error"))
                );
            }
            assert_eq!(
                call(name, &[f64::INFINITY]),
                Ok(MathValue::Float(f64::INFINITY))
            );
            assert!(float(call(name, &[f64::NAN]).unwrap()).is_nan());
        }
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
