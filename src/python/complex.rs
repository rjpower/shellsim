//! Native implementation of Python's builtin `complex` type.
//!
//! A complex number is an immutable pair of IEEE-754 doubles stored in the object arena (see
//! `Object::Complex`), so allocation is metered like any other heap value. Arithmetic, string
//! parsing, `repr`, and `hash` follow CPython 3.14's `complexobject.c`, including its mixed-mode
//! rules for real operands and its recovery of infinities in products and quotients. Every
//! operation here is a constant amount of floating-point work.
//!
//! Operations that are undefined for complex numbers (ordering, floor division, modulo, and the
//! real-only conversions elsewhere in the runtime) raise `TypeError` explicitly. Format
//! specifications are not implemented for complex values.

use num_traits::ToPrimitive;

use super::native::{
    CallArgs, GetterDef, MethodDef, NativeTypeDef, PyError, PyResult, PyRuntime, PyValue,
};
use super::number::NumberRef;

pub(super) static COMPLEX_TYPE: NativeTypeDef = NativeTypeDef {
    name: "complex",
    methods: &[MethodDef {
        type_name: "complex",
        name: "conjugate",
        call: conjugate,
    }],
    getters: &[
        GetterDef {
            owner: "complex",
            name: "real",
            get: real_part,
        },
        GetterDef {
            owner: "complex",
            name: "imag",
            get: imaginary_part,
        },
    ],
};

/// The value of a complex number. `PartialEq` compares components with IEEE semantics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    pub(super) const fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }
}

/// A numeric operand after Python's implicit conversion for complex arithmetic.
///
/// CPython 3.14 keeps real operands real in mixed-mode `+`, `-`, `*`, and `/`, which preserves
/// signed zeros and avoids spurious NaNs from multiplying an infinite real part by a zero
/// imaginary part.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Operand {
    Real(f64),
    Complex(Complex),
}

impl Operand {
    fn widen(self) -> Complex {
        match self {
            Self::Real(value) => Complex::new(value, 0.0),
            Self::Complex(value) => value,
        }
    }
}

/// Read one builtin number as a complex operand, or `None` for a non-numeric value.
fn operand(runtime: &dyn PyRuntime, value: &PyValue) -> PyResult<Option<Operand>> {
    Ok(match runtime.number(value) {
        Some(NumberRef::Int(value)) => Some(Operand::Real(value as f64)),
        Some(NumberRef::BigInt(value)) => Some(Operand::Real(bigint_to_f64(value)?)),
        Some(NumberRef::Float(value)) => Some(Operand::Real(value)),
        Some(NumberRef::Complex(real, imag)) => Some(Operand::Complex(Complex::new(real, imag))),
        None => None,
    })
}

fn bigint_to_f64(value: &num_bigint::BigInt) -> PyResult<f64> {
    value
        .to_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| PyError::overflow_error("int too large to convert to float"))
}

/// Read the receiver of a `complex` method, slot, or getter.
fn receiver(runtime: &dyn PyRuntime, value: &PyValue) -> PyResult<Complex> {
    match runtime.number(value) {
        Some(NumberRef::Complex(real, imag)) => Ok(Complex::new(real, imag)),
        _ => Err(PyError::type_error(
            "descriptor requires a 'complex' object",
        )),
    }
}

fn real_part(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(PyValue::Float(receiver(runtime, &value)?.real))
}

fn imaginary_part(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(PyValue::Float(receiver(runtime, &value)?.imag))
}

fn conjugate(runtime: &mut dyn PyRuntime, value: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("complex.conjugate", 0, 0)?;
    args.reject_keywords("complex.conjugate")?;
    let value = receiver(runtime, &value)?;
    runtime.new_complex(value.real, -value.imag)
}

/// Construct `complex(real=0, imag=0)`.
///
/// Accepts a string in CPython's literal-like syntax (`"1+2j"`, `"(-j)"`, `"inf+nanj"`), or up
/// to two numbers. Complex arguments are combined as `real + imag * 1j`, which CPython still
/// accepts with a deprecation warning.
pub(super) fn construct(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("complex", 0, 2)?;
    args.reject_unknown_keywords("complex", &["real", "imag"])?;
    let mut real = args.positional().first().copied();
    let mut imag = args.positional().get(1).copied();
    for (name, slot) in [("real", &mut real), ("imag", &mut imag)] {
        if let Some(value) = args.keyword("complex", name)? {
            if slot.is_some() {
                return Err(PyError::type_error(format!(
                    "argument for complex() given by name ('{name}') and position"
                )));
            }
            *slot = Some(*value);
        }
    }
    if let Some(text) = real
        .map(|value| runtime.string_value(&value))
        .transpose()?
        .flatten()
    {
        if imag.is_some() {
            return Err(PyError::type_error(
                "complex() argument 'real' must be a real number, not str",
            ));
        }
        runtime.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
        let value = parse(&text)?;
        return runtime.new_complex(value.real, value.imag);
    }
    let real = match real {
        Some(value) => match operand(runtime, &value)? {
            Some(Operand::Complex(_)) if imag.is_none() => return Ok(value),
            Some(operand) => operand,
            None => {
                let actual = runtime.type_name(&value)?;
                return Err(PyError::type_error(format!(
                    "complex() argument must be a string or a number, not {actual}"
                )));
            }
        },
        None => Operand::Real(0.0),
    };
    let imag = match imag {
        Some(value) => match operand(runtime, &value)? {
            Some(operand) => Some(operand),
            None => {
                let actual = runtime.type_name(&value)?;
                return Err(PyError::type_error(format!(
                    "complex() argument 'imag' must be a real number, not {actual}"
                )));
            }
        },
        None => None,
    };
    // Combine as CPython does, touching only components that exist so signed zeros survive.
    let mut result = Complex::new(real.widen().real, 0.0);
    match imag {
        Some(Operand::Real(imag)) => result.imag = imag,
        Some(Operand::Complex(imag)) => {
            result.real -= imag.imag;
            result.imag = imag.real;
        }
        None => {}
    }
    if let (Operand::Complex(real), Some(_)) = (real, imag) {
        result.imag += real.imag;
    }
    runtime.new_complex(result.real, result.imag)
}

/// Parse the string forms accepted by `complex()`.
///
/// The grammar is CPython's: optional surrounding whitespace and one pair of parentheses around
/// `<float>`, `<float>j`, `<float><signed-float>j`, `<float><sign>j`, or `[<sign>]j`, with
/// underscores allowed only between digits.
pub(super) fn parse(text: &str) -> PyResult<Complex> {
    let malformed = || PyError::value_error("complex() arg is a malformed string");
    let cleaned = strip_digit_underscores(text).ok_or_else(|| {
        PyError::value_error(format!("could not convert string to complex: {text:?}"))
    })?;
    let mut rest = cleaned.trim_matches(is_python_space);
    let parenthesized = rest.starts_with('(');
    if parenthesized {
        rest = rest
            .strip_prefix('(')
            .and_then(|inner| inner.strip_suffix(')'))
            .ok_or_else(malformed)?
            .trim_matches(is_python_space);
    }
    let bytes = rest.as_bytes();
    let (value, end) = match scan_float(bytes, 0) {
        Some((first, after_first)) => match bytes.get(after_first) {
            Some(b'+' | b'-') => {
                let (imag, after_imag) = match scan_float(bytes, after_first) {
                    Some(parsed) => parsed,
                    None => (sign_unit(bytes[after_first]), after_first + 1),
                };
                if !matches!(bytes.get(after_imag), Some(b'j' | b'J')) {
                    return Err(malformed());
                }
                (Complex::new(first, imag), after_imag + 1)
            }
            Some(b'j' | b'J') => (Complex::new(0.0, first), after_first + 1),
            _ => (Complex::new(first, 0.0), after_first),
        },
        None => {
            let (imag, position) = match bytes.first() {
                Some(sign @ (b'+' | b'-')) => (sign_unit(*sign), 1),
                _ => (1.0, 0),
            };
            if !matches!(bytes.get(position), Some(b'j' | b'J')) {
                return Err(malformed());
            }
            (Complex::new(0.0, imag), position + 1)
        }
    };
    if end != bytes.len() || rest.is_empty() {
        return Err(malformed());
    }
    Ok(value)
}

fn is_python_space(character: char) -> bool {
    character.is_whitespace()
}

fn sign_unit(sign: u8) -> f64 {
    if sign == b'-' {
        -1.0
    } else {
        1.0
    }
}

/// Remove underscores that separate two digits; reject any other underscore.
fn strip_digit_underscores(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut cleaned = String::with_capacity(text.len());
    for (index, character) in text.char_indices() {
        if character != '_' {
            cleaned.push(character);
            continue;
        }
        let before = index
            .checked_sub(1)
            .and_then(|previous| bytes.get(previous));
        let after = bytes.get(index + 1);
        if !before.is_some_and(u8::is_ascii_digit) || !after.is_some_and(u8::is_ascii_digit) {
            return None;
        }
    }
    Some(cleaned)
}

/// Scan the longest float literal starting at `start`, as `PyOS_string_to_double` does.
///
/// Returns the value and the index after the literal. Overflowing literals become infinities.
fn scan_float(bytes: &[u8], start: usize) -> Option<(f64, usize)> {
    let mut position = start;
    if matches!(bytes.get(position), Some(b'+' | b'-')) {
        position += 1;
    }
    let body = &bytes[position..];
    for word in ["infinity", "inf", "nan"] {
        if body.len() >= word.len() && body[..word.len()].eq_ignore_ascii_case(word.as_bytes()) {
            let end = position + word.len();
            let text = std::str::from_utf8(&bytes[start..end]).ok()?;
            return Some((text.parse().ok()?, end));
        }
    }
    let digits = |from: usize| {
        bytes[from..]
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count()
    };
    let integer = digits(position);
    position += integer;
    let mut fraction = 0;
    if bytes.get(position) == Some(&b'.') {
        fraction = digits(position + 1);
        if integer == 0 && fraction == 0 {
            return None;
        }
        position += 1 + fraction;
    }
    if integer == 0 && fraction == 0 {
        return None;
    }
    if matches!(bytes.get(position), Some(b'e' | b'E')) {
        let mut exponent = position + 1;
        if matches!(bytes.get(exponent), Some(b'+' | b'-')) {
            exponent += 1;
        }
        let exponent_digits = digits(exponent);
        if exponent_digits > 0 {
            position = exponent + exponent_digits;
        }
    }
    let text = std::str::from_utf8(&bytes[start..position]).ok()?;
    Some((text.parse().ok()?, position))
}

/// Render `repr(complex)`: `(1+2j)`, `2j`, `(-0-1j)`, `(nan+infj)`.
///
/// A real part of positive zero is omitted along with the parentheses, as in CPython.
pub(super) fn repr(value: Complex) -> String {
    if value.real == 0.0 && value.real.is_sign_positive() {
        return format!("{}j", repr_component(value.imag, false));
    }
    format!(
        "({}{}j)",
        repr_component(value.real, false),
        repr_component(value.imag, true)
    )
}

/// Format one component with CPython's shortest round-trip `repr` digits.
///
/// Unlike `float.__repr__`, integral components carry no `.0`. `signed` forces a leading `+`
/// for non-negative values and NaN, matching `Py_DTSF_SIGN`.
pub(super) fn repr_component(value: f64, signed: bool) -> String {
    let sign = if value.is_sign_negative() && !value.is_nan() {
        "-"
    } else if signed {
        "+"
    } else {
        ""
    };
    if value.is_nan() {
        return format!("{sign}nan");
    }
    if value.is_infinite() {
        return format!("{sign}inf");
    }
    // `{:e}` yields the shortest round-trip digits as `d[.ddd]e<exponent>`.
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("scientific formatting has an exponent");
    let exponent: i32 = exponent.parse().expect("scientific exponent is an integer");
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let body = if !(-4..16).contains(&exponent) {
        let (first, remainder) = digits.split_at(1);
        let fraction = if remainder.is_empty() {
            String::new()
        } else {
            format!(".{remainder}")
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!("{first}{fraction}e{exponent_sign}{:02}", exponent.abs())
    } else if exponent < 0 {
        let zeros = "0".repeat(usize::try_from(-exponent - 1).expect("negative exponent"));
        format!("0.{zeros}{digits}")
    } else {
        let point = usize::try_from(exponent + 1).expect("non-negative exponent");
        if digits.len() <= point {
            format!("{digits}{}", "0".repeat(point - digits.len()))
        } else {
            format!("{}.{}", &digits[..point], &digits[point..])
        }
    };
    format!("{sign}{body}")
}

const HASH_BITS: u32 = 61;
const HASH_MODULUS: u64 = (1 << HASH_BITS) - 1;
const HASH_INF: i64 = 314_159;
const HASH_IMAG: u64 = 1_000_003;

/// CPython's `hash(complex)`: `hash(real) + 1000003 * hash(imag)` with wrapping arithmetic.
///
/// CPython hashes NaN by object identity; this runtime has no `hash()` builtin that could
/// observe that, so NaN components hash as `0`, CPython's pre-3.10 behavior.
pub(super) fn hash(value: Complex) -> i64 {
    let real = hash_float(value.real) as u64;
    let imag = hash_float(value.imag) as u64;
    let combined = real.wrapping_add(HASH_IMAG.wrapping_mul(imag)) as i64;
    if combined == -1 {
        -2
    } else {
        combined
    }
}

/// CPython's `_Py_HashDouble`: reduce a finite double modulo the Mersenne prime `2**61 - 1`.
fn hash_float(value: f64) -> i64 {
    if value.is_nan() {
        return 0;
    }
    if value.is_infinite() {
        return if value > 0.0 { HASH_INF } else { -HASH_INF };
    }
    let (mut mantissa, mut exponent) = frexp(value);
    let negative = mantissa < 0.0;
    mantissa = mantissa.abs();
    let mut hashed = 0u64;
    while mantissa != 0.0 {
        hashed = ((hashed << 28) & HASH_MODULUS) | hashed >> (HASH_BITS - 28);
        mantissa *= 268_435_456.0;
        exponent -= 28;
        let integer = mantissa as u64;
        mantissa -= integer as f64;
        hashed += integer;
        if hashed >= HASH_MODULUS {
            hashed -= HASH_MODULUS;
        }
    }
    let bits = HASH_BITS as i32;
    let exponent = if exponent >= 0 {
        exponent % bits
    } else {
        bits - 1 - ((-1 - exponent) % bits)
    } as u32;
    hashed = ((hashed << exponent) & HASH_MODULUS) | hashed >> (HASH_BITS - exponent);
    let hashed = if negative {
        (hashed as i64).wrapping_neg()
    } else {
        hashed as i64
    };
    if hashed == -1 {
        -2
    } else {
        hashed
    }
}

/// Split a finite, nonzero-or-zero double into a mantissa in `[0.5, 1)` and a power of two.
fn frexp(value: f64) -> (f64, i32) {
    if value == 0.0 {
        return (value, 0);
    }
    let bits = value.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: scale into the normal range first.
        let (mantissa, exponent) = frexp(value * 2f64.powi(54));
        return (mantissa, exponent - 54);
    }
    let mantissa = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
    (mantissa, biased - 1022)
}

/// CPython's `_Py_c_prod`, including C11 Annex G recovery of infinities.
fn multiply(left: Complex, right: Complex) -> Complex {
    let (mut a, mut b, mut c, mut d) = (left.real, left.imag, right.real, right.imag);
    let (ac, bd, ad, bc) = (a * c, b * d, a * d, b * c);
    let result = Complex::new(ac - bd, ad + bc);
    if !(result.real.is_nan() && result.imag.is_nan()) {
        return result;
    }
    let boxed = |value: f64| (if value.is_infinite() { 1.0f64 } else { 0.0 }).copysign(value);
    let zero_nan = |value: f64| {
        if value.is_nan() {
            0.0f64.copysign(value)
        } else {
            value
        }
    };
    let mut recalculate = false;
    if a.is_infinite() || b.is_infinite() {
        (a, b, c, d) = (boxed(a), boxed(b), zero_nan(c), zero_nan(d));
        recalculate = true;
    }
    if c.is_infinite() || d.is_infinite() {
        (a, b, c, d) = (zero_nan(a), zero_nan(b), boxed(c), boxed(d));
        recalculate = true;
    }
    if !recalculate && [ac, bd, ad, bc].iter().any(|value| value.is_infinite()) {
        (a, b, c, d) = (zero_nan(a), zero_nan(b), zero_nan(c), zero_nan(d));
        recalculate = true;
    }
    if recalculate {
        Complex::new(
            f64::INFINITY * (a * c - b * d),
            f64::INFINITY * (a * d + b * c),
        )
    } else {
        result
    }
}

/// CPython's `_Py_c_quot`. Division by zero is reported as `None`.
fn divide(left: Complex, right: Complex) -> Option<Complex> {
    let abs_real = right.real.abs();
    let abs_imag = right.imag.abs();
    let mut result = if abs_real >= abs_imag {
        if abs_real == 0.0 {
            return None;
        }
        let ratio = right.imag / right.real;
        let denominator = right.real + right.imag * ratio;
        Complex::new(
            (left.real + left.imag * ratio) / denominator,
            (left.imag - left.real * ratio) / denominator,
        )
    } else if abs_imag >= abs_real {
        let ratio = right.real / right.imag;
        let denominator = right.real * ratio + right.imag;
        Complex::new(
            (left.real * ratio + left.imag) / denominator,
            (left.imag * ratio - left.real) / denominator,
        )
    } else {
        Complex::new(f64::NAN, f64::NAN)
    };
    if result.real.is_nan() && result.imag.is_nan() {
        let unit = |value: f64| (if value.is_infinite() { 1.0f64 } else { 0.0 }).copysign(value);
        if (left.real.is_infinite() || left.imag.is_infinite())
            && right.real.is_finite()
            && right.imag.is_finite()
        {
            let (x, y) = (unit(left.real), unit(left.imag));
            result = Complex::new(
                f64::INFINITY * (x * right.real + y * right.imag),
                f64::INFINITY * (y * right.real - x * right.imag),
            );
        } else if (abs_real.is_infinite() || abs_imag.is_infinite())
            && left.real.is_finite()
            && left.imag.is_finite()
        {
            let (x, y) = (unit(right.real), unit(right.imag));
            result = Complex::new(
                0.0 * (left.real * x + left.imag * y),
                0.0 * (left.imag * x - left.real * y),
            );
        }
    }
    Some(result)
}

/// Failure modes of complex exponentiation, mirroring CPython's `errno` checks.
#[derive(Debug, PartialEq)]
enum PowerError {
    ZeroToNegativeOrComplex,
    Overflow,
}

/// CPython's `complex.__pow__`: exact repeated squaring for small integral exponents and the
/// polar formula otherwise. Any infinite component in the result is an overflow.
fn power(base: Complex, exponent: Complex) -> Result<Complex, PowerError> {
    let result = if exponent.imag == 0.0
        && exponent.real == exponent.real.floor()
        && exponent.real.abs() <= 100.0
    {
        integer_power(base, exponent.real as i64)?
    } else {
        polar_power(base, exponent)?
    };
    if result.real.is_infinite() || result.imag.is_infinite() {
        return Err(PowerError::Overflow);
    }
    Ok(result)
}

fn integer_power(base: Complex, exponent: i64) -> Result<Complex, PowerError> {
    let one = Complex::new(1.0, 0.0);
    let unsigned_power = |mut remaining: u64| {
        let mut result = one;
        let mut factor = base;
        while remaining > 0 {
            if remaining & 1 == 1 {
                result = multiply(result, factor);
            }
            remaining >>= 1;
            factor = multiply(factor, factor);
        }
        result
    };
    if exponent > 0 {
        return Ok(unsigned_power(exponent.unsigned_abs()));
    }
    divide(one, unsigned_power(exponent.unsigned_abs())).ok_or(PowerError::ZeroToNegativeOrComplex)
}

fn polar_power(base: Complex, exponent: Complex) -> Result<Complex, PowerError> {
    if exponent.real == 0.0 && exponent.imag == 0.0 {
        return Ok(Complex::new(1.0, 0.0));
    }
    if base.real == 0.0 && base.imag == 0.0 {
        if exponent.imag != 0.0 || exponent.real < 0.0 {
            return Err(PowerError::ZeroToNegativeOrComplex);
        }
        return Ok(Complex::new(0.0, 0.0));
    }
    let magnitude = base.real.hypot(base.imag);
    let mut length = magnitude.powf(exponent.real);
    let angle = base.imag.atan2(base.real);
    let mut phase = angle * exponent.real;
    if exponent.imag != 0.0 {
        length /= (angle * exponent.imag).exp();
        phase += exponent.imag * magnitude.ln();
    }
    Ok(Complex::new(length * phase.cos(), length * phase.sin()))
}

/// Binary operations with a complex slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
}

/// Evaluate `left <operation> right` where at least one operand is complex.
fn evaluate(operation: Operation, left: Operand, right: Operand) -> PyResult<Complex> {
    use Operand::{Complex as C, Real as R};
    Ok(match (operation, left, right) {
        (Operation::Add, C(left), R(right)) | (Operation::Add, R(right), C(left)) => {
            Complex::new(left.real + right, left.imag)
        }
        (Operation::Subtract, C(left), R(right)) => Complex::new(left.real - right, left.imag),
        (Operation::Subtract, R(left), C(right)) => Complex::new(left - right.real, -right.imag),
        (Operation::Multiply, C(left), R(right)) | (Operation::Multiply, R(right), C(left)) => {
            Complex::new(left.real * right, left.imag * right)
        }
        (Operation::Divide, C(left), R(right)) => {
            if right == 0.0 {
                return Err(PyError::zero_division_error("division by zero"));
            }
            Complex::new(left.real / right, left.imag / right)
        }
        (Operation::Add, left, right) => {
            let (left, right) = (left.widen(), right.widen());
            Complex::new(left.real + right.real, left.imag + right.imag)
        }
        (Operation::Subtract, left, right) => {
            let (left, right) = (left.widen(), right.widen());
            Complex::new(left.real - right.real, left.imag - right.imag)
        }
        (Operation::Multiply, left, right) => multiply(left.widen(), right.widen()),
        (Operation::Divide, left, right) => divide(left.widen(), right.widen())
            .ok_or_else(|| PyError::zero_division_error("division by zero"))?,
        (Operation::Power, left, right) => {
            power(left.widen(), right.widen()).map_err(|error| match error {
                PowerError::ZeroToNegativeOrComplex => {
                    PyError::zero_division_error("zero to a negative or complex power")
                }
                PowerError::Overflow => PyError::overflow_error("complex exponentiation"),
            })?
        }
    })
}

fn binary(
    runtime: &mut dyn PyRuntime,
    operation: Operation,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let (Some(left), Some(right)) = (operand(runtime, &left)?, operand(runtime, &right)?) else {
        return Ok(None);
    };
    let result = evaluate(operation, left, right)?;
    runtime.new_complex(result.real, result.imag).map(Some)
}

pub(super) fn slot_add(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    binary(runtime, Operation::Add, left, right)
}

pub(super) fn slot_subtract(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    binary(runtime, Operation::Subtract, left, right)
}

pub(super) fn slot_reflected_subtract(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    binary(runtime, Operation::Subtract, left, right)
}

pub(super) fn slot_multiply(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    binary(runtime, Operation::Multiply, left, right)
}

pub(super) fn slot_divide(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    binary(runtime, Operation::Divide, left, right)
}

pub(super) fn slot_reflected_divide(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    binary(runtime, Operation::Divide, left, right)
}

pub(super) fn slot_power(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    binary(runtime, Operation::Power, left, right)
}

pub(super) fn slot_reflected_power(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    binary(runtime, Operation::Power, left, right)
}

/// Raise CPython's `TypeError` for an operator that complex numbers do not define.
fn unsupported(
    runtime: &mut dyn PyRuntime,
    symbol: &str,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let left = runtime.type_name(&left)?;
    let right = runtime.type_name(&right)?;
    Err(PyError::type_error(format!(
        "unsupported operand type(s) for {symbol}: '{left}' and '{right}'"
    )))
}

pub(super) fn slot_floor_divide(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    unsupported(runtime, "//", left, right)
}

pub(super) fn slot_reflected_floor_divide(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    unsupported(runtime, "//", left, right)
}

pub(super) fn slot_remainder(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    unsupported(runtime, "%", left, right)
}

pub(super) fn slot_reflected_remainder(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    unsupported(runtime, "%", left, right)
}

/// Raise CPython's `TypeError` for ordering involving a complex number.
fn unordered(
    runtime: &mut dyn PyRuntime,
    symbol: &str,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let left = runtime.type_name(&left)?;
    let right = runtime.type_name(&right)?;
    Err(PyError::type_error(format!(
        "'{symbol}' not supported between instances of '{left}' and '{right}'"
    )))
}

pub(super) fn slot_less_than(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    unordered(runtime, "<", left, right)
}

pub(super) fn slot_less_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    unordered(runtime, "<=", left, right)
}

pub(super) fn slot_greater_than(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    unordered(runtime, ">", left, right)
}

pub(super) fn slot_greater_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    unordered(runtime, ">=", left, right)
}

pub(super) fn slot_positive(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    receiver(runtime, &value)?;
    Ok(Some(value))
}

pub(super) fn slot_negative(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let value = receiver(runtime, &value)?;
    runtime.new_complex(-value.real, -value.imag).map(Some)
}

pub(super) fn slot_absolute(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let value = receiver(runtime, &value)?;
    let magnitude = value.real.hypot(value.imag);
    if magnitude.is_infinite() && value.real.is_finite() && value.imag.is_finite() {
        return Err(PyError::overflow_error("absolute value too large"));
    }
    Ok(Some(PyValue::Float(magnitude)))
}

pub(super) fn slot_hash(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Option<PyValue>> {
    Ok(Some(PyValue::Int(hash(receiver(runtime, &value)?))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_matches_cpython_component_formatting() {
        let nan = f64::NAN;
        let inf = f64::INFINITY;
        for (real, imag, expected) in [
            (1.0, 2.0, "(1+2j)"),
            (0.0, 2.0, "2j"),
            (-0.0, -1.0, "(-0-1j)"),
            (0.0, -0.0, "-0j"),
            (-0.0, 0.0, "(-0+0j)"),
            (nan, inf, "(nan+infj)"),
            (1.0, -nan, "(1+nanj)"),
            (1e16, 1e-7, "(1e+16+1e-07j)"),
            (123456789012345678.0, 1.0, "(1.2345678901234568e+17+1j)"),
            (1e-5, 0.0001, "(1e-05+0.0001j)"),
            (0.1, 1.0 / 3.0, "(0.1+0.3333333333333333j)"),
            (1.5, -2000.0, "(1.5-2000j)"),
        ] {
            assert_eq!(repr(Complex::new(real, imag)), expected);
        }
    }

    #[test]
    fn hash_matches_cpython() {
        assert_eq!(hash(Complex::new(1.0, 2.0)), 2_000_007);
        assert_eq!(hash(Complex::new(0.0, 2.0)), 2_000_006);
        assert_eq!(hash(Complex::new(0.0, 1.0)), 1_000_003);
        assert_eq!(hash(Complex::new(1.5, -2.25)), -576_460_752_305_423_493);
        assert_eq!(hash_float(-1.0), -2);
        assert_eq!(hash_float(f64::INFINITY), 314_159);
        assert_eq!(hash_float(0.5), 1 << 60);
    }

    #[test]
    fn parser_accepts_cpython_forms_and_rejects_malformed_text() {
        for (text, real, imag) in [
            ("1+2j", 1.0, 2.0),
            (" ( 1.5-2e3J ) ", 1.5, -2000.0),
            ("j", 0.0, 1.0),
            ("-j", 0.0, -1.0),
            ("+J", 0.0, 1.0),
            ("1", 1.0, 0.0),
            ("1_0+2_0j", 10.0, 20.0),
            ("1.j", 0.0, 1.0),
            (".5j", 0.0, 0.5),
            ("1e400j", 0.0, f64::INFINITY),
            ("-1.5e-3-infj", -0.0015, f64::NEG_INFINITY),
        ] {
            assert_eq!(parse(text).unwrap(), Complex::new(real, imag), "{text}");
        }
        let parsed = parse("inf+nanj").unwrap();
        assert!(parsed.real.is_infinite() && parsed.imag.is_nan());
        for text in [
            "1+2", "1+2jj", "(1+2j", "1 + 2j", "", "()", "1__0", "_1", "j1", "nanjj", "1e", ".j",
        ] {
            assert_eq!(
                parse(text).unwrap_err().kind,
                super::super::native::PyErrorKind::Value
            );
        }
    }

    #[test]
    fn arithmetic_follows_cpython_mixed_mode_and_annex_g_rules() {
        let complex = |real, imag| Operand::Complex(Complex::new(real, imag));
        let product = evaluate(Operation::Multiply, complex(1.0, 2.0), complex(3.0, -4.0));
        assert_eq!(product.unwrap(), Complex::new(11.0, 2.0));
        let quotient = evaluate(Operation::Divide, complex(1.0, 2.0), complex(3.0, -4.0));
        assert_eq!(quotient.unwrap(), Complex::new(-0.2, 0.4));
        let infinite = evaluate(
            Operation::Multiply,
            complex(1.0, 2.0),
            Operand::Real(f64::INFINITY),
        );
        assert_eq!(
            infinite.unwrap(),
            Complex::new(f64::INFINITY, f64::INFINITY)
        );
        let sum = evaluate(Operation::Add, complex(1.0, -0.0), Operand::Real(1.0)).unwrap();
        assert!(sum.imag == 0.0 && sum.imag.is_sign_negative());
        assert_eq!(
            evaluate(Operation::Power, complex(1.0, 2.0), Operand::Real(-2.0)).unwrap(),
            Complex::new(-0.12, -0.16)
        );
        assert_eq!(
            power(Complex::new(10.0, 0.0), Complex::new(400.0, 0.0)),
            Err(PowerError::Overflow)
        );
        assert_eq!(
            power(Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0)),
            Err(PowerError::ZeroToNegativeOrComplex)
        );
        assert_eq!(
            power(Complex::new(0.0, 0.0), Complex::new(0.0, 1.0)),
            Err(PowerError::ZeroToNegativeOrComplex)
        );
        assert!(evaluate(Operation::Divide, complex(1.0, 1.0), Operand::Real(0.0)).is_err());
    }
}
