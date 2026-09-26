//! Per-element numeric semantics shared by ufuncs, reductions, and scalar arithmetic.
//!
//! [`Numeric`] is implemented once per element type, so every loop computes at its dtype's
//! width. The rules follow NumPy rather than Python:
//!
//! - Integers wrap. Division or remainder by zero gives 0 and raises the `divide` flag, and
//!   `MIN // -1` wraps with the `overflow` flag. Wrapping add, subtract, and multiply also set
//!   `overflow`; array loops ignore it and scalar operations report it, as NumPy does.
//! - Floats are IEEE. A finite nonzero value divided by zero sets `divide`; an operation that
//!   turns non-NaN inputs into NaN sets `invalid`; one that turns finite inputs into infinity
//!   sets `overflow`.
//! - `float16` computes in `f32` and rounds once when stored.
//! - Complex values order lexicographically, as NumPy sorts and compares them.

use std::cmp::Ordering;

use super::element::{Complex, Element, C128, C64, F16};

/// Floating-point exception flags raised by one loop, mirroring NumPy's FPE flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::python) struct FpFlags {
    pub divide: bool,
    pub overflow: bool,
    pub underflow: bool,
    pub invalid: bool,
}

impl FpFlags {
    pub(in crate::python) fn any(self) -> bool {
        self.divide || self.overflow || self.underflow || self.invalid
    }

    pub(in crate::python) fn merge(&mut self, other: Self) {
        self.divide |= other.divide;
        self.overflow |= other.overflow;
        self.underflow |= other.underflow;
        self.invalid |= other.invalid;
    }
}

/// Arithmetic every numeric element type supports.
pub(in crate::python) trait Numeric: Element {
    const IS_INTEGER: bool = false;
    const IS_COMPLEX: bool = false;

    fn add(self, other: Self, flags: &mut FpFlags) -> Self;
    fn subtract(self, other: Self, flags: &mut FpFlags) -> Self;
    fn multiply(self, other: Self, flags: &mut FpFlags) -> Self;
    /// True division. Integer loops never reach it: their inputs are cast to float first.
    fn divide(self, other: Self, flags: &mut FpFlags) -> Self;
    fn floor_divide(self, other: Self, flags: &mut FpFlags) -> Self;
    fn remainder(self, other: Self, flags: &mut FpFlags) -> Self;
    fn power(self, other: Self, flags: &mut FpFlags) -> Self;
    fn negative(self, flags: &mut FpFlags) -> Self;
    /// `abs` within the same dtype; complex magnitudes change dtype and are handled separately.
    fn absolute(self, flags: &mut FpFlags) -> Self;
    fn sign(self) -> Self;
    /// IEEE partial order; complex values compare lexicographically.
    fn compare(self, other: Self) -> Option<Ordering>;
    fn is_nan(self) -> bool {
        false
    }
    fn is_true(self) -> bool {
        self != Self::zero()
    }

    /// `np.maximum`: propagates NaN.
    fn maximum(self, other: Self) -> Self {
        if self.is_nan() {
            return self;
        }
        if other.is_nan() {
            return other;
        }
        if self.compare(other) == Some(Ordering::Less) {
            other
        } else {
            self
        }
    }

    /// `np.minimum`: propagates NaN.
    fn minimum(self, other: Self) -> Self {
        if self.is_nan() {
            return self;
        }
        if other.is_nan() {
            return other;
        }
        if self.compare(other) == Some(Ordering::Greater) {
            other
        } else {
            self
        }
    }

    /// `np.fmax`: ignores NaN when the other operand is a number.
    fn fmax(self, other: Self) -> Self {
        if self.is_nan() {
            return other;
        }
        if other.is_nan() {
            return self;
        }
        self.maximum(other)
    }

    /// `np.fmin`: ignores NaN when the other operand is a number.
    fn fmin(self, other: Self) -> Self {
        if self.is_nan() {
            return other;
        }
        if other.is_nan() {
            return self;
        }
        self.minimum(other)
    }

    /// Sort order with NaN last, as `np.sort` uses.
    fn sort_order(self, other: Self) -> Ordering {
        match (self.is_nan(), other.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => self.compare(other).unwrap_or(Ordering::Equal),
        }
    }
}

/// Bitwise operations on integers and bool.
pub(in crate::python) trait Integer: Numeric {
    fn bit_and(self, other: Self) -> Self;
    fn bit_or(self, other: Self) -> Self;
    fn bit_xor(self, other: Self) -> Self;
    fn invert(self) -> Self;
    fn left_shift(self, other: Self) -> Self;
    fn right_shift(self, other: Self) -> Self;
    /// The value as `i128`, for exact comparisons across integer kinds and gcd/lcm.
    fn wide(self) -> i128;
}

/// Real floating-point types, computed in `f64` or `f32` as their precision requires.
pub(in crate::python) trait Real: Numeric {
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
    /// Apply a function at this type's working precision.
    fn map(self, double: fn(f64) -> f64, single: fn(f32) -> f32) -> Self;
    fn zip(self, other: Self, double: fn(f64, f64) -> f64, single: fn(f32, f32) -> f32) -> Self;
}

fn float_flags(result: f64, inputs: &[f64], flags: &mut FpFlags) {
    if result.is_nan() && !inputs.iter().any(|value| value.is_nan()) {
        flags.invalid = true;
    } else if result.is_infinite() && inputs.iter().all(|value| value.is_finite()) {
        flags.overflow = true;
    }
}

macro_rules! signed {
    ($type:ty) => {
        impl Numeric for $type {
            const IS_INTEGER: bool = true;

            fn add(self, other: Self, flags: &mut FpFlags) -> Self {
                let (value, overflow) = self.overflowing_add(other);
                flags.overflow |= overflow;
                value
            }

            fn subtract(self, other: Self, flags: &mut FpFlags) -> Self {
                let (value, overflow) = self.overflowing_sub(other);
                flags.overflow |= overflow;
                value
            }

            fn multiply(self, other: Self, flags: &mut FpFlags) -> Self {
                let (value, overflow) = self.overflowing_mul(other);
                flags.overflow |= overflow;
                value
            }

            fn divide(self, other: Self, flags: &mut FpFlags) -> Self {
                self.floor_divide(other, flags)
            }

            fn floor_divide(self, other: Self, flags: &mut FpFlags) -> Self {
                if other == 0 {
                    flags.divide = true;
                    return 0;
                }
                if self == <$type>::MIN && other == -1 {
                    flags.overflow = true;
                    return <$type>::MIN;
                }
                let quotient = self / other;
                if (self % other != 0) && ((self < 0) != (other < 0)) {
                    quotient - 1
                } else {
                    quotient
                }
            }

            fn remainder(self, other: Self, flags: &mut FpFlags) -> Self {
                if other == 0 {
                    flags.divide = true;
                    return 0;
                }
                if other == -1 {
                    return 0;
                }
                let remainder = self % other;
                if remainder != 0 && ((remainder < 0) != (other < 0)) {
                    remainder + other
                } else {
                    remainder
                }
            }

            fn power(self, other: Self, flags: &mut FpFlags) -> Self {
                // Negative exponents are rejected before the loop runs.
                let mut result: $type = 1;
                let mut base = self;
                let mut exponent = other.max(0) as u64;
                while exponent > 0 {
                    if exponent & 1 == 1 {
                        result = result.multiply(base, flags);
                    }
                    exponent >>= 1;
                    if exponent > 0 {
                        base = base.multiply(base, flags);
                    }
                }
                result
            }

            fn negative(self, flags: &mut FpFlags) -> Self {
                let (value, overflow) = self.overflowing_neg();
                flags.overflow |= overflow;
                value
            }

            fn absolute(self, flags: &mut FpFlags) -> Self {
                if self == <$type>::MIN {
                    flags.overflow = true;
                }
                self.wrapping_abs()
            }

            fn sign(self) -> Self {
                self.signum()
            }

            fn compare(self, other: Self) -> Option<Ordering> {
                Some(self.cmp(&other))
            }
        }

        impl Integer for $type {
            fn bit_and(self, other: Self) -> Self {
                self & other
            }
            fn bit_or(self, other: Self) -> Self {
                self | other
            }
            fn bit_xor(self, other: Self) -> Self {
                self ^ other
            }
            fn invert(self) -> Self {
                !self
            }
            fn left_shift(self, other: Self) -> Self {
                if other < 0 || other as u32 >= <$type>::BITS {
                    0
                } else {
                    self.wrapping_shl(other as u32)
                }
            }
            fn right_shift(self, other: Self) -> Self {
                if other < 0 || other as u32 >= <$type>::BITS {
                    if self < 0 {
                        -1
                    } else {
                        0
                    }
                } else {
                    self >> other
                }
            }
            fn wide(self) -> i128 {
                i128::from(self)
            }
        }
    };
}

macro_rules! unsigned {
    ($type:ty) => {
        impl Numeric for $type {
            const IS_INTEGER: bool = true;

            fn add(self, other: Self, flags: &mut FpFlags) -> Self {
                let (value, overflow) = self.overflowing_add(other);
                flags.overflow |= overflow;
                value
            }

            fn subtract(self, other: Self, flags: &mut FpFlags) -> Self {
                let (value, overflow) = self.overflowing_sub(other);
                flags.overflow |= overflow;
                value
            }

            fn multiply(self, other: Self, flags: &mut FpFlags) -> Self {
                let (value, overflow) = self.overflowing_mul(other);
                flags.overflow |= overflow;
                value
            }

            fn divide(self, other: Self, flags: &mut FpFlags) -> Self {
                self.floor_divide(other, flags)
            }

            fn floor_divide(self, other: Self, flags: &mut FpFlags) -> Self {
                if other == 0 {
                    flags.divide = true;
                    return 0;
                }
                self / other
            }

            fn remainder(self, other: Self, flags: &mut FpFlags) -> Self {
                if other == 0 {
                    flags.divide = true;
                    return 0;
                }
                self % other
            }

            fn power(self, other: Self, flags: &mut FpFlags) -> Self {
                let mut result: $type = 1;
                let mut base = self;
                let mut exponent = other as u64;
                while exponent > 0 {
                    if exponent & 1 == 1 {
                        result = result.multiply(base, flags);
                    }
                    exponent >>= 1;
                    if exponent > 0 {
                        base = base.multiply(base, flags);
                    }
                }
                result
            }

            fn negative(self, flags: &mut FpFlags) -> Self {
                let (value, overflow) = self.overflowing_neg();
                flags.overflow |= overflow;
                value
            }

            fn absolute(self, _flags: &mut FpFlags) -> Self {
                self
            }

            fn sign(self) -> Self {
                <$type>::from(self != 0)
            }

            fn compare(self, other: Self) -> Option<Ordering> {
                Some(self.cmp(&other))
            }
        }

        impl Integer for $type {
            fn bit_and(self, other: Self) -> Self {
                self & other
            }
            fn bit_or(self, other: Self) -> Self {
                self | other
            }
            fn bit_xor(self, other: Self) -> Self {
                self ^ other
            }
            fn invert(self) -> Self {
                !self
            }
            fn left_shift(self, other: Self) -> Self {
                if other as u64 >= u64::from(<$type>::BITS) {
                    0
                } else {
                    self.wrapping_shl(other as u32)
                }
            }
            fn right_shift(self, other: Self) -> Self {
                if other as u64 >= u64::from(<$type>::BITS) {
                    0
                } else {
                    self >> other
                }
            }
            fn wide(self) -> i128 {
                i128::from(self)
            }
        }
    };
}

signed!(i8);
signed!(i16);
signed!(i32);
signed!(i64);
unsigned!(u8);
unsigned!(u16);
unsigned!(u32);
unsigned!(u64);

/// Bool arithmetic: `+` is logical or and `*` is logical and. Subtraction and negation are
/// rejected by the ufunc resolvers before reaching these loops.
impl Numeric for bool {
    const IS_INTEGER: bool = true;

    fn add(self, other: Self, _flags: &mut FpFlags) -> Self {
        self || other
    }
    fn subtract(self, other: Self, _flags: &mut FpFlags) -> Self {
        self ^ other
    }
    fn multiply(self, other: Self, _flags: &mut FpFlags) -> Self {
        self && other
    }
    fn divide(self, other: Self, flags: &mut FpFlags) -> Self {
        self.floor_divide(other, flags)
    }
    fn floor_divide(self, other: Self, flags: &mut FpFlags) -> Self {
        if !other {
            flags.divide = true;
            return false;
        }
        self
    }
    fn remainder(self, other: Self, flags: &mut FpFlags) -> Self {
        if !other {
            flags.divide = true;
        }
        false
    }
    fn power(self, other: Self, _flags: &mut FpFlags) -> Self {
        self || !other
    }
    fn negative(self, _flags: &mut FpFlags) -> Self {
        !self
    }
    fn absolute(self, _flags: &mut FpFlags) -> Self {
        self
    }
    fn sign(self) -> Self {
        self
    }
    fn compare(self, other: Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl Integer for bool {
    fn bit_and(self, other: Self) -> Self {
        self & other
    }
    fn bit_or(self, other: Self) -> Self {
        self | other
    }
    fn bit_xor(self, other: Self) -> Self {
        self ^ other
    }
    fn invert(self) -> Self {
        !self
    }
    fn left_shift(self, other: Self) -> Self {
        self && !other
    }
    fn right_shift(self, other: Self) -> Self {
        self && !other
    }
    fn wide(self) -> i128 {
        i128::from(self)
    }
}

/// Python's float floor division: `fmod`-based so the result is exact where possible.
fn floor_divide_f64(left: f64, right: f64) -> f64 {
    if right == 0.0 {
        return left / right;
    }
    let remainder = left % right;
    let mut quotient = (left - remainder) / right;
    if remainder != 0.0 && ((right < 0.0) != (remainder < 0.0)) {
        quotient -= 1.0;
    }
    if quotient != 0.0 {
        let floored = quotient.floor();
        if quotient - floored > 0.5 {
            floored + 1.0
        } else {
            floored
        }
    } else {
        0.0f64.copysign(left / right)
    }
}

/// Python's float modulo: the result takes the divisor's sign.
fn remainder_f64(left: f64, right: f64) -> f64 {
    if right == 0.0 {
        return f64::NAN;
    }
    let remainder = left % right;
    if remainder != 0.0 {
        if (right < 0.0) != (remainder < 0.0) {
            remainder + right
        } else {
            remainder
        }
    } else {
        0.0f64.copysign(right)
    }
}

/// Binary arithmetic at a float type's working precision: `f32` for half and single, `f64`
/// for double, as NumPy's loops compute.
trait RealBinary: Sized {
    type Wide;

    /// Apply `operation` to both operands at working precision, round, and record flags.
    fn binary(
        self,
        other: Self,
        flags: &mut FpFlags,
        operation: fn(Self::Wide, Self::Wide) -> Self::Wide,
    ) -> Self;
}

macro_rules! real {
    ($type:ty, $wide:ty, $to:expr, $from:expr) => {
        impl Numeric for $type {
            fn add(self, other: Self, flags: &mut FpFlags) -> Self {
                self.binary(other, flags, |a, b| a + b)
            }
            fn subtract(self, other: Self, flags: &mut FpFlags) -> Self {
                self.binary(other, flags, |a, b| a - b)
            }
            fn multiply(self, other: Self, flags: &mut FpFlags) -> Self {
                self.binary(other, flags, |a, b| a * b)
            }
            fn divide(self, other: Self, flags: &mut FpFlags) -> Self {
                let (a, b) = (self.to_f64(), other.to_f64());
                if b == 0.0 && a != 0.0 && !a.is_nan() {
                    flags.divide = true;
                    return Self::from_f64(a / b);
                }
                self.binary(other, flags, |a, b| a / b)
            }
            fn floor_divide(self, other: Self, flags: &mut FpFlags) -> Self {
                let (a, b) = (self.to_f64(), other.to_f64());
                if b == 0.0 {
                    if a != 0.0 && !a.is_nan() {
                        flags.divide = true;
                    } else if !a.is_nan() {
                        flags.invalid = true;
                    }
                    return Self::from_f64(a / b);
                }
                self.binary(other, flags, |a, b| {
                    floor_divide_f64(a.into(), b.into()) as $wide
                })
            }
            fn remainder(self, other: Self, flags: &mut FpFlags) -> Self {
                let (a, b) = (self.to_f64(), other.to_f64());
                if b == 0.0 {
                    if !a.is_nan() {
                        flags.invalid = true;
                    }
                    return Self::from_f64(f64::NAN);
                }
                self.binary(other, flags, |a, b| {
                    remainder_f64(a.into(), b.into()) as $wide
                })
            }
            fn power(self, other: Self, flags: &mut FpFlags) -> Self {
                self.binary(other, flags, |a, b| a.powf(b))
            }
            fn negative(self, _flags: &mut FpFlags) -> Self {
                Self::from_f64(-self.to_f64())
            }
            fn absolute(self, _flags: &mut FpFlags) -> Self {
                Self::from_f64(self.to_f64().abs())
            }
            fn sign(self) -> Self {
                let value = self.to_f64();
                Self::from_f64(if value.is_nan() {
                    value
                } else if value > 0.0 {
                    1.0
                } else if value < 0.0 {
                    -1.0
                } else {
                    0.0
                })
            }
            fn compare(self, other: Self) -> Option<Ordering> {
                self.to_f64().partial_cmp(&other.to_f64())
            }
            fn is_nan(self) -> bool {
                self.to_f64().is_nan()
            }
            fn is_true(self) -> bool {
                self.to_f64() != 0.0
            }
        }

        impl RealBinary for $type {
            type Wide = $wide;

            fn binary(
                self,
                other: Self,
                flags: &mut FpFlags,
                operation: fn($wide, $wide) -> $wide,
            ) -> Self {
                let a: $wide = $to(self);
                let b: $wide = $to(other);
                let result = Self::from_f64(f64::from(operation(a, b)));
                float_flags(result.to_f64(), &[f64::from(a), f64::from(b)], flags);
                result
            }
        }
    };
}

real!(f64, f64, |value: f64| value, |value: f64| value);
real!(f32, f32, |value: f32| value, |value: f32| value);
real!(F16, f32, |value: F16| value.to_f32(), |value: f32| {
    F16::from_f32(value)
});

impl Real for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(value: f64) -> Self {
        value
    }
    fn map(self, double: fn(f64) -> f64, _single: fn(f32) -> f32) -> Self {
        double(self)
    }
    fn zip(self, other: Self, double: fn(f64, f64) -> f64, _single: fn(f32, f32) -> f32) -> Self {
        double(self, other)
    }
}

impl Real for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn map(self, _double: fn(f64) -> f64, single: fn(f32) -> f32) -> Self {
        single(self)
    }
    fn zip(self, other: Self, _double: fn(f64, f64) -> f64, single: fn(f32, f32) -> f32) -> Self {
        single(self, other)
    }
}

impl Real for F16 {
    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }
    fn from_f64(value: f64) -> Self {
        F16::from_f64(value)
    }
    fn map(self, _double: fn(f64) -> f64, single: fn(f32) -> f32) -> Self {
        F16::from_f32(single(self.to_f32()))
    }
    fn zip(self, other: Self, _double: fn(f64, f64) -> f64, single: fn(f32, f32) -> f32) -> Self {
        F16::from_f32(single(self.to_f32(), other.to_f32()))
    }
}

/// Complex arithmetic at the precision of its parts.
pub(in crate::python) trait ComplexParts: Numeric {
    fn parts(self) -> (f64, f64);
    fn from_parts(real: f64, imag: f64) -> Self;
}

impl ComplexParts for C64 {
    fn parts(self) -> (f64, f64) {
        (f64::from(self.re), f64::from(self.im))
    }
    fn from_parts(real: f64, imag: f64) -> Self {
        Complex {
            re: real as f32,
            im: imag as f32,
        }
    }
}

impl ComplexParts for C128 {
    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }
    fn from_parts(real: f64, imag: f64) -> Self {
        Complex { re: real, im: imag }
    }
}

/// Smith's algorithm for complex division, as NumPy uses.
pub(in crate::python) fn complex_divide(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let ((ar, ai), (br, bi)) = (a, b);
    if br.abs() >= bi.abs() {
        if br == 0.0 && bi == 0.0 {
            return (ar / br.abs(), ai / br.abs());
        }
        let ratio = bi / br;
        let denominator = br + bi * ratio;
        (
            (ar + ai * ratio) / denominator,
            (ai - ar * ratio) / denominator,
        )
    } else {
        let ratio = br / bi;
        let denominator = br * ratio + bi;
        (
            (ar * ratio + ai) / denominator,
            (ai * ratio - ar) / denominator,
        )
    }
}

/// `a ** b` for complex values, with exact results for small integer exponents.
pub(in crate::python) fn complex_power(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    if b == (0.0, 0.0) {
        return (1.0, 0.0);
    }
    if a == (0.0, 0.0) {
        return if b.0 > 0.0 && b.1 == 0.0 {
            (0.0, 0.0)
        } else {
            (f64::NAN, f64::NAN)
        };
    }
    if b.1 == 0.0 && b.0.fract() == 0.0 && b.0.abs() <= 100.0 {
        let mut result = (1.0, 0.0);
        let mut base = a;
        let mut exponent = b.0.abs() as u32;
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = complex_multiply(result, base);
            }
            base = complex_multiply(base, base);
            exponent >>= 1;
        }
        return if b.0 < 0.0 {
            complex_divide((1.0, 0.0), result)
        } else {
            result
        };
    }
    let magnitude = a.0.hypot(a.1);
    let angle = a.1.atan2(a.0);
    let log_magnitude = magnitude.ln();
    let real = (b.0 * log_magnitude - b.1 * angle).exp();
    let phase = b.1 * log_magnitude + b.0 * angle;
    (real * phase.cos(), real * phase.sin())
}

pub(in crate::python) fn complex_multiply(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

macro_rules! complex {
    ($type:ty) => {
        impl Numeric for $type {
            const IS_COMPLEX: bool = true;

            fn add(self, other: Self, flags: &mut FpFlags) -> Self {
                self.combine(other, flags, |a, b| (a.0 + b.0, a.1 + b.1))
            }
            fn subtract(self, other: Self, flags: &mut FpFlags) -> Self {
                self.combine(other, flags, |a, b| (a.0 - b.0, a.1 - b.1))
            }
            fn multiply(self, other: Self, flags: &mut FpFlags) -> Self {
                self.combine(other, flags, complex_multiply)
            }
            fn divide(self, other: Self, flags: &mut FpFlags) -> Self {
                let (b_real, b_imag) = other.parts();
                if b_real == 0.0 && b_imag == 0.0 {
                    let (a_real, a_imag) = self.parts();
                    if a_real.is_nan() || a_imag.is_nan() {
                    } else if a_real != 0.0 || a_imag != 0.0 {
                        flags.divide = true;
                    } else {
                        flags.invalid = true;
                    }
                    let (real, imag) = complex_divide(self.parts(), other.parts());
                    return Self::from_parts(real, imag);
                }
                self.combine(other, flags, complex_divide)
            }
            fn floor_divide(self, other: Self, flags: &mut FpFlags) -> Self {
                // Rejected by the resolver; kept total for generic reductions.
                self.divide(other, flags)
            }
            fn remainder(self, _other: Self, _flags: &mut FpFlags) -> Self {
                Self::from_parts(f64::NAN, f64::NAN)
            }
            fn power(self, other: Self, flags: &mut FpFlags) -> Self {
                self.combine(other, flags, complex_power)
            }
            fn negative(self, _flags: &mut FpFlags) -> Self {
                let (real, imag) = self.parts();
                Self::from_parts(-real, -imag)
            }
            fn absolute(self, _flags: &mut FpFlags) -> Self {
                let (real, imag) = self.parts();
                Self::from_parts(real.hypot(imag), 0.0)
            }
            fn sign(self) -> Self {
                // NumPy 2: z / |z|, and 0 for 0.
                let (real, imag) = self.parts();
                let magnitude = real.hypot(imag);
                if magnitude == 0.0 {
                    Self::from_parts(0.0, 0.0)
                } else {
                    Self::from_parts(real / magnitude, imag / magnitude)
                }
            }
            fn compare(self, other: Self) -> Option<Ordering> {
                let (a, b) = (self.parts(), other.parts());
                match a.0.partial_cmp(&b.0)? {
                    Ordering::Equal => a.1.partial_cmp(&b.1),
                    ordering => Some(ordering),
                }
            }
            fn is_nan(self) -> bool {
                let (real, imag) = self.parts();
                real.is_nan() || imag.is_nan()
            }
            fn is_true(self) -> bool {
                let (real, imag) = self.parts();
                real != 0.0 || imag != 0.0
            }
        }

        impl $type {
            fn combine(
                self,
                other: Self,
                flags: &mut FpFlags,
                operation: fn((f64, f64), (f64, f64)) -> (f64, f64),
            ) -> Self {
                let (a, b) = (self.parts(), other.parts());
                let (real, imag) = operation(a, b);
                let result = Self::from_parts(real, imag);
                let (real, imag) = result.parts();
                let inputs = [a.0, a.1, b.0, b.1];
                float_flags(real, &inputs, flags);
                float_flags(imag, &inputs, flags);
                result
            }
        }
    };
}

complex!(C64);
complex!(C128);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_wrap_and_floor_like_numpy() {
        let mut flags = FpFlags::default();
        assert_eq!(127i8.add(1, &mut flags), -128);
        assert!(flags.overflow);
        let mut flags = FpFlags::default();
        assert_eq!((-7i64).floor_divide(2, &mut flags), -4);
        assert_eq!((-7i64).remainder(3, &mut flags), 2);
        assert_eq!(7i64.remainder(-3, &mut flags), -2);
        assert!(!flags.any());
        assert_eq!(5i64.floor_divide(0, &mut flags), 0);
        assert!(flags.divide);
        assert_eq!(0u8.subtract(1, &mut FpFlags::default()), 255);
        assert_eq!(2i64.power(10, &mut FpFlags::default()), 1024);
    }

    #[test]
    fn floats_follow_ieee_with_numpy_flags() {
        let mut flags = FpFlags::default();
        assert_eq!(1.0f64.divide(0.0, &mut flags), f64::INFINITY);
        assert!(flags.divide && !flags.invalid);
        let mut flags = FpFlags::default();
        assert!(0.0f64.divide(0.0, &mut flags).is_nan());
        assert!(flags.invalid);
        let mut flags = FpFlags::default();
        assert_eq!(1e308f64.multiply(10.0, &mut flags), f64::INFINITY);
        assert!(flags.overflow);
        let mut flags = FpFlags::default();
        assert_eq!((-7.5f64).floor_divide(2.0, &mut flags), -4.0);
        assert_eq!((-7.5f64).remainder(2.0, &mut flags), 0.5);
        assert!(!flags.any());
    }

    #[test]
    fn half_and_single_precision_round_every_result() {
        let mut flags = FpFlags::default();
        assert_eq!(
            F16::from_f32(2048.0)
                .add(F16::from_f32(1.0), &mut flags)
                .to_f32(),
            2048.0
        );
        assert_eq!(16_777_216f32.add(1.0, &mut flags), 16_777_216.0);
        assert!(Numeric::maximum(f64::NAN, 1.0).is_nan());
        assert_eq!(f64::NAN.fmax(1.0), 1.0);
    }

    #[test]
    fn complex_values_order_lexicographically() {
        let a = C128 { re: 1.0, im: 2.0 };
        let b = C128 { re: 1.0, im: 3.0 };
        assert_eq!(a.compare(b), Some(Ordering::Less));
        let quotient = a.divide(b, &mut FpFlags::default());
        // (1+2j)/(1+3j) = (1+2j)(1-3j)/10 = 0.7-0.1j
        assert!((quotient.re - 0.7).abs() < 1e-12 && (quotient.im + 0.1).abs() < 1e-12);
    }
}
