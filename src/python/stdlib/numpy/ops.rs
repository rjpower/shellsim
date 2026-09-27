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
use super::underflow;

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
    /// `np.reciprocal` for inexact types. Integer loops handle it separately.
    fn reciprocal(self, flags: &mut FpFlags) -> Self {
        Self::one().divide(self, flags)
    }
    fn negative(self, flags: &mut FpFlags) -> Self;
    /// `abs` within the same dtype; complex magnitudes change dtype and are handled separately.
    fn absolute(self, flags: &mut FpFlags) -> Self;
    fn sign(self) -> Self;
    /// IEEE partial order; complex values compare lexicographically.
    fn compare(self, other: Self) -> Option<Ordering>;
    fn is_nan(self) -> bool {
        false
    }

    /// `np.maximum`: propagates NaN. Equal operands give the second one, so
    /// `maximum(-0.0, 0.0)` is `0.0` and `maximum(0.0, -0.0)` is `-0.0`, as in NumPy.
    fn maximum(self, other: Self) -> Self {
        if self.is_nan() {
            return self;
        }
        if other.is_nan() {
            return other;
        }
        if self.compare(other) == Some(Ordering::Greater) {
            self
        } else {
            other
        }
    }

    /// `np.minimum`: propagates NaN. Equal operands give the second one.
    fn minimum(self, other: Self) -> Self {
        if self.is_nan() {
            return self;
        }
        if other.is_nan() {
            return other;
        }
        if self.compare(other) == Some(Ordering::Less) {
            self
        } else {
            other
        }
    }

    /// `self + sum(lane)`, as NumPy's `add` reduction combines a running total with one run of
    /// elements. Floating types sum the run pairwise ([`pairwise_sum`]); others add in order.
    fn add_lane(self, lane: &[Self], flags: &mut FpFlags) -> Self {
        lane.iter()
            .fold(self, |total, value| total.add(*value, flags))
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
}

/// Bitwise operations on integers and bool.
pub(in crate::python) trait Integer: Numeric {
    fn bit_and(self, other: Self) -> Self;
    fn bit_or(self, other: Self) -> Self;
    fn bit_xor(self, other: Self) -> Self;
    fn invert(self) -> Self;
    fn left_shift(self, other: Self) -> Self;
    fn right_shift(self, other: Self) -> Self;
}

/// Real floating-point types, computed in `f64` or `f32` as their precision requires.
pub(in crate::python) trait Real: Numeric {
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
    /// Apply a function at this type's working precision.
    fn map(self, double: fn(f64) -> f64, single: fn(f32) -> f32) -> Self;
    fn zip(self, other: Self, double: fn(f64, f64) -> f64, single: fn(f32, f32) -> f32) -> Self;
}

/// The sum of `values` with rounding error that grows like `log n` rather than `n`, grouped
/// exactly as NumPy's floating-point `add` reductions group it, so contiguous sums match NumPy
/// bit for bit. An empty slice sums to `-0.0`, the exact identity for float addition, so a lane
/// with no elements never flips another value's sign of zero.
pub(in crate::python) fn pairwise_sum<F>(values: &[F]) -> F
where
    F: Copy + std::ops::Add<Output = F> + From<f32>,
{
    pairwise(values, 8, F::from(-0.0f32), |left, right| left + right)
}

/// [`pairwise_sum`] of complex values, returned as `(real, imaginary)`. NumPy counts the real
/// and imaginary parts as separate scalars, so the grouping keeps four complex partial sums
/// where a real sum keeps eight.
pub(in crate::python) fn pairwise_complex_sum<F>(values: &[Complex<F>]) -> (F, F)
where
    F: Copy + std::ops::Add<Output = F> + From<f32>,
{
    let zero = Complex {
        re: F::from(-0.0f32),
        im: F::from(-0.0f32),
    };
    let total = pairwise(values, 4, zero, |left, right| Complex {
        re: left.re + right.re,
        im: left.im + right.im,
    });
    (total.re, total.im)
}

/// Pairwise summation with `lanes` (at most 8) interleaved partial sums. A run shorter than
/// `lanes` adds left to right. A run of at most 16 groups of `lanes` adds element `i` into
/// partial sum `i % lanes`, combines the partial sums as a balanced tree, then adds the leftover
/// tail. A longer run splits at half its length rounded down to a whole group and recurses, so
/// the depth is `O(log n)`.
fn pairwise<T: Copy>(values: &[T], lanes: usize, zero: T, add: impl Fn(T, T) -> T + Copy) -> T {
    let length = values.len();
    if length < lanes {
        return values.iter().fold(zero, |total, &value| add(total, value));
    }
    if length > 16 * lanes {
        let (left, right) = values.split_at(length / (2 * lanes) * lanes);
        return add(
            pairwise(left, lanes, zero, add),
            pairwise(right, lanes, zero, add),
        );
    }
    let grouped = length - length % lanes;
    let mut partial = [zero; 8];
    partial[..lanes].copy_from_slice(&values[..lanes]);
    for group in values[lanes..grouped].chunks_exact(lanes) {
        for (sum, &value) in partial.iter_mut().zip(group) {
            *sum = add(*sum, value);
        }
    }
    let mut width = lanes;
    while width > 1 {
        width /= 2;
        for index in 0..width {
            partial[index] = add(partial[2 * index], partial[2 * index + 1]);
        }
    }
    values[grouped..]
        .iter()
        .fold(partial[0], |total, &value| add(total, value))
}

/// [`float_flags`] for a result computed from many inputs.
fn lane_flags(result: f64, inputs: impl Iterator<Item = f64> + Clone, flags: &mut FpFlags) {
    if result.is_nan() && !inputs.clone().any(f64::is_nan) {
        flags.invalid = true;
    } else if result.is_infinite() && inputs.clone().all(f64::is_finite) {
        flags.overflow = true;
    }
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
    (
        $type:ty, $wide:ty, $to:expr, $from:expr,
        product: $product:expr, quotient: $quotient:expr, narrow: $narrow:expr
    ) => {
        impl Numeric for $type {
            fn add(self, other: Self, flags: &mut FpFlags) -> Self {
                self.binary(other, flags, |a, b| a + b)
            }
            fn subtract(self, other: Self, flags: &mut FpFlags) -> Self {
                self.binary(other, flags, |a, b| a - b)
            }
            fn multiply(self, other: Self, flags: &mut FpFlags) -> Self {
                let result = self.binary(other, flags, |a, b| a * b);
                flags.underflow |= $product(self, other, result);
                result
            }
            fn divide(self, other: Self, flags: &mut FpFlags) -> Self {
                let (a, b) = (self.to_f64(), other.to_f64());
                if b == 0.0 && a != 0.0 && !a.is_nan() {
                    flags.divide = true;
                    return Self::from_f64(a / b);
                }
                let result = self.binary(other, flags, |a, b| a / b);
                flags.underflow |= $quotient(self, other, result);
                result
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
            fn add_lane(self, lane: &[Self], flags: &mut FpFlags) -> Self {
                let values = lane.iter().map(|value| $to(*value)).collect::<Vec<$wide>>();
                let total: $wide = $to(self) + pairwise_sum(&values);
                let result = $from(total);
                lane_flags(
                    result.to_f64(),
                    std::iter::once(self.to_f64()).chain(lane.iter().map(|value| value.to_f64())),
                    flags,
                );
                result
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
                let wide = operation(a, b);
                let result = Self::from_f64(f64::from(wide));
                float_flags(result.to_f64(), &[f64::from(a), f64::from(b)], flags);
                flags.underflow |= $narrow(wide);
                result
            }
        }
    };
}

real!(
    f64, f64, |value: f64| value, |value: f64| value,
    product: underflow::product_f64,
    quotient: underflow::quotient_f64,
    narrow: |_: f64| false
);
real!(
    f32, f32, |value: f32| value, |value: f32| value,
    product: underflow::product_f32,
    quotient: underflow::quotient_f32,
    narrow: |_: f32| false
);
// Half arithmetic runs in single precision; the rounding back to half raises underflow.
real!(
    F16, f32, |value: F16| value.to_f32(), |value: f32| F16::from_f32(value),
    product: |_: F16, _: F16, _: F16| false,
    quotient: |_: F16, _: F16, _: F16| false,
    narrow: underflow::to_half
);

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

/// `a / b` for complex values, as `np.divide` gives it for `complex128`.
///
/// Smith's algorithm (1962): scale by the ratio of the divisor's smaller component to its
/// larger one, so every division stays within a factor of the divisor's own magnitude instead
/// of squaring both components the way the textbook `(ac+bd)/(c²+d²)` formula does, which
/// overflows for a divisor whose `|c|` or `|d|` is past roughly 1e154 even when the true
/// quotient is representable. A zero divisor (either sign of either zero) divides by a literal
/// `+0.0` rather than by `b` itself: black-box testing against the reference interpreter showed
/// all four sign combinations of a `0.0 + 0.0i` divisor give an identical result, matching `a /
/// +0.0` component-wise. [`Numeric::divide`] raises the floating-point flags for that zero-
/// divisor case (component-wise: `NaN / 0` raises none, a finite nonzero numerator component
/// raises `divide`, and `0 / 0` raises `invalid`); flags for every other divisor are inferred
/// generically by comparing this function's result against its operands.
///
/// This does not special-case divisors whose magnitude is within about 1e8 of `f64::MAX`: an
/// intermediate `c + d*r` can overflow to infinity there even though the quotient is finite, a
/// limitation confirmed to match NumPy's own `complex128` division bit-for-bit on such inputs
/// (for example `(1.7e308+1.7e308i) / (1.7e308+1.7e308i)` gives `NaN+0i` in both), though NumPy
/// additionally raises an `overflow` flag there that this module's generic flag inference cannot
/// see, since the final result is finite.
///
/// Scales by `1.0 / denom` once and multiplies both components, rather than dividing each
/// component by `denom` separately: the two roundings are not equivalent (a division and a
/// reciprocal-then-multiply can differ in their last bit), and matching NumPy's own last bit
/// requires the multiply form — confirmed black-box, bit-for-bit against `hex()`, on
/// `(-1 - 1.2246467991473532e-16j) / sqrt(2)`, where dividing each component by `denom` directly
/// gives an imaginary part one ULP away from NumPy's.
pub(in crate::python) fn complex_divide(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let (a_re, a_im) = a;
    let (b_re, b_im) = b;
    if b_re == 0.0 && b_im == 0.0 {
        return (a_re / 0.0, a_im / 0.0);
    }
    if b_re.abs() >= b_im.abs() {
        let r = b_im / b_re;
        let scl = 1.0 / (b_re + b_im * r);
        ((a_re + a_im * r) * scl, (a_im - a_re * r) * scl)
    } else {
        let r = b_re / b_im;
        let scl = 1.0 / (b_im + b_re * r);
        ((a_re * r + a_im) * scl, (a_im * r - a_re) * scl)
    }
}

/// `1 / (x, y)` by Smith's algorithm specialized for a numerator of exactly `1 + 0i`.
///
/// This is not simply [`complex_divide`] called with `a = (1.0, 0.0)`: NumPy's `reciprocal` and
/// its ordinary division disagree on the sign of a resulting zero at the same inputs (black-box
/// testing found `np.reciprocal(inf+0j)` is `0-0i` but `1/(inf+0j)` computed through division is
/// `0+0i`). The difference traces to which IEEE 754 operation produces the imaginary part: this
/// function negates `r` directly, where general division subtracts it from `a_im = 0.0`, and
/// `-(+0.0) == -0.0` while `0.0 - (+0.0) == +0.0`. Like [`complex_divide`], it scales by `1.0 /
/// denom` once and multiplies rather than dividing each component by `denom` separately, to
/// match NumPy's last bit.
fn smith_reciprocal(x: f64, y: f64) -> (f64, f64, f64) {
    if x.abs() >= y.abs() {
        let r = y / x;
        let denom = x + y * r;
        let scl = 1.0 / denom;
        (scl, -r * scl, denom)
    } else {
        let r = x / y;
        let denom = y + x * r;
        let scl = 1.0 / denom;
        (r * scl, -scl, denom)
    }
}

/// `np.reciprocal` of a complex value, raising the floating-point flags NumPy reports for it.
///
/// Unlike [`complex_divide`] and [`complex_power`], whose callers infer flags by comparing the
/// returned value against the original operands, `complex_reciprocal` sets its own: `invalid`
/// when either result component is `NaN`, and `overflow` when [`smith_reciprocal`]'s scaled
/// denominator overflows to infinity even though `x` and `y` were both finite, since that
/// intermediate overflow does not show up in the final result (which rounds to a finite `±0`).
pub(in crate::python) fn complex_reciprocal(a: (f64, f64), flags: &mut FpFlags) -> (f64, f64) {
    let (x, y) = a;
    let (real, imag, denom) = smith_reciprocal(x, y);
    if denom.is_infinite() && x.is_finite() && y.is_finite() {
        flags.overflow = true;
    }
    if real.is_nan() || imag.is_nan() {
        flags.invalid = true;
    }
    (real, imag)
}

/// `exp(re + im*i)`, exact on the real axis (`im == 0` or `-0`) the way `exp(x) * (cos(0),
/// sin(0))` is not, since `cos` and `sin` of exactly `0.0` still round to `1.0` and a
/// sign-losing `0.0` rather than preserving `im`'s own sign.
fn complex_exp((re, im): (f64, f64)) -> (f64, f64) {
    let scale = re.exp();
    if im == 0.0 {
        return (scale, im);
    }
    (scale * im.cos(), scale * im.sin())
}

/// `base ** n` for a nonzero complex `base` and non-negative integer `n`, by exponentiation by
/// squaring.
///
/// Seeds the accumulator with `base` itself at `n`'s lowest set bit instead of starting from the
/// multiplicative identity `1 + 0i`: identity-seeded squaring computes one extra `(1+0i) *
/// base`, which is enough to flip a resulting zero's sign. NumPy's own complex integer power
/// does not make that extra multiplication either — confirmed against `(1 - 0i) ** 2` and `(1 -
/// 0i) ** 3`, both of which keep the negative sign on their zero imaginary part.
fn integer_power(base: (f64, f64), n: u64) -> (f64, f64) {
    let mut result = None;
    let mut square = base;
    let mut bit = n;
    while bit > 0 {
        if bit & 1 == 1 {
            result = Some(match result {
                None => square,
                Some(accumulated) => complex_multiply(accumulated, square),
            });
        }
        bit >>= 1;
        if bit > 0 {
            square = complex_multiply(square, square);
        }
    }
    result.unwrap_or((1.0, 0.0))
}

/// `a ** b` for complex values, as `np.power` gives it for `complex128`.
///
/// `0 ** b` is a pole or a removable singularity depending on `b`'s real part, confirmed
/// black-box against the reference interpreter across `b in {0, positive, negative,
/// positive-real complex, zero-real complex}`: `0 ** 0` is `1`, `0 ** b` is `0` when `b`'s real
/// part is positive, and `0 ** b` is `NaN + NaNi` (with the generic flag inference below raising
/// `invalid`) otherwise, which includes a purely imaginary `b`.
///
/// A `b` with an exactly integer value and zero imaginary part uses [`integer_power`] (inverted
/// through [`smith_reciprocal`] for negative `b`), which is exact where the general formula
/// below is not: squaring `1e150 + 1e150i` cancels its real part to exactly `0` because both
/// terms of `x² - y²` round to the identical value before the subtraction, while `exp(b *
/// log(a))` accumulates rounding error in the angle long before the final `cos`/`sin`. Every
/// other exponent uses the principal branch, `exp(b * log(a))`, log's own principal branch cut
/// along the negative real axis.
pub(in crate::python) fn complex_power(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    if a == (0.0, 0.0) {
        if b == (0.0, 0.0) {
            return (1.0, 0.0);
        }
        return if b.0 > 0.0 {
            (0.0, 0.0)
        } else {
            (f64::NAN, f64::NAN)
        };
    }
    if b.1 == 0.0 && b.0.fract() == 0.0 && b.0.abs() < 9.2e18 {
        let n = b.0 as i64;
        let magnitude = integer_power(a, n.unsigned_abs());
        return if n >= 0 {
            magnitude
        } else {
            let (real, imag, _denom) = smith_reciprocal(magnitude.0, magnitude.1);
            (real, imag)
        };
    }
    let log_a = (a.0.hypot(a.1).ln(), a.1.atan2(a.0));
    complex_exp(complex_multiply(b, log_a))
}

pub(in crate::python) fn complex_multiply(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

macro_rules! complex {
    ($type:ty) => {
        impl Numeric for $type {
            const IS_COMPLEX: bool = true;

            fn add_lane(self, lane: &[Self], flags: &mut FpFlags) -> Self {
                let (re, im) = pairwise_complex_sum(lane);
                let result = Complex {
                    re: self.re + re,
                    im: self.im + im,
                };
                let parts = |value: &Self| {
                    let (re, im) = value.parts();
                    [re, im]
                };
                let inputs = std::iter::once(&self).chain(lane).flat_map(parts);
                let (re, im) = result.parts();
                lane_flags(re, inputs.clone(), flags);
                lane_flags(im, inputs, flags);
                result
            }

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
                    // NumPy sets a zero-divisor's flags per component, independent of the
                    // other component's NaN-ness: confirmed black-box, `(nan+1j) / (0+0j)`
                    // still warns "divide by zero" from its finite `1j` alone, and `(inf+0j) /
                    // (0+0j)` warns only "invalid" (an already-infinite component dividing by
                    // zero raises neither flag; only a finite-nonzero component raises
                    // `divide`, and an exact `0` raises `invalid`).
                    let (a_real, a_imag) = self.parts();
                    let mut classify = |component: f64| {
                        if component.is_nan() || component.is_infinite() {
                            // No flag: `nan / 0` is silent, and an already-infinite component
                            // dividing by zero does not newly overflow or invalidate.
                        } else if component == 0.0 {
                            flags.invalid = true;
                        } else {
                            flags.divide = true;
                        }
                    };
                    classify(a_real);
                    classify(a_imag);
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
            fn reciprocal(self, flags: &mut FpFlags) -> Self {
                let (real, imag) = complex_reciprocal(self.parts(), flags);
                Self::from_parts(real, imag)
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
