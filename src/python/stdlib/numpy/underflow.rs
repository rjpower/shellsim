//! Underflow detection for NumPy's floating-point error flags.
//!
//! shellsim treats a scalar arithmetic result as underflowed when its operands are finite and
//! nonzero but the result is subnormal or flushed to zero: a real loss of magnitude too small to
//! represent at normal precision. `float16` arithmetic runs in `float32` and rounds to half
//! precision afterwards, so its underflow check applies the same rule to that rounding step:
//! the pre-rounding value plays the role of the operand, and the half-precision result plays the
//! role of the result. NumPy's own default `errstate` for underflow is `'ignore'`, so this flag
//! only matters to code that explicitly asks to see it.

use super::element::F16;

/// The smallest positive half-precision normal number, `2**-14`.
const HALF_MIN_POSITIVE: f32 = 6.103_515_6e-5;

/// Whether `a` and `b`, both finite, are nonzero and `result` is subnormal or zero.
fn underflowed(a: f64, b: f64, result: f64, smallest_normal: f64) -> bool {
    a.is_finite() && b.is_finite() && a != 0.0 && b != 0.0 && result.abs() < smallest_normal
}

/// Whether `a * b`, rounded to `result`, underflows.
pub(super) fn product_f64(a: f64, b: f64, result: f64) -> bool {
    underflowed(a, b, result, f64::MIN_POSITIVE)
}

/// Whether `a / b`, rounded to `result`, underflows.
pub(super) fn quotient_f64(a: f64, b: f64, result: f64) -> bool {
    underflowed(a, b, result, f64::MIN_POSITIVE)
}

/// Whether the single-precision `a * b`, rounded to `result`, underflows.
pub(super) fn product_f32(a: f32, b: f32, result: f32) -> bool {
    underflowed(
        f64::from(a),
        f64::from(b),
        f64::from(result),
        f64::from(f32::MIN_POSITIVE),
    )
}

/// Whether the single-precision `a / b`, rounded to `result`, underflows.
pub(super) fn quotient_f32(a: f32, b: f32, result: f32) -> bool {
    underflowed(
        f64::from(a),
        f64::from(b),
        f64::from(result),
        f64::from(f32::MIN_POSITIVE),
    )
}

/// Whether rounding `value` (already computed at `f32` precision) down to half precision
/// underflows: `value` is the nonzero finite operand, and the narrowed half value, promoted back
/// to `f32`, is the result.
pub(super) fn to_half(value: f32) -> bool {
    value.is_finite() && value != 0.0 && F16::from_f32(value).to_f32().abs() < HALF_MIN_POSITIVE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_products_and_quotients_need_nonzero_finite_operands() {
        // Both operands finite and nonzero, result subnormal (or flushed to zero): underflow.
        assert!(product_f64(1e-308, 1e-308, 1e-308 * 1e-308));
        let (a, b) = (2f64.powi(-1000), 2f64.powi(-50));
        assert!(product_f64(a, b, a * b));
        let smallest = f64::from_bits(1);
        assert!(quotient_f64(smallest, 3.0, smallest / 3.0));
        // A normal result never underflows, however tiny the operands.
        assert!(!product_f64(1e-200, 1e-100, 1e-200 * 1e-100));
        // A zero or non-finite operand never counts as underflow.
        assert!(!product_f64(0.0, 1e-308, 0.0));
        assert!(!quotient_f64(1e-308, f64::INFINITY, 0.0));
        assert!(!product_f64(f64::NAN, 1e-308, f64::NAN));
    }

    #[test]
    fn single_and_half_precision() {
        let (a, b) = (1e-30f32, 1e-30f32);
        assert!(product_f32(a, b, a * b));
        assert!(quotient_f32(
            f32::from_bits(1),
            3.0,
            f32::from_bits(1) / 3.0
        ));
        assert!(!product_f32(1e-15, 1e-15, 1e-15f32 * 1e-15f32));
        assert!(to_half(1e-6));
        // Below the smallest half normal, so the rounded half result is subnormal regardless of
        // whether narrowing was exact.
        assert!(to_half(2f32.powi(-20)));
        assert!(!to_half(1e-3));
        assert!(!to_half(0.0));
        assert!(!to_half(f32::NAN));
    }
}
