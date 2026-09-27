//! IEEE underflow detection for NumPy's floating-point error flags.
//!
//! NumPy reads the underflow flag from the FPU after a loop. The flag means the result is tiny
//! and inexact. On x86, the reference platform, tininess is judged after rounding: the result
//! rounded to full precision with an unbounded exponent is below the smallest normal number. A
//! tiny result that is exact, such as the difference of two subnormals, raises nothing.
//!
//! Only multiplication and division can produce inexact tiny results from `f64` or `f32`
//! operands; sums of subnormals are exact. `float16` arithmetic runs in `float32` and rounds to
//! half precision afterwards, and that conversion raises the flag as NumPy's
//! `npy_floatbits_to_halfbits` does, judging tininess before rounding.

/// Whether `a * b`, rounded to `result`, underflows.
pub(super) fn product_f64(a: f64, b: f64, result: f64) -> bool {
    if !candidate(result, a, b) || a == 0.0 || b == 0.0 {
        return false;
    }
    let ((a, a_exponent), (b, b_exponent)) = (split(a), split(b));
    // The product of the mantissas is normal, so `mul_add` recovers its rounding error exactly.
    let product = a * b;
    let error = a.mul_add(b, -product);
    let exponent = a_exponent + b_exponent;
    let tiny = product.abs() < scale(f64::MIN_POSITIVE, -exponent);
    tiny && (error != 0.0 || scale(result, -exponent) != product)
}

/// Whether `a / b`, rounded to `result`, underflows.
pub(super) fn quotient_f64(a: f64, b: f64, result: f64) -> bool {
    if !candidate(result, a, b) || a == 0.0 || b == 0.0 {
        return false;
    }
    let ((a, a_exponent), (b, b_exponent)) = (split(a), split(b));
    let quotient = a / b;
    let remainder = (-quotient).mul_add(b, a);
    let exponent = a_exponent - b_exponent;
    let tiny = quotient.abs() < scale(f64::MIN_POSITIVE, -exponent);
    tiny && (remainder != 0.0 || scale(result, -exponent) != quotient)
}

/// Whether the single-precision `a * b`, rounded to `result`, underflows. The product of two
/// `f32` values is exact in `f64`.
pub(super) fn product_f32(a: f32, b: f32, result: f32) -> bool {
    let exact = f64::from(a) * f64::from(b);
    tiny_f32(exact) && f64::from(result) != exact
}

/// Whether the single-precision `a / b`, rounded to `result`, underflows.
pub(super) fn quotient_f32(a: f32, b: f32, result: f32) -> bool {
    if !candidate(f64::from(result), f64::from(a), f64::from(b)) || b == 0.0 {
        return false;
    }
    let inexact = f64::from(result) * f64::from(b) != f64::from(a);
    inexact && tiny_f32(f64::from(a) / f64::from(b))
}

/// Whether rounding the single-precision `value` to half precision underflows.
pub(super) fn to_half(value: f32) -> bool {
    // 2**-14 is the smallest normal half; half subnormals are multiples of 2**-24.
    value != 0.0 && value.abs() < 2f32.powi(-14) && (value * 2f32.powi(24)).fract() != 0.0
}

/// A result can only have underflowed when it is below the normal range and both operands are
/// finite.
fn candidate(result: f64, a: f64, b: f64) -> bool {
    result.abs() < f64::MIN_POSITIVE && a.is_finite() && b.is_finite()
}

/// Whether a nonzero `f64` value is tiny for `f32` after rounding to 24 bits. Scaling by 2**200
/// keeps every quotient or product of `f32` values in the normal `f32` range while rounding.
fn tiny_f32(exact: f64) -> bool {
    exact != 0.0 && ((exact * 2f64.powi(200)) as f32).abs() < 2f32.powi(-126 + 200)
}

/// Split a finite nonzero value into a mantissa in `[1, 2)` with its sign, and an exponent.
fn split(value: f64) -> (f64, i32) {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent == 0 {
        let (mantissa, exponent) = split(value * 2f64.powi(64));
        return (mantissa, exponent - 64);
    }
    let mantissa = f64::from_bits((bits & !(0x7ff << 52)) | (1023 << 52));
    (mantissa, exponent - 1023)
}

/// `value * 2**exponent`, stepping so no intermediate power of two overflows.
fn scale(mut value: f64, mut exponent: i32) -> f64 {
    while exponent > 1000 {
        value *= 2f64.powi(1000);
        exponent -= 1000;
    }
    while exponent < -1000 {
        value *= 2f64.powi(-1000);
        exponent += 1000;
    }
    value * 2f64.powi(exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_products_and_quotients() {
        // 1e-308 * 1e-308 rounds to zero: tiny and inexact.
        assert!(product_f64(1e-308, 1e-308, 1e-308 * 1e-308));
        // 2**-1000 * 2**-50 is exactly 2**-1050, a representable subnormal.
        let (a, b) = (2f64.powi(-1000), 2f64.powi(-50));
        assert!(!product_f64(a, b, a * b));
        // 3 * 2**-1074 is exact; 2**-1074 / 3 is not.
        let smallest = f64::from_bits(1);
        assert!(!product_f64(3.0, smallest, 3.0 * smallest));
        assert!(quotient_f64(smallest, 3.0, smallest / 3.0));
        assert!(!quotient_f64(4.0 * smallest, 2.0, 2.0 * smallest));
        // Normal results and exact zeros never underflow.
        assert!(!product_f64(1e-200, 1e-100, 1e-200 * 1e-100));
        assert!(!product_f64(0.0, 1e-308, 0.0));
        assert!(!quotient_f64(1e-308, f64::INFINITY, 0.0));
    }

    #[test]
    fn single_and_half_precision() {
        let (a, b) = (1e-30f32, 1e-30f32);
        assert!(product_f32(a, b, a * b));
        let exact = 2f32.powi(-140);
        assert!(!product_f32(exact, 2.0, exact * 2.0));
        assert!(quotient_f32(
            f32::from_bits(1),
            3.0,
            f32::from_bits(1) / 3.0
        ));
        assert!(to_half(1e-6));
        assert!(!to_half(2f32.powi(-20)));
        assert!(!to_half(1e-3));
        assert!(!to_half(0.0));
    }
}
