//! Shortest round-trip decimal text for floats, as CPython's `repr` and NumPy's Dragon4 print it.
//!
//! Rust's `{:e}` finds the fewest digits that round-trip, but when the value lies exactly halfway
//! between two such digit strings it rounds up. CPython's `_Py_dg_dtoa` and NumPy's Dragon4
//! round half to even, so `repr(16.5042266845703125)` is `16.504226684570312`, not `...313`.

use std::fmt::LowerExp;
use std::str::FromStr;

/// A finite nonzero magnitude as `digits[0].digits[1..] × 10^exponent`, without trailing zeros.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Decimal {
    pub digits: String,
    pub exponent: i32,
}

/// Shortest round-trip digits of a finite nonzero `f64`'s magnitude.
pub(crate) fn shortest(value: f64) -> Decimal {
    shortest_of(value.abs())
}

/// Shortest digits that round-trip through `f32`, for NumPy's single-precision scalars.
pub(crate) fn shortest_f32(value: f32) -> Decimal {
    shortest_of(value.abs())
}

fn shortest_of<T>(magnitude: T) -> Decimal
where
    T: LowerExp + FromStr + PartialEq + Copy,
{
    let shortest = format!("{magnitude:e}");
    let length = parse(&shortest).digits.len();
    // `{:.N$e}` rounds the exact value half to even. At a power of two the round-trip interval
    // is lopsided, so the nearest string of this length may not round-trip; keep Rust's then.
    let even = format!("{magnitude:.*e}", length - 1);
    if even.parse::<T>().ok() == Some(magnitude) {
        parse(&even)
    } else {
        parse(&shortest)
    }
}

/// Correctly-rounded digits of a finite, non-negative magnitude at a fixed count of significant
/// digits, keeping trailing zeros (unlike [`shortest`]): `magnitude == digits[0].digits[1..] ×
/// 10^exponent`. Ties round to even, via `{:.N$e}`. Used where a caller needs an exact digit
/// count rather than the shortest round-trip text, e.g. NumPy's `precision` and
/// `min_digits` options.
pub(crate) fn fixed_digits(magnitude: f64, significant_digits: usize) -> (String, i32) {
    debug_assert!(significant_digits >= 1);
    debug_assert!(magnitude.is_finite() && magnitude >= 0.0);
    let text = format!("{magnitude:.*e}", significant_digits - 1);
    let (mantissa, exponent) = text.split_once('e').expect("exponent notation");
    (
        mantissa.replace('.', ""),
        exponent.parse().expect("integer exponent"),
    )
}

fn parse(text: &str) -> Decimal {
    let (mantissa, exponent) = text.split_once('e').expect("exponent notation");
    let digits = mantissa.replace('.', "");
    let digits = digits.trim_end_matches('0');
    Decimal {
        digits: if digits.is_empty() { "0" } else { digits }.to_string(),
        exponent: exponent.parse().expect("integer exponent"),
    }
}

/// CPython's `repr(float)`: shortest round-trip digits, positional when `1e-4 <= |x| < 1e16`
/// and scientific otherwise, with `.0` on integral positional values.
///
/// ```text
/// repr(0.1) == "0.1"; repr(1e16) == "1e+16"; repr(5e-324) == "5e-324"; repr(-2.0) == "-2.0"
/// ```
pub(crate) fn repr(value: f64) -> String {
    let sign = if value.is_sign_negative() && !value.is_nan() {
        "-"
    } else {
        ""
    };
    format!("{sign}{}", magnitude_repr(value, true))
}

/// The unsigned text of `repr(value)`. `dot_zero` appends `.0` to integral positional values,
/// as `float.__repr__` does and `complex.__repr__` does not.
pub(crate) fn magnitude_repr(value: f64, dot_zero: bool) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return "inf".into();
    }
    if value == 0.0 {
        return if dot_zero { "0.0" } else { "0" }.into();
    }
    let Decimal { digits, exponent } = shortest(value);
    if !(-4..16).contains(&exponent) {
        let (first, rest) = digits.split_at(1);
        let fraction = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        return format!(
            "{first}{fraction}e{exponent_sign}{:02}",
            exponent.unsigned_abs()
        );
    }
    if exponent < 0 {
        let zeros = "0".repeat(exponent.unsigned_abs() as usize - 1);
        return format!("0.{zeros}{digits}");
    }
    let point = exponent as usize + 1;
    if digits.len() > point {
        return format!("{}.{}", &digits[..point], &digits[point..]);
    }
    let padded = format!("{digits}{}", "0".repeat(point - digits.len()));
    if dot_zero {
        format!("{padded}.0")
    } else {
        padded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected text comes from CPython 3.14.4.
    #[test]
    fn repr_matches_cpython() {
        // 16.504226684570312 is exactly 16.5042266845703125, halfway between two 17-digit
        // strings.
        for (value, text) in [
            (16.504_226_684_570_312, "16.504226684570312"),
            (-28.182_485_580_444_336, "-28.182485580444336"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (5e-324, "5e-324"),
            (1e300, "1e+300"),
            (1.000_000_000_000_000_2, "1.0000000000000002"),
            (9_007_199_254_740_993.0, "9007199254740992.0"),
            (-0.0, "-0.0"),
            (f64::NAN, "nan"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(repr(value), text, "{value:e}");
        }
    }

    #[test]
    fn ties_round_to_even() {
        assert_eq!(shortest(16.504_226_684_570_312).digits, "16504226684570312");
        assert_eq!(shortest_f32(0.1).digits, "1");
    }
}
