//! Text for NumPy scalars and arrays.
//!
//! Scalar text uses the shortest decimal that round-trips at the value's own precision. It is
//! positional when `1e-4 <= |x| < 10^k` and scientific otherwise, where `k` is 3, 6, or 16 for
//! half, single, and double precision, as in NumPy. So `np.float32(0.1)` prints `0.1` although
//! its `float()` value is `0.10000000149011612`, and `np.float16(2048)` prints `2.048e+03`.

use super::super::super::float_text;
use super::element::{Number, F16};

/// Floating-point precision of a value being printed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum Precision {
    Half,
    Single,
    Double,
}

/// Shortest round-trip decimal digits and decimal exponent of a finite, nonzero value:
/// `value == 0.{digits} × 10^(exponent + 1)`, i.e. `digits[0]` is the units digit at
/// `10^exponent`. Shared with [`super::dragon4`], which extends this to fixed digit counts.
pub(super) fn shortest_digits(value: f64, precision: Precision) -> (String, i32) {
    let decimal = match precision {
        Precision::Double => float_text::shortest(value),
        Precision::Single => float_text::shortest_f32(value as f32),
        Precision::Half => return half_shortest_digits(value),
    };
    (decimal.digits, decimal.exponent)
}

/// Shortest digits that round-trip through `float16`, found by trying one to five digits.
fn half_shortest_digits(value: f64) -> (String, i32) {
    let target = F16::from_f64(value).0 & 0x7fff;
    let magnitude = value.abs();
    let text = (1..=5)
        .map(|digits| format!("{:.*e}", digits - 1, magnitude))
        .find(|text| {
            text.parse::<f64>()
                .is_ok_and(|parsed| F16::from_f64(parsed).0 & 0x7fff == target)
        })
        .unwrap_or_else(|| format!("{magnitude:e}"));
    let (mantissa, exponent) = text.split_once('e').expect("exponent notation");
    let digits = mantissa.replace('.', "");
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    (digits.to_string(), exponent.parse().expect("exponent"))
}

/// Python-style `repr` of a real value at the given precision: `1.5`, `2.0`, `1e-05`, `1e+20`,
/// `nan`, `inf`, `-0.0`.
pub(in crate::python) fn float_repr(value: f64, precision: Precision) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if value == 0.0 {
        return format!("{sign}0.0");
    }
    let (digits, exponent) = shortest_digits(value, precision);
    let upper = match precision {
        Precision::Half => 1e3,
        Precision::Single => 1e6,
        Precision::Double => 1e16,
    };
    let body = if (1e-4..upper).contains(&value.abs()) {
        if exponent < 0 {
            format!("0.{}{digits}", "0".repeat((-exponent - 1) as usize))
        } else if digits.len() as i32 > exponent + 1 {
            let split = (exponent + 1) as usize;
            format!("{}.{}", &digits[..split], &digits[split..])
        } else {
            format!(
                "{digits}{}.0",
                "0".repeat((exponent + 1) as usize - digits.len())
            )
        }
    } else {
        let mantissa = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!("{mantissa}e{exponent_sign}{:02}", exponent.abs())
    };
    format!("{sign}{body}")
}

/// Python-style `repr` of a complex value: `1j`, `(1+2j)`, `(-1.5+0j)`.
pub(in crate::python) fn complex_repr(real: f64, imag: f64, precision: Precision) -> String {
    let part = |value: f64| {
        let text = float_repr(value, precision);
        text.strip_suffix(".0").map(str::to_string).unwrap_or(text)
    };
    if real == 0.0 && !real.is_sign_negative() {
        return format!("{}j", part(imag));
    }
    let imag_text = part(imag);
    let joiner = if imag_text.starts_with('-') || imag.is_nan() && imag.is_sign_negative() {
        ""
    } else {
        "+"
    };
    format!("({}{joiner}{imag_text}j)", part(real))
}

/// The text `str()` gives for one scalar value of a numeric dtype.
pub(in crate::python) fn number_str(value: Number, precision: Precision) -> String {
    match value {
        Number::Bool(value) => if value { "True" } else { "False" }.to_string(),
        Number::Int(value) => value.to_string(),
        Number::UInt(value) => value.to_string(),
        Number::Float(value) => float_repr(value, precision),
        Number::Complex(real, imag) => complex_repr(real, imag, precision),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_use_python_layout_at_their_own_precision() {
        assert_eq!(float_repr(1.5, Precision::Double), "1.5");
        assert_eq!(float_repr(2.0, Precision::Double), "2.0");
        assert_eq!(float_repr(1e-5, Precision::Double), "1e-05");
        assert_eq!(float_repr(1e20, Precision::Double), "1e+20");
        assert_eq!(float_repr(1e16, Precision::Double), "1e+16");
        assert_eq!(float_repr(123456.0, Precision::Double), "123456.0");
        assert_eq!(float_repr(0.0001, Precision::Double), "0.0001");
        assert_eq!(float_repr(-0.0, Precision::Double), "-0.0");
        assert_eq!(float_repr(f64::from(0.1f32), Precision::Single), "0.1");
        assert_eq!(float_repr(0.099_975_585_937_5, Precision::Half), "0.1");
        assert_eq!(float_repr(65504.0, Precision::Half), "6.55e+04");
        assert_eq!(float_repr(2048.0, Precision::Half), "2.048e+03");
        assert_eq!(float_repr(999.0, Precision::Half), "999.0");
        assert_eq!(float_repr(16_777_216.0, Precision::Single), "1.6777216e+07");
        assert_eq!(float_repr(f64::from(0.0001f32), Precision::Single), "1e-04");
        assert_eq!(float_repr(1e15, Precision::Double), "1000000000000000.0");
    }

    #[test]
    fn complex_values_match_python() {
        assert_eq!(complex_repr(0.0, 1.0, Precision::Double), "1j");
        assert_eq!(complex_repr(1.0, 2.0, Precision::Double), "(1+2j)");
        assert_eq!(complex_repr(-1.5, 0.0, Precision::Double), "(-1.5+0j)");
        assert_eq!(complex_repr(1.0, -4.0, Precision::Double), "(1-4j)");
    }
}
