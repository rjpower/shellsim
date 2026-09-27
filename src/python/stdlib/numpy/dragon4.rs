//! Dragon4-equivalent float text: NumPy's positional and scientific formatting, with its
//! rounding, trimming and padding rules.
//!
//! [`positional`] and [`scientific`] generate decimal text for one finite value at the caller's
//! choice of digit count and layout, matching NumPy's `dragon4_positional`/`dragon4_scientific`
//! C functions. Python sees them as `np.format_float_positional`/`np.format_float_scientific`
//! (thin argument-validating wrappers in `numpy._arrayprint`) and array printing calls them once
//! per element with explicit padding so a column of numbers lines up.
//!
//! Digit generation has two regimes, chosen by `unique`:
//! - `unique = true` asks for the *shortest* decimal that round-trips back to the value at its
//!   own precision (half/single/double, see [`super::format`]), optionally capped by `precision`
//!   and/or extended by `min_digits`. When neither option changes the natural digit count, the
//!   shortest digits are reused directly; otherwise the value is correctly rounded to the
//!   resolved digit count, per NumPy's docs ("the last digit is rounded with unbiased rounding").
//! - `unique = false` always correctly rounds the exact value to exactly `precision` digits, as
//!   if printing infinitely many digits and stopping (NumPy's "IEEE unbiased", i.e.
//!   round-half-to-even, rounding).
//!
//! `precision`/`min_digits` count differently depending on the mode: positional's `fractional`
//! flag switches between digits after the decimal point (including leading zeros) and total
//! significant digits; scientific always counts digits after the leading digit. This module
//! never reimplements decimal rounding itself: fixed-digit-count text comes from Rust's
//! correctly-rounded float formatting (`{:.N$}`/`{:.N$e}`), and shortest text comes from the
//! round-trip search in [`crate::python::float_text`] and [`super::format`]. Trimming and padding
//! are string post-processing over the assembled integer/fraction/exponent parts.
//!
//! A generated digit count or padding width beyond [`MAX_DIGITS`] is rejected rather than
//! formatted, so `precision`, `min_digits`, `pad_left`, `pad_right` and `exp_digits` cannot force
//! an unbounded allocation; NumPy's own Dragon4 buffer is bounded similarly, if more tightly.

use super::super::super::native::{PyError, PyResult};
use super::format::{shortest_digits, Precision};
use crate::python::float_text;

/// Positional/scientific formatting options, in NumPy's `dragon4_positional`/`dragon4_scientific`
/// argument conventions: a negative integer field means "unset". Argument validation (negative
/// values a caller supplied explicitly, mutually inconsistent options) happens in
/// `numpy._arrayprint` before these options are built; by the time code here runs, a negative
/// field always means "not given".
#[derive(Clone, Copy)]
pub(in crate::python) struct Options {
    pub precision: i32,
    pub unique: bool,
    pub fractional: bool,
    pub sign: bool,
    pub trim: Trim,
    pub pad_left: i32,
    pub pad_right: i32,
    pub exp_digits: i32,
    pub min_digits: i32,
}

impl Options {
    pub(in crate::python) fn new() -> Self {
        Options {
            precision: -1,
            unique: true,
            fractional: false,
            sign: false,
            trim: Trim::Keep,
            pad_left: -1,
            pad_right: -1,
            exp_digits: -1,
            min_digits: -1,
        }
    }
}

/// Trailing-digit trimming, from NumPy's `trim` codes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::python) enum Trim {
    /// `'k'`: keep trailing zeros and the decimal point.
    Keep,
    /// `'.'`: drop trailing zeros, keep the decimal point.
    Fraction,
    /// `'0'`: drop trailing zeros but leave one digit after the decimal point.
    Zero,
    /// `'-'`: drop trailing zeros and the decimal point itself if nothing follows it.
    Bare,
}

impl Trim {
    pub(in crate::python) fn parse(code: &str) -> Option<Self> {
        match code {
            "k" => Some(Trim::Keep),
            "." => Some(Trim::Fraction),
            "0" => Some(Trim::Zero),
            "-" => Some(Trim::Bare),
            _ => None,
        }
    }
}

/// A digit count or padding width beyond this is rejected. Order-of-magnitude match for NumPy's
/// own internal formatting buffer; nothing in the supported surface needs anywhere near this
/// many digits (the largest legitimate request is the ~1074-digit exact expansion of the
/// smallest subnormal).
const MAX_DIGITS: i32 = 4096;

/// `value < 0` means "unset" (`None`); otherwise the value, rejecting anything past
/// [`MAX_DIGITS`] before any formatting work happens.
fn bounded(value: i32, label: &str) -> PyResult<Option<i64>> {
    if value < 0 {
        return Ok(None);
    }
    if value > MAX_DIGITS {
        return Err(PyError::resource_error(format!(
            "{label} exceeds shellsim's float formatting limit of {MAX_DIGITS}"
        )));
    }
    Ok(Some(i64::from(value)))
}

/// The natural (shortest round-trip) digits and decimal exponent of a value's magnitude, `0.0`
/// included: `magnitude == digits[0].digits[1..] × 10^exponent`. [`shortest_digits`] only
/// accepts nonzero values, so zero is synthesized directly.
fn natural(magnitude: f64, precision: Precision) -> (String, i32) {
    if magnitude == 0.0 {
        return ("0".to_string(), 0);
    }
    shortest_digits(magnitude, precision)
}

/// Clip a natural digit count to at most `cap` and at least `floor`, never below `minimum`.
fn resolve_target(natural: i64, cap: Option<i64>, floor: Option<i64>, minimum: i64) -> i64 {
    let mut target = natural;
    if let Some(cap) = cap {
        target = target.min(cap);
    }
    if let Some(floor) = floor {
        target = target.max(floor);
    }
    target.max(minimum)
}

/// Digits and exponent holding exactly `significant` significant digits of `magnitude`: the
/// natural digits when they already have that count and `unique` allows reusing them (this keeps
/// NumPy's round-trip guarantee, which can differ from naive correct rounding for a lopsided
/// rounding interval, see [`crate::python::float_text`]), otherwise the magnitude correctly
/// rounded fresh to that many digits.
fn digits_at(
    magnitude: f64,
    natural_digits: &str,
    natural_exponent: i32,
    unique: bool,
    significant: i64,
) -> (String, i32) {
    if unique && significant == natural_digits.len() as i64 {
        (natural_digits.to_string(), natural_exponent)
    } else {
        float_text::fixed_digits(magnitude, significant.max(1) as usize)
    }
}

/// Split scientific-normalized digits into a positional integer part and fraction part, padding
/// with zeros on whichever side the exponent demands: `123` with `exponent = 4` (i.e.
/// `1.23×10^4`) is `("123", "00")`.
fn split_positional(digits: &str, exponent: i32) -> (String, String) {
    if exponent >= 0 {
        let point = (exponent + 1) as usize;
        if digits.len() >= point {
            (digits[..point].to_string(), digits[point..].to_string())
        } else {
            let zeros = "0".repeat(point - digits.len());
            (format!("{digits}{zeros}"), String::new())
        }
    } else {
        let zeros = "0".repeat((-exponent - 1) as usize);
        ("0".to_string(), format!("{zeros}{digits}"))
    }
}

/// Post-process a fraction digit string per NumPy's `trim` option. `None` means the decimal
/// point itself should be omitted (`Trim::Bare` with nothing left after trimming).
fn trim_fraction(fraction: &str, mode: Trim) -> Option<String> {
    match mode {
        Trim::Keep => Some(fraction.to_string()),
        Trim::Fraction => Some(fraction.trim_end_matches('0').to_string()),
        Trim::Zero => {
            let trimmed = fraction.trim_end_matches('0');
            Some(if trimmed.is_empty() { "0" } else { trimmed }.to_string())
        }
        Trim::Bare => {
            let trimmed = fraction.trim_end_matches('0');
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
    }
}

/// `nan`/`inf` text, unaffected by every other option except `sign` (NaN never carries a sign).
fn special_text(value: f64, show_sign: bool) -> Option<String> {
    if value.is_nan() {
        return Some("nan".to_string());
    }
    value.is_infinite().then(|| {
        let sign = if value < 0.0 {
            "-"
        } else if show_sign {
            "+"
        } else {
            ""
        };
        format!("{sign}inf")
    })
}

/// Assemble sign, integer part and left padding, then a decimal point/fraction and right
/// padding. `fraction: None` means trimming removed the decimal point entirely; the column it
/// would have occupied still counts toward `pad_right`, as whitespace.
fn assemble(
    negative: bool,
    show_sign: bool,
    integer: &str,
    fraction: Option<String>,
    pad_left: Option<i64>,
    pad_right: Option<i64>,
) -> String {
    let sign = if negative {
        "-"
    } else if show_sign {
        "+"
    } else {
        ""
    };
    let left = format!("{sign}{integer}");
    let left_width = pad_left
        .map_or(0, |width| width as usize)
        .max(left.chars().count());

    let right_content = fraction.map(|digits| format!(".{digits}"));
    let natural_right_width = right_content.as_deref().map_or(0, str::len);
    let right_width = pad_right
        .map_or(0, |width| width as usize + 1)
        .max(natural_right_width);

    format!(
        "{left:>left_width$}{:<right_width$}",
        right_content.unwrap_or_default()
    )
}

/// `dragon4_positional`: decimal text in positional (non-exponential) notation. See the module
/// documentation for the digit-generation regimes; `options.fractional` chooses whether
/// `precision`/`min_digits` count fractional digits or total significant digits.
pub(in crate::python) fn positional(
    value: f64,
    precision: Precision,
    options: &Options,
    work: &mut u64,
) -> PyResult<String> {
    if let Some(text) = special_text(value, options.sign) {
        *work += text.len() as u64;
        return Ok(text);
    }
    let cap = bounded(options.precision, "precision")?;
    let floor = bounded(options.min_digits, "min_digits")?;
    let pad_left = bounded(options.pad_left, "pad_left")?;
    let pad_right = bounded(options.pad_right, "pad_right")?;

    let magnitude = value.abs();
    let (natural_digits, natural_exponent) = natural(magnitude, precision);

    let (integer, fraction) = if options.fractional {
        let natural_frac = (natural_digits.len() as i64 - i64::from(natural_exponent) - 1).max(0);
        let target = if options.unique {
            resolve_target(natural_frac, cap, floor, 0)
        } else {
            cap.expect("numpy._arrayprint requires precision when unique is False")
        };
        if options.unique && target == natural_frac {
            split_positional(&natural_digits, natural_exponent)
        } else {
            let text = format!("{magnitude:.*}", target as usize);
            let (integer, fraction) = text.split_once('.').expect("fixed-point notation");
            (integer.to_string(), fraction.to_string())
        }
    } else {
        let natural_sig = natural_digits.len() as i64;
        let target = if options.unique {
            resolve_target(natural_sig, cap, floor, 1)
        } else {
            cap.expect("numpy._arrayprint requires precision when unique is False")
        };
        let (digits, exponent) = digits_at(
            magnitude,
            &natural_digits,
            natural_exponent,
            options.unique,
            target,
        );
        split_positional(&digits, exponent)
    };

    let fraction = trim_fraction(&fraction, options.trim);
    let text = assemble(
        value.is_sign_negative(),
        options.sign,
        &integer,
        fraction,
        pad_left,
        pad_right,
    );
    *work += text.len() as u64 + 8;
    Ok(text)
}

/// `dragon4_scientific`: decimal text in scientific (exponential) notation, always with exactly
/// one digit before the decimal point and an `e±EE` suffix.
pub(in crate::python) fn scientific(
    value: f64,
    precision: Precision,
    options: &Options,
    work: &mut u64,
) -> PyResult<String> {
    if let Some(text) = special_text(value, options.sign) {
        *work += text.len() as u64;
        return Ok(text);
    }
    let cap = bounded(options.precision, "precision")?;
    let floor = bounded(options.min_digits, "min_digits")?;
    let pad_left = bounded(options.pad_left, "pad_left")?;
    let exp_digits = bounded(options.exp_digits, "exp_digits")?;

    let magnitude = value.abs();
    let (natural_digits, natural_exponent) = natural(magnitude, precision);
    let natural_fraction = natural_digits.len() as i64 - 1;
    let target_fraction = if options.unique {
        resolve_target(natural_fraction, cap, floor, 0)
    } else {
        cap.expect("numpy._arrayprint requires precision when unique is False")
    };
    let (digits, exponent) = digits_at(
        magnitude,
        &natural_digits,
        natural_exponent,
        options.unique,
        target_fraction + 1,
    );
    let (leading, fraction) = digits.split_at(1);

    let fraction = trim_fraction(fraction, options.trim);
    let mantissa = assemble(
        value.is_sign_negative(),
        options.sign,
        leading,
        fraction,
        pad_left,
        None,
    );
    let exp_width = exp_digits.map_or(2, |width| width as usize).max(2);
    let exp_sign = if exponent < 0 { '-' } else { '+' };
    let text = format!(
        "{mantissa}e{exp_sign}{:0exp_width$}",
        exponent.unsigned_abs()
    );
    *work += text.len() as u64 + 8;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn positional_text(value: f64, options: Options) -> String {
        let mut work = 0;
        positional(value, Precision::Double, &options, &mut work).unwrap()
    }

    fn scientific_text(value: f64, options: Options) -> String {
        let mut work = 0;
        scientific(value, Precision::Double, &options, &mut work).unwrap()
    }

    // Expected text comes from NumPy 2.5.3 (`np.format_float_positional`/`_scientific`).
    #[test]
    fn positional_matches_numpy_defaults() {
        assert_eq!(positional_text(1.0, Options::new()), "1.");
        assert_eq!(positional_text(0.0, Options::new()), "0.");
        assert_eq!(positional_text(-0.0, Options::new()), "-0.");
        assert_eq!(
            positional_text(
                0.1,
                Options {
                    precision: 20,
                    unique: false,
                    fractional: true,
                    ..Options::new()
                }
            ),
            "0.10000000000000000555"
        );
    }

    #[test]
    fn positional_pads_left_and_right_of_the_point() {
        assert_eq!(
            positional_text(
                1.5,
                Options {
                    fractional: true,
                    pad_left: 3,
                    pad_right: 4,
                    ..Options::new()
                }
            ),
            "  1.5   "
        );
        assert_eq!(
            positional_text(
                -1.5,
                Options {
                    fractional: true,
                    pad_left: 3,
                    ..Options::new()
                }
            ),
            " -1.5"
        );
    }

    #[test]
    fn positional_trim_modes() {
        assert_eq!(positional_text(2.0, Options::new()), "2.");
        assert_eq!(
            positional_text(
                2.0,
                Options {
                    trim: Trim::Bare,
                    ..Options::new()
                }
            ),
            "2"
        );
        let min3 = Options {
            fractional: true,
            min_digits: 3,
            ..Options::new()
        };
        assert_eq!(positional_text(2.0, min3), "2.000");
        assert_eq!(
            positional_text(
                2.0,
                Options {
                    trim: Trim::Fraction,
                    ..min3
                }
            ),
            "2."
        );
        assert_eq!(
            positional_text(
                2.0,
                Options {
                    trim: Trim::Zero,
                    ..min3
                }
            ),
            "2.0"
        );
    }

    #[test]
    fn positional_precision_caps_unique_digits_but_does_not_pad() {
        // 0.1's shortest form has one fractional digit, well under the precision cap.
        assert_eq!(
            positional_text(
                0.1,
                Options {
                    fractional: true,
                    precision: 8,
                    ..Options::new()
                }
            ),
            "0.1"
        );
    }

    #[test]
    fn positional_significant_digit_mode_rounds_the_integer_part() {
        let options = Options {
            precision: 3,
            unique: false,
            fractional: false,
            ..Options::new()
        };
        assert_eq!(positional_text(12345.0, options), "12300.");
        assert_eq!(positional_text(99.95, options), "100.");
    }

    #[test]
    fn scientific_matches_numpy_examples() {
        assert_eq!(scientific_text(0.0, Options::new()), "0.e+00");
        assert_eq!(
            scientific_text(
                1.23e24,
                Options {
                    exp_digits: 4,
                    ..Options::new()
                }
            ),
            "1.23e+0024"
        );
        assert_eq!(scientific_text(5e-324, Options::new()), "5.e-324");
    }

    #[test]
    fn special_values_ignore_padding() {
        assert_eq!(
            positional_text(
                f64::INFINITY,
                Options {
                    pad_left: 6,
                    ..Options::new()
                }
            ),
            "inf"
        );
        assert_eq!(positional_text(f64::NAN, Options::new()), "nan");
        assert_eq!(
            positional_text(
                f64::NAN,
                Options {
                    sign: true,
                    ..Options::new()
                }
            ),
            "nan"
        );
    }

    #[test]
    fn oversized_requests_are_rejected_before_formatting() {
        let mut work = 0;
        let error = positional(
            1.5,
            Precision::Double,
            &Options {
                min_digits: MAX_DIGITS + 1,
                ..Options::new()
            },
            &mut work,
        )
        .unwrap_err();
        assert_eq!(
            error.message,
            "min_digits exceeds shellsim's float formatting limit of 4096"
        );
    }
}
