//! NumPy's Dragon4 float formatter: `dragon4_positional` and `dragon4_scientific`.
//!
//! This is a port of `numpy/_core/src/multiarray/dragon4.c` (itself derived from Ryan
//! Juckett's implementation of Steele & White / Burger & Dybvig). Array printing and
//! `np.format_float_positional`/`np.format_float_scientific` need its exact digit choices:
//! in unique mode it prints the shortest digits that identify the value among values of its
//! own width (half, single, or double), stops at a precision cutoff with round-half-even on the
//! exact binary value, and can continue past the shortest digits up to `min_digits`, where the
//! extra digits come from the exact binary expansion rather than zero padding. The trimming
//! and padding rules then reproduce NumPy's `trim`, `pad_left`, `pad_right` and `exp_digits`.
//!
//! Arithmetic uses arbitrary-precision integers, so every finite value is exact. Output is
//! capped at NumPy's 16384-byte buffer, which bounds the work any caller-supplied precision
//! can request. Each call adds its work (digits generated times the big-integer width) to a
//! caller-owned counter, which callers charge as CPU after every value.

use num_bigint::BigUint;
use num_traits::Zero;

use super::super::super::native::{PyError, PyResult};
use super::element::F16;
use super::format::Precision;

/// NumPy's scratch buffer size, which caps every formatted value.
const BUFFER_SIZE: i32 = 16384;

/// How trailing zeros and the decimal point are trimmed (NumPy's `trim` argument).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum Trim {
    /// `'k'`: keep trailing zeros and the decimal point.
    Keep,
    /// `'.'`: trim trailing zeros, keep the decimal point.
    Zeros,
    /// `'0'`: trim trailing zeros but keep one zero after the decimal point.
    LeaveOneZero,
    /// `'-'`: trim trailing zeros and the decimal point.
    DropPoint,
}

impl Trim {
    /// Parse NumPy's one-character trim code.
    pub(in crate::python) fn parse(code: &str) -> Option<Self> {
        Some(match code {
            "k" => Self::Keep,
            "." => Self::Zeros,
            "0" => Self::LeaveOneZero,
            "-" => Self::DropPoint,
            _ => return None,
        })
    }
}

/// Options shared by the positional and scientific formats. Negative numbers mean "unset",
/// as in NumPy's C interface.
#[derive(Clone, Copy, Debug)]
pub(in crate::python) struct Options {
    /// Shortest-unique digits (`true`) or the exact expansion cut at `precision` (`false`).
    pub unique: bool,
    /// Positional only: `precision` and `min_digits` count digits after the point rather than
    /// significant digits.
    pub fractional: bool,
    pub precision: i32,
    pub min_digits: i32,
    /// Print `+` for non-negative values.
    pub sign: bool,
    pub trim: Trim,
    /// Pad with spaces until this many characters precede the decimal point.
    pub pad_left: i32,
    /// Positional only: pad with spaces until this many characters follow the decimal point.
    pub pad_right: i32,
    /// Scientific only: minimum exponent digits; negative means 2.
    pub exp_digits: i32,
}

impl Options {
    /// Unique digits with no cutoff, padding, or trimming, like NumPy's defaults.
    pub(in crate::python) const fn new() -> Self {
        Self {
            unique: true,
            fractional: true,
            precision: -1,
            min_digits: -1,
            sign: false,
            trim: Trim::Keep,
            pad_left: -1,
            pad_right: -1,
            exp_digits: -1,
        }
    }
}

/// A finite value as `mantissa × 2^exponent`, with the facts Dragon4 needs about its format.
struct Parts {
    mantissa: u64,
    exponent: i32,
    /// Index of the highest set mantissa bit.
    mantissa_bit: u32,
    /// Whether the gap to the next value above is twice the gap below (a power of two).
    unequal_margins: bool,
}

enum Decoded {
    Finite(Parts),
    Infinite,
    NaN,
}

/// Split `value`, read at `precision`, into sign and parts. The value must already be exactly
/// representable at that precision, as elements of a `float16` or `float32` array are.
fn decode(value: f64, precision: Precision) -> (bool, Decoded) {
    let (bits, fraction_bits, exponent_bits, bias) = match precision {
        Precision::Half => (u64::from(F16::from_f64(value).0), 10, 5, 15),
        Precision::Single => (u64::from((value as f32).to_bits()), 23, 8, 127),
        Precision::Double => (value.to_bits(), 52, 11, 1023),
    };
    let negative = bits >> (fraction_bits + exponent_bits) != 0;
    let fraction = bits & ((1 << fraction_bits) - 1);
    let biased = ((bits >> fraction_bits) & ((1 << exponent_bits) - 1)) as i32;
    if biased == (1 << exponent_bits) - 1 {
        let decoded = if fraction == 0 {
            Decoded::Infinite
        } else {
            Decoded::NaN
        };
        return (negative, decoded);
    }
    let parts = if biased != 0 {
        Parts {
            mantissa: (1 << fraction_bits) | fraction,
            exponent: biased - bias - fraction_bits as i32,
            mantissa_bit: fraction_bits,
            unequal_margins: biased != 1 && fraction == 0,
        }
    } else {
        Parts {
            mantissa: fraction,
            exponent: 1 - bias - fraction_bits as i32,
            mantissa_bit: fraction.checked_ilog2().unwrap_or(0),
            unequal_margins: false,
        }
    };
    (negative, Decoded::Finite(parts))
}

/// Text for infinities and NaNs; NaN never shows a sign.
fn special(decoded: &Decoded, sign: Option<u8>) -> Option<String> {
    match decoded {
        Decoded::Finite(_) => None,
        Decoded::NaN => Some("nan".to_string()),
        Decoded::Infinite => {
            let sign = sign.map(char::from).map(String::from).unwrap_or_default();
            Some(format!("{sign}inf"))
        }
    }
}

/// The digits Dragon4 generates and the decimal exponent of the first one.
///
/// `cutoff_max` stops generation after that many digits (total or fractional, per
/// `total_length`); `cutoff_min` keeps generating past the unique point until that many digits
/// exist. `buffer_size` caps the digit count. The loop's cost is added to `work`.
fn dragon4(
    parts: &Parts,
    unique: bool,
    total_length: bool,
    cutoff_max: i32,
    cutoff_min: i32,
    buffer_size: i32,
    work: &mut u64,
) -> (Vec<u8>, i32) {
    if parts.mantissa == 0 {
        return (vec![b'0'], 0);
    }
    let is_even = parts.mantissa.is_multiple_of(2);
    let exponent = parts.exponent;
    let unequal = parts.unequal_margins;
    // value = scaled_value / scale; the margins are half the gaps to the neighbouring floats.
    let mut scaled_value = BigUint::from(parts.mantissa);
    let mut scale;
    let mut margin_low;
    let shift = if unequal { 2 } else { 1 };
    if exponent > 0 {
        scaled_value <<= (exponent + shift) as usize;
        scale = BigUint::from(1u32 << shift);
        margin_low = BigUint::from(1u32) << exponent as usize;
    } else {
        scaled_value <<= shift as usize;
        scale = BigUint::from(1u32) << (shift - exponent) as usize;
        margin_low = BigUint::from(1u32);
    }
    let high_margin = |low: &BigUint| if unequal { low << 1usize } else { low.clone() };
    let mut margin_high = high_margin(&margin_low);

    // An estimate of floor(log10(value)) + 1 that is correct or one too small.
    let mut digit_exponent =
        (f64::from(parts.mantissa_bit as i32 + exponent) * std::f64::consts::LOG10_2 - 0.69).ceil()
            as i32;
    if cutoff_max >= 0 && !total_length && digit_exponent <= -cutoff_max {
        digit_exponent = -cutoff_max + 1;
    }
    if digit_exponent > 0 {
        scale *= BigUint::from(10u32).pow(digit_exponent as u32);
    } else if digit_exponent < 0 {
        let power = BigUint::from(10u32).pow(digit_exponent.unsigned_abs());
        scaled_value *= &power;
        margin_low *= &power;
        margin_high = high_margin(&margin_low);
    }
    if scaled_value >= scale {
        digit_exponent += 1;
    } else {
        scaled_value *= 10u32;
        margin_low *= 10u32;
        margin_high = high_margin(&margin_low);
    }

    let mut cutoff_max_exponent = digit_exponent - buffer_size;
    if cutoff_max >= 0 {
        let desired = if total_length {
            digit_exponent - cutoff_max
        } else {
            -cutoff_max
        };
        cutoff_max_exponent = cutoff_max_exponent.max(desired);
    }
    let mut cutoff_min_exponent = digit_exponent;
    if cutoff_min >= 0 {
        let desired = if total_length {
            digit_exponent - cutoff_min
        } else {
            -cutoff_min
        };
        cutoff_min_exponent = cutoff_min_exponent.min(desired);
    }
    let mut out_exponent = digit_exponent - 1;

    let mut digits = Vec::new();
    let (mut low, mut high) = (false, false);
    let mut output_digit;
    let word_cost = 1 + scale.bits() / 64;
    loop {
        *work = work.saturating_add(word_cost);
        digit_exponent -= 1;
        output_digit = 0u8;
        while scaled_value >= scale {
            scaled_value -= &scale;
            output_digit += 1;
        }
        if unique {
            let value_high = &scaled_value + &margin_high;
            let compare_low = scaled_value.cmp(&margin_low);
            low = if is_even {
                compare_low.is_le()
            } else {
                compare_low.is_lt()
            };
            let compare_high = value_high.cmp(&scale);
            high = if is_even {
                compare_high.is_ge()
            } else {
                compare_high.is_gt()
            };
            if ((low || high) && digit_exponent <= cutoff_min_exponent)
                || digit_exponent == cutoff_max_exponent
            {
                break;
            }
        } else if scaled_value.is_zero() || digit_exponent == cutoff_max_exponent {
            break;
        }
        digits.push(b'0' + output_digit);
        scaled_value *= 10u32;
        if unique {
            margin_low *= 10u32;
            margin_high = high_margin(&margin_low);
        }
    }

    // Round the final digit: toward the closer neighbour, or half to even when both are legal.
    let mut round_down = low;
    if low == high {
        scaled_value <<= 1usize;
        let compare = scaled_value.cmp(&scale);
        round_down = compare.is_lt() || (compare.is_eq() && output_digit % 2 == 0);
    }
    if round_down {
        digits.push(b'0' + output_digit);
    } else if output_digit == 9 {
        // Carry into the previous digits, dropping the nines that become zeros.
        loop {
            match digits.pop() {
                None => {
                    digits.push(b'1');
                    out_exponent += 1;
                    break;
                }
                Some(b'9') => {}
                Some(digit) => {
                    digits.push(digit + 1);
                    break;
                }
            }
        }
    } else {
        digits.push(b'0' + output_digit + 1);
    }
    (digits, out_exponent)
}

fn too_large() -> PyError {
    PyError::runtime_error("Float formatting result too large")
}

fn sign_byte(negative: bool, options: &Options) -> Option<u8> {
    if negative {
        Some(b'-')
    } else if options.sign {
        Some(b'+')
    } else {
        None
    }
}

/// NumPy's `dragon4_positional`: `ddd.ddd` with the requested cutoff, trimming, and padding.
///
/// ```ignore
/// let options = Options { precision: 3, ..Options::new() };
/// assert_eq!(positional(1.0 / 3.0, Precision::Double, &options, &mut work)?, "0.333");
/// ```
pub(in crate::python) fn positional(
    value: f64,
    precision: Precision,
    options: &Options,
    work: &mut u64,
) -> PyResult<String> {
    let (negative, decoded) = decode(value, precision);
    let sign = sign_byte(negative, options);
    if let Some(text) = special(&decoded, sign) {
        return Ok(text);
    }
    let Decoded::Finite(parts) = decoded else {
        unreachable!("special values returned above")
    };
    let max_print_len = BUFFER_SIZE - 1;
    let has_sign = i32::from(sign.is_some());
    let mut buffer: Vec<u8> = sign.into_iter().collect();
    let (digits, print_exponent) = dragon4(
        &parts,
        options.unique,
        !options.fractional,
        options.precision,
        options.min_digits,
        max_print_len - has_sign,
        work,
    );
    let num_digits = digits.len() as i32;
    let num_whole_digits;
    let mut num_fraction_digits = 0;
    if print_exponent >= 0 {
        num_whole_digits = print_exponent + 1;
        if num_digits <= num_whole_digits {
            let count = num_whole_digits - num_digits;
            if count > max_print_len - (has_sign + num_digits) {
                return Err(too_large());
            }
            buffer.extend_from_slice(&digits);
            buffer.resize(buffer.len() + count as usize, b'0');
        } else {
            let max_fraction_digits = max_print_len - num_whole_digits - 1 - has_sign;
            num_fraction_digits = (num_digits - num_whole_digits).min(max_fraction_digits);
            let whole = num_whole_digits as usize;
            buffer.extend_from_slice(&digits[..whole]);
            buffer.push(b'.');
            buffer.extend_from_slice(&digits[whole..whole + num_fraction_digits as usize]);
        }
    } else {
        let max_fraction_zeros = max_print_len - 2 - has_sign;
        let fraction_zeros = (-(print_exponent + 1)).min(max_fraction_zeros);
        let shown = num_digits.min(max_print_len - 2 - fraction_zeros - has_sign);
        buffer.extend_from_slice(b"0.");
        buffer.resize(buffer.len() + fraction_zeros as usize, b'0');
        buffer.extend_from_slice(&digits[..shown as usize]);
        num_fraction_digits = shown + fraction_zeros;
        num_whole_digits = 1;
    }
    let room = |buffer: &Vec<u8>| (buffer.len() as i32) < max_print_len;
    if options.trim != Trim::DropPoint && num_fraction_digits == 0 && room(&buffer) {
        buffer.push(b'.');
    }
    let add_digits = if options.unique {
        options.min_digits
    } else {
        options.precision
    };
    let desired_fraction_digits = if options.fractional {
        add_digits.max(0)
    } else {
        add_digits - num_whole_digits
    };
    if options.trim == Trim::LeaveOneZero {
        if num_fraction_digits == 0 && room(&buffer) {
            buffer.push(b'0');
            num_fraction_digits += 1;
        }
    } else if options.trim == Trim::Keep
        && desired_fraction_digits > num_fraction_digits
        && room(&buffer)
    {
        let count = desired_fraction_digits - num_fraction_digits;
        if count > max_print_len - buffer.len() as i32 {
            return Err(too_large());
        }
        num_fraction_digits += count;
        buffer.resize(buffer.len() + count as usize, b'0');
    }
    if options.trim != Trim::Keep && num_fraction_digits > 0 {
        while buffer.last() == Some(&b'0') {
            buffer.pop();
            num_fraction_digits -= 1;
        }
        if buffer.last() == Some(&b'.') {
            match options.trim {
                Trim::LeaveOneZero => {
                    buffer.push(b'0');
                    num_fraction_digits += 1;
                }
                Trim::DropPoint => {
                    buffer.pop();
                }
                Trim::Keep | Trim::Zeros => {}
            }
        }
    }
    if options.pad_right >= num_fraction_digits {
        let count = options.pad_right - num_fraction_digits;
        if options.trim == Trim::DropPoint && num_fraction_digits == 0 && room(&buffer) {
            buffer.push(b' ');
        }
        if count > max_print_len - buffer.len() as i32 {
            return Err(too_large());
        }
        buffer.resize(buffer.len() + count as usize, b' ');
    }
    if options.pad_left > num_whole_digits + has_sign {
        let shift = options.pad_left - (num_whole_digits + has_sign);
        if buffer.len() as i32 > max_print_len - shift {
            return Err(too_large());
        }
        buffer.splice(0..0, std::iter::repeat_n(b' ', shift as usize));
    }
    Ok(String::from_utf8(buffer).expect("formatted floats are ASCII"))
}

/// NumPy's `dragon4_scientific`: `d.ddde+XX` with the requested precision, trimming, left
/// padding, and exponent width.
pub(in crate::python) fn scientific(
    value: f64,
    precision: Precision,
    options: &Options,
    work: &mut u64,
) -> String {
    let (negative, decoded) = decode(value, precision);
    let sign = sign_byte(negative, options);
    if let Some(text) = special(&decoded, sign) {
        return text;
    }
    let Decoded::Finite(parts) = decoded else {
        unreachable!("special values returned above")
    };
    let mut buffer = Vec::new();
    let mut buffer_size = BUFFER_SIZE;
    let left_chars = 1 + i32::from(sign.is_some());
    if options.pad_left > left_chars {
        for _ in 0..options.pad_left - left_chars {
            if buffer_size <= 1 {
                break;
            }
            buffer.push(b' ');
            buffer_size -= 1;
        }
    }
    if let Some(sign) = sign.filter(|_| buffer_size > 1) {
        buffer.push(sign);
        buffer_size -= 1;
    }
    let plus_one = |digits: i32| if digits < 0 { -1 } else { digits + 1 };
    let (digits, print_exponent) = dragon4(
        &parts,
        options.unique,
        true,
        plus_one(options.precision),
        plus_one(options.min_digits),
        buffer_size,
        work,
    );
    buffer.push(digits[0]);
    if buffer_size > 1 {
        buffer_size -= 1;
    }
    let mut num_fraction_digits = digits.len() as i32 - 1;
    if num_fraction_digits > 0 && buffer_size > 1 {
        num_fraction_digits = num_fraction_digits.min(buffer_size - 2);
        buffer.push(b'.');
        buffer.extend_from_slice(&digits[1..1 + num_fraction_digits as usize]);
        buffer_size -= 1 + num_fraction_digits;
    }
    if options.trim != Trim::DropPoint && num_fraction_digits == 0 && buffer_size > 1 {
        buffer.push(b'.');
        buffer_size -= 1;
    }
    let add_digits = if options.unique {
        options.min_digits
    } else {
        options.precision
    }
    .max(0);
    if options.trim == Trim::LeaveOneZero {
        if num_fraction_digits == 0 && buffer_size > 1 {
            buffer.push(b'0');
            buffer_size -= 1;
            num_fraction_digits += 1;
        }
    } else if options.trim == Trim::Keep && add_digits > num_fraction_digits {
        let zeros = (add_digits - num_fraction_digits).min(buffer_size - 1);
        buffer.resize(buffer.len() + zeros.max(0) as usize, b'0');
        num_fraction_digits += zeros;
    }
    if options.trim != Trim::Keep && num_fraction_digits > 0 {
        while buffer.last() == Some(&b'0') {
            buffer.pop();
            buffer_size += 1;
        }
        if options.trim == Trim::LeaveOneZero && buffer.last() == Some(&b'.') {
            buffer.push(b'0');
            buffer_size -= 1;
        }
    }
    if buffer_size > 1 {
        let exp_digits = match options.exp_digits {
            digits if digits < 0 => 2,
            digits => digits.min(5),
        } as usize;
        buffer.push(b'e');
        buffer.push(if print_exponent >= 0 { b'+' } else { b'-' });
        let magnitude = print_exponent.unsigned_abs().to_string();
        let width = magnitude.len().max(exp_digits);
        // An exponent of zero with `exp_digits=0` prints no digits, as in NumPy.
        let text = if print_exponent == 0 && exp_digits == 0 {
            String::new()
        } else {
            format!("{magnitude:0>width$}")
        };
        buffer.extend_from_slice(text.as_bytes());
    }
    String::from_utf8(buffer).expect("formatted floats are ASCII")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique(precision: i32, trim: Trim) -> Options {
        Options {
            precision,
            trim,
            ..Options::new()
        }
    }

    #[test]
    fn positional_matches_numpy() {
        let double = Precision::Double;
        assert_eq!(
            positional(0.1, double, &Options::new(), &mut 0).unwrap(),
            "0.1"
        );
        assert_eq!(
            positional(1.0, double, &Options::new(), &mut 0).unwrap(),
            "1."
        );
        assert_eq!(
            positional(1.0 / 3.0, double, &unique(8, Trim::Zeros), &mut 0).unwrap(),
            "0.33333333"
        );
        assert_eq!(
            positional(2.0 / 3.0, double, &unique(8, Trim::Zeros), &mut 0).unwrap(),
            "0.66666667"
        );
        assert_eq!(
            positional(0.125, double, &unique(2, Trim::Zeros), &mut 0).unwrap(),
            "0.12"
        );
        assert_eq!(
            positional(0.375, double, &unique(2, Trim::Zeros), &mut 0).unwrap(),
            "0.38"
        );
        assert_eq!(
            positional(9.999, double, &unique(2, Trim::Zeros), &mut 0).unwrap(),
            "10."
        );
        assert_eq!(
            positional(-0.0, double, &Options::new(), &mut 0).unwrap(),
            "-0."
        );
        assert_eq!(
            positional(1e-5, double, &unique(8, Trim::Zeros), &mut 0).unwrap(),
            "0.00001"
        );
        assert_eq!(
            positional(1e-10, double, &unique(8, Trim::Zeros), &mut 0).unwrap(),
            "0."
        );
        assert_eq!(
            positional(
                f64::from(0.1f32),
                Precision::Single,
                &Options::new(),
                &mut 0
            )
            .unwrap(),
            "0.1"
        );
        let padded = Options {
            pad_left: 3,
            pad_right: 4,
            ..unique(8, Trim::Zeros)
        };
        assert_eq!(
            positional(1.5, double, &padded, &mut 0).unwrap(),
            "  1.5   "
        );
        let exact = Options {
            unique: false,
            ..unique(20, Trim::Keep)
        };
        assert_eq!(
            positional(0.1, double, &exact, &mut 0).unwrap(),
            "0.10000000000000000555"
        );
        let min_digits = Options {
            min_digits: 20,
            ..Options::new()
        };
        assert_eq!(
            positional(0.1, double, &min_digits, &mut 0).unwrap(),
            "0.10000000000000000555"
        );
    }

    #[test]
    fn scientific_matches_numpy() {
        let double = Precision::Double;
        assert_eq!(
            scientific(1e-5, double, &unique(8, Trim::Zeros), &mut 0),
            "1.e-05"
        );
        assert_eq!(
            scientific(1.1e-4, double, &unique(8, Trim::Zeros), &mut 0),
            "1.1e-04"
        );
        assert_eq!(
            scientific(123_456_789.0, double, &unique(8, Trim::Zeros), &mut 0),
            "1.23456789e+08"
        );
        assert_eq!(
            scientific(1e300, double, &unique(8, Trim::Zeros), &mut 0),
            "1.e+300"
        );
        assert_eq!(
            scientific(5e-324, double, &Options::new(), &mut 0),
            "5.e-324"
        );
        let fixed = Options {
            min_digits: 3,
            pad_left: 2,
            exp_digits: 2,
            ..unique(3, Trim::Keep)
        };
        assert_eq!(scientific(1.0, double, &fixed, &mut 0), " 1.000e+00");
        assert_eq!(scientific(f64::NAN, double, &Options::new(), &mut 0), "nan");
        assert_eq!(
            scientific(f64::NEG_INFINITY, double, &Options::new(), &mut 0),
            "-inf"
        );
        assert_eq!(
            scientific(2048.0, Precision::Half, &Options::new(), &mut 0),
            "2.048e+03"
        );
    }
}
