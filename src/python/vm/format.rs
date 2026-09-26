//! Python's format-specification mini-language for f-strings, `format`, and `str.format`.
//!
//! A specification is parsed once into [`FormatSpec`], then rendered for an integer, float, or
//! string. The supported grammar is `[align][sign][#][0][width][,][.precision][type]` with the
//! default space fill. Custom fill characters, `=` alignment, the space sign, `_` grouping, and
//! the `n`/`c` presentations are rejected explicitly rather than approximated.

use super::BigInt;
use num_bigint::Sign;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Align {
    Left,
    Right,
    Center,
}

/// A parsed format specification. Widths and precisions are bounded by the caller's resource
/// reservation before rendering allocates padding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct FormatSpec {
    pub(super) align: Option<Align>,
    pub(super) plus: bool,
    pub(super) alternate: bool,
    pub(super) zero: bool,
    pub(super) width: usize,
    pub(super) grouping: bool,
    pub(super) precision: Option<usize>,
    pub(super) presentation: Option<char>,
}

impl FormatSpec {
    /// Parse `spec`, for example `">+12,.2f"` or `"#010x"`.
    pub(super) fn parse(spec: &str) -> Result<Self, String> {
        let unsupported = || format!("unsupported format specification {spec:?}");
        let mut parsed = Self::default();
        let mut rest = spec;
        if let Some(align) = rest.chars().next().and_then(|first| match first {
            '<' => Some(Align::Left),
            '>' => Some(Align::Right),
            '^' => Some(Align::Center),
            _ => None,
        }) {
            parsed.align = Some(align);
            rest = &rest[1..];
        }
        if let Some(stripped) = rest.strip_prefix('+') {
            parsed.plus = true;
            rest = stripped;
        } else if let Some(stripped) = rest.strip_prefix('-') {
            rest = stripped;
        }
        if let Some(stripped) = rest.strip_prefix('#') {
            parsed.alternate = true;
            rest = stripped;
        }
        if let Some(stripped) = rest.strip_prefix('0') {
            parsed.zero = true;
            rest = stripped;
        }
        let (width, stripped) = leading_number(rest).map_err(|_| unsupported())?;
        parsed.width = width.unwrap_or(0);
        rest = stripped;
        if let Some(stripped) = rest.strip_prefix(',') {
            parsed.grouping = true;
            rest = stripped;
        }
        if let Some(stripped) = rest.strip_prefix('.') {
            let (precision, stripped) = leading_number(stripped).map_err(|_| unsupported())?;
            parsed.precision = Some(precision.ok_or_else(unsupported)?);
            rest = stripped;
        }
        let mut chars = rest.chars();
        parsed.presentation = chars.next();
        if chars.next().is_some() {
            return Err(unsupported());
        }
        Ok(parsed)
    }

    fn sign(&self, negative: bool) -> &'static str {
        if negative {
            "-"
        } else if self.plus {
            "+"
        } else {
            ""
        }
    }

    /// Pad a rendered number. Zero padding goes between the sign and the digits, and explicit
    /// alignment takes precedence over it, as in CPython.
    fn pad_number(&self, sign: &str, body: String) -> String {
        let body = if self.grouping {
            group_decimal(&body, self.zero_width(sign.len()))
        } else {
            body
        };
        let rendered = format!("{sign}{body}");
        if self.align.is_none() && self.zero {
            let padding = self.width.saturating_sub(rendered.len());
            return format!("{sign}{}{body}", "0".repeat(padding));
        }
        align(&rendered, self.width, self.align.unwrap_or(Align::Right))
    }

    /// The minimum grouped digit-field width when zero padding applies.
    fn zero_width(&self, sign: usize) -> usize {
        if self.zero && self.align.is_none() {
            self.width.saturating_sub(sign)
        } else {
            0
        }
    }
}

fn leading_number(text: &str) -> Result<(Option<usize>, &str), std::num::ParseIntError> {
    let digits = text.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return Ok((None, text));
    }
    Ok((Some(text[..digits].parse()?), &text[digits..]))
}

/// Format an integer with the `d`, `b`, `o`, `x`, `X`, or default presentation.
pub(super) fn format_integer(value: BigInt, spec: &FormatSpec) -> Result<String, String> {
    if spec.precision.is_some() {
        return Err("precision is not allowed in integer format".into());
    }
    let (radix, prefix, uppercase) = match spec.presentation {
        None | Some('d') => (10, "", false),
        Some('b') => (2, "0b", false),
        Some('o') => (8, "0o", false),
        Some('x') => (16, "0x", false),
        Some('X') => (16, "0X", true),
        Some(other) => return Err(format!("unsupported integer format type {other:?}")),
    };
    if spec.alternate && radix == 10 {
        return Err("alternate form is not allowed for decimal integers".into());
    }
    if spec.grouping && radix != 10 {
        return Err("comma grouping requires decimal presentation".into());
    }
    let negative = value.sign() == Sign::Minus;
    let magnitude = if negative { -value } else { value };
    let mut digits = magnitude.to_str_radix(radix);
    if uppercase {
        digits.make_ascii_uppercase();
    }
    let prefix = if spec.alternate { prefix } else { "" };
    if !prefix.is_empty() && spec.zero && spec.align.is_none() {
        let content = spec.sign(negative).len() + prefix.len() + digits.len();
        let padding = spec.width.saturating_sub(content);
        return Ok(format!(
            "{}{prefix}{}{digits}",
            spec.sign(negative),
            "0".repeat(padding)
        ));
    }
    Ok(spec.pad_number(spec.sign(negative), format!("{prefix}{digits}")))
}

/// Format a float with the `f`, `e`, `E`, `g`, `G`, `%`, or default presentation. `repr` is
/// Python's shortest round-trip text, used when neither a presentation nor a precision is given.
pub(super) fn format_float(value: f64, repr: &str, spec: &FormatSpec) -> Result<String, String> {
    if spec.alternate {
        return Err("alternate form is not supported for floats".into());
    }
    let precision = spec.precision.unwrap_or(6);
    let magnitude = value.abs();
    let body = match spec.presentation {
        _ if !value.is_finite() => non_finite(value, spec.presentation),
        Some('f') => format!("{magnitude:.precision$}"),
        Some('e') => scientific(magnitude, precision, 'e'),
        Some('E') => scientific(magnitude, precision, 'E'),
        Some('g') => format_general(magnitude, precision, false, false),
        Some('G') => format_general(magnitude, precision, false, true),
        Some('%') => format!("{:.precision$}%", magnitude * 100.0),
        None => match spec.precision {
            Some(precision) => format_general(magnitude, precision, true, false),
            None => repr.trim_start_matches('-').to_string(),
        },
        Some(other) => return Err(format!("unsupported floating-point format type {other:?}")),
    };
    let body = match spec.presentation {
        Some('%') if !value.is_finite() => format!("{body}%"),
        _ => body,
    };
    Ok(spec.pad_number(spec.sign(value.is_sign_negative()), body))
}

/// Format a string with the `s` or default presentation.
pub(super) fn format_text(value: &str, spec: &FormatSpec) -> Result<String, String> {
    if spec.plus || spec.alternate || spec.zero || spec.grouping || spec.precision.is_some() {
        return Err("numeric format options are not allowed for strings".into());
    }
    if !matches!(spec.presentation, None | Some('s')) {
        return Err(format!(
            "unsupported string format type {:?}",
            spec.presentation.unwrap_or_default()
        ));
    }
    Ok(align(value, spec.width, spec.align.unwrap_or(Align::Left)))
}

fn non_finite(value: f64, presentation: Option<char>) -> String {
    let label = if value.is_nan() { "nan" } else { "inf" };
    if matches!(presentation, Some('E' | 'G')) {
        label.to_uppercase()
    } else {
        label.into()
    }
}

fn scientific(value: f64, precision: usize, marker: char) -> String {
    let rendered = format!("{value:.precision$e}");
    let (mantissa, exponent) = rendered.split_once('e').expect("Rust scientific format");
    let exponent = exponent.parse::<i32>().expect("Rust scientific exponent");
    format!("{mantissa}{marker}{exponent:+03}")
}

/// Format significant digits of a nonnegative finite value using Python's fixed/scientific
/// thresholds. The default presentation keeps a trailing `.0` and switches to scientific one
/// digit earlier than `g`.
fn format_general(value: f64, precision: usize, default_type: bool, uppercase: bool) -> String {
    let precision = precision.max(1);
    let decimals = precision.saturating_sub(1);
    let scientific = format!("{value:.decimals$e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("Rust scientific format");
    let exponent = exponent.parse::<i32>().expect("Rust scientific exponent");
    let threshold = if default_type {
        precision.saturating_sub(1)
    } else {
        precision
    };
    let use_scientific = exponent < -4 || usize::try_from(exponent).is_ok_and(|e| e >= threshold);
    if use_scientific {
        let mantissa = trim_fraction(mantissa, false);
        return format!(
            "{mantissa}{}{exponent:+03}",
            if uppercase { 'E' } else { 'e' }
        );
    }
    let decimals = usize::try_from(precision as i128 - i128::from(exponent) - 1)
        .expect("fixed precision is nonnegative");
    trim_fraction(&format!("{value:.decimals$}"), default_type)
}

fn trim_fraction(value: &str, preserve_decimal: bool) -> String {
    let Some((integer, fraction)) = value.split_once('.') else {
        return value.to_string();
    };
    let fraction = fraction.trim_end_matches('0');
    if fraction.is_empty() {
        if preserve_decimal {
            format!("{integer}.0")
        } else {
            integer.to_string()
        }
    } else {
        format!("{integer}.{fraction}")
    }
}

/// Insert thousands separators in the leading digits of an unsigned `body`, first zero-filling
/// those digits until the grouped field reaches `minimum` characters. As in CPython, a separator
/// can make the result one character wider than `minimum`.
fn group_decimal(body: &str, minimum: usize) -> String {
    let digits = body.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return body.into();
    }
    let rest = &body[digits..];
    let mut filled = digits;
    while filled + (filled - 1) / 3 + rest.len() < minimum {
        filled += 1;
    }
    let padded = format!("{}{}", "0".repeat(filled - digits), &body[..digits]);
    let mut result = String::with_capacity(filled + filled / 3 + rest.len());
    for (index, digit) in padded.chars().enumerate() {
        if index > 0 && (filled - index).is_multiple_of(3) {
            result.push(',');
        }
        result.push(digit);
    }
    result.push_str(rest);
    result
}

fn align(value: &str, width: usize, alignment: Align) -> String {
    let padding = width.saturating_sub(value.chars().count());
    let left = match alignment {
        Align::Right => padding,
        Align::Center => padding / 2,
        Align::Left => 0,
    };
    let right = padding - left;
    format!("{}{}{}", " ".repeat(left), value, " ".repeat(right))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float(value: f64, spec: &str) -> String {
        format_float(value, &value.to_string(), &FormatSpec::parse(spec).unwrap()).unwrap()
    }

    fn integer(value: i64, spec: &str) -> Result<String, String> {
        format_integer(value.into(), &FormatSpec::parse(spec)?)
    }

    #[test]
    fn parses_every_supported_field() {
        assert_eq!(
            FormatSpec::parse(">+#012,.3f").unwrap(),
            FormatSpec {
                align: Some(Align::Right),
                plus: true,
                alternate: true,
                zero: true,
                width: 12,
                grouping: true,
                precision: Some(3),
                presentation: Some('f'),
            }
        );
        for invalid in ["*<5", "1,,.2", "5.2ff", ".f", " d"] {
            assert!(FormatSpec::parse(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn grouping_zero_fills_before_inserting_separators() {
        assert_eq!(float(-1234.5, "+12,.2f"), "   -1,234.50");
        assert_eq!(float(1234.0, "012,.0f"), "0,000,001,234");
        assert_eq!(float(12345.0, ",.0"), "1e+04");
        assert_eq!(integer(1234, "+012,d").unwrap(), "+000,001,234");
        assert_eq!(integer(-1234567, ",").unwrap(), "-1,234,567");
        assert!(integer(42, ",x").is_err());
    }

    #[test]
    fn percent_and_signs_match_python() {
        assert_eq!(float(0.125, ".1%"), "12.5%");
        assert_eq!(float(-0.5, "+08.1%"), "-0050.0%");
        assert_eq!(float(f64::INFINITY, "+%"), "+inf%");
        assert_eq!(float(f64::NAN, "E"), "NAN");
        assert_eq!(integer(255, "#06x").unwrap(), "0x00ff");
        assert_eq!(integer(5, "^+5").unwrap(), " +5  ");
    }
}
