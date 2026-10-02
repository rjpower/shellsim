//! Python's format-specification mini-language for f-strings, `format`, and `str.format`.
//!
//! A specification is parsed into [`FormatSpec`], then rendered for an integer, float, complex
//! number or string. The grammar is CPython 3.14's:
//!
//! ```text
//! [[fill]align][sign]["z"]["#"]["0"][width][grouping]["." precision [grouping]][type]
//! ```
//!
//! Numbers render as a sign, a prefix such as `0x`, integer digits, and a remainder holding the
//! fraction, exponent and `%`. Grouping separators go into the integer digits; with `0` fill
//! and `=` alignment the padding becomes leading zero digits, so separators appear inside it
//! (`format(1234, "012,")` is `0,000,001,234`). Checks run in CPython's order, and each failure
//! carries CPython's exception type and message. The `n` presentation uses the C locale, as
//! CPython does before a program calls `setlocale`, so it never groups.

use super::BigInt;
use num_bigint::Sign as BigSign;
use num_traits::ToPrimitive;

/// A formatting failure: the exception CPython raises and its message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct FormatError {
    pub(super) kind: &'static str,
    pub(super) message: String,
}

fn value_error(message: impl Into<String>) -> FormatError {
    FormatError {
        kind: "ValueError",
        message: message.into(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Align {
    Left,
    Right,
    Center,
    /// `=`: padding goes between the sign (and prefix) and the digits.
    Numeric,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Sign {
    Plus,
    Minus,
    Space,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Grouping {
    Comma,
    Underscore,
}

impl Grouping {
    fn separator(self) -> char {
        match self {
            Self::Comma => ',',
            Self::Underscore => '_',
        }
    }
}

/// A parsed format specification. Widths and precisions are bounded by the caller's resource
/// reservation before rendering allocates padding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct FormatSpec {
    pub(super) fill: Option<char>,
    pub(super) align: Option<Align>,
    pub(super) sign: Option<Sign>,
    /// `z`: show a value that rounds to negative zero without its sign.
    pub(super) no_negative_zero: bool,
    pub(super) alternate: bool,
    /// The `0` flag: zero fill unless a fill character is given, and `=` alignment for numbers
    /// unless an alignment is given.
    pub(super) zero: bool,
    pub(super) width: usize,
    pub(super) grouping: Option<Grouping>,
    pub(super) precision: Option<usize>,
    /// Separators between groups of three fraction digits (new in Python 3.14).
    pub(super) fraction_grouping: Option<Grouping>,
    pub(super) presentation: Option<char>,
}

fn align_of(character: char) -> Option<Align> {
    Some(match character {
        '<' => Align::Left,
        '>' => Align::Right,
        '^' => Align::Center,
        '=' => Align::Numeric,
        _ => return None,
    })
}

/// A decimal number at the front of `characters`, or `None` when it starts with no digit.
fn leading_number(characters: &[char], position: &mut usize) -> Result<Option<usize>, FormatError> {
    let start = *position;
    let mut number = 0usize;
    while let Some(digit) = characters.get(*position).and_then(|c| c.to_digit(10)) {
        number = number
            .checked_mul(10)
            .and_then(|number| number.checked_add(digit as usize))
            .ok_or_else(|| value_error("Too many decimal digits in format string"))?;
        *position += 1;
    }
    Ok((*position > start).then_some(number))
}

fn grouping_type_error(grouping: Grouping, presentation: char) -> FormatError {
    value_error(format!(
        "Cannot specify '{}' with '{presentation}'.",
        grouping.separator()
    ))
}

fn unknown_code(presentation: char, type_name: &str) -> FormatError {
    value_error(format!(
        "Unknown format code '{presentation}' for object of type '{type_name}'"
    ))
}

impl FormatSpec {
    /// Parse `spec` for a value of `type_name` whose presentation defaults to
    /// `default_presentation` (`d` for ints, `s` for strings, none for floats), which decides
    /// whether a grouping separator is allowed. For example `"*>+12,.2f"` or `"#010_x"`.
    pub(super) fn parse(
        spec: &str,
        type_name: &str,
        default_presentation: Option<char>,
    ) -> Result<Self, FormatError> {
        let characters = spec.chars().collect::<Vec<_>>();
        let mut parsed = Self::default();
        let mut position = 0;
        if let Some(align) = characters.get(1).copied().and_then(align_of) {
            parsed.fill = Some(characters[0]);
            parsed.align = Some(align);
            position = 2;
        } else if let Some(align) = characters.first().copied().and_then(align_of) {
            parsed.align = Some(align);
            position = 1;
        }
        parsed.sign = match characters.get(position) {
            Some('+') => Some(Sign::Plus),
            Some('-') => Some(Sign::Minus),
            Some(' ') => Some(Sign::Space),
            _ => None,
        };
        position += usize::from(parsed.sign.is_some());
        if characters.get(position) == Some(&'z') {
            parsed.no_negative_zero = true;
            position += 1;
        }
        if characters.get(position) == Some(&'#') {
            parsed.alternate = true;
            position += 1;
        }
        if parsed.fill.is_none() && characters.get(position) == Some(&'0') {
            parsed.zero = true;
            position += 1;
        }
        parsed.width = leading_number(&characters, &mut position)?.unwrap_or(0);
        parsed.grouping = parse_grouping(&characters, &mut position)?;
        if characters.get(position) == Some(&'.') {
            position += 1;
            parsed.precision = Some(
                leading_number(&characters, &mut position)?
                    .ok_or_else(|| value_error("Format specifier missing precision"))?,
            );
            parsed.fraction_grouping = parse_grouping(&characters, &mut position)?;
        }
        match &characters[position..] {
            [] => parsed.presentation = default_presentation,
            [presentation] => parsed.presentation = Some(*presentation),
            _ => {
                return Err(value_error(format!(
                    "Invalid format specifier '{spec}' for object of type '{type_name}'"
                )))
            }
        }
        if let Some(grouping) = parsed.grouping {
            match (parsed.presentation, grouping) {
                (None | Some('d' | 'e' | 'f' | 'g' | 'E' | 'G' | '%' | 'F'), _) => {}
                (Some('b' | 'o' | 'x' | 'X'), Grouping::Underscore) => {}
                (Some(presentation), _) => return Err(grouping_type_error(grouping, presentation)),
            }
        }
        Ok(parsed)
    }

    fn sign_text(&self, negative: bool) -> &'static str {
        match (negative, self.sign) {
            (true, _) => "-",
            (false, Some(Sign::Plus)) => "+",
            (false, Some(Sign::Space)) => " ",
            (false, _) => "",
        }
    }

    fn fill_char(&self) -> char {
        self.fill.unwrap_or(if self.zero { '0' } else { ' ' })
    }

    /// Lay out a number from its sign, prefix, integer digits and remainder.
    fn render_number(
        &self,
        negative: bool,
        prefix: &str,
        digits: &str,
        remainder: &str,
        group_size: usize,
    ) -> String {
        let sign = self.sign_text(negative);
        let fill = self.fill_char();
        let align = self.align.unwrap_or(if self.zero {
            Align::Numeric
        } else {
            Align::Right
        });
        let fixed = sign.len() + prefix.len() + remainder.chars().count();
        // Zero fill with `=` alignment extends the digits themselves, so grouping covers it.
        let minimum = if fill == '0' && align == Align::Numeric {
            self.width.saturating_sub(fixed)
        } else {
            0
        };
        let digits = match self.grouping {
            Some(grouping) if !digits.is_empty() => {
                group_digits(digits, group_size, grouping.separator(), minimum)
            }
            _ => format!(
                "{}{digits}",
                "0".repeat(minimum.saturating_sub(digits.len()))
            ),
        };
        let length = fixed + digits.chars().count();
        let padding = self.width.saturating_sub(length);
        let fill_text = |count: usize| fill.to_string().repeat(count);
        match align {
            Align::Numeric => format!("{sign}{prefix}{}{digits}{remainder}", fill_text(padding)),
            Align::Left => format!("{sign}{prefix}{digits}{remainder}{}", fill_text(padding)),
            Align::Right => format!("{}{sign}{prefix}{digits}{remainder}", fill_text(padding)),
            Align::Center => format!(
                "{}{sign}{prefix}{digits}{remainder}{}",
                fill_text(padding / 2),
                fill_text(padding - padding / 2)
            ),
        }
    }
}

fn parse_grouping(
    characters: &[char],
    position: &mut usize,
) -> Result<Option<Grouping>, FormatError> {
    let both = || value_error("Cannot specify both ',' and '_'.");
    let mut grouping = None;
    if characters.get(*position) == Some(&',') {
        grouping = Some(Grouping::Comma);
        *position += 1;
    }
    if characters.get(*position) == Some(&'_') {
        if grouping.is_some() {
            return Err(both());
        }
        grouping = Some(Grouping::Underscore);
        *position += 1;
    }
    if characters.get(*position) == Some(&',') && grouping == Some(Grouping::Underscore) {
        return Err(both());
    }
    Ok(grouping)
}

/// Format an integer. `type_name` is `int` or `bool`, for error messages. Float presentations
/// convert the value to a float first, as `int.__format__` does.
pub(super) fn format_integer(
    value: BigInt,
    spec: &str,
    type_name: &str,
) -> Result<String, FormatError> {
    let spec = FormatSpec::parse(spec, type_name, Some('d'))?;
    let presentation = spec.presentation.unwrap_or('d');
    let (radix, prefix, uppercase) = match presentation {
        'd' | 'n' | 'c' => (10, "", false),
        'b' => (2, "0b", false),
        'o' => (8, "0o", false),
        'x' => (16, "0x", false),
        'X' => (16, "0X", true),
        'e' | 'E' | 'f' | 'F' | 'g' | 'G' | '%' => {
            let float = value
                .to_f64()
                .filter(|float| float.is_finite())
                .ok_or(FormatError {
                    kind: "OverflowError",
                    message: "int too large to convert to float".into(),
                })?;
            return render_float(float, "", &spec, type_name);
        }
        other => return Err(unknown_code(other, type_name)),
    };
    if spec.precision.is_some() {
        return Err(value_error(
            "Precision not allowed in integer format specifier",
        ));
    }
    if spec.no_negative_zero {
        return Err(value_error(
            "Negative zero coercion (z) not allowed in integer format specifier",
        ));
    }
    if presentation == 'c' {
        if spec.sign.is_some() {
            return Err(value_error(
                "Sign not allowed with integer format specifier 'c'",
            ));
        }
        if spec.alternate {
            return Err(value_error(
                "Alternate form (#) not allowed with integer format specifier 'c'",
            ));
        }
        let character = value.to_u32().and_then(char::from_u32).ok_or(FormatError {
            kind: "OverflowError",
            message: "%c arg not in range(0x110000)".into(),
        })?;
        return Ok(spec.render_number(false, "", "", &character.to_string(), 3));
    }
    let negative = value.sign() == BigSign::Minus;
    let magnitude = if negative { -value } else { value };
    if radix == 10 && super::super::number::exceeds_str_digits(&magnitude) {
        return Err(value_error(format!(
            "Exceeds the limit ({} digits) for integer string conversion; use \
             sys.set_int_max_str_digits() to increase the limit",
            super::super::number::INT_MAX_STR_DIGITS
        )));
    }
    let mut digits = magnitude.to_str_radix(radix);
    if uppercase {
        digits.make_ascii_uppercase();
    }
    let prefix = if spec.alternate { prefix } else { "" };
    let group_size = if radix == 10 { 3 } else { 4 };
    Ok(spec.render_number(negative, prefix, &digits, "", group_size))
}

/// Format a float. `repr` is Python's shortest round-trip text for the value, used when neither
/// a presentation nor a precision is given.
pub(super) fn format_float(value: f64, repr: &str, spec: &str) -> Result<String, FormatError> {
    let spec = FormatSpec::parse(spec, "float", None)?;
    render_float(value, repr, &spec, "float")
}

fn render_float(
    value: f64,
    repr: &str,
    spec: &FormatSpec,
    type_name: &str,
) -> Result<String, FormatError> {
    let presentation = spec.presentation;
    if let Some(other) = presentation.filter(|p| !"eEfFgGn%".contains(*p)) {
        return Err(unknown_code(other, type_name));
    }
    if let (Some('n'), Some(grouping)) = (presentation, spec.fraction_grouping) {
        return Err(grouping_type_error(grouping, 'n'));
    }
    let alternate = spec.alternate;
    let magnitude = value.abs();
    let uppercase = matches!(presentation, Some('E' | 'F' | 'G'));
    let text = if value.is_finite() {
        let precision = spec.precision.unwrap_or(6);
        match presentation {
            Some('f' | 'F') => fixed(magnitude, precision, alternate),
            Some('%') => format!("{}%", fixed(magnitude * 100.0, precision, alternate)),
            Some('e' | 'E') => scientific(magnitude, precision, alternate, uppercase),
            Some('g' | 'G' | 'n') => general(magnitude, precision, false, uppercase, alternate),
            _ => match spec.precision {
                Some(precision) => general(magnitude, precision, true, false, alternate),
                None if alternate => with_point(repr.trim_start_matches('-')),
                None => repr.trim_start_matches('-').to_string(),
            },
        }
    } else {
        let label = if value.is_nan() { "nan" } else { "inf" };
        let label = if uppercase {
            label.to_uppercase()
        } else {
            label.to_string()
        };
        if presentation == Some('%') {
            format!("{label}%")
        } else {
            label
        }
    };
    let mantissa_end = text.find(['e', 'E', '%']).unwrap_or(text.len());
    let rounds_to_zero = text[..mantissa_end].bytes().all(|b| b == b'0' || b == b'.');
    // CPython never shows a sign on NaN, whatever its sign bit.
    let negative = value.is_sign_negative()
        && !value.is_nan()
        && !(spec.no_negative_zero && value.is_finite() && rounds_to_zero);
    let split = text.bytes().take_while(u8::is_ascii_digit).count();
    let (digits, remainder) = text.split_at(split);
    let remainder = match spec.fraction_grouping {
        Some(grouping) => group_fraction(remainder, grouping.separator()),
        None => remainder.to_string(),
    };
    Ok(spec.render_number(negative, "", digits, &remainder, 3))
}

/// Format a complex number as CPython's `complex.__format__` does, for a non-empty `spec`.
///
/// Each part is formatted as a float without padding, and the width, fill and alignment then
/// apply to the whole number. The imaginary part always carries a sign. Without a presentation
/// type, the parts use their `repr` digits (`g` when a precision is given), the real part is
/// dropped when it is `+0`, and otherwise the number is parenthesized, as in `repr`. For
/// example `format(1+2j, ">12.1f")` is `"    1.0+2.0j"` and `format(1+2j, "")` is `"(1+2j)"`.
pub(super) fn format_complex(real: f64, imag: f64, spec: &str) -> Result<String, FormatError> {
    let spec = FormatSpec::parse(spec, "complex", None)?;
    if let Some(other) = spec.presentation.filter(|p| !"eEfFgGn".contains(*p)) {
        return Err(unknown_code(other, "complex"));
    }
    if spec.fill_char() == '0' {
        return Err(value_error(
            "Zero padding is not allowed in complex format specifier",
        ));
    }
    if spec.align == Some(Align::Numeric) {
        return Err(value_error(
            "'=' alignment flag is not allowed in complex format specifier",
        ));
    }
    let bare = spec.presentation.is_none();
    let skip_real = bare && real == 0.0 && real.is_sign_positive();
    let mut part = FormatSpec {
        fill: None,
        align: None,
        zero: false,
        width: 0,
        ..spec
    };
    if bare && spec.precision.is_some() {
        part.presentation = Some('g');
    }
    let render = |value: f64, sign: Option<Sign>| {
        let repr = super::super::complex::repr_component(value, false);
        render_float(value, &repr, &FormatSpec { sign, ..part }, "complex")
    };
    let imaginary_sign = if skip_real {
        spec.sign
    } else {
        Some(Sign::Plus)
    };
    let mut body = render(imag, imaginary_sign)?;
    body.push('j');
    if !skip_real {
        body = render(real, spec.sign)? + &body;
        if bare {
            body = format!("({body})");
        }
    }
    let padding = spec.width.saturating_sub(body.chars().count());
    let (left, right) = match spec.align.unwrap_or(Align::Right) {
        Align::Left | Align::Numeric => (0, padding),
        Align::Right => (padding, 0),
        Align::Center => (padding / 2, padding - padding / 2),
    };
    let fill = spec.fill_char().to_string();
    Ok(format!("{}{body}{}", fill.repeat(left), fill.repeat(right)))
}

/// Format a string with the `s` or default presentation. `type_name` is `str` unless the text
/// came from a `!r`/`!s` conversion of another type.
pub(super) fn format_text(value: &str, spec: &str) -> Result<String, FormatError> {
    let spec = FormatSpec::parse(spec, "str", Some('s'))?;
    if let Some(other) = spec.presentation.filter(|p| *p != 's') {
        return Err(unknown_code(other, "str"));
    }
    if spec.sign.is_some() {
        return Err(value_error("Sign not allowed in string format specifier"));
    }
    if spec.no_negative_zero {
        return Err(value_error(
            "Negative zero coercion (z) not allowed in string format specifier",
        ));
    }
    if spec.alternate {
        return Err(value_error(
            "Alternate form (#) not allowed in string format specifier",
        ));
    }
    if spec.align == Some(Align::Numeric) {
        return Err(value_error(
            "'=' alignment not allowed in string format specifier",
        ));
    }
    let text = match spec.precision {
        Some(precision) => value.chars().take(precision).collect::<String>(),
        None => value.to_string(),
    };
    let padding = spec.width.saturating_sub(text.chars().count());
    let fill = spec.fill_char().to_string();
    let (left, right) = match spec.align.unwrap_or(Align::Left) {
        Align::Left | Align::Numeric => (0, padding),
        Align::Right => (padding, 0),
        Align::Center => (padding / 2, padding - padding / 2),
    };
    Ok(format!("{}{text}{}", fill.repeat(left), fill.repeat(right)))
}

/// `text` with a decimal point before its exponent if it has none, for the `#` form.
fn with_point(text: &str) -> String {
    if text.contains('.') {
        return text.to_string();
    }
    match text.find(['e', 'E']) {
        Some(exponent) => format!("{}.{}", &text[..exponent], &text[exponent..]),
        None => format!("{text}."),
    }
}

fn fixed(value: f64, precision: usize, alternate: bool) -> String {
    let text = format!("{value:.precision$}");
    if alternate && precision == 0 {
        format!("{text}.")
    } else {
        text
    }
}

fn scientific(value: f64, precision: usize, alternate: bool, uppercase: bool) -> String {
    let rendered = format!("{value:.precision$e}");
    let (mantissa, exponent) = rendered.split_once('e').expect("Rust scientific format");
    let exponent = exponent.parse::<i32>().expect("Rust scientific exponent");
    let point = if alternate && precision == 0 { "." } else { "" };
    let marker = if uppercase { 'E' } else { 'e' };
    format!("{mantissa}{point}{marker}{exponent:+03}")
}

/// Format significant digits of a nonnegative finite value using Python's fixed/scientific
/// thresholds. The default presentation (`add_dot_0`) keeps a trailing `.0` and switches to
/// scientific one digit earlier than `g`; the `#` form keeps trailing zeros and the point.
fn general(
    value: f64,
    precision: usize,
    add_dot_0: bool,
    uppercase: bool,
    alternate: bool,
) -> String {
    let precision = precision.max(1);
    let decimals = precision - 1;
    let scientific = format!("{value:.decimals$e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("Rust scientific format");
    let exponent = exponent.parse::<i32>().expect("Rust scientific exponent");
    let threshold = if add_dot_0 { precision - 1 } else { precision };
    let use_scientific = exponent < -4 || usize::try_from(exponent).is_ok_and(|e| e >= threshold);
    let marker = if uppercase { 'E' } else { 'e' };
    if use_scientific {
        let mantissa = if alternate {
            with_point(mantissa)
        } else {
            trim_fraction(mantissa, false)
        };
        return format!("{mantissa}{marker}{exponent:+03}");
    }
    let decimals = usize::try_from(precision as i128 - i128::from(exponent) - 1)
        .expect("fixed precision is nonnegative");
    let text = format!("{value:.decimals$}");
    if alternate {
        with_point(&text)
    } else {
        trim_fraction(&text, add_dot_0)
    }
}

fn trim_fraction(value: &str, preserve_decimal: bool) -> String {
    let Some((integer, fraction)) = value.split_once('.') else {
        return if preserve_decimal {
            format!("{value}.0")
        } else {
            value.to_string()
        };
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

/// Insert `separator` between groups of `size` digits, counting from the right, after
/// zero-filling the digits until the grouped text reaches `minimum` characters. As in CPython,
/// a separator can make the result one character wider than `minimum`.
fn group_digits(digits: &str, size: usize, separator: char, minimum: usize) -> String {
    let mut filled = digits.len();
    while filled + (filled - 1) / size < minimum {
        filled += 1;
    }
    let padded = format!("{}{digits}", "0".repeat(filled - digits.len()));
    let mut result = String::with_capacity(filled + filled / size);
    for (index, digit) in padded.chars().enumerate() {
        if index > 0 && (filled - index).is_multiple_of(size) {
            result.push(separator);
        }
        result.push(digit);
    }
    result
}

/// Group the fraction digits that follow the decimal point in `remainder` in threes from the
/// left, leaving any exponent or `%` untouched.
fn group_fraction(remainder: &str, separator: char) -> String {
    let Some(fraction) = remainder.strip_prefix('.') else {
        return remainder.to_string();
    };
    let digits = fraction.bytes().take_while(u8::is_ascii_digit).count();
    let mut result = String::from(".");
    for (index, digit) in fraction[..digits].chars().enumerate() {
        if index > 0 && index.is_multiple_of(3) {
            result.push(separator);
        }
        result.push(digit);
    }
    result.push_str(&fraction[digits..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float(value: f64, spec: &str) -> String {
        format_float(value, &value.to_string(), spec).unwrap()
    }

    fn integer(value: i64, spec: &str) -> Result<String, FormatError> {
        format_integer(value.into(), spec, "int")
    }

    #[test]
    fn parses_every_field() {
        assert_eq!(
            FormatSpec::parse("*>+z#012,.3_f", "float", None).unwrap(),
            FormatSpec {
                fill: Some('*'),
                align: Some(Align::Right),
                sign: Some(Sign::Plus),
                no_negative_zero: true,
                alternate: true,
                zero: false,
                width: 12,
                grouping: Some(Grouping::Comma),
                precision: Some(3),
                fraction_grouping: Some(Grouping::Underscore),
                presentation: Some('f'),
            }
        );
        let invalid = FormatSpec::parse("5.2ff", "int", Some('d')).unwrap_err();
        assert_eq!(
            invalid.message,
            "Invalid format specifier '5.2ff' for object of type 'int'"
        );
        let both = FormatSpec::parse(",_", "int", Some('d')).unwrap_err();
        assert_eq!(both.message, "Cannot specify both ',' and '_'.");
        let missing = FormatSpec::parse(".", "int", Some('d')).unwrap_err();
        assert_eq!(missing.message, "Format specifier missing precision");
    }

    #[test]
    fn fill_alignment_and_zero_flag_match_python() {
        assert_eq!(integer(5, "*^9").unwrap(), "****5****");
        assert_eq!(integer(5, "=+6").unwrap(), "+    5");
        assert_eq!(integer(5, "<05").unwrap(), "50000");
        assert_eq!(integer(-5, "0^8").unwrap(), "000-5000");
        assert_eq!(integer(5, " 3d").unwrap(), "  5");
        assert_eq!(integer(5, "x<+#8x").unwrap(), "+0x5xxxx");
        assert_eq!(format_text("ab", "05").unwrap(), "ab000");
        assert_eq!(
            format_text("abc", "\u{e9}^7.2").unwrap(),
            "\u{e9}\u{e9}ab\u{e9}\u{e9}\u{e9}"
        );
    }

    #[test]
    fn grouping_zero_fills_before_inserting_separators() {
        assert_eq!(float(-1234.5, "+12,.2f"), "   -1,234.50");
        assert_eq!(float(1234.0, "012,.0f"), "0,000,001,234");
        assert_eq!(float(1234.5, "*=12,.1f"), "*****1,234.5");
        assert_eq!(float(12345.0, ",.0"), "1e+04");
        assert_eq!(float(1234.56789, ",.5_f"), "1,234.567_89");
        assert_eq!(integer(255, "#010_b").unwrap(), "0b1111_1111");
        assert_eq!(integer(-1_234_567, ",").unwrap(), "-1,234,567");
        assert_eq!(
            integer(42, ",x").unwrap_err().message,
            "Cannot specify ',' with 'x'."
        );
    }

    #[test]
    fn float_forms_match_python() {
        assert_eq!(float(0.125, ".1%"), "12.5%");
        assert_eq!(float(-0.5, "+08.1%"), "-0050.0%");
        assert_eq!(float(f64::INFINITY, "+%"), "+inf%");
        assert_eq!(float(f64::NAN, "F"), "NAN");
        assert_eq!(float(f64::NAN, "+010.2f"), "+000000nan");
        assert_eq!(float(-0.001, "z.1f"), "0.0");
        assert_eq!(float(-0.06, "z.1f"), "-0.1");
        assert_eq!(float(1.0, "#.0f"), "1.");
        assert_eq!(float(1e20, "#.3g"), "1.00e+20");
        assert_eq!(float(1.0, "#g"), "1.00000");
        assert_eq!(format_float(1e16, "1e+16", "#").unwrap(), "1.e+16");
    }

    #[test]
    fn errors_match_python() {
        assert_eq!(
            integer(5, ".2").unwrap_err().message,
            "Precision not allowed in integer format specifier"
        );
        assert_eq!(
            integer(65, "+c").unwrap_err().message,
            "Sign not allowed with integer format specifier 'c'"
        );
        assert_eq!(integer(-1, "c").unwrap_err().kind, "OverflowError");
        assert_eq!(integer(65, "5c").unwrap(), "    A");
        assert_eq!(
            format_float(1.5, "1.5", "d").unwrap_err().message,
            "Unknown format code 'd' for object of type 'float'"
        );
        assert_eq!(
            format_text("a", "=5").unwrap_err().message,
            "'=' alignment not allowed in string format specifier"
        );
        assert_eq!(
            format_text("a", ",").unwrap_err().message,
            "Cannot specify ',' with 's'."
        );
        assert_eq!(
            format_integer(BigInt::from(1), "s", "bool")
                .unwrap_err()
                .message,
            "Unknown format code 's' for object of type 'bool'"
        );
    }
}
