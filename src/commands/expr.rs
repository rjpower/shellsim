//! POSIX `expr`: argument-vector arithmetic and string expressions.
//!
//! `expr` never touches shell state: each invocation parses its argv as one self-contained
//! expression and prints the result. Precedence (lowest to highest) follows POSIX: `|`, `&`,
//! the relational operators, additive, multiplicative, then `:` (anchored basic-regex match).
//! GNU extensions `match`, `substr`, `index`, `length`, the `+ TOKEN` string escape, and a
//! leading `--` end-of-options marker are supported at the primary level. Integers are GNU-style
//! arbitrary precision (`num_bigint::BigInt`, already a dependency via `bc`), matching real
//! `expr 9223372036854775807 + 1`. A magnitude cap ([`MAX_DIGITS`]) charged against the process
//! CPU meter before any multiplication/division bounds the work a hostile expression can force,
//! since unlike a fixed-width integer this arithmetic has no natural ceiling of its own.
//!
//! `:` reuses the same BRE-to-Rust-regex translation grep and sed already depend on
//! ([`crate::commands::regex_compat::basic_regex_to_rust`]) rather than adding a second engine.

use std::collections::HashMap;

use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};

use crate::commands::regex_compat::basic_regex_to_rust;
use crate::commands::util::{ewln, wln};
use crate::commands::{CommandSpec, Io, Trust};
use crate::program::ProcessContext;
use crate::syscalls::System;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    super::reg_system(m, "/usr/bin/expr", Trust::Real, run);
}

/// GNU-style syntax error status; also used for division by zero and the digit-cap diagnostic.
const STATUS_ERROR: i32 = 2;

/// Cap on total decimal digits kept in any one integer value. Arbitrary-precision arithmetic has
/// no natural ceiling, so without this a chain of multiplications in a single invocation could
/// force unbounded host allocation; work proportional to operand size is charged against the CPU
/// meter before each multiplication/division, and this cap is the backstop.
const MAX_DIGITS: usize = 100_000;

fn run(context: &mut ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    // GNU `expr` treats a single leading `--` as an end-of-options marker so an expression can
    // start with a token that looks like an option, e.g. `expr -- -5 + 2`.
    let args: &[String] = if args.first().map(String::as_str) == Some("--") {
        &args[1..]
    } else {
        args
    };
    // Bound parse/regex work by the size of the argv actually supplied.
    let cost: u64 = args.iter().map(|a| a.len() as u64 + 1).sum();
    if !context.system.charge_cpu(cost.max(1)) {
        return context.system.stop_status();
    }
    if args.is_empty() {
        ewln(io.err, "expr: missing operand");
        ewln(io.err, "Try 'expr --help' for more information.");
        return STATUS_ERROR;
    }
    let owned: Vec<String> = args.to_vec();
    let mut parser = Parser {
        tokens: &owned,
        pos: 0,
        meter: SystemMeter(context.system),
    };
    let result = parser.parse_expr().and_then(|value| {
        if parser.pos != parser.tokens.len() {
            Err(EvalError::Message(format!(
                "syntax error: unexpected argument '{}'",
                parser.tokens[parser.pos]
            )))
        } else {
            Ok(value)
        }
    });
    match result {
        Ok(value) => {
            let text = value.to_display();
            wln(io.out, &text);
            if text.is_empty() || text == "0" {
                1
            } else {
                0
            }
        }
        Err(EvalError::Message(message)) => {
            ewln(io.err, &format!("expr: {message}"));
            STATUS_ERROR
        }
        Err(EvalError::Resource(status)) => status,
    }
}

/// One evaluated expr value. Integers keep their numeric form so later arithmetic can reuse it
/// without reparsing; every value also has a canonical string form for display and comparison.
#[derive(Clone, Debug)]
enum Value {
    Int(BigInt),
    Str(String),
}

impl Value {
    fn to_display(&self) -> String {
        match self {
            Value::Int(n) => n.to_string(),
            Value::Str(s) => s.clone(),
        }
    }

    fn as_int(&self) -> Option<BigInt> {
        match self {
            Value::Int(n) => Some(n.clone()),
            Value::Str(s) => parse_integer(s),
        }
    }

    fn is_falsy(&self) -> bool {
        match self {
            Value::Int(n) => n.is_zero(),
            Value::Str(s) => s.is_empty() || s == "0",
        }
    }
}

/// Parse a POSIX integer literal: optional sign, then one or more decimal digits. A literal
/// longer than [`MAX_DIGITS`] is treated as non-integer (it falls back to string comparison),
/// which keeps parsing itself bounded regardless of what an operation later does with it.
fn parse_integer(s: &str) -> Option<BigInt> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let (sign_len, negative) = match bytes[0] {
        b'-' => (1, true),
        b'+' => (1, false),
        _ => (0, false),
    };
    if bytes.len() == sign_len || !bytes[sign_len..].iter().all(u8::is_ascii_digit) {
        return None;
    }
    if bytes.len() - sign_len > MAX_DIGITS {
        return None;
    }
    let magnitude: BigInt = s[sign_len..].parse().ok()?;
    Some(if negative { -magnitude } else { magnitude })
}

fn digit_count(n: &BigInt) -> usize {
    n.magnitude().to_str_radix(10).len()
}

fn enforce_cap(n: BigInt) -> Result<BigInt, String> {
    if digit_count(&n) > MAX_DIGITS {
        Err("number too large".to_string())
    } else {
        Ok(n)
    }
}

/// Distinguishes an ordinary diagnostic (printed, exit 2) from CPU-budget exhaustion (which
/// carries the process's own stop status and must propagate immediately, matching how other
/// typed commands report a resource limit).
enum EvalError {
    Message(String),
    Resource(i32),
}

impl From<String> for EvalError {
    fn from(message: String) -> Self {
        EvalError::Message(message)
    }
}

/// The CPU-charging boundary, factored out of [`Parser`] so unit tests can exercise the parser
/// and arithmetic without a full [`System`] double (`System` has dozens of unrelated methods).
trait CpuMeter {
    fn charge(&mut self, units: u64) -> Result<(), EvalError>;
}

struct SystemMeter<'a>(&'a mut dyn System);

impl CpuMeter for SystemMeter<'_> {
    fn charge(&mut self, units: u64) -> Result<(), EvalError> {
        if self.0.charge_cpu(units.max(1)) {
            Ok(())
        } else {
            Err(EvalError::Resource(self.0.stop_status()))
        }
    }
}

#[cfg(test)]
struct NoopMeter;

#[cfg(test)]
impl CpuMeter for NoopMeter {
    fn charge(&mut self, _units: u64) -> Result<(), EvalError> {
        Ok(())
    }
}

struct Parser<'a, M: CpuMeter> {
    tokens: &'a [String],
    pos: usize,
    meter: M,
}

impl<'a, M: CpuMeter> Parser<'a, M> {
    fn peek(&self) -> Option<&'a str> {
        self.tokens.get(self.pos).map(String::as_str)
    }

    fn advance(&mut self) -> Option<&'a str> {
        let token = self.peek();
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    fn expect(&mut self, token: &str) -> Result<(), EvalError> {
        match self.advance() {
            Some(t) if t == token => Ok(()),
            Some(t) => Err(format!("syntax error: expecting '{token}' but got '{t}'").into()),
            None => Err(self.missing_argument_error().into()),
        }
    }

    /// GNU `expr` reports an unexpected end of input as "missing argument after '<X>'", where
    /// `<X>` is simply the last token supplied, regardless of which production hit the end.
    fn missing_argument_error(&self) -> String {
        format!(
            "syntax error: missing argument after '{}'",
            self.tokens.last().map(String::as_str).unwrap_or("")
        )
    }

    fn parse_expr(&mut self) -> Result<Value, EvalError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Value, EvalError> {
        let mut left = self.parse_and()?;
        while self.peek() == Some("|") {
            self.advance();
            let right = self.parse_and()?;
            // POSIX: expr1 if it is neither null nor 0; otherwise expr2 if it is not the empty
            // string; otherwise 0. Note expr2 == "0" is returned as-is (only *emptiness* falls
            // through), which is why `0 | ''` is "0", not "".
            left = if !left.is_falsy() {
                left
            } else if !right.to_display().is_empty() {
                right
            } else {
                Value::Int(BigInt::zero())
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Value, EvalError> {
        let mut left = self.parse_cmp()?;
        while self.peek() == Some("&") {
            self.advance();
            let right = self.parse_cmp()?;
            left = if left.is_falsy() || right.is_falsy() {
                Value::Int(BigInt::zero())
            } else {
                left
            };
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> Result<Value, EvalError> {
        let mut left = self.parse_add()?;
        while let Some(op @ ("=" | "!=" | "<" | "<=" | ">" | ">=")) = self.peek() {
            self.advance();
            let right = self.parse_add()?;
            let outcome = compare(&left, op, &right)?;
            left = Value::Int(BigInt::from(u8::from(outcome)));
        }
        Ok(left)
    }

    fn parse_add(&mut self) -> Result<Value, EvalError> {
        let mut left = self.parse_mul()?;
        while let Some(op @ ("+" | "-")) = self.peek() {
            self.advance();
            let right = self.parse_mul()?;
            let a = require_int(&left)?;
            let b = require_int(&right)?;
            self.meter
                .charge(digit_count(&a).max(digit_count(&b)) as u64)?;
            let result = if op == "+" { a + b } else { a - b };
            left = Value::Int(enforce_cap(result)?);
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<Value, EvalError> {
        let mut left = self.parse_colon()?;
        while let Some(op @ ("*" | "/" | "%")) = self.peek() {
            self.advance();
            let right = self.parse_colon()?;
            let a = require_int(&left)?;
            let b = require_int(&right)?;
            // Multiplication (and long division) are the only superlinear operations here;
            // charge proportional to the schoolbook cost before doing the work.
            let cost = (digit_count(&a) as u64) * (digit_count(&b) as u64) / 8 + 1;
            self.meter.charge(cost)?;
            let result = match op {
                "*" => a * b,
                "/" => {
                    if b.is_zero() {
                        return Err("division by zero".to_string().into());
                    }
                    a / b
                }
                _ => {
                    if b.is_zero() {
                        return Err("division by zero".to_string().into());
                    }
                    a % b
                }
            };
            left = Value::Int(enforce_cap(result)?);
        }
        Ok(left)
    }

    fn parse_colon(&mut self) -> Result<Value, EvalError> {
        let mut left = self.parse_primary()?;
        while self.peek() == Some(":") {
            self.advance();
            let right = self.parse_primary()?;
            left = Value::Str(bre_match(&left.to_display(), &right.to_display())?);
        }
        Ok(left)
    }

    fn parse_primary(&mut self) -> Result<Value, EvalError> {
        match self.peek() {
            Some("(") => {
                self.advance();
                let inner = self.parse_expr()?;
                self.expect(")")?;
                Ok(inner)
            }
            Some("+") => {
                self.advance();
                let token = self
                    .advance()
                    .ok_or_else(|| self.missing_argument_error())?;
                Ok(literal(token))
            }
            // GNU keywords are recognized wherever a primary is expected, unconditionally: a
            // bare `length`/`match`/`index`/`substr` always requires its full argument list, it
            // never silently falls back to being a literal string (use `+ length` for that).
            Some("length") => {
                self.advance();
                let value = self.parse_primary()?;
                Ok(Value::Int(BigInt::from(
                    value.to_display().chars().count() as i64
                )))
            }
            Some("match") => {
                self.advance();
                let haystack = self.parse_primary()?.to_display();
                let pattern = self.parse_primary()?.to_display();
                Ok(Value::Str(bre_match(&haystack, &pattern)?))
            }
            Some("index") => {
                self.advance();
                let haystack = self.parse_primary()?.to_display();
                let chars = self.parse_primary()?.to_display();
                let position = haystack
                    .chars()
                    .position(|c| chars.contains(c))
                    .map_or(0, |i| i + 1);
                Ok(Value::Int(BigInt::from(position as i64)))
            }
            Some("substr") => {
                self.advance();
                let haystack = self.parse_primary()?.to_display();
                let start = require_int(&self.parse_primary()?)?;
                let len = require_int(&self.parse_primary()?)?;
                Ok(Value::Str(substr(&haystack, &start, &len)))
            }
            Some(_) => Ok(literal(self.advance().expect("peeked"))),
            None => Err(self.missing_argument_error().into()),
        }
    }
}

fn literal(token: &str) -> Value {
    match parse_integer(token) {
        Some(n) => Value::Int(n),
        None => Value::Str(token.to_string()),
    }
}

fn require_int(value: &Value) -> Result<BigInt, String> {
    value
        .as_int()
        .ok_or_else(|| format!("non-integer argument '{}'", value.to_display()))
}

fn compare(left: &Value, op: &str, right: &Value) -> Result<bool, String> {
    let outcome = match (left.as_int(), right.as_int()) {
        (Some(a), Some(b)) => a.cmp(&b),
        _ => left.to_display().cmp(&right.to_display()),
    };
    Ok(match op {
        "=" => outcome.is_eq(),
        "!=" => outcome.is_ne(),
        "<" => outcome.is_lt(),
        "<=" => outcome.is_le(),
        ">" => outcome.is_gt(),
        ">=" => outcome.is_ge(),
        _ => unreachable!("caller only passes known relational operators"),
    })
}

/// Match `pattern` (a POSIX basic regular expression) anchored at the start of `haystack`.
/// Returns the first `\(...\)` capture when the pattern has one, otherwise the match length;
/// an unmatched pattern yields an empty string, matching GNU `expr`.
fn bre_match(haystack: &str, pattern: &str) -> Result<String, String> {
    let translated =
        basic_regex_to_rust(pattern).map_err(|e| format!("invalid regular expression: {e}"))?;
    let anchored = format!("^(?:{translated})");
    let re =
        regex::Regex::new(&anchored).map_err(|e| format!("invalid regular expression: {e}"))?;
    match re.captures(haystack) {
        Some(caps) if caps.len() > 1 => Ok(caps
            .get(1)
            .map(|m| m.as_str().to_string())
            .unwrap_or_default()),
        Some(caps) => Ok(caps
            .get(0)
            .map(|m| m.as_str().len())
            .unwrap_or(0)
            .to_string()),
        None if re.captures_len() > 1 => Ok(String::new()),
        None => Ok("0".to_string()),
    }
}

/// 1-indexed, POSIX/GNU-`substr` style: out-of-range positions or non-positive lengths yield
/// the empty string rather than an error. `start`/`len` are clamped through `i64` since they
/// only ever index a small in-memory string; a huge literal just clamps to "out of range".
fn substr(s: &str, start: &BigInt, len: &BigInt) -> String {
    let start = start.to_i64().unwrap_or(i64::MAX);
    let len = len.to_i64().unwrap_or(i64::MAX);
    if len <= 0 || start < 1 {
        return String::new();
    }
    let chars: Vec<char> = s.chars().collect();
    let start_index = (start - 1) as usize;
    if start_index >= chars.len() {
        return String::new();
    }
    let end_index = start_index.saturating_add(len as usize).min(chars.len());
    chars[start_index..end_index].iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(args: &[&str]) -> Result<Value, String> {
        let owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let mut parser = Parser {
            tokens: &owned,
            pos: 0,
            meter: NoopMeter,
        };
        parser.parse_expr().map_err(|e| match e {
            EvalError::Message(m) => m,
            EvalError::Resource(_) => "resource".to_string(),
        })
    }

    #[test]
    fn arithmetic_precedence() {
        assert_eq!(eval(&["2", "+", "3", "*", "4"]).unwrap().to_display(), "14");
        assert_eq!(
            eval(&["(", "2", "+", "3", ")", "*", "4"])
                .unwrap()
                .to_display(),
            "20"
        );
    }

    #[test]
    fn arbitrary_precision_matches_gnu_expr() {
        assert_eq!(
            eval(&["9223372036854775807", "+", "1"])
                .unwrap()
                .to_display(),
            "9223372036854775808"
        );
    }

    #[test]
    fn oversize_result_hits_the_documented_cap() {
        let big = "9".repeat(MAX_DIGITS);
        let err = eval(&[&big, "*", &big]).unwrap_err();
        assert_eq!(err, "number too large");
    }

    #[test]
    fn division_by_zero_is_rejected() {
        assert_eq!(eval(&["1", "/", "0"]).unwrap_err(), "division by zero");
    }

    #[test]
    fn colon_returns_capture_or_length() {
        assert_eq!(
            eval(&["hello", ":", r"h\(...\)o"]).unwrap().to_display(),
            "ell"
        );
        assert_eq!(eval(&["hello", ":", "hel"]).unwrap().to_display(), "3");
        assert_eq!(eval(&["hello", ":", "xyz"]).unwrap().to_display(), "0");
    }

    #[test]
    fn string_comparison_falls_back_when_non_numeric() {
        assert_eq!(eval(&["abc", "<", "abd"]).unwrap().to_display(), "1");
    }

    #[test]
    fn or_of_two_falsy_operands_is_zero_not_empty() {
        assert_eq!(eval(&["0", "|", ""]).unwrap().to_display(), "0");
    }

    #[test]
    fn missing_argument_names_the_last_token() {
        assert_eq!(
            eval(&["1", "+"]).unwrap_err(),
            "syntax error: missing argument after '+'"
        );
        assert_eq!(
            eval(&["match", "abc"]).unwrap_err(),
            "syntax error: missing argument after 'abc'"
        );
    }

    #[test]
    fn bare_keyword_always_requires_its_arguments() {
        // Real GNU expr never lets `length`/`match`/`index`/`substr` fall back to a literal
        // string; only `+ TOKEN` can force that.
        assert_eq!(
            eval(&["length"]).unwrap_err(),
            "syntax error: missing argument after 'length'"
        );
        assert_eq!(eval(&["+", "length"]).unwrap().to_display(), "length");
    }
}
