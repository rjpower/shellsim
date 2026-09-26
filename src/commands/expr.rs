//! POSIX `expr`: argument-vector arithmetic and string expressions.
//!
//! `expr` never touches shell state: each invocation parses its argv as one self-contained
//! expression and prints the result. Precedence (lowest to highest) follows POSIX: `|`, `&`,
//! the relational operators, additive, multiplicative, then `:` (anchored basic-regex match).
//! GNU extensions `match`, `substr`, `index`, `length`, and the `+ TOKEN` string escape are
//! supported at the primary level. Integers are evaluated with checked `i64` arithmetic so
//! overflow is reported instead of wrapping; anything else compares and concatenates as bytes.
//!
//! `:` reuses the same BRE-to-Rust-regex translation grep and sed already depend on
//! ([`crate::commands::regex_compat::basic_regex_to_rust`]) rather than adding a second engine.

use std::collections::HashMap;

use crate::commands::regex_compat::basic_regex_to_rust;
use crate::commands::util::{ewln, wln};
use crate::commands::{CommandSpec, Io, Trust};
use crate::program::ProcessContext;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    super::reg_system(m, "/usr/bin/expr", Trust::Real, run);
}

/// GNU-style syntax error status; also used for division by zero and overflow, matching the
/// task's relaxed status policy (GNU sometimes differentiates 2 vs 3; this build always uses 2).
const STATUS_ERROR: i32 = 2;

fn run(context: &mut ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    // Bound parse/regex work by the size of the argv actually supplied.
    let cost: u64 = args.iter().map(|a| a.len() as u64 + 1).sum();
    if !context.system.charge_cpu(cost.max(1)) {
        return context.system.stop_status();
    }
    if args.is_empty() {
        ewln(io.err, "expr: missing operand");
        return STATUS_ERROR;
    }
    let owned: Vec<String> = args.to_vec();
    let mut parser = Parser {
        tokens: &owned,
        pos: 0,
    };
    let result = parser.parse_expr().and_then(|value| {
        if parser.pos != parser.tokens.len() {
            Err(format!(
                "syntax error: unexpected argument '{}'",
                parser.tokens[parser.pos]
            ))
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
        Err(message) => {
            ewln(io.err, &format!("expr: {message}"));
            STATUS_ERROR
        }
    }
}

/// One evaluated expr value. Integers keep their numeric form so later arithmetic can reuse it
/// without reparsing; every value also has a canonical string form for display and comparison.
#[derive(Clone, Debug)]
enum Value {
    Int(i64),
    Str(String),
}

impl Value {
    fn to_display(&self) -> String {
        match self {
            Value::Int(n) => n.to_string(),
            Value::Str(s) => s.clone(),
        }
    }

    fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(n) => Some(*n),
            Value::Str(s) => parse_integer(s),
        }
    }

    fn is_falsy(&self) -> bool {
        matches!(self, Value::Int(0)) || matches!(self, Value::Str(s) if s.is_empty() || s == "0")
    }
}

/// Parse a POSIX integer literal: optional sign, then one or more decimal digits.
fn parse_integer(s: &str) -> Option<i64> {
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
    let magnitude = s[sign_len..].parse::<i64>().ok()?;
    if negative {
        magnitude.checked_neg()
    } else {
        Some(magnitude)
    }
}

struct Parser<'a> {
    tokens: &'a [String],
    pos: usize,
}

impl<'a> Parser<'a> {
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

    fn expect(&mut self, token: &str) -> Result<(), String> {
        match self.advance() {
            Some(t) if t == token => Ok(()),
            Some(t) => Err(format!("syntax error: expecting '{token}' but got '{t}'")),
            None => Err(format!("syntax error: expecting '{token}'")),
        }
    }

    fn parse_expr(&mut self) -> Result<Value, String> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Value, String> {
        let mut left = self.parse_and()?;
        while self.peek() == Some("|") {
            self.advance();
            let right = self.parse_and()?;
            left = if left.is_falsy() { right } else { left };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Value, String> {
        let mut left = self.parse_cmp()?;
        while self.peek() == Some("&") {
            self.advance();
            let right = self.parse_cmp()?;
            left = if left.is_falsy() || right.is_falsy() {
                Value::Int(0)
            } else {
                left
            };
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> Result<Value, String> {
        let mut left = self.parse_add()?;
        while let Some(op @ ("=" | "!=" | "<" | "<=" | ">" | ">=")) = self.peek() {
            self.advance();
            let right = self.parse_add()?;
            let outcome = compare(&left, op, &right)?;
            left = Value::Int(i64::from(outcome));
        }
        Ok(left)
    }

    fn parse_add(&mut self) -> Result<Value, String> {
        let mut left = self.parse_mul()?;
        while let Some(op @ ("+" | "-")) = self.peek() {
            self.advance();
            let right = self.parse_mul()?;
            let a = require_int(&left)?;
            let b = require_int(&right)?;
            let result = if op == "+" {
                a.checked_add(b)
            } else {
                a.checked_sub(b)
            };
            left = Value::Int(result.ok_or_else(too_large)?);
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<Value, String> {
        let mut left = self.parse_colon()?;
        while let Some(op @ ("*" | "/" | "%")) = self.peek() {
            self.advance();
            let right = self.parse_colon()?;
            let a = require_int(&left)?;
            let b = require_int(&right)?;
            let result = match op {
                "*" => a.checked_mul(b),
                "/" => {
                    if b == 0 {
                        return Err("division by zero".to_string());
                    }
                    a.checked_div(b)
                }
                _ => {
                    if b == 0 {
                        return Err("division by zero".to_string());
                    }
                    a.checked_rem(b)
                }
            };
            left = Value::Int(result.ok_or_else(too_large)?);
        }
        Ok(left)
    }

    fn parse_colon(&mut self) -> Result<Value, String> {
        let mut left = self.parse_primary()?;
        while self.peek() == Some(":") {
            self.advance();
            let right = self.parse_primary()?;
            left = Value::Str(bre_match(&left.to_display(), &right.to_display())?);
        }
        Ok(left)
    }

    fn parse_primary(&mut self) -> Result<Value, String> {
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
                    .ok_or_else(|| "syntax error: expecting operand after '+'".to_string())?;
                Ok(literal(token))
            }
            Some("length") if self.lookahead_is_keyword_call(1) => {
                self.advance();
                let value = self.parse_primary()?;
                Ok(Value::Int(value.to_display().chars().count() as i64))
            }
            Some("match") if self.lookahead_is_keyword_call(2) => {
                self.advance();
                let haystack = self.parse_primary()?.to_display();
                let pattern = self.parse_primary()?.to_display();
                Ok(Value::Str(bre_match(&haystack, &pattern)?))
            }
            Some("index") if self.lookahead_is_keyword_call(2) => {
                self.advance();
                let haystack = self.parse_primary()?.to_display();
                let chars = self.parse_primary()?.to_display();
                let position = haystack
                    .chars()
                    .position(|c| chars.contains(c))
                    .map_or(0, |i| i + 1);
                Ok(Value::Int(position as i64))
            }
            Some("substr") if self.lookahead_is_keyword_call(3) => {
                self.advance();
                let haystack = self.parse_primary()?.to_display();
                let start = require_int(&self.parse_primary()?)?;
                let len = require_int(&self.parse_primary()?)?;
                Ok(Value::Str(substr(&haystack, start, len)))
            }
            Some(_) => Ok(literal(self.advance().expect("peeked"))),
            None => Err("syntax error: unexpected end of expression".to_string()),
        }
    }

    /// A GNU keyword (`length`, `match`, `substr`, `index`) only introduces the special form
    /// when it is followed by exactly the operand tokens it needs, none of which are the end of
    /// input. This keeps a bare use of the word (e.g. as a literal string) working via the
    /// ordinary literal fallback.
    fn lookahead_is_keyword_call(&self, arity: usize) -> bool {
        self.pos + arity < self.tokens.len()
    }
}

fn literal(token: &str) -> Value {
    match parse_integer(token) {
        Some(n) => Value::Int(n),
        None => Value::Str(token.to_string()),
    }
}

fn require_int(value: &Value) -> Result<i64, String> {
    value
        .as_int()
        .ok_or_else(|| format!("non-integer argument '{}'", value.to_display()))
}

fn too_large() -> String {
    "integer result too large".to_string()
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
/// the empty string rather than an error.
fn substr(s: &str, start: i64, len: i64) -> String {
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
        };
        parser.parse_expr()
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
    fn overflow_is_rejected() {
        let err = eval(&[&i64::MAX.to_string(), "+", "1"]).unwrap_err();
        assert_eq!(err, "integer result too large");
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
}
