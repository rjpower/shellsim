//! A bounded POSIX `bc` subset: arbitrary-precision decimal arithmetic driven by a small
//! recursive-descent interpreter over `bc` source text.
//!
//! Design constraints that keep this a *simulated* calculator rather than a full `bc`:
//! - Numbers are `(BigInt, scale)` pairs (`value == BigInt * 10^-scale`); magnitude is capped at
//!   [`MAX_DIGITS`] decimal digits and `scale` at [`MAX_SCALE`], both charged against the
//!   process CPU meter before the (potentially expensive) bignum operation runs, so a hostile
//!   script cannot force unbounded host allocation.
//! - `ibase`/`obase` accept only the value `10`; other bases are rejected with a runtime
//!   diagnostic rather than silently misinterpreted.
//! - User-defined functions (`define`, `auto`, `return`) and arrays are rejected explicitly at
//!   parse time: this build only evaluates a flat script of statements and global scalars.
//! - A syntax error aborts the whole program (nothing after it could execute correctly); a
//!   runtime error (divide by zero, bad `ibase`/`obase`, oversize number) is printed and only
//!   the current top-level statement is abandoned, matching common `bc` behavior.
//!
//! Test strategy: unit tests below cover the bignum arithmetic and parser directly; integration
//! coverage (`tests/commands/bc.rs`) drives the command end to end including CLI flags, stdin,
//! and file operands.

use std::collections::HashMap;

use num_bigint::BigInt;
use num_traits::{Signed, Zero};

use crate::commands::util::{ewln, w, wln};
use crate::commands::{CommandSpec, Io, Trust};
use crate::exec::ShellPoll;
use crate::program::ProcessContext;
use crate::syscalls::System;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    super::reg_system_poll(m, "/usr/bin/bc", Trust::Partial, run);
}

/// Cap on total decimal digits (integer + fractional) kept in any one number. Chosen to bound
/// per-operation work (schoolbook multiplication is O(digits^2)) to a few hundred million
/// primitive digit operations in the worst case, which is fast but not free, hence the CPU
/// charge below.
const MAX_DIGITS: usize = 100_000;
/// Cap on the `scale` builtin/variable and on any single operation's target scale.
const MAX_SCALE: usize = 20_000;
/// GNU `bc`'s default `BC_LINE_LENGTH` is 70, and that count *includes* the trailing backslash
/// and newline of a continuation line (`man bc`: "This includes the backslash and newline
/// characters for long numbers"). So each continuation line carries 70 - 1 (`\`) - 1 (`\n`) = 68
/// digits; verified against GNU bc 1.07.1's actual output for `2^1000`.
const LINE_WIDTH: usize = 68;

fn run(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    if let Err(poll) = context.read_standard_input(io) {
        return poll;
    }
    let args = context.args;
    let mut files = Vec::new();
    for arg in args {
        match arg.as_str() {
            "-l" | "--mathlib" => {
                ewln(io.err, "bc: -l (math library) is not supported in shellsim");
                return ShellPoll::Ready(2);
            }
            "-q" | "--quiet" => {}
            other if other.starts_with('-') => {
                ewln(io.err, &format!("bc: unsupported option '{other}'"));
                return ShellPoll::Ready(2);
            }
            other => files.push(other.to_string()),
        }
    }
    let mut source = String::new();
    let cwd = context.system.cwd().to_string();
    for file in &files {
        match context
            .system
            .read_file_limited(&cwd, file, 16 * 1024 * 1024)
        {
            Ok(bytes) => source.push_str(&String::from_utf8_lossy(&bytes)),
            Err(error) => {
                ewln(io.err, &format!("bc: {file}: {error}"));
                return ShellPoll::Ready(1);
            }
        }
        source.push('\n');
    }
    source.push_str(&String::from_utf8_lossy(&io.stdin));

    if !context
        .system
        .charge_cpu((source.len() as u64).saturating_add(1))
    {
        return ShellPoll::Ready(context.system.stop_status());
    }

    let tokens = match lex(&source) {
        Ok(tokens) => tokens,
        Err(error) => {
            ewln(io.err, &format!("bc: syntax error: {error}"));
            return ShellPoll::Ready(1);
        }
    };
    let program = match Parser::new(&tokens).parse_program() {
        Ok(program) => program,
        Err(error) => {
            ewln(io.err, &format!("bc: syntax error: {error}"));
            return ShellPoll::Ready(1);
        }
    };
    let mut ctx = Ctx {
        vars: HashMap::new(),
        scale: 0,
        system: context.system,
    };
    ShellPoll::Ready(run_program(&mut ctx, &program, io))
}

// ---------------------------------------------------------------------------------------------
// Arbitrary-precision decimal
// ---------------------------------------------------------------------------------------------

/// A decimal number: `value * 10^-scale`. `scale` is the number of digits kept after the
/// decimal point, independent of trailing zeros (so `1/4` at `scale=4` is exactly `0.2500`).
#[derive(Clone, Debug, PartialEq, Eq)]
struct BigDec {
    value: BigInt,
    scale: usize,
}

impl BigDec {
    fn zero() -> Self {
        BigDec {
            value: BigInt::zero(),
            scale: 0,
        }
    }

    fn from_i64(n: i64) -> Self {
        BigDec {
            value: BigInt::from(n),
            scale: 0,
        }
    }

    fn digit_count(&self) -> usize {
        // `.to_string()` on the magnitude is the simplest correct digit count; call sites bound
        // the operands first so this never runs on unbounded input.
        self.value.magnitude().to_str_radix(10).len()
    }

    fn is_zero_value(&self) -> bool {
        self.value.is_zero()
    }

    /// Re-express at a new scale, truncating extra fractional digits toward zero (never
    /// rounding), matching every `bc` implementation's truncation behavior.
    fn rescale_to(&self, scale: usize) -> BigDec {
        if scale == self.scale {
            return self.clone();
        }
        if scale > self.scale {
            let factor = pow10(scale - self.scale);
            BigDec {
                value: &self.value * factor,
                scale,
            }
        } else {
            let factor = pow10(self.scale - scale);
            BigDec {
                value: &self.value / factor,
                scale,
            }
        }
    }

    fn add(a: &BigDec, b: &BigDec) -> BigDec {
        let scale = a.scale.max(b.scale);
        let av = a.rescale_to(scale).value;
        let bv = b.rescale_to(scale).value;
        BigDec {
            value: av + bv,
            scale,
        }
    }

    fn sub(a: &BigDec, b: &BigDec) -> BigDec {
        BigDec::add(a, &BigDec::neg(b))
    }

    fn neg(a: &BigDec) -> BigDec {
        BigDec {
            value: -&a.value,
            scale: a.scale,
        }
    }

    fn mul(a: &BigDec, b: &BigDec, current_scale: usize) -> BigDec {
        let raw_scale = a.scale + b.scale;
        let target_scale = raw_scale.min(a.scale.max(b.scale).max(current_scale));
        BigDec {
            value: &a.value * &b.value,
            scale: raw_scale,
        }
        .rescale_to(target_scale)
    }

    /// `a / b`, truncated to exactly `target_scale` fractional digits.
    fn div(a: &BigDec, b: &BigDec, target_scale: usize) -> Result<BigDec, String> {
        if b.is_zero_value() {
            return Err("Divide by zero".to_string());
        }
        let num_exp = b.scale + target_scale;
        let den_exp = a.scale;
        let numerator = &a.value * pow10(num_exp);
        let denominator = &b.value * pow10(den_exp);
        Ok(BigDec {
            value: numerator / denominator,
            scale: target_scale,
        })
    }

    /// `a % b`, defined (as POSIX does) as `a - (a/b)*b` where the internal division truncates
    /// to `current_scale`; the result's scale is `max(a.scale, b.scale + current_scale)`.
    fn rem(a: &BigDec, b: &BigDec, current_scale: usize) -> Result<BigDec, String> {
        let q = BigDec::div(a, b, current_scale)?;
        let prod = BigDec::mul(&q, b, current_scale);
        let result = BigDec::sub(a, &prod);
        let target_scale = a.scale.max(b.scale + current_scale);
        Ok(result.rescale_to(target_scale))
    }

    /// `a ^ exponent`, exponent must be an integer (POSIX allows a negative one: `a^-n` is
    /// `1 / a^n` computed at `current_scale`, matching GNU `bc`).
    fn pow(a: &BigDec, exponent: &BigDec, current_scale: usize) -> Result<BigDec, String> {
        if exponent.scale != 0 {
            return Err("non-integer exponent".to_string());
        }
        let negative_exponent = exponent.value.is_negative();
        let magnitude = exponent.value.magnitude().clone();
        let e: u32 = magnitude
            .try_into()
            .map_err(|_| "exponent too large".to_string())?;
        let raw_scale = a
            .scale
            .checked_mul(e as usize)
            .filter(|s| *s <= MAX_SCALE * 4)
            .ok_or_else(|| "exponent too large".to_string())?;
        // Reject up front (before allocating) if the magnitude would blow past the digit cap:
        // digit count roughly scales with exponent * bit length of the base.
        let estimated_bits = a.value.bits().saturating_mul(u64::from(e));
        if estimated_bits > (MAX_DIGITS as u64) * 4 {
            return Err("result too large".to_string());
        }
        let value = a.value.pow(e);
        let target_scale = raw_scale.min(a.scale.max(current_scale));
        let result = BigDec {
            value,
            scale: raw_scale,
        }
        .rescale_to(target_scale);
        if !negative_exponent {
            return Ok(result);
        }
        if result.is_zero_value() {
            return Err("Divide by zero".to_string());
        }
        BigDec::div(&BigDec::from_i64(1), &result, current_scale)
    }

    fn sqrt(a: &BigDec, target_scale: usize) -> Result<BigDec, String> {
        if a.value.is_negative() {
            return Err("square root of a negative number".to_string());
        }
        let magnitude = a.value.magnitude().clone();
        // We want floor(sqrt(A * 10^(2*ts - sa))) where A is the stored integer and sa its scale.
        let exponent = 2 * target_scale as i64 - a.scale as i64;
        let scaled = if exponent >= 0 {
            &magnitude * pow10(exponent as usize).magnitude().clone()
        } else {
            &magnitude / pow10((-exponent) as usize).magnitude().clone()
        };
        let root = isqrt(&scaled);
        Ok(BigDec {
            value: BigInt::from(root),
            scale: target_scale,
        })
    }
}

fn pow10(n: usize) -> BigInt {
    BigInt::from(10u32).pow(n as u32)
}

/// Floor integer square root via Newton's method; converges in O(log(bits)) iterations.
fn isqrt(n: &num_bigint::BigUint) -> num_bigint::BigUint {
    use num_bigint::BigUint;
    if n.is_zero() {
        return BigUint::zero();
    }
    let mut x = BigUint::from(1u32) << ((n.bits() as usize) / 2 + 1);
    loop {
        let y = (&x + n / &x) >> 1u32;
        if y >= x {
            return x;
        }
        x = y;
    }
}

fn format_bigdec(v: &BigDec) -> String {
    if v.is_zero_value() {
        return "0".to_string();
    }
    let negative = v.value.is_negative();
    let digits = v.value.magnitude().to_str_radix(10);
    let (int_str, frac_str) = if v.scale == 0 {
        (digits.as_str(), "")
    } else if digits.len() <= v.scale {
        // Fewer digits than the scale: the integer part is entirely a leading zero we drop and
        // the fraction is left-padded with zeros to `scale` digits.
        let pad = "0".repeat(v.scale - digits.len());
        return finish_number(negative, "", &format!("{pad}{digits}"));
    } else {
        let split = digits.len() - v.scale;
        (&digits[..split], &digits[split..])
    };
    finish_number(negative, int_str, frac_str)
}

fn finish_number(negative: bool, int_str: &str, frac_str: &str) -> String {
    let mut s = String::new();
    if negative {
        s.push('-');
    }
    if !int_str.is_empty() {
        s.push_str(int_str);
    }
    if !frac_str.is_empty() {
        s.push('.');
        s.push_str(frac_str);
    }
    wrap_number(&s)
}

/// Break a long printed number into `LINE_WIDTH`-character chunks joined by `\` + newline,
/// matching GNU `bc`'s default 70-column output wrapping.
fn wrap_number(s: &str) -> String {
    if s.len() <= LINE_WIDTH {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + s.len() / LINE_WIDTH * 2);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && i % LINE_WIDTH == 0 {
            out.push('\\');
            out.push('\n');
        }
        out.push(ch);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Num(String),
    Str(String),
    Ident(String),
    If,
    While,
    For,
    Break,
    Quit,
    Define,
    Sqrt,
    Length,
    Scale,
    Ibase,
    Obase,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    Assign,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    CaretEq,
    EqEq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Inc,
    Dec,
    LParen,
    RParen,
    LBrace,
    RBrace,
    Semicolon,
    Newline,
}

fn lex(source: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = source.chars().collect();
    let mut i = 0;
    let mut tokens = Vec::new();
    let mut paren_depth: i32 = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\r' => i += 1,
            '\n' => {
                if paren_depth == 0 {
                    tokens.push(Token::Newline);
                }
                i += 1;
            }
            '#' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '/' if chars.get(i + 1) == Some(&'*') => {
                i += 2;
                loop {
                    if i + 1 >= chars.len() {
                        return Err("unterminated comment".to_string());
                    }
                    if chars[i] == '*' && chars[i + 1] == '/' {
                        i += 2;
                        break;
                    }
                    i += 1;
                }
            }
            '0'..='9' | '.' => {
                let start = i;
                let mut seen_dot = false;
                while i < chars.len()
                    && (chars[i].is_ascii_digit() || (chars[i] == '.' && !seen_dot))
                {
                    if chars[i] == '.' {
                        seen_dot = true;
                    }
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                if text.chars().filter(|c| *c == '.').count() > 1 || text == "." {
                    return Err(format!("invalid number '{text}'"));
                }
                let digit_count = text.chars().filter(char::is_ascii_digit).count();
                if digit_count > MAX_DIGITS {
                    return Err("number exceeds the supported digit limit".to_string());
                }
                tokens.push(Token::Num(text));
            }
            '"' => {
                let mut s = String::new();
                i += 1;
                loop {
                    match chars.get(i) {
                        None => return Err("unterminated string".to_string()),
                        Some('"') => {
                            i += 1;
                            break;
                        }
                        Some(ch) => {
                            s.push(*ch);
                            i += 1;
                        }
                    }
                }
                tokens.push(Token::Str(s));
            }
            'a'..='z' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_lowercase()
                        || chars[i].is_ascii_digit()
                        || chars[i] == '_')
                {
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                tokens.push(match text.as_str() {
                    "if" => Token::If,
                    "while" => Token::While,
                    "for" => Token::For,
                    "break" => Token::Break,
                    "quit" => Token::Quit,
                    "define" | "auto" | "return" => Token::Define,
                    "sqrt" => Token::Sqrt,
                    "length" => Token::Length,
                    "scale" => Token::Scale,
                    "ibase" => Token::Ibase,
                    "obase" => Token::Obase,
                    _ => Token::Ident(text),
                });
            }
            '+' if chars.get(i + 1) == Some(&'+') => {
                tokens.push(Token::Inc);
                i += 2;
            }
            '-' if chars.get(i + 1) == Some(&'-') => {
                tokens.push(Token::Dec);
                i += 2;
            }
            '+' if chars.get(i + 1) == Some(&'=') => {
                tokens.push(Token::PlusEq);
                i += 2;
            }
            '-' if chars.get(i + 1) == Some(&'=') => {
                tokens.push(Token::MinusEq);
                i += 2;
            }
            '*' if chars.get(i + 1) == Some(&'=') => {
                tokens.push(Token::StarEq);
                i += 2;
            }
            '/' if chars.get(i + 1) == Some(&'=') => {
                tokens.push(Token::SlashEq);
                i += 2;
            }
            '%' if chars.get(i + 1) == Some(&'=') => {
                tokens.push(Token::PercentEq);
                i += 2;
            }
            '^' if chars.get(i + 1) == Some(&'=') => {
                tokens.push(Token::CaretEq);
                i += 2;
            }
            '=' if chars.get(i + 1) == Some(&'=') => {
                tokens.push(Token::EqEq);
                i += 2;
            }
            '!' if chars.get(i + 1) == Some(&'=') => {
                tokens.push(Token::Ne);
                i += 2;
            }
            '<' if chars.get(i + 1) == Some(&'=') => {
                tokens.push(Token::Le);
                i += 2;
            }
            '>' if chars.get(i + 1) == Some(&'=') => {
                tokens.push(Token::Ge);
                i += 2;
            }
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '%' => {
                tokens.push(Token::Percent);
                i += 1;
            }
            '^' => {
                tokens.push(Token::Caret);
                i += 1;
            }
            '=' => {
                tokens.push(Token::Assign);
                i += 1;
            }
            '<' => {
                tokens.push(Token::Lt);
                i += 1;
            }
            '>' => {
                tokens.push(Token::Gt);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                paren_depth += 1;
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                paren_depth = (paren_depth - 1).max(0);
                i += 1;
            }
            '{' => {
                tokens.push(Token::LBrace);
                i += 1;
            }
            '}' => {
                tokens.push(Token::RBrace);
                i += 1;
            }
            ';' => {
                tokens.push(Token::Semicolon);
                i += 1;
            }
            '&' | '|' | '!' => {
                return Err(format!("logical operator '{c}' is not supported"));
            }
            '[' | ']' => {
                return Err("arrays are not supported".to_string());
            }
            other => return Err(format!("unexpected character '{other}'")),
        }
    }
    Ok(tokens)
}

// ---------------------------------------------------------------------------------------------
// Parser / AST
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum LValue {
    Var(String),
    Scale,
    Ibase,
    Obase,
}

#[derive(Clone, Copy, Debug)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
}

#[derive(Clone, Copy, Debug)]
enum RelOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Debug)]
enum Expr {
    Num(BigDec),
    Var(LValue),
    Assign(LValue, Box<Expr>),
    CompoundAssign(LValue, BinOp, Box<Expr>),
    PreIncDec(LValue, bool),
    PostIncDec(LValue, bool),
    Neg(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Rel(RelOp, Box<Expr>, Box<Expr>),
    Sqrt(Box<Expr>),
    Length(Box<Expr>),
    ScaleOf(Box<Expr>),
}

#[derive(Clone, Debug)]
enum Stmt {
    Empty,
    Eval(Expr),
    PrintString(String),
    Block(Vec<Stmt>),
    If(Expr, Box<Stmt>),
    While(Expr, Box<Stmt>),
    For(Option<Expr>, Option<Expr>, Option<Expr>, Box<Stmt>),
    Break,
    Quit,
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        match self.advance() {
            Some(t) if t == expected => Ok(()),
            Some(t) => Err(format!("expected {expected:?} but found {t:?}")),
            None => Err(format!("expected {expected:?} but reached end of input")),
        }
    }

    fn skip_separators(&mut self) {
        while matches!(self.peek(), Some(Token::Semicolon) | Some(Token::Newline)) {
            self.pos += 1;
        }
    }

    fn parse_program(&mut self) -> Result<Vec<Stmt>, String> {
        let mut stmts = Vec::new();
        self.skip_separators();
        while self.peek().is_some() {
            stmts.push(self.parse_stmt()?);
            self.skip_separators();
        }
        Ok(stmts)
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, String> {
        let mut stmts = Vec::new();
        self.skip_separators();
        while !matches!(self.peek(), Some(Token::RBrace) | None) {
            stmts.push(self.parse_stmt()?);
            self.skip_separators();
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        match self.peek() {
            Some(Token::LBrace) => {
                self.advance();
                let body = self.parse_block()?;
                self.expect(&Token::RBrace)?;
                Ok(Stmt::Block(body))
            }
            Some(Token::If) => {
                self.advance();
                self.expect(&Token::LParen)?;
                let cond = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                self.skip_separators_within_stmt();
                let body = Box::new(self.parse_stmt()?);
                Ok(Stmt::If(cond, body))
            }
            Some(Token::While) => {
                self.advance();
                self.expect(&Token::LParen)?;
                let cond = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                self.skip_separators_within_stmt();
                let body = Box::new(self.parse_stmt()?);
                Ok(Stmt::While(cond, body))
            }
            Some(Token::For) => {
                self.advance();
                self.expect(&Token::LParen)?;
                let init = if matches!(self.peek(), Some(Token::Semicolon)) {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                self.expect(&Token::Semicolon)?;
                let cond = if matches!(self.peek(), Some(Token::Semicolon)) {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                self.expect(&Token::Semicolon)?;
                let step = if matches!(self.peek(), Some(Token::RParen)) {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                self.expect(&Token::RParen)?;
                self.skip_separators_within_stmt();
                let body = Box::new(self.parse_stmt()?);
                Ok(Stmt::For(init, cond, step, body))
            }
            Some(Token::Break) => {
                self.advance();
                Ok(Stmt::Break)
            }
            Some(Token::Quit) => {
                self.advance();
                Ok(Stmt::Quit)
            }
            Some(Token::Define) => Err(
                "user-defined functions ('define'/'auto'/'return') are not supported".to_string(),
            ),
            Some(Token::Str(_)) => {
                let Some(Token::Str(s)) = self.advance().cloned() else {
                    unreachable!("peeked Str above")
                };
                Ok(Stmt::PrintString(s))
            }
            Some(_) => Ok(Stmt::Eval(self.parse_expr()?)),
            None => Ok(Stmt::Empty),
        }
    }

    /// `if (c) \n stmt` is common in scripts; a single newline directly after the header is not
    /// a statement separator here, it just precedes the body statement.
    fn skip_separators_within_stmt(&mut self) {
        while matches!(self.peek(), Some(Token::Newline)) {
            self.pos += 1;
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_assign()
    }

    fn parse_assign(&mut self) -> Result<Expr, String> {
        let left = self.parse_rel()?;
        let op = match self.peek() {
            Some(Token::Assign) => None,
            Some(Token::PlusEq) => Some(BinOp::Add),
            Some(Token::MinusEq) => Some(BinOp::Sub),
            Some(Token::StarEq) => Some(BinOp::Mul),
            Some(Token::SlashEq) => Some(BinOp::Div),
            Some(Token::PercentEq) => Some(BinOp::Mod),
            Some(Token::CaretEq) => Some(BinOp::Pow),
            _ => return Ok(left),
        };
        let Expr::Var(lvalue) = left else {
            return Err("assignment target must be a variable".to_string());
        };
        self.advance();
        let right = self.parse_assign()?;
        Ok(match op {
            None => Expr::Assign(lvalue, Box::new(right)),
            Some(op) => Expr::CompoundAssign(lvalue, op, Box::new(right)),
        })
    }

    fn parse_rel(&mut self) -> Result<Expr, String> {
        let left = self.parse_add()?;
        let op = match self.peek() {
            Some(Token::EqEq) => RelOp::Eq,
            Some(Token::Ne) => RelOp::Ne,
            Some(Token::Lt) => RelOp::Lt,
            Some(Token::Le) => RelOp::Le,
            Some(Token::Gt) => RelOp::Gt,
            Some(Token::Ge) => RelOp::Ge,
            _ => return Ok(left),
        };
        self.advance();
        let right = self.parse_add()?;
        Ok(Expr::Rel(op, Box::new(left), Box::new(right)))
    }

    fn parse_add(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Some(Token::Plus) => BinOp::Add,
                Some(Token::Minus) => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_mul()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Some(Token::Star) => BinOp::Mul,
                Some(Token::Slash) => BinOp::Div,
                Some(Token::Percent) => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        if matches!(self.peek(), Some(Token::Minus)) {
            self.advance();
            return Ok(Expr::Neg(Box::new(self.parse_unary()?)));
        }
        self.parse_pow()
    }

    fn parse_pow(&mut self) -> Result<Expr, String> {
        let left = self.parse_postfix()?;
        if matches!(self.peek(), Some(Token::Caret)) {
            self.advance();
            let right = self.parse_unary()?;
            return Ok(Expr::Bin(BinOp::Pow, Box::new(left), Box::new(right)));
        }
        Ok(left)
    }

    fn parse_postfix(&mut self) -> Result<Expr, String> {
        let primary = self.parse_primary()?;
        if let Expr::Var(lvalue) = &primary {
            match self.peek() {
                Some(Token::Inc) => {
                    self.advance();
                    return Ok(Expr::PostIncDec(lvalue.clone(), true));
                }
                Some(Token::Dec) => {
                    self.advance();
                    return Ok(Expr::PostIncDec(lvalue.clone(), false));
                }
                _ => {}
            }
        }
        Ok(primary)
    }

    fn parse_lvalue(&mut self) -> Result<LValue, String> {
        match self.advance() {
            Some(Token::Ident(name)) => Ok(LValue::Var(name.clone())),
            Some(Token::Scale) => Ok(LValue::Scale),
            Some(Token::Ibase) => Ok(LValue::Ibase),
            Some(Token::Obase) => Ok(LValue::Obase),
            other => Err(format!("expected a variable, found {other:?}")),
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.peek().cloned() {
            Some(Token::Num(text)) => {
                self.advance();
                Ok(Expr::Num(parse_number(&text)))
            }
            Some(Token::LParen) => {
                self.advance();
                let inner = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            Some(Token::Inc) => {
                self.advance();
                Ok(Expr::PreIncDec(self.parse_lvalue()?, true))
            }
            Some(Token::Dec) => {
                self.advance();
                Ok(Expr::PreIncDec(self.parse_lvalue()?, false))
            }
            Some(Token::Sqrt) => {
                self.advance();
                self.expect(&Token::LParen)?;
                let inner = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(Expr::Sqrt(Box::new(inner)))
            }
            Some(Token::Length) => {
                self.advance();
                self.expect(&Token::LParen)?;
                let inner = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(Expr::Length(Box::new(inner)))
            }
            Some(Token::Scale) => {
                self.advance();
                if matches!(self.peek(), Some(Token::LParen)) {
                    self.advance();
                    let inner = self.parse_expr()?;
                    self.expect(&Token::RParen)?;
                    Ok(Expr::ScaleOf(Box::new(inner)))
                } else {
                    Ok(Expr::Var(LValue::Scale))
                }
            }
            Some(Token::Ibase) => {
                self.advance();
                Ok(Expr::Var(LValue::Ibase))
            }
            Some(Token::Obase) => {
                self.advance();
                Ok(Expr::Var(LValue::Obase))
            }
            Some(Token::Ident(name)) => {
                self.advance();
                Ok(Expr::Var(LValue::Var(name)))
            }
            Some(Token::Str(_)) => Err("strings are not valid inside expressions".to_string()),
            other => Err(format!("unexpected token {other:?}")),
        }
    }
}

fn parse_number(text: &str) -> BigDec {
    let (int_part, frac_part) = match text.split_once('.') {
        Some((i, f)) => (i, f),
        None => (text, ""),
    };
    let digits = format!("{int_part}{frac_part}");
    let digits = if digits.is_empty() { "0" } else { &digits };
    let value = digits.parse::<BigInt>().unwrap_or_else(|_| BigInt::zero());
    BigDec {
        value,
        scale: frac_part.len(),
    }
}

// ---------------------------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------------------------

enum Flow {
    Normal,
    Break,
    Quit,
}

enum Abort {
    /// Printed to stderr; only the current top-level statement is abandoned.
    Runtime(String),
    /// The process CPU budget ran out; stop the whole command immediately.
    Resource,
}

struct Ctx<'a> {
    vars: HashMap<String, BigDec>,
    scale: usize,
    system: &'a mut dyn System,
}

impl Ctx<'_> {
    fn charge(&mut self, units: u64) -> Result<(), Abort> {
        if self.system.charge_cpu(units.max(1)) {
            Ok(())
        } else {
            Err(Abort::Resource)
        }
    }

    fn read_var(&self, lvalue: &LValue) -> BigDec {
        match lvalue {
            LValue::Var(name) => self.vars.get(name).cloned().unwrap_or_else(BigDec::zero),
            LValue::Scale => BigDec::from_i64(self.scale as i64),
            LValue::Ibase | LValue::Obase => BigDec::from_i64(10),
        }
    }

    fn write_var(&mut self, lvalue: &LValue, value: BigDec) -> Result<(), Abort> {
        match lvalue {
            LValue::Var(name) => {
                if value.digit_count() > MAX_DIGITS {
                    return Err(Abort::Runtime(
                        "number exceeds the supported digit limit".to_string(),
                    ));
                }
                self.vars.insert(name.clone(), value);
                Ok(())
            }
            LValue::Scale => {
                if value.scale != 0 || value.value.is_negative() {
                    return Err(Abort::Runtime(
                        "scale must be a non-negative integer".into(),
                    ));
                }
                let n: usize = value
                    .value
                    .try_into()
                    .map_err(|_| Abort::Runtime("scale value is out of range".into()))?;
                if n > MAX_SCALE {
                    return Err(Abort::Runtime(format!(
                        "scale exceeds the supported limit of {MAX_SCALE}"
                    )));
                }
                self.scale = n;
                Ok(())
            }
            LValue::Ibase | LValue::Obase => {
                if value.scale != 0 || value.value != BigInt::from(10) {
                    return Err(Abort::Runtime(
                        "only base 10 is supported for ibase/obase".to_string(),
                    ));
                }
                Ok(())
            }
        }
    }
}

fn eval_expr(ctx: &mut Ctx, expr: &Expr) -> Result<BigDec, Abort> {
    match expr {
        Expr::Num(n) => Ok(n.clone()),
        Expr::Var(lvalue) => Ok(ctx.read_var(lvalue)),
        Expr::Assign(lvalue, rhs) => {
            let value = eval_expr(ctx, rhs)?;
            ctx.write_var(lvalue, value.clone())?;
            Ok(value)
        }
        Expr::CompoundAssign(lvalue, op, rhs) => {
            let current = ctx.read_var(lvalue);
            let rhs_value = eval_expr(ctx, rhs)?;
            let value = apply_binop(ctx, *op, &current, &rhs_value)?;
            ctx.write_var(lvalue, value.clone())?;
            Ok(value)
        }
        Expr::PreIncDec(lvalue, inc) => {
            let current = ctx.read_var(lvalue);
            let value = step(&current, *inc);
            ctx.write_var(lvalue, value.clone())?;
            Ok(value)
        }
        Expr::PostIncDec(lvalue, inc) => {
            let current = ctx.read_var(lvalue);
            let value = step(&current, *inc);
            ctx.write_var(lvalue, value)?;
            Ok(current)
        }
        Expr::Neg(inner) => Ok(BigDec::neg(&eval_expr(ctx, inner)?)),
        Expr::Bin(op, l, r) => {
            let lv = eval_expr(ctx, l)?;
            let rv = eval_expr(ctx, r)?;
            apply_binop(ctx, *op, &lv, &rv)
        }
        Expr::Rel(op, l, r) => {
            let lv = eval_expr(ctx, l)?;
            let rv = eval_expr(ctx, r)?;
            let scale = lv.scale.max(rv.scale);
            let ordering = lv.rescale_to(scale).value.cmp(&rv.rescale_to(scale).value);
            let truth = match op {
                RelOp::Eq => ordering.is_eq(),
                RelOp::Ne => ordering.is_ne(),
                RelOp::Lt => ordering.is_lt(),
                RelOp::Le => ordering.is_le(),
                RelOp::Gt => ordering.is_gt(),
                RelOp::Ge => ordering.is_ge(),
            };
            Ok(BigDec::from_i64(i64::from(truth)))
        }
        Expr::Sqrt(inner) => {
            let value = eval_expr(ctx, inner)?;
            ctx.charge(value.digit_count() as u64 * 4 + 16)?;
            BigDec::sqrt(&value, ctx.scale.max(value.scale)).map_err(Abort::Runtime)
        }
        Expr::Length(inner) => {
            let value = eval_expr(ctx, inner)?;
            Ok(BigDec::from_i64(length_of(&value) as i64))
        }
        Expr::ScaleOf(inner) => {
            let value = eval_expr(ctx, inner)?;
            Ok(BigDec::from_i64(value.scale as i64))
        }
    }
}

fn step(current: &BigDec, inc: bool) -> BigDec {
    let one = BigDec::from_i64(1);
    if inc {
        BigDec::add(current, &one)
    } else {
        BigDec::sub(current, &one)
    }
}

fn length_of(v: &BigDec) -> usize {
    if v.is_zero_value() {
        return 1;
    }
    let digits = v.value.magnitude().to_str_radix(10);
    let int_len = digits.len().saturating_sub(v.scale);
    int_len + v.scale
}

fn apply_binop(ctx: &mut Ctx, op: BinOp, a: &BigDec, b: &BigDec) -> Result<BigDec, Abort> {
    // Charge roughly proportional to the underlying bignum work before doing it, per the
    // repository's "meter before unbounded host work" rule; multiplication and power are the
    // only superlinear operations here.
    let cost = match op {
        BinOp::Add | BinOp::Sub => a.digit_count().max(b.digit_count()) as u64,
        BinOp::Mul => (a.digit_count() as u64) * (b.digit_count() as u64) / 8 + 1,
        BinOp::Div | BinOp::Mod => (a.digit_count().max(b.digit_count()) as u64) * 4 + 16,
        BinOp::Pow => a.digit_count() as u64 * 4 + 16,
    };
    ctx.charge(cost)?;
    let result = match op {
        BinOp::Add => Ok(BigDec::add(a, b)),
        BinOp::Sub => Ok(BigDec::sub(a, b)),
        BinOp::Mul => Ok(BigDec::mul(a, b, ctx.scale)),
        BinOp::Div => BigDec::div(a, b, ctx.scale),
        BinOp::Mod => BigDec::rem(a, b, ctx.scale),
        BinOp::Pow => BigDec::pow(a, b, ctx.scale),
    }
    .map_err(Abort::Runtime)?;
    if result.digit_count() > MAX_DIGITS {
        return Err(Abort::Runtime(
            "number exceeds the supported digit limit".to_string(),
        ));
    }
    Ok(result)
}

fn is_zero_value(v: &BigDec) -> bool {
    v.is_zero_value()
}

fn exec_stmt(ctx: &mut Ctx, stmt: &Stmt, io: &mut Io) -> Result<Flow, Abort> {
    ctx.charge(2)?;
    match stmt {
        Stmt::Empty => Ok(Flow::Normal),
        Stmt::Eval(expr) => {
            let value = eval_expr(ctx, expr)?;
            if !matches!(expr, Expr::Assign(..) | Expr::CompoundAssign(..)) {
                let text = format_bigdec(&value);
                wln(io.out, &text);
            }
            Ok(Flow::Normal)
        }
        Stmt::PrintString(s) => {
            // A `bc` string literal statement prints its raw bytes (no escape processing, no
            // automatic trailing newline); only expression statements print a trailing `\n`.
            w(io.out, s);
            Ok(Flow::Normal)
        }
        Stmt::Block(list) => exec_list(ctx, list, io),
        Stmt::If(cond, body) => {
            if !is_zero_value(&eval_expr(ctx, cond)?) {
                exec_stmt(ctx, body, io)
            } else {
                Ok(Flow::Normal)
            }
        }
        Stmt::While(cond, body) => {
            loop {
                ctx.charge(2)?;
                if is_zero_value(&eval_expr(ctx, cond)?) {
                    break;
                }
                match exec_stmt(ctx, body, io)? {
                    Flow::Break => break,
                    Flow::Quit => return Ok(Flow::Quit),
                    Flow::Normal => {}
                }
            }
            Ok(Flow::Normal)
        }
        Stmt::For(init, cond, step_expr, body) => {
            if let Some(e) = init {
                eval_expr(ctx, e)?;
            }
            loop {
                ctx.charge(2)?;
                let go = match cond {
                    Some(e) => !is_zero_value(&eval_expr(ctx, e)?),
                    None => true,
                };
                if !go {
                    break;
                }
                match exec_stmt(ctx, body, io)? {
                    Flow::Break => break,
                    Flow::Quit => return Ok(Flow::Quit),
                    Flow::Normal => {}
                }
                if let Some(e) = step_expr {
                    eval_expr(ctx, e)?;
                }
            }
            Ok(Flow::Normal)
        }
        Stmt::Break => Ok(Flow::Break),
        Stmt::Quit => Ok(Flow::Quit),
    }
}

fn exec_list(ctx: &mut Ctx, list: &[Stmt], io: &mut Io) -> Result<Flow, Abort> {
    for stmt in list {
        match exec_stmt(ctx, stmt, io)? {
            Flow::Normal => {}
            other => return Ok(other),
        }
    }
    Ok(Flow::Normal)
}

fn run_program(ctx: &mut Ctx, program: &[Stmt], io: &mut Io) -> i32 {
    for stmt in program {
        match exec_stmt(ctx, stmt, io) {
            Ok(Flow::Quit) => return 0,
            Ok(_) => {}
            Err(Abort::Runtime(message)) => {
                // GNU bc's format is "Runtime error (func=(main), adr=N): <message>"; `adr` is
                // an internal bytecode address with no equivalent here, so it is omitted.
                ewln(io.err, &format!("Runtime error (func=(main)): {message}"));
            }
            Err(Abort::Resource) => return ctx.system.stop_status(),
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dec(text: &str) -> BigDec {
        parse_number(text)
    }

    #[test]
    fn add_sub_align_scales() {
        let a = dec("1.5");
        let b = dec("2.25");
        assert_eq!(format_bigdec(&BigDec::add(&a, &b)), "3.75");
        assert_eq!(format_bigdec(&BigDec::sub(&a, &b)), "-.75");
    }

    #[test]
    fn division_truncates_to_target_scale() {
        let a = dec("10");
        let b = dec("3");
        let q = BigDec::div(&a, &b, 4).unwrap();
        assert_eq!(format_bigdec(&q), "3.3333");
    }

    #[test]
    fn division_by_zero_is_reported_not_panicked() {
        let a = dec("1");
        let b = dec("0");
        assert_eq!(BigDec::div(&a, &b, 2).unwrap_err(), "Divide by zero");
    }

    #[test]
    fn multiplication_scale_matches_posix_rule() {
        // scale(a) + scale(b) = 2, current scale = 0, so the min(2, max(1,1,0)) = 1 rule kicks
        // in and the result keeps only one fractional digit.
        let a = dec("1.5");
        let b = dec("1.5");
        assert_eq!(format_bigdec(&BigDec::mul(&a, &b, 0)), "2.2");
    }

    #[test]
    fn sqrt_matches_known_values() {
        let two = dec("2");
        assert_eq!(format_bigdec(&BigDec::sqrt(&two, 4).unwrap()), "1.4142");
    }

    #[test]
    fn negative_sqrt_is_rejected() {
        let neg = dec("-4");
        assert!(BigDec::sqrt(&neg, 2).is_err());
    }

    #[test]
    fn negative_exponent_is_a_reciprocal_at_the_current_scale() {
        let two = dec("2");
        let neg_one = dec("-1");
        assert_eq!(format_bigdec(&BigDec::pow(&two, &neg_one, 0).unwrap()), "0");
        let neg_two = dec("-2");
        assert_eq!(
            format_bigdec(&BigDec::pow(&two, &neg_two, 3).unwrap()),
            ".250"
        );
    }

    #[test]
    fn zero_to_a_negative_power_is_divide_by_zero() {
        let zero = dec("0");
        let neg_one = dec("-1");
        assert_eq!(
            BigDec::pow(&zero, &neg_one, 0).unwrap_err(),
            "Divide by zero"
        );
    }

    #[test]
    fn number_formatting_drops_leading_zero() {
        assert_eq!(format_bigdec(&dec(".5")), ".5");
        assert_eq!(format_bigdec(&dec("0.001")), ".001");
        assert_eq!(format_bigdec(&dec("0")), "0");
    }

    #[test]
    fn long_numbers_wrap_at_68_digits_per_line() {
        // GNU bc's default BC_LINE_LENGTH (70) counts the trailing `\` and `\n` themselves, so
        // 68 digits + `\` + `\n` = 70 columns; verified against GNU bc 1.07.1's real output for
        // `2^1000` (68 digits on each continuation line).
        let digits = "1".repeat(140);
        let wrapped = wrap_number(&digits);
        assert!(wrapped.contains("\\\n"));
        assert_eq!(wrapped.split("\\\n").next().unwrap().len(), 68);
        assert_eq!(LINE_WIDTH, 68);
    }

    #[test]
    fn lexer_rejects_logical_operators() {
        assert!(lex("1 && 2").is_err());
    }

    #[test]
    fn lexer_rejects_arrays() {
        assert!(lex("a[0] = 1").is_err());
    }

    #[test]
    fn parser_rejects_define() {
        let tokens = lex("define f(x) { return x }").unwrap();
        assert!(Parser::new(&tokens).parse_program().is_err());
    }

    #[test]
    fn parser_builds_expected_shape_for_control_flow() {
        let tokens = lex("i = 0\nwhile (i < 3) { i += 1 }\n").unwrap();
        let program = Parser::new(&tokens).parse_program().expect("parse");
        assert_eq!(program.len(), 2);
        assert!(matches!(program[1], Stmt::While(..)));
    }

    #[test]
    fn assignment_expression_is_distinguished_from_other_exprs() {
        let tokens = lex("a = 5\n").unwrap();
        let program = Parser::new(&tokens).parse_program().expect("parse");
        match &program[0] {
            Stmt::Eval(Expr::Assign(..)) => {}
            other => panic!("expected a silent assignment statement, got {other:?}"),
        }
    }
}
