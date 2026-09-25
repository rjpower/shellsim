//! A small UTF-8-aware lexer. It owns Python escape handling rather than reusing shell rules.

use num_bigint::BigInt;
use num_traits::ToPrimitive;

use super::source::Span;
use super::token::{Token, TokenKind};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    Lexer::new(source).lex()
}

struct Lexer<'a> {
    source: &'a str,
    offset: usize,
    line: usize,
    column: usize,
    at_line_start: bool,
    nesting: usize,
    indents: Vec<usize>,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            offset: 0,
            line: 1,
            column: 1,
            at_line_start: true,
            nesting: 0,
            indents: vec![0],
        }
    }

    fn lex(mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::new();
        while self.peek().is_some() {
            if self.at_line_start && self.nesting == 0 {
                self.indentation(&mut tokens)?;
                if self.peek().is_none() {
                    break;
                }
            }
            let ch = self.peek().expect("checked above");
            if matches!(ch, ' ' | '\t' | '\r') {
                self.bump();
                continue;
            }
            let start = self.span_start();
            let kind = match ch {
                '\n' => {
                    self.bump();
                    if self.nesting == 0 {
                        self.at_line_start = true;
                        TokenKind::Newline
                    } else {
                        continue;
                    }
                }
                '#' => {
                    while !matches!(self.peek(), None | Some('\n')) {
                        self.bump();
                    }
                    continue;
                }
                'f' | 'F' if self.followed_by_quote() => self.fstring(start, 1, false)?,
                'f' | 'F' if self.followed_by_raw_quote() => self.fstring(start, 2, true)?,
                'r' | 'R' if self.followed_by_format_quote() => self.fstring(start, 2, true)?,
                'r' | 'R' if self.followed_by_byte_quote() => self.raw_byte_string(start)?,
                'r' | 'R' if self.followed_by_quote() => self.raw_string(start)?,
                'b' | 'B' if self.followed_by_raw_quote() => self.raw_byte_string(start)?,
                'b' | 'B' if self.followed_by_quote() => {
                    self.bump();
                    self.byte_string(start)?
                }
                'a'..='z' | 'A'..='Z' | '_' => self.name(),
                '0'..='9' => self.number(start, false)?,
                '\'' | '"' => self.string(start)?,
                '+' => self.either('=', TokenKind::PlusEqual, TokenKind::Plus),
                '-' => {
                    self.bump();
                    if self.peek() == Some('>') {
                        self.bump();
                        TokenKind::Arrow
                    } else if self.peek() == Some('=') {
                        self.bump();
                        TokenKind::MinusEqual
                    } else {
                        TokenKind::Minus
                    }
                }
                '*' => {
                    self.bump();
                    if self.peek() == Some('*') {
                        self.bump();
                        if self.peek() == Some('=') {
                            self.bump();
                            TokenKind::DoubleStarEqual
                        } else {
                            TokenKind::DoubleStar
                        }
                    } else if self.peek() == Some('=') {
                        self.bump();
                        TokenKind::StarEqual
                    } else {
                        TokenKind::Star
                    }
                }
                '%' => self.either('=', TokenKind::PercentEqual, TokenKind::Percent),
                '&' => self.either('=', TokenKind::AmpersandEqual, TokenKind::Ampersand),
                '^' => self.either('=', TokenKind::CaretEqual, TokenKind::Caret),
                '~' => self.single(TokenKind::Tilde),
                '|' => self.either('=', TokenKind::PipeEqual, TokenKind::Pipe),
                '\\' if self.source[self.offset..].starts_with("\\\n") => {
                    self.bump();
                    self.bump();
                    continue;
                }
                '=' => self.either('=', TokenKind::EqualEqual, TokenKind::Equal),
                '!' => {
                    self.bump();
                    if self.peek() == Some('=') {
                        self.bump();
                        TokenKind::NotEqual
                    } else {
                        return Err(self.error(start, "expected '=' after '!'"));
                    }
                }
                '<' => {
                    self.bump();
                    if self.peek() == Some('<') {
                        self.bump();
                        if self.peek() == Some('=') {
                            self.bump();
                            TokenKind::LeftShiftEqual
                        } else {
                            TokenKind::LeftShift
                        }
                    } else if self.peek() == Some('=') {
                        self.bump();
                        TokenKind::LessEqual
                    } else {
                        TokenKind::Less
                    }
                }
                '>' => {
                    self.bump();
                    if self.peek() == Some('>') {
                        self.bump();
                        if self.peek() == Some('=') {
                            self.bump();
                            TokenKind::RightShiftEqual
                        } else {
                            TokenKind::RightShift
                        }
                    } else if self.peek() == Some('=') {
                        self.bump();
                        TokenKind::GreaterEqual
                    } else {
                        TokenKind::Greater
                    }
                }
                '.' if matches!(self.source[self.offset..].chars().nth(1), Some('0'..='9')) => {
                    self.number(start, true)?
                }
                '.' => self.single(TokenKind::Dot),
                '@' => self.single(TokenKind::At),
                ',' => self.single(TokenKind::Comma),
                ':' => {
                    self.bump();
                    if self.peek() == Some('=') {
                        self.bump();
                        TokenKind::ColonEqual
                    } else {
                        TokenKind::Colon
                    }
                }
                '(' => self.open(TokenKind::LeftParen),
                ')' => self.close(start, TokenKind::RightParen)?,
                '[' => self.open(TokenKind::LeftBracket),
                ']' => self.close(start, TokenKind::RightBracket)?,
                '{' => self.open(TokenKind::LeftBrace),
                '}' => self.close(start, TokenKind::RightBrace)?,
                ';' => self.single(TokenKind::Semicolon),
                '/' => {
                    self.bump();
                    if self.peek() == Some('/') {
                        self.bump();
                        if self.peek() == Some('=') {
                            self.bump();
                            TokenKind::DoubleSlashEqual
                        } else {
                            TokenKind::DoubleSlash
                        }
                    } else if self.peek() == Some('=') {
                        self.bump();
                        TokenKind::SlashEqual
                    } else {
                        TokenKind::Slash
                    }
                }
                _ => return Err(self.error(start, format!("unexpected character {ch:?}"))),
            };
            tokens.push(Token {
                kind,
                span: start.through(self.span_end()),
            });
        }
        if self.nesting != 0 {
            return Err(self.error(self.span_start(), "unclosed delimiter"));
        }
        let span = self.span_start();
        if !tokens.is_empty()
            && !tokens
                .last()
                .is_some_and(|token| matches!(token.kind, TokenKind::Newline))
        {
            tokens.push(Token {
                kind: TokenKind::Newline,
                span,
            });
        }
        while self.indents.len() > 1 {
            self.indents.pop();
            tokens.push(Token {
                kind: TokenKind::Dedent,
                span,
            });
        }
        tokens.push(Token {
            kind: TokenKind::Eof,
            span,
        });
        Ok(tokens)
    }

    fn name(&mut self) -> TokenKind {
        let start = self.offset;
        while matches!(self.peek(), Some('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) {
            self.bump();
        }
        match &self.source[start..self.offset] {
            "import" => TokenKind::Import,
            "from" => TokenKind::From,
            "as" => TokenKind::As,
            "del" => TokenKind::Del,
            "None" => TokenKind::None,
            "True" => TokenKind::True,
            "False" => TokenKind::False,
            "and" => TokenKind::And,
            "or" => TokenKind::Or,
            "not" => TokenKind::Not,
            "in" => TokenKind::In,
            "is" => TokenKind::Is,
            "if" => TokenKind::If,
            "elif" => TokenKind::Elif,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "for" => TokenKind::For,
            "def" => TokenKind::Def,
            "class" => TokenKind::Class,
            "lambda" => TokenKind::Lambda,
            "return" => TokenKind::Return,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "global" => TokenKind::Global,
            "nonlocal" => TokenKind::Nonlocal,
            "pass" => TokenKind::Pass,
            "assert" => TokenKind::Assert,
            "try" => TokenKind::Try,
            "except" => TokenKind::Except,
            "finally" => TokenKind::Finally,
            "raise" => TokenKind::Raise,
            "with" => TokenKind::With,
            "yield" => TokenKind::Yield,
            "async" => TokenKind::Async,
            "await" => TokenKind::Await,
            name => TokenKind::Name((*name).to_string()),
        }
    }

    /// Lex a decimal integer or floating-point literal.  The grammar is kept
    /// intentionally narrow: decimal digits, an optional decimal point, and
    /// an optional e/E exponent.  Underscores are accepted only between
    /// digits, matching Python's decimal-literal rules.
    fn number(&mut self, start: Span, leading_dot: bool) -> Result<TokenKind, LexError> {
        let offset = self.offset;
        let mut is_float = leading_dot;

        if !leading_dot && self.peek() == Some('0') {
            if let Some(prefix @ ('x' | 'X' | 'o' | 'O' | 'b' | 'B')) =
                self.source[self.offset..].chars().nth(1)
            {
                return self.radix_integer(start, prefix);
            }
        }

        if leading_dot {
            self.bump(); // '.'
            if !self.digit_run(start)? {
                return Err(self.error(start, "invalid decimal literal: expected digits after '.'"));
            }
        } else {
            self.digit_run(start)?;
            if self.peek() == Some('.') {
                is_float = true;
                self.bump();
                // A fractional part is optional, so 1. is valid.  An
                // underscore here is not: 1._2 is not a Python literal.
                if self.peek() == Some('_') {
                    return Err(self.error(start, "invalid decimal literal: misplaced underscore"));
                }
                if matches!(self.peek(), Some('0'..='9')) {
                    self.digit_run(start)?;
                }
            }
        }

        if matches!(self.peek(), Some('e' | 'E')) {
            is_float = true;
            self.bump();
            if matches!(self.peek(), Some('+' | '-')) {
                self.bump();
            }
            if !self.digit_run(start)? {
                return Err(self.error(start, "invalid decimal literal: exponent requires digits"));
            }
        }

        let spelling = self.source[offset..self.offset].replace('_', "");
        if matches!(self.peek(), Some('j' | 'J')) {
            self.bump();
            return spelling
                .parse::<f64>()
                .map(TokenKind::Imaginary)
                .map_err(|_| self.error(start, "invalid imaginary literal"));
        }
        if is_float {
            spelling
                .parse::<f64>()
                .map(TokenKind::Float)
                .map_err(|_| self.error(start, "invalid decimal floating-point literal"))
        } else {
            Ok(spelling
                .parse::<i64>()
                .map(TokenKind::Integer)
                .unwrap_or_else(|_| TokenKind::BigInteger(spelling)))
        }
    }

    fn radix_integer(&mut self, start: Span, prefix: char) -> Result<TokenKind, LexError> {
        let radix = match prefix.to_ascii_lowercase() {
            'x' => 16,
            'o' => 8,
            'b' => 2,
            _ => unreachable!("validated radix prefix"),
        };
        self.bump();
        self.bump();
        let digits_start = self.offset;
        while self
            .peek()
            .is_some_and(|character| character == '_' || character.is_digit(radix))
        {
            self.bump();
        }
        let spelling = &self.source[digits_start..self.offset];
        if spelling.is_empty()
            || spelling == "_"
            || spelling.ends_with('_')
            || spelling.contains("__")
        {
            return Err(self.error(start, "invalid prefixed integer literal"));
        }
        let digits = spelling
            .strip_prefix('_')
            .unwrap_or(spelling)
            .replace('_', "");
        let value = BigInt::parse_bytes(digits.as_bytes(), radix)
            .ok_or_else(|| self.error(start, "invalid prefixed integer literal"))?;
        Ok(value
            .to_i64()
            .map(TokenKind::Integer)
            .unwrap_or_else(|| TokenKind::BigInteger(value.to_string())))
    }

    /// Consume one run of decimal digits and separators.  Return whether at
    /// least one digit was consumed; callers use that for optional fractional
    /// parts and required exponents.
    fn digit_run(&mut self, start: Span) -> Result<bool, LexError> {
        let mut digits = false;
        let mut underscore = false;
        while matches!(self.peek(), Some('0'..='9' | '_')) {
            if self.peek() == Some('_') {
                if !digits || underscore {
                    return Err(self.error(start, "invalid decimal literal: misplaced underscore"));
                }
                underscore = true;
            } else {
                digits = true;
                underscore = false;
            }
            self.bump();
        }
        if underscore {
            return Err(self.error(start, "invalid decimal literal: trailing underscore"));
        }
        Ok(digits)
    }

    fn string(&mut self, start: Span) -> Result<TokenKind, LexError> {
        let quote = self.bump().expect("the caller observed a quote");
        let triple =
            self.peek() == Some(quote) && self.source[self.offset..].chars().nth(1) == Some(quote);
        if triple {
            self.bump();
            self.bump();
        }
        let mut value = String::new();
        loop {
            let Some(ch) = self.bump() else {
                return Err(self.error(start, "unterminated string literal"));
            };
            match ch {
                ch if ch == quote && self.consume_triple_end(quote, triple) => break,
                '\n' if !triple => return Err(self.error(start, "unterminated string literal")),
                '\\' => {
                    let Some(escaped) = self.bump() else {
                        return Err(self.error(start, "unterminated escape sequence"));
                    };
                    if escaped == '\n' {
                        continue;
                    }
                    let decoded = match escaped {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        'b' => '\u{0008}',
                        'f' => '\u{000c}',
                        'v' => '\u{000b}',
                        '\\' => '\\',
                        '\'' => '\'',
                        '"' => '"',
                        '0' => '\0',
                        other => {
                            value.push('\\');
                            other
                        }
                    };
                    value.push(decoded);
                }
                other => value.push(other),
            }
        }
        Ok(TokenKind::String(value))
    }

    fn raw_string(&mut self, start: Span) -> Result<TokenKind, LexError> {
        self.bump(); // r/R prefix
        let quote = self.bump().expect("followed_by_quote checked above");
        let triple =
            self.peek() == Some(quote) && self.source[self.offset..].chars().nth(1) == Some(quote);
        if triple {
            self.bump();
            self.bump();
        }
        let mut value = String::new();
        loop {
            let Some(ch) = self.bump() else {
                return Err(self.error(start, "unterminated raw string literal"));
            };
            if ch == '\\' {
                value.push(ch);
                let escaped = self
                    .bump()
                    .ok_or_else(|| self.error(start, "unterminated raw string literal"))?;
                value.push(escaped);
                continue;
            }
            if ch == quote && self.consume_triple_end(quote, triple) {
                break;
            }
            if ch == '\n' && !triple {
                return Err(self.error(start, "unterminated raw string literal"));
            }
            value.push(ch);
        }
        Ok(TokenKind::String(value))
    }

    /// Decode a bytes literal without passing arbitrary octets through UTF-8 text storage.
    fn byte_string(&mut self, start: Span) -> Result<TokenKind, LexError> {
        let quote = self.bump().expect("the caller observed a quote");
        let triple =
            self.peek() == Some(quote) && self.source[self.offset..].chars().nth(1) == Some(quote);
        if triple {
            self.bump();
            self.bump();
        }
        let mut value = Vec::new();
        loop {
            let Some(ch) = self.bump() else {
                return Err(self.error(start, "unterminated bytes literal"));
            };
            match ch {
                ch if ch == quote && self.consume_triple_end(quote, triple) => break,
                '\n' if !triple => return Err(self.error(start, "unterminated bytes literal")),
                '\\' => {
                    let escaped = self
                        .bump()
                        .ok_or_else(|| self.error(start, "unterminated bytes escape"))?;
                    if escaped == '\n' {
                        continue;
                    }
                    let byte = match escaped {
                        'n' => b'\n',
                        'r' => b'\r',
                        't' => b'\t',
                        'b' => 8,
                        'f' => 12,
                        'v' => 11,
                        '\\' => b'\\',
                        '\'' => b'\'',
                        '"' => b'"',
                        'x' => {
                            let high = self.bump().and_then(|ch| ch.to_digit(16));
                            let low = self.bump().and_then(|ch| ch.to_digit(16));
                            let (Some(high), Some(low)) = (high, low) else {
                                return Err(self.error(start, "invalid hexadecimal bytes escape"));
                            };
                            u8::try_from(high * 16 + low).expect("two hex digits fit in a byte")
                        }
                        '0'..='7' => {
                            let mut number = escaped.to_digit(8).expect("matched octal digit");
                            for _ in 0..2 {
                                let Some(digit) = self.peek().and_then(|ch| ch.to_digit(8)) else {
                                    break;
                                };
                                self.bump();
                                number = number * 8 + digit;
                            }
                            u8::try_from(number).map_err(|_| {
                                self.error(start, "octal bytes escape is out of range")
                            })?
                        }
                        other if other.is_ascii() => {
                            value.push(b'\\');
                            other as u8
                        }
                        _ => return Err(self.error(start, "bytes literals must contain ASCII")),
                    };
                    value.push(byte);
                }
                other if other.is_ascii() => value.push(other as u8),
                _ => return Err(self.error(start, "bytes literals must contain ASCII")),
            }
        }
        Ok(TokenKind::Bytes(value))
    }

    fn raw_byte_string(&mut self, start: Span) -> Result<TokenKind, LexError> {
        self.bump();
        self.bump();
        let quote = self.bump().expect("prefix is followed by a quote");
        let triple =
            self.peek() == Some(quote) && self.source[self.offset..].chars().nth(1) == Some(quote);
        if triple {
            self.bump();
            self.bump();
        }
        let mut value = Vec::new();
        loop {
            let Some(ch) = self.bump() else {
                return Err(self.error(start, "unterminated raw bytes literal"));
            };
            if ch == '\\' {
                let escaped = self
                    .bump()
                    .ok_or_else(|| self.error(start, "unterminated raw bytes literal"))?;
                if !escaped.is_ascii() {
                    return Err(self.error(start, "bytes literals must contain ASCII"));
                }
                value.push(b'\\');
                value.push(escaped as u8);
                continue;
            }
            if ch == quote && self.consume_triple_end(quote, triple) {
                break;
            }
            if ch == '\n' && !triple {
                return Err(self.error(start, "unterminated raw bytes literal"));
            }
            if !ch.is_ascii() {
                return Err(self.error(start, "bytes literals must contain ASCII"));
            }
            value.push(ch as u8);
        }
        Ok(TokenKind::Bytes(value))
    }

    /// Capture an f-string body without interpreting braces.  The parser owns
    /// brace matching and embedded-expression parsing; keeping the raw body in
    /// one token prevents the ordinary lexer from confusing expression tokens
    /// with the surrounding literal text.
    fn fstring(
        &mut self,
        start: Span,
        prefix_length: usize,
        raw: bool,
    ) -> Result<TokenKind, LexError> {
        for _ in 0..prefix_length {
            self.bump();
        }
        let quote = self.bump().expect("followed_by_quote checked above");
        let triple =
            self.peek() == Some(quote) && self.source[self.offset..].chars().nth(1) == Some(quote);
        if triple {
            self.bump();
            self.bump();
        }
        let body_start = self.offset;
        while let Some(ch) = self.bump() {
            if ch == quote && self.consume_triple_end(quote, triple) {
                let delimiter_length = if triple { 3 } else { 1 };
                return Ok(TokenKind::FString {
                    body: self.source[body_start..self.offset - delimiter_length].into(),
                    raw,
                });
            }
            if ch == '\n' && !triple {
                return Err(self.error(start, "unterminated f-string literal"));
            }
            if ch == '\\' && !raw {
                self.bump()
                    .ok_or_else(|| self.error(start, "unterminated escape sequence"))?;
            }
        }
        Err(self.error(start, "unterminated f-string literal"))
    }

    fn followed_by_quote(&self) -> bool {
        matches!(self.source[self.offset..].chars().nth(1), Some('\'' | '"'))
    }

    fn followed_by_raw_quote(&self) -> bool {
        matches!(
            (
                self.source[self.offset..].chars().nth(1),
                self.source[self.offset..].chars().nth(2)
            ),
            (Some('r' | 'R'), Some('\'' | '"'))
        )
    }

    fn followed_by_format_quote(&self) -> bool {
        matches!(
            (
                self.source[self.offset..].chars().nth(1),
                self.source[self.offset..].chars().nth(2)
            ),
            (Some('f' | 'F'), Some('\'' | '"'))
        )
    }

    fn followed_by_byte_quote(&self) -> bool {
        matches!(
            (
                self.source[self.offset..].chars().nth(1),
                self.source[self.offset..].chars().nth(2)
            ),
            (Some('b' | 'B'), Some('\'' | '"'))
        )
    }

    fn consume_triple_end(&mut self, quote: char, triple: bool) -> bool {
        if !triple {
            return true;
        }
        if self.peek() == Some(quote) && self.source[self.offset..].chars().nth(1) == Some(quote) {
            self.bump();
            self.bump();
            true
        } else {
            false
        }
    }

    fn single(&mut self, kind: TokenKind) -> TokenKind {
        self.bump();
        kind
    }

    fn open(&mut self, kind: TokenKind) -> TokenKind {
        self.nesting += 1;
        self.single(kind)
    }

    fn close(&mut self, start: Span, kind: TokenKind) -> Result<TokenKind, LexError> {
        if self.nesting == 0 {
            return Err(self.error(start, "unmatched closing delimiter"));
        }
        self.nesting -= 1;
        Ok(self.single(kind))
    }

    fn indentation(&mut self, tokens: &mut Vec<Token>) -> Result<(), LexError> {
        let start = self.span_start();
        let mut width = 0usize;
        loop {
            match self.peek() {
                Some(' ') => {
                    self.bump();
                    width += 1;
                }
                Some('\t') => {
                    self.bump();
                    width = (width / 8 + 1) * 8;
                }
                Some('\u{000c}') => {
                    self.bump();
                    width = 0;
                }
                _ => break,
            }
        }
        // A CRLF blank line must not change indentation. The main token loop already ignores
        // `\r`; consuming it here keeps the indentation stack untouched before the `\n` token.
        if self.source[self.offset..].starts_with("\r\n") {
            self.bump();
        }
        if matches!(self.peek(), None | Some('\n' | '#')) {
            return Ok(());
        }
        self.at_line_start = false;
        let current = *self.indents.last().expect("indent stack is never empty");
        if width > current {
            self.indents.push(width);
            tokens.push(Token {
                kind: TokenKind::Indent,
                span: start.through(self.span_end()),
            });
        } else if width < current {
            while width < *self.indents.last().expect("indent stack is never empty") {
                self.indents.pop();
                tokens.push(Token {
                    kind: TokenKind::Dedent,
                    span: start.through(self.span_end()),
                });
            }
            if width != *self.indents.last().expect("indent stack is never empty") {
                return Err(self.error(start, "unindent does not match an outer indentation level"));
            }
        }
        Ok(())
    }

    fn either(&mut self, next: char, two: TokenKind, one: TokenKind) -> TokenKind {
        self.bump();
        if self.peek() == Some(next) {
            self.bump();
            two
        } else {
            one
        }
    }

    fn peek(&self) -> Option<char> {
        self.source[self.offset..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.offset += ch.len_utf8();
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(ch)
    }

    fn span_start(&self) -> Span {
        Span::new(self.offset, self.offset, self.line, self.column)
    }

    fn span_end(&self) -> Span {
        Span::new(self.offset, self.offset, self.line, self.column)
    }

    fn error(&self, span: Span, message: impl Into<String>) -> LexError {
        LexError {
            message: message.into(),
            span: span.through(self.span_end()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexes_unicode_strings_and_python_escapes() {
        let tokens = lex("name = 'café\\n' # ignored\n").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Name("name".into()));
        assert_eq!(tokens[2].kind, TokenKind::String("café\n".into()));
        assert_eq!(tokens[3].kind, TokenKind::Newline);
    }

    #[test]
    fn preserves_out_of_range_integer_spelling_for_bigint_parsing() {
        let tokens = lex("999999999999999999999999999999").unwrap();
        assert_eq!(
            tokens[0].kind,
            TokenKind::BigInteger("999999999999999999999999999999".into())
        );
    }

    #[test]
    fn imaginary_literals_require_a_valid_decimal_component() {
        let tokens = lex("1j .5J 1e2j").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Imaginary(1.0));
        assert_eq!(tokens[1].kind, TokenKind::Imaginary(0.5));
        assert_eq!(tokens[2].kind, TokenKind::Imaginary(100.0));
        assert!(lex("1_j").is_err());
    }

    #[test]
    fn prefixed_integer_literals_are_normalized_to_decimal_values() {
        let tokens = lex("0xff 0o20 0b1_010 0x8000000000000000").unwrap();
        assert!(matches!(tokens[0].kind, TokenKind::Integer(255)));
        assert!(matches!(tokens[1].kind, TokenKind::Integer(16)));
        assert!(matches!(tokens[2].kind, TokenKind::Integer(10)));
        assert!(matches!(
            &tokens[3].kind,
            TokenKind::BigInteger(value) if value == "9223372036854775808"
        ));
        assert!(lex("0x_ ").is_err());
        assert!(lex("0b2").is_err());
    }

    #[test]
    fn emits_python_indentation_boundaries() {
        let tokens = lex("if True:\n    print(1)\nprint(2)\n").unwrap();
        let kinds = tokens
            .into_iter()
            .map(|token| token.kind)
            .collect::<Vec<_>>();
        assert!(matches!(kinds[4], TokenKind::Indent));
        assert!(kinds.iter().any(|kind| matches!(kind, TokenKind::Dedent)));
    }
}
