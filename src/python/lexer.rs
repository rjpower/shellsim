//! A small UTF-8-aware lexer. It owns Python escape handling rather than reusing shell rules.

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
                'f' | 'F' if self.followed_by_quote() => self.fstring(start)?,
                'r' | 'R' if self.followed_by_quote() => self.raw_string(start)?,
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
                '*' => self.single(TokenKind::Star),
                '%' => self.single(TokenKind::Percent),
                '|' => self.single(TokenKind::Pipe),
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
                '<' => self.either('=', TokenKind::LessEqual, TokenKind::Less),
                '>' => self.either('=', TokenKind::GreaterEqual, TokenKind::Greater),
                '.' if matches!(self.source[self.offset..].chars().nth(1), Some('0'..='9')) => {
                    self.number(start, true)?
                }
                '.' => self.single(TokenKind::Dot),
                '@' => self.single(TokenKind::At),
                ',' => self.single(TokenKind::Comma),
                ':' => self.single(TokenKind::Colon),
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
                        TokenKind::DoubleSlash
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
            "nonlocal" => TokenKind::Nonlocal,
            "pass" => TokenKind::Pass,
            "assert" => TokenKind::Assert,
            "try" => TokenKind::Try,
            "except" => TokenKind::Except,
            "finally" => TokenKind::Finally,
            "raise" => TokenKind::Raise,
            "with" => TokenKind::With,
            "yield" => TokenKind::Yield,
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
        if is_float {
            spelling
                .parse::<f64>()
                .map(TokenKind::Float)
                .map_err(|_| self.error(start, "invalid decimal floating-point literal"))
        } else {
            spelling
                .parse::<i64>()
                .map(TokenKind::Integer)
                .map_err(|_| self.error(start, "integer is outside this implementation's range"))
        }
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
        let mut value = String::new();
        loop {
            let Some(ch) = self.bump() else {
                return Err(self.error(start, "unterminated string literal"));
            };
            match ch {
                ch if ch == quote => break,
                '\n' => return Err(self.error(start, "unterminated string literal")),
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
        let mut value = String::new();
        loop {
            let Some(ch) = self.bump() else {
                return Err(self.error(start, "unterminated raw string literal"));
            };
            if ch == quote {
                break;
            }
            if ch == '\n' {
                return Err(self.error(start, "unterminated raw string literal"));
            }
            value.push(ch);
        }
        Ok(TokenKind::String(value))
    }

    /// Capture an f-string body without interpreting braces.  The parser owns
    /// brace matching and embedded-expression parsing; keeping the raw body in
    /// one token prevents the ordinary lexer from confusing expression tokens
    /// with the surrounding literal text.
    fn fstring(&mut self, start: Span) -> Result<TokenKind, LexError> {
        self.bump(); // f/F prefix
        let quote = self.bump().expect("followed_by_quote checked above");
        let body_start = self.offset;
        while let Some(ch) = self.bump() {
            if ch == quote {
                return Ok(TokenKind::FString(
                    self.source[body_start..self.offset - 1].into(),
                ));
            }
            if ch == '\n' {
                return Err(self.error(start, "unterminated f-string literal"));
            }
            if ch == '\\' {
                self.bump()
                    .ok_or_else(|| self.error(start, "unterminated escape sequence"))?;
            }
        }
        Err(self.error(start, "unterminated f-string literal"))
    }

    fn followed_by_quote(&self) -> bool {
        matches!(self.source[self.offset..].chars().nth(1), Some('\'' | '"'))
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
    fn rejects_out_of_range_integer_instead_of_wrapping() {
        let error = lex("999999999999999999999999999999").unwrap_err();
        assert!(error.message.contains("outside"));
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
