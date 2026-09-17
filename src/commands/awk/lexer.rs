//! Bounded tokenizer for awk source.

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Token {
    Ident(String),
    Number(f64),
    String(String),
    Regex(String),
    Begin,
    End,
    If,
    Else,
    While,
    For,
    In,
    Break,
    Continue,
    Delete,
    Next,
    NextFile,
    Exit,
    Print,
    Printf,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
    Dollar,
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    PlusPlus,
    MinusMinus,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Match,
    NotMatch,
    And,
    Or,
    Not,
    Eof,
}

/// Tokenize a complete awk program before any simulated input is consumed.
pub(super) fn lex(source: &str) -> Result<Vec<Token>, String> {
    let mut lexer = Lexer {
        source,
        cursor: 0,
        tokens: Vec::new(),
        expects_operand: true,
    };
    while lexer.cursor < source.len() {
        lexer.token()?;
    }
    lexer.tokens.push(Token::Eof);
    Ok(lexer.tokens)
}

struct Lexer<'a> {
    source: &'a str,
    cursor: usize,
    tokens: Vec<Token>,
    expects_operand: bool,
}

impl Lexer<'_> {
    fn token(&mut self) -> Result<(), String> {
        let ch = self.peek().expect("cursor is in bounds");
        match ch {
            ' ' | '\t' | '\r' => {
                self.bump();
            }
            '\n' | ';' => {
                self.bump();
                self.push(Token::Semicolon, true);
            }
            '#' => {
                while self.peek().is_some_and(|value| value != '\n') {
                    self.bump();
                }
            }
            '"' => {
                let value = self.quoted_string()?;
                self.push(Token::String(value), false);
            }
            '0'..='9' | '.'
                if ch != '.' || self.peek_second().is_some_and(|c| c.is_ascii_digit()) =>
            {
                let value = self.number()?;
                self.push(Token::Number(value), false);
            }
            value if value == '_' || value.is_ascii_alphabetic() => {
                let ident = self.ident();
                if matches!(ident.as_str(), "function" | "getline") {
                    return Err(format!("unsupported keyword '{ident}'"));
                }
                let token = match ident.as_str() {
                    "BEGIN" => Token::Begin,
                    "END" => Token::End,
                    "if" => Token::If,
                    "else" => Token::Else,
                    "while" => Token::While,
                    "for" => Token::For,
                    "in" => Token::In,
                    "break" => Token::Break,
                    "continue" => Token::Continue,
                    "delete" => Token::Delete,
                    "next" => Token::Next,
                    "nextfile" => Token::NextFile,
                    "exit" => Token::Exit,
                    "print" => Token::Print,
                    "printf" => Token::Printf,
                    _ => Token::Ident(ident),
                };
                let expects_operand = matches!(
                    token,
                    Token::If
                        | Token::While
                        | Token::For
                        | Token::Delete
                        | Token::Print
                        | Token::Printf
                        | Token::Exit
                );
                self.push(token, expects_operand);
            }
            '/' if self.expects_operand => {
                let regex = self.regex()?;
                self.push(Token::Regex(regex), false);
            }
            '(' => self.single(Token::LParen, true),
            ')' => self.single(Token::RParen, false),
            '{' => self.single(Token::LBrace, true),
            '}' => self.single(Token::RBrace, false),
            '[' => self.single(Token::LBracket, true),
            ']' => self.single(Token::RBracket, false),
            ',' => self.single(Token::Comma, true),
            '$' => self.single(Token::Dollar, true),
            '+' => self.operator('+', Token::Plus, Token::PlusPlus, Token::AddAssign)?,
            '-' => self.operator('-', Token::Minus, Token::MinusMinus, Token::SubAssign)?,
            '*' => self.assignment_operator(Token::Star, Token::MulAssign),
            '%' => self.assignment_operator(Token::Percent, Token::RemAssign),
            '/' => self.assignment_operator(Token::Slash, Token::DivAssign),
            '=' => self.two_char('=', Token::Assign, Token::Equal, true),
            '!' => {
                if self.peek_second() == Some('=') {
                    self.bump();
                    self.bump();
                    self.push(Token::NotEqual, true);
                } else if self.peek_second() == Some('~') {
                    self.bump();
                    self.bump();
                    self.push(Token::NotMatch, true);
                } else {
                    self.single(Token::Not, true);
                }
            }
            '<' => self.two_char('=', Token::Less, Token::LessEqual, true),
            '>' => self.two_char('=', Token::Greater, Token::GreaterEqual, true),
            '~' => self.single(Token::Match, true),
            '&' if self.peek_second() == Some('&') => {
                self.bump();
                self.bump();
                self.push(Token::And, true);
            }
            '|' if self.peek_second() == Some('|') => {
                self.bump();
                self.bump();
                self.push(Token::Or, true);
            }
            _ => {
                return Err(format!(
                    "unexpected character '{ch}' at byte {}",
                    self.cursor
                ))
            }
        }
        Ok(())
    }

    fn quoted_string(&mut self) -> Result<String, String> {
        self.bump();
        let mut output = String::new();
        while let Some(ch) = self.bump() {
            match ch {
                '"' => return Ok(output),
                '\\' => {
                    let escaped = self
                        .bump()
                        .ok_or_else(|| "unterminated string escape".to_string())?;
                    output.push(match escaped {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        'b' => '\u{0008}',
                        'f' => '\u{000c}',
                        other => other,
                    });
                }
                '\n' => return Err("newline in string literal".to_string()),
                other => output.push(other),
            }
        }
        Err("unterminated string literal".to_string())
    }

    fn regex(&mut self) -> Result<String, String> {
        self.bump();
        let mut output = String::new();
        let mut escaped = false;
        while let Some(ch) = self.bump() {
            if escaped {
                if ch == '/' {
                    output.push('/');
                } else {
                    output.push('\\');
                    output.push(ch);
                }
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '/' {
                return Ok(output);
            } else if ch == '\n' {
                return Err("newline in regular expression".to_string());
            } else {
                output.push(ch);
            }
        }
        Err("unterminated regular expression".to_string())
    }

    fn number(&mut self) -> Result<f64, String> {
        let start = self.cursor;
        while self
            .peek()
            .is_some_and(|ch| ch.is_ascii_digit() || ch == '.')
        {
            self.bump();
        }
        if self.peek().is_some_and(|ch| matches!(ch, 'e' | 'E')) {
            self.bump();
            if self.peek().is_some_and(|ch| matches!(ch, '+' | '-')) {
                self.bump();
            }
            while self.peek().is_some_and(|ch| ch.is_ascii_digit()) {
                self.bump();
            }
        }
        self.source[start..self.cursor]
            .parse()
            .map_err(|_| format!("invalid number '{}';", &self.source[start..self.cursor]))
    }

    fn ident(&mut self) -> String {
        let start = self.cursor;
        while self
            .peek()
            .is_some_and(|ch| ch == '_' || ch.is_ascii_alphanumeric())
        {
            self.bump();
        }
        self.source[start..self.cursor].to_string()
    }

    fn operator(
        &mut self,
        character: char,
        plain: Token,
        repeated: Token,
        assigned: Token,
    ) -> Result<(), String> {
        debug_assert_eq!(self.peek(), Some(character));
        self.bump();
        if self.peek() == Some(character) {
            self.bump();
            self.push(repeated, false);
        } else if self.peek() == Some('=') {
            self.bump();
            self.push(assigned, true);
        } else {
            self.push(plain, true);
        }
        Ok(())
    }

    fn assignment_operator(&mut self, plain: Token, assigned: Token) {
        self.bump();
        if self.peek() == Some('=') {
            self.bump();
            self.push(assigned, true);
        } else {
            self.push(plain, true);
        }
    }

    fn two_char(&mut self, second: char, plain: Token, combined: Token, expects_operand: bool) {
        self.bump();
        if self.peek() == Some(second) {
            self.bump();
            self.push(combined, expects_operand);
        } else {
            self.push(plain, expects_operand);
        }
    }

    fn single(&mut self, token: Token, expects_operand: bool) {
        self.bump();
        self.push(token, expects_operand);
    }

    fn push(&mut self, token: Token, expects_operand: bool) {
        self.tokens.push(token);
        self.expects_operand = expects_operand;
    }

    fn peek(&self) -> Option<char> {
        self.source[self.cursor..].chars().next()
    }

    fn peek_second(&self) -> Option<char> {
        self.source[self.cursor..].chars().nth(1)
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.cursor += ch.len_utf8();
        Some(ch)
    }
}

#[cfg(test)]
mod tests {
    use super::{lex, Token};

    #[test]
    fn distinguishes_regex_from_division_and_preserves_blocks() {
        let tokens = lex(r#"$1 / 2 > 1 { if ($0 ~ /a\//) print "yes" }"#).unwrap();
        assert!(tokens.contains(&Token::Slash));
        assert!(tokens.contains(&Token::Regex("a/".to_string())));
        assert!(tokens.contains(&Token::If));
    }

    #[test]
    fn rejects_unterminated_literals() {
        assert!(lex("BEGIN { print \"x }")
            .unwrap_err()
            .contains("unterminated"));
        assert!(lex("/abc").unwrap_err().contains("unterminated"));
    }
}
