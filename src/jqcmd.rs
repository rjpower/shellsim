//! Typed, metered `jq` subset over `serde_json::Value`.
//!
//! The parser covers ordinary selection, construction, comparison, conditional, sorting, and
//! conversion filters used by the TaskTrove sample. Evaluation deliberately materializes values
//! and uses simple loops. Unsupported syntax fails before evaluation and never reaches host jq.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::commands::options::{parse_options_or_report, OptionSpec};
use crate::interp::Interp;
use serde::Serialize;
use serde_json::{Map, Value};

type Out<'a> = &'a mut Vec<u8>;
const MAX_FILTER_TOKENS: usize = 1_024;
const MAX_EXPRESSION_DEPTH: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Key {
    Raw,
    Compact,
    NullInput,
    Slurp,
    ExitStatus,
    Indent,
    Help,
}

const OPTIONS: &[OptionSpec<Key>] = &[
    OptionSpec::flag(Key::Raw, Some('r'), Some("raw-output")),
    OptionSpec::flag(Key::Compact, Some('c'), Some("compact-output")),
    OptionSpec::flag(Key::NullInput, Some('n'), Some("null-input")),
    OptionSpec::flag(Key::Slurp, Some('s'), Some("slurp")),
    OptionSpec::flag(Key::ExitStatus, Some('e'), Some("exit-status")),
    OptionSpec::required(Key::Indent, None, Some("indent")),
    OptionSpec::flag(Key::Help, None, Some("help")),
];

/// Run the bounded jq interpreter over stdin or VFS files.
pub fn jq(interp: &mut Interp, args: &[String], stdin: Vec<u8>, out: Out, err: Out) -> i32 {
    let memory_mark = interp.resources.memory_mark();
    let status = jq_inner(interp, args, stdin, out, err);
    interp.resources.restore_memory(memory_mark);
    status
}

fn jq_inner(interp: &mut Interp, args: &[String], stdin: Vec<u8>, out: Out, err: Out) -> i32 {
    let (args, variables) = match extract_variables(args) {
        Ok(parsed) => parsed,
        Err(error) => {
            ewln(err, &format!("jq: {error}"));
            return 2;
        }
    };
    let parsed = match parse_options_or_report(
        "jq",
        &args,
        OPTIONS,
        (
            Key::Help,
            "usage: jq [-rcnse] [--arg NAME VALUE] [--argjson NAME JSON] FILTER [FILE...]\n",
        ),
        out,
        err,
    ) {
        Ok(parsed) => parsed,
        Err(status) => return status,
    };
    let mut raw = false;
    let mut compact = false;
    let mut null_input = false;
    let mut slurp = false;
    let mut exit_status = false;
    let mut indent = 2;
    for option in parsed.options {
        match option.key {
            Key::Raw => raw = true,
            Key::Compact => compact = true,
            Key::NullInput => null_input = true,
            Key::Slurp => slurp = true,
            Key::ExitStatus => exit_status = true,
            Key::Indent => {
                let value = option.value.expect("required option value");
                indent = match value.parse::<usize>() {
                    Ok(value @ 0..=7) => value,
                    _ => {
                        ewln(err, "jq: --indent must be between 0 and 7");
                        return 2;
                    }
                };
            }
            Key::Help => unreachable!("help is handled by the shared option parser"),
        }
    }

    let mut operands = parsed.operands.into_iter();
    let filter = operands.next().unwrap_or_else(|| ".".to_string());
    if !interp.resources.charge_cpu(filter.len() as u64) {
        return 137;
    }
    let expression = match Parser::parse(&filter) {
        Ok(expression) => expression,
        Err(error) => {
            interp.note_unsupported(&format!("jq:{error}"));
            ewln(err, &format!("jq: unsupported filter: {error}"));
            return 3;
        }
    };

    let inputs = if null_input {
        vec![Value::Null]
    } else {
        let files = operands.collect::<Vec<_>>();
        let mut inputs = Vec::new();
        if files.is_empty() {
            if let Err(error) = parse_values(&stdin, &mut inputs) {
                ewln(err, &format!("jq: parse error: {error}"));
                return 2;
            }
        } else {
            for file in files {
                let bytes = match interp.vfs.read(&interp.cwd, &file) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        ewln(err, &format!("jq: error: {error}"));
                        return 2;
                    }
                };
                if let Err(error) = parse_values(&bytes, &mut inputs) {
                    ewln(err, &format!("jq: parse error in {file}: {error}"));
                    return 2;
                }
            }
        }
        if slurp {
            vec![Value::Array(inputs)]
        } else {
            inputs
        }
    };

    let mut evaluator = Evaluator {
        interp,
        variables,
        depth: 0,
    };
    let mut results = Vec::new();
    for input in &inputs {
        match evaluator.eval(&expression, input) {
            Ok(mut values) => results.append(&mut values),
            Err(EvalError::Message(error)) => {
                ewln(err, &format!("jq: error: {error}"));
                return 5;
            }
            Err(EvalError::Unsupported(error)) => {
                evaluator.interp.note_unsupported(&format!("jq:{error}"));
                ewln(err, &format!("jq: unsupported filter: {error}"));
                return 3;
            }
            Err(EvalError::Exhausted) => return 137,
        }
    }

    for value in &results {
        if !emit(
            evaluator.interp,
            value,
            raw,
            if compact { 0 } else { indent },
            out,
        ) {
            return 137;
        }
    }
    if exit_status {
        match results.last() {
            None => 4,
            Some(Value::Null | Value::Bool(false)) => 1,
            Some(_) => 0,
        }
    } else {
        0
    }
}

fn extract_variables(args: &[String]) -> Result<(Vec<String>, HashMap<String, Value>), String> {
    let mut filtered = Vec::new();
    let mut variables = HashMap::new();
    let mut index = 0;
    let mut only_operands = false;
    while index < args.len() {
        if args[index] == "--" {
            only_operands = true;
            filtered.push(args[index].clone());
            index += 1;
            continue;
        }
        if !only_operands && matches!(args[index].as_str(), "--arg" | "--argjson") {
            let option = args[index].as_str();
            let Some(name) = args.get(index + 1) else {
                return Err(format!("option {option} requires NAME VALUE"));
            };
            let Some(text) = args.get(index + 2) else {
                return Err(format!("option {option} requires NAME VALUE"));
            };
            let value = if option == "--arg" {
                Value::String(text.clone())
            } else {
                serde_json::from_str(text)
                    .map_err(|error| format!("invalid JSON for --argjson {name}: {error}"))?
            };
            variables.insert(name.clone(), value);
            index += 3;
        } else {
            filtered.push(args[index].clone());
            index += 1;
        }
    }
    Ok((filtered, variables))
}

fn parse_values(bytes: &[u8], values: &mut Vec<Value>) -> Result<(), serde_json::Error> {
    for value in serde_json::Deserializer::from_slice(bytes).into_iter::<Value>() {
        values.push(value?);
    }
    Ok(())
}

fn emit(interp: &mut Interp, value: &Value, raw: bool, indent: usize, out: Out) -> bool {
    let bound = render_bound(value, indent as u64, 0).saturating_add(1);
    if bound > interp.resources.output_remaining()
        || !interp.resources.reserve_memory(bound)
        || !interp.resources.charge_cpu(bound)
    {
        if bound > interp.resources.output_remaining() {
            let request = interp.resources.output_remaining().saturating_add(1);
            let _ = interp.resources.charge_output(request);
        }
        return false;
    }
    if raw {
        if let Value::String(text) = value {
            out.extend_from_slice(text.as_bytes());
            out.push(b'\n');
            return true;
        }
    }
    let Some(capacity) = bound
        .checked_sub(1)
        .and_then(|value| usize::try_from(value).ok())
    else {
        return false;
    };
    let mut rendered = Vec::with_capacity(capacity);
    let result = if indent == 0 {
        serde_json::to_writer(&mut rendered, value)
    } else {
        let indentation = vec![b' '; indent];
        let formatter = serde_json::ser::PrettyFormatter::with_indent(&indentation);
        let mut serializer = serde_json::Serializer::with_formatter(&mut rendered, formatter);
        value.serialize(&mut serializer)
    };
    match result {
        Ok(()) => {
            out.extend_from_slice(&rendered);
            out.push(b'\n');
            true
        }
        Err(_) => false,
    }
}

fn render_bound(value: &Value, indent: u64, depth: u64) -> u64 {
    match value {
        Value::Null => 4,
        Value::Bool(_) => 5,
        Value::Number(number) => number.to_string().len() as u64,
        Value::String(text) => (text.len() as u64).saturating_mul(6).saturating_add(2),
        Value::Array(values) => {
            let children = values.iter().fold(2_u64, |size, value| {
                size.saturating_add(render_bound(value, indent, depth.saturating_add(1)))
                    .saturating_add(1)
            });
            pretty_whitespace(children, values.len(), indent, depth)
        }
        Value::Object(values) => {
            let children = values.iter().fold(2_u64, |size, (key, value)| {
                size.saturating_add((key.len() as u64).saturating_mul(6))
                    .saturating_add(render_bound(value, indent, depth.saturating_add(1)))
                    .saturating_add(5)
            });
            pretty_whitespace(children, values.len(), indent, depth)
        }
    }
}

fn pretty_whitespace(compact_bound: u64, items: usize, indent: u64, depth: u64) -> u64 {
    if indent == 0 || items == 0 {
        return compact_bound;
    }
    let item_count = u64::try_from(items).unwrap_or(u64::MAX);
    compact_bound
        .saturating_add(item_count.saturating_add(1))
        .saturating_add(
            item_count
                .saturating_mul(depth.saturating_add(1))
                .saturating_add(depth)
                .saturating_mul(indent),
        )
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Dot,
    Dollar(String),
    Identifier(String),
    String(String),
    Number(serde_json::Number),
    Null,
    True,
    False,
    If,
    Then,
    Else,
    End,
    And,
    Or,
    Pipe,
    Comma,
    Colon,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
}

struct Lexer<'a> {
    source: &'a str,
    offset: usize,
}

impl Lexer<'_> {
    fn tokenize(source: &str) -> Result<Vec<Token>, String> {
        let mut lexer = Lexer { source, offset: 0 };
        let mut tokens = Vec::new();
        while let Some(token) = lexer.next()? {
            tokens.push(token);
        }
        Ok(tokens)
    }

    fn next(&mut self) -> Result<Option<Token>, String> {
        self.skip_space();
        let Some(character) = self.peek() else {
            return Ok(None);
        };
        let token = match character {
            '.' => {
                self.bump();
                Token::Dot
            }
            '$' => {
                self.bump();
                Token::Dollar(self.identifier()?)
            }
            '"' => Token::String(self.string()?),
            '0'..='9' => self.number()?,
            '-' if self.peek_next().is_some_and(|next| next.is_ascii_digit()) => self.number()?,
            '|' => {
                self.bump();
                Token::Pipe
            }
            ',' => {
                self.bump();
                Token::Comma
            }
            ':' => {
                self.bump();
                Token::Colon
            }
            '(' => {
                self.bump();
                Token::LeftParen
            }
            ')' => {
                self.bump();
                Token::RightParen
            }
            '[' => {
                self.bump();
                Token::LeftBracket
            }
            ']' => {
                self.bump();
                Token::RightBracket
            }
            '{' => {
                self.bump();
                Token::LeftBrace
            }
            '}' => {
                self.bump();
                Token::RightBrace
            }
            '=' if self.peek_next() == Some('=') => {
                self.bump();
                self.bump();
                Token::Eq
            }
            '!' if self.peek_next() == Some('=') => {
                self.bump();
                self.bump();
                Token::Ne
            }
            '<' => {
                self.bump();
                if self.peek() == Some('=') {
                    self.bump();
                    Token::Le
                } else {
                    Token::Lt
                }
            }
            '>' => {
                self.bump();
                if self.peek() == Some('=') {
                    self.bump();
                    Token::Ge
                } else {
                    Token::Gt
                }
            }
            '+' => {
                self.bump();
                Token::Plus
            }
            '-' => {
                self.bump();
                Token::Minus
            }
            character if identifier_start(character) => {
                let identifier = self.identifier()?;
                match identifier.as_str() {
                    "null" => Token::Null,
                    "true" => Token::True,
                    "false" => Token::False,
                    "if" => Token::If,
                    "then" => Token::Then,
                    "else" => Token::Else,
                    "end" => Token::End,
                    "and" => Token::And,
                    "or" => Token::Or,
                    _ => Token::Identifier(identifier),
                }
            }
            _ => return Err(format!("unexpected character {character:?}")),
        };
        Ok(Some(token))
    }

    fn string(&mut self) -> Result<String, String> {
        let start = self.offset;
        self.bump();
        let mut escaped = false;
        while let Some(character) = self.bump() {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                return serde_json::from_str(&self.source[start..self.offset])
                    .map_err(|error| format!("invalid string: {error}"));
            }
        }
        Err("unterminated string".to_string())
    }

    fn number(&mut self) -> Result<Token, String> {
        let start = self.offset;
        if self.peek() == Some('-') {
            self.bump();
        }
        while self.peek().is_some_and(|character| {
            character.is_ascii_digit() || matches!(character, '.' | 'e' | 'E' | '+' | '-')
        }) {
            self.bump();
        }
        let text = &self.source[start..self.offset];
        serde_json::from_str::<Value>(text)
            .ok()
            .and_then(|value| value.as_number().cloned())
            .map(Token::Number)
            .ok_or_else(|| format!("invalid number {text:?}"))
    }

    fn identifier(&mut self) -> Result<String, String> {
        let start = self.offset;
        if !self.peek().is_some_and(identifier_start) {
            return Err("expected identifier".to_string());
        }
        self.bump();
        while self.peek().is_some_and(identifier_continue) {
            self.bump();
        }
        Ok(self.source[start..self.offset].to_string())
    }

    fn skip_space(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.bump();
        }
    }

    fn peek(&self) -> Option<char> {
        self.source[self.offset..].chars().next()
    }

    fn peek_next(&self) -> Option<char> {
        let mut characters = self.source[self.offset..].chars();
        characters.next()?;
        characters.next()
    }

    fn bump(&mut self) -> Option<char> {
        let character = self.peek()?;
        self.offset += character.len_utf8();
        Some(character)
    }
}

fn identifier_start(character: char) -> bool {
    character.is_ascii_alphabetic() || character == '_'
}

fn identifier_continue(character: char) -> bool {
    identifier_start(character) || character.is_ascii_digit()
}

#[derive(Clone, Debug)]
enum Expr {
    Identity,
    Literal(Value),
    Variable(String),
    Field(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Slice(Box<Expr>, Option<i64>, Option<i64>),
    Iterate(Box<Expr>),
    Pipe(Box<Expr>, Box<Expr>),
    Array(Vec<Expr>),
    Object(Vec<(String, Expr)>),
    Call(String, Vec<Expr>),
    If {
        condition: Box<Expr>,
        then_value: Box<Expr>,
        else_value: Box<Expr>,
    },
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    Negate(Box<Expr>),
}

#[derive(Clone, Copy, Debug)]
enum BinaryOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Add,
}

struct Parser {
    tokens: Vec<Token>,
    offset: usize,
    depth: usize,
}

impl Parser {
    fn parse(source: &str) -> Result<Expr, String> {
        let tokens = Lexer::tokenize(source)?;
        if tokens.len() > MAX_FILTER_TOKENS {
            return Err(format!(
                "filter exceeds the {MAX_FILTER_TOKENS}-token limit"
            ));
        }
        let mut parser = Self {
            tokens,
            offset: 0,
            depth: 0,
        };
        let expression = parser.expression()?;
        if let Some(token) = parser.peek() {
            return Err(format!("unexpected token {token:?}"));
        }
        Ok(expression)
    }

    fn expression(&mut self) -> Result<Expr, String> {
        if self.depth >= MAX_EXPRESSION_DEPTH {
            return Err(format!(
                "expression nesting exceeds the {MAX_EXPRESSION_DEPTH}-level limit"
            ));
        }
        self.depth += 1;
        let result = self.pipe();
        self.depth -= 1;
        result
    }

    fn pipe(&mut self) -> Result<Expr, String> {
        let mut expression = self.or()?;
        while self.take(&Token::Pipe) {
            expression = Expr::Pipe(Box::new(expression), Box::new(self.or()?));
        }
        Ok(expression)
    }

    fn or(&mut self) -> Result<Expr, String> {
        let mut expression = self.and()?;
        while self.take(&Token::Or) {
            expression = Expr::Binary(Box::new(expression), BinaryOp::Or, Box::new(self.and()?));
        }
        Ok(expression)
    }

    fn and(&mut self) -> Result<Expr, String> {
        let mut expression = self.comparison()?;
        while self.take(&Token::And) {
            expression = Expr::Binary(
                Box::new(expression),
                BinaryOp::And,
                Box::new(self.comparison()?),
            );
        }
        Ok(expression)
    }

    fn comparison(&mut self) -> Result<Expr, String> {
        let mut expression = self.additive()?;
        loop {
            let operator = if self.take(&Token::Eq) {
                Some(BinaryOp::Eq)
            } else if self.take(&Token::Ne) {
                Some(BinaryOp::Ne)
            } else if self.take(&Token::Le) {
                Some(BinaryOp::Le)
            } else if self.take(&Token::Lt) {
                Some(BinaryOp::Lt)
            } else if self.take(&Token::Ge) {
                Some(BinaryOp::Ge)
            } else if self.take(&Token::Gt) {
                Some(BinaryOp::Gt)
            } else {
                None
            };
            let Some(operator) = operator else {
                break;
            };
            expression = Expr::Binary(Box::new(expression), operator, Box::new(self.additive()?));
        }
        Ok(expression)
    }

    fn additive(&mut self) -> Result<Expr, String> {
        let mut expression = self.unary()?;
        while self.take(&Token::Plus) {
            expression = Expr::Binary(Box::new(expression), BinaryOp::Add, Box::new(self.unary()?));
        }
        Ok(expression)
    }

    fn unary(&mut self) -> Result<Expr, String> {
        if self.take(&Token::Minus) {
            if self.depth >= MAX_EXPRESSION_DEPTH {
                return Err(format!(
                    "expression nesting exceeds the {MAX_EXPRESSION_DEPTH}-level limit"
                ));
            }
            self.depth += 1;
            let value = self.unary();
            self.depth -= 1;
            return Ok(Expr::Negate(Box::new(value?)));
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Expr, String> {
        let mut expression = self.primary()?;
        loop {
            if self.take(&Token::Dot) {
                let Some(Token::Identifier(name)) = self.advance() else {
                    return Err("expected field name after '.'".to_string());
                };
                expression = Expr::Field(Box::new(expression), name);
            } else if self.take(&Token::LeftBracket) {
                if self.take(&Token::RightBracket) {
                    expression = Expr::Iterate(Box::new(expression));
                    continue;
                }
                let start = self.signed_integer();
                if self.take(&Token::Colon) {
                    let end = self.signed_integer();
                    self.expect(Token::RightBracket)?;
                    expression = Expr::Slice(Box::new(expression), start, end);
                } else {
                    let index = if let Some(start) = start {
                        Expr::Literal(Value::from(start))
                    } else {
                        self.expression()?
                    };
                    self.expect(Token::RightBracket)?;
                    expression = Expr::Index(Box::new(expression), Box::new(index));
                }
            } else {
                break;
            }
        }
        Ok(expression)
    }

    fn primary(&mut self) -> Result<Expr, String> {
        let Some(token) = self.advance() else {
            return Err("expected expression".to_string());
        };
        match token {
            Token::Dot => {
                let mut expression = Expr::Identity;
                if let Some(Token::Identifier(name)) = self.peek().cloned() {
                    self.offset += 1;
                    expression = Expr::Field(Box::new(expression), name);
                }
                Ok(expression)
            }
            Token::Dollar(name) => Ok(Expr::Variable(name)),
            Token::String(value) => Ok(Expr::Literal(Value::String(value))),
            Token::Number(value) => Ok(Expr::Literal(Value::Number(value))),
            Token::Null => Ok(Expr::Literal(Value::Null)),
            Token::True => Ok(Expr::Literal(Value::Bool(true))),
            Token::False => Ok(Expr::Literal(Value::Bool(false))),
            Token::LeftParen => {
                let expression = self.expression()?;
                self.expect(Token::RightParen)?;
                Ok(expression)
            }
            Token::LeftBracket => {
                if self.take(&Token::RightBracket) {
                    Ok(Expr::Array(Vec::new()))
                } else {
                    let mut expressions = Vec::new();
                    loop {
                        expressions.push(self.expression()?);
                        if self.take(&Token::RightBracket) {
                            break;
                        }
                        self.expect(Token::Comma)?;
                    }
                    Ok(Expr::Array(expressions))
                }
            }
            Token::LeftBrace => self.object(),
            Token::If => self.conditional(),
            Token::Identifier(name) => {
                if self.take(&Token::LeftParen) {
                    let mut arguments = Vec::new();
                    if !self.take(&Token::RightParen) {
                        loop {
                            arguments.push(self.expression()?);
                            if self.take(&Token::RightParen) {
                                break;
                            }
                            self.expect(Token::Comma)?;
                        }
                    }
                    Ok(Expr::Call(name, arguments))
                } else {
                    Ok(Expr::Call(name, Vec::new()))
                }
            }
            token => Err(format!("unexpected token {token:?}")),
        }
    }

    fn object(&mut self) -> Result<Expr, String> {
        let mut fields = Vec::new();
        if self.take(&Token::RightBrace) {
            return Ok(Expr::Object(fields));
        }
        loop {
            let key = match self.advance() {
                Some(Token::Identifier(key) | Token::String(key)) => key,
                token => return Err(format!("expected object key, found {token:?}")),
            };
            self.expect(Token::Colon)?;
            fields.push((key, self.expression()?));
            if self.take(&Token::RightBrace) {
                break;
            }
            self.expect(Token::Comma)?;
        }
        Ok(Expr::Object(fields))
    }

    fn conditional(&mut self) -> Result<Expr, String> {
        let condition = self.expression()?;
        self.expect(Token::Then)?;
        let then_value = self.expression()?;
        self.expect(Token::Else)?;
        let else_value = self.expression()?;
        self.expect(Token::End)?;
        Ok(Expr::If {
            condition: Box::new(condition),
            then_value: Box::new(then_value),
            else_value: Box::new(else_value),
        })
    }

    fn signed_integer(&mut self) -> Option<i64> {
        let negative = self.take(&Token::Minus);
        let Some(Token::Number(number)) = self.peek() else {
            if negative {
                self.offset = self.offset.saturating_sub(1);
            }
            return None;
        };
        let value = number.as_i64()?;
        self.offset += 1;
        Some(if negative { -value } else { value })
    }

    fn expect(&mut self, expected: Token) -> Result<(), String> {
        if self.take(&expected) {
            Ok(())
        } else {
            Err(format!("expected {expected:?}, found {:?}", self.peek()))
        }
    }

    fn take(&mut self, expected: &Token) -> bool {
        if self.peek() == Some(expected) {
            self.offset += 1;
            true
        } else {
            false
        }
    }

    fn advance(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.offset).cloned()?;
        self.offset += 1;
        Some(token)
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.offset)
    }
}

enum EvalError {
    Message(String),
    Unsupported(String),
    Exhausted,
}

struct Evaluator<'a> {
    interp: &'a mut Interp,
    variables: HashMap<String, Value>,
    depth: usize,
}

impl Evaluator<'_> {
    fn push(&mut self, output: &mut Vec<Value>, value: Value) -> Result<(), EvalError> {
        let bytes = render_bound(&value, 0, 0).saturating_add(32);
        if !self.interp.resources.charge_cpu(1) || !self.interp.resources.reserve_memory(bytes) {
            return Err(EvalError::Exhausted);
        }
        output.push(value);
        Ok(())
    }

    fn one(&mut self, value: Value) -> Result<Vec<Value>, EvalError> {
        let mut output = Vec::new();
        self.push(&mut output, value)?;
        Ok(output)
    }

    fn append(&mut self, output: &mut Vec<Value>, values: Vec<Value>) -> Result<(), EvalError> {
        for value in values {
            self.push(output, value)?;
        }
        Ok(())
    }

    fn eval(&mut self, expression: &Expr, input: &Value) -> Result<Vec<Value>, EvalError> {
        if self.depth >= MAX_EXPRESSION_DEPTH {
            return Err(EvalError::Message(format!(
                "expression nesting exceeds the {MAX_EXPRESSION_DEPTH}-level limit"
            )));
        }
        self.depth += 1;
        let result = self.eval_inner(expression, input);
        self.depth -= 1;
        result
    }

    fn eval_inner(&mut self, expression: &Expr, input: &Value) -> Result<Vec<Value>, EvalError> {
        if !self.interp.resources.charge_cpu(1) {
            return Err(EvalError::Exhausted);
        }
        match expression {
            Expr::Identity => self.one(input.clone()),
            Expr::Literal(value) => self.one(value.clone()),
            Expr::Variable(name) => {
                let value = self
                    .variables
                    .get(name)
                    .cloned()
                    .ok_or_else(|| EvalError::Message(format!("undefined variable ${name}")))?;
                self.one(value)
            }
            Expr::Field(base, name) => {
                let mut output = Vec::new();
                for value in self.eval(base, input)? {
                    match value {
                        Value::Object(object) => {
                            let value = object.get(name).cloned().unwrap_or(Value::Null);
                            self.push(&mut output, value)?;
                        }
                        Value::Null => self.push(&mut output, Value::Null)?,
                        value => {
                            return Err(EvalError::Message(format!(
                                "cannot index {} with {name:?}",
                                type_of(&value)
                            )));
                        }
                    }
                }
                Ok(output)
            }
            Expr::Index(base, index) => {
                let indices = self.eval(index, input)?;
                let mut output = Vec::new();
                for value in self.eval(base, input)? {
                    for index in &indices {
                        let value = index_value(&value, index);
                        self.push(&mut output, value)?;
                    }
                }
                Ok(output)
            }
            Expr::Slice(base, start, end) => {
                let mut output = Vec::new();
                for value in self.eval(base, input)? {
                    let value = slice_value(value, *start, *end)?;
                    self.push(&mut output, value)?;
                }
                Ok(output)
            }
            Expr::Iterate(base) => {
                let mut output = Vec::new();
                for value in self.eval(base, input)? {
                    match value {
                        Value::Array(values) => self.append(&mut output, values)?,
                        Value::Object(values) => {
                            self.append(&mut output, values.into_values().collect())?
                        }
                        value => {
                            return Err(EvalError::Message(format!(
                                "cannot iterate over {}",
                                type_of(&value)
                            )));
                        }
                    }
                }
                Ok(output)
            }
            Expr::Pipe(left, right) => {
                let mut output = Vec::new();
                for value in self.eval(left, input)? {
                    let values = self.eval(right, &value)?;
                    self.append(&mut output, values)?;
                }
                Ok(output)
            }
            Expr::Array(expressions) => {
                let mut values = Vec::new();
                for expression in expressions {
                    let evaluated = self.eval(expression, input)?;
                    self.append(&mut values, evaluated)?;
                }
                self.one(Value::Array(values))
            }
            Expr::Object(fields) => self.object(fields, input),
            Expr::Call(name, arguments) => self.call(name, arguments, input),
            Expr::If {
                condition,
                then_value,
                else_value,
            } => {
                if self.eval(condition, input)?.iter().any(truthy) {
                    self.eval(then_value, input)
                } else {
                    self.eval(else_value, input)
                }
            }
            Expr::Binary(left, operator, right) => {
                let left = self.eval(left, input)?;
                let right = self.eval(right, input)?;
                let mut output = Vec::new();
                for left in &left {
                    for right in &right {
                        let value = binary(left, *operator, right)?;
                        self.push(&mut output, value)?;
                    }
                }
                Ok(output)
            }
            Expr::Negate(value) => {
                let mut output = Vec::new();
                for value in self.eval(value, input)? {
                    let Value::Number(number) = value else {
                        return Err(EvalError::Message("cannot negate non-number".to_string()));
                    };
                    let value = Value::Number(negate_number(&number)?);
                    self.push(&mut output, value)?;
                }
                Ok(output)
            }
        }
    }

    fn object(
        &mut self,
        fields: &[(String, Expr)],
        input: &Value,
    ) -> Result<Vec<Value>, EvalError> {
        let mut objects = vec![Map::new()];
        for (name, expression) in fields {
            let values = self.eval(expression, input)?;
            let mut next = Vec::new();
            for object in &objects {
                for value in &values {
                    let mut object = object.clone();
                    object.insert(name.clone(), value.clone());
                    let value = Value::Object(object);
                    let bytes = render_bound(&value, 0, 0).saturating_add(32);
                    if !self.interp.resources.charge_cpu(1)
                        || !self.interp.resources.reserve_memory(bytes)
                    {
                        return Err(EvalError::Exhausted);
                    }
                    let Value::Object(object) = value else {
                        unreachable!("object candidate changed JSON type")
                    };
                    next.push(object);
                }
            }
            objects = next;
        }
        Ok(objects.into_iter().map(Value::Object).collect())
    }

    fn call(
        &mut self,
        name: &str,
        arguments: &[Expr],
        input: &Value,
    ) -> Result<Vec<Value>, EvalError> {
        match name {
            "select" => {
                arity(name, arguments, 1)?;
                if self.eval(&arguments[0], input)?.iter().any(truthy) {
                    Ok(vec![input.clone()])
                } else {
                    Ok(Vec::new())
                }
            }
            "length" => {
                arity(name, arguments, 0)?;
                let value = match input {
                    Value::Array(values) => Value::from(values.len()),
                    Value::Object(values) => Value::from(values.len()),
                    Value::String(value) => Value::from(value.chars().count()),
                    Value::Number(value) => Value::Number(absolute_number(value)?),
                    Value::Null => Value::from(0),
                    _ => {
                        return Err(EvalError::Message(format!(
                            "length is not defined for {}",
                            type_of(input)
                        )))
                    }
                };
                self.one(value)
            }
            "split" => {
                arity(name, arguments, 1)?;
                let separator = first(self.eval(&arguments[0], input)?, "split separator")?;
                let Some(text) = input.as_str() else {
                    return Err(EvalError::Message(
                        "split input must be a string".to_string(),
                    ));
                };
                let Some(separator) = separator.as_str() else {
                    return Err(EvalError::Message(
                        "split separator must be a string".to_string(),
                    ));
                };
                Ok(vec![Value::Array(
                    text.split(separator)
                        .map(|part| Value::String(part.to_string()))
                        .collect(),
                )])
            }
            "sort_by" => {
                if arguments.is_empty() {
                    return Err(EvalError::Message(
                        "sort_by requires an expression".to_string(),
                    ));
                }
                let Value::Array(values) = input else {
                    return Err(EvalError::Message(
                        "sort_by input must be an array".to_string(),
                    ));
                };
                let mut keyed = Vec::with_capacity(values.len());
                for value in values {
                    let mut keys = Vec::new();
                    for argument in arguments {
                        keys.extend(self.eval(argument, value)?);
                    }
                    keyed.push((keys, value.clone()));
                }
                keyed.sort_by(|left, right| compare_slices(&left.0, &right.0));
                Ok(vec![Value::Array(
                    keyed.into_iter().map(|(_, value)| value).collect(),
                )])
            }
            "contains" => {
                arity(name, arguments, 1)?;
                let needle = first(self.eval(&arguments[0], input)?, "contains argument")?;
                Ok(vec![Value::Bool(contains(input, &needle))])
            }
            "tonumber" => {
                arity(name, arguments, 0)?;
                if input.is_number() {
                    Ok(vec![input.clone()])
                } else if let Some(text) = input.as_str() {
                    serde_json::from_str::<Value>(text)
                        .ok()
                        .filter(Value::is_number)
                        .map(|value| vec![value])
                        .ok_or_else(|| {
                            EvalError::Message(format!("cannot parse {text:?} as number"))
                        })
                } else {
                    Err(EvalError::Message(
                        "tonumber input must be a string or number".to_string(),
                    ))
                }
            }
            "keys" | "keys_unsorted" => {
                arity(name, arguments, 0)?;
                let values = match input {
                    Value::Array(values) => (0..values.len()).map(Value::from).collect(),
                    Value::Object(values) => {
                        let mut keys = values.keys().cloned().collect::<Vec<_>>();
                        if name == "keys" {
                            keys.sort();
                        }
                        keys.into_iter().map(Value::String).collect()
                    }
                    _ => {
                        return Err(EvalError::Message(
                            "keys input must be array or object".to_string(),
                        ))
                    }
                };
                Ok(vec![Value::Array(values)])
            }
            "values" => {
                arity(name, arguments, 0)?;
                Ok(match input {
                    Value::Null => Vec::new(),
                    _ => vec![input.clone()],
                })
            }
            "type" => {
                arity(name, arguments, 0)?;
                Ok(vec![Value::String(type_of(input).to_string())])
            }
            "has" => {
                arity(name, arguments, 1)?;
                let key = first(self.eval(&arguments[0], input)?, "has argument")?;
                let present = match (input, key) {
                    (Value::Object(values), Value::String(key)) => values.contains_key(&key),
                    (Value::Array(values), Value::Number(index)) => index
                        .as_u64()
                        .and_then(|index| usize::try_from(index).ok())
                        .is_some_and(|index| index < values.len()),
                    _ => false,
                };
                Ok(vec![Value::Bool(present)])
            }
            "to_entries" => {
                arity(name, arguments, 0)?;
                let Value::Object(values) = input else {
                    return Err(EvalError::Message(
                        "to_entries input must be an object".to_string(),
                    ));
                };
                Ok(vec![Value::Array(
                    values
                        .iter()
                        .map(|(key, value)| serde_json::json!({"key": key, "value": value}))
                        .collect(),
                )])
            }
            "add" => {
                arity(name, arguments, 0)?;
                let Value::Array(values) = input else {
                    return Err(EvalError::Message("add input must be an array".to_string()));
                };
                let mut sum = Value::Null;
                for value in values {
                    sum = add(&sum, value)?;
                }
                Ok(vec![sum])
            }
            _ => Err(EvalError::Unsupported(format!(
                "unsupported function {name:?}"
            ))),
        }
    }
}

fn arity(name: &str, arguments: &[Expr], expected: usize) -> Result<(), EvalError> {
    if arguments.len() == expected {
        Ok(())
    } else {
        Err(EvalError::Message(format!(
            "{name} expects {expected} argument(s), got {}",
            arguments.len()
        )))
    }
}

fn first(values: Vec<Value>, name: &str) -> Result<Value, EvalError> {
    values
        .into_iter()
        .next()
        .ok_or_else(|| EvalError::Message(format!("{name} produced no value")))
}

fn index_value(value: &Value, index: &Value) -> Value {
    match (value, index) {
        (Value::Array(values), Value::Number(index)) => {
            let index = index.as_i64().unwrap_or_default();
            let index = if index < 0 {
                values.len() as i64 + index
            } else {
                index
            };
            usize::try_from(index)
                .ok()
                .and_then(|index| values.get(index))
                .cloned()
                .unwrap_or(Value::Null)
        }
        (Value::Object(values), Value::String(index)) => {
            values.get(index).cloned().unwrap_or(Value::Null)
        }
        _ => Value::Null,
    }
}

fn slice_value(value: Value, start: Option<i64>, end: Option<i64>) -> Result<Value, EvalError> {
    match value {
        Value::Array(values) => {
            let (start, end) = slice_bounds(values.len(), start, end);
            Ok(Value::Array(values[start..end].to_vec()))
        }
        Value::String(value) => {
            let characters = value.chars().collect::<Vec<_>>();
            let (start, end) = slice_bounds(characters.len(), start, end);
            Ok(Value::String(characters[start..end].iter().collect()))
        }
        _ => Err(EvalError::Message(
            "slice input must be array or string".to_string(),
        )),
    }
}

fn slice_bounds(length: usize, start: Option<i64>, end: Option<i64>) -> (usize, usize) {
    fn bound(length: usize, value: i64) -> usize {
        if value < 0 {
            usize::try_from((length as i64 + value).max(0)).unwrap_or_default()
        } else {
            usize::try_from(value).unwrap_or(usize::MAX).min(length)
        }
    }
    let start = start.map_or(0, |value| bound(length, value));
    let end = end.map_or(length, |value| bound(length, value));
    (start.min(end), end)
}

fn binary(left: &Value, operator: BinaryOp, right: &Value) -> Result<Value, EvalError> {
    let value = match operator {
        BinaryOp::Eq => Value::Bool(left == right),
        BinaryOp::Ne => Value::Bool(left != right),
        BinaryOp::Lt => Value::Bool(compare(left, right) == Ordering::Less),
        BinaryOp::Le => Value::Bool(compare(left, right) != Ordering::Greater),
        BinaryOp::Gt => Value::Bool(compare(left, right) == Ordering::Greater),
        BinaryOp::Ge => Value::Bool(compare(left, right) != Ordering::Less),
        BinaryOp::And => Value::Bool(truthy(left) && truthy(right)),
        BinaryOp::Or => Value::Bool(truthy(left) || truthy(right)),
        BinaryOp::Add => add(left, right)?,
    };
    Ok(value)
}

fn add(left: &Value, right: &Value) -> Result<Value, EvalError> {
    match (left, right) {
        (Value::Null, value) | (value, Value::Null) => Ok(value.clone()),
        (Value::Number(left), Value::Number(right)) => {
            if let (Some(left), Some(right)) = (integer_value(left), integer_value(right)) {
                let value = left
                    .checked_add(right)
                    .ok_or_else(|| EvalError::Message("numeric addition overflow".to_string()))?;
                return integer_number(value).map(Value::Number);
            }
            let value = left.as_f64().unwrap_or_default() + right.as_f64().unwrap_or_default();
            serde_json::Number::from_f64(value)
                .map(Value::Number)
                .ok_or_else(|| EvalError::Message("numeric addition overflow".to_string()))
        }
        (Value::String(left), Value::String(right)) => Ok(Value::String(format!("{left}{right}"))),
        (Value::Array(left), Value::Array(right)) => {
            let mut values = left.clone();
            values.extend(right.clone());
            Ok(Value::Array(values))
        }
        (Value::Object(left), Value::Object(right)) => {
            let mut values = left.clone();
            values.extend(right.clone());
            Ok(Value::Object(values))
        }
        _ => Err(EvalError::Message(format!(
            "cannot add {} and {}",
            type_of(left),
            type_of(right)
        ))),
    }
}

fn integer_value(number: &serde_json::Number) -> Option<i128> {
    number
        .as_i64()
        .map(i128::from)
        .or_else(|| number.as_u64().map(i128::from))
}

fn integer_number(value: i128) -> Result<serde_json::Number, EvalError> {
    if let Ok(value) = i64::try_from(value) {
        Ok(serde_json::Number::from(value))
    } else if let Ok(value) = u64::try_from(value) {
        Ok(serde_json::Number::from(value))
    } else {
        Err(EvalError::Message(
            "integer exceeds jq's supported 64-bit range".to_string(),
        ))
    }
}

fn negate_number(number: &serde_json::Number) -> Result<serde_json::Number, EvalError> {
    if let Some(value) = integer_value(number) {
        return value
            .checked_neg()
            .ok_or_else(|| EvalError::Message("numeric negation overflow".to_string()))
            .and_then(integer_number);
    }
    number
        .as_f64()
        .and_then(|value| serde_json::Number::from_f64(-value))
        .ok_or_else(|| EvalError::Message("numeric negation overflow".to_string()))
}

fn absolute_number(number: &serde_json::Number) -> Result<serde_json::Number, EvalError> {
    if let Some(value) = integer_value(number) {
        return value
            .checked_abs()
            .ok_or_else(|| EvalError::Message("numeric absolute value overflow".to_string()))
            .and_then(integer_number);
    }
    number
        .as_f64()
        .and_then(|value| serde_json::Number::from_f64(value.abs()))
        .ok_or_else(|| EvalError::Message("numeric absolute value overflow".to_string()))
}

fn compare(left: &Value, right: &Value) -> Ordering {
    match (left, right) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Bool(left), Value::Bool(right)) => left.cmp(right),
        (Value::Number(left), Value::Number(right)) => left
            .as_f64()
            .partial_cmp(&right.as_f64())
            .unwrap_or(Ordering::Equal),
        (Value::String(left), Value::String(right)) => left.cmp(right),
        (Value::Array(left), Value::Array(right)) => compare_slices(left, right),
        _ => type_rank(left).cmp(&type_rank(right)),
    }
}

fn compare_slices(left: &[Value], right: &[Value]) -> Ordering {
    left.iter()
        .zip(right)
        .map(|(left, right)| compare(left, right))
        .find(|order| *order != Ordering::Equal)
        .unwrap_or_else(|| left.len().cmp(&right.len()))
}

fn type_rank(value: &Value) -> u8 {
    match value {
        Value::Null => 0,
        Value::Bool(false) => 1,
        Value::Bool(true) => 2,
        Value::Number(_) => 3,
        Value::String(_) => 4,
        Value::Array(_) => 5,
        Value::Object(_) => 6,
    }
}

fn contains(value: &Value, needle: &Value) -> bool {
    match (value, needle) {
        (Value::String(value), Value::String(needle)) => value.contains(needle),
        (Value::Array(values), Value::Array(needles)) => needles
            .iter()
            .all(|needle| values.iter().any(|value| contains(value, needle))),
        (Value::Object(values), Value::Object(needles)) => needles
            .iter()
            .all(|(key, needle)| values.get(key).is_some_and(|value| contains(value, needle))),
        _ => value == needle,
    }
}

fn truthy(value: &Value) -> bool {
    !matches!(value, Value::Null | Value::Bool(false))
}

fn type_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn ewln(err: Out, message: &str) {
    err.extend_from_slice(message.as_bytes());
    err.push(b'\n');
}

#[cfg(test)]
mod tests {
    use super::{Expr, Parser};

    #[test]
    fn parses_tasktrove_filter_shape() {
        let parsed = Parser::parse(
            r#"[.[] | select(.status == "active") | {name: .username, role: (.roles | if length > 0 then .[0] else null end)}] | sort_by(.name)"#,
        );
        assert!(matches!(parsed, Ok(Expr::Pipe(_, _))), "{parsed:?}");
    }

    #[test]
    fn rejects_incomplete_filters() {
        for source in ["select(", "{name}", "if . then . end", ".[x"] {
            assert!(Parser::parse(source).is_err(), "{source}");
        }
    }
}
