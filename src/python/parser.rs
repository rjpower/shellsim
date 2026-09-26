//! Recursive-descent parser for the first expression-and-simple-statement slice.

use super::ast::{
    AssignmentTarget, BinaryOperator, BooleanOperator, CallArgument, CallArgumentKind,
    ComparisonOperator, ComprehensionClause, Constant, DictEntry, ExceptHandler, Expression,
    ExpressionKind, FStringPart, Parameter, ParameterKind, Program, Statement, StatementKind,
    UnaryOperator,
};
use super::source::Span;
use super::token::{Token, TokenKind};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

pub fn parse(tokens: Vec<Token>) -> Result<Program, ParseError> {
    Parser {
        tokens,
        current: 0,
        expression_depth: 0,
        compound_depth: 0,
    }
    .program()
}

struct Parser {
    tokens: Vec<Token>,
    current: usize,
    expression_depth: usize,
    compound_depth: usize,
}

enum ParsedSubscript {
    Index(Expression),
    Slice {
        start: Option<Expression>,
        stop: Option<Expression>,
        step: Option<Expression>,
    },
}

impl ParsedSubscript {
    fn into_expression(self, span: Span) -> Expression {
        match self {
            Self::Index(expression) => expression,
            Self::Slice { start, stop, step } => Expression {
                kind: ExpressionKind::SliceValue {
                    start: start.map(Box::new),
                    stop: stop.map(Box::new),
                    step: step.map(Box::new),
                },
                span,
            },
        }
    }
}

impl Parser {
    fn program(mut self) -> Result<Program, ParseError> {
        let mut statements = Vec::new();
        self.separators();
        while !self.at(|kind| matches!(kind, TokenKind::Eof)) {
            let statement = self.statement()?;
            let compound = statement.kind.is_compound();
            statements.push(statement);
            if !compound
                && !self.at(|kind| {
                    matches!(
                        kind,
                        TokenKind::Semicolon | TokenKind::Newline | TokenKind::Eof
                    )
                })
            {
                return Err(self.error("expected a newline or ';' after statement"));
            }
            self.separators();
        }
        Ok(Program { statements })
    }

    fn statement(&mut self) -> Result<Statement, ParseError> {
        let start = self.peek().span;
        let mut decorators = Vec::new();
        while self.take(|kind| matches!(kind, TokenKind::At)).is_some() {
            decorators.push(self.expression()?);
            self.expect(
                |kind| matches!(kind, TokenKind::Newline),
                "expected a newline after decorator",
            )?;
            self.separators();
        }
        let is_async = self.take(|kind| matches!(kind, TokenKind::Async)).is_some();
        if is_async
            && !self.at(|kind| matches!(kind, TokenKind::Def | TokenKind::For | TokenKind::With))
        {
            return Err(self.error("expected 'def', 'for', or 'with' after 'async'"));
        }
        let kind = if self.take(|kind| matches!(kind, TokenKind::If)).is_some() {
            self.if_statement(start)?
        } else if self.take(|kind| matches!(kind, TokenKind::Try)).is_some() {
            let body = self.suite()?;
            let mut handlers = Vec::new();
            while self
                .take(|kind| matches!(kind, TokenKind::Except))
                .is_some()
            {
                let kind = if self.at(|kind| matches!(kind, TokenKind::Colon | TokenKind::As)) {
                    None
                } else {
                    Some(self.expression()?)
                };
                let name = if self.take(|kind| matches!(kind, TokenKind::As)).is_some() {
                    Some(self.name("expected an exception binding after 'as'")?)
                } else {
                    None
                };
                handlers.push(ExceptHandler {
                    kind,
                    name,
                    body: self.suite()?,
                });
            }
            let otherwise = if self.take(|kind| matches!(kind, TokenKind::Else)).is_some() {
                self.suite()?
            } else {
                Vec::new()
            };
            let finalbody = if self
                .take(|kind| matches!(kind, TokenKind::Finally))
                .is_some()
            {
                self.suite()?
            } else {
                Vec::new()
            };
            if handlers.is_empty() && finalbody.is_empty() {
                return Err(self.error("try statement requires except or finally"));
            }
            StatementKind::Try {
                body,
                handlers,
                otherwise,
                finalbody,
            }
        } else if self.take(|kind| matches!(kind, TokenKind::With)).is_some() {
            let context = self.expression()?;
            let target = if self.take(|kind| matches!(kind, TokenKind::As)).is_some() {
                Some(assignment_target(self.tuple_expression()?)?)
            } else {
                None
            };
            let body = self.suite()?;
            if is_async {
                StatementKind::AsyncWith {
                    context,
                    target,
                    body,
                }
            } else {
                StatementKind::With {
                    context,
                    target,
                    body,
                }
            }
        } else if self.take(|kind| matches!(kind, TokenKind::While)).is_some() {
            let test = self.expression()?;
            let body = self.suite()?;
            let otherwise = if self.take(|kind| matches!(kind, TokenKind::Else)).is_some() {
                self.suite()?
            } else {
                Vec::new()
            };
            StatementKind::While {
                test,
                body,
                otherwise,
            }
        } else if self.take(|kind| matches!(kind, TokenKind::For)).is_some() {
            let target = assignment_target(self.for_target_expression()?)?;
            self.expect(
                |kind| matches!(kind, TokenKind::In),
                "expected 'in' after loop target",
            )?;
            let iterable = self.expression()?;
            let body = self.suite()?;
            let otherwise = if self.take(|kind| matches!(kind, TokenKind::Else)).is_some() {
                self.suite()?
            } else {
                Vec::new()
            };
            if is_async {
                StatementKind::AsyncFor {
                    target,
                    iterable,
                    body,
                    otherwise,
                }
            } else {
                StatementKind::For {
                    target,
                    iterable,
                    body,
                    otherwise,
                }
            }
        } else if self.take(|kind| matches!(kind, TokenKind::Def)).is_some() {
            let name = self.name("expected a function name after 'def'")?;
            self.expect(
                |kind| matches!(kind, TokenKind::LeftParen),
                "expected '(' after function name",
            )?;
            let mut parameters = Vec::new();
            let mut saw_default = false;
            let mut saw_variadic = false;
            let mut keyword_only = false;
            if !self.at(|kind| matches!(kind, TokenKind::RightParen)) {
                loop {
                    let keyword_variadic = self
                        .take(|kind| matches!(kind, TokenKind::DoubleStar))
                        .is_some();
                    let variadic = self.take(|kind| matches!(kind, TokenKind::Star)).is_some();
                    if variadic && self.take(|kind| matches!(kind, TokenKind::Comma)).is_some() {
                        if keyword_only {
                            return Err(self.error("multiple '*' parameters are not allowed"));
                        }
                        keyword_only = true;
                        continue;
                    }
                    if variadic && saw_variadic {
                        return Err(self.error("multiple '*args' parameters are not allowed"));
                    }
                    let name = self.name("expected a parameter name")?;
                    if self.take(|kind| matches!(kind, TokenKind::Colon)).is_some() {
                        self.skip_annotation(|kind| {
                            matches!(
                                kind,
                                TokenKind::Comma | TokenKind::RightParen | TokenKind::Equal
                            )
                        })?;
                    }
                    let default = if self.take(|kind| matches!(kind, TokenKind::Equal)).is_some() {
                        if variadic || keyword_variadic {
                            return Err(self.error("variadic parameters cannot have a default"));
                        }
                        if !keyword_only {
                            saw_default = true;
                        }
                        Some(self.expression()?)
                    } else {
                        if saw_default && !keyword_only && !variadic && !keyword_variadic {
                            return Err(self.error("non-default argument follows default argument"));
                        }
                        None
                    };
                    parameters.push(Parameter {
                        name,
                        default,
                        kind: if keyword_variadic {
                            ParameterKind::KeywordVariadic
                        } else if variadic {
                            ParameterKind::Variadic
                        } else if keyword_only {
                            ParameterKind::KeywordOnly
                        } else {
                            ParameterKind::Positional
                        },
                    });
                    if keyword_variadic {
                        if self.take(|kind| matches!(kind, TokenKind::Comma)).is_some()
                            && !self.at(|kind| matches!(kind, TokenKind::RightParen))
                        {
                            return Err(self.error("parameters cannot follow **kwargs"));
                        }
                        break;
                    }
                    if variadic {
                        saw_variadic = true;
                        keyword_only = true;
                    }
                    if self.take(|kind| matches!(kind, TokenKind::Comma)).is_none() {
                        break;
                    }
                    if self.at(|kind| matches!(kind, TokenKind::RightParen)) {
                        break;
                    }
                }
            }
            self.expect(
                |kind| matches!(kind, TokenKind::RightParen),
                "expected ')' after parameters",
            )?;
            if self.take(|kind| matches!(kind, TokenKind::Arrow)).is_some() {
                self.skip_annotation(|kind| matches!(kind, TokenKind::Colon))?;
            }
            StatementKind::Function {
                name,
                parameters,
                body: self.suite()?,
                is_async,
            }
        } else if self.take(|kind| matches!(kind, TokenKind::Class)).is_some() {
            let name = self.name("expected a class name after 'class'")?;
            let mut bases = Vec::new();
            let mut metaclass = None;
            if self
                .take(|kind| matches!(kind, TokenKind::LeftParen))
                .is_some()
            {
                if !self.at(|kind| matches!(kind, TokenKind::RightParen)) {
                    loop {
                        let keyword = match (
                            &self.peek().kind,
                            self.tokens.get(self.current + 1).map(|token| &token.kind),
                        ) {
                            (TokenKind::Name(name), Some(TokenKind::Equal)) => Some(name.clone()),
                            _ => None,
                        };
                        if let Some(keyword) = keyword {
                            self.advance();
                            self.advance();
                            if keyword != "metaclass" {
                                return Err(self.error("unsupported class keyword argument"));
                            }
                            if metaclass.is_some() {
                                return Err(self.error("metaclass passed more than once"));
                            }
                            metaclass = Some(self.expression()?);
                        } else {
                            if metaclass.is_some() {
                                return Err(
                                    self.error("positional class base follows metaclass keyword")
                                );
                            }
                            bases.push(self.expression()?);
                        }
                        if self.take(|kind| matches!(kind, TokenKind::Comma)).is_none() {
                            break;
                        }
                        if self.at(|kind| matches!(kind, TokenKind::RightParen)) {
                            break;
                        }
                    }
                }
                self.expect(
                    |kind| matches!(kind, TokenKind::RightParen),
                    "expected ')' after class bases",
                )?;
            }
            StatementKind::Class {
                name,
                bases,
                metaclass,
                body: self.suite()?,
            }
        } else if self.take(|kind| matches!(kind, TokenKind::From)).is_some() {
            let mut module = String::new();
            while self.take(|kind| matches!(kind, TokenKind::Dot)).is_some() {
                module.push('.');
            }
            if !self.at(|kind| matches!(kind, TokenKind::Import)) {
                module.push_str(&self.module_name("expected a module name after 'from'")?);
            } else if module.is_empty() {
                return Err(self.error("expected a module name after 'from'"));
            }
            self.expect(
                |kind| matches!(kind, TokenKind::Import),
                "expected 'import' after module name",
            )?;
            let mut names = Vec::new();
            let parenthesized = self
                .take(|kind| matches!(kind, TokenKind::LeftParen))
                .is_some();
            loop {
                let imported = self.name("expected a name to import")?;
                let binding = if self.take(|kind| matches!(kind, TokenKind::As)).is_some() {
                    self.name("expected a binding after 'as'")?
                } else {
                    imported.clone()
                };
                names.push((imported, binding));
                if self.take(|kind| matches!(kind, TokenKind::Comma)).is_none() {
                    break;
                }
                if parenthesized && self.at(|kind| matches!(kind, TokenKind::RightParen)) {
                    break;
                }
            }
            if parenthesized {
                self.expect(
                    |kind| matches!(kind, TokenKind::RightParen),
                    "expected ')' after imported names",
                )?;
            }
            StatementKind::ImportFrom { module, names }
        } else if self
            .take(|kind| matches!(kind, TokenKind::Import))
            .is_some()
        {
            let mut modules = Vec::new();
            loop {
                let module = self.module_name("expected a module name after 'import'")?;
                let binding = if self.take(|kind| matches!(kind, TokenKind::As)).is_some() {
                    self.name("expected a binding after 'as'")?
                } else {
                    module
                        .split('.')
                        .next()
                        .expect("module is not empty")
                        .to_string()
                };
                modules.push((module, binding));
                if self.take(|kind| matches!(kind, TokenKind::Comma)).is_none() {
                    break;
                }
            }
            StatementKind::Import { modules }
        } else if self.take(|kind| matches!(kind, TokenKind::Del)).is_some() {
            StatementKind::Delete(assignment_target(self.postfix()?)?)
        } else if self
            .take(|kind| matches!(kind, TokenKind::Return))
            .is_some()
        {
            let value = if self.at(|kind| {
                matches!(
                    kind,
                    TokenKind::Semicolon | TokenKind::Newline | TokenKind::Dedent
                )
            }) {
                None
            } else {
                Some(self.tuple_expression()?)
            };
            StatementKind::Return(value)
        } else if self.take(|kind| matches!(kind, TokenKind::Break)).is_some() {
            StatementKind::Break
        } else if self
            .take(|kind| matches!(kind, TokenKind::Continue))
            .is_some()
        {
            StatementKind::Continue
        } else if self
            .take(|kind| matches!(kind, TokenKind::Global))
            .is_some()
        {
            let mut names = Vec::new();
            loop {
                names.push(self.name("expected a binding after 'global'")?);
                if self.take(|kind| matches!(kind, TokenKind::Comma)).is_none() {
                    break;
                }
            }
            StatementKind::Global(names)
        } else if self
            .take(|kind| matches!(kind, TokenKind::Nonlocal))
            .is_some()
        {
            let mut names = Vec::new();
            loop {
                names.push(self.name("expected a binding after 'nonlocal'")?);
                if self.take(|kind| matches!(kind, TokenKind::Comma)).is_none() {
                    break;
                }
            }
            StatementKind::Nonlocal(names)
        } else if self.take(|kind| matches!(kind, TokenKind::Pass)).is_some() {
            StatementKind::Pass
        } else if self
            .take(|kind| matches!(kind, TokenKind::Assert))
            .is_some()
        {
            let test = self.expression()?;
            let message = if self.take(|kind| matches!(kind, TokenKind::Comma)).is_some() {
                Some(self.expression()?)
            } else {
                None
            };
            StatementKind::Assert { test, message }
        } else if self.take(|kind| matches!(kind, TokenKind::Raise)).is_some() {
            let value = if self.at(|kind| {
                matches!(
                    kind,
                    TokenKind::Semicolon | TokenKind::Newline | TokenKind::Dedent
                )
            }) {
                None
            } else {
                Some(self.tuple_expression()?)
            };
            StatementKind::Raise(value)
        } else {
            let expression = self.tuple_expression()?;
            if self.take(|kind| matches!(kind, TokenKind::Colon)).is_some() {
                self.skip_annotation(|kind| {
                    matches!(
                        kind,
                        TokenKind::Equal
                            | TokenKind::Semicolon
                            | TokenKind::Newline
                            | TokenKind::Dedent
                    )
                })?;
                if self.take(|kind| matches!(kind, TokenKind::Equal)).is_some() {
                    StatementKind::AnnotatedAssign {
                        target: assignment_target(expression)?,
                        value: Some(self.tuple_expression()?),
                    }
                } else {
                    StatementKind::AnnotatedAssign {
                        target: assignment_target(expression)?,
                        value: None,
                    }
                }
            } else if self.take(|kind| matches!(kind, TokenKind::Equal)).is_some() {
                let mut targets = vec![assignment_target(expression)?];
                let mut value = self.tuple_expression()?;
                while self.take(|kind| matches!(kind, TokenKind::Equal)).is_some() {
                    targets.push(assignment_target(value)?);
                    value = self.tuple_expression()?;
                }
                StatementKind::Assign { targets, value }
            } else if let Some(operator) = self.augmented_operator() {
                StatementKind::AugmentedAssign {
                    target: assignment_target(expression)?,
                    operator,
                    value: self.tuple_expression()?,
                }
            } else {
                StatementKind::Expression(expression)
            }
        };
        let kind = if decorators.is_empty() {
            kind
        } else {
            if !matches!(
                kind,
                StatementKind::Class { .. } | StatementKind::Function { .. }
            ) {
                return Err(self.error("decorators may only be applied to functions and classes"));
            }
            StatementKind::Decorated {
                decorators,
                statement: Box::new(kind),
            }
        };
        Ok(Statement {
            kind,
            span: start.through(self.previous().span),
        })
    }

    fn if_statement(&mut self, start: Span) -> Result<StatementKind, ParseError> {
        let test = self.expression()?;
        let body = self.suite()?;
        let otherwise = if self.take(|kind| matches!(kind, TokenKind::Elif)).is_some() {
            let nested_start = self.previous().span;
            let nested_kind = self.if_statement(nested_start)?;
            vec![Statement {
                kind: nested_kind,
                span: nested_start.through(self.previous().span),
            }]
        } else if self.take(|kind| matches!(kind, TokenKind::Else)).is_some() {
            self.suite()?
        } else {
            Vec::new()
        };
        let _ = start;
        Ok(StatementKind::If {
            test,
            body,
            otherwise,
        })
    }

    fn suite(&mut self) -> Result<Vec<Statement>, ParseError> {
        const MAX_COMPOUND_DEPTH: usize = 256;
        if self.compound_depth == MAX_COMPOUND_DEPTH {
            return Err(self.error("compound-statement nesting limit exceeded"));
        }
        self.compound_depth += 1;
        let result = self.suite_inner();
        self.compound_depth -= 1;
        result
    }

    fn suite_inner(&mut self) -> Result<Vec<Statement>, ParseError> {
        self.expect(
            |kind| matches!(kind, TokenKind::Colon),
            "expected ':' before suite",
        )?;
        if !self.at(|kind| matches!(kind, TokenKind::Newline)) {
            let mut statements = Vec::new();
            loop {
                let statement = self.statement()?;
                if statement.kind.is_compound() {
                    return Err(self.error("compound statement is not allowed in a simple suite"));
                }
                statements.push(statement);
                if self
                    .take(|kind| matches!(kind, TokenKind::Semicolon))
                    .is_none()
                {
                    break;
                }
                if self.at(|kind| matches!(kind, TokenKind::Newline)) {
                    break;
                }
            }
            self.expect(
                |kind| matches!(kind, TokenKind::Newline),
                "expected a newline after simple suite",
            )?;
            return Ok(statements);
        }
        self.advance();
        self.separators();
        self.expect(
            |kind| matches!(kind, TokenKind::Indent),
            "expected an indented suite",
        )?;
        let mut statements = Vec::new();
        self.separators();
        while !self.at(|kind| matches!(kind, TokenKind::Dedent | TokenKind::Eof)) {
            let statement = self.statement()?;
            let compound = statement.kind.is_compound();
            statements.push(statement);
            if !compound
                && !self.at(|kind| {
                    matches!(
                        kind,
                        TokenKind::Semicolon | TokenKind::Newline | TokenKind::Dedent
                    )
                })
            {
                return Err(self.error("expected a newline or ';' after statement"));
            }
            self.separators();
        }
        self.expect(
            |kind| matches!(kind, TokenKind::Dedent),
            "expected the end of an indented suite",
        )?;
        if statements.is_empty() {
            return Err(self.error("expected at least one statement in suite"));
        }
        Ok(statements)
    }

    fn expression(&mut self) -> Result<Expression, ParseError> {
        const MAX_EXPRESSION_DEPTH: usize = 256;
        if self.expression_depth == MAX_EXPRESSION_DEPTH {
            return Err(self.error("expression nesting limit exceeded"));
        }
        self.expression_depth += 1;
        let result = if self
            .take(|kind| matches!(kind, TokenKind::Lambda))
            .is_some()
        {
            self.lambda_expression()
        } else {
            self.named_expression()
        };
        self.expression_depth -= 1;
        result
    }

    fn named_expression(&mut self) -> Result<Expression, ParseError> {
        let target = self.conditional_expression()?;
        if self
            .take(|kind| matches!(kind, TokenKind::ColonEqual))
            .is_none()
        {
            return Ok(target);
        }
        let ExpressionKind::Name(name) = target.kind else {
            return Err(ParseError {
                message: "assignment expression target must be a name".into(),
                span: target.span,
            });
        };
        let value = self.expression()?;
        Ok(Expression {
            span: target.span.through(value.span),
            kind: ExpressionKind::NamedExpression {
                name,
                value: Box::new(value),
            },
        })
    }

    fn expression_item(&mut self) -> Result<Expression, ParseError> {
        if self.take(|kind| matches!(kind, TokenKind::Star)).is_some() {
            let value = self.expression()?;
            let span = self.previous().span;
            Ok(Expression {
                span: value.span.through(span),
                kind: ExpressionKind::Starred(Box::new(value)),
            })
        } else {
            self.expression()
        }
    }

    fn fstring_expression(
        &self,
        body: String,
        raw: bool,
        span: Span,
    ) -> Result<ExpressionKind, ParseError> {
        let mut parts = Vec::new();
        let mut text = String::new();
        let chars: Vec<char> = body.chars().collect();
        let mut index = 0;
        while index < chars.len() {
            match chars[index] {
                '{' if index + 1 < chars.len() && chars[index + 1] == '{' => {
                    text.push('{');
                    index += 2;
                }
                '}' if index + 1 < chars.len() && chars[index + 1] == '}' => {
                    text.push('}');
                    index += 2;
                }
                '{' => {
                    if !text.is_empty() {
                        parts.push(FStringPart::Text(std::mem::take(&mut text)));
                    }
                    let start = index + 1;
                    let mut cursor = start;
                    let mut nesting = 0usize;
                    let mut quote = None;
                    while cursor < chars.len() {
                        let ch = chars[cursor];
                        if let Some(expected) = quote {
                            if ch == expected && (cursor == 0 || chars[cursor - 1] != '\\') {
                                quote = None;
                            }
                        } else if ch == '\'' || ch == '"' {
                            quote = Some(ch);
                        } else if matches!(ch, '(' | '[' | '{') {
                            nesting += 1;
                        } else if matches!(ch, ')' | ']' | '}') {
                            if nesting == 0 {
                                break;
                            }
                            nesting -= 1;
                        }
                        cursor += 1;
                    }
                    if cursor >= chars.len() || quote.is_some() {
                        return Err(ParseError {
                            message: "unterminated f-string expression".into(),
                            span,
                        });
                    }
                    let field: String = chars[start..cursor].iter().collect();
                    let (source, conversion, format_spec) = split_fstring_field(&field, span)?;
                    let embedded_tokens =
                        super::lexer::lex(&source).map_err(|error| ParseError {
                            message: format!("invalid f-string expression: {}", error.message),
                            span,
                        })?;
                    let embedded =
                        super::parser::parse(embedded_tokens).map_err(|error| ParseError {
                            message: format!("invalid f-string expression: {}", error.message),
                            span,
                        })?;
                    let Some(Statement {
                        kind: StatementKind::Expression(expression),
                        ..
                    }) = embedded.statements.into_iter().next()
                    else {
                        return Err(ParseError {
                            message: "f-string braces must contain an expression".into(),
                            span,
                        });
                    };
                    if conversion.is_some() || !format_spec.is_empty() {
                        parts.push(FStringPart::Formatted {
                            expression,
                            conversion,
                            format_spec,
                        });
                    } else {
                        parts.push(FStringPart::Expression(expression));
                    }
                    index = cursor + 1;
                }
                '}' => {
                    return Err(ParseError {
                        message: "single '}' is not allowed in an f-string".into(),
                        span,
                    });
                }
                '\\' if !raw && index + 1 < chars.len() => {
                    let escaped = match chars[index + 1] {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        '\\' => '\\',
                        '\'' => '\'',
                        '"' => '"',
                        other => other,
                    };
                    text.push(escaped);
                    index += 2;
                }
                ch => {
                    text.push(ch);
                    index += 1;
                }
            }
        }
        if !text.is_empty() {
            parts.push(FStringPart::Text(text));
        }
        Ok(ExpressionKind::FString(parts))
    }

    fn lambda_expression(&mut self) -> Result<Expression, ParseError> {
        let start = self.previous().span;
        let mut parameters = Vec::new();
        let mut saw_default = false;
        let mut saw_variadic = false;
        let mut keyword_only = false;
        if !self.at(|kind| matches!(kind, TokenKind::Colon)) {
            loop {
                let keyword_variadic = self
                    .take(|kind| matches!(kind, TokenKind::DoubleStar))
                    .is_some();
                let variadic = self.take(|kind| matches!(kind, TokenKind::Star)).is_some();
                if variadic && self.take(|kind| matches!(kind, TokenKind::Comma)).is_some() {
                    if keyword_only {
                        return Err(self.error("multiple '*' parameters are not allowed"));
                    }
                    keyword_only = true;
                    continue;
                }
                if variadic && saw_variadic {
                    return Err(self.error("multiple '*args' parameters are not allowed"));
                }
                let name = self.name("expected a lambda parameter")?;
                let default = if self.take(|kind| matches!(kind, TokenKind::Equal)).is_some() {
                    if variadic || keyword_variadic {
                        return Err(self.error("variadic parameters cannot have a default"));
                    }
                    if !keyword_only {
                        saw_default = true;
                    }
                    Some(self.expression()?)
                } else {
                    if saw_default && !keyword_only && !variadic && !keyword_variadic {
                        return Err(self.error("non-default argument follows default argument"));
                    }
                    None
                };
                parameters.push(Parameter {
                    name,
                    default,
                    kind: if keyword_variadic {
                        ParameterKind::KeywordVariadic
                    } else if variadic {
                        ParameterKind::Variadic
                    } else if keyword_only {
                        ParameterKind::KeywordOnly
                    } else {
                        ParameterKind::Positional
                    },
                });
                if keyword_variadic {
                    if self.take(|kind| matches!(kind, TokenKind::Comma)).is_some()
                        && !self.at(|kind| matches!(kind, TokenKind::Colon))
                    {
                        return Err(self.error("parameters cannot follow **kwargs"));
                    }
                    break;
                }
                if variadic {
                    saw_variadic = true;
                    keyword_only = true;
                }
                if self.take(|kind| matches!(kind, TokenKind::Comma)).is_none() {
                    break;
                }
            }
        }
        self.expect(
            |kind| matches!(kind, TokenKind::Colon),
            "expected ':' after lambda parameters",
        )?;
        let body = self.expression()?;
        Ok(Expression {
            span: start.through(body.span),
            kind: ExpressionKind::Lambda {
                parameters,
                body: Box::new(body),
            },
        })
    }

    /// Parse the comma-separated expression form used by assignment, return, and `for` targets.
    /// Calls and displays continue to parse each element with `expression`, so their commas remain
    /// delimiters rather than silently becoming nested tuples.
    fn tuple_expression(&mut self) -> Result<Expression, ParseError> {
        let first = self.expression_item()?;
        if self.take(|kind| matches!(kind, TokenKind::Comma)).is_none() {
            return Ok(first);
        }
        let start = first.span;
        let mut values = vec![first];
        while !self.at(|kind| {
            matches!(
                kind,
                TokenKind::Equal
                    | TokenKind::PlusEqual
                    | TokenKind::MinusEqual
                    | TokenKind::In
                    | TokenKind::Colon
                    | TokenKind::Semicolon
                    | TokenKind::Newline
                    | TokenKind::Dedent
                    | TokenKind::Eof
            )
        }) {
            values.push(self.expression_item()?);
            if self.take(|kind| matches!(kind, TokenKind::Comma)).is_none() {
                break;
            }
        }
        Ok(Expression {
            span: start.through(self.previous().span),
            kind: ExpressionKind::Tuple(values),
        })
    }

    fn for_target_expression(&mut self) -> Result<Expression, ParseError> {
        // `in` is a comparison operator in ordinary expressions, so a loop target must stop at it
        // before parsing the iterable expression.
        let first = self.target_item()?;
        if self.take(|kind| matches!(kind, TokenKind::Comma)).is_none() {
            return Ok(first);
        }
        let start = first.span;
        let mut values = vec![first];
        while !self.at(|kind| matches!(kind, TokenKind::In)) {
            values.push(self.target_item()?);
            if self.take(|kind| matches!(kind, TokenKind::Comma)).is_none() {
                break;
            }
        }
        Ok(Expression {
            span: start.through(self.previous().span),
            kind: ExpressionKind::Tuple(values),
        })
    }

    fn target_item(&mut self) -> Result<Expression, ParseError> {
        if self.take(|kind| matches!(kind, TokenKind::Star)).is_some() {
            let value = self.postfix()?;
            Ok(Expression {
                span: value.span,
                kind: ExpressionKind::Starred(Box::new(value)),
            })
        } else {
            self.postfix()
        }
    }

    fn boolean_or(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.boolean_and()?;
        while self.take(|kind| matches!(kind, TokenKind::Or)).is_some() {
            let right = self.boolean_and()?;
            let span = left.span.through(right.span);
            left = Expression {
                kind: ExpressionKind::Boolean {
                    left: Box::new(left),
                    operator: BooleanOperator::Or,
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn conditional_expression(&mut self) -> Result<Expression, ParseError> {
        let body = self.boolean_or()?;
        if self.take(|kind| matches!(kind, TokenKind::If)).is_none() {
            return Ok(body);
        }
        let test = self.boolean_or()?;
        self.expect(
            |kind| matches!(kind, TokenKind::Else),
            "expected 'else' in conditional expression",
        )?;
        let otherwise = self.expression()?;
        Ok(Expression {
            span: body.span.through(otherwise.span),
            kind: ExpressionKind::Conditional {
                test: Box::new(test),
                body: Box::new(body),
                otherwise: Box::new(otherwise),
            },
        })
    }

    fn boolean_and(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.boolean_not()?;
        while self.take(|kind| matches!(kind, TokenKind::And)).is_some() {
            let right = self.boolean_not()?;
            let span = left.span.through(right.span);
            left = Expression {
                kind: ExpressionKind::Boolean {
                    left: Box::new(left),
                    operator: BooleanOperator::And,
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn boolean_not(&mut self) -> Result<Expression, ParseError> {
        let start = self.peek().span;
        if self.take(|kind| matches!(kind, TokenKind::Not)).is_some() {
            let operand = self.boolean_not()?;
            let span = start.through(operand.span);
            Ok(Expression {
                kind: ExpressionKind::Unary {
                    operator: UnaryOperator::Not,
                    operand: Box::new(operand),
                },
                span,
            })
        } else {
            self.comparison()
        }
    }

    fn comparison(&mut self) -> Result<Expression, ParseError> {
        let left = self.bitwise_or()?;
        let mut comparisons = Vec::new();
        loop {
            let operator = if self
                .take(|kind| matches!(kind, TokenKind::EqualEqual))
                .is_some()
            {
                Some(ComparisonOperator::Equal)
            } else if self
                .take(|kind| matches!(kind, TokenKind::NotEqual))
                .is_some()
            {
                Some(ComparisonOperator::NotEqual)
            } else if self
                .take(|kind| matches!(kind, TokenKind::LessEqual))
                .is_some()
            {
                Some(ComparisonOperator::LessEqual)
            } else if self.take(|kind| matches!(kind, TokenKind::Less)).is_some() {
                Some(ComparisonOperator::Less)
            } else if self
                .take(|kind| matches!(kind, TokenKind::GreaterEqual))
                .is_some()
            {
                Some(ComparisonOperator::GreaterEqual)
            } else if self
                .take(|kind| matches!(kind, TokenKind::Greater))
                .is_some()
            {
                Some(ComparisonOperator::Greater)
            } else if self.take(|kind| matches!(kind, TokenKind::In)).is_some() {
                Some(ComparisonOperator::In)
            } else if self.take(|kind| matches!(kind, TokenKind::Is)).is_some() {
                if self.take(|kind| matches!(kind, TokenKind::Not)).is_some() {
                    Some(ComparisonOperator::IsNot)
                } else {
                    Some(ComparisonOperator::Is)
                }
            } else if self.take(|kind| matches!(kind, TokenKind::Not)).is_some() {
                self.expect(
                    |kind| matches!(kind, TokenKind::In),
                    "expected 'in' after 'not' in comparison",
                )?;
                Some(ComparisonOperator::NotIn)
            } else {
                None
            };
            let Some(operator) = operator else { break };
            comparisons.push((operator, self.bitwise_or()?));
        }
        if comparisons.is_empty() {
            Ok(left)
        } else {
            let span = left
                .span
                .through(comparisons.last().expect("non-empty").1.span);
            Ok(Expression {
                kind: ExpressionKind::Comparison {
                    left: Box::new(left),
                    comparisons,
                },
                span,
            })
        }
    }

    fn bitwise_or(&mut self) -> Result<Expression, ParseError> {
        self.binary_chain(
            Self::bitwise_xor,
            TokenKind::Pipe,
            BinaryOperator::BitwiseOr,
        )
    }

    fn bitwise_xor(&mut self) -> Result<Expression, ParseError> {
        self.binary_chain(
            Self::bitwise_and,
            TokenKind::Caret,
            BinaryOperator::BitwiseXor,
        )
    }

    fn bitwise_and(&mut self) -> Result<Expression, ParseError> {
        self.binary_chain(
            Self::shift,
            TokenKind::Ampersand,
            BinaryOperator::BitwiseAnd,
        )
    }

    fn shift(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.additive()?;
        loop {
            let operator = if self
                .take(|kind| matches!(kind, TokenKind::LeftShift))
                .is_some()
            {
                BinaryOperator::LeftShift
            } else if self
                .take(|kind| matches!(kind, TokenKind::RightShift))
                .is_some()
            {
                BinaryOperator::RightShift
            } else {
                break;
            };
            let right = self.additive()?;
            left = Expression {
                span: left.span.through(right.span),
                kind: ExpressionKind::Binary {
                    left: Box::new(left),
                    operator,
                    right: Box::new(right),
                },
            };
        }
        Ok(left)
    }

    fn binary_chain(
        &mut self,
        operand: fn(&mut Self) -> Result<Expression, ParseError>,
        token: TokenKind,
        operator: BinaryOperator,
    ) -> Result<Expression, ParseError> {
        let mut left = operand(self)?;
        while self
            .take(|kind| std::mem::discriminant(kind) == std::mem::discriminant(&token))
            .is_some()
        {
            let right = operand(self)?;
            left = Expression {
                span: left.span.through(right.span),
                kind: ExpressionKind::Binary {
                    left: Box::new(left),
                    operator,
                    right: Box::new(right),
                },
            };
        }
        Ok(left)
    }

    fn additive(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.multiplicative()?;
        loop {
            let operator = if self.take(|kind| matches!(kind, TokenKind::Plus)).is_some() {
                Some(BinaryOperator::Add)
            } else if self.take(|kind| matches!(kind, TokenKind::Minus)).is_some() {
                Some(BinaryOperator::Subtract)
            } else {
                None
            };
            let Some(operator) = operator else { break };
            let right = self.multiplicative()?;
            let span = left.span.through(right.span);
            left = Expression {
                kind: ExpressionKind::Binary {
                    left: Box::new(left),
                    operator,
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn multiplicative(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.unary()?;
        loop {
            let operator = if self.take(|kind| matches!(kind, TokenKind::Star)).is_some() {
                Some(BinaryOperator::Multiply)
            } else if self.take(|kind| matches!(kind, TokenKind::At)).is_some() {
                Some(BinaryOperator::MatrixMultiply)
            } else if self
                .take(|kind| matches!(kind, TokenKind::DoubleSlash))
                .is_some()
            {
                Some(BinaryOperator::FloorDivide)
            } else if self.take(|kind| matches!(kind, TokenKind::Slash)).is_some() {
                Some(BinaryOperator::Divide)
            } else if self
                .take(|kind| matches!(kind, TokenKind::Percent))
                .is_some()
            {
                Some(BinaryOperator::Remainder)
            } else {
                None
            };
            let Some(operator) = operator else { break };
            let right = self.unary()?;
            let span = left.span.through(right.span);
            left = Expression {
                kind: ExpressionKind::Binary {
                    left: Box::new(left),
                    operator,
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Expression, ParseError> {
        let start = self.peek().span;
        if self.take(|kind| matches!(kind, TokenKind::Await)).is_some() {
            let value = self.postfix()?;
            return Ok(Expression {
                span: start.through(value.span),
                kind: ExpressionKind::Await(Box::new(value)),
            });
        }
        let operator = if self.take(|kind| matches!(kind, TokenKind::Plus)).is_some() {
            Some(UnaryOperator::Positive)
        } else if self.take(|kind| matches!(kind, TokenKind::Minus)).is_some() {
            Some(UnaryOperator::Negative)
        } else if self.take(|kind| matches!(kind, TokenKind::Tilde)).is_some() {
            Some(UnaryOperator::Invert)
        } else {
            None
        };
        if let Some(operator) = operator {
            let operand = self.unary()?;
            let span = start.through(operand.span);
            return Ok(Expression {
                kind: ExpressionKind::Unary {
                    operator,
                    operand: Box::new(operand),
                },
                span,
            });
        }
        self.power()
    }

    fn power(&mut self) -> Result<Expression, ParseError> {
        let left = self.postfix()?;
        if self
            .take(|kind| matches!(kind, TokenKind::DoubleStar))
            .is_none()
        {
            return Ok(left);
        }
        let right = self.unary()?;
        Ok(Expression {
            span: left.span.through(right.span),
            kind: ExpressionKind::Binary {
                left: Box::new(left),
                operator: BinaryOperator::Power,
                right: Box::new(right),
            },
        })
    }

    fn augmented_operator(&mut self) -> Option<BinaryOperator> {
        let operator = match self.peek().kind {
            TokenKind::PlusEqual => BinaryOperator::Add,
            TokenKind::MinusEqual => BinaryOperator::Subtract,
            TokenKind::StarEqual => BinaryOperator::Multiply,
            TokenKind::DoubleStarEqual => BinaryOperator::Power,
            TokenKind::SlashEqual => BinaryOperator::Divide,
            TokenKind::DoubleSlashEqual => BinaryOperator::FloorDivide,
            TokenKind::PercentEqual => BinaryOperator::Remainder,
            TokenKind::LeftShiftEqual => BinaryOperator::LeftShift,
            TokenKind::RightShiftEqual => BinaryOperator::RightShift,
            TokenKind::AmpersandEqual => BinaryOperator::BitwiseAnd,
            TokenKind::CaretEqual => BinaryOperator::BitwiseXor,
            TokenKind::PipeEqual => BinaryOperator::BitwiseOr,
            _ => return None,
        };
        self.advance();
        Some(operator)
    }

    fn postfix(&mut self) -> Result<Expression, ParseError> {
        let mut value = self.atom()?;
        loop {
            if self.take(|kind| matches!(kind, TokenKind::Dot)).is_some() {
                let name = self.name("expected an attribute name after '.'")?;
                let span = value.span.through(self.previous().span);
                value = Expression {
                    kind: ExpressionKind::Attribute {
                        value: Box::new(value),
                        name,
                    },
                    span,
                };
            } else if self
                .take(|kind| matches!(kind, TokenKind::LeftBracket))
                .is_some()
            {
                let opening = self.previous().span;
                let mut components = vec![self.subscript_component()?];
                let mut saw_comma = false;
                while self.take(|kind| matches!(kind, TokenKind::Comma)).is_some() {
                    saw_comma = true;
                    if self.at(|kind| matches!(kind, TokenKind::RightBracket)) {
                        break;
                    }
                    components.push(self.subscript_component()?);
                }
                let end = self.expect(
                    |kind| matches!(kind, TokenKind::RightBracket),
                    "expected ']' after subscript",
                )?;
                let span = value.span.through(end.span);
                value = match components.pop() {
                    Some(component) if !saw_comma => Expression {
                        kind: ExpressionKind::Subscript {
                            value: Box::new(value),
                            index: Box::new(component.into_expression(opening.through(end.span))),
                        },
                        span,
                    },
                    last => {
                        if let Some(last) = last {
                            components.push(last);
                        }
                        let indices = components
                            .into_iter()
                            .map(|component| component.into_expression(opening.through(end.span)))
                            .collect();
                        let index = Expression {
                            kind: ExpressionKind::Tuple(indices),
                            span: opening.through(end.span),
                        };
                        Expression {
                            kind: ExpressionKind::Subscript {
                                value: Box::new(value),
                                index: Box::new(index),
                            },
                            span,
                        }
                    }
                };
            } else if self
                .take(|kind| matches!(kind, TokenKind::LeftParen))
                .is_some()
            {
                let mut arguments = Vec::new();
                let mut saw_keyword = false;
                if !self.at(|kind| matches!(kind, TokenKind::RightParen)) {
                    loop {
                        let keyword_unpack = self
                            .take(|kind| matches!(kind, TokenKind::DoubleStar))
                            .is_some();
                        let name = if keyword_unpack {
                            None
                        } else {
                            match (
                                &self.peek().kind,
                                self.tokens.get(self.current + 1).map(|token| &token.kind),
                            ) {
                                (TokenKind::Name(name), Some(TokenKind::Equal)) => {
                                    let name = name.clone();
                                    self.advance();
                                    self.advance();
                                    Some(name)
                                }
                                _ => None,
                            }
                        };
                        let starred = !keyword_unpack
                            && self.take(|kind| matches!(kind, TokenKind::Star)).is_some();
                        if (name.is_none() && !keyword_unpack && !starred) && saw_keyword {
                            return Err(self.error("positional argument follows keyword argument"));
                        }
                        if starred && saw_keyword {
                            return Err(self.error("positional argument follows keyword argument"));
                        }
                        if name.is_some() || keyword_unpack {
                            saw_keyword = true;
                        }
                        let mut argument_value = self.expression()?;
                        // Python permits the unparenthesized generator form in a call, e.g.
                        // ``sum(value for value in values)``.  The surrounding call already
                        // supplies the closing delimiter, so only the clauses belong here.
                        if name.is_none()
                            && !keyword_unpack
                            && !starred
                            && self.take(|kind| matches!(kind, TokenKind::For)).is_some()
                        {
                            let (clauses, _) = self.comprehension_clauses(argument_value.span)?;
                            argument_value = Expression {
                                span: argument_value.span,
                                kind: ExpressionKind::GeneratorExpression {
                                    element: Box::new(argument_value),
                                    clauses,
                                },
                            };
                        }
                        arguments.push(CallArgument {
                            value: argument_value,
                            kind: if keyword_unpack {
                                CallArgumentKind::KeywordUnpack
                            } else if starred {
                                CallArgumentKind::PositionalUnpack
                            } else if let Some(name) = name {
                                CallArgumentKind::Keyword(name)
                            } else {
                                CallArgumentKind::Positional
                            },
                        });
                        if self.take(|kind| matches!(kind, TokenKind::Comma)).is_none() {
                            break;
                        }
                        if self.at(|kind| matches!(kind, TokenKind::RightParen)) {
                            break;
                        }
                    }
                }
                let end = self.expect(
                    |kind| matches!(kind, TokenKind::RightParen),
                    "expected ')' after call arguments",
                )?;
                let span = value.span.through(end.span);
                value = Expression {
                    kind: ExpressionKind::Call {
                        function: Box::new(value),
                        arguments,
                    },
                    span,
                };
            } else {
                break;
            }
        }
        Ok(value)
    }

    fn subscript_component(&mut self) -> Result<ParsedSubscript, ParseError> {
        let start = if self.at(|kind| matches!(kind, TokenKind::Colon)) {
            None
        } else {
            Some(self.expression()?)
        };
        if self.take(|kind| matches!(kind, TokenKind::Colon)).is_none() {
            return start
                .map(ParsedSubscript::Index)
                .ok_or_else(|| self.error("expected subscript index"));
        }
        let stop = if self.at(|kind| {
            matches!(
                kind,
                TokenKind::Colon | TokenKind::Comma | TokenKind::RightBracket
            )
        }) {
            None
        } else {
            Some(self.expression()?)
        };
        let step = if self.take(|kind| matches!(kind, TokenKind::Colon)).is_some() {
            if self.at(|kind| matches!(kind, TokenKind::Comma | TokenKind::RightBracket)) {
                None
            } else {
                Some(self.expression()?)
            }
        } else {
            None
        };
        Ok(ParsedSubscript::Slice { start, stop, step })
    }

    fn atom(&mut self) -> Result<Expression, ParseError> {
        let token = self.advance().clone();
        if matches!(
            token.kind,
            TokenKind::String(_) | TokenKind::Bytes(_) | TokenKind::FString { .. }
        ) {
            return self.string_atom(token);
        }
        let kind = match token.kind {
            TokenKind::Integer(value) => ExpressionKind::Constant(Constant::Integer(value)),
            TokenKind::BigInteger(value) => ExpressionKind::Constant(Constant::BigInteger(value)),
            TokenKind::Float(value) => ExpressionKind::Constant(Constant::Float(value)),
            TokenKind::Imaginary(value) => ExpressionKind::Constant(Constant::Imaginary(value)),
            TokenKind::None => ExpressionKind::Constant(Constant::None),
            TokenKind::True => ExpressionKind::Constant(Constant::Bool(true)),
            TokenKind::False => ExpressionKind::Constant(Constant::Bool(false)),
            TokenKind::Name(name) => ExpressionKind::Name(name),
            TokenKind::Yield => {
                if self.take(|kind| matches!(kind, TokenKind::From)).is_some() {
                    let value = self.expression()?;
                    return Ok(Expression {
                        kind: ExpressionKind::YieldFrom(Box::new(value)),
                        span: token.span.through(self.previous().span),
                    });
                }
                let value = if self.at(|kind| {
                    matches!(
                        kind,
                        TokenKind::Newline
                            | TokenKind::Dedent
                            | TokenKind::Eof
                            | TokenKind::Comma
                            | TokenKind::Colon
                            | TokenKind::RightParen
                            | TokenKind::RightBracket
                            | TokenKind::RightBrace
                    )
                }) {
                    None
                } else {
                    Some(Box::new(self.expression()?))
                };
                return Ok(Expression {
                    kind: ExpressionKind::Yield(value),
                    span: token.span.through(self.previous().span),
                });
            }
            TokenKind::LeftBracket => {
                if self.at(|kind| matches!(kind, TokenKind::RightBracket)) {
                    self.advance();
                    return Ok(Expression {
                        span: token.span.through(self.previous().span),
                        kind: ExpressionKind::List(Vec::new()),
                    });
                }
                let first = self.expression_item()?;
                if self.take(|kind| matches!(kind, TokenKind::For)).is_some() {
                    let (clauses, _) = self.comprehension_clauses(first.span)?;
                    let end = self.expect(
                        |kind| matches!(kind, TokenKind::RightBracket),
                        "expected ']' after list comprehension",
                    )?;
                    return Ok(Expression {
                        span: token.span.through(end.span),
                        kind: ExpressionKind::ListComprehension {
                            element: Box::new(first),
                            clauses,
                        },
                    });
                }
                let mut values = vec![first];
                while self.take(|kind| matches!(kind, TokenKind::Comma)).is_some()
                    && !self.at(|kind| matches!(kind, TokenKind::RightBracket))
                {
                    values.push(self.expression_item()?);
                }
                let end = self.expect(
                    |kind| matches!(kind, TokenKind::RightBracket),
                    "expected ']' after list display",
                )?;
                return Ok(Expression {
                    span: token.span.through(end.span),
                    kind: ExpressionKind::List(values),
                });
            }
            TokenKind::LeftBrace => return self.brace_display(token.span),
            TokenKind::LeftParen => {
                if self
                    .take(|kind| matches!(kind, TokenKind::RightParen))
                    .is_some()
                {
                    return Ok(Expression {
                        span: token.span.through(self.previous().span),
                        kind: ExpressionKind::Tuple(Vec::new()),
                    });
                }
                let expression = self.expression_item()?;
                if self.take(|kind| matches!(kind, TokenKind::For)).is_some() {
                    let (clauses, _) = self.comprehension_clauses(expression.span)?;
                    let end = self.expect(
                        |kind| matches!(kind, TokenKind::RightParen),
                        "expected ')' after generator expression",
                    )?;
                    return Ok(Expression {
                        span: token.span.through(end.span),
                        kind: ExpressionKind::GeneratorExpression {
                            element: Box::new(expression),
                            clauses,
                        },
                    });
                }
                if self.take(|kind| matches!(kind, TokenKind::Comma)).is_some() {
                    let mut values = vec![expression];
                    while !self.at(|kind| matches!(kind, TokenKind::RightParen)) {
                        values.push(self.expression_item()?);
                        if self.take(|kind| matches!(kind, TokenKind::Comma)).is_none() {
                            break;
                        }
                    }
                    self.expect(
                        |kind| matches!(kind, TokenKind::RightParen),
                        "expected ')' after tuple",
                    )?;
                    return Ok(Expression {
                        span: token.span.through(self.previous().span),
                        kind: ExpressionKind::Tuple(values),
                    });
                }
                self.expect(
                    |kind| matches!(kind, TokenKind::RightParen),
                    "expected ')' after expression",
                )?;
                return Ok(Expression {
                    span: token.span.through(self.previous().span),
                    ..expression
                });
            }
            _ => {
                return Err(ParseError {
                    message: "expected an expression".into(),
                    span: token.span,
                })
            }
        };
        Ok(Expression {
            kind,
            span: token.span,
        })
    }

    /// Fold Python's adjacent literal syntax while it is still a sequence of tokens.
    /// Keeping this in the parser avoids introducing a runtime concatenation path for
    /// values that CPython also combines at compile time.
    fn string_atom(&mut self, first: Token) -> Result<Expression, ParseError> {
        let start = first.span;
        let mut end = first.span;
        let mut bytes = None::<Vec<u8>>;
        let mut parts = Vec::new();
        let mut saw_fstring = false;
        let mut token = Some(first);

        while let Some(current) = token.take() {
            end = current.span;
            match current.kind {
                TokenKind::String(value) => {
                    if bytes.is_some() {
                        return Err(ParseError {
                            message: "cannot mix bytes and nonbytes literals".into(),
                            span: current.span,
                        });
                    }
                    parts.push(FStringPart::Text(value));
                }
                TokenKind::Bytes(value) => {
                    if !parts.is_empty() || saw_fstring {
                        return Err(ParseError {
                            message: "cannot mix bytes and nonbytes literals".into(),
                            span: current.span,
                        });
                    }
                    bytes.get_or_insert_default().extend(value);
                }
                TokenKind::FString { body, raw } => {
                    if bytes.is_some() {
                        return Err(ParseError {
                            message: "cannot mix bytes and nonbytes literals".into(),
                            span: current.span,
                        });
                    }
                    saw_fstring = true;
                    let ExpressionKind::FString(mut next) =
                        self.fstring_expression(body, raw, current.span)?
                    else {
                        unreachable!("f-string parsing always returns an f-string")
                    };
                    parts.append(&mut next);
                }
                _ => unreachable!("string_atom receives only literal tokens"),
            }
            if self.at(|kind| {
                matches!(
                    kind,
                    TokenKind::String(_) | TokenKind::Bytes(_) | TokenKind::FString { .. }
                )
            }) {
                token = Some(self.advance().clone());
            }
        }

        let kind = if let Some(bytes) = bytes {
            ExpressionKind::Constant(Constant::Bytes(bytes))
        } else if saw_fstring {
            ExpressionKind::FString(parts)
        } else {
            let mut value = String::new();
            for part in parts {
                let FStringPart::Text(text) = part else {
                    unreachable!("plain strings contain only text")
                };
                value.push_str(&text);
            }
            ExpressionKind::Constant(Constant::String(value))
        };
        Ok(Expression {
            kind,
            span: start.through(end),
        })
    }

    fn brace_display(&mut self, start: Span) -> Result<Expression, ParseError> {
        if self
            .take(|kind| matches!(kind, TokenKind::RightBrace))
            .is_some()
        {
            return Ok(Expression {
                span: start.through(self.previous().span),
                kind: ExpressionKind::Dict(Vec::new()),
            });
        }
        if self
            .take(|kind| matches!(kind, TokenKind::DoubleStar))
            .is_some()
        {
            let mut entries = vec![DictEntry::Unpack(self.expression()?)];
            while self.take(|kind| matches!(kind, TokenKind::Comma)).is_some()
                && !self.at(|kind| matches!(kind, TokenKind::RightBrace))
            {
                entries.push(self.dict_entry()?);
            }
            self.expect(
                |kind| matches!(kind, TokenKind::RightBrace),
                "expected '}' after dictionary display",
            )?;
            return Ok(Expression {
                span: start.through(self.previous().span),
                kind: ExpressionKind::Dict(entries),
            });
        }
        let first = self.expression()?;
        if self.take(|kind| matches!(kind, TokenKind::Colon)).is_some() {
            let value = self.expression()?;
            if self.take(|kind| matches!(kind, TokenKind::For)).is_some() {
                let (clauses, _) = self.comprehension_clauses(first.span)?;
                let end = self.expect(
                    |kind| matches!(kind, TokenKind::RightBrace),
                    "expected '}' after dictionary comprehension",
                )?;
                return Ok(Expression {
                    span: start.through(end.span),
                    kind: ExpressionKind::DictComprehension {
                        key: Box::new(first),
                        value: Box::new(value),
                        clauses,
                    },
                });
            }
            let mut entries = vec![DictEntry::Pair(first, value)];
            while self.take(|kind| matches!(kind, TokenKind::Comma)).is_some()
                && !self.at(|kind| matches!(kind, TokenKind::RightBrace))
            {
                entries.push(self.dict_entry()?);
            }
            self.expect(
                |kind| matches!(kind, TokenKind::RightBrace),
                "expected '}' after dictionary display",
            )?;
            Ok(Expression {
                span: start.through(self.previous().span),
                kind: ExpressionKind::Dict(entries),
            })
        } else {
            if self.take(|kind| matches!(kind, TokenKind::For)).is_some() {
                let (clauses, _) = self.comprehension_clauses(first.span)?;
                let end = self.expect(
                    |kind| matches!(kind, TokenKind::RightBrace),
                    "expected '}' after set comprehension",
                )?;
                return Ok(Expression {
                    span: start.through(end.span),
                    kind: ExpressionKind::SetComprehension {
                        element: Box::new(first),
                        clauses,
                    },
                });
            }
            let mut values = vec![first];
            while self.take(|kind| matches!(kind, TokenKind::Comma)).is_some()
                && !self.at(|kind| matches!(kind, TokenKind::RightBrace))
            {
                values.push(self.expression()?);
            }
            self.expect(
                |kind| matches!(kind, TokenKind::RightBrace),
                "expected '}' after set display",
            )?;
            Ok(Expression {
                span: start.through(self.previous().span),
                kind: ExpressionKind::Set(values),
            })
        }
    }

    fn dict_entry(&mut self) -> Result<DictEntry, ParseError> {
        if self
            .take(|kind| matches!(kind, TokenKind::DoubleStar))
            .is_some()
        {
            return Ok(DictEntry::Unpack(self.expression()?));
        }
        let key = self.expression()?;
        self.expect(
            |kind| matches!(kind, TokenKind::Colon),
            "expected ':' between dictionary key and value",
        )?;
        Ok(DictEntry::Pair(key, self.expression()?))
    }

    fn comprehension_clauses(
        &mut self,
        _element_span: Span,
    ) -> Result<(Vec<ComprehensionClause>, Span), ParseError> {
        let mut clauses = Vec::new();
        loop {
            let target = assignment_target(self.for_target_expression()?)?;
            self.expect(
                |kind| matches!(kind, TokenKind::In),
                "expected 'in' after comprehension target",
            )?;
            // An unparenthesized comprehension iterable stops before the following `if`/`for`.
            // Conditional expressions remain available when parenthesized inside this position.
            let iterable = self.boolean_or()?;
            let mut conditions = Vec::new();
            while self.take(|kind| matches!(kind, TokenKind::If)).is_some() {
                conditions.push(self.boolean_or()?);
            }
            clauses.push(ComprehensionClause {
                target,
                iterable,
                conditions,
            });
            if self.take(|kind| matches!(kind, TokenKind::For)).is_none() {
                break;
            }
        }
        Ok((clauses, self.previous().span))
    }

    fn separators(&mut self) {
        while self
            .take(|kind| matches!(kind, TokenKind::Semicolon | TokenKind::Newline))
            .is_some()
        {}
    }

    fn name(&mut self, message: &str) -> Result<String, ParseError> {
        let token = self.advance().clone();
        if let TokenKind::Name(name) = token.kind {
            Ok(name)
        } else {
            Err(ParseError {
                message: message.into(),
                span: token.span,
            })
        }
    }

    fn module_name(&mut self, message: &str) -> Result<String, ParseError> {
        let mut module = self.name(message)?;
        while self.take(|kind| matches!(kind, TokenKind::Dot)).is_some() {
            module.push('.');
            module.push_str(&self.name("expected a name after '.'")?);
        }
        Ok(module)
    }

    fn skip_annotation(&mut self, stop: impl Fn(&TokenKind) -> bool) -> Result<(), ParseError> {
        let mut depth = 0usize;
        let mut consumed = false;
        loop {
            let kind = &self.peek().kind;
            if depth == 0 && stop(kind) {
                break;
            }
            match kind {
                TokenKind::LeftParen | TokenKind::LeftBracket | TokenKind::LeftBrace => depth += 1,
                TokenKind::RightParen | TokenKind::RightBracket | TokenKind::RightBrace => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                }
                TokenKind::Newline | TokenKind::Dedent | TokenKind::Eof if depth == 0 => break,
                _ => {}
            }
            consumed = true;
            self.advance();
        }
        if consumed {
            Ok(())
        } else {
            Err(self.error("expected an annotation"))
        }
    }

    fn expect(
        &mut self,
        predicate: impl FnOnce(&TokenKind) -> bool,
        message: &str,
    ) -> Result<Token, ParseError> {
        if predicate(&self.peek().kind) {
            Ok(self.advance().clone())
        } else {
            Err(self.error(message))
        }
    }

    fn take(&mut self, predicate: impl FnOnce(&TokenKind) -> bool) -> Option<Token> {
        if predicate(&self.peek().kind) {
            Some(self.advance().clone())
        } else {
            None
        }
    }

    fn at(&self, predicate: impl FnOnce(&TokenKind) -> bool) -> bool {
        predicate(&self.peek().kind)
    }

    fn advance(&mut self) -> &Token {
        let index = self.current;
        if !matches!(self.tokens[index].kind, TokenKind::Eof) {
            self.current += 1;
        }
        &self.tokens[index]
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.current.saturating_sub(1)]
    }

    fn error(&self, message: impl Into<String>) -> ParseError {
        ParseError {
            message: message.into(),
            span: self.peek().span,
        }
    }
}

fn split_fstring_field(
    source: &str,
    span: Span,
) -> Result<(String, Option<char>, String), ParseError> {
    let mut nesting = 0usize;
    let mut quote = None;
    let mut escaped = false;
    let characters = source.chars().collect::<Vec<_>>();
    let mut marker = None;
    for (index, ch) in characters.iter().copied().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if let Some(expected) = quote {
            if ch == expected {
                quote = None;
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
        } else if matches!(ch, '(' | '[' | '{') {
            nesting += 1;
        } else if matches!(ch, ')' | ']' | '}') {
            nesting = nesting.saturating_sub(1);
        } else if nesting == 0 && matches!(ch, '!' | ':') {
            marker = Some((index, ch));
            break;
        }
    }
    let Some((index, marker)) = marker else {
        return Ok((source.to_string(), None, String::new()));
    };
    let expression = characters[..index].iter().collect::<String>();
    if expression.trim().is_empty() {
        return Err(ParseError {
            message: "f-string field requires an expression".into(),
            span,
        });
    }
    if marker == ':' {
        return Ok((expression, None, characters[index + 1..].iter().collect()));
    }
    let conversion = characters
        .get(index + 1)
        .copied()
        .ok_or_else(|| ParseError {
            message: "f-string conversion requires a conversion character".into(),
            span,
        })?;
    if !matches!(conversion, 's' | 'r' | 'a') {
        return Err(ParseError {
            message: format!("unsupported f-string conversion !{conversion}"),
            span,
        });
    }
    let format_spec = match characters.get(index + 2) {
        None => String::new(),
        Some(':') => characters[index + 3..].iter().collect(),
        Some(_) => {
            return Err(ParseError {
                message: "expected ':' or '}' after f-string conversion".into(),
                span,
            })
        }
    };
    Ok((expression, Some(conversion), format_spec))
}

fn assignment_target(expression: Expression) -> Result<AssignmentTarget, ParseError> {
    let span = expression.span;
    match expression.kind {
        ExpressionKind::Name(name) => Ok(AssignmentTarget::Name(name)),
        ExpressionKind::Starred(value) => {
            Ok(AssignmentTarget::Star(Box::new(assignment_target(*value)?)))
        }
        ExpressionKind::Tuple(values) | ExpressionKind::List(values) => values
            .into_iter()
            .map(assignment_target)
            .collect::<Result<Vec<_>, _>>()
            .map(AssignmentTarget::Sequence),
        ExpressionKind::Attribute { value, name } => Ok(AssignmentTarget::Attribute {
            value: *value,
            name,
        }),
        ExpressionKind::Subscript { value, index } => Ok(AssignmentTarget::Subscript {
            value: *value,
            index: *index,
        }),
        _ => Err(ParseError {
            message: "cannot assign to this expression".into(),
            span,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::python::lexer::lex;

    #[test]
    fn multiplication_binds_tighter_than_addition() {
        let program = parse(lex("answer = 1 + 2 * 3").unwrap()).unwrap();
        let StatementKind::Assign { value, .. } = &program.statements[0].kind else {
            panic!("expected assignment")
        };
        let ExpressionKind::Binary { right, .. } = &value.kind else {
            panic!("expected addition")
        };
        assert!(matches!(right.kind, ExpressionKind::Binary { .. }));
    }

    #[test]
    fn parses_chained_postfix_operations() {
        parse(lex("sys.stdout.write(sys.argv[1])").unwrap()).unwrap();
    }

    #[test]
    fn parses_comma_separated_imports_and_rejects_a_missing_module() {
        let program = parse(lex("import math, urllib.error as errors, random").unwrap()).unwrap();
        let StatementKind::Import { modules } = &program.statements[0].kind else {
            panic!("expected import statement")
        };
        assert_eq!(
            modules,
            &[
                ("math".into(), "math".into()),
                ("urllib.error".into(), "errors".into()),
                ("random".into(), "random".into())
            ]
        );
        for source in ["import math,", "import , math"] {
            let error = parse(lex(source).unwrap()).expect_err("missing module must fail");
            assert!(error.message.contains("expected a module name"));
        }
    }

    #[test]
    fn parses_tuple_subscripts_and_rejects_empty_tuple_members() {
        parse(lex("pair = values[1, 2]\nsingle = values[1,]").unwrap()).unwrap();
        let error = parse(lex("values[1,,2]").unwrap())
            .expect_err("a tuple subscript cannot contain an empty member");
        assert!(error.message.contains("expected an expression"));
    }

    #[test]
    fn rejects_excessive_compound_statement_nesting() {
        let mut source = String::new();
        for depth in 0..300 {
            source.push_str(&"    ".repeat(depth));
            source.push_str("if True:\n");
        }
        source.push_str(&"    ".repeat(300));
        source.push_str("pass\n");
        let error = parse(lex(&source).expect("the adversarial source should lex"))
            .expect_err("compound nesting must be bounded");
        assert!(error.message.contains("compound-statement nesting limit"));
    }
}
