//! Recursive-descent parser for the supported awk grammar.

use super::ast::{AssignOp, BinaryOp, Expr, LValue, Pattern, Program, Rule, Stmt, UnaryOp};
use super::lexer::{lex, Token};

pub(super) fn parse(source: &str) -> Result<Program, String> {
    let tokens = lex(source)?;
    Parser { tokens, cursor: 0 }.program()
}

struct Parser {
    tokens: Vec<Token>,
    cursor: usize,
}

impl Parser {
    fn program(&mut self) -> Result<Program, String> {
        let mut rules = Vec::new();
        self.separators();
        while !self.at(&Token::Eof) {
            let pattern = if self.take(&Token::Begin) {
                Pattern::Begin
            } else if self.take(&Token::End) {
                Pattern::End
            } else if self.at(&Token::LBrace) {
                Pattern::Always
            } else {
                Pattern::Expr(self.expression()?)
            };
            self.separators();
            let body = if self.at(&Token::LBrace) {
                match self.statement()? {
                    Stmt::Block(statements) => statements,
                    _ => unreachable!("brace parser returns a block"),
                }
            } else {
                vec![Stmt::Print(Vec::new())]
            };
            rules.push(Rule { pattern, body });
            self.separators();
        }
        if rules.is_empty() {
            return Err("empty program".to_string());
        }
        Ok(Program { rules })
    }

    fn statement(&mut self) -> Result<Stmt, String> {
        self.separators();
        if self.take(&Token::LBrace) {
            let mut statements = Vec::new();
            self.separators();
            while !self.take(&Token::RBrace) {
                if self.at(&Token::Eof) {
                    return Err("unterminated statement block".to_string());
                }
                statements.push(self.statement()?);
                self.separators();
            }
            return Ok(Stmt::Block(statements));
        }
        if self.take(&Token::If) {
            self.expect(&Token::LParen, "expected '(' after if")?;
            let condition = self.expression()?;
            self.expect(&Token::RParen, "expected ')' after if condition")?;
            let then_branch = Box::new(self.statement()?);
            self.separators();
            let else_branch = if self.take(&Token::Else) {
                Some(Box::new(self.statement()?))
            } else {
                None
            };
            return Ok(Stmt::If {
                condition,
                then_branch,
                else_branch,
            });
        }
        if self.take(&Token::While) {
            self.expect(&Token::LParen, "expected '(' after while")?;
            let condition = self.expression()?;
            self.expect(&Token::RParen, "expected ')' after while condition")?;
            return Ok(Stmt::While {
                condition,
                body: Box::new(self.statement()?),
            });
        }
        if self.take(&Token::For) {
            return self.for_statement();
        }
        if self.take(&Token::Break) {
            return Ok(Stmt::Break);
        }
        if self.take(&Token::Continue) {
            return Ok(Stmt::Continue);
        }
        if self.take(&Token::Next) {
            return Ok(Stmt::Next);
        }
        if self.take(&Token::NextFile) {
            return Ok(Stmt::NextFile);
        }
        if self.take(&Token::Exit) {
            let value = if self.statement_end() {
                None
            } else {
                Some(self.expression()?)
            };
            return Ok(Stmt::Exit(value));
        }
        if self.take(&Token::Delete) {
            let expression = self.expression()?;
            let target = expression
                .into_lvalue()
                .ok_or_else(|| "delete requires an array element".to_string())?;
            if !matches!(target, LValue::Array { .. }) {
                return Err("delete requires an array element".to_string());
            }
            return Ok(Stmt::Delete(target));
        }
        if self.take(&Token::Print) {
            return Ok(Stmt::Print(self.output_arguments()?));
        }
        if self.take(&Token::Printf) {
            let args = self.output_arguments()?;
            if args.is_empty() {
                return Err("printf requires a format expression".to_string());
            }
            return Ok(Stmt::Printf(args));
        }
        Ok(Stmt::Expr(self.expression()?))
    }

    fn for_statement(&mut self) -> Result<Stmt, String> {
        self.expect(&Token::LParen, "expected '(' after for")?;
        if let (Some(Token::Ident(name)), Some(Token::In), Some(Token::Ident(array))) = (
            self.tokens.get(self.cursor).cloned(),
            self.tokens.get(self.cursor + 1),
            self.tokens.get(self.cursor + 2).cloned(),
        ) {
            self.cursor += 3;
            self.expect(&Token::RParen, "expected ')' after for-in clause")?;
            return Ok(Stmt::ForIn {
                name,
                array,
                body: Box::new(self.statement()?),
            });
        }
        let init = if self.at(&Token::Semicolon) {
            None
        } else {
            Some(self.expression()?)
        };
        self.expect(&Token::Semicolon, "expected ';' in for clause")?;
        let condition = if self.at(&Token::Semicolon) {
            None
        } else {
            Some(self.expression()?)
        };
        self.expect(&Token::Semicolon, "expected second ';' in for clause")?;
        let update = if self.at(&Token::RParen) {
            None
        } else {
            Some(self.expression()?)
        };
        self.expect(&Token::RParen, "expected ')' after for clause")?;
        Ok(Stmt::For {
            init,
            condition,
            update,
            body: Box::new(self.statement()?),
        })
    }

    fn output_arguments(&mut self) -> Result<Vec<Expr>, String> {
        if self.statement_end() {
            return Ok(Vec::new());
        }
        let parenthesized = self.take(&Token::LParen);
        let mut args = Vec::new();
        if parenthesized && self.take(&Token::RParen) {
            return Ok(args);
        }
        loop {
            let start = self.cursor;
            args.push(self.expression()?);
            if !parenthesized && self.has_top_level_output_redirect(start, self.cursor) {
                return Err("output redirection is not supported".to_string());
            }
            if !self.take(&Token::Comma) {
                break;
            }
        }
        if parenthesized {
            self.expect(&Token::RParen, "expected ')' after output arguments")?;
        }
        Ok(args)
    }

    fn has_top_level_output_redirect(&self, start: usize, end: usize) -> bool {
        let mut depth = 0usize;
        self.tokens[start..end].iter().any(|token| match token {
            Token::LParen | Token::LBracket => {
                depth = depth.saturating_add(1);
                false
            }
            Token::RParen | Token::RBracket => {
                depth = depth.saturating_sub(1);
                false
            }
            Token::Greater if depth == 0 => true,
            _ => false,
        })
    }

    fn expression(&mut self) -> Result<Expr, String> {
        self.assignment()
    }

    fn assignment(&mut self) -> Result<Expr, String> {
        let left = self.logical_or()?;
        let op = if self.take(&Token::Assign) {
            Some(AssignOp::Set)
        } else if self.take(&Token::AddAssign) {
            Some(AssignOp::Add)
        } else if self.take(&Token::SubAssign) {
            Some(AssignOp::Subtract)
        } else if self.take(&Token::MulAssign) {
            Some(AssignOp::Multiply)
        } else if self.take(&Token::DivAssign) {
            Some(AssignOp::Divide)
        } else if self.take(&Token::RemAssign) {
            Some(AssignOp::Remainder)
        } else {
            None
        };
        let Some(op) = op else { return Ok(left) };
        let target = left
            .into_lvalue()
            .ok_or_else(|| "assignment target is not writable".to_string())?;
        Ok(Expr::Assign {
            target,
            op,
            value: Box::new(self.assignment()?),
        })
    }

    fn logical_or(&mut self) -> Result<Expr, String> {
        self.binary(Self::logical_and, &[(Token::Or, BinaryOp::Or)])
    }

    fn logical_and(&mut self) -> Result<Expr, String> {
        self.binary(Self::comparison, &[(Token::And, BinaryOp::And)])
    }

    fn comparison(&mut self) -> Result<Expr, String> {
        self.binary(
            Self::concat,
            &[
                (Token::Equal, BinaryOp::Equal),
                (Token::NotEqual, BinaryOp::NotEqual),
                (Token::Less, BinaryOp::Less),
                (Token::LessEqual, BinaryOp::LessEqual),
                (Token::Greater, BinaryOp::Greater),
                (Token::GreaterEqual, BinaryOp::GreaterEqual),
                (Token::Match, BinaryOp::Match),
                (Token::NotMatch, BinaryOp::NotMatch),
                (Token::In, BinaryOp::In),
            ],
        )
    }

    fn concat(&mut self) -> Result<Expr, String> {
        let mut left = self.additive()?;
        while self.starts_expression() {
            let right = self.additive()?;
            left = Expr::Binary {
                left: Box::new(left),
                op: BinaryOp::Concat,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn additive(&mut self) -> Result<Expr, String> {
        self.binary(
            Self::multiplicative,
            &[
                (Token::Plus, BinaryOp::Add),
                (Token::Minus, BinaryOp::Subtract),
            ],
        )
    }

    fn multiplicative(&mut self) -> Result<Expr, String> {
        self.binary(
            Self::unary,
            &[
                (Token::Star, BinaryOp::Multiply),
                (Token::Slash, BinaryOp::Divide),
                (Token::Percent, BinaryOp::Remainder),
            ],
        )
    }

    fn binary(
        &mut self,
        operand: fn(&mut Self) -> Result<Expr, String>,
        operators: &[(Token, BinaryOp)],
    ) -> Result<Expr, String> {
        let mut left = operand(self)?;
        while let Some((_, op)) = operators.iter().find(|(token, _)| self.at(token)) {
            let op = *op;
            self.cursor += 1;
            let right = operand(self)?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Expr, String> {
        if self.take(&Token::Not) {
            return Ok(Expr::Unary {
                op: UnaryOp::Not,
                value: Box::new(self.unary()?),
            });
        }
        if self.take(&Token::Plus) {
            return Ok(Expr::Unary {
                op: UnaryOp::Positive,
                value: Box::new(self.unary()?),
            });
        }
        if self.take(&Token::Minus) {
            return Ok(Expr::Unary {
                op: UnaryOp::Negative,
                value: Box::new(self.unary()?),
            });
        }
        if self.take(&Token::PlusPlus) || self.take(&Token::MinusMinus) {
            let delta = if matches!(self.tokens[self.cursor - 1], Token::PlusPlus) {
                1
            } else {
                -1
            };
            let target = self
                .postfix()?
                .into_lvalue()
                .ok_or_else(|| "increment target is not writable".to_string())?;
            return Ok(Expr::Increment {
                target,
                delta,
                prefix: true,
            });
        }
        let value = self.postfix()?;
        if self.take(&Token::PlusPlus) || self.take(&Token::MinusMinus) {
            let delta = if matches!(self.tokens[self.cursor - 1], Token::PlusPlus) {
                1
            } else {
                -1
            };
            let target = value
                .into_lvalue()
                .ok_or_else(|| "increment target is not writable".to_string())?;
            Ok(Expr::Increment {
                target,
                delta,
                prefix: false,
            })
        } else {
            Ok(value)
        }
    }

    fn postfix(&mut self) -> Result<Expr, String> {
        if self.take(&Token::Dollar) {
            return Ok(Expr::Field(Box::new(self.unary()?)));
        }
        let mut value = match self.next().clone() {
            Token::String(value) => Expr::String(value),
            Token::Number(value) => Expr::Number(value),
            Token::Regex(value) => Expr::Regex(value),
            Token::Ident(name) => Expr::Variable(name),
            Token::LParen => {
                let value = self.expression()?;
                self.expect(&Token::RParen, "expected ')' after expression")?;
                value
            }
            token => return Err(format!("expected expression, found {token:?}")),
        };
        loop {
            if self.take(&Token::LParen) {
                let Expr::Variable(name) = value else {
                    return Err("only named functions can be called".to_string());
                };
                let args = self.expression_list(&Token::RParen)?;
                value = Expr::Call { name, args };
            } else if self.take(&Token::LBracket) {
                let Expr::Variable(name) = value else {
                    return Err("only named arrays can be indexed".to_string());
                };
                let indices = self.expression_list(&Token::RBracket)?;
                if indices.is_empty() {
                    return Err("array index cannot be empty".to_string());
                }
                value = Expr::Array { name, indices };
            } else {
                break;
            }
        }
        Ok(value)
    }

    fn expression_list(&mut self, close: &Token) -> Result<Vec<Expr>, String> {
        let mut values = Vec::new();
        if self.take(close) {
            return Ok(values);
        }
        loop {
            values.push(self.expression()?);
            if !self.take(&Token::Comma) {
                break;
            }
        }
        self.expect(close, "unterminated argument or index list")?;
        Ok(values)
    }

    fn starts_expression(&self) -> bool {
        matches!(
            self.peek(),
            Token::Ident(_)
                | Token::Number(_)
                | Token::String(_)
                | Token::Regex(_)
                | Token::Dollar
                | Token::LParen
                | Token::Not
                | Token::Plus
                | Token::Minus
                | Token::PlusPlus
                | Token::MinusMinus
        )
    }

    fn statement_end(&self) -> bool {
        matches!(self.peek(), Token::Semicolon | Token::RBrace | Token::Eof)
    }

    fn separators(&mut self) {
        while self.take(&Token::Semicolon) {}
    }

    fn expect(&mut self, token: &Token, message: &str) -> Result<(), String> {
        if self.take(token) {
            Ok(())
        } else {
            Err(format!("{message}; found {:?}", self.peek()))
        }
    }

    fn take(&mut self, token: &Token) -> bool {
        if self.at(token) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }

    fn at(&self, token: &Token) -> bool {
        std::mem::discriminant(self.peek()) == std::mem::discriminant(token)
    }

    fn next(&mut self) -> &Token {
        let token = &self.tokens[self.cursor];
        self.cursor += 1;
        token
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.cursor]
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn parses_control_flow_arrays_and_functions() {
        parse(
            r#"
            BEGIN { total = 0 }
            $2 > 1 {
                if ($1 in seen) { next }
                seen[$1] = 1
                for (i = 0; i < 3; i++) total += i
                print substr($1, 2), total
            }
            END { for (key in seen) print key }
            "#,
        )
        .unwrap();
    }

    #[test]
    fn rejects_incomplete_programs() {
        assert!(parse("BEGIN { if (1) print 1").is_err());
        assert!(parse("BEGIN { x()() }").is_err());
        assert!(parse("BEGIN { print 1 > \"out\" }")
            .unwrap_err()
            .contains("redirection"));
        assert!(parse("BEGIN { print (2 > 1) }").is_ok());
    }
}
