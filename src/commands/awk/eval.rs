//! Direct, metered evaluator for the typed awk syntax tree.

use std::collections::HashMap;

use super::ast::{AssignOp, BinaryOp, Expr, LValue, Pattern, Program, Stmt, UnaryOp};
use crate::commands::util::ewln;
use crate::commands::{CommandContext, Io};

const SUPPORTED_FUNCTIONS: &[&str] = &[
    "length", "substr", "index", "split", "sub", "gsub", "match", "tolower", "toupper", "int",
    "sprintf", "log",
];

#[derive(Clone, Debug, Default)]
pub(super) struct Scalar {
    text: String,
    numeric: Option<f64>,
}

impl Scalar {
    pub fn string(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            numeric: None,
        }
    }

    fn number(value: f64) -> Self {
        Self {
            text: number_text(value),
            numeric: Some(value),
        }
    }

    fn as_number(&self) -> f64 {
        self.numeric
            .or_else(|| self.text.trim().parse().ok())
            .unwrap_or(0.0)
    }

    fn numeric_value(&self) -> Option<f64> {
        self.numeric.or_else(|| self.text.trim().parse().ok())
    }

    fn truthy(&self) -> bool {
        self.numeric_value().is_some_and(|value| value != 0.0)
            || (self.numeric_value().is_none() && !self.text.is_empty())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Begin,
    Record,
    End,
}

enum Flow {
    Normal,
    Break,
    Continue,
    Next,
    NextFile,
    Exit(i32),
    Error(String),
    Exhausted,
}

#[derive(Default)]
struct State {
    vars: HashMap<String, Scalar>,
    arrays: HashMap<String, HashMap<String, Scalar>>,
    record: String,
    fields: Vec<String>,
    nr: usize,
    fnr: usize,
    filename: String,
    fs: String,
    ofs: String,
    ors: String,
}

/// Validate static language boundaries before reading input or producing output.
pub(super) fn validate(program: &Program) -> Result<(), String> {
    for rule in &program.rules {
        if let Pattern::Expr(expression) = &rule.pattern {
            validate_expr(expression)?;
        }
        for statement in &rule.body {
            validate_stmt(statement)?;
        }
    }
    Ok(())
}

fn validate_stmt(statement: &Stmt) -> Result<(), String> {
    match statement {
        Stmt::Block(statements) => {
            for statement in statements {
                validate_stmt(statement)?;
            }
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            validate_expr(condition)?;
            validate_stmt(then_branch)?;
            if let Some(branch) = else_branch {
                validate_stmt(branch)?;
            }
        }
        Stmt::While { condition, body } => {
            validate_expr(condition)?;
            validate_stmt(body)?;
        }
        Stmt::For {
            init,
            condition,
            update,
            body,
        } => {
            for expression in [init, condition, update].into_iter().flatten() {
                validate_expr(expression)?;
            }
            validate_stmt(body)?;
        }
        Stmt::ForIn { name, body, .. } => {
            if assignment_is_unsupported(name) {
                return Err(format!("assignment to '{name}' is not supported"));
            }
            validate_stmt(body)?;
        }
        Stmt::Delete(target) => validate_lvalue(target)?,
        Stmt::Exit(value) => {
            if let Some(value) = value {
                validate_expr(value)?;
            }
        }
        Stmt::Print(values) | Stmt::Printf(values) => {
            for value in values {
                validate_expr(value)?;
            }
        }
        Stmt::Expr(value) => validate_expr(value)?,
        Stmt::Break | Stmt::Continue | Stmt::Next | Stmt::NextFile => {}
    }
    Ok(())
}

fn validate_lvalue(target: &LValue) -> Result<(), String> {
    match target {
        LValue::Variable(name) if assignment_is_unsupported(name) => {
            Err(format!("assignment to '{name}' is not supported"))
        }
        LValue::Variable(_) => Ok(()),
        LValue::Field(index) => validate_expr(index),
        LValue::Array { indices, .. } => {
            for index in indices {
                validate_expr(index)?;
            }
            Ok(())
        }
    }
}

fn validate_expr(expression: &Expr) -> Result<(), String> {
    match expression {
        Expr::Regex(pattern) => regex::Regex::new(pattern)
            .map(|_| ())
            .map_err(|error| format!("invalid regular expression: {error}")),
        Expr::Field(index) | Expr::Unary { value: index, .. } => validate_expr(index),
        Expr::Array { indices, .. } => {
            for index in indices {
                validate_expr(index)?;
            }
            Ok(())
        }
        Expr::Call { name, args } => {
            if !SUPPORTED_FUNCTIONS.contains(&name.as_str()) {
                return Err(format!("unsupported function '{name}'"));
            }
            for argument in args {
                validate_expr(argument)?;
            }
            Ok(())
        }
        Expr::Assign { target, value, .. } => {
            validate_lvalue(target)?;
            validate_expr(value)
        }
        Expr::Binary { left, right, .. } => {
            validate_expr(left)?;
            validate_expr(right)
        }
        Expr::Increment { target, .. } => validate_lvalue(target),
        Expr::String(_) | Expr::Number(_) | Expr::Variable(_) => Ok(()),
    }
}

/// Execute an already parsed program over validated UTF-8 input files.
pub(super) fn execute(
    program: &Program,
    env: &mut CommandContext<'_>,
    io: &mut Io,
    variables: HashMap<String, String>,
    fs: String,
    inputs: Vec<(String, String)>,
) -> i32 {
    let mut state = State {
        fs,
        ofs: " ".to_string(),
        ors: "\n".to_string(),
        ..State::default()
    };
    for (name, value) in variables {
        match name.as_str() {
            "FS" => state.fs = value,
            "OFS" => state.ofs = value,
            "ORS" => state.ors = value,
            _ => {
                state.vars.insert(name, Scalar::string(value));
            }
        }
    }
    let mut runtime = Runtime {
        env,
        io,
        state,
        literal_regexes: HashMap::new(),
    };

    let mut requested_exit = match runtime.phase(program, Phase::Begin) {
        Flow::Normal => None,
        Flow::Exit(status) => Some(status),
        flow => return runtime.finish_unexpected(flow, "BEGIN"),
    };

    if requested_exit.is_none() {
        'files: for (filename, data) in inputs {
            runtime.state.filename = filename;
            runtime.state.fnr = 0;
            for line in data.lines() {
                runtime.state.nr = runtime.state.nr.saturating_add(1);
                runtime.state.fnr = runtime.state.fnr.saturating_add(1);
                if let Err(flow) = runtime.set_record(line.to_string()) {
                    return runtime.finish_unexpected(flow, "field splitting");
                }
                match runtime.phase(program, Phase::Record) {
                    Flow::Normal | Flow::Next => {}
                    Flow::NextFile => continue 'files,
                    Flow::Exit(status) => {
                        requested_exit = Some(status);
                        break 'files;
                    }
                    flow => return runtime.finish_unexpected(flow, "record action"),
                }
            }
        }
    }

    match runtime.phase(program, Phase::End) {
        Flow::Normal => requested_exit.unwrap_or(0),
        Flow::Exit(status) => status,
        flow => runtime.finish_unexpected(flow, "END"),
    }
}

struct Runtime<'a, 'env, 'io> {
    env: &'a mut CommandContext<'env>,
    io: &'a mut Io<'io>,
    state: State,
    literal_regexes: HashMap<String, regex::Regex>,
}

impl Runtime<'_, '_, '_> {
    fn phase(&mut self, program: &Program, phase: Phase) -> Flow {
        for rule in &program.rules {
            let selected = match (&rule.pattern, phase) {
                (Pattern::Begin, Phase::Begin) | (Pattern::End, Phase::End) => true,
                (Pattern::Always, Phase::Record) => true,
                (Pattern::Expr(expression), Phase::Record) => match self.expr(expression) {
                    Ok(value) => value.truthy(),
                    Err(flow) => return flow,
                },
                _ => false,
            };
            if !selected {
                continue;
            }
            match self.statements(&rule.body) {
                Flow::Normal => {}
                flow => return flow,
            }
        }
        Flow::Normal
    }

    fn statements(&mut self, statements: &[Stmt]) -> Flow {
        for statement in statements {
            let flow = self.statement(statement);
            if !matches!(flow, Flow::Normal) {
                return flow;
            }
        }
        Flow::Normal
    }

    fn statement(&mut self, statement: &Stmt) -> Flow {
        if !self.env.charge_cpu(1) {
            return Flow::Exhausted;
        }
        match statement {
            Stmt::Block(statements) => self.statements(statements),
            Stmt::If {
                condition,
                then_branch,
                else_branch,
            } => match self.expr(condition) {
                Ok(value) if value.truthy() => self.statement(then_branch),
                Ok(_) => else_branch
                    .as_deref()
                    .map_or(Flow::Normal, |branch| self.statement(branch)),
                Err(flow) => flow,
            },
            Stmt::While { condition, body } => loop {
                match self.expr(condition) {
                    Ok(value) if !value.truthy() => return Flow::Normal,
                    Ok(_) => {}
                    Err(flow) => return flow,
                }
                match self.statement(body) {
                    Flow::Normal | Flow::Continue => {}
                    Flow::Break => return Flow::Normal,
                    flow => return flow,
                }
            },
            Stmt::For {
                init,
                condition,
                update,
                body,
            } => {
                if let Some(init) = init {
                    if let Err(flow) = self.expr(init) {
                        return flow;
                    }
                }
                loop {
                    if let Some(condition) = condition {
                        match self.expr(condition) {
                            Ok(value) if !value.truthy() => return Flow::Normal,
                            Ok(_) => {}
                            Err(flow) => return flow,
                        }
                    }
                    match self.statement(body) {
                        Flow::Normal | Flow::Continue => {}
                        Flow::Break => return Flow::Normal,
                        flow => return flow,
                    }
                    if let Some(update) = update {
                        if let Err(flow) = self.expr(update) {
                            return flow;
                        }
                    }
                }
            }
            Stmt::ForIn { name, array, body } => {
                let mut keys = self
                    .state
                    .arrays
                    .get(array)
                    .map(|values| values.keys().cloned().collect::<Vec<_>>())
                    .unwrap_or_default();
                keys.sort();
                for key in keys {
                    if let Err(flow) = self.set_variable(name, Scalar::string(key)) {
                        return flow;
                    }
                    match self.statement(body) {
                        Flow::Normal | Flow::Continue => {}
                        Flow::Break => return Flow::Normal,
                        flow => return flow,
                    }
                }
                Flow::Normal
            }
            Stmt::Break => Flow::Break,
            Stmt::Continue => Flow::Continue,
            Stmt::Delete(target) => match self.array_target(target) {
                Ok((name, key)) => {
                    if let Some(array) = self.state.arrays.get_mut(&name) {
                        array.remove(&key);
                    }
                    Flow::Normal
                }
                Err(flow) => flow,
            },
            Stmt::Next => Flow::Next,
            Stmt::NextFile => Flow::NextFile,
            Stmt::Exit(value) => match value {
                Some(value) => match self.expr(value) {
                    Ok(value) => Flow::Exit(value.as_number() as i32),
                    Err(flow) => flow,
                },
                None => Flow::Exit(0),
            },
            Stmt::Print(values) => {
                let result = if values.is_empty() {
                    self.state.record.clone()
                } else {
                    let mut rendered = Vec::with_capacity(values.len());
                    for value in values {
                        match self.expr(value) {
                            Ok(value) => rendered.push(value.text),
                            Err(flow) => return flow,
                        }
                    }
                    rendered.join(&self.state.ofs)
                };
                self.io.out.extend_from_slice(result.as_bytes());
                self.io.out.extend_from_slice(self.state.ors.as_bytes());
                Flow::Normal
            }
            Stmt::Printf(values) => {
                let mut evaluated = Vec::with_capacity(values.len());
                for value in values {
                    match self.expr(value) {
                        Ok(value) => evaluated.push(value),
                        Err(flow) => return flow,
                    }
                }
                let limit = usize::try_from(
                    self.env
                        .resources
                        .output_remaining()
                        .min(self.env.resources.limits().memory),
                )
                .unwrap_or(usize::MAX);
                match format_printf(&evaluated[0].text, &evaluated[1..], limit) {
                    Ok(output) => {
                        self.io.out.extend_from_slice(output.as_bytes());
                        Flow::Normal
                    }
                    Err(FormatError::Invalid(error)) => Flow::Error(error),
                    Err(FormatError::Limit) => {
                        let remaining = self.env.resources.output_remaining();
                        let _ = self
                            .env
                            .resources
                            .charge_output(remaining.saturating_add(1));
                        Flow::Exhausted
                    }
                }
            }
            Stmt::Expr(value) => self.expr(value).map_or_else(|flow| flow, |_| Flow::Normal),
        }
    }

    fn expr(&mut self, expression: &Expr) -> Result<Scalar, Flow> {
        if !self.env.charge_cpu(1) {
            return Err(Flow::Exhausted);
        }
        match expression {
            Expr::String(value) => Ok(Scalar::string(value.clone())),
            Expr::Number(value) => Ok(Scalar::number(*value)),
            Expr::Regex(pattern) => {
                if !self.literal_regexes.contains_key(pattern) {
                    if !self.env.charge_cpu(pattern.len() as u64) {
                        return Err(Flow::Exhausted);
                    }
                    let regex = regex::Regex::new(pattern).map_err(|error| {
                        Flow::Error(format!("invalid regular expression: {error}"))
                    })?;
                    self.literal_regexes.insert(pattern.clone(), regex);
                }
                let matched = self
                    .literal_regexes
                    .get(pattern)
                    .expect("literal regex was inserted")
                    .is_match(&self.state.record);
                Ok(Scalar::number(matched as u8 as f64))
            }
            Expr::Variable(name) => Ok(self.variable(name)),
            Expr::Field(index) => {
                let index = self.expr(index)?.as_number().max(0.0) as usize;
                Ok(self.field(index))
            }
            Expr::Array { name, indices } => {
                let key = self.array_key(indices)?;
                Ok(self
                    .state
                    .arrays
                    .get(name)
                    .and_then(|array| array.get(&key))
                    .cloned()
                    .unwrap_or_default())
            }
            Expr::Call { name, args } => self.call(name, args),
            Expr::Assign { target, op, value } => {
                let right = self.expr(value)?;
                let value = if matches!(op, AssignOp::Set) {
                    right
                } else {
                    let left = self.read_lvalue(target)?;
                    let operand = right.as_number();
                    let result = match op {
                        AssignOp::Add => left.as_number() + operand,
                        AssignOp::Subtract => left.as_number() - operand,
                        AssignOp::Multiply => left.as_number() * operand,
                        AssignOp::Divide if operand != 0.0 => left.as_number() / operand,
                        AssignOp::Remainder if operand != 0.0 => left.as_number() % operand,
                        AssignOp::Divide | AssignOp::Remainder => {
                            return Err(Flow::Error("division by zero".to_string()))
                        }
                        AssignOp::Set => unreachable!(),
                    };
                    Scalar::number(result)
                };
                self.write_lvalue(target, value.clone())?;
                Ok(value)
            }
            Expr::Binary { left, op, right } => self.binary(left, *op, right),
            Expr::Unary { op, value } => {
                let value = self.expr(value)?;
                Ok(match op {
                    UnaryOp::Not => Scalar::number((!value.truthy()) as u8 as f64),
                    UnaryOp::Positive => Scalar::number(value.as_number()),
                    UnaryOp::Negative => Scalar::number(-value.as_number()),
                })
            }
            Expr::Increment {
                target,
                delta,
                prefix,
            } => {
                let old = self.read_lvalue(target)?;
                let new = Scalar::number(old.as_number() + f64::from(*delta));
                self.write_lvalue(target, new.clone())?;
                Ok(if *prefix { new } else { old })
            }
        }
    }

    fn binary(&mut self, left: &Expr, op: BinaryOp, right: &Expr) -> Result<Scalar, Flow> {
        if matches!(op, BinaryOp::Or) {
            let left = self.expr(left)?;
            return if left.truthy() {
                Ok(Scalar::number(1.0))
            } else {
                Ok(Scalar::number(self.expr(right)?.truthy() as u8 as f64))
            };
        }
        if matches!(op, BinaryOp::And) {
            let left = self.expr(left)?;
            return if !left.truthy() {
                Ok(Scalar::number(0.0))
            } else {
                Ok(Scalar::number(self.expr(right)?.truthy() as u8 as f64))
            };
        }
        if matches!(op, BinaryOp::In) {
            let key = self.expr(left)?.text;
            let Expr::Variable(array) = right else {
                return Err(Flow::Error(
                    "right side of 'in' must be an array name".to_string(),
                ));
            };
            return Ok(Scalar::number(
                self.state
                    .arrays
                    .get(array)
                    .is_some_and(|values| values.contains_key(&key)) as u8 as f64,
            ));
        }
        if matches!(op, BinaryOp::Match | BinaryOp::NotMatch) {
            let left = self.expr(left)?;
            let pattern = self.regex_source(right)?;
            let matched = regex::Regex::new(&pattern)
                .map_err(|error| Flow::Error(format!("invalid regular expression: {error}")))?
                .is_match(&left.text);
            return Ok(Scalar::number(
                (if matches!(op, BinaryOp::Match) {
                    matched
                } else {
                    !matched
                }) as u8 as f64,
            ));
        }
        let left = self.expr(left)?;
        let right = self.expr(right)?;
        let boolean = |value: bool| Scalar::number(value as u8 as f64);
        Ok(match op {
            BinaryOp::Equal | BinaryOp::NotEqual => {
                let equal = match (left.numeric_value(), right.numeric_value()) {
                    (Some(left), Some(right)) => left == right,
                    _ => left.text == right.text,
                };
                boolean(if matches!(op, BinaryOp::Equal) {
                    equal
                } else {
                    !equal
                })
            }
            BinaryOp::Less | BinaryOp::LessEqual | BinaryOp::Greater | BinaryOp::GreaterEqual => {
                let ordering = match (left.numeric_value(), right.numeric_value()) {
                    (Some(left), Some(right)) => left.partial_cmp(&right),
                    _ => Some(left.text.cmp(&right.text)),
                };
                boolean(match op {
                    BinaryOp::Less => ordering.is_some_and(|value| value.is_lt()),
                    BinaryOp::LessEqual => ordering.is_some_and(|value| value.is_le()),
                    BinaryOp::Greater => ordering.is_some_and(|value| value.is_gt()),
                    BinaryOp::GreaterEqual => ordering.is_some_and(|value| value.is_ge()),
                    _ => unreachable!(),
                })
            }
            BinaryOp::Concat => return self.concatenate(&left.text, &right.text),
            BinaryOp::Add => Scalar::number(left.as_number() + right.as_number()),
            BinaryOp::Subtract => Scalar::number(left.as_number() - right.as_number()),
            BinaryOp::Multiply => Scalar::number(left.as_number() * right.as_number()),
            BinaryOp::Divide if right.as_number() != 0.0 => {
                Scalar::number(left.as_number() / right.as_number())
            }
            BinaryOp::Remainder if right.as_number() != 0.0 => {
                Scalar::number(left.as_number() % right.as_number())
            }
            BinaryOp::Divide | BinaryOp::Remainder => {
                return Err(Flow::Error("division by zero".to_string()))
            }
            BinaryOp::Or | BinaryOp::And | BinaryOp::Match | BinaryOp::NotMatch | BinaryOp::In => {
                unreachable!()
            }
        })
    }

    fn concatenate(&mut self, left: &str, right: &str) -> Result<Scalar, Flow> {
        let bytes = (left.len() as u64).saturating_add(right.len() as u64);
        if !self.env.reserve_memory(bytes) {
            return Err(Flow::Exhausted);
        }
        if !self.env.charge_cpu(bytes) {
            self.env.resources.release_memory(bytes);
            return Err(Flow::Exhausted);
        }
        let mut output = String::with_capacity(bytes as usize);
        output.push_str(left);
        output.push_str(right);
        self.env.resources.release_memory(bytes);
        Ok(Scalar::string(output))
    }

    fn call(&mut self, name: &str, args: &[Expr]) -> Result<Scalar, Flow> {
        match name {
            "length" => {
                self.arity(name, args, 0, 1)?;
                let value = if let Some(value) = args.first() {
                    self.expr(value)?.text
                } else {
                    self.state.record.clone()
                };
                Ok(Scalar::number(value.chars().count() as f64))
            }
            "substr" => {
                self.arity(name, args, 2, 3)?;
                let value = self.expr(&args[0])?.text;
                let start = self.expr(&args[1])?.as_number().max(1.0) as usize - 1;
                let count = args
                    .get(2)
                    .map(|value| self.expr(value).map(|v| v.as_number().max(0.0) as usize))
                    .transpose()?;
                let text = value.chars().skip(start);
                Ok(Scalar::string(match count {
                    Some(count) => text.take(count).collect::<String>(),
                    None => text.collect::<String>(),
                }))
            }
            "index" => {
                self.arity(name, args, 2, 2)?;
                let haystack = self.expr(&args[0])?.text;
                let needle = self.expr(&args[1])?.text;
                let index = haystack
                    .find(&needle)
                    .map(|byte| haystack[..byte].chars().count() + 1)
                    .unwrap_or(0);
                Ok(Scalar::number(index as f64))
            }
            "tolower" | "toupper" | "int" | "log" => {
                self.arity(name, args, 1, 1)?;
                let value = self.expr(&args[0])?;
                Ok(match name {
                    "tolower" => Scalar::string(value.text.to_lowercase()),
                    "toupper" => Scalar::string(value.text.to_uppercase()),
                    "int" => Scalar::number(value.as_number().trunc()),
                    "log" => Scalar::number(value.as_number().ln()),
                    _ => unreachable!(),
                })
            }
            "split" => self.split(args),
            "sub" => self.substitute(args, false),
            "gsub" => self.substitute(args, true),
            "match" => self.match_function(args),
            "sprintf" => {
                self.arity(name, args, 1, usize::MAX)?;
                let mut values = Vec::with_capacity(args.len());
                for argument in args {
                    values.push(self.expr(argument)?);
                }
                let limit =
                    usize::try_from(self.env.resources.limits().memory).unwrap_or(usize::MAX);
                match format_printf(&values[0].text, &values[1..], limit) {
                    Ok(output) => {
                        let bytes = output.len() as u64;
                        if !self.env.reserve_memory(bytes) {
                            return Err(Flow::Exhausted);
                        }
                        self.env.resources.release_memory(bytes);
                        Ok(Scalar::string(output))
                    }
                    Err(FormatError::Invalid(error)) => Err(Flow::Error(error)),
                    Err(FormatError::Limit) => {
                        let request = self.env.resources.limits().memory.saturating_add(1);
                        let _ = self.env.reserve_memory(request);
                        Err(Flow::Exhausted)
                    }
                }
            }
            _ => Err(Flow::Error(format!("unsupported function '{name}'"))),
        }
    }

    fn split(&mut self, args: &[Expr]) -> Result<Scalar, Flow> {
        self.arity("split", args, 2, 3)?;
        let input = self.expr(&args[0])?.text;
        let Expr::Variable(array_name) = &args[1] else {
            return Err(Flow::Error(
                "split second argument must be an array name".to_string(),
            ));
        };
        let separator = if let Some(separator) = args.get(2) {
            self.regex_source(separator)?
        } else {
            self.state.fs.clone()
        };
        let values = split_fields(&input, &separator)?;
        let array = self.state.arrays.entry(array_name.clone()).or_default();
        array.clear();
        for (index, value) in values.iter().enumerate() {
            array.insert((index + 1).to_string(), Scalar::string(value.clone()));
        }
        Ok(Scalar::number(values.len() as f64))
    }

    fn substitute(&mut self, args: &[Expr], global: bool) -> Result<Scalar, Flow> {
        let name = if global { "gsub" } else { "sub" };
        self.arity(name, args, 2, 3)?;
        let pattern = self.regex_source(&args[0])?;
        let regex = regex::Regex::new(&pattern)
            .map_err(|error| Flow::Error(format!("invalid regular expression: {error}")))?;
        let replacement = awk_replacement(&self.expr(&args[1])?.text);
        let original = if let Some(target) = args.get(2) {
            let target = target
                .clone()
                .into_lvalue()
                .ok_or_else(|| Flow::Error(format!("{name} target is not writable")))?;
            let value = self.read_lvalue(&target)?.text;
            (Some(target), value)
        } else {
            (None, self.state.record.clone())
        };
        let count = if global {
            regex.find_iter(&original.1).count()
        } else {
            usize::from(regex.is_match(&original.1))
        };
        let replaced = if global {
            regex
                .replace_all(&original.1, replacement.as_str())
                .into_owned()
        } else {
            regex
                .replace(&original.1, replacement.as_str())
                .into_owned()
        };
        if let Some(target) = original.0 {
            self.write_lvalue(&target, Scalar::string(replaced))?;
        } else {
            self.set_record(replaced)?;
        }
        Ok(Scalar::number(count as f64))
    }

    fn match_function(&mut self, args: &[Expr]) -> Result<Scalar, Flow> {
        self.arity("match", args, 2, 2)?;
        let value = self.expr(&args[0])?.text;
        let pattern = self.regex_source(&args[1])?;
        let regex = regex::Regex::new(&pattern)
            .map_err(|error| Flow::Error(format!("invalid regular expression: {error}")))?;
        let found = regex.find(&value);
        let start = found.map_or(0, |matched| value[..matched.start()].chars().count() + 1);
        let length = found.map_or(-1, |matched| matched.as_str().chars().count() as i64);
        self.set_variable("RSTART", Scalar::number(start as f64))?;
        self.set_variable("RLENGTH", Scalar::number(length as f64))?;
        Ok(Scalar::number(start as f64))
    }

    fn regex_source(&mut self, expression: &Expr) -> Result<String, Flow> {
        match expression {
            Expr::Regex(pattern) => Ok(pattern.clone()),
            _ => Ok(self.expr(expression)?.text),
        }
    }

    fn arity(&self, name: &str, args: &[Expr], min: usize, max: usize) -> Result<(), Flow> {
        if args.len() < min || args.len() > max {
            let expected = if min == max {
                min.to_string()
            } else if max == usize::MAX {
                format!("at least {min}")
            } else {
                format!("{min} to {max}")
            };
            return Err(Flow::Error(format!(
                "{name} expects {expected} arguments, got {}",
                args.len()
            )));
        }
        Ok(())
    }

    fn variable(&self, name: &str) -> Scalar {
        match name {
            "NR" => Scalar::number(self.state.nr as f64),
            "FNR" => Scalar::number(self.state.fnr as f64),
            "NF" => Scalar::number(self.state.fields.len() as f64),
            "FILENAME" => Scalar::string(self.state.filename.clone()),
            "FS" => Scalar::string(self.state.fs.clone()),
            "OFS" => Scalar::string(self.state.ofs.clone()),
            "ORS" => Scalar::string(self.state.ors.clone()),
            "RS" => Scalar::string("\n"),
            _ => self.state.vars.get(name).cloned().unwrap_or_default(),
        }
    }

    fn set_variable(&mut self, name: &str, value: Scalar) -> Result<(), Flow> {
        match name {
            "FS" => self.state.fs = value.text,
            "OFS" => self.state.ofs = value.text,
            "ORS" => self.state.ors = value.text,
            name if assignment_is_unsupported(name) => {
                return Err(Flow::Error(format!(
                    "assignment to '{name}' is not supported"
                )))
            }
            _ => {
                self.state.vars.insert(name.to_string(), value);
            }
        }
        Ok(())
    }

    fn field(&self, index: usize) -> Scalar {
        if index == 0 {
            Scalar::string(self.state.record.clone())
        } else {
            self.state
                .fields
                .get(index - 1)
                .cloned()
                .map(Scalar::string)
                .unwrap_or_default()
        }
    }

    fn set_record(&mut self, record: String) -> Result<(), Flow> {
        self.state.fields = split_fields(&record, &self.state.fs)?;
        self.state.record = record;
        Ok(())
    }

    fn read_lvalue(&mut self, target: &LValue) -> Result<Scalar, Flow> {
        match target {
            LValue::Variable(name) => Ok(self.variable(name)),
            LValue::Field(index) => {
                let index = self.expr(index)?.as_number().max(0.0) as usize;
                Ok(self.field(index))
            }
            LValue::Array { name, indices } => {
                let key = self.array_key(indices)?;
                Ok(self
                    .state
                    .arrays
                    .get(name)
                    .and_then(|array| array.get(&key))
                    .cloned()
                    .unwrap_or_default())
            }
        }
    }

    fn write_lvalue(&mut self, target: &LValue, value: Scalar) -> Result<(), Flow> {
        match target {
            LValue::Variable(name) => self.set_variable(name, value)?,
            LValue::Field(index) => {
                let index = self.expr(index)?.as_number().max(0.0) as usize;
                if index == 0 {
                    self.set_record(value.text)?;
                } else {
                    let missing = index.saturating_sub(self.state.fields.len());
                    let vector_bytes =
                        (missing as u64).saturating_mul(std::mem::size_of::<String>() as u64);
                    let field_bytes = self
                        .state
                        .fields
                        .iter()
                        .enumerate()
                        .filter(|(position, _)| *position != index - 1)
                        .map(|(_, field)| field.len() as u64)
                        .fold(value.text.len() as u64, u64::saturating_add);
                    let joined_bytes = field_bytes.saturating_add(
                        ((index - 1) as u64).saturating_mul(self.state.ofs.len() as u64),
                    );
                    let scratch = vector_bytes.saturating_add(joined_bytes);
                    if !self.env.reserve_memory(scratch) {
                        return Err(Flow::Exhausted);
                    }
                    if !self.env.charge_cpu(scratch) {
                        self.env.resources.release_memory(scratch);
                        return Err(Flow::Exhausted);
                    }
                    if self.state.fields.len() < index {
                        self.state.fields.resize(index, String::new());
                    }
                    self.state.fields[index - 1] = value.text;
                    self.state.record = self.state.fields.join(&self.state.ofs);
                    self.env.resources.release_memory(scratch);
                }
            }
            LValue::Array { name, indices } => {
                let key = self.array_key(indices)?;
                self.state
                    .arrays
                    .entry(name.clone())
                    .or_default()
                    .insert(key, value);
            }
        }
        Ok(())
    }

    fn array_key(&mut self, indices: &[Expr]) -> Result<String, Flow> {
        let mut values = Vec::with_capacity(indices.len());
        for index in indices {
            values.push(self.expr(index)?.text);
        }
        Ok(values.join("\u{001c}"))
    }

    fn array_target(&mut self, target: &LValue) -> Result<(String, String), Flow> {
        let LValue::Array { name, indices } = target else {
            return Err(Flow::Error("expected array element".to_string()));
        };
        Ok((name.clone(), self.array_key(indices)?))
    }

    fn finish_unexpected(&mut self, flow: Flow, context: &str) -> i32 {
        match flow {
            Flow::Error(error) => {
                self.env.note_unsupported(&format!("awk:{error}"));
                ewln(self.io.err, &format!("awk: {error}"));
                2
            }
            Flow::Exhausted => 137,
            Flow::Break | Flow::Continue => {
                ewln(
                    self.io.err,
                    &format!("awk: loop control outside loop in {context}"),
                );
                2
            }
            Flow::Next | Flow::NextFile => {
                ewln(self.io.err, &format!("awk: record control in {context}"));
                2
            }
            Flow::Exit(status) => status,
            Flow::Normal => 0,
        }
    }
}

pub(super) fn assignment_is_unsupported(name: &str) -> bool {
    matches!(name, "NR" | "FNR" | "NF" | "FILENAME" | "RS")
}

fn split_fields(record: &str, separator: &str) -> Result<Vec<String>, Flow> {
    if separator == " " || separator.is_empty() {
        return Ok(record.split_whitespace().map(str::to_string).collect());
    }
    regex::Regex::new(separator)
        .map(|regex| regex.split(record).map(str::to_string).collect())
        .map_err(|error| Flow::Error(format!("invalid field separator: {error}")))
}

/// Check an initial field separator before input is read or output is produced.
pub(super) fn validate_field_separator(separator: &str) -> Result<(), String> {
    split_fields("", separator)
        .map(|_| ())
        .map_err(|flow| match flow {
            Flow::Error(error) => error,
            _ => "invalid field separator".to_string(),
        })
}

fn awk_replacement(value: &str) -> String {
    let mut output = String::new();
    let mut escaped = false;
    for character in value.chars() {
        if escaped {
            output.push(character);
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character == '&' {
            output.push_str("${0}");
        } else if character == '$' {
            output.push_str("$$");
        } else {
            output.push(character);
        }
    }
    if escaped {
        output.push('\\');
    }
    output
}

enum FormatError {
    Invalid(String),
    Limit,
}

fn format_printf(
    format: &str,
    values: &[Scalar],
    maximum_bytes: usize,
) -> Result<String, FormatError> {
    let mut output = String::new();
    let mut chars = format.chars().peekable();
    let mut value_index = 0usize;
    while let Some(character) = chars.next() {
        if character != '%' {
            if output
                .len()
                .checked_add(character.len_utf8())
                .is_none_or(|length| length > maximum_bytes)
            {
                return Err(FormatError::Limit);
            }
            output.push(character);
            continue;
        }
        if chars.peek() == Some(&'%') {
            chars.next();
            if output.len() >= maximum_bytes {
                return Err(FormatError::Limit);
            }
            output.push('%');
            continue;
        }
        let mut left = false;
        let mut zero = false;
        while let Some(flag) = chars.peek().copied() {
            match flag {
                '-' => left = true,
                '0' => zero = true,
                '+' | ' ' => {}
                '#' => {
                    return Err(FormatError::Invalid(
                        "unsupported printf flag '#'".to_string(),
                    ))
                }
                _ => break,
            }
            chars.next();
        }
        let width = take_digits(&mut chars)
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| FormatError::Invalid("printf width is too large".to_string()))
            })
            .transpose()?;
        let precision = if chars.peek() == Some(&'.') {
            chars.next();
            Some(
                take_digits(&mut chars)
                    .unwrap_or_default()
                    .parse::<usize>()
                    .map_err(|_| {
                        FormatError::Invalid("printf precision is too large".to_string())
                    })?,
            )
        } else {
            None
        };
        if width.is_some_and(|value| value > maximum_bytes)
            || precision.is_some_and(|value| value > maximum_bytes)
        {
            return Err(FormatError::Limit);
        }
        let conversion = chars
            .next()
            .ok_or_else(|| FormatError::Invalid("unterminated printf conversion".to_string()))?;
        if !matches!(conversion, 's' | 'd' | 'i' | 'f' | 'g' | 'e' | 'c') {
            return Err(FormatError::Invalid(format!(
                "unsupported printf conversion '%{conversion}'"
            )));
        }
        let value = values.get(value_index).cloned().unwrap_or_default();
        value_index += 1;
        if precision
            .is_some_and(|limit| value.text.len().min(limit.saturating_mul(4)) > maximum_bytes)
        {
            return Err(FormatError::Limit);
        }
        let mut rendered = match conversion {
            's' if precision.is_none() && value.text.len() > maximum_bytes => {
                return Err(FormatError::Limit)
            }
            's' => precision.map_or(value.text.clone(), |limit| {
                value.text.chars().take(limit).collect()
            }),
            'd' | 'i' => (value.as_number() as i64).to_string(),
            'f' => format!("{:.*}", precision.unwrap_or(6), value.as_number()),
            'e' => format!("{:.*e}", precision.unwrap_or(6), value.as_number()),
            'g' => number_text(value.as_number()),
            'c' => value.text.chars().next().unwrap_or('\0').to_string(),
            _ => unreachable!(),
        };
        if rendered.len() > maximum_bytes {
            return Err(FormatError::Limit);
        }
        if let Some(width) = width.filter(|width| *width > rendered.chars().count()) {
            let padding = width - rendered.chars().count();
            let fill = if zero && !left { '0' } else { ' ' };
            if left {
                rendered.extend(std::iter::repeat_n(fill, padding));
            } else {
                rendered = format!("{}{}", fill.to_string().repeat(padding), rendered);
            }
        }
        if output
            .len()
            .checked_add(rendered.len())
            .is_none_or(|length| length > maximum_bytes)
        {
            return Err(FormatError::Limit);
        }
        output.push_str(&rendered);
    }
    Ok(output)
}

fn take_digits(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Option<String> {
    let mut digits = String::new();
    while chars.peek().is_some_and(char::is_ascii_digit) {
        digits.push(chars.next().expect("peeked digit"));
    }
    (!digits.is_empty()).then_some(digits)
}

fn number_text(value: f64) -> String {
    if value.fract() == 0.0 && value.is_finite() {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}
