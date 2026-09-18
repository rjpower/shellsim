//! Compiler from the owned AST to typed stack-machine operations.

use super::ast::{
    AssignmentTarget, BooleanOperator, ComprehensionClause, Constant, DictEntry, Expression,
    ExpressionKind, FStringPart, Program, Statement, StatementKind,
};
use super::bytecode::{ClassField, Code, Instruction, Operation};
use super::source::Span;
use std::collections::HashSet;

pub fn compile(program: Program) -> Code {
    let mut compiler = Compiler {
        instructions: Vec::new(),
        loops: Vec::new(),
        finalizers: Vec::new(),
        protected_regions: Vec::new(),
        in_function: false,
        globals: HashSet::new(),
        nonlocals: HashSet::new(),
        structural_depth: 0,
    };
    compiler.statements(program.statements);
    compiler.emit(Operation::Halt, Span::default());
    Code {
        instructions: compiler.instructions,
        parameters: Vec::new(),
    }
}

struct Compiler {
    instructions: Vec<Instruction>,
    loops: Vec<LoopContext>,
    /// Lexically active ``finally`` bodies.  Abrupt control flow is lowered by
    /// running these bodies before the control-flow operation, which keeps the
    /// VM's instruction stream simple and makes cleanup ordering explicit.
    finalizers: Vec<Cleanup>,
    /// A yield cannot safely suspend while the VM has a protected handler or an
    /// entered context manager that must be unwound.  Such yields are rejected
    /// at compile time rather than losing that state in a suspended generator.
    protected_regions: Vec<&'static str>,
    in_function: bool,
    globals: HashSet<String>,
    nonlocals: HashSet<String>,
    structural_depth: usize,
}

struct LoopContext {
    continue_target: usize,
    iterator_on_stack: bool,
    finalizer_depth: usize,
    breaks: Vec<usize>,
}

#[derive(Clone)]
enum Cleanup {
    Finally(Vec<Statement>),
    WithExit,
}

fn contains_star(target: &AssignmentTarget) -> bool {
    match target {
        AssignmentTarget::Star(_) => true,
        AssignmentTarget::Sequence(targets) => targets.iter().any(contains_star),
        AssignmentTarget::Name(_)
        | AssignmentTarget::Attribute { .. }
        | AssignmentTarget::Subscript { .. } => false,
    }
}

impl Compiler {
    fn statements(&mut self, statements: Vec<Statement>) {
        for statement in statements {
            self.statement(statement);
        }
    }

    fn statement(&mut self, statement: Statement) {
        const MAX_STRUCTURAL_DEPTH: usize = 256;
        if self.structural_depth == MAX_STRUCTURAL_DEPTH {
            self.emit(
                Operation::RuntimeError("compiler structural nesting limit exceeded".into()),
                statement.span,
            );
            return;
        }
        self.structural_depth += 1;
        self.statement_inner(statement);
        self.structural_depth -= 1;
    }

    fn statement_inner(&mut self, statement: Statement) {
        let span = statement.span;
        match statement.kind {
            StatementKind::Import { module, binding } => {
                let bind_root = module.split('.').next().is_some_and(|root| root == binding);
                self.emit(
                    Operation::Import {
                        name: module,
                        bind_root,
                    },
                    span,
                );
                self.emit(Operation::StoreName(binding), span);
            }
            StatementKind::ImportFrom { module, names: _ } if module == "__future__" => {}
            StatementKind::ImportFrom { module, names } => {
                self.emit(
                    Operation::Import {
                        name: module,
                        bind_root: false,
                    },
                    span,
                );
                for (imported, binding) in names {
                    self.emit(Operation::Copy(1), span);
                    self.emit(Operation::LoadAttribute(imported), span);
                    self.emit(Operation::StoreName(binding), span);
                }
                self.emit(Operation::PopTop, span);
            }
            StatementKind::Assign { targets, value } => {
                self.expression(value);
                let last = targets.len().saturating_sub(1);
                for (index, target) in targets.into_iter().enumerate() {
                    if index != last {
                        self.emit(Operation::Copy(1), span);
                    }
                    self.store_target(target, span);
                }
            }
            StatementKind::AnnotatedAssign { target, value } => {
                if let Some(value) = value {
                    self.expression(value);
                    self.store_target(target, span);
                }
            }
            StatementKind::AugmentedAssign {
                target,
                operator,
                value,
            } => {
                self.augmented_assignment(target, operator, value, span);
            }
            StatementKind::Delete(target) => match target {
                AssignmentTarget::Name(name) => {
                    if self.globals.contains(&name) {
                        self.emit(Operation::DeleteGlobal(name), span);
                    } else {
                        self.emit(Operation::DeleteName(name), span);
                    }
                }
                AssignmentTarget::Subscript { value, index } => {
                    self.expression(value);
                    self.expression(index);
                    self.emit(Operation::DeleteSubscript, span);
                }
                AssignmentTarget::Attribute { .. }
                | AssignmentTarget::Sequence(_)
                | AssignmentTarget::Star(_) => {
                    self.emit(
                        Operation::RuntimeError(
                            "this deletion target is not implemented; names and subscripts are supported"
                                .into(),
                        ),
                        span,
                    );
                }
            },
            StatementKind::Expression(expression) => {
                self.expression(expression);
                self.emit(Operation::PopExpression, span);
            }
            StatementKind::If {
                test,
                body,
                otherwise,
            } => {
                self.expression(test);
                let otherwise_jump = self.emit(Operation::PopJumpIfFalse(usize::MAX), span);
                self.statements(body);
                let finished_jump = self.emit(Operation::Jump(usize::MAX), span);
                self.patch_jump(otherwise_jump, self.instructions.len());
                self.statements(otherwise);
                self.patch_jump(finished_jump, self.instructions.len());
            }
            StatementKind::While {
                test,
                body,
                otherwise,
            } => {
                let condition = self.instructions.len();
                self.expression(test);
                let exhausted = self.emit(Operation::PopJumpIfFalse(usize::MAX), span);
                self.loops.push(LoopContext {
                    continue_target: condition,
                    iterator_on_stack: false,
                    finalizer_depth: self.finalizers.len(),
                    breaks: Vec::new(),
                });
                self.statements(body);
                self.emit(Operation::Jump(condition), span);
                let otherwise_start = self.instructions.len();
                self.patch_jump(exhausted, otherwise_start);
                self.statements(otherwise);
                let finished = self.instructions.len();
                self.finish_loop(finished);
            }
            StatementKind::For {
                target,
                iterable,
                body,
                otherwise,
            } => {
                self.expression(iterable);
                self.emit(Operation::GetIterator, span);
                let next = self.instructions.len();
                let exhausted = self.emit(Operation::ForIterator(usize::MAX), span);
                self.store_target(target, span);
                self.loops.push(LoopContext {
                    continue_target: next,
                    iterator_on_stack: true,
                    finalizer_depth: self.finalizers.len(),
                    breaks: Vec::new(),
                });
                self.statements(body);
                self.emit(Operation::Jump(next), span);
                let otherwise_start = self.instructions.len();
                self.patch_jump(exhausted, otherwise_start);
                self.statements(otherwise);
                let finished = self.instructions.len();
                self.finish_loop(finished);
            }
            StatementKind::Function {
                name,
                parameters,
                body,
            } => {
                let defaults = parameters
                    .iter()
                    .filter_map(|parameter| parameter.default.clone())
                    .collect::<Vec<_>>();
                for default in &defaults {
                    self.expression(default.clone());
                }
                let mut nested = Compiler {
                    instructions: Vec::new(),
                    loops: Vec::new(),
                    finalizers: Vec::new(),
                    protected_regions: Vec::new(),
                    in_function: true,
                    globals: HashSet::new(),
                    nonlocals: HashSet::new(),
                    structural_depth: self.structural_depth,
                };
                nested.statements(body);
                nested.emit(Operation::LoadConstant(Constant::None), span);
                nested.emit(Operation::Return, span);
                let code = Code {
                    instructions: nested.instructions,
                    parameters: parameters
                        .iter()
                        .map(|parameter| super::bytecode::Parameter {
                            name: parameter.name.clone(),
                            has_default: parameter.default.is_some(),
                            variadic: parameter.variadic,
                            keyword_only: parameter.keyword_only,
                        })
                        .collect(),
                };
                self.emit(
                    Operation::MakeFunction {
                        name: name.clone(),
                        code: Box::new(code),
                        defaults: defaults.len(),
                    },
                    span,
                );
                self.emit(Operation::StoreName(name), span);
            }
            StatementKind::Class {
                name,
                bases,
                metaclass,
                body,
            } => {
                let base_count = bases.len();
                let fields = body
                    .iter()
                    .filter_map(|statement| match &statement.kind {
                        StatementKind::AnnotatedAssign {
                            target: AssignmentTarget::Name(name),
                            ..
                        } => Some(ClassField { name: name.clone() }),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let mut nested = Compiler {
                    instructions: Vec::new(),
                    loops: Vec::new(),
                    finalizers: Vec::new(),
                    protected_regions: Vec::new(),
                    in_function: false,
                    globals: HashSet::new(),
                    nonlocals: HashSet::new(),
                    structural_depth: self.structural_depth,
                };
                nested.statements(body);
                nested.emit(Operation::Halt, span);
                for base in bases {
                    self.expression(base);
                }
                let has_metaclass = metaclass.is_some();
                if let Some(metaclass) = metaclass {
                    self.expression(metaclass);
                }
                self.emit(
                    Operation::MakeClass {
                        name: name.clone(),
                        code: Box::new(Code {
                            instructions: nested.instructions,
                            parameters: Vec::new(),
                        }),
                        bases: base_count,
                        has_metaclass,
                        fields,
                    },
                    span,
                );
                self.emit(Operation::StoreName(name), span);
            }
            StatementKind::Decorated {
                decorators,
                statement,
            } => {
                let name = match statement.as_ref() {
                    StatementKind::Class { name, .. } | StatementKind::Function { name, .. } => {
                        name.clone()
                    }
                    _ => {
                        self.emit(
                            Operation::RuntimeError(
                                "decorators may only be applied to functions and classes".into(),
                            ),
                            span,
                        );
                        return;
                    }
                };
                for decorator in &decorators {
                    self.expression(decorator.clone());
                }
                self.statement(Statement {
                    kind: *statement,
                    span,
                });
                for _ in decorators.into_iter().rev() {
                    self.emit(Operation::LoadName(name.clone()), span);
                    self.emit(
                        Operation::Call {
                            positional: 1,
                            keywords: Vec::new(),
                            starred: vec![false],
                        },
                        span,
                    );
                    self.emit(Operation::StoreName(name.clone()), span);
                }
            }
            StatementKind::Return(value) => {
                if !self.in_function {
                    self.emit(
                        Operation::RuntimeError("'return' outside function".into()),
                        span,
                    );
                    return;
                }
                if let Some(value) = value {
                    self.expression(value);
                } else {
                    self.emit(Operation::LoadConstant(Constant::None), span);
                }
                self.emit_finalizers(span);
                self.emit(Operation::Return, span);
            }
            StatementKind::Break => {
                let Some((iterator_on_stack, finalizer_depth)) = self
                    .loops
                    .last()
                    .map(|loop_| (loop_.iterator_on_stack, loop_.finalizer_depth))
                else {
                    self.emit(Operation::RuntimeError("'break' outside loop".into()), span);
                    return;
                };
                if iterator_on_stack {
                    self.emit(Operation::PopTop, span);
                }
                self.emit_finalizers_from(finalizer_depth, span);
                let jump = self.emit(Operation::Jump(usize::MAX), span);
                self.loops
                    .last_mut()
                    .expect("checked above")
                    .breaks
                    .push(jump);
            }
            StatementKind::Continue => {
                let Some((target, finalizer_depth)) = self
                    .loops
                    .last()
                    .map(|loop_| (loop_.continue_target, loop_.finalizer_depth))
                else {
                    self.emit(
                        Operation::RuntimeError("'continue' outside loop".into()),
                        span,
                    );
                    return;
                };
                self.emit_finalizers_from(finalizer_depth, span);
                self.emit(Operation::Jump(target), span);
            }
            StatementKind::Global(names) => {
                self.globals.extend(names);
            }
            StatementKind::Nonlocal(names) => {
                if !self.in_function {
                    self.emit(
                        Operation::RuntimeError("nonlocal declaration at module scope".into()),
                        span,
                    );
                } else {
                    self.nonlocals.extend(names);
                }
            }
            StatementKind::Pass => {}
            StatementKind::Assert { test, message } => {
                self.expression(test);
                // JumpIfTrueOrPop retains a true condition while bypassing the diagnostic path;
                // false conditions are popped and continue with the lazily evaluated message.
                let passed = self.emit(Operation::JumpIfTrueOrPop(usize::MAX), span);
                self.emit(Operation::LoadConstant(Constant::Bool(false)), span);
                if let Some(message) = message {
                    self.expression(message);
                } else {
                    self.emit(Operation::LoadConstant(Constant::None), span);
                }
                self.emit(Operation::Assert, span);
                // The true branch retained its condition for the conditional jump; discard it
                // after landing so assert has no observable stack effect.
                let finished = self.emit(Operation::PopTop, span);
                self.patch_jump(passed, finished);
            }
            StatementKind::Raise(value) => {
                if let Some(value) = value {
                    self.expression(value);
                    self.emit(Operation::Raise(true), span);
                } else {
                    self.emit(Operation::Raise(false), span);
                }
            }
            StatementKind::Try {
                body,
                handlers,
                otherwise,
                finalbody,
            } => {
                let begin = self.emit(Operation::TryBegin(usize::MAX), span);
                self.protected_regions.push("try/except/finally");
                self.finalizers.push(Cleanup::Finally(finalbody.clone()));
                self.statements(body);
                self.finalizers.pop();
                self.protected_regions.pop();
                self.emit(Operation::TryEnd, span);
                let normal = self.emit(Operation::Jump(usize::MAX), span);
                let handler_start = self.instructions.len();
                self.patch_try_begin(begin, handler_start);
                let mut next_handler = None;
                let mut handled_jumps = Vec::new();
                for handler in handlers {
                    if let Some(previous) = next_handler.take() {
                        self.patch_jump(previous, self.instructions.len());
                    }
                    let typed = handler.kind.is_some();
                    if let Some(kind) = handler.kind {
                        self.expression(kind);
                    }
                    self.emit(Operation::MatchException { typed }, span);
                    let skip = self.emit(Operation::PopJumpIfFalse(usize::MAX), span);
                    if let Some(name) = handler.name {
                        self.emit(Operation::StoreName(name), span);
                    } else {
                        self.emit(Operation::PopTop, span);
                    }
                    // An exception raised by a handler must still pass through
                    // this try statement's finally body.  The nested handler
                    // catches only that new exception, runs cleanup, then
                    // re-raises it; the original exception remains below it.
                    let handler_body_try = if finalbody.is_empty() {
                        None
                    } else {
                        Some(self.emit(Operation::TryBegin(usize::MAX), span))
                    };
                    self.protected_regions.push("try/except/finally");
                    self.finalizers.push(Cleanup::Finally(finalbody.clone()));
                    self.statements(handler.body);
                    self.finalizers.pop();
                    self.protected_regions.pop();
                    if let Some(handler_body_try) = handler_body_try {
                        self.emit(Operation::TryEnd, span);
                        self.emit(Operation::ClearException, span);
                        self.statements(finalbody.clone());
                        let handler_body_done = self.emit(Operation::Jump(usize::MAX), span);
                        let handler_body_cleanup = self.instructions.len();
                        self.patch_try_begin(handler_body_try, handler_body_cleanup);
                        self.emit(Operation::PopTop, span);
                        self.statements(finalbody.clone());
                        self.emit(Operation::Reraise, span);
                        self.patch_jump(handler_body_done, self.instructions.len());
                    }
                    self.emit(Operation::ClearException, span);
                    self.statements(finalbody.clone());
                    let done = self.emit(Operation::Jump(usize::MAX), span);
                    handled_jumps.push(done);
                    next_handler = Some(skip);
                }
                if let Some(previous) = next_handler {
                    self.patch_jump(previous, self.instructions.len());
                }
                self.emit(Operation::PopTop, span);
                self.statements(finalbody.clone());
                self.emit(Operation::Reraise, span);
                let normal_start = self.instructions.len();
                self.patch_jump(normal, normal_start);
                self.protected_regions.push("try/except/finally");
                self.finalizers.push(Cleanup::Finally(finalbody.clone()));
                self.statements(otherwise);
                self.finalizers.pop();
                self.protected_regions.pop();
                self.statements(finalbody);
                let finished = self.instructions.len();
                for jump in handled_jumps {
                    self.patch_jump(jump, finished);
                }
            }
            StatementKind::With {
                context,
                target,
                body,
            } => {
                self.expression(context);
                self.emit(Operation::WithEnter, span);
                if let Some(target) = target {
                    self.store_target(target, span);
                } else {
                    self.emit(Operation::PopTop, span);
                }
                let begin = self.emit(Operation::TryBegin(usize::MAX), span);
                self.protected_regions.push("with");
                // ``with`` cleanup is represented by the VM's exception handler
                // rather than a source-level finally body.  A return/break/
                // continue here cannot currently invoke __exit__ safely, so
                // reject it through the same explicit cleanup boundary below.
                self.finalizers.push(Cleanup::WithExit);
                self.statements(body);
                self.finalizers.pop();
                self.protected_regions.pop();
                self.emit(Operation::TryEnd, span);
                self.emit(Operation::WithExit, span);
                let done = self.emit(Operation::Jump(usize::MAX), span);
                let handler = self.instructions.len();
                self.patch_try_begin(begin, handler);
                self.emit(Operation::WithExitException, span);
                self.patch_jump(done, self.instructions.len());
            }
        }
    }

    fn emit_finalizers(&mut self, span: Span) {
        self.emit_finalizers_from(0, span);
    }

    fn emit_finalizers_from(&mut self, depth: usize, span: Span) {
        // The innermost cleanup runs first, matching Python's nested finally
        // semantics.  Suppress the finalizer stack while compiling the cleanup
        // itself: a return in a finally body replaces the pending operation.
        let saved = std::mem::take(&mut self.finalizers);
        let finalizers = saved[depth..].to_vec();
        self.finalizers.extend_from_slice(&saved[..depth]);
        for finalizer in finalizers.iter().rev() {
            match finalizer {
                Cleanup::Finally(statements) => self.statements(statements.clone()),
                Cleanup::WithExit => {
                    // The normal ``with`` path closes its TryBegin before
                    // calling __exit__.  Do the same for an abrupt return,
                    // break, or continue so an exception raised by __exit__
                    // is not mistaken for an exception from the body.
                    self.emit(Operation::TryEnd, span);
                    self.emit(Operation::WithExit, span);
                }
            }
        }
        self.finalizers = saved;
    }

    fn finish_loop(&mut self, target: usize) {
        let loop_ = self.loops.pop().expect("a loop was pushed");
        for jump in loop_.breaks {
            self.patch_jump(jump, target);
        }
    }

    fn store_target(&mut self, target: AssignmentTarget, span: Span) {
        match target {
            AssignmentTarget::Name(name) => {
                self.store_name(name, span);
            }
            AssignmentTarget::Sequence(targets) => {
                let mut star_index = None;
                for (index, target) in targets.iter().enumerate() {
                    if contains_star(target) && star_index.replace(index).is_some() {
                        self.emit(
                            Operation::RuntimeError(
                                "multiple starred expressions in assignment".into(),
                            ),
                            span,
                        );
                        return;
                    }
                }
                self.emit(
                    Operation::UnpackSequence {
                        count: targets.len(),
                        star_index,
                    },
                    span,
                );
                for target in targets {
                    self.store_target(target, span);
                }
            }
            AssignmentTarget::Star(target) => self.store_target(*target, span),
            AssignmentTarget::Attribute { value, name } => {
                self.expression(value);
                self.emit(Operation::StoreAttribute(name), span);
            }
            AssignmentTarget::Subscript { value, index } => {
                self.expression(value);
                self.expression(index);
                self.emit(Operation::StoreSubscript, span);
            }
        }
    }

    fn augmented_assignment(
        &mut self,
        target: AssignmentTarget,
        operator: super::ast::BinaryOperator,
        value: Expression,
        span: Span,
    ) {
        match target {
            AssignmentTarget::Name(name) => {
                self.emit(Operation::LoadName(name.clone()), span);
                self.expression(value);
                self.emit(Operation::Binary(operator), span);
                self.store_name(name, span);
            }
            AssignmentTarget::Subscript {
                value: owner,
                index,
            } => {
                self.expression(owner);
                self.expression(index);
                self.emit(Operation::Copy(2), span);
                self.emit(Operation::Copy(2), span);
                self.emit(Operation::LoadSubscript, span);
                self.expression(value);
                self.emit(Operation::Binary(operator), span);
                self.emit(Operation::Swap(3), span);
                self.emit(Operation::Swap(2), span);
                self.emit(Operation::StoreSubscript, span);
            }
            AssignmentTarget::Attribute { value: owner, name } => {
                self.expression(owner);
                self.emit(Operation::Copy(1), span);
                self.emit(Operation::LoadAttribute(name.clone()), span);
                self.expression(value);
                self.emit(Operation::Binary(operator), span);
                self.emit(Operation::Swap(2), span);
                self.emit(Operation::StoreAttribute(name), span);
            }
            AssignmentTarget::Sequence(_) => {
                self.emit(
                    Operation::RuntimeError("augmented assignment cannot unpack a sequence".into()),
                    span,
                );
            }
            AssignmentTarget::Star(_) => {
                self.emit(
                    Operation::RuntimeError("augmented assignment cannot unpack a sequence".into()),
                    span,
                );
            }
        };
    }

    fn store_name(&mut self, name: String, span: Span) {
        if self.globals.contains(&name) {
            self.emit(Operation::StoreGlobal(name), span);
        } else if self.nonlocals.contains(&name) {
            self.emit(Operation::StoreNonlocal(name), span);
        } else {
            self.emit(Operation::StoreName(name), span);
        }
    }

    fn expression(&mut self, expression: Expression) {
        let span = expression.span;
        match expression.kind {
            ExpressionKind::Constant(value) => {
                self.emit(Operation::LoadConstant(value), span);
            }
            ExpressionKind::FString(parts) => {
                self.emit(
                    Operation::LoadConstant(Constant::String(String::new())),
                    span,
                );
                for part in parts {
                    match part {
                        FStringPart::Text(text) => {
                            self.emit(Operation::LoadConstant(Constant::String(text)), span);
                        }
                        FStringPart::Expression(expression) => {
                            self.emit(Operation::LoadName("str".into()), span);
                            self.expression(expression);
                            self.emit(
                                Operation::Call {
                                    positional: 1,
                                    keywords: Vec::new(),
                                    starred: vec![false],
                                },
                                span,
                            );
                        }
                        FStringPart::Formatted {
                            expression,
                            conversion,
                            format_spec,
                        } => {
                            self.expression(expression);
                            self.emit(
                                Operation::FormatValue {
                                    conversion,
                                    format_spec,
                                },
                                span,
                            );
                        }
                    }
                    self.emit(Operation::Binary(super::ast::BinaryOperator::Add), span);
                }
            }
            ExpressionKind::Starred(_) => {
                self.emit(
                    Operation::RuntimeError(
                        "starred expressions are only supported in calls and assignments".into(),
                    ),
                    span,
                );
            }
            ExpressionKind::Name(name) => {
                self.emit(Operation::LoadName(name), span);
            }
            ExpressionKind::List(values) => {
                let count = values.len();
                for value in values {
                    self.expression(value);
                }
                self.emit(Operation::BuildList(count), span);
            }
            ExpressionKind::Tuple(values) => {
                let count = values.len();
                for value in values {
                    self.expression(value);
                }
                self.emit(Operation::BuildTuple(count), span);
            }
            ExpressionKind::Dict(entries) => {
                let mut unpacked = Vec::with_capacity(entries.len());
                for entry in entries {
                    match entry {
                        DictEntry::Pair(key, value) => {
                            self.expression(key);
                            self.expression(value);
                            unpacked.push(false);
                        }
                        DictEntry::Unpack(value) => {
                            self.expression(value);
                            unpacked.push(true);
                        }
                    }
                }
                self.emit(Operation::BuildDict(unpacked), span);
            }
            ExpressionKind::Set(values) => {
                let count = values.len();
                for value in values {
                    self.expression(value);
                }
                self.emit(Operation::BuildSet(count), span);
            }
            ExpressionKind::ListComprehension { element, clauses } => {
                self.emit_comprehension(ComprehensionKind::List, *element, clauses, span);
            }
            ExpressionKind::SetComprehension { element, clauses } => {
                self.emit_comprehension(ComprehensionKind::Set, *element, clauses, span);
            }
            ExpressionKind::DictComprehension {
                key,
                value,
                clauses,
            } => {
                self.emit_dict_comprehension(*key, *value, clauses, span);
            }
            ExpressionKind::GeneratorExpression { element, clauses } => {
                // The first implementation intentionally materializes generator expressions.
                // It is still a separate function scope (like CPython's generator frame), and
                // all iteration remains metered by the ordinary bytecode loop.
                self.emit_comprehension(ComprehensionKind::List, *element, clauses, span);
            }
            ExpressionKind::Yield(value) => {
                if !self.in_function {
                    self.emit(
                        Operation::RuntimeError("'yield' outside function".into()),
                        span,
                    );
                } else if let Some(region) = self.protected_regions.last() {
                    self.emit(
                        Operation::RuntimeError(format!("yield is not supported inside a {region} cleanup region; suspending here would lose exception or context-manager state")),
                        span,
                    );
                } else {
                    if let Some(value) = value {
                        self.expression(*value);
                    } else {
                        self.emit(Operation::LoadConstant(Constant::None), span);
                    }
                    self.emit(Operation::Yield, span);
                }
            }
            ExpressionKind::Attribute { value, name } => {
                self.expression(*value);
                self.emit(Operation::LoadAttribute(name), span);
            }
            ExpressionKind::Subscript { value, index } => {
                self.expression(*value);
                self.expression(*index);
                self.emit(Operation::LoadSubscript, span);
            }
            ExpressionKind::Slice {
                value,
                start,
                stop,
                step,
            } => {
                self.expression(*value);
                let has_start = start.is_some();
                let has_stop = stop.is_some();
                let has_step = step.is_some();
                if let Some(start) = start {
                    self.expression(*start);
                }
                if let Some(stop) = stop {
                    self.expression(*stop);
                }
                if let Some(step) = step {
                    self.expression(*step);
                }
                self.emit(
                    Operation::LoadSlice {
                        has_start,
                        has_stop,
                        has_step,
                    },
                    span,
                );
            }
            ExpressionKind::SliceValue { start, stop, step } => {
                let has_start = start.is_some();
                let has_stop = stop.is_some();
                let has_step = step.is_some();
                if let Some(start) = start {
                    self.expression(*start);
                }
                if let Some(stop) = stop {
                    self.expression(*stop);
                }
                if let Some(step) = step {
                    self.expression(*step);
                }
                self.emit(
                    Operation::BuildSlice {
                        has_start,
                        has_stop,
                        has_step,
                    },
                    span,
                );
            }
            ExpressionKind::Call {
                function,
                arguments,
            } => {
                self.expression(*function);
                let mut positional = 0;
                let mut keywords = Vec::new();
                let mut starred = Vec::new();
                for argument in arguments {
                    if let Some(name) = argument.name {
                        keywords.push(name);
                    } else {
                        positional += 1;
                    }
                    starred.push(argument.starred);
                    self.expression(argument.value);
                }
                self.emit(
                    Operation::Call {
                        positional,
                        keywords,
                        starred,
                    },
                    span,
                );
            }
            ExpressionKind::Lambda { parameters, body } => {
                let defaults = parameters
                    .iter()
                    .filter_map(|parameter| parameter.default.clone())
                    .collect::<Vec<_>>();
                for default in &defaults {
                    self.expression(default.clone());
                }
                let mut nested = Compiler {
                    instructions: Vec::new(),
                    loops: Vec::new(),
                    finalizers: Vec::new(),
                    protected_regions: Vec::new(),
                    in_function: true,
                    globals: HashSet::new(),
                    nonlocals: HashSet::new(),
                    structural_depth: self.structural_depth,
                };
                nested.expression(*body);
                nested.emit(Operation::Return, span);
                self.emit(
                    Operation::MakeFunction {
                        name: "<lambda>".into(),
                        code: Box::new(Code {
                            instructions: nested.instructions,
                            parameters: parameters
                                .iter()
                                .map(|parameter| super::bytecode::Parameter {
                                    name: parameter.name.clone(),
                                    has_default: parameter.default.is_some(),
                                    variadic: parameter.variadic,
                                    keyword_only: parameter.keyword_only,
                                })
                                .collect(),
                        }),
                        defaults: defaults.len(),
                    },
                    span,
                );
            }
            ExpressionKind::Unary { operator, operand } => {
                self.expression(*operand);
                self.emit(Operation::Unary(operator), span);
            }
            ExpressionKind::Binary {
                left,
                operator,
                right,
            } => {
                self.expression(*left);
                self.expression(*right);
                self.emit(Operation::Binary(operator), span);
            }
            ExpressionKind::Conditional {
                test,
                body,
                otherwise,
            } => {
                self.expression(*test);
                let otherwise_jump = self.emit(Operation::PopJumpIfFalse(usize::MAX), span);
                self.expression(*body);
                let finished_jump = self.emit(Operation::Jump(usize::MAX), span);
                self.patch_jump(otherwise_jump, self.instructions.len());
                self.expression(*otherwise);
                self.patch_jump(finished_jump, self.instructions.len());
            }
            ExpressionKind::Boolean {
                left,
                operator,
                right,
            } => {
                self.expression(*left);
                let jump = match operator {
                    BooleanOperator::And => Operation::JumpIfFalseOrPop(usize::MAX),
                    BooleanOperator::Or => Operation::JumpIfTrueOrPop(usize::MAX),
                };
                let jump = self.emit(jump, span);
                self.expression(*right);
                self.patch_jump(jump, self.instructions.len());
            }
            ExpressionKind::Comparison { left, comparisons } => {
                self.expression(*left);
                let count = comparisons.len();
                let mut cleanup_jumps = Vec::new();
                for (index, (operator, right)) in comparisons.into_iter().enumerate() {
                    self.expression(right);
                    if index + 1 != count {
                        self.emit(Operation::Swap(2), span);
                        self.emit(Operation::Copy(2), span);
                    }
                    self.emit(Operation::Compare(operator), span);
                    if index + 1 != count {
                        cleanup_jumps
                            .push(self.emit(Operation::JumpIfFalseOrPop(usize::MAX), span));
                    }
                }
                if !cleanup_jumps.is_empty() {
                    let finished = self.emit(Operation::Jump(usize::MAX), span);
                    let cleanup = self.instructions.len();
                    self.emit(Operation::Swap(2), span);
                    self.emit(Operation::PopTop, span);
                    let end = self.instructions.len();
                    for jump in cleanup_jumps {
                        self.patch_jump(jump, cleanup);
                    }
                    self.patch_jump(finished, end);
                }
            }
        }
    }

    fn emit(&mut self, operation: Operation, span: Span) -> usize {
        let index = self.instructions.len();
        self.instructions.push(Instruction { operation, span });
        index
    }

    fn emit_comprehension(
        &mut self,
        kind: ComprehensionKind,
        element: Expression,
        clauses: Vec<ComprehensionClause>,
        span: Span,
    ) {
        let mut nested = Compiler {
            instructions: Vec::new(),
            loops: Vec::new(),
            finalizers: Vec::new(),
            protected_regions: Vec::new(),
            in_function: true,
            globals: HashSet::new(),
            nonlocals: HashSet::new(),
            structural_depth: self.structural_depth,
        };
        let result_name = "$__shellsim_comprehension_result".to_string();
        nested.emit(
            match kind {
                ComprehensionKind::List => Operation::BuildList(0),
                ComprehensionKind::Set => Operation::BuildSet(0),
            },
            span,
        );
        nested.emit(Operation::StoreName(result_name.clone()), span);
        nested.emit_comprehension_body(&clauses, 0, &element, kind, &result_name, span);
        nested.emit(Operation::LoadName(result_name), span);
        nested.emit(Operation::Return, span);
        self.emit(
            Operation::MakeFunction {
                name: "<comprehension>".into(),
                code: Box::new(Code {
                    instructions: nested.instructions,
                    parameters: Vec::new(),
                }),
                defaults: 0,
            },
            span,
        );
        self.emit(
            Operation::Call {
                positional: 0,
                keywords: Vec::new(),
                starred: Vec::new(),
            },
            span,
        );
    }

    fn emit_dict_comprehension(
        &mut self,
        key: Expression,
        value: Expression,
        clauses: Vec<ComprehensionClause>,
        span: Span,
    ) {
        let mut nested = Compiler {
            instructions: Vec::new(),
            loops: Vec::new(),
            finalizers: Vec::new(),
            protected_regions: Vec::new(),
            in_function: true,
            globals: HashSet::new(),
            nonlocals: HashSet::new(),
            structural_depth: self.structural_depth,
        };
        let result_name = "$__shellsim_comprehension_result".to_string();
        nested.emit(Operation::BuildDict(Vec::new()), span);
        nested.emit(Operation::StoreName(result_name.clone()), span);
        nested.emit_dict_comprehension_body(&clauses, 0, &key, &value, &result_name, span);
        nested.emit(Operation::LoadName(result_name), span);
        nested.emit(Operation::Return, span);
        self.emit(
            Operation::MakeFunction {
                name: "<dictcomp>".into(),
                code: Box::new(Code {
                    instructions: nested.instructions,
                    parameters: Vec::new(),
                }),
                defaults: 0,
            },
            span,
        );
        self.emit(
            Operation::Call {
                positional: 0,
                keywords: Vec::new(),
                starred: Vec::new(),
            },
            span,
        );
    }

    fn emit_comprehension_body(
        &mut self,
        clauses: &[ComprehensionClause],
        index: usize,
        element: &Expression,
        kind: ComprehensionKind,
        result_name: &str,
        span: Span,
    ) {
        let clause = &clauses[index];
        self.expression(clause.iterable.clone());
        self.emit(Operation::GetIterator, span);
        let next = self.emit(Operation::ForIterator(usize::MAX), span);
        self.store_target(clause.target.clone(), span);
        for condition in &clause.conditions {
            self.expression(condition.clone());
            let skip = self.emit(Operation::PopJumpIfFalse(usize::MAX), span);
            self.patch_jump(skip, next);
        }
        if index + 1 < clauses.len() {
            self.emit_comprehension_body(clauses, index + 1, element, kind, result_name, span);
        } else {
            self.expression(element.clone());
            self.emit(Operation::LoadName(result_name.into()), span);
            self.emit(
                Operation::LoadAttribute(match kind {
                    ComprehensionKind::List => "append".into(),
                    ComprehensionKind::Set => "add".into(),
                }),
                span,
            );
            self.emit(Operation::Swap(2), span);
            self.emit(
                Operation::Call {
                    positional: 1,
                    keywords: Vec::new(),
                    starred: vec![false],
                },
                span,
            );
            self.emit(Operation::PopTop, span);
        }
        self.emit(Operation::Jump(next), span);
        let exhausted = self.instructions.len();
        self.patch_jump(next, exhausted);
    }

    fn emit_dict_comprehension_body(
        &mut self,
        clauses: &[ComprehensionClause],
        index: usize,
        key: &Expression,
        value: &Expression,
        result_name: &str,
        span: Span,
    ) {
        let clause = &clauses[index];
        self.expression(clause.iterable.clone());
        self.emit(Operation::GetIterator, span);
        let next = self.emit(Operation::ForIterator(usize::MAX), span);
        self.store_target(clause.target.clone(), span);
        for condition in &clause.conditions {
            self.expression(condition.clone());
            let skip = self.emit(Operation::PopJumpIfFalse(usize::MAX), span);
            self.patch_jump(skip, next);
        }
        if index + 1 < clauses.len() {
            self.emit_dict_comprehension_body(clauses, index + 1, key, value, result_name, span);
        } else {
            self.expression(value.clone());
            self.emit(Operation::LoadName(result_name.into()), span);
            self.expression(key.clone());
            self.emit(Operation::StoreSubscript, span);
        }
        self.emit(Operation::Jump(next), span);
        let exhausted = self.instructions.len();
        self.patch_jump(next, exhausted);
    }

    fn patch_jump(&mut self, instruction: usize, target: usize) {
        match &mut self.instructions[instruction].operation {
            Operation::Jump(slot)
            | Operation::JumpIfFalseOrPop(slot)
            | Operation::JumpIfTrueOrPop(slot)
            | Operation::PopJumpIfFalse(slot)
            | Operation::ForIterator(slot) => *slot = target,
            _ => unreachable!("compiler attempted to patch a non-jump operation"),
        }
    }

    fn patch_try_begin(&mut self, instruction: usize, target: usize) {
        match &mut self.instructions[instruction].operation {
            Operation::TryBegin(slot) => *slot = target,
            _ => unreachable!("compiler attempted to patch a non-handler operation"),
        }
    }
}

#[derive(Clone, Copy)]
enum ComprehensionKind {
    List,
    Set,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::python::lexer::lex;
    use crate::python::parser::parse;

    #[test]
    fn assignment_has_an_explicit_stack_contract() {
        let code = compile(parse(lex("x = 40 + 2").unwrap()).unwrap());
        assert!(matches!(
            code.instructions[2].operation,
            Operation::Binary(_)
        ));
        assert_eq!(
            code.instructions[3].operation,
            Operation::StoreName("x".into())
        );
        assert_eq!(code.instructions.last().unwrap().operation, Operation::Halt);
    }

    #[test]
    fn caps_structural_nesting_without_recursing_forever() {
        let mut body = vec![Statement {
            kind: StatementKind::Pass,
            span: Span::default(),
        }];
        for _ in 0..300 {
            body = vec![Statement {
                kind: StatementKind::If {
                    test: Expression {
                        kind: ExpressionKind::Constant(Constant::Bool(true)),
                        span: Span::default(),
                    },
                    body,
                    otherwise: Vec::new(),
                },
                span: Span::default(),
            }];
        }
        let code = compile(Program { statements: body });
        assert!(code.instructions.iter().any(|instruction| {
            matches!(
                &instruction.operation,
                Operation::RuntimeError(message)
                    if message.contains("compiler structural nesting limit")
            )
        }));
    }
}
