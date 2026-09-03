//! Metered stack VM for the first vertical slice.

use crate::interp::Interp;

use std::cmp::Ordering;
use std::collections::HashMap;

use regex::{Regex, RegexBuilder};

use super::ast::{BinaryOperator, ComparisonOperator, Constant, UnaryOperator};
use super::bytecode::{ClassField, Code, Operation};
use super::heap::{Method, Object, ScopeId};
use super::{protocol, ExecResult, Out, ReplState, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeValue {
    Module(Module),
    Function(Builtin),
    Stream(Stream),
    Environment,
    ExceptionType(ExceptionType),
    /// Runtime representation of ``typing.List`` used by the generic-alias probe.
    TypingList,
    /// Marker used as the only supported base for the capability-free enum slice.
    EnumBase,
    /// Marker used as the only supported base for the capability-free unittest slice.
    UnitTestBase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExceptionType(pub &'static str);

#[derive(Clone, Debug)]
struct RaisedException {
    kind: String,
    value: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Module {
    Sys,
    Os,
    Json,
    Collections,
    Math,
    String,
    Itertools,
    Heapq,
    Bisect,
    Functools,
    Typing,
    Subprocess,
    Re,
    Argparse,
    Dataclasses,
    Enum,
    Pytest,
    Unittest,
    Time,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stream {
    Stdout,
    Stderr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Builtin {
    Print,
    Exit,
    String,
    Repr,
    Integer,
    Length,
    Sorted,
    Minimum,
    Maximum,
    Sum,
    Absolute,
    Boolean,
    List,
    Tuple,
    Set,
    Range,
    Enumerate,
    Zip,
    Any,
    All,
    Next,
    JsonDumps,
    DefaultDict,
    Math(MathBuiltin),
    Write(Stream),
    GetEnv,
    EnvironmentGet,
    ItertoolsCount,
    ItertoolsIslice,
    HeapqHeapify,
    HeapqHeappop,
    BisectLeft,
    Reduce,
    ReCompile,
    ReSearch,
    ReMatch,
    ReFullMatch,
    ReFindAll,
    ReFindIter,
    ReSub,
    ReEscape,
    ArgparseArgumentParser,
    ArgparseNamespace,
    Dataclass,
    PytestFail,
    PytestSkip,
    PytestRaises,
    Time,
    TimeNs,
    Monotonic,
    MonotonicNs,
    ProcessTime,
    ProcessTimeNs,
    Sleep,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MathBuiltin {
    Ceil,
    Exp,
    IsInf,
    IsNan,
    Log,
    Sin,
    Sqrt,
}

impl MathBuiltin {
    fn name(self) -> &'static str {
        match self {
            Self::Ceil => "ceil",
            Self::Exp => "exp",
            Self::IsInf => "isinf",
            Self::IsNan => "isnan",
            Self::Log => "log",
            Self::Sin => "sin",
            Self::Sqrt => "sqrt",
        }
    }
}

pub(super) fn execute(
    interp: &mut Interp,
    source: &str,
    argv: &[String],
    state: &mut ReplState,
    interactive: bool,
    out: Out,
    err: Out,
) -> ExecResult {
    let tokens = match super::lexer::lex(source) {
        Ok(tokens) => tokens,
        Err(error) => {
            return ExecResult::Unsupported(format!(
                "{} at line {}, column {}",
                error.message, error.span.line, error.span.column
            ))
        }
    };
    let program = match super::parser::parse(tokens) {
        Ok(program) => program,
        Err(error) => {
            return ExecResult::Unsupported(format!(
                "{} at line {}, column {}",
                error.message, error.span.line, error.span.column
            ))
        }
    };
    let code = super::compiler::compile(program);
    Vm::new(interp, argv, state, interactive, out, err).run(&code)
}

struct Vm<'a> {
    interp: &'a mut Interp,
    argv: &'a [String],
    state: &'a mut ReplState,
    interactive: bool,
    out: Out<'a>,
    err: Out<'a>,
    stack: Vec<Value>,
    local_scopes: Vec<ScopeId>,
    class_scopes: Vec<ScopeId>,
    class_bindings: Vec<Vec<String>>,
    call_depth: usize,
    pending_exception: Option<RaisedException>,
    exception_stack: Vec<RaisedException>,
    with_contexts: Vec<Value>,
}

impl<'a> Vm<'a> {
    fn new(
        interp: &'a mut Interp,
        argv: &'a [String],
        state: &'a mut ReplState,
        interactive: bool,
        out: Out<'a>,
        err: Out<'a>,
    ) -> Self {
        Self {
            interp,
            argv,
            state,
            interactive,
            out,
            err,
            stack: Vec::new(),
            local_scopes: Vec::new(),
            class_scopes: Vec::new(),
            class_bindings: Vec::new(),
            call_depth: 0,
            pending_exception: None,
            exception_stack: Vec::new(),
            with_contexts: Vec::new(),
        }
    }

    fn run(mut self, code: &Code) -> ExecResult {
        let retained_heap = self.state.heap.modeled_bytes();
        if retained_heap != 0 && !self.interp.resources.reserve_memory(retained_heap) {
            return ExecResult::Exit(137);
        }
        match self.execute_code(code) {
            Ok(Execution::Halt) | Ok(Execution::Return(_)) | Ok(Execution::Yield(_, _)) => {
                ExecResult::Continue
            }
            Ok(Execution::Exit(status)) => ExecResult::Exit(status),
            Err((error, span)) => {
                if let Some(reason) = self.interp.resources.stop_reason() {
                    ExecResult::Exit(reason.exit_status())
                } else if let Some(exception) = &self.pending_exception {
                    let rendered = protocol::display(&self.state.heap, &exception.value)
                        .unwrap_or_else(|_| exception.kind.clone());
                    ExecResult::Unsupported(format!(
                        "{}: {} at line {}, column {}",
                        exception.kind, rendered, span.line, span.column
                    ))
                } else {
                    ExecResult::Unsupported(format!(
                        "{} at line {}, column {}",
                        error, span.line, span.column
                    ))
                }
            }
        }
    }

    fn execute_code(&mut self, code: &Code) -> Result<Execution, (String, super::source::Span)> {
        let mut handlers: Vec<(usize, usize)> = Vec::new();
        self.execute_code_from(code, 0, &mut handlers)
    }

    fn execute_code_from(
        &mut self,
        code: &Code,
        mut instruction_pointer: usize,
        handlers: &mut Vec<(usize, usize)>,
    ) -> Result<Execution, (String, super::source::Span)> {
        loop {
            if self.interp.deadline_interrupt.is_some() {
                return Ok(Execution::Exit(124));
            }
            let Some(instruction) = code.instructions.get(instruction_pointer) else {
                return Err((
                    "instruction pointer left the code object".into(),
                    super::source::Span::default(),
                ));
            };
            if !self.interp.resources.charge_cpu(1) {
                return Ok(Execution::Exit(137));
            }
            let result = match &instruction.operation {
                Operation::LoadConstant(constant) => {
                    self.stack.push(value_from_constant(constant));
                    Ok(())
                }
                Operation::LoadName(name) => self.load_name(name),
                Operation::StoreName(name) => self.store_name(name),
                Operation::StoreNonlocal(name) => self.store_nonlocal(name),
                Operation::StoreAttribute(name) => {
                    let owner = self.pop().map_err(|error| (error, instruction.span))?;
                    let value = self.pop().map_err(|error| (error, instruction.span))?;
                    self.store_attribute(owner, name, value)
                }
                Operation::StoreSubscript => self.store_subscript(),
                Operation::DeleteName(name) => {
                    if let Some(scope) = self.local_scopes.last().copied() {
                        self.state.heap.scope_remove(scope, name).map(|_| ())
                    } else {
                        self.state.locals.remove(name);
                        Ok(())
                    }
                }
                Operation::Import(name) => self.import(name),
                Operation::LoadAttribute(name) => self.load_attribute(name),
                Operation::LoadSubscript => self.load_subscript(),
                Operation::BuildList(count) => self.build_sequence(*count, SequenceKind::List),
                Operation::BuildTuple(count) => self.build_sequence(*count, SequenceKind::Tuple),
                Operation::BuildDict(count) => self.build_dict(*count),
                Operation::BuildSet(count) => self.build_set(*count),
                Operation::UnpackSequence { count, star_index } => {
                    self.unpack_sequence(*count, *star_index)
                }
                Operation::MakeFunction {
                    name,
                    code,
                    defaults,
                } => self.make_function(name.clone(), (**code).clone(), *defaults),
                Operation::MakeClass {
                    name,
                    code,
                    bases,
                    fields,
                } => self.make_class(name.clone(), code, *bases, fields),
                Operation::GetIterator => self.get_iterator(),
                Operation::ForIterator(target) => match self.for_iterator() {
                    Ok(true) => Ok(()),
                    Ok(false) => {
                        instruction_pointer = *target;
                        continue;
                    }
                    Err(error) => Err(error),
                },
                Operation::Unary(operator) => self.unary(*operator),
                Operation::Binary(operator) => self.binary(*operator),
                Operation::Compare(operator) => self.compare(*operator),
                Operation::Call {
                    positional,
                    keywords,
                    starred,
                } => match self.call(*positional, keywords, starred) {
                    Ok(CallResult::Value(value)) => {
                        self.stack.push(value);
                        Ok(())
                    }
                    Ok(CallResult::Exit(status)) => return Ok(Execution::Exit(status)),
                    Err(error) => Err(error),
                },
                Operation::Copy(depth) => self.copy(*depth),
                Operation::Swap(depth) => self.swap(*depth),
                Operation::PopTop => self.pop().map(|_| ()),
                Operation::Jump(target) => {
                    instruction_pointer = *target;
                    continue;
                }
                Operation::JumpIfFalseOrPop(target) => match self.jump_if_or_pop(false) {
                    Ok(true) => {
                        instruction_pointer = *target;
                        continue;
                    }
                    Ok(false) => Ok(()),
                    Err(error) => Err(error),
                },
                Operation::JumpIfTrueOrPop(target) => match self.jump_if_or_pop(true) {
                    Ok(true) => {
                        instruction_pointer = *target;
                        continue;
                    }
                    Ok(false) => Ok(()),
                    Err(error) => Err(error),
                },
                Operation::PopJumpIfFalse(target) => {
                    let value = self.pop().map_err(|error| (error, instruction.span))?;
                    if !protocol::truth(&self.state.heap, &value)
                        .map_err(|error| (error, instruction.span))?
                    {
                        instruction_pointer = *target;
                        continue;
                    }
                    Ok(())
                }
                Operation::Return => {
                    let value = self.pop().map_err(|error| (error, instruction.span))?;
                    return Ok(Execution::Return(value));
                }
                Operation::Yield => {
                    let value = self.pop().map_err(|error| (error, instruction.span))?;
                    return Ok(Execution::Yield(value, instruction_pointer + 1));
                }
                Operation::RuntimeError(error) => Err(error.clone()),
                Operation::Assert => {
                    let message = self.pop().map_err(|error| (error, instruction.span))?;
                    let condition = self.pop().map_err(|error| (error, instruction.span))?;
                    if protocol::truth(&self.state.heap, &condition)
                        .map_err(|error| (error, instruction.span))?
                    {
                        Ok(())
                    } else {
                        let message = if matches!(message, Value::None) {
                            String::new()
                        } else {
                            protocol::display(&self.state.heap, &message)
                                .map_err(|error| (error, instruction.span))?
                        };
                        let value = Value::Exception {
                            kind: "AssertionError".into(),
                            message,
                        };
                        self.pending_exception = Some(RaisedException {
                            kind: "AssertionError".into(),
                            value,
                        });
                        Err("assertion failed".into())
                    }
                }
                Operation::TryBegin(target) => {
                    handlers.push((*target, self.stack.len()));
                    Ok(())
                }
                Operation::TryEnd => {
                    handlers
                        .pop()
                        .ok_or("invalid bytecode exception handler")
                        .map_err(|e| (e.to_string(), instruction.span))?;
                    Ok(())
                }
                Operation::MatchException(expected) => {
                    let exception = self
                        .exception_stack
                        .last()
                        .ok_or("no active exception")
                        .map_err(|e| (e.to_string(), instruction.span))?;
                    let matches = expected.as_deref().is_none_or(|name| {
                        name == "BaseException" || name == "Exception" || name == exception.kind
                    });
                    self.stack.push(Value::Bool(matches));
                    Ok(())
                }
                Operation::ClearException => {
                    self.exception_stack
                        .pop()
                        .ok_or("no active exception")
                        .map_err(|e| (e.to_string(), instruction.span))?;
                    Ok(())
                }
                Operation::Reraise => {
                    let exception = self
                        .exception_stack
                        .last()
                        .cloned()
                        .ok_or("no active exception")
                        .map_err(|e| (e.to_string(), instruction.span))?;
                    self.pending_exception = Some(exception);
                    Err("exception raised".into())
                }
                Operation::Raise(has_value) => {
                    let exception = if *has_value {
                        let value = self.pop().map_err(|e| (e, instruction.span))?;
                        match value {
                            Value::Exception { kind, message } => RaisedException {
                                kind: kind.clone(),
                                value: Value::Exception {
                                    kind: kind.clone(),
                                    message,
                                },
                            },
                            Value::Native(NativeValue::ExceptionType(ExceptionType(kind))) => {
                                let value = Value::Exception {
                                    kind: kind.to_string(),
                                    message: String::new(),
                                };
                                RaisedException {
                                    kind: kind.to_string(),
                                    value,
                                }
                            }
                            _ => {
                                return Err((
                                    "exceptions must derive from BaseException".into(),
                                    instruction.span,
                                ))
                            }
                        }
                    } else {
                        self.exception_stack
                            .last()
                            .cloned()
                            .ok_or("No active exception to reraise")
                            .map_err(|e| (e.to_string(), instruction.span))?
                    };
                    self.pending_exception = Some(exception);
                    Err("exception raised".into())
                }
                Operation::WithEnter => {
                    let context = self.pop().map_err(|e| (e, instruction.span))?;
                    self.stack.push(context.clone());
                    self.with_contexts.push(context);
                    self.load_attribute("__enter__")
                        .map_err(|e| (e, instruction.span))?;
                    match self.call(0, &[], &[]).map_err(|e| (e, instruction.span))? {
                        CallResult::Value(value) => self.stack.push(value),
                        CallResult::Exit(status) => return Ok(Execution::Exit(status)),
                    }
                    Ok(())
                }
                Operation::WithExit => {
                    let context = self
                        .with_contexts
                        .pop()
                        .ok_or("with stack underflow")
                        .map_err(|e| (e.to_string(), instruction.span))?;
                    self.stack.push(context);
                    self.load_attribute("__exit__")
                        .map_err(|e| (e, instruction.span))?;
                    self.stack.extend([Value::None, Value::None, Value::None]);
                    match self
                        .call(3, &[], &[false, false, false])
                        .map_err(|e| (e, instruction.span))?
                    {
                        CallResult::Value(_) => Ok(()),
                        CallResult::Exit(status) => return Ok(Execution::Exit(status)),
                    }
                }
                Operation::WithExitException => {
                    let context = self
                        .with_contexts
                        .pop()
                        .ok_or("with stack underflow")
                        .map_err(|e| (e.to_string(), instruction.span))?;
                    let exception = self
                        .exception_stack
                        .last()
                        .cloned()
                        .ok_or("no active exception")
                        .map_err(|e| (e.to_string(), instruction.span))?;
                    self.stack.push(context);
                    self.load_attribute("__exit__")
                        .map_err(|e| (e, instruction.span))?;
                    self.stack.extend([
                        Value::String(exception.kind.clone()),
                        exception.value.clone(),
                        Value::None,
                    ]);
                    let result = match self
                        .call(3, &[], &[false, false, false])
                        .map_err(|e| (e, instruction.span))?
                    {
                        CallResult::Value(value) => value,
                        CallResult::Exit(status) => return Ok(Execution::Exit(status)),
                    };
                    if protocol::truth(&self.state.heap, &result)
                        .map_err(|e| (e, instruction.span))?
                    {
                        self.pending_exception = None;
                        self.exception_stack.pop();
                        Ok(())
                    } else {
                        self.pending_exception = Some(exception);
                        Err("exception raised".into())
                    }
                }
                Operation::PopExpression => match self.pop() {
                    Ok(value) => {
                        if self.interactive && !matches!(value, Value::None) {
                            let rendered = match protocol::repr(&self.state.heap, &value) {
                                Ok(rendered) => rendered,
                                Err(error) => return Err((error, instruction.span)),
                            };
                            self.out.extend_from_slice(rendered.as_bytes());
                            self.out.push(b'\n');
                        }
                        Ok(())
                    }
                    Err(error) => Err(error),
                },
                Operation::Halt => return Ok(Execution::Halt),
            };
            if let Err(error) = result {
                if let Some(exception) = self.pending_exception.take() {
                    if let Some((target, depth)) = handlers.pop() {
                        self.stack.truncate(depth);
                        self.stack.push(exception.value.clone());
                        self.exception_stack.push(exception);
                        instruction_pointer = target;
                        continue;
                    }
                    self.pending_exception = Some(exception);
                }
                return Err((error, instruction.span));
            }
            instruction_pointer += 1;
        }
    }

    fn load_name(&mut self, name: &str) -> Result<(), String> {
        let local_scope = self.local_scopes.last().copied();
        let scoped = local_scope.and_then(|scope| self.state.heap.scope_get(scope, name).cloned());
        let can_use_globals = local_scope
            .map(|scope| self.state.heap.scope_uses_repl_globals(scope))
            .transpose()?
            .unwrap_or(true);
        let value = scoped
            .or_else(|| {
                can_use_globals
                    .then(|| self.state.locals.get(name).cloned())
                    .flatten()
            })
            .or_else(|| {
                let builtin = match name {
                    "print" => Builtin::Print,
                    "exit" | "quit" => Builtin::Exit,
                    "str" => Builtin::String,
                    "repr" => Builtin::Repr,
                    "int" => Builtin::Integer,
                    "len" => Builtin::Length,
                    "sorted" => Builtin::Sorted,
                    "min" => Builtin::Minimum,
                    "max" => Builtin::Maximum,
                    "sum" => Builtin::Sum,
                    "abs" => Builtin::Absolute,
                    "bool" => Builtin::Boolean,
                    "list" => Builtin::List,
                    "tuple" => Builtin::Tuple,
                    "set" => Builtin::Set,
                    "range" => Builtin::Range,
                    "enumerate" => Builtin::Enumerate,
                    "zip" => Builtin::Zip,
                    "any" => Builtin::Any,
                    "all" => Builtin::All,
                    "next" => Builtin::Next,
                    "Exception" => {
                        return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                            "Exception",
                        ))))
                    }
                    "BaseException" => {
                        return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                            "BaseException",
                        ))))
                    }
                    "RuntimeError" => {
                        return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                            "RuntimeError",
                        ))))
                    }
                    "ValueError" => {
                        return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                            "ValueError",
                        ))))
                    }
                    "TypeError" => {
                        return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                            "TypeError",
                        ))))
                    }
                    "KeyError" => {
                        return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                            "KeyError",
                        ))))
                    }
                    "IndexError" => {
                        return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                            "IndexError",
                        ))))
                    }
                    "StopIteration" => {
                        return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                            "StopIteration",
                        ))))
                    }
                    "AssertionError" => {
                        return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                            "AssertionError",
                        ))))
                    }
                    "Skipped" => {
                        return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                            "Skipped",
                        ))))
                    }
                    _ => return None,
                };
                Some(Value::Native(NativeValue::Function(builtin)))
            });
        self.stack
            .push(value.ok_or_else(|| format!("name {name:?} is not defined"))?);
        Ok(())
    }

    fn store_name(&mut self, name: &str) -> Result<(), String> {
        let value = self.pop()?;
        if let Some(scope) = self.local_scopes.last().copied() {
            if self.class_scopes.last().copied() == Some(scope)
                && self
                    .class_bindings
                    .last()
                    .is_some_and(|bindings| !bindings.iter().any(|binding| binding == name))
            {
                self.class_bindings
                    .last_mut()
                    .expect("class binding stack is present")
                    .push(name.to_string());
            }
            self.state.heap.scope_insert(
                scope,
                name.to_string(),
                value,
                &mut self.interp.resources,
            )?;
        } else {
            self.state.locals.insert(name.to_string(), value);
        }
        Ok(())
    }

    fn store_nonlocal(&mut self, name: &str) -> Result<(), String> {
        let value = self.pop()?;
        let scope = self
            .local_scopes
            .last()
            .copied()
            .ok_or_else(|| format!("no binding for nonlocal {name:?} found"))?;
        self.state.heap.scope_store_nonlocal(scope, name, value)
    }

    fn import(&mut self, name: &str) -> Result<(), String> {
        if let Some(module) = match name {
            "sys" => Some(Module::Sys),
            "os" => Some(Module::Os),
            "json" => Some(Module::Json),
            "collections" => Some(Module::Collections),
            "math" => Some(Module::Math),
            "string" => Some(Module::String),
            "itertools" => Some(Module::Itertools),
            "heapq" => Some(Module::Heapq),
            "bisect" => Some(Module::Bisect),
            "functools" => Some(Module::Functools),
            "typing" => Some(Module::Typing),
            "subprocess" => Some(Module::Subprocess),
            "re" => Some(Module::Re),
            "argparse" => Some(Module::Argparse),
            "dataclasses" => Some(Module::Dataclasses),
            "enum" => Some(Module::Enum),
            "pytest" => Some(Module::Pytest),
            "unittest" => Some(Module::Unittest),
            "time" => Some(Module::Time),
            _ => None,
        } {
            self.stack.push(Value::Native(NativeValue::Module(module)));
            return Ok(());
        }
        if let Some(module) = self.state.modules.get(name).cloned() {
            self.stack.push(module);
            return Ok(());
        }

        let relative = name.replace('.', "/");
        let roots = if self.state.import_paths.is_empty() {
            vec![self.interp.cwd.clone()]
        } else {
            self.state.import_paths.clone()
        };
        let mut source = None;
        for root in roots {
            for suffix in [format!("{relative}.py"), format!("{relative}/__init__.py")] {
                let candidate = crate::vfs::resolve_against(&root, &suffix);
                if let Ok(contents) = self.interp.vfs.read_string("/", &candidate) {
                    source = Some((candidate, contents));
                    break;
                }
            }
            if source.is_some() {
                break;
            }
        }
        let Some((path, source)) = source else {
            return Err(format!(
                "no module named {name:?} in the virtual filesystem"
            ));
        };
        let parse_memory = u64::try_from(source.len())
            .ok()
            .and_then(|bytes| bytes.checked_mul(4))
            .ok_or("module source is too large")?;
        if !self.interp.resources.reserve_memory(parse_memory)
            || !self.interp.resources.charge_cpu(source.len() as u64)
        {
            return Err("resource limit exceeded while importing module".into());
        }
        let tokens = super::lexer::lex(&source).map_err(|error| {
            format!(
                "{} in {path} at line {}, column {}",
                error.message, error.span.line, error.span.column
            )
        })?;
        let program = super::parser::parse(tokens).map_err(|error| {
            format!(
                "{} in {path} at line {}, column {}",
                error.message, error.span.line, error.span.column
            )
        })?;
        let code = super::compiler::compile(program);
        let scope = self.state.heap.allocate_scope(
            None,
            false,
            HashMap::new(),
            &mut self.interp.resources,
        )?;
        let module = self.allocate_object(Object::Module {
            name: name.to_string(),
            scope,
        })?;
        self.state.modules.insert(name.to_string(), module.clone());

        let outer_stack = std::mem::take(&mut self.stack);
        self.local_scopes.push(scope);
        let execution = self.execute_code(&code);
        self.local_scopes.pop();
        self.stack = outer_stack;
        match execution {
            Ok(Execution::Halt) => {
                self.stack.push(module);
                Ok(())
            }
            Ok(Execution::Return(_)) => {
                self.state.modules.remove(name);
                Err(format!("'return' outside function in module {name:?}"))
            }
            Ok(Execution::Yield(_, _)) => {
                self.state.modules.remove(name);
                Err(format!("'yield' outside function in module {name:?}"))
            }
            Ok(Execution::Exit(status)) => {
                self.state.modules.remove(name);
                Err(format!("module {name:?} exited with status {status}"))
            }
            Err((error, span)) => {
                self.state.modules.remove(name);
                Err(format!(
                    "{error} in {path} at line {}, column {}",
                    span.line, span.column
                ))
            }
        }
    }

    fn load_attribute(&mut self, name: &str) -> Result<(), String> {
        let owner = self.pop()?;
        if let Some(method) = self.method_for(&owner, name)? {
            let method = self.state.heap.allocate(
                Object::BoundMethod {
                    receiver: owner,
                    method,
                },
                &mut self.interp.resources,
            )?;
            self.stack.push(method);
            return Ok(());
        }
        if let Value::Object(id) = &owner {
            match self.state.heap.get(*id)?.clone() {
                Object::Module { scope, .. } => {
                    let value = self
                        .state
                        .heap
                        .scope_get(scope, name)
                        .cloned()
                        .ok_or_else(|| format!("module has no attribute {name:?}"))?;
                    self.stack.push(value);
                    return Ok(());
                }
                Object::Class { attributes, .. } => {
                    let value = attributes
                        .get(name)
                        .cloned()
                        .ok_or_else(|| format!("class has no attribute {name:?}"))?;
                    self.stack.push(value);
                    return Ok(());
                }
                Object::EnumMember {
                    name: member_name,
                    value,
                } => {
                    let value = match name {
                        "name" => Value::String(member_name),
                        "value" => value,
                        _ => return Err(format!("enum member has no attribute {name:?}")),
                    };
                    self.stack.push(value);
                    return Ok(());
                }
                Object::Instance { class, attributes } => {
                    if let Some(value) = attributes.get(name).cloned() {
                        self.stack.push(value);
                        return Ok(());
                    }
                    let Object::Class { attributes, .. } = self.state.heap.get(class)? else {
                        return Err("instance has an invalid class".into());
                    };
                    if attributes.contains_key("__shellsim_unittest__") {
                        let method = match name {
                            "assertEqual" => Some(Method::UnitTestAssertEqual),
                            "assertTrue" => Some(Method::UnitTestAssertTrue),
                            "assertFalse" => Some(Method::UnitTestAssertFalse),
                            "assertIsNone" => Some(Method::UnitTestAssertIsNone),
                            "assertRaises" => Some(Method::UnitTestAssertRaises),
                            _ => None,
                        };
                        if let Some(method) = method {
                            let method = self.state.heap.allocate(
                                Object::BoundMethod {
                                    receiver: owner,
                                    method,
                                },
                                &mut self.interp.resources,
                            )?;
                            self.stack.push(method);
                            return Ok(());
                        }
                    }
                    let value = attributes
                        .get(name)
                        .cloned()
                        .ok_or_else(|| format!("instance has no attribute {name:?}"))?;
                    if let Value::Object(function) = value {
                        if matches!(self.state.heap.get(function)?, Object::Function { .. }) {
                            let bound = self.allocate_object(Object::PythonBoundMethod {
                                receiver: owner,
                                function,
                            })?;
                            self.stack.push(bound);
                            return Ok(());
                        }
                    }
                    self.stack.push(value);
                    return Ok(());
                }
                Object::Match { .. } => {}
                Object::ArgumentParser { prog, .. } => {
                    if name == "prog" {
                        self.stack.push(Value::String(prog));
                        return Ok(());
                    }
                }
                Object::Namespace { values } => {
                    let value = values
                        .iter()
                        .find(|(key, _)| key == name)
                        .map(|(_, value)| value.clone())
                        .ok_or_else(|| format!("namespace has no attribute {name:?}"))?;
                    self.stack.push(value);
                    return Ok(());
                }
                _ => {}
            }
        }
        let value = match (owner, name) {
            (Value::Native(NativeValue::Module(Module::Sys)), "version") => {
                Value::String("3.14.0 (shellsim)".into())
            }
            (Value::Native(NativeValue::Module(Module::Sys)), "version_info") => {
                Value::String("(3, 14, 0, 'final', 0)".into())
            }
            (Value::Native(NativeValue::Module(Module::Sys)), "executable") => {
                Value::String("/usr/bin/python3.14".into())
            }
            (Value::Native(NativeValue::Module(Module::Sys)), "prefix") => {
                Value::String("/usr".into())
            }
            (Value::Native(NativeValue::Module(Module::Sys)), "argv") => self.state.heap.allocate(
                Object::List(self.argv.iter().cloned().map(Value::String).collect()),
                &mut self.interp.resources,
            )?,
            (Value::Native(NativeValue::Module(Module::Sys)), "stdout") => {
                Value::Native(NativeValue::Stream(Stream::Stdout))
            }
            (Value::Native(NativeValue::Module(Module::Sys)), "stderr") => {
                Value::Native(NativeValue::Stream(Stream::Stderr))
            }
            (Value::Native(NativeValue::Module(Module::Os)), "getenv") => {
                Value::Native(NativeValue::Function(Builtin::GetEnv))
            }
            (Value::Native(NativeValue::Module(Module::Os)), "environ") => {
                Value::Native(NativeValue::Environment)
            }
            (Value::Native(NativeValue::Module(Module::Json)), "dumps") => {
                Value::Native(NativeValue::Function(Builtin::JsonDumps))
            }
            (Value::Native(NativeValue::Module(Module::Collections)), "defaultdict") => {
                Value::Native(NativeValue::Function(Builtin::DefaultDict))
            }
            (Value::Native(NativeValue::Module(Module::Math)), name) => {
                if let Some(value) = super::stdlib::math::constant(name) {
                    match value {
                        super::stdlib::math::MathValue::Float(value) => Value::Float(value),
                        super::stdlib::math::MathValue::Int(value) => Value::Int(value),
                        super::stdlib::math::MathValue::Bool(value) => Value::Bool(value),
                    }
                } else {
                    let function = match name {
                        "ceil" => MathBuiltin::Ceil,
                        "exp" => MathBuiltin::Exp,
                        "isinf" => MathBuiltin::IsInf,
                        "isnan" => MathBuiltin::IsNan,
                        "log" => MathBuiltin::Log,
                        "sin" => MathBuiltin::Sin,
                        "sqrt" => MathBuiltin::Sqrt,
                        _ => return Err(format!("math has no attribute {name:?}")),
                    };
                    Value::Native(NativeValue::Function(Builtin::Math(function)))
                }
            }
            (Value::Native(NativeValue::Module(Module::String)), name) => {
                let value = super::stdlib::string::constant(name)
                    .ok_or_else(|| format!("string has no attribute {name:?}"))?;
                Value::String(value.into())
            }
            (Value::Native(NativeValue::Module(Module::Itertools)), "count") => {
                Value::Native(NativeValue::Function(Builtin::ItertoolsCount))
            }
            (Value::Native(NativeValue::Module(Module::Itertools)), "islice") => {
                Value::Native(NativeValue::Function(Builtin::ItertoolsIslice))
            }
            (Value::Native(NativeValue::Module(Module::Heapq)), "heapify") => {
                Value::Native(NativeValue::Function(Builtin::HeapqHeapify))
            }
            (Value::Native(NativeValue::Module(Module::Heapq)), "heappop") => {
                Value::Native(NativeValue::Function(Builtin::HeapqHeappop))
            }
            (Value::Native(NativeValue::Module(Module::Bisect)), "bisect_left") => {
                Value::Native(NativeValue::Function(Builtin::BisectLeft))
            }
            (Value::Native(NativeValue::Module(Module::Functools)), "reduce") => {
                Value::Native(NativeValue::Function(Builtin::Reduce))
            }
            (Value::Native(NativeValue::Module(Module::Typing)), "List") => {
                Value::Native(NativeValue::TypingList)
            }
            (Value::Native(NativeValue::Module(Module::Re)), "IGNORECASE") => Value::Int(2),
            (Value::Native(NativeValue::Module(Module::Re)), "MULTILINE") => Value::Int(8),
            (Value::Native(NativeValue::Module(Module::Re)), "compile") => {
                Value::Native(NativeValue::Function(Builtin::ReCompile))
            }
            (Value::Native(NativeValue::Module(Module::Re)), "search") => {
                Value::Native(NativeValue::Function(Builtin::ReSearch))
            }
            (Value::Native(NativeValue::Module(Module::Re)), "match") => {
                Value::Native(NativeValue::Function(Builtin::ReMatch))
            }
            (Value::Native(NativeValue::Module(Module::Re)), "fullmatch") => {
                Value::Native(NativeValue::Function(Builtin::ReFullMatch))
            }
            (Value::Native(NativeValue::Module(Module::Re)), "findall") => {
                Value::Native(NativeValue::Function(Builtin::ReFindAll))
            }
            (Value::Native(NativeValue::Module(Module::Re)), "finditer") => {
                Value::Native(NativeValue::Function(Builtin::ReFindIter))
            }
            (Value::Native(NativeValue::Module(Module::Re)), "sub") => {
                Value::Native(NativeValue::Function(Builtin::ReSub))
            }
            (Value::Native(NativeValue::Module(Module::Re)), "escape") => {
                Value::Native(NativeValue::Function(Builtin::ReEscape))
            }
            (Value::Native(NativeValue::Module(Module::Argparse)), "ArgumentParser") => {
                Value::Native(NativeValue::Function(Builtin::ArgparseArgumentParser))
            }
            (Value::Native(NativeValue::Module(Module::Argparse)), "Namespace") => {
                Value::Native(NativeValue::Function(Builtin::ArgparseNamespace))
            }
            (Value::Native(NativeValue::Module(Module::Dataclasses)), "dataclass") => {
                Value::Native(NativeValue::Function(Builtin::Dataclass))
            }
            (Value::Native(NativeValue::Module(Module::Dataclasses)), "fields") => {
                return Err("dataclasses.fields is not implemented".into())
            }
            (Value::Native(NativeValue::Module(Module::Enum)), "Enum") => {
                Value::Native(NativeValue::EnumBase)
            }
            (Value::Native(NativeValue::Module(Module::Pytest)), "fail") => {
                Value::Native(NativeValue::Function(Builtin::PytestFail))
            }
            (Value::Native(NativeValue::Module(Module::Pytest)), "skip") => {
                Value::Native(NativeValue::Function(Builtin::PytestSkip))
            }
            (Value::Native(NativeValue::Module(Module::Pytest)), "raises") => {
                Value::Native(NativeValue::Function(Builtin::PytestRaises))
            }
            (Value::Native(NativeValue::Module(Module::Unittest)), "TestCase") => {
                Value::Native(NativeValue::UnitTestBase)
            }
            (Value::Native(NativeValue::Module(Module::Time)), name) => {
                let function = match name {
                    "time" => Builtin::Time,
                    "time_ns" => Builtin::TimeNs,
                    "monotonic" | "perf_counter" => Builtin::Monotonic,
                    "monotonic_ns" | "perf_counter_ns" => Builtin::MonotonicNs,
                    "process_time" => Builtin::ProcessTime,
                    "process_time_ns" => Builtin::ProcessTimeNs,
                    "sleep" => Builtin::Sleep,
                    _ => return Err(format!("time has no attribute {name:?}")),
                };
                Value::Native(NativeValue::Function(function))
            }
            (Value::Native(NativeValue::UnitTestBase), "__name__") => {
                Value::String("TestCase".into())
            }
            (Value::Native(NativeValue::Stream(stream)), "write") => {
                Value::Native(NativeValue::Function(Builtin::Write(stream)))
            }
            (Value::Native(NativeValue::Environment), "get") => {
                Value::Native(NativeValue::Function(Builtin::EnvironmentGet))
            }
            _ => return Err(format!("attribute {name:?} is not implemented")),
        };
        self.stack.push(value);
        Ok(())
    }

    fn store_attribute(&mut self, owner: Value, name: &str, value: Value) -> Result<(), String> {
        let Value::Object(id) = owner else {
            return Err("object does not support attribute assignment".into());
        };
        let is_new = match self.state.heap.get(id)? {
            Object::Instance { attributes, .. } => !attributes.contains_key(name),
            _ => return Err("object does not support attribute assignment".into()),
        };
        if is_new {
            self.state
                .heap
                .reserve_growth(48, &mut self.interp.resources)?;
        }
        let Object::Instance { attributes, .. } = self.state.heap.get_mut(id)? else {
            unreachable!()
        };
        attributes.insert(name.to_string(), value);
        Ok(())
    }

    fn method_for(&self, owner: &Value, name: &str) -> Result<Option<Method>, String> {
        let method = match owner {
            Value::String(_) => match name {
                "strip" => Some(Method::StringStrip),
                "lstrip" => Some(Method::StringLStrip),
                "rstrip" => Some(Method::StringRStrip),
                "startswith" => Some(Method::StringStartsWith),
                "endswith" => Some(Method::StringEndsWith),
                "split" => Some(Method::StringSplit),
                _ => None,
            },
            Value::Object(id) => match self.state.heap.get(*id)? {
                Object::List(_) => match name {
                    "append" => Some(Method::ListAppend),
                    "extend" => Some(Method::ListExtend),
                    "pop" => Some(Method::ListPop),
                    "remove" => Some(Method::ListRemove),
                    "sort" => Some(Method::ListSort),
                    _ => None,
                },
                Object::Dict(_) | Object::DefaultDict { .. } => match name {
                    "get" => Some(Method::DictGet),
                    "keys" => Some(Method::DictKeys),
                    "values" => Some(Method::DictValues),
                    "items" => Some(Method::DictItems),
                    "setdefault" => Some(Method::DictSetDefault),
                    _ => None,
                },
                Object::Set(_) => match name {
                    "add" => Some(Method::SetAdd),
                    "update" => Some(Method::SetUpdate),
                    "remove" => Some(Method::SetRemove),
                    "discard" => Some(Method::SetDiscard),
                    _ => None,
                },
                Object::Regex { .. } => match name {
                    "search" => Some(Method::RegexSearch),
                    "match" => Some(Method::RegexMatch),
                    "fullmatch" => Some(Method::RegexFullMatch),
                    "findall" => Some(Method::RegexFindAll),
                    "finditer" => Some(Method::RegexFindIter),
                    "sub" => Some(Method::RegexSub),
                    _ => None,
                },
                Object::Match { .. } => match name {
                    "group" => Some(Method::MatchGroup),
                    "groups" => Some(Method::MatchGroups),
                    "start" => Some(Method::MatchStart),
                    "end" => Some(Method::MatchEnd),
                    _ => None,
                },
                Object::ArgumentParser { .. } => match name {
                    "add_argument" => Some(Method::ArgumentParserAddArgument),
                    "parse_args" => Some(Method::ArgumentParserParseArgs),
                    _ => None,
                },
                Object::RaisesContext { .. } => match name {
                    "__enter__" => Some(Method::RaisesEnter),
                    "__exit__" => Some(Method::RaisesExit),
                    _ => None,
                },
                Object::Tuple(_)
                | Object::Function { .. }
                | Object::Class { .. }
                | Object::Instance { .. }
                | Object::PythonBoundMethod { .. }
                | Object::Iterator { .. }
                | Object::CountIterator { .. }
                | Object::Generator { .. }
                | Object::BoundMethod { .. }
                | Object::Module { .. }
                | Object::Namespace { .. } => None,
                Object::EnumMember { .. } => None,
            },
            _ => None,
        };
        Ok(method)
    }

    fn load_subscript(&mut self) -> Result<(), String> {
        let index = self.pop()?;
        let owner = self.pop()?;
        let value = match owner {
            Value::Native(NativeValue::TypingList) => {
                let parameter = match index {
                    Value::Native(NativeValue::Function(Builtin::Integer)) => "int".to_string(),
                    Value::Native(NativeValue::Function(Builtin::String)) => "str".to_string(),
                    other => protocol::repr(&self.state.heap, &other)?,
                };
                Value::String(format!("typing.List[{parameter}]"))
            }
            Value::String(value) => {
                let index = index.as_int().ok_or("string index must be an integer")?;
                let chars: Vec<char> = value.chars().collect();
                let len = chars.len() as i64;
                let index = if index < 0 { len + index } else { index };
                Value::String(
                    chars
                        .get(usize::try_from(index).map_err(|_| "string index out of range")?)
                        .ok_or("string index out of range")?
                        .to_string(),
                )
            }
            Value::Object(id) => match self.state.heap.get(id)?.clone() {
                Object::List(values) | Object::Tuple(values) => {
                    let index = index.as_int().ok_or("sequence index must be an integer")?;
                    let len = values.len() as i64;
                    let index = if index < 0 { len + index } else { index };
                    values
                        .get(usize::try_from(index).map_err(|_| "index out of range")?)
                        .cloned()
                        .ok_or("index out of range")?
                }
                Object::Dict(entries) => {
                    let mut found = None;
                    for (key, value) in &entries {
                        self.charge_cpu(1)?;
                        if protocol::equals(&self.state.heap, key, &index)? {
                            found = Some(value.clone());
                            break;
                        }
                    }
                    found.ok_or("key not found")?
                }
                Object::DefaultDict { factory, entries } => {
                    let mut found = None;
                    for (key, value) in &entries {
                        self.charge_cpu(1)?;
                        if protocol::equals(&self.state.heap, key, &index)? {
                            found = Some(value.clone());
                            break;
                        }
                    }
                    if let Some(value) = found {
                        value
                    } else {
                        self.stack.push(factory);
                        let value = match self.call(0, &[], &[])? {
                            CallResult::Value(value) => value,
                            CallResult::Exit(status) => {
                                return Err(format!("default factory exited with status {status}"))
                            }
                        };
                        self.state
                            .heap
                            .reserve_growth(48, &mut self.interp.resources)?;
                        let Object::DefaultDict { entries, .. } = self.state.heap.get_mut(id)?
                        else {
                            unreachable!()
                        };
                        entries.push((index, value.clone()));
                        value
                    }
                }
                Object::Set(_) => return Err("set object is not subscriptable".into()),
                Object::Function { .. }
                | Object::Class { .. }
                | Object::Instance { .. }
                | Object::PythonBoundMethod { .. }
                | Object::Iterator { .. }
                | Object::CountIterator { .. }
                | Object::Generator { .. }
                | Object::BoundMethod { .. }
                | Object::Module { .. }
                | Object::Regex { .. }
                | Object::Match { .. }
                | Object::ArgumentParser { .. }
                | Object::Namespace { .. } => return Err("object is not subscriptable".into()),
                Object::EnumMember { .. } => return Err("object is not subscriptable".into()),
                Object::RaisesContext { .. } => return Err("object is not subscriptable".into()),
            },
            _ => return Err("object is not subscriptable".into()),
        };
        self.stack.push(value);
        Ok(())
    }

    fn store_subscript(&mut self) -> Result<(), String> {
        let index = self.pop()?;
        let owner = self.pop()?;
        let value = self.pop()?;
        let Value::Object(id) = owner else {
            return Err("object does not support item assignment".into());
        };
        match self.state.heap.get(id)?.clone() {
            Object::List(values) => {
                let index = index.as_int().ok_or("list index must be an integer")?;
                let len = values.len() as i64;
                let index = if index < 0 { len + index } else { index };
                let index =
                    usize::try_from(index).map_err(|_| "list assignment index out of range")?;
                let Object::List(values) = self.state.heap.get_mut(id)? else {
                    unreachable!()
                };
                let slot = values
                    .get_mut(index)
                    .ok_or("list assignment index out of range")?;
                *slot = value;
            }
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                let mut found = None;
                for (position, (candidate, _)) in entries.iter().enumerate() {
                    self.charge_cpu(1)?;
                    if protocol::equals(&self.state.heap, candidate, &index)? {
                        found = Some(position);
                        break;
                    }
                }
                if let Some(position) = found {
                    let entries = match self.state.heap.get_mut(id)? {
                        Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
                        _ => unreachable!(),
                    };
                    entries[position].1 = value;
                } else {
                    self.state
                        .heap
                        .reserve_growth(48, &mut self.interp.resources)?;
                    let entries = match self.state.heap.get_mut(id)? {
                        Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
                        _ => unreachable!(),
                    };
                    entries.push((index, value));
                }
            }
            Object::Tuple(_) => return Err("tuple object does not support item assignment".into()),
            Object::Set(_)
            | Object::Function { .. }
            | Object::Class { .. }
            | Object::Instance { .. }
            | Object::PythonBoundMethod { .. }
            | Object::Iterator { .. }
            | Object::CountIterator { .. }
            | Object::Generator { .. }
            | Object::BoundMethod { .. }
            | Object::Module { .. }
            | Object::Regex { .. }
            | Object::Match { .. }
            | Object::ArgumentParser { .. }
            | Object::Namespace { .. } => {
                return Err("object does not support item assignment".into())
            }
            Object::EnumMember { .. } => {
                return Err("object does not support item assignment".into())
            }
            Object::RaisesContext { .. } => {
                return Err("object does not support item assignment".into())
            }
        }
        Ok(())
    }

    fn make_function(
        &mut self,
        name: String,
        code: Code,
        default_count: usize,
    ) -> Result<(), String> {
        if self.stack.len() < default_count {
            return Err("invalid bytecode stack effect while creating function".into());
        }
        let defaults = self.stack.split_off(self.stack.len() - default_count);
        let mut closure = self.local_scopes.last().copied();
        if closure.is_some() && closure == self.class_scopes.last().copied() {
            closure = self
                .state
                .heap
                .scope_parent(closure.expect("checked above"))?;
        }
        let function = self.state.heap.allocate(
            Object::Function {
                name,
                code,
                closure,
                defaults,
            },
            &mut self.interp.resources,
        )?;
        self.stack.push(function);
        Ok(())
    }

    fn make_class(
        &mut self,
        name: String,
        code: &Code,
        base_count: usize,
        fields: &[ClassField],
    ) -> Result<(), String> {
        if self.stack.len() < base_count {
            return Err("invalid bytecode stack effect while creating class".into());
        }
        let bases = self.stack.split_off(self.stack.len() - base_count);
        let is_enum = bases.len() == 1 && matches!(bases[0], Value::Native(NativeValue::EnumBase));
        let is_unittest =
            bases.len() == 1 && matches!(bases[0], Value::Native(NativeValue::UnitTestBase));
        if !bases.is_empty() && !is_enum && !is_unittest {
            return Err("only enum.Enum and unittest.TestCase inheritance is supported".into());
        }
        let parent = self.local_scopes.last().copied();
        let uses_repl_globals = parent
            .map(|scope| self.state.heap.scope_uses_repl_globals(scope))
            .transpose()?
            .unwrap_or(true);
        let scope = self.state.heap.allocate_scope(
            parent,
            uses_repl_globals,
            HashMap::new(),
            &mut self.interp.resources,
        )?;
        let outer_stack = std::mem::take(&mut self.stack);
        self.local_scopes.push(scope);
        self.class_scopes.push(scope);
        self.class_bindings.push(Vec::new());
        let execution = self.execute_code(code);
        let bindings = self
            .class_bindings
            .pop()
            .expect("class binding stack is present");
        self.class_scopes.pop();
        self.local_scopes.pop();
        self.stack = outer_stack;
        match execution {
            Ok(Execution::Halt) => {}
            Ok(Execution::Return(_)) => return Err("'return' outside function".into()),
            Ok(Execution::Yield(_, _)) => return Err("'yield' outside function".into()),
            Ok(Execution::Exit(status)) => {
                return Err(format!("class body exited with status {status}"))
            }
            Err((error, span)) => {
                return Err(format!(
                    "{error} in class {name} at line {}, column {}",
                    span.line, span.column
                ))
            }
        }
        let mut attributes = self.state.heap.scope_values(scope)?;
        if is_unittest {
            attributes.insert("__shellsim_unittest__".into(), Value::Bool(true));
        }
        let dataclass_fields = fields
            .iter()
            .map(|field| (field.name.clone(), attributes.get(&field.name).cloned()))
            .collect::<Vec<_>>();
        let mut enum_members = Vec::new();
        if is_enum {
            for member_name in bindings {
                if member_name.starts_with('_') {
                    continue;
                }
                let Some(value) = attributes.get(&member_name).cloned() else {
                    continue;
                };
                if matches!(value, Value::Object(id) if matches!(self.state.heap.get(id)?, Object::Function { .. }))
                {
                    continue;
                }
                let member = self.allocate_object(Object::EnumMember {
                    name: member_name.clone(),
                    value,
                })?;
                attributes.insert(member_name, member.clone());
                enum_members.push(member);
            }
        }
        let class = self.state.heap.allocate(
            Object::Class {
                name,
                attributes,
                is_dataclass: false,
                dataclass_fields,
                enum_members,
            },
            &mut self.interp.resources,
        )?;
        self.stack.push(class);
        Ok(())
    }

    fn get_iterator(&mut self) -> Result<(), String> {
        let iterable = self.pop()?;
        if let Value::Object(id) = iterable {
            match self.state.heap.get(id)? {
                Object::CountIterator { .. } | Object::Generator { .. } => {
                    // These iterators remain lazy; materializing either one here would permit an
                    // unbounded host allocation before the caller's loop can meter each item.
                    self.stack.push(Value::Object(id));
                    return Ok(());
                }
                _ => {}
            }
        }
        let values = self.iterable_values(&iterable)?;
        let iterator = self.state.heap.allocate(
            Object::Iterator {
                values,
                position: 0,
            },
            &mut self.interp.resources,
        )?;
        self.stack.push(iterator);
        Ok(())
    }

    fn unpack_sequence(
        &mut self,
        expected: usize,
        star_index: Option<usize>,
    ) -> Result<(), String> {
        let value = self.pop()?;
        let values = self.iterable_values(&value)?;
        let mut outputs = Vec::new();
        if let Some(star_index) = star_index {
            if star_index >= expected || values.len() < expected.saturating_sub(1) {
                return Err(format!(
                    "not enough values to unpack (expected at least {}, got {})",
                    expected.saturating_sub(1),
                    values.len()
                ));
            }
            let tail_start = star_index;
            let tail_end = values.len() - (expected - star_index - 1);
            for index in 0..expected {
                if index == star_index {
                    let mut tail = Vec::new();
                    for value in values[tail_start..tail_end].iter().cloned() {
                        self.push_materialized(&mut tail, value)?;
                    }
                    outputs.push(self.allocate_object(Object::List(tail))?);
                } else {
                    let source_index = if index < star_index {
                        index
                    } else {
                        tail_end + (index - star_index - 1)
                    };
                    self.push_materialized(&mut outputs, values[source_index].clone())?;
                }
            }
        } else {
            if values.len() != expected {
                return Err(format!(
                    "cannot unpack sequence of length {} into {expected} targets",
                    values.len()
                ));
            }
            for value in values {
                self.push_materialized(&mut outputs, value)?;
            }
        }
        // Store operations pop their input, so the leftmost target must be on top.
        self.stack.extend(outputs.into_iter().rev());
        Ok(())
    }

    /// Advance the iterator kept at the top of the operand stack. The iterator remains below the
    /// yielded value until exhaustion, which gives `for` a small and explicit stack contract.
    fn for_iterator(&mut self) -> Result<bool, String> {
        let Value::Object(id) = self
            .stack
            .last()
            .cloned()
            .ok_or("invalid bytecode stack effect")?
        else {
            return Err("for-loop stack does not contain an iterator".into());
        };
        let next = match self.state.heap.get_mut(id)? {
            Object::Iterator { values, position } => {
                let next = values.get(*position).cloned();
                if next.is_some() {
                    *position += 1;
                }
                next
            }
            Object::CountIterator { current, step } => {
                let value = *current;
                *current =
                    super::stdlib::itertools::count_next(value, *step).map_err(str::to_string)?;
                Some(Value::Int(value))
            }
            Object::Generator { .. } => match self.resume_generator(id)? {
                Some(value) => {
                    self.stack.push(value);
                    return Ok(true);
                }
                None => {
                    self.stack.pop();
                    return Ok(false);
                }
            },
            _ => return Err("for-loop stack does not contain an iterator".into()),
        };
        if let Some(value) = next {
            self.stack.push(value);
            Ok(true)
        } else {
            self.stack.pop();
            Ok(false)
        }
    }

    /// Resume one generator frame until its next yield or terminal return. A generator's operand
    /// stack is kept separate from its caller's stack, while its lexical scope remains in the
    /// shared heap so closures and mutations preserve normal Python aliasing.
    fn resume_generator(&mut self, id: super::heap::ObjectId) -> Result<Option<Value>, String> {
        const MAX_GENERATOR_DEPTH: usize = 256;
        if self.call_depth >= MAX_GENERATOR_DEPTH {
            return Err("maximum recursion depth exceeded".into());
        }
        let (code, scope, instruction_pointer, mut handlers, frame_stack, exhausted, running) =
            match self.state.heap.get(id)?.clone() {
                Object::Generator {
                    code,
                    scope,
                    instruction_pointer,
                    handlers,
                    stack,
                    exhausted,
                    running,
                    ..
                } => (
                    code,
                    scope,
                    instruction_pointer,
                    handlers,
                    stack,
                    exhausted,
                    running,
                ),
                _ => return Err("object is not a generator".into()),
            };
        if exhausted {
            return Ok(None);
        }
        if running {
            return Err("generator already executing".into());
        }
        if let Object::Generator { running, .. } = self.state.heap.get_mut(id)? {
            *running = true;
        }
        let outer_stack = std::mem::take(&mut self.stack);
        self.stack = frame_stack;
        if instruction_pointer != 0 {
            // `next()` resumes a yield expression with None in this slice (send(value) is not
            // exposed yet), which also supplies the value popped by a yield statement's wrapper.
            self.stack.push(Value::None);
        }
        self.local_scopes.push(scope);
        self.call_depth += 1;
        let result = self.execute_code_from(&code, instruction_pointer, &mut handlers);
        self.call_depth -= 1;
        self.local_scopes.pop();
        let frame_result_stack = std::mem::take(&mut self.stack);
        self.stack = outer_stack;

        match result {
            Ok(Execution::Yield(value, next_instruction)) => {
                if let Object::Generator {
                    instruction_pointer,
                    handlers: saved_handlers,
                    stack: saved_stack,
                    running,
                    ..
                } = self.state.heap.get_mut(id)?
                {
                    *instruction_pointer = next_instruction;
                    *saved_handlers = handlers;
                    *saved_stack = frame_result_stack;
                    *running = false;
                }
                Ok(Some(value))
            }
            Ok(Execution::Return(_)) | Ok(Execution::Halt) | Ok(Execution::Exit(_)) => {
                if let Object::Generator {
                    exhausted, running, ..
                } = self.state.heap.get_mut(id)?
                {
                    *exhausted = true;
                    *running = false;
                }
                Ok(None)
            }
            Err((error, span)) => {
                if let Object::Generator {
                    exhausted, running, ..
                } = self.state.heap.get_mut(id)?
                {
                    *exhausted = true;
                    *running = false;
                }
                Err(format!(
                    "{error} at line {}, column {}",
                    span.line, span.column
                ))
            }
        }
    }

    fn unary(&mut self, operator: UnaryOperator) -> Result<(), String> {
        let value = self.pop()?;
        let value = match (operator, value) {
            (UnaryOperator::Not, value) => Value::Bool(!protocol::truth(&self.state.heap, &value)?),
            (UnaryOperator::Positive, Value::Int(value)) => Value::Int(value),
            (UnaryOperator::Negative, Value::Int(value)) => Value::Int(
                value
                    .checked_neg()
                    .ok_or("integer arithmetic exceeds the current bounded integer range")?,
            ),
            (UnaryOperator::Positive, Value::Float(value)) => Value::Float(value),
            (UnaryOperator::Negative, Value::Float(value)) => Value::Float(-value),
            _ => return Err("bad operand type for unary arithmetic".into()),
        };
        self.stack.push(value);
        Ok(())
    }

    fn build_sequence(&mut self, count: usize, kind: SequenceKind) -> Result<(), String> {
        let values = self.take(count)?;
        let object = match kind {
            SequenceKind::List => Object::List(values),
            SequenceKind::Tuple => Object::Tuple(values),
        };
        let value = self
            .state
            .heap
            .allocate(object, &mut self.interp.resources)?;
        self.stack.push(value);
        Ok(())
    }

    fn build_dict(&mut self, count: usize) -> Result<(), String> {
        let values = self.take(count.checked_mul(2).ok_or("dictionary is too large")?)?;
        let mut entries: Vec<(Value, Value)> = Vec::with_capacity(count);
        for pair in values.chunks_exact(2) {
            let key = pair[0].clone();
            let value = pair[1].clone();
            let mut replaced = false;
            for (existing_key, existing_value) in &mut entries {
                if protocol::equals(&self.state.heap, existing_key, &key)? {
                    *existing_value = value.clone();
                    replaced = true;
                    break;
                }
            }
            if !replaced {
                entries.push((key, value));
            }
        }
        let value = self
            .state
            .heap
            .allocate(Object::Dict(entries), &mut self.interp.resources)?;
        self.stack.push(value);
        Ok(())
    }

    fn build_set(&mut self, count: usize) -> Result<(), String> {
        let candidates = self.take(count)?;
        let mut values = Vec::with_capacity(count);
        for candidate in candidates {
            let mut exists = false;
            for value in &values {
                if protocol::equals(&self.state.heap, value, &candidate)? {
                    exists = true;
                    break;
                }
            }
            if !exists {
                values.push(candidate);
            }
        }
        let value = self
            .state
            .heap
            .allocate(Object::Set(values), &mut self.interp.resources)?;
        self.stack.push(value);
        Ok(())
    }

    fn compare(&mut self, operator: ComparisonOperator) -> Result<(), String> {
        let right = self.pop()?;
        let left = self.pop()?;
        let result = match operator {
            ComparisonOperator::Equal => protocol::equals(&self.state.heap, &left, &right)?,
            ComparisonOperator::NotEqual => !protocol::equals(&self.state.heap, &left, &right)?,
            ComparisonOperator::Less => {
                protocol::compare(&self.state.heap, &left, &right)? == Ordering::Less
            }
            ComparisonOperator::LessEqual => {
                protocol::compare(&self.state.heap, &left, &right)? != Ordering::Greater
            }
            ComparisonOperator::Greater => {
                protocol::compare(&self.state.heap, &left, &right)? == Ordering::Greater
            }
            ComparisonOperator::GreaterEqual => {
                protocol::compare(&self.state.heap, &left, &right)? != Ordering::Less
            }
            ComparisonOperator::In => protocol::contains(&self.state.heap, &right, &left)?,
            ComparisonOperator::NotIn => !protocol::contains(&self.state.heap, &right, &left)?,
            ComparisonOperator::Is => identity(&left, &right),
            ComparisonOperator::IsNot => !identity(&left, &right),
        };
        self.stack.push(Value::Bool(result));
        Ok(())
    }

    fn binary(&mut self, operator: BinaryOperator) -> Result<(), String> {
        let right = self.pop()?;
        let left = self.pop()?;
        let value = match (operator, left, right) {
            (BinaryOperator::Add, Value::Object(left_id), Value::Object(right_id)) => {
                let left_values = match self.state.heap.get(left_id)?.clone() {
                    Object::List(values) => values,
                    Object::Tuple(values) => values,
                    _ => return Err("can only concatenate list or tuple sequences".into()),
                };
                let right_values = match self.state.heap.get(right_id)?.clone() {
                    Object::List(values) => values,
                    Object::Tuple(values) => values,
                    _ => return Err("can only concatenate list or tuple sequences".into()),
                };
                let is_list = matches!(self.state.heap.get(left_id)?, Object::List(_));
                if is_list != matches!(self.state.heap.get(right_id)?, Object::List(_)) {
                    return Err("can only concatenate list (not tuple) to list".into());
                }
                let mut values = Vec::new();
                for value in left_values.into_iter().chain(right_values) {
                    self.push_materialized(&mut values, value)?;
                }
                self.allocate_object(if is_list {
                    Object::List(values)
                } else {
                    Object::Tuple(values)
                })?
            }
            (BinaryOperator::Multiply, Value::Object(id), Value::Int(count))
            | (BinaryOperator::Multiply, Value::Int(count), Value::Object(id)) => {
                let (is_list, values) = match self.state.heap.get(id)?.clone() {
                    Object::List(values) => (true, values),
                    Object::Tuple(values) => (false, values),
                    _ => return Err("can only multiply a sequence by an integer".into()),
                };
                let count =
                    usize::try_from(count.max(0)).map_err(|_| "sequence repeat is too large")?;
                let length = values
                    .len()
                    .checked_mul(count)
                    .ok_or("sequence repeat is too large")?;
                self.reserve_result(length.saturating_mul(64))?;
                let mut repeated = Vec::new();
                for _ in 0..count {
                    for value in values.iter().cloned() {
                        self.charge_cpu(1)?;
                        repeated.push(value);
                    }
                }
                self.allocate_object(if is_list {
                    Object::List(repeated)
                } else {
                    Object::Tuple(repeated)
                })?
            }
            (BinaryOperator::Add, Value::String(left), Value::String(right)) => {
                let bytes = left
                    .len()
                    .checked_add(right.len())
                    .ok_or("string result is too large")?;
                self.reserve_result(bytes)?;
                Value::String(left + &right)
            }
            (BinaryOperator::Multiply, Value::String(value), Value::Int(count))
            | (BinaryOperator::Multiply, Value::Int(count), Value::String(value)) => {
                let count =
                    usize::try_from(count.max(0)).map_err(|_| "string repeat is too large")?;
                let bytes = value
                    .len()
                    .checked_mul(count)
                    .ok_or("string result is too large")?;
                self.reserve_result(bytes)?;
                Value::String(value.repeat(count))
            }
            (operator, Value::Int(left), Value::Int(right)) => {
                integer_binary(operator, left, right)?
            }
            (operator, left, right) => {
                let left = as_float(&left).ok_or("unsupported arithmetic operands")?;
                let right = as_float(&right).ok_or("unsupported arithmetic operands")?;
                float_binary(operator, left, right)?
            }
        };
        self.stack.push(value);
        Ok(())
    }

    fn call(
        &mut self,
        positional: usize,
        keyword_names: &[String],
        starred: &[bool],
    ) -> Result<CallResult, String> {
        let count = positional
            .checked_add(keyword_names.len())
            .ok_or("too many call arguments")?;
        if starred.len() != count {
            return Err("invalid bytecode call argument metadata".into());
        }
        if self.stack.len() < count + 1 {
            return Err("invalid bytecode stack effect".into());
        }
        let mut raw_arguments = self.stack.split_off(self.stack.len() - count);
        let keyword_values = raw_arguments.split_off(positional);
        if starred[..positional].iter().any(|expanded| *expanded)
            && keyword_names
                .iter()
                .zip(&starred[positional..])
                .any(|(_, expanded)| *expanded)
        {
            return Err("invalid starred keyword argument metadata".into());
        }
        let positional_starred = &starred[..positional];
        let mut arguments = Vec::new();
        for (argument, expanded) in raw_arguments.into_iter().zip(positional_starred) {
            if *expanded {
                for value in self.iterable_values(&argument)? {
                    self.push_materialized(&mut arguments, value)?;
                }
            } else {
                self.push_materialized(&mut arguments, argument)?;
            }
        }
        let keyword_arguments = keyword_names
            .iter()
            .cloned()
            .zip(keyword_values)
            .collect::<Vec<_>>();
        let function = self.pop()?;
        if let Value::Object(id) = function {
            return match self.state.heap.get(id)?.clone() {
                Object::Function {
                    name,
                    code,
                    closure,
                    defaults,
                } => self.call_python_function(
                    &name,
                    &code,
                    closure,
                    &defaults,
                    arguments,
                    keyword_arguments,
                ),
                Object::BoundMethod { receiver, method } => {
                    if !keyword_arguments.is_empty()
                        && !matches!(method, Method::ListSort | Method::ArgumentParserAddArgument)
                    {
                        return Err("method keyword arguments are not implemented".into());
                    }
                    self.call_method(receiver, method, arguments, keyword_arguments)
                }
                Object::PythonBoundMethod { receiver, function } => {
                    let Object::Function {
                        name,
                        code,
                        closure,
                        defaults,
                    } = self.state.heap.get(function)?.clone()
                    else {
                        return Err("bound method has an invalid function".into());
                    };
                    arguments.insert(0, receiver);
                    self.call_python_function(
                        &name,
                        &code,
                        closure,
                        &defaults,
                        arguments,
                        keyword_arguments,
                    )
                }
                Object::Class {
                    name,
                    attributes,
                    is_dataclass,
                    dataclass_fields,
                    enum_members,
                } => {
                    if !enum_members.is_empty() {
                        if !keyword_arguments.is_empty() || arguments.len() != 1 {
                            return Err(format!("{name}() expects one value"));
                        }
                        for member in enum_members {
                            let Object::EnumMember { value, .. } =
                                self.state.heap.get(match member.clone() {
                                    Value::Object(id) => id,
                                    _ => return Err("invalid enum member".into()),
                                })?
                            else {
                                return Err("invalid enum member".into());
                            };
                            if protocol::equals(&self.state.heap, value, &arguments[0])? {
                                return Ok(CallResult::Value(member));
                            }
                        }
                        return Err(format!("value is not a valid {name}"));
                    }
                    let instance = self.allocate_object(Object::Instance {
                        class: id,
                        attributes: HashMap::new(),
                    })?;
                    if is_dataclass {
                        let mut values = Vec::new();
                        for (index, (field, default)) in dataclass_fields.iter().enumerate() {
                            if index < arguments.len()
                                && keyword_arguments.iter().any(|(name, _)| name == field)
                            {
                                return Err(format!(
                                    "{name}() got multiple values for argument {field:?}"
                                ));
                            }
                            let value = keyword_arguments
                                .iter()
                                .find(|(name, _)| name == field)
                                .map(|(_, value)| value.clone())
                                .or_else(|| arguments.get(index).cloned())
                                .or_else(|| default.clone())
                                .ok_or_else(|| {
                                    format!("{name}() missing required argument: {field:?}")
                                })?;
                            if keyword_arguments
                                .iter()
                                .filter(|(name, _)| name == field)
                                .count()
                                > 1
                            {
                                return Err(format!(
                                    "{name}() got multiple values for argument {field:?}"
                                ));
                            }
                            values.push((field.clone(), value));
                        }
                        if arguments.len() > dataclass_fields.len() {
                            return Err(format!(
                                "{name}() takes {} positional arguments but {} were given",
                                dataclass_fields.len(),
                                arguments.len()
                            ));
                        }
                        for (field, _) in &keyword_arguments {
                            if !dataclass_fields.iter().any(|(name, _)| name == field) {
                                return Err(format!(
                                    "{name}() got an unexpected keyword argument {field:?}"
                                ));
                            }
                        }
                        let Object::Instance { attributes, .. } =
                            self.state.heap.get_mut(match instance {
                                Value::Object(id) => id,
                                _ => unreachable!(),
                            })?
                        else {
                            unreachable!()
                        };
                        attributes.extend(values);
                    } else if let Some(initializer) = attributes.get("__init__") {
                        let Value::Object(function) = initializer else {
                            return Err(format!("{name}.__init__ is not callable"));
                        };
                        let Object::Function {
                            name: function_name,
                            code,
                            closure,
                            defaults,
                        } = self.state.heap.get(*function)?.clone()
                        else {
                            return Err(format!("{name}.__init__ is not a function"));
                        };
                        arguments.insert(0, instance.clone());
                        match self.call_python_function(
                            &function_name,
                            &code,
                            closure,
                            &defaults,
                            arguments,
                            keyword_arguments,
                        )? {
                            CallResult::Value(Value::None) => {}
                            CallResult::Value(_) => {
                                return Err("__init__() should return None".into())
                            }
                            CallResult::Exit(status) => return Ok(CallResult::Exit(status)),
                        }
                    } else if !arguments.is_empty() || !keyword_arguments.is_empty() {
                        return Err(format!("{name}() takes no arguments"));
                    }
                    Ok(CallResult::Value(instance))
                }
                Object::Module { .. } => Err("module object is not callable".into()),
                _ => Err("object is not callable".into()),
            };
        }
        if let Value::Native(NativeValue::ExceptionType(exception_type)) = function {
            expect_arity(&arguments, 0, 1)?;
            let message = arguments
                .first()
                .map(|value| protocol::display(&self.state.heap, value))
                .transpose()?
                .unwrap_or_default();
            return Ok(CallResult::Value(Value::Exception {
                kind: exception_type.0.to_string(),
                message,
            }));
        }
        let Value::Native(NativeValue::Function(function)) = function else {
            return Err("object is not callable".into());
        };
        if !keyword_arguments.is_empty()
            && !matches!(
                function,
                Builtin::JsonDumps
                    | Builtin::Sorted
                    | Builtin::ReCompile
                    | Builtin::ReSearch
                    | Builtin::ReMatch
                    | Builtin::ReFullMatch
                    | Builtin::ReFindAll
                    | Builtin::ReFindIter
                    | Builtin::ReSub
                    | Builtin::ArgparseArgumentParser
                    | Builtin::ArgparseNamespace
                    | Builtin::Dataclass
            )
        {
            return Err("this builtin does not accept keyword arguments".into());
        }
        match function {
            Builtin::Print => {
                let text = arguments
                    .iter()
                    .map(|value| protocol::display(&self.state.heap, value))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" ");
                self.write_output(Stream::Stdout, text.as_bytes());
                self.write_output(Stream::Stdout, b"\n");
                Ok(CallResult::Value(Value::None))
            }
            Builtin::Exit => {
                expect_arity(&arguments, 0, 1)?;
                let status = arguments
                    .first()
                    .and_then(Value::as_int)
                    .unwrap_or_default();
                Ok(CallResult::Exit(status as i32))
            }
            Builtin::String => {
                expect_arity(&arguments, 1, 1)?;
                Ok(CallResult::Value(Value::String(protocol::display(
                    &self.state.heap,
                    &arguments[0],
                )?)))
            }
            Builtin::Repr => {
                expect_arity(&arguments, 1, 1)?;
                Ok(CallResult::Value(Value::String(protocol::repr(
                    &self.state.heap,
                    &arguments[0],
                )?)))
            }
            Builtin::Integer => {
                expect_arity(&arguments, 0, 1)?;
                let value = arguments
                    .first()
                    .map(|value| value.as_int().ok_or("int() argument is not supported"))
                    .transpose()?
                    .unwrap_or(0);
                Ok(CallResult::Value(Value::Int(value)))
            }
            Builtin::Length => {
                expect_arity(&arguments, 1, 1)?;
                let length = match &arguments[0] {
                    Value::String(value) => value.chars().count(),
                    Value::Object(id) => match self.state.heap.get(*id)? {
                        Object::List(values) | Object::Tuple(values) | Object::Set(values) => {
                            values.len()
                        }
                        Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                            entries.len()
                        }
                        Object::Function { .. }
                        | Object::Class { .. }
                        | Object::Instance { .. }
                        | Object::PythonBoundMethod { .. }
                        | Object::Iterator { .. }
                        | Object::CountIterator { .. }
                        | Object::Generator { .. }
                        | Object::BoundMethod { .. }
                        | Object::Module { .. }
                        | Object::Regex { .. }
                        | Object::Match { .. }
                        | Object::ArgumentParser { .. }
                        | Object::Namespace { .. } => return Err("object has no len()".into()),
                        Object::EnumMember { .. } => return Err("object has no len()".into()),
                        Object::RaisesContext { .. } => return Err("object has no len()".into()),
                    },
                    _ => return Err("object has no len()".into()),
                };
                Ok(CallResult::Value(Value::Int(length as i64)))
            }
            Builtin::Sorted => {
                expect_arity(&arguments, 1, 1)?;
                let values = self.iterable_values(&arguments[0])?;
                let mut key_function = None;
                let mut reverse = false;
                let mut saw_reverse = false;
                for (name, value) in keyword_arguments {
                    match name.as_str() {
                        "key" if key_function.is_none() => key_function = Some(value),
                        "reverse" if !saw_reverse => {
                            reverse = protocol::truth(&self.state.heap, &value)?;
                            saw_reverse = true;
                        }
                        "key" | "reverse" => {
                            return Err(format!(
                                "sorted() got multiple values for keyword {name:?}"
                            ))
                        }
                        _ => return Err(format!("sorted() got an unexpected keyword {name:?}")),
                    }
                }
                let mut keyed = Vec::new();
                for value in values {
                    let key = if let Some(function) = &key_function {
                        self.stack.push(function.clone());
                        self.stack.push(value.clone());
                        match self.call(1, &[], &[false])? {
                            CallResult::Value(key) => key,
                            CallResult::Exit(status) => return Ok(CallResult::Exit(status)),
                        }
                    } else {
                        value.clone()
                    };
                    self.reserve_result(64)?;
                    keyed.push((key, value));
                }
                // Stable insertion sort keeps comparison dispatch and failure order obvious.
                for index in 1..keyed.len() {
                    let mut current = index;
                    while current > 0 {
                        self.charge_cpu(1)?;
                        if protocol::compare(
                            &self.state.heap,
                            &keyed[current].0,
                            &keyed[current - 1].0,
                        )? != if reverse {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        } {
                            break;
                        }
                        keyed.swap(current, current - 1);
                        current -= 1;
                    }
                }
                let values = keyed.into_iter().map(|(_, value)| value).collect();
                Ok(CallResult::Value(
                    self.allocate_object(Object::List(values))?,
                ))
            }
            Builtin::Minimum | Builtin::Maximum => {
                if arguments.is_empty() {
                    return Err("expected at least one argument".into());
                }
                let values = if arguments.len() == 1 {
                    self.iterable_values(&arguments[0])?
                } else {
                    arguments
                };
                let mut values = values.into_iter();
                let mut selected = values.next().ok_or("argument is an empty sequence")?;
                for value in values {
                    self.charge_cpu(1)?;
                    let ordering = protocol::compare(&self.state.heap, &value, &selected)?;
                    let replace = match function {
                        Builtin::Minimum => ordering == Ordering::Less,
                        Builtin::Maximum => ordering == Ordering::Greater,
                        _ => unreachable!(),
                    };
                    if replace {
                        selected = value;
                    }
                }
                Ok(CallResult::Value(selected))
            }
            Builtin::Sum => {
                expect_arity(&arguments, 1, 2)?;
                let values = self.iterable_values(&arguments[0])?;
                let mut total = arguments.get(1).cloned().unwrap_or(Value::Int(0));
                for value in values {
                    self.charge_cpu(1)?;
                    total = add_numbers(total, value)?;
                }
                Ok(CallResult::Value(total))
            }
            Builtin::Absolute => {
                expect_arity(&arguments, 1, 1)?;
                let value =
                    match &arguments[0] {
                        Value::Int(value) => Value::Int(value.checked_abs().ok_or(
                            "integer arithmetic exceeds the current bounded integer range",
                        )?),
                        Value::Float(value) => Value::Float(value.abs()),
                        _ => return Err("bad operand type for abs()".into()),
                    };
                Ok(CallResult::Value(value))
            }
            Builtin::Boolean => {
                expect_arity(&arguments, 0, 1)?;
                let value = match arguments.first() {
                    Some(value) => protocol::truth(&self.state.heap, value)?,
                    None => false,
                };
                Ok(CallResult::Value(Value::Bool(value)))
            }
            Builtin::Next => {
                expect_arity(&arguments, 1, 2)?;
                let Value::Object(id) = arguments[0] else {
                    return Err("next() argument is not an iterator".into());
                };
                let value = match self.state.heap.get(id)?.clone() {
                    Object::CountIterator { current, step } => {
                        let next = super::stdlib::itertools::count_next(current, step)
                            .map_err(str::to_string)?;
                        if let Object::CountIterator { current, .. } =
                            self.state.heap.get_mut(id)?
                        {
                            *current = next;
                        }
                        Some(Value::Int(current))
                    }
                    Object::Generator { .. } => self.resume_generator(id)?,
                    Object::Iterator { values, position } => {
                        let value = values.get(position).cloned();
                        if value.is_some() {
                            if let Object::Iterator { position, .. } =
                                self.state.heap.get_mut(id)?
                            {
                                *position += 1;
                            }
                        }
                        value
                    }
                    _ => return Err("next() argument is not an iterator".into()),
                };
                let value = match value {
                    Some(value) => value,
                    None => arguments.get(1).cloned().ok_or("StopIteration")?,
                };
                Ok(CallResult::Value(value))
            }
            Builtin::List | Builtin::Tuple | Builtin::Set => {
                expect_arity(&arguments, 0, 1)?;
                let values = arguments
                    .first()
                    .map(|value| self.iterable_values(value))
                    .transpose()?
                    .unwrap_or_default();
                let object = match function {
                    Builtin::List => Object::List(values),
                    Builtin::Tuple => Object::Tuple(values),
                    Builtin::Set => {
                        let mut unique = Vec::new();
                        for value in values {
                            self.charge_cpu(1)?;
                            if self.find_value(&unique, &value)?.is_none() {
                                self.reserve_result(64)?;
                                unique.push(value);
                            }
                        }
                        Object::Set(unique)
                    }
                    _ => unreachable!(),
                };
                Ok(CallResult::Value(self.allocate_object(object)?))
            }
            Builtin::Range => {
                expect_arity(&arguments, 1, 3)?;
                let integers = arguments
                    .iter()
                    .map(|value| value.as_int().ok_or("range arguments must be integers"))
                    .collect::<Result<Vec<_>, _>>()?;
                let (start, stop, step) = match integers.as_slice() {
                    [stop] => (0, *stop, 1),
                    [start, stop] => (*start, *stop, 1),
                    [start, stop, step] => (*start, *stop, *step),
                    _ => unreachable!(),
                };
                let values = self.range_values(start, stop, step)?;
                Ok(CallResult::Value(
                    self.allocate_object(Object::List(values))?,
                ))
            }
            Builtin::Enumerate => {
                expect_arity(&arguments, 1, 2)?;
                let values = self.iterable_values(&arguments[0])?;
                let start = arguments.get(1).map_or(Ok(0), |value| {
                    value.as_int().ok_or("enumerate start must be an integer")
                })?;
                let mut result = Vec::new();
                for (offset, value) in values.into_iter().enumerate() {
                    self.reserve_result(64)?;
                    self.charge_cpu(1)?;
                    let offset = i64::try_from(offset).map_err(|_| "enumerate is too large")?;
                    let index = start
                        .checked_add(offset)
                        .ok_or("enumerate index exceeds the bounded integer range")?;
                    result
                        .push(self.allocate_object(Object::Tuple(vec![Value::Int(index), value]))?);
                }
                Ok(CallResult::Value(
                    self.allocate_object(Object::List(result))?,
                ))
            }
            Builtin::Zip => {
                let sequences = arguments
                    .iter()
                    .map(|value| self.iterable_values(value))
                    .collect::<Result<Vec<_>, _>>()?;
                let length = sequences.iter().map(Vec::len).min().unwrap_or(0);
                let mut result = Vec::new();
                for index in 0..length {
                    self.reserve_result(64)?;
                    self.charge_cpu(1)?;
                    let tuple = sequences
                        .iter()
                        .map(|values| values[index].clone())
                        .collect();
                    result.push(self.allocate_object(Object::Tuple(tuple))?);
                }
                Ok(CallResult::Value(
                    self.allocate_object(Object::List(result))?,
                ))
            }
            Builtin::Any | Builtin::All => {
                expect_arity(&arguments, 1, 1)?;
                let values = self.iterable_values(&arguments[0])?;
                let mut result = matches!(function, Builtin::All);
                for value in values {
                    self.charge_cpu(1)?;
                    let truth = protocol::truth(&self.state.heap, &value)?;
                    if matches!(function, Builtin::Any) && truth {
                        result = true;
                        break;
                    }
                    if matches!(function, Builtin::All) && !truth {
                        result = false;
                        break;
                    }
                }
                Ok(CallResult::Value(Value::Bool(result)))
            }
            Builtin::ItertoolsCount => {
                expect_arity(&arguments, 0, 2)?;
                let current = arguments
                    .first()
                    .map(|value| value.as_int().ok_or("count start must be an integer"))
                    .transpose()?
                    .unwrap_or(0);
                let step = arguments
                    .get(1)
                    .map(|value| value.as_int().ok_or("count step must be an integer"))
                    .transpose()?
                    .unwrap_or(1);
                Ok(CallResult::Value(self.allocate_object(
                    Object::CountIterator { current, step },
                )?))
            }
            Builtin::ItertoolsIslice => {
                if !keyword_arguments.is_empty() {
                    return Err("itertools.islice does not accept keyword arguments".into());
                }
                if !(2..=4).contains(&arguments.len()) {
                    return Err("itertools.islice expected 2 to 4 arguments".into());
                }
                let (iterable, start, stop, step) = match arguments.as_slice() {
                    [iterable, stop] => (
                        iterable,
                        0_i64,
                        stop.as_int().ok_or("islice stop must be an integer")?,
                        1_i64,
                    ),
                    [iterable, start, stop] => (
                        iterable,
                        start.as_int().ok_or("islice start must be an integer")?,
                        stop.as_int().ok_or("islice stop must be an integer")?,
                        1_i64,
                    ),
                    [iterable, start, stop, step] => (
                        iterable,
                        start.as_int().ok_or("islice start must be an integer")?,
                        stop.as_int().ok_or("islice stop must be an integer")?,
                        step.as_int().ok_or("islice step must be an integer")?,
                    ),
                    _ => unreachable!(),
                };
                if step <= 0 {
                    return Err("islice step must be greater than zero".into());
                }
                let values = if let Value::Object(id) = iterable {
                    match self.state.heap.get(*id)?.clone() {
                        Object::CountIterator {
                            current,
                            step: count_step,
                        } => {
                            if start < 0 || stop < 0 {
                                return Err("islice indices must be non-negative".into());
                            }
                            let skip = start
                                .checked_mul(count_step)
                                .ok_or("islice arithmetic overflow")?;
                            let first = current
                                .checked_add(skip)
                                .ok_or("islice arithmetic overflow")?;
                            let length = if stop <= start {
                                0
                            } else {
                                usize::try_from((stop - start + step - 1) / step)
                                    .map_err(|_| "islice result is too large")?
                            };
                            let increment = count_step
                                .checked_mul(step)
                                .ok_or("islice arithmetic overflow")?;
                            self.reserve_result(length.saturating_mul(64))?;
                            let generated =
                                super::stdlib::itertools::count_slice(first, increment, length)
                                    .map_err(str::to_string)?;
                            let mut values = Vec::new();
                            for current in generated {
                                self.charge_cpu(1)?;
                                values.push(Value::Int(current));
                            }
                            let consumed = count_step
                                .checked_mul(stop)
                                .ok_or("islice arithmetic overflow")?;
                            let next_current = current
                                .checked_add(consumed)
                                .ok_or("islice arithmetic overflow")?;
                            let Object::CountIterator { current, .. } =
                                self.state.heap.get_mut(*id)?
                            else {
                                unreachable!()
                            };
                            *current = next_current;
                            values
                        }
                        _ => {
                            let source = self.iterable_values(iterable)?;
                            let begin = usize::try_from(start.max(0))
                                .map_err(|_| "islice start is too large")?;
                            let end = usize::try_from(stop.max(0))
                                .map_err(|_| "islice stop is too large")?;
                            let mut values = Vec::new();
                            let step =
                                usize::try_from(step).map_err(|_| "islice step is too large")?;
                            for value in source
                                .into_iter()
                                .skip(begin)
                                .take(end.saturating_sub(begin))
                                .step_by(step)
                            {
                                self.push_materialized(&mut values, value)?;
                            }
                            values
                        }
                    }
                } else {
                    return Err("islice argument is not iterable".into());
                };
                Ok(CallResult::Value(self.allocate_object(
                    Object::Iterator {
                        values,
                        position: 0,
                    },
                )?))
            }
            Builtin::HeapqHeapify => {
                expect_arity(&arguments, 1, 1)?;
                let Value::Object(id) = arguments[0] else {
                    return Err("heapify argument must be a list".into());
                };
                let mut values = match self.state.heap.get(id)?.clone() {
                    Object::List(values) => values,
                    _ => return Err("heapify argument must be a list".into()),
                };
                dynamic_heapify(&self.state.heap, &mut values)?;
                let Object::List(destination) = self.state.heap.get_mut(id)? else {
                    unreachable!()
                };
                *destination = values;
                Ok(CallResult::Value(Value::None))
            }
            Builtin::HeapqHeappop => {
                expect_arity(&arguments, 1, 1)?;
                let Value::Object(id) = arguments[0] else {
                    return Err("heappop argument must be a list".into());
                };
                let mut values = match self.state.heap.get(id)?.clone() {
                    Object::List(values) => values,
                    _ => return Err("heappop argument must be a list".into()),
                };
                let last = values.pop().ok_or("index out of range")?;
                if values.is_empty() {
                    let Object::List(destination) = self.state.heap.get_mut(id)? else {
                        unreachable!()
                    };
                    *destination = values;
                    return Ok(CallResult::Value(last));
                }
                let smallest = std::mem::replace(&mut values[0], last);
                dynamic_sift_down(&self.state.heap, &mut values, 0)?;
                let Object::List(destination) = self.state.heap.get_mut(id)? else {
                    unreachable!()
                };
                *destination = values;
                Ok(CallResult::Value(smallest))
            }
            Builtin::BisectLeft => {
                expect_arity(&arguments, 2, 2)?;
                let Value::Object(id) = arguments[0] else {
                    return Err("bisect argument must be a sequence".into());
                };
                let values = match self.state.heap.get(id)? {
                    Object::List(values) | Object::Tuple(values) => values.clone(),
                    _ => return Err("bisect argument must be a sequence".into()),
                };
                let mut low = 0;
                let mut high = values.len();
                while low < high {
                    let middle = low + (high - low) / 2;
                    if protocol::compare(&self.state.heap, &values[middle], &arguments[1])?
                        == Ordering::Less
                    {
                        low = middle + 1;
                    } else {
                        high = middle;
                    }
                }
                Ok(CallResult::Value(Value::Int(low as i64)))
            }
            Builtin::Reduce => {
                if !(2..=3).contains(&arguments.len()) {
                    return Err("reduce expected 2 or 3 arguments".into());
                }
                let function = arguments[0].clone();
                let values = self.iterable_values(&arguments[1])?;
                let mut iterator = values.into_iter();
                let mut accumulator = match arguments.get(2) {
                    Some(initial) => initial.clone(),
                    None => iterator
                        .next()
                        .ok_or("reduce() of empty sequence with no initial value")?,
                };
                for value in iterator {
                    self.stack.push(function.clone());
                    self.stack.extend([accumulator, value]);
                    accumulator = match self.call(2, &[], &[false, false])? {
                        CallResult::Value(value) => value,
                        CallResult::Exit(status) => return Ok(CallResult::Exit(status)),
                    };
                }
                Ok(CallResult::Value(accumulator))
            }
            Builtin::JsonDumps => {
                expect_arity(&arguments, 1, 1)?;
                let mut item_separator = ", ".to_string();
                let mut key_separator = ": ".to_string();
                let mut sort_keys = false;
                let mut saw_separators = false;
                let mut saw_sort_keys = false;
                for (name, value) in keyword_arguments {
                    match name.as_str() {
                        "separators" if !saw_separators => {
                            let values = self.iterable_values(&value)?;
                            let [Value::String(item), Value::String(key)] = values.as_slice()
                            else {
                                return Err(
                                    "json.dumps separators must be a pair of strings".into()
                                );
                            };
                            item_separator = item.clone();
                            key_separator = key.clone();
                            saw_separators = true;
                        }
                        "sort_keys" if !saw_sort_keys => {
                            sort_keys = protocol::truth(&self.state.heap, &value)?;
                            saw_sort_keys = true;
                        }
                        "separators" | "sort_keys" => {
                            return Err(format!(
                                "json.dumps got multiple values for keyword {name:?}"
                            ))
                        }
                        _ => {
                            return Err(format!(
                                "json.dumps keyword argument {name:?} is not implemented"
                            ))
                        }
                    }
                }
                let bound = self.json_size_bound(&arguments[0], 0, &mut Vec::new())?;
                let bound = bound
                    .checked_add(bound / 2)
                    .and_then(|bound| bound.checked_add(256))
                    .ok_or("json.dumps result is too large")?;
                self.reserve_result(bound)?;
                let rendered = self.json_dump_value(
                    &arguments[0],
                    &item_separator,
                    &key_separator,
                    sort_keys,
                    0,
                    &mut Vec::new(),
                )?;
                Ok(CallResult::Value(Value::String(rendered)))
            }
            Builtin::ReCompile => {
                let flags = keyword_int(&keyword_arguments, "flags")?
                    .unwrap_or_else(|| arguments.get(1).and_then(Value::as_int).unwrap_or(0));
                expect_arity(&arguments, 1, 2)?;
                let pattern = string_argument(&arguments[0], "regex pattern")?;
                Ok(CallResult::Value(self.allocate_regex(pattern, flags)?))
            }
            Builtin::ReSearch | Builtin::ReMatch | Builtin::ReFullMatch => {
                let (pattern, flags) = regex_argument(&arguments, &keyword_arguments)?;
                let text = string_argument(
                    arguments.get(1).ok_or("regex input is required")?,
                    "regex input",
                )?;
                expect_arity(&arguments, 2, 3)?;
                let regex = build_regex(&pattern, flags)?;
                let captures = match function {
                    Builtin::ReSearch => regex.captures(&text),
                    Builtin::ReMatch => regex.captures_at(&text, 0),
                    Builtin::ReFullMatch => regex.captures(&text).filter(|captures| {
                        captures.get(0).is_some_and(|matched| {
                            matched.start() == 0 && matched.end() == text.len()
                        })
                    }),
                    _ => unreachable!(),
                };
                match captures {
                    Some(captures) => Ok(CallResult::Value(self.allocate_match(&text, &captures)?)),
                    None => Ok(CallResult::Value(Value::None)),
                }
            }
            Builtin::ReFindAll | Builtin::ReFindIter => {
                let (pattern, flags) = regex_argument(&arguments, &keyword_arguments)?;
                let text = string_argument(
                    arguments.get(1).ok_or("regex input is required")?,
                    "regex input",
                )?;
                expect_arity(&arguments, 2, 3)?;
                let regex = build_regex(&pattern, flags)?;
                if matches!(function, Builtin::ReFindIter) {
                    let mut matches = Vec::new();
                    for captures in regex.captures_iter(&text) {
                        matches.push(self.allocate_match(&text, &captures)?);
                        if matches.len() > 100_000 {
                            return Err("regex result exceeds the bounded match limit".into());
                        }
                    }
                    return Ok(CallResult::Value(
                        self.allocate_object(Object::List(matches))?,
                    ));
                }
                let capture_count = regex.captures_len().saturating_sub(1);
                let mut values = Vec::new();
                for captures in regex.captures_iter(&text) {
                    let value = match capture_count {
                        0 => Value::String(captures[0].to_string()),
                        1 => Value::String(
                            captures
                                .get(1)
                                .map_or_else(String::new, |m| m.as_str().into()),
                        ),
                        _ => {
                            let fields = (1..=capture_count)
                                .map(|index| {
                                    Value::String(
                                        captures
                                            .get(index)
                                            .map_or_else(String::new, |m| m.as_str().into()),
                                    )
                                })
                                .collect();
                            self.allocate_object(Object::Tuple(fields))?
                        }
                    };
                    values.push(value);
                    if values.len() > 100_000 {
                        return Err("regex result exceeds the bounded match limit".into());
                    }
                }
                Ok(CallResult::Value(
                    self.allocate_object(Object::List(values))?,
                ))
            }
            Builtin::ReSub => {
                if arguments.len() < 3 || arguments.len() > 5 {
                    return Err("re.sub() expected 3 to 5 arguments".into());
                }
                let pattern = string_argument(&arguments[0], "regex pattern")?;
                let mut flags = arguments.get(4).and_then(Value::as_int).unwrap_or(0) as u32;
                if let Some(value) = keyword_int(&keyword_arguments, "flags")? {
                    if arguments.len() > 4 {
                        return Err("regex flags passed more than once".into());
                    }
                    flags = value as u32;
                }
                if flags & !(RE_IGNORECASE | RE_MULTILINE) != 0 {
                    return Err("regex flags are limited to IGNORECASE and MULTILINE".into());
                }
                let replacement = string_argument(&arguments[1], "replacement")?;
                let text = string_argument(&arguments[2], "regex input")?;
                let count = arguments.get(3).and_then(Value::as_int).unwrap_or(0);
                if count < 0 {
                    return Err("re.sub() count must be non-negative".into());
                }
                let regex = build_regex(&pattern, flags)?;
                // Rust's regex crate uses `$1`; accept Python's common `\\1` spelling without
                // attempting to emulate the full replacement mini-language.
                let replacement = normalize_replacement(&replacement)?;
                let rendered = if count == 0 {
                    regex.replace_all(&text, replacement.as_str()).into_owned()
                } else {
                    regex
                        .replacen(&text, count as usize, replacement.as_str())
                        .into_owned()
                };
                Ok(CallResult::Value(Value::String(rendered)))
            }
            Builtin::ReEscape => {
                expect_arity(&arguments, 1, 1)?;
                let text = string_argument(&arguments[0], "escape input")?;
                Ok(CallResult::Value(Value::String(python_re_escape(&text))))
            }
            Builtin::ArgparseArgumentParser => {
                expect_arity(&arguments, 0, 0)?;
                let prog = keyword_string(&keyword_arguments, "prog")?
                    .unwrap_or_else(|| self.argv.first().cloned().unwrap_or_else(|| "-".into()));
                reject_unknown_keywords(&keyword_arguments, &["prog", "description"])?;
                Ok(CallResult::Value(self.allocate_object(
                    Object::ArgumentParser {
                        prog,
                        arguments: Vec::new(),
                    },
                )?))
            }
            Builtin::ArgparseNamespace => {
                if !arguments.is_empty() {
                    return Err("argparse.Namespace accepts keyword arguments only".into());
                }
                let mut values = Vec::new();
                for (name, value) in keyword_arguments {
                    values.push((name, value));
                }
                Ok(CallResult::Value(
                    self.allocate_object(Object::Namespace { values })?,
                ))
            }
            Builtin::Dataclass => {
                expect_arity(&arguments, 1, 1)?;
                if !keyword_arguments.is_empty() {
                    return Err("dataclass options are not implemented".into());
                }
                let Value::Object(id) = arguments[0] else {
                    return Err("dataclass expects a class".into());
                };
                let Object::Class { is_dataclass, .. } = self.state.heap.get_mut(id)? else {
                    return Err("dataclass expects a class".into());
                };
                *is_dataclass = true;
                Ok(CallResult::Value(arguments[0].clone()))
            }
            Builtin::PytestFail | Builtin::PytestSkip => {
                expect_arity(&arguments, 0, 1)?;
                let message = arguments
                    .first()
                    .map(|value| protocol::display(&self.state.heap, value))
                    .transpose()?
                    .unwrap_or_default();
                let kind = if matches!(function, Builtin::PytestSkip) {
                    "Skipped"
                } else {
                    "Failed"
                };
                let value = Value::Exception {
                    kind: kind.into(),
                    message,
                };
                self.pending_exception = Some(RaisedException {
                    kind: kind.into(),
                    value,
                });
                Err(if kind == "Skipped" {
                    "pytest.skip()".into()
                } else {
                    "pytest.fail()".into()
                })
            }
            Builtin::PytestRaises => {
                expect_arity(&arguments, 1, 1)?;
                let Value::Native(NativeValue::ExceptionType(exception)) = arguments[0] else {
                    return Err("pytest.raises() expects an exception type".into());
                };
                Ok(CallResult::Value(self.allocate_object(
                    Object::RaisesContext {
                        expected: exception.0.to_string(),
                    },
                )?))
            }
            Builtin::DefaultDict => {
                expect_arity(&arguments, 1, 1)?;
                let factory = arguments[0].clone();
                let callable = matches!(factory, Value::Native(NativeValue::Function(_)))
                    || matches!(
                        &factory,
                        Value::Object(id)
                            if matches!(
                                self.state.heap.get(*id)?,
                                Object::Function { .. }
                                    | Object::BoundMethod { .. }
                                    | Object::PythonBoundMethod { .. }
                            )
                    );
                if !callable {
                    return Err("defaultdict first argument must be callable".into());
                }
                Ok(CallResult::Value(self.allocate_object(
                    Object::DefaultDict {
                        factory,
                        entries: Vec::new(),
                    },
                )?))
            }
            Builtin::Math(function) => {
                let arguments = arguments
                    .iter()
                    .map(|value| as_float(value).ok_or("math argument must be a real number"))
                    .collect::<Result<Vec<_>, _>>()?;
                let value = super::stdlib::math::call(function.name(), &arguments)
                    .map_err(|error| error.to_string())?;
                Ok(CallResult::Value(match value {
                    super::stdlib::math::MathValue::Float(value) => Value::Float(value),
                    super::stdlib::math::MathValue::Int(value) => Value::Int(value),
                    super::stdlib::math::MathValue::Bool(value) => Value::Bool(value),
                }))
            }
            Builtin::Write(stream) => {
                expect_arity(&arguments, 1, 1)?;
                let text = protocol::display(&self.state.heap, &arguments[0])?;
                self.write_output(stream, text.as_bytes());
                Ok(CallResult::Value(Value::Int(text.chars().count() as i64)))
            }
            Builtin::GetEnv | Builtin::EnvironmentGet => {
                expect_arity(&arguments, 1, 2)?;
                let Value::String(key) = &arguments[0] else {
                    return Err("environment variable name must be a string".into());
                };
                let value = self
                    .interp
                    .get_var(key)
                    .map(Value::String)
                    .unwrap_or_else(|| arguments.get(1).cloned().unwrap_or(Value::None));
                Ok(CallResult::Value(value))
            }
            Builtin::Time => {
                expect_arity(&arguments, 0, 0)?;
                let seconds = self
                    .interp
                    .clock
                    .wall_time_seconds()
                    .map_err(|error| error.to_string())?;
                Ok(CallResult::Value(Value::Float(seconds)))
            }
            Builtin::TimeNs => {
                expect_arity(&arguments, 0, 0)?;
                let nanos = self
                    .interp
                    .clock
                    .wall_time_ns()
                    .map_err(|error| error.to_string())?;
                let nanos =
                    i64::try_from(nanos).map_err(|_| "wall clock is outside Python int range")?;
                Ok(CallResult::Value(Value::Int(nanos)))
            }
            Builtin::Monotonic => {
                expect_arity(&arguments, 0, 0)?;
                Ok(CallResult::Value(Value::Float(
                    self.interp.clock.monotonic_seconds(),
                )))
            }
            Builtin::MonotonicNs => {
                expect_arity(&arguments, 0, 0)?;
                let nanos = i64::try_from(self.interp.clock.monotonic_ns())
                    .map_err(|_| "monotonic clock is outside Python int range")?;
                Ok(CallResult::Value(Value::Int(nanos)))
            }
            Builtin::ProcessTime => {
                expect_arity(&arguments, 0, 0)?;
                Ok(CallResult::Value(Value::Float(
                    self.interp.resources.process_time_seconds(),
                )))
            }
            Builtin::ProcessTimeNs => {
                expect_arity(&arguments, 0, 0)?;
                let nanos = i64::try_from(self.interp.resources.process_time_ns())
                    .map_err(|_| "process clock is outside Python int range")?;
                Ok(CallResult::Value(Value::Int(nanos)))
            }
            Builtin::Sleep => {
                expect_arity(&arguments, 1, 1)?;
                let seconds =
                    as_float(&arguments[0]).ok_or("time.sleep() argument must be a real number")?;
                if !seconds.is_finite() || seconds < 0.0 {
                    return Err("time.sleep() length must be a finite non-negative number".into());
                }
                let nanos = seconds * crate::clock::NANOS_PER_SECOND as f64;
                if nanos > u64::MAX as f64 {
                    return Err("time.sleep() length is too large".into());
                }
                match self
                    .interp
                    .clock
                    .block_task(crate::clock::MAIN_TASK_ID, nanos as u64)
                    .map_err(|error| error.to_string())?
                {
                    crate::clock::BlockOutcome::Completed => {}
                    crate::clock::BlockOutcome::Interrupted(event) => {
                        self.interp.deadline_interrupt = Some(event.id);
                    }
                }
                Ok(CallResult::Value(Value::None))
            }
        }
    }

    fn call_python_function(
        &mut self,
        name: &str,
        code: &Code,
        closure: Option<ScopeId>,
        defaults: &[Value],
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<CallResult, String> {
        if code
            .instructions
            .iter()
            .any(|instruction| matches!(&instruction.operation, Operation::Yield))
        {
            return self.create_generator(
                name,
                code,
                closure,
                defaults,
                arguments,
                keyword_arguments,
            );
        }
        const MAX_CALL_DEPTH: usize = 256;
        if self.call_depth == MAX_CALL_DEPTH {
            return Err("maximum recursion depth exceeded".into());
        }
        let variadic_index = code
            .parameters
            .iter()
            .position(|parameter| parameter.variadic);
        let fixed_len = variadic_index.unwrap_or(code.parameters.len());
        if variadic_index.is_none() && arguments.len() > fixed_len {
            return Err(format!(
                "{name}() takes {} positional arguments but {} were given",
                fixed_len,
                arguments.len()
            ));
        }
        let mut positional = arguments;
        let extra_positional = if variadic_index.is_some() && positional.len() > fixed_len {
            positional.split_off(fixed_len)
        } else {
            Vec::new()
        };
        let mut locals = code
            .parameters
            .iter()
            .take(fixed_len)
            .map(|parameter| parameter.name.clone())
            .zip(positional)
            .collect::<HashMap<_, _>>();
        if let Some(index) = variadic_index {
            let parameter = &code.parameters[index];
            let values = self.allocate_object(Object::Tuple(extra_positional))?;
            locals.insert(parameter.name.clone(), values);
        }
        for (keyword, value) in keyword_arguments {
            if !code
                .parameters
                .iter()
                .any(|parameter| parameter.name == keyword)
            {
                return Err(format!(
                    "{name}() got an unexpected keyword argument {keyword:?}"
                ));
            }
            if locals.insert(keyword.clone(), value).is_some() {
                return Err(format!(
                    "{name}() got multiple values for argument {keyword:?}"
                ));
            }
        }
        if let Some(missing) =
            code.parameters.iter().enumerate().find(|(_, parameter)| {
                !locals.contains_key(&parameter.name) && !parameter.has_default
            })
        {
            return Err(format!(
                "{name}() missing required argument {:?}",
                missing.1.name
            ));
        }
        let default_start = fixed_len.saturating_sub(defaults.len());
        if defaults.len() > fixed_len
            || code.parameters[default_start..fixed_len]
                .iter()
                .any(|parameter| !parameter.has_default)
        {
            return Err(format!("{name}() has invalid default argument metadata"));
        }
        for (index, default) in defaults.iter().enumerate() {
            let parameter = &code.parameters[default_start + index];
            locals
                .entry(parameter.name.clone())
                .or_insert_with(|| default.clone());
        }
        let uses_repl_globals = closure
            .map(|scope| self.state.heap.scope_uses_repl_globals(scope))
            .transpose()?
            .unwrap_or(true);
        let scope = self.state.heap.allocate_scope(
            closure,
            uses_repl_globals,
            locals,
            &mut self.interp.resources,
        )?;
        let outer_stack = std::mem::take(&mut self.stack);
        self.local_scopes.push(scope);
        self.call_depth += 1;
        let result = self.execute_code(code);
        self.call_depth -= 1;
        self.local_scopes.pop();
        self.stack = outer_stack;
        match result {
            Ok(Execution::Return(value)) => Ok(CallResult::Value(value)),
            Ok(Execution::Halt) => Ok(CallResult::Value(Value::None)),
            Ok(Execution::Yield(_, _)) => {
                Err(format!("unexpected yield in ordinary function {name}"))
            }
            Ok(Execution::Exit(status)) => Ok(CallResult::Exit(status)),
            Err((error, span)) => Err(format!(
                "{error} in {name} at line {}, column {}",
                span.line, span.column
            )),
        }
    }

    fn create_generator(
        &mut self,
        name: &str,
        code: &Code,
        closure: Option<ScopeId>,
        defaults: &[Value],
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<CallResult, String> {
        // Binding happens at call time, while the body itself starts only on the first next().
        // Keeping this validation identical to ordinary functions prevents generators from being
        // an accidental argument-checking escape hatch.
        let variadic_index = code
            .parameters
            .iter()
            .position(|parameter| parameter.variadic);
        let fixed_len = variadic_index.unwrap_or(code.parameters.len());
        if variadic_index.is_none() && arguments.len() > fixed_len {
            return Err(format!(
                "{name}() takes {} positional arguments but {} were given",
                fixed_len,
                arguments.len()
            ));
        }
        let mut positional = arguments;
        let extra_positional = if variadic_index.is_some() && positional.len() > fixed_len {
            positional.split_off(fixed_len)
        } else {
            Vec::new()
        };
        let mut locals = code
            .parameters
            .iter()
            .take(fixed_len)
            .map(|parameter| parameter.name.clone())
            .zip(positional)
            .collect::<HashMap<_, _>>();
        if let Some(index) = variadic_index {
            let values = self.allocate_object(super::heap::Object::Tuple(extra_positional))?;
            locals.insert(code.parameters[index].name.clone(), values);
        }
        for (keyword, value) in keyword_arguments {
            if !code
                .parameters
                .iter()
                .any(|parameter| parameter.name == keyword)
            {
                return Err(format!(
                    "{name}() got an unexpected keyword argument {keyword:?}"
                ));
            }
            if locals.insert(keyword.clone(), value).is_some() {
                return Err(format!(
                    "{name}() got multiple values for argument {keyword:?}"
                ));
            }
        }
        if let Some(missing) = code
            .parameters
            .iter()
            .find(|parameter| !locals.contains_key(&parameter.name) && !parameter.has_default)
        {
            return Err(format!(
                "{name}() missing required argument {:?}",
                missing.name
            ));
        }
        let default_start = fixed_len.saturating_sub(defaults.len());
        if defaults.len() > fixed_len
            || code.parameters[default_start..fixed_len]
                .iter()
                .any(|parameter| !parameter.has_default)
        {
            return Err(format!("{name}() has invalid default argument metadata"));
        }
        for (index, default) in defaults.iter().enumerate() {
            locals
                .entry(code.parameters[default_start + index].name.clone())
                .or_insert_with(|| default.clone());
        }
        let uses_repl_globals = closure
            .map(|scope| self.state.heap.scope_uses_repl_globals(scope))
            .transpose()?
            .unwrap_or(true);
        let scope = self.state.heap.allocate_scope(
            closure,
            uses_repl_globals,
            locals,
            &mut self.interp.resources,
        )?;
        let generator = self.allocate_object(Object::Generator {
            name: name.to_string(),
            code: code.clone(),
            scope,
            instruction_pointer: 0,
            handlers: Vec::new(),
            stack: Vec::new(),
            exhausted: false,
            running: false,
        })?;
        Ok(CallResult::Value(generator))
    }

    fn unittest_failure(&mut self, message: String) -> Result<CallResult, String> {
        let value = Value::Exception {
            kind: "AssertionError".into(),
            message,
        };
        self.pending_exception = Some(RaisedException {
            kind: "AssertionError".into(),
            value,
        });
        Err("unittest assertion failed".into())
    }

    fn call_method(
        &mut self,
        receiver: Value,
        method: Method,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<CallResult, String> {
        if !keyword_arguments.is_empty()
            && !matches!(method, Method::ListSort | Method::ArgumentParserAddArgument)
        {
            return Err("method keyword arguments are not implemented".into());
        }
        match method {
            Method::StringStrip | Method::StringLStrip | Method::StringRStrip => {
                expect_arity(&arguments, 0, 1)?;
                let Value::String(value) = receiver else {
                    return Err("invalid string method receiver".into());
                };
                let characters = match arguments.first() {
                    None | Some(Value::None) => None,
                    Some(Value::String(characters)) => Some(characters.as_str()),
                    _ => return Err("strip argument must be a string or None".into()),
                };
                let stripped = match (method, characters) {
                    (Method::StringStrip, None) => value.trim().to_string(),
                    (Method::StringLStrip, None) => value.trim_start().to_string(),
                    (Method::StringRStrip, None) => value.trim_end().to_string(),
                    (Method::StringStrip, Some(chars)) => {
                        value.trim_matches(|ch| chars.contains(ch)).to_string()
                    }
                    (Method::StringLStrip, Some(chars)) => value
                        .trim_start_matches(|ch| chars.contains(ch))
                        .to_string(),
                    (Method::StringRStrip, Some(chars)) => {
                        value.trim_end_matches(|ch| chars.contains(ch)).to_string()
                    }
                    _ => unreachable!(),
                };
                Ok(CallResult::Value(Value::String(stripped)))
            }
            Method::StringStartsWith | Method::StringEndsWith => {
                expect_arity(&arguments, 1, 1)?;
                let (Value::String(value), Value::String(needle)) = (&receiver, &arguments[0])
                else {
                    return Err("prefix/suffix must be a string".into());
                };
                Ok(CallResult::Value(Value::Bool(match method {
                    Method::StringStartsWith => value.starts_with(needle),
                    Method::StringEndsWith => value.ends_with(needle),
                    _ => unreachable!(),
                })))
            }
            Method::StringSplit => {
                expect_arity(&arguments, 0, 2)?;
                let Value::String(value) = receiver else {
                    return Err("invalid string method receiver".into());
                };
                let separator = match arguments.first() {
                    None | Some(Value::None) => None,
                    Some(Value::String(separator)) if separator.is_empty() => {
                        return Err("empty separator".into())
                    }
                    Some(Value::String(separator)) => Some(separator.as_str()),
                    _ => return Err("separator must be a string or None".into()),
                };
                let maximum = arguments
                    .get(1)
                    .map(|value| value.as_int().ok_or("maxsplit must be an integer"))
                    .transpose()?;
                let parts = split_string(&value, separator, maximum)
                    .into_iter()
                    .map(Value::String)
                    .collect();
                Ok(CallResult::Value(
                    self.allocate_object(Object::List(parts))?,
                ))
            }
            Method::ListAppend => {
                expect_arity(&arguments, 1, 1)?;
                let id = object_receiver(receiver)?;
                self.reserve_slots(1)?;
                let Object::List(values) = self.state.heap.get_mut(id)? else {
                    return Err("invalid list method receiver".into());
                };
                values.push(arguments[0].clone());
                Ok(CallResult::Value(Value::None))
            }
            Method::ListExtend => {
                expect_arity(&arguments, 1, 1)?;
                let id = object_receiver(receiver)?;
                let added = self.iterable_values(&arguments[0])?;
                self.reserve_slots(added.len())?;
                let Object::List(values) = self.state.heap.get_mut(id)? else {
                    return Err("invalid list method receiver".into());
                };
                values.extend(added);
                Ok(CallResult::Value(Value::None))
            }
            Method::ListPop => {
                expect_arity(&arguments, 0, 1)?;
                let id = object_receiver(receiver)?;
                let Object::List(values) = self.state.heap.get_mut(id)? else {
                    return Err("invalid list method receiver".into());
                };
                if values.is_empty() {
                    return Err("pop from empty list".into());
                }
                let index = arguments
                    .first()
                    .map_or(-1, |value| value.as_int().unwrap_or(i64::MIN));
                let index = normalize_index(index, values.len(), "pop index out of range")?;
                Ok(CallResult::Value(values.remove(index)))
            }
            Method::ListRemove => {
                expect_arity(&arguments, 1, 1)?;
                let id = object_receiver(receiver)?;
                let position = match self.state.heap.get(id)? {
                    Object::List(values) => {
                        let values = values.clone();
                        self.find_value(&values, &arguments[0])?
                    }
                    _ => return Err("invalid list method receiver".into()),
                };
                let Some(position) = position else {
                    return Err("list.remove(x): x not in list".into());
                };
                let Object::List(values) = self.state.heap.get_mut(id)? else {
                    unreachable!()
                };
                values.remove(position);
                Ok(CallResult::Value(Value::None))
            }
            Method::ListSort => {
                expect_arity(&arguments, 0, 0)?;
                let id = object_receiver(receiver)?;
                let mut key_function = None;
                let mut saw_key = false;
                let mut reverse = false;
                let mut saw_reverse = false;
                for (name, value) in keyword_arguments {
                    match name.as_str() {
                        "key" if !saw_key => {
                            if !matches!(value, Value::None) {
                                key_function = Some(value);
                            }
                            saw_key = true;
                        }
                        "reverse" if !saw_reverse => {
                            reverse = protocol::truth(&self.state.heap, &value)?;
                            saw_reverse = true;
                        }
                        "key" | "reverse" => {
                            return Err(format!(
                                "list.sort() got multiple values for keyword {name:?}"
                            ))
                        }
                        _ => return Err(format!("list.sort() got an unexpected keyword {name:?}")),
                    }
                }
                let values = match self.state.heap.get(id)? {
                    Object::List(values) => values.clone(),
                    _ => return Err("invalid list method receiver".into()),
                };
                let mut keyed = Vec::new();
                for value in values {
                    let key = if let Some(function) = &key_function {
                        self.stack.push(function.clone());
                        self.stack.push(value.clone());
                        match self.call(1, &[], &[false])? {
                            CallResult::Value(key) => key,
                            CallResult::Exit(status) => return Ok(CallResult::Exit(status)),
                        }
                    } else {
                        value.clone()
                    };
                    self.reserve_result(64)?;
                    keyed.push((key, value));
                }
                for index in 1..keyed.len() {
                    let mut current = index;
                    while current > 0 {
                        self.charge_cpu(1)?;
                        if protocol::compare(
                            &self.state.heap,
                            &keyed[current].0,
                            &keyed[current - 1].0,
                        )? != if reverse {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        } {
                            break;
                        }
                        keyed.swap(current, current - 1);
                        current -= 1;
                    }
                }
                let Object::List(values) = self.state.heap.get_mut(id)? else {
                    unreachable!()
                };
                *values = keyed.into_iter().map(|(_, value)| value).collect();
                Ok(CallResult::Value(Value::None))
            }
            Method::DictGet | Method::DictSetDefault => {
                expect_arity(&arguments, 1, 2)?;
                let id = object_receiver(receiver)?;
                let key = &arguments[0];
                let default = arguments.get(1).cloned().unwrap_or(Value::None);
                let position = self.dict_position(id, key)?;
                if let Some(position) = position {
                    let entries = match self.state.heap.get(id)? {
                        Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
                        _ => unreachable!(),
                    };
                    return Ok(CallResult::Value(entries[position].1.clone()));
                }
                if matches!(method, Method::DictSetDefault) {
                    self.reserve_slots(2)?;
                    let entries = match self.state.heap.get_mut(id)? {
                        Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
                        _ => unreachable!(),
                    };
                    entries.push((key.clone(), default.clone()));
                }
                Ok(CallResult::Value(default))
            }
            Method::DictKeys | Method::DictValues | Method::DictItems => {
                expect_arity(&arguments, 0, 0)?;
                let id = object_receiver(receiver)?;
                let entries = match self.state.heap.get(id)? {
                    Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
                    _ => return Err("invalid dictionary method receiver".into()),
                };
                let entries = entries.clone();
                let mut values = Vec::new();
                for (key, value) in entries {
                    self.charge_cpu(1)?;
                    let item = match method {
                        Method::DictKeys => key,
                        Method::DictValues => value,
                        Method::DictItems => {
                            self.allocate_object(Object::Tuple(vec![key, value]))?
                        }
                        _ => unreachable!(),
                    };
                    self.push_materialized(&mut values, item)?;
                }
                Ok(CallResult::Value(
                    self.allocate_object(Object::List(values))?,
                ))
            }
            Method::SetAdd => {
                expect_arity(&arguments, 1, 1)?;
                let id = object_receiver(receiver)?;
                if self.set_position(id, &arguments[0])?.is_none() {
                    self.reserve_slots(1)?;
                    let Object::Set(values) = self.state.heap.get_mut(id)? else {
                        unreachable!()
                    };
                    values.push(arguments[0].clone());
                }
                Ok(CallResult::Value(Value::None))
            }
            Method::SetUpdate => {
                expect_arity(&arguments, 1, 1)?;
                let id = object_receiver(receiver)?;
                for value in self.iterable_values(&arguments[0])? {
                    if self.set_position(id, &value)?.is_none() {
                        self.reserve_slots(1)?;
                        let Object::Set(values) = self.state.heap.get_mut(id)? else {
                            unreachable!()
                        };
                        values.push(value);
                    }
                }
                Ok(CallResult::Value(Value::None))
            }
            Method::SetRemove | Method::SetDiscard => {
                expect_arity(&arguments, 1, 1)?;
                let id = object_receiver(receiver)?;
                let position = self.set_position(id, &arguments[0])?;
                if let Some(position) = position {
                    let Object::Set(values) = self.state.heap.get_mut(id)? else {
                        unreachable!()
                    };
                    values.remove(position);
                } else if matches!(method, Method::SetRemove) {
                    return Err("set element not found".into());
                }
                Ok(CallResult::Value(Value::None))
            }
            Method::RegexSearch
            | Method::RegexMatch
            | Method::RegexFullMatch
            | Method::RegexFindAll
            | Method::RegexFindIter
            | Method::RegexSub => {
                let Value::Object(id) = receiver else {
                    return Err("invalid regex method receiver".into());
                };
                let Object::Regex { pattern, flags } = self.state.heap.get(id)?.clone() else {
                    return Err("invalid regex method receiver".into());
                };
                let regex = build_regex(&pattern, flags)?;
                match method {
                    Method::RegexSub => {
                        expect_arity(&arguments, 2, 3)?;
                        let replacement = string_argument(&arguments[0], "replacement")?;
                        let text = string_argument(&arguments[1], "regex input")?;
                        let count = arguments.get(2).and_then(Value::as_int).unwrap_or(0);
                        if count < 0 {
                            return Err("regex count must be non-negative".into());
                        }
                        let replacement = normalize_replacement(&replacement)?;
                        // Empty patterns can match at every UTF-8 boundary. Reserve a
                        // conservative output bound before regex::replace allocates its String.
                        let potential_matches = text.len().saturating_add(1);
                        let matches = if count == 0 {
                            potential_matches
                        } else {
                            potential_matches.min(count as usize)
                        };
                        let bound = text
                            .len()
                            .checked_add(
                                matches
                                    .checked_mul(replacement.len())
                                    .ok_or("regex substitution result is too large")?,
                            )
                            .ok_or("regex substitution result is too large")?;
                        self.reserve_result(bound.saturating_add(64))?;
                        self.charge_cpu(
                            u64::try_from(text.len().saturating_add(replacement.len()))
                                .unwrap_or(u64::MAX),
                        )?;
                        let rendered = if count == 0 {
                            regex.replace_all(&text, replacement.as_str()).into_owned()
                        } else {
                            regex
                                .replacen(&text, count as usize, replacement.as_str())
                                .into_owned()
                        };
                        Ok(CallResult::Value(Value::String(rendered)))
                    }
                    Method::RegexFindAll | Method::RegexFindIter => {
                        let text = string_argument(
                            arguments.first().ok_or("regex input is required")?,
                            "regex input",
                        )?;
                        let capture_count = regex.captures_len().saturating_sub(1);
                        let mut values = Vec::new();
                        for captures in regex.captures_iter(&text) {
                            self.charge_cpu(1)?;
                            if matches!(method, Method::RegexFindIter) {
                                let item = self.allocate_match(&text, &captures)?;
                                self.push_materialized(&mut values, item)?;
                            } else {
                                let item = match capture_count {
                                    0 => Value::String(captures[0].to_string()),
                                    1 => Value::String(
                                        captures
                                            .get(1)
                                            .map_or_else(String::new, |m| m.as_str().into()),
                                    ),
                                    _ => self.allocate_object(Object::Tuple(
                                        (1..=capture_count)
                                            .map(|index| {
                                                Value::String(
                                                    captures
                                                        .get(index)
                                                        .map_or_else(String::new, |m| {
                                                            m.as_str().into()
                                                        }),
                                                )
                                            })
                                            .collect(),
                                    ))?,
                                };
                                self.push_materialized(&mut values, item)?;
                            }
                            if values.len() > 100_000 {
                                return Err("regex result exceeds the bounded match limit".into());
                            }
                        }
                        Ok(CallResult::Value(
                            self.allocate_object(Object::List(values))?,
                        ))
                    }
                    _ => {
                        let text = string_argument(
                            arguments.first().ok_or("regex input is required")?,
                            "regex input",
                        )?;
                        let captures = match method {
                            Method::RegexSearch => regex.captures(&text),
                            Method::RegexMatch => regex.captures_at(&text, 0),
                            Method::RegexFullMatch => regex.captures(&text).filter(|captures| {
                                captures.get(0).is_some_and(|matched| {
                                    matched.start() == 0 && matched.end() == text.len()
                                })
                            }),
                            _ => unreachable!(),
                        };
                        match captures {
                            Some(captures) => {
                                Ok(CallResult::Value(self.allocate_match(&text, &captures)?))
                            }
                            None => Ok(CallResult::Value(Value::None)),
                        }
                    }
                }
            }
            Method::MatchGroup | Method::MatchGroups | Method::MatchStart | Method::MatchEnd => {
                let Value::Object(id) = receiver else {
                    return Err("invalid match method receiver".into());
                };
                let Object::Match {
                    groups, start, end, ..
                } = self.state.heap.get(id)?.clone()
                else {
                    return Err("invalid match method receiver".into());
                };
                match method {
                    Method::MatchGroup => {
                        expect_arity(&arguments, 0, 1)?;
                        let index = arguments
                            .first()
                            .map_or(0, |value| value.as_int().unwrap_or(-1));
                        let index = usize::try_from(index).map_err(|_| "no such group")?;
                        Ok(CallResult::Value(
                            groups
                                .get(index)
                                .cloned()
                                .flatten()
                                .map_or(Value::None, Value::String),
                        ))
                    }
                    Method::MatchGroups => {
                        expect_arity(&arguments, 0, 0)?;
                        let values = groups
                            .iter()
                            .skip(1)
                            .map(|value| value.clone().map_or(Value::None, Value::String))
                            .collect();
                        Ok(CallResult::Value(
                            self.allocate_object(Object::Tuple(values))?,
                        ))
                    }
                    Method::MatchStart | Method::MatchEnd => {
                        expect_arity(&arguments, 0, 1)?;
                        let position = if matches!(method, Method::MatchStart) {
                            start
                        } else {
                            end
                        };
                        Ok(CallResult::Value(Value::Int(position as i64)))
                    }
                    _ => unreachable!(),
                }
            }
            Method::UnitTestAssertEqual => {
                expect_arity(&arguments, 2, 3)?;
                if protocol::equals(&self.state.heap, &arguments[0], &arguments[1])? {
                    return Ok(CallResult::Value(Value::None));
                }
                let message = arguments
                    .get(2)
                    .map(|value| protocol::display(&self.state.heap, value))
                    .transpose()?
                    .unwrap_or_else(|| {
                        let left = protocol::repr(&self.state.heap, &arguments[0])
                            .unwrap_or_else(|_| "<value>".into());
                        let right = protocol::repr(&self.state.heap, &arguments[1])
                            .unwrap_or_else(|_| "<value>".into());
                        format!("{left} != {right}")
                    });
                self.unittest_failure(message)
            }
            Method::UnitTestAssertTrue | Method::UnitTestAssertFalse => {
                expect_arity(&arguments, 1, 2)?;
                let actual = protocol::truth(&self.state.heap, &arguments[0])?;
                let expected = matches!(method, Method::UnitTestAssertTrue);
                if actual == expected {
                    return Ok(CallResult::Value(Value::None));
                }
                let message = arguments
                    .get(1)
                    .map(|value| protocol::display(&self.state.heap, value))
                    .transpose()?
                    .unwrap_or_else(|| {
                        if expected {
                            "False is not true".into()
                        } else {
                            "True is not false".into()
                        }
                    });
                self.unittest_failure(message)
            }
            Method::UnitTestAssertIsNone => {
                expect_arity(&arguments, 1, 2)?;
                if matches!(arguments[0], Value::None) {
                    return Ok(CallResult::Value(Value::None));
                }
                let message = arguments
                    .get(1)
                    .map(|value| protocol::display(&self.state.heap, value))
                    .transpose()?
                    .unwrap_or_else(|| "value is not None".into());
                self.unittest_failure(message)
            }
            Method::UnitTestAssertRaises => {
                expect_arity(&arguments, 1, 1)?;
                let Value::Native(NativeValue::ExceptionType(exception)) = arguments[0] else {
                    return Err("assertRaises() expects an exception type".into());
                };
                Ok(CallResult::Value(self.allocate_object(
                    Object::RaisesContext {
                        expected: exception.0.to_string(),
                    },
                )?))
            }
            Method::RaisesEnter => {
                expect_arity(&arguments, 0, 0)?;
                Ok(CallResult::Value(receiver))
            }
            Method::RaisesExit => {
                expect_arity(&arguments, 3, 3)?;
                let Value::Object(id) = receiver.clone() else {
                    return Err("invalid pytest.raises receiver".into());
                };
                let Object::RaisesContext { expected } = self.state.heap.get(id)?.clone() else {
                    return Err("invalid pytest.raises receiver".into());
                };
                let kind = match &arguments[0] {
                    Value::String(kind) => Some(kind.as_str()),
                    Value::None => None,
                    _ => return Err("invalid exception context".into()),
                };
                let Some(kind) = kind else {
                    let value = Value::Exception {
                        kind: "Failed".into(),
                        message: "DID NOT RAISE".into(),
                    };
                    self.pending_exception = Some(RaisedException {
                        kind: "Failed".into(),
                        value,
                    });
                    return Err("pytest.raises() did not catch an exception".into());
                };
                Ok(CallResult::Value(Value::Bool(
                    expected == "Exception" || expected == "BaseException" || *kind == expected,
                )))
            }
            Method::ArgumentParserAddArgument => {
                let Value::Object(id) = receiver else {
                    return Err("invalid ArgumentParser receiver".into());
                };
                if arguments.is_empty() {
                    return Err("add_argument() requires at least one name".into());
                }
                let names = arguments
                    .iter()
                    .map(|value| string_argument(value, "argument name"))
                    .collect::<Result<Vec<_>, _>>()?;
                let dest = keyword_string(&keyword_arguments, "dest")?.unwrap_or_else(|| {
                    names
                        .iter()
                        .find(|name| name.starts_with("--"))
                        .unwrap_or(&names[0])
                        .trim_start_matches('-')
                        .replace('-', "_")
                });
                let required = keyword_bool(&keyword_arguments, "required")?.unwrap_or(false);
                let store_true = keyword_string(&keyword_arguments, "action")?
                    .is_some_and(|action| action == "store_true");
                let integer = matches!(
                    keyword_arguments
                        .iter()
                        .find(|(name, _)| name == "type")
                        .map(|(_, value)| value),
                    Some(Value::Native(NativeValue::Function(Builtin::Integer)))
                );
                let default =
                    keyword_value(&keyword_arguments, "default")?.unwrap_or(if store_true {
                        Value::Bool(false)
                    } else {
                        Value::None
                    });
                reject_unknown_keywords(
                    &keyword_arguments,
                    &["dest", "required", "action", "type", "default", "help"],
                )?;
                self.state
                    .heap
                    .reserve_growth(48, &mut self.interp.resources)?;
                let Object::ArgumentParser { arguments, .. } = self.state.heap.get_mut(id)? else {
                    return Err("invalid ArgumentParser receiver".into());
                };
                arguments.push(super::heap::ArgumentSpec {
                    names,
                    dest,
                    required,
                    default,
                    store_true,
                    integer,
                });
                Ok(CallResult::Value(Value::None))
            }
            Method::ArgumentParserParseArgs => {
                let Value::Object(id) = receiver else {
                    return Err("invalid ArgumentParser receiver".into());
                };
                let (specs, prog) = match self.state.heap.get(id)?.clone() {
                    Object::ArgumentParser { prog, arguments } => (arguments, prog),
                    _ => return Err("invalid ArgumentParser receiver".into()),
                };
                expect_arity(&arguments, 0, 1)?;
                let input = if let Some(value) = arguments.first() {
                    self.iterable_values(value)?
                        .into_iter()
                        .map(|value| string_argument(&value, "argument value"))
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    self.argv.iter().skip(1).cloned().collect()
                };
                let values = parse_argument_values(&specs, &input, &prog)?;
                Ok(CallResult::Value(
                    self.allocate_object(Object::Namespace { values })?,
                ))
            }
        }
    }

    fn allocate_object(&mut self, object: Object) -> Result<Value, String> {
        self.state.heap.allocate(object, &mut self.interp.resources)
    }

    fn allocate_regex(&mut self, pattern: String, flags: i64) -> Result<Value, String> {
        if flags < 0 || flags as u64 > u32::MAX as u64 {
            return Err("regex flags are out of range".into());
        }
        let flags = flags as u32;
        build_regex(&pattern, flags)?;
        self.allocate_object(Object::Regex { pattern, flags })
    }

    fn allocate_match(
        &mut self,
        text: &str,
        captures: &regex::Captures<'_>,
    ) -> Result<Value, String> {
        let whole = captures
            .get(0)
            .ok_or("regex engine returned no whole match")?;
        let groups = captures
            .iter()
            .map(|capture| capture.map(|matched| matched.as_str().to_string()))
            .collect();
        let start = text[..whole.start()].chars().count();
        let end = text[..whole.end()].chars().count();
        self.allocate_object(Object::Match {
            text: whole.as_str().to_string(),
            groups,
            start,
            end,
        })
    }

    fn range_values(&mut self, start: i64, stop: i64, step: i64) -> Result<Vec<Value>, String> {
        if step == 0 {
            return Err("range() arg 3 must not be zero".into());
        }
        let start = i128::from(start);
        let stop = i128::from(stop);
        let step = i128::from(step);
        let count = if step > 0 && start < stop {
            (stop - start - 1) / step + 1
        } else if step < 0 && start > stop {
            (start - stop - 1) / -step + 1
        } else {
            0
        };
        let count = usize::try_from(count).map_err(|_| "range is too large")?;
        let bytes = count.checked_mul(24).ok_or("range result is too large")?;
        self.reserve_result(bytes)?;
        if !self
            .interp
            .resources
            .charge_cpu(u64::try_from(count).map_err(|_| "range is too large")?)
        {
            return Err("resource limit exceeded while constructing range".into());
        }
        let mut values = Vec::with_capacity(count);
        let mut value = start;
        for _ in 0..count {
            values.push(Value::Int(
                i64::try_from(value).map_err(|_| "range value exceeds bounded integer range")?,
            ));
            value += step;
        }
        Ok(values)
    }

    fn json_dump_value(
        &mut self,
        value: &Value,
        item_separator: &str,
        key_separator: &str,
        sort_keys: bool,
        depth: usize,
        active: &mut Vec<super::heap::ObjectId>,
    ) -> Result<String, String> {
        if depth == 256 {
            return Err("maximum JSON nesting depth exceeded".into());
        }
        if !self.interp.resources.charge_cpu(1) {
            return Err("resource limit exceeded while encoding JSON".into());
        }
        match value {
            Value::None => Ok("null".into()),
            Value::Bool(value) => Ok(value.to_string()),
            Value::Int(value) => Ok(value.to_string()),
            Value::Float(value) if value.is_finite() => {
                serde_json::to_string(value).map_err(|error| error.to_string())
            }
            Value::String(value) => serde_json::to_string(value).map_err(|error| error.to_string()),
            Value::Exception { .. } => Err("object is not JSON serializable".into()),
            Value::Object(id) => {
                if active.contains(id) {
                    return Err("circular reference detected while encoding JSON".into());
                }
                let object = self.state.heap.get(*id)?.clone();
                match object {
                    Object::List(values) | Object::Tuple(values) => {
                        active.push(*id);
                        let rendered = values
                            .iter()
                            .map(|value| {
                                self.json_dump_value(
                                    value,
                                    item_separator,
                                    key_separator,
                                    sort_keys,
                                    depth + 1,
                                    active,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>();
                        active.pop();
                        Ok(format!("[{}]", rendered?.join(item_separator)))
                    }
                    Object::Dict(mut entries) | Object::DefaultDict { mut entries, .. } => {
                        if sort_keys {
                            for (key, _) in &entries {
                                if !matches!(key, Value::String(_)) {
                                    return Err("json.dumps sort_keys requires string keys".into());
                                }
                            }
                            // Keep comparison work inside the VM meter. This insertion sort is
                            // intentionally slow but makes every native comparison auditable.
                            for index in 1..entries.len() {
                                let mut current = index;
                                while current > 0 {
                                    self.charge_cpu(1)?;
                                    let should_swap =
                                        match (&entries[current - 1].0, &entries[current].0) {
                                            (Value::String(left), Value::String(right)) => {
                                                left > right
                                            }
                                            _ => unreachable!(),
                                        };
                                    if !should_swap {
                                        break;
                                    }
                                    entries.swap(current - 1, current);
                                    current -= 1;
                                }
                            }
                        }
                        active.push(*id);
                        let rendered = entries
                            .iter()
                            .map(|(key, value)| {
                                let Value::String(key) = key else {
                                    return Err("json.dumps currently requires string keys".into());
                                };
                                let key = serde_json::to_string(key)
                                    .map_err(|error| error.to_string())?;
                                let value = self.json_dump_value(
                                    value,
                                    item_separator,
                                    key_separator,
                                    sort_keys,
                                    depth + 1,
                                    active,
                                )?;
                                Ok(format!("{key}{key_separator}{value}"))
                            })
                            .collect::<Result<Vec<_>, String>>();
                        active.pop();
                        Ok(format!("{{{}}}", rendered?.join(item_separator)))
                    }
                    Object::Set(_)
                    | Object::Function { .. }
                    | Object::Class { .. }
                    | Object::Instance { .. }
                    | Object::PythonBoundMethod { .. }
                    | Object::Iterator { .. }
                    | Object::CountIterator { .. }
                    | Object::Generator { .. }
                    | Object::BoundMethod { .. }
                    | Object::Module { .. }
                    | Object::Regex { .. }
                    | Object::Match { .. }
                    | Object::ArgumentParser { .. }
                    | Object::Namespace { .. } => Err("object is not JSON serializable".into()),
                    Object::EnumMember { .. } => Err("object is not JSON serializable".into()),
                    Object::RaisesContext { .. } => Err("object is not JSON serializable".into()),
                }
            }
            Value::Float(_) => Err("non-finite float is not JSON serializable".into()),
            Value::Native(_) => Err("object is not JSON serializable".into()),
        }
    }

    fn reserve_slots(&mut self, slots: usize) -> Result<(), String> {
        let bytes = u64::try_from(slots)
            .ok()
            .and_then(|slots| slots.checked_mul(24))
            .ok_or("modeled object size overflow")?;
        self.state
            .heap
            .reserve_growth(bytes, &mut self.interp.resources)
    }

    /// Compute a deliberately generous bound for every host String/Vec built by JSON encoding.
    /// This runs before `json_dump_value`, whose convenient recursive implementation constructs
    /// child strings eagerly. The bound includes per-node allocator slack, not just final JSON
    /// bytes, so the convenience implementation remains within the modeled memory budget.
    fn json_size_bound(
        &mut self,
        value: &Value,
        depth: usize,
        active: &mut Vec<super::heap::ObjectId>,
    ) -> Result<usize, String> {
        if depth == 256 {
            return Err("maximum JSON nesting depth exceeded".into());
        }
        self.charge_cpu(1)?;
        let scalar = |size: usize| {
            size.checked_add(64)
                .ok_or_else(|| "json result is too large".to_string())
        };
        match value {
            Value::None => scalar(4),
            Value::Bool(value) => scalar(if *value { 4 } else { 5 }),
            Value::Int(_) => scalar(32),
            Value::Float(value) if value.is_finite() => scalar(64),
            // JSON escaping uses at most six ASCII bytes per input byte (`\\u00xx`), plus quotes.
            Value::String(value) => scalar(
                value
                    .len()
                    .checked_mul(6)
                    .and_then(|size| size.checked_add(2))
                    .ok_or("json result is too large")?,
            ),
            Value::Float(_) => Err("non-finite float is not JSON serializable".into()),
            Value::Exception { .. } | Value::Native(_) => {
                Err("object is not JSON serializable".into())
            }
            Value::Object(id) => {
                if active.contains(id) {
                    return Err("circular reference detected while encoding JSON".into());
                }
                active.push(*id);
                let result = match self.state.heap.get(*id)?.clone() {
                    Object::List(values) | Object::Tuple(values) => {
                        let mut size = 2usize;
                        for (index, child) in values.iter().enumerate() {
                            if index != 0 {
                                size = size.checked_add(1).ok_or("json result is too large")?;
                            }
                            size = size
                                .checked_add(self.json_size_bound(child, depth + 1, active)?)
                                .and_then(|size| size.checked_add(64))
                                .ok_or("json result is too large")?;
                        }
                        Ok(size)
                    }
                    Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                        let mut size = 2usize;
                        for (index, (key, child)) in entries.iter().enumerate() {
                            let Value::String(key) = key else {
                                return Err("json.dumps currently requires string keys".into());
                            };
                            if index != 0 {
                                size = size.checked_add(1).ok_or("json result is too large")?;
                            }
                            let key_size = key
                                .len()
                                .checked_mul(6)
                                .and_then(|size| size.checked_add(2))
                                .ok_or("json result is too large")?;
                            let child_size = self.json_size_bound(child, depth + 1, active)?;
                            size = size
                                .checked_add(key_size)
                                .and_then(|size| size.checked_add(1))
                                .and_then(|size| size.checked_add(child_size))
                                .and_then(|size| size.checked_add(64))
                                .ok_or("json result is too large")?;
                        }
                        Ok(size)
                    }
                    _ => Err("object is not JSON serializable".into()),
                };
                active.pop();
                result
            }
        }
    }

    fn iterable_values(&mut self, value: &Value) -> Result<Vec<Value>, String> {
        let mut result = Vec::new();
        match value {
            Value::String(value) => {
                for character in value.chars() {
                    self.push_materialized(&mut result, Value::String(character.to_string()))?;
                }
            }
            Value::Object(id) => match self.state.heap.get(*id)?.clone() {
                Object::List(values) | Object::Tuple(values) | Object::Set(values) => {
                    for value in values {
                        self.push_materialized(&mut result, value)?;
                    }
                }
                Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                    for (key, _) in entries {
                        self.push_materialized(&mut result, key)?;
                    }
                }
                Object::Iterator { values, position } => {
                    for value in values.into_iter().skip(position) {
                        self.push_materialized(&mut result, value)?;
                    }
                }
                Object::CountIterator { .. } => {
                    return Err(
                        "cannot materialize infinite itertools.count without a bound".into(),
                    )
                }
                Object::Generator { .. } => {
                    while let Some(value) = self.resume_generator(*id)? {
                        self.push_materialized(&mut result, value)?;
                    }
                }
                Object::Class { enum_members, .. } if !enum_members.is_empty() => {
                    for value in enum_members {
                        self.push_materialized(&mut result, value)?;
                    }
                }
                Object::Module { .. } => return Err("module object is not iterable".into()),
                _ => return Err("object is not iterable".into()),
            },
            _ => return Err("object is not iterable".into()),
        };
        Ok(result)
    }

    fn push_materialized(&mut self, values: &mut Vec<Value>, value: Value) -> Result<(), String> {
        // A host Vec has allocator/capacity overhead that is not represented in the Python heap.
        // Reserve a deliberately generous per-item amount before every push, including string
        // payloads, so repeated materialization cannot grow outside the memory budget.
        let payload = match &value {
            Value::String(text) => text.len().saturating_mul(2),
            _ => 0,
        };
        self.reserve_result(64usize.saturating_add(payload))?;
        self.charge_cpu(1)?;
        values.push(value);
        Ok(())
    }

    fn find_value(&mut self, values: &[Value], needle: &Value) -> Result<Option<usize>, String> {
        for (position, value) in values.iter().enumerate() {
            self.charge_cpu(1)?;
            if protocol::equals(&self.state.heap, value, needle)? {
                return Ok(Some(position));
            }
        }
        Ok(None)
    }

    fn dict_position(
        &mut self,
        id: super::heap::ObjectId,
        key: &Value,
    ) -> Result<Option<usize>, String> {
        let entries = match self.state.heap.get(id)?.clone() {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
            _ => return Err("invalid dictionary method receiver".into()),
        };
        for (position, (candidate, _)) in entries.iter().enumerate() {
            self.charge_cpu(1)?;
            if protocol::equals(&self.state.heap, candidate, key)? {
                return Ok(Some(position));
            }
        }
        Ok(None)
    }

    fn set_position(
        &mut self,
        id: super::heap::ObjectId,
        value: &Value,
    ) -> Result<Option<usize>, String> {
        let Object::Set(values) = self.state.heap.get(id)?.clone() else {
            return Err("invalid set method receiver".into());
        };
        self.find_value(&values, value)
    }

    fn pop(&mut self) -> Result<Value, String> {
        self.stack
            .pop()
            .ok_or_else(|| "invalid bytecode stack effect".into())
    }

    fn take(&mut self, count: usize) -> Result<Vec<Value>, String> {
        if self.stack.len() < count {
            return Err("invalid bytecode stack effect".into());
        }
        Ok(self.stack.split_off(self.stack.len() - count))
    }

    fn copy(&mut self, depth: usize) -> Result<(), String> {
        if depth == 0 || depth > self.stack.len() {
            return Err("invalid bytecode copy depth".into());
        }
        self.stack
            .push(self.stack[self.stack.len() - depth].clone());
        Ok(())
    }

    fn jump_if_or_pop(&mut self, jump_when: bool) -> Result<bool, String> {
        let value = self.stack.last().ok_or("invalid bytecode stack effect")?;
        if protocol::truth(&self.state.heap, value)? == jump_when {
            Ok(true)
        } else {
            self.stack.pop();
            Ok(false)
        }
    }

    fn swap(&mut self, depth: usize) -> Result<(), String> {
        if depth == 0 || depth > self.stack.len() {
            return Err("invalid bytecode swap depth".into());
        }
        let top = self.stack.len() - 1;
        let other = self.stack.len() - depth;
        self.stack.swap(top, other);
        Ok(())
    }

    fn reserve_result(&mut self, bytes: usize) -> Result<(), String> {
        let bytes = u64::try_from(bytes).map_err(|_| "string result is too large")?;
        if self.interp.resources.reserve_memory(bytes) {
            Ok(())
        } else {
            Err("memory limit exceeded".into())
        }
    }

    /// Append at most the output budget's remaining bytes. Charging happens before touching the
    /// host-owned Vec, so a Python print/write cannot bypass the hard output cap. The command
    /// dispatcher observes this already-accounted output and therefore will not charge it again.
    fn write_output(&mut self, stream: Stream, bytes: &[u8]) {
        let allowed = bytes
            .len()
            .min(self.interp.resources.output_remaining() as usize);
        let _ = self
            .interp
            .resources
            .charge_output(bytes.len().try_into().unwrap_or(u64::MAX));
        if allowed == 0 {
            return;
        }
        match stream {
            Stream::Stdout => self.out.extend_from_slice(&bytes[..allowed]),
            Stream::Stderr => self.err.extend_from_slice(&bytes[..allowed]),
        }
    }

    /// Charge a unit of VM-native work and turn exhaustion into the same bounded failure used by
    /// bytecode instructions. Native loops must use this rather than relying on a later opcode;
    /// otherwise a large host-side operation could complete after the budget was exhausted.
    fn charge_cpu(&mut self, units: u64) -> Result<(), String> {
        if self.interp.resources.charge_cpu(units) {
            Ok(())
        } else {
            Err("resource limit exceeded while executing Python".into())
        }
    }
}

enum CallResult {
    Value(Value),
    Exit(i32),
}

enum Execution {
    Halt,
    Return(Value),
    Yield(Value, usize),
    Exit(i32),
}

enum SequenceKind {
    List,
    Tuple,
}

fn identity(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::None, Value::None) => true,
        (Value::Bool(left), Value::Bool(right)) => left == right,
        (Value::Object(left), Value::Object(right)) => left == right,
        (Value::Native(left), Value::Native(right)) => left == right,
        _ => false,
    }
}

fn add_numbers(left: Value, right: Value) -> Result<Value, String> {
    match (left, right) {
        (Value::Int(left), Value::Int(right)) => left
            .checked_add(right)
            .map(Value::Int)
            .ok_or_else(|| "integer arithmetic exceeds the current bounded integer range".into()),
        (Value::Int(left), Value::Float(right)) => Ok(Value::Float(left as f64 + right)),
        (Value::Float(left), Value::Int(right)) => Ok(Value::Float(left + right as f64)),
        (Value::Float(left), Value::Float(right)) => Ok(Value::Float(left + right)),
        _ => Err("unsupported operands for sum()".into()),
    }
}

fn dynamic_heapify(heap: &super::heap::Heap, values: &mut [Value]) -> Result<(), String> {
    if values.len() < 2 {
        return Ok(());
    }
    for index in (0..values.len() / 2).rev() {
        dynamic_sift_down(heap, values, index)?;
    }
    Ok(())
}

fn dynamic_sift_down(
    heap: &super::heap::Heap,
    values: &mut [Value],
    mut parent: usize,
) -> Result<(), String> {
    loop {
        let left = parent * 2 + 1;
        if left >= values.len() {
            return Ok(());
        }
        let right = left + 1;
        let child = if right < values.len()
            && protocol::compare(heap, &values[right], &values[left])? == Ordering::Less
        {
            right
        } else {
            left
        };
        if protocol::compare(heap, &values[parent], &values[child])? != Ordering::Greater {
            return Ok(());
        }
        values.swap(parent, child);
        parent = child;
    }
}

fn object_receiver(value: Value) -> Result<super::heap::ObjectId, String> {
    if let Value::Object(id) = value {
        Ok(id)
    } else {
        Err("invalid method receiver".into())
    }
}

fn normalize_index(index: i64, length: usize, message: &str) -> Result<usize, String> {
    let length = i64::try_from(length).map_err(|_| message)?;
    let index = if index < 0 { length + index } else { index };
    if index < 0 || index >= length {
        Err(message.into())
    } else {
        usize::try_from(index).map_err(|_| message.into())
    }
}

fn split_string(value: &str, separator: Option<&str>, maximum: Option<i64>) -> Vec<String> {
    let unlimited = maximum.is_none_or(|maximum| maximum < 0);
    let limit = maximum
        .and_then(|maximum| usize::try_from(maximum).ok())
        .unwrap_or(usize::MAX);
    match separator {
        Some(separator) if unlimited => value.split(separator).map(str::to_string).collect(),
        Some(separator) => value
            .splitn(limit.saturating_add(1), separator)
            .map(str::to_string)
            .collect(),
        None if unlimited => value.split_whitespace().map(str::to_string).collect(),
        None => {
            let mut parts = value.split_whitespace();
            let mut result = Vec::new();
            for _ in 0..limit {
                let Some(part) = parts.next() else {
                    return result;
                };
                result.push(part.to_string());
            }
            let remainder = parts.collect::<Vec<_>>().join(" ");
            if !remainder.is_empty() {
                result.push(remainder);
            }
            result
        }
    }
}

fn value_from_constant(value: &Constant) -> Value {
    match value {
        Constant::None => Value::None,
        Constant::Bool(value) => Value::Bool(*value),
        Constant::Integer(value) => Value::Int(*value),
        Constant::Float(value) => Value::Float(*value),
        Constant::String(value) => Value::String(value.clone()),
    }
}

fn integer_binary(operator: BinaryOperator, left: i64, right: i64) -> Result<Value, String> {
    const OVERFLOW: &str = "integer arithmetic exceeds the current bounded integer range";
    match operator {
        BinaryOperator::Add => left
            .checked_add(right)
            .map(Value::Int)
            .ok_or(OVERFLOW.into()),
        BinaryOperator::Subtract => left
            .checked_sub(right)
            .map(Value::Int)
            .ok_or(OVERFLOW.into()),
        BinaryOperator::Multiply => left
            .checked_mul(right)
            .map(Value::Int)
            .ok_or(OVERFLOW.into()),
        BinaryOperator::Divide => {
            if right == 0 {
                Err("division by zero".into())
            } else {
                Ok(Value::Float(left as f64 / right as f64))
            }
        }
        BinaryOperator::FloorDivide => floor_div(left, right).map(Value::Int),
        BinaryOperator::Remainder => {
            let quotient = floor_div(left, right)?;
            left.checked_sub(quotient.checked_mul(right).ok_or(OVERFLOW)?)
                .map(Value::Int)
                .ok_or(OVERFLOW.into())
        }
    }
}

fn floor_div(left: i64, right: i64) -> Result<i64, String> {
    if right == 0 {
        return Err("integer division or modulo by zero".into());
    }
    let mut quotient = left
        .checked_div(right)
        .ok_or("integer arithmetic exceeds the current bounded integer range")?;
    let remainder = left % right;
    if remainder != 0 && (remainder < 0) != (right < 0) {
        quotient -= 1;
    }
    Ok(quotient)
}

fn float_binary(operator: BinaryOperator, left: f64, right: f64) -> Result<Value, String> {
    match operator {
        BinaryOperator::Add => Ok(Value::Float(left + right)),
        BinaryOperator::Subtract => Ok(Value::Float(left - right)),
        BinaryOperator::Multiply => Ok(Value::Float(left * right)),
        BinaryOperator::Divide if right != 0.0 => Ok(Value::Float(left / right)),
        BinaryOperator::FloorDivide if right != 0.0 => Ok(Value::Float((left / right).floor())),
        BinaryOperator::Remainder if right != 0.0 => {
            Ok(Value::Float(left - (left / right).floor() * right))
        }
        _ => Err("division by zero".into()),
    }
}

fn as_float(value: &Value) -> Option<f64> {
    match value {
        Value::Int(value) => Some(*value as f64),
        Value::Float(value) => Some(*value),
        _ => None,
    }
}

fn expect_arity(arguments: &[Value], minimum: usize, maximum: usize) -> Result<(), String> {
    if (minimum..=maximum).contains(&arguments.len()) {
        Ok(())
    } else {
        Err(format!(
            "expected {minimum}..={maximum} arguments, got {}",
            arguments.len()
        ))
    }
}

const RE_IGNORECASE: u32 = 2;
const RE_MULTILINE: u32 = 8;
const MAX_REGEX_PATTERN: usize = 4096;
const MAX_REGEX_INPUT: usize = 1_048_576;

fn string_argument(value: &Value, label: &str) -> Result<String, String> {
    match value {
        Value::String(value) if value.len() <= MAX_REGEX_INPUT => Ok(value.clone()),
        Value::String(_) => Err(format!("{label} exceeds the bounded regex input limit")),
        _ => Err(format!("{label} must be a string")),
    }
}

fn keyword_int(keywords: &[(String, Value)], name: &str) -> Result<Option<i64>, String> {
    let mut result = None;
    for (key, value) in keywords {
        if key == name {
            if result.is_some() {
                return Err(format!("got multiple values for keyword {name:?}"));
            }
            result = Some(
                value
                    .as_int()
                    .ok_or(format!("keyword {name:?} must be an integer"))?,
            );
        }
    }
    Ok(result)
}

fn keyword_string(keywords: &[(String, Value)], name: &str) -> Result<Option<String>, String> {
    let mut result = None;
    for (key, value) in keywords {
        if key == name {
            if result.is_some() {
                return Err(format!("got multiple values for keyword {name:?}"));
            }
            result = Some(string_argument(value, name)?);
        }
    }
    Ok(result)
}

fn keyword_value(keywords: &[(String, Value)], name: &str) -> Result<Option<Value>, String> {
    let mut result = None;
    for (key, value) in keywords {
        if key == name {
            if result.is_some() {
                return Err(format!("got multiple values for keyword {name:?}"));
            }
            result = Some(value.clone());
        }
    }
    Ok(result)
}

fn keyword_bool(keywords: &[(String, Value)], name: &str) -> Result<Option<bool>, String> {
    keyword_value(keywords, name)?
        .map(|value| match value {
            Value::Bool(value) => Ok(value),
            Value::Int(value) => Ok(value != 0),
            _ => Err(format!("keyword {name:?} must be a boolean")),
        })
        .transpose()
}

fn reject_unknown_keywords(keywords: &[(String, Value)], allowed: &[&str]) -> Result<(), String> {
    for (name, _) in keywords {
        if !allowed.contains(&name.as_str()) {
            return Err(format!("unexpected keyword argument {name:?}"));
        }
    }
    Ok(())
}

fn parse_argument_values(
    specs: &[super::heap::ArgumentSpec],
    input: &[String],
    _prog: &str,
) -> Result<Vec<(String, Value)>, String> {
    let mut values: Vec<(String, Value)> = specs
        .iter()
        .map(|spec| (spec.dest.clone(), spec.default.clone()))
        .collect();
    let mut positionals = specs
        .iter()
        .filter(|spec| !spec.names.iter().any(|name| name.starts_with('-')));
    let mut index = 0;
    while index < input.len() {
        let token = &input[index];
        if token == "--" {
            index += 1;
            continue;
        }
        let (name, attached) = token
            .split_once('=')
            .map_or((token.as_str(), None), |(name, value)| (name, Some(value)));
        let spec = specs
            .iter()
            .find(|spec| spec.names.iter().any(|candidate| candidate == name));
        if let Some(spec) = spec {
            let slot = values
                .iter_mut()
                .find(|(dest, _)| dest == &spec.dest)
                .ok_or("invalid parser state")?;
            if spec.store_true {
                slot.1 = Value::Bool(true);
            } else {
                let raw = if let Some(attached) = attached {
                    attached.to_string()
                } else {
                    index += 1;
                    input
                        .get(index)
                        .cloned()
                        .ok_or_else(|| format!("argument {name:?} expected one value"))?
                };
                slot.1 = if spec.integer {
                    Value::Int(
                        raw.parse()
                            .map_err(|_| format!("argument {name:?} must be an integer"))?,
                    )
                } else {
                    Value::String(raw)
                };
            }
        } else if token.starts_with('-') {
            return Err(format!("unrecognized argument {token:?}"));
        } else {
            let spec = positionals
                .next()
                .ok_or_else(|| format!("unrecognized argument {token:?}"))?;
            let slot = values
                .iter_mut()
                .find(|(dest, _)| dest == &spec.dest)
                .ok_or("invalid parser state")?;
            slot.1 = if spec.integer {
                Value::Int(
                    token
                        .parse()
                        .map_err(|_| format!("argument {token:?} must be an integer"))?,
                )
            } else {
                Value::String(token.clone())
            };
        }
        index += 1;
    }
    for spec in specs {
        if spec.required {
            let value = values
                .iter()
                .find(|(dest, _)| dest == &spec.dest)
                .map(|(_, value)| value);
            if value.is_none() || matches!(value, Some(Value::None)) {
                return Err(format!(
                    "the following arguments are required: {}",
                    spec.names.join(", ")
                ));
            }
        }
    }
    Ok(values)
}

fn regex_argument(
    arguments: &[Value],
    keywords: &[(String, Value)],
) -> Result<(String, u32), String> {
    let pattern = arguments.first().ok_or("regex pattern is required")?;
    let (pattern, mut flags) = match pattern {
        Value::String(pattern) => (pattern.clone(), 0),
        Value::Object(_) => {
            return Err("compiled regex values are only supported by methods".into())
        }
        _ => return Err("regex pattern must be a string".into()),
    };
    if let Some(value) = arguments.get(2) {
        flags = value.as_int().ok_or("regex flags must be an integer")? as u32;
    }
    if let Some(value) = keyword_int(keywords, "flags")? {
        if arguments.len() > 2 {
            return Err("regex flags passed more than once".into());
        }
        flags = value as u32;
    }
    if flags & !(RE_IGNORECASE | RE_MULTILINE) != 0 {
        return Err("regex flags are limited to IGNORECASE and MULTILINE".into());
    }
    Ok((pattern, flags))
}

fn build_regex(pattern: &str, flags: u32) -> Result<Regex, String> {
    if pattern.len() > MAX_REGEX_PATTERN {
        return Err("regex pattern exceeds the bounded pattern limit".into());
    }
    RegexBuilder::new(pattern)
        .case_insensitive(flags & RE_IGNORECASE != 0)
        .multi_line(flags & RE_MULTILINE != 0)
        .build()
        .map_err(|error| format!("unsupported regex pattern: {error}"))
}

fn normalize_replacement(replacement: &str) -> Result<String, String> {
    let mut output = String::with_capacity(replacement.len());
    let mut chars = replacement.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }
        let Some(next) = chars.next() else {
            return Err("bad escape in replacement".into());
        };
        if next.is_ascii_digit() {
            output.push('$');
            output.push(next);
        } else if next == '\\' {
            output.push('\\');
            output.push('\\');
        } else {
            output.push('\\');
            output.push(next);
        }
    }
    Ok(output)
}

fn python_re_escape(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ",:;!/".contains(ch) {
                ch.to_string()
            } else {
                format!("\\{ch}")
            }
        })
        .collect()
}
