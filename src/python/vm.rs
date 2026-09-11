//! Metered stack VM for the first vertical slice.

use crate::interp::Interp;

use std::cmp::Ordering;
use std::collections::HashMap;

use super::ast::{BinaryOperator, ComparisonOperator, Constant, UnaryOperator};
use super::bytecode::{ClassField, Code, Operation};
use super::heap::{ClassLayout, InstancePayload, Method, Object, ScopeId};
use super::native::{
    CallArgs, FunctionDef, ModuleDef, NativeTypeDef, PyCallable, PyClass, PyClock, PyDict,
    PyEnvironment, PyError, PyErrorKind, PyInstance, PyIterator, PyKind, PyList, PyMarker, PyMatch,
    PyMatchData, PyNativeKind, PyRegex, PyResult, PyRuntime, PyTuple, PyValueCast,
};
use super::{protocol, ExecResult, Out, ReplState, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeValue {
    Module(&'static ModuleDef),
    Function(Builtin),
    NativeFunction(&'static FunctionDef),
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
    Type,
    IsInstance,
    IsSubclass,
    Object,
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
    Write(Stream),
    EnvironmentGet,
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
                    has_metaclass,
                    fields,
                } => self.make_class(name.clone(), code, *bases, *has_metaclass, fields),
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
                    "type" => Builtin::Type,
                    "isinstance" => Builtin::IsInstance,
                    "issubclass" => Builtin::IsSubclass,
                    "object" => Builtin::Object,
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
        if let Some(module) = super::stdlib::native_module(name) {
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
        if let Value::Native(NativeValue::Module(module)) = &owner {
            if let Some(function) = module.function(name) {
                self.stack
                    .push(Value::Native(NativeValue::NativeFunction(function)));
                return Ok(());
            }
            if let Some(value) = module.value(name) {
                let value = value.get(self).map_err(|error| error.to_string())?;
                self.stack.push(value);
                return Ok(());
            }
        }
        if let Some(method) = self
            .native_type_def(&owner)?
            .and_then(|native_type| native_type.method(name))
        {
            let method = self.allocate_object(Object::NativeBoundMethod {
                receiver: owner,
                method,
            })?;
            self.stack.push(method);
            return Ok(());
        }
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
                Object::Class { .. } => {
                    let mut value = self.class_attribute(*id, name)?;
                    if value.is_none() {
                        let Object::Class { metaclass, .. } = self.state.heap.get(*id)? else {
                            unreachable!()
                        };
                        let metaclass = metaclass.clone();
                        if let Value::Object(metaclass) = metaclass {
                            value = self.class_attribute(metaclass, name)?;
                        }
                    }
                    let value = value.ok_or_else(|| format!("class has no attribute {name:?}"))?;
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
                Object::Instance { class, .. } => {
                    let instance = owner
                        .clone()
                        .cast::<PyInstance>(self)
                        .map_err(|error| error.to_string())?;
                    if let Some(value) = instance
                        .attribute(self, name)
                        .map_err(|error| error.to_string())?
                    {
                        self.stack.push(value);
                        return Ok(());
                    }
                    if self
                        .class_attribute(class, "__shellsim_unittest__")?
                        .is_some()
                    {
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
                    let value = self
                        .class_attribute(class, name)?
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
                | Object::NativeBoundMethod { .. }
                | Object::Iterator { .. }
                | Object::CountIterator { .. }
                | Object::Generator { .. }
                | Object::BoundMethod { .. }
                | Object::Module { .. }
                | Object::Regex { .. }
                | Object::Match { .. }
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
                | Object::NativeBoundMethod { .. }
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
            | Object::NativeBoundMethod { .. }
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
        has_metaclass: bool,
        fields: &[ClassField],
    ) -> Result<(), String> {
        let stack_values = base_count
            .checked_add(usize::from(has_metaclass))
            .ok_or("too many class construction values")?;
        if self.stack.len() < stack_values {
            return Err("invalid bytecode stack effect while creating class".into());
        }
        let explicit_metaclass = has_metaclass.then(|| self.stack.pop().expect("checked above"));
        let bases = self.stack.split_off(self.stack.len() - base_count);
        let is_enum = bases.len() == 1 && matches!(bases[0], Value::Native(NativeValue::EnumBase));
        let is_unittest =
            bases.len() == 1 && matches!(bases[0], Value::Native(NativeValue::UnitTestBase));
        let has_int_base = bases
            .iter()
            .any(|base| matches!(base, Value::Native(NativeValue::Function(Builtin::Integer))));
        let has_object_base = bases
            .iter()
            .any(|base| matches!(base, Value::Native(NativeValue::Function(Builtin::Object))));
        let has_type_base = bases
            .iter()
            .any(|base| matches!(base, Value::Native(NativeValue::Function(Builtin::Type))));
        let user_bases = if is_enum || is_unittest {
            Vec::new()
        } else {
            bases
                .iter()
                .filter_map(|base| match base {
                    Value::Object(id)
                        if matches!(self.state.heap.get(*id), Ok(Object::Class { .. })) =>
                    {
                        Some(Ok(*id))
                    }
                    Value::Native(NativeValue::Function(Builtin::Integer)) => None,
                    Value::Native(NativeValue::Function(Builtin::Object)) => None,
                    Value::Native(NativeValue::Function(Builtin::Type)) => None,
                    _ => Some(Err("class bases must be classes".to_string())),
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        if has_int_base && (bases.len() != 1 || is_enum || is_unittest) {
            return Err("int inheritance cannot yet be combined with another direct base".into());
        }
        if has_object_base && bases.len() != 1 {
            return Err("object cannot be combined with another direct base in this slice".into());
        }
        if has_type_base && bases.len() != 1 {
            return Err("type cannot be combined with another direct base in this slice".into());
        }
        let inherited_int_layouts = user_bases
            .iter()
            .filter_map(|base| match self.state.heap.get(*base) {
                Ok(Object::Class { layout, .. }) => Some(*layout == ClassLayout::Int),
                _ => None,
            })
            .filter(|is_int| *is_int)
            .count();
        if inherited_int_layouts > 1 {
            return Err("multiple bases have incompatible int instance layouts".into());
        }
        let inherited_type_layouts = user_bases
            .iter()
            .filter_map(|base| match self.state.heap.get(*base) {
                Ok(Object::Class { layout, .. }) => Some(*layout == ClassLayout::Type),
                _ => None,
            })
            .filter(|is_type| *is_type)
            .count();
        if inherited_type_layouts > 1 || (inherited_type_layouts == 1 && inherited_int_layouts == 1)
        {
            return Err("multiple bases have incompatible instance layouts".into());
        }
        let layout = if has_type_base || inherited_type_layouts == 1 {
            ClassLayout::Type
        } else if has_int_base || inherited_int_layouts == 1 {
            ClassLayout::Int
        } else {
            ClassLayout::Object
        };
        let default_metaclass = user_bases
            .first()
            .and_then(|base| match self.state.heap.get(*base) {
                Ok(Object::Class { metaclass, .. }) => Some(metaclass.clone()),
                _ => None,
            })
            .unwrap_or(Value::Native(NativeValue::Function(Builtin::Type)));
        let metaclass = explicit_metaclass.unwrap_or(default_metaclass);
        let valid_metaclass = matches!(
            metaclass,
            Value::Native(NativeValue::Function(Builtin::Type))
        ) || matches!(
            metaclass,
            Value::Object(id)
                if matches!(self.state.heap.get(id), Ok(Object::Class { layout: ClassLayout::Type, .. }))
        );
        if !valid_metaclass {
            return Err("metaclass must derive from type".into());
        }
        let mro = self.linearize_bases(&user_bases)?;
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
        if layout == ClassLayout::Type
            && ["__new__", "__init__", "__call__"]
                .iter()
                .any(|name| attributes.contains_key(*name))
        {
            return Err("custom metaclass construction hooks are not implemented".into());
        }
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
                bases: user_bases,
                mro,
                metaclass,
                layout,
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

    fn linearize_bases(
        &mut self,
        bases: &[super::heap::ObjectId],
    ) -> Result<Vec<super::heap::ObjectId>, String> {
        for (index, base) in bases.iter().enumerate() {
            if bases[..index].contains(base) {
                return Err("duplicate base class".into());
            }
        }
        let mut sequences = Vec::with_capacity(bases.len().saturating_add(1));
        for base in bases {
            let Object::Class { mro, .. } = self.state.heap.get(*base)? else {
                return Err("class base changed object kind".into());
            };
            let mut sequence = Vec::with_capacity(mro.len().saturating_add(1));
            sequence.push(*base);
            sequence.extend(mro.iter().copied());
            sequences.push(sequence);
        }
        sequences.push(bases.to_vec());

        let mut result = Vec::new();
        loop {
            sequences.retain(|sequence| !sequence.is_empty());
            if sequences.is_empty() {
                return Ok(result);
            }
            let candidate = sequences.iter().find_map(|sequence| {
                let head = sequence[0];
                (!sequences
                    .iter()
                    .any(|other| other.iter().skip(1).any(|item| *item == head)))
                .then_some(head)
            });
            let Some(candidate) = candidate else {
                return Err("cannot create a consistent method resolution order".into());
            };
            self.charge_cpu(u64::try_from(sequences.len()).unwrap_or(u64::MAX))?;
            result.push(candidate);
            for sequence in &mut sequences {
                if sequence.first() == Some(&candidate) {
                    sequence.remove(0);
                }
            }
        }
    }

    fn class_attribute(
        &mut self,
        class: super::heap::ObjectId,
        name: &str,
    ) -> Result<Option<Value>, String> {
        let Object::Class {
            attributes, mro, ..
        } = self.state.heap.get(class)?
        else {
            return Err("instance has an invalid class".into());
        };
        if let Some(value) = attributes.get(name) {
            return Ok(Some(value.clone()));
        }
        let ancestors = mro.clone();
        for ancestor in ancestors {
            self.charge_cpu(1)?;
            let Object::Class { attributes, .. } = self.state.heap.get(ancestor)? else {
                return Err("class MRO contains a non-class object".into());
            };
            if let Some(value) = attributes.get(name) {
                return Ok(Some(value.clone()));
            }
        }
        Ok(None)
    }

    fn type_of(&self, value: &Value) -> Result<Value, String> {
        let builtin = match value {
            Value::Int(_) => Some(Builtin::Integer),
            Value::Bool(_) => Some(Builtin::Boolean),
            Value::String(_) => Some(Builtin::String),
            Value::Object(id) => match self.state.heap.get(*id)? {
                Object::Instance { class, .. } => return Ok(Value::Object(*class)),
                Object::Class { metaclass, .. } => return Ok(metaclass.clone()),
                Object::List(_) => Some(Builtin::List),
                Object::Tuple(_) => Some(Builtin::Tuple),
                Object::Set(_) => Some(Builtin::Set),
                _ => None,
            },
            Value::Native(NativeValue::Function(_)) if self.is_type_value(value) => {
                Some(Builtin::Type)
            }
            _ => None,
        };
        builtin
            .map(|builtin| Value::Native(NativeValue::Function(builtin)))
            .ok_or_else(|| "type() is not implemented for this value kind".into())
    }

    fn is_instance(&self, value: &Value, class: &Value) -> Result<bool, String> {
        if matches!(class, Value::Native(NativeValue::Function(Builtin::Object))) {
            return Ok(true);
        }
        if matches!(
            class,
            Value::Native(NativeValue::Function(Builtin::Integer))
        ) && protocol::int_value(&self.state.heap, value).is_some()
        {
            return Ok(true);
        }
        let actual = self.type_of(value)?;
        self.is_subclass(&actual, class)
    }

    fn is_subclass(&self, class: &Value, base: &Value) -> Result<bool, String> {
        if !self.is_type_value(class) || !self.is_type_value(base) {
            return Err("isinstance() and issubclass() require a class argument".into());
        }
        if identity(class, base)
            || matches!(base, Value::Native(NativeValue::Function(Builtin::Object)))
        {
            return Ok(true);
        }
        if matches!(
            (class, base),
            (
                Value::Native(NativeValue::Function(Builtin::Boolean)),
                Value::Native(NativeValue::Function(Builtin::Integer))
            )
        ) {
            return Ok(true);
        }
        let Value::Object(class_id) = class else {
            return Ok(false);
        };
        let Object::Class { mro, layout, .. } = self.state.heap.get(*class_id)? else {
            return Ok(false);
        };
        match base {
            Value::Object(base_id) => Ok(mro.contains(base_id)),
            Value::Native(NativeValue::Function(Builtin::Integer)) => {
                Ok(*layout == ClassLayout::Int)
            }
            Value::Native(NativeValue::Function(Builtin::Type)) => Ok(*layout == ClassLayout::Type),
            _ => Ok(false),
        }
    }

    fn is_type_value(&self, value: &Value) -> bool {
        match value {
            Value::Native(NativeValue::Function(
                Builtin::Integer
                | Builtin::String
                | Builtin::Boolean
                | Builtin::List
                | Builtin::Tuple
                | Builtin::Set
                | Builtin::Object
                | Builtin::Type,
            )) => true,
            Value::Object(id) => matches!(self.state.heap.get(*id), Ok(Object::Class { .. })),
            _ => false,
        }
    }

    fn native_type_def(&self, value: &Value) -> Result<Option<&'static NativeTypeDef>, String> {
        let Value::Object(id) = value else {
            return Ok(None);
        };
        Ok(match self.state.heap.get(*id)? {
            Object::Regex { .. } => Some(&super::stdlib::re::PATTERN_TYPE),
            Object::Match { .. } => Some(&super::stdlib::re::MATCH_TYPE),
            _ => None,
        })
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
            (operator, value) => {
                let value = protocol::int_value(&self.state.heap, &value)
                    .ok_or("bad operand type for unary arithmetic")?;
                match operator {
                    UnaryOperator::Positive => Value::Int(value),
                    UnaryOperator::Negative => {
                        Value::Int(value.checked_neg().ok_or(
                            "integer arithmetic exceeds the current bounded integer range",
                        )?)
                    }
                    UnaryOperator::Not => unreachable!(),
                }
            }
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
        let right = match (&right, protocol::int_value(&self.state.heap, &right)) {
            (Value::Object(_), Some(value)) => Value::Int(value),
            _ => right,
        };
        let left = match (&left, protocol::int_value(&self.state.heap, &left)) {
            (Value::Object(_), Some(value)) => Value::Int(value),
            _ => left,
        };
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
                Object::NativeBoundMethod { receiver, method } => {
                    let call = CallArgs::new(arguments, keyword_arguments);
                    match (method.call)(self, receiver, call) {
                        Ok(value) => Ok(CallResult::Value(value)),
                        Err(PyError {
                            kind: PyErrorKind::Exit(status),
                            ..
                        }) => Ok(CallResult::Exit(status)),
                        Err(error) => Err(self.record_native_error(error)),
                    }
                }
                Object::Class {
                    name,
                    layout,
                    is_dataclass,
                    dataclass_fields,
                    enum_members,
                    ..
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
                    let payload = match layout {
                        ClassLayout::Object => InstancePayload::Object,
                        ClassLayout::Int => {
                            if !keyword_arguments.is_empty() {
                                return Err(format!("{name}() does not accept keyword arguments"));
                            }
                            let value = arguments
                                .first()
                                .map(|value| {
                                    protocol::int_value(&self.state.heap, value)
                                        .or_else(|| value.as_int())
                                        .ok_or_else(|| {
                                            format!("{name}() argument is not supported")
                                        })
                                })
                                .transpose()?
                                .unwrap_or(0);
                            if arguments.len() > 1 {
                                return Err(format!("{name}() expects at most one value"));
                            }
                            InstancePayload::Int(value)
                        }
                        ClassLayout::Type => {
                            return Err("direct custom metaclass calls are not implemented".into())
                        }
                    };
                    let instance = self.allocate_object(Object::Instance {
                        class: id,
                        payload,
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
                    } else if let Some(initializer) = self.class_attribute(id, "__init__")? {
                        let Value::Object(function) = initializer else {
                            return Err(format!("{name}.__init__ is not callable"));
                        };
                        let Object::Function {
                            name: function_name,
                            code,
                            closure,
                            defaults,
                        } = self.state.heap.get(function)?.clone()
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
                    } else if layout == ClassLayout::Object
                        && (!arguments.is_empty() || !keyword_arguments.is_empty())
                    {
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
        if let Value::Native(NativeValue::NativeFunction(function)) = function {
            let call = CallArgs::new(arguments, keyword_arguments);
            return match (function.call)(self, call) {
                Ok(value) => Ok(CallResult::Value(value)),
                Err(PyError {
                    kind: PyErrorKind::Exit(status),
                    ..
                }) => Ok(CallResult::Exit(status)),
                Err(error) => Err(self.record_native_error(error)),
            };
        }
        let Value::Native(NativeValue::Function(function)) = function else {
            return Err("object is not callable".into());
        };
        if !keyword_arguments.is_empty() && !matches!(function, Builtin::Sorted) {
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
                    .map(|value| {
                        protocol::int_value(&self.state.heap, value)
                            .or_else(|| value.as_int())
                            .ok_or("int() argument is not supported")
                    })
                    .transpose()?
                    .unwrap_or(0);
                Ok(CallResult::Value(Value::Int(value)))
            }
            Builtin::Type => {
                expect_arity(&arguments, 1, 1)?;
                Ok(CallResult::Value(self.type_of(&arguments[0])?))
            }
            Builtin::IsInstance => {
                expect_arity(&arguments, 2, 2)?;
                Ok(CallResult::Value(Value::Bool(
                    self.is_instance(&arguments[0], &arguments[1])?,
                )))
            }
            Builtin::IsSubclass => {
                expect_arity(&arguments, 2, 2)?;
                Ok(CallResult::Value(Value::Bool(
                    self.is_subclass(&arguments[0], &arguments[1])?,
                )))
            }
            Builtin::Object => {
                expect_arity(&arguments, 0, 0)?;
                if !keyword_arguments.is_empty() {
                    return Err("object() does not accept keyword arguments".into());
                }
                Err("direct object() instances are not implemented".into())
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
                        | Object::NativeBoundMethod { .. }
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
            Builtin::Write(stream) => {
                expect_arity(&arguments, 1, 1)?;
                let text = protocol::display(&self.state.heap, &arguments[0])?;
                self.write_output(stream, text.as_bytes());
                Ok(CallResult::Value(Value::Int(text.chars().count() as i64)))
            }
            Builtin::EnvironmentGet => {
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

    fn record_native_error(&mut self, error: PyError) -> String {
        let kind = match error.kind {
            PyErrorKind::Type => Some("TypeError"),
            PyErrorKind::Value => Some("ValueError"),
            PyErrorKind::ZeroDivision => Some("ZeroDivisionError"),
            PyErrorKind::Overflow => Some("OverflowError"),
            PyErrorKind::Runtime => Some("RuntimeError"),
            PyErrorKind::Exception(kind) => Some(kind),
            PyErrorKind::Resource | PyErrorKind::Exit(_) => None,
        };
        if let Some(kind) = kind {
            let value = Value::Exception {
                kind: kind.to_string(),
                message: error.message.clone(),
            };
            self.pending_exception = Some(RaisedException {
                kind: kind.to_string(),
                value,
            });
        }
        error.message
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

    fn reserve_slots(&mut self, slots: usize) -> Result<(), String> {
        let bytes = u64::try_from(slots)
            .ok()
            .and_then(|slots| slots.checked_mul(24))
            .ok_or("modeled object size overflow")?;
        self.state
            .heap
            .reserve_growth(bytes, &mut self.interp.resources)
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

impl PyRuntime for Vm<'_> {
    fn reserve_memory(&mut self, bytes: usize) -> PyResult<()> {
        self.reserve_result(bytes).map_err(PyError::resource_error)
    }

    fn charge_cpu(&mut self, units: u64) -> PyResult<()> {
        Vm::charge_cpu(self, units).map_err(PyError::resource_error)
    }

    fn kind(&self, value: &Value) -> PyResult<PyKind> {
        Ok(match value {
            Value::None => PyKind::None,
            Value::Bool(_) => PyKind::Bool,
            Value::Int(_) => PyKind::Int,
            Value::Float(_) => PyKind::Float,
            Value::String(_) => PyKind::String,
            Value::Exception { .. } | Value::Native(_) => PyKind::Native,
            Value::Object(id) => match self.state.heap.get(*id).map_err(PyError::runtime_error)? {
                Object::List(_) => PyKind::List,
                Object::Tuple(_) => PyKind::Tuple,
                Object::Dict(_) | Object::DefaultDict { .. } => PyKind::Dict,
                Object::Set(_) => PyKind::Set,
                Object::Function { .. }
                | Object::PythonBoundMethod { .. }
                | Object::NativeBoundMethod { .. }
                | Object::BoundMethod { .. } => PyKind::Function,
                Object::Class { .. } => PyKind::Class,
                Object::Instance { .. } | Object::EnumMember { .. } => PyKind::Instance,
                Object::Iterator { .. } | Object::CountIterator { .. } => PyKind::Iterator,
                Object::Generator { .. } => PyKind::Generator,
                Object::Module { .. } => PyKind::Module,
                Object::Regex { .. }
                | Object::Match { .. }
                | Object::ArgumentParser { .. }
                | Object::Namespace { .. }
                | Object::RaisesContext { .. } => PyKind::Native,
            },
        })
    }

    fn native_kind(&self, value: &Value) -> PyResult<Option<PyNativeKind>> {
        let Value::Object(id) = value else {
            return Ok(None);
        };
        Ok(
            match self.state.heap.get(*id).map_err(PyError::runtime_error)? {
                Object::Regex { .. } => Some(PyNativeKind::Regex),
                Object::Match { .. } => Some(PyNativeKind::Match),
                _ => None,
            },
        )
    }

    fn int_value(&self, value: &Value) -> Option<i64> {
        protocol::int_value(&self.state.heap, value)
    }

    fn truth(&self, value: &Value) -> PyResult<bool> {
        protocol::truth(&self.state.heap, value).map_err(PyError::runtime_error)
    }

    fn display(&self, value: &Value) -> PyResult<String> {
        protocol::display(&self.state.heap, value).map_err(PyError::runtime_error)
    }

    fn compare(&self, left: &Value, right: &Value) -> PyResult<Ordering> {
        protocol::compare(&self.state.heap, left, right).map_err(PyError::type_error)
    }

    fn list_items(&mut self, list: PyList) -> PyResult<Vec<Value>> {
        let id = list.object_id();
        let length = match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::List(items) => items.len(),
            _ => return Err(PyError::runtime_error("list handle changed object kind")),
        };
        let bytes = length
            .checked_mul(std::mem::size_of::<Value>())
            .ok_or_else(|| PyError::resource_error("list snapshot size overflow"))?;
        self.reserve_memory(bytes)?;
        match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::List(items) => Ok(items.clone()),
            _ => Err(PyError::runtime_error("list handle changed object kind")),
        }
    }

    fn tuple_items(&mut self, tuple: PyTuple) -> PyResult<Vec<Value>> {
        let id = tuple.object_id();
        let length = match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::Tuple(items) => items.len(),
            _ => return Err(PyError::runtime_error("tuple handle changed object kind")),
        };
        let bytes = length
            .checked_mul(std::mem::size_of::<Value>())
            .ok_or_else(|| PyError::resource_error("tuple snapshot size overflow"))?;
        self.reserve_memory(bytes)?;
        match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::Tuple(items) => Ok(items.clone()),
            _ => Err(PyError::runtime_error("tuple handle changed object kind")),
        }
    }

    fn dict_items(&mut self, dict: PyDict) -> PyResult<Vec<(Value, Value)>> {
        let id = dict.object_id();
        let length = match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::Dict(items) | Object::DefaultDict { entries: items, .. } => items.len(),
            _ => return Err(PyError::runtime_error("dict handle changed object kind")),
        };
        let bytes = length
            .checked_mul(std::mem::size_of::<(Value, Value)>())
            .ok_or_else(|| PyError::resource_error("dict snapshot size overflow"))?;
        self.reserve_memory(bytes)?;
        match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::Dict(items) | Object::DefaultDict { entries: items, .. } => Ok(items.clone()),
            _ => Err(PyError::runtime_error("dict handle changed object kind")),
        }
    }

    fn instance_attribute(&self, instance: PyInstance, name: &str) -> PyResult<Option<Value>> {
        match self
            .state
            .heap
            .get(instance.object_id())
            .map_err(PyError::runtime_error)?
        {
            Object::Instance { attributes, .. } => Ok(attributes.get(name).cloned()),
            _ => Err(PyError::runtime_error(
                "instance handle changed object kind",
            )),
        }
    }

    fn replace_list_items(&mut self, list: PyList, items: Vec<Value>) -> PyResult<()> {
        let Object::List(destination) = self
            .state
            .heap
            .get_mut(list.object_id())
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("list handle changed object kind"));
        };
        *destination = items;
        Ok(())
    }

    fn call_value(&mut self, callable: Value, args: CallArgs) -> PyResult<Value> {
        let (positional, keywords) = args.into_parts();
        let argument_count = positional.len();
        let total = argument_count
            .checked_add(keywords.len())
            .ok_or_else(|| PyError::resource_error("too many call arguments"))?;
        let unpacked = vec![false; total];
        let keyword_names = keywords
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        self.stack.push(callable);
        self.stack.extend(positional);
        self.stack
            .extend(keywords.into_iter().map(|(_, value)| value));
        match self
            .call(argument_count, &keyword_names, &unpacked)
            .map_err(PyError::runtime_error)?
        {
            CallResult::Value(value) => Ok(value),
            CallResult::Exit(status) => Err(PyError::exit(status)),
        }
    }

    fn is_callable(&self, value: &Value) -> PyResult<bool> {
        Ok(match value {
            Value::Native(
                NativeValue::Function(_)
                | NativeValue::NativeFunction(_)
                | NativeValue::ExceptionType(_),
            ) => true,
            Value::Object(id) => matches!(
                self.state.heap.get(*id).map_err(PyError::runtime_error)?,
                Object::Function { .. }
                    | Object::Class { .. }
                    | Object::BoundMethod { .. }
                    | Object::PythonBoundMethod { .. }
            ),
            _ => false,
        })
    }

    fn iterator(&mut self, value: Value) -> PyResult<PyIterator> {
        if let Value::Object(id) = value {
            if matches!(
                self.state.heap.get(id).map_err(PyError::runtime_error)?,
                Object::Iterator { .. } | Object::CountIterator { .. } | Object::Generator { .. }
            ) {
                return Value::Object(id).cast(self);
            }
        }
        let values = self.iterable_values(&value).map_err(PyError::type_error)?;
        self.new_iterator(values)?.cast(self)
    }

    fn iterator_next(&mut self, iterator: PyIterator) -> PyResult<Option<Value>> {
        let id = iterator.object_id();
        match self
            .state
            .heap
            .get(id)
            .map_err(PyError::runtime_error)?
            .clone()
        {
            Object::Iterator { values, position } => {
                let value = values.get(position).cloned();
                if value.is_some() {
                    let Object::Iterator { position, .. } = self
                        .state
                        .heap
                        .get_mut(id)
                        .map_err(PyError::runtime_error)?
                    else {
                        return Err(PyError::runtime_error("iterator changed object kind"));
                    };
                    *position += 1;
                }
                Ok(value)
            }
            Object::CountIterator { current, step } => {
                let value = current;
                let next = super::stdlib::itertools::count_next(current, step)
                    .map_err(PyError::overflow_error)?;
                let Object::CountIterator { current, .. } = self
                    .state
                    .heap
                    .get_mut(id)
                    .map_err(PyError::runtime_error)?
                else {
                    return Err(PyError::runtime_error("iterator changed object kind"));
                };
                *current = next;
                Ok(Some(Value::Int(value)))
            }
            Object::Generator { .. } => self.resume_generator(id).map_err(PyError::runtime_error),
            _ => Err(PyError::type_error("expected an iterator")),
        }
    }

    fn new_iterator(&mut self, values: Vec<Value>) -> PyResult<Value> {
        Vm::allocate_object(
            self,
            Object::Iterator {
                values,
                position: 0,
            },
        )
        .map_err(PyError::resource_error)
    }

    fn new_count_iterator(&mut self, start: i64, step: i64) -> PyResult<Value> {
        Vm::allocate_object(
            self,
            Object::CountIterator {
                current: start,
                step,
            },
        )
        .map_err(PyError::resource_error)
    }

    fn new_default_dict(&mut self, factory: PyCallable) -> PyResult<Value> {
        Vm::allocate_object(
            self,
            Object::DefaultDict {
                factory: factory.into_value(),
                entries: Vec::new(),
            },
        )
        .map_err(PyError::resource_error)
    }

    fn new_list(&mut self, items: Vec<Value>) -> PyResult<Value> {
        Vm::allocate_object(self, Object::List(items)).map_err(PyError::resource_error)
    }

    fn new_tuple(&mut self, items: Vec<Value>) -> PyResult<Value> {
        Vm::allocate_object(self, Object::Tuple(items)).map_err(PyError::resource_error)
    }

    fn new_dict(&mut self, items: Vec<(Value, Value)>) -> PyResult<Value> {
        Vm::allocate_object(self, Object::Dict(items)).map_err(PyError::resource_error)
    }

    fn new_regex(&mut self, pattern: String, flags: u32) -> PyResult<Value> {
        Vm::allocate_object(self, Object::Regex { pattern, flags }).map_err(PyError::resource_error)
    }

    fn new_match(
        &mut self,
        text: String,
        groups: Vec<Option<String>>,
        start: usize,
        end: usize,
    ) -> PyResult<Value> {
        Vm::allocate_object(
            self,
            Object::Match {
                text,
                groups,
                start,
                end,
            },
        )
        .map_err(PyError::resource_error)
    }

    fn regex_parts(&mut self, regex: PyRegex) -> PyResult<(String, u32)> {
        let Object::Regex { pattern, flags } = self
            .state
            .heap
            .get(regex.object_id())
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("regex handle changed object kind"));
        };
        let pattern = pattern.clone();
        let flags = *flags;
        self.reserve_memory(pattern.len())?;
        Ok((pattern, flags))
    }

    fn match_data(&mut self, matched: PyMatch) -> PyResult<PyMatchData> {
        let Object::Match {
            groups, start, end, ..
        } = self
            .state
            .heap
            .get(matched.object_id())
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("match handle changed object kind"));
        };
        let bytes = groups.iter().try_fold(0usize, |total, group| {
            total.checked_add(group.as_ref().map_or(0, String::len))
        });
        let bytes = bytes.ok_or_else(|| PyError::resource_error("match snapshot is too large"))?;
        let groups = groups.clone();
        let start = *start;
        let end = *end;
        self.reserve_memory(bytes)?;
        Ok(PyMatchData { groups, start, end })
    }

    fn marker(&self, marker: PyMarker) -> Value {
        Value::Native(match marker {
            PyMarker::TypingList => NativeValue::TypingList,
            PyMarker::EnumBase => NativeValue::EnumBase,
            PyMarker::UnitTestBase => NativeValue::UnitTestBase,
            PyMarker::Environment => NativeValue::Environment,
            PyMarker::Stdout => NativeValue::Stream(Stream::Stdout),
            PyMarker::Stderr => NativeValue::Stream(Stream::Stderr),
        })
    }

    fn mark_dataclass(&mut self, class: PyClass) -> PyResult<()> {
        let Object::Class { is_dataclass, .. } = self
            .state
            .heap
            .get_mut(class.object_id())
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("class handle changed object kind"));
        };
        *is_dataclass = true;
        Ok(())
    }

    fn argv0(&self) -> String {
        self.argv.first().cloned().unwrap_or_else(|| "-".into())
    }

    fn new_argv(&mut self) -> PyResult<Value> {
        self.new_list(self.argv.iter().cloned().map(Value::String).collect())
    }

    fn new_argument_parser(&mut self, program: String) -> PyResult<Value> {
        Vm::allocate_object(
            self,
            Object::ArgumentParser {
                prog: program,
                arguments: Vec::new(),
            },
        )
        .map_err(PyError::resource_error)
    }

    fn new_namespace(&mut self, values: Vec<(String, Value)>) -> PyResult<Value> {
        Vm::allocate_object(self, Object::Namespace { values }).map_err(PyError::resource_error)
    }

    fn new_raises_context(&mut self, expected: String) -> PyResult<Value> {
        Vm::allocate_object(self, Object::RaisesContext { expected })
            .map_err(PyError::resource_error)
    }

    fn exception_type_name(&self, value: &Value) -> Option<&'static str> {
        match value {
            Value::Native(NativeValue::ExceptionType(ExceptionType(name))) => Some(name),
            _ => None,
        }
    }

    fn clock(&mut self) -> &mut dyn PyClock {
        self
    }

    fn environment(&self) -> &dyn PyEnvironment {
        self
    }
}

impl PyClock for Vm<'_> {
    fn wall_time(&self) -> PyResult<f64> {
        self.interp
            .clock
            .wall_time_seconds()
            .map_err(|error| PyError::runtime_error(error.to_string()))
    }

    fn wall_time_ns(&self) -> PyResult<i64> {
        let nanos = self
            .interp
            .clock
            .wall_time_ns()
            .map_err(|error| PyError::runtime_error(error.to_string()))?;
        i64::try_from(nanos)
            .map_err(|_| PyError::overflow_error("wall clock is outside Python int range"))
    }

    fn monotonic(&self) -> f64 {
        self.interp.clock.monotonic_seconds()
    }

    fn monotonic_ns(&self) -> PyResult<i64> {
        i64::try_from(self.interp.clock.monotonic_ns())
            .map_err(|_| PyError::overflow_error("monotonic clock is outside Python int range"))
    }

    fn process_time(&self) -> f64 {
        self.interp.resources.process_time_seconds()
    }

    fn process_time_ns(&self) -> PyResult<i64> {
        i64::try_from(self.interp.resources.process_time_ns())
            .map_err(|_| PyError::overflow_error("process clock is outside Python int range"))
    }

    fn sleep(&mut self, seconds: f64) -> PyResult<()> {
        let nanos = seconds * crate::clock::NANOS_PER_SECOND as f64;
        if nanos > u64::MAX as f64 {
            return Err(PyError::overflow_error("time.sleep() length is too large"));
        }
        match self
            .interp
            .clock
            .block_task(crate::clock::MAIN_TASK_ID, nanos as u64)
            .map_err(|error| PyError::runtime_error(error.to_string()))?
        {
            crate::clock::BlockOutcome::Completed => {}
            crate::clock::BlockOutcome::Interrupted(event) => {
                self.interp.deadline_interrupt = Some(event.id);
            }
        }
        Ok(())
    }
}

impl PyEnvironment for Vm<'_> {
    fn get(&self, name: &str) -> Option<String> {
        self.interp.get_var(name)
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

const MAX_REGEX_INPUT: usize = 1_048_576;

fn string_argument(value: &Value, label: &str) -> Result<String, String> {
    match value {
        Value::String(value) if value.len() <= MAX_REGEX_INPUT => Ok(value.clone()),
        Value::String(_) => Err(format!("{label} exceeds the bounded regex input limit")),
        _ => Err(format!("{label} must be a string")),
    }
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
