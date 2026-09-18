//! Metered stack VM for the first vertical slice.

use crate::interp::Interp;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use num_bigint::BigInt;
use num_traits::ToPrimitive;

use super::ast::{BinaryOperator, ComparisonOperator, Constant, UnaryOperator};
use super::bytecode::{ClassField, Code, Operation};
use super::filesystem::PyModuleLoader;
use super::heap::{ClassLayout, InstancePayload, Object, ScopeId};
use super::native::{
    CallArgs, FunctionDef, ModuleDef, PyArgumentParser, PyArgumentParserData, PyArgumentSpec,
    PyArray, PyArrayDtype, PyArrayLayout, PyBinaryOp, PyByteArray, PyCallable, PyClass, PyClock,
    PyDict, PyEnvironment, PyError, PyErrorKind, PyFilesystem, PyIdentity, PyInstance, PyIterator,
    PyKind, PyList, PyMarker, PyMatch, PyMatchData, PyModule, PyNativeKind, PyProcessHandle,
    PyProcessOutput, PyProcessRunner, PyProcessStartRequest, PyProperty, PyRaisesContext, PyRegex,
    PyResult, PyRuntime, PySet, PySubcommandSpec, PySubparsersSpec, PyTuple, PyValueCast,
};
use super::object_model::{BuiltinType, PyLayout, Slot, SlotValue, TypeId};
use super::{protocol, ExecResult, Out, ReplState, Value, ValueTag};

const VM_POLL_QUANTUM: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeValue {
    Module(&'static ModuleDef),
    Function(Builtin),
    BuiltinType(BuiltinType),
    NativeFunction(&'static FunctionDef),
    NativeMethod(&'static super::native::MethodDef),
    ValueKind(&'static super::native::ValueKindDef),
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

impl NativeValue {
    const MODULE: u8 = 0;
    const FUNCTION: u8 = 1;
    const BUILTIN_TYPE: u8 = 2;
    const NATIVE_FUNCTION: u8 = 3;
    const NATIVE_METHOD: u8 = 4;
    const STREAM: u8 = 5;
    const ENVIRONMENT: u8 = 6;
    const EXCEPTION_TYPE: u8 = 7;
    const TYPING_LIST: u8 = 8;
    const ENUM_BASE: u8 = 9;
    const UNITTEST_BASE: u8 = 10;
    const VALUE_KIND: u8 = 11;

    pub(super) fn encode(self) -> (u64, u8) {
        match self {
            Self::Module(value) => (value as *const ModuleDef as usize as u64, Self::MODULE),
            Self::Function(value) => (value as u64, Self::FUNCTION),
            Self::BuiltinType(value) => (value as u64, Self::BUILTIN_TYPE),
            Self::NativeFunction(value) => (
                value as *const FunctionDef as usize as u64,
                Self::NATIVE_FUNCTION,
            ),
            Self::NativeMethod(value) => (
                value as *const super::native::MethodDef as usize as u64,
                Self::NATIVE_METHOD,
            ),
            Self::ValueKind(value) => (
                value as *const super::native::ValueKindDef as usize as u64,
                Self::VALUE_KIND,
            ),
            Self::Stream(value) => (value as u64, Self::STREAM),
            Self::Environment => (0, Self::ENVIRONMENT),
            Self::ExceptionType(value) => (exception_type_code(value.0), Self::EXCEPTION_TYPE),
            Self::TypingList => (0, Self::TYPING_LIST),
            Self::EnumBase => (0, Self::ENUM_BASE),
            Self::UnitTestBase => (0, Self::UNITTEST_BASE),
        }
    }

    /// Decode a handle produced by [`Self::encode`]. Static definition pointers are safe to
    /// recover because the private constructor accepts only `'static` references and values never
    /// cross interpreter processes or serialization boundaries.
    pub(super) fn decode(payload: u64, kind: u8) -> Self {
        match kind {
            Self::MODULE => {
                // SAFETY: `encode` stores a non-null pointer to a static `ModuleDef`.
                Self::Module(unsafe { &*(payload as usize as *const ModuleDef) })
            }
            Self::FUNCTION => {
                // SAFETY: the payload originates from the fieldless `Builtin` enum below.
                Self::Function(unsafe { std::mem::transmute::<u8, Builtin>(payload as u8) })
            }
            Self::BUILTIN_TYPE => Self::BuiltinType(
                // SAFETY: the payload originates from the `repr(u32)` `BuiltinType` enum.
                unsafe { std::mem::transmute::<u32, BuiltinType>(payload as u32) },
            ),
            Self::NATIVE_FUNCTION => {
                // SAFETY: `encode` stores a non-null pointer to a static `FunctionDef`.
                Self::NativeFunction(unsafe { &*(payload as usize as *const FunctionDef) })
            }
            Self::NATIVE_METHOD => {
                // SAFETY: `encode` stores a non-null pointer to a static `MethodDef`.
                Self::NativeMethod(unsafe {
                    &*(payload as usize as *const super::native::MethodDef)
                })
            }
            Self::STREAM => Self::Stream(match payload {
                0 => Stream::Stdin,
                1 => Stream::Stdout,
                _ => Stream::Stderr,
            }),
            Self::ENVIRONMENT => Self::Environment,
            Self::EXCEPTION_TYPE => {
                Self::ExceptionType(ExceptionType(exception_type_name(payload)))
            }
            Self::TYPING_LIST => Self::TypingList,
            Self::ENUM_BASE => Self::EnumBase,
            Self::UNITTEST_BASE => Self::UnitTestBase,
            Self::VALUE_KIND => {
                // SAFETY: `encode` stores a non-null pointer to a static `ValueKindDef`.
                Self::ValueKind(unsafe {
                    &*(payload as usize as *const super::native::ValueKindDef)
                })
            }
            _ => unreachable!("invalid private native-value tag"),
        }
    }
}

fn exception_type_code(name: &str) -> u64 {
    match name {
        "Exception" => 0,
        "BaseException" => 1,
        "AssertionError" => 2,
        "TypeError" => 3,
        "ValueError" => 4,
        "RuntimeError" => 5,
        "ZeroDivisionError" => 6,
        "OverflowError" => 7,
        "KeyError" => 8,
        "IndexError" => 9,
        "StopIteration" => 10,
        "Skipped" => 11,
        "Failed" => 12,
        "CalledProcessError" => 13,
        "TimeoutExpired" => 14,
        "EOFError" => 15,
        "OSError" => 16,
        "FileNotFoundError" => 17,
        "FileExistsError" => 18,
        "IsADirectoryError" => 19,
        "NotADirectoryError" => 20,
        "PermissionError" => 21,
        "SystemExit" => 22,
        _ => unreachable!("exception type must come from the closed builtin table"),
    }
}

fn exception_type_name(code: u64) -> &'static str {
    match code {
        0 => "Exception",
        1 => "BaseException",
        2 => "AssertionError",
        3 => "TypeError",
        4 => "ValueError",
        5 => "RuntimeError",
        6 => "ZeroDivisionError",
        7 => "OverflowError",
        8 => "KeyError",
        9 => "IndexError",
        10 => "StopIteration",
        11 => "Skipped",
        12 => "Failed",
        13 => "CalledProcessError",
        14 => "TimeoutExpired",
        15 => "EOFError",
        16 => "OSError",
        17 => "FileNotFoundError",
        18 => "FileExistsError",
        19 => "IsADirectoryError",
        20 => "NotADirectoryError",
        21 => "PermissionError",
        22 => "SystemExit",
        _ => unreachable!("invalid private exception-type handle"),
    }
}

impl NativeValue {
    pub(super) fn repr(self) -> String {
        match self {
            Self::BuiltinType(builtin_type) => format!("<class '{}'>", builtin_type.name()),
            Self::ValueKind(kind) => format!("<class '{}'>", kind.name),
            _ => "<native object>".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExceptionType(pub &'static str);

#[derive(Clone, Debug)]
struct RaisedException {
    kind: String,
    value: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Stream {
    Stdin,
    Stdout,
    Stderr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Builtin {
    Print,
    Input,
    Exit,
    Repr,
    IsInstance,
    IsSubclass,
    Length,
    Sorted,
    Minimum,
    Maximum,
    Sum,
    Absolute,
    Range,
    Enumerate,
    Zip,
    Any,
    All,
    Iter,
    Next,
    Property,
    StaticMethod,
    ClassMethod,
    Super,
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
    state
        .locals
        .entry("__name__".into())
        .or_insert_with(|| Value::inline_string("__main__").expect("short builtin string"));
    let mut program = match VmProgram::compile(source) {
        Ok(program) => program,
        Err(result) => return result,
    };
    loop {
        match program.poll(
            interp,
            ProcessInput { argv, stdin: &[] },
            state,
            VmMode::synchronous(interactive),
            out,
            err,
        ) {
            VmPoll::Runnable => {}
            VmPoll::Blocked(_) => unreachable!("synchronous Python cannot suspend"),
            VmPoll::Ready(result) => return result,
        }
    }
}

/// Compiled Python and the transient bytecode state retained between bounded VM polls.
#[derive(Clone)]
pub(super) struct VmProgram {
    code: Code,
    execution: VmState,
    started: bool,
}

pub(super) enum VmPoll {
    Runnable,
    Blocked(crate::scheduler::WaitReason),
    Ready(ExecResult),
}

#[derive(Clone, Copy)]
pub(super) struct VmMode {
    interactive: bool,
    scheduler_owned: bool,
}

impl VmMode {
    pub(super) fn synchronous(interactive: bool) -> Self {
        Self {
            interactive,
            scheduler_owned: false,
        }
    }

    pub(super) fn scheduled() -> Self {
        Self {
            interactive: false,
            scheduler_owned: true,
        }
    }
}

impl VmProgram {
    pub(super) fn compile(source: &str) -> Result<Self, ExecResult> {
        let tokens = super::lexer::lex(source).map_err(|error| {
            ExecResult::Unsupported(format!(
                "{} at line {}, column {}",
                error.message, error.span.line, error.span.column
            ))
        })?;
        let program = super::parser::parse(tokens).map_err(|error| {
            ExecResult::Unsupported(format!(
                "{} at line {}, column {}",
                error.message, error.span.line, error.span.column
            ))
        })?;
        Ok(Self {
            code: super::compiler::compile(program),
            execution: VmState::default(),
            started: false,
        })
    }

    /// Execute one bounded bytecode quantum, returning `None` while work remains runnable.
    pub(super) fn poll(
        &mut self,
        interp: &mut Interp,
        input: ProcessInput<'_>,
        state: &mut ReplState,
        mode: VmMode,
        out: Out,
        err: Out,
    ) -> VmPoll {
        let mut vm = Vm::new(interp, input, state, &mut self.execution, mode, out, err);
        vm.release_transient_memory();
        if !self.started {
            if !vm.state.sync_type_memory(&mut vm.interp.resources) {
                return VmPoll::Ready(ExecResult::Exit(137));
            }
            vm.bytecode_frames.push(BytecodeFrame {
                code: self.code.clone(),
                instruction_pointer: 0,
                handlers: Vec::new(),
                function_return: None,
                pending_native_call: None,
            });
            self.started = true;
        }
        if vm.state.heap.should_collect(
            vm.interp.resources.limits().memory,
            vm.interp.resources.memory_remaining(),
        ) {
            if let Err(error) = vm.collect_heap() {
                return VmPoll::Ready(ExecResult::Unsupported(error));
            }
        }
        let execution = vm.execute_active_frame(VM_POLL_QUANTUM);
        vm.release_transient_memory();
        if !vm.state.sync_type_memory(&mut vm.interp.resources) {
            return VmPoll::Ready(ExecResult::Exit(137));
        }
        match &execution {
            Ok(Execution::Pending) => return VmPoll::Runnable,
            Ok(Execution::Blocked(reason)) => return VmPoll::Blocked(*reason),
            _ => {}
        }
        vm.bytecode_frames
            .pop()
            .expect("completed program must retain its root frame");
        vm.release_retained_memory();
        VmPoll::Ready(vm.render_execution(execution))
    }
}

struct Vm<'a> {
    interp: &'a mut Interp,
    argv: &'a [String],
    stdin: &'a [u8],
    state: &'a mut ReplState,
    execution: &'a mut VmState,
    mode: VmMode,
    out: Out<'a>,
    err: Out<'a>,
}

#[derive(Clone, Copy)]
pub(super) struct ProcessInput<'a> {
    pub(super) argv: &'a [String],
    pub(super) stdin: &'a [u8],
}

/// State that must survive when bytecode execution yields to the process scheduler.
///
/// Keeping this state independent from the VM's temporary borrows is the first half of making a
/// Python process resumable. `Interp`, output descriptors, and the persistent Python heap are
/// borrowed only while a scheduler quantum is being polled; operand and semantic stacks belong to
/// the process continuation.
#[derive(Clone, Default)]
struct VmState {
    stack: Vec<Value>,
    bytecode_frames: Vec<BytecodeFrame>,
    local_scopes: Vec<ScopeId>,
    class_scopes: Vec<ScopeId>,
    class_bindings: Vec<Vec<String>>,
    call_depth: usize,
    pending_exception: Option<RaisedException>,
    pending_wait: Option<crate::scheduler::WaitReason>,
    native_suspend_allowed: bool,
    exception_stack: Vec<RaisedException>,
    with_contexts: Vec<Value>,
    method_frames: Vec<(super::heap::ObjectId, Value)>,
    stdin_position: usize,
    stdin_text: Option<String>,
    transient_memory: u64,
    retained_memory: u64,
}

/// An executing code object's resumable control state.
#[derive(Clone)]
struct BytecodeFrame {
    code: Code,
    instruction_pointer: usize,
    handlers: Vec<(usize, usize)>,
    function_return: Option<FunctionReturn>,
    pending_native_call: Option<PendingNativeCall>,
}

/// A normalized native invocation retained while its modeled resource is unavailable.
///
/// Starred arguments have already been expanded and the calling instruction has advanced, so a
/// wake retries exactly the native operation without repeating Python-visible argument work.
#[derive(Clone)]
struct PendingNativeCall {
    function: &'static FunctionDef,
    arguments: CallArgs,
    call_span: super::source::Span,
}

#[derive(Clone)]
struct FunctionReturn {
    name: String,
    call_span: super::source::Span,
    outer_stack: Vec<Value>,
    pop_method_frame: bool,
}

struct FunctionInvocation {
    arguments: Vec<Value>,
    keyword_arguments: Vec<(String, Value)>,
    mode: CallMode,
    pop_method_frame: bool,
}

impl Deref for Vm<'_> {
    type Target = VmState;

    fn deref(&self) -> &Self::Target {
        self.execution
    }
}

impl DerefMut for Vm<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.execution
    }
}

impl<'a> Vm<'a> {
    fn new(
        interp: &'a mut Interp,
        input: ProcessInput<'a>,
        state: &'a mut ReplState,
        execution: &'a mut VmState,
        mode: VmMode,
        out: Out<'a>,
        err: Out<'a>,
    ) -> Self {
        Self {
            interp,
            argv: input.argv,
            stdin: input.stdin,
            state,
            execution,
            mode,
            out,
            err,
        }
    }

    fn collect_heap(&mut self) -> Result<(), String> {
        let mut values = self.state.locals.values().copied().collect::<Vec<_>>();
        values.extend(self.state.modules.values().copied());
        values.extend(self.state.sys_path);
        values.extend(self.state.types.heap_roots());
        values.extend(self.stack.iter().copied());
        values.extend(self.with_contexts.iter().copied());
        if let Some(exception) = &self.pending_exception {
            values.push(exception.value);
        }
        values.extend(self.exception_stack.iter().map(|exception| exception.value));
        for (owner, value) in &self.method_frames {
            values.extend([Value::Object(*owner), *value]);
        }
        for frame in &self.bytecode_frames {
            if let Some(function_return) = &frame.function_return {
                values.extend(function_return.outer_stack.iter().copied());
            }
            if let Some(pending) = &frame.pending_native_call {
                values.extend(pending.arguments.positional().iter().copied());
                values.extend(pending.arguments.keywords().iter().map(|(_, value)| *value));
            }
        }
        let mut scopes = self.local_scopes.clone();
        scopes.extend(self.class_scopes.iter().copied());
        self.state
            .heap
            .collect(&values, &scopes, &mut self.interp.resources)?;
        Ok(())
    }

    fn release_transient_memory(&mut self) {
        let bytes = std::mem::take(&mut self.transient_memory);
        self.interp.resources.release_memory(bytes);
    }

    fn release_retained_memory(&mut self) {
        let bytes = std::mem::take(&mut self.retained_memory);
        self.interp.resources.release_memory(bytes);
    }

    fn reserve_retained_memory(&mut self, bytes: usize) -> Result<(), String> {
        let bytes = u64::try_from(bytes).map_err(|_| "Python allocation is too large")?;
        let next = self
            .retained_memory
            .checked_add(bytes)
            .ok_or("modeled Python memory overflow")?;
        if !self.interp.resources.reserve_memory(bytes) {
            return Err("memory limit exceeded".into());
        }
        self.retained_memory = next;
        Ok(())
    }

    fn render_execution(
        &mut self,
        execution: Result<Execution, (String, super::source::Span)>,
    ) -> ExecResult {
        match execution {
            Ok(Execution::Pending) => unreachable!("execute_code drains pending quanta"),
            Ok(Execution::Blocked(_)) => unreachable!("synchronous Python cannot suspend"),
            Ok(Execution::Halt) | Ok(Execution::Return(_)) | Ok(Execution::Yield(_, _)) => {
                ExecResult::Continue
            }
            Ok(Execution::Exit(status)) => ExecResult::Exit(status),
            Err((error, span)) => {
                if let Some(reason) = self.interp.resources.stop_reason() {
                    ExecResult::Exit(reason.exit_status())
                } else if self
                    .pending_exception
                    .as_ref()
                    .is_some_and(|exception| exception.kind == "SystemExit")
                {
                    let exception = self
                        .pending_exception
                        .as_ref()
                        .expect("SystemExit exception checked above");
                    let status = protocol::display(&self.state.heap, &exception.value)
                        .ok()
                        .and_then(|value| value.parse::<i32>().ok())
                        .unwrap_or(1);
                    ExecResult::Exit(status)
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
        instruction_pointer: usize,
        handlers: &mut Vec<(usize, usize)>,
    ) -> Result<Execution, (String, super::source::Span)> {
        self.bytecode_frames.push(BytecodeFrame {
            code: code.clone(),
            instruction_pointer,
            handlers: std::mem::take(handlers),
            function_return: None,
            pending_native_call: None,
        });
        let result = loop {
            match self.execute_active_frame(VM_POLL_QUANTUM) {
                Ok(Execution::Pending) => {}
                result => break result,
            }
        };
        let frame = self
            .bytecode_frames
            .pop()
            .expect("active bytecode frame must remain installed");
        *handlers = frame.handlers;
        result
    }

    fn execute_active_frame(
        &mut self,
        budget: usize,
    ) -> Result<Execution, (String, super::source::Span)> {
        'execution: for _ in 0..budget.max(1) {
            // Native helper snapshots live for one semantic instruction. Releasing the previous
            // instruction's scratch here avoids double-counting a materialized result after it
            // has moved into an arena object.
            self.release_transient_memory();
            if self.interp.deadline_interrupt.is_some() {
                return Ok(Execution::Exit(124));
            }
            if let Some(pending) = self.active_frame_mut().pending_native_call.take() {
                let span = pending.call_span;
                match self.resume_native_call(pending) {
                    Ok(CallResult::Value(value)) => {
                        self.stack.push(value);
                        continue;
                    }
                    Ok(CallResult::Blocked(reason, value)) => {
                        self.stack.push(value);
                        return Ok(Execution::Blocked(reason));
                    }
                    Ok(CallResult::Retry(reason, pending)) => {
                        self.active_frame_mut().pending_native_call = Some(pending);
                        return Ok(Execution::Blocked(reason));
                    }
                    Ok(CallResult::Exit(status)) => return Ok(Execution::Exit(status)),
                    Ok(CallResult::EnteredFrame) => {
                        unreachable!("a retained native call cannot enter a Python frame")
                    }
                    Err(error) => {
                        if self.propagate_error(error, span)? {
                            continue 'execution;
                        }
                        unreachable!("propagate_error either enters a handler or returns an error")
                    }
                }
            }
            let instruction_pointer = self
                .bytecode_frames
                .last()
                .expect("bytecode execution requires an active frame")
                .instruction_pointer;
            let Some(instruction) = self
                .bytecode_frames
                .last()
                .and_then(|frame| frame.code.instructions.get(instruction_pointer))
                .cloned()
            else {
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
                    self.value_from_constant(constant).map(|value| {
                        self.stack.push(value);
                    })
                }
                Operation::LoadName(name) => self.load_name(name),
                Operation::StoreName(name) => self.store_name(name),
                Operation::StoreEnclosing { name, scope_hops } => {
                    self.store_enclosing(name, *scope_hops)
                }
                Operation::StoreNonlocal(name) => self.store_nonlocal(name),
                Operation::StoreGlobal(name) => {
                    let value = self.pop().map_err(|error| (error, instruction.span))?;
                    self.store_global(name, value)
                }
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
                Operation::DeleteGlobal(name) => self.delete_global(name),
                Operation::DeleteSubscript => self.delete_subscript(),
                Operation::Import { name, bind_root } => self.import(name, *bind_root),
                Operation::LoadAttribute(name) => self.load_attribute(name),
                Operation::LoadSubscript => self.load_subscript(),
                Operation::LoadSlice {
                    has_start,
                    has_stop,
                    has_step,
                } => self.load_slice(*has_start, *has_stop, *has_step),
                Operation::BuildSlice {
                    has_start,
                    has_stop,
                    has_step,
                } => self.build_slice(*has_start, *has_stop, *has_step),
                Operation::BuildList(count) => self.build_sequence(*count, SequenceKind::List),
                Operation::BuildTuple(count) => self.build_sequence(*count, SequenceKind::Tuple),
                Operation::BuildDict(unpacked) => self.build_dict(unpacked),
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
                        self.active_frame_mut().instruction_pointer = *target;
                        continue;
                    }
                    Err(error) => Err(error),
                },
                Operation::Unary(operator) => self.unary(*operator),
                Operation::Binary(operator) => self.binary(*operator),
                Operation::FormatValue {
                    conversion,
                    format_spec,
                } => self.format_value(*conversion, format_spec),
                Operation::Compare(operator) => self.compare(*operator),
                Operation::Call {
                    positional,
                    keywords,
                    starred,
                } => match self.call(
                    *positional,
                    keywords,
                    starred,
                    CallMode::Deferred(instruction.span),
                ) {
                    Ok(CallResult::Value(value)) => {
                        self.stack.push(value);
                        Ok(())
                    }
                    Ok(CallResult::EnteredFrame) => {
                        let caller = self.bytecode_frames.len() - 2;
                        self.bytecode_frames[caller].instruction_pointer += 1;
                        continue;
                    }
                    Ok(CallResult::Blocked(reason, value)) => {
                        self.stack.push(value);
                        self.active_frame_mut().instruction_pointer += 1;
                        return Ok(Execution::Blocked(reason));
                    }
                    Ok(CallResult::Retry(reason, pending)) => {
                        self.active_frame_mut().instruction_pointer += 1;
                        self.active_frame_mut().pending_native_call = Some(pending);
                        return Ok(Execution::Blocked(reason));
                    }
                    Ok(CallResult::Exit(status)) => return Ok(Execution::Exit(status)),
                    Err(error) => Err(error),
                },
                Operation::Copy(depth) => self.copy(*depth),
                Operation::Swap(depth) => self.swap(*depth),
                Operation::PopTop => self.pop().map(|_| ()),
                Operation::Jump(target) => {
                    self.active_frame_mut().instruction_pointer = *target;
                    continue;
                }
                Operation::JumpIfFalseOrPop(target) => match self.jump_if_or_pop(false) {
                    Ok(true) => {
                        self.active_frame_mut().instruction_pointer = *target;
                        continue;
                    }
                    Ok(false) => Ok(()),
                    Err(error) => Err(error),
                },
                Operation::JumpIfTrueOrPop(target) => match self.jump_if_or_pop(true) {
                    Ok(true) => {
                        self.active_frame_mut().instruction_pointer = *target;
                        continue;
                    }
                    Ok(false) => Ok(()),
                    Err(error) => Err(error),
                },
                Operation::PopJumpIfFalse(target) => {
                    let value = self.pop().map_err(|error| (error, instruction.span))?;
                    if !self
                        .truth_value(&value)
                        .map_err(|error| (error, instruction.span))?
                    {
                        self.active_frame_mut().instruction_pointer = *target;
                        continue;
                    }
                    Ok(())
                }
                Operation::Return => {
                    let value = self.pop().map_err(|error| (error, instruction.span))?;
                    if self.finish_deferred_frame(value) {
                        continue;
                    }
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
                    if self
                        .truth_value(&condition)
                        .map_err(|error| (error, instruction.span))?
                    {
                        Ok(())
                    } else {
                        let message = if message.is_none() {
                            String::new()
                        } else {
                            protocol::display(&self.state.heap, &message)
                                .map_err(|error| (error, instruction.span))?
                        };
                        let value = self
                            .allocate_exception("AssertionError".into(), message)
                            .map_err(|error| (error, instruction.span))?;
                        self.pending_exception = Some(RaisedException {
                            kind: "AssertionError".into(),
                            value,
                        });
                        Err("assertion failed".into())
                    }
                }
                Operation::TryBegin(target) => {
                    let depth = self.stack.len();
                    self.active_frame_mut().handlers.push((*target, depth));
                    Ok(())
                }
                Operation::TryEnd => {
                    self.active_frame_mut()
                        .handlers
                        .pop()
                        .ok_or("invalid bytecode exception handler")
                        .map_err(|e| (e.to_string(), instruction.span))?;
                    Ok(())
                }
                Operation::MatchException { typed } => {
                    let actual = self
                        .exception_stack
                        .last()
                        .ok_or("no active exception")
                        .map_err(|e| (e.to_string(), instruction.span))?
                        .kind
                        .clone();
                    let matches = if *typed {
                        let expected = self.pop().map_err(|error| (error, instruction.span))?;
                        self.exception_type_matches(expected, &actual)
                            .map_err(|error| (error, instruction.span))?
                    } else {
                        true
                    };
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
                        if let Some((kind, _)) = protocol::exception_parts(&self.state.heap, &value)
                            .map_err(|error| (error, instruction.span))?
                        {
                            RaisedException { kind, value }
                        } else if let Some(NativeValue::ExceptionType(ExceptionType(kind))) =
                            value.native_value()
                        {
                            let value = self
                                .allocate_exception(kind.to_string(), String::new())
                                .map_err(|error| (error, instruction.span))?;
                            RaisedException {
                                kind: kind.to_string(),
                                value,
                            }
                        } else {
                            return Err((
                                "exceptions must derive from BaseException".into(),
                                instruction.span,
                            ));
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
                    self.stack.push(context);
                    self.with_contexts.push(context);
                    self.load_attribute("__enter__")
                        .map_err(|e| (e, instruction.span))?;
                    match self
                        .call(0, &[], &[], CallMode::Immediate)
                        .map_err(|e| (e, instruction.span))?
                    {
                        CallResult::Value(value) => self.stack.push(value),
                        CallResult::Exit(status) => return Ok(Execution::Exit(status)),
                        CallResult::EnteredFrame => unreachable!("immediate call entered a frame"),
                        CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                            unreachable!("immediate call cannot suspend")
                        }
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
                        .call(3, &[], &[false, false, false], CallMode::Immediate)
                        .map_err(|e| (e, instruction.span))?
                    {
                        CallResult::Value(_) => Ok(()),
                        CallResult::Exit(status) => return Ok(Execution::Exit(status)),
                        CallResult::EnteredFrame => unreachable!("immediate call entered a frame"),
                        CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                            unreachable!("immediate call cannot suspend")
                        }
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
                    let exception_kind = self
                        .allocate_string(exception.kind.clone())
                        .map_err(|error| (error, instruction.span))?;
                    self.stack
                        .extend([exception_kind, exception.value, Value::None]);
                    let result = match self
                        .call(3, &[], &[false, false, false], CallMode::Immediate)
                        .map_err(|e| (e, instruction.span))?
                    {
                        CallResult::Value(value) => value,
                        CallResult::Exit(status) => return Ok(Execution::Exit(status)),
                        CallResult::EnteredFrame => unreachable!("immediate call entered a frame"),
                        CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                            unreachable!("immediate call cannot suspend")
                        }
                    };
                    if self
                        .truth_value(&result)
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
                        if self.mode.interactive && !matches!(value, Value::None) {
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
                Operation::Halt => {
                    if self.finish_deferred_frame(Value::None) {
                        continue;
                    }
                    return Ok(Execution::Halt);
                }
            };
            if let Err(error) = result {
                if self.propagate_error(error, instruction.span)? {
                    continue 'execution;
                }
                unreachable!("propagate_error either enters a handler or returns an error")
            }
            self.active_frame_mut().instruction_pointer += 1;
        }
        Ok(Execution::Pending)
    }

    fn active_frame_mut(&mut self) -> &mut BytecodeFrame {
        self.bytecode_frames
            .last_mut()
            .expect("bytecode execution requires an active frame")
    }

    fn propagate_error(
        &mut self,
        mut error: String,
        mut span: super::source::Span,
    ) -> Result<bool, (String, super::source::Span)> {
        loop {
            if self.enter_exception_handler() {
                return Ok(true);
            }
            let Some(function_return) = self.unwind_deferred_frame() else {
                return Err((error, span));
            };
            error = format!(
                "{error} in {} at line {}, column {}",
                function_return.name, span.line, span.column
            );
            span = function_return.call_span;
        }
    }

    fn finish_deferred_frame(&mut self, value: Value) -> bool {
        let Some(_) = self
            .bytecode_frames
            .last()
            .and_then(|frame| frame.function_return.as_ref())
        else {
            return false;
        };
        self.unwind_deferred_frame()
            .expect("deferred function frame was checked above");
        self.stack.push(value);
        true
    }

    fn unwind_deferred_frame(&mut self) -> Option<FunctionReturn> {
        self.bytecode_frames
            .last()
            .and_then(|frame| frame.function_return.as_ref())?;
        let frame = self
            .bytecode_frames
            .pop()
            .expect("deferred function frame was checked above");
        let mut function_return = frame
            .function_return
            .expect("deferred function frame must own return state");
        self.call_depth = self.call_depth.saturating_sub(1);
        self.local_scopes
            .pop()
            .expect("deferred function frame must own a local scope");
        if function_return.pop_method_frame {
            self.method_frames
                .pop()
                .expect("deferred method frame must remain installed");
        }
        self.stack = std::mem::take(&mut function_return.outer_stack);
        Some(function_return)
    }

    fn enter_exception_handler(&mut self) -> bool {
        let Some(exception) = self.pending_exception.take() else {
            return false;
        };
        let Some((target, depth)) = self.active_frame_mut().handlers.pop() else {
            self.pending_exception = Some(exception);
            return false;
        };
        self.stack.truncate(depth);
        self.stack.push(exception.value);
        self.exception_stack.push(exception);
        self.active_frame_mut().instruction_pointer = target;
        true
    }

    fn load_name(&mut self, name: &str) -> Result<(), String> {
        if name == "__class__" {
            let class = self
                .method_frames
                .last()
                .map(|(class, _)| *class)
                .ok_or("__class__ is only defined inside a class method body")?;
            self.stack.push(Value::Object(class));
            return Ok(());
        }
        let local_scope = self.local_scopes.last().copied();
        let scoped = local_scope.and_then(|scope| self.state.heap.scope_get(scope, name).cloned());
        let can_use_globals = local_scope
            .map(|scope| self.state.heap.scope_uses_repl_globals(scope))
            .transpose()?
            .unwrap_or(true);
        let value = scoped.or_else(|| {
            can_use_globals
                .then(|| self.state.locals.get(name).cloned())
                .flatten()
        });
        if let Some(value) = value {
            self.stack.push(value);
            return Ok(());
        }
        if let Some((module_name, attribute)) = super::stdlib::frozen_builtin(name) {
            self.import(module_name, false)?;
            let module = self.pop()?;
            let callable = self.resolve_attribute(module, attribute)?.ok_or_else(|| {
                format!("frozen module {module_name:?} does not define {attribute:?}")
            })?;
            self.stack.push(callable);
            return Ok(());
        }
        let value = (|| {
            let builtin_type = match name {
                "type" => Some(BuiltinType::Type),
                "object" => Some(BuiltinType::Object),
                "bool" => Some(BuiltinType::Bool),
                "int" => Some(BuiltinType::Int),
                "float" => Some(BuiltinType::Float),
                "str" => Some(BuiltinType::String),
                "bytes" => Some(BuiltinType::Bytes),
                "bytearray" => Some(BuiltinType::ByteArray),
                "list" => Some(BuiltinType::List),
                "tuple" => Some(BuiltinType::Tuple),
                "dict" => Some(BuiltinType::Dict),
                "set" => Some(BuiltinType::Set),
                _ => None,
            };
            if let Some(builtin_type) = builtin_type {
                return Some(Value::Native(NativeValue::BuiltinType(builtin_type)));
            }
            if let Some(function) = super::stdlib::core::builtin_function(name) {
                return Some(Value::Native(NativeValue::NativeFunction(function)));
            }
            let builtin = match name {
                "print" => Builtin::Print,
                "input" => Builtin::Input,
                "exit" | "quit" => Builtin::Exit,
                "repr" => Builtin::Repr,
                "isinstance" => Builtin::IsInstance,
                "issubclass" => Builtin::IsSubclass,
                "len" => Builtin::Length,
                "sorted" => Builtin::Sorted,
                "min" => Builtin::Minimum,
                "max" => Builtin::Maximum,
                "sum" => Builtin::Sum,
                "abs" => Builtin::Absolute,
                "range" => Builtin::Range,
                "enumerate" => Builtin::Enumerate,
                "zip" => Builtin::Zip,
                "any" => Builtin::Any,
                "all" => Builtin::All,
                "iter" => Builtin::Iter,
                "next" => Builtin::Next,
                "property" => Builtin::Property,
                "staticmethod" => Builtin::StaticMethod,
                "classmethod" => Builtin::ClassMethod,
                "super" => Builtin::Super,
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
                "EOFError" => {
                    return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                        "EOFError",
                    ))))
                }
                "OSError" => {
                    return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                        "OSError",
                    ))))
                }
                "FileNotFoundError" => {
                    return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                        "FileNotFoundError",
                    ))))
                }
                "FileExistsError" => {
                    return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                        "FileExistsError",
                    ))))
                }
                "IsADirectoryError" => {
                    return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                        "IsADirectoryError",
                    ))))
                }
                "NotADirectoryError" => {
                    return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                        "NotADirectoryError",
                    ))))
                }
                "PermissionError" => {
                    return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                        "PermissionError",
                    ))))
                }
                "SystemExit" => {
                    return Some(Value::Native(NativeValue::ExceptionType(ExceptionType(
                        "SystemExit",
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
        })();
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

    fn store_enclosing(&mut self, name: &str, scope_hops: usize) -> Result<(), String> {
        let value = self.pop()?;
        let mut target = self.local_scopes.last().copied();
        for _ in 0..scope_hops {
            target = target
                .map(|scope| self.state.heap.scope_parent(scope))
                .transpose()?
                .flatten();
        }
        if let Some(scope) = target {
            self.state
                .heap
                .scope_insert(scope, name.to_string(), value, &mut self.interp.resources)
        } else {
            self.state.locals.insert(name.to_string(), value);
            Ok(())
        }
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

    fn exception_type_matches(&mut self, expected: Value, actual: &str) -> Result<bool, String> {
        if let Some(NativeValue::ExceptionType(ExceptionType(name))) = expected.native_value() {
            let os_error = matches!(
                actual,
                "OSError"
                    | "FileNotFoundError"
                    | "FileExistsError"
                    | "IsADirectoryError"
                    | "NotADirectoryError"
                    | "PermissionError"
            );
            return Ok(name == "BaseException"
                || (name == "Exception" && actual != "SystemExit")
                || name == actual
                || (name == "OSError" && os_error));
        }
        let Some(id) = expected.object_id() else {
            return Err(
                "catching classes that do not inherit from BaseException is not allowed".into(),
            );
        };
        let Object::Tuple(types) = self.state.heap.get(id)? else {
            return Err(
                "catching classes that do not inherit from BaseException is not allowed".into(),
            );
        };
        let types = types.clone();
        for expected in types {
            self.charge_cpu(1)?;
            if self.exception_type_matches(expected, actual)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn store_global(&mut self, name: &str, value: Value) -> Result<(), String> {
        let Some(scope) = self.local_scopes.last().copied() else {
            self.state.locals.insert(name.to_string(), value);
            return Ok(());
        };
        let root = self.state.heap.scope_root(scope)?;
        if self.state.heap.scope_uses_repl_globals(root)? {
            self.state.locals.insert(name.to_string(), value);
            Ok(())
        } else {
            self.state
                .heap
                .scope_insert(root, name.to_string(), value, &mut self.interp.resources)
        }
    }

    fn delete_global(&mut self, name: &str) -> Result<(), String> {
        let Some(scope) = self.local_scopes.last().copied() else {
            self.state.locals.remove(name);
            return Ok(());
        };
        let root = self.state.heap.scope_root(scope)?;
        if self.state.heap.scope_uses_repl_globals(root)? {
            self.state.locals.remove(name);
        } else {
            self.state.heap.scope_remove(root, name)?;
        }
        Ok(())
    }

    fn import_roots(&mut self) -> Result<Vec<String>, String> {
        let mut roots = self.state.temporary_import_paths.clone();
        if let Some(path) = self.state.sys_path {
            let Object::List(values) = self
                .state
                .heap
                .get(path.object_id().ok_or("sys.path lost list identity")?)?
            else {
                return Err("sys.path must remain a list".into());
            };
            for value in values.clone() {
                let path = self
                    .string_value(&value)
                    .map_err(|error| self.record_native_error(error))?
                    .ok_or("sys.path entries must be strings")?;
                roots.push(path);
            }
        } else {
            roots.extend(self.state.import_paths.clone());
        }
        if roots.is_empty() {
            roots.push(self.interp.cwd.clone());
        }
        Ok(roots)
    }

    fn import(&mut self, name: &str, bind_root: bool) -> Result<(), String> {
        if let Some(module) = super::stdlib::native_module(name) {
            return self.finish_import(name, Value::Native(NativeValue::Module(module)), bind_root);
        }
        if let Some(module) = self.state.modules.get(name).cloned() {
            return self.finish_import(name, module, bind_root);
        }

        let relative_name = name.trim_start_matches('.');
        let source = if let Some(source) = super::stdlib::frozen_module(relative_name) {
            Some((format!("<frozen {name}>"), source.to_string()))
        } else {
            let roots = self.import_roots()?;
            self.interp
                .load_module_source(&roots, relative_name)
                .map_err(|error| self.record_native_error(error))?
        };
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
        let module_name = self.allocate_string(relative_name.to_string())?;
        let scope = self.state.heap.allocate_scope(
            None,
            false,
            HashMap::from([("__name__".into(), module_name)]),
            &mut self.interp.resources,
        )?;
        let module = self.allocate_object(Object::Module {
            name: name.to_string(),
            scope,
        })?;
        self.state.modules.insert(name.to_string(), module);

        let outer_stack = std::mem::take(&mut self.stack);
        let temporary_import_path = (!path.starts_with('<')).then(|| {
            path.rsplit_once('/')
                .map_or_else(|| "/".to_string(), |(parent, _)| parent.to_string())
        });
        if let Some(path) = &temporary_import_path {
            self.state.temporary_import_paths.insert(0, path.clone());
        }
        self.local_scopes.push(scope);
        let execution = self.execute_code(&code);
        self.local_scopes.pop();
        if temporary_import_path.is_some() {
            self.state.temporary_import_paths.remove(0);
        }
        self.stack = outer_stack;
        match execution {
            Ok(Execution::Pending) => unreachable!("execute_code drains pending quanta"),
            Ok(Execution::Blocked(_)) => unreachable!("immediate code cannot suspend"),
            Ok(Execution::Halt) => self.finish_import(name, module, bind_root),
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

    /// Install synthetic package parents for a dotted import and push the value selected by
    /// Python's ordinary import binding rule. Package objects contain only VM module references.
    fn finish_import(&mut self, name: &str, leaf: Value, bind_root: bool) -> Result<(), String> {
        if !bind_root || !name.contains('.') {
            self.stack.push(leaf);
            return Ok(());
        }
        let parts = name.split('.').collect::<Vec<_>>();
        let mut child = leaf;
        for parent_end in (1..parts.len()).rev() {
            let parent_name = parts[..parent_end].join(".");
            let child_name = parts[parent_end];
            let parent = if let Some(parent) = self.state.modules.get(&parent_name).copied() {
                let Some(parent_id) = parent.object_id() else {
                    return Err(format!("module {parent_name:?} cannot contain submodules"));
                };
                let Object::Module { scope, .. } = self.state.heap.get(parent_id)? else {
                    return Err(format!("module {parent_name:?} changed object kind"));
                };
                let scope = *scope;
                self.state.heap.scope_insert(
                    scope,
                    child_name.to_string(),
                    child,
                    &mut self.interp.resources,
                )?;
                parent
            } else {
                let module_name = self.allocate_string(parent_name.clone())?;
                let scope = self.state.heap.allocate_scope(
                    None,
                    false,
                    HashMap::from([
                        ("__name__".into(), module_name),
                        (child_name.to_string(), child),
                    ]),
                    &mut self.interp.resources,
                )?;
                let parent = self.allocate_object(Object::Module {
                    name: parent_name.clone(),
                    scope,
                })?;
                self.state.modules.insert(parent_name, parent);
                parent
            };
            child = parent;
        }
        self.stack.push(child);
        Ok(())
    }

    fn load_attribute(&mut self, name: &str) -> Result<(), String> {
        let owner = self.pop()?;
        let value = self
            .resolve_attribute(owner, name)?
            .ok_or_else(|| format!("attribute {name:?} is not implemented"))?;
        self.stack.push(value);
        Ok(())
    }

    /// Resolve one attribute without involving the operand stack.
    ///
    /// Missing attributes return `None`; errors raised while invoking descriptors remain errors.
    /// This distinction lets `getattr` and `hasattr` share the bytecode lookup path.
    fn resolve_attribute(&mut self, owner: Value, name: &str) -> Result<Option<Value>, String> {
        if let Some(NativeValue::Module(module)) = owner.native_value() {
            if let Some(function) = module.function(name) {
                return Ok(Some(Value::Native(NativeValue::NativeFunction(function))));
            }
            if let Some(value) = module.value(name) {
                let value = value.get(self).map_err(|error| error.to_string())?;
                return Ok(Some(value));
            }
        }
        let owner_type = self.type_id(&owner)?;
        if let Some(method) = self
            .state
            .types
            .attribute(owner_type, name)?
            .and_then(|value| match value.native_value() {
                Some(NativeValue::NativeMethod(method)) => Some(method),
                _ => None,
            })
        {
            if method.name == "__new__" {
                return Ok(Some(Value::Native(NativeValue::NativeMethod(method))));
            }
            let bound = self.allocate_object(Object::DescriptorBoundMethod {
                receiver: owner,
                descriptor: Value::Native(NativeValue::NativeMethod(method)),
                owner: None,
            })?;
            return Ok(Some(bound));
        }
        if let Some(id) = owner.object_id() {
            match self.state.heap.get(id)?.clone() {
                Object::Module { scope, .. } => {
                    return Ok(self.state.heap.scope_get(scope, name).copied());
                }
                Object::Class { .. } => {
                    let mut entry = self.class_attribute_entry(id, name)?;
                    if entry.is_none() {
                        let Object::Class { metaclass, .. } = self.state.heap.get(id)? else {
                            unreachable!()
                        };
                        let metaclass = *metaclass;
                        if let Some(metaclass) = metaclass.object_id() {
                            entry = self.class_attribute_entry(metaclass, name)?;
                        }
                    }
                    let Some((defining_class, descriptor)) = entry else {
                        return Ok(None);
                    };
                    let value = self.bind_descriptor(descriptor, None, id, defining_class)?;
                    return Ok(Some(value));
                }
                Object::EnumMember {
                    name: member_name,
                    value,
                } => {
                    let value = match name {
                        "name" => self.allocate_string(member_name)?,
                        "value" => value,
                        _ => return Ok(None),
                    };
                    return Ok(Some(value));
                }
                Object::Instance { class, .. } => {
                    let class_entry = self.class_attribute_entry(class, name)?;
                    if let Some((defining_class, descriptor)) = class_entry {
                        if self.is_data_descriptor(&descriptor)? {
                            let value = self.bind_descriptor(
                                descriptor,
                                Some(owner),
                                class,
                                defining_class,
                            )?;
                            return Ok(Some(value));
                        }
                    }
                    let instance = owner
                        .cast::<PyInstance>(self)
                        .map_err(|error| error.to_string())?;
                    if let Some(value) = instance
                        .attribute(self, name)
                        .map_err(|error| error.to_string())?
                    {
                        return Ok(Some(value));
                    }
                    let Some((defining_class, descriptor)) = class_entry else {
                        return Ok(None);
                    };
                    let value =
                        self.bind_descriptor(descriptor, Some(owner), class, defining_class)?;
                    return Ok(Some(value));
                }
                Object::Super {
                    start_class,
                    receiver,
                } => {
                    let (defining_class, descriptor, accessed_class) =
                        self.super_attribute(start_class, &receiver, name)?;
                    let value = self.bind_descriptor(
                        descriptor,
                        Some(receiver),
                        accessed_class,
                        defining_class,
                    )?;
                    return Ok(Some(value));
                }
                Object::Match { .. } => {}
                Object::ArgumentParser { prog, .. } => {
                    if name == "prog" {
                        let prog = self.allocate_string(prog)?;
                        return Ok(Some(prog));
                    }
                }
                Object::Namespace { values } => {
                    let value = values
                        .iter()
                        .find(|(key, _)| key == name)
                        .map(|(_, value)| *value);
                    return Ok(value);
                }
                Object::Array { layout, dtype, .. } => {
                    let value = match name {
                        "shape" => Some(
                            self.allocate_object(Object::Tuple(
                                layout
                                    .shape
                                    .iter()
                                    .map(|value| Value::Int(*value as i64))
                                    .collect(),
                            ))?,
                        ),
                        "ndim" => Some(Value::Int(layout.shape.len() as i64)),
                        "size" => Some(Value::Int(
                            layout
                                .shape
                                .iter()
                                .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
                                .ok_or("array size overflow")? as i64,
                        )),
                        "dtype" => Some(self.allocate_string(dtype.name().to_string())?),
                        "T" => {
                            let array = owner
                                .cast::<PyArray>(self)
                                .map_err(|error| error.to_string())?;
                            Some(
                                super::stdlib::numpy::transpose(self, array, None)
                                    .map_err(|error| error.to_string())?,
                            )
                        }
                        _ => None,
                    };
                    return Ok(value);
                }
                _ => {}
            }
        }
        let value = if matches!(owner.native_value(), Some(NativeValue::UnitTestBase))
            && name == "__name__"
        {
            Some(self.allocate_string("TestCase".into())?)
        } else {
            None
        };
        Ok(value)
    }

    fn store_attribute(&mut self, owner: Value, name: &str, value: Value) -> Result<(), String> {
        let Some(id) = owner.object_id() else {
            return Err("object does not support attribute assignment".into());
        };
        let Object::Instance { class, .. } = self.state.heap.get(id)? else {
            return Err("object does not support attribute assignment".into());
        };
        let class = *class;
        if let Some((_, descriptor)) = self.class_attribute_entry(class, name)? {
            if let Some(descriptor_id) = descriptor.object_id() {
                match self.state.heap.get(descriptor_id)?.clone() {
                    Object::Property {
                        setter: Some(setter),
                        ..
                    } => {
                        self.invoke_value(setter, vec![Value::Object(id), value])?;
                        return Ok(());
                    }
                    Object::Property { setter: None, .. } => {
                        return Err(format!("property {name:?} has no setter"));
                    }
                    Object::Instance {
                        class: descriptor_class,
                        ..
                    } => {
                        if let Some((set_owner, set)) =
                            self.class_attribute_entry(descriptor_class, "__set__")?
                        {
                            let set = self.bind_descriptor(
                                set,
                                Some(descriptor),
                                descriptor_class,
                                set_owner,
                            )?;
                            self.invoke_value(set, vec![Value::Object(id), value])?;
                            return Ok(());
                        }
                    }
                    _ => {}
                }
            }
        }
        let is_new = !self.state.heap.has_attribute(id, name)?;
        if is_new {
            self.state
                .heap
                .reserve_object_growth(id, 48, &mut self.interp.resources)?;
        }
        self.state
            .heap
            .insert_attribute(id, name.to_string(), value)?;
        Ok(())
    }

    fn load_subscript(&mut self) -> Result<(), String> {
        let index = self.pop()?;
        let owner = self.pop()?;
        if let Some(value) = self.invoke_slot(&owner, Slot::GetItem, "__getitem__", vec![index])? {
            self.stack.push(value);
            return Ok(());
        }
        let value = if matches!(owner.native_value(), Some(NativeValue::TypingList)) {
            let parameter = match index.native_value() {
                Some(NativeValue::BuiltinType(BuiltinType::Int)) => "int".to_string(),
                Some(NativeValue::BuiltinType(BuiltinType::String)) => "str".to_string(),
                _ => protocol::repr(&self.state.heap, &index)?,
            };
            self.allocate_string(format!("typing.List[{parameter}]"))?
        } else if let Some(character) = protocol::string_index(&self.state.heap, &owner, &index)? {
            self.allocate_string(character.to_string())?
        } else if let Some(id) = owner.object_id() {
            match self.state.heap.get(id)?.clone() {
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
                        if protocol::identical(key, &index)
                            || protocol::equals(&self.state.heap, key, &index)?
                        {
                            found = Some(*value);
                            break;
                        }
                    }
                    found.ok_or("key not found")?
                }
                Object::DefaultDict { factory, entries } => {
                    let mut found = None;
                    for (key, value) in &entries {
                        self.charge_cpu(1)?;
                        if protocol::identical(key, &index)
                            || protocol::equals(&self.state.heap, key, &index)?
                        {
                            found = Some(*value);
                            break;
                        }
                    }
                    if let Some(value) = found {
                        value
                    } else {
                        self.stack.push(factory);
                        let value = match self.call(0, &[], &[], CallMode::Immediate)? {
                            CallResult::Value(value) => value,
                            CallResult::Exit(status) => {
                                return Err(format!("default factory exited with status {status}"))
                            }
                            CallResult::EnteredFrame => {
                                unreachable!("immediate call entered a frame")
                            }
                            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                                unreachable!("immediate call cannot suspend")
                            }
                        };
                        self.state.heap.reserve_object_growth(
                            id,
                            48,
                            &mut self.interp.resources,
                        )?;
                        let Object::DefaultDict { entries, .. } = self.state.heap.get_mut(id)?
                        else {
                            unreachable!()
                        };
                        entries.push((index, value));
                        value
                    }
                }
                Object::Set(_) => return Err("set object is not subscriptable".into()),
                Object::String(_)
                | Object::Bytes(_)
                | Object::ByteArray(_)
                | Object::Slice { .. }
                | Object::Exception { .. }
                | Object::BigInt(_)
                | Object::Function { .. }
                | Object::Class { .. }
                | Object::Instance { .. }
                | Object::DescriptorBoundMethod { .. }
                | Object::Iterator { .. }
                | Object::CountIterator { .. }
                | Object::Generator { .. }
                | Object::Module { .. }
                | Object::ArrayStorage(_)
                | Object::Array { .. }
                | Object::Regex { .. }
                | Object::Match { .. }
                | Object::ArgumentParser { .. }
                | Object::Namespace { .. } => return Err("object is not subscriptable".into()),
                Object::EnumMember { .. } => return Err("object is not subscriptable".into()),
                Object::RaisesContext { .. } => return Err("object is not subscriptable".into()),
                Object::Property { .. }
                | Object::StaticMethod { .. }
                | Object::ClassMethod { .. }
                | Object::Super { .. } => return Err("object is not subscriptable".into()),
            }
        } else {
            return Err("object is not subscriptable".into());
        };
        self.stack.push(value);
        Ok(())
    }

    fn load_slice(
        &mut self,
        has_start: bool,
        has_stop: bool,
        has_step: bool,
    ) -> Result<(), String> {
        let step = if has_step {
            Some(
                self.pop()?
                    .as_int()
                    .ok_or("slice step must be an integer")?,
            )
        } else {
            None
        };
        let stop = if has_stop {
            Some(
                self.pop()?
                    .as_int()
                    .ok_or("slice stop must be an integer")?,
            )
        } else {
            None
        };
        let start = if has_start {
            Some(
                self.pop()?
                    .as_int()
                    .ok_or("slice start must be an integer")?,
            )
        } else {
            None
        };
        let owner = self.pop()?;
        if let Some(SlotValue::NativeSlice(call)) =
            self.state.types.slot(self.type_id(&owner)?, Slot::Slice)?
        {
            let value = call(self, owner, start, stop, step)
                .map_err(|error| self.record_native_error(error))?
                .ok_or("slice operation declined by its registered type")?;
            self.stack.push(value);
            return Ok(());
        }
        let string_slice = protocol::string_ref(&self.state.heap, &owner)?
            .map(|text| select_string_slice(text.as_str(), text.is_ascii(), start, stop, step))
            .transpose()?;
        let value = if let Some((selected, units)) = string_slice {
            self.charge_cpu(units)?;
            self.allocate_string(selected)?
        } else if let Some(bytes) = protocol::bytes_value(&self.state.heap, &owner)? {
            let indices = slice_indices(bytes.len(), start, stop, step)?;
            self.charge_cpu(u64::try_from(indices.len()).unwrap_or(u64::MAX))?;
            let selected = indices.into_iter().map(|index| bytes[index]).collect();
            if self.type_id(&owner)? == BuiltinType::ByteArray.id() {
                self.allocate_bytearray(selected)?
            } else {
                self.allocate_bytes(selected)?
            }
        } else if let Some(id) = owner.object_id() {
            let (values, tuple) = match self.state.heap.get(id)?.clone() {
                Object::List(values) => (values, false),
                Object::Tuple(values) => (values, true),
                _ => return Err("object is not sliceable".into()),
            };
            let indices = slice_indices(values.len(), start, stop, step)?;
            self.charge_cpu(u64::try_from(indices.len()).unwrap_or(u64::MAX))?;
            let selected = indices.into_iter().map(|index| values[index]).collect();
            self.allocate_object(if tuple {
                Object::Tuple(selected)
            } else {
                Object::List(selected)
            })?
        } else {
            return Err("object is not sliceable".into());
        };
        self.stack.push(value);
        Ok(())
    }

    fn build_slice(
        &mut self,
        has_start: bool,
        has_stop: bool,
        has_step: bool,
    ) -> Result<(), String> {
        let step = if has_step {
            Some(
                self.pop()?
                    .as_int()
                    .ok_or("slice step must be an integer")?,
            )
        } else {
            None
        };
        let stop = if has_stop {
            Some(
                self.pop()?
                    .as_int()
                    .ok_or("slice stop must be an integer")?,
            )
        } else {
            None
        };
        let start = if has_start {
            Some(
                self.pop()?
                    .as_int()
                    .ok_or("slice start must be an integer")?,
            )
        } else {
            None
        };
        let value = self.allocate_object(Object::Slice { start, stop, step })?;
        self.stack.push(value);
        Ok(())
    }

    fn store_subscript(&mut self) -> Result<(), String> {
        let index = self.pop()?;
        let owner = self.pop()?;
        let value = self.pop()?;
        if self
            .invoke_slot(&owner, Slot::SetItem, "__setitem__", vec![index, value])?
            .is_some()
        {
            return Ok(());
        }
        let Some(id) = owner.object_id() else {
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
                    if protocol::identical(candidate, &index)
                        || protocol::equals(&self.state.heap, candidate, &index)?
                    {
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
                        .reserve_object_growth(id, 48, &mut self.interp.resources)?;
                    let entries = match self.state.heap.get_mut(id)? {
                        Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
                        _ => unreachable!(),
                    };
                    entries.push((index, value));
                }
            }
            Object::Tuple(_) => return Err("tuple object does not support item assignment".into()),
            Object::String(_)
            | Object::Bytes(_)
            | Object::ByteArray(_)
            | Object::Slice { .. }
            | Object::Exception { .. }
            | Object::Set(_)
            | Object::BigInt(_)
            | Object::Function { .. }
            | Object::Class { .. }
            | Object::Instance { .. }
            | Object::DescriptorBoundMethod { .. }
            | Object::Iterator { .. }
            | Object::CountIterator { .. }
            | Object::Generator { .. }
            | Object::Module { .. }
            | Object::ArrayStorage(_)
            | Object::Array { .. }
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
            Object::Property { .. }
            | Object::StaticMethod { .. }
            | Object::ClassMethod { .. }
            | Object::Super { .. } => return Err("object does not support item assignment".into()),
        }
        Ok(())
    }

    fn delete_subscript(&mut self) -> Result<(), String> {
        let index = self.pop()?;
        let owner = self.pop()?;
        self.invoke_slot(&owner, Slot::DeleteItem, "__delitem__", vec![index])?
            .ok_or("object does not support item deletion")?;
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
        let defaults_start = self.stack.len() - default_count;
        let defaults = self.stack.split_off(defaults_start);
        let mut closure = self.local_scopes.last().copied();
        if closure.is_some() && closure == self.class_scopes.last().copied() {
            closure = self
                .state
                .heap
                .scope_parent(closure.expect("checked above"))?;
        }
        let function = self.state.heap.allocate(
            Object::Function {
                name: name.clone(),
                code,
                closure,
                defaults,
                defining_class: None,
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
        let bases_start = self.stack.len() - base_count;
        let bases = self.stack.split_off(bases_start);
        let is_enum =
            bases.len() == 1 && matches!(bases[0].native_value(), Some(NativeValue::EnumBase));
        let is_unittest =
            bases.len() == 1 && matches!(bases[0].native_value(), Some(NativeValue::UnitTestBase));
        let has_int_base = bases.iter().any(|base| {
            matches!(
                base.native_value(),
                Some(NativeValue::BuiltinType(BuiltinType::Int))
            )
        });
        let has_object_base = bases.iter().any(|base| {
            matches!(
                base.native_value(),
                Some(NativeValue::BuiltinType(BuiltinType::Object))
            )
        });
        let has_type_base = bases.iter().any(|base| {
            matches!(
                base.native_value(),
                Some(NativeValue::BuiltinType(BuiltinType::Type))
            )
        });
        let user_bases = if is_enum || is_unittest {
            Vec::new()
        } else {
            bases
                .iter()
                .filter_map(|base| {
                    if let Some(id) = base.object_id() {
                        if matches!(self.state.heap.get(id), Ok(Object::Class { .. })) {
                            Some(Ok(id))
                        } else {
                            Some(Err("class bases must be classes".to_string()))
                        }
                    } else if matches!(
                        base.native_value(),
                        Some(NativeValue::BuiltinType(
                            BuiltinType::Int | BuiltinType::Object | BuiltinType::Type
                        ))
                    ) {
                        None
                    } else {
                        Some(Err("class bases must be classes".to_string()))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        if has_int_base && (bases.len() != 1 || is_enum || is_unittest) {
            return Err("int inheritance with another direct base is unsupported".into());
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
        let has_explicit_metaclass = explicit_metaclass.is_some();
        let mut metaclass = explicit_metaclass
            .unwrap_or(Value::Native(NativeValue::BuiltinType(BuiltinType::Type)));
        for base in &user_bases {
            let Object::Class {
                metaclass: base_metaclass,
                ..
            } = self.state.heap.get(*base)?
            else {
                unreachable!()
            };
            let base_metaclass = *base_metaclass;
            let winner_type = self
                .class_type_id(&metaclass)?
                .ok_or("metaclass must be a type")?;
            let candidate_type = self
                .class_type_id(&base_metaclass)?
                .ok_or("base class has an invalid metaclass")?;
            if self.state.types.is_subclass(winner_type, candidate_type)? {
                continue;
            }
            if !has_explicit_metaclass
                && self.state.types.is_subclass(candidate_type, winner_type)?
            {
                metaclass = base_metaclass;
                continue;
            }
            return Err("metaclass conflict between bases or explicit metaclass".into());
        }
        let valid_metaclass = matches!(
            metaclass.native_value(),
            Some(NativeValue::BuiltinType(BuiltinType::Type))
        ) || metaclass.object_id().is_some_and(|id| {
            matches!(
                self.state.heap.get(id),
                Ok(Object::Class {
                    layout: ClassLayout::Type,
                    ..
                })
            )
        });
        if !valid_metaclass {
            return Err("metaclass must derive from type".into());
        }
        let mro = self.linearize_bases(&user_bases)?;
        let mut prepared_namespace = HashMap::new();
        if let Some(metaclass_id) = metaclass.object_id() {
            if let Some((owner, prepare)) =
                self.class_attribute_entry(metaclass_id, "__prepare__")?
            {
                let prepare = self.bind_descriptor(prepare, None, metaclass_id, owner)?;
                let bases_value = self.allocate_object(Object::Tuple(bases.clone()))?;
                let class_name = self.allocate_string(name.clone())?;
                let namespace = self.invoke_value(prepare, vec![class_name, bases_value])?;
                let Some(namespace_id) = namespace.object_id() else {
                    return Err("metaclass __prepare__() must return a mapping".into());
                };
                let Object::Dict(entries) = self.state.heap.get(namespace_id)?.clone() else {
                    return Err("metaclass __prepare__() must return a dict in this slice".into());
                };
                for (key, value) in entries {
                    let Some(key) = protocol::string_value(&self.state.heap, &key)? else {
                        return Err("metaclass namespace keys must be strings".into());
                    };
                    prepared_namespace.insert(key, value);
                }
            }
        }
        let parent = self.local_scopes.last().copied();
        let uses_repl_globals = parent
            .map(|scope| self.state.heap.scope_uses_repl_globals(scope))
            .transpose()?
            .unwrap_or(true);
        let scope = self.state.heap.allocate_scope(
            parent,
            uses_repl_globals,
            prepared_namespace,
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
            Ok(Execution::Pending) => unreachable!("execute_code drains pending quanta"),
            Ok(Execution::Blocked(_)) => unreachable!("immediate code cannot suspend"),
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
            for method in super::stdlib::unittest::TEST_CASE_TYPE.methods {
                attributes.insert(
                    method.name.into(),
                    Value::Native(NativeValue::NativeMethod(method)),
                );
            }
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
                if value.object_id().is_some_and(|id| {
                    matches!(self.state.heap.get(id), Ok(Object::Function { .. }))
                }) {
                    continue;
                }
                let member = self.allocate_object(Object::EnumMember {
                    name: member_name.clone(),
                    value,
                })?;
                attributes.insert(member_name, member);
                enum_members.push(member);
            }
        }
        let descriptor_candidates = attributes
            .iter()
            .map(|(name, value)| (name.clone(), *value))
            .collect::<Vec<_>>();
        let class = if let Some(metaclass_id) = metaclass.object_id() {
            if let Some((owner, constructor)) =
                self.class_attribute_entry(metaclass_id, "__new__")?
            {
                let constructor =
                    self.bind_descriptor(constructor, Some(metaclass), metaclass_id, owner)?;
                let class_name = self.allocate_string(name.clone())?;
                let bases_value = self.allocate_object(Object::Tuple(bases.clone()))?;
                let mut namespace_entries = Vec::with_capacity(descriptor_candidates.len());
                for (attribute_name, value) in &descriptor_candidates {
                    namespace_entries.push((self.allocate_string(attribute_name.clone())?, *value));
                }
                let namespace = self.allocate_object(Object::Dict(namespace_entries))?;
                self.invoke_value(constructor, vec![class_name, bases_value, namespace])?
            } else {
                self.allocate_class(ClassDefinition {
                    name: name.clone(),
                    bases: bases.clone(),
                    user_bases: user_bases.clone(),
                    mro,
                    metaclass,
                    layout,
                    attributes,
                    dataclass_fields,
                    enum_members,
                })?
            }
        } else {
            self.allocate_class(ClassDefinition {
                name: name.clone(),
                bases: bases.clone(),
                user_bases: user_bases.clone(),
                mro,
                metaclass,
                layout,
                attributes,
                dataclass_fields,
                enum_members,
            })?
        };
        let class_id = class
            .object_id()
            .ok_or("metaclass __new__() must return a class")?;
        if !matches!(self.state.heap.get(class_id)?, Object::Class { .. }) {
            return Err("metaclass __new__() must return a class in this slice".into());
        }
        if let Some(base) = user_bases.first().copied() {
            if let Some((owner, initializer)) =
                self.class_attribute_entry(base, "__init_subclass__")?
            {
                let initializer =
                    self.bind_descriptor(initializer, Some(Value::Object(class_id)), base, owner)?;
                self.invoke_value(initializer, Vec::new())?;
            }
        }
        if let Some(metaclass_id) = metaclass.object_id() {
            if let Some((owner, initializer)) =
                self.class_attribute_entry(metaclass_id, "__init__")?
            {
                let initializer = self.bind_descriptor(
                    initializer,
                    Some(Value::Object(class_id)),
                    metaclass_id,
                    owner,
                )?;
                let bases = self.allocate_object(Object::Tuple(bases))?;
                let mut entries = Vec::with_capacity(descriptor_candidates.len());
                for (name, value) in descriptor_candidates {
                    entries.push((self.allocate_string(name)?, value));
                }
                let namespace = self.allocate_object(Object::Dict(entries))?;
                let class_name = self.allocate_string(name.clone())?;
                let result = self.invoke_value(initializer, vec![class_name, bases, namespace])?;
                if !result.is_none() {
                    return Err("metaclass __init__() should return None".into());
                }
            }
        }
        self.stack.push(Value::Object(class_id));
        Ok(())
    }

    /// Allocate and finish a class after metaclass policy has selected its layout and C3 MRO.
    /// Both ordinary class statements and `type.__new__` use this path.
    fn allocate_class(&mut self, definition: ClassDefinition) -> Result<Value, String> {
        let ClassDefinition {
            name,
            bases,
            user_bases,
            mro,
            metaclass,
            layout,
            attributes,
            dataclass_fields,
            enum_members,
        } = definition;
        let mut type_bases = Vec::new();
        for base in &bases {
            if let Some(base) = self.class_type_id(base)? {
                type_bases.push(base);
            }
        }
        if type_bases.is_empty() {
            type_bases.push(BuiltinType::Object.id());
        }
        let mut type_mro = mro
            .iter()
            .map(|ancestor| match self.state.heap.get(*ancestor)? {
                Object::Class { instance_type, .. } => Ok(*instance_type),
                _ => Err("class MRO contains a non-class object".into()),
            })
            .collect::<Result<Vec<_>, String>>()?;
        let builtin_ancestor = match layout {
            ClassLayout::Object => None,
            ClassLayout::Int => Some(BuiltinType::Int.id()),
            ClassLayout::Type => Some(BuiltinType::Type.id()),
        };
        if let Some(ancestor) = builtin_ancestor {
            if !type_mro.contains(&ancestor) {
                type_mro.push(ancestor);
            }
        }
        if !type_mro.contains(&BuiltinType::Object.id()) {
            type_mro.push(BuiltinType::Object.id());
        }
        let metaclass_type = self
            .class_type_id(&metaclass)?
            .ok_or("metaclass must be a type")?;
        let python_layout = match layout {
            ClassLayout::Object => PyLayout::Object,
            ClassLayout::Int => PyLayout::Int,
            ClassLayout::Type => PyLayout::Type,
        };
        let instance_type = self.state.types.register(
            name.clone(),
            type_bases,
            type_mro,
            metaclass_type,
            attributes.clone(),
            python_layout,
        )?;
        let descriptors = attributes
            .iter()
            .map(|(name, value)| (name.clone(), *value))
            .collect::<Vec<_>>();
        let class = self.allocate_object(Object::Class {
            instance_type,
            name,
            bases: user_bases,
            mro,
            metaclass,
            layout,
            attributes,
            is_dataclass: false,
            dataclass_fields,
            enum_members,
        })?;
        self.state.types.finish(instance_type, class)?;
        let class_id = class.object_id().expect("allocated class has an object id");
        for (_, descriptor) in &descriptors {
            let Some(function_id) = descriptor.object_id() else {
                continue;
            };
            if let Object::Function { defining_class, .. } = self.state.heap.get_mut(function_id)? {
                *defining_class = Some(class_id);
            }
        }
        for (attribute_name, descriptor) in descriptors {
            let Some(descriptor_id) = descriptor.object_id() else {
                continue;
            };
            let Object::Instance {
                class: descriptor_class,
                ..
            } = self.state.heap.get(descriptor_id)?.clone()
            else {
                continue;
            };
            let Some((owner, set_name)) =
                self.class_attribute_entry(descriptor_class, "__set_name__")?
            else {
                continue;
            };
            let set_name =
                self.bind_descriptor(set_name, Some(descriptor), descriptor_class, owner)?;
            let attribute_name = self.allocate_string(attribute_name)?;
            self.invoke_value(set_name, vec![class, attribute_name])?;
        }
        Ok(class)
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
        Ok(self
            .class_attribute_entry(class, name)?
            .map(|(_, value)| value))
    }

    fn class_attribute_entry(
        &mut self,
        class: super::heap::ObjectId,
        name: &str,
    ) -> Result<Option<(super::heap::ObjectId, Value)>, String> {
        let Object::Class {
            attributes, mro, ..
        } = self.state.heap.get(class)?
        else {
            return Err("instance has an invalid class".into());
        };
        if let Some(value) = attributes.get(name) {
            return Ok(Some((class, *value)));
        }
        let ancestors = mro.clone();
        for ancestor in ancestors {
            self.charge_cpu(1)?;
            let Object::Class { attributes, .. } = self.state.heap.get(ancestor)? else {
                return Err("class MRO contains a non-class object".into());
            };
            if let Some(value) = attributes.get(name) {
                return Ok(Some((ancestor, *value)));
            }
        }
        Ok(None)
    }

    fn is_data_descriptor(&mut self, value: &Value) -> Result<bool, String> {
        let Some(id) = value.object_id() else {
            return Ok(false);
        };
        let object = self.state.heap.get(id)?.clone();
        Ok(match object {
            Object::Property { .. } => true,
            Object::Instance { class, .. } => {
                self.class_attribute(class, "__set__")?.is_some()
                    || self.class_attribute(class, "__delete__")?.is_some()
            }
            _ => false,
        })
    }

    fn bind_descriptor(
        &mut self,
        descriptor: Value,
        receiver: Option<Value>,
        accessed_class: super::heap::ObjectId,
        defining_class: super::heap::ObjectId,
    ) -> Result<Value, String> {
        if matches!(
            descriptor.native_value(),
            Some(NativeValue::NativeMethod(method)) if method.name == "__new__"
        ) {
            return Ok(descriptor);
        }
        if matches!(
            descriptor.native_value(),
            Some(NativeValue::NativeMethod(_))
        ) {
            return match receiver {
                Some(receiver) => self.allocate_object(Object::DescriptorBoundMethod {
                    receiver,
                    descriptor,
                    owner: Some(defining_class),
                }),
                None => Ok(descriptor),
            };
        }
        let Some(id) = descriptor.object_id() else {
            return Ok(descriptor);
        };
        match self.state.heap.get(id)?.clone() {
            Object::Function { .. } => match receiver {
                Some(receiver) => self.allocate_object(Object::DescriptorBoundMethod {
                    receiver,
                    descriptor: Value::Object(id),
                    owner: Some(defining_class),
                }),
                None => Ok(descriptor),
            },
            Object::Property { getter, .. } => match receiver {
                Some(receiver) => self.invoke_value(getter, vec![receiver]),
                None => Ok(descriptor),
            },
            Object::StaticMethod { callable } => Ok(callable),
            Object::ClassMethod { callable } => {
                if let Some(function) = callable.object_id() {
                    if matches!(self.state.heap.get(function)?, Object::Function { .. }) {
                        return self.allocate_object(Object::DescriptorBoundMethod {
                            receiver: Value::Object(accessed_class),
                            descriptor: Value::Object(function),
                            owner: Some(defining_class),
                        });
                    }
                }
                Ok(callable)
            }
            Object::Instance { class, .. } => {
                let Some((get_owner, get)) = self.class_attribute_entry(class, "__get__")? else {
                    return Ok(descriptor);
                };
                let get = self.bind_descriptor(get, Some(descriptor), class, get_owner)?;
                self.invoke_value(
                    get,
                    vec![
                        receiver.unwrap_or(Value::None),
                        Value::Object(accessed_class),
                    ],
                )
            }
            _ => Ok(descriptor),
        }
    }

    fn invoke_value(&mut self, callable: Value, arguments: Vec<Value>) -> Result<Value, String> {
        match self.invoke_call(callable, arguments, Vec::new())? {
            CallResult::Value(value) => Ok(value),
            CallResult::Exit(status) => Err(format!("callable exited with status {status}")),
            CallResult::EnteredFrame => unreachable!("invoke_call is immediate"),
            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                unreachable!("immediate call cannot suspend")
            }
        }
    }

    fn invoke_call(
        &mut self,
        callable: Value,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<CallResult, String> {
        let positional = arguments.len();
        let total = positional
            .checked_add(keyword_arguments.len())
            .ok_or("too many call arguments")?;
        let keyword_names = keyword_arguments
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        self.stack.push(callable);
        self.stack.extend(arguments);
        self.stack
            .extend(keyword_arguments.into_iter().map(|(_, value)| value));
        self.call(
            positional,
            &keyword_names,
            &vec![false; total],
            CallMode::Immediate,
        )
    }

    fn invoke_slot(
        &mut self,
        receiver: &Value,
        slot: Slot,
        method_name: &str,
        arguments: Vec<Value>,
    ) -> Result<Option<Value>, String> {
        let type_id = self.type_id(receiver)?;
        let Some(slot_value) = self.state.types.slot(type_id, slot)? else {
            return Ok(None);
        };
        let slot_descriptor = match slot_value {
            SlotValue::NativeBinary(call) => {
                let [argument] = arguments.as_slice() else {
                    return Err(
                        "binary protocol slot received the wrong number of arguments".into(),
                    );
                };
                return call(self, *receiver, *argument)
                    .map_err(|error| self.record_native_error(error));
            }
            SlotValue::NativeTernary(call) => {
                let [first, second] = arguments.as_slice() else {
                    return Err(
                        "ternary protocol slot received the wrong number of arguments".into(),
                    );
                };
                return call(self, *receiver, *first, *second)
                    .map_err(|error| self.record_native_error(error));
            }
            SlotValue::NativeUnary(call) => {
                if !arguments.is_empty() {
                    return Err("unary protocol slot received arguments".into());
                }
                return call(self, *receiver).map_err(|error| self.record_native_error(error));
            }
            SlotValue::NativeSlice(_) => {
                return Err("slice protocol invoked through the wrong operation".into())
            }
            SlotValue::Descriptor(descriptor) => descriptor,
        };
        let Some(id) = receiver.object_id() else {
            return Ok(None);
        };
        let Object::Instance { class, .. } = self.state.heap.get(id)? else {
            return Ok(None);
        };
        let class = *class;
        let (defining_class, _) = self
            .class_attribute_entry(class, method_name)?
            .ok_or("cached type slot has no descriptor")?;
        let callable =
            self.bind_descriptor(slot_descriptor, Some(*receiver), class, defining_class)?;
        self.invoke_value(callable, arguments).map(Some)
    }

    fn truth_value(&mut self, value: &Value) -> Result<bool, String> {
        if let Some(result) = self.invoke_slot(value, Slot::Bool, "__bool__", Vec::new())? {
            return result
                .bool_value()
                .ok_or_else(|| "__bool__ should return bool".into());
        }
        if let Some(result) = self.invoke_slot(value, Slot::Length, "__len__", Vec::new())? {
            let length = protocol::int_value(&self.state.heap, &result)
                .ok_or_else(|| "__len__ should return int".to_string())?;
            if length < 0 {
                return Err("__len__ should return >= 0".into());
            }
            return Ok(length != 0);
        }
        protocol::truth(&self.state.heap, value)
    }

    fn repr_value(&mut self, value: &Value) -> Result<String, String> {
        if let Some(result) = self.invoke_slot(value, Slot::Repr, "__repr__", Vec::new())? {
            return protocol::string_value(&self.state.heap, &result)?
                .ok_or_else(|| "__repr__ should return str".into());
        }
        protocol::repr(&self.state.heap, value)
    }

    fn display_value(&mut self, value: &Value) -> Result<String, String> {
        if let Some(result) = self.invoke_slot(value, Slot::String, "__str__", Vec::new())? {
            return protocol::string_value(&self.state.heap, &result)?
                .ok_or_else(|| "__str__ should return str".into());
        }
        if self
            .state
            .types
            .slot(self.type_id(value)?, Slot::Repr)?
            .is_some()
        {
            return self.repr_value(value);
        }
        protocol::display(&self.state.heap, value)
    }

    fn compare_values(&mut self, left: &Value, right: &Value) -> Result<Ordering, String> {
        if let (Some(left_id), Some(right_id)) = (left.object_id(), right.object_id()) {
            let sequences = match (
                self.state.heap.get(left_id)?,
                self.state.heap.get(right_id)?,
            ) {
                (Object::List(left), Object::List(right))
                | (Object::Tuple(left), Object::Tuple(right)) => {
                    Some((left.clone(), right.clone()))
                }
                _ => None,
            };
            if let Some((left, right)) = sequences {
                for (left, right) in left.iter().zip(&right) {
                    self.charge_cpu(1)?;
                    if protocol::identical(left, right) {
                        continue;
                    }
                    let ordering = self.compare_values(left, right)?;
                    if ordering != Ordering::Equal {
                        return Ok(ordering);
                    }
                }
                return Ok(left.len().cmp(&right.len()));
            }
        }
        if let Some(equal) = self.invoke_slot(left, Slot::Equal, "__eq__", vec![*right])? {
            if self.truth_value(&equal)? {
                return Ok(Ordering::Equal);
            }
        }
        if let Some(less) = self.invoke_slot(left, Slot::LessThan, "__lt__", vec![*right])? {
            if self.truth_value(&less)? {
                return Ok(Ordering::Less);
            }
        }
        if let Some(less) = self.invoke_slot(right, Slot::LessThan, "__lt__", vec![*left])? {
            if self.truth_value(&less)? {
                return Ok(Ordering::Greater);
            }
        }
        protocol::compare(&self.state.heap, left, right)
    }

    fn super_attribute(
        &mut self,
        start_class: super::heap::ObjectId,
        receiver: &Value,
        name: &str,
    ) -> Result<(super::heap::ObjectId, Value, super::heap::ObjectId), String> {
        let accessed_class = if let Some(id) = receiver.object_id() {
            match self.state.heap.get(id)? {
                Object::Instance { class, .. } => *class,
                Object::Class { .. } => id,
                _ => return Err("super() receiver is not an instance or class".into()),
            }
        } else {
            return Err("super() receiver is not an instance or class".into());
        };
        let Object::Class { mro, .. } = self.state.heap.get(accessed_class)? else {
            return Err("super() receiver has an invalid class".into());
        };
        let mut classes = Vec::with_capacity(mro.len().saturating_add(1));
        classes.push(accessed_class);
        classes.extend(mro.iter().copied());
        let start = classes
            .iter()
            .position(|class| *class == start_class)
            .ok_or("super(type, obj): obj is not an instance or subtype of type")?;
        for class in classes.into_iter().skip(start.saturating_add(1)) {
            self.charge_cpu(1)?;
            let Object::Class { attributes, .. } = self.state.heap.get(class)? else {
                return Err("super MRO contains a non-class object".into());
            };
            if let Some(value) = attributes.get(name) {
                return Ok((class, *value, accessed_class));
            }
        }
        if name == "__new__"
            && matches!(
                self.state.heap.get(start_class)?,
                Object::Class {
                    layout: ClassLayout::Type,
                    ..
                }
            )
        {
            if let Some(descriptor) = self
                .state
                .types
                .attribute(BuiltinType::Type.id(), "__new__")?
            {
                return Ok((start_class, descriptor, accessed_class));
            }
        }
        Err(format!("super object has no attribute {name:?}"))
    }

    /// Invoke a canonical builtin type object.
    ///
    /// Construction is centralized here so type identity, `type()`, and calling a type do not
    /// depend on the unrelated builtin-function dispatch table. Collection construction remains
    /// metered through the normal iterator and allocation paths.
    fn call_builtin_type(
        &mut self,
        builtin_type: BuiltinType,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<CallResult, String> {
        if !keyword_arguments.is_empty() {
            return Err(format!(
                "{}() does not accept keyword arguments in this slice",
                builtin_type.name()
            ));
        }
        let value = match builtin_type {
            BuiltinType::Type => match arguments.as_slice() {
                [value] => self.type_of(value)?,
                [name, bases, namespace] => {
                    let name = protocol::string_value(&self.state.heap, name)?
                        .ok_or("type name must be a string")?;
                    self.new_type(
                        Value::Native(NativeValue::BuiltinType(BuiltinType::Type)),
                        name,
                        *bases,
                        *namespace,
                    )
                    .map_err(|error| error.to_string())?
                }
                _ => return Err("type() expects one or three arguments".into()),
            },
            BuiltinType::Object => {
                expect_arity(&arguments, 0, 0)?;
                return Err("direct object() instances are not implemented".into());
            }
            BuiltinType::None => {
                expect_arity(&arguments, 0, 0)?;
                Value::None
            }
            BuiltinType::Bool => {
                expect_arity(&arguments, 0, 1)?;
                Value::Bool(match arguments.first() {
                    Some(value) => self.truth_value(value)?,
                    None => false,
                })
            }
            BuiltinType::Int => {
                expect_arity(&arguments, 0, 2)?;
                if let Some(base) = arguments.get(1) {
                    let base = protocol::int_value(&self.state.heap, base).ok_or_else(|| {
                        self.record_native_error(PyError::type_error(
                            "int() base must be an integer",
                        ))
                    })?;
                    let text = protocol::string_value(&self.state.heap, &arguments[0])?
                        .ok_or_else(|| {
                            self.record_native_error(PyError::type_error(
                                "int() can't convert non-string with explicit base",
                            ))
                        })?;
                    self.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
                    let decimal = super::number::parse_integer_text(&text, base)
                        .map_err(|error| self.record_native_error(error))?;
                    self.new_integer(&decimal)
                        .map_err(|error| self.record_native_error(error))?
                } else {
                    match arguments.first() {
                        None => Value::Int(0),
                        Some(value) if self.is_bigint(value)? => *value,
                        Some(value)
                            if protocol::string_value(&self.state.heap, value)?.is_some() =>
                        {
                            let text =
                                protocol::string_value(&self.state.heap, value)?.expect("guarded");
                            self.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
                            let decimal = super::number::parse_integer_text(&text, 10)
                                .map_err(|error| self.record_native_error(error))?;
                            self.new_integer(&decimal)
                                .map_err(|error| self.record_native_error(error))?
                        }
                        Some(value) => Value::Int(
                            protocol::int_value(&self.state.heap, value)
                                .or_else(|| value.as_int())
                                .ok_or("int() argument is not supported")?,
                        ),
                    }
                }
            }
            BuiltinType::Float => {
                expect_arity(&arguments, 0, 1)?;
                let converted = match arguments.first() {
                    None => 0.0,
                    Some(value)
                        if matches!(
                            super::number::view(&self.state.heap, value),
                            Some(super::number::NumberRef::Float(_))
                        ) =>
                    {
                        let Some(super::number::NumberRef::Float(value)) =
                            super::number::view(&self.state.heap, value)
                        else {
                            unreachable!()
                        };
                        value
                    }
                    Some(value) if protocol::string_value(&self.state.heap, value)?.is_some() => {
                        protocol::string_value(&self.state.heap, value)?
                            .expect("guarded")
                            .parse::<f64>()
                            .map_err(|_| "could not convert string to float")?
                    }
                    Some(value) => self
                        .numeric_float(value)
                        .map_err(|_| "float() argument is not supported")?,
                };
                Value::Float(converted)
            }
            BuiltinType::String => {
                expect_arity(&arguments, 0, 1)?;
                let value = match arguments.first() {
                    Some(value) => self.display_value(value)?,
                    None => String::new(),
                };
                self.allocate_string(value)?
            }
            BuiltinType::Bytes | BuiltinType::ByteArray => {
                expect_arity(&arguments, 0, 2)?;
                let value = match arguments.as_slice() {
                    [] => Vec::new(),
                    [value] if protocol::bytes_value(&self.state.heap, value)?.is_some() => {
                        protocol::bytes_value(&self.state.heap, value)?.expect("guarded")
                    }
                    [value] if protocol::int_value(&self.state.heap, value).is_some() => {
                        let length = usize::try_from(
                            protocol::int_value(&self.state.heap, value).expect("guarded"),
                        )
                        .map_err(|_| "negative count")?;
                        self.reserve_result(length)?;
                        vec![0; length]
                    }
                    [value] => {
                        let items = self.iterable_values(value)?;
                        let mut bytes = Vec::with_capacity(items.len());
                        for item in items {
                            let byte = protocol::int_value(&self.state.heap, &item)
                                .and_then(|value| u8::try_from(value).ok())
                                .ok_or("bytes must be in range(0, 256)")?;
                            bytes.push(byte);
                        }
                        bytes
                    }
                    [value, encoding] => {
                        let text = protocol::string_value(&self.state.heap, value)?
                            .ok_or("encoding without a string argument")?;
                        let encoding = protocol::string_value(&self.state.heap, encoding)?
                            .ok_or("bytes() encoding must be a string")?;
                        if !matches!(encoding.to_ascii_lowercase().as_str(), "utf-8" | "utf8") {
                            return Err("only UTF-8 encoding is supported".into());
                        }
                        text.into_bytes()
                    }
                    _ => unreachable!("arity checked"),
                };
                if builtin_type == BuiltinType::Bytes {
                    self.allocate_bytes(value)?
                } else {
                    self.allocate_bytearray(value)?
                }
            }
            BuiltinType::List | BuiltinType::Tuple | BuiltinType::Set => {
                expect_arity(&arguments, 0, 1)?;
                let values = arguments
                    .first()
                    .map(|value| self.iterable_values(value))
                    .transpose()?
                    .unwrap_or_default();
                let object = match builtin_type {
                    BuiltinType::List => Object::List(values),
                    BuiltinType::Tuple => Object::Tuple(values),
                    BuiltinType::Set => {
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
                self.allocate_object(object)?
            }
            BuiltinType::Dict => {
                expect_arity(&arguments, 0, 1)?;
                let entries = match arguments.first() {
                    None => Vec::new(),
                    Some(value) if value.object_id().is_some() => {
                        match self.state.heap.get(value.object_id().unwrap())? {
                            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                                entries.clone()
                            }
                            _ => {
                                return Err("dict() argument is not a mapping in this slice".into())
                            }
                        }
                    }
                    Some(_) => return Err("dict() argument is not a mapping in this slice".into()),
                };
                self.allocate_object(Object::Dict(entries))?
            }
            BuiltinType::Function
            | BuiltinType::Module
            | BuiltinType::Iterator
            | BuiltinType::Generator
            | BuiltinType::Exception
            | BuiltinType::Native
            | BuiltinType::Stream
            | BuiltinType::Environment
            | BuiltinType::ArgumentParser
            | BuiltinType::RaisesContext
            | BuiltinType::Property
            | BuiltinType::Regex
            | BuiltinType::Match
            | BuiltinType::Array => {
                return Err(format!("cannot create '{}' instances", builtin_type.name()));
            }
        };
        Ok(CallResult::Value(value))
    }

    fn class_type_id(&self, value: &Value) -> Result<Option<TypeId>, String> {
        Ok(match value.native_value() {
            Some(NativeValue::BuiltinType(builtin)) => Some(builtin.id()),
            Some(NativeValue::ValueKind(kind)) => self.state.types.value_kind_type_id(kind),
            _ if value.object_id().is_some() => {
                match self.state.heap.get(value.object_id().unwrap())? {
                    Object::Class { instance_type, .. } => Some(*instance_type),
                    _ => None,
                }
            }
            _ => None,
        })
    }

    /// Return the Python-level type independently of the value's physical storage shape.
    fn type_id(&self, value: &Value) -> Result<TypeId, String> {
        if value.inline_string_len().is_some() {
            return Ok(BuiltinType::String.id());
        }
        Ok(match value.tag() {
            ValueTag::None => BuiltinType::None.id(),
            ValueTag::Bool => BuiltinType::Bool.id(),
            ValueTag::Int => BuiltinType::Int.id(),
            ValueTag::Float => BuiltinType::Float.id(),
            ValueTag::Registered => self
                .state
                .types
                .value_kind_type_id_by_index(value.registered_parts().expect("tag checked").0)
                .ok_or("invalid registered value kind")?,
            ValueTag::Native => match value.native_value().expect("native tag checked") {
                NativeValue::BuiltinType(_) | NativeValue::ValueKind(_) => BuiltinType::Type.id(),
                NativeValue::Function(_)
                | NativeValue::NativeFunction(_)
                | NativeValue::NativeMethod(_) => BuiltinType::Function.id(),
                NativeValue::Module(_) => BuiltinType::Module.id(),
                NativeValue::Stream(_) => BuiltinType::Stream.id(),
                NativeValue::Environment => BuiltinType::Environment.id(),
                _ => BuiltinType::Native.id(),
            },
            ValueTag::Object => self
                .state
                .heap
                .type_id(value.object_id().expect("object tag checked"))?,
            ValueTag::SmallString0
            | ValueTag::SmallString1
            | ValueTag::SmallString2
            | ValueTag::SmallString3
            | ValueTag::SmallString4
            | ValueTag::SmallString5
            | ValueTag::SmallString6
            | ValueTag::SmallString7
            | ValueTag::SmallString8
            | ValueTag::SmallString9
            | ValueTag::SmallString10
            | ValueTag::SmallString11
            | ValueTag::SmallString12
            | ValueTag::SmallString13
            | ValueTag::SmallString14
            | ValueTag::SmallString15 => unreachable!("handled above"),
        })
    }

    fn type_of(&self, value: &Value) -> Result<Value, String> {
        self.state.types.value(self.type_id(value)?)
    }

    fn is_instance(&mut self, value: &Value, class: &Value) -> Result<bool, String> {
        if let Some(class_id) = class.object_id() {
            if let Object::Tuple(classes) = self.state.heap.get(class_id)? {
                let classes = classes.clone();
                for class in classes {
                    self.charge_cpu(1)?;
                    if self.is_instance(value, &class)? {
                        return Ok(true);
                    }
                }
                return Ok(false);
            }
        }
        let class = self
            .class_type_id(class)?
            .ok_or("isinstance() requires a class argument")?;
        self.state.types.is_subclass(self.type_id(value)?, class)
    }

    fn is_subclass(&mut self, class: &Value, base: &Value) -> Result<bool, String> {
        if let Some(base_id) = base.object_id() {
            if let Object::Tuple(bases) = self.state.heap.get(base_id)? {
                let bases = bases.clone();
                for base in bases {
                    self.charge_cpu(1)?;
                    if self.is_subclass(class, &base)? {
                        return Ok(true);
                    }
                }
                return Ok(false);
            }
        }
        let class = self
            .class_type_id(class)?
            .ok_or("issubclass() requires a class argument")?;
        let base = self
            .class_type_id(base)?
            .ok_or("issubclass() requires a class argument")?;
        self.state.types.is_subclass(class, base)
    }

    fn get_iterator(&mut self) -> Result<(), String> {
        let iterable = self.pop()?;
        if let Some(id) = iterable.object_id() {
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
                    self.push_materialized(&mut outputs, values[source_index])?;
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
        let Some(id) = self
            .stack
            .last()
            .cloned()
            .ok_or("invalid bytecode stack effect")?
            .object_id()
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
            Ok(Execution::Pending) => unreachable!("execute_code_from drains pending quanta"),
            Ok(Execution::Blocked(_)) => unreachable!("generator execution cannot suspend"),
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
        if operator == UnaryOperator::Not {
            let value = Value::Bool(!self.truth_value(&value)?);
            self.stack.push(value);
            return Ok(());
        };
        let (slot, name) = match operator {
            UnaryOperator::Positive => (Slot::Positive, "__pos__"),
            UnaryOperator::Negative => (Slot::Negative, "__neg__"),
            UnaryOperator::Invert => (Slot::Invert, "__invert__"),
            UnaryOperator::Not => unreachable!("handled above"),
        };
        let result = self
            .invoke_slot(&value, slot, name, Vec::new())?
            .ok_or("bad operand type for unary arithmetic")?;
        self.stack.push(result);
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

    fn build_dict(&mut self, unpacked: &[bool]) -> Result<(), String> {
        let value_count = unpacked
            .iter()
            .try_fold(0usize, |count, unpacked| {
                count.checked_add(if *unpacked { 1 } else { 2 })
            })
            .ok_or("dictionary is too large")?;
        let values = self.take(value_count)?;
        let mut values = values.into_iter();
        let mut entries: Vec<(Value, Value)> = Vec::with_capacity(unpacked.len());
        for unpacked in unpacked {
            let additions = if *unpacked {
                let mapping = values.next().expect("dictionary stack contract");
                match mapping
                    .object_id()
                    .map(|id| self.state.heap.get(id))
                    .transpose()?
                {
                    Some(Object::Dict(entries)) | Some(Object::DefaultDict { entries, .. }) => {
                        entries.clone()
                    }
                    _ => return Err("'**' argument must be a mapping".into()),
                }
            } else {
                vec![(
                    values.next().expect("dictionary key stack contract"),
                    values.next().expect("dictionary value stack contract"),
                )]
            };
            for (key, value) in additions {
                let mut replaced = false;
                for (existing_key, existing_value) in &mut entries {
                    if protocol::identical(existing_key, &key)
                        || protocol::equals(&self.state.heap, existing_key, &key)?
                    {
                        *existing_value = value;
                        replaced = true;
                        break;
                    }
                }
                if !replaced {
                    entries.push((key, value));
                }
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
                if protocol::identical(value, &candidate)
                    || protocol::equals(&self.state.heap, value, &candidate)?
                {
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
        let mut slot_result = match operator {
            ComparisonOperator::Equal => {
                self.invoke_slot(&left, Slot::Equal, "__eq__", vec![right])?
            }
            ComparisonOperator::NotEqual => {
                self.invoke_slot(&left, Slot::NotEqual, "__ne__", vec![right])?
            }
            ComparisonOperator::Less => {
                self.invoke_slot(&left, Slot::LessThan, "__lt__", vec![right])?
            }
            ComparisonOperator::LessEqual => {
                self.invoke_slot(&left, Slot::LessEqual, "__le__", vec![right])?
            }
            ComparisonOperator::Greater => {
                self.invoke_slot(&left, Slot::GreaterThan, "__gt__", vec![right])?
            }
            ComparisonOperator::GreaterEqual => {
                self.invoke_slot(&left, Slot::GreaterEqual, "__ge__", vec![right])?
            }
            ComparisonOperator::In | ComparisonOperator::NotIn => {
                self.invoke_slot(&right, Slot::Contains, "__contains__", vec![left])?
            }
            ComparisonOperator::Is | ComparisonOperator::IsNot => None,
        };
        if slot_result.is_none() {
            let reflected = match operator {
                ComparisonOperator::Equal => Some((Slot::Equal, "__eq__")),
                ComparisonOperator::NotEqual => Some((Slot::NotEqual, "__ne__")),
                ComparisonOperator::Less => Some((Slot::GreaterThan, "__gt__")),
                ComparisonOperator::LessEqual => Some((Slot::GreaterEqual, "__ge__")),
                ComparisonOperator::Greater => Some((Slot::LessThan, "__lt__")),
                ComparisonOperator::GreaterEqual => Some((Slot::LessEqual, "__le__")),
                _ => None,
            };
            if let Some((slot, name)) = reflected {
                slot_result = self.invoke_slot(&right, slot, name, vec![left])?;
            }
        }
        if slot_result.is_none() && matches!(operator, ComparisonOperator::NotEqual) {
            let mut equality = self.invoke_slot(&left, Slot::Equal, "__eq__", vec![right])?;
            if equality.is_none() {
                equality = self.invoke_slot(&right, Slot::Equal, "__eq__", vec![left])?;
            }
            if let Some(value) = equality {
                slot_result = Some(Value::Bool(!self.truth_value(&value)?));
            }
        }
        if let Some(value) = slot_result {
            if matches!(operator, ComparisonOperator::NotIn) {
                let result = !self.truth_value(&value)?;
                self.stack.push(Value::Bool(result));
            } else {
                self.stack.push(value);
            }
            return Ok(());
        }
        let result = match operator {
            ComparisonOperator::Equal => protocol::equals(&self.state.heap, &left, &right)?,
            ComparisonOperator::NotEqual => !protocol::equals(&self.state.heap, &left, &right)?,
            ComparisonOperator::Less => self.compare_values(&left, &right)? == Ordering::Less,
            ComparisonOperator::LessEqual => {
                self.compare_values(&left, &right)? != Ordering::Greater
            }
            ComparisonOperator::Greater => self.compare_values(&left, &right)? == Ordering::Greater,
            ComparisonOperator::GreaterEqual => {
                self.compare_values(&left, &right)? != Ordering::Less
            }
            ComparisonOperator::In => protocol::contains(&self.state.heap, &right, &left)?,
            ComparisonOperator::NotIn => !protocol::contains(&self.state.heap, &right, &left)?,
            ComparisonOperator::Is => protocol::identical(&left, &right),
            ComparisonOperator::IsNot => !protocol::identical(&left, &right),
        };
        self.stack.push(Value::Bool(result));
        Ok(())
    }

    fn binary(&mut self, operator: BinaryOperator) -> Result<(), String> {
        let right = self.pop()?;
        let left = self.pop()?;
        let value = self.binary_value(operator, left, right)?;
        self.stack.push(value);
        Ok(())
    }

    fn binary_value(
        &mut self,
        operator: BinaryOperator,
        left: Value,
        right: Value,
    ) -> Result<Value, String> {
        let (slot, name, reflected_slot, reflected_name) = match operator {
            BinaryOperator::Add => (Slot::Add, "__add__", Slot::ReflectedAdd, "__radd__"),
            BinaryOperator::Subtract => (
                Slot::Subtract,
                "__sub__",
                Slot::ReflectedSubtract,
                "__rsub__",
            ),
            BinaryOperator::Multiply => (
                Slot::Multiply,
                "__mul__",
                Slot::ReflectedMultiply,
                "__rmul__",
            ),
            BinaryOperator::MatrixMultiply => (
                Slot::MatrixMultiply,
                "__matmul__",
                Slot::ReflectedMatrixMultiply,
                "__rmatmul__",
            ),
            BinaryOperator::Power => (Slot::Power, "__pow__", Slot::ReflectedPower, "__rpow__"),
            BinaryOperator::Divide => (
                Slot::Divide,
                "__truediv__",
                Slot::ReflectedDivide,
                "__rtruediv__",
            ),
            BinaryOperator::FloorDivide => (
                Slot::FloorDivide,
                "__floordiv__",
                Slot::ReflectedFloorDivide,
                "__rfloordiv__",
            ),
            BinaryOperator::Remainder => (
                Slot::Remainder,
                "__mod__",
                Slot::ReflectedRemainder,
                "__rmod__",
            ),
            BinaryOperator::LeftShift => (
                Slot::LeftShift,
                "__lshift__",
                Slot::ReflectedLeftShift,
                "__rlshift__",
            ),
            BinaryOperator::RightShift => (
                Slot::RightShift,
                "__rshift__",
                Slot::ReflectedRightShift,
                "__rrshift__",
            ),
            BinaryOperator::BitwiseAnd => (
                Slot::BitwiseAnd,
                "__and__",
                Slot::ReflectedBitwiseAnd,
                "__rand__",
            ),
            BinaryOperator::BitwiseXor => (
                Slot::BitwiseXor,
                "__xor__",
                Slot::ReflectedBitwiseXor,
                "__rxor__",
            ),
            BinaryOperator::BitwiseOr => (
                Slot::BitwiseOr,
                "__or__",
                Slot::ReflectedBitwiseOr,
                "__ror__",
            ),
        };
        if let Some(value) = self.invoke_slot(&left, slot, name, vec![right])? {
            return Ok(value);
        }
        if let Some(value) = self.invoke_slot(&right, reflected_slot, reflected_name, vec![left])? {
            return Ok(value);
        }
        Err("unsupported arithmetic operands".into())
    }

    fn format_value(&mut self, conversion: Option<char>, format_spec: &str) -> Result<(), String> {
        let value = self.pop()?;
        let converted = match conversion {
            Some('r' | 'a') => Some(protocol::repr(&self.state.heap, &value)?),
            Some('s') => Some(protocol::display(&self.state.heap, &value)?),
            Some(other) => return Err(format!("unsupported f-string conversion !{other}")),
            None => None,
        };
        let rendered = if format_spec.is_empty() {
            converted.unwrap_or(protocol::display(&self.state.heap, &value)?)
        } else if format_spec.contains(['{', '}']) {
            return Err("nested f-string format specifications are not implemented".into());
        } else if let Some(converted) = converted {
            format_text(&converted, format_spec)?
        } else {
            self.format_unconverted_value(&value, format_spec)?
        };
        self.charge_cpu(u64::try_from(rendered.len()).unwrap_or(u64::MAX))?;
        let rendered = self.allocate_string(rendered)?;
        self.stack.push(rendered);
        Ok(())
    }

    fn format_unconverted_value(&self, value: &Value, spec: &str) -> Result<String, String> {
        let presentation = spec.chars().last().unwrap_or(' ');
        if matches!(presentation, 'f' | 'e' | 'E') {
            let number = super::number::as_f64(&self.state.heap, value)
                .ok_or("floating-point format requires a number")?;
            return format_float(number, spec);
        }
        if presentation == 'd' {
            let text = self
                .bigint_operand(value)
                .map_err(|_| "integer format requires an integer")?
                .to_string();
            return pad_number(text, &spec[..spec.len() - 1]);
        }
        if let Some(text) = protocol::string_value(&self.state.heap, value)? {
            return format_text(&text, spec);
        }
        Err(format!("unsupported format specification {spec:?}"))
    }

    fn is_bigint(&self, value: &Value) -> Result<bool, String> {
        Ok(matches!(
            super::number::view(&self.state.heap, value),
            Some(super::number::NumberRef::BigInt(_))
        ))
    }

    fn bigint_operand(&self, value: &Value) -> Result<BigInt, String> {
        match super::number::view(&self.state.heap, value) {
            Some(super::number::NumberRef::Int(value)) => Ok(BigInt::from(value)),
            Some(super::number::NumberRef::BigInt(value)) => Ok(value.clone()),
            Some(super::number::NumberRef::Float(_)) | None => {
                Err("unsupported arithmetic operands".into())
            }
        }
    }

    fn numeric_float(&self, value: &Value) -> Result<f64, String> {
        if let Some(id) = value.object_id() {
            if let Object::BigInt(value) = self.state.heap.get(id)? {
                return value
                    .to_f64()
                    .ok_or_else(|| "int too large to convert to float".into());
            }
        }
        super::number::as_f64(&self.state.heap, value)
            .ok_or_else(|| "unsupported arithmetic operands".into())
    }

    fn add_numbers(&mut self, left: Value, right: Value) -> Result<Value, String> {
        self.binary_value(BinaryOperator::Add, left, right)
    }

    fn call(
        &mut self,
        positional: usize,
        keyword_names: &[String],
        starred: &[bool],
        mode: CallMode,
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
        let arguments_start = self.stack.len() - count;
        let mut raw_arguments = self.stack.split_off(arguments_start);
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
        if let Some(id) = function.object_id() {
            return match self.state.heap.get(id)?.clone() {
                Object::Function {
                    name,
                    code,
                    closure,
                    defaults,
                    defining_class,
                } => {
                    let method_frame = defining_class.zip(arguments.first().cloned());
                    if let Some((owner, receiver)) = method_frame {
                        self.method_frames.push((owner, receiver));
                    }
                    let result = self.call_python_function(
                        &name,
                        &code,
                        closure,
                        &defaults,
                        FunctionInvocation {
                            arguments,
                            keyword_arguments,
                            mode,
                            pop_method_frame: method_frame.is_some(),
                        },
                    );
                    if method_frame.is_some() && !matches!(result, Ok(CallResult::EnteredFrame)) {
                        self.method_frames.pop();
                    }
                    result
                }
                Object::DescriptorBoundMethod {
                    receiver,
                    descriptor,
                    owner,
                } => {
                    if let Some(function) = descriptor.object_id() {
                        let Object::Function {
                            name,
                            code,
                            closure,
                            defaults,
                            ..
                        } = self.state.heap.get(function)?.clone()
                        else {
                            return Err("bound descriptor is not callable".into());
                        };
                        arguments.insert(0, receiver);
                        if let Some(owner) = owner {
                            self.method_frames.push((owner, receiver));
                        }
                        let result = self.call_python_function(
                            &name,
                            &code,
                            closure,
                            &defaults,
                            FunctionInvocation {
                                arguments,
                                keyword_arguments,
                                mode,
                                pop_method_frame: owner.is_some(),
                            },
                        );
                        if owner.is_some() && !matches!(result, Ok(CallResult::EnteredFrame)) {
                            self.method_frames.pop();
                        }
                        result
                    } else if let Some(NativeValue::NativeMethod(method)) =
                        descriptor.native_value()
                    {
                        let call = CallArgs::new(arguments, keyword_arguments);
                        match (method.call)(self, receiver, call) {
                            Ok(value) => Ok(CallResult::Value(value)),
                            Err(PyError {
                                kind: PyErrorKind::Exit(status),
                                ..
                            }) => Ok(CallResult::Exit(status)),
                            Err(error) => Err(self.record_native_error(error)),
                        }
                    } else {
                        Err("bound descriptor is not callable".into())
                    }
                }
                Object::Instance { class, .. } => {
                    let type_id = self.state.heap.type_id(id)?;
                    if self.state.types.slot(type_id, Slot::Call)?.is_none() {
                        return Err("object is not callable".into());
                    }
                    let (defining_class, descriptor) = self
                        .class_attribute_entry(class, "__call__")?
                        .ok_or("call slot has no descriptor")?;
                    let callable = self.bind_descriptor(
                        descriptor,
                        Some(Value::Object(id)),
                        class,
                        defining_class,
                    )?;
                    self.invoke_call(callable, arguments, keyword_arguments)
                }
                Object::Class {
                    name,
                    metaclass,
                    layout,
                    is_dataclass,
                    dataclass_fields,
                    enum_members,
                    ..
                } => {
                    if let Some(metaclass_id) = metaclass.object_id() {
                        if let Some((owner, descriptor)) =
                            self.class_attribute_entry(metaclass_id, "__call__")?
                        {
                            let callable = self.bind_descriptor(
                                descriptor,
                                Some(Value::Object(id)),
                                metaclass_id,
                                owner,
                            )?;
                            return self.invoke_call(callable, arguments, keyword_arguments);
                        }
                    }
                    if !enum_members.is_empty() {
                        if !keyword_arguments.is_empty() || arguments.len() != 1 {
                            return Err(format!("{name}() expects one value"));
                        }
                        for member in enum_members {
                            let Object::EnumMember { value, .. } = self
                                .state
                                .heap
                                .get(member.object_id().ok_or("invalid enum member")?)?
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
                            let created = if let Some((owner, constructor)) =
                                self.class_attribute_entry(id, "__new__")?
                            {
                                let constructor = self.bind_descriptor(
                                    constructor,
                                    Some(Value::Object(id)),
                                    id,
                                    owner,
                                )?;
                                match self.invoke_call(
                                    constructor,
                                    arguments.clone(),
                                    keyword_arguments.clone(),
                                )? {
                                    CallResult::Value(value) => value,
                                    CallResult::Exit(status) => {
                                        return Ok(CallResult::Exit(status))
                                    }
                                    CallResult::EnteredFrame => {
                                        unreachable!("invoke_call is immediate")
                                    }
                                    CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                                        unreachable!("immediate call cannot suspend")
                                    }
                                }
                            } else {
                                if !keyword_arguments.is_empty() || arguments.len() != 3 {
                                    return Err(
                                        "type construction expects name, bases, and namespace"
                                            .into(),
                                    );
                                }
                                let name = protocol::string_value(&self.state.heap, &arguments[0])?
                                    .ok_or("type name must be a string")?;
                                self.new_type(Value::Object(id), name, arguments[1], arguments[2])
                                    .map_err(|error| error.to_string())?
                            };
                            if created.object_id().is_some_and(|created_id| {
                                matches!(self.state.heap.get(created_id), Ok(Object::Class { .. }))
                            }) {
                                if let Some((owner, initializer)) =
                                    self.class_attribute_entry(id, "__init__")?
                                {
                                    let initializer = self.bind_descriptor(
                                        initializer,
                                        Some(created),
                                        id,
                                        owner,
                                    )?;
                                    let result = match self.invoke_call(
                                        initializer,
                                        arguments,
                                        keyword_arguments,
                                    )? {
                                        CallResult::Value(value) => value,
                                        CallResult::Exit(status) => {
                                            return Ok(CallResult::Exit(status))
                                        }
                                        CallResult::EnteredFrame => {
                                            unreachable!("invoke_call is immediate")
                                        }
                                        CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                                            unreachable!("immediate call cannot suspend")
                                        }
                                    };
                                    if !result.is_none() {
                                        return Err(
                                            "metaclass __init__() should return None".into()
                                        );
                                    }
                                }
                            }
                            return Ok(CallResult::Value(created));
                        }
                    };
                    let instance = self.allocate_object(Object::Instance { class: id, payload })?;
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
                                .map(|(_, value)| *value)
                                .or_else(|| arguments.get(index).cloned())
                                .or(*default)
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
                        self.state.heap.extend_attributes(
                            instance.object_id().expect("instances are heap objects"),
                            values,
                            &mut self.interp.resources,
                        )?;
                    } else if let Some(initializer) = self.class_attribute(id, "__init__")? {
                        let Some(function) = initializer.object_id() else {
                            return Err(format!("{name}.__init__ is not callable"));
                        };
                        let Object::Function {
                            name: function_name,
                            code,
                            closure,
                            defaults,
                            ..
                        } = self.state.heap.get(function)?.clone()
                        else {
                            return Err(format!("{name}.__init__ is not a function"));
                        };
                        arguments.insert(0, instance);
                        match self.call_python_function(
                            &function_name,
                            &code,
                            closure,
                            &defaults,
                            FunctionInvocation {
                                arguments,
                                keyword_arguments,
                                mode: CallMode::Immediate,
                                pop_method_frame: false,
                            },
                        )? {
                            CallResult::Value(value) if value.is_none() => {}
                            CallResult::Value(_) => {
                                return Err("__init__() should return None".into())
                            }
                            CallResult::Exit(status) => return Ok(CallResult::Exit(status)),
                            CallResult::EnteredFrame => {
                                unreachable!("immediate initializer entered a frame")
                            }
                            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                                unreachable!("immediate initializer cannot suspend")
                            }
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
        if let Some(NativeValue::BuiltinType(builtin_type)) = function.native_value() {
            return self.call_builtin_type(builtin_type, arguments, keyword_arguments);
        }
        if let Some(NativeValue::ValueKind(kind)) = function.native_value() {
            let call = CallArgs::new(arguments, keyword_arguments);
            return (kind.construct)(self, call)
                .map(CallResult::Value)
                .map_err(|error| self.record_native_error(error));
        }
        if let Some(NativeValue::ExceptionType(exception_type)) = function.native_value() {
            expect_arity(&arguments, 0, 1)?;
            let message = arguments
                .first()
                .map(|value| protocol::display(&self.state.heap, value))
                .transpose()?
                .unwrap_or_default();
            return Ok(CallResult::Value(
                self.allocate_exception(exception_type.0.to_string(), message)?,
            ));
        }
        if let Some(NativeValue::NativeMethod(method)) = function.native_value() {
            if method.name != "__new__" || arguments.is_empty() {
                return Err("unbound native method requires a receiver".into());
            }
            let receiver = arguments.remove(0);
            let call = CallArgs::new(arguments, keyword_arguments);
            return match (method.call)(self, receiver, call) {
                Ok(value) => Ok(CallResult::Value(value)),
                Err(PyError {
                    kind: PyErrorKind::Exit(status),
                    ..
                }) => Ok(CallResult::Exit(status)),
                Err(error) => Err(self.record_native_error(error)),
            };
        }
        if let Some(NativeValue::NativeFunction(function)) = function.native_value() {
            let call = CallArgs::new(arguments, keyword_arguments);
            let retry = match mode {
                CallMode::Deferred(call_span) => Some(PendingNativeCall {
                    function,
                    arguments: call.clone(),
                    call_span,
                }),
                CallMode::Immediate => None,
            };
            let previous_suspend = self.native_suspend_allowed;
            self.native_suspend_allowed = matches!(mode, CallMode::Deferred(_));
            let result = (function.call)(self, call);
            self.native_suspend_allowed = previous_suspend;
            return match result {
                Ok(value) => match self.pending_wait.take() {
                    Some(reason) => Ok(CallResult::Blocked(reason, value)),
                    None => Ok(CallResult::Value(value)),
                },
                Err(PyError {
                    kind: PyErrorKind::Exit(status),
                    ..
                }) => Ok(CallResult::Exit(status)),
                Err(PyError {
                    kind: PyErrorKind::Suspend(reason),
                    ..
                }) => retry
                    .map(|pending| CallResult::Retry(reason, pending))
                    .ok_or_else(|| "native call suspended outside scheduler dispatch".into()),
                Err(error) => Err(self.record_native_error(error)),
            };
        }
        let Some(NativeValue::Function(function)) = function.native_value() else {
            return Err("object is not callable".into());
        };
        if !keyword_arguments.is_empty() && !matches!(function, Builtin::Sorted) {
            return Err("this builtin does not accept keyword arguments".into());
        }
        match function {
            Builtin::Print => {
                let mut rendered = Vec::with_capacity(arguments.len());
                for value in &arguments {
                    rendered.push(self.display_value(value)?);
                }
                let text = rendered.join(" ");
                self.write_output(Stream::Stdout, text.as_bytes());
                self.write_output(Stream::Stdout, b"\n");
                Ok(CallResult::Value(Value::None))
            }
            Builtin::Input => {
                expect_arity(&arguments, 0, 1)?;
                if let Some(prompt) = arguments.first() {
                    let prompt = self.display_value(prompt)?;
                    self.write_output(Stream::Stdout, prompt.as_bytes());
                }
                let marker = Value::Native(NativeValue::Stream(Stream::Stdin));
                let mut text = self
                    .read_stream(&marker, None, true)
                    .map_err(|error| self.record_native_error(error))?;
                if text.is_empty() {
                    return Err(self.record_native_error(PyError::exception(
                        "EOFError",
                        "EOF when reading a line",
                    )));
                }
                if text.ends_with('\n') {
                    text.pop();
                    if text.ends_with('\r') {
                        text.pop();
                    }
                }
                Ok(CallResult::Value(self.allocate_string(text)?))
            }
            Builtin::Exit => {
                expect_arity(&arguments, 0, 1)?;
                let status = arguments
                    .first()
                    .and_then(Value::as_int)
                    .unwrap_or_default();
                Ok(CallResult::Exit(status as i32))
            }
            Builtin::Repr => {
                expect_arity(&arguments, 1, 1)?;
                let value = self.repr_value(&arguments[0])?;
                Ok(CallResult::Value(self.allocate_string(value)?))
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
            Builtin::Length => {
                expect_arity(&arguments, 1, 1)?;
                if let Some(value) =
                    self.invoke_slot(&arguments[0], Slot::Length, "__len__", Vec::new())?
                {
                    let length = value.as_int().ok_or("__len__() should return an integer")?;
                    if length < 0 {
                        return Err("__len__() should return >= 0".into());
                    }
                    return Ok(CallResult::Value(Value::Int(length)));
                }
                let length = if let Some(length) =
                    protocol::string_length(&self.state.heap, &arguments[0])?
                {
                    length
                } else if let Some(id) = arguments[0].object_id() {
                    match self.state.heap.get(id)? {
                        Object::List(values) | Object::Tuple(values) | Object::Set(values) => {
                            values.len()
                        }
                        Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                            entries.len()
                        }
                        Object::BigInt(_)
                        | Object::String(_)
                        | Object::Bytes(_)
                        | Object::ByteArray(_)
                        | Object::Slice { .. }
                        | Object::Exception { .. }
                        | Object::Function { .. }
                        | Object::Class { .. }
                        | Object::Instance { .. }
                        | Object::DescriptorBoundMethod { .. }
                        | Object::Iterator { .. }
                        | Object::CountIterator { .. }
                        | Object::Generator { .. }
                        | Object::Module { .. }
                        | Object::ArrayStorage(_)
                        | Object::Array { .. }
                        | Object::Regex { .. }
                        | Object::Match { .. }
                        | Object::ArgumentParser { .. }
                        | Object::Namespace { .. } => return Err("object has no len()".into()),
                        Object::EnumMember { .. } => return Err("object has no len()".into()),
                        Object::RaisesContext { .. } => return Err("object has no len()".into()),
                        Object::Property { .. }
                        | Object::StaticMethod { .. }
                        | Object::ClassMethod { .. }
                        | Object::Super { .. } => return Err("object has no len()".into()),
                    }
                } else {
                    return Err("object has no len()".into());
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
                            reverse = self.truth_value(&value)?;
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
                        self.stack.push(*function);
                        self.stack.push(value);
                        match self.call(1, &[], &[false], CallMode::Immediate)? {
                            CallResult::Value(key) => key,
                            CallResult::Exit(status) => return Ok(CallResult::Exit(status)),
                            CallResult::EnteredFrame => {
                                unreachable!("immediate call entered a frame")
                            }
                            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                                unreachable!("immediate call cannot suspend")
                            }
                        }
                    } else {
                        value
                    };
                    self.reserve_result(64)?;
                    keyed.push((key, value));
                }
                // Stable insertion sort keeps comparison dispatch and failure order obvious.
                for index in 1..keyed.len() {
                    let mut current = index;
                    while current > 0 {
                        self.charge_cpu(1)?;
                        if self.compare_values(&keyed[current].0, &keyed[current - 1].0)?
                            != if reverse {
                                Ordering::Greater
                            } else {
                                Ordering::Less
                            }
                        {
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
                    let ordering = self.compare_values(&value, &selected)?;
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
                    total = self.add_numbers(total, value)?;
                }
                Ok(CallResult::Value(total))
            }
            Builtin::Absolute => {
                expect_arity(&arguments, 1, 1)?;
                let value = self
                    .invoke_slot(&arguments[0], Slot::Absolute, "__abs__", Vec::new())?
                    .ok_or("bad operand type for abs()")?;
                Ok(CallResult::Value(value))
            }
            Builtin::Iter => {
                expect_arity(&arguments, 1, 1)?;
                let values = self.iterable_values(&arguments[0])?;
                Ok(CallResult::Value(self.allocate_object(
                    Object::Iterator {
                        values,
                        position: 0,
                    },
                )?))
            }
            Builtin::Next => {
                expect_arity(&arguments, 1, 2)?;
                let Some(id) = arguments[0].object_id() else {
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
                    _ => {
                        match self.invoke_slot(&arguments[0], Slot::Next, "__next__", Vec::new()) {
                            Ok(Some(value)) => Some(value),
                            Ok(None) => return Err("next() argument is not an iterator".into()),
                            Err(_error)
                                if self
                                    .pending_exception
                                    .as_ref()
                                    .is_some_and(|exception| exception.kind == "StopIteration")
                                    && arguments.len() == 2 =>
                            {
                                self.pending_exception = None;
                                arguments.get(1).copied()
                            }
                            Err(error) => return Err(error),
                        }
                    }
                };
                let value = match value {
                    Some(value) => value,
                    None => arguments.get(1).cloned().ok_or("StopIteration")?,
                };
                Ok(CallResult::Value(value))
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
                    let tuple = sequences.iter().map(|values| values[index]).collect();
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
                    let truth = self.truth_value(&value)?;
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
            Builtin::Property => {
                expect_arity(&arguments, 1, 1)?;
                Ok(CallResult::Value(self.allocate_object(
                    Object::Property {
                        getter: arguments[0],
                        setter: None,
                    },
                )?))
            }
            Builtin::StaticMethod => {
                expect_arity(&arguments, 1, 1)?;
                Ok(CallResult::Value(self.allocate_object(
                    Object::StaticMethod {
                        callable: arguments[0],
                    },
                )?))
            }
            Builtin::ClassMethod => {
                expect_arity(&arguments, 1, 1)?;
                Ok(CallResult::Value(self.allocate_object(
                    Object::ClassMethod {
                        callable: arguments[0],
                    },
                )?))
            }
            Builtin::Super => {
                expect_arity(&arguments, 0, 2)?;
                let (start_class, receiver) = match arguments.as_slice() {
                    [] => self
                        .method_frames
                        .last()
                        .cloned()
                        .ok_or("super(): no current method context")?,
                    [start_class, receiver]
                        if start_class.object_id().is_some_and(|id| {
                            matches!(self.state.heap.get(id), Ok(Object::Class { .. }))
                        }) =>
                    {
                        (start_class.object_id().unwrap(), *receiver)
                    }
                    _ => return Err("super() expects a class and instance".into()),
                };
                Ok(CallResult::Value(self.allocate_object(Object::Super {
                    start_class,
                    receiver,
                })?))
            }
        }
    }

    fn call_python_function(
        &mut self,
        name: &str,
        code: &Code,
        closure: Option<ScopeId>,
        defaults: &[Value],
        invocation: FunctionInvocation,
    ) -> Result<CallResult, String> {
        let FunctionInvocation {
            arguments,
            keyword_arguments,
            mode,
            pop_method_frame,
        } = invocation;
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
        let positional_len = code
            .parameters
            .iter()
            .position(|parameter| parameter.variadic || parameter.keyword_only)
            .unwrap_or(code.parameters.len());
        if variadic_index.is_none() && arguments.len() > positional_len {
            return Err(format!(
                "{name}() takes {} positional arguments but {} were given",
                positional_len,
                arguments.len()
            ));
        }
        let mut positional = arguments;
        let extra_positional = if variadic_index.is_some() && positional.len() > positional_len {
            positional.split_off(positional_len)
        } else {
            Vec::new()
        };
        let mut locals = code
            .parameters
            .iter()
            .take(positional_len)
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
                .any(|parameter| parameter.name == keyword && !parameter.variadic)
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
        if defaults.len()
            != code
                .parameters
                .iter()
                .filter(|parameter| parameter.has_default)
                .count()
        {
            return Err(format!("{name}() has invalid default argument metadata"));
        }
        for (parameter, default) in code
            .parameters
            .iter()
            .filter(|parameter| parameter.has_default)
            .zip(defaults)
        {
            locals
                .entry(parameter.name.clone())
                .or_insert_with(|| *default);
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
        if let CallMode::Deferred(call_span) = mode {
            self.bytecode_frames.push(BytecodeFrame {
                code: code.clone(),
                instruction_pointer: 0,
                handlers: Vec::new(),
                function_return: Some(FunctionReturn {
                    name: name.to_string(),
                    call_span,
                    outer_stack,
                    pop_method_frame,
                }),
                pending_native_call: None,
            });
            return Ok(CallResult::EnteredFrame);
        }
        let result = self.execute_code(code);
        self.call_depth -= 1;
        self.local_scopes.pop();
        self.stack = outer_stack;
        match result {
            Ok(Execution::Pending) => unreachable!("execute_code drains pending quanta"),
            Ok(Execution::Blocked(_)) => unreachable!("immediate call cannot suspend"),
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
        let positional_len = code
            .parameters
            .iter()
            .position(|parameter| parameter.variadic || parameter.keyword_only)
            .unwrap_or(code.parameters.len());
        if variadic_index.is_none() && arguments.len() > positional_len {
            return Err(format!(
                "{name}() takes {} positional arguments but {} were given",
                positional_len,
                arguments.len()
            ));
        }
        let mut positional = arguments;
        let extra_positional = if variadic_index.is_some() && positional.len() > positional_len {
            positional.split_off(positional_len)
        } else {
            Vec::new()
        };
        let mut locals = code
            .parameters
            .iter()
            .take(positional_len)
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
                .any(|parameter| parameter.name == keyword && !parameter.variadic)
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
        if defaults.len()
            != code
                .parameters
                .iter()
                .filter(|parameter| parameter.has_default)
                .count()
        {
            return Err(format!("{name}() has invalid default argument metadata"));
        }
        for (parameter, default) in code
            .parameters
            .iter()
            .filter(|parameter| parameter.has_default)
            .zip(defaults)
        {
            locals
                .entry(parameter.name.clone())
                .or_insert_with(|| *default);
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

    fn record_native_error(&mut self, error: PyError) -> String {
        let kind = match error.kind {
            PyErrorKind::Type => Some("TypeError"),
            PyErrorKind::Value => Some("ValueError"),
            PyErrorKind::ZeroDivision => Some("ZeroDivisionError"),
            PyErrorKind::Overflow => Some("OverflowError"),
            PyErrorKind::Runtime => Some("RuntimeError"),
            PyErrorKind::Exception(kind) => Some(kind),
            PyErrorKind::Resource
            | PyErrorKind::Raised
            | PyErrorKind::Exit(_)
            | PyErrorKind::Suspend(_) => None,
        };
        if let Some(kind) = kind {
            if let Ok(value) = self.allocate_exception(kind.to_string(), error.message.clone()) {
                self.pending_exception = Some(RaisedException {
                    kind: kind.to_string(),
                    value,
                });
            }
        }
        error.message
    }

    fn resume_native_call(&mut self, pending: PendingNativeCall) -> Result<CallResult, String> {
        let retry = PendingNativeCall {
            function: pending.function,
            arguments: pending.arguments.clone(),
            call_span: pending.call_span,
        };
        let previous_suspend = self.native_suspend_allowed;
        self.native_suspend_allowed = true;
        let result = (pending.function.call)(self, pending.arguments);
        self.native_suspend_allowed = previous_suspend;
        match result {
            Ok(value) => match self.pending_wait.take() {
                Some(reason) => Ok(CallResult::Blocked(reason, value)),
                None => Ok(CallResult::Value(value)),
            },
            Err(PyError {
                kind: PyErrorKind::Exit(status),
                ..
            }) => Ok(CallResult::Exit(status)),
            Err(PyError {
                kind: PyErrorKind::Suspend(reason),
                ..
            }) => Ok(CallResult::Retry(reason, retry)),
            Err(error) => Err(self.record_native_error(error)),
        }
    }

    fn allocate_object(&mut self, object: Object) -> Result<Value, String> {
        self.state.heap.allocate(object, &mut self.interp.resources)
    }

    fn allocate_string(&mut self, value: String) -> Result<Value, String> {
        if let Some(value) = Value::inline_string(&value) {
            Ok(value)
        } else {
            self.allocate_object(Object::String(value))
        }
    }

    fn allocate_bytes(&mut self, value: Vec<u8>) -> Result<Value, String> {
        self.allocate_object(Object::Bytes(value))
    }

    fn allocate_bytearray(&mut self, value: Vec<u8>) -> Result<Value, String> {
        self.allocate_object(Object::ByteArray(value))
    }

    fn allocate_exception(&mut self, kind: String, message: String) -> Result<Value, String> {
        self.allocate_object(Object::Exception { kind, message })
    }

    fn value_from_constant(&mut self, value: &Constant) -> Result<Value, String> {
        Ok(match value {
            Constant::None => Value::None,
            Constant::Bool(value) => Value::Bool(*value),
            Constant::Integer(value) => Value::Int(*value),
            Constant::BigInteger(value) => {
                self.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
                self.reserve_result(value.len().saturating_mul(2))?;
                let value = value
                    .parse::<BigInt>()
                    .map_err(|_| "invalid arbitrary-precision integer literal")?;
                self.allocate_object(Object::BigInt(value))?
            }
            Constant::Float(value) => Value::Float(*value),
            Constant::String(value) => self.allocate_string(value.clone())?,
            Constant::Bytes(value) => self.allocate_bytes(value.clone())?,
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

    fn iterable_values(&mut self, value: &Value) -> Result<Vec<Value>, String> {
        if let Some(iterable) = self.invoke_slot(value, Slot::Iter, "__iter__", Vec::new())? {
            if protocol::identical(value, &iterable) {
                let mut result = Vec::new();
                loop {
                    match self.invoke_slot(&iterable, Slot::Next, "__next__", Vec::new()) {
                        Ok(Some(item)) => self.push_materialized(&mut result, item)?,
                        Ok(None) => return Err("iterator does not define __next__".into()),
                        Err(_error)
                            if self
                                .pending_exception
                                .as_ref()
                                .is_some_and(|exception| exception.kind == "StopIteration") =>
                        {
                            self.pending_exception = None;
                            break;
                        }
                        Err(error) => return Err(error),
                    }
                }
                return Ok(result);
            }
            return self.iterable_values(&iterable);
        }
        let mut result = Vec::new();
        if let Some(value) = protocol::string_value(&self.state.heap, value)? {
            for character in value.chars() {
                let character = self.allocate_string(character.to_string())?;
                self.push_materialized(&mut result, character)?;
            }
        } else if let Some(value) = protocol::bytes_value(&self.state.heap, value)? {
            for byte in value {
                self.push_materialized(&mut result, Value::Int(i64::from(byte)))?;
            }
        } else if let Some(id) = value.object_id() {
            match self.state.heap.get(id)?.clone() {
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
                    while let Some(value) = self.resume_generator(id)? {
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
            }
        } else {
            return Err("object is not iterable".into());
        }
        Ok(result)
    }

    fn push_materialized(&mut self, values: &mut Vec<Value>, value: Value) -> Result<(), String> {
        // A host Vec has allocator/capacity overhead that is not represented in the Python heap.
        // Reserve a deliberately generous per-item amount before every push, including string
        // payloads, so repeated materialization cannot grow outside the memory budget.
        let payload = protocol::string_ref(&self.state.heap, &value)?
            .map_or(0, |text| text.byte_len().saturating_mul(2));
        self.reserve_result(64usize.saturating_add(payload))?;
        self.charge_cpu(1)?;
        values.push(value);
        Ok(())
    }

    fn find_value(&mut self, values: &[Value], needle: &Value) -> Result<Option<usize>, String> {
        for (position, value) in values.iter().enumerate() {
            self.charge_cpu(1)?;
            if protocol::identical(value, needle)
                || protocol::equals(&self.state.heap, value, needle)?
            {
                return Ok(Some(position));
            }
        }
        Ok(None)
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
        let start = self.stack.len() - count;
        Ok(self.stack.split_off(start))
    }

    fn copy(&mut self, depth: usize) -> Result<(), String> {
        if depth == 0 || depth > self.stack.len() {
            return Err("invalid bytecode copy depth".into());
        }
        let value = self.stack[self.stack.len() - depth];
        self.stack.push(value);
        Ok(())
    }

    fn jump_if_or_pop(&mut self, jump_when: bool) -> Result<bool, String> {
        let value = self
            .stack
            .last()
            .cloned()
            .ok_or("invalid bytecode stack effect")?;
        if self.truth_value(&value)? == jump_when {
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
        let next = self
            .transient_memory
            .checked_add(bytes)
            .ok_or("modeled Python memory overflow")?;
        if !self.interp.resources.reserve_memory(bytes) {
            return Err("memory limit exceeded".into());
        }
        self.transient_memory = next;
        Ok(())
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
            Stream::Stdin => {}
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
        if value.inline_string_len().is_some() {
            return Ok(PyKind::String);
        }
        Ok(match value.tag() {
            ValueTag::None => PyKind::None,
            ValueTag::Bool => PyKind::Bool,
            ValueTag::Int => PyKind::Int,
            ValueTag::Float => PyKind::Float,
            ValueTag::Registered => PyKind::Native,
            ValueTag::Native => PyKind::Native,
            ValueTag::Object => match self
                .state
                .heap
                .get(value.object_id().expect("tag checked"))
                .map_err(PyError::runtime_error)?
            {
                Object::String(_) => PyKind::String,
                Object::Bytes(_) => PyKind::Bytes,
                Object::ByteArray(_) => PyKind::ByteArray,
                Object::Exception { .. } => PyKind::Native,
                Object::List(_) => PyKind::List,
                Object::BigInt(_) => PyKind::Int,
                Object::Tuple(_) => PyKind::Tuple,
                Object::Slice { .. } => PyKind::Native,
                Object::Dict(_) | Object::DefaultDict { .. } => PyKind::Dict,
                Object::Set(_) => PyKind::Set,
                Object::Function { .. } | Object::DescriptorBoundMethod { .. } => PyKind::Function,
                Object::Class { .. } => PyKind::Class,
                Object::Instance { .. } | Object::EnumMember { .. } => PyKind::Instance,
                Object::Iterator { .. } | Object::CountIterator { .. } => PyKind::Iterator,
                Object::Generator { .. } => PyKind::Generator,
                Object::Module { .. } => PyKind::Module,
                Object::Array { .. } => PyKind::Array,
                Object::ArrayStorage(_) => PyKind::Native,
                Object::Regex { .. }
                | Object::Match { .. }
                | Object::ArgumentParser { .. }
                | Object::Namespace { .. }
                | Object::RaisesContext { .. } => PyKind::Native,
                Object::Property { .. }
                | Object::StaticMethod { .. }
                | Object::ClassMethod { .. }
                | Object::Super { .. } => PyKind::Native,
            },
            _ => unreachable!("inline strings handled above"),
        })
    }

    fn string_value(&self, value: &Value) -> PyResult<Option<String>> {
        protocol::string_value(&self.state.heap, value).map_err(PyError::runtime_error)
    }

    fn bytes_value(&self, value: &Value) -> PyResult<Option<Vec<u8>>> {
        protocol::bytes_value(&self.state.heap, value).map_err(PyError::runtime_error)
    }

    fn new_string(&mut self, value: String) -> PyResult<Value> {
        self.allocate_string(value).map_err(PyError::resource_error)
    }

    fn new_bytes(&mut self, value: Vec<u8>) -> PyResult<Value> {
        self.allocate_bytes(value).map_err(PyError::resource_error)
    }

    fn new_bytearray(&mut self, value: Vec<u8>) -> PyResult<Value> {
        self.allocate_bytearray(value)
            .map_err(PyError::resource_error)
    }

    fn native_kind(&self, value: &Value) -> PyResult<Option<PyNativeKind>> {
        let Some(id) = value.object_id() else {
            return Ok(None);
        };
        Ok(
            match self.state.heap.get(id).map_err(PyError::runtime_error)? {
                Object::Regex { .. } => Some(PyNativeKind::Regex),
                Object::Match { .. } => Some(PyNativeKind::Match),
                Object::ArgumentParser { .. } => Some(PyNativeKind::ArgumentParser),
                Object::RaisesContext { .. } => Some(PyNativeKind::RaisesContext),
                Object::Property { .. } => Some(PyNativeKind::Property),
                Object::Array { .. } => Some(PyNativeKind::Array),
                _ => None,
            },
        )
    }

    fn identity(&self, value: &Value) -> Option<PyIdentity> {
        value.object_id().map(PyIdentity)
    }

    fn int_value(&self, value: &Value) -> Option<i64> {
        protocol::int_value(&self.state.heap, value)
    }

    fn is_integer_type(&self, value: &Value) -> bool {
        matches!(
            value.native_value(),
            Some(NativeValue::BuiltinType(BuiltinType::Int))
        )
    }

    fn is_string_type(&self, value: &Value) -> bool {
        matches!(
            value.native_value(),
            Some(NativeValue::BuiltinType(BuiltinType::String))
        )
    }

    fn integer_text(&self, value: &Value) -> PyResult<Option<String>> {
        Ok(match super::number::view(&self.state.heap, value) {
            Some(super::number::NumberRef::Int(value)) => Some(value.to_string()),
            Some(super::number::NumberRef::BigInt(value)) => Some(value.to_string()),
            Some(super::number::NumberRef::Float(_)) | None => None,
        })
    }

    fn write_stream(&mut self, stream: &Value, text: &str) -> PyResult<usize> {
        let Some(NativeValue::Stream(stream)) = stream.native_value() else {
            return Err(PyError::type_error("expected a simulated stream"));
        };
        if stream == Stream::Stdin {
            return Err(PyError::value_error("standard input is not writable"));
        }
        self.write_output(stream, text.as_bytes());
        Ok(text.chars().count())
    }

    fn read_stream(&mut self, stream: &Value, size: Option<usize>, line: bool) -> PyResult<String> {
        if stream.native_value() != Some(NativeValue::Stream(Stream::Stdin)) {
            return Err(PyError::value_error("only standard input is readable"));
        }
        if self.stdin_text.is_none() {
            self.charge_cpu(u64::try_from(self.stdin.len()).unwrap_or(u64::MAX))
                .map_err(PyError::runtime_error)?;
            std::str::from_utf8(self.stdin)
                .map_err(|_| PyError::value_error("standard input is not valid UTF-8"))?;
            self.reserve_retained_memory(self.stdin.len())
                .map_err(PyError::resource_error)?;
            self.stdin_text = Some(
                String::from_utf8(self.stdin.to_vec())
                    .expect("stdin was validated as UTF-8 immediately above"),
            );
        }
        let start = self
            .stdin_position
            .min(self.stdin_text.as_ref().map_or(0, String::len));
        let available_len = self
            .stdin_text
            .as_ref()
            .map_or(0, |text| text.len().saturating_sub(start));
        self.charge_cpu(u64::try_from(available_len).unwrap_or(u64::MAX))
            .map_err(PyError::runtime_error)?;
        let length = {
            let available = &self.stdin_text.as_ref().expect("initialized above")[start..];
            let size_end = size.map_or(available.len(), |characters| {
                available
                    .char_indices()
                    .nth(characters)
                    .map_or(available.len(), |(offset, _)| offset)
            });
            let mut length = size_end;
            if line {
                if let Some(newline) = available.as_bytes()[..size_end]
                    .iter()
                    .position(|byte| *byte == b'\n')
                {
                    length = newline + 1;
                }
            }
            length
        };
        self.reserve_memory(length)?;
        let text = self.stdin_text.as_ref().expect("initialized above")
            [start..start.saturating_add(length)]
            .to_string();
        self.stdin_position = self.stdin_position.saturating_add(length);
        Ok(text)
    }

    fn truth(&mut self, value: &Value) -> PyResult<bool> {
        self.truth_value(value).map_err(PyError::runtime_error)
    }

    fn display(&mut self, value: &Value) -> PyResult<String> {
        self.display_value(value).map_err(PyError::runtime_error)
    }

    fn repr(&mut self, value: &Value) -> PyResult<String> {
        self.repr_value(value).map_err(PyError::runtime_error)
    }

    fn equals(&mut self, left: &Value, right: &Value) -> PyResult<bool> {
        protocol::equals(&self.state.heap, left, right).map_err(PyError::runtime_error)
    }

    fn compare(&mut self, left: &Value, right: &Value) -> PyResult<Ordering> {
        self.compare_values(left, right)
            .map_err(PyError::type_error)
    }

    fn get_attribute(&mut self, value: Value, name: &str) -> PyResult<Option<Value>> {
        self.resolve_attribute(value, name)
            .map_err(PyError::runtime_error)
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

    fn bytearray_items(&mut self, value: PyByteArray) -> PyResult<Vec<u8>> {
        let id = value.object_id();
        let length = match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::ByteArray(items) => items.len(),
            _ => {
                return Err(PyError::runtime_error(
                    "bytearray handle changed object kind",
                ))
            }
        };
        self.reserve_memory(length)?;
        match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::ByteArray(items) => Ok(items.clone()),
            _ => Err(PyError::runtime_error(
                "bytearray handle changed object kind",
            )),
        }
    }

    fn replace_bytearray_items(&mut self, value: PyByteArray, items: Vec<u8>) -> PyResult<()> {
        let id = value.object_id();
        if !matches!(
            self.state.heap.get(id).map_err(PyError::runtime_error)?,
            Object::ByteArray(_)
        ) {
            return Err(PyError::runtime_error(
                "bytearray handle changed object kind",
            ));
        }
        self.state
            .heap
            .replace_payload(id, Object::ByteArray(items), &mut self.interp.resources)
            .map_err(PyError::resource_error)
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

    fn slice_parts(&self, value: &Value) -> Option<(Option<i64>, Option<i64>, Option<i64>)> {
        let Object::Slice { start, stop, step } = self.state.heap.get(value.object_id()?).ok()?
        else {
            return None;
        };
        Some((*start, *stop, *step))
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

    fn replace_dict_items(&mut self, dict: PyDict, items: Vec<(Value, Value)>) -> PyResult<()> {
        let id = dict.object_id();
        let replacement = match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::Dict(_) => Object::Dict(items),
            Object::DefaultDict { factory, .. } => Object::DefaultDict {
                factory: *factory,
                entries: items,
            },
            _ => return Err(PyError::runtime_error("dict handle changed object kind")),
        };
        self.state
            .heap
            .replace_payload(id, replacement, &mut self.interp.resources)
            .map_err(PyError::resource_error)
    }

    fn set_items(&mut self, set: PySet) -> PyResult<Vec<Value>> {
        let items = match self
            .state
            .heap
            .get(set.object_id())
            .map_err(PyError::runtime_error)?
        {
            Object::Set(items) => items,
            _ => return Err(PyError::runtime_error("set handle changed object kind")),
        };
        let bytes = items
            .len()
            .checked_mul(std::mem::size_of::<Value>())
            .ok_or_else(|| PyError::resource_error("set snapshot size overflow"))?;
        self.reserve_memory(bytes)?;
        match self
            .state
            .heap
            .get(set.object_id())
            .map_err(PyError::runtime_error)?
        {
            Object::Set(items) => Ok(items.clone()),
            _ => Err(PyError::runtime_error("set handle changed object kind")),
        }
    }

    fn replace_set_items(&mut self, set: PySet, items: Vec<Value>) -> PyResult<()> {
        let id = set.object_id();
        if !matches!(
            self.state.heap.get(id).map_err(PyError::runtime_error)?,
            Object::Set(_)
        ) {
            return Err(PyError::runtime_error("set handle changed object kind"));
        }
        self.state
            .heap
            .replace_payload(id, Object::Set(items), &mut self.interp.resources)
            .map_err(PyError::resource_error)
    }

    fn property_getter(&self, property: PyProperty) -> PyResult<Value> {
        match self
            .state
            .heap
            .get(property.object_id())
            .map_err(PyError::runtime_error)?
        {
            Object::Property { getter, .. } => Ok(*getter),
            _ => Err(PyError::runtime_error(
                "property handle changed object kind",
            )),
        }
    }

    fn new_property(&mut self, getter: Value, setter: Option<Value>) -> PyResult<Value> {
        self.allocate_object(Object::Property { getter, setter })
            .map_err(PyError::resource_error)
    }

    fn new_type(
        &mut self,
        metaclass: Value,
        name: String,
        bases: Value,
        namespace: Value,
    ) -> PyResult<Value> {
        let bases = match bases
            .object_id()
            .and_then(|id| self.state.heap.get(id).ok())
        {
            Some(Object::Tuple(values)) => values.clone(),
            _ => return Err(PyError::type_error("type.__new__() bases must be a tuple")),
        };
        let entries = match namespace
            .object_id()
            .and_then(|id| self.state.heap.get(id).ok())
        {
            Some(Object::Dict(entries)) => entries.clone(),
            _ => {
                return Err(PyError::type_error(
                    "type.__new__() namespace must be a dict",
                ))
            }
        };
        let mut attributes = HashMap::new();
        for (key, value) in entries {
            let key = protocol::string_value(&self.state.heap, &key)
                .map_err(PyError::runtime_error)?
                .ok_or_else(|| PyError::type_error("type.__new__() keys must be strings"))?;
            attributes.insert(key, value);
        }
        let mut user_bases = Vec::new();
        let mut layout = ClassLayout::Object;
        for base in &bases {
            if let Some(id) = base.object_id() {
                let Object::Class {
                    layout: base_layout,
                    ..
                } = self.state.heap.get(id).map_err(PyError::runtime_error)?
                else {
                    return Err(PyError::type_error("type.__new__() bases must be classes"));
                };
                if layout != ClassLayout::Object
                    && *base_layout != ClassLayout::Object
                    && layout != *base_layout
                {
                    return Err(PyError::type_error(
                        "multiple bases have incompatible instance layouts",
                    ));
                }
                if *base_layout != ClassLayout::Object {
                    layout = *base_layout;
                }
                user_bases.push(id);
            } else {
                match base.native_value() {
                    Some(NativeValue::BuiltinType(BuiltinType::Object)) => {}
                    Some(NativeValue::BuiltinType(BuiltinType::Int))
                        if layout == ClassLayout::Object =>
                    {
                        layout = ClassLayout::Int;
                    }
                    Some(NativeValue::BuiltinType(BuiltinType::Type))
                        if layout == ClassLayout::Object =>
                    {
                        layout = ClassLayout::Type;
                    }
                    _ => return Err(PyError::type_error("type.__new__() bases must be classes")),
                }
            }
        }
        let mro = self
            .linearize_bases(&user_bases)
            .map_err(PyError::type_error)?;
        self.allocate_class(ClassDefinition {
            name,
            bases,
            user_bases,
            mro,
            metaclass,
            layout,
            attributes,
            dataclass_fields: Vec::new(),
            enum_members: Vec::new(),
        })
        .map_err(PyError::runtime_error)
    }

    fn instance_attribute(&self, instance: PyInstance, name: &str) -> PyResult<Option<Value>> {
        if !matches!(
            self.state
                .heap
                .get(instance.object_id())
                .map_err(PyError::runtime_error)?,
            Object::Instance { .. }
        ) {
            return Err(PyError::runtime_error(
                "instance handle changed object kind",
            ));
        }
        self.state
            .heap
            .attribute(instance.object_id(), name)
            .map(|value| value.cloned())
            .map_err(PyError::runtime_error)
    }

    fn replace_list_items(&mut self, list: PyList, items: Vec<Value>) -> PyResult<()> {
        let id = list.object_id();
        if !matches!(
            self.state.heap.get(id).map_err(PyError::runtime_error)?,
            Object::List(_)
        ) {
            return Err(PyError::runtime_error("list handle changed object kind"));
        }
        self.state
            .heap
            .replace_payload(id, Object::List(items), &mut self.interp.resources)
            .map_err(PyError::resource_error)
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
            .call(
                argument_count,
                &keyword_names,
                &unpacked,
                CallMode::Immediate,
            )
            .map_err(PyError::runtime_error)?
        {
            CallResult::Value(value) => Ok(value),
            CallResult::Exit(status) => Err(PyError::exit(status)),
            CallResult::EnteredFrame => unreachable!("runtime callback is immediate"),
            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                unreachable!("runtime callback cannot suspend")
            }
        }
    }

    fn is_callable(&self, value: &Value) -> PyResult<bool> {
        Ok(
            if matches!(
                value.native_value(),
                Some(
                    NativeValue::Function(_)
                        | NativeValue::BuiltinType(_)
                        | NativeValue::ValueKind(_)
                        | NativeValue::NativeFunction(_)
                        | NativeValue::ExceptionType(_)
                )
            ) {
                true
            } else if let Some(id) = value.object_id() {
                matches!(
                    self.state.heap.get(id).map_err(PyError::runtime_error)?,
                    Object::Function { .. }
                        | Object::Class { .. }
                        | Object::DescriptorBoundMethod { .. }
                )
            } else {
                false
            },
        )
    }

    fn iterator(&mut self, value: Value) -> PyResult<PyIterator> {
        if let Some(id) = value.object_id() {
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

    fn new_import_path(&mut self) -> PyResult<Value> {
        if let Some(path) = self.state.sys_path {
            return Ok(path);
        }
        let mut values = Vec::with_capacity(self.state.import_paths.len());
        for path in self.state.import_paths.clone() {
            values.push(
                self.allocate_string(path)
                    .map_err(PyError::resource_error)?,
            );
        }
        let path = self.new_list(values)?;
        self.state.sys_path = Some(path);
        Ok(path)
    }

    fn new_tuple(&mut self, items: Vec<Value>) -> PyResult<Value> {
        Vm::allocate_object(self, Object::Tuple(items)).map_err(PyError::resource_error)
    }

    fn new_dict(&mut self, items: Vec<(Value, Value)>) -> PyResult<Value> {
        Vm::allocate_object(self, Object::Dict(items)).map_err(PyError::resource_error)
    }

    fn new_set(&mut self, items: Vec<Value>) -> PyResult<Value> {
        Vm::allocate_object(self, Object::Set(items)).map_err(PyError::resource_error)
    }

    fn new_value_kind(
        &self,
        kind: &'static super::native::ValueKindDef,
        payload: u64,
    ) -> PyResult<Value> {
        let index = self
            .state
            .types
            .value_kind_index(kind)
            .ok_or_else(|| PyError::runtime_error("value kind is not registered"))?;
        Ok(Value::registered(index, payload))
    }

    fn value_kind_payload(
        &self,
        value: &Value,
        kind: &'static super::native::ValueKindDef,
    ) -> Option<u64> {
        let (index, payload) = value.registered_parts()?;
        std::ptr::eq(self.state.types.value_kind(index)?, kind).then_some(payload)
    }

    fn value_kind_type(&self, kind: &'static super::native::ValueKindDef) -> PyResult<Value> {
        self.state
            .types
            .value_kind_type_id(kind)
            .ok_or_else(|| PyError::runtime_error("value kind is not registered"))
            .and_then(|type_id| {
                self.state
                    .types
                    .value(type_id)
                    .map_err(PyError::runtime_error)
            })
    }

    fn new_array(
        &mut self,
        items: Vec<Value>,
        shape: Vec<usize>,
        dtype: PyArrayDtype,
    ) -> PyResult<Value> {
        let count = shape
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
            .ok_or_else(|| PyError::value_error("array is too large"))?;
        if count != items.len() {
            return Err(PyError::runtime_error(
                "array storage does not match its shape",
            ));
        }
        let strides = super::stdlib::numpy::contiguous_strides(&shape)?;
        let storage = Vm::allocate_object(self, Object::ArrayStorage(items))
            .map_err(PyError::resource_error)?
            .object_id()
            .expect("allocated storage is an object");
        Vm::allocate_object(
            self,
            Object::Array {
                storage,
                layout: PyArrayLayout {
                    shape,
                    strides,
                    offset: 0,
                },
                dtype,
            },
        )
        .map_err(PyError::resource_error)
    }

    fn new_array_view(&mut self, array: PyArray, layout: PyArrayLayout) -> PyResult<Value> {
        if layout.shape.len() != layout.strides.len() {
            return Err(PyError::runtime_error(
                "array shape and strides have different ranks",
            ));
        }
        let (storage, dtype) = match self
            .state
            .heap
            .get(array.object_id())
            .map_err(PyError::runtime_error)?
        {
            Object::Array { storage, dtype, .. } => (*storage, *dtype),
            _ => return Err(PyError::runtime_error("array handle changed object kind")),
        };
        let storage_len = match self
            .state
            .heap
            .get(storage)
            .map_err(PyError::runtime_error)?
        {
            Object::ArrayStorage(values) => values.len(),
            _ => return Err(PyError::runtime_error("array storage changed object kind")),
        };
        validate_array_layout(&layout, storage_len)?;
        Vm::allocate_object(
            self,
            Object::Array {
                storage,
                layout,
                dtype,
            },
        )
        .map_err(PyError::resource_error)
    }

    fn array_layout(&self, array: PyArray) -> PyResult<(PyArrayLayout, PyArrayDtype)> {
        match self
            .state
            .heap
            .get(array.object_id())
            .map_err(PyError::runtime_error)?
        {
            Object::Array { layout, dtype, .. } => Ok((layout.clone(), *dtype)),
            _ => Err(PyError::runtime_error("array handle changed object kind")),
        }
    }

    fn array_get(&mut self, array: PyArray, index: &[usize]) -> PyResult<Value> {
        self.charge_cpu(1).map_err(PyError::resource_error)?;
        let (storage, layout) = match self
            .state
            .heap
            .get(array.object_id())
            .map_err(PyError::runtime_error)?
        {
            Object::Array {
                storage, layout, ..
            } => (*storage, layout.clone()),
            _ => return Err(PyError::runtime_error("array handle changed object kind")),
        };
        let offset = array_offset(&layout, index)?;
        match self
            .state
            .heap
            .get(storage)
            .map_err(PyError::runtime_error)?
        {
            Object::ArrayStorage(values) => values
                .get(offset)
                .copied()
                .ok_or_else(|| PyError::runtime_error("array offset is outside storage")),
            _ => Err(PyError::runtime_error("array storage changed object kind")),
        }
    }

    fn array_set(&mut self, array: PyArray, index: &[usize], value: Value) -> PyResult<()> {
        self.charge_cpu(1).map_err(PyError::resource_error)?;
        let (storage, layout) = match self
            .state
            .heap
            .get(array.object_id())
            .map_err(PyError::runtime_error)?
        {
            Object::Array {
                storage, layout, ..
            } => (*storage, layout.clone()),
            _ => return Err(PyError::runtime_error("array handle changed object kind")),
        };
        let offset = array_offset(&layout, index)?;
        match self
            .state
            .heap
            .get_mut(storage)
            .map_err(PyError::runtime_error)?
        {
            Object::ArrayStorage(values) => {
                let destination = values
                    .get_mut(offset)
                    .ok_or_else(|| PyError::runtime_error("array offset is outside storage"))?;
                *destination = value;
                Ok(())
            }
            _ => Err(PyError::runtime_error("array storage changed object kind")),
        }
    }

    fn binary_op(&mut self, operation: PyBinaryOp, left: Value, right: Value) -> PyResult<Value> {
        let operation = match operation {
            PyBinaryOp::Add => BinaryOperator::Add,
            PyBinaryOp::Subtract => BinaryOperator::Subtract,
            PyBinaryOp::Multiply => BinaryOperator::Multiply,
            PyBinaryOp::Divide => BinaryOperator::Divide,
        };
        self.binary_value(operation, left, right)
            .map_err(|message| {
                if self.pending_exception.is_some() {
                    PyError::new(PyErrorKind::Raised, message)
                } else {
                    PyError::type_error(message)
                }
            })
    }

    fn new_integer(&mut self, decimal: &str) -> PyResult<Value> {
        self.charge_cpu(u64::try_from(decimal.len()).unwrap_or(u64::MAX))
            .map_err(PyError::resource_error)?;
        self.reserve_result(decimal.len().saturating_mul(2))
            .map_err(PyError::resource_error)?;
        let value = decimal
            .parse::<BigInt>()
            .map_err(|_| PyError::value_error("invalid integer"))?;
        if let Some(value) = value.to_i64() {
            Ok(Value::Int(value))
        } else {
            Vm::allocate_object(self, Object::BigInt(value)).map_err(PyError::resource_error)
        }
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
            PyMarker::Stdin => NativeValue::Stream(Stream::Stdin),
            PyMarker::Stdout => NativeValue::Stream(Stream::Stdout),
            PyMarker::Stderr => NativeValue::Stream(Stream::Stderr),
            PyMarker::ArrayType => NativeValue::BuiltinType(BuiltinType::Array),
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
        let mut values = Vec::with_capacity(self.argv.len());
        for argument in self.argv {
            values.push(self.new_string(argument.clone())?);
        }
        self.new_list(values)
    }

    fn new_argument_parser(
        &mut self,
        program: String,
        description: Option<String>,
        add_help: bool,
        is_subcommand: bool,
    ) -> PyResult<Value> {
        Vm::allocate_object(
            self,
            Object::ArgumentParser {
                prog: program,
                description,
                add_help,
                is_subcommand,
                arguments: Vec::new(),
                subparsers: None,
            },
        )
        .map_err(PyError::resource_error)
    }

    fn argument_parser_parts(
        &mut self,
        parser: PyArgumentParser,
    ) -> PyResult<PyArgumentParserData> {
        let Object::ArgumentParser {
            prog,
            description,
            add_help,
            arguments,
            subparsers,
            ..
        } = self
            .state
            .heap
            .get(parser.object_id())
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("parser handle changed object kind"));
        };
        let bytes = prog
            .len()
            .saturating_add(description.as_ref().map_or(0, String::len))
            .saturating_add(arguments.len().saturating_mul(128))
            .saturating_add(
                subparsers
                    .as_ref()
                    .map_or(0, |value| value.commands.len().saturating_mul(96)),
            );
        let result = PyArgumentParserData {
            prog: prog.clone(),
            description: description.clone(),
            add_help: *add_help,
            arguments: arguments.clone(),
            subparsers: subparsers.clone(),
        };
        self.reserve_memory(bytes)?;
        Ok(result)
    }

    fn append_argument(
        &mut self,
        parser: PyArgumentParser,
        argument: PyArgumentSpec,
    ) -> PyResult<()> {
        self.state
            .heap
            .reserve_object_growth(parser.object_id(), 96, &mut self.interp.resources)
            .map_err(PyError::resource_error)?;
        let Object::ArgumentParser { arguments, .. } = self
            .state
            .heap
            .get_mut(parser.object_id())
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("parser handle changed object kind"));
        };
        arguments.push(argument);
        Ok(())
    }

    fn configure_subparsers(
        &mut self,
        parser: PyArgumentParser,
        subparsers: PySubparsersSpec,
    ) -> PyResult<()> {
        let Object::ArgumentParser {
            is_subcommand,
            subparsers: current,
            ..
        } = self
            .state
            .heap
            .get_mut(parser.object_id())
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("parser handle changed object kind"));
        };
        if *is_subcommand {
            return Err(PyError::value_error(
                "nested argparse subparsers are not supported",
            ));
        }
        if current.is_some() {
            return Err(PyError::value_error("parser already has subparsers"));
        }
        *current = Some(subparsers);
        Ok(())
    }

    fn append_subcommand(
        &mut self,
        parser: PyArgumentParser,
        command: PySubcommandSpec,
    ) -> PyResult<()> {
        self.state
            .heap
            .reserve_object_growth(parser.object_id(), 96, &mut self.interp.resources)
            .map_err(PyError::resource_error)?;
        let Object::ArgumentParser { subparsers, .. } = self
            .state
            .heap
            .get_mut(parser.object_id())
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("parser handle changed object kind"));
        };
        let subparsers = subparsers
            .as_mut()
            .ok_or_else(|| PyError::value_error("add_subparsers() must be called first"))?;
        if subparsers
            .commands
            .iter()
            .any(|candidate| candidate.name == command.name)
        {
            return Err(PyError::value_error(format!(
                "conflicting subparser: {}",
                command.name
            )));
        }
        subparsers.commands.push(command);
        Ok(())
    }

    fn command_arguments(&self) -> Vec<String> {
        self.argv.iter().skip(1).cloned().collect()
    }

    fn new_module(
        &mut self,
        name: String,
        path: String,
        spec: Value,
        loader: Value,
    ) -> PyResult<Value> {
        let module_name = self
            .allocate_string(name.clone())
            .map_err(PyError::resource_error)?;
        let module_path = self
            .allocate_string(path)
            .map_err(PyError::resource_error)?;
        let package = name
            .rsplit_once('.')
            .map_or("", |(package, _)| package)
            .to_string();
        let package = self
            .allocate_string(package)
            .map_err(PyError::resource_error)?;
        let scope = self
            .state
            .heap
            .allocate_scope(
                None,
                false,
                HashMap::from([
                    ("__name__".into(), module_name),
                    ("__file__".into(), module_path),
                    ("__package__".into(), package),
                    ("__spec__".into(), spec),
                    ("__loader__".into(), loader),
                ]),
                &mut self.interp.resources,
            )
            .map_err(PyError::resource_error)?;
        self.allocate_object(Object::Module { name, scope })
            .map_err(PyError::resource_error)
    }

    fn exec_module(&mut self, module: PyModule, path: &str) -> PyResult<()> {
        let source = self.interp.read_text(path)?;
        let parse_memory = u64::try_from(source.len())
            .ok()
            .and_then(|bytes| bytes.checked_mul(4))
            .ok_or_else(|| PyError::resource_error("module source is too large"))?;
        if !self.interp.resources.reserve_memory(parse_memory)
            || !self.interp.resources.charge_cpu(source.len() as u64)
        {
            return Err(PyError::resource_error(
                "resource limit exceeded while loading module",
            ));
        }
        let tokens = super::lexer::lex(&source).map_err(|error| {
            PyError::runtime_error(format!(
                "{} in {path} at line {}, column {}",
                error.message, error.span.line, error.span.column
            ))
        })?;
        let program = super::parser::parse(tokens).map_err(|error| {
            PyError::runtime_error(format!(
                "{} in {path} at line {}, column {}",
                error.message, error.span.line, error.span.column
            ))
        })?;
        let code = super::compiler::compile(program);
        let Object::Module { scope, .. } = self
            .state
            .heap
            .get(module.object_id())
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("module handle changed object kind"));
        };
        let scope = *scope;
        let outer_stack = std::mem::take(&mut self.stack);
        let import_root = path
            .rsplit_once('/')
            .map_or_else(|| "/".to_string(), |(parent, _)| parent.to_string());
        self.state.temporary_import_paths.insert(0, import_root);
        self.local_scopes.push(scope);
        let execution = self.execute_code(&code);
        self.local_scopes.pop();
        self.state.temporary_import_paths.remove(0);
        self.stack = outer_stack;
        match execution {
            Ok(Execution::Pending) => unreachable!("execute_code drains pending quanta"),
            Ok(Execution::Blocked(_)) => unreachable!("immediate code cannot suspend"),
            Ok(Execution::Halt) => Ok(()),
            Ok(Execution::Return(_)) => Err(PyError::runtime_error(format!(
                "'return' outside function in module loaded from {path:?}"
            ))),
            Ok(Execution::Yield(_, _)) => Err(PyError::runtime_error(format!(
                "'yield' outside function in module loaded from {path:?}"
            ))),
            Ok(Execution::Exit(status)) => Err(PyError::exit(status)),
            Err((error, span)) => Err(PyError::runtime_error(format!(
                "{error} in {path} at line {}, column {}",
                span.line, span.column
            ))),
        }
    }

    fn new_namespace(&mut self, values: Vec<(String, Value)>) -> PyResult<Value> {
        Vm::allocate_object(self, Object::Namespace { values }).map_err(PyError::resource_error)
    }

    fn new_raises_context(&mut self, expected: String) -> PyResult<Value> {
        Vm::allocate_object(self, Object::RaisesContext { expected })
            .map_err(PyError::resource_error)
    }

    fn raises_expected(&self, context: PyRaisesContext) -> PyResult<String> {
        let Object::RaisesContext { expected } = self
            .state
            .heap
            .get(context.object_id())
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("raises handle changed object kind"));
        };
        Ok(expected.clone())
    }

    fn exception_type_name(&self, value: &Value) -> Option<&'static str> {
        match value.native_value() {
            Some(NativeValue::ExceptionType(ExceptionType(name))) => Some(name),
            _ => None,
        }
    }

    fn exception_type(&self, name: &'static str) -> Value {
        Value::Native(NativeValue::ExceptionType(ExceptionType(name)))
    }

    fn clock(&mut self) -> &mut dyn PyClock {
        self
    }

    fn environment(&self) -> &dyn PyEnvironment {
        self
    }

    fn filesystem(&mut self) -> &mut dyn PyFilesystem {
        self.interp
    }

    fn processes(&mut self) -> &mut dyn PyProcessRunner {
        self
    }
}

impl PyProcessRunner for Vm<'_> {
    fn start(&mut self, request: PyProcessStartRequest) -> PyResult<PyProcessHandle> {
        super::process::start(self.interp, request)
    }

    fn poll(&mut self, handle: PyProcessHandle) -> PyResult<Option<i32>> {
        super::process::poll(self.interp, handle)
    }

    fn wait(
        &mut self,
        handle: PyProcessHandle,
        timeout_ns: Option<u64>,
    ) -> PyResult<PyProcessOutput> {
        let mut output = if self.mode.scheduler_owned && self.native_suspend_allowed {
            match super::process::wait_if_ready(self.interp, handle, timeout_ns)? {
                Ok(output) => output,
                Err(reason) => {
                    return Err(PyError::suspend(reason));
                }
            }
        } else {
            super::process::wait(self.interp, handle, timeout_ns)?
        };
        self.out
            .extend_from_slice(&std::mem::take(&mut output.inherited_stdout));
        self.err
            .extend_from_slice(&std::mem::take(&mut output.inherited_stderr));
        Ok(output)
    }

    fn communicate(
        &mut self,
        handle: PyProcessHandle,
        input: Vec<u8>,
        timeout_ns: Option<u64>,
    ) -> PyResult<PyProcessOutput> {
        let mut output = if self.mode.scheduler_owned && self.native_suspend_allowed {
            match super::process::communicate_if_ready(self.interp, handle, input, timeout_ns)? {
                Ok(output) => output,
                Err(reason) => {
                    return Err(PyError::suspend(reason));
                }
            }
        } else {
            super::process::communicate(self.interp, handle, input, timeout_ns)?
        };
        self.out
            .extend_from_slice(&std::mem::take(&mut output.inherited_stdout));
        self.err
            .extend_from_slice(&std::mem::take(&mut output.inherited_stderr));
        Ok(output)
    }

    fn read_pipe(
        &mut self,
        handle: PyProcessHandle,
        fd: i32,
        amount: Option<usize>,
    ) -> PyResult<Vec<u8>> {
        if self.mode.scheduler_owned && self.native_suspend_allowed {
            return match super::process::read_pipe_if_ready(self.interp, handle, fd, amount)? {
                Ok(bytes) => Ok(bytes),
                Err(reason) => Err(PyError::suspend(reason)),
            };
        }
        super::process::read_pipe(self.interp, handle, fd, amount)
    }

    fn write_pipe(&mut self, handle: PyProcessHandle, input: Vec<u8>) -> PyResult<usize> {
        if self.mode.scheduler_owned && self.native_suspend_allowed {
            return match super::process::write_pipe_if_ready(self.interp, handle, input)? {
                Ok(written) => Ok(written),
                Err(reason) => Err(PyError::suspend(reason)),
            };
        }
        super::process::write_pipe(self.interp, handle, input)
    }

    fn close_pipe(&mut self, handle: PyProcessHandle, fd: i32) -> PyResult<()> {
        super::process::close_pipe(self.interp, handle, fd)
    }

    fn send_signal(
        &mut self,
        handle: PyProcessHandle,
        signal: crate::process::Signal,
    ) -> PyResult<()> {
        super::process::send_signal(self.interp, handle, signal)
    }
}

fn array_offset(layout: &PyArrayLayout, index: &[usize]) -> PyResult<usize> {
    if index.len() != layout.shape.len() {
        return Err(PyError::value_error("array index has the wrong rank"));
    }
    let mut offset = layout.offset;
    for (axis, selected) in index.iter().enumerate() {
        if *selected >= layout.shape[axis] {
            return Err(PyError::value_error("array index is out of bounds"));
        }
        let selected = isize::try_from(*selected)
            .map_err(|_| PyError::value_error("array offset overflow"))?;
        offset = offset
            .checked_add(
                layout.strides[axis]
                    .checked_mul(selected)
                    .ok_or_else(|| PyError::value_error("array offset overflow"))?,
            )
            .ok_or_else(|| PyError::value_error("array offset overflow"))?;
    }
    usize::try_from(offset).map_err(|_| PyError::runtime_error("array offset is negative"))
}

fn validate_array_layout(layout: &PyArrayLayout, storage_len: usize) -> PyResult<()> {
    super::stdlib::numpy::validate_rank(&layout.shape)?;
    if layout.shape.contains(&0) {
        return Ok(());
    }
    let mut minimum = layout.offset;
    let mut maximum = layout.offset;
    for (length, stride) in layout.shape.iter().zip(&layout.strides) {
        let span = isize::try_from(length.saturating_sub(1))
            .ok()
            .and_then(|length| stride.checked_mul(length))
            .ok_or_else(|| PyError::value_error("array offset overflow"))?;
        if span < 0 {
            minimum = minimum
                .checked_add(span)
                .ok_or_else(|| PyError::value_error("array offset overflow"))?;
        } else {
            maximum = maximum
                .checked_add(span)
                .ok_or_else(|| PyError::value_error("array offset overflow"))?;
        }
    }
    if minimum < 0 || usize::try_from(maximum).map_or(true, |value| value >= storage_len) {
        return Err(PyError::runtime_error("array view is outside storage"));
    }
    Ok(())
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
        if self.mode.scheduler_owned && self.native_suspend_allowed {
            let event = self
                .interp
                .clock
                .schedule_wake_after(u64::from(self.interp.process.pid), nanos as u64)
                .map_err(|error| PyError::runtime_error(error.to_string()))?;
            self.pending_wait = Some(crate::scheduler::WaitReason::Timer(event.deadline_ns()));
            return Ok(());
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
    EnteredFrame,
    Blocked(crate::scheduler::WaitReason, Value),
    Retry(crate::scheduler::WaitReason, PendingNativeCall),
}

#[derive(Clone, Copy)]
enum CallMode {
    Immediate,
    Deferred(super::source::Span),
}

enum Execution {
    Pending,
    Blocked(crate::scheduler::WaitReason),
    Halt,
    Return(Value),
    Yield(Value, usize),
    Exit(i32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SequenceKind {
    List,
    Tuple,
}

/// Fully resolved inputs to the single class allocator shared by class statements and
/// `type.__new__`.
struct ClassDefinition {
    name: String,
    bases: Vec<Value>,
    user_bases: Vec<super::heap::ObjectId>,
    mro: Vec<super::heap::ObjectId>,
    metaclass: Value,
    layout: ClassLayout,
    attributes: HashMap<String, Value>,
    dataclass_fields: Vec<(String, Option<Value>)>,
    enum_members: Vec<Value>,
}

fn format_float(value: f64, spec: &str) -> Result<String, String> {
    let presentation = spec.chars().last().ok_or("empty float format")?;
    let options = &spec[..spec.len() - presentation.len_utf8()];
    let (width, precision, zero_pad) = parse_numeric_format(options)?;
    let precision = precision.unwrap_or(6);
    let rendered = match presentation {
        'f' => format!("{value:.precision$}"),
        'e' => format!("{value:.precision$e}"),
        'E' => format!("{value:.precision$e}").to_uppercase(),
        _ => return Err(format!("unsupported floating-point format {spec:?}")),
    };
    Ok(pad_rendered_number(rendered, width, zero_pad))
}

fn pad_number(value: String, options: &str) -> Result<String, String> {
    let (width, precision, zero_pad) = parse_numeric_format(options)?;
    if precision.is_some() {
        return Err("precision is not allowed in integer format".into());
    }
    Ok(pad_rendered_number(value, width, zero_pad))
}

fn parse_numeric_format(options: &str) -> Result<(usize, Option<usize>, bool), String> {
    let zero_pad = options.starts_with('0') && options.len() > 1;
    let (width, precision) = options
        .split_once('.')
        .map_or((options, None), |(width, precision)| {
            (width, Some(precision))
        });
    let width = if width.is_empty() {
        0
    } else {
        width
            .parse::<usize>()
            .map_err(|_| format!("unsupported numeric format {options:?}"))?
    };
    let precision = precision
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("unsupported numeric format {options:?}"))
        })
        .transpose()?;
    Ok((width, precision, zero_pad))
}

fn pad_rendered_number(value: String, width: usize, zero_pad: bool) -> String {
    let padding = width.saturating_sub(value.len());
    if padding == 0 {
        return value;
    }
    let fill = if zero_pad { '0' } else { ' ' };
    if zero_pad && value.starts_with('-') {
        format!("-{}{}", fill.to_string().repeat(padding), &value[1..])
    } else {
        format!("{}{}", fill.to_string().repeat(padding), value)
    }
}

fn format_text(value: &str, spec: &str) -> Result<String, String> {
    let (alignment, width_text) = match spec.chars().next() {
        Some(alignment @ ('<' | '>' | '^')) => (alignment, &spec[alignment.len_utf8()..]),
        _ => ('<', spec),
    };
    let width = width_text
        .parse::<usize>()
        .map_err(|_| format!("unsupported string format {spec:?}"))?;
    let padding = width.saturating_sub(value.chars().count());
    let left = match alignment {
        '>' => padding,
        '^' => padding / 2,
        _ => 0,
    };
    let right = padding - left;
    Ok(format!(
        "{}{}{}",
        " ".repeat(left),
        value,
        " ".repeat(right)
    ))
}

fn slice_indices(
    length: usize,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> Result<Vec<usize>, String> {
    let length = i64::try_from(length).map_err(|_| "sequence is too large to slice")?;
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err("slice step cannot be zero".into());
    }
    let normalize = |value: i64, minimum: i64, maximum: i64| {
        let value = if value < 0 {
            value.saturating_add(length)
        } else {
            value
        };
        value.clamp(minimum, maximum)
    };
    let (mut current, stop) = if step > 0 {
        (
            start.map_or(0, |value| normalize(value, 0, length)),
            stop.map_or(length, |value| normalize(value, 0, length)),
        )
    } else {
        (
            start.map_or(length - 1, |value| normalize(value, -1, length - 1)),
            stop.map_or(-1, |value| normalize(value, -1, length - 1)),
        )
    };
    let mut indices = Vec::new();
    while if step > 0 {
        current < stop
    } else {
        current > stop
    } {
        indices.push(usize::try_from(current).map_err(|_| "slice index out of range")?);
        current = current
            .checked_add(step)
            .ok_or("slice index arithmetic overflow")?;
    }
    Ok(indices)
}

fn select_string_slice(
    value: &str,
    is_ascii: bool,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> Result<(String, u64), String> {
    debug_assert_eq!(value.is_ascii(), is_ascii);
    let characters = (!is_ascii).then(|| value.chars().collect::<Vec<_>>());
    let length = characters.as_ref().map_or(value.len(), Vec::len);
    let indices = slice_indices(length, start, stop, step)?;
    let units = u64::try_from(indices.len()).unwrap_or(u64::MAX);
    let selected = if is_ascii && step.unwrap_or(1) == 1 {
        match (indices.first(), indices.last()) {
            (Some(first), Some(last)) => value
                .get(*first..last.saturating_add(1))
                .expect("ASCII slice indices are byte boundaries")
                .to_owned(),
            _ => String::new(),
        }
    } else if is_ascii {
        indices
            .into_iter()
            .map(|index| char::from(value.as_bytes()[index]))
            .collect()
    } else {
        let characters = characters.expect("non-ASCII slices materialize code points");
        indices.into_iter().map(|index| characters[index]).collect()
    };
    Ok((selected, units))
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
