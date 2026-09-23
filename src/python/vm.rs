//! Metered Python VM façade and persistent execution state.
//!
//! Child modules partition bytecode dispatch, calls, object protocols, iteration, native-module
//! integration, and host adapters. They extend the single [`Vm`] type rather than introducing
//! subsystem traits or independent state owners; resumable state remains centralized here.

use crate::interp::Interp;

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;

use super::ast::{BinaryOperator, ComparisonOperator, Constant, UnaryOperator};
use super::bytecode::{CallId, ClassField, CodeRef, NameId, Opcode};
use super::filesystem::PyModuleLoader;
use super::heap::{
    ClassLayout, InstanceAttributeSlot, InstanceAttributes, InstancePayload, Object, ObjectId,
    ScopeId, SymbolId, MODELED_MAPPING_ENTRY_BYTES, MODELED_VALUE_BYTES,
};
use super::native::{
    CallArgs, FunctionDef, ModuleDef, PyArgumentParser, PyArgumentParserData, PyArgumentSpec,
    PyArray, PyArrayDtype, PyArrayLayout, PyBinaryOp, PyByteArray, PyCallable, PyClass, PyClock,
    PyDict, PyEnvironment, PyError, PyErrorKind, PyFilesystem, PyHttpClient, PyIdentity,
    PyIterator, PyKind, PyList, PyMarker, PyMatch, PyMatchData, PyModule, PyNativeKind,
    PyProcessHandle, PyProcessOutput, PyProcessPoll, PyProcessRunner, PyProcessStartRequest,
    PyProperty, PyRaisesContext, PyRegex, PyResult, PyRuntime, PySet, PySubcommandSpec,
    PySubparsersSpec, PyTuple, PyValueCast,
};
use super::number;
use super::object_model::{BuiltinType, Slot, SlotValue, TypeId};
use super::slice::SlicePlan;
use super::{protocol, ExecResult, Out, ReplState, Value, ValueTag};

mod calls;
mod dispatch;
mod host;
mod iteration;
mod namespace;
mod native_runtime;
mod objects;
mod operations;
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

const EXCEPTION_TYPES: [&str; 25] = [
    "Exception",
    "BaseException",
    "AssertionError",
    "TypeError",
    "ValueError",
    "RuntimeError",
    "ZeroDivisionError",
    "OverflowError",
    "KeyError",
    "IndexError",
    "StopIteration",
    "Skipped",
    "Failed",
    "CalledProcessError",
    "TimeoutExpired",
    "EOFError",
    "OSError",
    "FileNotFoundError",
    "FileExistsError",
    "IsADirectoryError",
    "NotADirectoryError",
    "PermissionError",
    "SystemExit",
    "TimeoutError",
    "StopAsyncIteration",
];

fn known_exception_type(name: &str) -> Option<&'static str> {
    EXCEPTION_TYPES
        .iter()
        .copied()
        .find(|candidate| *candidate == name)
}

fn exception_type_code(name: &str) -> u64 {
    EXCEPTION_TYPES
        .iter()
        .position(|candidate| *candidate == name)
        .and_then(|index| u64::try_from(index).ok())
        .expect("exception type must come from the closed builtin table")
}

fn exception_type_name(code: u64) -> &'static str {
    usize::try_from(code)
        .ok()
        .and_then(|index| EXCEPTION_TYPES.get(index))
        .copied()
        .expect("invalid private exception-type handle")
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
    Character,
    Ordinal,
    Binary,
    Octal,
    Hexadecimal,
    Repr,
    Dir,
    IsInstance,
    IsSubclass,
    Length,
    Sorted,
    Minimum,
    Maximum,
    Sum,
    Absolute,
    Power,
    Divmod,
    Callable,
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
    let name_symbol = match state.heap.intern_symbol("__name__", &mut interp.resources) {
        Ok(symbol) => symbol,
        Err(_) => return ExecResult::Exit(137),
    };
    if state.globals.get(name_symbol).is_none()
        && state
            .globals
            .insert(
                name_symbol,
                Value::inline_string("__main__").expect("short builtin string"),
                &mut interp.resources,
            )
            .is_err()
    {
        return ExecResult::Exit(137);
    }
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
    code: CodeRef,
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
                stack_base: 0,
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
            Ok(Execution::Blocked(reason)) => return VmPoll::Blocked(reason.clone()),
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
    async_timer_deadlines: BTreeSet<u64>,
    native_suspend_allowed: bool,
    exception_stack: Vec<RaisedException>,
    with_contexts: Vec<Value>,
    method_frames: Vec<(super::heap::ObjectId, Value)>,
    code_caches: Vec<CodeCaches>,
    stdin_position: usize,
    stdin_text: Option<String>,
    transient_memory: u64,
    retained_memory: u64,
}

#[derive(Clone)]
struct CodeCaches {
    code: CodeRef,
    names: Vec<Option<SymbolId>>,
    attributes: Option<Vec<Option<LoadAttributeCache>>>,
}

#[derive(Clone, Copy)]
struct LoadAttributeCache {
    class: ObjectId,
    location: InstanceAttributeSlot,
}

/// An executing code object's resumable control state.
#[derive(Clone)]
struct BytecodeFrame {
    code: CodeRef,
    instruction_pointer: usize,
    /// First operand owned by this frame in the VM's shared value stack.
    stack_base: usize,
    handlers: Vec<(usize, usize)>,
    function_return: Option<FunctionReturn>,
    pending_native_call: Option<PendingNativeCall>,
}

/// Hot dispatch state retained in registers for one scheduler quantum.
///
/// The resumable frame remains the source of truth at suspension and frame boundaries. Between
/// those boundaries the cursor avoids rediscovering the active frame and rewriting its instruction
/// pointer after every opcode.
struct DispatchCursor {
    code: CodeRef,
    code_cache: usize,
    op_index: usize,
}

/// Control-flow effect of one successfully decoded opcode.
enum DispatchControl {
    Next,
    Jump(usize),
    RefreshFrame,
    Complete(Execution),
}

/// Result of classifying one arena-backed iterator at its mutation boundary.
enum IteratorAdvance {
    Yield(Value),
    Exhausted,
    Callable { callable: Value, sentinel: Value },
    Generator,
    Invalid,
}

enum BuiltinSubscript {
    Value(Value),
    Mapping { factory: Option<Value> },
    Set,
    Unsupported,
}

#[inline(always)]
fn dispatch_next(result: Result<(), String>) -> Result<DispatchControl, String> {
    result.map(|()| DispatchControl::Next)
}

impl DispatchCursor {
    fn for_active(vm: &mut Vm<'_>) -> Result<Self, String> {
        let frame = vm
            .bytecode_frames
            .last()
            .expect("bytecode execution requires an active frame");
        let code = frame.code.clone();
        let op_index = frame.instruction_pointer;
        let code_cache = vm.ensure_code_cache(&code)?;
        Ok(Self {
            code,
            code_cache,
            op_index,
        })
    }

    fn refresh(&mut self, vm: &mut Vm<'_>) -> Result<(), String> {
        *self = Self::for_active(vm)?;
        Ok(())
    }

    fn span(&self) -> super::source::Span {
        self.code.spans[self.op_index]
    }

    fn sync(&self, vm: &mut Vm<'_>) {
        let frame = vm
            .bytecode_frames
            .last_mut()
            .expect("bytecode execution requires an active frame");
        debug_assert!(Arc::ptr_eq(&self.code, &frame.code));
        frame.instruction_pointer = self.op_index;
    }
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
        let mut values = self.state.globals.values().collect::<Vec<_>>();
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
        if self.transient_memory == 0 {
            return;
        }
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
                    let rendered = protocol::display(&self.state.heap, &exception.value)
                        .unwrap_or_else(|_| "SystemExit".to_string());
                    let status = rendered.parse::<i32>().unwrap_or_else(|_| {
                        self.err.extend_from_slice(rendered.as_bytes());
                        self.err.push(b'\n');
                        1
                    });
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

    fn allocate_object(&mut self, object: Object) -> Result<Value, String> {
        self.state.heap.allocate(object, &mut self.interp.resources)
    }

    fn allocate_string(&mut self, value: String) -> Result<Value, String> {
        if let Some(value) = Value::inline_string(&value) {
            Ok(value)
        } else {
            self.allocate_object(Object::String(value.into()))
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
        let count = range_length(start, stop, step)?;
        let start = i128::from(start);
        let step = i128::from(step);
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
                Object::List(values)
                | Object::Tuple(values)
                | Object::Set(values)
                | Object::FrozenSet(values) => {
                    for value in values {
                        self.push_materialized(&mut result, value)?;
                    }
                }
                Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                    for (key, _) in entries {
                        self.push_materialized(&mut result, key)?;
                    }
                }
                Object::Range { start, stop, step } => {
                    for value in self.range_values(start, stop, step)? {
                        self.push_materialized(&mut result, value)?;
                    }
                }
                Object::Iterator { .. }
                | Object::SequenceIterator { .. }
                | Object::RangeIterator { .. } => {
                    while let Some(value) = self.next_stored_iterator(id)? {
                        self.push_materialized(&mut result, value)?;
                    }
                }
                Object::CountIterator { .. } => {
                    return Err(
                        "cannot materialize infinite itertools.count without a bound".into(),
                    )
                }
                Object::CallableIterator {
                    callable,
                    sentinel,
                    exhausted,
                } => {
                    if !exhausted {
                        loop {
                            self.charge_cpu(1)?;
                            let item = <Self as PyRuntime>::call_value(
                                self,
                                callable,
                                CallArgs::new(Vec::new(), Vec::new()),
                            )
                            .map_err(|error| error.to_string())?;
                            if protocol::equals(&self.state.heap, &item, &sentinel)? {
                                if let Object::CallableIterator { exhausted, .. } =
                                    self.state.heap.get_mut(id)?
                                {
                                    *exhausted = true;
                                }
                                break;
                            }
                            self.push_materialized(&mut result, item)?;
                        }
                    }
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

    fn find_mapping_entry(
        &mut self,
        id: ObjectId,
        needle: &Value,
    ) -> Result<Option<usize>, String> {
        let heap = &self.state.heap;
        let entries = match heap.get(id)? {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
            _ => return Err("dict handle changed object kind".into()),
        };
        for position in entries.candidate_positions(needle) {
            let candidate = entries[position].0;
            if !self.interp.resources.charge_cpu(1) {
                return Err("resource limit exceeded while executing Python".into());
            }
            if protocol::identical(&candidate, needle)
                || protocol::equals(heap, &candidate, needle)?
            {
                return Ok(Some(position));
            }
        }
        Ok(None)
    }

    fn find_set_entry(&mut self, id: ObjectId, needle: &Value) -> Result<Option<usize>, String> {
        let length = match self.state.heap.get(id)? {
            Object::Set(values) | Object::FrozenSet(values) => values.len(),
            _ => return Err("set handle changed object kind".into()),
        };
        for position in 0..length {
            let candidate = match self.state.heap.get(id)? {
                Object::Set(values) | Object::FrozenSet(values) => values[position],
                _ => return Err("set handle changed object kind".into()),
            };
            self.charge_cpu(1)?;
            if protocol::identical(&candidate, needle)
                || protocol::equals(&self.state.heap, &candidate, needle)?
            {
                return Ok(Some(position));
            }
        }
        Ok(None)
    }

    fn pop(&mut self) -> Result<Value, String> {
        if self.frame_stack_len() == 0 {
            return Err("invalid bytecode stack effect".into());
        }
        Ok(self.stack.pop().expect("non-empty frame stack was checked"))
    }

    fn take(&mut self, count: usize) -> Result<Vec<Value>, String> {
        if self.frame_stack_len() < count {
            return Err("invalid bytecode stack effect".into());
        }
        let start = self.stack.len() - count;
        Ok(self.stack.split_off(start))
    }

    fn copy(&mut self, depth: usize) -> Result<(), String> {
        if depth == 0 || depth > self.frame_stack_len() {
            return Err("invalid bytecode copy depth".into());
        }
        let value = self.stack[self.stack.len() - depth];
        self.stack.push(value);
        Ok(())
    }

    fn jump_if_or_pop(&mut self, jump_when: bool) -> Result<bool, String> {
        if self.frame_stack_len() == 0 {
            return Err("invalid bytecode stack effect".into());
        }
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
        if depth == 0 || depth > self.frame_stack_len() {
            return Err("invalid bytecode swap depth".into());
        }
        let top = self.stack.len() - 1;
        let other = self.stack.len() - depth;
        self.stack.swap(top, other);
        Ok(())
    }

    fn frame_stack_len(&self) -> usize {
        let stack_base = self
            .bytecode_frames
            .last()
            .map_or(0, |frame| frame.stack_base);
        self.stack.len().saturating_sub(stack_base)
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
    exception_base: Option<&'static str>,
    attributes: HashMap<String, Value>,
    dataclass_fields: Vec<(String, Option<Value>)>,
    enum_members: Vec<Value>,
}

fn is_os_error(name: &str) -> bool {
    matches!(
        name,
        "OSError"
            | "FileNotFoundError"
            | "FileExistsError"
            | "IsADirectoryError"
            | "NotADirectoryError"
            | "PermissionError"
    )
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

fn format_integer(value: BigInt, spec: &str) -> Result<String, String> {
    let presentation = spec
        .chars()
        .last()
        .ok_or_else(|| "empty integer format".to_string())?;
    let mut options = &spec[..spec.len() - presentation.len_utf8()];
    let alternate = options.starts_with('#');
    if alternate {
        options = &options[1..];
    }
    if options.contains('#') {
        return Err(format!("unsupported numeric format {spec:?}"));
    }
    let (radix, prefix, uppercase) = match presentation {
        'd' => (10, "", false),
        'b' => (2, "0b", false),
        'o' => (8, "0o", false),
        'x' => (16, "0x", false),
        'X' => (16, "0X", true),
        _ => return Err(format!("unsupported integer format {spec:?}")),
    };
    if alternate && radix == 10 {
        return Err(format!("alternate form is not allowed for {presentation}"));
    }
    let negative = value.sign() == Sign::Minus;
    let magnitude = if negative { -value } else { value };
    let mut digits = magnitude.to_str_radix(radix);
    if uppercase {
        digits.make_ascii_uppercase();
    }
    let prefix = if alternate { prefix } else { "" };
    let (width, precision, zero_pad) = parse_numeric_format(options)?;
    if precision.is_some() {
        return Err("precision is not allowed in integer format".into());
    }
    let sign = if negative { "-" } else { "" };
    let content_width = sign.len() + prefix.len() + digits.len();
    let padding = width.saturating_sub(content_width);
    if zero_pad {
        Ok(format!("{sign}{prefix}{}{digits}", "0".repeat(padding)))
    } else {
        Ok(format!("{}{sign}{prefix}{digits}", " ".repeat(padding)))
    }
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
    let plan = SlicePlan::new(length, start, stop, step)?;
    let units = u64::try_from(plan.len()).unwrap_or(u64::MAX);
    let selected = if is_ascii && plan.step() == 1 {
        let range = plan
            .contiguous_range()
            .expect("unit-step slices are contiguous");
        value
            .get(range)
            .expect("ASCII slice indices are byte boundaries")
            .to_owned()
    } else if is_ascii {
        plan.indices()
            .map(|index| char::from(value.as_bytes()[index]))
            .collect()
    } else {
        let characters = characters.expect("non-ASCII slices materialize code points");
        plan.indices().map(|index| characters[index]).collect()
    };
    Ok((selected, units))
}

fn range_length(start: i64, stop: i64, step: i64) -> Result<usize, String> {
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
    usize::try_from(count).map_err(|_| "range is too large".into())
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
