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

use num_bigint::BigInt;
use num_traits::ToPrimitive;

use super::ast::{BinaryOperator, ComparisonOperator, Constant, UnaryOperator};
use super::attributes::InstanceAttributeSlot;
use super::bytecode::{CallId, ClassField, CodeRef, DisplayKind, NameId, Opcode};
use super::cpython_names;
use super::definitions::DefinitionTable;
use super::exception_types;
use super::filesystem::PyModuleLoader;
use super::heap::{
    Builder, ClassLayout, InstancePayload, KeyHash, NamespaceTarget, Object, OrderedMap,
    OrderedSet, ProxyTarget, Ref, Roots, ValueStack, MODELED_MAPPING_ENTRY_BYTES,
    MODELED_SET_MEMBER_BYTES, MODELED_VALUE_BYTES,
};
use super::native::{
    CallArgs, FunctionDef, ModuleDef, PyArgumentParser, PyArgumentParserData, PyArgumentSpec,
    PyArray, PyArrayBuffer, PyArrayData, PyArrayDataMut, PyArrayDtype, PyArrayMut, PyArrayReader,
    PyArrayRef, PyArrayView, PyByteArray, PyCallable, PyClass, PyClock, PyDict, PyEnvironment,
    PyError, PyErrorKind, PyFilesystem, PyHttpClient, PyIdentity, PyIterator, PyKind, PyList,
    PyMarker, PyMatch, PyMatchData, PyModule, PyNativeKind, PyOperator, PyProcessHandle,
    PyProcessOutput, PyProcessPoll, PyProcessRunner, PyProcessStartRequest, PyProperty,
    PyRaisesContext, PyRegex, PyResult, PyRuntime, PySet, PyStreamRead, PySubcommandSpec,
    PySubparsersSpec, PyTuple, PyTypeObject, PyValueCast,
};
use super::number;
use super::object_model::{BuiltinType, Slot, SlotValue, TypeId};
use super::slice::SlicePlan;
use super::source::Span;
use super::symbols::SymbolId;
use super::{protocol, ExecResult, Out, ReplState, Value};

mod calls;
mod dispatch;
mod equality;
mod format;
mod handles;
mod hashing;
mod host;
mod iteration;
mod mappings;
mod namespace;
mod native_runtime;
mod objects;
mod operations;
mod summation;
const VM_POLL_QUANTUM: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeValue {
    Module(&'static ModuleDef),
    Function(Builtin),
    BuiltinType(BuiltinType),
    NativeFunction(&'static FunctionDef),
    NativeMethod(&'static super::native::MethodDef),
    /// Native method that attribute lookup binds to the type, as CPython binds a classmethod,
    /// whether it is reached through the type or through an instance.
    NativeClassMethod(&'static super::native::MethodDef),
    /// A callable view of a native slot owned by one type, independent of subclass overrides.
    SlotWrapper {
        owner: TypeId,
        slot: Slot,
    },
    /// Read-only data descriptor stored in a native type's attribute table.
    NativeGetter(&'static super::native::GetterDef),
    ValueKind(&'static super::native::ValueKindDef),
    Stream(Stream),
    Environment,
    ExceptionType(ExceptionType),
    /// Runtime representation of ``typing.List`` used by the generic-alias probe.
    TypingList,
    /// The `Ellipsis` singleton written as `...`. An immediate marker keeps identity, equality,
    /// and dictionary hashing canonical without allocating a heap object.
    Ellipsis,
    /// The `NotImplemented` singleton returned by binary and comparison methods that decline.
    NotImplemented,
}

static MODULES: DefinitionTable<ModuleDef> = DefinitionTable::new();
static FUNCTIONS: DefinitionTable<FunctionDef> = DefinitionTable::new();
static METHODS: DefinitionTable<super::native::MethodDef> = DefinitionTable::new();
static GETTERS: DefinitionTable<super::native::GetterDef> = DefinitionTable::new();
static VALUE_KINDS: DefinitionTable<super::native::ValueKindDef> = DefinitionTable::new();

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
    const VALUE_KIND: u8 = 9;
    const NATIVE_GETTER: u8 = 10;
    const ELLIPSIS: u8 = 11;
    const NOT_IMPLEMENTED: u8 = 12;
    const NATIVE_CLASS_METHOD: u8 = 13;
    const SLOT_WRAPPER: u8 = 14;

    pub(super) fn encode(self) -> (u64, u8) {
        match self {
            Self::Module(value) => (u64::from(MODULES.intern(value)), Self::MODULE),
            Self::Function(value) => (value.index() as u64, Self::FUNCTION),
            Self::BuiltinType(value) => (value as u64, Self::BUILTIN_TYPE),
            Self::NativeFunction(value) => {
                (u64::from(FUNCTIONS.intern(value)), Self::NATIVE_FUNCTION)
            }
            Self::NativeMethod(value) => (u64::from(METHODS.intern(value)), Self::NATIVE_METHOD),
            Self::NativeClassMethod(value) => {
                (u64::from(METHODS.intern(value)), Self::NATIVE_CLASS_METHOD)
            }
            Self::SlotWrapper { owner, slot } => (
                (u64::from(owner.raw()) << 8) | slot as u64,
                Self::SLOT_WRAPPER,
            ),
            Self::NativeGetter(value) => (u64::from(GETTERS.intern(value)), Self::NATIVE_GETTER),
            Self::ValueKind(value) => (u64::from(VALUE_KINDS.intern(value)), Self::VALUE_KIND),
            Self::Stream(value) => (value as u64, Self::STREAM),
            Self::Environment => (0, Self::ENVIRONMENT),
            Self::ExceptionType(value) => (exception_type_code(value.0), Self::EXCEPTION_TYPE),
            Self::TypingList => (0, Self::TYPING_LIST),
            Self::Ellipsis => (0, Self::ELLIPSIS),
            Self::NotImplemented => (0, Self::NOT_IMPLEMENTED),
        }
    }

    /// Decode a handle produced by [`Self::encode`]. Definitions are recovered from the
    /// process-wide tables in [`super::definitions`], so a payload can only name a definition
    /// that was interned.
    pub(super) fn decode(payload: u64, kind: u8) -> Self {
        const INTERNED: &str = "native handle names an interned definition";
        match kind {
            Self::MODULE => Self::Module(MODULES.get(payload).expect(INTERNED)),
            Self::FUNCTION => Self::Function(Builtin::from_index(payload)),
            Self::BUILTIN_TYPE => Self::BuiltinType(
                usize::try_from(payload)
                    .ok()
                    .and_then(|index| BuiltinType::ALL.get(index))
                    .copied()
                    .expect("builtin type handle is in range"),
            ),
            Self::NATIVE_FUNCTION => Self::NativeFunction(FUNCTIONS.get(payload).expect(INTERNED)),
            Self::NATIVE_METHOD => Self::NativeMethod(METHODS.get(payload).expect(INTERNED)),
            Self::NATIVE_CLASS_METHOD => {
                Self::NativeClassMethod(METHODS.get(payload).expect(INTERNED))
            }
            Self::SLOT_WRAPPER => Self::SlotWrapper {
                owner: TypeId::from_raw((payload >> 8) as u32),
                slot: Slot::from_index(payload as u8),
            },
            Self::NATIVE_GETTER => Self::NativeGetter(GETTERS.get(payload).expect(INTERNED)),
            Self::STREAM => Self::Stream(match payload {
                0 => Stream::Stdin,
                1 => Stream::Stdout,
                2 => Stream::Stderr,
                _ => Stream::StdinBuffer,
            }),
            Self::ENVIRONMENT => Self::Environment,
            Self::EXCEPTION_TYPE => {
                Self::ExceptionType(ExceptionType(exception_type_name(payload)))
            }
            Self::TYPING_LIST => Self::TypingList,
            Self::ELLIPSIS => Self::Ellipsis,
            Self::NOT_IMPLEMENTED => Self::NotImplemented,
            Self::VALUE_KIND => Self::ValueKind(VALUE_KINDS.get(payload).expect(INTERNED)),
            _ => unreachable!("invalid private native-value tag"),
        }
    }
}

fn known_exception_type(name: &str) -> Option<&'static str> {
    exception_types::exception_type(name).map(|definition| definition.name)
}

fn exception_type_code(name: &str) -> u64 {
    exception_types::EXCEPTION_TYPES
        .iter()
        .position(|definition| definition.name == name)
        .and_then(|index| u64::try_from(index).ok())
        .expect("exception type must come from the closed exception table")
}

fn exception_type_name(code: u64) -> &'static str {
    usize::try_from(code)
        .ok()
        .and_then(|index| exception_types::EXCEPTION_TYPES.get(index))
        .map(|definition| definition.name)
        .expect("invalid private exception-type handle")
}

impl NativeValue {
    pub(super) fn repr(self) -> String {
        match self {
            Self::BuiltinType(builtin_type) => format!("<class '{}'>", builtin_type.name()),
            Self::ValueKind(kind) => format!("<class '{}'>", kind.name),
            Self::ExceptionType(ExceptionType(name)) => {
                match exception_types::exception_type(name).map(|definition| definition.module) {
                    Some(module) if module != "builtins" => format!("<class '{module}.{name}'>"),
                    _ => format!("<class '{name}'>"),
                }
            }
            Self::NativeGetter(getter) => format!("{getter:?}"),
            Self::Ellipsis => "Ellipsis".into(),
            Self::NotImplemented => "NotImplemented".into(),
            Self::Function(builtin) => format!("<built-in function {}>", builtin.name()),
            Self::NativeFunction(function) => format!("<built-in function {}>", function.name),
            Self::NativeMethod(method) | Self::NativeClassMethod(method) => {
                format!(
                    "<method '{}' of '{}' objects>",
                    method.name, method.type_name
                )
            }
            Self::SlotWrapper { owner, slot } => {
                let name = super::object_model::SLOT_DEFS[slot as usize].1;
                match BuiltinType::ALL.get(owner.raw() as usize) {
                    Some(builtin) => {
                        format!("<slot wrapper '{name}' of '{}' objects>", builtin.name())
                    }
                    None => format!("<slot wrapper '{name}'>"),
                }
            }
            _ => "<native object>".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExceptionType(pub &'static str);

#[derive(Debug)]
struct RaisedException {
    kind: String,
    value: Ref,
}

impl Clone for RaisedException {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            value: self.value.dup(),
        }
    }
}

/// One call-chain entry captured while an uncaught exception unwinds the frame stack, rendered
/// as a CPython-style `File "...", line N, in <scope>` line.
///
/// [`Vm::propagate_error`] rebuilds this list from scratch on every unwind attempt, so a nested
/// unwind that is later discarded (for example inside a generator or `exec` sub-frame) never
/// leaks into the traceback that is finally reported for the real, uncaught error.
#[derive(Clone, Debug)]
struct TracebackFrame {
    /// Function name active at this call level, or `<module>` for the top-level frame.
    name: String,
    /// Source location this frame was executing when the exception passed through it.
    span: Span,
    /// `__file__` of the imported module that owns this frame's code, or `None` for the main
    /// program, whose name depends on how it was started.
    file: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Stream {
    Stdin,
    Stdout,
    Stderr,
    /// `sys.stdin.buffer`: the same descriptor as `Stdin`, read as raw bytes instead of text.
    StdinBuffer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Builtin {
    Print,
    Input,
    Exec,
    Eval,
    Exit,
    Character,
    Ordinal,
    Binary,
    Octal,
    Hexadecimal,
    Repr,
    Format,
    Hash,
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
    SetAttribute,
    DeleteAttribute,
    Globals,
    Locals,
    Vars,
}

/// The builtin functions the VM implements itself, by Python name. `exit` and `quit` share one.
pub(super) const BUILTIN_FUNCTIONS: &[(&str, Builtin)] = &[
    ("print", Builtin::Print),
    ("input", Builtin::Input),
    ("exec", Builtin::Exec),
    ("eval", Builtin::Eval),
    ("exit", Builtin::Exit),
    ("quit", Builtin::Exit),
    ("chr", Builtin::Character),
    ("ord", Builtin::Ordinal),
    ("bin", Builtin::Binary),
    ("oct", Builtin::Octal),
    ("hex", Builtin::Hexadecimal),
    ("repr", Builtin::Repr),
    ("format", Builtin::Format),
    ("hash", Builtin::Hash),
    ("dir", Builtin::Dir),
    ("isinstance", Builtin::IsInstance),
    ("issubclass", Builtin::IsSubclass),
    ("len", Builtin::Length),
    ("sorted", Builtin::Sorted),
    ("min", Builtin::Minimum),
    ("max", Builtin::Maximum),
    ("sum", Builtin::Sum),
    ("abs", Builtin::Absolute),
    ("pow", Builtin::Power),
    ("divmod", Builtin::Divmod),
    ("callable", Builtin::Callable),
    ("enumerate", Builtin::Enumerate),
    ("zip", Builtin::Zip),
    ("any", Builtin::Any),
    ("all", Builtin::All),
    ("iter", Builtin::Iter),
    ("next", Builtin::Next),
    ("property", Builtin::Property),
    ("staticmethod", Builtin::StaticMethod),
    ("classmethod", Builtin::ClassMethod),
    ("super", Builtin::Super),
    ("setattr", Builtin::SetAttribute),
    ("delattr", Builtin::DeleteAttribute),
    ("globals", Builtin::Globals),
    ("locals", Builtin::Locals),
    ("vars", Builtin::Vars),
];

impl Builtin {
    /// Position of the builtin's first entry in [`BUILTIN_FUNCTIONS`], its handle payload.
    fn index(self) -> usize {
        BUILTIN_FUNCTIONS
            .iter()
            .position(|(_, builtin)| *builtin == self)
            .expect("every builtin function has a name")
    }

    fn from_index(payload: u64) -> Self {
        usize::try_from(payload)
            .ok()
            .and_then(|index| BUILTIN_FUNCTIONS.get(index))
            .map(|(_, builtin)| *builtin)
            .expect("builtin function handle is in range")
    }

    /// The builtin's Python name, as `__name__` and `repr` report it.
    pub(super) fn name(self) -> &'static str {
        BUILTIN_FUNCTIONS
            .iter()
            .find(|(_, builtin)| *builtin == self)
            .map(|(name, _)| *name)
            .expect("every builtin function has a name")
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
    let name_symbol = match state.symbols.intern("__name__", &mut interp.resources) {
        Ok(symbol) => symbol,
        Err(_) => return ExecResult::Exit(137),
    };
    if state.globals.get(&state.heap, name_symbol).is_none()
        && state
            .globals
            .insert(
                &state.heap,
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
                exception_base: 0,
                function_return: None,
                pending_native_call: None,
            });
            self.started = true;
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

/// The interpreter for one scheduler quantum, and the handle scope for every value it touches.
/// See [`handles`] for the scope discipline.
struct Vm<'s> {
    interp: &'s mut Interp,
    argv: &'s [String],
    stdin: &'s [u8],
    state: &'s mut ReplState,
    execution: &'s mut VmState,
    mode: VmMode,
    out: Out<'s>,
    err: Out<'s>,
    /// Handle-stack height when this scope opened; dropping the scope truncates back to it.
    handle_base: usize,
    /// Host scratch reserved before this scope opened. The scope owns everything reserved above
    /// it and releases that on each instruction boundary and when it closes, so a nested
    /// execution never refunds scratch an enclosing native call still holds.
    transient_base: u64,
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
#[derive(Default)]
struct VmState {
    stack: ValueStack,
    bytecode_frames: Vec<BytecodeFrame>,
    local_scopes: Vec<Ref>,
    class_scopes: Vec<Ref>,
    class_bindings: Vec<Vec<String>>,
    call_depth: usize,
    pending_exception: Option<RaisedException>,
    /// Frames collected by the most recent [`Vm::propagate_error`] unwind, freshest overwrites
    /// stale. Consumed by `render_execution` when reporting an uncaught exception's traceback.
    traceback_frames: Vec<TracebackFrame>,
    pending_wait: Option<crate::scheduler::WaitReason>,
    async_timer_deadlines: BTreeSet<u64>,
    native_suspend_allowed: bool,
    /// Synchronous executions (imports, class bodies, generators, `exec`, and Python calls made
    /// from native code) now running. Their callers cannot resume a suspended frame, so while
    /// any is active, natives finish blocking operations instead of suspending.
    synchronous_frames: usize,
    exception_stack: Vec<RaisedException>,
    with_contexts: Vec<Ref>,
    /// (class, receiver) pairs for frames executing a method, for zero-argument `super()`.
    method_frames: Vec<(Ref, Ref)>,
    code_caches: CodeCacheTable,
    stdin_position: usize,
    stdin_text: Option<String>,
    /// Bytes read from fd 0 but not yet consumed by a completed `read`/`readline`, kept across
    /// scheduler quanta so a retry after [`WaitReason`](crate::scheduler::WaitReason) does not
    /// lose data already drained from the pipe. Only used in scheduler-owned (streaming) mode;
    /// synchronous execution still uses `stdin_text`/`stdin_position` over the fully supplied
    /// `ProcessInput::stdin` slice.
    stdin_stream_pending: Vec<u8>,
    /// Whether fd 0 has reported end-of-file. Once set, further reads only drain
    /// `stdin_stream_pending` and never touch the descriptor again.
    stdin_stream_eof: bool,
    /// Host scratch charged by `reserve_result` while a native helper builds a result. Each
    /// `Vm` scope releases the part reserved above its own base at every instruction boundary
    /// and when it closes.
    transient_memory: u64,
    /// Host bytes held by the VM itself across quanta (stdin text, pipe buffers), released when
    /// the program completes.
    retained_memory: u64,
}

impl Clone for VmState {
    fn clone(&self) -> Self {
        Self {
            stack: self.stack.clone(),
            bytecode_frames: self.bytecode_frames.clone(),
            local_scopes: self.local_scopes.iter().map(Ref::dup).collect(),
            class_scopes: self.class_scopes.iter().map(Ref::dup).collect(),
            class_bindings: self.class_bindings.clone(),
            call_depth: self.call_depth,
            pending_exception: self.pending_exception.clone(),
            traceback_frames: self.traceback_frames.clone(),
            pending_wait: self.pending_wait.clone(),
            async_timer_deadlines: self.async_timer_deadlines.clone(),
            native_suspend_allowed: self.native_suspend_allowed,
            synchronous_frames: self.synchronous_frames,
            exception_stack: self.exception_stack.clone(),
            with_contexts: self.with_contexts.iter().map(Ref::dup).collect(),
            method_frames: self
                .method_frames
                .iter()
                .map(|(class, receiver)| (class.dup(), receiver.dup()))
                .collect(),
            code_caches: self.code_caches.clone(),
            stdin_position: self.stdin_position,
            stdin_text: self.stdin_text.clone(),
            stdin_stream_pending: self.stdin_stream_pending.clone(),
            stdin_stream_eof: self.stdin_stream_eof,
            transient_memory: self.transient_memory,
            retained_memory: self.retained_memory,
        }
    }
}

impl Roots for VmState {
    fn visit_refs(&mut self, visitor: &mut dyn FnMut(&mut Ref)) {
        self.stack.visit_refs(visitor);
        for slot in self
            .local_scopes
            .iter_mut()
            .chain(&mut self.class_scopes)
            .chain(&mut self.with_contexts)
        {
            visitor(slot);
        }
        if let Some(exception) = &mut self.pending_exception {
            visitor(&mut exception.value);
        }
        for exception in &mut self.exception_stack {
            visitor(&mut exception.value);
        }
        for (class, receiver) in &mut self.method_frames {
            visitor(class);
            visitor(receiver);
        }
        for caches in self.code_caches.iter_mut() {
            if let Some(attributes) = &mut caches.attributes {
                for cache in attributes.iter_mut().flatten() {
                    visitor(&mut cache.class);
                }
            }
        }
        for frame in &mut self.bytecode_frames {
            match &mut frame.pending_native_call {
                Some(PendingNativeCall::Function { arguments, .. }) => {
                    arguments.visit_refs(visitor);
                }
                Some(PendingNativeCall::Method {
                    receiver,
                    arguments,
                    ..
                }) => {
                    visitor(receiver);
                    arguments.visit_refs(visitor);
                }
                Some(PendingNativeCall::Input { .. }) | None => {}
            }
        }
    }
}

struct CodeCaches {
    code: CodeRef,
    names: Vec<Option<SymbolId>>,
    attributes: Option<Vec<Option<LoadAttributeCache>>>,
}

/// Per-code inline caches in stable slots.
///
/// A slot index stays valid while its code object is alive, which lets the dispatch cursor
/// hold one for a whole quantum. Code that only the table still references, such as a
/// finished `eval` or `exec` expression, is dropped when the table is pruned, so a program that
/// compiles source in a loop does not accumulate caches without bound or pay a linear lookup
/// for each one.
#[derive(Clone, Default)]
struct CodeCacheTable {
    slots: Vec<Option<CodeCaches>>,
    /// Slot of each live code object, keyed by the `Arc` address.
    by_code: HashMap<usize, usize>,
    free: Vec<usize>,
    /// Live slot count at which the next prune runs; doubles with the surviving set.
    prune_at: usize,
}

/// Fewest live caches kept before pruning is considered.
const CODE_CACHE_PRUNE_FLOOR: usize = 64;

impl CodeCacheTable {
    fn slot_of(&self, code: &CodeRef) -> Option<usize> {
        self.by_code.get(&(Arc::as_ptr(code) as usize)).copied()
    }

    fn get(&self, slot: usize) -> Option<&CodeCaches> {
        self.slots.get(slot)?.as_ref()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut CodeCaches> {
        self.slots.iter_mut().flatten()
    }

    /// Install caches for `code` and return their slot. Before growing past the prune
    /// threshold, release every cache whose code nothing else references; `release` receives
    /// the modeled bytes each one held.
    fn insert(&mut self, caches: CodeCaches, mut release: impl FnMut(usize)) -> usize {
        if self.by_code.len() >= self.prune_at.max(CODE_CACHE_PRUNE_FLOOR) {
            for (slot, entry) in self.slots.iter_mut().enumerate() {
                let stale = entry
                    .as_ref()
                    .is_some_and(|caches| Arc::strong_count(&caches.code) == 1);
                if stale {
                    let caches = entry.take().expect("stale slot holds caches");
                    self.by_code.remove(&(Arc::as_ptr(&caches.code) as usize));
                    self.free.push(slot);
                    release(code_cache_bytes(&caches));
                }
            }
            self.prune_at = self.by_code.len().saturating_mul(2);
        }
        let key = Arc::as_ptr(&caches.code) as usize;
        let slot = match self.free.pop() {
            Some(slot) => {
                self.slots[slot] = Some(caches);
                slot
            }
            None => {
                self.slots.push(Some(caches));
                self.slots.len() - 1
            }
        };
        self.by_code.insert(key, slot);
        slot
    }
}

impl std::ops::Index<usize> for CodeCacheTable {
    type Output = CodeCaches;

    fn index(&self, slot: usize) -> &CodeCaches {
        self.slots[slot].as_ref().expect("code cache slot is live")
    }
}

impl std::ops::IndexMut<usize> for CodeCacheTable {
    fn index_mut(&mut self, slot: usize) -> &mut CodeCaches {
        self.slots[slot].as_mut().expect("code cache slot is live")
    }
}

/// Modeled bytes retained by one code object's caches.
fn code_cache_bytes(caches: &CodeCaches) -> usize {
    let names = caches
        .names
        .len()
        .saturating_mul(std::mem::size_of::<Option<SymbolId>>());
    let attributes = caches.attributes.as_ref().map_or(0, |attributes| {
        attributes
            .len()
            .saturating_mul(std::mem::size_of::<Option<LoadAttributeCache>>())
    });
    names
        .saturating_add(attributes)
        .saturating_add(std::mem::size_of::<CodeCaches>())
}

impl Clone for CodeCaches {
    fn clone(&self) -> Self {
        Self {
            code: self.code.clone(),
            names: self.names.clone(),
            attributes: self.attributes.as_ref().map(|caches| {
                caches
                    .iter()
                    .map(|cache| {
                        cache.as_ref().map(|cache| LoadAttributeCache {
                            class: cache.class.dup(),
                            location: cache.location,
                        })
                    })
                    .collect()
            }),
        }
    }
}

/// One `LoadAttribute` site's last shaped lookup: valid while the receiver has this class and
/// shape.
struct LoadAttributeCache {
    class: Ref,
    location: InstanceAttributeSlot,
}

/// An executing code object's resumable control state.
#[derive(Clone)]
struct BytecodeFrame {
    code: CodeRef,
    instruction_pointer: usize,
    /// First operand owned by this frame in the VM's shared value stack.
    stack_base: usize,
    /// Active `try` regions as `(handler target, operand stack depth, exception stack depth)`.
    handlers: Vec<(usize, usize, usize)>,
    /// Handled exceptions below this depth belong to enclosing frames. Leaving the frame by any
    /// route truncates the exception stack here, so a `return` inside an `except` body or an
    /// exception escaping a handler cannot leave its handled exception behind.
    exception_base: usize,
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

/// Result of classifying one heap iterator at its mutation boundary.
enum IteratorAdvance<'s> {
    Yield(Value<'s>),
    Exhausted,
    Callable {
        callable: Value<'s>,
        sentinel: Value<'s>,
    },
    Generator,
    /// An object of a class that defines `__next__`, advanced by calling it.
    Protocol,
    /// Advancing a `StreamIterator` would block on fd 0; suspend the enclosing `for` loop.
    Blocked(crate::scheduler::WaitReason),
}

/// Outcome of one `ForIterator` opcode: advance and jump into the loop body, fall through past
/// it, or suspend the whole process because the iterator's next value isn't available yet.
pub(super) enum ForIterOutcome {
    Yielded,
    Exhausted,
    Blocked(crate::scheduler::WaitReason),
}

enum BuiltinSubscript<'s> {
    Value(Value<'s>),
    Mapping { factory: Option<Value<'s>> },
    Set,
    Unsupported,
}

/// What [`Vm::iterable_values`] read from a heap object before it starts running guest code.
enum MaterializeSource<'s> {
    Values(Vec<Value<'s>>),
    Range(i64, i64, i64),
    StoredIterator,
    Callable,
    Generator,
    Instance,
    NotIterable,
}

/// Distinct set members in first-seen order with their hashes, from [`Vm::distinct_members`].
/// Hashing and comparison run guest code that can allocate, so members stay handles until
/// [`Self::into_set`] stores them inside an allocation or mutation builder.
pub(super) struct HashedMembers<'s>(Vec<(KeyHash, Value<'s>)>);

impl HashedMembers<'_> {
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn into_set(self, builder: &Builder<'_>) -> OrderedSet {
        let mut set = OrderedSet::default();
        for (hash, member) in self.0 {
            set.push(hash, builder.store(member));
        }
        set
    }
}

/// Deduplicated dict entries in first-seen key order with their key hashes, from
/// [`Vm::ordered_map`]; [`Self::into_map`] stores them inside a builder.
pub(super) struct HashedEntries<'s>(Vec<(KeyHash, Value<'s>, Value<'s>)>);

impl HashedEntries<'_> {
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn into_map(self, builder: &Builder<'_>) -> OrderedMap {
        let mut map = OrderedMap::default();
        for (hash, key, value) in self.0 {
            map.push(hash, (builder.store(key), builder.store(value)));
        }
        map
    }
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
/// Both module-level functions and bound methods (e.g. `sys.stdin.readline()`) can suspend, so
/// this covers either shape of native call.
enum PendingNativeCall {
    Function {
        function: &'static FunctionDef,
        arguments: StoredCallArgs,
        call_span: super::source::Span,
    },
    Method {
        method: &'static super::native::MethodDef,
        receiver: Ref,
        arguments: StoredCallArgs,
        call_span: super::source::Span,
    },
    /// `input()`: not a table-driven native call, but it reads the same modeled stdin stream and
    /// so can suspend the same way. The prompt (if any) is already written by the time this is
    /// installed, so retrying only re-attempts the read.
    Input { call_span: super::source::Span },
}

impl Clone for PendingNativeCall {
    fn clone(&self) -> Self {
        match self {
            Self::Function {
                function,
                arguments,
                call_span,
            } => Self::Function {
                function,
                arguments: arguments.clone(),
                call_span: *call_span,
            },
            Self::Method {
                method,
                receiver,
                arguments,
                call_span,
            } => Self::Method {
                method,
                receiver: receiver.dup(),
                arguments: arguments.clone(),
                call_span: *call_span,
            },
            Self::Input { call_span } => Self::Input {
                call_span: *call_span,
            },
        }
    }
}

/// Native call arguments kept across a scheduler suspension, as stored references.
struct StoredCallArgs {
    positional: Vec<Ref>,
    keywords: Vec<(String, Ref)>,
}

impl StoredCallArgs {
    fn store(vm: &Vm<'_>, arguments: &CallArgs<'_>) -> Self {
        Self {
            positional: arguments
                .positional()
                .iter()
                .map(|value| vm.store(*value))
                .collect(),
            keywords: arguments
                .keywords()
                .iter()
                .map(|(name, value)| (name.clone(), vm.store(*value)))
                .collect(),
        }
    }

    fn load<'s>(&self, vm: &Vm<'s>) -> CallArgs<'s> {
        CallArgs::new(
            vm.handles(&self.positional),
            self.keywords
                .iter()
                .map(|(name, value)| (name.clone(), vm.handle(value)))
                .collect(),
        )
    }

    fn visit_refs(&mut self, visitor: &mut dyn FnMut(&mut Ref)) {
        for slot in &mut self.positional {
            visitor(slot);
        }
        for (_, slot) in &mut self.keywords {
            visitor(slot);
        }
    }
}

impl Clone for StoredCallArgs {
    fn clone(&self) -> Self {
        Self {
            positional: self.positional.iter().map(Ref::dup).collect(),
            keywords: self
                .keywords
                .iter()
                .map(|(name, value)| (name.clone(), value.dup()))
                .collect(),
        }
    }
}

impl PendingNativeCall {
    fn call_span(&self) -> super::source::Span {
        match self {
            Self::Function { call_span, .. }
            | Self::Method { call_span, .. }
            | Self::Input { call_span } => *call_span,
        }
    }
}

#[derive(Clone)]
struct FunctionReturn {
    name: String,
    call_span: super::source::Span,
    pop_method_frame: bool,
}

struct FunctionInvocation<'s> {
    arguments: Vec<Value<'s>>,
    keyword_arguments: Vec<(String, Value<'s>)>,
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

impl<'s> Vm<'s> {
    fn new(
        interp: &'s mut Interp,
        input: ProcessInput<'s>,
        state: &'s mut ReplState,
        execution: &'s mut VmState,
        mode: VmMode,
        out: Out<'s>,
        err: Out<'s>,
    ) -> Self {
        let handle_base = state.heap.handle_count();
        let transient_base = execution.transient_memory;
        Self {
            interp,
            argv: input.argv,
            stdin: input.stdin,
            state,
            execution,
            mode,
            out,
            err,
            handle_base,
            transient_base,
        }
    }

    /// Release the host scratch reserved since this scope opened.
    fn release_transient_memory(&mut self) {
        let bytes = self.transient_memory.saturating_sub(self.transient_base);
        if bytes == 0 {
            return;
        }
        self.transient_memory = self.transient_base;
        self.interp.resources.release_memory(bytes);
    }

    fn release_retained_memory(&mut self) {
        let bytes = std::mem::take(&mut self.retained_memory);
        self.interp.resources.release_memory(bytes);
    }

    /// Give back part of the retained reservation, for caches dropped before the run ends.
    fn release_retained_memory_bytes(&mut self, bytes: usize) {
        let bytes = u64::try_from(bytes)
            .unwrap_or(u64::MAX)
            .min(self.retained_memory);
        self.retained_memory -= bytes;
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
                    let value = self.handle(&exception.value);
                    let rendered = protocol::display(self.state, value)
                        .unwrap_or_else(|_| "SystemExit".to_string());
                    let status = rendered.parse::<i32>().unwrap_or_else(|_| {
                        self.err.extend_from_slice(rendered.as_bytes());
                        self.err.push(b'\n');
                        1
                    });
                    ExecResult::Exit(status)
                } else if let Some(exception) = self.pending_exception.clone() {
                    ExecResult::Exit(self.render_uncaught_exception(&exception, span))
                } else {
                    ExecResult::Unsupported(format!(
                        "{} at line {}, column {}",
                        error, span.line, span.column
                    ))
                }
            }
        }
    }

    /// Report an uncaught, genuine Python exception the way CPython does: a traceback on stderr
    /// and exit status 1. This is distinct from [`super::unsupported`], which stays reserved for
    /// syntax, modules, or builtins shellsim does not model at all.
    fn render_uncaught_exception(&mut self, exception: &RaisedException, fallback: Span) -> i32 {
        // `protocol::display` renders a message-less builtin exception as its type name (to match
        // `print(exc)` elsewhere), which would duplicate the type name we print explicitly below.
        // Read the raw message instead so an empty `ValueError()` prints as bare `ValueError`.
        let value = self.handle(&exception.value);
        let message = match protocol::exception_parts(self.state, value) {
            Ok(Some((_, message))) => message,
            _ => protocol::display(self.state, value).unwrap_or_else(|_| String::new()),
        };
        let filename = self.traceback_filename();
        let frames = std::mem::take(&mut self.traceback_frames);
        self.err
            .extend_from_slice(b"Traceback (most recent call last):\n");
        if frames.is_empty() {
            // `propagate_error` always populates this when it runs, but fall back to the
            // location `render_execution` already has so a traceback is never frame-less.
            self.err.extend_from_slice(
                format!(
                    "  File \"{filename}\", line {}, in <module>\n",
                    fallback.line
                )
                .as_bytes(),
            );
        }
        for frame in &frames {
            self.err.extend_from_slice(
                format!(
                    "  File \"{}\", line {}, in {}\n",
                    frame.file.as_deref().unwrap_or(&filename),
                    frame.span.line,
                    frame.name
                )
                .as_bytes(),
            );
        }
        let summary = if message.is_empty() {
            format!("{}\n", exception.kind)
        } else {
            format!("{}: {message}\n", exception.kind)
        };
        self.err.extend_from_slice(summary.as_bytes());
        1
    }

    /// The `File "..."` name CPython would use for this process's source: `<string>` for `-c`,
    /// `<stdin>` for a piped or REPL script, otherwise the script path as given on the command
    /// line.
    fn traceback_filename(&self) -> String {
        match self.argv.first().map(String::as_str) {
            Some("-c") => "<string>".to_string(),
            Some("-" | "") | None => "<stdin>".to_string(),
            Some(other) => other.to_string(),
        }
    }

    fn allocate_object(&mut self, object: Object) -> Result<Value<'s>, String> {
        self.alloc(object)
    }

    fn allocate_string(&mut self, value: String) -> Result<Value<'s>, String> {
        if let Some(value) = Value::inline_string(&value) {
            Ok(value)
        } else {
            self.allocate_object(Object::String(value.into()))
        }
    }

    fn allocate_bytes(&mut self, value: Vec<u8>) -> Result<Value<'s>, String> {
        self.allocate_object(Object::Bytes(value))
    }

    fn allocate_bytearray(&mut self, value: Vec<u8>) -> Result<Value<'s>, String> {
        self.allocate_object(Object::ByteArray(value))
    }

    /// A builtin exception whose only argument is `message`, or with no arguments when the
    /// message is empty, as the VM and native code raise them.
    fn allocate_exception(&mut self, kind: String, message: String) -> Result<Value<'s>, String> {
        let args = if message.is_empty() {
            Vec::new()
        } else {
            vec![self.allocate_string(message)?]
        };
        self.alloc_with(|builder| Object::Exception {
            kind,
            args: builder.refs(args),
        })
    }

    /// Raise a builtin exception with the constructor arguments `args`.
    fn raise_exception_args(&mut self, kind: &str, args: Vec<Value<'s>>) -> String {
        let value = match self.alloc_with(|builder| Object::Exception {
            kind: kind.to_string(),
            args: builder.refs(args),
        }) {
            Ok(value) => value,
            Err(error) => return error,
        };
        let message = protocol::exception_parts(self.state, value)
            .ok()
            .flatten()
            .map(|(_, message)| message)
            .unwrap_or_default();
        self.pending_exception = Some(RaisedException {
            kind: kind.to_string(),
            value: self.store(value),
        });
        message
    }

    /// Raise a builtin Python exception from VM code and return the error string that carries
    /// it. The pending exception makes the error catchable by `except`; uncaught, it prints a
    /// CPython traceback rather than an unsupported-feature report.
    fn raise_exception(&mut self, kind: &'static str, message: impl Into<String>) -> String {
        self.record_native_error(PyError::exception(kind, message))
    }

    /// Raise CPython's `TypeError: '<type>' object <complaint>`, as in "is not callable".
    fn raise_object_type_error(&mut self, value: &Value<'s>, complaint: &str) -> String {
        match self.type_name_of(value) {
            Ok(name) => self.raise_exception("TypeError", format!("'{name}' object {complaint}")),
            Err(error) => error,
        }
    }

    /// The type name CPython prints in error messages, such as `int` or a user class name.
    fn type_name_of(&self, value: &Value<'s>) -> Result<String, String> {
        if let Some(kind) = protocol::exception_kind(self.state, *value)? {
            return Ok(kind);
        }
        Ok(self.state.types.get(self.type_id(value)?)?.name.clone())
    }

    fn value_from_constant(&mut self, value: &Constant) -> Result<Value<'s>, String> {
        Ok(match value {
            Constant::None => Value::None,
            Constant::Ellipsis => Value::Native(NativeValue::Ellipsis),
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
            Constant::Imaginary(value) => {
                number::create_complex(self, 0.0, *value).map_err(|error| error.to_string())?
            }
            Constant::String(value) => self.allocate_string(value.clone())?,
            Constant::Bytes(value) => self.allocate_bytes(value.clone())?,
        })
    }

    fn range_values(&mut self, start: i64, stop: i64, step: i64) -> Result<Vec<Value<'s>>, String> {
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

    /// Every item of an iterable, as handles.
    ///
    /// Builtin containers, strings, bytes, ranges and the runtime's own iterators are copied
    /// natively, charging each item through `push_materialized`. Anything whose items come from
    /// Python code (generators, classes with `__iter__`, `iter(callable, sentinel)`) is drained by
    /// a bytecode loop in the frozen `_iteration` module instead, so the items accumulate in a
    /// metered heap list rather than in a host vector no accounting can see while guest frames run.
    fn iterable_values(&mut self, value: &Value<'s>) -> Result<Vec<Value<'s>>, String> {
        if self.is_unbounded_iterator(value)? {
            return Err("cannot materialize infinite itertools.count without a bound".into());
        }
        if self.has_python_iter(value)? {
            return self.materialize_through_bytecode(*value);
        }
        if let Some(value) = protocol::builtin_payload(&self.state.heap, *value)? {
            return self.iterable_values(&value);
        }
        let mut result = Vec::new();
        if let Some(value) = protocol::string_value(&self.state.heap, *value)? {
            for character in value.chars() {
                let character = self.allocate_string(character.to_string())?;
                self.push_materialized(&mut result, character)?;
            }
        } else if let Some(value) = protocol::bytes_value(&self.state.heap, *value)? {
            for byte in value {
                self.push_materialized(&mut result, Value::Int(i64::from(byte)))?;
            }
        } else if value.is_object() {
            let source = match self.get(*value)? {
                Object::List(values) | Object::Tuple(values) => {
                    MaterializeSource::Values(self.handles(values))
                }
                Object::Set(values) | Object::FrozenSet(values) => {
                    MaterializeSource::Values(self.handles(values.iter()))
                }
                Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                    MaterializeSource::Values(self.handles(entries.iter().map(|(key, _)| key)))
                }
                Object::Range { start, stop, step } => {
                    MaterializeSource::Range(*start, *stop, *step)
                }
                Object::Iterator { .. }
                | Object::SequenceIterator { .. }
                | Object::ReverseIterator { .. }
                | Object::RangeIterator { .. } => MaterializeSource::StoredIterator,
                Object::CountIterator { .. } => {
                    return Err(
                        "cannot materialize infinite itertools.count without a bound".into(),
                    )
                }
                Object::CallableIterator { .. } => MaterializeSource::Callable,
                Object::Generator { .. } => MaterializeSource::Generator,
                Object::Class(class_object) if !class_object.enum_members.is_empty() => {
                    MaterializeSource::Values(self.handles(&class_object.enum_members))
                }
                Object::Instance { .. } => MaterializeSource::Instance,
                _ => MaterializeSource::NotIterable,
            };
            match source {
                MaterializeSource::Values(values) => {
                    for value in values {
                        self.push_materialized(&mut result, value)?;
                    }
                }
                MaterializeSource::Range(start, stop, step) => {
                    for value in self.range_values(start, stop, step)? {
                        self.push_materialized(&mut result, value)?;
                    }
                }
                MaterializeSource::StoredIterator => {
                    while let Some(item) = self.next_stored_iterator(*value)? {
                        self.push_materialized(&mut result, item)?;
                    }
                }
                MaterializeSource::Callable | MaterializeSource::Generator => {
                    return self.materialize_through_bytecode(*value);
                }
                MaterializeSource::Instance
                    if self
                        .state
                        .types
                        .slot(self.type_id(value)?, Slot::GetItem)?
                        .is_some() =>
                {
                    self.legacy_sequence_values(value, &mut result)?;
                }
                MaterializeSource::Instance | MaterializeSource::NotIterable => {
                    return Err(self.raise_object_type_error(value, "is not iterable"))
                }
            }
        } else {
            return Err(self.raise_object_type_error(value, "is not iterable"));
        }
        Ok(result)
    }

    /// Whether `value` iterates through an `__iter__` slot rather than one of the builtin kinds
    /// `iterable_values` copies natively. Native value kinds such as dict views count, since their
    /// `__iter__` is the only way to reach their items.
    fn has_python_iter(&self, value: &Value<'s>) -> Result<bool, String> {
        if !value.is_object() {
            return Ok(false);
        }
        if matches!(
            self.get(*value)?,
            Object::String(_)
                | Object::Bytes(_)
                | Object::ByteArray(_)
                | Object::List(_)
                | Object::Tuple(_)
                | Object::Set(_)
                | Object::FrozenSet(_)
                | Object::Dict(_)
                | Object::DefaultDict { .. }
                | Object::Range { .. }
                | Object::Iterator { .. }
                | Object::SequenceIterator { .. }
                | Object::ReverseIterator { .. }
                | Object::RangeIterator { .. }
                | Object::CountIterator { .. }
                | Object::Class(_)
        ) {
            return Ok(false);
        }
        Ok(matches!(
            self.state.types.slot(self.type_id(value)?, Slot::Iter)?,
            Some(slot) if !matches!(&slot, SlotValue::Descriptor(descriptor) if descriptor.is_none())
        ))
    }

    /// Drain `iterable` with the frozen `_iteration.materialize` loop and return its items. The
    /// helper runs as ordinary bytecode: each item is one instruction, and the list it builds is
    /// charged by the heap as it grows.
    fn materialize_through_bytecode(
        &mut self,
        iterable: Value<'s>,
    ) -> Result<Vec<Value<'s>>, String> {
        const MODULE: &str = "_iteration";
        let module = match self.loaded_module(MODULE) {
            Some(module) => module,
            None => <Self as PyRuntime>::import_module(self, MODULE)
                .map_err(|error| error.to_string())?,
        };
        let scope = self.module_scope(MODULE, module)?;
        let helper = self
            .scope_get(scope, "materialize")?
            .ok_or("frozen _iteration module lacks materialize")?;
        let items = <Self as PyRuntime>::call_value(
            self,
            helper,
            CallArgs::new(vec![iterable], Vec::new()),
        )
        .map_err(|error| error.to_string())?;
        let Object::List(items) = self.get(items)? else {
            return Err("_iteration.materialize did not return a list".into());
        };
        Ok(self.handles(items))
    }

    fn is_unbounded_iterator(&self, value: &Value<'s>) -> Result<bool, String> {
        if !value.is_object() {
            return Ok(false);
        }
        Ok(matches!(self.get(*value)?, Object::CountIterator { .. }))
    }

    /// Iterate an instance without `__iter__` through CPython's legacy sequence protocol:
    /// call `__getitem__(0)`, `__getitem__(1)`, ... until it raises `IndexError` or
    /// `StopIteration`. Any other exception propagates. Each item is metered, so a
    /// `__getitem__` that never raises exhausts the CPU budget instead of looping forever.
    fn legacy_sequence_values(
        &mut self,
        value: &Value<'s>,
        result: &mut Vec<Value<'s>>,
    ) -> Result<(), String> {
        let mut index: i64 = 0;
        loop {
            match self.invoke_slot(value, Slot::GetItem, "__getitem__", vec![Value::Int(index)]) {
                Ok(Some(item)) => self.push_materialized(result, item)?,
                Ok(None) => return Err("__getitem__ slot disappeared during iteration".into()),
                Err(error) => {
                    let Some(exception) = self.pending_exception.as_ref() else {
                        return Err(error);
                    };
                    let exception_value = self.handle(&exception.value);
                    let kind = match self.user_exception_base(&exception_value)? {
                        Some(base) => base,
                        None => exception.kind.as_str(),
                    };
                    if !exception_types::exception_is_subclass(kind, "IndexError")
                        && !exception_types::exception_is_subclass(kind, "StopIteration")
                    {
                        return Err(error);
                    }
                    self.pending_exception = None;
                    return Ok(());
                }
            }
            index = index
                .checked_add(1)
                .ok_or("sequence index exceeds bounded integer range")?;
        }
    }

    fn push_materialized(
        &mut self,
        values: &mut Vec<Value<'s>>,
        value: Value<'s>,
    ) -> Result<(), String> {
        // A host Vec has allocator/capacity overhead that is not represented in the Python heap.
        // Reserve a deliberately generous per-item amount before every push, including string
        // payloads, so repeated materialization cannot grow outside the memory budget.
        let payload = protocol::string_ref(&self.state.heap, value)?
            .map_or(0, |text| text.byte_len().saturating_mul(2));
        self.reserve_result(64usize.saturating_add(payload))?;
        self.charge_cpu(1)?;
        values.push(value);
        Ok(())
    }

    /// The distinct `candidates` in first-seen order, as the `set` constructor and set displays
    /// keep them. Each candidate is hashed once and compared only with members of equal hash.
    fn distinct_members(
        &mut self,
        candidates: Vec<Value<'s>>,
    ) -> Result<HashedMembers<'s>, String> {
        let mut members = Vec::new();
        let mut index = HashMap::<KeyHash, Vec<usize>>::new();
        for candidate in candidates {
            let hash = self.hash_value(&candidate)?;
            let mut present = false;
            // `members` is local, so a guest `__eq__` cannot change the candidate positions.
            for position in index.get(&hash).cloned().unwrap_or_default() {
                let (_, member) = members[position];
                self.charge_cpu(1)?;
                if self.values_equal(&member, &candidate)? {
                    present = true;
                    break;
                }
            }
            if !present {
                index.entry(hash).or_default().push(members.len());
                members.push((hash, candidate));
            }
        }
        Ok(HashedMembers(members))
    }

    /// Dict entries for `entries` in order. A repeated key keeps its first position and takes
    /// the last value, as a dict display does.
    fn ordered_map(
        &mut self,
        entries: Vec<(Value<'s>, Value<'s>)>,
    ) -> Result<HashedEntries<'s>, String> {
        let mut map: Vec<(KeyHash, Value<'s>, Value<'s>)> = Vec::new();
        let mut index = HashMap::<KeyHash, Vec<usize>>::new();
        for (key, value) in entries {
            let hash = self.hash_value(&key)?;
            let mut existing = None;
            for position in index.get(&hash).cloned().unwrap_or_default() {
                let member = map[position].1;
                self.charge_cpu(1)?;
                if self.values_equal(&member, &key)? {
                    existing = Some(position);
                    break;
                }
            }
            match existing {
                Some(position) => map[position].2 = value,
                None => {
                    index.entry(hash).or_default().push(map.len());
                    map.push((hash, key, value));
                }
            }
        }
        Ok(HashedEntries(map))
    }

    /// Allocate a dict holding `entries`, deduplicated as [`Self::ordered_map`] does.
    fn allocate_dict(&mut self, entries: Vec<(Value<'s>, Value<'s>)>) -> Result<Value<'s>, String> {
        let entries = self.ordered_map(entries)?;
        self.alloc_with(|builder| Object::Dict(entries.into_map(builder)))
    }

    /// The position of the entry whose key equals `needle`.
    fn find_mapping_entry(
        &mut self,
        mapping: Value<'s>,
        needle: &Value<'s>,
    ) -> Result<Option<usize>, String> {
        Ok(self.lookup_mapping_entry(mapping, needle)?.1)
    }

    /// `needle`'s hash and the position of the entry whose key equals it. Hashing and a user
    /// `__eq__` may run guest code that mutates the dict, so each candidate key is read again
    /// before it is compared.
    fn lookup_mapping_entry(
        &mut self,
        mapping: Value<'s>,
        needle: &Value<'s>,
    ) -> Result<(KeyHash, Option<usize>), String> {
        let hash = self.hash_value(needle)?;
        let candidates = match self.get(mapping)? {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                entries.candidate_positions(hash).to_vec()
            }
            _ => return Err("dict handle changed object kind".into()),
        };
        for position in candidates {
            self.charge_cpu(1)?;
            let candidate = match self.get(mapping)? {
                Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                    entries.get(position).map(|entry| self.handle(&entry.0))
                }
                _ => return Err("dict handle changed object kind".into()),
            };
            let Some(candidate) = candidate else {
                continue;
            };
            if self.values_equal(&candidate, needle)? {
                return Ok((hash, Some(position)));
            }
        }
        Ok((hash, None))
    }

    /// The position of the member equal to `needle`.
    fn find_set_entry(
        &mut self,
        set: Value<'s>,
        needle: &Value<'s>,
    ) -> Result<Option<usize>, String> {
        Ok(self.lookup_set_entry(set, needle)?.1)
    }

    /// `needle`'s hash and the position of the member equal to it, rereading each candidate
    /// because guest code may mutate the set during the comparison.
    fn lookup_set_entry(
        &mut self,
        set: Value<'s>,
        needle: &Value<'s>,
    ) -> Result<(KeyHash, Option<usize>), String> {
        let hash = self.hash_value(needle)?;
        let candidates = match self.get(set)? {
            Object::Set(values) | Object::FrozenSet(values) => {
                values.candidate_positions(hash).to_vec()
            }
            _ => return Err("set handle changed object kind".into()),
        };
        for position in candidates {
            let candidate = match self.get(set)? {
                Object::Set(values) | Object::FrozenSet(values) => match values.get(position) {
                    Some(value) => self.handle(value),
                    None => continue,
                },
                _ => return Err("set handle changed object kind".into()),
            };
            self.charge_cpu(1)?;
            if self.values_equal(&candidate, needle)? {
                return Ok((hash, Some(position)));
            }
        }
        Ok((hash, None))
    }

    fn pop(&mut self) -> Result<Value<'s>, String> {
        if self.frame_stack_len() == 0 {
            return Err("invalid bytecode stack effect".into());
        }
        Ok(self
            .execution
            .stack
            .pop(&self.state.heap)
            .expect("non-empty frame stack was checked"))
    }

    fn take(&mut self, count: usize) -> Result<Vec<Value<'s>>, String> {
        if self.frame_stack_len() < count {
            return Err("invalid bytecode stack effect".into());
        }
        self.pop_many(count)
    }

    fn copy(&mut self, depth: usize) -> Result<(), String> {
        if depth == 0 || depth > self.frame_stack_len() {
            return Err("invalid bytecode copy depth".into());
        }
        let value = self.peek(depth - 1)?;
        self.push(value);
        Ok(())
    }

    fn jump_if_or_pop(&mut self, jump_when: bool) -> Result<bool, String> {
        if self.frame_stack_len() == 0 {
            return Err("invalid bytecode stack effect".into());
        }
        let value = self
            .execution
            .stack
            .peek(&self.state.heap, 0)
            .ok_or("invalid bytecode stack effect")?;
        if self.truth_value(&value)? == jump_when {
            Ok(true)
        } else {
            self.execution.stack.pop_ref();
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
        super::heap::charge_construction(bytes, &mut self.interp.resources)?;
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
            Stream::Stdin | Stream::StdinBuffer => {}
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

enum CallResult<'s> {
    Value(Value<'s>),
    Exit(i32),
    EnteredFrame,
    Blocked(crate::scheduler::WaitReason, Value<'s>),
    Retry(crate::scheduler::WaitReason, PendingNativeCall),
}

#[derive(Clone, Copy)]
enum CallMode {
    Immediate,
    Deferred(super::source::Span),
}

/// Outcome of executing a frame. `Return` and `Yield` carry stored references because they
/// cross the boundary of the scope that produced them; the receiver re-roots them with
/// `Vm::handle` before anything can allocate.
enum Execution {
    Pending,
    Blocked(crate::scheduler::WaitReason),
    Halt,
    Return(Ref),
    Yield(Ref, usize),
    Exit(i32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SequenceKind {
    List,
    Tuple,
}

/// Fully resolved inputs to the single class allocator shared by class statements and
/// `type.__new__`.
struct ClassDefinition<'s> {
    name: String,
    bases: Vec<Value<'s>>,
    mro: Vec<Value<'s>>,
    metaclass: Value<'s>,
    layout: ClassLayout,
    exception_base: Option<&'static str>,
    attributes: HashMap<String, Value<'s>>,
    dataclass_fields: Vec<(String, Option<Value<'s>>)>,
    enum_members: Vec<Value<'s>>,
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

fn expect_arity(arguments: &[Value<'_>], minimum: usize, maximum: usize) -> Result<(), String> {
    if (minimum..=maximum).contains(&arguments.len()) {
        Ok(())
    } else {
        Err(format!(
            "expected {minimum}..={maximum} arguments, got {}",
            arguments.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_handles_round_trip_without_pointers() {
        for builtin in BuiltinType::ALL {
            let (payload, kind) = NativeValue::BuiltinType(builtin).encode();
            assert_eq!(
                NativeValue::decode(payload, kind),
                NativeValue::BuiltinType(builtin)
            );
        }
        for (_, builtin) in BUILTIN_FUNCTIONS {
            let (payload, kind) = NativeValue::Function(*builtin).encode();
            assert_eq!(
                NativeValue::decode(payload, kind),
                NativeValue::Function(*builtin)
            );
        }
        let module = super::super::stdlib::native_module("math").expect("math is native");
        let (payload, kind) = NativeValue::Module(module).encode();
        assert!(
            payload < 1 << 32,
            "a module handle is a table index, not an address"
        );
        assert_eq!(
            NativeValue::decode(payload, kind),
            NativeValue::Module(module)
        );
        let function = &module.functions[0];
        let (payload, kind) = NativeValue::NativeFunction(function).encode();
        assert!(std::ptr::eq(
            match NativeValue::decode(payload, kind) {
                NativeValue::NativeFunction(decoded) => decoded,
                other => panic!("decoded {other:?}"),
            },
            function
        ));
    }
}
