//! Metered Python VM façade and persistent execution state.
//!
//! Child modules partition bytecode dispatch, calls, object protocols, iteration, native-module
//! integration, and host adapters. They extend the single [`Vm`] type rather than introducing
//! subsystem traits or independent state owners; resumable state remains centralized here.

use crate::interp::Interp;
use crate::resources::Resources;

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::ToPrimitive;

use super::ast::{BinaryOperator, ComparisonOperator, Constant, UnaryOperator};
use super::attributes::ShapeId;
use super::bytecode::{CallId, ClassField, CodeRef, DisplayKind, Linker, Opcode};
use super::cpython_names;
use super::definitions::DefinitionTable;
use super::exception_types;
use super::filesystem::PyModuleLoader;
use super::heap::{
    ClassLayout, KeyHash, Namespace, NamespaceTarget, Object, OrderedMap, OrderedSet, ProxyTarget,
    Ref, Roots, ValueStack, MODELED_MAPPING_ENTRY_BYTES, MODELED_SET_MEMBER_BYTES,
    MODELED_VALUE_BYTES,
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
use super::string;
use super::symbols::SymbolId;
use super::{protocol, ExecResult, Out, ReplState, Value};

mod calls;
mod compare;
mod dispatch;
mod equality;
mod format;
mod hashing;
mod heap_access;
mod host;
mod iteration;
mod mappings;
mod namespace;
mod native_runtime;
mod objects;
mod operations;

pub(super) use objects::scan_cost;
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
    /// `LoadMethod`'s marker for an attribute that was bound rather than loaded as a method.
    /// It lives only on the operand stack between `LoadMethod` and `CallMethod`.
    NoReceiver,
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
    const NO_RECEIVER: u8 = 15;

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
            Self::NoReceiver => (0, Self::NO_RECEIVER),
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
            Self::NO_RECEIVER => Self::NoReceiver,
            Self::VALUE_KIND => Self::ValueKind(VALUE_KINDS.get(payload).expect(INTERNED)),
            _ => unreachable!("invalid private native-value tag"),
        }
    }
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

/// One call-chain entry captured while an uncaught exception unwinds the frame stack, rendered
/// as a CPython-style `File "...", line N, in <scope>` line.
///
/// [`Vm::propagate_error`] rebuilds this list from scratch on every unwind attempt, so a nested
/// unwind that is later discarded (for example inside a generator or `exec` sub-frame) never
/// leaks into the traceback that is finally reported for the real, uncaught error. Recording a
/// frame copies two references; names and files are read only when a traceback is printed.
#[derive(Debug)]
struct TracebackFrame {
    /// The function running at this call level, or `None` for the top-level frame.
    function: Option<Ref>,
    /// The outermost scope of the frame's code, whose `__file__` names an imported module's
    /// source; `None` when the frame has no lexical scope.
    module: Option<Ref>,
    /// Source location this frame was executing when the exception passed through it.
    span: Span,
}

impl Clone for TracebackFrame {
    fn clone(&self) -> Self {
        Self {
            function: self.function.as_ref().map(Ref::dup),
            module: self.module.as_ref().map(Ref::dup),
            span: self.span,
        }
    }
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
    let mut program = match VmProgram::compile(source, state, &mut interp.resources) {
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
    /// Compile a script or REPL entry against `state`'s symbol and code tables.
    pub(super) fn compile(
        source: &str,
        state: &mut ReplState,
        resources: &mut Resources,
    ) -> Result<Self, ExecResult> {
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
        let mut link = Linker::new(&mut state.symbols, &mut state.codes, resources);
        let code = super::compiler::compile(program, &mut link);
        if let Err(error) = link.finish() {
            return Err(match error.kind() {
                Some(PyErrorKind::Resource) => ExecResult::Exit(137),
                _ => ExecResult::Unsupported(error.message().to_string()),
            });
        }
        Ok(Self {
            code,
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
            let docstring = self.code.docstring.clone();
            let Ok(main) = vm.main_scope(docstring.as_deref()) else {
                return VmPoll::Ready(ExecResult::Exit(137));
            };
            let entry = FrameEntry::module(main);
            let Ok(frame) = vm.enter_frame(&self.code, 0, 0, entry, FrameKind::Entry) else {
                return VmPoll::Ready(ExecResult::Exit(137));
            };
            vm.bytecode_frames.push(frame);
            self.started = true;
        }
        let execution = vm.execute_active_frame(VM_POLL_QUANTUM);
        vm.release_transient_memory();
        if !vm.state.sync_type_memory(&mut vm.interp.resources) {
            return VmPoll::Ready(ExecResult::Exit(137));
        }
        match &execution {
            Ok(Flow::Pending) => return VmPoll::Runnable,
            Ok(Flow::Blocked) => {
                let suspension = vm
                    .suspension
                    .as_ref()
                    .expect("a blocked frame records its suspension");
                return VmPoll::Blocked(suspension.reason.clone());
            }
            _ => {}
        }
        vm.bytecode_frames
            .pop()
            .expect("completed program must retain its root frame");
        vm.release_retained_memory();
        VmPoll::Ready(vm.render_execution(execution))
    }
}

/// The interpreter for one scheduler quantum, and the pin scope for every value it touches.
/// See [`heap_access`] for the scope discipline.
struct Vm<'s> {
    interp: &'s mut Interp,
    argv: &'s [String],
    stdin: &'s [u8],
    state: &'s mut ReplState,
    execution: &'s mut VmState,
    mode: VmMode,
    out: Out<'s>,
    err: Out<'s>,
    /// Pin-stack height when this scope opened; dropping the scope truncates back to it.
    pin_base: usize,
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
    test_timeout: Option<TestTimeout>,
    stack: ValueStack,
    bytecode_frames: Vec<BytecodeFrame>,
    /// Local slots of every frame that keeps its locals in the VM, frames stacked in call
    /// order; `None` is an unbound local. Rooted like the operand stack.
    locals: Vec<Option<Ref>>,
    /// Open `try` regions of every frame as `(handler target, operand stack depth, exception
    /// stack depth)`, each frame's above its `handler_base`.
    handlers: Vec<(usize, usize, usize)>,
    call_depth: usize,
    /// The exception object being raised, between the operation that raised it and the handler
    /// that takes it. Operations that raise return [`PyError::pending`] beside it.
    pending_exception: Option<Ref>,
    /// Nesting of builtin sequence ordering in progress, bounded like other recursion.
    compare_depth: usize,
    /// Frames collected by the most recent [`Vm::propagate_error`] unwind, freshest overwrites
    /// stale. Consumed by `render_execution` when reporting an uncaught exception's traceback.
    traceback_frames: Vec<TracebackFrame>,
    pending_wait: Option<crate::scheduler::WaitReason>,
    /// Why the process last left the dispatch loop to wait, and the native call to retry when
    /// it resumes. At most one frame can be suspended, so this lives beside the frames rather
    /// than in each of them.
    suspension: Option<Box<Suspension>>,
    async_timer_deadlines: BTreeSet<u64>,
    native_suspend_allowed: bool,
    /// Frames Rust code is running with [`Vm::run_frame`] (imports, class bodies, `exec`,
    /// generators resumed by natives, and Python calls made from native code). Their callers
    /// cannot resume a suspended frame, so while any is active, natives finish blocking
    /// operations instead of suspending.
    synchronous_frames: usize,
    /// Exception objects being handled by `except` blocks, innermost last.
    exception_stack: Vec<Ref>,
    /// Context managers entered by every frame, each frame's above its `context_base`.
    with_contexts: Vec<Ref>,
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
            test_timeout: self.test_timeout,
            stack: self.stack.clone(),
            bytecode_frames: self.bytecode_frames.clone(),
            locals: self
                .locals
                .iter()
                .map(|slot| slot.as_ref().map(Ref::dup))
                .collect(),
            handlers: self.handlers.clone(),
            call_depth: self.call_depth,
            pending_exception: self.pending_exception.as_ref().map(Ref::dup),
            compare_depth: self.compare_depth,
            traceback_frames: self.traceback_frames.clone(),
            pending_wait: self.pending_wait.clone(),
            suspension: self.suspension.clone(),
            async_timer_deadlines: self.async_timer_deadlines.clone(),
            native_suspend_allowed: self.native_suspend_allowed,
            synchronous_frames: self.synchronous_frames,
            exception_stack: self.exception_stack.iter().map(Ref::dup).collect(),
            with_contexts: self.with_contexts.iter().map(Ref::dup).collect(),
            stdin_position: self.stdin_position,
            stdin_text: self.stdin_text.clone(),
            stdin_stream_pending: self.stdin_stream_pending.clone(),
            stdin_stream_eof: self.stdin_stream_eof,
            transient_memory: self.transient_memory,
            retained_memory: self.retained_memory,
        }
    }
}

/// Test deadlines include virtual waiting and modeled CPU time, never host elapsed time.
#[derive(Clone, Copy)]
struct TestTimeout {
    wall_start: u64,
    cpu_start: u64,
    duration: u64,
}

impl TestTimeout {
    fn remaining(self, interp: &Interp) -> u64 {
        let wall = interp.clock.monotonic_ns().saturating_sub(self.wall_start);
        let cpu = interp
            .resources
            .process_time_ns()
            .saturating_sub(self.cpu_start);
        self.duration.saturating_sub(wall.max(cpu))
    }
}

impl Roots for VmState {
    fn visit_refs(&self, visitor: &mut dyn FnMut(&Ref)) {
        self.stack.visit_refs(visitor);
        for slot in self.locals.iter().flatten() {
            visitor(slot);
        }
        for slot in &self.with_contexts {
            visitor(slot);
        }
        for exception in self.pending_exception.iter().chain(&self.exception_stack) {
            visitor(exception);
        }
        for frame in &self.traceback_frames {
            for slot in frame.function.iter().chain(&frame.module) {
                visitor(slot);
            }
        }
        for frame in &self.bytecode_frames {
            visitor(&frame.scope);
            visitor(&frame.globals);
            if let Some(callee) = &frame.callee {
                visitor(callee);
            }
        }
        if let Some(suspension) = &self.suspension {
            match &suspension.retry {
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

/// One code object's entry in the [`CodeTable`].
pub(super) struct CodeCaches {
    code: CodeRef,
    /// One entry per instruction, filled for `LoadAttribute` and `LoadMethod` sites.
    sites: Option<Vec<Option<SiteCache>>>,
}

impl CodeCaches {
    fn visit_refs(&self, visitor: &mut dyn FnMut(&Ref)) {
        for site in self.sites.iter().flatten().flatten() {
            match &site.resolved {
                Resolved::InstanceSlot(_) => {}
                Resolved::ClassValue(value) => visitor(value),
                Resolved::Method { descriptor, owner } => {
                    visitor(descriptor);
                    visitor(owner);
                }
            }
        }
    }
}

/// Every code object one interpreter has compiled, each with its inline caches.
///
/// The compiler registers each code object it makes and records the slot in
/// [`Code::cache_slot`](super::bytecode::Code::cache_slot), so a frame finds its caches by
/// index. The table keeps a reference to each code object. Once the table has grown to twice
/// the size that survived the last prune, registration first drops the code nothing else
/// references, such as a finished `exec` string, and reuses its slot, so a program that
/// compiles source in a loop does not accumulate entries. Entries and their site caches charge
/// their storage to the guest.
#[derive(Clone, Default)]
pub(super) struct CodeTable {
    slots: Vec<Option<CodeCaches>>,
    free: Vec<u32>,
    live: usize,
    /// Live entry count at which the next registration prunes; doubles with the survivors.
    prune_at: usize,
    /// Slots whose code has site caches, which a class attribute store clears.
    with_sites: Vec<u32>,
    modeled_bytes: u64,
}

/// Fewest live entries kept before pruning is considered.
const CODE_TABLE_PRUNE_FLOOR: usize = 64;

impl CodeTable {
    /// Charge a new entry and choose its slot; [`Self::install`] fills it with the code.
    pub(super) fn reserve_slot(&mut self, resources: &mut Resources) -> PyResult<u32> {
        if self.live >= self.prune_at.max(CODE_TABLE_PRUNE_FLOOR) {
            self.prune(resources);
        }
        self.reserve(std::mem::size_of::<CodeCaches>(), resources)?;
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None => {
                let slot = u32::try_from(self.slots.len()).map_err(|_| "too many code objects")?;
                self.slots.push(None);
                slot
            }
        };
        self.live += 1;
        Ok(slot)
    }

    pub(super) fn install(&mut self, slot: u32, code: CodeRef) {
        self.slots[slot as usize] = Some(CodeCaches { code, sites: None });
    }

    /// Drop every entry whose code only the table references and release its storage.
    fn prune(&mut self, resources: &mut Resources) {
        let mut released = 0usize;
        for (slot, entry) in self.slots.iter_mut().enumerate() {
            let stale = entry
                .as_ref()
                .is_some_and(|caches| Arc::strong_count(&caches.code) == 1);
            if stale {
                let caches = entry.take().expect("stale slot holds caches");
                released = released.saturating_add(code_cache_bytes(&caches));
                self.free
                    .push(u32::try_from(slot).expect("slots are numbered by u32"));
                self.live -= 1;
            }
        }
        let slots = &self.slots;
        self.with_sites.retain(|&slot| {
            slots[slot as usize]
                .as_ref()
                .is_some_and(|caches| caches.sites.is_some())
        });
        self.prune_at = self.live.saturating_mul(2);
        self.release(released, resources);
    }

    fn get(&self, slot: usize) -> Option<&CodeCaches> {
        self.slots.get(slot)?.as_ref()
    }

    /// Give `slot` an empty site cache per instruction, charging it; `false` when the memory
    /// limit leaves no room, in which case the code simply runs uncached.
    fn allocate_sites(&mut self, slot: usize, resources: &mut Resources) -> PyResult<bool> {
        let instructions = self[slot].code.instructions.len();
        let bytes = instructions
            .checked_mul(std::mem::size_of::<Option<SiteCache>>())
            .ok_or("site cache size overflow")?;
        if u64::try_from(bytes).unwrap_or(u64::MAX) > resources.memory_remaining() {
            return Ok(false);
        }
        self.reserve(bytes, resources)?;
        self[slot].sites = Some(std::iter::repeat_with(|| None).take(instructions).collect());
        self.with_sites
            .push(u32::try_from(slot).expect("slots are numbered by u32"));
        Ok(true)
    }

    /// Clear every site cache, after a class attribute store that any of them may depend on.
    fn clear_sites(&mut self, resources: &mut Resources) {
        let mut released = 0usize;
        for slot in std::mem::take(&mut self.with_sites) {
            if let Some(caches) = self.slots[slot as usize].as_mut() {
                let before = code_cache_bytes(caches);
                caches.sites = None;
                released = released.saturating_add(before - code_cache_bytes(caches));
            }
        }
        self.release(released, resources);
    }

    fn reserve(&mut self, bytes: usize, resources: &mut Resources) -> PyResult<()> {
        let bytes = u64::try_from(bytes).map_err(|_| "code table entry is too large")?;
        let next = self
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled code table size overflow")?;
        if !resources.reserve_memory(bytes) {
            return Err(PyError::resource_error("memory limit exceeded"));
        }
        self.modeled_bytes = next;
        Ok(())
    }

    fn release(&mut self, bytes: usize, resources: &mut Resources) {
        let bytes = u64::try_from(bytes)
            .unwrap_or(u64::MAX)
            .min(self.modeled_bytes);
        self.modeled_bytes -= bytes;
        resources.release_memory(bytes);
    }

    /// Transfer the table's accounting to the caller when an interpreter is discarded.
    pub(super) fn take_modeled_bytes(&mut self) -> u64 {
        std::mem::take(&mut self.modeled_bytes)
    }
}

impl Roots for CodeTable {
    fn visit_refs(&self, visitor: &mut dyn FnMut(&Ref)) {
        for caches in self.slots.iter().flatten() {
            caches.visit_refs(visitor);
        }
    }
}

impl std::fmt::Debug for CodeTable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CodeTable")
            .field("live", &self.live)
            .field("modeled_bytes", &self.modeled_bytes)
            .finish_non_exhaustive()
    }
}

impl std::ops::Index<usize> for CodeTable {
    type Output = CodeCaches;

    #[inline(always)]
    fn index(&self, slot: usize) -> &CodeCaches {
        self.slots[slot].as_ref().expect("code table slot is live")
    }
}

impl std::ops::IndexMut<usize> for CodeTable {
    fn index_mut(&mut self, slot: usize) -> &mut CodeCaches {
        self.slots[slot].as_mut().expect("code table slot is live")
    }
}

/// Modeled bytes retained by one code object's entry.
fn code_cache_bytes(caches: &CodeCaches) -> usize {
    let sites = caches.sites.as_ref().map_or(0, |sites| {
        sites
            .len()
            .saturating_mul(std::mem::size_of::<Option<SiteCache>>())
    });
    sites.saturating_add(std::mem::size_of::<CodeCaches>())
}

impl Clone for CodeCaches {
    fn clone(&self) -> Self {
        Self {
            code: self.code.clone(),
            sites: self.sites.as_ref().map(|sites| {
                sites
                    .iter()
                    .map(|site| {
                        site.as_ref().map(|site| SiteCache {
                            type_id: site.type_id,
                            shape: site.shape,
                            resolved: match &site.resolved {
                                Resolved::InstanceSlot(slot) => Resolved::InstanceSlot(*slot),
                                Resolved::ClassValue(value) => Resolved::ClassValue(value.dup()),
                                Resolved::Method { descriptor, owner } => Resolved::Method {
                                    descriptor: descriptor.dup(),
                                    owner: owner.dup(),
                                },
                            },
                        })
                    })
                    .collect()
            }),
        }
    }
}

/// One name-keyed site's last lookup, valid while the receiver has the same type and the same
/// instance-attribute shape, which is what decides every plain attribute lookup: a shaped
/// slot, a class attribute no instance attribute shadows, or a method of the type. Every
/// class attribute store clears the sites of every code object, so a hit needs no version
/// check.
struct SiteCache {
    type_id: TypeId,
    /// The receiver's instance-attribute shape; `None` for a value without instance attributes.
    shape: Option<ShapeId>,
    resolved: Resolved,
}

/// What a site resolved to.
enum Resolved {
    /// The attribute lives in this shaped slot of the receiver.
    InstanceSlot(usize),
    /// A class attribute that binding leaves as it is, such as a constant or a nested class.
    ClassValue(Ref),
    /// A plain function or native method of the receiver's type: `LoadMethod` pushes it with
    /// the receiver, `LoadAttribute` binds it to the receiver with `owner`, the defining class.
    Method { descriptor: Ref, owner: Ref },
}

/// An executing code object's activation record: everything the frame owns is either in
/// this struct or on a shared VM stack it indexes (`stack`, `locals`, `handlers`,
/// `with_contexts`, `exception_stack`), so entering a frame allocates nothing and leaving it
/// truncates those stacks back to the recorded bases.
///
/// Names resolve through `scope`. When `own_scope` is set it is the frame's own heap scope,
/// holding its locals and dynamically bound names: module and class bodies, generators, and
/// functions whose code sets [`CallSignature::heap_locals`](super::bytecode::CallSignature).
/// Otherwise the frame keeps its local slots on the shared locals stack from `locals_base`
/// and `scope` is where free-name lookup continues, the function's closure. `globals` is the
/// scope of the module the code belongs to, the root of the `scope` chain, kept here so a
/// global lookup does not walk the chain.
struct BytecodeFrame {
    code: CodeRef,
    instruction_pointer: usize,
    /// Handled exceptions below this depth belong to enclosing frames. Leaving the frame by any
    /// route truncates the exception stack here, so a `return` inside an `except` body or an
    /// exception escaping a handler cannot leave its handled exception behind.
    exception_base: u32,
    /// First operand owned by this frame in the VM's shared value stack.
    stack_base: u32,
    handler_base: u32,
    context_base: u32,
    /// First slot owned by this frame in the VM's shared locals stack, when its locals live
    /// there rather than in `scope`.
    locals_base: Option<u32>,
    scope: Ref,
    globals: Ref,
    /// The function this frame runs, for tracebacks and zero-argument `super()`; `None` for
    /// module, class and dynamic code.
    callee: Option<Ref>,
    /// Whether `scope` is the frame's own rather than its closure.
    own_scope: bool,
    /// Who resumes when this frame returns or yields.
    kind: FrameKind,
    class_body: bool,
}

/// How a frame hands back control. See [`BytecodeFrame::kind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameKind {
    /// Rust code runs the frame with [`Vm::run_frame`] and takes its result.
    Entry,
    /// A Python call entered the frame from the frame below, which resumes with the returned
    /// value.
    Call,
    /// A generator's frame. Its generator object sits on the operand stack just below the
    /// frame's operands; when the frame yields it saves its state into that object, and when
    /// it returns or raises it marks the generator finished. Either way `Consumer` resumes.
    Generator(Consumer),
}

/// What resumes when a generator frame yields or finishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Consumer {
    /// Rust code that resumed the generator (`next`, `send`, `throw`, or a native iterating
    /// it) takes the result from [`Vm::run_frame`].
    Rust,
    /// The frame below, whose instruction just before its instruction pointer resumed the
    /// generator. After a `ForIterator`, a yielded value goes to the loop body and finishing
    /// leaves the loop. After a `YieldFromSend`, that frame yields a yielded value on to its own
    /// consumer, and finishing completes the `yield from` with the returned value.
    Frame,
}

impl Clone for BytecodeFrame {
    fn clone(&self) -> Self {
        Self {
            code: self.code.clone(),
            instruction_pointer: self.instruction_pointer,
            exception_base: self.exception_base,
            stack_base: self.stack_base,
            handler_base: self.handler_base,
            context_base: self.context_base,
            locals_base: self.locals_base,
            scope: self.scope.dup(),
            globals: self.globals.dup(),
            callee: self.callee.as_ref().map(Ref::dup),
            own_scope: self.own_scope,
            kind: self.kind,
            class_body: self.class_body,
        }
    }
}

impl BytecodeFrame {
    /// The frame's own heap scope, when its names live in one.
    fn active_scope(&self) -> Option<&Ref> {
        self.own_scope.then_some(&self.scope)
    }

    fn stack_base(&self) -> usize {
        self.stack_base as usize
    }

    fn locals_base(&self) -> Option<usize> {
        self.locals_base.map(|base| base as usize)
    }
}

/// A stack depth recorded in a frame.
fn frame_index(depth: usize) -> PyResult<u32> {
    u32::try_from(depth).map_err(|_| "VM stack depth exceeds the frame index range".into())
}

/// The scopes and local slots a new frame starts with. See [`BytecodeFrame`].
struct FrameEntry {
    scope: Value,
    globals: Value,
    own_scope: bool,
    /// Base of the frame's already-bound local slots on the VM's locals stack, for a frame
    /// that keeps them there rather than in `scope`.
    locals_base: Option<usize>,
    callee: Option<Value>,
    class_body: bool,
}

impl FrameEntry {
    /// A module body, the main script's included: its names are the module's globals, which
    /// live in `scope`.
    fn module(scope: Value) -> FrameEntry {
        FrameEntry {
            scope,
            globals: scope,
            own_scope: true,
            locals_base: None,
            callee: None,
            class_body: false,
        }
    }

    /// A class body: its own scope, whose bindings become the class namespace, and which
    /// functions defined inside skip when they close over names.
    fn class_body(scope: Value, globals: Value) -> FrameEntry {
        FrameEntry {
            globals,
            class_body: true,
            ..Self::module(scope)
        }
    }

    /// A function whose local slots are already bound on the locals stack from `locals_base`
    /// and whose free names resolve through `closure`.
    fn with_locals(
        function: Value,
        closure: Value,
        globals: Value,
        locals_base: usize,
    ) -> FrameEntry {
        FrameEntry {
            scope: closure,
            globals,
            own_scope: false,
            locals_base: Some(locals_base),
            callee: Some(function),
            class_body: false,
        }
    }

    /// A function whose locals live in `scope`, a heap scope of its own.
    fn function(function: Value, scope: Value, globals: Value) -> FrameEntry {
        FrameEntry {
            globals,
            callee: Some(function),
            ..Self::module(scope)
        }
    }

    /// `exec`/`eval` code: no names of its own, free names resolved through `enclosing`.
    fn dynamic(enclosing: Value, globals: Value) -> FrameEntry {
        FrameEntry {
            scope: enclosing,
            globals,
            own_scope: false,
            locals_base: None,
            callee: None,
            class_body: false,
        }
    }
}

/// Where the active frame's local slots live, fixed for one dispatch quantum.
#[derive(Clone, Copy)]
enum LocalsLocation {
    /// Slots start at this index of the VM's shared locals stack.
    Stack(usize),
    /// Slots live in the frame's heap scope.
    Heap,
    /// The code declares no local slots.
    None,
}

/// Hot dispatch state retained in registers for one scheduler quantum.
///
/// The resumable frame remains the source of truth at suspension and frame boundaries. Between
/// those boundaries the cursor avoids rediscovering the active frame and rewriting its instruction
/// pointer after every opcode.
struct DispatchCursor {
    code: CodeRef,
    op_index: usize,
    locals: LocalsLocation,
}

/// Control leaving an opcode arm, a call, or a frame: the one type every routine that touches
/// the dispatch loop returns.
///
/// A value a call or an arm produces is left on the operand stack, which roots it, so no
/// variant carries a value. `Return` and `Yield` carry stored references because they cross
/// the pin scope of the frame that produced them; the receiver pins them again with
/// `Vm::value` before anything can allocate. `Blocked` reports a wait whose reason, and the
/// native call to retry, sit in [`VmState::suspension`].
#[derive(Debug)]
enum Flow {
    /// Continue with the next instruction.
    Next,
    Jump(usize),
    /// The active frame changed; the dispatch cursor reloads from it.
    Refresh,
    /// The active frame returned this value.
    Return(Ref),
    /// A generator frame run for Rust code yielded this value and saved itself into its
    /// generator.
    Yield(Ref),
    /// The code object ran off its end: a module, class body or the main program finished.
    Halt,
    Exit(i32),
    /// The quantum's budget ran out with work remaining.
    Pending,
    /// The process must wait; see [`VmState::suspension`].
    Blocked,
}

/// Why the process is waiting and what to retry when it wakes.
struct Suspension {
    reason: crate::scheduler::WaitReason,
    /// A native call interrupted by the wait, retried before the next instruction. `None`
    /// when the native completed and only asked the scheduler to wait afterwards.
    retry: Option<PendingNativeCall>,
}

impl Clone for Suspension {
    fn clone(&self) -> Self {
        Self {
            reason: self.reason.clone(),
            retry: self.retry.clone(),
        }
    }
}

/// Result of classifying one heap iterator at its mutation boundary.
enum IteratorAdvance {
    Yield(Value),
    Exhausted,
    Callable {
        callable: Value,
        sentinel: Value,
    },
    Generator,
    /// An object of a class that defines `__next__`, advanced by calling it.
    Protocol,
    /// Advancing a `StreamIterator` would block on fd 0; suspend the enclosing `for` loop.
    Blocked(crate::scheduler::WaitReason),
}

/// Outcome of one `ForIterator` opcode: advance and jump into the loop body, fall through past
/// it, enter a generator's frame that produces the next value, or suspend the whole process
/// because the iterator's next value isn't available yet.
pub(super) enum ForIterOutcome {
    Yielded,
    Exhausted,
    /// A generator frame is now active; its consumer frame resumes when it yields or finishes.
    Entered,
    Blocked(crate::scheduler::WaitReason),
}

enum BuiltinSubscript {
    Value(Value),
    Mapping { factory: Option<Value> },
    Set,
    Unsupported,
}

/// What [`Vm::iterable_values`] read from a heap object before it starts running guest code.
enum MaterializeSource {
    Values(Vec<Value>),
    Range(i64, i64, i64),
    StoredIterator,
    Callable,
    Generator,
    Instance,
    NotIterable,
}

/// Distinct set members in first-seen order with their hashes, from [`Vm::distinct_members`].
/// Hashing and comparison run guest code that can allocate, so members stay pinned values until
/// [`Self::into_set`] turns them into stored references for the set that holds them.
pub(super) struct HashedMembers(Vec<(KeyHash, Value)>);

impl HashedMembers {
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn into_set(self) -> OrderedSet {
        let mut set = OrderedSet::default();
        for (hash, member) in self.0 {
            set.push(hash, Ref::from(member));
        }
        set
    }
}

/// Deduplicated dict entries in first-seen key order with their key hashes, from
/// [`Vm::ordered_map`]; [`Self::into_map`] turns them into stored references.
pub(super) struct HashedEntries(Vec<(KeyHash, Value, Value)>);

impl HashedEntries {
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn into_map(self) -> OrderedMap {
        let mut map = OrderedMap::default();
        for (hash, key, value) in self.0 {
            map.push(hash, (Ref::from(key), Ref::from(value)));
        }
        map
    }
}

#[inline(always)]
fn dispatch_next(result: PyResult<()>) -> PyResult<Flow> {
    result.map(|()| Flow::Next)
}

impl DispatchCursor {
    fn for_active(vm: &mut Vm<'_>) -> PyResult<Self> {
        let frame = vm
            .bytecode_frames
            .last()
            .expect("bytecode execution requires an active frame");
        let code = frame.code.clone();
        let op_index = frame.instruction_pointer;
        let locals = match (frame.locals_base(), frame.own_scope) {
            (Some(base), _) => LocalsLocation::Stack(base),
            (None, true) => LocalsLocation::Heap,
            (None, false) => LocalsLocation::None,
        };
        Ok(Self {
            code,
            op_index,
            locals,
        })
    }

    fn refresh(&mut self, vm: &mut Vm<'_>) -> PyResult<()> {
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
        if !Arc::ptr_eq(&self.code, &frame.code) {
            let frames = vm
                .bytecode_frames
                .iter()
                .map(|f| {
                    format!(
                        "(code={:p} kind={:?} ip={})",
                        Arc::as_ptr(&f.code),
                        f.kind,
                        f.instruction_pointer
                    )
                })
                .collect::<Vec<_>>();
            panic!(
                "SYNC MISMATCH op_index={} cursor_code={:p} frames={frames:?}",
                self.op_index,
                Arc::as_ptr(&self.code)
            );
        }
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
    fn store(vm: &Vm<'_>, arguments: &CallArgs) -> Self {
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

    fn load(&self, vm: &Vm<'_>) -> CallArgs {
        CallArgs::new(
            vm.values(&self.positional),
            self.keywords
                .iter()
                .map(|(name, value)| (name.clone(), vm.value(value)))
                .collect(),
        )
    }

    fn visit_refs(&self, visitor: &mut dyn FnMut(&Ref)) {
        for slot in &self.positional {
            visitor(slot);
        }
        for (_, slot) in &self.keywords {
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
        let pin_base = state.heap.pin_count();
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
            pin_base,
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
    fn reserve_retained_memory(&mut self, bytes: usize) -> PyResult<()> {
        let bytes = u64::try_from(bytes).map_err(|_| "Python allocation is too large")?;
        let next = self
            .retained_memory
            .checked_add(bytes)
            .ok_or("modeled Python memory overflow")?;
        if !self.interp.resources.reserve_memory(bytes) {
            return Err(PyError::resource_error("memory limit exceeded"));
        }
        self.retained_memory = next;
        Ok(())
    }

    fn render_execution(
        &mut self,
        execution: Result<Flow, (PyError, super::source::Span)>,
    ) -> ExecResult {
        match execution {
            Ok(Flow::Halt | Flow::Return(_) | Flow::Yield(_)) => ExecResult::Continue,
            Ok(Flow::Exit(status)) => ExecResult::Exit(status),
            Ok(flow) => unreachable!("a completed program cannot end with {flow:?}"),
            Err((error, span)) => {
                if let Some(reason) = self.interp.resources.stop_reason() {
                    return ExecResult::Exit(reason.exit_status());
                }
                // An error that reached the top without unwinding a frame may not be raised yet.
                let error = self.raise_error(error);
                if let Some(PyErrorKind::Exit(status)) = error.kind() {
                    return ExecResult::Exit(*status);
                }
                let exception = match &self.pending_exception {
                    Some(exception) if error.is_pending() => self.value(exception),
                    _ => {
                        let fault = if error.is_pending() {
                            "exception unwound without a pending exception"
                        } else {
                            error.message()
                        };
                        return ExecResult::Unsupported(format!(
                            "{fault} at line {}, column {}",
                            span.line, span.column
                        ));
                    }
                };
                if self.pending_exception_is("SystemExit") {
                    let rendered = protocol::display(self.state, exception)
                        .unwrap_or_else(|_| "SystemExit".to_string());
                    let status = rendered.parse::<i32>().unwrap_or_else(|_| {
                        self.err.extend_from_slice(rendered.as_bytes());
                        self.err.push(b'\n');
                        1
                    });
                    ExecResult::Exit(status)
                } else {
                    ExecResult::Exit(self.render_uncaught_exception(exception, span))
                }
            }
        }
    }

    /// Report an uncaught, genuine Python exception the way CPython does: a traceback on stderr
    /// and exit status 1. This is distinct from [`super::unsupported`], which stays reserved for
    /// syntax, modules, or builtins shellsim does not model at all.
    fn render_uncaught_exception(&mut self, value: Value, fallback: Span) -> i32 {
        // `protocol::display` renders a message-less builtin exception as its type name (to match
        // `print(exc)` elsewhere), which would duplicate the type name we print explicitly below.
        // Read the raw message instead so an empty `ValueError()` prints as bare `ValueError`.
        let kind = exception_types::exception_type_name(self.state, value)
            .ok()
            .flatten()
            .unwrap_or_else(|| "BaseException".into());
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
            let name = match &frame.function {
                Some(function) => self.function_name(function),
                None => "<module>".to_string(),
            };
            let file = frame
                .module
                .as_ref()
                .and_then(|module| self.module_file(module));
            self.err.extend_from_slice(
                format!(
                    "  File \"{}\", line {}, in {name}\n",
                    file.as_deref().unwrap_or(&filename),
                    frame.span.line,
                )
                .as_bytes(),
            );
        }
        let summary = if message.is_empty() {
            format!("{kind}\n")
        } else {
            format!("{kind}: {message}\n")
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

    fn allocate_object(&mut self, object: Object) -> PyResult<Value> {
        self.alloc(object)
    }

    fn allocate_string(&mut self, value: String) -> PyResult<Value> {
        if let Some(value) = Value::inline_string(&value) {
            Ok(value)
        } else {
            self.allocate_object(Object::String(value.into()))
        }
    }

    fn allocate_bytes(&mut self, value: Vec<u8>) -> PyResult<Value> {
        self.allocate_object(Object::Bytes(value))
    }

    fn allocate_bytearray(&mut self, value: Vec<u8>) -> PyResult<Value> {
        self.allocate_object(Object::ByteArray(value))
    }

    /// A builtin exception whose only argument is `message`, or with no arguments when the
    /// message is empty, as the VM and native code raise them.
    fn allocate_exception(&mut self, kind: &str, message: String) -> PyResult<Value> {
        let args = if message.is_empty() {
            Vec::new()
        } else {
            vec![self.allocate_string(message)?]
        };
        self.allocate_exception_object(kind, args)
    }

    /// An instance of the builtin exception class `kind` with the constructor arguments `args`.
    /// The object header carries the class's type id, so `type()`, `isinstance` and slot
    /// lookups read it directly.
    fn allocate_exception_object(&mut self, kind: &str, args: Vec<Value>) -> PyResult<Value> {
        let type_id = self
            .state
            .types
            .exception_type_id(kind)
            .ok_or_else(|| format!("exception type {kind:?} is not registered"))?;
        self.allocate_typed(type_id, Object::Exception(Ref::all(args)))
    }

    /// Make the exception object `exception` the pending exception and return the marker that
    /// unwinds to its handler.
    fn raise_value(&mut self, exception: Value) -> PyError {
        self.pending_exception = Some(Ref::from(exception));
        PyError::pending()
    }

    /// Raise a builtin exception with the constructor arguments `args`.
    fn raise_exception_args(&mut self, kind: &str, args: Vec<Value>) -> PyError {
        match self.allocate_exception_object(kind, args) {
            Ok(value) => self.raise_value(value),
            Err(error) => error,
        }
    }

    /// Raise CPython's `TypeError: '<type>' object <complaint>`, as in "is not callable".
    fn raise_object_type_error(&self, value: &Value, complaint: &str) -> PyError {
        match self.type_name_of(value) {
            Ok(name) => PyError::exception("TypeError", format!("'{name}' object {complaint}")),
            Err(error) => error,
        }
    }

    /// Whether the exception object `exception` is an instance of the builtin exception class
    /// `kind` or a subclass of it, as `except kind:` would decide.
    fn exception_is(&self, exception: Value, kind: &str) -> PyResult<bool> {
        let expected = self
            .state
            .types
            .exception_type_id(kind)
            .ok_or_else(|| format!("exception type {kind:?} is not registered"))?;
        self.state
            .types
            .is_subclass(self.type_id(&exception)?, expected)
    }

    /// Whether the pending exception is an instance of the builtin exception class `kind`.
    fn pending_exception_is(&self, kind: &str) -> bool {
        self.pending_exception.as_ref().is_some_and(|exception| {
            matches!(self.exception_is(self.value(exception), kind), Ok(true))
        })
    }

    /// The type name CPython prints in error messages, such as `int` or a user class name.
    fn type_name_of(&self, value: &Value) -> PyResult<String> {
        Ok(self.state.types.get(self.type_id(value)?)?.name.clone())
    }

    fn value_from_constant(&mut self, value: &Constant) -> PyResult<Value> {
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
            Constant::Imaginary(value) => number::create_complex(self, 0.0, *value)?,
            Constant::String(value) => self.allocate_string(value.clone())?,
            Constant::Bytes(value) => self.allocate_bytes(value.clone())?,
        })
    }

    fn range_values(&mut self, start: i64, stop: i64, step: i64) -> PyResult<Vec<Value>> {
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

    /// Every item of an iterable, as pinned values.
    ///
    /// Builtin containers, strings, bytes, ranges and the runtime's own iterators are copied
    /// natively, charging each item through `push_materialized`. Anything whose items come from
    /// Python code (generators, classes with `__iter__`, `iter(callable, sentinel)`) is drained by
    /// a bytecode loop in the frozen `_iteration` module instead, so the items accumulate in a
    /// metered heap list rather than in a host vector no accounting can see while guest frames run.
    fn iterable_values(&mut self, value: &Value) -> PyResult<Vec<Value>> {
        if self.is_unbounded_iterator(value)? {
            return Err("cannot materialize infinite itertools.count without a bound".into());
        }
        if self.has_python_iter(value)? {
            return self.materialize_through_bytecode(*value);
        }
        let mut result = Vec::new();
        if let Some(value) = string::string_value(&self.state.heap, *value)? {
            for character in value.chars() {
                let character = self.allocate_string(character.to_string())?;
                self.push_materialized(&mut result, character)?;
            }
        } else if let Some(value) = string::bytes_value(&self.state.heap, *value)? {
            for byte in value {
                self.push_materialized(&mut result, Value::Int(i64::from(byte)))?;
            }
        } else if value.is_object() {
            let source = match self.get(*value)? {
                Object::List(values) | Object::Tuple(values) => {
                    MaterializeSource::Values(self.values(values))
                }
                Object::Set(values) | Object::FrozenSet(values) => {
                    MaterializeSource::Values(self.values(values.iter()))
                }
                Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                    MaterializeSource::Values(self.values(entries.iter().map(|(key, _)| key)))
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
                    MaterializeSource::Values(self.values(&class_object.enum_members))
                }
                _ if self.instance_class(*value)?.is_some() => MaterializeSource::Instance,
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
    fn has_python_iter(&self, value: &Value) -> PyResult<bool> {
        if !value.is_object() {
            return Ok(false);
        }
        // An instance of a user class iterates through Python code only when the class (or a
        // user ancestor) defines `__iter__`; an inherited builtin slot reads the payload natively.
        if self.instance_class(*value)?.is_some() {
            return Ok(matches!(
                self.state.types.slot(self.type_id(value)?, Slot::Iter)?,
                Some(SlotValue::Descriptor { value: descriptor, .. }) if !descriptor.is_none()
            ));
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
            Some(slot) if !matches!(&slot, SlotValue::Descriptor { value, .. } if value.is_none())
        ))
    }

    /// Drain `iterable` with the frozen `_iteration.materialize` loop and return its items. The
    /// helper runs as ordinary bytecode: each item is one instruction, and the list it builds is
    /// charged by the heap as it grows.
    fn materialize_through_bytecode(&mut self, iterable: Value) -> PyResult<Vec<Value>> {
        const MODULE: &str = "_iteration";
        let module = match self.loaded_module(MODULE) {
            Some(module) => module,
            None => <Self as PyRuntime>::import_module(self, MODULE)?,
        };
        let scope = self.module_scope(MODULE, module)?;
        let helper = self
            .scope_get_name(scope, "materialize")?
            .ok_or("frozen _iteration module lacks materialize")?;
        let items = <Self as PyRuntime>::call_value(
            self,
            helper,
            CallArgs::new(vec![iterable], Vec::new()),
        )?;
        let Object::List(items) = self.get(items)? else {
            return Err("_iteration.materialize did not return a list".into());
        };
        Ok(self.values(items))
    }

    fn is_unbounded_iterator(&self, value: &Value) -> PyResult<bool> {
        if !value.is_object() {
            return Ok(false);
        }
        Ok(matches!(self.get(*value)?, Object::CountIterator { .. }))
    }

    /// Iterate an instance without `__iter__` through CPython's legacy sequence protocol:
    /// call `__getitem__(0)`, `__getitem__(1)`, ... until it raises `IndexError` or
    /// `StopIteration`. Any other exception propagates. Each item is metered, so a
    /// `__getitem__` that never raises exhausts the CPU budget instead of looping forever.
    fn legacy_sequence_values(&mut self, value: &Value, result: &mut Vec<Value>) -> PyResult<()> {
        let mut index: i64 = 0;
        loop {
            match self.invoke_slot(value, Slot::GetItem, "__getitem__", vec![Value::Int(index)]) {
                Ok(Some(item)) => self.push_materialized(result, item)?,
                Ok(None) => return Err("__getitem__ slot disappeared during iteration".into()),
                Err(error) => {
                    return self
                        .catch(error, "IndexError")
                        .or_else(|error| self.catch(error, "StopIteration"));
                }
            }
            index = index
                .checked_add(1)
                .ok_or("sequence index exceeds bounded integer range")?;
        }
    }

    fn push_materialized(&mut self, values: &mut Vec<Value>, value: Value) -> PyResult<()> {
        // A host Vec has allocator/capacity overhead that is not represented in the Python heap.
        // Reserve a deliberately generous per-item amount before every push, including string
        // payloads, so repeated materialization cannot grow outside the memory budget.
        let payload = string::string_ref(&self.state.heap, value)?
            .map_or(0, |text| text.byte_len().saturating_mul(2));
        self.reserve_result(64usize.saturating_add(payload))?;
        self.charge_cpu(1)?;
        values.push(value);
        Ok(())
    }

    /// The distinct `candidates` in first-seen order, as the `set` constructor and set displays
    /// keep them. Each candidate is hashed once and compared only with members of equal hash.
    fn distinct_members(&mut self, candidates: Vec<Value>) -> PyResult<HashedMembers> {
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
    fn ordered_map(&mut self, entries: Vec<(Value, Value)>) -> PyResult<HashedEntries> {
        let mut map: Vec<(KeyHash, Value, Value)> = Vec::new();
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
    fn allocate_dict(&mut self, entries: Vec<(Value, Value)>) -> PyResult<Value> {
        let entries = self.ordered_map(entries)?;
        self.alloc(Object::Dict(entries.into_map()))
    }

    /// The position of the entry whose key equals `needle`.
    fn find_mapping_entry(&mut self, mapping: Value, needle: &Value) -> PyResult<Option<usize>> {
        Ok(self.lookup_mapping_entry(mapping, needle)?.1)
    }

    /// `needle`'s hash and the position of the entry whose key equals it. Hashing and a user
    /// `__eq__` may run guest code that mutates the dict, so each candidate key is read again
    /// before it is compared.
    fn lookup_mapping_entry(
        &mut self,
        mapping: Value,
        needle: &Value,
    ) -> PyResult<(KeyHash, Option<usize>)> {
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
                    entries.get(position).map(|entry| self.value(&entry.0))
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
    fn find_set_entry(&mut self, set: Value, needle: &Value) -> PyResult<Option<usize>> {
        Ok(self.lookup_set_entry(set, needle)?.1)
    }

    /// `needle`'s hash and the position of the member equal to it, rereading each candidate
    /// because guest code may mutate the set during the comparison.
    fn lookup_set_entry(
        &mut self,
        set: Value,
        needle: &Value,
    ) -> PyResult<(KeyHash, Option<usize>)> {
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
                    Some(value) => self.value(value),
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

    #[inline(always)]
    fn pop(&mut self) -> PyResult<Value> {
        if self.frame_stack_len() == 0 {
            return Err("invalid bytecode stack effect".into());
        }
        self.execution
            .stack
            .pop(&self.state.heap)
            .ok_or_else(|| "invalid bytecode stack effect".into())
    }

    fn take(&mut self, count: usize) -> PyResult<Vec<Value>> {
        if self.frame_stack_len() < count {
            return Err("invalid bytecode stack effect".into());
        }
        self.pop_many(count)
    }

    fn copy(&mut self, depth: usize) -> PyResult<()> {
        if depth == 0 || depth > self.frame_stack_len() {
            return Err("invalid bytecode copy depth".into());
        }
        let value = self.peek_ref(depth - 1)?.dup();
        self.execution.stack.push_ref(&value);
        Ok(())
    }

    fn jump_if_or_pop(&mut self, jump_when: bool) -> PyResult<bool> {
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

    fn swap(&mut self, depth: usize) -> PyResult<()> {
        if depth == 0 || depth > self.frame_stack_len() {
            return Err("invalid bytecode swap depth".into());
        }
        let top = self.stack.len() - 1;
        let other = self.stack.len() - depth;
        self.stack.swap(top, other);
        Ok(())
    }

    #[inline(always)]
    fn frame_stack_len(&self) -> usize {
        let stack_base = self
            .bytecode_frames
            .last()
            .map_or(0, BytecodeFrame::stack_base);
        self.stack.len().saturating_sub(stack_base)
    }

    fn reserve_result(&mut self, bytes: usize) -> PyResult<()> {
        let bytes = u64::try_from(bytes).map_err(|_| "string result is too large")?;
        super::heap::charge_construction(bytes, &mut self.interp.resources)?;
        let next = self
            .transient_memory
            .checked_add(bytes)
            .ok_or("modeled Python memory overflow")?;
        if !self.interp.resources.reserve_memory(bytes) {
            return Err(PyError::resource_error("memory limit exceeded"));
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
    fn charge_cpu(&mut self, units: u64) -> PyResult<()> {
        if self.interp.resources.charge_cpu(units) {
            Ok(())
        } else {
            Err("resource limit exceeded while executing Python".into())
        }
    }
}

#[derive(Clone, Copy)]
enum CallMode {
    Immediate,
    Deferred(super::source::Span),
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
    mro: Vec<Value>,
    metaclass: Value,
    layout: ClassLayout,
    exception_base: Option<&'static str>,
    /// The class namespace in definition order, holding pinned values.
    attributes: Namespace,
    dataclass_fields: Vec<(String, Option<Value>)>,
    enum_members: Vec<(SymbolId, Value)>,
}

fn select_string_slice(
    value: &str,
    is_ascii: bool,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> PyResult<(String, u64)> {
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

fn range_length(start: i64, stop: i64, step: i64) -> PyResult<usize> {
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

fn expect_arity(arguments: &[Value], minimum: usize, maximum: usize) -> PyResult<()> {
    if (minimum..=maximum).contains(&arguments.len()) {
        Ok(())
    } else {
        Err(format!(
            "expected {minimum}..={maximum} arguments, got {}",
            arguments.len()
        )
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The control type every opcode arm returns is one stored reference plus a tag; a payload
    /// that would grow it moves to `VmState` instead, as the suspension did.
    #[test]
    fn control_flow_stays_small() {
        assert_eq!(
            std::mem::size_of::<Flow>(),
            std::mem::size_of::<Ref>() + std::mem::size_of::<usize>()
        );
        assert!(std::mem::size_of::<PyResult<Flow>>() <= 32);
        // Five words of indices, three references (scope, globals and the optional callee), two
        // flags and the frame kind: pushed and popped by value on every call.
        assert!(std::mem::size_of::<BytecodeFrame>() <= 96);
    }

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
