//! Uniform value, argument, error, and runtime interfaces for native Python modules.
//!
//! Native functions are type-erased at the call boundary: every function accepts [`CallArgs`]
//! and returns a [`PyValue`]. Implementations recover small checked views such as [`PyNumber`] or
//! [`PyList`] locally. The runtime trait exposes allocation and metering, but no ambient host
//! capability.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

use super::heap::{self, DictViewKind};
use super::Value;

/// The erased value exchanged by native modules and the VM: a handle bound to the runtime's
/// scope `'s`. Immediates are `PyValue<'static>` and usable in any scope.
pub(super) type PyValue<'s> = Value<'s>;

/// Opaque stable identity of a heap-backed Python value. It survives collection, unlike a
/// handle's position, so it may key memo tables and cycle checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct PyIdentity(pub(super) u32);

/// Result of a Python runtime operation. `'s` is the scope of any value it carries; results
/// without values ignore it.
pub(super) type PyResult<'s, T = PyValue<'s>> = Result<T, PyError>;

/// Native implementation stored directly in a binary protocol slot.
pub(super) type BinarySlotFn = for<'s> fn(
    &mut dyn PyRuntime<'s>,
    PyValue<'s>,
    PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>>;

/// Native implementation stored directly in a three-operand protocol slot.
pub(super) type TernarySlotFn = for<'s> fn(
    &mut dyn PyRuntime<'s>,
    PyValue<'s>,
    PyValue<'s>,
    PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>>;

/// Native implementation stored directly in a unary protocol slot.
pub(super) type UnarySlotFn =
    for<'s> fn(&mut dyn PyRuntime<'s>, PyValue<'s>) -> PyResult<'s, Option<PyValue<'s>>>;

/// Protocol functions attached to one opaque inline value kind.
///
/// The runtime only dispatches these functions. It does not interpret the value payload.
#[derive(Clone, Copy, Default)]
pub(super) struct ValueKindSlots {
    pub repr: Option<UnarySlotFn>,
    pub str_: Option<UnarySlotFn>,
    pub bool_: Option<UnarySlotFn>,
    pub get_item: Option<BinarySlotFn>,
    pub positive: Option<UnarySlotFn>,
    pub negative: Option<UnarySlotFn>,
    pub invert: Option<UnarySlotFn>,
    pub absolute: Option<UnarySlotFn>,
    pub add: Option<BinarySlotFn>,
    pub reflected_add: Option<BinarySlotFn>,
    pub subtract: Option<BinarySlotFn>,
    pub reflected_subtract: Option<BinarySlotFn>,
    pub multiply: Option<BinarySlotFn>,
    pub reflected_multiply: Option<BinarySlotFn>,
    pub divide: Option<BinarySlotFn>,
    pub reflected_divide: Option<BinarySlotFn>,
    pub floor_divide: Option<BinarySlotFn>,
    pub reflected_floor_divide: Option<BinarySlotFn>,
    pub remainder: Option<BinarySlotFn>,
    pub reflected_remainder: Option<BinarySlotFn>,
    pub divmod: Option<BinarySlotFn>,
    pub reflected_divmod: Option<BinarySlotFn>,
    pub power: Option<BinarySlotFn>,
    pub reflected_power: Option<BinarySlotFn>,
    pub left_shift: Option<BinarySlotFn>,
    pub reflected_left_shift: Option<BinarySlotFn>,
    pub right_shift: Option<BinarySlotFn>,
    pub reflected_right_shift: Option<BinarySlotFn>,
    pub bitwise_and: Option<BinarySlotFn>,
    pub reflected_bitwise_and: Option<BinarySlotFn>,
    pub bitwise_xor: Option<BinarySlotFn>,
    pub reflected_bitwise_xor: Option<BinarySlotFn>,
    pub bitwise_or: Option<BinarySlotFn>,
    pub reflected_bitwise_or: Option<BinarySlotFn>,
    pub equal: Option<BinarySlotFn>,
    pub not_equal: Option<BinarySlotFn>,
    pub less_than: Option<BinarySlotFn>,
    pub less_equal: Option<BinarySlotFn>,
    pub greater_than: Option<BinarySlotFn>,
    pub greater_equal: Option<BinarySlotFn>,
}

/// Static registration for a type-erased Python value with a module-owned payload.
///
/// Instances carry either an inline `u64` payload ([`PyRuntime::new_value_kind`]) or a 16-byte
/// heap payload ([`PyRuntime::new_wide_value_kind`]); one kind uses one width consistently.
/// `methods` and `getters` are installed on the registered Python type exactly like the tables of
/// a [`NativeTypeDef`]; the payload stays opaque to the runtime.
///
/// Kinds are registered in iteration order, so every kind named in `bases` must be registered
/// before the kinds that derive from it. The runtime linearizes `bases` with C3 like a class
/// statement would.
pub(super) struct ValueKindDef {
    pub name: &'static str,
    pub construct: NativeFn,
    pub slots: ValueKindSlots,
    pub methods: &'static [MethodDef],
    pub getters: &'static [GetterDef],
    /// Direct Python bases, most specific first. An empty list means `object`.
    pub bases: &'static [KindBase],
    /// Calls an instance, e.g. a NumPy ufunc, with the instance as receiver.
    pub call: Option<NativeMethodFn>,
    /// The Python number an instance stands for, such as the value of a NumPy scalar. It backs
    /// `int()`, `float()`, `complex()`, `__index__` (integers only), formatting, and equality
    /// with builtin numbers, including inside containers where no runtime is at hand.
    pub numeric: Option<KindNumericFn>,
}

/// Pure numeric view of one registered instance, given its kind and payload. Inline kinds pass
/// `[payload, 0]`; wide kinds pass both words.
pub(super) type KindNumericFn = fn(&'static ValueKindDef, [u64; 2]) -> Option<KindNumber>;

impl ValueKindDef {
    /// Whether instances are builtin `float`s too, directly or through a registered base, as
    /// NumPy's `float64` is.
    pub(super) fn is_float_subclass(&self) -> bool {
        self.bases.iter().any(|base| match base {
            KindBase::Float => true,
            KindBase::Kind(kind) => kind.is_float_subclass(),
            KindBase::Complex => false,
        })
    }

    /// Whether instances are builtin `complex`es too, as NumPy's `complex128` is.
    pub(super) fn is_complex_subclass(&self) -> bool {
        self.bases.iter().any(|base| match base {
            KindBase::Complex => true,
            KindBase::Kind(kind) => kind.is_complex_subclass(),
            KindBase::Float => false,
        })
    }
}

/// One direct base of a registered value kind.
#[derive(Clone, Copy, Debug)]
pub(super) enum KindBase {
    Kind(&'static ValueKindDef),
    Float,
    Complex,
}

/// The builtin Python number a registered value stands for.
///
/// `UInt` keeps unsigned 64-bit values above `i64::MAX` exact. Only `Int` and `UInt` support
/// `__index__`; like NumPy's `bool`, a `Bool` converts with `int()` but is not an index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum KindNumber {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Complex(f64, f64),
}

impl PartialEq for ValueKindDef {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Eq for ValueKindDef {}

impl fmt::Debug for ValueKindDef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ValueKindDef")
            .field(&self.name)
            .finish()
    }
}

/// Stable error categories produced by native Python operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PyErrorKind {
    Type,
    Value,
    ZeroDivision,
    Overflow,
    Runtime,
    Resource,
    /// A valid Python operation that shellsim does not model. It is not a Python exception:
    /// the program stops with the minimal-shim diagnostic, as for an unmodeled builtin.
    Unsupported,
    Exception(&'static str),
    /// A nested VM operation already stored the concrete Python exception.
    Raised,
    Exit(i32),
    /// Internal cooperative control flow. This must be consumed by the bytecode VM and never
    /// materialized as a Python exception.
    Suspend(crate::scheduler::WaitReason),
}

/// A structured Python error. Formatting is deferred to the VM boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PyError {
    pub kind: PyErrorKind,
    pub message: String,
}

impl PyError {
    pub fn new(kind: PyErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn type_error(message: impl Into<String>) -> Self {
        Self::new(PyErrorKind::Type, message)
    }

    pub fn value_error(message: impl Into<String>) -> Self {
        Self::new(PyErrorKind::Value, message)
    }

    pub fn overflow_error(message: impl Into<String>) -> Self {
        Self::new(PyErrorKind::Overflow, message)
    }

    pub fn zero_division_error(message: impl Into<String>) -> Self {
        Self::new(PyErrorKind::ZeroDivision, message)
    }

    pub fn runtime_error(message: impl Into<String>) -> Self {
        Self::new(PyErrorKind::Runtime, message)
    }

    pub fn resource_error(message: impl Into<String>) -> Self {
        Self::new(PyErrorKind::Resource, message)
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(PyErrorKind::Unsupported, message)
    }

    /// A catchable `NotImplementedError` for a library feature outside shellsim's subset, such
    /// as an unsupported dtype or option in NumPy or SciPy.
    pub fn not_implemented_error(message: impl Into<String>) -> Self {
        Self::new(PyErrorKind::Exception("NotImplementedError"), message)
    }

    pub fn exception(kind: &'static str, message: impl Into<String>) -> Self {
        Self::new(PyErrorKind::Exception(kind), message)
    }

    pub fn exit(status: i32) -> Self {
        Self::new(PyErrorKind::Exit(status), "Python callable requested exit")
    }

    /// Suspend a scheduler-owned native call until its modeled resource becomes ready.
    pub fn suspend(reason: crate::scheduler::WaitReason) -> Self {
        Self::new(PyErrorKind::Suspend(reason), "Python native call suspended")
    }
}

impl fmt::Display for PyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

/// Coarse storage kind used for checked native casts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PyKind {
    None,
    Bool,
    Int,
    Float,
    String,
    Bytes,
    ByteArray,
    List,
    Tuple,
    Dict,
    Set,
    Function,
    Class,
    Instance,
    Iterator,
    Generator,
    Module,
    Array,
    Complex,
    Native,
}

/// Interpreter-defined object payload kinds exposed to checked native views.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PyNativeKind {
    Regex,
    Match,
    ArgumentParser,
    RaisesContext,
    Property,
    Array,
}

/// Module-owned element type of an array, opaque to the runtime except for its storage needs.
///
/// The runtime uses `itemsize` to bounds-check views and `values` to decide whether elements are
/// packed bytes or traced Python references. `tag` belongs to the module that created the array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyArrayDtype {
    tag: u32,
    itemsize: u32,
    values: bool,
}

impl PyArrayDtype {
    /// Size in bytes of one element stored as a Python reference.
    pub const VALUE_ITEMSIZE: usize = 8;

    /// A dtype whose elements are `itemsize` packed little-endian bytes.
    pub const fn bytes(tag: u32, itemsize: u32) -> Self {
        Self {
            tag,
            itemsize,
            values: false,
        }
    }

    /// A dtype whose elements are Python references (NumPy's `object`).
    pub const fn values(tag: u32) -> Self {
        Self {
            tag,
            itemsize: Self::VALUE_ITEMSIZE as u32,
            values: true,
        }
    }

    pub const fn tag(self) -> u32 {
        self.tag
    }

    pub const fn itemsize(self) -> usize {
        self.itemsize as usize
    }

    pub const fn is_values(self) -> bool {
        self.values
    }
}

/// Owned element storage for a new array.
///
/// `Values` element `i` is addressed at byte offset `i * PyArrayDtype::VALUE_ITEMSIZE`, so views
/// use byte strides and offsets for both kinds of storage.
#[derive(Clone, Debug)]
pub(super) enum PyArrayBuffer<'s> {
    Bytes(Vec<u8>),
    Values(Vec<PyValue<'s>>),
}

impl<'s> PyArrayBuffer<'s> {
    /// Length of the addressable storage in bytes.
    pub fn byte_len(&self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::Values(values) => values.len().saturating_mul(PyArrayDtype::VALUE_ITEMSIZE),
        }
    }
}

/// Shape, byte strides, and element type of one array view over shared storage.
///
/// Strides and `offset` are always in bytes. A zero stride repeats an element, which is how
/// broadcast views are expressed; such views are created read-only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PyArrayView {
    pub dtype: PyArrayDtype,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: usize,
    pub writeable: bool,
}

/// Borrowed storage of one array. Kernels read elements at `view` byte offsets. Object
/// elements are stored references; read one with the [`PyRefs`] the read callback receives.
pub(super) enum PyArrayData<'a> {
    Bytes(&'a [u8]),
    Values(&'a [heap::Ref]),
}

/// Mutably borrowed storage of one writeable array. Object elements are stored references;
/// make one with the [`heap::Builder`] the write callback receives.
pub(super) enum PyArrayDataMut<'a> {
    Bytes(&'a mut [u8]),
    Values(&'a mut Vec<heap::Ref>),
}

/// Reads stored references lent by [`PyRuntime::read_arrays`] as handles in the runtime's
/// scope. Implemented by the VM.
pub(super) trait PyRefs<'s> {
    fn handle(&self, r: &heap::Ref) -> PyValue<'s>;
}

/// One array lent to a read callback.
pub(super) struct PyArrayRef<'a> {
    pub view: &'a PyArrayView,
    pub data: PyArrayData<'a>,
}

/// One writeable array's storage lent to a write callback.
pub(super) struct PyArrayMut<'a> {
    pub data: PyArrayDataMut<'a>,
}

/// A builtin or registered type object, as passed to `np.dtype(float)` or `np.dtype(np.int8)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PyTypeObject {
    /// A builtin type, by its Python name, e.g. `int` or `str`.
    Builtin(&'static str),
    Kind(&'static ValueKindDef),
}

/// One parameter of a Python function, in declaration order, with its default when it has one.
#[derive(Clone, Debug)]
pub(super) struct PyParameter<'s> {
    pub name: String,
    pub kind: super::bytecode::ParameterKind,
    pub default: Option<PyValue<'s>>,
}

/// A Python operator applied through the VM's full protocol, including user dunder methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PyOperator {
    Unary(super::ast::UnaryOperator),
    Binary(super::ast::BinaryOperator),
    Compare(super::ast::ComparisonOperator),
    /// The builtin `abs()`.
    Absolute,
}

/// Interpreter-owned marker values exported by compatibility modules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PyMarker {
    TypingList,
    EnumType,
    TestCaseType,
    Environment,
    Stdin,
    /// `sys.stdin.buffer`: the standard-input descriptor read as raw bytes.
    StdinBuffer,
    Stdout,
    Stderr,
    ArrayType,
}

/// Explicit access to shellsim's virtual clock and CPU-time counters.
pub(super) trait PyClock {
    fn wall_time(&self) -> PyResult<'static, f64>;
    fn wall_time_ns(&self) -> PyResult<'static, i64>;
    fn monotonic(&self) -> f64;
    fn monotonic_ns(&self) -> PyResult<'static, i64>;
    fn process_time(&self) -> f64;
    fn process_time_ns(&self) -> PyResult<'static, i64>;
    fn sleep(&mut self, seconds: f64) -> PyResult<'static, ()>;
}

/// Read-only access to the modeled process environment.
pub(super) trait PyEnvironment {
    fn get(&self, name: &str) -> Option<String>;
}

/// Fully-owned request passed across Python's single virtual-HTTP capability boundary.
pub(super) struct PyHttpRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// Fully-owned response returned by the virtual-HTTP capability.
pub(super) struct PyHttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// Explicit HTTP-only capability backed by shellsim's route table.
///
/// `None` means no route matched. Implementations must never attempt DNS, sockets, TLS, or host
/// network fallback.
pub(super) trait PyHttpClient {
    fn request(&mut self, request: PyHttpRequest) -> PyResult<'static, Option<PyHttpResponse>>;
}

/// Standard-stream disposition for one simulated child process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PyStdio {
    Inherit,
    Pipe,
    DevNull,
    MergeStdout,
}

/// Snapshot returned by a live logical process operation.
pub(super) struct PyProcessOutput {
    pub status: i32,
    pub stdout: Option<Vec<u8>>,
    pub stderr: Option<Vec<u8>>,
    pub timed_out: bool,
    /// Bytes destined for the calling Python command's inherited streams.
    pub inherited_stdout: Vec<u8>,
    pub inherited_stderr: Vec<u8>,
}

/// Result of probing a modeled process operation without driving the scheduler.
pub(super) enum PyProcessPoll<T> {
    Ready(T),
    Blocked(crate::scheduler::WaitReason),
}

/// Fully-owned request for a child that outlives the native launch call.
pub(super) struct PyProcessStartRequest {
    pub argv: Vec<String>,
    pub cwd: Option<String>,
    pub environment: Option<BTreeMap<String, String>>,
    pub stdin: PyStdio,
    pub stdout: PyStdio,
    pub stderr: PyStdio,
    /// Establish the child as leader of a new modeled process group.
    pub start_new_session: bool,
}

/// Stable logical child identity returned by a live launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyProcessHandle {
    pub pid: u32,
}

/// Result of a modeled stream read: decoded text for `sys.stdin`, raw bytes for its `.buffer`.
pub(super) enum PyStreamRead {
    Text(String),
    Bytes(Vec<u8>),
}

impl PyStreamRead {
    /// Whether the read produced no data, the exhaustion signal used by `readline`/iteration.
    pub(super) fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Bytes(bytes) => bytes.is_empty(),
        }
    }
}

/// Explicit process-launch capability. Implementations must dispatch only modeled commands and
/// VFS scripts and must never fall back to an ambient host process.
pub(super) trait PyProcessRunner {
    fn start(&mut self, request: PyProcessStartRequest) -> PyResult<'static, PyProcessHandle>;
    fn poll(&mut self, handle: PyProcessHandle) -> PyResult<'static, Option<i32>>;
    fn wait(
        &mut self,
        handle: PyProcessHandle,
        timeout_ns: Option<u64>,
    ) -> PyResult<'static, PyProcessOutput>;
    fn communicate(
        &mut self,
        handle: PyProcessHandle,
        input: Vec<u8>,
        timeout_ns: Option<u64>,
    ) -> PyResult<'static, PyProcessOutput>;
    /// Read at most `amount` bytes, or through EOF when omitted, from a captured child stream.
    fn read_pipe(
        &mut self,
        handle: PyProcessHandle,
        fd: i32,
        amount: Option<usize>,
    ) -> PyResult<'static, Vec<u8>>;
    fn try_read_pipe(
        &mut self,
        handle: PyProcessHandle,
        fd: i32,
        amount: Option<usize>,
    ) -> PyResult<'static, PyProcessPoll<Vec<u8>>>;
    /// Write bytes to a captured child stdin, cooperatively scheduling while the pipe is full.
    fn write_pipe(&mut self, handle: PyProcessHandle, input: Vec<u8>) -> PyResult<'static, usize>;
    fn try_write_pipe(
        &mut self,
        handle: PyProcessHandle,
        input: Vec<u8>,
    ) -> PyResult<'static, PyProcessPoll<usize>>;
    /// Close one parent-side captured stream endpoint.
    fn close_pipe(&mut self, handle: PyProcessHandle, fd: i32) -> PyResult<'static, ()>;
    fn send_signal(
        &mut self,
        handle: PyProcessHandle,
        signal: crate::process::Signal,
    ) -> PyResult<'static, ()>;
}

/// Metered access to shellsim's simulated filesystem.
///
/// Implementations must remain confined to the interpreter-owned VFS. Native modules receive
/// this capability explicitly so neither the bytecode VM nor stdlib facades need to know VFS
/// path, quota, or mutation-time policy.
pub(super) trait PyFilesystem {
    /// Return the current directory of the simulated Python process.
    fn current_dir(&self) -> String;
    /// Change only the simulated process directory after validating it in the VFS.
    fn change_dir(&mut self, path: &str) -> PyResult<'static, ()>;
    fn read_text(&mut self, path: &str) -> PyResult<'static, String>;
    fn write_text(&mut self, path: &str, contents: &str) -> PyResult<'static, ()>;
    fn append_text(&mut self, path: &str, contents: &str) -> PyResult<'static, usize>;
    fn read_bytes(&mut self, path: &str) -> PyResult<'static, Vec<u8>>;
    fn write_bytes(&mut self, path: &str, contents: &[u8]) -> PyResult<'static, ()>;
    fn append_bytes(&mut self, path: &str, contents: &[u8]) -> PyResult<'static, usize>;
    fn remove_file(&mut self, path: &str) -> PyResult<'static, ()>;
    fn remove_tree(&mut self, path: &str) -> PyResult<'static, ()>;
    fn rename(&mut self, source: &str, destination: &str) -> PyResult<'static, ()>;
    fn exists(&self, path: &str) -> bool;
    fn is_file(&self, path: &str) -> bool;
    fn is_dir(&self, path: &str) -> bool;
    fn is_symlink(&self, path: &str) -> bool;
    fn list_dir(&mut self, path: &str) -> PyResult<'static, Vec<String>>;
    fn metadata(&self, path: &str) -> PyResult<'static, PyFileMetadata>;
    fn mkdir(&mut self, path: &str, parents: bool, exist_ok: bool) -> PyResult<'static, ()>;
    fn glob(&mut self, pattern: &str) -> PyResult<'static, Vec<String>>;
}

/// Stable metadata fields exposed by the modeled VFS to capability-scoped stdlib facades.
pub(super) struct PyFileMetadata {
    pub mode: u32,
    pub size: usize,
}

/// Read-only metadata supplied by the `type` data descriptors.
#[derive(Clone, Copy)]
pub(super) enum TypeMetadata {
    Name,
    Module,
    Bases,
    Mro,
}

/// Marker handed to `PyRuntime::nested` closures. Its well-formedness implies `'s: 'c`, which
/// is what lets a closure use the parent scope's handles inside the child scope.
pub(super) type PyScope<'c, 's> = std::marker::PhantomData<&'c &'s ()>;

/// Callback lent the storage of borrowed arrays by [`PyRuntime::read_arrays`]. Object elements
/// arrive as stored references and become handles through the [`PyRefs`] argument.
pub(super) type PyArrayReader<'a, 's> =
    dyn FnMut(&dyn PyRefs<'s>, &[PyArrayRef<'_>]) -> PyResult<'s, ()> + 'a;

/// Runtime value protocols and explicitly modeled services available to native modules.
pub(super) trait PyRuntime<'s> {
    fn reserve_memory(&mut self, bytes: usize) -> PyResult<'s, ()>;
    fn charge_cpu(&mut self, units: u64) -> PyResult<'s, ()>;
    fn kind(&self, value: &PyValue<'s>) -> PyResult<'s, PyKind>;
    /// The Python class of a value, including a user class's metaclass.
    fn class_of(&self, value: &PyValue<'s>) -> PyResult<'s, PyValue<'s>>;
    /// A class, module or instance's live Python namespace, when it has one.
    fn dictionary_of(&mut self, value: PyValue<'s>) -> PyResult<'s, Option<PyValue<'s>>>;
    fn type_metadata(
        &mut self,
        value: PyValue<'s>,
        field: TypeMetadata,
    ) -> PyResult<'s, Option<PyValue<'s>>>;
    fn native_kind(&self, value: &PyValue<'s>) -> PyResult<'s, Option<PyNativeKind>>;
    fn identity(&self, value: &PyValue<'s>) -> Option<PyIdentity>;
    /// Whether two values are the same object or the same immediate (Python's `is`).
    fn identical(&self, left: &PyValue<'s>, right: &PyValue<'s>) -> bool;
    /// Run `f` in a child handle scope whose values are released when it returns. Use it for
    /// loops whose iteration count is not bounded by an existing container.
    ///
    /// The closure's second argument is a marker that proves the child scope `'c` is shorter than
    /// the parent scope `'s`, so outer `PyValue<'s>` handles coerce to `PyValue<'c>` inside it.
    fn nested(
        &mut self,
        f: &mut dyn for<'c> FnMut(&mut dyn PyRuntime<'c>, PyScope<'c, 's>) -> PyResult<'c, ()>,
    ) -> PyResult<'s, ()>;
    fn int_value(&self, value: &PyValue<'s>) -> Option<i64>;
    fn string_value(&self, value: &PyValue<'s>) -> PyResult<'s, Option<String>>;
    fn bytes_value(&self, value: &PyValue<'s>) -> PyResult<'s, Option<Vec<u8>>>;
    fn bytearray_items(&mut self, value: PyByteArray<'s>) -> PyResult<'s, Vec<u8>>;
    fn replace_bytearray_items(
        &mut self,
        value: PyByteArray<'s>,
        items: Vec<u8>,
    ) -> PyResult<'s, ()>;
    fn is_integer_type(&self, value: &PyValue<'s>) -> bool;
    fn is_string_type(&self, value: &PyValue<'s>) -> bool;
    /// Return whether `value` is the `Ellipsis` singleton that the `...` literal evaluates to.
    ///
    /// Native subscript handlers need this because `array[..., 0]` reaches `__getitem__` as the
    /// tuple `(Ellipsis, 0)`, and the singleton has no other checked view.
    fn is_ellipsis(&self, value: &PyValue<'s>) -> bool;
    /// The `NotImplemented` singleton, which a comparison or arithmetic method returns to let
    /// the other operand answer.
    fn not_implemented(&self) -> PyValue<'s>;
    fn is_not_implemented(&self, value: &PyValue<'s>) -> bool;
    /// View a builtin `bool`, `int`, `float`, or `complex` without copying its storage.
    fn number(&self, value: &PyValue<'s>) -> Option<super::number::NumberRef<'_>>;
    /// Compare builtin payloads without entering Python rich-comparison slots again.
    fn physical_compare(
        &self,
        left: &PyValue<'s>,
        right: &PyValue<'s>,
    ) -> PyResult<'s, super::protocol::Comparison>;
    /// Allocate a builtin `complex` on the metered heap.
    fn new_complex(&mut self, real: f64, imag: f64) -> PyResult<'s, PyValue<'s>>;
    /// The exact value of a builtin integer, or `None` for any other value.
    fn integer_bigint(&self, value: &PyValue<'s>) -> PyResult<'s, Option<num_bigint::BigInt>>;
    /// Write only to an interpreter-owned simulated stream marker.
    fn write_stream(&mut self, stream: &PyValue<'s>, text: &str) -> PyResult<'s, usize>;
    /// Read from the invocation's modeled standard-input stream, incrementally through the
    /// process's fd 0. `sys.stdin` decodes text; `sys.stdin.buffer` returns raw bytes. A read that
    /// would block on an empty, not-yet-closed pipe suspends the calling process (see
    /// [`PyError::suspend`]) rather than draining the producer eagerly.
    fn read_stream(
        &mut self,
        stream: &PyValue<'s>,
        size: Option<usize>,
        line: bool,
    ) -> PyResult<'s, PyStreamRead>;
    fn truth(&mut self, value: &PyValue<'s>) -> PyResult<'s, bool>;
    fn display(&mut self, value: &PyValue<'s>) -> PyResult<'s, String>;
    fn repr(&mut self, value: &PyValue<'s>) -> PyResult<'s, String>;
    /// The base object's identity representation, without dispatching to an override.
    fn default_object_repr(&self, value: &PyValue<'s>) -> PyResult<'s, String>;
    /// The physical builtin length, without invoking a user-defined `__len__` slot.
    fn physical_length(&self, value: PyValue<'s>) -> PyResult<'s, Option<usize>>;
    /// Render through the same bounded formatting protocol used by f-strings.
    fn format_value(
        &mut self,
        value: &PyValue<'s>,
        conversion: Option<char>,
        specification: &str,
    ) -> PyResult<'s, String>;
    /// Format a builtin payload directly, without calling its `__format__` slot again.
    fn builtin_format(&mut self, value: &PyValue<'s>, specification: &str) -> PyResult<'s, String>;
    /// Dispatch `reversed` through the type slot, then through the sequence protocol.
    fn reverse_value(&mut self, value: PyValue<'s>) -> PyResult<'s, PyValue<'s>>;
    /// Reverse a builtin sequence directly for its native `__reversed__` slot.
    fn reverse_builtin_sequence(&mut self, value: PyValue<'s>) -> PyResult<'s, PyValue<'s>>;
    /// Create a metered generic alias for a builtin container's class subscription.
    fn new_generic_alias(
        &mut self,
        origin: PyValue<'s>,
        item: PyValue<'s>,
    ) -> PyResult<'s, PyValue<'s>>;
    fn equals(&mut self, left: &PyValue<'s>, right: &PyValue<'s>) -> PyResult<'s, bool>;
    fn compare(&mut self, left: &PyValue<'s>, right: &PyValue<'s>) -> PyResult<'s, Ordering>;
    /// `left < right` through the rich-comparison protocol. Sorting asks only this, as CPython
    /// does, so a user `__lt__` runs once per comparison instead of twice.
    fn less_than(&mut self, left: &PyValue<'s>, right: &PyValue<'s>) -> PyResult<'s, bool>;
    /// Resolve an attribute through the runtime's descriptor and MRO protocol.
    fn get_attribute(
        &mut self,
        value: PyValue<'s>,
        name: &str,
    ) -> PyResult<'s, Option<PyValue<'s>>>;
    /// Invoke the default object lookup directly, without `__getattr__` fallback.
    fn get_attribute_default(
        &mut self,
        value: PyValue<'s>,
        name: &str,
    ) -> PyResult<'s, Option<PyValue<'s>>>;
    /// Assign an attribute through the runtime's descriptor protocol, as `setattr` does.
    fn set_attribute(
        &mut self,
        value: PyValue<'s>,
        name: &str,
        item: PyValue<'s>,
    ) -> PyResult<'s, ()>;
    /// Assign an attribute without consulting a class's `__setattr__`, as `object.__setattr__`
    /// does: data descriptors still apply, and other names go to the instance.
    fn set_attribute_default(
        &mut self,
        value: PyValue<'s>,
        name: &str,
        item: PyValue<'s>,
    ) -> PyResult<'s, ()>;
    /// Raise the builtin exception `kind` with constructor arguments `args`, such as the key of
    /// a `KeyError`, and return the error that carries it.
    fn exception_with_args(&mut self, kind: &'static str, args: Vec<PyValue<'s>>) -> PyError;
    /// Raise the `StopIteration` that ends `generator`, carrying its return value.
    fn generator_stop(&mut self, generator: PyIterator<'s>) -> PyError;
    /// The builtin exception class and constructor arguments of an exception instance. For an
    /// instance of a user exception class, the class is its closest builtin ancestor.
    fn exception_args(
        &mut self,
        value: &PyValue<'s>,
    ) -> PyResult<'s, Option<(String, Vec<PyValue<'s>>)>>;
    /// Delete an attribute without consulting a class's `__delattr__`, as
    /// `object.__delattr__` does.
    fn delete_attribute_default(&mut self, value: PyValue<'s>, name: &str) -> PyResult<'s, ()>;
    fn list_len(&self, list: PyList<'s>) -> PyResult<'s, usize>;
    fn list_items(&mut self, list: PyList<'s>) -> PyResult<'s, Vec<PyValue<'s>>>;
    fn list_append(&mut self, list: PyList<'s>, value: PyValue<'s>) -> PyResult<'s, ()>;
    fn list_insert(
        &mut self,
        list: PyList<'s>,
        index: usize,
        value: PyValue<'s>,
    ) -> PyResult<'s, ()>;
    fn list_extend(&mut self, list: PyList<'s>, values: Vec<PyValue<'s>>) -> PyResult<'s, ()>;
    fn list_pop(&mut self, list: PyList<'s>, index: usize) -> PyResult<'s, PyValue<'s>>;
    fn list_position(
        &mut self,
        list: PyList<'s>,
        needle: &PyValue<'s>,
        start: usize,
        stop: usize,
    ) -> PyResult<'s, Option<usize>>;
    fn list_reverse(&mut self, list: PyList<'s>) -> PyResult<'s, ()>;
    fn list_clear(&mut self, list: PyList<'s>) -> PyResult<'s, ()>;
    fn tuple_items(&mut self, tuple: PyTuple<'s>) -> PyResult<'s, Vec<PyValue<'s>>>;
    /// The indices a slice selects with, each bound read through `__index__`, or `None` when
    /// `value` is not a slice. Raises `TypeError` for a bound that is not an index.
    fn slice_parts(
        &mut self,
        value: &PyValue<'s>,
    ) -> PyResult<'s, Option<super::slice::SliceBounds>>;
    fn dict_items(&mut self, dict: PyDict<'s>) -> PyResult<'s, Vec<(PyValue<'s>, PyValue<'s>)>>;
    fn dict_get(
        &mut self,
        dict: PyDict<'s>,
        key: &PyValue<'s>,
    ) -> PyResult<'s, Option<PyValue<'s>>>;
    fn dict_insert(
        &mut self,
        dict: PyDict<'s>,
        key: PyValue<'s>,
        value: PyValue<'s>,
    ) -> PyResult<'s, ()>;
    fn dict_remove(
        &mut self,
        dict: PyDict<'s>,
        key: &PyValue<'s>,
    ) -> PyResult<'s, Option<PyValue<'s>>>;
    /// The most recently inserted key of `dict`, or `None` when it is empty.
    fn dict_last_key(&mut self, dict: PyDict<'s>) -> PyResult<'s, Option<PyValue<'s>>>;
    fn replace_dict_items(
        &mut self,
        dict: PyDict<'s>,
        items: Vec<(PyValue<'s>, PyValue<'s>)>,
    ) -> PyResult<'s, ()>;
    /// A shallow copy of `dict` of the same kind: a `defaultdict` copy keeps its factory.
    fn dict_copy(&mut self, dict: PyDict<'s>) -> PyResult<'s, PyValue<'s>>;
    /// `container[key]`, running the container's `__getitem__` or builtin subscript.
    fn get_item(&mut self, container: PyValue<'s>, key: PyValue<'s>) -> PyResult<'s, PyValue<'s>>;
    /// Index a builtin payload without reentering its `__getitem__` slot.
    fn builtin_get_item(
        &mut self,
        container: PyValue<'s>,
        key: PyValue<'s>,
    ) -> PyResult<'s, PyValue<'s>>;
    /// Membership in a builtin payload without reentering its `__contains__` slot.
    fn builtin_contains(&mut self, container: PyValue<'s>, item: PyValue<'s>)
        -> PyResult<'s, bool>;
    /// The `(key, value)` entries of `value` if it is a mapping, or `None` when it has no
    /// `keys` method. Mappings other than dicts are read through `keys()` and `__getitem__`, as
    /// `dict(m)` and `f(**m)` read them in CPython.
    fn mapping_items(
        &mut self,
        value: PyValue<'s>,
    ) -> PyResult<'s, Option<Vec<(PyValue<'s>, PyValue<'s>)>>>;
    /// A live `keys()`, `values()` or `items()` view of `mapping`, which is a dict, a namespace
    /// view or a mapping proxy.
    fn new_dict_view(
        &mut self,
        kind: DictViewKind,
        mapping: PyValue<'s>,
    ) -> PyResult<'s, PyValue<'s>>;
    /// The projection and viewed mapping of a dict view, or `None` when `value` is not one.
    fn dict_view(&self, value: &PyValue<'s>) -> PyResult<'s, Option<(DictViewKind, PyValue<'s>)>>;
    fn set_items(&mut self, set: PySet<'s>) -> PyResult<'s, Vec<PyValue<'s>>>;
    fn set_is_frozen(&self, set: PySet<'s>) -> PyResult<'s, bool>;
    fn set_insert(&mut self, set: PySet<'s>, value: PyValue<'s>) -> PyResult<'s, bool>;
    fn set_remove(&mut self, set: PySet<'s>, value: &PyValue<'s>) -> PyResult<'s, bool>;
    /// The earliest inserted member of `set`, or `None` when it is empty.
    fn set_first(&mut self, set: PySet<'s>) -> PyResult<'s, Option<PyValue<'s>>>;
    /// Replace a mutable set's members with already-deduplicated `items`, metering the change
    /// in retained size. Frozen sets are rejected.
    fn replace_set_items(&mut self, set: PySet<'s>, items: Vec<PyValue<'s>>) -> PyResult<'s, ()>;
    fn replace_list_items(&mut self, list: PyList<'s>, items: Vec<PyValue<'s>>)
        -> PyResult<'s, ()>;
    fn call_value(
        &mut self,
        callable: PyValue<'s>,
        args: CallArgs<'s>,
    ) -> PyResult<'s, PyValue<'s>>;
    /// Invoke a class through the default `type.__call__` path, bypassing a metaclass override.
    fn call_type_default(
        &mut self,
        class: PyValue<'s>,
        args: CallArgs<'s>,
    ) -> PyResult<'s, PyValue<'s>>;
    fn is_callable(&self, value: &PyValue<'s>) -> PyResult<'s, bool>;
    /// Whether `value` is an iterator: a builtin iterator or generator, or an object whose class
    /// defines `__next__`.
    fn is_iterator(&self, value: &PyValue<'s>) -> PyResult<'s, bool>;
    /// Whether a native iterator is known to be unbounded and cannot be collected into memory.
    fn is_unbounded_iterator(&self, value: &PyValue<'s>) -> PyResult<'s, bool>;
    /// `iter(value)`: an iterator is returned as is; any other iterable produces one.
    fn iterator(&mut self, value: PyValue<'s>) -> PyResult<'s, PyIterator<'s>>;
    /// The next item of `iterator`, or `None` once it is exhausted.
    fn iterator_next(&mut self, iterator: PyIterator<'s>) -> PyResult<'s, Option<PyValue<'s>>>;
    fn generator_send(
        &mut self,
        generator: PyIterator<'s>,
        value: PyValue<'s>,
    ) -> PyResult<'s, Option<PyValue<'s>>>;
    fn generator_return_value(&self, generator: PyIterator<'s>) -> PyResult<'s, PyValue<'s>>;
    /// Resume a coroutine and return 0 for yield, 1 for return, or 2 for an exception.
    fn coroutine_step(
        &mut self,
        coroutine: PyIterator<'s>,
        value: PyValue<'s>,
    ) -> PyResult<'s, (u8, PyValue<'s>)>;
    fn generator_close(&mut self, generator: PyIterator<'s>) -> PyResult<'s, ()>;
    fn generator_throw(
        &mut self,
        generator: PyIterator<'s>,
        exception: PyValue<'s>,
    ) -> PyResult<'s>;
    fn new_iterator(&mut self, values: Vec<PyValue<'s>>) -> PyResult<'s, PyValue<'s>>;
    fn new_count_iterator(&mut self, start: i64, step: i64) -> PyResult<'s, PyValue<'s>>;
    fn new_default_dict(&mut self, factory: PyCallable<'s>) -> PyResult<'s, PyValue<'s>>;
    fn new_list(&mut self, items: Vec<PyValue<'s>>) -> PyResult<'s, PyValue<'s>>;
    fn new_tuple(&mut self, items: Vec<PyValue<'s>>) -> PyResult<'s, PyValue<'s>>;
    fn new_dict(&mut self, items: Vec<(PyValue<'s>, PyValue<'s>)>) -> PyResult<'s, PyValue<'s>>;
    fn new_set(&mut self, items: Vec<PyValue<'s>>) -> PyResult<'s, PyValue<'s>>;
    fn new_frozen_set(&mut self, items: Vec<PyValue<'s>>) -> PyResult<'s, PyValue<'s>>;
    fn new_value_kind(
        &self,
        kind: &'static ValueKindDef,
        payload: u64,
    ) -> PyResult<'s, PyValue<'s>>;
    fn value_kind_payload(&self, value: &PyValue<'s>, kind: &'static ValueKindDef) -> Option<u64>;
    /// Allocate a value of `kind` whose payload needs 16 bytes, such as a complex128 scalar.
    fn new_wide_value_kind(
        &mut self,
        kind: &'static ValueKindDef,
        payload: [u64; 2],
    ) -> PyResult<'s, PyValue<'s>>;
    fn wide_value_kind_payload(
        &self,
        value: &PyValue<'s>,
        kind: &'static ValueKindDef,
    ) -> Option<[u64; 2]>;
    /// The registered kind of `value`, for inline and wide values alike.
    fn value_kind_of(&self, value: &PyValue<'s>) -> Option<&'static ValueKindDef>;
    /// `value` itself as a builtin or registered type object, if it is one.
    fn type_object(&self, value: &PyValue<'s>) -> Option<PyTypeObject>;
    /// The builtin type object with this Python name, such as `str` or `float`.
    fn builtin_type(&self, name: &str) -> Option<PyValue<'s>>;
    fn value_kind_type(&self, kind: &'static ValueKindDef) -> PyResult<'s, PyValue<'s>>;
    /// Allocate an array that owns `buffer`, with its elements at `strides`. The buffer length
    /// must equal the element count times the dtype's item size, object dtypes need `Values`
    /// storage, and every addressed element must lie inside the buffer.
    fn new_array(
        &mut self,
        buffer: PyArrayBuffer<'s>,
        dtype: PyArrayDtype,
        shape: Vec<usize>,
        strides: Vec<isize>,
    ) -> PyResult<'s, PyValue<'s>>;
    /// Allocate another view of `base`'s storage after checking that it stays inside the
    /// storage and matches its element kind. A view of a read-only array stays read-only.
    fn new_array_view(&mut self, base: PyArray<'s>, view: PyArrayView)
        -> PyResult<'s, PyValue<'s>>;
    fn array_view(&self, array: PyArray<'s>) -> PyResult<'s, PyArrayView>;
    /// Identity of the storage behind `array`; views of one buffer share it.
    fn array_storage(&self, array: PyArray<'s>) -> PyResult<'s, PyIdentity>;
    /// The array that owns `array`'s storage, or `None` when `array` owns it.
    fn array_base(&self, array: PyArray<'s>) -> PyResult<'s, Option<PyValue<'s>>>;
    fn set_array_writeable(&mut self, array: PyArray<'s>, writeable: bool) -> PyResult<'s, ()>;
    /// Lend the storage of `arrays` to `read`. The callback cannot reach the runtime, so it
    /// cannot run Python code while the storage is borrowed.
    fn read_arrays(
        &self,
        arrays: &[PyArray<'s>],
        read: &mut PyArrayReader<'_, 's>,
    ) -> PyResult<'s, ()>;
    /// Lend the storage of one writeable array to `write`. Read-only arrays raise NumPy's
    /// `ValueError: assignment destination is read-only`.
    fn write_array(
        &mut self,
        array: PyArray<'s>,
        write: &mut dyn FnMut(&heap::Builder<'_>, PyArrayMut<'_>) -> PyResult<'s, ()>,
    ) -> PyResult<'s, ()>;
    /// Apply a Python operator with the VM's complete protocol, including user dunders.
    fn apply_operator(
        &mut self,
        operator: PyOperator,
        operands: &[PyValue<'s>],
    ) -> PyResult<'s, PyValue<'s>>;
    fn new_string(&mut self, value: String) -> PyResult<'s, PyValue<'s>>;
    fn new_bytes(&mut self, value: Vec<u8>) -> PyResult<'s, PyValue<'s>>;
    fn new_bytearray(&mut self, value: Vec<u8>) -> PyResult<'s, PyValue<'s>>;
    fn property_getter(&self, property: PyProperty<'s>) -> PyResult<'s, PyValue<'s>>;
    fn new_property(
        &mut self,
        getter: PyValue<'s>,
        setter: Option<PyValue<'s>>,
    ) -> PyResult<'s, PyValue<'s>>;
    /// The builtin value an instance of a subclass of a builtin type such as `tuple` holds.
    fn builtin_payload(&self, value: &PyValue<'s>) -> PyResult<'s, Option<PyValue<'s>>>;
    /// `builtin.__new__(class, ...)`, such as `tuple.__new__(cls, iterable)`: the builtin value,
    /// held by a new instance of `class` when it is a subclass of `builtin`.
    fn new_builtin_instance(
        &mut self,
        builtin: super::object_model::BuiltinType,
        class: PyValue<'s>,
        args: CallArgs<'s>,
    ) -> PyResult<'s, PyValue<'s>>;
    /// `object.__new__(class, ...)`: a new, attribute-free instance of `class`. `has_arguments`
    /// says whether the call passed anything beyond the class.
    fn new_instance(
        &mut self,
        class: PyValue<'s>,
        has_arguments: bool,
    ) -> PyResult<'s, PyValue<'s>>;
    /// Allocate a class through the runtime's single `type.__new__` implementation.
    fn new_type(
        &mut self,
        metaclass: PyValue<'s>,
        name: String,
        bases: PyValue<'s>,
        namespace: PyValue<'s>,
    ) -> PyResult<'s, PyValue<'s>>;
    /// Parse and allocate a Python integer without imposing an immediate-width limit.
    fn new_integer(&mut self, decimal: &str) -> PyResult<'s, PyValue<'s>>;
    /// An `int` holding `value`, immediate when it fits in `i64`.
    fn new_bigint(&mut self, value: num_bigint::BigInt) -> PyResult<'s, PyValue<'s>>;
    fn new_regex(&mut self, pattern: String, flags: u32) -> PyResult<'s, PyValue<'s>>;
    fn new_match(
        &mut self,
        text: String,
        groups: Vec<Option<String>>,
        group_names: Vec<Option<String>>,
        start: usize,
        end: usize,
    ) -> PyResult<'s, PyValue<'s>>;
    fn regex_parts(&mut self, regex: PyRegex<'s>) -> PyResult<'s, (String, u32)>;
    fn match_data(&mut self, matched: PyMatch<'s>) -> PyResult<'s, PyMatchData>;
    fn marker(&self, marker: PyMarker) -> PyValue<'s>;
    fn mark_dataclass(&mut self, class: PyClass<'s>) -> PyResult<'s, ()>;
    fn argv0(&self) -> String;
    fn new_argv(&mut self) -> PyResult<'s, PyValue<'s>>;
    /// Return the mutable list consulted for subsequent VFS module imports.
    fn new_import_path(&mut self) -> PyResult<'s, PyValue<'s>>;
    /// Import one module through the VM's closed native, frozen, and VFS lookup rules.
    fn import_module(&mut self, name: &str) -> PyResult<'s, PyValue<'s>>;
    fn new_argument_parser(
        &mut self,
        program: String,
        description: Option<String>,
        add_help: bool,
        is_subcommand: bool,
    ) -> PyResult<'s, PyValue<'s>>;
    fn argument_parser_parts(
        &mut self,
        parser: PyArgumentParser<'s>,
    ) -> PyResult<'s, PyArgumentParserData<'s>>;
    fn append_argument(
        &mut self,
        parser: PyArgumentParser<'s>,
        argument: PyArgumentSpec<'s>,
    ) -> PyResult<'s, ()>;
    fn configure_subparsers(
        &mut self,
        parser: PyArgumentParser<'s>,
        subparsers: PySubparsersSpec<'s>,
    ) -> PyResult<'s, ()>;
    fn append_subcommand(
        &mut self,
        parser: PyArgumentParser<'s>,
        command: PySubcommandSpec<'s>,
    ) -> PyResult<'s, ()>;
    fn command_arguments(&self) -> Vec<String>;
    /// Allocate an empty VM module whose globals are isolated from the caller.
    fn new_module(
        &mut self,
        name: String,
        path: String,
        spec: PyValue<'s>,
        loader: PyValue<'s>,
    ) -> PyResult<'s, PyValue<'s>>;
    /// Execute one bounded VFS source file in an existing VM module namespace.
    fn exec_module(&mut self, module: PyModule<'s>, path: &str) -> PyResult<'s, ()>;
    fn new_namespace(&mut self, values: Vec<(String, PyValue<'s>)>) -> PyResult<'s, PyValue<'s>>;
    fn new_raises_context(&mut self, expected: String) -> PyResult<'s, PyValue<'s>>;
    fn raises_expected(&self, context: PyRaisesContext<'s>) -> PyResult<'s, String>;
    fn exception_type_name(&self, value: &PyValue<'s>) -> Option<&'static str>;
    /// Return one interpreter-owned exception class from the runtime's closed type table.
    fn exception_type(&self, name: &'static str) -> PyValue<'s>;
    /// Suspend the enclosing logical process until any modeled resource becomes ready.
    fn wait_on(&mut self, reasons: Vec<crate::scheduler::WaitReason>) -> PyResult<'s, ()>;
    fn clock(&mut self) -> &mut dyn PyClock;
    fn environment(&self) -> &dyn PyEnvironment;
    fn filesystem(&mut self) -> &mut dyn PyFilesystem;
    fn http(&mut self) -> &mut dyn PyHttpClient;
    fn processes(&mut self) -> &mut dyn PyProcessRunner;
    /// The script path and source line executing `depth` Python frames below the innermost one,
    /// or `None` when the stack is shallower. Depth 0 is the innermost frame.
    fn caller_location(&self, depth: usize) -> Option<(String, u32)>;
    /// The `__name__` of the module whose code runs `depth` Python frames below the innermost
    /// one, as `sys._getframemodulename` reports it, or `None` when the stack is shallower.
    fn frame_module_name(&mut self, depth: usize) -> PyResult<'s, Option<PyValue<'s>>>;
    /// The parameters of a Python function, or of the function a bound method wraps without its
    /// bound first parameter. `None` for any other callable.
    fn function_parameters(
        &self,
        value: &PyValue<'s>,
    ) -> PyResult<'s, Option<Vec<PyParameter<'s>>>>;
    /// PID of the logical process running this Python interpreter.
    fn current_pid(&self) -> u32;
    /// PID of the logical parent of the process running this Python interpreter.
    fn current_ppid(&self) -> u32;
    /// Deliver a signal to an arbitrary modeled process, not only an owned subprocess handle.
    fn send_os_signal(&mut self, pid: u32, signal: crate::process::Signal) -> PyResult<'s, ()>;

    /// The name CPython prints for a value's type in error messages, such as `int`,
    /// `float32` or a user class name.
    fn type_name(&self, value: &PyValue<'s>) -> PyResult<'s, String>;
}

/// Checked handle to an interpreter-owned array view.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyArray<'s>(PyValue<'s>);

impl<'s> FromPyValue<'s> for PyArray<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        if !value.is_object() {
            return Err(PyError::type_error("expected numpy.ndarray"));
        }
        if runtime.native_kind(&value)? == Some(PyNativeKind::Array) {
            Ok(Self(value))
        } else {
            Err(PyError::type_error("expected numpy.ndarray"))
        }
    }
}

impl<'s> PyArray<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }
}

/// Conversion from an erased Python value into a checked native view.
pub(super) trait FromPyValue<'s>: Sized {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self>;
}

pub(super) trait PyValueCast<'s> {
    fn cast<T: FromPyValue<'s>>(self, runtime: &dyn PyRuntime<'s>) -> PyResult<'s, T>;
}

impl<'s> PyValueCast<'s> for PyValue<'s> {
    fn cast<T: FromPyValue<'s>>(self, runtime: &dyn PyRuntime<'s>) -> PyResult<'s, T> {
        T::from_py_value(runtime, self)
    }
}

/// Owned string extracted from an erased value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct OwnedPyString(pub String);

impl<'s> FromPyValue<'s> for OwnedPyString {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        if let Some(value) = runtime.string_value(&value)? {
            Ok(Self(value))
        } else {
            let actual = runtime.type_name(&value)?;
            Err(PyError::type_error(format!(
                "expected a string, got {actual}"
            )))
        }
    }
}

/// Owned byte sequence extracted from an erased value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PyBytes(pub Vec<u8>);

impl<'s> FromPyValue<'s> for PyBytes {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        if let Some(value) = runtime.bytes_value(&value)? {
            Ok(Self(value))
        } else {
            let actual = runtime.type_name(&value)?;
            Err(PyError::type_error(format!(
                "expected a bytes-like object, got {actual}"
            )))
        }
    }
}

/// Checked handle to a mutable byte array.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyByteArray<'s>(PyValue<'s>);

impl<'s> FromPyValue<'s> for PyByteArray<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        let payload = runtime.builtin_payload(&value)?.unwrap_or(value);
        if !payload.is_object() {
            return Err(PyError::type_error("expected a bytearray"));
        }
        if runtime.kind(&payload)? == PyKind::ByteArray {
            Ok(Self(payload))
        } else {
            Err(PyError::type_error("expected a bytearray"))
        }
    }
}

impl<'s> PyByteArray<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }
}

/// Checked handle to one of the exception classes modeled by the VM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyExceptionType(pub &'static str);

impl<'s> FromPyValue<'s> for PyExceptionType {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        runtime
            .exception_type_name(&value)
            .map(Self)
            .ok_or_else(|| PyError::type_error("expected an exception type"))
    }
}

/// Checked handle to a compiled regular-expression object.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyRegex<'s>(PyValue<'s>);

impl<'s> PyRegex<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }
}

impl<'s> FromPyValue<'s> for PyRegex<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        if !value.is_object() {
            return Err(PyError::type_error("expected a compiled regex"));
        }
        if runtime.native_kind(&value)? == Some(PyNativeKind::Regex) {
            Ok(Self(value))
        } else {
            Err(PyError::type_error("expected a compiled regex"))
        }
    }
}

/// Checked handle to a regular-expression match object.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyMatch<'s>(PyValue<'s>);

impl<'s> PyMatch<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }
}

impl<'s> FromPyValue<'s> for PyMatch<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        if !value.is_object() {
            return Err(PyError::type_error("expected a regex match"));
        }
        if runtime.native_kind(&value)? == Some(PyNativeKind::Match) {
            Ok(Self(value))
        } else {
            Err(PyError::type_error("expected a regex match"))
        }
    }
}

/// Owned, metered snapshot of a match payload.
pub(super) struct PyMatchData {
    pub groups: Vec<Option<String>>,
    pub group_names: Vec<Option<String>>,
    pub start: usize,
    pub end: usize,
}

/// Owned definition of one bounded ``argparse`` argument.
#[derive(Clone, Debug)]
pub(super) struct PyArgumentSpec<'s> {
    pub names: Vec<String>,
    pub dest: String,
    pub required: bool,
    pub default: PyValue<'s>,
    pub store_true: bool,
    pub store_false: bool,
    pub integer: bool,
    pub choices: Vec<PyValue<'s>>,
    pub help: Option<String>,
}

/// One command registered on an ``argparse`` subparser collection.
#[derive(Clone, Debug)]
pub(super) struct PySubcommandSpec<'s> {
    pub name: String,
    pub help: Option<String>,
    pub parser: PyArgumentParser<'s>,
}

/// Owned definition of the deliberately one-level subparser surface.
#[derive(Clone, Debug)]
pub(super) struct PySubparsersSpec<'s> {
    pub dest: Option<String>,
    pub required: bool,
    pub help: Option<String>,
    pub commands: Vec<PySubcommandSpec<'s>>,
}

/// Metered snapshot of an interpreter-owned argument parser.
#[derive(Clone, Debug)]
pub(super) struct PyArgumentParserData<'s> {
    pub prog: String,
    pub description: Option<String>,
    pub add_help: bool,
    pub arguments: Vec<PyArgumentSpec<'s>>,
    pub subparsers: Option<PySubparsersSpec<'s>>,
}

/// Checked handle to an interpreter-owned argument parser.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyArgumentParser<'s>(PyValue<'s>);

impl<'s> PyArgumentParser<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }
}

impl<'s> FromPyValue<'s> for PyArgumentParser<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        if !value.is_object() {
            return Err(PyError::type_error("expected ArgumentParser"));
        }
        if runtime.native_kind(&value)? == Some(PyNativeKind::ArgumentParser) {
            Ok(Self(value))
        } else {
            Err(PyError::type_error("expected ArgumentParser"))
        }
    }
}

/// Checked handle to an interpreter-owned Python module.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyModule<'s>(PyValue<'s>);

impl<'s> PyModule<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }
}

impl<'s> FromPyValue<'s> for PyModule<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        if !value.is_object() {
            return Err(PyError::type_error("expected module"));
        }
        if runtime.kind(&value)? == PyKind::Module {
            Ok(Self(value))
        } else {
            Err(PyError::type_error("expected module"))
        }
    }
}

/// Checked handle to a context returned by ``pytest.raises``.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyRaisesContext<'s>(PyValue<'s>);

impl<'s> PyRaisesContext<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }
}

impl<'s> FromPyValue<'s> for PyRaisesContext<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        if !value.is_object() {
            return Err(PyError::type_error("expected pytest.raises context"));
        }
        if runtime.native_kind(&value)? == Some(PyNativeKind::RaisesContext) {
            Ok(Self(value))
        } else {
            Err(PyError::type_error("expected pytest.raises context"))
        }
    }
}

/// Checked handle to a heap-backed Python list.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyList<'s>(PyValue<'s>);

impl<'s> FromPyValue<'s> for PyList<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        let payload = runtime.builtin_payload(&value)?.unwrap_or(value);
        if !payload.is_object() {
            let actual = runtime.type_name(&value)?;
            return Err(PyError::type_error(format!("expected list, got {actual}")));
        }
        if runtime.kind(&payload)? == PyKind::List {
            Ok(Self(payload))
        } else {
            let actual = runtime.type_name(&payload)?;
            Err(PyError::type_error(format!("expected list, got {actual}")))
        }
    }
}

impl<'s> PyList<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }

    /// Snapshot list items so callers may allocate or invoke protocols while iterating.
    pub fn items(self, runtime: &mut dyn PyRuntime<'s>) -> PyResult<'s, Vec<PyValue<'s>>> {
        runtime.list_items(self)
    }
}

/// Checked handle to a heap-backed Python tuple.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyTuple<'s>(PyValue<'s>);

impl<'s> FromPyValue<'s> for PyTuple<'s> {
    /// Accepts a tuple, or an instance of a `tuple` subclass through the tuple it holds.
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        let tuple = runtime.builtin_payload(&value)?.unwrap_or(value);
        match tuple {
            _ if tuple.is_object() && runtime.kind(&tuple)? == PyKind::Tuple => Ok(Self(tuple)),
            _ => {
                let actual = runtime.type_name(&value)?;
                Err(PyError::type_error(format!("expected tuple, got {actual}")))
            }
        }
    }
}

impl<'s> PyTuple<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }

    pub fn items(self, runtime: &mut dyn PyRuntime<'s>) -> PyResult<'s, Vec<PyValue<'s>>> {
        runtime.tuple_items(self)
    }
}

/// Checked view over the sequence kinds accepted by small stdlib algorithms.
#[derive(Clone, Copy, Debug)]
pub(super) enum PySequence<'s> {
    List(PyList<'s>),
    Tuple(PyTuple<'s>),
}

impl<'s> FromPyValue<'s> for PySequence<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        match runtime.kind(&value)? {
            PyKind::List => value.cast(runtime).map(Self::List),
            PyKind::Tuple => value.cast(runtime).map(Self::Tuple),
            kind => Err(PyError::type_error(format!(
                "expected a sequence, got {kind:?}"
            ))),
        }
    }
}

impl<'s> PySequence<'s> {
    pub fn items(self, runtime: &mut dyn PyRuntime<'s>) -> PyResult<'s, Vec<PyValue<'s>>> {
        match self {
            Self::List(list) => list.items(runtime),
            Self::Tuple(tuple) => tuple.items(runtime),
        }
    }
}

/// Integer accepted by operations requiring Python's index protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyIndex(pub i64);

impl<'s> FromPyValue<'s> for PyIndex {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        if let Some(value) = runtime.int_value(&value) {
            return Ok(Self(value));
        }
        let actual = runtime.type_name(&value)?;
        Err(PyError::type_error(format!(
            "expected an integer index, got {actual}"
        )))
    }
}

/// Checked handle to one of the runtime's lazy iterator representations.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyIterator<'s>(PyValue<'s>);

impl<'s> PyIterator<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }
}

/// Checked callable value. Invocation stays on the runtime so modules do not inspect functions.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyCallable<'s>(PyValue<'s>);

impl<'s> FromPyValue<'s> for PyCallable<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        if runtime.is_callable(&value)? {
            Ok(Self(value))
        } else {
            let actual = runtime.type_name(&value)?;
            Err(PyError::type_error(format!(
                "expected a callable, got {actual}"
            )))
        }
    }
}

impl<'s> PyCallable<'s> {
    pub fn call(self, runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
        runtime.call_value(self.0, args)
    }

    pub(super) fn into_value(self) -> PyValue<'s> {
        self.0
    }
}

impl<'s> FromPyValue<'s> for PyIterator<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        match value {
            _ if value.is_object() && runtime.is_iterator(&value)? => Ok(Self(value)),
            _ => Err(PyError::type_error("expected an iterator")),
        }
    }
}

/// Checked handle to a heap-backed Python dictionary.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyDict<'s>(PyValue<'s>);

impl<'s> FromPyValue<'s> for PyDict<'s> {
    /// Accepts a dict, or an instance of a `dict` subclass through the dict it holds.
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        let dict = runtime.builtin_payload(&value)?.unwrap_or(value);
        match dict {
            _ if dict.is_object() && runtime.kind(&dict)? == PyKind::Dict => Ok(Self(dict)),
            _ => {
                let actual = runtime.type_name(&value)?;
                Err(PyError::type_error(format!("expected dict, got {actual}")))
            }
        }
    }
}

impl<'s> PyDict<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }

    /// Snapshot entries so callers do not retain a heap borrow across Python work.
    pub fn items(
        self,
        runtime: &mut dyn PyRuntime<'s>,
    ) -> PyResult<'s, Vec<(PyValue<'s>, PyValue<'s>)>> {
        runtime.dict_items(self)
    }
}

/// Checked handle to a heap-backed Python set.
#[derive(Clone, Copy, Debug)]
pub(super) struct PySet<'s>(PyValue<'s>);

impl<'s> FromPyValue<'s> for PySet<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        let payload = runtime.builtin_payload(&value)?.unwrap_or(value);
        if !payload.is_object() {
            return Err(PyError::type_error("expected set"));
        }
        if runtime.kind(&payload)? == PyKind::Set {
            Ok(Self(payload))
        } else {
            Err(PyError::type_error("expected set"))
        }
    }
}

impl<'s> PySet<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }

    pub fn items(self, runtime: &mut dyn PyRuntime<'s>) -> PyResult<'s, Vec<PyValue<'s>>> {
        runtime.set_items(self)
    }
}

/// Checked handle to the standard `property` descriptor payload.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyProperty<'s>(PyValue<'s>);

impl<'s> FromPyValue<'s> for PyProperty<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        if !value.is_object() {
            return Err(PyError::type_error("expected property"));
        }
        if runtime.native_kind(&value)? == Some(PyNativeKind::Property) {
            Ok(Self(value))
        } else {
            Err(PyError::type_error("expected property"))
        }
    }
}

impl<'s> PyProperty<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }
}

/// Checked handle to a heap-backed Python class.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyClass<'s>(PyValue<'s>);

impl<'s> PyClass<'s> {
    pub fn value(self) -> PyValue<'s> {
        self.0
    }
}

impl<'s> FromPyValue<'s> for PyClass<'s> {
    fn from_py_value(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Self> {
        if !value.is_object() {
            return Err(PyError::type_error("expected a class"));
        }
        if runtime.kind(&value)? == PyKind::Class {
            Ok(Self(value))
        } else {
            Err(PyError::type_error("expected a class"))
        }
    }
}

/// Owned arguments passed through the uniform native-call ABI.
#[derive(Clone)]
pub(super) struct CallArgs<'s> {
    positional: Vec<PyValue<'s>>,
    keywords: Vec<(String, PyValue<'s>)>,
}

impl<'s> CallArgs<'s> {
    pub fn new(positional: Vec<PyValue<'s>>, keywords: Vec<(String, PyValue<'s>)>) -> Self {
        Self {
            positional,
            keywords,
        }
    }

    pub fn positional(&self) -> &[PyValue<'s>] {
        &self.positional
    }

    pub fn keywords(&self) -> &[(String, PyValue<'s>)] {
        &self.keywords
    }

    pub fn into_parts(self) -> (Vec<PyValue<'s>>, Vec<(String, PyValue<'s>)>) {
        (self.positional, self.keywords)
    }

    pub fn expect_positional(
        &self,
        function: &str,
        minimum: usize,
        maximum: usize,
    ) -> PyResult<'s, ()> {
        if (minimum..=maximum).contains(&self.positional.len()) {
            Ok(())
        } else {
            Err(PyError::type_error(format!(
                "{function}() expected {minimum}..={maximum} positional arguments, got {}",
                self.positional.len()
            )))
        }
    }

    pub fn reject_keywords(&self, function: &str) -> PyResult<'s, ()> {
        if self.keywords.is_empty() {
            Ok(())
        } else {
            Err(PyError::type_error(format!(
                "{function}() does not accept keyword arguments"
            )))
        }
    }

    pub fn keyword(&self, function: &str, name: &str) -> PyResult<'s, Option<&PyValue<'s>>> {
        let mut found = None;
        for (candidate, value) in &self.keywords {
            if candidate == name {
                if found.is_some() {
                    return Err(PyError::type_error(format!(
                        "{function}() got multiple values for keyword {name:?}"
                    )));
                }
                found = Some(value);
            }
        }
        Ok(found)
    }

    pub fn reject_unknown_keywords(&self, function: &str, names: &[&str]) -> PyResult<'s, ()> {
        if let Some((name, _)) = self
            .keywords
            .iter()
            .find(|(name, _)| !names.contains(&name.as_str()))
        {
            Err(PyError::type_error(format!(
                "{function}() keyword argument {name:?} is not implemented"
            )))
        } else {
            Ok(())
        }
    }
}

/// Uniform implementation type for capability-free native functions.
pub(super) type NativeFn = for<'s> fn(&mut dyn PyRuntime<'s>, CallArgs<'s>) -> PyResult<'s>;

/// Uniform implementation type for a native method after descriptor binding.
pub(super) type NativeMethodFn =
    for<'s> fn(&mut dyn PyRuntime<'s>, PyValue<'s>, CallArgs<'s>) -> PyResult<'s>;

/// Declarative native function definition stored in a module table.
pub(super) struct FunctionDef {
    pub module: &'static str,
    pub name: &'static str,
    pub call: NativeFn,
}

/// One method stored on an interpreter-defined native type.
pub(super) struct MethodDef {
    pub type_name: &'static str,
    pub name: &'static str,
    pub call: NativeMethodFn,
}

impl fmt::Debug for MethodDef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "<native method {}.{}>",
            self.type_name, self.name
        )
    }
}

impl PartialEq for MethodDef {
    fn eq(&self, other: &Self) -> bool {
        self.type_name == other.type_name && self.name == other.name
    }
}

impl Eq for MethodDef {}

/// Uniform implementation type for a native data attribute read from its receiver.
pub(super) type NativeGetterFn = for<'s> fn(&mut dyn PyRuntime<'s>, PyValue<'s>) -> PyResult<'s>;

/// A read-only data attribute computed from its receiver, e.g. `int.real` or `ndarray.shape`.
///
/// Getters are data descriptors: attribute lookup on an instance calls `get` with the instance
/// before consulting any instance attributes, lookup through the type object returns the
/// descriptor itself, and assignment raises `AttributeError`. There are no native setters.
pub(super) struct GetterDef {
    /// Python type name used in descriptor reprs and error messages.
    pub owner: &'static str,
    pub name: &'static str,
    pub get: NativeGetterFn,
}

impl fmt::Debug for GetterDef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "<attribute '{}' of '{}' objects>",
            self.name, self.owner
        )
    }
}

impl PartialEq for GetterDef {
    fn eq(&self, other: &Self) -> bool {
        self.owner == other.owner && self.name == other.name
    }
}

impl Eq for GetterDef {}

/// Method and data-attribute tables for an interpreter-defined native object type.
pub(super) struct NativeTypeDef {
    pub name: &'static str,
    pub methods: &'static [MethodDef],
    pub getters: &'static [GetterDef],
}

/// Static values supported directly by declarative module definitions.
pub(super) enum PyConstant {
    Int(i64),
    Float(f64),
    String(&'static str),
}

/// A module value, either a scalar constant or a runtime-constructed marker/type.
pub(super) enum ValueDef {
    Constant {
        name: &'static str,
        value: PyConstant,
    },
    Factory {
        name: &'static str,
        get: for<'s> fn(&mut dyn PyRuntime<'s>) -> PyResult<'s>,
    },
    /// An inline registered value, such as a NumPy ufunc or `np.True_`.
    Registered {
        name: &'static str,
        kind: &'static ValueKindDef,
        payload: u64,
    },
}

impl ValueDef {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Constant { name, .. }
            | Self::Factory { name, .. }
            | Self::Registered { name, .. } => name,
        }
    }

    pub fn get<'s>(&self, runtime: &mut dyn PyRuntime<'s>) -> PyResult<'s> {
        match self {
            Self::Constant { value, .. } => Ok(match value {
                PyConstant::Int(value) => Value::Int(*value),
                PyConstant::Float(value) => Value::Float(*value),
                PyConstant::String(value) => return runtime.new_string((*value).to_string()),
            }),
            Self::Factory { get, .. } => get(runtime),
            Self::Registered { kind, payload, .. } => runtime.new_value_kind(kind, *payload),
        }
    }
}

impl fmt::Debug for FunctionDef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "<native function {}.{}>", self.module, self.name)
    }
}

impl PartialEq for FunctionDef {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module && self.name == other.name
    }
}

impl Eq for FunctionDef {}

/// Declarative native module definition.
pub(super) struct ModuleDef {
    pub name: &'static str,
    pub functions: &'static [FunctionDef],
    pub values: &'static [ValueDef],
}

impl fmt::Debug for ModuleDef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "<native module {}>", self.name)
    }
}

impl PartialEq for ModuleDef {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for ModuleDef {}

impl ModuleDef {
    pub fn function(&'static self, name: &str) -> Option<&'static FunctionDef> {
        self.functions.iter().find(|function| function.name == name)
    }

    pub fn value(&'static self, name: &str) -> Option<&'static ValueDef> {
        self.values.iter().find(|value| value.name() == name)
    }
}
