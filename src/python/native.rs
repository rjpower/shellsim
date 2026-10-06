//! Uniform value, argument, error, and runtime interfaces for native Python modules.
//!
//! Native functions are type-erased at the call boundary: every function accepts [`CallArgs`]
//! and returns a [`PyValue`]. Implementations recover small checked views such as [`PyNumber`] or
//! [`PyList`] locally. The runtime trait exposes allocation and metering, but no ambient host
//! capability.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

use super::ast::ComparisonOperator;
use super::heap::{self, DictViewKind};
use super::Value;

/// The value exchanged by native modules and the VM. A heap value is pinned by the runtime
/// scope that produced it and must be stored in a root to outlive that scope.
pub(super) type PyValue = Value;

/// Opaque identity of a live heap-backed Python value. Objects never move, so it is stable for
/// the object's lifetime and may key memo tables and cycle checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct PyIdentity(pub(super) u32);

/// Result of a Python runtime operation. `'s` is the scope of any value it carries; results
/// without values ignore it.
pub(super) use super::error::{PyError, PyErrorKind, PyResult};

/// Native implementation of a type's six rich comparisons, stored in each comparison slot. The
/// operands are the builtin values the receiver and argument stand for; `None` means the type
/// does not compare with that operand (`NotImplemented`), so the reflected slot is tried next.
pub(super) type CompareSlotFn =
    fn(&mut dyn PyRuntime, PyValue, PyValue, ComparisonOperator) -> PyResult<Option<bool>>;

/// Native implementation stored directly in a binary protocol slot.
pub(super) type BinarySlotFn =
    fn(&mut dyn PyRuntime, PyValue, PyValue) -> PyResult<Option<PyValue>>;

/// Native implementation stored directly in a three-operand protocol slot.
pub(super) type TernarySlotFn =
    fn(&mut dyn PyRuntime, PyValue, PyValue, PyValue) -> PyResult<Option<PyValue>>;

/// Native implementation stored directly in a unary protocol slot.
pub(super) type UnarySlotFn = fn(&mut dyn PyRuntime, PyValue) -> PyResult<Option<PyValue>>;

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
pub(super) enum PyArrayBuffer {
    Bytes(Vec<u8>),
    Values(Vec<PyValue>),
}

impl PyArrayBuffer {
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
/// make one from a value with `Ref::from`.
pub(super) enum PyArrayDataMut<'a> {
    Bytes(&'a mut [u8]),
    Values(&'a mut Vec<heap::Ref>),
}

/// Reads stored references lent by [`PyRuntime::read_arrays`] as values pinned in the
/// runtime's scope. Implemented by the VM.
pub(super) trait PyRefs {
    fn value(&self, r: &heap::Ref) -> PyValue;
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
pub(super) struct PyParameter {
    pub name: String,
    pub kind: super::bytecode::ParameterKind,
    pub default: Option<PyValue>,
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
    fn wall_time(&self) -> PyResult<f64>;
    fn wall_time_ns(&self) -> PyResult<i64>;
    fn monotonic(&self) -> f64;
    fn monotonic_ns(&self) -> PyResult<i64>;
    fn process_time(&self) -> f64;
    fn process_time_ns(&self) -> PyResult<i64>;
    fn sleep(&mut self, seconds: f64) -> PyResult<()>;
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
    fn request(&mut self, request: PyHttpRequest) -> PyResult<Option<PyHttpResponse>>;
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
    fn start(&mut self, request: PyProcessStartRequest) -> PyResult<PyProcessHandle>;
    fn poll(&mut self, handle: PyProcessHandle) -> PyResult<Option<i32>>;
    fn wait(
        &mut self,
        handle: PyProcessHandle,
        timeout_ns: Option<u64>,
    ) -> PyResult<PyProcessOutput>;
    fn communicate(
        &mut self,
        handle: PyProcessHandle,
        input: Vec<u8>,
        timeout_ns: Option<u64>,
    ) -> PyResult<PyProcessOutput>;
    /// Read at most `amount` bytes, or through EOF when omitted, from a captured child stream.
    fn read_pipe(
        &mut self,
        handle: PyProcessHandle,
        fd: i32,
        amount: Option<usize>,
    ) -> PyResult<Vec<u8>>;
    fn try_read_pipe(
        &mut self,
        handle: PyProcessHandle,
        fd: i32,
        amount: Option<usize>,
    ) -> PyResult<PyProcessPoll<Vec<u8>>>;
    /// Write bytes to a captured child stdin, cooperatively scheduling while the pipe is full.
    fn write_pipe(&mut self, handle: PyProcessHandle, input: Vec<u8>) -> PyResult<usize>;
    fn try_write_pipe(
        &mut self,
        handle: PyProcessHandle,
        input: Vec<u8>,
    ) -> PyResult<PyProcessPoll<usize>>;
    /// Close one parent-side captured stream endpoint.
    fn close_pipe(&mut self, handle: PyProcessHandle, fd: i32) -> PyResult<()>;
    fn send_signal(
        &mut self,
        handle: PyProcessHandle,
        signal: crate::process::Signal,
    ) -> PyResult<()>;
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
    fn change_dir(&mut self, path: &str) -> PyResult<()>;
    fn read_text(&mut self, path: &str) -> PyResult<String>;
    fn write_text(&mut self, path: &str, contents: &str) -> PyResult<()>;
    fn append_text(&mut self, path: &str, contents: &str) -> PyResult<usize>;
    fn read_bytes(&mut self, path: &str) -> PyResult<Vec<u8>>;
    fn write_bytes(&mut self, path: &str, contents: &[u8]) -> PyResult<()>;
    fn append_bytes(&mut self, path: &str, contents: &[u8]) -> PyResult<usize>;
    fn remove_file(&mut self, path: &str) -> PyResult<()>;
    fn remove_tree(&mut self, path: &str) -> PyResult<()>;
    fn rename(&mut self, source: &str, destination: &str) -> PyResult<()>;
    fn exists(&self, path: &str) -> bool;
    fn is_file(&self, path: &str) -> bool;
    fn is_dir(&self, path: &str) -> bool;
    fn is_symlink(&self, path: &str) -> bool;
    fn list_dir(&mut self, path: &str) -> PyResult<Vec<String>>;
    /// Metadata of `path`; `follow` resolves a final symlink as `stat` does, else as `lstat`.
    fn metadata(&self, path: &str, follow: bool) -> PyResult<PyFileMetadata>;
    fn mkdir(&mut self, path: &str, parents: bool, exist_ok: bool) -> PyResult<()>;
    fn rmdir(&mut self, path: &str) -> PyResult<()>;
    fn chmod(&mut self, path: &str, mode: u32) -> PyResult<()>;
    fn symlink(&mut self, target: &str, link_path: &str) -> PyResult<()>;
    fn read_link(&self, path: &str) -> PyResult<String>;
    /// Set the modification time to the virtual clock, creating an empty file if needed.
    fn touch(&mut self, path: &str) -> PyResult<()>;
    /// `(used, limit)` bytes of the simulated disk, for `os.statvfs` and `shutil.disk_usage`.
    fn disk_usage(&self) -> (u64, u64);
    fn glob(&mut self, pattern: &str) -> PyResult<Vec<String>>;
}

/// Stable metadata fields exposed by the modeled VFS to capability-scoped stdlib facades.
pub(super) struct PyFileMetadata {
    pub mode: u32,
    pub size: usize,
    /// Modification time on the virtual clock, in milliseconds since the epoch.
    pub mtime_ms: u64,
    pub kind: PyFileKind,
}

/// The node kinds the modeled VFS distinguishes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum PyFileKind {
    File,
    Dir,
    Symlink,
    Other,
}

/// Read-only metadata supplied by the `type` data descriptors.
#[derive(Clone, Copy)]
pub(super) enum TypeMetadata {
    Name,
    Module,
    Bases,
    Mro,
}

/// Callback lent the storage of borrowed arrays by [`PyRuntime::read_arrays`]. Object elements
/// arrive as stored references and become values through the [`PyRefs`] argument.
pub(super) type PyArrayReader<'a> = dyn FnMut(&dyn PyRefs, &[PyArrayRef<'_>]) -> PyResult<()> + 'a;

/// Runtime value protocols and explicitly modeled services available to native modules.
pub(super) trait PyRuntime {
    fn reserve_memory(&mut self, bytes: usize) -> PyResult<()>;
    fn charge_cpu(&mut self, units: u64) -> PyResult<()>;
    /// Arm or clear the runner's per-item virtual-time limit; global resource limits still apply.
    fn set_test_timeout(&mut self, nanoseconds: Option<u64>);
    fn kind(&self, value: &PyValue) -> PyResult<PyKind>;
    /// The Python class of a value, including a user class's metaclass.
    fn class_of(&self, value: &PyValue) -> PyResult<PyValue>;
    /// A class, module or instance's live Python namespace, when it has one.
    fn dictionary_of(&mut self, value: PyValue) -> PyResult<Option<PyValue>>;
    fn type_metadata(&mut self, value: PyValue, field: TypeMetadata) -> PyResult<Option<PyValue>>;
    fn native_kind(&self, value: &PyValue) -> PyResult<Option<PyNativeKind>>;
    fn identity(&self, value: &PyValue) -> Option<PyIdentity>;
    /// Whether two values are the same object or the same immediate (Python's `is`).
    fn identical(&self, left: &PyValue, right: &PyValue) -> bool;
    /// Run `f` in a child scope whose pinned values are released when it returns. Use it for
    /// loops whose iteration count is not bounded by an existing container. A value made inside
    /// `f` must not be used after it returns unless it was stored in a root.
    fn nested(&mut self, f: &mut dyn FnMut(&mut dyn PyRuntime) -> PyResult<()>) -> PyResult<()>;
    fn int_value(&self, value: &PyValue) -> Option<i64>;
    fn string_value(&self, value: &PyValue) -> PyResult<Option<String>>;
    fn bytes_value(&self, value: &PyValue) -> PyResult<Option<Vec<u8>>>;
    fn bytearray_items(&mut self, value: PyByteArray) -> PyResult<Vec<u8>>;
    fn replace_bytearray_items(&mut self, value: PyByteArray, items: Vec<u8>) -> PyResult<()>;
    fn is_integer_type(&self, value: &PyValue) -> bool;
    fn is_string_type(&self, value: &PyValue) -> bool;
    /// Return whether `value` is the `Ellipsis` singleton that the `...` literal evaluates to.
    ///
    /// Native subscript handlers need this because `array[..., 0]` reaches `__getitem__` as the
    /// tuple `(Ellipsis, 0)`, and the singleton has no other checked view.
    fn is_ellipsis(&self, value: &PyValue) -> bool;
    /// The `NotImplemented` singleton, which a comparison or arithmetic method returns to let
    /// the other operand answer.
    fn not_implemented(&self) -> PyValue;
    fn is_not_implemented(&self, value: &PyValue) -> bool;
    /// View a builtin `bool`, `int`, `float`, or `complex` without copying its storage.
    fn number(&self, value: &PyValue) -> Option<super::number::NumberRef<'_>>;
    /// Compare builtin payloads without entering Python rich-comparison slots again.
    fn physical_compare(
        &self,
        left: &PyValue,
        right: &PyValue,
    ) -> PyResult<super::protocol::Comparison>;
    /// Structural equality of builtin payloads without entering rich-comparison slots: the
    /// `==` of ranges, slices and generic aliases, whose parts are compared as values.
    fn physical_equals(&self, left: &PyValue, right: &PyValue) -> PyResult<bool>;
    /// The comparison slot of the builtin containers: lists and tuples order element by
    /// element, and lists, tuples, dicts and sets compare equal element by element. `None`
    /// when the operands are not containers of one kind.
    fn container_compare(
        &mut self,
        operator: ComparisonOperator,
        left: &PyValue,
        right: &PyValue,
    ) -> PyResult<Option<bool>>;
    /// Allocate a builtin `complex` on the metered heap.
    fn new_complex(&mut self, real: f64, imag: f64) -> PyResult<PyValue>;
    /// The exact value of a builtin integer, or `None` for any other value.
    fn integer_bigint(&self, value: &PyValue) -> PyResult<Option<num_bigint::BigInt>>;
    /// Write only to an interpreter-owned simulated stream marker.
    fn write_stream(&mut self, stream: &PyValue, text: &str) -> PyResult<usize>;
    /// Read from the invocation's modeled standard-input stream, incrementally through the
    /// process's fd 0. `sys.stdin` decodes text; `sys.stdin.buffer` returns raw bytes. A read that
    /// would block on an empty, not-yet-closed pipe suspends the calling process (see
    /// [`PyError::suspend`]) rather than draining the producer eagerly.
    fn read_stream(
        &mut self,
        stream: &PyValue,
        size: Option<usize>,
        line: bool,
    ) -> PyResult<PyStreamRead>;
    fn truth(&mut self, value: &PyValue) -> PyResult<bool>;
    fn display(&mut self, value: &PyValue) -> PyResult<String>;
    fn repr(&mut self, value: &PyValue) -> PyResult<String>;
    /// The builtin `repr()` of `value`'s payload, ignoring any `__repr__` its class defines: what
    /// `tuple.__repr__(instance)` renders when a subclass's `__repr__` delegates to it.
    fn payload_repr(&mut self, value: &PyValue) -> PyResult<String>;
    /// The base object's identity representation, without dispatching to an override.
    fn default_object_repr(&self, value: &PyValue) -> PyResult<String>;
    /// The physical builtin length, without invoking a user-defined `__len__` slot.
    fn physical_length(&self, value: PyValue) -> PyResult<Option<usize>>;
    /// Render through the same bounded formatting protocol used by f-strings.
    fn format_value(
        &mut self,
        value: &PyValue,
        conversion: Option<char>,
        specification: &str,
    ) -> PyResult<String>;
    /// Format a builtin payload directly, without calling its `__format__` slot again.
    fn builtin_format(&mut self, value: &PyValue, specification: &str) -> PyResult<String>;
    /// Dispatch `reversed` through the type slot, then through the sequence protocol.
    fn reverse_value(&mut self, value: PyValue) -> PyResult<PyValue>;
    /// Reverse a builtin sequence directly for its native `__reversed__` slot.
    fn reverse_builtin_sequence(&mut self, value: PyValue) -> PyResult<PyValue>;
    /// Create a metered generic alias for a builtin container's class subscription.
    fn new_generic_alias(&mut self, origin: PyValue, item: PyValue) -> PyResult<PyValue>;
    fn equals(&mut self, left: &PyValue, right: &PyValue) -> PyResult<bool>;
    fn compare(&mut self, left: &PyValue, right: &PyValue) -> PyResult<Ordering>;
    /// `left < right` through the rich-comparison protocol. Sorting asks only this, as CPython
    /// does, so a user `__lt__` runs once per comparison instead of twice.
    fn less_than(&mut self, left: &PyValue, right: &PyValue) -> PyResult<bool>;
    /// Resolve an attribute through the runtime's descriptor and MRO protocol.
    fn get_attribute(&mut self, value: PyValue, name: &str) -> PyResult<Option<PyValue>>;
    /// Invoke the default object lookup directly, without `__getattr__` fallback.
    fn get_attribute_default(&mut self, value: PyValue, name: &str) -> PyResult<Option<PyValue>>;
    /// Assign an attribute through the runtime's descriptor protocol, as `setattr` does.
    fn set_attribute(&mut self, value: PyValue, name: &str, item: PyValue) -> PyResult<()>;
    /// Assign an attribute without consulting a class's `__setattr__`, as `object.__setattr__`
    /// does: data descriptors still apply, and other names go to the instance.
    fn set_attribute_default(&mut self, value: PyValue, name: &str, item: PyValue) -> PyResult<()>;
    /// Raise the builtin exception `kind` with constructor arguments `args`, such as the key of
    /// a `KeyError`, and return the error that carries it.
    fn exception_with_args(&mut self, kind: &'static str, args: Vec<PyValue>) -> PyError;
    /// Raise the `StopIteration` that ends `generator`, carrying its return value.
    fn generator_stop(&mut self, generator: PyIterator) -> PyError;
    /// The builtin exception class and constructor arguments of an exception instance. For an
    /// instance of a user exception class, the class is its closest builtin ancestor.
    fn exception_args(&mut self, value: &PyValue) -> PyResult<Option<(String, Vec<PyValue>)>>;
    /// Delete an attribute without consulting a class's `__delattr__`, as
    /// `object.__delattr__` does.
    fn delete_attribute_default(&mut self, value: PyValue, name: &str) -> PyResult<()>;
    fn list_len(&self, list: PyList) -> PyResult<usize>;
    fn list_items(&mut self, list: PyList) -> PyResult<Vec<PyValue>>;
    fn list_append(&mut self, list: PyList, value: PyValue) -> PyResult<()>;
    fn list_insert(&mut self, list: PyList, index: usize, value: PyValue) -> PyResult<()>;
    fn list_extend(&mut self, list: PyList, values: Vec<PyValue>) -> PyResult<()>;
    fn list_pop(&mut self, list: PyList, index: usize) -> PyResult<PyValue>;
    fn list_position(
        &mut self,
        list: PyList,
        needle: &PyValue,
        start: usize,
        stop: usize,
    ) -> PyResult<Option<usize>>;
    fn list_reverse(&mut self, list: PyList) -> PyResult<()>;
    fn list_clear(&mut self, list: PyList) -> PyResult<()>;
    fn tuple_items(&mut self, tuple: PyTuple) -> PyResult<Vec<PyValue>>;
    /// The indices a slice selects with, each bound read through `__index__`, or `None` when
    /// `value` is not a slice. Raises `TypeError` for a bound that is not an index.
    fn slice_parts(&mut self, value: &PyValue) -> PyResult<Option<super::slice::SliceBounds>>;
    fn dict_items(&mut self, dict: PyDict) -> PyResult<Vec<(PyValue, PyValue)>>;
    fn dict_get(&mut self, dict: PyDict, key: &PyValue) -> PyResult<Option<PyValue>>;
    fn dict_insert(&mut self, dict: PyDict, key: PyValue, value: PyValue) -> PyResult<()>;
    fn dict_remove(&mut self, dict: PyDict, key: &PyValue) -> PyResult<Option<PyValue>>;
    /// The most recently inserted key of `dict`, or `None` when it is empty.
    fn dict_last_key(&mut self, dict: PyDict) -> PyResult<Option<PyValue>>;
    fn replace_dict_items(&mut self, dict: PyDict, items: Vec<(PyValue, PyValue)>) -> PyResult<()>;
    /// A shallow copy of `dict` of the same kind: a `defaultdict` copy keeps its factory.
    fn dict_copy(&mut self, dict: PyDict) -> PyResult<PyValue>;
    /// `container[key]`, running the container's `__getitem__` or builtin subscript.
    fn get_item(&mut self, container: PyValue, key: PyValue) -> PyResult<PyValue>;
    /// Index a builtin payload without reentering its `__getitem__` slot.
    fn builtin_get_item(&mut self, container: PyValue, key: PyValue) -> PyResult<PyValue>;
    /// Membership in a builtin payload without reentering its `__contains__` slot.
    fn builtin_contains(&mut self, container: PyValue, item: PyValue) -> PyResult<bool>;
    /// The `(key, value)` entries of `value` if it is a mapping, or `None` when it has no
    /// `keys` method. Mappings other than dicts are read through `keys()` and `__getitem__`, as
    /// `dict(m)` and `f(**m)` read them in CPython.
    fn mapping_items(&mut self, value: PyValue) -> PyResult<Option<Vec<(PyValue, PyValue)>>>;
    /// A live `keys()`, `values()` or `items()` view of `mapping`, which is a dict, a namespace
    /// view or a mapping proxy.
    fn new_dict_view(&mut self, kind: DictViewKind, mapping: PyValue) -> PyResult<PyValue>;
    /// The projection and viewed mapping of a dict view, or `None` when `value` is not one.
    fn dict_view(&self, value: &PyValue) -> PyResult<Option<(DictViewKind, PyValue)>>;
    fn set_items(&mut self, set: PySet) -> PyResult<Vec<PyValue>>;
    fn set_is_frozen(&self, set: PySet) -> PyResult<bool>;
    fn set_insert(&mut self, set: PySet, value: PyValue) -> PyResult<bool>;
    fn set_remove(&mut self, set: PySet, value: &PyValue) -> PyResult<bool>;
    /// The earliest inserted member of `set`, or `None` when it is empty.
    fn set_first(&mut self, set: PySet) -> PyResult<Option<PyValue>>;
    /// Replace a mutable set's members with already-deduplicated `items`, metering the change
    /// in retained size. Frozen sets are rejected.
    fn replace_set_items(&mut self, set: PySet, items: Vec<PyValue>) -> PyResult<()>;
    fn replace_list_items(&mut self, list: PyList, items: Vec<PyValue>) -> PyResult<()>;
    fn call_value(&mut self, callable: PyValue, args: CallArgs) -> PyResult<PyValue>;
    /// Invoke a class through the default `type.__call__` path, bypassing a metaclass override.
    fn call_type_default(&mut self, class: PyValue, args: CallArgs) -> PyResult<PyValue>;
    fn is_callable(&self, value: &PyValue) -> PyResult<bool>;
    /// Whether `value` is an iterator: a builtin iterator or generator, or an object whose class
    /// defines `__next__`.
    fn is_iterator(&self, value: &PyValue) -> PyResult<bool>;
    /// Whether a native iterator is known to be unbounded and cannot be collected into memory.
    fn is_unbounded_iterator(&self, value: &PyValue) -> PyResult<bool>;
    /// `iter(value)`: an iterator is returned as is; any other iterable produces one.
    fn iterator(&mut self, value: PyValue) -> PyResult<PyIterator>;
    /// An iterator over `value`'s builtin payload, ignoring any `__iter__` its class defines, or
    /// `None` when the payload is not a builtin iterable. The builtin `__iter__` slots use it.
    fn payload_iterator(&mut self, value: PyValue) -> PyResult<Option<PyValue>>;
    /// Whether `value` is an instance of a user-defined class, whatever payload it carries. A
    /// named tuple is a tuple to [`PyRuntime::kind`] and a user instance here.
    fn is_user_instance(&self, value: &PyValue) -> PyResult<bool>;
    /// The next item of `iterator`, or `None` once it is exhausted.
    fn iterator_next(&mut self, iterator: PyIterator) -> PyResult<Option<PyValue>>;
    fn generator_send(
        &mut self,
        generator: PyIterator,
        value: PyValue,
    ) -> PyResult<Option<PyValue>>;
    fn generator_return_value(&self, generator: PyIterator) -> PyResult<PyValue>;
    /// Resume a coroutine and return 0 for yield, 1 for return, or 2 for an exception.
    fn coroutine_step(&mut self, coroutine: PyIterator, value: PyValue) -> PyResult<(u8, PyValue)>;
    fn generator_close(&mut self, generator: PyIterator) -> PyResult<()>;
    fn generator_throw(&mut self, generator: PyIterator, exception: PyValue) -> PyResult;
    fn new_iterator(&mut self, values: Vec<PyValue>) -> PyResult<PyValue>;
    fn new_count_iterator(&mut self, start: i64, step: i64) -> PyResult<PyValue>;
    fn new_default_dict(&mut self, factory: PyCallable) -> PyResult<PyValue>;
    fn new_list(&mut self, items: Vec<PyValue>) -> PyResult<PyValue>;
    fn new_tuple(&mut self, items: Vec<PyValue>) -> PyResult<PyValue>;
    fn new_dict(&mut self, items: Vec<(PyValue, PyValue)>) -> PyResult<PyValue>;
    fn new_set(&mut self, items: Vec<PyValue>) -> PyResult<PyValue>;
    fn new_frozen_set(&mut self, items: Vec<PyValue>) -> PyResult<PyValue>;
    fn new_value_kind(&self, kind: &'static ValueKindDef, payload: u64) -> PyResult<PyValue>;
    fn value_kind_payload(&self, value: &PyValue, kind: &'static ValueKindDef) -> Option<u64>;
    /// Allocate a value of `kind` whose payload needs 16 bytes, such as a complex128 scalar.
    fn new_wide_value_kind(
        &mut self,
        kind: &'static ValueKindDef,
        payload: [u64; 2],
    ) -> PyResult<PyValue>;
    fn wide_value_kind_payload(
        &self,
        value: &PyValue,
        kind: &'static ValueKindDef,
    ) -> Option<[u64; 2]>;
    /// The registered kind of `value`, for inline and wide values alike.
    fn value_kind_of(&self, value: &PyValue) -> Option<&'static ValueKindDef>;
    /// `value` itself as a builtin or registered type object, if it is one.
    fn type_object(&self, value: &PyValue) -> Option<PyTypeObject>;
    /// The builtin type object with this Python name, such as `str` or `float`.
    fn builtin_type(&self, name: &str) -> Option<PyValue>;
    fn value_kind_type(&self, kind: &'static ValueKindDef) -> PyResult<PyValue>;
    /// Allocate an array that owns `buffer`, with its elements at `strides`. The buffer length
    /// must equal the element count times the dtype's item size, object dtypes need `Values`
    /// storage, and every addressed element must lie inside the buffer.
    fn new_array(
        &mut self,
        buffer: PyArrayBuffer,
        dtype: PyArrayDtype,
        shape: Vec<usize>,
        strides: Vec<isize>,
    ) -> PyResult<PyValue>;
    /// Allocate another view of `base`'s storage after checking that it stays inside the
    /// storage and matches its element kind. A view of a read-only array stays read-only.
    fn new_array_view(&mut self, base: PyArray, view: PyArrayView) -> PyResult<PyValue>;
    fn array_view(&self, array: PyArray) -> PyResult<PyArrayView>;
    /// Identity of the storage behind `array`; views of one buffer share it.
    fn array_storage(&self, array: PyArray) -> PyResult<PyIdentity>;
    /// The array that owns `array`'s storage, or `None` when `array` owns it.
    fn array_base(&self, array: PyArray) -> PyResult<Option<PyValue>>;
    fn set_array_writeable(&mut self, array: PyArray, writeable: bool) -> PyResult<()>;
    /// Lend the storage of `arrays` to `read`. The callback cannot reach the runtime, so it
    /// cannot run Python code while the storage is borrowed.
    fn read_arrays(&self, arrays: &[PyArray], read: &mut PyArrayReader<'_>) -> PyResult<()>;
    /// Lend the storage of one writeable array to `write`. Read-only arrays raise NumPy's
    /// `ValueError: assignment destination is read-only`.
    fn write_array(
        &mut self,
        array: PyArray,
        write: &mut dyn FnMut(PyArrayMut<'_>) -> PyResult<()>,
    ) -> PyResult<()>;
    /// Apply a Python operator with the VM's complete protocol, including user dunders.
    fn apply_operator(&mut self, operator: PyOperator, operands: &[PyValue]) -> PyResult<PyValue>;
    fn new_string(&mut self, value: String) -> PyResult<PyValue>;
    fn new_bytes(&mut self, value: Vec<u8>) -> PyResult<PyValue>;
    fn new_bytearray(&mut self, value: Vec<u8>) -> PyResult<PyValue>;
    fn property_getter(&self, property: PyProperty) -> PyResult<PyValue>;
    /// The setter of a property, or `None` for a read-only one.
    fn property_setter(&self, property: PyProperty) -> PyResult<Option<PyValue>>;
    fn new_property(&mut self, getter: PyValue, setter: Option<PyValue>) -> PyResult<PyValue>;
    /// The builtin value an instance of a subclass of a builtin type such as `tuple` holds.
    /// `builtin.__new__(class, ...)`, such as `tuple.__new__(cls, iterable)`: the builtin value,
    /// held by a new instance of `class` when it is a subclass of `builtin`.
    fn new_builtin_instance(
        &mut self,
        builtin: super::object_model::BuiltinType,
        class: PyValue,
        args: CallArgs,
    ) -> PyResult<PyValue>;
    /// `object.__new__(class, ...)`: a new, attribute-free instance of `class`. `has_arguments`
    /// says whether the call passed anything beyond the class.
    fn new_instance(&mut self, class: PyValue, has_arguments: bool) -> PyResult<PyValue>;
    /// Allocate a class through the runtime's single `type.__new__` implementation.
    fn new_type(
        &mut self,
        metaclass: PyValue,
        name: String,
        bases: PyValue,
        namespace: PyValue,
    ) -> PyResult<PyValue>;
    /// Parse and allocate a Python integer without imposing an immediate-width limit.
    fn new_integer(&mut self, decimal: &str) -> PyResult<PyValue>;
    /// An `int` holding `value`, immediate when it fits in `i64`.
    fn new_bigint(&mut self, value: num_bigint::BigInt) -> PyResult<PyValue>;
    fn new_regex(&mut self, pattern: String, flags: u32) -> PyResult<PyValue>;
    fn new_match(&mut self, data: PyMatchData) -> PyResult<PyValue>;
    fn regex_parts(&mut self, regex: PyRegex) -> PyResult<(String, u32)>;
    fn match_data(&mut self, matched: PyMatch) -> PyResult<PyMatchData>;
    fn marker(&self, marker: PyMarker) -> PyValue;
    fn mark_dataclass(&mut self, class: PyClass) -> PyResult<()>;
    fn argv0(&self) -> String;
    fn new_argv(&mut self) -> PyResult<PyValue>;
    /// Return the mutable list consulted for subsequent VFS module imports.
    fn new_import_path(&mut self) -> PyResult<PyValue>;
    /// Import one module through the VM's closed native, frozen, and VFS lookup rules.
    fn import_module(&mut self, name: &str) -> PyResult<PyValue>;
    fn new_argument_parser(
        &mut self,
        program: String,
        description: Option<String>,
        add_help: bool,
        is_subcommand: bool,
    ) -> PyResult<PyValue>;
    fn argument_parser_parts(&mut self, parser: PyArgumentParser)
        -> PyResult<PyArgumentParserData>;
    fn append_argument(
        &mut self,
        parser: PyArgumentParser,
        argument: PyArgumentSpec,
    ) -> PyResult<()>;
    fn configure_subparsers(
        &mut self,
        parser: PyArgumentParser,
        subparsers: PySubparsersSpec,
    ) -> PyResult<()>;
    fn append_subcommand(
        &mut self,
        parser: PyArgumentParser,
        command: PySubcommandSpec,
    ) -> PyResult<()>;
    fn command_arguments(&self) -> Vec<String>;
    /// Allocate an empty VM module whose globals are isolated from the caller.
    fn new_module(
        &mut self,
        name: String,
        path: String,
        spec: PyValue,
        loader: PyValue,
    ) -> PyResult<PyValue>;
    /// Execute one bounded VFS source file in an existing VM module namespace.
    fn exec_module(&mut self, module: PyModule, path: &str) -> PyResult<()>;
    fn new_namespace(&mut self, values: Vec<(String, PyValue)>) -> PyResult<PyValue>;
    fn new_raises_context(&mut self, expected: String) -> PyResult<PyValue>;
    fn raises_expected(&self, context: PyRaisesContext) -> PyResult<String>;
    fn exception_type_name(&self, value: &PyValue) -> Option<&'static str>;
    /// Return one interpreter-owned exception class from the runtime's closed type table.
    fn exception_type(&self, name: &'static str) -> PyValue;
    /// Suspend the enclosing logical process until any modeled resource becomes ready.
    fn wait_on(&mut self, reasons: Vec<crate::scheduler::WaitReason>) -> PyResult<()>;
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
    fn frame_module_name(&mut self, depth: usize) -> PyResult<Option<PyValue>>;
    /// The parameters of a Python function, or of the function a bound method wraps without its
    /// bound first parameter. `None` for any other callable.
    fn function_parameters(&self, value: &PyValue) -> PyResult<Option<Vec<PyParameter>>>;
    /// `(is_generator, is_coroutine)` for a Python function or bound method; `None` otherwise.
    fn function_flags(&self, value: &PyValue) -> PyResult<Option<(bool, bool)>>;
    /// PID of the logical process running this Python interpreter.
    fn current_pid(&self) -> u32;
    /// PID of the logical parent of the process running this Python interpreter.
    fn current_ppid(&self) -> u32;
    /// Deliver a signal to an arbitrary modeled process, not only an owned subprocess handle.
    fn send_os_signal(&mut self, pid: u32, signal: crate::process::Signal) -> PyResult<()>;

    /// The name CPython prints for a value's type in error messages, such as `int`,
    /// `float32` or a user class name.
    fn type_name(&self, value: &PyValue) -> PyResult<String>;
    /// The exception being handled by the innermost active `except` block, for
    /// `sys.exc_info()` and `sys.exception()`.
    fn active_exception(&self) -> Option<PyValue>;
    /// A dict of the modules loaded so far, keyed by name, for `sys.modules`.
    fn loaded_modules(&mut self) -> PyResult<PyValue>;
    /// Register `module` under `name` so later imports of `name` return it, as assigning to
    /// `sys.modules` does; `module` must be a module object.
    fn register_module(&mut self, name: &str, module: PyValue) -> PyResult<()>;
    /// Forget the module registered under `name`, as `del sys.modules[name]` does. Returns
    /// whether a registration existed.
    fn unregister_module(&mut self, name: &str) -> bool;
}

/// Checked handle to an interpreter-owned array view.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyArray(PyValue);

impl FromPyValue for PyArray {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
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

impl PyArray {
    pub fn value(self) -> PyValue {
        self.0
    }
}

/// Conversion from an erased Python value into a checked native view.
pub(super) trait FromPyValue: Sized {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self>;
}

pub(super) trait PyValueCast {
    fn cast<T: FromPyValue>(self, runtime: &dyn PyRuntime) -> PyResult<T>;
}

impl PyValueCast for PyValue {
    fn cast<T: FromPyValue>(self, runtime: &dyn PyRuntime) -> PyResult<T> {
        T::from_py_value(runtime, self)
    }
}

/// Owned string extracted from an erased value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct OwnedPyString(pub String);

impl FromPyValue for OwnedPyString {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
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

impl FromPyValue for PyBytes {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
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
pub(super) struct PyByteArray(PyValue);

impl FromPyValue for PyByteArray {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let payload = value;
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

impl PyByteArray {
    pub fn value(self) -> PyValue {
        self.0
    }
}

/// Checked handle to one of the exception classes modeled by the VM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyExceptionType(pub &'static str);

impl FromPyValue for PyExceptionType {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        runtime
            .exception_type_name(&value)
            .map(Self)
            .ok_or_else(|| PyError::type_error("expected an exception type"))
    }
}

/// Checked handle to a compiled regular-expression object.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyRegex(PyValue);

impl PyRegex {
    pub fn value(self) -> PyValue {
        self.0
    }
}

impl FromPyValue for PyRegex {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
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
pub(super) struct PyMatch(PyValue);

impl PyMatch {
    pub fn value(self) -> PyValue {
        self.0
    }
}

impl FromPyValue for PyMatch {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
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

/// Owned, metered copy of a match payload, also the parts a new match is built from.
///
/// `spans` holds character offsets into the subject per group, `None` for a group that did not
/// participate; `spans[0]` is the whole match. `pos` and `endpos` are the searched window.
pub(super) struct PyMatchData {
    pub subject: PyValue,
    pub regex: PyValue,
    pub text: String,
    pub groups: Vec<Option<String>>,
    pub group_names: Vec<Option<String>>,
    pub spans: Vec<Option<(usize, usize)>>,
    pub pos: usize,
    pub endpos: usize,
}

/// Owned definition of one bounded ``argparse`` argument.
#[derive(Clone, Debug)]
pub(super) struct PyArgumentSpec {
    pub names: Vec<String>,
    pub dest: String,
    pub required: bool,
    pub default: PyValue,
    pub store_true: bool,
    pub store_false: bool,
    pub integer: bool,
    pub choices: Vec<PyValue>,
    pub help: Option<String>,
}

/// One command registered on an ``argparse`` subparser collection.
#[derive(Clone, Debug)]
pub(super) struct PySubcommandSpec {
    pub name: String,
    pub help: Option<String>,
    pub parser: PyArgumentParser,
}

/// Owned definition of the deliberately one-level subparser surface.
#[derive(Clone, Debug)]
pub(super) struct PySubparsersSpec {
    pub dest: Option<String>,
    pub required: bool,
    pub help: Option<String>,
    pub commands: Vec<PySubcommandSpec>,
}

/// Metered snapshot of an interpreter-owned argument parser.
#[derive(Clone, Debug)]
pub(super) struct PyArgumentParserData {
    pub prog: String,
    pub description: Option<String>,
    pub add_help: bool,
    pub arguments: Vec<PyArgumentSpec>,
    pub subparsers: Option<PySubparsersSpec>,
}

/// Checked handle to an interpreter-owned argument parser.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyArgumentParser(PyValue);

impl PyArgumentParser {
    pub fn value(self) -> PyValue {
        self.0
    }
}

impl FromPyValue for PyArgumentParser {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
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
pub(super) struct PyModule(PyValue);

impl PyModule {
    pub fn value(self) -> PyValue {
        self.0
    }
}

impl FromPyValue for PyModule {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
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
pub(super) struct PyRaisesContext(PyValue);

impl PyRaisesContext {
    pub fn value(self) -> PyValue {
        self.0
    }
}

impl FromPyValue for PyRaisesContext {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
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
pub(super) struct PyList(PyValue);

impl FromPyValue for PyList {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let payload = value;
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

impl PyList {
    pub fn value(self) -> PyValue {
        self.0
    }

    /// Snapshot list items so callers may allocate or invoke protocols while iterating.
    pub fn items(self, runtime: &mut dyn PyRuntime) -> PyResult<Vec<PyValue>> {
        runtime.list_items(self)
    }
}

/// Checked handle to a heap-backed Python tuple.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyTuple(PyValue);

impl FromPyValue for PyTuple {
    /// Accepts a tuple, or an instance of a `tuple` subclass through the tuple it holds.
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let tuple = value;
        match tuple {
            _ if tuple.is_object() && runtime.kind(&tuple)? == PyKind::Tuple => Ok(Self(tuple)),
            _ => {
                let actual = runtime.type_name(&value)?;
                Err(PyError::type_error(format!("expected tuple, got {actual}")))
            }
        }
    }
}

impl PyTuple {
    pub fn value(self) -> PyValue {
        self.0
    }

    pub fn items(self, runtime: &mut dyn PyRuntime) -> PyResult<Vec<PyValue>> {
        runtime.tuple_items(self)
    }
}

/// Checked view over the sequence kinds accepted by small stdlib algorithms.
#[derive(Clone, Copy, Debug)]
pub(super) enum PySequence {
    List(PyList),
    Tuple(PyTuple),
}

impl FromPyValue for PySequence {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        match runtime.kind(&value)? {
            PyKind::List => value.cast(runtime).map(Self::List),
            PyKind::Tuple => value.cast(runtime).map(Self::Tuple),
            kind => Err(PyError::type_error(format!(
                "expected a sequence, got {kind:?}"
            ))),
        }
    }
}

impl PySequence {
    pub fn items(self, runtime: &mut dyn PyRuntime) -> PyResult<Vec<PyValue>> {
        match self {
            Self::List(list) => list.items(runtime),
            Self::Tuple(tuple) => tuple.items(runtime),
        }
    }
}

/// Integer accepted by operations requiring Python's index protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyIndex(pub i64);

impl FromPyValue for PyIndex {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
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
pub(super) struct PyIterator(PyValue);

impl PyIterator {
    pub fn value(self) -> PyValue {
        self.0
    }
}

/// Checked callable value. Invocation stays on the runtime so modules do not inspect functions.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyCallable(PyValue);

impl FromPyValue for PyCallable {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
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

impl PyCallable {
    pub fn call(self, runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
        runtime.call_value(self.0, args)
    }

    pub(super) fn into_value(self) -> PyValue {
        self.0
    }
}

impl FromPyValue for PyIterator {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        match value {
            _ if value.is_object() && runtime.is_iterator(&value)? => Ok(Self(value)),
            _ => Err(PyError::type_error("expected an iterator")),
        }
    }
}

/// Checked handle to a heap-backed Python dictionary.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyDict(PyValue);

impl FromPyValue for PyDict {
    /// Accepts a dict, or an instance of a `dict` subclass through the dict it holds.
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let dict = value;
        match dict {
            _ if dict.is_object() && runtime.kind(&dict)? == PyKind::Dict => Ok(Self(dict)),
            _ => {
                let actual = runtime.type_name(&value)?;
                Err(PyError::type_error(format!("expected dict, got {actual}")))
            }
        }
    }
}

impl PyDict {
    pub fn value(self) -> PyValue {
        self.0
    }

    /// Snapshot entries so callers do not retain a heap borrow across Python work.
    pub fn items(self, runtime: &mut dyn PyRuntime) -> PyResult<Vec<(PyValue, PyValue)>> {
        runtime.dict_items(self)
    }
}

/// Checked handle to a heap-backed Python set.
#[derive(Clone, Copy, Debug)]
pub(super) struct PySet(PyValue);

impl FromPyValue for PySet {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let payload = value;
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

impl PySet {
    pub fn value(self) -> PyValue {
        self.0
    }

    pub fn items(self, runtime: &mut dyn PyRuntime) -> PyResult<Vec<PyValue>> {
        runtime.set_items(self)
    }
}

/// Checked handle to the standard `property` descriptor payload.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyProperty(PyValue);

impl FromPyValue for PyProperty {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
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

impl PyProperty {
    pub fn value(self) -> PyValue {
        self.0
    }
}

/// Checked handle to a heap-backed Python class.
#[derive(Clone, Copy, Debug)]
pub(super) struct PyClass(PyValue);

impl PyClass {
    pub fn value(self) -> PyValue {
        self.0
    }
}

impl FromPyValue for PyClass {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
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
pub(super) struct CallArgs {
    positional: Vec<PyValue>,
    keywords: Vec<(String, PyValue)>,
}

impl CallArgs {
    pub fn new(positional: Vec<PyValue>, keywords: Vec<(String, PyValue)>) -> Self {
        Self {
            positional,
            keywords,
        }
    }

    pub fn positional(&self) -> &[PyValue] {
        &self.positional
    }

    pub fn keywords(&self) -> &[(String, PyValue)] {
        &self.keywords
    }

    pub fn into_parts(self) -> (Vec<PyValue>, Vec<(String, PyValue)>) {
        (self.positional, self.keywords)
    }

    pub fn expect_positional(
        &self,
        function: &str,
        minimum: usize,
        maximum: usize,
    ) -> PyResult<()> {
        if (minimum..=maximum).contains(&self.positional.len()) {
            Ok(())
        } else {
            Err(PyError::type_error(format!(
                "{function}() expected {minimum}..={maximum} positional arguments, got {}",
                self.positional.len()
            )))
        }
    }

    pub fn reject_keywords(&self, function: &str) -> PyResult<()> {
        if self.keywords.is_empty() {
            Ok(())
        } else {
            Err(PyError::type_error(format!(
                "{function}() does not accept keyword arguments"
            )))
        }
    }

    pub fn keyword(&self, function: &str, name: &str) -> PyResult<Option<&PyValue>> {
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

    pub fn reject_unknown_keywords(&self, function: &str, names: &[&str]) -> PyResult<()> {
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
pub(super) type NativeFn = fn(&mut dyn PyRuntime, CallArgs) -> PyResult;

/// Uniform implementation type for a native method after descriptor binding.
pub(super) type NativeMethodFn = fn(&mut dyn PyRuntime, PyValue, CallArgs) -> PyResult;

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
pub(super) type NativeGetterFn = fn(&mut dyn PyRuntime, PyValue) -> PyResult;

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
        get: fn(&mut dyn PyRuntime) -> PyResult,
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

    pub fn get(&self, runtime: &mut dyn PyRuntime) -> PyResult {
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
