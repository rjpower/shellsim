//! Uniform value, argument, error, and runtime interfaces for native Python modules.
//!
//! Native functions are type-erased at the call boundary: every function accepts [`CallArgs`]
//! and returns a [`PyValue`]. Implementations recover small checked views such as [`PyNumber`] or
//! [`PyList`] locally. The runtime trait exposes allocation and metering, but no ambient host
//! capability.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

use super::heap::ObjectId;
use super::Value;

/// The erased value exchanged by native modules and the VM.
pub(super) type PyValue = Value;

/// Opaque stable identity for an arena-backed Python value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyIdentity(pub(super) ObjectId);

/// Result of a Python runtime operation.
pub(super) type PyResult<T = PyValue> = Result<T, PyError>;

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
    pub bool_: Option<UnarySlotFn>,
    pub add: Option<BinarySlotFn>,
    pub reflected_add: Option<BinarySlotFn>,
    pub subtract: Option<BinarySlotFn>,
    pub reflected_subtract: Option<BinarySlotFn>,
    pub multiply: Option<BinarySlotFn>,
    pub reflected_multiply: Option<BinarySlotFn>,
    pub divide: Option<BinarySlotFn>,
    pub reflected_divide: Option<BinarySlotFn>,
    pub equal: Option<BinarySlotFn>,
    pub not_equal: Option<BinarySlotFn>,
    pub less_than: Option<BinarySlotFn>,
    pub less_equal: Option<BinarySlotFn>,
    pub greater_than: Option<BinarySlotFn>,
    pub greater_equal: Option<BinarySlotFn>,
}

/// Static registration for a type-erased, inline Python value.
pub(super) struct ValueKindDef {
    pub name: &'static str,
    pub construct: NativeFn,
    pub slots: ValueKindSlots,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PyErrorKind {
    Type,
    Value,
    ZeroDivision,
    Overflow,
    Runtime,
    Resource,
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

/// Opaque module-owned scalar identity carried by a type-erased array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyArrayDtype {
    id: u8,
    name: &'static str,
}

impl PyArrayDtype {
    pub const fn new(id: u8, name: &'static str) -> Self {
        Self { id, name }
    }

    pub const fn name(self) -> &'static str {
        self.name
    }
}

/// Shape and storage mapping for an opaque array view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PyArrayLayout {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: isize,
}

/// Scalar operation used by generic array kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PyBinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
}

/// Interpreter-owned marker values exported by compatibility modules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PyMarker {
    TypingList,
    EnumBase,
    UnitTestBase,
    Environment,
    Stdin,
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
    /// Write bytes to a captured child stdin, cooperatively scheduling while the pipe is full.
    fn write_pipe(&mut self, handle: PyProcessHandle, input: Vec<u8>) -> PyResult<usize>;
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
    fn metadata(&self, path: &str) -> PyResult<PyFileMetadata>;
    fn mkdir(&mut self, path: &str, parents: bool, exist_ok: bool) -> PyResult<()>;
    fn glob(&mut self, pattern: &str) -> PyResult<Vec<String>>;
}

/// Stable metadata fields exposed by the modeled VFS to capability-scoped stdlib facades.
pub(super) struct PyFileMetadata {
    pub mode: u32,
    pub size: usize,
}

/// Runtime value protocols and explicitly modeled services available to native modules.
pub(super) trait PyRuntime {
    fn reserve_memory(&mut self, bytes: usize) -> PyResult<()>;
    fn charge_cpu(&mut self, units: u64) -> PyResult<()>;
    fn kind(&self, value: &PyValue) -> PyResult<PyKind>;
    fn native_kind(&self, value: &PyValue) -> PyResult<Option<PyNativeKind>>;
    fn identity(&self, value: &PyValue) -> Option<PyIdentity>;
    fn int_value(&self, value: &PyValue) -> Option<i64>;
    fn string_value(&self, value: &PyValue) -> PyResult<Option<String>>;
    fn bytes_value(&self, value: &PyValue) -> PyResult<Option<Vec<u8>>>;
    fn bytearray_items(&mut self, value: PyByteArray) -> PyResult<Vec<u8>>;
    fn replace_bytearray_items(&mut self, value: PyByteArray, items: Vec<u8>) -> PyResult<()>;
    fn is_integer_type(&self, value: &PyValue) -> bool;
    fn is_string_type(&self, value: &PyValue) -> bool;
    /// Return an exact decimal rendering for any Python integer representation.
    fn integer_text(&self, value: &PyValue) -> PyResult<Option<String>>;
    /// Write only to an interpreter-owned simulated stream marker.
    fn write_stream(&mut self, stream: &PyValue, text: &str) -> PyResult<usize>;
    /// Read text only from the invocation's modeled standard-input stream.
    fn read_stream(
        &mut self,
        stream: &PyValue,
        size: Option<usize>,
        line: bool,
    ) -> PyResult<String>;
    fn truth(&mut self, value: &PyValue) -> PyResult<bool>;
    fn display(&mut self, value: &PyValue) -> PyResult<String>;
    fn repr(&mut self, value: &PyValue) -> PyResult<String>;
    fn equals(&mut self, left: &PyValue, right: &PyValue) -> PyResult<bool>;
    fn compare(&mut self, left: &PyValue, right: &PyValue) -> PyResult<Ordering>;
    /// Resolve an attribute through the runtime's descriptor and MRO protocol.
    fn get_attribute(&mut self, value: PyValue, name: &str) -> PyResult<Option<PyValue>>;
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
    fn slice_parts(&self, value: &PyValue) -> Option<(Option<i64>, Option<i64>, Option<i64>)>;
    fn dict_items(&mut self, dict: PyDict) -> PyResult<Vec<(PyValue, PyValue)>>;
    fn dict_get(&mut self, dict: PyDict, key: &PyValue) -> PyResult<Option<PyValue>>;
    fn dict_insert(&mut self, dict: PyDict, key: PyValue, value: PyValue) -> PyResult<()>;
    fn dict_remove(&mut self, dict: PyDict, key: &PyValue) -> PyResult<Option<PyValue>>;
    fn replace_dict_items(&mut self, dict: PyDict, items: Vec<(PyValue, PyValue)>) -> PyResult<()>;
    fn set_items(&mut self, set: PySet) -> PyResult<Vec<PyValue>>;
    fn set_insert(&mut self, set: PySet, value: PyValue) -> PyResult<bool>;
    fn set_remove(&mut self, set: PySet, value: &PyValue) -> PyResult<bool>;
    fn replace_list_items(&mut self, list: PyList, items: Vec<PyValue>) -> PyResult<()>;
    fn call_value(&mut self, callable: PyValue, args: CallArgs) -> PyResult<PyValue>;
    fn is_callable(&self, value: &PyValue) -> PyResult<bool>;
    fn iterator(&mut self, value: PyValue) -> PyResult<PyIterator>;
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
    fn new_value_kind(&self, kind: &'static ValueKindDef, payload: u64) -> PyResult<PyValue>;
    fn value_kind_payload(&self, value: &PyValue, kind: &'static ValueKindDef) -> Option<u64>;
    fn value_kind_type(&self, kind: &'static ValueKindDef) -> PyResult<PyValue>;
    fn new_array(
        &mut self,
        items: Vec<PyValue>,
        shape: Vec<usize>,
        dtype: PyArrayDtype,
    ) -> PyResult<PyValue>;
    fn new_array_view(&mut self, array: PyArray, layout: PyArrayLayout) -> PyResult<PyValue>;
    fn array_layout(&self, array: PyArray) -> PyResult<(PyArrayLayout, PyArrayDtype)>;
    fn array_get(&mut self, array: PyArray, index: &[usize]) -> PyResult<PyValue>;
    fn array_set(&mut self, array: PyArray, index: &[usize], value: PyValue) -> PyResult<()>;
    fn binary_op(
        &mut self,
        operation: PyBinaryOp,
        left: PyValue,
        right: PyValue,
    ) -> PyResult<PyValue>;
    fn new_string(&mut self, value: String) -> PyResult<PyValue>;
    fn new_bytes(&mut self, value: Vec<u8>) -> PyResult<PyValue>;
    fn new_bytearray(&mut self, value: Vec<u8>) -> PyResult<PyValue>;
    fn property_getter(&self, property: PyProperty) -> PyResult<PyValue>;
    fn new_property(&mut self, getter: PyValue, setter: Option<PyValue>) -> PyResult<PyValue>;
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
    fn new_regex(&mut self, pattern: String, flags: u32) -> PyResult<PyValue>;
    fn new_match(
        &mut self,
        text: String,
        groups: Vec<Option<String>>,
        start: usize,
        end: usize,
    ) -> PyResult<PyValue>;
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
    fn clock(&mut self) -> &mut dyn PyClock;
    fn environment(&self) -> &dyn PyEnvironment;
    fn filesystem(&mut self) -> &mut dyn PyFilesystem;
    fn http(&mut self) -> &mut dyn PyHttpClient;
    fn processes(&mut self) -> &mut dyn PyProcessRunner;

    fn type_name(&self, value: &PyValue) -> PyResult<&'static str> {
        Ok(match self.kind(value)? {
            PyKind::None => "NoneType",
            PyKind::Bool => "bool",
            PyKind::Int => "int",
            PyKind::Float => "float",
            PyKind::String => "str",
            PyKind::Bytes => "bytes",
            PyKind::ByteArray => "bytearray",
            PyKind::List => "list",
            PyKind::Tuple => "tuple",
            PyKind::Dict => "dict",
            PyKind::Set => "set",
            PyKind::Function => "function",
            PyKind::Class => "type",
            PyKind::Instance => "object",
            PyKind::Iterator => "iterator",
            PyKind::Generator => "generator",
            PyKind::Module => "module",
            PyKind::Array => "numpy.ndarray",
            PyKind::Native => "object",
        })
    }
}

/// Checked handle to an interpreter-owned array view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyArray(ObjectId);

impl FromPyValue for PyArray {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            return Err(PyError::type_error("expected numpy.ndarray"));
        };
        if runtime.native_kind(&Value::Object(id))? == Some(PyNativeKind::Array) {
            Ok(Self(id))
        } else {
            Err(PyError::type_error("expected numpy.ndarray"))
        }
    }
}

impl PyArray {
    pub(super) fn object_id(self) -> ObjectId {
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyByteArray(ObjectId);

impl FromPyValue for PyByteArray {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            return Err(PyError::type_error("expected a bytearray"));
        };
        if runtime.kind(&value)? == PyKind::ByteArray {
            Ok(Self(id))
        } else {
            Err(PyError::type_error("expected a bytearray"))
        }
    }
}

impl PyByteArray {
    pub(super) fn object_id(self) -> ObjectId {
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyRegex(ObjectId);

impl PyRegex {
    pub(super) fn object_id(self) -> ObjectId {
        self.0
    }
}

impl FromPyValue for PyRegex {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            return Err(PyError::type_error("expected a compiled regex"));
        };
        if runtime.native_kind(&Value::Object(id))? == Some(PyNativeKind::Regex) {
            Ok(Self(id))
        } else {
            Err(PyError::type_error("expected a compiled regex"))
        }
    }
}

/// Checked handle to a regular-expression match object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyMatch(ObjectId);

impl PyMatch {
    pub(super) fn object_id(self) -> ObjectId {
        self.0
    }
}

impl FromPyValue for PyMatch {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            return Err(PyError::type_error("expected a regex match"));
        };
        if runtime.native_kind(&Value::Object(id))? == Some(PyNativeKind::Match) {
            Ok(Self(id))
        } else {
            Err(PyError::type_error("expected a regex match"))
        }
    }
}

/// Owned, metered snapshot of a match payload.
pub(super) struct PyMatchData {
    pub groups: Vec<Option<String>>,
    pub start: usize,
    pub end: usize,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyArgumentParser(ObjectId);

impl PyArgumentParser {
    pub(super) fn object_id(self) -> ObjectId {
        self.0
    }
}

impl FromPyValue for PyArgumentParser {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            return Err(PyError::type_error("expected ArgumentParser"));
        };
        if runtime.native_kind(&Value::Object(id))? == Some(PyNativeKind::ArgumentParser) {
            Ok(Self(id))
        } else {
            Err(PyError::type_error("expected ArgumentParser"))
        }
    }
}

/// Checked handle to an interpreter-owned Python module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyModule(ObjectId);

impl PyModule {
    pub(super) fn object_id(self) -> ObjectId {
        self.0
    }
}

impl FromPyValue for PyModule {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            return Err(PyError::type_error("expected module"));
        };
        if runtime.kind(&Value::Object(id))? == PyKind::Module {
            Ok(Self(id))
        } else {
            Err(PyError::type_error("expected module"))
        }
    }
}

/// Checked handle to a context returned by ``pytest.raises``.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyRaisesContext(ObjectId);

impl PyRaisesContext {
    pub(super) fn object_id(self) -> ObjectId {
        self.0
    }
}

impl FromPyValue for PyRaisesContext {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            return Err(PyError::type_error("expected pytest.raises context"));
        };
        if runtime.native_kind(&Value::Object(id))? == Some(PyNativeKind::RaisesContext) {
            Ok(Self(id))
        } else {
            Err(PyError::type_error("expected pytest.raises context"))
        }
    }
}

/// Checked handle to an arena-backed Python list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyList(ObjectId);

impl FromPyValue for PyList {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            let actual = runtime.type_name(&value)?;
            return Err(PyError::type_error(format!("expected list, got {actual}")));
        };
        if runtime.kind(&Value::Object(id))? == PyKind::List {
            Ok(Self(id))
        } else {
            let actual = runtime.type_name(&Value::Object(id))?;
            Err(PyError::type_error(format!("expected list, got {actual}")))
        }
    }
}

impl PyList {
    pub(super) fn object_id(self) -> ObjectId {
        self.0
    }

    /// Snapshot list items so callers may allocate or invoke protocols while iterating.
    pub fn items(self, runtime: &mut dyn PyRuntime) -> PyResult<Vec<PyValue>> {
        runtime.list_items(self)
    }
}

/// Checked handle to an arena-backed Python tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyTuple(ObjectId);

impl FromPyValue for PyTuple {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            let actual = runtime.type_name(&value)?;
            return Err(PyError::type_error(format!("expected tuple, got {actual}")));
        };
        if runtime.kind(&Value::Object(id))? == PyKind::Tuple {
            Ok(Self(id))
        } else {
            let actual = runtime.type_name(&Value::Object(id))?;
            Err(PyError::type_error(format!("expected tuple, got {actual}")))
        }
    }
}

impl PyTuple {
    pub(super) fn object_id(self) -> ObjectId {
        self.0
    }

    pub fn items(self, runtime: &mut dyn PyRuntime) -> PyResult<Vec<PyValue>> {
        runtime.tuple_items(self)
    }
}

/// Checked view over the sequence kinds accepted by small stdlib algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyIterator(ObjectId);

impl PyIterator {
    pub(super) fn object_id(self) -> ObjectId {
        self.0
    }
}

/// Checked callable value. Invocation stays on the runtime so modules do not inspect functions.
#[derive(Clone, Debug)]
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
        let Some(id) = value.object_id() else {
            return Err(PyError::type_error("expected an iterator"));
        };
        if matches!(
            runtime.kind(&Value::Object(id))?,
            PyKind::Iterator | PyKind::Generator
        ) {
            Ok(Self(id))
        } else {
            Err(PyError::type_error("expected an iterator"))
        }
    }
}

/// Checked handle to an arena-backed Python dictionary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyDict(ObjectId);

impl FromPyValue for PyDict {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            let actual = runtime.type_name(&value)?;
            return Err(PyError::type_error(format!("expected dict, got {actual}")));
        };
        if runtime.kind(&Value::Object(id))? == PyKind::Dict {
            Ok(Self(id))
        } else {
            let actual = runtime.type_name(&Value::Object(id))?;
            Err(PyError::type_error(format!("expected dict, got {actual}")))
        }
    }
}

impl PyDict {
    pub(super) fn object_id(self) -> ObjectId {
        self.0
    }

    /// Snapshot entries so callers do not retain an arena borrow across Python work.
    pub fn items(self, runtime: &mut dyn PyRuntime) -> PyResult<Vec<(PyValue, PyValue)>> {
        runtime.dict_items(self)
    }
}

/// Checked handle to an arena-backed Python set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PySet(ObjectId);

impl FromPyValue for PySet {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            return Err(PyError::type_error("expected set"));
        };
        if runtime.kind(&Value::Object(id))? == PyKind::Set {
            Ok(Self(id))
        } else {
            Err(PyError::type_error("expected set"))
        }
    }
}

impl PySet {
    pub(super) fn object_id(self) -> ObjectId {
        self.0
    }

    pub fn items(self, runtime: &mut dyn PyRuntime) -> PyResult<Vec<PyValue>> {
        runtime.set_items(self)
    }
}

/// Checked handle to the standard `property` descriptor payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyProperty(ObjectId);

impl FromPyValue for PyProperty {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            return Err(PyError::type_error("expected property"));
        };
        if runtime.native_kind(&Value::Object(id))? == Some(PyNativeKind::Property) {
            Ok(Self(id))
        } else {
            Err(PyError::type_error("expected property"))
        }
    }
}

impl PyProperty {
    pub(super) fn object_id(self) -> ObjectId {
        self.0
    }
}

/// Checked handle to an arena-backed Python class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyClass(ObjectId);

impl PyClass {
    pub(super) fn object_id(self) -> ObjectId {
        self.0
    }
}

impl FromPyValue for PyClass {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            return Err(PyError::type_error("expected a class"));
        };
        if runtime.kind(&Value::Object(id))? == PyKind::Class {
            Ok(Self(id))
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

/// Method table for an interpreter-defined native object type.
pub(super) struct NativeTypeDef {
    pub name: &'static str,
    pub methods: &'static [MethodDef],
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
}

impl ValueDef {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Constant { name, .. } | Self::Factory { name, .. } => name,
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
