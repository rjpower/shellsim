//! Uniform value, argument, error, and runtime interfaces for native Python modules.
//!
//! Native functions are type-erased at the call boundary: every function accepts [`CallArgs`]
//! and returns a [`PyValue`]. Implementations recover small checked views such as [`PyNumber`] or
//! [`PyList`] locally. The runtime trait exposes allocation and metering, but no ambient host
//! capability.

use std::cmp::Ordering;
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
    Exit(i32),
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
}

/// Interpreter-owned marker values exported by compatibility modules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PyMarker {
    TypingList,
    EnumBase,
    UnitTestBase,
    Environment,
    Stdout,
    Stderr,
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

/// Runtime value protocols and explicitly modeled services available to native modules.
pub(super) trait PyRuntime {
    fn reserve_memory(&mut self, bytes: usize) -> PyResult<()>;
    fn charge_cpu(&mut self, units: u64) -> PyResult<()>;
    fn kind(&self, value: &PyValue) -> PyResult<PyKind>;
    fn native_kind(&self, value: &PyValue) -> PyResult<Option<PyNativeKind>>;
    fn identity(&self, value: &PyValue) -> Option<PyIdentity>;
    fn int_value(&self, value: &PyValue) -> Option<i64>;
    fn string_value(&self, value: &PyValue) -> PyResult<Option<String>>;
    fn is_integer_type(&self, value: &PyValue) -> bool;
    /// Return an exact decimal rendering for any Python integer representation.
    fn integer_text(&self, value: &PyValue) -> PyResult<Option<String>>;
    /// Write only to an interpreter-owned simulated stream marker.
    fn write_stream(&mut self, stream: &PyValue, text: &str) -> PyResult<usize>;
    fn truth(&mut self, value: &PyValue) -> PyResult<bool>;
    fn display(&mut self, value: &PyValue) -> PyResult<String>;
    fn repr(&mut self, value: &PyValue) -> PyResult<String>;
    fn equals(&mut self, left: &PyValue, right: &PyValue) -> PyResult<bool>;
    fn compare(&mut self, left: &PyValue, right: &PyValue) -> PyResult<Ordering>;
    /// Resolve an attribute through the runtime's descriptor and MRO protocol.
    fn get_attribute(&mut self, value: PyValue, name: &str) -> PyResult<Option<PyValue>>;
    fn list_items(&mut self, list: PyList) -> PyResult<Vec<PyValue>>;
    fn tuple_items(&mut self, tuple: PyTuple) -> PyResult<Vec<PyValue>>;
    fn dict_items(&mut self, dict: PyDict) -> PyResult<Vec<(PyValue, PyValue)>>;
    fn replace_dict_items(&mut self, dict: PyDict, items: Vec<(PyValue, PyValue)>) -> PyResult<()>;
    fn set_items(&mut self, set: PySet) -> PyResult<Vec<PyValue>>;
    fn replace_set_items(&mut self, set: PySet, items: Vec<PyValue>) -> PyResult<()>;
    fn instance_attribute(&self, instance: PyInstance, name: &str) -> PyResult<Option<PyValue>>;
    fn replace_list_items(&mut self, list: PyList, items: Vec<PyValue>) -> PyResult<()>;
    fn call_value(&mut self, callable: PyValue, args: CallArgs) -> PyResult<PyValue>;
    fn is_callable(&self, value: &PyValue) -> PyResult<bool>;
    fn iterator(&mut self, value: PyValue) -> PyResult<PyIterator>;
    fn iterator_next(&mut self, iterator: PyIterator) -> PyResult<Option<PyValue>>;
    fn new_iterator(&mut self, values: Vec<PyValue>) -> PyResult<PyValue>;
    fn new_count_iterator(&mut self, start: i64, step: i64) -> PyResult<PyValue>;
    fn new_default_dict(&mut self, factory: PyCallable) -> PyResult<PyValue>;
    fn new_list(&mut self, items: Vec<PyValue>) -> PyResult<PyValue>;
    fn new_tuple(&mut self, items: Vec<PyValue>) -> PyResult<PyValue>;
    fn new_dict(&mut self, items: Vec<(PyValue, PyValue)>) -> PyResult<PyValue>;
    fn new_set(&mut self, items: Vec<PyValue>) -> PyResult<PyValue>;
    fn new_string(&mut self, value: String) -> PyResult<PyValue>;
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
    fn new_argument_parser(&mut self, program: String) -> PyResult<PyValue>;
    fn argument_parser_parts(
        &mut self,
        parser: PyArgumentParser,
    ) -> PyResult<(String, Vec<PyArgumentSpec>)>;
    fn append_argument(
        &mut self,
        parser: PyArgumentParser,
        argument: PyArgumentSpec,
    ) -> PyResult<()>;
    fn command_arguments(&self) -> Vec<String>;
    fn new_namespace(&mut self, values: Vec<(String, PyValue)>) -> PyResult<PyValue>;
    fn new_raises_context(&mut self, expected: String) -> PyResult<PyValue>;
    fn raises_expected(&self, context: PyRaisesContext) -> PyResult<String>;
    fn exception_type_name(&self, value: &PyValue) -> Option<&'static str>;
    fn clock(&mut self) -> &mut dyn PyClock;
    fn environment(&self) -> &dyn PyEnvironment;

    fn type_name(&self, value: &PyValue) -> PyResult<&'static str> {
        Ok(match self.kind(value)? {
            PyKind::None => "NoneType",
            PyKind::Bool => "bool",
            PyKind::Int => "int",
            PyKind::Float => "float",
            PyKind::String => "str",
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
            PyKind::Native => "object",
        })
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
pub(super) struct PyString(pub String);

impl FromPyValue for PyString {
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
    pub integer: bool,
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

/// Checked handle to an arena-backed Python instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PyInstance(ObjectId);

impl FromPyValue for PyInstance {
    fn from_py_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Self> {
        let Some(id) = value.object_id() else {
            let actual = runtime.type_name(&value)?;
            return Err(PyError::type_error(format!(
                "expected instance, got {actual}"
            )));
        };
        if runtime.kind(&Value::Object(id))? == PyKind::Instance {
            Ok(Self(id))
        } else {
            let actual = runtime.type_name(&Value::Object(id))?;
            Err(PyError::type_error(format!(
                "expected instance, got {actual}"
            )))
        }
    }
}

impl PyInstance {
    pub(super) fn object_id(self) -> ObjectId {
        self.0
    }

    /// Read an instance attribute without exposing the arena representation.
    pub fn attribute(self, runtime: &dyn PyRuntime, name: &str) -> PyResult<Option<PyValue>> {
        runtime.instance_attribute(self, name)
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
