//! Native-module runtime bridge backed by the metered Python VM.

use super::super::heap::{self, Builder, FunctionObject, OrderedMap};
use super::super::stdlib::argparse::{
    ArgumentParserObject, ArgumentSpec, NamespaceObject, SubcommandSpec, SubparsersSpec,
};
use super::super::stdlib::numpy::storage::{ArrayObject, ArrayStorage};
use super::super::stdlib::re::{MatchObject, RegexObject};
use super::super::stdlib::unittest::RaisesContextObject;
use super::namespace::NamespaceHandle;
use super::{
    exception_types, number, protocol, string, Arc, BigInt, BuiltinType, CallArgs, CallMode,
    CallResult, ClassDefinition, ClassLayout, ExceptionType, Execution, FrameEntry, HashMap,
    NativeValue, Object, Ordering, PyArgumentParser, PyArgumentParserData, PyArgumentSpec, PyArray,
    PyArrayBuffer, PyArrayData, PyArrayDataMut, PyArrayDtype, PyArrayMut, PyArrayReader,
    PyArrayRef, PyArrayView, PyByteArray, PyCallable, PyClass, PyClock, PyDict, PyEnvironment,
    PyError, PyErrorKind, PyFilesystem, PyHttpClient, PyIdentity, PyIterator, PyKind, PyList,
    PyMarker, PyMatch, PyMatchData, PyModule, PyNativeKind, PyOperator, PyProcessRunner,
    PyProperty, PyRaisesContext, PyRegex, PyResult, PyRuntime, PySet, PyStreamRead,
    PySubcommandSpec, PySubparsersSpec, PyTuple, PyTypeObject, PyValueCast, RaisedException,
    Stream, ToPrimitive, Value, Vm, MODELED_MAPPING_ENTRY_BYTES, MODELED_SET_MEMBER_BYTES,
    MODELED_VALUE_BYTES,
};
use crate::python::bytecode::ParameterKind;
use crate::python::heap::DictViewKind;
use crate::python::native::{PyParameter, PyRefs, PyScope};
use std::marker::PhantomData;

/// Check that every element `view` addresses lies inside storage of `byte_len` bytes and suits
/// its element kind (`values` storage holds object references).
fn validate_array_view(view: &PyArrayView, values: bool, byte_len: usize) -> Result<(), PyError> {
    let invalid = |message: &str| Err(PyError::runtime_error(message.to_string()));
    if view.shape.len() != view.strides.len() {
        return invalid("array shape and strides have different ranks");
    }
    if view.dtype.is_values() != values {
        return invalid("array view dtype does not match its storage");
    }
    if view.dtype.is_values() {
        let aligned = |value: isize| value % PyArrayDtype::VALUE_ITEMSIZE as isize == 0;
        if !view.offset.is_multiple_of(PyArrayDtype::VALUE_ITEMSIZE)
            || !view.strides.iter().all(|stride| aligned(*stride))
        {
            return invalid("object array view is not aligned to its elements");
        }
    }
    if view.shape.contains(&0) {
        return Ok(());
    }
    let overflow = || PyError::value_error("array is too big.");
    let mut minimum = isize::try_from(view.offset).map_err(|_| overflow())?;
    let mut maximum = minimum;
    for (length, stride) in view.shape.iter().zip(&view.strides) {
        let span = isize::try_from(length - 1)
            .ok()
            .and_then(|length| stride.checked_mul(length))
            .ok_or_else(overflow)?;
        if span < 0 {
            minimum = minimum.checked_add(span).ok_or_else(overflow)?;
        } else {
            maximum = maximum.checked_add(span).ok_or_else(overflow)?;
        }
    }
    let end = usize::try_from(maximum)
        .ok()
        .and_then(|maximum| maximum.checked_add(view.dtype.itemsize()));
    if minimum < 0 || end.is_none_or(|end| end > byte_len) {
        return invalid("array view is outside storage");
    }
    Ok(())
}

fn wait_reason_ready(
    interp: &crate::interp::Interp,
    reason: &crate::scheduler::WaitReason,
    now: u64,
) -> bool {
    use crate::scheduler::WaitReason;

    match reason {
        WaitReason::ShellSession(_) | WaitReason::HostHttp(_) => false,
        WaitReason::Timer(deadline) => *deadline <= now,
        WaitReason::InputReadable(description) => interp.descriptors.input_readable(*description),
        WaitReason::PipeReadable(pipe) => interp.descriptors.pipe_readable(*pipe),
        WaitReason::PipeWritable(pipe) => interp.descriptors.pipe_writable(*pipe),
        WaitReason::Child(pid) | WaitReason::ChildActivity(pid) => matches!(
            interp.processes.get(*pid).map(|record| record.status),
            Some(crate::process::ProcessStatus::Exited(_))
        ),
        WaitReason::ChildDeadline(pid, deadline)
        | WaitReason::ChildActivityDeadline(pid, deadline) => {
            *deadline <= now
                || matches!(
                    interp.processes.get(*pid).map(|record| record.status),
                    Some(crate::process::ProcessStatus::Exited(_))
                )
        }
        WaitReason::Any(reasons) => reasons
            .iter()
            .any(|reason| wait_reason_ready(interp, reason, now)),
    }
}

fn stdin_wait_reason(wait: crate::descriptors::IoWait) -> crate::scheduler::WaitReason {
    match wait {
        crate::descriptors::IoWait::InputReadable(description) => {
            crate::scheduler::WaitReason::InputReadable(description)
        }
        crate::descriptors::IoWait::PipeReadable(pipe) => {
            crate::scheduler::WaitReason::PipeReadable(pipe)
        }
        crate::descriptors::IoWait::PipeWritable(pipe) => {
            crate::scheduler::WaitReason::PipeWritable(pipe)
        }
    }
}

impl<'s> Vm<'s> {
    /// Charge for moving `count` list slots, as inserting or removing before the end does.
    fn charge_shift(&mut self, count: usize) -> PyResult<'s, ()> {
        let bytes = (count as u64).saturating_mul(MODELED_VALUE_BYTES);
        super::super::heap::charge_construction(bytes, &mut self.interp.resources)
            .map_err(PyError::resource_error)
    }

    /// Replace the `old_len` elements of a list, dict or set payload with `new_len` elements
    /// stored by `install`, charging exactly what replacing the whole payload would: the
    /// construction CPU of the new payload, growth before installing, and shrink after.
    /// `element_bytes` is the modeled size of one element.
    fn replace_container_payload(
        &mut self,
        container: Value<'s>,
        old_len: usize,
        new_len: usize,
        element_bytes: u64,
        install: impl FnOnce(&Builder<'_>, &mut Object),
    ) -> PyResult<'s, ()> {
        let overflow = || PyError::resource_error("modeled object size overflow");
        let size = |length: usize| {
            u64::try_from(length)
                .ok()
                .and_then(|length| length.checked_mul(element_bytes))
        };
        let old_bytes = size(old_len).ok_or_else(overflow)?;
        let new_bytes = size(new_len).ok_or_else(overflow)?;
        let current = self
            .heap()
            .object_bytes(container)
            .map_err(PyError::runtime_error)?;
        let next = current
            .saturating_sub(old_bytes)
            .checked_add(new_bytes)
            .ok_or_else(overflow)?;
        super::super::heap::charge_construction(next, &mut self.interp.resources)
            .map_err(PyError::resource_error)?;
        if new_bytes > old_bytes {
            self.reserve_object_growth(container, new_bytes - old_bytes)
                .map_err(PyError::resource_error)?;
        }
        self.modify(container, install)
            .map_err(PyError::runtime_error)?;
        if old_bytes > new_bytes {
            self.release_object_shrink(container, old_bytes - new_bytes)
                .map_err(PyError::resource_error)?;
        }
        Ok(())
    }

    /// A VM failure as a native error: the Python exception it raised, if any, or else an
    /// internal runtime error.
    fn raised_or_runtime_error(&self, message: String) -> PyError {
        if self.pending_exception.is_some() {
            PyError::new(PyErrorKind::Raised, message)
        } else {
            PyError::runtime_error(message)
        }
    }

    /// The registered kind of an inline or wide registered value.
    pub(super) fn registered_kind(
        &self,
        value: &Value<'_>,
    ) -> Option<&'static super::super::native::ValueKindDef> {
        if let Some((index, _)) = value.registered_parts() {
            return self.state.types.value_kind(index);
        }
        if !value.is_object() {
            return None;
        }
        match self.state.heap.get(*value).ok()? {
            Object::WideValue { kind, .. } => self.state.types.value_kind(*kind),
            _ => None,
        }
    }

    /// The heap view object behind a checked array handle.
    fn array_object(&self, array: PyArray<'s>) -> PyResult<'s, &ArrayObject> {
        self.state
            .heap
            .get(array.value())
            .map_err(PyError::runtime_error)?
            .native::<ArrayObject>()
            .ok_or_else(|| PyError::runtime_error("array handle changed object kind"))
    }

    /// The element storage behind an array, borrowed for reading.
    fn array_storage_object(&self, storage: Value<'s>) -> PyResult<'s, &ArrayStorage> {
        self.get(storage)
            .map_err(PyError::runtime_error)?
            .native::<ArrayStorage>()
            .ok_or_else(|| PyError::runtime_error("array storage changed object kind"))
    }

    /// Read `sys.stdin`/`sys.stdin.buffer` from the fully-supplied `ProcessInput::stdin` slice.
    ///
    /// Used by synchronous execution: the interactive REPL, nested/legacy command dispatch (a
    /// command invoked from inside another native command cannot suspend), and the direct
    /// `shellsim::python::run_python` embedding API. All of these already receive their complete
    /// standard input up front, so there is nothing to stream and no descriptor to block on.
    fn read_stream_buffered(
        &mut self,
        size: Option<usize>,
        line: bool,
        binary: bool,
    ) -> PyResult<'s, PyStreamRead> {
        if binary {
            let start = self.stdin_position.min(self.stdin.len());
            let available = &self.stdin[start..];
            self.charge_cpu(u64::try_from(available.len()).unwrap_or(u64::MAX))
                .map_err(PyError::runtime_error)?;
            let size_end = size.map_or(available.len(), |bytes| bytes.min(available.len()));
            let mut length = size_end;
            if line {
                if let Some(newline) = available[..size_end].iter().position(|byte| *byte == b'\n')
                {
                    length = newline + 1;
                }
            }
            self.reserve_memory(length)?;
            let bytes = available[..length].to_vec();
            self.stdin_position = self.stdin_position.saturating_add(length);
            return Ok(PyStreamRead::Bytes(bytes));
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
        Ok(PyStreamRead::Text(text))
    }

    /// Read `sys.stdin`/`sys.stdin.buffer` incrementally through the process's fd 0.
    ///
    /// Bytes already pulled off the descriptor but not yet consumed by a completed read live in
    /// `stdin_stream_pending`, so a suspend-and-retry (a blocked pipe read) never drops data. A
    /// read that cannot be satisfied without more input attempts exactly one bounded `read_fd`
    /// per call; if that would block, this suspends the calling process via `PyError::suspend`
    /// instead of waiting synchronously, exactly as `subprocess`'s pipe reads already do.
    fn read_stream_streaming(
        &mut self,
        size: Option<usize>,
        line: bool,
        binary: bool,
    ) -> PyResult<'s, PyStreamRead> {
        loop {
            if let Some(read) = self.take_pending_stdin(size, line, binary)? {
                return Ok(read);
            }
            if self.stdin_stream_eof {
                return Ok(if binary {
                    PyStreamRead::Bytes(Vec::new())
                } else {
                    PyStreamRead::Text(String::new())
                });
            }
            match self
                .interp
                .read_fd(0, crate::descriptors::DEVICE_READ_QUANTUM)
            {
                Ok(crate::descriptors::IoPoll::Ready(bytes)) if bytes.is_empty() => {
                    self.stdin_stream_eof = true;
                }
                Ok(crate::descriptors::IoPoll::Ready(bytes)) => {
                    self.charge_cpu(u64::try_from(bytes.len()).unwrap_or(u64::MAX))
                        .map_err(PyError::runtime_error)?;
                    self.reserve_retained_memory(bytes.len())
                        .map_err(PyError::resource_error)?;
                    self.stdin_stream_pending.extend_from_slice(&bytes);
                }
                Ok(crate::descriptors::IoPoll::Blocked(wait)) => {
                    return Err(PyError::suspend(stdin_wait_reason(wait)));
                }
                Err(message) => return Err(PyError::runtime_error(message)),
            }
        }
    }

    /// Try to satisfy one read from bytes already buffered from fd 0. Returns `None` when more
    /// input is required (and end-of-file has not yet been observed).
    ///
    /// `size` counts bytes in binary mode and *characters* in text mode, matching
    /// [`Vm::read_stream_buffered`]. Binary reads slice the pending bytes directly; text reads go
    /// through [`Vm::take_pending_stdin_text`], which only ever consumes complete UTF-8
    /// characters so a multi-byte codepoint split across two `read_fd` quanta is not mistaken for
    /// invalid input.
    fn take_pending_stdin(
        &mut self,
        size: Option<usize>,
        line: bool,
        binary: bool,
    ) -> PyResult<'s, Option<PyStreamRead>> {
        if binary {
            let cap = size.unwrap_or(usize::MAX);
            let length = if line {
                let bound = self.stdin_stream_pending.len().min(cap);
                if let Some(newline) = self.stdin_stream_pending[..bound]
                    .iter()
                    .position(|byte| *byte == b'\n')
                {
                    Some(newline + 1)
                } else if bound == cap && cap != usize::MAX {
                    Some(cap)
                } else if self.stdin_stream_eof {
                    Some(bound)
                } else {
                    None
                }
            } else if let Some(cap) = size {
                if self.stdin_stream_pending.len() >= cap {
                    Some(cap)
                } else if self.stdin_stream_eof {
                    Some(self.stdin_stream_pending.len())
                } else {
                    None
                }
            } else if self.stdin_stream_eof {
                Some(self.stdin_stream_pending.len())
            } else {
                None
            };
            let Some(length) = length else {
                return Ok(None);
            };
            return Ok(Some(PyStreamRead::Bytes(self.drain_pending_stdin(length))));
        }
        self.take_pending_stdin_text(size, line)
    }

    /// Text-mode counterpart of [`Vm::take_pending_stdin`]'s binary branch.
    ///
    /// A partial multi-byte sequence at the tail of the buffered bytes is a pipe-quantum
    /// boundary artifact, not malformed input, so it is treated as "more data needed" rather than
    /// a decoding error unless end-of-file has already been observed (in which case no further
    /// byte can ever arrive to complete it, and CPython's own `UnicodeDecodeError` behavior is
    /// approximated with the same `ValueError` the fully-buffered path raises).
    fn take_pending_stdin_text(
        &mut self,
        size: Option<usize>,
        line: bool,
    ) -> PyResult<'s, Option<PyStreamRead>> {
        let valid_len = match std::str::from_utf8(&self.stdin_stream_pending) {
            Ok(text) => text.len(),
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            Err(_) => return Err(PyError::value_error("standard input is not valid UTF-8")),
        };
        if self.stdin_stream_eof && valid_len < self.stdin_stream_pending.len() {
            // Bytes past `valid_len` will never be completed by a later read.
            return Err(PyError::value_error("standard input is not valid UTF-8"));
        }
        let valid_text = std::str::from_utf8(&self.stdin_stream_pending[..valid_len])
            .expect("validated as UTF-8 immediately above");
        let requested_chars_available =
            size.is_none_or(|characters| valid_text.chars().count() >= characters);
        let size_end = size.map_or(valid_text.len(), |characters| {
            valid_text
                .char_indices()
                .nth(characters)
                .map_or(valid_text.len(), |(offset, _)| offset)
        });
        let length = if line {
            if let Some(newline) = valid_text.as_bytes()[..size_end]
                .iter()
                .position(|byte| *byte == b'\n')
            {
                newline + 1
            } else if size.is_some() && requested_chars_available {
                size_end
            } else if self.stdin_stream_eof {
                valid_len
            } else {
                return Ok(None);
            }
        } else if size.is_some() {
            if requested_chars_available {
                size_end
            } else if self.stdin_stream_eof {
                valid_len
            } else {
                return Ok(None);
            }
        } else if self.stdin_stream_eof {
            valid_len
        } else {
            return Ok(None);
        };
        let bytes = self.drain_pending_stdin(length);
        let text = String::from_utf8(bytes).expect("length only spans the validated prefix");
        Ok(Some(PyStreamRead::Text(text)))
    }

    /// Remove and account for `length` bytes at the front of the buffered, not-yet-consumed fd 0
    /// bytes retained by [`Vm::read_stream_streaming`].
    fn drain_pending_stdin(&mut self, length: usize) -> Vec<u8> {
        let bytes = self
            .stdin_stream_pending
            .drain(..length)
            .collect::<Vec<u8>>();
        let released = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        self.retained_memory = self.retained_memory.saturating_sub(released);
        self.interp.resources.release_memory(released);
        bytes
    }
}

impl<'s> Vm<'s> {
    /// The namespace a dict handle views, when it is a namespace view rather than a stored
    /// dict.
    fn namespace_view(&self, dict: Value<'s>) -> PyResult<'s, Option<NamespaceHandle<'s>>> {
        match self.get(dict).map_err(PyError::runtime_error)? {
            Object::NamespaceDict(target) => Ok(Some(self.namespace_handle(target))),
            _ => Ok(None),
        }
    }

    /// The name to bind for a key stored in a namespace view. CPython's namespaces are dicts
    /// and accept any hashable key; shellsim's live on name-keyed storage, so other keys are
    /// rejected.
    fn namespace_key(&self, key: &Value<'s>) -> PyResult<'s, String> {
        match self.string_value(key)? {
            Some(name) => Ok(name),
            None => Err(PyError::type_error(format!(
                "namespace keys must be str, not {}",
                self.type_name(key)?
            ))),
        }
    }
}

/// The stored form of a subcommand registered on a parser's subparsers.
fn stored_subcommand(builder: &Builder<'_>, command: PySubcommandSpec<'_>) -> SubcommandSpec {
    SubcommandSpec {
        name: command.name,
        help: command.help,
        parser: builder.store(command.parser.value()),
    }
}

/// The state one step of a lazy iterator needs, copied out of its heap object.
enum LazyStep<'s> {
    Count {
        current: i64,
        step: i64,
    },
    Callable {
        callable: Value<'s>,
        sentinel: Value<'s>,
        exhausted: bool,
    },
}

impl<'s> PyRefs<'s> for Vm<'s> {
    fn handle(&self, r: &heap::Ref) -> Value<'s> {
        Vm::handle(self, r)
    }
}

impl<'s> PyRuntime<'s> for Vm<'s> {
    fn set_test_timeout(&mut self, nanoseconds: Option<u64>) {
        self.execution.test_timeout =
            nanoseconds
                .filter(|value| *value > 0)
                .map(|duration| super::TestTimeout {
                    wall_start: self.interp.clock.monotonic_ns(),
                    cpu_start: self.interp.resources.process_time_ns(),
                    duration,
                });
    }

    fn reserve_memory(&mut self, bytes: usize) -> PyResult<'s, ()> {
        self.reserve_result(bytes).map_err(PyError::resource_error)
    }

    fn charge_cpu(&mut self, units: u64) -> PyResult<'s, ()> {
        Vm::charge_cpu(self, units).map_err(PyError::resource_error)
    }

    fn class_of(&self, value: &Value<'s>) -> PyResult<'s, Value<'s>> {
        self.type_of(value).map_err(PyError::runtime_error)
    }

    fn dictionary_of(&mut self, value: Value<'s>) -> PyResult<'s, Option<Value<'s>>> {
        Vm::dictionary_of(self, value).map_err(PyError::resource_error)
    }

    fn type_metadata(
        &mut self,
        value: Value<'s>,
        field: super::super::native::TypeMetadata,
    ) -> PyResult<'s, Option<Value<'s>>> {
        Vm::type_metadata(self, value, field)
            .map_err(|message| self.raised_or_runtime_error(message))
    }

    fn kind(&self, value: &Value<'s>) -> PyResult<'s, PyKind> {
        if value.inline_string_len().is_some() {
            return Ok(PyKind::String);
        }
        if !value.is_object() {
            return Ok(if value.is_none() {
                PyKind::None
            } else if value.bool_value().is_some() {
                PyKind::Bool
            } else if value.immediate_int().is_some() {
                PyKind::Int
            } else if value.float_value().is_some() {
                PyKind::Float
            } else {
                // Registered values and native markers.
                PyKind::Native
            });
        }
        if self
            .instance_class(*value)
            .map_err(PyError::runtime_error)?
            .is_some()
        {
            if let Object::Bare = self.get(*value).map_err(PyError::runtime_error)? {
                return Ok(PyKind::Instance);
            }
        }
        let object = self.get(*value).map_err(PyError::runtime_error)?;
        Ok(match object {
            Object::Bare => PyKind::Native,
            Object::Float(_) => PyKind::Float,
            Object::String(_) => PyKind::String,
            Object::Bytes(_) => PyKind::Bytes,
            Object::ByteArray(_) => PyKind::ByteArray,
            Object::Exception(_) => PyKind::Native,
            Object::List(_) => PyKind::List,
            Object::BigInt(_) => PyKind::Int,
            Object::Complex { .. } => PyKind::Complex,
            Object::Tuple(_) => PyKind::Tuple,
            Object::Slice { .. } => PyKind::Native,
            Object::Dict(_) | Object::DefaultDict { .. } => PyKind::Dict,
            Object::Set(_) | Object::FrozenSet(_) => PyKind::Set,
            Object::Range { .. } => PyKind::Native,
            Object::Function { .. } | Object::DescriptorBoundMethod { .. } => PyKind::Function,
            Object::Class { .. } => PyKind::Class,
            Object::GenericAlias { .. } => PyKind::Native,
            Object::Iterator { .. }
            | Object::SequenceIterator { .. }
            | Object::ReverseIterator { .. }
            | Object::RangeIterator { .. }
            | Object::CountIterator { .. }
            | Object::StreamIterator { .. }
            | Object::CallableIterator { .. } => PyKind::Iterator,
            Object::Generator { .. } => PyKind::Generator,
            Object::Module { .. } => PyKind::Module,
            Object::NamespaceDict(_) => PyKind::Dict,
            Object::DictView { .. } | Object::MappingProxy(_) => PyKind::Native,
            Object::WideValue { .. } => PyKind::Native,
            Object::Native(_) if object.native::<ArrayObject>().is_some() => PyKind::Array,
            Object::Native(_) => PyKind::Native,
            Object::Property { .. }
            | Object::StaticMethod { .. }
            | Object::ClassMethod { .. }
            | Object::Super { .. }
            | Object::Scope(_) => PyKind::Native,
        })
    }

    fn string_value(&self, value: &Value<'s>) -> PyResult<'s, Option<String>> {
        string::string_value(&self.state.heap, *value).map_err(PyError::runtime_error)
    }

    fn bytes_value(&self, value: &Value<'s>) -> PyResult<'s, Option<Vec<u8>>> {
        string::bytes_value(&self.state.heap, *value).map_err(PyError::runtime_error)
    }

    fn new_string(&mut self, value: String) -> PyResult<'s, Value<'s>> {
        self.allocate_string(value).map_err(PyError::resource_error)
    }

    fn new_bytes(&mut self, value: Vec<u8>) -> PyResult<'s, Value<'s>> {
        self.allocate_bytes(value).map_err(PyError::resource_error)
    }

    fn new_bytearray(&mut self, value: Vec<u8>) -> PyResult<'s, Value<'s>> {
        self.allocate_bytearray(value)
            .map_err(PyError::resource_error)
    }

    fn native_kind(&self, value: &Value<'s>) -> PyResult<'s, Option<PyNativeKind>> {
        if !value.is_object() {
            return Ok(None);
        }
        let object = self.get(*value).map_err(PyError::runtime_error)?;
        Ok(match object {
            Object::Property { .. } => Some(PyNativeKind::Property),
            _ if object.native::<ArrayObject>().is_some() => Some(PyNativeKind::Array),
            _ if object.native::<RegexObject>().is_some() => Some(PyNativeKind::Regex),
            _ if object.native::<MatchObject>().is_some() => Some(PyNativeKind::Match),
            _ if object.native::<ArgumentParserObject>().is_some() => {
                Some(PyNativeKind::ArgumentParser)
            }
            _ if object.native::<RaisesContextObject>().is_some() => {
                Some(PyNativeKind::RaisesContext)
            }
            _ => None,
        })
    }

    fn identity(&self, value: &Value<'s>) -> Option<PyIdentity> {
        Vm::identity(self, *value).ok().flatten().map(PyIdentity)
    }

    fn identical(&self, left: &Value<'s>, right: &Value<'s>) -> bool {
        Vm::identical(self, *left, *right)
    }

    fn nested(
        &mut self,
        f: &mut dyn for<'c> FnMut(&mut dyn PyRuntime<'c>, PyScope<'c, 's>) -> PyResult<'c, ()>,
    ) -> PyResult<'s, ()> {
        let mut child = self.scope();
        f(&mut child, PhantomData)
    }

    fn int_value(&self, value: &Value<'s>) -> Option<i64> {
        number::int_value(&self.state.heap, *value)
    }

    fn is_integer_type(&self, value: &Value<'s>) -> bool {
        matches!(
            value.native_value(),
            Some(NativeValue::BuiltinType(BuiltinType::Int))
        )
    }

    fn is_string_type(&self, value: &Value<'s>) -> bool {
        matches!(
            value.native_value(),
            Some(NativeValue::BuiltinType(BuiltinType::String))
        )
    }

    fn type_name(&self, value: &Value<'s>) -> PyResult<'s, String> {
        self.type_name_of(value).map_err(PyError::runtime_error)
    }

    fn is_ellipsis(&self, value: &Value<'s>) -> bool {
        value.native_value() == Some(NativeValue::Ellipsis)
    }

    fn not_implemented(&self) -> Value<'s> {
        Value::Native(NativeValue::NotImplemented)
    }

    fn is_not_implemented(&self, value: &Value<'s>) -> bool {
        value.native_value() == Some(NativeValue::NotImplemented)
    }

    fn number(&self, value: &Value<'s>) -> Option<super::number::NumberRef<'_>> {
        super::number::view(&self.state.heap, value)
    }

    fn physical_compare(
        &self,
        left: &Value<'s>,
        right: &Value<'s>,
    ) -> PyResult<'s, protocol::Comparison> {
        protocol::compare(&self.state.heap, *left, *right).map_err(PyError::runtime_error)
    }

    fn new_complex(&mut self, real: f64, imag: f64) -> PyResult<'s, Value<'s>> {
        self.allocate_object(Object::Complex { real, imag })
            .map_err(PyError::resource_error)
    }

    fn integer_bigint(&self, value: &Value<'s>) -> PyResult<'s, Option<BigInt>> {
        Ok(match super::number::view(&self.state.heap, value) {
            Some(super::number::NumberRef::Int(value)) => Some(value.into()),
            Some(super::number::NumberRef::BigInt(value)) => Some(value.clone()),
            Some(super::number::NumberRef::UInt(value)) => Some(value.into()),
            Some(super::number::NumberRef::Float(_) | super::number::NumberRef::Complex(..))
            | None => None,
        })
    }

    fn write_stream(&mut self, stream: &Value<'s>, text: &str) -> PyResult<'s, usize> {
        let Some(NativeValue::Stream(stream)) = stream.native_value() else {
            return Err(PyError::type_error("expected a simulated stream"));
        };
        if matches!(stream, Stream::Stdin | Stream::StdinBuffer) {
            return Err(PyError::value_error("standard input is not writable"));
        }
        self.write_output(stream, text.as_bytes());
        Ok(text.chars().count())
    }

    fn read_stream(
        &mut self,
        stream: &Value<'s>,
        size: Option<usize>,
        line: bool,
    ) -> PyResult<'s, PyStreamRead> {
        let binary = match stream.native_value() {
            Some(NativeValue::Stream(Stream::Stdin)) => false,
            Some(NativeValue::Stream(Stream::StdinBuffer)) => true,
            _ => return Err(PyError::value_error("only standard input is readable")),
        };
        if self.mode.scheduler_owned {
            self.read_stream_streaming(size, line, binary)
        } else {
            self.read_stream_buffered(size, line, binary)
        }
    }

    fn truth(&mut self, value: &Value<'s>) -> PyResult<'s, bool> {
        self.truth_value(value).map_err(PyError::runtime_error)
    }

    fn display(&mut self, value: &Value<'s>) -> PyResult<'s, String> {
        self.display_value(value)
            .map_err(|error| self.raised_or_runtime_error(error))
    }

    fn repr(&mut self, value: &Value<'s>) -> PyResult<'s, String> {
        self.repr_value(value)
            .map_err(|error| self.raised_or_runtime_error(error))
    }

    fn payload_repr(&mut self, value: &Value<'s>) -> PyResult<'s, String> {
        self.repr_payload(value, &mut std::collections::BTreeSet::new())
            .map_err(|error| self.raised_or_runtime_error(error))
    }

    fn default_object_repr(&self, value: &Value<'s>) -> PyResult<'s, String> {
        Vm::default_object_repr(self, value).map_err(PyError::runtime_error)
    }

    fn physical_length(&self, value: Value<'s>) -> PyResult<'s, Option<usize>> {
        Vm::physical_length(self, value).map_err(PyError::runtime_error)
    }

    fn format_value(
        &mut self,
        value: &Value<'s>,
        conversion: Option<char>,
        specification: &str,
    ) -> PyResult<'s, String> {
        self.render_formatted_value(value, conversion, specification)
            .map_err(|message| {
                if self.pending_exception.is_some() {
                    PyError::new(PyErrorKind::Raised, message)
                } else {
                    PyError::unsupported(message)
                }
            })
    }

    fn builtin_format(&mut self, value: &Value<'s>, specification: &str) -> PyResult<'s, String> {
        if let Some(rendered) = self
            .format_registered_number(value, specification)
            .map_err(|message| self.raised_or_runtime_error(message))?
        {
            return Ok(rendered);
        }
        let result = if specification.is_empty() {
            self.display_value(value)
        } else {
            self.format_unconverted_value(value, specification)
        };
        result.map_err(|message| self.raised_or_runtime_error(message))
    }

    fn reverse_value(&mut self, value: Value<'s>) -> PyResult<'s, Value<'s>> {
        Vm::reverse_value(self, value).map_err(|message| self.raised_or_runtime_error(message))
    }

    fn reverse_builtin_sequence(&mut self, value: Value<'s>) -> PyResult<'s, Value<'s>> {
        Vm::reverse_builtin_sequence(self, value)
            .map_err(|message| self.raised_or_runtime_error(message))
    }

    fn new_generic_alias(&mut self, origin: Value<'s>, item: Value<'s>) -> PyResult<'s, Value<'s>> {
        Vm::new_generic_alias(self, origin, item)
            .map_err(|message| self.raised_or_runtime_error(message))
    }

    fn equals(&mut self, left: &Value<'s>, right: &Value<'s>) -> PyResult<'s, bool> {
        self.values_equal(left, right)
            .map_err(|message| self.raised_or_runtime_error(message))
    }

    fn physical_equals(&self, left: &Value<'s>, right: &Value<'s>) -> PyResult<'s, bool> {
        protocol::equals(&self.state.heap, *left, *right).map_err(PyError::runtime_error)
    }

    fn container_compare(
        &mut self,
        operator: super::super::ast::ComparisonOperator,
        left: &Value<'s>,
        right: &Value<'s>,
    ) -> PyResult<'s, Option<bool>> {
        Vm::container_compare(self, operator, *left, *right)
            .map_err(|message| self.raised_or_runtime_error(message))
    }

    fn compare(&mut self, left: &Value<'s>, right: &Value<'s>) -> PyResult<'s, Ordering> {
        self.sort_order(left, right).map_err(|message| {
            if self.pending_exception.is_some() {
                PyError::new(PyErrorKind::Raised, message)
            } else {
                PyError::type_error(message)
            }
        })
    }

    fn less_than(&mut self, left: &Value<'s>, right: &Value<'s>) -> PyResult<'s, bool> {
        self.compare_truth(super::super::ast::ComparisonOperator::Less, left, right)
            .map_err(|message| {
                if self.pending_exception.is_some() {
                    PyError::new(PyErrorKind::Raised, message)
                } else {
                    PyError::type_error(message)
                }
            })
    }

    fn get_attribute(&mut self, value: Value<'s>, name: &str) -> PyResult<'s, Option<Value<'s>>> {
        self.resolve_optional_attribute(value, name)
            .map_err(|message| {
                if self.pending_exception.is_some() {
                    PyError::new(PyErrorKind::Raised, message)
                } else {
                    PyError::runtime_error(message)
                }
            })
    }

    fn get_attribute_default(
        &mut self,
        value: Value<'s>,
        name: &str,
    ) -> PyResult<'s, Option<Value<'s>>> {
        let symbol = self.symbol_id(name);
        self.lookup_attribute_default(value, symbol, name)
            .map_err(|message| self.raised_or_runtime_error(message))
    }

    fn set_attribute(&mut self, value: Value<'s>, name: &str, item: Value<'s>) -> PyResult<'s, ()> {
        let symbol = self.intern_symbol(name).map_err(PyError::runtime_error)?;
        self.store_attribute_by_symbol(value, symbol, name, item)
            .map_err(|message| self.raised_or_runtime_error(message))
    }

    fn set_attribute_default(
        &mut self,
        value: Value<'s>,
        name: &str,
        item: Value<'s>,
    ) -> PyResult<'s, ()> {
        let symbol = self.intern_symbol(name).map_err(PyError::runtime_error)?;
        self.store_attribute_default(value, symbol, name, item)
            .map_err(|message| self.raised_or_runtime_error(message))
    }

    fn exception_with_args(&mut self, kind: &'static str, args: Vec<Value<'s>>) -> PyError {
        PyError::new(PyErrorKind::Raised, self.raise_exception_args(kind, args))
    }

    fn generator_stop(&mut self, generator: PyIterator<'s>) -> PyError {
        let generator = generator.value();
        PyError::new(PyErrorKind::Raised, self.raise_stop_iteration(&generator))
    }

    fn exception_args(
        &mut self,
        value: &Value<'s>,
    ) -> PyResult<'s, Option<(String, Vec<Value<'s>>)>> {
        Ok(exception_types::exception_args(self.state, *value)
            .map_err(PyError::runtime_error)?
            .map(|(base, args)| (base.to_string(), args)))
    }

    fn delete_attribute_default(&mut self, value: Value<'s>, name: &str) -> PyResult<'s, ()> {
        let symbol = self.intern_symbol(name).map_err(PyError::runtime_error)?;
        Vm::delete_attribute_default(self, value, symbol, name)
            .map_err(|message| self.raised_or_runtime_error(message))
    }

    fn list_len(&self, list: PyList<'s>) -> PyResult<'s, usize> {
        match self
            .state
            .heap
            .get(list.value())
            .map_err(PyError::runtime_error)?
        {
            Object::List(items) => Ok(items.len()),
            _ => Err(PyError::runtime_error("list handle changed object kind")),
        }
    }

    fn list_items(&mut self, list: PyList<'s>) -> PyResult<'s, Vec<Value<'s>>> {
        let list = list.value();
        let length = match self.get(list).map_err(PyError::runtime_error)? {
            Object::List(items) => items.len(),
            _ => return Err(PyError::runtime_error("list handle changed object kind")),
        };
        let bytes = length
            .checked_mul(std::mem::size_of::<Value<'s>>())
            .ok_or_else(|| PyError::resource_error("list snapshot size overflow"))?;
        self.reserve_memory(bytes)?;
        match self.get(list).map_err(PyError::runtime_error)? {
            Object::List(items) => Ok(self.handles(items)),
            _ => Err(PyError::runtime_error("list handle changed object kind")),
        }
    }

    fn list_append(&mut self, list: PyList<'s>, value: Value<'s>) -> PyResult<'s, ()> {
        let list = list.value();
        if !matches!(self.get(list), Ok(Object::List(_))) {
            return Err(PyError::runtime_error("list handle changed object kind"));
        }
        self.reserve_object_growth(list, MODELED_VALUE_BYTES)
            .map_err(PyError::resource_error)?;
        self.modify(list, |builder, object| {
            let Object::List(items) = object else {
                unreachable!("list kind was checked before reserving growth")
            };
            items.push(builder.store(value));
        })
        .map_err(PyError::runtime_error)
    }

    fn list_insert(
        &mut self,
        list: PyList<'s>,
        index: usize,
        value: Value<'s>,
    ) -> PyResult<'s, ()> {
        let length = self.list_len(list)?;
        let list = list.value();
        let index = index.min(length);
        self.charge_shift(length - index)?;
        self.reserve_object_growth(list, MODELED_VALUE_BYTES)
            .map_err(PyError::resource_error)?;
        self.modify(list, |builder, object| {
            let Object::List(items) = object else {
                unreachable!("list kind was checked before reserving growth")
            };
            items.insert(index, builder.store(value));
        })
        .map_err(PyError::runtime_error)
    }

    fn list_extend(&mut self, list: PyList<'s>, values: Vec<Value<'s>>) -> PyResult<'s, ()> {
        self.list_len(list)?;
        let list = list.value();
        let count = u64::try_from(values.len())
            .map_err(|_| PyError::resource_error("list growth overflow"))?;
        let bytes = count
            .checked_mul(MODELED_VALUE_BYTES)
            .ok_or_else(|| PyError::resource_error("list growth overflow"))?;
        self.reserve_object_growth(list, bytes)
            .map_err(PyError::resource_error)?;
        self.modify(list, |builder, object| {
            let Object::List(items) = object else {
                unreachable!("list kind was checked before reserving growth")
            };
            items.extend(values.into_iter().map(|value| builder.store(value)));
        })
        .map_err(PyError::runtime_error)
    }

    fn list_pop(&mut self, list: PyList<'s>, index: usize) -> PyResult<'s, Value<'s>> {
        let length = self.list_len(list)?;
        let list = list.value();
        self.charge_shift(length.saturating_sub(index))?;
        let value = match self.get(list).map_err(PyError::runtime_error)? {
            Object::List(items) if index < items.len() => self.handle(&items[index]),
            Object::List(_) => {
                return Err(PyError::exception("IndexError", "pop index out of range"))
            }
            _ => return Err(PyError::runtime_error("list handle changed object kind")),
        };
        self.modify(list, |_, object| {
            let Object::List(items) = object else {
                unreachable!("list kind was checked before removal")
            };
            items.remove(index);
        })
        .map_err(PyError::runtime_error)?;
        self.release_object_shrink(list, MODELED_VALUE_BYTES)
            .map_err(PyError::runtime_error)?;
        Ok(value)
    }

    fn list_position(
        &mut self,
        list: PyList<'s>,
        needle: &Value<'s>,
        start: usize,
        stop: usize,
    ) -> PyResult<'s, Option<usize>> {
        let length = self.list_len(list)?;
        let list = list.value();
        for position in start.min(length)..stop.min(length) {
            let candidate = match self.get(list).map_err(PyError::runtime_error)? {
                Object::List(items) => self.handle(&items[position]),
                _ => return Err(PyError::runtime_error("list handle changed object kind")),
            };
            self.charge_cpu(1).map_err(PyError::resource_error)?;
            if self
                .values_equal(&candidate, needle)
                .map_err(|message| self.raised_or_runtime_error(message))?
            {
                return Ok(Some(position));
            }
        }
        Ok(None)
    }

    fn list_reverse(&mut self, list: PyList<'s>) -> PyResult<'s, ()> {
        let length = self.list_len(list)?;
        self.charge_cpu(u64::try_from(length).unwrap_or(u64::MAX))
            .map_err(PyError::resource_error)?;
        let Object::List(items) = self.get_mut(list.value()).map_err(PyError::runtime_error)?
        else {
            unreachable!("list kind was checked before reversal")
        };
        items.reverse();
        Ok(())
    }

    fn list_clear(&mut self, list: PyList<'s>) -> PyResult<'s, ()> {
        let length = self.list_len(list)?;
        let list = list.value();
        let Object::List(items) = self.get_mut(list).map_err(PyError::runtime_error)? else {
            unreachable!("list kind was checked before clearing")
        };
        items.clear();
        let bytes = u64::try_from(length)
            .unwrap_or(u64::MAX)
            .saturating_mul(MODELED_VALUE_BYTES);
        self.release_object_shrink(list, bytes)
            .map_err(PyError::runtime_error)
    }

    fn bytearray_items(&mut self, value: PyByteArray<'s>) -> PyResult<'s, Vec<u8>> {
        let value = value.value();
        let length = match self.get(value).map_err(PyError::runtime_error)? {
            Object::ByteArray(items) => items.len(),
            _ => {
                return Err(PyError::runtime_error(
                    "bytearray handle changed object kind",
                ))
            }
        };
        self.reserve_memory(length)?;
        match self.get(value).map_err(PyError::runtime_error)? {
            Object::ByteArray(items) => Ok(items.clone()),
            _ => Err(PyError::runtime_error(
                "bytearray handle changed object kind",
            )),
        }
    }

    fn replace_bytearray_items(
        &mut self,
        value: PyByteArray<'s>,
        items: Vec<u8>,
    ) -> PyResult<'s, ()> {
        let value = value.value();
        if !matches!(
            self.get(value).map_err(PyError::runtime_error)?,
            Object::ByteArray(_)
        ) {
            return Err(PyError::runtime_error(
                "bytearray handle changed object kind",
            ));
        }
        self.replace_payload(value, Object::ByteArray(items))
            .map_err(PyError::resource_error)
    }

    fn tuple_items(&mut self, tuple: PyTuple<'s>) -> PyResult<'s, Vec<Value<'s>>> {
        let tuple = tuple.value();
        let length = match self.get(tuple).map_err(PyError::runtime_error)? {
            Object::Tuple(items) => items.len(),
            _ => return Err(PyError::runtime_error("tuple handle changed object kind")),
        };
        let bytes = length
            .checked_mul(std::mem::size_of::<Value<'s>>())
            .ok_or_else(|| PyError::resource_error("tuple snapshot size overflow"))?;
        self.reserve_memory(bytes)?;
        match self.get(tuple).map_err(PyError::runtime_error)? {
            Object::Tuple(items) => Ok(self.handles(items)),
            _ => Err(PyError::runtime_error("tuple handle changed object kind")),
        }
    }

    fn slice_parts(
        &mut self,
        value: &Value<'s>,
    ) -> PyResult<'s, Option<super::super::slice::SliceBounds>> {
        self.slice_bounds(value)
            .map_err(|message| self.raised_or_runtime_error(message))
    }

    fn dict_items(&mut self, dict: PyDict<'s>) -> PyResult<'s, Vec<(Value<'s>, Value<'s>)>> {
        let id = dict.value();
        if let Some(target) = self.namespace_view(id)? {
            return self
                .namespace_items(target)
                .map_err(|message| self.raised_or_runtime_error(message));
        }
        let length = match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::Dict(items) | Object::DefaultDict { entries: items, .. } => items.len(),
            _ => return Err(PyError::runtime_error("dict handle changed object kind")),
        };
        let bytes = length
            .checked_mul(std::mem::size_of::<(Value<'s>, Value<'s>)>())
            .ok_or_else(|| PyError::resource_error("dict snapshot size overflow"))?;
        self.reserve_memory(bytes)?;
        match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::Dict(items) | Object::DefaultDict { entries: items, .. } => Ok(items
                .iter()
                .map(|(key, value)| (self.handle(key), self.handle(value)))
                .collect()),
            _ => Err(PyError::runtime_error("dict handle changed object kind")),
        }
    }

    fn dict_last_key(&mut self, dict: PyDict<'s>) -> PyResult<'s, Option<Value<'s>>> {
        let id = dict.value();
        if let Some(target) = self.namespace_view(id)? {
            let items = self
                .namespace_items(target)
                .map_err(|message| self.raised_or_runtime_error(message))?;
            return Ok(items.last().map(|(key, _)| *key));
        }
        match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::Dict(items) | Object::DefaultDict { entries: items, .. } => {
                Ok(items.iter().next_back().map(|(key, _)| self.handle(key)))
            }
            _ => Err(PyError::runtime_error("dict handle changed object kind")),
        }
    }

    fn dict_get(&mut self, dict: PyDict<'s>, key: &Value<'s>) -> PyResult<'s, Option<Value<'s>>> {
        let id = dict.value();
        if let Some(target) = self.namespace_view(id)? {
            // A namespace binds only names, so any other key is absent.
            let Some(name) = self.string_value(key)? else {
                return Ok(None);
            };
            return self
                .namespace_lookup(target, &name)
                .map_err(PyError::runtime_error);
        }
        let Some(position) = self
            .find_mapping_entry(id, key)
            .map_err(|message| self.raised_or_runtime_error(message))?
        else {
            return Ok(None);
        };
        match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                // A key's `__eq__` may have removed the entry after the lookup found it.
                Ok(entries.get(position).map(|entry| self.handle(&entry.1)))
            }
            _ => Err(PyError::runtime_error("dict handle changed object kind")),
        }
    }

    fn dict_insert(
        &mut self,
        dict: PyDict<'s>,
        key: Value<'s>,
        value: Value<'s>,
    ) -> PyResult<'s, ()> {
        let id = dict.value();
        if let Some(target) = self.namespace_view(id)? {
            let name = self.namespace_key(&key)?;
            return self
                .namespace_store(target, name, value)
                .map_err(PyError::resource_error);
        }
        let (hash, position) = self
            .lookup_mapping_entry(id, &key)
            .map_err(|message| self.raised_or_runtime_error(message))?;
        if let Some(position) = position {
            let replaced = self
                .modify(id, |builder, object| match object {
                    Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                        entries.set_value(position, builder.store(value));
                        true
                    }
                    _ => false,
                })
                .map_err(PyError::runtime_error)?;
            if !replaced {
                return Err(PyError::runtime_error("dict handle changed object kind"));
            }
            return Ok(());
        }
        self.reserve_object_growth(id, MODELED_MAPPING_ENTRY_BYTES)
            .map_err(PyError::resource_error)?;
        self.modify(id, |builder, object| match object {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                entries.push(hash, (builder.store(key), builder.store(value)));
            }
            _ => unreachable!("dict kind was checked during lookup"),
        })
        .map_err(PyError::runtime_error)
    }

    fn dict_remove(
        &mut self,
        dict: PyDict<'s>,
        key: &Value<'s>,
    ) -> PyResult<'s, Option<Value<'s>>> {
        let id = dict.value();
        if let Some(target) = self.namespace_view(id)? {
            let Some(name) = self.string_value(key)? else {
                return Ok(None);
            };
            // Removing an attribute can convert a shaped instance to dictionary storage, which
            // charges memory.
            return self
                .namespace_delete(target, &name)
                .map_err(PyError::resource_error);
        }
        let Some(position) = self
            .find_mapping_entry(id, key)
            .map_err(|message| self.raised_or_runtime_error(message))?
        else {
            return Ok(None);
        };
        let value = match self.get(id).map_err(PyError::runtime_error)? {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                // A key's `__eq__` may already have removed the entry the lookup found.
                match entries.get(position) {
                    Some(entry) => self.handle(&entry.1),
                    None => return Ok(None),
                }
            }
            _ => return Err(PyError::runtime_error("dict handle changed object kind")),
        };
        self.modify(id, |_, object| {
            let (Object::Dict(entries) | Object::DefaultDict { entries, .. }) = object else {
                unreachable!("dict kind was checked before removal")
            };
            entries.remove(position);
        })
        .map_err(PyError::runtime_error)?;
        self.release_object_shrink(id, MODELED_MAPPING_ENTRY_BYTES)
            .map_err(PyError::runtime_error)?;
        Ok(Some(value))
    }

    fn replace_dict_items(
        &mut self,
        dict: PyDict<'s>,
        items: Vec<(Value<'s>, Value<'s>)>,
    ) -> PyResult<'s, ()> {
        let id = dict.value();
        if let Some(target) = self.namespace_view(id)? {
            let mut kept = Vec::with_capacity(items.len());
            for (key, value) in items {
                kept.push((self.namespace_key(&key)?, value));
            }
            let names = kept
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<std::collections::HashSet<_>>();
            let removed = self
                .namespace_entries(target)
                .map_err(PyError::runtime_error)?
                .into_iter()
                .map(|(name, _)| name)
                .filter(|name| !names.contains(name.as_str()))
                .collect::<Vec<_>>();
            PyRuntime::charge_cpu(
                self,
                u64::try_from(removed.len() + kept.len()).unwrap_or(u64::MAX),
            )?;
            for name in removed {
                self.namespace_delete(target, &name)
                    .map_err(PyError::resource_error)?;
            }
            for (name, value) in kept {
                self.namespace_store(target, name, value)
                    .map_err(PyError::resource_error)?;
            }
            return Ok(());
        }
        let entries = self
            .ordered_map(items)
            .map_err(|message| self.raised_or_runtime_error(message))?;
        let old_len = match self.get(id).map_err(PyError::runtime_error)? {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries.len(),
            _ => return Err(PyError::runtime_error("dict handle changed object kind")),
        };
        let new_len = entries.len();
        self.replace_container_payload(
            id,
            old_len,
            new_len,
            MODELED_MAPPING_ENTRY_BYTES,
            |builder, object| {
                let (Object::Dict(current)
                | Object::DefaultDict {
                    entries: current, ..
                }) = object
                else {
                    unreachable!("dict kind was checked before replacement")
                };
                *current = entries.into_map(builder);
            },
        )
    }

    fn dict_copy(&mut self, dict: PyDict<'s>) -> PyResult<'s, Value<'s>> {
        if let Some(target) = self.namespace_view(dict.value())? {
            return self
                .namespace_snapshot_dict(target)
                .map_err(PyError::resource_error);
        }
        // Read the entries as handles first: the copy's allocation may move the source.
        let (factory, entries) = match self.get(dict.value()).map_err(PyError::runtime_error)? {
            Object::Dict(entries) => (None, entries),
            Object::DefaultDict { factory, entries } => (Some(self.handle(factory)), entries),
            _ => return Err(PyError::runtime_error("dict handle changed object kind")),
        };
        let entries = entries
            .iter_hashed()
            .map(|(hash, (key, value))| (hash, self.handle(key), self.handle(value)))
            .collect::<Vec<_>>();
        self.alloc_with(|builder| {
            let mut copy = OrderedMap::default();
            for (hash, key, value) in entries {
                copy.push(hash, (builder.store(key), builder.store(value)));
            }
            match factory {
                None => Object::Dict(copy),
                Some(factory) => Object::DefaultDict {
                    factory: builder.store(factory),
                    entries: copy,
                },
            }
        })
        .map_err(PyError::resource_error)
    }

    fn get_item(&mut self, container: Value<'s>, key: Value<'s>) -> PyResult<'s, Value<'s>> {
        self.subscript_value(container, key)
            .map_err(|message| self.raised_or_runtime_error(message))
    }

    fn builtin_get_item(
        &mut self,
        container: Value<'s>,
        key: Value<'s>,
    ) -> PyResult<'s, Value<'s>> {
        self.subscript_builtin(container, key)
            .map_err(|message| self.raised_or_runtime_error(message))
    }

    fn builtin_contains(&mut self, container: Value<'s>, item: Value<'s>) -> PyResult<'s, bool> {
        self.contains_value(&container, &item)
            .map_err(|message| self.raised_or_runtime_error(message))
    }

    fn mapping_items(
        &mut self,
        value: Value<'s>,
    ) -> PyResult<'s, Option<Vec<(Value<'s>, Value<'s>)>>> {
        Vm::mapping_items(self, value).map_err(|message| self.raised_or_runtime_error(message))
    }

    fn new_dict_view(&mut self, kind: DictViewKind, mapping: Value<'s>) -> PyResult<'s, Value<'s>> {
        if !mapping.is_object() {
            return Err(PyError::runtime_error("a dict view needs a mapping object"));
        }
        self.alloc_with(|builder| Object::DictView {
            kind,
            mapping: builder.store(mapping),
        })
        .map_err(PyError::resource_error)
    }

    fn dict_view(&self, value: &Value<'s>) -> PyResult<'s, Option<(DictViewKind, Value<'s>)>> {
        if !value.is_object() {
            return Ok(None);
        }
        match self.get(*value).map_err(PyError::runtime_error)? {
            Object::DictView { kind, mapping } => Ok(Some((*kind, self.handle(mapping)))),
            _ => Ok(None),
        }
    }

    fn set_items(&mut self, set: PySet<'s>) -> PyResult<'s, Vec<Value<'s>>> {
        let set = set.value();
        let length = match self.get(set).map_err(PyError::runtime_error)? {
            Object::Set(items) | Object::FrozenSet(items) => items.len(),
            _ => return Err(PyError::runtime_error("set handle changed object kind")),
        };
        let bytes = length
            .checked_mul(std::mem::size_of::<Value<'s>>())
            .ok_or_else(|| PyError::resource_error("set snapshot size overflow"))?;
        self.reserve_memory(bytes)?;
        match self.get(set).map_err(PyError::runtime_error)? {
            Object::Set(items) | Object::FrozenSet(items) => Ok(self.handles(items.iter())),
            _ => Err(PyError::runtime_error("set handle changed object kind")),
        }
    }

    fn set_is_frozen(&self, set: PySet<'s>) -> PyResult<'s, bool> {
        match self.get(set.value()).map_err(PyError::runtime_error)? {
            Object::FrozenSet(_) => Ok(true),
            Object::Set(_) => Ok(false),
            _ => Err(PyError::runtime_error("set handle changed object kind")),
        }
    }

    fn set_insert(&mut self, set: PySet<'s>, value: Value<'s>) -> PyResult<'s, bool> {
        let id = set.value();
        let (hash, position) = self
            .lookup_set_entry(id, &value)
            .map_err(|message| self.raised_or_runtime_error(message))?;
        if position.is_some() {
            return Ok(false);
        }
        self.reserve_object_growth(id, MODELED_SET_MEMBER_BYTES)
            .map_err(PyError::resource_error)?;
        self.modify(id, |builder, object| {
            let Object::Set(items) = object else {
                unreachable!("set kind was checked during lookup")
            };
            items.push(hash, builder.store(value));
        })
        .map_err(PyError::runtime_error)?;
        Ok(true)
    }

    fn set_first(&mut self, set: PySet<'s>) -> PyResult<'s, Option<Value<'s>>> {
        match self.get(set.value()).map_err(PyError::runtime_error)? {
            Object::Set(items) | Object::FrozenSet(items) => {
                Ok(items.iter().next().map(|item| self.handle(item)))
            }
            _ => Err(PyError::runtime_error("set handle changed object kind")),
        }
    }

    fn set_remove(&mut self, set: PySet<'s>, value: &Value<'s>) -> PyResult<'s, bool> {
        let id = set.value();
        let Some(position) = self
            .find_set_entry(id, value)
            .map_err(|message| self.raised_or_runtime_error(message))?
        else {
            return Ok(false);
        };
        let removed = self
            .modify(id, |_, object| match object {
                Object::Set(items) => Some(items.remove(position).is_some()),
                _ => None,
            })
            .map_err(PyError::runtime_error)?;
        match removed {
            None => Err(PyError::runtime_error("set handle changed object kind")),
            Some(false) => Ok(false),
            Some(true) => {
                self.release_object_shrink(id, MODELED_SET_MEMBER_BYTES)
                    .map_err(PyError::runtime_error)?;
                Ok(true)
            }
        }
    }

    fn replace_set_items(&mut self, set: PySet<'s>, items: Vec<Value<'s>>) -> PyResult<'s, ()> {
        let id = set.value();
        match self.get(id).map_err(PyError::runtime_error)? {
            Object::Set(_) => {}
            Object::FrozenSet(_) => {
                return Err(PyError::runtime_error("frozenset items cannot be replaced"))
            }
            _ => return Err(PyError::runtime_error("set handle changed object kind")),
        }
        let members = self
            .distinct_members(items)
            .map_err(|message| self.raised_or_runtime_error(message))?;
        // Hashing and `__eq__` ran guest code, which may have changed the set.
        let old_len = match self.get(id).map_err(PyError::runtime_error)? {
            Object::Set(items) => items.len(),
            _ => return Err(PyError::runtime_error("set handle changed object kind")),
        };
        let new_len = members.len();
        self.replace_container_payload(
            id,
            old_len,
            new_len,
            MODELED_SET_MEMBER_BYTES,
            |builder, object| {
                let Object::Set(current) = object else {
                    unreachable!("set kind was checked before replacement")
                };
                *current = members.into_set(builder);
            },
        )
    }

    fn property_getter(&self, property: PyProperty<'s>) -> PyResult<'s, Value<'s>> {
        match self.get(property.value()).map_err(PyError::runtime_error)? {
            Object::Property { getter, .. } => Ok(self.handle(getter)),
            _ => Err(PyError::runtime_error(
                "property handle changed object kind",
            )),
        }
    }

    fn property_setter(&self, property: PyProperty<'s>) -> PyResult<'s, Option<Value<'s>>> {
        match self.get(property.value()).map_err(PyError::runtime_error)? {
            Object::Property { setter, .. } => Ok(self.handle_optional(setter.as_ref())),
            _ => Err(PyError::runtime_error(
                "property handle changed object kind",
            )),
        }
    }

    fn new_property(
        &mut self,
        getter: Value<'s>,
        setter: Option<Value<'s>>,
    ) -> PyResult<'s, Value<'s>> {
        self.alloc_with(|builder| Object::Property {
            getter: builder.store(getter),
            setter: builder.optional(setter),
        })
        .map_err(PyError::resource_error)
    }

    fn new_builtin_instance(
        &mut self,
        builtin: BuiltinType,
        class: Value<'s>,
        args: CallArgs<'s>,
    ) -> PyResult<'s, Value<'s>> {
        let (arguments, keyword_arguments) = args.into_parts();
        Vm::new_builtin_instance(self, builtin, class, arguments, keyword_arguments)
            .map_err(|error| self.raised_or_runtime_error(error))
    }

    fn new_instance(&mut self, class: Value<'s>, has_arguments: bool) -> PyResult<'s, Value<'s>> {
        Vm::new_instance(self, class, has_arguments)
            .map_err(|error| self.raised_or_runtime_error(error))
    }

    fn new_type(
        &mut self,
        metaclass: Value<'s>,
        name: String,
        bases: Value<'s>,
        namespace: Value<'s>,
    ) -> PyResult<'s, Value<'s>> {
        let bases = match bases.is_object().then(|| self.get(bases).ok()).flatten() {
            Some(Object::Tuple(values)) => self.handles(values),
            _ => return Err(PyError::type_error("type.__new__() bases must be a tuple")),
        };
        let entries = match namespace
            .is_object()
            .then(|| self.get(namespace).ok())
            .flatten()
        {
            Some(Object::Dict(entries)) => entries
                .iter()
                .map(|(key, value)| (self.handle(key), self.handle(value)))
                .collect::<Vec<_>>(),
            _ => {
                return Err(PyError::type_error(
                    "type.__new__() namespace must be a dict",
                ))
            }
        };
        let mut attributes = HashMap::new();
        for (key, value) in entries {
            let key = string::string_value(&self.state.heap, key)
                .map_err(PyError::runtime_error)?
                .ok_or_else(|| PyError::type_error("type.__new__() keys must be strings"))?;
            attributes.insert(key, value);
        }
        let mut user_bases = Vec::new();
        let mut layout = ClassLayout::Object;
        let mut exception_base = None;
        for base in &bases {
            if base.is_object() {
                let Object::Class(base_object) = self.get(*base).map_err(PyError::runtime_error)?
                else {
                    return Err(PyError::type_error("type.__new__() bases must be classes"));
                };
                let (base_layout, base_exception) =
                    (&base_object.layout, &base_object.exception_base);
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
                if let Some(base_exception) = base_exception {
                    exception_base.get_or_insert(*base_exception);
                }
                user_bases.push(*base);
            } else {
                match base.native_value() {
                    Some(NativeValue::BuiltinType(
                        BuiltinType::Object | BuiltinType::Enum | BuiltinType::TestCase,
                    )) => {}
                    Some(NativeValue::BuiltinType(builtin))
                        if super::objects::is_subclassable_builtin(builtin)
                            && layout == ClassLayout::Object =>
                    {
                        layout = ClassLayout::Builtin(builtin);
                    }
                    Some(NativeValue::BuiltinType(BuiltinType::Type))
                        if layout == ClassLayout::Object =>
                    {
                        layout = ClassLayout::Type;
                    }
                    Some(NativeValue::ExceptionType(ExceptionType(name)))
                        if layout == ClassLayout::Object =>
                    {
                        exception_base.get_or_insert(name);
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
            mro,
            metaclass,
            layout,
            exception_base,
            attributes,
            dataclass_fields: Vec::new(),
            enum_members: Vec::new(),
        })
        .map_err(PyError::runtime_error)
    }

    fn replace_list_items(&mut self, list: PyList<'s>, items: Vec<Value<'s>>) -> PyResult<'s, ()> {
        let id = list.value();
        let old_len = match self.get(id).map_err(PyError::runtime_error)? {
            Object::List(current) => current.len(),
            _ => return Err(PyError::runtime_error("list handle changed object kind")),
        };
        let new_len = items.len();
        self.replace_container_payload(
            id,
            old_len,
            new_len,
            MODELED_VALUE_BYTES,
            |builder, object| {
                let Object::List(current) = object else {
                    unreachable!("list kind was checked before replacement")
                };
                *current = builder.refs(items);
            },
        )
    }

    fn call_type_default(
        &mut self,
        class: Value<'s>,
        args: CallArgs<'s>,
    ) -> PyResult<'s, Value<'s>> {
        match Vm::call_type_default(self, class, args)
            .map_err(|message| self.raised_or_runtime_error(message))?
        {
            CallResult::Value(value) => Ok(value),
            CallResult::Exit(status) => Err(PyError::exit(status)),
            CallResult::EnteredFrame | CallResult::Blocked(..) | CallResult::Retry(..) => {
                Err(PyError::runtime_error("default type call did not finish"))
            }
        }
    }

    fn call_value(&mut self, callable: Value<'s>, args: CallArgs<'s>) -> PyResult<'s, Value<'s>> {
        let (positional, keywords) = args.into_parts();
        let argument_count = positional.len();
        let total = argument_count
            .checked_add(keywords.len())
            .ok_or_else(|| PyError::resource_error("too many call arguments"))?;
        let unpacked = vec![false; total];
        let keyword_names = keywords
            .iter()
            .map(|(name, _)| Some(name.clone()))
            .collect::<Vec<_>>();
        self.push(callable);
        for value in positional {
            self.push(value);
        }
        for (_, value) in keywords {
            self.push(value);
        }
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

    fn is_callable(&self, value: &Value<'s>) -> PyResult<'s, bool> {
        Ok(
            if matches!(
                value.native_value(),
                Some(
                    NativeValue::Function(_)
                        | NativeValue::BuiltinType(_)
                        | NativeValue::ValueKind(_)
                        | NativeValue::NativeFunction(_)
                        | NativeValue::NativeMethod(_)
                        | NativeValue::NativeClassMethod(_)
                        | NativeValue::ExceptionType(_)
                )
            ) || self
                .registered_kind(value)
                .is_some_and(|kind| kind.call.is_some())
            {
                true
            } else if let Some(class) = self
                .instance_class(*value)
                .map_err(PyError::runtime_error)?
            {
                // An instance is callable when its class or an ancestor defines `__call__`.
                let heap = &self.state.heap;
                let defines_call = |class: &heap::Ref| {
                    matches!(
                        heap.get(heap.handle(class)),
                        Ok(Object::Class(class_object))
                            if class_object.attributes.contains_key("__call__")
                    )
                };
                match heap.get(class).map_err(PyError::runtime_error)? {
                    Object::Class(class_object) => {
                        defines_call(&heap.store(class))
                            || class_object.mro.iter().any(defines_call)
                    }
                    _ => false,
                }
            } else if value.is_object() {
                matches!(
                    self.get(*value).map_err(PyError::runtime_error)?,
                    Object::Function { .. }
                        | Object::Class { .. }
                        | Object::DescriptorBoundMethod { .. }
                )
            } else {
                false
            },
        )
    }

    fn is_iterator(&self, value: &Value<'s>) -> PyResult<'s, bool> {
        Vm::is_iterator(self, value).map_err(PyError::runtime_error)
    }

    fn is_unbounded_iterator(&self, value: &Value<'s>) -> PyResult<'s, bool> {
        Vm::is_unbounded_iterator(self, value).map_err(PyError::runtime_error)
    }

    fn payload_iterator(&mut self, value: Value<'s>) -> PyResult<'s, Option<Value<'s>>> {
        Vm::payload_iterator(self, value).map_err(PyError::runtime_error)
    }

    fn is_user_instance(&self, value: &Value<'s>) -> PyResult<'s, bool> {
        Ok(self
            .instance_class(*value)
            .map_err(PyError::runtime_error)?
            .is_some())
    }

    fn iterator(&mut self, value: Value<'s>) -> PyResult<'s, PyIterator<'s>> {
        if Vm::is_iterator(self, &value).map_err(PyError::runtime_error)? {
            return value.cast(self);
        }
        self.make_iterator(value)
            .map_err(|error| {
                if self.pending_exception.is_some() {
                    PyError::new(PyErrorKind::Raised, error)
                } else {
                    PyError::type_error(error)
                }
            })?
            .cast(self)
    }

    fn iterator_next(&mut self, iterator: PyIterator<'s>) -> PyResult<'s, Option<Value<'s>>> {
        let id = iterator.value();
        // Step a materialized iterator in place: cloning its payload would copy every remaining
        // value on each step.
        let materialized = match self.get(id).map_err(PyError::runtime_error)? {
            Object::Iterator { values, position } => {
                Some(values.get(*position).map(|value| self.handle(value)))
            }
            _ => None,
        };
        if let Some(value) = materialized {
            if value.is_some() {
                let Object::Iterator { position, .. } =
                    self.get_mut(id).map_err(PyError::runtime_error)?
                else {
                    unreachable!("iterator kind was checked above")
                };
                *position += 1;
            }
            return Ok(value);
        }
        // Copy out only the fields a step needs; cloning a payload such as an instance or a
        // generator frame on every step would make iteration quadratic.
        let step = match self.get(id).map_err(PyError::runtime_error)? {
            Object::CountIterator { current, step } => LazyStep::Count {
                current: *current,
                step: *step,
            },
            Object::CallableIterator {
                callable,
                sentinel,
                exhausted,
            } => LazyStep::Callable {
                callable: self.handle(callable),
                sentinel: self.handle(sentinel),
                exhausted: *exhausted,
            },
            Object::SequenceIterator { .. }
            | Object::ReverseIterator { .. }
            | Object::RangeIterator { .. }
            | Object::StreamIterator { .. } => {
                return self
                    .next_stored_iterator(id)
                    .map_err(|error| self.raised_or_runtime_error(error));
            }
            Object::Generator { .. } => {
                return self.resume_generator(id).map_err(PyError::runtime_error);
            }
            _ => {
                return self
                    .next_until_stop(&id)
                    .map_err(|error| self.raised_or_runtime_error(error));
            }
        };
        match step {
            LazyStep::Count { current, step } => {
                let value = current;
                let next = super::super::stdlib::itertools::count_next(current, step)
                    .map_err(PyError::overflow_error)?;
                let Object::CountIterator { current, .. } =
                    self.get_mut(id).map_err(PyError::runtime_error)?
                else {
                    return Err(PyError::runtime_error("iterator changed object kind"));
                };
                *current = next;
                Ok(Some(Value::Int(value)))
            }
            LazyStep::Callable {
                callable,
                sentinel,
                exhausted,
            } => {
                if exhausted {
                    return Ok(None);
                }
                self.charge_cpu(1).map_err(PyError::resource_error)?;
                let value = self.call_value(callable, CallArgs::new(Vec::new(), Vec::new()))?;
                if self
                    .values_equal(&value, &sentinel)
                    .map_err(|message| self.raised_or_runtime_error(message))?
                {
                    let Object::CallableIterator { exhausted, .. } =
                        self.get_mut(id).map_err(PyError::runtime_error)?
                    else {
                        return Err(PyError::runtime_error("iterator changed object kind"));
                    };
                    *exhausted = true;
                    Ok(None)
                } else {
                    Ok(Some(value))
                }
            }
        }
    }

    fn generator_send(
        &mut self,
        generator: PyIterator<'s>,
        value: Value<'s>,
    ) -> PyResult<'s, Option<Value<'s>>> {
        let id = generator.value();
        if !matches!(
            self.state.heap.get(id).map_err(PyError::runtime_error)?,
            Object::Generator { .. }
        ) {
            return Err(PyError::type_error("expected a generator"));
        }
        self.resume_generator_with(id, value)
            .map_err(PyError::runtime_error)
    }

    fn generator_return_value(&self, generator: PyIterator<'s>) -> PyResult<'s, Value<'s>> {
        match self
            .get(generator.value())
            .map_err(PyError::runtime_error)?
        {
            Object::Generator(generator) if generator.exhausted => {
                Ok(self.handle(&generator.return_value))
            }
            Object::Generator(_) => Err(PyError::runtime_error("coroutine has not completed")),
            _ => Err(PyError::type_error("expected a coroutine")),
        }
    }

    fn coroutine_step(
        &mut self,
        coroutine: PyIterator<'s>,
        value: Value<'s>,
    ) -> PyResult<'s, (u8, Value<'s>)> {
        let id = coroutine.value();
        if !matches!(
            self.state.heap.get(id).map_err(PyError::runtime_error)?,
            Object::Generator { .. }
        ) {
            return Err(PyError::type_error("expected a coroutine"));
        }
        match self.resume_generator_with(id, value) {
            Ok(Some(value)) => Ok((0, value)),
            Ok(None) => self
                .generator_return_value(coroutine)
                .map(|value| (1, value)),
            Err(error) => match self.pending_exception.take() {
                Some(exception) => Ok((2, self.handle(&exception.value))),
                None => Err(PyError::runtime_error(error)),
            },
        }
    }

    /// `generator.close()`: raise `GeneratorExit` at the suspended `yield`. The generator may run
    /// cleanup code, but yielding another value is an error.
    fn generator_close(&mut self, generator: PyIterator<'s>) -> PyResult<'s, ()> {
        let id = generator.value();
        if !matches!(
            self.state.heap.get(id).map_err(PyError::runtime_error)?,
            Object::Generator { .. }
        ) {
            return Err(PyError::type_error("expected a generator"));
        }
        self.close_generator(id)
            .map_err(|error| self.raised_or_runtime_error(error))
    }

    /// `generator.throw(exception)`: raise `exception`, an instance or an exception class, at the
    /// suspended `yield` and return the next value the generator yields.
    fn generator_throw(&mut self, generator: PyIterator<'s>, exception: Value<'s>) -> PyResult<'s> {
        let id = generator.value();
        if !matches!(
            self.state.heap.get(id).map_err(PyError::runtime_error)?,
            Object::Generator { .. }
        ) {
            return Err(PyError::type_error("expected a generator"));
        }
        // The exception is stored into a `RaisedException` only at the call, so no stored
        // reference is held across the allocation below.
        let (kind, value) = if let Some(kind) =
            exception_types::exception_type_name(self.state, exception)
                .map_err(PyError::runtime_error)?
        {
            (kind, exception)
        } else if let Some(NativeValue::ExceptionType(ExceptionType(kind))) =
            exception.native_value()
        {
            let value = self
                .allocate_exception(kind.into(), String::new())
                .map_err(PyError::resource_error)?;
            (kind.into(), value)
        } else {
            return Err(PyError::type_error(
                "exceptions must be classes or instances deriving from BaseException, not "
                    .to_string()
                    + &self.type_name(&exception)?,
            ));
        };
        let raised = RaisedException {
            kind,
            value: self.store(value),
        };
        match self.throw_into_generator(id, raised) {
            Ok(Some(value)) => Ok(value),
            Ok(None) => Err(PyError::new(
                PyErrorKind::Raised,
                self.raise_stop_iteration(&id),
            )),
            Err(error) if self.pending_exception.is_some() => {
                Err(PyError::new(PyErrorKind::Raised, error))
            }
            Err(error) => Err(PyError::runtime_error(error)),
        }
    }

    fn new_iterator(&mut self, values: Vec<Value<'s>>) -> PyResult<'s, Value<'s>> {
        self.alloc_with(|builder| Object::Iterator {
            values: builder.refs(values),
            position: 0,
        })
        .map_err(PyError::resource_error)
    }

    fn new_count_iterator(&mut self, start: i64, step: i64) -> PyResult<'s, Value<'s>> {
        Vm::allocate_object(
            self,
            Object::CountIterator {
                current: start,
                step,
            },
        )
        .map_err(PyError::resource_error)
    }

    fn new_default_dict(&mut self, factory: PyCallable<'s>) -> PyResult<'s, Value<'s>> {
        self.alloc_with(|builder| Object::DefaultDict {
            factory: builder.store(factory.into_value()),
            entries: Default::default(),
        })
        .map_err(PyError::resource_error)
    }

    fn new_list(&mut self, items: Vec<Value<'s>>) -> PyResult<'s, Value<'s>> {
        self.alloc_with(|builder| Object::List(builder.refs(items)))
            .map_err(PyError::resource_error)
    }

    fn new_import_path(&mut self) -> PyResult<'s, Value<'s>> {
        if let Some(path) = &self.state.sys_path {
            return Ok(self.handle(path));
        }
        let mut values = Vec::with_capacity(self.state.import_paths.len());
        for path in self.state.import_paths.clone() {
            values.push(
                self.allocate_string(path)
                    .map_err(PyError::resource_error)?,
            );
        }
        let path = self.new_list(values)?;
        self.state.sys_path = Some(self.store(path));
        Ok(path)
    }

    fn new_tuple(&mut self, items: Vec<Value<'s>>) -> PyResult<'s, Value<'s>> {
        self.alloc_with(|builder| Object::Tuple(builder.refs(items)))
            .map_err(PyError::resource_error)
    }

    fn new_dict(&mut self, items: Vec<(Value<'s>, Value<'s>)>) -> PyResult<'s, Value<'s>> {
        let entries = self
            .ordered_map(items)
            .map_err(|message| self.raised_or_runtime_error(message))?;
        self.alloc_with(|builder| Object::Dict(entries.into_map(builder)))
            .map_err(PyError::resource_error)
    }

    fn new_set(&mut self, items: Vec<Value<'s>>) -> PyResult<'s, Value<'s>> {
        let members = self
            .distinct_members(items)
            .map_err(|message| self.raised_or_runtime_error(message))?;
        self.alloc_with(|builder| Object::Set(members.into_set(builder)))
            .map_err(PyError::resource_error)
    }

    fn new_frozen_set(&mut self, items: Vec<Value<'s>>) -> PyResult<'s, Value<'s>> {
        let members = self
            .distinct_members(items)
            .map_err(|message| self.raised_or_runtime_error(message))?;
        self.alloc_with(|builder| Object::FrozenSet(members.into_set(builder)))
            .map_err(PyError::resource_error)
    }

    fn new_value_kind(
        &self,
        kind: &'static super::super::native::ValueKindDef,
        payload: u64,
    ) -> PyResult<'s, Value<'s>> {
        let index = self
            .state
            .types
            .value_kind_index(kind)
            .ok_or_else(|| PyError::runtime_error("value kind is not registered"))?;
        Ok(Value::registered(index, payload))
    }

    fn value_kind_payload(
        &self,
        value: &Value<'s>,
        kind: &'static super::super::native::ValueKindDef,
    ) -> Option<u64> {
        let (index, payload) = value.registered_parts()?;
        std::ptr::eq(self.state.types.value_kind(index)?, kind).then_some(payload)
    }

    fn new_wide_value_kind(
        &mut self,
        kind: &'static super::super::native::ValueKindDef,
        payload: [u64; 2],
    ) -> PyResult<'s, Value<'s>> {
        let index = self
            .state
            .types
            .value_kind_index(kind)
            .ok_or_else(|| PyError::runtime_error("value kind is not registered"))?;
        let type_id = self
            .state
            .types
            .value_kind_type_id_by_index(index)
            .ok_or_else(|| PyError::runtime_error("value kind is not registered"))?;
        Vm::allocate_object(
            self,
            Object::WideValue {
                type_id,
                kind: index,
                payload,
            },
        )
        .map_err(PyError::resource_error)
    }

    fn wide_value_kind_payload(
        &self,
        value: &Value<'s>,
        kind: &'static super::super::native::ValueKindDef,
    ) -> Option<[u64; 2]> {
        let Object::WideValue {
            kind: index,
            payload,
            ..
        } = self
            .get(value.immediate().is_none().then_some(*value)?)
            .ok()?
        else {
            return None;
        };
        std::ptr::eq(self.state.types.value_kind(*index)?, kind).then_some(*payload)
    }

    fn value_kind_of(
        &self,
        value: &Value<'s>,
    ) -> Option<&'static super::super::native::ValueKindDef> {
        self.registered_kind(value)
    }

    fn builtin_type(&self, name: &str) -> Option<Value<'s>> {
        BuiltinType::ALL
            .iter()
            .find(|builtin| builtin.name() == name)
            .map(|builtin| Value::Native(NativeValue::BuiltinType(*builtin)))
    }

    fn type_object(&self, value: &Value<'s>) -> Option<PyTypeObject> {
        match value.native_value()? {
            NativeValue::BuiltinType(builtin) => Some(PyTypeObject::Builtin(builtin.name())),
            NativeValue::ValueKind(kind) => Some(PyTypeObject::Kind(kind)),
            _ => None,
        }
    }

    fn value_kind_type(
        &self,
        kind: &'static super::super::native::ValueKindDef,
    ) -> PyResult<'s, Value<'s>> {
        self.state
            .types
            .value_kind_type_id(kind)
            .ok_or_else(|| PyError::runtime_error("value kind is not registered"))
            .and_then(|type_id| self.type_value(type_id).map_err(PyError::runtime_error))
    }

    fn new_array(
        &mut self,
        buffer: PyArrayBuffer<'s>,
        dtype: PyArrayDtype,
        shape: Vec<usize>,
        strides: Vec<isize>,
    ) -> PyResult<'s, Value<'s>> {
        let count = shape
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
            .and_then(|count| count.checked_mul(dtype.itemsize()))
            .ok_or_else(|| PyError::value_error("array is too big."))?;
        let values = matches!(buffer, PyArrayBuffer::Values(_));
        if count != buffer.byte_len() || dtype.is_values() != values {
            return Err(PyError::runtime_error(
                "array storage does not match its shape and dtype",
            ));
        }
        let view = PyArrayView {
            strides,
            dtype,
            shape,
            offset: 0,
            writeable: true,
        };
        validate_array_view(&view, values, buffer.byte_len())?;
        let storage = match buffer {
            PyArrayBuffer::Bytes(bytes) => {
                self.alloc(Object::Native(Box::new(ArrayStorage::Bytes(bytes))))
            }
            PyArrayBuffer::Values(values) => self.alloc_with(|builder| {
                Object::Native(Box::new(ArrayStorage::Values(builder.refs(values))))
            }),
        }
        .map_err(PyError::resource_error)?;
        self.alloc_with(|builder| {
            Object::Native(Box::new(ArrayObject {
                storage: builder.store(storage),
                view,
                base: None,
            }))
        })
        .map_err(PyError::resource_error)
    }

    fn new_array_view(&mut self, base: PyArray<'s>, view: PyArrayView) -> PyResult<'s, Value<'s>> {
        let array = self.array_object(base)?;
        let (storage, base_writeable, owner) = (
            self.handle(&array.storage),
            array.view.writeable,
            self.handle_optional(array.base.as_ref())
                .unwrap_or(base.value()),
        );
        if view.writeable && !base_writeable {
            return Err(PyError::runtime_error(
                "a view of a read-only array cannot be writeable",
            ));
        }
        let buffer = self.array_storage_object(storage)?;
        validate_array_view(
            &view,
            matches!(buffer, ArrayStorage::Values(_)),
            buffer.byte_len(),
        )?;
        self.alloc_with(|builder| {
            Object::Native(Box::new(ArrayObject {
                storage: builder.store(storage),
                view,
                base: Some(builder.store(owner)),
            }))
        })
        .map_err(PyError::resource_error)
    }

    fn array_view(&self, array: PyArray<'s>) -> PyResult<'s, PyArrayView> {
        Ok(self.array_object(array)?.view.clone())
    }

    fn array_storage(&self, array: PyArray<'s>) -> PyResult<'s, PyIdentity> {
        let storage = self.handle(&self.array_object(array)?.storage);
        Vm::identity(self, storage)
            .map_err(PyError::runtime_error)?
            .map(PyIdentity)
            .ok_or_else(|| PyError::runtime_error("array storage is not an object"))
    }

    fn array_base(&self, array: PyArray<'s>) -> PyResult<'s, Option<Value<'s>>> {
        Ok(self.handle_optional(self.array_object(array)?.base.as_ref()))
    }

    fn set_array_writeable(&mut self, array: PyArray<'s>, writeable: bool) -> PyResult<'s, ()> {
        let array = self
            .get_mut(array.value())
            .map_err(PyError::runtime_error)?
            .native_mut::<ArrayObject>()
            .ok_or_else(|| PyError::runtime_error("array handle changed object kind"))?;
        array.view.writeable = writeable;
        Ok(())
    }

    fn read_arrays(
        &self,
        arrays: &[PyArray<'s>],
        read: &mut PyArrayReader<'_, 's>,
    ) -> PyResult<'s, ()> {
        let mut lent = Vec::with_capacity(arrays.len());
        for array in arrays {
            let ArrayObject { storage, view, .. } = self.array_object(*array)?;
            let data = match self.array_storage_object(self.handle(storage))? {
                ArrayStorage::Bytes(bytes) => PyArrayData::Bytes(bytes),
                ArrayStorage::Values(values) => PyArrayData::Values(values),
            };
            lent.push(PyArrayRef { view, data });
        }
        read(self, &lent)
    }

    fn write_array(
        &mut self,
        array: PyArray<'s>,
        write: &mut dyn FnMut(&Builder<'_>, PyArrayMut<'_>) -> PyResult<'s, ()>,
    ) -> PyResult<'s, ()> {
        let ArrayObject { storage, view, .. } = self.array_object(array)?;
        if !view.writeable {
            return Err(PyError::value_error("assignment destination is read-only"));
        }
        let storage = self.handle(storage);
        self.modify(storage, |builder, object| {
            let data = match object.native_mut::<ArrayStorage>() {
                Some(ArrayStorage::Bytes(bytes)) => PyArrayDataMut::Bytes(bytes),
                Some(ArrayStorage::Values(values)) => PyArrayDataMut::Values(values),
                None => return Err(PyError::runtime_error("array storage changed object kind")),
            };
            write(builder, PyArrayMut { data })
        })
        .map_err(PyError::runtime_error)?
    }

    fn apply_operator(
        &mut self,
        operator: PyOperator,
        operands: &[Value<'s>],
    ) -> PyResult<'s, Value<'s>> {
        let result = match (operator, operands) {
            (PyOperator::Binary(operator), [left, right]) => {
                self.binary_value(operator, *left, *right)
            }
            (PyOperator::Unary(operator), [operand]) => {
                self.push(*operand);
                self.unary(operator).and_then(|()| self.pop())
            }
            (PyOperator::Compare(operator), [left, right]) => {
                self.push(*left);
                self.push(*right);
                self.compare(operator).and_then(|()| self.pop())
            }
            (PyOperator::Absolute, [operand]) => self.absolute(*operand),
            _ => return Err(PyError::runtime_error("operator received the wrong arity")),
        };
        result.map_err(|message| {
            if self.pending_exception.is_some() {
                PyError::new(PyErrorKind::Raised, message)
            } else {
                PyError::type_error(message)
            }
        })
    }

    fn new_integer(&mut self, decimal: &str) -> PyResult<'s, Value<'s>> {
        self.charge_cpu(u64::try_from(decimal.len()).unwrap_or(u64::MAX))
            .map_err(PyError::resource_error)?;
        self.reserve_result(decimal.len().saturating_mul(2))
            .map_err(PyError::resource_error)?;
        let value = decimal
            .parse::<BigInt>()
            .map_err(|_| PyError::value_error("invalid integer"))?;
        self.new_bigint(value)
    }

    fn new_bigint(&mut self, value: BigInt) -> PyResult<'s, Value<'s>> {
        if let Some(value) = value.to_i64() {
            return Ok(Value::Int(value));
        }
        self.charge_cpu(super::super::number::words(&value))
            .map_err(PyError::resource_error)?;
        Vm::allocate_object(self, Object::BigInt(value)).map_err(PyError::resource_error)
    }

    fn new_regex(&mut self, pattern: String, flags: u32) -> PyResult<'s, Value<'s>> {
        Vm::allocate_object(
            self,
            Object::Native(Box::new(RegexObject { pattern, flags })),
        )
        .map_err(PyError::resource_error)
    }

    fn new_match(&mut self, data: PyMatchData<'s>) -> PyResult<'s, Value<'s>> {
        let PyMatchData {
            subject,
            regex,
            text,
            groups,
            group_names,
            spans,
            pos,
            endpos,
        } = data;
        self.alloc_with(|builder| {
            Object::Native(Box::new(MatchObject {
                subject: builder.store(subject),
                regex: builder.store(regex),
                text,
                groups,
                group_names,
                spans,
                pos,
                endpos,
            }))
        })
        .map_err(PyError::resource_error)
    }

    fn regex_parts(&mut self, regex: PyRegex<'s>) -> PyResult<'s, (String, u32)> {
        let RegexObject { pattern, flags } = self
            .state
            .heap
            .get(regex.value())
            .map_err(PyError::runtime_error)?
            .native::<RegexObject>()
            .ok_or_else(|| PyError::runtime_error("regex handle changed object kind"))?;
        let pattern = pattern.clone();
        let flags = *flags;
        self.reserve_memory(pattern.len())?;
        Ok((pattern, flags))
    }

    fn match_data(&mut self, matched: PyMatch<'s>) -> PyResult<'s, PyMatchData<'s>> {
        let match_object = self
            .state
            .heap
            .get(matched.value())
            .map_err(PyError::runtime_error)?
            .native::<MatchObject>()
            .ok_or_else(|| PyError::runtime_error("match handle changed object kind"))?;
        let bytes = match_object
            .groups
            .iter()
            .chain(&match_object.group_names)
            .try_fold(match_object.text.len(), |total, value| {
                total.checked_add(value.as_ref().map_or(0, String::len))
            })
            .and_then(|total| total.checked_add(match_object.spans.len().saturating_mul(16)));
        let bytes = bytes.ok_or_else(|| PyError::resource_error("match snapshot is too large"))?;
        let data = PyMatchData {
            subject: self.handle(&match_object.subject),
            regex: self.handle(&match_object.regex),
            text: match_object.text.clone(),
            groups: match_object.groups.clone(),
            group_names: match_object.group_names.clone(),
            spans: match_object.spans.clone(),
            pos: match_object.pos,
            endpos: match_object.endpos,
        };
        self.reserve_memory(bytes)?;
        Ok(data)
    }

    fn marker(&self, marker: PyMarker) -> Value<'s> {
        Value::Native(match marker {
            PyMarker::TypingList => NativeValue::TypingList,
            PyMarker::EnumType => NativeValue::BuiltinType(BuiltinType::Enum),
            PyMarker::TestCaseType => NativeValue::BuiltinType(BuiltinType::TestCase),
            PyMarker::Environment => NativeValue::Environment,
            PyMarker::Stdin => NativeValue::Stream(Stream::Stdin),
            PyMarker::StdinBuffer => NativeValue::Stream(Stream::StdinBuffer),
            PyMarker::Stdout => NativeValue::Stream(Stream::Stdout),
            PyMarker::Stderr => NativeValue::Stream(Stream::Stderr),
            PyMarker::ArrayType => NativeValue::BuiltinType(BuiltinType::Array),
        })
    }

    fn mark_dataclass(&mut self, class: PyClass<'s>) -> PyResult<'s, ()> {
        let Object::Class(class_object) = self
            .state
            .heap
            .get_mut(class.value())
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("class handle changed object kind"));
        };
        class_object.is_dataclass = true;
        Ok(())
    }

    fn argv0(&self) -> String {
        self.argv.first().cloned().unwrap_or_else(|| "-".into())
    }

    fn new_argv(&mut self) -> PyResult<'s, Value<'s>> {
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
    ) -> PyResult<'s, Value<'s>> {
        Vm::allocate_object(
            self,
            Object::Native(Box::new(ArgumentParserObject {
                prog: program,
                description,
                add_help,
                is_subcommand,
                arguments: Vec::new(),
                subparsers: None,
            })),
        )
        .map_err(PyError::resource_error)
    }

    fn argument_parser_parts(
        &mut self,
        parser: PyArgumentParser<'s>,
    ) -> PyResult<'s, PyArgumentParserData<'s>> {
        let ArgumentParserObject {
            prog,
            description,
            add_help,
            arguments,
            subparsers,
            ..
        } = self
            .get(parser.value())
            .map_err(PyError::runtime_error)?
            .native::<ArgumentParserObject>()
            .ok_or_else(|| PyError::runtime_error("parser handle changed object kind"))?;
        let bytes = prog
            .len()
            .saturating_add(description.as_ref().map_or(0, String::len))
            .saturating_add(arguments.len().saturating_mul(128))
            .saturating_add(
                subparsers
                    .as_ref()
                    .map_or(0, |value| value.commands.len().saturating_mul(96)),
            );
        let arguments = arguments
            .iter()
            .map(|argument| PyArgumentSpec {
                names: argument.names.clone(),
                dest: argument.dest.clone(),
                required: argument.required,
                default: self.handle(&argument.default),
                store_true: argument.store_true,
                store_false: argument.store_false,
                integer: argument.integer,
                choices: self.handles(&argument.choices),
                help: argument.help.clone(),
            })
            .collect();
        let subparsers = match subparsers {
            Some(subparsers) => {
                let mut commands = Vec::with_capacity(subparsers.commands.len());
                for command in &subparsers.commands {
                    commands.push(PySubcommandSpec {
                        name: command.name.clone(),
                        help: command.help.clone(),
                        parser: self.handle(&command.parser).cast(self)?,
                    });
                }
                Some(PySubparsersSpec {
                    dest: subparsers.dest.clone(),
                    required: subparsers.required,
                    help: subparsers.help.clone(),
                    commands,
                })
            }
            None => None,
        };
        let result = PyArgumentParserData {
            prog: prog.clone(),
            description: description.clone(),
            add_help: *add_help,
            arguments,
            subparsers,
        };
        self.reserve_memory(bytes)?;
        Ok(result)
    }

    fn append_argument(
        &mut self,
        parser: PyArgumentParser<'s>,
        argument: PyArgumentSpec<'s>,
    ) -> PyResult<'s, ()> {
        self.reserve_object_growth(parser.value(), 96)
            .map_err(PyError::resource_error)?;
        self.modify(parser.value(), |builder, object| {
            let Some(parser_object) = object.native_mut::<ArgumentParserObject>() else {
                return Err(PyError::runtime_error("parser handle changed object kind"));
            };
            let PyArgumentSpec {
                names,
                dest,
                required,
                default,
                store_true,
                store_false,
                integer,
                choices,
                help,
            } = argument;
            parser_object.arguments.push(ArgumentSpec {
                names,
                dest,
                required,
                default: builder.store(default),
                store_true,
                store_false,
                integer,
                choices: builder.refs(choices),
                help,
            });
            Ok(())
        })
        .map_err(PyError::runtime_error)?
    }

    fn configure_subparsers(
        &mut self,
        parser: PyArgumentParser<'s>,
        subparsers: PySubparsersSpec<'s>,
    ) -> PyResult<'s, ()> {
        self.modify(parser.value(), |builder, object| {
            let Some(parser_object) = object.native_mut::<ArgumentParserObject>() else {
                return Err(PyError::runtime_error("parser handle changed object kind"));
            };
            let ArgumentParserObject {
                is_subcommand,
                subparsers: current,
                ..
            } = parser_object;
            if *is_subcommand {
                return Err(PyError::value_error(
                    "nested argparse subparsers are not supported",
                ));
            }
            if current.is_some() {
                return Err(PyError::value_error("parser already has subparsers"));
            }
            let PySubparsersSpec {
                dest,
                required,
                help,
                commands,
            } = subparsers;
            *current = Some(SubparsersSpec {
                dest,
                required,
                help,
                commands: commands
                    .into_iter()
                    .map(|command| stored_subcommand(builder, command))
                    .collect(),
            });
            Ok(())
        })
        .map_err(PyError::runtime_error)?
    }

    fn append_subcommand(
        &mut self,
        parser: PyArgumentParser<'s>,
        command: PySubcommandSpec<'s>,
    ) -> PyResult<'s, ()> {
        self.reserve_object_growth(parser.value(), 96)
            .map_err(PyError::resource_error)?;
        self.modify(parser.value(), |builder, object| {
            let Some(parser_object) = object.native_mut::<ArgumentParserObject>() else {
                return Err(PyError::runtime_error("parser handle changed object kind"));
            };
            let ArgumentParserObject { subparsers, .. } = parser_object;
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
            subparsers
                .commands
                .push(stored_subcommand(builder, command));
            Ok(())
        })
        .map_err(PyError::runtime_error)?
    }

    fn command_arguments(&self) -> Vec<String> {
        self.argv.iter().skip(1).cloned().collect()
    }

    fn import_module(&mut self, name: &str) -> PyResult<'s, Value<'s>> {
        let stack_len = self.stack.len();
        self.import(name, false).map_err(PyError::runtime_error)?;
        let module = self
            .execution
            .stack
            .pop(&self.state.heap)
            .ok_or_else(|| PyError::runtime_error("module import produced no value"))?;
        debug_assert_eq!(self.stack.len(), stack_len);
        Ok(module)
    }

    fn new_module(
        &mut self,
        name: String,
        path: String,
        spec: Value<'s>,
        loader: Value<'s>,
    ) -> PyResult<'s, Value<'s>> {
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
            .alloc_scope_named(
                None,
                false,
                Arc::from([]),
                HashMap::from([
                    ("__name__".into(), module_name),
                    ("__file__".into(), module_path),
                    ("__package__".into(), package),
                    ("__spec__".into(), spec),
                    ("__loader__".into(), loader),
                ]),
            )
            .map_err(PyError::resource_error)?;
        self.alloc_with(|builder| Object::Module {
            name,
            scope: builder.store(scope),
        })
        .map_err(PyError::resource_error)
    }

    fn exec_module(&mut self, module: PyModule<'s>, path: &str) -> PyResult<'s, ()> {
        let source = self.interp.read_text(path)?;
        let code = self.compile_module_source(&source, path).map_err(|error| {
            // A stopped resource meter is the only resource failure the front end reports.
            if self.interp.resources.stop_reason().is_some() {
                PyError::resource_error(error)
            } else {
                PyError::runtime_error(error)
            }
        })?;
        let Object::Module { scope, .. } =
            self.get(module.value()).map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("module handle changed object kind"));
        };
        let scope = self.handle(scope);
        let import_root = path
            .rsplit_once('/')
            .map_or_else(|| "/".to_string(), |(parent, _)| parent.to_string());
        self.state.temporary_import_paths.insert(0, import_root);
        let execution = self.execute_code(&code, FrameEntry::scoped(scope));
        self.state.temporary_import_paths.remove(0);
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

    fn new_namespace(&mut self, values: Vec<(String, Value<'s>)>) -> PyResult<'s, Value<'s>> {
        self.alloc_with(|builder| {
            Object::Native(Box::new(NamespaceObject {
                values: values
                    .into_iter()
                    .map(|(name, value)| (name, builder.store(value)))
                    .collect(),
            }))
        })
        .map_err(PyError::resource_error)
    }

    fn new_raises_context(&mut self, expected: String) -> PyResult<'s, Value<'s>> {
        Vm::allocate_object(
            self,
            Object::Native(Box::new(RaisesContextObject { expected })),
        )
        .map_err(PyError::resource_error)
    }

    fn raises_expected(&self, context: PyRaisesContext<'s>) -> PyResult<'s, String> {
        let RaisesContextObject { expected } = self
            .state
            .heap
            .get(context.value())
            .map_err(PyError::runtime_error)?
            .native::<RaisesContextObject>()
            .ok_or_else(|| PyError::runtime_error("raises handle changed object kind"))?;
        Ok(expected.clone())
    }

    fn exception_type_name(&self, value: &Value<'s>) -> Option<&'static str> {
        match value.native_value() {
            Some(NativeValue::ExceptionType(ExceptionType(name))) => Some(name),
            _ => None,
        }
    }

    fn exception_type(&self, name: &'static str) -> Value<'s> {
        Value::Native(NativeValue::ExceptionType(ExceptionType(name)))
    }

    fn wait_on(&mut self, reasons: Vec<crate::scheduler::WaitReason>) -> PyResult<'s, ()> {
        if reasons.is_empty() {
            return Err(PyError::runtime_error(
                "cannot wait on an empty resource set",
            ));
        }
        if !self.native_suspend_allowed {
            return Err(PyError::runtime_error(
                "resource waits require bytecode scheduler dispatch",
            ));
        }
        let now = self.interp.clock.monotonic_ns();
        if reasons
            .iter()
            .any(|reason| wait_reason_ready(self.interp, reason, now))
        {
            return Ok(());
        }
        self.execution
            .async_timer_deadlines
            .retain(|deadline| *deadline > now);
        for reason in &reasons {
            let crate::scheduler::WaitReason::Timer(deadline) = reason else {
                continue;
            };
            if *deadline <= now {
                return Ok(());
            }
            if self.execution.async_timer_deadlines.insert(*deadline) {
                self.interp
                    .clock
                    .schedule_at(
                        *deadline,
                        crate::clock::EventKind::WakeTask {
                            task: u64::from(self.interp.process.pid),
                        },
                    )
                    .map_err(|error| PyError::resource_error(error.to_string()))?;
            }
        }
        if !self.mode.scheduler_owned {
            let deadline = reasons
                .iter()
                .filter_map(|reason| match reason {
                    crate::scheduler::WaitReason::Timer(deadline) => Some(*deadline),
                    _ => None,
                })
                .min();
            let target = self
                .interp
                .live_children
                .iter()
                .find(|(_, child)| child.owner == self.interp.process.pid)
                .map(|(pid, _)| *pid);
            if let Some(target) = target {
                crate::exec::drive_scheduler_step(self.interp, target, deadline)
                    .map_err(PyError::runtime_error)?;
            } else if let Some(deadline) = deadline {
                self.interp
                    .wait_for_time(deadline)
                    .map_err(PyError::resource_error)?;
            } else {
                return Err(PyError::runtime_error(
                    "asyncio resource wait has no live modeled child",
                ));
            }
            return Ok(());
        }
        self.pending_wait = Some(if reasons.len() == 1 {
            reasons.into_iter().next().expect("length was checked")
        } else {
            crate::scheduler::WaitReason::Any(reasons)
        });
        Ok(())
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

    fn http(&mut self) -> &mut dyn PyHttpClient {
        self
    }

    fn processes(&mut self) -> &mut dyn PyProcessRunner {
        self
    }

    fn caller_location(&self, depth: usize) -> Option<(String, u32)> {
        let index = self
            .bytecode_frames
            .len()
            .checked_sub(depth.checked_add(1)?)?;
        let frame = &self.bytecode_frames[index];
        // Call sites record the instruction pointer past the executing call.
        let span = frame
            .code
            .spans
            .get(frame.instruction_pointer.checked_sub(1)?)?;
        Some((self.traceback_filename(), u32::try_from(span.line).ok()?))
    }

    fn frame_module_name(&mut self, depth: usize) -> PyResult<'s, Option<Value<'s>>> {
        // Each frame resolves names through a scope that roots in its module; the main script's
        // frames root in the global namespace.
        let Some(frame) = self.bytecode_frames.iter().rev().nth(depth) else {
            return Ok(None);
        };
        let scope = self.handle_optional(frame.lookup_scope());
        self.module_name_of(scope).map_err(PyError::runtime_error)
    }

    fn function_flags(&self, value: &Value<'s>) -> PyResult<'s, Option<(bool, bool)>> {
        if !value.is_object() {
            return Ok(None);
        }
        let function = match self.get(*value).map_err(PyError::runtime_error)? {
            Object::Function { .. } => *value,
            Object::DescriptorBoundMethod { descriptor, .. } if descriptor.is_object() => {
                self.handle(descriptor)
            }
            // A suspended generator or coroutine reports the flags of the code it runs.
            Object::Generator(generator) => {
                let signature = &generator.code.call_signature;
                return Ok(Some((signature.is_generator, signature.is_coroutine)));
            }
            _ => return Ok(None),
        };
        let Object::Function(function_object) =
            self.get(function).map_err(PyError::runtime_error)?
        else {
            return Ok(None);
        };
        let signature = &function_object.code.call_signature;
        Ok(Some((signature.is_generator, signature.is_coroutine)))
    }

    fn function_parameters(&self, value: &Value<'s>) -> PyResult<'s, Option<Vec<PyParameter<'s>>>> {
        if !value.is_object() {
            return Ok(None);
        }
        let (function, bound) = match self.get(*value).map_err(PyError::runtime_error)? {
            Object::Function { .. } => (*value, false),
            Object::DescriptorBoundMethod { descriptor, .. } if descriptor.is_object() => {
                (self.handle(descriptor), true)
            }
            Object::DescriptorBoundMethod { .. } => return Ok(None),
            _ => return Ok(None),
        };
        let Object::Function(function_object) =
            self.get(function).map_err(PyError::runtime_error)?
        else {
            return Ok(None);
        };
        let FunctionObject { code, defaults, .. } = &**function_object;
        let mut parameters = code
            .parameters
            .iter()
            .map(|parameter| PyParameter {
                name: parameter.name.clone(),
                kind: parameter.kind,
                default: None,
            })
            .collect::<Vec<_>>();
        // Defaults fill local slots, which the compiler names after their parameters.
        for (&slot, default) in code.call_signature.default_slots.iter().zip(defaults) {
            let name = code.local_names.get(slot).map(String::as_str);
            if let Some(parameter) = parameters
                .iter_mut()
                .find(|parameter| Some(parameter.name.as_str()) == name)
            {
                parameter.default = Some(self.handle(default));
            }
        }
        if bound {
            // As `inspect.signature` does, a bound method drops its first positional parameter
            // and keeps a leading `*args`.
            match parameters.first().map(|parameter| parameter.kind) {
                Some(ParameterKind::PositionalOnly | ParameterKind::Positional) => {
                    parameters.remove(0);
                }
                Some(ParameterKind::Variadic) => {}
                _ => return Err(PyError::value_error("invalid method signature")),
            }
        }
        Ok(Some(parameters))
    }

    fn active_exception(&self) -> Option<Value<'s>> {
        self.exception_stack
            .last()
            .map(|exception| self.handle(&exception.value))
    }

    fn loaded_modules(&mut self) -> PyResult<'s, Value<'s>> {
        let mut names: Vec<String> = self.state.modules.keys().cloned().collect();
        names.sort();
        let mut items = Vec::with_capacity(names.len());
        for name in names {
            let module = self.handle(&self.state.modules[&name]);
            items.push((self.new_string(name)?, module));
        }
        self.new_dict(items)
    }

    fn register_module(&mut self, name: &str, module: Value<'s>) -> PyResult<'s, ()> {
        let is_module = matches!(module.native_value(), Some(NativeValue::Module(_)))
            || (module.is_object()
                && matches!(
                    self.get(module).map_err(PyError::runtime_error)?,
                    Object::Module { .. }
                ));
        if !is_module {
            return Err(PyError::type_error("sys.modules values must be modules"));
        }
        let stored = self.store(module);
        self.state.modules.insert(name.to_string(), stored);
        Ok(())
    }

    fn unregister_module(&mut self, name: &str) -> bool {
        self.state.modules.remove(name).is_some()
    }

    fn current_pid(&self) -> u32 {
        self.interp.process.pid
    }

    fn current_ppid(&self) -> u32 {
        self.interp.process.ppid
    }

    fn send_os_signal(&mut self, pid: u32, signal: crate::process::Signal) -> PyResult<'s, ()> {
        self.interp.send_signal(pid, signal).map_err(|error| {
            if error.contains("does not exist") {
                PyError::os_error(3, "No such process", None)
            } else {
                PyError::runtime_error(error)
            }
        })
    }
}
