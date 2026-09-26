//! Native-module runtime bridge backed by the metered Python VM.

use super::{
    protocol, Arc, BigInt, BinaryOperator, BuiltinType, CallArgs, CallMode, CallResult,
    ClassDefinition, ClassLayout, ExceptionType, Execution, HashMap, NativeValue, Object, Ordering,
    PyArgumentParser, PyArgumentParserData, PyArgumentSpec, PyArray, PyArrayDtype, PyArrayLayout,
    PyBinaryOp, PyByteArray, PyCallable, PyClass, PyClock, PyDict, PyEnvironment, PyError,
    PyErrorKind, PyFilesystem, PyHttpClient, PyIdentity, PyIterator, PyKind, PyList, PyMarker,
    PyMatch, PyMatchData, PyModule, PyNativeKind, PyProcessRunner, PyProperty, PyRaisesContext,
    PyRegex, PyResult, PyRuntime, PySet, PyStreamRead, PySubcommandSpec, PySubparsersSpec, PyTuple,
    PyValueCast, RaisedException, Stream, ToPrimitive, Value, ValueTag, Vm,
    MODELED_MAPPING_ENTRY_BYTES, MODELED_VALUE_BYTES,
};

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
    super::super::stdlib::numpy::validate_rank(&layout.shape)?;
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

fn wait_reason_ready(
    interp: &crate::interp::Interp,
    reason: &crate::scheduler::WaitReason,
    now: u64,
) -> bool {
    use crate::scheduler::WaitReason;

    match reason {
        WaitReason::ShellSession(_) => false,
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

impl Vm<'_> {
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
    ) -> PyResult<PyStreamRead> {
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
    ) -> PyResult<PyStreamRead> {
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
    ) -> PyResult<Option<PyStreamRead>> {
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
    ) -> PyResult<Option<PyStreamRead>> {
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
                Object::Complex { .. } => PyKind::Complex,
                Object::Tuple(_) => PyKind::Tuple,
                Object::Slice { .. } => PyKind::Native,
                Object::Dict(_) | Object::DefaultDict { .. } => PyKind::Dict,
                Object::Set(_) | Object::FrozenSet(_) => PyKind::Set,
                Object::Range { .. } => PyKind::Native,
                Object::Function { .. } | Object::DescriptorBoundMethod { .. } => PyKind::Function,
                Object::Class { .. } => PyKind::Class,
                Object::Instance { .. } | Object::EnumMember { .. } => PyKind::Instance,
                Object::Iterator { .. }
                | Object::SequenceIterator { .. }
                | Object::RangeIterator { .. }
                | Object::CountIterator { .. }
                | Object::StreamIterator { .. }
                | Object::CallableIterator { .. } => PyKind::Iterator,
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

    fn number(&self, value: &Value) -> Option<super::number::NumberRef<'_>> {
        super::number::view(&self.state.heap, value)
    }

    fn new_complex(&mut self, real: f64, imag: f64) -> PyResult<Value> {
        self.allocate_object(Object::Complex { real, imag })
            .map_err(PyError::resource_error)
    }

    fn integer_text(&self, value: &Value) -> PyResult<Option<String>> {
        Ok(match super::number::view(&self.state.heap, value) {
            Some(super::number::NumberRef::Int(value)) => Some(value.to_string()),
            Some(super::number::NumberRef::BigInt(value)) => Some(value.to_string()),
            Some(super::number::NumberRef::Float(_) | super::number::NumberRef::Complex(..))
            | None => None,
        })
    }

    fn write_stream(&mut self, stream: &Value, text: &str) -> PyResult<usize> {
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
        stream: &Value,
        size: Option<usize>,
        line: bool,
    ) -> PyResult<PyStreamRead> {
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

    fn truth(&mut self, value: &Value) -> PyResult<bool> {
        self.truth_value(value).map_err(PyError::runtime_error)
    }

    fn display(&mut self, value: &Value) -> PyResult<String> {
        self.display_value(value).map_err(PyError::runtime_error)
    }

    fn repr(&mut self, value: &Value) -> PyResult<String> {
        self.repr_value(value).map_err(PyError::runtime_error)
    }

    fn format_value(
        &mut self,
        value: &Value,
        conversion: Option<char>,
        specification: &str,
    ) -> PyResult<String> {
        self.render_formatted_value(value, conversion, specification)
            .map_err(PyError::value_error)
    }

    fn equals(&mut self, left: &Value, right: &Value) -> PyResult<bool> {
        protocol::equals(&self.state.heap, left, right).map_err(PyError::runtime_error)
    }

    fn compare(&mut self, left: &Value, right: &Value) -> PyResult<Ordering> {
        self.sort_order(left, right).map_err(|message| {
            if self.pending_exception.is_some() {
                PyError::new(PyErrorKind::Raised, message)
            } else {
                PyError::type_error(message)
            }
        })
    }

    fn get_attribute(&mut self, value: Value, name: &str) -> PyResult<Option<Value>> {
        self.resolve_attribute(value, name)
            .map_err(PyError::runtime_error)
    }

    fn list_len(&self, list: PyList) -> PyResult<usize> {
        match self
            .state
            .heap
            .get(list.object_id())
            .map_err(PyError::runtime_error)?
        {
            Object::List(items) => Ok(items.len()),
            _ => Err(PyError::runtime_error("list handle changed object kind")),
        }
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

    fn list_append(&mut self, list: PyList, value: Value) -> PyResult<()> {
        let id = list.object_id();
        if !matches!(self.state.heap.get(id), Ok(Object::List(_))) {
            return Err(PyError::runtime_error("list handle changed object kind"));
        }
        self.state
            .heap
            .reserve_object_growth(id, MODELED_VALUE_BYTES, &mut self.interp.resources)
            .map_err(PyError::resource_error)?;
        let Object::List(items) = self
            .state
            .heap
            .get_mut(id)
            .map_err(PyError::runtime_error)?
        else {
            unreachable!("list kind was checked before reserving growth")
        };
        items.push(value);
        Ok(())
    }

    fn list_insert(&mut self, list: PyList, index: usize, value: Value) -> PyResult<()> {
        let id = list.object_id();
        let length = self.list_len(list)?;
        let index = index.min(length);
        self.state
            .heap
            .reserve_object_growth(id, MODELED_VALUE_BYTES, &mut self.interp.resources)
            .map_err(PyError::resource_error)?;
        let Object::List(items) = self
            .state
            .heap
            .get_mut(id)
            .map_err(PyError::runtime_error)?
        else {
            unreachable!("list kind was checked before reserving growth")
        };
        items.insert(index, value);
        Ok(())
    }

    fn list_extend(&mut self, list: PyList, values: Vec<Value>) -> PyResult<()> {
        let id = list.object_id();
        self.list_len(list)?;
        let count = u64::try_from(values.len())
            .map_err(|_| PyError::resource_error("list growth overflow"))?;
        let bytes = count
            .checked_mul(MODELED_VALUE_BYTES)
            .ok_or_else(|| PyError::resource_error("list growth overflow"))?;
        self.state
            .heap
            .reserve_object_growth(id, bytes, &mut self.interp.resources)
            .map_err(PyError::resource_error)?;
        let Object::List(items) = self
            .state
            .heap
            .get_mut(id)
            .map_err(PyError::runtime_error)?
        else {
            unreachable!("list kind was checked before reserving growth")
        };
        items.extend(values);
        Ok(())
    }

    fn list_pop(&mut self, list: PyList, index: usize) -> PyResult<Value> {
        let id = list.object_id();
        let value = match self
            .state
            .heap
            .get_mut(id)
            .map_err(PyError::runtime_error)?
        {
            Object::List(items) if index < items.len() => items.remove(index),
            Object::List(_) => {
                return Err(PyError::exception("IndexError", "pop index out of range"))
            }
            _ => return Err(PyError::runtime_error("list handle changed object kind")),
        };
        self.state
            .heap
            .release_object_shrink(id, MODELED_VALUE_BYTES, &mut self.interp.resources)
            .map_err(PyError::runtime_error)?;
        Ok(value)
    }

    fn list_position(
        &mut self,
        list: PyList,
        needle: &Value,
        start: usize,
        stop: usize,
    ) -> PyResult<Option<usize>> {
        let id = list.object_id();
        let length = self.list_len(list)?;
        for position in start.min(length)..stop.min(length) {
            let candidate = match self.state.heap.get(id).map_err(PyError::runtime_error)? {
                Object::List(items) => items[position],
                _ => return Err(PyError::runtime_error("list handle changed object kind")),
            };
            self.charge_cpu(1).map_err(PyError::resource_error)?;
            if protocol::identical(&candidate, needle)
                || protocol::equals(&self.state.heap, &candidate, needle)
                    .map_err(PyError::runtime_error)?
            {
                return Ok(Some(position));
            }
        }
        Ok(None)
    }

    fn list_reverse(&mut self, list: PyList) -> PyResult<()> {
        let id = list.object_id();
        let length = self.list_len(list)?;
        self.charge_cpu(u64::try_from(length).unwrap_or(u64::MAX))
            .map_err(PyError::resource_error)?;
        let Object::List(items) = self
            .state
            .heap
            .get_mut(id)
            .map_err(PyError::runtime_error)?
        else {
            unreachable!("list kind was checked before reversal")
        };
        items.reverse();
        Ok(())
    }

    fn list_clear(&mut self, list: PyList) -> PyResult<()> {
        let id = list.object_id();
        let length = self.list_len(list)?;
        let Object::List(items) = self
            .state
            .heap
            .get_mut(id)
            .map_err(PyError::runtime_error)?
        else {
            unreachable!("list kind was checked before clearing")
        };
        items.clear();
        let bytes = u64::try_from(length)
            .unwrap_or(u64::MAX)
            .saturating_mul(MODELED_VALUE_BYTES);
        self.state
            .heap
            .release_object_shrink(id, bytes, &mut self.interp.resources)
            .map_err(PyError::runtime_error)
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
            Object::Dict(items) | Object::DefaultDict { entries: items, .. } => Ok(items.to_vec()),
            _ => Err(PyError::runtime_error("dict handle changed object kind")),
        }
    }

    fn dict_get(&mut self, dict: PyDict, key: &Value) -> PyResult<Option<Value>> {
        let id = dict.object_id();
        let Some(position) = self
            .find_mapping_entry(id, key)
            .map_err(PyError::runtime_error)?
        else {
            return Ok(None);
        };
        match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                Ok(Some(entries[position].1))
            }
            _ => Err(PyError::runtime_error("dict handle changed object kind")),
        }
    }

    fn dict_insert(&mut self, dict: PyDict, key: Value, value: Value) -> PyResult<()> {
        let id = dict.object_id();
        if let Some(position) = self
            .find_mapping_entry(id, &key)
            .map_err(PyError::runtime_error)?
        {
            let entries = match self
                .state
                .heap
                .get_mut(id)
                .map_err(PyError::runtime_error)?
            {
                Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
                _ => return Err(PyError::runtime_error("dict handle changed object kind")),
            };
            entries.set_value(position, value);
            return Ok(());
        }
        self.state
            .heap
            .reserve_object_growth(id, MODELED_MAPPING_ENTRY_BYTES, &mut self.interp.resources)
            .map_err(PyError::resource_error)?;
        let entries = match self
            .state
            .heap
            .get_mut(id)
            .map_err(PyError::runtime_error)?
        {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
            _ => unreachable!("dict kind was checked during lookup"),
        };
        entries.push((key, value));
        Ok(())
    }

    fn dict_remove(&mut self, dict: PyDict, key: &Value) -> PyResult<Option<Value>> {
        let id = dict.object_id();
        let Some(position) = self
            .find_mapping_entry(id, key)
            .map_err(PyError::runtime_error)?
        else {
            return Ok(None);
        };
        let value = match self
            .state
            .heap
            .get_mut(id)
            .map_err(PyError::runtime_error)?
        {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                entries.remove(position).1
            }
            _ => return Err(PyError::runtime_error("dict handle changed object kind")),
        };
        self.state
            .heap
            .release_object_shrink(id, MODELED_MAPPING_ENTRY_BYTES, &mut self.interp.resources)
            .map_err(PyError::runtime_error)?;
        Ok(Some(value))
    }

    fn replace_dict_items(&mut self, dict: PyDict, items: Vec<(Value, Value)>) -> PyResult<()> {
        let id = dict.object_id();
        let replacement = match self.state.heap.get(id).map_err(PyError::runtime_error)? {
            Object::Dict(_) => Object::Dict(items.into()),
            Object::DefaultDict { factory, .. } => Object::DefaultDict {
                factory: *factory,
                entries: items.into(),
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
            Object::Set(items) | Object::FrozenSet(items) => items,
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
            Object::Set(items) | Object::FrozenSet(items) => Ok(items.clone()),
            _ => Err(PyError::runtime_error("set handle changed object kind")),
        }
    }

    fn set_is_frozen(&self, set: PySet) -> PyResult<bool> {
        match self
            .state
            .heap
            .get(set.object_id())
            .map_err(PyError::runtime_error)?
        {
            Object::FrozenSet(_) => Ok(true),
            Object::Set(_) => Ok(false),
            _ => Err(PyError::runtime_error("set handle changed object kind")),
        }
    }

    fn set_insert(&mut self, set: PySet, value: Value) -> PyResult<bool> {
        let id = set.object_id();
        if self
            .find_set_entry(id, &value)
            .map_err(PyError::runtime_error)?
            .is_some()
        {
            return Ok(false);
        }
        self.state
            .heap
            .reserve_object_growth(id, MODELED_VALUE_BYTES, &mut self.interp.resources)
            .map_err(PyError::resource_error)?;
        let Object::Set(items) = self
            .state
            .heap
            .get_mut(id)
            .map_err(PyError::runtime_error)?
        else {
            unreachable!("set kind was checked during lookup")
        };
        items.push(value);
        Ok(true)
    }

    fn set_remove(&mut self, set: PySet, value: &Value) -> PyResult<bool> {
        let id = set.object_id();
        let Some(position) = self
            .find_set_entry(id, value)
            .map_err(PyError::runtime_error)?
        else {
            return Ok(false);
        };
        let Object::Set(items) = self
            .state
            .heap
            .get_mut(id)
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("set handle changed object kind"));
        };
        items.remove(position);
        self.state
            .heap
            .release_object_shrink(id, MODELED_VALUE_BYTES, &mut self.interp.resources)
            .map_err(PyError::runtime_error)?;
        Ok(true)
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
        let mut exception_base = None;
        for base in &bases {
            if let Some(id) = base.object_id() {
                let Object::Class {
                    layout: base_layout,
                    exception_base: base_exception,
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
                if let Some(base_exception) = base_exception {
                    if exception_base.replace(*base_exception).is_some() {
                        return Err(PyError::type_error(
                            "multiple exception bases are unsupported",
                        ));
                    }
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
                    Some(NativeValue::ExceptionType(ExceptionType(name)))
                        if layout == ClassLayout::Object && exception_base.is_none() =>
                    {
                        exception_base = Some(name);
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
            exception_base,
            attributes,
            dataclass_fields: Vec::new(),
            enum_members: Vec::new(),
        })
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
            .map(|(name, _)| Some(name.clone()))
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
                Object::Iterator { .. }
                    | Object::SequenceIterator { .. }
                    | Object::RangeIterator { .. }
                    | Object::CountIterator { .. }
                    | Object::CallableIterator { .. }
                    | Object::Generator { .. }
            ) {
                return Value::Object(id).cast(self);
            }
        }
        self.make_iterator(value)
            .map_err(PyError::type_error)?
            .cast(self)
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
            Object::SequenceIterator { .. } | Object::RangeIterator { .. } => self
                .next_stored_iterator(id)
                .map_err(PyError::runtime_error),
            Object::CountIterator { current, step } => {
                let value = current;
                let next = super::super::stdlib::itertools::count_next(current, step)
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
            Object::CallableIterator {
                callable,
                sentinel,
                exhausted,
            } => {
                if exhausted {
                    return Ok(None);
                }
                self.charge_cpu(1).map_err(PyError::resource_error)?;
                let value = self.call_value(callable, CallArgs::new(Vec::new(), Vec::new()))?;
                if protocol::equals(&self.state.heap, &value, &sentinel)
                    .map_err(PyError::runtime_error)?
                {
                    let Object::CallableIterator { exhausted, .. } = self
                        .state
                        .heap
                        .get_mut(id)
                        .map_err(PyError::runtime_error)?
                    else {
                        return Err(PyError::runtime_error("iterator changed object kind"));
                    };
                    *exhausted = true;
                    Ok(None)
                } else {
                    Ok(Some(value))
                }
            }
            Object::Generator { .. } => self.resume_generator(id).map_err(PyError::runtime_error),
            _ => Err(PyError::type_error("expected an iterator")),
        }
    }

    fn generator_send(&mut self, generator: PyIterator, value: Value) -> PyResult<Option<Value>> {
        let id = generator.object_id();
        if !matches!(
            self.state.heap.get(id).map_err(PyError::runtime_error)?,
            Object::Generator { .. }
        ) {
            return Err(PyError::type_error("expected a generator"));
        }
        self.resume_generator_with(id, value)
            .map_err(PyError::runtime_error)
    }

    fn generator_return_value(&self, generator: PyIterator) -> PyResult<Value> {
        match self
            .state
            .heap
            .get(generator.object_id())
            .map_err(PyError::runtime_error)?
        {
            Object::Generator {
                exhausted: true,
                return_value,
                ..
            } => Ok(*return_value),
            Object::Generator { .. } => Err(PyError::runtime_error("coroutine has not completed")),
            _ => Err(PyError::type_error("expected a coroutine")),
        }
    }

    fn coroutine_step(&mut self, coroutine: PyIterator, value: Value) -> PyResult<(u8, Value)> {
        let id = coroutine.object_id();
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
                Some(exception) => Ok((2, exception.value)),
                None => Err(PyError::runtime_error(error)),
            },
        }
    }

    fn generator_close(&mut self, generator: PyIterator) -> PyResult<()> {
        let id = generator.object_id();
        if !matches!(
            self.state.heap.get(id).map_err(PyError::runtime_error)?,
            Object::Generator { .. }
        ) {
            return Err(PyError::type_error("expected a generator"));
        }
        // A compact approximation of GeneratorExit: drive the bounded frame through its
        // cleanup path and discard values yielded while closing.
        while self
            .resume_generator(id)
            .map_err(PyError::runtime_error)?
            .is_some()
        {}
        Ok(())
    }

    fn generator_throw(&mut self, generator: PyIterator, exception: Value) -> PyResult {
        self.generator_close(generator)?;
        let raised = if let Some((kind, _)) =
            protocol::exception_parts(&self.state.heap, &exception)
                .map_err(PyError::runtime_error)?
        {
            RaisedException {
                kind,
                value: exception,
            }
        } else if let Some(NativeValue::ExceptionType(ExceptionType(kind))) =
            exception.native_value()
        {
            let value = self
                .allocate_exception(kind.into(), String::new())
                .map_err(PyError::resource_error)?;
            RaisedException {
                kind: kind.into(),
                value,
            }
        } else {
            return Err(PyError::type_error("generator.throw expects an exception"));
        };
        self.pending_exception = Some(raised);
        Err(PyError::new(
            PyErrorKind::Raised,
            "generator exception raised",
        ))
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
                entries: Default::default(),
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
        Vm::allocate_object(self, Object::Dict(items.into())).map_err(PyError::resource_error)
    }

    fn new_set(&mut self, items: Vec<Value>) -> PyResult<Value> {
        Vm::allocate_object(self, Object::Set(items)).map_err(PyError::resource_error)
    }

    fn new_frozen_set(&mut self, items: Vec<Value>) -> PyResult<Value> {
        Vm::allocate_object(self, Object::FrozenSet(items)).map_err(PyError::resource_error)
    }

    fn new_value_kind(
        &self,
        kind: &'static super::super::native::ValueKindDef,
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
        kind: &'static super::super::native::ValueKindDef,
    ) -> Option<u64> {
        let (index, payload) = value.registered_parts()?;
        std::ptr::eq(self.state.types.value_kind(index)?, kind).then_some(payload)
    }

    fn value_kind_type(
        &self,
        kind: &'static super::super::native::ValueKindDef,
    ) -> PyResult<Value> {
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
        let strides = super::super::stdlib::numpy::contiguous_strides(&shape)?;
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
        group_names: Vec<Option<String>>,
        start: usize,
        end: usize,
    ) -> PyResult<Value> {
        Vm::allocate_object(
            self,
            Object::Match {
                text,
                groups,
                group_names,
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
            groups,
            group_names,
            start,
            end,
            ..
        } = self
            .state
            .heap
            .get(matched.object_id())
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("match handle changed object kind"));
        };
        let bytes = groups
            .iter()
            .chain(group_names)
            .try_fold(0usize, |total, value| {
                total.checked_add(value.as_ref().map_or(0, String::len))
            });
        let bytes = bytes.ok_or_else(|| PyError::resource_error("match snapshot is too large"))?;
        let groups = groups.clone();
        let group_names = group_names.clone();
        let start = *start;
        let end = *end;
        self.reserve_memory(bytes)?;
        Ok(PyMatchData {
            groups,
            group_names,
            start,
            end,
        })
    }

    fn marker(&self, marker: PyMarker) -> Value {
        Value::Native(match marker {
            PyMarker::TypingList => NativeValue::TypingList,
            PyMarker::EnumBase => NativeValue::EnumBase,
            PyMarker::UnitTestBase => NativeValue::UnitTestBase,
            PyMarker::Environment => NativeValue::Environment,
            PyMarker::Stdin => NativeValue::Stream(Stream::Stdin),
            PyMarker::StdinBuffer => NativeValue::Stream(Stream::StdinBuffer),
            PyMarker::Stdout => NativeValue::Stream(Stream::Stdout),
            PyMarker::Stderr => NativeValue::Stream(Stream::Stderr),
            PyMarker::ArrayType => NativeValue::BuiltinType(BuiltinType::Array),
            PyMarker::ComplexType => NativeValue::BuiltinType(BuiltinType::Complex),
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

    fn import_module(&mut self, name: &str) -> PyResult<Value> {
        let stack_len = self.stack.len();
        self.import(name, false).map_err(PyError::runtime_error)?;
        let module = self
            .stack
            .pop()
            .ok_or_else(|| PyError::runtime_error("module import produced no value"))?;
        debug_assert_eq!(self.stack.len(), stack_len);
        Ok(module)
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
                Arc::from([]),
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
        let tokens = super::super::lexer::lex(&source).map_err(|error| {
            PyError::runtime_error(format!(
                "{} in {path} at line {}, column {}",
                error.message, error.span.line, error.span.column
            ))
        })?;
        let program = super::super::parser::parse(tokens).map_err(|error| {
            PyError::runtime_error(format!(
                "{} in {path} at line {}, column {}",
                error.message, error.span.line, error.span.column
            ))
        })?;
        let code = super::super::compiler::compile(program);
        let Object::Module { scope, .. } = self
            .state
            .heap
            .get(module.object_id())
            .map_err(PyError::runtime_error)?
        else {
            return Err(PyError::runtime_error("module handle changed object kind"));
        };
        let scope = *scope;
        let import_root = path
            .rsplit_once('/')
            .map_or_else(|| "/".to_string(), |(parent, _)| parent.to_string());
        self.state.temporary_import_paths.insert(0, import_root);
        self.local_scopes.push(scope);
        let execution = self.execute_code(&code);
        self.local_scopes.pop();
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

    fn wait_on(&mut self, reasons: Vec<crate::scheduler::WaitReason>) -> PyResult<()> {
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
        self.interp
    }

    fn processes(&mut self) -> &mut dyn PyProcessRunner {
        self
    }

    fn current_pid(&self) -> u32 {
        self.interp.process.pid
    }

    fn current_ppid(&self) -> u32 {
        self.interp.process.ppid
    }

    fn send_os_signal(&mut self, pid: u32, signal: crate::process::Signal) -> PyResult<()> {
        self.interp.send_signal(pid, signal).map_err(|error| {
            if error.contains("does not exist") {
                PyError::exception("ProcessLookupError", format!("[Errno 3] {error}"))
            } else {
                PyError::runtime_error(error)
            }
        })
    }
}
