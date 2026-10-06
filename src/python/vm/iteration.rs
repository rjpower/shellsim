//! Iteration, generator suspension, and sequence unpacking.
//!
//! A suspended generator keeps its operands, open `try` regions, entered contexts and handled
//! exceptions as stored references inside its heap object. Resuming it pushes its frame on the
//! shared VM stacks above the generator object, as a call would, and moves that saved state on
//! top; a `yield` moves it back. A `for` loop or `yield from` in Python code resumes a generator
//! on the same dispatch loop, so no Rust frame sits between the generator and its consumer;
//! natives, `next` and `send` run the frame with [`Vm::run_frames_above`].

use super::super::heap::Ref;
use super::{
    exception_types, number, string, CallArgs, Consumer, Flow, ForIterOutcome, FrameEntry,
    FrameKind, IteratorAdvance, NativeValue, Object, Opcode, PyError, PyRuntime, PyStreamRead,
    Slot, SlotValue, Stream, Value, Vm,
};
use crate::python::error::{Control, PyResult};

/// How a suspended generator resumes: with the value of its `yield` expression, with an
/// exception raised at that `yield`, or, after a thrown exception made the subiterator of its
/// `yield from` return, with that `yield from` finished at instruction `finished`.
#[derive(Clone, Copy)]
enum GeneratorResume {
    Send(Value),
    Throw(Value),
    SendFinished { value: Value, finished: usize },
}

/// What [`Vm::delegated_throw`] leaves to do after offering a thrown exception to the
/// subiterator of a `yield from`.
enum Delegated {
    /// Resume the delegating generator this way.
    Resume(GeneratorResume),
    /// The subiterator yielded this value; the delegating generator stays suspended.
    Yielded(Value),
}

/// What became of an exception thrown into a generator suspended in `yield from`, after
/// [`Vm::forward_throw`] passed it to the subiterator.
enum ForwardedThrow {
    /// The subiterator yielded this value; the delegating generator stays suspended.
    Yielded(Value),
    /// The subiterator returned this value, which becomes the value of `yield from`.
    Returned(Value),
    /// Raise this exception at the delegating generator's `yield from`.
    Raise(Value),
}

/// An iterator classified by [`Vm::advance_iterator`] whose advance needs a second pass over
/// `self`: reading a stored reference after the position update, a subscript, or a stream read.
enum SlowPathAdvance {
    /// A materialized iterator's item at this index.
    Materialized(usize),
    /// A sequence iterator's owner at this position.
    Sequence(usize),
    /// A reverse iterator's owner at this index.
    Reverse(usize),
    Callable,
    Stream(bool),
}

impl<'s> Vm<'s> {
    /// `reversed(x)` consults the type's reverse slot before the length and indexed sequence
    /// protocol. It does not accept an arbitrary iterable as a sequence.
    pub(super) fn reverse_value(&mut self, value: Value) -> PyResult<Value> {
        if let Some(iterator) =
            self.invoke_slot(&value, Slot::Reversed, "__reversed__", Vec::new())?
        {
            return Ok(iterator);
        }
        let subject = value;
        let physical_sequence = if subject.is_object() {
            matches!(
                self.get(subject)?,
                Object::List(_)
                    | Object::Tuple(_)
                    | Object::Range { .. }
                    | Object::String(_)
                    | Object::Bytes(_)
                    | Object::ByteArray(_)
            )
        } else {
            string::string_value(&self.state.heap, subject)?.is_some()
        };
        let getitem = self
            .state
            .types
            .slot(self.type_id(&value)?, Slot::GetItem)?
            .is_some();
        if !physical_sequence && !getitem {
            let name = self.type_name_of(&value)?;
            return Err(PyError::exception(
                "TypeError",
                format!("'{name}' object is not reversible"),
            ));
        }
        let length = match self.invoke_slot(&value, Slot::Length, "__len__", Vec::new())? {
            Some(length) => {
                let length = number::int_value(&self.state.heap, length).ok_or_else(|| {
                    PyError::exception("TypeError", "__len__() should return an integer")
                })?;
                usize::try_from(length)
                    .map_err(|_| PyError::exception("ValueError", "__len__() should return >= 0"))?
            }
            None => self
                .physical_length(value)?
                .ok_or_else(|| PyError::exception("TypeError", "reversed() requires a sequence"))?,
        };
        self.reverse_items(value, length)
    }

    /// The native list, tuple and range reverse slots use physical payload indexing, so an
    /// inherited slot does not re-enter a subclass's `__getitem__` override.
    pub(super) fn reverse_builtin_sequence(&mut self, value: Value) -> PyResult<Value> {
        let subject = value;
        let length = self
            .physical_length(subject)?
            .ok_or("native reverse slot requires a sequence")?;
        self.reverse_items(subject, length)
    }

    fn reverse_items(&mut self, value: Value, length: usize) -> PyResult<Value> {
        self.alloc(Object::ReverseIterator {
            owner: Ref::from(value),
            next: length,
        })
    }

    pub(super) fn get_iterator(&mut self) -> PyResult<()> {
        let iterable = self.pop()?;
        let iterator = self.make_iterator(iterable)?;
        self.push(iterator);
        Ok(())
    }

    pub(super) fn make_iterator(&mut self, iterable: Value) -> PyResult<Value> {
        if let Some(NativeValue::Stream(stream)) = iterable.native_value() {
            if matches!(stream, Stream::Stdin | Stream::StdinBuffer) {
                // `for line in sys.stdin` needs a heap-object iterator (not the generic
                // `__iter__`/`__next__` dispatch materialized below): only a heap-object iterator
                // can suspend a `for` loop, and this one can block on fd 0. See
                // `advance_iterator`'s `Object::StreamIterator` arm.
                return self.alloc(Object::StreamIterator {
                    binary: matches!(stream, Stream::StdinBuffer),
                });
            }
        }
        if self.instance_class(iterable)?.is_none() {
            if let Some(iterator) = self.payload_iterator(iterable)? {
                return Ok(iterator);
            }
        }
        if let Some(iterator) = self.class_iterator(&iterable)? {
            return Ok(iterator);
        }
        let values = self.iterable_values(&iterable)?;
        self.alloc(Object::Iterator {
            values: Ref::all(values),
            position: 0,
        })
    }

    /// An iterator over a builtin payload, ignoring any `__iter__` the object's class defines:
    /// what the builtin `__iter__` slots return, so `str.__iter__(instance)` does not dispatch
    /// back into a subclass's override. Returns `None` when the payload is not a builtin
    /// iterable.
    pub(super) fn payload_iterator(&mut self, iterable: Value) -> PyResult<Option<Value>> {
        if iterable.is_object() {
            match self.get(iterable)? {
                Object::List(_) | Object::Tuple(_) => {
                    return self
                        .alloc(Object::SequenceIterator {
                            owner: Ref::from(iterable),
                            position: 0,
                        })
                        .map(Some);
                }
                Object::Range { start, stop, step } => {
                    let (current, stop, step) = (*start, *stop, *step);
                    return self
                        .alloc(Object::RangeIterator {
                            current,
                            stop,
                            step,
                            exhausted: false,
                        })
                        .map(Some);
                }
                Object::Iterator { .. }
                | Object::SequenceIterator { .. }
                | Object::ReverseIterator { .. }
                | Object::RangeIterator { .. }
                | Object::CountIterator { .. }
                | Object::StreamIterator { .. }
                | Object::CallableIterator { .. }
                | Object::Generator { .. } => {
                    // These iterators remain lazy; materializing either one here would permit an
                    // unbounded host allocation before the caller's loop can meter each item.
                    return Ok(Some(iterable));
                }
                _ => {}
            }
        }
        self.snapshot_builtin_iterator(iterable)
    }

    /// Builtin strings, byte strings and hash containers keep their existing snapshot iteration
    /// behavior, but enter it before their new `__iter__` slots to avoid redispatching into the
    /// native wrapper. Copying one element at a time charges its retained size before the next.
    fn snapshot_builtin_iterator(&mut self, iterable: Value) -> PyResult<Option<Value>> {
        let mut values = Vec::new();
        if let Some(text) = string::string_value(&self.state.heap, iterable)? {
            for character in text.chars() {
                let character = self.allocate_string(character.to_string())?;
                self.push_materialized(&mut values, character)?;
            }
        } else if let Some(bytes) = string::bytes_value(&self.state.heap, iterable)? {
            for byte in bytes {
                self.push_materialized(&mut values, Value::Int(i64::from(byte)))?;
            }
        } else if iterable.is_object() {
            let members = match self.get(iterable)? {
                Object::Set(items) | Object::FrozenSet(items) => self.values(items.iter()),
                Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                    self.values(entries.iter().map(|(key, _)| key))
                }
                _ => return Ok(None),
            };
            for item in members {
                self.push_materialized(&mut values, item)?;
            }
        } else {
            return Ok(None);
        }
        self.alloc(Object::Iterator {
            values: Ref::all(values),
            position: 0,
        })
        .map(Some)
    }

    /// Whether `value` is an iterator: one of the runtime's lazy iterator objects, or an object
    /// whose class defines `__next__`.
    pub(super) fn is_iterator(&self, value: &Value) -> PyResult<bool> {
        if value.is_object()
            && matches!(
                self.get(*value)?,
                Object::Iterator { .. }
                    | Object::SequenceIterator { .. }
                    | Object::ReverseIterator { .. }
                    | Object::RangeIterator { .. }
                    | Object::CountIterator { .. }
                    | Object::StreamIterator { .. }
                    | Object::CallableIterator { .. }
                    | Object::Generator { .. }
            )
        {
            return Ok(true);
        }
        Ok(matches!(
            self.state.types.slot(self.type_id(value)?, Slot::Next)?,
            Some(slot) if !matches!(&slot, SlotValue::Descriptor { value, .. } if value.is_none())
        ))
    }

    /// The iterator that the `__iter__` of `iterable`'s class returns, or `None` when the class
    /// defines no `__iter__`. The iterator is returned unadvanced, so a loop over it calls
    /// `__next__` once per item, as CPython does.
    ///
    /// A class that sets `__iter__ = None` declares its instances not iterable, even when it
    /// defines `__getitem__`, and an `__iter__` that returns something other than an iterator
    /// raises `TypeError`.
    pub(super) fn class_iterator(&mut self, iterable: &Value) -> PyResult<Option<Value>> {
        match self.state.types.slot(self.type_id(iterable)?, Slot::Iter)? {
            None => return Ok(None),
            Some(SlotValue::Descriptor {
                value: descriptor, ..
            }) if descriptor.is_none() => {
                return Err(self.raise_object_type_error(iterable, "is not iterable"));
            }
            Some(_) => {}
        }
        let Some(iterator) = self.invoke_slot(iterable, Slot::Iter, "__iter__", Vec::new())? else {
            return Ok(None);
        };
        if !self.is_iterator(&iterator)? {
            let type_name = self.type_name_of(&iterator)?;
            return Err(PyError::exception(
                "TypeError",
                format!("iter() returned non-iterator of type '{type_name}'"),
            ));
        }
        Ok(Some(iterator))
    }

    /// The next item of `iterator`, or `None` once it is exhausted. A `StopIteration` raised by
    /// a class's `__next__` ends the iteration rather than propagating.
    pub(super) fn next_until_stop(&mut self, iterator: &Value) -> PyResult<Option<Value>> {
        match self.iterator_next(iterator) {
            Ok(item) => Ok(item),
            Err(error) => self.catch(error, "StopIteration").map(|()| None),
        }
    }

    /// Classify and, where possible, advance one iterator with a single heap lookup.
    fn advance_iterator(&mut self, iterator: Value) -> PyResult<IteratorAdvance> {
        let slow_path = match self.get_mut(iterator)? {
            Object::Iterator { values, position } => {
                if *position >= values.len() {
                    return Ok(IteratorAdvance::Exhausted);
                }
                *position += 1;
                SlowPathAdvance::Materialized(*position - 1)
            }
            Object::RangeIterator {
                current,
                stop,
                step,
                exhausted,
            } => {
                if *exhausted
                    || (*step > 0 && *current >= *stop)
                    || (*step < 0 && *current <= *stop)
                {
                    *exhausted = true;
                    return Ok(IteratorAdvance::Exhausted);
                }
                let value = *current;
                if let Some(next) = current.checked_add(*step) {
                    *current = next;
                } else {
                    *exhausted = true;
                }
                return Ok(IteratorAdvance::Yield(Value::Int(value)));
            }
            Object::CountIterator { current, step } => {
                let value = *current;
                *current = super::super::stdlib::itertools::count_next(value, *step)
                    .map_err(str::to_string)?;
                return Ok(IteratorAdvance::Yield(Value::Int(value)));
            }
            Object::SequenceIterator { position, .. } => SlowPathAdvance::Sequence(*position),
            Object::ReverseIterator { next, .. } => {
                if *next == 0 {
                    return Ok(IteratorAdvance::Exhausted);
                }
                *next -= 1;
                SlowPathAdvance::Reverse(*next)
            }
            Object::CallableIterator { exhausted, .. } => {
                if *exhausted {
                    return Ok(IteratorAdvance::Exhausted);
                }
                SlowPathAdvance::Callable
            }
            Object::Generator { .. } => return Ok(IteratorAdvance::Generator),
            Object::StreamIterator { binary } => SlowPathAdvance::Stream(*binary),
            _ => return Ok(IteratorAdvance::Protocol),
        };
        match slow_path {
            SlowPathAdvance::Materialized(index) => {
                let Object::Iterator { values, .. } = self.get(iterator)? else {
                    unreachable!("iterator kind was checked above")
                };
                let value = self.value(&values[index]);
                Ok(IteratorAdvance::Yield(value))
            }
            SlowPathAdvance::Sequence(position) => {
                let Object::SequenceIterator { owner, .. } = self.get(iterator)? else {
                    unreachable!("iterator kind was checked above")
                };
                let owner = self.value(owner);
                let value = match self.get(owner)? {
                    Object::List(values) | Object::Tuple(values) => {
                        self.value_optional(values.get(position))
                    }
                    _ => return Err("iterator source changed object kind".into()),
                };
                let Some(value) = value else {
                    return Ok(IteratorAdvance::Exhausted);
                };
                let Object::SequenceIterator { position, .. } = self.get_mut(iterator)? else {
                    unreachable!("iterator kind was checked above")
                };
                *position += 1;
                Ok(IteratorAdvance::Yield(value))
            }
            SlowPathAdvance::Reverse(index) => {
                let Object::ReverseIterator { owner, .. } = self.get(iterator)? else {
                    unreachable!("iterator kind was checked above")
                };
                let owner = self.value(owner);
                self.charge_cpu(1)?;
                let index = i64::try_from(index)
                    .map_err(|_| PyError::exception("OverflowError", "sequence is too large"))?;
                self.subscript_value(owner, Value::Int(index))
                    .map(IteratorAdvance::Yield)
            }
            SlowPathAdvance::Callable => {
                let Object::CallableIterator {
                    callable, sentinel, ..
                } = self.get(iterator)?
                else {
                    unreachable!("iterator kind was checked above")
                };
                Ok(IteratorAdvance::Callable {
                    callable: self.value(callable),
                    sentinel: self.value(sentinel),
                })
            }
            SlowPathAdvance::Stream(binary) => self.advance_stream_iterator(binary),
        }
    }

    /// Read one line from `sys.stdin`/`sys.stdin.buffer` for a `for` loop, suspending on the same
    /// `WaitReason` a direct `readline()` call would produce.
    fn advance_stream_iterator(&mut self, binary: bool) -> PyResult<IteratorAdvance> {
        let stream = if binary {
            Stream::StdinBuffer
        } else {
            Stream::Stdin
        };
        let marker = Value::Native(NativeValue::Stream(stream));
        match self
            .read_stream(&marker, None, true)
            .map_err(PyError::into_control)
        {
            Ok(read) if read.is_empty() => Ok(IteratorAdvance::Exhausted),
            Ok(PyStreamRead::Text(text)) => Ok(IteratorAdvance::Yield(self.allocate_string(text)?)),
            Ok(PyStreamRead::Bytes(bytes)) => {
                let value = self.new_bytes(bytes)?;
                Ok(IteratorAdvance::Yield(value))
            }
            Err(Ok(Control::Exit(status))) => Err(PyError::exit(status)),
            Err(Ok(Control::Suspend(reason))) => Ok(IteratorAdvance::Blocked(reason)),
            Err(Err(error)) => Err(error),
        }
    }

    pub(super) fn next_stored_iterator(&mut self, iterator: Value) -> PyResult<Option<Value>> {
        match self.advance_iterator(iterator)? {
            IteratorAdvance::Yield(value) => Ok(Some(value)),
            IteratorAdvance::Exhausted => Ok(None),
            IteratorAdvance::Callable { .. }
            | IteratorAdvance::Generator
            | IteratorAdvance::Protocol => Err("object is not a stored iterator".into()),
            // `next(sys.stdin)` and unpacking outside a `for` loop cannot suspend the way the
            // `ForIterator` opcode can; this is a narrower surface than CPython's `next()`.
            IteratorAdvance::Blocked(_) => {
                Err("reading standard input would block outside a for loop".into())
            }
        }
    }

    fn exhaust_callable_iterator(&mut self, iterator: Value) -> PyResult<()> {
        let Object::CallableIterator { exhausted, .. } = self.get_mut(iterator)? else {
            return Err("iterator changed object kind".into());
        };
        *exhausted = true;
        Ok(())
    }

    pub(super) fn unpack_sequence(
        &mut self,
        expected: usize,
        star_index: Option<usize>,
    ) -> PyResult<()> {
        let value = self.pop()?;
        let Some(star_index) = star_index else {
            return self.unpack_exactly(value, expected);
        };
        let values = self.iterable_values(&value)?;
        let mut outputs = Vec::new();
        if star_index >= expected || values.len() < expected.saturating_sub(1) {
            return Err(PyError::exception(
                "ValueError",
                format!(
                    "not enough values to unpack (expected at least {}, got {})",
                    expected.saturating_sub(1),
                    values.len()
                ),
            ));
        }
        let tail_start = star_index;
        let tail_end = values.len() - (expected - star_index - 1);
        for index in 0..expected {
            if index == star_index {
                let mut tail = Vec::new();
                for value in values[tail_start..tail_end].iter().copied() {
                    self.push_materialized(&mut tail, value)?;
                }
                outputs.push(self.alloc(Object::List(Ref::all(tail)))?);
            } else {
                let source_index = if index < star_index {
                    index
                } else {
                    tail_end + (index - star_index - 1)
                };
                self.push_materialized(&mut outputs, values[source_index])?;
            }
        }
        // Store operations pop their input, so the leftmost target must be on top.
        for output in outputs.into_iter().rev() {
            self.push(output);
        }
        Ok(())
    }

    /// Unpack `value` into exactly `expected` targets. As in CPython, only an exact list, tuple
    /// or dict reports how many items it held when there are too many; any other iterable is
    /// read one item past `expected`, so unpacking an infinite iterator still fails.
    fn unpack_exactly(&mut self, value: Value, expected: usize) -> PyResult<()> {
        let known_length = if value.is_object() {
            match self.get(value)? {
                Object::List(values) | Object::Tuple(values) => Some(values.len()),
                Object::Dict(entries) => Some(entries.len()),
                _ => None,
            }
        } else {
            None
        };
        let values = match known_length {
            Some(length) if length != expected => {
                let problem = if length > expected {
                    "too many"
                } else {
                    "not enough"
                };
                return Err(PyError::exception(
                    "ValueError",
                    format!("{problem} values to unpack (expected {expected}, got {length})"),
                ));
            }
            Some(_) => self.iterable_values(&value)?,
            None => {
                let iterator = self.make_iterator(value)?;
                let mut values = Vec::new();
                while values.len() <= expected {
                    let Some(item) = self.next_until_stop(&iterator)? else {
                        break;
                    };
                    self.push_materialized(&mut values, item)?;
                }
                if values.len() != expected {
                    let message = if values.len() < expected {
                        format!(
                            "not enough values to unpack (expected {expected}, got {})",
                            values.len()
                        )
                    } else {
                        format!("too many values to unpack (expected {expected})")
                    };
                    return Err(PyError::exception("ValueError", message));
                }
                values
            }
        };
        // Store operations pop their input, so the leftmost target must be on top.
        for value in values.into_iter().rev() {
            self.push(value);
        }
        Ok(())
    }

    /// Advance the iterator kept at the top of the operand stack. The iterator remains below the
    /// yielded value until exhaustion, which gives `for` a small and explicit stack contract.
    ///
    /// A generator is resumed on the dispatch loop: its frame becomes active with the loop's
    /// frame, resuming after the `ForIterator` at `op_index`, as consumer
    /// ([`ForIterOutcome::Entered`]).
    ///
    /// A [`ForIterOutcome::Blocked`] result leaves the iterator on the stack untouched: retrying
    /// the same `ForIterator` opcode next quantum re-advances the same iterator object, which is
    /// safe because [`Vm::advance_stream_iterator`] only mutates position after a successful read.
    pub(super) fn for_iterator(&mut self, op_index: usize) -> PyResult<ForIterOutcome> {
        if self.frame_stack_len() == 0 {
            return Err("invalid bytecode stack effect".into());
        }
        let iterator = self
            .execution
            .stack
            .peek(&self.state.heap, 0)
            .ok_or("invalid bytecode stack effect")?;
        if !iterator.is_object() {
            return Err("for-loop stack does not contain an iterator".into());
        }
        match self.advance_iterator(iterator)? {
            IteratorAdvance::Yield(value) => {
                self.push(value);
                Ok(ForIterOutcome::Yielded)
            }
            IteratorAdvance::Exhausted => {
                self.execution.stack.pop_ref();
                Ok(ForIterOutcome::Exhausted)
            }
            IteratorAdvance::Callable { callable, sentinel } => {
                self.charge_cpu(1)?;
                let value = <Self as PyRuntime>::call_value(
                    self,
                    callable,
                    CallArgs::new(Vec::new(), Vec::new()),
                )?;
                if self.values_equal(&value, &sentinel)? {
                    self.exhaust_callable_iterator(iterator)?;
                    self.execution.stack.pop_ref();
                    return Ok(ForIterOutcome::Exhausted);
                }
                self.push(value);
                Ok(ForIterOutcome::Yielded)
            }
            IteratorAdvance::Generator => {
                self.active_frame_mut().instruction_pointer = op_index + 1;
                if self.enter_generator_frame(iterator, Value::None, Consumer::Frame)? {
                    return Ok(ForIterOutcome::Entered);
                }
                self.execution.stack.pop_ref();
                Ok(ForIterOutcome::Exhausted)
            }
            IteratorAdvance::Protocol => match self.next_until_stop(&iterator)? {
                Some(value) => {
                    self.push(value);
                    Ok(ForIterOutcome::Yielded)
                }
                None => {
                    self.execution.stack.pop_ref();
                    Ok(ForIterOutcome::Exhausted)
                }
            },
            IteratorAdvance::Blocked(reason) => Ok(ForIterOutcome::Blocked(reason)),
        }
    }

    /// One step of `yield from`, with the subiterator and the sent value on top of the stack.
    ///
    /// A generator subiterator receives the value through `send`. Any other iterator advances as
    /// `next` would when the value is `None`, and otherwise receives it through its own `send`
    /// method; CPython's builtin iterators have none, so sending them a value raises
    /// `AttributeError`. [`ForIterOutcome::Yielded`] leaves the subiterator and its value on the
    /// stack; [`ForIterOutcome::Exhausted`] replaces the subiterator with its return value, which
    /// is `None` unless a generator returned one or a `send` method raised `StopIteration` with
    /// one; [`ForIterOutcome::Blocked`] restores the stack for a retry. A generator subiterator
    /// resumes on the dispatch loop with the delegating frame, resuming after the
    /// `YieldFromSend` at `op_index`, as consumer ([`ForIterOutcome::Entered`]).
    pub(super) fn yield_from_send(&mut self, op_index: usize) -> PyResult<ForIterOutcome> {
        let sent = self.pop()?;
        let subiterator = self
            .execution
            .stack
            .peek(&self.state.heap, 0)
            .ok_or("invalid bytecode stack effect")?;
        if let Some(generator) = self.suspendable_generator(&subiterator)? {
            self.active_frame_mut().instruction_pointer = op_index + 1;
            if self.enter_generator_frame(generator, sent, Consumer::Frame)? {
                return Ok(ForIterOutcome::Entered);
            }
            self.execution.stack.pop_ref();
            self.push(Value::None);
            return Ok(ForIterOutcome::Exhausted);
        }
        if !sent.is_none() {
            let Some(send) = self.resolve_attribute(subiterator, "send")? else {
                let type_name = self.type_name_of(&subiterator)?;
                return Err(PyError::exception(
                    "AttributeError",
                    format!("'{type_name}' object has no attribute 'send'"),
                ));
            };
            return Ok(match self.call_subiterator_method(send, sent)? {
                ForwardedThrow::Yielded(value) => {
                    self.push(value);
                    ForIterOutcome::Yielded
                }
                ForwardedThrow::Returned(value) => {
                    self.execution.stack.pop_ref();
                    self.push(value);
                    ForIterOutcome::Exhausted
                }
                ForwardedThrow::Raise(exception) => return Err(self.raise_value(exception)),
            });
        }
        let outcome = self.for_iterator(op_index)?;
        match outcome {
            ForIterOutcome::Yielded | ForIterOutcome::Entered => {}
            ForIterOutcome::Exhausted => self.push(Value::None),
            ForIterOutcome::Blocked(_) => self.push(sent),
        }
        Ok(outcome)
    }

    /// The generator `value` refers to, unless it is some other kind of iterator.
    fn suspendable_generator(&self, value: &Value) -> PyResult<Option<Value>> {
        if !value.is_object() {
            return Ok(None);
        }
        Ok(matches!(self.get(*value)?, Object::Generator { .. }).then_some(*value))
    }

    /// Take the value a generator returned, leaving `None` behind so that it is reported once.
    fn take_return_value(&mut self, generator: Value) -> PyResult<Value> {
        let returned = self.modify(generator, |object| match object {
            Object::Generator(state) => Ok(std::mem::replace(
                &mut state.return_value,
                Ref::from(Value::None),
            )),
            _ => Err(String::from("object is not a generator")),
        })??;
        Ok(self.value(&returned))
    }

    /// Close a generator as `generator.close()` does: raise `GeneratorExit` at its suspension
    /// point and let it run its cleanup. A generator that yields again instead raises
    /// `RuntimeError`; any other exception it raises stays pending.
    pub(super) fn close_generator(&mut self, generator: Value) -> PyResult<()> {
        let exit = self.allocate_exception("GeneratorExit", String::new())?;
        match self.resume_generator_frame(generator, GeneratorResume::Throw(exit)) {
            Ok(Some(_)) => Err(PyError::exception(
                "RuntimeError",
                "generator ignored GeneratorExit",
            )),
            Ok(None) => Ok(()),
            Err(error) => self
                .catch(error, "GeneratorExit")
                .or_else(|error| self.catch(error, "StopIteration")),
        }
    }

    /// Pass an exception thrown into a generator suspended in `yield from` to its subiterator.
    ///
    /// `GeneratorExit` closes the subiterator, through `close` for a subiterator that is not a
    /// generator, and is then raised in the delegating generator. Any other exception goes to
    /// the subiterator's `throw`; the result says whether it yielded, returned, or raised. A
    /// subiterator without `close` or `throw`, such as a builtin iterator, is skipped, so the
    /// exception is raised in the delegating generator.
    fn forward_throw(&mut self, subiterator: &Value, exception: Value) -> PyResult<ForwardedThrow> {
        let generator = self.suspendable_generator(subiterator)?;
        if self.exception_is(exception, "GeneratorExit")? {
            let closed = match generator {
                Some(generator) => self.close_generator(generator),
                None => match self.resolve_attribute(*subiterator, "close")? {
                    Some(close) => <Self as PyRuntime>::call_value(
                        self,
                        close,
                        CallArgs::new(Vec::new(), Vec::new()),
                    )
                    .map(|_| ()),
                    None => Ok(()),
                },
            };
            return match closed {
                Ok(()) => Ok(ForwardedThrow::Raise(exception)),
                Err(error) => self.take_exception(error).map(ForwardedThrow::Raise),
            };
        }
        if let Some(generator) = generator {
            return match self.resume_generator_frame(generator, GeneratorResume::Throw(exception)) {
                Ok(Some(value)) => Ok(ForwardedThrow::Yielded(value)),
                Ok(None) => Ok(ForwardedThrow::Returned(self.take_return_value(generator)?)),
                Err(error) => self.take_exception(error).map(ForwardedThrow::Raise),
            };
        }
        match self.resolve_attribute(*subiterator, "throw")? {
            Some(throw) => self.call_subiterator_method(throw, exception),
            None => Ok(ForwardedThrow::Raise(exception)),
        }
    }

    /// Call a subiterator's `send` or `throw` method for `yield from`. Its result is the next
    /// value to yield, and a `StopIteration` it raises ends the delegation with the exception's
    /// value.
    fn call_subiterator_method(
        &mut self,
        method: Value,
        argument: Value,
    ) -> PyResult<ForwardedThrow> {
        let result = <Self as PyRuntime>::call_value(
            self,
            method,
            CallArgs::new(vec![argument], Vec::new()),
        );
        let exception = match result {
            Ok(value) => return Ok(ForwardedThrow::Yielded(value)),
            Err(error) => self.take_exception(error)?,
        };
        if !self.exception_is(exception, "StopIteration")? {
            return Ok(ForwardedThrow::Raise(exception));
        }
        let value = exception_types::exception_args(self.state, exception)?
            .and_then(|(_, args)| args.first().copied())
            .unwrap_or(Value::None);
        Ok(ForwardedThrow::Returned(value))
    }

    /// Resume `generator` for Rust code until its next yield, returning the value, or until it
    /// returns (`None`).
    pub(super) fn resume_generator(&mut self, generator: Value) -> PyResult<Option<Value>> {
        self.resume_generator_with(generator, Value::None)
    }

    pub(super) fn resume_generator_with(
        &mut self,
        generator: Value,
        sent: Value,
    ) -> PyResult<Option<Value>> {
        self.resume_generator_frame(generator, GeneratorResume::Send(sent))
    }

    /// Raise the `StopIteration` that ends iteration over `iterator`. A generator that has just
    /// returned passes its return value as the exception's `value`, once; later calls, and
    /// other iterators, raise it without arguments.
    pub(super) fn raise_stop_iteration(&mut self, iterator: &Value) -> PyError {
        let returned = if self
            .suspendable_generator(iterator)
            .ok()
            .flatten()
            .is_some()
        {
            self.take_return_value(*iterator).unwrap_or(Value::None)
        } else {
            Value::None
        };
        let args = if returned.is_none() {
            Vec::new()
        } else {
            vec![returned]
        };
        self.raise_exception_args("StopIteration", args)
    }

    fn set_generator_running(&mut self, generator: Value, value: bool) -> PyResult<()> {
        if let Object::Generator(state) = self.get_mut(generator)? {
            state.running = value;
        }
        Ok(())
    }

    /// Raise `exception` at the generator's suspended `yield`, as `generator.throw` does.
    ///
    /// The frame's innermost active handler receives it, so `except`, `finally` and `with`
    /// blocks run as they would for an exception raised there. The result is the next yielded
    /// value, or `None` once the generator returns. An exception the generator does not handle,
    /// including one thrown before it starts or after it finishes, stays pending and closes it.
    pub(super) fn throw_into_generator(
        &mut self,
        generator: Value,
        exception: Value,
    ) -> PyResult<Option<Value>> {
        self.resume_generator_frame(generator, GeneratorResume::Throw(exception))
    }

    /// Run one generator step for Rust code in a child pin scope, so the pins the step
    /// creates are released when it suspends; the yielded value crosses back as a stored
    /// reference.
    fn resume_generator_frame(
        &mut self,
        generator: Value,
        resume: GeneratorResume,
    ) -> PyResult<Option<Value>> {
        let yielded = {
            let mut vm = self.scope();
            vm.push(generator);
            let step = vm.run_generator_step(generator, resume);
            vm.execution.stack.pop_ref();
            step?
        };
        Ok(yielded.map(|value| self.value(&value)))
    }

    /// One generator step for Rust code, with `generator` on top of the operand stack: enter
    /// its frame and run it until it yields, returns or raises.
    fn run_generator_step(
        &mut self,
        generator: Value,
        resume: GeneratorResume,
    ) -> PyResult<Option<Ref>> {
        let resume = match resume {
            GeneratorResume::Throw(exception) => {
                match self.delegated_throw(generator, exception)? {
                    Delegated::Resume(resume) => resume,
                    Delegated::Yielded(value) => return Ok(Some(self.store(value))),
                }
            }
            resume => resume,
        };
        let depth = self.bytecode_frames.len();
        let sent = match resume {
            GeneratorResume::Send(sent) | GeneratorResume::SendFinished { value: sent, .. } => sent,
            GeneratorResume::Throw(_) => Value::None,
        };
        if !self.enter_generator_frame(generator, sent, Consumer::Rust)? {
            return match resume {
                // A finished generator raises a thrown exception at once.
                GeneratorResume::Throw(exception) => Err(self.raise_value(exception)),
                _ => Ok(None),
            };
        }
        let result = match resume {
            GeneratorResume::Send(_) => self.run_frames_above(depth),
            GeneratorResume::SendFinished { finished, .. } => {
                // The subiterator returned the sent value: `yield from` finishes with it.
                self.execution.stack.remove(1);
                self.active_frame_mut().instruction_pointer = finished;
                self.run_frames_above(depth)
            }
            GeneratorResume::Throw(exception) => {
                let error = self.raise_value(exception);
                let span = self.suspension_span();
                self.propagate_error(error, span)
                    .and_then(|_| self.run_frames_above(depth))
            }
        };
        match result {
            Ok(Flow::Yield(value)) => Ok(Some(value)),
            Ok(Flow::Return(_)) => Ok(None),
            Ok(Flow::Exit(status)) => Err(PyError::exit(status)),
            Ok(flow) => unreachable!("a generator step cannot end with {flow:?}"),
            Err((error, _)) => Err(error),
        }
    }

    /// The span of the `yield` the active generator frame is suspended after.
    fn suspension_span(&mut self) -> super::super::source::Span {
        let frame = self.active_frame_mut();
        frame
            .instruction_pointer
            .checked_sub(1)
            .and_then(|site| frame.code.spans.get(site).copied())
            .unwrap_or_default()
    }

    /// For a generator suspended in `yield from`, pass a thrown exception to the subiterator
    /// first; the result says how the generator itself resumes, or what the subiterator yielded.
    fn delegated_throw(&mut self, generator: Value, exception: Value) -> PyResult<Delegated> {
        let Object::Generator(state) = self.get(generator)? else {
            return Err("object is not a generator".into());
        };
        // A frame suspended in `yield from` sits at its `YieldFromSend`; a plain `yield`
        // resumes after its `Yield`, which is never followed by a `YieldFromSend`.
        let finished = match state.code.instructions.get(state.instruction_pointer) {
            Some(instruction) if !state.exhausted && !state.running => match instruction.opcode {
                Opcode::YieldFromSend(finished) => finished,
                _ => return Ok(Delegated::Resume(GeneratorResume::Throw(exception))),
            },
            _ => return Ok(Delegated::Resume(GeneratorResume::Throw(exception))),
        };
        let subiterator = self
            .value_optional(state.stack.last())
            .ok_or("yield from lost its subiterator")?;
        self.set_generator_running(generator, true)?;
        let forwarded = self.forward_throw(&subiterator, exception);
        self.set_generator_running(generator, false)?;
        Ok(match forwarded? {
            ForwardedThrow::Yielded(value) => Delegated::Yielded(value),
            ForwardedThrow::Returned(value) => {
                Delegated::Resume(GeneratorResume::SendFinished { value, finished })
            }
            ForwardedThrow::Raise(exception) => {
                Delegated::Resume(GeneratorResume::Throw(exception))
            }
        })
    }

    /// Push the frame of `generator`, which sits on top of the operand stack, so that it
    /// resumes for `consumer` with `sent` as the value of the `yield` it stopped at. Its saved
    /// operands, `try` regions, contexts and handled exceptions move onto the shared stacks
    /// above the generator. `false` when the generator has finished; resuming it reports the
    /// end once more and clears its return value.
    pub(super) fn enter_generator_frame(
        &mut self,
        generator: Value,
        sent: Value,
        consumer: Consumer,
    ) -> PyResult<bool> {
        if self.call_depth >= Self::MAX_CALL_DEPTH {
            return Err(PyError::exception(
                "RecursionError",
                "maximum recursion depth exceeded",
            ));
        }
        let Object::Generator(state) = self.get(generator)? else {
            return Err("object is not a generator".into());
        };
        if state.running {
            return Err(PyError::value_error("generator already executing"));
        }
        if state.exhausted {
            self.take_return_value(generator)?;
            return Ok(false);
        }
        let instruction_pointer = state.instruction_pointer;
        if instruction_pointer == 0 && !sent.is_none() {
            return Err(PyError::type_error(
                "can't send non-None value to a just-started generator",
            ));
        }
        let code = state.code.clone();
        let entry = FrameEntry::function(self.value(&state.function), self.value(&state.scope));
        let stack_base = self.execution.stack.len();
        let frame = self.enter_frame(
            &code,
            instruction_pointer,
            stack_base,
            entry,
            FrameKind::Generator(consumer),
        )?;
        let Object::Generator(state) = self.state.heap.get_mut(generator)? else {
            unreachable!("generator kind was checked above")
        };
        state.running = true;
        let exception_base = frame.exception_base as usize;
        self.execution
            .stack
            .extend_owned(std::mem::take(&mut state.stack));
        self.execution.exception_stack.append(&mut state.exceptions);
        self.execution.handlers.extend(state.handlers.drain(..).map(
            |(target, depth, exception_depth)| {
                (target, depth + stack_base, exception_depth + exception_base)
            },
        ));
        self.execution.with_contexts.append(&mut state.contexts);
        if instruction_pointer != 0 {
            self.push(sent);
        }
        self.call_depth += 1;
        self.bytecode_frames.push(frame);
        Ok(true)
    }

    /// Pop the active generator frame at a `yield`, saving its position, operands, open `try`
    /// regions (relative to its bases), entered contexts and handled exceptions into its
    /// generator, which is left on top of the operand stack. Allocates nothing.
    pub(super) fn suspend_generator_frame(&mut self) -> PyResult<()> {
        let frame = self
            .bytecode_frames
            .pop()
            .expect("a generator frame is active");
        self.call_depth = self.call_depth.saturating_sub(1);
        let stack_base = frame.stack_base();
        let exception_base = frame.exception_base as usize;
        let operands = self
            .execution
            .stack
            .drain_refs(stack_base)
            .collect::<Vec<_>>();
        let exceptions = self.execution.exception_stack.split_off(exception_base);
        let handlers = self
            .execution
            .handlers
            .split_off(frame.handler_base as usize)
            .into_iter()
            .map(|(target, depth, exception_depth)| {
                (
                    target,
                    depth.saturating_sub(stack_base),
                    exception_depth.saturating_sub(exception_base),
                )
            })
            .collect();
        let contexts = self
            .execution
            .with_contexts
            .split_off(frame.context_base as usize);
        let generator = self.peek(0)?;
        self.modify(generator, |object| {
            if let Object::Generator(state) = object {
                state.instruction_pointer = frame.instruction_pointer;
                state.stack = operands;
                state.exceptions = exceptions;
                state.handlers = handlers;
                state.contexts = contexts;
                state.running = false;
            }
        })
    }

    /// Pop the active generator frame for good, after it returned or raised: what it owned on
    /// the shared stacks goes, and its generator, left on top of the operand stack, is finished.
    pub(super) fn close_generator_frame(&mut self) -> PyResult<()> {
        self.pop_frame();
        let generator = self.peek(0)?;
        if let Object::Generator(state) = self.get_mut(generator)? {
            state.exhausted = true;
            state.running = false;
        }
        Ok(())
    }

    /// Record `value` as what a generator that Rust code resumed returned, for the
    /// `StopIteration` that reports it. The generator is on top of the operand stack.
    pub(super) fn set_generator_return(&mut self, value: &Ref) -> PyResult<()> {
        let generator = self.peek(0)?;
        let value = value.dup();
        self.modify(generator, |object| {
            if let Object::Generator(state) = object {
                state.return_value = value;
            }
        })
    }
}
