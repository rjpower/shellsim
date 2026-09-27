//! Iteration, generator suspension, and sequence unpacking.

use super::{
    protocol, CallArgs, Execution, ForIterOutcome, IteratorAdvance, NativeValue, Object, Opcode,
    PyError, PyErrorKind, PyRuntime, PyStreamRead, Slot, SlotValue, Stream, Value, Vm,
};

/// How a suspended generator resumes: with the value of its `yield` expression, or with an
/// exception raised at that `yield`.
enum GeneratorResume {
    Send(Value),
    Throw(super::RaisedException),
}

/// What became of an exception thrown into a generator suspended in `yield from`, after
/// [`Vm::forward_throw`] passed it to the subiterator.
enum ForwardedThrow {
    /// The subiterator yielded this value; the delegating generator stays suspended.
    Yielded(Value),
    /// The subiterator returned this value, which becomes the value of `yield from`.
    Returned(Value),
    /// Raise this exception at the delegating generator's `yield from`.
    Raise(super::RaisedException),
}

/// An iterator classified by [`Vm::advance_iterator`] whose advance needs a second, unborrowed
/// pass over `self` (a heap lookup for a sequence, or a descriptor read for a stream).
enum SlowPathAdvance {
    Sequence(super::super::heap::ObjectId, usize),
    Stream(bool),
}

impl Vm<'_> {
    pub(super) fn get_iterator(&mut self) -> Result<(), String> {
        let iterable = self.pop()?;
        let iterator = self.make_iterator(iterable)?;
        self.stack.push(iterator);
        Ok(())
    }

    pub(super) fn make_iterator(&mut self, iterable: Value) -> Result<Value, String> {
        if let Some(NativeValue::Stream(stream)) = iterable.native_value() {
            if matches!(stream, Stream::Stdin | Stream::StdinBuffer) {
                // `for line in sys.stdin` needs a heap-object iterator (not the generic
                // `__iter__`/`__next__` dispatch materialized below): only a heap-object iterator
                // can suspend a `for` loop, and this one can block on fd 0. See
                // `advance_iterator`'s `Object::StreamIterator` arm.
                return self.allocate_object(Object::StreamIterator {
                    binary: matches!(stream, Stream::StdinBuffer),
                });
            }
        }
        if let Some(id) = iterable.object_id() {
            match self.state.heap.get(id)? {
                Object::List(_) | Object::Tuple(_) => {
                    return self.allocate_object(Object::SequenceIterator {
                        owner: id,
                        position: 0,
                    });
                }
                Object::Range { start, stop, step } => {
                    let (current, stop, step) = (*start, *stop, *step);
                    return self.allocate_object(Object::RangeIterator {
                        current,
                        stop,
                        step,
                        exhausted: false,
                    });
                }
                Object::Iterator { .. }
                | Object::SequenceIterator { .. }
                | Object::RangeIterator { .. }
                | Object::CountIterator { .. }
                | Object::StreamIterator { .. }
                | Object::CallableIterator { .. }
                | Object::Generator { .. } => {
                    // These iterators remain lazy; materializing either one here would permit an
                    // unbounded host allocation before the caller's loop can meter each item.
                    return Ok(Value::Object(id));
                }
                _ => {}
            }
        }
        if let Some(iterator) = self.class_iterator(&iterable)? {
            return Ok(iterator);
        }
        let values = self.iterable_values(&iterable)?;
        let iterator = self.state.heap.allocate(
            Object::Iterator {
                values,
                position: 0,
            },
            &mut self.interp.resources,
        )?;
        Ok(iterator)
    }

    /// Whether `value` is an iterator: one of the runtime's lazy iterator objects, or an object
    /// whose class defines `__next__`.
    pub(super) fn is_iterator(&self, value: &Value) -> Result<bool, String> {
        if let Some(id) = value.object_id() {
            if matches!(
                self.state.heap.get(id)?,
                Object::Iterator { .. }
                    | Object::SequenceIterator { .. }
                    | Object::RangeIterator { .. }
                    | Object::CountIterator { .. }
                    | Object::StreamIterator { .. }
                    | Object::CallableIterator { .. }
                    | Object::Generator { .. }
            ) {
                return Ok(true);
            }
        }
        Ok(matches!(
            self.state.types.slot(self.type_id(value)?, Slot::Next)?,
            Some(slot) if !matches!(slot, SlotValue::Descriptor(Value::None))
        ))
    }

    /// The iterator that the `__iter__` of `iterable`'s class returns, or `None` when the class
    /// defines no `__iter__`. The iterator is returned unadvanced, so a loop over it calls
    /// `__next__` once per item, as CPython does.
    ///
    /// A class that sets `__iter__ = None` declares its instances not iterable, even when it
    /// defines `__getitem__`, and an `__iter__` that returns something other than an iterator
    /// raises `TypeError`.
    pub(super) fn class_iterator(&mut self, iterable: &Value) -> Result<Option<Value>, String> {
        match self.state.types.slot(self.type_id(iterable)?, Slot::Iter)? {
            None => return Ok(None),
            Some(SlotValue::Descriptor(Value::None)) => {
                return Err(self.raise_object_type_error(iterable, "is not iterable"));
            }
            Some(_) => {}
        }
        let Some(iterator) = self.invoke_slot(iterable, Slot::Iter, "__iter__", Vec::new())? else {
            return Ok(None);
        };
        if !self.is_iterator(&iterator)? {
            let type_name = self.type_name_of(&iterator)?;
            return Err(self.raise_exception(
                "TypeError",
                format!("iter() returned non-iterator of type '{type_name}'"),
            ));
        }
        Ok(Some(iterator))
    }

    /// The next item of `iterator`, or `None` once it is exhausted. A `StopIteration` raised by
    /// a class's `__next__` ends the iteration rather than propagating.
    pub(super) fn next_until_stop(&mut self, iterator: &Value) -> Result<Option<Value>, String> {
        match self.iterator_next(iterator) {
            Err(_) if self.pending_stop_iteration() => {
                self.pending_exception = None;
                Ok(None)
            }
            result => result,
        }
    }

    /// Classify and, where possible, advance one iterator with a single arena lookup.
    fn advance_iterator(
        &mut self,
        iterator: super::super::heap::ObjectId,
    ) -> Result<IteratorAdvance, String> {
        let sequence = match self.state.heap.get_mut(iterator)? {
            Object::Iterator { values, position } => {
                let value = values.get(*position).copied();
                if value.is_some() {
                    *position += 1;
                }
                return Ok(value
                    .map(IteratorAdvance::Yield)
                    .unwrap_or(IteratorAdvance::Exhausted));
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
            Object::SequenceIterator { owner, position } => {
                SlowPathAdvance::Sequence(*owner, *position)
            }
            Object::CallableIterator {
                callable,
                sentinel,
                exhausted,
            } => {
                return Ok(if *exhausted {
                    IteratorAdvance::Exhausted
                } else {
                    IteratorAdvance::Callable {
                        callable: *callable,
                        sentinel: *sentinel,
                    }
                });
            }
            Object::Generator { .. } => return Ok(IteratorAdvance::Generator),
            Object::StreamIterator { binary } => SlowPathAdvance::Stream(*binary),
            _ => return Ok(IteratorAdvance::Protocol),
        };
        match sequence {
            SlowPathAdvance::Sequence(owner, position) => {
                let value = match self.state.heap.get(owner)? {
                    Object::List(values) | Object::Tuple(values) => values.get(position).copied(),
                    _ => return Err("iterator source changed object kind".into()),
                };
                let Some(value) = value else {
                    return Ok(IteratorAdvance::Exhausted);
                };
                let Object::SequenceIterator { position, .. } =
                    self.state.heap.get_mut(iterator)?
                else {
                    unreachable!("iterator kind was checked above")
                };
                *position += 1;
                Ok(IteratorAdvance::Yield(value))
            }
            SlowPathAdvance::Stream(binary) => self.advance_stream_iterator(binary),
        }
    }

    /// Read one line from `sys.stdin`/`sys.stdin.buffer` for a `for` loop, suspending on the same
    /// `WaitReason` a direct `readline()` call would produce.
    fn advance_stream_iterator(&mut self, binary: bool) -> Result<IteratorAdvance, String> {
        let stream = if binary {
            Stream::StdinBuffer
        } else {
            Stream::Stdin
        };
        let marker = Value::Native(NativeValue::Stream(stream));
        match self.read_stream(&marker, None, true) {
            Ok(read) if read.is_empty() => Ok(IteratorAdvance::Exhausted),
            Ok(PyStreamRead::Text(text)) => Ok(IteratorAdvance::Yield(self.allocate_string(text)?)),
            Ok(PyStreamRead::Bytes(bytes)) => {
                let value = self
                    .new_bytes(bytes)
                    .map_err(|error| self.record_native_error(error))?;
                Ok(IteratorAdvance::Yield(value))
            }
            Err(PyError {
                kind: PyErrorKind::Suspend(reason),
                ..
            }) => Ok(IteratorAdvance::Blocked(reason)),
            Err(error) => Err(self.record_native_error(error)),
        }
    }

    pub(super) fn next_stored_iterator(
        &mut self,
        iterator: super::super::heap::ObjectId,
    ) -> Result<Option<Value>, String> {
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

    fn exhaust_callable_iterator(
        &mut self,
        iterator: super::super::heap::ObjectId,
    ) -> Result<(), String> {
        let Object::CallableIterator { exhausted, .. } = self.state.heap.get_mut(iterator)? else {
            return Err("iterator changed object kind".into());
        };
        *exhausted = true;
        Ok(())
    }

    pub(super) fn unpack_sequence(
        &mut self,
        expected: usize,
        star_index: Option<usize>,
    ) -> Result<(), String> {
        let value = self.pop()?;
        let values = self.iterable_values(&value)?;
        let mut outputs = Vec::new();
        if let Some(star_index) = star_index {
            if star_index >= expected || values.len() < expected.saturating_sub(1) {
                return Err(self.raise_exception(
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
                    for value in values[tail_start..tail_end].iter().cloned() {
                        self.push_materialized(&mut tail, value)?;
                    }
                    outputs.push(self.allocate_object(Object::List(tail))?);
                } else {
                    let source_index = if index < star_index {
                        index
                    } else {
                        tail_end + (index - star_index - 1)
                    };
                    self.push_materialized(&mut outputs, values[source_index])?;
                }
            }
        } else {
            if values.len() != expected {
                let problem = if values.len() > expected {
                    "too many"
                } else {
                    "not enough"
                };
                return Err(self.raise_exception(
                    "ValueError",
                    format!(
                        "{problem} values to unpack (expected {expected}, got {})",
                        values.len()
                    ),
                ));
            }
            for value in values {
                self.push_materialized(&mut outputs, value)?;
            }
        }
        // Store operations pop their input, so the leftmost target must be on top.
        self.stack.extend(outputs.into_iter().rev());
        Ok(())
    }

    /// Advance the iterator kept at the top of the operand stack. The iterator remains below the
    /// yielded value until exhaustion, which gives `for` a small and explicit stack contract.
    ///
    /// A [`ForIterOutcome::Blocked`] result leaves the iterator on the stack untouched: retrying
    /// the same `ForIterator` opcode next quantum re-advances the same iterator object, which is
    /// safe because [`Vm::advance_stream_iterator`] only mutates position after a successful read.
    pub(super) fn for_iterator(&mut self) -> Result<ForIterOutcome, String> {
        if self.frame_stack_len() == 0 {
            return Err("invalid bytecode stack effect".into());
        }
        let Some(id) = self
            .stack
            .last()
            .cloned()
            .ok_or("invalid bytecode stack effect")?
            .object_id()
        else {
            return Err("for-loop stack does not contain an iterator".into());
        };
        match self.advance_iterator(id)? {
            IteratorAdvance::Yield(value) => {
                self.stack.push(value);
                Ok(ForIterOutcome::Yielded)
            }
            IteratorAdvance::Exhausted => {
                self.stack.pop();
                Ok(ForIterOutcome::Exhausted)
            }
            IteratorAdvance::Callable { callable, sentinel } => {
                self.charge_cpu(1)?;
                let value = <Self as PyRuntime>::call_value(
                    self,
                    callable,
                    CallArgs::new(Vec::new(), Vec::new()),
                )
                .map_err(|error| error.to_string())?;
                if self.values_equal(&value, &sentinel)? {
                    self.exhaust_callable_iterator(id)?;
                    self.stack.pop();
                    return Ok(ForIterOutcome::Exhausted);
                }
                self.stack.push(value);
                Ok(ForIterOutcome::Yielded)
            }
            IteratorAdvance::Generator => match self.resume_generator(id)? {
                Some(value) => {
                    self.stack.push(value);
                    Ok(ForIterOutcome::Yielded)
                }
                None => {
                    self.stack.pop();
                    Ok(ForIterOutcome::Exhausted)
                }
            },
            IteratorAdvance::Protocol => match self.next_until_stop(&Value::Object(id))? {
                Some(value) => {
                    self.stack.push(value);
                    Ok(ForIterOutcome::Yielded)
                }
                None => {
                    self.stack.pop();
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
    /// one; [`ForIterOutcome::Blocked`] restores the stack for a retry.
    pub(super) fn yield_from_send(&mut self) -> Result<ForIterOutcome, String> {
        let sent = self.pop()?;
        let subiterator = *self.stack.last().ok_or("invalid bytecode stack effect")?;
        if let Some(id) = self.suspendable_generator(&subiterator)? {
            let outcome = match self.resume_generator_with(id, sent)? {
                Some(value) => {
                    self.stack.push(value);
                    ForIterOutcome::Yielded
                }
                None => {
                    self.stack.pop();
                    let returned = self.take_return_value(id)?;
                    self.stack.push(returned);
                    ForIterOutcome::Exhausted
                }
            };
            return Ok(outcome);
        }
        if !sent.is_none() {
            let Some(send) = self.resolve_attribute(subiterator, "send")? else {
                let type_name = self.type_name_of(&subiterator)?;
                return Err(self.raise_exception(
                    "AttributeError",
                    format!("'{type_name}' object has no attribute 'send'"),
                ));
            };
            return Ok(match self.call_subiterator_method(send, sent)? {
                ForwardedThrow::Yielded(value) => {
                    self.stack.push(value);
                    ForIterOutcome::Yielded
                }
                ForwardedThrow::Returned(value) => {
                    self.stack.pop();
                    self.stack.push(value);
                    ForIterOutcome::Exhausted
                }
                ForwardedThrow::Raise(exception) => {
                    let message = format!("{} raised by send()", exception.kind);
                    self.pending_exception = Some(exception);
                    return Err(message);
                }
            });
        }
        let outcome = self.for_iterator()?;
        match outcome {
            ForIterOutcome::Yielded => {}
            ForIterOutcome::Exhausted => self.stack.push(Value::None),
            ForIterOutcome::Blocked(_) => self.stack.push(sent),
        }
        Ok(outcome)
    }

    /// The generator `value` refers to, unless it is some other kind of iterator.
    fn suspendable_generator(
        &self,
        value: &Value,
    ) -> Result<Option<super::super::heap::ObjectId>, String> {
        let Some(id) = value.object_id() else {
            return Ok(None);
        };
        Ok(matches!(self.state.heap.get(id)?, Object::Generator { .. }).then_some(id))
    }

    /// Take the value a generator returned, leaving `None` behind so that it is reported once.
    fn take_return_value(&mut self, id: super::super::heap::ObjectId) -> Result<Value, String> {
        match self.state.heap.get_mut(id)? {
            Object::Generator { return_value, .. } => {
                Ok(std::mem::replace(return_value, Value::None))
            }
            _ => Err("object is not a generator".into()),
        }
    }

    /// Close a generator as `generator.close()` does: raise `GeneratorExit` at its suspension
    /// point and let it run its cleanup. A generator that yields again instead raises
    /// `RuntimeError`; any other exception it raises stays pending.
    pub(super) fn close_generator(
        &mut self,
        id: super::super::heap::ObjectId,
    ) -> Result<(), String> {
        let value = self.allocate_exception("GeneratorExit".into(), String::new())?;
        let exit = super::RaisedException {
            kind: "GeneratorExit".into(),
            value,
        };
        match self.throw_into_generator(id, exit) {
            Ok(Some(_)) => {
                Err(self.raise_exception("RuntimeError", "generator ignored GeneratorExit"))
            }
            Ok(None) => Ok(()),
            Err(error) => match self.pending_exception.take() {
                Some(exception)
                    if matches!(exception.kind.as_str(), "GeneratorExit" | "StopIteration") =>
                {
                    Ok(())
                }
                exception => {
                    self.pending_exception = exception;
                    Err(error)
                }
            },
        }
    }

    /// Pass an exception thrown into a generator suspended in `yield from` to its subiterator.
    ///
    /// `GeneratorExit` closes the subiterator, through `close` for a subiterator that is not a
    /// generator, and is then raised in the delegating generator. Any other exception goes to
    /// the subiterator's `throw`; the result says whether it yielded, returned, or raised. A
    /// subiterator without `close` or `throw`, such as a builtin iterator, is skipped, so the
    /// exception is raised in the delegating generator.
    fn forward_throw(
        &mut self,
        subiterator: &Value,
        exception: super::RaisedException,
    ) -> Result<ForwardedThrow, String> {
        let generator = self.suspendable_generator(subiterator)?;
        if exception.kind == "GeneratorExit" {
            let closed = match generator {
                Some(id) => self.close_generator(id),
                None => match self.resolve_attribute(*subiterator, "close")? {
                    Some(close) => <Self as PyRuntime>::call_value(
                        self,
                        close,
                        CallArgs::new(Vec::new(), Vec::new()),
                    )
                    .map(|_| ())
                    .map_err(|error| self.record_native_error(error)),
                    None => Ok(()),
                },
            };
            return match closed {
                Ok(()) => Ok(ForwardedThrow::Raise(exception)),
                Err(error) => self
                    .pending_exception
                    .take()
                    .map(ForwardedThrow::Raise)
                    .ok_or(error),
            };
        }
        if let Some(id) = generator {
            return match self.throw_into_generator(id, exception) {
                Ok(Some(value)) => Ok(ForwardedThrow::Yielded(value)),
                Ok(None) => Ok(ForwardedThrow::Returned(self.take_return_value(id)?)),
                Err(error) => self
                    .pending_exception
                    .take()
                    .map(ForwardedThrow::Raise)
                    .ok_or(error),
            };
        }
        match self.resolve_attribute(*subiterator, "throw")? {
            Some(throw) => self.call_subiterator_method(throw, exception.value),
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
    ) -> Result<ForwardedThrow, String> {
        let result = <Self as PyRuntime>::call_value(
            self,
            method,
            CallArgs::new(vec![argument], Vec::new()),
        );
        let error = match result {
            Ok(value) => return Ok(ForwardedThrow::Yielded(value)),
            Err(error) => self.record_native_error(error),
        };
        let Some(exception) = self.pending_exception.take() else {
            return Err(error);
        };
        if exception.kind != "StopIteration" {
            return Ok(ForwardedThrow::Raise(exception));
        }
        let value = protocol::exception_args(&self.state.heap, &exception.value)?
            .and_then(|(_, args)| args.first().copied())
            .unwrap_or(Value::None);
        Ok(ForwardedThrow::Returned(value))
    }

    /// Resume one generator frame until its next yield or terminal return. A generator's operand
    /// stack is kept separate from its caller's stack, while its lexical scope remains in the
    /// shared heap so closures and mutations preserve normal Python aliasing.
    pub(super) fn resume_generator(
        &mut self,
        id: super::super::heap::ObjectId,
    ) -> Result<Option<Value>, String> {
        self.resume_generator_with(id, Value::None)
    }

    pub(super) fn resume_generator_with(
        &mut self,
        id: super::super::heap::ObjectId,
        sent: Value,
    ) -> Result<Option<Value>, String> {
        self.resume_generator_frame(id, GeneratorResume::Send(sent))
    }

    /// Raise the `StopIteration` that ends iteration over `iterator`. A generator that has just
    /// returned passes its return value as the exception's `value`, once; later calls, and
    /// other iterators, raise it without arguments.
    pub(super) fn raise_stop_iteration(&mut self, iterator: &Value) -> String {
        let returned = match iterator.object_id().map(|id| self.state.heap.get_mut(id)) {
            Some(Ok(Object::Generator { return_value, .. })) => {
                std::mem::replace(return_value, Value::None)
            }
            _ => Value::None,
        };
        let args = if returned.is_none() {
            Vec::new()
        } else {
            vec![returned]
        };
        self.raise_exception_args("StopIteration", args)
    }

    /// Raise `exception` at the generator's suspended `yield`, as `generator.throw` does.
    ///
    /// The frame's innermost active handler receives it, so `except`, `finally` and `with`
    /// blocks run as they would for an exception raised there. The result is the next yielded
    /// value, or `None` once the generator returns. An exception the generator does not handle,
    /// including one thrown before it starts or after it finishes, stays pending and closes it.
    fn set_generator_running(
        &mut self,
        id: super::super::heap::ObjectId,
        value: bool,
    ) -> Result<(), String> {
        if let Object::Generator { running, .. } = self.state.heap.get_mut(id)? {
            *running = value;
        }
        Ok(())
    }

    pub(super) fn throw_into_generator(
        &mut self,
        id: super::super::heap::ObjectId,
        exception: super::RaisedException,
    ) -> Result<Option<Value>, String> {
        self.resume_generator_frame(id, GeneratorResume::Throw(exception))
    }

    fn resume_generator_frame(
        &mut self,
        id: super::super::heap::ObjectId,
        resume: GeneratorResume,
    ) -> Result<Option<Value>, String> {
        const MAX_GENERATOR_DEPTH: usize = 256;
        if self.call_depth >= MAX_GENERATOR_DEPTH {
            return Err(self.raise_exception("RecursionError", "maximum recursion depth exceeded"));
        }
        let (
            code,
            scope,
            mut instruction_pointer,
            mut handlers,
            exceptions,
            mut frame_stack,
            exhausted,
            running,
        ) = match self.state.heap.get(id)?.clone() {
            Object::Generator {
                code,
                scope,
                instruction_pointer,
                handlers,
                exceptions,
                stack,
                exhausted,
                running,
                ..
            } => (
                code,
                scope,
                instruction_pointer,
                handlers,
                exceptions,
                stack,
                exhausted,
                running,
            ),
            _ => return Err("object is not a generator".into()),
        };
        if running {
            return Err("generator already executing".into());
        }
        // A frame suspended in `yield from` sits at its `YieldFromSend`; a plain `yield` resumes
        // after its `Yield`, which is never followed by a `YieldFromSend`.
        let delegating = match code.instructions.get(instruction_pointer) {
            Some(instruction) if !exhausted => match instruction.opcode {
                Opcode::YieldFromSend(finished) => Some(finished),
                _ => None,
            },
            _ => None,
        };
        let resume = match (resume, delegating) {
            (GeneratorResume::Throw(exception), Some(finished)) => {
                let subiterator = *frame_stack
                    .last()
                    .ok_or("yield from lost its subiterator")?;
                self.set_generator_running(id, true)?;
                let forwarded = self.forward_throw(&subiterator, exception);
                self.set_generator_running(id, false)?;
                match forwarded? {
                    ForwardedThrow::Yielded(value) => return Ok(Some(value)),
                    ForwardedThrow::Returned(value) => {
                        frame_stack.pop();
                        instruction_pointer = finished;
                        GeneratorResume::Send(value)
                    }
                    ForwardedThrow::Raise(exception) => GeneratorResume::Throw(exception),
                }
            }
            (resume, _) => resume,
        };
        let handler = match &resume {
            GeneratorResume::Send(sent) => {
                if exhausted {
                    // A finished generator reports its return value only to the resume that
                    // finished it.
                    self.take_return_value(id)?;
                    return Ok(None);
                }
                if instruction_pointer == 0 && !sent.is_none() {
                    return Err("can't send non-None value to a just-started generator".into());
                }
                None
            }
            GeneratorResume::Throw(_) if exhausted => None,
            GeneratorResume::Throw(_) => handlers.pop(),
        };
        if let GeneratorResume::Throw(exception) = &resume {
            if handler.is_none() {
                // Nothing at the suspension point handles it, so it leaves the generator at once.
                if let Object::Generator { exhausted, .. } = self.state.heap.get_mut(id)? {
                    *exhausted = true;
                }
                self.pending_exception = Some(exception.clone());
                return Err(format!("{} thrown into generator", exception.kind));
            }
        }
        self.set_generator_running(id, true)?;
        let outer_stack = std::mem::take(&mut self.stack);
        self.stack = frame_stack;
        let generator_exceptions = exceptions
            .into_iter()
            .map(|(kind, value)| super::RaisedException { kind, value })
            .collect();
        let outer_exceptions = std::mem::replace(&mut self.exception_stack, generator_exceptions);
        let start = match (resume, handler) {
            (GeneratorResume::Send(sent), _) => {
                if instruction_pointer != 0 {
                    self.stack.push(sent);
                }
                instruction_pointer
            }
            (GeneratorResume::Throw(exception), Some((target, depth))) => {
                self.stack.truncate(depth);
                self.stack.push(exception.value);
                self.exception_stack.push(exception);
                target
            }
            (GeneratorResume::Throw(_), None) => unreachable!("an unhandled throw returned above"),
        };
        self.local_scopes.push(scope);
        self.call_depth += 1;
        let result = self.execute_code_from(&code, start, &mut handlers, 0);
        self.call_depth -= 1;
        self.local_scopes.pop();
        let generator_exceptions = std::mem::replace(&mut self.exception_stack, outer_exceptions)
            .into_iter()
            .map(|exception| (exception.kind, exception.value))
            .collect();
        let frame_result_stack = std::mem::take(&mut self.stack);
        self.stack = outer_stack;

        match result {
            Ok(Execution::Pending) => unreachable!("execute_code_from drains pending quanta"),
            Ok(Execution::Blocked(_)) => unreachable!("generator execution cannot suspend"),
            Ok(Execution::Yield(value, next_instruction)) => {
                if let Object::Generator {
                    instruction_pointer,
                    handlers: saved_handlers,
                    exceptions: saved_exceptions,
                    stack: saved_stack,
                    running,
                    ..
                } = self.state.heap.get_mut(id)?
                {
                    *instruction_pointer = next_instruction;
                    *saved_handlers = handlers;
                    *saved_exceptions = generator_exceptions;
                    *saved_stack = frame_result_stack;
                    *running = false;
                }
                Ok(Some(value))
            }
            Ok(Execution::Return(value)) => {
                if let Object::Generator {
                    exhausted,
                    running,
                    return_value,
                    ..
                } = self.state.heap.get_mut(id)?
                {
                    *exhausted = true;
                    *running = false;
                    *return_value = value;
                }
                Ok(None)
            }
            Ok(Execution::Halt) | Ok(Execution::Exit(_)) => {
                if let Object::Generator {
                    exhausted, running, ..
                } = self.state.heap.get_mut(id)?
                {
                    *exhausted = true;
                    *running = false;
                }
                Ok(None)
            }
            Err((error, span)) => {
                if let Object::Generator {
                    exhausted, running, ..
                } = self.state.heap.get_mut(id)?
                {
                    *exhausted = true;
                    *running = false;
                }
                Err(format!(
                    "{error} at line {}, column {}",
                    span.line, span.column
                ))
            }
        }
    }
}
