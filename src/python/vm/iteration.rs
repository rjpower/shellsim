//! Iteration, generator suspension, and sequence unpacking.

use super::{protocol, CallArgs, Execution, IteratorAdvance, Object, PyRuntime, Value, Vm};

impl Vm<'_> {
    pub(super) fn get_iterator(&mut self) -> Result<(), String> {
        let iterable = self.pop()?;
        let iterator = self.make_iterator(iterable)?;
        self.stack.push(iterator);
        Ok(())
    }

    pub(super) fn make_iterator(&mut self, iterable: Value) -> Result<Value, String> {
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
                | Object::CallableIterator { .. }
                | Object::Generator { .. } => {
                    // These iterators remain lazy; materializing either one here would permit an
                    // unbounded host allocation before the caller's loop can meter each item.
                    return Ok(Value::Object(id));
                }
                _ => {}
            }
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
            Object::SequenceIterator { owner, position } => Some((*owner, *position)),
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
            _ => return Ok(IteratorAdvance::Invalid),
        };
        let (owner, position) = sequence.expect("only sequence iterators reach the slow path");
        let value = match self.state.heap.get(owner)? {
            Object::List(values) | Object::Tuple(values) => values.get(position).copied(),
            _ => return Err("iterator source changed object kind".into()),
        };
        let Some(value) = value else {
            return Ok(IteratorAdvance::Exhausted);
        };
        let Object::SequenceIterator { position, .. } = self.state.heap.get_mut(iterator)? else {
            unreachable!("iterator kind was checked above")
        };
        *position += 1;
        Ok(IteratorAdvance::Yield(value))
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
            | IteratorAdvance::Invalid => Err("object is not a stored iterator".into()),
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
                return Err(format!(
                    "not enough values to unpack (expected at least {}, got {})",
                    expected.saturating_sub(1),
                    values.len()
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
                return Err(format!(
                    "cannot unpack sequence of length {} into {expected} targets",
                    values.len()
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
    pub(super) fn for_iterator(&mut self) -> Result<bool, String> {
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
                Ok(true)
            }
            IteratorAdvance::Exhausted => {
                self.stack.pop();
                Ok(false)
            }
            IteratorAdvance::Callable { callable, sentinel } => {
                self.charge_cpu(1)?;
                let value = <Self as PyRuntime>::call_value(
                    self,
                    callable,
                    CallArgs::new(Vec::new(), Vec::new()),
                )
                .map_err(|error| error.to_string())?;
                if protocol::equals(&self.state.heap, &value, &sentinel)? {
                    self.exhaust_callable_iterator(id)?;
                    self.stack.pop();
                    return Ok(false);
                }
                self.stack.push(value);
                Ok(true)
            }
            IteratorAdvance::Generator => match self.resume_generator(id)? {
                Some(value) => {
                    self.stack.push(value);
                    Ok(true)
                }
                None => {
                    self.stack.pop();
                    Ok(false)
                }
            },
            IteratorAdvance::Invalid => Err("for-loop stack does not contain an iterator".into()),
        }
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
        const MAX_GENERATOR_DEPTH: usize = 256;
        if self.call_depth >= MAX_GENERATOR_DEPTH {
            return Err("maximum recursion depth exceeded".into());
        }
        let (
            code,
            scope,
            instruction_pointer,
            mut handlers,
            exceptions,
            frame_stack,
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
        if exhausted {
            return Ok(None);
        }
        if instruction_pointer == 0 && !sent.is_none() {
            return Err("can't send non-None value to a just-started generator".into());
        }
        if running {
            return Err("generator already executing".into());
        }
        if let Object::Generator { running, .. } = self.state.heap.get_mut(id)? {
            *running = true;
        }
        let outer_stack = std::mem::take(&mut self.stack);
        self.stack = frame_stack;
        if instruction_pointer != 0 {
            self.stack.push(sent);
        }
        let generator_exceptions = exceptions
            .into_iter()
            .map(|(kind, value)| super::RaisedException { kind, value })
            .collect();
        let outer_exceptions = std::mem::replace(&mut self.exception_stack, generator_exceptions);
        self.local_scopes.push(scope);
        self.call_depth += 1;
        let result = self.execute_code_from(&code, instruction_pointer, &mut handlers, 0);
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
