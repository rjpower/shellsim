//! Bytecode dispatch, frame transitions, exception unwind, and scheduler suspension.

use super::{
    dispatch_next, protocol, BytecodeFrame, CallId, CallMode, CallResult, CodeRef, DispatchControl,
    DispatchCursor, ExceptionType, Execution, ForIterOutcome, FunctionReturn, NativeValue, Opcode,
    RaisedException, SequenceKind, TracebackFrame, Value, Vm, VM_POLL_QUANTUM,
};

impl Vm<'_> {
    pub(super) fn execute_code(
        &mut self,
        code: &CodeRef,
    ) -> Result<Execution, (String, super::super::source::Span)> {
        let mut handlers: Vec<(usize, usize)> = Vec::new();
        let stack_base = self.stack.len();
        self.execute_code_from(code, 0, &mut handlers, stack_base)
    }

    pub(super) fn execute_code_from(
        &mut self,
        code: &CodeRef,
        instruction_pointer: usize,
        handlers: &mut Vec<(usize, usize)>,
        stack_base: usize,
    ) -> Result<Execution, (String, super::super::source::Span)> {
        self.bytecode_frames.push(BytecodeFrame {
            code: code.clone(),
            instruction_pointer,
            stack_base,
            handlers: std::mem::take(handlers),
            function_return: None,
            pending_native_call: None,
        });
        let result = loop {
            match self.execute_active_frame(VM_POLL_QUANTUM) {
                Ok(Execution::Pending) => {}
                result => break result,
            }
        };
        let frame = self
            .bytecode_frames
            .pop()
            .expect("active bytecode frame must remain installed");
        if !matches!(result, Ok(Execution::Yield(_, _))) {
            self.stack.truncate(frame.stack_base);
        }
        *handlers = frame.handlers;
        result
    }

    pub(super) fn execute_active_frame(
        &mut self,
        budget: usize,
    ) -> Result<Execution, (String, super::super::source::Span)> {
        // The VM runs synchronously for one bounded quantum. Interrupt state can change only when
        // control returns to the scheduler, so one check defines the quantum's safe-point edge.
        if self.interp.deadline_interrupt.is_some() {
            return Ok(Execution::Exit(124));
        }
        self.release_transient_memory();
        if let Some(execution) = self.resume_pending_native_call()? {
            return Ok(execution);
        }
        let mut dispatch = DispatchCursor::for_active(self)
            .map_err(|error| (error, super::super::source::Span::default()))?;
        'execution: for _ in 0..budget.max(1) {
            // Native helper snapshots live for one semantic instruction. Releasing the previous
            // instruction's scratch here avoids double-counting a materialized result after it
            // has moved into an arena object.
            self.release_transient_memory();
            let instruction_pointer = dispatch.op_index;
            let code = &dispatch.code;
            let code_cache = dispatch.code_cache;
            let Some(instruction) = code.instructions.get(instruction_pointer) else {
                return Err((
                    "instruction pointer left the code object".into(),
                    super::super::source::Span::default(),
                ));
            };
            if !self.interp.resources.charge_cpu(1) {
                return Ok(Execution::Exit(137));
            }
            let opcode = instruction.opcode;
            let result: Result<DispatchControl, String> = match opcode {
                Opcode::LoadConstant(constant) => self
                    .value_from_constant(code.constant(constant))
                    .map(|value| {
                        self.stack.push(value);
                    })
                    .map(|()| DispatchControl::Next),
                Opcode::LoadName(name) => {
                    let symbol = self
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(self.load_name(symbol, code.name(name)))
                }
                Opcode::LoadGlobal(name) => {
                    let symbol = self
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(self.load_global(symbol, code, name))
                }
                Opcode::StoreName(name) => {
                    let symbol = self
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(self.store_name(symbol, code.name(name)))
                }
                Opcode::LoadLocal(slot) => dispatch_next(self.load_local(slot)),
                Opcode::StoreLocal(slot) => dispatch_next(self.store_local(slot)),
                Opcode::StoreEnclosing { name, scope_hops } => {
                    let symbol = self
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(self.store_enclosing(symbol, code.name(name), scope_hops))
                }
                Opcode::StoreNonlocal(name) => dispatch_next(self.store_nonlocal(code.name(name))),
                Opcode::StoreGlobal(name) => {
                    let symbol = self
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    let value = self.pop().map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(self.store_global(symbol, code, name, value))
                }
                Opcode::StoreAttribute(name) => {
                    let symbol = self
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    let owner = self.pop().map_err(|error| (error, dispatch.span()))?;
                    let value = self.pop().map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(self.store_attribute_by_symbol(
                        owner,
                        symbol,
                        code.name(name),
                        value,
                    ))
                }
                Opcode::StoreSubscript => dispatch_next(self.store_subscript()),
                Opcode::DeleteName(name) => {
                    let symbol = self
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    let name = code.name(name);
                    let result = if let Some(scope) = self.local_scopes.last().copied() {
                        self.state.heap.scope_remove(scope, name).map(|_| ())
                    } else {
                        self.state.globals.remove(symbol);
                        Ok(())
                    };
                    dispatch_next(result)
                }
                Opcode::DeleteLocal(slot) => dispatch_next(self.delete_local(slot)),
                Opcode::DeleteGlobal(name) => {
                    let symbol = self
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(self.delete_global(symbol, code, name))
                }
                Opcode::DeleteSubscript => dispatch_next(self.delete_subscript()),
                Opcode::Import { name, bind_root } => {
                    dispatch_next(self.import(code.name(name), bind_root))
                }
                Opcode::ImportFrom(name) => dispatch_next(self.import_from(code.name(name))),
                Opcode::LoadAttribute(name) => {
                    let symbol = self
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(self.load_attribute_at(
                        code,
                        code_cache,
                        instruction_pointer,
                        symbol,
                        code.name(name),
                    ))
                }
                Opcode::LoadSubscript => dispatch_next(self.load_subscript()),
                Opcode::BuildSlice {
                    has_start,
                    has_stop,
                    has_step,
                } => dispatch_next(self.build_slice(has_start, has_stop, has_step)),
                Opcode::BuildList(count) => {
                    dispatch_next(self.build_sequence(count, SequenceKind::List))
                }
                Opcode::BuildTuple(count) => {
                    dispatch_next(self.build_sequence(count, SequenceKind::Tuple))
                }
                Opcode::BuildDict(dict) => dispatch_next(self.build_dict(code.unpack_flags(dict))),
                Opcode::BuildSet(count) => dispatch_next(self.build_set(count)),
                Opcode::BuildUnpacked { kind, starred } => {
                    dispatch_next(self.build_unpacked(kind, code.unpack_flags(starred)))
                }
                Opcode::UnpackSequence { count, star_index } => {
                    dispatch_next(self.unpack_sequence(count, star_index))
                }
                Opcode::MakeFunction(function) => {
                    let function = code.function(function);
                    dispatch_next(self.make_function(
                        code.name(function.name).to_owned(),
                        function.code.clone(),
                        function.defaults,
                    ))
                }
                Opcode::MakeClass(class) => {
                    let class = code.class(class);
                    dispatch_next(self.make_class(
                        code.name(class.name).to_owned(),
                        &class.code,
                        class.bases,
                        class.has_metaclass,
                        &class.fields,
                    ))
                }
                Opcode::GetIterator => dispatch_next(self.get_iterator()),
                Opcode::ForIterator(target) => match self.for_iterator() {
                    Ok(ForIterOutcome::Yielded) => Ok(DispatchControl::Next),
                    Ok(ForIterOutcome::Exhausted) => Ok(DispatchControl::Jump(target)),
                    Ok(ForIterOutcome::Blocked(reason)) => {
                        // The iterator is still on the stack, unconsumed; leaving the frame's
                        // instruction pointer at this same opcode makes the next quantum retry
                        // the identical advance instead of skipping or repeating a yielded value.
                        self.active_frame_mut().instruction_pointer = instruction_pointer;
                        Ok(DispatchControl::Complete(Execution::Blocked(reason)))
                    }
                    Err(error) => Err(error),
                },
                Opcode::Unary(operator) => dispatch_next(self.unary(operator)),
                Opcode::Binary(operator) => dispatch_next(self.binary(operator)),
                Opcode::FormatValue(format) => {
                    let format = code.format(format);
                    dispatch_next(self.format_value(format.conversion, &format.format_spec))
                }
                Opcode::Compare(operator) => dispatch_next(self.compare(operator)),
                Opcode::Call(call) => {
                    self.dispatch_call(&dispatch.code, call, instruction_pointer, dispatch.span())
                }
                Opcode::Copy(depth) => dispatch_next(self.copy(depth)),
                Opcode::Swap(depth) => dispatch_next(self.swap(depth)),
                Opcode::PopTop => self.pop().map(|_| DispatchControl::Next),
                Opcode::Jump(target) => Ok(DispatchControl::Jump(target)),
                Opcode::JumpIfFalseOrPop(target) => match self.jump_if_or_pop(false) {
                    Ok(true) => Ok(DispatchControl::Jump(target)),
                    Ok(false) => Ok(DispatchControl::Next),
                    Err(error) => Err(error),
                },
                Opcode::JumpIfTrueOrPop(target) => match self.jump_if_or_pop(true) {
                    Ok(true) => Ok(DispatchControl::Jump(target)),
                    Ok(false) => Ok(DispatchControl::Next),
                    Err(error) => Err(error),
                },
                Opcode::PopJumpIfFalse(target) => {
                    let value = self.pop().map_err(|error| (error, dispatch.span()))?;
                    if !self
                        .truth_value(&value)
                        .map_err(|error| (error, dispatch.span()))?
                    {
                        Ok(DispatchControl::Jump(target))
                    } else {
                        Ok(DispatchControl::Next)
                    }
                }
                Opcode::Return => {
                    let value = self.pop().map_err(|error| (error, dispatch.span()))?;
                    if self.finish_deferred_frame(value) {
                        Ok(DispatchControl::RefreshFrame)
                    } else {
                        Ok(DispatchControl::Complete(Execution::Return(value)))
                    }
                }
                Opcode::Yield => {
                    let value = self.pop().map_err(|error| (error, dispatch.span()))?;
                    Ok(DispatchControl::Complete(Execution::Yield(
                        value,
                        instruction_pointer + 1,
                    )))
                }
                Opcode::AwaitResult => self.dispatch_await_result(),
                Opcode::RuntimeError(error) => Err(code.error(error).to_owned()),
                Opcode::Assert => self.dispatch_assert(),
                Opcode::TryBegin(target) => {
                    let depth = self.stack.len();
                    self.active_frame_mut().handlers.push((target, depth));
                    Ok(DispatchControl::Next)
                }
                Opcode::TryEnd => {
                    self.active_frame_mut()
                        .handlers
                        .pop()
                        .ok_or("invalid bytecode exception handler")
                        .map_err(|e| (e.to_string(), dispatch.span()))?;
                    Ok(DispatchControl::Next)
                }
                Opcode::MatchException { typed } => {
                    let actual = self
                        .exception_stack
                        .last()
                        .ok_or("no active exception")
                        .map_err(|e| (e.to_string(), dispatch.span()))?
                        .clone();
                    let matches = if typed {
                        let expected = self.pop().map_err(|error| (error, dispatch.span()))?;
                        self.exception_type_matches(expected, &actual)
                            .map_err(|error| (error, dispatch.span()))?
                    } else {
                        true
                    };
                    self.stack.push(Value::Bool(matches));
                    Ok(DispatchControl::Next)
                }
                Opcode::ClearException => {
                    self.exception_stack
                        .pop()
                        .ok_or("no active exception")
                        .map_err(|e| (e.to_string(), dispatch.span()))?;
                    Ok(DispatchControl::Next)
                }
                Opcode::Reraise => {
                    let exception = self
                        .exception_stack
                        .last()
                        .cloned()
                        .ok_or("no active exception")
                        .map_err(|e| (e.to_string(), dispatch.span()))?;
                    self.pending_exception = Some(exception);
                    Err("exception raised".into())
                }
                Opcode::Raise(has_value) => self.dispatch_raise(has_value),
                Opcode::WithEnter => self.dispatch_with_enter(),
                Opcode::WithExit => self.dispatch_with_exit(),
                Opcode::WithExitException => self.dispatch_with_exit_exception(),
                Opcode::AsyncWithExitException => self.dispatch_async_with_exit_exception(),
                Opcode::AsyncWithFinishException => self.dispatch_async_with_finish_exception(),
                Opcode::PopExpression => self.dispatch_pop_expression(),
                Opcode::Halt => {
                    if self.finish_deferred_frame(Value::None) {
                        Ok(DispatchControl::RefreshFrame)
                    } else {
                        Ok(DispatchControl::Complete(Execution::Halt))
                    }
                }
            };
            match result {
                Ok(DispatchControl::Next) => dispatch.op_index += 1,
                Ok(DispatchControl::Jump(target)) => dispatch.op_index = target,
                Ok(DispatchControl::RefreshFrame) => {
                    let span = dispatch.span();
                    dispatch.refresh(self).map_err(|error| (error, span))?;
                }
                Ok(DispatchControl::Complete(execution)) => return Ok(execution),
                Err(error) => {
                    dispatch.sync(self);
                    if self.propagate_error(error, dispatch.span())? {
                        let span = dispatch.span();
                        dispatch.refresh(self).map_err(|error| (error, span))?;
                        continue 'execution;
                    }
                    unreachable!("propagate_error either enters a handler or returns an error")
                }
            }
        }
        dispatch.sync(self);
        Ok(Execution::Pending)
    }

    #[inline(never)]
    fn dispatch_call(
        &mut self,
        code: &CodeRef,
        call: CallId,
        op_index: usize,
        span: super::super::source::Span,
    ) -> Result<DispatchControl, String> {
        let call = code.call(call);
        let keywords = call
            .keywords
            .iter()
            .map(|name| name.map(|name| code.name(name).to_owned()))
            .collect::<Vec<_>>();
        match self.call(
            call.positional,
            &keywords,
            &call.starred,
            CallMode::Deferred(span),
        ) {
            Ok(CallResult::Value(value)) => {
                self.stack.push(value);
                Ok(DispatchControl::Next)
            }
            Ok(CallResult::EnteredFrame) => {
                let caller = self.bytecode_frames.len() - 2;
                self.bytecode_frames[caller].instruction_pointer = op_index + 1;
                Ok(DispatchControl::RefreshFrame)
            }
            Ok(CallResult::Blocked(reason, value)) => {
                self.stack.push(value);
                self.active_frame_mut().instruction_pointer = op_index + 1;
                Ok(DispatchControl::Complete(Execution::Blocked(reason)))
            }
            Ok(CallResult::Retry(reason, pending)) => {
                let frame = self.active_frame_mut();
                frame.instruction_pointer = op_index + 1;
                frame.pending_native_call = Some(pending);
                Ok(DispatchControl::Complete(Execution::Blocked(reason)))
            }
            Ok(CallResult::Exit(status)) => Ok(DispatchControl::Complete(Execution::Exit(status))),
            Err(error) => Err(error),
        }
    }

    #[inline(never)]
    fn dispatch_assert(&mut self) -> Result<DispatchControl, String> {
        let message = self.pop()?;
        let condition = self.pop()?;
        if self.truth_value(&condition)? {
            return Ok(DispatchControl::Next);
        }
        let message = if message.is_none() {
            String::new()
        } else {
            protocol::display(&self.state.heap, &message)?
        };
        let value = self.allocate_exception("AssertionError".into(), message)?;
        self.pending_exception = Some(RaisedException {
            kind: "AssertionError".into(),
            value,
        });
        Err("assertion failed".into())
    }

    fn dispatch_await_result(&mut self) -> Result<DispatchControl, String> {
        let outcome = self.pop()?;
        let Some(id) = outcome.object_id() else {
            return Err("invalid coroutine scheduler outcome".into());
        };
        let (success, value) = match self.state.heap.get(id)? {
            super::super::heap::Object::Tuple(items) if items.len() == 2 => (items[0], items[1]),
            _ => return Err("invalid coroutine scheduler outcome".into()),
        };
        if self.truth_value(&success)? {
            self.stack.push(value);
            return Ok(DispatchControl::Next);
        }
        let kind = if let Some((kind, _)) = protocol::exception_parts(&self.state.heap, &value)? {
            kind
        } else if let Some(kind) = self.user_exception_kind(&value)? {
            kind
        } else {
            return Err("coroutine scheduler injected a non-exception".into());
        };
        self.pending_exception = Some(RaisedException { kind, value });
        Err("exception raised across await".into())
    }

    #[cold]
    #[inline(never)]
    fn dispatch_raise(&mut self, has_value: bool) -> Result<DispatchControl, String> {
        let exception = if has_value {
            let value = self.pop()?;
            if let Some((kind, _)) = protocol::exception_parts(&self.state.heap, &value)? {
                RaisedException { kind, value }
            } else if let Some(kind) = self.user_exception_kind(&value)? {
                RaisedException { kind, value }
            } else if let Some(NativeValue::ExceptionType(ExceptionType(kind))) =
                value.native_value()
            {
                let value = self.allocate_exception(kind.to_string(), String::new())?;
                RaisedException {
                    kind: kind.to_string(),
                    value,
                }
            } else {
                return Err("exceptions must derive from BaseException".into());
            }
        } else {
            self.exception_stack
                .last()
                .cloned()
                .ok_or("No active exception to reraise")?
        };
        self.pending_exception = Some(exception);
        Err("exception raised".into())
    }

    #[inline(never)]
    fn dispatch_with_enter(&mut self) -> Result<DispatchControl, String> {
        let context = self.pop()?;
        self.stack.push(context);
        self.with_contexts.push(context);
        self.load_attribute("__enter__")?;
        match self.call(0, &[], &[], CallMode::Immediate)? {
            CallResult::Value(value) => self.stack.push(value),
            CallResult::Exit(status) => {
                return Ok(DispatchControl::Complete(Execution::Exit(status)))
            }
            CallResult::EnteredFrame => unreachable!("immediate call entered a frame"),
            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                unreachable!("immediate call cannot suspend")
            }
        }
        Ok(DispatchControl::Next)
    }

    #[inline(never)]
    fn dispatch_with_exit(&mut self) -> Result<DispatchControl, String> {
        let context = self.with_contexts.pop().ok_or("with stack underflow")?;
        self.stack.push(context);
        self.load_attribute("__exit__")?;
        self.stack.extend([Value::None, Value::None, Value::None]);
        match self.call(3, &[], &[false, false, false], CallMode::Immediate)? {
            CallResult::Value(_) => Ok(DispatchControl::Next),
            CallResult::Exit(status) => Ok(DispatchControl::Complete(Execution::Exit(status))),
            CallResult::EnteredFrame => unreachable!("immediate call entered a frame"),
            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                unreachable!("immediate call cannot suspend")
            }
        }
    }

    #[inline(never)]
    fn dispatch_with_exit_exception(&mut self) -> Result<DispatchControl, String> {
        let context = self.with_contexts.pop().ok_or("with stack underflow")?;
        let exception = self
            .exception_stack
            .last()
            .cloned()
            .ok_or("no active exception")?;
        self.stack.push(context);
        self.load_attribute("__exit__")?;
        let exception_kind = self.exception_class(&exception)?;
        self.stack
            .extend([exception_kind, exception.value, Value::None]);
        let result = match self.call(3, &[], &[false, false, false], CallMode::Immediate)? {
            CallResult::Value(value) => value,
            CallResult::Exit(status) => {
                return Ok(DispatchControl::Complete(Execution::Exit(status)))
            }
            CallResult::EnteredFrame => unreachable!("immediate call entered a frame"),
            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                unreachable!("immediate call cannot suspend")
            }
        };
        if self.truth_value(&result)? {
            self.pending_exception = None;
            self.exception_stack.pop();
            Ok(DispatchControl::Next)
        } else {
            self.pending_exception = Some(exception);
            Err("exception raised".into())
        }
    }

    fn dispatch_async_with_exit_exception(&mut self) -> Result<DispatchControl, String> {
        let context = self.pop()?;
        self.pop()?;
        let exception = self
            .exception_stack
            .last()
            .cloned()
            .ok_or("no active exception")?;
        self.stack.push(context);
        self.load_attribute("__aexit__")?;
        let exception_kind = self.exception_class(&exception)?;
        self.stack
            .extend([exception_kind, exception.value, Value::None]);
        match self.call(3, &[], &[false, false, false], CallMode::Immediate)? {
            CallResult::Value(value) => {
                self.stack.push(value);
                Ok(DispatchControl::Next)
            }
            CallResult::Exit(status) => Ok(DispatchControl::Complete(Execution::Exit(status))),
            CallResult::EnteredFrame => unreachable!("immediate call entered a frame"),
            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                unreachable!("immediate call cannot suspend")
            }
        }
    }

    fn exception_class(&self, exception: &super::RaisedException) -> Result<Value, String> {
        if protocol::exception_parts(&self.state.heap, &exception.value)?.is_some() {
            let name = super::known_exception_type(&exception.kind).ok_or_else(|| {
                format!(
                    "exception type metadata is not modeled for {:?}",
                    exception.kind
                )
            })?;
            Ok(Value::Native(NativeValue::ExceptionType(ExceptionType(
                name,
            ))))
        } else {
            self.type_of(&exception.value)
        }
    }

    fn dispatch_async_with_finish_exception(&mut self) -> Result<DispatchControl, String> {
        let suppress = self.pop()?;
        let exception = self.exception_stack.pop().ok_or("no active exception")?;
        if self.truth_value(&suppress)? {
            self.pending_exception = None;
            Ok(DispatchControl::Next)
        } else {
            self.pending_exception = Some(exception);
            Err("exception raised".into())
        }
    }

    #[inline(never)]
    fn dispatch_pop_expression(&mut self) -> Result<DispatchControl, String> {
        let value = self.pop()?;
        if self.mode.interactive && !matches!(value, Value::None) {
            let rendered = protocol::repr(&self.state.heap, &value)?;
            self.out.extend_from_slice(rendered.as_bytes());
            self.out.push(b'\n');
        }
        Ok(DispatchControl::Next)
    }

    /// Resume work retained by a blocking native call before entering the opcode loop.
    ///
    /// A pending call can only be installed while returning to the scheduler, so checking it once
    /// at the next quantum boundary is sufficient. Ordinary opcodes never need to probe the frame.
    fn resume_pending_native_call(
        &mut self,
    ) -> Result<Option<Execution>, (String, super::super::source::Span)> {
        let Some(pending) = self.active_frame_mut().pending_native_call.take() else {
            return Ok(None);
        };
        let span = pending.call_span();
        match self.resume_native_call(pending) {
            Ok(CallResult::Value(value)) => {
                self.stack.push(value);
                Ok(None)
            }
            Ok(CallResult::Blocked(reason, value)) => {
                self.stack.push(value);
                Ok(Some(Execution::Blocked(reason)))
            }
            Ok(CallResult::Retry(reason, pending)) => {
                self.active_frame_mut().pending_native_call = Some(pending);
                Ok(Some(Execution::Blocked(reason)))
            }
            Ok(CallResult::Exit(status)) => Ok(Some(Execution::Exit(status))),
            Ok(CallResult::EnteredFrame) => {
                unreachable!("a retained native call cannot enter a Python frame")
            }
            Err(error) => {
                if self.propagate_error(error, span)? {
                    Ok(None)
                } else {
                    unreachable!("propagate_error either enters a handler or returns an error")
                }
            }
        }
    }

    fn active_frame_mut(&mut self) -> &mut BytecodeFrame {
        self.bytecode_frames
            .last_mut()
            .expect("bytecode execution requires an active frame")
    }

    fn propagate_error(
        &mut self,
        mut error: String,
        mut span: super::super::source::Span,
    ) -> Result<bool, (String, super::super::source::Span)> {
        // Rebuilt from scratch on every call: a handler found partway through means the
        // in-progress frame list here is irrelevant, and a fully uncaught error overwrites
        // whatever an earlier, since-discarded unwind (e.g. inside a generator sub-frame) left
        // behind. `render_execution` only ever reads the list left by the unwind that actually
        // reaches the top of the program.
        let mut frames = Vec::new();
        loop {
            if self.enter_exception_handler() {
                return Ok(true);
            }
            let Some(function_return) = self.unwind_deferred_frame() else {
                frames.push(TracebackFrame {
                    name: "<module>".to_string(),
                    span,
                });
                frames.reverse();
                self.traceback_frames = frames;
                return Err((error, span));
            };
            frames.push(TracebackFrame {
                name: function_return.name.clone(),
                span,
            });
            error = format!(
                "{error} in {} at line {}, column {}",
                function_return.name, span.line, span.column
            );
            span = function_return.call_span;
        }
    }

    fn finish_deferred_frame(&mut self, value: Value) -> bool {
        let Some(_) = self
            .bytecode_frames
            .last()
            .and_then(|frame| frame.function_return.as_ref())
        else {
            return false;
        };
        self.unwind_deferred_frame()
            .expect("deferred function frame was checked above");
        self.stack.push(value);
        true
    }

    fn unwind_deferred_frame(&mut self) -> Option<FunctionReturn> {
        self.bytecode_frames
            .last()
            .and_then(|frame| frame.function_return.as_ref())?;
        let frame = self
            .bytecode_frames
            .pop()
            .expect("deferred function frame was checked above");
        let function_return = frame
            .function_return
            .expect("deferred function frame must own return state");
        self.call_depth = self.call_depth.saturating_sub(1);
        self.local_scopes
            .pop()
            .expect("deferred function frame must own a local scope");
        if function_return.pop_method_frame {
            self.method_frames
                .pop()
                .expect("deferred method frame must remain installed");
        }
        self.stack.truncate(frame.stack_base);
        Some(function_return)
    }

    fn enter_exception_handler(&mut self) -> bool {
        let Some(exception) = self.pending_exception.take() else {
            return false;
        };
        let Some((target, depth)) = self.active_frame_mut().handlers.pop() else {
            self.pending_exception = Some(exception);
            return false;
        };
        self.stack.truncate(depth);
        self.stack.push(exception.value);
        self.exception_stack.push(exception);
        self.active_frame_mut().instruction_pointer = target;
        true
    }
}
