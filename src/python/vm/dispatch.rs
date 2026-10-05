//! Bytecode dispatch, frame transitions, exception unwind, and scheduler suspension.

use super::super::scopes;
use super::{
    dispatch_next, exception_types, protocol, string, BytecodeFrame, CallId, CallMode, CallResult,
    CodeRef, DispatchControl, DispatchCursor, ExceptionType, Execution, ForIterOutcome, FrameEntry,
    FunctionReturn, LocalsLocation, NativeValue, Opcode, RaisedException, SequenceKind,
    TracebackFrame, Value, Vm, VM_POLL_QUANTUM,
};

impl<'s> Vm<'s> {
    /// Run `code` to completion in a new frame, synchronously.
    pub(super) fn execute_code(
        &mut self,
        code: &CodeRef,
        entry: FrameEntry<'_>,
    ) -> Result<Execution, (String, super::super::source::Span)> {
        let mut handlers: Vec<(usize, usize, usize)> = Vec::new();
        let stack_base = self.stack.len();
        self.execute_code_from(code, 0, &mut handlers, stack_base, entry)
    }

    /// Run `code` from `instruction_pointer` in a new frame, synchronously, as `execute_code`
    /// does for a fresh start and generator resumption does from a suspension point.
    pub(super) fn execute_code_from(
        &mut self,
        code: &CodeRef,
        instruction_pointer: usize,
        handlers: &mut Vec<(usize, usize, usize)>,
        stack_base: usize,
        entry: FrameEntry<'_>,
    ) -> Result<Execution, (String, super::super::source::Span)> {
        let exception_base = self.exception_stack.len();
        let frame = self
            .enter_frame(code, instruction_pointer, stack_base, entry, None)
            .map_err(|error| (error, super::super::source::Span::default()))?;
        self.bytecode_frames.push(BytecodeFrame {
            handlers: std::mem::take(handlers),
            exception_base,
            ..frame
        });
        self.synchronous_frames += 1;
        let result = loop {
            match self.execute_active_frame(VM_POLL_QUANTUM) {
                Ok(Execution::Pending) => {}
                result => break result,
            }
        };
        self.synchronous_frames -= 1;
        let frame = self
            .bytecode_frames
            .pop()
            .expect("active bytecode frame must remain installed");
        if !matches!(result, Ok(Execution::Yield(_, _))) {
            self.stack.truncate(frame.stack_base);
            self.exception_stack.truncate(frame.exception_base);
        }
        if let Some(base) = frame.locals_base {
            self.locals.truncate(base);
        }
        *handlers = frame.handlers;
        result
    }

    /// Build a frame for `code`. The frame owns the local slots above `entry.locals_base`;
    /// popping it truncates the locals stack back there. `handlers` and `exception_base`
    /// start empty and at the current depth.
    pub(super) fn enter_frame(
        &mut self,
        code: &CodeRef,
        instruction_pointer: usize,
        stack_base: usize,
        entry: FrameEntry<'_>,
        function_return: Option<FunctionReturn>,
    ) -> Result<BytecodeFrame, String> {
        let code_cache = self.ensure_code_cache(code)?;
        Ok(BytecodeFrame {
            code: code.clone(),
            code_cache,
            instruction_pointer,
            stack_base,
            locals_base: entry.locals_base,
            scope: entry.scope.map(|scope| self.store(scope)),
            enclosing: entry.enclosing.map(|scope| self.store(scope)),
            handlers: Vec::new(),
            exception_base: self.exception_stack.len(),
            function_return,
            pending_native_call: None,
        })
    }

    pub(super) fn execute_active_frame(
        &mut self,
        budget: usize,
    ) -> Result<Execution, (String, super::super::source::Span)> {
        // Handles made while executing live in this child scope and die with each instruction;
        // only stored references in the VM's roots carry values from one instruction to the next.
        let mut vm = self.scope();
        // The VM runs synchronously for one bounded quantum. Interrupt state can change only when
        // control returns to the scheduler, so one check defines the quantum's safe-point edge.
        if vm.interp.deadline_interrupt.is_some() {
            return Ok(Execution::Exit(124));
        }
        if let Some(execution) = vm.resume_pending_native_call()? {
            return Ok(execution);
        }
        let mut dispatch = DispatchCursor::for_active(&mut vm)
            .map_err(|error| (error, super::super::source::Span::default()))?;
        'execution: for _ in 0..budget.max(1) {
            // Handles and native scratch live for one semantic instruction. Releasing the
            // previous instruction's here avoids double-counting a materialized result after it
            // has moved into a heap object.
            vm.reset_scope();
            let instruction_pointer = dispatch.op_index;
            let code = &dispatch.code;
            let code_cache = dispatch.code_cache;
            let Some(instruction) = code.instructions.get(instruction_pointer) else {
                return Err((
                    "instruction pointer left the code object".into(),
                    super::super::source::Span::default(),
                ));
            };
            if !vm.interp.resources.charge_cpu(1) {
                return Ok(Execution::Exit(137));
            }
            let opcode = instruction.opcode;
            let result: Result<DispatchControl, String> = match opcode {
                Opcode::LoadConstant(constant) => vm
                    .value_from_constant(code.constant(constant))
                    .map(|value| {
                        vm.push(value);
                    })
                    .map(|()| DispatchControl::Next),
                Opcode::LoadName(name) => {
                    let symbol = vm
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(vm.load_name(symbol, code.name(name)))
                }
                Opcode::LoadGlobal(name) => {
                    let symbol = vm
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(vm.load_global(symbol, code, name))
                }
                Opcode::StoreName(name) => {
                    let symbol = vm
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(vm.store_name(symbol, code.name(name)))
                }
                Opcode::LoadLocal(slot) => dispatch_next(vm.load_fast(dispatch.locals, slot)),
                Opcode::StoreLocal(slot) => dispatch_next(vm.store_fast(dispatch.locals, slot)),
                Opcode::StoreEnclosing { name, scope_hops } => {
                    let symbol = vm
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(vm.store_enclosing(symbol, code.name(name), scope_hops))
                }
                Opcode::StoreNonlocal(name) => dispatch_next(vm.store_nonlocal(code.name(name))),
                Opcode::StoreGlobal(name) => {
                    let symbol = vm
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    let value = vm.pop().map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(vm.store_global(symbol, code, name, value))
                }
                Opcode::StoreAttribute(name) => {
                    let symbol = vm
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    let owner = vm.pop().map_err(|error| (error, dispatch.span()))?;
                    let value = vm.pop().map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(vm.store_attribute_by_symbol(
                        owner,
                        symbol,
                        code.name(name),
                        value,
                    ))
                }
                Opcode::StoreSubscript => dispatch_next(vm.store_subscript()),
                Opcode::DeleteAttribute(name) => {
                    let symbol = vm
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    let owner = vm.pop().map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(vm.delete_attribute_by_symbol(owner, symbol, code.name(name)))
                }
                Opcode::DeleteName(name) => {
                    let symbol = vm
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    let name = code.name(name);
                    let scope = vm.active_scope();
                    let result = if let Some(scope) = scope {
                        scopes::remove(&mut vm.state.heap, scope, name).map(|_| ())
                    } else {
                        vm.state.globals.remove(&vm.state.heap, symbol);
                        Ok(())
                    };
                    dispatch_next(result)
                }
                Opcode::DeleteLocal(slot) => dispatch_next(vm.delete_local(dispatch.locals, slot)),
                Opcode::DeleteGlobal(name) => {
                    let symbol = vm
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(vm.delete_global(symbol, code, name))
                }
                Opcode::DeleteSubscript => dispatch_next(vm.delete_subscript()),
                Opcode::Import { name, bind_root } => {
                    dispatch_next(vm.import(code.name(name), bind_root))
                }
                Opcode::ImportFrom(name) => dispatch_next(vm.import_from(code.name(name))),
                Opcode::ImportStar => dispatch_next(vm.import_star()),
                Opcode::LoadAttribute(name) => {
                    let symbol = vm
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(vm.load_attribute_at(
                        code,
                        code_cache,
                        instruction_pointer,
                        symbol,
                        code.name(name),
                    ))
                }
                Opcode::LoadSubscript => dispatch_next(vm.load_subscript()),
                Opcode::BuildSlice {
                    has_start,
                    has_stop,
                    has_step,
                } => dispatch_next(vm.build_slice(has_start, has_stop, has_step)),
                Opcode::BuildList(count) => {
                    dispatch_next(vm.build_sequence(count, SequenceKind::List))
                }
                Opcode::BuildTuple(count) => {
                    dispatch_next(vm.build_sequence(count, SequenceKind::Tuple))
                }
                Opcode::BuildDict(dict) => dispatch_next(vm.build_dict(code.unpack_flags(dict))),
                Opcode::BuildSet(count) => dispatch_next(vm.build_set(count)),
                Opcode::BuildUnpacked { kind, starred } => {
                    dispatch_next(vm.build_unpacked(kind, code.unpack_flags(starred)))
                }
                Opcode::UnpackSequence { count, star_index } => {
                    dispatch_next(vm.unpack_sequence(count, star_index))
                }
                Opcode::MakeFunction(function) => {
                    let function = code.function(function);
                    dispatch_next(vm.make_function(
                        code.name(function.name).to_owned(),
                        function.code.clone(),
                        function.defaults,
                    ))
                }
                Opcode::MakeClass(class) => {
                    let class = code.class(class);
                    dispatch_next(vm.make_class(
                        code.name(class.name).to_owned(),
                        &class.code,
                        class.bases,
                        class.has_metaclass,
                        &class.fields,
                    ))
                }
                Opcode::GetIterator => dispatch_next(vm.get_iterator()),
                Opcode::ForIterator(target) => match vm.for_iterator() {
                    Ok(ForIterOutcome::Yielded) => Ok(DispatchControl::Next),
                    Ok(ForIterOutcome::Exhausted) => Ok(DispatchControl::Jump(target)),
                    Ok(ForIterOutcome::Blocked(reason)) => {
                        // The iterator is still on the stack, unconsumed; leaving the frame's
                        // instruction pointer at this same opcode makes the next quantum retry
                        // the identical advance instead of skipping or repeating a yielded value.
                        vm.active_frame_mut().instruction_pointer = instruction_pointer;
                        Ok(DispatchControl::Complete(Execution::Blocked(reason)))
                    }
                    Err(error) => Err(error),
                },
                Opcode::Unary(operator) => dispatch_next(vm.unary(operator)),
                Opcode::Binary(operator) => dispatch_next(vm.binary(operator)),
                Opcode::InPlaceBinary(operator) => dispatch_next(vm.inplace_binary(operator)),
                Opcode::FormatValue(format) => {
                    let format = code.format(format);
                    dispatch_next(vm.format_value(format.conversion, &format.format_spec))
                }
                Opcode::Compare(operator) => dispatch_next(vm.compare(operator)),
                Opcode::Call(call) => {
                    vm.dispatch_call(&dispatch.code, call, instruction_pointer, dispatch.span())
                }
                Opcode::LoadMethod(name) => {
                    let symbol = vm
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(vm.load_method_at(
                        code,
                        code_cache,
                        instruction_pointer,
                        symbol,
                        code.name(name),
                    ))
                }
                Opcode::CallMethod(call) => vm.dispatch_call_method(
                    &dispatch.code,
                    call,
                    instruction_pointer,
                    dispatch.span(),
                ),
                Opcode::Copy(depth) => dispatch_next(vm.copy(depth)),
                Opcode::Swap(depth) => dispatch_next(vm.swap(depth)),
                Opcode::PopTop => vm.pop().map(|_| DispatchControl::Next),
                Opcode::Jump(target) => Ok(DispatchControl::Jump(target)),
                Opcode::JumpIfFalseOrPop(target) => match vm.jump_if_or_pop(false) {
                    Ok(true) => Ok(DispatchControl::Jump(target)),
                    Ok(false) => Ok(DispatchControl::Next),
                    Err(error) => Err(error),
                },
                Opcode::JumpIfTrueOrPop(target) => match vm.jump_if_or_pop(true) {
                    Ok(true) => Ok(DispatchControl::Jump(target)),
                    Ok(false) => Ok(DispatchControl::Next),
                    Err(error) => Err(error),
                },
                Opcode::PopJumpIfFalse(target) => {
                    let value = vm.pop().map_err(|error| (error, dispatch.span()))?;
                    if !vm
                        .truth_value(&value)
                        .map_err(|error| (error, dispatch.span()))?
                    {
                        Ok(DispatchControl::Jump(target))
                    } else {
                        Ok(DispatchControl::Next)
                    }
                }
                Opcode::Return => {
                    let value = vm.pop().map_err(|error| (error, dispatch.span()))?;
                    if vm.finish_deferred_frame(value) {
                        Ok(DispatchControl::RefreshFrame)
                    } else {
                        Ok(DispatchControl::Complete(Execution::Return(
                            vm.store(value),
                        )))
                    }
                }
                Opcode::Yield => {
                    let value = vm.pop().map_err(|error| (error, dispatch.span()))?;
                    Ok(DispatchControl::Complete(Execution::Yield(
                        vm.store(value),
                        instruction_pointer + 1,
                    )))
                }
                Opcode::YieldFromSend(target) => match vm.yield_from_send() {
                    Ok(ForIterOutcome::Yielded) => {
                        // Suspend at this instruction so the next `send` repeats the step.
                        let value = vm.pop().map_err(|error| (error, dispatch.span()))?;
                        Ok(DispatchControl::Complete(Execution::Yield(
                            vm.store(value),
                            instruction_pointer,
                        )))
                    }
                    Ok(ForIterOutcome::Exhausted) => Ok(DispatchControl::Jump(target)),
                    Ok(ForIterOutcome::Blocked(reason)) => {
                        vm.active_frame_mut().instruction_pointer = instruction_pointer;
                        Ok(DispatchControl::Complete(Execution::Blocked(reason)))
                    }
                    Err(error) => Err(error),
                },
                Opcode::AwaitResult => vm.dispatch_await_result(),
                Opcode::RuntimeError(error) => Err(code.error(error).to_owned()),
                Opcode::Assert => vm.dispatch_assert(),
                Opcode::TryBegin(target) => {
                    let depth = vm.stack.len();
                    let exception_depth = vm.exception_stack.len();
                    vm.active_frame_mut()
                        .handlers
                        .push((target, depth, exception_depth));
                    Ok(DispatchControl::Next)
                }
                Opcode::TryEnd => {
                    vm.active_frame_mut()
                        .handlers
                        .pop()
                        .ok_or("invalid bytecode exception handler")
                        .map_err(|e| (e.to_string(), dispatch.span()))?;
                    Ok(DispatchControl::Next)
                }
                Opcode::MatchException { typed } => {
                    let actual = vm
                        .exception_stack
                        .last()
                        .ok_or("no active exception")
                        .map_err(|e| (e.to_string(), dispatch.span()))?
                        .clone();
                    let matches = if typed {
                        let expected = vm.pop().map_err(|error| (error, dispatch.span()))?;
                        vm.exception_type_matches(expected, &actual)
                            .map_err(|error| (error, dispatch.span()))?
                    } else {
                        true
                    };
                    vm.push(Value::Bool(matches));
                    Ok(DispatchControl::Next)
                }
                Opcode::ClearException => {
                    vm.exception_stack
                        .pop()
                        .ok_or("no active exception")
                        .map_err(|e| (e.to_string(), dispatch.span()))?;
                    Ok(DispatchControl::Next)
                }
                Opcode::Reraise => {
                    let exception = vm
                        .exception_stack
                        .last()
                        .cloned()
                        .ok_or("no active exception")
                        .map_err(|e| (e.to_string(), dispatch.span()))?;
                    vm.pending_exception = Some(exception);
                    Err("exception raised".into())
                }
                Opcode::Raise(has_value) => vm.dispatch_raise(has_value),
                Opcode::RaiseFrom => vm.dispatch_raise_from(),
                Opcode::WithEnter => vm.dispatch_with_enter(),
                Opcode::WithExit => vm.dispatch_with_exit(),
                Opcode::WithExitException => vm.dispatch_with_exit_exception(),
                Opcode::AsyncWithExitException => vm.dispatch_async_with_exit_exception(),
                Opcode::AsyncWithFinishException => vm.dispatch_async_with_finish_exception(),
                Opcode::PopExpression => vm.dispatch_pop_expression(),
                Opcode::Halt => {
                    if vm.finish_deferred_frame(Value::None) {
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
                    dispatch.refresh(&mut vm).map_err(|error| (error, span))?;
                }
                Ok(DispatchControl::Complete(execution)) => return Ok(execution),
                Err(error) => {
                    dispatch.sync(&mut vm);
                    if vm.propagate_error(error, dispatch.span())? {
                        let span = dispatch.span();
                        dispatch.refresh(&mut vm).map_err(|error| (error, span))?;
                        continue 'execution;
                    }
                    unreachable!("propagate_error either enters a handler or returns an error")
                }
            }
        }
        dispatch.sync(&mut vm);
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
        self.dispatch_call_spec(
            code,
            call.positional,
            &call.keywords,
            &call.starred,
            op_index,
            span,
        )
    }

    /// `CallMethod`: the receiver slot `LoadMethod` left below the arguments is either the
    /// receiver, called as the first positional argument, or a marker to drop first.
    #[inline(never)]
    fn dispatch_call_method(
        &mut self,
        code: &CodeRef,
        call: CallId,
        op_index: usize,
        span: super::super::source::Span,
    ) -> Result<DispatchControl, String> {
        let call = code.call(call);
        let count = call.positional + call.keywords.len();
        let receiver_depth = count - 1;
        let bound = matches!(
            self.peek(receiver_depth)?.native_value(),
            Some(NativeValue::NoReceiver)
        );
        if bound {
            self.execution
                .stack
                .remove(receiver_depth)
                .ok_or("stack underflow")?;
            return self.dispatch_call_spec(
                code,
                call.positional - 1,
                &call.keywords,
                &call.starred[1..],
                op_index,
                span,
            );
        }
        self.dispatch_call_spec(
            code,
            call.positional,
            &call.keywords,
            &call.starred,
            op_index,
            span,
        )
    }

    fn dispatch_call_spec(
        &mut self,
        code: &CodeRef,
        positional: usize,
        keyword_names: &[Option<super::NameId>],
        starred: &[bool],
        op_index: usize,
        span: super::super::source::Span,
    ) -> Result<DispatchControl, String> {
        // Record the executing call so native code can attribute work to this line, as
        // `warnings.warn` does for its caller.
        self.active_frame_mut().instruction_pointer = op_index + 1;
        let keywords = keyword_names
            .iter()
            .map(|name| name.map(|name| code.name(name).to_owned()))
            .collect::<Vec<_>>();
        match self.call(positional, &keywords, starred, CallMode::Deferred(span)) {
            Ok(CallResult::Value(value)) => {
                self.push(value);
                Ok(DispatchControl::Next)
            }
            Ok(CallResult::EnteredFrame) => {
                let caller = self.bytecode_frames.len() - 2;
                self.bytecode_frames[caller].instruction_pointer = op_index + 1;
                Ok(DispatchControl::RefreshFrame)
            }
            Ok(CallResult::Blocked(reason, value)) => {
                self.push(value);
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
            protocol::display(self.state, message)?
        };
        let value = self.allocate_exception("AssertionError".into(), message)?;
        self.pending_exception = Some(RaisedException {
            kind: "AssertionError".into(),
            value: self.store(value),
        });
        Err("assertion failed".into())
    }

    fn dispatch_await_result(&mut self) -> Result<DispatchControl, String> {
        let outcome = self.pop()?;
        if !outcome.is_object() {
            return Err("invalid coroutine scheduler outcome".into());
        }
        let (success, value) = match self.get(outcome)? {
            super::super::heap::Object::Tuple(items) if items.len() == 2 => {
                (self.handle(&items[0]), self.handle(&items[1]))
            }
            _ => return Err("invalid coroutine scheduler outcome".into()),
        };
        if self.truth_value(&success)? {
            self.push(value);
            return Ok(DispatchControl::Next);
        }
        let Some(kind) = exception_types::exception_type_name(self.state, value)? else {
            return Err("coroutine scheduler injected a non-exception".into());
        };
        self.pending_exception = Some(RaisedException {
            kind,
            value: self.store(value),
        });
        Err("exception raised across await".into())
    }

    #[cold]
    #[inline(never)]
    fn dispatch_raise(&mut self, has_value: bool) -> Result<DispatchControl, String> {
        let exception = if has_value {
            let value = self.pop()?;
            if let Some(kind) = exception_types::exception_type_name(self.state, value)? {
                RaisedException {
                    kind,
                    value: self.store(value),
                }
            } else if let Some(NativeValue::ExceptionType(ExceptionType(kind))) =
                value.native_value()
            {
                let value = self.allocate_exception(kind.to_string(), String::new())?;
                RaisedException {
                    kind: kind.to_string(),
                    value: self.store(value),
                }
            } else if self.exception_class_base(&value)?.is_some() {
                // `raise Cls` raises `Cls()`, running any user `__init__`.
                self.push(value);
                let instance = match self.call(0, &[], &[], CallMode::Immediate)? {
                    CallResult::Value(instance) => instance,
                    CallResult::Exit(status) => {
                        return Ok(DispatchControl::Complete(Execution::Exit(status)))
                    }
                    CallResult::EnteredFrame => unreachable!("immediate call entered a frame"),
                    CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                        unreachable!("immediate call cannot suspend")
                    }
                };
                let kind = exception_types::exception_type_name(self.state, instance)?
                    .ok_or("exception class produced a non-exception instance")?;
                RaisedException {
                    kind,
                    value: self.store(instance),
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

    /// `raise exception from cause`. The cause must be an exception, an exception class, or
    /// `None`. Exceptions do not record `__cause__` yet, so a valid cause is checked and then
    /// dropped; an uncaught chained exception prints without its cause.
    #[cold]
    #[inline(never)]
    fn dispatch_raise_from(&mut self) -> Result<DispatchControl, String> {
        let cause = self.pop()?;
        let valid = cause.is_none()
            || exception_types::exception_type_name(self.state, cause)?.is_some()
            || self.exception_class_base(&cause)?.is_some();
        if !valid {
            return Err(self.raise_exception(
                "TypeError",
                "exception causes must derive from BaseException",
            ));
        }
        self.dispatch_raise(true)
    }

    #[inline(never)]
    fn dispatch_with_enter(&mut self) -> Result<DispatchControl, String> {
        let context = self.pop()?;
        // CPython looks up `__exit__` first, then `__enter__`.
        for method in ["__exit__", "__enter__"] {
            if self.resolve_attribute(context, method)?.is_none() {
                let message = format!(
                    "'{}' object does not support the context manager protocol (missed {method} \
                     method)",
                    self.type_name_of(&context)?
                );
                return Err(self.raise_exception("TypeError", message));
            }
        }
        self.push(context);
        self.load_attribute("__enter__")?;
        match self.call(0, &[], &[], CallMode::Immediate)? {
            // The context joins the cleanup stack only once `__enter__` has succeeded, so an
            // exception from `__enter__` reaches the enclosing handler, not this `__exit__`.
            CallResult::Value(value) => {
                let context = self.store(context);
                self.with_contexts.push(context);
                self.push(value);
            }
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
        self.execution.stack.push_ref(&context);
        self.load_attribute("__exit__")?;
        for _ in 0..3 {
            self.push(Value::None);
        }
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
        // The handler pushed the exception; `__exit__` receives it from the exception stack.
        self.pop()?;
        let context = self.with_contexts.pop().ok_or("with stack underflow")?;
        self.execution.stack.push_ref(&context);
        let (kind, exception) = self.active_exception()?;
        self.load_attribute("__exit__")?;
        let exception_kind = self.exception_class(&kind, &exception)?;
        self.push(exception_kind);
        self.push(exception);
        self.push(Value::None);
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
            self.pending_exception = Some(RaisedException {
                kind,
                value: self.store(exception),
            });
            Err("exception raised".into())
        }
    }

    fn dispatch_async_with_exit_exception(&mut self) -> Result<DispatchControl, String> {
        let context = self.pop()?;
        self.pop()?;
        let (kind, exception) = self.active_exception()?;
        self.push(context);
        self.load_attribute("__aexit__")?;
        let exception_kind = self.exception_class(&kind, &exception)?;
        self.push(exception_kind);
        self.push(exception);
        self.push(Value::None);
        match self.call(3, &[], &[false, false, false], CallMode::Immediate)? {
            CallResult::Value(value) => {
                self.push(value);
                Ok(DispatchControl::Next)
            }
            CallResult::Exit(status) => Ok(DispatchControl::Complete(Execution::Exit(status))),
            CallResult::EnteredFrame => unreachable!("immediate call entered a frame"),
            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                unreachable!("immediate call cannot suspend")
            }
        }
    }

    /// The innermost handled exception's kind and value, as a handle that stays valid while
    /// `__exit__` runs guest code.
    fn active_exception(&self) -> Result<(String, Value<'s>), String> {
        let exception = self.exception_stack.last().ok_or("no active exception")?;
        Ok((exception.kind.clone(), self.handle(&exception.value)))
    }

    fn exception_class(&self, kind: &str, value: &Value<'s>) -> Result<Value<'s>, String> {
        // A builtin exception instance has no class object; its class is the registered type.
        if exception_types::exception_base(self.state, *value)?.is_some()
            && self.instance_class(*value)?.is_none()
        {
            let name = super::known_exception_type(kind)
                .ok_or_else(|| format!("exception type metadata is not modeled for {kind:?}"))?;
            Ok(Value::Native(NativeValue::ExceptionType(ExceptionType(
                name,
            ))))
        } else {
            self.type_of(value)
        }
    }

    fn dispatch_async_with_finish_exception(&mut self) -> Result<DispatchControl, String> {
        let suppress = self.pop()?;
        let exception = self.exception_stack.pop().ok_or("no active exception")?;
        let (kind, value) = (exception.kind, self.handle(&exception.value));
        if self.truth_value(&suppress)? {
            self.pending_exception = None;
            Ok(DispatchControl::Next)
        } else {
            self.pending_exception = Some(RaisedException {
                kind,
                value: self.store(value),
            });
            Err("exception raised".into())
        }
    }

    #[inline(never)]
    fn dispatch_pop_expression(&mut self) -> Result<DispatchControl, String> {
        let value = self.pop()?;
        if self.mode.interactive && !value.is_none() {
            let rendered = self.repr_value(&value)?;
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
                self.push(value);
                Ok(None)
            }
            Ok(CallResult::Blocked(reason, value)) => {
                self.push(value);
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

    /// `LoadLocal`: copy a local slot's stored reference onto the operand stack without making
    /// a handle.
    #[inline(always)]
    fn load_fast(&mut self, locals: LocalsLocation, slot: usize) -> Result<(), String> {
        let value = match locals {
            LocalsLocation::Stack(base) => {
                let value = self
                    .execution
                    .locals
                    .get(base + slot)
                    .and_then(Option::as_ref)
                    .ok_or("local variable referenced before assignment")?
                    .dup();
                self.execution.stack.push_ref(&value);
                return Ok(());
            }
            LocalsLocation::Heap => {
                let scope = self.active_scope().expect("heap locals have a scope");
                scopes::local_ref(&self.state.heap, scope, slot)?
            }
            LocalsLocation::None => return Err("local bytecode requires local slots".into()),
        };
        let value = value.ok_or("local variable referenced before assignment")?;
        self.execution.stack.push_ref(value);
        Ok(())
    }

    /// `StoreLocal`: move the operand stack's top reference into a local slot without making a
    /// handle for it.
    #[inline(always)]
    fn store_fast(&mut self, locals: LocalsLocation, slot: usize) -> Result<(), String> {
        if self.frame_stack_len() == 0 {
            return Err("invalid bytecode stack effect".into());
        }
        let value = self
            .execution
            .stack
            .pop_ref()
            .expect("non-empty frame stack was checked");
        match locals {
            LocalsLocation::Stack(base) => {
                let local = self
                    .locals
                    .get_mut(base + slot)
                    .ok_or("invalid local slot")?;
                *local = Some(value);
                Ok(())
            }
            LocalsLocation::Heap => {
                let scope = self.active_scope().expect("heap locals have a scope");
                scopes::store_local_ref(&mut self.state.heap, scope, slot, value)
            }
            LocalsLocation::None => Err("local bytecode requires local slots".into()),
        }
    }

    /// `DeleteLocal`: unbind a local slot.
    fn delete_local(&mut self, locals: LocalsLocation, slot: usize) -> Result<(), String> {
        match locals {
            LocalsLocation::Stack(base) => {
                let local = self
                    .locals
                    .get_mut(base + slot)
                    .ok_or("invalid local slot")?;
                *local = None;
                Ok(())
            }
            LocalsLocation::Heap => {
                let scope = self.active_scope().expect("heap locals have a scope");
                scopes::remove_local(&mut self.state.heap, scope, slot).map(|_| ())
            }
            LocalsLocation::None => Err("local bytecode requires local slots".into()),
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
            let file = self.imported_module_file();
            let Some(function_return) = self.unwind_deferred_frame() else {
                frames.push(TracebackFrame {
                    name: "<module>".to_string(),
                    span,
                    file,
                });
                frames.reverse();
                self.traceback_frames = frames;
                return Err((error, span));
            };
            let name = self.function_name(&function_return.function);
            frames.push(TracebackFrame {
                name: name.clone(),
                span,
                file,
            });
            error = format!(
                "{error} in {name} at line {}, column {}",
                span.line, span.column
            );
            span = function_return.call_span;
        }
    }

    fn finish_deferred_frame(&mut self, value: Value<'_>) -> bool {
        let Some(_) = self
            .bytecode_frames
            .last()
            .and_then(|frame| frame.function_return.as_ref())
        else {
            return false;
        };
        self.unwind_deferred_frame()
            .expect("deferred function frame was checked above");
        self.push(value);
        true
    }

    /// `__file__` of the module whose code the active frame runs, when that module was imported.
    /// Imported code runs under the module's scope; the main program keeps its globals outside
    /// the scope chain, so its frames find no `__file__` here.
    fn imported_module_file(&self) -> Option<String> {
        let heap = &self.state.heap;
        let root = scopes::root(heap, self.lookup_scope()?).ok()?;
        let file = scopes::get(heap, root, "__file__").ok()??;
        string::string_value(heap, file).ok().flatten()
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
        if let Some(base) = frame.locals_base {
            self.locals.truncate(base);
        }
        if function_return.pop_method_frame {
            self.method_frames
                .pop()
                .expect("deferred method frame must remain installed");
        }
        self.stack.truncate(frame.stack_base);
        self.exception_stack.truncate(frame.exception_base);
        Some(function_return)
    }

    fn enter_exception_handler(&mut self) -> bool {
        let Some(exception) = self.pending_exception.take() else {
            return false;
        };
        let Some((target, depth, exception_depth)) = self.active_frame_mut().handlers.pop() else {
            self.pending_exception = Some(exception);
            return false;
        };
        self.stack.truncate(depth);
        // Exceptions handled inside the `try` body, including one a handler was processing when
        // this exception escaped it, end with the body.
        self.exception_stack.truncate(exception_depth);
        self.execution.stack.push_ref(&exception.value);
        self.exception_stack.push(exception);
        self.active_frame_mut().instruction_pointer = target;
        true
    }
}
