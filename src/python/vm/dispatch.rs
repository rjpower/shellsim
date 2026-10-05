//! Bytecode dispatch, frame transitions, exception unwind, and scheduler suspension.

use super::super::heap::Ref;
use super::super::scopes;
use super::{
    dispatch_next, exception_types, frame_index, protocol, string, BytecodeFrame, CallId, CallMode,
    CodeRef, DispatchCursor, ExceptionType, Flow, ForIterOutcome, FrameEntry, LocalsLocation,
    NativeValue, Opcode, PendingNativeCall, RaisedException, SequenceKind, Suspension,
    TracebackFrame, Value, Vm, VM_POLL_QUANTUM,
};

/// Where a frame that suspends at `yield` stops, which `try` regions it has open and which
/// context managers it has entered, so a generator can resume from exactly that point.
pub(super) struct ResumePoint {
    pub(super) instruction_pointer: usize,
    pub(super) handlers: Vec<(usize, usize, usize)>,
    pub(super) contexts: Vec<Ref>,
}

impl<'s> Vm<'s> {
    /// Run `code` to completion in a new frame, synchronously.
    pub(super) fn execute_code(
        &mut self,
        code: &CodeRef,
        entry: FrameEntry<'_>,
    ) -> Result<Flow, (String, super::super::source::Span)> {
        let mut resume = ResumePoint {
            instruction_pointer: 0,
            handlers: Vec::new(),
            contexts: Vec::new(),
        };
        let stack_base = self.stack.len();
        self.execute_code_from(code, &mut resume, stack_base, entry)
    }

    /// Run `code` from `resume` in a new frame, synchronously, as `execute_code` does for a
    /// fresh start and generator resumption does from a suspension point. On return `resume`
    /// holds where the frame stopped.
    pub(super) fn execute_code_from(
        &mut self,
        code: &CodeRef,
        resume: &mut ResumePoint,
        stack_base: usize,
        entry: FrameEntry<'_>,
    ) -> Result<Flow, (String, super::super::source::Span)> {
        let frame = self
            .enter_frame(code, resume.instruction_pointer, stack_base, entry, false)
            .map_err(|error| (error, super::super::source::Span::default()))?;
        self.handlers.append(&mut resume.handlers);
        self.with_contexts.append(&mut resume.contexts);
        let depth = self.bytecode_frames.len();
        self.bytecode_frames.push(frame);
        self.synchronous_frames += 1;
        let result = loop {
            match self.execute_active_frame(VM_POLL_QUANTUM) {
                Ok(Flow::Pending) => {}
                result => break result,
            }
        };
        self.synchronous_frames -= 1;
        // A process exit leaves the frames this execution called installed above its own;
        // they end with it.
        self.bytecode_frames.truncate(depth + 1);
        let frame = self
            .bytecode_frames
            .pop()
            .expect("active bytecode frame must remain installed");
        // A yielding frame leaves its operands and handled exceptions for the generator to
        // save; everything else it owned is released here.
        let yielded = matches!(result, Ok(Flow::Yield(_)));
        resume.instruction_pointer = frame.instruction_pointer;
        resume.handlers = self.handlers.split_off(frame.handler_base as usize);
        resume.contexts = self.with_contexts.split_off(frame.context_base as usize);
        self.leave_frame(&frame, yielded);
        result
    }

    /// Build a frame for `code` whose shared-stack bases are the current depths: it owns the
    /// local slots above `entry.locals_base` and whatever the shared stacks gain while it
    /// runs. `called` marks a frame a Python call entered, which returns to the frame below.
    pub(super) fn enter_frame(
        &mut self,
        code: &CodeRef,
        instruction_pointer: usize,
        stack_base: usize,
        entry: FrameEntry<'_>,
        called: bool,
    ) -> Result<BytecodeFrame, String> {
        let code_cache = frame_index(self.ensure_code_cache(code)?)?;
        Ok(BytecodeFrame {
            code: code.clone(),
            instruction_pointer,
            code_cache,
            exception_base: frame_index(self.exception_stack.len())?,
            stack_base: frame_index(stack_base)?,
            handler_base: frame_index(self.handlers.len())?,
            context_base: frame_index(self.with_contexts.len())?,
            locals_base: entry.locals_base.map(frame_index).transpose()?,
            scope: entry.scope.map(|scope| self.store(scope)),
            callee: entry.callee.map(|callee| self.store(callee)),
            own_scope: entry.own_scope,
            called,
            class_body: entry.class_body,
        })
    }

    /// Release what a popped frame owned on the shared stacks. A yielding frame keeps its
    /// operands and handled exceptions, which the generator saves.
    fn leave_frame(&mut self, frame: &BytecodeFrame, yielded: bool) {
        if !yielded {
            self.stack.truncate(frame.stack_base());
            self.exception_stack.truncate(frame.exception_base as usize);
        }
        if let Some(base) = frame.locals_base() {
            self.locals.truncate(base);
        }
        self.handlers.truncate(frame.handler_base as usize);
        self.with_contexts.truncate(frame.context_base as usize);
    }

    pub(super) fn execute_active_frame(
        &mut self,
        budget: usize,
    ) -> Result<Flow, (String, super::super::source::Span)> {
        // Handles made while executing live in this child scope and die with each instruction;
        // only stored references in the VM's roots carry values from one instruction to the next.
        // An instruction that works in place on the operand stack makes no handle at all, so
        // the operand stack, a root, is what keeps its values alive.
        let mut vm = self.scope();
        // The VM runs synchronously for one bounded quantum. Interrupt state can change only when
        // control returns to the scheduler, so one check defines the quantum's safe-point edge.
        if vm.interp.deadline_interrupt.is_some() {
            return Ok(Flow::Exit(124));
        }
        if let Some(flow) = vm.resume_suspension()? {
            return Ok(flow);
        }
        // CPU is metered per quantum: the loop counts instructions and charges them in one
        // call when it leaves or when a call hands control to Rust code, so natives see an
        // exact meter. The quantum never exceeds what the limit has left, so the count of
        // instructions executed before exhaustion is the same as charging each one.
        let remaining = vm.interp.resources.cpu_remaining();
        if remaining == 0 || vm.interp.resources.is_stopped() {
            // The failed charge records the stop reason, as charging the instruction would.
            vm.interp.resources.charge_cpu(1);
            return Ok(Flow::Exit(137));
        }
        let quantum = usize::try_from(remaining).map_or(budget, |remaining| budget.min(remaining));
        let mut dispatch = DispatchCursor::for_active(&mut vm)
            .map_err(|error| (error, super::super::source::Span::default()))?;
        let mut executed = 0;
        let result = vm.run_quantum(&mut dispatch, quantum.max(1), &mut executed);
        vm.flush_cpu(&mut executed);
        result
    }

    /// Charge the instructions counted since the last flush.
    #[inline(always)]
    fn flush_cpu(&mut self, executed: &mut u64) {
        if *executed != 0 {
            // A failed charge records the stop; the next quantum ends the process.
            self.interp.resources.charge_cpu(*executed);
            *executed = 0;
        }
    }

    /// Run up to `quantum` instructions of the active frame, counting them in `executed`.
    fn run_quantum(
        &mut self,
        dispatch: &mut DispatchCursor,
        quantum: usize,
        executed: &mut u64,
    ) -> Result<Flow, (String, super::super::source::Span)> {
        'execution: for _ in 0..quantum {
            // A resource limit a native or a nested execution hit stops the process before the
            // next instruction, as the per-instruction charge did; the quantum itself cannot
            // exceed the CPU limit.
            if self.interp.resources.is_stopped() {
                return Ok(Flow::Exit(137));
            }
            *executed += 1;
            // Handles and native scratch live for one semantic instruction. Releasing the
            // previous instruction's here avoids double-counting a materialized result after it
            // has moved into a heap object.
            self.reset_scope_if_used();
            let instruction_pointer = dispatch.op_index;
            let code = &dispatch.code;
            let code_cache = dispatch.code_cache;
            let Some(instruction) = code.instructions.get(instruction_pointer) else {
                return Err((
                    "instruction pointer left the code object".into(),
                    super::super::source::Span::default(),
                ));
            };
            if self
                .execution
                .test_timeout
                .is_some_and(|timeout| timeout.remaining(self.interp) == 0)
            {
                self.execution.test_timeout = None;
                dispatch.sync(self);
                let error =
                    self.record_native_error(super::PyError::exception("Failed", "test timed out"));
                self.propagate_error(error, dispatch.span())?;
                let span = dispatch.span();
                dispatch.refresh(self).map_err(|error| (error, span))?;
                continue 'execution;
            }
            let opcode = instruction.opcode;
            let result: Result<Flow, String> = match opcode {
                Opcode::LoadConstant(constant) => self
                    .value_from_constant(code.constant(constant))
                    .map(|value| {
                        self.push(value);
                    })
                    .map(|()| Flow::Next),
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
                Opcode::LoadLocal(slot) => dispatch_next(self.load_fast(dispatch.locals, slot)),
                Opcode::StoreLocal(slot) => dispatch_next(self.store_fast(dispatch.locals, slot)),
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
                Opcode::DeleteAttribute(name) => {
                    let symbol = self
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    let owner = self.pop().map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(self.delete_attribute_by_symbol(owner, symbol, code.name(name)))
                }
                Opcode::DeleteName(name) => {
                    let symbol = self
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    let name = code.name(name);
                    let scope = self.active_scope();
                    let result = if let Some(scope) = scope {
                        scopes::remove(&mut self.state.heap, scope, name).map(|_| ())
                    } else {
                        self.state.globals.remove(&self.state.heap, symbol);
                        Ok(())
                    };
                    dispatch_next(result)
                }
                Opcode::DeleteLocal(slot) => {
                    dispatch_next(self.delete_local(dispatch.locals, slot))
                }
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
                Opcode::ImportStar => dispatch_next(self.import_star()),
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
                    Ok(ForIterOutcome::Yielded) => Ok(Flow::Next),
                    Ok(ForIterOutcome::Exhausted) => Ok(Flow::Jump(target)),
                    Ok(ForIterOutcome::Blocked(reason)) => {
                        // The iterator is still on the stack, unconsumed; leaving the frame's
                        // instruction pointer at this same opcode makes the next quantum retry
                        // the identical advance instead of skipping or repeating a yielded value.
                        self.active_frame_mut().instruction_pointer = instruction_pointer;
                        Ok(self.suspend(reason, None))
                    }
                    Err(error) => Err(error),
                },
                Opcode::Unary(operator) => dispatch_next(self.unary(operator)),
                Opcode::Binary(operator) => dispatch_next(self.binary(operator)),
                Opcode::InPlaceBinary(operator) => dispatch_next(self.inplace_binary(operator)),
                Opcode::FormatValue(format) => {
                    let format = code.format(format);
                    dispatch_next(self.format_value(format.conversion, &format.format_spec))
                }
                Opcode::Compare(operator) => dispatch_next(self.compare(operator)),
                Opcode::Call(call) => {
                    self.flush_cpu(executed);
                    self.dispatch_call(&dispatch.code, call, instruction_pointer, dispatch.span())
                }
                Opcode::LoadMethod(name) => {
                    let symbol = self
                        .symbol_for(code, code_cache, name)
                        .map_err(|error| (error, dispatch.span()))?;
                    dispatch_next(self.load_method_at(
                        code,
                        code_cache,
                        instruction_pointer,
                        symbol,
                        code.name(name),
                    ))
                }
                Opcode::CallMethod(call) => {
                    self.flush_cpu(executed);
                    self.dispatch_call_method(
                        &dispatch.code,
                        call,
                        instruction_pointer,
                        dispatch.span(),
                    )
                }
                Opcode::Copy(depth) => dispatch_next(self.copy(depth)),
                Opcode::Swap(depth) => dispatch_next(self.swap(depth)),
                Opcode::PopTop => self.pop_ref().map(|_| Flow::Next),
                Opcode::Jump(target) => Ok(Flow::Jump(target)),
                Opcode::JumpIfFalseOrPop(target) => match self.jump_if_or_pop(false) {
                    Ok(true) => Ok(Flow::Jump(target)),
                    Ok(false) => Ok(Flow::Next),
                    Err(error) => Err(error),
                },
                Opcode::JumpIfTrueOrPop(target) => match self.jump_if_or_pop(true) {
                    Ok(true) => Ok(Flow::Jump(target)),
                    Ok(false) => Ok(Flow::Next),
                    Err(error) => Err(error),
                },
                Opcode::PopJumpIfFalse(target) => {
                    let value = self.pop().map_err(|error| (error, dispatch.span()))?;
                    if !self
                        .truth_value(&value)
                        .map_err(|error| (error, dispatch.span()))?
                    {
                        Ok(Flow::Jump(target))
                    } else {
                        Ok(Flow::Next)
                    }
                }
                Opcode::Return => {
                    let value = self.pop_ref().map_err(|error| (error, dispatch.span()))?;
                    match self.finish_called_frame(value) {
                        Ok(()) => Ok(Flow::Refresh),
                        Err(value) => Ok(Flow::Return(value)),
                    }
                }
                Opcode::Yield => {
                    let value = self.pop_ref().map_err(|error| (error, dispatch.span()))?;
                    self.active_frame_mut().instruction_pointer = instruction_pointer + 1;
                    Ok(Flow::Yield(value))
                }
                Opcode::YieldFromSend(target) => match self.yield_from_send() {
                    Ok(ForIterOutcome::Yielded) => {
                        // Suspend at this instruction so the next `send` repeats the step.
                        let value = self.pop().map_err(|error| (error, dispatch.span()))?;
                        self.active_frame_mut().instruction_pointer = instruction_pointer;
                        Ok(Flow::Yield(self.store(value)))
                    }
                    Ok(ForIterOutcome::Exhausted) => Ok(Flow::Jump(target)),
                    Ok(ForIterOutcome::Blocked(reason)) => {
                        self.active_frame_mut().instruction_pointer = instruction_pointer;
                        Ok(self.suspend(reason, None))
                    }
                    Err(error) => Err(error),
                },
                Opcode::AwaitResult => self.dispatch_await_result(),
                Opcode::RuntimeError(error) => Err(code.error(error).to_owned()),
                Opcode::Assert => self.dispatch_assert(),
                Opcode::TryBegin(target) => {
                    let depth = self.stack.len();
                    let exception_depth = self.exception_stack.len();
                    self.handlers.push((target, depth, exception_depth));
                    Ok(Flow::Next)
                }
                Opcode::TryEnd => {
                    self.pop_handler()
                        .ok_or("invalid bytecode exception handler")
                        .map_err(|e| (e.to_string(), dispatch.span()))?;
                    Ok(Flow::Next)
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
                    self.push(Value::Bool(matches));
                    Ok(Flow::Next)
                }
                Opcode::ClearException => {
                    self.exception_stack
                        .pop()
                        .ok_or("no active exception")
                        .map_err(|e| (e.to_string(), dispatch.span()))?;
                    Ok(Flow::Next)
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
                Opcode::RaiseFrom => self.dispatch_raise_from(),
                Opcode::WithEnter => self.dispatch_with_enter(),
                Opcode::WithExit => self.dispatch_with_exit(),
                Opcode::WithExitException => self.dispatch_with_exit_exception(),
                Opcode::AsyncWithExitException => self.dispatch_async_with_exit_exception(),
                Opcode::AsyncWithFinishException => self.dispatch_async_with_finish_exception(),
                Opcode::PopExpression => self.dispatch_pop_expression(),
                Opcode::Halt => match self.finish_called_frame(Ref::from_immediate(Value::None)) {
                    Ok(()) => Ok(Flow::Refresh),
                    Err(_) => Ok(Flow::Halt),
                },
            };
            match result {
                Ok(Flow::Next) => dispatch.op_index += 1,
                Ok(Flow::Jump(target)) => dispatch.op_index = target,
                Ok(Flow::Refresh) => {
                    let span = dispatch.span();
                    dispatch.refresh(self).map_err(|error| (error, span))?;
                }
                Ok(flow) => return Ok(flow),
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
        Ok(Flow::Pending)
    }

    /// Leave the dispatch loop to wait for `reason`, retrying `retry` when the process wakes.
    pub(super) fn suspend(
        &mut self,
        reason: crate::scheduler::WaitReason,
        retry: Option<PendingNativeCall>,
    ) -> Flow {
        self.suspension = Some(Box::new(Suspension { reason, retry }));
        Flow::Blocked
    }

    #[inline(never)]
    fn dispatch_call(
        &mut self,
        code: &CodeRef,
        call: CallId,
        op_index: usize,
        span: super::super::source::Span,
    ) -> Result<Flow, String> {
        let call = code.call(call);
        self.dispatch_call_spec(
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
    ) -> Result<Flow, String> {
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
                call.positional - 1,
                &call.keywords,
                &call.starred[1..],
                op_index,
                span,
            );
        }
        self.dispatch_call_spec(
            call.positional,
            &call.keywords,
            &call.starred,
            op_index,
            span,
        )
    }

    fn dispatch_call_spec(
        &mut self,
        positional: usize,
        keywords: &[super::super::bytecode::KeywordName],
        starred: &[bool],
        op_index: usize,
        span: super::super::source::Span,
    ) -> Result<Flow, String> {
        // The calling frame resumes past the call whether the callee runs in its own frame,
        // suspends, or returns here; recording that first also lets native code attribute
        // work to this line, as `warnings.warn` does for its caller.
        self.active_frame_mut().instruction_pointer = op_index + 1;
        self.call(positional, keywords, starred, CallMode::Deferred(span))
    }

    /// The value of a call made in [`CallMode::Immediate`], which leaves it on the operand
    /// stack; an exit request propagates as the flow to return from the current arm.
    pub(super) fn immediate_value(
        &mut self,
        flow: Flow,
    ) -> Result<Result<Value<'s>, Flow>, String> {
        match flow {
            Flow::Next => self.pop().map(Ok),
            Flow::Exit(status) => Ok(Err(Flow::Exit(status))),
            flow => unreachable!("an immediate call cannot end with {flow:?}"),
        }
    }

    #[inline(never)]
    fn dispatch_assert(&mut self) -> Result<Flow, String> {
        let message = self.pop()?;
        let condition = self.pop()?;
        if self.truth_value(&condition)? {
            return Ok(Flow::Next);
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

    fn dispatch_await_result(&mut self) -> Result<Flow, String> {
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
            return Ok(Flow::Next);
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
    fn dispatch_raise(&mut self, has_value: bool) -> Result<Flow, String> {
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
                let flow = self.call(0, &[], &[], CallMode::Immediate)?;
                let instance = match self.immediate_value(flow)? {
                    Ok(instance) => instance,
                    Err(flow) => return Ok(flow),
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
    fn dispatch_raise_from(&mut self) -> Result<Flow, String> {
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
    fn dispatch_with_enter(&mut self) -> Result<Flow, String> {
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
        let flow = self.call(0, &[], &[], CallMode::Immediate)?;
        // The context joins the cleanup stack only once `__enter__` has succeeded, so an
        // exception from `__enter__` reaches the enclosing handler, not this `__exit__`.
        match self.immediate_value(flow)? {
            Ok(value) => {
                let context = self.store(context);
                self.with_contexts.push(context);
                self.push(value);
                Ok(Flow::Next)
            }
            Err(flow) => Ok(flow),
        }
    }

    #[inline(never)]
    fn dispatch_with_exit(&mut self) -> Result<Flow, String> {
        let context = self.with_contexts.pop().ok_or("with stack underflow")?;
        self.execution.stack.push_ref(&context);
        self.load_attribute("__exit__")?;
        for _ in 0..3 {
            self.push(Value::None);
        }
        let flow = self.call(3, &[], &[false, false, false], CallMode::Immediate)?;
        match self.immediate_value(flow)? {
            Ok(_) => Ok(Flow::Next),
            Err(flow) => Ok(flow),
        }
    }

    #[inline(never)]
    fn dispatch_with_exit_exception(&mut self) -> Result<Flow, String> {
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
        let flow = self.call(3, &[], &[false, false, false], CallMode::Immediate)?;
        let result = match self.immediate_value(flow)? {
            Ok(result) => result,
            Err(flow) => return Ok(flow),
        };
        if self.truth_value(&result)? {
            self.pending_exception = None;
            self.exception_stack.pop();
            Ok(Flow::Next)
        } else {
            self.pending_exception = Some(RaisedException {
                kind,
                value: self.store(exception),
            });
            Err("exception raised".into())
        }
    }

    fn dispatch_async_with_exit_exception(&mut self) -> Result<Flow, String> {
        let context = self.pop()?;
        self.pop()?;
        let (kind, exception) = self.active_exception()?;
        self.push(context);
        self.load_attribute("__aexit__")?;
        let exception_kind = self.exception_class(&kind, &exception)?;
        self.push(exception_kind);
        self.push(exception);
        self.push(Value::None);
        // The awaited `__aexit__` result stays on the stack for the following await.
        self.call(3, &[], &[false, false, false], CallMode::Immediate)
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

    fn dispatch_async_with_finish_exception(&mut self) -> Result<Flow, String> {
        let suppress = self.pop()?;
        let exception = self.exception_stack.pop().ok_or("no active exception")?;
        let (kind, value) = (exception.kind, self.handle(&exception.value));
        if self.truth_value(&suppress)? {
            self.pending_exception = None;
            Ok(Flow::Next)
        } else {
            self.pending_exception = Some(RaisedException {
                kind,
                value: self.store(value),
            });
            Err("exception raised".into())
        }
    }

    #[inline(never)]
    fn dispatch_pop_expression(&mut self) -> Result<Flow, String> {
        let value = self.pop()?;
        if self.mode.interactive && !value.is_none() {
            let rendered = self.repr_value(&value)?;
            self.out.extend_from_slice(rendered.as_bytes());
            self.out.push(b'\n');
        }
        Ok(Flow::Next)
    }

    /// Finish the wait the previous quantum left, retrying its interrupted native call before
    /// the opcode loop runs. `None` means the loop can proceed.
    ///
    /// A suspension can only be installed while returning to the scheduler, so checking it
    /// once at the next quantum boundary is sufficient.
    fn resume_suspension(&mut self) -> Result<Option<Flow>, (String, super::super::source::Span)> {
        let Some(suspension) = self.suspension.take() else {
            return Ok(None);
        };
        let Some(pending) = suspension.retry else {
            return Ok(None);
        };
        let span = pending.call_span();
        match self.resume_native_call(pending) {
            Ok(Flow::Next) => Ok(None),
            Ok(flow) => Ok(Some(flow)),
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
            let Some(callee) = self.unwind_called_frame() else {
                frames.push(TracebackFrame {
                    name: "<module>".to_string(),
                    span,
                    file,
                });
                frames.reverse();
                self.traceback_frames = frames;
                return Err((error, span));
            };
            let name = self.function_name(&callee);
            frames.push(TracebackFrame {
                name: name.clone(),
                span,
                file,
            });
            error = format!(
                "{error} in {name} at line {}, column {}",
                span.line, span.column
            );
            span = self.call_site_span();
        }
    }

    /// The span of the call the active frame is executing: its instruction pointer already
    /// points past the call.
    fn call_site_span(&self) -> super::super::source::Span {
        let frame = self
            .bytecode_frames
            .last()
            .expect("a called frame has a caller");
        frame
            .instruction_pointer
            .checked_sub(1)
            .and_then(|site| frame.code.spans.get(site).copied())
            .unwrap_or_default()
    }

    /// Return `value` from a called frame to its caller. `false` when the active frame was
    /// entered synchronously by Rust code, which takes the value from the dispatch loop.
    /// Pop the active frame when a call entered it and leave `value` on the caller's stack.
    /// A frame Rust code entered stays in place and the value comes back for it to return.
    fn finish_called_frame(&mut self, value: Ref) -> Result<(), Ref> {
        if self.unwind_called_frame().is_none() {
            return Err(value);
        }
        self.execution.stack.push_ref(&value);
        Ok(())
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

    /// Pop the active frame when a call entered it, returning the function it ran; `None`
    /// leaves a frame Rust code entered in place.
    fn unwind_called_frame(&mut self) -> Option<Ref> {
        if !self.bytecode_frames.last()?.called {
            return None;
        }
        let frame = self
            .bytecode_frames
            .pop()
            .expect("called frame was checked above");
        self.call_depth = self.call_depth.saturating_sub(1);
        self.leave_frame(&frame, false);
        frame.callee
    }

    /// Close the active frame's innermost open `try` region.
    fn pop_handler(&mut self) -> Option<(usize, usize, usize)> {
        let base = self.bytecode_frames.last()?.handler_base as usize;
        if self.handlers.len() <= base {
            return None;
        }
        self.handlers.pop()
    }

    fn enter_exception_handler(&mut self) -> bool {
        let Some(exception) = self.pending_exception.take() else {
            return false;
        };
        let Some((target, depth, exception_depth)) = self.pop_handler() else {
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
