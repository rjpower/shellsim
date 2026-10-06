//! Bytecode dispatch, frame transitions, exception unwind, and scheduler suspension.

use super::super::heap::Ref;
use super::super::scopes;
use super::{
    dispatch_next, exception_types, frame_index, protocol, string, BytecodeFrame, CallId, CallMode,
    CodeRef, Consumer, DispatchCursor, ExceptionType, Flow, ForIterOutcome, FrameEntry, FrameKind,
    LocalsLocation, NativeValue, Opcode, PendingNativeCall, SequenceKind, Suspension,
    TracebackFrame, Value, Vm, VM_POLL_QUANTUM,
};
use crate::python::error::PyError;
use crate::python::error::PyResult;

impl<'s> Vm<'s> {
    /// Run `code` to completion in a new frame, synchronously.
    pub(super) fn execute_code(
        &mut self,
        code: &CodeRef,
        entry: FrameEntry,
    ) -> Result<Flow, (PyError, super::super::source::Span)> {
        let stack_base = self.stack.len();
        let frame = self
            .enter_frame(code, 0, stack_base, entry, FrameKind::Entry)
            .map_err(|error| (error, super::super::source::Span::default()))?;
        self.run_frame(frame)
    }

    /// Push `frame` and run the dispatch loop until it is gone, for Rust code that needs its
    /// result. Calls it makes, and generators it advances, run on the same loop above it.
    pub(super) fn run_frame(
        &mut self,
        frame: BytecodeFrame,
    ) -> Result<Flow, (PyError, super::super::source::Span)> {
        let depth = self.bytecode_frames.len();
        self.bytecode_frames.push(frame);
        self.run_frames_above(depth)
    }

    /// Run the dispatch loop until the frame at `depth` returns, yields or raises, then pop
    /// whatever is left above `depth`: an entry frame that finished, or, after a process exit,
    /// every frame it called.
    pub(super) fn run_frames_above(
        &mut self,
        depth: usize,
    ) -> Result<Flow, (PyError, super::super::source::Span)> {
        self.synchronous_frames += 1;
        let result = loop {
            match self.execute_active_frame(VM_POLL_QUANTUM) {
                Ok(Flow::Pending) => {}
                // Rust code cannot resume a frame that suspends, so a wait here is a stop, as it
                // is for a native call that would block.
                Ok(Flow::Blocked) => {
                    self.suspension = None;
                    let span = self.active_span();
                    let stop = PyError::unsupported("waiting for input inside a synchronous call");
                    break Err((stop, span));
                }
                result => break result,
            }
        };
        self.synchronous_frames -= 1;
        while self.bytecode_frames.len() > depth {
            let frame = self
                .bytecode_frames
                .pop()
                .expect("the length was checked above");
            self.leave_frame(&frame);
        }
        result
    }

    /// Build a frame for `code` whose shared-stack bases are the current depths: it owns the
    /// local slots above `entry.locals_base` and whatever the shared stacks gain while it
    /// runs. `kind` says who resumes when it returns.
    pub(super) fn enter_frame(
        &mut self,
        code: &CodeRef,
        instruction_pointer: usize,
        stack_base: usize,
        entry: FrameEntry,
        kind: FrameKind,
    ) -> PyResult<BytecodeFrame> {
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
            kind,
            class_body: entry.class_body,
        })
    }

    /// Release what a popped frame owned on the shared stacks.
    pub(super) fn leave_frame(&mut self, frame: &BytecodeFrame) {
        self.stack.truncate(frame.stack_base());
        self.exception_stack.truncate(frame.exception_base as usize);
        if let Some(base) = frame.locals_base() {
            self.locals.truncate(base);
        }
        self.handlers.truncate(frame.handler_base as usize);
        self.with_contexts.truncate(frame.context_base as usize);
    }

    pub(super) fn execute_active_frame(
        &mut self,
        budget: usize,
    ) -> Result<Flow, (PyError, super::super::source::Span)> {
        // Pins made while executing live in this child scope and die with each instruction;
        // only stored references in the VM's roots carry values from one instruction to the next.
        // An instruction that works in place on the operand stack pins nothing at all, so the
        // operand stack, a root, is what keeps its values alive.
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
    ) -> Result<Flow, (PyError, super::super::source::Span)> {
        'execution: for _ in 0..quantum {
            // A resource limit a native or a nested execution hit stops the process before the
            // next instruction, as the per-instruction charge did; the quantum itself cannot
            // exceed the CPU limit.
            if self.interp.resources.is_stopped() {
                return Ok(Flow::Exit(137));
            }
            *executed += 1;
            // Pins and native scratch live for one semantic instruction. Releasing the
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
                let error = super::PyError::exception("Failed", "test timed out");
                self.propagate_error(error, dispatch.span())?;
                let span = dispatch.span();
                dispatch.refresh(self).map_err(|error| (error, span))?;
                continue 'execution;
            }
            let opcode = instruction.opcode;
            let result: PyResult<Flow> = match opcode {
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
                Opcode::ForIterator(target) => match self.for_iterator(instruction_pointer) {
                    Ok(ForIterOutcome::Yielded) => Ok(Flow::Next),
                    Ok(ForIterOutcome::Exhausted) => Ok(Flow::Jump(target)),
                    Ok(ForIterOutcome::Entered) => Ok(Flow::Refresh),
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
                Opcode::Return => match self.pop_ref() {
                    Ok(value) => self.finish_frame(value, false),
                    Err(error) => Err(error),
                },
                Opcode::Yield => match self.pop_ref() {
                    Ok(value) => self.yield_frame(value, instruction_pointer + 1),
                    Err(error) => Err(error),
                },
                Opcode::YieldFromSend(target) => {
                    match self.yield_from_send(instruction_pointer) {
                        // Suspend at this instruction so the next `send` repeats the step.
                        Ok(ForIterOutcome::Yielded) => match self.pop_ref() {
                            Ok(value) => self.yield_frame(value, instruction_pointer),
                            Err(error) => Err(error),
                        },
                        Ok(ForIterOutcome::Exhausted) => Ok(Flow::Jump(target)),
                        Ok(ForIterOutcome::Entered) => Ok(Flow::Refresh),
                        Ok(ForIterOutcome::Blocked(reason)) => {
                            self.active_frame_mut().instruction_pointer = instruction_pointer;
                            Ok(self.suspend(reason, None))
                        }
                        Err(error) => Err(error),
                    }
                }
                Opcode::AwaitResult => self.dispatch_await_result(),
                Opcode::RuntimeError(error) => Err(code.error(error).to_owned().into()),
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
                        .map_err(|e| (PyError::from(e), dispatch.span()))?;
                    Ok(Flow::Next)
                }
                Opcode::MatchException { typed } => self.dispatch_match_exception(typed),
                Opcode::ClearException => match self.exception_stack.pop() {
                    Some(_) => Ok(Flow::Next),
                    None => Err("no active exception".into()),
                },
                Opcode::Reraise => match self.exception_stack.last() {
                    Some(exception) => {
                        let exception = self.value(exception);
                        Err(self.raise_value(exception))
                    }
                    None => Err("no active exception".into()),
                },
                Opcode::Raise(has_value) => self.dispatch_raise(has_value),
                Opcode::RaiseFrom => self.dispatch_raise_from(),
                Opcode::WithEnter => self.dispatch_with_enter(),
                Opcode::WithExit => self.dispatch_with_exit(),
                Opcode::WithExitException => self.dispatch_with_exit_exception(),
                Opcode::AsyncWithExitException => self.dispatch_async_with_exit_exception(),
                Opcode::AsyncWithFinishException => self.dispatch_async_with_finish_exception(),
                Opcode::PopExpression => self.dispatch_pop_expression(),
                Opcode::Halt => self.finish_frame(Ref::from_immediate(Value::None), true),
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
    ) -> PyResult<Flow> {
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
    ) -> PyResult<Flow> {
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
    ) -> PyResult<Flow> {
        // The calling frame resumes past the call whether the callee runs in its own frame,
        // suspends, or returns here; recording that first also lets native code attribute
        // work to this line, as `warnings.warn` does for its caller.
        self.active_frame_mut().instruction_pointer = op_index + 1;
        self.call(positional, keywords, starred, CallMode::Deferred(span))
    }

    /// The value of a call made in [`CallMode::Immediate`], which leaves it on the operand
    /// stack; an exit request propagates as the flow to return from the current arm.
    pub(super) fn immediate_value(&mut self, flow: Flow) -> PyResult<Result<Value, Flow>> {
        match flow {
            Flow::Next => self.pop().map(Ok),
            Flow::Exit(status) => Ok(Err(Flow::Exit(status))),
            flow => unreachable!("an immediate call cannot end with {flow:?}"),
        }
    }

    #[inline(never)]
    fn dispatch_assert(&mut self) -> PyResult<Flow> {
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
        let value = self.allocate_exception("AssertionError", message)?;
        Err(self.raise_value(value))
    }

    /// `except` matching: push whether the innermost handled exception is an instance of the
    /// class or tuple on the stack, or `True` for a bare `except:`.
    fn dispatch_match_exception(&mut self, typed: bool) -> PyResult<Flow> {
        let actual = self.value(self.exception_stack.last().ok_or("no active exception")?);
        let matches = if typed {
            let expected = self.pop()?;
            self.exception_type_matches(expected, actual)?
        } else {
            true
        };
        self.push(Value::Bool(matches));
        Ok(Flow::Next)
    }

    fn dispatch_await_result(&mut self) -> PyResult<Flow> {
        let outcome = self.pop()?;
        if !outcome.is_object() {
            return Err("invalid coroutine scheduler outcome".into());
        }
        let (success, value) = match self.get(outcome)? {
            super::super::heap::Object::Tuple(items) if items.len() == 2 => {
                (self.value(&items[0]), self.value(&items[1]))
            }
            _ => return Err("invalid coroutine scheduler outcome".into()),
        };
        if self.truth_value(&success)? {
            self.push(value);
            return Ok(Flow::Next);
        }
        if exception_types::exception_base(self.state, value)?.is_none() {
            return Err("coroutine scheduler injected a non-exception".into());
        }
        Err(self.raise_value(value))
    }

    #[cold]
    #[inline(never)]
    fn dispatch_raise(&mut self, has_value: bool) -> PyResult<Flow> {
        let exception = if has_value {
            let value = self.pop()?;
            if exception_types::exception_base(self.state, value)?.is_some() {
                value
            } else if let Some(NativeValue::ExceptionType(ExceptionType(kind))) =
                value.native_value()
            {
                self.allocate_exception(kind, String::new())?
            } else if self.exception_class_base(&value)?.is_some() {
                // `raise Cls` raises `Cls()`, running any user `__init__`.
                self.push(value);
                let flow = self.call(0, &[], &[], CallMode::Immediate)?;
                let instance = match self.immediate_value(flow)? {
                    Ok(instance) => instance,
                    Err(flow) => return Ok(flow),
                };
                if exception_types::exception_base(self.state, instance)?.is_none() {
                    return Err("exception class produced a non-exception instance".into());
                }
                instance
            } else {
                return Err("exceptions must derive from BaseException".into());
            }
        } else {
            self.value(
                self.exception_stack
                    .last()
                    .ok_or("No active exception to reraise")?,
            )
        };
        Err(self.raise_value(exception))
    }

    /// `raise exception from cause`. The cause must be an exception, an exception class, or
    /// `None`. Exceptions do not record `__cause__` yet, so a valid cause is checked and then
    /// dropped; an uncaught chained exception prints without its cause.
    #[cold]
    #[inline(never)]
    fn dispatch_raise_from(&mut self) -> PyResult<Flow> {
        let cause = self.pop()?;
        let valid = cause.is_none()
            || exception_types::exception_type_name(self.state, cause)?.is_some()
            || self.exception_class_base(&cause)?.is_some();
        if !valid {
            return Err(PyError::exception(
                "TypeError",
                "exception causes must derive from BaseException",
            ));
        }
        self.dispatch_raise(true)
    }

    #[inline(never)]
    fn dispatch_with_enter(&mut self) -> PyResult<Flow> {
        let context = self.pop()?;
        // CPython looks up `__exit__` first, then `__enter__`.
        for method in ["__exit__", "__enter__"] {
            if self.resolve_attribute(context, method)?.is_none() {
                let message = format!(
                    "'{}' object does not support the context manager protocol (missed {method} \
                     method)",
                    self.type_name_of(&context)?
                );
                return Err(PyError::exception("TypeError", message));
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
    fn dispatch_with_exit(&mut self) -> PyResult<Flow> {
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
    fn dispatch_with_exit_exception(&mut self) -> PyResult<Flow> {
        // The handler pushed the exception; `__exit__` receives it from the exception stack.
        self.pop()?;
        let context = self.with_contexts.pop().ok_or("with stack underflow")?;
        self.execution.stack.push_ref(&context);
        let exception = self.active_exception()?;
        self.load_attribute("__exit__")?;
        let exception_class = self.exception_class(exception)?;
        self.push(exception_class);
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
            Err(self.raise_value(exception))
        }
    }

    fn dispatch_async_with_exit_exception(&mut self) -> PyResult<Flow> {
        let context = self.pop()?;
        self.pop()?;
        let exception = self.active_exception()?;
        self.push(context);
        self.load_attribute("__aexit__")?;
        let exception_class = self.exception_class(exception)?;
        self.push(exception_class);
        self.push(exception);
        self.push(Value::None);
        // The awaited `__aexit__` result stays on the stack for the following await.
        self.call(3, &[], &[false, false, false], CallMode::Immediate)
    }

    /// The innermost handled exception, pinned so it stays valid while `__exit__` runs guest
    /// code.
    fn active_exception(&self) -> PyResult<Value> {
        let exception = self.exception_stack.last().ok_or("no active exception")?;
        Ok(self.value(exception))
    }

    fn exception_class(&self, exception: Value) -> PyResult<Value> {
        // A builtin exception instance has no class object; its class is the registered type.
        if self.instance_class(exception)?.is_none() {
            if let Some(name) = exception_types::exception_base(self.state, exception)? {
                return Ok(Value::Native(NativeValue::ExceptionType(ExceptionType(
                    name,
                ))));
            }
        }
        self.type_of(&exception)
    }

    fn dispatch_async_with_finish_exception(&mut self) -> PyResult<Flow> {
        let suppress = self.pop()?;
        let exception = self.exception_stack.pop().ok_or("no active exception")?;
        let exception = self.value(&exception);
        if self.truth_value(&suppress)? {
            self.pending_exception = None;
            Ok(Flow::Next)
        } else {
            Err(self.raise_value(exception))
        }
    }

    #[inline(never)]
    fn dispatch_pop_expression(&mut self) -> PyResult<Flow> {
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
    fn resume_suspension(&mut self) -> Result<Option<Flow>, (PyError, super::super::source::Span)> {
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

    /// `LoadLocal`: copy a local slot's stored reference onto the operand stack without pinning
    /// it.
    #[inline(always)]
    fn load_fast(&mut self, locals: LocalsLocation, slot: usize) -> PyResult<()> {
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

    /// `StoreLocal`: move the operand stack's top reference into a local slot without pinning
    /// it.
    #[inline(always)]
    fn store_fast(&mut self, locals: LocalsLocation, slot: usize) -> PyResult<()> {
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
    fn delete_local(&mut self, locals: LocalsLocation, slot: usize) -> PyResult<()> {
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

    pub(super) fn active_frame_mut(&mut self) -> &mut BytecodeFrame {
        self.bytecode_frames
            .last_mut()
            .expect("bytecode execution requires an active frame")
    }

    /// Raise `error` if it describes an exception, then unwind frames until a `try` handler
    /// takes the pending exception (`Ok(true)`) or the error reaches Rust code: an entry frame,
    /// or a generator frame Rust code resumed, which closes. A fault is never handled; each
    /// frame it leaves adds its location to the message.
    pub(super) fn propagate_error(
        &mut self,
        error: PyError,
        mut span: super::super::source::Span,
    ) -> Result<bool, (PyError, super::super::source::Span)> {
        // Rebuilt from scratch on every call: a handler found partway through means the
        // in-progress frame list here is irrelevant, and a fully uncaught error overwrites
        // whatever an earlier, since-discarded unwind (e.g. inside a generator sub-frame) left
        // behind. `render_execution` only ever reads the list left by the unwind that actually
        // reaches the top of the program.
        let mut error = self.raise_error(error);
        let mut frames = Vec::new();
        loop {
            if error.is_pending() && self.enter_exception_handler() {
                return Ok(true);
            }
            let module = self.frame_module();
            let unwound = match self.unwind_frame() {
                Ok(unwound) => unwound,
                Err(fault) => return Err((fault, span)),
            };
            let Some((callee, to_rust)) = unwound else {
                frames.push(TracebackFrame {
                    function: None,
                    module,
                    span,
                });
                frames.reverse();
                self.traceback_frames = frames;
                return Err((error, span));
            };
            error = error.located(|| {
                format!(
                    " in {} at line {}, column {}",
                    self.function_name(&callee),
                    span.line,
                    span.column
                )
            });
            frames.push(TracebackFrame {
                function: Some(callee),
                module,
                span,
            });
            if to_rust {
                frames.reverse();
                self.traceback_frames = frames;
                return Err((error, span));
            }
            span = self.call_site_span();
        }
    }

    /// The span of the instruction the active frame stopped at.
    fn active_span(&self) -> super::super::source::Span {
        let frame = self
            .bytecode_frames
            .last()
            .expect("bytecode execution requires an active frame");
        frame
            .code
            .spans
            .get(frame.instruction_pointer)
            .copied()
            .unwrap_or_default()
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

    /// Finish the active frame with `value`, from `return` or, when `halted`, from running off
    /// the end of its code. A called frame pops and leaves `value` on its caller's stack; a
    /// generator's frame pops, finishes its generator and resumes its consumer; an entry frame
    /// stays for [`Self::run_frame`] and the value goes back to it.
    fn finish_frame(&mut self, value: Ref, halted: bool) -> PyResult<Flow> {
        let kind = self
            .bytecode_frames
            .last()
            .expect("bytecode execution requires an active frame")
            .kind;
        match kind {
            FrameKind::Entry if halted => Ok(Flow::Halt),
            FrameKind::Entry => Ok(Flow::Return(value)),
            FrameKind::Call => {
                self.pop_frame();
                self.execution.stack.push_ref(&value);
                Ok(Flow::Refresh)
            }
            FrameKind::Generator(consumer) => {
                // Nothing below allocates before `value` is stored again.
                self.close_generator_frame()?;
                if consumer == Consumer::Rust {
                    self.set_generator_return(&value)?;
                    return Ok(Flow::Return(value));
                }
                // The generator leaves the consumer's stack; `yield from` evaluates to `value`.
                self.execution.stack.pop_ref();
                let target = match self.resuming_instruction()? {
                    Opcode::ForIterator(exit) => exit,
                    Opcode::YieldFromSend(finished) => {
                        self.execution.stack.push_ref(&value);
                        finished
                    }
                    _ => return Err("generator consumer is not iterating".into()),
                };
                self.active_frame_mut().instruction_pointer = target;
                Ok(Flow::Refresh)
            }
        }
    }

    /// Suspend the active frame at `yield value`, to resume at instruction `resume`. A
    /// generator's frame saves itself into its generator and hands `value` to its consumer: a
    /// `for` loop's body, Rust code, or a delegating generator, which suspends at its
    /// `yield from` and yields the value on in turn.
    fn yield_frame(&mut self, value: Ref, resume: usize) -> PyResult<Flow> {
        let mut resume = resume;
        loop {
            let frame = self.active_frame_mut();
            frame.instruction_pointer = resume;
            let FrameKind::Generator(consumer) = frame.kind else {
                // Only a generator's code yields; Rust code running other code reports it.
                return Ok(Flow::Yield(value));
            };
            // Nothing below allocates before `value` is stored again.
            self.suspend_generator_frame()?;
            if consumer == Consumer::Rust {
                return Ok(Flow::Yield(value));
            }
            match self.resuming_instruction()? {
                Opcode::ForIterator(_) => {
                    self.execution.stack.push_ref(&value);
                    return Ok(Flow::Refresh);
                }
                // The delegating frame suspends at its `YieldFromSend`.
                Opcode::YieldFromSend(_) => {
                    resume = self.active_frame_mut().instruction_pointer - 1;
                }
                _ => return Err("generator consumer is not iterating".into()),
            }
        }
    }

    /// The opcode with which the active frame resumed the generator frame just popped from
    /// above it: the instruction before its instruction pointer.
    fn resuming_instruction(&mut self) -> PyResult<Opcode> {
        let frame = self.active_frame_mut();
        frame
            .instruction_pointer
            .checked_sub(1)
            .and_then(|site| frame.code.instructions.get(site))
            .map(|instruction| instruction.opcode)
            .ok_or_else(|| "generator consumer has no resuming instruction".into())
    }

    /// Pop the active frame and release what it owned on the shared stacks.
    pub(super) fn pop_frame(&mut self) {
        let frame = self
            .bytecode_frames
            .pop()
            .expect("bytecode execution requires an active frame");
        self.call_depth = self.call_depth.saturating_sub(1);
        self.leave_frame(&frame);
    }

    /// The outermost scope of the active frame's code, recorded in a traceback frame so the
    /// module's `__file__` can be read if the traceback is printed.
    fn frame_module(&self) -> Option<Ref> {
        let root = scopes::root(&self.state.heap, self.lookup_scope()?).ok()?;
        Some(Ref::from(root))
    }

    /// `__file__` of an imported module, given its scope from [`Self::frame_module`]. Imported
    /// code runs under the module's scope; the main program keeps its globals outside the scope
    /// chain, so its frames find no `__file__`.
    pub(super) fn module_file(&self, module: &Ref) -> Option<String> {
        let heap = &self.state.heap;
        let file = scopes::get(heap, self.value(module), "__file__").ok()??;
        string::string_value(heap, file).ok().flatten()
    }

    /// Pop the active frame as an exception leaves it, finishing a generator's frame for good.
    /// Returns the function the frame ran and whether Rust code, rather than the frame below,
    /// receives the exception next; `None` leaves an entry frame in place for
    /// [`Self::run_frame`].
    fn unwind_frame(&mut self) -> PyResult<Option<(Ref, bool)>> {
        let frame = self
            .bytecode_frames
            .last()
            .expect("bytecode execution requires an active frame");
        let to_rust = match frame.kind {
            FrameKind::Entry => return Ok(None),
            FrameKind::Call => false,
            FrameKind::Generator(consumer) => consumer == Consumer::Rust,
        };
        let callee = frame
            .callee
            .as_ref()
            .map(Ref::dup)
            .ok_or("a called frame has no function")?;
        if let FrameKind::Generator(_) = frame.kind {
            self.close_generator_frame()?;
        } else {
            self.pop_frame();
        }
        Ok(Some((callee, to_rust)))
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
        self.execution.stack.push_ref(&exception);
        self.exception_stack.push(exception);
        self.active_frame_mut().instruction_pointer = target;
        true
    }
}
