//! Call preparation, callable dispatch, argument binding, and Python frame entry.

use super::super::ast::{Program, Statement, StatementKind};
use super::{
    expect_arity, protocol, range_length, BigInt, BinaryOperator, Builtin, BuiltinType,
    BytecodeFrame, CallArgs, CallMode, CallResult, ClassLayout, CodeRef, ComparisonOperator,
    ExceptionType, Execution, FunctionInvocation, FunctionReturn, HashMap, InstanceAttributes,
    InstancePayload, NativeValue, Object, PendingNativeCall, PyError, PyErrorKind, PyRuntime,
    PyStreamRead, RaisedException, ScopeId, Slot, Stream, Value, Vm,
};
use num_traits::{One, Signed, Zero};

impl Vm<'_> {
    /// Length of a builtin representation without consulting Python slots. Native length slots
    /// and the `len()` fallback share this path so their metering and results cannot diverge.
    pub(super) fn physical_length(&self, value: Value) -> Result<Option<usize>, String> {
        let subject = self.builtin_view(value)?;
        if let Some(length) = protocol::string_length(&self.state.heap, &subject)? {
            return Ok(Some(length));
        }
        let Some(id) = subject.object_id() else {
            return Ok(None);
        };
        Ok(match self.state.heap.get(id)? {
            Object::List(values)
            | Object::Tuple(values)
            | Object::Set(values)
            | Object::FrozenSet(values) => Some(values.len()),
            Object::Range { start, stop, step } => Some(range_length(*start, *stop, *step)?),
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => Some(entries.len()),
            _ => None,
        })
    }

    /// The builtin `abs(value)`, through the `__abs__` slot.
    pub(super) fn absolute(&mut self, value: Value) -> Result<Value, String> {
        if let Some(result) = self.invoke_slot(&value, Slot::Absolute, "__abs__", Vec::new())? {
            return Ok(result);
        }
        let message = format!(
            "bad operand type for abs(): '{}'",
            self.type_name_of(&value)?
        );
        Err(self.raise_exception("TypeError", message))
    }

    /// Parse the source string passed to `exec` or `eval` (`builtin`). Parsing is charged
    /// before it runs, and nesting is bounded so dynamic source cannot recurse without limit.
    /// `eval` ignores leading spaces and tabs, as CPython does.
    fn parse_dynamic_source(&mut self, builtin: &str, source: &Value) -> Result<Program, String> {
        let source = protocol::string_value(&self.state.heap, source)?.ok_or_else(|| {
            self.record_native_error(PyError::type_error(format!(
                "{builtin}() arg 1 must be a string, bytes or code object"
            )))
        })?;
        let source = if builtin == "eval" {
            source.trim_start_matches([' ', '\t'])
        } else {
            &source
        };
        if self.bytecode_frames.len() >= 256 {
            return Err(format!("maximum {builtin} depth exceeded"));
        }
        let parse_memory = source
            .len()
            .checked_mul(4)
            .ok_or_else(|| format!("{builtin} source is too large"))?;
        self.charge_cpu(u64::try_from(source.len()).unwrap_or(u64::MAX))?;
        self.reserve_result(parse_memory)?;
        let tokens = super::super::lexer::lex(source).map_err(|error| {
            format!(
                "{} at line {}, column {}",
                error.message, error.span.line, error.span.column
            )
        })?;
        super::super::parser::parse(tokens).map_err(|error| {
            format!(
                "{} at line {}, column {}",
                error.message, error.span.line, error.span.column
            )
        })
    }

    fn pow_integer_argument(&self, value: &Value) -> Result<(BigInt, usize), PyError> {
        let decimal = <Self as PyRuntime>::integer_text(self, value)?.ok_or_else(|| {
            PyError::type_error("pow() 3rd argument not allowed unless all arguments are integers")
        })?;
        let integer = decimal
            .parse::<BigInt>()
            .map_err(|_| PyError::runtime_error("invalid internal integer representation"))?;
        Ok((integer, decimal.len()))
    }

    /// `dir(value)` through a `__dir__` that the value's class defines: the names it returns,
    /// sorted, or `None` when the class defines none.
    fn custom_dir(&mut self, value: &Value) -> Result<Option<Value>, String> {
        let Some(id) = value.object_id() else {
            return Ok(None);
        };
        let Object::Instance { class, .. } = self.state.heap.get(id)? else {
            return Ok(None);
        };
        let class = *class;
        if self.class_attribute(class, "__dir__")?.is_none() {
            return Ok(None);
        }
        let method = self
            .resolve_attribute(*value, "__dir__")?
            .ok_or("__dir__ disappeared during lookup")?;
        let result = self.invoke_value(method, Vec::new())?;
        let mut names = Vec::new();
        for item in self.iterable_values(&result)? {
            let Some(name) = protocol::string_value(&self.state.heap, &item)? else {
                return Err(self.raise_exception("TypeError", "__dir__() must return strings"));
            };
            names.push((name, item));
        }
        self.charge_cpu(u64::try_from(names.len()).unwrap_or(u64::MAX))?;
        names.sort_by(|left, right| left.0.cmp(&right.0));
        let values = names.into_iter().map(|(_, item)| item).collect();
        self.allocate_object(Object::List(values)).map(Some)
    }

    fn dir_names(&self, value: &Value) -> Result<Vec<String>, String> {
        if let Some(NativeValue::Module(module)) = value.native_value() {
            return Ok(module
                .functions
                .iter()
                .map(|function| function.name.to_string())
                .chain(module.values.iter().map(|value| value.name().to_string()))
                .collect());
        }
        let Some(id) = value.object_id() else {
            return Ok(Vec::new());
        };
        match self.state.heap.get(id)? {
            Object::Module { scope, .. } => {
                Ok(self.state.heap.scope_values(*scope)?.into_keys().collect())
            }
            Object::Class {
                attributes, mro, ..
            } => {
                let mut names = attributes.keys().cloned().collect::<Vec<_>>();
                for ancestor in mro {
                    if let Object::Class { attributes, .. } = self.state.heap.get(*ancestor)? {
                        names.extend(attributes.keys().cloned());
                    }
                }
                Ok(names)
            }
            Object::Instance { class, .. } => {
                let mut names = self.state.heap.instance_attribute_names(id)?;
                if let Object::Class {
                    attributes, mro, ..
                } = self.state.heap.get(*class)?
                {
                    names.extend(attributes.keys().cloned());
                    for ancestor in mro {
                        if let Object::Class { attributes, .. } = self.state.heap.get(*ancestor)? {
                            names.extend(attributes.keys().cloned());
                        }
                    }
                }
                Ok(names)
            }
            _ => Ok(Vec::new()),
        }
    }

    pub(super) fn call(
        &mut self,
        positional: usize,
        keyword_names: &[Option<String>],
        starred: &[bool],
        mode: CallMode,
    ) -> Result<CallResult, String> {
        let count = positional
            .checked_add(keyword_names.len())
            .ok_or("too many call arguments")?;
        if starred.len() != count {
            return Err("invalid bytecode call argument metadata".into());
        }
        if self.frame_stack_len() < count + 1 {
            return Err("invalid bytecode stack effect".into());
        }
        let arguments_start = self.stack.len() - count;
        let mut raw_arguments = self.stack.split_off(arguments_start);
        let keyword_values = raw_arguments.split_off(positional);
        let positional_starred = &starred[..positional];
        let mut arguments = Vec::new();
        for (argument, expanded) in raw_arguments.into_iter().zip(positional_starred) {
            if *expanded {
                for value in self.iterable_values(&argument)? {
                    self.push_materialized(&mut arguments, value)?;
                }
            } else {
                self.push_materialized(&mut arguments, argument)?;
            }
        }
        let mut keyword_arguments = Vec::new();
        for ((name, value), expanded) in keyword_names
            .iter()
            .zip(keyword_values)
            .zip(&starred[positional..])
        {
            let additions = match (name, expanded) {
                (Some(name), false) => vec![(name.clone(), value)],
                (None, true) => {
                    let Some(entries) = self.mapping_items(value)? else {
                        let message = format!(
                            "argument after ** must be a mapping, not {}",
                            self.type_name_of(&value)?
                        );
                        return Err(self.raise_exception("TypeError", message));
                    };
                    self.reserve_result(entries.len().saturating_mul(64))?;
                    let mut additions = Vec::with_capacity(entries.len());
                    for (key, value) in entries {
                        let Some(name) = protocol::string_value(&self.state.heap, &key)? else {
                            return Err(
                                self.raise_exception("TypeError", "keywords must be strings")
                            );
                        };
                        self.reserve_result(64usize.saturating_add(name.len()))?;
                        self.charge_cpu(1)?;
                        additions.push((name, value));
                    }
                    additions
                }
                _ => return Err("invalid keyword argument metadata".into()),
            };
            for (name, value) in additions {
                if keyword_arguments
                    .iter()
                    .any(|(existing, _)| existing == &name)
                {
                    let message = format!("got multiple values for keyword argument '{name}'");
                    return Err(self.raise_exception("TypeError", message));
                }
                self.reserve_result(64usize.saturating_add(name.len()))?;
                self.charge_cpu(1)?;
                keyword_arguments.push((name, value));
            }
        }
        let function = self.pop()?;
        if let Some(call) = self.registered_kind(&function).and_then(|kind| kind.call) {
            return call(self, function, CallArgs::new(arguments, keyword_arguments))
                .map(CallResult::Value)
                .map_err(|error| self.record_native_error(error));
        }
        if let Some(id) = function.object_id() {
            return match self.state.heap.get(id)?.clone() {
                Object::Function {
                    name,
                    code,
                    closure,
                    defaults,
                    defining_class,
                    ..
                } => {
                    let method_frame = defining_class.zip(arguments.first().cloned());
                    if let Some((owner, receiver)) = method_frame {
                        self.method_frames.push((owner, receiver));
                    }
                    let result = self.call_python_function(
                        &name,
                        &code,
                        closure,
                        &defaults,
                        FunctionInvocation {
                            arguments,
                            keyword_arguments,
                            mode,
                            pop_method_frame: method_frame.is_some(),
                        },
                    );
                    if method_frame.is_some() && !matches!(result, Ok(CallResult::EnteredFrame)) {
                        self.method_frames.pop();
                    }
                    result
                }
                Object::DescriptorBoundMethod {
                    receiver,
                    descriptor,
                    owner,
                } => {
                    if let Some(function) = descriptor.object_id() {
                        let Object::Function {
                            name,
                            code,
                            closure,
                            defaults,
                            ..
                        } = self.state.heap.get(function)?.clone()
                        else {
                            return Err("bound descriptor is not callable".into());
                        };
                        arguments.insert(0, receiver);
                        if let Some(owner) = owner {
                            self.method_frames.push((owner, receiver));
                        }
                        let result = self.call_python_function(
                            &name,
                            &code,
                            closure,
                            &defaults,
                            FunctionInvocation {
                                arguments,
                                keyword_arguments,
                                mode,
                                pop_method_frame: owner.is_some(),
                            },
                        );
                        if owner.is_some() && !matches!(result, Ok(CallResult::EnteredFrame)) {
                            self.method_frames.pop();
                        }
                        result
                    } else if let Some(NativeValue::SlotWrapper { owner, slot }) =
                        descriptor.native_value()
                    {
                        self.call_slot_wrapper(owner, slot, receiver, arguments, keyword_arguments)
                            .map(CallResult::Value)
                    } else if let Some(NativeValue::NativeMethod(method)) =
                        descriptor.native_value()
                    {
                        let call = CallArgs::new(arguments, keyword_arguments);
                        let retry = match mode {
                            CallMode::Deferred(call_span) => Some(PendingNativeCall::Method {
                                method,
                                receiver,
                                arguments: call.clone(),
                                call_span,
                            }),
                            CallMode::Immediate => None,
                        };
                        let previous_suspend = self.native_suspend_allowed;
                        self.native_suspend_allowed = self.may_suspend(mode);
                        let native_receiver = if receiver.object_id().is_some_and(|id| {
                            matches!(self.state.heap.get(id), Ok(Object::EnumMember { .. }))
                        }) {
                            self.builtin_view(receiver)?
                        } else {
                            receiver
                        };
                        let result = (method.call)(self, native_receiver, call);
                        self.native_suspend_allowed = previous_suspend;
                        match result {
                            Ok(value) => Ok(CallResult::Value(value)),
                            Err(PyError {
                                kind: PyErrorKind::Exit(status),
                                ..
                            }) => Ok(CallResult::Exit(status)),
                            Err(PyError {
                                kind: PyErrorKind::Suspend(reason),
                                ..
                            }) => retry
                                .map(|pending| CallResult::Retry(reason, pending))
                                .ok_or_else(|| {
                                    "native call suspended outside scheduler dispatch".into()
                                }),
                            Err(error) => Err(self.record_native_error(error)),
                        }
                    } else {
                        Err("bound descriptor is not callable".into())
                    }
                }
                Object::GenericAlias { origin, .. } => {
                    self.invoke_call(origin, arguments, keyword_arguments)
                }
                Object::Instance { class, .. } => {
                    let type_id = self.state.heap.type_id(id)?;
                    if self.state.types.slot(type_id, Slot::Call)?.is_none() {
                        return Err(self.raise_object_type_error(&function, "is not callable"));
                    }
                    let (defining_class, descriptor) = self
                        .class_attribute_entry(class, "__call__")?
                        .ok_or("call slot has no descriptor")?;
                    let callable = self.bind_descriptor(
                        descriptor,
                        Some(Value::Object(id)),
                        class,
                        defining_class,
                    )?;
                    self.invoke_call(callable, arguments, keyword_arguments)
                }
                Object::Class { .. } => {
                    self.call_user_class(id, arguments, keyword_arguments, true)
                }
                _ => Err(self.raise_object_type_error(&function, "is not callable")),
            };
        }
        if let Some(NativeValue::BuiltinType(builtin_type)) = function.native_value() {
            return self.call_builtin_type(builtin_type, arguments, keyword_arguments);
        }
        if let Some(NativeValue::ValueKind(kind)) = function.native_value() {
            let call = CallArgs::new(arguments, keyword_arguments);
            return (kind.construct)(self, call)
                .map(CallResult::Value)
                .map_err(|error| self.record_native_error(error));
        }
        if let Some(NativeValue::ExceptionType(exception_type)) = function.native_value() {
            if !keyword_arguments.is_empty() {
                let message = format!("{}() takes no keyword arguments", exception_type.0);
                return Err(self.raise_exception("TypeError", message));
            }
            return Ok(CallResult::Value(self.allocate_object(
                Object::Exception {
                    kind: exception_type.0.to_string(),
                    args: arguments,
                },
            )?));
        }
        if let Some(NativeValue::SlotWrapper { owner, slot }) = function.native_value() {
            if arguments.is_empty() {
                return Err(self.raise_exception("TypeError", "slot wrapper requires a receiver"));
            }
            let receiver = arguments.remove(0);
            return self
                .call_slot_wrapper(owner, slot, receiver, arguments, keyword_arguments)
                .map(CallResult::Value);
        }
        if let Some(NativeValue::NativeMethod(method)) = function.native_value() {
            if arguments.is_empty() {
                return Err("unbound native method requires a receiver".into());
            }
            let receiver = arguments.remove(0);
            let call = CallArgs::new(arguments, keyword_arguments);
            let retry = match mode {
                CallMode::Deferred(call_span) => Some(PendingNativeCall::Method {
                    method,
                    receiver,
                    arguments: call.clone(),
                    call_span,
                }),
                CallMode::Immediate => None,
            };
            let previous_suspend = self.native_suspend_allowed;
            self.native_suspend_allowed = self.may_suspend(mode);
            let result = (method.call)(self, receiver, call);
            self.native_suspend_allowed = previous_suspend;
            return match result {
                Ok(value) => Ok(CallResult::Value(value)),
                Err(PyError {
                    kind: PyErrorKind::Exit(status),
                    ..
                }) => Ok(CallResult::Exit(status)),
                Err(PyError {
                    kind: PyErrorKind::Suspend(reason),
                    ..
                }) => retry
                    .map(|pending| CallResult::Retry(reason, pending))
                    .ok_or_else(|| "native call suspended outside scheduler dispatch".into()),
                Err(error) => Err(self.record_native_error(error)),
            };
        }
        if let Some(NativeValue::NativeFunction(function)) = function.native_value() {
            let call = CallArgs::new(arguments, keyword_arguments);
            let retry = match mode {
                CallMode::Deferred(call_span) => Some(PendingNativeCall::Function {
                    function,
                    arguments: call.clone(),
                    call_span,
                }),
                CallMode::Immediate => None,
            };
            let previous_suspend = self.native_suspend_allowed;
            self.native_suspend_allowed = self.may_suspend(mode);
            let result = (function.call)(self, call);
            self.native_suspend_allowed = previous_suspend;
            return match result {
                Ok(value) => match self.pending_wait.take() {
                    Some(reason) => Ok(CallResult::Blocked(reason, value)),
                    None => Ok(CallResult::Value(value)),
                },
                Err(PyError {
                    kind: PyErrorKind::Exit(status),
                    ..
                }) => Ok(CallResult::Exit(status)),
                Err(PyError {
                    kind: PyErrorKind::Suspend(reason),
                    ..
                }) => retry
                    .map(|pending| CallResult::Retry(reason, pending))
                    .ok_or_else(|| "native call suspended outside scheduler dispatch".into()),
                Err(error) => Err(self.record_native_error(error)),
            };
        }
        let Some(NativeValue::Function(function)) = function.native_value() else {
            return Err(self.raise_object_type_error(&function, "is not callable"));
        };
        if !keyword_arguments.is_empty()
            && !matches!(
                function,
                Builtin::Print
                    | Builtin::Sorted
                    | Builtin::Minimum
                    | Builtin::Maximum
                    | Builtin::Format
            )
        {
            return Err("this builtin does not accept keyword arguments".into());
        }
        match function {
            Builtin::Print => {
                let mut separator = " ".to_string();
                let mut ending = "\n".to_string();
                let mut stream = Stream::Stdout;
                for (name, value) in &keyword_arguments {
                    match name.as_str() {
                        "sep" => {
                            if value.is_none() {
                                continue;
                            }
                            separator = protocol::string_value(&self.state.heap, value)?
                                .ok_or("sep must be None or a string")?;
                        }
                        "end" => {
                            if value.is_none() {
                                continue;
                            }
                            ending = protocol::string_value(&self.state.heap, value)?
                                .ok_or("end must be None or a string")?;
                        }
                        "file" => match value.native_value() {
                            Some(NativeValue::Stream(selected)) => stream = selected,
                            _ if value.is_none() => {}
                            _ => return Err("print file must be a modeled text stream".into()),
                        },
                        "flush" => {}
                        _ => {
                            return Err(format!(
                                "print() got an unexpected keyword argument {name:?}"
                            ))
                        }
                    }
                }
                let mut rendered = Vec::with_capacity(arguments.len());
                for value in &arguments {
                    rendered.push(self.display_value(value)?);
                }
                let text = rendered.join(&separator);
                self.write_output(stream, text.as_bytes());
                self.write_output(stream, ending.as_bytes());
                Ok(CallResult::Value(Value::None))
            }
            Builtin::Input => {
                expect_arity(&arguments, 0, 1)?;
                if let Some(prompt) = arguments.first() {
                    let prompt = self.display_value(prompt)?;
                    self.write_output(Stream::Stdout, prompt.as_bytes());
                }
                let retry = match mode {
                    CallMode::Deferred(call_span) => Some(PendingNativeCall::Input { call_span }),
                    CallMode::Immediate => None,
                };
                let marker = Value::Native(NativeValue::Stream(Stream::Stdin));
                let previous_suspend = self.native_suspend_allowed;
                self.native_suspend_allowed = self.may_suspend(mode);
                let result = self.read_stream(&marker, None, true);
                self.native_suspend_allowed = previous_suspend;
                match result {
                    Ok(read) => self.finish_input(read).map(CallResult::Value),
                    Err(PyError {
                        kind: PyErrorKind::Suspend(reason),
                        ..
                    }) => retry
                        .map(|pending| CallResult::Retry(reason, pending))
                        .ok_or_else(|| "native call suspended outside scheduler dispatch".into()),
                    Err(error) => Err(self.record_native_error(error)),
                }
            }
            Builtin::Exec => {
                expect_arity(&arguments, 1, 1)?;
                let program = self.parse_dynamic_source("exec", &arguments[0])?;
                let code = super::super::compiler::compile(program);
                match self.execute_code(&code) {
                    Ok(Execution::Halt) => Ok(CallResult::Value(Value::None)),
                    Ok(Execution::Exit(status)) => Ok(CallResult::Exit(status)),
                    Ok(
                        Execution::Pending
                        | Execution::Blocked(_)
                        | Execution::Return(_)
                        | Execution::Yield(_, _),
                    ) => Err("exec source did not finish normally".into()),
                    Err((error, span)) => Err(format!(
                        "{error} in exec source at line {}, column {}",
                        span.line, span.column
                    )),
                }
            }
            Builtin::Eval => {
                expect_arity(&arguments, 1, 1)?;
                let mut program = self.parse_dynamic_source("eval", &arguments[0])?;
                let expression = match program.statements.pop() {
                    Some(Statement {
                        kind: StatementKind::Expression(expression),
                        ..
                    }) if program.statements.is_empty() => expression,
                    _ => return Err(self.raise_exception("SyntaxError", "invalid syntax")),
                };
                let code = super::super::compiler::compile_expression(expression);
                match self.execute_code(&code) {
                    Ok(Execution::Return(value)) => Ok(CallResult::Value(value)),
                    Ok(Execution::Exit(status)) => Ok(CallResult::Exit(status)),
                    Ok(
                        Execution::Halt
                        | Execution::Pending
                        | Execution::Blocked(_)
                        | Execution::Yield(_, _),
                    ) => Err("eval source did not finish normally".into()),
                    Err((error, span)) => Err(format!(
                        "{error} in eval source at line {}, column {}",
                        span.line, span.column
                    )),
                }
            }
            Builtin::Exit => {
                expect_arity(&arguments, 0, 1)?;
                let status = arguments
                    .first()
                    .and_then(Value::as_int)
                    .unwrap_or_default();
                Ok(CallResult::Exit(status as i32))
            }
            Builtin::Character => {
                expect_arity(&arguments, 1, 1)?;
                let value = protocol::int_value(&self.state.heap, &arguments[0])
                    .ok_or("an integer is required for chr()")?;
                let Some(codepoint) = u32::try_from(value).ok().and_then(char::from_u32) else {
                    return Err(
                        self.raise_exception("ValueError", "chr() arg not in range(0x110000)")
                    );
                };
                Ok(CallResult::Value(
                    self.allocate_string(codepoint.to_string())?,
                ))
            }
            Builtin::Ordinal => {
                expect_arity(&arguments, 1, 1)?;
                let value = if let Some(text) =
                    protocol::string_value(&self.state.heap, &arguments[0])?
                {
                    let mut characters = text.chars();
                    let character = characters.next().ok_or("ord() expected a character")?;
                    if characters.next().is_some() {
                        return Err("ord() expected a character".into());
                    }
                    u32::from(character) as i64
                } else if let Some(bytes) = <Self as PyRuntime>::bytes_value(self, &arguments[0])
                    .map_err(|error| error.to_string())?
                {
                    if bytes.len() != 1 {
                        return Err("ord() expected a character".into());
                    }
                    i64::from(bytes[0])
                } else {
                    return Err("ord() expected string of length 1".into());
                };
                Ok(CallResult::Value(Value::Int(value)))
            }
            Builtin::Binary | Builtin::Octal | Builtin::Hexadecimal => {
                expect_arity(&arguments, 1, 1)?;
                let decimal = <Self as PyRuntime>::integer_text(self, &arguments[0])
                    .map_err(|error| error.to_string())?
                    .ok_or("integer argument expected")?;
                let integer = decimal
                    .parse::<BigInt>()
                    .map_err(|_| "invalid internal integer representation")?;
                let output_bound = decimal
                    .len()
                    .checked_mul(4)
                    .and_then(|length| length.checked_add(3))
                    .ok_or("integer representation is too large")?;
                <Self as PyRuntime>::reserve_memory(self, output_bound)
                    .map_err(|error| error.to_string())?;
                <Self as PyRuntime>::charge_cpu(
                    self,
                    u64::try_from(output_bound).unwrap_or(u64::MAX),
                )
                .map_err(|error| error.to_string())?;
                let (prefix, digits) = match function {
                    Builtin::Binary => ("0b", format!("{integer:b}")),
                    Builtin::Octal => ("0o", format!("{integer:o}")),
                    Builtin::Hexadecimal => ("0x", format!("{integer:x}")),
                    _ => unreachable!(),
                };
                let rendered = if let Some(digits) = digits.strip_prefix('-') {
                    format!("-{prefix}{digits}")
                } else {
                    format!("{prefix}{digits}")
                };
                Ok(CallResult::Value(self.allocate_string(rendered)?))
            }
            Builtin::Repr => {
                expect_arity(&arguments, 1, 1)?;
                let value = self.repr_value(&arguments[0])?;
                Ok(CallResult::Value(self.allocate_string(value)?))
            }
            Builtin::Format => {
                if !keyword_arguments.is_empty() {
                    let message = "format() takes no keyword arguments".to_string();
                    return Err(self.raise_exception("TypeError", message));
                }
                if !(1..=2).contains(&arguments.len()) {
                    let (bound, count) = if arguments.is_empty() {
                        ("least 1 argument", 0)
                    } else {
                        ("most 2 arguments", arguments.len())
                    };
                    let message = format!("format expected at {bound}, got {count}");
                    return Err(self.raise_exception("TypeError", message));
                }
                let spec = match arguments.get(1) {
                    Some(spec) => {
                        protocol::string_value(&self.state.heap, spec)?.ok_or_else(|| {
                            let message = format!(
                                "format() argument 2 must be str, not {}",
                                self.type_name_of(spec).unwrap_or_default()
                            );
                            self.raise_exception("TypeError", message)
                        })?
                    }
                    None => String::new(),
                };
                self.reserve_format_spec(&spec)?;
                let value = self.format_object(&arguments[0], &spec)?;
                Ok(CallResult::Value(self.allocate_string(value)?))
            }
            Builtin::Hash => {
                expect_arity(&arguments, 1, 1)?;
                let hash = self.hash_value(&arguments[0])?;
                Ok(CallResult::Value(Value::Int(hash)))
            }
            Builtin::Dir => {
                expect_arity(&arguments, 0, 1)?;
                if let Some(value) = arguments.first() {
                    if let Some(names) = self.custom_dir(value)? {
                        return Ok(CallResult::Value(names));
                    }
                }
                let mut names = if let Some(value) = arguments.first() {
                    self.dir_names(value)?
                } else {
                    self.state
                        .globals
                        .values
                        .iter()
                        .enumerate()
                        .filter_map(|(index, value)| {
                            value.and_then(|_| {
                                let symbol = super::super::heap::SymbolId::from_index(index)?;
                                self.state.heap.symbol_name(symbol).map(str::to_string)
                            })
                        })
                        .collect()
                };
                let name_bytes = names.iter().try_fold(0usize, |total, name| {
                    total
                        .checked_add(name.len())
                        .ok_or("dir() result is too large")
                })?;
                self.reserve_result(name_bytes)?;
                self.charge_cpu(u64::try_from(names.len()).unwrap_or(u64::MAX))?;
                names.sort();
                names.dedup();
                let values = names
                    .into_iter()
                    .map(|name| self.allocate_string(name))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(CallResult::Value(
                    self.allocate_object(Object::List(values))?,
                ))
            }
            Builtin::IsInstance => {
                expect_arity(&arguments, 2, 2)?;
                Ok(CallResult::Value(Value::Bool(
                    self.is_instance(&arguments[0], &arguments[1])?,
                )))
            }
            Builtin::IsSubclass => {
                expect_arity(&arguments, 2, 2)?;
                Ok(CallResult::Value(Value::Bool(
                    self.is_subclass(&arguments[0], &arguments[1])?,
                )))
            }
            Builtin::Length => {
                expect_arity(&arguments, 1, 1)?;
                if let Some(value) =
                    self.invoke_slot(&arguments[0], Slot::Length, "__len__", Vec::new())?
                {
                    let length = protocol::int_value(&self.state.heap, &value)
                        .ok_or("__len__() should return an integer")?;
                    if length < 0 {
                        return Err("__len__() should return >= 0".into());
                    }
                    return Ok(CallResult::Value(Value::Int(length)));
                }
                let Some(length) = self.physical_length(arguments[0])? else {
                    let message = format!(
                        "object of type '{}' has no len()",
                        self.type_name_of(&arguments[0])?
                    );
                    return Err(self.raise_exception("TypeError", message));
                };
                let length = i64::try_from(length)
                    .map_err(|_| self.raise_exception("OverflowError", "length is too large"))?;
                Ok(CallResult::Value(Value::Int(length)))
            }
            Builtin::Sorted => {
                expect_arity(&arguments, 1, 1)?;
                let values = self.iterable_values(&arguments[0])?;
                let mut key_function = None;
                let mut reverse = false;
                let mut saw_reverse = false;
                for (name, value) in keyword_arguments {
                    match name.as_str() {
                        "key" if key_function.is_none() => key_function = Some(value),
                        "reverse" if !saw_reverse => {
                            reverse = self.truth_value(&value)?;
                            saw_reverse = true;
                        }
                        "key" | "reverse" => {
                            return Err(format!(
                                "sorted() got multiple values for keyword {name:?}"
                            ))
                        }
                        _ => return Err(format!("sorted() got an unexpected keyword {name:?}")),
                    }
                }
                let mut keyed = Vec::new();
                for value in values {
                    let key = if let Some(function) = &key_function {
                        self.stack.push(*function);
                        self.stack.push(value);
                        match self.call(1, &[], &[false], CallMode::Immediate)? {
                            CallResult::Value(key) => key,
                            CallResult::Exit(status) => return Ok(CallResult::Exit(status)),
                            CallResult::EnteredFrame => {
                                unreachable!("immediate call entered a frame")
                            }
                            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                                unreachable!("immediate call cannot suspend")
                            }
                        }
                    } else {
                        value
                    };
                    self.reserve_result(64)?;
                    keyed.push((key, value));
                }
                // Stable insertion sort keeps comparison dispatch and failure order obvious. Like
                // CPython's sort it only asks `<`, and a reversed sort keeps equal items in order.
                for index in 1..keyed.len() {
                    let mut current = index;
                    while current > 0 {
                        self.charge_cpu(1)?;
                        let (left, right) = if reverse {
                            (keyed[current - 1].0, keyed[current].0)
                        } else {
                            (keyed[current].0, keyed[current - 1].0)
                        };
                        if !self.compare_truth(ComparisonOperator::Less, &left, &right)? {
                            break;
                        }
                        keyed.swap(current, current - 1);
                        current -= 1;
                    }
                }
                let values = keyed.into_iter().map(|(_, value)| value).collect();
                Ok(CallResult::Value(
                    self.allocate_object(Object::List(values))?,
                ))
            }
            Builtin::Minimum | Builtin::Maximum => {
                let (name, operator) = match function {
                    Builtin::Minimum => ("min", ComparisonOperator::Less),
                    _ => ("max", ComparisonOperator::Greater),
                };
                let mut key_function = None;
                let mut default = None;
                for (keyword, value) in keyword_arguments {
                    match keyword.as_str() {
                        "key" => key_function = Some(value).filter(|value| !value.is_none()),
                        "default" => default = Some(value),
                        _ => {
                            let message =
                                format!("{name}() got an unexpected keyword argument '{keyword}'");
                            return Err(self.raise_exception("TypeError", message));
                        }
                    }
                }
                let values = match arguments.len() {
                    0 => {
                        let message = format!("{name} expected at least 1 argument, got 0");
                        return Err(self.raise_exception("TypeError", message));
                    }
                    1 => self.iterable_values(&arguments[0])?,
                    _ if default.is_some() => {
                        let message = format!(
                            "Cannot specify a default for {name}() with multiple positional \
                             arguments"
                        );
                        return Err(self.raise_exception("TypeError", message));
                    }
                    _ => arguments,
                };
                // CPython keeps the first of equal items: only a strictly smaller (for `min`) or
                // larger (for `max`) key replaces the selection.
                let mut selected: Option<(Value, Value)> = None;
                for value in values {
                    self.charge_cpu(1)?;
                    let key = match key_function {
                        Some(function) => {
                            self.stack.push(function);
                            self.stack.push(value);
                            match self.call(1, &[], &[false], CallMode::Immediate)? {
                                CallResult::Value(key) => key,
                                CallResult::Exit(status) => return Ok(CallResult::Exit(status)),
                                CallResult::EnteredFrame => {
                                    unreachable!("immediate call entered a frame")
                                }
                                CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                                    unreachable!("immediate call cannot suspend")
                                }
                            }
                        }
                        None => value,
                    };
                    let replace = match &selected {
                        None => true,
                        Some((selected_key, _)) => {
                            self.compare_truth(operator, &key, &selected_key.clone())?
                        }
                    };
                    if replace {
                        selected = Some((key, value));
                    }
                }
                match (selected, default) {
                    (Some((_, value)), _) => Ok(CallResult::Value(value)),
                    (None, Some(default)) => Ok(CallResult::Value(default)),
                    (None, None) => {
                        let message = format!("{name}() iterable argument is empty");
                        Err(self.raise_exception("ValueError", message))
                    }
                }
            }
            Builtin::Sum => {
                expect_arity(&arguments, 1, 2)?;
                let values = self.iterable_values(&arguments[0])?;
                let start = arguments.get(1).copied().unwrap_or(Value::Int(0));
                Ok(CallResult::Value(self.builtin_sum(values, start)?))
            }
            Builtin::Absolute => {
                expect_arity(&arguments, 1, 1)?;
                Ok(CallResult::Value(self.absolute(arguments[0])?))
            }
            Builtin::Power => {
                expect_arity(&arguments, 2, 3)?;
                if arguments.len() == 3 && arguments[2] != Value::None {
                    let parsed = self.pow_integer_argument(&arguments[0]);
                    let (base, base_len) =
                        parsed.map_err(|error| self.record_native_error(error))?;
                    let parsed = self.pow_integer_argument(&arguments[1]);
                    let (exponent, exponent_len) =
                        parsed.map_err(|error| self.record_native_error(error))?;
                    let parsed = self.pow_integer_argument(&arguments[2]);
                    let (modulus, modulus_len) =
                        parsed.map_err(|error| self.record_native_error(error))?;
                    if modulus.is_zero() {
                        let error = PyError::zero_division_error("pow() 3rd argument cannot be 0");
                        return Err(self.record_native_error(error));
                    }
                    let positive_modulus = modulus.abs();
                    // As in CPython, `pow(b, -e, m)` is `pow(inverse(b), e, m)`, and every
                    // value is its own inverse modulo one.
                    let (base, exponent) = if !exponent.is_negative() {
                        (base, exponent)
                    } else if positive_modulus.is_one() {
                        (BigInt::zero(), -exponent)
                    } else {
                        let Some(inverse) = base.modinv(&positive_modulus) else {
                            let message = "base is not invertible for the given modulus";
                            return Err(self.record_native_error(PyError::value_error(message)));
                        };
                        (inverse, -exponent)
                    };
                    let work = base_len
                        .saturating_add(modulus_len)
                        .saturating_mul(exponent_len.saturating_mul(4).max(1));
                    <Self as PyRuntime>::charge_cpu(self, u64::try_from(work).unwrap_or(u64::MAX))
                        .map_err(|error| error.to_string())?;
                    <Self as PyRuntime>::reserve_memory(self, modulus_len.saturating_mul(4).max(1))
                        .map_err(|error| error.to_string())?;
                    let mut result = base.modpow(&exponent, &positive_modulus);
                    if modulus.is_negative() && !result.is_zero() {
                        result += modulus;
                    }
                    let result = <Self as PyRuntime>::new_integer(self, &result.to_string())
                        .map_err(|error| error.to_string())?;
                    return Ok(CallResult::Value(result));
                }
                Ok(CallResult::Value(self.binary_value(
                    BinaryOperator::Power,
                    arguments[0],
                    arguments[1],
                )?))
            }
            Builtin::Divmod => {
                expect_arity(&arguments, 2, 2)?;
                Ok(CallResult::Value(
                    self.divmod_value(arguments[0], arguments[1])?,
                ))
            }
            Builtin::SetAttribute => {
                if arguments.len() != 3 {
                    let message = format!("setattr expected 3 arguments, got {}", arguments.len());
                    return Err(self.raise_exception("TypeError", message));
                }
                let Some(name) = protocol::string_value(&self.state.heap, &arguments[1])? else {
                    let message = format!(
                        "attribute name must be string, not '{}'",
                        self.type_name_of(&arguments[1])?
                    );
                    return Err(self.raise_exception("TypeError", message));
                };
                // `setattr(owner, name, value)` is `owner.name = value` with a computed name.
                let symbol = self
                    .state
                    .heap
                    .intern_symbol(&name, &mut self.interp.resources)?;
                self.store_attribute_by_symbol(arguments[0], symbol, &name, arguments[2])?;
                Ok(CallResult::Value(Value::None))
            }
            Builtin::DeleteAttribute => {
                if arguments.len() != 2 {
                    let message = format!("delattr expected 2 arguments, got {}", arguments.len());
                    return Err(self.raise_exception("TypeError", message));
                }
                let Some(name) = protocol::string_value(&self.state.heap, &arguments[1])? else {
                    let message = format!(
                        "attribute name must be string, not '{}'",
                        self.type_name_of(&arguments[1])?
                    );
                    return Err(self.raise_exception("TypeError", message));
                };
                let symbol = self
                    .state
                    .heap
                    .intern_symbol(&name, &mut self.interp.resources)?;
                self.delete_attribute_by_symbol(arguments[0], symbol, &name)?;
                Ok(CallResult::Value(Value::None))
            }
            Builtin::Callable => {
                expect_arity(&arguments, 1, 1)?;
                let callable = <Self as PyRuntime>::is_callable(self, &arguments[0])
                    .map_err(|error| error.to_string())?;
                Ok(CallResult::Value(Value::Bool(callable)))
            }
            Builtin::Iter => {
                expect_arity(&arguments, 1, 2)?;
                if arguments.len() == 2 {
                    if !<Self as PyRuntime>::is_callable(self, &arguments[0])
                        .map_err(|error| error.to_string())?
                    {
                        return Err("iter(v, w): v must be callable".into());
                    }
                    return Ok(CallResult::Value(self.allocate_object(
                        Object::CallableIterator {
                            callable: arguments[0],
                            sentinel: arguments[1],
                            exhausted: false,
                        },
                    )?));
                }
                Ok(CallResult::Value(self.make_iterator(arguments[0])?))
            }
            Builtin::Next => {
                expect_arity(&arguments, 1, 2)?;
                let value = match self.iterator_next(&arguments[0]) {
                    Ok(value) => value,
                    Err(_) if arguments.len() == 2 && self.pending_stop_iteration() => {
                        self.pending_exception = None;
                        None
                    }
                    Err(error) => return Err(error),
                };
                let value = match value {
                    Some(value) => value,
                    None => match arguments.get(1) {
                        Some(default) => *default,
                        None => return Err(self.raise_stop_iteration(&arguments[0])),
                    },
                };
                Ok(CallResult::Value(value))
            }
            Builtin::Enumerate => {
                expect_arity(&arguments, 1, 2)?;
                let values = self.iterable_values(&arguments[0])?;
                let start = match arguments.get(1) {
                    Some(value) => self.index_argument(value)?,
                    None => 0,
                };
                let mut result = Vec::new();
                for (offset, value) in values.into_iter().enumerate() {
                    self.reserve_result(64)?;
                    self.charge_cpu(1)?;
                    let offset = i64::try_from(offset).map_err(|_| "enumerate is too large")?;
                    let index = start
                        .checked_add(offset)
                        .ok_or("enumerate index exceeds the bounded integer range")?;
                    result
                        .push(self.allocate_object(Object::Tuple(vec![Value::Int(index), value]))?);
                }
                Ok(CallResult::Value(
                    self.allocate_object(Object::List(result))?,
                ))
            }
            Builtin::Zip => {
                let sequences = arguments
                    .iter()
                    .map(|value| self.iterable_values(value))
                    .collect::<Result<Vec<_>, _>>()?;
                let length = sequences.iter().map(Vec::len).min().unwrap_or(0);
                let mut result = Vec::new();
                for index in 0..length {
                    self.reserve_result(64)?;
                    self.charge_cpu(1)?;
                    let tuple = sequences.iter().map(|values| values[index]).collect();
                    result.push(self.allocate_object(Object::Tuple(tuple))?);
                }
                Ok(CallResult::Value(
                    self.allocate_object(Object::List(result))?,
                ))
            }
            Builtin::Any | Builtin::All => {
                expect_arity(&arguments, 1, 1)?;
                let values = self.iterable_values(&arguments[0])?;
                let mut result = matches!(function, Builtin::All);
                for value in values {
                    self.charge_cpu(1)?;
                    let truth = self.truth_value(&value)?;
                    if matches!(function, Builtin::Any) && truth {
                        result = true;
                        break;
                    }
                    if matches!(function, Builtin::All) && !truth {
                        result = false;
                        break;
                    }
                }
                Ok(CallResult::Value(Value::Bool(result)))
            }
            Builtin::Property => {
                expect_arity(&arguments, 1, 1)?;
                Ok(CallResult::Value(self.allocate_object(
                    Object::Property {
                        getter: arguments[0],
                        setter: None,
                    },
                )?))
            }
            Builtin::StaticMethod => {
                expect_arity(&arguments, 1, 1)?;
                Ok(CallResult::Value(self.allocate_object(
                    Object::StaticMethod {
                        callable: arguments[0],
                    },
                )?))
            }
            Builtin::ClassMethod => {
                expect_arity(&arguments, 1, 1)?;
                Ok(CallResult::Value(self.allocate_object(
                    Object::ClassMethod {
                        callable: arguments[0],
                    },
                )?))
            }
            Builtin::Super => {
                expect_arity(&arguments, 0, 2)?;
                let (start_class, receiver) = match arguments.as_slice() {
                    [] => self
                        .method_frames
                        .last()
                        .cloned()
                        .ok_or("super(): no current method context")?,
                    [start_class, receiver]
                        if start_class.object_id().is_some_and(|id| {
                            matches!(self.state.heap.get(id), Ok(Object::Class { .. }))
                        }) =>
                    {
                        (start_class.object_id().unwrap(), *receiver)
                    }
                    _ => return Err("super() expects a class and instance".into()),
                };
                Ok(CallResult::Value(self.allocate_object(Object::Super {
                    start_class,
                    receiver,
                })?))
            }
            Builtin::Globals => {
                expect_arity(&arguments, 0, 0)?;
                let target = self.current_globals_target()?;
                Ok(CallResult::Value(
                    self.allocate_object(Object::NamespaceDict(target))?,
                ))
            }
            Builtin::Locals => {
                expect_arity(&arguments, 0, 0)?;
                Ok(CallResult::Value(self.current_locals()?))
            }
            Builtin::Vars => {
                expect_arity(&arguments, 0, 1)?;
                let Some(owner) = arguments.first() else {
                    return Ok(CallResult::Value(self.current_locals()?));
                };
                // `vars(obj)` is `obj.__dict__`: a namespace view, or a class's or native
                // module's read-only proxy.
                let Some(namespace) = self.resolve_optional_attribute(*owner, "__dict__")? else {
                    return Err(self.raise_exception(
                        "TypeError",
                        "vars() argument must have __dict__ attribute",
                    ));
                };
                Ok(CallResult::Value(namespace))
            }
        }
    }

    /// Explicit `type.__call__` skips the metaclass override while retaining ordinary class
    /// construction and the native type constructors.
    pub(super) fn call_type_default(
        &mut self,
        class: Value,
        args: CallArgs,
    ) -> Result<CallResult, String> {
        let (arguments, keyword_arguments) = args.into_parts();
        if let Some(id) = class.object_id() {
            if matches!(self.state.heap.get(id)?, Object::Class { .. }) {
                return self.call_user_class(id, arguments, keyword_arguments, false);
            }
        }
        if matches!(
            class.native_value(),
            Some(
                NativeValue::BuiltinType(_)
                    | NativeValue::ValueKind(_)
                    | NativeValue::ExceptionType(_)
            )
        ) {
            return self.invoke_call(class, arguments, keyword_arguments);
        }
        Err(self.raise_exception("TypeError", "type.__call__ requires a class"))
    }

    /// Apply the default class constructor after any metaclass `__call__` override has had
    /// its turn. Direct `type.__call__` enters here with metaclass dispatch disabled.
    fn call_user_class(
        &mut self,
        id: super::super::heap::ObjectId,
        mut arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
        dispatch_metaclass: bool,
    ) -> Result<CallResult, String> {
        let Object::Class {
            name,
            metaclass,
            layout,
            exception_base,
            is_dataclass,
            dataclass_fields,
            enum_members,
            ..
        } = self.state.heap.get(id)?.clone()
        else {
            return Err("type.__call__ requires a class".into());
        };
        if dispatch_metaclass {
            if let Some(metaclass_id) = metaclass.object_id() {
                if let Some((owner, descriptor)) =
                    self.class_attribute_entry(metaclass_id, "__call__")?
                {
                    let callable = self.bind_descriptor(
                        descriptor,
                        Some(Value::Object(id)),
                        metaclass_id,
                        owner,
                    )?;
                    return self.invoke_call(callable, arguments, keyword_arguments);
                }
            }
        }
        if !enum_members.is_empty() {
            if !keyword_arguments.is_empty() || arguments.len() != 1 {
                return Err(format!("{name}() expects one value"));
            }
            for member in enum_members {
                let Object::EnumMember { value, .. } = self
                    .state
                    .heap
                    .get(member.object_id().ok_or("invalid enum member")?)?
                else {
                    return Err("invalid enum member".into());
                };
                let value = *value;
                if self.values_equal(&value, &arguments[0])? {
                    return Ok(CallResult::Value(member));
                }
            }
            return Err(format!("value is not a valid {name}"));
        }
        if layout != ClassLayout::Type && exception_base.is_none() && !is_dataclass {
            let class_type = self
                .class_type_id(&Value::Object(id))?
                .ok_or("class has no registered type")?;
            let use_type_constructor = layout == ClassLayout::Object;
            if use_type_constructor || self.class_attribute_entry(id, "__new__")?.is_some() {
                let (owner, constructor) = self
                    .type_lookup(class_type, "__new__")?
                    .ok_or("class constructor has no descriptor")?;
                return self.construct_with_new(
                    id,
                    class_type,
                    owner,
                    constructor,
                    arguments,
                    keyword_arguments,
                );
            }
        }
        let payload = match layout {
            ClassLayout::Object => InstancePayload::Object,
            // Mutable builtin `__new__` creates empty storage. An override can then
            // populate it, usually through the base `__init__`.
            ClassLayout::Builtin(
                builtin @ (BuiltinType::List
                | BuiltinType::Set
                | BuiltinType::Dict
                | BuiltinType::ByteArray),
            ) if self.class_attribute(id, "__init__")?.is_some() => {
                InstancePayload::Builtin(self.builtin_value(builtin, Vec::new(), Vec::new())?)
            }
            ClassLayout::Builtin(builtin) => InstancePayload::Builtin(self.builtin_value(
                builtin,
                arguments.clone(),
                keyword_arguments.clone(),
            )?),
            ClassLayout::Type => {
                let created = if let Some((owner, constructor)) =
                    self.class_attribute_entry(id, "__new__")?
                {
                    let constructor =
                        self.bind_descriptor(constructor, Some(Value::Object(id)), id, owner)?;
                    match self.invoke_call(
                        constructor,
                        arguments.clone(),
                        keyword_arguments.clone(),
                    )? {
                        CallResult::Value(value) => value,
                        CallResult::Exit(status) => return Ok(CallResult::Exit(status)),
                        CallResult::EnteredFrame => {
                            unreachable!("invoke_call is immediate")
                        }
                        CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                            unreachable!("immediate call cannot suspend")
                        }
                    }
                } else {
                    if !keyword_arguments.is_empty() || arguments.len() != 3 {
                        return Err("type construction expects name, bases, and namespace".into());
                    }
                    let name = protocol::string_value(&self.state.heap, &arguments[0])?
                        .ok_or("type name must be a string")?;
                    self.new_type(Value::Object(id), name, arguments[1], arguments[2])
                        .map_err(|error| error.to_string())?
                };
                if created.object_id().is_some_and(|created_id| {
                    matches!(self.state.heap.get(created_id), Ok(Object::Class { .. }))
                }) {
                    if let Some((owner, initializer)) =
                        self.class_attribute_entry(id, "__init__")?
                    {
                        let initializer =
                            self.bind_descriptor(initializer, Some(created), id, owner)?;
                        let result =
                            match self.invoke_call(initializer, arguments, keyword_arguments)? {
                                CallResult::Value(value) => value,
                                CallResult::Exit(status) => return Ok(CallResult::Exit(status)),
                                CallResult::EnteredFrame => {
                                    unreachable!("invoke_call is immediate")
                                }
                                CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                                    unreachable!("immediate call cannot suspend")
                                }
                            };
                        if !result.is_none() {
                            return Err("metaclass __init__() should return None".into());
                        }
                    }
                }
                return Ok(CallResult::Value(created));
            }
        };
        let instance = self.allocate_object(Object::Instance {
            class: id,
            payload,
            attributes: InstanceAttributes::default(),
        })?;
        if exception_base.is_some() {
            let exception_args = self.allocate_object(Object::Tuple(arguments.clone()))?;
            self.state.heap.insert_attribute(
                instance.object_id().expect("instances are heap objects"),
                "args".into(),
                exception_args,
                &mut self.interp.resources,
            )?;
        }
        if is_dataclass {
            let mut values = Vec::new();
            for (index, (field, default)) in dataclass_fields.iter().enumerate() {
                if index < arguments.len()
                    && keyword_arguments.iter().any(|(name, _)| name == field)
                {
                    return Err(format!(
                        "{name}() got multiple values for argument {field:?}"
                    ));
                }
                let value = keyword_arguments
                    .iter()
                    .find(|(name, _)| name == field)
                    .map(|(_, value)| *value)
                    .or_else(|| arguments.get(index).cloned())
                    .or(*default)
                    .ok_or_else(|| format!("{name}() missing required argument: {field:?}"))?;
                if keyword_arguments
                    .iter()
                    .filter(|(name, _)| name == field)
                    .count()
                    > 1
                {
                    return Err(format!(
                        "{name}() got multiple values for argument {field:?}"
                    ));
                }
                values.push((field.clone(), value));
            }
            if arguments.len() > dataclass_fields.len() {
                return Err(format!(
                    "{name}() takes {} positional arguments but {} were given",
                    dataclass_fields.len(),
                    arguments.len()
                ));
            }
            for (field, _) in &keyword_arguments {
                if !dataclass_fields.iter().any(|(name, _)| name == field) {
                    return Err(format!(
                        "{name}() got an unexpected keyword argument {field:?}"
                    ));
                }
            }
            self.state.heap.extend_attributes(
                instance.object_id().expect("instances are heap objects"),
                values,
                &mut self.interp.resources,
            )?;
        } else if let Some(initializer) = self.class_attribute(id, "__init__")? {
            let Some(function) = initializer.object_id() else {
                return Err(format!("{name}.__init__ is not callable"));
            };
            let Object::Function {
                name: function_name,
                code,
                closure,
                defaults,
                defining_class,
                ..
            } = self.state.heap.get(function)?.clone()
            else {
                return Err(format!("{name}.__init__ is not a function"));
            };
            arguments.insert(0, instance);
            // Zero-argument `super()` in the initializer reads this frame.
            if let Some(owner) = defining_class {
                self.method_frames.push((owner, instance));
            }
            let result = self.call_python_function(
                &function_name,
                &code,
                closure,
                &defaults,
                FunctionInvocation {
                    arguments,
                    keyword_arguments,
                    mode: CallMode::Immediate,
                    pop_method_frame: false,
                },
            );
            if defining_class.is_some() {
                self.method_frames.pop();
            }
            match result? {
                CallResult::Value(value) if value.is_none() => {}
                CallResult::Value(_) => return Err("__init__() should return None".into()),
                CallResult::Exit(status) => return Ok(CallResult::Exit(status)),
                CallResult::EnteredFrame => {
                    unreachable!("immediate initializer entered a frame")
                }
                CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                    unreachable!("immediate initializer cannot suspend")
                }
            }
        } else if exception_base.is_some() && !keyword_arguments.is_empty() {
            return Err(format!("{name}() does not accept keyword arguments"));
        } else if layout == ClassLayout::Object
            && exception_base.is_none()
            && (!arguments.is_empty() || !keyword_arguments.is_empty())
        {
            return Err(format!("{name}() takes no arguments"));
        }
        Ok(CallResult::Value(instance))
    }

    fn call_python_function(
        &mut self,
        name: &str,
        code: &CodeRef,
        closure: Option<ScopeId>,
        defaults: &[Value],
        invocation: FunctionInvocation,
    ) -> Result<CallResult, String> {
        let FunctionInvocation {
            arguments,
            keyword_arguments,
            mode,
            pop_method_frame,
        } = invocation;
        if code.call_signature.is_generator || code.call_signature.is_coroutine {
            return self.create_generator(
                name,
                code,
                closure,
                defaults,
                arguments,
                keyword_arguments,
            );
        }
        const MAX_CALL_DEPTH: usize = 256;
        if self.call_depth == MAX_CALL_DEPTH {
            return Err(self.raise_exception("RecursionError", "maximum recursion depth exceeded"));
        }
        let scope =
            self.bind_function_scope(name, code, closure, defaults, arguments, keyword_arguments)?;
        self.local_scopes.push(scope);
        self.call_depth += 1;
        if let CallMode::Deferred(call_span) = mode {
            let stack_base = self.stack.len();
            self.bytecode_frames.push(BytecodeFrame {
                code: code.clone(),
                instruction_pointer: 0,
                stack_base,
                handlers: Vec::new(),
                function_return: Some(FunctionReturn {
                    name: name.to_string(),
                    call_span,
                    pop_method_frame,
                }),
                pending_native_call: None,
            });
            return Ok(CallResult::EnteredFrame);
        }
        let result = self.execute_code(code);
        self.call_depth -= 1;
        self.local_scopes.pop();
        match result {
            Ok(Execution::Pending) => unreachable!("execute_code drains pending quanta"),
            Ok(Execution::Blocked(_)) => unreachable!("immediate call cannot suspend"),
            Ok(Execution::Return(value)) => Ok(CallResult::Value(value)),
            Ok(Execution::Halt) => Ok(CallResult::Value(Value::None)),
            Ok(Execution::Yield(_, _)) => {
                Err(format!("unexpected yield in ordinary function {name}"))
            }
            Ok(Execution::Exit(status)) => Ok(CallResult::Exit(status)),
            Err((error, span)) => Err(format!(
                "{error} in {name} at line {}, column {}",
                span.line, span.column
            )),
        }
    }

    fn create_generator(
        &mut self,
        name: &str,
        code: &CodeRef,
        closure: Option<ScopeId>,
        defaults: &[Value],
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<CallResult, String> {
        let scope =
            self.bind_function_scope(name, code, closure, defaults, arguments, keyword_arguments)?;
        let generator = self.allocate_object(Object::Generator {
            name: name.to_string(),
            code: code.clone(),
            scope,
            instruction_pointer: 0,
            handlers: Vec::new(),
            exceptions: Vec::new(),
            stack: Vec::new(),
            exhausted: false,
            running: false,
            return_value: Value::None,
        })?;
        Ok(CallResult::Value(generator))
    }

    /// Bind one invocation directly into the compiler's local-slot layout.
    fn bind_function_scope(
        &mut self,
        name: &str,
        code: &CodeRef,
        closure: Option<ScopeId>,
        defaults: &[Value],
        mut arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<ScopeId, String> {
        let signature = &code.call_signature;
        if signature.variadic_slot.is_none() && arguments.len() > signature.positional_count {
            let required = code.parameters[..signature.positional_count]
                .iter()
                .filter(|parameter| !parameter.has_default)
                .count();
            let takes = match (required, signature.positional_count) {
                (required, count) if required == count => format!(
                    "{count} positional argument{}",
                    if count == 1 { "" } else { "s" }
                ),
                (required, count) => format!("from {required} to {count} positional arguments"),
            };
            let given = arguments.len();
            let keyword_only = keyword_arguments
                .iter()
                .filter(|(keyword, _)| {
                    code.parameters.iter().any(|parameter| {
                        parameter.name == *keyword
                            && parameter.kind == super::super::bytecode::ParameterKind::KeywordOnly
                    })
                })
                .count();
            let keyword_detail = if keyword_only == 0 {
                String::new()
            } else {
                format!(
                    " positional argument{} (and {keyword_only} keyword-only argument{})",
                    if given == 1 { "" } else { "s" },
                    if keyword_only == 1 { "" } else { "s" }
                )
            };
            let verb = if given == 1 && keyword_only == 0 {
                "was"
            } else {
                "were"
            };
            let message =
                format!("{name}() takes {takes} but {given}{keyword_detail} {verb} given");
            return Err(self.raise_exception("TypeError", message));
        }
        let extra_positional =
            if signature.variadic_slot.is_some() && arguments.len() > signature.positional_count {
                arguments.split_off(signature.positional_count)
            } else {
                Vec::new()
            };
        let mut locals = vec![None; code.local_names.len()];
        for (slot, value) in arguments.into_iter().enumerate() {
            locals[slot] = Some(value);
        }
        if let Some(slot) = signature.variadic_slot {
            locals[slot] = Some(self.allocate_object(Object::Tuple(extra_positional))?);
        }
        let mut extra_keywords = Vec::new();
        for (keyword, value) in keyword_arguments {
            let slot = code.parameters.iter().position(|parameter| {
                parameter.name == keyword
                    && matches!(
                        parameter.kind,
                        super::super::bytecode::ParameterKind::Positional
                            | super::super::bytecode::ParameterKind::KeywordOnly
                    )
            });
            let Some(slot) = slot else {
                if signature.keyword_variadic_slot.is_some() {
                    let key = self.allocate_string(keyword)?;
                    self.reserve_result(64)?;
                    extra_keywords.push((key, value));
                    continue;
                }
                let positional_only = code.parameters.iter().any(|parameter| {
                    parameter.name == keyword
                        && parameter.kind == super::super::bytecode::ParameterKind::PositionalOnly
                });
                let message = if positional_only {
                    format!(
                        "{name}() got some positional-only arguments passed as keyword \
                         arguments: '{keyword}'"
                    )
                } else {
                    format!("{name}() got an unexpected keyword argument '{keyword}'")
                };
                return Err(self.raise_exception("TypeError", message));
            };
            if locals[slot].replace(value).is_some() {
                let message = format!("{name}() got multiple values for argument '{keyword}'");
                return Err(self.raise_exception("TypeError", message));
            }
        }
        if let Some(slot) = signature.keyword_variadic_slot {
            locals[slot] = Some(self.allocate_object(Object::Dict(extra_keywords.into()))?);
        }
        if defaults.len() != signature.default_slots.len() {
            return Err(format!("{name}() has invalid default argument metadata"));
        }
        for (&slot, default) in signature.default_slots.iter().zip(defaults) {
            if locals[slot].is_none() {
                locals[slot] = Some(*default);
            }
        }
        let missing = |keyword_only: bool| {
            code.parameters
                .iter()
                .enumerate()
                .filter(|(slot, parameter)| {
                    let kind_matches = match parameter.kind {
                        super::super::bytecode::ParameterKind::PositionalOnly
                        | super::super::bytecode::ParameterKind::Positional => !keyword_only,
                        super::super::bytecode::ParameterKind::KeywordOnly => keyword_only,
                        _ => false,
                    };
                    kind_matches && locals[*slot].is_none() && !parameter.has_default
                })
                .map(|(_, parameter)| format!("'{}'", parameter.name))
                .collect::<Vec<_>>()
        };
        for (keyword_only, description) in [(false, "positional"), (true, "keyword-only")] {
            let names = missing(keyword_only);
            if names.is_empty() {
                continue;
            }
            let plural = if names.len() == 1 { "" } else { "s" };
            let message = format!(
                "{name}() missing {} required {description} argument{plural}: {}",
                names.len(),
                english_list(&names)
            );
            return Err(self.raise_exception("TypeError", message));
        }
        let uses_repl_globals = closure
            .map(|scope| self.state.heap.scope_uses_repl_globals(scope))
            .transpose()?
            .unwrap_or(true);
        self.state.heap.allocate_scope_slots(
            closure,
            uses_repl_globals,
            code.local_names.clone(),
            locals,
            HashMap::new(),
            &mut self.interp.resources,
        )
    }

    /// Advance any Python iterator by one item; `Ok(None)` means a builtin iterator is exhausted.
    /// A user iterator's `StopIteration` stays pending as an error, so callers that treat it as
    /// exhaustion check [`Self::pending_stop_iteration`].
    /// Instantiate a class whose MRO defines `__new__`, as `type.__call__` does: call `__new__`
    /// with the class and the call's arguments, then run `__init__` with the same arguments
    /// when the result is an instance of the class. Any other result is returned as is.
    fn construct_with_new(
        &mut self,
        class: super::super::heap::ObjectId,
        class_type: super::super::object_model::TypeId,
        owner: super::super::object_model::TypeId,
        constructor: Value,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<CallResult, String> {
        // `__new__` is a static method that receives the class explicitly.
        let constructor = self
            .bind_type_attribute(constructor, None, class_type, owner)?
            .ok_or("__new__ descriptor has no value")?;
        let mut new_arguments = Vec::with_capacity(arguments.len().saturating_add(1));
        new_arguments.push(Value::Object(class));
        new_arguments.extend(arguments.iter().copied());
        let created =
            match self.invoke_call(constructor, new_arguments, keyword_arguments.clone())? {
                CallResult::Value(value) => value,
                CallResult::Exit(status) => return Ok(CallResult::Exit(status)),
                CallResult::EnteredFrame => unreachable!("invoke_call is immediate"),
                CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                    unreachable!("immediate call cannot suspend")
                }
            };
        if !self.is_instance(&created, &Value::Object(class))? {
            return Ok(CallResult::Value(created));
        }
        let layout = match self.state.heap.get(class)? {
            Object::Class { layout, .. } => *layout,
            _ => return Err("type.__call__ requires a class".into()),
        };
        let initializer = if layout == ClassLayout::Object {
            let initializer = self.type_lookup(class_type, "__init__")?;
            // CPython accepts constructor arguments when a class replaces `__new__` but keeps
            // `object.__init__`; that base initializer has no state to populate.
            initializer.filter(|(defining_type, _)| {
                *defining_type != BuiltinType::Object.id() || owner == BuiltinType::Object.id()
            })
        } else {
            self.class_attribute_entry(class, "__init__")?
                .map(|(owner, value)| {
                    self.class_type_id(&Value::Object(owner))
                        .and_then(|owner| owner.ok_or("initializer class has no type".into()))
                        .map(|owner| (owner, value))
                })
                .transpose()?
        };
        let Some((owner, initializer)) = initializer else {
            return Ok(CallResult::Value(created));
        };
        let initializer = self
            .bind_type_attribute(initializer, Some(created), class_type, owner)?
            .ok_or("__init__ descriptor has no value")?;
        match self.invoke_call(initializer, arguments, keyword_arguments)? {
            CallResult::Value(value) if value.is_none() => Ok(CallResult::Value(created)),
            CallResult::Value(value) => {
                let type_name = self.type_name_of(&value)?;
                Err(self.raise_exception(
                    "TypeError",
                    format!("__init__() should return None, not '{type_name}'"),
                ))
            }
            CallResult::Exit(status) => Ok(CallResult::Exit(status)),
            CallResult::EnteredFrame => unreachable!("invoke_call is immediate"),
            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                unreachable!("immediate call cannot suspend")
            }
        }
    }

    /// The value `builtin(*arguments, **keyword_arguments)` constructs, such as the tuple that
    /// `tuple(iterable)` builds.
    fn builtin_value(
        &mut self,
        builtin: BuiltinType,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<Value, String> {
        match self.call_builtin_type(builtin, arguments, keyword_arguments)? {
            CallResult::Value(value) => Ok(value),
            _ => Err(format!("{}() did not produce a value", builtin.name())),
        }
    }

    /// `builtin.__new__(class, ...)` for a builtin type that user classes may derive from, such
    /// as `tuple.__new__(cls, iterable)`: the builtin value itself when `class` is `builtin`, and
    /// otherwise an instance of the subclass `class` that holds the value.
    pub(super) fn new_builtin_instance(
        &mut self,
        builtin: BuiltinType,
        class: Value,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<Value, String> {
        let name = builtin.name();
        let subclass = match class.native_value() {
            Some(NativeValue::BuiltinType(class_type)) if class_type == builtin => None,
            Some(
                native @ (NativeValue::BuiltinType(_)
                | NativeValue::ExceptionType(_)
                | NativeValue::ValueKind(_)),
            ) => {
                let class_name = match native {
                    NativeValue::BuiltinType(class_type) => class_type.name(),
                    NativeValue::ExceptionType(ExceptionType(class_name)) => class_name,
                    NativeValue::ValueKind(kind) => kind
                        .name
                        .rsplit_once('.')
                        .map_or(kind.name, |(_, class_name)| class_name),
                    _ => unreachable!("matched above"),
                };
                let message = format!(
                    "{name}.__new__({class_name}): {class_name} is not a subtype of {name}"
                );
                return Err(self.raise_exception("TypeError", message));
            }
            _ => match class.object_id().map(|id| (id, self.state.heap.get(id))) {
                Some((
                    id,
                    Ok(Object::Class {
                        layout: ClassLayout::Builtin(layout),
                        ..
                    }),
                )) if *layout == builtin => Some(id),
                Some((
                    _,
                    Ok(Object::Class {
                        name: class_name, ..
                    }),
                )) => {
                    let message = format!(
                        "{name}.__new__({class_name}): {class_name} is not a subtype of {name}"
                    );
                    return Err(self.raise_exception("TypeError", message));
                }
                _ => {
                    let type_name = self.type_name_of(&class)?;
                    let message =
                        format!("{name}.__new__(X): X is not a type object ({type_name})");
                    return Err(self.raise_exception("TypeError", message));
                }
            },
        };
        let value = self.builtin_value(builtin, arguments, keyword_arguments)?;
        let Some(class) = subclass else {
            return Ok(value);
        };
        self.allocate_object(Object::Instance {
            class,
            payload: InstancePayload::Builtin(value),
            attributes: InstanceAttributes::default(),
        })
    }

    /// `object.__new__(class)`: a new instance of `class` with no attributes set.
    ///
    /// As in CPython, extra arguments are an error when the class overrides `__new__` (they were
    /// meant for it) or when it keeps `object.__init__` (nothing would accept them). A class whose
    /// instances have a builtin layout, such as an `int` or exception subclass, must be created
    /// by that builtin's `__new__`.
    pub(super) fn new_instance(
        &mut self,
        class: Value,
        has_arguments: bool,
    ) -> Result<Value, String> {
        if matches!(
            class.native_value(),
            Some(NativeValue::BuiltinType(BuiltinType::Object))
        ) {
            if has_arguments {
                return Err(self.raise_exception("TypeError", "object() takes no arguments"));
            }
            return self.allocate_object(Object::Bare);
        }
        let builtin = match class.native_value() {
            Some(NativeValue::BuiltinType(builtin)) => Some(builtin.name()),
            Some(NativeValue::ExceptionType(ExceptionType(name))) => Some(name),
            _ => None,
        };
        if let Some(name) = builtin {
            return Err(self.raise_exception(
                "TypeError",
                format!("object.__new__({name}) is not safe, use {name}.__new__()"),
            ));
        }
        let Some((id, name, layout, exception_base)) =
            class
                .object_id()
                .and_then(|id| match self.state.heap.get(id) {
                    Ok(Object::Class {
                        name,
                        layout,
                        exception_base,
                        ..
                    }) => Some((id, name.clone(), *layout, *exception_base)),
                    _ => None,
                })
        else {
            let type_name = self.type_name_of(&class)?;
            return Err(self.raise_exception(
                "TypeError",
                format!("object.__new__(X): X is not a type object ({type_name})"),
            ));
        };
        if layout != ClassLayout::Object || exception_base.is_some() {
            return Err(self.raise_exception(
                "TypeError",
                format!("object.__new__({name}) is not safe, use {name}.__new__()"),
            ));
        }
        if has_arguments {
            if self.class_attribute_entry(id, "__new__")?.is_some() {
                return Err(self.raise_exception(
                    "TypeError",
                    "object.__new__() takes exactly one argument (the type to instantiate)",
                ));
            }
            if self.class_attribute_entry(id, "__init__")?.is_none() {
                return Err(
                    self.raise_exception("TypeError", format!("{name}() takes no arguments"))
                );
            }
        }
        self.allocate_object(Object::Instance {
            class: id,
            payload: InstancePayload::Object,
            attributes: InstanceAttributes::default(),
        })
    }

    pub(super) fn iterator_next(&mut self, iterator: &Value) -> Result<Option<Value>, String> {
        let Some(id) = iterator.object_id() else {
            return Err(self.raise_object_type_error(iterator, "is not an iterator"));
        };
        Ok(match self.state.heap.get(id)?.clone() {
            Object::CallableIterator {
                callable,
                sentinel,
                exhausted,
            } => {
                if exhausted {
                    None
                } else {
                    self.charge_cpu(1)?;
                    let value = <Self as PyRuntime>::call_value(
                        self,
                        callable,
                        CallArgs::new(Vec::new(), Vec::new()),
                    )
                    .map_err(|error| error.to_string())?;
                    if self.values_equal(&value, &sentinel)? {
                        if let Object::CallableIterator { exhausted, .. } =
                            self.state.heap.get_mut(id)?
                        {
                            *exhausted = true;
                        }
                        None
                    } else {
                        Some(value)
                    }
                }
            }
            Object::CountIterator { current, step } => {
                let next = super::super::stdlib::itertools::count_next(current, step)
                    .map_err(str::to_string)?;
                if let Object::CountIterator { current, .. } = self.state.heap.get_mut(id)? {
                    *current = next;
                }
                Some(Value::Int(current))
            }
            Object::Generator { .. } => self.resume_generator(id)?,
            Object::Iterator { .. }
            | Object::SequenceIterator { .. }
            | Object::ReverseIterator { .. }
            | Object::RangeIterator { .. }
            | Object::StreamIterator { .. } => self.next_stored_iterator(id)?,
            _ => match self.invoke_slot(iterator, Slot::Next, "__next__", Vec::new())? {
                Some(value) => Some(value),
                None => return Err(self.raise_object_type_error(iterator, "is not an iterator")),
            },
        })
    }

    /// Whether the error being propagated is a `StopIteration`.
    pub(super) fn pending_stop_iteration(&self) -> bool {
        self.pending_exception
            .as_ref()
            .is_some_and(|exception| exception.kind == "StopIteration")
    }

    pub(super) fn record_native_error(&mut self, error: PyError) -> String {
        // Native code converts VM failures with `PyError::runtime_error`. When the VM already
        // raised a Python exception for that failure, it is the one propagating; a generic
        // RuntimeError must not replace it.
        if matches!(error.kind, PyErrorKind::Runtime) && self.pending_exception.is_some() {
            return error.message;
        }
        let kind = match error.kind {
            PyErrorKind::Type => Some("TypeError"),
            PyErrorKind::Value => Some("ValueError"),
            PyErrorKind::ZeroDivision => Some("ZeroDivisionError"),
            PyErrorKind::Overflow => Some("OverflowError"),
            PyErrorKind::Runtime => Some("RuntimeError"),
            PyErrorKind::Exception(kind) => Some(kind),
            PyErrorKind::Resource
            | PyErrorKind::Unsupported
            | PyErrorKind::Raised
            | PyErrorKind::Exit(_)
            | PyErrorKind::Suspend(_) => None,
        };
        if let Some(kind) = kind {
            if let Ok(value) = self.allocate_exception(kind.to_string(), error.message.clone()) {
                self.pending_exception = Some(RaisedException {
                    kind: kind.to_string(),
                    value,
                });
            }
        }
        error.message
    }

    /// Whether a native called in `mode` may suspend its process: only a deferred call made
    /// by scheduler-dispatched bytecode, outside any synchronous execution.
    fn may_suspend(&self, mode: CallMode) -> bool {
        matches!(mode, CallMode::Deferred(_)) && self.synchronous_frames == 0
    }

    pub(super) fn resume_native_call(
        &mut self,
        pending: PendingNativeCall,
    ) -> Result<CallResult, String> {
        if matches!(pending, PendingNativeCall::Input { .. }) {
            return self.resume_input(pending);
        }
        let retry = pending.clone();
        let previous_suspend = self.native_suspend_allowed;
        self.native_suspend_allowed = self.synchronous_frames == 0;
        let result = match pending {
            PendingNativeCall::Function {
                function,
                arguments,
                ..
            } => (function.call)(self, arguments),
            PendingNativeCall::Method {
                method,
                receiver,
                arguments,
                ..
            } => (method.call)(self, receiver, arguments),
            PendingNativeCall::Input { .. } => unreachable!("handled above"),
        };
        self.native_suspend_allowed = previous_suspend;
        match result {
            Ok(value) => match self.pending_wait.take() {
                Some(reason) => Ok(CallResult::Blocked(reason, value)),
                None => Ok(CallResult::Value(value)),
            },
            Err(PyError {
                kind: PyErrorKind::Exit(status),
                ..
            }) => Ok(CallResult::Exit(status)),
            Err(PyError {
                kind: PyErrorKind::Suspend(reason),
                ..
            }) => Ok(CallResult::Retry(reason, retry)),
            Err(error) => Err(self.record_native_error(error)),
        }
    }

    fn resume_input(&mut self, pending: PendingNativeCall) -> Result<CallResult, String> {
        let PendingNativeCall::Input { call_span } = pending else {
            unreachable!("caller checked the variant");
        };
        let marker = Value::Native(NativeValue::Stream(Stream::Stdin));
        let previous_suspend = self.native_suspend_allowed;
        self.native_suspend_allowed = self.synchronous_frames == 0;
        let result = self.read_stream(&marker, None, true);
        self.native_suspend_allowed = previous_suspend;
        match result {
            Ok(read) => self.finish_input(read).map(CallResult::Value),
            Err(PyError {
                kind: PyErrorKind::Suspend(reason),
                ..
            }) => Ok(CallResult::Retry(
                reason,
                PendingNativeCall::Input { call_span },
            )),
            Err(error) => Err(self.record_native_error(error)),
        }
    }

    /// Strip the trailing newline `input()` reads and raise `EOFError` on an empty read.
    fn finish_input(&mut self, read: PyStreamRead) -> Result<Value, String> {
        let mut text = match read {
            PyStreamRead::Text(text) => text,
            PyStreamRead::Bytes(_) => {
                unreachable!("input() only reads the modeled text stdin stream")
            }
        };
        if text.is_empty() {
            return Err(
                self.record_native_error(PyError::exception("EOFError", "EOF when reading a line"))
            );
        }
        if text.ends_with('\n') {
            text.pop();
            if text.ends_with('\r') {
                text.pop();
            }
        }
        self.allocate_string(text)
    }
}

/// Join names the way CPython's argument errors do: `'a'`, `'a' and 'b'`, `'a', 'b', and 'c'`.
fn english_list(names: &[String]) -> String {
    match names {
        [] => String::new(),
        [only] => only.clone(),
        [first, second] => format!("{first} and {second}"),
        [rest @ .., last] => format!("{}, and {last}", rest.join(", ")),
    }
}
