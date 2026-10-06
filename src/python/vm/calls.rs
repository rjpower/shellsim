//! Call preparation, callable dispatch, argument binding, and Python frame entry.

use std::sync::Arc;

use super::super::ast::{Program, Statement, StatementKind};
use super::super::bytecode::{KeywordName, ParameterKind};
use super::super::heap::{GeneratorObject, Ref};
use super::super::scopes;
use super::{
    expect_arity, number, range_length, string, BigInt, BinaryOperator, Builtin, BuiltinType,
    CallArgs, CallMode, ClassLayout, CodeRef, ComparisonOperator, ExceptionType, Flow, FrameEntry,
    FrameKind, HashMap, NativeValue, Object, PendingNativeCall, PyError, PyErrorKind, PyRuntime,
    PyStreamRead, Slot, StoredCallArgs, Stream, Value, Vm,
};
use crate::python::error::{Control, PyResult};
use num_traits::{One, Signed, Zero};

impl<'s> Vm<'s> {
    /// Length of a builtin representation without consulting Python slots. Native length slots
    /// and the `len()` fallback share this path so their metering and results cannot diverge.
    pub(super) fn physical_length(&self, value: Value) -> PyResult<Option<usize>> {
        let subject = value;
        if let Some(length) = string::string_length(&self.state.heap, subject)? {
            return Ok(Some(length));
        }
        if !subject.is_object() {
            return Ok(None);
        }
        Ok(match self.get(subject)? {
            Object::List(values) | Object::Tuple(values) => Some(values.len()),
            Object::Set(values) | Object::FrozenSet(values) => Some(values.len()),
            Object::Range { start, stop, step } => Some(range_length(*start, *stop, *step)?),
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => Some(entries.len()),
            _ => None,
        })
    }

    /// The builtin `abs(value)`, through the `__abs__` slot.
    pub(super) fn absolute(&mut self, value: Value) -> PyResult<Value> {
        if let Some(result) = self.invoke_slot(&value, Slot::Absolute, "__abs__", Vec::new())? {
            return Ok(result);
        }
        let message = format!(
            "bad operand type for abs(): '{}'",
            self.type_name_of(&value)?
        );
        Err(PyError::exception("TypeError", message))
    }

    /// Parse the source string passed to `exec` or `eval` (`builtin`). Parsing is charged
    /// before it runs, and nesting is bounded so dynamic source cannot recurse without limit.
    /// `eval` ignores leading spaces and tabs, as CPython does.
    fn parse_dynamic_source(&mut self, builtin: &str, source: &Value) -> PyResult<Program> {
        let source = string::string_value(&self.state.heap, *source)?.ok_or_else(|| {
            PyError::type_error(format!(
                "{builtin}() arg 1 must be a string, bytes or code object"
            ))
        })?;
        let source = if builtin == "eval" {
            source.trim_start_matches([' ', '\t'])
        } else {
            &source
        };
        if self.bytecode_frames.len() >= 256 {
            return Err(format!("maximum {builtin} depth exceeded").into());
        }
        let parse_memory = super::super::source::front_end_memory(source.len())
            .ok_or_else(|| format!("{builtin} source is too large"))?;
        self.charge_cpu(u64::try_from(source.len()).unwrap_or(u64::MAX))?;
        self.reserve_result(parse_memory)?;
        // The front end rejects valid syntax it does not model with the same error as invalid
        // syntax, so neither is a catchable `SyntaxError`.
        let syntax_error = |message: &str, span: &super::super::source::Span| {
            PyError::unsupported(format!(
                "{message} at line {}, column {}",
                span.line, span.column
            ))
        };
        let tokens = super::super::lexer::lex(source)
            .map_err(|error| syntax_error(&error.message, &error.span))?;
        let token_memory = super::super::source::token_memory(tokens.len())
            .ok_or_else(|| format!("{builtin} source is too large"))?;
        self.reserve_result(token_memory)?;
        super::super::parser::parse(tokens)
            .map_err(|error| syntax_error(&error.message, &error.span))
    }

    fn pow_integer_argument(&self, value: &Value) -> Result<(BigInt, usize), PyError> {
        let integer = <Self as PyRuntime>::integer_bigint(self, value)?.ok_or_else(|| {
            PyError::type_error("pow() 3rd argument not allowed unless all arguments are integers")
        })?;
        let digits = super::super::number::decimal_digits(&integer);
        Ok((integer, digits))
    }

    /// `dir(value)` through a `__dir__` that the value's class defines: the names it returns,
    /// sorted, or `None` when the class defines none.
    fn custom_dir(&mut self, value: &Value) -> PyResult<Option<Value>> {
        let Some(class) = self.instance_class(*value)? else {
            return Ok(None);
        };
        if self.class_attribute(class, "__dir__")?.is_none() {
            return Ok(None);
        }
        let method = self
            .resolve_attribute(*value, "__dir__")?
            .ok_or("__dir__ disappeared during lookup")?;
        let result = self.invoke_value(method, Vec::new())?;
        let mut names = Vec::new();
        for item in self.iterable_values(&result)? {
            let Some(name) = string::string_value(&self.state.heap, item)? else {
                return Err(PyError::exception(
                    "TypeError",
                    "__dir__() must return strings",
                ));
            };
            names.push((name, item));
        }
        self.charge_cpu(u64::try_from(names.len()).unwrap_or(u64::MAX))?;
        names.sort_by(|left, right| left.0.cmp(&right.0));
        let values = names.into_iter().map(|(_, item)| item).collect::<Vec<_>>();
        self.alloc(Object::List(Ref::all(values))).map(Some)
    }

    fn dir_names(&self, value: &Value) -> PyResult<Vec<String>> {
        if let Some(NativeValue::Module(module)) = value.native_value() {
            return Ok(module
                .functions
                .iter()
                .map(|function| function.name.to_string())
                .chain(module.values.iter().map(|value| value.name().to_string()))
                .collect());
        }
        if !value.is_object() {
            return self.type_attribute_names(value);
        }
        if let Some(class) = self.instance_class(*value)? {
            let mut names = self.instance_attribute_names(*value)?;
            if let Object::Class(class_object) = self.get(class)? {
                names.extend(class_object.attributes.keys().cloned());
                for ancestor in &class_object.mro {
                    if let Object::Class(ancestor) = self.get(self.value(ancestor))? {
                        names.extend(ancestor.attributes.keys().cloned());
                    }
                }
            }
            return Ok(names);
        }
        match self.get(*value)? {
            Object::Module { scope, .. } => {
                let namespace = self.module_namespace(self.value(scope))?;
                Ok(self
                    .namespace_entries(namespace)?
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect())
            }
            Object::Class(class_object) => {
                let mut names = class_object.attributes.keys().cloned().collect::<Vec<_>>();
                for ancestor in &class_object.mro {
                    if let Object::Class(ancestor) = self.get(self.value(ancestor))? {
                        names.extend(ancestor.attributes.keys().cloned());
                    }
                }
                Ok(names)
            }
            _ => self.type_attribute_names(value),
        }
    }

    /// The names a builtin value's registered type and its ancestors publish, for `dir()`.
    fn type_attribute_names(&self, value: &Value) -> PyResult<Vec<String>> {
        let type_id = self.type_id(value)?;
        let ty = self.state.types.get(type_id)?;
        let mut names = ty.attributes.keys().cloned().collect::<Vec<_>>();
        let ancestors = ty.mro.clone();
        for ancestor in ancestors {
            names.extend(self.state.types.get(ancestor)?.attributes.keys().cloned());
        }
        Ok(names)
    }

    /// Call the value below the arguments on the operand stack. `positional` values follow the
    /// callee, then one value per `keywords` entry; `starred` marks the `*iterable` and
    /// `**mapping` operands among them. Python functions bind straight from the stack through
    /// [`Self::call_python`]; every other callee receives owned arguments.
    pub(super) fn call(
        &mut self,
        positional: usize,
        keywords: &[KeywordName],
        starred: &[bool],
        mode: CallMode,
    ) -> PyResult<Flow> {
        let count = positional
            .checked_add(keywords.len())
            .ok_or("too many call arguments")?;
        if starred.len() != count {
            return Err("invalid bytecode call argument metadata".into());
        }
        if self.frame_stack_len() < count + 1 {
            return Err("invalid bytecode stack effect".into());
        }
        if starred.contains(&true) {
            let (positional, keywords) =
                self.expand_starred_arguments(positional, keywords, starred)?;
            return self.call_expanded(positional, &keywords, mode);
        }
        self.call_expanded(positional, keywords, mode)
    }

    /// Replace the call operands on the stack with their expansion: each `*iterable` by its
    /// items and each `**mapping` by its values, whose names join the returned keyword list.
    /// Returns the expanded positional count and keyword names.
    fn expand_starred_arguments(
        &mut self,
        positional: usize,
        keywords: &[KeywordName],
        starred: &[bool],
    ) -> PyResult<(usize, Vec<KeywordName>)> {
        let arguments_start = self.stack.len() - positional - keywords.len();
        let mut raw_arguments = self
            .execution
            .stack
            .split_off(&self.state.heap, arguments_start);
        let keyword_values = raw_arguments.split_off(positional);
        let mut expanded = 0usize;
        for (argument, starred) in raw_arguments.into_iter().zip(&starred[..positional]) {
            if *starred {
                for value in self.iterable_values(&argument)? {
                    self.push_argument(value)?;
                    expanded += 1;
                }
            } else {
                self.push_argument(argument)?;
                expanded += 1;
            }
        }
        let mut names = Vec::with_capacity(keywords.len());
        for ((name, value), starred) in keywords
            .iter()
            .zip(keyword_values)
            .zip(&starred[positional..])
        {
            match (name, starred) {
                (Some(name), false) => {
                    self.push_keyword_argument(&mut names, name.clone(), value)?;
                }
                (None, true) => {
                    let Some(entries) = self.mapping_items(value)? else {
                        let message = format!(
                            "argument after ** must be a mapping, not {}",
                            self.type_name_of(&value)?
                        );
                        return Err(PyError::exception("TypeError", message));
                    };
                    for (key, value) in entries {
                        let Some(name) = string::string_value(&self.state.heap, key)? else {
                            return Err(PyError::exception(
                                "TypeError",
                                "keywords must be strings",
                            ));
                        };
                        self.push_keyword_argument(&mut names, Arc::from(name), value)?;
                    }
                }
                _ => return Err("invalid keyword argument metadata".into()),
            }
        }
        Ok((expanded, names))
    }

    /// Push one expanded positional argument, metering the stack growth.
    fn push_argument(&mut self, value: Value) -> PyResult<()> {
        self.reserve_result(64)?;
        self.charge_cpu(1)?;
        self.push(value);
        Ok(())
    }

    /// Push one expanded keyword argument and record its name, rejecting a repeated name.
    fn push_keyword_argument(
        &mut self,
        names: &mut Vec<KeywordName>,
        name: Arc<str>,
        value: Value,
    ) -> PyResult<()> {
        if names
            .iter()
            .any(|existing| existing.as_deref() == Some(&*name))
        {
            let message = format!("got multiple values for keyword argument '{name}'");
            return Err(PyError::exception("TypeError", message));
        }
        self.reserve_result(64usize.saturating_add(name.len()))?;
        self.charge_cpu(1)?;
        self.push(value);
        names.push(Some(name));
        Ok(())
    }

    /// Dispatch a call whose operands hold no starred entries.
    fn call_expanded(
        &mut self,
        positional: usize,
        keywords: &[KeywordName],
        mode: CallMode,
    ) -> PyResult<Flow> {
        let count = positional + keywords.len();
        let callee = self.peek(count)?;
        if callee.is_object() {
            match self.get(callee)? {
                Object::Function(_) => {
                    return self.call_python(callee, None, positional, keywords, mode);
                }
                Object::DescriptorBoundMethod {
                    receiver,
                    descriptor,
                    ..
                } if descriptor.is_object() => {
                    let (receiver, descriptor) = (self.value(receiver), self.value(descriptor));
                    return self.call_python(
                        descriptor,
                        Some(receiver),
                        positional,
                        keywords,
                        mode,
                    );
                }
                _ => {}
            }
        }
        self.reserve_result(count.saturating_mul(64))?;
        self.charge_cpu(count as u64)?;
        let arguments_start = self.stack.len() - count;
        let mut arguments = self
            .execution
            .stack
            .split_off(&self.state.heap, arguments_start);
        let keyword_values = arguments.split_off(positional);
        let mut keyword_arguments = Vec::with_capacity(keywords.len());
        for (name, value) in keywords.iter().zip(keyword_values) {
            let name = name.as_ref().ok_or("invalid keyword argument metadata")?;
            keyword_arguments.push((name.to_string(), value));
        }
        let function = self.pop()?;
        if let Some(call) = self.registered_kind(&function).and_then(|kind| kind.call) {
            return call(self, function, CallArgs::new(arguments, keyword_arguments))
                .map(|value| self.produce(value));
        }
        if function.is_object() {
            /// What a heap callable is, with the values its call needs.
            enum Callee {
                BoundMethod { receiver: Value, descriptor: Value },
                GenericAlias(Value),
                Instance(Value),
                Class,
                Other,
            }
            let callee = match self.get(function)? {
                Object::DescriptorBoundMethod {
                    receiver,
                    descriptor,
                    ..
                } => Callee::BoundMethod {
                    receiver: self.value(receiver),
                    descriptor: self.value(descriptor),
                },
                Object::GenericAlias { origin, .. } => Callee::GenericAlias(self.value(origin)),
                Object::Class(_) => Callee::Class,
                _ => match self.instance_class(function)? {
                    Some(class) => Callee::Instance(class),
                    None => Callee::Other,
                },
            };
            return match callee {
                Callee::BoundMethod {
                    receiver,
                    descriptor,
                } => {
                    if let Some(NativeValue::SlotWrapper { owner, slot }) =
                        descriptor.native_value()
                    {
                        self.call_slot_wrapper(owner, slot, receiver, arguments, keyword_arguments)
                            .map(|value| self.produce(value))
                    } else if let Some(NativeValue::NativeMethod(method)) =
                        descriptor.native_value()
                    {
                        self.call_native_method(
                            method,
                            receiver,
                            receiver,
                            CallArgs::new(arguments, keyword_arguments),
                            mode,
                        )
                    } else {
                        Err("bound descriptor is not callable".into())
                    }
                }
                Callee::GenericAlias(origin) => {
                    self.invoke_call(origin, arguments, keyword_arguments)
                }
                Callee::Instance(class) => {
                    let type_id = self.object_type_id(function)?;
                    if self.state.types.slot(type_id, Slot::Call)?.is_none() {
                        return Err(self.raise_object_type_error(&function, "is not callable"));
                    }
                    let (defining_class, descriptor) = self
                        .class_attribute_entry(class, "__call__")?
                        .ok_or("call slot has no descriptor")?;
                    let callable =
                        self.bind_descriptor(descriptor, Some(function), class, defining_class)?;
                    self.invoke_call(callable, arguments, keyword_arguments)
                }
                Callee::Class => self.call_user_class(function, arguments, keyword_arguments, true),
                Callee::Other => Err(self.raise_object_type_error(&function, "is not callable")),
            };
        }
        if let Some(NativeValue::BuiltinType(builtin_type)) = function.native_value() {
            return self.call_builtin_type(builtin_type, arguments, keyword_arguments);
        }
        if let Some(NativeValue::ValueKind(kind)) = function.native_value() {
            let call = CallArgs::new(arguments, keyword_arguments);
            return (kind.construct)(self, call).map(|value| self.produce(value));
        }
        if let Some(NativeValue::ExceptionType(exception_type)) = function.native_value() {
            if !keyword_arguments.is_empty() {
                let message = format!("{}() takes no keyword arguments", exception_type.0);
                return Err(PyError::exception("TypeError", message));
            }
            // `OSError(errno, strerror, ...)` constructs the errno's subclass, as CPython does.
            let kind = match arguments.first().and_then(|value| value.immediate_int()) {
                Some(errno) if exception_type.0 == "OSError" && arguments.len() >= 2 => {
                    i32::try_from(errno)
                        .map_or("OSError", super::super::exception_types::os_error_subclass)
                }
                _ => exception_type.0,
            };
            return {
                let value = self.allocate_exception_object(kind, arguments)?;
                Ok(self.produce(value))
            };
        }
        if let Some(NativeValue::SlotWrapper { owner, slot }) = function.native_value() {
            if arguments.is_empty() {
                return Err(PyError::exception(
                    "TypeError",
                    "slot wrapper requires a receiver",
                ));
            }
            let receiver = arguments.remove(0);
            return self
                .call_slot_wrapper(owner, slot, receiver, arguments, keyword_arguments)
                .map(|value| self.produce(value));
        }
        if let Some(NativeValue::NativeMethod(method)) = function.native_value() {
            if arguments.is_empty() {
                return Err("unbound native method requires a receiver".into());
            }
            let receiver = arguments.remove(0);
            return self.call_native_method(
                method,
                receiver,
                receiver,
                CallArgs::new(arguments, keyword_arguments),
                mode,
            );
        }
        if let Some(NativeValue::NativeFunction(function)) = function.native_value() {
            let call = CallArgs::new(arguments, keyword_arguments);
            // Kept as pinned values so a suspension can store them for the retry.
            let retry_arguments = matches!(mode, CallMode::Deferred(_)).then(|| call.clone());
            let previous_suspend = self.native_suspend_allowed;
            self.native_suspend_allowed = self.may_suspend(mode);
            let result = (function.call)(self, call);
            self.native_suspend_allowed = previous_suspend;
            return match result.map_err(PyError::into_control) {
                Ok(value) => Ok(self.native_result(value)),
                Err(Ok(Control::Exit(status))) => Ok(Flow::Exit(status)),
                Err(Ok(Control::Suspend(reason))) => match (mode, retry_arguments) {
                    (CallMode::Deferred(call_span), Some(arguments)) => {
                        let retry = PendingNativeCall::Function {
                            function,
                            arguments: StoredCallArgs::store(self, &arguments),
                            call_span,
                        };
                        Ok(self.suspend(reason, Some(retry)))
                    }
                    _ => Err("native call suspended outside scheduler dispatch".into()),
                },
                Err(Err(error)) => Err(error),
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
                // `file` is a modeled stream or, as with `io.StringIO`, any object whose
                // `write` method takes the text. Without `file`, a `sys.stdout` the program
                // rebound is used, so `contextlib.redirect_stdout` captures output.
                let mut target = None;
                let mut flush = false;
                for (name, value) in &keyword_arguments {
                    match name.as_str() {
                        "sep" => {
                            if value.is_none() {
                                continue;
                            }
                            separator = string::string_value(&self.state.heap, *value)?
                                .ok_or("sep must be None or a string")?;
                        }
                        "end" => {
                            if value.is_none() {
                                continue;
                            }
                            ending = string::string_value(&self.state.heap, *value)?
                                .ok_or("end must be None or a string")?;
                        }
                        "file" if !value.is_none() => target = Some(*value),
                        "file" => {}
                        "flush" => flush = self.truth_value(value)?,
                        _ => {
                            return Err(format!(
                                "print() got an unexpected keyword argument {name:?}"
                            )
                            .into())
                        }
                    }
                }
                if target.is_none() {
                    if let Some(sys) = self.loaded_module("sys") {
                        target = self.resolve_attribute(sys, "stdout")?;
                    }
                }
                let mut rendered = Vec::with_capacity(arguments.len());
                for value in &arguments {
                    rendered.push(self.display_value(value)?);
                }
                let text = rendered.join(&separator);
                let target = target.unwrap_or(Value::Native(NativeValue::Stream(Stream::Stdout)));
                if let Some(NativeValue::Stream(stream)) = target.native_value() {
                    self.write_output(stream, text.as_bytes());
                    self.write_output(stream, ending.as_bytes());
                    return Ok(self.produce(Value::None));
                }
                let write = self
                    .resolve_attribute(target, "write")?
                    .ok_or_else(|| self.raise_object_type_error(&target, "has no write method"))?;
                for piece in [text, ending] {
                    let piece = self.allocate_string(piece)?;
                    self.invoke_value(write, vec![piece])?;
                }
                if flush {
                    if let Some(flush) = self.resolve_attribute(target, "flush")? {
                        self.invoke_value(flush, Vec::new())?;
                    }
                }
                Ok(self.produce(Value::None))
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
                match result.map_err(PyError::into_control) {
                    Ok(read) => {
                        let value = self.finish_input(read)?;
                        Ok(self.produce(value))
                    }
                    Err(Ok(Control::Exit(status))) => Ok(Flow::Exit(status)),
                    Err(Ok(Control::Suspend(reason))) => match retry {
                        Some(pending) => Ok(self.suspend(reason, Some(pending))),
                        None => Err("native call suspended outside scheduler dispatch".into()),
                    },
                    Err(Err(error)) => Err(error),
                }
            }
            Builtin::Exec => {
                expect_arity(&arguments, 1, 1)?;
                let program = self.parse_dynamic_source("exec", &arguments[0])?;
                let code = super::super::compiler::compile(program);
                let entry = self.dynamic_code_entry()?;
                match self.execute_code(&code, entry) {
                    Ok(Flow::Halt) => Ok(self.produce(Value::None)),
                    Ok(Flow::Exit(status)) => Ok(Flow::Exit(status)),
                    Ok(_) => Err("exec source did not finish normally".into()),
                    Err((error, span)) => Err(error.located(|| {
                        format!(
                            " in exec source at line {}, column {}",
                            span.line, span.column
                        )
                    })),
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
                    _ => return Err(PyError::exception("SyntaxError", "invalid syntax")),
                };
                let code = super::super::compiler::compile_expression(expression);
                let entry = self.dynamic_code_entry()?;
                match self.execute_code(&code, entry) {
                    Ok(Flow::Return(value)) => {
                        self.execution.stack.push_ref(&value);
                        Ok(Flow::Next)
                    }
                    Ok(Flow::Exit(status)) => Ok(Flow::Exit(status)),
                    Ok(_) => Err("eval source did not finish normally".into()),
                    Err((error, span)) => Err(error.located(|| {
                        format!(
                            " in eval source at line {}, column {}",
                            span.line, span.column
                        )
                    })),
                }
            }
            Builtin::Exit => {
                expect_arity(&arguments, 0, 1)?;
                let status = arguments
                    .first()
                    .and_then(|value| value.as_int())
                    .unwrap_or_default();
                Ok(Flow::Exit(status as i32))
            }
            Builtin::Character => {
                expect_arity(&arguments, 1, 1)?;
                let value = number::int_value(&self.state.heap, arguments[0])
                    .ok_or("an integer is required for chr()")?;
                let Some(codepoint) = u32::try_from(value).ok().and_then(char::from_u32) else {
                    return Err(PyError::exception(
                        "ValueError",
                        "chr() arg not in range(0x110000)",
                    ));
                };
                {
                    let value = self.allocate_string(codepoint.to_string())?;
                    Ok(self.produce(value))
                }
            }
            Builtin::Ordinal => {
                expect_arity(&arguments, 1, 1)?;
                let value = if let Some(text) =
                    string::string_value(&self.state.heap, arguments[0])?
                {
                    let mut characters = text.chars();
                    let character = characters.next().ok_or("ord() expected a character")?;
                    if characters.next().is_some() {
                        return Err("ord() expected a character".into());
                    }
                    u32::from(character) as i64
                } else if let Some(bytes) = <Self as PyRuntime>::bytes_value(self, &arguments[0])? {
                    if bytes.len() != 1 {
                        return Err("ord() expected a character".into());
                    }
                    i64::from(bytes[0])
                } else {
                    return Err("ord() expected string of length 1".into());
                };
                Ok(self.produce(Value::Int(value)))
            }
            Builtin::Binary | Builtin::Octal | Builtin::Hexadecimal => {
                expect_arity(&arguments, 1, 1)?;
                let integer = <Self as PyRuntime>::integer_bigint(self, &arguments[0])?
                    .ok_or("integer argument expected")?;
                // Binary needs one digit per bit, the widest of the three spellings.
                let output_bound = usize::try_from(integer.bits())
                    .ok()
                    .and_then(|length| length.checked_add(3))
                    .ok_or("integer representation is too large")?;
                <Self as PyRuntime>::reserve_memory(self, output_bound)?;
                <Self as PyRuntime>::charge_cpu(
                    self,
                    u64::try_from(output_bound).unwrap_or(u64::MAX),
                )?;
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
                {
                    let value = self.allocate_string(rendered)?;
                    Ok(self.produce(value))
                }
            }
            Builtin::Repr => {
                expect_arity(&arguments, 1, 1)?;
                let value = self.repr_value(&arguments[0])?;
                {
                    let value = self.allocate_string(value)?;
                    Ok(self.produce(value))
                }
            }
            Builtin::Format => {
                if !keyword_arguments.is_empty() {
                    let message = "format() takes no keyword arguments".to_string();
                    return Err(PyError::exception("TypeError", message));
                }
                if !(1..=2).contains(&arguments.len()) {
                    let (bound, count) = if arguments.is_empty() {
                        ("least 1 argument", 0)
                    } else {
                        ("most 2 arguments", arguments.len())
                    };
                    let message = format!("format expected at {bound}, got {count}");
                    return Err(PyError::exception("TypeError", message));
                }
                let spec = match arguments.get(1) {
                    Some(spec) => {
                        string::string_value(&self.state.heap, *spec)?.ok_or_else(|| {
                            let message = format!(
                                "format() argument 2 must be str, not {}",
                                self.type_name_of(spec).unwrap_or_default()
                            );
                            PyError::exception("TypeError", message)
                        })?
                    }
                    None => String::new(),
                };
                self.reserve_format_spec(&spec)?;
                let value = self.format_object(&arguments[0], &spec)?;
                {
                    let value = self.allocate_string(value)?;
                    Ok(self.produce(value))
                }
            }
            Builtin::Hash => {
                expect_arity(&arguments, 1, 1)?;
                let hash = self.hash_value(&arguments[0])?;
                Ok(self.produce(Value::Int(hash)))
            }
            Builtin::Dir => {
                expect_arity(&arguments, 0, 1)?;
                if let Some(value) = arguments.first() {
                    if let Some(names) = self.custom_dir(value)? {
                        return Ok(self.produce(names));
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
                            value.as_ref().and_then(|_| {
                                let symbol = super::super::symbols::SymbolId::from_index(index)?;
                                self.symbol_name(symbol).map(str::to_string)
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
                {
                    let value = self.alloc(Object::List(Ref::all(values)))?;
                    Ok(self.produce(value))
                }
            }
            Builtin::IsInstance => {
                expect_arity(&arguments, 2, 2)?;
                {
                    let value = Value::Bool(self.is_instance(&arguments[0], &arguments[1])?);
                    Ok(self.produce(value))
                }
            }
            Builtin::IsSubclass => {
                expect_arity(&arguments, 2, 2)?;
                {
                    let value = Value::Bool(self.is_subclass(&arguments[0], &arguments[1])?);
                    Ok(self.produce(value))
                }
            }
            Builtin::Length => {
                expect_arity(&arguments, 1, 1)?;
                if let Some(value) =
                    self.invoke_slot(&arguments[0], Slot::Length, "__len__", Vec::new())?
                {
                    let length = number::int_value(&self.state.heap, value)
                        .ok_or("__len__() should return an integer")?;
                    if length < 0 {
                        return Err("__len__() should return >= 0".into());
                    }
                    return Ok(self.produce(Value::Int(length)));
                }
                let Some(length) = self.physical_length(arguments[0])? else {
                    let message = format!(
                        "object of type '{}' has no len()",
                        self.type_name_of(&arguments[0])?
                    );
                    return Err(PyError::exception("TypeError", message));
                };
                let length = i64::try_from(length)
                    .map_err(|_| PyError::exception("OverflowError", "length is too large"))?;
                Ok(self.produce(Value::Int(length)))
            }
            Builtin::Sorted => {
                expect_arity(&arguments, 1, 1)?;
                let values = self.iterable_values(&arguments[0])?;
                let mut key_function = None;
                let mut saw_key = false;
                let mut reverse = false;
                let mut saw_reverse = false;
                for (name, value) in keyword_arguments {
                    match name.as_str() {
                        // `key=None` means no key function, as in CPython.
                        "key" if key_function.is_none() && !saw_key => {
                            key_function = (!value.is_none()).then_some(value);
                            saw_key = true;
                        }
                        "reverse" if !saw_reverse => {
                            reverse = self.truth_value(&value)?;
                            saw_reverse = true;
                        }
                        "key" | "reverse" => {
                            return Err(format!(
                                "sorted() got multiple values for keyword {name:?}"
                            )
                            .into())
                        }
                        _ => {
                            return Err(
                                format!("sorted() got an unexpected keyword {name:?}").into()
                            )
                        }
                    }
                }
                let mut keyed = Vec::new();
                for value in values {
                    let key = if let Some(function) = &key_function {
                        self.push(*function);
                        self.push(value);
                        let flow = self.call(1, &[], &[false], CallMode::Immediate)?;
                        match self.immediate_value(flow)? {
                            Ok(key) => key,
                            Err(flow) => return Ok(flow),
                        }
                    } else {
                        value
                    };
                    self.reserve_result(64)?;
                    keyed.push((key, value));
                }
                // Like CPython's sort this only asks `<`, and a reversed sort keeps equal items
                // in their original order.
                super::super::sort::merge_sort(&mut keyed, |right, left| {
                    self.charge_cpu(1)?;
                    let (lesser, greater) = if reverse {
                        (left.0, right.0)
                    } else {
                        (right.0, left.0)
                    };
                    self.compare_truth(ComparisonOperator::Less, &lesser, &greater)
                })?;
                let values = keyed.into_iter().map(|(_, value)| value);
                {
                    let value = self.alloc(Object::List(Ref::all(values)))?;
                    Ok(self.produce(value))
                }
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
                            return Err(PyError::exception("TypeError", message));
                        }
                    }
                }
                let values = match arguments.len() {
                    0 => {
                        let message = format!("{name} expected at least 1 argument, got 0");
                        return Err(PyError::exception("TypeError", message));
                    }
                    1 => self.iterable_values(&arguments[0])?,
                    _ if default.is_some() => {
                        let message = format!(
                            "Cannot specify a default for {name}() with multiple positional \
                             arguments"
                        );
                        return Err(PyError::exception("TypeError", message));
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
                            self.push(function);
                            self.push(value);
                            let flow = self.call(1, &[], &[false], CallMode::Immediate)?;
                            match self.immediate_value(flow)? {
                                Ok(key) => key,
                                Err(flow) => return Ok(flow),
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
                    (Some((_, value)), _) => Ok(self.produce(value)),
                    (None, Some(default)) => Ok(self.produce(default)),
                    (None, None) => {
                        let message = format!("{name}() iterable argument is empty");
                        Err(PyError::exception("ValueError", message))
                    }
                }
            }
            Builtin::Sum => {
                expect_arity(&arguments, 1, 2)?;
                let values = self.iterable_values(&arguments[0])?;
                let start = arguments.get(1).copied().unwrap_or(Value::Int(0));
                {
                    let value = self.builtin_sum(values, start)?;
                    Ok(self.produce(value))
                }
            }
            Builtin::Absolute => {
                expect_arity(&arguments, 1, 1)?;
                {
                    let value = self.absolute(arguments[0])?;
                    Ok(self.produce(value))
                }
            }
            Builtin::Power => {
                expect_arity(&arguments, 2, 3)?;
                if arguments.len() == 3 && !arguments[2].is_none() {
                    let parsed = self.pow_integer_argument(&arguments[0]);
                    let (base, base_len) = parsed?;
                    let parsed = self.pow_integer_argument(&arguments[1]);
                    let (exponent, exponent_len) = parsed?;
                    let parsed = self.pow_integer_argument(&arguments[2]);
                    let (modulus, modulus_len) = parsed?;
                    if modulus.is_zero() {
                        let error = PyError::zero_division_error("pow() 3rd argument cannot be 0");
                        return Err(error);
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
                            return Err(PyError::value_error(message));
                        };
                        (inverse, -exponent)
                    };
                    let work = base_len
                        .saturating_add(modulus_len)
                        .saturating_mul(exponent_len.saturating_mul(4).max(1));
                    <Self as PyRuntime>::charge_cpu(self, u64::try_from(work).unwrap_or(u64::MAX))?;
                    <Self as PyRuntime>::reserve_memory(
                        self,
                        modulus_len.saturating_mul(4).max(1),
                    )?;
                    let mut result = base.modpow(&exponent, &positive_modulus);
                    if modulus.is_negative() && !result.is_zero() {
                        result += modulus;
                    }
                    let result = <Self as PyRuntime>::new_bigint(self, result)?;
                    return Ok(self.produce(result));
                }
                {
                    let value =
                        self.binary_value(BinaryOperator::Power, arguments[0], arguments[1])?;
                    Ok(self.produce(value))
                }
            }
            Builtin::Divmod => {
                expect_arity(&arguments, 2, 2)?;
                {
                    let value = self.divmod_value(arguments[0], arguments[1])?;
                    Ok(self.produce(value))
                }
            }
            Builtin::SetAttribute => {
                if arguments.len() != 3 {
                    let message = format!("setattr expected 3 arguments, got {}", arguments.len());
                    return Err(PyError::exception("TypeError", message));
                }
                let Some(name) = string::string_value(&self.state.heap, arguments[1])? else {
                    let message = format!(
                        "attribute name must be string, not '{}'",
                        self.type_name_of(&arguments[1])?
                    );
                    return Err(PyError::exception("TypeError", message));
                };
                // `setattr(owner, name, value)` is `owner.name = value` with a computed name.
                let symbol = self.intern_symbol(&name)?;
                self.store_attribute_by_symbol(arguments[0], symbol, &name, arguments[2])?;
                Ok(self.produce(Value::None))
            }
            Builtin::DeleteAttribute => {
                if arguments.len() != 2 {
                    let message = format!("delattr expected 2 arguments, got {}", arguments.len());
                    return Err(PyError::exception("TypeError", message));
                }
                let Some(name) = string::string_value(&self.state.heap, arguments[1])? else {
                    let message = format!(
                        "attribute name must be string, not '{}'",
                        self.type_name_of(&arguments[1])?
                    );
                    return Err(PyError::exception("TypeError", message));
                };
                let symbol = self.intern_symbol(&name)?;
                self.delete_attribute_by_symbol(arguments[0], symbol, &name)?;
                Ok(self.produce(Value::None))
            }
            Builtin::Callable => {
                expect_arity(&arguments, 1, 1)?;
                let callable = <Self as PyRuntime>::is_callable(self, &arguments[0])?;
                Ok(self.produce(Value::Bool(callable)))
            }
            Builtin::Iter => {
                expect_arity(&arguments, 1, 2)?;
                if arguments.len() == 2 {
                    if !<Self as PyRuntime>::is_callable(self, &arguments[0])? {
                        return Err("iter(v, w): v must be callable".into());
                    }
                    return {
                        let value = self.alloc(Object::CallableIterator {
                            callable: Ref::from(arguments[0]),
                            sentinel: Ref::from(arguments[1]),
                            exhausted: false,
                        })?;
                        Ok(self.produce(value))
                    };
                }
                {
                    let value = self.make_iterator(arguments[0])?;
                    Ok(self.produce(value))
                }
            }
            Builtin::Next => {
                expect_arity(&arguments, 1, 2)?;
                let value = match self.iterator_next(&arguments[0]) {
                    Ok(value) => value,
                    Err(error) if arguments.len() == 2 => {
                        self.catch(error, "StopIteration")?;
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
                Ok(self.produce(value))
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
                    result.push(self.alloc({
                        Object::Tuple(vec![Ref::from(Value::Int(index)), Ref::from(value)])
                    })?);
                }
                {
                    let value = self.alloc(Object::List(Ref::all(result)))?;
                    Ok(self.produce(value))
                }
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
                    let tuple = sequences.iter().map(|values| values[index]);
                    result.push(self.alloc(Object::Tuple(Ref::all(tuple)))?);
                }
                {
                    let value = self.alloc(Object::List(Ref::all(result)))?;
                    Ok(self.produce(value))
                }
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
                Ok(self.produce(Value::Bool(result)))
            }
            Builtin::Super => {
                expect_arity(&arguments, 0, 2)?;
                let (start_class, receiver) = match arguments.as_slice() {
                    [] => self
                        .method_context()?
                        .ok_or("super(): no current method context")?,
                    [start_class, receiver]
                        if start_class.is_object()
                            && matches!(self.get(*start_class), Ok(Object::Class { .. })) =>
                    {
                        (*start_class, *receiver)
                    }
                    _ => return Err("super() expects a class and instance".into()),
                };
                {
                    let value = self.alloc(Object::Super {
                        start_class: Ref::from(start_class),
                        receiver: Ref::from(receiver),
                    })?;
                    Ok(self.produce(value))
                }
            }
            Builtin::Globals => {
                expect_arity(&arguments, 0, 0)?;
                let target = self.current_globals_target()?;
                {
                    let value = self.alloc(Object::NamespaceDict(target.store()))?;
                    Ok(self.produce(value))
                }
            }
            Builtin::Locals => {
                expect_arity(&arguments, 0, 0)?;
                {
                    let value = self.current_locals()?;
                    Ok(self.produce(value))
                }
            }
            Builtin::Vars => {
                expect_arity(&arguments, 0, 1)?;
                let Some(owner) = arguments.first() else {
                    return {
                        let value = self.current_locals()?;
                        Ok(self.produce(value))
                    };
                };
                // `vars(obj)` is `obj.__dict__`: a namespace view, or a class's or native
                // module's read-only proxy.
                let Some(namespace) = self.resolve_optional_attribute(*owner, "__dict__")? else {
                    return Err(PyError::exception(
                        "TypeError",
                        "vars() argument must have __dict__ attribute",
                    ));
                };
                Ok(self.produce(namespace))
            }
        }
    }

    /// Explicit `type.__call__` skips the metaclass override while retaining ordinary class
    /// construction and the native type constructors.
    pub(super) fn call_type_default(&mut self, class: Value, args: CallArgs) -> PyResult<Flow> {
        let (arguments, keyword_arguments) = args.into_parts();
        if class.is_object() && matches!(self.get(class)?, Object::Class { .. }) {
            return self.call_user_class(class, arguments, keyword_arguments, false);
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
        Err(PyError::exception(
            "TypeError",
            "type.__call__ requires a class",
        ))
    }

    /// The name, code, closure, defaults and defining class of a Python function object, or
    /// `None` when `function` is some other value.
    /// Call a native method. A deferred call that suspends returns a retry that passes
    /// `receiver` and the same arguments again; this call passes `native_receiver`.
    fn call_native_method(
        &mut self,
        method: &'static super::super::native::MethodDef,
        receiver: Value,
        native_receiver: Value,
        call: CallArgs,
        mode: CallMode,
    ) -> PyResult<Flow> {
        // Kept as pinned values so a suspension can store them for the retry.
        let retry_arguments = matches!(mode, CallMode::Deferred(_)).then(|| call.clone());
        let previous_suspend = self.native_suspend_allowed;
        self.native_suspend_allowed = self.may_suspend(mode);
        let result = (method.call)(self, native_receiver, call);
        self.native_suspend_allowed = previous_suspend;
        match result.map_err(PyError::into_control) {
            Ok(value) => Ok(self.produce(value)),
            Err(Ok(Control::Exit(status))) => Ok(Flow::Exit(status)),
            Err(Ok(Control::Suspend(reason))) => match (mode, retry_arguments) {
                (CallMode::Deferred(call_span), Some(arguments)) => {
                    let retry = PendingNativeCall::Method {
                        method,
                        receiver: self.store(receiver),
                        arguments: StoredCallArgs::store(self, &arguments),
                        call_span,
                    };
                    Ok(self.suspend(reason, Some(retry)))
                }
                _ => Err("native call suspended outside scheduler dispatch".into()),
            },
            Err(Err(error)) => Err(error),
        }
    }

    /// Apply the default class constructor after any metaclass `__call__` override has had
    /// its turn. Direct `type.__call__` enters here with metaclass dispatch disabled.
    /// `TypeError` when `class` carries a non-empty `__abstractmethods__`, which `abc.ABCMeta`
    /// sets on every class it creates. Only the class's own attribute is consulted, so ordinary
    /// instantiation costs one hash lookup.
    fn reject_abstract_instantiation(&mut self, class: Value, name: &str) -> PyResult<()> {
        let Object::Class(class_object) = self.get(class)? else {
            return Ok(());
        };
        let Some(abstract_methods) = class_object.attributes.get("__abstractmethods__") else {
            return Ok(());
        };
        let abstract_methods = self.value(abstract_methods);
        if !abstract_methods.is_object() {
            return Ok(());
        }
        let names = match self.get(abstract_methods)? {
            Object::Set(members) | Object::FrozenSet(members) => self.values(members),
            _ => return Ok(()),
        };
        if names.is_empty() {
            return Ok(());
        }
        let mut names = names
            .iter()
            .filter_map(|value| {
                string::string_value(&self.state.heap, *value)
                    .ok()
                    .flatten()
            })
            .map(|name| format!("'{name}'"))
            .collect::<Vec<_>>();
        names.sort();
        let plural = if names.len() == 1 {
            "method"
        } else {
            "methods"
        };
        Err(PyError::exception(
            "TypeError",
            format!(
                "Can't instantiate abstract class {name} without an implementation for abstract {plural} {}",
                names.join(", ")
            ),
        ))
    }

    fn call_user_class(
        &mut self,
        class: Value,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
        dispatch_metaclass: bool,
    ) -> PyResult<Flow> {
        let Object::Class(class_object) = self.get(class)? else {
            return Err("type.__call__ requires a class".into());
        };
        let name = class_object.name.clone();
        let metaclass = self.value(&class_object.metaclass);
        let layout = class_object.layout;
        let exception_base = class_object.exception_base;
        let is_dataclass = class_object.is_dataclass;
        let dataclass_fields = class_object
            .dataclass_fields
            .iter()
            .map(|(field, default)| (field.clone(), self.value_optional(default.as_ref())))
            .collect::<Vec<_>>();
        let enum_members = self.values(&class_object.enum_members);
        if dispatch_metaclass && metaclass.is_object() {
            if let Some((owner, descriptor)) = self.class_attribute_entry(metaclass, "__call__")? {
                let callable = self.bind_descriptor(descriptor, Some(class), metaclass, owner)?;
                return self.invoke_call(callable, arguments, keyword_arguments);
            }
        }
        self.reject_abstract_instantiation(class, &name)?;
        if !enum_members.is_empty() {
            if !keyword_arguments.is_empty() || arguments.len() != 1 {
                return Err(format!("{name}() expects one value").into());
            }
            let value_symbol = self.intern_symbol("_value_")?;
            for member in enum_members {
                let value = self
                    .attribute_by_symbol(member, value_symbol)?
                    .ok_or("enum member has no value")?;
                if self.values_equal(&value, &arguments[0])? {
                    return Ok(self.produce(member));
                }
            }
            let rendered = self.repr_value(&arguments[0])?;
            return Err(PyError::exception(
                "ValueError",
                format!("{rendered} is not a valid {name}"),
            ));
        }
        if layout != ClassLayout::Type && exception_base.is_none() && !is_dataclass {
            let class_type = self
                .class_type_id(&class)?
                .ok_or("class has no registered type")?;
            let use_type_constructor = layout == ClassLayout::Object;
            if use_type_constructor || self.class_attribute_entry(class, "__new__")?.is_some() {
                let (owner, constructor) = self
                    .type_lookup(class_type, "__new__")?
                    .ok_or("class constructor has no descriptor")?;
                return self.construct_with_new(
                    class,
                    class_type,
                    owner,
                    constructor,
                    arguments,
                    keyword_arguments,
                );
            }
        }
        // The builtin value an instance of a builtin subclass carries.
        let builtin_payload = match layout {
            ClassLayout::Object => None,
            // Mutable builtin `__new__` creates empty storage. An override can then
            // populate it, usually through the base `__init__`.
            ClassLayout::Builtin(
                builtin @ (BuiltinType::List
                | BuiltinType::Set
                | BuiltinType::Dict
                | BuiltinType::ByteArray),
            ) if self.class_attribute(class, "__init__")?.is_some() => {
                Some(self.builtin_value(builtin, Vec::new(), Vec::new())?)
            }
            ClassLayout::Builtin(builtin) => {
                Some(self.builtin_value(builtin, arguments.clone(), keyword_arguments.clone())?)
            }
            ClassLayout::Type => {
                let created = if let Some((owner, constructor)) =
                    self.class_attribute_entry(class, "__new__")?
                {
                    let constructor =
                        self.bind_descriptor(constructor, Some(class), class, owner)?;
                    let flow = self.invoke_call(
                        constructor,
                        arguments.clone(),
                        keyword_arguments.clone(),
                    )?;
                    match self.immediate_value(flow)? {
                        Ok(value) => value,
                        Err(flow) => return Ok(flow),
                    }
                } else {
                    if !keyword_arguments.is_empty() || arguments.len() != 3 {
                        return Err("type construction expects name, bases, and namespace".into());
                    }
                    let name = string::string_value(&self.state.heap, arguments[0])?
                        .ok_or("type name must be a string")?;
                    self.new_type(class, name, arguments[1], arguments[2])?
                };
                if created.is_object() && matches!(self.get(created), Ok(Object::Class { .. })) {
                    if let Some((owner, initializer)) =
                        self.class_attribute_entry(class, "__init__")?
                    {
                        let initializer =
                            self.bind_descriptor(initializer, Some(created), class, owner)?;
                        let flow = self.invoke_call(initializer, arguments, keyword_arguments)?;
                        let result = match self.immediate_value(flow)? {
                            Ok(value) => value,
                            Err(flow) => return Ok(flow),
                        };
                        if !result.is_none() {
                            return Err("metaclass __init__() should return None".into());
                        }
                    }
                }
                return Ok(self.produce(created));
            }
        };
        let instance_type = self
            .class_type_id(&class)?
            .ok_or("class has no registered type")?;
        let instance = if exception_base.is_some() {
            let args = arguments.clone();
            self.allocate_typed(instance_type, Object::Exception(Ref::all(args)))?
        } else {
            let payload = match builtin_payload {
                Some(value) => self.state.heap.copy_builtin_payload(value)?,
                None => Object::Bare,
            };
            self.allocate_typed(instance_type, payload)?
        };
        if is_dataclass {
            let mut values = Vec::new();
            for (index, (field, default)) in dataclass_fields.iter().enumerate() {
                if index < arguments.len()
                    && keyword_arguments.iter().any(|(name, _)| name == field)
                {
                    return Err(
                        format!("{name}() got multiple values for argument {field:?}").into(),
                    );
                }
                let value = keyword_arguments
                    .iter()
                    .find(|(name, _)| name == field)
                    .map(|(_, value)| *value)
                    .or_else(|| arguments.get(index).copied())
                    .or(*default)
                    .ok_or_else(|| format!("{name}() missing required argument: {field:?}"))?;
                if keyword_arguments
                    .iter()
                    .filter(|(name, _)| name == field)
                    .count()
                    > 1
                {
                    return Err(
                        format!("{name}() got multiple values for argument {field:?}").into(),
                    );
                }
                values.push((field.clone(), value));
            }
            if arguments.len() > dataclass_fields.len() {
                return Err(format!(
                    "{name}() takes {} positional arguments but {} were given",
                    dataclass_fields.len(),
                    arguments.len()
                )
                .into());
            }
            for (field, _) in &keyword_arguments {
                if !dataclass_fields.iter().any(|(name, _)| name == field) {
                    return Err(
                        format!("{name}() got an unexpected keyword argument {field:?}").into(),
                    );
                }
            }
            self.with_attributes(|store| store.extend(instance, values))?;
        } else if let Some(initializer) = self.class_attribute(class, "__init__")? {
            if !initializer.is_object() {
                return Err(format!("{name}.__init__ is not callable").into());
            }
            if !matches!(self.get(initializer)?, Object::Function(_)) {
                return Err(format!("{name}.__init__ is not a function").into());
            }
            let positional = arguments.len();
            self.push(initializer);
            for argument in arguments {
                self.push(argument);
            }
            let mut keywords = Vec::with_capacity(keyword_arguments.len());
            for (keyword, value) in keyword_arguments {
                keywords.push(Some(Arc::from(keyword)));
                self.push(value);
            }
            let flow = self.call_python(
                initializer,
                Some(instance),
                positional,
                &keywords,
                CallMode::Immediate,
            )?;
            match self.immediate_value(flow)? {
                Ok(value) if value.is_none() => {}
                Ok(_) => return Err("__init__() should return None".into()),
                Err(flow) => return Ok(flow),
            }
        } else if exception_base.is_some() && !keyword_arguments.is_empty() {
            return Err(format!("{name}() does not accept keyword arguments").into());
        } else if layout == ClassLayout::Object
            && exception_base.is_none()
            && (!arguments.is_empty() || !keyword_arguments.is_empty())
        {
            return Err(format!("{name}() takes no arguments").into());
        }
        Ok(self.produce(instance))
    }

    pub(super) const MAX_CALL_DEPTH: usize = 256;

    /// Call the Python `function` whose arguments sit on top of the operand stack above the
    /// callee slot: `positional` values, then one value per `keywords` entry. `receiver`
    /// becomes the first argument. Binding validates the call against the signature first,
    /// then moves the stored references straight into the new frame's slots on the shared
    /// locals stack, so a call with no `*args` or `**kwargs` parameter allocates nothing. A
    /// generator or coroutine, or code whose nested scopes read its locals, then moves the
    /// bound slots into a heap scope.
    fn call_python(
        &mut self,
        function: Value,
        receiver: Option<Value>,
        positional: usize,
        keywords: &[KeywordName],
        mode: CallMode,
    ) -> PyResult<Flow> {
        let Object::Function(function_object) = self.get(function)? else {
            return Err("bound descriptor is not callable".into());
        };
        let code = function_object.code.clone();
        let closure = self.value_optional(function_object.closure.as_ref());
        let signature = &code.call_signature;
        let suspends = signature.is_generator || signature.is_coroutine;
        if !suspends && self.call_depth == Self::MAX_CALL_DEPTH {
            return Err(PyError::exception(
                "RecursionError",
                "maximum recursion depth exceeded",
            ));
        }
        let locals_base = self.execution.locals.len();
        if let Err(error) =
            self.bind_arguments(function, &code, receiver, positional, keywords, locals_base)
        {
            self.execution.locals.truncate(locals_base);
            return Err(error);
        }
        if suspends || signature.heap_locals {
            let locals = self.execution.locals[locals_base..]
                .iter()
                .map(|slot| self.value_optional(slot.as_ref()))
                .collect::<Vec<_>>();
            self.execution.locals.truncate(locals_base);
            if suspends {
                return self.create_generator(function, &code, closure, locals);
            }
            let scope = self.function_scope(&code, closure, locals)?;
            let entry = FrameEntry::function(function, scope);
            return self.enter_python_function(function, &code, entry, mode);
        }
        let entry = FrameEntry::with_locals(function, closure, locals_base);
        self.enter_python_function(function, &code, entry, mode)
    }

    /// Bind the call operands on the stack into `code`'s local slots at `locals_base` on the
    /// locals stack, consuming the operands and the callee slot. Every `TypeError` the
    /// signature can raise is detected before the stack changes; the caller discards the
    /// partially extended locals on error.
    fn bind_arguments(
        &mut self,
        function: Value,
        code: &CodeRef,
        receiver: Option<Value>,
        positional: usize,
        keywords: &[KeywordName],
        locals_base: usize,
    ) -> PyResult<()> {
        let signature = &code.call_signature;
        let receiver_offset = usize::from(receiver.is_some());
        let given = positional + receiver_offset;
        let count = positional + keywords.len();
        let arguments_start = self.stack.len() - count;
        if given > signature.positional_count && signature.variadic_slot.is_none() {
            return Err(self.too_many_positional(function, code, given, keywords));
        }
        // Slots the positional arguments fill; the rest of them go to `*args`.
        let filled = given.min(signature.positional_count);
        // Validate the keywords and the required parameters before touching the stack.
        let mut extra_keywords = 0usize;
        for (index, keyword) in keywords.iter().enumerate() {
            let name = keyword
                .as_deref()
                .ok_or("invalid keyword argument metadata")?;
            let repeated = keywords[..index]
                .iter()
                .any(|earlier| earlier.as_deref() == Some(name));
            match keyword_target(code, name) {
                KeywordTarget::Slot(slot) if slot < filled || repeated => {
                    let message = format!(
                        "{}() got multiple values for argument '{name}'",
                        self.function_name(&self.store(function))
                    );
                    return Err(PyError::exception("TypeError", message));
                }
                KeywordTarget::Slot(_) => {}
                // A name that binds no slot, including a positional-only parameter's, lands in
                // `**kwargs` when the signature has one.
                KeywordTarget::Unknown | KeywordTarget::PositionalOnly
                    if signature.keyword_variadic_slot.is_some() =>
                {
                    if repeated {
                        let message = format!(
                            "{}() got multiple values for argument '{name}'",
                            self.function_name(&self.store(function))
                        );
                        return Err(PyError::exception("TypeError", message));
                    }
                    extra_keywords += 1;
                }
                KeywordTarget::Unknown => {
                    let message = format!(
                        "{}() got an unexpected keyword argument '{name}'",
                        self.function_name(&self.store(function))
                    );
                    return Err(PyError::exception("TypeError", message));
                }
                KeywordTarget::PositionalOnly => {
                    let message = format!(
                        "{}() got some positional-only arguments passed as keyword \
                         arguments: '{name}'",
                        self.function_name(&self.store(function))
                    );
                    return Err(PyError::exception("TypeError", message));
                }
            }
        }
        if given < signature.required_positional
            || signature.keyword_only_required
            || !keywords.is_empty()
        {
            self.check_required_parameters(function, code, filled, keywords)?;
        }
        // Allocate the variadic containers from the stack values, which stay rooted meanwhile.
        let variadic = match signature.variadic_slot {
            Some(_) if given > signature.positional_count => {
                let first_extra = signature.positional_count.saturating_sub(receiver_offset);
                let mut extras = Vec::with_capacity(given - signature.positional_count);
                if signature.positional_count == 0 {
                    extras.extend(receiver);
                }
                for index in first_extra..positional {
                    extras.push(self.peek(count - 1 - index)?);
                }
                self.reserve_result(extras.len().saturating_mul(64))?;
                Some(self.alloc(Object::Tuple(Ref::all(extras)))?)
            }
            Some(_) => Some(self.alloc(Object::Tuple(Ref::all([])))?),
            None => None,
        };
        let keyword_variadic = match signature.keyword_variadic_slot {
            Some(_) => {
                let mut entries = Vec::with_capacity(extra_keywords);
                for (index, keyword) in keywords.iter().enumerate() {
                    let name = keyword.as_deref().expect("keywords were validated above");
                    if !matches!(keyword_target(code, name), KeywordTarget::Slot(_)) {
                        let key = self.allocate_string(name.to_string())?;
                        self.reserve_result(64)?;
                        entries.push((key, self.peek(keywords.len() - 1 - index)?));
                    }
                }
                Some(self.allocate_dict(entries)?)
            }
            None => None,
        };
        // Move the operands into the slots.
        let receiver = receiver.map(|receiver| self.store(receiver));
        let variadic = variadic.map(|value| self.store(value));
        let keyword_variadic = keyword_variadic.map(|value| self.store(value));
        let super::VmState { locals, stack, .. } = &mut *self.execution;
        locals.extend(std::iter::repeat_with(|| None).take(code.local_names.len()));
        let slots = &mut locals[locals_base..];
        if let (Some(receiver), true) = (receiver, signature.positional_count > 0) {
            slots[0] = Some(receiver);
        }
        for (index, value) in stack.drain_refs(arguments_start).enumerate() {
            if index < positional {
                let slot = index + receiver_offset;
                if slot < signature.positional_count {
                    slots[slot] = Some(value);
                }
            } else {
                let name = keywords[index - positional]
                    .as_deref()
                    .expect("keywords were validated above");
                if let KeywordTarget::Slot(slot) = keyword_target(code, name) {
                    slots[slot] = Some(value);
                }
            }
        }
        stack.pop_ref();
        if let (Some(slot), Some(value)) = (signature.variadic_slot, variadic) {
            slots[slot] = Some(value);
        }
        if let (Some(slot), Some(value)) = (signature.keyword_variadic_slot, keyword_variadic) {
            slots[slot] = Some(value);
        }
        let Object::Function(function_object) = self.state.heap.get(function)? else {
            unreachable!("function was checked by the caller and remains a function");
        };
        if function_object.defaults.len() != signature.default_slots.len() {
            return Err(format!(
                "{}() has invalid default argument metadata",
                function_object.name
            )
            .into());
        }
        for (&slot, default) in signature
            .default_slots
            .iter()
            .zip(&function_object.defaults)
        {
            let local = &mut self.execution.locals[locals_base + slot];
            if local.is_none() {
                *local = Some(default.dup());
            }
        }
        Ok(())
    }

    /// The `TypeError` for a call that passes more positional arguments than the signature
    /// takes, worded as CPython words it.
    fn too_many_positional(
        &mut self,
        function: Value,
        code: &CodeRef,
        given: usize,
        keywords: &[KeywordName],
    ) -> PyError {
        let count = code.call_signature.positional_count;
        let required = code.parameters[..count]
            .iter()
            .filter(|parameter| !parameter.has_default)
            .count();
        let takes = if required == count {
            format!(
                "{count} positional argument{}",
                if count == 1 { "" } else { "s" }
            )
        } else {
            format!("from {required} to {count} positional arguments")
        };
        let keyword_only = keywords
            .iter()
            .filter(|keyword| {
                code.parameters.iter().any(|parameter| {
                    Some(parameter.name.as_str()) == keyword.as_deref()
                        && parameter.kind == ParameterKind::KeywordOnly
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
        let message = format!(
            "{}() takes {takes} but {given}{keyword_detail} {verb} given",
            self.function_name(&self.store(function))
        );
        PyError::exception("TypeError", message)
    }

    /// Raise the `TypeError` for parameters without a default that neither the `filled`
    /// positional slots nor `keywords` supply.
    fn check_required_parameters(
        &mut self,
        function: Value,
        code: &CodeRef,
        filled: usize,
        keywords: &[KeywordName],
    ) -> PyResult<()> {
        let missing = |keyword_only: bool| {
            code.parameters
                .iter()
                .enumerate()
                .filter(|(slot, parameter)| {
                    let kind_matches = match parameter.kind {
                        ParameterKind::PositionalOnly | ParameterKind::Positional => !keyword_only,
                        ParameterKind::KeywordOnly => keyword_only,
                        ParameterKind::Variadic | ParameterKind::KeywordVariadic => false,
                    };
                    let supplied = *slot < filled
                        || keywords
                            .iter()
                            .any(|keyword| keyword.as_deref() == Some(parameter.name.as_str()));
                    kind_matches && !supplied && !parameter.has_default
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
                "{}() missing {} required {description} argument{plural}: {}",
                self.function_name(&self.store(function)),
                names.len(),
                english_list(&names)
            );
            return Err(PyError::exception("TypeError", message));
        }
        Ok(())
    }

    /// Run `code` for `function` in a frame whose locals are already bound. A deferred call
    /// leaves the frame installed for the dispatch loop; an immediate call runs it to
    /// completion here.
    fn enter_python_function(
        &mut self,
        function: Value,
        code: &CodeRef,
        entry: FrameEntry,
        mode: CallMode,
    ) -> PyResult<Flow> {
        self.call_depth += 1;
        if let CallMode::Deferred(_) = mode {
            let stack_base = self.stack.len();
            let frame = self.enter_frame(code, 0, stack_base, entry, FrameKind::Call)?;
            self.bytecode_frames.push(frame);
            return Ok(Flow::Refresh);
        }
        let result = self.execute_code(code, entry);
        self.call_depth -= 1;
        match result {
            Ok(Flow::Return(value)) => {
                self.execution.stack.push_ref(&value);
                Ok(Flow::Next)
            }
            Ok(Flow::Halt) => Ok(self.produce(Value::None)),
            Ok(Flow::Yield(_)) => Err(format!(
                "unexpected yield in ordinary function {}",
                self.function_name(&self.store(function))
            )
            .into()),
            Ok(Flow::Exit(status)) => Ok(Flow::Exit(status)),
            Ok(flow) => unreachable!("an immediate call cannot end with {flow:?}"),
            Err((error, span)) => Err(error.located(|| {
                format!(
                    " in {} at line {}, column {}",
                    self.function_name(&self.store(function)),
                    span.line,
                    span.column
                )
            })),
        }
    }

    /// The `__name__` of a function object, for tracebacks and error messages.
    pub(super) fn function_name(&self, function: &super::super::heap::Ref) -> String {
        match self.state.heap.get(self.value(function)) {
            Ok(Object::Function(function_object)) => function_object.name.clone(),
            _ => "<function>".to_string(),
        }
    }

    fn create_generator(
        &mut self,
        function: Value,
        code: &CodeRef,
        closure: Option<Value>,
        locals: Vec<Option<Value>>,
    ) -> PyResult<Flow> {
        let scope = self.function_scope(code, closure, locals)?;
        let generator = self.alloc({
            Object::Generator(Box::new(GeneratorObject {
                function: Ref::from(function),
                code: code.clone(),
                scope: Ref::from(scope),
                instruction_pointer: 0,
                handlers: Vec::new(),
                contexts: Vec::new(),
                exceptions: Vec::new(),
                stack: Vec::new(),
                exhausted: false,
                running: false,
                return_value: Ref::from(Value::None),
            }))
        })?;
        Ok(self.produce(generator))
    }

    /// The frame entry for `exec`/`eval` code: no locals of its own, names resolved through the
    /// calling frame's scope. A caller that keeps its locals in the frame exposes a snapshot of
    /// them, which the dynamic code can read but, as in CPython, not rebind.
    fn dynamic_code_entry(&mut self) -> PyResult<FrameEntry> {
        let enclosing = self.lookup_scope();
        let Some((base, code)) = self
            .bytecode_frames
            .last()
            .and_then(|frame| Some((frame.locals_base()?, frame.code.clone())))
        else {
            return Ok(FrameEntry::dynamic(enclosing));
        };
        let locals = (0..code.local_names.len())
            .map(|slot| self.value_optional(self.locals.get(base + slot)?.as_ref()))
            .collect();
        let snapshot = self.function_scope(&code, enclosing, locals)?;
        Ok(FrameEntry::dynamic(Some(snapshot)))
    }

    /// Allocate the heap scope of an activation whose locals must outlive the frame or be
    /// visible to nested scopes.
    fn function_scope(
        &mut self,
        code: &CodeRef,
        closure: Option<Value>,
        locals: Vec<Option<Value>>,
    ) -> PyResult<Value> {
        let uses_repl_globals = closure
            .map(|scope| scopes::uses_repl_globals(self.heap(), scope))
            .transpose()?
            .unwrap_or(true);
        self.alloc_scope(
            closure,
            uses_repl_globals,
            code.local_names.clone(),
            locals,
            HashMap::new(),
        )
    }

    /// Instantiate a class whose MRO defines `__new__`, as `type.__call__` does: call `__new__`
    /// with the class and the call's arguments, then run `__init__` with the same arguments
    /// when the result is an instance of the class. Any other result is returned as is.
    fn construct_with_new(
        &mut self,
        class: Value,
        class_type: super::super::object_model::TypeId,
        owner: super::super::object_model::TypeId,
        constructor: Value,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> PyResult<Flow> {
        // `__new__` is a static method that receives the class explicitly.
        let constructor = self
            .bind_type_attribute(constructor, None, class_type, owner)?
            .ok_or("__new__ descriptor has no value")?;
        let mut new_arguments = Vec::with_capacity(arguments.len().saturating_add(1));
        new_arguments.push(class);
        new_arguments.extend(arguments.iter().copied());
        let flow = self.invoke_call(constructor, new_arguments, keyword_arguments.clone())?;
        let created = match self.immediate_value(flow)? {
            Ok(value) => value,
            Err(flow) => return Ok(flow),
        };
        if !self.is_instance(&created, &class)? {
            return Ok(self.produce(created));
        }
        let layout = match self.get(class)? {
            Object::Class(class_object) => class_object.layout,
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
                    self.class_type_id(&owner)
                        .and_then(|owner| owner.ok_or("initializer class has no type".into()))
                        .map(|owner| (owner, value))
                })
                .transpose()?
        };
        let Some((owner, initializer)) = initializer else {
            return Ok(self.produce(created));
        };
        let initializer = self
            .bind_type_attribute(initializer, Some(created), class_type, owner)?
            .ok_or("__init__ descriptor has no value")?;
        let flow = self.invoke_call(initializer, arguments, keyword_arguments)?;
        match self.immediate_value(flow)? {
            Ok(value) if value.is_none() => Ok(self.produce(created)),
            Ok(value) => {
                let type_name = self.type_name_of(&value)?;
                Err(PyError::exception(
                    "TypeError",
                    format!("__init__() should return None, not '{type_name}'"),
                ))
            }
            Err(flow) => Ok(flow),
        }
    }

    /// The value `builtin(*arguments, **keyword_arguments)` constructs, such as the tuple that
    /// `tuple(iterable)` builds.
    fn builtin_value(
        &mut self,
        builtin: BuiltinType,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> PyResult<Value> {
        match self.call_builtin_type(builtin, arguments, keyword_arguments)? {
            Flow::Next => self.pop(),
            _ => Err(format!("{}() did not produce a value", builtin.name()).into()),
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
    ) -> PyResult<Value> {
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
                return Err(PyError::exception("TypeError", message));
            }
            _ => match class.is_object().then(|| self.get(class)) {
                Some(Ok(Object::Class(class_object)))
                    if class_object.layout == ClassLayout::Builtin(builtin) =>
                {
                    Some(class)
                }
                Some(Ok(Object::Class(class_object))) => {
                    let class_name = &class_object.name;
                    let message = format!(
                        "{name}.__new__({class_name}): {class_name} is not a subtype of {name}"
                    );
                    return Err(PyError::exception("TypeError", message));
                }
                _ => {
                    let type_name = self.type_name_of(&class)?;
                    let message =
                        format!("{name}.__new__(X): X is not a type object ({type_name})");
                    return Err(PyError::exception("TypeError", message));
                }
            },
        };
        let value = self.builtin_value(builtin, arguments, keyword_arguments)?;
        let Some(class) = subclass else {
            return Ok(value);
        };
        let instance_type = self
            .class_type_id(&class)?
            .ok_or("builtin subclass has no registered type")?;
        let payload = self.state.heap.copy_builtin_payload(value)?;
        self.allocate_typed(instance_type, payload)
    }

    /// `object.__new__(class)`: a new instance of `class` with no attributes set.
    ///
    /// As in CPython, extra arguments are an error when the class overrides `__new__` (they were
    /// meant for it) or when it keeps `object.__init__` (nothing would accept them). A class whose
    /// instances have a builtin layout, such as an `int` or exception subclass, must be created
    /// by that builtin's `__new__`.
    pub(super) fn new_instance(&mut self, class: Value, has_arguments: bool) -> PyResult<Value> {
        if matches!(
            class.native_value(),
            Some(NativeValue::BuiltinType(BuiltinType::Object))
        ) {
            if has_arguments {
                return Err(PyError::exception(
                    "TypeError",
                    "object() takes no arguments",
                ));
            }
            return self.alloc(Object::Bare);
        }
        let builtin = match class.native_value() {
            Some(NativeValue::BuiltinType(builtin)) => Some(builtin.name()),
            Some(NativeValue::ExceptionType(ExceptionType(name))) => Some(name),
            _ => None,
        };
        if let Some(name) = builtin {
            return Err(PyError::exception(
                "TypeError",
                format!("object.__new__({name}) is not safe, use {name}.__new__()"),
            ));
        }
        let Some((name, layout, exception_base)) = class
            .is_object()
            .then(|| match self.get(class) {
                Ok(Object::Class(class_object)) => Some((
                    class_object.name.clone(),
                    class_object.layout,
                    class_object.exception_base,
                )),
                _ => None,
            })
            .flatten()
        else {
            let type_name = self.type_name_of(&class)?;
            return Err(PyError::exception(
                "TypeError",
                format!("object.__new__(X): X is not a type object ({type_name})"),
            ));
        };
        if layout != ClassLayout::Object || exception_base.is_some() {
            return Err(PyError::exception(
                "TypeError",
                format!("object.__new__({name}) is not safe, use {name}.__new__()"),
            ));
        }
        if has_arguments {
            if self.class_attribute_entry(class, "__new__")?.is_some() {
                return Err(PyError::exception(
                    "TypeError",
                    "object.__new__() takes exactly one argument (the type to instantiate)",
                ));
            }
            if self.class_attribute_entry(class, "__init__")?.is_none() {
                return Err(PyError::exception(
                    "TypeError",
                    format!("{name}() takes no arguments"),
                ));
            }
        }
        let instance_type = self
            .class_type_id(&class)?
            .ok_or("class has no registered type")?;
        self.allocate_typed(instance_type, Object::Bare)
    }

    /// Advance any Python iterator by one item; `Ok(None)` means a builtin iterator is exhausted.
    /// A user iterator's `StopIteration` stays an error, so callers that treat it as exhaustion
    /// [`catch`](Self::catch) it.
    pub(super) fn iterator_next(&mut self, iterator: &Value) -> PyResult<Option<Value>> {
        if !iterator.is_object() {
            return Err(self.raise_object_type_error(iterator, "is not an iterator"));
        }
        let iterator = *iterator;
        // Copy out only the fields a step needs. Cloning a materialized iterator, generator or
        // instance payload on every step would make iteration quadratic in host time.
        enum Step {
            Callable {
                callable: Value,
                sentinel: Value,
                exhausted: bool,
            },
            Count {
                current: i64,
                step: i64,
            },
            Protocol,
        }
        let step = match self.get(iterator)? {
            Object::CallableIterator {
                callable,
                sentinel,
                exhausted,
            } => Step::Callable {
                callable: self.value(callable),
                sentinel: self.value(sentinel),
                exhausted: *exhausted,
            },
            Object::CountIterator { current, step } => Step::Count {
                current: *current,
                step: *step,
            },
            Object::Generator { .. } => return self.resume_generator(iterator),
            Object::Iterator { .. }
            | Object::SequenceIterator { .. }
            | Object::ReverseIterator { .. }
            | Object::RangeIterator { .. }
            | Object::StreamIterator { .. } => return self.next_stored_iterator(iterator),
            _ => Step::Protocol,
        };
        Ok(match step {
            Step::Callable {
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
                    )?;
                    if self.values_equal(&value, &sentinel)? {
                        if let Object::CallableIterator { exhausted, .. } =
                            self.get_mut(iterator)?
                        {
                            *exhausted = true;
                        }
                        None
                    } else {
                        Some(value)
                    }
                }
            }
            Step::Count { current, step } => {
                let next = super::super::stdlib::itertools::count_next(current, step)
                    .map_err(str::to_string)?;
                if let Object::CountIterator { current, .. } = self.get_mut(iterator)? {
                    *current = next;
                }
                Some(Value::Int(current))
            }
            Step::Protocol => {
                match self.invoke_slot(&iterator, Slot::Next, "__next__", Vec::new())? {
                    Some(value) => Some(value),
                    None => {
                        return Err(self.raise_object_type_error(&iterator, "is not an iterator"))
                    }
                }
            }
        })
    }

    /// Make the exception `error` describes the pending exception, allocating its object, and
    /// return the pending signal. A pending exception and a stop that is not an exception come
    /// back unchanged.
    pub(super) fn raise_error(&mut self, error: PyError) -> PyError {
        let Some((kind, message)) = error.into_parts() else {
            return PyError::pending();
        };
        let kind = match kind {
            PyErrorKind::Exception(kind) => kind,
            PyErrorKind::OsError { errno, filename } => {
                let kind = super::super::exception_types::os_error_subclass(errno);
                let mut args = vec![Value::Int(i64::from(errno))];
                for text in std::iter::once(message).chain(filename) {
                    match self.allocate_string(text) {
                        Ok(text) => args.push(text),
                        Err(error) => return error,
                    }
                }
                return self.raise_exception_args(kind, args);
            }
            kind => return PyError::new(kind, message),
        };
        match self.allocate_exception(kind, message) {
            Ok(value) => self.raise_value(value),
            Err(error) => error,
        }
    }

    /// Catch the exception `error` stands for: clear it and return its object. A stop that is
    /// not a Python exception comes back as the error.
    pub(super) fn take_exception(&mut self, error: PyError) -> PyResult<Value> {
        let error = self.raise_error(error);
        if !error.is_pending() {
            return Err(error);
        }
        let exception = self
            .pending_exception
            .take()
            .ok_or("exception unwound without a pending exception")?;
        Ok(self.value(&exception))
    }

    /// Catch `error` if it is an exception of the builtin class `kind` or a subclass, as
    /// `except kind:` would; any other error comes back. An exception not yet raised is matched
    /// by its class name without allocating its object.
    pub(super) fn catch(&mut self, error: PyError, kind: &str) -> PyResult<()> {
        if let Some(PyErrorKind::Exception(name)) = error.kind() {
            if super::super::exception_types::exception_is_subclass(name, kind) {
                return Ok(());
            }
            return Err(error);
        }
        if error.is_pending() && self.pending_exception_is(kind) {
            self.pending_exception = None;
            return Ok(());
        }
        Err(error)
    }

    /// Whether a native called in `mode` may suspend its process: only a deferred call made
    /// by scheduler-dispatched bytecode, outside any synchronous execution.
    fn may_suspend(&self, mode: CallMode) -> bool {
        matches!(mode, CallMode::Deferred(_)) && self.synchronous_frames == 0
    }

    pub(super) fn resume_native_call(&mut self, pending: PendingNativeCall) -> PyResult<Flow> {
        /// The native a retried call invokes, with its receiver pinned.
        #[derive(Clone, Copy)]
        enum Target {
            Function(&'static super::FunctionDef),
            Method(&'static super::super::native::MethodDef, Value),
        }
        if matches!(pending, PendingNativeCall::Input { .. }) {
            return self.resume_input(pending);
        }
        let call_span = pending.call_span();
        // Pin the stored arguments before anything can allocate.
        let (target, arguments) = match pending {
            PendingNativeCall::Function {
                function,
                arguments,
                ..
            } => (Target::Function(function), arguments.load(self)),
            PendingNativeCall::Method {
                method,
                receiver,
                arguments,
                ..
            } => (
                Target::Method(method, self.value(&receiver)),
                arguments.load(self),
            ),
            PendingNativeCall::Input { .. } => unreachable!("handled above"),
        };
        let retry_arguments = arguments.clone();
        let previous_suspend = self.native_suspend_allowed;
        self.native_suspend_allowed = self.synchronous_frames == 0;
        let result = match target {
            Target::Function(function) => (function.call)(self, arguments),
            Target::Method(method, receiver) => (method.call)(self, receiver, arguments),
        };
        self.native_suspend_allowed = previous_suspend;
        match result.map_err(PyError::into_control) {
            Ok(value) => Ok(self.native_result(value)),
            Err(Ok(Control::Exit(status))) => Ok(Flow::Exit(status)),
            Err(Ok(Control::Suspend(reason))) => {
                let arguments = StoredCallArgs::store(self, &retry_arguments);
                let retry = match target {
                    Target::Function(function) => PendingNativeCall::Function {
                        function,
                        arguments,
                        call_span,
                    },
                    Target::Method(method, receiver) => PendingNativeCall::Method {
                        method,
                        receiver: self.store(receiver),
                        arguments,
                        call_span,
                    },
                };
                Ok(self.suspend(reason, Some(retry)))
            }
            Err(Err(error)) => Err(error),
        }
    }

    /// A native function's result goes on the stack; a wait it requested afterwards, through
    /// `pending_wait`, blocks the process once the result is in place.
    fn native_result(&mut self, value: Value) -> Flow {
        self.push(value);
        match self.pending_wait.take() {
            Some(reason) => self.suspend(reason, None),
            None => Flow::Next,
        }
    }

    fn resume_input(&mut self, pending: PendingNativeCall) -> PyResult<Flow> {
        let PendingNativeCall::Input { call_span } = pending else {
            unreachable!("caller checked the variant");
        };
        let marker = Value::Native(NativeValue::Stream(Stream::Stdin));
        let previous_suspend = self.native_suspend_allowed;
        self.native_suspend_allowed = self.synchronous_frames == 0;
        let result = self.read_stream(&marker, None, true);
        self.native_suspend_allowed = previous_suspend;
        match result.map_err(PyError::into_control) {
            Ok(read) => {
                let value = self.finish_input(read)?;
                Ok(self.produce(value))
            }
            Err(Ok(Control::Exit(status))) => Ok(Flow::Exit(status)),
            Err(Ok(Control::Suspend(reason))) => {
                Ok(self.suspend(reason, Some(PendingNativeCall::Input { call_span })))
            }
            Err(Err(error)) => Err(error),
        }
    }

    /// Strip the trailing newline `input()` reads and raise `EOFError` on an empty read.
    fn finish_input(&mut self, read: PyStreamRead) -> PyResult<Value> {
        let mut text = match read {
            PyStreamRead::Text(text) => text,
            PyStreamRead::Bytes(_) => {
                unreachable!("input() only reads the modeled text stdin stream")
            }
        };
        if text.is_empty() {
            return Err(PyError::exception("EOFError", "EOF when reading a line"));
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

/// Where a keyword argument lands in a signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeywordTarget {
    /// The local slot of a parameter the keyword may bind.
    Slot(usize),
    /// A positional-only parameter, which a keyword cannot bind.
    PositionalOnly,
    /// No parameter of that name; `**kwargs` collects it when the signature has one.
    Unknown,
}

fn keyword_target(code: &CodeRef, name: &str) -> KeywordTarget {
    match code
        .parameters
        .iter()
        .position(|parameter| parameter.name == name)
    {
        Some(slot) => match code.parameters[slot].kind {
            ParameterKind::Positional | ParameterKind::KeywordOnly => KeywordTarget::Slot(slot),
            ParameterKind::PositionalOnly => KeywordTarget::PositionalOnly,
            ParameterKind::Variadic | ParameterKind::KeywordVariadic => KeywordTarget::Unknown,
        },
        None => KeywordTarget::Unknown,
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
