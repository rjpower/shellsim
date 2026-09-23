//! Call preparation, callable dispatch, argument binding, and Python frame entry.

use super::{
    expect_arity, protocol, range_length, BigInt, BinaryOperator, Builtin, BytecodeFrame, CallArgs,
    CallMode, CallResult, ClassLayout, CodeRef, Execution, FunctionInvocation, FunctionReturn,
    HashMap, InstanceAttributes, InstancePayload, NativeValue, Object, Ordering, PendingNativeCall,
    PyError, PyErrorKind, PyRuntime, RaisedException, ScopeId, Slot, Stream, Value, Vm,
};
use num_traits::{Signed, Zero};

impl Vm<'_> {
    fn pow_integer_argument(&self, value: &Value) -> Result<(BigInt, usize), PyError> {
        let decimal = <Self as PyRuntime>::integer_text(self, value)?.ok_or_else(|| {
            PyError::type_error("pow() 3rd argument not allowed unless all arguments are integers")
        })?;
        let integer = decimal
            .parse::<BigInt>()
            .map_err(|_| PyError::runtime_error("invalid internal integer representation"))?;
        Ok((integer, decimal.len()))
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
        keyword_names: &[String],
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
        if starred[..positional].iter().any(|expanded| *expanded)
            && keyword_names
                .iter()
                .zip(&starred[positional..])
                .any(|(_, expanded)| *expanded)
        {
            return Err("invalid starred keyword argument metadata".into());
        }
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
        let keyword_arguments = keyword_names
            .iter()
            .cloned()
            .zip(keyword_values)
            .collect::<Vec<_>>();
        let function = self.pop()?;
        if let Some(id) = function.object_id() {
            return match self.state.heap.get(id)?.clone() {
                Object::Function {
                    name,
                    code,
                    closure,
                    defaults,
                    defining_class,
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
                    } else if let Some(NativeValue::NativeMethod(method)) =
                        descriptor.native_value()
                    {
                        let call = CallArgs::new(arguments, keyword_arguments);
                        match (method.call)(self, receiver, call) {
                            Ok(value) => Ok(CallResult::Value(value)),
                            Err(PyError {
                                kind: PyErrorKind::Exit(status),
                                ..
                            }) => Ok(CallResult::Exit(status)),
                            Err(error) => Err(self.record_native_error(error)),
                        }
                    } else {
                        Err("bound descriptor is not callable".into())
                    }
                }
                Object::Instance { class, .. } => {
                    let type_id = self.state.heap.type_id(id)?;
                    if self.state.types.slot(type_id, Slot::Call)?.is_none() {
                        return Err("object is not callable".into());
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
                Object::Class {
                    name,
                    metaclass,
                    layout,
                    exception_base,
                    is_dataclass,
                    dataclass_fields,
                    enum_members,
                    ..
                } => {
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
                            if protocol::equals(&self.state.heap, value, &arguments[0])? {
                                return Ok(CallResult::Value(member));
                            }
                        }
                        return Err(format!("value is not a valid {name}"));
                    }
                    let payload = match layout {
                        ClassLayout::Object => InstancePayload::Object,
                        ClassLayout::Int => {
                            if !keyword_arguments.is_empty() {
                                return Err(format!("{name}() does not accept keyword arguments"));
                            }
                            let value = arguments
                                .first()
                                .map(|value| {
                                    protocol::int_value(&self.state.heap, value)
                                        .or_else(|| value.as_int())
                                        .ok_or_else(|| {
                                            format!("{name}() argument is not supported")
                                        })
                                })
                                .transpose()?
                                .unwrap_or(0);
                            if arguments.len() > 1 {
                                return Err(format!("{name}() expects at most one value"));
                            }
                            InstancePayload::Int(value)
                        }
                        ClassLayout::Type => {
                            let created = if let Some((owner, constructor)) =
                                self.class_attribute_entry(id, "__new__")?
                            {
                                let constructor = self.bind_descriptor(
                                    constructor,
                                    Some(Value::Object(id)),
                                    id,
                                    owner,
                                )?;
                                match self.invoke_call(
                                    constructor,
                                    arguments.clone(),
                                    keyword_arguments.clone(),
                                )? {
                                    CallResult::Value(value) => value,
                                    CallResult::Exit(status) => {
                                        return Ok(CallResult::Exit(status))
                                    }
                                    CallResult::EnteredFrame => {
                                        unreachable!("invoke_call is immediate")
                                    }
                                    CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                                        unreachable!("immediate call cannot suspend")
                                    }
                                }
                            } else {
                                if !keyword_arguments.is_empty() || arguments.len() != 3 {
                                    return Err(
                                        "type construction expects name, bases, and namespace"
                                            .into(),
                                    );
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
                                    let initializer = self.bind_descriptor(
                                        initializer,
                                        Some(created),
                                        id,
                                        owner,
                                    )?;
                                    let result = match self.invoke_call(
                                        initializer,
                                        arguments,
                                        keyword_arguments,
                                    )? {
                                        CallResult::Value(value) => value,
                                        CallResult::Exit(status) => {
                                            return Ok(CallResult::Exit(status))
                                        }
                                        CallResult::EnteredFrame => {
                                            unreachable!("invoke_call is immediate")
                                        }
                                        CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                                            unreachable!("immediate call cannot suspend")
                                        }
                                    };
                                    if !result.is_none() {
                                        return Err(
                                            "metaclass __init__() should return None".into()
                                        );
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
                        let exception_args =
                            self.allocate_object(Object::Tuple(arguments.clone()))?;
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
                                .ok_or_else(|| {
                                    format!("{name}() missing required argument: {field:?}")
                                })?;
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
                            ..
                        } = self.state.heap.get(function)?.clone()
                        else {
                            return Err(format!("{name}.__init__ is not a function"));
                        };
                        arguments.insert(0, instance);
                        match self.call_python_function(
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
                        )? {
                            CallResult::Value(value) if value.is_none() => {}
                            CallResult::Value(_) => {
                                return Err("__init__() should return None".into())
                            }
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
                Object::Module { .. } => Err("module object is not callable".into()),
                _ => Err("object is not callable".into()),
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
            expect_arity(&arguments, 0, 1)?;
            let message = arguments
                .first()
                .map(|value| protocol::display(&self.state.heap, value))
                .transpose()?
                .unwrap_or_default();
            return Ok(CallResult::Value(
                self.allocate_exception(exception_type.0.to_string(), message)?,
            ));
        }
        if let Some(NativeValue::NativeMethod(method)) = function.native_value() {
            if arguments.is_empty() {
                return Err("unbound native method requires a receiver".into());
            }
            let receiver = arguments.remove(0);
            let call = CallArgs::new(arguments, keyword_arguments);
            return match (method.call)(self, receiver, call) {
                Ok(value) => Ok(CallResult::Value(value)),
                Err(PyError {
                    kind: PyErrorKind::Exit(status),
                    ..
                }) => Ok(CallResult::Exit(status)),
                Err(error) => Err(self.record_native_error(error)),
            };
        }
        if let Some(NativeValue::NativeFunction(function)) = function.native_value() {
            let call = CallArgs::new(arguments, keyword_arguments);
            let retry = match mode {
                CallMode::Deferred(call_span) => Some(PendingNativeCall {
                    function,
                    arguments: call.clone(),
                    call_span,
                }),
                CallMode::Immediate => None,
            };
            let previous_suspend = self.native_suspend_allowed;
            self.native_suspend_allowed = matches!(mode, CallMode::Deferred(_));
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
            return Err("object is not callable".into());
        };
        if !keyword_arguments.is_empty() && !matches!(function, Builtin::Print | Builtin::Sorted) {
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
                let marker = Value::Native(NativeValue::Stream(Stream::Stdin));
                let mut text = self
                    .read_stream(&marker, None, true)
                    .map_err(|error| self.record_native_error(error))?;
                if text.is_empty() {
                    return Err(self.record_native_error(PyError::exception(
                        "EOFError",
                        "EOF when reading a line",
                    )));
                }
                if text.ends_with('\n') {
                    text.pop();
                    if text.ends_with('\r') {
                        text.pop();
                    }
                }
                Ok(CallResult::Value(self.allocate_string(text)?))
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
                let codepoint = u32::try_from(value)
                    .ok()
                    .and_then(char::from_u32)
                    .ok_or("chr() arg not in range(0x110000)")?;
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
            Builtin::Dir => {
                expect_arity(&arguments, 0, 1)?;
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
                    let length = value.as_int().ok_or("__len__() should return an integer")?;
                    if length < 0 {
                        return Err("__len__() should return >= 0".into());
                    }
                    return Ok(CallResult::Value(Value::Int(length)));
                }
                let length = if let Some(length) =
                    protocol::string_length(&self.state.heap, &arguments[0])?
                {
                    length
                } else if let Some(id) = arguments[0].object_id() {
                    match self.state.heap.get(id)? {
                        Object::List(values) | Object::Tuple(values) | Object::Set(values) => {
                            values.len()
                        }
                        Object::Range { start, stop, step } => range_length(*start, *stop, *step)?,
                        Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                            entries.len()
                        }
                        Object::BigInt(_)
                        | Object::String(_)
                        | Object::Bytes(_)
                        | Object::ByteArray(_)
                        | Object::Slice { .. }
                        | Object::Exception { .. }
                        | Object::Function { .. }
                        | Object::Class { .. }
                        | Object::Instance { .. }
                        | Object::DescriptorBoundMethod { .. }
                        | Object::Iterator { .. }
                        | Object::SequenceIterator { .. }
                        | Object::RangeIterator { .. }
                        | Object::CountIterator { .. }
                        | Object::CallableIterator { .. }
                        | Object::Generator { .. }
                        | Object::Module { .. }
                        | Object::ArrayStorage(_)
                        | Object::Array { .. }
                        | Object::Regex { .. }
                        | Object::Match { .. }
                        | Object::ArgumentParser { .. }
                        | Object::Namespace { .. } => return Err("object has no len()".into()),
                        Object::EnumMember { .. } => return Err("object has no len()".into()),
                        Object::RaisesContext { .. } => return Err("object has no len()".into()),
                        Object::Property { .. }
                        | Object::StaticMethod { .. }
                        | Object::ClassMethod { .. }
                        | Object::Super { .. } => return Err("object has no len()".into()),
                    }
                } else {
                    return Err("object has no len()".into());
                };
                Ok(CallResult::Value(Value::Int(length as i64)))
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
                // Stable insertion sort keeps comparison dispatch and failure order obvious.
                for index in 1..keyed.len() {
                    let mut current = index;
                    while current > 0 {
                        self.charge_cpu(1)?;
                        if self.compare_values(&keyed[current].0, &keyed[current - 1].0)?
                            != if reverse {
                                Ordering::Greater
                            } else {
                                Ordering::Less
                            }
                        {
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
                if arguments.is_empty() {
                    return Err("expected at least one argument".into());
                }
                let values = if arguments.len() == 1 {
                    self.iterable_values(&arguments[0])?
                } else {
                    arguments
                };
                let mut values = values.into_iter();
                let mut selected = values.next().ok_or("argument is an empty sequence")?;
                for value in values {
                    self.charge_cpu(1)?;
                    let ordering = self.compare_values(&value, &selected)?;
                    let replace = match function {
                        Builtin::Minimum => ordering == Ordering::Less,
                        Builtin::Maximum => ordering == Ordering::Greater,
                        _ => unreachable!(),
                    };
                    if replace {
                        selected = value;
                    }
                }
                Ok(CallResult::Value(selected))
            }
            Builtin::Sum => {
                expect_arity(&arguments, 1, 2)?;
                let values = self.iterable_values(&arguments[0])?;
                let mut total = arguments.get(1).cloned().unwrap_or(Value::Int(0));
                for value in values {
                    self.charge_cpu(1)?;
                    total = self.add_numbers(total, value)?;
                }
                Ok(CallResult::Value(total))
            }
            Builtin::Absolute => {
                expect_arity(&arguments, 1, 1)?;
                let value = self
                    .invoke_slot(&arguments[0], Slot::Absolute, "__abs__", Vec::new())?
                    .ok_or("bad operand type for abs()")?;
                Ok(CallResult::Value(value))
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
                    if exponent.is_negative() {
                        let error =
                            PyError::value_error("negative exponent with modulus is not supported");
                        return Err(self.record_native_error(error));
                    }
                    let work = base_len
                        .saturating_add(modulus_len)
                        .saturating_mul(exponent_len.saturating_mul(4).max(1));
                    <Self as PyRuntime>::charge_cpu(self, u64::try_from(work).unwrap_or(u64::MAX))
                        .map_err(|error| error.to_string())?;
                    <Self as PyRuntime>::reserve_memory(self, modulus_len.saturating_mul(4).max(1))
                        .map_err(|error| error.to_string())?;
                    let positive_modulus = modulus.abs();
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
                let quotient =
                    self.binary_value(BinaryOperator::FloorDivide, arguments[0], arguments[1])?;
                let remainder =
                    self.binary_value(BinaryOperator::Remainder, arguments[0], arguments[1])?;
                Ok(CallResult::Value(self.allocate_object(Object::Tuple(
                    vec![quotient, remainder],
                ))?))
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
                let Some(id) = arguments[0].object_id() else {
                    return Err("next() argument is not an iterator".into());
                };
                let value = match self.state.heap.get(id)?.clone() {
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
                            if protocol::equals(&self.state.heap, &value, &sentinel)? {
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
                        if let Object::CountIterator { current, .. } =
                            self.state.heap.get_mut(id)?
                        {
                            *current = next;
                        }
                        Some(Value::Int(current))
                    }
                    Object::Generator { .. } => self.resume_generator(id)?,
                    Object::Iterator { .. }
                    | Object::SequenceIterator { .. }
                    | Object::RangeIterator { .. } => self.next_stored_iterator(id)?,
                    _ => {
                        match self.invoke_slot(&arguments[0], Slot::Next, "__next__", Vec::new()) {
                            Ok(Some(value)) => Some(value),
                            Ok(None) => return Err("next() argument is not an iterator".into()),
                            Err(_error)
                                if self
                                    .pending_exception
                                    .as_ref()
                                    .is_some_and(|exception| exception.kind == "StopIteration")
                                    && arguments.len() == 2 =>
                            {
                                self.pending_exception = None;
                                arguments.get(1).copied()
                            }
                            Err(error) => return Err(error),
                        }
                    }
                };
                let value = match value {
                    Some(value) => value,
                    None => arguments.get(1).cloned().ok_or("StopIteration")?,
                };
                Ok(CallResult::Value(value))
            }
            Builtin::Range => {
                expect_arity(&arguments, 1, 3)?;
                let integers = arguments
                    .iter()
                    .map(|value| value.as_int().ok_or("range arguments must be integers"))
                    .collect::<Result<Vec<_>, _>>()?;
                let (start, stop, step) = match integers.as_slice() {
                    [stop] => (0, *stop, 1),
                    [start, stop] => (*start, *stop, 1),
                    [start, stop, step] => (*start, *stop, *step),
                    _ => unreachable!(),
                };
                if step == 0 {
                    return Err("range() arg 3 must not be zero".into());
                }
                Ok(CallResult::Value(self.allocate_object(Object::Range {
                    start,
                    stop,
                    step,
                })?))
            }
            Builtin::Enumerate => {
                expect_arity(&arguments, 1, 2)?;
                let values = self.iterable_values(&arguments[0])?;
                let start = arguments.get(1).map_or(Ok(0), |value| {
                    value.as_int().ok_or("enumerate start must be an integer")
                })?;
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
        }
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
            return Err("maximum recursion depth exceeded".into());
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
            return Err(format!(
                "{name}() takes {} positional arguments but {} were given",
                signature.positional_count,
                arguments.len()
            ));
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
        for (keyword, value) in keyword_arguments {
            let Some(slot) = code
                .parameters
                .iter()
                .position(|parameter| parameter.name == keyword && !parameter.variadic)
            else {
                return Err(format!(
                    "{name}() got an unexpected keyword argument {keyword:?}"
                ));
            };
            if locals[slot].replace(value).is_some() {
                return Err(format!(
                    "{name}() got multiple values for argument {keyword:?}"
                ));
            }
        }
        if defaults.len() != signature.default_slots.len() {
            return Err(format!("{name}() has invalid default argument metadata"));
        }
        for (&slot, default) in signature.default_slots.iter().zip(defaults) {
            if locals[slot].is_none() {
                locals[slot] = Some(*default);
            }
        }
        if let Some((slot, parameter)) = code
            .parameters
            .iter()
            .enumerate()
            .find(|(slot, parameter)| locals[*slot].is_none() && !parameter.has_default)
        {
            debug_assert!(slot < locals.len());
            return Err(format!(
                "{name}() missing required argument {:?}",
                parameter.name
            ));
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

    pub(super) fn record_native_error(&mut self, error: PyError) -> String {
        let kind = match error.kind {
            PyErrorKind::Type => Some("TypeError"),
            PyErrorKind::Value => Some("ValueError"),
            PyErrorKind::ZeroDivision => Some("ZeroDivisionError"),
            PyErrorKind::Overflow => Some("OverflowError"),
            PyErrorKind::Runtime => Some("RuntimeError"),
            PyErrorKind::Exception(kind) => Some(kind),
            PyErrorKind::Resource
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

    pub(super) fn resume_native_call(
        &mut self,
        pending: PendingNativeCall,
    ) -> Result<CallResult, String> {
        let retry = PendingNativeCall {
            function: pending.function,
            arguments: pending.arguments.clone(),
            call_span: pending.call_span,
        };
        let previous_suspend = self.native_suspend_allowed;
        self.native_suspend_allowed = true;
        let result = (pending.function.call)(self, pending.arguments);
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
}
