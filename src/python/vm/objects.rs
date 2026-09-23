//! Runtime object operations for attributes, subscription, descriptors, classes, and types.

use super::{
    expect_arity, protocol, range_length, select_string_slice, Arc, BuiltinSubscript, BuiltinType,
    CallMode, CallResult, ClassDefinition, ClassField, ClassLayout, CodeCaches, CodeRef,
    ExceptionType, Execution, HashMap, LoadAttributeCache, NameId, NativeValue, Object, Ordering,
    PyArray, PyError, PyRuntime, PyValueCast, SlicePlan, Slot, SlotValue, SymbolId, TypeId, Value,
    ValueTag, Vm, MODELED_MAPPING_ENTRY_BYTES,
};

impl Vm<'_> {
    pub(super) fn load_attribute(&mut self, name: &str) -> Result<(), String> {
        let owner = self.pop()?;
        let value = self
            .resolve_attribute(owner, name)?
            .ok_or_else(|| format!("attribute {name:?} is not implemented"))?;
        self.stack.push(value);
        Ok(())
    }

    pub(super) fn load_attribute_at(
        &mut self,
        code: &CodeRef,
        code_cache: usize,
        site: usize,
        symbol: SymbolId,
        name: &str,
    ) -> Result<(), String> {
        let owner = self.pop()?;
        if let (Some(id), Some(cache)) = (owner.object_id(), self.attribute_cache(code_cache, site))
        {
            if let Some(value) =
                self.state
                    .heap
                    .cached_instance_attribute(id, cache.class, cache.location)?
            {
                self.stack.push(value);
                return Ok(());
            }
        }
        let value = self
            .resolve_attribute_by_symbol(owner, symbol, name)?
            .ok_or_else(|| format!("attribute {name:?} is not implemented"))?;
        if let Some(cache) = self.cacheable_instance_attribute(owner, symbol, name)? {
            self.remember_attribute_cache(code, code_cache, site, cache)?;
        }
        self.stack.push(value);
        Ok(())
    }

    fn attribute_cache(&self, code_cache: usize, site: usize) -> Option<LoadAttributeCache> {
        self.execution
            .code_caches
            .get(code_cache)?
            .attributes
            .as_ref()?
            .get(site)
            .copied()
            .flatten()
    }

    fn remember_attribute_cache(
        &mut self,
        code: &CodeRef,
        code_cache: usize,
        site: usize,
        cache: LoadAttributeCache,
    ) -> Result<(), String> {
        if let Some(attributes) = &mut self.execution.code_caches[code_cache].attributes {
            attributes[site] = Some(cache);
            return Ok(());
        }
        let bytes = code
            .instructions
            .len()
            .checked_mul(std::mem::size_of::<Option<LoadAttributeCache>>())
            .ok_or("attribute cache size overflow")?;
        if u64::try_from(bytes).unwrap_or(u64::MAX) > self.interp.resources.memory_remaining() {
            return Ok(());
        }
        self.reserve_retained_memory(bytes)?;
        let mut attributes = vec![None; code.instructions.len()];
        attributes[site] = Some(cache);
        self.execution.code_caches[code_cache].attributes = Some(attributes);
        Ok(())
    }

    #[inline(always)]
    pub(super) fn symbol_for(
        &mut self,
        code: &CodeRef,
        cache: usize,
        name: NameId,
    ) -> Result<SymbolId, String> {
        if let Some(symbol) = self.execution.code_caches[cache].names[name.index()] {
            return Ok(symbol);
        }
        self.resolve_symbol(code, cache, name)
    }

    #[cold]
    #[inline(never)]
    fn resolve_symbol(
        &mut self,
        code: &CodeRef,
        cache: usize,
        name: NameId,
    ) -> Result<SymbolId, String> {
        let symbol = self
            .state
            .heap
            .intern_symbol(code.name(name), &mut self.interp.resources)?;
        self.execution.code_caches[cache].names[name.index()] = Some(symbol);
        Ok(symbol)
    }

    pub(super) fn ensure_code_cache(&mut self, code: &CodeRef) -> Result<usize, String> {
        if let Some(index) = self
            .execution
            .code_caches
            .iter()
            .position(|cache| Arc::ptr_eq(&cache.code, code))
        {
            return Ok(index);
        }
        let bytes = code
            .name_count()
            .checked_mul(std::mem::size_of::<Option<SymbolId>>())
            .and_then(|names| names.checked_add(std::mem::size_of::<CodeCaches>()))
            .ok_or("code cache size overflow")?;
        self.reserve_retained_memory(bytes)?;
        let index = self.execution.code_caches.len();
        self.execution.code_caches.push(CodeCaches {
            code: code.clone(),
            names: vec![None; code.name_count()],
            attributes: None,
        });
        Ok(index)
    }

    fn cacheable_instance_attribute(
        &mut self,
        owner: Value,
        symbol: SymbolId,
        name: &str,
    ) -> Result<Option<LoadAttributeCache>, String> {
        let Some(id) = owner.object_id() else {
            return Ok(None);
        };
        let Object::Instance { class, .. } = self.state.heap.get(id)? else {
            return Ok(None);
        };
        let class = *class;
        if let Some((_, descriptor)) = self.class_attribute_entry(class, name)? {
            if self.is_data_descriptor(&descriptor)? {
                return Ok(None);
            }
        }
        Ok(self
            .state
            .heap
            .instance_attribute_slot_by_symbol(id, symbol)?
            .map(|location| LoadAttributeCache { class, location }))
    }

    /// Resolve one attribute without involving the operand stack.
    ///
    /// Missing attributes return `None`; errors raised while invoking descriptors remain errors.
    /// This distinction lets `getattr` and `hasattr` share the bytecode lookup path.
    pub(super) fn resolve_attribute(
        &mut self,
        owner: Value,
        name: &str,
    ) -> Result<Option<Value>, String> {
        let symbol = self.state.heap.symbol_id(name);
        self.resolve_attribute_inner(owner, symbol, name)
    }

    fn resolve_attribute_by_symbol(
        &mut self,
        owner: Value,
        symbol: SymbolId,
        name: &str,
    ) -> Result<Option<Value>, String> {
        self.resolve_attribute_inner(owner, Some(symbol), name)
    }

    fn resolve_attribute_inner(
        &mut self,
        owner: Value,
        symbol: Option<SymbolId>,
        name: &str,
    ) -> Result<Option<Value>, String> {
        if let Some(NativeValue::Module(module)) = owner.native_value() {
            if let Some(function) = module.function(name) {
                return Ok(Some(Value::Native(NativeValue::NativeFunction(function))));
            }
            if let Some(value) = module.value(name) {
                let value = value.get(self).map_err(|error| error.to_string())?;
                return Ok(Some(value));
            }
        }
        let owner_type = self.type_id(&owner)?;
        if let Some(method) = self
            .state
            .types
            .attribute(owner_type, name)?
            .and_then(|value| match value.native_value() {
                Some(NativeValue::NativeMethod(method)) => Some(method),
                _ => None,
            })
        {
            if method.name == "__new__" {
                return Ok(Some(Value::Native(NativeValue::NativeMethod(method))));
            }
            let bound = self.allocate_object(Object::DescriptorBoundMethod {
                receiver: owner,
                descriptor: Value::Native(NativeValue::NativeMethod(method)),
                owner: None,
            })?;
            return Ok(Some(bound));
        }
        if let Some(id) = owner.object_id() {
            match self.state.heap.get(id)?.clone() {
                Object::Module { scope, .. } => {
                    return Ok(self.state.heap.scope_get(scope, name).copied());
                }
                Object::Class { .. } => {
                    let mut entry = self.class_attribute_entry(id, name)?;
                    if entry.is_none() {
                        let Object::Class { metaclass, .. } = self.state.heap.get(id)? else {
                            unreachable!()
                        };
                        let metaclass = *metaclass;
                        if let Some(metaclass) = metaclass.object_id() {
                            entry = self.class_attribute_entry(metaclass, name)?;
                        }
                    }
                    let Some((defining_class, descriptor)) = entry else {
                        return Ok(None);
                    };
                    let value = self.bind_descriptor(descriptor, None, id, defining_class)?;
                    return Ok(Some(value));
                }
                Object::EnumMember {
                    name: member_name,
                    value,
                } => {
                    let value = match name {
                        "name" => self.allocate_string(member_name)?,
                        "value" => value,
                        _ => return Ok(None),
                    };
                    return Ok(Some(value));
                }
                Object::Instance { class, .. } => {
                    let class_entry = self.class_attribute_entry(class, name)?;
                    if let Some((defining_class, descriptor)) = class_entry {
                        if self.is_data_descriptor(&descriptor)? {
                            let value = self.bind_descriptor(
                                descriptor,
                                Some(owner),
                                class,
                                defining_class,
                            )?;
                            return Ok(Some(value));
                        }
                    }
                    let instance_value = match symbol {
                        Some(symbol) => self.state.heap.attribute_by_symbol(id, symbol)?.copied(),
                        None => None,
                    };
                    if let Some(value) = instance_value {
                        return Ok(Some(value));
                    }
                    let Some((defining_class, descriptor)) = class_entry else {
                        return Ok(None);
                    };
                    let value =
                        self.bind_descriptor(descriptor, Some(owner), class, defining_class)?;
                    return Ok(Some(value));
                }
                Object::Super {
                    start_class,
                    receiver,
                } => {
                    let (defining_class, descriptor, accessed_class) =
                        self.super_attribute(start_class, &receiver, name)?;
                    let value = self.bind_descriptor(
                        descriptor,
                        Some(receiver),
                        accessed_class,
                        defining_class,
                    )?;
                    return Ok(Some(value));
                }
                Object::Match { .. } => {}
                Object::ArgumentParser { prog, .. } => {
                    if name == "prog" {
                        let prog = self.allocate_string(prog)?;
                        return Ok(Some(prog));
                    }
                }
                Object::Namespace { values } => {
                    let value = values
                        .iter()
                        .find(|(key, _)| key == name)
                        .map(|(_, value)| *value);
                    return Ok(value);
                }
                Object::Slice { start, stop, step } => {
                    let component = match name {
                        "start" => start,
                        "stop" => stop,
                        "step" => step,
                        _ => return Ok(None),
                    };
                    return Ok(Some(component.map_or(Value::None, Value::Int)));
                }
                Object::Array { layout, dtype, .. } => {
                    let value = match name {
                        "shape" => Some(
                            self.allocate_object(Object::Tuple(
                                layout
                                    .shape
                                    .iter()
                                    .map(|value| Value::Int(*value as i64))
                                    .collect(),
                            ))?,
                        ),
                        "ndim" => Some(Value::Int(layout.shape.len() as i64)),
                        "size" => Some(Value::Int(
                            layout
                                .shape
                                .iter()
                                .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
                                .ok_or("array size overflow")? as i64,
                        )),
                        "dtype" => Some(self.allocate_string(dtype.name().to_string())?),
                        "T" => {
                            let array = owner
                                .cast::<PyArray>(self)
                                .map_err(|error| error.to_string())?;
                            Some(
                                super::super::stdlib::numpy::transpose(self, array, None)
                                    .map_err(|error| error.to_string())?,
                            )
                        }
                        _ => None,
                    };
                    return Ok(value);
                }
                _ => {}
            }
        }
        let value = if matches!(owner.native_value(), Some(NativeValue::UnitTestBase))
            && name == "__name__"
        {
            Some(self.allocate_string("TestCase".into())?)
        } else {
            None
        };
        Ok(value)
    }

    pub(super) fn store_attribute_by_symbol(
        &mut self,
        owner: Value,
        symbol: SymbolId,
        name: &str,
        value: Value,
    ) -> Result<(), String> {
        let Some(id) = owner.object_id() else {
            return Err("object does not support attribute assignment".into());
        };
        let Object::Instance { class, .. } = self.state.heap.get(id)? else {
            return Err("object does not support attribute assignment".into());
        };
        let class = *class;
        if let Some((_, descriptor)) = self.class_attribute_entry(class, name)? {
            if let Some(descriptor_id) = descriptor.object_id() {
                match self.state.heap.get(descriptor_id)?.clone() {
                    Object::Property {
                        setter: Some(setter),
                        ..
                    } => {
                        self.invoke_value(setter, vec![Value::Object(id), value])?;
                        return Ok(());
                    }
                    Object::Property { setter: None, .. } => {
                        return Err(format!("property {name:?} has no setter"));
                    }
                    Object::Instance {
                        class: descriptor_class,
                        ..
                    } => {
                        if let Some((set_owner, set)) =
                            self.class_attribute_entry(descriptor_class, "__set__")?
                        {
                            let set = self.bind_descriptor(
                                set,
                                Some(descriptor),
                                descriptor_class,
                                set_owner,
                            )?;
                            self.invoke_value(set, vec![Value::Object(id), value])?;
                            return Ok(());
                        }
                    }
                    _ => {}
                }
            }
        }
        self.state.heap.insert_attribute_by_symbol(
            id,
            symbol,
            value,
            &mut self.interp.resources,
        )?;
        Ok(())
    }

    pub(super) fn load_subscript(&mut self) -> Result<(), String> {
        let index = self.pop()?;
        let owner = self.pop()?;
        if let Some(value) = self.invoke_slot(&owner, Slot::GetItem, "__getitem__", vec![index])? {
            self.stack.push(value);
            return Ok(());
        }
        if let Some((start, stop, step)) = self.slice_parts(&index) {
            let value = self.load_builtin_slice(owner, start, stop, step)?;
            self.stack.push(value);
            return Ok(());
        }
        let value = if matches!(owner.native_value(), Some(NativeValue::TypingList)) {
            let parameter = match index.native_value() {
                Some(NativeValue::BuiltinType(BuiltinType::Int)) => "int".to_string(),
                Some(NativeValue::BuiltinType(BuiltinType::String)) => "str".to_string(),
                _ => protocol::repr(&self.state.heap, &index)?,
            };
            self.allocate_string(format!("typing.List[{parameter}]"))?
        } else if matches!(owner.native_value(), Some(NativeValue::Environment)) {
            let name = protocol::string_ref(&self.state.heap, &index)?
                .ok_or("environment key must be a string")?
                .as_str()
                .to_string();
            let value = self
                .interp
                .get_var(&name)
                .ok_or_else(|| format!("environment key not found: {name}"))?;
            self.allocate_string(value)?
        } else if let Some(character) = protocol::string_index(&self.state.heap, &owner, &index)? {
            self.allocate_string(character.to_string())?
        } else if let Some(id) = owner.object_id() {
            let target = match self.state.heap.get(id)? {
                Object::List(values) | Object::Tuple(values) => {
                    let index = index.as_int().ok_or("sequence index must be an integer")?;
                    let len = values.len() as i64;
                    let index = if index < 0 { len + index } else { index };
                    BuiltinSubscript::Value(
                        values
                            .get(usize::try_from(index).map_err(|_| "index out of range")?)
                            .copied()
                            .ok_or("index out of range")?,
                    )
                }
                Object::Range { start, stop, step } => {
                    let length = range_length(*start, *stop, *step)?;
                    let index = index.as_int().ok_or("range index must be an integer")?;
                    let index = if index < 0 {
                        i128::try_from(length).map_err(|_| "range is too large")?
                            + i128::from(index)
                    } else {
                        i128::from(index)
                    };
                    if index < 0 || index >= i128::try_from(length).unwrap_or(i128::MAX) {
                        return Err("range index out of range".into());
                    }
                    let value = i128::from(*start)
                        .checked_add(
                            i128::from(*step)
                                .checked_mul(index)
                                .ok_or("range value overflow")?,
                        )
                        .ok_or("range value overflow")?;
                    BuiltinSubscript::Value(Value::Int(
                        i64::try_from(value)
                            .map_err(|_| "range value exceeds bounded integer range")?,
                    ))
                }
                Object::Dict(_) => BuiltinSubscript::Mapping { factory: None },
                Object::DefaultDict { factory, .. } => BuiltinSubscript::Mapping {
                    factory: Some(*factory),
                },
                Object::Set(_) => BuiltinSubscript::Set,
                Object::String(_)
                | Object::Bytes(_)
                | Object::ByteArray(_)
                | Object::Slice { .. }
                | Object::Exception { .. }
                | Object::BigInt(_)
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
                | Object::Namespace { .. }
                | Object::EnumMember { .. }
                | Object::RaisesContext { .. } => BuiltinSubscript::Unsupported,
                Object::Property { .. }
                | Object::StaticMethod { .. }
                | Object::ClassMethod { .. }
                | Object::Super { .. } => BuiltinSubscript::Unsupported,
            };
            match target {
                BuiltinSubscript::Value(value) => value,
                BuiltinSubscript::Mapping { factory } => {
                    if let Some(position) = self.find_mapping_entry(id, &index)? {
                        match self.state.heap.get(id)? {
                            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                                entries[position].1
                            }
                            _ => unreachable!("mapping kind was classified before lookup"),
                        }
                    } else if let Some(factory) = factory {
                        self.stack.push(factory);
                        let value = match self.call(0, &[], &[], CallMode::Immediate)? {
                            CallResult::Value(value) => value,
                            CallResult::Exit(status) => {
                                return Err(format!("default factory exited with status {status}"))
                            }
                            CallResult::EnteredFrame => {
                                unreachable!("immediate call entered a frame")
                            }
                            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                                unreachable!("immediate call cannot suspend")
                            }
                        };
                        self.state.heap.reserve_object_growth(
                            id,
                            MODELED_MAPPING_ENTRY_BYTES,
                            &mut self.interp.resources,
                        )?;
                        let Object::DefaultDict { entries, .. } = self.state.heap.get_mut(id)?
                        else {
                            unreachable!("defaultdict kind was classified before insertion")
                        };
                        entries.push((index, value));
                        value
                    } else {
                        return Err("key not found".into());
                    }
                }
                BuiltinSubscript::Set => return Err("set object is not subscriptable".into()),
                BuiltinSubscript::Unsupported => return Err("object is not subscriptable".into()),
            }
        } else {
            return Err("object is not subscriptable".into());
        };
        self.stack.push(value);
        Ok(())
    }

    fn load_builtin_slice(
        &mut self,
        owner: Value,
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    ) -> Result<Value, String> {
        let string_slice = protocol::string_ref(&self.state.heap, &owner)?
            .map(|text| select_string_slice(text.as_str(), text.is_ascii(), start, stop, step))
            .transpose()?;
        if let Some((selected, units)) = string_slice {
            self.charge_cpu(units)?;
            return self.allocate_string(selected);
        }
        if let Some(bytes) = protocol::bytes_value(&self.state.heap, &owner)? {
            let plan = SlicePlan::new(bytes.len(), start, stop, step)?;
            self.charge_cpu(u64::try_from(plan.len()).unwrap_or(u64::MAX))?;
            let selected = plan.indices().map(|index| bytes[index]).collect();
            let selected = if self.type_id(&owner)? == BuiltinType::ByteArray.id() {
                self.allocate_bytearray(selected)?
            } else {
                self.allocate_bytes(selected)?
            };
            return Ok(selected);
        }
        if let Some(id) = owner.object_id() {
            let (values, tuple) = match self.state.heap.get(id)?.clone() {
                Object::List(values) => (values, false),
                Object::Tuple(values) => (values, true),
                _ => return Err("object is not sliceable".into()),
            };
            let plan = SlicePlan::new(values.len(), start, stop, step)?;
            self.charge_cpu(u64::try_from(plan.len()).unwrap_or(u64::MAX))?;
            let selected = plan.indices().map(|index| values[index]).collect();
            return self.allocate_object(if tuple {
                Object::Tuple(selected)
            } else {
                Object::List(selected)
            });
        }
        Err("object is not sliceable".into())
    }

    pub(super) fn build_slice(
        &mut self,
        has_start: bool,
        has_stop: bool,
        has_step: bool,
    ) -> Result<(), String> {
        let step = if has_step {
            Some(
                self.pop()?
                    .as_int()
                    .ok_or("slice step must be an integer")?,
            )
        } else {
            None
        };
        let stop = if has_stop {
            Some(
                self.pop()?
                    .as_int()
                    .ok_or("slice stop must be an integer")?,
            )
        } else {
            None
        };
        let start = if has_start {
            Some(
                self.pop()?
                    .as_int()
                    .ok_or("slice start must be an integer")?,
            )
        } else {
            None
        };
        let value = self.allocate_object(Object::Slice { start, stop, step })?;
        self.stack.push(value);
        Ok(())
    }

    pub(super) fn store_subscript(&mut self) -> Result<(), String> {
        let index = self.pop()?;
        let owner = self.pop()?;
        let value = self.pop()?;
        if self
            .invoke_slot(&owner, Slot::SetItem, "__setitem__", vec![index, value])?
            .is_some()
        {
            return Ok(());
        }
        let Some(id) = owner.object_id() else {
            return Err("object does not support item assignment".into());
        };
        match self.state.heap.get(id)? {
            Object::List(values) => {
                let index = index.as_int().ok_or("list index must be an integer")?;
                let len = values.len() as i64;
                let index = if index < 0 { len + index } else { index };
                let index =
                    usize::try_from(index).map_err(|_| "list assignment index out of range")?;
                let Object::List(values) = self.state.heap.get_mut(id)? else {
                    unreachable!()
                };
                let slot = values
                    .get_mut(index)
                    .ok_or("list assignment index out of range")?;
                *slot = value;
            }
            Object::Dict(_) | Object::DefaultDict { .. } => {
                if let Some(position) = self.find_mapping_entry(id, &index)? {
                    let entries = match self.state.heap.get_mut(id)? {
                        Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
                        _ => unreachable!(),
                    };
                    entries.set_value(position, value);
                } else {
                    self.state.heap.reserve_object_growth(
                        id,
                        MODELED_MAPPING_ENTRY_BYTES,
                        &mut self.interp.resources,
                    )?;
                    let entries = match self.state.heap.get_mut(id)? {
                        Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
                        _ => unreachable!(),
                    };
                    entries.push((index, value));
                }
            }
            Object::Tuple(_) => return Err("tuple object does not support item assignment".into()),
            Object::String(_)
            | Object::Bytes(_)
            | Object::ByteArray(_)
            | Object::Slice { .. }
            | Object::Exception { .. }
            | Object::Set(_)
            | Object::BigInt(_)
            | Object::Range { .. }
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
            | Object::Namespace { .. } => {
                return Err("object does not support item assignment".into())
            }
            Object::EnumMember { .. } => {
                return Err("object does not support item assignment".into())
            }
            Object::RaisesContext { .. } => {
                return Err("object does not support item assignment".into())
            }
            Object::Property { .. }
            | Object::StaticMethod { .. }
            | Object::ClassMethod { .. }
            | Object::Super { .. } => return Err("object does not support item assignment".into()),
        }
        Ok(())
    }

    pub(super) fn delete_subscript(&mut self) -> Result<(), String> {
        let index = self.pop()?;
        let owner = self.pop()?;
        self.invoke_slot(&owner, Slot::DeleteItem, "__delitem__", vec![index])?
            .ok_or("object does not support item deletion")?;
        Ok(())
    }

    pub(super) fn make_function(
        &mut self,
        name: String,
        code: CodeRef,
        default_count: usize,
    ) -> Result<(), String> {
        if self.frame_stack_len() < default_count {
            return Err("invalid bytecode stack effect while creating function".into());
        }
        let defaults_start = self.stack.len() - default_count;
        let defaults = self.stack.split_off(defaults_start);
        let mut closure = self.local_scopes.last().copied();
        if closure.is_some() && closure == self.class_scopes.last().copied() {
            closure = self
                .state
                .heap
                .scope_parent(closure.expect("checked above"))?;
        }
        let function = self.state.heap.allocate(
            Object::Function {
                name: name.clone(),
                code,
                closure,
                defaults,
                defining_class: None,
            },
            &mut self.interp.resources,
        )?;
        self.stack.push(function);
        Ok(())
    }

    pub(super) fn make_class(
        &mut self,
        name: String,
        code: &CodeRef,
        base_count: usize,
        has_metaclass: bool,
        fields: &[ClassField],
    ) -> Result<(), String> {
        let stack_values = base_count
            .checked_add(usize::from(has_metaclass))
            .ok_or("too many class construction values")?;
        if self.frame_stack_len() < stack_values {
            return Err("invalid bytecode stack effect while creating class".into());
        }
        let explicit_metaclass = has_metaclass.then(|| self.stack.pop().expect("checked above"));
        let bases_start = self.stack.len() - base_count;
        let bases = self.stack.split_off(bases_start);
        let is_enum =
            bases.len() == 1 && matches!(bases[0].native_value(), Some(NativeValue::EnumBase));
        let is_unittest =
            bases.len() == 1 && matches!(bases[0].native_value(), Some(NativeValue::UnitTestBase));
        let has_int_base = bases.iter().any(|base| {
            matches!(
                base.native_value(),
                Some(NativeValue::BuiltinType(BuiltinType::Int))
            )
        });
        let has_object_base = bases.iter().any(|base| {
            matches!(
                base.native_value(),
                Some(NativeValue::BuiltinType(BuiltinType::Object))
            )
        });
        let has_type_base = bases.iter().any(|base| {
            matches!(
                base.native_value(),
                Some(NativeValue::BuiltinType(BuiltinType::Type))
            )
        });
        let direct_exception_bases = bases
            .iter()
            .filter_map(|base| match base.native_value() {
                Some(NativeValue::ExceptionType(ExceptionType(name))) => Some(name),
                _ => None,
            })
            .collect::<Vec<_>>();
        let user_bases = if is_enum || is_unittest {
            Vec::new()
        } else {
            bases
                .iter()
                .filter_map(|base| {
                    if let Some(id) = base.object_id() {
                        if matches!(self.state.heap.get(id), Ok(Object::Class { .. })) {
                            Some(Ok(id))
                        } else {
                            Some(Err("class bases must be classes".to_string()))
                        }
                    } else if matches!(
                        base.native_value(),
                        Some(NativeValue::BuiltinType(
                            BuiltinType::Int | BuiltinType::Object | BuiltinType::Type
                        )) | Some(NativeValue::ExceptionType(_))
                    ) {
                        None
                    } else {
                        Some(Err("class bases must be classes".to_string()))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let inherited_exception_bases = user_bases
            .iter()
            .filter_map(|base| match self.state.heap.get(*base) {
                Ok(Object::Class { exception_base, .. }) => *exception_base,
                _ => None,
            })
            .collect::<Vec<_>>();
        let exception_base = direct_exception_bases
            .iter()
            .chain(inherited_exception_bases.iter())
            .copied()
            .next();
        if direct_exception_bases.len() + inherited_exception_bases.len() > 1 {
            return Err("multiple exception bases are unsupported".into());
        }
        if exception_base.is_some() && (has_int_base || has_type_base || is_enum || is_unittest) {
            return Err("exception classes cannot use another instance layout".into());
        }
        if has_int_base && (bases.len() != 1 || is_enum || is_unittest) {
            return Err("int inheritance with another direct base is unsupported".into());
        }
        if has_object_base && bases.len() != 1 {
            return Err("object cannot be combined with another direct base in this slice".into());
        }
        if has_type_base && bases.len() != 1 {
            return Err("type cannot be combined with another direct base in this slice".into());
        }
        let inherited_int_layouts = user_bases
            .iter()
            .filter_map(|base| match self.state.heap.get(*base) {
                Ok(Object::Class { layout, .. }) => Some(*layout == ClassLayout::Int),
                _ => None,
            })
            .filter(|is_int| *is_int)
            .count();
        if inherited_int_layouts > 1 {
            return Err("multiple bases have incompatible int instance layouts".into());
        }
        let inherited_type_layouts = user_bases
            .iter()
            .filter_map(|base| match self.state.heap.get(*base) {
                Ok(Object::Class { layout, .. }) => Some(*layout == ClassLayout::Type),
                _ => None,
            })
            .filter(|is_type| *is_type)
            .count();
        if inherited_type_layouts > 1 || (inherited_type_layouts == 1 && inherited_int_layouts == 1)
        {
            return Err("multiple bases have incompatible instance layouts".into());
        }
        let layout = if has_type_base || inherited_type_layouts == 1 {
            ClassLayout::Type
        } else if has_int_base || inherited_int_layouts == 1 {
            ClassLayout::Int
        } else {
            ClassLayout::Object
        };
        let has_explicit_metaclass = explicit_metaclass.is_some();
        let mut metaclass = explicit_metaclass
            .unwrap_or(Value::Native(NativeValue::BuiltinType(BuiltinType::Type)));
        for base in &user_bases {
            let Object::Class {
                metaclass: base_metaclass,
                ..
            } = self.state.heap.get(*base)?
            else {
                unreachable!()
            };
            let base_metaclass = *base_metaclass;
            let winner_type = self
                .class_type_id(&metaclass)?
                .ok_or("metaclass must be a type")?;
            let candidate_type = self
                .class_type_id(&base_metaclass)?
                .ok_or("base class has an invalid metaclass")?;
            if self.state.types.is_subclass(winner_type, candidate_type)? {
                continue;
            }
            if !has_explicit_metaclass
                && self.state.types.is_subclass(candidate_type, winner_type)?
            {
                metaclass = base_metaclass;
                continue;
            }
            return Err("metaclass conflict between bases or explicit metaclass".into());
        }
        let valid_metaclass = matches!(
            metaclass.native_value(),
            Some(NativeValue::BuiltinType(BuiltinType::Type))
        ) || metaclass.object_id().is_some_and(|id| {
            matches!(
                self.state.heap.get(id),
                Ok(Object::Class {
                    layout: ClassLayout::Type,
                    ..
                })
            )
        });
        if !valid_metaclass {
            return Err("metaclass must derive from type".into());
        }
        let mro = self.linearize_bases(&user_bases)?;
        let mut prepared_namespace = HashMap::new();
        if let Some(metaclass_id) = metaclass.object_id() {
            if let Some((owner, prepare)) =
                self.class_attribute_entry(metaclass_id, "__prepare__")?
            {
                let prepare = self.bind_descriptor(prepare, None, metaclass_id, owner)?;
                let bases_value = self.allocate_object(Object::Tuple(bases.clone()))?;
                let class_name = self.allocate_string(name.clone())?;
                let namespace = self.invoke_value(prepare, vec![class_name, bases_value])?;
                let Some(namespace_id) = namespace.object_id() else {
                    return Err("metaclass __prepare__() must return a mapping".into());
                };
                let Object::Dict(entries) = self.state.heap.get(namespace_id)?.clone() else {
                    return Err("metaclass __prepare__() must return a dict in this slice".into());
                };
                for (key, value) in entries {
                    let Some(key) = protocol::string_value(&self.state.heap, &key)? else {
                        return Err("metaclass namespace keys must be strings".into());
                    };
                    prepared_namespace.insert(key, value);
                }
            }
        }
        let parent = self.local_scopes.last().copied();
        let uses_repl_globals = parent
            .map(|scope| self.state.heap.scope_uses_repl_globals(scope))
            .transpose()?
            .unwrap_or(true);
        let scope = self.state.heap.allocate_scope(
            parent,
            uses_repl_globals,
            Arc::from([]),
            prepared_namespace,
            &mut self.interp.resources,
        )?;
        self.local_scopes.push(scope);
        self.class_scopes.push(scope);
        self.class_bindings.push(Vec::new());
        let execution = self.execute_code(code);
        let bindings = self
            .class_bindings
            .pop()
            .expect("class binding stack is present");
        self.class_scopes.pop();
        self.local_scopes.pop();
        match execution {
            Ok(Execution::Pending) => unreachable!("execute_code drains pending quanta"),
            Ok(Execution::Blocked(_)) => unreachable!("immediate code cannot suspend"),
            Ok(Execution::Halt) => {}
            Ok(Execution::Return(_)) => return Err("'return' outside function".into()),
            Ok(Execution::Yield(_, _)) => return Err("'yield' outside function".into()),
            Ok(Execution::Exit(status)) => {
                return Err(format!("class body exited with status {status}"))
            }
            Err((error, span)) => {
                return Err(format!(
                    "{error} in class {name} at line {}, column {}",
                    span.line, span.column
                ))
            }
        }
        let mut attributes = self.state.heap.scope_values(scope)?;
        if is_unittest {
            attributes.insert("__shellsim_unittest__".into(), Value::Bool(true));
            for method in super::super::stdlib::unittest::TEST_CASE_TYPE.methods {
                attributes.insert(
                    method.name.into(),
                    Value::Native(NativeValue::NativeMethod(method)),
                );
            }
        }
        let dataclass_fields = fields
            .iter()
            .map(|field| (field.name.clone(), attributes.get(&field.name).cloned()))
            .collect::<Vec<_>>();
        let mut enum_members = Vec::new();
        if is_enum {
            for member_name in bindings {
                if member_name.starts_with('_') {
                    continue;
                }
                let Some(value) = attributes.get(&member_name).cloned() else {
                    continue;
                };
                if value.object_id().is_some_and(|id| {
                    matches!(self.state.heap.get(id), Ok(Object::Function { .. }))
                }) {
                    continue;
                }
                let member = self.allocate_object(Object::EnumMember {
                    name: member_name.clone(),
                    value,
                })?;
                attributes.insert(member_name, member);
                enum_members.push(member);
            }
        }
        let descriptor_candidates = attributes
            .iter()
            .map(|(name, value)| (name.clone(), *value))
            .collect::<Vec<_>>();
        let class = if let Some(metaclass_id) = metaclass.object_id() {
            if let Some((owner, constructor)) =
                self.class_attribute_entry(metaclass_id, "__new__")?
            {
                let constructor =
                    self.bind_descriptor(constructor, Some(metaclass), metaclass_id, owner)?;
                let class_name = self.allocate_string(name.clone())?;
                let bases_value = self.allocate_object(Object::Tuple(bases.clone()))?;
                let mut namespace_entries = Vec::with_capacity(descriptor_candidates.len());
                for (attribute_name, value) in &descriptor_candidates {
                    namespace_entries.push((self.allocate_string(attribute_name.clone())?, *value));
                }
                let namespace = self.allocate_object(Object::Dict(namespace_entries.into()))?;
                self.invoke_value(constructor, vec![class_name, bases_value, namespace])?
            } else {
                self.allocate_class(ClassDefinition {
                    name: name.clone(),
                    bases: bases.clone(),
                    user_bases: user_bases.clone(),
                    mro,
                    metaclass,
                    layout,
                    exception_base,
                    attributes,
                    dataclass_fields,
                    enum_members,
                })?
            }
        } else {
            self.allocate_class(ClassDefinition {
                name: name.clone(),
                bases: bases.clone(),
                user_bases: user_bases.clone(),
                mro,
                metaclass,
                layout,
                exception_base,
                attributes,
                dataclass_fields,
                enum_members,
            })?
        };
        let class_id = class
            .object_id()
            .ok_or("metaclass __new__() must return a class")?;
        if !matches!(self.state.heap.get(class_id)?, Object::Class { .. }) {
            return Err("metaclass __new__() must return a class in this slice".into());
        }
        if let Some(base) = user_bases.first().copied() {
            if let Some((owner, initializer)) =
                self.class_attribute_entry(base, "__init_subclass__")?
            {
                let initializer =
                    self.bind_descriptor(initializer, Some(Value::Object(class_id)), base, owner)?;
                self.invoke_value(initializer, Vec::new())?;
            }
        }
        if let Some(metaclass_id) = metaclass.object_id() {
            if let Some((owner, initializer)) =
                self.class_attribute_entry(metaclass_id, "__init__")?
            {
                let initializer = self.bind_descriptor(
                    initializer,
                    Some(Value::Object(class_id)),
                    metaclass_id,
                    owner,
                )?;
                let bases = self.allocate_object(Object::Tuple(bases))?;
                let mut entries = Vec::with_capacity(descriptor_candidates.len());
                for (name, value) in descriptor_candidates {
                    entries.push((self.allocate_string(name)?, value));
                }
                let namespace = self.allocate_object(Object::Dict(entries.into()))?;
                let class_name = self.allocate_string(name.clone())?;
                let result = self.invoke_value(initializer, vec![class_name, bases, namespace])?;
                if !result.is_none() {
                    return Err("metaclass __init__() should return None".into());
                }
            }
        }
        self.stack.push(Value::Object(class_id));
        Ok(())
    }

    /// Allocate and finish a class after metaclass policy has selected its layout and C3 MRO.
    /// Both ordinary class statements and `type.__new__` use this path.
    pub(super) fn allocate_class(&mut self, definition: ClassDefinition) -> Result<Value, String> {
        let ClassDefinition {
            name,
            bases,
            user_bases,
            mro,
            metaclass,
            layout,
            exception_base,
            attributes,
            dataclass_fields,
            enum_members,
        } = definition;
        let mut type_bases = Vec::new();
        for base in &bases {
            let base = match base.native_value() {
                Some(NativeValue::ExceptionType(_)) => Some(BuiltinType::Exception.id()),
                _ => self.class_type_id(base)?,
            };
            if let Some(base) = base {
                type_bases.push(base);
            }
        }
        if type_bases.is_empty() {
            type_bases.push(BuiltinType::Object.id());
        }
        let mut type_mro = mro
            .iter()
            .map(|ancestor| match self.state.heap.get(*ancestor)? {
                Object::Class { instance_type, .. } => Ok(*instance_type),
                _ => Err("class MRO contains a non-class object".into()),
            })
            .collect::<Result<Vec<_>, String>>()?;
        if exception_base.is_some() && !type_mro.contains(&BuiltinType::Exception.id()) {
            type_mro.push(BuiltinType::Exception.id());
        }
        let builtin_ancestor = match layout {
            ClassLayout::Object => None,
            ClassLayout::Int => Some(BuiltinType::Int.id()),
            ClassLayout::Type => Some(BuiltinType::Type.id()),
        };
        if let Some(ancestor) = builtin_ancestor {
            if !type_mro.contains(&ancestor) {
                type_mro.push(ancestor);
            }
        }
        if !type_mro.contains(&BuiltinType::Object.id()) {
            type_mro.push(BuiltinType::Object.id());
        }
        self.class_type_id(&metaclass)?
            .ok_or("metaclass must be a type")?;
        let instance_type =
            self.state
                .types
                .register(name.clone(), type_bases, type_mro, &attributes)?;
        let descriptors = attributes
            .iter()
            .map(|(name, value)| (name.clone(), *value))
            .collect::<Vec<_>>();
        let class = self.allocate_object(Object::Class {
            instance_type,
            name,
            bases: user_bases,
            mro,
            metaclass,
            layout,
            exception_base,
            attributes,
            is_dataclass: false,
            dataclass_fields,
            enum_members,
        })?;
        self.state.types.finish(instance_type, class)?;
        let class_id = class.object_id().expect("allocated class has an object id");
        for (_, descriptor) in &descriptors {
            let Some(function_id) = descriptor.object_id() else {
                continue;
            };
            if let Object::Function { defining_class, .. } = self.state.heap.get_mut(function_id)? {
                *defining_class = Some(class_id);
            }
        }
        for (attribute_name, descriptor) in descriptors {
            let Some(descriptor_id) = descriptor.object_id() else {
                continue;
            };
            let Object::Instance {
                class: descriptor_class,
                ..
            } = self.state.heap.get(descriptor_id)?.clone()
            else {
                continue;
            };
            let Some((owner, set_name)) =
                self.class_attribute_entry(descriptor_class, "__set_name__")?
            else {
                continue;
            };
            let set_name =
                self.bind_descriptor(set_name, Some(descriptor), descriptor_class, owner)?;
            let attribute_name = self.allocate_string(attribute_name)?;
            self.invoke_value(set_name, vec![class, attribute_name])?;
        }
        Ok(class)
    }

    pub(super) fn linearize_bases(
        &mut self,
        bases: &[super::super::heap::ObjectId],
    ) -> Result<Vec<super::super::heap::ObjectId>, String> {
        for (index, base) in bases.iter().enumerate() {
            if bases[..index].contains(base) {
                return Err("duplicate base class".into());
            }
        }
        let mut sequences = Vec::with_capacity(bases.len().saturating_add(1));
        for base in bases {
            let Object::Class { mro, .. } = self.state.heap.get(*base)? else {
                return Err("class base changed object kind".into());
            };
            let mut sequence = Vec::with_capacity(mro.len().saturating_add(1));
            sequence.push(*base);
            sequence.extend(mro.iter().copied());
            sequences.push(sequence);
        }
        sequences.push(bases.to_vec());

        let mut result = Vec::new();
        loop {
            sequences.retain(|sequence| !sequence.is_empty());
            if sequences.is_empty() {
                return Ok(result);
            }
            let candidate = sequences.iter().find_map(|sequence| {
                let head = sequence[0];
                (!sequences
                    .iter()
                    .any(|other| other.iter().skip(1).any(|item| *item == head)))
                .then_some(head)
            });
            let Some(candidate) = candidate else {
                return Err("cannot create a consistent method resolution order".into());
            };
            self.charge_cpu(u64::try_from(sequences.len()).unwrap_or(u64::MAX))?;
            result.push(candidate);
            for sequence in &mut sequences {
                if sequence.first() == Some(&candidate) {
                    sequence.remove(0);
                }
            }
        }
    }

    pub(super) fn class_attribute(
        &mut self,
        class: super::super::heap::ObjectId,
        name: &str,
    ) -> Result<Option<Value>, String> {
        Ok(self
            .class_attribute_entry(class, name)?
            .map(|(_, value)| value))
    }

    pub(super) fn class_attribute_entry(
        &mut self,
        class: super::super::heap::ObjectId,
        name: &str,
    ) -> Result<Option<(super::super::heap::ObjectId, Value)>, String> {
        let Object::Class {
            attributes, mro, ..
        } = self.state.heap.get(class)?
        else {
            return Err("instance has an invalid class".into());
        };
        if let Some(value) = attributes.get(name) {
            return Ok(Some((class, *value)));
        }
        let ancestors = mro.clone();
        for ancestor in ancestors {
            self.charge_cpu(1)?;
            let Object::Class { attributes, .. } = self.state.heap.get(ancestor)? else {
                return Err("class MRO contains a non-class object".into());
            };
            if let Some(value) = attributes.get(name) {
                return Ok(Some((ancestor, *value)));
            }
        }
        Ok(None)
    }

    fn is_data_descriptor(&mut self, value: &Value) -> Result<bool, String> {
        let Some(id) = value.object_id() else {
            return Ok(false);
        };
        let object = self.state.heap.get(id)?.clone();
        Ok(match object {
            Object::Property { .. } => true,
            Object::Instance { class, .. } => {
                self.class_attribute(class, "__set__")?.is_some()
                    || self.class_attribute(class, "__delete__")?.is_some()
            }
            _ => false,
        })
    }

    pub(super) fn bind_descriptor(
        &mut self,
        descriptor: Value,
        receiver: Option<Value>,
        accessed_class: super::super::heap::ObjectId,
        defining_class: super::super::heap::ObjectId,
    ) -> Result<Value, String> {
        if matches!(
            descriptor.native_value(),
            Some(NativeValue::NativeMethod(method)) if method.name == "__new__"
        ) {
            return Ok(descriptor);
        }
        if matches!(
            descriptor.native_value(),
            Some(NativeValue::NativeMethod(_))
        ) {
            return match receiver {
                Some(receiver) => self.allocate_object(Object::DescriptorBoundMethod {
                    receiver,
                    descriptor,
                    owner: Some(defining_class),
                }),
                None => Ok(descriptor),
            };
        }
        let Some(id) = descriptor.object_id() else {
            return Ok(descriptor);
        };
        match self.state.heap.get(id)?.clone() {
            Object::Function { .. } => match receiver {
                Some(receiver) => self.allocate_object(Object::DescriptorBoundMethod {
                    receiver,
                    descriptor: Value::Object(id),
                    owner: Some(defining_class),
                }),
                None => Ok(descriptor),
            },
            Object::Property { getter, .. } => match receiver {
                Some(receiver) => self.invoke_value(getter, vec![receiver]),
                None => Ok(descriptor),
            },
            Object::StaticMethod { callable } => Ok(callable),
            Object::ClassMethod { callable } => {
                if let Some(function) = callable.object_id() {
                    if matches!(self.state.heap.get(function)?, Object::Function { .. }) {
                        return self.allocate_object(Object::DescriptorBoundMethod {
                            receiver: Value::Object(accessed_class),
                            descriptor: Value::Object(function),
                            owner: Some(defining_class),
                        });
                    }
                }
                Ok(callable)
            }
            Object::Instance { class, .. } => {
                let Some((get_owner, get)) = self.class_attribute_entry(class, "__get__")? else {
                    return Ok(descriptor);
                };
                let get = self.bind_descriptor(get, Some(descriptor), class, get_owner)?;
                self.invoke_value(
                    get,
                    vec![
                        receiver.unwrap_or(Value::None),
                        Value::Object(accessed_class),
                    ],
                )
            }
            _ => Ok(descriptor),
        }
    }

    fn invoke_value(&mut self, callable: Value, arguments: Vec<Value>) -> Result<Value, String> {
        match self.invoke_call(callable, arguments, Vec::new())? {
            CallResult::Value(value) => Ok(value),
            CallResult::Exit(status) => Err(format!("callable exited with status {status}")),
            CallResult::EnteredFrame => unreachable!("invoke_call is immediate"),
            CallResult::Blocked(_, _) | CallResult::Retry(_, _) => {
                unreachable!("immediate call cannot suspend")
            }
        }
    }

    pub(super) fn invoke_call(
        &mut self,
        callable: Value,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<CallResult, String> {
        let positional = arguments.len();
        let total = positional
            .checked_add(keyword_arguments.len())
            .ok_or("too many call arguments")?;
        let keyword_names = keyword_arguments
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        self.stack.push(callable);
        self.stack.extend(arguments);
        self.stack
            .extend(keyword_arguments.into_iter().map(|(_, value)| value));
        self.call(
            positional,
            &keyword_names,
            &vec![false; total],
            CallMode::Immediate,
        )
    }

    pub(super) fn invoke_slot(
        &mut self,
        receiver: &Value,
        slot: Slot,
        method_name: &str,
        arguments: Vec<Value>,
    ) -> Result<Option<Value>, String> {
        let type_id = self.type_id(receiver)?;
        let Some(slot_value) = self.state.types.slot(type_id, slot)? else {
            return Ok(None);
        };
        let slot_descriptor = match slot_value {
            SlotValue::NativeBinary(call) => {
                let [argument] = arguments.as_slice() else {
                    return Err(
                        "binary protocol slot received the wrong number of arguments".into(),
                    );
                };
                return call(self, *receiver, *argument)
                    .map_err(|error| self.record_native_error(error));
            }
            SlotValue::NativeTernary(call) => {
                let [first, second] = arguments.as_slice() else {
                    return Err(
                        "ternary protocol slot received the wrong number of arguments".into(),
                    );
                };
                return call(self, *receiver, *first, *second)
                    .map_err(|error| self.record_native_error(error));
            }
            SlotValue::NativeUnary(call) => {
                if !arguments.is_empty() {
                    return Err("unary protocol slot received arguments".into());
                }
                return call(self, *receiver).map_err(|error| self.record_native_error(error));
            }
            SlotValue::Descriptor(descriptor) => descriptor,
        };
        let Some(id) = receiver.object_id() else {
            return Ok(None);
        };
        let Object::Instance { class, .. } = self.state.heap.get(id)? else {
            return Ok(None);
        };
        let class = *class;
        let (defining_class, _) = self
            .class_attribute_entry(class, method_name)?
            .ok_or("cached type slot has no descriptor")?;
        let callable =
            self.bind_descriptor(slot_descriptor, Some(*receiver), class, defining_class)?;
        self.invoke_value(callable, arguments).map(Some)
    }

    pub(super) fn truth_value(&mut self, value: &Value) -> Result<bool, String> {
        if let Some(result) = self.invoke_slot(value, Slot::Bool, "__bool__", Vec::new())? {
            return result
                .bool_value()
                .ok_or_else(|| "__bool__ should return bool".into());
        }
        if let Some(result) = self.invoke_slot(value, Slot::Length, "__len__", Vec::new())? {
            let length = protocol::int_value(&self.state.heap, &result)
                .ok_or_else(|| "__len__ should return int".to_string())?;
            if length < 0 {
                return Err("__len__ should return >= 0".into());
            }
            return Ok(length != 0);
        }
        protocol::truth(&self.state.heap, value)
    }

    pub(super) fn repr_value(&mut self, value: &Value) -> Result<String, String> {
        if let Some(result) = self.invoke_slot(value, Slot::Repr, "__repr__", Vec::new())? {
            return protocol::string_value(&self.state.heap, &result)?
                .ok_or_else(|| "__repr__ should return str".into());
        }
        protocol::repr(&self.state.heap, value)
    }

    pub(super) fn display_value(&mut self, value: &Value) -> Result<String, String> {
        if let Some(result) = self.invoke_slot(value, Slot::String, "__str__", Vec::new())? {
            return protocol::string_value(&self.state.heap, &result)?
                .ok_or_else(|| "__str__ should return str".into());
        }
        if self
            .state
            .types
            .slot(self.type_id(value)?, Slot::Repr)?
            .is_some()
        {
            return self.repr_value(value);
        }
        protocol::display(&self.state.heap, value)
    }

    pub(super) fn compare_values(
        &mut self,
        left: &Value,
        right: &Value,
    ) -> Result<Ordering, String> {
        if let (Some(left_id), Some(right_id)) = (left.object_id(), right.object_id()) {
            let sequences = match (
                self.state.heap.get(left_id)?,
                self.state.heap.get(right_id)?,
            ) {
                (Object::List(left), Object::List(right))
                | (Object::Tuple(left), Object::Tuple(right)) => {
                    Some((left.clone(), right.clone()))
                }
                _ => None,
            };
            if let Some((left, right)) = sequences {
                for (left, right) in left.iter().zip(&right) {
                    self.charge_cpu(1)?;
                    if protocol::identical(left, right) {
                        continue;
                    }
                    let ordering = self.compare_values(left, right)?;
                    if ordering != Ordering::Equal {
                        return Ok(ordering);
                    }
                }
                return Ok(left.len().cmp(&right.len()));
            }
        }
        if let Some(equal) = self.invoke_slot(left, Slot::Equal, "__eq__", vec![*right])? {
            if self.truth_value(&equal)? {
                return Ok(Ordering::Equal);
            }
        }
        if let Some(less) = self.invoke_slot(left, Slot::LessThan, "__lt__", vec![*right])? {
            if self.truth_value(&less)? {
                return Ok(Ordering::Less);
            }
        }
        if let Some(less) = self.invoke_slot(right, Slot::LessThan, "__lt__", vec![*left])? {
            if self.truth_value(&less)? {
                return Ok(Ordering::Greater);
            }
        }
        protocol::compare(&self.state.heap, left, right)
    }

    fn super_attribute(
        &mut self,
        start_class: super::super::heap::ObjectId,
        receiver: &Value,
        name: &str,
    ) -> Result<
        (
            super::super::heap::ObjectId,
            Value,
            super::super::heap::ObjectId,
        ),
        String,
    > {
        let accessed_class = if let Some(id) = receiver.object_id() {
            match self.state.heap.get(id)? {
                Object::Instance { class, .. } => *class,
                Object::Class { .. } => id,
                _ => return Err("super() receiver is not an instance or class".into()),
            }
        } else {
            return Err("super() receiver is not an instance or class".into());
        };
        let Object::Class { mro, .. } = self.state.heap.get(accessed_class)? else {
            return Err("super() receiver has an invalid class".into());
        };
        let mut classes = Vec::with_capacity(mro.len().saturating_add(1));
        classes.push(accessed_class);
        classes.extend(mro.iter().copied());
        let start = classes
            .iter()
            .position(|class| *class == start_class)
            .ok_or("super(type, obj): obj is not an instance or subtype of type")?;
        for class in classes.into_iter().skip(start.saturating_add(1)) {
            self.charge_cpu(1)?;
            let Object::Class { attributes, .. } = self.state.heap.get(class)? else {
                return Err("super MRO contains a non-class object".into());
            };
            if let Some(value) = attributes.get(name) {
                return Ok((class, *value, accessed_class));
            }
        }
        if name == "__new__"
            && matches!(
                self.state.heap.get(start_class)?,
                Object::Class {
                    layout: ClassLayout::Type,
                    ..
                }
            )
        {
            if let Some(descriptor) = self
                .state
                .types
                .attribute(BuiltinType::Type.id(), "__new__")?
            {
                return Ok((start_class, descriptor, accessed_class));
            }
        }
        Err(format!("super object has no attribute {name:?}"))
    }

    /// Invoke a canonical builtin type object.
    ///
    /// Construction is centralized here so type identity, `type()`, and calling a type do not
    /// depend on the unrelated builtin-function dispatch table. Collection construction remains
    /// metered through the normal iterator and allocation paths.
    pub(super) fn call_builtin_type(
        &mut self,
        builtin_type: BuiltinType,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<CallResult, String> {
        if !keyword_arguments.is_empty() {
            return Err(format!(
                "{}() does not accept keyword arguments in this slice",
                builtin_type.name()
            ));
        }
        let value = match builtin_type {
            BuiltinType::Type => match arguments.as_slice() {
                [value] => self.type_of(value)?,
                [name, bases, namespace] => {
                    let name = protocol::string_value(&self.state.heap, name)?
                        .ok_or("type name must be a string")?;
                    self.new_type(
                        Value::Native(NativeValue::BuiltinType(BuiltinType::Type)),
                        name,
                        *bases,
                        *namespace,
                    )
                    .map_err(|error| error.to_string())?
                }
                _ => return Err("type() expects one or three arguments".into()),
            },
            BuiltinType::Object => {
                expect_arity(&arguments, 0, 0)?;
                return Err("direct object() instances are not implemented".into());
            }
            BuiltinType::None => {
                expect_arity(&arguments, 0, 0)?;
                Value::None
            }
            BuiltinType::Bool => {
                expect_arity(&arguments, 0, 1)?;
                Value::Bool(match arguments.first() {
                    Some(value) => self.truth_value(value)?,
                    None => false,
                })
            }
            BuiltinType::Int => {
                expect_arity(&arguments, 0, 2)?;
                if let Some(base) = arguments.get(1) {
                    let base = protocol::int_value(&self.state.heap, base).ok_or_else(|| {
                        self.record_native_error(PyError::type_error(
                            "int() base must be an integer",
                        ))
                    })?;
                    let text = protocol::string_value(&self.state.heap, &arguments[0])?
                        .ok_or_else(|| {
                            self.record_native_error(PyError::type_error(
                                "int() can't convert non-string with explicit base",
                            ))
                        })?;
                    self.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
                    let decimal = super::number::parse_integer_text(&text, base)
                        .map_err(|error| self.record_native_error(error))?;
                    self.new_integer(&decimal)
                        .map_err(|error| self.record_native_error(error))?
                } else {
                    match arguments.first() {
                        None => Value::Int(0),
                        Some(value) if self.is_bigint(value)? => *value,
                        Some(value)
                            if protocol::string_value(&self.state.heap, value)?.is_some() =>
                        {
                            let text =
                                protocol::string_value(&self.state.heap, value)?.expect("guarded");
                            self.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
                            let decimal = super::number::parse_integer_text(&text, 10)
                                .map_err(|error| self.record_native_error(error))?;
                            self.new_integer(&decimal)
                                .map_err(|error| self.record_native_error(error))?
                        }
                        Some(value) => Value::Int(
                            protocol::int_value(&self.state.heap, value)
                                .or_else(|| value.as_int())
                                .ok_or("int() argument is not supported")?,
                        ),
                    }
                }
            }
            BuiltinType::Float => {
                expect_arity(&arguments, 0, 1)?;
                let converted = match arguments.first() {
                    None => 0.0,
                    Some(value)
                        if matches!(
                            super::number::view(&self.state.heap, value),
                            Some(super::number::NumberRef::Float(_))
                        ) =>
                    {
                        let Some(super::number::NumberRef::Float(value)) =
                            super::number::view(&self.state.heap, value)
                        else {
                            unreachable!()
                        };
                        value
                    }
                    Some(value) if protocol::string_value(&self.state.heap, value)?.is_some() => {
                        protocol::string_value(&self.state.heap, value)?
                            .expect("guarded")
                            .parse::<f64>()
                            .map_err(|_| "could not convert string to float")?
                    }
                    Some(value) => self
                        .numeric_float(value)
                        .map_err(|_| "float() argument is not supported")?,
                };
                Value::Float(converted)
            }
            BuiltinType::String => {
                expect_arity(&arguments, 0, 1)?;
                let value = match arguments.first() {
                    Some(value) => self.display_value(value)?,
                    None => String::new(),
                };
                self.allocate_string(value)?
            }
            BuiltinType::Bytes | BuiltinType::ByteArray => {
                expect_arity(&arguments, 0, 2)?;
                let value = match arguments.as_slice() {
                    [] => Vec::new(),
                    [value] if protocol::bytes_value(&self.state.heap, value)?.is_some() => {
                        protocol::bytes_value(&self.state.heap, value)?.expect("guarded")
                    }
                    [value] if protocol::int_value(&self.state.heap, value).is_some() => {
                        let length = usize::try_from(
                            protocol::int_value(&self.state.heap, value).expect("guarded"),
                        )
                        .map_err(|_| "negative count")?;
                        self.reserve_result(length)?;
                        vec![0; length]
                    }
                    [value] => {
                        let items = self.iterable_values(value)?;
                        let mut bytes = Vec::with_capacity(items.len());
                        for item in items {
                            let byte = protocol::int_value(&self.state.heap, &item)
                                .and_then(|value| u8::try_from(value).ok())
                                .ok_or("bytes must be in range(0, 256)")?;
                            bytes.push(byte);
                        }
                        bytes
                    }
                    [value, encoding] => {
                        let text = protocol::string_value(&self.state.heap, value)?
                            .ok_or("encoding without a string argument")?;
                        let encoding = protocol::string_value(&self.state.heap, encoding)?
                            .ok_or("bytes() encoding must be a string")?;
                        if !matches!(encoding.to_ascii_lowercase().as_str(), "utf-8" | "utf8") {
                            return Err("only UTF-8 encoding is supported".into());
                        }
                        text.into_bytes()
                    }
                    _ => unreachable!("arity checked"),
                };
                if builtin_type == BuiltinType::Bytes {
                    self.allocate_bytes(value)?
                } else {
                    self.allocate_bytearray(value)?
                }
            }
            BuiltinType::List | BuiltinType::Tuple | BuiltinType::Set => {
                expect_arity(&arguments, 0, 1)?;
                let values = arguments
                    .first()
                    .map(|value| self.iterable_values(value))
                    .transpose()?
                    .unwrap_or_default();
                let object = match builtin_type {
                    BuiltinType::List => Object::List(values),
                    BuiltinType::Tuple => Object::Tuple(values),
                    BuiltinType::Set => {
                        let mut unique = Vec::new();
                        for value in values {
                            self.charge_cpu(1)?;
                            if self.find_value(&unique, &value)?.is_none() {
                                self.reserve_result(64)?;
                                unique.push(value);
                            }
                        }
                        Object::Set(unique)
                    }
                    _ => unreachable!(),
                };
                self.allocate_object(object)?
            }
            BuiltinType::Dict => {
                expect_arity(&arguments, 0, 1)?;
                let entries = match arguments.first() {
                    None => Vec::new(),
                    Some(value) if value.object_id().is_some() => {
                        match self.state.heap.get(value.object_id().unwrap())? {
                            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                                entries.to_vec()
                            }
                            _ => {
                                return Err("dict() argument is not a mapping in this slice".into())
                            }
                        }
                    }
                    Some(_) => return Err("dict() argument is not a mapping in this slice".into()),
                };
                self.allocate_object(Object::Dict(entries.into()))?
            }
            BuiltinType::Function
            | BuiltinType::Range
            | BuiltinType::Module
            | BuiltinType::Iterator
            | BuiltinType::Generator
            | BuiltinType::Exception
            | BuiltinType::Native
            | BuiltinType::Stream
            | BuiltinType::Environment
            | BuiltinType::ArgumentParser
            | BuiltinType::RaisesContext
            | BuiltinType::Property
            | BuiltinType::Regex
            | BuiltinType::Match
            | BuiltinType::Array => {
                return Err(format!("cannot create '{}' instances", builtin_type.name()));
            }
        };
        Ok(CallResult::Value(value))
    }

    fn class_type_id(&self, value: &Value) -> Result<Option<TypeId>, String> {
        Ok(match value.native_value() {
            Some(NativeValue::BuiltinType(builtin)) => Some(builtin.id()),
            Some(NativeValue::ValueKind(kind)) => self.state.types.value_kind_type_id(kind),
            _ if value.object_id().is_some() => {
                match self.state.heap.get(value.object_id().unwrap())? {
                    Object::Class { instance_type, .. } => Some(*instance_type),
                    _ => None,
                }
            }
            _ => None,
        })
    }

    /// Return the Python-level type independently of the value's physical storage shape.
    pub(super) fn type_id(&self, value: &Value) -> Result<TypeId, String> {
        if value.inline_string_len().is_some() {
            return Ok(BuiltinType::String.id());
        }
        Ok(match value.tag() {
            ValueTag::None => BuiltinType::None.id(),
            ValueTag::Bool => BuiltinType::Bool.id(),
            ValueTag::Int => BuiltinType::Int.id(),
            ValueTag::Float => BuiltinType::Float.id(),
            ValueTag::Registered => self
                .state
                .types
                .value_kind_type_id_by_index(value.registered_parts().expect("tag checked").0)
                .ok_or("invalid registered value kind")?,
            ValueTag::Native => match value.native_value().expect("native tag checked") {
                NativeValue::BuiltinType(_) | NativeValue::ValueKind(_) => BuiltinType::Type.id(),
                NativeValue::Function(_)
                | NativeValue::NativeFunction(_)
                | NativeValue::NativeMethod(_) => BuiltinType::Function.id(),
                NativeValue::Module(_) => BuiltinType::Module.id(),
                NativeValue::Stream(_) => BuiltinType::Stream.id(),
                NativeValue::Environment => BuiltinType::Environment.id(),
                _ => BuiltinType::Native.id(),
            },
            ValueTag::Object => self
                .state
                .heap
                .type_id(value.object_id().expect("object tag checked"))?,
            ValueTag::SmallString0
            | ValueTag::SmallString1
            | ValueTag::SmallString2
            | ValueTag::SmallString3
            | ValueTag::SmallString4
            | ValueTag::SmallString5
            | ValueTag::SmallString6
            | ValueTag::SmallString7
            | ValueTag::SmallString8
            | ValueTag::SmallString9
            | ValueTag::SmallString10
            | ValueTag::SmallString11
            | ValueTag::SmallString12
            | ValueTag::SmallString13
            | ValueTag::SmallString14
            | ValueTag::SmallString15 => unreachable!("handled above"),
        })
    }

    fn type_of(&self, value: &Value) -> Result<Value, String> {
        self.state.types.value(self.type_id(value)?)
    }

    pub(super) fn is_instance(&mut self, value: &Value, class: &Value) -> Result<bool, String> {
        if let Some(NativeValue::ExceptionType(ExceptionType(expected))) = class.native_value() {
            let actual =
                if let Some((kind, _)) = protocol::exception_parts(&self.state.heap, value)? {
                    Some(kind)
                } else {
                    self.user_exception_kind(value)?
                };
            let Some(actual) = actual else {
                return Ok(false);
            };
            let os_error = matches!(
                actual.as_str(),
                "OSError"
                    | "FileNotFoundError"
                    | "FileExistsError"
                    | "IsADirectoryError"
                    | "NotADirectoryError"
                    | "PermissionError"
            );
            return Ok(expected == "BaseException"
                || (expected == "Exception"
                    && actual != "BaseException"
                    && actual != "SystemExit")
                || expected == actual
                || (expected == "OSError" && os_error));
        }
        if let Some(class_id) = class.object_id() {
            if let Object::Tuple(classes) = self.state.heap.get(class_id)? {
                let classes = classes.clone();
                for class in classes {
                    self.charge_cpu(1)?;
                    if self.is_instance(value, &class)? {
                        return Ok(true);
                    }
                }
                return Ok(false);
            }
        }
        let class = self
            .class_type_id(class)?
            .ok_or("isinstance() requires a class argument")?;
        self.state.types.is_subclass(self.type_id(value)?, class)
    }

    pub(super) fn is_subclass(&mut self, class: &Value, base: &Value) -> Result<bool, String> {
        if let Some(base_id) = base.object_id() {
            if let Object::Tuple(bases) = self.state.heap.get(base_id)? {
                let bases = bases.clone();
                for base in bases {
                    self.charge_cpu(1)?;
                    if self.is_subclass(class, &base)? {
                        return Ok(true);
                    }
                }
                return Ok(false);
            }
        }
        let class = self
            .class_type_id(class)?
            .ok_or("issubclass() requires a class argument")?;
        let base = self
            .class_type_id(base)?
            .ok_or("issubclass() requires a class argument")?;
        self.state.types.is_subclass(class, base)
    }
}
