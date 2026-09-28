//! Runtime object operations for attributes, subscription, descriptors, classes, and types.

use std::collections::BTreeSet;

use super::super::heap::{DictViewKind, NamespaceTarget, ObjectId, ProxyTarget};
use super::{
    exception_types, expect_arity, protocol, range_length, select_string_slice, Arc,
    BuiltinSubscript, BuiltinType, CallMode, CallResult, ClassDefinition, ClassField, ClassLayout,
    CodeCaches, CodeRef, ComparisonOperator, ExceptionType, Execution, HashMap, LoadAttributeCache,
    NameId, NativeValue, Object, Ordering, PyError, PyRuntime, SlicePlan, Slot, SlotValue,
    SymbolId, TypeId, Value, ValueTag, Vm, MODELED_MAPPING_ENTRY_BYTES,
};

/// Items of a builtin container whose `repr` the VM renders item by item.
enum ContainerItems {
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Set(Vec<Value>),
    FrozenSet(Vec<Value>),
    Dict(Vec<(Value, Value)>),
}

/// Modeled bytes for one class attribute added after the class statement, beyond its name.
const CLASS_ATTRIBUTE_BYTES: u64 = 48;
impl ContainerItems {
    fn len(&self) -> usize {
        match self {
            Self::List(items) | Self::Tuple(items) | Self::Set(items) | Self::FrozenSet(items) => {
                items.len()
            }
            Self::Dict(entries) => entries.len(),
        }
    }

    /// CPython's text for a container met again while rendering itself.
    fn placeholder(&self) -> &'static str {
        match self {
            Self::List(_) => "[...]",
            Self::Tuple(_) => "(...)",
            Self::Set(_) => "set(...)",
            Self::FrozenSet(_) => "frozenset(...)",
            Self::Dict(_) => "{...}",
        }
    }
}

/// Whether user classes may derive from `builtin`, holding its values as
/// [`InstancePayload::Builtin`](super::super::heap::InstancePayload::Builtin).
pub(super) fn is_subclassable_builtin(builtin: BuiltinType) -> bool {
    matches!(
        builtin,
        BuiltinType::Int
            | BuiltinType::Tuple
            | BuiltinType::Dict
            | BuiltinType::String
            | BuiltinType::Bytes
            | BuiltinType::Float
            | BuiltinType::Complex
    )
}

impl Vm<'_> {
    pub(super) fn load_attribute(&mut self, name: &str) -> Result<(), String> {
        let owner = self.pop()?;
        let Some(value) = self.resolve_attribute(owner, name)? else {
            return Err(self.missing_attribute(&owner, name));
        };
        self.stack.push(value);
        Ok(())
    }

    /// Report an attribute lookup that found nothing. Modules, user classes and their instances
    /// raise CPython's `AttributeError`, so `try`/`except` fallbacks for a missing module member
    /// work. On a builtin value a missing name is almost always a method shellsim does not model,
    /// so it stays an unsupported-feature error that user code cannot catch and misread as
    /// absence; `hasattr` and `getattr` defaults still observe the absence without raising.
    pub(super) fn missing_attribute(&mut self, owner: &Value, name: &str) -> String {
        if let Some(NativeValue::Module(module)) = owner.native_value() {
            let message = format!("module '{}' has no attribute '{name}'", module.name);
            return self.raise_exception("AttributeError", message);
        }
        let message = match owner.object_id().map(|id| self.state.heap.get(id)) {
            Some(Ok(Object::Module { name: module, .. })) => {
                format!("module '{module}' has no attribute '{name}'")
            }
            Some(Ok(Object::Class {
                name: class_name, ..
            })) => format!("type object '{class_name}' has no attribute '{name}'"),
            Some(Ok(Object::Instance { .. } | Object::Bare)) => match self.type_name_of(owner) {
                Ok(type_name) => format!("'{type_name}' object has no attribute '{name}'"),
                Err(error) => return error,
            },
            Some(Err(error)) => return error,
            _ => return format!("attribute {name:?} is not implemented"),
        };
        self.raise_exception("AttributeError", message)
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
        let Some(value) = self.resolve_attribute_by_symbol(owner, symbol, name)? else {
            return Err(self.missing_attribute(&owner, name));
        };
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
        let class_type = self
            .class_type_id(&Value::Object(class))?
            .ok_or("instance has no registered class")?;
        if let Some((_, descriptor)) = self.type_lookup(class_type, name)? {
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

    /// Resolve an attribute where an `AttributeError` raised during the lookup means the
    /// attribute is absent, as in CPython's `PyObject_GetOptionalAttr`. `getattr` defaults,
    /// `hasattr` and `from module import name` use this, so a module `__getattr__` or property
    /// that raises `AttributeError` reads as a missing name while other exceptions propagate.
    pub(super) fn resolve_optional_attribute(
        &mut self,
        owner: Value,
        name: &str,
    ) -> Result<Option<Value>, String> {
        let error = match self.resolve_attribute(owner, name) {
            Ok(value) => return Ok(value),
            Err(error) => error,
        };
        let Some(exception) = self.pending_exception.take() else {
            return Err(error);
        };
        let attribute_error =
            Value::Native(NativeValue::ExceptionType(ExceptionType("AttributeError")));
        if self.exception_type_matches(attribute_error, &exception)? {
            return Ok(None);
        }
        self.pending_exception = Some(exception);
        Err(error)
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
        let getattribute: Result<Option<Value>, String> = if owner.object_id().is_some_and(|id| {
            matches!(
                self.state.heap.get(id),
                Ok(Object::Instance { .. } | Object::Class { .. })
            )
        }) && self
            .state
            .types
            .slot(self.type_id(&owner)?, Slot::GetAttribute)?
            .is_some()
        {
            let name = self.allocate_string(name.to_string())?;
            self.invoke_slot(&owner, Slot::GetAttribute, "__getattribute__", vec![name])
        } else {
            self.lookup_attribute_default(owner, symbol, name)
        };
        let found = match getattribute {
            Ok(found) => found,
            Err(error) => {
                let Some(exception) = self.pending_exception.take() else {
                    return Err(error);
                };
                let attribute_error =
                    Value::Native(NativeValue::ExceptionType(ExceptionType("AttributeError")));
                if self.exception_type_matches(attribute_error, &exception)? {
                    if let Some(value) = self.call_type_getattr_hook(owner, name)? {
                        return Ok(Some(value));
                    }
                }
                self.pending_exception = Some(exception);
                return Err(error);
            }
        };
        match found {
            Some(value) => Ok(Some(value)),
            None => self.call_type_getattr_hook(owner, name),
        }
    }

    /// The default object lookup without the outer `__getattr__` fallback. Direct calls to
    /// `object.__getattribute__` and ordinary attribute reads share this descriptor algorithm.
    pub(super) fn lookup_attribute_default(
        &mut self,
        owner: Value,
        symbol: Option<SymbolId>,
        name: &str,
    ) -> Result<Option<Value>, String> {
        let found = self.lookup_attribute(owner, symbol, name)?;
        if found.is_none() && name == "__class__" {
            return self.type_of(&owner).map(Some);
        }
        Ok(found)
    }

    fn call_type_getattr_hook(
        &mut self,
        owner: Value,
        name: &str,
    ) -> Result<Option<Value>, String> {
        let owner_type = self.type_id(&owner)?;
        let Some((defining_type, hook)) = self.type_lookup(owner_type, "__getattr__")? else {
            return Ok(None);
        };
        let hook = self
            .bind_type_attribute(hook, Some(owner), owner_type, defining_type)?
            .ok_or("__getattr__ descriptor has no value")?;
        let name = self.allocate_string(name.to_string())?;
        self.invoke_value(hook, vec![name]).map(Some)
    }

    fn lookup_attribute(
        &mut self,
        owner: Value,
        symbol: Option<SymbolId>,
        name: &str,
    ) -> Result<Option<Value>, String> {
        if let Some(NativeValue::Module(module)) = owner.native_value() {
            if name == "__dict__" {
                let proxy = Object::MappingProxy(ProxyTarget::NativeModule(module));
                return self.allocate_object(proxy).map(Some);
            }
            if let Some(function) = module.function(name) {
                return Ok(Some(Value::Native(NativeValue::NativeFunction(function))));
            }
            if let Some(value) = module.value(name) {
                let value = value.get(self).map_err(|error| error.to_string())?;
                return Ok(Some(value));
            }
        }
        if let Some(NativeValue::BuiltinType(builtin)) = owner.native_value() {
            if let Some((_, value)) = self.type_lookup(builtin.id(), name)? {
                if let Some(NativeValue::NativeClassMethod(method)) = value.native_value() {
                    return self.bind_native_class_method(owner, method).map(Some);
                }
                return Ok(Some(value));
            }
        }
        if let Some(NativeValue::ExceptionType(_)) = owner.native_value() {
            // Every builtin exception class shares the `BaseException` methods, then `object`'s.
            for ancestor in [BuiltinType::Exception.id(), BuiltinType::Object.id()] {
                if let Some(value) = self.state.types.attribute(ancestor, name)? {
                    return Ok(Some(value));
                }
            }
        }
        if let Some(NativeValue::ValueKind(kind)) = owner.native_value() {
            if let Some(type_id) = self.state.types.value_kind_type_id(kind) {
                if let Some((_, value)) = self.type_lookup(type_id, name)? {
                    return Ok(Some(value));
                }
            }
        }
        let owner_type = self.type_id(&owner)?;
        // These objects apply their own MRO or proxy lookup in the arms below.
        let owner_has_custom_lookup = owner.object_id().is_some_and(|id| {
            matches!(
                self.state.heap.get(id),
                Ok(Object::Class { .. } | Object::Instance { .. } | Object::Super { .. })
            )
        });
        if !owner_has_custom_lookup {
            if let Some((defining_type, descriptor)) = self.type_lookup(owner_type, name)? {
                match descriptor.native_value() {
                    Some(
                        NativeValue::NativeMethod(_)
                        | NativeValue::NativeClassMethod(_)
                        | NativeValue::SlotWrapper { .. },
                    ) => {
                        return self.bind_type_attribute(
                            descriptor,
                            Some(owner),
                            owner_type,
                            defining_type,
                        );
                    }
                    // Native getters are data descriptors. Builtin receivers have no instance
                    // dictionary, so reaching the type table first gives CPython precedence.
                    Some(NativeValue::NativeGetter(getter)) => {
                        return self.call_native_getter(getter, owner);
                    }
                    _ => {}
                }
            }
        }
        if let Some(id) = owner.object_id() {
            match self.state.heap.get(id)?.clone() {
                Object::Module { scope, .. } => {
                    if name == "__dict__" {
                        let namespace = Object::NamespaceDict(NamespaceTarget::Scope(scope));
                        return self.allocate_object(namespace).map(Some);
                    }
                    if let Some(value) = self.state.heap.scope_get(scope, name).copied() {
                        return Ok(Some(value));
                    }
                    // PEP 562: a module-level `__getattr__` supplies names the module does not
                    // define, which is how packages import submodules lazily. It runs after
                    // the type's attributes, of which modules model only `__class__`.
                    let hook = self.state.heap.scope_get(scope, "__getattr__").copied();
                    let Some(hook) = hook.filter(|_| name != "__class__") else {
                        return Ok(None);
                    };
                    let name = self.allocate_string(name.to_string())?;
                    return self.invoke_value(hook, vec![name]).map(Some);
                }
                Object::Function {
                    name: function_name,
                    closure,
                    attributes,
                    ..
                } => {
                    if name == "__name__" {
                        return Ok(Some(self.allocate_string(function_name)?));
                    }
                    if let Some(value) = attributes.get(name) {
                        return Ok(Some(*value));
                    }
                    if name == "__module__" {
                        return self.module_name_of(closure);
                    }
                }
                Object::Class {
                    name: class_name, ..
                } => {
                    let class_type = self
                        .class_type_id(&owner)?
                        .ok_or("class has no registered type")?;
                    let metaclass_entry = self.type_lookup(owner_type, name)?;
                    if let Some((defining_type, descriptor)) = metaclass_entry {
                        if self.is_data_descriptor(&descriptor)? {
                            return self.bind_type_attribute(
                                descriptor,
                                Some(owner),
                                owner_type,
                                defining_type,
                            );
                        }
                    }
                    if name == "__name__" {
                        return Ok(Some(self.allocate_string(class_name)?));
                    }
                    // `type.__dict__` is a data descriptor, so it wins over the class's own
                    // attributes. The proxy is read-only; `setattr(cls, ...)` still works.
                    if name == "__dict__" {
                        let proxy = Object::MappingProxy(ProxyTarget::Class(id));
                        return self.allocate_object(proxy).map(Some);
                    }
                    if let Some(value) = self.class_metadata(id, name)? {
                        return Ok(Some(value));
                    }
                    if let Some((defining_type, descriptor)) = self.type_lookup(class_type, name)? {
                        return self.bind_type_attribute(
                            descriptor,
                            None,
                            class_type,
                            defining_type,
                        );
                    }
                    if let Some((defining_type, descriptor)) = metaclass_entry {
                        return self.bind_type_attribute(
                            descriptor,
                            Some(owner),
                            owner_type,
                            defining_type,
                        );
                    }
                    return Ok(None);
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
                    // `object.__dict__` is a data descriptor on every class, so it wins over
                    // anything stored on the instance.
                    if name == "__dict__" {
                        let namespace = Object::NamespaceDict(NamespaceTarget::Instance(id));
                        return self.allocate_object(namespace).map(Some);
                    }
                    let class_type = self
                        .class_type_id(&Value::Object(class))?
                        .ok_or("instance has no registered class")?;
                    let class_entry = self.type_lookup(class_type, name)?;
                    if let Some((defining_type, descriptor)) = class_entry {
                        if self.is_data_descriptor(&descriptor)? {
                            return self.bind_type_attribute(
                                descriptor,
                                Some(owner),
                                class_type,
                                defining_type,
                            );
                        }
                    }
                    let instance_value = match symbol {
                        Some(symbol) => self.state.heap.attribute_by_symbol(id, symbol)?.copied(),
                        None => None,
                    };
                    if let Some(value) = instance_value {
                        return Ok(Some(value));
                    }
                    let Some((defining_type, descriptor)) = class_entry else {
                        return Ok(None);
                    };
                    return self.bind_type_attribute(
                        descriptor,
                        Some(owner),
                        class_type,
                        defining_type,
                    );
                }
                Object::Super {
                    start_class,
                    receiver,
                } => {
                    let (defining_type, descriptor, accessed_type) =
                        self.super_attribute(start_class, &receiver, name)?;
                    return self.bind_type_attribute(
                        descriptor,
                        Some(receiver),
                        accessed_type,
                        defining_type,
                    );
                }
                // A bound method exposes its receiver and function and reads other attributes,
                // such as `__name__`, from the function, as CPython's method objects do.
                Object::DescriptorBoundMethod {
                    receiver,
                    descriptor,
                    ..
                } if name != "__class__" => {
                    return match name {
                        "__self__" => Ok(Some(receiver)),
                        "__func__" => Ok(Some(descriptor)),
                        _ => self.lookup_attribute(descriptor, None, name),
                    };
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
                    return Ok(Some(component));
                }
                _ => {}
            }
        }
        if let Some(value) = self.native_type_metadata(owner, name)? {
            return Ok(Some(value));
        }
        let native_name =
            match owner.native_value() {
                Some(NativeValue::UnitTestBase) if name == "__name__" => Some("TestCase"),
                // Builtin types and native value kinds record their qualified names, as `repr` shows
                // them; `__name__` is the last component.
                Some(NativeValue::BuiltinType(builtin)) if name == "__name__" => {
                    builtin.name().rsplit('.').next()
                }
                Some(NativeValue::ValueKind(kind)) if name == "__name__" => {
                    kind.name.rsplit('.').next()
                }
                Some(NativeValue::ExceptionType(ExceptionType(exception_name)))
                    if name == "__name__" =>
                {
                    Some(exception_name)
                }
                Some(NativeValue::Function(builtin)) if name == "__name__" => Some(builtin.name()),
                Some(NativeValue::NativeFunction(function)) if name == "__name__" => {
                    Some(function.name)
                }
                Some(
                    NativeValue::NativeMethod(method) | NativeValue::NativeClassMethod(method),
                ) if name == "__name__" => Some(method.name),
                _ => None,
            };
        let value = native_name
            .map(str::to_string)
            .map(|name| self.allocate_string(name))
            .transpose()?;
        Ok(value)
    }

    /// Assign `owner.name = value`. A class that defines `__setattr__` receives the assignment;
    /// otherwise [`Vm::store_attribute_default`] performs it.
    pub(super) fn store_attribute_by_symbol(
        &mut self,
        owner: Value,
        symbol: SymbolId,
        name: &str,
        value: Value,
    ) -> Result<(), String> {
        if let Some(Object::Instance { class, .. }) = owner
            .object_id()
            .map(|id| self.state.heap.get(id))
            .transpose()?
        {
            let class = *class;
            if let Some((defining_class, hook)) =
                self.class_attribute_entry(class, "__setattr__")?
            {
                let hook = self.bind_descriptor(hook, Some(owner), class, defining_class)?;
                let name = self.allocate_string(name.to_string())?;
                self.invoke_value(hook, vec![name, value])?;
                return Ok(());
            }
        }
        self.store_attribute_default(owner, symbol, name, value)
    }

    /// The assignment `object.__setattr__` performs: a data descriptor's setter, or else the
    /// instance's own attribute storage.
    pub(super) fn store_attribute_default(
        &mut self,
        owner: Value,
        symbol: SymbolId,
        name: &str,
        value: Value,
    ) -> Result<(), String> {
        let class = match owner.object_id() {
            Some(id) => match self.state.heap.get(id)? {
                Object::Instance { class, .. } => Some((id, *class)),
                _ => None,
            },
            None => None,
        };
        let Some((id, class)) = class else {
            if let Some(id) = owner.object_id() {
                match self.state.heap.get(id)? {
                    Object::Class { .. } => return self.set_class_attribute(id, name, Some(value)),
                    Object::Function { .. } => {
                        return self.set_function_attribute(id, name, Some(value))
                    }
                    // `BaseException.args` is writable and stores any iterable as a tuple.
                    Object::Exception { args, .. } if name == "args" => {
                        let current = args.len();
                        let items = self.iterable_values(&value)?;
                        let bytes = u64::try_from(items.len().saturating_sub(current))
                            .unwrap_or(u64::MAX)
                            .saturating_mul(super::MODELED_VALUE_BYTES);
                        self.state.heap.reserve_object_growth(
                            id,
                            bytes,
                            &mut self.interp.resources,
                        )?;
                        let Object::Exception { args, .. } = self.state.heap.get_mut(id)? else {
                            unreachable!("checked above")
                        };
                        *args = items;
                        return Ok(());
                    }
                    _ => {}
                }
            }
            return Err(self.reject_builtin_attribute_store(owner, name));
        };
        if name == "__dict__" {
            return self.replace_instance_attributes(id, value);
        }
        let class_type = self
            .class_type_id(&Value::Object(class))?
            .ok_or("instance has no registered class")?;
        if let Some((_, descriptor)) = self.type_lookup(class_type, name)? {
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
                        let message = format!(
                            "property '{name}' of '{}' object has no setter",
                            self.type_name_of(&owner)?
                        );
                        return Err(self.raise_exception("AttributeError", message));
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
            if matches!(
                descriptor.native_value(),
                Some(NativeValue::NativeGetter(_))
            ) && !(name == "args" && self.user_exception_base(&owner)?.is_some())
            {
                let type_name = self.type_name_of(&owner)?;
                return Err(self.raise_exception(
                    "AttributeError",
                    format!("attribute '{name}' of '{type_name}' objects is not writable"),
                ));
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

    /// `obj.__dict__ = mapping`: replace every attribute of instance `id` with `mapping`'s
    /// entries. The entries are copied, so later changes to `mapping` do not reach the instance;
    /// CPython instead makes the instance adopt the dict itself.
    fn replace_instance_attributes(&mut self, id: ObjectId, mapping: Value) -> Result<(), String> {
        let source = mapping
            .object_id()
            .map(|source| self.state.heap.get(source))
            .transpose()?;
        let entries = match source {
            Some(Object::Dict(entries) | Object::DefaultDict { entries, .. }) => {
                let entries = entries.to_vec();
                let mut named = Vec::with_capacity(entries.len());
                for (key, value) in entries {
                    let Some(name) = protocol::string_value(&self.state.heap, &key)? else {
                        let message = format!(
                            "namespace keys must be str, not {}",
                            self.type_name_of(&key)?
                        );
                        return Err(self.raise_exception("TypeError", message));
                    };
                    named.push((name, value));
                }
                named
            }
            Some(Object::NamespaceDict(target)) => {
                let target = *target;
                self.namespace_entries(target)?
            }
            _ => {
                let message = format!(
                    "__dict__ must be set to a dictionary, not a '{}'",
                    self.type_name_of(&mapping)?
                );
                return Err(self.raise_exception("TypeError", message));
            }
        };
        let current = self.state.heap.instance_attribute_names(id)?;
        self.charge_cpu(u64::try_from(current.len() + entries.len()).unwrap_or(u64::MAX))?;
        for name in current {
            self.namespace_delete(NamespaceTarget::Instance(id), &name)?;
        }
        for (name, value) in entries {
            self.namespace_store(NamespaceTarget::Instance(id), name, value)?;
        }
        Ok(())
    }

    /// Delete `owner.name`. A class that defines `__delattr__` receives the deletion;
    /// otherwise [`Vm::delete_attribute_default`] performs it.
    pub(super) fn delete_attribute_by_symbol(
        &mut self,
        owner: Value,
        symbol: SymbolId,
        name: &str,
    ) -> Result<(), String> {
        if let Some(Object::Instance { class, .. }) = owner
            .object_id()
            .map(|id| self.state.heap.get(id))
            .transpose()?
        {
            let class = *class;
            if let Some((defining_class, hook)) =
                self.class_attribute_entry(class, "__delattr__")?
            {
                let hook = self.bind_descriptor(hook, Some(owner), class, defining_class)?;
                let name = self.allocate_string(name.to_string())?;
                self.invoke_value(hook, vec![name])?;
                return Ok(());
            }
        }
        self.delete_attribute_default(owner, symbol, name)
    }

    /// The deletion `object.__delattr__` performs: a data descriptor's `__delete__`, or else
    /// removal from the instance's own attributes. Classes and modules delete from their
    /// namespaces.
    pub(super) fn delete_attribute_default(
        &mut self,
        owner: Value,
        symbol: SymbolId,
        name: &str,
    ) -> Result<(), String> {
        let Some(id) = owner.object_id() else {
            return Err(self.reject_builtin_attribute_store(owner, name));
        };
        let class = match self.state.heap.get(id)? {
            Object::Instance { class, .. } => *class,
            Object::Class { .. } => return self.set_class_attribute(id, name, None),
            Object::Function { .. } => return self.set_function_attribute(id, name, None),
            Object::Module { scope, .. } => {
                let scope = *scope;
                if self.state.heap.scope_remove(scope, name)?.is_none() {
                    return Err(self.missing_attribute(&owner, name));
                }
                return Ok(());
            }
            _ => return Err(self.reject_builtin_attribute_store(owner, name)),
        };
        let class_type = self
            .class_type_id(&Value::Object(class))?
            .ok_or("instance has no registered class")?;
        if let Some((_, descriptor)) = self.type_lookup(class_type, name)? {
            if let Some(descriptor_id) = descriptor.object_id() {
                match self.state.heap.get(descriptor_id)?.clone() {
                    Object::Property { .. } => {
                        let message = format!(
                            "property '{name}' of '{}' object has no deleter",
                            self.type_name_of(&owner)?
                        );
                        return Err(self.raise_exception("AttributeError", message));
                    }
                    Object::Instance {
                        class: descriptor_class,
                        ..
                    } => {
                        if let Some((delete_owner, delete)) =
                            self.class_attribute_entry(descriptor_class, "__delete__")?
                        {
                            let delete = self.bind_descriptor(
                                delete,
                                Some(descriptor),
                                descriptor_class,
                                delete_owner,
                            )?;
                            self.invoke_value(delete, vec![owner])?;
                            return Ok(());
                        }
                    }
                    _ => {}
                }
            }
            if matches!(
                descriptor.native_value(),
                Some(NativeValue::NativeGetter(_))
            ) {
                let type_name = self.type_name_of(&owner)?;
                return Err(self.raise_exception(
                    "AttributeError",
                    format!("attribute '{name}' of '{type_name}' objects is not deletable"),
                ));
            }
        }
        if self
            .state
            .heap
            .remove_attribute_by_symbol(id, symbol, &mut self.interp.resources)?
            .is_none()
        {
            return Err(self.missing_attribute(&owner, name));
        }
        Ok(())
    }

    /// Assign (`Some`) or delete (`None`) a class attribute after the class statement. The
    /// slots derived from dunder methods are recomputed for the class and its subclasses, and
    /// cached instance lookups are dropped, since a new data descriptor can shadow an instance
    /// attribute.
    /// Assign (`Some`) or delete (`None`) an attribute of a function object. `__name__` renames
    /// the function; every other name lives in the function's own `__dict__`.
    fn set_function_attribute(
        &mut self,
        function: ObjectId,
        name: &str,
        value: Option<Value>,
    ) -> Result<(), String> {
        if name == "__name__" {
            let Some(text) = value
                .map(|value| protocol::string_ref(&self.state.heap, &value))
                .transpose()?
                .flatten()
                .map(|text| text.as_str().to_string())
            else {
                return Err(
                    self.raise_exception("TypeError", "__name__ must be set to a string object")
                );
            };
            let Object::Function { name, .. } = self.state.heap.get_mut(function)? else {
                unreachable!("checked by the caller")
            };
            *name = text;
            return Ok(());
        }
        let Object::Function { attributes, .. } = self.state.heap.get(function)? else {
            return Err("function attribute store on a non-function".into());
        };
        let exists = attributes.contains_key(name);
        match value {
            Some(value) => {
                if !exists {
                    let bytes = u64::try_from(name.len())
                        .unwrap_or(u64::MAX)
                        .saturating_add(MODELED_MAPPING_ENTRY_BYTES);
                    self.state.heap.reserve_object_growth(
                        function,
                        bytes,
                        &mut self.interp.resources,
                    )?;
                }
                let Object::Function { attributes, .. } = self.state.heap.get_mut(function)? else {
                    unreachable!("checked above")
                };
                attributes.insert(name.to_string(), value);
            }
            None if exists => {
                let Object::Function { attributes, .. } = self.state.heap.get_mut(function)? else {
                    unreachable!("checked above")
                };
                attributes.remove(name);
            }
            None => {
                let message = format!("'function' object has no attribute '{name}'");
                return Err(self.raise_exception("AttributeError", message));
            }
        }
        Ok(())
    }

    fn set_class_attribute(
        &mut self,
        class: ObjectId,
        name: &str,
        value: Option<Value>,
    ) -> Result<(), String> {
        if matches!(name, "__name__" | "__bases__" | "__mro__") {
            return Err(format!(
                "assigning or deleting a class's {name} is not supported"
            ));
        }
        let Object::Class {
            attributes,
            instance_type,
            ..
        } = self.state.heap.get(class)?
        else {
            return Err("class attribute store on a non-class".into());
        };
        let instance_type = *instance_type;
        match value {
            Some(value) => {
                if !attributes.contains_key(name) {
                    let bytes = u64::try_from(name.len())
                        .unwrap_or(u64::MAX)
                        .saturating_add(CLASS_ATTRIBUTE_BYTES);
                    self.state.heap.reserve_object_growth(
                        class,
                        bytes,
                        &mut self.interp.resources,
                    )?;
                }
                let Object::Class { attributes, .. } = self.state.heap.get_mut(class)? else {
                    unreachable!("checked above")
                };
                attributes.insert(name.to_string(), value);
            }
            None => {
                let Object::Class { attributes, .. } = self.state.heap.get_mut(class)? else {
                    unreachable!("checked above")
                };
                if attributes.remove(name).is_none() {
                    return Err(self.missing_attribute(&Value::Object(class), name));
                }
            }
        }
        self.charge_cpu(u64::try_from(self.state.types.len()).unwrap_or(u64::MAX))?;
        for type_id in self.state.types.subtypes(instance_type) {
            let Some(id) = self.state.types.value(type_id)?.object_id() else {
                continue;
            };
            let state = &mut *self.state;
            let Object::Class { attributes, .. } = state.heap.get(id)? else {
                continue;
            };
            state.types.replace_slots(type_id, attributes)?;
        }
        for cache in &mut self.execution.code_caches {
            cache.attributes = None;
        }
        Ok(())
    }

    /// Raise CPython's `AttributeError` for a store on a receiver without an instance
    /// dictionary: data descriptors are not writable, other type attributes are read-only, and
    /// new names have nowhere to go.
    fn reject_builtin_attribute_store(&mut self, owner: Value, name: &str) -> String {
        let attribute = self
            .type_id(&owner)
            .and_then(|owner_type| self.type_lookup(owner_type, name))
            .ok()
            .flatten()
            .map(|(_, value)| value);
        let type_name = match self.type_name_of(&owner) {
            Ok(type_name) => type_name,
            Err(error) => return error,
        };
        let message = match attribute.and_then(|value| value.native_value()) {
            Some(NativeValue::NativeGetter(getter)) => format!(
                "attribute '{}' of '{}' objects is not writable",
                getter.name, getter.owner
            ),
            _ if attribute.is_some()
                || matches!(self.resolve_attribute(owner, name), Ok(Some(_))) =>
            {
                format!("'{type_name}' object attribute '{name}' is read-only")
            }
            _ => format!(
                "'{type_name}' object has no attribute '{name}' and no __dict__ for setting new \
                 attributes"
            ),
        };
        self.raise_exception("AttributeError", message)
    }

    /// `value`, or the builtin value it holds when it is an instance of a builtin subclass such
    /// as a `tuple` subclass. Builtin operations that the class does not override act on it.
    pub(super) fn builtin_view(&self, value: Value) -> Result<Value, String> {
        Ok(protocol::builtin_payload(&self.state.heap, &value)?.unwrap_or(value))
    }

    pub(super) fn load_subscript(&mut self) -> Result<(), String> {
        let index = self.pop()?;
        let owner = self.pop()?;
        let value = self.subscript_value(owner, index)?;
        self.stack.push(value);
        Ok(())
    }

    /// `owner[index]`: the owner's `__getitem__` or its builtin subscript.
    pub(super) fn subscript_value(&mut self, owner: Value, index: Value) -> Result<Value, String> {
        if let Some(value) = self.invoke_slot(&owner, Slot::GetItem, "__getitem__", vec![index])? {
            return Ok(value);
        }
        let subject = owner;
        let owner = self.builtin_view(owner)?;
        // A slice is an ordinary key to a mapping; only sequences slice with it.
        let mapping = match owner.object_id() {
            Some(id) => matches!(
                self.state.heap.get(id)?,
                Object::Dict(_) | Object::DefaultDict { .. }
            ),
            None => false,
        };
        if !mapping {
            if let Some((start, stop, step)) = self.slice_bounds(&index)? {
                let value = self.load_builtin_slice(owner, start, stop, step)?;
                return Ok(value);
            }
        }
        let index = self.sequence_index(&owner, index)?;
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
        } else if let Some(character) = self.string_subscript(&owner, &index)? {
            self.allocate_string(character.to_string())?
        } else if let Some(id) = owner.object_id() {
            let target = match self.state.heap.get(id)? {
                Object::List(values) | Object::Tuple(values) => {
                    let sequence = if matches!(self.state.heap.get(id)?, Object::List(_)) {
                        "list"
                    } else {
                        "tuple"
                    };
                    let Some(index) = protocol::int_value(&self.state.heap, &index) else {
                        let message = format!(
                            "{sequence} indices must be integers or slices, not {}",
                            self.type_name_of(&index)?
                        );
                        return Err(self.raise_exception("TypeError", message));
                    };
                    let len = values.len() as i64;
                    let index = if index < 0 { len + index } else { index };
                    match usize::try_from(index)
                        .ok()
                        .and_then(|index| values.get(index))
                    {
                        Some(value) => BuiltinSubscript::Value(*value),
                        None => {
                            return Err(self.raise_exception(
                                "IndexError",
                                format!("{sequence} index out of range"),
                            ))
                        }
                    }
                }
                Object::Range { start, stop, step } => {
                    let length = range_length(*start, *stop, *step)?;
                    let Some(index) = protocol::int_value(&self.state.heap, &index) else {
                        let message = format!(
                            "range indices must be integers or slices, not {}",
                            self.type_name_of(&index)?
                        );
                        return Err(self.raise_exception("TypeError", message));
                    };
                    let index = if index < 0 {
                        i128::try_from(length).map_err(|_| "range is too large")?
                            + i128::from(index)
                    } else {
                        i128::from(index)
                    };
                    if index < 0 || index >= i128::try_from(length).unwrap_or(i128::MAX) {
                        return Err(
                            self.raise_exception("IndexError", "range object index out of range")
                        );
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
                Object::Bare
                | Object::String(_)
                | Object::Bytes(_)
                | Object::ByteArray(_)
                | Object::Slice { .. }
                | Object::Exception { .. }
                | Object::FrozenSet(_)
                | Object::BigInt(_)
                | Object::Complex { .. }
                | Object::Function { .. }
                | Object::Class { .. }
                | Object::Instance { .. }
                | Object::DescriptorBoundMethod { .. }
                | Object::Iterator { .. }
                | Object::SequenceIterator { .. }
                | Object::RangeIterator { .. }
                | Object::CountIterator { .. }
                | Object::StreamIterator { .. }
                | Object::CallableIterator { .. }
                | Object::Generator { .. }
                | Object::Module { .. }
                | Object::ArrayStorage(_)
                | Object::Array { .. }
                | Object::WideValue { .. }
                | Object::Regex { .. }
                | Object::Match { .. }
                | Object::ArgumentParser { .. }
                | Object::Namespace { .. }
                | Object::EnumMember { .. }
                | Object::RaisesContext { .. }
                | Object::NamespaceDict(_)
                | Object::DictView { .. }
                | Object::MappingProxy(_) => BuiltinSubscript::Unsupported,
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
                    } else if let Some(value) = self.missing_key(&subject, index)? {
                        value
                    } else {
                        return Err(self.raise_exception_args("KeyError", vec![index]));
                    }
                }
                BuiltinSubscript::Set | BuiltinSubscript::Unsupported => {
                    return Err(self.raise_object_type_error(&owner, "is not subscriptable"))
                }
            }
        } else {
            return Err(self.raise_object_type_error(&owner, "is not subscriptable"));
        };
        Ok(value)
    }

    /// The value a dict subclass's `__missing__(key)` supplies for an absent key, or `None` when
    /// `subject` is not an instance whose class defines `__missing__`.
    fn missing_key(&mut self, subject: &Value, key: Value) -> Result<Option<Value>, String> {
        let Some(id) = subject.object_id() else {
            return Ok(None);
        };
        let Object::Instance { class, .. } = self.state.heap.get(id)? else {
            return Ok(None);
        };
        let class = *class;
        if self.class_attribute(class, "__missing__")?.is_none() {
            return Ok(None);
        }
        let method = self
            .resolve_attribute(*subject, "__missing__")?
            .ok_or("__missing__ disappeared during lookup")?;
        self.invoke_value(method, vec![key]).map(Some)
    }

    /// Index a string, raising CPython's errors for a bad index. `None` means `owner` is not a
    /// string.
    fn string_subscript(&mut self, owner: &Value, index: &Value) -> Result<Option<char>, String> {
        match protocol::string_index(&self.state.heap, owner, index)? {
            protocol::StringIndex::NotString => Ok(None),
            protocol::StringIndex::Character(character) => Ok(Some(character)),
            protocol::StringIndex::NotInteger => {
                let message = format!(
                    "string indices must be integers, not '{}'",
                    self.type_name_of(index)?
                );
                Err(self.raise_exception("TypeError", message))
            }
            protocol::StringIndex::OutOfRange => {
                Err(self.raise_exception("IndexError", "string index out of range"))
            }
        }
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

    /// `value` as a machine integer, read through `__index__` as CPython's `PyNumber_Index` reads
    /// sequence indices and slice bounds; NumPy integer scalars and 0-d integer arrays qualify.
    /// `None` means `value` is not an integer and its type defines no `__index__`.
    fn index_value(&mut self, value: &Value) -> Result<Option<i64>, String> {
        if let Some(index) = protocol::int_value(&self.state.heap, value) {
            return Ok(Some(index));
        }
        let Some(result) = self.int_by_method(value, &["__index__"])? else {
            return Ok(None);
        };
        match protocol::int_value(&self.state.heap, &result) {
            Some(index) => Ok(Some(index)),
            None => {
                Err(self
                    .raise_exception("IndexError", "cannot fit 'int' into an index-sized integer"))
            }
        }
    }

    /// A builtin sequence's subscript with a non-int index, such as a NumPy integer, replaced by
    /// the int its `__index__` returns. Other owners and indices are returned unchanged.
    fn sequence_index(&mut self, owner: &Value, index: Value) -> Result<Value, String> {
        let Some(id) = owner.object_id() else {
            return Ok(index);
        };
        let sequence = matches!(
            self.state.heap.get(id)?,
            Object::List(_)
                | Object::Tuple(_)
                | Object::Range { .. }
                | Object::String(_)
                | Object::Bytes(_)
                | Object::ByteArray(_)
        );
        if !sequence || protocol::int_value(&self.state.heap, &index).is_some() {
            return Ok(index);
        }
        Ok(self.index_value(&index)?.map_or(index, Value::Int))
    }

    /// An integer argument read through `__index__`, such as a `range()` bound. Anything else
    /// raises CPython's `TypeError`.
    pub(super) fn index_argument(&mut self, value: &Value) -> Result<i64, String> {
        if let Some(index) = self.index_value(value)? {
            return Ok(index);
        }
        let message = format!(
            "'{}' object cannot be interpreted as an integer",
            self.type_name_of(value)?
        );
        Err(self.raise_exception("TypeError", message))
    }

    /// Call `value`'s conversion method `method` (`__int__`, `__float__`, or `__index__`), as
    /// CPython's `int()` and `float()` do for values that are not builtin numbers or strings:
    /// NumPy scalars and arrays, and instances of classes. `None` means the value is a builtin
    /// number, string, or bytes, or its type does not define `method`.
    fn conversion_method(&mut self, value: &Value, method: &str) -> Result<Option<Value>, String> {
        let builtin = self.registered_kind(value).is_none()
            && (super::number::view(&self.state.heap, value).is_some()
                || protocol::string_value(&self.state.heap, value)?.is_some()
                || protocol::bytes_value(&self.state.heap, value)?.is_some());
        if builtin {
            return Ok(None);
        }
        let Some(method) = self.special_method(value, method)? else {
            return Ok(None);
        };
        self.invoke_value(method, Vec::new()).map(Some)
    }

    /// `value`'s special method `name`, bound to `value`. CPython looks special methods up on
    /// the type, so an instance attribute of the same name does not replace the class's method.
    /// Native kinds have no instance dictionary, so ordinary attribute lookup finds the same
    /// method for them.
    pub(super) fn special_method(
        &mut self,
        value: &Value,
        name: &str,
    ) -> Result<Option<Value>, String> {
        let class = match value
            .object_id()
            .map(|id| self.state.heap.get(id))
            .transpose()?
        {
            Some(Object::Instance { class, .. }) => *class,
            _ => return self.resolve_attribute(*value, name),
        };
        let Some((defining_class, descriptor)) = self.class_attribute_entry(class, name)? else {
            return Ok(None);
        };
        self.bind_descriptor(descriptor, Some(*value), class, defining_class)
            .map(Some)
    }

    /// An int from the first of `methods` that `value` defines, checking that the method
    /// returned an int as CPython does. `int()` tries `__int__` then `__index__`; `float()` falls
    /// back to `__index__` only.
    fn int_by_method(
        &mut self,
        value: &Value,
        methods: &[&'static str],
    ) -> Result<Option<Value>, String> {
        for &method in methods {
            let Some(result) = self.conversion_method(value, method)? else {
                continue;
            };
            if protocol::int_value(&self.state.heap, &result).is_none()
                && !self.is_bigint(&result)?
            {
                let message = format!(
                    "{method} returned non-int (type {})",
                    self.type_name_of(&result)?
                );
                return Err(self.raise_exception("TypeError", message));
            }
            return Ok(Some(result));
        }
        Ok(None)
    }

    /// `int(value)` for a number: integers stay exact and floats truncate toward zero, as
    /// `int.__new__` does through `__index__` and `__trunc__`.
    fn truncate_to_int(&mut self, value: &Value) -> Result<Value, String> {
        use super::number::NumberRef;
        let number = super::number::view(&self.state.heap, value);
        let float = match number {
            Some(NumberRef::Float(float)) => float,
            Some(number @ (NumberRef::Int(_) | NumberRef::BigInt(_) | NumberRef::UInt(_))) => {
                let text = number.to_bigint().expect("integer view").to_string();
                return self
                    .new_integer(&text)
                    .map_err(|error| self.record_native_error(error));
            }
            Some(NumberRef::Complex(..)) => {
                let message = format!(
                    "int() argument must be a string, a bytes-like object or a real number, not '{}'",
                    self.type_name_of(value)?
                );
                return Err(self.raise_exception("TypeError", message));
            }
            None => {
                let message = format!(
                    "int() argument must be a string, a bytes-like object or a real number, not \
                     '{}'",
                    self.type_name_of(value)?
                );
                return Err(self.raise_exception("TypeError", message));
            }
        };
        if float.is_nan() {
            return Err(self.raise_exception("ValueError", "cannot convert float NaN to integer"));
        }
        if float.is_infinite() {
            return Err(
                self.raise_exception("OverflowError", "cannot convert float infinity to integer")
            );
        }
        let text = format!("{:.0}", float.trunc());
        self.new_integer(&text)
            .map_err(|error| self.record_native_error(error))
    }

    /// The indices a slice selects with, when `value` is a slice: each bound goes through
    /// `__index__` (`None` stays open). Integers beyond the machine range clamp to it, as
    /// CPython's `_PyEval_SliceIndex` does, since no sequence is that long.
    pub(super) fn slice_bounds(
        &mut self,
        value: &Value,
    ) -> Result<Option<super::super::slice::SliceBounds>, String> {
        let Some(id) = value.object_id() else {
            return Ok(None);
        };
        let Object::Slice { start, stop, step } = self.state.heap.get(id)? else {
            return Ok(None);
        };
        let (start, stop, step) = (*start, *stop, *step);
        let bounds = (
            self.slice_index(&start)?,
            self.slice_index(&stop)?,
            self.slice_index(&step)?,
        );
        if bounds.2 == Some(0) {
            return Err(self.raise_exception("ValueError", "slice step cannot be zero"));
        }
        Ok(Some(bounds))
    }

    fn slice_index(&mut self, bound: &Value) -> Result<Option<i64>, String> {
        if bound.is_none() {
            return Ok(None);
        }
        let integer =
            if protocol::int_value(&self.state.heap, bound).is_some() || self.is_bigint(bound)? {
                *bound
            } else {
                match self.int_by_method(bound, &["__index__"])? {
                    Some(integer) => integer,
                    None => {
                        return Err(self.raise_exception(
                            "TypeError",
                            "slice indices must be integers or None or have an __index__ method",
                        ))
                    }
                }
            };
        if let Some(index) = protocol::int_value(&self.state.heap, &integer) {
            return Ok(Some(index));
        }
        let negative = matches!(
            super::number::view(&self.state.heap, &integer),
            Some(super::number::NumberRef::BigInt(value)) if num_traits::Signed::is_negative(value)
        );
        Ok(Some(if negative { -i64::MAX } else { i64::MAX }))
    }

    /// Pop one slice bound; an omitted bound is `None`.
    fn pop_slice_bound(&mut self, present: bool) -> Result<Value, String> {
        if present {
            self.pop()
        } else {
            Ok(Value::None)
        }
    }

    pub(super) fn build_slice(
        &mut self,
        has_start: bool,
        has_stop: bool,
        has_step: bool,
    ) -> Result<(), String> {
        let step = self.pop_slice_bound(has_step)?;
        let stop = self.pop_slice_bound(has_stop)?;
        let start = self.pop_slice_bound(has_start)?;
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
            return Err(self.raise_object_type_error(&owner, "does not support item assignment"));
        };
        let index = self.sequence_index(&owner, index)?;
        match self.state.heap.get(id)? {
            Object::List(values) => {
                let Some(index) = protocol::int_value(&self.state.heap, &index) else {
                    let message = format!(
                        "list indices must be integers or slices, not {}",
                        self.type_name_of(&index)?
                    );
                    return Err(self.raise_exception("TypeError", message));
                };
                let len = values.len() as i64;
                let index = if index < 0 { len + index } else { index };
                let Some(index) = usize::try_from(index)
                    .ok()
                    .filter(|index| *index < values.len())
                else {
                    return Err(
                        self.raise_exception("IndexError", "list assignment index out of range")
                    );
                };
                let Object::List(values) = self.state.heap.get_mut(id)? else {
                    unreachable!()
                };
                values[index] = value;
            }
            Object::Dict(_) | Object::DefaultDict { .. } => {
                self.dict_set_entry(id, index, value)?
            }
            // Item assignment on these would be a shellsim gap, not a CPython TypeError.
            Object::ByteArray(_) | Object::ArrayStorage(_) | Object::Array { .. } => {
                return Err("object does not support item assignment".into())
            }
            _ => {
                return Err(self.raise_object_type_error(&owner, "does not support item assignment"))
            }
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
                attributes: HashMap::new(),
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
        let named_tuple_bases = bases
            .iter()
            .filter(|base| {
                matches!(
                    base.native_value(),
                    Some(NativeValue::NativeFunction(function))
                        if super::super::stdlib::typing::is_named_tuple(function)
                )
            })
            .count();
        if named_tuple_bases > 0 && bases.len() > 1 {
            return Err(self.raise_exception(
                "TypeError",
                "can only inherit from a NamedTuple type and Generic",
            ));
        }
        let is_named_tuple = named_tuple_bases == 1;
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
        let user_bases = if is_enum || is_unittest || is_named_tuple {
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
                    } else {
                        match base.native_value() {
                            Some(NativeValue::BuiltinType(
                                BuiltinType::Object | BuiltinType::Type,
                            ))
                            | Some(NativeValue::ExceptionType(_)) => None,
                            Some(NativeValue::BuiltinType(builtin))
                                if is_subclassable_builtin(builtin) =>
                            {
                                None
                            }
                            Some(NativeValue::BuiltinType(builtin)) => Some(Err(format!(
                                "subclassing the builtin type '{}' is not supported",
                                builtin.name()
                            ))),
                            _ => Some(Err("class bases must be classes".to_string())),
                        }
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        // Direct builtin bases such as `tuple` and those inherited through user classes must agree
        // on one instance layout, as in CPython.
        let mut builtin_layouts = bases
            .iter()
            .filter_map(|base| match base.native_value() {
                Some(NativeValue::BuiltinType(builtin)) if is_subclassable_builtin(builtin) => {
                    Some(builtin)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        for base in &user_bases {
            if let Object::Class {
                layout: ClassLayout::Builtin(builtin),
                ..
            } = self.state.heap.get(*base)?
            {
                builtin_layouts.push(*builtin);
            }
        }
        builtin_layouts.dedup();
        if builtin_layouts.len() > 1 {
            return Err(
                self.raise_exception("TypeError", "multiple bases have instance lay-out conflict")
            );
        }
        let builtin_layout = builtin_layouts.first().copied();
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
        if exception_base.is_some()
            && (builtin_layout.is_some() || has_type_base || is_enum || is_unittest)
        {
            return Err("exception classes cannot use another instance layout".into());
        }
        if has_object_base && bases.len() != 1 {
            return Err("object cannot be combined with another direct base in this slice".into());
        }
        if has_type_base && bases.len() != 1 {
            return Err("type cannot be combined with another direct base in this slice".into());
        }
        let inherited_type_layouts = user_bases
            .iter()
            .filter_map(|base| match self.state.heap.get(*base) {
                Ok(Object::Class { layout, .. }) => Some(*layout == ClassLayout::Type),
                _ => None,
            })
            .filter(|is_type| *is_type)
            .count();
        if inherited_type_layouts > 1
            || ((has_type_base || inherited_type_layouts == 1) && builtin_layout.is_some())
        {
            return Err("multiple bases have incompatible instance layouts".into());
        }
        let layout = if has_type_base || inherited_type_layouts == 1 {
            ClassLayout::Type
        } else if let Some(builtin) = builtin_layout {
            ClassLayout::Builtin(builtin)
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
        if is_named_tuple {
            let class = self.named_tuple_class(&name, fields, &bindings, attributes)?;
            self.stack.push(class);
            return Ok(());
        }
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

    /// `class Name(typing.NamedTuple)`: `collections._namedtuple_from_class` builds the class
    /// from the annotated fields and the class body's namespace, both in definition order.
    fn named_tuple_class(
        &mut self,
        name: &str,
        fields: &[ClassField],
        bindings: &[String],
        mut attributes: HashMap<String, Value>,
    ) -> Result<Value, String> {
        let module = match attributes.get("__module__") {
            Some(module) => *module,
            None => self.current_module_name()?.unwrap_or(Value::None),
        };
        // A name annotated twice keeps its first position, as in `__annotations__`.
        let mut names = Vec::<&str>::with_capacity(fields.len());
        for field in fields {
            if !names.contains(&field.name.as_str()) {
                names.push(&field.name);
            }
        }
        let mut field_names = Vec::with_capacity(names.len());
        for name in names {
            field_names.push(self.allocate_string(name.to_string())?);
        }
        // Names bound by the class body keep their order; any others follow sorted, so the
        // namespace never depends on hash order.
        let mut entries = Vec::with_capacity(attributes.len());
        for binding in bindings {
            if let Some(value) = attributes.remove(binding) {
                entries.push((self.allocate_string(binding.clone())?, value));
            }
        }
        let mut rest = attributes.into_iter().collect::<Vec<_>>();
        rest.sort_by(|left, right| left.0.cmp(&right.0));
        for (key, value) in rest {
            entries.push((self.allocate_string(key)?, value));
        }
        let class_name = self.allocate_string(name.to_string())?;
        let field_names = self.allocate_object(Object::Tuple(field_names))?;
        let namespace = self.allocate_object(Object::Dict(entries.into()))?;
        let collections = PyRuntime::import_module(self, "collections")
            .map_err(|error| self.record_native_error(error))?;
        let build = PyRuntime::get_attribute(self, collections, "_namedtuple_from_class")
            .map_err(|error| self.record_native_error(error))?
            .ok_or("collections._namedtuple_from_class is missing")?;
        self.invoke_value(build, vec![class_name, field_names, namespace, module])
    }

    /// Allocate and finish a class after metaclass policy has selected its layout and C3 MRO.
    /// Both ordinary class statements and `type.__new__` use this path.
    /// `__bases__`, `__mro__` and `mro` of a user class, which CPython's `type` provides as
    /// data descriptors and a method ahead of the class's own attributes.
    fn class_metadata(&mut self, class: ObjectId, name: &str) -> Result<Option<Value>, String> {
        let items = match name {
            "__bases__" => {
                let Object::Class { bases, .. } = self.state.heap.get(class)? else {
                    return Err("class metadata requested for a non-class".into());
                };
                if bases.is_empty() {
                    vec![Value::Native(NativeValue::BuiltinType(BuiltinType::Object))]
                } else {
                    bases.clone()
                }
            }
            "__mro__" => self.class_mro(class)?,
            "mro" => {
                let descriptor = self
                    .state
                    .types
                    .attribute(BuiltinType::Type.id(), "mro")?
                    .ok_or("type.mro is not installed")?;
                return self
                    .allocate_object(Object::DescriptorBoundMethod {
                        receiver: Value::Object(class),
                        descriptor,
                        owner: None,
                    })
                    .map(Some);
            }
            _ => return Ok(None),
        };
        self.allocate_object(Object::Tuple(items)).map(Some)
    }

    /// A user class's method resolution order: the class and its C3-linearized user ancestors,
    /// then the native types their bases derive from, ending with `object`.
    fn class_mro(&self, class: ObjectId) -> Result<Vec<Value>, String> {
        let Object::Class { mro, .. } = self.state.heap.get(class)? else {
            return Err("class metadata requested for a non-class".into());
        };
        let classes = std::iter::once(class)
            .chain(mro.iter().copied())
            .collect::<Vec<_>>();
        let mut order = classes
            .iter()
            .copied()
            .map(Value::Object)
            .collect::<Vec<_>>();
        for ancestor in classes {
            let Object::Class { bases, .. } = self.state.heap.get(ancestor)? else {
                return Err("class MRO contains a non-class object".into());
            };
            for base in bases.iter().filter(|base| base.object_id().is_none()) {
                for native in self.native_mro(*base)? {
                    if !order.contains(&native) {
                        order.push(native);
                    }
                }
            }
        }
        let object = Value::Native(NativeValue::BuiltinType(BuiltinType::Object));
        order.retain(|value| *value != object);
        order.push(object);
        Ok(order)
    }

    /// The MRO of a native type, starting with the type itself.
    fn native_mro(&self, native: Value) -> Result<Vec<Value>, String> {
        let registered = match native.native_value() {
            Some(NativeValue::BuiltinType(builtin)) => Some(builtin.id()),
            Some(NativeValue::ValueKind(kind)) => self.state.types.value_kind_type_id(kind),
            Some(NativeValue::ExceptionType(ExceptionType(name))) => {
                let mut order = Vec::new();
                let mut current = Some(name);
                while let Some(name) = current {
                    order.push(Value::Native(NativeValue::ExceptionType(ExceptionType(
                        name,
                    ))));
                    current = super::super::exception_types::exception_type(name)
                        .and_then(|definition| definition.parent);
                }
                order.push(Value::Native(NativeValue::BuiltinType(BuiltinType::Object)));
                return Ok(order);
            }
            _ => None,
        };
        let Some(type_id) = registered else {
            return Ok(vec![
                native,
                Value::Native(NativeValue::BuiltinType(BuiltinType::Object)),
            ]);
        };
        let mut order = vec![native];
        for ancestor in &self.state.types.get(type_id)?.mro {
            let value = self.state.types.value(*ancestor)?;
            if !order.contains(&value) {
                order.push(value);
            }
        }
        Ok(order)
    }

    /// Fixed namespace and type metadata of builtin types, native value kinds and exception
    /// classes, plus the defining module of builtin and native functions.
    fn native_type_metadata(&mut self, owner: Value, name: &str) -> Result<Option<Value>, String> {
        let Some(native) = owner.native_value() else {
            return Ok(None);
        };
        let qualified_module = |qualified: &'static str| {
            qualified
                .rsplit_once('.')
                .map_or("builtins", |(module, _)| module)
        };
        let is_type = matches!(
            native,
            NativeValue::BuiltinType(_) | NativeValue::ValueKind(_) | NativeValue::ExceptionType(_)
        );
        match name {
            "__dict__" if is_type => {
                let type_id = match native {
                    NativeValue::BuiltinType(builtin) => builtin.id(),
                    NativeValue::ValueKind(kind) => self
                        .state
                        .types
                        .value_kind_type_id(kind)
                        .ok_or("value kind is not registered")?,
                    NativeValue::ExceptionType(_) => BuiltinType::Exception.id(),
                    _ => unreachable!("checked by is_type"),
                };
                self.allocate_object(Object::MappingProxy(ProxyTarget::RegisteredType(type_id)))
                    .map(Some)
            }
            "__module__" => {
                let module = match native {
                    NativeValue::BuiltinType(builtin) => qualified_module(builtin.name()),
                    NativeValue::ValueKind(kind) => qualified_module(kind.name),
                    NativeValue::ExceptionType(ExceptionType(name)) => {
                        super::super::exception_types::exception_type(name)
                            .map_or("builtins", |definition| definition.module)
                    }
                    NativeValue::Function(_) => "builtins",
                    NativeValue::NativeFunction(function) => function.module,
                    _ => return Ok(None),
                };
                self.allocate_string(module.to_string()).map(Some)
            }
            "__mro__" if is_type => {
                let order = self.native_mro(owner)?;
                self.allocate_object(Object::Tuple(order)).map(Some)
            }
            "__bases__" if is_type => {
                let bases = match native {
                    NativeValue::ExceptionType(ExceptionType(name)) => {
                        let parent = super::super::exception_types::exception_type(name)
                            .and_then(|definition| definition.parent);
                        vec![match parent {
                            Some(parent) => {
                                Value::Native(NativeValue::ExceptionType(ExceptionType(parent)))
                            }
                            None => Value::Native(NativeValue::BuiltinType(BuiltinType::Object)),
                        }]
                    }
                    _ => {
                        let type_id = match native {
                            NativeValue::BuiltinType(builtin) => builtin.id(),
                            NativeValue::ValueKind(kind) => self
                                .state
                                .types
                                .value_kind_type_id(kind)
                                .ok_or("value kind is not registered")?,
                            _ => unreachable!("checked by is_type"),
                        };
                        self.state
                            .types
                            .get(type_id)?
                            .bases
                            .iter()
                            .map(|base| self.state.types.value(*base))
                            .collect::<Result<Vec<_>, _>>()?
                    }
                };
                self.allocate_object(Object::Tuple(bases)).map(Some)
            }
            _ => Ok(None),
        }
    }

    /// The `__name__` global of the module whose code is running.
    fn current_module_name(&mut self) -> Result<Option<Value>, String> {
        self.module_name_of(self.local_scopes.last().copied())
    }

    /// The `__name__` global of the module that owns `scope`; `None` is the main script's
    /// global namespace.
    pub(super) fn module_name_of(
        &mut self,
        scope: Option<super::ScopeId>,
    ) -> Result<Option<Value>, String> {
        if let Some(scope) = scope {
            let root = self.state.heap.scope_root(scope)?;
            if !self.state.heap.scope_uses_repl_globals(root)? {
                return Ok(self.state.heap.scope_get(root, "__name__").copied());
            }
        }
        let symbol = self
            .state
            .heap
            .intern_symbol("__name__", &mut self.interp.resources)?;
        Ok(self.state.globals.get(symbol))
    }

    pub(super) fn allocate_class(&mut self, definition: ClassDefinition) -> Result<Value, String> {
        let ClassDefinition {
            name,
            bases,
            mro,
            metaclass,
            layout,
            exception_base,
            mut attributes,
            dataclass_fields,
            enum_members,
        } = definition;
        // As in CPython, a class records the defining module's `__name__` unless its namespace
        // already sets `__module__`.
        if !attributes.contains_key("__module__") {
            if let Some(module) = self.current_module_name()? {
                attributes.insert("__module__".into(), module);
            }
        }
        if attributes.contains_key("__eq__") && !attributes.contains_key("__hash__") {
            attributes.insert("__hash__".into(), Value::None);
        }
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
            ClassLayout::Builtin(builtin) => Some(builtin.id()),
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
            bases,
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

    /// Search each type's own namespace in MRO order. User namespaces remain on their heap
    /// class objects, while builtin namespaces remain in the registry.
    pub(super) fn type_lookup(
        &mut self,
        type_id: TypeId,
        name: &str,
    ) -> Result<Option<(TypeId, Value)>, String> {
        let mro_len = self.state.types.get(type_id)?.mro.len();
        for index in 0..=mro_len {
            let ancestor = if index == 0 {
                type_id
            } else {
                self.state.types.get(type_id)?.mro[index - 1]
            };
            self.charge_cpu(1)?;
            if let Some(value) = self.type_namespace_attribute(ancestor, name)? {
                return Ok(Some((ancestor, value)));
            }
        }
        Ok(None)
    }

    fn type_namespace_attribute(
        &self,
        type_id: TypeId,
        name: &str,
    ) -> Result<Option<Value>, String> {
        let ty = self.state.types.get(type_id)?;
        match self.state.types.value(type_id)?.object_id() {
            Some(id) => match self.state.heap.get(id)? {
                Object::Class { attributes, .. } => Ok(attributes.get(name).copied()),
                _ => Ok(ty.attributes.get(name).copied()),
            },
            None => Ok(ty.attributes.get(name).copied()),
        }
    }

    /// Bind a descriptor found by `type_lookup` to an instance or leave it unbound for class
    /// access. Native descriptors have no heap class object, so they are handled here.
    fn bind_type_attribute(
        &mut self,
        descriptor: Value,
        receiver: Option<Value>,
        accessed_type: TypeId,
        defining_type: TypeId,
    ) -> Result<Option<Value>, String> {
        match descriptor.native_value() {
            Some(NativeValue::NativeGetter(getter)) => {
                return match receiver {
                    Some(receiver) => self.call_native_getter(getter, receiver),
                    None => Ok(Some(descriptor)),
                };
            }
            Some(NativeValue::NativeClassMethod(method)) => {
                let class = self.state.types.value(accessed_type)?;
                return self.bind_native_class_method(class, method).map(Some);
            }
            Some(NativeValue::NativeMethod(method)) if method.name == "__new__" => {
                return Ok(Some(descriptor));
            }
            Some(NativeValue::NativeMethod(_)) => {
                return match receiver {
                    Some(receiver) => self
                        .allocate_object(Object::DescriptorBoundMethod {
                            receiver,
                            descriptor,
                            owner: None,
                        })
                        .map(Some),
                    None => Ok(Some(descriptor)),
                };
            }
            Some(NativeValue::SlotWrapper { .. }) => {
                return match receiver {
                    Some(receiver) => self
                        .allocate_object(Object::DescriptorBoundMethod {
                            receiver,
                            descriptor,
                            owner: None,
                        })
                        .map(Some),
                    None => Ok(Some(descriptor)),
                };
            }
            _ => {}
        }
        let Some(defining_class) = self.state.types.value(defining_type)?.object_id() else {
            return Ok(Some(descriptor));
        };
        let accessed_class = self
            .state
            .types
            .value(accessed_type)?
            .object_id()
            .ok_or("user descriptor has no class receiver")?;
        self.bind_descriptor(descriptor, receiver, accessed_class, defining_class)
            .map(Some)
    }

    fn call_native_getter(
        &mut self,
        getter: &'static super::super::native::GetterDef,
        receiver: Value,
    ) -> Result<Option<Value>, String> {
        match (getter.get)(self, receiver) {
            Ok(value) => Ok(Some(value)),
            Err(error) => Err(self.record_native_error(error)),
        }
    }

    fn is_data_descriptor(&mut self, value: &Value) -> Result<bool, String> {
        if matches!(value.native_value(), Some(NativeValue::NativeGetter(_))) {
            return Ok(true);
        }
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

    /// Bind a native class method to `class`, which the method receives in place of an instance.
    fn bind_native_class_method(
        &mut self,
        class: Value,
        method: &'static super::super::native::MethodDef,
    ) -> Result<Value, String> {
        self.allocate_object(Object::DescriptorBoundMethod {
            receiver: class,
            descriptor: Value::Native(NativeValue::NativeMethod(method)),
            owner: None,
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
            Some(NativeValue::NativeMethod(_) | NativeValue::SlotWrapper { .. })
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

    pub(super) fn invoke_value(
        &mut self,
        callable: Value,
        arguments: Vec<Value>,
    ) -> Result<Value, String> {
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
            .map(|(name, _)| Some(name.clone()))
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
        // A native slot implements a builtin type's behavior, which an instance of a builtin
        // subclass, as receiver or operand, takes part in through the value it holds.
        let slot_descriptor = match slot_value {
            SlotValue::NativeBinary(call) => {
                let [argument] = arguments.as_slice() else {
                    return Err(
                        "binary protocol slot received the wrong number of arguments".into(),
                    );
                };
                let (receiver, argument) =
                    (self.builtin_view(*receiver)?, self.builtin_view(*argument)?);
                return call(self, receiver, argument)
                    .map_err(|error| self.record_native_error(error));
            }
            SlotValue::NativeTernary(call) => {
                let [first, second] = arguments.as_slice() else {
                    return Err(
                        "ternary protocol slot received the wrong number of arguments".into(),
                    );
                };
                let receiver = self.builtin_view(*receiver)?;
                let (first, second) = (self.builtin_view(*first)?, self.builtin_view(*second)?);
                return call(self, receiver, first, second)
                    .map_err(|error| self.record_native_error(error));
            }
            SlotValue::NativeUnary(call) => {
                if !arguments.is_empty() {
                    return Err("unary protocol slot received arguments".into());
                }
                let receiver = self.builtin_view(*receiver)?;
                return call(self, receiver).map_err(|error| self.record_native_error(error));
            }
            SlotValue::Descriptor(descriptor) => descriptor,
        };
        let Some(id) = receiver.object_id() else {
            return Ok(None);
        };
        let class = match self.state.heap.get(id)? {
            Object::Instance { class, .. } => *class,
            Object::Class { metaclass, .. } => match metaclass.object_id() {
                Some(class) => class,
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        let (defining_class, _) = self
            .class_attribute_entry(class, method_name)?
            .ok_or("cached type slot has no descriptor")?;
        let callable =
            self.bind_descriptor(slot_descriptor, Some(*receiver), class, defining_class)?;
        self.invoke_value(callable, arguments).map(Some)
    }

    /// Execute the defining type's native slot for an explicit dunder call. The receiver check
    /// and local-slot lookup keep `int.__add__(x, y)` independent of overrides on `type(x)`.
    pub(super) fn call_slot_wrapper(
        &mut self,
        owner: TypeId,
        slot: Slot,
        receiver: Value,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<Value, String> {
        let (_, name, arity) = super::super::object_model::SLOT_DEFS[slot as usize];
        if !keyword_arguments.is_empty() || arguments.len() != usize::from(arity) {
            return Err(
                self.raise_exception("TypeError", format!("{name}() received invalid arguments"))
            );
        }
        let receiver_type = self.type_id(&receiver)?;
        if !self.state.types.is_subclass(receiver_type, owner)? {
            let owner_name = self.state.types.get(owner)?.name.clone();
            return Err(self.raise_exception(
                "TypeError",
                format!("descriptor {name} requires a '{owner_name}' object"),
            ));
        }
        let implementation = self
            .state
            .types
            .local_slot(owner, slot)?
            .ok_or("slot wrapper has no local implementation")?;
        let receiver = self.builtin_view(receiver)?;
        let result = match implementation {
            SlotValue::NativeUnary(call) => call(self, receiver),
            SlotValue::NativeBinary(call) => {
                let argument = self.builtin_view(arguments[0])?;
                call(self, receiver, argument)
            }
            SlotValue::NativeTernary(call) => {
                let first = self.builtin_view(arguments[0])?;
                let second = self.builtin_view(arguments[1])?;
                call(self, receiver, first, second)
            }
            SlotValue::Descriptor(_) => return Err("slot wrapper is not native".into()),
        };
        result
            .map(|result| result.unwrap_or(Value::Native(NativeValue::NotImplemented)))
            .map_err(|error| self.record_native_error(error))
    }

    /// Invoke a binary-operator or rich-comparison slot. A method that returns `NotImplemented`
    /// declines the operation, so the result is `None` exactly as if the slot were absent and the
    /// caller goes on to the reflected method or the default behavior, as CPython does.
    pub(super) fn invoke_operator_slot(
        &mut self,
        receiver: &Value,
        slot: Slot,
        method_name: &str,
        arguments: Vec<Value>,
    ) -> Result<Option<Value>, String> {
        Ok(self
            .invoke_slot(receiver, slot, method_name, arguments)?
            .filter(|value| value.native_value() != Some(NativeValue::NotImplemented)))
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
        self.repr_nested(value, &mut BTreeSet::new())
    }

    /// `repr()` that renders container items through their own `__repr__` and protocol slots,
    /// as CPython does. `active` holds the containers being rendered, so a container that
    /// contains itself prints as `[...]`.
    fn repr_nested(
        &mut self,
        value: &Value,
        active: &mut BTreeSet<ObjectId>,
    ) -> Result<String, String> {
        if let Some(result) = self.invoke_slot(value, Slot::Repr, "__repr__", Vec::new())? {
            return protocol::string_value(&self.state.heap, &result)?
                .ok_or_else(|| "__repr__ should return str".into());
        }
        if let Some(id) = value.object_id() {
            match *self.state.heap.get(id)? {
                Object::NamespaceDict(target) => {
                    return self.repr_namespace_dict(id, target, active);
                }
                Object::DictView { kind, mapping } => {
                    return self.repr_dict_view(id, kind, mapping, active);
                }
                Object::MappingProxy(target) => {
                    return self.repr_mapping_proxy(id, target, active);
                }
                _ => {}
            }
        }
        let Some((id, container)) = self.container_items(value)? else {
            return protocol::repr(&self.state.heap, value);
        };
        if !active.insert(id) {
            return Ok(container.placeholder().into());
        }
        self.charge_cpu(u64::try_from(container.len()).unwrap_or(u64::MAX))?;
        let rendered = match container {
            ContainerItems::List(items) => format!("[{}]", self.repr_items(&items, active)?),
            ContainerItems::Tuple(items) => match items.as_slice() {
                [only] => format!("({},)", self.repr_nested(only, active)?),
                _ => format!("({})", self.repr_items(&items, active)?),
            },
            ContainerItems::Set(items) if items.is_empty() => "set()".into(),
            ContainerItems::Set(items) => format!("{{{}}}", self.repr_items(&items, active)?),
            ContainerItems::FrozenSet(items) if items.is_empty() => "frozenset()".into(),
            ContainerItems::FrozenSet(items) => {
                format!("frozenset({{{}}})", self.repr_items(&items, active)?)
            }
            ContainerItems::Dict(entries) => self.repr_entries(&entries, active)?,
        };
        active.remove(&id);
        Ok(rendered)
    }

    /// Dict-literal text for `entries`: `{key: value, ...}`.
    fn repr_entries(
        &mut self,
        entries: &[(Value, Value)],
        active: &mut BTreeSet<ObjectId>,
    ) -> Result<String, String> {
        let mut parts = Vec::with_capacity(entries.len());
        for (key, item) in entries {
            let key = self.repr_nested(key, active)?;
            let item = self.repr_nested(item, active)?;
            parts.push(format!("{key}: {item}"));
        }
        Ok(format!("{{{}}}", parts.join(", ")))
    }

    /// `dict_keys([...])`, `dict_values([...])` or `dict_items([...])` for a view of the
    /// mapping's current entries. A view reached again while it renders prints `...`, as in
    /// CPython.
    fn repr_dict_view(
        &mut self,
        id: ObjectId,
        kind: DictViewKind,
        mapping: ObjectId,
        active: &mut BTreeSet<ObjectId>,
    ) -> Result<String, String> {
        if !active.insert(id) {
            return Ok("...".into());
        }
        let entries = self
            .mapping_items(Value::Object(mapping))?
            .ok_or("a dict view needs a mapping")?;
        self.charge_cpu(u64::try_from(entries.len()).unwrap_or(u64::MAX))?;
        let mut parts = Vec::with_capacity(entries.len());
        for (key, item) in &entries {
            parts.push(match kind {
                DictViewKind::Keys => self.repr_nested(key, active)?,
                DictViewKind::Values => self.repr_nested(item, active)?,
                DictViewKind::Items => format!(
                    "({}, {})",
                    self.repr_nested(key, active)?,
                    self.repr_nested(item, active)?
                ),
            });
        }
        active.remove(&id);
        let name = match kind {
            DictViewKind::Keys => "dict_keys",
            DictViewKind::Values => "dict_values",
            DictViewKind::Items => "dict_items",
        };
        Ok(format!("{name}([{}])", parts.join(", ")))
    }

    /// `mappingproxy({...})` over a class's or native module's current attributes.
    fn repr_mapping_proxy(
        &mut self,
        id: ObjectId,
        target: ProxyTarget,
        active: &mut BTreeSet<ObjectId>,
    ) -> Result<String, String> {
        if !active.insert(id) {
            return Ok("...".into());
        }
        let entries = self.proxy_items(target)?;
        let rendered = self.repr_entries(&entries, active)?;
        active.remove(&id);
        Ok(format!("mappingproxy({rendered})"))
    }

    /// `repr` of a namespace view: rendered like a dict literal, sharing `active` with
    /// `repr_nested` so a namespace that holds its own view (`g = globals()` at module level)
    /// prints `{...}` for the cycle instead of recursing without limit.
    fn repr_namespace_dict(
        &mut self,
        id: ObjectId,
        target: NamespaceTarget,
        active: &mut BTreeSet<ObjectId>,
    ) -> Result<String, String> {
        if !active.insert(id) {
            return Ok("{...}".into());
        }
        let entries = self.namespace_entries(target)?;
        self.charge_cpu(u64::try_from(entries.len()).unwrap_or(u64::MAX))?;
        let mut parts = Vec::with_capacity(entries.len());
        for (name, value) in entries {
            let value = self.repr_nested(&value, active)?;
            parts.push(format!("{}: {value}", protocol::quote_string(&name)));
        }
        active.remove(&id);
        Ok(format!("{{{}}}", parts.join(", ")))
    }

    fn repr_items(
        &mut self,
        items: &[Value],
        active: &mut BTreeSet<ObjectId>,
    ) -> Result<String, String> {
        let mut parts = Vec::with_capacity(items.len());
        for item in items {
            parts.push(self.repr_nested(item, active)?);
        }
        Ok(parts.join(", "))
    }

    /// A snapshot of a builtin container's items, for rendering them through the VM.
    fn container_items(&self, value: &Value) -> Result<Option<(ObjectId, ContainerItems)>, String> {
        let Some(id) = value.object_id() else {
            return Ok(None);
        };
        let items = match self.state.heap.get(id)? {
            Object::List(items) => ContainerItems::List(items.clone()),
            Object::Tuple(items) => ContainerItems::Tuple(items.clone()),
            Object::Set(items) => ContainerItems::Set(items.clone()),
            Object::FrozenSet(items) => ContainerItems::FrozenSet(items.clone()),
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                ContainerItems::Dict(entries.iter().copied().collect())
            }
            _ => return Ok(None),
        };
        Ok(Some((id, items)))
    }

    /// Whether `value` is a namespace view, dict view or mapping proxy, which `repr_nested`
    /// renders through the VM because their entries live outside the object.
    fn is_mapping_view(&self, value: &Value) -> Result<bool, String> {
        let Some(id) = value.object_id() else {
            return Ok(false);
        };
        Ok(matches!(
            self.state.heap.get(id)?,
            Object::NamespaceDict(_) | Object::DictView { .. } | Object::MappingProxy(_)
        ))
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
            || self.container_items(value)?.is_some()
            || self.is_mapping_view(value)?
        {
            return self.repr_value(value);
        }
        protocol::display(&self.state.heap, value)
    }

    /// Order two values by Python's rich comparisons, including user `__eq__` and `__lt__`.
    pub(super) fn compare_values(
        &mut self,
        left: &Value,
        right: &Value,
    ) -> Result<protocol::Comparison, String> {
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
                    let comparison = self.compare_values(left, right)?;
                    if comparison != protocol::Comparison::Ordered(Ordering::Equal) {
                        return Ok(comparison);
                    }
                }
                return Ok(protocol::Comparison::Ordered(left.len().cmp(&right.len())));
            }
        }
        if let Some(equal) = self.invoke_operator_slot(left, Slot::Equal, "__eq__", vec![*right])? {
            if self.truth_value(&equal)? {
                return Ok(protocol::Comparison::Ordered(Ordering::Equal));
            }
        }
        if let Some(less) =
            self.invoke_operator_slot(left, Slot::LessThan, "__lt__", vec![*right])?
        {
            if self.truth_value(&less)? {
                return Ok(protocol::Comparison::Ordered(Ordering::Less));
            }
        }
        if let Some(less) =
            self.invoke_operator_slot(right, Slot::LessThan, "__lt__", vec![*left])?
        {
            if self.truth_value(&less)? {
                return Ok(protocol::Comparison::Ordered(Ordering::Greater));
            }
        }
        let (builtin_left, builtin_right) = (self.builtin_view(*left)?, self.builtin_view(*right)?);
        if builtin_left != *left || builtin_right != *right {
            return self.compare_values(&builtin_left, &builtin_right);
        }
        protocol::compare(&self.state.heap, left, right)
    }

    /// Order two values for native sorting, heap and bisection helpers with the `<` operator
    /// alone, as CPython's do: `left < right` is `Less`, `right < left` is `Greater`, and
    /// anything else, such as a NaN, is `Equal`, so the earlier value stays in place.
    pub(super) fn sort_order(&mut self, left: &Value, right: &Value) -> Result<Ordering, String> {
        if self.compare_truth(ComparisonOperator::Less, left, right)? {
            Ok(Ordering::Less)
        } else if self.compare_truth(ComparisonOperator::Less, right, left)? {
            Ok(Ordering::Greater)
        } else {
            Ok(Ordering::Equal)
        }
    }

    /// Raise CPython's `TypeError` for an ordering comparison between unrelated types.
    pub(super) fn raise_unorderable(
        &mut self,
        symbol: &str,
        left: &Value,
        right: &Value,
    ) -> String {
        let message = match (self.type_name_of(left), self.type_name_of(right)) {
            (Ok(left), Ok(right)) => {
                format!("'{symbol}' not supported between instances of '{left}' and '{right}'")
            }
            (Err(error), _) | (_, Err(error)) => return error,
        };
        self.raise_exception("TypeError", message)
    }

    fn super_attribute(
        &mut self,
        start_class: super::super::heap::ObjectId,
        receiver: &Value,
        name: &str,
    ) -> Result<(TypeId, Value, TypeId), String> {
        let accessed_class = if let Some(id) = receiver.object_id() {
            match self.state.heap.get(id)? {
                Object::Instance { class, .. } => *class,
                Object::Class { .. } => id,
                _ => return Err("super() receiver is not an instance or class".into()),
            }
        } else {
            return Err("super() receiver is not an instance or class".into());
        };
        let accessed_type = self
            .class_type_id(&Value::Object(accessed_class))?
            .ok_or("super() receiver has an invalid class")?;
        let start_type = self
            .class_type_id(&Value::Object(start_class))?
            .ok_or("super() start has an invalid class")?;
        let mro = &self.state.types.get(accessed_type)?.mro;
        let start = if accessed_type == start_type {
            0
        } else {
            mro.iter()
                .position(|class| *class == start_type)
                .map(|position| position + 1)
                .ok_or("super(type, obj): obj is not an instance or subtype of type")?
        };
        let mro_len = mro.len();
        for index in start..mro_len {
            let ancestor = self.state.types.get(accessed_type)?.mro[index];
            self.charge_cpu(1)?;
            if let Some(descriptor) = self.type_namespace_attribute(ancestor, name)? {
                return Ok((ancestor, descriptor, accessed_type));
            }
        }
        Err(format!("super object has no attribute {name:?}"))
    }

    /// Invoke a canonical builtin type object.
    ///
    /// Construction is centralized here so type identity, `type()`, and calling a type do not
    /// depend on the unrelated builtin-function dispatch table. Collection construction remains
    /// metered through the normal iterator and allocation paths.
    /// `d[key] = value` for the dict `id`: replace the value of an equal key, or append the pair.
    fn dict_set_entry(&mut self, id: ObjectId, key: Value, value: Value) -> Result<(), String> {
        if let Some(position) = self.find_mapping_entry(id, &key)? {
            let entries = match self.state.heap.get_mut(id)? {
                Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
                _ => unreachable!("dict kind was checked during lookup"),
            };
            entries.set_value(position, value);
            return Ok(());
        }
        self.state.heap.reserve_object_growth(
            id,
            MODELED_MAPPING_ENTRY_BYTES,
            &mut self.interp.resources,
        )?;
        let entries = match self.state.heap.get_mut(id)? {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
            _ => unreachable!("dict kind was checked during lookup"),
        };
        entries.push((key, value));
        Ok(())
    }

    /// `dict(source=(), /, **keywords)`: the entries of a mapping (a dict, or any object with
    /// `keys()` and `__getitem__`) or the pairs of an iterable, followed by the keywords.
    fn construct_dict(
        &mut self,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<Value, String> {
        if arguments.len() > 1 {
            let message = format!("dict expected at most 1 argument, got {}", arguments.len());
            return Err(self.raise_exception("TypeError", message));
        }
        let dict = self.allocate_object(Object::Dict(Vec::new().into()))?;
        let id = dict.object_id().expect("dicts are arena objects");
        if let Some(source) = arguments.first() {
            for (key, value) in self.dict_source_entries(source)? {
                self.charge_cpu(1)?;
                self.dict_set_entry(id, key, value)?;
            }
        }
        for (name, value) in keyword_arguments {
            let key = self.allocate_string(name)?;
            self.dict_set_entry(id, key, value)?;
        }
        Ok(dict)
    }

    fn dict_source_entries(&mut self, source: &Value) -> Result<Vec<(Value, Value)>, String> {
        // A dict subclass contributes the entries it holds, as CPython's dict merge does.
        if let Some(id) = self.builtin_view(*source)?.object_id() {
            if let Object::Dict(entries) | Object::DefaultDict { entries, .. } =
                self.state.heap.get(id)?
            {
                return Ok(entries.to_vec());
            }
        }
        if let Some(keys) = self.resolve_optional_attribute(*source, "keys")? {
            let keys = <Self as PyRuntime>::call_value(
                self,
                keys,
                super::CallArgs::new(Vec::new(), Vec::new()),
            )
            .map_err(|error| self.record_native_error(error))?;
            let mut entries = Vec::new();
            for key in self.iterable_values(&keys)? {
                let Some(value) =
                    self.invoke_slot(source, Slot::GetItem, "__getitem__", vec![key])?
                else {
                    return Err(self.raise_object_type_error(source, "is not subscriptable"));
                };
                entries.push((key, value));
            }
            return Ok(entries);
        }
        let mut entries = Vec::new();
        for (index, item) in self.iterable_values(source)?.into_iter().enumerate() {
            let pair = match self.iterable_values(&item) {
                Ok(pair) => pair,
                // CPython 3.14 replaces the "'int' object is not iterable" of a non-iterable
                // element with a message that omits the type name.
                Err(message)
                    if message.ends_with("object is not iterable")
                        && self
                            .pending_exception
                            .as_ref()
                            .is_some_and(|exception| exception.kind == "TypeError") =>
                {
                    self.pending_exception = None;
                    return Err(self.raise_exception("TypeError", "object is not iterable"));
                }
                Err(error) => return Err(error),
            };
            let [key, value] = pair.as_slice() else {
                let message = format!(
                    "dictionary update sequence element #{index} has length {}; 2 is required",
                    pair.len()
                );
                return Err(self.raise_exception("ValueError", message));
            };
            entries.push((*key, *value));
        }
        Ok(entries)
    }

    pub(super) fn call_builtin_type(
        &mut self,
        builtin_type: BuiltinType,
        arguments: Vec<Value>,
        keyword_arguments: Vec<(String, Value)>,
    ) -> Result<CallResult, String> {
        if builtin_type == BuiltinType::Dict {
            return self
                .construct_dict(arguments, keyword_arguments)
                .map(CallResult::Value);
        }
        if builtin_type == BuiltinType::Complex {
            let arguments = super::CallArgs::new(arguments, keyword_arguments);
            let value = super::super::complex::construct(self, arguments)
                .map_err(|error| self.record_native_error(error))?;
            return Ok(CallResult::Value(value));
        }
        if matches!(builtin_type, BuiltinType::Int | BuiltinType::Float)
            && arguments
                .first()
                .is_some_and(|value| super::number::is_complex(&self.state.heap, value))
        {
            let error = PyError::type_error(if builtin_type == BuiltinType::Int {
                "int() argument must be a string, a bytes-like object or a real number, not 'complex'"
            } else {
                "float() argument must be a string or a real number, not 'complex'"
            });
            return Err(self.record_native_error(error));
        }
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
                self.allocate_object(Object::Bare)?
            }
            BuiltinType::Slice => {
                if arguments.is_empty() || arguments.len() > 3 {
                    return Err(self.raise_exception(
                        "TypeError",
                        format!(
                            "slice expected at least 1 argument, got {}",
                            arguments.len()
                        ),
                    ));
                }
                let mut bounds = arguments.clone();
                if bounds.len() == 1 {
                    bounds.insert(0, Value::None);
                }
                bounds.resize(3, Value::None);
                self.allocate_object(Object::Slice {
                    start: bounds[0],
                    stop: bounds[1],
                    step: bounds[2],
                })?
            }
            BuiltinType::Range => {
                let bound = match arguments.len() {
                    0 => Some("least 1 argument"),
                    1..=3 => None,
                    _ => Some("most 3 arguments"),
                };
                if let Some(bound) = bound {
                    let message = format!("range expected at {bound}, got {}", arguments.len());
                    return Err(self.raise_exception("TypeError", message));
                }
                let mut integers = Vec::with_capacity(arguments.len());
                for value in &arguments {
                    integers.push(self.index_argument(value)?);
                }
                let (start, stop, step) = match integers.as_slice() {
                    [stop] => (0, *stop, 1),
                    [start, stop] => (*start, *stop, 1),
                    [start, stop, step] => (*start, *stop, *step),
                    _ => unreachable!(),
                };
                if step == 0 {
                    return Err(
                        self.raise_exception("ValueError", "range() arg 3 must not be zero")
                    );
                }
                self.allocate_object(Object::Range { start, stop, step })?
            }
            BuiltinType::None => {
                expect_arity(&arguments, 0, 0)?;
                Value::None
            }
            BuiltinType::Ellipsis => {
                expect_arity(&arguments, 0, 0)?;
                Value::Native(NativeValue::Ellipsis)
            }
            BuiltinType::NotImplemented => {
                expect_arity(&arguments, 0, 0)?;
                Value::Native(NativeValue::NotImplemented)
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
                        Some(value) => {
                            match self.int_by_method(value, &["__int__", "__index__"])? {
                                Some(converted) => converted,
                                None => self.truncate_to_int(value)?,
                            }
                        }
                    }
                }
            }
            BuiltinType::Float => {
                expect_arity(&arguments, 0, 1)?;
                let mut argument = arguments.first().copied();
                if let Some(value) = argument {
                    if let Some(converted) = self.conversion_method(&value, "__float__")? {
                        if converted.tag() != ValueTag::Float {
                            let message = format!(
                                "{}.__float__ returned non-float (type {})",
                                self.type_name_of(&value)?,
                                self.type_name_of(&converted)?
                            );
                            return Err(self.raise_exception("TypeError", message));
                        }
                        return Ok(CallResult::Value(converted));
                    }
                    // CPython falls back to `__index__` and converts the int it returns.
                    if let Some(index) = self.int_by_method(&value, &["__index__"])? {
                        argument = Some(index);
                    }
                }
                let converted = match argument.as_ref() {
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
                        let text =
                            protocol::string_value(&self.state.heap, value)?.expect("guarded");
                        match text.trim().parse::<f64>() {
                            Ok(parsed) => parsed,
                            Err(_) => {
                                let message = format!(
                                    "could not convert string to float: {}",
                                    protocol::quote_string(&text)
                                );
                                return Err(self.raise_exception("ValueError", message));
                            }
                        }
                    }
                    Some(value) => match self.numeric_float(value) {
                        Ok(converted) => converted,
                        Err(_) if self.is_bigint(value)? => {
                            return Err(self.raise_exception(
                                "OverflowError",
                                "int too large to convert to float",
                            ))
                        }
                        Err(_) => {
                            let message = format!(
                                "float() argument must be a string or a real number, not '{}'",
                                self.type_name_of(value)?
                            );
                            return Err(self.raise_exception("TypeError", message));
                        }
                    },
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
                        let Ok(length) = usize::try_from(
                            protocol::int_value(&self.state.heap, value).expect("guarded"),
                        ) else {
                            return Err(self.raise_exception("ValueError", "negative count"));
                        };
                        self.reserve_result(length)?;
                        vec![0; length]
                    }
                    [value] => {
                        let items = self.iterable_values(value)?;
                        let mut bytes = Vec::with_capacity(items.len());
                        for item in items {
                            let byte = match super::number::index(&self.state.heap, &item) {
                                Some(super::number::NumberRef::Int(value)) => {
                                    u8::try_from(value).ok()
                                }
                                Some(_) => None,
                                None => {
                                    return Err(self.raise_object_type_error(
                                        &item,
                                        "cannot be interpreted as an integer",
                                    ))
                                }
                            };
                            let Some(byte) = byte else {
                                return Err(self.raise_exception(
                                    "ValueError",
                                    "bytes must be in range(0, 256)",
                                ));
                            };
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
            BuiltinType::List | BuiltinType::Tuple | BuiltinType::Set | BuiltinType::FrozenSet => {
                expect_arity(&arguments, 0, 1)?;
                let values = arguments
                    .first()
                    .map(|value| self.iterable_values(value))
                    .transpose()?
                    .unwrap_or_default();
                let object = match builtin_type {
                    BuiltinType::List => Object::List(values),
                    BuiltinType::Tuple => Object::Tuple(values),
                    BuiltinType::Set | BuiltinType::FrozenSet => {
                        let mut unique = Vec::new();
                        for value in values {
                            self.charge_cpu(1)?;
                            if self.find_value(&unique, &value)?.is_none() {
                                self.reserve_result(64)?;
                                unique.push(value);
                            }
                        }
                        if builtin_type == BuiltinType::Set {
                            Object::Set(unique)
                        } else {
                            Object::FrozenSet(unique)
                        }
                    }
                    _ => unreachable!(),
                };
                self.allocate_object(object)?
            }
            BuiltinType::Dict => unreachable!("dict construction returned above"),
            BuiltinType::Function
            | BuiltinType::Module
            | BuiltinType::Iterator
            | BuiltinType::Generator
            | BuiltinType::Exception
            | BuiltinType::Native
            | BuiltinType::Stream
            | BuiltinType::Environment
            | BuiltinType::NamespaceDict
            | BuiltinType::DictKeys
            | BuiltinType::DictValues
            | BuiltinType::DictItems
            | BuiltinType::MappingProxy
            | BuiltinType::ArgumentParser
            | BuiltinType::RaisesContext
            | BuiltinType::Property
            | BuiltinType::Regex
            | BuiltinType::Match
            | BuiltinType::Array => {
                return Err(format!("cannot create '{}' instances", builtin_type.name()));
            }
            BuiltinType::Complex => unreachable!("complex construction returned above"),
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
                NativeValue::BuiltinType(_)
                | NativeValue::ValueKind(_)
                | NativeValue::ExceptionType(_) => BuiltinType::Type.id(),
                NativeValue::Function(_)
                | NativeValue::NativeFunction(_)
                | NativeValue::NativeMethod(_)
                | NativeValue::SlotWrapper { .. } => BuiltinType::Function.id(),
                NativeValue::Module(_) => BuiltinType::Module.id(),
                NativeValue::Stream(_) => BuiltinType::Stream.id(),
                NativeValue::Environment => BuiltinType::Environment.id(),
                NativeValue::Ellipsis => BuiltinType::Ellipsis.id(),
                NativeValue::NotImplemented => BuiltinType::NotImplemented.id(),
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

    pub(super) fn type_of(&self, value: &Value) -> Result<Value, String> {
        // Builtin exception instances share one heap type; their class is the modeled
        // exception type, so `type(error) is ValueError` and `type(error).__name__` work.
        if let Some((kind, _)) = protocol::exception_parts(&self.state.heap, value)? {
            if let Some(definition) = exception_types::exception_type(&kind) {
                return Ok(Value::Native(NativeValue::ExceptionType(ExceptionType(
                    definition.name,
                ))));
            }
        }
        self.state.types.value(self.type_id(value)?)
    }

    pub(super) fn is_instance(&mut self, value: &Value, class: &Value) -> Result<bool, String> {
        if let Some(NativeValue::ExceptionType(ExceptionType(expected))) = class.native_value() {
            let actual =
                if let Some((kind, _)) = protocol::exception_parts(&self.state.heap, value)? {
                    kind
                } else if let Some(base) = self.user_exception_base(value)? {
                    base.to_string()
                } else {
                    return Ok(false);
                };
            return Ok(exception_types::exception_is_subclass(&actual, expected));
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
        if let Some(NativeValue::ExceptionType(ExceptionType(base))) = base.native_value() {
            if let Some(kind) = self.exception_class_base(class)? {
                return Ok(exception_types::exception_is_subclass(kind, base));
            }
            self.class_type_id(class)?
                .ok_or("issubclass() requires a class argument")?;
            return Ok(false);
        }
        if let Some(NativeValue::ExceptionType(_)) = class.native_value() {
            // A builtin exception class is never a subclass of a user class, and its only
            // non-exception ancestor is `object`.
            return Ok(matches!(
                base.native_value(),
                Some(NativeValue::BuiltinType(BuiltinType::Object))
            ));
        }
        let class = self
            .class_type_id(class)?
            .ok_or("issubclass() requires a class argument")?;
        let base = self
            .class_type_id(base)?
            .ok_or("issubclass() requires a class argument")?;
        self.state.types.is_subclass(class, base)
    }

    /// The modeled exception class that `class` is or derives from: the class itself for a
    /// builtin exception type, the recorded exception base for a user exception class, and
    /// `None` for anything else.
    pub(super) fn exception_class_base(
        &self,
        class: &Value,
    ) -> Result<Option<&'static str>, String> {
        if let Some(NativeValue::ExceptionType(ExceptionType(name))) = class.native_value() {
            return Ok(Some(name));
        }
        let Some(id) = class.object_id() else {
            return Ok(None);
        };
        Ok(match self.state.heap.get(id)? {
            Object::Class { exception_base, .. } => *exception_base,
            _ => None,
        })
    }
}
