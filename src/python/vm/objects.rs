//! Runtime object operations for attributes, subscription, descriptors, classes, and types.

use std::collections::BTreeSet;

use super::super::attributes;
use super::super::heap::{ClassObject, DictViewKind, FunctionObject, NamespaceTarget, ProxyTarget};
use super::super::scopes;
use super::namespace::{NamespaceHandle, ProxyHandle};
use super::{
    exception_types, expect_arity, number, protocol, range_length, select_string_slice, string,
    Arc, BuiltinSubscript, BuiltinType, CallArgs, CallMode, ClassDefinition, ClassField,
    ClassLayout, CodeCaches, CodeRef, ComparisonOperator, ExceptionType, Flow, FrameEntry, HashMap,
    LoadAttributeCache, NameId, NativeValue, Object, PyError, PyRuntime, RaisedException,
    SlicePlan, Slot, SlotValue, SymbolId, TypeId, Value, Vm, MODELED_MAPPING_ENTRY_BYTES,
};

/// Positional and keyword arguments of one call, as the call machinery passes them.
type CallArguments<'s> = (Vec<Value<'s>>, Vec<(String, Value<'s>)>);

/// Items of a builtin container whose `repr` the VM renders item by item.
enum ContainerItems<'s> {
    List(Vec<Value<'s>>),
    Tuple(Vec<Value<'s>>),
    Set(Vec<Value<'s>>),
    FrozenSet(Vec<Value<'s>>),
    Dict(Vec<(Value<'s>, Value<'s>)>),
}

/// Modeled bytes for one class attribute added after the class statement, beyond its name.
const CLASS_ATTRIBUTE_BYTES: u64 = 48;

/// Container nesting bound for `repr()`, matching the VM's call-depth limit.
const MAX_REPR_DEPTH: usize = 256;

/// Bytes a linear scan over text or bytes covers per CPU unit. Searches, comparisons and code
/// point walks run at memory speed, so one unit per word keeps the charge proportional without
/// making ordinary string handling expensive.
const SCAN_BYTES_PER_CPU_UNIT: usize = 16;

/// CPU units for scanning `bytes` bytes of text or binary data.
pub(crate) fn scan_cost(bytes: usize) -> u64 {
    u64::try_from(bytes / SCAN_BYTES_PER_CPU_UNIT).unwrap_or(u64::MAX) + 1
}
impl ContainerItems<'_> {
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

/// Whether user classes may derive from `builtin`. Their instances carry the builtin's payload
/// under the subclass's type id (see [`ClassLayout::Builtin`]).
pub(super) fn is_subclassable_builtin(builtin: BuiltinType) -> bool {
    matches!(
        builtin,
        BuiltinType::Int
            | BuiltinType::List
            | BuiltinType::Tuple
            | BuiltinType::Dict
            | BuiltinType::Set
            | BuiltinType::FrozenSet
            | BuiltinType::String
            | BuiltinType::Bytes
            | BuiltinType::ByteArray
            | BuiltinType::Float
            | BuiltinType::Complex
    )
}

impl<'s> Vm<'s> {
    pub(super) fn load_attribute(&mut self, name: &str) -> Result<(), String> {
        let owner = self.pop()?;
        let Some(value) = self.resolve_attribute(owner, name)? else {
            return Err(self.missing_attribute(&owner, name));
        };
        self.push(value);
        Ok(())
    }

    /// Report an attribute lookup that found nothing. Modules, user classes and their instances
    /// raise CPython's `AttributeError`, so `try`/`except` fallbacks for a missing module member
    /// work. On a builtin value a missing name is almost always a method shellsim does not model,
    /// so it stays an unsupported-feature error that user code cannot catch and misread as
    /// absence; `hasattr` and `getattr` defaults still observe the absence without raising. A
    /// missing special method (`__exit__`, `__bytes__`, ...) is the exception: programs probe
    /// those with `try`/`except AttributeError` to pick a protocol, and the set of special names
    /// a builtin type supports is a modeled decision rather than a gap.
    pub(super) fn missing_attribute(&mut self, owner: &Value<'s>, name: &str) -> String {
        if let Some(NativeValue::Module(module)) = owner.native_value() {
            let message = format!("module '{}' has no attribute '{name}'", module.name);
            return self.raise_exception("AttributeError", message);
        }
        let is_special = name.len() > 4 && name.starts_with("__") && name.ends_with("__");
        if is_special && !owner.is_object() {
            let message = match owner.native_value() {
                Some(NativeValue::BuiltinType(builtin)) => {
                    format!("type object '{}' has no attribute '{name}'", builtin.name())
                }
                _ => match self.type_name_of(owner) {
                    Ok(type_name) => format!("'{type_name}' object has no attribute '{name}'"),
                    Err(error) => return error,
                },
            };
            return self.raise_exception("AttributeError", message);
        }
        match self.instance_class(*owner) {
            Ok(Some(_)) => {
                let message = match self.type_name_of(owner) {
                    Ok(type_name) => format!("'{type_name}' object has no attribute '{name}'"),
                    Err(error) => return error,
                };
                return self.raise_exception("AttributeError", message);
            }
            Ok(None) => {}
            Err(error) => return error,
        }
        let message = match owner.is_object().then(|| self.get(*owner)) {
            Some(Ok(Object::Module { name: module, .. })) => {
                format!("module '{module}' has no attribute '{name}'")
            }
            Some(Ok(Object::Class(class_object))) => {
                let class_name = &class_object.name;
                format!("type object '{class_name}' has no attribute '{name}'")
            }
            Some(Ok(Object::Bare)) => match self.type_name_of(owner) {
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
        if owner.is_object() {
            if let Some(cache) = self.attribute_cache(code_cache, site) {
                if let Some(value) =
                    attributes::cached_attribute(self.heap(), owner, cache.type_id, cache.location)?
                {
                    self.push(value);
                    return Ok(());
                }
            }
        }
        let Some(value) = self.resolve_attribute_by_symbol(owner, symbol, name)? else {
            return Err(self.missing_attribute(&owner, name));
        };
        if let Some(cache) = self.cacheable_instance_attribute(owner, symbol, name)? {
            self.remember_attribute_cache(code, code_cache, site, cache)?;
        }
        self.push(value);
        Ok(())
    }

    /// `LoadMethod`: push the callable of `owner.name` and then its receiver, when `name` is a
    /// plain function or native method of `owner`'s type that no instance attribute shadows.
    /// `CallMethod` then passes the receiver as the first argument, which is what binding the
    /// method and calling the bound method would do, without creating it. Any other attribute
    /// is loaded as `LoadAttribute` would, followed by a no-receiver marker.
    pub(super) fn load_method_at(
        &mut self,
        code: &CodeRef,
        code_cache: usize,
        site: usize,
        symbol: SymbolId,
        name: &str,
    ) -> Result<(), String> {
        let owner = self.peek(0)?;
        if let Some(method) = self.unbound_method(owner, symbol, name)? {
            self.execution
                .stack
                .set(&self.state.heap, self.stack.len() - 1, method);
            self.push(owner);
            return Ok(());
        }
        self.load_attribute_at(code, code_cache, site, symbol, name)?;
        self.push(Value::Native(NativeValue::NoReceiver));
        Ok(())
    }

    fn unbound_method(
        &mut self,
        owner: Value<'s>,
        symbol: SymbolId,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
        // Only owners whose lookup is a plain walk of their type's MRO qualify: user instances
        // without an attribute hook, and builtin values. Classes, `super` proxies and native
        // module or type values resolve names their own way in `lookup_attribute`.
        if owner.native_value().is_some() || self.class_type_id(&owner)?.is_some() {
            return Ok(None);
        }
        let type_id = self.type_id(&owner)?;
        if owner.is_object() {
            if matches!(self.get(owner)?, Object::Class(_) | Object::Super { .. }) {
                return Ok(None);
            }
            if self.instance_class(owner)?.is_some()
                && self
                    .state
                    .types
                    .slot(type_id, Slot::GetAttribute)?
                    .is_some()
            {
                return Ok(None);
            }
        }
        let Some((_, descriptor)) = self.type_lookup(type_id, name)? else {
            return Ok(None);
        };
        let is_method = match descriptor.native_value() {
            Some(NativeValue::NativeMethod(method)) => method.name != "__new__",
            Some(_) => false,
            None => descriptor.is_object() && matches!(self.get(descriptor)?, Object::Function(_)),
        };
        if !is_method {
            return Ok(None);
        }
        // An instance attribute of the same name shadows a non-data descriptor.
        if owner.is_object() && self.attribute_by_symbol(owner, symbol)?.is_some() {
            return Ok(None);
        }
        Ok(Some(descriptor))
    }

    fn attribute_cache(&self, code_cache: usize, site: usize) -> Option<&LoadAttributeCache> {
        self.execution
            .code_caches
            .get(code_cache)?
            .attributes
            .as_ref()?
            .get(site)?
            .as_ref()
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
        let mut attributes: Vec<Option<LoadAttributeCache>> = std::iter::repeat_with(|| None)
            .take(code.instructions.len())
            .collect();
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
        let symbol = self.intern_symbol(code.name(name))?;
        self.execution.code_caches[cache].names[name.index()] = Some(symbol);
        Ok(symbol)
    }

    pub(super) fn ensure_code_cache(&mut self, code: &CodeRef) -> Result<usize, String> {
        if let Some(slot) = self.execution.code_caches.slot_of(code) {
            return Ok(slot);
        }
        let bytes = code
            .name_count()
            .checked_mul(std::mem::size_of::<Option<SymbolId>>())
            .and_then(|names| names.checked_add(std::mem::size_of::<CodeCaches>()))
            .ok_or("code cache size overflow")?;
        self.reserve_retained_memory(bytes)?;
        let caches = CodeCaches {
            code: code.clone(),
            names: vec![None; code.name_count()],
            attributes: None,
        };
        let mut released = 0usize;
        let slot = self
            .execution
            .code_caches
            .insert(caches, |bytes| released = released.saturating_add(bytes));
        if released > 0 {
            self.release_retained_memory_bytes(released);
        }
        Ok(slot)
    }

    fn cacheable_instance_attribute(
        &mut self,
        owner: Value<'s>,
        symbol: SymbolId,
        name: &str,
    ) -> Result<Option<LoadAttributeCache>, String> {
        if self.instance_class(owner)?.is_none() {
            return Ok(None);
        }
        let class_type = self.type_id(&owner)?;
        if let Some((_, descriptor)) = self.type_lookup(class_type, name)? {
            if self.is_data_descriptor(&descriptor)? {
                return Ok(None);
            }
        }
        Ok(self
            .instance_attribute_slot_by_symbol(owner, symbol)?
            .map(|location| LoadAttributeCache {
                type_id: class_type,
                location,
            }))
    }

    /// Resolve one attribute without involving the operand stack.
    ///
    /// Missing attributes return `None`; errors raised while invoking descriptors remain errors.
    /// This distinction lets `getattr` and `hasattr` share the bytecode lookup path.
    pub(super) fn resolve_attribute(
        &mut self,
        owner: Value<'s>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
        let symbol = self.symbol_id(name);
        self.resolve_attribute_inner(owner, symbol, name)
    }

    /// Resolve an attribute where an `AttributeError` raised during the lookup means the
    /// attribute is absent, as in CPython's `PyObject_GetOptionalAttr`. `getattr` defaults,
    /// `hasattr` and `from module import name` use this, so a module `__getattr__` or property
    /// that raises `AttributeError` reads as a missing name while other exceptions propagate.
    pub(super) fn resolve_optional_attribute(
        &mut self,
        owner: Value<'s>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
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
        owner: Value<'s>,
        symbol: SymbolId,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
        self.resolve_attribute_inner(owner, Some(symbol), name)
    }

    fn resolve_attribute_inner(
        &mut self,
        owner: Value<'s>,
        symbol: Option<SymbolId>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
        let getattribute: Result<Option<Value<'s>>, String> = if owner.is_object()
            && (self.instance_class(owner)?.is_some()
                || matches!(self.get(owner), Ok(Object::Class { .. })))
            && self
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
                let matches = self.exception_type_matches(attribute_error, &exception)?;
                // The hook runs guest code that may allocate, so hold the taken exception
                // through a handle rather than its stored reference.
                let RaisedException { kind, value } = exception;
                let value = self.handle(&value);
                if matches {
                    if let Some(value) = self.call_type_getattr_hook(owner, name)? {
                        return Ok(Some(value));
                    }
                }
                self.pending_exception = Some(RaisedException {
                    kind,
                    value: self.store(value),
                });
                return Err(error);
            }
        };
        match found {
            Some(value) => Ok(Some(value)),
            None => self.call_type_getattr_hook(owner, name),
        }
    }

    /// The default attribute lookup for an instance of user class `class`: a data descriptor on
    /// the class wins, then the instance's own attributes, then any other class attribute.
    fn lookup_instance_attribute(
        &mut self,
        owner: Value<'s>,
        class: Value<'s>,
        symbol: Option<SymbolId>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
        let class_type = self
            .class_type_id(&class)?
            .ok_or("instance has no registered class")?;
        let class_entry = self.type_lookup(class_type, name)?;
        if let Some((defining_type, descriptor)) = class_entry {
            // A native getter on a user exception instance models a C struct slot such as
            // `OSError.filename`: an assigned value replaces the derived one.
            let overridable_getter = matches!(
                descriptor.native_value(),
                Some(NativeValue::NativeGetter(_))
            ) && exception_types::exception_base(self.state, owner)?
                .is_some();
            let instance_value = match (overridable_getter, symbol) {
                (true, Some(symbol)) => self.attribute_by_symbol(owner, symbol)?,
                _ => None,
            };
            if let Some(value) = instance_value {
                return Ok(Some(value));
            }
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
            Some(symbol) => self.attribute_by_symbol(owner, symbol)?,
            None => None,
        };
        if let Some(value) = instance_value {
            return Ok(Some(value));
        }
        let Some((defining_type, descriptor)) = class_entry else {
            return Ok(None);
        };
        self.bind_type_attribute(descriptor, Some(owner), class_type, defining_type)
    }

    /// The default object lookup without the outer `__getattr__` fallback. Direct calls to
    /// `object.__getattribute__` and ordinary attribute reads share this descriptor algorithm.
    pub(super) fn lookup_attribute_default(
        &mut self,
        owner: Value<'s>,
        symbol: Option<SymbolId>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
        self.lookup_attribute(owner, symbol, name)
    }

    fn call_type_getattr_hook(
        &mut self,
        owner: Value<'s>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
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
        owner: Value<'s>,
        symbol: Option<SymbolId>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
        // A metatype data descriptor wins over a class's own MRO, for native and user classes.
        if self.class_type_id(&owner)?.is_some() {
            let metatype = self.type_id(&owner)?;
            if let Some((defining_type, descriptor)) = self.type_lookup(metatype, name)? {
                if self.is_data_descriptor(&descriptor)? {
                    return self.bind_type_attribute(
                        descriptor,
                        Some(owner),
                        metatype,
                        defining_type,
                    );
                }
            }
        }
        if let Some(NativeValue::Module(module)) = owner.native_value() {
            if let Some(function) = module.function(name) {
                return Ok(Some(Value::Native(NativeValue::NativeFunction(function))));
            }
            if let Some(value) = module.value(name) {
                let value = value.get(self).map_err(|error| error.to_string())?;
                return Ok(Some(value));
            }
        }
        if let Some(NativeValue::BuiltinType(builtin)) = owner.native_value() {
            if let Some((defining_type, value)) = self.type_lookup(builtin.id(), name)? {
                if let Some(NativeValue::NativeClassMethod(method)) = value.native_value() {
                    return self.bind_native_class_method(owner, method).map(Some);
                }
                if matches!(
                    value.native_value(),
                    Some(NativeValue::SlotWrapper {
                        slot: Slot::ClassGetItem,
                        ..
                    })
                ) {
                    return self.bind_type_attribute(
                        value,
                        Some(owner),
                        builtin.id(),
                        defining_type,
                    );
                }
                return Ok(Some(value));
            }
        }
        if let Some(NativeValue::ExceptionType(ExceptionType(kind))) = owner.native_value() {
            let type_id = self
                .state
                .types
                .exception_type_id(kind)
                .ok_or("exception type is not registered")?;
            if let Some((_, value)) = self.type_lookup(type_id, name)? {
                return Ok(Some(value));
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
        let instance_class = self.instance_class(owner)?;
        let owner_has_custom_lookup = instance_class.is_some()
            || (owner.is_object()
                && matches!(
                    self.get(owner),
                    Ok(Object::Class { .. } | Object::Super { .. })
                ));
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
        if let Some(class) = instance_class {
            return self.lookup_instance_attribute(owner, class, symbol, name);
        }
        if let Some(symbol) = symbol {
            if exception_types::exception_base(self.state, owner)?.is_some() {
                if let Some(value) = self.attribute_by_symbol(owner, symbol)? {
                    return Ok(Some(value));
                }
            }
        }
        if owner.is_object() {
            match self.get(owner)? {
                Object::Module { scope, .. } => {
                    let namespace = self.module_namespace(self.handle(scope))?;
                    if let Some(value) = self.namespace_lookup(namespace, name)? {
                        return Ok(Some(value));
                    }
                    // PEP 562: a module-level `__getattr__` supplies names the module does not
                    // define, which is how packages import submodules lazily. It runs after
                    // the type's attributes, of which modules model only `__class__`.
                    let hook = self.namespace_lookup(namespace, "__getattr__")?;
                    let Some(hook) = hook.filter(|_| name != "__class__") else {
                        return Ok(None);
                    };
                    let name = self.allocate_string(name.to_string())?;
                    return self.invoke_value(hook, vec![name]).map(Some);
                }
                Object::Function(function_object) => {
                    let FunctionObject {
                        name: function_name,
                        closure,
                        attributes,
                        ..
                    } = &**function_object;
                    if name == "__name__" {
                        let function_name = function_name.clone();
                        return Ok(Some(self.allocate_string(function_name)?));
                    }
                    if let Some(value) = attributes.get(name) {
                        return Ok(Some(self.handle(value)));
                    }
                    if name == "__doc__" {
                        let Some(docstring) = function_object.code.docstring.clone() else {
                            return Ok(Some(Value::None));
                        };
                        return Ok(Some(self.allocate_string(docstring.to_string())?));
                    }
                    if name == "__module__" {
                        let closure = self.handle_optional(closure.as_ref());
                        return self.module_name_of(closure);
                    }
                }
                Object::Class { .. } => {
                    let class_type = self
                        .class_type_id(&owner)?
                        .ok_or("class has no registered type")?;
                    let metaclass_entry = self.type_lookup(owner_type, name)?;
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
                Object::GenericAlias { origin, arguments } => {
                    return match name {
                        "__origin__" => Ok(Some(self.handle(origin))),
                        "__args__" => {
                            let arguments = self.handles(arguments);
                            self.alloc_with(|b| Object::Tuple(b.refs(arguments)))
                                .map(Some)
                        }
                        _ => Ok(None),
                    };
                }
                Object::Super {
                    start_class,
                    receiver,
                } => {
                    let start_class = self.handle(start_class);
                    let receiver = self.handle(receiver);
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
                    let receiver = self.handle(receiver);
                    let descriptor = self.handle(descriptor);
                    return match name {
                        "__self__" => Ok(Some(receiver)),
                        "__func__" => Ok(Some(descriptor)),
                        _ => self.lookup_attribute(descriptor, None, name),
                    };
                }
                Object::Native(native) => {
                    if let Some(slot) = native.attribute(name) {
                        return Ok(Some(self.handle(slot)));
                    }
                }
                // `classmethod` and `staticmethod` expose the wrapped callable and forward the
                // metadata attributes `functools.wraps` and `abc` read from it.
                Object::StaticMethod { callable } | Object::ClassMethod { callable } => {
                    let callable = self.handle(callable);
                    if name == "__func__" || name == "__wrapped__" {
                        return Ok(Some(callable));
                    }
                    let forwarded = self.lookup_attribute(callable, None, name)?;
                    if name == "__isabstractmethod__" {
                        return Ok(Some(forwarded.unwrap_or(Value::Bool(false))));
                    }
                    return Ok(forwarded);
                }
                Object::Slice { start, stop, step } => {
                    let component = match name {
                        "start" => start,
                        "stop" => stop,
                        "step" => step,
                        _ => return Ok(None),
                    };
                    return Ok(Some(self.handle(component)));
                }
                _ => {}
            }
        }
        if let Some(value) = self.native_type_metadata(owner, name)? {
            return Ok(Some(value));
        }
        let native_name =
            match owner.native_value() {
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

    /// Materialize the live namespace selected by a `__dict__` data descriptor.
    pub(super) fn dictionary_of(&mut self, owner: Value<'s>) -> Result<Option<Value<'s>>, String> {
        let namespace = match owner.native_value() {
            Some(NativeValue::Module(module)) => {
                Some(Object::MappingProxy(ProxyTarget::NativeModule(module)))
            }
            Some(NativeValue::BuiltinType(builtin)) => Some(Object::MappingProxy(
                ProxyTarget::RegisteredType(builtin.id()),
            )),
            Some(NativeValue::ValueKind(kind)) => {
                let type_id = self
                    .state
                    .types
                    .value_kind_type_id(kind)
                    .ok_or("value kind is not registered")?;
                Some(Object::MappingProxy(ProxyTarget::RegisteredType(type_id)))
            }
            Some(NativeValue::ExceptionType(ExceptionType(name))) => {
                let type_id = self
                    .state
                    .types
                    .exception_type_id(name)
                    .ok_or("exception type is not registered")?;
                Some(Object::MappingProxy(ProxyTarget::RegisteredType(type_id)))
            }
            _ if self.has_instance_dict(owner)? => {
                return self
                    .alloc_with(|b| {
                        Object::NamespaceDict(NamespaceTarget::Instance(b.store(owner)))
                    })
                    .map(Some);
            }
            _ if owner.is_object() => {
                return match self.get(owner)? {
                    Object::Class { .. } => self
                        .alloc_with(|b| Object::MappingProxy(ProxyTarget::Class(b.store(owner))))
                        .map(Some),
                    Object::Module { scope, .. } => {
                        let namespace = self.module_namespace(self.handle(scope))?;
                        self.alloc_with(|b| Object::NamespaceDict(namespace.store(b)))
                            .map(Some)
                    }
                    _ => Ok(None),
                };
            }
            _ => None,
        };
        namespace
            .map(|namespace| self.allocate_object(namespace))
            .transpose()
    }

    /// Return class metadata through `type`'s data descriptors. User classes keep their own
    /// namespace and MRO; native classes read the same fields from the type registry.
    pub(super) fn type_metadata(
        &mut self,
        owner: Value<'s>,
        field: super::super::native::TypeMetadata,
    ) -> Result<Option<Value<'s>>, String> {
        use super::super::native::TypeMetadata;
        if owner.is_object() {
            if let Object::Class(class_object) = self.get(owner)? {
                return match field {
                    TypeMetadata::Name => {
                        let name = class_object.name.clone();
                        self.allocate_string(name).map(Some)
                    }
                    TypeMetadata::Module => {
                        Ok(self.handle_optional(class_object.attributes.get("__module__")))
                    }
                    TypeMetadata::Bases => self.class_metadata(owner, "__bases__"),
                    TypeMetadata::Mro => self.class_metadata(owner, "__mro__"),
                };
            }
        }
        match field {
            TypeMetadata::Name => {
                let name = match owner.native_value() {
                    Some(NativeValue::BuiltinType(builtin)) => builtin.name(),
                    Some(NativeValue::ValueKind(kind)) => kind.name,
                    Some(NativeValue::ExceptionType(ExceptionType(name))) => name,
                    _ => return Ok(None),
                };
                self.allocate_string(name.rsplit('.').next().unwrap_or(name).to_string())
                    .map(Some)
            }
            TypeMetadata::Module => self.native_type_metadata(owner, "__module__"),
            TypeMetadata::Bases => self.native_type_metadata(owner, "__bases__"),
            TypeMetadata::Mro => self.native_type_metadata(owner, "__mro__"),
        }
    }

    /// Assign `owner.name = value`. A class that defines `__setattr__` receives the assignment;
    /// otherwise [`Vm::store_attribute_default`] performs it.
    pub(super) fn store_attribute_by_symbol(
        &mut self,
        owner: Value<'s>,
        symbol: SymbolId,
        name: &str,
        value: Value<'s>,
    ) -> Result<(), String> {
        if let Some(class) = self.instance_class(owner)? {
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
        owner: Value<'s>,
        symbol: SymbolId,
        name: &str,
        value: Value<'s>,
    ) -> Result<(), String> {
        // `BaseException.args` is writable and stores any iterable as a tuple.
        if name == "args" && exception_types::exception_base(self.state, owner)?.is_some() {
            let current = match self.get(owner)? {
                Object::Exception(args) => args.len(),
                _ => return Err("exception instance has a non-exception layout".into()),
            };
            let items = self.iterable_values(&value)?;
            let bytes = u64::try_from(items.len().saturating_sub(current))
                .unwrap_or(u64::MAX)
                .saturating_mul(super::MODELED_VALUE_BYTES);
            self.reserve_object_growth(owner, bytes)?;
            return self.modify(owner, |b, object| {
                let Object::Exception(args) = object else {
                    unreachable!("checked above")
                };
                *args = b.refs(items);
            });
        }
        let Some(class) = self.instance_class(owner)? else {
            if owner.is_object() {
                match self.get(owner)? {
                    Object::Class { .. } => {
                        return self.set_class_attribute(owner, name, Some(value))
                    }
                    Object::Function { .. } => {
                        return self.set_function_attribute(owner, name, Some(value))
                    }
                    Object::Module { scope, .. } => {
                        let namespace = self.module_namespace(self.handle(scope))?;
                        return self.namespace_store(namespace, name.to_string(), value);
                    }
                    _ => {}
                }
                // A builtin exception instance has an attribute dictionary like any instance.
                if exception_types::exception_base(self.state, owner)?.is_some() {
                    return self.insert_attribute(owner, name, value);
                }
            }
            return Err(self.reject_builtin_attribute_store(owner, name));
        };
        if name == "__dict__" {
            return self.replace_instance_attributes(owner, value);
        }
        let class_type = self
            .class_type_id(&class)?
            .ok_or("instance has no registered class")?;
        if let Some((_, descriptor)) = self.type_lookup(class_type, name)? {
            if descriptor.is_object() {
                let descriptor_class = self.instance_class(descriptor)?;
                match self.get(descriptor)? {
                    Object::Property {
                        setter: Some(setter),
                        ..
                    } => {
                        let setter = self.handle(setter);
                        self.invoke_value(setter, vec![owner, value])?;
                        return Ok(());
                    }
                    Object::Property { setter: None, .. } => {
                        let message = format!(
                            "property '{name}' of '{}' object has no setter",
                            self.type_name_of(&owner)?
                        );
                        return Err(self.raise_exception("AttributeError", message));
                    }
                    _ if descriptor_class.is_some() => {
                        let descriptor_class = descriptor_class.expect("checked above");
                        if let Some((set_owner, set)) =
                            self.class_attribute_entry(descriptor_class, "__set__")?
                        {
                            let set = self.bind_descriptor(
                                set,
                                Some(descriptor),
                                descriptor_class,
                                set_owner,
                            )?;
                            self.invoke_value(set, vec![owner, value])?;
                            return Ok(());
                        }
                    }
                    _ => {}
                }
            }
            if matches!(
                descriptor.native_value(),
                Some(NativeValue::NativeGetter(_))
            ) && !(matches!(
                name,
                "args" | "errno" | "strerror" | "filename" | "filename2"
            ) && exception_types::exception_base(self.state, owner)?.is_some())
            {
                let type_name = self.type_name_of(&owner)?;
                return Err(self.raise_exception(
                    "AttributeError",
                    format!("attribute '{name}' of '{type_name}' objects is not writable"),
                ));
            }
        }
        self.insert_attribute_by_symbol(owner, symbol, value)?;
        Ok(())
    }

    /// `obj.__dict__ = mapping`: replace every attribute of instance `id` with `mapping`'s
    /// entries. The entries are copied, so later changes to `mapping` do not reach the instance;
    /// CPython instead makes the instance adopt the dict itself.
    fn replace_instance_attributes(
        &mut self,
        instance: Value<'s>,
        mapping: Value<'s>,
    ) -> Result<(), String> {
        let source = mapping.is_object().then(|| self.get(mapping)).transpose()?;
        let entries = match source {
            Some(Object::Dict(entries) | Object::DefaultDict { entries, .. }) => {
                let entries: Vec<_> = entries
                    .iter()
                    .map(|(key, value)| (self.handle(key), self.handle(value)))
                    .collect();
                let mut named = Vec::with_capacity(entries.len());
                for (key, value) in entries {
                    let Some(name) = string::string_value(&self.state.heap, key)? else {
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
                let target = self.namespace_handle(target);
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
        let current = self.instance_attribute_names(instance)?;
        self.charge_cpu(u64::try_from(current.len() + entries.len()).unwrap_or(u64::MAX))?;
        for name in current {
            self.namespace_delete(NamespaceHandle::Instance(instance), &name)?;
        }
        for (name, value) in entries {
            self.namespace_store(NamespaceHandle::Instance(instance), name, value)?;
        }
        Ok(())
    }

    /// Delete `owner.name` through its type slot, then use the default descriptor algorithm if
    /// the type supplies no deletion override.
    pub(super) fn delete_attribute_by_symbol(
        &mut self,
        owner: Value<'s>,
        symbol: SymbolId,
        name: &str,
    ) -> Result<(), String> {
        if self
            .state
            .types
            .slot(self.type_id(&owner)?, Slot::DeleteAttribute)?
            .is_some()
        {
            let name = self.allocate_string(name.to_string())?;
            self.invoke_slot(&owner, Slot::DeleteAttribute, "__delattr__", vec![name])?;
            return Ok(());
        }
        self.delete_attribute_default(owner, symbol, name)
    }

    /// The deletion `object.__delattr__` performs: a data descriptor's `__delete__`, or else
    /// removal from the instance's own attributes. Classes and modules delete from their
    /// namespaces.
    pub(super) fn delete_attribute_default(
        &mut self,
        owner: Value<'s>,
        symbol: SymbolId,
        name: &str,
    ) -> Result<(), String> {
        if !owner.is_object() {
            return Err(self.reject_builtin_attribute_store(owner, name));
        }
        let class = match self.get(owner)? {
            _ if self.instance_class(owner)?.is_some() => {
                self.instance_class(owner)?.expect("checked above")
            }
            _ if exception_types::exception_base(self.state, owner)?.is_some() => {
                if self.remove_attribute_by_symbol(owner, symbol)?.is_none() {
                    return Err(self.missing_attribute(&owner, name));
                }
                return Ok(());
            }
            Object::Class { .. } => return self.set_class_attribute(owner, name, None),
            Object::Function { .. } => return self.set_function_attribute(owner, name, None),
            Object::Module { scope, .. } => {
                let namespace = self.module_namespace(self.handle(scope))?;
                if self.namespace_delete(namespace, name)?.is_none() {
                    return Err(self.missing_attribute(&owner, name));
                }
                return Ok(());
            }
            _ => return Err(self.reject_builtin_attribute_store(owner, name)),
        };
        let class_type = self
            .class_type_id(&class)?
            .ok_or("instance has no registered class")?;
        if let Some((_, descriptor)) = self.type_lookup(class_type, name)? {
            if descriptor.is_object() {
                let descriptor_class = self.instance_class(descriptor)?;
                match self.get(descriptor)? {
                    Object::Property { .. } => {
                        let message = format!(
                            "property '{name}' of '{}' object has no deleter",
                            self.type_name_of(&owner)?
                        );
                        return Err(self.raise_exception("AttributeError", message));
                    }
                    _ if descriptor_class.is_some() => {
                        let descriptor_class = descriptor_class.expect("checked above");
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
        if self.remove_attribute_by_symbol(owner, symbol)?.is_none() {
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
        function: Value<'s>,
        name: &str,
        value: Option<Value<'s>>,
    ) -> Result<(), String> {
        if name == "__name__" {
            let Some(text) = value
                .map(|value| string::string_ref(&self.state.heap, value))
                .transpose()?
                .flatten()
                .map(|text| text.as_str().to_string())
            else {
                return Err(
                    self.raise_exception("TypeError", "__name__ must be set to a string object")
                );
            };
            let Object::Function(function_object) = self.get_mut(function)? else {
                unreachable!("checked by the caller")
            };
            let FunctionObject { name, .. } = &mut **function_object;
            *name = text;
            return Ok(());
        }
        let Object::Function(function_object) = self.get(function)? else {
            return Err("function attribute store on a non-function".into());
        };
        let FunctionObject { attributes, .. } = &**function_object;
        let exists = attributes.contains_key(name);
        match value {
            Some(value) => {
                if !exists {
                    let bytes = u64::try_from(name.len())
                        .unwrap_or(u64::MAX)
                        .saturating_add(MODELED_MAPPING_ENTRY_BYTES);
                    self.reserve_object_growth(function, bytes)?;
                }
                self.modify(function, |b, object| {
                    let Object::Function(function_object) = object else {
                        unreachable!("checked above")
                    };
                    let FunctionObject { attributes, .. } = &mut **function_object;
                    attributes.insert(name.to_string(), b.store(value));
                })?;
            }
            None if exists => {
                let Object::Function(function_object) = self.get_mut(function)? else {
                    unreachable!("checked above")
                };
                let FunctionObject { attributes, .. } = &mut **function_object;
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
        class: Value<'s>,
        name: &str,
        value: Option<Value<'s>>,
    ) -> Result<(), String> {
        if matches!(name, "__dict__" | "__name__" | "__bases__" | "__mro__") {
            return Err(format!(
                "assigning or deleting a class's {name} is not supported"
            ));
        }
        let Object::Class(class_object) = self.get(class)? else {
            return Err("class attribute store on a non-class".into());
        };
        let (attributes, instance_type) = (&class_object.attributes, class_object.instance_type);
        match value {
            Some(value) => {
                if !attributes.contains_key(name) {
                    let bytes = u64::try_from(name.len())
                        .unwrap_or(u64::MAX)
                        .saturating_add(CLASS_ATTRIBUTE_BYTES);
                    self.reserve_object_growth(class, bytes)?;
                }
                self.modify(class, |b, object| {
                    let Object::Class(class_object) = object else {
                        unreachable!("checked above")
                    };
                    class_object
                        .attributes
                        .insert(name.to_string(), b.store(value));
                })?;
            }
            None => {
                let Object::Class(class_object) = self.get_mut(class)? else {
                    unreachable!("checked above")
                };
                if class_object.attributes.remove(name).is_none() {
                    return Err(self.missing_attribute(&class, name));
                }
            }
        }
        self.charge_cpu(u64::try_from(self.state.types.len()).unwrap_or(u64::MAX))?;
        for type_id in self.state.types.subtypes(instance_type) {
            let subclass = self.type_value(type_id)?;
            if !subclass.is_object() {
                continue;
            }
            let state = &mut *self.state;
            let Object::Class(class_object) = state.heap.get(subclass)? else {
                continue;
            };
            state
                .types
                .replace_slots(type_id, &class_object.attributes)?;
        }
        for cache in self.execution.code_caches.iter_mut() {
            cache.attributes = None;
        }
        Ok(())
    }

    /// Raise CPython's `AttributeError` for a store on a receiver without an instance
    /// dictionary: data descriptors are not writable, other type attributes are read-only, and
    /// new names have nowhere to go.
    fn reject_builtin_attribute_store(&mut self, owner: Value<'s>, name: &str) -> String {
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

    /// The class object of `value` when it is an instance of a user class, whatever payload the
    /// class's layout gave it; `None` for builtin values and class objects.
    pub(super) fn instance_class(&self, value: Value<'s>) -> Result<Option<Value<'s>>, String> {
        self.state.types.instance_class(&self.state.heap, value)
    }

    /// Whether `value` keeps its own attribute dictionary: an instance of a user class, or an
    /// exception instance, whose builtin classes also give instances a `__dict__`.
    pub(super) fn has_instance_dict(&self, value: Value<'s>) -> Result<bool, String> {
        Ok(self.instance_class(value)?.is_some()
            || exception_types::exception_base(self.state, value)?.is_some())
    }

    /// Whether `value` is an instance of a user class with the plain `object` layout and no
    /// exception base: the objects whose default `repr()` and `str()` are `<Name object at ...>`.
    fn is_plain_instance(&self, value: &Value<'s>) -> Result<bool, String> {
        let Some(class) = self.instance_class(*value)? else {
            return Ok(false);
        };
        if !matches!(self.get(*value)?, Object::Bare) {
            return Ok(false);
        }
        Ok(matches!(
            self.get(class)?,
            Object::Class(class_object) if class_object.exception_base.is_none()
        ))
    }

    pub(super) fn load_subscript(&mut self) -> Result<(), String> {
        let index = self.pop()?;
        let owner = self.pop()?;
        let value = self.subscript_value(owner, index)?;
        self.push(value);
        Ok(())
    }

    /// Build an alias after the class-subscription slot has selected a builtin container.
    pub(super) fn new_generic_alias(
        &mut self,
        origin: Value<'s>,
        item: Value<'s>,
    ) -> Result<Value<'s>, String> {
        let count = match item.is_object().then(|| self.get(item)).transpose()? {
            Some(Object::Tuple(arguments)) => arguments.len(),
            _ => 1,
        };
        self.charge_cpu(u64::try_from(count).unwrap_or(u64::MAX))?;
        self.reserve_result(count.saturating_mul(std::mem::size_of::<Value>()))?;
        let arguments = match item.is_object().then(|| self.get(item)).transpose()? {
            Some(Object::Tuple(arguments)) => self.handles(arguments),
            _ => vec![item],
        };
        self.alloc_with(|b| Object::GenericAlias {
            origin: b.store(origin),
            arguments: b.refs(arguments),
        })
    }

    /// `owner[index]`: the owner's `__getitem__` or its builtin subscript.
    pub(super) fn subscript_value(
        &mut self,
        owner: Value<'s>,
        index: Value<'s>,
    ) -> Result<Value<'s>, String> {
        if let Some(value) = self.invoke_slot(&owner, Slot::GetItem, "__getitem__", vec![index])? {
            return Ok(value);
        }
        if let Some(class_type) = self.class_type_id(&owner)? {
            if let Some((defining_type, descriptor)) =
                self.type_lookup(class_type, "__class_getitem__")?
            {
                let method = self
                    .bind_type_attribute(descriptor, Some(owner), class_type, defining_type)?
                    .ok_or("__class_getitem__ descriptor has no value")?;
                return self.invoke_value(method, vec![index]);
            }
        }
        self.subscript_builtin(owner, index)
    }

    /// Index a builtin payload directly. A native `__getitem__` slot calls this after type-slot
    /// dispatch, so the implementation cannot redispatch into its own wrapper.
    pub(super) fn subscript_builtin(
        &mut self,
        owner: Value<'s>,
        index: Value<'s>,
    ) -> Result<Value<'s>, String> {
        let subject = owner;
        // A slice is an ordinary key to a mapping; only sequences slice with it.
        let mapping = match owner.is_object() {
            true => matches!(
                self.get(owner)?,
                Object::Dict(_) | Object::DefaultDict { .. }
            ),
            false => false,
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
                _ => protocol::repr(self.state, index)?,
            };
            self.allocate_string(format!("typing.List[{parameter}]"))?
        } else if matches!(owner.native_value(), Some(NativeValue::Environment)) {
            let name = string::string_ref(&self.state.heap, index)?
                .ok_or("environment key must be a string")?
                .as_str()
                .to_string();
            let Some(value) = self.interp.get_var(&name) else {
                let key = self.allocate_string(name)?;
                return Err(self.raise_exception_args("KeyError", vec![key]));
            };
            self.allocate_string(value)?
        } else if let Some(character) = self.string_subscript(&owner, &index)? {
            self.allocate_string(character.to_string())?
        } else if owner.is_object() {
            let target = match self.get(owner)? {
                Object::List(values) | Object::Tuple(values) => {
                    let sequence = if matches!(self.get(owner)?, Object::List(_)) {
                        "list"
                    } else {
                        "tuple"
                    };
                    let Some(index) = number::int_value(&self.state.heap, index) else {
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
                        Some(value) => BuiltinSubscript::Value(self.handle(value)),
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
                    let Some(index) = number::int_value(&self.state.heap, index) else {
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
                    factory: Some(self.handle(factory)),
                },
                Object::Set(_) => BuiltinSubscript::Set,
                Object::Bare
                | Object::String(_)
                | Object::Bytes(_)
                | Object::ByteArray(_)
                | Object::Slice { .. }
                | Object::Exception(_)
                | Object::FrozenSet(_)
                | Object::BigInt(_)
                | Object::Complex { .. }
                | Object::Function { .. }
                | Object::Class { .. }
                | Object::Float(_)
                | Object::DescriptorBoundMethod { .. }
                | Object::Iterator { .. }
                | Object::SequenceIterator { .. }
                | Object::ReverseIterator { .. }
                | Object::RangeIterator { .. }
                | Object::CountIterator { .. }
                | Object::StreamIterator { .. }
                | Object::CallableIterator { .. }
                | Object::Generator { .. }
                | Object::Module { .. }
                | Object::Scope(_)
                | Object::WideValue { .. }
                | Object::Native(_)
                | Object::NamespaceDict(_)
                | Object::DictView { .. }
                | Object::MappingProxy(_) => BuiltinSubscript::Unsupported,
                Object::GenericAlias { .. } => BuiltinSubscript::Unsupported,
                Object::Property { .. }
                | Object::StaticMethod { .. }
                | Object::ClassMethod { .. }
                | Object::Super { .. } => BuiltinSubscript::Unsupported,
            };
            match target {
                BuiltinSubscript::Value(value) => value,
                BuiltinSubscript::Mapping { factory } => {
                    let (hash, position) = self.lookup_mapping_entry(owner, &index)?;
                    if let Some(position) = position {
                        match self.get(owner)? {
                            Object::Dict(entries) | Object::DefaultDict { entries, .. } => self
                                .handle(
                                    &entries
                                        .get(position)
                                        .ok_or("dictionary changed size during lookup")?
                                        .1,
                                ),
                            _ => unreachable!("mapping kind was classified before lookup"),
                        }
                    } else if let Some(factory) = factory {
                        self.push(factory);
                        let result = self.call(0, &[], &[], CallMode::Immediate);
                        let value = self.immediate_call_value(result, "default factory")?;
                        self.reserve_object_growth(owner, MODELED_MAPPING_ENTRY_BYTES)?;
                        self.modify(owner, |b, object| {
                            let Object::DefaultDict { entries, .. } = object else {
                                unreachable!("defaultdict kind was classified before insertion")
                            };
                            entries.push(hash, (b.store(index), b.store(value)));
                        })?;
                        value
                    } else if let Some(value) = self.missing_key(&subject, index)? {
                        value
                    } else {
                        return Err(self.raise_exception_args("KeyError", vec![index]));
                    }
                }
                BuiltinSubscript::Unsupported => {
                    // A native type such as `re.Match` is subscriptable through the
                    // `__getitem__` method it publishes; the method sees the object itself.
                    let type_id = self.type_id(&owner)?;
                    let native_method =
                        self.type_lookup(type_id, "__getitem__")?
                            .and_then(|(_, descriptor)| match descriptor.native_value() {
                                Some(NativeValue::NativeMethod(method)) => Some(method),
                                _ => None,
                            });
                    let Some(method) = native_method else {
                        return Err(self.raise_object_type_error(&owner, "is not subscriptable"));
                    };
                    (method.call)(self, owner, CallArgs::new(vec![index], Vec::new()))
                        .map_err(|error| self.record_native_error(error))?
                }
                BuiltinSubscript::Set => {
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
    fn missing_key(
        &mut self,
        subject: &Value<'s>,
        key: Value<'s>,
    ) -> Result<Option<Value<'s>>, String> {
        let Some(class) = self.instance_class(*subject)? else {
            return Ok(None);
        };
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
    fn string_subscript(
        &mut self,
        owner: &Value<'s>,
        index: &Value<'s>,
    ) -> Result<Option<char>, String> {
        // A non-ASCII string is indexed by code point, which scans the UTF-8 up to the index.
        let non_ascii_bytes = string::string_ref(&self.state.heap, *owner)?
            .filter(|text| !text.is_ascii())
            .map(|text| text.byte_len());
        if let Some(bytes) = non_ascii_bytes {
            self.charge_cpu(scan_cost(bytes))?;
        }
        match string::string_index(&self.state.heap, *owner, *index)? {
            string::StringIndex::NotString => Ok(None),
            string::StringIndex::Character(character) => Ok(Some(character)),
            string::StringIndex::NotInteger => {
                let message = format!(
                    "string indices must be integers, not '{}'",
                    self.type_name_of(index)?
                );
                Err(self.raise_exception("TypeError", message))
            }
            string::StringIndex::OutOfRange => {
                Err(self.raise_exception("IndexError", "string index out of range"))
            }
        }
    }

    fn load_builtin_slice(
        &mut self,
        owner: Value<'s>,
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    ) -> Result<Value<'s>, String> {
        let non_ascii_bytes = string::string_ref(&self.state.heap, owner)?
            .filter(|text| !text.is_ascii())
            .map(|text| text.byte_len());
        if let Some(bytes) = non_ascii_bytes {
            // Slicing by code point materializes the characters first.
            self.charge_cpu(scan_cost(bytes))?;
            self.reserve_result(bytes.saturating_mul(std::mem::size_of::<char>()))?;
        }
        let string_slice = string::string_ref(&self.state.heap, owner)?
            .map(|text| select_string_slice(text.as_str(), text.is_ascii(), start, stop, step))
            .transpose()?;
        if let Some((selected, units)) = string_slice {
            self.charge_cpu(units)?;
            return self.allocate_string(selected);
        }
        if let Some(bytes) = string::bytes_value(&self.state.heap, owner)? {
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
        if owner.is_object() {
            let (length, tuple) = match self.get(owner)? {
                Object::List(values) => (values.len(), false),
                Object::Tuple(values) => (values.len(), true),
                _ => return Err("object is not sliceable".into()),
            };
            let plan = SlicePlan::new(length, start, stop, step)?;
            self.charge_cpu(u64::try_from(plan.len()).unwrap_or(u64::MAX))?;
            let (Object::List(values) | Object::Tuple(values)) = self.get(owner)? else {
                unreachable!("sequence kind was checked above")
            };
            let selected = plan
                .indices()
                .map(|index| self.handle(&values[index]))
                .collect::<Vec<_>>();
            return self.alloc_with(|b| {
                let selected = b.refs(selected);
                if tuple {
                    Object::Tuple(selected)
                } else {
                    Object::List(selected)
                }
            });
        }
        Err("object is not sliceable".into())
    }

    /// `value` as a machine integer, read through `__index__` as CPython's `PyNumber_Index` reads
    /// sequence indices and slice bounds; NumPy integer scalars and 0-d integer arrays qualify.
    /// `None` means `value` is not an integer and its type defines no `__index__`.
    fn index_value(&mut self, value: &Value<'s>) -> Result<Option<i64>, String> {
        if let Some(index) = number::int_value(&self.state.heap, *value) {
            return Ok(Some(index));
        }
        let Some(result) = self.int_by_method(value, &["__index__"])? else {
            return Ok(None);
        };
        match number::int_value(&self.state.heap, result) {
            Some(index) => Ok(Some(index)),
            None => {
                Err(self
                    .raise_exception("IndexError", "cannot fit 'int' into an index-sized integer"))
            }
        }
    }

    /// A builtin sequence's subscript with a non-int index, such as a NumPy integer, replaced by
    /// the int its `__index__` returns. Other owners and indices are returned unchanged.
    fn sequence_index(&mut self, owner: &Value<'s>, index: Value<'s>) -> Result<Value<'s>, String> {
        if !owner.is_object() {
            return Ok(index);
        }
        let sequence = matches!(
            self.get(*owner)?,
            Object::List(_)
                | Object::Tuple(_)
                | Object::Range { .. }
                | Object::String(_)
                | Object::Bytes(_)
                | Object::ByteArray(_)
        );
        if !sequence || number::int_value(&self.state.heap, index).is_some() {
            return Ok(index);
        }
        Ok(match self.index_value(&index)? {
            Some(converted) => Value::Int(converted),
            None => index,
        })
    }

    /// An integer argument read through `__index__`, such as a `range()` bound. Anything else
    /// raises CPython's `TypeError`.
    pub(super) fn index_argument(&mut self, value: &Value<'s>) -> Result<i64, String> {
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
    fn conversion_method(
        &mut self,
        value: &Value<'s>,
        method: &str,
    ) -> Result<Option<Value<'s>>, String> {
        let builtin = self.registered_kind(value).is_none()
            && (super::number::view(&self.state.heap, value).is_some()
                || string::string_value(&self.state.heap, *value)?.is_some()
                || string::bytes_value(&self.state.heap, *value)?.is_some());
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
        value: &Value<'s>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
        let Some(class) = self.instance_class(*value)? else {
            return self.resolve_attribute(*value, name);
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
        value: &Value<'s>,
        methods: &[&'static str],
    ) -> Result<Option<Value<'s>>, String> {
        for &method in methods {
            let Some(result) = self.conversion_method(value, method)? else {
                continue;
            };
            if number::int_value(&self.state.heap, result).is_none() && !self.is_bigint(&result)? {
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
    fn truncate_to_int(&mut self, value: &Value<'s>) -> Result<Value<'s>, String> {
        use super::number::NumberRef;
        let number = super::number::view(&self.state.heap, value);
        let float = match number {
            Some(NumberRef::Float(float)) => float,
            Some(number @ (NumberRef::Int(_) | NumberRef::BigInt(_) | NumberRef::UInt(_))) => {
                return self
                    .new_bigint(number.to_bigint().expect("integer view"))
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
        let integer = <num_bigint::BigInt as num_traits::FromPrimitive>::from_f64(float.trunc())
            .ok_or("float is not finite")?;
        self.new_bigint(integer)
            .map_err(|error| self.record_native_error(error))
    }

    /// The indices a slice selects with, when `value` is a slice: each bound goes through
    /// `__index__` (`None` stays open). Integers beyond the machine range clamp to it, as
    /// CPython's `_PyEval_SliceIndex` does, since no sequence is that long.
    pub(super) fn slice_bounds(
        &mut self,
        value: &Value<'s>,
    ) -> Result<Option<super::super::slice::SliceBounds>, String> {
        if !value.is_object() {
            return Ok(None);
        }
        let Object::Slice { start, stop, step } = self.get(*value)? else {
            return Ok(None);
        };
        let (start, stop, step) = (self.handle(start), self.handle(stop), self.handle(step));
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

    fn slice_index(&mut self, bound: &Value<'s>) -> Result<Option<i64>, String> {
        if bound.is_none() {
            return Ok(None);
        }
        let integer =
            if number::int_value(&self.state.heap, *bound).is_some() || self.is_bigint(bound)? {
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
        if let Some(index) = number::int_value(&self.state.heap, integer) {
            return Ok(Some(index));
        }
        let negative = matches!(
            super::number::view(&self.state.heap, &integer),
            Some(super::number::NumberRef::BigInt(value)) if num_traits::Signed::is_negative(value)
        );
        Ok(Some(if negative { -i64::MAX } else { i64::MAX }))
    }

    /// Pop one slice bound; an omitted bound is `None`.
    fn pop_slice_bound(&mut self, present: bool) -> Result<Value<'s>, String> {
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
        let value = self.alloc_with(|b| Object::Slice {
            start: b.store(start),
            stop: b.store(stop),
            step: b.store(step),
        })?;
        self.push(value);
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
        if !owner.is_object() {
            return Err(self.raise_object_type_error(&owner, "does not support item assignment"));
        }
        let index = self.sequence_index(&owner, index)?;
        match self.get(owner)? {
            Object::List(values) => {
                let Some(index) = number::int_value(&self.state.heap, index) else {
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
                self.modify(owner, |b, object| {
                    let Object::List(values) = object else {
                        unreachable!()
                    };
                    values[index] = b.store(value);
                })?;
            }
            Object::Dict(_) | Object::DefaultDict { .. } => {
                self.dict_set_entry(owner, index, value)?
            }
            // Item assignment on a bytearray would be a shellsim gap, not a CPython TypeError.
            Object::ByteArray(_) => return Err("object does not support item assignment".into()),
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
        let defaults = self
            .execution
            .stack
            .split_off(&self.state.heap, defaults_start);
        let mut closure = self.lookup_scope();
        let class_scope = self.handle_optional(self.class_scopes.last());
        if let (Some(scope), Some(class_scope)) = (closure, class_scope) {
            if self.identical(scope, class_scope) {
                closure = scopes::parent(self.heap(), scope)?;
            }
        }
        let function = self.alloc_with(|b| {
            Object::Function(Box::new(FunctionObject {
                name: name.clone(),
                code,
                closure: b.optional(closure),
                defaults: b.refs(defaults),
                defining_class: None,
                attributes: HashMap::new(),
            }))
        })?;
        self.push(function);
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
        let explicit_metaclass = if has_metaclass {
            Some(self.pop()?)
        } else {
            None
        };
        let bases_start = self.stack.len() - base_count;
        let bases = self
            .execution
            .stack
            .split_off(&self.state.heap, bases_start);
        let is_enum = bases.iter().any(|base| {
            matches!(
                base.native_value(),
                Some(NativeValue::BuiltinType(BuiltinType::Enum))
            )
        });
        let is_unittest = bases.iter().any(|base| {
            matches!(
                base.native_value(),
                Some(NativeValue::BuiltinType(BuiltinType::TestCase))
            )
        });
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
        let user_bases = if is_named_tuple {
            Vec::new()
        } else {
            bases
                .iter()
                .filter_map(|base| {
                    if base.is_object() {
                        if matches!(self.get(*base), Ok(Object::Class { .. })) {
                            Some(Ok(*base))
                        } else {
                            Some(Err("class bases must be classes".to_string()))
                        }
                    } else {
                        match base.native_value() {
                            Some(NativeValue::BuiltinType(
                                BuiltinType::Object
                                | BuiltinType::Type
                                | BuiltinType::Enum
                                | BuiltinType::TestCase,
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
                .collect::<Result<Vec<_>, _>>()
                .map_err(|message| self.raise_exception("TypeError", message))?
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
            if let Object::Class(class_object) = self.get(*base)? {
                if let ClassLayout::Builtin(builtin) = class_object.layout {
                    builtin_layouts.push(builtin);
                }
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
            .filter_map(|base| match self.get(*base) {
                Ok(Object::Class(class_object)) => class_object.exception_base,
                _ => None,
            })
            .collect::<Vec<_>>();
        let exception_base = direct_exception_bases
            .iter()
            .chain(inherited_exception_bases.iter())
            .copied()
            .next();
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
            .filter_map(|base| match self.get(*base) {
                Ok(Object::Class(class_object)) => Some(class_object.layout == ClassLayout::Type),
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
            let Object::Class(class_object) = self.get(*base)? else {
                unreachable!()
            };
            let base_metaclass = self.handle(&class_object.metaclass);
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
        ) || metaclass.is_object()
            && matches!(
                self.get(metaclass),
                Ok(Object::Class(class_object)) if class_object.layout == ClassLayout::Type
            );
        if !valid_metaclass {
            return Err("metaclass must derive from type".into());
        }
        let mro = self.linearize_bases(&user_bases)?;
        let mut prepared_namespace = HashMap::new();
        if metaclass.is_object() {
            if let Some((owner, prepare)) = self.class_attribute_entry(metaclass, "__prepare__")? {
                let prepare = self.bind_descriptor(prepare, None, metaclass, owner)?;
                let bases_value = self.alloc_with(|b| Object::Tuple(b.refs(bases.clone())))?;
                let class_name = self.allocate_string(name.clone())?;
                let namespace = self.invoke_value(prepare, vec![class_name, bases_value])?;
                if !namespace.is_object() {
                    return Err("metaclass __prepare__() must return a mapping".into());
                }
                let Object::Dict(entries) = self.get(namespace)? else {
                    return Err("metaclass __prepare__() must return a dict in this slice".into());
                };
                let entries = entries
                    .iter()
                    .map(|(key, value)| (self.handle(key), self.handle(value)))
                    .collect::<Vec<_>>();
                for (key, value) in entries {
                    let Some(key) = string::string_value(&self.state.heap, key)? else {
                        return Err("metaclass namespace keys must be strings".into());
                    };
                    prepared_namespace.insert(key, value);
                }
            }
        }
        // The class body starts with `__module__` bound to the defining module's `__name__`, as
        // CPython's compiler arranges, so a metaclass `__new__` running in another module does
        // not relabel the class.
        if !prepared_namespace.contains_key("__module__") {
            if let Some(module) = self.current_module_name()? {
                prepared_namespace.insert("__module__".into(), module);
            }
        }
        let parent = self.lookup_scope();
        let uses_repl_globals = parent
            .map(|scope| scopes::uses_repl_globals(self.heap(), scope))
            .transpose()?
            .unwrap_or(true);
        let scope =
            self.alloc_scope_named(parent, uses_repl_globals, Arc::from([]), prepared_namespace)?;
        let stored = self.store(scope);
        self.class_scopes.push(stored);
        self.class_bindings.push(Vec::new());
        let execution = self.execute_code(code, FrameEntry::scoped(scope));
        let bindings = self
            .class_bindings
            .pop()
            .expect("class binding stack is present");
        self.class_scopes.pop();
        match execution {
            Ok(Flow::Halt) => {}
            Ok(Flow::Return(_)) => return Err("'return' outside function".into()),
            Ok(Flow::Yield(_)) => return Err("'yield' outside function".into()),
            Ok(Flow::Exit(status)) => {
                return Err(format!("class body exited with status {status}"))
            }
            Ok(flow) => unreachable!("a class body cannot end with {flow:?}"),
            Err((error, span)) => {
                return Err(format!(
                    "{error} in class {name} at line {}, column {}",
                    span.line, span.column
                ))
            }
        }
        let mut attributes = scopes::values(self.heap(), scope)?;
        if !attributes.contains_key("__doc__") {
            let docstring = match code.docstring.clone() {
                Some(text) => self.allocate_string(text.to_string())?,
                None => Value::None,
            };
            attributes.insert("__doc__".into(), docstring);
        }
        if is_named_tuple {
            let class = self.named_tuple_class(&name, fields, &bindings, attributes)?;
            self.push(class);
            return Ok(());
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
                if value.is_object() && matches!(self.get(value), Ok(Object::Function { .. })) {
                    continue;
                }
                enum_members.push((member_name, value));
            }
            // An enum that mixes in a data type keeps Enum's repr and str, which the data type
            // would otherwise shadow in the MRO. CPython's EnumType.__new__ makes the same
            // substitution for methods the class body does not define.
            if matches!(layout, ClassLayout::Builtin(_)) {
                for (name, slot) in [("__repr__", Slot::Repr), ("__str__", Slot::String)] {
                    attributes.entry(name.into()).or_insert(Value::Native(
                        NativeValue::SlotWrapper {
                            owner: BuiltinType::Enum.id(),
                            slot,
                        },
                    ));
                }
            }
        }
        let descriptor_candidates = attributes
            .iter()
            .map(|(name, value)| (name.clone(), *value))
            .collect::<Vec<_>>();
        let class = if metaclass.is_object() {
            if let Some((owner, constructor)) = self.class_attribute_entry(metaclass, "__new__")? {
                let constructor =
                    self.bind_descriptor(constructor, Some(metaclass), metaclass, owner)?;
                let class_name = self.allocate_string(name.clone())?;
                let bases_value = self.alloc_with(|b| Object::Tuple(b.refs(bases.clone())))?;
                let mut namespace_entries = Vec::with_capacity(descriptor_candidates.len());
                for (attribute_name, value) in &descriptor_candidates {
                    namespace_entries.push((self.allocate_string(attribute_name.clone())?, *value));
                }
                let namespace = self.allocate_dict(namespace_entries)?;
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
        if !class.is_object() {
            return Err("metaclass __new__() must return a class".into());
        }
        if !matches!(self.get(class)?, Object::Class { .. }) {
            return Err("metaclass __new__() must return a class in this slice".into());
        }
        if let Some(base) = bases.first() {
            if let Some(base_type) = self.class_type_id(base)? {
                if let Some((defining_type, initializer)) =
                    self.type_lookup(base_type, "__init_subclass__")?
                {
                    let child_type = self
                        .class_type_id(&class)?
                        .ok_or("class has no registered type")?;
                    let initializer = self
                        .bind_type_attribute(initializer, Some(class), child_type, defining_type)?
                        .ok_or("__init_subclass__ descriptor has no value")?;
                    self.invoke_value(initializer, Vec::new())?;
                }
            }
        }
        if metaclass.is_object() {
            if let Some((owner, initializer)) = self.class_attribute_entry(metaclass, "__init__")? {
                let initializer =
                    self.bind_descriptor(initializer, Some(class), metaclass, owner)?;
                let bases = self.alloc_with(|b| Object::Tuple(b.refs(bases)))?;
                let mut entries = Vec::with_capacity(descriptor_candidates.len());
                for (name, value) in descriptor_candidates {
                    entries.push((self.allocate_string(name)?, value));
                }
                let namespace = self.allocate_dict(entries)?;
                let class_name = self.allocate_string(name.clone())?;
                let result = self.invoke_value(initializer, vec![class_name, bases, namespace])?;
                if !result.is_none() {
                    return Err("metaclass __init__() should return None".into());
                }
            }
        }
        self.push(class);
        Ok(())
    }

    /// `class Name(typing.NamedTuple)`: `collections._namedtuple_from_class` builds the class
    /// from the annotated fields and the class body's namespace, both in definition order.
    fn named_tuple_class(
        &mut self,
        name: &str,
        fields: &[ClassField],
        bindings: &[String],
        mut attributes: HashMap<String, Value<'s>>,
    ) -> Result<Value<'s>, String> {
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
        let field_names = self.alloc_with(|b| Object::Tuple(b.refs(field_names)))?;
        let namespace = self.allocate_dict(entries)?;
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
    fn class_metadata(
        &mut self,
        class: Value<'s>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
        let items = match name {
            "__bases__" => {
                let Object::Class(class_object) = self.get(class)? else {
                    return Err("class metadata requested for a non-class".into());
                };
                let bases = &class_object.bases;
                if bases.is_empty() {
                    vec![Value::Native(NativeValue::BuiltinType(BuiltinType::Object))]
                } else {
                    self.handles(bases)
                }
            }
            "__mro__" => self.class_mro(class)?,
            _ => return Ok(None),
        };
        self.alloc_with(|b| Object::Tuple(b.refs(items))).map(Some)
    }

    /// A user class's method resolution order: the class and its C3-linearized user ancestors,
    /// then the native types their bases derive from, ending with `object`.
    fn class_mro(&self, class: Value<'s>) -> Result<Vec<Value<'s>>, String> {
        let Object::Class(class_object) = self.get(class)? else {
            return Err("class metadata requested for a non-class".into());
        };
        let instance_type = class_object.instance_type;
        let ty = self.state.types.get(instance_type)?;
        std::iter::once(instance_type)
            .chain(ty.mro.iter().copied())
            .map(|id| self.type_value(id))
            .collect()
    }

    /// The MRO of a native type, starting with the type itself.
    fn native_mro(&self, native: Value<'s>) -> Result<Vec<Value<'s>>, String> {
        let registered = match native.native_value() {
            Some(NativeValue::BuiltinType(builtin)) => Some(builtin.id()),
            Some(NativeValue::ValueKind(kind)) => self.state.types.value_kind_type_id(kind),
            Some(NativeValue::ExceptionType(ExceptionType(name))) => {
                self.state.types.exception_type_id(name)
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
            let value = self.type_value(*ancestor)?;
            if !order.iter().any(|known| self.identical(*known, value)) {
                order.push(value);
            }
        }
        Ok(order)
    }

    /// Fixed namespace and type metadata of builtin types, native value kinds and exception
    /// classes, plus the defining module of builtin and native functions.
    fn native_type_metadata(
        &mut self,
        owner: Value<'s>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
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
                self.alloc_with(|b| Object::Tuple(b.refs(order))).map(Some)
            }
            "__bases__" if is_type => {
                let type_id = match native {
                    NativeValue::BuiltinType(builtin) => builtin.id(),
                    NativeValue::ValueKind(kind) => self
                        .state
                        .types
                        .value_kind_type_id(kind)
                        .ok_or("value kind is not registered")?,
                    NativeValue::ExceptionType(ExceptionType(name)) => self
                        .state
                        .types
                        .exception_type_id(name)
                        .ok_or("exception type is not registered")?,
                    _ => unreachable!("checked by is_type"),
                };
                let bases = self
                    .state
                    .types
                    .get(type_id)?
                    .bases
                    .iter()
                    .map(|base| self.type_value(*base))
                    .collect::<Result<Vec<_>, _>>()?;
                self.alloc_with(|b| Object::Tuple(b.refs(bases))).map(Some)
            }
            _ => Ok(None),
        }
    }

    /// The `__name__` global of the module whose code is running.
    /// The namespace a module object's attributes live in. The `__main__` module's scope defers
    /// to the script's global table, so its attributes are the script's globals.
    pub(super) fn module_namespace(
        &self,
        scope: Value<'s>,
    ) -> Result<super::namespace::NamespaceHandle<'s>, String> {
        if scopes::uses_repl_globals(self.heap(), scope)? {
            Ok(super::namespace::NamespaceHandle::Repl)
        } else {
            Ok(super::namespace::NamespaceHandle::Scope(scope))
        }
    }

    fn current_module_name(&mut self) -> Result<Option<Value<'s>>, String> {
        let scope = self.lookup_scope();
        self.module_name_of(scope)
    }

    /// The `__name__` global of the module that owns `scope`; `None` is the main script's
    /// global namespace.
    pub(super) fn module_name_of(
        &mut self,
        scope: Option<Value<'s>>,
    ) -> Result<Option<Value<'s>>, String> {
        if let Some(scope) = scope {
            let root = scopes::root(self.heap(), scope)?;
            if !scopes::uses_repl_globals(self.heap(), root)? {
                return self.scope_get(root, "__name__");
            }
        }
        let symbol = self.intern_symbol("__name__")?;
        Ok(self.state.globals.get(&self.state.heap, symbol))
    }

    pub(super) fn allocate_class(
        &mut self,
        definition: ClassDefinition<'s>,
    ) -> Result<Value<'s>, String> {
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
        // A heap class introduces the instance-dictionary descriptor when no heap ancestor
        // already supplies it. Metaclasses inherit type's own descriptor instead.
        if mro.is_empty() && layout != ClassLayout::Type && !attributes.contains_key("__dict__") {
            attributes.insert(
                "__dict__".into(),
                Value::Native(NativeValue::NativeGetter(
                    &super::super::stdlib::core::INSTANCE_DICT_GETTER,
                )),
            );
        }
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
            let base = self.class_type_id(base)?;
            if let Some(base) = base {
                type_bases.push(base);
            }
        }
        if type_bases.is_empty() {
            type_bases.push(BuiltinType::Object.id());
        }
        let entries = type_bases.iter().try_fold(0usize, |total, base| {
            total
                .checked_add(self.state.types.get(*base)?.mro.len().saturating_add(1))
                .ok_or_else(|| "class MRO is too large".to_string())
        })?;
        self.charge_cpu(
            u64::try_from(entries.saturating_mul(type_bases.len())).unwrap_or(u64::MAX),
        )?;
        let type_mro = self.state.types.linearize_bases(&type_bases)?;
        self.class_type_id(&metaclass)?
            .ok_or("metaclass must be a type")?;
        let descriptors = attributes
            .iter()
            .map(|(name, value)| (name.clone(), *value))
            .collect::<Vec<_>>();
        // The type registry reads slot methods from the class's stored attributes, so the class
        // object exists first and learns its type once the registry has assigned one.
        let class = self.alloc_with(|b| {
            Object::Class(Box::new(ClassObject {
                instance_type: BuiltinType::Object.id(),
                name: name.clone(),
                bases: b.refs(bases),
                mro: b.refs(mro),
                metaclass: b.store(metaclass),
                layout,
                exception_base,
                attributes: b.named(attributes),
                is_dataclass: false,
                dataclass_fields: dataclass_fields
                    .into_iter()
                    .map(|(field, default)| (field, b.optional(default)))
                    .collect(),
                enum_members: Vec::new(),
            }))
        })?;
        let state = &mut *self.state;
        let Object::Class(class_object) = state.heap.get(class)? else {
            unreachable!("allocated a class")
        };
        let instance_type =
            state
                .types
                .register(name, type_bases, type_mro, &class_object.attributes)?;
        let Object::Class(class_object) = self.get_mut(class)? else {
            unreachable!("allocated a class")
        };
        class_object.instance_type = instance_type;
        let stored = self.store(class);
        self.state.types.finish(instance_type, stored)?;
        // Members are instances of the class, so they exist once the class has its type. Their
        // name and value are instance attributes, as in CPython; a data mixin also gives the
        // member the value as its payload so the mixin's methods act on it.
        for (member_name, value) in enum_members {
            let payload = match layout {
                ClassLayout::Builtin(_) => self.state.heap.copy_builtin_payload(value)?,
                _ => Object::Bare,
            };
            let member = self.allocate_typed(instance_type, payload)?;
            let name_value = self.allocate_string(member_name.clone())?;
            self.insert_attribute(member, "_name_", name_value)?;
            self.insert_attribute(member, "_value_", value)?;
            self.modify(class, |b, object| {
                let Object::Class(class_object) = object else {
                    unreachable!("allocated a class");
                };
                class_object
                    .attributes
                    .insert(member_name.clone(), b.store(member));
                class_object.enum_members.push(b.store(member));
            })?;
        }
        for (_, descriptor) in &descriptors {
            if !descriptor.is_object() {
                continue;
            }
            self.modify(*descriptor, |b, object| {
                if let Object::Function(function_object) = object {
                    let FunctionObject { defining_class, .. } = &mut **function_object;
                    *defining_class = Some(b.store(class));
                }
            })?;
        }
        for (attribute_name, descriptor) in descriptors {
            if !descriptor.is_object() {
                continue;
            }
            let Some(descriptor_class) = self.instance_class(descriptor)? else {
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
        bases: &[Value<'s>],
    ) -> Result<Vec<Value<'s>>, String> {
        for (index, base) in bases.iter().enumerate() {
            if bases[..index]
                .iter()
                .any(|earlier| self.identical(*earlier, *base))
            {
                return Err("duplicate base class".into());
            }
        }
        let mut sequences = Vec::with_capacity(bases.len().saturating_add(1));
        for base in bases {
            let Object::Class(class_object) = self.get(*base)? else {
                return Err("class base changed object kind".into());
            };
            let mro = &class_object.mro;
            let mut sequence = Vec::with_capacity(mro.len().saturating_add(1));
            sequence.push(*base);
            sequence.extend(self.handles(mro));
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
                    .any(|other| other.iter().skip(1).any(|item| self.identical(*item, head))))
                .then_some(head)
            });
            let Some(candidate) = candidate else {
                return Err("cannot create a consistent method resolution order".into());
            };
            self.charge_cpu(u64::try_from(sequences.len()).unwrap_or(u64::MAX))?;
            result.push(candidate);
            for sequence in &mut sequences {
                if sequence
                    .first()
                    .is_some_and(|first| self.identical(*first, candidate))
                {
                    sequence.remove(0);
                }
            }
        }
    }

    pub(super) fn class_attribute(
        &mut self,
        class: Value<'s>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
        Ok(self
            .class_attribute_entry(class, name)?
            .map(|(_, value)| value))
    }

    pub(super) fn class_attribute_entry(
        &mut self,
        class: Value<'s>,
        name: &str,
    ) -> Result<Option<(Value<'s>, Value<'s>)>, String> {
        let Object::Class(class_object) = self.get(class)? else {
            return Err("instance has an invalid class".into());
        };
        let instance_type = class_object.instance_type;
        let ancestors = std::iter::once(instance_type)
            .chain(self.state.types.get(instance_type)?.mro.iter().copied())
            .collect::<Vec<_>>();
        for ancestor in ancestors {
            self.charge_cpu(1)?;
            let Some(value) = self.type_namespace_attribute(ancestor, name)? else {
                continue;
            };
            let owner = self.type_value(ancestor)?;
            if owner.is_object() && matches!(self.get(owner)?, Object::Class { .. }) {
                return Ok(Some((owner, value)));
            }
            // This helper returns heap-owned definitions only. A native definition before a
            // later user class still wins and must be handled by the caller's native path.
            return Ok(None);
        }
        Ok(None)
    }

    /// Search each type's own namespace in MRO order. User namespaces remain on their heap
    /// class objects, while builtin namespaces remain in the registry.
    pub(super) fn type_lookup(
        &mut self,
        type_id: TypeId,
        name: &str,
    ) -> Result<Option<(TypeId, Value<'s>)>, String> {
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
    ) -> Result<Option<Value<'s>>, String> {
        let ty = self.state.types.get(type_id)?;
        let value = self.type_value(type_id)?;
        if value.is_object() {
            if let Object::Class(class_object) = self.get(value)? {
                return Ok(self.handle_optional(class_object.attributes.get(name)));
            }
        }
        Ok(self.handle_optional(ty.attributes.get(name)))
    }

    /// Bind a descriptor found by `type_lookup` to an instance or leave it unbound for class
    /// access. Native descriptors have no heap class object, so they are handled here.
    pub(super) fn bind_type_attribute(
        &mut self,
        descriptor: Value<'s>,
        receiver: Option<Value<'s>>,
        accessed_type: TypeId,
        defining_type: TypeId,
    ) -> Result<Option<Value<'s>>, String> {
        match descriptor.native_value() {
            Some(NativeValue::NativeGetter(getter)) => {
                return match receiver {
                    Some(receiver) => self.call_native_getter(getter, receiver),
                    None => Ok(Some(descriptor)),
                };
            }
            Some(NativeValue::NativeClassMethod(method)) => {
                let class = self.type_value(accessed_type)?;
                return self.bind_native_class_method(class, method).map(Some);
            }
            Some(NativeValue::NativeMethod(method)) if method.name == "__new__" => {
                return Ok(Some(descriptor));
            }
            Some(NativeValue::NativeMethod(_)) => {
                return match receiver {
                    Some(receiver) => self
                        .alloc_with(|b| Object::DescriptorBoundMethod {
                            receiver: b.store(receiver),
                            descriptor: b.store(descriptor),
                            owner: None,
                        })
                        .map(Some),
                    None => Ok(Some(descriptor)),
                };
            }
            Some(NativeValue::SlotWrapper { .. }) => {
                return match receiver {
                    Some(receiver) => self
                        .alloc_with(|b| Object::DescriptorBoundMethod {
                            receiver: b.store(receiver),
                            descriptor: b.store(descriptor),
                            owner: None,
                        })
                        .map(Some),
                    None => Ok(Some(descriptor)),
                };
            }
            _ => {}
        }
        let defining_class = self.type_value(defining_type)?;
        if !defining_class.is_object() {
            return Ok(Some(descriptor));
        }
        let accessed_class = self.type_value(accessed_type)?;
        if !accessed_class.is_object() {
            return Err("user descriptor has no class receiver".into());
        }
        self.bind_descriptor(descriptor, receiver, accessed_class, defining_class)
            .map(Some)
    }

    fn call_native_getter(
        &mut self,
        getter: &'static super::super::native::GetterDef,
        receiver: Value<'s>,
    ) -> Result<Option<Value<'s>>, String> {
        match (getter.get)(self, receiver) {
            Ok(value) => Ok(Some(value)),
            Err(error) => Err(self.record_native_error(error)),
        }
    }

    fn is_data_descriptor(&mut self, value: &Value<'s>) -> Result<bool, String> {
        if matches!(value.native_value(), Some(NativeValue::NativeGetter(_))) {
            return Ok(true);
        }
        if !value.is_object() {
            return Ok(false);
        }
        if let Some(class) = self.instance_class(*value)? {
            return Ok(self.class_attribute(class, "__set__")?.is_some()
                || self.class_attribute(class, "__delete__")?.is_some());
        }
        Ok(matches!(self.get(*value)?, Object::Property { .. }))
    }

    /// Bind a native class method to `class`, which the method receives in place of an instance.
    fn bind_native_class_method(
        &mut self,
        class: Value<'s>,
        method: &'static super::super::native::MethodDef,
    ) -> Result<Value<'s>, String> {
        self.alloc_with(|b| Object::DescriptorBoundMethod {
            receiver: b.store(class),
            descriptor: b.store(Value::Native(NativeValue::NativeMethod(method))),
            owner: None,
        })
    }

    pub(super) fn bind_descriptor(
        &mut self,
        descriptor: Value<'s>,
        receiver: Option<Value<'s>>,
        accessed_class: Value<'s>,
        defining_class: Value<'s>,
    ) -> Result<Value<'s>, String> {
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
                Some(receiver) => self.alloc_with(|b| Object::DescriptorBoundMethod {
                    receiver: b.store(receiver),
                    descriptor: b.store(descriptor),
                    owner: Some(b.store(defining_class)),
                }),
                None => Ok(descriptor),
            };
        }
        if !descriptor.is_object() {
            return Ok(descriptor);
        }
        let descriptor_class = self.instance_class(descriptor)?;
        match self.get(descriptor)? {
            Object::Function { .. } => match receiver {
                Some(receiver) => self.alloc_with(|b| Object::DescriptorBoundMethod {
                    receiver: b.store(receiver),
                    descriptor: b.store(descriptor),
                    owner: Some(b.store(defining_class)),
                }),
                None => Ok(descriptor),
            },
            Object::Property { getter, .. } => match receiver {
                Some(receiver) => {
                    let getter = self.handle(getter);
                    self.invoke_value(getter, vec![receiver])
                }
                None => Ok(descriptor),
            },
            Object::StaticMethod { callable } => Ok(self.handle(callable)),
            Object::ClassMethod { callable } => {
                let callable = self.handle(callable);
                if callable.is_object() && matches!(self.get(callable)?, Object::Function { .. }) {
                    return self.alloc_with(|b| Object::DescriptorBoundMethod {
                        receiver: b.store(accessed_class),
                        descriptor: b.store(callable),
                        owner: Some(b.store(defining_class)),
                    });
                }
                Ok(callable)
            }
            _ if descriptor_class.is_some() => {
                let class = descriptor_class.expect("checked above");
                let Some((get_owner, get)) = self.class_attribute_entry(class, "__get__")? else {
                    return Ok(descriptor);
                };
                let get = self.bind_descriptor(get, Some(descriptor), class, get_owner)?;
                self.invoke_value(get, vec![receiver.unwrap_or(Value::None), accessed_class])
            }
            _ => Ok(descriptor),
        }
    }

    pub(super) fn invoke_value(
        &mut self,
        callable: Value<'s>,
        arguments: Vec<Value<'s>>,
    ) -> Result<Value<'s>, String> {
        let result = self.invoke_call(callable, arguments, Vec::new());
        self.immediate_call_value(result, "callable")
    }

    pub(super) fn invoke_call(
        &mut self,
        callable: Value<'s>,
        arguments: Vec<Value<'s>>,
        keyword_arguments: Vec<(String, Value<'s>)>,
    ) -> Result<Flow, String> {
        let positional = arguments.len();
        let total = positional
            .checked_add(keyword_arguments.len())
            .ok_or("too many call arguments")?;
        let keyword_names = keyword_arguments
            .iter()
            .map(|(name, _)| Some(name.clone()))
            .collect::<Vec<_>>();
        self.push(callable);
        for argument in arguments {
            self.push(argument);
        }
        for (_, value) in keyword_arguments {
            self.push(value);
        }
        self.call(
            positional,
            &keyword_names,
            &vec![false; total],
            CallMode::Immediate,
        )
    }

    pub(super) fn invoke_slot(
        &mut self,
        receiver: &Value<'s>,
        slot: Slot,
        method_name: &str,
        arguments: Vec<Value<'s>>,
    ) -> Result<Option<Value<'s>>, String> {
        let type_id = self.type_id(receiver)?;
        let Some(slot_value) = self.state.types.slot(type_id, slot)? else {
            return Ok(None);
        };
        // A native slot implements a builtin type's behavior, which an instance of a builtin
        // subclass, as receiver or operand, takes part in through the value it holds.
        let slot_descriptor = match slot_value {
            SlotValue::NativeCompare(call) => {
                let [argument] = arguments.as_slice() else {
                    return Err("comparison slot received the wrong number of arguments".into());
                };
                let operator = ComparisonOperator::from_slot(slot)
                    .ok_or("comparison slot invoked for a non-comparison operator")?;
                let (receiver, argument) = (*receiver, *argument);
                return call(self, receiver, argument, operator)
                    .map(|result| result.map(Value::Bool))
                    .map_err(|error| self.record_native_error(error));
            }
            SlotValue::VmHash => {
                if !arguments.is_empty() {
                    return Err("hash slot received arguments".into());
                }
                let receiver = *receiver;
                let hash = self.payload_hash(&receiver, 0)?;
                return Ok(Some(Value::Int(hash)));
            }
            SlotValue::VmRepr => {
                if !arguments.is_empty() {
                    return Err("representation slot received arguments".into());
                }
                let rendered = self.repr_payload(receiver, &mut BTreeSet::new())?;
                return self.allocate_string(rendered).map(Some);
            }
            SlotValue::NativeMethod(method) => {
                let receiver = *receiver;
                return (method.call)(self, receiver, CallArgs::new(arguments, Vec::new()))
                    .map(Some)
                    .map_err(|error| self.record_native_error(error));
            }
            SlotValue::NativeBinary(call) => {
                let [argument] = arguments.as_slice() else {
                    return Err(
                        "binary protocol slot received the wrong number of arguments".into(),
                    );
                };
                let (receiver, argument) = (*receiver, *argument);
                return call(self, receiver, argument)
                    .map_err(|error| self.record_native_error(error));
            }
            SlotValue::NativeTernary(call) => {
                let [first, second] = arguments.as_slice() else {
                    return Err(
                        "ternary protocol slot received the wrong number of arguments".into(),
                    );
                };
                let receiver = *receiver;
                let (first, second) = (*first, *second);
                return call(self, receiver, first, second)
                    .map_err(|error| self.record_native_error(error));
            }
            SlotValue::NativeUnary(call) => {
                if !arguments.is_empty() {
                    return Err("unary protocol slot received arguments".into());
                }
                let receiver = *receiver;
                return call(self, receiver).map_err(|error| self.record_native_error(error));
            }
            SlotValue::Descriptor { value, owner } => (self.handle(&value), owner),
        };
        if !receiver.is_object() {
            return Ok(None);
        }
        let (descriptor, owner) = slot_descriptor;
        // A plain function is called with the receiver prepended, which is what binding it
        // and calling the bound method would do, without the bound method.
        if descriptor.is_object() && matches!(self.get(descriptor)?, Object::Function(_)) {
            let positional = arguments.len().saturating_add(1);
            self.push(descriptor);
            self.push(*receiver);
            for argument in arguments {
                self.push(argument);
            }
            let result = self.call(
                positional,
                &[],
                &vec![false; positional],
                CallMode::Immediate,
            );
            return self.immediate_call_value(result, method_name).map(Some);
        }
        let class = match (self.instance_class(*receiver)?, self.get(*receiver)?) {
            (Some(class), _) => class,
            (None, Object::Class(class_object)) if class_object.metaclass.is_object() => {
                self.handle(&class_object.metaclass)
            }
            _ => return Ok(None),
        };
        let defining_class = self.type_value(owner)?;
        let callable = self.bind_descriptor(descriptor, Some(*receiver), class, defining_class)?;
        self.invoke_value(callable, arguments).map(Some)
    }

    /// The value of an immediate call made on behalf of Rust code, which has no frame to exit
    /// from: an exit request becomes an error naming `what` asked for it.
    fn immediate_call_value(
        &mut self,
        result: Result<Flow, String>,
        what: &str,
    ) -> Result<Value<'s>, String> {
        match self.immediate_value(result?)? {
            Ok(value) => Ok(value),
            Err(Flow::Exit(status)) => Err(format!("{what} exited with status {status}")),
            Err(flow) => unreachable!("an immediate call cannot end with {flow:?}"),
        }
    }

    /// Execute the defining type's native slot for an explicit dunder call. The receiver check
    /// and local-slot lookup keep `int.__add__(x, y)` independent of overrides on `type(x)`.
    pub(super) fn call_slot_wrapper(
        &mut self,
        owner: TypeId,
        slot: Slot,
        receiver: Value<'s>,
        arguments: Vec<Value<'s>>,
        keyword_arguments: Vec<(String, Value<'s>)>,
    ) -> Result<Value<'s>, String> {
        let (_, name, arity) = super::super::object_model::SLOT_DEFS[slot as usize];
        if arity != 255 && (!keyword_arguments.is_empty() || arguments.len() != usize::from(arity))
        {
            return Err(
                self.raise_exception("TypeError", format!("{name}() received invalid arguments"))
            );
        }
        let receiver_type = if slot == Slot::ClassGetItem {
            self.class_type_id(&receiver)?
                .ok_or("class subscription receiver has no registered type")?
        } else {
            self.type_id(&receiver)?
        };
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
        // The builtin `__repr__` and `__hash__` act on the receiver's payload: called from a
        // subclass's own `__repr__`, they must not dispatch back into it.
        if matches!(implementation, SlotValue::VmRepr) {
            let rendered = self.repr_payload(&receiver, &mut BTreeSet::new())?;
            return self.allocate_string(rendered);
        }
        if matches!(implementation, SlotValue::VmHash) {
            return self.payload_hash(&receiver, 0).map(Value::Int);
        }
        let result = match implementation {
            SlotValue::VmRepr | SlotValue::VmHash => {
                unreachable!("handled before the native slot dispatch")
            }
            SlotValue::NativeCompare(call) => {
                let operator = ComparisonOperator::from_slot(slot)
                    .ok_or("comparison slot invoked for a non-comparison operator")?;
                let argument = arguments[0];
                call(self, receiver, argument, operator).map(|result| result.map(Value::Bool))
            }
            SlotValue::NativeMethod(method) => {
                return (method.call)(self, receiver, CallArgs::new(arguments, keyword_arguments))
                    .map_err(|error| self.record_native_error(error));
            }
            SlotValue::NativeUnary(call) => call(self, receiver),
            SlotValue::NativeBinary(call) => {
                let argument = arguments[0];
                call(self, receiver, argument)
            }
            SlotValue::NativeTernary(call) => {
                let first = arguments[0];
                let second = arguments[1];
                call(self, receiver, first, second)
            }
            SlotValue::Descriptor { .. } => return Err("slot wrapper is not native".into()),
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
        receiver: &Value<'s>,
        slot: Slot,
        method_name: &str,
        arguments: Vec<Value<'s>>,
    ) -> Result<Option<Value<'s>>, String> {
        Ok(self
            .invoke_slot(receiver, slot, method_name, arguments)?
            .filter(|value| value.native_value() != Some(NativeValue::NotImplemented)))
    }

    pub(super) fn truth_value(&mut self, value: &Value<'s>) -> Result<bool, String> {
        // Immediate bools, ints, floats and None have no slots to consult; every comparison
        // result passes through here, so the slot lookups would dominate a sort.
        if value.bool_value().is_some()
            || value.immediate_int().is_some()
            || value.float_value().is_some()
            || value.is_none()
        {
            return protocol::truth(&self.state.heap, *value);
        }
        if let Some(result) = self.invoke_slot(value, Slot::Bool, "__bool__", Vec::new())? {
            return result
                .bool_value()
                .ok_or_else(|| "__bool__ should return bool".into());
        }
        if let Some(result) = self.invoke_slot(value, Slot::Length, "__len__", Vec::new())? {
            let length = number::int_value(&self.state.heap, result)
                .ok_or_else(|| "__len__ should return int".to_string())?;
            if length < 0 {
                return Err("__len__ should return >= 0".into());
            }
            return Ok(length != 0);
        }
        protocol::truth(&self.state.heap, *value)
    }

    pub(super) fn repr_value(&mut self, value: &Value<'s>) -> Result<String, String> {
        self.repr_nested(value, &mut BTreeSet::new())
    }

    /// Render the base object's identity without consulting the value's `__repr__` slot.
    pub(super) fn default_object_repr(&self, value: &Value<'s>) -> Result<String, String> {
        let mut name = self.type_name_of(value)?;
        if let Some(class) = self.instance_class(*value)? {
            if let Object::Class(class_object) = self.get(class)? {
                if let Some(module) = class_object.attributes.get("__module__") {
                    let module = self.handle(module);
                    if let Some(module) = string::string_value(&self.state.heap, module)? {
                        if module != "builtins" {
                            name = format!("{module}.{name}");
                        }
                    }
                }
            }
        }
        let address = self
            .identity(*value)?
            .map(protocol::address)
            .unwrap_or_else(|| {
                format!(
                    "0x{:x}",
                    super::super::stdlib::core::immediate_identity(value)
                )
            });
        Ok(format!("<{name} object at {address}>"))
    }

    /// `repr()` that renders container items through their own `__repr__` and protocol slots,
    /// as CPython does. `active` holds the containers being rendered, so a container that
    /// contains itself prints as `[...]`.
    fn repr_nested(
        &mut self,
        value: &Value<'s>,
        active: &mut BTreeSet<u32>,
    ) -> Result<String, String> {
        crate::stack::grow(|| self.repr_nested_inner(value, active))
    }

    fn repr_nested_inner(
        &mut self,
        value: &Value<'s>,
        active: &mut BTreeSet<u32>,
    ) -> Result<String, String> {
        if !matches!(
            self.state.types.slot(self.type_id(value)?, Slot::Repr)?,
            Some(SlotValue::VmRepr)
        ) {
            if let Some(result) = self.invoke_slot(value, Slot::Repr, "__repr__", Vec::new())? {
                return string::string_value(&self.state.heap, result)?
                    .ok_or_else(|| "__repr__ should return str".into());
            }
        }
        self.repr_payload(value, active)
    }

    /// The builtin `repr()` of `value`'s payload, ignoring any `__repr__` its class defines.
    /// Items of a container still render through their own classes.
    pub(super) fn repr_payload(
        &mut self,
        value: &Value<'s>,
        active: &mut BTreeSet<u32>,
    ) -> Result<String, String> {
        if let Some(id) = self.identity(*value)? {
            if self.is_plain_instance(value)? {
                return self.default_object_repr(value);
            }
            match self.get(*value)? {
                Object::NamespaceDict(target) => {
                    let target = self.namespace_handle(target);
                    return self.repr_namespace_dict(id, target, active);
                }
                Object::DictView { kind, mapping } => {
                    let (kind, mapping) = (*kind, self.handle(mapping));
                    return self.repr_dict_view(id, kind, mapping, active);
                }
                Object::MappingProxy(target) => {
                    let target = self.proxy_handle(target);
                    return self.repr_mapping_proxy(id, target, active);
                }
                _ => {}
            }
        }
        let Some((id, container)) = self.container_items(value)? else {
            self.check_int_str_digits(value)?;
            return protocol::repr(self.state, *value).map_err(|error| self.render_error(error));
        };
        if active.contains(&id) {
            return Ok(container.placeholder().into());
        }
        // `active` holds the containers on the current path, so its size is the nesting depth.
        if active.len() >= MAX_REPR_DEPTH {
            return Err(self.raise_exception(
                "RecursionError",
                "maximum recursion depth exceeded while getting the repr of an object",
            ));
        }
        active.insert(id);
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
        entries: &[(Value<'s>, Value<'s>)],
        active: &mut BTreeSet<u32>,
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
        id: u32,
        kind: DictViewKind,
        mapping: Value<'s>,
        active: &mut BTreeSet<u32>,
    ) -> Result<String, String> {
        if !active.insert(id) {
            return Ok("...".into());
        }
        let entries = self
            .mapping_items(mapping)?
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
        id: u32,
        target: ProxyHandle<'s>,
        active: &mut BTreeSet<u32>,
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
        id: u32,
        target: NamespaceHandle<'s>,
        active: &mut BTreeSet<u32>,
    ) -> Result<String, String> {
        if !active.insert(id) {
            return Ok("{...}".into());
        }
        let entries = self.namespace_entries(target)?;
        self.charge_cpu(u64::try_from(entries.len()).unwrap_or(u64::MAX))?;
        let mut parts = Vec::with_capacity(entries.len());
        for (name, value) in entries {
            let value = self.repr_nested(&value, active)?;
            parts.push(format!("{}: {value}", string::quote_string(&name)));
        }
        active.remove(&id);
        Ok(format!("{{{}}}", parts.join(", ")))
    }

    fn repr_items(
        &mut self,
        items: &[Value<'s>],
        active: &mut BTreeSet<u32>,
    ) -> Result<String, String> {
        let mut parts = Vec::with_capacity(items.len());
        for item in items {
            parts.push(self.repr_nested(item, active)?);
        }
        Ok(parts.join(", "))
    }

    /// A snapshot of a builtin container's items, for rendering them through the VM.
    fn container_items(
        &self,
        value: &Value<'s>,
    ) -> Result<Option<(u32, ContainerItems<'s>)>, String> {
        let Some(id) = self.identity(*value)? else {
            return Ok(None);
        };
        let items = match self.get(*value)? {
            Object::List(items) => ContainerItems::List(self.handles(items)),
            Object::Tuple(items) => ContainerItems::Tuple(self.handles(items)),
            Object::Set(items) => ContainerItems::Set(self.handles(items.iter())),
            Object::FrozenSet(items) => ContainerItems::FrozenSet(self.handles(items.iter())),
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => ContainerItems::Dict(
                entries
                    .iter()
                    .map(|(key, value)| (self.handle(key), self.handle(value)))
                    .collect(),
            ),
            _ => return Ok(None),
        };
        Ok(Some((id, items)))
    }

    /// Whether `value` is a namespace view, dict view or mapping proxy, which `repr_nested`
    /// renders through the VM because their entries live outside the object.
    fn is_mapping_view(&self, value: &Value<'s>) -> Result<bool, String> {
        if !value.is_object() {
            return Ok(false);
        }
        Ok(matches!(
            self.get(*value)?,
            Object::NamespaceDict(_) | Object::DictView { .. } | Object::MappingProxy(_)
        ))
    }

    pub(super) fn display_value(&mut self, value: &Value<'s>) -> Result<String, String> {
        if let Some(result) = self.invoke_slot(value, Slot::String, "__str__", Vec::new())? {
            return string::string_value(&self.state.heap, result)?
                .ok_or_else(|| "__str__ should return str".into());
        }
        if let Some(text) = string::string_value(&self.state.heap, *value)? {
            return Ok(text);
        }
        if self
            .state
            .types
            .is_subclass(self.type_id(value)?, BuiltinType::Exception.id())?
        {
            return protocol::display(self.state, *value).map_err(|error| self.render_error(error));
        }
        let plain_instance = self.is_plain_instance(value)?;
        if self
            .state
            .types
            .slot(self.type_id(value)?, Slot::Repr)?
            .is_some()
            || plain_instance
            || self.container_items(value)?.is_some()
            || self.is_mapping_view(value)?
        {
            return self.repr_value(value);
        }
        self.check_int_str_digits(value)?;
        protocol::display(self.state, *value).map_err(|error| self.render_error(error))
    }

    /// Turn a rendering failure into the Python exception CPython raises for it: nesting past
    /// the render bound is `RecursionError`; anything else stays an internal error.
    fn render_error(&mut self, error: String) -> String {
        if error.starts_with("maximum recursion depth exceeded") {
            self.raise_exception("RecursionError", error)
        } else {
            error
        }
    }

    /// Raise `ValueError` before rendering an int with more decimal digits than CPython's
    /// default `sys.int_max_str_digits` allows.
    fn check_int_str_digits(&mut self, value: &Value<'s>) -> Result<(), String> {
        if let Some(super::number::NumberRef::BigInt(integer)) =
            super::number::view(&self.state.heap, value)
        {
            if super::number::exceeds_str_digits(integer) {
                return Err(self.record_native_error(super::number::int_str_digits_error()));
            }
        }
        Ok(())
    }

    fn super_attribute(
        &mut self,
        start_class: Value<'s>,
        receiver: &Value<'s>,
        name: &str,
    ) -> Result<(TypeId, Value<'s>, TypeId), String> {
        let start_type = self
            .class_type_id(&start_class)?
            .ok_or("super() start has an invalid class")?;
        if !receiver.is_object() {
            return Err("super() receiver is not an instance or class".into());
        }
        let accessed_type = match self.get(*receiver)? {
            _ if self.instance_class(*receiver)?.is_some() => self.type_id(receiver)?,
            Object::Class { .. } => {
                let class_type = self
                    .class_type_id(receiver)?
                    .ok_or("super() receiver has an invalid class")?;
                if self.state.types.is_subclass(class_type, start_type)? {
                    class_type
                } else {
                    // A method on a metaclass receives a class object. Its `super()` walks
                    // that object's metaclass MRO rather than the class's own MRO.
                    self.type_id(receiver)?
                }
            }
            _ => return Err("super() receiver is not an instance or class".into()),
        };
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
    fn dict_set_entry(
        &mut self,
        dict: Value<'s>,
        key: Value<'s>,
        value: Value<'s>,
    ) -> Result<(), String> {
        let (hash, position) = self.lookup_mapping_entry(dict, &key)?;
        if let Some(position) = position {
            return self.modify(dict, |b, object| {
                let entries = match object {
                    Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
                    _ => unreachable!("dict kind was checked during lookup"),
                };
                entries.set_value(position, b.store(value));
            });
        }
        self.reserve_object_growth(dict, MODELED_MAPPING_ENTRY_BYTES)?;
        self.modify(dict, |b, object| {
            let entries = match object {
                Object::Dict(entries) | Object::DefaultDict { entries, .. } => entries,
                _ => unreachable!("dict kind was checked during lookup"),
            };
            entries.push(hash, (b.store(key), b.store(value)));
        })
    }

    /// `dict(source=(), /, **keywords)`: the entries of a mapping (a dict, or any object with
    /// `keys()` and `__getitem__`) or the pairs of an iterable, followed by the keywords.
    fn construct_dict(
        &mut self,
        arguments: Vec<Value<'s>>,
        keyword_arguments: Vec<(String, Value<'s>)>,
    ) -> Result<Value<'s>, String> {
        if arguments.len() > 1 {
            let message = format!("dict expected at most 1 argument, got {}", arguments.len());
            return Err(self.raise_exception("TypeError", message));
        }
        let dict = self.allocate_object(Object::Dict(Default::default()))?;
        if let Some(source) = arguments.first() {
            for (key, value) in self.dict_source_entries(source)? {
                self.charge_cpu(1)?;
                self.dict_set_entry(dict, key, value)?;
            }
        }
        for (name, value) in keyword_arguments {
            let key = self.allocate_string(name)?;
            self.dict_set_entry(dict, key, value)?;
        }
        Ok(dict)
    }

    fn dict_source_entries(
        &mut self,
        source: &Value<'s>,
    ) -> Result<Vec<(Value<'s>, Value<'s>)>, String> {
        // A dict subclass contributes the entries it holds, as CPython's dict merge does.
        let view = *source;
        if view.is_object() {
            if let Object::Dict(entries) | Object::DefaultDict { entries, .. } = self.get(view)? {
                return Ok(entries
                    .iter()
                    .map(|(key, value)| (self.handle(key), self.handle(value)))
                    .collect());
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

    /// `int(text, base=n)`: move the `base` keyword into the positional slot `int()` parses.
    fn int_base_keyword(
        &mut self,
        mut arguments: Vec<Value<'s>>,
        mut keyword_arguments: Vec<(String, Value<'s>)>,
    ) -> Result<CallArguments<'s>, String> {
        let Some(position) = keyword_arguments
            .iter()
            .position(|(name, _)| name == "base")
        else {
            return Ok((arguments, keyword_arguments));
        };
        if arguments.len() != 1 {
            let message = "int() missing string argument";
            return Err(self.raise_exception("TypeError", message));
        }
        arguments.push(keyword_arguments.remove(position).1);
        Ok((arguments, keyword_arguments))
    }

    pub(super) fn call_builtin_type(
        &mut self,
        builtin_type: BuiltinType,
        arguments: Vec<Value<'s>>,
        keyword_arguments: Vec<(String, Value<'s>)>,
    ) -> Result<Flow, String> {
        if builtin_type == BuiltinType::Dict {
            return self
                .construct_dict(arguments, keyword_arguments)
                .map(|value| self.produce(value));
        }
        if builtin_type == BuiltinType::Complex {
            let arguments = super::CallArgs::new(arguments, keyword_arguments);
            let value = super::super::complex::construct(self, arguments)
                .map_err(|error| self.record_native_error(error))?;
            return Ok(self.produce(value));
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
        let (arguments, keyword_arguments) = if builtin_type == BuiltinType::Int {
            self.int_base_keyword(arguments, keyword_arguments)?
        } else {
            (arguments, keyword_arguments)
        };
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
                    let name = string::string_value(&self.state.heap, *name)?
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
                self.alloc_with(|b| Object::Slice {
                    start: b.store(bounds[0]),
                    stop: b.store(bounds[1]),
                    step: b.store(bounds[2]),
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
                    let base = number::int_value(&self.state.heap, *base).ok_or_else(|| {
                        self.record_native_error(PyError::type_error(
                            "int() base must be an integer",
                        ))
                    })?;
                    let text =
                        string::string_value(&self.state.heap, arguments[0])?.ok_or_else(|| {
                            self.record_native_error(PyError::type_error(
                                "int() can't convert non-string with explicit base",
                            ))
                        })?;
                    self.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
                    let integer = super::number::parse_integer_text(&text, base)
                        .map_err(|error| self.record_native_error(error))?;
                    self.new_bigint(integer)
                        .map_err(|error| self.record_native_error(error))?
                } else {
                    match arguments.first() {
                        None => Value::Int(0),
                        Some(value) if self.is_bigint(value)? => *value,
                        Some(value)
                            if string::string_value(&self.state.heap, *value)?.is_some() =>
                        {
                            let text =
                                string::string_value(&self.state.heap, *value)?.expect("guarded");
                            self.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
                            let integer = super::number::parse_integer_text(&text, 10)
                                .map_err(|error| self.record_native_error(error))?;
                            self.new_bigint(integer)
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
                        if converted.float_value().is_none() {
                            let message = format!(
                                "{}.__float__ returned non-float (type {})",
                                self.type_name_of(&value)?,
                                self.type_name_of(&converted)?
                            );
                            return Err(self.raise_exception("TypeError", message));
                        }
                        return Ok(self.produce(converted));
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
                    Some(value) if string::string_value(&self.state.heap, *value)?.is_some() => {
                        let text =
                            string::string_value(&self.state.heap, *value)?.expect("guarded");
                        match text.trim().parse::<f64>() {
                            Ok(parsed) => parsed,
                            Err(_) => {
                                let message = format!(
                                    "could not convert string to float: {}",
                                    string::quote_string(&text)
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
                    [value] if string::bytes_value(&self.state.heap, *value)?.is_some() => {
                        string::bytes_value(&self.state.heap, *value)?.expect("guarded")
                    }
                    [value] if number::int_value(&self.state.heap, *value).is_some() => {
                        let Ok(length) = usize::try_from(
                            number::int_value(&self.state.heap, *value).expect("guarded"),
                        ) else {
                            return Err(self.raise_exception("ValueError", "negative count"));
                        };
                        self.reserve_result(length)?;
                        vec![0; length]
                    }
                    [value]
                        if builtin_type == BuiltinType::Bytes
                            && self.special_method(value, "__bytes__")?.is_some() =>
                    {
                        let converted = self
                            .conversion_method(value, "__bytes__")?
                            .expect("special method found above");
                        match string::bytes_value(&self.state.heap, converted)? {
                            Some(bytes) => bytes,
                            None => {
                                let message = format!(
                                    "__bytes__ returned non-bytes (type {})",
                                    self.type_name_of(&converted)?
                                );
                                return Err(self.raise_exception("TypeError", message));
                            }
                        }
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
                        let text = string::string_value(&self.state.heap, *value)?
                            .ok_or("encoding without a string argument")?;
                        let encoding = string::string_value(&self.state.heap, *encoding)?
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
                match builtin_type {
                    BuiltinType::List => self.alloc_with(|b| Object::List(b.refs(values)))?,
                    BuiltinType::Tuple => self.alloc_with(|b| Object::Tuple(b.refs(values)))?,
                    BuiltinType::Set | BuiltinType::FrozenSet => {
                        let unique = self.distinct_members(values)?;
                        self.reserve_result(unique.len().saturating_mul(64))?;
                        self.alloc_with(|b| {
                            let unique = unique.into_set(b);
                            if builtin_type == BuiltinType::Set {
                                Object::Set(unique)
                            } else {
                                Object::FrozenSet(unique)
                            }
                        })?
                    }
                    _ => unreachable!(),
                }
            }
            BuiltinType::Dict => unreachable!("dict construction returned above"),
            BuiltinType::Property => {
                expect_arity(&arguments, 0, 2)?;
                let getter = arguments.first().copied().unwrap_or(Value::None);
                let setter = arguments.get(1).copied().filter(|value| !value.is_none());
                self.alloc_with(|builder| Object::Property {
                    getter: builder.store(getter),
                    setter: setter.map(|value| builder.store(value)),
                })?
            }
            BuiltinType::StaticMethod => {
                expect_arity(&arguments, 1, 1)?;
                self.alloc_with(|builder| Object::StaticMethod {
                    callable: builder.store(arguments[0]),
                })?
            }
            BuiltinType::ClassMethod => {
                expect_arity(&arguments, 1, 1)?;
                self.alloc_with(|builder| Object::ClassMethod {
                    callable: builder.store(arguments[0]),
                })?
            }
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
            | BuiltinType::Regex
            | BuiltinType::Match
            | BuiltinType::Array
            | BuiltinType::GenericAlias
            | BuiltinType::Enum
            | BuiltinType::TestCase => {
                return Err(format!("cannot create '{}' instances", builtin_type.name()));
            }
            BuiltinType::Complex => unreachable!("complex construction returned above"),
        };
        Ok(self.produce(value))
    }

    pub(super) fn class_type_id(&self, value: &Value<'s>) -> Result<Option<TypeId>, String> {
        Ok(match value.native_value() {
            Some(NativeValue::BuiltinType(builtin)) => Some(builtin.id()),
            Some(NativeValue::ValueKind(kind)) => self.state.types.value_kind_type_id(kind),
            Some(NativeValue::ExceptionType(ExceptionType(name))) => {
                self.state.types.exception_type_id(name)
            }
            _ if value.is_object() => match self.get(*value)? {
                Object::Class(class_object) => Some(class_object.instance_type),
                _ => None,
            },
            _ => None,
        })
    }

    /// Return the Python-level type independently of the value's physical storage shape.
    /// The Python type of a value: a tag switch for an immediate, the object header for a
    /// heap object. The header is set when the object is allocated and re-typed only by
    /// [`Heap::set_type_id`](super::super::heap::Heap::set_type_id) for a builtin exception
    /// instance or an enum member whose class is known only later.
    pub(super) fn type_id(&self, value: &Value<'s>) -> Result<TypeId, String> {
        if value.inline_string_len().is_some() {
            return Ok(BuiltinType::String.id());
        }
        if value.is_object() {
            return self.object_type_id(*value);
        }
        if value.is_none() {
            return Ok(BuiltinType::None.id());
        }
        if value.bool_value().is_some() {
            return Ok(BuiltinType::Bool.id());
        }
        if value.immediate_int().is_some() {
            return Ok(BuiltinType::Int.id());
        }
        if value.float_value().is_some() {
            return Ok(BuiltinType::Float.id());
        }
        if let Some((kind, _)) = value.registered_parts() {
            return Ok(self
                .state
                .types
                .value_kind_type_id_by_index(kind)
                .ok_or("invalid registered value kind")?);
        }
        Ok(
            match value
                .native_value()
                .expect("inline strings were handled above")
            {
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
        )
    }

    pub(super) fn type_of(&self, value: &Value<'s>) -> Result<Value<'s>, String> {
        self.type_value(self.type_id(value)?)
    }

    pub(super) fn is_instance(
        &mut self,
        value: &Value<'s>,
        class: &Value<'s>,
    ) -> Result<bool, String> {
        if class.is_object() {
            if let Object::Tuple(classes) = self.get(*class)? {
                let classes = self.handles(classes);
                for class in classes {
                    self.charge_cpu(1)?;
                    if self.is_instance(value, &class)? {
                        return Ok(true);
                    }
                }
                return Ok(false);
            }
        }
        if let Some(answer) = self.metaclass_check(class, "__instancecheck__", value)? {
            return Ok(answer);
        }
        let class = self
            .class_type_id(class)?
            .ok_or("isinstance() requires a class argument")?;
        self.state.types.is_subclass(self.type_id(value)?, class)
    }

    /// Ask a user metaclass's `__instancecheck__` or `__subclasscheck__` about `subject`, when
    /// `class` is a class whose metaclass defines the hook. `None` leaves the decision to the
    /// ordinary type hierarchy, as for classes created by `type` itself.
    fn metaclass_check(
        &mut self,
        class: &Value<'s>,
        hook: &str,
        subject: &Value<'s>,
    ) -> Result<Option<bool>, String> {
        if !class.is_object() {
            return Ok(None);
        }
        let Object::Class(class_object) = self.get(*class)? else {
            return Ok(None);
        };
        let metaclass = self.handle(&class_object.metaclass);
        if !metaclass.is_object() || !matches!(self.get(metaclass)?, Object::Class(_)) {
            return Ok(None);
        }
        let Some((owner, descriptor)) = self.class_attribute_entry(metaclass, hook)? else {
            return Ok(None);
        };
        let method = self.bind_descriptor(descriptor, Some(*class), metaclass, owner)?;
        let answer = self.invoke_value(method, vec![*subject])?;
        self.truth_value(&answer).map(Some)
    }

    pub(super) fn is_subclass(
        &mut self,
        class: &Value<'s>,
        base: &Value<'s>,
    ) -> Result<bool, String> {
        if base.is_object() {
            if let Object::Tuple(bases) = self.get(*base)? {
                let bases = self.handles(bases);
                for base in bases {
                    self.charge_cpu(1)?;
                    if self.is_subclass(class, &base)? {
                        return Ok(true);
                    }
                }
                return Ok(false);
            }
        }
        if let Some(answer) = self.metaclass_check(base, "__subclasscheck__", class)? {
            return Ok(answer);
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
        class: &Value<'s>,
    ) -> Result<Option<&'static str>, String> {
        if let Some(NativeValue::ExceptionType(ExceptionType(name))) = class.native_value() {
            return Ok(Some(name));
        }
        if !class.is_object() {
            return Ok(None);
        }
        Ok(match self.get(*class)? {
            Object::Class(class_object) => class_object.exception_base,

            _ => None,
        })
    }
}
