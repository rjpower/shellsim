//! Name, scope, import, and per-code cache operations used by bytecode execution.
//!
//! Lexical scopes are heap objects. The VM's scope stacks hold stored references to them, and
//! every operation here takes a handle from those roots before it touches the heap, so a scope
//! stays valid while a lookup or store allocates.

use super::super::heap::Builder;
use super::super::object_model::TypeId;
use super::super::scopes;
use super::{
    cpython_names, exception_types, protocol, Arc, BuiltinType, CodeRef, ExceptionType, Execution,
    HashMap, ModuleDef, NameId, NamespaceTarget, NativeValue, Object, ProxyTarget, PyModuleLoader,
    PyRuntime, RaisedException, SymbolId, Value, Vm, BUILTIN_FUNCTIONS,
};

/// A [`NamespaceTarget`] whose references are handles in the current scope, so it can be held
/// across allocation. Store it back into an object with [`NamespaceHandle::store`].
#[derive(Clone, Copy)]
pub(super) enum NamespaceHandle<'s> {
    Scope(Value<'s>),
    Repl,
    Instance(Value<'s>),
}

impl NamespaceHandle<'_> {
    /// The stored form, for an `Object::NamespaceDict` built with `alloc_with` or `modify`.
    pub(super) fn store(self, builder: &Builder<'_>) -> NamespaceTarget {
        match self {
            Self::Scope(scope) => NamespaceTarget::Scope(builder.store(scope)),
            Self::Repl => NamespaceTarget::Repl,
            Self::Instance(instance) => NamespaceTarget::Instance(builder.store(instance)),
        }
    }
}

/// A [`ProxyTarget`] whose references are handles in the current scope.
#[derive(Clone, Copy)]
pub(super) enum ProxyHandle<'s> {
    Class(Value<'s>),
    RegisteredType(TypeId),
    NativeModule(&'static ModuleDef),
}

impl<'s> Vm<'s> {
    /// Handles for a namespace view's target, read while the view object is borrowed.
    pub(super) fn namespace_handle(&self, target: &NamespaceTarget) -> NamespaceHandle<'s> {
        match target {
            NamespaceTarget::Scope(scope) => NamespaceHandle::Scope(self.handle(scope)),
            NamespaceTarget::Repl => NamespaceHandle::Repl,
            NamespaceTarget::Instance(instance) => NamespaceHandle::Instance(self.handle(instance)),
        }
    }

    /// Handles for a mapping proxy's target, read while the proxy object is borrowed.
    pub(super) fn proxy_handle(&self, target: &ProxyTarget) -> ProxyHandle<'s> {
        match target {
            ProxyTarget::Class(class) => ProxyHandle::Class(self.handle(class)),
            ProxyTarget::RegisteredType(type_id) => ProxyHandle::RegisteredType(*type_id),
            ProxyTarget::NativeModule(module) => ProxyHandle::NativeModule(module),
        }
    }

    /// The innermost lexical scope, if code is running inside one.
    pub(super) fn current_scope(&self) -> Option<Value<'s>> {
        self.local_scopes.last().map(|scope| self.handle(scope))
    }

    pub(super) fn load_name(&mut self, symbol: SymbolId, name: &str) -> Result<(), String> {
        if name == "__class__" {
            let class = self
                .method_frames
                .last()
                .map(|(class, _)| self.handle(class))
                .ok_or("__class__ is only defined inside a class method body")?;
            self.push(class);
            return Ok(());
        }
        let local_scope = self.current_scope();
        let scoped = match local_scope {
            Some(scope) => self.scope_get(scope, name)?,
            None => None,
        };
        let can_use_globals = local_scope
            .map(|scope| scopes::uses_repl_globals(self.heap(), scope))
            .transpose()?
            .unwrap_or(true);
        let value = scoped.or_else(|| {
            can_use_globals
                .then(|| self.state.globals.get(&self.state.heap, symbol))
                .flatten()
        });
        if let Some(value) = value {
            self.push(value);
            return Ok(());
        }
        self.load_builtin(name)
    }

    #[inline(always)]
    pub(super) fn load_global(
        &mut self,
        symbol: SymbolId,
        code: &CodeRef,
        name: NameId,
    ) -> Result<(), String> {
        if self.local_scopes.is_empty() {
            if let Some(value) = self.state.globals.get_ref(symbol) {
                self.execution.stack.push_ref(value);
                return Ok(());
            }
            return self.load_builtin(code.name(name));
        }
        self.load_scoped_global(symbol, code, name)
    }

    #[cold]
    #[inline(never)]
    fn load_scoped_global(
        &mut self,
        symbol: SymbolId,
        code: &CodeRef,
        name: NameId,
    ) -> Result<(), String> {
        let scope = self.current_scope().expect("checked by load_global");
        let root = scopes::root(self.heap(), scope)?;
        let value = if scopes::uses_repl_globals(self.heap(), root)? {
            self.state.globals.get(&self.state.heap, symbol)
        } else {
            self.scope_get(root, code.name(name))?
        };
        if let Some(value) = value {
            self.push(value);
            return Ok(());
        }
        self.load_builtin(code.name(name))
    }

    fn load_builtin(&mut self, name: &str) -> Result<(), String> {
        if let Some((module_name, attribute)) = super::super::stdlib::frozen_builtin(name) {
            self.import(module_name, false)?;
            let module = self.pop()?;
            let callable = self.resolve_attribute(module, attribute)?.ok_or_else(|| {
                format!("frozen module {module_name:?} does not define {attribute:?}")
            })?;
            self.push(callable);
            return Ok(());
        }
        let value = (|| {
            let builtin_type = match name {
                "type" => Some(BuiltinType::Type),
                "object" => Some(BuiltinType::Object),
                "bool" => Some(BuiltinType::Bool),
                "int" => Some(BuiltinType::Int),
                "float" => Some(BuiltinType::Float),
                "complex" => Some(BuiltinType::Complex),
                "str" => Some(BuiltinType::String),
                "bytes" => Some(BuiltinType::Bytes),
                "bytearray" => Some(BuiltinType::ByteArray),
                "list" => Some(BuiltinType::List),
                "tuple" => Some(BuiltinType::Tuple),
                "dict" => Some(BuiltinType::Dict),
                "set" => Some(BuiltinType::Set),
                "frozenset" => Some(BuiltinType::FrozenSet),
                "slice" => Some(BuiltinType::Slice),
                "range" => Some(BuiltinType::Range),
                _ => None,
            };
            if let Some(builtin_type) = builtin_type {
                return Some(Value::Native(NativeValue::BuiltinType(builtin_type)));
            }
            match name {
                "Ellipsis" => return Some(Value::Native(NativeValue::Ellipsis)),
                "NotImplemented" => return Some(Value::Native(NativeValue::NotImplemented)),
                _ => {}
            }
            if let Some(function) = super::super::stdlib::core::builtin_function(name) {
                return Some(Value::Native(NativeValue::NativeFunction(function)));
            }
            let Some(builtin) = BUILTIN_FUNCTIONS
                .iter()
                .find(|(builtin_name, _)| *builtin_name == name)
                .map(|(_, builtin)| *builtin)
            else {
                return exception_types::exception_type(name)
                    .filter(|definition| definition.builtin)
                    .map(|definition| {
                        Value::Native(NativeValue::ExceptionType(ExceptionType(definition.name)))
                    });
            };
            Some(Value::Native(NativeValue::Function(builtin)))
        })();
        let Some(value) = value else {
            if cpython_names::is_cpython_builtin(name) {
                return Err(format!("builtin {name:?} is not implemented"));
            }
            return Err(self.raise_exception("NameError", format!("name '{name}' is not defined")));
        };
        self.push(value);
        Ok(())
    }

    pub(super) fn delete_local(&mut self, slot: usize) -> Result<(), String> {
        let scope = self
            .current_scope()
            .ok_or("local bytecode requires a lexical scope")?;
        scopes::remove_local(&mut self.state.heap, scope, slot).map(|_| ())
    }

    pub(super) fn store_name(&mut self, symbol: SymbolId, name: &str) -> Result<(), String> {
        let value = self.pop()?;
        if let Some(scope) = self.current_scope() {
            let in_class_body = self
                .class_scopes
                .last()
                .is_some_and(|class_scope| self.state.heap.identical_ref(scope, class_scope));
            if in_class_body
                && self
                    .class_bindings
                    .last()
                    .is_some_and(|bindings| !bindings.iter().any(|binding| binding == name))
            {
                self.class_bindings
                    .last_mut()
                    .expect("class binding stack is present")
                    .push(name.to_string());
            }
            self.scope_insert(scope, name.to_string(), value)?;
        } else {
            self.state.globals.insert(
                &self.state.heap,
                symbol,
                value,
                &mut self.interp.resources,
            )?;
        }
        Ok(())
    }

    pub(super) fn store_enclosing(
        &mut self,
        symbol: SymbolId,
        name: &str,
        scope_hops: usize,
    ) -> Result<(), String> {
        let value = self.pop()?;
        let mut target = self.current_scope();
        for _ in 0..scope_hops {
            target = target
                .map(|scope| scopes::parent(self.heap(), scope))
                .transpose()?
                .flatten();
        }
        if let Some(scope) = target {
            self.scope_insert(scope, name.to_string(), value)
        } else {
            self.state.globals.insert(
                &self.state.heap,
                symbol,
                value,
                &mut self.interp.resources,
            )?;
            Ok(())
        }
    }

    pub(super) fn store_nonlocal(&mut self, name: &str) -> Result<(), String> {
        let value = self.pop()?;
        let scope = self
            .current_scope()
            .ok_or_else(|| format!("no binding for nonlocal {name:?} found"))?;
        scopes::store_nonlocal(&mut self.state.heap, scope, name, value)
    }

    pub(super) fn exception_type_matches(
        &mut self,
        expected: Value<'s>,
        actual: &RaisedException,
    ) -> Result<bool, String> {
        if let Some(NativeValue::ExceptionType(ExceptionType(name))) = expected.native_value() {
            let expected_type = self
                .state
                .types
                .exception_type_id(name)
                .ok_or("exception type is not registered")?;
            let actual_value = self.handle(&actual.value);
            return self
                .state
                .types
                .is_subclass(self.type_id(&actual_value)?, expected_type);
        }
        if !expected.is_object() {
            return Err(
                "catching classes that do not inherit from BaseException is not allowed".into(),
            );
        }
        enum Expected<'v> {
            Class(TypeId),
            Tuple(Vec<Value<'v>>),
        }
        let expected = match self.get(expected)? {
            Object::Class(class_object) if class_object.exception_base.is_some() => {
                Expected::Class(class_object.instance_type)
            }
            Object::Tuple(types) => Expected::Tuple(self.handles(types)),
            _ => {
                return Err(
                    "catching classes that do not inherit from BaseException is not allowed".into(),
                )
            }
        };
        match expected {
            Expected::Class(instance_type) => {
                let actual_value = self.handle(&actual.value);
                let actual_type = self.type_id(&actual_value)?;
                self.state.types.is_subclass(actual_type, instance_type)
            }
            Expected::Tuple(types) => {
                for expected in types {
                    self.charge_cpu(1)?;
                    if self.exception_type_matches(expected, actual)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
        }
    }

    pub(super) fn user_exception_kind(&self, value: &Value<'_>) -> Result<Option<String>, String> {
        if !value.is_object() {
            return Ok(None);
        }
        let Object::Instance { class, .. } = self.get(*value)? else {
            return Ok(None);
        };
        let Object::Class(class_object) = self.get(self.handle(class))? else {
            return Ok(None);
        };
        Ok(class_object
            .exception_base
            .map(|_| class_object.name.clone()))
    }

    /// The modeled exception class a user exception instance derives from, if `value` is one.
    pub(super) fn user_exception_base(
        &self,
        value: &Value<'_>,
    ) -> Result<Option<&'static str>, String> {
        if !value.is_object() {
            return Ok(None);
        }
        let Object::Instance { class, .. } = self.get(*value)? else {
            return Ok(None);
        };
        let Object::Class(class_object) = self.get(self.handle(class))? else {
            return Ok(None);
        };
        Ok(class_object.exception_base)
    }

    #[inline(always)]
    pub(super) fn store_global(
        &mut self,
        symbol: SymbolId,
        code: &CodeRef,
        name: NameId,
        value: Value<'s>,
    ) -> Result<(), String> {
        let Some(scope) = self.current_scope() else {
            self.state.globals.insert(
                &self.state.heap,
                symbol,
                value,
                &mut self.interp.resources,
            )?;
            return Ok(());
        };
        self.store_scoped_global(scope, symbol, code, name, value)
    }

    #[cold]
    #[inline(never)]
    fn store_scoped_global(
        &mut self,
        scope: Value<'s>,
        symbol: SymbolId,
        code: &CodeRef,
        name: NameId,
        value: Value<'s>,
    ) -> Result<(), String> {
        let root = scopes::root(self.heap(), scope)?;
        if scopes::uses_repl_globals(self.heap(), root)? {
            self.state.globals.insert(
                &self.state.heap,
                symbol,
                value,
                &mut self.interp.resources,
            )?;
            Ok(())
        } else {
            self.scope_insert(root, code.name(name).to_owned(), value)
        }
    }

    pub(super) fn delete_global(
        &mut self,
        symbol: SymbolId,
        code: &CodeRef,
        name: NameId,
    ) -> Result<(), String> {
        let Some(scope) = self.current_scope() else {
            self.state.globals.remove(&self.state.heap, symbol);
            return Ok(());
        };
        let root = scopes::root(self.heap(), scope)?;
        if scopes::uses_repl_globals(self.heap(), root)? {
            self.state.globals.remove(&self.state.heap, symbol);
        } else {
            scopes::remove(&mut self.state.heap, root, code.name(name))?;
        }
        Ok(())
    }

    /// The target a fresh `globals()` call should reference for the code currently running: an
    /// imported module's own scope, or the REPL/script's flat table when no scope roots the call
    /// (the entry-point program) or the root scope defers to it (a function or class body defined
    /// at that program's top level). Mirrors `load_global`'s and `store_global`'s resolution
    /// exactly, so `globals()` always names the same namespace a bare name lookup would.
    pub(super) fn current_globals_target(&self) -> Result<NamespaceHandle<'s>, String> {
        let Some(scope) = self.current_scope() else {
            return Ok(NamespaceHandle::Repl);
        };
        let root = scopes::root(self.heap(), scope)?;
        if scopes::uses_repl_globals(self.heap(), root)? {
            Ok(NamespaceHandle::Repl)
        } else {
            Ok(NamespaceHandle::Scope(root))
        }
    }

    /// `vars()` and `locals()` with no argument. At a module's or the script's top level this
    /// is the same live view `globals()` returns, as `locals() is globals()` there in CPython.
    /// Inside a function it is a detached `dict` of that call's local variables: CPython's
    /// `locals()` there is a snapshot too, and writing to it never rebinds a local.
    pub(super) fn current_locals(&mut self) -> Result<Value<'s>, String> {
        let Some(scope) = self.current_scope() else {
            return self.alloc(Object::NamespaceDict(NamespaceTarget::Repl));
        };
        // A module's own scope is the only one with no parent that does not defer to the
        // REPL/script table; function calls either have a lexical parent or defer to it.
        let is_module_scope = !scopes::uses_repl_globals(self.heap(), scope)?
            && scopes::parent(self.heap(), scope)?.is_none();
        if is_module_scope {
            return self.alloc_with(|builder| {
                Object::NamespaceDict(NamespaceTarget::Scope(builder.store(scope)))
            });
        }
        self.namespace_snapshot_dict(NamespaceHandle::Scope(scope))
    }

    /// A detached `dict` holding `target`'s current bindings.
    pub(super) fn namespace_snapshot_dict(
        &mut self,
        target: NamespaceHandle<'s>,
    ) -> Result<Value<'s>, String> {
        let items = self.namespace_items(target)?;
        self.allocate_dict(items)
    }

    /// `target`'s bindings as `(key, value)` pairs with freshly allocated string keys, in the
    /// order [`Self::namespace_entries`] gives.
    pub(super) fn namespace_items(
        &mut self,
        target: NamespaceHandle<'s>,
    ) -> Result<Vec<(Value<'s>, Value<'s>)>, String> {
        let entries = self.namespace_entries(target)?;
        let mut items = Vec::with_capacity(entries.len());
        for (name, value) in entries {
            items.push((self.allocate_string(name)?, value));
        }
        Ok(items)
    }

    /// The entries of a class's or native module's read-only `__dict__`, sorted by name, with
    /// freshly allocated string keys. Class attributes live in a `HashMap` and native modules
    /// declare functions apart from values, so neither keeps CPython's definition order.
    pub(super) fn proxy_items(
        &mut self,
        target: ProxyHandle<'s>,
    ) -> Result<Vec<(Value<'s>, Value<'s>)>, String> {
        let mut entries: Vec<(String, Value<'s>)> = match target {
            ProxyHandle::Class(class) => {
                let Object::Class(class_object) = self.get(class)? else {
                    return Err("mappingproxy target is not a class".into());
                };
                class_object
                    .attributes
                    .iter()
                    .map(|(name, value)| (name.clone(), self.handle(value)))
                    .collect()
            }
            ProxyHandle::RegisteredType(type_id) => self
                .state
                .types
                .get(type_id)?
                .attributes
                .iter()
                .map(|(name, value)| (name.clone(), self.handle(value)))
                .collect(),
            ProxyHandle::NativeModule(module) => {
                let mut entries = Vec::with_capacity(module.functions.len() + module.values.len());
                for function in module.functions {
                    let value = Value::Native(NativeValue::NativeFunction(function));
                    entries.push((function.name.to_string(), value));
                }
                for definition in module.values {
                    let value = definition.get(self).map_err(|error| error.to_string())?;
                    entries.push((definition.name().to_string(), value));
                }
                entries
            }
        };
        self.charge_cpu(u64::try_from(entries.len()).unwrap_or(u64::MAX))?;
        entries.sort_by(|(left, _), (right, _)| left.cmp(right));
        let mut items = Vec::with_capacity(entries.len());
        for (name, value) in entries {
            items.push((self.allocate_string(name)?, value));
        }
        Ok(items)
    }

    /// Every binding a namespace target currently holds, in a deterministic order.
    ///
    /// Module and REPL/script namespaces are sorted by name. Neither backing store keeps
    /// Python's insertion order cheaply: a module scope keeps its dynamic bindings in a
    /// `HashMap`, and the REPL/script table is indexed by interned symbol. Instance attributes
    /// come in the order [`Vm::instance_attribute_names`] gives.
    pub(super) fn namespace_entries(
        &self,
        target: NamespaceHandle<'s>,
    ) -> Result<Vec<(String, Value<'s>)>, String> {
        let mut entries: Vec<(String, Value<'s>)> = match target {
            NamespaceHandle::Scope(scope) => {
                scopes::values(self.heap(), scope)?.into_iter().collect()
            }
            NamespaceHandle::Repl => self
                .state
                .globals
                .entries(&self.state.heap, &self.state.symbols),
            NamespaceHandle::Instance(instance) => {
                let mut entries = Vec::new();
                for name in self.instance_attribute_names(instance)? {
                    if let Some(value) = self.namespace_lookup(target, &name)? {
                        entries.push((name, value));
                    }
                }
                return Ok(entries);
            }
        };
        entries.sort_by(|(left, _), (right, _)| left.cmp(right));
        Ok(entries)
    }

    pub(super) fn namespace_lookup(
        &self,
        target: NamespaceHandle<'s>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
        if let NamespaceHandle::Scope(scope) = target {
            return self.scope_get(scope, name);
        }
        // REPL/script names and instance attributes are keyed by interned symbol, so a name that
        // was never interned is not bound there.
        let Some(symbol) = self.symbol_id(name) else {
            return Ok(None);
        };
        Ok(match target {
            NamespaceHandle::Repl => self.state.globals.get(&self.state.heap, symbol),
            NamespaceHandle::Instance(instance) => self.attribute_by_symbol(instance, symbol)?,
            NamespaceHandle::Scope(_) => unreachable!("scope targets returned above"),
        })
    }

    /// Bind `name` in `target`. An instance write goes straight to the instance's own
    /// attributes, bypassing `__setattr__` and descriptors, as a write to CPython's
    /// `obj.__dict__` does.
    pub(super) fn namespace_store(
        &mut self,
        target: NamespaceHandle<'s>,
        name: String,
        value: Value<'s>,
    ) -> Result<(), String> {
        if let NamespaceHandle::Scope(scope) = target {
            return self.scope_insert(scope, name, value);
        }
        let symbol = self.intern_symbol(&name)?;
        match target {
            NamespaceHandle::Repl => self.state.globals.insert(
                &self.state.heap,
                symbol,
                value,
                &mut self.interp.resources,
            ),
            NamespaceHandle::Instance(instance) => {
                self.insert_attribute_by_symbol(instance, symbol, value)
            }
            NamespaceHandle::Scope(_) => unreachable!("scope targets returned above"),
        }
    }

    pub(super) fn namespace_delete(
        &mut self,
        target: NamespaceHandle<'s>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
        if let NamespaceHandle::Scope(scope) = target {
            return scopes::remove(&mut self.state.heap, scope, name);
        }
        let Some(symbol) = self.symbol_id(name) else {
            return Ok(None);
        };
        match target {
            NamespaceHandle::Repl => Ok(self.state.globals.remove(&self.state.heap, symbol)),
            NamespaceHandle::Instance(instance) => {
                self.remove_attribute_by_symbol(instance, symbol)
            }
            NamespaceHandle::Scope(_) => unreachable!("scope targets returned above"),
        }
    }

    fn import_roots(&mut self) -> Result<Vec<String>, String> {
        let mut roots = self.state.temporary_import_paths.clone();
        if let Some(path) = self.handle_optional(self.state.sys_path.as_ref()) {
            if !path.is_object() {
                return Err("sys.path lost list identity".into());
            }
            let Object::List(values) = self.get(path)? else {
                return Err("sys.path must remain a list".into());
            };
            for value in self.handles(values) {
                let path = self
                    .string_value(&value)
                    .map_err(|error| self.record_native_error(error))?
                    .ok_or("sys.path entries must be strings")?;
                roots.push(path);
            }
        } else {
            roots.extend(self.state.import_paths.clone());
        }
        if roots.is_empty() {
            roots.push(self.interp.cwd.clone());
        }
        Ok(roots)
    }

    /// The module object imported under `name`, if any.
    fn loaded_module(&self, name: &str) -> Option<Value<'s>> {
        self.handle_optional(self.state.modules.get(name))
    }

    /// Allocate a module object over a fresh module scope binding `values`.
    fn allocate_module(
        &mut self,
        name: String,
        values: HashMap<String, Value<'s>>,
    ) -> Result<(Value<'s>, Value<'s>), String> {
        let scope = self.alloc_scope(None, false, Arc::from([]), Vec::new(), values)?;
        let module = self.alloc_with(|builder| Object::Module {
            name,
            scope: builder.store(scope),
        })?;
        Ok((module, scope))
    }

    pub(super) fn import(&mut self, name: &str, bind_root: bool) -> Result<(), String> {
        let name = self.resolve_import_name(name)?;
        if let Some(module) = super::super::stdlib::native_module(&name) {
            return self.finish_import(
                &name,
                Value::Native(NativeValue::Module(module)),
                bind_root,
            );
        }
        if let Some(module) = self.loaded_module(&name) {
            return self.finish_import(&name, module, bind_root);
        }

        self.ensure_package_parent(&name)?;
        // A package initializer may import the requested child itself. Reuse that exact module
        // object so classes and exceptions retain identity across the import cycle.
        if let Some(module) = self.loaded_module(&name) {
            return self.finish_import(&name, module, bind_root);
        }

        let frozen = super::super::stdlib::frozen_module(&name);
        let source = if let Some(frozen) = frozen {
            Some((format!("<frozen {name}>"), frozen.source.to_string()))
        } else {
            let roots = self.import_roots()?;
            self.interp
                .load_module_source(&roots, &name)
                .map_err(|error| self.record_native_error(error))?
        };
        let Some((path, source)) = source else {
            // A standard package that shellsim does not provide at all is unsupported. A missing
            // submodule of a provided package is absent, like a missing module attribute.
            let top_level = name.split('.').next().unwrap_or(&name);
            let provided = super::super::stdlib::native_module(top_level).is_some()
                || super::super::stdlib::frozen_module(top_level).is_some()
                || self.state.modules.contains_key(top_level);
            if !provided && cpython_names::is_cpython_stdlib_module(top_level) {
                return Err(format!(
                    "standard-library module {name:?} is not implemented"
                ));
            }
            return Err(
                self.raise_exception("ModuleNotFoundError", format!("No module named '{name}'"))
            );
        };
        let parse_memory = u64::try_from(source.len())
            .ok()
            .and_then(|bytes| bytes.checked_mul(4))
            .ok_or("module source is too large")?;
        if !self.interp.resources.reserve_memory(parse_memory)
            || !self.interp.resources.charge_cpu(source.len() as u64)
        {
            return Err("resource limit exceeded while importing module".into());
        }
        let tokens = super::super::lexer::lex(&source).map_err(|error| {
            format!(
                "{} in {path} at line {}, column {}",
                error.message, error.span.line, error.span.column
            )
        })?;
        let program = super::super::parser::parse(tokens).map_err(|error| {
            format!(
                "{} in {path} at line {}, column {}",
                error.message, error.span.line, error.span.column
            )
        })?;
        let code = super::super::compiler::compile(program);
        // A frozen module's path is a synthetic `<frozen name>` marker, so its package-ness comes
        // from the bundled file name instead.
        let is_package = path.ends_with("/__init__.py") || frozen.is_some_and(|f| f.is_package());
        let package_name = if is_package {
            name.clone()
        } else {
            name.rsplit_once('.')
                .map_or_else(String::new, |(package, _)| package.to_string())
        };
        let module_name = self.allocate_string(name.clone())?;
        let module_package = self.allocate_string(package_name)?;
        let module_file = self.allocate_string(path.clone())?;
        let (module, scope) = self.allocate_module(
            name.clone(),
            HashMap::from([
                ("__name__".into(), module_name),
                ("__package__".into(), module_package),
                ("__file__".into(), module_file),
            ]),
        )?;
        let stored = self.store(module);
        self.state.modules.insert(name.clone(), stored);

        let temporary_import_path = (!path.starts_with('<')).then(|| {
            path.rsplit_once('/')
                .map_or_else(|| "/".to_string(), |(parent, _)| parent.to_string())
        });
        if let Some(path) = &temporary_import_path {
            self.state.temporary_import_paths.insert(0, path.clone());
        }
        let stored_scope = self.store(scope);
        self.local_scopes.push(stored_scope);
        let execution = self.execute_code(&code);
        self.local_scopes.pop();
        if temporary_import_path.is_some() {
            self.state.temporary_import_paths.remove(0);
        }
        match execution {
            Ok(Execution::Pending) => unreachable!("execute_code drains pending quanta"),
            Ok(Execution::Blocked(_)) => unreachable!("immediate code cannot suspend"),
            Ok(Execution::Halt) => self.finish_import(&name, module, bind_root),
            Ok(Execution::Return(_)) => {
                self.state.modules.remove(&name);
                Err(format!("'return' outside function in module {name:?}"))
            }
            Ok(Execution::Yield(_, _)) => {
                self.state.modules.remove(&name);
                Err(format!("'yield' outside function in module {name:?}"))
            }
            Ok(Execution::Exit(status)) => {
                self.state.modules.remove(&name);
                Err(format!("module {name:?} exited with status {status}"))
            }
            Err((error, span)) => {
                self.state.modules.remove(&name);
                Err(format!(
                    "{error} in {path} at line {}, column {}",
                    span.line, span.column
                ))
            }
        }
    }

    fn resolve_import_name(&self, requested: &str) -> Result<String, String> {
        let level = requested
            .chars()
            .take_while(|character| *character == '.')
            .count();
        if level == 0 {
            return Ok(requested.to_string());
        }
        let scope = self
            .current_scope()
            .ok_or("relative import requires a package context")?;
        let package = self
            .scope_get(scope, "__package__")?
            .ok_or("relative import requires __package__")?;
        let package = protocol::string_value(&self.state.heap, package)?
            .filter(|package| !package.is_empty())
            .ok_or("relative import requires a non-empty package")?;
        let mut parts = package.split('.').collect::<Vec<_>>();
        if level > parts.len() {
            return Err("attempted relative import beyond top-level package".into());
        }
        parts.truncate(parts.len() + 1 - level);
        let suffix = &requested[level..];
        if !suffix.is_empty() {
            parts.push(suffix);
        }
        Ok(parts.join("."))
    }

    fn ensure_package_parent(&mut self, name: &str) -> Result<(), String> {
        let Some((parent, _)) = name.rsplit_once('.') else {
            return Ok(());
        };
        if self.state.modules.contains_key(parent) {
            return Ok(());
        }
        let standard = super::super::stdlib::native_module(parent).is_some()
            || super::super::stdlib::frozen_module(parent).is_some();
        let vfs_package = if standard {
            false
        } else {
            let roots = self.import_roots()?;
            self.interp
                .load_module_source(&roots, parent)
                .map_err(|error| self.record_native_error(error))?
                .is_some_and(|(path, _)| path.ends_with("/__init__.py"))
        };
        if standard || vfs_package {
            self.import(parent, false)?;
            self.execution
                .stack
                .pop_ref()
                .ok_or("parent import produced no value")?;
        }
        Ok(())
    }

    /// `from module import name` with the module on top of the stack: push the module attribute,
    /// else the submodule `module.name`, else raise CPython's `ImportError`.
    pub(super) fn import_from(&mut self, name: &str) -> Result<(), String> {
        let module = self
            .execution
            .stack
            .peek(&self.state.heap, 0)
            .ok_or("import stack underflow")?;
        if let Some(value) = self.resolve_optional_attribute(module, name)? {
            self.push(value);
            return Ok(());
        }
        let module_name = match module.native_value() {
            Some(NativeValue::Module(definition)) => definition.name.to_string(),
            _ => match module.is_object().then(|| self.get(module)).transpose()? {
                Some(Object::Module { name, .. }) => name.clone(),
                _ => return Err(self.missing_attribute(&module, name)),
            },
        };
        match self.import(&format!("{module_name}.{name}"), false) {
            Err(_)
                if self
                    .pending_exception
                    .as_ref()
                    .is_some_and(|exception| exception.kind == "ModuleNotFoundError") =>
            {
                self.pending_exception = None;
                let message =
                    format!("cannot import name '{name}' from '{module_name}' (unknown location)");
                Err(self.raise_exception("ImportError", message))
            }
            result => result,
        }
    }

    /// `from module import *` with the module on top of the stack: bind every name in the
    /// module's `__all__`, or else every name without a leading underscore, in the current scope.
    pub(super) fn import_star(&mut self) -> Result<(), String> {
        let module = self.pop()?;
        let names = match module.native_value() {
            Some(NativeValue::Module(definition)) => definition
                .functions
                .iter()
                .map(|function| function.name)
                .chain(definition.values.iter().map(|value| value.name()))
                .filter(|name| !name.starts_with('_'))
                .map(str::to_string)
                .collect::<Vec<_>>(),
            _ => {
                let Some(Object::Module { scope, name }) =
                    module.is_object().then(|| self.get(module)).transpose()?
                else {
                    return Err(self.missing_attribute(&module, "__all__"));
                };
                let (scope, module_name) = (self.handle(scope), name.clone());
                match self.scope_get(scope, "__all__")? {
                    Some(all) => {
                        let items = match all.is_object().then(|| self.get(all)).transpose()? {
                            Some(Object::List(items) | Object::Tuple(items)) => self.handles(items),
                            // CPython indexes `__all__`; lists and tuples are what modules use.
                            _ => {
                                let message = format!(
                                    "'{}' object does not support indexing",
                                    self.type_name_of(&all)?
                                );
                                return Err(self.raise_exception("TypeError", message));
                            }
                        };
                        let mut names = Vec::with_capacity(items.len());
                        for item in items {
                            let Some(name) = protocol::string_value(&self.state.heap, item)? else {
                                let message = format!(
                                    "Item in {module_name}.__all__ must be str, not {}",
                                    self.type_name_of(&item)?
                                );
                                return Err(self.raise_exception("TypeError", message));
                            };
                            names.push(name);
                        }
                        names
                    }
                    None => {
                        let mut names = scopes::values(self.heap(), scope)?
                            .into_keys()
                            .filter(|name| !name.starts_with('_'))
                            .collect::<Vec<_>>();
                        // Bind in a stable order so later metering and errors are deterministic.
                        names.sort_unstable();
                        names
                    }
                }
            }
        };
        for name in names {
            self.charge_cpu(1)?;
            let Some(value) = self.resolve_attribute(module, &name)? else {
                return Err(self.missing_attribute(&module, &name));
            };
            if let Some(scope) = self.current_scope() {
                self.scope_insert(scope, name, value)?;
            } else {
                let symbol = self.intern_symbol(&name)?;
                self.state.globals.insert(
                    &self.state.heap,
                    symbol,
                    value,
                    &mut self.interp.resources,
                )?;
            }
        }
        Ok(())
    }

    /// The scope of the module imported as `name`, for binding a submodule into it.
    fn module_scope(&self, name: &str, module: Value<'s>) -> Result<Value<'s>, String> {
        if !module.is_object() {
            return Err(format!("module {name:?} cannot contain submodules"));
        }
        let Object::Module { scope, .. } = self.get(module)? else {
            return Err(format!("module {name:?} changed object kind"));
        };
        Ok(self.handle(scope))
    }

    /// Install synthetic package parents for a dotted import and push the value selected by
    /// Python's ordinary import binding rule. Package objects contain only VM module references.
    fn finish_import(
        &mut self,
        name: &str,
        leaf: Value<'s>,
        bind_root: bool,
    ) -> Result<(), String> {
        if let Some((parent_name, child_name)) = name.rsplit_once('.') {
            if let Some(parent) = self.loaded_module(parent_name) {
                let scope = self.module_scope(parent_name, parent)?;
                self.scope_insert(scope, child_name.to_string(), leaf)?;
            }
        }
        if !bind_root || !name.contains('.') {
            self.push(leaf);
            return Ok(());
        }
        let parts = name.split('.').collect::<Vec<_>>();
        let mut child = leaf;
        for parent_end in (1..parts.len()).rev() {
            let parent_name = parts[..parent_end].join(".");
            let child_name = parts[parent_end];
            let parent = if let Some(parent) = self.loaded_module(&parent_name) {
                let scope = self.module_scope(&parent_name, parent)?;
                self.scope_insert(scope, child_name.to_string(), child)?;
                parent
            } else {
                let module_name = self.allocate_string(parent_name.clone())?;
                let (parent, _) = self.allocate_module(
                    parent_name.clone(),
                    HashMap::from([
                        ("__name__".into(), module_name),
                        (child_name.to_string(), child),
                    ]),
                )?;
                let stored = self.store(parent);
                self.state.modules.insert(parent_name, stored);
                parent
            };
            child = parent;
        }
        self.push(child);
        Ok(())
    }
}
