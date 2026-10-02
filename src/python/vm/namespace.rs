//! Name, scope, import, and per-code cache operations used by bytecode execution.

use super::{
    cpython_names, exception_types, protocol, Arc, BuiltinType, CodeRef, ExceptionType, Execution,
    HashMap, NameId, NamespaceTarget, NativeValue, Object, ProxyTarget, PyModuleLoader, PyRuntime,
    RaisedException, ScopeId, SymbolId, Value, Vm, BUILTIN_FUNCTIONS,
};

impl Vm<'_> {
    pub(super) fn load_name(&mut self, symbol: SymbolId, name: &str) -> Result<(), String> {
        if name == "__class__" {
            let class = self
                .method_frames
                .last()
                .map(|(class, _)| *class)
                .ok_or("__class__ is only defined inside a class method body")?;
            self.stack.push(Value::Object(class));
            return Ok(());
        }
        let local_scope = self.local_scopes.last().copied();
        let scoped = local_scope.and_then(|scope| self.state.heap.scope_get(scope, name).cloned());
        let can_use_globals = local_scope
            .map(|scope| self.state.heap.scope_uses_repl_globals(scope))
            .transpose()?
            .unwrap_or(true);
        let value = scoped.or_else(|| {
            can_use_globals
                .then(|| self.state.globals.get(symbol))
                .flatten()
        });
        if let Some(value) = value {
            self.stack.push(value);
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
            if let Some(value) = self.state.globals.get(symbol) {
                self.stack.push(value);
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
        let scope = *self.local_scopes.last().expect("checked by load_global");
        let root = self.state.heap.scope_root(scope)?;
        let value = if self.state.heap.scope_uses_repl_globals(root)? {
            self.state.globals.get(symbol)
        } else {
            self.state.heap.scope_get(root, code.name(name)).copied()
        };
        if let Some(value) = value {
            self.stack.push(value);
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
            self.stack.push(callable);
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
        self.stack.push(value);
        Ok(())
    }

    pub(super) fn load_local(&mut self, slot: usize) -> Result<(), String> {
        let scope = self
            .local_scopes
            .last()
            .copied()
            .ok_or("local bytecode requires a lexical scope")?;
        let value = self
            .state
            .heap
            .scope_get_local(scope, slot)?
            .ok_or_else(|| "local variable referenced before assignment".to_string())?;
        self.stack.push(value);
        Ok(())
    }

    pub(super) fn store_local(&mut self, slot: usize) -> Result<(), String> {
        let value = self.pop()?;
        let scope = self
            .local_scopes
            .last()
            .copied()
            .ok_or("local bytecode requires a lexical scope")?;
        self.state.heap.scope_store_local(scope, slot, value)
    }

    pub(super) fn delete_local(&mut self, slot: usize) -> Result<(), String> {
        let scope = self
            .local_scopes
            .last()
            .copied()
            .ok_or("local bytecode requires a lexical scope")?;
        self.state.heap.scope_remove_local(scope, slot).map(|_| ())
    }

    pub(super) fn store_name(&mut self, symbol: SymbolId, name: &str) -> Result<(), String> {
        let value = self.pop()?;
        if let Some(scope) = self.local_scopes.last().copied() {
            if self.class_scopes.last().copied() == Some(scope)
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
            self.state.heap.scope_insert(
                scope,
                name.to_string(),
                value,
                &mut self.interp.resources,
            )?;
        } else {
            self.state
                .globals
                .insert(symbol, value, &mut self.interp.resources)?;
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
        let mut target = self.local_scopes.last().copied();
        for _ in 0..scope_hops {
            target = target
                .map(|scope| self.state.heap.scope_parent(scope))
                .transpose()?
                .flatten();
        }
        if let Some(scope) = target {
            self.state
                .heap
                .scope_insert(scope, name.to_string(), value, &mut self.interp.resources)
        } else {
            self.state
                .globals
                .insert(symbol, value, &mut self.interp.resources)?;
            Ok(())
        }
    }

    pub(super) fn store_nonlocal(&mut self, name: &str) -> Result<(), String> {
        let value = self.pop()?;
        let scope = self
            .local_scopes
            .last()
            .copied()
            .ok_or_else(|| format!("no binding for nonlocal {name:?} found"))?;
        self.state.heap.scope_store_nonlocal(scope, name, value)
    }

    pub(super) fn exception_type_matches(
        &mut self,
        expected: Value,
        actual: &RaisedException,
    ) -> Result<bool, String> {
        if let Some(NativeValue::ExceptionType(ExceptionType(name))) = expected.native_value() {
            let expected_type = self
                .state
                .types
                .exception_type_id(name)
                .ok_or("exception type is not registered")?;
            return self
                .state
                .types
                .is_subclass(self.type_id(&actual.value)?, expected_type);
        }
        let Some(id) = expected.object_id() else {
            return Err(
                "catching classes that do not inherit from BaseException is not allowed".into(),
            );
        };
        match self.state.heap.get(id)?.clone() {
            Object::Class {
                instance_type,
                exception_base: Some(_),
                ..
            } => {
                let actual_type = self.type_id(&actual.value)?;
                self.state.types.is_subclass(actual_type, instance_type)
            }
            Object::Tuple(types) => {
                for expected in types {
                    self.charge_cpu(1)?;
                    if self.exception_type_matches(expected, actual)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            _ => {
                Err("catching classes that do not inherit from BaseException is not allowed".into())
            }
        }
    }

    pub(super) fn user_exception_kind(&self, value: &Value) -> Result<Option<String>, String> {
        let Some(id) = value.object_id() else {
            return Ok(None);
        };
        let Object::Instance { class, .. } = self.state.heap.get(id)? else {
            return Ok(None);
        };
        let Object::Class {
            name,
            exception_base,
            ..
        } = self.state.heap.get(*class)?
        else {
            return Ok(None);
        };
        Ok(exception_base.map(|_| name.clone()))
    }

    /// The modeled exception class a user exception instance derives from, if `value` is one.
    pub(super) fn user_exception_base(
        &self,
        value: &Value,
    ) -> Result<Option<&'static str>, String> {
        let Some(id) = value.object_id() else {
            return Ok(None);
        };
        let Object::Instance { class, .. } = self.state.heap.get(id)? else {
            return Ok(None);
        };
        let Object::Class { exception_base, .. } = self.state.heap.get(*class)? else {
            return Ok(None);
        };
        Ok(*exception_base)
    }

    #[inline(always)]
    pub(super) fn store_global(
        &mut self,
        symbol: SymbolId,
        code: &CodeRef,
        name: NameId,
        value: Value,
    ) -> Result<(), String> {
        let Some(scope) = self.local_scopes.last().copied() else {
            self.state
                .globals
                .insert(symbol, value, &mut self.interp.resources)?;
            return Ok(());
        };
        self.store_scoped_global(scope, symbol, code, name, value)
    }

    #[cold]
    #[inline(never)]
    fn store_scoped_global(
        &mut self,
        scope: ScopeId,
        symbol: SymbolId,
        code: &CodeRef,
        name: NameId,
        value: Value,
    ) -> Result<(), String> {
        let root = self.state.heap.scope_root(scope)?;
        if self.state.heap.scope_uses_repl_globals(root)? {
            self.state
                .globals
                .insert(symbol, value, &mut self.interp.resources)?;
            Ok(())
        } else {
            self.state.heap.scope_insert(
                root,
                code.name(name).to_owned(),
                value,
                &mut self.interp.resources,
            )
        }
    }

    pub(super) fn delete_global(
        &mut self,
        symbol: SymbolId,
        code: &CodeRef,
        name: NameId,
    ) -> Result<(), String> {
        let Some(scope) = self.local_scopes.last().copied() else {
            self.state.globals.remove(symbol);
            return Ok(());
        };
        let root = self.state.heap.scope_root(scope)?;
        if self.state.heap.scope_uses_repl_globals(root)? {
            self.state.globals.remove(symbol);
        } else {
            self.state.heap.scope_remove(root, code.name(name))?;
        }
        Ok(())
    }

    /// The target a fresh `globals()` call should reference for the code currently running: an
    /// imported module's own scope, or the REPL/script's flat table when no scope roots the call
    /// (the entry-point program) or the root scope defers to it (a function or class body defined
    /// at that program's top level). Mirrors `load_global`'s and `store_global`'s resolution
    /// exactly, so `globals()` always names the same namespace a bare name lookup would.
    pub(super) fn current_globals_target(&self) -> Result<NamespaceTarget, String> {
        let Some(scope) = self.local_scopes.last().copied() else {
            return Ok(NamespaceTarget::Repl);
        };
        let root = self.state.heap.scope_root(scope)?;
        if self.state.heap.scope_uses_repl_globals(root)? {
            Ok(NamespaceTarget::Repl)
        } else {
            Ok(NamespaceTarget::Scope(root))
        }
    }

    /// `vars()` and `locals()` with no argument. At a module's or the script's top level this
    /// is the same live view `globals()` returns, as `locals() is globals()` there in CPython.
    /// Inside a function it is a detached `dict` of that call's local variables: CPython's
    /// `locals()` there is a snapshot too, and writing to it never rebinds a local.
    pub(super) fn current_locals(&mut self) -> Result<Value, String> {
        let Some(scope) = self.local_scopes.last().copied() else {
            return self.allocate_object(Object::NamespaceDict(NamespaceTarget::Repl));
        };
        // A module's own scope is the only one with no parent that does not defer to the
        // REPL/script table; function calls either have a lexical parent or defer to it.
        let is_module_scope = !self.state.heap.scope_uses_repl_globals(scope)?
            && self.state.heap.scope_parent(scope)?.is_none();
        if is_module_scope {
            return self.allocate_object(Object::NamespaceDict(NamespaceTarget::Scope(scope)));
        }
        self.namespace_snapshot_dict(NamespaceTarget::Scope(scope))
    }

    /// A detached `dict` holding `target`'s current bindings.
    pub(super) fn namespace_snapshot_dict(
        &mut self,
        target: NamespaceTarget,
    ) -> Result<Value, String> {
        let items = self.namespace_items(target)?;
        self.allocate_dict(items)
    }

    /// `target`'s bindings as `(key, value)` pairs with freshly allocated string keys, in the
    /// order [`Self::namespace_entries`] gives.
    pub(super) fn namespace_items(
        &mut self,
        target: NamespaceTarget,
    ) -> Result<Vec<(Value, Value)>, String> {
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
        target: ProxyTarget,
    ) -> Result<Vec<(Value, Value)>, String> {
        let mut entries: Vec<(String, Value)> = match target {
            ProxyTarget::Class(class) => {
                let Object::Class { attributes, .. } = self.state.heap.get(class)? else {
                    return Err("mappingproxy target is not a class".into());
                };
                attributes
                    .iter()
                    .map(|(name, value)| (name.clone(), *value))
                    .collect()
            }
            ProxyTarget::RegisteredType(type_id) => self
                .state
                .types
                .get(type_id)?
                .attributes
                .iter()
                .map(|(name, value)| (name.clone(), *value))
                .collect(),
            ProxyTarget::NativeModule(module) => {
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
    /// come in the order [`Heap::instance_attribute_names`] gives.
    ///
    /// [`Heap::instance_attribute_names`]: super::super::heap::Heap::instance_attribute_names
    pub(super) fn namespace_entries(
        &self,
        target: NamespaceTarget,
    ) -> Result<Vec<(String, Value)>, String> {
        let mut entries: Vec<(String, Value)> = match target {
            NamespaceTarget::Scope(scope) => {
                self.state.heap.scope_values(scope)?.into_iter().collect()
            }
            NamespaceTarget::Repl => self.state.globals.entries(&self.state.heap),
            NamespaceTarget::Instance(id) => {
                let mut entries = Vec::new();
                for name in self.state.heap.instance_attribute_names(id)? {
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
        target: NamespaceTarget,
        name: &str,
    ) -> Result<Option<Value>, String> {
        if let NamespaceTarget::Scope(scope) = target {
            return Ok(self.state.heap.scope_get(scope, name).copied());
        }
        // REPL/script names and instance attributes are keyed by interned symbol, so a name that
        // was never interned is not bound there.
        let Some(symbol) = self.state.heap.symbol_id(name) else {
            return Ok(None);
        };
        Ok(match target {
            NamespaceTarget::Repl => self.state.globals.get(symbol),
            NamespaceTarget::Instance(id) => {
                self.state.heap.attribute_by_symbol(id, symbol)?.copied()
            }
            NamespaceTarget::Scope(_) => unreachable!("scope targets returned above"),
        })
    }

    /// Bind `name` in `target`. An instance write goes straight to the instance's own
    /// attributes, bypassing `__setattr__` and descriptors, as a write to CPython's
    /// `obj.__dict__` does.
    pub(super) fn namespace_store(
        &mut self,
        target: NamespaceTarget,
        name: String,
        value: Value,
    ) -> Result<(), String> {
        if let NamespaceTarget::Scope(scope) = target {
            return self
                .state
                .heap
                .scope_insert(scope, name, value, &mut self.interp.resources);
        }
        let symbol = self
            .state
            .heap
            .intern_symbol(&name, &mut self.interp.resources)?;
        match target {
            NamespaceTarget::Repl => {
                self.state
                    .globals
                    .insert(symbol, value, &mut self.interp.resources)
            }
            NamespaceTarget::Instance(id) => self.state.heap.insert_attribute_by_symbol(
                id,
                symbol,
                value,
                &mut self.interp.resources,
            ),
            NamespaceTarget::Scope(_) => unreachable!("scope targets returned above"),
        }
    }

    pub(super) fn namespace_delete(
        &mut self,
        target: NamespaceTarget,
        name: &str,
    ) -> Result<Option<Value>, String> {
        if let NamespaceTarget::Scope(scope) = target {
            return self.state.heap.scope_remove(scope, name);
        }
        let Some(symbol) = self.state.heap.symbol_id(name) else {
            return Ok(None);
        };
        match target {
            NamespaceTarget::Repl => Ok(self.state.globals.remove(symbol)),
            NamespaceTarget::Instance(id) => {
                self.state
                    .heap
                    .remove_attribute_by_symbol(id, symbol, &mut self.interp.resources)
            }
            NamespaceTarget::Scope(_) => unreachable!("scope targets returned above"),
        }
    }

    fn import_roots(&mut self) -> Result<Vec<String>, String> {
        let mut roots = self.state.temporary_import_paths.clone();
        if let Some(path) = self.state.sys_path {
            let Object::List(values) = self
                .state
                .heap
                .get(path.object_id().ok_or("sys.path lost list identity")?)?
            else {
                return Err("sys.path must remain a list".into());
            };
            for value in values.clone() {
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

    pub(super) fn import(&mut self, name: &str, bind_root: bool) -> Result<(), String> {
        let name = self.resolve_import_name(name)?;
        if let Some(module) = super::super::stdlib::native_module(&name) {
            return self.finish_import(
                &name,
                Value::Native(NativeValue::Module(module)),
                bind_root,
            );
        }
        if let Some(module) = self.state.modules.get(&name).cloned() {
            return self.finish_import(&name, module, bind_root);
        }

        self.ensure_package_parent(&name)?;
        // A package initializer may import the requested child itself. Reuse that exact module
        // object so classes and exceptions retain identity across the import cycle.
        if let Some(module) = self.state.modules.get(&name).copied() {
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
        let scope = self.state.heap.allocate_scope(
            None,
            false,
            Arc::from([]),
            HashMap::from([
                ("__name__".into(), module_name),
                ("__package__".into(), module_package),
                ("__file__".into(), module_file),
            ]),
            &mut self.interp.resources,
        )?;
        let module = self.allocate_object(Object::Module {
            name: name.clone(),
            scope,
        })?;
        self.state.modules.insert(name.clone(), module);

        let temporary_import_path = (!path.starts_with('<')).then(|| {
            path.rsplit_once('/')
                .map_or_else(|| "/".to_string(), |(parent, _)| parent.to_string())
        });
        if let Some(path) = &temporary_import_path {
            self.state.temporary_import_paths.insert(0, path.clone());
        }
        self.local_scopes.push(scope);
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
            .local_scopes
            .last()
            .copied()
            .ok_or("relative import requires a package context")?;
        let package = self
            .state
            .heap
            .scope_get(scope, "__package__")
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
            self.stack.pop().ok_or("parent import produced no value")?;
        }
        Ok(())
    }

    /// `from module import name` with the module on top of the stack: push the module attribute,
    /// else the submodule `module.name`, else raise CPython's `ImportError`.
    pub(super) fn import_from(&mut self, name: &str) -> Result<(), String> {
        let module = *self.stack.last().ok_or("import stack underflow")?;
        if let Some(value) = self.resolve_optional_attribute(module, name)? {
            self.stack.push(value);
            return Ok(());
        }
        let module_name = match module.native_value() {
            Some(NativeValue::Module(definition)) => definition.name.to_string(),
            _ => match module
                .object_id()
                .map(|id| self.state.heap.get(id))
                .transpose()?
            {
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
                let Some(Object::Module { scope, name }) = module
                    .object_id()
                    .map(|id| self.state.heap.get(id))
                    .transpose()?
                else {
                    return Err(self.missing_attribute(&module, "__all__"));
                };
                let (scope, module_name) = (*scope, name.clone());
                match self.state.heap.scope_get(scope, "__all__").copied() {
                    Some(all) => {
                        let items = match all
                            .object_id()
                            .map(|id| self.state.heap.get(id))
                            .transpose()?
                        {
                            Some(Object::List(items) | Object::Tuple(items)) => items.clone(),
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
                            let Some(name) = protocol::string_value(&self.state.heap, &item)?
                            else {
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
                        let mut names = self
                            .state
                            .heap
                            .scope_values(scope)?
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
            if let Some(scope) = self.local_scopes.last().copied() {
                self.state
                    .heap
                    .scope_insert(scope, name, value, &mut self.interp.resources)?;
            } else {
                let symbol = self
                    .state
                    .heap
                    .intern_symbol(&name, &mut self.interp.resources)?;
                self.state
                    .globals
                    .insert(symbol, value, &mut self.interp.resources)?;
            }
        }
        Ok(())
    }

    /// Install synthetic package parents for a dotted import and push the value selected by
    /// Python's ordinary import binding rule. Package objects contain only VM module references.
    fn finish_import(&mut self, name: &str, leaf: Value, bind_root: bool) -> Result<(), String> {
        if let Some((parent_name, child_name)) = name.rsplit_once('.') {
            if let Some(parent) = self.state.modules.get(parent_name).copied() {
                let parent_id = parent
                    .object_id()
                    .ok_or_else(|| format!("module {parent_name:?} cannot contain submodules"))?;
                let Object::Module { scope, .. } = self.state.heap.get(parent_id)? else {
                    return Err(format!("module {parent_name:?} changed object kind"));
                };
                self.state.heap.scope_insert(
                    *scope,
                    child_name.to_string(),
                    leaf,
                    &mut self.interp.resources,
                )?;
            }
        }
        if !bind_root || !name.contains('.') {
            self.stack.push(leaf);
            return Ok(());
        }
        let parts = name.split('.').collect::<Vec<_>>();
        let mut child = leaf;
        for parent_end in (1..parts.len()).rev() {
            let parent_name = parts[..parent_end].join(".");
            let child_name = parts[parent_end];
            let parent = if let Some(parent) = self.state.modules.get(&parent_name).copied() {
                let Some(parent_id) = parent.object_id() else {
                    return Err(format!("module {parent_name:?} cannot contain submodules"));
                };
                let Object::Module { scope, .. } = self.state.heap.get(parent_id)? else {
                    return Err(format!("module {parent_name:?} changed object kind"));
                };
                let scope = *scope;
                self.state.heap.scope_insert(
                    scope,
                    child_name.to_string(),
                    child,
                    &mut self.interp.resources,
                )?;
                parent
            } else {
                let module_name = self.allocate_string(parent_name.clone())?;
                let scope = self.state.heap.allocate_scope(
                    None,
                    false,
                    Arc::from([]),
                    HashMap::from([
                        ("__name__".into(), module_name),
                        (child_name.to_string(), child),
                    ]),
                    &mut self.interp.resources,
                )?;
                let parent = self.allocate_object(Object::Module {
                    name: parent_name.clone(),
                    scope,
                })?;
                self.state.modules.insert(parent_name, parent);
                parent
            };
            child = parent;
        }
        self.stack.push(child);
        Ok(())
    }
}
