//! Name, scope, and import operations used by bytecode execution.
//!
//! Lexical scopes are heap objects. The VM's scope stacks hold stored references to them, and
//! every operation here pins a scope it reads from those roots before it touches the heap, so
//! the scope stays live while a lookup or store allocates.

use super::super::heap::Ref;
use super::super::object_model::TypeId;
use super::super::scopes;
use super::{
    cpython_names, exception_types, string, Arc, BuiltinType, BytecodeFrame, ExceptionType, Flow,
    FrameEntry, HashMap, Linker, ModuleDef, NamespaceTarget, NativeValue, Object, ProxyTarget,
    PyModuleLoader, PyRuntime, SymbolId, Value, Vm, BUILTIN_FUNCTIONS,
};
use crate::python::error::{PyError, PyResult};

/// A [`NamespaceTarget`] whose references are pinned values, so it can be held across
/// allocation. Store it back into an object with [`NamespaceHandle::store`].
#[derive(Clone, Copy)]
pub(super) enum NamespaceHandle {
    Scope(Value),
    Repl,
    Instance(Value),
}

impl NamespaceHandle {
    /// The stored form, for an `Object::NamespaceDict` payload.
    pub(super) fn store(self) -> NamespaceTarget {
        match self {
            Self::Scope(scope) => NamespaceTarget::Scope(Ref::from(scope)),
            Self::Repl => NamespaceTarget::Repl,
            Self::Instance(instance) => NamespaceTarget::Instance(Ref::from(instance)),
        }
    }
}

/// A [`ProxyTarget`] whose references are pinned values.
#[derive(Clone, Copy)]
pub(super) enum ProxyHandle {
    Class(Value),
    RegisteredType(TypeId),
    NativeModule(&'static ModuleDef),
}

impl<'s> Vm<'s> {
    /// Pinned values for a namespace view's target, read while the view object is borrowed.
    pub(super) fn namespace_handle(&self, target: &NamespaceTarget) -> NamespaceHandle {
        match target {
            NamespaceTarget::Scope(scope) => NamespaceHandle::Scope(self.value(scope)),
            NamespaceTarget::Repl => NamespaceHandle::Repl,
            NamespaceTarget::Instance(instance) => NamespaceHandle::Instance(self.value(instance)),
        }
    }

    /// Pinned values for a mapping proxy's target, read while the proxy object is borrowed.
    pub(super) fn proxy_handle(&self, target: &ProxyTarget) -> ProxyHandle {
        match target {
            ProxyTarget::Class(class) => ProxyHandle::Class(self.value(class)),
            ProxyTarget::RegisteredType(type_id) => ProxyHandle::RegisteredType(*type_id),
            ProxyTarget::NativeModule(module) => ProxyHandle::NativeModule(module),
        }
    }

    /// The active frame's own heap scope, when its names live in one.
    pub(super) fn active_scope(&self) -> Option<Value> {
        self.value_optional(
            self.bytecode_frames
                .last()
                .and_then(BytecodeFrame::active_scope),
        )
    }

    /// The scope the active frame resolves free names through: its own scope, else the
    /// closure of the function it runs. `None` means the main program's global table.
    #[inline(always)]
    pub(super) fn lookup_scope(&self) -> Option<Value> {
        self.value_optional(
            self.bytecode_frames
                .last()
                .and_then(|frame| frame.scope.as_ref()),
        )
    }

    /// The class and receiver zero-argument `super()` and `__class__` refer to: the defining
    /// class of the function the nearest method frame runs, and that frame's first local.
    pub(super) fn method_context(&self) -> PyResult<Option<(Value, Value)>> {
        for frame in self.bytecode_frames.iter().rev() {
            let Some(callee) = &frame.callee else {
                continue;
            };
            let Object::Function(function) = self.get(self.value(callee))? else {
                continue;
            };
            let Some(class) = self.value_optional(function.defining_class.as_ref()) else {
                continue;
            };
            let receiver = match (frame.locals_base(), frame.active_scope()) {
                (Some(base), _) => {
                    self.value_optional(self.locals.get(base).and_then(Option::as_ref))
                }
                (None, Some(scope)) => scopes::local_ref(self.heap(), self.value(scope), 0)?
                    .map(|slot| self.value(slot)),
                (None, None) => None,
            };
            return Ok(receiver.map(|receiver| (class, receiver)));
        }
        Ok(None)
    }

    /// The scope `hops` lexical levels above the active frame's own level; `None` past the
    /// outermost scope. A frame without a scope of its own counts its closure as one hop.
    fn enclosing_scope(&self, hops: usize) -> PyResult<Option<Value>> {
        let frame = self
            .bytecode_frames
            .last()
            .ok_or("scope walk requires an active frame")?;
        let scope = self.value_optional(frame.scope.as_ref());
        let (mut target, hops) = match (frame.own_scope, hops) {
            (true, hops) => (scope, hops),
            (false, 0) => return Err("invalid enclosing scope hop count".into()),
            (false, hops) => (scope, hops - 1),
        };
        for _ in 0..hops {
            target = target
                .map(|scope| scopes::parent(self.heap(), scope))
                .transpose()?
                .flatten();
        }
        Ok(target)
    }

    pub(super) fn load_name(&mut self, symbol: SymbolId) -> PyResult<()> {
        if self.symbol_name(symbol) == "__class__" {
            let (class, _) = self
                .method_context()?
                .ok_or("__class__ is only defined inside a class method body")?;
            self.push(class);
            return Ok(());
        }
        let local_scope = self.lookup_scope();
        let scoped = match local_scope {
            Some(scope) => self.scope_get(scope, self.symbol_name(symbol))?,
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
        self.load_builtin(symbol)
    }

    #[inline(always)]
    pub(super) fn load_global(&mut self, symbol: SymbolId) -> PyResult<()> {
        if self.lookup_scope().is_none() {
            if let Some(value) = self.state.globals.get_ref(symbol) {
                self.execution.stack.push_ref(value);
                return Ok(());
            }
            return self.load_builtin(symbol);
        }
        self.load_scoped_global(symbol)
    }

    #[cold]
    #[inline(never)]
    fn load_scoped_global(&mut self, symbol: SymbolId) -> PyResult<()> {
        let scope = self.lookup_scope().expect("checked by load_global");
        let root = scopes::root(self.heap(), scope)?;
        let value = if scopes::uses_repl_globals(self.heap(), root)? {
            self.state.globals.get(&self.state.heap, symbol)
        } else {
            self.scope_get(root, self.symbol_name(symbol))?
        };
        if let Some(value) = value {
            self.push(value);
            return Ok(());
        }
        self.load_builtin(symbol)
    }

    fn load_builtin(&mut self, symbol: SymbolId) -> PyResult<()> {
        if let Some((module_name, attribute)) =
            super::super::stdlib::frozen_builtin(self.symbol_name(symbol))
        {
            self.import(module_name, false)?;
            let module = self.pop()?;
            let callable = self.resolve_attribute(module, attribute)?.ok_or_else(|| {
                format!("frozen module {module_name:?} does not define {attribute:?}")
            })?;
            self.push(callable);
            return Ok(());
        }
        let name = self.symbol_name(symbol);
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
                "property" => Some(BuiltinType::Property),
                "staticmethod" => Some(BuiltinType::StaticMethod),
                "classmethod" => Some(BuiltinType::ClassMethod),
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
                return Err(format!("builtin {name:?} is not implemented").into());
            }
            return Err(PyError::exception(
                "NameError",
                format!("name '{name}' is not defined"),
            ));
        };
        self.push(value);
        Ok(())
    }

    pub(super) fn store_name(&mut self, symbol: SymbolId) -> PyResult<()> {
        let value = self.pop()?;
        if let Some(scope) = self.active_scope() {
            self.scope_insert(scope, self.symbol_name(symbol).to_owned(), value)?;
        } else {
            self.state
                .globals
                .insert(symbol, value, &mut self.interp.resources)?;
        }
        Ok(())
    }

    pub(super) fn store_enclosing(&mut self, symbol: SymbolId, scope_hops: usize) -> PyResult<()> {
        let value = self.pop()?;
        if let Some(scope) = self.enclosing_scope(scope_hops)? {
            self.scope_insert(scope, self.symbol_name(symbol).to_owned(), value)
        } else {
            self.state
                .globals
                .insert(symbol, value, &mut self.interp.resources)?;
            Ok(())
        }
    }

    pub(super) fn store_nonlocal(&mut self, symbol: SymbolId) -> PyResult<()> {
        let value = self.pop()?;
        let start = self.enclosing_scope(1)?.ok_or_else(|| {
            format!(
                "no binding for nonlocal {:?} found",
                self.symbol_name(symbol)
            )
        })?;
        let state = &mut *self.state;
        scopes::store_nonlocal(&mut state.heap, start, state.symbols.issued(symbol), value)
    }

    pub(super) fn exception_type_matches(
        &mut self,
        expected: Value,
        actual: Value,
    ) -> PyResult<bool> {
        if let Some(NativeValue::ExceptionType(ExceptionType(name))) = expected.native_value() {
            let expected_type = self
                .state
                .types
                .exception_type_id(name)
                .ok_or("exception type is not registered")?;
            return self
                .state
                .types
                .is_subclass(self.type_id(&actual)?, expected_type);
        }
        if !expected.is_object() {
            return Err(
                "catching classes that do not inherit from BaseException is not allowed".into(),
            );
        }
        enum Expected {
            Class(TypeId),
            Tuple(Vec<Value>),
        }
        let expected = match self.get(expected)? {
            Object::Class(class_object) if class_object.exception_base.is_some() => {
                Expected::Class(class_object.instance_type)
            }
            Object::Tuple(types) => Expected::Tuple(self.values(types)),
            _ => {
                return Err(
                    "catching classes that do not inherit from BaseException is not allowed".into(),
                )
            }
        };
        match expected {
            Expected::Class(instance_type) => {
                let actual_type = self.type_id(&actual)?;
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

    #[inline(always)]
    pub(super) fn store_global(&mut self, symbol: SymbolId, value: Value) -> PyResult<()> {
        let Some(scope) = self.lookup_scope() else {
            self.state
                .globals
                .insert(symbol, value, &mut self.interp.resources)?;
            return Ok(());
        };
        self.store_scoped_global(scope, symbol, value)
    }

    #[cold]
    #[inline(never)]
    fn store_scoped_global(
        &mut self,
        scope: Value,
        symbol: SymbolId,
        value: Value,
    ) -> PyResult<()> {
        let root = scopes::root(self.heap(), scope)?;
        if scopes::uses_repl_globals(self.heap(), root)? {
            self.state
                .globals
                .insert(symbol, value, &mut self.interp.resources)?;
            Ok(())
        } else {
            self.scope_insert(root, self.symbol_name(symbol).to_owned(), value)
        }
    }

    pub(super) fn delete_global(&mut self, symbol: SymbolId) -> PyResult<()> {
        let Some(scope) = self.lookup_scope() else {
            self.state.globals.remove(&self.state.heap, symbol);
            return Ok(());
        };
        let root = scopes::root(self.heap(), scope)?;
        if scopes::uses_repl_globals(self.heap(), root)? {
            self.state.globals.remove(&self.state.heap, symbol);
        } else {
            let state = &mut *self.state;
            scopes::remove(&mut state.heap, root, state.symbols.issued(symbol))?;
        }
        Ok(())
    }

    /// The target a fresh `globals()` call should reference for the code currently running: an
    /// imported module's own scope, or the REPL/script's flat table when no scope roots the call
    /// (the entry-point program) or the root scope defers to it (a function or class body defined
    /// at that program's top level). Mirrors `load_global`'s and `store_global`'s resolution
    /// exactly, so `globals()` always names the same namespace a bare name lookup would.
    pub(super) fn current_globals_target(&self) -> PyResult<NamespaceHandle> {
        let Some(scope) = self.lookup_scope() else {
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
    pub(super) fn current_locals(&mut self) -> PyResult<Value> {
        // `exec`/`eval` code reports the locals of the frame that ran it.
        let Some(frame) = self
            .bytecode_frames
            .iter()
            .rev()
            .find(|frame| frame.own_scope || frame.locals_base.is_some())
        else {
            return self.alloc(Object::NamespaceDict(NamespaceTarget::Repl));
        };
        let Some(scope) = self.value_optional(frame.active_scope()) else {
            let base = frame
                .locals_base()
                .expect("frame without a scope was chosen for its slots");
            let names = frame.code.local_names.clone();
            let mut items = Vec::with_capacity(names.len());
            for (slot, name) in names.iter().enumerate() {
                if let Some(value) = self.locals.get(base + slot).and_then(Option::as_ref) {
                    let value = self.value(value);
                    items.push((self.allocate_string(name.clone())?, value));
                }
            }
            return self.allocate_dict(items);
        };
        // A module's own scope is the only one with no parent that does not defer to the
        // REPL/script table; function calls either have a lexical parent or defer to it.
        let is_module_scope = !scopes::uses_repl_globals(self.heap(), scope)?
            && scopes::parent(self.heap(), scope)?.is_none();
        if is_module_scope {
            return self.alloc(Object::NamespaceDict(NamespaceTarget::Scope(Ref::from(
                scope,
            ))));
        }
        self.namespace_snapshot_dict(NamespaceHandle::Scope(scope))
    }

    /// A detached `dict` holding `target`'s current bindings.
    pub(super) fn namespace_snapshot_dict(&mut self, target: NamespaceHandle) -> PyResult<Value> {
        let items = self.namespace_items(target)?;
        self.allocate_dict(items)
    }

    /// `target`'s bindings as `(key, value)` pairs with freshly allocated string keys, in the
    /// order [`Self::namespace_entries`] gives.
    pub(super) fn namespace_items(
        &mut self,
        target: NamespaceHandle,
    ) -> PyResult<Vec<(Value, Value)>> {
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
    pub(super) fn proxy_items(&mut self, target: ProxyHandle) -> PyResult<Vec<(Value, Value)>> {
        let mut entries: Vec<(String, Value)> = match target {
            ProxyHandle::Class(class) => {
                let Object::Class(class_object) = self.get(class)? else {
                    return Err("mappingproxy target is not a class".into());
                };
                class_object
                    .attributes
                    .iter()
                    .map(|(name, value)| (name.clone(), self.value(value)))
                    .collect()
            }
            ProxyHandle::RegisteredType(type_id) => self
                .state
                .types
                .get(type_id)?
                .attributes
                .iter()
                .map(|(name, value)| (name.clone(), self.value(value)))
                .collect(),
            ProxyHandle::NativeModule(module) => {
                let mut entries = Vec::with_capacity(module.functions.len() + module.values.len());
                for function in module.functions {
                    let value = Value::Native(NativeValue::NativeFunction(function));
                    entries.push((function.name.to_string(), value));
                }
                for definition in module.values {
                    let value = definition.get(self)?;
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
        target: NamespaceHandle,
    ) -> PyResult<Vec<(String, Value)>> {
        let mut entries: Vec<(String, Value)> = match target {
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
        target: NamespaceHandle,
        name: &str,
    ) -> PyResult<Option<Value>> {
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
        target: NamespaceHandle,
        name: String,
        value: Value,
    ) -> PyResult<()> {
        if let NamespaceHandle::Scope(scope) = target {
            return self.scope_insert(scope, name, value);
        }
        let symbol = self.intern_symbol(&name)?;
        match target {
            NamespaceHandle::Repl => {
                self.state
                    .globals
                    .insert(symbol, value, &mut self.interp.resources)
            }
            NamespaceHandle::Instance(instance) => {
                self.insert_attribute_by_symbol(instance, symbol, value)
            }
            NamespaceHandle::Scope(_) => unreachable!("scope targets returned above"),
        }
    }

    pub(super) fn namespace_delete(
        &mut self,
        target: NamespaceHandle,
        name: &str,
    ) -> PyResult<Option<Value>> {
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

    fn import_roots(&mut self) -> PyResult<Vec<String>> {
        let mut roots = self.state.temporary_import_paths.clone();
        if let Some(path) = self.value_optional(self.state.sys_path.as_ref()) {
            if !path.is_object() {
                return Err("sys.path lost list identity".into());
            }
            let Object::List(values) = self.get(path)? else {
                return Err("sys.path must remain a list".into());
            };
            for value in self.values(values) {
                let path = self
                    .string_value(&value)?
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
    pub(super) fn loaded_module(&self, name: &str) -> Option<Value> {
        self.value_optional(self.state.modules.get(name))
    }

    /// Allocate a module object over a fresh module scope binding `values`.
    fn allocate_module(
        &mut self,
        name: String,
        values: HashMap<String, Value>,
    ) -> PyResult<(Value, Value)> {
        let scope = self.alloc_scope(None, false, Arc::from([]), Vec::new(), values)?;
        let module = self.alloc(Object::Module {
            name,
            scope: Ref::from(scope),
        })?;
        Ok((module, scope))
    }

    /// Lex, parse and compile a module's source. The front end's working memory (tokens, syntax
    /// tree, pending instructions) is reserved from the source length and token count before it
    /// is allocated and released once the code exists, so a large module counts against the
    /// limit while it is compiled but not for the life of the program. The names and code
    /// objects the module keeps are charged to the symbol and code tables as they are compiled.
    pub(super) fn compile_module_source(
        &mut self,
        source: &str,
        path: &str,
    ) -> PyResult<super::super::bytecode::CodeRef> {
        let parse_memory = super::super::source::front_end_memory(source.len())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or("module source is too large")?;
        if !self.interp.resources.reserve_memory(parse_memory)
            || !self.interp.resources.charge_cpu(source.len() as u64)
        {
            return Err(PyError::resource_error(
                "resource limit exceeded while importing module",
            ));
        }
        let result = self.compile_module_tokens(source, path);
        self.interp.resources.release_memory(parse_memory);
        result
    }

    fn compile_module_tokens(
        &mut self,
        source: &str,
        path: &str,
    ) -> PyResult<super::super::bytecode::CodeRef> {
        // As for `exec`, a front-end error stays a fault rather than a catchable `SyntaxError`.
        let syntax_error = |message: &str, span: &super::super::source::Span| {
            PyError::unsupported(format!(
                "{message} in {path} at line {}, column {}",
                span.line, span.column
            ))
        };
        let tokens = super::super::lexer::lex(source)
            .map_err(|error| syntax_error(&error.message, &error.span))?;
        let token_memory = super::super::source::token_memory(tokens.len())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or("module source is too large")?;
        if !self.interp.resources.reserve_memory(token_memory) {
            return Err(PyError::resource_error(
                "resource limit exceeded while importing module",
            ));
        }
        let result = super::super::parser::parse(tokens)
            .map_err(|error| syntax_error(&error.message, &error.span))
            .and_then(|program| self.link(|link| super::super::compiler::compile(program, link)));
        self.interp.resources.release_memory(token_memory);
        result
    }

    /// Run `compile` against this interpreter's symbol and code tables. Fails when either
    /// could not charge what the new code needs; the code must not run then.
    pub(super) fn link<R>(&mut self, compile: impl FnOnce(&mut Linker) -> R) -> PyResult<R> {
        let state = &mut *self.state;
        let mut link = Linker::new(
            &mut state.symbols,
            &mut state.codes,
            &mut self.interp.resources,
        );
        let compiled = compile(&mut link);
        link.finish()?;
        Ok(compiled)
    }

    pub(super) fn import(&mut self, name: &str, bind_root: bool) -> PyResult<()> {
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
            self.interp.load_module_source(&roots, &name)?
        };
        let Some((path, source)) = source else {
            // A standard package that shellsim does not provide at all is unsupported. A missing
            // submodule of a provided package is absent, like a missing module attribute.
            let top_level = name.split('.').next().unwrap_or(&name);
            let provided = super::super::stdlib::native_module(top_level).is_some()
                || super::super::stdlib::frozen_module(top_level).is_some()
                || self.state.modules.contains_key(top_level);
            if !provided && cpython_names::is_cpython_stdlib_module(top_level) {
                return Err(format!("standard-library module {name:?} is not implemented").into());
            }
            return Err(PyError::exception(
                "ModuleNotFoundError",
                format!("No module named '{name}'"),
            ));
        };
        let code = self.compile_module_source(&source, &path)?;
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
        let module_doc = match code.docstring.clone() {
            Some(text) => self.allocate_string(text.to_string())?,
            None => Value::None,
        };
        let (module, scope) = self.allocate_module(
            name.clone(),
            HashMap::from([
                ("__name__".into(), module_name),
                ("__package__".into(), module_package),
                ("__file__".into(), module_file),
                ("__doc__".into(), module_doc),
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
        let execution = self.execute_code(&code, FrameEntry::scoped(scope));
        if temporary_import_path.is_some() {
            self.state.temporary_import_paths.remove(0);
        }
        match execution {
            Ok(Flow::Halt) => self.finish_import(&name, module, bind_root),
            Ok(Flow::Return(_)) => {
                self.state.modules.remove(&name);
                Err(format!("'return' outside function in module {name:?}").into())
            }
            Ok(Flow::Yield(_)) => {
                self.state.modules.remove(&name);
                Err(format!("'yield' outside function in module {name:?}").into())
            }
            Ok(Flow::Exit(status)) => {
                self.state.modules.remove(&name);
                Err(format!("module {name:?} exited with status {status}").into())
            }
            Ok(flow) => unreachable!("a module body cannot end with {flow:?}"),
            Err((error, span)) => {
                self.state.modules.remove(&name);
                Err(error.located(|| {
                    format!(" in {path} at line {}, column {}", span.line, span.column)
                }))
            }
        }
    }

    fn resolve_import_name(&self, requested: &str) -> PyResult<String> {
        let level = requested
            .chars()
            .take_while(|character| *character == '.')
            .count();
        if level == 0 {
            return Ok(requested.to_string());
        }
        let scope = self
            .lookup_scope()
            .ok_or("relative import requires a package context")?;
        let package = self
            .scope_get(scope, "__package__")?
            .ok_or("relative import requires __package__")?;
        let package = string::string_value(&self.state.heap, package)?
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

    fn ensure_package_parent(&mut self, name: &str) -> PyResult<()> {
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
                .load_module_source(&roots, parent)?
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
    pub(super) fn import_from(&mut self, name: &str) -> PyResult<()> {
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
        let Err(error) = self.import(&format!("{module_name}.{name}"), false) else {
            return Ok(());
        };
        self.catch(error, "ModuleNotFoundError")?;
        let message =
            format!("cannot import name '{name}' from '{module_name}' (unknown location)");
        Err(PyError::exception("ImportError", message))
    }

    /// `from module import *` with the module on top of the stack: bind every name in the
    /// module's `__all__`, or else every name without a leading underscore, in the current scope.
    pub(super) fn import_star(&mut self) -> PyResult<()> {
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
                let (scope, module_name) = (self.value(scope), name.clone());
                match self.scope_get(scope, "__all__")? {
                    Some(all) => {
                        let items = match all.is_object().then(|| self.get(all)).transpose()? {
                            Some(Object::List(items) | Object::Tuple(items)) => self.values(items),
                            // CPython indexes `__all__`; lists and tuples are what modules use.
                            _ => {
                                let message = format!(
                                    "'{}' object does not support indexing",
                                    self.type_name_of(&all)?
                                );
                                return Err(PyError::exception("TypeError", message));
                            }
                        };
                        let mut names = Vec::with_capacity(items.len());
                        for item in items {
                            let Some(name) = string::string_value(&self.state.heap, item)? else {
                                let message = format!(
                                    "Item in {module_name}.__all__ must be str, not {}",
                                    self.type_name_of(&item)?
                                );
                                return Err(PyError::exception("TypeError", message));
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
            if let Some(scope) = self.active_scope() {
                self.scope_insert(scope, name, value)?;
            } else {
                let symbol = self.intern_symbol(&name)?;
                self.state
                    .globals
                    .insert(symbol, value, &mut self.interp.resources)?;
            }
        }
        Ok(())
    }

    /// The scope of the module imported as `name`, for binding a submodule into it.
    pub(super) fn module_scope(&self, name: &str, module: Value) -> PyResult<Value> {
        if !module.is_object() {
            return Err(format!("module {name:?} cannot contain submodules").into());
        }
        let Object::Module { scope, .. } = self.get(module)? else {
            return Err(format!("module {name:?} changed object kind").into());
        };
        Ok(self.value(scope))
    }

    /// Install synthetic package parents for a dotted import and push the value selected by
    /// Python's ordinary import binding rule. Package objects contain only VM module references.
    fn finish_import(&mut self, name: &str, leaf: Value, bind_root: bool) -> PyResult<()> {
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
