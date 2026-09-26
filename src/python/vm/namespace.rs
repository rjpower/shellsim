//! Name, scope, import, and per-code cache operations used by bytecode execution.

use super::{
    cpython_names, exception_types, protocol, Arc, Builtin, BuiltinType, CodeRef, ExceptionType,
    Execution, HashMap, NameId, NativeValue, Object, PyModuleLoader, PyRuntime, RaisedException,
    ScopeId, SymbolId, Value, Vm,
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
                _ => None,
            };
            if let Some(builtin_type) = builtin_type {
                return Some(Value::Native(NativeValue::BuiltinType(builtin_type)));
            }
            if let Some(function) = super::super::stdlib::core::builtin_function(name) {
                return Some(Value::Native(NativeValue::NativeFunction(function)));
            }
            let builtin = match name {
                "print" => Builtin::Print,
                "input" => Builtin::Input,
                "exec" => Builtin::Exec,
                "exit" | "quit" => Builtin::Exit,
                "chr" => Builtin::Character,
                "ord" => Builtin::Ordinal,
                "bin" => Builtin::Binary,
                "oct" => Builtin::Octal,
                "hex" => Builtin::Hexadecimal,
                "repr" => Builtin::Repr,
                "dir" => Builtin::Dir,
                "isinstance" => Builtin::IsInstance,
                "issubclass" => Builtin::IsSubclass,
                "len" => Builtin::Length,
                "sorted" => Builtin::Sorted,
                "min" => Builtin::Minimum,
                "max" => Builtin::Maximum,
                "sum" => Builtin::Sum,
                "abs" => Builtin::Absolute,
                "pow" => Builtin::Power,
                "divmod" => Builtin::Divmod,
                "callable" => Builtin::Callable,
                "range" => Builtin::Range,
                "enumerate" => Builtin::Enumerate,
                "zip" => Builtin::Zip,
                "any" => Builtin::Any,
                "all" => Builtin::All,
                "iter" => Builtin::Iter,
                "next" => Builtin::Next,
                "property" => Builtin::Property,
                "staticmethod" => Builtin::StaticMethod,
                "classmethod" => Builtin::ClassMethod,
                "super" => Builtin::Super,
                _ => {
                    return exception_types::exception_type(name)
                        .filter(|definition| definition.builtin)
                        .map(|definition| {
                            Value::Native(NativeValue::ExceptionType(ExceptionType(
                                definition.name,
                            )))
                        })
                }
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
            let kind = match self.user_exception_base(&actual.value)? {
                Some(base) => base,
                None => actual.kind.as_str(),
            };
            return Ok(exception_types::exception_is_subclass(kind, name));
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

        let source = if let Some(source) = super::super::stdlib::frozen_module(&name) {
            Some((format!("<frozen {name}>"), source.to_string()))
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
        let is_package = path.ends_with("/__init__.py");
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
        if let Some(value) = self.resolve_attribute(module, name)? {
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
