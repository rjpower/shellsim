//! Arena-backed mutable Python objects.
//!
//! IDs make aliasing explicit without pervasive interior mutability. Collection happens only at
//! VM instruction boundaries, where every live `Value` and lexical scope is available as a root.
//! A traced free list handles cycles while preserving the compact, copyable `Value` contract.

use crate::resources::Resources;
use num_bigint::BigInt;
use std::collections::HashMap;

use super::bytecode::Code;
use super::native::{PyArgumentSpec, PyArrayDtype, PyArrayLayout};
use super::object_model::{BuiltinType, TypeId};
use super::Value;

/// Storage layout inherited by user-defined classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassLayout {
    Object,
    Int,
    Type,
}

/// Type-erased payload carried by an instance while preserving its user-defined class identity.
#[derive(Clone, Debug)]
pub enum InstancePayload {
    Object,
    Int(i64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectId(usize);

impl ObjectId {
    pub(super) const fn from_raw(value: usize) -> Self {
        Self(value)
    }

    pub(super) const fn as_raw(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScopeId(usize);

#[derive(Clone, Debug)]
pub enum Object {
    String(String),
    Bytes(Vec<u8>),
    ByteArray(Vec<u8>),
    Exception {
        kind: String,
        message: String,
    },
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Slice {
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    },
    Dict(Vec<(Value, Value)>),
    DefaultDict {
        factory: Value,
        entries: Vec<(Value, Value)>,
    },
    Set(Vec<Value>),
    BigInt(BigInt),
    Function {
        name: String,
        code: Code,
        closure: Option<ScopeId>,
        defaults: Vec<Value>,
        /// Class captured when this function is installed by a class body.
        defining_class: Option<ObjectId>,
    },
    Class {
        /// Semantic type identity used by instances of this class.
        instance_type: TypeId,
        name: String,
        /// Direct user-defined bases in source order.
        bases: Vec<ObjectId>,
        /// C3-linearized user-defined ancestors, excluding this class.
        mro: Vec<ObjectId>,
        /// The callable type object responsible for this class.
        metaclass: Value,
        layout: ClassLayout,
        attributes: HashMap<String, Value>,
        is_dataclass: bool,
        dataclass_fields: Vec<(String, Option<Value>)>,
        enum_members: Vec<Value>,
    },
    Instance {
        class: ObjectId,
        payload: InstancePayload,
    },
    EnumMember {
        name: String,
        value: Value,
    },
    DescriptorBoundMethod {
        receiver: Value,
        descriptor: Value,
        owner: Option<ObjectId>,
    },
    Iterator {
        values: Vec<Value>,
        position: usize,
    },
    /// An infinite arithmetic iterator kept lazy so it cannot allocate without a caller bound.
    CountIterator {
        current: i64,
        step: i64,
    },
    /// The two-argument `iter(callable, sentinel)` form. Calls remain lazy so an unbounded
    /// producer is still governed by the VM's ordinary instruction budget.
    CallableIterator {
        callable: Value,
        sentinel: Value,
        exhausted: bool,
    },
    /// A suspended Python generator frame. The bytecode is immutable; the instruction pointer,
    /// exception-handler stack, and lexical scope are the complete resumable state.
    Generator {
        name: String,
        code: Code,
        scope: ScopeId,
        instruction_pointer: usize,
        handlers: Vec<(usize, usize)>,
        stack: Vec<Value>,
        exhausted: bool,
        running: bool,
    },
    Module {
        name: String,
        scope: ScopeId,
    },
    /// Flat type-erased storage shared by one or more array views.
    ArrayStorage(Vec<Value>),
    /// An ndarray view. Indices are mapped into `ArrayStorage` by the layout.
    Array {
        storage: ObjectId,
        layout: PyArrayLayout,
        dtype: PyArrayDtype,
    },
    /// A compiled regular expression.  The pattern is compiled at the operation boundary so
    /// regex execution never gets a host capability; keeping the source and flags here also
    /// makes the object cheap to clone and deterministic to inspect.
    Regex {
        pattern: String,
        flags: u32,
    },
    /// A bounded regular-expression match result.  Captures are stored as owned text rather than
    /// references into a host regex object, which keeps the arena self-contained.
    Match {
        text: String,
        groups: Vec<Option<String>>,
        start: usize,
        end: usize,
    },
    ArgumentParser {
        prog: String,
        description: Option<String>,
        add_help: bool,
        is_subcommand: bool,
        arguments: Vec<PyArgumentSpec>,
        subparsers: Option<super::native::PySubparsersSpec>,
    },
    Namespace {
        values: Vec<(String, Value)>,
    },
    RaisesContext {
        expected: String,
    },
    Property {
        getter: Value,
        setter: Option<Value>,
    },
    StaticMethod {
        callable: Value,
    },
    ClassMethod {
        callable: Value,
    },
    Super {
        start_class: ObjectId,
        receiver: Value,
    },
}

#[derive(Clone, Default, Debug)]
pub struct Heap {
    objects: Vec<Option<HeapObject>>,
    free_objects: Vec<usize>,
    scopes: Vec<Option<Scope>>,
    free_scopes: Vec<usize>,
    modeled_bytes: u64,
    bytes_since_collection: u64,
}

/// Common header shared by every arena-backed Python object.
#[derive(Clone, Debug)]
struct HeapObject {
    type_id: TypeId,
    attributes: HashMap<String, Value>,
    payload: Object,
    string_is_ascii: Option<bool>,
    modeled_bytes: u64,
}

#[derive(Clone, Debug)]
struct Scope {
    parent: Option<ScopeId>,
    uses_repl_globals: bool,
    values: HashMap<String, Value>,
    modeled_bytes: u64,
}

impl Heap {
    pub fn get(&self, id: ObjectId) -> Result<&Object, String> {
        self.objects
            .get(id.0)
            .and_then(Option::as_ref)
            .map(|object| &object.payload)
            .ok_or_else(|| "invalid object reference".into())
    }

    pub fn get_mut(&mut self, id: ObjectId) -> Result<&mut Object, String> {
        self.objects
            .get_mut(id.0)
            .and_then(Option::as_mut)
            .map(|object| &mut object.payload)
            .ok_or_else(|| "invalid object reference".into())
    }

    /// Return the cached ASCII property of an immutable string payload.
    pub fn string_is_ascii(&self, id: ObjectId) -> Result<Option<bool>, String> {
        self.objects
            .get(id.0)
            .and_then(Option::as_ref)
            .map(|object| object.string_is_ascii)
            .ok_or_else(|| "invalid object reference".into())
    }

    /// Replace an object payload while charging growth before installing it and releasing shrink
    /// after the old payload is no longer live. The object's identity and instance attributes are
    /// preserved.
    pub fn replace_payload(
        &mut self,
        id: ObjectId,
        payload: Object,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let next_bytes = modeled_size(&payload)?;
        let current_bytes = {
            let current = self
                .objects
                .get(id.0)
                .and_then(Option::as_ref)
                .ok_or("invalid object reference")?;
            modeled_size(&current.payload)?
        };
        let next_type = self.infer_type_id(&payload)?;
        if next_bytes > current_bytes {
            self.reserve_object_growth(id, next_bytes - current_bytes, resources)?;
        }
        let string_is_ascii = match &payload {
            Object::String(value) => Some(value.is_ascii()),
            _ => None,
        };
        let object = self
            .objects
            .get_mut(id.0)
            .and_then(Option::as_mut)
            .ok_or("invalid object reference")?;
        object.type_id = next_type;
        object.payload = payload;
        object.string_is_ascii = string_is_ascii;
        if current_bytes > next_bytes {
            let released = current_bytes - next_bytes;
            object.modeled_bytes = object.modeled_bytes.saturating_sub(released);
            self.modeled_bytes = self.modeled_bytes.saturating_sub(released);
            resources.release_memory(released);
        }
        Ok(())
    }

    pub fn type_id(&self, id: ObjectId) -> Result<TypeId, String> {
        self.objects
            .get(id.0)
            .and_then(Option::as_ref)
            .map(|object| object.type_id)
            .ok_or_else(|| "invalid object reference".into())
    }

    pub fn attribute(&self, id: ObjectId, name: &str) -> Result<Option<&Value>, String> {
        Ok(self
            .objects
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or("invalid object reference")?
            .attributes
            .get(name))
    }

    pub fn has_attribute(&self, id: ObjectId, name: &str) -> Result<bool, String> {
        Ok(self
            .objects
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or("invalid object reference")?
            .attributes
            .contains_key(name))
    }

    pub fn insert_attribute(
        &mut self,
        id: ObjectId,
        name: String,
        value: Value,
    ) -> Result<(), String> {
        self.objects
            .get_mut(id.0)
            .and_then(Option::as_mut)
            .ok_or("invalid object reference")?
            .attributes
            .insert(name, value);
        Ok(())
    }

    pub fn extend_attributes(
        &mut self,
        id: ObjectId,
        values: impl IntoIterator<Item = (String, Value)>,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let values = values.into_iter().collect::<Vec<_>>();
        let object = self
            .objects
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or("invalid object reference")?;
        let new_attributes = values
            .iter()
            .filter(|(name, _)| !object.attributes.contains_key(name))
            .count();
        let growth = u64::try_from(new_attributes)
            .ok()
            .and_then(|count| count.checked_mul(48))
            .ok_or("modeled attribute size overflow")?;
        self.reserve_object_growth(id, growth, resources)?;
        self.objects
            .get_mut(id.0)
            .and_then(Option::as_mut)
            .ok_or("invalid object reference")?
            .attributes
            .extend(values);
        Ok(())
    }

    /// Return whether enough allocation has occurred to justify tracing the arena.
    pub fn should_collect(&self, memory_limit: u64, memory_remaining: u64) -> bool {
        // Scan at most about sixteen times while approaching a limit. Small test environments
        // still collect promptly, while ordinary 64-256 MiB runs avoid scanning the live graph
        // for every megabyte of temporary allocation.
        let allocation_interval = (memory_limit / 16).clamp(64 * 1024, 16 * 1024 * 1024);
        self.bytes_since_collection >= allocation_interval
            || (self.bytes_since_collection != 0 && memory_remaining < allocation_interval)
    }

    /// Transfer all live heap accounting to the caller when an interpreter is discarded.
    pub fn take_modeled_bytes(&mut self) -> u64 {
        self.bytes_since_collection = 0;
        std::mem::take(&mut self.modeled_bytes)
    }

    pub fn allocate_scope(
        &mut self,
        parent: Option<ScopeId>,
        uses_repl_globals: bool,
        values: HashMap<String, Value>,
        resources: &mut Resources,
    ) -> Result<ScopeId, String> {
        let slots = u64::try_from(values.len()).map_err(|_| "modeled scope size overflow")?;
        let bytes = 32u64
            .checked_add(slots.checked_mul(48).ok_or("modeled scope size overflow")?)
            .ok_or("modeled scope size overflow")?;
        self.reserve_growth(bytes, resources)?;
        let scope = Scope {
            parent,
            uses_repl_globals,
            values,
            modeled_bytes: bytes,
        };
        let id = if let Some(index) = self.free_scopes.pop() {
            self.scopes[index] = Some(scope);
            ScopeId(index)
        } else {
            let id = ScopeId(self.scopes.len());
            self.scopes.push(Some(scope));
            id
        };
        Ok(id)
    }

    pub fn scope_get(&self, mut scope: ScopeId, name: &str) -> Option<&Value> {
        loop {
            let current = self.scopes.get(scope.0)?.as_ref()?;
            if let Some(value) = current.values.get(name) {
                return Some(value);
            }
            scope = current.parent?;
        }
    }

    pub fn scope_uses_repl_globals(&self, mut scope: ScopeId) -> Result<bool, String> {
        loop {
            let current = self
                .scopes
                .get(scope.0)
                .and_then(Option::as_ref)
                .ok_or("invalid scope reference")?;
            if let Some(parent) = current.parent {
                scope = parent;
            } else {
                return Ok(current.uses_repl_globals);
            }
        }
    }

    pub fn scope_root(&self, mut scope: ScopeId) -> Result<ScopeId, String> {
        loop {
            let current = self
                .scopes
                .get(scope.0)
                .and_then(Option::as_ref)
                .ok_or("invalid scope reference")?;
            if let Some(parent) = current.parent {
                scope = parent;
            } else {
                return Ok(scope);
            }
        }
    }

    pub fn scope_parent(&self, scope: ScopeId) -> Result<Option<ScopeId>, String> {
        Ok(self
            .scopes
            .get(scope.0)
            .and_then(Option::as_ref)
            .ok_or("invalid scope reference")?
            .parent)
    }

    pub fn scope_values(&self, scope: ScopeId) -> Result<HashMap<String, Value>, String> {
        Ok(self
            .scopes
            .get(scope.0)
            .and_then(Option::as_ref)
            .ok_or("invalid scope reference")?
            .values
            .clone())
    }

    pub fn scope_insert(
        &mut self,
        scope: ScopeId,
        name: String,
        value: Value,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let is_new = !self
            .scopes
            .get(scope.0)
            .and_then(Option::as_ref)
            .ok_or("invalid scope reference")?
            .values
            .contains_key(&name);
        if is_new {
            self.reserve_growth(48, resources)?;
            let scope = self
                .scopes
                .get_mut(scope.0)
                .and_then(Option::as_mut)
                .ok_or("invalid scope reference")?;
            scope.modeled_bytes = scope
                .modeled_bytes
                .checked_add(48)
                .ok_or("modeled scope size overflow")?;
        }
        self.scopes
            .get_mut(scope.0)
            .and_then(Option::as_mut)
            .ok_or("invalid scope reference")?
            .values
            .insert(name, value);
        Ok(())
    }

    pub fn scope_store_nonlocal(
        &mut self,
        scope: ScopeId,
        name: &str,
        value: Value,
    ) -> Result<(), String> {
        let mut current = self
            .scopes
            .get(scope.0)
            .and_then(Option::as_ref)
            .ok_or("invalid scope reference")?
            .parent
            .ok_or_else(|| format!("no binding for nonlocal {name:?} found"))?;
        loop {
            let parent = self
                .scopes
                .get(current.0)
                .and_then(Option::as_ref)
                .ok_or("invalid scope reference")?
                .parent;
            if self.scopes[current.0]
                .as_ref()
                .expect("validated scope")
                .values
                .contains_key(name)
            {
                self.scopes[current.0]
                    .as_mut()
                    .expect("validated scope")
                    .values
                    .insert(name.to_string(), value);
                return Ok(());
            }
            current = parent.ok_or_else(|| format!("no binding for nonlocal {name:?} found"))?;
        }
    }

    pub fn scope_remove(&mut self, scope: ScopeId, name: &str) -> Result<Option<Value>, String> {
        Ok(self
            .scopes
            .get_mut(scope.0)
            .and_then(Option::as_mut)
            .ok_or("invalid scope reference")?
            .values
            .remove(name))
    }

    pub fn allocate(&mut self, object: Object, resources: &mut Resources) -> Result<Value, String> {
        let bytes = modeled_size(&object)?;
        if !resources.reserve_memory(bytes) {
            return Err("memory limit exceeded".into());
        }
        self.modeled_bytes = self
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled heap size overflow")?;
        self.bytes_since_collection = self.bytes_since_collection.saturating_add(bytes);
        let type_id = self.infer_type_id(&object)?;
        let string_is_ascii = match &object {
            Object::String(value) => Some(value.is_ascii()),
            _ => None,
        };
        let object = HeapObject {
            type_id,
            attributes: HashMap::new(),
            payload: object,
            string_is_ascii,
            modeled_bytes: bytes,
        };
        let id = if let Some(index) = self.free_objects.pop() {
            self.objects[index] = Some(object);
            ObjectId(index)
        } else {
            let id = ObjectId(self.objects.len());
            self.objects.push(Some(object));
            id
        };
        Ok(Value::Object(id))
    }

    fn infer_type_id(&self, object: &Object) -> Result<TypeId, String> {
        Ok(match object {
            Object::String(_) => BuiltinType::String.id(),
            Object::Bytes(_) => BuiltinType::Bytes.id(),
            Object::ByteArray(_) => BuiltinType::ByteArray.id(),
            Object::Exception { .. } => BuiltinType::Exception.id(),
            Object::List(_) => BuiltinType::List.id(),
            Object::Tuple(_) => BuiltinType::Tuple.id(),
            Object::Slice { .. } => BuiltinType::Native.id(),
            Object::Dict(_) | Object::DefaultDict { .. } => BuiltinType::Dict.id(),
            Object::Set(_) => BuiltinType::Set.id(),
            Object::BigInt(_) => BuiltinType::Int.id(),
            Object::Function { .. } | Object::DescriptorBoundMethod { .. } => {
                BuiltinType::Function.id()
            }
            Object::Class { metaclass, .. } => match metaclass.native_value() {
                Some(super::vm::NativeValue::BuiltinType(builtin)) => builtin.id(),
                _ if metaclass.object_id().is_some() => {
                    match self.get(metaclass.object_id().expect("checked above"))? {
                        Object::Class { instance_type, .. } => *instance_type,
                        _ => return Err("class metaclass is not a class".into()),
                    }
                }
                _ => return Err("class has an invalid metaclass".into()),
            },
            Object::Instance { class, .. } => match self.get(*class)? {
                Object::Class { instance_type, .. } => *instance_type,
                _ => return Err("instance class is not a class".into()),
            },
            Object::Iterator { .. }
            | Object::CountIterator { .. }
            | Object::CallableIterator { .. } => BuiltinType::Iterator.id(),
            Object::Generator { .. } => BuiltinType::Generator.id(),
            Object::Module { .. } => BuiltinType::Module.id(),
            Object::ArrayStorage(_) => BuiltinType::Native.id(),
            Object::Array { .. } => BuiltinType::Array.id(),
            Object::Regex { .. } => BuiltinType::Regex.id(),
            Object::Match { .. } => BuiltinType::Match.id(),
            Object::ArgumentParser { .. } => BuiltinType::ArgumentParser.id(),
            Object::RaisesContext { .. } => BuiltinType::RaisesContext.id(),
            Object::EnumMember { .. } | Object::Namespace { .. } => BuiltinType::Native.id(),
            Object::Property { .. } => BuiltinType::Property.id(),
            Object::StaticMethod { .. } | Object::ClassMethod { .. } | Object::Super { .. } => {
                BuiltinType::Native.id()
            }
        })
    }

    pub fn reserve_object_growth(
        &mut self,
        id: ObjectId,
        bytes: u64,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let next_heap_bytes = self
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled heap size overflow")?;
        let next_object_bytes = self
            .objects
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or("invalid object reference")?
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled object size overflow")?;
        if !resources.reserve_memory(bytes) {
            return Err("memory limit exceeded".into());
        }
        self.modeled_bytes = next_heap_bytes;
        self.bytes_since_collection = self.bytes_since_collection.saturating_add(bytes);
        self.objects[id.0]
            .as_mut()
            .expect("object was validated before reserving memory")
            .modeled_bytes = next_object_bytes;
        Ok(())
    }

    fn reserve_growth(&mut self, bytes: u64, resources: &mut Resources) -> Result<(), String> {
        let next_heap_bytes = self
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled heap size overflow")?;
        if !resources.reserve_memory(bytes) {
            return Err("memory limit exceeded".into());
        }
        self.modeled_bytes = next_heap_bytes;
        self.bytes_since_collection = self.bytes_since_collection.saturating_add(bytes);
        Ok(())
    }

    /// Reclaim objects and lexical scopes unreachable from a complete VM safe-point root set.
    ///
    /// The collector deliberately does not run from `allocate`: native helpers may temporarily
    /// hold values in Rust locals that are not VM roots. Callers invoke this only between bytecode
    /// instructions, when all Python-visible state has returned to the arena or resumable VM.
    pub fn collect(
        &mut self,
        value_roots: &[Value],
        scope_roots: &[ScopeId],
        resources: &mut Resources,
    ) -> Result<u64, String> {
        let mut marked_objects = vec![false; self.objects.len()];
        let mut marked_scopes = vec![false; self.scopes.len()];
        let mut object_work = Vec::new();
        let mut scope_work = scope_roots.to_vec();
        for value in value_roots {
            if let Some(id) = value.object_id() {
                object_work.push(id);
            }
        }

        while !object_work.is_empty() || !scope_work.is_empty() {
            while let Some(id) = object_work.pop() {
                let marked = marked_objects
                    .get_mut(id.0)
                    .ok_or("invalid object reference during collection")?;
                if *marked {
                    continue;
                }
                let object = self
                    .objects
                    .get(id.0)
                    .and_then(Option::as_ref)
                    .ok_or("invalid object reference during collection")?;
                *marked = true;
                trace_object(object, &mut object_work, &mut scope_work);
            }

            while let Some(id) = scope_work.pop() {
                let marked = marked_scopes
                    .get_mut(id.0)
                    .ok_or("invalid scope reference during collection")?;
                if *marked {
                    continue;
                }
                let scope = self
                    .scopes
                    .get(id.0)
                    .and_then(Option::as_ref)
                    .ok_or("invalid scope reference during collection")?;
                *marked = true;
                if let Some(parent) = scope.parent {
                    scope_work.push(parent);
                }
                for value in scope.values.values() {
                    if let Some(id) = value.object_id() {
                        object_work.push(id);
                    }
                }
            }
        }

        let mut released = 0u64;
        for (index, slot) in self.objects.iter_mut().enumerate() {
            if !marked_objects[index] {
                if let Some(object) = slot.take() {
                    released = released.saturating_add(object.modeled_bytes);
                    self.free_objects.push(index);
                }
            }
        }
        for (index, slot) in self.scopes.iter_mut().enumerate() {
            if !marked_scopes[index] {
                if let Some(scope) = slot.take() {
                    released = released.saturating_add(scope.modeled_bytes);
                    self.free_scopes.push(index);
                }
            }
        }
        self.modeled_bytes = self.modeled_bytes.saturating_sub(released);
        self.bytes_since_collection = 0;
        resources.release_memory(released);
        Ok(released)
    }
}

fn trace_value(value: Value, object_work: &mut Vec<ObjectId>) {
    if let Some(id) = value.object_id() {
        object_work.push(id);
    }
}

fn trace_values(values: impl IntoIterator<Item = Value>, object_work: &mut Vec<ObjectId>) {
    for value in values {
        trace_value(value, object_work);
    }
}

fn trace_object(
    object: &HeapObject,
    object_work: &mut Vec<ObjectId>,
    scope_work: &mut Vec<ScopeId>,
) {
    trace_values(object.attributes.values().copied(), object_work);
    match &object.payload {
        Object::List(items)
        | Object::Tuple(items)
        | Object::Set(items)
        | Object::ArrayStorage(items) => trace_values(items.iter().copied(), object_work),
        Object::Dict(entries) => {
            for (key, value) in entries {
                trace_values([*key, *value], object_work);
            }
        }
        Object::DefaultDict { factory, entries } => {
            trace_value(*factory, object_work);
            for (key, value) in entries {
                trace_values([*key, *value], object_work);
            }
        }
        Object::Function {
            closure,
            defaults,
            defining_class,
            ..
        } => {
            trace_values(defaults.iter().copied(), object_work);
            scope_work.extend(*closure);
            object_work.extend(*defining_class);
        }
        Object::Class {
            bases,
            mro,
            metaclass,
            attributes,
            dataclass_fields,
            enum_members,
            ..
        } => {
            object_work.extend(bases.iter().copied());
            object_work.extend(mro.iter().copied());
            trace_value(*metaclass, object_work);
            trace_values(attributes.values().copied(), object_work);
            trace_values(
                dataclass_fields.iter().filter_map(|(_, value)| *value),
                object_work,
            );
            trace_values(enum_members.iter().copied(), object_work);
        }
        Object::Instance { class, .. } => object_work.push(*class),
        Object::EnumMember { value, .. } => trace_value(*value, object_work),
        Object::DescriptorBoundMethod {
            receiver,
            descriptor,
            owner,
        } => {
            trace_values([*receiver, *descriptor], object_work);
            object_work.extend(*owner);
        }
        Object::Iterator { values: items, .. } => {
            trace_values(items.iter().copied(), object_work);
        }
        Object::CallableIterator {
            callable, sentinel, ..
        } => {
            trace_value(*callable, object_work);
            trace_value(*sentinel, object_work);
        }
        Object::Generator { scope, stack, .. } => {
            scope_work.push(*scope);
            trace_values(stack.iter().copied(), object_work);
        }
        Object::Module { scope, .. } => scope_work.push(*scope),
        Object::Array { storage, .. } => object_work.push(*storage),
        Object::ArgumentParser {
            arguments,
            subparsers,
            ..
        } => {
            for argument in arguments {
                trace_value(argument.default, object_work);
                trace_values(argument.choices.iter().copied(), object_work);
            }
            if let Some(subparsers) = subparsers {
                object_work.extend(
                    subparsers
                        .commands
                        .iter()
                        .map(|command| command.parser.object_id()),
                );
            }
        }
        Object::Namespace { values: entries } => {
            trace_values(entries.iter().map(|(_, value)| *value), object_work);
        }
        Object::Property { getter, setter } => {
            trace_value(*getter, object_work);
            trace_values(*setter, object_work);
        }
        Object::StaticMethod { callable } | Object::ClassMethod { callable } => {
            trace_value(*callable, object_work);
        }
        Object::Super {
            start_class,
            receiver,
        } => {
            object_work.push(*start_class);
            trace_value(*receiver, object_work);
        }
        Object::String(_)
        | Object::Bytes(_)
        | Object::ByteArray(_)
        | Object::Exception { .. }
        | Object::Slice { .. }
        | Object::BigInt(_)
        | Object::CountIterator { .. }
        | Object::Regex { .. }
        | Object::Match { .. }
        | Object::RaisesContext { .. } => {}
    }
}

fn modeled_size(object: &Object) -> Result<u64, String> {
    const HEADER: u64 = 32;
    const VALUE: u64 = 24;
    let slots = match object {
        Object::String(value) => value.len(),
        Object::Bytes(value) => value.len(),
        Object::ByteArray(value) => value.len(),
        Object::Exception { kind, message } => kind
            .len()
            .checked_add(message.len())
            .ok_or("modeled object size overflow")?,
        Object::List(values) | Object::Tuple(values) | Object::Set(values) => values.len(),
        Object::Slice { .. } => 3,
        Object::BigInt(value) => usize::try_from(value.bits().saturating_add(7) / 8)
            .map_err(|_| "modeled big integer size overflow")?,
        Object::Dict(entries) => entries
            .len()
            .checked_mul(2)
            .ok_or("modeled object size overflow")?,
        Object::DefaultDict { entries, .. } => entries
            .len()
            .checked_mul(2)
            .and_then(|slots| slots.checked_add(1))
            .ok_or("modeled object size overflow")?,
        Object::Function {
            name,
            code,
            defaults,
            ..
        } => name
            .len()
            .checked_add(code.instructions.len())
            .and_then(|size| size.checked_add(defaults.len()))
            .ok_or("modeled object size overflow")?,
        Object::Class {
            instance_type: _,
            name,
            bases,
            mro,
            metaclass: _,
            layout: _,
            attributes,
            is_dataclass,
            dataclass_fields,
            enum_members,
        } => name
            .len()
            .checked_add(bases.len())
            .and_then(|size| size.checked_add(mro.len()))
            .and_then(|size| size.checked_add(1))
            .and_then(|size| size.checked_add(attributes.len()))
            .and_then(|size| size.checked_add(usize::from(*is_dataclass)))
            .and_then(|size| size.checked_add(dataclass_fields.len()))
            .and_then(|size| size.checked_add(enum_members.len()))
            .ok_or("modeled object size overflow")?,
        Object::Instance { .. } => 0,
        Object::EnumMember { name, .. } => name
            .len()
            .checked_add(1)
            .ok_or("modeled object size overflow")?,
        Object::DescriptorBoundMethod { .. } => 3,
        Object::Iterator { values, .. } => values.len(),
        Object::CountIterator { .. } => 2,
        Object::CallableIterator { .. } => 3,
        Object::Generator {
            name,
            code,
            handlers,
            stack,
            ..
        } => name
            .len()
            .checked_add(code.instructions.len())
            .and_then(|size| size.checked_add(handlers.len()))
            .and_then(|size| size.checked_add(stack.len()))
            .ok_or("modeled object size overflow")?,
        Object::Module { name, .. } => name.len(),
        Object::ArrayStorage(values) => values.len(),
        Object::Array { layout, .. } => layout
            .shape
            .len()
            .checked_add(layout.strides.len())
            .and_then(|size| size.checked_add(3))
            .ok_or("modeled object size overflow")?,
        Object::Regex { pattern, .. } => pattern.len(),
        Object::Match { text, groups, .. } => text
            .len()
            .checked_add(
                groups
                    .iter()
                    .map(|group| group.as_ref().map_or(0, String::len))
                    .sum(),
            )
            .ok_or("modeled object size overflow")?,
        Object::ArgumentParser {
            prog,
            description,
            arguments,
            subparsers,
            ..
        } => prog
            .len()
            .checked_add(description.as_ref().map_or(0, String::len))
            .and_then(|size| size.checked_add(arguments.len()))
            .and_then(|size| {
                size.checked_add(
                    subparsers
                        .as_ref()
                        .map_or(0, |subparsers| subparsers.commands.len()),
                )
            })
            .ok_or("modeled object size overflow")?,
        Object::Namespace { values } => values.len(),
        Object::RaisesContext { expected } => expected.len(),
        Object::Property { setter, .. } => 1 + usize::from(setter.is_some()),
        Object::StaticMethod { .. } | Object::ClassMethod { .. } => 1,
        Object::Super { .. } => 2,
    };
    let slots = u64::try_from(slots).map_err(|_| "modeled object size overflow")?;
    HEADER
        .checked_add(
            slots
                .checked_mul(VALUE)
                .ok_or("modeled object size overflow")?,
        )
        .ok_or_else(|| "modeled object size overflow".into())
}

#[cfg(test)]
mod tests {
    use super::{Heap, Object};
    use crate::python::Value;
    use crate::resources::{Limits, Resources};
    use std::collections::HashMap;

    #[test]
    fn traced_roots_keep_aliases_and_unreachable_cycles_are_reclaimed() {
        let mut resources = Resources::new(Limits::unlimited());
        let mut heap = Heap::default();
        let first = heap
            .allocate(Object::List(Vec::new()), &mut resources)
            .unwrap();
        let second = heap
            .allocate(Object::List(vec![first]), &mut resources)
            .unwrap();
        let first_id = first.object_id().unwrap();
        heap.reserve_object_growth(first_id, 24, &mut resources)
            .unwrap();
        let Object::List(values) = heap.get_mut(first_id).unwrap() else {
            unreachable!()
        };
        values.push(second);

        assert_eq!(heap.collect(&[first], &[], &mut resources).unwrap(), 0);
        assert!(heap.get(first_id).is_ok());
        let released = heap.collect(&[], &[], &mut resources).unwrap();
        assert!(released > 0);
        assert!(heap.get(first_id).is_err());
        assert_eq!(resources.outcome(0, 0, 0).usage.memory_current, 0);
    }

    #[test]
    fn lexical_scope_roots_trace_values_and_vacant_slots_are_reused() {
        let mut resources = Resources::new(Limits::unlimited());
        let mut heap = Heap::default();
        let value = heap
            .allocate(Object::String("retained".into()), &mut resources)
            .unwrap();
        let scope = heap
            .allocate_scope(
                None,
                false,
                HashMap::from([("value".into(), value)]),
                &mut resources,
            )
            .unwrap();
        let old_id = value.object_id().unwrap();

        assert_eq!(heap.collect(&[], &[scope], &mut resources).unwrap(), 0);
        heap.collect(&[], &[], &mut resources).unwrap();
        let replacement = heap
            .allocate(Object::String("replacement".into()), &mut resources)
            .unwrap();
        assert_eq!(replacement.object_id(), Some(old_id));
        assert!(matches!(heap.get(old_id), Ok(Object::String(value)) if value == "replacement"));
    }

    #[test]
    fn immediate_values_do_not_create_false_arena_roots() {
        let mut resources = Resources::new(Limits::unlimited());
        let mut heap = Heap::default();
        let value = heap
            .allocate(Object::List(Vec::new()), &mut resources)
            .unwrap();
        assert!(value.object_id().is_some());
        heap.collect(&[Value::Int(1), Value::None], &[], &mut resources)
            .unwrap();
        assert!(heap.get(value.object_id().unwrap()).is_err());
    }
}
