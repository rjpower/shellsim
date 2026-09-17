//! Arena-backed mutable Python objects.
//!
//! IDs make aliasing explicit and keep cycle collection possible without pervasive interior
//! mutability. The initial arena is append-only; mark/sweep can reuse vacant slots later without
//! changing `Value` or bytecode.

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
        arguments: Vec<PyArgumentSpec>,
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
    objects: Vec<HeapObject>,
    scopes: Vec<Scope>,
    modeled_bytes: u64,
}

/// Common header shared by every arena-backed Python object.
#[derive(Clone, Debug)]
struct HeapObject {
    type_id: TypeId,
    attributes: HashMap<String, Value>,
    payload: Object,
}

#[derive(Clone, Debug)]
struct Scope {
    parent: Option<ScopeId>,
    uses_repl_globals: bool,
    values: HashMap<String, Value>,
}

impl Heap {
    pub fn get(&self, id: ObjectId) -> Result<&Object, String> {
        self.objects
            .get(id.0)
            .map(|object| &object.payload)
            .ok_or_else(|| "invalid object reference".into())
    }

    pub fn get_mut(&mut self, id: ObjectId) -> Result<&mut Object, String> {
        self.objects
            .get_mut(id.0)
            .map(|object| &mut object.payload)
            .ok_or_else(|| "invalid object reference".into())
    }

    pub fn type_id(&self, id: ObjectId) -> Result<TypeId, String> {
        self.objects
            .get(id.0)
            .map(|object| object.type_id)
            .ok_or_else(|| "invalid object reference".into())
    }

    pub fn attribute(&self, id: ObjectId, name: &str) -> Result<Option<&Value>, String> {
        Ok(self
            .objects
            .get(id.0)
            .ok_or("invalid object reference")?
            .attributes
            .get(name))
    }

    pub fn has_attribute(&self, id: ObjectId, name: &str) -> Result<bool, String> {
        Ok(self
            .objects
            .get(id.0)
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
            .ok_or("invalid object reference")?
            .attributes
            .insert(name, value);
        Ok(())
    }

    pub fn extend_attributes(
        &mut self,
        id: ObjectId,
        values: impl IntoIterator<Item = (String, Value)>,
    ) -> Result<(), String> {
        self.objects
            .get_mut(id.0)
            .ok_or("invalid object reference")?
            .attributes
            .extend(values);
        Ok(())
    }

    pub fn modeled_bytes(&self) -> u64 {
        self.modeled_bytes
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
        let id = ScopeId(self.scopes.len());
        self.scopes.push(Scope {
            parent,
            uses_repl_globals,
            values,
        });
        Ok(id)
    }

    pub fn scope_get(&self, mut scope: ScopeId, name: &str) -> Option<&Value> {
        loop {
            let current = self.scopes.get(scope.0)?;
            if let Some(value) = current.values.get(name) {
                return Some(value);
            }
            scope = current.parent?;
        }
    }

    pub fn scope_uses_repl_globals(&self, mut scope: ScopeId) -> Result<bool, String> {
        loop {
            let current = self.scopes.get(scope.0).ok_or("invalid scope reference")?;
            if let Some(parent) = current.parent {
                scope = parent;
            } else {
                return Ok(current.uses_repl_globals);
            }
        }
    }

    pub fn scope_root(&self, mut scope: ScopeId) -> Result<ScopeId, String> {
        loop {
            let current = self.scopes.get(scope.0).ok_or("invalid scope reference")?;
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
            .ok_or("invalid scope reference")?
            .parent)
    }

    pub fn scope_values(&self, scope: ScopeId) -> Result<HashMap<String, Value>, String> {
        Ok(self
            .scopes
            .get(scope.0)
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
            .ok_or("invalid scope reference")?
            .values
            .contains_key(&name);
        if is_new {
            self.reserve_growth(48, resources)?;
        }
        self.scopes
            .get_mut(scope.0)
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
            .ok_or("invalid scope reference")?
            .parent
            .ok_or_else(|| format!("no binding for nonlocal {name:?} found"))?;
        loop {
            let parent = self
                .scopes
                .get(current.0)
                .ok_or("invalid scope reference")?
                .parent;
            if self.scopes[current.0].values.contains_key(name) {
                self.scopes[current.0]
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
        let id = ObjectId(self.objects.len());
        let type_id = self.infer_type_id(&object)?;
        self.objects.push(HeapObject {
            type_id,
            attributes: HashMap::new(),
            payload: object,
        });
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
            Object::Iterator { .. } | Object::CountIterator { .. } => BuiltinType::Iterator.id(),
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

    pub fn reserve_growth(&mut self, bytes: u64, resources: &mut Resources) -> Result<(), String> {
        if !resources.reserve_memory(bytes) {
            return Err("memory limit exceeded".into());
        }
        self.modeled_bytes = self
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled heap size overflow")?;
        Ok(())
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
        Object::ArgumentParser { prog, arguments } => prog
            .len()
            .checked_add(arguments.len())
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
