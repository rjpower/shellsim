//! Arena-backed mutable Python objects.
//!
//! IDs make aliasing explicit and keep cycle collection possible without pervasive interior
//! mutability. The initial arena is append-only; mark/sweep can reuse vacant slots later without
//! changing `Value` or bytecode.

use crate::resources::Resources;
use std::collections::HashMap;

use super::bytecode::Code;
use super::native::MethodDef;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScopeId(usize);

#[derive(Clone, Debug)]
pub enum Object {
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    DefaultDict {
        factory: Value,
        entries: Vec<(Value, Value)>,
    },
    Set(Vec<Value>),
    Function {
        name: String,
        code: Code,
        closure: Option<ScopeId>,
        defaults: Vec<Value>,
    },
    Class {
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
        attributes: HashMap<String, Value>,
    },
    EnumMember {
        name: String,
        value: Value,
    },
    PythonBoundMethod {
        receiver: Value,
        function: ObjectId,
    },
    NativeBoundMethod {
        receiver: Value,
        method: &'static MethodDef,
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
    BoundMethod {
        receiver: Value,
        method: Method,
    },
    Module {
        name: String,
        scope: ScopeId,
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
        arguments: Vec<ArgumentSpec>,
    },
    Namespace {
        values: Vec<(String, Value)>,
    },
    RaisesContext {
        expected: String,
    },
}

#[derive(Clone, Debug)]
pub struct ArgumentSpec {
    pub names: Vec<String>,
    pub dest: String,
    pub required: bool,
    pub default: Value,
    pub store_true: bool,
    pub integer: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum Method {
    StringStrip,
    StringLStrip,
    StringRStrip,
    StringStartsWith,
    StringEndsWith,
    StringSplit,
    ListAppend,
    ListExtend,
    ListPop,
    ListRemove,
    ListSort,
    DictGet,
    DictKeys,
    DictValues,
    DictItems,
    DictSetDefault,
    SetAdd,
    SetUpdate,
    SetRemove,
    SetDiscard,
    ArgumentParserAddArgument,
    ArgumentParserParseArgs,
    RaisesEnter,
    RaisesExit,
    UnitTestAssertEqual,
    UnitTestAssertTrue,
    UnitTestAssertFalse,
    UnitTestAssertIsNone,
    UnitTestAssertRaises,
}

#[derive(Default, Debug)]
pub struct Heap {
    objects: Vec<Object>,
    scopes: Vec<Scope>,
    modeled_bytes: u64,
}

#[derive(Debug)]
struct Scope {
    parent: Option<ScopeId>,
    uses_repl_globals: bool,
    values: HashMap<String, Value>,
}

impl Heap {
    pub fn get(&self, id: ObjectId) -> Result<&Object, String> {
        self.objects
            .get(id.0)
            .ok_or_else(|| "invalid object reference".into())
    }

    pub fn get_mut(&mut self, id: ObjectId) -> Result<&mut Object, String> {
        self.objects
            .get_mut(id.0)
            .ok_or_else(|| "invalid object reference".into())
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
        self.objects.push(object);
        Ok(Value::Object(id))
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
        Object::List(values) | Object::Tuple(values) | Object::Set(values) => values.len(),
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
        Object::Instance { attributes, .. } => attributes.len(),
        Object::EnumMember { name, .. } => name
            .len()
            .checked_add(1)
            .ok_or("modeled object size overflow")?,
        Object::PythonBoundMethod { .. } => 2,
        Object::NativeBoundMethod { .. } => 2,
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
        Object::BoundMethod { .. } => 1,
        Object::Module { name, .. } => name.len(),
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
