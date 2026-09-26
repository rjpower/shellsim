//! Arena-backed mutable Python objects.
//!
//! IDs make aliasing explicit without pervasive interior mutability. Collection happens only at
//! VM instruction boundaries, where every live `Value` and lexical scope is available as a root.
//! A traced free list handles cycles while preserving the compact, copyable `Value` contract.

use crate::resources::Resources;
use num_bigint::BigInt;
use std::collections::HashMap;
use std::sync::Arc;

use super::bytecode::CodeRef;
use super::mapping::OrderedMap;
use super::native::{PyArgumentSpec, PyArrayDtype, PyArrayLayout};
use super::object_model::{BuiltinType, TypeId};
use super::string::PyString;
use super::Value;

pub(super) const MODELED_VALUE_BYTES: u64 = 24;
pub(super) const MODELED_MAPPING_ENTRY_BYTES: u64 = MODELED_VALUE_BYTES * 3;

const MAX_SHAPED_ATTRIBUTES: usize = 32;
const INSTANCE_SLOT_BYTES: u64 = 16;
const INSTANCE_DICT_ENTRY_BYTES: u64 = 48;
const SHAPE_BYTES: u64 = 24;
const SYMBOL_NAME_BYTES: u64 = 24;

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

/// Runtime-local identity for an interned Python identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SymbolId(u32);

impl SymbolId {
    pub(super) const fn index(self) -> usize {
        self.0 as usize
    }

    pub(super) fn from_index(index: usize) -> Option<Self> {
        u32::try_from(index).ok().map(Self)
    }
}

/// Runtime-local identity for one append-only instance storage layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShapeId(u32);

/// Guard and slot for one shaped instance attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstanceAttributeSlot {
    shape: ShapeId,
    slot: usize,
}

/// Attribute storage for one user-defined instance.
#[derive(Clone, Debug)]
pub enum InstanceAttributes {
    Shaped { shape: ShapeId, values: Vec<Value> },
    Dictionary(HashMap<SymbolId, Value>),
}

impl Default for InstanceAttributes {
    fn default() -> Self {
        Self::Shaped {
            shape: ShapeId(0),
            values: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct Shape {
    parent: Option<ShapeId>,
    added: Option<SymbolId>,
    slots: u32,
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
    String(PyString),
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
    Dict(OrderedMap),
    DefaultDict {
        factory: Value,
        entries: OrderedMap,
    },
    Set(Vec<Value>),
    FrozenSet(Vec<Value>),
    BigInt(BigInt),
    /// Reusable arithmetic sequence. Iteration state lives in a separate iterator object.
    Range {
        start: i64,
        stop: i64,
        step: i64,
    },
    Function {
        name: String,
        code: CodeRef,
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
        /// Closest native exception ancestor, when instances may be raised.
        exception_base: Option<&'static str>,
        attributes: HashMap<String, Value>,
        is_dataclass: bool,
        dataclass_fields: Vec<(String, Option<Value>)>,
        enum_members: Vec<Value>,
    },
    Instance {
        class: ObjectId,
        payload: InstancePayload,
        attributes: InstanceAttributes,
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
    /// Cursor over an arena sequence. Keeping the owner preserves mutation and lifetime semantics
    /// without copying every item when a loop starts.
    SequenceIterator {
        owner: ObjectId,
        position: usize,
    },
    RangeIterator {
        current: i64,
        stop: i64,
        step: i64,
        exhausted: bool,
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
    /// `for line in sys.stdin` (or `sys.stdin.buffer`). Kept as its own iterator object, rather
    /// than the generic native-value `__iter__`/`__next__` dispatch, because only a heap-object
    /// iterator can suspend a `for` loop: `ForIterator` requires an `ObjectId` to advance, and
    /// advancing this one can block on fd 0 (see `vm::iteration::advance_iterator`).
    StreamIterator {
        binary: bool,
    },
    /// A suspended Python generator frame. The bytecode is immutable; the instruction pointer,
    /// exception state, and lexical scope are the complete resumable state.
    Generator {
        name: String,
        code: CodeRef,
        scope: ScopeId,
        instruction_pointer: usize,
        handlers: Vec<(usize, usize)>,
        exceptions: Vec<(String, Value)>,
        stack: Vec<Value>,
        exhausted: bool,
        running: bool,
        return_value: Value,
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
        group_names: Vec<Option<String>>,
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

#[derive(Clone, Debug)]
pub struct Heap {
    objects: Vec<Option<HeapObject>>,
    free_objects: Vec<usize>,
    scopes: Vec<Option<Scope>>,
    free_scopes: Vec<usize>,
    symbol_ids: HashMap<Arc<str>, SymbolId>,
    symbol_names: Vec<Arc<str>>,
    shapes: Vec<Shape>,
    shape_transitions: HashMap<(ShapeId, SymbolId), ShapeId>,
    modeled_bytes: u64,
    bytes_since_collection: u64,
}

impl Default for Heap {
    fn default() -> Self {
        Self {
            objects: Vec::new(),
            free_objects: Vec::new(),
            scopes: Vec::new(),
            free_scopes: Vec::new(),
            symbol_ids: HashMap::new(),
            symbol_names: Vec::new(),
            shapes: vec![Shape {
                parent: None,
                added: None,
                slots: 0,
            }],
            shape_transitions: HashMap::new(),
            modeled_bytes: 0,
            bytes_since_collection: 0,
        }
    }
}

/// Common header shared by every arena-backed Python object.
#[derive(Clone, Debug)]
struct HeapObject {
    type_id: TypeId,
    payload: Object,
    modeled_bytes: u64,
}

#[derive(Clone, Debug)]
struct Scope {
    parent: Option<ScopeId>,
    uses_repl_globals: bool,
    local_names: Arc<[String]>,
    locals: Vec<Option<Value>>,
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

    /// Replace an object payload while charging growth before installing it and releasing shrink
    /// after the old payload is no longer live. The object's identity is preserved.
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
        let object = self
            .objects
            .get_mut(id.0)
            .and_then(Option::as_mut)
            .ok_or("invalid object reference")?;
        object.type_id = next_type;
        object.payload = payload;
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

    /// Return the runtime identity of an identifier already known to this heap.
    pub fn symbol_id(&self, name: &str) -> Option<SymbolId> {
        self.symbol_ids.get(name).copied()
    }

    /// Resolve a symbol identity back to its interpreter-owned name.
    pub fn symbol_name(&self, symbol: SymbolId) -> Option<&str> {
        self.symbol_names.get(symbol.index()).map(AsRef::as_ref)
    }

    /// Snapshot the names stored directly on an instance in either attribute representation.
    pub fn instance_attribute_names(&self, id: ObjectId) -> Result<Vec<String>, String> {
        let object = self
            .objects
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or("invalid object reference")?;
        let Object::Instance { attributes, .. } = &object.payload else {
            return Err("object does not have instance attributes".into());
        };
        match attributes {
            InstanceAttributes::Shaped { shape, values } => (0..values.len())
                .map(|slot| {
                    let symbol = self
                        .shape_attribute_at(*shape, slot)
                        .ok_or("invalid instance shape slot")?;
                    self.symbol_name(symbol)
                        .map(str::to_string)
                        .ok_or_else(|| "invalid instance attribute symbol".into())
                })
                .collect(),
            InstanceAttributes::Dictionary(values) => values
                .keys()
                .map(|symbol| {
                    self.symbol_name(*symbol)
                        .map(str::to_string)
                        .ok_or_else(|| "invalid instance attribute symbol".into())
                })
                .collect(),
        }
    }

    /// Intern one identifier, charging its process-lifetime storage before mutation.
    pub fn intern_symbol(
        &mut self,
        name: &str,
        resources: &mut Resources,
    ) -> Result<SymbolId, String> {
        if let Some(symbol) = self.symbol_id(name) {
            return Ok(symbol);
        }
        let symbol = SymbolId(
            u32::try_from(self.symbol_names.len()).map_err(|_| "too many Python identifiers")?,
        );
        let bytes = SYMBOL_NAME_BYTES
            .checked_add(u64::try_from(name.len()).unwrap_or(u64::MAX))
            .ok_or("identifier storage size overflow")?;
        let modeled_bytes = self
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled heap size overflow")?;
        if !resources.reserve_memory(bytes) {
            return Err("memory limit exceeded".into());
        }
        self.modeled_bytes = modeled_bytes;
        self.bytes_since_collection = self.bytes_since_collection.saturating_add(bytes);
        let name: Arc<str> = name.into();
        self.symbol_ids.insert(name.clone(), symbol);
        self.symbol_names.push(name);
        Ok(symbol)
    }

    #[cfg(test)]
    pub fn attribute(&self, id: ObjectId, name: &str) -> Result<Option<&Value>, String> {
        let Some(symbol) = self.symbol_id(name) else {
            return Ok(None);
        };
        self.attribute_by_symbol(id, symbol)
    }

    pub fn attribute_by_symbol(
        &self,
        id: ObjectId,
        symbol: SymbolId,
    ) -> Result<Option<&Value>, String> {
        let object = self
            .objects
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or("invalid object reference")?;
        let Object::Instance { attributes, .. } = &object.payload else {
            return Err("object does not have instance attributes".into());
        };
        match attributes {
            InstanceAttributes::Dictionary(values) => Ok(values.get(&symbol)),
            InstanceAttributes::Shaped { shape, values } => Ok(self
                .shape_slot(*shape, symbol)
                .and_then(|slot| values.get(slot))),
        }
    }

    pub fn instance_attribute_slot_by_symbol(
        &self,
        id: ObjectId,
        symbol: SymbolId,
    ) -> Result<Option<InstanceAttributeSlot>, String> {
        let object = self
            .objects
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or("invalid object reference")?;
        let Object::Instance { attributes, .. } = &object.payload else {
            return Ok(None);
        };
        let InstanceAttributes::Shaped { shape, values } = attributes else {
            return Ok(None);
        };
        Ok(self
            .shape_slot(*shape, symbol)
            .filter(|slot| *slot < values.len())
            .map(|slot| InstanceAttributeSlot {
                shape: *shape,
                slot,
            }))
    }

    pub fn cached_instance_attribute(
        &self,
        id: ObjectId,
        class: ObjectId,
        location: InstanceAttributeSlot,
    ) -> Result<Option<Value>, String> {
        let object = self
            .objects
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or("invalid object reference")?;
        let Object::Instance {
            class: actual_class,
            attributes,
            ..
        } = &object.payload
        else {
            return Ok(None);
        };
        if *actual_class != class {
            return Ok(None);
        }
        let InstanceAttributes::Shaped { shape, values } = attributes else {
            return Ok(None);
        };
        if *shape != location.shape {
            return Ok(None);
        }
        Ok(values.get(location.slot).copied())
    }

    pub fn insert_attribute(
        &mut self,
        id: ObjectId,
        name: String,
        value: Value,
        resources: &mut Resources,
    ) -> Result<(), String> {
        match &self
            .objects
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or("invalid object reference")?
            .payload
        {
            Object::Instance { .. } => {}
            _ => return Err("object does not have instance attributes".into()),
        }
        let symbol = self.intern_symbol(&name, resources)?;
        self.insert_attribute_by_symbol(id, symbol, value, resources)
    }

    pub fn insert_attribute_by_symbol(
        &mut self,
        id: ObjectId,
        symbol: SymbolId,
        value: Value,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let storage = match &self
            .objects
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or("invalid object reference")?
            .payload
        {
            Object::Instance { attributes, .. } => attributes,
            _ => return Err("object does not have instance attributes".into()),
        };
        match storage {
            InstanceAttributes::Dictionary(values) => {
                let growth = if values.contains_key(&symbol) {
                    0
                } else {
                    INSTANCE_DICT_ENTRY_BYTES
                };
                self.reserve_object_growth(id, growth, resources)?;
                let Object::Instance { attributes, .. } = &mut self
                    .objects
                    .get_mut(id.0)
                    .and_then(Option::as_mut)
                    .expect("instance was validated before growth")
                    .payload
                else {
                    unreachable!("instance was validated before growth")
                };
                let InstanceAttributes::Dictionary(values) = attributes else {
                    unreachable!("instance representation changed without yielding")
                };
                values.insert(symbol, value);
                Ok(())
            }
            InstanceAttributes::Shaped { shape, values } => {
                let shape = *shape;
                if let Some(slot) = self.shape_slot(shape, symbol) {
                    let Object::Instance { attributes, .. } = &mut self
                        .objects
                        .get_mut(id.0)
                        .and_then(Option::as_mut)
                        .expect("instance was validated before update")
                        .payload
                    else {
                        unreachable!("instance was validated before update")
                    };
                    let InstanceAttributes::Shaped { values, .. } = attributes else {
                        unreachable!("instance representation changed without yielding")
                    };
                    values[slot] = value;
                    return Ok(());
                }
                if values.len() >= MAX_SHAPED_ATTRIBUTES {
                    return self.insert_dictionary_attribute(id, symbol, value, resources);
                }
                self.append_shaped_attribute(id, shape, symbol, value, resources)
            }
        }
    }

    pub fn extend_attributes(
        &mut self,
        id: ObjectId,
        values: impl IntoIterator<Item = (String, Value)>,
        resources: &mut Resources,
    ) -> Result<(), String> {
        for (name, value) in values {
            self.insert_attribute(id, name, value, resources)?;
        }
        Ok(())
    }

    fn append_shaped_attribute(
        &mut self,
        id: ObjectId,
        shape: ShapeId,
        symbol: SymbolId,
        value: Value,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let existing_transition = self.shape_transitions.get(&(shape, symbol)).copied();
        let next_shape = existing_transition
            .unwrap_or_else(|| ShapeId(u32::try_from(self.shapes.len()).unwrap_or(u32::MAX)));
        if next_shape.0 == u32::MAX {
            return Err("too many instance shapes".into());
        }
        let shape_bytes = if existing_transition.is_none() {
            SHAPE_BYTES
        } else {
            0
        };
        self.reserve_instance_growth(id, INSTANCE_SLOT_BYTES, shape_bytes, resources)?;
        if existing_transition.is_none() {
            let slots = self
                .shapes
                .get(shape.0 as usize)
                .ok_or("invalid instance shape")?
                .slots
                .checked_add(1)
                .ok_or("too many shaped instance attributes")?;
            self.shapes.push(Shape {
                parent: Some(shape),
                added: Some(symbol),
                slots,
            });
            self.shape_transitions.insert((shape, symbol), next_shape);
        }
        let Object::Instance { attributes, .. } = &mut self
            .objects
            .get_mut(id.0)
            .and_then(Option::as_mut)
            .expect("instance was validated before growth")
            .payload
        else {
            unreachable!("instance was validated before growth")
        };
        let InstanceAttributes::Shaped { shape, values } = attributes else {
            unreachable!("instance representation changed without yielding")
        };
        *shape = next_shape;
        values.push(value);
        Ok(())
    }

    fn insert_dictionary_attribute(
        &mut self,
        id: ObjectId,
        symbol: SymbolId,
        value: Value,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let (shape, shaped_len) = match &self
            .objects
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or("invalid object reference")?
            .payload
        {
            Object::Instance {
                attributes: InstanceAttributes::Shaped { shape, values },
                ..
            } => (*shape, values.len()),
            Object::Instance {
                attributes: InstanceAttributes::Dictionary(_),
                ..
            } => return self.insert_attribute_by_symbol(id, symbol, value, resources),
            _ => return Err("object does not have instance attributes".into()),
        };
        let existing = u64::try_from(shaped_len)
            .unwrap_or(u64::MAX)
            .saturating_mul(INSTANCE_DICT_ENTRY_BYTES.saturating_sub(INSTANCE_SLOT_BYTES));
        self.reserve_object_growth(
            id,
            existing.saturating_add(INSTANCE_DICT_ENTRY_BYTES),
            resources,
        )?;
        let shaped_values = match &self
            .objects
            .get(id.0)
            .and_then(Option::as_ref)
            .expect("instance was validated before dictionary conversion")
            .payload
        {
            Object::Instance {
                attributes: InstanceAttributes::Shaped { values, .. },
                ..
            } => values.clone(),
            _ => unreachable!("instance representation changed without yielding"),
        };
        let mut values = HashMap::with_capacity(shaped_len.saturating_add(1));
        for (slot, value) in shaped_values.into_iter().enumerate() {
            let attribute = self
                .shape_attribute_at(shape, slot)
                .ok_or("invalid instance shape slot")?;
            values.insert(attribute, value);
        }
        values.insert(symbol, value);
        let Object::Instance { attributes, .. } = &mut self
            .objects
            .get_mut(id.0)
            .and_then(Option::as_mut)
            .expect("instance was validated before dictionary conversion")
            .payload
        else {
            unreachable!("instance was validated before dictionary conversion")
        };
        *attributes = InstanceAttributes::Dictionary(values);
        Ok(())
    }

    fn shape_slot(&self, mut shape: ShapeId, attribute: SymbolId) -> Option<usize> {
        while shape.0 != 0 {
            let current = self.shapes.get(shape.0 as usize)?;
            if current.added == Some(attribute) {
                return usize::try_from(current.slots.checked_sub(1)?).ok();
            }
            shape = current.parent?;
        }
        None
    }

    fn shape_attribute_at(&self, mut shape: ShapeId, slot: usize) -> Option<SymbolId> {
        while shape.0 != 0 {
            let current = self.shapes.get(shape.0 as usize)?;
            if usize::try_from(current.slots.checked_sub(1)?).ok()? == slot {
                return current.added;
            }
            shape = current.parent?;
        }
        None
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
        local_names: Arc<[String]>,
        mut values: HashMap<String, Value>,
        resources: &mut Resources,
    ) -> Result<ScopeId, String> {
        let mut locals = Vec::with_capacity(local_names.len());
        for name in local_names.iter() {
            locals.push(values.remove(name));
        }
        self.allocate_scope_slots(
            parent,
            uses_repl_globals,
            local_names,
            locals,
            values,
            resources,
        )
    }

    /// Allocate a lexical scope whose compiler-assigned local slots are already populated.
    pub fn allocate_scope_slots(
        &mut self,
        parent: Option<ScopeId>,
        uses_repl_globals: bool,
        local_names: Arc<[String]>,
        locals: Vec<Option<Value>>,
        values: HashMap<String, Value>,
        resources: &mut Resources,
    ) -> Result<ScopeId, String> {
        if locals.len() != local_names.len() {
            return Err("local slot metadata mismatch".into());
        }
        let slots = u64::try_from(locals.len()).map_err(|_| "modeled scope size overflow")?;
        let dynamic = u64::try_from(values.len()).map_err(|_| "modeled scope size overflow")?;
        let bytes = 32u64
            .checked_add(slots.checked_mul(16).ok_or("modeled scope size overflow")?)
            .and_then(|bytes| bytes.checked_add(dynamic.checked_mul(48)?))
            .ok_or("modeled scope size overflow")?;
        self.reserve_growth(bytes, resources)?;
        let scope = Scope {
            parent,
            uses_repl_globals,
            local_names,
            locals,
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
            if let Some(index) = current.local_names.iter().position(|local| local == name) {
                if let Some(value) = current.locals.get(index)?.as_ref() {
                    return Some(value);
                }
            }
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
        let scope = self
            .scopes
            .get(scope.0)
            .and_then(Option::as_ref)
            .ok_or("invalid scope reference")?;
        let mut values = scope.values.clone();
        values.extend(
            scope
                .local_names
                .iter()
                .zip(&scope.locals)
                .filter_map(|(name, value)| value.map(|value| (name.clone(), value))),
        );
        Ok(values)
    }

    pub fn scope_get_local(&self, scope: ScopeId, slot: usize) -> Result<Option<Value>, String> {
        self.scopes
            .get(scope.0)
            .and_then(Option::as_ref)
            .ok_or("invalid scope reference")?
            .locals
            .get(slot)
            .copied()
            .ok_or_else(|| "invalid local slot".into())
    }

    pub fn scope_store_local(
        &mut self,
        scope: ScopeId,
        slot: usize,
        value: Value,
    ) -> Result<(), String> {
        *self
            .scopes
            .get_mut(scope.0)
            .and_then(Option::as_mut)
            .ok_or("invalid scope reference")?
            .locals
            .get_mut(slot)
            .ok_or("invalid local slot")? = Some(value);
        Ok(())
    }

    pub fn scope_remove_local(
        &mut self,
        scope: ScopeId,
        slot: usize,
    ) -> Result<Option<Value>, String> {
        Ok(self
            .scopes
            .get_mut(scope.0)
            .and_then(Option::as_mut)
            .ok_or("invalid scope reference")?
            .locals
            .get_mut(slot)
            .ok_or("invalid local slot")?
            .take())
    }

    pub fn scope_insert(
        &mut self,
        scope: ScopeId,
        name: String,
        value: Value,
        resources: &mut Resources,
    ) -> Result<(), String> {
        if let Some(slot) = self
            .scopes
            .get(scope.0)
            .and_then(Option::as_ref)
            .ok_or("invalid scope reference")?
            .local_names
            .iter()
            .position(|local| local == &name)
        {
            return self.scope_store_local(scope, slot, value);
        }
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
            let candidate = self
                .scopes
                .get(current.0)
                .and_then(Option::as_ref)
                .ok_or("invalid scope reference")?;
            let parent = candidate.parent;
            if let Some(slot) = candidate.local_names.iter().position(|local| local == name) {
                if candidate.locals[slot].is_some() {
                    self.scope_store_local(current, slot, value)?;
                    return Ok(());
                }
            }
            if candidate.values.contains_key(name) {
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
        let current = self
            .scopes
            .get_mut(scope.0)
            .and_then(Option::as_mut)
            .ok_or("invalid scope reference")?;
        if let Some(slot) = current.local_names.iter().position(|local| local == name) {
            return Ok(current.locals[slot].take());
        }
        Ok(current.values.remove(name))
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
        let object = HeapObject {
            type_id,
            payload: object,
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
            Object::FrozenSet(_) => BuiltinType::FrozenSet.id(),
            Object::BigInt(_) => BuiltinType::Int.id(),
            Object::Range { .. } => BuiltinType::Range.id(),
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
            | Object::SequenceIterator { .. }
            | Object::RangeIterator { .. }
            | Object::CountIterator { .. }
            | Object::CallableIterator { .. }
            | Object::StreamIterator { .. } => BuiltinType::Iterator.id(),
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

    /// Release modeled storage removed from one live object's payload.
    pub fn release_object_shrink(
        &mut self,
        id: ObjectId,
        bytes: u64,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let object = self
            .objects
            .get_mut(id.0)
            .and_then(Option::as_mut)
            .ok_or("invalid object reference")?;
        if object.modeled_bytes < bytes || self.modeled_bytes < bytes {
            return Err("modeled object size underflow".into());
        }
        object.modeled_bytes -= bytes;
        self.modeled_bytes -= bytes;
        resources.release_memory(bytes);
        Ok(())
    }

    fn reserve_instance_growth(
        &mut self,
        id: ObjectId,
        object_bytes: u64,
        metadata_bytes: u64,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let growth = object_bytes
            .checked_add(metadata_bytes)
            .ok_or("modeled attribute size overflow")?;
        let next_heap_bytes = self
            .modeled_bytes
            .checked_add(growth)
            .ok_or("modeled heap size overflow")?;
        let next_object_bytes = self
            .objects
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or("invalid object reference")?
            .modeled_bytes
            .checked_add(object_bytes)
            .ok_or("modeled object size overflow")?;
        if !resources.reserve_memory(growth) {
            return Err("memory limit exceeded".into());
        }
        self.modeled_bytes = next_heap_bytes;
        self.bytes_since_collection = self.bytes_since_collection.saturating_add(growth);
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
                for value in scope.locals.iter().flatten() {
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
    match &object.payload {
        Object::List(items)
        | Object::Tuple(items)
        | Object::Set(items)
        | Object::FrozenSet(items)
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
        Object::Instance {
            class, attributes, ..
        } => {
            object_work.push(*class);
            match attributes {
                InstanceAttributes::Shaped { values, .. } => {
                    trace_values(values.iter().copied(), object_work);
                }
                InstanceAttributes::Dictionary(values) => {
                    trace_values(values.values().copied(), object_work);
                }
            }
        }
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
        Object::SequenceIterator { owner, .. } => object_work.push(*owner),
        Object::CallableIterator {
            callable, sentinel, ..
        } => {
            trace_value(*callable, object_work);
            trace_value(*sentinel, object_work);
        }
        Object::Generator {
            scope,
            exceptions,
            stack,
            return_value,
            ..
        } => {
            scope_work.push(*scope);
            trace_values(exceptions.iter().map(|(_, value)| *value), object_work);
            trace_values(stack.iter().copied(), object_work);
            trace_value(*return_value, object_work);
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
        | Object::Range { .. }
        | Object::RangeIterator { .. }
        | Object::CountIterator { .. }
        | Object::StreamIterator { .. }
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
        Object::List(values)
        | Object::Tuple(values)
        | Object::Set(values)
        | Object::FrozenSet(values) => values.len(),
        Object::Slice { .. } => 3,
        Object::BigInt(value) => usize::try_from(value.bits().saturating_add(7) / 8)
            .map_err(|_| "modeled big integer size overflow")?,
        Object::Range { .. } => 3,
        Object::Dict(entries) => entries
            .len()
            .checked_mul(3)
            .ok_or("modeled object size overflow")?,
        Object::DefaultDict { entries, .. } => entries
            .len()
            .checked_mul(3)
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
            exception_base,
            attributes,
            is_dataclass,
            dataclass_fields,
            enum_members,
        } => name
            .len()
            .checked_add(bases.len())
            .and_then(|size| size.checked_add(mro.len()))
            .and_then(|size| size.checked_add(1))
            .and_then(|size| size.checked_add(usize::from(exception_base.is_some())))
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
        Object::SequenceIterator { .. } => 2,
        Object::RangeIterator { .. } => 4,
        Object::CountIterator { .. } => 2,
        Object::CallableIterator { .. } => 3,
        Object::StreamIterator { .. } => 1,
        Object::Generator {
            name,
            code,
            handlers,
            exceptions,
            stack,
            ..
        } => name
            .len()
            .checked_add(code.instructions.len())
            .and_then(|size| size.checked_add(handlers.len()))
            .and_then(|size| size.checked_add(exceptions.len()))
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
        Object::Match {
            text,
            groups,
            group_names,
            ..
        } => text
            .len()
            .checked_add(
                groups
                    .iter()
                    .map(|group| group.as_ref().map_or(0, String::len))
                    .sum(),
            )
            .and_then(|size| {
                size.checked_add(
                    group_names
                        .iter()
                        .map(|name| name.as_ref().map_or(0, String::len))
                        .sum(),
                )
            })
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
    use super::*;
    use crate::python::vm::NativeValue;
    use crate::resources::{Limits, Resources};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn allocate_instance(heap: &mut Heap, resources: &mut Resources) -> Value {
        let class = heap
            .allocate(
                Object::Class {
                    instance_type: BuiltinType::Object.id(),
                    name: "Example".into(),
                    bases: Vec::new(),
                    mro: Vec::new(),
                    metaclass: Value::Native(NativeValue::BuiltinType(BuiltinType::Type)),
                    layout: ClassLayout::Object,
                    exception_base: None,
                    attributes: HashMap::new(),
                    is_dataclass: false,
                    dataclass_fields: Vec::new(),
                    enum_members: Vec::new(),
                },
                resources,
            )
            .unwrap();
        heap.allocate(
            Object::Instance {
                class: class.object_id().unwrap(),
                payload: InstancePayload::Object,
                attributes: InstanceAttributes::default(),
            },
            resources,
        )
        .unwrap()
    }

    fn instance_shape(heap: &Heap, value: Value) -> ShapeId {
        match heap.get(value.object_id().unwrap()).unwrap() {
            Object::Instance {
                attributes: InstanceAttributes::Shaped { shape, .. },
                ..
            } => *shape,
            _ => panic!("expected shaped instance"),
        }
    }

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
                Arc::from([]),
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

    #[test]
    fn instances_share_shape_transitions_and_diverge_by_attribute_order() {
        let mut resources = Resources::new(Limits::unlimited());
        let mut heap = Heap::default();
        let first = allocate_instance(&mut heap, &mut resources);
        let second = allocate_instance(&mut heap, &mut resources);
        let reversed = allocate_instance(&mut heap, &mut resources);

        for instance in [first, second] {
            let id = instance.object_id().unwrap();
            heap.insert_attribute(id, "left".into(), Value::Int(1), &mut resources)
                .unwrap();
            heap.insert_attribute(id, "right".into(), Value::Int(2), &mut resources)
                .unwrap();
        }
        let reversed_id = reversed.object_id().unwrap();
        heap.insert_attribute(reversed_id, "right".into(), Value::Int(3), &mut resources)
            .unwrap();
        heap.insert_attribute(reversed_id, "left".into(), Value::Int(4), &mut resources)
            .unwrap();

        assert_eq!(instance_shape(&heap, first), instance_shape(&heap, second));
        assert_ne!(
            instance_shape(&heap, first),
            instance_shape(&heap, reversed)
        );
        assert_eq!(
            heap.attribute(first.object_id().unwrap(), "right").unwrap(),
            Some(&Value::Int(2))
        );

        let original_shape = instance_shape(&heap, first);
        heap.insert_attribute(
            first.object_id().unwrap(),
            "right".into(),
            Value::Int(5),
            &mut resources,
        )
        .unwrap();
        assert_eq!(instance_shape(&heap, first), original_shape);
        assert_eq!(
            heap.attribute(first.object_id().unwrap(), "right").unwrap(),
            Some(&Value::Int(5))
        );
    }

    #[test]
    fn highly_dynamic_instances_fall_back_to_dictionary_storage() {
        let mut resources = Resources::new(Limits::unlimited());
        let mut heap = Heap::default();
        let instance = allocate_instance(&mut heap, &mut resources);
        let id = instance.object_id().unwrap();

        for index in 0..=MAX_SHAPED_ATTRIBUTES {
            heap.insert_attribute(
                id,
                format!("field_{index}"),
                Value::Int(index as i64),
                &mut resources,
            )
            .unwrap();
        }

        assert!(matches!(
            heap.get(id).unwrap(),
            Object::Instance {
                attributes: InstanceAttributes::Dictionary(_),
                ..
            }
        ));
        for index in 0..=MAX_SHAPED_ATTRIBUTES {
            assert_eq!(
                heap.attribute(id, &format!("field_{index}")).unwrap(),
                Some(&Value::Int(index as i64))
            );
        }
    }

    #[test]
    fn shaped_values_are_traced_and_cloned_heaps_diverge_cleanly() {
        let mut resources = Resources::new(Limits::unlimited());
        let mut heap = Heap::default();
        let instance = allocate_instance(&mut heap, &mut resources);
        let instance_id = instance.object_id().unwrap();
        let retained = heap
            .allocate(Object::String("retained".into()), &mut resources)
            .unwrap();
        let retained_id = retained.object_id().unwrap();
        heap.insert_attribute(instance_id, "value".into(), retained, &mut resources)
            .unwrap();

        heap.collect(&[instance], &[], &mut resources).unwrap();
        assert!(heap.get(retained_id).is_ok());

        let mut cloned_heap = heap.clone();
        let mut cloned_resources = resources.clone();
        heap.insert_attribute(
            instance_id,
            "original".into(),
            Value::Int(1),
            &mut resources,
        )
        .unwrap();
        cloned_heap
            .insert_attribute(
                instance_id,
                "cloned".into(),
                Value::Int(2),
                &mut cloned_resources,
            )
            .unwrap();
        assert_eq!(heap.attribute(instance_id, "cloned").unwrap(), None);
        assert_eq!(
            cloned_heap.attribute(instance_id, "original").unwrap(),
            None
        );
        assert_eq!(
            cloned_heap.attribute(instance_id, "cloned").unwrap(),
            Some(&Value::Int(2))
        );
    }

    #[test]
    fn shape_growth_fails_before_mutating_the_instance_or_tables() {
        let mut resources = Resources::new(Limits {
            memory: 4 * 1024,
            ..Limits::unlimited()
        });
        let mut heap = Heap::default();
        let instance = allocate_instance(&mut heap, &mut resources);
        let id = instance.object_id().unwrap();
        let shapes_before = heap.shapes.len();
        let names_before = heap.symbol_names.len();

        let error = heap
            .insert_attribute(id, "x".repeat(8 * 1024), Value::Int(1), &mut resources)
            .unwrap_err();
        assert_eq!(error, "memory limit exceeded");
        assert_eq!(heap.shapes.len(), shapes_before);
        assert_eq!(heap.symbol_names.len(), names_before);
        assert!(matches!(
            heap.get(id).unwrap(),
            Object::Instance {
                attributes: InstanceAttributes::Shaped { values, .. },
                ..
            } if values.is_empty()
        ));
    }
}
