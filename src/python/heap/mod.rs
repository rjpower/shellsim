//! Generational, non-moving storage for Python objects.
//!
//! Every Python object lives in one slot of an arena for its whole life, so its [`ObjectId`] (a
//! slot index and the slot's generation) never changes and Rust code names objects directly.
//! Collection is generational without moving anything: a new object is young until it survives
//! a minor collection, which marks from the roots and the remembered old objects, follows only
//! young objects, and frees the unmarked young ones. A mark-and-sweep major collection reclaims
//! old objects when the old generation has doubled since the last one. Collection can run inside
//! any allocation, so a program is out of memory only when its live data exceeds the limit.
//!
//! Code outside this module tree sees three value types ([`value`]):
//!
//! - a [`Value`], an immediate or an object id, pinned on the heap's pin stack from the moment it
//!   is made until the code that made it releases its pins;
//! - a [`Ref`], the stored reference inside an object or VM root, made from a pinned value;
//! - immediates, which need no pin.
//!
//! The root set is the pin stack, the slots the VM reports through [`Roots`], and, for a minor
//! collection, the old objects remembered since the last one. [`Heap::get_mut`] and
//! [`Heap::modify`] remember an old object whenever it is mutated, so a young reference stored
//! into an old object is always found.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use crate::resources::Resources;
use num_bigint::BigInt;

use super::attributes::ShapeId;
use super::bytecode::CodeRef;
use super::object_model::{BuiltinType, TypeId};
use super::string::PyString;

mod gc;
pub mod mapping;
mod native_object;
mod snapshot;
mod stack;
pub mod value;

pub use gc::Roots;
pub use mapping::{KeyHash, OrderedMap, OrderedSet};
pub use native_object::NativeObject;
pub use stack::ValueStack;
pub use value::{Ref, Value};

use value::Raw;

pub(crate) const MODELED_VALUE_BYTES: u64 = 24;
/// A dict entry holds its key, value, and hash in a slot plus one hash-index entry.
const MAPPING_ENTRY_VALUES: usize = 4;
pub(super) const MODELED_MAPPING_ENTRY_BYTES: u64 =
    MODELED_VALUE_BYTES * MAPPING_ENTRY_VALUES as u64;
/// A set member holds its value and hash in a slot plus one hash-index entry.
const SET_MEMBER_VALUES: usize = 3;
pub(super) const MODELED_SET_MEMBER_BYTES: u64 = MODELED_VALUE_BYTES * SET_MEMBER_VALUES as u64;
/// Building an object writes every byte it stores, so creating or replacing a payload also
/// charges one CPU unit per this many modeled bytes. Copies such as `s += t` then cost in
/// proportion to the bytes they move.
pub(super) const BYTES_PER_CPU_UNIT: u64 = 64;

/// Charge the CPU cost of writing `bytes` of new object storage.
pub(super) fn charge_construction(bytes: u64, resources: &mut Resources) -> Result<(), String> {
    if resources.charge_cpu(bytes / BYTES_PER_CPU_UNIT) {
        Ok(())
    } else {
        Err("resource limit exceeded while executing Python".into())
    }
}

/// Storage layout inherited by user-defined classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassLayout {
    Object,
    /// A subclass of a builtin value type such as `int` or `tuple`. Its instances carry that
    /// builtin's payload (an [`Object::BigInt`], [`Object::Tuple`], ...) under the subclass's
    /// type id, so builtin operations the class does not override act on the payload directly.
    Builtin(BuiltinType),
    Type,
}

/// Attribute storage for one instance of a user-defined class.
///
/// It hangs off the object header rather than the payload, like CPython's `__dict__` pointer, so
/// an instance of a `list` subclass is an [`Object::List`] with attributes and an instance of a
/// plain class is an [`Object::Bare`] with attributes. The box is allocated on the first
/// attribute write; most builtin objects never pay for it.
#[derive(Debug)]
pub enum InstanceAttributes {
    Shaped {
        shape: ShapeId,
        values: Vec<Ref>,
    },
    /// Boxed so the common shaped representation sets the size of an instance. The map's
    /// 48-byte header would otherwise widen every heap slot, which is the cost the
    /// `box_collection` lint does not see.
    #[allow(clippy::box_collection)]
    Dictionary(Box<HashMap<super::symbols::SymbolId, Ref>>),
}

impl Default for InstanceAttributes {
    fn default() -> Self {
        Self::Shaped {
            shape: ShapeId::ROOT,
            values: Vec::new(),
        }
    }
}

/// The heap's address of one object: a slot index and that slot's generation. A slot's
/// generation advances whenever its object is freed, so an id that outlives its object fails
/// loudly instead of naming a later one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct ObjectId(u64);

impl Object {
    /// The module-owned payload behind [`Object::Native`] when it is a `T`; `None` for any other
    /// payload, including another native type.
    pub fn native<T: NativeObject>(&self) -> Option<&T> {
        match self {
            Object::Native(native) => (native.as_ref() as &dyn std::any::Any).downcast_ref(),
            _ => None,
        }
    }

    /// Mutable access to the module-owned payload when it is a `T`.
    pub fn native_mut<T: NativeObject>(&mut self) -> Option<&mut T> {
        match self {
            Object::Native(native) => (native.as_mut() as &mut dyn std::any::Any).downcast_mut(),
            _ => None,
        }
    }
}

impl ObjectId {
    const fn new(index: u32, generation: u32) -> Self {
        Self(((generation as u64) << 32) | index as u64)
    }

    pub(super) const fn bits(self) -> u64 {
        self.0
    }

    pub(super) const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    const fn index(self) -> usize {
        (self.0 & 0xffff_ffff) as usize
    }

    const fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }
}

/// Where a namespace view's bindings actually live. `globals()`, `vars()` and `obj.__dict__`
/// return an [`Object::NamespaceDict`] over one of these, and every read and write through the
/// view acts on this storage.
///
/// An imported module's top-level code runs with its own lexical [`ScopeObject`]
/// (`uses_repl_globals` false), so its namespace is that scope. The entry-point script or an
/// interactive REPL line runs with no scope of its own; its names, and those of any function or
/// class body defined at that top level, live in the flat REPL/script table instead (see
/// `ReplState::globals` and `Vm::scope_uses_repl_globals`).
#[derive(Debug)]
pub enum NamespaceTarget {
    /// A module's own scope.
    Scope(Ref),
    /// The script and REPL global table.
    Repl,
    /// One instance's own attributes, for `obj.__dict__` and `vars(obj)`.
    Instance(Ref),
}

/// Which projection of a mapping a [`Object::DictView`] presents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DictViewKind {
    Keys,
    Values,
    Items,
}

/// What a [`Object::MappingProxy`] reads: a namespace that shellsim exposes read-only.
#[derive(Debug)]
pub enum ProxyTarget {
    /// A class's own attributes, for `cls.__dict__` and `vars(cls)`.
    Class(Ref),
    /// A builtin or native value-kind type's fixed registry namespace.
    RegisteredType(TypeId),
    /// A native module's functions and values, which cannot be rebound.
    NativeModule(&'static super::native::ModuleDef),
}

/// A user-defined class. Boxed inside [`Object::Class`] because it is much larger than the
/// common objects that share the heap slot size.
#[derive(Debug)]
pub struct ClassObject {
    /// Semantic type identity used by instances of this class.
    pub instance_type: TypeId,
    pub name: String,
    /// Every direct base in source order, native bases included: `__bases__`.
    pub bases: Vec<Ref>,
    /// C3-linearized user-defined ancestors, excluding this class.
    pub mro: Vec<Ref>,
    /// The callable type object responsible for this class.
    pub metaclass: Ref,
    pub layout: ClassLayout,
    /// Closest native exception ancestor, when instances may be raised.
    pub exception_base: Option<&'static str>,
    pub attributes: HashMap<String, Ref>,
    pub is_dataclass: bool,
    pub dataclass_fields: Vec<(String, Option<Ref>)>,
    pub enum_members: Vec<Ref>,
}

/// A Python function. Boxed inside [`Object::Function`] to keep heap slots small.
#[derive(Debug)]
pub struct FunctionObject {
    pub name: String,
    pub code: CodeRef,
    pub closure: Option<Ref>,
    pub defaults: Vec<Ref>,
    /// Class captured when this function is installed by a class body.
    pub defining_class: Option<Ref>,
    /// Names assigned on the function object, its `__dict__`.
    pub attributes: HashMap<String, Ref>,
}

/// A suspended generator frame. Boxed inside [`Object::Generator`] to keep heap slots small.
#[derive(Debug)]
pub struct GeneratorObject {
    /// The generator function, for its name and for zero-argument `super()` in its body.
    pub function: Ref,
    pub code: CodeRef,
    pub scope: Ref,
    pub instruction_pointer: usize,
    /// Active `try` regions as `(handler target, operand stack depth, exception stack depth)`.
    pub handlers: Vec<(usize, usize, usize)>,
    /// Context managers entered and not yet exited at the suspension point.
    pub contexts: Vec<Ref>,
    pub exceptions: Vec<(String, Ref)>,
    pub stack: Vec<Ref>,
    pub exhausted: bool,
    pub running: bool,
    pub return_value: Ref,
}

/// A lexical scope: a function activation's compiler-assigned local slots, any dynamically
/// bound names (module and class bodies, `exec`), and its enclosing scope. Name resolution
/// lives in [`scopes`](super::scopes); the heap only stores and traces the object.
#[derive(Debug)]
pub struct ScopeObject {
    pub parent: Option<Ref>,
    /// Whether names that this scope does not bind resolve in the flat REPL/script table rather
    /// than in a module scope.
    pub uses_repl_globals: bool,
    pub local_names: std::sync::Arc<[String]>,
    pub locals: Vec<Option<Ref>>,
    /// Dynamic names in first-binding order; `values` holds their current bindings.
    pub order: Vec<String>,
    pub values: HashMap<String, Ref>,
}

const LOCAL_SLOT_BYTES: u64 = 16;
/// Modeled bytes of one dynamically bound scope name, charged when a scope gains one.
pub const DYNAMIC_NAME_BYTES: u64 = 48;

fn modeled_scope_size(scope: &ScopeObject) -> Result<u64, String> {
    let slots = u64::try_from(scope.locals.len()).map_err(|_| "modeled scope size overflow")?;
    let dynamic = u64::try_from(scope.values.len()).map_err(|_| "modeled scope size overflow")?;
    OBJECT_HEADER
        .checked_add(
            slots
                .checked_mul(LOCAL_SLOT_BYTES)
                .ok_or("modeled scope size overflow")?,
        )
        .and_then(|bytes| bytes.checked_add(dynamic.checked_mul(DYNAMIC_NAME_BYTES)?))
        .ok_or_else(|| "modeled scope size overflow".into())
}

#[derive(Debug)]
pub enum Object {
    /// A direct `object()` instance: identity only, with no attributes.
    Bare,
    String(PyString),
    Bytes(Vec<u8>),
    ByteArray(Vec<u8>),
    /// The layout every exception instance inherits from `BaseException`: its constructor
    /// arguments, `args`. The class is the object's type, builtin or user-defined.
    Exception(Vec<Ref>),
    List(Vec<Ref>),
    Tuple(Vec<Ref>),
    /// `slice(start, stop, step)`. Bounds are any objects, as in CPython (`None` when
    /// omitted); they become indices only when the slice subscripts a sequence.
    Slice {
        start: Ref,
        stop: Ref,
        step: Ref,
    },
    Dict(OrderedMap),
    DefaultDict {
        factory: Ref,
        entries: OrderedMap,
    },
    Set(OrderedSet),
    FrozenSet(OrderedSet),
    BigInt(BigInt),
    /// Immutable builtin `complex`. Two doubles exceed the inline value payload, so complex
    /// numbers are heap objects like arbitrary-precision integers.
    Complex {
        real: f64,
        imag: f64,
    },
    /// Reusable arithmetic sequence. Iteration state lives in a separate iterator object.
    Range {
        start: i64,
        stop: i64,
        step: i64,
    },
    /// A boxed `float`, which only an instance of a `float` subclass needs: an exact float is an
    /// immediate value.
    Float(f64),
    Function(Box<FunctionObject>),
    Class(Box<ClassObject>),
    DescriptorBoundMethod {
        receiver: Ref,
        descriptor: Ref,
        owner: Option<Ref>,
    },
    /// A parameterized builtin class such as `list[int]`.
    GenericAlias {
        origin: Ref,
        arguments: Vec<Ref>,
    },
    Iterator {
        values: Vec<Ref>,
        position: usize,
    },
    /// Cursor over a heap sequence. Keeping the owner preserves mutation and lifetime semantics
    /// without copying every item when a loop starts.
    SequenceIterator {
        owner: Ref,
        position: usize,
    },
    /// Reverse traversal indexes the source on demand, preserving bounded iterator storage.
    ReverseIterator {
        owner: Ref,
        next: usize,
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
        callable: Ref,
        sentinel: Ref,
        exhausted: bool,
    },
    /// `for line in sys.stdin` (or `sys.stdin.buffer`). Kept as its own iterator object, rather
    /// than the generic native-value `__iter__`/`__next__` dispatch, because only a heap-object
    /// iterator can suspend a `for` loop: advancing this one can block on fd 0 (see
    /// `vm::iteration::advance_iterator`).
    StreamIterator {
        binary: bool,
    },
    /// A suspended Python generator frame. The bytecode is immutable; the instruction pointer,
    /// exception state, and lexical scope are the complete resumable state.
    Generator(Box<GeneratorObject>),
    Module {
        name: String,
        scope: Ref,
    },
    /// A lexical scope: a function activation, class body or module namespace. Scopes are
    /// ordinary objects so that closures, generators and frames keep them alive by reference
    /// and the collector reclaims them like everything else.
    Scope(Box<ScopeObject>),
    /// A live, dict-like view over a [`NamespaceTarget`]: what `globals()`, `vars()` and
    /// `obj.__dict__` return. Reads, writes and deletes through it act on the same storage that
    /// name and attribute lookups use.
    NamespaceDict(NamespaceTarget),
    /// `dict.keys()`, `dict.values()` or `dict.items()`: a live view that reads `mapping`'s
    /// current entries on every use. `mapping` is a dict, a namespace view or a mapping proxy.
    DictView {
        kind: DictViewKind,
        mapping: Ref,
    },
    /// A read-only mapping over a [`ProxyTarget`], like CPython's `mappingproxy`.
    MappingProxy(ProxyTarget),
    /// A registered value kind whose payload does not fit inline, such as a complex128 scalar.
    WideValue {
        type_id: TypeId,
        kind: u8,
        payload: [u64; 2],
    },
    /// A payload owned by one stdlib module, such as a compiled regex or an argument parser. The
    /// heap traces, sizes, copies and renders it through [`NativeObject`] alone.
    Native(Box<dyn NativeObject>),
    Property {
        getter: Ref,
        setter: Option<Ref>,
    },
    StaticMethod {
        callable: Ref,
    },
    ClassMethod {
        callable: Ref,
    },
    Super {
        start_class: Ref,
        receiver: Ref,
    },
}

/// Common header shared by every heap object.
#[derive(Debug)]
struct HeapObject {
    type_id: TypeId,
    /// The slot's generation in the low bits, and the collector's young, marked and remembered
    /// bits above it. Packed so the header stays within the slot.
    flags: Cell<u32>,
    payload: Object,
    /// Attributes assigned on an instance of a user class; `None` until the first write.
    attributes: Option<Box<InstanceAttributes>>,
    modeled_bytes: u64,
}

/// Allocated since the last minor collection.
const YOUNG_FLAG: u32 = 1 << 31;
/// Reached by the collection in progress.
const MARK_FLAG: u32 = 1 << 30;
/// An old object mutated since the last minor collection, so its slots may name young objects.
const DIRTY_FLAG: u32 = 1 << 29;
const GENERATION_MASK: u32 = DIRTY_FLAG - 1;

impl HeapObject {
    fn generation(&self) -> u32 {
        self.flags.get() & GENERATION_MASK
    }

    fn has(&self, flag: u32) -> bool {
        self.flags.get() & flag != 0
    }

    fn set(&self, flag: u32, on: bool) {
        let flags = self.flags.get();
        self.flags
            .set(if on { flags | flag } else { flags & !flag });
    }
}

/// Collection statistics, for tests and diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GcStats {
    pub minor_collections: u64,
    pub major_collections: u64,
    pub promoted_objects: u64,
}

pub struct Heap {
    /// Every object, by slot index; `None` marks a free slot.
    slots: Vec<Option<HeapObject>>,
    /// Free slots with the generation the next object there gets.
    free: Vec<(u32, u32)>,
    /// Slots allocated since the last minor collection.
    young: Vec<u32>,
    young_bytes: u64,
    old_bytes: u64,
    /// Old-generation size that triggers the next major collection.
    major_threshold: u64,
    /// Old objects mutated since the last minor collection; their slots may name young objects.
    remembered: Vec<u32>,
    /// Objects named by live [`Value`]s. Truncated by whoever released the values.
    pins: RefCell<Vec<ObjectId>>,
    /// Modeled bytes of every live object.
    modeled_bytes: u64,
    stats: GcStats,
}

impl Default for Heap {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            young: Vec::new(),
            young_bytes: 0,
            old_bytes: 0,
            major_threshold: 0,
            remembered: Vec::new(),
            pins: RefCell::new(Vec::new()),
            modeled_bytes: 0,
            stats: GcStats::default(),
        }
    }
}

impl std::fmt::Debug for Heap {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Heap")
            .field("slots", &self.slots.len())
            .field("young", &self.young.len())
            .field("modeled_bytes", &self.modeled_bytes)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl Clone for Heap {
    /// A deep copy for process-state snapshots. Pins belong to running Rust code and do not
    /// survive.
    fn clone(&self) -> Self {
        Self {
            slots: self
                .slots
                .iter()
                .map(|slot| slot.as_ref().map(HeapObject::dup))
                .collect(),
            free: self.free.clone(),
            young: self.young.clone(),
            young_bytes: self.young_bytes,
            old_bytes: self.old_bytes,
            major_threshold: self.major_threshold,
            remembered: self.remembered.clone(),
            pins: RefCell::new(Vec::new()),
            modeled_bytes: self.modeled_bytes,
            stats: self.stats,
        }
    }
}

impl Heap {
    // ----- pins ----------------------------------------------------------------------------

    /// Number of pinned values; whoever releases a group of values records this first.
    #[inline(always)]
    pub fn pin_count(&self) -> usize {
        self.pins.borrow().len()
    }

    /// Release every value pinned since the pin stack had `len` entries. The objects stay valid
    /// until a collection finds them unreachable.
    #[inline(always)]
    pub fn truncate_pins(&self, len: usize) {
        self.pins.borrow_mut().truncate(len);
    }

    /// The value a stored reference holds, pinned so it survives collection.
    #[inline(always)]
    pub fn value(&self, slot: &Ref) -> Value {
        self.pin_raw(slot.0)
    }

    pub fn value_optional(&self, slot: Option<&Ref>) -> Option<Value> {
        slot.map(|slot| self.value(slot))
    }

    pub fn values<'a>(&self, slots: impl IntoIterator<Item = &'a Ref>) -> Vec<Value> {
        slots.into_iter().map(|slot| self.value(slot)).collect()
    }

    #[inline(always)]
    fn pin_raw(&self, raw: Raw) -> Value {
        if let Some(id) = raw.object_id() {
            self.pins.borrow_mut().push(id);
        }
        Value::from_raw(raw)
    }

    // ----- object access -------------------------------------------------------------------

    #[inline(always)]
    fn object_id(value: Value) -> Result<ObjectId, String> {
        value
            .raw()
            .object_id()
            .ok_or_else(|| "expected a heap object".into())
    }

    #[inline(always)]
    fn object(&self, id: ObjectId) -> Result<&HeapObject, String> {
        match self.slots.get(id.index()) {
            Some(Some(object)) if object.generation() == id.generation() => Ok(object),
            _ => Err(stale_reference()),
        }
    }

    /// Mutable access to an object header. Old objects are remembered so a minor collection
    /// finds any young reference the caller stores into them.
    fn object_mut(&mut self, id: ObjectId) -> Result<&mut HeapObject, String> {
        let object = match self.slots.get_mut(id.index()) {
            Some(Some(object)) if object.generation() == id.generation() => object,
            _ => return Err(stale_reference()),
        };
        if !object.has(YOUNG_FLAG | DIRTY_FLAG) {
            object.set(DIRTY_FLAG, true);
            self.remembered.push(id.index() as u32);
        }
        Ok(object)
    }

    pub fn get(&self, value: Value) -> Result<&Object, String> {
        Ok(&self.object(Self::object_id(value)?)?.payload)
    }

    /// Mutable payload access. Old objects are remembered, so references may be stored through
    /// it.
    pub fn get_mut(&mut self, value: Value) -> Result<&mut Object, String> {
        let id = Self::object_id(value)?;
        Ok(&mut self.object_mut(id)?.payload)
    }

    /// Mutate an object's payload.
    pub fn modify<R>(
        &mut self,
        value: Value,
        f: impl FnOnce(&mut Object) -> R,
    ) -> Result<R, String> {
        Ok(f(self.get_mut(value)?))
    }

    /// The attributes stored on `value`, or `None` when none has been assigned (or `value` is
    /// not an instance of a user class).
    pub fn attributes(&self, value: Value) -> Result<Option<&InstanceAttributes>, String> {
        Ok(self.object(Self::object_id(value)?)?.attributes.as_deref())
    }

    /// `value`'s type together with its attributes, for inline caches that guard on both.
    pub fn typed_attributes(
        &self,
        value: Value,
    ) -> Result<(TypeId, Option<&InstanceAttributes>), String> {
        let object = self.object(Self::object_id(value)?)?;
        Ok((object.type_id, object.attributes.as_deref()))
    }

    /// Mutate `value`'s attribute storage, creating or replacing it.
    pub fn modify_attributes<R>(
        &mut self,
        value: Value,
        f: impl FnOnce(&mut Option<Box<InstanceAttributes>>) -> R,
    ) -> Result<R, String> {
        let id = Self::object_id(value)?;
        Ok(f(&mut self.object_mut(id)?.attributes))
    }

    pub fn type_id(&self, value: Value) -> Result<TypeId, String> {
        Ok(self.object(Self::object_id(value)?)?.type_id)
    }

    /// A stable identity for `id()` and identity hashing: one more than the object's slot index,
    /// unique among live objects as CPython's addresses are. Immediates have none.
    pub fn identity(&self, value: Value) -> Result<Option<u32>, String> {
        let Some(id) = value.raw().object_id() else {
            return Ok(None);
        };
        self.object(id)?;
        u32::try_from(id.index() + 1)
            .map(Some)
            .map_err(|_| "too many Python object identities".into())
    }

    /// Modeled size of one object's storage, as charged against the memory limit.
    pub fn object_bytes(&self, value: Value) -> Result<u64, String> {
        Ok(self.object(Self::object_id(value)?)?.modeled_bytes)
    }

    /// Collection counters, for tests that assert when collections happen.
    #[cfg(test)]
    pub fn stats(&self) -> GcStats {
        self.stats
    }

    /// Modeled bytes of every live object, for accounting tests.
    #[cfg(test)]
    pub fn modeled_bytes(&self) -> u64 {
        self.modeled_bytes
    }

    // ----- allocation ----------------------------------------------------------------------

    /// Allocate `object`, collecting first when the young generation is full or the memory limit
    /// would be exceeded. The object's references are roots of that collection. The object's
    /// type is the builtin type of its payload.
    pub fn alloc(
        &mut self,
        object: Object,
        roots: &dyn Roots,
        resources: &mut Resources,
    ) -> Result<Value, String> {
        let type_id = self.infer_type_id(&object)?;
        self.alloc_typed(type_id, object, roots, resources)
    }

    /// Allocate `object` as an instance of `type_id`, which is how an instance of a user class
    /// comes to carry a builtin payload: a `list` subclass instance is an [`Object::List`]
    /// whose type is the subclass.
    pub fn alloc_typed(
        &mut self,
        type_id: TypeId,
        object: Object,
        roots: &dyn Roots,
        resources: &mut Resources,
    ) -> Result<Value, String> {
        let bytes = modeled_size(&object)?;
        charge_construction(bytes, resources)?;
        self.make_room(bytes, Some(&object), roots, resources)?;
        self.modeled_bytes = self
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled heap size overflow")?;
        self.young_bytes = self.young_bytes.saturating_add(bytes);
        let (index, generation) = match self.free.pop() {
            Some(free) => free,
            None => {
                let index =
                    u32::try_from(self.slots.len()).map_err(|_| "too many Python objects")?;
                self.slots.push(None);
                (index, 0)
            }
        };
        self.slots[index as usize] = Some(HeapObject {
            type_id,
            flags: Cell::new(generation | YOUNG_FLAG),
            payload: object,
            attributes: None,
            modeled_bytes: bytes,
        });
        self.young.push(index);
        Ok(self.pin_raw(Raw::object(ObjectId::new(index, generation))))
    }

    /// A fresh payload holding the same builtin value as `value`, for the instance of a builtin
    /// subclass that `Class(value)` creates. Immediates are boxed; heap payloads are copied, so
    /// the instance never aliases the object it was built from.
    pub fn copy_builtin_payload(&self, value: Value) -> Result<Object, String> {
        if let Some(value) = value.inline_string_ref() {
            return Ok(Object::String(PyString::from(value.as_str())));
        }
        if let Some(value) = value.float_value() {
            return Ok(Object::Float(value));
        }
        if let Some(value) = value.bool_value() {
            return Ok(Object::BigInt(BigInt::from(value)));
        }
        if let Some(value) = value.immediate_int() {
            return Ok(Object::BigInt(BigInt::from(value)));
        }
        if !value.is_object() {
            return Err("value cannot be the payload of a builtin subclass instance".into());
        }
        match self.get(value)? {
            object @ (Object::String(_)
            | Object::Bytes(_)
            | Object::ByteArray(_)
            | Object::List(_)
            | Object::Tuple(_)
            | Object::Dict(_)
            | Object::Set(_)
            | Object::FrozenSet(_)
            | Object::BigInt(_)
            | Object::Float(_)
            | Object::Complex { .. }) => Ok(snapshot::dup_object(object)),
            _ => Err("value cannot be the payload of a builtin subclass instance".into()),
        }
    }

    /// Make the young generation and the memory limit accommodate `bytes` more, collecting as
    /// needed. `pending` is an object under construction whose references must survive the
    /// collection.
    fn make_room(
        &mut self,
        bytes: u64,
        pending: Option<&Object>,
        roots: &dyn Roots,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let limit = resources.limits().memory;
        let mut collected_fully = false;
        if self.old_bytes >= self.major_threshold.max(gc::major_floor(limit)) {
            self.collect_full(roots, pending, resources)?;
            collected_fully = true;
        } else if self.young_bytes.saturating_add(bytes) > gc::young_budget(limit)
            || collect_every_allocation()
        {
            self.collect_young(roots, pending, resources)?;
        }
        // A failed reservation stops the process for good, so only reserve once the headroom
        // is known to exist; otherwise collect everything first and let the final attempt fail.
        if resources.memory_remaining() >= bytes && resources.reserve_memory(bytes) {
            return Ok(());
        }
        if !collected_fully {
            self.collect_full(roots, pending, resources)?;
        }
        if resources.reserve_memory(bytes) {
            return Ok(());
        }
        Err("memory limit exceeded".into())
    }

    /// Replace an object payload while charging growth before installing it and releasing shrink
    /// after the old payload is no longer live. The object's identity is preserved.
    pub fn replace_payload(
        &mut self,
        value: Value,
        payload: Object,
        roots: &dyn Roots,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let next_bytes = modeled_size(&payload)?;
        let current_bytes = modeled_size(self.get(value)?)?;
        charge_construction(next_bytes, resources)?;
        if next_bytes > current_bytes {
            self.reserve_object_growth_pending(
                value,
                next_bytes - current_bytes,
                Some(&payload),
                roots,
                resources,
            )?;
        }
        let next_type = self.infer_type_id(&payload)?;
        let id = Self::object_id(value)?;
        let object = self.object_mut(id)?;
        object.type_id = next_type;
        object.payload = payload;
        if current_bytes > next_bytes {
            let released = current_bytes - next_bytes;
            object.modeled_bytes = object.modeled_bytes.saturating_sub(released);
            self.release_generation_bytes(id, released);
            self.modeled_bytes = self.modeled_bytes.saturating_sub(released);
            resources.release_memory(released);
        }
        Ok(())
    }

    /// Reserve `bytes` more modeled storage for one live object, collecting if needed.
    pub fn reserve_object_growth(
        &mut self,
        value: Value,
        bytes: u64,
        roots: &dyn Roots,
        resources: &mut Resources,
    ) -> Result<(), String> {
        self.reserve_object_growth_pending(value, bytes, None, roots, resources)
    }

    fn reserve_object_growth_pending(
        &mut self,
        value: Value,
        bytes: u64,
        pending: Option<&Object>,
        roots: &dyn Roots,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let next_heap_bytes = self
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled heap size overflow")?;
        let id = Self::object_id(value)?;
        self.object(id)?
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled object size overflow")?;
        self.make_room(bytes, pending, roots, resources)?;
        self.modeled_bytes = next_heap_bytes;
        // The object may have been promoted by the collection; charge the generation it is in.
        let object = self.object(id)?;
        let young = object.has(YOUNG_FLAG);
        let object = self.slots[id.index()].as_mut().expect("checked live above");
        object.modeled_bytes = object.modeled_bytes.saturating_add(bytes);
        if young {
            self.young_bytes = self.young_bytes.saturating_add(bytes);
        } else {
            self.old_bytes = self.old_bytes.saturating_add(bytes);
        }
        Ok(())
    }

    /// Release modeled storage removed from one live object's payload.
    pub fn release_object_shrink(
        &mut self,
        value: Value,
        bytes: u64,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let id = Self::object_id(value)?;
        let object = match self.slots.get_mut(id.index()) {
            Some(Some(object)) if object.generation() == id.generation() => object,
            _ => return Err(stale_reference()),
        };
        if object.modeled_bytes < bytes || self.modeled_bytes < bytes {
            return Err("modeled object size underflow".into());
        }
        object.modeled_bytes -= bytes;
        self.modeled_bytes -= bytes;
        self.release_generation_bytes(id, bytes);
        resources.release_memory(bytes);
        Ok(())
    }

    fn release_generation_bytes(&mut self, id: ObjectId, bytes: u64) {
        let young =
            matches!(self.slots.get(id.index()), Some(Some(object)) if object.has(YOUNG_FLAG));
        if young {
            self.young_bytes = self.young_bytes.saturating_sub(bytes);
        } else {
            self.old_bytes = self.old_bytes.saturating_sub(bytes);
        }
    }

    /// Transfer all live heap accounting to the caller when an interpreter is discarded.
    pub fn take_modeled_bytes(&mut self) -> u64 {
        std::mem::take(&mut self.modeled_bytes)
    }

    fn infer_type_id(&self, object: &Object) -> Result<TypeId, String> {
        Ok(match object {
            Object::Bare => BuiltinType::Object.id(),
            Object::String(_) => BuiltinType::String.id(),
            Object::Bytes(_) => BuiltinType::Bytes.id(),
            Object::ByteArray(_) => BuiltinType::ByteArray.id(),
            Object::Exception(_) => BuiltinType::Exception.id(),
            Object::List(_) => BuiltinType::List.id(),
            Object::Tuple(_) => BuiltinType::Tuple.id(),
            Object::Slice { .. } => BuiltinType::Slice.id(),
            Object::Dict(_) | Object::DefaultDict { .. } => BuiltinType::Dict.id(),
            Object::Set(_) => BuiltinType::Set.id(),
            Object::FrozenSet(_) => BuiltinType::FrozenSet.id(),
            Object::BigInt(_) => BuiltinType::Int.id(),
            Object::Complex { .. } => BuiltinType::Complex.id(),
            Object::Float(_) => BuiltinType::Float.id(),
            Object::Range { .. } => BuiltinType::Range.id(),
            Object::Function { .. } | Object::DescriptorBoundMethod { .. } => {
                BuiltinType::Function.id()
            }
            Object::Class(class_object) => {
                let metaclass = class_object.metaclass.0;
                match Value::from_raw(metaclass).native_value() {
                    Some(super::vm::NativeValue::BuiltinType(builtin)) => builtin.id(),
                    _ => match metaclass.object_id() {
                        Some(metaclass) => match &self.object(metaclass)?.payload {
                            Object::Class(metaclass) => metaclass.instance_type,
                            _ => return Err("class metaclass is not a class".into()),
                        },
                        None => return Err("class has an invalid metaclass".into()),
                    },
                }
            }
            Object::Iterator { .. }
            | Object::SequenceIterator { .. }
            | Object::ReverseIterator { .. }
            | Object::RangeIterator { .. }
            | Object::CountIterator { .. }
            | Object::CallableIterator { .. }
            | Object::StreamIterator { .. } => BuiltinType::Iterator.id(),
            Object::Generator { .. } => BuiltinType::Generator.id(),
            Object::Module { .. } => BuiltinType::Module.id(),
            Object::Scope(_) => BuiltinType::Native.id(),
            Object::NamespaceDict(_) => BuiltinType::NamespaceDict.id(),
            Object::DictView { kind, .. } => match kind {
                DictViewKind::Keys => BuiltinType::DictKeys.id(),
                DictViewKind::Values => BuiltinType::DictValues.id(),
                DictViewKind::Items => BuiltinType::DictItems.id(),
            },
            Object::MappingProxy(_) => BuiltinType::MappingProxy.id(),
            Object::GenericAlias { .. } => BuiltinType::GenericAlias.id(),
            Object::WideValue { type_id, .. } => *type_id,
            Object::Native(native) => native.python_type(),
            Object::Property { .. } => BuiltinType::Property.id(),
            Object::StaticMethod { .. } => BuiltinType::StaticMethod.id(),
            Object::ClassMethod { .. } => BuiltinType::ClassMethod.id(),
            Object::Super { .. } => BuiltinType::Native.id(),
        })
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only stress mode: run a minor collection before every allocation, so a value that
    /// Rust code holds without a pin is freed at the first chance and its use reports a stale
    /// reference instead of passing by luck.
    pub(crate) static COLLECT_EVERY_ALLOCATION: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[inline(always)]
fn collect_every_allocation() -> bool {
    #[cfg(test)]
    return COLLECT_EVERY_ALLOCATION.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

/// The error for an id whose slot was freed (or reused) since the id was made: a value that was
/// used after its pin was released, which is an interpreter bug.
fn stale_reference() -> String {
    "stale reference to a freed object".into()
}

/// Fixed charge for one heap slot: the host size of the slot itself, so that many small objects
/// cost the guest what they cost the host.
const OBJECT_HEADER: u64 = std::mem::size_of::<Option<HeapObject>>() as u64;

fn modeled_size(object: &Object) -> Result<u64, String> {
    const VALUE: u64 = MODELED_VALUE_BYTES;
    // Text, bytes, big-integer magnitudes, and packed array elements are charged at their byte
    // length rather than per value slot.
    let packed = |length: usize| {
        u64::try_from(length)
            .ok()
            .and_then(|bytes| bytes.checked_add(OBJECT_HEADER))
            .ok_or_else(|| String::from("modeled object size overflow"))
    };
    let slots = match object {
        Object::String(value) => return packed(value.len()),
        Object::Bytes(value) | Object::ByteArray(value) => return packed(value.len()),
        Object::Exception(values) | Object::List(values) | Object::Tuple(values) => values.len(),
        Object::Set(values) | Object::FrozenSet(values) => values
            .len()
            .checked_mul(SET_MEMBER_VALUES)
            .ok_or("modeled object size overflow")?,
        Object::Bare => 0,
        Object::Slice { .. } => 3,
        Object::BigInt(value) => {
            return packed(
                usize::try_from(value.bits().div_ceil(8))
                    .map_err(|_| "modeled big integer size overflow")?,
            )
        }
        // Up to sixteen bytes of payload rounded up to one modeled value slot.
        Object::Complex { .. } | Object::WideValue { .. } | Object::Float(_) => 1,
        Object::Range { .. } => 3,
        Object::Dict(entries) => entries
            .len()
            .checked_mul(MAPPING_ENTRY_VALUES)
            .ok_or("modeled object size overflow")?,
        Object::DefaultDict { entries, .. } => entries
            .len()
            .checked_mul(MAPPING_ENTRY_VALUES)
            .and_then(|slots| slots.checked_add(1))
            .ok_or("modeled object size overflow")?,
        Object::Function(function_object) => {
            let FunctionObject {
                name,
                code,
                defaults,
                attributes,
                ..
            } = &**function_object;
            name.len()
                .checked_add(code.instructions.len())
                .and_then(|size| size.checked_add(defaults.len()))
                .and_then(|size| size.checked_add(attributes.len()))
                .ok_or("modeled object size overflow")?
        }
        Object::Class(class_object) => {
            let ClassObject {
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
            } = &**class_object;
            name.len()
                .checked_add(bases.len())
                .and_then(|size| size.checked_add(mro.len()))
                .and_then(|size| size.checked_add(1))
                .and_then(|size| size.checked_add(usize::from(exception_base.is_some())))
                .and_then(|size| size.checked_add(attributes.len()))
                .and_then(|size| size.checked_add(usize::from(*is_dataclass)))
                .and_then(|size| size.checked_add(dataclass_fields.len()))
                .and_then(|size| size.checked_add(enum_members.len()))
                .ok_or("modeled object size overflow")?
        }
        Object::DescriptorBoundMethod { .. } => 3,
        Object::GenericAlias { arguments, .. } => arguments.len().saturating_add(1),
        Object::Iterator { values, .. } => values.len(),
        Object::SequenceIterator { .. } => 2,
        Object::ReverseIterator { .. } => 2,
        Object::RangeIterator { .. } => 4,
        Object::CountIterator { .. } => 2,
        Object::CallableIterator { .. } => 3,
        Object::StreamIterator { .. } => 1,
        Object::Generator(generator_object) => {
            let GeneratorObject {
                code,
                handlers,
                exceptions,
                stack,
                ..
            } = &**generator_object;
            code.instructions
                .len()
                .checked_add(handlers.len())
                .and_then(|size| size.checked_add(exceptions.len()))
                .and_then(|size| size.checked_add(stack.len()))
                .ok_or("modeled object size overflow")?
        }
        Object::Module { name, .. } => name.len(),
        Object::Scope(scope) => return modeled_scope_size(scope),
        // The namespace it views is charged where that namespace actually lives (the scope or
        // the REPL/script global table), so the view itself is a fixed, minimal handle.
        Object::NamespaceDict(_) | Object::DictView { .. } | Object::MappingProxy(_) => 1,
        Object::Native(native) => {
            return OBJECT_HEADER
                .checked_add(native.modeled_bytes()?)
                .ok_or_else(|| "modeled object size overflow".into())
        }
        Object::Property { setter, .. } => 1 + usize::from(setter.is_some()),
        Object::StaticMethod { .. } | Object::ClassMethod { .. } => 1,
        Object::Super { .. } => 2,
    };
    let slots = u64::try_from(slots).map_err(|_| "modeled object size overflow")?;
    OBJECT_HEADER
        .checked_add(
            slots
                .checked_mul(VALUE)
                .ok_or("modeled object size overflow")?,
        )
        .ok_or_else(|| "modeled object size overflow".into())
}

#[cfg(test)]
mod tests;
