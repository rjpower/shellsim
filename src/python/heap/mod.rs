//! Generational, handle-rooted storage for Python objects.
//!
//! Every Python object lives in one of two spaces. New objects are bump-allocated into the young
//! space; when it fills, a minor collection copies the survivors into the old space, which a
//! mark-and-sweep major collection reclaims when it has doubled since the last one. Collection
//! can therefore run inside any allocation, so a program is out of memory only when its live
//! data exceeds the limit.
//!
//! Moving objects is sound because no Rust code holds a raw object address. Code outside this
//! module tree sees three value types ([`value`]):
//!
//! - a [`Value<'s>`] handle, bound to a scope and read through the heap's handle stack, which
//!   the collector rewrites when an object moves;
//! - a [`Ref`], the stored reference inside an object or VM root, readable only through a
//!   handle and never copyable;
//! - immediates, which need no scope at all.
//!
//! The root set is the handle stack, the slots the VM reports through [`Roots`], and the old
//! objects remembered since the last minor collection. [`Heap::get_mut`] remembers an old object
//! whenever it is mutated, so a young reference stored into an old object is always found. The
//! only way to create a stored reference is through a [`Builder`] inside an allocation or a
//! mutation, or [`Heap::store`] for the VM's own root containers.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use crate::resources::Resources;
use num_bigint::BigInt;

use super::attributes::ShapeId;
use super::bytecode::CodeRef;
use super::native::PyArrayView;
use super::object_model::{BuiltinType, TypeId};
use super::string::PyString;

mod gc;
pub mod mapping;
mod snapshot;
mod stack;
pub mod value;

pub use gc::Roots;
pub use mapping::{KeyHash, OrderedMap, OrderedSet};
pub use stack::ValueStack;
pub use value::{Ref, Value};

use value::Raw;

pub(super) const MODELED_VALUE_BYTES: u64 = 24;
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
    /// A subclass of a builtin value type such as `int` or `tuple`, whose instances carry a
    /// value of that type in [`InstancePayload::Builtin`].
    Builtin(BuiltinType),
    Type,
}

/// Type-erased payload carried by an instance while preserving its user-defined class identity.
#[derive(Debug)]
pub enum InstancePayload {
    Object,
    /// The builtin value an instance of a builtin subclass stands for, such as the `int` of
    /// `class Flag(int)` or the `tuple` of a named tuple. Builtin operations that the class does
    /// not override act on this value.
    Builtin(Ref),
}

/// Attribute storage for one user-defined instance.
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

/// The heap's address of one object. Young addresses carry the young space's epoch so a stale
/// reference to a moved object fails loudly instead of aliasing a later allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct ObjectId(u64);

const YOUNG_FLAG: u64 = 1 << 63;

impl ObjectId {
    const fn young(epoch: u32, index: usize) -> Self {
        Self(YOUNG_FLAG | ((epoch as u64) << 32) | index as u64)
    }

    const fn old(index: usize) -> Self {
        Self(index as u64)
    }

    pub(super) const fn bits(self) -> u64 {
        self.0
    }

    pub(super) const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    const fn is_young(self) -> bool {
        self.0 & YOUNG_FLAG != 0
    }

    const fn index(self) -> usize {
        (self.0 & 0xffff_ffff) as usize
    }

    const fn epoch(self) -> u32 {
        ((self.0 >> 32) & 0x7fff_ffff) as u32
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
    pub name: String,
    pub code: CodeRef,
    pub scope: Ref,
    pub instruction_pointer: usize,
    /// Active `try` regions as `(handler target, operand stack depth, exception stack depth)`.
    pub handlers: Vec<(usize, usize, usize)>,
    pub exceptions: Vec<(String, Ref)>,
    pub stack: Vec<Ref>,
    pub exhausted: bool,
    pub running: bool,
    pub return_value: Ref,
}

/// A regular-expression match. Boxed inside [`Object::Match`] to keep heap slots small.
///
/// The subject string and the `re.Pattern` are held by reference so a match costs its own
/// group text, not a copy of the string it was found in. `spans` are character offsets into
/// the subject per group, `None` for a group that did not participate.
#[derive(Debug)]
pub struct MatchObject {
    pub subject: Ref,
    pub regex: Ref,
    pub text: String,
    pub groups: Vec<Option<String>>,
    pub group_names: Vec<Option<String>>,
    pub spans: Vec<Option<(usize, usize)>>,
    pub pos: usize,
    pub endpos: usize,
}

/// One declared `argparse` argument, with its default and choices stored as references.
#[derive(Debug)]
pub struct ArgumentSpec {
    pub names: Vec<String>,
    pub dest: String,
    pub required: bool,
    pub default: Ref,
    pub store_true: bool,
    pub store_false: bool,
    pub integer: bool,
    pub choices: Vec<Ref>,
    pub help: Option<String>,
}

/// One command of an `argparse` subparser collection; `parser` is the sub-parser object.
#[derive(Debug)]
pub struct SubcommandSpec {
    pub name: String,
    pub help: Option<String>,
    pub parser: Ref,
}

/// The one-level subparser surface of an `argparse.ArgumentParser`.
#[derive(Debug)]
pub struct SubparsersSpec {
    pub dest: Option<String>,
    pub required: bool,
    pub help: Option<String>,
    pub commands: Vec<SubcommandSpec>,
}

/// An `argparse.ArgumentParser`. Boxed inside [`Object::ArgumentParser`] to keep heap slots small.
#[derive(Debug)]
pub struct ArgumentParserObject {
    pub prog: String,
    pub description: Option<String>,
    pub add_help: bool,
    pub is_subcommand: bool,
    pub arguments: Vec<ArgumentSpec>,
    pub subparsers: Option<SubparsersSpec>,
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

/// Flat element storage shared by one or more array views: packed bytes, or traced Python
/// references for object arrays.
#[derive(Debug)]
pub enum ArrayStorage {
    Bytes(Vec<u8>),
    Values(Vec<Ref>),
}

impl ArrayStorage {
    /// Length of the addressable storage in bytes.
    pub fn byte_len(&self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::Values(values) => values.len().saturating_mul(16),
        }
    }
}

#[derive(Debug)]
pub enum Object {
    /// A direct `object()` instance: identity only, with no attributes.
    Bare,
    String(PyString),
    Bytes(Vec<u8>),
    ByteArray(Vec<u8>),
    /// An instance of a builtin exception class, with the constructor arguments `args`.
    Exception {
        kind: String,
        args: Vec<Ref>,
    },
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
    Function(Box<FunctionObject>),
    Class(Box<ClassObject>),
    Instance {
        class: Ref,
        payload: InstancePayload,
        attributes: InstanceAttributes,
    },
    EnumMember {
        class: Option<Ref>,
        name: String,
        value: Ref,
    },
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
    /// Flat element storage shared by one or more array views.
    ArrayStorage(ArrayStorage),
    /// An ndarray view. Byte strides and offset map indices into `ArrayStorage`.
    Array {
        storage: Ref,
        view: Box<PyArrayView>,
        /// The array that owns the storage, for `ndarray.base`; `None` for owners.
        base: Option<Ref>,
    },
    /// A registered value kind whose payload does not fit inline, such as a complex128 scalar.
    WideValue {
        type_id: TypeId,
        kind: u8,
        payload: [u64; 2],
    },
    /// A compiled regular expression.  The pattern is compiled at the operation boundary so
    /// regex execution never gets a host capability; keeping the source and flags here also
    /// makes the object cheap to copy and deterministic to inspect.
    Regex {
        pattern: String,
        flags: u32,
    },
    /// A bounded regular-expression match result.  Captures are stored as owned text rather than
    /// references into a host regex object, which keeps the heap self-contained.
    Match(Box<MatchObject>),
    ArgumentParser(Box<ArgumentParserObject>),
    Namespace {
        values: Vec<(String, Ref)>,
    },
    RaisesContext {
        expected: String,
    },
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
    /// Bit 31: remembered since the last minor collection (old objects only). Bits 0..31: the
    /// lazily assigned identity, 0 while unassigned. Packed so a header stays within the slot.
    flags: Cell<u32>,
    payload: Object,
    modeled_bytes: u64,
}

const DIRTY_FLAG: u32 = 1 << 31;
const IDENTITY_MASK: u32 = DIRTY_FLAG - 1;

impl HeapObject {
    fn is_dirty(&self) -> bool {
        self.flags.get() & DIRTY_FLAG != 0
    }

    fn set_dirty(&self, dirty: bool) {
        let flags = self.flags.get();
        self.flags.set(if dirty {
            flags | DIRTY_FLAG
        } else {
            flags & IDENTITY_MASK
        });
    }

    fn identity(&self) -> u32 {
        self.flags.get() & IDENTITY_MASK
    }

    fn set_identity(&self, identity: u32) {
        self.flags
            .set((self.flags.get() & DIRTY_FLAG) | (identity & IDENTITY_MASK));
    }
}

/// Converts scoped handles into stored references while an allocation or mutation is in
/// progress. A builder exists only inside [`Heap::alloc_with`] and [`Heap::modify`], so the
/// references it makes are stored before the next allocation can move anything.
pub struct Builder<'h> {
    handles: &'h RefCell<Vec<Raw>>,
}

impl Builder<'_> {
    /// The stored reference for a handle.
    pub fn store(&self, value: Value<'_>) -> Ref {
        Ref(resolve(self.handles, value))
    }

    pub fn optional(&self, value: Option<Value<'_>>) -> Option<Ref> {
        value.map(|value| self.store(value))
    }

    pub fn refs<'s>(&self, values: impl IntoIterator<Item = Value<'s>>) -> Vec<Ref> {
        // Collecting straight from a `Vec<Value>` would reuse its allocation in place, so a
        // list cut down to a few items could keep the capacity of the snapshot it came from.
        // The modeled size counts items, so build exactly the storage the items need.
        let values = values.into_iter();
        let mut refs = Vec::with_capacity(values.size_hint().0);
        refs.extend(values.map(|value| self.store(value)));
        refs
    }

    pub fn named<'s>(
        &self,
        values: impl IntoIterator<Item = (String, Value<'s>)>,
    ) -> HashMap<String, Ref> {
        values
            .into_iter()
            .map(|(name, value)| (name, self.store(value)))
            .collect()
    }
}

fn resolve(handles: &RefCell<Vec<Raw>>, value: Value<'_>) -> Raw {
    match value.raw().handle_index() {
        Some(index) => *handles
            .borrow()
            .get(index)
            .expect("a handle cannot outlive its scope"),
        None => value.raw(),
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
    young: Vec<Option<HeapObject>>,
    young_epoch: u32,
    young_bytes: u64,
    old: Vec<Option<HeapObject>>,
    free_old: Vec<usize>,
    old_bytes: u64,
    /// Old-space size that triggers the next major collection.
    major_threshold: u64,
    /// Old objects mutated since the last minor collection; their slots may name young objects.
    remembered: Vec<usize>,
    handles: RefCell<Vec<Raw>>,
    next_identity: Cell<u32>,
    /// Modeled bytes of every object in both spaces.
    modeled_bytes: u64,
    stats: GcStats,
}

impl Default for Heap {
    fn default() -> Self {
        Self {
            young: Vec::new(),
            young_epoch: 0,
            young_bytes: 0,
            old: Vec::new(),
            free_old: Vec::new(),
            old_bytes: 0,
            major_threshold: 0,
            remembered: Vec::new(),
            handles: RefCell::new(Vec::new()),
            next_identity: Cell::new(1),
            modeled_bytes: 0,
            stats: GcStats::default(),
        }
    }
}

impl std::fmt::Debug for Heap {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Heap")
            .field("young", &self.young.len())
            .field("old", &self.old.len())
            .field("modeled_bytes", &self.modeled_bytes)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl Clone for Heap {
    /// A deep copy for process-state snapshots. Handles are scope-local and do not survive.
    fn clone(&self) -> Self {
        Self {
            young: self
                .young
                .iter()
                .map(|slot| slot.as_ref().map(HeapObject::dup))
                .collect(),
            young_epoch: self.young_epoch,
            young_bytes: self.young_bytes,
            old: self
                .old
                .iter()
                .map(|slot| slot.as_ref().map(HeapObject::dup))
                .collect(),
            free_old: self.free_old.clone(),
            old_bytes: self.old_bytes,
            major_threshold: self.major_threshold,
            remembered: self.remembered.clone(),
            handles: RefCell::new(Vec::new()),
            next_identity: self.next_identity.clone(),
            modeled_bytes: self.modeled_bytes,
            stats: self.stats,
        }
    }
}

impl Heap {
    // ----- handles -------------------------------------------------------------------------

    /// Number of live handle-stack entries; a scope records this when it opens.
    pub fn handle_count(&self) -> usize {
        self.handles.borrow().len()
    }

    /// Drop every handle created since the stack had `len` entries; a scope does this when it
    /// closes. The entries above `len` can no longer be named, so nothing dangles.
    pub fn truncate_handles(&self, len: usize) {
        self.handles.borrow_mut().truncate(len);
    }

    /// A scoped handle for a stored reference. Immediates pass through without a stack entry.
    pub fn handle<'s>(&self, slot: &Ref) -> Value<'s> {
        self.handle_raw(slot.0)
    }

    pub fn handle_optional<'s>(&self, slot: Option<&Ref>) -> Option<Value<'s>> {
        slot.map(|slot| self.handle(slot))
    }

    pub fn handles<'a, 's>(&self, slots: impl IntoIterator<Item = &'a Ref>) -> Vec<Value<'s>> {
        slots.into_iter().map(|slot| self.handle(slot)).collect()
    }

    fn handle_raw<'s>(&self, raw: Raw) -> Value<'s> {
        if raw.is_object() {
            let mut handles = self.handles.borrow_mut();
            handles.push(raw);
            Value::from_raw(Raw::handle(handles.len() - 1))
        } else {
            Value::from_raw(raw)
        }
    }

    /// The stored form of a handle, for the VM's own root containers. The result must be placed
    /// in a root before the next allocation; it is not a handle and the collector cannot see it
    /// in a Rust local.
    pub fn store(&self, value: Value<'_>) -> Ref {
        Ref(self.resolve(value))
    }

    fn resolve(&self, value: Value<'_>) -> Raw {
        resolve(&self.handles, value)
    }

    fn builder(&self) -> Builder<'_> {
        Builder {
            handles: &self.handles,
        }
    }

    /// Whether two handles name the same object or the same immediate (`is`).
    pub fn identical(&self, left: Value<'_>, right: Value<'_>) -> bool {
        self.resolve(left) == self.resolve(right)
    }

    /// Whether a handle and a stored reference name the same object or immediate.
    pub fn identical_ref(&self, value: Value<'_>, slot: &Ref) -> bool {
        self.resolve(value) == slot.0
    }

    // ----- object access -------------------------------------------------------------------

    fn object_id(&self, value: Value<'_>) -> Result<ObjectId, String> {
        self.resolve(value)
            .object_id()
            .ok_or_else(|| "expected a heap object".into())
    }

    fn object(&self, id: ObjectId) -> Result<&HeapObject, String> {
        let slot = if id.is_young() {
            if id.epoch() != self.young_epoch {
                return Err("stale reference to a moved young object".into());
            }
            self.young.get(id.index())
        } else {
            self.old.get(id.index())
        };
        slot.and_then(Option::as_ref)
            .ok_or_else(|| "invalid object reference".into())
    }

    /// Mutable access to an object header. Old objects are remembered so a minor collection
    /// finds any young reference the caller stores into them.
    fn object_mut(&mut self, id: ObjectId) -> Result<&mut HeapObject, String> {
        if id.is_young() {
            if id.epoch() != self.young_epoch {
                return Err("stale reference to a moved young object".into());
            }
            return self
                .young
                .get_mut(id.index())
                .and_then(Option::as_mut)
                .ok_or_else(|| "invalid object reference".into());
        }
        let object = self
            .old
            .get_mut(id.index())
            .and_then(Option::as_mut)
            .ok_or("invalid object reference")?;
        if !object.is_dirty() {
            object.set_dirty(true);
            self.remembered.push(id.index());
        }
        Ok(object)
    }

    pub fn get(&self, value: Value<'_>) -> Result<&Object, String> {
        Ok(&self.object(self.object_id(value)?)?.payload)
    }

    /// Mutable payload access. Store references into the payload only through [`Self::modify`];
    /// this entry point is for payload fields that hold no references, such as iterator
    /// positions, byte buffers and generator control state.
    pub fn get_mut(&mut self, value: Value<'_>) -> Result<&mut Object, String> {
        let id = self.object_id(value)?;
        Ok(&mut self.object_mut(id)?.payload)
    }

    /// Mutate an object with a [`Builder`] for turning handles into stored references.
    pub fn modify<R>(
        &mut self,
        value: Value<'_>,
        f: impl FnOnce(&Builder<'_>, &mut Object) -> R,
    ) -> Result<R, String> {
        let id = self.object_id(value)?;
        let builder = Builder {
            handles: &self.handles,
        };
        let object = if id.is_young() {
            if id.epoch() != self.young_epoch {
                return Err("stale reference to a moved young object".into());
            }
            self.young
                .get_mut(id.index())
                .and_then(Option::as_mut)
                .ok_or("invalid object reference")?
        } else {
            let object = self
                .old
                .get_mut(id.index())
                .and_then(Option::as_mut)
                .ok_or("invalid object reference")?;
            if !object.is_dirty() {
                object.set_dirty(true);
                self.remembered.push(id.index());
            }
            object
        };
        Ok(f(&builder, &mut object.payload))
    }

    pub fn type_id(&self, value: Value<'_>) -> Result<TypeId, String> {
        Ok(self.object(self.object_id(value)?)?.type_id)
    }

    /// A stable identity for `id()` and identity hashing, assigned on first use and unchanged
    /// when the object moves. Immediates have none.
    pub fn identity(&self, value: Value<'_>) -> Result<Option<u32>, String> {
        let Some(id) = self.resolve(value).object_id() else {
            return Ok(None);
        };
        let object = self.object(id)?;
        let identity = object.identity();
        if identity != 0 {
            return Ok(Some(identity));
        }
        let identity = self.next_identity.get();
        if identity & IDENTITY_MASK == 0 {
            return Err("too many Python object identities".into());
        }
        self.next_identity.set(identity.wrapping_add(1));
        object.set_identity(identity);
        Ok(Some(identity))
    }

    /// Modeled size of one object's storage, as charged against the memory limit.
    pub fn object_bytes(&self, value: Value<'_>) -> Result<u64, String> {
        Ok(self.object(self.object_id(value)?)?.modeled_bytes)
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

    /// Allocate `object` in the young space, collecting first when the space is full or the
    /// memory limit would be exceeded. `object`'s own references are rewritten if that
    /// collection moves their targets.
    pub fn alloc<'s>(
        &mut self,
        mut object: Object,
        roots: &mut dyn Roots,
        resources: &mut Resources,
    ) -> Result<Value<'s>, String> {
        let bytes = modeled_size(&object)?;
        charge_construction(bytes, resources)?;
        self.make_room(bytes, Some(&mut object), roots, resources)?;
        let type_id = self.infer_type_id(&object)?;
        self.modeled_bytes = self
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled heap size overflow")?;
        self.young_bytes = self.young_bytes.saturating_add(bytes);
        let id = ObjectId::young(self.young_epoch, self.young.len());
        self.young.push(Some(HeapObject {
            type_id,
            flags: Cell::new(0),
            payload: object,
            modeled_bytes: bytes,
        }));
        Ok(self.handle_raw(Raw::object(id)))
    }

    /// Allocate an object whose payload holds references, built from handles by a [`Builder`].
    pub fn alloc_with<'s>(
        &mut self,
        roots: &mut dyn Roots,
        resources: &mut Resources,
        build: impl FnOnce(&Builder<'_>) -> Object,
    ) -> Result<Value<'s>, String> {
        let object = build(&self.builder());
        self.alloc(object, roots, resources)
    }

    /// Make the young space and the memory limit accommodate `bytes` more, collecting as needed.
    /// `pending` is an object under construction whose references must survive the collection.
    fn make_room(
        &mut self,
        bytes: u64,
        mut pending: Option<&mut Object>,
        roots: &mut dyn Roots,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let limit = resources.limits().memory;
        let mut collected_fully = false;
        if self.old_bytes >= self.major_threshold.max(gc::major_floor(limit)) {
            self.collect_full(roots, pending.as_deref_mut(), resources)?;
            collected_fully = true;
        } else if self.young_bytes.saturating_add(bytes) > gc::young_budget(limit) {
            self.collect_young(roots, pending.as_deref_mut(), resources)?;
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
        value: Value<'_>,
        payload: Object,
        roots: &mut dyn Roots,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let next_bytes = modeled_size(&payload)?;
        let current_bytes = modeled_size(self.get(value)?)?;
        charge_construction(next_bytes, resources)?;
        let mut payload = payload;
        if next_bytes > current_bytes {
            self.reserve_object_growth_pending(
                value,
                next_bytes - current_bytes,
                Some(&mut payload),
                roots,
                resources,
            )?;
        }
        let next_type = self.infer_type_id(&payload)?;
        let id = self.object_id(value)?;
        let object = self.object_mut(id)?;
        object.type_id = next_type;
        object.payload = payload;
        if current_bytes > next_bytes {
            let released = current_bytes - next_bytes;
            object.modeled_bytes = object.modeled_bytes.saturating_sub(released);
            self.release_space_bytes(id, released);
            self.modeled_bytes = self.modeled_bytes.saturating_sub(released);
            resources.release_memory(released);
        }
        Ok(())
    }

    /// Reserve `bytes` more modeled storage for one live object, collecting if needed.
    pub fn reserve_object_growth(
        &mut self,
        value: Value<'_>,
        bytes: u64,
        roots: &mut dyn Roots,
        resources: &mut Resources,
    ) -> Result<(), String> {
        self.reserve_object_growth_pending(value, bytes, None, roots, resources)
    }

    fn reserve_object_growth_pending(
        &mut self,
        value: Value<'_>,
        bytes: u64,
        pending: Option<&mut Object>,
        roots: &mut dyn Roots,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let next_heap_bytes = self
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled heap size overflow")?;
        let id = self.object_id(value)?;
        self.object(id)?
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled object size overflow")?;
        self.make_room(bytes, pending, roots, resources)?;
        // The collection may have moved the object; look it up again through the handle.
        let id = self.object_id(value)?;
        self.modeled_bytes = next_heap_bytes;
        self.add_space_bytes(id, bytes);
        let object = if id.is_young() {
            self.young.get_mut(id.index()).and_then(Option::as_mut)
        } else {
            self.old.get_mut(id.index()).and_then(Option::as_mut)
        }
        .ok_or("invalid object reference")?;
        object.modeled_bytes = object.modeled_bytes.saturating_add(bytes);
        Ok(())
    }

    /// Release modeled storage removed from one live object's payload.
    pub fn release_object_shrink(
        &mut self,
        value: Value<'_>,
        bytes: u64,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let id = self.object_id(value)?;
        let object = if id.is_young() {
            self.young.get_mut(id.index()).and_then(Option::as_mut)
        } else {
            self.old.get_mut(id.index()).and_then(Option::as_mut)
        }
        .ok_or("invalid object reference")?;
        if object.modeled_bytes < bytes || self.modeled_bytes < bytes {
            return Err("modeled object size underflow".into());
        }
        object.modeled_bytes -= bytes;
        self.modeled_bytes -= bytes;
        self.release_space_bytes(id, bytes);
        resources.release_memory(bytes);
        Ok(())
    }

    fn add_space_bytes(&mut self, id: ObjectId, bytes: u64) {
        if id.is_young() {
            self.young_bytes = self.young_bytes.saturating_add(bytes);
        } else {
            self.old_bytes = self.old_bytes.saturating_add(bytes);
        }
    }

    fn release_space_bytes(&mut self, id: ObjectId, bytes: u64) {
        if id.is_young() {
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
            Object::Exception { .. } => BuiltinType::Exception.id(),
            Object::List(_) => BuiltinType::List.id(),
            Object::Tuple(_) => BuiltinType::Tuple.id(),
            Object::Slice { .. } => BuiltinType::Slice.id(),
            Object::Dict(_) | Object::DefaultDict { .. } => BuiltinType::Dict.id(),
            Object::Set(_) => BuiltinType::Set.id(),
            Object::FrozenSet(_) => BuiltinType::FrozenSet.id(),
            Object::BigInt(_) => BuiltinType::Int.id(),
            Object::Complex { .. } => BuiltinType::Complex.id(),
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
            Object::Instance { class, .. } => {
                let class = class.0.object_id().ok_or("instance class is not a class")?;
                match &self.object(class)?.payload {
                    Object::Class(class_object) => class_object.instance_type,
                    _ => return Err("instance class is not a class".into()),
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
            Object::ArrayStorage(_) => BuiltinType::Native.id(),
            Object::Array { .. } => BuiltinType::Array.id(),
            Object::WideValue { type_id, .. } => *type_id,
            Object::Regex { .. } => BuiltinType::Regex.id(),
            Object::Match { .. } => BuiltinType::Match.id(),
            Object::ArgumentParser { .. } => BuiltinType::ArgumentParser.id(),
            Object::RaisesContext { .. } => BuiltinType::RaisesContext.id(),
            Object::EnumMember { .. } | Object::Namespace { .. } => BuiltinType::Native.id(),
            Object::Property { .. } => BuiltinType::Property.id(),
            Object::StaticMethod { .. } => BuiltinType::StaticMethod.id(),
            Object::ClassMethod { .. } => BuiltinType::ClassMethod.id(),
            Object::Super { .. } => BuiltinType::Native.id(),
        })
    }
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
        Object::Exception { kind, args } => kind
            .len()
            .checked_add(args.len())
            .ok_or("modeled object size overflow")?,
        Object::List(values) | Object::Tuple(values) => values.len(),
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
        // Sixteen bytes of payload rounded up to one modeled value slot.
        Object::Complex { .. } | Object::WideValue { .. } => 1,
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
        Object::Instance { .. } => 0,
        Object::EnumMember { name, .. } => name
            .len()
            .checked_add(1)
            .ok_or("modeled object size overflow")?,
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
                name,
                code,
                handlers,
                exceptions,
                stack,
                ..
            } = &**generator_object;
            name.len()
                .checked_add(code.instructions.len())
                .and_then(|size| size.checked_add(handlers.len()))
                .and_then(|size| size.checked_add(exceptions.len()))
                .and_then(|size| size.checked_add(stack.len()))
                .ok_or("modeled object size overflow")?
        }
        Object::Module { name, .. } => name.len(),
        Object::Scope(scope) => return modeled_scope_size(scope),
        // The namespace it views is charged where that namespace actually lives (the scope or
        // the REPL/script global table), so the view itself is a fixed, minimal handle.
        Object::NamespaceDict(_) | Object::DictView { .. } | Object::MappingProxy(_) => 1,
        Object::ArrayStorage(ArrayStorage::Bytes(bytes)) => return packed(bytes.len()),
        Object::ArrayStorage(ArrayStorage::Values(values)) => values.len(),
        Object::Array { view, .. } => view
            .shape
            .len()
            .checked_add(view.strides.len())
            .and_then(|size| size.checked_add(3))
            .ok_or("modeled object size overflow")?,
        Object::Regex { pattern, .. } => pattern.len(),
        Object::Match(match_object) => {
            let MatchObject {
                text,
                groups,
                group_names,
                spans,
                ..
            } = &**match_object;
            text.len()
                .checked_add(spans.len().saturating_mul(16))
                .and_then(|size| {
                    size.checked_add(
                        groups
                            .iter()
                            .map(|group| group.as_ref().map_or(0, String::len))
                            .sum(),
                    )
                })
                .and_then(|size| {
                    size.checked_add(
                        group_names
                            .iter()
                            .map(|name| name.as_ref().map_or(0, String::len))
                            .sum(),
                    )
                })
                .ok_or("modeled object size overflow")?
        }
        Object::ArgumentParser(parser_object) => {
            let ArgumentParserObject {
                prog,
                description,
                arguments,
                subparsers,
                ..
            } = &**parser_object;
            prog.len()
                .checked_add(description.as_ref().map_or(0, String::len))
                .and_then(|size| size.checked_add(arguments.len()))
                .and_then(|size| {
                    size.checked_add(
                        subparsers
                            .as_ref()
                            .map_or(0, |subparsers| subparsers.commands.len()),
                    )
                })
                .ok_or("modeled object size overflow")?
        }
        Object::Namespace { values } => values.len(),
        Object::RaisesContext { expected } => expected.len(),
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
