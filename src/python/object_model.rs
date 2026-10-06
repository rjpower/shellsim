//! Python semantic types and bootstrapped builtin type metadata.
//!
//! `TypeId` is independent of storage shape: immediate values, heap objects, and native markers
//! all identify their Python type through the same registry. The registry stores only modeled
//! metadata and Python values, so looking up a type cannot acquire host capabilities.
//!
//! The registry is a heap root: it holds stored references ([`Ref`]) to class objects, class
//! attributes and cached descriptors, and reports them to the collector through
//! [`Roots`](heap::Roots). Slot tables copy descriptor references out of the attribute maps
//! with [`Ref::dup`], which is sound only because every copy stays inside this root.

use std::collections::HashMap;

use super::heap::{self, Heap, Object, Ref};
use super::native::{
    BinarySlotFn, CompareSlotFn, MethodDef, PyError, PyResult, PyRuntime, TernarySlotFn,
    UnarySlotFn,
};
use super::Value;

/// Stable identity of a Python type within a [`ReplState`](super::ReplState).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeId(u32);

impl TypeId {
    const fn builtin(value: BuiltinType) -> Self {
        Self(value as u32)
    }

    pub(super) const fn raw(self) -> u32 {
        self.0
    }

    pub(super) const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
}

/// Canonical builtin type objects and their stable registry indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub(super) enum BuiltinType {
    Object,
    Type,
    None,
    /// The type of the `Ellipsis` singleton, which CPython names `ellipsis`.
    Ellipsis,
    /// The type of the `NotImplemented` singleton.
    NotImplemented,
    Bool,
    Int,
    Float,
    String,
    Bytes,
    ByteArray,
    List,
    Tuple,
    Dict,
    Set,
    FrozenSet,
    Range,
    Function,
    Module,
    Iterator,
    Generator,
    Exception,
    Native,
    Regex,
    Match,
    Stream,
    Environment,
    /// The live namespace view behind `globals()`, `vars()` and `obj.__dict__`. See
    /// `heap::Object::NamespaceDict`.
    NamespaceDict,
    DictKeys,
    DictValues,
    DictItems,
    /// A read-only mapping: a class's `__dict__` or a native module's namespace.
    MappingProxy,
    ArgumentParser,
    RaisesContext,
    Property,
    Array,
    Complex,
    Slice,
    GenericAlias,
    Enum,
    TestCase,
    StaticMethod,
    ClassMethod,
}

impl BuiltinType {
    pub(super) const ALL: [Self; 43] = [
        Self::Object,
        Self::Type,
        Self::None,
        Self::Ellipsis,
        Self::NotImplemented,
        Self::Bool,
        Self::Int,
        Self::Float,
        Self::String,
        Self::Bytes,
        Self::ByteArray,
        Self::List,
        Self::Tuple,
        Self::Dict,
        Self::Set,
        Self::FrozenSet,
        Self::Range,
        Self::Function,
        Self::Module,
        Self::Iterator,
        Self::Generator,
        Self::Exception,
        Self::Native,
        Self::Regex,
        Self::Match,
        Self::Stream,
        Self::Environment,
        Self::NamespaceDict,
        Self::DictKeys,
        Self::DictValues,
        Self::DictItems,
        Self::MappingProxy,
        Self::ArgumentParser,
        Self::RaisesContext,
        Self::Property,
        Self::Array,
        Self::Complex,
        Self::Slice,
        Self::GenericAlias,
        Self::Enum,
        Self::TestCase,
        Self::StaticMethod,
        Self::ClassMethod,
    ];

    pub(super) const fn id(self) -> TypeId {
        TypeId::builtin(self)
    }

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Object => "object",
            Self::Type => "type",
            Self::None => "NoneType",
            Self::Ellipsis => "ellipsis",
            Self::NotImplemented => "NotImplementedType",
            Self::Bool => "bool",
            Self::Int => "int",
            Self::Float => "float",
            Self::String => "str",
            Self::Bytes => "bytes",
            Self::ByteArray => "bytearray",
            Self::List => "list",
            Self::Tuple => "tuple",
            Self::Dict => "dict",
            Self::Set => "set",
            Self::FrozenSet => "frozenset",
            Self::Range => "range",
            Self::Function => "function",
            Self::Module => "module",
            Self::Iterator => "iterator",
            Self::Generator => "generator",
            Self::Exception => "BaseException",
            Self::Native => "native",
            Self::Regex => "re.Pattern",
            Self::Match => "re.Match",
            Self::Stream => "shellsim.stream",
            Self::Environment => "shellsim.environment",
            Self::NamespaceDict => "shellsim.namespace_dict",
            Self::DictKeys => "dict_keys",
            Self::DictValues => "dict_values",
            Self::DictItems => "dict_items",
            Self::MappingProxy => "mappingproxy",
            Self::ArgumentParser => "argparse.ArgumentParser",
            Self::RaisesContext => "pytest.raises",
            Self::Property => "property",
            Self::StaticMethod => "staticmethod",
            Self::ClassMethod => "classmethod",
            Self::Array => "numpy.ndarray",
            Self::Complex => "complex",
            Self::Slice => "slice",
            Self::GenericAlias => "GenericAlias",
            Self::Enum => "enum.Enum",
            Self::TestCase => "unittest.TestCase",
        }
    }
}

/// Cached protocol methods resolved from a type dictionary, one entry per [`Slot`].
///
/// A user slot holds its ordinary Python descriptor. Builtin operator slots hold a direct native
/// function with the erased runtime ABI. An empty entry means the type does not implement the
/// protocol. The table is indexed by the slot's discriminant, so a lookup is one bounds check.
#[derive(Debug)]
pub struct TypeSlots([Option<SlotValue>; SLOT_COUNT]);

impl Default for TypeSlots {
    fn default() -> Self {
        Self([const { None }; SLOT_COUNT])
    }
}

/// A cached Python descriptor or a native implementation attached directly to a builtin type.
#[derive(Debug)]
pub enum SlotValue {
    /// A descriptor from a user class's namespace, with the type that defines it so a
    /// dispatch can bind it without searching the MRO again.
    Descriptor {
        value: Ref,
        owner: TypeId,
    },
    NativeMethod(&'static MethodDef),
    /// The VM's representation renderer, which shares cycle tracking across nested values.
    VmRepr,
    /// Hash compound values through the VM so their elements use Python's hash protocol.
    VmHash,
    /// A builtin type's six rich comparisons, one function answering for every operator.
    NativeCompare(CompareSlotFn),
    NativeBinary(BinarySlotFn),
    NativeTernary(TernarySlotFn),
    NativeUnary(UnarySlotFn),
}

/// Protocol operations cached on each type after MRO resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Slot {
    Call,
    New,
    Init,
    GetAttribute,
    SetAttribute,
    Repr,
    String,
    Bool,
    Hash,
    Iter,
    Next,
    Length,
    GetItem,
    SetItem,
    Positive,
    Negative,
    Invert,
    Absolute,
    Add,
    ReflectedAdd,
    Subtract,
    ReflectedSubtract,
    Multiply,
    ReflectedMultiply,
    MatrixMultiply,
    ReflectedMatrixMultiply,
    Power,
    ReflectedPower,
    Divide,
    ReflectedDivide,
    FloorDivide,
    ReflectedFloorDivide,
    Remainder,
    ReflectedRemainder,
    DivMod,
    ReflectedDivMod,
    LeftShift,
    ReflectedLeftShift,
    RightShift,
    ReflectedRightShift,
    BitwiseAnd,
    ReflectedBitwiseAnd,
    BitwiseXor,
    ReflectedBitwiseXor,
    BitwiseOr,
    ReflectedBitwiseOr,
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    Contains,
    DeleteItem,
    Format,
    Reversed,
    DeleteAttribute,
    ClassGetItem,
    InitSubclass,
    InplaceAdd,
    InplaceSubtract,
    InplaceMultiply,
    InplaceMatrixMultiply,
    InplacePower,
    InplaceDivide,
    InplaceFloorDivide,
    InplaceRemainder,
    InplaceLeftShift,
    InplaceRightShift,
    InplaceBitwiseAnd,
    InplaceBitwiseXor,
    InplaceBitwiseOr,
}

impl Slot {
    pub(super) const ALL: [Self; 72] = [
        Self::Call,
        Self::New,
        Self::Init,
        Self::GetAttribute,
        Self::SetAttribute,
        Self::Repr,
        Self::String,
        Self::Bool,
        Self::Hash,
        Self::Iter,
        Self::Next,
        Self::Length,
        Self::GetItem,
        Self::SetItem,
        Self::Positive,
        Self::Negative,
        Self::Invert,
        Self::Absolute,
        Self::Add,
        Self::ReflectedAdd,
        Self::Subtract,
        Self::ReflectedSubtract,
        Self::Multiply,
        Self::ReflectedMultiply,
        Self::MatrixMultiply,
        Self::ReflectedMatrixMultiply,
        Self::Power,
        Self::ReflectedPower,
        Self::Divide,
        Self::ReflectedDivide,
        Self::FloorDivide,
        Self::ReflectedFloorDivide,
        Self::Remainder,
        Self::ReflectedRemainder,
        Self::DivMod,
        Self::ReflectedDivMod,
        Self::LeftShift,
        Self::ReflectedLeftShift,
        Self::RightShift,
        Self::ReflectedRightShift,
        Self::BitwiseAnd,
        Self::ReflectedBitwiseAnd,
        Self::BitwiseXor,
        Self::ReflectedBitwiseXor,
        Self::BitwiseOr,
        Self::ReflectedBitwiseOr,
        Self::Equal,
        Self::NotEqual,
        Self::LessThan,
        Self::LessEqual,
        Self::GreaterThan,
        Self::GreaterEqual,
        Self::Contains,
        Self::DeleteItem,
        Self::Format,
        Self::Reversed,
        Self::DeleteAttribute,
        Self::ClassGetItem,
        Self::InitSubclass,
        Self::InplaceAdd,
        Self::InplaceSubtract,
        Self::InplaceMultiply,
        Self::InplaceMatrixMultiply,
        Self::InplacePower,
        Self::InplaceDivide,
        Self::InplaceFloorDivide,
        Self::InplaceRemainder,
        Self::InplaceLeftShift,
        Self::InplaceRightShift,
        Self::InplaceBitwiseAnd,
        Self::InplaceBitwiseXor,
        Self::InplaceBitwiseOr,
    ];

    pub(super) fn from_index(index: u8) -> Self {
        *Self::ALL
            .get(index as usize)
            .expect("invalid private slot index")
    }
}

/// Python names and call shapes for implicitly dispatched slots. Native wrappers are created
/// from this table only when the defining type supplies the corresponding local slot.
/// The number of protocol slots, which sizes every type's slot table.
pub(super) const SLOT_COUNT: usize = 72;

pub(super) const SLOT_DEFS: [(Slot, &str, u8); SLOT_COUNT] = [
    (Slot::Call, "__call__", 255),
    (Slot::New, "__new__", 255),
    (Slot::Init, "__init__", 255),
    (Slot::GetAttribute, "__getattribute__", 1),
    (Slot::SetAttribute, "__setattr__", 2),
    (Slot::Repr, "__repr__", 0),
    (Slot::String, "__str__", 0),
    (Slot::Bool, "__bool__", 0),
    (Slot::Hash, "__hash__", 0),
    (Slot::Iter, "__iter__", 0),
    (Slot::Next, "__next__", 0),
    (Slot::Length, "__len__", 0),
    (Slot::GetItem, "__getitem__", 1),
    (Slot::SetItem, "__setitem__", 2),
    (Slot::Positive, "__pos__", 0),
    (Slot::Negative, "__neg__", 0),
    (Slot::Invert, "__invert__", 0),
    (Slot::Absolute, "__abs__", 0),
    (Slot::Add, "__add__", 1),
    (Slot::ReflectedAdd, "__radd__", 1),
    (Slot::Subtract, "__sub__", 1),
    (Slot::ReflectedSubtract, "__rsub__", 1),
    (Slot::Multiply, "__mul__", 1),
    (Slot::ReflectedMultiply, "__rmul__", 1),
    (Slot::MatrixMultiply, "__matmul__", 1),
    (Slot::ReflectedMatrixMultiply, "__rmatmul__", 1),
    (Slot::Power, "__pow__", 1),
    (Slot::ReflectedPower, "__rpow__", 1),
    (Slot::Divide, "__truediv__", 1),
    (Slot::ReflectedDivide, "__rtruediv__", 1),
    (Slot::FloorDivide, "__floordiv__", 1),
    (Slot::ReflectedFloorDivide, "__rfloordiv__", 1),
    (Slot::Remainder, "__mod__", 1),
    (Slot::ReflectedRemainder, "__rmod__", 1),
    (Slot::DivMod, "__divmod__", 1),
    (Slot::ReflectedDivMod, "__rdivmod__", 1),
    (Slot::LeftShift, "__lshift__", 1),
    (Slot::ReflectedLeftShift, "__rlshift__", 1),
    (Slot::RightShift, "__rshift__", 1),
    (Slot::ReflectedRightShift, "__rrshift__", 1),
    (Slot::BitwiseAnd, "__and__", 1),
    (Slot::ReflectedBitwiseAnd, "__rand__", 1),
    (Slot::BitwiseXor, "__xor__", 1),
    (Slot::ReflectedBitwiseXor, "__rxor__", 1),
    (Slot::BitwiseOr, "__or__", 1),
    (Slot::ReflectedBitwiseOr, "__ror__", 1),
    (Slot::Equal, "__eq__", 1),
    (Slot::NotEqual, "__ne__", 1),
    (Slot::LessThan, "__lt__", 1),
    (Slot::LessEqual, "__le__", 1),
    (Slot::GreaterThan, "__gt__", 1),
    (Slot::GreaterEqual, "__ge__", 1),
    (Slot::Contains, "__contains__", 1),
    (Slot::DeleteItem, "__delitem__", 1),
    (Slot::Format, "__format__", 1),
    (Slot::Reversed, "__reversed__", 0),
    (Slot::DeleteAttribute, "__delattr__", 1),
    (Slot::ClassGetItem, "__class_getitem__", 1),
    (Slot::InitSubclass, "__init_subclass__", 255),
    (Slot::InplaceAdd, "__iadd__", 1),
    (Slot::InplaceSubtract, "__isub__", 1),
    (Slot::InplaceMultiply, "__imul__", 1),
    (Slot::InplaceMatrixMultiply, "__imatmul__", 1),
    (Slot::InplacePower, "__ipow__", 1),
    (Slot::InplaceDivide, "__itruediv__", 1),
    (Slot::InplaceFloorDivide, "__ifloordiv__", 1),
    (Slot::InplaceRemainder, "__imod__", 1),
    (Slot::InplaceLeftShift, "__ilshift__", 1),
    (Slot::InplaceRightShift, "__irshift__", 1),
    (Slot::InplaceBitwiseAnd, "__iand__", 1),
    (Slot::InplaceBitwiseXor, "__ixor__", 1),
    (Slot::InplaceBitwiseOr, "__ior__", 1),
];

impl TypeSlots {
    fn from_attributes(owner: TypeId, attributes: &HashMap<String, Ref>) -> Self {
        let mut slots = Self::default();
        for (slot, name, _) in SLOT_DEFS {
            if let Some(value) = attributes.get(name) {
                // The slot table lives in the registry root alongside the attributes, so the
                // duplicate reference is traced and rewritten with them.
                slots.set(
                    slot,
                    SlotValue::Descriptor {
                        value: value.dup(),
                        owner,
                    },
                );
            }
        }
        slots
    }

    fn populated_count(&self) -> usize {
        self.0.iter().filter(|slot| slot.is_some()).count()
    }

    pub fn get(&self, slot: Slot) -> Option<&SlotValue> {
        self.0[slot as usize].as_ref()
    }

    fn set(&mut self, slot: Slot, value: SlotValue) {
        self.0[slot as usize] = Some(value);
    }

    /// Every cached descriptor reference, for the registry's root set.
    fn visit_refs(&self, visitor: &mut dyn FnMut(&Ref)) {
        for entry in &self.0 {
            if let Some(SlotValue::Descriptor { value, .. }) = entry {
                visitor(value);
            }
        }
    }
}

impl Clone for TypeSlots {
    fn clone(&self) -> Self {
        Self(std::array::from_fn(|index| self.0[index].clone()))
    }
}

/// Copies descriptor references with [`Ref::dup`]: slot values live only in the registry root.
impl Clone for SlotValue {
    fn clone(&self) -> Self {
        match self {
            Self::Descriptor { value, owner } => Self::Descriptor {
                value: value.dup(),
                owner: *owner,
            },
            Self::NativeMethod(method) => Self::NativeMethod(method),
            Self::VmRepr => Self::VmRepr,
            Self::VmHash => Self::VmHash,
            Self::NativeCompare(function) => Self::NativeCompare(*function),
            Self::NativeBinary(function) => Self::NativeBinary(*function),
            Self::NativeTernary(function) => Self::NativeTernary(*function),
            Self::NativeUnary(function) => Self::NativeUnary(*function),
        }
    }
}

/// Duplicate the stored references of an attribute map that stays inside the registry root.
fn dup_attributes(attributes: &HashMap<String, Ref>) -> HashMap<String, Ref> {
    attributes
        .iter()
        .map(|(name, value)| (name.clone(), value.dup()))
        .collect()
}

/// Metadata shared by builtin, native, and user-defined Python types.
#[derive(Debug)]
pub struct PyType {
    pub name: String,
    pub kind: TypeKind,
    pub bases: Vec<TypeId>,
    pub mro: Vec<TypeId>,
    pub attributes: HashMap<String, Ref>,
    /// Slots defined by this type, before MRO resolution.
    pub local_slots: TypeSlots,
    pub slots: TypeSlots,
    value: Option<Ref>,
}

impl Clone for PyType {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            kind: self.kind,
            bases: self.bases.clone(),
            mro: self.mro.clone(),
            attributes: dup_attributes(&self.attributes),
            local_slots: self.local_slots.clone(),
            slots: self.slots.clone(),
            value: self.value.as_ref().map(Ref::dup),
        }
    }
}

/// How a type came to be registered, which decides what its instances are made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeKind {
    /// A builtin type or builtin exception type, whose instances are payloads with no attribute
    /// storage.
    Builtin,
    /// A native value kind, whose instances are immediates or wide values.
    ValueKind,
    /// A class created by a class statement or `type()`. Its instances are heap objects whose
    /// header names the class and may carry attributes; their payload is the layout the class
    /// inherits, [`Object::Bare`](super::heap::Object::Bare) for a plain class.
    Class,
}

/// Per-runtime registry containing all semantic Python types.
#[derive(Debug)]
pub struct TypeRegistry {
    types: Vec<PyType>,
    value_kinds: Vec<&'static super::native::ValueKindDef>,
    exception_types: HashMap<&'static str, TypeId>,
    /// The inverse of `exception_types`, for the builtin ancestor of an exception type.
    exception_names: HashMap<TypeId, &'static str>,
    modeled_bytes: u64,
}

/// A copy of the registry is a separate root; the heap it describes must be copied with it.
impl Clone for TypeRegistry {
    fn clone(&self) -> Self {
        Self {
            types: self.types.clone(),
            value_kinds: self.value_kinds.clone(),
            exception_types: self.exception_types.clone(),
            exception_names: self.exception_names.clone(),
            modeled_bytes: self.modeled_bytes,
        }
    }
}

/// Heap values retained by semantic type metadata.
///
/// Completed user class objects own their Python-visible namespace. The registry roots those
/// class objects, builtin attributes, and cached protocol descriptors in both slot tables.
impl heap::Roots for TypeRegistry {
    fn visit_refs(&self, visitor: &mut dyn FnMut(&Ref)) {
        for ty in &self.types {
            for value in ty.attributes.values() {
                visitor(value);
            }
            if let Some(value) = &ty.value {
                visitor(value);
            }
            ty.local_slots.visit_refs(visitor);
            ty.slots.visit_refs(visitor);
        }
    }
}

/// Modeled memory of one type: its name, bases, MRO, attributes and the slots it defines. The
/// resolved slot table is a cache of the MRO, not data the program allocated, so it is free.
fn modeled_type_bytes(ty: &PyType) -> u64 {
    let bytes = 64usize
        .saturating_add(ty.name.len())
        .saturating_add(ty.bases.len().saturating_mul(4))
        .saturating_add(ty.mro.len().saturating_mul(4))
        .saturating_add(ty.attributes.len().saturating_mul(48))
        .saturating_add(ty.local_slots.populated_count().saturating_mul(24));
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

impl Default for TypeRegistry {
    fn default() -> Self {
        let mut types = Vec::with_capacity(BuiltinType::ALL.len());
        for builtin in BuiltinType::ALL {
            let (bases, mro) = builtin_metadata(builtin);
            types.push(PyType {
                name: builtin.name().into(),
                kind: TypeKind::Builtin,
                bases,
                mro,
                attributes: HashMap::new(),
                local_slots: TypeSlots::default(),
                slots: TypeSlots::default(),
                value: Some(Ref::from_immediate(Value::Native(
                    super::vm::NativeValue::BuiltinType(builtin),
                ))),
            });
        }
        install_native_attributes(
            &mut types[BuiltinType::Object as usize],
            &super::stdlib::core::OBJECT_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Enum as usize],
            &super::stdlib::r#enum::ENUM_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Type as usize],
            &super::stdlib::core::TYPE_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Module as usize],
            &super::stdlib::core::MODULE_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Exception as usize],
            &super::stdlib::core::EXCEPTION_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Iterator as usize],
            &super::stdlib::core::ITERATOR_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::String as usize],
            &super::stdlib::core::STRING_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Bytes as usize],
            &super::stdlib::core::BYTES_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::ByteArray as usize],
            &super::stdlib::core::BYTEARRAY_TYPE,
        );
        install_native_class_methods(
            &mut types[BuiltinType::Bytes as usize],
            super::stdlib::core::BYTES_CLASS_METHODS,
        );
        install_native_class_methods(
            &mut types[BuiltinType::ByteArray as usize],
            super::stdlib::core::BYTEARRAY_CLASS_METHODS,
        );
        install_native_attributes(
            &mut types[BuiltinType::List as usize],
            &super::stdlib::core::LIST_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Slice as usize],
            &super::stdlib::core::SLICE_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Tuple as usize],
            &super::stdlib::core::TUPLE_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Dict as usize],
            &super::stdlib::core::DICT_TYPE,
        );
        install_native_class_methods(
            &mut types[BuiltinType::Dict as usize],
            super::stdlib::core::DICT_CLASS_METHODS,
        );
        install_native_attributes(
            &mut types[BuiltinType::DictKeys as usize],
            &super::stdlib::mapping_views::DICT_KEYS_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::DictValues as usize],
            &super::stdlib::mapping_views::DICT_VALUES_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::DictItems as usize],
            &super::stdlib::mapping_views::DICT_ITEMS_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::MappingProxy as usize],
            &super::stdlib::mapping_views::MAPPING_PROXY_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Set as usize],
            &super::stdlib::core::SET_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::FrozenSet as usize],
            &super::stdlib::core::FROZENSET_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Generator as usize],
            &super::stdlib::core::GENERATOR_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Property as usize],
            &super::stdlib::core::PROPERTY_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Regex as usize],
            &super::stdlib::re::PATTERN_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Match as usize],
            &super::stdlib::re::MATCH_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Stream as usize],
            &super::stdlib::sys::STREAM_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::Environment as usize],
            &super::stdlib::os::ENVIRONMENT_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::ArgumentParser as usize],
            &super::stdlib::argparse::ARGUMENT_PARSER_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::RaisesContext as usize],
            &super::stdlib::unittest::RAISES_CONTEXT_TYPE,
        );
        install_native_attributes(
            &mut types[BuiltinType::TestCase as usize],
            &super::stdlib::unittest::TEST_CASE_TYPE,
        );
        for definition in super::stdlib::numpy::array_types() {
            install_native_attributes(&mut types[BuiltinType::Array as usize], definition);
        }
        install_number_attributes(&mut types);
        // A namespace view is a `dict` whose storage is a module scope, the script table or an
        // instance's attributes. The runtime's dict accessors read and write that storage, so
        // `dict`'s own methods serve it unchanged.
        types[BuiltinType::NamespaceDict as usize].attributes =
            dup_attributes(&types[BuiltinType::Dict as usize].attributes);
        for ty in &mut types {
            install_native_method_slots(ty);
        }
        install_builtin_slots(&mut types);
        for ty in &mut types {
            ty.local_slots = ty.slots.clone();
        }
        // bool uses int's arithmetic slots, but defines only these bitwise operators itself.
        let bool_slots = &types[BuiltinType::Bool as usize].slots;
        let mut bool_local = TypeSlots::default();
        for slot in [
            Slot::Invert,
            Slot::BitwiseAnd,
            Slot::ReflectedBitwiseAnd,
            Slot::BitwiseOr,
            Slot::ReflectedBitwiseOr,
            Slot::BitwiseXor,
            Slot::ReflectedBitwiseXor,
        ] {
            bool_local.set(
                slot,
                bool_slots.get(slot).expect("bool slot installed").clone(),
            );
        }
        types[BuiltinType::Bool as usize].local_slots = bool_local;
        resolve_builtin_slots(&mut types);
        install_slot_wrappers(&mut types);
        // Mutable containers publish `__hash__ = None`, which is how `collections.abc.Hashable`
        // and user code detect that they are unhashable.
        for builtin in [
            BuiltinType::List,
            BuiltinType::Dict,
            BuiltinType::Set,
            BuiltinType::ByteArray,
        ] {
            types[builtin as usize]
                .attributes
                .insert("__hash__".into(), Ref::from_immediate(Value::None));
        }
        let builtin_bytes = types
            .iter()
            .map(modeled_type_bytes)
            .fold(0u64, u64::saturating_add);
        let mut registry = Self {
            types,
            value_kinds: Vec::new(),
            exception_types: HashMap::new(),
            exception_names: HashMap::new(),
            modeled_bytes: builtin_bytes,
        };
        for kind in super::stdlib::value_kinds() {
            registry.register_value_kind(kind);
        }
        registry.register_exception_types();
        registry
    }
}

impl TypeRegistry {
    /// Builtin exception classes keep their public values while sharing the registry's C3 MRO.
    /// Registering them after value kinds preserves the compact registered-value indices.
    fn register_exception_types(&mut self) {
        use super::vm::{ExceptionType, NativeValue};

        let base = BuiltinType::Exception.id();
        self.types[base.raw() as usize].value = Some(Ref::from_immediate(Value::Native(
            NativeValue::ExceptionType(ExceptionType("BaseException")),
        )));
        // Exception instances keep an attribute dictionary, like instances of heap classes.
        let base_type = &mut self.types[base.raw() as usize];
        let before = modeled_type_bytes(base_type);
        base_type.attributes.insert(
            "__dict__".into(),
            Ref::from_immediate(Value::Native(NativeValue::NativeGetter(
                &super::stdlib::core::INSTANCE_DICT_GETTER,
            ))),
        );
        let after = modeled_type_bytes(base_type);
        self.modeled_bytes = self
            .modeled_bytes
            .saturating_sub(before)
            .saturating_add(after);
        self.exception_types.insert("BaseException", base);
        self.exception_names.insert(base, "BaseException");
        for definition in super::exception_types::EXCEPTION_TYPES.iter().skip(1) {
            let parent = definition.parent.expect("non-root exception has a parent");
            let parent = self.exception_types[parent];
            let mut bases = vec![parent];
            if let Some(secondary) = definition.secondary_parent {
                bases.push(self.exception_types[secondary]);
            }
            let mro = self
                .linearize_bases(&bases)
                .expect("exception MRO is valid");
            let value = Ref::from_immediate(Value::Native(NativeValue::ExceptionType(
                ExceptionType(definition.name),
            )));
            let mut ty = PyType {
                name: definition.name.into(),
                kind: TypeKind::Builtin,
                bases,
                mro: mro.clone(),
                attributes: HashMap::new(),
                local_slots: TypeSlots::default(),
                slots: self.inherit_slots(TypeSlots::default(), &mro),
                value: Some(value),
            };
            if definition.name == super::stdlib::core::OS_ERROR_TYPE.name {
                install_native_attributes(&mut ty, &super::stdlib::core::OS_ERROR_TYPE);
            }
            let id = TypeId(u32::try_from(self.types.len()).expect("too many registered types"));
            self.modeled_bytes = self.modeled_bytes.saturating_add(modeled_type_bytes(&ty));
            self.types.push(ty);
            self.exception_types.insert(definition.name, id);
            self.exception_names.insert(id, definition.name);
        }
    }

    pub(super) fn exception_type_id(&self, name: &str) -> Option<TypeId> {
        self.exception_types.get(name).copied()
    }

    /// Whether instances of `id` are exceptions: `id` derives from `BaseException`.
    pub fn is_exception_type(&self, id: TypeId) -> PyResult<bool> {
        self.is_subclass(id, BuiltinType::Exception.id())
    }

    /// The closest builtin exception class that `id` is or derives from, which decides how its
    /// instances render and which native behavior they inherit. `None` for non-exception types.
    pub fn exception_base(&self, id: TypeId) -> PyResult<Option<&'static str>> {
        if let Some(name) = self.exception_names.get(&id) {
            return Ok(Some(name));
        }
        Ok(self
            .get(id)?
            .mro
            .iter()
            .find_map(|ancestor| self.exception_names.get(ancestor).copied()))
    }

    /// Conservative modeled size of registry metadata retained between executions.
    pub fn modeled_bytes(&self) -> u64 {
        self.modeled_bytes
    }

    pub fn get(&self, id: TypeId) -> PyResult<&PyType> {
        self.types
            .get(id.0 as usize)
            .ok_or_else(|| "invalid type reference".into())
    }

    /// The stored reference to the type object registered for `id`.
    pub fn value_ref(&self, id: TypeId) -> PyResult<&Ref> {
        self.get(id)?
            .value
            .as_ref()
            .ok_or_else(|| "type construction is incomplete".into())
    }

    /// The class object of `value` when `value` is an instance of a user class, whatever payload
    /// the class's layout gave it. Class objects and enum members resolve through their own
    /// paths and return `None` here, as do builtin values.
    pub fn instance_class(&self, heap: &Heap, value: Value) -> PyResult<Option<Value>> {
        if !value.is_object() {
            return Ok(None);
        }
        let ty = self.get(heap.type_id(value)?)?;
        if ty.kind != TypeKind::Class {
            return Ok(None);
        }
        if matches!(heap.get(value)?, Object::Class(_)) {
            return Ok(None);
        }
        let class = ty
            .value
            .as_ref()
            .ok_or("instance of a class whose construction is incomplete")?;
        Ok(Some(heap.value(class)))
    }

    pub fn register(
        &mut self,
        name: String,
        bases: Vec<TypeId>,
        mro: Vec<TypeId>,
        attributes: &HashMap<String, Ref>,
    ) -> PyResult<TypeId> {
        let index = u32::try_from(self.types.len()).map_err(|_| "too many Python types")?;
        let local_slots = TypeSlots::from_attributes(TypeId(index), attributes);
        let slots = self.inherit_slots(local_slots.clone(), &mro);
        let ty = PyType {
            name,
            kind: TypeKind::Class,
            bases,
            mro,
            attributes: HashMap::new(),
            local_slots,
            slots,
            value: None,
        };
        self.modeled_bytes = self.modeled_bytes.saturating_add(modeled_type_bytes(&ty));
        self.types.push(ty);
        Ok(TypeId(index))
    }

    /// Fill the slots a type does not define from the first ancestor in `mro` that does.
    fn inherit_slots(&self, mut slots: TypeSlots, mro: &[TypeId]) -> TypeSlots {
        for slot in Slot::ALL {
            if slots.get(slot).is_some() {
                continue;
            }
            if let Some(value) = mro
                .iter()
                .find_map(|ancestor| self.get(*ancestor).ok()?.slots.get(slot).cloned())
            {
                slots.set(slot, value);
            }
        }
        slots
    }

    /// The number of registered types, which bounds the work of [`TypeRegistry::subtypes`].
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// `id` and the types that derive from it, in registration order, which places every base
    /// before its subclasses.
    pub fn subtypes(&self, id: TypeId) -> Vec<TypeId> {
        (0..self.types.len())
            .filter_map(|index| u32::try_from(index).ok().map(TypeId))
            .filter(|candidate| {
                *candidate == id || self.types[candidate.0 as usize].mro.contains(&id)
            })
            .collect()
    }

    /// Recompute a user type's slots from its class attributes after one of them changed,
    /// inheriting the rest through its MRO as [`TypeRegistry::register`] does.
    pub fn replace_slots(&mut self, id: TypeId, attributes: &HashMap<String, Ref>) -> PyResult<()> {
        let mro = self.get(id)?.mro.clone();
        let local_slots = TypeSlots::from_attributes(id, attributes);
        let slots = self.inherit_slots(local_slots.clone(), &mro);
        let ty = self
            .types
            .get_mut(id.0 as usize)
            .ok_or("invalid type reference")?;
        let before = modeled_type_bytes(ty);
        ty.local_slots = local_slots;
        ty.slots = slots;
        let after = modeled_type_bytes(ty);
        self.modeled_bytes = self
            .modeled_bytes
            .saturating_sub(before)
            .saturating_add(after);
        Ok(())
    }

    pub fn finish(&mut self, id: TypeId, value: Ref) -> PyResult<()> {
        let ty = self
            .types
            .get_mut(id.0 as usize)
            .ok_or("invalid type reference")?;
        if ty.value.is_some() {
            return Err("type construction finished more than once".into());
        }
        ty.value = Some(value);
        Ok(())
    }

    pub fn is_subclass(&self, class: TypeId, base: TypeId) -> PyResult<bool> {
        Ok(class == base || self.get(class)?.mro.contains(&base))
    }

    /// Linearize every registered base, builtin or heap-owned, with the same C3 rule used to
    /// register native value kinds. The result excludes the class being created.
    pub(super) fn linearize_bases(&self, bases: &[TypeId]) -> PyResult<Vec<TypeId>> {
        for (index, base) in bases.iter().enumerate() {
            if bases[..index].contains(base) {
                return Err(PyError::type_error("duplicate base class"));
            }
        }
        let mut sequences = Vec::with_capacity(bases.len().saturating_add(1));
        for base in bases {
            let ty = self.get(*base)?;
            let mut sequence = Vec::with_capacity(ty.mro.len().saturating_add(1));
            sequence.push(*base);
            sequence.extend(&ty.mro);
            sequences.push(sequence);
        }
        sequences.push(bases.to_vec());
        c3_merge(sequences).ok_or_else(|| {
            PyError::type_error("cannot create a consistent method resolution order")
        })
    }

    pub fn slot(&self, type_id: TypeId, slot: Slot) -> PyResult<Option<SlotValue>> {
        Ok(self.get(type_id)?.slots.get(slot).cloned())
    }

    pub(super) fn local_slot(&self, type_id: TypeId, slot: Slot) -> PyResult<Option<SlotValue>> {
        Ok(self.get(type_id)?.local_slots.get(slot).cloned())
    }

    /// Register a module-owned value kind after its bases, linearizing them with C3 as a class
    /// statement would. Slots and attributes not defined by the kind are inherited through the
    /// MRO, so `np.float64` finds `float` methods.
    fn register_value_kind(&mut self, kind: &'static super::native::ValueKindDef) {
        use super::native::KindBase;
        let object = BuiltinType::Object.id();
        let type_id = TypeId(u32::try_from(self.types.len()).expect("too many registered types"));
        let bases = if kind.bases.is_empty() {
            vec![object]
        } else {
            kind.bases
                .iter()
                .map(|base| match base {
                    KindBase::Kind(base) => self
                        .value_kind_type_id(base)
                        .expect("value kind bases are registered first"),
                    KindBase::Float => BuiltinType::Float.id(),
                    KindBase::Complex => BuiltinType::Complex.id(),
                })
                .collect::<Vec<_>>()
        };
        let mut sequences = bases
            .iter()
            .map(|base| {
                let mut sequence = vec![*base];
                sequence.extend(&self.types[base.0 as usize].mro);
                sequence
            })
            .collect::<Vec<_>>();
        sequences.push(bases.clone());
        let mro = c3_merge(sequences).expect("value kind bases have a consistent MRO");
        let mut local_slots = value_kind_slots(kind.slots);
        local_slots.set(
            Slot::Format,
            SlotValue::NativeBinary(super::stdlib::core::slot_builtin_format),
        );
        let mut slots = local_slots.clone();
        for slot in Slot::ALL {
            if slots.get(slot).is_none() {
                if let Some(value) = mro
                    .iter()
                    .find_map(|ancestor| self.types[ancestor.0 as usize].slots.get(slot).cloned())
                {
                    slots.set(slot, value);
                }
            }
        }
        let mut ty = PyType {
            name: kind.name.into(),
            kind: TypeKind::ValueKind,
            bases,
            mro,
            attributes: HashMap::new(),
            local_slots,
            slots,
            value: Some(Ref::from_immediate(Value::Native(
                super::vm::NativeValue::ValueKind(kind),
            ))),
        };
        insert_native_attributes(&mut ty, kind.methods, kind.getters);
        install_slot_wrappers_for_type(&mut ty, type_id);
        self.modeled_bytes = self.modeled_bytes.saturating_add(modeled_type_bytes(&ty));
        self.types.push(ty);
        self.value_kinds.push(kind);
        debug_assert_eq!(self.value_kind_type_id(kind), Some(type_id));
    }

    pub(super) fn value_kind_index(
        &self,
        kind: &'static super::native::ValueKindDef,
    ) -> Option<u8> {
        self.value_kinds
            .iter()
            .position(|candidate| std::ptr::eq(*candidate, kind))
            .and_then(|index| u8::try_from(index).ok())
    }

    pub(super) fn value_kind_type_id(
        &self,
        kind: &'static super::native::ValueKindDef,
    ) -> Option<TypeId> {
        let index = self.value_kind_index(kind)?;
        self.value_kind_type_id_by_index(index)
    }

    pub(super) fn value_kind_type_id_by_index(&self, index: u8) -> Option<TypeId> {
        self.value_kinds.get(index as usize)?;
        let offset = BuiltinType::ALL.len().checked_add(index as usize)?;
        Some(TypeId(u32::try_from(offset).ok()?))
    }

    pub(super) fn value_kind(&self, index: u8) -> Option<&'static super::native::ValueKindDef> {
        self.value_kinds.get(index as usize).copied()
    }
}

/// C3 linearization of `sequences`, each a base followed by its MRO, plus the base list itself.
fn c3_merge(mut sequences: Vec<Vec<TypeId>>) -> Option<Vec<TypeId>> {
    let mut merged = Vec::new();
    loop {
        sequences.retain(|sequence| !sequence.is_empty());
        if sequences.is_empty() {
            return Some(merged);
        }
        let head = sequences
            .iter()
            .map(|sequence| sequence[0])
            .find(|candidate| {
                sequences
                    .iter()
                    .all(|sequence| !sequence[1..].contains(candidate))
            })?;
        merged.push(head);
        for sequence in &mut sequences {
            if sequence[0] == head {
                sequence.remove(0);
            }
        }
    }
}

fn value_kind_slots(slots: super::native::ValueKindSlots) -> TypeSlots {
    let mut table = TypeSlots::default();
    let unary = SlotValue::NativeUnary;
    let binary = SlotValue::NativeBinary;
    if let Some(function) = slots.repr {
        table.set(Slot::Repr, unary(function));
    }
    if let Some(function) = slots.str_ {
        table.set(Slot::String, unary(function));
    }
    if let Some(function) = slots.bool_ {
        table.set(Slot::Bool, unary(function));
    }
    if let Some(function) = slots.get_item {
        table.set(Slot::GetItem, binary(function));
    }
    if let Some(function) = slots.positive {
        table.set(Slot::Positive, unary(function));
    }
    if let Some(function) = slots.negative {
        table.set(Slot::Negative, unary(function));
    }
    if let Some(function) = slots.invert {
        table.set(Slot::Invert, unary(function));
    }
    if let Some(function) = slots.absolute {
        table.set(Slot::Absolute, unary(function));
    }
    if let Some(function) = slots.add {
        table.set(Slot::Add, binary(function));
    }
    if let Some(function) = slots.reflected_add {
        table.set(Slot::ReflectedAdd, binary(function));
    }
    if let Some(function) = slots.subtract {
        table.set(Slot::Subtract, binary(function));
    }
    if let Some(function) = slots.reflected_subtract {
        table.set(Slot::ReflectedSubtract, binary(function));
    }
    if let Some(function) = slots.multiply {
        table.set(Slot::Multiply, binary(function));
    }
    if let Some(function) = slots.reflected_multiply {
        table.set(Slot::ReflectedMultiply, binary(function));
    }
    if let Some(function) = slots.divide {
        table.set(Slot::Divide, binary(function));
    }
    if let Some(function) = slots.reflected_divide {
        table.set(Slot::ReflectedDivide, binary(function));
    }
    if let Some(function) = slots.floor_divide {
        table.set(Slot::FloorDivide, binary(function));
    }
    if let Some(function) = slots.reflected_floor_divide {
        table.set(Slot::ReflectedFloorDivide, binary(function));
    }
    if let Some(function) = slots.remainder {
        table.set(Slot::Remainder, binary(function));
    }
    if let Some(function) = slots.reflected_remainder {
        table.set(Slot::ReflectedRemainder, binary(function));
    }
    if let Some(function) = slots.divmod {
        table.set(Slot::DivMod, binary(function));
    }
    if let Some(function) = slots.reflected_divmod {
        table.set(Slot::ReflectedDivMod, binary(function));
    }
    if let Some(function) = slots.power {
        table.set(Slot::Power, binary(function));
    }
    if let Some(function) = slots.reflected_power {
        table.set(Slot::ReflectedPower, binary(function));
    }
    if let Some(function) = slots.left_shift {
        table.set(Slot::LeftShift, binary(function));
    }
    if let Some(function) = slots.reflected_left_shift {
        table.set(Slot::ReflectedLeftShift, binary(function));
    }
    if let Some(function) = slots.right_shift {
        table.set(Slot::RightShift, binary(function));
    }
    if let Some(function) = slots.reflected_right_shift {
        table.set(Slot::ReflectedRightShift, binary(function));
    }
    if let Some(function) = slots.bitwise_and {
        table.set(Slot::BitwiseAnd, binary(function));
    }
    if let Some(function) = slots.reflected_bitwise_and {
        table.set(Slot::ReflectedBitwiseAnd, binary(function));
    }
    if let Some(function) = slots.bitwise_xor {
        table.set(Slot::BitwiseXor, binary(function));
    }
    if let Some(function) = slots.reflected_bitwise_xor {
        table.set(Slot::ReflectedBitwiseXor, binary(function));
    }
    if let Some(function) = slots.bitwise_or {
        table.set(Slot::BitwiseOr, binary(function));
    }
    if let Some(function) = slots.reflected_bitwise_or {
        table.set(Slot::ReflectedBitwiseOr, binary(function));
    }
    if let Some(function) = slots.equal {
        table.set(Slot::Equal, binary(function));
    }
    if let Some(function) = slots.not_equal {
        table.set(Slot::NotEqual, binary(function));
    }
    if let Some(function) = slots.less_than {
        table.set(Slot::LessThan, binary(function));
    }
    if let Some(function) = slots.less_equal {
        table.set(Slot::LessEqual, binary(function));
    }
    if let Some(function) = slots.greater_than {
        table.set(Slot::GreaterThan, binary(function));
    }
    if let Some(function) = slots.greater_equal {
        table.set(Slot::GreaterEqual, binary(function));
    }
    table
}

fn builtin_metadata(builtin: BuiltinType) -> (Vec<TypeId>, Vec<TypeId>) {
    let object = BuiltinType::Object.id();
    match builtin {
        BuiltinType::Object => (Vec::new(), Vec::new()),
        BuiltinType::Bool => (
            vec![BuiltinType::Int.id()],
            vec![BuiltinType::Int.id(), object],
        ),
        BuiltinType::NamespaceDict => (
            vec![BuiltinType::Dict.id()],
            vec![BuiltinType::Dict.id(), object],
        ),
        _ => (vec![object], vec![object]),
    }
}

fn install_native_attributes(ty: &mut PyType, definition: &'static super::native::NativeTypeDef) {
    debug_assert_eq!(ty.name, definition.name);
    debug_assert!(definition
        .methods
        .iter()
        .all(|method| method.type_name == definition.name));
    debug_assert!(definition
        .getters
        .iter()
        .all(|getter| getter.owner == definition.name));
    insert_native_attributes(ty, definition.methods, definition.getters);
}

fn insert_native_attributes(
    ty: &mut PyType,
    methods: &'static [super::native::MethodDef],
    getters: &'static [super::native::GetterDef],
) {
    for method in methods {
        ty.attributes.insert(
            method.name.into(),
            Ref::from_immediate(Value::Native(super::vm::NativeValue::NativeMethod(method))),
        );
    }
    for getter in getters {
        ty.attributes.insert(
            getter.name.into(),
            Ref::from_immediate(Value::Native(super::vm::NativeValue::NativeGetter(getter))),
        );
    }
}

/// Native methods with protocol names participate in the same slot resolution as user methods.
/// Specialized builtin slots installed later replace these when a direct implementation exists.
fn install_native_method_slots(ty: &mut PyType) {
    for (slot, name, _) in SLOT_DEFS {
        if !matches!(
            slot,
            Slot::Call
                | Slot::New
                | Slot::Init
                | Slot::InitSubclass
                | Slot::ClassGetItem
                | Slot::Format
                | Slot::Iter
                | Slot::Next
        ) {
            continue;
        }
        if let Some(method) =
            ty.attributes
                .get(name)
                .and_then(|value| match value.immediate()?.native_value() {
                    Some(super::vm::NativeValue::NativeMethod(method)) => Some(method),
                    _ => None,
                })
        {
            ty.slots.set(slot, SlotValue::NativeMethod(method));
        }
    }
}

/// Install methods such as `dict.fromkeys` that receive the type rather than an instance.
fn install_native_class_methods(ty: &mut PyType, methods: &'static [super::native::MethodDef]) {
    debug_assert!(methods.iter().all(|method| method.type_name == ty.name));
    for method in methods {
        ty.attributes.insert(
            method.name.into(),
            Ref::from_immediate(Value::Native(super::vm::NativeValue::NativeClassMethod(
                method,
            ))),
        );
    }
}

/// Install the numeric-tower accessors (`real`, `imag`, `conjugate`, and the rational
/// accessors on integers), the integer methods such as `to_bytes` and `from_bytes`, and the
/// float methods on the builtin number types, including `complex`.
///
/// The builtin namespace contains only methods defined by that type. Lookup walks the MRO,
/// so `bool` inherits integer methods without copying their entries.
fn install_number_attributes(types: &mut [PyType]) {
    install_native_attributes(
        &mut types[BuiltinType::Int as usize],
        &super::number::INT_TYPE,
    );
    install_native_attributes(
        &mut types[BuiltinType::Int as usize],
        &super::number::INT_CONSTRUCTOR,
    );
    install_native_class_methods(
        &mut types[BuiltinType::Int as usize],
        super::number::INT_CLASS_METHODS,
    );
    install_native_attributes(
        &mut types[BuiltinType::Float as usize],
        &super::number::FLOAT_TYPE,
    );
    install_native_attributes(
        &mut types[BuiltinType::Complex as usize],
        &super::complex::COMPLEX_TYPE,
    );
}

/// Give every comparison slot of a builtin type the one native function that answers them all.
fn install_compare(slots: &mut TypeSlots, call: CompareSlotFn) {
    for slot in [
        Slot::Equal,
        Slot::NotEqual,
        Slot::LessThan,
        Slot::LessEqual,
        Slot::GreaterThan,
        Slot::GreaterEqual,
    ] {
        slots.set(slot, SlotValue::NativeCompare(call));
    }
}

fn install_builtin_slots(types: &mut [PyType]) {
    let intrinsic = SlotValue::NativeBinary;
    let unary = SlotValue::NativeUnary;
    // Every builtin value type compares through a comparison slot; `bool` inherits `int`'s.
    for builtin in [BuiltinType::Int, BuiltinType::Float, BuiltinType::Complex] {
        install_compare(
            &mut types[builtin as usize].slots,
            super::number::slot_number_compare,
        );
    }
    install_compare(
        &mut types[BuiltinType::String as usize].slots,
        super::stdlib::core::slot_str_compare,
    );
    for builtin in [BuiltinType::Bytes, BuiltinType::ByteArray] {
        install_compare(
            &mut types[builtin as usize].slots,
            super::stdlib::core::slot_bytes_compare,
        );
    }
    for builtin in [
        BuiltinType::Range,
        BuiltinType::Slice,
        BuiltinType::GenericAlias,
    ] {
        install_compare(
            &mut types[builtin as usize].slots,
            super::stdlib::core::slot_structural_compare,
        );
    }
    types[BuiltinType::None as usize]
        .slots
        .set(Slot::Bool, unary(super::stdlib::core::slot_none_bool));
    types[BuiltinType::None as usize]
        .slots
        .set(Slot::Hash, unary(super::stdlib::core::slot_none_hash));
    for builtin in [BuiltinType::Tuple, BuiltinType::Range, BuiltinType::Slice] {
        types[builtin as usize]
            .slots
            .set(Slot::Hash, SlotValue::VmHash);
    }
    for builtin in [
        BuiltinType::None,
        BuiltinType::Ellipsis,
        BuiltinType::NotImplemented,
        BuiltinType::Bool,
        BuiltinType::Int,
        BuiltinType::Float,
        BuiltinType::String,
        BuiltinType::Bytes,
        BuiltinType::ByteArray,
        BuiltinType::List,
        BuiltinType::Tuple,
        BuiltinType::Dict,
        BuiltinType::Set,
        BuiltinType::FrozenSet,
        BuiltinType::Range,
        BuiltinType::NamespaceDict,
        BuiltinType::DictKeys,
        BuiltinType::DictValues,
        BuiltinType::DictItems,
        BuiltinType::MappingProxy,
        BuiltinType::Complex,
        BuiltinType::Slice,
        BuiltinType::Exception,
        BuiltinType::Module,
        BuiltinType::Function,
    ] {
        types[builtin as usize]
            .slots
            .set(Slot::Repr, SlotValue::VmRepr);
    }
    for builtin in [
        BuiltinType::List,
        BuiltinType::Tuple,
        BuiltinType::Dict,
        BuiltinType::Set,
        BuiltinType::FrozenSet,
        BuiltinType::Type,
    ] {
        types[builtin as usize].slots.set(
            Slot::ClassGetItem,
            intrinsic(super::stdlib::core::slot_generic_alias),
        );
    }
    types[BuiltinType::Object as usize].slots.set(
        Slot::Format,
        intrinsic(super::stdlib::core::slot_object_format),
    );
    for builtin in [
        BuiltinType::Int,
        BuiltinType::Float,
        BuiltinType::Complex,
        BuiltinType::String,
    ] {
        types[builtin as usize].slots.set(
            Slot::Format,
            intrinsic(super::stdlib::core::slot_builtin_format),
        );
    }
    for builtin in [BuiltinType::Bool, BuiltinType::Int, BuiltinType::Float] {
        let slots = &mut types[builtin as usize].slots;
        slots.set(Slot::Positive, unary(super::number::slot_positive));
        slots.set(Slot::Negative, unary(super::number::slot_negative));
        slots.set(Slot::Invert, unary(super::number::slot_invert));
        slots.set(Slot::Absolute, unary(super::number::slot_absolute));
        slots.set(Slot::Add, intrinsic(super::number::slot_add));
        slots.set(Slot::ReflectedAdd, intrinsic(super::number::slot_add));
        slots.set(Slot::Subtract, intrinsic(super::number::slot_subtract));
        slots.set(
            Slot::ReflectedSubtract,
            intrinsic(super::number::slot_reflected_subtract),
        );
        slots.set(Slot::Multiply, intrinsic(super::number::slot_multiply));
        slots.set(
            Slot::ReflectedMultiply,
            intrinsic(super::number::slot_multiply),
        );
        slots.set(Slot::Power, intrinsic(super::number::slot_power));
        slots.set(
            Slot::ReflectedPower,
            intrinsic(super::number::slot_reflected_power),
        );
        slots.set(Slot::Divide, intrinsic(super::number::slot_divide));
        slots.set(
            Slot::ReflectedDivide,
            intrinsic(super::number::slot_reflected_divide),
        );
        slots.set(
            Slot::FloorDivide,
            intrinsic(super::number::slot_floor_divide),
        );
        slots.set(
            Slot::ReflectedFloorDivide,
            intrinsic(super::number::slot_reflected_floor_divide),
        );
        slots.set(Slot::Remainder, intrinsic(super::number::slot_remainder));
        slots.set(
            Slot::ReflectedRemainder,
            intrinsic(super::number::slot_reflected_remainder),
        );
        slots.set(Slot::DivMod, intrinsic(super::number::slot_divmod));
        slots.set(
            Slot::ReflectedDivMod,
            intrinsic(super::number::slot_reflected_divmod),
        );
        slots.set(Slot::LeftShift, intrinsic(super::number::slot_left_shift));
        slots.set(
            Slot::ReflectedLeftShift,
            intrinsic(super::number::slot_left_shift),
        );
        slots.set(Slot::RightShift, intrinsic(super::number::slot_right_shift));
        slots.set(
            Slot::ReflectedRightShift,
            intrinsic(super::number::slot_right_shift),
        );
        slots.set(Slot::BitwiseAnd, intrinsic(super::number::slot_bitwise_and));
        slots.set(
            Slot::ReflectedBitwiseAnd,
            intrinsic(super::number::slot_bitwise_and),
        );
        slots.set(Slot::BitwiseXor, intrinsic(super::number::slot_bitwise_xor));
        slots.set(
            Slot::ReflectedBitwiseXor,
            intrinsic(super::number::slot_bitwise_xor),
        );
        slots.set(Slot::BitwiseOr, intrinsic(super::number::slot_bitwise_or));
        slots.set(
            Slot::ReflectedBitwiseOr,
            intrinsic(super::number::slot_bitwise_or),
        );
        slots.set(Slot::Equal, intrinsic(super::number::slot_equal));
        slots.set(Slot::NotEqual, intrinsic(super::number::slot_not_equal));
        slots.set(Slot::Hash, unary(super::number::slot_hash));
        slots.set(Slot::Bool, unary(super::number::slot_bool));
        slots.set(Slot::LessThan, intrinsic(super::number::slot_less));
        slots.set(Slot::LessEqual, intrinsic(super::number::slot_less_equal));
        slots.set(Slot::GreaterThan, intrinsic(super::number::slot_greater));
        slots.set(
            Slot::GreaterEqual,
            intrinsic(super::number::slot_greater_equal),
        );
    }
    let slots = &mut types[BuiltinType::String as usize].slots;
    slots.set(
        Slot::Equal,
        intrinsic(super::stdlib::core::slot_string_equal),
    );
    slots.set(
        Slot::NotEqual,
        intrinsic(super::stdlib::core::slot_string_not_equal),
    );
    slots.set(
        Slot::LessThan,
        intrinsic(super::stdlib::core::slot_string_less),
    );
    slots.set(
        Slot::LessEqual,
        intrinsic(super::stdlib::core::slot_string_less_equal),
    );
    slots.set(
        Slot::GreaterThan,
        intrinsic(super::stdlib::core::slot_string_greater),
    );
    slots.set(
        Slot::GreaterEqual,
        intrinsic(super::stdlib::core::slot_string_greater_equal),
    );
    slots.set(Slot::Hash, unary(super::stdlib::core::slot_string_hash));
    slots.set(
        Slot::Length,
        unary(super::stdlib::core::slot_builtin_length),
    );
    slots.set(Slot::Iter, unary(super::stdlib::core::slot_sequence_iter));
    slots.set(
        Slot::GetItem,
        intrinsic(super::stdlib::core::slot_builtin_get_item),
    );
    slots.set(
        Slot::Contains,
        intrinsic(super::stdlib::core::slot_builtin_contains),
    );
    slots.set(Slot::Add, intrinsic(super::stdlib::core::slot_string_add));
    slots.set(
        Slot::Multiply,
        intrinsic(super::stdlib::core::slot_string_multiply),
    );
    slots.set(
        Slot::ReflectedMultiply,
        intrinsic(super::stdlib::core::slot_string_multiply),
    );
    slots.set(
        Slot::Remainder,
        intrinsic(super::stdlib::core::slot_string_remainder),
    );

    let slots = &mut types[BuiltinType::Bytes as usize].slots;
    slots.set(Slot::Hash, unary(super::stdlib::core::slot_bytes_hash));
    slots.set(Slot::Length, unary(super::stdlib::core::slot_bytes_length));
    slots.set(Slot::Iter, unary(super::stdlib::core::slot_sequence_iter));
    slots.set(
        Slot::Contains,
        intrinsic(super::stdlib::core::slot_builtin_contains),
    );
    slots.set(
        Slot::GetItem,
        intrinsic(super::stdlib::core::slot_bytes_get_item),
    );
    slots.set(Slot::Add, intrinsic(super::stdlib::core::slot_bytes_add));
    slots.set(
        Slot::Multiply,
        intrinsic(super::stdlib::core::slot_bytes_multiply),
    );
    slots.set(
        Slot::ReflectedMultiply,
        intrinsic(super::stdlib::core::slot_bytes_multiply),
    );

    let slots = &mut types[BuiltinType::ByteArray as usize].slots;
    slots.set(
        Slot::Length,
        unary(super::stdlib::core::slot_bytearray_length),
    );
    slots.set(Slot::Iter, unary(super::stdlib::core::slot_sequence_iter));
    slots.set(
        Slot::Contains,
        intrinsic(super::stdlib::core::slot_builtin_contains),
    );
    slots.set(
        Slot::GetItem,
        intrinsic(super::stdlib::core::slot_bytearray_get_item),
    );
    slots.set(
        Slot::Add,
        intrinsic(super::stdlib::core::slot_bytearray_add),
    );
    slots.set(
        Slot::Multiply,
        intrinsic(super::stdlib::core::slot_bytearray_multiply),
    );
    slots.set(
        Slot::ReflectedMultiply,
        intrinsic(super::stdlib::core::slot_bytearray_multiply),
    );
    slots.set(
        Slot::SetItem,
        SlotValue::NativeTernary(super::stdlib::core::slot_bytearray_set_item),
    );
    slots.set(
        Slot::DeleteItem,
        intrinsic(super::stdlib::core::slot_bytearray_delete_item),
    );

    let slots = &mut types[BuiltinType::List as usize].slots;
    install_compare(slots, super::stdlib::core::slot_container_compare);
    slots.set(
        Slot::Length,
        unary(super::stdlib::core::slot_builtin_length),
    );
    slots.set(
        Slot::GetItem,
        intrinsic(super::stdlib::core::slot_builtin_get_item),
    );
    slots.set(
        Slot::Contains,
        intrinsic(super::stdlib::core::slot_builtin_contains),
    );
    slots.set(Slot::Iter, unary(super::stdlib::core::slot_sequence_iter));
    slots.set(
        Slot::Reversed,
        unary(super::stdlib::core::slot_sequence_reversed),
    );
    slots.set(Slot::Add, intrinsic(super::stdlib::core::slot_list_add));
    slots.set(
        Slot::Multiply,
        intrinsic(super::stdlib::core::slot_list_multiply),
    );
    slots.set(
        Slot::ReflectedMultiply,
        intrinsic(super::stdlib::core::slot_list_multiply),
    );
    slots.set(
        Slot::SetItem,
        SlotValue::NativeTernary(super::stdlib::core::slot_list_set_item),
    );
    slots.set(
        Slot::DeleteItem,
        intrinsic(super::stdlib::core::slot_list_delete_item),
    );

    let slots = &mut types[BuiltinType::Dict as usize].slots;
    install_compare(slots, super::stdlib::core::slot_container_compare);
    slots.set(
        Slot::Length,
        unary(super::stdlib::core::slot_builtin_length),
    );
    slots.set(Slot::Iter, unary(super::stdlib::core::slot_sequence_iter));
    slots.set(
        Slot::Contains,
        intrinsic(super::stdlib::core::slot_builtin_contains),
    );
    slots.set(
        Slot::Reversed,
        unary(super::stdlib::core::slot_dict_reversed),
    );
    slots.set(
        Slot::DeleteItem,
        intrinsic(super::stdlib::core::slot_dict_delete_item),
    );
    slots.set(
        Slot::BitwiseOr,
        intrinsic(super::stdlib::core::slot_dict_union),
    );
    slots.set(
        Slot::ReflectedBitwiseOr,
        intrinsic(super::stdlib::core::slot_dict_reflected_union),
    );

    // The VM's builtin subscript, `len`, `in` and iteration read stored dicts directly, so a
    // namespace view supplies those through slots and inherits the rest from `dict`.
    types[BuiltinType::NamespaceDict as usize].slots =
        types[BuiltinType::Dict as usize].slots.clone();
    let slots = &mut types[BuiltinType::NamespaceDict as usize].slots;
    slots.set(
        Slot::GetItem,
        intrinsic(super::stdlib::core::slot_namespace_dict_get_item),
    );
    slots.set(
        Slot::SetItem,
        SlotValue::NativeTernary(super::stdlib::core::slot_namespace_dict_set_item),
    );
    slots.set(
        Slot::Length,
        unary(super::stdlib::core::slot_namespace_dict_length),
    );
    slots.set(
        Slot::Contains,
        intrinsic(super::stdlib::core::slot_namespace_dict_contains),
    );
    slots.set(
        Slot::Iter,
        unary(super::stdlib::core::slot_namespace_dict_iter),
    );
    // No `slots.repr`: `Vm::repr_nested` renders `Object::NamespaceDict` directly, sharing the
    // same cycle-tracking set as `dict`, `list` and `set`. A slot implemented through the erased
    // `PyRuntime::repr` would start a fresh cycle-tracking set per nested call and recurse
    // forever on `g = globals()`.

    for view in [
        BuiltinType::DictKeys,
        BuiltinType::DictValues,
        BuiltinType::DictItems,
    ] {
        use super::stdlib::mapping_views as views;
        let slots = &mut types[view as usize].slots;
        slots.set(Slot::Length, unary(views::slot_view_length));
        slots.set(Slot::Iter, unary(views::slot_view_iter));
        slots.set(Slot::Reversed, unary(views::slot_view_reversed));
        slots.set(Slot::Contains, intrinsic(views::slot_view_contains));
        if view == BuiltinType::DictValues {
            continue;
        }
        // Keys and items views are set-like; a values view compares by identity.
        slots.set(Slot::Equal, intrinsic(views::slot_view_equal));
        slots.set(Slot::LessThan, intrinsic(views::slot_view_less));
        slots.set(Slot::LessEqual, intrinsic(views::slot_view_less_equal));
        slots.set(Slot::GreaterThan, intrinsic(views::slot_view_greater));
        slots.set(
            Slot::GreaterEqual,
            intrinsic(views::slot_view_greater_equal),
        );
        slots.set(Slot::BitwiseAnd, intrinsic(views::slot_view_and));
        slots.set(
            Slot::ReflectedBitwiseAnd,
            intrinsic(views::slot_view_reflected_and),
        );
        slots.set(Slot::BitwiseOr, intrinsic(views::slot_view_or));
        slots.set(
            Slot::ReflectedBitwiseOr,
            intrinsic(views::slot_view_reflected_or),
        );
        slots.set(Slot::BitwiseXor, intrinsic(views::slot_view_xor));
        slots.set(
            Slot::ReflectedBitwiseXor,
            intrinsic(views::slot_view_reflected_xor),
        );
        slots.set(Slot::Subtract, intrinsic(views::slot_view_subtract));
        slots.set(
            Slot::ReflectedSubtract,
            intrinsic(views::slot_view_reflected_subtract),
        );
    }
    let slots = &mut types[BuiltinType::MappingProxy as usize].slots;
    install_compare(slots, super::stdlib::core::slot_container_compare);
    slots.set(
        Slot::GetItem,
        intrinsic(super::stdlib::mapping_views::slot_proxy_get_item),
    );
    slots.set(
        Slot::Length,
        unary(super::stdlib::mapping_views::slot_proxy_length),
    );
    slots.set(
        Slot::Contains,
        intrinsic(super::stdlib::mapping_views::slot_proxy_contains),
    );
    slots.set(
        Slot::Iter,
        unary(super::stdlib::mapping_views::slot_proxy_iter),
    );

    let slots = &mut types[BuiltinType::Set as usize].slots;
    install_compare(slots, super::stdlib::core::slot_set_compare);
    slots.set(
        Slot::Length,
        unary(super::stdlib::core::slot_builtin_length),
    );
    slots.set(Slot::Iter, unary(super::stdlib::core::slot_sequence_iter));
    slots.set(
        Slot::Contains,
        intrinsic(super::stdlib::core::slot_builtin_contains),
    );
    slots.set(
        Slot::Subtract,
        intrinsic(super::stdlib::core::slot_set_subtract),
    );
    slots.set(
        Slot::BitwiseAnd,
        intrinsic(super::stdlib::core::slot_set_intersection),
    );
    slots.set(
        Slot::ReflectedBitwiseAnd,
        intrinsic(super::stdlib::core::slot_set_intersection),
    );
    slots.set(
        Slot::BitwiseXor,
        intrinsic(super::stdlib::core::slot_set_symmetric_difference),
    );
    slots.set(
        Slot::ReflectedBitwiseXor,
        intrinsic(super::stdlib::core::slot_set_symmetric_difference),
    );
    slots.set(
        Slot::BitwiseOr,
        intrinsic(super::stdlib::core::slot_set_union),
    );
    slots.set(
        Slot::ReflectedBitwiseOr,
        intrinsic(super::stdlib::core::slot_set_union),
    );

    types[BuiltinType::FrozenSet as usize].slots = types[BuiltinType::Set as usize].slots.clone();
    types[BuiltinType::FrozenSet as usize]
        .slots
        .set(Slot::Hash, SlotValue::VmHash);

    let slots = &mut types[BuiltinType::Tuple as usize].slots;
    install_compare(slots, super::stdlib::core::slot_container_compare);
    slots.set(
        Slot::Length,
        unary(super::stdlib::core::slot_builtin_length),
    );
    slots.set(
        Slot::GetItem,
        intrinsic(super::stdlib::core::slot_builtin_get_item),
    );
    slots.set(
        Slot::Contains,
        intrinsic(super::stdlib::core::slot_builtin_contains),
    );
    slots.set(Slot::Iter, unary(super::stdlib::core::slot_sequence_iter));
    slots.set(
        Slot::Reversed,
        unary(super::stdlib::core::slot_sequence_reversed),
    );
    slots.set(Slot::Add, intrinsic(super::stdlib::core::slot_tuple_add));
    slots.set(
        Slot::Multiply,
        intrinsic(super::stdlib::core::slot_tuple_multiply),
    );
    slots.set(
        Slot::ReflectedMultiply,
        intrinsic(super::stdlib::core::slot_tuple_multiply),
    );

    types[BuiltinType::Range as usize]
        .slots
        .set(Slot::Iter, unary(super::stdlib::core::slot_sequence_iter));
    types[BuiltinType::Range as usize].slots.set(
        Slot::Length,
        unary(super::stdlib::core::slot_builtin_length),
    );
    types[BuiltinType::Range as usize].slots.set(
        Slot::GetItem,
        intrinsic(super::stdlib::core::slot_builtin_get_item),
    );
    types[BuiltinType::Range as usize].slots.set(
        Slot::Contains,
        intrinsic(super::stdlib::core::slot_builtin_contains),
    );
    types[BuiltinType::Range as usize].slots.set(
        Slot::Reversed,
        unary(super::stdlib::core::slot_sequence_reversed),
    );

    let slots = &mut types[BuiltinType::Array as usize].slots;
    {
        use super::stdlib::numpy as np;
        slots.set(Slot::Repr, unary(np::slot_repr));
        slots.set(Slot::String, unary(np::slot_str));
        slots.set(Slot::Bool, unary(np::slot_bool));
        slots.set(Slot::Iter, unary(np::slot_iter));
        slots.set(Slot::Length, unary(np::slot_length));
        slots.set(Slot::GetItem, intrinsic(np::slot_get_item));
        slots.set(Slot::SetItem, SlotValue::NativeTernary(np::slot_set_item));
        slots.set(Slot::Positive, unary(np::slot_positive));
        slots.set(Slot::Negative, unary(np::slot_negative));
        slots.set(Slot::Invert, unary(np::slot_invert));
        slots.set(Slot::Absolute, unary(np::slot_absolute));
        slots.set(Slot::Add, intrinsic(np::slot_add));
        slots.set(Slot::ReflectedAdd, intrinsic(np::slot_reflected_add));
        slots.set(Slot::Subtract, intrinsic(np::slot_subtract));
        slots.set(
            Slot::ReflectedSubtract,
            intrinsic(np::slot_reflected_subtract),
        );
        slots.set(Slot::Multiply, intrinsic(np::slot_multiply));
        slots.set(
            Slot::ReflectedMultiply,
            intrinsic(np::slot_reflected_multiply),
        );
        slots.set(Slot::MatrixMultiply, intrinsic(np::slot_matrix_multiply));
        slots.set(
            Slot::ReflectedMatrixMultiply,
            intrinsic(np::slot_reflected_matrix_multiply),
        );
        slots.set(Slot::Power, intrinsic(np::slot_power));
        slots.set(Slot::ReflectedPower, intrinsic(np::slot_reflected_power));
        slots.set(Slot::Divide, intrinsic(np::slot_divide));
        slots.set(Slot::ReflectedDivide, intrinsic(np::slot_reflected_divide));
        slots.set(Slot::FloorDivide, intrinsic(np::slot_floor_divide));
        slots.set(
            Slot::ReflectedFloorDivide,
            intrinsic(np::slot_reflected_floor_divide),
        );
        slots.set(Slot::Remainder, intrinsic(np::slot_remainder));
        slots.set(
            Slot::ReflectedRemainder,
            intrinsic(np::slot_reflected_remainder),
        );
        slots.set(Slot::DivMod, intrinsic(np::slot_divmod));
        slots.set(Slot::ReflectedDivMod, intrinsic(np::slot_reflected_divmod));
        slots.set(Slot::LeftShift, intrinsic(np::slot_left_shift));
        slots.set(
            Slot::ReflectedLeftShift,
            intrinsic(np::slot_reflected_left_shift),
        );
        slots.set(Slot::RightShift, intrinsic(np::slot_right_shift));
        slots.set(
            Slot::ReflectedRightShift,
            intrinsic(np::slot_reflected_right_shift),
        );
        slots.set(Slot::BitwiseAnd, intrinsic(np::slot_bitwise_and));
        slots.set(
            Slot::ReflectedBitwiseAnd,
            intrinsic(np::slot_reflected_bitwise_and),
        );
        slots.set(Slot::BitwiseXor, intrinsic(np::slot_bitwise_xor));
        slots.set(
            Slot::ReflectedBitwiseXor,
            intrinsic(np::slot_reflected_bitwise_xor),
        );
        slots.set(Slot::BitwiseOr, intrinsic(np::slot_bitwise_or));
        slots.set(
            Slot::ReflectedBitwiseOr,
            intrinsic(np::slot_reflected_bitwise_or),
        );
        slots.set(Slot::Equal, intrinsic(np::slot_equal));
        slots.set(Slot::NotEqual, intrinsic(np::slot_not_equal));
        slots.set(Slot::LessThan, intrinsic(np::slot_less_than));
        slots.set(Slot::LessEqual, intrinsic(np::slot_less_equal));
        slots.set(Slot::GreaterThan, intrinsic(np::slot_greater_than));
        slots.set(Slot::GreaterEqual, intrinsic(np::slot_greater_equal));
        for (slot, call) in [
            (Slot::InplaceAdd, np::slot_iadd as BinarySlotFn),
            (Slot::InplaceSubtract, np::slot_isub),
            (Slot::InplaceMultiply, np::slot_imul),
            (Slot::InplacePower, np::slot_ipow),
            (Slot::InplaceDivide, np::slot_itruediv),
            (Slot::InplaceFloorDivide, np::slot_ifloordiv),
            (Slot::InplaceRemainder, np::slot_imod),
            (Slot::InplaceLeftShift, np::slot_ilshift),
            (Slot::InplaceRightShift, np::slot_irshift),
            (Slot::InplaceBitwiseAnd, np::slot_iand),
            (Slot::InplaceBitwiseXor, np::slot_ixor),
            (Slot::InplaceBitwiseOr, np::slot_ior),
        ] {
            slots.set(slot, intrinsic(call));
        }
    }

    let slots = &mut types[BuiltinType::Complex as usize].slots;
    slots.set(Slot::Equal, intrinsic(super::number::slot_equal));
    slots.set(Slot::NotEqual, intrinsic(super::number::slot_not_equal));
    slots.set(Slot::Hash, unary(super::number::slot_hash));
    slots.set(Slot::Bool, unary(super::number::slot_bool));
    slots.set(Slot::Positive, unary(super::complex::slot_positive));
    slots.set(Slot::Negative, unary(super::complex::slot_negative));
    slots.set(Slot::Absolute, unary(super::complex::slot_absolute));
    slots.set(Slot::Hash, unary(super::complex::slot_hash));
    slots.set(Slot::Add, intrinsic(super::complex::slot_add));
    slots.set(Slot::ReflectedAdd, intrinsic(super::complex::slot_add));
    slots.set(Slot::Subtract, intrinsic(super::complex::slot_subtract));
    slots.set(
        Slot::ReflectedSubtract,
        intrinsic(super::complex::slot_reflected_subtract),
    );
    slots.set(Slot::Multiply, intrinsic(super::complex::slot_multiply));
    slots.set(
        Slot::ReflectedMultiply,
        intrinsic(super::complex::slot_multiply),
    );
    slots.set(Slot::Divide, intrinsic(super::complex::slot_divide));
    slots.set(
        Slot::ReflectedDivide,
        intrinsic(super::complex::slot_reflected_divide),
    );
    slots.set(Slot::Power, intrinsic(super::complex::slot_power));
    slots.set(
        Slot::ReflectedPower,
        intrinsic(super::complex::slot_reflected_power),
    );
    slots.set(
        Slot::FloorDivide,
        intrinsic(super::complex::slot_floor_divide),
    );
    slots.set(
        Slot::ReflectedFloorDivide,
        intrinsic(super::complex::slot_reflected_floor_divide),
    );
    slots.set(Slot::Remainder, intrinsic(super::complex::slot_remainder));
    slots.set(
        Slot::ReflectedRemainder,
        intrinsic(super::complex::slot_reflected_remainder),
    );

    let stream = &mut types[BuiltinType::Stream as usize].slots;
    stream.set(Slot::Iter, unary(super::stdlib::sys::slot_iter));
    stream.set(Slot::Next, unary(super::stdlib::sys::slot_next));

    types[BuiltinType::NotImplemented as usize]
        .slots
        .set(Slot::Bool, unary(not_implemented_bool));
    let enum_slots = &mut types[BuiltinType::Enum as usize].slots;
    enum_slots.set(Slot::Repr, unary(super::stdlib::r#enum::slot_repr));
    enum_slots.set(Slot::String, unary(super::stdlib::r#enum::slot_str));
    enum_slots.set(Slot::Hash, unary(super::stdlib::r#enum::slot_hash));
}

/// Fill resolved builtin slots from the first defining ancestor. Local slots were captured
/// before this pass so inherited native wrappers appear only in their defining namespace.
fn resolve_builtin_slots(types: &mut [PyType]) {
    for index in 0..types.len() {
        let mro = types[index].mro.clone();
        for slot in Slot::ALL {
            if types[index].slots.get(slot).is_some() {
                continue;
            }
            if let Some(value) = mro
                .iter()
                .find_map(|ancestor| types[ancestor.0 as usize].local_slots.get(slot).cloned())
            {
                types[index].slots.set(slot, value);
            }
        }
    }
}

fn install_slot_wrappers(types: &mut [PyType]) {
    for (index, ty) in types.iter_mut().enumerate() {
        let id = TypeId(u32::try_from(index).expect("too many builtin types"));
        install_slot_wrappers_for_type(ty, id);
    }
}

fn install_slot_wrappers_for_type(ty: &mut PyType, owner: TypeId) {
    for (slot, name, _) in SLOT_DEFS {
        if matches!(
            ty.local_slots.get(slot),
            Some(
                SlotValue::NativeBinary(_)
                    | SlotValue::NativeTernary(_)
                    | SlotValue::NativeUnary(_)
                    | SlotValue::VmRepr
                    | SlotValue::VmHash
                    | SlotValue::NativeCompare(_)
            )
        ) {
            ty.attributes.entry(name.into()).or_insert_with(|| {
                Ref::from_immediate(Value::Native(super::vm::NativeValue::SlotWrapper {
                    owner,
                    slot,
                }))
            });
        }
    }
}

/// CPython 3.14 rejects `NotImplemented` in a boolean context. Truth-testing it usually means an
/// operator method's result was used without checking whether the method declined.
fn not_implemented_bool(_: &mut dyn PyRuntime, _: Value) -> PyResult<Option<Value>> {
    Err(PyError::type_error(
        "NotImplemented should not be used in a boolean context",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_builds_bool_int_hierarchy_and_slots() {
        let registry = TypeRegistry::default();
        assert!(registry
            .is_subclass(BuiltinType::Bool.id(), BuiltinType::Int.id())
            .unwrap());
        assert!(registry
            .is_subclass(BuiltinType::Int.id(), BuiltinType::Object.id())
            .unwrap());
        assert!(matches!(
            registry
                .slot(BuiltinType::Int.id(), Slot::Multiply)
                .unwrap(),
            Some(SlotValue::NativeBinary(_))
        ));
    }

    #[test]
    fn slot_definitions_follow_encoded_slot_indices() {
        for (index, (slot, _, _)) in SLOT_DEFS.iter().enumerate() {
            assert_eq!(*slot as usize, index);
        }
    }

    #[test]
    fn registered_bases_use_c3_order_and_reject_conflicts() {
        let registry = TypeRegistry::default();
        assert_eq!(
            registry
                .linearize_bases(&[BuiltinType::Bool.id(), BuiltinType::Int.id()])
                .unwrap(),
            vec![
                BuiltinType::Bool.id(),
                BuiltinType::Int.id(),
                BuiltinType::Object.id()
            ]
        );
        assert!(registry
            .linearize_bases(&[BuiltinType::Int.id(), BuiltinType::Bool.id()])
            .is_err());
        assert!(registry
            .linearize_bases(&[BuiltinType::Int.id(), BuiltinType::Int.id()])
            .is_err());
    }

    #[test]
    fn exception_types_have_distinct_registry_ids_and_multiple_ancestors() {
        let registry = TypeRegistry::default();
        let axis = registry.exception_type_id("AxisError").unwrap();
        let value = registry.exception_type_id("ValueError").unwrap();
        let index = registry.exception_type_id("IndexError").unwrap();
        assert_ne!(value, index);
        assert!(registry.is_subclass(axis, value).unwrap());
        assert!(registry.is_subclass(axis, index).unwrap());
        assert_eq!(registry.get(axis).unwrap().bases, vec![value, index]);
    }

    #[test]
    fn modeled_memory_counts_builtin_and_registered_types() {
        let mut registry = TypeRegistry::default();
        let builtin_bytes = registry
            .types
            .iter()
            .take(BuiltinType::ALL.len())
            .map(modeled_type_bytes)
            .fold(0u64, u64::saturating_add);
        let value_kind_bytes = registry
            .types
            .iter()
            .skip(BuiltinType::ALL.len())
            .map(modeled_type_bytes)
            .fold(0u64, u64::saturating_add);
        assert_eq!(registry.modeled_bytes(), builtin_bytes + value_kind_bytes);

        let before = registry.modeled_bytes();
        let id = registry
            .register(
                "Example".into(),
                vec![BuiltinType::Object.id()],
                vec![BuiltinType::Object.id()],
                &HashMap::from([("value".into(), Ref::from_immediate(Value::Int(1)))]),
            )
            .unwrap();
        let registered_bytes = modeled_type_bytes(registry.get(id).unwrap());
        assert_eq!(registry.get(id).unwrap().attributes.get("value"), None);
        assert_eq!(
            registry.modeled_bytes(),
            before.saturating_add(registered_bytes)
        );

        registry
            .finish(id, Ref::from_immediate(Value::None))
            .unwrap();
        assert_eq!(
            registry.modeled_bytes(),
            before.saturating_add(registered_bytes)
        );
    }
}
