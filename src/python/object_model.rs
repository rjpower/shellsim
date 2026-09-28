//! Python semantic types and bootstrapped builtin type metadata.
//!
//! `TypeId` is independent of storage shape: immediate values, arena objects, and native markers
//! all identify their Python type through the same registry. The registry stores only modeled
//! metadata and Python values, so looking up a type cannot acquire host capabilities.

use std::collections::HashMap;

use super::native::{
    BinarySlotFn, MethodDef, PyError, PyResult, PyRuntime, TernarySlotFn, UnarySlotFn,
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
}

impl BuiltinType {
    pub(super) const ALL: [Self; 41] = [
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
            Self::Array => "numpy.ndarray",
            Self::Complex => "complex",
            Self::Slice => "slice",
            Self::GenericAlias => "GenericAlias",
            Self::Enum => "enum.Enum",
            Self::TestCase => "unittest.TestCase",
        }
    }
}

/// Cached protocol methods resolved from a type dictionary.
///
/// A user slot holds its ordinary Python descriptor. Builtin operator slots hold a direct native
/// function with the erased runtime ABI. Empty slots mean the type does not implement the
/// protocol.
#[derive(Clone, Debug, Default)]
pub struct TypeSlots {
    pub call: Option<SlotValue>,
    pub new: Option<SlotValue>,
    pub init: Option<SlotValue>,
    pub getattribute: Option<SlotValue>,
    pub setattr: Option<SlotValue>,
    pub repr: Option<SlotValue>,
    pub str_: Option<SlotValue>,
    pub bool_: Option<SlotValue>,
    pub hash: Option<SlotValue>,
    pub iter: Option<SlotValue>,
    pub next: Option<SlotValue>,
    pub length: Option<SlotValue>,
    pub get_item: Option<SlotValue>,
    pub set_item: Option<SlotValue>,
    pub positive: Option<SlotValue>,
    pub negative: Option<SlotValue>,
    pub invert: Option<SlotValue>,
    pub absolute: Option<SlotValue>,
    pub add: Option<SlotValue>,
    pub reflected_add: Option<SlotValue>,
    pub subtract: Option<SlotValue>,
    pub reflected_subtract: Option<SlotValue>,
    pub multiply: Option<SlotValue>,
    pub reflected_multiply: Option<SlotValue>,
    pub matrix_multiply: Option<SlotValue>,
    pub reflected_matrix_multiply: Option<SlotValue>,
    pub power: Option<SlotValue>,
    pub reflected_power: Option<SlotValue>,
    pub divide: Option<SlotValue>,
    pub reflected_divide: Option<SlotValue>,
    pub floor_divide: Option<SlotValue>,
    pub reflected_floor_divide: Option<SlotValue>,
    pub remainder: Option<SlotValue>,
    pub reflected_remainder: Option<SlotValue>,
    pub divmod: Option<SlotValue>,
    pub reflected_divmod: Option<SlotValue>,
    pub left_shift: Option<SlotValue>,
    pub reflected_left_shift: Option<SlotValue>,
    pub right_shift: Option<SlotValue>,
    pub reflected_right_shift: Option<SlotValue>,
    pub bitwise_and: Option<SlotValue>,
    pub reflected_bitwise_and: Option<SlotValue>,
    pub bitwise_xor: Option<SlotValue>,
    pub reflected_bitwise_xor: Option<SlotValue>,
    pub bitwise_or: Option<SlotValue>,
    pub reflected_bitwise_or: Option<SlotValue>,
    pub equal: Option<SlotValue>,
    pub not_equal: Option<SlotValue>,
    pub less_than: Option<SlotValue>,
    pub less_equal: Option<SlotValue>,
    pub greater_than: Option<SlotValue>,
    pub greater_equal: Option<SlotValue>,
    pub contains: Option<SlotValue>,
    pub delete_item: Option<SlotValue>,
    extra: [Option<SlotValue>; 18],
}

/// A cached Python descriptor or a native implementation attached directly to a builtin type.
#[derive(Clone, Debug)]
pub enum SlotValue {
    Descriptor(Value),
    NativeMethod(&'static MethodDef),
    /// The VM's representation renderer, which shares cycle tracking across nested values.
    VmRepr,
    /// Enum member text needs its defining class, which is stored in the VM's heap.
    VmEnumString,
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
pub(super) const SLOT_DEFS: [(Slot, &str, u8); 72] = [
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
    fn from_attributes(attributes: &HashMap<String, Value>) -> Self {
        let mut slots = Self::default();
        for (slot, name, _) in SLOT_DEFS {
            if let Some(value) = attributes.get(name) {
                slots.set(slot, SlotValue::Descriptor(*value));
            }
        }
        slots
    }

    fn populated_count(&self) -> usize {
        [
            &self.call,
            &self.new,
            &self.init,
            &self.getattribute,
            &self.setattr,
            &self.repr,
            &self.str_,
            &self.bool_,
            &self.hash,
            &self.iter,
            &self.next,
            &self.length,
            &self.get_item,
            &self.set_item,
            &self.positive,
            &self.negative,
            &self.invert,
            &self.absolute,
            &self.add,
            &self.reflected_add,
            &self.subtract,
            &self.reflected_subtract,
            &self.multiply,
            &self.reflected_multiply,
            &self.matrix_multiply,
            &self.reflected_matrix_multiply,
            &self.power,
            &self.reflected_power,
            &self.divide,
            &self.reflected_divide,
            &self.floor_divide,
            &self.reflected_floor_divide,
            &self.remainder,
            &self.reflected_remainder,
            &self.divmod,
            &self.reflected_divmod,
            &self.left_shift,
            &self.reflected_left_shift,
            &self.right_shift,
            &self.reflected_right_shift,
            &self.bitwise_and,
            &self.reflected_bitwise_and,
            &self.bitwise_xor,
            &self.reflected_bitwise_xor,
            &self.bitwise_or,
            &self.reflected_bitwise_or,
            &self.equal,
            &self.not_equal,
            &self.less_than,
            &self.less_equal,
            &self.greater_than,
            &self.greater_equal,
            &self.contains,
            &self.delete_item,
        ]
        .into_iter()
        .filter(|slot| slot.is_some())
        .count()
            + self.extra.iter().filter(|slot| slot.is_some()).count()
    }

    pub fn get(&self, slot: Slot) -> Option<&SlotValue> {
        match slot {
            Slot::Call => self.call.as_ref(),
            Slot::New => self.new.as_ref(),
            Slot::Init => self.init.as_ref(),
            Slot::GetAttribute => self.getattribute.as_ref(),
            Slot::SetAttribute => self.setattr.as_ref(),
            Slot::Repr => self.repr.as_ref(),
            Slot::String => self.str_.as_ref(),
            Slot::Bool => self.bool_.as_ref(),
            Slot::Hash => self.hash.as_ref(),
            Slot::Iter => self.iter.as_ref(),
            Slot::Next => self.next.as_ref(),
            Slot::Length => self.length.as_ref(),
            Slot::GetItem => self.get_item.as_ref(),
            Slot::SetItem => self.set_item.as_ref(),
            Slot::Positive => self.positive.as_ref(),
            Slot::Negative => self.negative.as_ref(),
            Slot::Invert => self.invert.as_ref(),
            Slot::Absolute => self.absolute.as_ref(),
            Slot::Add => self.add.as_ref(),
            Slot::ReflectedAdd => self.reflected_add.as_ref(),
            Slot::Subtract => self.subtract.as_ref(),
            Slot::ReflectedSubtract => self.reflected_subtract.as_ref(),
            Slot::Multiply => self.multiply.as_ref(),
            Slot::ReflectedMultiply => self.reflected_multiply.as_ref(),
            Slot::MatrixMultiply => self.matrix_multiply.as_ref(),
            Slot::ReflectedMatrixMultiply => self.reflected_matrix_multiply.as_ref(),
            Slot::Power => self.power.as_ref(),
            Slot::ReflectedPower => self.reflected_power.as_ref(),
            Slot::Divide => self.divide.as_ref(),
            Slot::ReflectedDivide => self.reflected_divide.as_ref(),
            Slot::FloorDivide => self.floor_divide.as_ref(),
            Slot::ReflectedFloorDivide => self.reflected_floor_divide.as_ref(),
            Slot::Remainder => self.remainder.as_ref(),
            Slot::ReflectedRemainder => self.reflected_remainder.as_ref(),
            Slot::DivMod => self.divmod.as_ref(),
            Slot::ReflectedDivMod => self.reflected_divmod.as_ref(),
            Slot::LeftShift => self.left_shift.as_ref(),
            Slot::ReflectedLeftShift => self.reflected_left_shift.as_ref(),
            Slot::RightShift => self.right_shift.as_ref(),
            Slot::ReflectedRightShift => self.reflected_right_shift.as_ref(),
            Slot::BitwiseAnd => self.bitwise_and.as_ref(),
            Slot::ReflectedBitwiseAnd => self.reflected_bitwise_and.as_ref(),
            Slot::BitwiseXor => self.bitwise_xor.as_ref(),
            Slot::ReflectedBitwiseXor => self.reflected_bitwise_xor.as_ref(),
            Slot::BitwiseOr => self.bitwise_or.as_ref(),
            Slot::ReflectedBitwiseOr => self.reflected_bitwise_or.as_ref(),
            Slot::Equal => self.equal.as_ref(),
            Slot::NotEqual => self.not_equal.as_ref(),
            Slot::LessThan => self.less_than.as_ref(),
            Slot::LessEqual => self.less_equal.as_ref(),
            Slot::GreaterThan => self.greater_than.as_ref(),
            Slot::GreaterEqual => self.greater_equal.as_ref(),
            Slot::Contains => self.contains.as_ref(),
            Slot::DeleteItem => self.delete_item.as_ref(),
            _ => self.extra[slot as usize - 54].as_ref(),
        }
    }

    fn set(&mut self, slot: Slot, value: SlotValue) {
        *match slot {
            Slot::Call => &mut self.call,
            Slot::New => &mut self.new,
            Slot::Init => &mut self.init,
            Slot::GetAttribute => &mut self.getattribute,
            Slot::SetAttribute => &mut self.setattr,
            Slot::Repr => &mut self.repr,
            Slot::String => &mut self.str_,
            Slot::Bool => &mut self.bool_,
            Slot::Hash => &mut self.hash,
            Slot::Iter => &mut self.iter,
            Slot::Next => &mut self.next,
            Slot::Length => &mut self.length,
            Slot::GetItem => &mut self.get_item,
            Slot::SetItem => &mut self.set_item,
            Slot::Positive => &mut self.positive,
            Slot::Negative => &mut self.negative,
            Slot::Invert => &mut self.invert,
            Slot::Absolute => &mut self.absolute,
            Slot::Add => &mut self.add,
            Slot::ReflectedAdd => &mut self.reflected_add,
            Slot::Subtract => &mut self.subtract,
            Slot::ReflectedSubtract => &mut self.reflected_subtract,
            Slot::Multiply => &mut self.multiply,
            Slot::ReflectedMultiply => &mut self.reflected_multiply,
            Slot::MatrixMultiply => &mut self.matrix_multiply,
            Slot::ReflectedMatrixMultiply => &mut self.reflected_matrix_multiply,
            Slot::Power => &mut self.power,
            Slot::ReflectedPower => &mut self.reflected_power,
            Slot::Divide => &mut self.divide,
            Slot::ReflectedDivide => &mut self.reflected_divide,
            Slot::FloorDivide => &mut self.floor_divide,
            Slot::ReflectedFloorDivide => &mut self.reflected_floor_divide,
            Slot::Remainder => &mut self.remainder,
            Slot::ReflectedRemainder => &mut self.reflected_remainder,
            Slot::DivMod => &mut self.divmod,
            Slot::ReflectedDivMod => &mut self.reflected_divmod,
            Slot::LeftShift => &mut self.left_shift,
            Slot::ReflectedLeftShift => &mut self.reflected_left_shift,
            Slot::RightShift => &mut self.right_shift,
            Slot::ReflectedRightShift => &mut self.reflected_right_shift,
            Slot::BitwiseAnd => &mut self.bitwise_and,
            Slot::ReflectedBitwiseAnd => &mut self.reflected_bitwise_and,
            Slot::BitwiseXor => &mut self.bitwise_xor,
            Slot::ReflectedBitwiseXor => &mut self.reflected_bitwise_xor,
            Slot::BitwiseOr => &mut self.bitwise_or,
            Slot::ReflectedBitwiseOr => &mut self.reflected_bitwise_or,
            Slot::Equal => &mut self.equal,
            Slot::NotEqual => &mut self.not_equal,
            Slot::LessThan => &mut self.less_than,
            Slot::LessEqual => &mut self.less_equal,
            Slot::GreaterThan => &mut self.greater_than,
            Slot::GreaterEqual => &mut self.greater_equal,
            Slot::Contains => &mut self.contains,
            Slot::DeleteItem => &mut self.delete_item,
            _ => &mut self.extra[slot as usize - 54],
        } = Some(value);
    }
}

/// Metadata shared by builtin, native, and user-defined Python types.
#[derive(Clone, Debug)]
pub struct PyType {
    pub name: String,
    pub bases: Vec<TypeId>,
    pub mro: Vec<TypeId>,
    pub attributes: HashMap<String, Value>,
    /// Slots defined by this type, before MRO resolution.
    pub local_slots: TypeSlots,
    pub slots: TypeSlots,
    value: Option<Value>,
}

/// Per-runtime registry containing all semantic Python types.
#[derive(Clone, Debug)]
pub struct TypeRegistry {
    types: Vec<PyType>,
    value_kinds: Vec<&'static super::native::ValueKindDef>,
    exception_types: HashMap<&'static str, TypeId>,
    modeled_bytes: u64,
}

fn modeled_type_bytes(ty: &PyType) -> u64 {
    let bytes = 64usize
        .saturating_add(ty.name.len())
        .saturating_add(ty.bases.len().saturating_mul(4))
        .saturating_add(ty.mro.len().saturating_mul(4))
        .saturating_add(ty.attributes.len().saturating_mul(48))
        .saturating_add(ty.local_slots.populated_count().saturating_mul(24))
        .saturating_add(ty.slots.populated_count().saturating_mul(24));
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

impl Default for TypeRegistry {
    fn default() -> Self {
        let mut types = Vec::with_capacity(BuiltinType::ALL.len());
        for builtin in BuiltinType::ALL {
            let (bases, mro) = builtin_metadata(builtin);
            types.push(PyType {
                name: builtin.name().into(),
                bases,
                mro,
                attributes: HashMap::new(),
                local_slots: TypeSlots::default(),
                slots: TypeSlots::default(),
                value: Some(Value::Native(super::vm::NativeValue::BuiltinType(builtin))),
            });
        }
        install_native_attributes(
            &mut types[BuiltinType::Object as usize],
            &super::stdlib::core::OBJECT_TYPE,
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
            types[BuiltinType::Dict as usize].attributes.clone();
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
        let builtin_bytes = types
            .iter()
            .map(modeled_type_bytes)
            .fold(0u64, u64::saturating_add);
        let mut registry = Self {
            types,
            value_kinds: Vec::new(),
            exception_types: HashMap::new(),
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
        self.types[base.raw() as usize].value = Some(Value::Native(NativeValue::ExceptionType(
            ExceptionType("BaseException"),
        )));
        self.exception_types.insert("BaseException", base);
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
            let value = Value::Native(NativeValue::ExceptionType(ExceptionType(definition.name)));
            let ty = PyType {
                name: definition.name.into(),
                bases,
                mro: mro.clone(),
                attributes: HashMap::new(),
                local_slots: TypeSlots::default(),
                slots: self.inherit_slots(TypeSlots::default(), &mro),
                value: Some(value),
            };
            let id = TypeId(u32::try_from(self.types.len()).expect("too many registered types"));
            self.modeled_bytes = self.modeled_bytes.saturating_add(modeled_type_bytes(&ty));
            self.types.push(ty);
            self.exception_types.insert(definition.name, id);
        }
    }

    pub(super) fn exception_type_id(&self, name: &str) -> Option<TypeId> {
        self.exception_types.get(name).copied()
    }

    /// Conservative modeled size of registry metadata retained between executions.
    pub fn modeled_bytes(&self) -> u64 {
        self.modeled_bytes
    }

    /// Heap values retained by semantic type metadata.
    ///
    /// Completed user class objects own their Python-visible namespace. The registry roots those
    /// class objects, builtin attributes, and cached protocol descriptors at every safe point.
    pub fn heap_roots(&self) -> Vec<Value> {
        let mut roots = Vec::new();
        for ty in &self.types {
            roots.extend(ty.attributes.values().copied());
            roots.extend(ty.value);
            for slot in Slot::ALL {
                if let Some(SlotValue::Descriptor(value)) = ty.slots.get(slot) {
                    roots.push(*value);
                }
            }
        }
        roots
    }

    pub fn get(&self, id: TypeId) -> Result<&PyType, String> {
        self.types
            .get(id.0 as usize)
            .ok_or_else(|| "invalid type reference".into())
    }

    pub fn value(&self, id: TypeId) -> Result<Value, String> {
        self.get(id)?
            .value
            .ok_or_else(|| "type construction is incomplete".into())
    }

    pub fn register(
        &mut self,
        name: String,
        bases: Vec<TypeId>,
        mro: Vec<TypeId>,
        attributes: &HashMap<String, Value>,
    ) -> Result<TypeId, String> {
        let index = u32::try_from(self.types.len()).map_err(|_| "too many Python types")?;
        let local_slots = TypeSlots::from_attributes(attributes);
        let slots = self.inherit_slots(local_slots.clone(), &mro);
        let ty = PyType {
            name,
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
    pub fn replace_slots(
        &mut self,
        id: TypeId,
        attributes: &HashMap<String, Value>,
    ) -> Result<(), String> {
        let mro = self.get(id)?.mro.clone();
        let local_slots = TypeSlots::from_attributes(attributes);
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

    pub fn finish(&mut self, id: TypeId, value: Value) -> Result<(), String> {
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

    pub fn is_subclass(&self, class: TypeId, base: TypeId) -> Result<bool, String> {
        Ok(class == base || self.get(class)?.mro.contains(&base))
    }

    /// Linearize every registered base, builtin or heap-owned, with the same C3 rule used to
    /// register native value kinds. The result excludes the class being created.
    pub(super) fn linearize_bases(&self, bases: &[TypeId]) -> Result<Vec<TypeId>, String> {
        for (index, base) in bases.iter().enumerate() {
            if bases[..index].contains(base) {
                return Err("duplicate base class".into());
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
        c3_merge(sequences)
            .ok_or_else(|| "cannot create a consistent method resolution order".into())
    }

    pub fn slot(&self, type_id: TypeId, slot: Slot) -> Result<Option<SlotValue>, String> {
        Ok(self.get(type_id)?.slots.get(slot).cloned())
    }

    pub(super) fn local_slot(
        &self,
        type_id: TypeId,
        slot: Slot,
    ) -> Result<Option<SlotValue>, String> {
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
            bases,
            mro,
            attributes: HashMap::new(),
            local_slots,
            slots,
            value: Some(Value::Native(super::vm::NativeValue::ValueKind(kind))),
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
    let binary = SlotValue::NativeBinary;
    let unary = SlotValue::NativeUnary;
    TypeSlots {
        repr: slots.repr.map(unary),
        str_: slots.str_.map(unary),
        bool_: slots.bool_.map(unary),
        get_item: slots.get_item.map(binary),
        positive: slots.positive.map(unary),
        negative: slots.negative.map(unary),
        invert: slots.invert.map(unary),
        absolute: slots.absolute.map(unary),
        add: slots.add.map(binary),
        reflected_add: slots.reflected_add.map(binary),
        subtract: slots.subtract.map(binary),
        reflected_subtract: slots.reflected_subtract.map(binary),
        multiply: slots.multiply.map(binary),
        reflected_multiply: slots.reflected_multiply.map(binary),
        divide: slots.divide.map(binary),
        reflected_divide: slots.reflected_divide.map(binary),
        floor_divide: slots.floor_divide.map(binary),
        reflected_floor_divide: slots.reflected_floor_divide.map(binary),
        remainder: slots.remainder.map(binary),
        reflected_remainder: slots.reflected_remainder.map(binary),
        divmod: slots.divmod.map(binary),
        reflected_divmod: slots.reflected_divmod.map(binary),
        power: slots.power.map(binary),
        reflected_power: slots.reflected_power.map(binary),
        left_shift: slots.left_shift.map(binary),
        reflected_left_shift: slots.reflected_left_shift.map(binary),
        right_shift: slots.right_shift.map(binary),
        reflected_right_shift: slots.reflected_right_shift.map(binary),
        bitwise_and: slots.bitwise_and.map(binary),
        reflected_bitwise_and: slots.reflected_bitwise_and.map(binary),
        bitwise_xor: slots.bitwise_xor.map(binary),
        reflected_bitwise_xor: slots.reflected_bitwise_xor.map(binary),
        bitwise_or: slots.bitwise_or.map(binary),
        reflected_bitwise_or: slots.reflected_bitwise_or.map(binary),
        equal: slots.equal.map(binary),
        not_equal: slots.not_equal.map(binary),
        less_than: slots.less_than.map(binary),
        less_equal: slots.less_equal.map(binary),
        greater_than: slots.greater_than.map(binary),
        greater_equal: slots.greater_equal.map(binary),
        ..TypeSlots::default()
    }
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
            Value::Native(super::vm::NativeValue::NativeMethod(method)),
        );
    }
    for getter in getters {
        ty.attributes.insert(
            getter.name.into(),
            Value::Native(super::vm::NativeValue::NativeGetter(getter)),
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
        if let Some(method) = ty
            .attributes
            .get(name)
            .and_then(|value| match value.native_value() {
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
            Value::Native(super::vm::NativeValue::NativeClassMethod(method)),
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

fn install_builtin_slots(types: &mut [PyType]) {
    let intrinsic = SlotValue::NativeBinary;
    let unary = SlotValue::NativeUnary;
    types[BuiltinType::None as usize].slots.bool_ =
        Some(unary(super::stdlib::core::slot_none_bool));
    types[BuiltinType::None as usize].slots.hash = Some(unary(super::stdlib::core::slot_none_hash));
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
        types[builtin as usize].slots.repr = Some(SlotValue::VmRepr);
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
        slots.positive = Some(unary(super::number::slot_positive));
        slots.negative = Some(unary(super::number::slot_negative));
        slots.invert = Some(unary(super::number::slot_invert));
        slots.absolute = Some(unary(super::number::slot_absolute));
        slots.add = Some(intrinsic(super::number::slot_add));
        slots.reflected_add = Some(intrinsic(super::number::slot_add));
        slots.subtract = Some(intrinsic(super::number::slot_subtract));
        slots.reflected_subtract = Some(intrinsic(super::number::slot_reflected_subtract));
        slots.multiply = Some(intrinsic(super::number::slot_multiply));
        slots.reflected_multiply = Some(intrinsic(super::number::slot_multiply));
        slots.power = Some(intrinsic(super::number::slot_power));
        slots.reflected_power = Some(intrinsic(super::number::slot_reflected_power));
        slots.divide = Some(intrinsic(super::number::slot_divide));
        slots.reflected_divide = Some(intrinsic(super::number::slot_reflected_divide));
        slots.floor_divide = Some(intrinsic(super::number::slot_floor_divide));
        slots.reflected_floor_divide = Some(intrinsic(super::number::slot_reflected_floor_divide));
        slots.remainder = Some(intrinsic(super::number::slot_remainder));
        slots.reflected_remainder = Some(intrinsic(super::number::slot_reflected_remainder));
        slots.divmod = Some(intrinsic(super::number::slot_divmod));
        slots.reflected_divmod = Some(intrinsic(super::number::slot_reflected_divmod));
        slots.left_shift = Some(intrinsic(super::number::slot_left_shift));
        slots.reflected_left_shift = Some(intrinsic(super::number::slot_left_shift));
        slots.right_shift = Some(intrinsic(super::number::slot_right_shift));
        slots.reflected_right_shift = Some(intrinsic(super::number::slot_right_shift));
        slots.bitwise_and = Some(intrinsic(super::number::slot_bitwise_and));
        slots.reflected_bitwise_and = Some(intrinsic(super::number::slot_bitwise_and));
        slots.bitwise_xor = Some(intrinsic(super::number::slot_bitwise_xor));
        slots.reflected_bitwise_xor = Some(intrinsic(super::number::slot_bitwise_xor));
        slots.bitwise_or = Some(intrinsic(super::number::slot_bitwise_or));
        slots.reflected_bitwise_or = Some(intrinsic(super::number::slot_bitwise_or));
        slots.equal = Some(intrinsic(super::number::slot_equal));
        slots.not_equal = Some(intrinsic(super::number::slot_not_equal));
        slots.hash = Some(unary(super::number::slot_hash));
        slots.bool_ = Some(unary(super::number::slot_bool));
    }
    let slots = &mut types[BuiltinType::String as usize].slots;
    slots.equal = Some(intrinsic(super::stdlib::core::slot_string_equal));
    slots.not_equal = Some(intrinsic(super::stdlib::core::slot_string_not_equal));
    slots.less_than = Some(intrinsic(super::stdlib::core::slot_string_less));
    slots.less_equal = Some(intrinsic(super::stdlib::core::slot_string_less_equal));
    slots.greater_than = Some(intrinsic(super::stdlib::core::slot_string_greater));
    slots.greater_equal = Some(intrinsic(super::stdlib::core::slot_string_greater_equal));
    slots.hash = Some(unary(super::stdlib::core::slot_string_hash));
    slots.length = Some(unary(super::stdlib::core::slot_builtin_length));
    slots.iter = Some(unary(super::stdlib::core::slot_sequence_iter));
    slots.get_item = Some(intrinsic(super::stdlib::core::slot_builtin_get_item));
    slots.contains = Some(intrinsic(super::stdlib::core::slot_builtin_contains));
    slots.add = Some(intrinsic(super::stdlib::core::slot_string_add));
    slots.multiply = Some(intrinsic(super::stdlib::core::slot_string_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::core::slot_string_multiply));
    slots.remainder = Some(intrinsic(super::stdlib::core::slot_string_remainder));

    let slots = &mut types[BuiltinType::Bytes as usize].slots;
    slots.hash = Some(unary(super::stdlib::core::slot_bytes_hash));
    slots.length = Some(unary(super::stdlib::core::slot_bytes_length));
    slots.iter = Some(unary(super::stdlib::core::slot_sequence_iter));
    slots.contains = Some(intrinsic(super::stdlib::core::slot_builtin_contains));
    slots.get_item = Some(intrinsic(super::stdlib::core::slot_bytes_get_item));
    slots.add = Some(intrinsic(super::stdlib::core::slot_bytes_add));
    slots.multiply = Some(intrinsic(super::stdlib::core::slot_bytes_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::core::slot_bytes_multiply));

    let slots = &mut types[BuiltinType::ByteArray as usize].slots;
    slots.length = Some(unary(super::stdlib::core::slot_bytearray_length));
    slots.iter = Some(unary(super::stdlib::core::slot_sequence_iter));
    slots.contains = Some(intrinsic(super::stdlib::core::slot_builtin_contains));
    slots.get_item = Some(intrinsic(super::stdlib::core::slot_bytearray_get_item));
    slots.add = Some(intrinsic(super::stdlib::core::slot_bytearray_add));
    slots.multiply = Some(intrinsic(super::stdlib::core::slot_bytearray_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::core::slot_bytearray_multiply));
    slots.set_item = Some(SlotValue::NativeTernary(
        super::stdlib::core::slot_bytearray_set_item,
    ));
    slots.delete_item = Some(intrinsic(super::stdlib::core::slot_bytearray_delete_item));

    let slots = &mut types[BuiltinType::List as usize].slots;
    slots.length = Some(unary(super::stdlib::core::slot_builtin_length));
    slots.get_item = Some(intrinsic(super::stdlib::core::slot_builtin_get_item));
    slots.contains = Some(intrinsic(super::stdlib::core::slot_builtin_contains));
    slots.iter = Some(unary(super::stdlib::core::slot_sequence_iter));
    slots.set(
        Slot::Reversed,
        unary(super::stdlib::core::slot_sequence_reversed),
    );
    slots.add = Some(intrinsic(super::stdlib::core::slot_list_add));
    slots.multiply = Some(intrinsic(super::stdlib::core::slot_list_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::core::slot_list_multiply));
    slots.set_item = Some(SlotValue::NativeTernary(
        super::stdlib::core::slot_list_set_item,
    ));
    slots.delete_item = Some(intrinsic(super::stdlib::core::slot_list_delete_item));

    let slots = &mut types[BuiltinType::Dict as usize].slots;
    slots.length = Some(unary(super::stdlib::core::slot_builtin_length));
    slots.iter = Some(unary(super::stdlib::core::slot_sequence_iter));
    slots.contains = Some(intrinsic(super::stdlib::core::slot_builtin_contains));
    slots.set(
        Slot::Reversed,
        unary(super::stdlib::core::slot_dict_reversed),
    );
    slots.delete_item = Some(intrinsic(super::stdlib::core::slot_dict_delete_item));
    slots.bitwise_or = Some(intrinsic(super::stdlib::core::slot_dict_union));
    slots.reflected_bitwise_or = Some(intrinsic(super::stdlib::core::slot_dict_reflected_union));

    // The VM's builtin subscript, `len`, `in` and iteration read stored dicts directly, so a
    // namespace view supplies those through slots and inherits the rest from `dict`.
    types[BuiltinType::NamespaceDict as usize].slots =
        types[BuiltinType::Dict as usize].slots.clone();
    let slots = &mut types[BuiltinType::NamespaceDict as usize].slots;
    slots.get_item = Some(intrinsic(super::stdlib::core::slot_namespace_dict_get_item));
    slots.set_item = Some(SlotValue::NativeTernary(
        super::stdlib::core::slot_namespace_dict_set_item,
    ));
    slots.length = Some(unary(super::stdlib::core::slot_namespace_dict_length));
    slots.contains = Some(intrinsic(super::stdlib::core::slot_namespace_dict_contains));
    slots.iter = Some(unary(super::stdlib::core::slot_namespace_dict_iter));
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
        slots.length = Some(unary(views::slot_view_length));
        slots.iter = Some(unary(views::slot_view_iter));
        slots.set(Slot::Reversed, unary(views::slot_view_reversed));
        slots.contains = Some(intrinsic(views::slot_view_contains));
        if view == BuiltinType::DictValues {
            continue;
        }
        // Keys and items views are set-like; a values view compares by identity.
        slots.equal = Some(intrinsic(views::slot_view_equal));
        slots.less_than = Some(intrinsic(views::slot_view_less));
        slots.less_equal = Some(intrinsic(views::slot_view_less_equal));
        slots.greater_than = Some(intrinsic(views::slot_view_greater));
        slots.greater_equal = Some(intrinsic(views::slot_view_greater_equal));
        slots.bitwise_and = Some(intrinsic(views::slot_view_and));
        slots.reflected_bitwise_and = Some(intrinsic(views::slot_view_reflected_and));
        slots.bitwise_or = Some(intrinsic(views::slot_view_or));
        slots.reflected_bitwise_or = Some(intrinsic(views::slot_view_reflected_or));
        slots.bitwise_xor = Some(intrinsic(views::slot_view_xor));
        slots.reflected_bitwise_xor = Some(intrinsic(views::slot_view_reflected_xor));
        slots.subtract = Some(intrinsic(views::slot_view_subtract));
        slots.reflected_subtract = Some(intrinsic(views::slot_view_reflected_subtract));
    }
    let slots = &mut types[BuiltinType::MappingProxy as usize].slots;
    slots.get_item = Some(intrinsic(super::stdlib::mapping_views::slot_proxy_get_item));
    slots.length = Some(unary(super::stdlib::mapping_views::slot_proxy_length));
    slots.contains = Some(intrinsic(super::stdlib::mapping_views::slot_proxy_contains));
    slots.iter = Some(unary(super::stdlib::mapping_views::slot_proxy_iter));

    let slots = &mut types[BuiltinType::Set as usize].slots;
    slots.length = Some(unary(super::stdlib::core::slot_builtin_length));
    slots.iter = Some(unary(super::stdlib::core::slot_sequence_iter));
    slots.contains = Some(intrinsic(super::stdlib::core::slot_builtin_contains));
    slots.subtract = Some(intrinsic(super::stdlib::core::slot_set_subtract));
    slots.bitwise_and = Some(intrinsic(super::stdlib::core::slot_set_intersection));
    slots.reflected_bitwise_and = Some(intrinsic(super::stdlib::core::slot_set_intersection));
    slots.bitwise_xor = Some(intrinsic(
        super::stdlib::core::slot_set_symmetric_difference,
    ));
    slots.reflected_bitwise_xor = Some(intrinsic(
        super::stdlib::core::slot_set_symmetric_difference,
    ));
    slots.bitwise_or = Some(intrinsic(super::stdlib::core::slot_set_union));
    slots.reflected_bitwise_or = Some(intrinsic(super::stdlib::core::slot_set_union));
    slots.less_than = Some(intrinsic(super::stdlib::core::slot_set_less));
    slots.less_equal = Some(intrinsic(super::stdlib::core::slot_set_less_equal));
    slots.greater_than = Some(intrinsic(super::stdlib::core::slot_set_greater));
    slots.greater_equal = Some(intrinsic(super::stdlib::core::slot_set_greater_equal));

    types[BuiltinType::FrozenSet as usize].slots = types[BuiltinType::Set as usize].slots.clone();

    let slots = &mut types[BuiltinType::Tuple as usize].slots;
    slots.length = Some(unary(super::stdlib::core::slot_builtin_length));
    slots.get_item = Some(intrinsic(super::stdlib::core::slot_builtin_get_item));
    slots.contains = Some(intrinsic(super::stdlib::core::slot_builtin_contains));
    slots.iter = Some(unary(super::stdlib::core::slot_sequence_iter));
    slots.set(
        Slot::Reversed,
        unary(super::stdlib::core::slot_sequence_reversed),
    );
    slots.add = Some(intrinsic(super::stdlib::core::slot_tuple_add));
    slots.multiply = Some(intrinsic(super::stdlib::core::slot_tuple_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::core::slot_tuple_multiply));

    types[BuiltinType::Range as usize].slots.iter =
        Some(unary(super::stdlib::core::slot_sequence_iter));
    types[BuiltinType::Range as usize].slots.length =
        Some(unary(super::stdlib::core::slot_builtin_length));
    types[BuiltinType::Range as usize].slots.get_item =
        Some(intrinsic(super::stdlib::core::slot_builtin_get_item));
    types[BuiltinType::Range as usize].slots.contains =
        Some(intrinsic(super::stdlib::core::slot_builtin_contains));
    types[BuiltinType::Range as usize].slots.set(
        Slot::Reversed,
        unary(super::stdlib::core::slot_sequence_reversed),
    );

    let slots = &mut types[BuiltinType::Array as usize].slots;
    {
        use super::stdlib::numpy as np;
        slots.repr = Some(unary(np::slot_repr));
        slots.str_ = Some(unary(np::slot_str));
        slots.bool_ = Some(unary(np::slot_bool));
        slots.iter = Some(unary(np::slot_iter));
        slots.length = Some(unary(np::slot_length));
        slots.get_item = Some(intrinsic(np::slot_get_item));
        slots.set_item = Some(SlotValue::NativeTernary(np::slot_set_item));
        slots.positive = Some(unary(np::slot_positive));
        slots.negative = Some(unary(np::slot_negative));
        slots.invert = Some(unary(np::slot_invert));
        slots.absolute = Some(unary(np::slot_absolute));
        slots.add = Some(intrinsic(np::slot_add));
        slots.reflected_add = Some(intrinsic(np::slot_reflected_add));
        slots.subtract = Some(intrinsic(np::slot_subtract));
        slots.reflected_subtract = Some(intrinsic(np::slot_reflected_subtract));
        slots.multiply = Some(intrinsic(np::slot_multiply));
        slots.reflected_multiply = Some(intrinsic(np::slot_reflected_multiply));
        slots.matrix_multiply = Some(intrinsic(np::slot_matrix_multiply));
        slots.reflected_matrix_multiply = Some(intrinsic(np::slot_reflected_matrix_multiply));
        slots.power = Some(intrinsic(np::slot_power));
        slots.reflected_power = Some(intrinsic(np::slot_reflected_power));
        slots.divide = Some(intrinsic(np::slot_divide));
        slots.reflected_divide = Some(intrinsic(np::slot_reflected_divide));
        slots.floor_divide = Some(intrinsic(np::slot_floor_divide));
        slots.reflected_floor_divide = Some(intrinsic(np::slot_reflected_floor_divide));
        slots.remainder = Some(intrinsic(np::slot_remainder));
        slots.reflected_remainder = Some(intrinsic(np::slot_reflected_remainder));
        slots.divmod = Some(intrinsic(np::slot_divmod));
        slots.reflected_divmod = Some(intrinsic(np::slot_reflected_divmod));
        slots.left_shift = Some(intrinsic(np::slot_left_shift));
        slots.reflected_left_shift = Some(intrinsic(np::slot_reflected_left_shift));
        slots.right_shift = Some(intrinsic(np::slot_right_shift));
        slots.reflected_right_shift = Some(intrinsic(np::slot_reflected_right_shift));
        slots.bitwise_and = Some(intrinsic(np::slot_bitwise_and));
        slots.reflected_bitwise_and = Some(intrinsic(np::slot_reflected_bitwise_and));
        slots.bitwise_xor = Some(intrinsic(np::slot_bitwise_xor));
        slots.reflected_bitwise_xor = Some(intrinsic(np::slot_reflected_bitwise_xor));
        slots.bitwise_or = Some(intrinsic(np::slot_bitwise_or));
        slots.reflected_bitwise_or = Some(intrinsic(np::slot_reflected_bitwise_or));
        slots.equal = Some(intrinsic(np::slot_equal));
        slots.not_equal = Some(intrinsic(np::slot_not_equal));
        slots.less_than = Some(intrinsic(np::slot_less_than));
        slots.less_equal = Some(intrinsic(np::slot_less_equal));
        slots.greater_than = Some(intrinsic(np::slot_greater_than));
        slots.greater_equal = Some(intrinsic(np::slot_greater_equal));
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
    slots.equal = Some(intrinsic(super::number::slot_equal));
    slots.not_equal = Some(intrinsic(super::number::slot_not_equal));
    slots.hash = Some(unary(super::number::slot_hash));
    slots.bool_ = Some(unary(super::number::slot_bool));
    slots.positive = Some(unary(super::complex::slot_positive));
    slots.negative = Some(unary(super::complex::slot_negative));
    slots.absolute = Some(unary(super::complex::slot_absolute));
    slots.hash = Some(unary(super::complex::slot_hash));
    slots.add = Some(intrinsic(super::complex::slot_add));
    slots.reflected_add = Some(intrinsic(super::complex::slot_add));
    slots.subtract = Some(intrinsic(super::complex::slot_subtract));
    slots.reflected_subtract = Some(intrinsic(super::complex::slot_reflected_subtract));
    slots.multiply = Some(intrinsic(super::complex::slot_multiply));
    slots.reflected_multiply = Some(intrinsic(super::complex::slot_multiply));
    slots.divide = Some(intrinsic(super::complex::slot_divide));
    slots.reflected_divide = Some(intrinsic(super::complex::slot_reflected_divide));
    slots.power = Some(intrinsic(super::complex::slot_power));
    slots.reflected_power = Some(intrinsic(super::complex::slot_reflected_power));
    slots.floor_divide = Some(intrinsic(super::complex::slot_floor_divide));
    slots.reflected_floor_divide = Some(intrinsic(super::complex::slot_reflected_floor_divide));
    slots.remainder = Some(intrinsic(super::complex::slot_remainder));
    slots.reflected_remainder = Some(intrinsic(super::complex::slot_reflected_remainder));

    let stream = &mut types[BuiltinType::Stream as usize].slots;
    stream.iter = Some(unary(super::stdlib::sys::slot_iter));
    stream.next = Some(unary(super::stdlib::sys::slot_next));

    types[BuiltinType::NotImplemented as usize].slots.bool_ = Some(unary(not_implemented_bool));
    types[BuiltinType::Enum as usize].slots.repr = Some(SlotValue::VmRepr);
    types[BuiltinType::Enum as usize].slots.str_ = Some(SlotValue::VmEnumString);
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
                    | SlotValue::VmEnumString
            )
        ) {
            ty.attributes.entry(name.into()).or_insert(Value::Native(
                super::vm::NativeValue::SlotWrapper { owner, slot },
            ));
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
                &HashMap::from([("value".into(), Value::Int(1))]),
            )
            .unwrap();
        let registered_bytes = modeled_type_bytes(registry.get(id).unwrap());
        assert_eq!(registry.get(id).unwrap().attributes.get("value"), None);
        assert_eq!(
            registry.modeled_bytes(),
            before.saturating_add(registered_bytes)
        );

        registry.finish(id, Value::None).unwrap();
        assert_eq!(
            registry.modeled_bytes(),
            before.saturating_add(registered_bytes)
        );
    }
}
