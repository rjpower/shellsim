//! Python semantic types and bootstrapped builtin type metadata.
//!
//! `TypeId` is independent of storage shape: immediate values, arena objects, and native markers
//! all identify their Python type through the same registry. The registry stores only modeled
//! metadata and Python values, so looking up a type cannot acquire host capabilities.

use std::collections::HashMap;

use super::native::{BinarySlotFn, TernarySlotFn, UnarySlotFn};
use super::Value;

// Builtins and native value kinds are immutable process metadata. A fixed charge keeps their
// accounting out of the VM hot path; user-defined types are charged when registered below.
const BUILTIN_TYPE_MEMORY_BYTES: u64 = 16 * 1024;

/// Stable identity of a Python type within a [`ReplState`](super::ReplState).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeId(u32);

impl TypeId {
    const fn builtin(value: BuiltinType) -> Self {
        Self(value as u32)
    }
}

/// Canonical builtin type objects and their stable registry indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub(super) enum BuiltinType {
    Object,
    Type,
    None,
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
    ArgumentParser,
    RaisesContext,
    Property,
    Array,
}

impl BuiltinType {
    pub(super) const ALL: [Self; 28] = [
        Self::Object,
        Self::Type,
        Self::None,
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
        Self::ArgumentParser,
        Self::RaisesContext,
        Self::Property,
        Self::Array,
    ];

    pub(super) const fn id(self) -> TypeId {
        TypeId::builtin(self)
    }

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Object => "object",
            Self::Type => "type",
            Self::None => "NoneType",
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
            Self::ArgumentParser => "argparse.ArgumentParser",
            Self::RaisesContext => "pytest.raises",
            Self::Property => "property",
            Self::Array => "numpy.ndarray",
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
}

/// A cached Python descriptor or a native implementation attached directly to a builtin type.
#[derive(Clone, Debug)]
pub enum SlotValue {
    Descriptor(Value),
    NativeBinary(BinarySlotFn),
    NativeTernary(TernarySlotFn),
    NativeUnary(UnarySlotFn),
}

/// Protocol operations cached on each type after MRO resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

impl Slot {
    const ALL: [Self; 52] = [
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
    ];
}

impl TypeSlots {
    fn from_attributes(attributes: &HashMap<String, Value>) -> Self {
        let get = |name: &str| attributes.get(name).cloned().map(SlotValue::Descriptor);
        Self {
            call: get("__call__"),
            new: get("__new__"),
            init: get("__init__"),
            getattribute: get("__getattribute__"),
            setattr: get("__setattr__"),
            repr: get("__repr__"),
            str_: get("__str__"),
            bool_: get("__bool__"),
            hash: get("__hash__"),
            iter: get("__iter__"),
            next: get("__next__"),
            length: get("__len__"),
            get_item: get("__getitem__"),
            set_item: get("__setitem__"),
            positive: get("__pos__"),
            negative: get("__neg__"),
            invert: get("__invert__"),
            absolute: get("__abs__"),
            add: get("__add__"),
            reflected_add: get("__radd__"),
            subtract: get("__sub__"),
            reflected_subtract: get("__rsub__"),
            multiply: get("__mul__"),
            reflected_multiply: get("__rmul__"),
            matrix_multiply: get("__matmul__"),
            reflected_matrix_multiply: get("__rmatmul__"),
            power: get("__pow__"),
            reflected_power: get("__rpow__"),
            divide: get("__truediv__"),
            reflected_divide: get("__rtruediv__"),
            floor_divide: get("__floordiv__"),
            reflected_floor_divide: get("__rfloordiv__"),
            remainder: get("__mod__"),
            reflected_remainder: get("__rmod__"),
            left_shift: get("__lshift__"),
            reflected_left_shift: get("__rlshift__"),
            right_shift: get("__rshift__"),
            reflected_right_shift: get("__rrshift__"),
            bitwise_and: get("__and__"),
            reflected_bitwise_and: get("__rand__"),
            bitwise_xor: get("__xor__"),
            reflected_bitwise_xor: get("__rxor__"),
            bitwise_or: get("__or__"),
            reflected_bitwise_or: get("__ror__"),
            equal: get("__eq__"),
            not_equal: get("__ne__"),
            less_than: get("__lt__"),
            less_equal: get("__le__"),
            greater_than: get("__gt__"),
            greater_equal: get("__ge__"),
            contains: get("__contains__"),
            delete_item: get("__delitem__"),
        }
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
    pub slots: TypeSlots,
    value: Option<Value>,
}

/// Per-runtime registry containing all semantic Python types.
#[derive(Clone, Debug)]
pub struct TypeRegistry {
    types: Vec<PyType>,
    value_kinds: Vec<&'static super::native::ValueKindDef>,
    modeled_bytes: u64,
}

fn modeled_type_bytes(ty: &PyType) -> u64 {
    let bytes = 64usize
        .saturating_add(ty.name.len())
        .saturating_add(ty.bases.len().saturating_mul(4))
        .saturating_add(ty.mro.len().saturating_mul(4))
        .saturating_add(ty.attributes.len().saturating_mul(48))
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
                slots: TypeSlots::default(),
                value: Some(Value::Native(super::vm::NativeValue::BuiltinType(builtin))),
            });
        }
        install_native_methods(
            &mut types[BuiltinType::Type as usize],
            &super::stdlib::core::TYPE_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::String as usize],
            &super::stdlib::core::STRING_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::Bytes as usize],
            &super::stdlib::core::BYTES_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::ByteArray as usize],
            &super::stdlib::core::BYTEARRAY_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::List as usize],
            &super::stdlib::core::LIST_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::Dict as usize],
            &super::stdlib::core::DICT_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::Set as usize],
            &super::stdlib::core::SET_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::Property as usize],
            &super::stdlib::core::PROPERTY_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::Regex as usize],
            &super::stdlib::re::PATTERN_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::Match as usize],
            &super::stdlib::re::MATCH_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::Stream as usize],
            &super::stdlib::sys::STREAM_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::Environment as usize],
            &super::stdlib::os::ENVIRONMENT_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::ArgumentParser as usize],
            &super::stdlib::argparse::ARGUMENT_PARSER_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::RaisesContext as usize],
            &super::stdlib::pytest::RAISES_CONTEXT_TYPE,
        );
        install_native_methods(
            &mut types[BuiltinType::Array as usize],
            &super::stdlib::numpy::ARRAY_TYPE,
        );
        install_builtin_slots(&mut types);
        let mut registry = Self {
            types,
            value_kinds: Vec::new(),
            modeled_bytes: BUILTIN_TYPE_MEMORY_BYTES,
        };
        for kind in super::stdlib::value_kinds() {
            registry.register_value_kind(kind);
        }
        registry
    }
}

impl TypeRegistry {
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
        let mut slots = TypeSlots::from_attributes(attributes);
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
        let ty = PyType {
            name,
            bases,
            mro,
            attributes: HashMap::new(),
            slots,
            value: None,
        };
        self.modeled_bytes = self.modeled_bytes.saturating_add(modeled_type_bytes(&ty));
        self.types.push(ty);
        Ok(TypeId(index))
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

    pub fn slot(&self, type_id: TypeId, slot: Slot) -> Result<Option<SlotValue>, String> {
        Ok(self.get(type_id)?.slots.get(slot).cloned())
    }

    pub fn attribute(&self, type_id: TypeId, name: &str) -> Result<Option<Value>, String> {
        Ok(self.get(type_id)?.attributes.get(name).cloned())
    }

    fn register_value_kind(&mut self, kind: &'static super::native::ValueKindDef) {
        let object = BuiltinType::Object.id();
        let type_id = TypeId(u32::try_from(self.types.len()).expect("too many registered types"));
        self.types.push(PyType {
            name: kind.name.into(),
            bases: vec![object],
            mro: vec![object],
            attributes: HashMap::new(),
            slots: value_kind_slots(kind.slots),
            value: Some(Value::Native(super::vm::NativeValue::ValueKind(kind))),
        });
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

fn value_kind_slots(slots: super::native::ValueKindSlots) -> TypeSlots {
    let binary = SlotValue::NativeBinary;
    let unary = SlotValue::NativeUnary;
    TypeSlots {
        repr: slots.repr.map(unary),
        bool_: slots.bool_.map(unary),
        add: slots.add.map(binary),
        reflected_add: slots.reflected_add.map(binary),
        subtract: slots.subtract.map(binary),
        reflected_subtract: slots.reflected_subtract.map(binary),
        multiply: slots.multiply.map(binary),
        reflected_multiply: slots.reflected_multiply.map(binary),
        divide: slots.divide.map(binary),
        reflected_divide: slots.reflected_divide.map(binary),
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
        _ => (vec![object], vec![object]),
    }
}

fn install_native_methods(ty: &mut PyType, definition: &'static super::native::NativeTypeDef) {
    debug_assert_eq!(ty.name, definition.name);
    for method in definition.methods {
        debug_assert_eq!(method.type_name, definition.name);
        ty.attributes.insert(
            method.name.into(),
            Value::Native(super::vm::NativeValue::NativeMethod(method)),
        );
    }
}

fn install_builtin_slots(types: &mut [PyType]) {
    let intrinsic = SlotValue::NativeBinary;
    let unary = SlotValue::NativeUnary;
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
    }
    let slots = &mut types[BuiltinType::String as usize].slots;
    slots.add = Some(intrinsic(super::stdlib::core::slot_string_add));
    slots.multiply = Some(intrinsic(super::stdlib::core::slot_string_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::core::slot_string_multiply));

    let slots = &mut types[BuiltinType::Bytes as usize].slots;
    slots.length = Some(unary(super::stdlib::core::slot_bytes_length));
    slots.get_item = Some(intrinsic(super::stdlib::core::slot_bytes_get_item));
    slots.add = Some(intrinsic(super::stdlib::core::slot_bytes_add));
    slots.multiply = Some(intrinsic(super::stdlib::core::slot_bytes_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::core::slot_bytes_multiply));

    let slots = &mut types[BuiltinType::ByteArray as usize].slots;
    slots.length = Some(unary(super::stdlib::core::slot_bytearray_length));
    slots.get_item = Some(intrinsic(super::stdlib::core::slot_bytearray_get_item));
    slots.add = Some(intrinsic(super::stdlib::core::slot_bytearray_add));
    slots.multiply = Some(intrinsic(super::stdlib::core::slot_bytearray_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::core::slot_bytearray_multiply));
    slots.set_item = Some(SlotValue::NativeTernary(
        super::stdlib::core::slot_bytearray_set_item,
    ));
    slots.delete_item = Some(intrinsic(super::stdlib::core::slot_bytearray_delete_item));

    let slots = &mut types[BuiltinType::List as usize].slots;
    slots.add = Some(intrinsic(super::stdlib::core::slot_list_add));
    slots.multiply = Some(intrinsic(super::stdlib::core::slot_list_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::core::slot_list_multiply));
    slots.set_item = Some(SlotValue::NativeTernary(
        super::stdlib::core::slot_list_set_item,
    ));
    slots.delete_item = Some(intrinsic(super::stdlib::core::slot_list_delete_item));

    types[BuiltinType::Dict as usize].slots.delete_item =
        Some(intrinsic(super::stdlib::core::slot_dict_delete_item));

    let slots = &mut types[BuiltinType::Set as usize].slots;
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

    let slots = &mut types[BuiltinType::Tuple as usize].slots;
    slots.add = Some(intrinsic(super::stdlib::core::slot_tuple_add));
    slots.multiply = Some(intrinsic(super::stdlib::core::slot_tuple_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::core::slot_tuple_multiply));

    let slots = &mut types[BuiltinType::Array as usize].slots;
    slots.repr = Some(unary(super::stdlib::numpy::slot_repr));
    slots.bool_ = Some(unary(super::stdlib::numpy::slot_bool));
    slots.iter = Some(unary(super::stdlib::numpy::slot_iter));
    slots.length = Some(unary(super::stdlib::numpy::slot_length));
    slots.get_item = Some(intrinsic(super::stdlib::numpy::slot_get_item));
    slots.set_item = Some(SlotValue::NativeTernary(
        super::stdlib::numpy::slot_set_item,
    ));
    slots.add = Some(intrinsic(super::stdlib::numpy::slot_add));
    slots.reflected_add = Some(intrinsic(super::stdlib::numpy::slot_reflected_add));
    slots.subtract = Some(intrinsic(super::stdlib::numpy::slot_subtract));
    slots.reflected_subtract = Some(intrinsic(super::stdlib::numpy::slot_reflected_subtract));
    slots.multiply = Some(intrinsic(super::stdlib::numpy::slot_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::numpy::slot_reflected_multiply));
    slots.matrix_multiply = Some(intrinsic(super::stdlib::numpy::slot_matrix_multiply));
    slots.reflected_matrix_multiply = Some(intrinsic(
        super::stdlib::numpy::slot_reflected_matrix_multiply,
    ));
    slots.divide = Some(intrinsic(super::stdlib::numpy::slot_divide));
    slots.reflected_divide = Some(intrinsic(super::stdlib::numpy::slot_reflected_divide));
    slots.positive = Some(unary(super::stdlib::numpy::slot_positive));
    slots.negative = Some(unary(super::stdlib::numpy::slot_negative));
    slots.invert = Some(unary(super::stdlib::numpy::slot_invert));
    slots.absolute = Some(unary(super::stdlib::numpy::slot_absolute));
    slots.equal = Some(intrinsic(super::stdlib::numpy::slot_equal));
    slots.not_equal = Some(intrinsic(super::stdlib::numpy::slot_not_equal));
    slots.less_than = Some(intrinsic(super::stdlib::numpy::slot_less_than));
    slots.less_equal = Some(intrinsic(super::stdlib::numpy::slot_less_equal));
    slots.greater_than = Some(intrinsic(super::stdlib::numpy::slot_greater_than));
    slots.greater_equal = Some(intrinsic(super::stdlib::numpy::slot_greater_equal));

    let stream = &mut types[BuiltinType::Stream as usize].slots;
    stream.iter = Some(unary(super::stdlib::sys::slot_iter));
    stream.next = Some(unary(super::stdlib::sys::slot_next));
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
    fn modeled_memory_uses_a_fixed_builtin_charge_and_tracks_registered_types() {
        let mut registry = TypeRegistry::default();
        assert_eq!(registry.modeled_bytes(), BUILTIN_TYPE_MEMORY_BYTES);

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
        assert_eq!(registry.attribute(id, "value").unwrap(), None);
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
