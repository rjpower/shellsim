//! Python semantic types and bootstrapped builtin type metadata.
//!
//! `TypeId` is independent of storage shape: immediate values, arena objects, and native markers
//! all identify their Python type through the same registry. The registry stores only modeled
//! metadata and Python values, so looking up a type cannot acquire host capabilities.

use std::collections::HashMap;

use super::native::BinarySlotFn;
use super::Value;

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
    List,
    Tuple,
    Dict,
    Set,
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
}

impl BuiltinType {
    pub(super) const ALL: [Self; 24] = [
        Self::Object,
        Self::Type,
        Self::None,
        Self::Bool,
        Self::Int,
        Self::Float,
        Self::String,
        Self::List,
        Self::Tuple,
        Self::Dict,
        Self::Set,
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
            Self::List => "list",
            Self::Tuple => "tuple",
            Self::Dict => "dict",
            Self::Set => "set",
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
        }
    }
}

/// Storage layout required by instances of a type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyLayout {
    Object,
    Int,
    Type,
    Native,
}

/// Cached protocol methods resolved from a type dictionary.
///
/// A user slot holds its ordinary Python descriptor. Builtin binary slots hold a direct native
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
    pub add: Option<SlotValue>,
    pub reflected_add: Option<SlotValue>,
    pub subtract: Option<SlotValue>,
    pub reflected_subtract: Option<SlotValue>,
    pub multiply: Option<SlotValue>,
    pub reflected_multiply: Option<SlotValue>,
    pub equal: Option<SlotValue>,
    pub less_than: Option<SlotValue>,
    pub contains: Option<SlotValue>,
}

/// A cached Python descriptor or a native implementation attached directly to a builtin type.
#[derive(Clone, Debug)]
pub enum SlotValue {
    Descriptor(Value),
    NativeBinary(BinarySlotFn),
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
    Add,
    ReflectedAdd,
    Subtract,
    ReflectedSubtract,
    Multiply,
    ReflectedMultiply,
    Equal,
    LessThan,
    Contains,
}

impl Slot {
    const ALL: [Self; 20] = [
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
        Self::Add,
        Self::ReflectedAdd,
        Self::Subtract,
        Self::ReflectedSubtract,
        Self::Multiply,
        Self::ReflectedMultiply,
        Self::Equal,
        Self::LessThan,
        Self::Contains,
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
            add: get("__add__"),
            reflected_add: get("__radd__"),
            subtract: get("__sub__"),
            reflected_subtract: get("__rsub__"),
            multiply: get("__mul__"),
            reflected_multiply: get("__rmul__"),
            equal: get("__eq__"),
            less_than: get("__lt__"),
            contains: get("__contains__"),
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
            &self.add,
            &self.reflected_add,
            &self.subtract,
            &self.reflected_subtract,
            &self.multiply,
            &self.reflected_multiply,
            &self.equal,
            &self.less_than,
            &self.contains,
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
            Slot::Add => self.add.as_ref(),
            Slot::ReflectedAdd => self.reflected_add.as_ref(),
            Slot::Subtract => self.subtract.as_ref(),
            Slot::ReflectedSubtract => self.reflected_subtract.as_ref(),
            Slot::Multiply => self.multiply.as_ref(),
            Slot::ReflectedMultiply => self.reflected_multiply.as_ref(),
            Slot::Equal => self.equal.as_ref(),
            Slot::LessThan => self.less_than.as_ref(),
            Slot::Contains => self.contains.as_ref(),
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
            Slot::Add => &mut self.add,
            Slot::ReflectedAdd => &mut self.reflected_add,
            Slot::Subtract => &mut self.subtract,
            Slot::ReflectedSubtract => &mut self.reflected_subtract,
            Slot::Multiply => &mut self.multiply,
            Slot::ReflectedMultiply => &mut self.reflected_multiply,
            Slot::Equal => &mut self.equal,
            Slot::LessThan => &mut self.less_than,
            Slot::Contains => &mut self.contains,
        } = Some(value);
    }
}

/// Metadata shared by builtin, native, and user-defined Python types.
#[derive(Clone, Debug)]
pub struct PyType {
    pub name: String,
    pub bases: Vec<TypeId>,
    pub mro: Vec<TypeId>,
    pub metaclass: TypeId,
    pub attributes: HashMap<String, Value>,
    pub layout: PyLayout,
    pub slots: TypeSlots,
    value: Option<Value>,
}

/// Per-runtime registry containing all semantic Python types.
#[derive(Clone, Debug)]
pub struct TypeRegistry {
    types: Vec<PyType>,
}

impl Default for TypeRegistry {
    fn default() -> Self {
        let mut types = Vec::with_capacity(BuiltinType::ALL.len());
        for builtin in BuiltinType::ALL {
            let (bases, mro, metaclass, layout) = builtin_metadata(builtin);
            types.push(PyType {
                name: builtin.name().into(),
                bases,
                mro,
                metaclass,
                attributes: HashMap::new(),
                layout,
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
        install_builtin_slots(&mut types);
        Self { types }
    }
}

impl TypeRegistry {
    /// Conservative modeled size of registry metadata retained between executions.
    pub fn modeled_bytes(&self) -> u64 {
        self.types.iter().fold(0u64, |total, ty| {
            let variable = ty
                .name
                .len()
                .saturating_add(ty.bases.len().saturating_mul(4))
                .saturating_add(ty.mro.len().saturating_mul(4))
                .saturating_add(ty.attributes.len().saturating_mul(48))
                .saturating_add(ty.slots.populated_count().saturating_mul(24));
            let fixed = match ty.layout {
                PyLayout::Object | PyLayout::Int | PyLayout::Type | PyLayout::Native => 64usize,
            };
            let _metaclass = ty.metaclass;
            total.saturating_add(u64::try_from(variable.saturating_add(fixed)).unwrap_or(u64::MAX))
        })
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
        metaclass: TypeId,
        attributes: HashMap<String, Value>,
        layout: PyLayout,
    ) -> Result<TypeId, String> {
        let index = u32::try_from(self.types.len()).map_err(|_| "too many Python types")?;
        let mut slots = TypeSlots::from_attributes(&attributes);
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
        self.types.push(PyType {
            name,
            bases,
            mro,
            metaclass,
            attributes,
            layout,
            slots,
            value: None,
        });
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
}

fn builtin_metadata(builtin: BuiltinType) -> (Vec<TypeId>, Vec<TypeId>, TypeId, PyLayout) {
    let object = BuiltinType::Object.id();
    let type_ = BuiltinType::Type.id();
    match builtin {
        BuiltinType::Object => (Vec::new(), Vec::new(), type_, PyLayout::Object),
        BuiltinType::Type => (vec![object], vec![object], type_, PyLayout::Type),
        BuiltinType::Bool => (
            vec![BuiltinType::Int.id()],
            vec![BuiltinType::Int.id(), object],
            type_,
            PyLayout::Int,
        ),
        BuiltinType::Int => (vec![object], vec![object], type_, PyLayout::Int),
        BuiltinType::Native
        | BuiltinType::Regex
        | BuiltinType::Match
        | BuiltinType::Stream
        | BuiltinType::Environment
        | BuiltinType::ArgumentParser
        | BuiltinType::RaisesContext => (vec![object], vec![object], type_, PyLayout::Native),
        BuiltinType::Property => (vec![object], vec![object], type_, PyLayout::Object),
        _ => (vec![object], vec![object], type_, PyLayout::Object),
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
    for builtin in [BuiltinType::Bool, BuiltinType::Int, BuiltinType::Float] {
        let slots = &mut types[builtin as usize].slots;
        slots.add = Some(intrinsic(super::number::slot_add));
        slots.reflected_add = Some(intrinsic(super::number::slot_add));
        slots.subtract = Some(intrinsic(super::number::slot_subtract));
        slots.reflected_subtract = Some(intrinsic(super::number::slot_reflected_subtract));
        slots.multiply = Some(intrinsic(super::number::slot_multiply));
        slots.reflected_multiply = Some(intrinsic(super::number::slot_multiply));
    }
    let slots = &mut types[BuiltinType::String as usize].slots;
    slots.add = Some(intrinsic(super::stdlib::core::slot_string_add));
    slots.multiply = Some(intrinsic(super::stdlib::core::slot_string_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::core::slot_string_multiply));

    let slots = &mut types[BuiltinType::List as usize].slots;
    slots.add = Some(intrinsic(super::stdlib::core::slot_list_add));
    slots.multiply = Some(intrinsic(super::stdlib::core::slot_list_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::core::slot_list_multiply));

    let slots = &mut types[BuiltinType::Tuple as usize].slots;
    slots.add = Some(intrinsic(super::stdlib::core::slot_tuple_add));
    slots.multiply = Some(intrinsic(super::stdlib::core::slot_tuple_multiply));
    slots.reflected_multiply = Some(intrinsic(super::stdlib::core::slot_tuple_multiply));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_closes_object_type_cycle_and_bool_int_hierarchy() {
        let registry = TypeRegistry::default();
        assert_eq!(
            registry.get(BuiltinType::Object.id()).unwrap().metaclass,
            BuiltinType::Type.id()
        );
        assert_eq!(
            registry.get(BuiltinType::Type.id()).unwrap().metaclass,
            BuiltinType::Type.id()
        );
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
}
