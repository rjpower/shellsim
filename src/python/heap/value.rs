//! Value representations: the heap's own word, stored references, and pinned values.
//!
//! Three types share one compact 16-byte layout (a tag, a 64-bit payload and seven auxiliary
//! bytes) and differ only in who may hold them:
//!
//! - [`Raw`] is the heap's own word: an immediate or an [`ObjectId`].
//! - [`Ref`] is a stored reference inside a heap object or a VM root, which the collector
//!   traces. It is not `Copy`, so a reference cannot be duplicated into an untraced place by
//!   accident.
//! - [`Value`] is the word Rust code works with. Objects never move, so a value names its object
//!   directly; it is rooted by the heap's pin stack from the moment it is made until the
//!   instruction or native loop that made it ends. A value is not `Send`, and interpreter state
//!   must be, so a value cannot be stored in the VM's persistent state.
//!
//! Immediates (ints, floats, bools, `None`, short strings, native markers and registered inline
//! values) need no pin.

use std::marker::PhantomData;

use super::ObjectId;
use crate::python::string::InlineString;
use crate::python::vm::NativeValue;

/// Physical storage discriminator kept separate from Python's semantic
/// [`TypeId`](crate::python::object_model::TypeId).
///
/// Tags describe storage only. Python semantics come from the value's registered `TypeId`, so a
/// short string and a heap string have the same Python type despite different physical tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub(super) enum ValueTag {
    SmallString0,
    SmallString1,
    SmallString2,
    SmallString3,
    SmallString4,
    SmallString5,
    SmallString6,
    SmallString7,
    SmallString8,
    SmallString9,
    SmallString10,
    SmallString11,
    SmallString12,
    SmallString13,
    SmallString14,
    SmallString15,
    Int,
    Float,
    Bool,
    None,
    Object,
    Native,
    Registered,
}

/// The heap's 16-byte word. For [`ValueTag::Object`] the payload is an [`ObjectId`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub(super) struct Raw {
    payload: u64,
    aux: [u8; 7],
    tag: ValueTag,
}

impl Raw {
    pub(super) const NONE: Self = Self {
        payload: 0,
        aux: [0; 7],
        tag: ValueTag::None,
    };

    pub(super) const fn int(value: i64) -> Self {
        Self {
            payload: value as u64,
            aux: [0; 7],
            tag: ValueTag::Int,
        }
    }

    pub(super) const fn float(value: f64) -> Self {
        Self {
            payload: value.to_bits(),
            aux: [0; 7],
            tag: ValueTag::Float,
        }
    }

    pub(super) const fn bool(value: bool) -> Self {
        Self {
            payload: value as u64,
            aux: [0; 7],
            tag: ValueTag::Bool,
        }
    }

    pub(super) const fn object(id: ObjectId) -> Self {
        Self {
            payload: id.bits(),
            aux: [0; 7],
            tag: ValueTag::Object,
        }
    }

    pub(super) fn registered(kind: u8, payload: u64) -> Self {
        let mut aux = [0; 7];
        aux[0] = kind;
        Self {
            payload,
            aux,
            tag: ValueTag::Registered,
        }
    }

    pub(super) fn native(value: NativeValue) -> Self {
        let (payload, native_tag) = value.encode();
        let mut aux = [0; 7];
        aux[0] = native_tag;
        Self {
            payload,
            aux,
            tag: ValueTag::Native,
        }
    }

    pub(super) fn inline_string(value: &str) -> Option<Self> {
        if value.len() > 15 {
            return None;
        }
        let mut bytes = [0; 15];
        bytes[..value.len()].copy_from_slice(value.as_bytes());
        let mut payload = [0; 8];
        payload.copy_from_slice(&bytes[..8]);
        let mut aux = [0; 7];
        aux.copy_from_slice(&bytes[8..]);
        Some(Self {
            payload: u64::from_ne_bytes(payload),
            aux,
            tag: string_tag(value.len()),
        })
    }

    pub(super) const fn object_id(self) -> Option<ObjectId> {
        match self.tag {
            ValueTag::Object => Some(ObjectId::from_bits(self.payload)),
            _ => None,
        }
    }

    pub(super) const fn is_object(self) -> bool {
        matches!(self.tag, ValueTag::Object)
    }

    const fn immediate_int(self) -> Option<i64> {
        match self.tag {
            ValueTag::Int | ValueTag::Bool => Some(self.payload as i64),
            _ => None,
        }
    }

    const fn float_value(self) -> Option<f64> {
        match self.tag {
            ValueTag::Float => Some(f64::from_bits(self.payload)),
            _ => None,
        }
    }

    const fn bool_value(self) -> Option<bool> {
        match self.tag {
            ValueTag::Bool => Some(self.payload != 0),
            _ => None,
        }
    }

    const fn registered_parts(self) -> Option<(u8, u64)> {
        match self.tag {
            ValueTag::Registered => Some((self.aux[0], self.payload)),
            _ => None,
        }
    }

    fn native_value(self) -> Option<NativeValue> {
        match self.tag {
            ValueTag::Native => Some(NativeValue::decode(self.payload, self.aux[0])),
            _ => None,
        }
    }

    const fn inline_string_len(self) -> Option<usize> {
        let raw = self.tag as u8;
        if raw <= ValueTag::SmallString15 as u8 {
            Some(raw as usize)
        } else {
            None
        }
    }

    fn inline_string_ref(self) -> Option<InlineString> {
        let length = self.inline_string_len()?;
        let mut bytes = [0; 15];
        bytes[..8].copy_from_slice(&self.payload.to_ne_bytes());
        bytes[8..].copy_from_slice(&self.aux);
        Some(InlineString::from_parts(bytes, length))
    }
}

impl std::fmt::Debug for Raw {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(value) = self.inline_string_ref() {
            return formatter
                .debug_tuple("String")
                .field(&value.as_str())
                .finish();
        }
        match self.tag {
            ValueTag::Int => formatter
                .debug_tuple("Int")
                .field(&(self.payload as i64))
                .finish(),
            ValueTag::Float => formatter
                .debug_tuple("Float")
                .field(&f64::from_bits(self.payload))
                .finish(),
            ValueTag::Bool => formatter
                .debug_tuple("Bool")
                .field(&(self.payload != 0))
                .finish(),
            ValueTag::None => formatter.write_str("None"),
            ValueTag::Object => formatter
                .debug_tuple("Object")
                .field(&self.payload)
                .finish(),
            ValueTag::Native => formatter
                .debug_tuple("Native")
                .field(&self.native_value())
                .finish(),
            ValueTag::Registered => formatter
                .debug_struct("Registered")
                .field("kind", &self.aux[0])
                .field("payload", &self.payload)
                .finish(),
            _ => unreachable!("small strings returned above"),
        }
    }
}

const fn string_tag(length: usize) -> ValueTag {
    match length {
        0 => ValueTag::SmallString0,
        1 => ValueTag::SmallString1,
        2 => ValueTag::SmallString2,
        3 => ValueTag::SmallString3,
        4 => ValueTag::SmallString4,
        5 => ValueTag::SmallString5,
        6 => ValueTag::SmallString6,
        7 => ValueTag::SmallString7,
        8 => ValueTag::SmallString8,
        9 => ValueTag::SmallString9,
        10 => ValueTag::SmallString10,
        11 => ValueTag::SmallString11,
        12 => ValueTag::SmallString12,
        13 => ValueTag::SmallString13,
        14 => ValueTag::SmallString14,
        15 => ValueTag::SmallString15,
        _ => panic!("inline string length exceeds payload"),
    }
}

/// A reference stored in a heap object or a VM root.
///
/// Slots are the collector's unit of work: it traces every slot reachable from the roots. A slot
/// is made from a pinned [`Value`] and read back as one with [`Heap::value`](super::Heap::value);
/// it cannot be copied, so the only references the collector cannot see are the pinned values
/// themselves. Equality is identity: two slots are equal when they name the same object or
/// immediate.
#[derive(PartialEq, Eq)]
#[repr(transparent)]
pub struct Ref(pub(super) Raw);

impl Ref {
    /// The immediate this slot holds, or `None` for an object reference.
    pub fn immediate(&self) -> Option<Value> {
        (!self.0.is_object()).then(|| Value::from_raw(self.0))
    }

    pub fn is_none(&self) -> bool {
        matches!(self.0.tag, ValueTag::None)
    }

    pub fn is_object(&self) -> bool {
        self.0.is_object()
    }

    /// A stored immediate, for root containers that hold Python values without a heap, such as
    /// the type registry's native method entries.
    ///
    /// # Panics
    ///
    /// Panics when `value` is an object.
    pub fn from_immediate(value: Value) -> Self {
        assert!(
            !value.is_object(),
            "only immediates can be stored without a value"
        );
        Self(value.raw())
    }

    /// Copy a stored reference for a root container's `Clone` implementation, or to move it
    /// between roots. Never hold the result in a Rust local across an allocation: it is not
    /// pinned and the collector cannot see it there.
    pub(crate) fn dup(&self) -> Self {
        Self(self.0)
    }

    /// The stored form of an optional value.
    pub fn optional(value: Option<Value>) -> Option<Ref> {
        value.map(Ref::from)
    }

    /// The stored form of each of `values`, sized exactly: collecting straight from a
    /// `Vec<Value>` would reuse its allocation, so a list cut down to a few items could keep the
    /// capacity of the snapshot it came from, while the modeled size counts items.
    pub fn all(values: impl IntoIterator<Item = Value>) -> Vec<Ref> {
        let values = values.into_iter();
        let mut refs = Vec::with_capacity(values.size_hint().0);
        refs.extend(values.map(Ref::from));
        refs
    }
}

impl From<Value> for Ref {
    /// The stored form of a pinned value. Place it in a heap object or a root before the value's
    /// pin is released.
    #[inline(always)]
    fn from(value: Value) -> Self {
        Self(value.raw())
    }
}

impl std::fmt::Debug for Ref {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// A Python value as Rust code holds it: an immediate, or an object's id.
///
/// Objects never move, so the id is all a value needs. A value made from a stored reference or
/// an allocation is pinned on the heap's pin stack, which keeps its object alive through any
/// collection until the instruction or native loop that made it releases its pins. The
/// `PhantomData` makes values neither `Send` nor `Sync`, so they cannot be kept in interpreter
/// state, which must be `Send`; store a [`Ref`] there instead.
#[derive(Clone, Copy)]
pub struct Value {
    raw: Raw,
    pinned: PhantomData<*const ()>,
}

// Immediate constructors keep the enum-variant spelling (`Value::Int(3)`, `Value::None`) that
// the interpreter and native modules read naturally.
impl Value {
    #[allow(non_upper_case_globals)]
    pub const None: Value = Value::from_raw(Raw::NONE);

    #[allow(non_snake_case)]
    pub const fn Int(value: i64) -> Value {
        Value::from_raw(Raw::int(value))
    }

    #[allow(non_snake_case)]
    pub const fn Float(value: f64) -> Value {
        Value::from_raw(Raw::float(value))
    }

    #[allow(non_snake_case)]
    pub const fn Bool(value: bool) -> Value {
        Value::from_raw(Raw::bool(value))
    }

    #[allow(non_snake_case)]
    pub(in crate::python) fn Native(value: NativeValue) -> Value {
        Value::from_raw(Raw::native(value))
    }

    pub fn registered(kind: u8, payload: u64) -> Value {
        Value::from_raw(Raw::registered(kind, payload))
    }

    /// A string of at most fifteen bytes stored inline, or `None` when it needs a heap object.
    pub fn inline_string(value: &str) -> Option<Value> {
        Raw::inline_string(value).map(Value::from_raw)
    }

    pub(super) const fn from_raw(raw: Raw) -> Value {
        Value {
            raw,
            pinned: PhantomData,
        }
    }

    pub(super) const fn raw(self) -> Raw {
        self.raw
    }

    /// Whether this value names a heap object rather than an immediate.
    pub const fn is_object(self) -> bool {
        self.raw.is_object()
    }

    /// This value when it is an immediate, or `None` for an object.
    pub const fn immediate(self) -> Option<Value> {
        if self.raw.is_object() {
            None
        } else {
            Some(Value::from_raw(self.raw))
        }
    }

    /// Whether both values are the same immediate. Objects never compare equal here; compare
    /// them with [`Value::is`].
    pub fn same_immediate(self, other: Value) -> bool {
        !self.raw.is_object() && !other.raw.is_object() && self.raw == other.raw
    }

    /// Python's `is`: the same object or the same immediate.
    #[inline(always)]
    pub fn is(self, other: Value) -> bool {
        self.raw == other.raw
    }

    /// Whether this value and a stored reference name the same object or immediate.
    #[inline(always)]
    pub fn is_ref(self, slot: &Ref) -> bool {
        self.raw == slot.0
    }

    pub const fn is_none(self) -> bool {
        matches!(self.raw.tag, ValueTag::None)
    }

    pub const fn immediate_int(self) -> Option<i64> {
        self.raw.immediate_int()
    }

    pub const fn float_value(self) -> Option<f64> {
        self.raw.float_value()
    }

    pub const fn bool_value(self) -> Option<bool> {
        self.raw.bool_value()
    }

    pub const fn registered_parts(self) -> Option<(u8, u64)> {
        self.raw.registered_parts()
    }

    pub(in crate::python) fn native_value(self) -> Option<NativeValue> {
        self.raw.native_value()
    }

    pub const fn inline_string_len(self) -> Option<usize> {
        self.raw.inline_string_len()
    }

    pub fn inline_string_ref(self) -> Option<InlineString> {
        self.raw.inline_string_ref()
    }

    pub fn inline_string_value(self) -> Option<String> {
        Some(self.inline_string_ref()?.as_str().to_owned())
    }

    /// Lenient integer view used by shell-facing conversions: ints and bools, finite floats
    /// truncated, and short decimal strings.
    pub fn as_int(self) -> Option<i64> {
        if let Some(value) = self.immediate_int() {
            return Some(value);
        }
        if let Some(value) = self.float_value().filter(|value| value.is_finite()) {
            return Some(value as i64);
        }
        self.inline_string_value()?.parse().ok()
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.raw.fmt(formatter)
    }
}

const _: () = assert!(std::mem::size_of::<Raw>() == 16);
const _: () = assert!(std::mem::size_of::<Ref>() == 16);
const _: () = assert!(std::mem::size_of::<Option<Ref>>() == 16);
const _: () = assert!(std::mem::size_of::<Value>() == 16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_values_are_exactly_sixteen_bytes() {
        assert_eq!(std::mem::size_of::<Value>(), 16);
        assert_eq!(
            Value::inline_string("123456789012345")
                .unwrap()
                .inline_string_len(),
            Some(15)
        );
        assert!(Value::inline_string("1234567890123456").is_none());
    }

    #[test]
    fn immediates_compare_and_objects_do_not() {
        assert!(Value::Int(3).same_immediate(Value::Int(3)));
        assert!(!Value::Int(3).same_immediate(Value::Float(3.0)));
        let object = Value::from_raw(Raw::object(ObjectId::from_bits(7)));
        assert!(!object.same_immediate(object));
        assert!(object.is(object));
        assert!(object.immediate().is_none());
        assert_eq!(Value::Bool(true).immediate_int(), Some(1));
    }
}
