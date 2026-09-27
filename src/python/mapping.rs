//! Insertion-ordered storage shared by Python mapping objects.
//!
//! Python dictionaries preserve insertion order, while key equality belongs to the runtime's
//! object protocol. This type owns ordering and mutation; callers supply protocol-aware lookup.

use std::collections::HashMap;
use std::ops::Deref;

use super::native::KindNumber;
use super::{Value, ValueTag};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum IndexedKey {
    Integer(i64),
    Float(u64),
    InlineString(Value),
    None,
    Native(Value),
    Registered(Value),
}

impl IndexedKey {
    fn from_value(value: &Value) -> Option<Self> {
        if let Some(value) = value.immediate_int() {
            return Some(Self::Integer(value));
        }
        if let Some(value) = value.float_value() {
            return Some(Self::float(value));
        }
        if value.inline_string_len().is_some() {
            return Some(Self::InlineString(*value));
        }
        match value.tag() {
            ValueTag::None => Some(Self::None),
            ValueTag::Native => Some(Self::Native(*value)),
            ValueTag::Registered => Self::registered(*value),
            ValueTag::Object
            | ValueTag::Int
            | ValueTag::Float
            | ValueTag::Bool
            | ValueTag::SmallString0
            | ValueTag::SmallString1
            | ValueTag::SmallString2
            | ValueTag::SmallString3
            | ValueTag::SmallString4
            | ValueTag::SmallString5
            | ValueTag::SmallString6
            | ValueTag::SmallString7
            | ValueTag::SmallString8
            | ValueTag::SmallString9
            | ValueTag::SmallString10
            | ValueTag::SmallString11
            | ValueTag::SmallString12
            | ValueTag::SmallString13
            | ValueTag::SmallString14
            | ValueTag::SmallString15 => None,
        }
    }

    /// Integral floats share the bucket of the equal integer, as `hash(2.0) == hash(2)`.
    fn float(value: f64) -> Self {
        let integer = value as i64;
        if value.is_finite() && value.fract() == 0.0 && integer as f64 == value {
            return Self::Integer(integer);
        }
        Self::Float(value.to_bits())
    }

    /// A registered number, such as a NumPy scalar, shares the bucket of the Python number it
    /// equals. Complex numbers with an imaginary part and integers beyond `i64` stay unindexed,
    /// like their builtin counterparts.
    fn registered(value: Value) -> Option<Self> {
        let (index, payload) = value.registered_parts()?;
        let Some(kind) = super::stdlib::value_kind(index) else {
            return Some(Self::Registered(value));
        };
        let Some(number) = kind.numeric.and_then(|numeric| numeric(kind, [payload, 0])) else {
            return Some(Self::Registered(value));
        };
        match number {
            KindNumber::Bool(value) => Some(Self::Integer(i64::from(value))),
            KindNumber::Int(value) => Some(Self::Integer(value)),
            KindNumber::UInt(value) => i64::try_from(value).ok().map(Self::Integer),
            KindNumber::Float(value) => Some(Self::float(value)),
            KindNumber::Complex(real, imag) => (imag == 0.0).then(|| Self::float(real)),
        }
    }
}

/// Ordered key/value entries for `dict` and mapping-derived objects.
#[derive(Clone, Debug, Default)]
pub(super) struct OrderedMap {
    entries: Vec<(Value, Value)>,
    index: HashMap<IndexedKey, Vec<usize>>,
    unindexed: Vec<usize>,
}

impl OrderedMap {
    pub(super) fn push(&mut self, entry: (Value, Value)) {
        let position = self.entries.len();
        if let Some(key) = IndexedKey::from_value(&entry.0) {
            self.index.entry(key).or_default().push(position);
        } else {
            self.unindexed.push(position);
        }
        self.entries.push(entry);
    }

    pub(super) fn remove(&mut self, index: usize) -> (Value, Value) {
        let entry = self.entries.remove(index);
        self.rebuild_index();
        entry
    }

    pub(super) fn set_value(&mut self, index: usize, value: Value) {
        self.entries[index].1 = value;
    }

    /// Yield only positions that can compare equal under the runtime's current scalar protocol.
    /// Unindexed object kinds remain a correctness fallback until they have a stable hash model.
    pub(super) fn candidate_positions(&self, key: &Value) -> impl Iterator<Item = usize> + '_ {
        let indexed = IndexedKey::from_value(key);
        let bucket = indexed
            .as_ref()
            .and_then(|key| self.index.get(key))
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let unindexed = if indexed.is_some() {
            self.unindexed.as_slice()
        } else {
            &[]
        };
        let fallback = indexed
            .is_none()
            .then_some(0..self.entries.len())
            .into_iter()
            .flatten();
        bucket
            .iter()
            .copied()
            .chain(unindexed.iter().copied())
            .chain(fallback)
    }

    fn rebuild_index(&mut self) {
        self.index.clear();
        self.unindexed.clear();
        for (position, (key, _)) in self.entries.iter().enumerate() {
            if let Some(key) = IndexedKey::from_value(key) {
                self.index.entry(key).or_default().push(position);
            } else {
                self.unindexed.push(position);
            }
        }
    }
}

impl From<Vec<(Value, Value)>> for OrderedMap {
    fn from(entries: Vec<(Value, Value)>) -> Self {
        let mut mapping = Self::default();
        mapping.extend(entries);
        mapping
    }
}

impl FromIterator<(Value, Value)> for OrderedMap {
    fn from_iter<T: IntoIterator<Item = (Value, Value)>>(iter: T) -> Self {
        let mut mapping = Self::default();
        mapping.extend(iter);
        mapping
    }
}

impl Extend<(Value, Value)> for OrderedMap {
    fn extend<T: IntoIterator<Item = (Value, Value)>>(&mut self, iter: T) {
        for entry in iter {
            self.push(entry);
        }
    }
}

impl Deref for OrderedMap {
    type Target = [(Value, Value)];

    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}

impl IntoIterator for OrderedMap {
    type Item = (Value, Value);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<'a> IntoIterator for &'a OrderedMap {
    type Item = &'a (Value, Value);
    type IntoIter = std::slice::Iter<'a, (Value, Value)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}
