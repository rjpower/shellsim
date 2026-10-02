//! Insertion-ordered storage shared by Python mapping objects.
//!
//! Python dictionaries preserve insertion order, while key equality belongs to the runtime's
//! object protocol. This type owns ordering and mutation; callers supply protocol-aware lookup.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

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
/// Hash index from scalar keys to slot positions, shared by ordered maps and sets.
///
/// Most scalar keys own one slot, so the common case stores a single position. Distinct members
/// whose keys share a bucket, such as two NaN objects, spill into `overflow`. Object kinds
/// without a stable hash model stay unindexed and are compared by every lookup.
#[derive(Clone, Debug, Default)]
struct PositionIndex {
    index: HashMap<IndexedKey, usize>,
    overflow: HashMap<IndexedKey, Vec<usize>>,
    /// Ascending positions of unindexed members.
    unindexed: Vec<usize>,
}

impl PositionIndex {
    fn add(&mut self, key: &Value, position: usize) {
        let Some(key) = IndexedKey::from_value(key) else {
            self.unindexed.push(position);
            return;
        };
        match self.index.entry(key) {
            Entry::Vacant(slot) => {
                slot.insert(position);
            }
            Entry::Occupied(_) => self.overflow.entry(key).or_default().push(position),
        }
    }

    fn remove(&mut self, key: &Value, position: usize) {
        let Some(key) = IndexedKey::from_value(key) else {
            if let Ok(found) = self.unindexed.binary_search(&position) {
                self.unindexed.remove(found);
            }
            return;
        };
        if self.index.get(&key) == Some(&position) {
            match self.overflow.get_mut(&key) {
                Some(spilled) => {
                    let next = spilled.remove(0);
                    if spilled.is_empty() {
                        self.overflow.remove(&key);
                    }
                    self.index.insert(key, next);
                }
                None => {
                    self.index.remove(&key);
                }
            }
        } else if let Some(spilled) = self.overflow.get_mut(&key) {
            spilled.retain(|&spilled| spilled != position);
            if spilled.is_empty() {
                self.overflow.remove(&key);
            }
        }
    }

    /// Positions that can hold a member equal to `key`. Unindexable keys fall back to every
    /// position below `length`, including tombstones, which callers skip.
    fn candidates(&self, key: &Value, length: usize) -> impl Iterator<Item = usize> + '_ {
        let indexed = IndexedKey::from_value(key);
        let first = indexed
            .as_ref()
            .and_then(|key| self.index.get(key))
            .copied();
        let spilled = indexed
            .as_ref()
            .and_then(|key| self.overflow.get(key))
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let unindexed = if indexed.is_some() {
            self.unindexed.as_slice()
        } else {
            &[]
        };
        let fallback = indexed.is_none().then_some(0..length).into_iter().flatten();
        first
            .into_iter()
            .chain(spilled.iter().copied())
            .chain(unindexed.iter().copied())
            .chain(fallback)
    }

    fn clear(&mut self) {
        self.index.clear();
        self.overflow.clear();
        self.unindexed.clear();
    }
}

/// Insertion-ordered slots where removal leaves a tombstone, so deleting one member is constant
/// time. Slots are compacted once tombstones outnumber live members, which keeps removal
/// amortized constant and storage within twice the live size. Positions stay valid until the
/// next removal.
#[derive(Clone, Debug)]
struct Slots<T> {
    slots: Vec<Option<T>>,
    /// No live member precedes this position, so taking the first member, as `set.pop` does,
    /// does not rescan leading tombstones. Trailing tombstones are popped eagerly.
    first: usize,
    live: usize,
    index: PositionIndex,
}

impl<T> Default for Slots<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            first: 0,
            live: 0,
            index: PositionIndex::default(),
        }
    }
}

impl<T> Slots<T> {
    fn push(&mut self, key: &Value, member: T) {
        self.index.add(key, self.slots.len());
        self.slots.push(Some(member));
        self.live += 1;
    }

    /// Remove the live member at `position`, or return `None` when a guest `__eq__` already
    /// removed it after the lookup.
    fn take(&mut self, position: usize, key_of: impl Fn(&T) -> &Value) -> Option<T> {
        let member = self.slots.get_mut(position)?.take()?;
        self.index.remove(key_of(&member), position);
        self.live -= 1;
        while self.slots.last().is_some_and(Option::is_none) {
            self.slots.pop();
        }
        while self.slots.get(self.first).is_some_and(Option::is_none) {
            self.first += 1;
        }
        self.first = self.first.min(self.slots.len());
        if self.slots.len() > 8 && self.slots.len() - self.live > self.live {
            self.first = 0;
            self.slots.retain(Option::is_some);
            self.index.clear();
            for (position, slot) in self.slots.iter().enumerate() {
                let member = slot.as_ref().expect("compacted slots are live");
                self.index.add(key_of(member), position);
            }
        }
        Some(member)
    }

    fn get(&self, position: usize) -> Option<&T> {
        self.slots.get(position).and_then(Option::as_ref)
    }

    fn iter(&self) -> impl DoubleEndedIterator<Item = &T> + Clone {
        self.slots[self.first..].iter().filter_map(Option::as_ref)
    }

    fn candidates(&self, key: &Value) -> impl Iterator<Item = usize> + '_ {
        self.index
            .candidates(key, self.slots.len())
            .filter(|&position| self.slots[position].is_some())
    }
}

/// Ordered key/value entries for `dict` and mapping-derived objects.
#[derive(Clone, Debug, Default)]
pub(super) struct OrderedMap {
    entries: Slots<(Value, Value)>,
}

impl OrderedMap {
    pub(super) fn push(&mut self, entry: (Value, Value)) {
        self.entries.push(&entry.0, entry);
    }

    /// Remove the live entry at `position`, invalidating previously returned positions.
    pub(super) fn remove(&mut self, position: usize) -> Option<(Value, Value)> {
        self.entries.take(position, |entry| &entry.0)
    }

    pub(super) fn set_value(&mut self, position: usize, value: Value) {
        if let Some(Some(entry)) = self.entries.slots.get_mut(position) {
            entry.1 = value;
        }
    }

    /// The live entry at a position from [`Self::candidate_positions`] or [`Self::positions`].
    pub(super) fn get(&self, position: usize) -> Option<&(Value, Value)> {
        self.entries.get(position)
    }

    pub(super) fn len(&self) -> usize {
        self.entries.live
    }

    pub(super) fn is_empty(&self) -> bool {
        self.entries.live == 0
    }

    pub(super) fn iter(&self) -> impl DoubleEndedIterator<Item = &(Value, Value)> + Clone {
        self.entries.iter()
    }

    pub(super) fn to_vec(&self) -> Vec<(Value, Value)> {
        self.iter().copied().collect()
    }

    /// Yield only live positions that can compare equal under the runtime's scalar protocol.
    /// Unindexed object kinds remain a correctness fallback until they have a stable hash model.
    pub(super) fn candidate_positions(&self, key: &Value) -> impl Iterator<Item = usize> + '_ {
        self.entries.candidates(key)
    }
}

/// Insertion-ordered members of a `set` or `frozenset`, indexed like [`OrderedMap`] keys.
#[derive(Clone, Debug, Default)]
pub(super) struct OrderedSet {
    values: Slots<Value>,
}

impl OrderedSet {
    pub(super) fn push(&mut self, value: Value) {
        self.values.push(&value, value);
    }

    /// Remove the live member at `position`, invalidating previously returned positions.
    pub(super) fn remove(&mut self, position: usize) -> Option<Value> {
        self.values.take(position, |value| value)
    }

    pub(super) fn get(&self, position: usize) -> Option<&Value> {
        self.values.get(position)
    }

    pub(super) fn len(&self) -> usize {
        self.values.live
    }

    pub(super) fn is_empty(&self) -> bool {
        self.values.live == 0
    }

    pub(super) fn iter(&self) -> impl DoubleEndedIterator<Item = &Value> + Clone {
        self.values.iter()
    }

    pub(super) fn to_vec(&self) -> Vec<Value> {
        self.iter().copied().collect()
    }

    /// Live positions of members that can compare equal to `value`.
    pub(super) fn candidate_positions(&self, value: &Value) -> impl Iterator<Item = usize> + '_ {
        self.values.candidates(value)
    }
}

impl From<Vec<Value>> for OrderedSet {
    fn from(values: Vec<Value>) -> Self {
        values.into_iter().collect()
    }
}

impl FromIterator<Value> for OrderedSet {
    fn from_iter<T: IntoIterator<Item = Value>>(iter: T) -> Self {
        let mut set = Self::default();
        for value in iter {
            set.push(value);
        }
        set
    }
}

impl IntoIterator for OrderedSet {
    type Item = Value;
    type IntoIter = std::iter::Flatten<std::vec::IntoIter<Option<Value>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.slots.into_iter().flatten()
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

impl IntoIterator for OrderedMap {
    type Item = (Value, Value);
    type IntoIter = std::iter::Flatten<std::vec::IntoIter<Option<(Value, Value)>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.slots.into_iter().flatten()
    }
}

impl<'a> IntoIterator for &'a OrderedMap {
    type Item = &'a (Value, Value);
    type IntoIter = std::iter::Flatten<std::slice::Iter<'a, Option<(Value, Value)>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.slots.iter().flatten()
    }
}

impl<'a> IntoIterator for &'a OrderedSet {
    type Item = &'a Value;
    type IntoIter = std::iter::Flatten<std::slice::Iter<'a, Option<Value>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.slots.iter().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(set: &OrderedSet, value: i64) -> Option<usize> {
        set.candidate_positions(&Value::Int(value))
            .find(|&position| set.get(position) == Some(&Value::Int(value)))
    }

    #[test]
    fn removal_keeps_order_and_compaction_keeps_lookup() {
        let mut set: OrderedSet = (0..20).map(Value::Int).collect();
        for value in (0..20).step_by(2) {
            let found = position(&set, value).expect("member is indexed");
            assert_eq!(set.remove(found), Some(Value::Int(value)));
            assert_eq!(set.remove(found), None, "a removed slot stays empty");
        }
        // Removing one more than half compacts the slots; positions are renumbered.
        let found = position(&set, 1).unwrap();
        set.remove(found);
        assert_eq!(set.values.slots.len(), set.len());
        let odd = (3..20).step_by(2).map(Value::Int).collect::<Vec<_>>();
        assert_eq!(set.to_vec(), odd);
        assert!(odd
            .iter()
            .all(|value| set.candidate_positions(value).count() == 1));
        assert_eq!(position(&set, 2), None);
    }

    #[test]
    fn popping_the_first_member_skips_leading_tombstones_once() {
        let mut map: OrderedMap = (0..6).map(|key| (Value::Int(key), Value::None)).collect();
        map.remove(0);
        map.remove(1);
        assert_eq!(map.entries.first, 2);
        assert_eq!(map.iter().next().map(|entry| entry.0), Some(Value::Int(2)));
        map.remove(5);
        assert_eq!(map.entries.slots.len(), 5, "trailing tombstones are popped");
        map.push((Value::Int(9), Value::None));
        let keys = map.iter().map(|entry| entry.0).collect::<Vec<_>>();
        assert_eq!(keys, [2, 3, 4, 9].map(Value::Int));
    }
}
