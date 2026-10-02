//! Insertion-ordered storage shared by Python mapping objects.
//!
//! Python dictionaries preserve insertion order, while hashing and key equality belong to the
//! runtime's object protocol. These types own ordering, mutation and a hash index; callers
//! supply each key's hash and test the candidates that share it for equality.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use super::Value;

/// A key's result under the runtime's `hash()` protocol. Callers compute it before every
/// insertion or lookup, so storage never needs the heap or guest code to place a key.
pub(super) type KeyHash = i64;

/// Positions of the live members that share one hash. Almost every bucket holds one member.
#[derive(Clone, Debug)]
enum Bucket {
    One(usize),
    Many(Vec<usize>),
}

impl Bucket {
    fn positions(&self) -> &[usize] {
        match self {
            Self::One(position) => std::slice::from_ref(position),
            Self::Many(positions) => positions,
        }
    }
}

/// Insertion-ordered slots where removal leaves a tombstone, so deleting one member is constant
/// time. Slots are compacted once tombstones outnumber live members, which keeps removal
/// amortized constant and storage within twice the live size. Positions stay valid until the
/// next removal.
///
/// Each slot keeps its member's hash, so the index can be rebuilt and two containers can be
/// compared without hashing again. Distinct members with equal hashes share a bucket and are
/// told apart by the caller's equality test, as in CPython.
#[derive(Clone, Debug)]
struct Slots<T> {
    slots: Vec<Option<(KeyHash, T)>>,
    /// No live member precedes this position, so taking the first member, as `set.pop` does,
    /// does not rescan leading tombstones. Trailing tombstones are popped eagerly.
    first: usize,
    live: usize,
    index: HashMap<KeyHash, Bucket>,
}

impl<T> Default for Slots<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            first: 0,
            live: 0,
            index: HashMap::new(),
        }
    }
}

fn index_add(index: &mut HashMap<KeyHash, Bucket>, hash: KeyHash, position: usize) {
    match index.entry(hash) {
        Entry::Vacant(slot) => {
            slot.insert(Bucket::One(position));
        }
        Entry::Occupied(mut slot) => match slot.get_mut() {
            Bucket::One(first) => {
                let first = *first;
                slot.insert(Bucket::Many(vec![first, position]));
            }
            Bucket::Many(positions) => positions.push(position),
        },
    }
}

fn index_remove(index: &mut HashMap<KeyHash, Bucket>, hash: KeyHash, position: usize) {
    let Entry::Occupied(mut slot) = index.entry(hash) else {
        return;
    };
    match slot.get_mut() {
        Bucket::One(_) => {
            slot.remove();
        }
        Bucket::Many(positions) => {
            positions.retain(|&candidate| candidate != position);
            if let [remaining] = positions[..] {
                slot.insert(Bucket::One(remaining));
            }
        }
    }
}

impl<T> Slots<T> {
    fn push(&mut self, hash: KeyHash, member: T) {
        index_add(&mut self.index, hash, self.slots.len());
        self.slots.push(Some((hash, member)));
        self.live += 1;
    }

    /// Remove the live member at `position`, or return `None` when a guest `__eq__` already
    /// removed it after the lookup.
    fn take(&mut self, position: usize) -> Option<T> {
        let (hash, member) = self.slots.get_mut(position)?.take()?;
        index_remove(&mut self.index, hash, position);
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
                let (hash, _) = slot.as_ref().expect("compacted slots are live");
                index_add(&mut self.index, *hash, position);
            }
        }
        Some(member)
    }

    fn get(&self, position: usize) -> Option<&T> {
        self.slots
            .get(position)
            .and_then(Option::as_ref)
            .map(|(_, member)| member)
    }

    fn get_mut(&mut self, position: usize) -> Option<&mut T> {
        self.slots
            .get_mut(position)
            .and_then(Option::as_mut)
            .map(|(_, member)| member)
    }

    fn iter(&self) -> impl DoubleEndedIterator<Item = &T> + Clone {
        self.iter_hashed().map(|(_, member)| member)
    }

    fn iter_hashed(&self) -> impl DoubleEndedIterator<Item = (KeyHash, &T)> + Clone {
        self.slots[self.first..]
            .iter()
            .filter_map(Option::as_ref)
            .map(|(hash, member)| (*hash, member))
    }

    /// Positions of live members whose hash is `hash`.
    fn candidates(&self, hash: KeyHash) -> &[usize] {
        self.index.get(&hash).map_or(&[], Bucket::positions)
    }
}

/// Ordered key/value entries for `dict` and mapping-derived objects. The storage is boxed so
/// a dict costs one pointer inside its heap object, keeping every object slot small.
#[derive(Clone, Debug, Default)]
pub(super) struct OrderedMap {
    entries: Box<Slots<(Value, Value)>>,
}

impl OrderedMap {
    /// Append an entry whose key is not already present; `hash` is the key's hash.
    pub(super) fn push(&mut self, hash: KeyHash, entry: (Value, Value)) {
        self.entries.push(hash, entry);
    }

    /// Remove the live entry at `position`, invalidating previously returned positions.
    pub(super) fn remove(&mut self, position: usize) -> Option<(Value, Value)> {
        self.entries.take(position)
    }

    pub(super) fn set_value(&mut self, position: usize, value: Value) {
        if let Some(entry) = self.entries.get_mut(position) {
            entry.1 = value;
        }
    }

    /// The live entry at a position from [`Self::candidate_positions`].
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

    /// Entries with their key hashes, for copying into another mapping without rehashing.
    pub(super) fn iter_hashed(
        &self,
    ) -> impl DoubleEndedIterator<Item = (KeyHash, &(Value, Value))> + Clone {
        self.entries.iter_hashed()
    }

    pub(super) fn to_vec(&self) -> Vec<(Value, Value)> {
        self.iter().copied().collect()
    }

    /// Live positions whose key has hash `hash`; the caller tests each key for equality.
    pub(super) fn candidate_positions(&self, hash: KeyHash) -> &[usize] {
        self.entries.candidates(hash)
    }
}

/// Insertion-ordered members of a `set` or `frozenset`, indexed and boxed like [`OrderedMap`].
#[derive(Clone, Debug, Default)]
pub(super) struct OrderedSet {
    values: Box<Slots<Value>>,
}

impl OrderedSet {
    /// Append a member that is not already present; `hash` is the member's hash.
    pub(super) fn push(&mut self, hash: KeyHash, value: Value) {
        self.values.push(hash, value);
    }

    /// Remove the live member at `position`, invalidating previously returned positions.
    pub(super) fn remove(&mut self, position: usize) -> Option<Value> {
        self.values.take(position)
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

    /// Members with their hashes, for copying into another set without rehashing.
    pub(super) fn iter_hashed(&self) -> impl DoubleEndedIterator<Item = (KeyHash, &Value)> + Clone {
        self.values.iter_hashed()
    }

    pub(super) fn to_vec(&self) -> Vec<Value> {
        self.iter().copied().collect()
    }

    /// Live positions of members with hash `hash`; the caller tests each for equality.
    pub(super) fn candidate_positions(&self, hash: KeyHash) -> &[usize] {
        self.values.candidates(hash)
    }
}

type Owned<T> = std::iter::Map<
    std::iter::Flatten<std::vec::IntoIter<Option<(KeyHash, T)>>>,
    fn((KeyHash, T)) -> T,
>;
type Borrowed<'a, T> = std::iter::Map<
    std::iter::Flatten<std::slice::Iter<'a, Option<(KeyHash, T)>>>,
    fn(&'a (KeyHash, T)) -> &'a T,
>;

fn member<T>((_, member): (KeyHash, T)) -> T {
    member
}

fn member_ref<T>((_, member): &(KeyHash, T)) -> &T {
    member
}

impl IntoIterator for OrderedSet {
    type Item = Value;
    type IntoIter = Owned<Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.slots.into_iter().flatten().map(member)
    }
}

impl IntoIterator for OrderedMap {
    type Item = (Value, Value);
    type IntoIter = Owned<(Value, Value)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.slots.into_iter().flatten().map(member)
    }
}

impl<'a> IntoIterator for &'a OrderedMap {
    type Item = &'a (Value, Value);
    type IntoIter = Borrowed<'a, (Value, Value)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.slots.iter().flatten().map(member_ref)
    }
}

impl<'a> IntoIterator for &'a OrderedSet {
    type Item = &'a Value;
    type IntoIter = Borrowed<'a, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.slots.iter().flatten().map(member_ref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Integer members use their own value as the hash, as `hash(n) == n` for small `n`.
    fn set_of(values: impl IntoIterator<Item = i64>) -> OrderedSet {
        let mut set = OrderedSet::default();
        for value in values {
            set.push(value, Value::Int(value));
        }
        set
    }

    fn position(set: &OrderedSet, value: i64) -> Option<usize> {
        set.candidate_positions(value)
            .iter()
            .copied()
            .find(|&position| set.get(position) == Some(&Value::Int(value)))
    }

    #[test]
    fn removal_keeps_order_and_compaction_keeps_lookup() {
        let mut set = set_of(0..20);
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
        assert!((3..20)
            .step_by(2)
            .all(|value| set.candidate_positions(value).len() == 1));
        assert_eq!(position(&set, 2), None);
    }

    #[test]
    fn members_with_equal_hashes_share_a_bucket() {
        let mut set = OrderedSet::default();
        for value in 0..3 {
            set.push(7, Value::Int(value));
        }
        assert_eq!(set.candidate_positions(7), [0, 1, 2]);
        set.remove(1);
        assert_eq!(set.candidate_positions(7), [0, 2]);
        set.remove(0);
        assert_eq!(set.candidate_positions(7), [2]);
        assert!(set.candidate_positions(8).is_empty());
    }

    #[test]
    fn popping_the_first_member_skips_leading_tombstones_once() {
        let mut map = OrderedMap::default();
        for key in 0..6 {
            map.push(key, (Value::Int(key), Value::None));
        }
        map.remove(0);
        map.remove(1);
        assert_eq!(map.entries.first, 2);
        assert_eq!(map.iter().next().map(|entry| entry.0), Some(Value::Int(2)));
        map.remove(5);
        assert_eq!(map.entries.slots.len(), 5, "trailing tombstones are popped");
        map.push(9, (Value::Int(9), Value::None));
        let keys = map.iter().map(|entry| entry.0).collect::<Vec<_>>();
        assert_eq!(keys, [2, 3, 4, 9].map(Value::Int));
    }
}
