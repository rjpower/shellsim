//! Process-wide index of the static native definitions that `Value` handles name.
//!
//! A `Value` is 16 bytes, so a native module, function, method, getter or value kind is stored
//! as a small index into a table rather than as a reference. A definition is interned the first
//! time a handle names it and is never removed, so an index stays valid for the life of the
//! process and across threads. Decoding is a bounds-checked lookup: a payload that did not come
//! from [`DefinitionTable::intern`] cannot reach arbitrary memory.

use std::collections::BTreeMap;
use std::sync::RwLock;

pub(super) struct DefinitionTable<T: 'static> {
    entries: RwLock<Entries<T>>,
}

struct Entries<T: 'static> {
    definitions: Vec<&'static T>,
    /// Index of each interned definition, keyed by its address.
    positions: BTreeMap<usize, u32>,
}

impl<T: 'static> DefinitionTable<T> {
    pub(super) const fn new() -> Self {
        Self {
            entries: RwLock::new(Entries {
                definitions: Vec::new(),
                positions: BTreeMap::new(),
            }),
        }
    }

    /// The stable index of `definition`, adding it on first use.
    pub(super) fn intern(&self, definition: &'static T) -> u32 {
        let address = std::ptr::from_ref(definition) as usize;
        if let Some(position) = self.read().positions.get(&address) {
            return *position;
        }
        let mut entries = self
            .entries
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(position) = entries.positions.get(&address) {
            return *position;
        }
        let position =
            u32::try_from(entries.definitions.len()).expect("static definitions fit in u32");
        entries.definitions.push(definition);
        entries.positions.insert(address, position);
        position
    }

    /// The definition interned at `index`, if any.
    pub(super) fn get(&self, index: u64) -> Option<&'static T> {
        let index = usize::try_from(index).ok()?;
        self.read().definitions.get(index).copied()
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Entries<T>> {
        // The table is only appended to, so a poisoned lock still holds consistent entries.
        self.entries
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static FIRST: u64 = 1;
    static SECOND: u64 = 2;

    #[test]
    fn interning_is_stable_and_lookup_is_bounds_checked() {
        let table = DefinitionTable::<u64>::new();
        let first = table.intern(&FIRST);
        let second = table.intern(&SECOND);
        assert_ne!(first, second);
        assert_eq!(table.intern(&FIRST), first);
        assert_eq!(table.get(u64::from(second)), Some(&SECOND));
        assert_eq!(table.get(99), None);
        assert_eq!(table.get(u64::MAX), None);
    }
}
