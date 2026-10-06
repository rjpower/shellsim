//! Symbol-keyed namespaces.
//!
//! [`Namespace`] is the one name-to-value table behind module globals, the dynamic names of
//! scopes, class and function attributes, and instance dictionaries. It keeps insertion order,
//! as Python's `__dict__` does, and finds a name by comparing 32-bit symbols: a scan while the
//! table is small, and an index hashed from the symbol itself once it grows. Removing a name
//! shifts the later entries, which is linear but rare next to lookups and stores.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use super::Ref;
use crate::python::symbols::{SymbolId, Symbols};

/// Entries a namespace holds before it builds an index; a scan of this many symbols is about as
/// fast as one hash probe.
const INDEX_THRESHOLD: usize = 8;

/// An insertion-ordered map from symbol to stored reference.
#[derive(Debug, Default)]
pub struct Namespace {
    entries: Vec<(SymbolId, Ref)>,
    /// Position of each symbol in `entries`, present once the namespace outgrows a scan. Boxed
    /// so that the many namespaces that never grow that large stay three words smaller; an
    /// inline map measured slower on call-heavy code.
    #[allow(clippy::box_collection)]
    index: Option<Box<HashMap<SymbolId, u32, BuildHasherDefault<SymbolHasher>>>>,
}

impl Namespace {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Where `symbol` sits in insertion order. A small namespace is scanned from its newest
    /// name back: a module's own dunders come first and its loop variables last, and the
    /// later names are the ones a loop reads.
    #[inline]
    pub fn position(&self, symbol: SymbolId) -> Option<usize> {
        match &self.index {
            Some(index) => index.get(&symbol).map(|&position| position as usize),
            None => self.entries.iter().rposition(|(key, _)| *key == symbol),
        }
    }

    #[inline]
    pub fn get(&self, symbol: SymbolId) -> Option<&Ref> {
        self.position(symbol)
            .map(|position| &self.entries[position].1)
    }

    #[inline]
    pub fn get_mut(&mut self, symbol: SymbolId) -> Option<&mut Ref> {
        self.position(symbol)
            .map(|position| &mut self.entries[position].1)
    }

    /// The binding of a name native code holds as a string. A name the interpreter never
    /// interned is bound in no namespace, so the lookup misses without a scan.
    pub fn get_name(&self, symbols: &Symbols, name: &str) -> Option<&Ref> {
        symbols.id(name).and_then(|symbol| self.get(symbol))
    }

    pub fn contains(&self, symbol: SymbolId) -> bool {
        self.position(symbol).is_some()
    }

    /// Bind `symbol`, keeping its place when it is already bound; returns the previous value.
    pub fn insert(&mut self, symbol: SymbolId, value: Ref) -> Option<Ref> {
        if let Some(position) = self.position(symbol) {
            return Some(std::mem::replace(&mut self.entries[position].1, value));
        }
        let position = self.entries.len();
        self.entries.push((symbol, value));
        match &mut self.index {
            Some(index) => {
                index.insert(symbol, position_u32(position));
            }
            None if self.entries.len() > INDEX_THRESHOLD => self.rebuild_index(),
            None => {}
        }
        None
    }

    /// Unbind `symbol`, keeping the order of the names after it.
    pub fn remove(&mut self, symbol: SymbolId) -> Option<Ref> {
        let position = self.position(symbol)?;
        let (_, value) = self.entries.remove(position);
        if self.index.is_some() {
            self.rebuild_index();
        }
        Some(value)
    }

    /// Every binding in insertion order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (SymbolId, &Ref)> {
        self.entries.iter().map(|(symbol, value)| (*symbol, value))
    }

    pub fn symbols(&self) -> impl ExactSizeIterator<Item = SymbolId> + '_ {
        self.entries.iter().map(|(symbol, _)| *symbol)
    }

    pub fn refs(&self) -> impl Iterator<Item = &Ref> {
        self.entries.iter().map(|(_, value)| value)
    }

    /// A copy whose references are duplicated for a heap or root snapshot.
    pub fn dup(&self) -> Self {
        Self {
            entries: self
                .entries
                .iter()
                .map(|(symbol, value)| (*symbol, value.dup()))
                .collect(),
            index: self.index.clone(),
        }
    }

    fn rebuild_index(&mut self) {
        self.index = (self.entries.len() > INDEX_THRESHOLD).then(|| {
            Box::new(
                self.entries
                    .iter()
                    .enumerate()
                    .map(|(position, (symbol, _))| (*symbol, position_u32(position)))
                    .collect(),
            )
        });
    }
}

impl FromIterator<(SymbolId, Ref)> for Namespace {
    fn from_iter<I: IntoIterator<Item = (SymbolId, Ref)>>(entries: I) -> Self {
        let mut namespace = Self::default();
        for (symbol, value) in entries {
            namespace.insert(symbol, value);
        }
        namespace
    }
}

fn position_u32(position: usize) -> u32 {
    // A namespace's entries are charged to the guest, so the memory limit bounds them far below
    // four billion.
    u32::try_from(position).expect("namespace positions fit in u32")
}

/// Hashes a symbol by multiplying it with a 64-bit odd constant: symbols are dense small
/// integers, which the multiplication spreads over both the bucket and the tag bits.
#[derive(Default)]
struct SymbolHasher(u64);

impl Hasher for SymbolHasher {
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(*byte)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }

    fn write_u32(&mut self, value: u32) {
        self.0 = u64::from(value).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::python::heap::Value;

    fn symbol(index: usize) -> SymbolId {
        SymbolId::from_index(index).unwrap()
    }

    fn entries(namespace: &Namespace) -> Vec<(usize, i64)> {
        namespace
            .iter()
            .map(|(symbol, value)| (symbol.index(), value.immediate().unwrap().as_int().unwrap()))
            .collect()
    }

    #[test]
    fn keeps_insertion_order_across_rebinding_and_removal_at_every_size() {
        for size in [3, INDEX_THRESHOLD + 5] {
            let mut namespace = Namespace::default();
            for index in (0..size).rev() {
                namespace.insert(symbol(index), Ref::from_immediate(Value::Int(index as i64)));
            }
            namespace.insert(symbol(1), Ref::from_immediate(Value::Int(-1)));
            assert!(namespace.remove(symbol(2)).is_some());
            assert!(namespace.remove(symbol(2)).is_none());
            let mut expected = (0..size)
                .rev()
                .filter(|index| *index != 2)
                .map(|index| (index, if index == 1 { -1 } else { index as i64 }))
                .collect::<Vec<_>>();
            assert_eq!(entries(&namespace), expected);
            namespace.insert(symbol(2), Ref::from_immediate(Value::Int(2)));
            expected.push((2, 2));
            assert_eq!(entries(&namespace), expected);
            for (index, _) in &expected {
                assert!(namespace.contains(symbol(*index)));
            }
            assert!(!namespace.contains(symbol(size)));
        }
    }
}
