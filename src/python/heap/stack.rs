//! The VM operand stack: a root container that moves values between slots without pinning.
//!
//! Loads and stores between the stack and scope locals are the interpreter's hottest path, so
//! the stack stores references directly and only pins a value when it leaves into Rust code
//! (`pop`, `peek`). The collector visits it through [`Roots`].

use super::{Heap, Ref, Roots, Value};

#[derive(Debug, Default)]
pub struct ValueStack {
    values: Vec<Ref>,
}

impl ValueStack {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
    }

    #[inline(always)]
    pub fn push(&mut self, value: Value) {
        self.values.push(Ref::from(value));
    }

    /// Push a stored reference without pinning it.
    #[inline(always)]
    pub fn push_ref(&mut self, slot: &Ref) {
        self.values.push(slot.dup());
    }

    #[inline(always)]
    pub fn pop(&mut self, heap: &Heap) -> Option<Value> {
        let slot = self.values.pop()?;
        Some(heap.value(&slot))
    }

    /// Pop the top value as a stored reference. The result must go straight into a root.
    #[inline(always)]
    pub fn pop_ref(&mut self) -> Option<Ref> {
        self.values.pop()
    }

    /// The stored reference `depth` entries below the top (0 is the top), without pinning it.
    #[inline(always)]
    pub fn top(&self, depth: usize) -> Option<&Ref> {
        self.values
            .len()
            .checked_sub(depth + 1)
            .and_then(|index| self.values.get(index))
    }

    /// The value `depth` entries below the top (0 is the top).
    #[inline(always)]
    pub fn peek(&self, heap: &Heap, depth: usize) -> Option<Value> {
        let slot = self.top(depth)?;
        Some(heap.value(slot))
    }

    pub fn set(&mut self, index: usize, value: Value) -> bool {
        match self.values.get_mut(index) {
            Some(slot) => {
                *slot = Ref::from(value);
                true
            }
            None => false,
        }
    }

    /// Pop `count` values, returned bottom-first, or `None` when the stack is shallower.
    pub fn pop_many(&mut self, heap: &Heap, count: usize) -> Option<Vec<Value>> {
        let base = self.values.len().checked_sub(count)?;
        Some(self.split_off(heap, base))
    }

    /// Move every stored reference at or above `base` out, bottom-first, without pinning.
    /// Each must go straight into a root.
    pub fn drain_refs(&mut self, base: usize) -> std::vec::Drain<'_, Ref> {
        let base = base.min(self.values.len());
        self.values.drain(base..)
    }

    /// Remove and return every value at or above `base`, bottom-first.
    pub fn split_off(&mut self, heap: &Heap, base: usize) -> Vec<Value> {
        let base = base.min(self.values.len());
        self.values
            .drain(base..)
            .map(|slot| heap.value(&slot))
            .collect()
    }

    /// Remove the value `depth` entries below the top, closing the gap.
    pub fn remove(&mut self, depth: usize) -> Option<Ref> {
        let index = self.values.len().checked_sub(depth + 1)?;
        Some(self.values.remove(index))
    }

    pub fn swap(&mut self, first: usize, second: usize) {
        self.values.swap(first, second);
    }

    /// Append stored references, for restoring a generator's saved stack.
    pub fn extend_refs<'a>(&mut self, slots: impl IntoIterator<Item = &'a Ref>) {
        self.values.extend(slots.into_iter().map(Ref::dup));
    }
}

impl Clone for ValueStack {
    fn clone(&self) -> Self {
        Self {
            values: self.values.iter().map(Ref::dup).collect(),
        }
    }
}

impl Roots for ValueStack {
    fn visit_refs(&self, visitor: &mut dyn FnMut(&Ref)) {
        for slot in &self.values {
            visitor(slot);
        }
    }
}
