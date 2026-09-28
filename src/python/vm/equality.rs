//! Element equality for containers, as CPython's `PyObject_RichCompareBool` defines it.
//!
//! Lists, tuples, dicts and sets compare their elements, find keys and answer `in` with the
//! rule `x is y or x == y`, where `==` is the full rich-comparison protocol. A user class's
//! `__eq__`, or a native type's equality slot such as a NumPy dtype's, therefore holds inside
//! containers: `[Fraction(1, 2)] == [0.5]`, `(np.dtype("f8"),) == (np.float64,)`, and an equal
//! key finds a dict entry. Builtin containers compare element by element through the same rule.
//! Values whose types define no equality slot, such as numbers and strings, keep the structural
//! fast path in `protocol::equals`.
//!
//! Nesting is bounded, so comparing two distinct self-containing lists raises `RecursionError`
//! as it does in CPython, instead of recursing without limit.

use super::super::ast::ComparisonOperator;
use super::super::heap::{Object, ObjectId};
use super::super::protocol;
use super::{Slot, Value, Vm};

/// Nesting bound for builtin container comparison, matching the hash nesting bound.
const MAX_EQUALITY_DEPTH: usize = 256;

/// Builtin containers of matching kinds, which compare element by element.
enum ContainerPair {
    Sequences,
    Mappings,
    Sets,
}

/// How `values_equal` must compare a value.
enum EqualityKind {
    /// Equality is structural and cannot run user code.
    Plain,
    /// A builtin list, tuple, dict or set, or a namespace view, compared element by element.
    Container,
    /// A user class instance, which may define `__eq__`, or a value whose native type has an
    /// equality slot.
    Rich,
}

impl Vm<'_> {
    /// `left is right or left == right`, as containers compare elements.
    pub(super) fn values_equal(&mut self, left: &Value, right: &Value) -> Result<bool, String> {
        self.values_equal_at(left, right, 0)
    }

    fn values_equal_at(
        &mut self,
        left: &Value,
        right: &Value,
        depth: usize,
    ) -> Result<bool, String> {
        if protocol::identical(left, right) {
            return Ok(true);
        }
        match (self.equality_kind(left)?, self.equality_kind(right)?) {
            (EqualityKind::Rich, _) | (_, EqualityKind::Rich) => {
                self.compare_truth(ComparisonOperator::Equal, left, right)
            }
            (EqualityKind::Plain, EqualityKind::Plain) => {
                protocol::equals(&self.state.heap, left, right)
            }
            _ => self.builtin_equality_at(left, right, depth),
        }
    }

    /// `left == right` once neither operand's `__eq__` has answered: builtin containers of the
    /// same kind compare their elements, and anything else compares structurally.
    pub(super) fn builtin_equality(&mut self, left: &Value, right: &Value) -> Result<bool, String> {
        self.builtin_equality_at(left, right, 0)
    }

    fn builtin_equality_at(
        &mut self,
        left: &Value,
        right: &Value,
        depth: usize,
    ) -> Result<bool, String> {
        let (Some(left_id), Some(right_id)) = (left.object_id(), right.object_id()) else {
            return protocol::equals(&self.state.heap, left, right);
        };
        // A namespace view compares as the dict of its current bindings, so
        // `globals() == globals()` and `vars(a) == {"x": 1}` hold as they do in CPython.
        if let Object::NamespaceDict(target) = *self.state.heap.get(left_id)? {
            let left = self.namespace_snapshot_dict(target)?;
            return self.builtin_equality_at(&left, right, depth);
        }
        if let Object::NamespaceDict(target) = *self.state.heap.get(right_id)? {
            let right = self.namespace_snapshot_dict(target)?;
            return self.builtin_equality_at(left, &right, depth);
        }
        let pair = match (
            self.state.heap.get(left_id)?,
            self.state.heap.get(right_id)?,
        ) {
            (Object::List(_), Object::List(_)) | (Object::Tuple(_), Object::Tuple(_)) => {
                ContainerPair::Sequences
            }
            (
                Object::Dict(_) | Object::DefaultDict { .. },
                Object::Dict(_) | Object::DefaultDict { .. },
            ) => ContainerPair::Mappings,
            (Object::Set(_) | Object::FrozenSet(_), Object::Set(_) | Object::FrozenSet(_)) => {
                ContainerPair::Sets
            }
            _ => return protocol::equals(&self.state.heap, left, right),
        };
        if depth == MAX_EQUALITY_DEPTH {
            return Err(self.raise_exception(
                "RecursionError",
                "maximum recursion depth exceeded in comparison",
            ));
        }
        match pair {
            ContainerPair::Sequences => self.sequences_equal(left_id, right_id, depth),
            ContainerPair::Mappings => self.mappings_equal(left_id, right_id, depth),
            ContainerPair::Sets => self.sets_equal(left_id, right_id),
        }
    }

    fn equality_kind(&self, value: &Value) -> Result<EqualityKind, String> {
        if let Some(id) = value.object_id() {
            match self.state.heap.get(id)? {
                Object::Instance { .. } => return Ok(EqualityKind::Rich),
                Object::List(_)
                | Object::Tuple(_)
                | Object::Dict(_)
                | Object::DefaultDict { .. }
                | Object::Set(_)
                | Object::FrozenSet(_)
                | Object::NamespaceDict(_) => return Ok(EqualityKind::Container),
                _ => {}
            }
        }
        let type_id = self.type_id(value)?;
        Ok(if self.state.types.slot(type_id, Slot::Equal)?.is_some() {
            EqualityKind::Rich
        } else {
            EqualityKind::Plain
        })
    }

    /// Lists or tuples: equal lengths and pairwise-equal items. Items are read by index on each
    /// step because a user `__eq__` may mutate either list, as in CPython.
    fn sequences_equal(
        &mut self,
        left: ObjectId,
        right: ObjectId,
        depth: usize,
    ) -> Result<bool, String> {
        if self.sequence_len(left)? != self.sequence_len(right)? {
            return Ok(false);
        }
        let mut index = 0;
        while let (Some(left_item), Some(right_item)) = (
            self.sequence_item(left, index)?,
            self.sequence_item(right, index)?,
        ) {
            self.charge_cpu(1)?;
            if !self.values_equal_at(&left_item, &right_item, depth + 1)? {
                return Ok(false);
            }
            index += 1;
        }
        Ok(self.sequence_len(left)? == self.sequence_len(right)?)
    }

    /// `needle in sequence` for a list or tuple, reading each item as the scan reaches it.
    pub(super) fn sequence_contains(
        &mut self,
        id: ObjectId,
        needle: &Value,
    ) -> Result<bool, String> {
        let mut index = 0;
        while let Some(item) = self.sequence_item(id, index)? {
            self.charge_cpu(1)?;
            if self.values_equal(&item, needle)? {
                return Ok(true);
            }
            index += 1;
        }
        Ok(false)
    }

    fn sequence_item(&self, id: ObjectId, index: usize) -> Result<Option<Value>, String> {
        match self.state.heap.get(id)? {
            Object::List(items) | Object::Tuple(items) => Ok(items.get(index).copied()),
            _ => Err("sequence handle changed object kind".into()),
        }
    }

    fn sequence_len(&self, id: ObjectId) -> Result<usize, String> {
        match self.state.heap.get(id)? {
            Object::List(items) | Object::Tuple(items) => Ok(items.len()),
            _ => Err("sequence handle changed object kind".into()),
        }
    }

    /// Dicts: the same number of entries, and every key of `left` found in `right` with an equal
    /// value.
    fn mappings_equal(
        &mut self,
        left: ObjectId,
        right: ObjectId,
        depth: usize,
    ) -> Result<bool, String> {
        if self.mapping_len(left)? != self.mapping_len(right)? {
            return Ok(false);
        }
        let mut index = 0;
        while let Some((key, value)) = self.mapping_entry(left, index)? {
            let Some(position) = self.find_mapping_entry(right, &key)? else {
                return Ok(false);
            };
            let Some((_, other)) = self.mapping_entry(right, position)? else {
                return Ok(false);
            };
            if !self.values_equal_at(&value, &other, depth + 1)? {
                return Ok(false);
            }
            index += 1;
        }
        Ok(true)
    }

    fn mapping_entry(&self, id: ObjectId, index: usize) -> Result<Option<(Value, Value)>, String> {
        match self.state.heap.get(id)? {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                Ok(entries.get(index).copied())
            }
            _ => Err("dict handle changed object kind".into()),
        }
    }

    fn mapping_len(&self, id: ObjectId) -> Result<usize, String> {
        match self.state.heap.get(id)? {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => Ok(entries.len()),
            _ => Err("dict handle changed object kind".into()),
        }
    }

    /// Sets and frozensets: the same size, and every element of `left` found in `right`.
    fn sets_equal(&mut self, left: ObjectId, right: ObjectId) -> Result<bool, String> {
        let size = |vm: &Self, id| match vm.state.heap.get(id)? {
            Object::Set(values) | Object::FrozenSet(values) => Ok(values.len()),
            _ => Err(String::from("set handle changed object kind")),
        };
        if size(self, left)? != size(self, right)? {
            return Ok(false);
        }
        let mut index = 0;
        loop {
            let element = match self.state.heap.get(left)? {
                Object::Set(values) | Object::FrozenSet(values) => values.get(index).copied(),
                _ => return Err("set handle changed object kind".into()),
            };
            let Some(element) = element else {
                return Ok(true);
            };
            if self.find_set_entry(right, &element)?.is_none() {
                return Ok(false);
            }
            index += 1;
        }
    }
}
