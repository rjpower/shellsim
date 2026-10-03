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
use super::super::heap::Object;
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
    /// A builtin list, tuple, dict or set, a namespace view or a mapping proxy, compared
    /// element by element.
    Container,
    /// A user class instance, which may define `__eq__`, or a value whose native type has an
    /// equality slot.
    Rich,
}

impl<'s> Vm<'s> {
    /// Charge the scan a structural comparison of two strings, byte strings or big integers
    /// performs. Their equality and ordering run at memory speed over the shorter operand, which
    /// is unbounded work the comparison itself does not otherwise account for.
    pub(super) fn charge_scan_pair(
        &mut self,
        left: &Value<'s>,
        right: &Value<'s>,
    ) -> Result<(), String> {
        let heap = self.heap();
        let scanned = match (
            protocol::string_ref(heap, *left)?,
            protocol::string_ref(heap, *right)?,
        ) {
            (Some(left), Some(right)) => left.byte_len().min(right.byte_len()),
            _ => match (
                protocol::bytes_ref(heap, *left)?,
                protocol::bytes_ref(heap, *right)?,
            ) {
                (Some(left), Some(right)) => left.len().min(right.len()),
                _ => match (
                    protocol::bigint_value(heap, *left),
                    protocol::bigint_value(heap, *right),
                ) {
                    (Some(left), Some(right)) => {
                        usize::try_from(left.bits().min(right.bits()) / 8).unwrap_or(usize::MAX)
                    }
                    _ => return Ok(()),
                },
            },
        };
        self.charge_cpu(super::objects::scan_cost(scanned))
    }

    /// `left is right or left == right`, as containers compare elements.
    pub(super) fn values_equal(
        &mut self,
        left: &Value<'s>,
        right: &Value<'s>,
    ) -> Result<bool, String> {
        self.values_equal_at(left, right, 0)
    }

    fn values_equal_at(
        &mut self,
        left: &Value<'s>,
        right: &Value<'s>,
        depth: usize,
    ) -> Result<bool, String> {
        if self.identical(*left, *right) {
            return Ok(true);
        }
        match (self.equality_kind(left)?, self.equality_kind(right)?) {
            (EqualityKind::Rich, _) | (_, EqualityKind::Rich) => {
                self.compare_truth(ComparisonOperator::Equal, left, right)
            }
            (EqualityKind::Plain, EqualityKind::Plain) => {
                protocol::equals(self.heap(), *left, *right)
            }
            _ => self.builtin_equality_at(left, right, depth),
        }
    }

    /// `left == right` once neither operand's `__eq__` has answered: builtin containers of the
    /// same kind compare their elements, and anything else compares structurally.
    pub(super) fn builtin_equality(
        &mut self,
        left: &Value<'s>,
        right: &Value<'s>,
    ) -> Result<bool, String> {
        self.builtin_equality_at(left, right, 0)
    }

    fn builtin_equality_at(
        &mut self,
        left: &Value<'s>,
        right: &Value<'s>,
        depth: usize,
    ) -> Result<bool, String> {
        if !left.is_object() || !right.is_object() {
            return protocol::equals(self.heap(), *left, *right);
        }
        self.charge_scan_pair(left, right)?;
        // A namespace view or mapping proxy compares as the dict of its current entries, so
        // `globals() == globals()`, `vars(a) == {"x": 1}` and `A.__dict__ == A.__dict__` hold
        // as they do in CPython.
        if let Some(left) = self.mapping_snapshot(*left)? {
            return self.builtin_equality_at(&left, right, depth);
        }
        if let Some(right) = self.mapping_snapshot(*right)? {
            return self.builtin_equality_at(left, &right, depth);
        }
        let pair = match (self.get(*left)?, self.get(*right)?) {
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
            _ => return protocol::equals(self.heap(), *left, *right),
        };
        if depth == MAX_EQUALITY_DEPTH {
            return Err(self.raise_exception(
                "RecursionError",
                "maximum recursion depth exceeded in comparison",
            ));
        }
        match pair {
            ContainerPair::Sequences => self.sequences_equal(*left, *right, depth),
            ContainerPair::Mappings => self.mappings_equal(*left, *right, depth),
            ContainerPair::Sets => self.sets_equal(*left, *right),
        }
    }

    /// A dict of the current entries when `value` is a namespace view or mapping proxy, whose
    /// entries live outside the object.
    fn mapping_snapshot(&mut self, value: Value<'s>) -> Result<Option<Value<'s>>, String> {
        let entries = match self.get(value)? {
            Object::NamespaceDict(target) => {
                let target = self.namespace_handle(target);
                self.namespace_items(target)?
            }
            Object::MappingProxy(target) => {
                let target = self.proxy_handle(target);
                self.proxy_items(target)?
            }
            _ => return Ok(None),
        };
        self.allocate_dict(entries).map(Some)
    }

    fn equality_kind(&self, value: &Value<'s>) -> Result<EqualityKind, String> {
        if value.is_object() {
            match self.get(*value)? {
                Object::Instance { .. } => return Ok(EqualityKind::Rich),
                Object::List(_)
                | Object::Tuple(_)
                | Object::Dict(_)
                | Object::DefaultDict { .. }
                | Object::Set(_)
                | Object::FrozenSet(_)
                | Object::NamespaceDict(_)
                | Object::MappingProxy(_) => return Ok(EqualityKind::Container),
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
        left: Value<'s>,
        right: Value<'s>,
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
        id: Value<'s>,
        needle: &Value<'s>,
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

    fn sequence_item(&self, id: Value<'s>, index: usize) -> Result<Option<Value<'s>>, String> {
        match self.get(id)? {
            Object::List(items) | Object::Tuple(items) => {
                Ok(self.handle_optional(items.get(index)))
            }
            _ => Err("sequence handle changed object kind".into()),
        }
    }

    fn sequence_len(&self, id: Value<'s>) -> Result<usize, String> {
        match self.get(id)? {
            Object::List(items) | Object::Tuple(items) => Ok(items.len()),
            _ => Err("sequence handle changed object kind".into()),
        }
    }

    /// Dicts: the same number of entries, and every key of `left` found in `right` with an equal
    /// value.
    fn mappings_equal(
        &mut self,
        left: Value<'s>,
        right: Value<'s>,
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

    fn mapping_entry(
        &self,
        id: Value<'s>,
        index: usize,
    ) -> Result<Option<(Value<'s>, Value<'s>)>, String> {
        match self.get(id)? {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => Ok(entries
                .get(index)
                .map(|(key, value)| (self.handle(key), self.handle(value)))),
            _ => Err("dict handle changed object kind".into()),
        }
    }

    fn mapping_len(&self, id: Value<'s>) -> Result<usize, String> {
        match self.get(id)? {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => Ok(entries.len()),
            _ => Err("dict handle changed object kind".into()),
        }
    }

    /// Sets and frozensets: the same size, and every element of `left` found in `right`.
    fn sets_equal(&mut self, left: Value<'s>, right: Value<'s>) -> Result<bool, String> {
        let size = |vm: &Self, id: Value<'s>| match vm.get(id)? {
            Object::Set(values) | Object::FrozenSet(values) => Ok(values.len()),
            _ => Err(String::from("set handle changed object kind")),
        };
        if size(self, left)? != size(self, right)? {
            return Ok(false);
        }
        let mut index = 0;
        loop {
            let element = match self.get(left)? {
                Object::Set(values) | Object::FrozenSet(values) => {
                    self.handle_optional(values.get(index))
                }
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
