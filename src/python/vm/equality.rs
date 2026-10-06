//! Element equality for containers, as CPython's `PyObject_RichCompareBool` defines it.
//!
//! Lists, tuples, dicts and sets compare their elements, find keys and answer `in` with the
//! rule `x is y or x == y`, where `==` is the full rich comparison through `rich_compare`. A
//! user class's `__eq__`, or a native type's equality slot such as a NumPy dtype's, therefore
//! holds inside containers: `[Fraction(1, 2)] == [0.5]`, `(np.dtype("f8"),) == (np.float64,)`,
//! and an equal key finds a dict entry.
//!
//! Nesting is bounded through the VM's comparison depth, so comparing two distinct
//! self-containing lists raises `RecursionError` as it does in CPython.

use super::super::ast::ComparisonOperator;
use super::super::heap::Object;
use super::{number, string, Value, Vm};

/// Builtin containers of matching kinds, which compare element by element.
enum ContainerPair {
    Sequences,
    Mappings,
    Sets,
}

impl<'s> Vm<'s> {
    /// Charge the scan a structural comparison of two strings, byte strings or big integers
    /// performs. Their equality and ordering run at memory speed over the shorter operand, which
    /// is unbounded work the comparison itself does not otherwise account for.
    pub(super) fn charge_scan_pair(&mut self, left: &Value, right: &Value) -> Result<(), String> {
        let heap = self.heap();
        let scanned = match (
            string::string_ref(heap, *left)?,
            string::string_ref(heap, *right)?,
        ) {
            (Some(left), Some(right)) => left.byte_len().min(right.byte_len()),
            _ => match (
                string::bytes_ref(heap, *left)?,
                string::bytes_ref(heap, *right)?,
            ) {
                (Some(left), Some(right)) => left.len().min(right.len()),
                _ => match (
                    number::bigint_value(heap, *left),
                    number::bigint_value(heap, *right),
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
    pub(super) fn values_equal(&mut self, left: &Value, right: &Value) -> Result<bool, String> {
        if self.identical(*left, *right) {
            return Ok(true);
        }
        self.compare_truth(ComparisonOperator::Equal, left, right)
    }

    /// `left == right` for two builtin containers of one kind, element by element. Containers
    /// of different kinds, such as a list and a tuple, are unequal.
    pub(super) fn builtin_equality(&mut self, left: &Value, right: &Value) -> Result<bool, String> {
        // A namespace view or mapping proxy compares as the dict of its current entries, so
        // `globals() == globals()`, `vars(a) == {"x": 1}` and `A.__dict__ == A.__dict__` hold
        // as they do in CPython.
        if let Some(left) = self.mapping_snapshot(*left)? {
            return self.builtin_equality(&left, right);
        }
        if let Some(right) = self.mapping_snapshot(*right)? {
            return self.builtin_equality(left, &right);
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
            _ => return Ok(false),
        };
        let (left, right) = (*left, *right);
        self.nested_comparison(|vm| match pair {
            ContainerPair::Sequences => vm.sequences_equal(left, right),
            ContainerPair::Mappings => vm.mappings_equal(left, right),
            ContainerPair::Sets => vm.sets_equal(left, right),
        })
    }

    /// A dict of the current entries when `value` is a namespace view or mapping proxy, whose
    /// entries live outside the object.
    fn mapping_snapshot(&mut self, value: Value) -> Result<Option<Value>, String> {
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

    /// Lists or tuples: equal lengths and pairwise-equal items. Items are read by index on each
    /// step because a user `__eq__` may mutate either list, as in CPython.
    fn sequences_equal(&mut self, left: Value, right: Value) -> Result<bool, String> {
        if self.sequence_len(left)? != self.sequence_len(right)? {
            return Ok(false);
        }
        let mut index = 0;
        while let (Some(left_item), Some(right_item)) = (
            self.sequence_item(left, index)?,
            self.sequence_item(right, index)?,
        ) {
            self.charge_cpu(1)?;
            if !self.values_equal(&left_item, &right_item)? {
                return Ok(false);
            }
            index += 1;
        }
        Ok(self.sequence_len(left)? == self.sequence_len(right)?)
    }

    /// `needle in sequence` for a list or tuple, reading each item as the scan reaches it.
    pub(super) fn sequence_contains(&mut self, id: Value, needle: &Value) -> Result<bool, String> {
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

    pub(super) fn sequence_item(&self, id: Value, index: usize) -> Result<Option<Value>, String> {
        match self.get(id)? {
            Object::List(items) | Object::Tuple(items) => Ok(self.value_optional(items.get(index))),
            _ => Err("sequence handle changed object kind".into()),
        }
    }

    pub(super) fn sequence_len(&self, id: Value) -> Result<usize, String> {
        match self.get(id)? {
            Object::List(items) | Object::Tuple(items) => Ok(items.len()),
            _ => Err("sequence handle changed object kind".into()),
        }
    }

    /// Dicts: the same number of entries, and every key of `left` found in `right` with an equal
    /// value.
    fn mappings_equal(&mut self, left: Value, right: Value) -> Result<bool, String> {
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
            if !self.values_equal(&value, &other)? {
                return Ok(false);
            }
            index += 1;
        }
        Ok(true)
    }

    fn mapping_entry(&self, id: Value, index: usize) -> Result<Option<(Value, Value)>, String> {
        match self.get(id)? {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => Ok(entries
                .get(index)
                .map(|(key, value)| (self.value(key), self.value(value)))),
            _ => Err("dict handle changed object kind".into()),
        }
    }

    fn mapping_len(&self, id: Value) -> Result<usize, String> {
        match self.get(id)? {
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => Ok(entries.len()),
            _ => Err("dict handle changed object kind".into()),
        }
    }

    /// Sets and frozensets: the same size, and every element of `left` found in `right`.
    fn sets_equal(&mut self, left: Value, right: Value) -> Result<bool, String> {
        let size = |vm: &Self, id: Value| match vm.get(id)? {
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
                    self.value_optional(values.get(index))
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
