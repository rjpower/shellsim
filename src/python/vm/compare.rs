//! Rich comparison: the one entry point for `==`, `!=`, `<`, `<=`, `>` and `>=`.
//!
//! [`Vm::rich_compare`] does what CPython's `PyObject_RichCompare` does. It consults the operand
//! types' comparison slots, reflected operand first when the right operand's type is a strict
//! subclass that overrides the reflected slot, and falls back to identity for equality or to
//! `TypeError` for ordering. Builtin ints, floats and strings take a direct path first because
//! they are the bulk of all comparisons and have no slots to consult. Builtin containers compare
//! element by element through this same entry point, so a user `__eq__` or `__lt__` holds inside
//! a tuple or list.
//!
//! [`Vm::compare_truth`] is the truth-valued form that sorting, `min`, `max`, membership and
//! container equality use. Its slot path runs in a child handle scope, so a loop of comparisons
//! releases the handles each one creates instead of accumulating them.

use std::cmp::Ordering;

use super::super::ast::ComparisonOperator;
use super::super::heap::{Heap, Object};
use super::{number, string, BuiltinType, Slot, Value, Vm};

/// Nesting bound for ordering comparisons of builtin sequences, matching the equality bound.
const MAX_COMPARE_DEPTH: usize = 256;

/// A protocol slot and the dunder name it is looked up under on a user class.
type NamedSlot = (Slot, &'static str);

impl ComparisonOperator {
    /// The operand slots that implement this operator: the left operand's own slot and the
    /// slot the reflected (right) operand answers with. `None` for identity and membership.
    fn slots(self) -> Option<(NamedSlot, NamedSlot)> {
        Some(match self {
            Self::Equal => ((Slot::Equal, "__eq__"), (Slot::Equal, "__eq__")),
            Self::NotEqual => ((Slot::NotEqual, "__ne__"), (Slot::NotEqual, "__ne__")),
            Self::Less => ((Slot::LessThan, "__lt__"), (Slot::GreaterThan, "__gt__")),
            Self::LessEqual => ((Slot::LessEqual, "__le__"), (Slot::GreaterEqual, "__ge__")),
            Self::Greater => ((Slot::GreaterThan, "__gt__"), (Slot::LessThan, "__lt__")),
            Self::GreaterEqual => ((Slot::GreaterEqual, "__ge__"), (Slot::LessEqual, "__le__")),
            Self::In | Self::NotIn | Self::Is | Self::IsNot => return None,
        })
    }

    /// The operator as Python spells it in error messages.
    pub(super) fn symbol(self) -> &'static str {
        match self {
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::In => "in",
            Self::NotIn => "not in",
            Self::Is => "is",
            Self::IsNot => "is not",
        }
    }

    /// Whether an ordering of two values satisfies this operator.
    fn accepts(self, ordering: Ordering) -> bool {
        match self {
            Self::Equal => ordering.is_eq(),
            Self::NotEqual => ordering.is_ne(),
            Self::Less => ordering.is_lt(),
            Self::LessEqual => ordering.is_le(),
            Self::Greater => ordering.is_gt(),
            Self::GreaterEqual => ordering.is_ge(),
            Self::In | Self::NotIn | Self::Is | Self::IsNot => false,
        }
    }

    /// The comparison slot this operator invokes on a builtin container.
    pub(super) fn from_slot(slot: Slot) -> Option<Self> {
        Some(match slot {
            Slot::Equal => Self::Equal,
            Slot::NotEqual => Self::NotEqual,
            Slot::LessThan => Self::Less,
            Slot::LessEqual => Self::LessEqual,
            Slot::GreaterThan => Self::Greater,
            Slot::GreaterEqual => Self::GreaterEqual,
            _ => return None,
        })
    }
}

impl<'s> Vm<'s> {
    /// `left <operator> right` for the six rich comparisons, as a Python value. A user slot may
    /// return any object (a NumPy array, for one), so the result is not necessarily a bool.
    pub(super) fn rich_compare(
        &mut self,
        operator: ComparisonOperator,
        left: Value<'s>,
        right: Value<'s>,
    ) -> Result<Value<'s>, String> {
        if let Some(result) = self.fast_compare(operator, left, right)? {
            return Ok(Value::Bool(result));
        }
        if let Some(value) = self.slot_compare(operator, left, right)? {
            return Ok(value);
        }
        self.default_compare(operator, left, right).map(Value::Bool)
    }

    /// The truth of `left <operator> right`, as CPython's `PyObject_RichCompareBool` computes
    /// it for `min`, `max`, sorting and container comparison, except that identity is not a
    /// shortcut: `values_equal` adds that where containers need it.
    pub(super) fn compare_truth(
        &mut self,
        operator: ComparisonOperator,
        left: &Value<'s>,
        right: &Value<'s>,
    ) -> Result<bool, String> {
        if let Some(result) = self.fast_compare(operator, *left, *right)? {
            return Ok(result);
        }
        let mut vm = self.scope();
        match vm.slot_compare(operator, *left, *right)? {
            Some(value) => vm.truth_value(&value),
            None => vm.default_compare(operator, *left, *right),
        }
    }

    /// Order `left` and `right` with `<` alone, as CPython's sorting, heap and bisection
    /// helpers do: `left < right` is `Less`, `right < left` is `Greater`, and anything else,
    /// such as a NaN, is `Equal`, so the earlier value stays in place.
    pub(super) fn sort_order(
        &mut self,
        left: &Value<'s>,
        right: &Value<'s>,
    ) -> Result<Ordering, String> {
        if self.compare_truth(ComparisonOperator::Less, left, right)? {
            Ok(Ordering::Less)
        } else if self.compare_truth(ComparisonOperator::Less, right, left)? {
            Ok(Ordering::Greater)
        } else {
            Ok(Ordering::Equal)
        }
    }

    /// Builtin ints, floats and exact strings, which have no slots to consult.
    fn fast_compare(
        &mut self,
        operator: ComparisonOperator,
        left: Value<'s>,
        right: Value<'s>,
    ) -> Result<Option<bool>, String> {
        if let Some(result) = number::exact_integer_comparison(operator, left, right)
            .or_else(|| number::exact_float_comparison(operator, left, right))
        {
            return Ok(Some(result));
        }
        self.exact_string_comparison(operator, left, right)
    }

    /// Compare two builtin strings directly. Code point order is UTF-8 byte order, so the
    /// comparison is a byte scan, charged like the other string scans.
    fn exact_string_comparison(
        &mut self,
        operator: ComparisonOperator,
        left: Value<'s>,
        right: Value<'s>,
    ) -> Result<Option<bool>, String> {
        // Only exact `str` values qualify: a `str` subclass holds the same payload but may define
        // its own comparison.
        let is_plain = |heap: &Heap, value: Value<'s>| -> Result<bool, String> {
            Ok(value.inline_string_ref().is_some()
                || (value.is_object() && heap.type_id(value)? == BuiltinType::String.id()))
        };
        let ordering = {
            let heap = &self.state.heap;
            if !is_plain(heap, left)? || !is_plain(heap, right)? {
                return Ok(None);
            }
            let (Some(left), Some(right)) = (
                string::string_ref(heap, left)?,
                string::string_ref(heap, right)?,
            ) else {
                return Ok(None);
            };
            left.as_str().cmp(right.as_str())
        };
        if operator.slots().is_none() {
            return Ok(None);
        }
        self.charge_scan_pair(&left, &right)?;
        Ok(Some(operator.accepts(ordering)))
    }

    /// The operand types' comparison slots, in CPython's order: the right operand first when
    /// its type is a strict subclass of the left's that overrides the reflected slot, then the
    /// left, then the right. `!=` without a `__ne__` answers with the negated `__eq__`.
    fn slot_compare(
        &mut self,
        operator: ComparisonOperator,
        left: Value<'s>,
        right: Value<'s>,
    ) -> Result<Option<Value<'s>>, String> {
        let Some(((left_slot, left_name), (right_slot, right_name))) = operator.slots() else {
            return Ok(None);
        };
        let left_type = self.type_id(&left)?;
        let right_type = self.type_id(&right)?;
        let right_first = right_type != left_type
            && self.state.types.is_subclass(right_type, left_type)?
            && self
                .state
                .types
                .local_slot(right_type, right_slot)?
                .is_some();
        let mut result = None;
        if right_first {
            result = self.invoke_operator_slot(&right, right_slot, right_name, vec![left])?;
        }
        if result.is_none() {
            result = self.invoke_operator_slot(&left, left_slot, left_name, vec![right])?;
        }
        if result.is_none() && !right_first {
            result = self.invoke_operator_slot(&right, right_slot, right_name, vec![left])?;
        }
        if result.is_none() && operator == ComparisonOperator::NotEqual {
            let mut equality =
                self.invoke_operator_slot(&left, Slot::Equal, "__eq__", vec![right])?;
            if equality.is_none() {
                equality = self.invoke_operator_slot(&right, Slot::Equal, "__eq__", vec![left])?;
            }
            if let Some(value) = equality {
                result = Some(Value::Bool(!self.truth_value(&value)?));
            }
        }
        Ok(result)
    }

    /// The comparison once no slot has answered, as `object` defines it: a value is equal only
    /// to itself, and nothing is ordered.
    fn default_compare(
        &mut self,
        operator: ComparisonOperator,
        left: Value<'s>,
        right: Value<'s>,
    ) -> Result<bool, String> {
        match operator {
            ComparisonOperator::Equal => Ok(self.identical(left, right)),
            ComparisonOperator::NotEqual => Ok(!self.identical(left, right)),
            _ => Err(self.raise_unorderable(operator.symbol(), &left, &right)),
        }
    }

    /// Raise CPython's `TypeError` for an ordering comparison between unrelated types.
    pub(super) fn raise_unorderable(
        &mut self,
        symbol: &str,
        left: &Value<'s>,
        right: &Value<'s>,
    ) -> String {
        let message = match (self.type_name_of(left), self.type_name_of(right)) {
            (Ok(left), Ok(right)) => {
                format!("'{symbol}' not supported between instances of '{left}' and '{right}'")
            }
            (Err(error), _) | (_, Err(error)) => return error,
        };
        self.raise_exception("TypeError", message)
    }

    /// The comparison slot of the builtin containers: lists and tuples order element by
    /// element, and lists, tuples, dicts and sets compare equal element by element. `None` when
    /// the operands are not containers of one kind, which the caller treats as `NotImplemented`.
    pub(super) fn container_compare(
        &mut self,
        operator: ComparisonOperator,
        left: Value<'s>,
        right: Value<'s>,
    ) -> Result<Option<bool>, String> {
        if !left.is_object() || !right.is_object() {
            return Ok(None);
        }
        let kinds = match (self.get(left)?, self.get(right)?) {
            (Object::List(_), Object::List(_)) | (Object::Tuple(_), Object::Tuple(_)) => {
                ContainerPair::Sequences
            }
            (
                Object::Dict(_)
                | Object::DefaultDict { .. }
                | Object::NamespaceDict(_)
                | Object::MappingProxy(_),
                Object::Dict(_)
                | Object::DefaultDict { .. }
                | Object::NamespaceDict(_)
                | Object::MappingProxy(_),
            ) => ContainerPair::Mappings,
            (Object::Set(_) | Object::FrozenSet(_), Object::Set(_) | Object::FrozenSet(_)) => {
                ContainerPair::Sets
            }
            _ => return Ok(None),
        };
        let result = match operator {
            ComparisonOperator::Equal => self.builtin_equality(&left, &right)?,
            ComparisonOperator::NotEqual => !self.builtin_equality(&left, &right)?,
            ComparisonOperator::Less
            | ComparisonOperator::LessEqual
            | ComparisonOperator::Greater
            | ComparisonOperator::GreaterEqual => match kinds {
                ContainerPair::Sequences => return self.sequence_order(operator, left, right),
                ContainerPair::Mappings | ContainerPair::Sets => return Ok(None),
            },
            ComparisonOperator::In
            | ComparisonOperator::NotIn
            | ComparisonOperator::Is
            | ComparisonOperator::IsNot => return Ok(None),
        };
        Ok(Some(result))
    }

    /// Order two lists or two tuples as CPython does: find the first index whose items differ
    /// under `==`, and let that pair decide with `operator`; when one sequence is a prefix of
    /// the other, the lengths decide. Items are re-read on each step because a user `__eq__`
    /// may mutate either list.
    fn sequence_order(
        &mut self,
        operator: ComparisonOperator,
        left: Value<'s>,
        right: Value<'s>,
    ) -> Result<Option<bool>, String> {
        self.nested_comparison(|vm| {
            let mut index = 0;
            loop {
                let (Some(left_item), Some(right_item)) = (
                    vm.sequence_item(left, index)?,
                    vm.sequence_item(right, index)?,
                ) else {
                    let ordering = vm.sequence_len(left)?.cmp(&vm.sequence_len(right)?);
                    return Ok(Some(operator.accepts(ordering)));
                };
                vm.charge_cpu(1)?;
                if !vm.values_equal(&left_item, &right_item)? {
                    return vm
                        .compare_truth(operator, &left_item, &right_item)
                        .map(Some);
                }
                index += 1;
            }
        })
    }

    /// Run one level of a container comparison, raising `RecursionError` instead of exhausting
    /// the host stack when containers nest past [`MAX_COMPARE_DEPTH`].
    pub(super) fn nested_comparison<T>(
        &mut self,
        body: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        if self.execution.compare_depth >= MAX_COMPARE_DEPTH {
            return Err(self.raise_exception(
                "RecursionError",
                "maximum recursion depth exceeded in comparison",
            ));
        }
        self.execution.compare_depth += 1;
        let result = body(self);
        self.execution.compare_depth -= 1;
        result
    }
}

/// Builtin containers of matching kinds, which compare element by element.
enum ContainerPair {
    Sequences,
    Mappings,
    Sets,
}
