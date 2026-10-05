//! VM adapters for unary, binary, comparison, construction, and formatting operations.

use super::super::heap::{Builder, MODELED_SET_MEMBER_BYTES, MODELED_VALUE_BYTES};
use super::super::native::KindNumber;
use super::format::{format_complex, format_float, format_integer, format_text, FormatError};
use super::{
    number, protocol, string, BigInt, BinaryOperator, BuiltinType, ComparisonOperator, DisplayKind,
    HashedMembers, NativeValue, Object, SequenceKind, Slot, ToPrimitive, UnaryOperator, Value, Vm,
};

/// The new contents of a builtin container updated in place by an augmented assignment.
enum Replacement<'s> {
    List(Vec<Value<'s>>),
    Set(HashedMembers<'s>),
    ByteArray(Vec<u8>),
}

impl<'s> Vm<'s> {
    pub(super) fn unary(&mut self, operator: UnaryOperator) -> Result<(), String> {
        // An exact number is rewritten in place on the stack; everything else leaves it as a
        // handle for the slot protocol.
        let value = self.peek(0)?;
        if let Some(result) = number::exact_unary(self, operator, value)
            .map_err(|error| self.record_native_error(error))?
        {
            let slot = self.stack.len() - 1;
            self.execution.stack.set(&self.state.heap, slot, result);
            return Ok(());
        }
        let value = self.pop()?;
        if operator == UnaryOperator::Not {
            let value = Value::Bool(!self.truth_value(&value)?);
            self.push(value);
            return Ok(());
        };
        let (slot, name, symbol) = match operator {
            UnaryOperator::Positive => (Slot::Positive, "__pos__", "+"),
            UnaryOperator::Negative => (Slot::Negative, "__neg__", "-"),
            UnaryOperator::Invert => (Slot::Invert, "__invert__", "~"),
            UnaryOperator::Not => unreachable!("handled above"),
        };
        let Some(result) = self.invoke_slot(&value, slot, name, Vec::new())? else {
            let message = format!(
                "bad operand type for unary {symbol}: '{}'",
                self.type_name_of(&value)?
            );
            return Err(self.raise_exception("TypeError", message));
        };
        self.push(result);
        Ok(())
    }

    pub(super) fn build_sequence(
        &mut self,
        count: usize,
        kind: SequenceKind,
    ) -> Result<(), String> {
        let values = self.take(count)?;
        let value = self.alloc_with(|builder| match kind {
            SequenceKind::List => Object::List(builder.refs(values)),
            SequenceKind::Tuple => Object::Tuple(builder.refs(values)),
        })?;
        self.push(value);
        Ok(())
    }

    pub(super) fn build_dict(&mut self, unpacked: &[bool]) -> Result<(), String> {
        let value_count = unpacked
            .iter()
            .try_fold(0usize, |count, unpacked| {
                count.checked_add(if *unpacked { 1 } else { 2 })
            })
            .ok_or("dictionary is too large")?;
        let values = self.take(value_count)?;
        let mut values = values.into_iter();
        let mut entries: Vec<(Value<'s>, Value<'s>)> = Vec::with_capacity(unpacked.len());
        for unpacked in unpacked {
            let additions = if *unpacked {
                let mapping = values.next().expect("dictionary stack contract");
                match self.mapping_items(mapping)? {
                    Some(entries) => entries,
                    None => {
                        let message =
                            format!("'{}' object is not a mapping", self.type_name_of(&mapping)?);
                        return Err(self.raise_exception("TypeError", message));
                    }
                }
            } else {
                vec![(
                    values.next().expect("dictionary key stack contract"),
                    values.next().expect("dictionary value stack contract"),
                )]
            };
            entries.extend(additions);
        }
        // A repeated key keeps its first position and takes the last value.
        let entries = self.ordered_map(entries)?;
        let value = self.alloc_with(|builder| Object::Dict(entries.into_map(builder)))?;
        self.push(value);
        Ok(())
    }

    /// Build a display whose `true` operands are iterables expanded in place, as in
    /// `[first, *rest]`.
    pub(super) fn build_unpacked(
        &mut self,
        kind: DisplayKind,
        starred: &[bool],
    ) -> Result<(), String> {
        let operands = self.take(starred.len())?;
        let mut values = Vec::with_capacity(operands.len());
        for (operand, expanded) in operands.into_iter().zip(starred) {
            if *expanded {
                for value in self.iterable_values(&operand)? {
                    self.push_materialized(&mut values, value)?;
                }
            } else {
                self.push_materialized(&mut values, operand)?;
            }
        }
        let value = match kind {
            DisplayKind::List => self.alloc_with(|builder| Object::List(builder.refs(values)))?,
            DisplayKind::Tuple => self.alloc_with(|builder| Object::Tuple(builder.refs(values)))?,
            DisplayKind::Set => return self.push_set(values),
        };
        self.push(value);
        Ok(())
    }

    pub(super) fn build_set(&mut self, count: usize) -> Result<(), String> {
        let candidates = self.take(count)?;
        self.push_set(candidates)
    }

    /// Push a set of the distinct `candidates`, metering each membership comparison as the `set`
    /// constructor does.
    fn push_set(&mut self, candidates: Vec<Value<'s>>) -> Result<(), String> {
        let members = self.distinct_members(candidates)?;
        let value = self.alloc_with(|builder| Object::Set(members.into_set(builder)))?;
        self.push(value);
        Ok(())
    }

    /// `COMPARE`: pop two operands and push `left <operator> right`. Membership and identity
    /// are their own protocols; the six rich comparisons go through [`Self::rich_compare`].
    pub(super) fn compare(&mut self, operator: ComparisonOperator) -> Result<(), String> {
        let right = self.pop()?;
        let left = self.pop()?;
        let result = match operator {
            ComparisonOperator::Is => Value::Bool(self.identical(left, right)),
            ComparisonOperator::IsNot => Value::Bool(!self.identical(left, right)),
            ComparisonOperator::In | ComparisonOperator::NotIn => {
                let contained =
                    match self.invoke_slot(&right, Slot::Contains, "__contains__", vec![left])? {
                        Some(value) => self.truth_value(&value)?,
                        None => {
                            let container = right;
                            self.contains_value(&container, &left)?
                        }
                    };
                Value::Bool(contained == (operator == ComparisonOperator::In))
            }
            _ => self.rich_compare(operator, left, right)?,
        };
        self.push(result);
        Ok(())
    }

    /// Answer `needle in container` once `__contains__` has declined. Builtin containers answer
    /// directly; any other iterable is searched item by item, stopping at the first match, as
    /// CPython does.
    pub(super) fn contains_value(
        &mut self,
        container: &Value<'s>,
        needle: &Value<'s>,
    ) -> Result<bool, String> {
        if string::string_ref(self.heap(), *container)?.is_some() {
            let Some(needle_text) = string::string_ref(self.heap(), *needle)? else {
                let message = format!(
                    "'in <string>' requires string as left operand, not {}",
                    self.type_name_of(needle)?
                );
                return Err(self.raise_exception("TypeError", message));
            };
            let scanned = string::string_ref(self.heap(), *container)?
                .map_or(0, |text| text.byte_len())
                .saturating_add(needle_text.byte_len());
            self.charge_cpu(super::objects::scan_cost(scanned))?;
            return protocol::contains(self.heap(), *container, *needle);
        }
        if !container.is_object() {
            let message = format!(
                "argument of type '{}' is not a container or iterable",
                self.type_name_of(container)?
            );
            return Err(self.raise_exception("TypeError", message));
        }
        let direct = match self.get(*container)? {
            Object::Bytes(_)
            | Object::ByteArray(_)
            | Object::List(_)
            | Object::Tuple(_)
            | Object::Set(_)
            | Object::FrozenSet(_)
            | Object::Dict(_)
            | Object::DefaultDict { .. } => true,
            Object::Range { .. } => number::int_value(self.heap(), *needle).is_some(),
            _ => false,
        };
        if direct {
            if let Object::Bytes(value) | Object::ByteArray(value) = self.get(*container)? {
                let length = value.len();
                // A bytes needle is searched for in linear time; an int needle is one byte.
                let needle_length = match string::bytes_ref(self.heap(), *needle)? {
                    Some(needle) => needle.len(),
                    None => match number::int_value(self.heap(), *needle) {
                        Some(byte) if (0..256).contains(&byte) => 1,
                        Some(_) => {
                            return Err(
                                self.raise_exception("ValueError", "byte must be in range(0, 256)")
                            );
                        }
                        None => {
                            let message = format!(
                                "a bytes-like object is required, not '{}'",
                                self.type_name_of(needle)?
                            );
                            return Err(self.raise_exception("TypeError", message));
                        }
                    },
                };
                self.charge_cpu(
                    u64::try_from(length.saturating_add(needle_length)).unwrap_or(u64::MAX),
                )?;
            }
            return match self.get(*container)? {
                Object::List(_) | Object::Tuple(_) => self.sequence_contains(*container, needle),
                Object::Set(_) | Object::FrozenSet(_) => {
                    Ok(self.find_set_entry(*container, needle)?.is_some())
                }
                Object::Dict(_) | Object::DefaultDict { .. } => {
                    Ok(self.find_mapping_entry(*container, needle)?.is_some())
                }
                _ => protocol::contains(self.heap(), *container, *needle),
            };
        }
        let iterator = self.make_iterator(*container)?;
        let needle = *needle;
        // The iterator may be unbounded, so each step's handles are released with its scope.
        loop {
            let mut vm = self.scope();
            vm.charge_cpu(1)?;
            let item = match vm.iterator_next(&iterator) {
                Ok(Some(item)) => item,
                Ok(None) => return Ok(false),
                Err(_) if vm.pending_stop_iteration() => {
                    vm.pending_exception = None;
                    return Ok(false);
                }
                Err(error) => return Err(error),
            };
            if vm.values_equal(&item, &needle)? {
                return Ok(true);
            }
        }
    }

    pub(super) fn binary(&mut self, operator: BinaryOperator) -> Result<(), String> {
        if self.frame_stack_len() < 2 {
            return Err("invalid bytecode stack effect".into());
        }
        let result_slot = self
            .stack
            .len()
            .checked_sub(2)
            .expect("frame operand count was checked");
        let left = self.peek(1)?;
        let right = self.peek(0)?;
        match number::exact_binary(self, operator, left, right)
            .map_err(|error| self.record_native_error(error))
        {
            Ok(Some(value)) => {
                self.stack.truncate(result_slot + 1);
                self.execution
                    .stack
                    .set(&self.state.heap, result_slot, value);
                return Ok(());
            }
            Ok(None) => {}
            Err(error) => {
                self.stack.truncate(result_slot);
                return Err(error);
            }
        }
        self.stack.truncate(result_slot);
        let value = self.binary_protocol(operator, left, right)?;
        self.push(value);
        Ok(())
    }

    pub(super) fn binary_value(
        &mut self,
        operator: BinaryOperator,
        left: Value<'s>,
        right: Value<'s>,
    ) -> Result<Value<'s>, String> {
        if let Some(value) = number::exact_binary(self, operator, left, right)
            .map_err(|error| self.record_native_error(error))?
        {
            return Ok(value);
        }
        self.binary_protocol(operator, left, right)
    }

    /// Augmented assignment. Numbers take the binary fast path. Builtin mutable containers
    /// update the left operand in place, as `list.__iadd__` and `set.__ior__` do, and other
    /// operands use their in-place method, such as `__iadd__`, when they define one. Everything
    /// else falls back to the binary operator.
    pub(super) fn inplace_binary(&mut self, operator: BinaryOperator) -> Result<(), String> {
        let right = self.pop()?;
        let left = self.pop()?;
        if let Some(value) = number::exact_binary(self, operator, left, right)
            .map_err(|error| self.record_native_error(error))?
        {
            self.push(value);
            return Ok(());
        }
        let value = match self.builtin_inplace(operator, left, right)? {
            Some(value) => value,
            None => match self.inplace_method(operator, left, right)? {
                Some(value) => value,
                None => self.binary_protocol_for(operator, left, right, true)?,
            },
        };
        self.push(value);
        Ok(())
    }

    fn builtin_inplace(
        &mut self,
        operator: BinaryOperator,
        left: Value<'s>,
        right: Value<'s>,
    ) -> Result<Option<Value<'s>>, String> {
        if !left.is_object() {
            return Ok(None);
        }
        // `dict |= other` is `dict.update(other)`, for a dict subclass too unless it defines its
        // own `__ior__`.
        if operator == BinaryOperator::BitwiseOr && self.updates_dict_in_place(left)? {
            let update = self
                .resolve_attribute(left, "update")?
                .ok_or("dict.update is not available")?;
            self.invoke_value(update, vec![right])?;
            return Ok(Some(left));
        }
        let replacement = match (self.get(left)?, operator) {
            // `list += iterable` extends with any iterable, unlike `list + list`.
            (Object::List(items), BinaryOperator::Add) => {
                let mut items = self.handles(items);
                for value in self.iterable_values(&right)? {
                    self.push_materialized(&mut items, value)?;
                }
                Replacement::List(items)
            }
            (Object::List(_), BinaryOperator::Multiply)
            | (
                Object::Set(_),
                BinaryOperator::BitwiseOr
                | BinaryOperator::BitwiseAnd
                | BinaryOperator::BitwiseXor
                | BinaryOperator::Subtract,
            )
            | (Object::ByteArray(_), BinaryOperator::Add | BinaryOperator::Multiply) => {
                let result = self.binary_protocol_for(operator, left, right, true)?;
                if !result.is_object() {
                    return Err("in-place container operation produced a non-object".into());
                }
                match self.get(result)? {
                    Object::List(items) => Replacement::List(self.handles(items)),
                    Object::Set(members) => Replacement::Set(HashedMembers(
                        members
                            .iter_hashed()
                            .map(|(hash, member)| (hash, self.handle(member)))
                            .collect(),
                    )),
                    Object::ByteArray(bytes) => Replacement::ByteArray(bytes.clone()),
                    _ => {
                        return Err(
                            "in-place container operation produced an unexpected object".into()
                        )
                    }
                }
            }
            _ => return Ok(None),
        };
        self.replace_contents(left, replacement)?;
        Ok(Some(left))
    }

    /// Install `replacement` as the payload of the list, set or bytearray `target`, keeping its
    /// identity, and charge or release the change in its modeled size.
    fn replace_contents(
        &mut self,
        target: Value<'s>,
        replacement: Replacement<'s>,
    ) -> Result<(), String> {
        let (old_len, new_len, unit) = match (self.get(target)?, &replacement) {
            (Object::List(items), Replacement::List(values)) => {
                (items.len(), values.len(), MODELED_VALUE_BYTES)
            }
            (Object::Set(members), Replacement::Set(values)) => {
                (members.len(), values.len(), MODELED_SET_MEMBER_BYTES)
            }
            (Object::ByteArray(_), Replacement::ByteArray(bytes)) => {
                return self.replace_payload(target, Object::ByteArray(bytes.clone()));
            }
            _ => return Err("in-place container operation changed the object kind".into()),
        };
        let bytes = |count: usize| {
            u64::try_from(count)
                .ok()
                .and_then(|count| count.checked_mul(unit))
                .ok_or_else(|| String::from("modeled object size overflow"))
        };
        if new_len > old_len {
            self.reserve_object_growth(target, bytes(new_len - old_len)?)?;
        }
        self.modify(target, |builder: &Builder<'_>, object| {
            *object = match replacement {
                Replacement::List(values) => Object::List(builder.refs(values)),
                Replacement::Set(members) => Object::Set(members.into_set(builder)),
                Replacement::ByteArray(_) => unreachable!("bytearrays are replaced above"),
            };
        })?;
        if old_len > new_len {
            self.release_object_shrink(target, bytes(old_len - new_len)?)?;
        }
        Ok(())
    }

    /// Call the left operand's in-place method, looked up on its type as CPython does. A method
    /// that returns `NotImplemented` declines, and the caller falls back to the binary operator.
    /// Whether `|=` on object `value` is `dict.update`: a dict or namespace view, or a dict
    /// subclass instance whose class does not define `__ior__`.
    fn updates_dict_in_place(&mut self, value: Value<'s>) -> Result<bool, String> {
        if let Some(class) = self.instance_class(value)? {
            let holds_dict = matches!(self.get(value)?, Object::Dict(_));
            return Ok(holds_dict && self.class_attribute(class, "__ior__")?.is_none());
        }
        Ok(matches!(
            self.get(value)?,
            Object::Dict(_) | Object::DefaultDict { .. } | Object::NamespaceDict(_)
        ))
    }

    fn inplace_method(
        &mut self,
        operator: BinaryOperator,
        left: Value<'s>,
        right: Value<'s>,
    ) -> Result<Option<Value<'s>>, String> {
        let slot = match operator {
            BinaryOperator::Add => Slot::InplaceAdd,
            BinaryOperator::Subtract => Slot::InplaceSubtract,
            BinaryOperator::Multiply => Slot::InplaceMultiply,
            BinaryOperator::MatrixMultiply => Slot::InplaceMatrixMultiply,
            BinaryOperator::Power => Slot::InplacePower,
            BinaryOperator::Divide => Slot::InplaceDivide,
            BinaryOperator::FloorDivide => Slot::InplaceFloorDivide,
            BinaryOperator::Remainder => Slot::InplaceRemainder,
            BinaryOperator::LeftShift => Slot::InplaceLeftShift,
            BinaryOperator::RightShift => Slot::InplaceRightShift,
            BinaryOperator::BitwiseAnd => Slot::InplaceBitwiseAnd,
            BinaryOperator::BitwiseXor => Slot::InplaceBitwiseXor,
            BinaryOperator::BitwiseOr => Slot::InplaceBitwiseOr,
        };
        let (_, name, _) = super::super::object_model::SLOT_DEFS[slot as usize];
        Ok(self
            .invoke_slot(&left, slot, name, vec![right])?
            .filter(|result| result.native_value() != Some(NativeValue::NotImplemented)))
    }

    #[cold]
    #[inline(never)]
    fn binary_protocol(
        &mut self,
        operator: BinaryOperator,
        left: Value<'s>,
        right: Value<'s>,
    ) -> Result<Value<'s>, String> {
        self.binary_protocol_for(operator, left, right, false)
    }

    /// The binary operator protocol. `inplace` only changes the operator an error names.
    fn binary_protocol_for(
        &mut self,
        operator: BinaryOperator,
        left: Value<'s>,
        right: Value<'s>,
        inplace: bool,
    ) -> Result<Value<'s>, String> {
        let (slot, name, reflected_slot, reflected_name) = match operator {
            BinaryOperator::Add => (Slot::Add, "__add__", Slot::ReflectedAdd, "__radd__"),
            BinaryOperator::Subtract => (
                Slot::Subtract,
                "__sub__",
                Slot::ReflectedSubtract,
                "__rsub__",
            ),
            BinaryOperator::Multiply => (
                Slot::Multiply,
                "__mul__",
                Slot::ReflectedMultiply,
                "__rmul__",
            ),
            BinaryOperator::MatrixMultiply => (
                Slot::MatrixMultiply,
                "__matmul__",
                Slot::ReflectedMatrixMultiply,
                "__rmatmul__",
            ),
            BinaryOperator::Power => (Slot::Power, "__pow__", Slot::ReflectedPower, "__rpow__"),
            BinaryOperator::Divide => (
                Slot::Divide,
                "__truediv__",
                Slot::ReflectedDivide,
                "__rtruediv__",
            ),
            BinaryOperator::FloorDivide => (
                Slot::FloorDivide,
                "__floordiv__",
                Slot::ReflectedFloorDivide,
                "__rfloordiv__",
            ),
            BinaryOperator::Remainder => (
                Slot::Remainder,
                "__mod__",
                Slot::ReflectedRemainder,
                "__rmod__",
            ),
            BinaryOperator::LeftShift => (
                Slot::LeftShift,
                "__lshift__",
                Slot::ReflectedLeftShift,
                "__rlshift__",
            ),
            BinaryOperator::RightShift => (
                Slot::RightShift,
                "__rshift__",
                Slot::ReflectedRightShift,
                "__rrshift__",
            ),
            BinaryOperator::BitwiseAnd => (
                Slot::BitwiseAnd,
                "__and__",
                Slot::ReflectedBitwiseAnd,
                "__rand__",
            ),
            BinaryOperator::BitwiseXor => (
                Slot::BitwiseXor,
                "__xor__",
                Slot::ReflectedBitwiseXor,
                "__rxor__",
            ),
            BinaryOperator::BitwiseOr => (
                Slot::BitwiseOr,
                "__or__",
                Slot::ReflectedBitwiseOr,
                "__ror__",
            ),
        };
        let symbol = match (operator, inplace) {
            (BinaryOperator::Power, true) => "**=".to_string(),
            (operator, true) => format!("{}=", binary_operator_symbol(operator)),
            (operator, false) => binary_operator_symbol(operator).to_string(),
        };
        self.binary_slot_protocol(
            left,
            right,
            (slot, name, reflected_slot, reflected_name),
            &symbol,
        )
    }

    /// `divmod(left, right)`: the binary protocol over `__divmod__` and `__rdivmod__`.
    pub(super) fn divmod_value(
        &mut self,
        left: Value<'s>,
        right: Value<'s>,
    ) -> Result<Value<'s>, String> {
        self.binary_slot_protocol(
            left,
            right,
            (
                Slot::DivMod,
                "__divmod__",
                Slot::ReflectedDivMod,
                "__rdivmod__",
            ),
            "divmod()",
        )
    }

    /// CPython's `binary_op1`: try the left operand's method, then the right operand's
    /// reflected method, and raise `TypeError` naming `symbol` when both decline.
    fn binary_slot_protocol(
        &mut self,
        left: Value<'s>,
        right: Value<'s>,
        (slot, name, reflected_slot, reflected_name): (Slot, &str, Slot, &str),
        symbol: &str,
    ) -> Result<Value<'s>, String> {
        // As in CPython, a right operand whose type is a proper subclass of the left operand's
        // type gets its reflected method first, so `1.0 + np.float64(2)` stays a NumPy scalar.
        let left_type = self.type_id(&left)?;
        let right_type = self.type_id(&right)?;
        let right_first = left_type != right_type
            && self.state.types.is_subclass(right_type, left_type)?
            && self.state.types.slot(right_type, reflected_slot)?.is_some();
        if right_first {
            if let Some(value) =
                self.invoke_operator_slot(&right, reflected_slot, reflected_name, vec![left])?
            {
                return Ok(value);
            }
        }
        if let Some(value) = self.invoke_operator_slot(&left, slot, name, vec![right])? {
            return Ok(value);
        }
        if !right_first {
            if let Some(value) =
                self.invoke_operator_slot(&right, reflected_slot, reflected_name, vec![left])?
            {
                return Ok(value);
            }
        }
        let message = match self.sequence_operator_message(&left, &right, symbol)? {
            Some(message) => message,
            None => format!(
                "unsupported operand type(s) for {symbol}: '{}' and '{}'",
                self.type_name_of(&left)?,
                self.type_name_of(&right)?
            ),
        };
        Err(self.raise_exception("TypeError", message))
    }

    /// CPython's message when `+` or `*` reaches a builtin sequence's concatenation or
    /// repetition with an operand it cannot combine, such as `"a" + 1`.
    fn sequence_operator_message(
        &mut self,
        left: &Value<'s>,
        right: &Value<'s>,
        symbol: &str,
    ) -> Result<Option<String>, String> {
        let left_sequence = self.builtin_sequence(left)?;
        match symbol.trim_end_matches('=') {
            "+" => {
                let Some(sequence) = left_sequence else {
                    return Ok(None);
                };
                let other = self.type_name_of(right)?;
                Ok(Some(match sequence {
                    BuiltinType::Bytes => format!("can't concat {other} to bytes"),
                    BuiltinType::ByteArray => format!("can't concat {other} to bytearray"),
                    _ => {
                        let name = builtin_sequence_name(sequence);
                        format!("can only concatenate {name} (not \"{other}\") to {name}")
                    }
                }))
            }
            "*" => {
                let other = if left_sequence.is_some() {
                    right
                } else if self.builtin_sequence(right)?.is_some() {
                    left
                } else {
                    return Ok(None);
                };
                let other = self.type_name_of(other)?;
                Ok(Some(format!(
                    "can't multiply sequence by non-int of type '{other}'"
                )))
            }
            _ => Ok(None),
        }
    }

    /// The builtin sequence type (`str`, `bytes`, `bytearray`, `list` or `tuple`) that
    /// `value`'s type derives from, if any.
    fn builtin_sequence(&self, value: &Value<'s>) -> Result<Option<BuiltinType>, String> {
        let type_id = self.type_id(value)?;
        for sequence in [
            BuiltinType::String,
            BuiltinType::Bytes,
            BuiltinType::ByteArray,
            BuiltinType::List,
            BuiltinType::Tuple,
        ] {
            if self.state.types.is_subclass(type_id, sequence.id())? {
                return Ok(Some(sequence));
            }
        }
        Ok(None)
    }

    pub(super) fn format_value(
        &mut self,
        conversion: Option<char>,
        format_spec: &str,
    ) -> Result<(), String> {
        let value = self.pop()?;
        let rendered = self.render_formatted_value(&value, conversion, format_spec)?;
        self.charge_cpu(u64::try_from(rendered.len()).unwrap_or(u64::MAX))?;
        let rendered = self.allocate_string(rendered)?;
        self.push(rendered);
        Ok(())
    }

    pub(super) fn render_formatted_value(
        &mut self,
        value: &Value<'s>,
        conversion: Option<char>,
        format_spec: &str,
    ) -> Result<String, String> {
        self.reserve_format_spec(format_spec)?;
        if format_spec.contains(['{', '}']) {
            return Err("nested f-string format specifications are not implemented".into());
        }
        let converted = match conversion {
            Some('r' | 'a') => self.repr_value(value)?,
            Some('s') => self.display_value(value)?,
            Some(other) => return Err(format!("unsupported f-string conversion !{other}")),
            None => return self.format_object(value, format_spec),
        };
        if format_spec.is_empty() {
            Ok(converted)
        } else {
            format_text(&converted, format_spec).map_err(|error| self.raise_format_error(error))
        }
    }

    fn raise_format_error(&mut self, error: FormatError) -> String {
        self.raise_exception(error.kind, error.message)
    }

    /// `format(value, spec)`, which CPython defines as `type(value).__format__(value, spec)`.
    /// A user class's `__format__` runs as written. Registered numbers such as NumPy scalars
    /// format as the Python number they stand for, as NumPy's `__format__` does. Other values
    /// accept only the empty spec, which gives `str(value)`.
    pub(super) fn format_object(
        &mut self,
        value: &Value<'s>,
        format_spec: &str,
    ) -> Result<String, String> {
        let spec = self.allocate_string(format_spec.to_string())?;
        if let Some(result) = self.invoke_slot(value, Slot::Format, "__format__", vec![spec])? {
            return string::string_value(self.heap(), result)?
                .ok_or_else(|| self.raise_exception("TypeError", "__format__ must return a str"));
        }
        if let Some(rendered) = self.format_registered_number(value, format_spec)? {
            return Ok(rendered);
        }
        if format_spec.is_empty() {
            return self.display_value(value);
        }
        self.format_unconverted_value(value, format_spec)
    }

    /// Registered scalar formats use their Python numeric value, including when the spec is empty.
    pub(super) fn format_registered_number(
        &mut self,
        value: &Value<'s>,
        format_spec: &str,
    ) -> Result<Option<String>, String> {
        let Some((_, number)) = super::number::registered_number(self.heap(), value) else {
            return Ok(None);
        };
        let number = match number {
            KindNumber::Bool(value) => Value::Bool(value),
            KindNumber::Int(value) => Value::Int(value),
            KindNumber::UInt(value) => self.allocate_object(Object::BigInt(BigInt::from(value)))?,
            KindNumber::Float(value) => Value::Float(value),
            KindNumber::Complex(real, imag) => {
                self.allocate_object(Object::Complex { real, imag })?
            }
        };
        self.format_object(&number, format_spec).map(Some)
    }

    /// Reserve the largest width or precision before formatting can allocate padding.
    pub(super) fn reserve_format_spec(&mut self, spec: &str) -> Result<(), String> {
        let mut largest = 0_usize;
        let mut digits = None::<usize>;
        for byte in spec.bytes() {
            if byte.is_ascii_digit() {
                let next = digits
                    .unwrap_or(0)
                    .saturating_mul(10)
                    .saturating_add(usize::from(byte - b'0'));
                digits = Some(next);
                largest = largest.max(next);
            } else {
                digits = None;
            }
        }
        self.charge_cpu(u64::try_from(largest.max(spec.len())).unwrap_or(u64::MAX))?;
        self.reserve_result(largest.saturating_add(spec.len()).saturating_mul(2))
    }

    /// `format(value, text)` for a builtin value with a non-empty specification: ints, bools
    /// and floats through the numeric mini-language, strings through the string one, and
    /// anything else with the `TypeError` of `object.__format__`.
    pub(super) fn format_unconverted_value(
        &mut self,
        value: &Value<'s>,
        text: &str,
    ) -> Result<String, String> {
        let result = match super::number::view(self.heap(), value) {
            Some(number::NumberRef::Complex(real, imag)) => format_complex(real, imag, text),
            Some(number::NumberRef::Float(float)) => {
                let repr = protocol::repr(self.state, Value::Float(float))?;
                format_float(float, &repr, text)
            }
            Some(_) => {
                let integer = self
                    .bigint_operand(value)
                    .map_err(|_| "integer format requires an integer")?;
                let type_name = self.type_name_of(value)?;
                format_integer(integer, text, &type_name)
            }
            None => match string::string_value(self.heap(), *value)? {
                Some(string) => format_text(&string, text),
                None => {
                    let type_name = self.type_name_of(value)?;
                    return Err(self.raise_exception(
                        "TypeError",
                        format!("unsupported format string passed to {type_name}.__format__"),
                    ));
                }
            },
        };
        result.map_err(|error| self.raise_format_error(error))
    }

    pub(super) fn is_bigint(&self, value: &Value<'s>) -> Result<bool, String> {
        Ok(matches!(
            super::number::view(self.heap(), value),
            Some(super::number::NumberRef::BigInt(_))
        ))
    }

    fn bigint_operand(&self, value: &Value<'s>) -> Result<BigInt, String> {
        super::number::view(self.heap(), value)
            .and_then(super::number::NumberRef::to_bigint)
            .ok_or_else(|| "unsupported arithmetic operands".into())
    }

    pub(super) fn numeric_float(&self, value: &Value<'s>) -> Result<f64, String> {
        if value.is_object() {
            if let Object::BigInt(value) = self.get(*value)? {
                return value
                    .to_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| "int too large to convert to float".into());
            }
        }
        super::number::as_f64(self.heap(), value)
            .ok_or_else(|| "unsupported arithmetic operands".into())
    }

    pub(super) fn add_numbers(
        &mut self,
        left: Value<'s>,
        right: Value<'s>,
    ) -> Result<Value<'s>, String> {
        self.binary_value(BinaryOperator::Add, left, right)
    }
}

/// The source spelling of a binary operator, as CPython prints it in `TypeError` messages.
fn builtin_sequence_name(sequence: BuiltinType) -> &'static str {
    match sequence {
        BuiltinType::String => "str",
        BuiltinType::Bytes => "bytes",
        BuiltinType::ByteArray => "bytearray",
        BuiltinType::List => "list",
        _ => "tuple",
    }
}

fn binary_operator_symbol(operator: BinaryOperator) -> &'static str {
    match operator {
        BinaryOperator::Add => "+",
        BinaryOperator::Subtract => "-",
        BinaryOperator::Multiply => "*",
        BinaryOperator::MatrixMultiply => "@",
        BinaryOperator::Power => "** or pow()",
        BinaryOperator::Divide => "/",
        BinaryOperator::FloorDivide => "//",
        BinaryOperator::Remainder => "%",
        BinaryOperator::LeftShift => "<<",
        BinaryOperator::RightShift => ">>",
        BinaryOperator::BitwiseAnd => "&",
        BinaryOperator::BitwiseXor => "^",
        BinaryOperator::BitwiseOr => "|",
    }
}
