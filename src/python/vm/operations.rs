//! VM adapters for unary, binary, comparison, construction, and formatting operations.

use super::super::native::KindNumber;
use super::format::{format_complex, format_float, format_integer, format_text, FormatError};
use super::{
    number, protocol, BigInt, BinaryOperator, BuiltinType, ComparisonOperator, DisplayKind,
    NativeValue, Object, Ordering, SequenceKind, Slot, ToPrimitive, UnaryOperator, Value, Vm,
};

impl Vm<'_> {
    pub(super) fn unary(&mut self, operator: UnaryOperator) -> Result<(), String> {
        let value = self.pop()?;
        if operator == UnaryOperator::Not {
            let value = Value::Bool(!self.truth_value(&value)?);
            self.stack.push(value);
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
        self.stack.push(result);
        Ok(())
    }

    pub(super) fn build_sequence(
        &mut self,
        count: usize,
        kind: SequenceKind,
    ) -> Result<(), String> {
        let values = self.take(count)?;
        let object = match kind {
            SequenceKind::List => Object::List(values),
            SequenceKind::Tuple => Object::Tuple(values),
        };
        let value = self
            .state
            .heap
            .allocate(object, &mut self.interp.resources)?;
        self.stack.push(value);
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
        let mut entries: Vec<(Value, Value)> = Vec::with_capacity(unpacked.len());
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
            for (key, value) in additions {
                let mut replaced = false;
                for entry in &mut entries {
                    if self.values_equal(&entry.0, &key)? {
                        entry.1 = value;
                        replaced = true;
                        break;
                    }
                }
                if !replaced {
                    entries.push((key, value));
                }
            }
        }
        let value = self
            .state
            .heap
            .allocate(Object::Dict(entries.into()), &mut self.interp.resources)?;
        self.stack.push(value);
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
        let object = match kind {
            DisplayKind::List => Object::List(values),
            DisplayKind::Tuple => Object::Tuple(values),
            DisplayKind::Set => return self.push_set(values),
        };
        let value = self
            .state
            .heap
            .allocate(object, &mut self.interp.resources)?;
        self.stack.push(value);
        Ok(())
    }

    pub(super) fn build_set(&mut self, count: usize) -> Result<(), String> {
        let candidates = self.take(count)?;
        self.push_set(candidates)
    }

    /// Push a set of the distinct `candidates`, metering each membership comparison as the `set`
    /// constructor does.
    fn push_set(&mut self, candidates: Vec<Value>) -> Result<(), String> {
        let mut values = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            self.charge_cpu(1)?;
            if self.find_value(&values, &candidate)?.is_none() {
                values.push(candidate);
            }
        }
        let value = self
            .state
            .heap
            .allocate(Object::Set(values), &mut self.interp.resources)?;
        self.stack.push(value);
        Ok(())
    }

    /// The truth of `left <operator> right` under the full rich-comparison protocol, including
    /// the reflected operand, as CPython's `PyObject_RichCompareBool` computes it for `min`,
    /// `max` and sorting.
    pub(super) fn compare_truth(
        &mut self,
        operator: ComparisonOperator,
        left: &Value,
        right: &Value,
    ) -> Result<bool, String> {
        self.stack.push(*left);
        self.stack.push(*right);
        self.compare(operator)?;
        let result = self.pop()?;
        self.truth_value(&result)
    }

    pub(super) fn compare(&mut self, operator: ComparisonOperator) -> Result<(), String> {
        let right = self.pop()?;
        let left = self.pop()?;
        if let Some(result) = number::exact_integer_comparison(operator, left, right) {
            self.stack.push(Value::Bool(result));
            return Ok(());
        }
        let comparison_slots = match operator {
            ComparisonOperator::Equal => Some(((Slot::Equal, "__eq__"), (Slot::Equal, "__eq__"))),
            ComparisonOperator::NotEqual => {
                Some(((Slot::NotEqual, "__ne__"), (Slot::NotEqual, "__ne__")))
            }
            ComparisonOperator::Less => {
                Some(((Slot::LessThan, "__lt__"), (Slot::GreaterThan, "__gt__")))
            }
            ComparisonOperator::LessEqual => {
                Some(((Slot::LessEqual, "__le__"), (Slot::GreaterEqual, "__ge__")))
            }
            ComparisonOperator::Greater => {
                Some(((Slot::GreaterThan, "__gt__"), (Slot::LessThan, "__lt__")))
            }
            ComparisonOperator::GreaterEqual => {
                Some(((Slot::GreaterEqual, "__ge__"), (Slot::LessEqual, "__le__")))
            }
            _ => None,
        };
        let mut slot_result = None;
        if let Some(((left_slot, left_name), (right_slot, right_name))) = comparison_slots {
            let left_type = self.type_id(&left)?;
            let right_type = self.type_id(&right)?;
            let right_first = right_type != left_type
                && self.state.types.is_subclass(right_type, left_type)?
                && self
                    .state
                    .types
                    .local_slot(right_type, right_slot)?
                    .is_some();
            if right_first {
                slot_result =
                    self.invoke_operator_slot(&right, right_slot, right_name, vec![left])?;
            }
            if slot_result.is_none() {
                slot_result =
                    self.invoke_operator_slot(&left, left_slot, left_name, vec![right])?;
            }
            if slot_result.is_none() && !right_first {
                slot_result =
                    self.invoke_operator_slot(&right, right_slot, right_name, vec![left])?;
            }
        } else if matches!(operator, ComparisonOperator::In | ComparisonOperator::NotIn) {
            slot_result = self.invoke_slot(&right, Slot::Contains, "__contains__", vec![left])?;
        }
        if slot_result.is_none() && matches!(operator, ComparisonOperator::NotEqual) {
            let mut equality =
                self.invoke_operator_slot(&left, Slot::Equal, "__eq__", vec![right])?;
            if equality.is_none() {
                equality = self.invoke_operator_slot(&right, Slot::Equal, "__eq__", vec![left])?;
            }
            if let Some(value) = equality {
                slot_result = Some(Value::Bool(!self.truth_value(&value)?));
            }
        }
        if let Some(value) = slot_result {
            if matches!(operator, ComparisonOperator::NotIn) {
                let result = !self.truth_value(&value)?;
                self.stack.push(Value::Bool(result));
            } else {
                self.stack.push(value);
            }
            return Ok(());
        }
        // Instances of builtin subclasses compare as the values they hold; errors still name the
        // subclasses.
        let (subject_left, subject_right) = (left, right);
        let (left, right) = (self.builtin_view(left)?, self.builtin_view(right)?);
        let result = match operator {
            ComparisonOperator::Equal => self.builtin_equality(&left, &right)?,
            ComparisonOperator::NotEqual => !self.builtin_equality(&left, &right)?,
            ComparisonOperator::Less
            | ComparisonOperator::LessEqual
            | ComparisonOperator::Greater
            | ComparisonOperator::GreaterEqual => {
                let (symbol, accepted): (&str, &[Ordering]) = match operator {
                    ComparisonOperator::Less => ("<", &[Ordering::Less]),
                    ComparisonOperator::LessEqual => ("<=", &[Ordering::Less, Ordering::Equal]),
                    ComparisonOperator::Greater => (">", &[Ordering::Greater]),
                    _ => (">=", &[Ordering::Greater, Ordering::Equal]),
                };
                match self.compare_values(&left, &right)? {
                    protocol::Comparison::Ordered(ordering) => accepted.contains(&ordering),
                    protocol::Comparison::Unordered => false,
                    protocol::Comparison::Unsupported => {
                        return Err(self.raise_unorderable(symbol, &subject_left, &subject_right))
                    }
                }
            }
            ComparisonOperator::In => self.contains_value(&right, &left)?,
            ComparisonOperator::NotIn => !self.contains_value(&right, &left)?,
            ComparisonOperator::Is => protocol::identical(&left, &right),
            ComparisonOperator::IsNot => !protocol::identical(&left, &right),
        };
        self.stack.push(Value::Bool(result));
        Ok(())
    }

    /// Answer `needle in container` once `__contains__` has declined. Builtin containers answer
    /// directly; any other iterable is searched item by item, stopping at the first match, as
    /// CPython does.
    fn contains_value(&mut self, container: &Value, needle: &Value) -> Result<bool, String> {
        if protocol::string_ref(&self.state.heap, container)?.is_some() {
            if protocol::string_ref(&self.state.heap, needle)?.is_none() {
                let message = format!(
                    "'in <string>' requires string as left operand, not {}",
                    self.type_name_of(needle)?
                );
                return Err(self.raise_exception("TypeError", message));
            }
            return protocol::contains(&self.state.heap, container, needle);
        }
        let Some(id) = container.object_id() else {
            let message = format!(
                "argument of type '{}' is not a container or iterable",
                self.type_name_of(container)?
            );
            return Err(self.raise_exception("TypeError", message));
        };
        let direct = match self.state.heap.get(id)? {
            Object::Bytes(_)
            | Object::ByteArray(_)
            | Object::List(_)
            | Object::Tuple(_)
            | Object::Set(_)
            | Object::FrozenSet(_)
            | Object::Dict(_)
            | Object::DefaultDict { .. } => true,
            Object::Range { .. } => protocol::int_value(&self.state.heap, needle).is_some(),
            _ => false,
        };
        if direct {
            if let Object::Bytes(value) | Object::ByteArray(value) = self.state.heap.get(id)? {
                let length = value.len();
                // A bytes needle is searched for in linear time; an int needle is one byte.
                let needle_length = match protocol::bytes_ref(&self.state.heap, needle)? {
                    Some(needle) => needle.len(),
                    None => match protocol::int_value(&self.state.heap, needle) {
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
            return match self.state.heap.get(id)? {
                Object::List(_) | Object::Tuple(_) => self.sequence_contains(id, needle),
                Object::Set(_) | Object::FrozenSet(_) => {
                    Ok(self.find_set_entry(id, needle)?.is_some())
                }
                Object::Dict(_) | Object::DefaultDict { .. } => {
                    Ok(self.find_mapping_entry(id, needle)?.is_some())
                }
                _ => protocol::contains(&self.state.heap, container, needle),
            };
        }
        let iterator = self.make_iterator(*container)?;
        loop {
            self.charge_cpu(1)?;
            let item = match self.iterator_next(&iterator) {
                Ok(Some(item)) => item,
                Ok(None) => return Ok(false),
                Err(_) if self.pending_stop_iteration() => {
                    self.pending_exception = None;
                    return Ok(false);
                }
                Err(error) => return Err(error),
            };
            if self.values_equal(&item, needle)? {
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
        let left = self.stack[result_slot];
        let right = self.stack[result_slot + 1];
        match number::exact_binary(self, operator, left, right)
            .map_err(|error| self.record_native_error(error))
        {
            Ok(Some(value)) => {
                self.stack.truncate(result_slot + 1);
                self.stack[result_slot] = value;
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
        self.stack.push(value);
        Ok(())
    }

    pub(super) fn binary_value(
        &mut self,
        operator: BinaryOperator,
        left: Value,
        right: Value,
    ) -> Result<Value, String> {
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
            self.stack.push(value);
            return Ok(());
        }
        let value = match self.builtin_inplace(operator, left, right)? {
            Some(value) => value,
            None => match self.inplace_method(operator, left, right)? {
                Some(value) => value,
                None => self.binary_protocol_for(operator, left, right, true)?,
            },
        };
        self.stack.push(value);
        Ok(())
    }

    fn builtin_inplace(
        &mut self,
        operator: BinaryOperator,
        left: Value,
        right: Value,
    ) -> Result<Option<Value>, String> {
        let Some(id) = left.object_id() else {
            return Ok(None);
        };
        // `dict |= other` is `dict.update(other)`, for a dict subclass too unless it defines its
        // own `__ior__`.
        if operator == BinaryOperator::BitwiseOr && self.updates_dict_in_place(id)? {
            let update = self
                .resolve_attribute(left, "update")?
                .ok_or("dict.update is not available")?;
            self.invoke_value(update, vec![right])?;
            return Ok(Some(left));
        }
        let replacement = match (self.state.heap.get(id)?, operator) {
            // `list += iterable` extends with any iterable, unlike `list + list`.
            (Object::List(items), BinaryOperator::Add) => {
                let mut items = items.clone();
                for value in self.iterable_values(&right)? {
                    self.push_materialized(&mut items, value)?;
                }
                Object::List(items)
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
                let result = result
                    .object_id()
                    .ok_or("in-place container operation produced a non-object")?;
                self.state.heap.get(result)?.clone()
            }
            _ => return Ok(None),
        };
        self.state
            .heap
            .replace_payload(id, replacement, &mut self.interp.resources)?;
        Ok(Some(left))
    }

    /// Call the left operand's in-place method, looked up on its type as CPython does. A method
    /// that returns `NotImplemented` declines, and the caller falls back to the binary operator.
    /// Whether `|=` on object `id` is `dict.update`: a dict or namespace view, or a dict
    /// subclass instance whose class does not define `__ior__`.
    fn updates_dict_in_place(&mut self, id: super::super::heap::ObjectId) -> Result<bool, String> {
        let class = match self.state.heap.get(id)? {
            Object::Dict(_) | Object::DefaultDict { .. } | Object::NamespaceDict(_) => {
                return Ok(true)
            }
            Object::Instance { class, .. } => *class,
            _ => return Ok(false),
        };
        let holds_dict = match protocol::builtin_payload(&self.state.heap, &Value::Object(id))?
            .and_then(|payload| payload.object_id())
        {
            Some(payload) => matches!(self.state.heap.get(payload)?, Object::Dict(_)),
            None => false,
        };
        Ok(holds_dict && self.class_attribute(class, "__ior__")?.is_none())
    }

    fn inplace_method(
        &mut self,
        operator: BinaryOperator,
        left: Value,
        right: Value,
    ) -> Result<Option<Value>, String> {
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
        left: Value,
        right: Value,
    ) -> Result<Value, String> {
        self.binary_protocol_for(operator, left, right, false)
    }

    /// The binary operator protocol. `inplace` only changes the operator an error names.
    fn binary_protocol_for(
        &mut self,
        operator: BinaryOperator,
        left: Value,
        right: Value,
        inplace: bool,
    ) -> Result<Value, String> {
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
    pub(super) fn divmod_value(&mut self, left: Value, right: Value) -> Result<Value, String> {
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
        left: Value,
        right: Value,
        (slot, name, reflected_slot, reflected_name): (Slot, &str, Slot, &str),
        symbol: &str,
    ) -> Result<Value, String> {
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
        left: &Value,
        right: &Value,
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
    fn builtin_sequence(&self, value: &Value) -> Result<Option<BuiltinType>, String> {
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
        self.stack.push(rendered);
        Ok(())
    }

    pub(super) fn render_formatted_value(
        &mut self,
        value: &Value,
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
        value: &Value,
        format_spec: &str,
    ) -> Result<String, String> {
        let spec = self.allocate_string(format_spec.to_string())?;
        if let Some(result) = self.invoke_slot(value, Slot::Format, "__format__", vec![spec])? {
            return protocol::string_value(&self.state.heap, &result)?
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
        value: &Value,
        format_spec: &str,
    ) -> Result<Option<String>, String> {
        let Some((_, number)) = super::number::registered_number(&self.state.heap, value) else {
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
        value: &Value,
        text: &str,
    ) -> Result<String, String> {
        let result = match super::number::view(&self.state.heap, value) {
            Some(number::NumberRef::Complex(real, imag)) => format_complex(real, imag, text),
            Some(number::NumberRef::Float(float)) => {
                let repr = protocol::repr(&self.state.heap, &Value::Float(float))?;
                format_float(float, &repr, text)
            }
            Some(_) => {
                let integer = self
                    .bigint_operand(value)
                    .map_err(|_| "integer format requires an integer")?;
                let type_name = self.type_name_of(value)?;
                format_integer(integer, text, &type_name)
            }
            None => match protocol::string_value(&self.state.heap, value)? {
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

    pub(super) fn is_bigint(&self, value: &Value) -> Result<bool, String> {
        Ok(matches!(
            super::number::view(&self.state.heap, value),
            Some(super::number::NumberRef::BigInt(_))
        ))
    }

    fn bigint_operand(&self, value: &Value) -> Result<BigInt, String> {
        super::number::view(&self.state.heap, value)
            .and_then(super::number::NumberRef::to_bigint)
            .ok_or_else(|| "unsupported arithmetic operands".into())
    }

    pub(super) fn numeric_float(&self, value: &Value) -> Result<f64, String> {
        if let Some(id) = value.object_id() {
            if let Object::BigInt(value) = self.state.heap.get(id)? {
                return value
                    .to_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| "int too large to convert to float".into());
            }
        }
        super::number::as_f64(&self.state.heap, value)
            .ok_or_else(|| "unsupported arithmetic operands".into())
    }

    pub(super) fn add_numbers(&mut self, left: Value, right: Value) -> Result<Value, String> {
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
