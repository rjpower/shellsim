//! VM adapters for unary, binary, comparison, construction, and formatting operations.

use super::format::{format_float, format_integer, format_text, FormatSpec};
use super::{
    number, protocol, BigInt, BinaryOperator, ComparisonOperator, DisplayKind, Object, Ordering,
    SequenceKind, Slot, ToPrimitive, UnaryOperator, Value, Vm,
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
                match mapping
                    .object_id()
                    .map(|id| self.state.heap.get(id))
                    .transpose()?
                {
                    Some(Object::Dict(entries)) | Some(Object::DefaultDict { entries, .. }) => {
                        entries.to_vec()
                    }
                    _ => return Err("'**' argument must be a mapping".into()),
                }
            } else {
                vec![(
                    values.next().expect("dictionary key stack contract"),
                    values.next().expect("dictionary value stack contract"),
                )]
            };
            for (key, value) in additions {
                let mut replaced = false;
                for (existing_key, existing_value) in &mut entries {
                    if protocol::identical(existing_key, &key)
                        || protocol::equals(&self.state.heap, existing_key, &key)?
                    {
                        *existing_value = value;
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

    pub(super) fn compare(&mut self, operator: ComparisonOperator) -> Result<(), String> {
        let right = self.pop()?;
        let left = self.pop()?;
        if let Some(result) = number::exact_integer_comparison(operator, left, right) {
            self.stack.push(Value::Bool(result));
            return Ok(());
        }
        let mut slot_result = match operator {
            ComparisonOperator::Equal => {
                self.invoke_slot(&left, Slot::Equal, "__eq__", vec![right])?
            }
            ComparisonOperator::NotEqual => {
                self.invoke_slot(&left, Slot::NotEqual, "__ne__", vec![right])?
            }
            ComparisonOperator::Less => {
                self.invoke_slot(&left, Slot::LessThan, "__lt__", vec![right])?
            }
            ComparisonOperator::LessEqual => {
                self.invoke_slot(&left, Slot::LessEqual, "__le__", vec![right])?
            }
            ComparisonOperator::Greater => {
                self.invoke_slot(&left, Slot::GreaterThan, "__gt__", vec![right])?
            }
            ComparisonOperator::GreaterEqual => {
                self.invoke_slot(&left, Slot::GreaterEqual, "__ge__", vec![right])?
            }
            ComparisonOperator::In | ComparisonOperator::NotIn => {
                self.invoke_slot(&right, Slot::Contains, "__contains__", vec![left])?
            }
            ComparisonOperator::Is | ComparisonOperator::IsNot => None,
        };
        if slot_result.is_none() {
            let reflected = match operator {
                ComparisonOperator::Equal => Some((Slot::Equal, "__eq__")),
                ComparisonOperator::NotEqual => Some((Slot::NotEqual, "__ne__")),
                ComparisonOperator::Less => Some((Slot::GreaterThan, "__gt__")),
                ComparisonOperator::LessEqual => Some((Slot::GreaterEqual, "__ge__")),
                ComparisonOperator::Greater => Some((Slot::LessThan, "__lt__")),
                ComparisonOperator::GreaterEqual => Some((Slot::LessEqual, "__le__")),
                _ => None,
            };
            if let Some((slot, name)) = reflected {
                slot_result = self.invoke_slot(&right, slot, name, vec![left])?;
            }
        }
        if slot_result.is_none() && matches!(operator, ComparisonOperator::NotEqual) {
            let mut equality = self.invoke_slot(&left, Slot::Equal, "__eq__", vec![right])?;
            if equality.is_none() {
                equality = self.invoke_slot(&right, Slot::Equal, "__eq__", vec![left])?;
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
        let result = match operator {
            ComparisonOperator::Equal => protocol::equals(&self.state.heap, &left, &right)?,
            ComparisonOperator::NotEqual => !protocol::equals(&self.state.heap, &left, &right)?,
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
                        return Err(self.raise_unorderable(symbol, &left, &right))
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
            return protocol::contains(&self.state.heap, container, needle);
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
            if self.item_equals(&item, needle)? {
                return Ok(true);
            }
        }
    }

    /// `item == needle` with user `__eq__` on either side, as membership tests use it.
    fn item_equals(&mut self, item: &Value, needle: &Value) -> Result<bool, String> {
        if protocol::identical(item, needle) {
            return Ok(true);
        }
        if let Some(result) = self.invoke_slot(item, Slot::Equal, "__eq__", vec![*needle])? {
            return self.truth_value(&result);
        }
        if let Some(result) = self.invoke_slot(needle, Slot::Equal, "__eq__", vec![*item])? {
            return self.truth_value(&result);
        }
        protocol::equals(&self.state.heap, item, needle)
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

    #[cold]
    #[inline(never)]
    fn binary_protocol(
        &mut self,
        operator: BinaryOperator,
        left: Value,
        right: Value,
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
        if let Some(value) = self.invoke_slot(&left, slot, name, vec![right])? {
            return Ok(value);
        }
        if let Some(value) = self.invoke_slot(&right, reflected_slot, reflected_name, vec![left])? {
            return Ok(value);
        }
        let message = format!(
            "unsupported operand type(s) for {}: '{}' and '{}'",
            binary_operator_symbol(operator),
            self.type_name_of(&left)?,
            self.type_name_of(&right)?
        );
        Err(self.raise_exception("TypeError", message))
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
        let converted = match conversion {
            Some('r' | 'a') => Some(protocol::repr(&self.state.heap, value)?),
            Some('s') => Some(protocol::display(&self.state.heap, value)?),
            Some(other) => return Err(format!("unsupported f-string conversion !{other}")),
            None => None,
        };
        let rendered = if format_spec.is_empty() {
            converted.unwrap_or(protocol::display(&self.state.heap, value)?)
        } else if format_spec.contains(['{', '}']) {
            return Err("nested f-string format specifications are not implemented".into());
        } else if let Some(converted) = converted {
            format_text(&converted, &FormatSpec::parse(format_spec)?)?
        } else {
            self.format_unconverted_value(value, format_spec)?
        };
        Ok(rendered)
    }

    /// Reserve the largest width or precision before formatting can allocate padding.
    fn reserve_format_spec(&mut self, spec: &str) -> Result<(), String> {
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

    fn format_unconverted_value(&self, value: &Value, text: &str) -> Result<String, String> {
        if super::number::is_complex(&self.state.heap, value) {
            return Err("format specifications for complex numbers are not implemented".into());
        }
        let spec = FormatSpec::parse(text)?;
        let number = super::number::view(&self.state.heap, value);
        match (spec.presentation, number) {
            (Some('f' | 'e' | 'E' | 'g' | 'G' | '%'), Some(_))
            | (None, Some(number::NumberRef::Float(_))) => {
                let float = super::number::as_f64(&self.state.heap, value)
                    .ok_or("floating-point format requires a number")?;
                format_float(
                    float,
                    &protocol::repr(&self.state.heap, &Value::Float(float))?,
                    &spec,
                )
            }
            (Some('d' | 'b' | 'o' | 'x' | 'X') | None, Some(_)) => {
                let integer = self
                    .bigint_operand(value)
                    .map_err(|_| "integer format requires an integer")?;
                format_integer(integer, &spec)
            }
            (_, Some(_)) => Err(format!("unsupported numeric format {text:?}")),
            (_, None) => match protocol::string_value(&self.state.heap, value)? {
                Some(text) => format_text(&text, &spec),
                None => Err(format!("unsupported format specification {text:?}")),
            },
        }
    }

    pub(super) fn is_bigint(&self, value: &Value) -> Result<bool, String> {
        Ok(matches!(
            super::number::view(&self.state.heap, value),
            Some(super::number::NumberRef::BigInt(_))
        ))
    }

    fn bigint_operand(&self, value: &Value) -> Result<BigInt, String> {
        match super::number::view(&self.state.heap, value) {
            Some(super::number::NumberRef::Int(value)) => Ok(BigInt::from(value)),
            Some(super::number::NumberRef::BigInt(value)) => Ok(value.clone()),
            Some(super::number::NumberRef::Float(_) | super::number::NumberRef::Complex(..))
            | None => Err("unsupported arithmetic operands".into()),
        }
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
