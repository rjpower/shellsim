//! Central Python value protocols for truth, representation, equality, ordering, and containment.
//!
//! Container algorithms are deliberately linear. Correct behavior and one auditable dispatch
//! point matter more than asymptotic performance for the bounded evaluator.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use num_bigint::BigInt;
use num_traits::{FromPrimitive, Zero};

use super::heap::{Heap, InstancePayload, Object, ObjectId};
pub use super::string::{string_ref, string_value};
use super::Value;

pub fn display(heap: &Heap, value: &Value) -> Result<String, String> {
    if let Some(value) = string_value(heap, value)? {
        return Ok(value);
    }
    if let Some((kind, message)) = exception_parts(heap, value)? {
        return Ok(if message.is_empty() { kind } else { message });
    }
    repr(heap, value)
}

pub fn repr(heap: &Heap, value: &Value) -> Result<String, String> {
    render(heap, value, &mut BTreeSet::new())
}

/// Return the integer payload of an immediate integer or an `int` subclass instance.
pub fn int_value(heap: &Heap, value: &Value) -> Option<i64> {
    match super::number::index(heap, value)? {
        super::number::NumberRef::Int(value) => Some(value),
        super::number::NumberRef::BigInt(_) | super::number::NumberRef::Float(_) => None,
    }
}

/// Return one Python string code point without materializing the complete string as characters.
///
/// ASCII strings, including the large text buffers used by the frozen I/O layer, support direct
/// byte indexing. Non-ASCII strings still index by Unicode code point to match Python semantics.
pub fn string_index(heap: &Heap, owner: &Value, index: &Value) -> Result<Option<char>, String> {
    let Some(text) = string_ref(heap, owner)? else {
        return Ok(None);
    };
    let index = index.as_int().ok_or("string index must be an integer")?;
    indexed_char(text.as_str(), index, text.is_ascii())
        .ok_or_else(|| "string index out of range".to_string())
        .map(Some)
}

/// Return a string's Python length without cloning its arena payload.
pub fn string_length(heap: &Heap, value: &Value) -> Result<Option<usize>, String> {
    let Some(text) = string_ref(heap, value)? else {
        return Ok(None);
    };
    Ok(Some(if text.is_ascii() {
        text.byte_len()
    } else {
        text.as_str().chars().count()
    }))
}

fn indexed_char(value: &str, index: i64, is_ascii: bool) -> Option<char> {
    if is_ascii {
        let index = normalize_index(value.len(), index)?;
        return value.as_bytes().get(index).copied().map(char::from);
    }

    let index = normalize_index(value.chars().count(), index)?;
    value.chars().nth(index)
}

fn normalize_index(length: usize, index: i64) -> Option<usize> {
    let index = if index < 0 {
        length.checked_sub(usize::try_from(index.unsigned_abs()).ok()?)?
    } else {
        usize::try_from(index).ok()?
    };
    (index < length).then_some(index)
}

pub fn bytes_value(heap: &Heap, value: &Value) -> Result<Option<Vec<u8>>, String> {
    let Some(id) = value.object_id() else {
        return Ok(None);
    };
    Ok(match heap.get(id)? {
        Object::Bytes(value) | Object::ByteArray(value) => Some(value.clone()),
        _ => None,
    })
}

pub fn exception_parts(heap: &Heap, value: &Value) -> Result<Option<(String, String)>, String> {
    let Some(id) = value.object_id() else {
        return Ok(None);
    };
    Ok(match heap.get(id)? {
        Object::Exception { kind, message } => Some((kind.clone(), message.clone())),
        _ => None,
    })
}

fn bigint_value<'a>(heap: &'a Heap, value: &Value) -> Option<&'a BigInt> {
    let id = value.object_id()?;
    match heap.get(id).ok()? {
        Object::BigInt(value) => Some(value),
        _ => None,
    }
}

fn render(heap: &Heap, value: &Value, active: &mut BTreeSet<ObjectId>) -> Result<String, String> {
    if let Some(value) = string_value(heap, value)? {
        return Ok(quote_string(&value));
    }
    if matches!(value.tag(), super::ValueTag::Int) {
        return Ok(value.immediate_int().expect("tag checked").to_string());
    }
    if let Some(value) = value.float_value() {
        if value.is_nan() {
            return Ok("nan".into());
        }
        if value.is_infinite() {
            return Ok(if value.is_sign_negative() {
                "-inf"
            } else {
                "inf"
            }
            .into());
        }
        let text = value.to_string();
        return Ok(if text.contains(['.', 'e', 'E']) {
            text
        } else {
            format!("{text}.0")
        });
    }
    if let Some(value) = value.bool_value() {
        return Ok(if value { "True" } else { "False" }.into());
    }
    if value.is_none() {
        return Ok("None".into());
    }
    if let Some(value) = value.native_value() {
        return Ok(value.repr());
    }
    if let Some((kind, message)) = exception_parts(heap, value)? {
        if message.is_empty() {
            return Ok(kind);
        } else {
            return Ok(format!("{kind}: {message}"));
        }
    }
    if let Some(id) = value.object_id() {
        if !active.insert(id) {
            return Ok(match heap.get(id)? {
                Object::String(_) => "<str ...>",
                Object::Bytes(_) => "<bytes ...>",
                Object::ByteArray(_) => "<bytearray ...>",
                Object::Exception { .. } => "<exception ...>",
                Object::List(_) => "[...]",
                Object::Tuple(_) => "(...)",
                Object::Slice { .. } => "slice(...)",
                Object::Dict(_) | Object::DefaultDict { .. } => "{...}",
                Object::Set(_) => "set(...)",
                Object::Range { .. } => "range(...)",
                Object::Function { .. } => "<function ...>",
                Object::Class { .. } => "<class ...>",
                Object::Instance { .. } => "<instance ...>",
                Object::DescriptorBoundMethod { .. } => "<bound method ...>",
                Object::Iterator { .. }
                | Object::SequenceIterator { .. }
                | Object::RangeIterator { .. } => "<iterator ...>",
                Object::CountIterator { .. } => "<iterator ...>",
                Object::CallableIterator { .. } => "<callable_iterator ...>",
                Object::Generator { .. } => "<generator ...>",
                Object::Module { .. } => "<module ...>",
                Object::ArrayStorage(_) => "<array storage ...>",
                Object::Array { .. } => "array(...)",
                Object::Regex { .. } => "re.compile(...) ",
                Object::Match { .. } => "<re.Match ...>",
                Object::ArgumentParser { .. } => "<argparse.ArgumentParser ...>",
                Object::Namespace { .. } => "<argparse.Namespace ...>",
                Object::EnumMember { .. } => "<enum member ...>",
                Object::RaisesContext { .. } => "<pytest.raises ...>",
                Object::Property { .. } => "<property ...>",
                Object::StaticMethod { .. } => "<staticmethod ...>",
                Object::ClassMethod { .. } => "<classmethod ...>",
                Object::Super { .. } => "<super ...>",
                Object::BigInt(_) => "<int ...>",
            }
            .into());
        }
        let rendered = match heap.get(id)? {
            Object::String(value) => quote_string(value),
            Object::Bytes(value) => quote_bytes(value),
            Object::ByteArray(value) => format!("bytearray({})", quote_bytes(value)),
            Object::Exception { kind, message } => {
                if message.is_empty() {
                    kind.clone()
                } else {
                    format!("{kind}: {message}")
                }
            }
            Object::List(values) => {
                format!("[{}]", render_values(heap, values, active)?.join(", "))
            }
            Object::Tuple(values) => {
                let values = render_values(heap, values, active)?;
                match values.as_slice() {
                    [] => "()".into(),
                    [only] => format!("({only},)"),
                    _ => format!("({})", values.join(", ")),
                }
            }
            Object::Slice { start, stop, step } => {
                format!("slice({start:?}, {stop:?}, {step:?})")
            }
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                let mut rendered = Vec::with_capacity(entries.len());
                for (key, value) in entries {
                    rendered.push(format!(
                        "{}: {}",
                        render(heap, key, active)?,
                        render(heap, value, active)?
                    ));
                }
                format!("{{{}}}", rendered.join(", "))
            }
            Object::Set(values) if values.is_empty() => "set()".into(),
            Object::Set(values) => {
                format!("{{{}}}", render_values(heap, values, active)?.join(", "))
            }
            Object::Range { start, stop, step } => {
                if *step == 1 && *start == 0 {
                    format!("range({stop})")
                } else if *step == 1 {
                    format!("range({start}, {stop})")
                } else {
                    format!("range({start}, {stop}, {step})")
                }
            }
            Object::Function { name, .. } => format!("<function {name}>"),
            Object::Class { name, .. } => format!("<class '{name}'>"),
            Object::Instance { class, payload, .. } => match payload {
                InstancePayload::Int(value) => value.to_string(),
                InstancePayload::Object => match heap.get(*class)? {
                    Object::Class { name, .. } => format!("<{name} object>"),
                    _ => return Err("instance has an invalid class".into()),
                },
            },
            Object::DescriptorBoundMethod { .. } => "<bound method>".into(),
            Object::Iterator { .. }
            | Object::SequenceIterator { .. }
            | Object::RangeIterator { .. } => "<iterator>".into(),
            Object::CountIterator { .. } => "<iterator>".into(),
            Object::CallableIterator { .. } => "<callable_iterator>".into(),
            Object::Generator { .. } => "<generator>".into(),
            Object::Module { name, .. } => format!("<module '{name}'>"),
            Object::ArrayStorage(_) => "<array storage>".into(),
            Object::Array { layout, dtype, .. } => {
                format!("array(shape={:?}, dtype={})", layout.shape, dtype.name())
            }
            Object::Regex { pattern, .. } => format!("re.compile({})", quote_string(pattern)),
            Object::Match {
                text, start, end, ..
            } => format!(
                "<re.Match object; span=({}, {}), match={}>",
                start,
                end,
                quote_string(text)
            ),
            Object::ArgumentParser { .. } => "<argparse.ArgumentParser>".into(),
            Object::Namespace { values } => {
                let rendered = values
                    .iter()
                    .map(|(name, value)| Ok(format!("{}={}", name, render(heap, value, active)?)))
                    .collect::<Result<Vec<_>, String>>()?;
                format!("Namespace({})", rendered.join(", "))
            }
            Object::EnumMember { name, .. } => {
                format!("<enum member {name}>")
            }
            Object::RaisesContext { .. } => "<pytest.raises>".into(),
            Object::Property { .. } => "<property>".into(),
            Object::StaticMethod { .. } => "<staticmethod>".into(),
            Object::ClassMethod { .. } => "<classmethod>".into(),
            Object::Super { .. } => "<super>".into(),
            Object::BigInt(value) => value.to_string(),
        };
        active.remove(&id);
        return Ok(rendered);
    }
    Err("invalid Python value tag".into())
}

fn render_values(
    heap: &Heap,
    values: &[Value],
    active: &mut BTreeSet<ObjectId>,
) -> Result<Vec<String>, String> {
    values
        .iter()
        .map(|value| render(heap, value, active))
        .collect()
}

pub fn truth(heap: &Heap, value: &Value) -> Result<bool, String> {
    if value.is_none() {
        return Ok(false);
    }
    if let Some(value) = value.bool_value() {
        return Ok(value);
    }
    if matches!(value.tag(), super::ValueTag::Int) {
        return Ok(value.immediate_int().expect("tag checked") != 0);
    }
    if let Some(value) = value.float_value() {
        return Ok(value != 0.0);
    }
    if let Some(value) = string_value(heap, value)? {
        return Ok(!value.is_empty());
    }
    if value.native_value().is_some() {
        return Ok(true);
    }
    let Some(id) = value.object_id() else {
        return Err("invalid Python value tag".into());
    };
    Ok(match heap.get(id)? {
        Object::String(value) => !value.is_empty(),
        Object::Bytes(value) => !value.is_empty(),
        Object::ByteArray(value) => !value.is_empty(),
        Object::Exception { .. } => true,
        Object::List(values) | Object::Tuple(values) | Object::Set(values) => !values.is_empty(),
        Object::Slice { .. } => true,
        Object::Dict(entries) | Object::DefaultDict { entries, .. } => !entries.is_empty(),
        Object::BigInt(value) => !value.is_zero(),
        Object::Range { start, stop, step } => {
            (*step > 0 && *start < *stop) || (*step < 0 && *start > *stop)
        }
        Object::Instance {
            payload: InstancePayload::Int(value),
            ..
        } => *value != 0,
        Object::Function { .. }
        | Object::Class { .. }
        | Object::Instance {
            payload: InstancePayload::Object,
            ..
        }
        | Object::DescriptorBoundMethod { .. }
        | Object::Iterator { .. }
        | Object::SequenceIterator { .. }
        | Object::RangeIterator { .. }
        | Object::CountIterator { .. }
        | Object::CallableIterator { .. }
        | Object::Generator { .. }
        | Object::Module { .. }
        | Object::ArrayStorage(_)
        | Object::Array { .. }
        | Object::Regex { .. }
        | Object::Match { .. }
        | Object::ArgumentParser { .. }
        | Object::Namespace { .. } => true,
        Object::EnumMember { .. } => true,
        Object::RaisesContext { .. } => true,
        Object::Property { .. }
        | Object::StaticMethod { .. }
        | Object::ClassMethod { .. }
        | Object::Super { .. } => true,
    })
}

pub fn equals(heap: &Heap, left: &Value, right: &Value) -> Result<bool, String> {
    if let Some(equal) = scalar_equality(heap, left, right)? {
        return Ok(equal);
    }
    equals_inner(heap, left, right, &mut BTreeSet::new())
}

/// Test runtime identity without requiring immediate values to carry pointers.
///
/// Immediate immutable values are canonical by representation: integers and strings use their
/// values, while floats use their exact bits so NaN remains identical to itself without becoming
/// equal to itself. Arena-backed values use their stable object handles.
pub fn identical(left: &Value, right: &Value) -> bool {
    left == right
}

fn equals_inner(
    heap: &Heap,
    left: &Value,
    right: &Value,
    active: &mut BTreeSet<(ObjectId, ObjectId)>,
) -> Result<bool, String> {
    if let Some(equal) = scalar_equality(heap, left, right)? {
        return Ok(equal);
    }
    match (left.object_id(), right.object_id()) {
        (Some(left), Some(right)) if left == right => Ok(true),
        (Some(left), Some(right)) => {
            if !active.insert((left, right)) {
                return Ok(true);
            }
            let result = match (heap.get(left)?, heap.get(right)?) {
                (Object::String(left), Object::String(right)) => left == right,
                (Object::Bytes(left), Object::Bytes(right)) => left == right,
                (Object::Bytes(left), Object::ByteArray(right))
                | (Object::ByteArray(left), Object::Bytes(right))
                | (Object::ByteArray(left), Object::ByteArray(right)) => left == right,
                (
                    Object::Exception {
                        kind: lk,
                        message: lm,
                    },
                    Object::Exception {
                        kind: rk,
                        message: rm,
                    },
                ) => lk == rk && lm == rm,
                (Object::List(left), Object::List(right))
                | (Object::Tuple(left), Object::Tuple(right)) => {
                    sequence_equal(heap, left, right, active)?
                }
                (
                    Object::Range {
                        start: left_start,
                        stop: left_stop,
                        step: left_step,
                    },
                    Object::Range {
                        start: right_start,
                        stop: right_stop,
                        step: right_step,
                    },
                ) => {
                    let count = |start: i64, stop: i64, step: i64| {
                        let start = i128::from(start);
                        let stop = i128::from(stop);
                        let step = i128::from(step);
                        if step > 0 && start < stop {
                            (stop - start - 1) / step + 1
                        } else if step < 0 && start > stop {
                            (start - stop - 1) / -step + 1
                        } else {
                            0
                        }
                    };
                    let left_count = count(*left_start, *left_stop, *left_step);
                    let right_count = count(*right_start, *right_stop, *right_step);
                    left_count == right_count
                        && (left_count == 0
                            || (left_start == right_start
                                && (left_count == 1 || left_step == right_step)))
                }
                (Object::Dict(left), Object::Dict(right))
                | (Object::Dict(left), Object::DefaultDict { entries: right, .. })
                | (Object::DefaultDict { entries: left, .. }, Object::Dict(right))
                | (
                    Object::DefaultDict { entries: left, .. },
                    Object::DefaultDict { entries: right, .. },
                ) => {
                    if left.len() != right.len() {
                        false
                    } else {
                        let mut all = true;
                        for (left_key, left_value) in left {
                            let mut found = false;
                            for (right_key, right_value) in right {
                                if identical(left_key, right_key)
                                    || equals_inner(heap, left_key, right_key, active)?
                                {
                                    found = identical(left_value, right_value)
                                        || equals_inner(heap, left_value, right_value, active)?;
                                    break;
                                }
                            }
                            if !found {
                                all = false;
                                break;
                            }
                        }
                        all
                    }
                }
                (Object::Set(left), Object::Set(right)) => {
                    if left.len() != right.len() {
                        false
                    } else {
                        let mut all = true;
                        for left_value in left {
                            let mut found = false;
                            for right_value in right {
                                if identical(left_value, right_value)
                                    || equals_inner(heap, left_value, right_value, active)?
                                {
                                    found = true;
                                    break;
                                }
                            }
                            if !found {
                                all = false;
                                break;
                            }
                        }
                        all
                    }
                }
                (Object::Function { .. }, Object::Function { .. })
                | (Object::Class { .. }, Object::Class { .. })
                | (Object::Instance { .. }, Object::Instance { .. })
                | (Object::DescriptorBoundMethod { .. }, Object::DescriptorBoundMethod { .. })
                | (Object::Iterator { .. }, Object::Iterator { .. })
                | (Object::CountIterator { .. }, Object::CountIterator { .. })
                | (Object::CallableIterator { .. }, Object::CallableIterator { .. })
                | (Object::Generator { .. }, Object::Generator { .. })
                | (Object::Module { .. }, Object::Module { .. })
                | (Object::Regex { .. }, Object::Regex { .. })
                | (Object::Match { .. }, Object::Match { .. })
                | (Object::ArgumentParser { .. }, Object::ArgumentParser { .. })
                | (Object::Namespace { .. }, Object::Namespace { .. }) => false,
                (Object::EnumMember { .. }, Object::EnumMember { .. }) => false,
                (Object::Property { .. }, Object::Property { .. })
                | (Object::StaticMethod { .. }, Object::StaticMethod { .. })
                | (Object::ClassMethod { .. }, Object::ClassMethod { .. })
                | (Object::Super { .. }, Object::Super { .. }) => false,
                _ => false,
            };
            active.remove(&(left, right));
            Ok(result)
        }
        _ => Ok(false),
    }
}

fn scalar_equality(heap: &Heap, left: &Value, right: &Value) -> Result<Option<bool>, String> {
    if let Some(left) = bigint_value(heap, left) {
        if let Some(right) = bigint_value(heap, right) {
            return Ok(Some(left == right));
        }
        if let Some(right) = int_value(heap, right) {
            return Ok(Some(left == &BigInt::from(right)));
        }
        if let Some(right) = right.float_value() {
            return Ok(Some(
                right.is_finite()
                    && right.fract() == 0.0
                    && BigInt::from_f64(right).is_some_and(|right| left == &right),
            ));
        }
    }
    if let Some(right) = bigint_value(heap, right) {
        if let Some(left) = int_value(heap, left) {
            return Ok(Some(&BigInt::from(left) == right));
        }
        if let Some(left) = left.float_value() {
            return Ok(Some(
                left.is_finite()
                    && left.fract() == 0.0
                    && BigInt::from_f64(left).is_some_and(|left| &left == right),
            ));
        }
    }
    if let Some(left) = int_value(heap, left) {
        if let Some(right) = int_value(heap, right) {
            return Ok(Some(left == right));
        }
        if let Some(right) = right.float_value() {
            return Ok(Some(
                right.is_finite()
                    && right.fract() == 0.0
                    && BigInt::from_f64(right).is_some_and(|right| BigInt::from(left) == right),
            ));
        }
    }
    if let (Some(left), Some(right)) = (left.float_value(), int_value(heap, right)) {
        return Ok(Some(
            left.is_finite()
                && left.fract() == 0.0
                && BigInt::from_f64(left).is_some_and(|left| left == BigInt::from(right)),
        ));
    }
    if left.is_none() || right.is_none() {
        return Ok(Some(left.is_none() && right.is_none()));
    }
    if let (Some(left), Some(right)) = (left.float_value(), right.float_value()) {
        return Ok(Some(left == right));
    }
    if let (Some(left), Some(right)) = (string_value(heap, left)?, string_value(heap, right)?) {
        return Ok(Some(left == right));
    }
    if let (Some(left), Some(right)) = (left.native_value(), right.native_value()) {
        return Ok(Some(left == right));
    }
    Ok(None)
}

fn sequence_equal(
    heap: &Heap,
    left: &[Value],
    right: &[Value],
    active: &mut BTreeSet<(ObjectId, ObjectId)>,
) -> Result<bool, String> {
    if left.len() != right.len() {
        return Ok(false);
    }
    for (left, right) in left.iter().zip(right) {
        if !identical(left, right) && !equals_inner(heap, left, right, active)? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn compare(heap: &Heap, left: &Value, right: &Value) -> Result<Ordering, String> {
    if let Some(left) = bigint_value(heap, left) {
        if let Some(right) = bigint_value(heap, right) {
            return Ok(left.cmp(right));
        }
        if let Some(right) = int_value(heap, right) {
            return Ok(left.cmp(&BigInt::from(right)));
        }
        if let Some(right) = right.float_value() {
            return compare_bigint_float(left, right);
        }
    }
    if let Some(right) = bigint_value(heap, right) {
        if let Some(left) = int_value(heap, left) {
            return Ok(BigInt::from(left).cmp(right));
        }
        if let Some(left) = left.float_value() {
            return compare_bigint_float(right, left).map(Ordering::reverse);
        }
    }
    if let Some(left) = int_value(heap, left) {
        if let Some(right) = int_value(heap, right) {
            return Ok(left.cmp(&right));
        }
        if let Some(right) = right.float_value() {
            return compare_bigint_float(&BigInt::from(left), right);
        }
    }
    if let (Some(left), Some(right)) = (left.float_value(), int_value(heap, right)) {
        return compare_bigint_float(&BigInt::from(right), left).map(Ordering::reverse);
    }
    if let (Some(left), Some(right)) = (left.float_value(), right.float_value()) {
        return left
            .partial_cmp(&right)
            .ok_or_else(|| "comparison with NaN is unordered".into());
    }
    if let (Some(left), Some(right)) = (string_value(heap, left)?, string_value(heap, right)?) {
        return Ok(left.cmp(&right));
    }
    if let (Some(left), Some(right)) = (bytes_value(heap, left)?, bytes_value(heap, right)?) {
        return Ok(left.cmp(&right));
    }
    match (left.object_id(), right.object_id()) {
        (Some(left), Some(right)) => match (heap.get(left)?, heap.get(right)?) {
            (Object::List(left), Object::List(right))
            | (Object::Tuple(left), Object::Tuple(right)) => sequence_compare(heap, left, right),
            _ => Err("objects do not define ordering".into()),
        },
        _ => Err("objects do not define ordering".into()),
    }
}

fn compare_bigint_float(integer: &BigInt, float: f64) -> Result<Ordering, String> {
    if float.is_nan() {
        return Err("comparison with NaN is unordered".into());
    }
    if float == f64::INFINITY {
        return Ok(Ordering::Less);
    }
    if float == f64::NEG_INFINITY {
        return Ok(Ordering::Greater);
    }
    let truncated = BigInt::from_f64(float).ok_or("float cannot be converted for comparison")?;
    let ordering = integer.cmp(&truncated);
    if ordering != Ordering::Equal || float.fract() == 0.0 {
        return Ok(ordering);
    }
    Ok(if float.is_sign_positive() {
        Ordering::Less
    } else {
        Ordering::Greater
    })
}

fn sequence_compare(heap: &Heap, left: &[Value], right: &[Value]) -> Result<Ordering, String> {
    for (left, right) in left.iter().zip(right) {
        if identical(left, right) || equals(heap, left, right)? {
            continue;
        }
        return compare(heap, left, right);
    }
    Ok(left.len().cmp(&right.len()))
}

pub fn contains(heap: &Heap, container: &Value, needle: &Value) -> Result<bool, String> {
    if let Some(container) = string_value(heap, container)? {
        let Some(needle) = string_value(heap, needle)? else {
            return Err("string containment requires a string operand".into());
        };
        return Ok(container.contains(&needle));
    }
    match container.object_id() {
        Some(id) => match heap.get(id)? {
            Object::String(_) | Object::Exception { .. } | Object::Slice { .. } => {
                Err("object is not a container".into())
            }
            Object::Bytes(value) | Object::ByteArray(value) => {
                let needle = int_value(heap, needle)
                    .and_then(|value| u8::try_from(value).ok())
                    .ok_or("bytes containment requires an integer in range(0, 256)")?;
                Ok(value.contains(&needle))
            }
            Object::List(values) | Object::Tuple(values) | Object::Set(values) => {
                for value in values {
                    if identical(value, needle) || equals(heap, value, needle)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Object::Range { start, stop, step } => {
                let value =
                    int_value(heap, needle).ok_or("range containment requires an integer")?;
                let within = (*step > 0 && value >= *start && value < *stop)
                    || (*step < 0 && value <= *start && value > *stop);
                Ok(within && (i128::from(value) - i128::from(*start)) % i128::from(*step) == 0)
            }
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                for (key, _) in entries {
                    if identical(key, needle) || equals(heap, key, needle)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Object::Function { .. }
            | Object::Class { .. }
            | Object::Instance { .. }
            | Object::DescriptorBoundMethod { .. }
            | Object::Iterator { .. }
            | Object::SequenceIterator { .. }
            | Object::RangeIterator { .. }
            | Object::CountIterator { .. }
            | Object::CallableIterator { .. }
            | Object::Generator { .. }
            | Object::Module { .. }
            | Object::ArrayStorage(_)
            | Object::Array { .. }
            | Object::Regex { .. }
            | Object::Match { .. }
            | Object::ArgumentParser { .. }
            | Object::Namespace { .. } => Err("object is not a container".into()),
            Object::EnumMember { .. } => Err("object is not a container".into()),
            Object::RaisesContext { .. } => Err("object is not a container".into()),
            Object::Property { .. }
            | Object::StaticMethod { .. }
            | Object::ClassMethod { .. }
            | Object::Super { .. } => Err("object is not a container".into()),
            Object::BigInt(_) => Err("object is not a container".into()),
        },
        None => Err("object is not a container".into()),
    }
}

fn quote_string(value: &str) -> String {
    format!(
        "'{}'",
        value
            .replace('\\', "\\\\")
            .replace('\'', "\\'")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    )
}

fn quote_bytes(value: &[u8]) -> String {
    let mut rendered = String::from("b'");
    for byte in value {
        match byte {
            b'\\' => rendered.push_str("\\\\"),
            b'\'' => rendered.push_str("\\'"),
            b'\n' => rendered.push_str("\\n"),
            b'\r' => rendered.push_str("\\r"),
            b'\t' => rendered.push_str("\\t"),
            0x20..=0x7e => rendered.push(char::from(*byte)),
            _ => rendered.push_str(&format!("\\x{byte:02x}")),
        }
    }
    rendered.push('\'');
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{Limits, Resources};

    #[test]
    fn dict_and_set_equality_are_order_independent() {
        let mut heap = Heap::default();
        let mut resources = Resources::new(Limits::unlimited());
        let first = heap
            .allocate(
                Object::Dict(
                    vec![
                        (Value::inline_string("a").unwrap(), Value::Int(1)),
                        (Value::inline_string("b").unwrap(), Value::Int(2)),
                    ]
                    .into(),
                ),
                &mut resources,
            )
            .unwrap();
        let second = heap
            .allocate(
                Object::Dict(
                    vec![
                        (Value::inline_string("b").unwrap(), Value::Int(2)),
                        (Value::inline_string("a").unwrap(), Value::Int(1)),
                    ]
                    .into(),
                ),
                &mut resources,
            )
            .unwrap();
        assert!(equals(&heap, &first, &second).unwrap());
    }

    #[test]
    fn nan_identity_is_distinct_from_nan_equality() {
        let mut heap = Heap::default();
        let mut resources = Resources::new(Limits::unlimited());
        let nan = Value::Float(f64::NAN);
        let first = heap
            .allocate(Object::List(vec![nan]), &mut resources)
            .unwrap();
        let second = heap
            .allocate(Object::List(vec![nan]), &mut resources)
            .unwrap();

        assert!(identical(&nan, &nan));
        assert!(!equals(&heap, &nan, &nan).unwrap());
        assert!(contains(&heap, &first, &nan).unwrap());
        assert!(equals(&heap, &first, &second).unwrap());
    }

    #[test]
    fn string_protocols_handle_inline_heap_ascii_and_unicode_values() {
        let mut heap = Heap::default();
        let mut resources = Resources::new(Limits::unlimited());
        let inline = Value::inline_string("café").unwrap();
        let ascii = heap
            .allocate(Object::String("a long ASCII string".into()), &mut resources)
            .unwrap();
        let unicode = heap
            .allocate(Object::String("☃ snow".into()), &mut resources)
            .unwrap();

        let inline_ref = string_ref(&heap, &inline).unwrap().unwrap();
        assert_eq!(inline_ref.as_str(), "café");
        assert!(!inline_ref.is_ascii());
        let ascii_ref = string_ref(&heap, &ascii).unwrap().unwrap();
        assert_eq!(ascii_ref.as_str(), "a long ASCII string");
        assert!(ascii_ref.is_ascii());
        assert_eq!(
            string_index(&heap, &inline, &Value::Int(3)).unwrap(),
            Some('é')
        );
        assert_eq!(
            string_index(&heap, &inline, &Value::Int(-4)).unwrap(),
            Some('c')
        );
        assert_eq!(
            string_index(&heap, &ascii, &Value::Int(7)).unwrap(),
            Some('A')
        );
        assert_eq!(
            string_index(&heap, &unicode, &Value::Int(-6)).unwrap(),
            Some('☃')
        );
        assert_eq!(string_length(&heap, &inline).unwrap(), Some(4));
        assert_eq!(string_length(&heap, &ascii).unwrap(), Some(19));
        assert_eq!(string_length(&heap, &unicode).unwrap(), Some(6));
        assert_eq!(
            string_index(&heap, &Value::Int(1), &Value::Int(0)).unwrap(),
            None
        );
        assert_eq!(
            string_index(&heap, &ascii, &Value::Int(99)).unwrap_err(),
            "string index out of range"
        );
        assert_eq!(
            string_index(&heap, &ascii, &Value::None).unwrap_err(),
            "string index must be an integer"
        );
    }
}
