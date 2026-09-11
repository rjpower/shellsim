//! Central Python value protocols for truth, representation, equality, ordering, and containment.
//!
//! Container algorithms are deliberately linear. Correct behavior and one auditable dispatch
//! point matter more than asymptotic performance for the bounded evaluator.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use num_bigint::BigInt;
use num_traits::{FromPrimitive, Zero};

use super::heap::{Heap, InstancePayload, Object, ObjectId};
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

pub fn string_value(heap: &Heap, value: &Value) -> Result<Option<String>, String> {
    if let Some(value) = value.inline_string_value() {
        return Ok(Some(value));
    }
    let Some(id) = value.object_id() else {
        return Ok(None);
    };
    Ok(match heap.get(id)? {
        Object::String(value) => Some(value.clone()),
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
    if value.tag() == super::ValueTag::Int {
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
                Object::Exception { .. } => "<exception ...>",
                Object::List(_) => "[...]",
                Object::Tuple(_) => "(...)",
                Object::Dict(_) | Object::DefaultDict { .. } => "{...}",
                Object::Set(_) => "set(...)",
                Object::Function { .. } => "<function ...>",
                Object::Class { .. } => "<class ...>",
                Object::Instance { .. } => "<instance ...>",
                Object::DescriptorBoundMethod { .. } => "<bound method ...>",
                Object::Iterator { .. } => "<iterator ...>",
                Object::CountIterator { .. } => "<iterator ...>",
                Object::Generator { .. } => "<generator ...>",
                Object::Module { .. } => "<module ...>",
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
            Object::Iterator { .. } => "<iterator>".into(),
            Object::CountIterator { .. } => "<iterator>".into(),
            Object::Generator { .. } => "<generator>".into(),
            Object::Module { name, .. } => format!("<module '{name}'>"),
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
    if value.tag() == super::ValueTag::Int {
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
        Object::Exception { .. } => true,
        Object::List(values) | Object::Tuple(values) | Object::Set(values) => !values.is_empty(),
        Object::Dict(entries) | Object::DefaultDict { entries, .. } => !entries.is_empty(),
        Object::BigInt(value) => !value.is_zero(),
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
        | Object::CountIterator { .. }
        | Object::Generator { .. }
        | Object::Module { .. }
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
    if let Some(left) = bigint_value(heap, left) {
        if let Some(right) = bigint_value(heap, right) {
            return Ok(left == right);
        }
        if let Some(right) = int_value(heap, right) {
            return Ok(left == &BigInt::from(right));
        }
        if let Some(right) = right.float_value() {
            return Ok(right.is_finite()
                && right.fract() == 0.0
                && BigInt::from_f64(right).is_some_and(|right| left == &right));
        }
    }
    if let Some(right) = bigint_value(heap, right) {
        if let Some(left) = int_value(heap, left) {
            return Ok(&BigInt::from(left) == right);
        }
        if let Some(left) = left.float_value() {
            return Ok(left.is_finite()
                && left.fract() == 0.0
                && BigInt::from_f64(left).is_some_and(|left| &left == right));
        }
    }
    if let Some(left) = int_value(heap, left) {
        if let Some(right) = int_value(heap, right) {
            return Ok(left == right);
        }
        if let Some(right) = right.float_value() {
            return Ok(right.is_finite()
                && right.fract() == 0.0
                && BigInt::from_f64(right).is_some_and(|right| BigInt::from(left) == right));
        }
    }
    if let (Some(left), Some(right)) = (left.float_value(), int_value(heap, right)) {
        return Ok(left.is_finite()
            && left.fract() == 0.0
            && BigInt::from_f64(left).is_some_and(|left| left == BigInt::from(right)));
    }
    if left.is_none() || right.is_none() {
        return Ok(left.is_none() && right.is_none());
    }
    if let (Some(left), Some(right)) = (left.float_value(), right.float_value()) {
        return Ok(left == right);
    }
    if let (Some(left), Some(right)) = (string_value(heap, left)?, string_value(heap, right)?) {
        return Ok(left == right);
    }
    if let (Some(left), Some(right)) = (left.native_value(), right.native_value()) {
        return Ok(left == right);
    }
    match (left.object_id(), right.object_id()) {
        (Some(left), Some(right)) if left == right => Ok(true),
        (Some(left), Some(right)) => {
            if !active.insert((left, right)) {
                return Ok(true);
            }
            let result = match (heap.get(left)?, heap.get(right)?) {
                (Object::String(left), Object::String(right)) => left == right,
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
            Object::String(_) | Object::Exception { .. } => Err("object is not a container".into()),
            Object::List(values) | Object::Tuple(values) | Object::Set(values) => {
                for value in values {
                    if identical(value, needle) || equals(heap, value, needle)? {
                        return Ok(true);
                    }
                }
                Ok(false)
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
            | Object::CountIterator { .. }
            | Object::Generator { .. }
            | Object::Module { .. }
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
                Object::Dict(vec![
                    (Value::inline_string("a").unwrap(), Value::Int(1)),
                    (Value::inline_string("b").unwrap(), Value::Int(2)),
                ]),
                &mut resources,
            )
            .unwrap();
        let second = heap
            .allocate(
                Object::Dict(vec![
                    (Value::inline_string("b").unwrap(), Value::Int(2)),
                    (Value::inline_string("a").unwrap(), Value::Int(1)),
                ]),
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
}
