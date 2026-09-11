//! Central Python value protocols for truth, representation, equality, ordering, and containment.
//!
//! Container algorithms are deliberately linear. Correct behavior and one auditable dispatch
//! point matter more than asymptotic performance for the bounded evaluator.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use super::heap::{Heap, InstancePayload, Object, ObjectId};
use super::Value;

pub fn display(heap: &Heap, value: &Value) -> Result<String, String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Exception { message, kind } => Ok(if message.is_empty() {
            kind.clone()
        } else {
            message.clone()
        }),
        _ => repr(heap, value),
    }
}

pub fn repr(heap: &Heap, value: &Value) -> Result<String, String> {
    render(heap, value, &mut BTreeSet::new())
}

/// Return the integer payload of an immediate integer or an `int` subclass instance.
pub fn int_value(heap: &Heap, value: &Value) -> Option<i64> {
    match value {
        Value::Int(value) => Some(*value),
        Value::Bool(value) => Some(i64::from(*value)),
        Value::Object(id) => match heap.get(*id).ok()? {
            Object::Instance {
                payload: InstancePayload::Int(value),
                ..
            } => Some(*value),
            _ => None,
        },
        _ => None,
    }
}

fn render(heap: &Heap, value: &Value, active: &mut BTreeSet<ObjectId>) -> Result<String, String> {
    match value {
        Value::String(value) => Ok(quote_string(value)),
        Value::Int(value) => Ok(value.to_string()),
        Value::Float(value) => {
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
            Ok(if text.contains(['.', 'e', 'E']) {
                text
            } else {
                format!("{text}.0")
            })
        }
        Value::Bool(value) => Ok(if *value { "True" } else { "False" }.into()),
        Value::None => Ok("None".into()),
        Value::Native(_) => Ok("<native object>".into()),
        Value::Exception { kind, message } => {
            if message.is_empty() {
                Ok(kind.clone())
            } else {
                Ok(format!("{kind}: {message}"))
            }
        }
        Value::Object(id) => {
            if !active.insert(*id) {
                return Ok(match heap.get(*id)? {
                    Object::List(_) => "[...]",
                    Object::Tuple(_) => "(...)",
                    Object::Dict(_) | Object::DefaultDict { .. } => "{...}",
                    Object::Set(_) => "set(...)",
                    Object::Function { .. } => "<function ...>",
                    Object::Class { .. } => "<class ...>",
                    Object::Instance { .. } => "<instance ...>",
                    Object::PythonBoundMethod { .. } => "<bound method ...>",
                    Object::NativeBoundMethod { .. } => "<bound native method ...>",
                    Object::Iterator { .. } => "<iterator ...>",
                    Object::CountIterator { .. } => "<iterator ...>",
                    Object::Generator { .. } => "<generator ...>",
                    Object::BoundMethod { .. } => "<bound method ...>",
                    Object::Module { .. } => "<module ...>",
                    Object::Regex { .. } => "re.compile(...) ",
                    Object::Match { .. } => "<re.Match ...>",
                    Object::ArgumentParser { .. } => "<argparse.ArgumentParser ...>",
                    Object::Namespace { .. } => "<argparse.Namespace ...>",
                    Object::EnumMember { .. } => "<enum member ...>",
                    Object::RaisesContext { .. } => "<pytest.raises ...>",
                }
                .into());
            }
            let rendered = match heap.get(*id)? {
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
                Object::PythonBoundMethod { .. } => "<bound method>".into(),
                Object::NativeBoundMethod { .. } => "<bound native method>".into(),
                Object::Iterator { .. } => "<iterator>".into(),
                Object::CountIterator { .. } => "<iterator>".into(),
                Object::Generator { .. } => "<generator>".into(),
                Object::BoundMethod { .. } => "<bound method>".into(),
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
                        .map(|(name, value)| {
                            Ok(format!("{}={}", name, render(heap, value, active)?))
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    format!("Namespace({})", rendered.join(", "))
                }
                Object::EnumMember { name, .. } => {
                    format!("<enum member {name}>")
                }
                Object::RaisesContext { .. } => "<pytest.raises>".into(),
            };
            active.remove(id);
            Ok(rendered)
        }
    }
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
    Ok(match value {
        Value::None => false,
        Value::Bool(value) => *value,
        Value::Int(value) => *value != 0,
        Value::Float(value) => *value != 0.0,
        Value::String(value) => !value.is_empty(),
        Value::Native(_) => true,
        Value::Exception { .. } => true,
        Value::Object(id) => match heap.get(*id)? {
            Object::List(values) | Object::Tuple(values) | Object::Set(values) => {
                !values.is_empty()
            }
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => !entries.is_empty(),
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
            | Object::PythonBoundMethod { .. }
            | Object::NativeBoundMethod { .. }
            | Object::Iterator { .. }
            | Object::CountIterator { .. }
            | Object::Generator { .. }
            | Object::BoundMethod { .. }
            | Object::Module { .. }
            | Object::Regex { .. }
            | Object::Match { .. }
            | Object::ArgumentParser { .. }
            | Object::Namespace { .. } => true,
            Object::EnumMember { .. } => true,
            Object::RaisesContext { .. } => true,
        },
    })
}

pub fn equals(heap: &Heap, left: &Value, right: &Value) -> Result<bool, String> {
    equals_inner(heap, left, right, &mut BTreeSet::new())
}

fn equals_inner(
    heap: &Heap,
    left: &Value,
    right: &Value,
    active: &mut BTreeSet<(ObjectId, ObjectId)>,
) -> Result<bool, String> {
    if let Some(left) = int_value(heap, left) {
        if let Some(right) = int_value(heap, right) {
            return Ok(left == right);
        }
        if let Value::Float(right) = right {
            return Ok(left as f64 == *right);
        }
    }
    if let (Value::Float(left), Some(right)) = (left, int_value(heap, right)) {
        return Ok(*left == right as f64);
    }
    match (left, right) {
        (Value::None, Value::None) => Ok(true),
        (Value::Bool(left), Value::Bool(right)) => Ok(left == right),
        (Value::Int(left), Value::Int(right)) => Ok(left == right),
        (Value::Float(left), Value::Float(right)) => Ok(left == right),
        (Value::Int(left), Value::Float(right)) => Ok(*left as f64 == *right),
        (Value::Float(left), Value::Int(right)) => Ok(*left == *right as f64),
        (Value::Bool(left), Value::Int(right)) | (Value::Int(right), Value::Bool(left)) => {
            Ok(i64::from(*left) == *right)
        }
        (Value::String(left), Value::String(right)) => Ok(left == right),
        (Value::Native(left), Value::Native(right)) => Ok(left == right),
        (
            Value::Exception {
                kind: lk,
                message: lm,
            },
            Value::Exception {
                kind: rk,
                message: rm,
            },
        ) => Ok(lk == rk && lm == rm),
        (Value::Object(left), Value::Object(right)) if left == right => Ok(true),
        (Value::Object(left), Value::Object(right)) => {
            if !active.insert((*left, *right)) {
                return Ok(true);
            }
            let result = match (heap.get(*left)?, heap.get(*right)?) {
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
                                if equals_inner(heap, left_key, right_key, active)? {
                                    found = equals_inner(heap, left_value, right_value, active)?;
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
                                if equals_inner(heap, left_value, right_value, active)? {
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
                | (Object::PythonBoundMethod { .. }, Object::PythonBoundMethod { .. })
                | (Object::Iterator { .. }, Object::Iterator { .. })
                | (Object::CountIterator { .. }, Object::CountIterator { .. })
                | (Object::Generator { .. }, Object::Generator { .. })
                | (Object::BoundMethod { .. }, Object::BoundMethod { .. })
                | (Object::Module { .. }, Object::Module { .. })
                | (Object::Regex { .. }, Object::Regex { .. })
                | (Object::Match { .. }, Object::Match { .. })
                | (Object::ArgumentParser { .. }, Object::ArgumentParser { .. })
                | (Object::Namespace { .. }, Object::Namespace { .. }) => false,
                (Object::EnumMember { .. }, Object::EnumMember { .. }) => false,
                _ => false,
            };
            active.remove(&(*left, *right));
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
        if !equals_inner(heap, left, right, active)? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn compare(heap: &Heap, left: &Value, right: &Value) -> Result<Ordering, String> {
    if let Some(left) = int_value(heap, left) {
        if let Some(right) = int_value(heap, right) {
            return Ok(left.cmp(&right));
        }
        if let Value::Float(right) = right {
            return (left as f64)
                .partial_cmp(right)
                .ok_or_else(|| "comparison with NaN is unordered".into());
        }
    }
    if let (Value::Float(left), Some(right)) = (left, int_value(heap, right)) {
        return left
            .partial_cmp(&(right as f64))
            .ok_or_else(|| "comparison with NaN is unordered".into());
    }
    match (left, right) {
        (Value::Int(left), Value::Int(right)) => Ok(left.cmp(right)),
        (Value::Float(left), Value::Float(right)) => left
            .partial_cmp(right)
            .ok_or_else(|| "comparison with NaN is unordered".into()),
        (Value::Int(left), Value::Float(right)) => (*left as f64)
            .partial_cmp(right)
            .ok_or_else(|| "comparison with NaN is unordered".into()),
        (Value::Float(left), Value::Int(right)) => left
            .partial_cmp(&(*right as f64))
            .ok_or_else(|| "comparison with NaN is unordered".into()),
        (Value::String(left), Value::String(right)) => Ok(left.cmp(right)),
        (Value::Object(left), Value::Object(right)) => {
            match (heap.get(*left)?, heap.get(*right)?) {
                (Object::List(left), Object::List(right))
                | (Object::Tuple(left), Object::Tuple(right)) => {
                    sequence_compare(heap, left, right)
                }
                _ => Err("objects do not define ordering".into()),
            }
        }
        _ => Err("objects do not define ordering".into()),
    }
}

fn sequence_compare(heap: &Heap, left: &[Value], right: &[Value]) -> Result<Ordering, String> {
    for (left, right) in left.iter().zip(right) {
        if equals(heap, left, right)? {
            continue;
        }
        return compare(heap, left, right);
    }
    Ok(left.len().cmp(&right.len()))
}

pub fn contains(heap: &Heap, container: &Value, needle: &Value) -> Result<bool, String> {
    match container {
        Value::String(container) => {
            let Value::String(needle) = needle else {
                return Err("string containment requires a string operand".into());
            };
            Ok(container.contains(needle))
        }
        Value::Object(id) => match heap.get(*id)? {
            Object::List(values) | Object::Tuple(values) | Object::Set(values) => {
                for value in values {
                    if equals(heap, value, needle)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                for (key, _) in entries {
                    if equals(heap, key, needle)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Object::Function { .. }
            | Object::Class { .. }
            | Object::Instance { .. }
            | Object::PythonBoundMethod { .. }
            | Object::NativeBoundMethod { .. }
            | Object::Iterator { .. }
            | Object::CountIterator { .. }
            | Object::Generator { .. }
            | Object::BoundMethod { .. }
            | Object::Module { .. }
            | Object::Regex { .. }
            | Object::Match { .. }
            | Object::ArgumentParser { .. }
            | Object::Namespace { .. } => Err("object is not a container".into()),
            Object::EnumMember { .. } => Err("object is not a container".into()),
            Object::RaisesContext { .. } => Err("object is not a container".into()),
        },
        _ => Err("object is not a container".into()),
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
                    (Value::String("a".into()), Value::Int(1)),
                    (Value::String("b".into()), Value::Int(2)),
                ]),
                &mut resources,
            )
            .unwrap();
        let second = heap
            .allocate(
                Object::Dict(vec![
                    (Value::String("b".into()), Value::Int(2)),
                    (Value::String("a".into()), Value::Int(1)),
                ]),
                &mut resources,
            )
            .unwrap();
        assert!(equals(&heap, &first, &second).unwrap());
    }
}
