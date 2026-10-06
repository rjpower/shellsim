//! Heap-only Python value protocols: representation, truth, equality, ordering and containment.
//!
//! These read builtin payloads directly and never run Python code, so the VM uses them as the
//! fallback once no slot on the value's type claims the operation, and the REPL uses them to
//! print values. Container algorithms are deliberately linear: correct behavior and one
//! auditable dispatch point matter more than asymptotic performance for the bounded evaluator.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use num_bigint::BigInt;
use num_traits::{FromPrimitive, Zero};

use super::exception_types::{exception_args, exception_type_name};
use super::heap::{Heap, NamespaceTarget, Object, Ref};
use super::number::{bigint_value, int_value};
use super::scopes;
use super::string::{bytes_ref, bytes_value, quote_bytes, quote_string, string_value};
use super::{ReplState, Value};
use crate::python::error::{PyError, PyResult};

/// The object behind `value`, or `None` for an immediate.
fn object(heap: &Heap, value: Value) -> PyResult<Option<&Object>> {
    if value.is_object() {
        heap.get(value).map(Some)
    } else {
        Ok(None)
    }
}

fn alias_item_repr(state: &ReplState, value: &Ref, active: &mut BTreeSet<u32>) -> PyResult<String> {
    let heap = &state.heap;
    let value = heap.value(value);
    if let Some(super::vm::NativeValue::BuiltinType(builtin)) = value.native_value() {
        return Ok(builtin.name().into());
    }
    if let Some(Object::Class(class_object)) = object(heap, value)? {
        return Ok(class_object.name.clone());
    }
    render(state, value, active)
}

/// `str()` without user `__str__` dispatch. Rendering takes the interpreter state rather than
/// the heap alone because a user exception's message comes from its `args` instance attribute,
/// which the shape and symbol tables resolve.
pub fn display(state: &ReplState, value: Value) -> PyResult<String> {
    display_inner(state, value, &mut BTreeSet::new())
}

/// [`display`] with the exceptions whose message is being rendered on the current path. An
/// exception whose `args` reach itself, directly or through another exception, would otherwise
/// recurse without bound; CPython raises `RecursionError` there too.
fn display_inner(state: &ReplState, value: Value, active: &mut BTreeSet<u32>) -> PyResult<String> {
    let heap = &state.heap;
    if let Some(value) = string_value(heap, value)? {
        return Ok(value);
    }
    let Some((base, args)) = exception_args(state, value)? else {
        return render(state, value, active);
    };
    let id = heap
        .identity(value)?
        .ok_or("exception is not a heap object")?;
    if !active.insert(id) || active.len() >= MAX_RENDER_DEPTH {
        return Err(PyError::exception("RecursionError", RECURSION_IN_STR));
    }
    let message = exception_message(state, base, &args, active);
    active.remove(&id);
    message
}

/// The error for a `str()` whose `args` nest deeper than [`MAX_RENDER_DEPTH`] or reach the
/// exception itself.
const RECURSION_IN_STR: &str =
    "maximum recursion depth exceeded while getting the str of an object";

/// The stand-in for a CPython object address in default reprs such as
/// `<object object at 0x7f0000000010>`: derived from the object's stable identity, so it is
/// deterministic across runs and distinct for live objects.
pub fn address(identity: u32) -> String {
    format!("0x{:x}", address_value(identity))
}

/// The numeric stand-in address of a heap object; see [`address`].
pub fn address_value(identity: u32) -> u64 {
    0x7f00_0000_0000_u64 + u64::from(identity) * 16
}

pub fn repr(state: &ReplState, value: Value) -> PyResult<String> {
    render(state, value, &mut BTreeSet::new())
}

/// The class name and `str()` of an exception instance, builtin or user-defined.
pub fn exception_parts(state: &ReplState, value: Value) -> PyResult<Option<(String, String)>> {
    let Some(name) = exception_type_name(state, value)? else {
        return Ok(None);
    };
    let (base, args) = exception_args(state, value)?.ok_or("exception lost its type")?;
    let mut active = BTreeSet::new();
    if let Some(id) = state.heap.identity(value)? {
        active.insert(id);
    }
    let message = exception_message(state, base, &args, &mut active)?;
    Ok(Some((name, message)))
}

/// `str()` of an exception of class `kind` (a builtin exception class name) with arguments
/// `args`: empty without arguments, the `str()` of a single argument, whose `repr()` a
/// `KeyError` shows because the argument is a key, and otherwise the `repr()` of the tuple.
///
/// `active` holds the exceptions on the current rendering path, so a self-referential `args`
/// stops instead of recursing without bound.
fn exception_message(
    state: &ReplState,
    kind: &str,
    args: &[Value],
    active: &mut BTreeSet<u32>,
) -> PyResult<String> {
    match args {
        [] => Ok(String::new()),
        [only] if super::exception_types::exception_is_subclass(kind, "KeyError") => {
            render(state, *only, active)
        }
        [only] => display_inner(state, *only, active),
        [errno, strerror, rest @ ..]
            if rest.len() <= 3
                && super::exception_types::exception_is_subclass(kind, "OSError") =>
        {
            let mut message = format!(
                "[Errno {}] {}",
                display_inner(state, *errno, active)?,
                display_inner(state, *strerror, active)?
            );
            if let Some(filename) = rest.first().filter(|value| !value.is_none()) {
                message.push_str(": ");
                message.push_str(&render(state, *filename, active)?);
                if let Some(filename2) = rest.get(2).filter(|value| !value.is_none()) {
                    message.push_str(" -> ");
                    message.push_str(&render(state, *filename2, active)?);
                }
            }
            Ok(message)
        }
        _ => {
            let values = args
                .iter()
                .map(|value| render(state, *value, active))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(format!("({})", values.join(", ")))
        }
    }
}

/// Nesting bound for heap-only rendering, matching the VM's call-depth limit.
const MAX_RENDER_DEPTH: usize = 256;

/// `active` holds the identities of the objects on the current rendering path.
fn render(state: &ReplState, value: Value, active: &mut BTreeSet<u32>) -> PyResult<String> {
    crate::stack::grow(|| render_inner(state, value, active))
}

fn render_inner(state: &ReplState, value: Value, active: &mut BTreeSet<u32>) -> PyResult<String> {
    let heap = &state.heap;
    if let Some(value) = string_value(heap, value)? {
        return Ok(quote_string(&value));
    }
    if let (Some(integer), None) = (value.immediate_int(), value.bool_value()) {
        return Ok(integer.to_string());
    }
    if let Some(value) = value.float_value() {
        return Ok(super::float_text::repr(value));
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
    if let Some(id) = heap.identity(value)? {
        // `active` holds the objects on the current path, so its size is the nesting depth.
        if !active.contains(&id) && active.len() >= MAX_RENDER_DEPTH {
            return Err(PyError::exception(
                "RecursionError",
                "maximum recursion depth exceeded while getting the repr of an object",
            ));
        }
        if !active.insert(id) {
            return Ok(match heap.get(value)? {
                Object::Bare => "<object ...>",
                Object::String(_) => "<str ...>",
                Object::Bytes(_) => "<bytes ...>",
                Object::ByteArray(_) => "<bytearray ...>",
                Object::Exception(_) => "<exception ...>",
                Object::List(_) => "[...]",
                Object::Tuple(_) => "(...)",
                Object::Slice { .. } => "slice(...)",
                Object::Dict(_) | Object::DefaultDict { .. } => "{...}",
                Object::Set(_) => "set(...)",
                Object::FrozenSet(_) => "frozenset(...)",
                Object::Range { .. } => "range(...)",
                Object::Function { .. } => "<function ...>",
                Object::Class { .. } => "<class ...>",
                Object::DescriptorBoundMethod { .. } => "<bound method ...>",
                Object::GenericAlias { .. } => "<generic alias ...>",
                Object::Iterator { .. }
                | Object::SequenceIterator { .. }
                | Object::ReverseIterator { .. }
                | Object::RangeIterator { .. } => "<iterator ...>",
                Object::CountIterator { .. } | Object::StreamIterator { .. } => "<iterator ...>",
                Object::CallableIterator { .. } => "<callable_iterator ...>",
                Object::Generator { .. } => "<generator ...>",
                Object::Module { .. } => "<module ...>",
                Object::NamespaceDict(_) => "{...}",
                Object::DictView { .. } => "<dict view ...>",
                Object::MappingProxy(_) => "mappingproxy(...)",
                Object::WideValue { .. } => "<value ...>",
                Object::Native(_) => "<native ...>",
                Object::Property { .. } => "<property ...>",
                Object::StaticMethod { .. } => "<staticmethod ...>",
                Object::ClassMethod { .. } => "<classmethod ...>",
                Object::Super { .. } => "<super ...>",
                Object::Scope(_) => "<scope ...>",
                Object::BigInt(_) => "<int ...>",
                Object::Float(_) => "<float ...>",
                Object::Complex { .. } => "<complex ...>",
            }
            .into());
        }
        let rendered = match heap.get(value)? {
            Object::Bare => match state.types.instance_class(heap, value)? {
                None => format!("<object object at {}>", address(id)),
                Some(class) => match heap.get(class)? {
                    Object::Class(class_object) => format!("<{} object>", class_object.name),
                    _ => return Err("instance has an invalid class".into()),
                },
            },
            Object::Exception(args) => {
                let name = state.types.get(heap.type_id(value)?)?.name.clone();
                format!("{name}({})", render_values(state, args, active)?.join(", "))
            }
            Object::Float(value) => super::float_text::repr(*value),
            Object::String(value) => quote_string(value),
            Object::Bytes(value) => quote_bytes(value),
            Object::ByteArray(value) => format!("bytearray({})", quote_bytes(value)),
            Object::List(values) => {
                format!("[{}]", render_values(state, values, active)?.join(", "))
            }
            Object::Tuple(values) => {
                let values = render_values(state, values, active)?;
                match values.as_slice() {
                    [] => "()".into(),
                    [only] => format!("({only},)"),
                    _ => format!("({})", values.join(", ")),
                }
            }
            Object::Slice { start, stop, step } => {
                format!(
                    "slice({})",
                    render_values(state, [start, stop, step], active)?.join(", ")
                )
            }
            Object::Dict(entries) | Object::DefaultDict { entries, .. } => {
                let mut rendered = Vec::with_capacity(entries.len());
                for (key, value) in entries {
                    rendered.push(format!(
                        "{}: {}",
                        render(state, heap.value(key), active)?,
                        render(state, heap.value(value), active)?
                    ));
                }
                format!("{{{}}}", rendered.join(", "))
            }
            Object::Set(values) if values.is_empty() => "set()".into(),
            Object::Set(values) => {
                format!("{{{}}}", render_values(state, values, active)?.join(", "))
            }
            Object::FrozenSet(values) if values.is_empty() => "frozenset()".into(),
            Object::FrozenSet(values) => format!(
                "frozenset({{{}}})",
                render_values(state, values, active)?.join(", ")
            ),
            Object::Range { start, stop, step } => {
                if *step == 1 && *start == 0 {
                    format!("range({stop})")
                } else if *step == 1 {
                    format!("range({start}, {stop})")
                } else {
                    format!("range({start}, {stop}, {step})")
                }
            }
            Object::Function(function) => format!("<function {}>", function.name),
            Object::Class(class_object) => match class_object.attributes.get("__module__") {
                Some(module) => match string_value(heap, heap.value(module))? {
                    Some(module) if module != "builtins" => {
                        format!("<class '{module}.{}'>", class_object.name)
                    }
                    _ => format!("<class '{}'>", class_object.name),
                },
                None => format!("<class '{}'>", class_object.name),
            },
            Object::DescriptorBoundMethod { .. } => "<bound method>".into(),
            Object::GenericAlias { origin, arguments } => {
                let origin = alias_item_repr(state, origin, active)?;
                let arguments = arguments
                    .iter()
                    .map(|argument| alias_item_repr(state, argument, active))
                    .collect::<Result<Vec<_>, _>>()?;
                format!("{origin}[{}]", arguments.join(", "))
            }
            Object::Iterator { .. }
            | Object::SequenceIterator { .. }
            | Object::ReverseIterator { .. }
            | Object::RangeIterator { .. } => "<iterator>".into(),
            Object::CountIterator { .. } | Object::StreamIterator { .. } => "<iterator>".into(),
            Object::CallableIterator { .. } => "<callable_iterator>".into(),
            Object::Generator { .. } => "<generator>".into(),
            Object::Module { name, .. } => format!("<module '{name}'>"),
            Object::NamespaceDict(NamespaceTarget::Scope(scope)) => {
                let mut entries = scopes::values(heap, heap.value(scope))?
                    .into_iter()
                    .collect::<Vec<_>>();
                entries.sort_by(|(left, _), (right, _)| left.cmp(right));
                let mut rendered = Vec::with_capacity(entries.len());
                for (name, value) in entries {
                    rendered.push(format!(
                        "{}: {}",
                        quote_string(&name),
                        render(state, value, active)?
                    ));
                }
                format!("{{{}}}", rendered.join(", "))
            }
            Object::NamespaceDict(NamespaceTarget::Instance(instance)) => {
                let mut rendered = Vec::new();
                let entries =
                    state
                        .shapes
                        .attribute_values(heap, &state.symbols, heap.value(instance))?;
                for (name, value) in entries {
                    rendered.push(format!(
                        "{}: {}",
                        quote_string(&name),
                        render(state, value, active)?
                    ));
                }
                format!("{{{}}}", rendered.join(", "))
            }
            // The REPL/script table isn't reachable from a bare `&Heap`. `Vm::repr_nested`
            // (via `repr_namespace_dict`) covers every ordinary `repr()`, `str()`, or `print()`
            // call, so this generic fallback is only reached by the interactive REPL auto-printing
            // a bare expression, which already skips a user `__repr__` for every other type too.
            Object::NamespaceDict(NamespaceTarget::Repl) => "<globals>".to_string(),
            // `Vm::repr_nested` renders views and proxies from their mapping's entries; the
            // heap alone cannot read every mapping they may view.
            Object::DictView { .. } => "<dict view>".to_string(),
            Object::MappingProxy(_) => "mappingproxy(...)".to_string(),
            Object::WideValue { .. } => "<value>".into(),
            Object::Native(native) => {
                native.repr(&mut |slot| render(state, heap.value(slot), active))?
            }
            Object::Property { .. } => "<property>".into(),
            Object::StaticMethod { .. } => "<staticmethod>".into(),
            Object::ClassMethod { .. } => "<classmethod>".into(),
            Object::Super { .. } => "<super>".into(),
            Object::Scope(_) => "<scope>".into(),
            Object::BigInt(value) => value.to_string(),
            Object::Complex { real, imag } => {
                super::complex::repr(super::complex::Complex::new(*real, *imag))
            }
        };
        active.remove(&id);
        return Ok(rendered);
    }
    Err("invalid Python value tag".into())
}

fn render_values<'a>(
    state: &ReplState,
    values: impl IntoIterator<Item = &'a Ref>,
    active: &mut BTreeSet<u32>,
) -> PyResult<Vec<String>> {
    values
        .into_iter()
        .map(|value| render(state, state.heap.value(value), active))
        .collect()
}

pub fn truth(heap: &Heap, value: Value) -> PyResult<bool> {
    if value.is_none() {
        return Ok(false);
    }
    if let Some(value) = value.bool_value() {
        return Ok(value);
    }
    if let (Some(integer), None) = (value.immediate_int(), value.bool_value()) {
        return Ok(integer != 0);
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
    let Some(object) = object(heap, value)? else {
        return Err("invalid Python value tag".into());
    };
    Ok(match object {
        Object::Bare => true,
        Object::String(value) => !value.is_empty(),
        Object::Bytes(value) => !value.is_empty(),
        Object::ByteArray(value) => !value.is_empty(),
        Object::Exception(_) => true,
        Object::List(values) | Object::Tuple(values) => !values.is_empty(),
        Object::Set(values) | Object::FrozenSet(values) => !values.is_empty(),
        Object::Slice { .. } => true,
        Object::Dict(entries) | Object::DefaultDict { entries, .. } => !entries.is_empty(),
        Object::BigInt(value) => !value.is_zero(),
        Object::Float(value) => *value != 0.0,
        Object::Complex { real, imag } => *real != 0.0 || *imag != 0.0,
        Object::Range { start, stop, step } => {
            (*step > 0 && *start < *stop) || (*step < 0 && *start > *stop)
        }
        Object::Function { .. }
        | Object::Class { .. }
        | Object::DescriptorBoundMethod { .. }
        | Object::Iterator { .. }
        | Object::SequenceIterator { .. }
        | Object::ReverseIterator { .. }
        | Object::RangeIterator { .. }
        | Object::CountIterator { .. }
        | Object::StreamIterator { .. }
        | Object::CallableIterator { .. }
        | Object::Generator { .. }
        | Object::Module { .. }
        | Object::WideValue { .. }
        | Object::Native(_)
        | Object::NamespaceDict(_)
        | Object::DictView { .. }
        | Object::MappingProxy(_) => true,
        Object::GenericAlias { .. } => true,
        Object::Property { .. }
        | Object::StaticMethod { .. }
        | Object::ClassMethod { .. }
        | Object::Super { .. }
        | Object::Scope(_) => true,
    })
}

pub fn equals(heap: &Heap, left: Value, right: Value) -> PyResult<bool> {
    if let Some(equal) = scalar_equality(heap, left, right)? {
        return Ok(equal);
    }
    equals_inner(heap, left, right, &mut BTreeSet::new())
}

fn equals_inner(
    heap: &Heap,
    left: Value,
    right: Value,
    active: &mut BTreeSet<(u32, u32)>,
) -> PyResult<bool> {
    if let Some(equal) = scalar_equality(heap, left, right)? {
        return Ok(equal);
    }
    match (heap.identity(left)?, heap.identity(right)?) {
        (Some(left_id), Some(right_id)) if left_id == right_id => Ok(true),
        (Some(left_id), Some(right_id)) => {
            if !active.insert((left_id, right_id)) {
                return Ok(true);
            }
            let result = match (heap.get(left)?, heap.get(right)?) {
                (Object::String(left), Object::String(right)) => left == right,
                (Object::Bytes(left), Object::Bytes(right)) => left == right,
                (Object::Bytes(left), Object::ByteArray(right))
                | (Object::ByteArray(left), Object::Bytes(right))
                | (Object::ByteArray(left), Object::ByteArray(right)) => left == right,
                (Object::List(left), Object::List(right))
                | (Object::Tuple(left), Object::Tuple(right)) => {
                    sequence_equal(heap, left, right, active)?
                }
                (
                    Object::GenericAlias {
                        origin: left_origin,
                        arguments: left_args,
                    },
                    Object::GenericAlias {
                        origin: right_origin,
                        arguments: right_args,
                    },
                ) => {
                    left_origin == right_origin
                        && sequence_equal(heap, left_args, right_args, active)?
                }
                (
                    Object::Slice {
                        start: left_start,
                        stop: left_stop,
                        step: left_step,
                    },
                    Object::Slice {
                        start: right_start,
                        stop: right_stop,
                        step: right_step,
                    },
                ) => sequence_equal(
                    heap,
                    [left_start, left_stop, left_step],
                    [right_start, right_stop, right_step],
                    active,
                )?,
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
                        // Equal keys have equal hashes, so each key is compared only with the
                        // keys in its bucket.
                        let mut all = true;
                        for (hash, (left_key, left_value)) in left.iter_hashed() {
                            let mut found = false;
                            for &position in right.candidate_positions(hash) {
                                let Some((right_key, right_value)) = right.get(position) else {
                                    continue;
                                };
                                if refs_equal(heap, left_key, right_key, active)? {
                                    found = refs_equal(heap, left_value, right_value, active)?;
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
                (
                    Object::Set(left) | Object::FrozenSet(left),
                    Object::Set(right) | Object::FrozenSet(right),
                ) => {
                    if left.len() != right.len() {
                        false
                    } else {
                        let mut all = true;
                        for (hash, left_value) in left.iter_hashed() {
                            let mut found = false;
                            for &position in right.candidate_positions(hash) {
                                let Some(right_value) = right.get(position) else {
                                    continue;
                                };
                                if refs_equal(heap, left_value, right_value, active)? {
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
                | (Object::DescriptorBoundMethod { .. }, Object::DescriptorBoundMethod { .. })
                | (Object::Iterator { .. }, Object::Iterator { .. })
                | (Object::CountIterator { .. }, Object::CountIterator { .. })
                | (Object::CallableIterator { .. }, Object::CallableIterator { .. })
                | (Object::Generator { .. }, Object::Generator { .. })
                | (Object::Module { .. }, Object::Module { .. })
                | (Object::Native(_), Object::Native(_)) => false,
                (Object::Property { .. }, Object::Property { .. })
                | (Object::StaticMethod { .. }, Object::StaticMethod { .. })
                | (Object::ClassMethod { .. }, Object::ClassMethod { .. })
                | (Object::Super { .. }, Object::Super { .. }) => false,
                _ => false,
            };
            active.remove(&(left_id, right_id));
            Ok(result)
        }
        _ => Ok(false),
    }
}

fn scalar_equality(heap: &Heap, left: Value, right: Value) -> PyResult<Option<bool>> {
    use super::number;
    // Registered numbers such as NumPy scalars equal the Python number with the same value, so
    // `np.int64(1)` finds the key `1` in a dict or list.
    if number::registered_number(heap, &left).is_some()
        || number::registered_number(heap, &right).is_some()
    {
        if let (Some(left), Some(right)) = (number::view(heap, &left), number::view(heap, &right)) {
            return Ok(Some(number::numbers_equal(left, right)));
        }
    }
    if let Some(equal) = complex_equality(heap, left, right) {
        return Ok(Some(equal));
    }
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

/// Compare a builtin `complex` with any value. Real numbers compare equal only to a zero
/// imaginary part and an exactly equal real part, as in CPython. Returns `None` when neither
/// operand is complex.
fn complex_equality(heap: &Heap, left: Value, right: Value) -> Option<bool> {
    use super::number::{view, NumberRef};
    let (left, right) = (view(heap, &left), view(heap, &right));
    let ((real, imag), other) = match (left, right) {
        (Some(NumberRef::Complex(real, imag)), other)
        | (other, Some(NumberRef::Complex(real, imag))) => ((real, imag), other),
        _ => return None,
    };
    let exact_integer = |integer: BigInt| {
        imag == 0.0
            && real.is_finite()
            && real.fract() == 0.0
            && BigInt::from_f64(real).is_some_and(|real| real == integer)
    };
    Some(match other {
        Some(NumberRef::Complex(other_real, other_imag)) => {
            real == other_real && imag == other_imag
        }
        Some(NumberRef::Float(other)) => imag == 0.0 && real == other,
        Some(NumberRef::Int(other)) => exact_integer(BigInt::from(other)),
        Some(NumberRef::BigInt(other)) => exact_integer(other.clone()),
        Some(NumberRef::UInt(other)) => exact_integer(BigInt::from(other)),
        None => false,
    })
}

fn sequence_equal<'a>(
    heap: &Heap,
    left: impl IntoIterator<Item = &'a Ref, IntoIter: ExactSizeIterator>,
    right: impl IntoIterator<Item = &'a Ref, IntoIter: ExactSizeIterator>,
    active: &mut BTreeSet<(u32, u32)>,
) -> PyResult<bool> {
    let (left, right) = (left.into_iter(), right.into_iter());
    if left.len() != right.len() {
        return Ok(false);
    }
    for (left, right) in left.zip(right) {
        if !refs_equal(heap, left, right, active)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Identity or equality of two stored references, as containers compare their members.
fn refs_equal(
    heap: &Heap,
    left: &Ref,
    right: &Ref,
    active: &mut BTreeSet<(u32, u32)>,
) -> PyResult<bool> {
    Ok(left == right || equals_inner(heap, heap.value(left), heap.value(right), active)?)
}

/// How two values order under Python's rich comparisons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comparison {
    Ordered(Ordering),
    /// A NaN is involved, so `<`, `<=`, `>` and `>=` are all false.
    Unordered,
    /// The types define no ordering; CPython raises `TypeError`.
    Unsupported,
}

impl Comparison {
    fn reverse(self) -> Self {
        match self {
            Self::Ordered(ordering) => Self::Ordered(ordering.reverse()),
            other => other,
        }
    }
}

pub fn compare(heap: &Heap, left: Value, right: Value) -> PyResult<Comparison> {
    compare_at(heap, left, right, 0)
}

/// Nesting bound for sequence ordering outside the VM, matching [`MAX_RENDER_DEPTH`].
const MAX_COMPARE_DEPTH: usize = 256;

fn compare_at(heap: &Heap, left: Value, right: Value, depth: usize) -> PyResult<Comparison> {
    if let Some(left) = bigint_value(heap, left) {
        if let Some(right) = bigint_value(heap, right) {
            return Ok(Comparison::Ordered(left.cmp(right)));
        }
        if let Some(right) = int_value(heap, right) {
            return Ok(Comparison::Ordered(left.cmp(&BigInt::from(right))));
        }
        if let Some(right) = right.float_value() {
            return compare_bigint_float(left, right);
        }
    }
    if let Some(right) = bigint_value(heap, right) {
        if let Some(left) = int_value(heap, left) {
            return Ok(Comparison::Ordered(BigInt::from(left).cmp(right)));
        }
        if let Some(left) = left.float_value() {
            return compare_bigint_float(right, left).map(Comparison::reverse);
        }
    }
    if let Some(left) = int_value(heap, left) {
        if let Some(right) = int_value(heap, right) {
            return Ok(Comparison::Ordered(left.cmp(&right)));
        }
        if let Some(right) = right.float_value() {
            return compare_bigint_float(&BigInt::from(left), right);
        }
    }
    if let (Some(left), Some(right)) = (left.float_value(), int_value(heap, right)) {
        return compare_bigint_float(&BigInt::from(right), left).map(Comparison::reverse);
    }
    if let (Some(left), Some(right)) = (left.float_value(), right.float_value()) {
        return Ok(left
            .partial_cmp(&right)
            .map_or(Comparison::Unordered, Comparison::Ordered));
    }
    if let (Some(left), Some(right)) = (string_value(heap, left)?, string_value(heap, right)?) {
        return Ok(Comparison::Ordered(left.cmp(&right)));
    }
    if let (Some(left), Some(right)) = (bytes_value(heap, left)?, bytes_value(heap, right)?) {
        return Ok(Comparison::Ordered(left.cmp(&right)));
    }
    match (object(heap, left)?, object(heap, right)?) {
        (Some(Object::List(left)), Some(Object::List(right)))
        | (Some(Object::Tuple(left)), Some(Object::Tuple(right))) => {
            if depth >= MAX_COMPARE_DEPTH {
                return Err(PyError::exception(
                    "RecursionError",
                    "maximum recursion depth exceeded in comparison",
                ));
            }
            sequence_compare(heap, left, right, depth + 1)
        }
        _ => Ok(Comparison::Unsupported),
    }
}

fn compare_bigint_float(integer: &BigInt, float: f64) -> PyResult<Comparison> {
    if float.is_nan() {
        return Ok(Comparison::Unordered);
    }
    if float == f64::INFINITY {
        return Ok(Comparison::Ordered(Ordering::Less));
    }
    if float == f64::NEG_INFINITY {
        return Ok(Comparison::Ordered(Ordering::Greater));
    }
    let truncated = BigInt::from_f64(float).ok_or("float cannot be converted for comparison")?;
    let ordering = integer.cmp(&truncated);
    if ordering != Ordering::Equal || float.fract() == 0.0 {
        return Ok(Comparison::Ordered(ordering));
    }
    Ok(Comparison::Ordered(if float.is_sign_positive() {
        Ordering::Less
    } else {
        Ordering::Greater
    }))
}

fn sequence_compare(
    heap: &Heap,
    left: &[Ref],
    right: &[Ref],
    depth: usize,
) -> PyResult<Comparison> {
    for (left, right) in left.iter().zip(right) {
        let (left, right) = (heap.value(left), heap.value(right));
        if left.is(right) || equals(heap, left, right)? {
            continue;
        }
        return compare_at(heap, left, right, depth);
    }
    Ok(Comparison::Ordered(left.len().cmp(&right.len())))
}

pub fn contains(heap: &Heap, container: Value, needle: Value) -> PyResult<bool> {
    if let Some(container) = string_value(heap, container)? {
        let Some(needle) = string_value(heap, needle)? else {
            return Err("string containment requires a string operand".into());
        };
        return Ok(container.contains(&needle));
    }
    match object(heap, container)? {
        Some(object) => match object {
            Object::Bare
            | Object::Float(_)
            | Object::String(_)
            | Object::Exception(_)
            | Object::Slice { .. } => Err("object is not a container".into()),
            Object::Bytes(value) | Object::ByteArray(value) => {
                if let Some(needle) = bytes_ref(heap, needle)? {
                    return Ok(memchr::memmem::find(value, needle).is_some());
                }
                let needle = int_value(heap, needle)
                    .and_then(|value| u8::try_from(value).ok())
                    .ok_or("bytes containment requires an integer in range(0, 256)")?;
                Ok(value.contains(&needle))
            }
            Object::List(values) | Object::Tuple(values) => {
                for value in values {
                    if needle.is_ref(value) || equals(heap, heap.value(value), needle)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Object::Set(values) | Object::FrozenSet(values) => {
                for value in values {
                    if needle.is_ref(value) || equals(heap, heap.value(value), needle)? {
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
                    if needle.is_ref(key) || equals(heap, heap.value(key), needle)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Object::Function { .. }
            | Object::Class { .. }
            | Object::DescriptorBoundMethod { .. }
            | Object::Iterator { .. }
            | Object::SequenceIterator { .. }
            | Object::ReverseIterator { .. }
            | Object::RangeIterator { .. }
            | Object::CountIterator { .. }
            | Object::StreamIterator { .. }
            | Object::CallableIterator { .. }
            | Object::Generator { .. }
            | Object::Module { .. }
            | Object::WideValue { .. }
            | Object::Native(_)
            | Object::NamespaceDict(_)
            | Object::DictView { .. }
            | Object::MappingProxy(_) => Err("object is not a container".into()),
            Object::GenericAlias { .. } => Err("object is not a container".into()),
            Object::Property { .. }
            | Object::StaticMethod { .. }
            | Object::ClassMethod { .. }
            | Object::Super { .. }
            | Object::Scope(_) => Err("object is not a container".into()),
            Object::BigInt(_) | Object::Complex { .. } => Err("object is not a container".into()),
        },
        None => Err("object is not a container".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{Limits, Resources};

    #[test]
    fn dict_and_set_equality_are_order_independent() {
        let mut heap = Heap::default();
        let mut resources = Resources::new(Limits::unlimited());
        let mut dict = |entries: [(&str, i64); 2]| {
            let mut map = super::super::heap::OrderedMap::default();
            for (key, value) in entries {
                let entry = (
                    Ref::from(Value::inline_string(key).unwrap()),
                    Ref::from(Value::Int(value)),
                );
                map.push(super::super::hash::string(key), entry);
            }
            heap.alloc(Object::Dict(map), &(), &mut resources).unwrap()
        };
        let first = dict([("a", 1), ("b", 2)]);
        let second = dict([("b", 2), ("a", 1)]);
        let swapped = dict([("a", 2), ("b", 1)]);
        assert!(equals(&heap, first, second).unwrap());
        assert!(!equals(&heap, first, swapped).unwrap());
    }

    #[test]
    fn nan_identity_is_distinct_from_nan_equality() {
        let mut heap = Heap::default();
        let mut resources = Resources::new(Limits::unlimited());
        let nan = Value::Float(f64::NAN);
        let first = heap
            .alloc(Object::List(vec![Ref::from(nan)]), &(), &mut resources)
            .unwrap();
        let second = heap
            .alloc(Object::List(vec![Ref::from(nan)]), &(), &mut resources)
            .unwrap();

        assert!(nan.is(nan));
        assert!(!equals(&heap, nan, nan).unwrap());
        assert!(contains(&heap, first, nan).unwrap());
        assert!(equals(&heap, first, second).unwrap());
    }
}
