//! Dict views (`dict_keys`, `dict_values`, `dict_items`) and the read-only `mappingproxy`.
//!
//! A view holds its mapping and reads it on every operation, so it reflects later insertions
//! and deletions as CPython's views do. Keys and items views are set-like: they compare with
//! sets and other set-like views, and `&`, `|`, `^` and `-` accept any iterable on either side
//! and return a `set`. A values view is only iterable.
//!
//! A `mappingproxy` is the read-only mapping a class or native module presents as `__dict__`.
//! It supports lookup, iteration and the mapping protocol, and has no mutating methods, so
//! `A.__dict__["x"] = 1` raises `TypeError`.
//!
//! Every operation reads the mapping through [`PyRuntime::mapping_items`], which reads dicts,
//! namespace views and proxies straight from storage. The Python-visible behavior is CPython's;
//! lookups in a proxy are linear in its size.

use super::super::ast::ComparisonOperator;
use super::super::heap::DictViewKind;
use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, MethodDef, NativeMethodFn, NativeTypeDef, PyDict, PyError, PyOperator, PyResult,
    PyRuntime, PySet, PyTuple, PyValue, PyValueCast,
};

pub(crate) static DICT_KEYS_TYPE: NativeTypeDef = NativeTypeDef {
    name: "dict_keys",
    methods: &[method("dict_keys", "isdisjoint", view_isdisjoint)],
    getters: &[],
};

pub(crate) static DICT_VALUES_TYPE: NativeTypeDef = NativeTypeDef {
    name: "dict_values",
    methods: &[],
    getters: &[],
};

pub(crate) static DICT_ITEMS_TYPE: NativeTypeDef = NativeTypeDef {
    name: "dict_items",
    methods: &[method("dict_items", "isdisjoint", view_isdisjoint)],
    getters: &[],
};

pub(crate) static MAPPING_PROXY_TYPE: NativeTypeDef = NativeTypeDef {
    name: "mappingproxy",
    methods: &[
        method("mappingproxy", "get", proxy_get),
        method("mappingproxy", "keys", proxy_keys),
        method("mappingproxy", "values", proxy_values),
        method("mappingproxy", "items", proxy_items),
        method("mappingproxy", "copy", proxy_copy),
    ],
    getters: &[],
};

const fn method(type_name: &'static str, name: &'static str, call: NativeMethodFn) -> MethodDef {
    MethodDef {
        type_name,
        name,
        call,
    }
}

/// The kind and mapping of a dict view.
fn view_parts<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    view: PyValue<'s>,
) -> PyResult<'s, (DictViewKind, PyValue<'s>)> {
    runtime
        .dict_view(&view)?
        .ok_or_else(|| PyError::type_error("descriptor requires a dict view"))
}

/// The current entries of a dict, namespace view or mapping proxy.
fn entries<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    mapping: PyValue<'s>,
) -> PyResult<'s, Vec<(PyValue<'s>, PyValue<'s>)>> {
    runtime
        .mapping_items(mapping)?
        .ok_or_else(|| PyError::type_error("a dict view needs a mapping"))
}

/// The members a view of `kind` over `mapping` currently yields, in iteration order.
fn view_members<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    kind: DictViewKind,
    mapping: PyValue<'s>,
) -> PyResult<'s, Vec<PyValue<'s>>> {
    let entries = entries(runtime, mapping)?;
    let mut members = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        runtime.charge_cpu(1)?;
        members.push(match kind {
            DictViewKind::Keys => key,
            DictViewKind::Values => value,
            DictViewKind::Items => runtime.new_tuple(vec![key, value])?,
        });
    }
    Ok(members)
}

/// `mapping[key]` without raising: a dict's own lookup, or a linear search of a proxy.
fn lookup<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    mapping: PyValue<'s>,
    key: &PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    if let Ok(dict) = mapping.cast::<PyDict<'s>>(runtime) {
        return runtime.dict_get(dict, key);
    }
    for (candidate, value) in entries(runtime, mapping)? {
        runtime.charge_cpu(1)?;
        if runtime.equals(&candidate, key)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn view_contains<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    view: PyValue<'s>,
    item: &PyValue<'s>,
) -> PyResult<'s, bool> {
    let (kind, mapping) = view_parts(runtime, view)?;
    match kind {
        DictViewKind::Keys => Ok(lookup(runtime, mapping, item)?.is_some()),
        DictViewKind::Items => {
            let Ok(pair) = (*item).cast::<PyTuple<'s>>(runtime) else {
                return Ok(false);
            };
            let [key, value] = pair.items(runtime)?[..] else {
                return Ok(false);
            };
            match lookup(runtime, mapping, &key)? {
                Some(found) => runtime.equals(&found, &value),
                None => Ok(false),
            }
        }
        DictViewKind::Values => {
            for (_, value) in entries(runtime, mapping)? {
                runtime.charge_cpu(1)?;
                if runtime.equals(&value, item)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }
}

pub(crate) fn slot_view_length<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    view: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    let (_, mapping) = view_parts(runtime, view)?;
    length(runtime, mapping).map(Some)
}

fn length<'s>(runtime: &mut dyn PyRuntime<'s>, mapping: PyValue<'s>) -> PyResult<'s, PyValue<'s>> {
    let length = entries(runtime, mapping)?.len();
    let length =
        i64::try_from(length).map_err(|_| PyError::overflow_error("mapping is too large"))?;
    Ok(Value::Int(length))
}

pub(crate) fn slot_view_iter<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    view: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    let (kind, mapping) = view_parts(runtime, view)?;
    let members = view_members(runtime, kind, mapping)?;
    runtime.new_iterator(members).map(Some)
}

pub(crate) fn slot_view_reversed<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    view: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    let (kind, mapping) = view_parts(runtime, view)?;
    let mut members = view_members(runtime, kind, mapping)?;
    runtime.charge_cpu(u64::try_from(members.len()).unwrap_or(u64::MAX))?;
    members.reverse();
    runtime.new_iterator(members).map(Some)
}

pub(crate) fn slot_view_contains<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    view: PyValue<'s>,
    item: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    view_contains(runtime, view, &item).map(|found| Some(Value::Bool(found)))
}

/// The members of a set, frozenset, or keys or items view; `None` for anything else, which a
/// set-like comparison declines.
fn set_like_members<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    value: PyValue<'s>,
) -> PyResult<'s, Option<Vec<PyValue<'s>>>> {
    if let Ok(set) = value.cast::<PySet<'s>>(runtime) {
        return set.items(runtime).map(Some);
    }
    match runtime.dict_view(&value)? {
        Some((kind @ (DictViewKind::Keys | DictViewKind::Items), mapping)) => {
            view_members(runtime, kind, mapping).map(Some)
        }
        _ => Ok(None),
    }
}

/// Whether every one of `members` is `in` the container.
fn all_in<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    members: &[PyValue<'s>],
    container: PyValue<'s>,
) -> PyResult<'s, bool> {
    let contains = PyOperator::Compare(ComparisonOperator::In);
    for member in members {
        runtime.charge_cpu(1)?;
        let found = runtime.apply_operator(contains, &[*member, container])?;
        if !runtime.truth(&found)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// How a set-like comparison relates the view to the other operand.
#[derive(Clone, Copy)]
enum Relation {
    Equal,
    Subset { proper: bool },
    Superset { proper: bool },
}

/// Compare a keys or items view with another set-like operand, as sets compare.
fn compare_view<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    view: PyValue<'s>,
    other: PyValue<'s>,
    relation: Relation,
) -> PyResult<'s, Option<PyValue<'s>>> {
    let Some(other_members) = set_like_members(runtime, other)? else {
        return Ok(None);
    };
    let (kind, mapping) = view_parts(runtime, view)?;
    let members = view_members(runtime, kind, mapping)?;
    let (size, other_size) = (members.len(), other_members.len());
    let result = match relation {
        Relation::Equal => size == other_size && all_in(runtime, &members, other)?,
        Relation::Subset { proper } => {
            (if proper {
                size < other_size
            } else {
                size <= other_size
            }) && all_in(runtime, &members, other)?
        }
        Relation::Superset { proper } => {
            (if proper {
                size > other_size
            } else {
                size >= other_size
            }) && all_in(runtime, &other_members, view)?
        }
    };
    Ok(Some(Value::Bool(result)))
}

pub(crate) fn slot_view_equal<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    view: PyValue<'s>,
    other: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    compare_view(runtime, view, other, Relation::Equal)
}

pub(crate) fn slot_view_less<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    view: PyValue<'s>,
    other: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    compare_view(runtime, view, other, Relation::Subset { proper: true })
}

pub(crate) fn slot_view_less_equal<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    view: PyValue<'s>,
    other: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    compare_view(runtime, view, other, Relation::Subset { proper: false })
}

pub(crate) fn slot_view_greater<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    view: PyValue<'s>,
    other: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    compare_view(runtime, view, other, Relation::Superset { proper: true })
}

pub(crate) fn slot_view_greater_equal<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    view: PyValue<'s>,
    other: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    compare_view(runtime, view, other, Relation::Superset { proper: false })
}

/// `left <op> right` where either side is a view: `set(left)` updated in place by the named
/// `set` method with `right`. Either operand may be any iterable, as in CPython.
fn set_operation<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    left: PyValue<'s>,
    right: PyValue<'s>,
    update: &str,
) -> PyResult<'s, Option<PyValue<'s>>> {
    let set_type = runtime
        .builtin_type("set")
        .ok_or_else(|| PyError::runtime_error("the set type is not registered"))?;
    let result = runtime.call_value(set_type, CallArgs::new(vec![left], Vec::new()))?;
    let update = runtime
        .get_attribute(result, update)?
        .ok_or_else(|| PyError::runtime_error("set lacks an update method"))?;
    runtime.call_value(update, CallArgs::new(vec![right], Vec::new()))?;
    Ok(Some(result))
}

pub(crate) fn slot_view_and<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    left: PyValue<'s>,
    right: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    set_operation(runtime, left, right, "intersection_update")
}

pub(crate) fn slot_view_reflected_and<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    right: PyValue<'s>,
    left: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    set_operation(runtime, left, right, "intersection_update")
}

pub(crate) fn slot_view_or<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    left: PyValue<'s>,
    right: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    set_operation(runtime, left, right, "update")
}

pub(crate) fn slot_view_reflected_or<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    right: PyValue<'s>,
    left: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    set_operation(runtime, left, right, "update")
}

pub(crate) fn slot_view_xor<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    left: PyValue<'s>,
    right: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    set_operation(runtime, left, right, "symmetric_difference_update")
}

pub(crate) fn slot_view_reflected_xor<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    right: PyValue<'s>,
    left: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    set_operation(runtime, left, right, "symmetric_difference_update")
}

pub(crate) fn slot_view_subtract<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    left: PyValue<'s>,
    right: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    set_operation(runtime, left, right, "difference_update")
}

pub(crate) fn slot_view_reflected_subtract<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    right: PyValue<'s>,
    left: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    set_operation(runtime, left, right, "difference_update")
}

/// `view.isdisjoint(iterable)`: no item of the iterable is in the view. Stops at the first
/// shared item.
fn view_isdisjoint<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("isdisjoint", 1, 1)?;
    args.reject_keywords("isdisjoint")?;
    let iterator = runtime.iterator(args.positional()[0])?;
    // The iterable is unbounded, so each step runs in its own handle scope.
    let mut disjoint = true;
    let mut done = false;
    while !done {
        runtime.nested(&mut |runtime, _| {
            let Some(item) = runtime.iterator_next(iterator)? else {
                done = true;
                return Ok(());
            };
            runtime.charge_cpu(1)?;
            if view_contains(runtime, receiver, &item)? {
                disjoint = false;
                done = true;
            }
            Ok(())
        })?;
    }
    Ok(Value::Bool(disjoint))
}

pub(crate) fn slot_proxy_get_item<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    proxy: PyValue<'s>,
    key: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    match lookup(runtime, proxy, &key)? {
        Some(value) => Ok(Some(value)),
        None => Err(runtime.exception_with_args("KeyError", vec![key])),
    }
}

pub(crate) fn slot_proxy_length<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    proxy: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    length(runtime, proxy).map(Some)
}

pub(crate) fn slot_proxy_contains<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    proxy: PyValue<'s>,
    key: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    Ok(Some(Value::Bool(lookup(runtime, proxy, &key)?.is_some())))
}

pub(crate) fn slot_proxy_iter<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    proxy: PyValue<'s>,
) -> PyResult<'s, Option<PyValue<'s>>> {
    let keys = view_members(runtime, DictViewKind::Keys, proxy)?;
    runtime.new_iterator(keys).map(Some)
}

fn proxy_get<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("mappingproxy.get", 1, 2)?;
    args.reject_keywords("mappingproxy.get")?;
    let default = args.positional().get(1).copied().unwrap_or(Value::None);
    Ok(lookup(runtime, receiver, &args.positional()[0])?.unwrap_or(default))
}

fn proxy_keys<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    proxy_view(runtime, receiver, args, DictViewKind::Keys)
}

fn proxy_values<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    proxy_view(runtime, receiver, args, DictViewKind::Values)
}

fn proxy_items<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    proxy_view(runtime, receiver, args, DictViewKind::Items)
}

fn proxy_view<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
    kind: DictViewKind,
) -> PyResult<'s> {
    args.expect_positional("mappingproxy view", 0, 0)?;
    args.reject_keywords("mappingproxy view")?;
    runtime.new_dict_view(kind, receiver)
}

/// `proxy.copy()`: a new `dict` of the current entries.
fn proxy_copy<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("mappingproxy.copy", 0, 0)?;
    args.reject_keywords("mappingproxy.copy")?;
    let entries = entries(runtime, receiver)?;
    runtime.new_dict(entries)
}
