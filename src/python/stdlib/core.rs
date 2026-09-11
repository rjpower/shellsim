//! Native descriptors for methods on builtin Python value types.
//!
//! Methods use checked, type-erased runtime views and snapshot-and-commit mutation. This keeps
//! collection layouts and compact scalar tags private to the runtime while giving builtin and
//! user-defined methods the same descriptor call path.

use std::cmp::Ordering;

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, MethodDef, NativeTypeDef, PyCallable, PyDict, PyError, PyKind, PyList,
    PyProperty, PyResult, PyRuntime, PySequence, PySet, PyString, PyValue, PyValueCast,
};

static BUILTINS: &[FunctionDef] = &[
    builtin("map", builtin_map),
    builtin("filter", builtin_filter),
    builtin("reversed", builtin_reversed),
    builtin("getattr", builtin_getattr),
    builtin("hasattr", builtin_hasattr),
];

/// Resolve capability-free builtins implemented through the erased runtime API.
pub(crate) fn builtin_function(name: &str) -> Option<&'static FunctionDef> {
    BUILTINS.iter().find(|function| function.name == name)
}

const fn builtin(
    name: &'static str,
    call: fn(&mut dyn PyRuntime, CallArgs) -> PyResult,
) -> FunctionDef {
    FunctionDef {
        module: "builtins",
        name,
        call,
    }
}

pub(crate) static STRING_TYPE: NativeTypeDef = NativeTypeDef {
    name: "str",
    methods: &[
        method("str", "strip", string_strip),
        method("str", "lstrip", string_lstrip),
        method("str", "rstrip", string_rstrip),
        method("str", "startswith", string_startswith),
        method("str", "endswith", string_endswith),
        method("str", "split", string_split),
        method("str", "join", string_join),
        method("str", "replace", string_replace),
        method("str", "format", string_format),
    ],
};

pub(crate) static LIST_TYPE: NativeTypeDef = NativeTypeDef {
    name: "list",
    methods: &[
        method("list", "append", list_append),
        method("list", "extend", list_extend),
        method("list", "pop", list_pop),
        method("list", "remove", list_remove),
        method("list", "reverse", list_reverse),
        method("list", "count", list_count),
        method("list", "index", list_index),
        method("list", "sort", list_sort),
    ],
};

pub(crate) static DICT_TYPE: NativeTypeDef = NativeTypeDef {
    name: "dict",
    methods: &[
        method("dict", "get", dict_get),
        method("dict", "keys", dict_keys),
        method("dict", "values", dict_values),
        method("dict", "items", dict_items),
        method("dict", "setdefault", dict_setdefault),
        method("dict", "update", dict_update),
        method("dict", "pop", dict_pop),
    ],
};

pub(crate) static SET_TYPE: NativeTypeDef = NativeTypeDef {
    name: "set",
    methods: &[
        method("set", "add", set_add),
        method("set", "update", set_update),
        method("set", "remove", set_remove),
        method("set", "discard", set_discard),
        method("set", "union", set_union),
    ],
};

pub(crate) static PROPERTY_TYPE: NativeTypeDef = NativeTypeDef {
    name: "property",
    methods: &[method("property", "setter", property_setter)],
};

pub(crate) static TYPE_TYPE: NativeTypeDef = NativeTypeDef {
    name: "type",
    methods: &[method("type", "__new__", type_new)],
};

const fn method(
    type_name: &'static str,
    name: &'static str,
    call: fn(&mut dyn PyRuntime, PyValue, CallArgs) -> PyResult,
) -> MethodDef {
    MethodDef {
        type_name,
        name,
        call,
    }
}

fn string_strip(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    strip(runtime, receiver, args, StripKind::Both)
}

fn string_lstrip(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    strip(runtime, receiver, args, StripKind::Left)
}

fn string_rstrip(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    strip(runtime, receiver, args, StripKind::Right)
}

enum StripKind {
    Both,
    Left,
    Right,
}

fn strip(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    kind: StripKind,
) -> PyResult {
    args.expect_positional("str.strip", 0, 1)?;
    args.reject_keywords("str.strip")?;
    let PyString(value) = receiver.cast(runtime)?;
    let characters = match args.positional().first() {
        None => None,
        Some(value) if runtime.kind(value)? == super::super::native::PyKind::None => None,
        Some(value) => Some((*value).cast::<PyString>(runtime)?.0),
    };
    let result = match (kind, characters.as_deref()) {
        (StripKind::Both, None) => value.trim().to_string(),
        (StripKind::Left, None) => value.trim_start().to_string(),
        (StripKind::Right, None) => value.trim_end().to_string(),
        (StripKind::Both, Some(chars)) => value.trim_matches(|ch| chars.contains(ch)).to_string(),
        (StripKind::Left, Some(chars)) => value
            .trim_start_matches(|ch| chars.contains(ch))
            .to_string(),
        (StripKind::Right, Some(chars)) => {
            value.trim_end_matches(|ch| chars.contains(ch)).to_string()
        }
    };
    runtime.new_string(result)
}

fn string_startswith(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_affix(runtime, receiver, args, true)
}

fn string_endswith(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_affix(runtime, receiver, args, false)
}

fn string_affix(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    prefix: bool,
) -> PyResult {
    args.expect_positional("str prefix test", 1, 1)?;
    args.reject_keywords("str prefix test")?;
    let PyString(value) = receiver.cast(runtime)?;
    let PyString(needle) = args.positional()[0].cast(runtime)?;
    Ok(Value::Bool(if prefix {
        value.starts_with(&needle)
    } else {
        value.ends_with(&needle)
    }))
}

fn string_split(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.split", 0, 2)?;
    args.reject_keywords("str.split")?;
    let PyString(value) = receiver.cast(runtime)?;
    let separator = match args.positional().first() {
        None => None,
        Some(value) if runtime.kind(value)? == super::super::native::PyKind::None => None,
        Some(value) => {
            let PyString(value) = (*value).cast(runtime)?;
            if value.is_empty() {
                return Err(PyError::value_error("empty separator"));
            }
            Some(value)
        }
    };
    let maximum = args
        .positional()
        .get(1)
        .map(|value| {
            runtime
                .int_value(value)
                .ok_or_else(|| PyError::type_error("maxsplit must be an integer"))
        })
        .transpose()?;
    let parts = split_text(&value, separator.as_deref(), maximum);
    let mut values = Vec::with_capacity(parts.len());
    for part in parts {
        values.push(runtime.new_string(part)?);
    }
    runtime.new_list(values)
}

fn string_join(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.join", 1, 1)?;
    args.reject_keywords("str.join")?;
    let PyString(separator) = receiver.cast(runtime)?;
    let iterator = runtime.iterator(args.positional()[0])?;
    let mut parts = Vec::new();
    let mut bytes = 0usize;
    while let Some(value) = runtime.iterator_next(iterator)? {
        runtime.charge_cpu(1)?;
        let PyString(value) = value.cast(runtime)?;
        bytes = bytes
            .checked_add(value.len())
            .ok_or_else(|| PyError::resource_error("joined string is too large"))?;
        parts.push(value);
    }
    bytes = bytes
        .checked_add(
            separator
                .len()
                .saturating_mul(parts.len().saturating_sub(1)),
        )
        .ok_or_else(|| PyError::resource_error("joined string is too large"))?;
    runtime.reserve_memory(bytes)?;
    runtime.charge_cpu(u64::try_from(bytes).unwrap_or(u64::MAX))?;
    runtime.new_string(parts.join(&separator))
}

fn string_replace(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.replace", 2, 3)?;
    args.reject_keywords("str.replace")?;
    let PyString(value) = receiver.cast(runtime)?;
    let PyString(old) = args.positional()[0].cast(runtime)?;
    let PyString(new) = args.positional()[1].cast(runtime)?;
    let count = args.positional().get(2).map_or(Ok(None), |value| {
        runtime
            .int_value(value)
            .ok_or_else(|| PyError::type_error("replace count must be an integer"))
            .map(|value| usize::try_from(value).ok())
    })?;
    let possible = if old.is_empty() {
        value.chars().count().saturating_add(1)
    } else {
        value.matches(&old).count()
    };
    let replacements = count.map_or(possible, |count| count.min(possible));
    let growth = new.len().saturating_sub(old.len());
    let bound = value
        .len()
        .checked_add(growth.saturating_mul(replacements))
        .ok_or_else(|| PyError::resource_error("replacement string is too large"))?;
    runtime.reserve_memory(bound)?;
    runtime.charge_cpu(u64::try_from(bound).unwrap_or(u64::MAX))?;
    let result = match count {
        Some(count) => value.replacen(&old, &new, count),
        None => value.replace(&old, &new),
    };
    runtime.new_string(result)
}

fn string_format(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let PyString(template) = receiver.cast(runtime)?;
    let mut result = String::new();
    let mut characters = template.chars().peekable();
    let mut automatic = 0usize;
    let mut used_automatic = false;
    let mut used_manual_index = false;
    while let Some(character) = characters.next() {
        runtime.charge_cpu(1)?;
        match character {
            '{' if characters.peek() == Some(&'{') => {
                characters.next();
                result.push('{');
            }
            '}' if characters.peek() == Some(&'}') => {
                characters.next();
                result.push('}');
            }
            '{' => {
                let mut field = String::new();
                loop {
                    match characters.next() {
                        Some('}') => break,
                        Some('{') | None => {
                            return Err(PyError::value_error("unmatched '{' in format string"))
                        }
                        Some(character) => field.push(character),
                    }
                }
                if field.contains(['!', ':']) {
                    return Err(PyError::value_error(
                        "format conversions and specifications are not implemented",
                    ));
                }
                let value = if field.is_empty() {
                    if used_manual_index {
                        return Err(PyError::value_error(
                            "cannot switch from manual field specification to automatic field numbering",
                        ));
                    }
                    used_automatic = true;
                    let value = args
                        .positional()
                        .get(automatic)
                        .ok_or_else(|| PyError::value_error("replacement index out of range"))?;
                    automatic = automatic.saturating_add(1);
                    *value
                } else if let Ok(index) = field.parse::<usize>() {
                    if used_automatic {
                        return Err(PyError::value_error(
                            "cannot switch from automatic field numbering to manual field specification",
                        ));
                    }
                    used_manual_index = true;
                    *args
                        .positional()
                        .get(index)
                        .ok_or_else(|| PyError::value_error("replacement index out of range"))?
                } else {
                    *args
                        .keywords()
                        .iter()
                        .find(|(name, _)| name == &field)
                        .map(|(_, value)| value)
                        .ok_or_else(|| PyError::exception("KeyError", field.clone()))?
                };
                result.push_str(&runtime.display(&value)?);
            }
            '}' => return Err(PyError::value_error("single '}' in format string")),
            character => result.push(character),
        }
    }
    runtime.reserve_memory(result.len())?;
    runtime.new_string(result)
}

fn split_text(value: &str, separator: Option<&str>, maximum: Option<i64>) -> Vec<String> {
    let unlimited = maximum.is_none_or(|maximum| maximum < 0);
    let limit = maximum
        .and_then(|maximum| usize::try_from(maximum).ok())
        .unwrap_or(usize::MAX);
    match separator {
        Some(separator) if unlimited => value.split(separator).map(str::to_string).collect(),
        Some(separator) => value
            .splitn(limit.saturating_add(1), separator)
            .map(str::to_string)
            .collect(),
        None if unlimited => value.split_whitespace().map(str::to_string).collect(),
        None => {
            let mut parts = value.split_whitespace();
            let mut result = Vec::new();
            for _ in 0..limit {
                let Some(part) = parts.next() else {
                    return result;
                };
                result.push(part.to_string());
            }
            let remainder = parts.collect::<Vec<_>>().join(" ");
            if !remainder.is_empty() {
                result.push(remainder);
            }
            result
        }
    }
}

fn list_append(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.append", 1, 1)?;
    args.reject_keywords("list.append")?;
    let list = receiver.cast::<PyList>(runtime)?;
    let mut values = list.items(runtime)?;
    runtime.reserve_memory(64)?;
    values.push(args.positional()[0]);
    runtime.replace_list_items(list, values)?;
    Ok(Value::None)
}

fn list_extend(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.extend", 1, 1)?;
    args.reject_keywords("list.extend")?;
    let list = receiver.cast::<PyList>(runtime)?;
    let mut values = list.items(runtime)?;
    let iterator = runtime.iterator(args.positional()[0])?;
    while let Some(value) = runtime.iterator_next(iterator)? {
        runtime.reserve_memory(64)?;
        values.push(value);
    }
    runtime.replace_list_items(list, values)?;
    Ok(Value::None)
}

fn list_pop(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.pop", 0, 1)?;
    args.reject_keywords("list.pop")?;
    let list = receiver.cast::<PyList>(runtime)?;
    let mut values = list.items(runtime)?;
    if values.is_empty() {
        return Err(PyError::value_error("pop from empty list"));
    }
    let raw = args.positional().first().map_or(Ok(-1), |value| {
        runtime
            .int_value(value)
            .ok_or_else(|| PyError::type_error("list index must be an integer"))
    })?;
    let len = i64::try_from(values.len()).map_err(|_| PyError::overflow_error("list too large"))?;
    let raw = if raw < 0 {
        len.saturating_add(raw)
    } else {
        raw
    };
    let index = usize::try_from(raw).map_err(|_| PyError::value_error("pop index out of range"))?;
    if index >= values.len() {
        return Err(PyError::value_error("pop index out of range"));
    }
    let value = values.remove(index);
    runtime.replace_list_items(list, values)?;
    Ok(value)
}

fn list_remove(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.remove", 1, 1)?;
    args.reject_keywords("list.remove")?;
    let list = receiver.cast::<PyList>(runtime)?;
    let mut values = list.items(runtime)?;
    let mut position = None;
    for (index, value) in values.iter().enumerate() {
        runtime.charge_cpu(1)?;
        if runtime.equals(value, &args.positional()[0])? {
            position = Some(index);
            break;
        }
    }
    let position = position.ok_or_else(|| PyError::value_error("list.remove(x): x not in list"))?;
    values.remove(position);
    runtime.replace_list_items(list, values)?;
    Ok(Value::None)
}

fn list_reverse(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.reverse", 0, 0)?;
    args.reject_keywords("list.reverse")?;
    let list = receiver.cast::<PyList>(runtime)?;
    let mut values = list.items(runtime)?;
    runtime.charge_cpu(u64::try_from(values.len()).unwrap_or(u64::MAX))?;
    values.reverse();
    runtime.replace_list_items(list, values)?;
    Ok(Value::None)
}

fn list_count(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.count", 1, 1)?;
    args.reject_keywords("list.count")?;
    let values = receiver.cast::<PyList>(runtime)?.items(runtime)?;
    let mut count = 0i64;
    for value in values {
        runtime.charge_cpu(1)?;
        if runtime.equals(&value, &args.positional()[0])? {
            count = count
                .checked_add(1)
                .ok_or_else(|| PyError::overflow_error("list is too large"))?;
        }
    }
    Ok(Value::Int(count))
}

fn list_index(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.index", 1, 3)?;
    args.reject_keywords("list.index")?;
    let values = receiver.cast::<PyList>(runtime)?.items(runtime)?;
    let length =
        i64::try_from(values.len()).map_err(|_| PyError::overflow_error("list too large"))?;
    let endpoint = |value: Option<&PyValue>, default: i64| -> PyResult<i64> {
        value.map_or(Ok(default), |value| {
            runtime
                .int_value(value)
                .ok_or_else(|| PyError::type_error("slice index must be an integer"))
        })
    };
    let normalize = |value: i64| {
        if value < 0 {
            length.saturating_add(value).max(0)
        } else {
            value.min(length)
        }
    };
    let start = normalize(endpoint(args.positional().get(1), 0)?);
    let stop = normalize(endpoint(args.positional().get(2), length)?);
    for index in start..stop {
        runtime.charge_cpu(1)?;
        if runtime.equals(&values[index as usize], &args.positional()[0])? {
            return Ok(Value::Int(index));
        }
    }
    Err(PyError::value_error("list.index(x): x not in list"))
}

fn list_sort(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.sort", 0, 0)?;
    let list = receiver.cast::<PyList>(runtime)?;
    let key = args.keyword("list.sort", "key")?.copied().filter(|value| {
        runtime
            .kind(value)
            .is_ok_and(|kind| kind != super::super::native::PyKind::None)
    });
    let reverse = args
        .keyword("list.sort", "reverse")?
        .map(|value| runtime.truth(value))
        .transpose()?
        .unwrap_or(false);
    args.reject_unknown_keywords("list.sort", &["key", "reverse"])?;
    let values = list.items(runtime)?;
    let mut keyed = Vec::with_capacity(values.len());
    for value in values {
        let sort_key = if let Some(callable) = key {
            runtime.call_value(callable, CallArgs::new(vec![value], Vec::new()))?
        } else {
            value
        };
        runtime.reserve_memory(64)?;
        keyed.push((sort_key, value));
    }
    for index in 1..keyed.len() {
        let mut current = index;
        while current > 0 {
            runtime.charge_cpu(1)?;
            let order = runtime.compare(&keyed[current].0, &keyed[current - 1].0)?;
            if order
                != if reverse {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            {
                break;
            }
            keyed.swap(current, current - 1);
            current -= 1;
        }
    }
    runtime.replace_list_items(list, keyed.into_iter().map(|(_, value)| value).collect())?;
    Ok(Value::None)
}

fn find_entry(
    runtime: &mut dyn PyRuntime,
    entries: &[(PyValue, PyValue)],
    key: &PyValue,
) -> PyResult<Option<usize>> {
    for (index, (candidate, _)) in entries.iter().enumerate() {
        runtime.charge_cpu(1)?;
        if runtime.equals(candidate, key)? {
            return Ok(Some(index));
        }
    }
    Ok(None)
}

fn dict_get(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    dict_lookup(runtime, receiver, args, false)
}

fn dict_setdefault(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    dict_lookup(runtime, receiver, args, true)
}

fn dict_lookup(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    insert: bool,
) -> PyResult {
    args.expect_positional("dict lookup", 1, 2)?;
    args.reject_keywords("dict lookup")?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    let mut entries = dict.items(runtime)?;
    if let Some(position) = find_entry(runtime, &entries, &args.positional()[0])? {
        return Ok(entries[position].1);
    }
    let default = args.positional().get(1).copied().unwrap_or(Value::None);
    if insert {
        runtime.reserve_memory(96)?;
        entries.push((args.positional()[0], default));
        runtime.replace_dict_items(dict, entries)?;
    }
    Ok(default)
}

fn dict_keys(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    dict_projection(runtime, receiver, args, 0)
}

fn dict_values(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    dict_projection(runtime, receiver, args, 1)
}

fn dict_items(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    dict_projection(runtime, receiver, args, 2)
}

fn dict_projection(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    projection: u8,
) -> PyResult {
    args.expect_positional("dict view", 0, 0)?;
    args.reject_keywords("dict view")?;
    let entries = receiver.cast::<PyDict>(runtime)?.items(runtime)?;
    let mut values = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        runtime.charge_cpu(1)?;
        values.push(match projection {
            0 => key,
            1 => value,
            _ => runtime.new_tuple(vec![key, value])?,
        });
    }
    runtime.new_list(values)
}

fn dict_update(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.update", 0, 1)?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    let mut entries = dict.items(runtime)?;
    let mut additions = Vec::new();
    if let Some(source) = args.positional().first() {
        if runtime.kind(source)? == PyKind::Dict {
            additions.extend(source.cast::<PyDict>(runtime)?.items(runtime)?);
        } else {
            let iterator = runtime.iterator(*source)?;
            while let Some(item) = runtime.iterator_next(iterator)? {
                let pair = item.cast::<PySequence>(runtime)?.items(runtime)?;
                if pair.len() != 2 {
                    return Err(PyError::value_error(
                        "dictionary update sequence element has length other than 2",
                    ));
                }
                additions.push((pair[0], pair[1]));
            }
        }
    }
    for (name, value) in args.keywords() {
        additions.push((runtime.new_string(name.clone())?, *value));
    }
    for (key, value) in additions {
        if let Some(position) = find_entry(runtime, &entries, &key)? {
            entries[position].1 = value;
        } else {
            runtime.reserve_memory(96)?;
            entries.push((key, value));
        }
    }
    runtime.replace_dict_items(dict, entries)?;
    Ok(Value::None)
}

fn dict_pop(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.pop", 1, 2)?;
    args.reject_keywords("dict.pop")?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    let mut entries = dict.items(runtime)?;
    if let Some(position) = find_entry(runtime, &entries, &args.positional()[0])? {
        let (_, value) = entries.remove(position);
        runtime.replace_dict_items(dict, entries)?;
        return Ok(value);
    }
    if let Some(default) = args.positional().get(1) {
        return Ok(*default);
    }
    Err(PyError::exception("KeyError", "key not found"))
}

fn set_add(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    set_modify(runtime, receiver, args, SetOperation::Add)
}

fn set_update(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    set_modify(runtime, receiver, args, SetOperation::Update)
}

fn set_remove(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    set_modify(runtime, receiver, args, SetOperation::Remove)
}

fn set_discard(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    set_modify(runtime, receiver, args, SetOperation::Discard)
}

enum SetOperation {
    Add,
    Update,
    Remove,
    Discard,
}

fn set_modify(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    operation: SetOperation,
) -> PyResult {
    args.expect_positional("set method", 1, 1)?;
    args.reject_keywords("set method")?;
    let set = receiver.cast::<PySet>(runtime)?;
    let mut values = set.items(runtime)?;
    let additions = if matches!(operation, SetOperation::Update) {
        let iterator = runtime.iterator(args.positional()[0])?;
        let mut items = Vec::new();
        while let Some(value) = runtime.iterator_next(iterator)? {
            items.push(value);
        }
        items
    } else {
        vec![args.positional()[0]]
    };
    for value in additions {
        let mut position = None;
        for (index, candidate) in values.iter().enumerate() {
            runtime.charge_cpu(1)?;
            if runtime.equals(candidate, &value)? {
                position = Some(index);
                break;
            }
        }
        match operation {
            SetOperation::Add | SetOperation::Update if position.is_none() => {
                runtime.reserve_memory(64)?;
                values.push(value);
            }
            SetOperation::Remove if position.is_none() => {
                return Err(PyError::value_error("set element not found"))
            }
            SetOperation::Remove | SetOperation::Discard if position.is_some() => {
                values.remove(position.unwrap());
            }
            _ => {}
        }
    }
    runtime.replace_set_items(set, values)?;
    Ok(Value::None)
}

fn set_union(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.reject_keywords("set.union")?;
    let mut values = receiver.cast::<PySet>(runtime)?.items(runtime)?;
    for source in args.positional() {
        let iterator = runtime.iterator(*source)?;
        while let Some(value) = runtime.iterator_next(iterator)? {
            let mut present = false;
            for candidate in &values {
                runtime.charge_cpu(1)?;
                if runtime.equals(candidate, &value)? {
                    present = true;
                    break;
                }
            }
            if !present {
                runtime.reserve_memory(64)?;
                values.push(value);
            }
        }
    }
    runtime.new_set(values)
}

fn builtin_map(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("map", 2, usize::MAX)?;
    args.reject_keywords("map")?;
    let function = args.positional()[0].cast::<PyCallable>(runtime)?;
    let mut iterators = Vec::new();
    for value in &args.positional()[1..] {
        iterators.push(runtime.iterator(*value)?);
    }
    let mut result = Vec::new();
    loop {
        let mut values = Vec::with_capacity(iterators.len());
        for iterator in &iterators {
            let Some(value) = runtime.iterator_next(*iterator)? else {
                return runtime.new_iterator(result);
            };
            values.push(value);
        }
        runtime.reserve_memory(64)?;
        result.push(
            function
                .clone()
                .call(runtime, CallArgs::new(values, Vec::new()))?,
        );
    }
}

fn builtin_filter(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("filter", 2, 2)?;
    args.reject_keywords("filter")?;
    let predicate = (runtime.kind(&args.positional()[0])? != PyKind::None)
        .then(|| args.positional()[0].cast::<PyCallable>(runtime))
        .transpose()?;
    let iterator = runtime.iterator(args.positional()[1])?;
    let mut result = Vec::new();
    while let Some(value) = runtime.iterator_next(iterator)? {
        let selected = match &predicate {
            Some(predicate) => {
                let result = predicate
                    .clone()
                    .call(runtime, CallArgs::new(vec![value], Vec::new()))?;
                runtime.truth(&result)?
            }
            None => runtime.truth(&value)?,
        };
        if selected {
            runtime.reserve_memory(64)?;
            result.push(value);
        }
    }
    runtime.new_iterator(result)
}

fn builtin_reversed(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("reversed", 1, 1)?;
    args.reject_keywords("reversed")?;
    let iterator = runtime.iterator(args.positional()[0])?;
    let mut values = Vec::new();
    while let Some(value) = runtime.iterator_next(iterator)? {
        runtime.reserve_memory(64)?;
        values.push(value);
    }
    runtime.charge_cpu(u64::try_from(values.len()).unwrap_or(u64::MAX))?;
    values.reverse();
    runtime.new_iterator(values)
}

fn builtin_getattr(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("getattr", 2, 3)?;
    args.reject_keywords("getattr")?;
    let PyString(name) = args.positional()[1].cast(runtime)?;
    match runtime.get_attribute(args.positional()[0], &name)? {
        Some(value) => Ok(value),
        None => args.positional().get(2).copied().ok_or_else(|| {
            PyError::exception("AttributeError", format!("attribute {name:?} not found"))
        }),
    }
}

fn builtin_hasattr(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("hasattr", 2, 2)?;
    args.reject_keywords("hasattr")?;
    let PyString(name) = args.positional()[1].cast(runtime)?;
    Ok(Value::Bool(
        runtime
            .get_attribute(args.positional()[0], &name)?
            .is_some(),
    ))
}

fn property_setter(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("property.setter", 1, 1)?;
    args.reject_keywords("property.setter")?;
    let property = receiver.cast::<PyProperty>(runtime)?;
    let getter = runtime.property_getter(property)?;
    runtime.new_property(getter, Some(args.positional()[0]))
}

fn type_new(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("type.__new__", 3, 3)?;
    args.reject_keywords("type.__new__")?;
    let PyString(name) = args.positional()[0].cast(runtime)?;
    runtime.new_type(receiver, name, args.positional()[1], args.positional()[2])
}

pub(crate) fn slot_string_add(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let Some(left) = runtime.string_value(&left)? else {
        return Ok(None);
    };
    let Some(right) = runtime.string_value(&right)? else {
        return Ok(None);
    };
    let bytes = left
        .len()
        .checked_add(right.len())
        .ok_or_else(|| PyError::resource_error("string result is too large"))?;
    runtime.reserve_memory(bytes)?;
    runtime.new_string(left + &right).map(Some)
}

pub(crate) fn slot_string_multiply(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    count: PyValue,
) -> PyResult<Option<PyValue>> {
    let Some(value) = runtime.string_value(&value)? else {
        return Ok(None);
    };
    let Some(count) = super::super::number::runtime_repeat_count(runtime, &count)? else {
        return Ok(None);
    };
    let bytes = value
        .len()
        .checked_mul(count)
        .ok_or_else(|| PyError::resource_error("string result is too large"))?;
    runtime.reserve_memory(bytes)?;
    runtime.charge_cpu(u64::try_from(bytes).unwrap_or(u64::MAX))?;
    runtime.new_string(value.repeat(count)).map(Some)
}

pub(crate) fn slot_list_add(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_sequence_add(runtime, PyKind::List, left, right)
}

pub(crate) fn slot_tuple_add(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_sequence_add(runtime, PyKind::Tuple, left, right)
}

fn slot_sequence_add(
    runtime: &mut dyn PyRuntime,
    kind: PyKind,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    if runtime.kind(&right)? != kind {
        return Ok(None);
    }
    let left = left.cast::<PySequence>(runtime)?;
    let right = right.cast::<PySequence>(runtime)?;
    let mut values = left.items(runtime)?;
    let additions = right.items(runtime)?;
    let bytes = values
        .len()
        .checked_add(additions.len())
        .and_then(|length| length.checked_mul(std::mem::size_of::<PyValue>()))
        .ok_or_else(|| PyError::resource_error("sequence result is too large"))?;
    runtime.reserve_memory(bytes)?;
    runtime.charge_cpu(u64::try_from(additions.len()).unwrap_or(u64::MAX))?;
    values.extend(additions);
    match kind {
        PyKind::List => runtime.new_list(values).map(Some),
        PyKind::Tuple => runtime.new_tuple(values).map(Some),
        _ => unreachable!("only concrete sequence slots call this helper"),
    }
}

pub(crate) fn slot_list_multiply(
    runtime: &mut dyn PyRuntime,
    sequence: PyValue,
    count: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_sequence_multiply(runtime, PyKind::List, sequence, count)
}

pub(crate) fn slot_tuple_multiply(
    runtime: &mut dyn PyRuntime,
    sequence: PyValue,
    count: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_sequence_multiply(runtime, PyKind::Tuple, sequence, count)
}

fn slot_sequence_multiply(
    runtime: &mut dyn PyRuntime,
    kind: PyKind,
    sequence: PyValue,
    count: PyValue,
) -> PyResult<Option<PyValue>> {
    let Some(count) = super::super::number::runtime_repeat_count(runtime, &count)? else {
        return Ok(None);
    };
    let values = sequence.cast::<PySequence>(runtime)?.items(runtime)?;
    let length = values
        .len()
        .checked_mul(count)
        .ok_or_else(|| PyError::resource_error("sequence repeat is too large"))?;
    let bytes = length
        .checked_mul(std::mem::size_of::<PyValue>())
        .ok_or_else(|| PyError::resource_error("sequence repeat is too large"))?;
    runtime.reserve_memory(bytes)?;
    runtime.charge_cpu(u64::try_from(length).unwrap_or(u64::MAX))?;
    let mut repeated = Vec::with_capacity(length);
    for _ in 0..count {
        repeated.extend(values.iter().copied());
    }
    match kind {
        PyKind::List => runtime.new_list(repeated).map(Some),
        PyKind::Tuple => runtime.new_tuple(repeated).map(Some),
        _ => unreachable!("only concrete sequence slots call this helper"),
    }
}
