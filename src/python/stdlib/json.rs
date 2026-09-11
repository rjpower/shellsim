//! Bounded JSON conversion over the erased Python runtime value interface.
//!
//! Parsing uses `serde_json` only after bounding and reserving for its temporary tree. Conversion
//! and encoding allocate through [`PyRuntime`], preserve object insertion order, cap recursive
//! depth, and reject values outside shellsim's current signed-64-bit integer model.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyDict, PyError, PyKind, PyList, PyResult, PyRuntime,
    PyTuple, PyValue, PyValueCast,
};
use super::super::Value;

const MAX_JSON_INPUT: usize = 1024 * 1024;
const MAX_JSON_DEPTH: usize = 128;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "json",
    functions: &[
        FunctionDef {
            module: "json",
            name: "dumps",
            call: dumps,
        },
        FunctionDef {
            module: "json",
            name: "loads",
            call: loads,
        },
    ],
    values: &[],
};

fn loads(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("loads", 1, 1)?;
    args.reject_keywords("loads")?;
    let Value::String(source) = &args.positional()[0] else {
        return Err(PyError::type_error("json.loads() expects a string"));
    };
    if source.len() > MAX_JSON_INPUT {
        return Err(PyError::value_error("JSON input exceeds 1 MiB"));
    }

    runtime.charge_cpu(u64::try_from(source.len()).unwrap_or(u64::MAX))?;
    let scratch = source
        .len()
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(4096))
        .ok_or_else(|| PyError::resource_error("JSON parser scratch overflow"))?;
    runtime.reserve_memory(scratch)?;
    let parsed: serde_json::Value = serde_json::from_str(source)
        .map_err(|error| PyError::value_error(format!("invalid JSON: {error}")))?;
    from_json(runtime, parsed, 0)
}

fn from_json(runtime: &mut dyn PyRuntime, value: serde_json::Value, depth: usize) -> PyResult {
    if depth >= MAX_JSON_DEPTH {
        return Err(PyError::value_error("maximum JSON nesting depth exceeded"));
    }
    runtime.charge_cpu(1)?;
    match value {
        serde_json::Value::Null => Ok(Value::None),
        serde_json::Value::Bool(value) => Ok(Value::Bool(value)),
        serde_json::Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                Ok(Value::Int(value))
            } else if let Some(value) = value.as_u64() {
                i64::try_from(value).map(Value::Int).map_err(|_| {
                    PyError::overflow_error("JSON integer exceeds bounded integer range")
                })
            } else {
                value
                    .as_f64()
                    .map(Value::Float)
                    .ok_or_else(|| PyError::value_error("invalid JSON number"))
            }
        }
        serde_json::Value::String(value) => {
            runtime.reserve_memory(value.len())?;
            Ok(Value::String(value))
        }
        serde_json::Value::Array(values) => {
            let mut items = Vec::with_capacity(values.len());
            for value in values {
                items.push(from_json(runtime, value, depth + 1)?);
            }
            runtime.new_list(items)
        }
        serde_json::Value::Object(values) => {
            let mut entries = Vec::with_capacity(values.len());
            for (key, value) in values {
                runtime.reserve_memory(key.len())?;
                entries.push((Value::String(key), from_json(runtime, value, depth + 1)?));
            }
            runtime.new_dict(entries)
        }
    }
}

fn dumps(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("dumps", 1, 1)?;
    let mut item_separator = ", ".to_string();
    let mut key_separator = ": ".to_string();
    let mut sort_keys = false;
    let mut saw_separators = false;
    let mut saw_sort_keys = false;
    for (name, value) in args.keywords() {
        match name.as_str() {
            "separators" if !saw_separators => {
                let separators = sequence_items(runtime, value.clone())?;
                let [Value::String(item), Value::String(key)] = separators.as_slice() else {
                    return Err(PyError::type_error(
                        "json.dumps separators must be a pair of strings",
                    ));
                };
                item_separator = item.clone();
                key_separator = key.clone();
                saw_separators = true;
            }
            "sort_keys" if !saw_sort_keys => {
                sort_keys = runtime.truth(value)?;
                saw_sort_keys = true;
            }
            "separators" | "sort_keys" => {
                return Err(PyError::type_error(format!(
                    "json.dumps got multiple values for keyword {name:?}"
                )))
            }
            _ => {
                return Err(PyError::type_error(format!(
                    "json.dumps keyword argument {name:?} is not implemented"
                )))
            }
        }
    }

    let value = args.positional()[0].clone();
    let bound = size_bound(runtime, value.clone(), 0, &mut Vec::new())?;
    let bound = bound
        .checked_add(bound / 2)
        .and_then(|bytes| bytes.checked_add(256))
        .ok_or_else(|| PyError::resource_error("json result is too large"))?;
    runtime.reserve_memory(bound)?;
    let rendered = dump_value(
        runtime,
        value,
        &item_separator,
        &key_separator,
        sort_keys,
        0,
        &mut Vec::new(),
    )?;
    Ok(Value::String(rendered))
}

fn sequence_items(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<PyValue>> {
    match runtime.kind(&value)? {
        PyKind::List => value.cast::<PyList>(runtime)?.items(runtime),
        PyKind::Tuple => value.cast::<PyTuple>(runtime)?.items(runtime),
        _ => Err(PyError::type_error("expected a list or tuple")),
    }
}

fn dump_value(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    item_separator: &str,
    key_separator: &str,
    sort_keys: bool,
    depth: usize,
    active: &mut Vec<super::super::heap::ObjectId>,
) -> PyResult<String> {
    if depth >= MAX_JSON_DEPTH {
        return Err(PyError::value_error("maximum JSON nesting depth exceeded"));
    }
    runtime.charge_cpu(1)?;
    match value {
        Value::None => Ok("null".into()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Int(value) => Ok(value.to_string()),
        Value::Float(value) if value.is_finite() => {
            serde_json::to_string(&value).map_err(|error| PyError::value_error(error.to_string()))
        }
        Value::String(value) => {
            serde_json::to_string(&value).map_err(|error| PyError::value_error(error.to_string()))
        }
        Value::Object(id) => {
            if active.contains(&id) {
                return Err(PyError::value_error(
                    "circular reference detected while encoding JSON",
                ));
            }
            match runtime.kind(&Value::Object(id))? {
                PyKind::List | PyKind::Tuple => {
                    active.push(id);
                    let values = sequence_items(runtime, Value::Object(id))?;
                    let mut rendered = Vec::with_capacity(values.len());
                    for value in values {
                        rendered.push(dump_value(
                            runtime,
                            value,
                            item_separator,
                            key_separator,
                            sort_keys,
                            depth + 1,
                            active,
                        )?);
                    }
                    active.pop();
                    Ok(format!("[{}]", rendered.join(item_separator)))
                }
                PyKind::Dict => {
                    active.push(id);
                    let mut entries = Value::Object(id).cast::<PyDict>(runtime)?.items(runtime)?;
                    if sort_keys {
                        for (key, _) in &entries {
                            if !matches!(key, Value::String(_)) {
                                return Err(PyError::type_error(
                                    "json.dumps sort_keys requires string keys",
                                ));
                            }
                        }
                        for index in 1..entries.len() {
                            let mut current = index;
                            while current > 0 {
                                runtime.charge_cpu(1)?;
                                let should_swap =
                                    match (&entries[current - 1].0, &entries[current].0) {
                                        (Value::String(left), Value::String(right)) => left > right,
                                        _ => unreachable!(),
                                    };
                                if !should_swap {
                                    break;
                                }
                                entries.swap(current - 1, current);
                                current -= 1;
                            }
                        }
                    }
                    let mut rendered = Vec::with_capacity(entries.len());
                    for (key, value) in entries {
                        let Value::String(key) = key else {
                            return Err(PyError::type_error(
                                "json.dumps currently requires string keys",
                            ));
                        };
                        let key = serde_json::to_string(&key)
                            .map_err(|error| PyError::value_error(error.to_string()))?;
                        let value = dump_value(
                            runtime,
                            value,
                            item_separator,
                            key_separator,
                            sort_keys,
                            depth + 1,
                            active,
                        )?;
                        rendered.push(format!("{key}{key_separator}{value}"));
                    }
                    active.pop();
                    Ok(format!("{{{}}}", rendered.join(item_separator)))
                }
                _ => Err(PyError::type_error("object is not JSON serializable")),
            }
        }
        Value::Float(_) => Err(PyError::value_error(
            "non-finite float is not JSON serializable",
        )),
        Value::Exception { .. } | Value::Native(_) => {
            Err(PyError::type_error("object is not JSON serializable"))
        }
    }
}

fn size_bound(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    depth: usize,
    active: &mut Vec<super::super::heap::ObjectId>,
) -> PyResult<usize> {
    if depth >= MAX_JSON_DEPTH {
        return Err(PyError::value_error("maximum JSON nesting depth exceeded"));
    }
    runtime.charge_cpu(1)?;
    let scalar = |size: usize| {
        size.checked_add(64)
            .ok_or_else(|| PyError::resource_error("json result is too large"))
    };
    match value {
        Value::None => scalar(4),
        Value::Bool(value) => scalar(if value { 4 } else { 5 }),
        Value::Int(_) => scalar(32),
        Value::Float(value) if value.is_finite() => scalar(64),
        Value::String(value) => scalar(
            value
                .len()
                .checked_mul(6)
                .and_then(|bytes| bytes.checked_add(2))
                .ok_or_else(|| PyError::resource_error("json result is too large"))?,
        ),
        Value::Float(_) => Err(PyError::value_error(
            "non-finite float is not JSON serializable",
        )),
        Value::Exception { .. } | Value::Native(_) => {
            Err(PyError::type_error("object is not JSON serializable"))
        }
        Value::Object(id) => {
            if active.contains(&id) {
                return Err(PyError::value_error(
                    "circular reference detected while encoding JSON",
                ));
            }
            active.push(id);
            let result = match runtime.kind(&Value::Object(id))? {
                PyKind::List | PyKind::Tuple => {
                    let values = sequence_items(runtime, Value::Object(id))?;
                    let mut size = 2usize;
                    for (index, child) in values.into_iter().enumerate() {
                        if index != 0 {
                            size = size.checked_add(1).ok_or_else(|| {
                                PyError::resource_error("json result is too large")
                            })?;
                        }
                        size = size
                            .checked_add(size_bound(runtime, child, depth + 1, active)?)
                            .and_then(|bytes| bytes.checked_add(64))
                            .ok_or_else(|| PyError::resource_error("json result is too large"))?;
                    }
                    Ok(size)
                }
                PyKind::Dict => {
                    let entries = Value::Object(id).cast::<PyDict>(runtime)?.items(runtime)?;
                    let mut size = 2usize;
                    for (index, (key, child)) in entries.into_iter().enumerate() {
                        let Value::String(key) = key else {
                            return Err(PyError::type_error(
                                "json.dumps currently requires string keys",
                            ));
                        };
                        if index != 0 {
                            size = size.checked_add(1).ok_or_else(|| {
                                PyError::resource_error("json result is too large")
                            })?;
                        }
                        let key_size = key
                            .len()
                            .checked_mul(6)
                            .and_then(|bytes| bytes.checked_add(2))
                            .ok_or_else(|| PyError::resource_error("json result is too large"))?;
                        let child_size = size_bound(runtime, child, depth + 1, active)?;
                        size = size
                            .checked_add(key_size)
                            .and_then(|bytes| bytes.checked_add(1))
                            .and_then(|bytes| bytes.checked_add(child_size))
                            .and_then(|bytes| bytes.checked_add(64))
                            .ok_or_else(|| PyError::resource_error("json result is too large"))?;
                    }
                    Ok(size)
                }
                _ => Err(PyError::type_error("object is not JSON serializable")),
            };
            active.pop();
            result
        }
    }
}
