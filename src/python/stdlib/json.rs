//! Bounded JSON conversion over the erased Python runtime value interface.
//!
//! Parsing uses `serde_json` only after bounding and reserving for its temporary tree. Conversion
//! and encoding allocate through [`PyRuntime`], preserve object insertion order, cap recursive
//! depth, and preserve arbitrary-precision JSON integers through the common Python integer API.

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyDict, PyError, PyIdentity, PyKind, PyList, PyResult,
    PyRuntime, PyTuple, PyValue, PyValueCast,
};
use super::super::number::PyNumber;

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
    let source = args.positional()[0]
        .cast::<super::super::native::PyString>(runtime)?
        .0;
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
    let parsed: serde_json::Value = serde_json::from_str(&source)
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
            } else {
                let spelling = value.to_string();
                if spelling.contains(['.', 'e', 'E']) {
                    value
                        .as_f64()
                        .map(Value::Float)
                        .ok_or_else(|| PyError::value_error("invalid JSON number"))
                } else {
                    runtime.new_integer(&spelling)
                }
            }
        }
        serde_json::Value::String(value) => runtime.new_string(value),
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
                entries.push((
                    runtime.new_string(key)?,
                    from_json(runtime, value, depth + 1)?,
                ));
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
                let separators = sequence_items(runtime, *value)?;
                let [item, key] = separators.as_slice() else {
                    return Err(PyError::type_error(
                        "json.dumps separators must be a pair of strings",
                    ));
                };
                item_separator = item.cast::<super::super::native::PyString>(runtime)?.0;
                key_separator = key.cast::<super::super::native::PyString>(runtime)?.0;
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

    let value = args.positional()[0];
    let bound = size_bound(runtime, value, 0, &mut Vec::new())?;
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
    runtime.new_string(rendered)
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
    active: &mut Vec<PyIdentity>,
) -> PyResult<String> {
    if depth >= MAX_JSON_DEPTH {
        return Err(PyError::value_error("maximum JSON nesting depth exceeded"));
    }
    runtime.charge_cpu(1)?;
    let kind = runtime.kind(&value)?;
    match kind {
        PyKind::None => return Ok("null".into()),
        PyKind::Bool => return Ok(runtime.truth(&value)?.to_string()),
        PyKind::Int => {
            return runtime
                .integer_text(&value)?
                .ok_or_else(|| PyError::type_error("invalid integer representation"));
        }
        PyKind::Float => {
            let PyNumber::Float(number) = value.cast::<PyNumber>(runtime)? else {
                return Err(PyError::runtime_error(
                    "float changed numeric representation",
                ));
            };
            return if number.is_finite() {
                serde_json::to_string(&number)
                    .map_err(|error| PyError::value_error(error.to_string()))
            } else {
                Err(PyError::value_error(
                    "non-finite float is not JSON serializable",
                ))
            };
        }
        PyKind::String => {
            let value = runtime
                .string_value(&value)?
                .ok_or_else(|| PyError::runtime_error("string changed representation"))?;
            return serde_json::to_string(&value)
                .map_err(|error| PyError::value_error(error.to_string()));
        }
        _ => {}
    }
    if matches!(kind, PyKind::List | PyKind::Tuple | PyKind::Dict) {
        let id = runtime
            .identity(&value)
            .ok_or_else(|| PyError::runtime_error("container has no stable identity"))?;
        if active.contains(&id) {
            return Err(PyError::value_error(
                "circular reference detected while encoding JSON",
            ));
        }
        return match kind {
            PyKind::List | PyKind::Tuple => {
                active.push(id);
                let values = sequence_items(runtime, value)?;
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
                let mut entries = value.cast::<PyDict>(runtime)?.items(runtime)?;
                if sort_keys {
                    for (key, _) in &entries {
                        if runtime.string_value(key)?.is_none() {
                            return Err(PyError::type_error(
                                "json.dumps sort_keys requires string keys",
                            ));
                        }
                    }
                    for index in 1..entries.len() {
                        let mut current = index;
                        while current > 0 {
                            runtime.charge_cpu(1)?;
                            let left = runtime
                                .string_value(&entries[current - 1].0)?
                                .expect("validated string key");
                            let right = runtime
                                .string_value(&entries[current].0)?
                                .expect("validated string key");
                            let should_swap = left > right;
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
                    let Some(key) = runtime.string_value(&key)? else {
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
        };
    }
    Err(PyError::type_error("object is not JSON serializable"))
}

fn size_bound(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    depth: usize,
    active: &mut Vec<PyIdentity>,
) -> PyResult<usize> {
    if depth >= MAX_JSON_DEPTH {
        return Err(PyError::value_error("maximum JSON nesting depth exceeded"));
    }
    runtime.charge_cpu(1)?;
    let scalar = |size: usize| {
        size.checked_add(64)
            .ok_or_else(|| PyError::resource_error("json result is too large"))
    };
    let kind = runtime.kind(&value)?;
    match kind {
        PyKind::None => return scalar(4),
        PyKind::Bool => return scalar(if runtime.truth(&value)? { 4 } else { 5 }),
        PyKind::Int => {
            let text = runtime
                .integer_text(&value)?
                .ok_or_else(|| PyError::type_error("invalid integer representation"))?;
            return scalar(text.len().max(32));
        }
        PyKind::Float => {
            let PyNumber::Float(number) = value.cast::<PyNumber>(runtime)? else {
                return Err(PyError::runtime_error(
                    "float changed numeric representation",
                ));
            };
            return if number.is_finite() {
                scalar(64)
            } else {
                Err(PyError::value_error(
                    "non-finite float is not JSON serializable",
                ))
            };
        }
        PyKind::String => {
            let value = runtime
                .string_value(&value)?
                .ok_or_else(|| PyError::runtime_error("string changed representation"))?;
            return scalar(
                value
                    .len()
                    .checked_mul(6)
                    .and_then(|bytes| bytes.checked_add(2))
                    .ok_or_else(|| PyError::resource_error("json result is too large"))?,
            );
        }
        _ => {}
    }
    if matches!(kind, PyKind::List | PyKind::Tuple | PyKind::Dict) {
        let id = runtime
            .identity(&value)
            .ok_or_else(|| PyError::runtime_error("container has no stable identity"))?;
        if active.contains(&id) {
            return Err(PyError::value_error(
                "circular reference detected while encoding JSON",
            ));
        }
        active.push(id);
        let result = match kind {
            PyKind::List | PyKind::Tuple => {
                let values = sequence_items(runtime, value)?;
                let mut size = 2usize;
                for (index, child) in values.into_iter().enumerate() {
                    if index != 0 {
                        size = size
                            .checked_add(1)
                            .ok_or_else(|| PyError::resource_error("json result is too large"))?;
                    }
                    size = size
                        .checked_add(size_bound(runtime, child, depth + 1, active)?)
                        .and_then(|bytes| bytes.checked_add(64))
                        .ok_or_else(|| PyError::resource_error("json result is too large"))?;
                }
                Ok(size)
            }
            PyKind::Dict => {
                let entries = value.cast::<PyDict>(runtime)?.items(runtime)?;
                let mut size = 2usize;
                for (index, (key, child)) in entries.into_iter().enumerate() {
                    let Some(key) = runtime.string_value(&key)? else {
                        return Err(PyError::type_error(
                            "json.dumps currently requires string keys",
                        ));
                    };
                    if index != 0 {
                        size = size
                            .checked_add(1)
                            .ok_or_else(|| PyError::resource_error("json result is too large"))?;
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
    } else {
        Err(PyError::type_error("object is not JSON serializable"))
    }
}
