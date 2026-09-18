//! Native descriptors for methods on builtin Python value types.
//!
//! Methods use checked, type-erased runtime views and snapshot-and-commit mutation. This keeps
//! collection layouts and compact scalar tags private to the runtime while giving builtin and
//! user-defined methods the same descriptor call path.

use std::cmp::Ordering;

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, MethodDef, NativeTypeDef, PyByteArray, PyBytes, PyCallable, PyDict,
    PyError, PyKind, PyList, PyProperty, PyResult, PyRuntime, PySequence, PySet, PyString, PyValue,
    PyValueCast,
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
        method("str", "splitlines", string_splitlines),
        method("str", "join", string_join),
        method("str", "replace", string_replace),
        method("str", "format", string_format),
        method("str", "encode", string_encode),
        method("str", "lower", string_lower),
        method("str", "upper", string_upper),
        method("str", "zfill", string_zfill),
        method("str", "isalnum", string_isalnum),
        method("str", "isalpha", string_isalpha),
        method("str", "isdigit", string_isdigit),
        method("str", "islower", string_islower),
        method("str", "isupper", string_isupper),
    ],
};

pub(crate) static BYTES_TYPE: NativeTypeDef = NativeTypeDef {
    name: "bytes",
    methods: &[
        method("bytes", "decode", bytes_decode),
        method("bytes", "hex", bytes_hex),
        method("bytes", "startswith", bytes_startswith),
        method("bytes", "endswith", bytes_endswith),
        method("bytes", "find", bytes_find),
    ],
};

pub(crate) static BYTEARRAY_TYPE: NativeTypeDef = NativeTypeDef {
    name: "bytearray",
    methods: &[
        method("bytearray", "append", bytearray_append),
        method("bytearray", "extend", bytearray_extend),
        method("bytearray", "decode", bytes_decode),
        method("bytearray", "hex", bytes_hex),
        method("bytearray", "find", bytes_find),
    ],
};

pub(crate) static LIST_TYPE: NativeTypeDef = NativeTypeDef {
    name: "list",
    methods: &[
        method("list", "append", list_append),
        method("list", "insert", list_insert),
        method("list", "extend", list_extend),
        method("list", "pop", list_pop),
        method("list", "remove", list_remove),
        method("list", "reverse", list_reverse),
        method("list", "clear", list_clear),
        method("list", "count", list_count),
        method("list", "index", list_index),
        method("list", "sort", list_sort),
        method("list", "copy", list_copy),
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
        method("dict", "copy", dict_copy),
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
        method("set", "copy", set_copy),
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

fn string_encode(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.encode", 0, 2)?;
    args.reject_keywords("str.encode")?;
    let PyString(value) = receiver.cast(runtime)?;
    let encoding = args
        .positional()
        .first()
        .map(|value| value.cast::<PyString>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or_else(|| "utf-8".into())
        .to_ascii_lowercase()
        .replace('_', "-");
    let errors = args
        .positional()
        .get(1)
        .map(|value| value.cast::<PyString>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or_else(|| "strict".into());
    if errors != "strict" {
        return Err(PyError::value_error(
            "only strict encoding errors are supported",
        ));
    }
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let encoded = match encoding.as_str() {
        "utf-8" | "utf8" => value.into_bytes(),
        "ascii" => {
            if !value.is_ascii() {
                return Err(PyError::exception(
                    "UnicodeEncodeError",
                    "character is outside the ASCII range",
                ));
            }
            value.into_bytes()
        }
        "latin-1" | "latin1" | "iso-8859-1" => value
            .chars()
            .map(|character| {
                u8::try_from(u32::from(character)).map_err(|_| {
                    PyError::exception(
                        "UnicodeEncodeError",
                        "character is outside the Latin-1 range",
                    )
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
        _ => return Err(PyError::value_error("unknown text encoding")),
    };
    runtime.new_bytes(encoded)
}

fn string_lower(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_transform(
        runtime,
        receiver,
        args,
        |value| value.to_lowercase(),
        "str.lower",
    )
}

fn string_upper(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_transform(
        runtime,
        receiver,
        args,
        |value| value.to_uppercase(),
        "str.upper",
    )
}

fn string_transform(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    transform: fn(&str) -> String,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 0, 0)?;
    args.reject_keywords(name)?;
    let PyString(value) = receiver.cast(runtime)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    runtime.new_string(transform(&value))
}

fn string_isalnum(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_predicate(
        runtime,
        receiver,
        args,
        char::is_alphanumeric,
        "str.isalnum",
    )
}

fn string_isalpha(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_predicate(runtime, receiver, args, char::is_alphabetic, "str.isalpha")
}

fn string_isdigit(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_predicate(runtime, receiver, args, char::is_numeric, "str.isdigit")
}

fn string_islower(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_case_predicate(runtime, receiver, args, false, "str.islower")
}

fn string_isupper(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_case_predicate(runtime, receiver, args, true, "str.isupper")
}

fn string_case_predicate(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    uppercase: bool,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 0, 0)?;
    args.reject_keywords(name)?;
    let PyString(value) = receiver.cast(runtime)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let mut cased = false;
    for character in value.chars() {
        if character.is_uppercase() || character.is_lowercase() {
            cased = true;
            if uppercase != character.is_uppercase() {
                return Ok(PyValue::Bool(false));
            }
        }
    }
    Ok(PyValue::Bool(cased))
}

fn string_predicate(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    predicate: fn(char) -> bool,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 0, 0)?;
    args.reject_keywords(name)?;
    let PyString(value) = receiver.cast(runtime)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    Ok(PyValue::Bool(
        !value.is_empty() && value.chars().all(predicate),
    ))
}

fn bytes_decode(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytes.decode", 0, 2)?;
    args.reject_keywords("bytes.decode")?;
    let encoding = args
        .positional()
        .first()
        .map(|value| value.cast::<PyString>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or_else(|| "utf-8".into());
    let errors = args
        .positional()
        .get(1)
        .map(|value| value.cast::<PyString>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or_else(|| "strict".into());
    if errors != "strict" {
        return Err(PyError::value_error(
            "only strict decoding errors are supported",
        ));
    }
    let PyBytes(value) = receiver.cast(runtime)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let normalized = encoding.to_ascii_lowercase().replace('_', "-");
    let decoded = match normalized.as_str() {
        "utf-8" | "utf8" => String::from_utf8(value)
            .map_err(|_| PyError::exception("UnicodeDecodeError", "invalid UTF-8 byte sequence"))?,
        "ascii" => {
            if !value.is_ascii() {
                return Err(PyError::exception(
                    "UnicodeDecodeError",
                    "byte is outside the ASCII range",
                ));
            }
            String::from_utf8(value).expect("ASCII is valid UTF-8")
        }
        "latin-1" | "latin1" | "iso-8859-1" => value.into_iter().map(char::from).collect(),
        _ => return Err(PyError::value_error("unknown text encoding")),
    };
    runtime.new_string(decoded)
}

fn bytes_hex(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytes.hex", 0, 0)?;
    args.reject_keywords("bytes.hex")?;
    let PyBytes(value) = receiver.cast(runtime)?;
    let length = value
        .len()
        .checked_mul(2)
        .ok_or_else(|| PyError::resource_error("hex result is too large"))?;
    runtime.reserve_memory(length)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let mut result = String::with_capacity(length);
    for byte in value {
        use std::fmt::Write;
        write!(&mut result, "{byte:02x}").expect("writing to a string cannot fail");
    }
    runtime.new_string(result)
}

fn bytes_startswith(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    bytes_affix(runtime, receiver, args, true)
}

fn bytes_endswith(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    bytes_affix(runtime, receiver, args, false)
}

fn bytes_affix(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    prefix: bool,
) -> PyResult {
    args.expect_positional("bytes affix test", 1, 1)?;
    args.reject_keywords("bytes affix test")?;
    let PyBytes(value) = receiver.cast(runtime)?;
    let PyBytes(needle) = args.positional()[0].cast(runtime)?;
    Ok(PyValue::Bool(if prefix {
        value.starts_with(&needle)
    } else {
        value.ends_with(&needle)
    }))
}

fn bytes_find(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytes.find", 1, 3)?;
    args.reject_keywords("bytes.find")?;
    let PyBytes(value) = receiver.cast(runtime)?;
    let PyBytes(needle) = args.positional()[0].cast(runtime)?;
    let optional_index = |value: Option<&PyValue>, default| -> PyResult<i64> {
        value
            .map(|value| {
                runtime
                    .int_value(value)
                    .ok_or_else(|| PyError::type_error("slice indices must be integers"))
            })
            .transpose()
            .map(|value| value.unwrap_or(default))
    };
    let start = optional_index(args.positional().get(1), 0)?;
    let end = optional_index(
        args.positional().get(2),
        i64::try_from(value.len()).unwrap_or(i64::MAX),
    )?;
    let length = i64::try_from(value.len()).unwrap_or(i64::MAX);
    let normalize = |index: i64| {
        usize::try_from(if index < 0 {
            index.saturating_add(length).max(0)
        } else {
            index
        })
        .unwrap_or(value.len())
        .min(value.len())
    };
    let start = normalize(start);
    let end = normalize(end);
    runtime.charge_cpu(u64::try_from(end.saturating_sub(start)).unwrap_or(u64::MAX))?;
    let found = if start <= end && needle.len() <= end - start {
        if needle.is_empty() {
            Some(start)
        } else {
            value[start..end]
                .windows(needle.len())
                .position(|window| window == needle)
                .map(|position| start + position)
        }
    } else {
        None
    };
    Ok(PyValue::Int(
        found
            .and_then(|value| i64::try_from(value).ok())
            .unwrap_or(-1),
    ))
}

fn bytearray_append(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytearray.append", 1, 1)?;
    args.reject_keywords("bytearray.append")?;
    let array = receiver.cast::<PyByteArray>(runtime)?;
    let byte = runtime
        .int_value(&args.positional()[0])
        .and_then(|value| u8::try_from(value).ok())
        .ok_or_else(|| PyError::value_error("byte must be in range(0, 256)"))?;
    let mut items = runtime.bytearray_items(array)?;
    items.push(byte);
    runtime.replace_bytearray_items(array, items)?;
    Ok(PyValue::None)
}

fn bytearray_extend(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytearray.extend", 1, 1)?;
    args.reject_keywords("bytearray.extend")?;
    let array = receiver.cast::<PyByteArray>(runtime)?;
    let iterator = runtime.iterator(args.positional()[0])?;
    let mut additions = Vec::new();
    while let Some(value) = runtime.iterator_next(iterator)? {
        additions.push(
            runtime
                .int_value(&value)
                .and_then(|value| u8::try_from(value).ok())
                .ok_or_else(|| PyError::value_error("byte must be in range(0, 256)"))?,
        );
    }
    let mut items = runtime.bytearray_items(array)?;
    let length = items
        .len()
        .checked_add(additions.len())
        .ok_or_else(|| PyError::resource_error("bytearray is too large"))?;
    runtime.reserve_memory(length)?;
    items.extend(additions);
    runtime.replace_bytearray_items(array, items)?;
    Ok(PyValue::None)
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

fn string_zfill(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.zfill", 1, 1)?;
    args.reject_keywords("str.zfill")?;
    let PyString(value) = receiver.cast(runtime)?;
    let width = runtime
        .int_value(&args.positional()[0])
        .ok_or_else(|| PyError::type_error("width must be an integer"))?;
    let width = usize::try_from(width).unwrap_or(0);
    let length = value.chars().count();
    if width <= length {
        return runtime.new_string(value);
    }
    let padding = width - length;
    let capacity = value
        .len()
        .checked_add(padding)
        .ok_or_else(|| PyError::resource_error("filled string is too large"))?;
    runtime.reserve_memory(capacity)?;
    runtime.charge_cpu(u64::try_from(capacity).unwrap_or(u64::MAX))?;
    let mut result = String::with_capacity(capacity);
    let (sign, digits) = value
        .strip_prefix(['+', '-'])
        .map_or((None, value.as_str()), |digits| {
            (value.chars().next(), digits)
        });
    if let Some(sign) = sign {
        result.push(sign);
    }
    result.extend(std::iter::repeat_n('0', padding));
    result.push_str(digits);
    runtime.new_string(result)
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

fn string_splitlines(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.splitlines", 0, 1)?;
    args.reject_keywords("str.splitlines")?;
    let keepends = args
        .positional()
        .first()
        .map(|value| runtime.truth(value))
        .transpose()?
        .unwrap_or(false);
    let PyString(value) = receiver.cast(runtime)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let mut lines = Vec::new();
    let mut start = 0;
    let mut characters = value.char_indices().peekable();
    while let Some((index, character)) = characters.next() {
        let mut end = index + character.len_utf8();
        let boundary = matches!(
            character,
            '\n' | '\r'
                | '\u{000b}'
                | '\u{000c}'
                | '\u{001c}'
                | '\u{001d}'
                | '\u{001e}'
                | '\u{0085}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if !boundary {
            continue;
        }
        if character == '\r' && characters.peek().is_some_and(|(_, next)| *next == '\n') {
            end = characters.next().expect("peeked LF").0 + 1;
        }
        let line_end = if keepends { end } else { index };
        lines.push(runtime.new_string(value[start..line_end].to_string())?);
        start = end;
    }
    if start < value.len() {
        lines.push(runtime.new_string(value[start..].to_string())?);
    }
    runtime.new_list(lines)
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

fn list_insert(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.insert", 2, 2)?;
    args.reject_keywords("list.insert")?;
    let list = receiver.cast::<PyList>(runtime)?;
    let mut values = list.items(runtime)?;
    let raw = runtime
        .int_value(&args.positional()[0])
        .ok_or_else(|| PyError::type_error("list index must be an integer"))?;
    let len = i64::try_from(values.len()).map_err(|_| PyError::overflow_error("list too large"))?;
    let index = if raw < 0 {
        usize::try_from(len.saturating_add(raw).max(0)).unwrap_or(0)
    } else {
        usize::try_from(raw).unwrap_or(usize::MAX).min(values.len())
    };
    runtime.reserve_memory(64)?;
    values.insert(index, args.positional()[1]);
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

fn list_clear(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.clear", 0, 0)?;
    args.reject_keywords("list.clear")?;
    let list = receiver.cast::<PyList>(runtime)?;
    runtime.replace_list_items(list, Vec::new())?;
    Ok(PyValue::None)
}

fn list_copy(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.copy", 0, 0)?;
    args.reject_keywords("list.copy")?;
    let values = receiver.cast::<PyList>(runtime)?.items(runtime)?;
    runtime.new_list(values)
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

fn dict_copy(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.copy", 0, 0)?;
    args.reject_keywords("dict.copy")?;
    let entries = receiver.cast::<PyDict>(runtime)?.items(runtime)?;
    runtime.new_dict(entries)
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

fn set_copy(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("set.copy", 0, 0)?;
    args.reject_keywords("set.copy")?;
    let values = receiver.cast::<PySet>(runtime)?.items(runtime)?;
    runtime.new_set(values)
}

fn set_is_subset(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<(bool, bool)> {
    let left = left.cast::<PySet>(runtime)?.items(runtime)?;
    let right = right.cast::<PySet>(runtime)?.items(runtime)?;
    let mut subset = true;
    for value in &left {
        let mut present = false;
        for candidate in &right {
            runtime.charge_cpu(1)?;
            if runtime.equals(value, candidate)? {
                present = true;
                break;
            }
        }
        if !present {
            subset = false;
            break;
        }
    }
    Ok((subset, left.len() < right.len()))
}

pub(crate) fn slot_set_less(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let (subset, smaller) = set_is_subset(runtime, left, right)?;
    Ok(Some(Value::Bool(subset && smaller)))
}

pub(crate) fn slot_set_less_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    Ok(Some(Value::Bool(set_is_subset(runtime, left, right)?.0)))
}

pub(crate) fn slot_set_greater(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_set_less(runtime, right, left)
}

pub(crate) fn slot_set_greater_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_set_less_equal(runtime, right, left)
}

pub(crate) fn slot_set_subtract(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let left = left.cast::<PySet>(runtime)?.items(runtime)?;
    let Ok(right) = right.cast::<PySet>(runtime) else {
        return Ok(None);
    };
    let right = right.items(runtime)?;
    let mut difference = Vec::new();
    for value in left {
        let mut present = false;
        for candidate in &right {
            runtime.charge_cpu(1)?;
            if runtime.equals(&value, candidate)? {
                present = true;
                break;
            }
        }
        if !present {
            difference.push(value);
        }
    }
    runtime.new_set(difference).map(Some)
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

pub(crate) fn slot_bytes_length(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let Some(value) = runtime.bytes_value(&value)? else {
        return Ok(None);
    };
    let length = i64::try_from(value.len())
        .map_err(|_| PyError::overflow_error("bytes object is too large"))?;
    Ok(Some(PyValue::Int(length)))
}

pub(crate) fn slot_bytes_get_item(
    runtime: &mut dyn PyRuntime,
    owner: PyValue,
    index: PyValue,
) -> PyResult<Option<PyValue>> {
    let Some(value) = runtime.bytes_value(&owner)? else {
        return Ok(None);
    };
    let Some(index) = runtime.int_value(&index) else {
        return Ok(None);
    };
    let length = i64::try_from(value.len()).unwrap_or(i64::MAX);
    let index = if index < 0 {
        index.checked_add(length)
    } else {
        Some(index)
    }
    .and_then(|index| usize::try_from(index).ok())
    .filter(|index| *index < value.len())
    .ok_or_else(|| PyError::exception("IndexError", "index out of range"))?;
    Ok(Some(PyValue::Int(i64::from(value[index]))))
}

pub(crate) fn slot_bytes_add(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let (Some(mut left), Some(right)) = (runtime.bytes_value(&left)?, runtime.bytes_value(&right)?)
    else {
        return Ok(None);
    };
    let length = left
        .len()
        .checked_add(right.len())
        .ok_or_else(|| PyError::resource_error("bytes result is too large"))?;
    runtime.reserve_memory(length)?;
    runtime.charge_cpu(u64::try_from(right.len()).unwrap_or(u64::MAX))?;
    left.extend(right);
    runtime.new_bytes(left).map(Some)
}

pub(crate) fn slot_bytes_multiply(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let (value, count) = if let Some(value) = runtime.bytes_value(&left)? {
        (value, right)
    } else if let Some(value) = runtime.bytes_value(&right)? {
        (value, left)
    } else {
        return Ok(None);
    };
    let Some(count) = super::super::number::runtime_repeat_count(runtime, &count)? else {
        return Ok(None);
    };
    let length = value
        .len()
        .checked_mul(count)
        .ok_or_else(|| PyError::resource_error("bytes result is too large"))?;
    runtime.reserve_memory(length)?;
    runtime.charge_cpu(u64::try_from(length).unwrap_or(u64::MAX))?;
    runtime.new_bytes(value.repeat(count)).map(Some)
}

pub(crate) fn slot_bytearray_length(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = value.cast::<PyByteArray>(runtime)?;
    let length = i64::try_from(runtime.bytearray_items(array)?.len())
        .map_err(|_| PyError::overflow_error("bytearray is too large"))?;
    Ok(Some(PyValue::Int(length)))
}

pub(crate) fn slot_bytearray_add(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    if runtime.kind(&left)? != PyKind::ByteArray {
        return Ok(None);
    }
    let (Some(mut left), Some(right)) = (runtime.bytes_value(&left)?, runtime.bytes_value(&right)?)
    else {
        return Ok(None);
    };
    let length = left
        .len()
        .checked_add(right.len())
        .ok_or_else(|| PyError::resource_error("bytearray result is too large"))?;
    runtime.reserve_memory(length)?;
    runtime.charge_cpu(u64::try_from(right.len()).unwrap_or(u64::MAX))?;
    left.extend(right);
    runtime.new_bytearray(left).map(Some)
}

pub(crate) fn slot_bytearray_multiply(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let (value, count) = if runtime.kind(&left)? == PyKind::ByteArray {
        (runtime.bytes_value(&left)?.unwrap_or_default(), right)
    } else if runtime.kind(&right)? == PyKind::ByteArray {
        (runtime.bytes_value(&right)?.unwrap_or_default(), left)
    } else {
        return Ok(None);
    };
    let Some(count) = super::super::number::runtime_repeat_count(runtime, &count)? else {
        return Ok(None);
    };
    let length = value
        .len()
        .checked_mul(count)
        .ok_or_else(|| PyError::resource_error("bytearray result is too large"))?;
    runtime.reserve_memory(length)?;
    runtime.charge_cpu(u64::try_from(length).unwrap_or(u64::MAX))?;
    runtime.new_bytearray(value.repeat(count)).map(Some)
}

pub(crate) fn slot_bytearray_get_item(
    runtime: &mut dyn PyRuntime,
    owner: PyValue,
    index: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = owner.cast::<PyByteArray>(runtime)?;
    let Some(index) = runtime.int_value(&index) else {
        return Ok(None);
    };
    let items = runtime.bytearray_items(array)?;
    let index = byte_index(index, items.len())?;
    Ok(Some(PyValue::Int(i64::from(items[index]))))
}

pub(crate) fn slot_bytearray_set_item(
    runtime: &mut dyn PyRuntime,
    owner: PyValue,
    index: PyValue,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = owner.cast::<PyByteArray>(runtime)?;
    let Some(index) = runtime.int_value(&index) else {
        return Ok(None);
    };
    let value = runtime
        .int_value(&value)
        .and_then(|value| u8::try_from(value).ok())
        .ok_or_else(|| PyError::value_error("byte must be in range(0, 256)"))?;
    let mut items = runtime.bytearray_items(array)?;
    let index = byte_index(index, items.len())?;
    items[index] = value;
    runtime.replace_bytearray_items(array, items)?;
    Ok(Some(PyValue::None))
}

fn byte_index(index: i64, length: usize) -> PyResult<usize> {
    let length = i64::try_from(length).unwrap_or(i64::MAX);
    let index = if index < 0 {
        index.checked_add(length)
    } else {
        Some(index)
    }
    .and_then(|index| usize::try_from(index).ok())
    .filter(|index| *index < usize::try_from(length).unwrap_or(usize::MAX))
    .ok_or_else(|| PyError::exception("IndexError", "bytearray index out of range"))?;
    Ok(index)
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

pub(crate) fn slot_list_delete_item(
    runtime: &mut dyn PyRuntime,
    owner: PyValue,
    index: PyValue,
) -> PyResult<Option<PyValue>> {
    let list = owner.cast::<PyList>(runtime)?;
    let Some(index) = runtime.int_value(&index) else {
        return Ok(None);
    };
    let mut items = runtime.list_items(list)?;
    let length = i64::try_from(items.len()).unwrap_or(i64::MAX);
    let index = if index < 0 {
        index.checked_add(length)
    } else {
        Some(index)
    }
    .and_then(|index| usize::try_from(index).ok())
    .filter(|index| *index < items.len())
    .ok_or_else(|| PyError::exception("IndexError", "list assignment index out of range"))?;
    items.remove(index);
    runtime.replace_list_items(list, items)?;
    Ok(Some(PyValue::None))
}

pub(crate) fn slot_dict_delete_item(
    runtime: &mut dyn PyRuntime,
    owner: PyValue,
    key: PyValue,
) -> PyResult<Option<PyValue>> {
    let dict = owner.cast::<PyDict>(runtime)?;
    let mut items = runtime.dict_items(dict)?;
    let mut found = None;
    for (index, (candidate, _)) in items.iter().enumerate() {
        if runtime.equals(candidate, &key)? {
            found = Some(index);
            break;
        }
    }
    let index = found.ok_or_else(|| PyError::exception("KeyError", "key not found"))?;
    items.remove(index);
    runtime.replace_dict_items(dict, items)?;
    Ok(Some(PyValue::None))
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
