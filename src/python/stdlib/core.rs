//! Native descriptors for methods on builtin Python value types.
//!
//! Methods use checked, type-erased runtime views and snapshot-and-commit mutation. This keeps
//! collection layouts and compact scalar tags private to the runtime while giving builtin and
//! user-defined methods the same descriptor call path.

use std::cmp::Ordering;

use num_bigint::{BigInt, Sign};
use num_traits::{Signed, ToPrimitive, Zero};

use super::super::ast::ComparisonOperator;
use super::super::heap::DictViewKind;
use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, GetterDef, MethodDef, NativeFn, NativeMethodFn, NativeTypeDef,
    OwnedPyString, PyByteArray, PyBytes, PyCallable, PyDict, PyError, PyIterator, PyKind, PyList,
    PyProperty, PyResult, PyRuntime, PySequence, PySet, PyTuple, PyValue, PyValueCast,
    TypeMetadata,
};
use super::super::number::{index_argument, PyNumber};
use super::super::object_model::BuiltinType;
use super::super::protocol;
use super::super::slice::SlicePlan;
use super::super::unicode;

static BUILTINS: &[FunctionDef] = &[
    builtin("__import__", builtin_import),
    builtin("ascii", builtin_ascii),
    builtin("id", builtin_id),
    builtin("map", builtin_map),
    builtin("filter", builtin_filter),
    builtin("reversed", builtin_reversed),
    builtin("getattr", builtin_getattr),
    builtin("hasattr", builtin_hasattr),
    builtin("round", builtin_round),
];

/// Resolve capability-free builtins implemented through the erased runtime API.
pub(crate) fn builtin_function(name: &str) -> Option<&'static FunctionDef> {
    BUILTINS.iter().find(|function| function.name == name)
}

const fn builtin(name: &'static str, call: NativeFn) -> FunctionDef {
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
        method("str", "find", string_find),
        method("str", "rfind", string_rfind),
        method("str", "index", string_index),
        method("str", "rindex", string_rindex),
        method("str", "count", string_count),
        method("str", "partition", string_partition),
        method("str", "rpartition", string_rpartition),
        method("str", "split", string_split),
        method("str", "rsplit", string_rsplit),
        method("str", "splitlines", string_splitlines),
        method("str", "join", string_join),
        method("str", "replace", string_replace),
        method("str", "format", string_format),
        method("str", "format_map", string_format_map),
        method("str", "ljust", string_ljust),
        method("str", "rjust", string_rjust),
        method("str", "center", string_center),
        method("str", "encode", string_encode),
        method("str", "lower", string_lower),
        method("str", "upper", string_upper),
        method("str", "title", string_title),
        method("str", "capitalize", string_capitalize),
        method("str", "swapcase", string_swapcase),
        method("str", "zfill", string_zfill),
        method("str", "expandtabs", string_expandtabs),
        method("str", "translate", string_translate),
        method("str", "removeprefix", string_removeprefix),
        method("str", "removesuffix", string_removesuffix),
        method("str", "isalnum", string_isalnum),
        method("str", "isalpha", string_isalpha),
        method("str", "isdigit", string_isdigit),
        method("str", "isdecimal", string_isdecimal),
        method("str", "isnumeric", string_isnumeric),
        method("str", "isspace", string_isspace),
        method("str", "isascii", string_isascii),
        method("str", "islower", string_islower),
        method("str", "isupper", string_isupper),
        method("str", "istitle", string_istitle),
        method("str", "isidentifier", string_isidentifier),
    ],
    getters: &[],
};

pub(crate) static BYTES_TYPE: NativeTypeDef = NativeTypeDef {
    name: "bytes",
    methods: &[
        method("bytes", "decode", bytes_decode),
        method("bytes", "hex", bytes_hex),
        method("bytes", "startswith", bytes_startswith),
        method("bytes", "endswith", bytes_endswith),
        method("bytes", "find", bytes_find),
        method("bytes", "join", bytes_join),
        method("bytes", "index", bytes_index),
        method("bytes", "count", bytes_count),
        method("bytes", "partition", bytes_partition),
        method("bytes", "rpartition", bytes_rpartition),
        method("bytes", "center", bytes_center),
        method("bytes", "strip", bytes_strip),
        method("bytes", "lstrip", bytes_lstrip),
        method("bytes", "rstrip", bytes_rstrip),
        method("bytes", "split", bytes_split),
        method("bytes", "splitlines", bytes_splitlines),
        method("bytes", "upper", bytes_upper),
        method("bytes", "lower", bytes_lower),
        method("bytes", "replace", bytes_replace),
    ],
    getters: &[],
};

pub(crate) static BYTEARRAY_TYPE: NativeTypeDef = NativeTypeDef {
    name: "bytearray",
    methods: &[
        method("bytearray", "__init__", bytearray_init),
        method("bytearray", "append", bytearray_append),
        method("bytearray", "extend", bytearray_extend),
        method("bytearray", "insert", bytearray_insert),
        method("bytearray", "pop", bytearray_pop),
        method("bytearray", "remove", bytearray_remove),
        method("bytearray", "clear", bytearray_clear),
        method("bytearray", "copy", bytearray_copy),
        method("bytearray", "decode", bytes_decode),
        method("bytearray", "hex", bytes_hex),
        method("bytearray", "startswith", bytes_startswith),
        method("bytearray", "endswith", bytes_endswith),
        method("bytearray", "find", bytes_find),
        method("bytearray", "join", bytes_join),
        method("bytearray", "index", bytes_index),
        method("bytearray", "count", bytes_count),
        method("bytearray", "partition", bytes_partition),
        method("bytearray", "rpartition", bytes_rpartition),
        method("bytearray", "center", bytes_center),
        method("bytearray", "strip", bytes_strip),
        method("bytearray", "lstrip", bytes_lstrip),
        method("bytearray", "rstrip", bytes_rstrip),
        method("bytearray", "split", bytes_split),
        method("bytearray", "splitlines", bytes_splitlines),
        method("bytearray", "upper", bytes_upper),
        method("bytearray", "lower", bytes_lower),
        method("bytearray", "replace", bytes_replace),
        method("bytearray", "reverse", bytearray_reverse),
    ],
    getters: &[],
};

pub(crate) static SLICE_TYPE: NativeTypeDef = NativeTypeDef {
    name: "slice",
    methods: &[method("slice", "indices", slice_indices)],
    getters: &[],
};

pub(crate) static LIST_TYPE: NativeTypeDef = NativeTypeDef {
    name: "list",
    methods: &[
        method("list", "__init__", list_init),
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
    getters: &[],
};

pub(crate) static TUPLE_TYPE: NativeTypeDef = NativeTypeDef {
    name: "tuple",
    methods: &[
        method("tuple", "__new__", tuple_new),
        method("tuple", "__repr__", tuple_repr),
        method("tuple", "count", tuple_count),
        method("tuple", "index", tuple_index),
    ],
    getters: &[],
};

pub(crate) static DICT_TYPE: NativeTypeDef = NativeTypeDef {
    name: "dict",
    methods: &[
        method("dict", "__init__", dict_init),
        method("dict", "__getitem__", dict_getitem),
        method("dict", "__setitem__", dict_setitem),
        method("dict", "__delitem__", dict_delitem),
        method("dict", "__contains__", dict_contains),
        method("dict", "__iter__", dict_iter),
        method("dict", "__repr__", dict_repr),
        method("dict", "get", dict_get),
        method("dict", "keys", dict_keys),
        method("dict", "values", dict_values),
        method("dict", "items", dict_items),
        method("dict", "setdefault", dict_setdefault),
        method("dict", "update", dict_update),
        method("dict", "pop", dict_pop),
        method("dict", "popitem", dict_popitem),
        method("dict", "clear", dict_clear),
        method("dict", "copy", dict_copy),
    ],
    getters: &[],
};

/// `dict` methods bound to the type, so `dict.fromkeys(...)` and `{}.fromkeys(...)` agree.
pub(crate) static DICT_CLASS_METHODS: &[MethodDef] = &[method("dict", "fromkeys", dict_fromkeys)];
/// `bytes.fromhex` and `bytearray.fromhex`, bound to their types.
pub(crate) static BYTES_CLASS_METHODS: &[MethodDef] = &[method("bytes", "fromhex", bytes_fromhex)];
pub(crate) static BYTEARRAY_CLASS_METHODS: &[MethodDef] =
    &[method("bytearray", "fromhex", bytearray_fromhex)];

pub(crate) static SET_TYPE: NativeTypeDef = NativeTypeDef {
    name: "set",
    methods: &[
        method("set", "__init__", set_init),
        method("set", "add", set_add),
        method("set", "update", set_update),
        method("set", "remove", set_remove),
        method("set", "discard", set_discard),
        method("set", "pop", set_pop),
        method("set", "clear", set_clear),
        method("set", "union", set_union),
        method("set", "intersection", set_intersection),
        method("set", "difference", set_difference),
        method("set", "symmetric_difference", set_symmetric_difference),
        method("set", "intersection_update", set_intersection_update),
        method("set", "difference_update", set_difference_update),
        method(
            "set",
            "symmetric_difference_update",
            set_symmetric_difference_update,
        ),
        method("set", "issubset", set_issubset),
        method("set", "issuperset", set_issuperset),
        method("set", "isdisjoint", set_isdisjoint),
        method("set", "copy", set_copy),
    ],
    getters: &[],
};

pub(crate) static FROZENSET_TYPE: NativeTypeDef = NativeTypeDef {
    name: "frozenset",
    methods: &[
        method("frozenset", "union", set_union),
        method("frozenset", "intersection", set_intersection),
        method("frozenset", "difference", set_difference),
        method(
            "frozenset",
            "symmetric_difference",
            set_symmetric_difference,
        ),
        method("frozenset", "issubset", set_issubset),
        method("frozenset", "issuperset", set_issuperset),
        method("frozenset", "isdisjoint", set_isdisjoint),
        method("frozenset", "copy", set_copy),
    ],
    getters: &[],
};

pub(crate) static PROPERTY_TYPE: NativeTypeDef = NativeTypeDef {
    name: "property",
    methods: &[method("property", "setter", property_setter)],
    getters: &[
        GetterDef {
            owner: "property",
            name: "fget",
            get: property_fget,
        },
        GetterDef {
            owner: "property",
            name: "fset",
            get: property_fset,
        },
        GetterDef {
            owner: "property",
            name: "__isabstractmethod__",
            get: property_is_abstract,
        },
    ],
};

/// `property.fget`: the getter function.
fn property_fget(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    runtime.property_getter(receiver.cast::<PyProperty>(runtime)?)
}

/// `property.fset`: the setter function, or `None` for a read-only property.
fn property_fset(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    Ok(runtime
        .property_setter(receiver.cast::<PyProperty>(runtime)?)?
        .unwrap_or(Value::None))
}

/// `property.__isabstractmethod__`: whether the getter or setter is marked abstract, so
/// `abc.ABCMeta` sees an `@property` over an `@abstractmethod`.
fn property_is_abstract(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    let property = receiver.cast::<PyProperty>(runtime)?;
    let mut functions = vec![runtime.property_getter(property)?];
    functions.extend(runtime.property_setter(property)?);
    for function in functions {
        if let Some(flag) = runtime.get_attribute(function, "__isabstractmethod__")? {
            if runtime.truth(&flag)? {
                return Ok(Value::Bool(true));
            }
        }
    }
    Ok(Value::Bool(false))
}

pub(crate) static OBJECT_TYPE: NativeTypeDef = NativeTypeDef {
    name: "object",
    methods: &[
        method("object", "__new__", object_new),
        method("object", "__init__", object_init),
        method("object", "__init_subclass__", object_init_subclass),
        method("object", "__repr__", object_repr),
        method("object", "__str__", object_str),
        method("object", "__hash__", object_hash),
        method("object", "__getattribute__", object_getattribute),
        method("object", "__eq__", object_eq),
        method("object", "__ne__", object_ne),
        method("object", "__setattr__", object_setattr),
        method("object", "__delattr__", object_delattr),
    ],
    getters: &[GetterDef {
        owner: "object",
        name: "__class__",
        get: object_class,
    }],
};

fn object_class(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    runtime.class_of(&receiver)
}

/// The read side of the class, module and heap-instance `__dict__` descriptors.
fn object_dict(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    runtime
        .dictionary_of(receiver)?
        .ok_or_else(|| PyError::exception("AttributeError", "this object has no __dict__"))
}

fn type_name(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    runtime
        .type_metadata(receiver, TypeMetadata::Name)?
        .ok_or_else(|| PyError::exception("AttributeError", "type has no __name__"))
}

fn type_module(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    runtime
        .type_metadata(receiver, TypeMetadata::Module)?
        .ok_or_else(|| PyError::exception("AttributeError", "type has no __module__"))
}

fn type_bases(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    runtime
        .type_metadata(receiver, TypeMetadata::Bases)?
        .ok_or_else(|| PyError::exception("AttributeError", "type has no __bases__"))
}

fn type_mro_getter(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    runtime
        .type_metadata(receiver, TypeMetadata::Mro)?
        .ok_or_else(|| PyError::exception("AttributeError", "type has no __mro__"))
}

pub(crate) static INSTANCE_DICT_GETTER: GetterDef = GetterDef {
    owner: "object",
    name: "__dict__",
    get: object_dict,
};

pub(crate) static MODULE_TYPE: NativeTypeDef = NativeTypeDef {
    name: "module",
    methods: &[],
    getters: &[GetterDef {
        owner: "module",
        name: "__dict__",
        get: object_dict,
    }],
};

pub(crate) static ITERATOR_TYPE: NativeTypeDef = NativeTypeDef {
    name: "iterator",
    methods: &[
        method("iterator", "__iter__", iterator_iter),
        method("iterator", "__next__", iterator_next),
    ],
    getters: &[],
};

pub(crate) static EXCEPTION_TYPE: NativeTypeDef = NativeTypeDef {
    name: "BaseException",
    methods: &[method("BaseException", "__init__", exception_init)],
    getters: &[
        GetterDef {
            owner: "BaseException",
            name: "args",
            get: exception_args,
        },
        GetterDef {
            owner: "BaseException",
            name: "value",
            get: stop_iteration_value,
        },
    ],
};

/// Attributes `OSError` derives from its constructor arguments: `errno`, `strerror`,
/// `filename` and `filename2`, each `None` unless the exception was built with an errno and a
/// message, as in CPython.
pub(crate) static OS_ERROR_TYPE: NativeTypeDef = NativeTypeDef {
    name: "OSError",
    methods: &[],
    getters: &[
        GetterDef {
            owner: "OSError",
            name: "errno",
            get: os_error_errno,
        },
        GetterDef {
            owner: "OSError",
            name: "strerror",
            get: os_error_strerror,
        },
        GetterDef {
            owner: "OSError",
            name: "filename",
            get: os_error_filename,
        },
        GetterDef {
            owner: "OSError",
            name: "filename2",
            get: os_error_filename2,
        },
    ],
};

/// The errno-form arguments `(errno, strerror[, filename[, winerror[, filename2]]])` of an
/// `OSError`, or `None` when it was built with a plain message.
fn os_error_parts(
    runtime: &mut dyn PyRuntime,
    receiver: &PyValue,
) -> PyResult<Option<Vec<PyValue>>> {
    let Some((kind, args)) = runtime.exception_args(receiver)? else {
        return Ok(None);
    };
    let errno_form = (2..=5).contains(&args.len())
        && super::super::exception_types::exception_is_subclass(&kind, "OSError");
    Ok(errno_form.then_some(args))
}

fn os_error_field(runtime: &mut dyn PyRuntime, receiver: PyValue, index: usize) -> PyResult {
    Ok(os_error_parts(runtime, &receiver)?
        .and_then(|args| args.get(index).copied())
        .unwrap_or(Value::None))
}

fn os_error_errno(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    os_error_field(runtime, receiver, 0)
}

fn os_error_strerror(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    os_error_field(runtime, receiver, 1)
}

fn os_error_filename(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    os_error_field(runtime, receiver, 2)
}

fn os_error_filename2(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    os_error_field(runtime, receiver, 4)
}

/// `exception.args`: the constructor arguments of a builtin exception. An `OSError` built with a
/// filename keeps only `(errno, strerror)` in `args`, as CPython does.
fn exception_args(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    if let Some(args) = os_error_parts(runtime, &receiver)? {
        if args.len() > 2 {
            return runtime.new_tuple(args[..2].to_vec());
        }
    }
    let (_, args) = runtime
        .exception_args(&receiver)?
        .ok_or_else(|| PyError::type_error("descriptor 'args' requires an exception"))?;
    runtime.new_tuple(args)
}

/// `StopIteration.value`: the first argument, which is a generator's return value, or `None`.
fn stop_iteration_value(runtime: &mut dyn PyRuntime, receiver: PyValue) -> PyResult {
    match runtime.exception_args(&receiver)? {
        Some((kind, args))
            if super::super::exception_types::exception_is_subclass(&kind, "StopIteration") =>
        {
            Ok(args.first().copied().unwrap_or(Value::None))
        }
        Some(_) => Err(PyError::exception(
            "AttributeError",
            format!(
                "'{}' object has no attribute 'value'",
                runtime.type_name(&receiver)?
            ),
        )),
        None => Err(PyError::type_error(
            "descriptor 'value' requires an exception",
        )),
    }
}

pub(crate) static TYPE_TYPE: NativeTypeDef = NativeTypeDef {
    name: "type",
    methods: &[
        method("type", "__new__", type_new),
        method("type", "__call__", type_call),
        method("type", "mro", type_mro),
    ],
    getters: &[
        GetterDef {
            owner: "type",
            name: "__dict__",
            get: object_dict,
        },
        GetterDef {
            owner: "type",
            name: "__name__",
            get: type_name,
        },
        GetterDef {
            owner: "type",
            name: "__qualname__",
            get: type_name,
        },
        GetterDef {
            owner: "type",
            name: "__module__",
            get: type_module,
        },
        GetterDef {
            owner: "type",
            name: "__bases__",
            get: type_bases,
        },
        GetterDef {
            owner: "type",
            name: "__mro__",
            get: type_mro_getter,
        },
    ],
};

pub(crate) static GENERATOR_TYPE: NativeTypeDef = NativeTypeDef {
    name: "generator",
    methods: &[
        method("generator", "__iter__", iterator_iter),
        method("generator", "__next__", generator_next),
        method("generator", "send", generator_send),
        method("generator", "throw", generator_throw),
        method("generator", "close", generator_close),
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

fn generator_next(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("generator.__next__", 0, 0)?;
    args.reject_keywords("generator.__next__")?;
    let generator = receiver.cast::<PyIterator>(runtime)?;
    match runtime.generator_send(generator, Value::None)? {
        Some(value) => Ok(value),
        None => Err(runtime.generator_stop(generator)),
    }
}

fn generator_send(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("generator.send", 1, 1)?;
    args.reject_keywords("generator.send")?;
    let generator = receiver.cast::<PyIterator>(runtime)?;
    match runtime.generator_send(generator, args.positional()[0])? {
        Some(value) => Ok(value),
        None => Err(runtime.generator_stop(generator)),
    }
}

fn generator_close(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("generator.close", 0, 0)?;
    args.reject_keywords("generator.close")?;
    let generator = receiver.cast::<PyIterator>(runtime)?;
    runtime.generator_close(generator)?;
    Ok(Value::None)
}

fn generator_throw(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("generator.throw", 1, 1)?;
    args.reject_keywords("generator.throw")?;
    let generator = receiver.cast::<PyIterator>(runtime)?;
    runtime.generator_throw(generator, args.positional()[0])
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
    args.reject_unknown_keywords("str.encode", &["encoding", "errors"])?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let (encoding, errors) = codec_args(runtime, &args, "str.encode")?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    // The widest code point the encoding can represent; characters above it go through the
    // error handler.
    let limit = match encoding.as_str() {
        "utf-8" | "utf8" => {
            return runtime.new_bytes(value.into_bytes());
        }
        "ascii" | "us-ascii" => 0x7f,
        "latin-1" | "latin1" | "iso-8859-1" | "iso8859-1" => 0xff,
        _ => return Err(PyError::value_error("unknown text encoding")),
    };
    let mut encoded = Vec::with_capacity(value.len());
    for character in value.chars() {
        let code = u32::from(character);
        if code <= limit {
            encoded.push(code as u8);
            continue;
        }
        match errors {
            CodecErrors::Strict => {
                return Err(PyError::exception(
                    "UnicodeEncodeError",
                    format!("'{encoding}' codec can't encode character {character:?}"),
                ))
            }
            CodecErrors::Ignore => {}
            CodecErrors::Replace => encoded.push(b'?'),
            CodecErrors::BackslashReplace => {
                encoded.extend_from_slice(backslash_escape(character).as_bytes())
            }
            CodecErrors::XmlCharRefReplace => {
                encoded.extend_from_slice(format!("&#{code};").as_bytes())
            }
        }
    }
    runtime.new_bytes(encoded)
}

/// How a codec treats characters or bytes it cannot convert.
#[derive(Clone, Copy)]
enum CodecErrors {
    Strict,
    Ignore,
    Replace,
    /// `\xNN`, `\uNNNN` or `\UNNNNNNNN` escapes for what the codec cannot represent.
    BackslashReplace,
    /// `&#NNN;` references; encoding only.
    XmlCharRefReplace,
}

/// The escape `backslashreplace` substitutes for one unencodable character.
fn backslash_escape(character: char) -> String {
    let code = u32::from(character);
    if code <= 0xff {
        format!("\\x{code:02x}")
    } else if code <= 0xffff {
        format!("\\u{code:04x}")
    } else {
        format!("\\U{code:08x}")
    }
}

/// The `encoding` and `errors` arguments of `str.encode` and `bytes.decode`, positional or by
/// keyword, with the encoding name normalized to lowercase hyphenated form.
fn codec_args(
    runtime: &dyn PyRuntime,
    args: &CallArgs,
    name: &str,
) -> PyResult<(String, CodecErrors)> {
    let encoding = args
        .positional()
        .first()
        .or(args.keyword(name, "encoding")?)
        .map(|value| value.cast::<OwnedPyString>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or_else(|| "utf-8".into())
        .to_ascii_lowercase()
        .replace('_', "-");
    let errors = args
        .positional()
        .get(1)
        .or(args.keyword(name, "errors")?)
        .map(|value| value.cast::<OwnedPyString>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or_else(|| "strict".into());
    let errors = match errors.as_str() {
        "strict" => CodecErrors::Strict,
        "ignore" => CodecErrors::Ignore,
        "replace" => CodecErrors::Replace,
        "backslashreplace" => CodecErrors::BackslashReplace,
        "xmlcharrefreplace" if name == "str.encode" => CodecErrors::XmlCharRefReplace,
        _ => {
            return Err(PyError::exception(
                "LookupError",
                format!("unknown error handler name '{errors}'"),
            ))
        }
    };
    Ok((encoding, errors))
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

fn string_title(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_transform(runtime, receiver, args, unicode::title, "str.title")
}

fn string_capitalize(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_transform(
        runtime,
        receiver,
        args,
        unicode::capitalize,
        "str.capitalize",
    )
}

fn string_swapcase(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_transform(runtime, receiver, args, unicode::swapcase, "str.swapcase")
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
    let OwnedPyString(value) = receiver.cast(runtime)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    runtime.new_string(transform(&value))
}

fn string_isalnum(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_predicate(
        runtime,
        receiver,
        args,
        unicode::is_alphanumeric,
        "str.isalnum",
    )
}

fn string_isalpha(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_predicate(runtime, receiver, args, char::is_alphabetic, "str.isalpha")
}

fn string_isdigit(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_predicate(runtime, receiver, args, unicode::is_digit, "str.isdigit")
}

fn string_isdecimal(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_predicate(
        runtime,
        receiver,
        args,
        unicode::is_decimal,
        "str.isdecimal",
    )
}

fn string_isnumeric(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_predicate(
        runtime,
        receiver,
        args,
        unicode::is_numeric,
        "str.isnumeric",
    )
}

fn string_isspace(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_predicate(runtime, receiver, args, unicode::is_space, "str.isspace")
}

fn string_isascii(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.isascii", 0, 0)?;
    args.reject_keywords("str.isascii")?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    Ok(PyValue::Bool(value.is_ascii()))
}

fn string_istitle(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.istitle", 0, 0)?;
    args.reject_keywords("str.istitle")?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    Ok(PyValue::Bool(unicode::is_title(&value)))
}

fn string_isidentifier(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.isidentifier", 0, 0)?;
    args.reject_keywords("str.isidentifier")?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    Ok(PyValue::Bool(unicode::is_identifier(&value)))
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
    let OwnedPyString(value) = receiver.cast(runtime)?;
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
    let OwnedPyString(value) = receiver.cast(runtime)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    Ok(PyValue::Bool(
        !value.is_empty() && value.chars().all(predicate),
    ))
}

fn bytes_decode(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytes.decode", 0, 2)?;
    args.reject_unknown_keywords("bytes.decode", &["encoding", "errors"])?;
    let (encoding, errors) = codec_args(runtime, &args, "bytes.decode")?;
    let PyBytes(value) = receiver.cast(runtime)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let decoded = match encoding.as_str() {
        "utf-8" | "utf8" => match errors {
            CodecErrors::Strict => String::from_utf8(value).map_err(|_| {
                PyError::exception("UnicodeDecodeError", "invalid UTF-8 byte sequence")
            })?,
            CodecErrors::Replace => String::from_utf8_lossy(&value).into_owned(),
            CodecErrors::Ignore
            | CodecErrors::BackslashReplace
            | CodecErrors::XmlCharRefReplace => {
                let mut decoded = String::with_capacity(value.len());
                for chunk in value.utf8_chunks() {
                    decoded.push_str(chunk.valid());
                    if matches!(errors, CodecErrors::BackslashReplace) {
                        for byte in chunk.invalid() {
                            decoded.push_str(&format!("\\x{byte:02x}"));
                        }
                    }
                }
                decoded
            }
        },
        "ascii" | "us-ascii" => {
            let mut decoded = String::with_capacity(value.len());
            for byte in value {
                if byte.is_ascii() {
                    decoded.push(char::from(byte));
                    continue;
                }
                match errors {
                    CodecErrors::Strict => {
                        return Err(PyError::exception(
                            "UnicodeDecodeError",
                            format!("'ascii' codec can't decode byte {byte:#04x}"),
                        ))
                    }
                    CodecErrors::Ignore | CodecErrors::XmlCharRefReplace => {}
                    CodecErrors::Replace => decoded.push('\u{fffd}'),
                    CodecErrors::BackslashReplace => decoded.push_str(&format!("\\x{byte:02x}")),
                }
            }
            decoded
        }
        "latin-1" | "latin1" | "iso-8859-1" | "iso8859-1" => {
            value.into_iter().map(char::from).collect()
        }
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

/// `bytes.startswith` and `bytes.endswith`: test one affix or each affix of a tuple against
/// the bytes in `[start, end)`, which follow slice clamping.
fn bytes_affix(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    prefix: bool,
) -> PyResult {
    let name = if prefix { "startswith" } else { "endswith" };
    args.expect_positional(name, 1, 3)?;
    args.reject_keywords(name)?;
    let PyBytes(value) = receiver.cast(runtime)?;
    let affix = args.positional()[0];
    let affixes = if runtime.kind(&affix)? == PyKind::Tuple {
        affix.cast::<PyTuple>(runtime)?.items(runtime)?
    } else if runtime.bytes_value(&affix)?.is_some() {
        vec![affix]
    } else {
        let actual = runtime.type_name(&affix)?;
        return Err(PyError::type_error(format!(
            "{name} first arg must be bytes or a tuple of bytes, not {actual}"
        )));
    };
    let (start, end) = string_bounds(runtime, args.positional(), value.len())?;
    let window = value.get(start..end).unwrap_or_default();
    for affix in affixes {
        let Some(affix) = runtime.bytes_value(&affix)? else {
            let actual = runtime.type_name(&affix)?;
            return Err(PyError::type_error(format!(
                "a bytes-like object is required, not '{actual}'"
            )));
        };
        runtime.charge_cpu(u64::try_from(affix.len()).unwrap_or(u64::MAX))?;
        let matched = start <= end
            && if prefix {
                window.starts_with(&affix)
            } else {
                window.ends_with(&affix)
            };
        if matched {
            return Ok(Value::Bool(true));
        }
    }
    Ok(Value::Bool(false))
}

/// `bytes.join(iterable)` and `bytearray.join`: the bytes-like items with the receiver between
/// them, as an object of the receiver's type.
fn bytes_join(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytes.join", 1, 1)?;
    args.reject_keywords("bytes.join")?;
    let PyBytes(separator) = receiver.cast(runtime)?;
    let iterator = runtime.iterator(args.positional()[0])?;
    let mut parts: Vec<Vec<u8>> = Vec::new();
    let mut length = 0usize;
    let mut exhausted = false;
    while !exhausted {
        let mut part = None;
        runtime.nested(&mut |runtime| {
            let Some(value) = runtime.iterator_next(iterator)? else {
                exhausted = true;
                return Ok(());
            };
            runtime.charge_cpu(1)?;
            let Some(bytes) = runtime.bytes_value(&value)? else {
                return Err(PyError::type_error(format!(
                    "sequence item {}: expected a bytes-like object, {} found",
                    parts.len(),
                    runtime.type_name(&value)?
                )));
            };
            part = Some(bytes);
            Ok(())
        })?;
        let Some(part) = part else { continue };
        length = length
            .checked_add(part.len())
            .ok_or_else(|| PyError::resource_error("joined bytes are too large"))?;
        // Each part is a host copy that lives until the join finishes. It is reserved here, in
        // the scope that keeps it, because scratch reserved inside the child scope is released
        // when that scope closes.
        runtime.reserve_memory(part.len().saturating_add(JOINED_PART_BYTES))?;
        parts.push(part);
    }
    length = length
        .checked_add(
            separator
                .len()
                .saturating_mul(parts.len().saturating_sub(1)),
        )
        .ok_or_else(|| PyError::resource_error("joined bytes are too large"))?;
    runtime.reserve_memory(length)?;
    runtime.charge_cpu(u64::try_from(length).unwrap_or(u64::MAX))?;
    let joined = parts.join(separator.as_slice());
    if runtime.kind(&receiver)? == PyKind::ByteArray {
        runtime.new_bytearray(joined)
    } else {
        runtime.new_bytes(joined)
    }
}

fn bytes_find(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let found = bytes_search(runtime, receiver, &args, "bytes.find")?;
    Ok(PyValue::Int(
        found
            .and_then(|value| i64::try_from(value).ok())
            .unwrap_or(-1),
    ))
}

fn bytes_index(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let found = bytes_search(runtime, receiver, &args, "bytes.index")?
        .ok_or_else(|| PyError::value_error("subsection not found"))?;
    i64::try_from(found)
        .map(PyValue::Int)
        .map_err(|_| PyError::overflow_error("byte index is too large"))
}

/// Position of the first occurrence of `sub` within the optional `[start:end]` window.
fn bytes_search(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: &CallArgs,
    name: &str,
) -> PyResult<Option<usize>> {
    args.expect_positional(name, 1, 3)?;
    args.reject_keywords(name)?;
    let PyBytes(value) = receiver.cast(runtime)?;
    let PyBytes(needle) = args.positional()[0].cast(runtime)?;
    let (start, end) = string_bounds(runtime, args.positional(), value.len())?;
    // Two-way search is linear in both lengths.
    let work = end.saturating_sub(start).saturating_add(needle.len());
    runtime.charge_cpu(u64::try_from(work).unwrap_or(u64::MAX))?;
    Ok(if start <= end && needle.len() <= end - start {
        memchr::memmem::find(&value[start..end], &needle).map(|position| start + position)
    } else {
        None
    })
}

fn bytes_count(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytes.count", 1, 3)?;
    args.reject_keywords("bytes.count")?;
    let PyBytes(value) = receiver.cast(runtime)?;
    let PyBytes(needle) = args.positional()[0].cast(runtime)?;
    let (start, end) = string_bounds(runtime, args.positional(), value.len())?;
    let mut count = 0usize;
    let mut index = start;
    if needle.is_empty() {
        count = end
            .saturating_sub(start)
            .saturating_add(usize::from(start <= end));
    } else {
        while index.saturating_add(needle.len()) <= end {
            runtime.charge_cpu(1)?;
            if value[index..].starts_with(&needle) {
                count = count.saturating_add(1);
                index = index.saturating_add(needle.len());
            } else {
                index = index.saturating_add(1);
            }
        }
    }
    i64::try_from(count)
        .map(PyValue::Int)
        .map_err(|_| PyError::overflow_error("byte count is too large"))
}

fn bytes_partition(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    bytes_partition_impl(runtime, receiver, args, false)
}

fn bytes_rpartition(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    bytes_partition_impl(runtime, receiver, args, true)
}

fn bytes_partition_impl(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    reverse: bool,
) -> PyResult {
    args.expect_positional("bytes.partition", 1, 1)?;
    args.reject_keywords("bytes.partition")?;
    let kind = runtime.kind(&receiver)?;
    let PyBytes(value) = receiver.cast(runtime)?;
    let PyBytes(separator) = args.positional()[0].cast(runtime)?;
    if separator.is_empty() {
        return Err(PyError::value_error("empty separator"));
    }
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let position = if reverse {
        value
            .windows(separator.len())
            .rposition(|window| window == separator)
    } else {
        value
            .windows(separator.len())
            .position(|window| window == separator)
    };
    let (left, middle, right) = match position {
        Some(position) => (
            value[..position].to_vec(),
            separator.clone(),
            value[position + separator.len()..].to_vec(),
        ),
        None if reverse => (Vec::new(), Vec::new(), value),
        None => (value, Vec::new(), Vec::new()),
    };
    let left = new_bytes_like(runtime, kind, left)?;
    let middle = new_bytes_like(runtime, kind, middle)?;
    let right = new_bytes_like(runtime, kind, right)?;
    runtime.new_tuple(vec![left, middle, right])
}

fn bytes_center(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytes.center", 1, 2)?;
    args.reject_keywords("bytes.center")?;
    let kind = runtime.kind(&receiver)?;
    let PyBytes(value) = receiver.cast(runtime)?;
    let width = runtime
        .int_value(&args.positional()[0])
        .ok_or_else(|| PyError::type_error("width must be an integer"))?;
    let fill = if let Some(fill) = args.positional().get(1) {
        let PyBytes(fill) = (*fill).cast(runtime)?;
        if fill.len() != 1 {
            return Err(PyError::type_error(
                "center() argument 2 must be a byte string of length 1",
            ));
        }
        fill[0]
    } else {
        b' '
    };
    let padding = usize::try_from(width)
        .ok()
        .unwrap_or_default()
        .saturating_sub(value.len());
    let capacity = value
        .len()
        .checked_add(padding)
        .ok_or_else(|| PyError::resource_error("centered bytes are too large"))?;
    runtime.reserve_memory(capacity)?;
    runtime.charge_cpu(u64::try_from(capacity).unwrap_or(u64::MAX))?;
    let left = center_left_padding(padding, width);
    let mut result = Vec::with_capacity(capacity);
    result.extend(std::iter::repeat_n(fill, left));
    result.extend(value);
    result.extend(std::iter::repeat_n(fill, padding - left));
    new_bytes_like(runtime, kind, result)
}

fn new_bytes_like(runtime: &mut dyn PyRuntime, kind: PyKind, value: Vec<u8>) -> PyResult {
    if kind == PyKind::ByteArray {
        runtime.new_bytearray(value)
    } else {
        runtime.new_bytes(value)
    }
}

/// ASCII whitespace as `bytes.isspace` defines it, which includes vertical tab.
fn is_bytes_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | b'\x0b' | b'\x0c')
}

fn bytes_strip(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    bytes_strip_impl(runtime, receiver, args, StripKind::Both, "bytes.strip")
}

fn bytes_lstrip(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    bytes_strip_impl(runtime, receiver, args, StripKind::Left, "bytes.lstrip")
}

fn bytes_rstrip(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    bytes_strip_impl(runtime, receiver, args, StripKind::Right, "bytes.rstrip")
}

/// Strip bytes found in the optional argument, or ASCII whitespace when it is absent or `None`.
fn bytes_strip_impl(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    strip: StripKind,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 0, 1)?;
    args.reject_keywords(name)?;
    let kind = runtime.kind(&receiver)?;
    let PyBytes(value) = receiver.cast(runtime)?;
    let characters = match args.positional().first() {
        None => None,
        Some(value) if runtime.kind(value)? == PyKind::None => None,
        Some(value) => Some((*value).cast::<PyBytes>(runtime)?.0),
    };
    let stripped = |byte: &u8| match &characters {
        Some(characters) => characters.contains(byte),
        None => is_bytes_whitespace(*byte),
    };
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let start = if matches!(strip, StripKind::Right) {
        0
    } else {
        value
            .iter()
            .position(|byte| !stripped(byte))
            .unwrap_or(value.len())
    };
    let end = if matches!(strip, StripKind::Left) {
        value.len()
    } else {
        value
            .iter()
            .rposition(|byte| !stripped(byte))
            .map_or(0, |position| position + 1)
    };
    let result = value[start..end.max(start)].to_vec();
    new_bytes_like(runtime, kind, result)
}

/// `split(sep=None, maxsplit=-1)` with positional arguments, as `str.split` accepts them.
fn bytes_split(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytes.split", 0, 2)?;
    args.reject_keywords("bytes.split")?;
    let kind = runtime.kind(&receiver)?;
    let PyBytes(value) = receiver.cast(runtime)?;
    let separator = match args.positional().first() {
        None => None,
        Some(value) if runtime.kind(value)? == PyKind::None => None,
        Some(value) => {
            let PyBytes(separator) = (*value).cast(runtime)?;
            if separator.is_empty() {
                return Err(PyError::value_error("empty separator"));
            }
            Some(separator)
        }
    };
    let maximum = args
        .positional()
        .get(1)
        .map(|value| index_argument(runtime, value))
        .transpose()?
        .and_then(|maximum| usize::try_from(maximum).ok());
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let mut parts = Vec::new();
    split_bytes(&value, separator.as_deref(), maximum, |part| {
        runtime.reserve_memory(std::mem::size_of::<PyValue>().saturating_add(part.len()))?;
        parts.push(new_bytes_like(runtime, kind, part.to_vec())?);
        Ok(())
    })?;
    runtime.new_list(parts)
}

/// `bytes.splitlines(keepends=False)`: split at `\n`, `\r` and `\r\n` only, unlike `str`.
fn bytes_splitlines(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytes.splitlines", 0, 1)?;
    args.reject_unknown_keywords("bytes.splitlines", &["keepends"])?;
    let keepends = match args
        .positional()
        .first()
        .or(args.keyword("bytes.splitlines", "keepends")?)
    {
        Some(value) => runtime.truth(value)?,
        None => false,
    };
    let kind = runtime.kind(&receiver)?;
    let PyBytes(value) = receiver.cast(runtime)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let mut parts = Vec::new();
    let mut start = 0;
    let mut index = 0;
    while index < value.len() {
        let byte = value[index];
        if byte != b'\n' && byte != b'\r' {
            index += 1;
            continue;
        }
        let mut end = index + 1;
        if byte == b'\r' && value.get(end) == Some(&b'\n') {
            end += 1;
        }
        let part = if keepends {
            &value[start..end]
        } else {
            &value[start..index]
        };
        runtime.reserve_memory(std::mem::size_of::<PyValue>().saturating_add(part.len()))?;
        parts.push(new_bytes_like(runtime, kind, part.to_vec())?);
        start = end;
        index = end;
    }
    if start < value.len() {
        let part = &value[start..];
        runtime.reserve_memory(std::mem::size_of::<PyValue>().saturating_add(part.len()))?;
        parts.push(new_bytes_like(runtime, kind, part.to_vec())?);
    }
    runtime.new_list(parts)
}

/// Pass each part of `value` to `part` in order, as CPython's `bytes.split` divides it, with at
/// most `maximum` splits when it is set.
///
/// Without a separator, runs of ASCII whitespace separate parts and never produce empty ones;
/// the unsplit remainder keeps its trailing whitespace. The callback lets the caller meter each
/// part before allocating it.
fn split_bytes(
    value: &[u8],
    separator: Option<&[u8]>,
    maximum: Option<usize>,
    mut part: impl FnMut(&[u8]) -> PyResult<()>,
) -> PyResult<()> {
    let mut splits = 0;
    let Some(separator) = separator else {
        let mut index = 0;
        loop {
            while index < value.len() && is_bytes_whitespace(value[index]) {
                index += 1;
            }
            if index == value.len() {
                return Ok(());
            }
            if maximum == Some(splits) {
                return part(&value[index..]);
            }
            let start = index;
            while index < value.len() && !is_bytes_whitespace(value[index]) {
                index += 1;
            }
            part(&value[start..index])?;
            splits += 1;
        }
    };
    let mut start = 0;
    while maximum != Some(splits) {
        let Some(offset) = value[start..]
            .windows(separator.len())
            .position(|window| window == separator)
        else {
            break;
        };
        part(&value[start..start + offset])?;
        start += offset + separator.len();
        splits += 1;
    }
    part(&value[start..])
}

fn bytes_upper(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    bytes_transform(
        runtime,
        receiver,
        args,
        "bytes.upper",
        <[u8]>::to_ascii_uppercase,
    )
}

fn bytes_lower(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    bytes_transform(
        runtime,
        receiver,
        args,
        "bytes.lower",
        <[u8]>::to_ascii_lowercase,
    )
}

/// Apply a length-preserving ASCII transform. Bytes outside ASCII are unchanged, as in CPython.
fn bytes_transform(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    name: &str,
    transform: fn(&[u8]) -> Vec<u8>,
) -> PyResult {
    args.expect_positional(name, 0, 0)?;
    args.reject_keywords(name)?;
    let kind = runtime.kind(&receiver)?;
    let PyBytes(value) = receiver.cast(runtime)?;
    runtime.reserve_memory(value.len())?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    new_bytes_like(runtime, kind, transform(&value))
}

/// `replace(old, new, count=-1)`, reserving the exact result size before building it.
fn bytes_replace(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytes.replace", 2, 3)?;
    args.reject_keywords("bytes.replace")?;
    let kind = runtime.kind(&receiver)?;
    let PyBytes(value) = receiver.cast(runtime)?;
    let PyBytes(old) = args.positional()[0].cast(runtime)?;
    let PyBytes(new) = args.positional()[1].cast(runtime)?;
    // A negative count, the default, replaces every occurrence.
    let limit = args
        .positional()
        .get(2)
        .map(|value| index_argument(runtime, value))
        .transpose()?
        .and_then(|count| usize::try_from(count).ok())
        .unwrap_or(usize::MAX);
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let mut count = 0usize;
    for_each_replacement(&value, &old, limit, |_| count += 1);
    // Matches never overlap, so they cover at most `value.len()` bytes.
    let length = count
        .checked_mul(new.len())
        .and_then(|inserted| inserted.checked_add(value.len() - count * old.len()))
        .ok_or_else(|| PyError::resource_error("replaced bytes are too large"))?;
    runtime.reserve_memory(length)?;
    runtime.charge_cpu(u64::try_from(length).unwrap_or(u64::MAX))?;
    let mut result = Vec::with_capacity(length);
    let mut copied = 0;
    for_each_replacement(&value, &old, limit, |position| {
        result.extend_from_slice(&value[copied..position]);
        result.extend_from_slice(&new);
        copied = position + old.len();
    });
    result.extend_from_slice(&value[copied..]);
    new_bytes_like(runtime, kind, result)
}

/// Visit the start offsets of the first `limit` non-overlapping occurrences of `old`, left to
/// right. An empty `old` matches before every byte and at the end, as in CPython.
fn for_each_replacement(value: &[u8], old: &[u8], limit: usize, mut visit: impl FnMut(usize)) {
    let mut found = 0;
    let mut index = 0;
    while found < limit && index <= value.len() {
        if value[index..].starts_with(old) {
            visit(index);
            found += 1;
            // An empty match must still advance to the next byte.
            index += old.len().max(1);
        } else {
            index += 1;
        }
    }
}

fn bytearray_append(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytearray.append", 1, 1)?;
    args.reject_keywords("bytearray.append")?;
    let array = receiver.cast::<PyByteArray>(runtime)?;
    let byte = byte_argument(runtime, &args.positional()[0])?;
    let mut items = runtime.bytearray_items(array)?;
    items.push(byte);
    runtime.replace_bytearray_items(array, items)?;
    Ok(PyValue::None)
}

/// Convert an integer argument to one byte, raising CPython's errors for other values.
fn byte_argument(runtime: &dyn PyRuntime, value: &PyValue) -> PyResult<u8> {
    let out_of_range = || PyError::value_error("byte must be in range(0, 256)");
    // An integer too large for an index is still just out of the byte range.
    if runtime.kind(value)? == PyKind::Int && runtime.int_value(value).is_none() {
        return Err(out_of_range());
    }
    u8::try_from(index_argument(runtime, value)?).map_err(|_| out_of_range())
}

fn bytearray_insert(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytearray.insert", 2, 2)?;
    args.reject_keywords("bytearray.insert")?;
    let array = receiver.cast::<PyByteArray>(runtime)?;
    let raw = index_argument(runtime, &args.positional()[0])?;
    let byte = byte_argument(runtime, &args.positional()[1])?;
    let mut items = runtime.bytearray_items(array)?;
    runtime.charge_cpu(u64::try_from(items.len()).unwrap_or(u64::MAX))?;
    items.insert(insert_index(raw, items.len()), byte);
    runtime.replace_bytearray_items(array, items)?;
    Ok(PyValue::None)
}

fn bytearray_pop(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytearray.pop", 0, 1)?;
    args.reject_keywords("bytearray.pop")?;
    let array = receiver.cast::<PyByteArray>(runtime)?;
    let raw = args
        .positional()
        .first()
        .map_or(Ok(-1), |value| index_argument(runtime, value))?;
    let mut items = runtime.bytearray_items(array)?;
    if items.is_empty() {
        return Err(PyError::exception("IndexError", "pop from empty bytearray"));
    }
    let index = pop_index(raw, items.len())?;
    runtime.charge_cpu(u64::try_from(items.len()).unwrap_or(u64::MAX))?;
    let byte = items.remove(index);
    runtime.replace_bytearray_items(array, items)?;
    Ok(PyValue::Int(i64::from(byte)))
}

fn bytearray_remove(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytearray.remove", 1, 1)?;
    args.reject_keywords("bytearray.remove")?;
    let array = receiver.cast::<PyByteArray>(runtime)?;
    let byte = byte_argument(runtime, &args.positional()[0])?;
    let mut items = runtime.bytearray_items(array)?;
    runtime.charge_cpu(u64::try_from(items.len()).unwrap_or(u64::MAX))?;
    let position = items
        .iter()
        .position(|item| *item == byte)
        .ok_or_else(|| PyError::value_error("value not found in bytearray"))?;
    items.remove(position);
    runtime.replace_bytearray_items(array, items)?;
    Ok(PyValue::None)
}

fn bytearray_clear(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytearray.clear", 0, 0)?;
    args.reject_keywords("bytearray.clear")?;
    let array = receiver.cast::<PyByteArray>(runtime)?;
    runtime.replace_bytearray_items(array, Vec::new())?;
    Ok(PyValue::None)
}

fn bytearray_copy(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytearray.copy", 0, 0)?;
    args.reject_keywords("bytearray.copy")?;
    let array = receiver.cast::<PyByteArray>(runtime)?;
    let items = runtime.bytearray_items(array)?;
    runtime.new_bytearray(items)
}

fn bytearray_extend(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytearray.extend", 1, 1)?;
    args.reject_keywords("bytearray.extend")?;
    let array = receiver.cast::<PyByteArray>(runtime)?;
    let additions = collect_bytes(runtime, args.positional()[0])?;
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

/// Mutable builtin initialization replaces the payload after reading its source.
fn bytearray_init(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytearray.__init__", 0, 1)?;
    args.reject_keywords("bytearray.__init__")?;
    let items = args
        .positional()
        .first()
        .map(|source| collect_bytes(runtime, *source))
        .transpose()?
        .unwrap_or_default();
    let array = receiver.cast::<PyByteArray>(runtime)?;
    runtime.replace_bytearray_items(array, items)?;
    Ok(Value::None)
}

fn bytearray_reverse(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytearray.reverse", 0, 0)?;
    args.reject_keywords("bytearray.reverse")?;
    let array = receiver.cast::<PyByteArray>(runtime)?;
    let mut items = runtime.bytearray_items(array)?;
    runtime.charge_cpu(u64::try_from(items.len()).unwrap_or(u64::MAX))?;
    items.reverse();
    runtime.replace_bytearray_items(array, items)?;
    Ok(PyValue::None)
}

fn collect_bytes(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<u8>> {
    if runtime.is_unbounded_iterator(&value)? {
        return Err(PyError::resource_error(
            "cannot materialize infinite itertools.count without a bound",
        ));
    }
    let iterator = runtime.iterator(value)?;
    let mut bytes = Vec::new();
    let mut exhausted = false;
    while !exhausted {
        let mut byte = None;
        runtime.nested(&mut |runtime| {
            let Some(value) = runtime.iterator_next(iterator)? else {
                exhausted = true;
                return Ok(());
            };
            byte = Some(byte_argument(runtime, &value)?);
            runtime.charge_cpu(1)
        })?;
        if let Some(byte) = byte {
            // Reserved outside the child scope, which releases its own scratch on close.
            runtime.reserve_memory(1)?;
            bytes.push(byte);
        }
    }
    Ok(bytes)
}

fn collect_values(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<PyValue>> {
    if runtime.is_unbounded_iterator(&value)? {
        return Err(PyError::resource_error(
            "cannot materialize infinite itertools.count without a bound",
        ));
    }
    let iterator = runtime.iterator(value)?;
    // Each step runs in its own pin scope; the list keeps the collected items alive.
    let values = runtime.new_list(Vec::new())?.cast::<PyList>(runtime)?;
    let mut exhausted = false;
    while !exhausted {
        runtime.nested(&mut |runtime| {
            let Some(value) = runtime.iterator_next(iterator)? else {
                exhausted = true;
                return Ok(());
            };
            runtime.charge_cpu(1)?;
            runtime.list_append(values, value)
        })?;
    }
    runtime.list_items(values)
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
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let characters = match args.positional().first() {
        None => None,
        Some(value) if runtime.kind(value)? == super::super::native::PyKind::None => None,
        Some(value) => Some((*value).cast::<OwnedPyString>(runtime)?.0),
    };
    // Each stripped character is tested against the whole set, and the result is a copy.
    let tests = value
        .len()
        .saturating_mul(characters.as_ref().map_or(1, |chars| chars.len().max(1)));
    runtime.charge_cpu(u64::try_from(tests / SEARCH_CHARS_PER_CPU_UNIT + 1).unwrap_or(u64::MAX))?;
    runtime.reserve_memory(value.len())?;
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
    let OwnedPyString(value) = receiver.cast(runtime)?;
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

/// `str.startswith` and `str.endswith`: test one affix or each affix of a tuple against the
/// characters in `[start, end)`, which follow slice clamping.
fn string_affix(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    prefix: bool,
) -> PyResult {
    let name = if prefix { "startswith" } else { "endswith" };
    args.expect_positional(name, 1, 3)?;
    args.reject_keywords(name)?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let affix = args.positional()[0];
    let affixes = match runtime.kind(&affix)? {
        PyKind::String => vec![affix],
        PyKind::Tuple => affix.cast::<PyTuple>(runtime)?.items(runtime)?,
        _ => {
            let actual = runtime.type_name(&affix)?;
            return Err(PyError::type_error(format!(
                "{name} first arg must be str or a tuple of str, not {actual}"
            )));
        }
    };
    // The window is addressed by code point, so the text is decoded once up front.
    runtime.charge_cpu(
        u64::try_from(value.len() / SEARCH_CHARS_PER_CPU_UNIT + 1).unwrap_or(u64::MAX),
    )?;
    runtime.reserve_memory(value.len().saturating_mul(std::mem::size_of::<char>()))?;
    let characters = value.chars().collect::<Vec<_>>();
    let (start, end) = string_bounds(runtime, args.positional(), characters.len())?;
    let window = characters.get(start..end).unwrap_or_default();
    for affix in affixes {
        if runtime.kind(&affix)? != PyKind::String {
            let actual = runtime.type_name(&affix)?;
            return Err(PyError::type_error(format!(
                "tuple for {name} must only contain str, not {actual}"
            )));
        }
        let OwnedPyString(affix) = affix.cast(runtime)?;
        runtime.charge_cpu(u64::try_from(affix.len()).unwrap_or(u64::MAX))?;
        let affix = affix.chars().collect::<Vec<_>>();
        // CPython rejects an empty affix when `start` lies past the end of the string.
        let matched = start <= end
            && if prefix {
                window.starts_with(&affix)
            } else {
                window.ends_with(&affix)
            };
        if matched {
            return Ok(Value::Bool(true));
        }
    }
    Ok(Value::Bool(false))
}

fn string_removeprefix(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_remove_affix(runtime, receiver, args, true)
}

fn string_removesuffix(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_remove_affix(runtime, receiver, args, false)
}

fn string_remove_affix(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    prefix: bool,
) -> PyResult {
    let name = if prefix {
        "removeprefix"
    } else {
        "removesuffix"
    };
    args.expect_positional(name, 1, 1)?;
    args.reject_keywords(name)?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let affix = args.positional()[0];
    if runtime.kind(&affix)? != PyKind::String {
        let actual = runtime.type_name(&affix)?;
        return Err(PyError::type_error(format!(
            "{name}() argument must be str, not {actual}"
        )));
    }
    let OwnedPyString(affix) = affix.cast(runtime)?;
    let stripped = if prefix {
        value.strip_prefix(affix.as_str())
    } else {
        value.strip_suffix(affix.as_str())
    };
    match stripped {
        Some(stripped) => runtime.new_string(stripped.to_owned()),
        None => Ok(receiver),
    }
}

/// `str.expandtabs`: replace each tab with spaces up to the next multiple of `tabsize` columns.
/// Newlines and carriage returns reset the column, and a non-positive `tabsize` deletes tabs.
fn string_expandtabs(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("expandtabs", 0, 1)?;
    args.reject_unknown_keywords("expandtabs", &["tabsize"])?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let tabsize = match args.keyword("expandtabs", "tabsize")? {
        Some(_) if !args.positional().is_empty() => {
            return Err(PyError::type_error(
                "expandtabs() got multiple values for argument 'tabsize'",
            ));
        }
        Some(tabsize) => Some(*tabsize),
        None => args.positional().first().copied(),
    };
    let tabsize = match tabsize {
        Some(tabsize) => runtime.int_value(&tabsize).ok_or_else(|| {
            let actual = runtime
                .type_name(&tabsize)
                .unwrap_or_else(|_| "object".into());
            PyError::type_error(format!(
                "'{actual}' object cannot be interpreted as an integer"
            ))
        })?,
        None => 8,
    };
    let tabsize = usize::try_from(tabsize).unwrap_or(0);
    let too_long = || PyError::overflow_error("new string is too long");
    // Size the result before building it so a large `tabsize` is charged, not allocated.
    let mut column = 0usize;
    let mut length = 0usize;
    for character in value.chars() {
        let width = match character {
            '\t' if tabsize > 0 => tabsize - column % tabsize,
            '\t' => 0,
            _ => 1,
        };
        column = match character {
            '\n' | '\r' => 0,
            _ => column.checked_add(width).ok_or_else(too_long)?,
        };
        length = length.checked_add(width).ok_or_else(too_long)?;
    }
    runtime.reserve_memory(length)?;
    runtime.charge_cpu(u64::try_from(length).unwrap_or(u64::MAX))?;
    let mut output = String::with_capacity(length);
    column = 0;
    for character in value.chars() {
        match character {
            '\t' if tabsize > 0 => {
                let spaces = tabsize - column % tabsize;
                output.extend(std::iter::repeat_n(' ', spaces));
                column += spaces;
            }
            '\t' => {}
            '\n' | '\r' => {
                output.push(character);
                column = 0;
            }
            _ => {
                output.push(character);
                column += 1;
            }
        }
    }
    runtime.new_string(output)
}

/// `str.translate` with a dict, list or tuple table indexed by code point. A missing key or
/// out-of-range index keeps the character; `None` deletes it. Other mapping types would need
/// a general subscript protocol here, so they are rejected as unsupported.
fn string_translate(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("translate", 1, 1)?;
    args.reject_keywords("translate")?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let table = args.positional()[0];
    let table_kind = runtime.kind(&table)?;
    let items = match table_kind {
        PyKind::Dict => None,
        PyKind::List | PyKind::Tuple => Some(table.cast::<PySequence>(runtime)?.items(runtime)?),
        PyKind::Instance | PyKind::Native => {
            return Err(PyError::unsupported(
                "str.translate() with a table that is not a dict, list or tuple",
            ));
        }
        _ => {
            let actual = runtime.type_name(&table)?;
            return Err(PyError::type_error(format!(
                "'{actual}' object is not subscriptable"
            )));
        }
    };
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let mut output = String::with_capacity(value.len());
    for character in value.chars() {
        let code = u32::from(character);
        let mapped = match &items {
            Some(items) => usize::try_from(code)
                .ok()
                .and_then(|index| items.get(index).copied()),
            None => runtime.dict_get(table.cast::<PyDict>(runtime)?, &Value::Int(code.into()))?,
        };
        let Some(mapped) = mapped else {
            output.push(character);
            continue;
        };
        match runtime.kind(&mapped)? {
            PyKind::None => {}
            PyKind::String => {
                let OwnedPyString(text) = mapped.cast(runtime)?;
                runtime.reserve_memory(text.len())?;
                output.push_str(&text);
            }
            _ => {
                let replacement = runtime
                    .int_value(&mapped)
                    .and_then(|code| u32::try_from(code).ok())
                    .and_then(char::from_u32);
                match replacement {
                    Some(replacement) => output.push(replacement),
                    None if matches!(runtime.kind(&mapped)?, PyKind::Int | PyKind::Bool) => {
                        return Err(PyError::value_error(
                            "character mapping must be in range(0x110000)",
                        ));
                    }
                    None => {
                        return Err(PyError::type_error(
                            "character mapping must return integer, None or str",
                        ));
                    }
                }
            }
        }
    }
    runtime.new_string(output)
}

fn string_find(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_find_impl(runtime, receiver, args, false, false)
}

fn string_rfind(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_find_impl(runtime, receiver, args, true, false)
}

fn string_index(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_find_impl(runtime, receiver, args, false, true)
}

fn string_rindex(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_find_impl(runtime, receiver, args, true, true)
}

/// Characters a substring search examines per CPU unit. Rust's substring search is linear in
/// the haystack and needle, so `str.find` and `str.count` cost the same for adversarial
/// inputs, such as `'a' * n` searched for `'a' * m + 'b'`, as for ordinary text.
const SEARCH_CHARS_PER_CPU_UNIT: usize = 16;

/// The slice of `text` holding characters `start..end`, or `None` when `start` is past the end
/// or after `end`.
fn char_range(text: &str, start: usize, end: usize) -> Option<&str> {
    if start > end {
        return None;
    }
    let mut offsets = text
        .char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(text.len()));
    let first = offsets.nth(start)?;
    let last = if end == start {
        first
    } else {
        offsets.nth(end - start - 1).unwrap_or(text.len())
    };
    Some(&text[first..last])
}

fn charge_search(runtime: &mut dyn PyRuntime, haystack: &str, needle: &str) -> PyResult<()> {
    let examined = haystack.len().saturating_add(needle.len());
    runtime.charge_cpu(u64::try_from(examined / SEARCH_CHARS_PER_CPU_UNIT + 1).unwrap_or(u64::MAX))
}

fn string_find_impl(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    reverse: bool,
    raise: bool,
) -> PyResult {
    args.expect_positional("str search", 1, 3)?;
    args.reject_keywords("str search")?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let OwnedPyString(needle) = args.positional()[0].cast(runtime)?;
    charge_search(runtime, &value, &needle)?;
    let length = value.chars().count();
    let (start, end) = string_bounds(runtime, args.positional(), length)?;
    let found = if needle.is_empty() {
        (start <= end).then_some(if reverse { end } else { start })
    } else {
        char_range(&value, start, end).and_then(|haystack| {
            let offset = if reverse {
                haystack.rfind(needle.as_str())
            } else {
                haystack.find(needle.as_str())
            }?;
            Some(start + haystack[..offset].chars().count())
        })
    };
    match found {
        Some(index) => i64::try_from(index)
            .map(Value::Int)
            .map_err(|_| PyError::overflow_error("string index is too large")),
        None if raise => Err(PyError::value_error("substring not found")),
        None => Ok(Value::Int(-1)),
    }
}

fn string_count(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.count", 1, 3)?;
    args.reject_keywords("str.count")?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let OwnedPyString(needle) = args.positional()[0].cast(runtime)?;
    charge_search(runtime, &value, &needle)?;
    let length = value.chars().count();
    let (start, end) = string_bounds(runtime, args.positional(), length)?;
    let count = if needle.is_empty() {
        if start <= end {
            end - start + 1
        } else {
            0
        }
    } else {
        // `matches` yields non-overlapping occurrences from the left, as `str.count` counts.
        char_range(&value, start, end)
            .map_or(0, |haystack| haystack.matches(needle.as_str()).count())
    };
    i64::try_from(count)
        .map(Value::Int)
        .map_err(|_| PyError::overflow_error("string count is too large"))
}

fn string_bounds(
    runtime: &dyn PyRuntime,
    arguments: &[PyValue],
    length: usize,
) -> PyResult<(usize, usize)> {
    let length_i64 = i64::try_from(length).unwrap_or(i64::MAX);
    let normalize_start = |value: i64| {
        if value < 0 {
            usize::try_from(length_i64.saturating_add(value).max(0)).unwrap_or_default()
        } else {
            usize::try_from(value).unwrap_or(usize::MAX)
        }
    };
    let normalize_end = |value: i64| {
        if value < 0 {
            usize::try_from(length_i64.saturating_add(value).max(0)).unwrap_or_default()
        } else {
            usize::try_from(value).unwrap_or(usize::MAX).min(length)
        }
    };
    let index = |value: &PyValue| -> PyResult<Option<i64>> {
        if value.is_none() {
            return Ok(None);
        }
        runtime.int_value(value).map(Some).ok_or_else(|| {
            PyError::type_error(
                "slice indices must be integers or None or have an __index__ method",
            )
        })
    };
    let start = match arguments.get(1) {
        Some(value) => index(value)?.map_or(0, normalize_start),
        None => 0,
    };
    let end = match arguments.get(2) {
        Some(value) => index(value)?.map_or(length, normalize_end),
        None => length,
    };
    Ok((start, end))
}

fn string_partition(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_partition_impl(runtime, receiver, args, false)
}

fn string_rpartition(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_partition_impl(runtime, receiver, args, true)
}

fn string_partition_impl(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    reverse: bool,
) -> PyResult {
    args.expect_positional("str.partition", 1, 1)?;
    args.reject_keywords("str.partition")?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let OwnedPyString(separator) = args.positional()[0].cast(runtime)?;
    if separator.is_empty() {
        return Err(PyError::value_error("empty separator"));
    }
    let parts = if reverse {
        value.rsplit_once(&separator)
    } else {
        value.split_once(&separator)
    };
    let (left, middle, right) = match parts {
        Some((left, right)) => (left.to_string(), separator, right.to_string()),
        None if reverse => (String::new(), String::new(), value),
        None => (value, String::new(), String::new()),
    };
    let left = runtime.new_string(left)?;
    let middle = runtime.new_string(middle)?;
    let right = runtime.new_string(right)?;
    runtime.new_tuple(vec![left, middle, right])
}

fn string_split(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.split", 0, 2)?;
    args.reject_keywords("str.split")?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let separator = match args.positional().first() {
        None => None,
        Some(value) if runtime.kind(value)? == super::super::native::PyKind::None => None,
        Some(value) => {
            let OwnedPyString(value) = (*value).cast(runtime)?;
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

fn string_rsplit(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.rsplit", 0, 2)?;
    args.reject_keywords("str.rsplit")?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let separator = match args.positional().first() {
        None => None,
        Some(value) if runtime.kind(value)? == PyKind::None => None,
        Some(value) => {
            let OwnedPyString(value) = (*value).cast(runtime)?;
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
        .transpose()?
        .unwrap_or(-1);
    let limit = if maximum < 0 {
        usize::MAX
    } else {
        usize::try_from(maximum).unwrap_or(usize::MAX)
    };
    let mut parts = match separator.as_deref() {
        Some(separator) => value
            .rsplitn(limit.saturating_add(1), separator)
            .map(str::to_string)
            .collect::<Vec<_>>(),
        None => whitespace_rsplit(&value, limit),
    };
    parts.reverse();
    let values = parts
        .into_iter()
        .map(|part| runtime.new_string(part))
        .collect::<PyResult<Vec<_>>>()?;
    runtime.new_list(values)
}

fn whitespace_rsplit(value: &str, limit: usize) -> Vec<String> {
    let value = value.trim_end_matches(char::is_whitespace);
    if value.is_empty() {
        return Vec::new();
    }
    if limit == 0 {
        return vec![value.to_string()];
    }
    let mut parts = Vec::new();
    let mut end = value.len();
    while parts.len() < limit {
        let mut word_start = end;
        for (index, character) in value[..end].char_indices().rev() {
            if character.is_whitespace() {
                break;
            }
            word_start = index;
        }
        let mut separator_start = word_start;
        for (index, character) in value[..word_start].char_indices().rev() {
            if !character.is_whitespace() {
                break;
            }
            separator_start = index;
        }
        if separator_start == word_start {
            break;
        }
        parts.push(value[word_start..end].to_string());
        end = separator_start;
    }
    if end > 0 {
        parts.push(value[..end].to_string());
    }
    parts
}

fn string_splitlines(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.splitlines", 0, 1)?;
    args.reject_unknown_keywords("str.splitlines", &["keepends"])?;
    let keepends = match args
        .positional()
        .first()
        .or(args.keyword("str.splitlines", "keepends")?)
    {
        Some(value) => runtime.truth(value)?,
        None => false,
    };
    let OwnedPyString(value) = receiver.cast(runtime)?;
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

/// Modeled header cost of one host string held while joining, on top of its bytes.
const JOINED_PART_BYTES: usize = 32;

fn string_join(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.join", 1, 1)?;
    args.reject_keywords("str.join")?;
    let OwnedPyString(separator) = receiver.cast(runtime)?;
    let iterator = runtime.iterator(args.positional()[0])?;
    let mut parts: Vec<String> = Vec::new();
    let mut bytes = 0usize;
    let mut exhausted = false;
    while !exhausted {
        let mut part = None;
        runtime.nested(&mut |runtime| {
            let Some(value) = runtime.iterator_next(iterator)? else {
                exhausted = true;
                return Ok(());
            };
            runtime.charge_cpu(1)?;
            let OwnedPyString(value) = value.cast(runtime)?;
            part = Some(value);
            Ok(())
        })?;
        let Some(part) = part else { continue };
        bytes = bytes
            .checked_add(part.len())
            .ok_or_else(|| PyError::resource_error("joined string is too large"))?;
        // Each part is a host string that lives until the join finishes. It is reserved here, in
        // the scope that keeps it, because scratch reserved inside the child scope is released
        // when that scope closes.
        runtime.reserve_memory(part.len().saturating_add(JOINED_PART_BYTES))?;
        parts.push(part);
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
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let OwnedPyString(old) = args.positional()[0].cast(runtime)?;
    let OwnedPyString(new) = args.positional()[1].cast(runtime)?;
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

fn string_ljust(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_justify(runtime, receiver, args, false)
}

fn string_rjust(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    string_justify(runtime, receiver, args, true)
}

/// The left share of `padding` for `str.center` and `bytes.center`: half, plus the odd unit when
/// both the padding and the requested width are odd, as CPython does. So `'ab'.center(5)` is
/// `'  ab '` but `'a'.center(4)` is `' a  '`.
fn center_left_padding(padding: usize, width: i64) -> usize {
    padding / 2 + (padding & usize::try_from(width).unwrap_or_default() & 1)
}

fn string_center(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("str.center", 1, 2)?;
    args.reject_keywords("str.center")?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let width = runtime
        .int_value(&args.positional()[0])
        .ok_or_else(|| PyError::type_error("width must be an integer"))?;
    let fill = if let Some(fill) = args.positional().get(1) {
        let OwnedPyString(fill) = (*fill).cast(runtime)?;
        if fill.chars().count() != 1 {
            return Err(PyError::type_error(
                "the fill character must be exactly one character long",
            ));
        }
        fill
    } else {
        " ".to_string()
    };
    let padding = usize::try_from(width)
        .ok()
        .unwrap_or_default()
        .saturating_sub(value.chars().count());
    let left = center_left_padding(padding, width);
    let right = padding - left;
    let fill_bytes = fill
        .len()
        .checked_mul(padding)
        .ok_or_else(|| PyError::resource_error("centered string is too large"))?;
    let capacity = value
        .len()
        .checked_add(fill_bytes)
        .ok_or_else(|| PyError::resource_error("centered string is too large"))?;
    runtime.reserve_memory(capacity)?;
    runtime.charge_cpu(u64::try_from(capacity).unwrap_or(u64::MAX))?;
    runtime.new_string(format!(
        "{}{value}{}",
        fill.repeat(left),
        fill.repeat(right)
    ))
}

fn string_justify(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    right: bool,
) -> PyResult {
    let name = if right { "str.rjust" } else { "str.ljust" };
    args.expect_positional(name, 1, 2)?;
    args.reject_keywords(name)?;
    let OwnedPyString(value) = receiver.cast(runtime)?;
    let width = runtime
        .int_value(&args.positional()[0])
        .ok_or_else(|| PyError::type_error("width must be an integer"))?;
    let fill = if let Some(fill) = args.positional().get(1) {
        let OwnedPyString(fill) = (*fill).cast(runtime)?;
        if fill.chars().count() != 1 {
            return Err(PyError::type_error(
                "the fill character must be exactly one character long",
            ));
        }
        fill
    } else {
        " ".to_string()
    };
    let padding = usize::try_from(width)
        .ok()
        .unwrap_or_default()
        .saturating_sub(value.chars().count());
    let added = fill
        .len()
        .checked_mul(padding)
        .ok_or_else(|| PyError::resource_error("justified string is too large"))?;
    let capacity = value
        .len()
        .checked_add(added)
        .ok_or_else(|| PyError::resource_error("justified string is too large"))?;
    runtime.reserve_memory(capacity)?;
    runtime.charge_cpu(u64::try_from(capacity).unwrap_or(u64::MAX))?;
    let padding = fill.repeat(padding);
    runtime.new_string(if right {
        format!("{padding}{value}")
    } else {
        format!("{value}{padding}")
    })
}

/// Implement the established string `%` protocol without routing numeric remainder through it.
pub(crate) fn slot_string_remainder(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let OwnedPyString(template) = left.cast(runtime)?;
    runtime.reserve_memory(template.len())?;
    let arguments = if runtime.kind(&right)? == PyKind::Tuple {
        right.cast::<PyTuple>(runtime)?.items(runtime)?
    } else {
        vec![right]
    };
    let mut argument = 0usize;
    let mut used_mapping = false;
    let mut output = String::new();
    let characters = template.chars().collect::<Vec<_>>();
    let mut index = 0usize;
    while index < characters.len() {
        runtime.charge_cpu(1)?;
        if characters[index] != '%' {
            output.push(characters[index]);
            index += 1;
            continue;
        }
        index += 1;
        if characters.get(index) == Some(&'%') {
            output.push('%');
            index += 1;
            continue;
        }
        let mapping_key = if characters.get(index) == Some(&'(') {
            index += 1;
            let start = index;
            while characters
                .get(index)
                .is_some_and(|character| *character != ')')
            {
                index += 1;
            }
            if characters.get(index) != Some(&')') {
                return Err(PyError::value_error("incomplete format key"));
            }
            let key = characters[start..index].iter().collect::<String>();
            index += 1;
            Some(key)
        } else {
            None
        };
        let mut left_align = false;
        let mut plus = false;
        let mut space = false;
        let mut alternate = false;
        let mut zero = false;
        while let Some(flag) = characters.get(index) {
            match flag {
                '-' => left_align = true,
                '+' => plus = true,
                ' ' => space = true,
                '#' => alternate = true,
                '0' => zero = true,
                _ => break,
            }
            index += 1;
        }
        let width = parse_format_digits(&characters, &mut index)?;
        let precision = if characters.get(index) == Some(&'.') {
            index += 1;
            Some(parse_format_digits(&characters, &mut index)?.unwrap_or(0))
        } else {
            None
        };
        let requested = width.unwrap_or_default().max(precision.unwrap_or_default());
        runtime.reserve_memory(requested)?;
        runtime.charge_cpu(u64::try_from(requested).unwrap_or(u64::MAX))?;
        let conversion_index = index;
        let conversion = *characters
            .get(index)
            .ok_or_else(|| PyError::value_error("incomplete format"))?;
        index += 1;
        if !"sradiuxXofFeEgGc".contains(conversion) {
            return Err(PyError::value_error(format!(
                "unsupported format character '{conversion}' ({:#x}) at index {conversion_index}",
                u32::from(conversion)
            )));
        }
        let value = if let Some(key) = mapping_key {
            used_mapping = true;
            // Any mapping works here, read through its `__getitem__`; a tuple or string is a
            // sequence of arguments instead.
            if matches!(runtime.kind(&right)?, PyKind::Tuple | PyKind::String) {
                return Err(PyError::type_error("format requires a mapping"));
            }
            let key = runtime.new_string(key.clone())?;
            runtime.get_item(right, key)?
        } else {
            let value = arguments
                .get(argument)
                .copied()
                .ok_or_else(|| PyError::type_error("not enough arguments for format string"))?;
            argument += 1;
            value
        };
        let rendered = match conversion {
            'f' | 'F' | 'e' | 'E' | 'g' | 'G' => {
                // The format-spec mini-language renders floats the same way, including zero
                // padding of nan and inf.
                let number = match value.cast::<PyNumber>(runtime) {
                    Ok(number) => number.into_f64()?,
                    Err(_) => {
                        let actual = runtime.type_name(&value)?;
                        return Err(PyError::type_error(format!(
                            "must be real number, not {actual}"
                        )));
                    }
                };
                let mut spec = String::new();
                if left_align {
                    spec.push('<');
                }
                if plus {
                    spec.push('+');
                } else if space {
                    spec.push(' ');
                }
                if alternate {
                    spec.push('#');
                }
                if zero && !left_align {
                    spec.push('0');
                }
                if let Some(width) = width {
                    spec.push_str(&width.to_string());
                }
                spec.push_str(&format!(".{}{conversion}", precision.unwrap_or(6)));
                runtime.format_value(&Value::Float(number), None, &spec)?
            }
            'd' | 'i' | 'u' | 'x' | 'X' | 'o' => {
                let integer = percent_integer(runtime, &value, conversion)?;
                let negative = integer.sign() == Sign::Minus;
                let magnitude = integer.magnitude();
                let digits = match conversion {
                    'x' => magnitude.to_str_radix(16),
                    'X' => magnitude.to_str_radix(16).to_ascii_uppercase(),
                    'o' => magnitude.to_str_radix(8),
                    _ => magnitude.to_string(),
                };
                let digits = pad_integer_precision(digits, precision);
                let prefix = match (alternate, conversion) {
                    (true, 'x') => "0x",
                    (true, 'X') => "0X",
                    (true, 'o') => "0o",
                    _ => "",
                };
                let sign = if negative {
                    "-"
                } else if plus {
                    "+"
                } else if space {
                    " "
                } else {
                    ""
                };
                let length = sign.len() + prefix.len() + digits.len();
                let padding = width.unwrap_or(0).saturating_sub(length);
                if left_align {
                    format!("{sign}{prefix}{digits}{}", " ".repeat(padding))
                } else if zero {
                    format!("{sign}{prefix}{}{digits}", "0".repeat(padding))
                } else {
                    format!("{}{sign}{prefix}{digits}", " ".repeat(padding))
                }
            }
            _ => {
                let mut text = match conversion {
                    's' => runtime.display(&value)?,
                    'r' | 'a' => runtime.repr(&value)?,
                    _ => percent_character(runtime, &value)?,
                };
                if conversion != 'c' {
                    if let Some(precision) = precision {
                        text = text.chars().take(precision).collect();
                    }
                }
                let padding = width.unwrap_or(0).saturating_sub(text.chars().count());
                if left_align {
                    text.extend(std::iter::repeat_n(' ', padding));
                    text
                } else {
                    format!("{}{text}", " ".repeat(padding))
                }
            }
        };
        runtime.reserve_memory(rendered.len())?;
        output.push_str(&rendered);
    }
    if !used_mapping && argument < arguments.len() {
        return Err(PyError::type_error(
            "not all arguments converted during string formatting",
        ));
    }
    runtime.reserve_memory(output.len())?;
    Ok(Some(runtime.new_string(output)?))
}

fn parse_format_digits(characters: &[char], index: &mut usize) -> PyResult<Option<usize>> {
    let start = *index;
    let mut value = 0usize;
    while let Some(character) = characters.get(*index).and_then(|value| value.to_digit(10)) {
        value = value
            .checked_mul(10)
            .and_then(|value| value.checked_add(character as usize))
            .ok_or_else(|| PyError::resource_error("format width is too large"))?;
        *index += 1;
    }
    Ok((*index != start).then_some(value))
}

/// Zero-extend unsigned `digits` to `precision` digits, which `%`-formatting treats as a
/// minimum digit count for integers.
fn pad_integer_precision(digits: String, precision: Option<usize>) -> String {
    match precision {
        Some(precision) if digits.len() < precision => {
            format!("{}{digits}", "0".repeat(precision - digits.len()))
        }
        _ => digits,
    }
}

/// The integer a `%d`-style conversion formats. `%d`, `%i` and `%u` truncate a float as
/// `int()` does; `%x`, `%X` and `%o` require an integer.
fn percent_integer(
    runtime: &mut dyn PyRuntime,
    value: &PyValue,
    conversion: char,
) -> PyResult<BigInt> {
    if let Some(integer) = runtime.integer_bigint(value)? {
        if matches!(conversion, 'd' | 'i' | 'u')
            && super::super::number::exceeds_str_digits(&integer)
        {
            return Err(super::super::number::int_str_digits_error());
        }
        return Ok(integer);
    }
    let actual = runtime.type_name(value)?;
    if !matches!(conversion, 'd' | 'i' | 'u') {
        return Err(PyError::type_error(format!(
            "%{conversion} format: an integer is required, not {actual}"
        )));
    }
    let number = value
        .cast::<PyNumber>(runtime)
        .and_then(PyNumber::into_f64)
        .map_err(|_| {
            PyError::type_error(format!(
                "%{conversion} format: a real number is required, not {actual}"
            ))
        })?;
    if number.is_nan() {
        return Err(PyError::value_error("cannot convert float NaN to integer"));
    }
    num_traits::FromPrimitive::from_f64(number.trunc())
        .ok_or_else(|| PyError::overflow_error("cannot convert float infinity to integer"))
}

/// The character `%c` formats: an int code point or a one-character string.
fn percent_character(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<String> {
    if let Some(integer) = runtime.integer_bigint(value)? {
        return integer
            .to_u32()
            .and_then(char::from_u32)
            .map(String::from)
            .ok_or_else(|| PyError::overflow_error("%c arg not in range(0x110000)"));
    }
    let requirement = "%c requires an int or a unicode character";
    if runtime.kind(value)? != PyKind::String {
        let actual = runtime.type_name(value)?;
        return Err(PyError::type_error(format!("{requirement}, not {actual}")));
    }
    let OwnedPyString(text) = value.cast(runtime)?;
    match text.chars().count() {
        1 => Ok(text),
        length => Err(PyError::type_error(format!(
            "{requirement}, not a string of length {length}"
        ))),
    }
}

fn string_format(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let OwnedPyString(template) = receiver.cast(runtime)?;
    format_template(
        runtime,
        &template,
        args.positional(),
        NamedFields::Keywords(args.keywords()),
    )
}

/// `str.format_map(mapping)`: `str.format` with named fields read from `mapping` through its
/// `__getitem__`, so a dict subclass's `__missing__` can supply absent names.
fn string_format_map(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("format_map", 1, 1)?;
    args.reject_keywords("format_map")?;
    let OwnedPyString(template) = receiver.cast(runtime)?;
    format_template(
        runtime,
        &template,
        &[],
        NamedFields::Mapping(args.positional()[0]),
    )
}

/// Where a named replacement field such as `{name}` finds its value.
enum NamedFields<'a> {
    /// `str.format`'s keyword arguments.
    Keywords(&'a [(String, PyValue)]),
    /// `str.format_map`'s mapping.
    Mapping(PyValue),
}

fn format_template(
    runtime: &mut dyn PyRuntime,
    template: &str,
    positional: &[PyValue],
    named: NamedFields<'_>,
) -> PyResult {
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
                let (field, conversion, specification) = parse_format_field(&field)?;
                let value = if field.is_empty() {
                    if used_manual_index {
                        return Err(PyError::value_error(
                            "cannot switch from manual field specification to automatic field numbering",
                        ));
                    }
                    used_automatic = true;
                    let value = positional
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
                    *positional
                        .get(index)
                        .ok_or_else(|| PyError::value_error("replacement index out of range"))?
                } else {
                    match named {
                        NamedFields::Keywords(keywords) => {
                            match keywords.iter().find(|(name, _)| name == field) {
                                Some((_, value)) => *value,
                                None => {
                                    let key = runtime.new_string(field.to_string())?;
                                    return Err(runtime.exception_with_args("KeyError", vec![key]));
                                }
                            }
                        }
                        NamedFields::Mapping(mapping) => {
                            let key = runtime.new_string(field.to_string())?;
                            runtime.get_item(mapping, key)?
                        }
                    }
                };
                result.push_str(&runtime.format_value(&value, conversion, specification)?);
            }
            '}' => return Err(PyError::value_error("single '}' in format string")),
            character => result.push(character),
        }
    }
    runtime.reserve_memory(result.len())?;
    runtime.new_string(result)
}

fn parse_format_field(field: &str) -> PyResult<(&str, Option<char>, &str)> {
    let (selector_and_conversion, specification) =
        field.split_once(':').map_or((field, ""), |parts| parts);
    let (selector, conversion) = match selector_and_conversion.split_once('!') {
        Some((selector, conversion)) => {
            let mut characters = conversion.chars();
            let conversion = characters
                .next()
                .filter(|conversion| matches!(conversion, 'r' | 's' | 'a'))
                .ok_or_else(|| PyError::value_error("unknown format conversion"))?;
            if characters.next().is_some() {
                return Err(PyError::value_error("invalid format conversion"));
            }
            (selector, Some(conversion))
        }
        None => (selector_and_conversion, None),
    };
    Ok((selector, conversion, specification))
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
    runtime.list_append(list, args.positional()[0])?;
    Ok(Value::None)
}

fn list_insert(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.insert", 2, 2)?;
    args.reject_keywords("list.insert")?;
    let list = receiver.cast::<PyList>(runtime)?;
    let raw = index_argument(runtime, &args.positional()[0])?;
    let index = insert_index(raw, runtime.list_len(list)?);
    runtime.list_insert(list, index, args.positional()[1])?;
    Ok(Value::None)
}

/// Resolve an `insert` index as CPython does: negative values count from the end and every
/// out-of-range value clamps to the nearest end.
fn insert_index(raw: i64, length: usize) -> usize {
    let resolved = if raw < 0 {
        i64::try_from(length)
            .unwrap_or(i64::MAX)
            .saturating_add(raw)
    } else {
        raw
    };
    usize::try_from(resolved.max(0))
        .unwrap_or(usize::MAX)
        .min(length)
}

fn list_extend(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.extend", 1, 1)?;
    args.reject_keywords("list.extend")?;
    let list = receiver.cast::<PyList>(runtime)?;
    let values = collect_values(runtime, args.positional()[0])?;
    runtime.list_extend(list, values)?;
    Ok(Value::None)
}

fn list_init(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.__init__", 0, 1)?;
    args.reject_keywords("list.__init__")?;
    let items = args
        .positional()
        .first()
        .map(|source| collect_values(runtime, *source))
        .transpose()?
        .unwrap_or_default();
    let list = receiver.cast::<PyList>(runtime)?;
    runtime.replace_list_items(list, items)?;
    Ok(Value::None)
}

fn list_pop(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.pop", 0, 1)?;
    args.reject_keywords("list.pop")?;
    let list = receiver.cast::<PyList>(runtime)?;
    let raw = args
        .positional()
        .first()
        .map_or(Ok(-1), |value| index_argument(runtime, value))?;
    let length = runtime.list_len(list)?;
    if length == 0 {
        return Err(PyError::exception("IndexError", "pop from empty list"));
    }
    let index = pop_index(raw, length)?;
    runtime.list_pop(list, index)
}

/// Resolve a possibly negative `pop` index against `length`, raising CPython's `IndexError`.
fn pop_index(raw: i64, length: usize) -> PyResult<usize> {
    let resolved = if raw < 0 {
        i64::try_from(length)
            .unwrap_or(i64::MAX)
            .saturating_add(raw)
    } else {
        raw
    };
    usize::try_from(resolved)
        .ok()
        .filter(|index| *index < length)
        .ok_or_else(|| PyError::exception("IndexError", "pop index out of range"))
}

/// `slice.indices(length)`: the normalized `(start, stop, step)` for a sequence of `length`.
fn slice_indices(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.reject_keywords("slice.indices")?;
    let [length] = args.positional() else {
        return Err(PyError::type_error(format!(
            "slice.indices() takes exactly one argument ({} given)",
            args.positional().len()
        )));
    };
    let length = index_argument(runtime, length)?;
    if length < 0 {
        return Err(PyError::value_error("length should not be negative"));
    }
    let (start, stop, step) = runtime
        .slice_parts(&receiver)?
        .ok_or_else(|| PyError::type_error("descriptor 'indices' requires a 'slice' object"))?;
    let (start, stop, step) = super::super::slice::slice_indices(length, start, stop, step)
        .map_err(PyError::value_error)?;
    runtime.new_tuple(vec![
        PyValue::Int(start),
        PyValue::Int(stop),
        PyValue::Int(step),
    ])
}

fn list_remove(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.remove", 1, 1)?;
    args.reject_keywords("list.remove")?;
    let list = receiver.cast::<PyList>(runtime)?;
    let position = runtime
        .list_position(list, &args.positional()[0], 0, usize::MAX)?
        .ok_or_else(|| PyError::value_error("list.remove(x): x not in list"))?;
    runtime.list_pop(list, position)?;
    Ok(Value::None)
}

fn list_reverse(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.reverse", 0, 0)?;
    args.reject_keywords("list.reverse")?;
    let list = receiver.cast::<PyList>(runtime)?;
    runtime.list_reverse(list)?;
    Ok(Value::None)
}

fn list_clear(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.clear", 0, 0)?;
    args.reject_keywords("list.clear")?;
    let list = receiver.cast::<PyList>(runtime)?;
    runtime.list_clear(list)?;
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
    count_equal(runtime, &values, &args.positional()[0], "list")
}

fn list_index(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("list.index", 1, 3)?;
    args.reject_keywords("list.index")?;
    let values = receiver.cast::<PyList>(runtime)?.items(runtime)?;
    index_of(runtime, &values, args.positional(), "list")
}

/// `tuple.__new__(cls, iterable=())`, which receives the class explicitly.
fn tuple_new(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    runtime.new_builtin_instance(BuiltinType::Tuple, receiver, args)
}

/// `tuple.__repr__(self)`, which a subclass's own `__repr__` may call.
fn tuple_repr(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("tuple.__repr__", 0, 0)?;
    args.reject_keywords("tuple.__repr__")?;
    let tuple = receiver.cast::<PyTuple>(runtime)?;
    let text = runtime.payload_repr(&tuple.value())?;
    runtime.new_string(text)
}

fn tuple_count(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("tuple.count", 1, 1)?;
    args.reject_keywords("tuple.count")?;
    let values = receiver.cast::<PyTuple>(runtime)?.items(runtime)?;
    count_equal(runtime, &values, &args.positional()[0], "tuple")
}

fn tuple_index(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("tuple.index", 1, 3)?;
    args.reject_keywords("tuple.index")?;
    let values = receiver.cast::<PyTuple>(runtime)?.items(runtime)?;
    index_of(runtime, &values, args.positional(), "tuple")
}

/// `sequence.count(value)` for a list or tuple of `values`.
fn count_equal(
    runtime: &mut dyn PyRuntime,
    values: &[PyValue],
    value: &PyValue,
    kind: &str,
) -> PyResult {
    let mut count = 0i64;
    for item in values {
        runtime.charge_cpu(1)?;
        if runtime.equals(item, value)? {
            count = count
                .checked_add(1)
                .ok_or_else(|| PyError::overflow_error(format!("{kind} is too large")))?;
        }
    }
    Ok(Value::Int(count))
}

/// `sequence.index(value, start=0, stop=len)` for a list or tuple of `values`; `arguments` are
/// the method's positional arguments.
fn index_of(
    runtime: &mut dyn PyRuntime,
    values: &[PyValue],
    arguments: &[PyValue],
    kind: &str,
) -> PyResult {
    let length = i64::try_from(values.len())
        .map_err(|_| PyError::overflow_error(format!("{kind} too large")))?;
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
    let start = normalize(endpoint(arguments.get(1), 0)?);
    let stop = normalize(endpoint(arguments.get(2), length)?);
    for index in start..stop {
        runtime.charge_cpu(1)?;
        if runtime.equals(&values[index as usize], &arguments[0])? {
            return Ok(Value::Int(index));
        }
    }
    Err(PyError::value_error(format!(
        "{kind}.index(x): x not in {kind}"
    )))
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
    // Like CPython's sort this asks only `<`, and a reversed sort keeps equal items in their
    // original order.
    super::super::sort::merge_sort(&mut keyed, |right, left| {
        runtime.charge_cpu(1)?;
        let (lesser, greater) = if reverse {
            (left.0, right.0)
        } else {
            (right.0, left.0)
        };
        runtime.less_than(&lesser, &greater)
    })?;
    runtime.replace_list_items(list, keyed.into_iter().map(|(_, value)| value).collect())?;
    Ok(Value::None)
}

/// `dict.__init__(self, other=(), **pairs)`: add the entries to `self`, which a subclass's own
/// `__init__` reaches through `super().__init__(...)`.
fn dict_init(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    if args.positional().len() > 1 {
        return Err(PyError::type_error(format!(
            "dict expected at most 1 argument, got {}",
            args.positional().len()
        )));
    }
    dict_update(runtime, receiver, args)
}

/// `dict.__getitem__(self, key)`, which a subclass's own `__getitem__` may call.
fn dict_getitem(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.__getitem__", 1, 1)?;
    args.reject_keywords("dict.__getitem__")?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    let key = args.positional()[0];
    if let Some(value) = runtime.dict_get(dict, &key)? {
        return Ok(value);
    }
    // A dict subclass supplies absent keys through `__missing__`, as `d[key]` does in CPython.
    if let Some(missing) = runtime.get_attribute_default(receiver, "__missing__")? {
        return runtime.call_value(missing, CallArgs::new(vec![key], Vec::new()));
    }
    Err(runtime.exception_with_args("KeyError", vec![key]))
}

fn dict_setitem(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.__setitem__", 2, 2)?;
    args.reject_keywords("dict.__setitem__")?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    runtime.dict_insert(dict, args.positional()[0], args.positional()[1])?;
    Ok(Value::None)
}

fn dict_delitem(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.__delitem__", 1, 1)?;
    args.reject_keywords("dict.__delitem__")?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    let key = args.positional()[0];
    match runtime.dict_remove(dict, &key)? {
        Some(_) => Ok(Value::None),
        None => Err(runtime.exception_with_args("KeyError", vec![key])),
    }
}

fn dict_contains(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.__contains__", 1, 1)?;
    args.reject_keywords("dict.__contains__")?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    Ok(Value::Bool(
        runtime.dict_get(dict, &args.positional()[0])?.is_some(),
    ))
}

fn dict_iter(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.__iter__", 0, 0)?;
    args.reject_keywords("dict.__iter__")?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    let iterator = runtime.iterator(dict.value())?;
    Ok(iterator.value())
}

/// `dict.__repr__(self)`, which a subclass's own `__repr__` may call.
fn dict_repr(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.__repr__", 0, 0)?;
    args.reject_keywords("dict.__repr__")?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    let text = runtime.payload_repr(&dict.value())?;
    runtime.new_string(text)
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
    if let Some(value) = runtime.dict_get(dict, &args.positional()[0])? {
        return Ok(value);
    }
    let default = args.positional().get(1).copied().unwrap_or(Value::None);
    if insert {
        runtime.dict_insert(dict, args.positional()[0], default)?;
    }
    Ok(default)
}

fn dict_keys(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    dict_view(runtime, receiver, args, DictViewKind::Keys)
}

fn dict_values(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    dict_view(runtime, receiver, args, DictViewKind::Values)
}

fn dict_items(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    dict_view(runtime, receiver, args, DictViewKind::Items)
}

/// A live view of the dict's keys, values or items, which `mapping_views` implements.
fn dict_view(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    kind: DictViewKind,
) -> PyResult {
    args.expect_positional("dict view", 0, 0)?;
    args.reject_keywords("dict view")?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    runtime.new_dict_view(kind, dict.value())
}

fn dict_update(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.update", 0, 1)?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    let mut additions = Vec::new();
    if let Some(source) = args.positional().first() {
        if let Some(entries) = runtime.mapping_items(*source)? {
            additions.extend(entries);
        } else {
            let iterator = runtime.iterator(*source)?;
            // Each step runs in its own pin scope; the flat key, value list keeps the pairs
            // alive until they are inserted.
            let pairs = runtime.new_list(Vec::new())?.cast::<PyList>(runtime)?;
            let mut exhausted = false;
            while !exhausted {
                runtime.nested(&mut |runtime| {
                    let Some(item) = runtime.iterator_next(iterator)? else {
                        exhausted = true;
                        return Ok(());
                    };
                    let pair = item.cast::<PySequence>(runtime)?.items(runtime)?;
                    if pair.len() != 2 {
                        return Err(PyError::value_error(
                            "dictionary update sequence element has length other than 2",
                        ));
                    }
                    runtime.list_append(pairs, pair[0])?;
                    runtime.list_append(pairs, pair[1])
                })?;
            }
            let pairs = runtime.list_items(pairs)?;
            additions.extend(pairs.chunks_exact(2).map(|pair| (pair[0], pair[1])));
        }
    }
    for (name, value) in args.keywords() {
        additions.push((runtime.new_string(name.clone())?, *value));
    }
    for (key, value) in additions {
        runtime.dict_insert(dict, key, value)?;
    }
    Ok(Value::None)
}

fn dict_pop(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.pop", 1, 2)?;
    args.reject_keywords("dict.pop")?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    if let Some(value) = runtime.dict_remove(dict, &args.positional()[0])? {
        return Ok(value);
    }
    if let Some(default) = args.positional().get(1) {
        return Ok(*default);
    }
    Err(runtime.exception_with_args("KeyError", vec![args.positional()[0]]))
}

/// Remove and return the most recently inserted `(key, value)` pair.
fn dict_popitem(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.popitem", 0, 0)?;
    args.reject_keywords("dict.popitem")?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    let Some(key) = runtime.dict_last_key(dict)? else {
        let message = runtime.new_string("popitem(): dictionary is empty".into())?;
        return Err(runtime.exception_with_args("KeyError", vec![message]));
    };
    let Some(value) = runtime.dict_remove(dict, &key)? else {
        return Err(runtime.exception_with_args("KeyError", vec![key]));
    };
    runtime.new_tuple(vec![key, value])
}

fn dict_clear(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.clear", 0, 0)?;
    args.reject_keywords("dict.clear")?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    runtime.replace_dict_items(dict, Vec::new())?;
    Ok(Value::None)
}

/// `dict.fromkeys(iterable, value=None)`: map every key to the same `value` object.
///
/// The receiver is the `dict` type, bound through `DICT_CLASS_METHODS`. Shellsim does not support
/// subclasses of builtin `dict`, so the result is always a plain `dict`.
fn dict_fromkeys(runtime: &mut dyn PyRuntime, _class: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.fromkeys", 1, 2)?;
    args.reject_keywords("dict.fromkeys")?;
    let value = args.positional().get(1).copied().unwrap_or(Value::None);
    let keys = collect_values(runtime, args.positional()[0])?;
    let result = runtime.new_dict(Vec::new())?;
    let dict = result.cast::<PyDict>(runtime)?;
    for key in keys {
        runtime.dict_insert(dict, key, value)?;
    }
    Ok(result)
}

/// Decode pairs of hex digits, ignoring ASCII whitespace between pairs as CPython does.
fn parse_hex_bytes(runtime: &mut dyn PyRuntime, text: PyValue) -> PyResult<Vec<u8>> {
    let OwnedPyString(text) = text.cast(runtime)?;
    runtime.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() {
            index += 1;
            continue;
        }
        let digit = |position: usize| -> PyResult<u8> {
            bytes
                .get(position)
                .and_then(|byte| char::from(*byte).to_digit(16))
                .map(|value| value as u8)
                .ok_or_else(|| {
                    PyError::value_error(format!(
                        "non-hexadecimal number found in fromhex() arg at position {position}"
                    ))
                })
        };
        out.push(digit(index)? * 16 + digit(index + 1)?);
        index += 2;
    }
    Ok(out)
}

fn bytes_fromhex(runtime: &mut dyn PyRuntime, _class: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytes.fromhex", 1, 1)?;
    args.reject_keywords("bytes.fromhex")?;
    let value = parse_hex_bytes(runtime, args.positional()[0])?;
    runtime.new_bytes(value)
}

fn bytearray_fromhex(runtime: &mut dyn PyRuntime, _class: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("bytearray.fromhex", 1, 1)?;
    args.reject_keywords("bytearray.fromhex")?;
    let value = parse_hex_bytes(runtime, args.positional()[0])?;
    runtime.new_bytearray(value)
}

fn dict_copy(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("dict.copy", 0, 0)?;
    args.reject_keywords("dict.copy")?;
    let dict = receiver.cast::<PyDict>(runtime)?;
    runtime.dict_copy(dict)
}

/// A builtin iterator. The VM handles supported physical payloads before consulting their
/// `__iter__` slots, so this call cannot redispatch into itself.
pub(crate) fn slot_sequence_iter(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
) -> PyResult<Option<PyValue>> {
    runtime.payload_iterator(receiver)
}

/// A builtin type's physical length, shared with `len()` after user-slot dispatch.
pub(crate) fn slot_builtin_length(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
) -> PyResult<Option<PyValue>> {
    let Some(length) = runtime.physical_length(receiver)? else {
        return Ok(None);
    };
    let length =
        i64::try_from(length).map_err(|_| PyError::overflow_error("length is too large"))?;
    Ok(Some(Value::Int(length)))
}

/// `__getitem__` for builtin sequences whose indexing is implemented by the VM's physical
/// payload reader. The reader also handles slices and their resource charges.
pub(crate) fn slot_builtin_get_item(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    index: PyValue,
) -> PyResult<Option<PyValue>> {
    runtime.builtin_get_item(receiver, index).map(Some)
}

/// Membership for builtin payloads, preserving the VM's bounded search and element equality.
pub(crate) fn slot_builtin_contains(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    item: PyValue,
) -> PyResult<Option<PyValue>> {
    Ok(Some(Value::Bool(runtime.builtin_contains(receiver, item)?)))
}

/// The base implementation accepts only an empty format specifier and uses the type's string
/// protocol. Builtin number and string types replace it with their own format slot.
pub(crate) fn slot_object_format(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    spec: PyValue,
) -> PyResult<Option<PyValue>> {
    let OwnedPyString(spec) = spec.cast(runtime)?;
    if !spec.is_empty() {
        let name = runtime.type_name(&receiver)?;
        return Err(PyError::type_error(format!(
            "unsupported format string passed to {name}.__format__"
        )));
    }
    let rendered = runtime.display(&receiver)?;
    runtime.new_string(rendered).map(Some)
}

/// Format a builtin payload without redispatching through `__format__`.
pub(crate) fn slot_builtin_format(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    spec: PyValue,
) -> PyResult<Option<PyValue>> {
    let OwnedPyString(spec) = spec.cast(runtime)?;
    let rendered = runtime.builtin_format(&receiver, &spec)?;
    runtime.new_string(rendered).map(Some)
}

/// The builtin subscription slot preserves the origin and tuple of type arguments.
pub(crate) fn slot_generic_alias(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    item: PyValue,
) -> PyResult<Option<PyValue>> {
    runtime.new_generic_alias(receiver, item).map(Some)
}

pub(crate) fn slot_namespace_dict_get_item(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    key: PyValue,
) -> PyResult<Option<PyValue>> {
    let dict = receiver.cast::<PyDict>(runtime)?;
    match runtime.dict_get(dict, &key)? {
        Some(value) => Ok(Some(value)),
        None => Err(runtime.exception_with_args("KeyError", vec![key])),
    }
}

pub(crate) fn slot_namespace_dict_set_item(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    key: PyValue,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let dict = receiver.cast::<PyDict>(runtime)?;
    runtime.dict_insert(dict, key, value)?;
    Ok(Some(Value::None))
}

pub(crate) fn slot_namespace_dict_length(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
) -> PyResult<Option<PyValue>> {
    let entries = receiver.cast::<PyDict>(runtime)?.items(runtime)?;
    let length =
        i64::try_from(entries.len()).map_err(|_| PyError::overflow_error("dict is too large"))?;
    Ok(Some(Value::Int(length)))
}

pub(crate) fn slot_namespace_dict_contains(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    key: PyValue,
) -> PyResult<Option<PyValue>> {
    let dict = receiver.cast::<PyDict>(runtime)?;
    Ok(Some(Value::Bool(runtime.dict_get(dict, &key)?.is_some())))
}

pub(crate) fn slot_namespace_dict_iter(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
) -> PyResult<Option<PyValue>> {
    let keys = receiver
        .cast::<PyDict>(runtime)?
        .items(runtime)?
        .into_iter()
        .map(|(key, _)| key)
        .collect();
    Ok(Some(runtime.new_iterator(keys)?))
}

fn set_add(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    set_modify(runtime, receiver, args, SetOperation::Add)
}

fn set_remove(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    set_modify(runtime, receiver, args, SetOperation::Remove)
}

fn set_discard(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    set_modify(runtime, receiver, args, SetOperation::Discard)
}

enum SetOperation {
    Add,
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
    let name = match operation {
        SetOperation::Add => "add",
        SetOperation::Remove => "remove",
        SetOperation::Discard => "discard",
    };
    let set = mutable_set(runtime, receiver, name)?;
    let value = args.positional()[0];
    match operation {
        SetOperation::Add => {
            runtime.set_insert(set, value)?;
        }
        SetOperation::Remove => {
            if !runtime.set_remove(set, &value)? {
                return Err(runtime.exception_with_args("KeyError", vec![value]));
            }
        }
        SetOperation::Discard => {
            runtime.set_remove(set, &value)?;
        }
    }
    Ok(Value::None)
}

/// Cast the receiver of a mutating `set` method.
///
/// Method lookup never finds these methods on a `frozenset`, but the unbound form
/// `set.add(frozenset(), 1)` still reaches them and must fail as it does in CPython.
fn mutable_set(runtime: &mut dyn PyRuntime, receiver: PyValue, method: &str) -> PyResult<PySet> {
    let set = receiver.cast::<PySet>(runtime)?;
    if runtime.set_is_frozen(set)? {
        return Err(PyError::type_error(format!(
            "descriptor '{method}' for 'set' objects doesn't apply to a 'frozenset' object"
        )));
    }
    Ok(set)
}

/// `set.update(*iterables)`: add the items of every iterable.
fn set_update(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.reject_keywords("set.update")?;
    let set = mutable_set(runtime, receiver, "update")?;
    for source in args.positional() {
        // Snapshot each iterable before inserting so an iterator over the receiver itself does
        // not observe the insertions.
        for value in collect_values(runtime, *source)? {
            runtime.set_insert(set, value)?;
        }
    }
    Ok(Value::None)
}

fn set_init(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("set.__init__", 0, 1)?;
    args.reject_keywords("set.__init__")?;
    let items = args
        .positional()
        .first()
        .map(|source| collect_values(runtime, *source))
        .transpose()?
        .unwrap_or_default();
    let set = mutable_set(runtime, receiver, "__init__")?;
    runtime.replace_set_items(set, Vec::new())?;
    for item in items {
        runtime.set_insert(set, item)?;
    }
    Ok(Value::None)
}

/// Remove and return one member. CPython picks a member by hash position; shellsim's sets keep
/// insertion order, so this removes the oldest member.
fn set_pop(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("set.pop", 0, 0)?;
    args.reject_keywords("set.pop")?;
    let set = mutable_set(runtime, receiver, "pop")?;
    let Some(value) = runtime.set_first(set)? else {
        let message = runtime.new_string("pop from an empty set".into())?;
        return Err(runtime.exception_with_args("KeyError", vec![message]));
    };
    runtime.set_remove(set, &value)?;
    Ok(value)
}

fn set_clear(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("set.clear", 0, 0)?;
    args.reject_keywords("set.clear")?;
    let set = mutable_set(runtime, receiver, "clear")?;
    runtime.replace_set_items(set, Vec::new())?;
    Ok(Value::None)
}

/// `intersection(*iterables)`: members present in the receiver and in every iterable.
fn set_intersection(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.reject_keywords("set.intersection")?;
    let set = receiver.cast::<PySet>(runtime)?;
    let members = set.items(runtime)?;
    let members = filter_set_members(runtime, members, args.positional(), true)?;
    new_set_like(runtime, set, members)
}

/// `difference(*iterables)`: receiver members absent from every iterable.
fn set_difference(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.reject_keywords("set.difference")?;
    let set = receiver.cast::<PySet>(runtime)?;
    let members = set.items(runtime)?;
    let members = filter_set_members(runtime, members, args.positional(), false)?;
    new_set_like(runtime, set, members)
}

fn set_symmetric_difference(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    args.expect_positional("set.symmetric_difference", 1, 1)?;
    args.reject_keywords("set.symmetric_difference")?;
    let set = receiver.cast::<PySet>(runtime)?;
    let members = set.items(runtime)?;
    let members = symmetric_difference_members(runtime, members, args.positional()[0])?;
    new_set_like(runtime, set, members)
}

fn set_intersection_update(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    args.reject_keywords("set.intersection_update")?;
    let set = mutable_set(runtime, receiver, "intersection_update")?;
    let members = set.items(runtime)?;
    let members = filter_set_members(runtime, members, args.positional(), true)?;
    runtime.replace_set_items(set, members)?;
    Ok(Value::None)
}

fn set_difference_update(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    args.reject_keywords("set.difference_update")?;
    let set = mutable_set(runtime, receiver, "difference_update")?;
    let members = set.items(runtime)?;
    let members = filter_set_members(runtime, members, args.positional(), false)?;
    runtime.replace_set_items(set, members)?;
    Ok(Value::None)
}

fn set_symmetric_difference_update(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    args.expect_positional("set.symmetric_difference_update", 1, 1)?;
    args.reject_keywords("set.symmetric_difference_update")?;
    let set = mutable_set(runtime, receiver, "symmetric_difference_update")?;
    let members = set.items(runtime)?;
    let members = symmetric_difference_members(runtime, members, args.positional()[0])?;
    runtime.replace_set_items(set, members)?;
    Ok(Value::None)
}

/// `issubset(iterable)`: every receiver member is an item of the iterable.
fn set_issubset(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("set.issubset", 1, 1)?;
    args.reject_keywords("set.issubset")?;
    let members = receiver.cast::<PySet>(runtime)?.items(runtime)?;
    let operand = collect_values(runtime, args.positional()[0])?;
    let operand = runtime.new_frozen_set(operand)?;
    for member in &members {
        if !indexed_contains(runtime, operand, member)? {
            return Ok(Value::Bool(false));
        }
    }
    Ok(Value::Bool(true))
}

/// `issuperset(iterable)`: every item of the iterable is a receiver member. As in CPython, this
/// stops consuming the iterable at the first item that is not a member.
fn set_issuperset(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    set_all_items(runtime, receiver, args, "set.issuperset", true)
}

/// `isdisjoint(iterable)`: no item of the iterable is a receiver member. As in CPython, this
/// stops consuming the iterable at the first shared item.
fn set_isdisjoint(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    set_all_items(runtime, receiver, args, "set.isdisjoint", false)
}

/// Whether every item of the single iterable argument has receiver membership `expected`.
fn set_all_items(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    name: &str,
    expected: bool,
) -> PyResult {
    args.expect_positional(name, 1, 1)?;
    args.reject_keywords(name)?;
    let members = receiver.cast::<PySet>(runtime)?.value();
    let iterator = runtime.iterator(args.positional()[0])?;
    let mut all = true;
    let mut exhausted = false;
    while all && !exhausted {
        runtime.nested(&mut |runtime| {
            let Some(item) = runtime.iterator_next(iterator)? else {
                exhausted = true;
                return Ok(());
            };
            all = indexed_contains(runtime, members, &item)? == expected;
            Ok(())
        })?;
    }
    Ok(Value::Bool(all))
}

/// Keep the members whose presence in each iterable operand equals `keep_present`.
///
/// `keep_present` selects intersection (`true`) or difference (`false`). Every operand is
/// consumed even after `members` becomes empty, so a non-iterable operand raises `TypeError` as
/// it does in CPython.
fn filter_set_members(
    runtime: &mut dyn PyRuntime,
    mut members: Vec<PyValue>,
    operands: &[PyValue],
    keep_present: bool,
) -> PyResult<Vec<PyValue>> {
    for operand in operands {
        let operand = collect_values(runtime, *operand)?;
        let operand = runtime.new_frozen_set(operand)?;
        let mut kept = Vec::new();
        for member in members {
            if indexed_contains(runtime, operand, &member)? == keep_present {
                runtime.reserve_memory(std::mem::size_of::<PyValue>())?;
                kept.push(member);
            }
        }
        members = kept;
    }
    Ok(members)
}

/// Members of exactly one of `members` and the iterable `operand`: receiver members first, then
/// the operand's remaining items without duplicates.
fn symmetric_difference_members(
    runtime: &mut dyn PyRuntime,
    members: Vec<PyValue>,
    operand: PyValue,
) -> PyResult<Vec<PyValue>> {
    let operand = collect_values(runtime, operand)?;
    let operand_index = runtime.new_frozen_set(operand.clone())?;
    let member_index = runtime.new_frozen_set(members.clone())?;
    // Operand items already added, so a repeated operand item is kept once.
    let added = runtime.new_set(Vec::new())?.cast::<PySet>(runtime)?;
    let mut result = Vec::new();
    for member in &members {
        if !indexed_contains(runtime, operand_index, member)? {
            runtime.reserve_memory(std::mem::size_of::<PyValue>())?;
            result.push(*member);
        }
    }
    for item in operand {
        if !indexed_contains(runtime, member_index, &item)? && runtime.set_insert(added, item)? {
            runtime.reserve_memory(std::mem::size_of::<PyValue>())?;
            result.push(item);
        }
    }
    Ok(result)
}

/// Allocate a result of the same builtin type, `set` or `frozenset`, as `like`.
fn new_set_like(runtime: &mut dyn PyRuntime, like: PySet, members: Vec<PyValue>) -> PyResult {
    if runtime.set_is_frozen(like)? {
        runtime.new_frozen_set(members)
    } else {
        runtime.new_set(members)
    }
}

fn set_union(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.reject_keywords("set.union")?;
    let receiver = receiver.cast::<PySet>(runtime)?;
    let frozen = runtime.set_is_frozen(receiver)?;
    let members = receiver.items(runtime)?;
    let union = runtime.new_set(members)?;
    let union_set = union.cast::<PySet>(runtime)?;
    for source in args.positional() {
        let iterator = runtime.iterator(*source)?;
        let mut exhausted = false;
        while !exhausted {
            runtime.nested(&mut |runtime| {
                let Some(value) = runtime.iterator_next(iterator)? else {
                    exhausted = true;
                    return Ok(());
                };
                runtime.charge_cpu(1)?;
                runtime.set_insert(union_set, value)?;
                Ok(())
            })?;
        }
    }
    if frozen {
        let members = union_set.items(runtime)?;
        runtime.new_frozen_set(members)
    } else {
        Ok(union)
    }
}

fn set_copy(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("set.copy", 0, 0)?;
    args.reject_keywords("set.copy")?;
    let receiver = receiver.cast::<PySet>(runtime)?;
    let frozen = runtime.set_is_frozen(receiver)?;
    let values = receiver.items(runtime)?;
    if frozen {
        runtime.new_frozen_set(values)
    } else {
        runtime.new_set(values)
    }
}

/// Whether `left`'s members are all in `right`, and whether `left` is smaller; `None` when
/// `right` is not a set, so the comparison declines and a set-like right operand, such as a
/// dict keys view, can answer through its reflected slot.
fn set_is_subset(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<(bool, bool)>> {
    let left_set = left.cast::<PySet>(runtime)?;
    let left = left_set.items(runtime)?;
    let Ok(right_set) = right.cast::<PySet>(runtime) else {
        return Ok(None);
    };
    let right_len = right_set.items(runtime)?.len();
    let mut subset = true;
    // Each member is probed by hash, so the test is linear in the smaller set.
    for value in &left {
        runtime.charge_cpu(1)?;
        if !runtime.builtin_contains(right, *value)? {
            subset = false;
            break;
        }
    }
    Ok(Some((subset, left.len() < right_len)))
}

/// Whether an ordering of two values satisfies `operator`.
fn ordering_accepts(operator: ComparisonOperator, ordering: Ordering) -> bool {
    match operator {
        ComparisonOperator::Equal => ordering.is_eq(),
        ComparisonOperator::NotEqual => ordering.is_ne(),
        ComparisonOperator::Less => ordering.is_lt(),
        ComparisonOperator::LessEqual => ordering.is_le(),
        ComparisonOperator::Greater => ordering.is_gt(),
        ComparisonOperator::GreaterEqual => ordering.is_ge(),
        ComparisonOperator::In
        | ComparisonOperator::NotIn
        | ComparisonOperator::Is
        | ComparisonOperator::IsNot => false,
    }
}

/// The comparison slot of `str`: strings order by code point, which is UTF-8 byte order, and
/// decline every other operand. Exact strings take the VM's direct path before any slot; this
/// serves `str` subclasses and mixed operands.
pub(crate) fn slot_str_compare(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    operator: ComparisonOperator,
) -> PyResult<Option<bool>> {
    let (Some(left), Some(right)) = (runtime.string_value(&left)?, runtime.string_value(&right)?)
    else {
        return Ok(None);
    };
    runtime.charge_cpu(super::super::vm::scan_cost(left.len().min(right.len())))?;
    Ok(Some(ordering_accepts(operator, left.cmp(&right))))
}

/// The comparison slot of `bytes` and `bytearray`, which compare with each other by value.
pub(crate) fn slot_bytes_compare(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    operator: ComparisonOperator,
) -> PyResult<Option<bool>> {
    let (Some(left), Some(right)) = (runtime.bytes_value(&left)?, runtime.bytes_value(&right)?)
    else {
        return Ok(None);
    };
    runtime.charge_cpu(super::super::vm::scan_cost(left.len().min(right.len())))?;
    Ok(Some(ordering_accepts(operator, left.cmp(&right))))
}

/// The comparison slot of `range`, `slice` and generic aliases: equal to another value of the
/// same type when their parts are, and unordered.
pub(crate) fn slot_structural_compare(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    operator: ComparisonOperator,
) -> PyResult<Option<bool>> {
    if !runtime.identical(&runtime.class_of(&left)?, &runtime.class_of(&right)?) {
        return Ok(None);
    }
    match operator {
        ComparisonOperator::Equal => runtime.physical_equals(&left, &right).map(Some),
        ComparisonOperator::NotEqual => runtime.physical_equals(&left, &right).map(|eq| Some(!eq)),
        _ => Ok(None),
    }
}

/// The comparison slot of `list`, `tuple` and `dict`: element by element through the runtime,
/// so a user `__eq__` or `__lt__` holds inside a container.
pub(crate) fn slot_container_compare(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    operator: ComparisonOperator,
) -> PyResult<Option<bool>> {
    runtime.container_compare(operator, &left, &right)
}

/// The comparison slot of `set` and `frozenset`: equality element by element, and the subset
/// relations for the ordering operators.
pub(crate) fn slot_set_compare(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    operator: ComparisonOperator,
) -> PyResult<Option<bool>> {
    if right.cast::<PySet>(runtime).is_err() {
        return Ok(None);
    }
    let relation = match operator {
        ComparisonOperator::Equal | ComparisonOperator::NotEqual => {
            return runtime.container_compare(operator, &left, &right)
        }
        ComparisonOperator::Less | ComparisonOperator::LessEqual => {
            set_is_subset(runtime, left, right)?
        }
        ComparisonOperator::Greater | ComparisonOperator::GreaterEqual => {
            set_is_subset(runtime, right, left)?
        }
        ComparisonOperator::In
        | ComparisonOperator::NotIn
        | ComparisonOperator::Is
        | ComparisonOperator::IsNot => return Ok(None),
    };
    Ok(relation.map(|(subset, strict)| match operator {
        ComparisonOperator::Less | ComparisonOperator::Greater => subset && strict,
        _ => subset,
    }))
}

pub(crate) fn slot_set_subtract(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    set_binary(runtime, left, right, SetBinaryOperation::Difference)
}

pub(crate) fn slot_set_intersection(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    set_binary(runtime, left, right, SetBinaryOperation::Intersection)
}

pub(crate) fn slot_set_symmetric_difference(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    set_binary(
        runtime,
        left,
        right,
        SetBinaryOperation::SymmetricDifference,
    )
}

pub(crate) fn slot_set_union(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    set_binary(runtime, left, right, SetBinaryOperation::Union)
}

#[derive(Clone, Copy)]
enum SetBinaryOperation {
    Difference,
    Intersection,
    SymmetricDifference,
    Union,
}

fn set_binary(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    operation: SetBinaryOperation,
) -> PyResult<Option<PyValue>> {
    let left_set = left.cast::<PySet>(runtime)?;
    let frozen = runtime.set_is_frozen(left_set)?;
    let left_members = left_set.items(runtime)?;
    let Ok(right_set) = right.cast::<PySet>(runtime) else {
        return Ok(None);
    };
    let right_members = right_set.items(runtime)?;
    let left = left_set.value();
    let right = right_set.value();
    let mut result = Vec::new();
    for value in &left_members {
        let present = indexed_contains(runtime, right, value)?;
        if matches!(operation, SetBinaryOperation::Union)
            || present == matches!(operation, SetBinaryOperation::Intersection)
        {
            runtime.reserve_memory(64)?;
            result.push(*value);
        }
    }
    if matches!(
        operation,
        SetBinaryOperation::Union | SetBinaryOperation::SymmetricDifference
    ) {
        for value in &right_members {
            if !indexed_contains(runtime, left, value)? {
                runtime.reserve_memory(64)?;
                result.push(*value);
            }
        }
    }
    if frozen {
        runtime.new_frozen_set(result).map(Some)
    } else {
        runtime.new_set(result).map(Some)
    }
}

/// Membership of `value` in the builtin set or frozenset `set`, through its hash index rather
/// than any `__contains__` override.
fn indexed_contains(runtime: &mut dyn PyRuntime, set: PyValue, value: &PyValue) -> PyResult<bool> {
    runtime.charge_cpu(1)?;
    runtime.builtin_contains(set, *value)
}

fn builtin_map(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("map", 2, usize::MAX)?;
    args.reject_keywords("map")?;
    let function = args.positional()[0].cast::<PyCallable>(runtime)?;
    let mut iterators = Vec::new();
    for value in &args.positional()[1..] {
        iterators.push(runtime.iterator(*value)?);
    }
    // Each step runs in its own pin scope; the list keeps the mapped results alive.
    let result = runtime.new_list(Vec::new())?.cast::<PyList>(runtime)?;
    let mut exhausted = false;
    while !exhausted {
        runtime.nested(&mut |runtime| {
            let mut values = Vec::with_capacity(iterators.len());
            for iterator in &iterators {
                let Some(value) = runtime.iterator_next(*iterator)? else {
                    exhausted = true;
                    return Ok(());
                };
                values.push(value);
            }
            let mapped = function.call(runtime, CallArgs::new(values, Vec::new()))?;
            runtime.list_append(result, mapped)
        })?;
    }
    let result = runtime.list_items(result)?;
    runtime.new_iterator(result)
}

fn builtin_filter(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("filter", 2, 2)?;
    args.reject_keywords("filter")?;
    let predicate = (runtime.kind(&args.positional()[0])? != PyKind::None)
        .then(|| args.positional()[0].cast::<PyCallable>(runtime))
        .transpose()?;
    let iterator = runtime.iterator(args.positional()[1])?;
    // Each step runs in its own pin scope; the list keeps the selected items alive.
    let result = runtime.new_list(Vec::new())?.cast::<PyList>(runtime)?;
    let mut exhausted = false;
    while !exhausted {
        runtime.nested(&mut |runtime| {
            let Some(value) = runtime.iterator_next(iterator)? else {
                exhausted = true;
                return Ok(());
            };
            let selected = match &predicate {
                Some(predicate) => {
                    let result = predicate.call(runtime, CallArgs::new(vec![value], Vec::new()))?;
                    runtime.truth(&result)?
                }
                None => runtime.truth(&value)?,
            };
            if selected {
                runtime.list_append(result, value)?;
            }
            Ok(())
        })?;
    }
    let result = runtime.list_items(result)?;
    runtime.new_iterator(result)
}

fn builtin_reversed(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("reversed", 1, 1)?;
    args.reject_keywords("reversed")?;
    runtime.reverse_value(args.positional()[0])
}

pub(crate) fn slot_sequence_reversed(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
) -> PyResult<Option<PyValue>> {
    runtime.reverse_builtin_sequence(receiver).map(Some)
}

pub(crate) fn slot_dict_reversed(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
) -> PyResult<Option<PyValue>> {
    let entries = runtime
        .mapping_items(receiver)?
        .ok_or_else(|| PyError::type_error("descriptor requires a dict"))?;
    let mut keys = entries.into_iter().map(|(key, _)| key).collect::<Vec<_>>();
    runtime.charge_cpu(u64::try_from(keys.len()).unwrap_or(u64::MAX))?;
    keys.reverse();
    runtime.new_iterator(keys).map(Some)
}

fn builtin_getattr(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("getattr", 2, 3)?;
    args.reject_keywords("getattr")?;
    let OwnedPyString(name) = args.positional()[1].cast(runtime)?;
    match runtime.get_attribute(args.positional()[0], &name)? {
        Some(value) => Ok(value),
        None => args.positional().get(2).copied().ok_or_else(|| {
            PyError::exception("AttributeError", format!("attribute {name:?} not found"))
        }),
    }
}

/// `id()`: an integer that two values share exactly when `is` holds between them.
///
/// A heap object's id is the stand-in address its default repr shows, so `hex(id(x))` matches
/// `<object object at 0x...>`. Immediate values such as small ints, floats and short strings
/// are identical when their contents are, so their id is a deterministic hash of the value,
/// placed below the heap addresses.
fn builtin_id(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("id", 1, 1)?;
    args.reject_keywords("id")?;
    let value = args.positional()[0];
    let id = match runtime.identity(&value) {
        Some(identity) => protocol::address_value(identity.0),
        None => immediate_identity(&value),
    };
    Ok(Value::Int(i64::try_from(id).expect("ids are below 2**47")))
}

/// The stand-in address of an immediate value, below the heap addresses. Equal immediates are
/// the same object, so this is a deterministic hash of the value.
pub(crate) fn immediate_identity(value: &PyValue) -> u64 {
    use std::hash::{Hash, Hasher};
    const IMMEDIATE_BASE: u64 = 0x5000_0000_0000;
    let mut hasher = std::hash::DefaultHasher::new();
    // Values have no `Hash`; hash each immediate kind with a distinct tag so that values of
    // different kinds with equal payloads (such as `True` and `1`) get different ids.
    if let Some(flag) = value.bool_value() {
        (0_u8, flag).hash(&mut hasher);
    } else if let Some(integer) = value.immediate_int() {
        (1_u8, integer).hash(&mut hasher);
    } else if let Some(float) = value.float_value() {
        (2_u8, float.to_bits()).hash(&mut hasher);
    } else if value.is_none() {
        3_u8.hash(&mut hasher);
    } else if let Some(text) = value.inline_string_ref() {
        (4_u8, text.as_str()).hash(&mut hasher);
    } else if let Some(native) = value.native_value() {
        (5_u8, native.encode()).hash(&mut hasher);
    } else if let Some(parts) = value.registered_parts() {
        (6_u8, parts).hash(&mut hasher);
    }
    IMMEDIATE_BASE + (hasher.finish() % (1 << 40)) * 16
}

/// `ascii()`: `repr()` with non-ASCII characters escaped as `\\x`, `\\u` or `\\U`.
fn builtin_ascii(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("ascii", 1, 1)?;
    args.reject_keywords("ascii")?;
    let text = runtime.repr(&args.positional()[0])?;
    runtime.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
    let mut output = String::with_capacity(text.len());
    for character in text.chars() {
        let code = u32::from(character);
        if character.is_ascii() {
            output.push(character);
        } else if code <= 0xff {
            output.push_str(&format!("\\x{code:02x}"));
        } else if code <= 0xffff {
            output.push_str(&format!("\\u{code:04x}"));
        } else {
            output.push_str(&format!("\\U{code:08x}"));
        }
    }
    runtime.new_string(output)
}

fn builtin_hasattr(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("hasattr", 2, 2)?;
    args.reject_keywords("hasattr")?;
    let OwnedPyString(name) = args.positional()[1].cast(runtime)?;
    Ok(Value::Bool(
        runtime
            .get_attribute(args.positional()[0], &name)?
            .is_some(),
    ))
}

fn builtin_round(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("round", 1, 2)?;
    args.reject_unknown_keywords("round", &["ndigits"])?;
    let keyword_digits = args.keyword("round", "ndigits")?;
    if args.positional().len() == 2 && keyword_digits.is_some() {
        return Err(PyError::type_error(
            "round() got multiple values for argument 'ndigits'",
        ));
    }
    let digits = match args.positional().get(1).or(keyword_digits) {
        Some(value) if runtime.kind(value)? != PyKind::None => Some(integer_argument(
            runtime,
            value,
            "ndigits must be an integer",
        )?),
        Some(_) | None => None,
    };
    let value = args.positional()[0];
    if runtime.value_kind_of(&value).is_some()
        || matches!(runtime.kind(&value)?, PyKind::Instance | PyKind::Complex)
    {
        // Like CPython, defer to the type's `__round__`, so a NumPy scalar keeps its dtype and
        // a user class such as `Fraction` rounds itself.
        let Some(method) = runtime.get_attribute(value, "__round__")? else {
            return Err(PyError::type_error(format!(
                "type {} doesn't define __round__ method",
                runtime.type_name(&value)?
            )));
        };
        let arguments = args.positional().get(1).or(keyword_digits).copied();
        return runtime.call_value(
            method,
            CallArgs::new(arguments.into_iter().collect(), Vec::new()),
        );
    }
    match value.cast::<PyNumber>(runtime)? {
        PyNumber::Int(value) => round_integer(runtime, BigInt::from(value), digits),
        PyNumber::BigInt(value) => round_integer(runtime, value, digits),
        PyNumber::Float(value) => round_float(runtime, value, digits),
    }
}

#[derive(Clone, Copy)]
enum IntegerArgument {
    Finite(i64),
    TooPositive,
    TooNegative,
}

fn integer_argument(
    runtime: &dyn PyRuntime,
    value: &PyValue,
    message: &str,
) -> PyResult<IntegerArgument> {
    let integer = runtime
        .integer_bigint(value)?
        .ok_or_else(|| PyError::type_error(message))?;
    Ok(match integer.to_i64() {
        Some(value) => IntegerArgument::Finite(value),
        None if integer.is_negative() => IntegerArgument::TooNegative,
        None => IntegerArgument::TooPositive,
    })
}

fn round_integer(
    runtime: &mut dyn PyRuntime,
    value: BigInt,
    digits: Option<IntegerArgument>,
) -> PyResult {
    let negative_digits = match digits {
        None | Some(IntegerArgument::TooPositive | IntegerArgument::Finite(0..)) => {
            return runtime.new_integer(&value.to_string());
        }
        Some(IntegerArgument::TooNegative) => return runtime.new_integer("0"),
        Some(IntegerArgument::Finite(value)) => value.unsigned_abs(),
    };
    // Rounding at or beyond the leading digit is still exact below, so an upper estimate of the
    // digit count only skips work.
    let decimal_digits = super::super::number::decimal_digits(&value) as u64;
    if negative_digits > decimal_digits {
        return runtime.new_integer("0");
    }
    let exponent = u32::try_from(negative_digits)
        .map_err(|_| PyError::resource_error("rounding precision is too large"))?;
    runtime.charge_cpu(negative_digits)?;
    let divisor = BigInt::from(10_u8).pow(exponent);
    let mut quotient = &value / &divisor;
    let remainder = (&value % &divisor).abs();
    let twice_remainder = remainder * 2_u8;
    if twice_remainder > divisor
        || (twice_remainder == divisor && (&quotient % 2_u8) != BigInt::zero())
    {
        quotient += if value.sign() == Sign::Minus { -1 } else { 1 };
    }
    runtime.new_bigint(quotient * divisor)
}

fn round_float(
    runtime: &mut dyn PyRuntime,
    value: f64,
    digits: Option<IntegerArgument>,
) -> PyResult {
    let Some(digits) = digits else {
        if value.is_nan() {
            return Err(PyError::value_error("cannot convert float NaN to integer"));
        }
        if value.is_infinite() {
            return Err(PyError::overflow_error(
                "cannot convert float infinity to integer",
            ));
        }
        return runtime.new_integer(&format!("{:.0}", value.round_ties_even()));
    };
    if !value.is_finite() {
        return Ok(Value::Float(value));
    }
    let rounded = match digits {
        IntegerArgument::TooPositive => value,
        IntegerArgument::TooNegative => value.signum() * 0.0,
        IntegerArgument::Finite(digits) if digits > 308 => value,
        IntegerArgument::Finite(digits) if digits < -308 => value.signum() * 0.0,
        IntegerArgument::Finite(digits) if digits >= 0 => {
            let precision = usize::try_from(digits).expect("nonnegative precision is bounded");
            runtime.reserve_memory(precision.saturating_add(320))?;
            runtime.charge_cpu(u64::try_from(precision).unwrap_or(u64::MAX))?;
            format!("{value:.precision$}")
                .parse::<f64>()
                .expect("formatted finite float remains a float")
        }
        IntegerArgument::Finite(digits) => {
            let scale = 10_f64.powi(i32::try_from(-digits).expect("precision is bounded"));
            (value / scale).round_ties_even() * scale
        }
    };
    Ok(Value::Float(rounded))
}

fn property_setter(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("property.setter", 1, 1)?;
    args.reject_keywords("property.setter")?;
    let property = receiver.cast::<PyProperty>(runtime)?;
    let getter = runtime.property_getter(property)?;
    runtime.new_property(getter, Some(args.positional()[0]))
}

/// `object.__init__`, which ends an initializer chain reached through `super().__init__()`.
/// It accepts only the instance.
/// `object.__new__(cls)`: the receiver is the class, since `__new__` is a static method.
fn object_new(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let has_arguments = !args.positional().is_empty() || !args.keywords().is_empty();
    runtime.new_instance(receiver, has_arguments)
}

fn object_init(_runtime: &mut dyn PyRuntime, _receiver: PyValue, args: CallArgs) -> PyResult {
    if !args.positional().is_empty() || !args.keywords().is_empty() {
        return Err(PyError::type_error(
            "object.__init__() takes exactly one argument (the instance to initialize)",
        ));
    }
    Ok(Value::None)
}

/// End a cooperative class-initialization chain after every parent has had its turn.
fn object_init_subclass(
    _runtime: &mut dyn PyRuntime,
    _receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    args.expect_positional("object.__init_subclass__", 0, 0)?;
    args.reject_keywords("object.__init_subclass__")?;
    Ok(Value::None)
}

/// The base representation uses identity even when a subclass overrides `__repr__`.
fn object_repr(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("object.__repr__", 0, 0)?;
    args.reject_keywords("object.__repr__")?;
    let rendered = runtime.default_object_repr(&receiver)?;
    runtime.new_string(rendered)
}

fn object_str(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("object.__str__", 0, 0)?;
    args.reject_keywords("object.__str__")?;
    let rendered = runtime.repr(&receiver)?;
    runtime.new_string(rendered)
}

/// `object.__hash__`: the identity hash an instance has unless its class overrides it, so a
/// class that defines `__eq__` can keep it with `__hash__ = object.__hash__`.
fn object_hash(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("object.__hash__", 0, 0)?;
    args.reject_keywords("object.__hash__")?;
    let identity = match runtime.identity(&receiver) {
        Some(identity) => u64::from(identity.0),
        None => immediate_identity(&receiver),
    };
    Ok(Value::Int(super::super::hash::identity(identity)))
}

/// Call the default descriptor algorithm without the outer `__getattr__` fallback.
fn object_getattribute(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("object.__getattribute__", 1, 1)?;
    args.reject_keywords("object.__getattribute__")?;
    let name = runtime
        .string_value(&args.positional()[0])?
        .ok_or_else(|| PyError::type_error("attribute name must be a string"))?;
    runtime
        .get_attribute_default(receiver, &name)?
        .ok_or_else(|| {
            PyError::exception("AttributeError", format!("attribute {name:?} not found"))
        })
}

/// `object.__delattr__(name)`: the default deletion, which a class's own `__delattr__` calls.
fn object_delattr(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.reject_keywords("object.__delattr__")?;
    let [name] = args.positional() else {
        return Err(PyError::type_error(format!(
            "expected 1 argument, got {}",
            args.positional().len()
        )));
    };
    let Some(name) = runtime.string_value(name)? else {
        return Err(PyError::type_error(format!(
            "attribute name must be string, not '{}'",
            runtime.type_name(name)?
        )));
    };
    runtime.delete_attribute_default(receiver, &name)?;
    Ok(Value::None)
}

/// `object.__setattr__(name, value)`: the default assignment, which a class's own
/// `__setattr__` calls to store a value after checking or transforming it.
fn object_setattr(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.reject_keywords("object.__setattr__")?;
    let [name, value] = args.positional() else {
        return Err(PyError::type_error(format!(
            "expected 2 arguments, got {}",
            args.positional().len()
        )));
    };
    let Some(name) = runtime.string_value(name)? else {
        return Err(PyError::type_error(format!(
            "attribute name must be string, not '{}'",
            runtime.type_name(name)?
        )));
    };
    runtime.set_attribute_default(receiver, &name, *value)?;
    Ok(Value::None)
}

/// `iterator.__iter__`: a builtin iterator is its own iterator.
fn iterator_iter(_runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("__iter__", 0, 0)?;
    args.reject_keywords("__iter__")?;
    Ok(receiver)
}

/// `iterator.__next__`: the next item of a builtin iterator, or `StopIteration` at the end, so
/// `it.__next__` works wherever `next(it)` does.
fn iterator_next(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("__next__", 0, 0)?;
    args.reject_keywords("__next__")?;
    let iterator = runtime.iterator(receiver)?;
    runtime
        .iterator_next(iterator)?
        .ok_or_else(|| PyError::exception("StopIteration", ""))
}

/// `object.__eq__`: an object equals itself, and any other comparison is left to the other
/// operand by returning `NotImplemented`.
fn object_eq(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("object.__eq__", 1, 1)?;
    args.reject_keywords("object.__eq__")?;
    if runtime.identical(&receiver, &args.positional()[0]) {
        return Ok(Value::Bool(true));
    }
    Ok(runtime.not_implemented())
}

/// `object.__ne__`: the inverse of the receiver's `__eq__`, or `NotImplemented` when that
/// declines.
fn object_ne(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("object.__ne__", 1, 1)?;
    args.reject_keywords("object.__ne__")?;
    let other = args.positional()[0];
    let equal = match runtime.get_attribute(receiver, "__eq__")? {
        Some(method) => runtime.call_value(method, CallArgs::new(vec![other], Vec::new()))?,
        None => object_eq(runtime, receiver, CallArgs::new(vec![other], Vec::new()))?,
    };
    if runtime.is_not_implemented(&equal) {
        return Ok(equal);
    }
    Ok(Value::Bool(!runtime.truth(&equal)?))
}

/// `BaseException.__init__`, which replaces the exception's `args` with its positional
/// arguments.
fn exception_init(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    if !args.keywords().is_empty() {
        let type_name = runtime.type_name(&receiver)?;
        return Err(PyError::type_error(format!(
            "{type_name}() takes no keyword arguments"
        )));
    }
    let items = runtime.new_tuple(args.positional().to_vec())?;
    runtime.set_attribute(receiver, "args", items)?;
    Ok(Value::None)
}

fn type_call(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    runtime.call_type_default(receiver, args)
}

fn type_new(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("type.__new__", 3, 3)?;
    args.reject_keywords("type.__new__")?;
    let OwnedPyString(name) = args.positional()[0].cast(runtime)?;
    runtime.new_type(receiver, name, args.positional()[1], args.positional()[2])
}

/// `cls.mro()`: the method resolution order as a list.
fn type_mro(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("type.mro", 0, 0)?;
    args.reject_keywords("type.mro")?;
    let order = runtime
        .get_attribute(receiver, "__mro__")?
        .ok_or_else(|| PyError::type_error("descriptor 'mro' requires a type"))?;
    let items = order.cast::<PyTuple>(runtime)?.items(runtime)?;
    runtime.new_list(items)
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

pub(crate) fn slot_string_hash(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let Some(text) = runtime.string_value(&value)? else {
        return Ok(None);
    };
    runtime.charge_cpu(u64::try_from(text.len() / 32).unwrap_or(u64::MAX))?;
    Ok(Some(Value::Int(super::super::hash::string(&text))))
}

pub(crate) fn slot_none_bool(
    _runtime: &mut dyn PyRuntime,
    _value: PyValue,
) -> PyResult<Option<PyValue>> {
    Ok(Some(Value::Bool(false)))
}

pub(crate) fn slot_none_hash(
    _runtime: &mut dyn PyRuntime,
    _value: PyValue,
) -> PyResult<Option<PyValue>> {
    Ok(Some(Value::Int(super::super::hash::NONE)))
}

pub(crate) fn slot_string_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let (Some(left), Some(right)) = (runtime.string_value(&left)?, runtime.string_value(&right)?)
    else {
        return Ok(None);
    };
    Ok(Some(Value::Bool(left == right)))
}

pub(crate) fn slot_string_not_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let (Some(left), Some(right)) = (runtime.string_value(&left)?, runtime.string_value(&right)?)
    else {
        return Ok(None);
    };
    Ok(Some(Value::Bool(left != right)))
}

fn slot_string_order(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    accepted: &[Ordering],
) -> PyResult<Option<PyValue>> {
    let (Some(left), Some(right)) = (runtime.string_value(&left)?, runtime.string_value(&right)?)
    else {
        return Ok(None);
    };
    Ok(Some(Value::Bool(accepted.contains(&left.cmp(&right)))))
}

pub(crate) fn slot_string_less(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_string_order(runtime, left, right, &[Ordering::Less])
}

pub(crate) fn slot_string_less_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_string_order(runtime, left, right, &[Ordering::Less, Ordering::Equal])
}

pub(crate) fn slot_string_greater(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_string_order(runtime, left, right, &[Ordering::Greater])
}

pub(crate) fn slot_string_greater_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_string_order(runtime, left, right, &[Ordering::Greater, Ordering::Equal])
}

pub(crate) fn slot_bytes_hash(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let Some(bytes) = runtime.bytes_value(&value)? else {
        return Ok(None);
    };
    runtime.charge_cpu(u64::try_from(bytes.len() / 32).unwrap_or(u64::MAX))?;
    Ok(Some(Value::Int(super::super::hash::bytes(&bytes))))
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
    let mut items = runtime.bytearray_items(array)?;
    if let Some(index) = runtime.int_value(&index) {
        let value = runtime
            .int_value(&value)
            .and_then(|value| u8::try_from(value).ok())
            .ok_or_else(|| PyError::value_error("byte must be in range(0, 256)"))?;
        let index = byte_index(index, items.len())?;
        items[index] = value;
    } else if let Some((start, stop, step)) = runtime.slice_parts(&index)? {
        let replacement = collect_bytes(runtime, value)?;
        let plan = SlicePlan::new(items.len(), start, stop, step).map_err(PyError::value_error)?;
        assign_slice(runtime, &mut items, plan, replacement)?;
    } else {
        return Ok(None);
    }
    runtime.replace_bytearray_items(array, items)?;
    Ok(Some(PyValue::None))
}

pub(crate) fn slot_bytearray_delete_item(
    runtime: &mut dyn PyRuntime,
    owner: PyValue,
    index: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = owner.cast::<PyByteArray>(runtime)?;
    let mut items = runtime.bytearray_items(array)?;
    if let Some(index) = runtime.int_value(&index) {
        let index = byte_index(index, items.len())?;
        items.remove(index);
    } else if let Some((start, stop, step)) = runtime.slice_parts(&index)? {
        let plan = SlicePlan::new(items.len(), start, stop, step).map_err(PyError::value_error)?;
        delete_slice(runtime, &mut items, plan)?;
    } else {
        return Ok(None);
    }
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
    let mut items = runtime.list_items(list)?;
    if let Some(index) = runtime.int_value(&index) {
        let index = normalized_list_index(index, items.len())?;
        items.remove(index);
    } else if let Some((start, stop, step)) = runtime.slice_parts(&index)? {
        let plan = SlicePlan::new(items.len(), start, stop, step).map_err(PyError::value_error)?;
        delete_slice(runtime, &mut items, plan)?;
    } else {
        return Ok(None);
    }
    runtime.replace_list_items(list, items)?;
    Ok(Some(PyValue::None))
}

pub(crate) fn slot_list_set_item(
    runtime: &mut dyn PyRuntime,
    owner: PyValue,
    index: PyValue,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let list = owner.cast::<PyList>(runtime)?;
    let mut items = runtime.list_items(list)?;
    if let Some(index) = runtime.int_value(&index) {
        let index = normalized_list_index(index, items.len())?;
        items[index] = value;
    } else if let Some((start, stop, step)) = runtime.slice_parts(&index)? {
        let replacement = collect_values(runtime, value)?;
        let plan = SlicePlan::new(items.len(), start, stop, step).map_err(PyError::value_error)?;
        assign_slice(runtime, &mut items, plan, replacement)?;
    } else {
        return Ok(None);
    }
    runtime.replace_list_items(list, items)?;
    Ok(Some(PyValue::None))
}

fn normalized_list_index(index: i64, length: usize) -> PyResult<usize> {
    let length = i64::try_from(length).unwrap_or(i64::MAX);
    let index = if index < 0 {
        index.checked_add(length)
    } else {
        Some(index)
    }
    .and_then(|index| usize::try_from(index).ok())
    .filter(|index| *index < usize::try_from(length).unwrap_or(usize::MAX))
    .ok_or_else(|| PyError::exception("IndexError", "list assignment index out of range"))?;
    Ok(index)
}

fn assign_slice<T>(
    runtime: &mut dyn PyRuntime,
    items: &mut Vec<T>,
    plan: SlicePlan,
    replacement: Vec<T>,
) -> PyResult<()> {
    if let Some(range) = plan.contiguous_range() {
        let final_length = items
            .len()
            .checked_sub(range.len())
            .and_then(|length| length.checked_add(replacement.len()))
            .ok_or_else(|| PyError::resource_error("slice result is too large"))?;
        let growth = final_length
            .saturating_sub(items.len())
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| PyError::resource_error("slice result is too large"))?;
        runtime.reserve_memory(growth)?;
        runtime.charge_cpu(
            u64::try_from(items.len().saturating_add(replacement.len())).unwrap_or(u64::MAX),
        )?;
        items.splice(range, replacement);
        return Ok(());
    }
    if replacement.len() != plan.len() {
        return Err(PyError::value_error(format!(
            "attempt to assign sequence of size {} to extended slice of size {}",
            replacement.len(),
            plan.len()
        )));
    }
    runtime.charge_cpu(u64::try_from(plan.len()).unwrap_or(u64::MAX))?;
    for (index, value) in plan.indices().zip(replacement) {
        items[index] = value;
    }
    Ok(())
}

fn delete_slice<T>(
    runtime: &mut dyn PyRuntime,
    items: &mut Vec<T>,
    plan: SlicePlan,
) -> PyResult<()> {
    runtime.charge_cpu(u64::try_from(items.len()).unwrap_or(u64::MAX))?;
    if let Some(range) = plan.contiguous_range() {
        items.drain(range);
        return Ok(());
    }
    runtime.reserve_memory(items.len())?;
    let mut deleted = vec![false; items.len()];
    for index in plan.indices() {
        deleted[index] = true;
    }
    let mut index = 0;
    items.retain(|_| {
        let keep = !deleted[index];
        index += 1;
        keep
    });
    Ok(())
}

/// `left | right` for two dicts: a new dict holding `left`'s entries updated by `right`'s.
/// Anything but a dict (or dict subclass) on either side declines, so `{} | [1]` raises
/// `TypeError`.
pub(crate) fn slot_dict_union(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    dict_union(runtime, left, right)
}

/// `left | right` reached through the right operand's dict type.
pub(crate) fn slot_dict_reflected_union(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    dict_union(runtime, left, right)
}

fn dict_union(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let (Ok(left), Ok(right)) = (left.cast::<PyDict>(runtime), right.cast::<PyDict>(runtime))
    else {
        return Ok(None);
    };
    let additions = right.items(runtime)?;
    let union = runtime.dict_copy(left)?;
    let dict = union.cast::<PyDict>(runtime)?;
    for (key, value) in additions {
        runtime.dict_insert(dict, key, value)?;
    }
    Ok(Some(union))
}

pub(crate) fn slot_dict_delete_item(
    runtime: &mut dyn PyRuntime,
    owner: PyValue,
    key: PyValue,
) -> PyResult<Option<PyValue>> {
    let dict = owner.cast::<PyDict>(runtime)?;
    if runtime.dict_remove(dict, &key)?.is_none() {
        return Err(runtime.exception_with_args("KeyError", vec![key]));
    }
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

/// Import only through the simulated module loader, preserving dotted-import return behavior.
fn builtin_import(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("__import__", 1, 5)?;
    args.reject_unknown_keywords("__import__", &["globals", "locals", "fromlist", "level"])?;
    let OwnedPyString(name) = args.positional()[0].cast(runtime)?;
    let level = args
        .positional()
        .get(4)
        .copied()
        .or(args.keyword("__import__", "level")?.copied());
    if let Some(level) = level {
        if !matches!(
            integer_argument(runtime, &level, "level must be an integer")?,
            IntegerArgument::Finite(0)
        ) {
            return Err(PyError::value_error("relative __import__ is not supported"));
        }
    }
    let fromlist = args
        .positional()
        .get(3)
        .copied()
        .or(args.keyword("__import__", "fromlist")?.copied());
    let module = runtime.import_module(&name)?;
    if let Some(fromlist) = fromlist {
        if runtime.truth(&fromlist)? {
            return Ok(module);
        }
    }
    runtime.import_module(name.split('.').next().unwrap_or(&name))
}
