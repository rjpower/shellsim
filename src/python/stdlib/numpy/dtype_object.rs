//! The Python `numpy.dtype` type: an inline value whose payload is a packed [`DType`].
//!
//! `np.dtype(x)` accepts everything a `dtype=` argument does. Equality coerces the other
//! operand the same way, so `np.dtype("i4") == np.int32` and `np.dtype("f8") == "float64"`
//! hold, while an operand that is not a dtype specification compares unequal.

use super::super::super::native::{
    CallArgs, GetterDef, MethodDef, NativeGetterFn, PyError, PyResult, PyRuntime, PyValue,
    ValueKindDef, ValueKindSlots,
};
use super::super::super::Value;
use super::dtype::{DType, Kind};
use super::scalar::NO_SLOTS;

pub(in crate::python) static DTYPE: ValueKindDef = ValueKindDef {
    name: "numpy.dtype",
    construct,
    slots: ValueKindSlots {
        repr: Some(slot_repr),
        str_: Some(slot_str),
        equal: Some(slot_equal),
        not_equal: Some(slot_not_equal),
        ..NO_SLOTS
    },
    methods: &[MethodDef {
        type_name: "numpy.dtype",
        name: "newbyteorder",
        call: method_newbyteorder,
    }],
    getters: &[
        getter("name", get_name),
        getter("kind", get_kind),
        getter("char", get_char),
        getter("str", get_descr),
        getter("descr", get_descr_list),
        getter("itemsize", get_itemsize),
        getter("alignment", get_itemsize),
        getter("byteorder", get_byteorder),
        getter("type", get_type),
        getter("ndim", get_ndim),
        getter("shape", get_shape),
        getter("names", get_none),
        getter("fields", get_none),
        getter("subdtype", get_none),
        getter("metadata", get_none),
        getter("base", get_base),
        getter("isbuiltin", get_isbuiltin),
        getter("isnative", get_isnative),
        getter("hasobject", get_hasobject),
        getter("num", get_num),
    ],
    bases: &[],
    call: None,
    numeric: None,
};

const fn getter(name: &'static str, get: NativeGetterFn) -> GetterDef {
    GetterDef {
        owner: "numpy.dtype",
        name,
        get,
    }
}

/// A new `numpy.dtype` value.
pub(in crate::python) fn new(runtime: &mut dyn PyRuntime, dtype: DType) -> PyResult {
    runtime.new_value_kind(&DTYPE, dtype.pack())
}

/// The dtype held by a `numpy.dtype` value.
pub(in crate::python) fn unpack(runtime: &dyn PyRuntime, value: &PyValue) -> Option<DType> {
    DType::unpack(runtime.value_kind_payload(value, &DTYPE)?)
}

fn receiver(runtime: &dyn PyRuntime, value: &PyValue) -> DType {
    unpack(runtime, value).expect("dtype methods receive dtype values")
}

fn construct(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: super::args::Signature =
        super::args::Signature::new("dtype", &["dtype", "align", "copy", "metadata"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let value = bound.required("dtype");
    // `np.dtype(None)` is NumPy's default dtype.
    let dtype = if value.is_none() {
        DType::FLOAT64
    } else {
        super::args::dtype(runtime, value)?
    };
    new(runtime, dtype)
}

fn slot_repr(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Option<PyValue>> {
    let dtype = receiver(runtime, &value);
    runtime.new_string(dtype.repr()).map(Some)
}

fn slot_str(runtime: &mut dyn PyRuntime, receiver_value: PyValue) -> PyResult<Option<PyValue>> {
    let dtype = receiver(runtime, &receiver_value);
    runtime.new_string(dtype.display()).map(Some)
}

/// Coerce the other operand of a comparison; anything that is not a dtype spec is unequal.
fn coerced(runtime: &mut dyn PyRuntime, value: PyValue) -> Option<DType> {
    if value.is_none() {
        return None;
    }
    super::args::dtype(runtime, value).ok()
}

fn slot_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let dtype = receiver(runtime, &left);
    Ok(Some(Value::Bool(coerced(runtime, right) == Some(dtype))))
}

fn slot_not_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let dtype = receiver(runtime, &left);
    Ok(Some(Value::Bool(coerced(runtime, right) != Some(dtype))))
}

fn get_name(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let dtype = receiver(runtime, &value);
    runtime.new_string(dtype.name())
}

fn get_kind(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let dtype = receiver(runtime, &value);
    runtime.new_string(dtype.kind_char().to_string())
}

fn get_char(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let dtype = receiver(runtime, &value);
    runtime.new_string(dtype.char().to_string())
}

fn get_descr(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let dtype = receiver(runtime, &value);
    runtime.new_string(dtype.descr())
}

fn get_descr_list(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let dtype = receiver(runtime, &value);
    let name = runtime.new_string(String::new())?;
    let descr = runtime.new_string(dtype.descr())?;
    let entry = runtime.new_tuple(vec![name, descr])?;
    runtime.new_list(vec![entry])
}

fn get_itemsize(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(Value::Int(receiver(runtime, &value).itemsize() as i64))
}

fn get_byteorder(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let dtype = receiver(runtime, &value);
    runtime.new_string(dtype.byte_order().to_string())
}

fn get_type(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let dtype = receiver(runtime, &value);
    match super::scalar::scalar_kind(dtype.kind()) {
        Some(kind) => runtime.value_kind_type(kind),
        // `str` and `object` elements box to builtin values, so their scalar types are the
        // builtin types.
        None => {
            let name = if dtype.kind() == Kind::Str {
                "str"
            } else {
                "object"
            };
            runtime.builtin_type(name).ok_or_else(|| {
                super::super::super::native::PyError::runtime_error("builtin type missing")
            })
        }
    }
}

fn get_ndim(_runtime: &mut dyn PyRuntime, _value: PyValue) -> PyResult {
    Ok(Value::Int(0))
}

fn get_shape(runtime: &mut dyn PyRuntime, _value: PyValue) -> PyResult {
    runtime.new_tuple(Vec::new())
}

fn get_none(_runtime: &mut dyn PyRuntime, _value: PyValue) -> PyResult {
    Ok(Value::None)
}

fn get_base(_runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(value)
}

/// Native dtypes are NumPy's builtin descriptors; a big-endian one is a derived instance.
fn get_isbuiltin(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(Value::Int(i64::from(receiver(runtime, &value).is_native())))
}

fn get_isnative(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(Value::Bool(receiver(runtime, &value).is_native()))
}

/// `dtype.newbyteorder(new_order='S')`: `S` swaps, `<`/`L`/`=`/`N`/`I`/`|` select native order,
/// and `>`/`B` big-endian.
fn method_newbyteorder(
    runtime: &mut dyn PyRuntime,
    receiver_value: PyValue,
    args: CallArgs,
) -> PyResult {
    static SIGNATURE: super::args::Signature =
        super::args::Signature::new("newbyteorder", &["new_order"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let dtype = receiver(runtime, &receiver_value);
    // `PyArray_ByteorderConverter` reads only the first character, so "big" and "swap" work.
    let order = match bound.value("new_order") {
        Some(value) => match runtime.string_value(&value)? {
            Some(text) => text,
            None => {
                return Err(PyError::type_error(format!(
                    "byteorder must be str, not {}",
                    runtime.type_name(&value)?
                )))
            }
        },
        None => "S".to_string(),
    };
    let dtype = match order.chars().next() {
        Some('S' | 's') if dtype.is_native() => dtype.big_endian(),
        Some('S' | 's' | '<' | 'L' | 'l' | '=' | 'N' | 'n' | 'I' | 'i' | '|') => dtype.native(),
        Some('>' | 'B' | 'b') => dtype.big_endian(),
        _ => {
            return Err(PyError::value_error(format!(
                "byteorder not recognized (got {order:?})"
            )))
        }
    };
    new(runtime, dtype)
}

fn get_hasobject(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(Value::Bool(
        receiver(runtime, &value).kind() == Kind::Object,
    ))
}

fn get_num(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    // NumPy's type numbers for the supported kinds.
    const NUMBERS: [i64; 16] = [0, 1, 3, 5, 7, 2, 4, 6, 8, 23, 11, 12, 14, 15, 19, 17];
    Ok(Value::Int(
        NUMBERS[receiver(runtime, &value).kind() as usize],
    ))
}
