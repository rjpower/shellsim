//! Small NumPy-compatible array operations over one opaque shape/stride contract.
//!
//! Kernels deliberately use logical index iteration and runtime scalar operations. This keeps
//! arrays type-erased, deterministic, and easy to meter; no host BLAS, native pointer, or dtype
//! implementation crosses the interpreter boundary.

use std::cmp::Ordering;

use super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyArray, PyArrayDtype,
    PyArrayLayout, PyBinaryOp, PyConstant, PyError, PyIndex, PyKind, PyMarker, PyResult, PyRuntime,
    PySequence, PyString, PyValue, PyValueCast, ValueDef, ValueKindDef, ValueKindSlots,
};
use super::super::number::{PyNumber, PyNumber as Number};
use super::super::Value;

const MAX_ARRAY_RANK: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumericClass {
    Bool,
    Signed,
    Unsigned,
    Float,
}

struct NumericKind {
    definition: ValueKindDef,
    dtype: PyArrayDtype,
    class: NumericClass,
    bits: u8,
    aliases: &'static [&'static str],
}

impl NumericKind {
    fn unpack(&'static self, runtime: &dyn PyRuntime, value: &PyValue) -> Option<Scalar> {
        let payload = runtime.value_kind_payload(value, &self.definition)?;
        let value = match self.class {
            NumericClass::Bool => ScalarValue::Bool(payload != 0),
            NumericClass::Signed => {
                let shift = 64 - self.bits;
                ScalarValue::Signed((((payload << shift) as i64) >> shift) as i128)
            }
            NumericClass::Unsigned => {
                ScalarValue::Unsigned((payload & bit_mask(self.bits)) as u128)
            }
            NumericClass::Float if self.bits == 32 => {
                ScalarValue::Float(f32::from_bits(payload as u32) as f64)
            }
            NumericClass::Float => ScalarValue::Float(f64::from_bits(payload)),
        };
        Some(Scalar {
            dtype: self.dtype,
            value,
        })
    }

    fn pack_payload(&'static self, runtime: &dyn PyRuntime, payload: u64) -> PyResult {
        runtime.new_value_kind(&self.definition, payload & bit_mask(self.bits))
    }
}

macro_rules! numeric_kind {
    ($static_name:ident, $constructor:ident, $getter:ident, $dtype:ident, $id:literal, $class:ident, $bits:literal, $dtype_name:literal, $python_name:literal, [$($alias:literal),+ $(,)?]) => {
        impl PyArrayDtype {
            #[allow(non_upper_case_globals)]
            const $dtype: Self = Self::new($id, $dtype_name);
        }

        static $static_name: NumericKind = NumericKind {
            definition: ValueKindDef {
                name: concat!("numpy.", $python_name),
                construct: $constructor,
                slots: scalar_slots(),
            },
            dtype: PyArrayDtype::$dtype,
            class: NumericClass::$class,
            bits: $bits,
            aliases: &[$($alias),+],
        };

        fn $constructor(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
            construct_scalar(runtime, args, &$static_name)
        }

        fn $getter(runtime: &mut dyn PyRuntime) -> PyResult {
            runtime.value_kind_type(&$static_name.definition)
        }
    };
}

macro_rules! numeric_kinds {
    ($(($static_name:ident, $constructor:ident, $getter:ident, $dtype:ident, $id:literal, $class:ident, $bits:literal, $dtype_name:literal, $python_name:literal, [$($alias:literal),+ $(,)?])),+ $(,)?) => {
        $(numeric_kind!(
            $static_name,
            $constructor,
            $getter,
            $dtype,
            $id,
            $class,
            $bits,
            $dtype_name,
            $python_name,
            [$($alias),+]
        );)+

        static NUMERIC_KINDS: &[&NumericKind] = &[$(&$static_name),+];
    };
}

numeric_kinds!(
    (
        BOOL,
        construct_bool,
        bool_type,
        Bool,
        0,
        Bool,
        1,
        "bool",
        "bool_",
        ["bool", "bool_", "?"]
    ),
    (
        INT8,
        construct_int8,
        int8_type,
        Int8,
        1,
        Signed,
        8,
        "int8",
        "int8",
        ["int8", "byte", "i1"]
    ),
    (
        INT16,
        construct_int16,
        int16_type,
        Int16,
        2,
        Signed,
        16,
        "int16",
        "int16",
        ["int16", "short", "i2"]
    ),
    (
        INT32,
        construct_int32,
        int32_type,
        Int32,
        3,
        Signed,
        32,
        "int32",
        "int32",
        ["int32", "intc", "i4"]
    ),
    (
        INT64,
        construct_int64,
        int64_type,
        Int64,
        4,
        Signed,
        64,
        "int64",
        "int64",
        ["int", "int64", "int_", "intp", "longlong", "i8"]
    ),
    (
        UINT8,
        construct_uint8,
        uint8_type,
        UInt8,
        5,
        Unsigned,
        8,
        "uint8",
        "uint8",
        ["uint8", "ubyte", "u1"]
    ),
    (
        UINT16,
        construct_uint16,
        uint16_type,
        UInt16,
        6,
        Unsigned,
        16,
        "uint16",
        "uint16",
        ["uint16", "ushort", "u2"]
    ),
    (
        UINT32,
        construct_uint32,
        uint32_type,
        UInt32,
        7,
        Unsigned,
        32,
        "uint32",
        "uint32",
        ["uint32", "uintc", "u4"]
    ),
    (
        UINT64,
        construct_uint64,
        uint64_type,
        UInt64,
        8,
        Unsigned,
        64,
        "uint64",
        "uint64",
        ["uint64", "uint", "uintp", "ulonglong", "u8"]
    ),
    (
        FLOAT32,
        construct_float32,
        float32_type,
        Float32,
        9,
        Float,
        32,
        "float32",
        "float32",
        ["float32", "single", "f4"]
    ),
    (
        FLOAT64,
        construct_float64,
        float64_type,
        Float64,
        10,
        Float,
        64,
        "float64",
        "float64",
        ["float", "float64", "double", "f8"]
    ),
);

pub(super) fn value_kinds() -> impl Iterator<Item = &'static ValueKindDef> {
    NUMERIC_KINDS.iter().map(|kind| &kind.definition)
}

const fn scalar_slots() -> ValueKindSlots {
    ValueKindSlots {
        repr: Some(slot_scalar_repr),
        bool_: Some(slot_scalar_bool),
        add: Some(slot_scalar_add),
        reflected_add: Some(slot_scalar_add),
        subtract: Some(slot_scalar_subtract),
        reflected_subtract: Some(slot_scalar_reflected_subtract),
        multiply: Some(slot_scalar_multiply),
        reflected_multiply: Some(slot_scalar_multiply),
        divide: Some(slot_scalar_divide),
        reflected_divide: Some(slot_scalar_reflected_divide),
        equal: Some(slot_scalar_equal),
        not_equal: Some(slot_scalar_not_equal),
        less_than: Some(slot_scalar_less_than),
        less_equal: Some(slot_scalar_less_equal),
        greater_than: Some(slot_scalar_greater_than),
        greater_equal: Some(slot_scalar_greater_equal),
    }
}

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "numpy",
    functions: &[
        function("array", array),
        function("asarray", asarray),
        function("zeros", zeros),
        function("ones", ones),
        function("full", full),
        function("zeros_like", zeros_like),
        function("ones_like", ones_like),
        function("full_like", full_like),
        function("arange", arange),
        function("linspace", linspace),
        function("eye", eye),
        function("identity", identity),
        function("reshape", module_reshape),
        function("transpose", module_transpose),
        function("squeeze", module_squeeze),
        function("expand_dims", expand_dims),
        function("swapaxes", module_swapaxes),
        function("broadcast_to", broadcast_to),
        function("concatenate", concatenate),
        function("stack", stack),
        function("vstack", vstack),
        function("hstack", hstack),
        function("where", where_),
        function("minimum", minimum),
        function("maximum", maximum),
        function("clip", clip),
        function("negative", negative),
        function("absolute", absolute),
        function("abs", absolute),
        function("sqrt", sqrt),
        function("exp", exp),
        function("log", log),
        function("log2", log2),
        function("sin", sin),
        function("cos", cos),
        function("tan", tan),
        function("floor", floor),
        function("ceil", ceil),
        function("rint", rint),
        function("sign", sign),
        function("isnan", isnan),
        function("isinf", isinf),
        function("sum", module_sum),
        function("prod", module_prod),
        function("mean", module_mean),
        function("min", module_min),
        function("max", module_max),
        function("var", module_var),
        function("std", module_std),
        function("median", module_median),
        function("all", module_all),
        function("any", module_any),
        function("argmin", module_argmin),
        function("argmax", module_argmax),
        function("cumsum", module_cumsum),
        function("cumprod", module_cumprod),
        function("dot", dot),
        function("inner", inner),
        function("outer", outer),
        function("matmul", matmul),
        function("diag", diag),
        function("allclose", allclose),
        function("argsort", argsort),
        function("percentile", percentile),
    ],
    values: &[
        ValueDef::Constant {
            name: "pi",
            value: PyConstant::Float(std::f64::consts::PI),
        },
        ValueDef::Constant {
            name: "inf",
            value: PyConstant::Float(f64::INFINITY),
        },
        ValueDef::Constant {
            name: "nan",
            value: PyConstant::Float(f64::NAN),
        },
        ValueDef::Constant {
            name: "__version__",
            value: PyConstant::String("2.0.0-shellsim"),
        },
        ValueDef::Factory {
            name: "ndarray",
            get: array_type,
        },
        dtype_value("bool", bool_type),
        dtype_value("bool_", bool_type),
        dtype_value("int8", int8_type),
        dtype_value("byte", int8_type),
        dtype_value("int16", int16_type),
        dtype_value("short", int16_type),
        dtype_value("int32", int32_type),
        dtype_value("intc", int32_type),
        dtype_value("int64", int64_type),
        dtype_value("int_", int64_type),
        dtype_value("intp", int64_type),
        dtype_value("longlong", int64_type),
        dtype_value("uint8", uint8_type),
        dtype_value("ubyte", uint8_type),
        dtype_value("uint16", uint16_type),
        dtype_value("ushort", uint16_type),
        dtype_value("uint32", uint32_type),
        dtype_value("uintc", uint32_type),
        dtype_value("uint64", uint64_type),
        dtype_value("uint", uint64_type),
        dtype_value("uintp", uint64_type),
        dtype_value("ulonglong", uint64_type),
        dtype_value("float32", float32_type),
        dtype_value("single", float32_type),
        dtype_value("float64", float64_type),
        dtype_value("double", float64_type),
    ],
};

fn array_type(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.marker(PyMarker::ArrayType))
}

const fn dtype_value(name: &'static str, get: fn(&mut dyn PyRuntime) -> PyResult) -> ValueDef {
    ValueDef::Factory { name, get }
}

const fn function(
    name: &'static str,
    call: fn(&mut dyn PyRuntime, CallArgs) -> PyResult,
) -> FunctionDef {
    FunctionDef {
        module: "numpy",
        name,
        call,
    }
}

pub(crate) static ARRAY_TYPE: NativeTypeDef = NativeTypeDef {
    name: "numpy.ndarray",
    methods: &[
        method("tolist", method_tolist),
        method("copy", method_copy),
        method("astype", method_astype),
        method("reshape", method_reshape),
        method("transpose", method_transpose),
        method("flatten", method_flatten),
        method("ravel", method_ravel),
        method("squeeze", method_squeeze),
        method("swapaxes", method_swapaxes),
        method("sum", method_sum),
        method("prod", method_prod),
        method("mean", method_mean),
        method("min", method_min),
        method("max", method_max),
        method("var", method_var),
        method("std", method_std),
        method("all", method_all),
        method("any", method_any),
        method("argmin", method_argmin),
        method("argmax", method_argmax),
        method("cumsum", method_cumsum),
        method("cumprod", method_cumprod),
    ],
};

const fn method(
    name: &'static str,
    call: fn(&mut dyn PyRuntime, PyValue, CallArgs) -> PyResult,
) -> MethodDef {
    MethodDef {
        type_name: "numpy.ndarray",
        name,
        call,
    }
}

fn array(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.array", 1, 1)?;
    args.reject_unknown_keywords("numpy.array", &["dtype"])?;
    let dtype = dtype_keyword(runtime, &args, "numpy.array")?;
    construct(runtime, args.positional()[0], dtype)
}

fn asarray(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.asarray", 1, 1)?;
    args.reject_unknown_keywords("numpy.asarray", &["dtype"])?;
    let dtype = dtype_keyword(runtime, &args, "numpy.asarray")?;
    if dtype.is_none() && runtime.kind(&args.positional()[0])? == PyKind::Array {
        return Ok(args.positional()[0]);
    }
    construct(runtime, args.positional()[0], dtype)
}

fn construct(
    runtime: &mut dyn PyRuntime,
    source: PyValue,
    requested_dtype: Option<PyArrayDtype>,
) -> PyResult {
    if runtime.kind(&source)? == PyKind::Array {
        let array = source.cast::<PyArray>(runtime)?;
        let (layout, source_dtype) = runtime.array_layout(array)?;
        let dtype = requested_dtype.unwrap_or(source_dtype);
        let count = element_count(&layout.shape)?;
        reserve_values(runtime, count)?;
        let mut values = Vec::with_capacity(count);
        for_each_index(&layout.shape, |index| {
            let value = runtime.array_get(array, index)?;
            values.push(convert(runtime, value, dtype)?);
            Ok(())
        })?;
        return runtime.new_array(values, layout.shape, dtype);
    }
    let mut values = Vec::new();
    let shape = flatten(runtime, source, &mut values, 0)?;
    let inferred = infer_dtype(runtime, &values)?;
    let dtype = requested_dtype.unwrap_or(inferred);
    reserve_values(runtime, values.len())?;
    let mut converted = Vec::with_capacity(values.len());
    for value in values {
        runtime.charge_cpu(1)?;
        converted.push(convert(runtime, value, dtype)?);
    }
    runtime.new_array(converted, shape, dtype)
}

fn flatten(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    output: &mut Vec<PyValue>,
    depth: usize,
) -> PyResult<Vec<usize>> {
    if depth > MAX_ARRAY_RANK {
        return Err(PyError::value_error(format!(
            "arrays support at most {MAX_ARRAY_RANK} dimensions"
        )));
    }
    if registered_scalar(runtime, &value).is_some() {
        push_value(runtime, output, value)?;
        return Ok(Vec::new());
    }
    match runtime.kind(&value)? {
        PyKind::List | PyKind::Tuple => {
            let items = value.cast::<PySequence>(runtime)?.items(runtime)?;
            let mut child_shape = None;
            for item in items.iter().copied() {
                runtime.charge_cpu(1)?;
                let shape = flatten(runtime, item, output, depth + 1)?;
                if child_shape
                    .as_ref()
                    .is_some_and(|expected| expected != &shape)
                {
                    return Err(PyError::value_error(
                        "setting an array element with a sequence",
                    ));
                }
                child_shape = Some(shape);
            }
            let mut shape = vec![items.len()];
            shape.extend(child_shape.unwrap_or_default());
            Ok(shape)
        }
        PyKind::Array => {
            let array = value.cast::<PyArray>(runtime)?;
            let (layout, _) = runtime.array_layout(array)?;
            for_each_index(&layout.shape, |index| {
                let value = runtime.array_get(array, index)?;
                push_value(runtime, output, value)?;
                Ok(())
            })?;
            Ok(layout.shape)
        }
        PyKind::Bool | PyKind::Int | PyKind::Float => {
            push_value(runtime, output, value)?;
            Ok(Vec::new())
        }
        _ => Err(PyError::type_error(
            "numpy arrays require numeric scalar values",
        )),
    }
}

fn infer_dtype(runtime: &dyn PyRuntime, values: &[PyValue]) -> PyResult<PyArrayDtype> {
    if values.is_empty() {
        return Ok(PyArrayDtype::Float64);
    }
    let mut dtype = PyArrayDtype::Bool;
    for value in values {
        dtype = promote_dtype(dtype, scalar(runtime, *value)?.dtype);
    }
    Ok(dtype)
}

const fn bit_mask(bits: u8) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn promote_dtype(left: PyArrayDtype, right: PyArrayDtype) -> PyArrayDtype {
    let left = dtype_kind(left);
    let right = dtype_kind(right);
    match (left.class, right.class) {
        (NumericClass::Bool, _) => right.dtype,
        (_, NumericClass::Bool) => left.dtype,
        (NumericClass::Float, NumericClass::Float) => {
            if left.bits.max(right.bits) == 64 {
                PyArrayDtype::Float64
            } else {
                PyArrayDtype::Float32
            }
        }
        (NumericClass::Float, _) | (_, NumericClass::Float) => {
            let float = if left.class == NumericClass::Float {
                left
            } else {
                right
            };
            let other = if left.class == NumericClass::Float {
                right
            } else {
                left
            };
            if float.bits == 64 || other.bits > 16 {
                PyArrayDtype::Float64
            } else {
                PyArrayDtype::Float32
            }
        }
        (NumericClass::Signed, NumericClass::Signed) => signed_dtype(left.bits.max(right.bits)),
        (NumericClass::Unsigned, NumericClass::Unsigned) => {
            unsigned_dtype(left.bits.max(right.bits))
        }
        (NumericClass::Signed, NumericClass::Unsigned)
        | (NumericClass::Unsigned, NumericClass::Signed) => {
            let signed_bits = if left.class == NumericClass::Signed {
                left.bits
            } else {
                right.bits
            };
            let unsigned_bits = if left.class == NumericClass::Unsigned {
                left.bits
            } else {
                right.bits
            };
            [8, 16, 32, 64]
                .into_iter()
                .find(|bits| *bits >= signed_bits && *bits > unsigned_bits)
                .map(signed_dtype)
                .unwrap_or(PyArrayDtype::Float64)
        }
    }
}

fn promote_operands(
    left: PyArrayDtype,
    right: PyArrayDtype,
    left_is_weak: bool,
    right_is_weak: bool,
) -> PyArrayDtype {
    match (left_is_weak, right_is_weak) {
        (true, false) => promote_weak_scalar(right, left),
        (false, true) => promote_weak_scalar(left, right),
        _ => promote_dtype(left, right),
    }
}

fn promote_weak_scalar(strong: PyArrayDtype, weak: PyArrayDtype) -> PyArrayDtype {
    match (dtype_kind(strong).class, dtype_kind(weak).class) {
        (_, NumericClass::Bool) => strong,
        (NumericClass::Bool, NumericClass::Signed | NumericClass::Unsigned) => weak,
        (NumericClass::Float, NumericClass::Signed | NumericClass::Unsigned) => strong,
        (
            NumericClass::Signed | NumericClass::Unsigned,
            NumericClass::Signed | NumericClass::Unsigned,
        ) => strong,
        (NumericClass::Float, NumericClass::Float) => strong,
        (_, NumericClass::Float) => PyArrayDtype::Float64,
    }
}

fn signed_dtype(bits: u8) -> PyArrayDtype {
    match bits {
        0..=8 => PyArrayDtype::Int8,
        9..=16 => PyArrayDtype::Int16,
        17..=32 => PyArrayDtype::Int32,
        _ => PyArrayDtype::Int64,
    }
}

fn unsigned_dtype(bits: u8) -> PyArrayDtype {
    match bits {
        0..=8 => PyArrayDtype::UInt8,
        9..=16 => PyArrayDtype::UInt16,
        17..=32 => PyArrayDtype::UInt32,
        _ => PyArrayDtype::UInt64,
    }
}

fn dtype_kind(dtype: PyArrayDtype) -> &'static NumericKind {
    NUMERIC_KINDS
        .iter()
        .copied()
        .find(|kind| kind.dtype == dtype)
        .expect("every array dtype has a registered scalar kind")
}

fn dtype_keyword(
    runtime: &mut dyn PyRuntime,
    args: &CallArgs,
    function: &str,
) -> PyResult<Option<PyArrayDtype>> {
    args.keyword(function, "dtype")?
        .copied()
        .map(|value| parse_dtype(runtime, value))
        .transpose()
}

fn parse_dtype(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<PyArrayDtype> {
    let is_string = runtime.kind(&value)? == PyKind::String;
    let rendered = if is_string {
        value.cast::<PyString>(runtime)?.0
    } else {
        runtime.repr(&value)?
    };
    let name = rendered
        .strip_prefix("<class '")
        .and_then(|name| name.strip_suffix("'>"))
        .unwrap_or(&rendered);
    let name = if is_string {
        name
    } else {
        name.strip_prefix("numpy.").unwrap_or(name)
    };
    NUMERIC_KINDS
        .iter()
        .find(|kind| kind.aliases.contains(&name))
        .map(|kind| kind.dtype)
        .ok_or_else(|| PyError::type_error(format!("unsupported numpy dtype {rendered:?}")))
}

fn construct_scalar(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    kind: &'static NumericKind,
) -> PyResult {
    args.expect_positional(kind.definition.name, 0, 1)?;
    args.reject_keywords(kind.definition.name)?;
    let default = match kind.class {
        NumericClass::Bool => Value::Bool(false),
        NumericClass::Signed | NumericClass::Unsigned => Value::Int(0),
        NumericClass::Float => Value::Float(0.0),
    };
    convert(
        runtime,
        args.positional().first().copied().unwrap_or(default),
        kind.dtype,
    )
}

#[derive(Clone, Copy)]
enum ScalarValue {
    Bool(bool),
    Signed(i128),
    Unsigned(u128),
    Float(f64),
}

#[derive(Clone, Copy)]
struct Scalar {
    dtype: PyArrayDtype,
    value: ScalarValue,
}

impl Scalar {
    fn as_f64(self) -> f64 {
        match self.value {
            ScalarValue::Bool(value) => i64::from(value) as f64,
            ScalarValue::Signed(value) => value as f64,
            ScalarValue::Unsigned(value) => value as f64,
            ScalarValue::Float(value) => value,
        }
    }

    fn as_i128(self) -> i128 {
        match self.value {
            ScalarValue::Bool(value) => i128::from(value),
            ScalarValue::Signed(value) => value,
            ScalarValue::Unsigned(value) => value as i128,
            ScalarValue::Float(_) => unreachable!("float is not an integer operand"),
        }
    }

    fn as_u128(self) -> u128 {
        match self.value {
            ScalarValue::Bool(value) => u128::from(value),
            ScalarValue::Unsigned(value) => value,
            ScalarValue::Signed(value) => value as u128,
            ScalarValue::Float(_) => unreachable!("float is not an integer operand"),
        }
    }

    fn truth(self) -> bool {
        match self.value {
            ScalarValue::Bool(value) => value,
            ScalarValue::Signed(value) => value != 0,
            ScalarValue::Unsigned(value) => value != 0,
            ScalarValue::Float(value) => value != 0.0,
        }
    }

    fn is_nan(self) -> bool {
        matches!(self.value, ScalarValue::Float(value) if value.is_nan())
    }

    fn compare(self, other: Self) -> Option<Ordering> {
        if matches!(self.value, ScalarValue::Float(_))
            || matches!(other.value, ScalarValue::Float(_))
        {
            return self.as_f64().partial_cmp(&other.as_f64());
        }
        match (self.value, other.value) {
            (ScalarValue::Signed(left), ScalarValue::Unsigned(right)) => {
                if left < 0 {
                    Some(Ordering::Less)
                } else {
                    (left as u128).partial_cmp(&right)
                }
            }
            (ScalarValue::Unsigned(left), ScalarValue::Signed(right)) => {
                if right < 0 {
                    Some(Ordering::Greater)
                } else {
                    left.partial_cmp(&(right as u128))
                }
            }
            _ => self.as_i128().partial_cmp(&other.as_i128()),
        }
    }
}

fn registered_scalar(runtime: &dyn PyRuntime, value: &PyValue) -> Option<Scalar> {
    NUMERIC_KINDS
        .iter()
        .find_map(|kind| kind.unpack(runtime, value))
}

fn scalar(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<Scalar> {
    if let Some(value) = registered_scalar(runtime, &value) {
        return Ok(value);
    }
    if runtime.kind(&value)? == PyKind::Bool {
        return Ok(Scalar {
            dtype: PyArrayDtype::Bool,
            value: ScalarValue::Bool(value.bool_value().expect("bool kind checked")),
        });
    }
    match value.cast::<Number>(runtime)? {
        Number::Int(value) => Ok(Scalar {
            dtype: PyArrayDtype::Int64,
            value: ScalarValue::Signed(value as i128),
        }),
        Number::BigInt(value) => {
            if let Ok(value) = value.parse::<i64>() {
                Ok(Scalar {
                    dtype: PyArrayDtype::Int64,
                    value: ScalarValue::Signed(value as i128),
                })
            } else if let Ok(value) = value.parse::<u64>() {
                Ok(Scalar {
                    dtype: PyArrayDtype::UInt64,
                    value: ScalarValue::Unsigned(value as u128),
                })
            } else {
                Err(PyError::overflow_error(
                    "integer is outside the supported NumPy range",
                ))
            }
        }
        Number::Float(value) => Ok(Scalar {
            dtype: PyArrayDtype::Float64,
            value: ScalarValue::Float(value),
        }),
    }
}

fn convert(runtime: &mut dyn PyRuntime, value: PyValue, dtype: PyArrayDtype) -> PyResult<PyValue> {
    let value = match scalar(runtime, value) {
        Ok(value) => value,
        Err(_) if dtype == PyArrayDtype::Bool => Scalar {
            dtype: PyArrayDtype::Bool,
            value: ScalarValue::Bool(runtime.truth(&value)?),
        },
        Err(error) => return Err(error),
    };
    pack_checked(runtime, value, dtype)
}

fn pack_checked(runtime: &dyn PyRuntime, value: Scalar, dtype: PyArrayDtype) -> PyResult {
    let kind = dtype_kind(dtype);
    let converted = match kind.class {
        NumericClass::Bool => ScalarValue::Bool(value.truth()),
        NumericClass::Float => ScalarValue::Float(value.as_f64()),
        NumericClass::Signed => {
            let minimum = -(1i128 << (kind.bits - 1));
            let maximum = (1i128 << (kind.bits - 1)) - 1;
            let value = match value.value {
                ScalarValue::Bool(value) => i128::from(value),
                ScalarValue::Signed(value) => value,
                ScalarValue::Unsigned(value) => i128::try_from(value).unwrap_or(i128::MAX),
                ScalarValue::Float(value) if value.is_finite() => {
                    let boundary = 2f64.powi(i32::from(kind.bits) - 1);
                    if value < -boundary || value >= boundary {
                        return Err(dtype_overflow(dtype));
                    }
                    value.trunc() as i128
                }
                ScalarValue::Float(_) => return Err(dtype_overflow(dtype)),
            };
            if !(minimum..=maximum).contains(&value) {
                return Err(dtype_overflow(dtype));
            }
            ScalarValue::Signed(value)
        }
        NumericClass::Unsigned => {
            let maximum = (1u128 << kind.bits) - 1;
            let value = match value.value {
                ScalarValue::Bool(value) => u128::from(value),
                ScalarValue::Signed(value) if value >= 0 => value as u128,
                ScalarValue::Signed(_) => return Err(dtype_overflow(dtype)),
                ScalarValue::Unsigned(value) => value,
                ScalarValue::Float(value) if value.is_finite() => {
                    let boundary = 2f64.powi(i32::from(kind.bits));
                    if value < 0.0 || value >= boundary {
                        return Err(dtype_overflow(dtype));
                    }
                    value.trunc() as u128
                }
                ScalarValue::Float(_) => return Err(dtype_overflow(dtype)),
            };
            if value > maximum {
                return Err(dtype_overflow(dtype));
            }
            ScalarValue::Unsigned(value)
        }
    };
    pack_wrapping(
        runtime,
        Scalar {
            dtype,
            value: converted,
        },
        dtype,
    )
}

fn pack_wrapping(runtime: &dyn PyRuntime, value: Scalar, dtype: PyArrayDtype) -> PyResult {
    let kind = dtype_kind(dtype);
    let payload = match kind.class {
        NumericClass::Bool => u64::from(value.truth()),
        NumericClass::Signed => value.as_i128() as u64,
        NumericClass::Unsigned => value.as_u128() as u64,
        NumericClass::Float if kind.bits == 32 => (value.as_f64() as f32).to_bits() as u64,
        NumericClass::Float => value.as_f64().to_bits(),
    };
    kind.pack_payload(runtime, payload)
}

fn cast_scalar(runtime: &dyn PyRuntime, value: Scalar, dtype: PyArrayDtype) -> PyResult<Scalar> {
    let packed = pack_checked(runtime, value, dtype)?;
    registered_scalar(runtime, &packed)
        .ok_or_else(|| PyError::runtime_error("numeric cast produced an unregistered scalar"))
}

fn pack_bool(runtime: &dyn PyRuntime, value: bool) -> PyResult {
    pack_wrapping(
        runtime,
        Scalar {
            dtype: PyArrayDtype::Bool,
            value: ScalarValue::Bool(value),
        },
        PyArrayDtype::Bool,
    )
}

fn pack_index(runtime: &dyn PyRuntime, value: usize) -> PyResult {
    let value = i128::try_from(value)
        .map_err(|_| PyError::overflow_error("array index exceeds numpy.int64"))?;
    pack_wrapping(
        runtime,
        Scalar {
            dtype: PyArrayDtype::Int64,
            value: ScalarValue::Signed(value),
        },
        PyArrayDtype::Int64,
    )
}

fn pack_float(runtime: &dyn PyRuntime, value: f64, dtype: PyArrayDtype) -> PyResult {
    pack_wrapping(
        runtime,
        Scalar {
            dtype,
            value: ScalarValue::Float(value),
        },
        dtype,
    )
}

fn dtype_overflow(dtype: PyArrayDtype) -> PyError {
    PyError::overflow_error(format!("integer does not fit in numpy.{}", dtype.name()))
}

fn scalar_binary(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    operation: PyBinaryOp,
) -> PyResult<Option<PyValue>> {
    let left_is_weak = registered_scalar(runtime, &left).is_none();
    let right_is_weak = registered_scalar(runtime, &right).is_none();
    let mut left = scalar(runtime, left)?;
    let mut right = scalar(runtime, right)?;
    if left.dtype == PyArrayDtype::Bool
        && right.dtype == PyArrayDtype::Bool
        && !matches!(operation, PyBinaryOp::Divide)
    {
        let value = match operation {
            PyBinaryOp::Add => left.truth() || right.truth(),
            PyBinaryOp::Multiply => left.truth() && right.truth(),
            PyBinaryOp::Subtract => {
                return Err(PyError::type_error(
                    "numpy boolean subtract is not supported",
                ))
            }
            PyBinaryOp::Divide => unreachable!(),
        };
        return pack_wrapping(
            runtime,
            Scalar {
                dtype: PyArrayDtype::Bool,
                value: ScalarValue::Bool(value),
            },
            PyArrayDtype::Bool,
        )
        .map(Some);
    }
    let mut dtype = promote_operands(left.dtype, right.dtype, left_is_weak, right_is_weak);
    if matches!(operation, PyBinaryOp::Divide) && dtype_kind(dtype).class != NumericClass::Float {
        dtype = PyArrayDtype::Float64;
    }
    if left_is_weak {
        left = cast_scalar(runtime, left, dtype)?;
    }
    if right_is_weak {
        right = cast_scalar(runtime, right, dtype)?;
    }
    let class = dtype_kind(dtype).class;
    let value = match class {
        NumericClass::Float => {
            let left = left.as_f64();
            let right = right.as_f64();
            if matches!(operation, PyBinaryOp::Divide) && right == 0.0 {
                return Err(PyError::zero_division_error("division by zero"));
            }
            let value = if dtype == PyArrayDtype::Float32 {
                let left = left as f32;
                let right = right as f32;
                (match operation {
                    PyBinaryOp::Add => left + right,
                    PyBinaryOp::Subtract => left - right,
                    PyBinaryOp::Multiply => left * right,
                    PyBinaryOp::Divide => left / right,
                }) as f64
            } else {
                match operation {
                    PyBinaryOp::Add => left + right,
                    PyBinaryOp::Subtract => left - right,
                    PyBinaryOp::Multiply => left * right,
                    PyBinaryOp::Divide => left / right,
                }
            };
            ScalarValue::Float(value)
        }
        NumericClass::Signed => {
            let left = left.as_i128();
            let right = right.as_i128();
            ScalarValue::Signed(match operation {
                PyBinaryOp::Add => left.wrapping_add(right),
                PyBinaryOp::Subtract => left.wrapping_sub(right),
                PyBinaryOp::Multiply => left.wrapping_mul(right),
                PyBinaryOp::Divide => unreachable!(),
            })
        }
        NumericClass::Unsigned => {
            let left = left.as_u128();
            let right = right.as_u128();
            ScalarValue::Unsigned(match operation {
                PyBinaryOp::Add => left.wrapping_add(right),
                PyBinaryOp::Subtract => left.wrapping_sub(right),
                PyBinaryOp::Multiply => left.wrapping_mul(right),
                PyBinaryOp::Divide => unreachable!(),
            })
        }
        NumericClass::Bool => unreachable!("boolean pairs returned above"),
    };
    pack_wrapping(runtime, Scalar { dtype, value }, dtype).map(Some)
}

pub(crate) fn slot_scalar_add(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    scalar_binary(runtime, left, right, PyBinaryOp::Add)
}

pub(crate) fn slot_scalar_subtract(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    scalar_binary(runtime, left, right, PyBinaryOp::Subtract)
}

pub(crate) fn slot_scalar_reflected_subtract(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    scalar_binary(runtime, left, right, PyBinaryOp::Subtract)
}

pub(crate) fn slot_scalar_multiply(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    scalar_binary(runtime, left, right, PyBinaryOp::Multiply)
}

pub(crate) fn slot_scalar_divide(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    scalar_binary(runtime, left, right, PyBinaryOp::Divide)
}

pub(crate) fn slot_scalar_reflected_divide(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    scalar_binary(runtime, left, right, PyBinaryOp::Divide)
}

fn slot_scalar_repr(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Option<PyValue>> {
    let scalar = scalar(runtime, value)?;
    let text = match scalar.value {
        ScalarValue::Bool(value) => if value { "True" } else { "False" }.into(),
        ScalarValue::Signed(value) => value.to_string(),
        ScalarValue::Unsigned(value) => value.to_string(),
        ScalarValue::Float(value) => {
            let text = if scalar.dtype == PyArrayDtype::Float32 {
                (value as f32).to_string()
            } else {
                value.to_string()
            };
            if text.contains(['.', 'e', 'E']) {
                text
            } else {
                format!("{text}.0")
            }
        }
    };
    runtime.new_string(text).map(Some)
}

fn slot_scalar_bool(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Option<PyValue>> {
    Ok(Some(Value::Bool(scalar(runtime, value)?.truth())))
}

fn slot_scalar_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    Ok(Some(Value::Bool(
        scalar(runtime, left)?.compare(scalar(runtime, right)?) == Some(Ordering::Equal),
    )))
}

fn slot_scalar_less_than(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    Ok(Some(Value::Bool(
        scalar(runtime, left)?.compare(scalar(runtime, right)?) == Some(Ordering::Less),
    )))
}

fn slot_scalar_not_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_scalar_equal(runtime, left, right).map(|value| {
        value.map(|value| Value::Bool(!value.bool_value().expect("comparison returns bool")))
    })
}

fn slot_scalar_less_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    Ok(Some(Value::Bool(matches!(
        scalar(runtime, left)?.compare(scalar(runtime, right)?),
        Some(Ordering::Less | Ordering::Equal)
    ))))
}

fn slot_scalar_greater_than(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    Ok(Some(Value::Bool(
        scalar(runtime, left)?.compare(scalar(runtime, right)?) == Some(Ordering::Greater),
    )))
}

fn slot_scalar_greater_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    Ok(Some(Value::Bool(matches!(
        scalar(runtime, left)?.compare(scalar(runtime, right)?),
        Some(Ordering::Greater | Ordering::Equal)
    ))))
}

fn zeros(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    filled(runtime, args, "numpy.zeros", Value::Int(0))
}

fn ones(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    filled(runtime, args, "numpy.ones", Value::Int(1))
}

fn full(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.full", 2, 2)?;
    args.reject_unknown_keywords("numpy.full", &["dtype"])?;
    let shape = parse_shape(runtime, args.positional()[0])?;
    let requested = dtype_keyword(runtime, &args, "numpy.full")?;
    let dtype = requested.unwrap_or(infer_dtype(runtime, &args.positional()[1..])?);
    let value = convert(runtime, args.positional()[1], dtype)?;
    allocate_filled(runtime, shape, value, dtype)
}

fn filled(runtime: &mut dyn PyRuntime, args: CallArgs, name: &str, value: PyValue) -> PyResult {
    args.expect_positional(name, 1, 1)?;
    args.reject_unknown_keywords(name, &["dtype"])?;
    let shape = parse_shape(runtime, args.positional()[0])?;
    let dtype = dtype_keyword(runtime, &args, name)?.unwrap_or(PyArrayDtype::Float64);
    let value = convert(runtime, value, dtype)?;
    allocate_filled(runtime, shape, value, dtype)
}

fn allocate_filled(
    runtime: &mut dyn PyRuntime,
    shape: Vec<usize>,
    value: PyValue,
    dtype: PyArrayDtype,
) -> PyResult {
    let count = element_count(&shape)?;
    reserve_values(runtime, count)?;
    runtime.charge_cpu(u64::try_from(count).unwrap_or(u64::MAX))?;
    runtime.new_array(vec![value; count], shape, dtype)
}

fn zeros_like(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    like_filled(runtime, args, "numpy.zeros_like", Value::Int(0))
}

fn ones_like(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    like_filled(runtime, args, "numpy.ones_like", Value::Int(1))
}

fn like_filled(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    name: &str,
    value: PyValue,
) -> PyResult {
    args.expect_positional(name, 1, 1)?;
    args.reject_unknown_keywords(name, &["dtype"])?;
    let array = coerce_array(runtime, args.positional()[0])?;
    let (layout, source_dtype) = runtime.array_layout(array)?;
    let dtype = dtype_keyword(runtime, &args, name)?.unwrap_or(source_dtype);
    let value = convert(runtime, value, dtype)?;
    allocate_filled(runtime, layout.shape, value, dtype)
}

fn full_like(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.full_like", 2, 2)?;
    args.reject_unknown_keywords("numpy.full_like", &["dtype"])?;
    let array = coerce_array(runtime, args.positional()[0])?;
    let (layout, source_dtype) = runtime.array_layout(array)?;
    let dtype = dtype_keyword(runtime, &args, "numpy.full_like")?.unwrap_or(source_dtype);
    let value = convert(runtime, args.positional()[1], dtype)?;
    allocate_filled(runtime, layout.shape, value, dtype)
}

fn parse_shape(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<usize>> {
    if let Some(value) = runtime.int_value(&value) {
        return Ok(vec![dimension(value)?]);
    }
    let items = value.cast::<PySequence>(runtime)?.items(runtime)?;
    items
        .into_iter()
        .map(|item| {
            let PyIndex(value) = item.cast(runtime)?;
            dimension(value)
        })
        .collect()
}

fn dimension(value: i64) -> PyResult<usize> {
    usize::try_from(value).map_err(|_| PyError::value_error("negative dimensions are not allowed"))
}

fn element_count(shape: &[usize]) -> PyResult<usize> {
    validate_rank(shape)?;
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| PyError::value_error("array is too large"))
    })
}

pub(in crate::python) fn validate_rank(shape: &[usize]) -> PyResult<()> {
    if shape.len() > MAX_ARRAY_RANK {
        Err(PyError::value_error(format!(
            "arrays support at most {MAX_ARRAY_RANK} dimensions"
        )))
    } else {
        Ok(())
    }
}

fn reserve_values(runtime: &mut dyn PyRuntime, count: usize) -> PyResult<()> {
    let bytes = count
        .checked_mul(std::mem::size_of::<PyValue>())
        .ok_or_else(|| PyError::value_error("array is too large"))?;
    runtime.reserve_memory(bytes)
}

fn push_value(
    runtime: &mut dyn PyRuntime,
    values: &mut Vec<PyValue>,
    value: PyValue,
) -> PyResult<()> {
    reserve_values(runtime, 1)?;
    values.push(value);
    Ok(())
}

fn arange(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.arange", 1, 3)?;
    args.reject_unknown_keywords("numpy.arange", &["dtype"])?;
    let numbers = args
        .positional()
        .iter()
        .copied()
        .map(|value| value.cast::<PyNumber>(runtime))
        .collect::<PyResult<Vec<_>>>()?;
    let float_input = numbers
        .iter()
        .any(|value| matches!(value, Number::Float(_)));
    let converted = numbers
        .into_iter()
        .map(Number::into_f64)
        .collect::<PyResult<Vec<_>>>()?;
    let (start, stop, step) = match converted.as_slice() {
        [stop] => (0.0, *stop, 1.0),
        [start, stop] => (*start, *stop, 1.0),
        [start, stop, step] => (*start, *stop, *step),
        _ => unreachable!(),
    };
    if step == 0.0 || !start.is_finite() || !stop.is_finite() || !step.is_finite() {
        return Err(PyError::value_error(
            "arange arguments must be finite and step must be nonzero",
        ));
    }
    let raw_count = ((stop - start) / step).ceil().max(0.0);
    if raw_count > usize::MAX as f64 {
        return Err(PyError::value_error("array is too large"));
    }
    let count = raw_count as usize;
    reserve_values(runtime, count)?;
    runtime.charge_cpu(u64::try_from(count).unwrap_or(u64::MAX))?;
    let requested = dtype_keyword(runtime, &args, "numpy.arange")?;
    let dtype = requested.unwrap_or(if float_input {
        PyArrayDtype::Float64
    } else {
        PyArrayDtype::Int64
    });
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let value = start + step * index as f64;
        values.push(convert(runtime, Value::Float(value), dtype)?);
    }
    runtime.new_array(values, vec![count], dtype)
}

fn linspace(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.linspace", 2, 3)?;
    args.reject_unknown_keywords("numpy.linspace", &["num", "endpoint", "dtype"])?;
    let start = scalar(runtime, args.positional()[0])?.as_f64();
    let stop = scalar(runtime, args.positional()[1])?.as_f64();
    let num_value = args
        .positional()
        .get(2)
        .copied()
        .or(args.keyword("numpy.linspace", "num")?.copied())
        .unwrap_or(Value::Int(50));
    let PyIndex(num) = num_value.cast(runtime)?;
    let num = dimension(num)?;
    let endpoint = match args.keyword("numpy.linspace", "endpoint")? {
        Some(value) => runtime.truth(value)?,
        None => true,
    };
    let dtype = dtype_keyword(runtime, &args, "numpy.linspace")?.unwrap_or(PyArrayDtype::Float64);
    reserve_values(runtime, num)?;
    let mut values = Vec::with_capacity(num);
    let denominator = if endpoint { num.saturating_sub(1) } else { num };
    for index in 0..num {
        runtime.charge_cpu(1)?;
        let value = if denominator == 0 {
            start
        } else {
            start + (stop - start) * index as f64 / denominator as f64
        };
        values.push(convert(runtime, Value::Float(value), dtype)?);
    }
    runtime.new_array(values, vec![num], dtype)
}

fn eye(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.eye", 1, 3)?;
    args.reject_unknown_keywords("numpy.eye", &["M", "k", "dtype"])?;
    let PyIndex(rows) = args.positional()[0].cast(runtime)?;
    let rows = dimension(rows)?;
    let columns = args
        .positional()
        .get(1)
        .copied()
        .or(args.keyword("numpy.eye", "M")?.copied())
        .map(|value| {
            value
                .cast::<PyIndex>(runtime)
                .and_then(|value| dimension(value.0))
        })
        .transpose()?
        .unwrap_or(rows);
    let diagonal = args
        .positional()
        .get(2)
        .copied()
        .or(args.keyword("numpy.eye", "k")?.copied())
        .map(|value| value.cast::<PyIndex>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or(0);
    let dtype = dtype_keyword(runtime, &args, "numpy.eye")?.unwrap_or(PyArrayDtype::Float64);
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| PyError::value_error("array is too large"))?;
    reserve_values(runtime, count)?;
    let zero = convert(runtime, Value::Int(0), dtype)?;
    let one = convert(runtime, Value::Int(1), dtype)?;
    let mut values = vec![zero; count];
    for row in 0..rows {
        runtime.charge_cpu(1)?;
        let column = i64::try_from(row)
            .ok()
            .and_then(|row| row.checked_add(diagonal))
            .and_then(|column| usize::try_from(column).ok());
        if let Some(column) = column.filter(|column| *column < columns) {
            values[row * columns + column] = one;
        }
    }
    runtime.new_array(values, vec![rows, columns], dtype)
}

fn identity(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.identity", 1, 1)?;
    args.reject_unknown_keywords("numpy.identity", &["dtype"])?;
    eye(runtime, args)
}

fn coerce_array(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<PyArray> {
    if runtime.kind(&value)? == PyKind::Array {
        value.cast(runtime)
    } else {
        construct(runtime, value, None)?.cast(runtime)
    }
}

pub(crate) fn slot_length(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = value.cast::<PyArray>(runtime)?;
    let (layout, _) = runtime.array_layout(array)?;
    let Some(length) = layout.shape.first() else {
        return Err(PyError::type_error("len() of unsized object"));
    };
    Ok(Some(Value::Int(i64::try_from(*length).map_err(|_| {
        PyError::overflow_error("array length overflow")
    })?)))
}

pub(crate) fn slot_bool(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Option<PyValue>> {
    let array = value.cast::<PyArray>(runtime)?;
    let (layout, _) = runtime.array_layout(array)?;
    if element_count(&layout.shape)? != 1 {
        return Err(PyError::value_error(
            "the truth value of an array with other than one element is ambiguous",
        ));
    }
    let index = vec![0; layout.shape.len()];
    let value = runtime.array_get(array, &index)?;
    Ok(Some(Value::Bool(runtime.truth(&value)?)))
}

pub(crate) fn slot_repr(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Option<PyValue>> {
    let array = value.cast::<PyArray>(runtime)?;
    let (layout, _) = runtime.array_layout(array)?;
    let values = tolist_axis(runtime, array, &layout.shape, &mut Vec::new())?;
    let rendered = runtime.repr(&values)?;
    runtime.new_string(format!("array({rendered})")).map(Some)
}

pub(crate) fn slot_iter(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Option<PyValue>> {
    let array = value.cast::<PyArray>(runtime)?;
    let (layout, _) = runtime.array_layout(array)?;
    let Some(length) = layout.shape.first() else {
        return Err(PyError::type_error("iteration over a 0-d array"));
    };
    reserve_values(runtime, *length)?;
    let mut values = Vec::with_capacity(*length);
    for index in 0..*length {
        values.push(get_item(runtime, array, Value::Int(index as i64))?);
    }
    runtime.new_iterator(values).map(Some)
}

pub(crate) fn slot_get_item(
    runtime: &mut dyn PyRuntime,
    array: PyValue,
    index: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = array.cast::<PyArray>(runtime)?;
    Ok(Some(get_item(runtime, array, index)?))
}

pub(crate) fn slot_slice(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> PyResult<Option<PyValue>> {
    let array = value.cast::<PyArray>(runtime)?;
    let (mut layout, _) = runtime.array_layout(array)?;
    let Some(&length) = layout.shape.first() else {
        return Err(PyError::value_error("cannot slice a 0-d array"));
    };
    let (first, count, step) = slice_plan(length, start, stop, step)?;
    if count != 0 {
        layout.offset = layout
            .offset
            .checked_add(
                layout.strides[0]
                    .checked_mul(first)
                    .ok_or_else(|| PyError::value_error("array offset overflow"))?,
            )
            .ok_or_else(|| PyError::value_error("array offset overflow"))?;
    }
    layout.shape[0] = count;
    layout.strides[0] = layout.strides[0]
        .checked_mul(step)
        .ok_or_else(|| PyError::value_error("array stride overflow"))?;
    runtime.new_array_view(array, layout).map(Some)
}

fn slice_plan(
    length: usize,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> PyResult<(isize, usize, isize)> {
    let length =
        i64::try_from(length).map_err(|_| PyError::overflow_error("array dimension overflow"))?;
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(PyError::value_error("slice step cannot be zero"));
    }
    let normalize = |value: i64, minimum: i64, maximum: i64| {
        let value = if value < 0 {
            value.saturating_add(length)
        } else {
            value
        };
        value.clamp(minimum, maximum)
    };
    let (first, stop) = if step > 0 {
        (
            start.map_or(0, |value| normalize(value, 0, length)),
            stop.map_or(length, |value| normalize(value, 0, length)),
        )
    } else {
        (
            start.map_or(length - 1, |value| normalize(value, -1, length - 1)),
            stop.map_or(-1, |value| normalize(value, -1, length - 1)),
        )
    };
    let count = if (step > 0 && first < stop) || (step < 0 && first > stop) {
        usize::try_from((stop - first - step.signum()) / step + 1)
            .map_err(|_| PyError::value_error("slice length overflow"))?
    } else {
        0
    };
    Ok((
        isize::try_from(first).map_err(|_| PyError::value_error("slice offset overflow"))?,
        count,
        isize::try_from(step).map_err(|_| PyError::value_error("slice stride overflow"))?,
    ))
}

fn get_item(runtime: &mut dyn PyRuntime, array: PyArray, index: PyValue) -> PyResult<PyValue> {
    let (layout, _) = runtime.array_layout(array)?;
    if runtime.kind(&index)? == PyKind::Tuple {
        let raw = index.cast::<PySequence>(runtime)?.items(runtime)?;
        if raw.iter().any(|value| runtime.slice_parts(value).is_some()) {
            let layout = basic_slice_layout(runtime, &layout, &raw)?;
            return runtime.new_array_view(array, layout);
        }
    }
    if matches!(runtime.kind(&index)?, PyKind::List | PyKind::Array) {
        let selection = advanced_selection(runtime, &layout.shape, index)?;
        reserve_values(runtime, selection.coordinates.len())?;
        let mut values = Vec::with_capacity(selection.coordinates.len());
        for coordinate in selection.coordinates {
            values.push(runtime.array_get(array, &coordinate)?);
        }
        let dtype = runtime.array_layout(array)?.1;
        return runtime.new_array(values, selection.shape, dtype);
    }
    let indices = parse_indices(runtime, index, &layout.shape)?;
    if indices.len() == layout.shape.len() {
        return runtime.array_get(array, &indices);
    }
    let mut offset = layout.offset;
    for (axis, index) in indices.iter().enumerate() {
        offset = offset
            .checked_add(
                layout.strides[axis]
                    .checked_mul(*index as isize)
                    .ok_or_else(|| PyError::value_error("array offset overflow"))?,
            )
            .ok_or_else(|| PyError::value_error("array offset overflow"))?;
    }
    runtime.new_array_view(
        array,
        PyArrayLayout {
            shape: layout.shape[indices.len()..].to_vec(),
            strides: layout.strides[indices.len()..].to_vec(),
            offset,
        },
    )
}

fn parse_indices(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    shape: &[usize],
) -> PyResult<Vec<usize>> {
    let raw = if runtime.int_value(&value).is_some() {
        vec![value]
    } else {
        value.cast::<PySequence>(runtime)?.items(runtime)?
    };
    if raw.len() > shape.len() {
        return Err(PyError::value_error("too many indices for array"));
    }
    raw.into_iter()
        .enumerate()
        .map(|(axis, value)| {
            let PyIndex(value) = value.cast(runtime)?;
            normalize_index(value, shape[axis])
        })
        .collect()
}

fn normalize_index(index: i64, length: usize) -> PyResult<usize> {
    let length_i64 =
        i64::try_from(length).map_err(|_| PyError::overflow_error("array dimension overflow"))?;
    let index = if index < 0 {
        length_i64.checked_add(index)
    } else {
        Some(index)
    }
    .ok_or_else(|| PyError::value_error("index out of bounds"))?;
    let index = usize::try_from(index).map_err(|_| PyError::value_error("index out of bounds"))?;
    if index >= length {
        Err(PyError::value_error("index out of bounds"))
    } else {
        Ok(index)
    }
}

fn boolean_index(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<bool> {
    match scalar(runtime, value)?.value {
        ScalarValue::Bool(value) => Ok(value),
        _ => Err(PyError::runtime_error(
            "boolean index array contains another scalar kind",
        )),
    }
}

fn integer_index(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<i64> {
    let value = match scalar(runtime, value)?.value {
        ScalarValue::Signed(value) => i64::try_from(value),
        ScalarValue::Unsigned(value) => i64::try_from(value),
        _ => {
            return Err(PyError::runtime_error(
                "integer index array contains another scalar kind",
            ))
        }
    };
    value.map_err(|_| PyError::value_error("index out of bounds"))
}

pub(crate) fn slot_set_item(
    runtime: &mut dyn PyRuntime,
    array: PyValue,
    index: PyValue,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = array.cast::<PyArray>(runtime)?;
    let (layout, dtype) = runtime.array_layout(array)?;
    let basic_components = if runtime.kind(&index)? == PyKind::Tuple {
        Some(index.cast::<PySequence>(runtime)?.items(runtime)?)
    } else if runtime.int_value(&index).is_some() && layout.shape.len() > 1 {
        Some(vec![index])
    } else {
        None
    };
    if let Some(raw) = basic_components {
        if raw.iter().any(|value| runtime.slice_parts(value).is_some())
            || raw.len() < layout.shape.len()
        {
            let selection = basic_slice_layout(runtime, &layout, &raw)?;
            let target = runtime.new_array_view(array, selection.clone())?;
            let target = target.cast::<PyArray>(runtime)?;
            let mut source = value;
            if matches!(runtime.kind(&source)?, PyKind::List | PyKind::Tuple) {
                source = construct(runtime, source, None)?;
            }
            let source_layout = if runtime.kind(&source)? == PyKind::Array {
                Some(runtime.array_layout(source.cast(runtime)?)?.0)
            } else {
                None
            };
            if let Some(source_layout) = &source_layout {
                if broadcast_shape(&selection.shape, &source_layout.shape)? != selection.shape {
                    return Err(PyError::value_error(
                        "assignment value cannot be broadcast to the indexed shape",
                    ));
                }
            }
            for_each_index(&selection.shape, |index| {
                let value = broadcast_get(
                    runtime,
                    source,
                    source_layout.as_ref(),
                    index,
                    &selection.shape,
                )?;
                let value = convert(runtime, value, dtype)?;
                runtime.array_set(target, index, value)
            })?;
            return Ok(Some(Value::None));
        }
    }
    if matches!(runtime.kind(&index)?, PyKind::List | PyKind::Array) {
        let selection = advanced_selection(runtime, &layout.shape, index)?;
        let mut source = value;
        if matches!(runtime.kind(&source)?, PyKind::List | PyKind::Tuple) {
            source = construct(runtime, source, None)?;
        }
        let source_layout = if runtime.kind(&source)? == PyKind::Array {
            Some(runtime.array_layout(source.cast(runtime)?)?.0)
        } else {
            None
        };
        if let Some(source_layout) = &source_layout {
            let broadcast = broadcast_shape(&selection.shape, &source_layout.shape)?;
            if broadcast != selection.shape {
                return Err(PyError::value_error(
                    "assignment value cannot be broadcast to the indexed shape",
                ));
            }
        }
        let mut position = 0usize;
        for_each_index(&selection.shape, |output_index| {
            let selected = selection.coordinates[position].clone();
            position += 1;
            let value = broadcast_get(
                runtime,
                source,
                source_layout.as_ref(),
                output_index,
                &selection.shape,
            )?;
            let value = convert(runtime, value, dtype)?;
            runtime.array_set(array, &selected, value)
        })?;
        return Ok(Some(Value::None));
    }
    let indices = parse_indices(runtime, index, &layout.shape)?;
    if indices.len() != layout.shape.len() {
        return Err(PyError::type_error(
            "partial basic-index assignment is not supported",
        ));
    }
    let value = convert(runtime, value, dtype)?;
    runtime.array_set(array, &indices, value)?;
    Ok(Some(Value::None))
}

fn basic_slice_layout(
    runtime: &dyn PyRuntime,
    layout: &PyArrayLayout,
    components: &[PyValue],
) -> PyResult<PyArrayLayout> {
    if components.len() > layout.shape.len() {
        return Err(PyError::value_error("too many indices for array"));
    }
    let mut offset = layout.offset;
    let mut shape = Vec::with_capacity(layout.shape.len());
    let mut strides = Vec::with_capacity(layout.strides.len());
    for (axis, component) in components.iter().enumerate() {
        if let Some((start, stop, step)) = runtime.slice_parts(component) {
            let (first, count, step) = slice_plan(layout.shape[axis], start, stop, step)?;
            if count != 0 {
                offset = offset
                    .checked_add(
                        layout.strides[axis]
                            .checked_mul(first)
                            .ok_or_else(|| PyError::value_error("array offset overflow"))?,
                    )
                    .ok_or_else(|| PyError::value_error("array offset overflow"))?;
            }
            shape.push(count);
            strides.push(
                layout.strides[axis]
                    .checked_mul(step)
                    .ok_or_else(|| PyError::value_error("array stride overflow"))?,
            );
        } else {
            let PyIndex(index) = (*component).cast(runtime)?;
            let index = normalize_index(index, layout.shape[axis])?;
            offset = offset
                .checked_add(
                    layout.strides[axis]
                        .checked_mul(index as isize)
                        .ok_or_else(|| PyError::value_error("array offset overflow"))?,
                )
                .ok_or_else(|| PyError::value_error("array offset overflow"))?;
        }
    }
    shape.extend_from_slice(&layout.shape[components.len()..]);
    strides.extend_from_slice(&layout.strides[components.len()..]);
    Ok(PyArrayLayout {
        shape,
        strides,
        offset,
    })
}

struct AdvancedSelection {
    coordinates: Vec<Vec<usize>>,
    shape: Vec<usize>,
}

fn advanced_selection(
    runtime: &mut dyn PyRuntime,
    shape: &[usize],
    index: PyValue,
) -> PyResult<AdvancedSelection> {
    if shape.is_empty() {
        return Err(PyError::value_error("cannot index a 0-d array"));
    }
    let index = if runtime.kind(&index)? == PyKind::List {
        let empty = index
            .cast::<PySequence>(runtime)?
            .items(runtime)?
            .is_empty();
        construct(runtime, index, empty.then_some(PyArrayDtype::Int64))?
    } else {
        index
    };
    let indices = index.cast::<PyArray>(runtime)?;
    let (index_layout, index_dtype) = runtime.array_layout(indices)?;
    let mut coordinates = Vec::new();
    let result_shape = match index_dtype {
        PyArrayDtype::Bool if index_layout.shape == shape => {
            for_each_index(shape, |coordinate| {
                let selected = runtime.array_get(indices, coordinate)?;
                if boolean_index(runtime, selected)? {
                    push_coordinate(runtime, &mut coordinates, coordinate.to_vec())?;
                }
                Ok(())
            })?;
            vec![coordinates.len()]
        }
        PyArrayDtype::Bool
            if index_layout.shape.len() == 1 && index_layout.shape[0] == shape[0] =>
        {
            let mut selected_rows = 0usize;
            for row in 0..shape[0] {
                let selected = runtime.array_get(indices, &[row])?;
                if !boolean_index(runtime, selected)? {
                    continue;
                }
                selected_rows += 1;
                for_each_index(&shape[1..], |tail| {
                    let mut coordinate = vec![row];
                    coordinate.extend_from_slice(tail);
                    push_coordinate(runtime, &mut coordinates, coordinate)
                })?;
            }
            let mut result = vec![selected_rows];
            result.extend_from_slice(&shape[1..]);
            result
        }
        PyArrayDtype::Bool => {
            return Err(PyError::value_error(
                "boolean index must match the array or its first axis",
            ))
        }
        dtype
            if matches!(
                dtype_kind(dtype).class,
                NumericClass::Signed | NumericClass::Unsigned
            ) =>
        {
            for_each_index(&index_layout.shape, |index_coordinate| {
                let selected = runtime.array_get(indices, index_coordinate)?;
                let selected = integer_index(runtime, selected)?;
                let row = normalize_index(selected, shape[0])?;
                for_each_index(&shape[1..], |tail| {
                    let mut coordinate = vec![row];
                    coordinate.extend_from_slice(tail);
                    push_coordinate(runtime, &mut coordinates, coordinate)
                })
            })?;
            let mut result = index_layout.shape;
            result.extend_from_slice(&shape[1..]);
            result
        }
        _ => {
            return Err(PyError::type_error(
                "arrays used as indices must be of integer or boolean type",
            ))
        }
    };
    Ok(AdvancedSelection {
        coordinates,
        shape: result_shape,
    })
}

fn push_coordinate(
    runtime: &mut dyn PyRuntime,
    coordinates: &mut Vec<Vec<usize>>,
    coordinate: Vec<usize>,
) -> PyResult<()> {
    runtime.reserve_memory(
        coordinate
            .len()
            .checked_mul(std::mem::size_of::<usize>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<usize>>()))
            .ok_or_else(|| PyError::value_error("index selection is too large"))?,
    )?;
    coordinates.push(coordinate);
    Ok(())
}

fn binary(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    operation: PyBinaryOp,
    reflected: bool,
) -> PyResult<Option<PyValue>> {
    let (mut left, mut right) = if reflected {
        (right, left)
    } else {
        (left, right)
    };
    if matches!(runtime.kind(&left)?, PyKind::List | PyKind::Tuple) {
        left = construct(runtime, left, None)?;
    }
    if matches!(runtime.kind(&right)?, PyKind::List | PyKind::Tuple) {
        right = construct(runtime, right, None)?;
    }
    if runtime.kind(&left)? != PyKind::Array && runtime.kind(&right)? != PyKind::Array {
        return Ok(None);
    }
    let left_array = if runtime.kind(&left)? == PyKind::Array {
        Some(runtime.array_layout(left.cast(runtime)?)?)
    } else {
        None
    };
    let right_array = if runtime.kind(&right)? == PyKind::Array {
        Some(runtime.array_layout(right.cast(runtime)?)?)
    } else {
        None
    };
    let shape = broadcast_shape(
        left_array
            .as_ref()
            .map(|value| value.0.shape.as_slice())
            .unwrap_or(&[]),
        right_array
            .as_ref()
            .map(|value| value.0.shape.as_slice())
            .unwrap_or(&[]),
    )?;
    let left_dtype = operand_dtype(runtime, left, left_array.as_ref().map(|value| value.1))?;
    let right_dtype = operand_dtype(runtime, right, right_array.as_ref().map(|value| value.1))?;
    let left_is_weak = left_array.is_none() && registered_scalar(runtime, &left).is_none();
    let right_is_weak = right_array.is_none() && registered_scalar(runtime, &right).is_none();
    let promoted = promote_operands(left_dtype, right_dtype, left_is_weak, right_is_weak);
    let dtype =
        if operation == PyBinaryOp::Divide && dtype_kind(promoted).class != NumericClass::Float {
            PyArrayDtype::Float64
        } else {
            promoted
        };
    let count = element_count(&shape)?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    for_each_index(&shape, |index| {
        let left_value = broadcast_get(
            runtime,
            left,
            left_array.as_ref().map(|value| &value.0),
            index,
            &shape,
        )?;
        let right_value = broadcast_get(
            runtime,
            right,
            right_array.as_ref().map(|value| &value.0),
            index,
            &shape,
        )?;
        let left_value = convert(runtime, left_value, dtype)?;
        let right_value = convert(runtime, right_value, dtype)?;
        let value = runtime.binary_op(operation, left_value, right_value)?;
        values.push(value);
        Ok(())
    })?;
    runtime.new_array(values, shape, dtype).map(Some)
}

fn operand_dtype(
    runtime: &dyn PyRuntime,
    value: PyValue,
    array_dtype: Option<PyArrayDtype>,
) -> PyResult<PyArrayDtype> {
    if let Some(dtype) = array_dtype {
        return Ok(dtype);
    }
    if let Some(value) = registered_scalar(runtime, &value) {
        return Ok(value.dtype);
    }
    match runtime.kind(&value)? {
        PyKind::Bool => Ok(PyArrayDtype::Bool),
        PyKind::Int => Ok(PyArrayDtype::Int64),
        PyKind::Float => Ok(PyArrayDtype::Float64),
        _ => Err(PyError::type_error("numpy operand is not numeric")),
    }
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> PyResult<Vec<usize>> {
    let rank = left.len().max(right.len());
    let mut shape = vec![1; rank];
    for offset in 0..rank {
        let a = left
            .get(left.len().wrapping_sub(1 + offset))
            .copied()
            .unwrap_or(1);
        let b = right
            .get(right.len().wrapping_sub(1 + offset))
            .copied()
            .unwrap_or(1);
        if a != b && a != 1 && b != 1 {
            return Err(PyError::value_error(
                "operands could not be broadcast together",
            ));
        }
        shape[rank - 1 - offset] = a.max(b);
    }
    Ok(shape)
}

fn broadcast_get(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    layout: Option<&PyArrayLayout>,
    output_index: &[usize],
    output_shape: &[usize],
) -> PyResult<PyValue> {
    let Some(layout) = layout else {
        return Ok(value);
    };
    let leading = output_shape
        .len()
        .checked_sub(layout.shape.len())
        .ok_or_else(|| PyError::value_error("operand has more dimensions than the output"))?;
    let index = layout
        .shape
        .iter()
        .enumerate()
        .map(|(axis, length)| {
            if *length == 1 {
                0
            } else {
                output_index[leading + axis]
            }
        })
        .collect::<Vec<_>>();
    runtime.array_get(value.cast(runtime)?, &index)
}

macro_rules! binary_slots {
    ($normal:ident, $reflected:ident, $operation:expr) => {
        pub(crate) fn $normal(
            runtime: &mut dyn PyRuntime,
            left: PyValue,
            right: PyValue,
        ) -> PyResult<Option<PyValue>> {
            binary(runtime, left, right, $operation, false)
        }
        pub(crate) fn $reflected(
            runtime: &mut dyn PyRuntime,
            left: PyValue,
            right: PyValue,
        ) -> PyResult<Option<PyValue>> {
            binary(runtime, left, right, $operation, true)
        }
    };
}

binary_slots!(slot_add, slot_reflected_add, PyBinaryOp::Add);
binary_slots!(slot_subtract, slot_reflected_subtract, PyBinaryOp::Subtract);
binary_slots!(slot_multiply, slot_reflected_multiply, PyBinaryOp::Multiply);
binary_slots!(slot_divide, slot_reflected_divide, PyBinaryOp::Divide);

pub(crate) fn slot_matrix_multiply(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    let left = coerce_array(runtime, left)?;
    let right = coerce_array(runtime, right)?;
    matmul_arrays(runtime, left, right).map(Some)
}

pub(crate) fn slot_reflected_matrix_multiply(
    runtime: &mut dyn PyRuntime,
    right: PyValue,
    left: PyValue,
) -> PyResult<Option<PyValue>> {
    slot_matrix_multiply(runtime, left, right)
}

#[derive(Clone, Copy)]
enum CompareMap {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

fn comparison(
    runtime: &mut dyn PyRuntime,
    mut left: PyValue,
    mut right: PyValue,
    operation: CompareMap,
) -> PyResult<Option<PyValue>> {
    if matches!(runtime.kind(&left)?, PyKind::List | PyKind::Tuple) {
        left = construct(runtime, left, None)?;
    }
    if matches!(runtime.kind(&right)?, PyKind::List | PyKind::Tuple) {
        right = construct(runtime, right, None)?;
    }
    if runtime.kind(&left)? != PyKind::Array && runtime.kind(&right)? != PyKind::Array {
        return Ok(None);
    }
    let left_layout = if runtime.kind(&left)? == PyKind::Array {
        Some(runtime.array_layout(left.cast(runtime)?)?.0)
    } else {
        None
    };
    let right_layout = if runtime.kind(&right)? == PyKind::Array {
        Some(runtime.array_layout(right.cast(runtime)?)?.0)
    } else {
        None
    };
    let shape = broadcast_shape(
        left_layout
            .as_ref()
            .map_or(&[], |value| value.shape.as_slice()),
        right_layout
            .as_ref()
            .map_or(&[], |value| value.shape.as_slice()),
    )?;
    let count = element_count(&shape)?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    for_each_index(&shape, |index| {
        let left = broadcast_get(runtime, left, left_layout.as_ref(), index, &shape)?;
        let right = broadcast_get(runtime, right, right_layout.as_ref(), index, &shape)?;
        let left = scalar(runtime, left)?;
        let right = scalar(runtime, right)?;
        let ordering = left.compare(right);
        let result = match operation {
            CompareMap::Equal => ordering == Some(Ordering::Equal),
            CompareMap::NotEqual => ordering != Some(Ordering::Equal),
            CompareMap::Less => ordering == Some(Ordering::Less),
            CompareMap::LessEqual => {
                matches!(ordering, Some(Ordering::Less | Ordering::Equal))
            }
            CompareMap::Greater => ordering == Some(Ordering::Greater),
            CompareMap::GreaterEqual => {
                matches!(ordering, Some(Ordering::Greater | Ordering::Equal))
            }
        };
        values.push(pack_bool(runtime, result)?);
        Ok(())
    })?;
    runtime
        .new_array(values, shape, PyArrayDtype::Bool)
        .map(Some)
}

macro_rules! comparison_slots {
    ($(($name:ident, $operation:expr)),+ $(,)?) => {
        $(pub(crate) fn $name(
            runtime: &mut dyn PyRuntime,
            left: PyValue,
            right: PyValue,
        ) -> PyResult<Option<PyValue>> {
            comparison(runtime, left, right, $operation)
        })+
    };
}

comparison_slots!(
    (slot_equal, CompareMap::Equal),
    (slot_not_equal, CompareMap::NotEqual),
    (slot_less_than, CompareMap::Less),
    (slot_less_equal, CompareMap::LessEqual),
    (slot_greater_than, CompareMap::Greater),
    (slot_greater_equal, CompareMap::GreaterEqual),
);

#[derive(Clone, Copy)]
enum UnaryMap {
    Positive,
    Negative,
    Invert,
    Absolute,
    Sqrt,
    Exp,
    Log,
    Log2,
    Sin,
    Cos,
    Tan,
    Floor,
    Ceil,
    Rint,
    Sign,
    IsNan,
    IsInf,
}

fn unary_function(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    name: &str,
    operation: UnaryMap,
) -> PyResult {
    args.expect_positional(name, 1, 1)?;
    args.reject_keywords(name)?;
    let array = coerce_array(runtime, args.positional()[0])?;
    unary_array(runtime, array, operation)
}

fn unary_array(runtime: &mut dyn PyRuntime, array: PyArray, operation: UnaryMap) -> PyResult {
    let (layout, input_dtype) = runtime.array_layout(array)?;
    let float_output = matches!(
        operation,
        UnaryMap::Sqrt
            | UnaryMap::Exp
            | UnaryMap::Log
            | UnaryMap::Log2
            | UnaryMap::Sin
            | UnaryMap::Cos
            | UnaryMap::Tan
            | UnaryMap::Floor
            | UnaryMap::Ceil
            | UnaryMap::Rint
    );
    if input_dtype == PyArrayDtype::Bool && matches!(operation, UnaryMap::Negative) {
        return Err(PyError::type_error(
            "this unary operation is not defined for boolean arrays",
        ));
    }
    let dtype = if matches!(operation, UnaryMap::IsNan | UnaryMap::IsInf) {
        PyArrayDtype::Bool
    } else if float_output {
        if input_dtype == PyArrayDtype::Float32 {
            PyArrayDtype::Float32
        } else {
            PyArrayDtype::Float64
        }
    } else {
        input_dtype
    };
    let count = element_count(&layout.shape)?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    for_each_index(&layout.shape, |index| {
        let raw = runtime.array_get(array, index)?;
        let value = scalar(runtime, raw)?;
        let mapped_value = match operation {
            UnaryMap::Positive => value,
            UnaryMap::Negative => match value.value {
                ScalarValue::Signed(value) => Scalar {
                    dtype,
                    value: ScalarValue::Signed(value.wrapping_neg()),
                },
                ScalarValue::Unsigned(value) => Scalar {
                    dtype,
                    value: ScalarValue::Unsigned(0u128.wrapping_sub(value)),
                },
                ScalarValue::Float(value) => Scalar {
                    dtype,
                    value: ScalarValue::Float(-value),
                },
                ScalarValue::Bool(_) => unreachable!(),
            },
            UnaryMap::Invert => match value.value {
                ScalarValue::Bool(value) => Scalar {
                    dtype,
                    value: ScalarValue::Bool(!value),
                },
                ScalarValue::Signed(value) => Scalar {
                    dtype,
                    value: ScalarValue::Signed(!value),
                },
                ScalarValue::Unsigned(value) => Scalar {
                    dtype,
                    value: ScalarValue::Unsigned(!value),
                },
                ScalarValue::Float(_) => {
                    return Err(PyError::type_error(
                        "bitwise invert is not defined for floating arrays",
                    ))
                }
            },
            UnaryMap::Absolute => match value.value {
                ScalarValue::Signed(value) => Scalar {
                    dtype,
                    value: ScalarValue::Signed(value.wrapping_abs()),
                },
                ScalarValue::Float(value) => Scalar {
                    dtype,
                    value: ScalarValue::Float(value.abs()),
                },
                ScalarValue::Bool(_) | ScalarValue::Unsigned(_) => value,
            },
            UnaryMap::Sqrt => float_scalar(dtype, value.as_f64().sqrt()),
            UnaryMap::Exp => float_scalar(dtype, value.as_f64().exp()),
            UnaryMap::Log => float_scalar(dtype, value.as_f64().ln()),
            UnaryMap::Log2 => float_scalar(dtype, value.as_f64().log2()),
            UnaryMap::Sin => float_scalar(dtype, value.as_f64().sin()),
            UnaryMap::Cos => float_scalar(dtype, value.as_f64().cos()),
            UnaryMap::Tan => float_scalar(dtype, value.as_f64().tan()),
            UnaryMap::Floor => float_scalar(dtype, value.as_f64().floor()),
            UnaryMap::Ceil => float_scalar(dtype, value.as_f64().ceil()),
            UnaryMap::Rint => float_scalar(dtype, value.as_f64().round_ties_even()),
            UnaryMap::Sign => {
                let signed = value.as_f64();
                let sign = if signed.is_nan() {
                    f64::NAN
                } else if signed > 0.0 {
                    1.0
                } else if signed < 0.0 {
                    -1.0
                } else {
                    signed
                };
                cast_scalar(runtime, float_scalar(PyArrayDtype::Float64, sign), dtype)?
            }
            UnaryMap::IsNan => Scalar {
                dtype,
                value: ScalarValue::Bool(value.as_f64().is_nan()),
            },
            UnaryMap::IsInf => Scalar {
                dtype,
                value: ScalarValue::Bool(value.as_f64().is_infinite()),
            },
        };
        values.push(pack_scalar(runtime, mapped_value, dtype)?);
        Ok(())
    })?;
    runtime.new_array(values, layout.shape, dtype)
}

pub(crate) fn slot_positive(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = value.cast(runtime)?;
    unary_array(runtime, array, UnaryMap::Positive).map(Some)
}

pub(crate) fn slot_negative(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = value.cast(runtime)?;
    unary_array(runtime, array, UnaryMap::Negative).map(Some)
}

pub(crate) fn slot_invert(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = value.cast(runtime)?;
    unary_array(runtime, array, UnaryMap::Invert).map(Some)
}

pub(crate) fn slot_absolute(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Option<PyValue>> {
    let array = value.cast(runtime)?;
    unary_array(runtime, array, UnaryMap::Absolute).map(Some)
}

fn pack_scalar(runtime: &dyn PyRuntime, value: Scalar, dtype: PyArrayDtype) -> PyResult<PyValue> {
    pack_wrapping(runtime, value, dtype)
}

fn float_scalar(dtype: PyArrayDtype, value: f64) -> Scalar {
    Scalar {
        dtype,
        value: ScalarValue::Float(value),
    }
}

macro_rules! unary_functions {
    ($(($name:ident, $operation:expr)),+ $(,)?) => {
        $(fn $name(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
            unary_function(runtime, args, concat!("numpy.", stringify!($name)), $operation)
        })+
    };
}

unary_functions!(
    (negative, UnaryMap::Negative),
    (absolute, UnaryMap::Absolute),
    (sqrt, UnaryMap::Sqrt),
    (exp, UnaryMap::Exp),
    (log, UnaryMap::Log),
    (log2, UnaryMap::Log2),
    (sin, UnaryMap::Sin),
    (cos, UnaryMap::Cos),
    (tan, UnaryMap::Tan),
    (floor, UnaryMap::Floor),
    (ceil, UnaryMap::Ceil),
    (rint, UnaryMap::Rint),
    (sign, UnaryMap::Sign),
    (isnan, UnaryMap::IsNan),
    (isinf, UnaryMap::IsInf),
);

fn minimum(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    extreme(runtime, args, "numpy.minimum", false)
}

fn maximum(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    extreme(runtime, args, "numpy.maximum", true)
}

fn extreme(runtime: &mut dyn PyRuntime, args: CallArgs, name: &str, maximum: bool) -> PyResult {
    args.expect_positional(name, 2, 2)?;
    args.reject_keywords(name)?;
    elementwise_extreme(runtime, args.positional()[0], args.positional()[1], maximum)
}

fn elementwise_extreme(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    maximum: bool,
) -> PyResult {
    let left_array = (runtime.kind(&left)? == PyKind::Array)
        .then(|| left.cast::<PyArray>(runtime))
        .transpose()?;
    let right_array = (runtime.kind(&right)? == PyKind::Array)
        .then(|| right.cast::<PyArray>(runtime))
        .transpose()?;
    let left_info = left_array
        .map(|array| runtime.array_layout(array))
        .transpose()?;
    let right_info = right_array
        .map(|array| runtime.array_layout(array))
        .transpose()?;
    let shape = broadcast_shape(
        left_info
            .as_ref()
            .map_or(&[], |value| value.0.shape.as_slice()),
        right_info
            .as_ref()
            .map_or(&[], |value| value.0.shape.as_slice()),
    )?;
    let left_dtype = operand_dtype(runtime, left, left_info.as_ref().map(|value| value.1))?;
    let right_dtype = operand_dtype(runtime, right, right_info.as_ref().map(|value| value.1))?;
    let dtype = promote_operands(
        left_dtype,
        right_dtype,
        left_info.is_none() && registered_scalar(runtime, &left).is_none(),
        right_info.is_none() && registered_scalar(runtime, &right).is_none(),
    );
    let count = element_count(&shape)?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    for_each_index(&shape, |index| {
        let left = broadcast_get(
            runtime,
            left,
            left_info.as_ref().map(|value| &value.0),
            index,
            &shape,
        )?;
        let right = broadcast_get(
            runtime,
            right,
            right_info.as_ref().map(|value| &value.0),
            index,
            &shape,
        )?;
        let left_scalar = scalar(runtime, left)?;
        let right_scalar = scalar(runtime, right)?;
        let selected = match left_scalar.compare(right_scalar) {
            Some(Ordering::Greater) if maximum => left,
            Some(Ordering::Less) if !maximum => left,
            Some(Ordering::Equal) => left,
            None if left_scalar.is_nan() => left,
            None if right_scalar.is_nan() => right,
            _ => right,
        };
        values.push(convert(runtime, selected, dtype)?);
        Ok(())
    })?;
    runtime.new_array(values, shape, dtype)
}

fn clip(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.clip", 3, 3)?;
    args.reject_keywords("numpy.clip")?;
    let lower = elementwise_extreme(runtime, args.positional()[0], args.positional()[1], true)?;
    elementwise_extreme(runtime, lower, args.positional()[2], false)
}

fn where_(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.where", 3, 3)?;
    args.reject_keywords("numpy.where")?;
    let condition = coerce_array(runtime, args.positional()[0])?;
    let condition_layout = runtime.array_layout(condition)?.0;
    let left = args.positional()[1];
    let right = args.positional()[2];
    let left_info = if runtime.kind(&left)? == PyKind::Array {
        Some(runtime.array_layout(left.cast(runtime)?)?)
    } else {
        None
    };
    let right_info = if runtime.kind(&right)? == PyKind::Array {
        Some(runtime.array_layout(right.cast(runtime)?)?)
    } else {
        None
    };
    let shape = broadcast_shape(
        &condition_layout.shape,
        &broadcast_shape(
            left_info
                .as_ref()
                .map_or(&[], |value| value.0.shape.as_slice()),
            right_info
                .as_ref()
                .map_or(&[], |value| value.0.shape.as_slice()),
        )?,
    )?;
    let left_dtype = operand_dtype(runtime, left, left_info.as_ref().map(|value| value.1))?;
    let right_dtype = operand_dtype(runtime, right, right_info.as_ref().map(|value| value.1))?;
    let dtype = promote_operands(
        left_dtype,
        right_dtype,
        left_info.is_none() && registered_scalar(runtime, &left).is_none(),
        right_info.is_none() && registered_scalar(runtime, &right).is_none(),
    );
    let count = element_count(&shape)?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    for_each_index(&shape, |index| {
        let condition_value = broadcast_get(
            runtime,
            Value::Object(condition.object_id()),
            Some(&condition_layout),
            index,
            &shape,
        )?;
        let (value, layout) = if scalar(runtime, condition_value)?.truth() {
            (left, left_info.as_ref().map(|value| &value.0))
        } else {
            (right, right_info.as_ref().map(|value| &value.0))
        };
        let value = broadcast_get(runtime, value, layout, index, &shape)?;
        values.push(convert(runtime, value, dtype)?);
        Ok(())
    })?;
    runtime.new_array(values, shape, dtype)
}

fn method_tolist(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("ndarray.tolist", 0, 0)?;
    args.reject_keywords("ndarray.tolist")?;
    let array = receiver.cast::<PyArray>(runtime)?;
    let (layout, _) = runtime.array_layout(array)?;
    tolist_axis(runtime, array, &layout.shape, &mut Vec::new())
}

fn tolist_axis(
    runtime: &mut dyn PyRuntime,
    array: PyArray,
    shape: &[usize],
    prefix: &mut Vec<usize>,
) -> PyResult {
    if prefix.len() == shape.len() {
        let value = runtime.array_get(array, prefix)?;
        return match registered_scalar(runtime, &value) {
            Some(value) => scalar_to_python(runtime, value),
            None => Ok(value),
        };
    }
    let axis = prefix.len();
    reserve_values(runtime, shape[axis])?;
    let mut values = Vec::with_capacity(shape[axis]);
    for index in 0..shape[axis] {
        runtime.charge_cpu(1)?;
        prefix.push(index);
        values.push(tolist_axis(runtime, array, shape, prefix)?);
        prefix.pop();
    }
    runtime.new_list(values)
}

fn scalar_to_python(runtime: &mut dyn PyRuntime, value: Scalar) -> PyResult {
    match value.value {
        ScalarValue::Bool(value) => Ok(Value::Bool(value)),
        ScalarValue::Signed(value) => i64::try_from(value)
            .map(Value::Int)
            .map_err(|_| PyError::overflow_error("signed scalar exceeds Python integer range")),
        ScalarValue::Unsigned(value) => match i64::try_from(value) {
            Ok(value) => Ok(Value::Int(value)),
            Err(_) => runtime.new_integer(&value.to_string()),
        },
        ScalarValue::Float(value) => Ok(Value::Float(value)),
    }
}

fn method_copy(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("ndarray.copy", 0, 0)?;
    args.reject_keywords("ndarray.copy")?;
    let array = receiver.cast(runtime)?;
    copy_array(runtime, array)
}

fn method_astype(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("ndarray.astype", 1, 1)?;
    args.reject_keywords("ndarray.astype")?;
    let array = receiver.cast::<PyArray>(runtime)?;
    let dtype = parse_dtype(runtime, args.positional()[0])?;
    let (layout, _) = runtime.array_layout(array)?;
    let count = element_count(&layout.shape)?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    for_each_index(&layout.shape, |index| {
        let value = runtime.array_get(array, index)?;
        values.push(convert(runtime, value, dtype)?);
        Ok(())
    })?;
    runtime.new_array(values, layout.shape, dtype)
}

fn copy_array(runtime: &mut dyn PyRuntime, array: PyArray) -> PyResult {
    let (layout, dtype) = runtime.array_layout(array)?;
    let count = element_count(&layout.shape)?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    for_each_index(&layout.shape, |index| {
        values.push(runtime.array_get(array, index)?);
        Ok(())
    })?;
    runtime.new_array(values, layout.shape, dtype)
}

fn method_reshape(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("ndarray.reshape", 1, usize::MAX)?;
    args.reject_keywords("ndarray.reshape")?;
    let array = receiver.cast::<PyArray>(runtime)?;
    let total = element_count(&runtime.array_layout(array)?.0.shape)?;
    let requested = if args.positional().len() == 1 {
        parse_signed_shape(runtime, args.positional()[0])?
    } else {
        args.positional()
            .iter()
            .map(|value| {
                let PyIndex(value) = (*value).cast(runtime)?;
                Ok(value)
            })
            .collect::<PyResult<Vec<_>>>()?
    };
    let shape = resolve_reshape_shape(requested, total)?;
    reshape(runtime, array, shape)
}

fn parse_signed_shape(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<i64>> {
    if let Some(value) = runtime.int_value(&value) {
        return Ok(vec![value]);
    }
    value
        .cast::<PySequence>(runtime)?
        .items(runtime)?
        .into_iter()
        .map(|item| item.cast::<PyIndex>(runtime).map(|value| value.0))
        .collect()
}

fn resolve_reshape_shape(requested: Vec<i64>, total: usize) -> PyResult<Vec<usize>> {
    let mut inferred = None;
    let mut known = 1usize;
    let mut shape = Vec::with_capacity(requested.len());
    for value in requested {
        if value == -1 {
            if inferred.replace(shape.len()).is_some() {
                return Err(PyError::value_error(
                    "can only specify one unknown dimension",
                ));
            }
            shape.push(1);
        } else {
            let value = dimension(value)?;
            known = known
                .checked_mul(value)
                .ok_or_else(|| PyError::value_error("array is too large"))?;
            shape.push(value);
        }
    }
    if let Some(axis) = inferred {
        if known == 0 || !total.is_multiple_of(known) {
            return Err(PyError::value_error("cannot infer reshape dimension"));
        }
        shape[axis] = total / known;
    }
    Ok(shape)
}

fn reshape(runtime: &mut dyn PyRuntime, array: PyArray, shape: Vec<usize>) -> PyResult {
    let (layout, _) = runtime.array_layout(array)?;
    if element_count(&shape)? != element_count(&layout.shape)? {
        return Err(PyError::value_error(
            "cannot reshape array to requested shape",
        ));
    }
    if contiguous_strides(&layout.shape)? == layout.strides {
        return runtime.new_array_view(
            array,
            PyArrayLayout {
                strides: contiguous_strides(&shape)?,
                shape,
                offset: layout.offset,
            },
        );
    }
    let copied = copy_array(runtime, array)?.cast(runtime)?;
    let (copied_layout, _) = runtime.array_layout(copied)?;
    runtime.new_array_view(
        copied,
        PyArrayLayout {
            strides: contiguous_strides(&shape)?,
            shape,
            offset: copied_layout.offset,
        },
    )
}

fn module_reshape(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.reshape", 2, 2)?;
    args.reject_keywords("numpy.reshape")?;
    let array = coerce_array(runtime, args.positional()[0])?;
    let total = element_count(&runtime.array_layout(array)?.0.shape)?;
    let shape = resolve_reshape_shape(parse_signed_shape(runtime, args.positional()[1])?, total)?;
    reshape(runtime, array, shape)
}

fn method_transpose(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.reject_keywords("ndarray.transpose")?;
    let axes = if args.positional().is_empty() {
        None
    } else {
        Some(
            if args.positional().len() == 1
                && matches!(
                    runtime.kind(&args.positional()[0])?,
                    PyKind::List | PyKind::Tuple
                )
            {
                parse_shape(runtime, args.positional()[0])?
            } else {
                args.positional()
                    .iter()
                    .map(|value| {
                        let PyIndex(value) = (*value).cast(runtime)?;
                        dimension(value)
                    })
                    .collect::<PyResult<Vec<_>>>()?
            },
        )
    };
    let array = receiver.cast(runtime)?;
    transpose(runtime, array, axes)
}

pub(in crate::python) fn transpose(
    runtime: &mut dyn PyRuntime,
    array: PyArray,
    axes: Option<Vec<usize>>,
) -> PyResult {
    let (layout, _) = runtime.array_layout(array)?;
    let axes = axes.unwrap_or_else(|| (0..layout.shape.len()).rev().collect());
    if axes.len() != layout.shape.len() {
        return Err(PyError::value_error("axes don't match array"));
    }
    let mut seen = vec![false; axes.len()];
    for axis in &axes {
        if *axis >= axes.len() || seen[*axis] {
            return Err(PyError::value_error("invalid transpose axes"));
        }
        seen[*axis] = true;
    }
    runtime.new_array_view(
        array,
        PyArrayLayout {
            shape: axes.iter().map(|axis| layout.shape[*axis]).collect(),
            strides: axes.iter().map(|axis| layout.strides[*axis]).collect(),
            offset: layout.offset,
        },
    )
}

fn module_transpose(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.transpose", 1, 2)?;
    args.reject_keywords("numpy.transpose")?;
    let axes = args
        .positional()
        .get(1)
        .copied()
        .map(|value| parse_shape(runtime, value))
        .transpose()?;
    let array = coerce_array(runtime, args.positional()[0])?;
    transpose(runtime, array, axes)
}

fn method_flatten(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("ndarray.flatten", 0, 0)?;
    args.reject_keywords("ndarray.flatten")?;
    let array = receiver.cast::<PyArray>(runtime)?;
    let count = element_count(&runtime.array_layout(array)?.0.shape)?;
    let copied = copy_array(runtime, array)?.cast(runtime)?;
    reshape(runtime, copied, vec![count])
}

fn method_ravel(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("ndarray.ravel", 0, 0)?;
    args.reject_keywords("ndarray.ravel")?;
    let array = receiver.cast::<PyArray>(runtime)?;
    let layout = runtime.array_layout(array)?.0;
    let count = element_count(&layout.shape)?;
    reshape(runtime, array, vec![count])
}

fn method_squeeze(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    squeeze(runtime, array, args, "ndarray.squeeze")
}

fn module_squeeze(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.squeeze", 1, 2)?;
    args.reject_unknown_keywords("numpy.squeeze", &["axis"])?;
    let array = coerce_array(runtime, args.positional()[0])?;
    let forwarded = CallArgs::new(args.positional()[1..].to_vec(), args.keywords().to_vec());
    squeeze(runtime, array, forwarded, "numpy.squeeze")
}

fn squeeze(runtime: &mut dyn PyRuntime, array: PyArray, args: CallArgs, name: &str) -> PyResult {
    args.expect_positional(name, 0, 1)?;
    args.reject_unknown_keywords(name, &["axis"])?;
    let (layout, _) = runtime.array_layout(array)?;
    let selected = args
        .positional()
        .first()
        .copied()
        .or(args.keyword(name, "axis")?.copied())
        .map(|value| {
            let PyIndex(axis) = value.cast(runtime)?;
            normalize_axis(axis, layout.shape.len()).map(Some)
        })
        .transpose()?
        .flatten();
    if let Some(axis) = selected {
        if layout.shape[axis] != 1 {
            return Err(PyError::value_error(
                "cannot squeeze an axis whose size is not one",
            ));
        }
    }
    let retained = (0..layout.shape.len())
        .filter(|axis| selected.map_or(layout.shape[*axis] != 1, |selected| selected != *axis))
        .collect::<Vec<_>>();
    runtime.new_array_view(
        array,
        PyArrayLayout {
            shape: retained.iter().map(|axis| layout.shape[*axis]).collect(),
            strides: retained.iter().map(|axis| layout.strides[*axis]).collect(),
            offset: layout.offset,
        },
    )
}

fn expand_dims(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.expand_dims", 2, 2)?;
    args.reject_keywords("numpy.expand_dims")?;
    let array = coerce_array(runtime, args.positional()[0])?;
    let (mut layout, _) = runtime.array_layout(array)?;
    let PyIndex(axis) = args.positional()[1].cast(runtime)?;
    let rank = layout
        .shape
        .len()
        .checked_add(1)
        .ok_or_else(|| PyError::overflow_error("array rank overflow"))?;
    let axis = normalize_axis(axis, rank)?;
    let stride = layout
        .strides
        .get(axis)
        .copied()
        .unwrap_or(1)
        .checked_mul(
            isize::try_from(layout.shape.get(axis).copied().unwrap_or(1))
                .map_err(|_| PyError::value_error("array stride overflow"))?,
        )
        .ok_or_else(|| PyError::value_error("array stride overflow"))?;
    layout.shape.insert(axis, 1);
    layout.strides.insert(axis, stride);
    runtime.new_array_view(array, layout)
}

fn method_swapaxes(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("ndarray.swapaxes", 2, 2)?;
    args.reject_keywords("ndarray.swapaxes")?;
    let array = receiver.cast(runtime)?;
    swapaxes(runtime, array, args.positional()[0], args.positional()[1])
}

fn module_swapaxes(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.swapaxes", 3, 3)?;
    args.reject_keywords("numpy.swapaxes")?;
    let array = coerce_array(runtime, args.positional()[0])?;
    swapaxes(runtime, array, args.positional()[1], args.positional()[2])
}

fn swapaxes(
    runtime: &mut dyn PyRuntime,
    array: PyArray,
    first: PyValue,
    second: PyValue,
) -> PyResult {
    let (layout, _) = runtime.array_layout(array)?;
    let PyIndex(first) = first.cast(runtime)?;
    let PyIndex(second) = second.cast(runtime)?;
    let first = normalize_axis(first, layout.shape.len())?;
    let second = normalize_axis(second, layout.shape.len())?;
    let mut axes = (0..layout.shape.len()).collect::<Vec<_>>();
    axes.swap(first, second);
    transpose(runtime, array, Some(axes))
}

fn broadcast_to(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.broadcast_to", 2, 2)?;
    args.reject_keywords("numpy.broadcast_to")?;
    let array = coerce_array(runtime, args.positional()[0])?;
    let target = parse_shape(runtime, args.positional()[1])?;
    let (layout, _) = runtime.array_layout(array)?;
    if target.len() < layout.shape.len() {
        return Err(PyError::value_error("cannot broadcast to fewer dimensions"));
    }
    let leading = target.len() - layout.shape.len();
    let mut strides = vec![0; leading];
    for (source, destination) in layout.shape.iter().zip(&target[leading..]) {
        if source != destination && *source != 1 {
            return Err(PyError::value_error(
                "operands could not be broadcast together",
            ));
        }
    }
    strides.extend(
        layout
            .shape
            .iter()
            .zip(&layout.strides)
            .map(|(length, stride)| if *length == 1 { 0 } else { *stride }),
    );
    runtime.new_array_view(
        array,
        PyArrayLayout {
            shape: target,
            strides,
            offset: layout.offset,
        },
    )
}

fn array_sequence(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<PyArray>> {
    let items = value.cast::<PySequence>(runtime)?.items(runtime)?;
    if items.is_empty() {
        return Err(PyError::value_error("need at least one array to join"));
    }
    items
        .into_iter()
        .map(|value| coerce_array(runtime, value))
        .collect()
}

fn concatenate(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.concatenate", 1, 2)?;
    args.reject_unknown_keywords("numpy.concatenate", &["axis"])?;
    let arrays = array_sequence(runtime, args.positional()[0])?;
    let axis = args
        .positional()
        .get(1)
        .copied()
        .or(args.keyword("numpy.concatenate", "axis")?.copied())
        .unwrap_or(Value::Int(0));
    concatenate_arrays(runtime, &arrays, axis)
}

fn concatenate_arrays(
    runtime: &mut dyn PyRuntime,
    arrays: &[PyArray],
    axis_value: PyValue,
) -> PyResult {
    let layouts = arrays
        .iter()
        .map(|array| runtime.array_layout(*array))
        .collect::<PyResult<Vec<_>>>()?;
    let rank = layouts[0].0.shape.len();
    let PyIndex(axis) = axis_value.cast(runtime)?;
    let axis = normalize_axis(axis, rank)?;
    let dtype = layouts.iter().fold(PyArrayDtype::Bool, |dtype, value| {
        promote_dtype(dtype, value.1)
    });
    let mut shape = layouts[0].0.shape.clone();
    shape[axis] = 0;
    for (layout, _) in &layouts {
        if layout.shape.len() != rank
            || layout
                .shape
                .iter()
                .enumerate()
                .any(|(candidate, length)| candidate != axis && *length != shape[candidate])
        {
            return Err(PyError::value_error(
                "all input arrays must have matching dimensions",
            ));
        }
        shape[axis] = shape[axis]
            .checked_add(layout.shape[axis])
            .ok_or_else(|| PyError::value_error("array is too large"))?;
    }
    let count = element_count(&shape)?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    for_each_index(&shape, |output_index| {
        let mut selected = output_index[axis];
        for (array, (layout, _)) in arrays.iter().zip(&layouts) {
            if selected < layout.shape[axis] {
                let mut input_index = output_index.to_vec();
                input_index[axis] = selected;
                let value = runtime.array_get(*array, &input_index)?;
                values.push(convert(runtime, value, dtype)?);
                return Ok(());
            }
            selected -= layout.shape[axis];
        }
        Err(PyError::runtime_error("concatenate index escaped inputs"))
    })?;
    runtime.new_array(values, shape, dtype)
}

fn stack(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.stack", 1, 2)?;
    args.reject_unknown_keywords("numpy.stack", &["axis"])?;
    let arrays = array_sequence(runtime, args.positional()[0])?;
    let first_shape = runtime.array_layout(arrays[0])?.0.shape;
    for array in arrays.iter().skip(1) {
        if runtime.array_layout(*array)?.0.shape != first_shape {
            return Err(PyError::value_error(
                "all input arrays must have the same shape",
            ));
        }
    }
    let axis_value = args
        .positional()
        .get(1)
        .copied()
        .or(args.keyword("numpy.stack", "axis")?.copied())
        .unwrap_or(Value::Int(0));
    let PyIndex(axis) = axis_value.cast(runtime)?;
    let rank = first_shape
        .len()
        .checked_add(1)
        .ok_or_else(|| PyError::overflow_error("array rank overflow"))?;
    let axis = normalize_axis(axis, rank)?;
    let mut expanded = Vec::with_capacity(arrays.len());
    for array in arrays {
        let (mut layout, _) = runtime.array_layout(array)?;
        layout.shape.insert(axis, 1);
        layout.strides.insert(axis, 0);
        expanded.push(runtime.new_array_view(array, layout)?.cast(runtime)?);
    }
    concatenate_arrays(runtime, &expanded, Value::Int(axis as i64))
}

fn vstack(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.vstack", 1, 1)?;
    args.reject_keywords("numpy.vstack")?;
    let arrays = array_sequence(runtime, args.positional()[0])?;
    let mut promoted = Vec::with_capacity(arrays.len());
    for array in arrays {
        let layout = runtime.array_layout(array)?.0;
        promoted.push(if layout.shape.len() == 1 {
            reshape(runtime, array, vec![1, layout.shape[0]])?.cast(runtime)?
        } else {
            array
        });
    }
    concatenate_arrays(runtime, &promoted, Value::Int(0))
}

fn hstack(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.hstack", 1, 1)?;
    args.reject_keywords("numpy.hstack")?;
    let arrays = array_sequence(runtime, args.positional()[0])?;
    let axis = if runtime.array_layout(arrays[0])?.0.shape.len() == 1 {
        0
    } else {
        1
    };
    concatenate_arrays(runtime, &arrays, Value::Int(axis))
}

fn method_sum(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    reduce(runtime, array, args, Reduction::Sum, "ndarray.sum")
}
fn method_prod(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    reduce(runtime, array, args, Reduction::Product, "ndarray.prod")
}
fn method_mean(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    reduce(runtime, array, args, Reduction::Mean, "ndarray.mean")
}
fn method_min(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    reduce(runtime, array, args, Reduction::Min, "ndarray.min")
}
fn method_max(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    reduce(runtime, array, args, Reduction::Max, "ndarray.max")
}

fn module_sum(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_reduce(runtime, args, Reduction::Sum, "numpy.sum")
}
fn module_prod(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_reduce(runtime, args, Reduction::Product, "numpy.prod")
}
fn module_mean(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_reduce(runtime, args, Reduction::Mean, "numpy.mean")
}
fn module_min(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_reduce(runtime, args, Reduction::Min, "numpy.min")
}
fn module_max(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_reduce(runtime, args, Reduction::Max, "numpy.max")
}

fn method_var(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    statistic(runtime, array, args, Statistic::Variance, "ndarray.var")
}

fn method_std(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    statistic(runtime, array, args, Statistic::StdDev, "ndarray.std")
}

fn method_all(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    statistic(runtime, array, args, Statistic::All, "ndarray.all")
}

fn method_any(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    statistic(runtime, array, args, Statistic::Any, "ndarray.any")
}

fn method_argmin(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    arg_reduce(runtime, array, args, false, "ndarray.argmin")
}

fn method_argmax(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    arg_reduce(runtime, array, args, true, "ndarray.argmax")
}

fn method_cumsum(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    cumulative(runtime, array, args, false, "ndarray.cumsum")
}

fn method_cumprod(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let array = receiver.cast(runtime)?;
    cumulative(runtime, array, args, true, "ndarray.cumprod")
}

fn module_var(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_statistic(runtime, args, Statistic::Variance, "numpy.var")
}

fn module_std(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_statistic(runtime, args, Statistic::StdDev, "numpy.std")
}

fn module_median(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_statistic(runtime, args, Statistic::Median, "numpy.median")
}

fn module_all(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_statistic(runtime, args, Statistic::All, "numpy.all")
}

fn module_any(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_statistic(runtime, args, Statistic::Any, "numpy.any")
}

fn module_argmin(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_arg_reduce(runtime, args, false, "numpy.argmin")
}

fn module_argmax(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_arg_reduce(runtime, args, true, "numpy.argmax")
}

fn module_cumsum(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_cumulative(runtime, args, false, "numpy.cumsum")
}

fn module_cumprod(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    module_cumulative(runtime, args, true, "numpy.cumprod")
}

fn module_arg_reduce(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    maximum: bool,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 1, 2)?;
    args.reject_unknown_keywords(name, &["axis"])?;
    let array = coerce_array(runtime, args.positional()[0])?;
    let forwarded = CallArgs::new(args.positional()[1..].to_vec(), args.keywords().to_vec());
    arg_reduce(runtime, array, forwarded, maximum, name)
}

fn arg_reduce(
    runtime: &mut dyn PyRuntime,
    array: PyArray,
    args: CallArgs,
    maximum: bool,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 0, 1)?;
    args.reject_unknown_keywords(name, &["axis"])?;
    let shape = runtime.array_layout(array)?.0.shape;
    let axis_value = args
        .positional()
        .first()
        .copied()
        .or(args.keyword(name, "axis")?.copied());
    if let Some(axis_value) = axis_value {
        let PyIndex(axis) = axis_value.cast(runtime)?;
        let axis = normalize_axis(axis, shape.len())?;
        let axis_length = shape[axis];
        if axis_length == 0 {
            return Err(PyError::value_error("arg reduction of an empty sequence"));
        }
        let mut output_shape = shape.clone();
        output_shape.remove(axis);
        let count = element_count(&output_shape)?;
        reserve_values(runtime, count)?;
        let mut values = Vec::with_capacity(count);
        for_each_index(&output_shape, |output_index| {
            let mut winner = 0usize;
            let mut index = output_index.to_vec();
            index.insert(axis, 0);
            let first = runtime.array_get(array, &index)?;
            let mut best = scalar(runtime, first)?;
            for selected in 1..axis_length {
                index[axis] = selected;
                let raw = runtime.array_get(array, &index)?;
                let candidate = scalar(runtime, raw)?;
                let ordering = candidate.compare(best);
                if (maximum && ordering == Some(Ordering::Greater))
                    || (!maximum && ordering == Some(Ordering::Less))
                {
                    best = candidate;
                    winner = selected;
                }
            }
            values.push(pack_index(runtime, winner)?);
            Ok(())
        })?;
        return runtime.new_array(values, output_shape, PyArrayDtype::Int64);
    }
    let mut winner = None;
    let mut best = None;
    let mut position = 0usize;
    for_each_index(&shape, |index| {
        let raw = runtime.array_get(array, index)?;
        let candidate = scalar(runtime, raw)?;
        let ordering = best.and_then(|best| candidate.compare(best));
        if winner.is_none()
            || (maximum && ordering == Some(Ordering::Greater))
            || (!maximum && ordering == Some(Ordering::Less))
        {
            winner = Some(position);
            best = Some(candidate);
        }
        position += 1;
        Ok(())
    })?;
    pack_index(
        runtime,
        winner.ok_or_else(|| PyError::value_error("arg reduction of an empty sequence"))?,
    )
}

fn module_cumulative(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    product: bool,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 1, 2)?;
    args.reject_unknown_keywords(name, &["axis"])?;
    let array = coerce_array(runtime, args.positional()[0])?;
    let forwarded = CallArgs::new(args.positional()[1..].to_vec(), args.keywords().to_vec());
    cumulative(runtime, array, forwarded, product, name)
}

fn cumulative(
    runtime: &mut dyn PyRuntime,
    array: PyArray,
    args: CallArgs,
    product: bool,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 0, 1)?;
    args.reject_unknown_keywords(name, &["axis"])?;
    let (layout, dtype) = runtime.array_layout(array)?;
    let axis_value = args
        .positional()
        .first()
        .copied()
        .or(args.keyword(name, "axis")?.copied());
    let (source, shape, axis) = if let Some(axis) = axis_value {
        let PyIndex(axis) = axis.cast(runtime)?;
        let rank = layout.shape.len();
        (array, layout.shape, normalize_axis(axis, rank)?)
    } else {
        let count = element_count(&layout.shape)?;
        let flattened = copy_array(runtime, array)?.cast(runtime)?;
        let flattened = reshape(runtime, flattened, vec![count])?.cast(runtime)?;
        (flattened, vec![count], 0)
    };
    let reduction = if product {
        Reduction::Product
    } else {
        Reduction::Sum
    };
    let output_dtype = reduction_dtype(dtype, reduction);
    let count = element_count(&shape)?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    for_each_index(&shape, |index| {
        let mut total = reduction_identity(runtime, output_dtype, reduction)?;
        for selected in 0..=index[axis] {
            let mut source_index = index.to_vec();
            source_index[axis] = selected;
            let value = runtime.array_get(source, &source_index)?;
            total = runtime.binary_op(
                if product {
                    PyBinaryOp::Multiply
                } else {
                    PyBinaryOp::Add
                },
                total,
                value,
            )?;
        }
        values.push(total);
        Ok(())
    })?;
    runtime.new_array(values, shape, output_dtype)
}

#[derive(Clone, Copy)]
enum Statistic {
    Variance,
    StdDev,
    Median,
    All,
    Any,
}

fn module_statistic(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    statistic_kind: Statistic,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 1, 2)?;
    args.reject_unknown_keywords(name, &["axis"])?;
    let array = coerce_array(runtime, args.positional()[0])?;
    let forwarded = CallArgs::new(args.positional()[1..].to_vec(), args.keywords().to_vec());
    statistic(runtime, array, forwarded, statistic_kind, name)
}

fn statistic(
    runtime: &mut dyn PyRuntime,
    array: PyArray,
    args: CallArgs,
    statistic_kind: Statistic,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 0, 1)?;
    args.reject_unknown_keywords(name, &["axis"])?;
    let (layout, input_dtype) = runtime.array_layout(array)?;
    let shape = layout.shape;
    let output_dtype = statistic_dtype(input_dtype, statistic_kind);
    let axis_value = args
        .positional()
        .first()
        .copied()
        .or(args.keyword(name, "axis")?.copied());
    if let Some(axis) = axis_value {
        let PyIndex(axis) = axis.cast(runtime)?;
        return statistic_axis(
            runtime,
            array,
            &shape,
            normalize_axis(axis, shape.len())?,
            statistic_kind,
            output_dtype,
        );
    }
    let mut state = StatisticState::new(runtime, statistic_kind, element_count(&shape)?)?;
    for_each_index(&shape, |index| {
        let value = runtime.array_get(array, index)?;
        state.observe(runtime, value)
    })?;
    state.finish(runtime, statistic_kind, output_dtype)
}

fn statistic_axis(
    runtime: &mut dyn PyRuntime,
    array: PyArray,
    shape: &[usize],
    axis: usize,
    statistic_kind: Statistic,
    output_dtype: PyArrayDtype,
) -> PyResult {
    let mut output_shape = shape.to_vec();
    let axis_length = output_shape.remove(axis);
    let count = element_count(&output_shape)?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    for_each_index(&output_shape, |output_index| {
        let mut state = StatisticState::new(runtime, statistic_kind, axis_length)?;
        for selected in 0..axis_length {
            let mut index = output_index.to_vec();
            index.insert(axis, selected);
            let value = runtime.array_get(array, &index)?;
            state.observe(runtime, value)?;
        }
        values.push(state.finish(runtime, statistic_kind, output_dtype)?);
        Ok(())
    })?;
    runtime.new_array(values, output_shape, output_dtype)
}

fn statistic_dtype(dtype: PyArrayDtype, statistic: Statistic) -> PyArrayDtype {
    if matches!(statistic, Statistic::All | Statistic::Any) {
        PyArrayDtype::Bool
    } else if dtype == PyArrayDtype::Float32 {
        PyArrayDtype::Float32
    } else {
        PyArrayDtype::Float64
    }
}

struct StatisticState {
    count: usize,
    mean: f64,
    squared_deviation: f64,
    ordered: Option<Vec<f64>>,
    all: bool,
    any: bool,
}

impl StatisticState {
    fn new(
        runtime: &mut dyn PyRuntime,
        statistic_kind: Statistic,
        capacity: usize,
    ) -> PyResult<Self> {
        let ordered = if matches!(statistic_kind, Statistic::Median) {
            runtime.reserve_memory(
                capacity
                    .checked_mul(std::mem::size_of::<f64>())
                    .ok_or_else(|| PyError::value_error("statistic input is too large"))?,
            )?;
            Some(Vec::with_capacity(capacity))
        } else {
            None
        };
        Ok(Self {
            count: 0,
            mean: 0.0,
            squared_deviation: 0.0,
            ordered,
            all: true,
            any: false,
        })
    }

    fn observe(&mut self, runtime: &dyn PyRuntime, value: PyValue) -> PyResult<()> {
        let value = scalar(runtime, value)?;
        let numeric = value.as_f64();
        self.count += 1;
        let delta = numeric - self.mean;
        self.mean += delta / self.count as f64;
        self.squared_deviation += delta * (numeric - self.mean);
        if let Some(ordered) = &mut self.ordered {
            ordered.push(numeric);
        }
        self.all &= value.truth();
        self.any |= value.truth();
        Ok(())
    }

    fn finish(
        mut self,
        runtime: &mut dyn PyRuntime,
        statistic_kind: Statistic,
        output_dtype: PyArrayDtype,
    ) -> PyResult<PyValue> {
        match statistic_kind {
            Statistic::All => pack_bool(runtime, self.all),
            Statistic::Any => pack_bool(runtime, self.any),
            Statistic::Variance | Statistic::StdDev => {
                let variance = if self.count == 0 {
                    f64::NAN
                } else {
                    self.squared_deviation / self.count as f64
                };
                pack_float(
                    runtime,
                    if matches!(statistic_kind, Statistic::StdDev) {
                        variance.sqrt()
                    } else {
                        variance
                    },
                    output_dtype,
                )
            }
            Statistic::Median => {
                let ordered = self.ordered.as_mut().expect("median collects values");
                let comparisons = ordered
                    .len()
                    .saturating_mul(ordered.len().max(1).ilog2() as usize);
                runtime.charge_cpu(u64::try_from(comparisons).unwrap_or(u64::MAX))?;
                ordered.sort_by(f64::total_cmp);
                let middle = ordered.len() / 2;
                let value = if ordered.is_empty() {
                    f64::NAN
                } else if ordered.len().is_multiple_of(2) {
                    (ordered[middle - 1] + ordered[middle]) / 2.0
                } else {
                    ordered[middle]
                };
                pack_float(runtime, value, output_dtype)
            }
        }
    }
}

fn module_reduce(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    reduction: Reduction,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 1, 2)?;
    args.reject_unknown_keywords(name, &["axis"])?;
    let array = coerce_array(runtime, args.positional()[0])?;
    let forwarded = CallArgs::new(args.positional()[1..].to_vec(), args.keywords().to_vec());
    reduce(runtime, array, forwarded, reduction, name)
}

#[derive(Clone, Copy)]
enum Reduction {
    Sum,
    Product,
    Mean,
    Min,
    Max,
}

fn reduce(
    runtime: &mut dyn PyRuntime,
    array: PyArray,
    args: CallArgs,
    reduction: Reduction,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 0, 1)?;
    args.reject_unknown_keywords(name, &["axis"])?;
    let axis_value = if let Some(value) = args.positional().first() {
        Some(*value)
    } else {
        args.keyword(name, "axis")?.copied()
    };
    let (layout, dtype) = runtime.array_layout(array)?;
    if let Some(axis_value) = axis_value {
        let PyIndex(axis) = axis_value.cast(runtime)?;
        let axis = normalize_axis(axis, layout.shape.len())?;
        return reduce_axis(runtime, array, &layout.shape, dtype, axis, reduction);
    }
    reduce_values(runtime, array, &layout.shape, dtype, reduction)
}

fn normalize_axis(axis: i64, rank: usize) -> PyResult<usize> {
    let rank = i64::try_from(rank).map_err(|_| PyError::overflow_error("array rank overflow"))?;
    let axis = if axis < 0 { rank + axis } else { axis };
    usize::try_from(axis)
        .ok()
        .filter(|axis| *axis < rank as usize)
        .ok_or_else(|| PyError::value_error("axis is out of bounds for array"))
}

fn reduce_values(
    runtime: &mut dyn PyRuntime,
    array: PyArray,
    shape: &[usize],
    dtype: PyArrayDtype,
    reduction: Reduction,
) -> PyResult {
    let mut accumulator = match reduction {
        Reduction::Sum | Reduction::Product | Reduction::Mean => {
            Some(reduction_identity(runtime, dtype, reduction)?)
        }
        Reduction::Min | Reduction::Max => None,
    };
    let mut count = 0usize;
    for_each_index(shape, |index| {
        let value = runtime.array_get(array, index)?;
        accumulator = Some(match (reduction, accumulator) {
            (Reduction::Sum | Reduction::Mean, Some(current)) => {
                runtime.binary_op(PyBinaryOp::Add, current, value)?
            }
            (Reduction::Product, Some(current)) => {
                runtime.binary_op(PyBinaryOp::Multiply, current, value)?
            }
            (Reduction::Min, Some(current)) => {
                if runtime.compare(&value, &current)? == Ordering::Less {
                    value
                } else {
                    current
                }
            }
            (Reduction::Max, Some(current)) => {
                if runtime.compare(&value, &current)? == Ordering::Greater {
                    value
                } else {
                    current
                }
            }
            (Reduction::Min | Reduction::Max, None) => value,
            _ => unreachable!(),
        });
        count += 1;
        Ok(())
    })?;
    let value = accumulator
        .ok_or_else(|| PyError::value_error("zero-size array reduction has no identity"))?;
    if matches!(reduction, Reduction::Mean) {
        if count == 0 {
            return Err(PyError::value_error("mean of empty array"));
        }
        let count = i64::try_from(count)
            .map_err(|_| PyError::overflow_error("array length exceeds numpy.int64"))?;
        let divisor = convert(
            runtime,
            Value::Int(count),
            reduction_dtype(dtype, reduction),
        )?;
        runtime.binary_op(PyBinaryOp::Divide, value, divisor)
    } else {
        Ok(value)
    }
}

fn reduce_axis(
    runtime: &mut dyn PyRuntime,
    array: PyArray,
    shape: &[usize],
    dtype: PyArrayDtype,
    axis: usize,
    reduction: Reduction,
) -> PyResult {
    let mut output_shape = shape.to_vec();
    let axis_length = output_shape.remove(axis);
    let count = element_count(&output_shape)?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    let output_dtype = reduction_dtype(dtype, reduction);
    for_each_index(&output_shape, |output_index| {
        let mut accumulator = match reduction {
            Reduction::Sum | Reduction::Product | Reduction::Mean => {
                Some(reduction_identity(runtime, dtype, reduction)?)
            }
            _ => None,
        };
        for selected in 0..axis_length {
            let mut index = output_index.to_vec();
            index.insert(axis, selected);
            let value = runtime.array_get(array, &index)?;
            accumulator = Some(match (reduction, accumulator) {
                (Reduction::Sum | Reduction::Mean, Some(current)) => {
                    runtime.binary_op(PyBinaryOp::Add, current, value)?
                }
                (Reduction::Product, Some(current)) => {
                    runtime.binary_op(PyBinaryOp::Multiply, current, value)?
                }
                (Reduction::Min, Some(current)) => {
                    if runtime.compare(&value, &current)? == Ordering::Less {
                        value
                    } else {
                        current
                    }
                }
                (Reduction::Max, Some(current)) => {
                    if runtime.compare(&value, &current)? == Ordering::Greater {
                        value
                    } else {
                        current
                    }
                }
                (_, None) => value,
            });
        }
        let mut value = accumulator
            .ok_or_else(|| PyError::value_error("zero-size array reduction has no identity"))?;
        if matches!(reduction, Reduction::Mean) {
            if axis_length == 0 {
                return Err(PyError::value_error("mean of empty array"));
            }
            let divisor = i64::try_from(axis_length)
                .map_err(|_| PyError::overflow_error("array length exceeds numpy.int64"))?;
            let divisor = convert(runtime, Value::Int(divisor), output_dtype)?;
            value = runtime.binary_op(PyBinaryOp::Divide, value, divisor)?;
        }
        values.push(value);
        Ok(())
    })?;
    runtime.new_array(values, output_shape, output_dtype)
}

fn reduction_dtype(dtype: PyArrayDtype, reduction: Reduction) -> PyArrayDtype {
    let kind = dtype_kind(dtype);
    match reduction {
        Reduction::Mean if dtype == PyArrayDtype::Float32 => PyArrayDtype::Float32,
        Reduction::Mean => PyArrayDtype::Float64,
        Reduction::Sum | Reduction::Product
            if matches!(kind.class, NumericClass::Bool | NumericClass::Signed)
                && kind.bits < 64 =>
        {
            PyArrayDtype::Int64
        }
        Reduction::Sum | Reduction::Product
            if kind.class == NumericClass::Unsigned && kind.bits < 64 =>
        {
            PyArrayDtype::UInt64
        }
        _ => dtype,
    }
}

fn reduction_identity(
    runtime: &dyn PyRuntime,
    dtype: PyArrayDtype,
    reduction: Reduction,
) -> PyResult<PyValue> {
    let one = matches!(reduction, Reduction::Product);
    let dtype = reduction_dtype(dtype, reduction);
    numeric_identity(runtime, dtype, one)
}

fn numeric_identity(runtime: &dyn PyRuntime, dtype: PyArrayDtype, one: bool) -> PyResult<PyValue> {
    let value = match dtype_kind(dtype).class {
        NumericClass::Bool => ScalarValue::Bool(one),
        NumericClass::Signed => ScalarValue::Signed(i128::from(one)),
        NumericClass::Unsigned => ScalarValue::Unsigned(u128::from(one)),
        NumericClass::Float => ScalarValue::Float(f64::from(one)),
    };
    pack_wrapping(runtime, Scalar { dtype, value }, dtype)
}

fn dot(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.dot", 2, 2)?;
    args.reject_keywords("numpy.dot")?;
    let left = coerce_array(runtime, args.positional()[0])?;
    let right = coerce_array(runtime, args.positional()[1])?;
    let (left_layout, left_dtype) = runtime.array_layout(left)?;
    let (right_layout, right_dtype) = runtime.array_layout(right)?;
    let left_shape = left_layout.shape;
    let right_shape = right_layout.shape;
    let dtype = product_dtype(left_dtype, right_dtype);
    if left_shape.len() == 1 && right_shape.len() == 1 {
        if left_shape[0] != right_shape[0] {
            return Err(PyError::value_error("shapes are not aligned"));
        }
        let mut total = numeric_identity(runtime, dtype, false)?;
        for inner in 0..left_shape[0] {
            let left_value = runtime.array_get(left, &[inner])?;
            let right_value = runtime.array_get(right, &[inner])?;
            let product = runtime.binary_op(PyBinaryOp::Multiply, left_value, right_value)?;
            total = runtime.binary_op(PyBinaryOp::Add, total, product)?;
        }
        Ok(total)
    } else {
        matmul_arrays(runtime, left, right)
    }
}

fn inner(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.inner", 2, 2)?;
    args.reject_keywords("numpy.inner")?;
    let left = coerce_array(runtime, args.positional()[0])?;
    let right = coerce_array(runtime, args.positional()[1])?;
    let (left_layout, left_dtype) = runtime.array_layout(left)?;
    let (right_layout, right_dtype) = runtime.array_layout(right)?;
    if left_layout.shape.is_empty() || right_layout.shape.is_empty() {
        return binary(
            runtime,
            Value::Object(left.object_id()),
            Value::Object(right.object_id()),
            PyBinaryOp::Multiply,
            false,
        )?
        .ok_or_else(|| PyError::runtime_error("inner product rejected array operands"));
    }
    let inner = *left_layout.shape.last().expect("nonempty shape checked");
    if right_layout.shape.last() != Some(&inner) {
        return Err(PyError::value_error("shapes are not aligned"));
    }
    let mut shape = left_layout.shape[..left_layout.shape.len() - 1].to_vec();
    shape.extend_from_slice(&right_layout.shape[..right_layout.shape.len() - 1]);
    let dtype = product_dtype(left_dtype, right_dtype);
    let count = element_count(&shape)?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    let left_outer_rank = left_layout.shape.len() - 1;
    for_each_index(&shape, |output| {
        let mut total = numeric_identity(runtime, dtype, false)?;
        for contracted in 0..inner {
            let mut left_index = output[..left_outer_rank].to_vec();
            left_index.push(contracted);
            let mut right_index = output[left_outer_rank..].to_vec();
            right_index.push(contracted);
            let left_value = runtime.array_get(left, &left_index)?;
            let right_value = runtime.array_get(right, &right_index)?;
            let product = runtime.binary_op(PyBinaryOp::Multiply, left_value, right_value)?;
            total = runtime.binary_op(PyBinaryOp::Add, total, product)?;
        }
        values.push(total);
        Ok(())
    })?;
    if shape.is_empty() {
        Ok(values[0])
    } else {
        runtime.new_array(values, shape, dtype)
    }
}

fn outer(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.outer", 2, 2)?;
    args.reject_keywords("numpy.outer")?;
    let left = coerce_array(runtime, args.positional()[0])?;
    let right = coerce_array(runtime, args.positional()[1])?;
    let left_count = element_count(&runtime.array_layout(left)?.0.shape)?;
    let right_count = element_count(&runtime.array_layout(right)?.0.shape)?;
    let left_copy = copy_array(runtime, left)?;
    let left_copy = left_copy.cast(runtime)?;
    let left = reshape(runtime, left_copy, vec![left_count])?.cast::<PyArray>(runtime)?;
    let right_copy = copy_array(runtime, right)?;
    let right_copy = right_copy.cast(runtime)?;
    let right = reshape(runtime, right_copy, vec![right_count])?.cast::<PyArray>(runtime)?;
    let left_dtype = runtime.array_layout(left)?.1;
    let right_dtype = runtime.array_layout(right)?.1;
    let dtype = product_dtype(left_dtype, right_dtype);
    let count = left_count
        .checked_mul(right_count)
        .ok_or_else(|| PyError::value_error("array is too large"))?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    for row in 0..left_count {
        for column in 0..right_count {
            let left_value = runtime.array_get(left, &[row])?;
            let right_value = runtime.array_get(right, &[column])?;
            values.push(runtime.binary_op(PyBinaryOp::Multiply, left_value, right_value)?);
        }
    }
    runtime.new_array(values, vec![left_count, right_count], dtype)
}

fn matmul(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.matmul", 2, 2)?;
    args.reject_keywords("numpy.matmul")?;
    let left = coerce_array(runtime, args.positional()[0])?;
    let right = coerce_array(runtime, args.positional()[1])?;
    matmul_arrays(runtime, left, right)
}

fn matmul_arrays(runtime: &mut dyn PyRuntime, left: PyArray, right: PyArray) -> PyResult {
    let (left_layout, left_dtype) = runtime.array_layout(left)?;
    let (right_layout, right_dtype) = runtime.array_layout(right)?;
    let left_shape = left_layout.shape;
    let right_shape = right_layout.shape;
    if left_shape.len() != 2 || right_shape.len() != 2 || left_shape[1] != right_shape[0] {
        return Err(PyError::value_error(
            "matmul requires aligned two-dimensional arrays",
        ));
    }
    let shape = vec![left_shape[0], right_shape[1]];
    let count = element_count(&shape)?;
    reserve_values(runtime, count)?;
    let mut values = Vec::with_capacity(count);
    let dtype = product_dtype(left_dtype, right_dtype);
    for row in 0..shape[0] {
        for column in 0..shape[1] {
            let mut total = numeric_identity(runtime, dtype, false)?;
            for inner in 0..left_shape[1] {
                runtime.charge_cpu(1)?;
                let left_value = runtime.array_get(left, &[row, inner])?;
                let right_value = runtime.array_get(right, &[inner, column])?;
                let product = runtime.binary_op(PyBinaryOp::Multiply, left_value, right_value)?;
                total = runtime.binary_op(PyBinaryOp::Add, total, product)?;
            }
            values.push(total);
        }
    }
    runtime.new_array(values, shape, dtype)
}

fn diag(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.diag", 1, 2)?;
    args.reject_unknown_keywords("numpy.diag", &["k"])?;
    let array = coerce_array(runtime, args.positional()[0])?;
    let (layout, dtype) = runtime.array_layout(array)?;
    let offset = args
        .positional()
        .get(1)
        .copied()
        .or(args.keyword("numpy.diag", "k")?.copied())
        .unwrap_or(Value::Int(0));
    let PyIndex(offset) = offset.cast(runtime)?;
    match layout.shape.as_slice() {
        [length] => {
            let padding = usize::try_from(offset.unsigned_abs())
                .map_err(|_| PyError::value_error("diagonal offset is too large"))?;
            let size = length
                .checked_add(padding)
                .ok_or_else(|| PyError::value_error("diagonal shape overflow"))?;
            let count = element_count(&[size, size])?;
            reserve_values(runtime, count)?;
            let zero = numeric_identity(runtime, dtype, false)?;
            let mut values = Vec::with_capacity(count);
            for row in 0..size {
                for column in 0..size {
                    let source = if offset >= 0 && column == row.saturating_add(padding) {
                        Some(row)
                    } else if offset < 0 && row == column.saturating_add(padding) {
                        Some(column)
                    } else {
                        None
                    };
                    values.push(
                        if let Some(index) = source.filter(|index| *index < *length) {
                            runtime.array_get(array, &[index])?
                        } else {
                            zero
                        },
                    );
                }
            }
            runtime.new_array(values, vec![size, size], dtype)
        }
        [rows, columns] => {
            let (row, column) = if offset >= 0 {
                (0usize, usize::try_from(offset).unwrap_or(usize::MAX))
            } else {
                (
                    usize::try_from(offset.unsigned_abs()).unwrap_or(usize::MAX),
                    0usize,
                )
            };
            let length = rows.saturating_sub(row).min(columns.saturating_sub(column));
            reserve_values(runtime, length)?;
            let mut values = Vec::with_capacity(length);
            for index in 0..length {
                values.push(runtime.array_get(array, &[row + index, column + index])?);
            }
            runtime.new_array(values, vec![length], dtype)
        }
        _ => Err(PyError::value_error(
            "numpy.diag requires a 1-D or 2-D array",
        )),
    }
}

fn allclose(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.allclose", 2, 4)?;
    args.reject_unknown_keywords("numpy.allclose", &["rtol", "atol", "equal_nan"])?;
    let left = coerce_array(runtime, args.positional()[0])?;
    let right = coerce_array(runtime, args.positional()[1])?;
    let rtol = args
        .positional()
        .get(2)
        .copied()
        .or(args.keyword("numpy.allclose", "rtol")?.copied())
        .map(|value| scalar(runtime, value).map(|value| value.as_f64()))
        .transpose()?
        .unwrap_or(1e-5);
    let atol = args
        .positional()
        .get(3)
        .copied()
        .or(args.keyword("numpy.allclose", "atol")?.copied())
        .map(|value| scalar(runtime, value).map(|value| value.as_f64()))
        .transpose()?
        .unwrap_or(1e-8);
    let equal_nan = args
        .keyword("numpy.allclose", "equal_nan")?
        .map(|value| runtime.truth(value))
        .transpose()?
        .unwrap_or(false);
    let (left_layout, _) = runtime.array_layout(left)?;
    let (right_layout, _) = runtime.array_layout(right)?;
    let shape = broadcast_shape(&left_layout.shape, &right_layout.shape)?;
    let mut close = true;
    for_each_index(&shape, |index| {
        let left = broadcast_get(
            runtime,
            Value::Object(left.object_id()),
            Some(&left_layout),
            index,
            &shape,
        )?;
        let right = broadcast_get(
            runtime,
            Value::Object(right.object_id()),
            Some(&right_layout),
            index,
            &shape,
        )?;
        let left = scalar(runtime, left)?.as_f64();
        let right = scalar(runtime, right)?.as_f64();
        close &= if left.is_nan() || right.is_nan() {
            equal_nan && left.is_nan() && right.is_nan()
        } else {
            left == right || (left - right).abs() <= atol + rtol * right.abs()
        };
        Ok(())
    })?;
    Ok(Value::Bool(close))
}

fn argsort(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.argsort", 1, 1)?;
    args.reject_unknown_keywords("numpy.argsort", &["axis", "kind"])?;
    if let Some(kind) = args.keyword("numpy.argsort", "kind")? {
        let PyString(kind) = (*kind).cast(runtime)?;
        if !matches!(
            kind.as_str(),
            "quicksort" | "mergesort" | "heapsort" | "stable"
        ) {
            return Err(PyError::value_error(format!(
                "unsupported numpy.argsort kind {kind:?}"
            )));
        }
    }
    let array = coerce_array(runtime, args.positional()[0])?;
    let (layout, _) = runtime.array_layout(array)?;
    let axis_value = args.keyword("numpy.argsort", "axis")?.copied();
    if axis_value.is_some_and(|value| value.is_none()) {
        let count = element_count(&layout.shape)?;
        reserve_values(runtime, count)?;
        let mut indexed = Vec::with_capacity(count);
        let mut ordinal = 0usize;
        for_each_index(&layout.shape, |index| {
            let value = runtime.array_get(array, index)?;
            indexed.push((scalar(runtime, value)?, ordinal));
            ordinal += 1;
            Ok(())
        })?;
        sort_indexed(runtime, &mut indexed)?;
        let values = indexed
            .into_iter()
            .map(|(_, index)| pack_index(runtime, index))
            .collect::<PyResult<Vec<_>>>()?;
        return runtime.new_array(values, vec![count], PyArrayDtype::Int64);
    }
    let axis = match axis_value {
        Some(value) => {
            let PyIndex(axis) = value.cast(runtime)?;
            normalize_axis(axis, layout.shape.len())?
        }
        None => layout
            .shape
            .len()
            .checked_sub(1)
            .ok_or_else(|| PyError::value_error("cannot argsort a 0-D array"))?,
    };
    let count = element_count(&layout.shape)?;
    reserve_values(runtime, count)?;
    let mut values = vec![Value::Int(0); count];
    let strides = contiguous_strides(&layout.shape)?;
    let mut outer_shape = layout.shape.clone();
    let axis_length = outer_shape.remove(axis);
    for_each_index(&outer_shape, |outer| {
        let mut indexed = Vec::with_capacity(axis_length);
        for selected in 0..axis_length {
            let mut index = outer.to_vec();
            index.insert(axis, selected);
            let value = runtime.array_get(array, &index)?;
            indexed.push((scalar(runtime, value)?, selected));
        }
        sort_indexed(runtime, &mut indexed)?;
        for (position, (_, source)) in indexed.into_iter().enumerate() {
            let mut output_index = outer.to_vec();
            output_index.insert(axis, position);
            let flat = output_index
                .iter()
                .zip(&strides)
                .try_fold(0usize, |total, (index, stride)| {
                    total.checked_add(index.saturating_mul(*stride as usize))
                })
                .ok_or_else(|| PyError::value_error("array offset overflow"))?;
            values[flat] = pack_index(runtime, source)?;
        }
        Ok(())
    })?;
    runtime.new_array(values, layout.shape, PyArrayDtype::Int64)
}

fn sort_indexed(runtime: &mut dyn PyRuntime, values: &mut [(Scalar, usize)]) -> PyResult<()> {
    let comparisons = values
        .len()
        .saturating_mul(values.len().max(1).ilog2() as usize);
    runtime.charge_cpu(u64::try_from(comparisons).unwrap_or(u64::MAX))?;
    values.sort_by(|(left, _), (right, _)| {
        left.compare(*right).unwrap_or_else(|| {
            match (left.as_f64().is_nan(), right.as_f64().is_nan()) {
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                _ => Ordering::Equal,
            }
        })
    });
    Ok(())
}

fn percentile(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("numpy.percentile", 2, 2)?;
    args.reject_unknown_keywords("numpy.percentile", &["axis"])?;
    if args
        .keyword("numpy.percentile", "axis")?
        .is_some_and(|axis| !axis.is_none())
    {
        return Err(PyError::value_error(
            "numpy.percentile axis is not supported",
        ));
    }
    let array = coerce_array(runtime, args.positional()[0])?;
    let quantile = scalar(runtime, args.positional()[1])?.as_f64();
    if !(0.0..=100.0).contains(&quantile) {
        return Err(PyError::value_error(
            "percentile must be in the range [0, 100]",
        ));
    }
    let (layout, _) = runtime.array_layout(array)?;
    let count = element_count(&layout.shape)?;
    runtime.reserve_memory(
        count
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| PyError::value_error("percentile input is too large"))?,
    )?;
    let mut values = Vec::with_capacity(count);
    for_each_index(&layout.shape, |index| {
        let value = runtime.array_get(array, index)?;
        values.push(scalar(runtime, value)?.as_f64());
        Ok(())
    })?;
    let comparisons = count.saturating_mul(count.max(1).ilog2() as usize);
    runtime.charge_cpu(u64::try_from(comparisons).unwrap_or(u64::MAX))?;
    values.sort_by(f64::total_cmp);
    let value = if values.is_empty() {
        f64::NAN
    } else {
        let position = quantile / 100.0 * (values.len() - 1) as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        values[lower] + (values[upper] - values[lower]) * (position - lower as f64)
    };
    pack_float(runtime, value, PyArrayDtype::Float64)
}

fn product_dtype(left: PyArrayDtype, right: PyArrayDtype) -> PyArrayDtype {
    promote_dtype(left, right)
}

pub(in crate::python) fn contiguous_strides(shape: &[usize]) -> PyResult<Vec<isize>> {
    validate_rank(shape)?;
    let mut stride = 1usize;
    let mut result = vec![0isize; shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] =
            isize::try_from(stride).map_err(|_| PyError::value_error("array stride overflow"))?;
        stride = stride
            .checked_mul(shape[axis])
            .ok_or_else(|| PyError::value_error("array stride overflow"))?;
    }
    Ok(result)
}

fn for_each_index(
    shape: &[usize],
    mut operation: impl FnMut(&[usize]) -> PyResult<()>,
) -> PyResult<()> {
    if shape.contains(&0) {
        return Ok(());
    }
    if shape.is_empty() {
        return operation(&[]);
    }
    let mut index = vec![0; shape.len()];
    loop {
        operation(&index)?;
        let mut axis = shape.len();
        loop {
            if axis == 0 {
                return Ok(());
            }
            axis -= 1;
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcasting_aligns_dimensions_from_the_right() {
        assert_eq!(broadcast_shape(&[2, 1, 3], &[4, 3]).unwrap(), vec![2, 4, 3]);
        assert!(broadcast_shape(&[2, 3], &[4]).is_err());
    }

    #[test]
    fn contiguous_layout_is_row_major() {
        assert_eq!(contiguous_strides(&[2, 3, 4]).unwrap(), vec![12, 4, 1]);
    }

    #[test]
    fn promotion_uses_numeric_class_and_width() {
        for (left, right, expected) in [
            (PyArrayDtype::Int8, PyArrayDtype::Int8, PyArrayDtype::Int8),
            (PyArrayDtype::Int8, PyArrayDtype::UInt8, PyArrayDtype::Int16),
            (
                PyArrayDtype::Int32,
                PyArrayDtype::UInt32,
                PyArrayDtype::Int64,
            ),
            (
                PyArrayDtype::Int64,
                PyArrayDtype::UInt64,
                PyArrayDtype::Float64,
            ),
            (
                PyArrayDtype::Float32,
                PyArrayDtype::Int16,
                PyArrayDtype::Float32,
            ),
            (
                PyArrayDtype::Float32,
                PyArrayDtype::Int32,
                PyArrayDtype::Float64,
            ),
            (
                PyArrayDtype::Float32,
                PyArrayDtype::Float64,
                PyArrayDtype::Float64,
            ),
        ] {
            assert_eq!(promote_dtype(left, right), expected);
            assert_eq!(promote_dtype(right, left), expected);
        }
    }

    #[test]
    fn numeric_registration_ids_and_aliases_are_unambiguous() {
        for (index, left) in NUMERIC_KINDS.iter().enumerate() {
            for right in &NUMERIC_KINDS[index + 1..] {
                assert_ne!(left.dtype, right.dtype);
                for alias in left.aliases {
                    assert!(!right.aliases.contains(alias), "duplicate alias {alias}");
                }
            }
        }
    }

    #[test]
    fn python_scalars_are_weak_against_registered_dtypes() {
        assert_eq!(
            promote_operands(PyArrayDtype::Int8, PyArrayDtype::Int64, false, true,),
            PyArrayDtype::Int8
        );
        assert_eq!(
            promote_operands(PyArrayDtype::Float32, PyArrayDtype::Float64, false, true,),
            PyArrayDtype::Float32
        );
        assert_eq!(
            promote_operands(PyArrayDtype::Int8, PyArrayDtype::Float64, false, true,),
            PyArrayDtype::Float64
        );
    }
}
