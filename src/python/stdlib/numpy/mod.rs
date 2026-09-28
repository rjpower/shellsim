//! shellsim's NumPy: typed ndarray storage, a dtype table, ufuncs, and the native modules behind
//! the frozen `numpy` package.
//!
//! Arrays are views over VM-owned storage: packed little-endian bytes for numeric and string
//! dtypes, and traced Python references for `object`. A [`dtype::DType`] indexes a constant
//! table, so metadata lookups are O(1), and kernels are monomorphized per element type through
//! the dispatch macros in [`element`].
//!
//! The Python-visible package is frozen source (`source/numpy/`). It star-imports the native
//! modules registered in [`native_module`]: `_numpy` holds the core (constructors, dtypes,
//! ufuncs, and scalar types), and each area module (`_numpy_reduce`, `_numpy_shape`, ...) holds
//! one family of functions. Areas also install their `ndarray` methods through [`array_types`],
//! so each area lives in its own file.
//!
//! Simulated code reaches no host capability through this module: files go through the virtual
//! filesystem, and every loop charges CPU and reserves memory before allocating.

mod args;
mod array;
mod construct;
mod convert;
mod dtype;
mod dtype_object;
mod element;
mod errstate;
mod format;
mod index;
mod io;
mod layout;
pub(in crate::python) mod linalg;
mod math;
mod ndarray;
mod ops;
mod products;
mod random;
mod reduce;
mod scalar;
mod select;
mod shape;
mod sort;
mod strings;
mod ufunc;
mod underflow;

pub(in crate::python) use ndarray::{
    slot_bool, slot_get_item, slot_iter, slot_length, slot_matrix_multiply,
    slot_reflected_matrix_multiply, slot_repr, slot_set_item, slot_str,
};
pub(in crate::python) use ufunc::{
    slot_absolute, slot_add, slot_bitwise_and, slot_bitwise_or, slot_bitwise_xor, slot_divide,
    slot_divmod, slot_equal, slot_floor_divide, slot_greater_equal, slot_greater_than, slot_invert,
    slot_left_shift, slot_less_equal, slot_less_than, slot_multiply, slot_negative, slot_not_equal,
    slot_positive, slot_power, slot_reflected_add, slot_reflected_bitwise_and,
    slot_reflected_bitwise_or, slot_reflected_bitwise_xor, slot_reflected_divide,
    slot_reflected_divmod, slot_reflected_floor_divide, slot_reflected_left_shift,
    slot_reflected_multiply, slot_reflected_power, slot_reflected_remainder,
    slot_reflected_right_shift, slot_reflected_subtract, slot_remainder, slot_right_shift,
    slot_subtract,
};

use super::super::native::{
    FunctionDef, ModuleDef, NativeFn, NativeTypeDef, PyConstant, PyMarker, PyResult, PyRuntime,
    ValueDef, ValueKindDef,
};

/// Registered value kinds in dependency order: every kind follows its bases.
pub(in crate::python) fn value_kinds() -> impl Iterator<Item = &'static ValueKindDef> {
    scalar::ABSTRACT
        .iter()
        .copied()
        .chain(scalar::SCALARS.iter())
        .chain([&dtype_object::DTYPE, &ufunc::UFUNC])
}

/// Attribute tables installed on `numpy.ndarray`, core first.
pub(in crate::python) fn array_types() -> [&'static NativeTypeDef; 8] {
    [
        &ndarray::ARRAY_TYPE,
        &reduce::ARRAY_METHODS,
        &shape::ARRAY_METHODS,
        &select::ARRAY_METHODS,
        &sort::ARRAY_METHODS,
        &products::ARRAY_METHODS,
        &math::ARRAY_METHODS,
        &io::ARRAY_METHODS,
    ]
}

/// The native modules the frozen `numpy` package imports.
pub(in crate::python) fn native_module(name: &str) -> Option<&'static ModuleDef> {
    Some(match name {
        "_numpy" => &MODULE,
        "_numpy_reduce" => &reduce::MODULE,
        "_numpy_shape" => &shape::MODULE,
        "_numpy_sort" => &sort::MODULE,
        "_numpy_products" => &products::MODULE,
        "_numpy_linalg" => &linalg::MODULE,
        "_numpy_random" => &random::MODULE,
        "_numpy_io" => &io::MODULE,
        "_numpy_math" => &math::MODULE,
        "_numpy_strings" => &strings::MODULE,
        _ => return None,
    })
}

const fn function(name: &'static str, call: NativeFn) -> FunctionDef {
    FunctionDef {
        module: "numpy",
        name,
        call,
    }
}

static MODULE: ModuleDef = ModuleDef {
    name: "_numpy",
    functions: &[
        function("array", construct::array),
        function("asarray", construct::asarray),
        function("asanyarray", construct::asarray),
        function("ascontiguousarray", construct::ascontiguousarray),
        function("asfortranarray", construct::asfortranarray),
        function("copy", construct::copy),
        function("zeros", construct::zeros),
        function("ones", construct::ones),
        function("empty", construct::empty),
        function("full", construct::full),
        function("zeros_like", construct::zeros_like),
        function("ones_like", construct::ones_like),
        function("empty_like", construct::empty_like),
        function("full_like", construct::full_like),
        function("arange", construct::arange),
        function("linspace", construct::linspace),
        function("logspace", construct::logspace),
        function("geomspace", construct::geomspace),
        function("eye", construct::eye),
        function("identity", construct::identity),
        function("diag", construct::diag),
        function("meshgrid", construct::meshgrid),
        function("fromiter", construct::fromiter),
        function("shares_memory", construct::shares_memory),
        function("may_share_memory", construct::may_share_memory),
        function("ndim", construct::ndim),
        function("shape", construct::shape),
        function("size", construct::size),
        function("take", construct::take),
        function("put", construct::put),
        function("nonzero", select::module_nonzero),
        function("where", select::module_where),
        function("copyto", select::module_copyto),
        function("result_type", construct::result_type),
        function("promote_types", construct::promote_types),
        function("can_cast", construct::can_cast),
        function("issubdtype", construct::issubdtype),
        function("isscalar", construct::isscalar),
        function("_c_contiguous", ndarray::c_contiguous),
        function("_writeable", ndarray::writeable),
    ],
    values: VALUES,
};

macro_rules! kind_types {
    ($($function:ident => $kind:expr;)*) => {
        $(
            fn $function(runtime: &mut dyn PyRuntime) -> PyResult {
                runtime.value_kind_type($kind)
            }
        )*
    };
}

kind_types! {
    generic_type => &scalar::GENERIC;
    number_type => &scalar::NUMBER;
    integer_type => &scalar::INTEGER;
    signed_integer_type => &scalar::SIGNED_INTEGER;
    unsigned_integer_type => &scalar::UNSIGNED_INTEGER;
    inexact_type => &scalar::INEXACT;
    floating_type => &scalar::FLOATING;
    complex_floating_type => &scalar::COMPLEX_FLOATING;
    bool_type => &scalar::SCALARS[dtype::Kind::Bool as usize];
    int8_type => &scalar::SCALARS[dtype::Kind::Int8 as usize];
    int16_type => &scalar::SCALARS[dtype::Kind::Int16 as usize];
    int32_type => &scalar::SCALARS[dtype::Kind::Int32 as usize];
    int64_type => &scalar::SCALARS[dtype::Kind::Int64 as usize];
    uint8_type => &scalar::SCALARS[dtype::Kind::UInt8 as usize];
    uint16_type => &scalar::SCALARS[dtype::Kind::UInt16 as usize];
    uint32_type => &scalar::SCALARS[dtype::Kind::UInt32 as usize];
    uint64_type => &scalar::SCALARS[dtype::Kind::UInt64 as usize];
    float16_type => &scalar::SCALARS[dtype::Kind::Float16 as usize];
    float32_type => &scalar::SCALARS[dtype::Kind::Float32 as usize];
    float64_type => &scalar::SCALARS[dtype::Kind::Float64 as usize];
    complex64_type => &scalar::SCALARS[dtype::Kind::Complex64 as usize];
    complex128_type => &scalar::SCALARS[dtype::Kind::Complex128 as usize];
    dtype_type => &dtype_object::DTYPE;
    ufunc_type => &ufunc::UFUNC;
}

fn ndarray_type(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.marker(PyMarker::ArrayType))
}

/// `np.str_` and `np.object_`: `str` and `object` elements box to builtin values, so their
/// scalar types are the builtin types.
fn str_type(runtime: &mut dyn PyRuntime) -> PyResult {
    builtin(runtime, "str")
}

fn object_type(runtime: &mut dyn PyRuntime) -> PyResult {
    builtin(runtime, "object")
}

fn builtin(runtime: &mut dyn PyRuntime, name: &str) -> PyResult {
    runtime.builtin_type(name).ok_or_else(|| {
        super::super::native::PyError::runtime_error(format!("builtin {name} is missing"))
    })
}

macro_rules! exception_types {
    ($($function:ident => $name:literal;)*) => {
        $(
            fn $function(runtime: &mut dyn PyRuntime) -> PyResult {
                Ok(runtime.exception_type($name))
            }
        )*
    };
}

exception_types! {
    axis_error => "AxisError";
    linalg_error => "LinAlgError";
    ufunc_type_error => "UFuncTypeError";
    dtype_promotion_error => "DTypePromotionError";
    complex_warning => "ComplexWarning";
}

const fn factory(name: &'static str, get: fn(&mut dyn PyRuntime) -> PyResult) -> ValueDef {
    ValueDef::Factory { name, get }
}

const fn constant(name: &'static str, value: f64) -> ValueDef {
    ValueDef::Constant {
        name,
        value: PyConstant::Float(value),
    }
}

const fn ufunc_value(name: &'static str) -> ValueDef {
    named_ufunc_value(name, name)
}

/// A module value named `name` for the ufunc named `ufunc`, which may be an alias; SciPy's
/// modules export their ufuncs through this.
pub(in crate::python) const fn named_ufunc_value(name: &'static str, ufunc: &str) -> ValueDef {
    ValueDef::Registered {
        name,
        kind: &ufunc::UFUNC,
        payload: ufunc::index_of(ufunc),
    }
}

/// A module value naming the `scipy.special` ufunc at `position` in its function table.
pub(in crate::python) const fn special_ufunc_value(
    name: &'static str,
    position: usize,
) -> ValueDef {
    ValueDef::Registered {
        name,
        kind: &ufunc::UFUNC,
        payload: ufunc::special_index(position),
    }
}

const fn bool_scalar(name: &'static str, value: bool) -> ValueDef {
    ValueDef::Registered {
        name,
        kind: &scalar::SCALARS[dtype::Kind::Bool as usize],
        payload: value as u64,
    }
}

/// Module values: constants, types, and every ufunc and alias. `ModuleDef` takes one slice,
/// so the macro appends the ufunc values to the fixed entries.
macro_rules! module_values {
    ([$($fixed:expr),* $(,)?], [$($ufunc:literal),* $(,)?]) => {
        &[$($fixed,)* $(ufunc_value($ufunc),)*]
    };
}

static VALUES: &[ValueDef] = module_values!(
    [
        constant("pi", std::f64::consts::PI),
        constant("e", std::f64::consts::E),
        constant("euler_gamma", 0.577_215_664_901_532_9),
        constant("inf", f64::INFINITY),
        constant("nan", f64::NAN),
        factory("ndarray", ndarray_type),
        factory("dtype", dtype_type),
        factory("ufunc", ufunc_type),
        factory("generic", generic_type),
        factory("number", number_type),
        factory("integer", integer_type),
        factory("signedinteger", signed_integer_type),
        factory("unsignedinteger", unsigned_integer_type),
        factory("inexact", inexact_type),
        factory("floating", floating_type),
        factory("complexfloating", complex_floating_type),
        factory("bool", bool_type),
        factory("bool_", bool_type),
        factory("int8", int8_type),
        factory("byte", int8_type),
        factory("int16", int16_type),
        factory("short", int16_type),
        factory("int32", int32_type),
        factory("intc", int32_type),
        factory("int64", int64_type),
        factory("int_", int64_type),
        factory("intp", int64_type),
        factory("long", int64_type),
        factory("longlong", int64_type),
        factory("uint8", uint8_type),
        factory("ubyte", uint8_type),
        factory("uint16", uint16_type),
        factory("ushort", uint16_type),
        factory("uint32", uint32_type),
        factory("uintc", uint32_type),
        factory("uint64", uint64_type),
        factory("uint", uint64_type),
        factory("uintp", uint64_type),
        factory("ulong", uint64_type),
        factory("ulonglong", uint64_type),
        factory("float16", float16_type),
        factory("half", float16_type),
        factory("float32", float32_type),
        factory("single", float32_type),
        factory("float64", float64_type),
        factory("double", float64_type),
        factory("complex64", complex64_type),
        factory("csingle", complex64_type),
        factory("complex128", complex128_type),
        factory("cdouble", complex128_type),
        factory("str_", str_type),
        factory("object_", object_type),
        bool_scalar("True_", true),
        bool_scalar("False_", false),
        factory("_AxisError", axis_error),
        factory("_LinAlgError", linalg_error),
        factory("_UFuncTypeError", ufunc_type_error),
        factory("_DTypePromotionError", dtype_promotion_error),
        factory("_ComplexWarning", complex_warning),
    ],
    [
        "add",
        "subtract",
        "multiply",
        "divide",
        "true_divide",
        "floor_divide",
        "remainder",
        "mod",
        "fmod",
        "power",
        "pow",
        "maximum",
        "minimum",
        "fmax",
        "fmin",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "left_shift",
        "right_shift",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "arctan2",
        "atan2",
        "hypot",
        "copysign",
        "logaddexp",
        "logaddexp2",
        "heaviside",
        "negative",
        "positive",
        "absolute",
        "abs",
        "square",
        "reciprocal",
        "sign",
        "conjugate",
        "conj",
        "floor",
        "ceil",
        "trunc",
        "invert",
        "bitwise_not",
        "bitwise_invert",
        "logical_not",
        "sqrt",
        "cbrt",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log2",
        "log10",
        "log1p",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "asin",
        "arccos",
        "acos",
        "arctan",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "arcsinh",
        "asinh",
        "arccosh",
        "acosh",
        "arctanh",
        "atanh",
        "deg2rad",
        "radians",
        "rad2deg",
        "degrees",
        "rint",
        "fabs",
        "isnan",
        "isinf",
        "isfinite",
        "signbit",
    ]
);
