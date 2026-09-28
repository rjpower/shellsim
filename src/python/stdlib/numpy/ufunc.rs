//! Universal functions: dtype resolution, broadcasting loops, and the `numpy.ufunc` type.
//!
//! A call runs in four steps:
//!
//! 1. **Operands.** Arrays and NumPy scalars are strong; Python `bool`/`int`/`float`/`complex`
//!    are weak (NEP 50). Other values convert with `np.asarray`.
//! 2. **Resolution.** The ufunc's [`Family`] picks the loop dtype from the promoted operand
//!    dtype (for example, `true_divide` computes integers in `float64` and `sqrt` uses the
//!    smallest float that holds the input), and the output dtype.
//! 3. **Loop.** Inputs are cast to the loop dtype, then one monomorphized kernel runs over the
//!    broadcast byte offsets. `str` and `object` loops are explicit families.
//! 4. **Result.** Results go to a fresh array, or are cast with `same_kind` into `out=`.
//!    Results with shape `()` box to NumPy scalars unless `out=` was given.
//!
//! A `where=` mask computes only the selected elements; see [`masked`].
//!
//! Floating-point flags raised by a loop go to [`super::errstate::report`].

mod masked;

use std::cmp::Ordering;

use super::super::super::ast::{BinaryOperator, ComparisonOperator, UnaryOperator};
use super::super::super::native::{
    CallArgs, GetterDef, MethodDef, PyArray, PyArrayBuffer, PyArrayData, PyError, PyErrorKind,
    PyKind, PyNativeKind, PyOperator, PyResult, PyRuntime, PyValue, ValueKindDef, ValueKindSlots,
};
use super::super::super::number::NumberRef;
use super::super::super::Value;
use super::super::scipy::special::{
    evaluate as evaluate_special, Function as SpecialFunction, ELEMENT_COST as SPECIAL_ELEMENT_COST,
};
use super::array::{
    broadcast_shapes, broadcast_strides, element_count, reserve_elements, Array, Offsets,
};
use super::convert::{self, Leaf};
use super::dtype::{self, Casting, Category, DType, Kind, Weak};
use super::element::{
    dispatch_complex, dispatch_integer, dispatch_numeric, dispatch_real, Element,
};
use super::layout::{self, Order};
use super::ops::{ComplexParts, FpFlags, Integer, Numeric, Real};
use super::scalar::NO_SLOTS;

/// Binary operations computed in the common input dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum ArithOp {
    Add,
    Subtract,
    Multiply,
    FloorDivide,
    Remainder,
    Fmod,
    Power,
    Maximum,
    Minimum,
    Fmax,
    Fmin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum CompareOp {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum LogicalOp {
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum BitOp {
    And,
    Or,
    Xor,
    LeftShift,
    RightShift,
}

/// Unary operations computed in the input dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum UnaryOp {
    Negative,
    Positive,
    Absolute,
    Square,
    Reciprocal,
    Sign,
    Conjugate,
    Floor,
    Ceil,
    Trunc,
}

/// Real functions of one argument, computed in the smallest float that holds the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum FloatOp {
    Sqrt,
    Cbrt,
    Exp,
    Exp2,
    Expm1,
    Log,
    Log2,
    Log10,
    Log1p,
    Sin,
    Cos,
    Tan,
    Arcsin,
    Arccos,
    Arctan,
    Sinh,
    Cosh,
    Tanh,
    Arcsinh,
    Arccosh,
    Arctanh,
    Deg2rad,
    Rad2deg,
    Rint,
    Fabs,
}

/// Real functions of two arguments, computed in the smallest float that holds the inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum Float2Op {
    Arctan2,
    Hypot,
    Copysign,
    Logaddexp,
    Logaddexp2,
    Heaviside,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum PredicateOp {
    IsNan,
    IsInf,
    IsFinite,
    Signbit,
}

/// How a ufunc treats boolean inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum BoolLoop {
    /// Bool inputs have a bool loop.
    Keep,
    /// Bool inputs compute in `int8`.
    Int8,
    /// NumPy rejects bool inputs with a specific message.
    Reject(&'static str),
    /// No bool loop exists; NumPy reports a missing loop signature.
    Missing,
}

/// Resolution and loop family of one ufunc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum Family {
    Arith {
        op: ArithOp,
        bools: BoolLoop,
        complex: bool,
    },
    TrueDivide,
    Compare(CompareOp),
    Logical(LogicalOp),
    Bitwise(BitOp),
    Float2(Float2Op),
    Unary {
        op: UnaryOp,
        bools: BoolLoop,
        complex: bool,
    },
    Float {
        op: FloatOp,
        complex: bool,
    },
    Invert,
    LogicalNot,
    Predicate(PredicateOp),
    /// A `scipy.special` function, computed in `float32` or `float64`.
    Special(SpecialFunction),
}

/// How a ufunc treats `object` elements. NumPy's `O` loops apply a Python operation, and its
/// `P` loops call the element's method named after the ufunc.
#[derive(Clone, Copy)]
pub(in crate::python) enum ObjectLoop {
    /// No object loop, so resolution rejects `object` operands.
    Missing,
    /// A Python operator or `abs()`, as `PyNumber_Add` or `PyNumber_Absolute` apply it.
    Operator(PyOperator),
    /// `x * x` (`Py_square`).
    Square,
    /// `1 / x` (`Py_reciprocal`).
    Reciprocal,
    /// The first operand if it compares `>=` to the second, else the second.
    Max,
    /// The first operand if it compares `<=` to the second, else the second.
    Min,
    /// Python's `and`, which returns one of its operands.
    And,
    /// Python's `or`.
    Or,
    /// `not x`, as a `bool` object.
    Not,
    /// `-1`, `1` or `0` by comparison with `0`.
    Sign,
    /// A `math` module function, such as `math.floor` for `floor`'s object loop.
    Math(&'static str),
    /// `x.name()` or `x.name(y)`, for the ufunc's `name`.
    Method,
}

/// Static description of one ufunc.
pub(in crate::python) struct UfuncDef {
    pub name: &'static str,
    pub family: Family,
    pub object: ObjectLoop,
}

impl UfuncDef {
    pub(in crate::python) fn nin(&self) -> usize {
        match self.family {
            Family::Arith { .. }
            | Family::TrueDivide
            | Family::Compare(_)
            | Family::Logical(_)
            | Family::Bitwise(_)
            | Family::Float2(_) => 2,
            Family::Special(function) => function.nin(),
            _ => 1,
        }
    }
}

const BOOL_SUBTRACT: &str = "numpy boolean subtract, the `-` operator, is not supported, use \
                             the bitwise_xor, the `^` operator, or the logical_xor function \
                             instead.";
const BOOL_NEGATIVE: &str = "The numpy boolean negative, the `-` operator, is not supported, \
                             use the `~` operator or the logical_not function instead.";

const fn arith(
    name: &'static str,
    op: ArithOp,
    bools: BoolLoop,
    complex: bool,
    object: ObjectLoop,
) -> UfuncDef {
    UfuncDef {
        name,
        family: Family::Arith { op, bools, complex },
        object,
    }
}

const fn binary_operator(operator: BinaryOperator) -> ObjectLoop {
    ObjectLoop::Operator(PyOperator::Binary(operator))
}

const fn unary_operator(operator: UnaryOperator) -> ObjectLoop {
    ObjectLoop::Operator(PyOperator::Unary(operator))
}

const fn compare(name: &'static str, op: CompareOp, operator: ComparisonOperator) -> UfuncDef {
    UfuncDef {
        name,
        family: Family::Compare(op),
        object: ObjectLoop::Operator(PyOperator::Compare(operator)),
    }
}

/// A floating-point function; like NumPy's, its object loop calls the element's method.
const fn float(name: &'static str, op: FloatOp, complex: bool) -> UfuncDef {
    UfuncDef {
        name,
        family: Family::Float { op, complex },
        object: ObjectLoop::Method,
    }
}

const fn unary_def(
    name: &'static str,
    op: UnaryOp,
    bools: BoolLoop,
    complex: bool,
    object: ObjectLoop,
) -> UfuncDef {
    UfuncDef {
        name,
        family: Family::Unary { op, bools, complex },
        object,
    }
}

const fn simple(name: &'static str, family: Family, object: ObjectLoop) -> UfuncDef {
    UfuncDef {
        name,
        family,
        object,
    }
}

/// A `scipy.special` ufunc; SciPy registers no object loops.
const fn special(function: SpecialFunction) -> UfuncDef {
    UfuncDef {
        name: function.name(),
        family: Family::Special(function),
        object: ObjectLoop::Missing,
    }
}

/// Every ufunc; a `numpy.ufunc` value's payload is its index here.
pub(in crate::python) const UFUNCS: &[UfuncDef] = &[
    arith(
        "add",
        ArithOp::Add,
        BoolLoop::Keep,
        true,
        binary_operator(BinaryOperator::Add),
    ),
    arith(
        "subtract",
        ArithOp::Subtract,
        BoolLoop::Reject(BOOL_SUBTRACT),
        true,
        binary_operator(BinaryOperator::Subtract),
    ),
    arith(
        "multiply",
        ArithOp::Multiply,
        BoolLoop::Keep,
        true,
        binary_operator(BinaryOperator::Multiply),
    ),
    simple(
        "divide",
        Family::TrueDivide,
        binary_operator(BinaryOperator::Divide),
    ),
    arith(
        "floor_divide",
        ArithOp::FloorDivide,
        BoolLoop::Int8,
        false,
        binary_operator(BinaryOperator::FloorDivide),
    ),
    arith(
        "remainder",
        ArithOp::Remainder,
        BoolLoop::Int8,
        false,
        binary_operator(BinaryOperator::Remainder),
    ),
    arith(
        "fmod",
        ArithOp::Fmod,
        BoolLoop::Int8,
        false,
        ObjectLoop::Method,
    ),
    arith(
        "power",
        ArithOp::Power,
        BoolLoop::Int8,
        true,
        binary_operator(BinaryOperator::Power),
    ),
    arith(
        "maximum",
        ArithOp::Maximum,
        BoolLoop::Keep,
        true,
        ObjectLoop::Max,
    ),
    arith(
        "minimum",
        ArithOp::Minimum,
        BoolLoop::Keep,
        true,
        ObjectLoop::Min,
    ),
    arith("fmax", ArithOp::Fmax, BoolLoop::Keep, true, ObjectLoop::Max),
    arith("fmin", ArithOp::Fmin, BoolLoop::Keep, true, ObjectLoop::Min),
    compare("equal", CompareOp::Equal, ComparisonOperator::Equal),
    compare(
        "not_equal",
        CompareOp::NotEqual,
        ComparisonOperator::NotEqual,
    ),
    compare("less", CompareOp::Less, ComparisonOperator::Less),
    compare(
        "less_equal",
        CompareOp::LessEqual,
        ComparisonOperator::LessEqual,
    ),
    compare("greater", CompareOp::Greater, ComparisonOperator::Greater),
    compare(
        "greater_equal",
        CompareOp::GreaterEqual,
        ComparisonOperator::GreaterEqual,
    ),
    simple(
        "logical_and",
        Family::Logical(LogicalOp::And),
        ObjectLoop::And,
    ),
    simple("logical_or", Family::Logical(LogicalOp::Or), ObjectLoop::Or),
    simple(
        "logical_xor",
        Family::Logical(LogicalOp::Xor),
        ObjectLoop::Method,
    ),
    simple(
        "bitwise_and",
        Family::Bitwise(BitOp::And),
        binary_operator(BinaryOperator::BitwiseAnd),
    ),
    simple(
        "bitwise_or",
        Family::Bitwise(BitOp::Or),
        binary_operator(BinaryOperator::BitwiseOr),
    ),
    simple(
        "bitwise_xor",
        Family::Bitwise(BitOp::Xor),
        binary_operator(BinaryOperator::BitwiseXor),
    ),
    simple(
        "left_shift",
        Family::Bitwise(BitOp::LeftShift),
        binary_operator(BinaryOperator::LeftShift),
    ),
    simple(
        "right_shift",
        Family::Bitwise(BitOp::RightShift),
        binary_operator(BinaryOperator::RightShift),
    ),
    simple(
        "arctan2",
        Family::Float2(Float2Op::Arctan2),
        ObjectLoop::Method,
    ),
    simple("hypot", Family::Float2(Float2Op::Hypot), ObjectLoop::Method),
    simple(
        "copysign",
        Family::Float2(Float2Op::Copysign),
        ObjectLoop::Missing,
    ),
    simple(
        "logaddexp",
        Family::Float2(Float2Op::Logaddexp),
        ObjectLoop::Missing,
    ),
    simple(
        "logaddexp2",
        Family::Float2(Float2Op::Logaddexp2),
        ObjectLoop::Missing,
    ),
    simple(
        "heaviside",
        Family::Float2(Float2Op::Heaviside),
        ObjectLoop::Missing,
    ),
    unary_def(
        "negative",
        UnaryOp::Negative,
        BoolLoop::Reject(BOOL_NEGATIVE),
        true,
        unary_operator(UnaryOperator::Negative),
    ),
    unary_def(
        "positive",
        UnaryOp::Positive,
        BoolLoop::Missing,
        true,
        unary_operator(UnaryOperator::Positive),
    ),
    unary_def(
        "absolute",
        UnaryOp::Absolute,
        BoolLoop::Keep,
        true,
        ObjectLoop::Operator(PyOperator::Absolute),
    ),
    unary_def(
        "square",
        UnaryOp::Square,
        BoolLoop::Int8,
        true,
        ObjectLoop::Square,
    ),
    unary_def(
        "reciprocal",
        UnaryOp::Reciprocal,
        BoolLoop::Int8,
        true,
        ObjectLoop::Reciprocal,
    ),
    unary_def(
        "sign",
        UnaryOp::Sign,
        BoolLoop::Missing,
        true,
        ObjectLoop::Sign,
    ),
    unary_def(
        "conjugate",
        UnaryOp::Conjugate,
        BoolLoop::Int8,
        true,
        ObjectLoop::Method,
    ),
    unary_def(
        "floor",
        UnaryOp::Floor,
        BoolLoop::Keep,
        false,
        ObjectLoop::Math("floor"),
    ),
    unary_def(
        "ceil",
        UnaryOp::Ceil,
        BoolLoop::Keep,
        false,
        ObjectLoop::Math("ceil"),
    ),
    unary_def(
        "trunc",
        UnaryOp::Trunc,
        BoolLoop::Keep,
        false,
        ObjectLoop::Math("trunc"),
    ),
    simple(
        "invert",
        Family::Invert,
        unary_operator(UnaryOperator::Invert),
    ),
    simple("logical_not", Family::LogicalNot, ObjectLoop::Not),
    float("sqrt", FloatOp::Sqrt, true),
    float("cbrt", FloatOp::Cbrt, false),
    float("exp", FloatOp::Exp, true),
    float("exp2", FloatOp::Exp2, false),
    float("expm1", FloatOp::Expm1, false),
    float("log", FloatOp::Log, true),
    float("log2", FloatOp::Log2, false),
    float("log10", FloatOp::Log10, true),
    float("log1p", FloatOp::Log1p, false),
    float("sin", FloatOp::Sin, true),
    float("cos", FloatOp::Cos, true),
    float("tan", FloatOp::Tan, false),
    float("arcsin", FloatOp::Arcsin, false),
    float("arccos", FloatOp::Arccos, false),
    float("arctan", FloatOp::Arctan, false),
    float("sinh", FloatOp::Sinh, false),
    float("cosh", FloatOp::Cosh, false),
    float("tanh", FloatOp::Tanh, false),
    float("arcsinh", FloatOp::Arcsinh, false),
    float("arccosh", FloatOp::Arccosh, false),
    float("arctanh", FloatOp::Arctanh, false),
    float("deg2rad", FloatOp::Deg2rad, false),
    float("rad2deg", FloatOp::Rad2deg, false),
    float("rint", FloatOp::Rint, false),
    float("fabs", FloatOp::Fabs, false),
    simple(
        "isnan",
        Family::Predicate(PredicateOp::IsNan),
        ObjectLoop::Missing,
    ),
    simple(
        "isinf",
        Family::Predicate(PredicateOp::IsInf),
        ObjectLoop::Missing,
    ),
    simple(
        "isfinite",
        Family::Predicate(PredicateOp::IsFinite),
        ObjectLoop::Missing,
    ),
    simple(
        "signbit",
        Family::Predicate(PredicateOp::Signbit),
        ObjectLoop::Missing,
    ),
    // scipy.special, exported by `_scipy_special` rather than NumPy.
    special(SpecialFunction::Erf),
    special(SpecialFunction::Erfc),
    special(SpecialFunction::Erfinv),
    special(SpecialFunction::Erfcinv),
    special(SpecialFunction::Gamma),
    special(SpecialFunction::Gammaln),
    special(SpecialFunction::Loggamma),
    special(SpecialFunction::Psi),
    special(SpecialFunction::Betainc),
    special(SpecialFunction::Betaincc),
    special(SpecialFunction::Betaincinv),
    special(SpecialFunction::Gammainc),
    special(SpecialFunction::Gammaincc),
    special(SpecialFunction::Gammaincinv),
    special(SpecialFunction::Gammainccinv),
    special(SpecialFunction::Ndtr),
    special(SpecialFunction::LogNdtr),
    special(SpecialFunction::Ndtri),
    special(SpecialFunction::Zeta),
];

/// Module-level aliases NumPy exports for some ufuncs.
pub(in crate::python) const ALIASES: &[(&str, &str)] = &[
    ("true_divide", "divide"),
    ("mod", "remainder"),
    ("abs", "absolute"),
    ("conj", "conjugate"),
    ("bitwise_not", "invert"),
    ("bitwise_invert", "invert"),
    ("degrees", "rad2deg"),
    ("radians", "deg2rad"),
    ("bitwise_left_shift", "left_shift"),
    ("bitwise_right_shift", "right_shift"),
    ("pow", "power"),
    ("acos", "arccos"),
    ("asin", "arcsin"),
    ("atan", "arctan"),
    ("atan2", "arctan2"),
    ("acosh", "arccosh"),
    ("asinh", "arcsinh"),
    ("atanh", "arctanh"),
];

const fn same_text(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

/// Index of the `scipy.special` ufunc at `position` in `SpecialFunction::ALL`, at compile time.
/// The special ufuncs close the table in that order. Some, such as `expm1`, share a name with a
/// NumPy ufunc, so they cannot be found by name.
pub(in crate::python) const fn special_index(position: usize) -> u64 {
    let index = UFUNCS.len() - SpecialFunction::ALL.len() + position;
    assert!(
        same_text(UFUNCS[index].name, SpecialFunction::ALL[position].name()),
        "special ufuncs must close the table in SpecialFunction::ALL order"
    );
    index as u64
}

/// Index of a ufunc by name or alias, at compile time, for module value tables.
pub(in crate::python) const fn index_of(name: &str) -> u64 {
    let mut alias = 0;
    let mut target = name;
    while alias < ALIASES.len() {
        if same_text(ALIASES[alias].0, name) {
            target = ALIASES[alias].1;
        }
        alias += 1;
    }
    let mut index = 0;
    while index < UFUNCS.len() {
        if same_text(UFUNCS[index].name, target) {
            return index as u64;
        }
        index += 1;
    }
    panic!("unknown ufunc name")
}

/// Index of a ufunc by name, including aliases.
pub(in crate::python) fn find(name: &str) -> Option<usize> {
    let name = ALIASES
        .iter()
        .find(|(alias, _)| *alias == name)
        .map_or(name, |(_, target)| target);
    UFUNCS.iter().position(|ufunc| ufunc.name == name)
}

pub(in crate::python) fn named(name: &str) -> usize {
    find(name).unwrap_or_else(|| panic!("ufunc {name} is registered"))
}

/// The `numpy.ufunc` type; values carry an index into [`UFUNCS`].
pub(in crate::python) static UFUNC: ValueKindDef = ValueKindDef {
    name: "numpy.ufunc",
    construct: construct_ufunc,
    slots: ValueKindSlots {
        repr: Some(slot_ufunc_repr),
        ..NO_SLOTS
    },
    methods: &[
        ufunc_method("reduce", method_reduce),
        ufunc_method("accumulate", method_accumulate),
        ufunc_method("outer", method_outer),
    ],
    getters: &[
        ufunc_getter("__name__", get_name),
        ufunc_getter("nin", get_nin),
        ufunc_getter("nout", get_nout),
        ufunc_getter("nargs", get_nargs),
        ufunc_getter("identity", get_identity),
    ],
    bases: &[],
    call: Some(call_ufunc_value),
    numeric: None,
};

const fn ufunc_method(
    name: &'static str,
    call: fn(&mut dyn PyRuntime, PyValue, CallArgs) -> PyResult,
) -> MethodDef {
    MethodDef {
        type_name: "numpy.ufunc",
        name,
        call,
    }
}

const fn ufunc_getter(
    name: &'static str,
    get: fn(&mut dyn PyRuntime, PyValue) -> PyResult,
) -> GetterDef {
    GetterDef {
        owner: "numpy.ufunc",
        name,
        get,
    }
}

fn construct_ufunc(_runtime: &mut dyn PyRuntime, _args: CallArgs) -> PyResult {
    Err(PyError::type_error("cannot create 'numpy.ufunc' instances"))
}

/// The ufunc index carried by a `numpy.ufunc` value.
pub(in crate::python) fn ufunc_index(runtime: &dyn PyRuntime, value: &PyValue) -> Option<usize> {
    let index = runtime.value_kind_payload(value, &UFUNC)? as usize;
    (index < UFUNCS.len()).then_some(index)
}

fn receiver(runtime: &dyn PyRuntime, value: &PyValue) -> usize {
    ufunc_index(runtime, value).expect("ufunc methods receive ufunc values")
}

fn slot_ufunc_repr(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Option<PyValue>> {
    let name = UFUNCS[receiver(runtime, &value)].name;
    runtime.new_string(format!("<ufunc '{name}'>")).map(Some)
}

fn get_name(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let name = UFUNCS[receiver(runtime, &value)].name;
    runtime.new_string(name.to_string())
}

fn get_nin(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(Value::Int(UFUNCS[receiver(runtime, &value)].nin() as i64))
}

fn get_nout(_runtime: &mut dyn PyRuntime, _value: PyValue) -> PyResult {
    Ok(Value::Int(1))
}

fn get_nargs(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    Ok(Value::Int(
        UFUNCS[receiver(runtime, &value)].nin() as i64 + 1,
    ))
}

fn get_identity(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    let ufunc = &UFUNCS[receiver(runtime, &value)];
    Ok(match ufunc.family {
        Family::Arith {
            op: ArithOp::Add, ..
        }
        | Family::Logical(LogicalOp::Or | LogicalOp::Xor)
        | Family::Bitwise(BitOp::Or | BitOp::Xor) => Value::Int(0),
        Family::Arith {
            op: ArithOp::Multiply,
            ..
        } => Value::Int(1),
        Family::Logical(LogicalOp::And) => Value::Bool(true),
        Family::Bitwise(BitOp::And) => Value::Int(-1),
        _ => Value::None,
    })
}

fn call_ufunc_value(
    runtime: &mut dyn PyRuntime,
    receiver_value: PyValue,
    args: CallArgs,
) -> PyResult {
    let index = receiver(runtime, &receiver_value);
    call(runtime, index, args)
}

fn method_reduce(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    let index = receiver(runtime, &receiver_value);
    super::reduce::ufunc_reduce(runtime, index, args)
}

fn method_accumulate(
    runtime: &mut dyn PyRuntime,
    receiver_value: PyValue,
    args: CallArgs,
) -> PyResult {
    let index = receiver(runtime, &receiver_value);
    super::reduce::ufunc_accumulate(runtime, index, args)
}

/// `ufunc.outer(a, b)`: apply the ufunc to every pair, with shape `a.shape + b.shape`.
fn method_outer(runtime: &mut dyn PyRuntime, receiver_value: PyValue, args: CallArgs) -> PyResult {
    let index = receiver(runtime, &receiver_value);
    let name = UFUNCS[index].name;
    if UFUNCS[index].nin() != 2 {
        return Err(PyError::value_error(
            "outer product only supported for binary functions",
        ))
        .map_err(|error: PyError| PyError::value_error(format!("{name}: {}", error.message)));
    }
    args.expect_positional("outer", 2, 2)?;
    let left = convert::as_array(runtime, args.positional()[0])?;
    let right = convert::as_array(runtime, args.positional()[1])?;
    let mut shape = left.shape().to_vec();
    shape.extend(std::iter::repeat_n(1, right.ndim()));
    let mut strides = left.strides().to_vec();
    strides.extend(std::iter::repeat_n(0, right.ndim()));
    let expanded =
        super::array::new_view(runtime, &left, left.dtype, shape, strides, left.view.offset)?;
    apply(
        runtime,
        index,
        &[expanded.value(), right.value()],
        &Options::default(),
    )
}

/// Keyword options shared by every ufunc call.
#[derive(Clone, Default)]
pub(in crate::python) struct Options {
    pub out: Option<Array>,
    pub dtype: Option<DType>,
    pub casting: Option<Casting>,
    /// The call comes from a Python operator such as `a + b`. When every operand is a scalar,
    /// NumPy's scalar math also reports integer overflow and names the operation `scalar add`.
    pub operator: bool,
    /// `out=...`: return a 0-d result as an array rather than a scalar.
    pub keep_array: bool,
    /// `where=`, unless it is the literal `True`.
    pub mask: Option<PyValue>,
    /// `order=` for an allocated output; `None` is NumPy's default, `K`.
    pub order: Option<Order>,
}

impl Options {
    fn operator() -> Self {
        Self {
            operator: true,
            ..Self::default()
        }
    }
}

/// `np.<ufunc>(*inputs, out=None, dtype=None, casting='same_kind')`.
pub(in crate::python) fn call(
    runtime: &mut dyn PyRuntime,
    index: usize,
    args: CallArgs,
) -> PyResult {
    let ufunc = &UFUNCS[index];
    let nin = ufunc.nin();
    let positional = args.positional();
    if positional.len() < nin || positional.len() > nin + 1 {
        return Err(PyError::type_error(format!(
            "{}() takes from {nin} to {} positional arguments but {} were given",
            ufunc.name,
            nin + 1,
            positional.len()
        )));
    }
    let (mut options, out) = keyword_options(runtime, ufunc.name, &args, positional.get(nin))?;
    if options.mask.is_some() && out.is_none() {
        super::errstate::warn_where_without_out(runtime)?;
    }
    match out {
        Some(out) if runtime.is_ellipsis(&out) => options.keep_array = true,
        out => options.out = out_array(runtime, out)?,
    }
    apply(runtime, index, &positional[..nin], &options)
}

/// Parse a ufunc call's keywords. Returns the options and the `out` argument, positional or
/// keyword, still unparsed.
fn keyword_options(
    runtime: &mut dyn PyRuntime,
    name: &str,
    args: &CallArgs,
    positional_out: Option<&PyValue>,
) -> PyResult<(Options, Option<PyValue>)> {
    let mut options = Options::default();
    let mut out = positional_out.copied();
    for (keyword, value) in args.keywords() {
        match keyword.as_str() {
            "out" => {
                if out.is_some() {
                    return Err(PyError::type_error(
                        "cannot specify 'out' as both a positional and keyword argument",
                    ));
                }
                out = Some(*value);
            }
            "dtype" => options.dtype = super::args::optional_dtype(runtime, Some(*value))?,
            "casting" => {
                let text = runtime.string_value(value)?.unwrap_or_default();
                options.casting = Some(Casting::parse(&text)?);
            }
            "where" if value.bool_value() == Some(true) => {}
            "where" => options.mask = Some(*value),
            "order" => options.order = Some(Order::parse(runtime, Some(*value), Order::K)?),
            "subok" => {}
            _ => {
                return Err(PyError::type_error(format!(
                    "{name}() got an unexpected keyword argument '{keyword}'"
                )))
            }
        }
    }
    Ok((options, out))
}

/// Accept `out=array`, `out=(array,)`, or `out=None`.
fn out_array(runtime: &mut dyn PyRuntime, value: Option<PyValue>) -> PyResult<Option<Array>> {
    let Some(value) = value.filter(|value| !value.is_none()) else {
        return Ok(None);
    };
    let value = if runtime.kind(&value)? == PyKind::Tuple {
        let tuple = value.cast(runtime)?;
        let items = runtime.tuple_items(tuple)?;
        match items.as_slice() {
            [single] if !single.is_none() => *single,
            [_] => return Ok(None),
            _ => {
                return Err(PyError::value_error(
                    "The 'out' tuple must have exactly one entry per ufunc output",
                ))
            }
        }
    } else {
        value
    };
    Array::from_value(runtime, value)
        .map(Some)
        .map_err(|_| PyError::type_error("return arrays must be of ArrayType"))
}

use super::super::super::native::PyValueCast;

/// One prepared ufunc operand.
pub(in crate::python) enum Operand {
    Array(Array),
    Weak {
        value: PyValue,
        leaf: Leaf,
        weak: Weak,
    },
}

impl Operand {
    pub(in crate::python) fn dtype(&self) -> Option<DType> {
        match self {
            Self::Array(array) => Some(array.dtype),
            Self::Weak { .. } => None,
        }
    }

    pub(in crate::python) fn shape(&self) -> &[usize] {
        match self {
            Self::Array(array) => array.shape(),
            Self::Weak { .. } => &[],
        }
    }
}

/// Prepare one input, reporting whether it is a scalar (a Python number or a NumPy scalar).
pub(in crate::python) fn operand(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<(Operand, bool)> {
    if let Some((weak, leaf)) = convert::weak_scalar(runtime, &value)? {
        return Ok((Operand::Weak { value, leaf, weak }, true));
    }
    let scalar = super::scalar::unbox(runtime, &value).is_some();
    let array = convert::as_array(runtime, value)?;
    Ok((Operand::Array(array), scalar))
}

/// The dtype operands promote to under NEP 50.
pub(in crate::python) fn common_dtype(operands: &[Operand]) -> PyResult<DType> {
    let strong = operands
        .iter()
        .filter_map(Operand::dtype)
        .collect::<Vec<_>>();
    let weak = operands
        .iter()
        .filter_map(|operand| match operand {
            Operand::Weak { weak, .. } => Some(*weak),
            Operand::Array(_) => None,
        })
        .collect::<Vec<_>>();
    dtype::result_type(&strong, &weak)
}

/// A `TypeError` for a ufunc with no registered loop accepting `inputs`.
fn no_loop(name: &str, inputs: &[DType]) -> PyError {
    let types = inputs
        .iter()
        .map(|dtype| dtype.repr())
        .collect::<Vec<_>>()
        .join(", ");
    PyError::type_error(format!("ufunc '{name}' not supported for dtypes ({types})"))
}

/// A ufunc that mixes `str` with numbers shares no loop for either type. `multiply` by an
/// integer gets a more specific message pointing at `numpy.strings`, since that combination has
/// its own dedicated replacement; every other mix falls through to [`no_loop`].
fn string_mix_error(ufunc: &UfuncDef, operands: &[Operand]) -> Option<PyError> {
    let kind = |operand: &Operand| operand.dtype().map(DType::kind);
    let strings = operands
        .iter()
        .filter(|operand| kind(operand) == Some(Kind::Str))
        .count();
    let objects = operands
        .iter()
        .any(|operand| kind(operand) == Some(Kind::Object));
    if strings == 0 || strings == operands.len() || objects {
        return None;
    }
    let integer = |operand: &Operand| match operand {
        Operand::Array(array) => array.dtype.is_integer(),
        Operand::Weak { weak, .. } => *weak == Weak::Int,
    };
    let multiply = matches!(
        ufunc.family,
        Family::Arith {
            op: ArithOp::Multiply,
            ..
        }
    );
    if multiply && operands.iter().any(integer) {
        return Some(PyError::type_error(
            "The 'out' kwarg is necessary when using the string multiply ufunc directly. Use \
             numpy.strings.multiply to multiply strings without specifying 'out'.",
        ));
    }
    let dtypes = operands
        .iter()
        .map(|operand| match operand {
            Operand::Array(array) => array.dtype,
            Operand::Weak { weak, .. } => weak.default_dtype(),
        })
        .collect::<Vec<_>>();
    Some(no_loop(ufunc.name, &dtypes))
}

/// A `TypeError` for a ufunc whose resolver categorically rejects the input dtype (for example
/// a complex operand to a real-only function), as opposed to [`no_loop`]'s dtype-combination
/// mismatch.
fn not_supported(name: &str) -> PyError {
    PyError::type_error(format!("ufunc '{name}' not supported for the input types"))
}

/// Loop and output dtypes chosen for one call.
pub(in crate::python) struct Resolved {
    /// Dtype every input is cast to.
    pub input: DType,
    pub output: DType,
}

pub(in crate::python) fn resolve(
    ufunc: &UfuncDef,
    common: DType,
    inputs: &[DType],
    requested: Option<DType>,
) -> PyResult<Resolved> {
    // Loops run in native byte order, so results are native even for big-endian operands.
    let common = common.native();
    let requested = requested.map(DType::native);
    let name = ufunc.name;
    let same = |dtype: DType| Resolved {
        input: dtype,
        output: dtype,
    };
    let object = common.kind() == Kind::Object;
    if object {
        if matches!(ufunc.object, ObjectLoop::Missing) {
            return Err(not_supported(name));
        }
        let output = match ufunc.family {
            Family::Compare(_) => DType::BOOL,
            _ => DType::OBJECT,
        };
        return Ok(Resolved {
            input: DType::OBJECT,
            output,
        });
    }
    let string = common.kind() == Kind::Str;
    match ufunc.family {
        Family::Arith { op, bools, complex } => {
            if string {
                return match op {
                    ArithOp::Add | ArithOp::Maximum | ArithOp::Minimum => Ok(same(common)),
                    _ => Err(no_loop(name, inputs)),
                };
            }
            let dtype = requested.unwrap_or(common);
            let dtype = bool_loop(name, dtype, bools, inputs)?;
            if dtype.category() == Category::Complex && !complex {
                return Err(not_supported(name));
            }
            Ok(same(dtype))
        }
        Family::TrueDivide => {
            if string {
                return Err(no_loop(name, inputs));
            }
            Ok(same(
                requested.unwrap_or_else(|| dtype::true_divide_dtype(common)),
            ))
        }
        Family::Compare(_) => Ok(Resolved {
            input: common,
            output: DType::BOOL,
        }),
        Family::Logical(_) | Family::LogicalNot => {
            if string {
                return Err(no_loop(name, inputs));
            }
            Ok(Resolved {
                input: DType::BOOL,
                output: DType::BOOL,
            })
        }
        Family::Bitwise(op) => {
            let dtype = requested.unwrap_or(common);
            let dtype = match (dtype.category(), op) {
                (Category::Bool, BitOp::LeftShift | BitOp::RightShift) => DType::INT8,
                (Category::Bool | Category::Signed | Category::Unsigned, _) => dtype,
                _ => return Err(not_supported(name)),
            };
            Ok(same(dtype))
        }
        Family::Invert => match common.category() {
            Category::Bool | Category::Signed | Category::Unsigned => Ok(same(common)),
            _ => Err(not_supported(name)),
        },
        Family::Float2(_) | Family::Float { .. } => {
            if string {
                return Err(no_loop(name, inputs));
            }
            let complex_ok = matches!(ufunc.family, Family::Float { complex: true, .. });
            let dtype = requested.unwrap_or_else(|| common.smallest_float_for());
            if dtype.category() == Category::Complex && !complex_ok {
                return Err(not_supported(name));
            }
            Ok(same(dtype))
        }
        Family::Unary { op, bools, complex } => {
            if string {
                return Err(no_loop(name, inputs));
            }
            let dtype = bool_loop(name, requested.unwrap_or(common), bools, inputs)?;
            if dtype.category() == Category::Complex && !complex {
                return Err(not_supported(name));
            }
            let output = if op == UnaryOp::Absolute {
                dtype.real_part()
            } else {
                dtype
            };
            Ok(Resolved {
                input: dtype,
                output,
            })
        }
        Family::Predicate(_) => {
            if string {
                return Err(not_supported(name));
            }
            Ok(Resolved {
                input: common,
                output: DType::BOOL,
            })
        }
        Family::Special(function) => resolve_special(function, common, requested).map(same),
    }
}

/// The loop dtype of a `scipy.special` function. SciPy registers `float32` and `float64`
/// loops, and NumPy takes the first one every input casts to safely, unless one matches the
/// inputs exactly: `int16` computes in `float32`, `int32` in `float64`.
fn resolve_special(
    function: SpecialFunction,
    common: DType,
    requested: Option<DType>,
) -> PyResult<DType> {
    let name = function.name();
    let complex = function.supports_complex();
    let complex_error = || {
        PyError::not_implemented_error(format!(
            "complex input to scipy.special.{name} is not supported by shellsim's SciPy"
        ))
    };
    if let Some(dtype) = requested {
        return match dtype.category() {
            Category::Float if dtype.kind() != Kind::Float16 => Ok(dtype),
            Category::Complex if complex => Err(complex_error()),
            _ => Err(PyError::type_error(format!(
                "No loop matching the specified signature and casting was found for ufunc {name}"
            ))),
        };
    }
    match common.category() {
        Category::Bool | Category::Signed | Category::Unsigned | Category::Float => {}
        Category::Complex if complex => return Err(complex_error()),
        Category::Complex | Category::Str | Category::Object => return Err(not_supported(name)),
    }
    let single = dtype::can_cast(common, DType::FLOAT32, Casting::Safe);
    Ok(if single {
        DType::FLOAT32
    } else {
        DType::FLOAT64
    })
}

fn bool_loop(name: &str, dtype: DType, bools: BoolLoop, inputs: &[DType]) -> PyResult<DType> {
    if dtype.kind() != Kind::Bool {
        return Ok(dtype);
    }
    match bools {
        BoolLoop::Keep => Ok(dtype),
        BoolLoop::Int8 => Ok(DType::INT8),
        BoolLoop::Reject(message) => Err(PyError::type_error(message)),
        BoolLoop::Missing => Err(no_loop(name, inputs)),
    }
}

/// Run ufunc `index` on `inputs` and report its floating-point errors.
pub(in crate::python) fn apply(
    runtime: &mut dyn PyRuntime,
    index: usize,
    inputs: &[PyValue],
    options: &Options,
) -> PyResult {
    let evaluated = evaluate(runtime, index, inputs, options)?;
    evaluated.report(runtime, UFUNCS[index].name)?;
    Ok(evaluated.value)
}

/// A ufunc result whose floating-point errors are not reported yet.
pub(in crate::python) struct Evaluated {
    pub value: PyValue,
    /// Flags to report; integer overflow is already dropped where NumPy ignores it.
    pub flags: FpFlags,
    /// Every operand was a scalar and the call came from an operator, so NumPy's scalar math
    /// names the operation `scalar <name>`.
    pub scalar_math: bool,
}

impl Evaluated {
    pub(in crate::python) fn report(
        &self,
        runtime: &mut dyn PyRuntime,
        name: &str,
    ) -> PyResult<()> {
        if !self.flags.any() {
            return Ok(());
        }
        let name = if self.scalar_math {
            format!("scalar {name}")
        } else {
            name.to_string()
        };
        super::errstate::report(runtime, &name, self.flags)
    }
}

/// Run ufunc `index` on `inputs`, storing into `out=` when given, and return the flags its
/// loop raised without reporting them.
pub(in crate::python) fn evaluate(
    runtime: &mut dyn PyRuntime,
    index: usize,
    inputs: &[PyValue],
    options: &Options,
) -> PyResult<Evaluated> {
    if let Some(mask) = options.mask {
        return masked::evaluate(runtime, index, inputs, options, mask);
    }
    let ufunc = &UFUNCS[index];
    let mut operands = Vec::with_capacity(inputs.len());
    let mut all_scalars = true;
    for value in inputs {
        let (operand, scalar) = operand(runtime, *value)?;
        all_scalars &= scalar;
        operands.push(operand);
    }
    if let Some(error) = string_mix_error(ufunc, &operands) {
        return Err(error);
    }
    let common = comparison_safe_common(ufunc, &operands)?;
    let input_dtypes = operands
        .iter()
        .map(|operand| operand.dtype().unwrap_or(common))
        .collect::<Vec<_>>();
    let resolved = resolve(ufunc, common, &input_dtypes, options.dtype)?;
    let casting = options.casting.unwrap_or(Casting::SameKind);
    if options.dtype.is_some() {
        for (position, dtype) in input_dtypes.iter().enumerate() {
            if !dtype::can_cast(*dtype, resolved.input, casting) {
                return Err(PyError::type_error(format!(
                    "Cannot cast ufunc '{}' input {position} from {} to {} with casting rule '{}'",
                    ufunc.name,
                    dtype.repr(),
                    resolved.input.repr(),
                    casting.name()
                )));
            }
        }
    }
    let shapes = operands.iter().map(Operand::shape).collect::<Vec<_>>();
    let shape = broadcast_shapes(&shapes)?;
    if let Some(out) = &options.out {
        if out.shape() != shape.as_slice() {
            let full = broadcast_shapes(&[out.shape(), &shape]).ok();
            if full.as_deref() != Some(out.shape()) {
                return Err(PyError::value_error(format!(
                    "non-broadcastable output operand with shape {} doesn't match the broadcast \
                     shape {}",
                    super::array::format_shape(out.shape()),
                    super::array::format_shape(&shape)
                )));
            }
        }
        check_output_cast(ufunc, resolved.output, out, casting)?;
    }
    let shape = match &options.out {
        Some(out) => out.shape().to_vec(),
        None => shape,
    };
    if let (
        Family::Arith {
            op: ArithOp::Power, ..
        },
        true,
    ) = (ufunc.family, resolved.input.is_integer())
    {
        reject_negative_exponent(runtime, &operands[1])?;
    }
    let prepared = operands
        .iter()
        .map(|operand| prepare(runtime, operand, resolved.input))
        .collect::<PyResult<Vec<_>>>()?;
    // The loop fills the output in its memory order: C order over the permuted axes.
    let axes = match options.out {
        Some(_) => (0..shape.len()).collect(),
        None => output_axes(&operands, &shape, options.order.unwrap_or(Order::K))?,
    };
    let loop_shape = axes.iter().map(|axis| shape[*axis]).collect::<Vec<_>>();
    let prepared = prepared
        .iter()
        .map(|input| layout::broadcast_reading_order(input, &shape, &axes))
        .collect::<PyResult<Vec<_>>>()?;
    // NumPy's scalar math calls `pow` directly, without the loops' constant-exponent cases.
    let constant_exponent = matches!(
        ufunc.family,
        Family::Arith {
            op: ArithOp::Power,
            ..
        }
    ) && !(options.operator && all_scalars)
        && constant_operand(&operands[1], element_count(&loop_shape)?);
    let (buffer, output_dtype, mut flags) = run(
        runtime,
        ufunc,
        &resolved,
        &prepared,
        &loop_shape,
        constant_exponent,
    )?;
    let scalar_math = options.operator && all_scalars;
    let division = matches!(
        ufunc.family,
        Family::Arith {
            op: ArithOp::FloorDivide,
            ..
        }
    );
    if resolved.input.is_integer() && !scalar_math && !division {
        // Integer loops wrap silently; scalar operators and integer division, whose only
        // overflow is `MIN // -1`, report it.
        flags.overflow = false;
    }
    let result = layout::new_array(runtime, buffer, output_dtype, shape, &axes)?;
    let value = if let Some(out) = &options.out {
        super::array::assign(runtime, out, &result)?;
        out.value()
    } else if result.ndim() == 0 && !options.keep_array {
        convert::element_to_scalar(runtime, &result, result.view.offset)?
    } else {
        result.value()
    };
    Ok(Evaluated {
        value,
        flags,
        scalar_math,
    })
}

/// Whether an operand is the same for every element of NumPy's inner loop, where its stride is
/// zero: a scalar or 0-d array, or a one-element array broadcast to `count > 1` elements. A
/// one-element array that is not stretched keeps its stride.
fn constant_operand(operand: &Operand, count: usize) -> bool {
    match operand {
        Operand::Array(array) => array.size() == 1 && (array.ndim() == 0 || count > 1),
        Operand::Weak { .. } => true,
    }
}

/// The memory order of an allocated output of `shape`, as NumPy's iterator applies `order`:
/// `A` means Fortran order when every array operand is Fortran-contiguous, and `K` follows
/// the operands' strides.
fn output_axes(operands: &[Operand], shape: &[usize], order: Order) -> PyResult<Vec<usize>> {
    let arrays = operands.iter().filter_map(|operand| match operand {
        Operand::Array(array) => Some(array),
        Operand::Weak { .. } => None,
    });
    let order = match order {
        Order::A if shape.len() > 1 && arrays.clone().all(layout::is_f_contiguous) => Order::F,
        Order::A => Order::C,
        order => order,
    };
    if order != Order::K {
        return Ok(layout::axes(order, shape.len()));
    }
    let strides = arrays
        .map(|array| broadcast_strides(&array.view, shape))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(layout::iteration_axes(&strides, shape))
}

/// Check that a loop's `output` dtype casts to `out=` under `casting`.
fn check_output_cast(
    ufunc: &UfuncDef,
    output: DType,
    out: &Array,
    casting: Casting,
) -> PyResult<()> {
    if dtype::can_cast(output, out.dtype, casting) {
        return Ok(());
    }
    Err(PyError::type_error(format!(
        "Cannot cast ufunc '{}' output from {} to {} with casting rule '{}'",
        ufunc.name,
        output.repr(),
        out.dtype.repr(),
        casting.name()
    )))
}

/// The `__divmod__`/`__rdivmod__` operator protocol on ndarrays: `(floor_divide(x1, x2),
/// remainder(x1, x2))` from one pass, so floating-point errors are reported once under the name
/// `divmod`. `np.divmod` itself is frozen Python (`numpy._math.divmod`), two independent ufunc
/// calls; this function exists only for `divmod(array, array)`, a different call path.
pub(in crate::python) fn divmod(
    runtime: &mut dyn PyRuntime,
    inputs: &[PyValue],
    options: &Options,
    outs: [Option<Array>; 2],
) -> PyResult {
    let [quotient_out, remainder_out] = outs;
    let quotient_options = Options {
        out: quotient_out,
        ..options.clone()
    };
    let remainder_options = Options {
        out: remainder_out,
        ..options.clone()
    };
    let quotient = evaluate(runtime, named("floor_divide"), inputs, &quotient_options)?;
    let remainder = evaluate(runtime, named("remainder"), inputs, &remainder_options)?;
    let mut merged = quotient.flags;
    merged.merge(remainder.flags);
    let combined = Evaluated {
        value: runtime.new_tuple(vec![quotient.value, remainder.value])?,
        flags: merged,
        scalar_math: quotient.scalar_math,
    };
    combined.report(runtime, "divmod")?;
    Ok(combined.value)
}

pub(in crate::python) fn slot_divmod(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    if defers_to(runtime, right)? {
        return Ok(None);
    }
    divmod(runtime, &[left, right], &Options::operator(), [None, None]).map(Some)
}

pub(in crate::python) fn slot_reflected_divmod(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    if defers_to(runtime, right)? {
        return Ok(None);
    }
    divmod(runtime, &[right, left], &Options::operator(), [None, None]).map(Some)
}

/// Comparisons with Python ints outside the array's integer range compare exactly instead of
/// raising, as NumPy 2 does.
fn comparison_safe_common(ufunc: &UfuncDef, operands: &[Operand]) -> PyResult<DType> {
    let common = common_dtype(operands)?;
    if !matches!(ufunc.family, Family::Compare(_)) || !common.is_integer() {
        return Ok(common);
    }
    let out_of_range = operands.iter().any(|operand| match operand {
        Operand::Weak {
            leaf: Leaf::Int(value),
            ..
        } => convert::checked_int(*value, common).is_err(),
        Operand::Weak {
            leaf: Leaf::BigInt(_),
            ..
        } => true,
        _ => false,
    });
    if !out_of_range {
        return Ok(common);
    }
    Ok(
        if common.category() == Category::Unsigned && common.kind() == Kind::UInt64 {
            DType::FLOAT64
        } else {
            DType::INT64
        },
    )
}

fn reject_negative_exponent(runtime: &mut dyn PyRuntime, exponent: &Operand) -> PyResult<()> {
    let negative = match exponent {
        Operand::Weak {
            leaf: Leaf::Int(value),
            ..
        } => *value < 0,
        Operand::Weak { .. } => false,
        Operand::Array(array) => match array.dtype.category() {
            Category::Signed => {
                let wide = convert::cast_array(runtime, array, DType::INT64, false)?;
                let values = super::array::read_elements::<i64>(runtime, &wide)?;
                values.iter().any(|value| *value < 0)
            }
            _ => false,
        },
    };
    if negative {
        return Err(PyError::value_error(
            "Integers to negative integer powers are not allowed.",
        ));
    }
    Ok(())
}

pub(in crate::python) fn prepare(
    runtime: &mut dyn PyRuntime,
    operand: &Operand,
    dtype: DType,
) -> PyResult<Array> {
    match operand {
        Operand::Array(array) if dtype.kind() == Kind::Str && array.dtype.kind() == Kind::Str => {
            Ok(array.clone())
        }
        // Storage is native whatever the dtype's byte order, so a big-endian operand reads as
        // its native twin without a copy.
        Operand::Array(array) if array.dtype.native() == dtype => Ok(Array {
            dtype,
            ..array.clone()
        }),
        Operand::Array(array) => convert::cast_array(runtime, array, dtype, false),
        Operand::Weak { value, leaf, .. } => convert::weak_array(runtime, *value, leaf, dtype),
    }
}

/// Relative cost of one element of a ufunc loop, in CPU units.
fn element_cost(family: Family) -> u64 {
    match family {
        Family::Float { .. } | Family::Float2(_) => 4,
        Family::Special(_) => SPECIAL_ELEMENT_COST,
        Family::Arith {
            op: ArithOp::Power, ..
        } => 4,
        _ => 1,
    }
}

/// Execute the loop into a fresh C-contiguous buffer. `constant_exponent` says that a `power`
/// exponent is the same for every element (see [`constant_operand`]).
fn run(
    runtime: &mut dyn PyRuntime,
    ufunc: &UfuncDef,
    resolved: &Resolved,
    inputs: &[Array],
    shape: &[usize],
    constant_exponent: bool,
) -> PyResult<(PyArrayBuffer, DType, FpFlags)> {
    let count = element_count(shape)?;
    runtime.charge_cpu((count as u64).saturating_mul(element_cost(ufunc.family)) + 1)?;
    if resolved.input.kind() == Kind::Object {
        return object_loop(runtime, ufunc, resolved, inputs, shape);
    }
    if resolved.input.kind() == Kind::Str {
        return string_loop(runtime, ufunc, inputs, shape);
    }
    let output = resolved.output;
    reserve_elements(runtime, output, count)?;
    let mut bytes = vec![0u8; count * output.itemsize()];
    let mut flags = FpFlags::default();
    let strides = inputs
        .iter()
        .map(|input| broadcast_strides(&input.view, shape))
        .collect::<PyResult<Vec<_>>>()?;
    let handles = inputs.iter().map(|input| input.handle).collect::<Vec<_>>();
    let kind = resolved.input.kind();
    let family = ufunc.family;
    if let Family::Special(function) = family {
        let offsets = inputs
            .iter()
            .zip(&strides)
            .map(|(input, strides)| Offsets::new(shape, strides, input.view.offset))
            .collect();
        match kind {
            Kind::Float32 => special_loop::<f32>(runtime, &handles, offsets, &mut bytes, |args| {
                function.eval_f32(args)
            })?,
            Kind::Float64 => special_loop::<f64>(runtime, &handles, offsets, &mut bytes, |args| {
                function.eval(args)
            })?,
            _ => {
                return Err(PyError::runtime_error(
                    "ufunc loop has no kernel for its dtype",
                ))
            }
        }
        return Ok((PyArrayBuffer::Bytes(bytes), output, flags));
    }
    runtime.read_arrays(&handles, &mut |arrays| {
        let data = arrays
            .iter()
            .map(|array| match array.data {
                PyArrayData::Bytes(bytes) => Ok(bytes),
                PyArrayData::Values(_) => Err(PyError::runtime_error("numeric loop saw objects")),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let offsets = inputs
            .iter()
            .zip(&strides)
            .map(|(input, strides)| Offsets::new(shape, strides, input.view.offset))
            .collect::<Vec<_>>();
        numeric_loop(
            family,
            kind,
            &data,
            offsets,
            &mut bytes,
            &mut flags,
            constant_exponent,
        )
    })?;
    Ok((PyArrayBuffer::Bytes(bytes), output, flags))
}

/// Dispatch one numeric family to its monomorphized kernel.
fn numeric_loop(
    family: Family,
    kind: Kind,
    data: &[&[u8]],
    mut offsets: Vec<Offsets>,
    output: &mut [u8],
    flags: &mut FpFlags,
    constant_exponent: bool,
) -> PyResult<()> {
    let unsupported = || {
        Err(PyError::runtime_error(
            "ufunc loop has no kernel for its dtype",
        ))
    };
    match family {
        // NumPy's `float32` and `float64` power loops take `sqrt` for a constant exponent of
        // 0.5, which keeps `-0.0` and gives NaN for `-inf` where `pow` gives `0.0` and `inf`.
        // Its other constant exponents (-1, 0, 1 and 2) round as `pow` does.
        Family::Arith {
            op: ArithOp::Power, ..
        } if constant_exponent && kind == Kind::Float64 => {
            binary_loop::<f64, f64>(data, &mut offsets, output, flags, |a, b, flags| {
                if b == 0.5 {
                    let root = a.sqrt();
                    flags.invalid |= root.is_nan() && !a.is_nan();
                    root
                } else {
                    a.power(b, flags)
                }
            })
        }
        Family::Arith {
            op: ArithOp::Power, ..
        } if constant_exponent && kind == Kind::Float32 => {
            binary_loop::<f32, f32>(data, &mut offsets, output, flags, |a, b, flags| {
                if b == 0.5 {
                    let root = a.sqrt();
                    flags.invalid |= root.is_nan() && !a.is_nan();
                    root
                } else {
                    a.power(b, flags)
                }
            })
        }
        Family::Arith { op, .. } => dispatch_numeric!(kind, T => {
            let operation = arith_fn::<T>(op);
            binary_loop::<T, T>(data, &mut offsets, output, flags, operation)
        }, _ => unsupported()),
        Family::TrueDivide => dispatch_numeric!(kind, T => {
            binary_loop::<T, T>(data, &mut offsets, output, flags, |a, b, flags| a.divide(b, flags))
        }, _ => unsupported()),
        Family::Compare(op) => dispatch_numeric!(kind, T => {
            let test = compare_fn(op);
            binary_loop::<T, bool>(data, &mut offsets, output, flags, move |a, b, _| test(a.compare(b)))
        }, _ => unsupported()),
        Family::Logical(op) => {
            let operation: fn(bool, bool) -> bool = match op {
                LogicalOp::And => |a, b| a && b,
                LogicalOp::Or => |a, b| a || b,
                LogicalOp::Xor => |a, b| a != b,
            };
            binary_loop::<bool, bool>(data, &mut offsets, output, flags, move |a, b, _| {
                operation(a, b)
            })
        }
        Family::LogicalNot => {
            unary_loop::<bool, bool>(data, &mut offsets, output, flags, |a, _| !a)
        }
        Family::Bitwise(op) => dispatch_integer!(kind, T => {
            let operation: fn(T, T) -> T = match op {
                BitOp::And => |a: T, b: T| a.bit_and(b),
                BitOp::Or => |a: T, b: T| a.bit_or(b),
                BitOp::Xor => |a: T, b: T| a.bit_xor(b),
                BitOp::LeftShift => |a: T, b: T| a.left_shift(b),
                BitOp::RightShift => |a: T, b: T| a.right_shift(b),
            };
            binary_loop::<T, T>(data, &mut offsets, output, flags, move |a, b, _| operation(a, b))
        }, _ => unsupported()),
        Family::Invert => dispatch_integer!(kind, T => {
            unary_loop::<T, T>(data, &mut offsets, output, flags, |a: T, _| a.invert())
        }, _ => unsupported()),
        Family::Unary { op, .. } => {
            if op == UnaryOp::Absolute {
                if let Some(result) = dispatch_complex!(kind, T => Some({
                    unary_loop::<T, <T as ComplexMagnitude>::Real>(data, &mut offsets, output, flags, |a: T, _| a.magnitude())
                }), _ => None)
                {
                    return result;
                }
            }
            dispatch_numeric!(kind, T => {
                let operation = unary_fn::<T>(op);
                unary_loop::<T, T>(data, &mut offsets, output, flags, operation)
            }, _ => unsupported())
        }
        Family::Float { op, .. } => {
            if let Some(result) = dispatch_real!(kind, T => Some({
                let (operation, pole) = float_fn(op);
                unary_loop::<T, T>(data, &mut offsets, output, flags, move |a: T, flags| {
                    let result = a.map(operation.0, operation.1);
                    float_unary_flags(a.to_f64(), result.to_f64(), pole, flags);
                    result
                })
            }), _ => None)
            {
                return result;
            }
            dispatch_complex!(kind, T => {
                let operation = complex_fn(op)
                    .ok_or_else(|| PyError::runtime_error("complex loop is missing"))?;
                unary_loop::<T, T>(data, &mut offsets, output, flags, move |a: T, _| {
                    let (real, imag) = operation(a.parts());
                    T::from_parts(real, imag)
                })
            }, _ => unsupported())
        }
        Family::Float2(op) => dispatch_real!(kind, T => {
            let operation = float2_fn(op);
            binary_loop::<T, T>(data, &mut offsets, output, flags, move |a: T, b: T, flags| {
                let result = a.zip(b, operation.0, operation.1);
                float_flags_binary(a.to_f64(), b.to_f64(), result.to_f64(), flags);
                result
            })
        }, _ => unsupported()),
        Family::Predicate(op) => dispatch_numeric!(kind, T => {
            unary_loop::<T, bool>(data, &mut offsets, output, flags, move |a: T, _| predicate(op, a.to_number()))
        }, _ => unsupported()),
        Family::Special(_) => unsupported(),
    }
}

/// Elements whose arguments a `scipy.special` loop copies out of its operands at a time.
const SPECIAL_CHUNK: usize = 1024;

/// A loop over any number of inputs of one element type, for `scipy.special` kernels, which
/// raise no floating-point flags. Arguments are copied out a chunk at a time so that kernels run
/// outside the operand borrow, where each element can be charged for the work it reports.
fn special_loop<T: Element>(
    runtime: &mut dyn PyRuntime,
    handles: &[PyArray],
    mut offsets: Vec<Offsets>,
    output: &mut [u8],
    kernel: impl Fn(&[T]) -> T,
) -> PyResult<()> {
    let nin = handles.len();
    let mut args = Vec::with_capacity(SPECIAL_CHUNK * nin);
    for chunk in output.chunks_mut(SPECIAL_CHUNK * T::SIZE) {
        let elements = chunk.len() / T::SIZE;
        args.clear();
        runtime.read_arrays(handles, &mut |arrays| {
            for _ in 0..elements {
                for (array, offsets) in arrays.iter().zip(offsets.iter_mut()) {
                    let PyArrayData::Bytes(bytes) = array.data else {
                        return Err(PyError::runtime_error("numeric loop saw objects"));
                    };
                    let offset = offsets
                        .next()
                        .ok_or_else(|| PyError::runtime_error("ufunc operand ended early"))?;
                    args.push(T::read(&bytes[offset..]));
                }
            }
            Ok(())
        })?;
        for (element, out) in args.chunks_exact(nin).zip(chunk.chunks_exact_mut(T::SIZE)) {
            evaluate_special(runtime, || kernel(element))?.write(out);
        }
    }
    Ok(())
}

/// Complex magnitude with the matching real element type.
trait ComplexMagnitude: ComplexParts {
    type Real: Element;
    fn magnitude(self) -> Self::Real;
}

impl ComplexMagnitude for super::element::C64 {
    type Real = f32;
    fn magnitude(self) -> f32 {
        self.re.hypot(self.im)
    }
}

impl ComplexMagnitude for super::element::C128 {
    type Real = f64;
    fn magnitude(self) -> f64 {
        self.re.hypot(self.im)
    }
}

fn binary_loop<T: Element, O: Element>(
    data: &[&[u8]],
    offsets: &mut [Offsets],
    output: &mut [u8],
    flags: &mut FpFlags,
    operation: impl Fn(T, T, &mut FpFlags) -> O,
) -> PyResult<()> {
    let (left, right) = (data[0], data[1]);
    let (first, rest) = offsets.split_at_mut(1);
    for ((a, b), chunk) in first[0]
        .by_ref()
        .zip(rest[0].by_ref())
        .zip(output.chunks_exact_mut(O::SIZE))
    {
        operation(T::read(&left[a..]), T::read(&right[b..]), flags).write(chunk);
    }
    Ok(())
}

fn unary_loop<T: Element, O: Element>(
    data: &[&[u8]],
    offsets: &mut [Offsets],
    output: &mut [u8],
    flags: &mut FpFlags,
    operation: impl Fn(T, &mut FpFlags) -> O,
) -> PyResult<()> {
    let input = data[0];
    for (a, chunk) in offsets[0].by_ref().zip(output.chunks_exact_mut(O::SIZE)) {
        operation(T::read(&input[a..]), flags).write(chunk);
    }
    Ok(())
}

/// The kernel for one arithmetic operation.
pub(in crate::python) fn arith_fn<T: Numeric>(op: ArithOp) -> fn(T, T, &mut FpFlags) -> T {
    match op {
        ArithOp::Add => T::add,
        ArithOp::Subtract => T::subtract,
        ArithOp::Multiply => T::multiply,
        ArithOp::FloorDivide => T::floor_divide,
        ArithOp::Remainder => T::remainder,
        ArithOp::Fmod => |a: T, b: T, flags: &mut FpFlags| {
            // C `fmod`: the result takes the dividend's sign.
            let remainder = a.remainder(b, flags);
            if remainder != T::zero()
                && (remainder.compare(T::zero()) == Some(Ordering::Less))
                    != (a.compare(T::zero()) == Some(Ordering::Less))
            {
                remainder.subtract(b, flags)
            } else {
                remainder
            }
        },
        ArithOp::Power => T::power,
        ArithOp::Maximum => |a: T, b: T, _: &mut FpFlags| a.maximum(b),
        ArithOp::Minimum => |a: T, b: T, _: &mut FpFlags| a.minimum(b),
        ArithOp::Fmax => |a: T, b: T, _: &mut FpFlags| a.fmax(b),
        ArithOp::Fmin => |a: T, b: T, _: &mut FpFlags| a.fmin(b),
    }
}

pub(in crate::python) fn compare_fn(op: CompareOp) -> fn(Option<Ordering>) -> bool {
    match op {
        CompareOp::Equal => |ordering| ordering == Some(Ordering::Equal),
        CompareOp::NotEqual => |ordering| ordering != Some(Ordering::Equal),
        CompareOp::Less => |ordering| ordering == Some(Ordering::Less),
        CompareOp::LessEqual => {
            |ordering| matches!(ordering, Some(Ordering::Less | Ordering::Equal))
        }
        CompareOp::Greater => |ordering| ordering == Some(Ordering::Greater),
        CompareOp::GreaterEqual => {
            |ordering| matches!(ordering, Some(Ordering::Greater | Ordering::Equal))
        }
    }
}

fn unary_fn<T: Numeric>(op: UnaryOp) -> fn(T, &mut FpFlags) -> T {
    match op {
        UnaryOp::Negative => T::negative,
        UnaryOp::Positive | UnaryOp::Conjugate if !T::IS_COMPLEX => |a: T, _: &mut FpFlags| a,
        UnaryOp::Positive => |a: T, _: &mut FpFlags| a,
        UnaryOp::Conjugate => |a: T, flags: &mut FpFlags| {
            // conj(z) = re - i·im, computed through the number view to stay generic.
            let _ = flags;
            match a.to_number() {
                super::element::Number::Complex(real, imag) => {
                    T::from_number(super::element::Number::Complex(real, -imag))
                }
                _ => a,
            }
        },
        UnaryOp::Absolute => T::absolute,
        UnaryOp::Square => |a: T, flags: &mut FpFlags| a.multiply(a, flags),
        UnaryOp::Reciprocal => |a: T, flags: &mut FpFlags| {
            if T::IS_INTEGER {
                // NumPy's integer reciprocal is 1 / x in integer arithmetic.
                if a == T::zero() {
                    flags.divide = true;
                    return T::zero();
                }
                T::one().floor_divide(a, flags).add(
                    if a.compare(T::zero()) == Some(Ordering::Less)
                        && a != T::zero().subtract(T::one(), flags)
                    {
                        T::one()
                    } else {
                        T::zero()
                    },
                    flags,
                )
            } else {
                a.reciprocal(flags)
            }
        },
        UnaryOp::Sign => |a: T, _: &mut FpFlags| a.sign(),
        UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Trunc if T::IS_INTEGER => {
            |a: T, _: &mut FpFlags| a
        }
        UnaryOp::Floor => |a: T, _: &mut FpFlags| {
            T::from_number(super::element::Number::Float(
                a.to_number().as_f64().floor(),
            ))
        },
        UnaryOp::Ceil => |a: T, _: &mut FpFlags| {
            T::from_number(super::element::Number::Float(a.to_number().as_f64().ceil()))
        },
        UnaryOp::Trunc => |a: T, _: &mut FpFlags| {
            T::from_number(super::element::Number::Float(
                a.to_number().as_f64().trunc(),
            ))
        },
    }
}

/// Which flag an infinite result from a finite input raises.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pole {
    /// `log(0)` and friends: a pole, reported as division by zero.
    Divide,
    /// `exp(1000)`: overflow.
    Overflow,
}

type FloatPair = (fn(f64) -> f64, fn(f32) -> f32);

fn float_fn(op: FloatOp) -> (FloatPair, Pole) {
    use FloatOp::*;
    let pair: FloatPair = match op {
        Sqrt => (f64::sqrt, f32::sqrt),
        Cbrt => (f64::cbrt, f32::cbrt),
        Exp => (f64::exp, f32::exp),
        Exp2 => (f64::exp2, f32::exp2),
        Expm1 => (f64::exp_m1, f32::exp_m1),
        Log => (f64::ln, f32::ln),
        Log2 => (f64::log2, f32::log2),
        Log10 => (f64::log10, f32::log10),
        Log1p => (f64::ln_1p, f32::ln_1p),
        Sin => (f64::sin, f32::sin),
        Cos => (f64::cos, f32::cos),
        Tan => (f64::tan, f32::tan),
        Arcsin => (f64::asin, f32::asin),
        Arccos => (f64::acos, f32::acos),
        Arctan => (f64::atan, f32::atan),
        Sinh => (f64::sinh, f32::sinh),
        Cosh => (f64::cosh, f32::cosh),
        Tanh => (f64::tanh, f32::tanh),
        Arcsinh => (f64::asinh, f32::asinh),
        Arccosh => (f64::acosh, f32::acosh),
        Arctanh => (f64::atanh, f32::atanh),
        Deg2rad => (f64::to_radians, f32::to_radians),
        Rad2deg => (f64::to_degrees, f32::to_degrees),
        Rint => (f64::round_ties_even, f32::round_ties_even),
        Fabs => (f64::abs, f32::abs),
    };
    let pole = match op {
        Log | Log2 | Log10 | Log1p | Arctanh => Pole::Divide,
        _ => Pole::Overflow,
    };
    (pair, pole)
}

fn float_unary_flags(input: f64, result: f64, pole: Pole, flags: &mut FpFlags) {
    if result.is_nan() && !input.is_nan() {
        flags.invalid = true;
    } else if result.is_infinite() && input.is_finite() {
        match pole {
            Pole::Divide => flags.divide = true,
            Pole::Overflow => flags.overflow = true,
        }
    }
}

pub(in crate::python) fn float_flags_binary(a: f64, b: f64, result: f64, flags: &mut FpFlags) {
    if result.is_nan() && !a.is_nan() && !b.is_nan() {
        flags.invalid = true;
    } else if result.is_infinite() && a.is_finite() && b.is_finite() {
        flags.overflow = true;
    }
}

pub(in crate::python) type Float2Pair = (fn(f64, f64) -> f64, fn(f32, f32) -> f32);

pub(in crate::python) fn float2_fn(op: Float2Op) -> Float2Pair {
    match op {
        Float2Op::Arctan2 => (f64::atan2, f32::atan2),
        Float2Op::Hypot => (f64::hypot, f32::hypot),
        Float2Op::Copysign => (f64::copysign, f32::copysign),
        Float2Op::Logaddexp => (
            |a, b| {
                if a == b {
                    a + std::f64::consts::LN_2
                } else {
                    let high = a.max(b);
                    high + (-(a - b).abs()).exp().ln_1p()
                }
            },
            |a, b| {
                if a == b {
                    a + std::f32::consts::LN_2
                } else {
                    let high = a.max(b);
                    high + (-(a - b).abs()).exp().ln_1p()
                }
            },
        ),
        Float2Op::Logaddexp2 => (
            |a, b| {
                if a == b {
                    a + 1.0
                } else {
                    let high = a.max(b);
                    high + (-(a - b).abs()).exp2().ln_1p() / std::f64::consts::LN_2
                }
            },
            |a, b| {
                if a == b {
                    a + 1.0
                } else {
                    let high = a.max(b);
                    high + (-(a - b).abs()).exp2().ln_1p() / std::f32::consts::LN_2
                }
            },
        ),
        Float2Op::Heaviside => (
            |x, h| {
                if x.is_nan() {
                    x
                } else if x == 0.0 {
                    h
                } else if x < 0.0 {
                    0.0
                } else {
                    1.0
                }
            },
            |x, h| {
                if x.is_nan() {
                    x
                } else if x == 0.0 {
                    h
                } else if x < 0.0 {
                    0.0
                } else {
                    1.0
                }
            },
        ),
    }
}

type ComplexFn = fn((f64, f64)) -> (f64, f64);

fn complex_fn(op: FloatOp) -> Option<ComplexFn> {
    Some(match op {
        FloatOp::Sqrt => complex_sqrt,
        FloatOp::Exp => |(re, im)| {
            let scale = re.exp();
            if im == 0.0 {
                return (scale, im);
            }
            (scale * im.cos(), scale * im.sin())
        },
        FloatOp::Log => |(re, im)| (re.hypot(im).ln(), im.atan2(re)),
        FloatOp::Log10 => |(re, im)| {
            (
                re.hypot(im).ln() / std::f64::consts::LN_10,
                im.atan2(re) / std::f64::consts::LN_10,
            )
        },
        FloatOp::Sin => |(re, im)| (re.sin() * im.cosh(), re.cos() * im.sinh()),
        FloatOp::Cos => |(re, im)| (re.cos() * im.cosh(), -(re.sin() * im.sinh())),
        _ => return None,
    })
}

/// `np.sqrt` of a complex value: the principal square root, with a nonnegative real part and
/// the branch cut along the negative real axis.
///
/// Special values (any operand `inf`, `nan`, or signed zero) follow C99 Annex G's `csqrt`,
/// confirmed black-box against the reference interpreter's `np.sqrt` across every sign and
/// `{finite, 0, inf, nan}` combination of the real and imaginary parts. The finite case uses
/// the classic half-angle formula (Abramowitz & Stegun 3.7.27: for `z = x + yi` with modulus
/// `m`, `re(sqrt z) = sqrt((m + x) / 2)` and `im(sqrt z) = y / (2 re(sqrt z))` when `x >= 0`,
/// mirrored through `im` when `x < 0`), which is exact on the real and imaginary axes and needs
/// only one `hypot` and one `sqrt`, unlike `exp(0.5 * log(z))`.
///
/// `re` and `im` are prescaled by a power of ten when either magnitude is extreme enough that
/// `hypot(re, im)` could overflow (or underflow to zero) even though the true square root is
/// representable, for example `sqrt(1.7e308 + 1.7e308i) ≈ 1.4e154 + 5.9e153i`: the identity
/// `sqrt(z) == k * sqrt(z / k^2)` recovers the true result from the rescaled inputs.
fn complex_sqrt((re, im): (f64, f64)) -> (f64, f64) {
    if im.is_infinite() {
        return (f64::INFINITY, im);
    }
    if re.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if re == f64::INFINITY {
        return if im.is_nan() {
            (f64::INFINITY, f64::NAN)
        } else {
            (f64::INFINITY, 0.0_f64.copysign(im))
        };
    }
    if re == f64::NEG_INFINITY {
        return if im.is_nan() {
            (f64::NAN, f64::INFINITY)
        } else {
            (0.0, f64::INFINITY.copysign(im))
        };
    }
    if im.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if re == 0.0 && im == 0.0 {
        return (0.0, im);
    }

    const HUGE: f64 = 1e150;
    const TINY: f64 = 1e-150;
    let largest = re.abs().max(im.abs());
    let (x, y, unscale) = if largest > HUGE {
        (re * 1e-200, im * 1e-200, 1e100)
    } else if largest < TINY {
        (re * 1e200, im * 1e200, 1e-100)
    } else {
        (re, im, 1.0)
    };

    let modulus = x.hypot(y);
    let (u, v) = if x >= 0.0 {
        let u = ((modulus + x) / 2.0).sqrt();
        let v = if u == 0.0 { 0.0 } else { y / (2.0 * u) };
        (u, v)
    } else {
        let t = ((modulus - x) / 2.0).sqrt();
        (y.abs() / (2.0 * t), t.copysign(y))
    };
    (u * unscale, v * unscale)
}

fn predicate(op: PredicateOp, value: super::element::Number) -> bool {
    use super::element::Number;
    let (real, imag) = value.as_complex();
    let is_float = matches!(value, Number::Float(_) | Number::Complex(..));
    match op {
        PredicateOp::IsNan => is_float && (real.is_nan() || imag.is_nan()),
        PredicateOp::IsInf => is_float && (real.is_infinite() || imag.is_infinite()),
        PredicateOp::IsFinite => !is_float || (real.is_finite() && imag.is_finite()),
        PredicateOp::Signbit => match value {
            Number::Int(value) => value < 0,
            Number::Float(value) => value.is_sign_negative(),
            _ => false,
        },
    }
}

/// Apply the ufunc's Python operator to every element of object arrays.
fn object_loop(
    runtime: &mut dyn PyRuntime,
    ufunc: &UfuncDef,
    resolved: &Resolved,
    inputs: &[Array],
    shape: &[usize],
) -> PyResult<(PyArrayBuffer, DType, FpFlags)> {
    let broadcast = inputs
        .iter()
        .map(|input| {
            let strides = broadcast_strides(&input.view, shape)?;
            super::array::new_view(
                runtime,
                input,
                input.dtype,
                shape.to_vec(),
                strides,
                input.view.offset,
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    let columns = broadcast
        .iter()
        .map(|input| super::array::read_objects(runtime, input))
        .collect::<PyResult<Vec<_>>>()?;
    let count = element_count(shape)?;
    reserve_elements(runtime, resolved.output, count)?;
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let operands = columns
            .iter()
            .map(|column| column[index])
            .collect::<Vec<_>>();
        values.push(object_element(runtime, ufunc, &operands)?);
    }
    if resolved.output.kind() == Kind::Bool {
        let mut bytes = Vec::with_capacity(count);
        for value in values {
            bytes.push(u8::from(runtime.truth(&value)?));
        }
        return Ok((PyArrayBuffer::Bytes(bytes), DType::BOOL, FpFlags::default()));
    }
    Ok((
        PyArrayBuffer::Values(values),
        DType::OBJECT,
        FpFlags::default(),
    ))
}

pub(in crate::python) fn object_element(
    runtime: &mut dyn PyRuntime,
    ufunc: &UfuncDef,
    operands: &[PyValue],
) -> PyResult {
    match ufunc.object {
        ObjectLoop::Missing => Err(not_supported(ufunc.name)),
        ObjectLoop::Operator(operator) => runtime.apply_operator(operator, operands),
        ObjectLoop::Square => runtime.apply_operator(
            PyOperator::Binary(BinaryOperator::Multiply),
            &[operands[0], operands[0]],
        ),
        ObjectLoop::Reciprocal => runtime.apply_operator(
            PyOperator::Binary(BinaryOperator::Divide),
            &[Value::Int(1), operands[0]],
        ),
        ObjectLoop::Max | ObjectLoop::Min => {
            let operator = if matches!(ufunc.object, ObjectLoop::Max) {
                ComparisonOperator::GreaterEqual
            } else {
                ComparisonOperator::LessEqual
            };
            let keep_first = runtime.apply_operator(PyOperator::Compare(operator), operands)?;
            Ok(if runtime.truth(&keep_first)? {
                operands[0]
            } else {
                operands[1]
            })
        }
        ObjectLoop::And => Ok(if runtime.truth(&operands[0])? {
            operands[1]
        } else {
            operands[0]
        }),
        ObjectLoop::Or => Ok(if runtime.truth(&operands[0])? {
            operands[0]
        } else {
            operands[1]
        }),
        ObjectLoop::Not => Ok(Value::Bool(!runtime.truth(&operands[0])?)),
        ObjectLoop::Sign => object_sign(runtime, operands[0]),
        ObjectLoop::Math(function) => {
            let math = runtime.import_module("math")?;
            let function = runtime
                .get_attribute(math, function)?
                .ok_or_else(|| PyError::runtime_error(format!("math.{function} is missing")))?;
            runtime.call_value(function, CallArgs::new(vec![operands[0]], Vec::new()))
        }
        ObjectLoop::Method => object_method(runtime, ufunc.name, operands),
    }
}

/// NumPy's `OBJECT_sign`: `-1`, `1` or `0` by comparing with `0`, and a `TypeError` for a value
/// such as NaN that compares false every way.
fn object_sign(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult {
    for (operator, sign) in [
        (ComparisonOperator::Less, -1),
        (ComparisonOperator::Greater, 1),
        (ComparisonOperator::Equal, 0),
    ] {
        let test =
            runtime.apply_operator(PyOperator::Compare(operator), &[value, Value::Int(0)])?;
        if runtime.truth(&test)? {
            return Ok(Value::Int(sign));
        }
    }
    Err(PyError::type_error("unorderable types for comparison"))
}

/// A `P` loop: call the first operand's method `name` with the other operands. A unary loop
/// reports a missing method as NumPy's `TypeError`; a binary one lets the `AttributeError`
/// through, as `PyObject_CallMethod` does.
fn object_method(runtime: &mut dyn PyRuntime, name: &str, operands: &[PyValue]) -> PyResult {
    let receiver = operands[0];
    let method = runtime.get_attribute(receiver, name)?;
    let method = match method {
        Some(method) if operands.len() > 1 || runtime.is_callable(&method)? => method,
        _ => {
            let type_name = runtime.type_name(&receiver)?;
            if operands.len() > 1 {
                return Err(PyError::exception(
                    "AttributeError",
                    format!("'{type_name}' object has no attribute '{name}'"),
                ));
            }
            return Err(PyError::type_error(format!(
                "loop of ufunc does not support argument 0 of type {type_name} which has no \
                 callable {name} method"
            )));
        }
    };
    runtime.call_value(method, CallArgs::new(operands[1..].to_vec(), Vec::new()))
}

/// `str` loops: concatenation, comparison, and `maximum`/`minimum` by code point order.
fn string_loop(
    runtime: &mut dyn PyRuntime,
    ufunc: &UfuncDef,
    inputs: &[Array],
    shape: &[usize],
) -> PyResult<(PyArrayBuffer, DType, FpFlags)> {
    let columns = inputs
        .iter()
        .map(|input| -> PyResult<Vec<String>> {
            let strides = broadcast_strides(&input.view, shape)?;
            let itemsize = input.itemsize();
            let mut texts = Vec::new();
            runtime.read_arrays(&[input.handle], &mut |arrays| {
                let PyArrayData::Bytes(bytes) = arrays[0].data else {
                    return Err(PyError::runtime_error("string array has object storage"));
                };
                texts.extend(
                    Offsets::new(shape, &strides, input.view.offset)
                        .map(|offset| convert::read_str(&bytes[offset..offset + itemsize])),
                );
                Ok(())
            })?;
            Ok(texts)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let count = element_count(shape)?;
    match ufunc.family {
        Family::Compare(op) => {
            let test = compare_fn(op);
            let bytes = (0..count)
                .map(|index| u8::from(test(Some(columns[0][index].cmp(&columns[1][index])))))
                .collect();
            Ok((PyArrayBuffer::Bytes(bytes), DType::BOOL, FpFlags::default()))
        }
        Family::Arith { op, .. } => {
            let texts = (0..count)
                .map(|index| {
                    let (a, b) = (&columns[0][index], &columns[1][index]);
                    match op {
                        ArithOp::Add => format!("{a}{b}"),
                        ArithOp::Maximum => a.max(b).clone(),
                        _ => a.min(b).clone(),
                    }
                })
                .collect::<Vec<_>>();
            let chars = match op {
                ArithOp::Add => inputs[0].dtype.chars() + inputs[1].dtype.chars(),
                _ => inputs[0].dtype.chars().max(inputs[1].dtype.chars()),
            };
            let dtype = DType::str(chars.max(1))?;
            reserve_elements(runtime, dtype, count)?;
            let mut bytes = Vec::with_capacity(count * dtype.itemsize());
            for text in &texts {
                convert::write_str(text, dtype.chars(), &mut bytes);
            }
            Ok((PyArrayBuffer::Bytes(bytes), dtype, FpFlags::default()))
        }
        _ => Err(no_loop(
            ufunc.name,
            &inputs.iter().map(|input| input.dtype).collect::<Vec<_>>(),
        )),
    }
}

macro_rules! operator_slots {
    ($($slot:ident, $reflected:ident => $name:literal;)*) => {
        $(
            pub(in crate::python) fn $slot(
                runtime: &mut dyn PyRuntime,
                left: PyValue,
                right: PyValue,
            ) -> PyResult<Option<PyValue>> {
                if defers_to(runtime, right)? {
                    return Ok(None);
                }
                apply(runtime, named($name), &[left, right], &Options::operator()).map(Some)
            }

            pub(in crate::python) fn $reflected(
                runtime: &mut dyn PyRuntime,
                left: PyValue,
                right: PyValue,
            ) -> PyResult<Option<PyValue>> {
                if defers_to(runtime, right)? {
                    return Ok(None);
                }
                apply(runtime, named($name), &[right, left], &Options::operator()).map(Some)
            }
        )*
    };
}

/// NEP 13's opt-out: an operand whose type sets `__array_ufunc__ = None` does not take part in
/// NumPy's operators, so an array or NumPy scalar operator returns `NotImplemented` and Python
/// tries the operand's reflected method instead. `pytest.approx` relies on this to compare
/// arrays itself. Every operator slot receives the array as `left` and the other operand as
/// `right`, reflected slots included.
pub(in crate::python) fn defers_to(runtime: &mut dyn PyRuntime, other: PyValue) -> PyResult<bool> {
    if runtime.kind(&other)? != PyKind::Instance {
        return Ok(false);
    }
    match runtime.get_attribute(other, "__array_ufunc__")? {
        Some(value) => Ok(runtime.kind(&value)? == PyKind::None),
        None => Ok(false),
    }
}

operator_slots! {
    slot_add, slot_reflected_add => "add";
    slot_subtract, slot_reflected_subtract => "subtract";
    slot_multiply, slot_reflected_multiply => "multiply";
    slot_divide, slot_reflected_divide => "divide";
    slot_floor_divide, slot_reflected_floor_divide => "floor_divide";
    slot_remainder, slot_reflected_remainder => "remainder";
    slot_bitwise_and, slot_reflected_bitwise_and => "bitwise_and";
    slot_bitwise_or, slot_reflected_bitwise_or => "bitwise_or";
    slot_bitwise_xor, slot_reflected_bitwise_xor => "bitwise_xor";
    slot_left_shift, slot_reflected_left_shift => "left_shift";
    slot_right_shift, slot_reflected_right_shift => "right_shift";
}

/// `a ** b` for an array `a`.
pub(in crate::python) fn slot_power(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    if defers_to(runtime, right)? {
        return Ok(None);
    }
    if let Some(name) = fast_power(runtime, left, right)? {
        return apply(runtime, named(name), &[left], &Options::operator()).map(Some);
    }
    apply(
        runtime,
        named("power"),
        &[left, right],
        &Options::operator(),
    )
    .map(Some)
}

pub(in crate::python) fn slot_reflected_power(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    if defers_to(runtime, right)? {
        return Ok(None);
    }
    apply(
        runtime,
        named("power"),
        &[right, left],
        &Options::operator(),
    )
    .map(Some)
}

/// The unary ufunc NumPy applies instead of `power` for `base ** exponent`, if any.
///
/// `array ** scalar` fast-paths three literal Python scalar exponents on a float or complex
/// base to a dedicated unary ufunc instead of the general binary `power`: `-1` (an exact Python
/// `int`) to `"reciprocal"`, `0.5` (an exact Python `float`) to `"sqrt"`, and `2` (an exact
/// Python `int` or `float`) to `"square"`. Eligibility is about the exponent's own Python type,
/// not its numeric value: a NumPy scalar or array holding the same value, such as
/// `np.float64(0.5)`, always takes the general `power` ufunc. This was confirmed black-box by
/// comparing the floating-point warning `np.array([-np.inf]) ** x` raises for each `x`: a bare
/// `0.5` warns "invalid value encountered in **sqrt**", while `np.float64(0.5)` warns
/// "...in **power**". Integer and boolean bases are never fast-pathed, matching NumPy's
/// restriction of the optimization to inexact loops.
fn fast_power(
    runtime: &mut dyn PyRuntime,
    base: PyValue,
    exponent: PyValue,
) -> PyResult<Option<&'static str>> {
    let name = match runtime.kind(&exponent)? {
        PyKind::Int => match runtime.int_value(&exponent) {
            Some(-1) => "reciprocal",
            Some(2) => "square",
            _ => return Ok(None),
        },
        PyKind::Float => match runtime.number(&exponent) {
            Some(NumberRef::Float(0.5)) => "sqrt",
            Some(NumberRef::Float(2.0)) => "square",
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };
    // A NumPy scalar (`np.float64(-0.0) ** 0.5`) shares this slot with arrays but is not one:
    // `**` between two scalars is "scalar math", which skips this fast path entirely (NumPy
    // calls `pow` directly, confirmed black-box by `np.float64(-0.0) ** 0.5` giving `+0.0`
    // where the array loop's constant-exponent case would keep `-0.0`), so only an actual array
    // base is eligible here.
    if runtime.native_kind(&base)? != Some(PyNativeKind::Array) {
        return Ok(None);
    }
    let base = Array::from_value(runtime, base)?;
    match base.dtype.category() {
        Category::Float | Category::Complex => Ok(Some(name)),
        _ => Ok(None),
    }
}

macro_rules! comparison_slots {
    ($($slot:ident => $name:literal;)*) => {
        $(
            pub(in crate::python) fn $slot(
                runtime: &mut dyn PyRuntime,
                left: PyValue,
                right: PyValue,
            ) -> PyResult<Option<PyValue>> {
                if defers_to(runtime, right)? {
                    return Ok(None);
                }
                apply(runtime, named($name), &[left, right], &Options::operator()).map(Some)
            }
        )*
    };
}

pub(in crate::python) fn slot_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    equality(runtime, left, right, "equal", false)
}

pub(in crate::python) fn slot_not_equal(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult<Option<PyValue>> {
    equality(runtime, left, right, "not_equal", true)
}

/// `==` and `!=` on arrays. When the operands share no comparison loop, such as a string array
/// and a number, shellsim answers "unequal" everywhere instead of raising; operands that do not
/// broadcast still raise.
fn equality(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    name: &str,
    unequal: bool,
) -> PyResult<Option<PyValue>> {
    if defers_to(runtime, right)? {
        return Ok(None);
    }
    let error = match apply(runtime, named(name), &[left, right], &Options::operator()) {
        Ok(value) => return Ok(Some(value)),
        Err(error) => error,
    };
    // `left` and `right` are already prepared operands (arrays or weak scalars) applied with
    // fixed, dtype-less options, so the only `TypeError` this call can raise is a missing
    // comparison loop or dtype-promotion failure, never an unrelated one.
    if error.kind != PyErrorKind::Type {
        return Err(error);
    }
    let (left, right) = (
        convert::as_array(runtime, left)?,
        convert::as_array(runtime, right)?,
    );
    let shape = broadcast_shapes(&[left.shape(), right.shape()])?;
    let count = element_count(&shape)?;
    reserve_elements(runtime, DType::BOOL, count)?;
    let result =
        super::array::array_from_elements(runtime, DType::BOOL, shape, &vec![unequal; count])?;
    if result.ndim() == 0 {
        return convert::element_to_scalar(runtime, &result, result.view.offset).map(Some);
    }
    Ok(Some(result.value()))
}

comparison_slots! {
    slot_less_than => "less";
    slot_less_equal => "less_equal";
    slot_greater_than => "greater";
    slot_greater_equal => "greater_equal";
}

macro_rules! unary_slots {
    ($($slot:ident => $name:literal;)*) => {
        $(
            pub(in crate::python) fn $slot(
                runtime: &mut dyn PyRuntime,
                value: PyValue,
            ) -> PyResult<Option<PyValue>> {
                apply(runtime, named($name), &[value], &Options::operator()).map(Some)
            }
        )*
    };
}

unary_slots! {
    slot_negative => "negative";
    slot_positive => "positive";
    slot_absolute => "absolute";
    slot_invert => "invert";
}

/// In-place operator on an array: `a += b` is `np.add(a, b, out=a)` with `same_kind` casting.
pub(in crate::python) fn inplace(
    runtime: &mut dyn PyRuntime,
    name: &str,
    target: PyValue,
    other: PyValue,
) -> PyResult {
    let out = Array::from_value(runtime, target)?;
    apply(
        runtime,
        named(name),
        &[target, other],
        &Options {
            out: Some(out),
            operator: true,
            ..Options::default()
        },
    )
}
