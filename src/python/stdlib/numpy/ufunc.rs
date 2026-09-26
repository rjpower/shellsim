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
//! Floating-point flags raised by a loop go to [`super::errstate::report`].

use std::cmp::Ordering;

use super::super::super::ast::{BinaryOperator, ComparisonOperator, UnaryOperator};
use super::super::super::native::{
    CallArgs, GetterDef, MethodDef, PyArrayBuffer, PyArrayData, PyError, PyKind, PyOperator,
    PyResult, PyRuntime, PyValue, ValueKindDef, ValueKindSlots,
};
use super::super::super::Value;
use super::array::{
    broadcast_shapes, broadcast_strides, element_count, new_array, reserve_elements, Array, Offsets,
};
use super::convert::{self, Leaf};
use super::dtype::{self, Casting, Category, DType, Kind, Weak};
use super::element::{
    dispatch_complex, dispatch_integer, dispatch_numeric, dispatch_real, Element,
};
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
}

/// Static description of one ufunc.
pub(in crate::python) struct UfuncDef {
    pub name: &'static str,
    pub family: Family,
    /// Python operator applied per element for `object` arrays.
    pub object: Option<PyOperator>,
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
    object: Option<PyOperator>,
) -> UfuncDef {
    UfuncDef {
        name,
        family: Family::Arith { op, bools, complex },
        object,
    }
}

const fn binary_operator(operator: BinaryOperator) -> Option<PyOperator> {
    Some(PyOperator::Binary(operator))
}

const fn compare(name: &'static str, op: CompareOp, operator: ComparisonOperator) -> UfuncDef {
    UfuncDef {
        name,
        family: Family::Compare(op),
        object: Some(PyOperator::Compare(operator)),
    }
}

const fn float(name: &'static str, op: FloatOp, complex: bool) -> UfuncDef {
    UfuncDef {
        name,
        family: Family::Float { op, complex },
        object: None,
    }
}

const fn unary_def(
    name: &'static str,
    op: UnaryOp,
    bools: BoolLoop,
    complex: bool,
    object: Option<PyOperator>,
) -> UfuncDef {
    UfuncDef {
        name,
        family: Family::Unary { op, bools, complex },
        object,
    }
}

const fn simple(name: &'static str, family: Family, object: Option<PyOperator>) -> UfuncDef {
    UfuncDef {
        name,
        family,
        object,
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
    arith("fmod", ArithOp::Fmod, BoolLoop::Int8, false, None),
    arith(
        "power",
        ArithOp::Power,
        BoolLoop::Int8,
        true,
        binary_operator(BinaryOperator::Power),
    ),
    arith("maximum", ArithOp::Maximum, BoolLoop::Keep, true, None),
    arith("minimum", ArithOp::Minimum, BoolLoop::Keep, true, None),
    arith("fmax", ArithOp::Fmax, BoolLoop::Keep, true, None),
    arith("fmin", ArithOp::Fmin, BoolLoop::Keep, true, None),
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
    simple("logical_and", Family::Logical(LogicalOp::And), None),
    simple("logical_or", Family::Logical(LogicalOp::Or), None),
    simple("logical_xor", Family::Logical(LogicalOp::Xor), None),
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
    simple("arctan2", Family::Float2(Float2Op::Arctan2), None),
    simple("hypot", Family::Float2(Float2Op::Hypot), None),
    simple("copysign", Family::Float2(Float2Op::Copysign), None),
    simple("logaddexp", Family::Float2(Float2Op::Logaddexp), None),
    simple("logaddexp2", Family::Float2(Float2Op::Logaddexp2), None),
    simple("heaviside", Family::Float2(Float2Op::Heaviside), None),
    unary_def(
        "negative",
        UnaryOp::Negative,
        BoolLoop::Reject(BOOL_NEGATIVE),
        true,
        Some(PyOperator::Unary(UnaryOperator::Negative)),
    ),
    unary_def(
        "positive",
        UnaryOp::Positive,
        BoolLoop::Missing,
        true,
        Some(PyOperator::Unary(UnaryOperator::Positive)),
    ),
    unary_def("absolute", UnaryOp::Absolute, BoolLoop::Keep, true, None),
    unary_def("square", UnaryOp::Square, BoolLoop::Int8, true, None),
    unary_def(
        "reciprocal",
        UnaryOp::Reciprocal,
        BoolLoop::Int8,
        true,
        None,
    ),
    unary_def("sign", UnaryOp::Sign, BoolLoop::Missing, true, None),
    unary_def("conjugate", UnaryOp::Conjugate, BoolLoop::Int8, true, None),
    unary_def("floor", UnaryOp::Floor, BoolLoop::Keep, false, None),
    unary_def("ceil", UnaryOp::Ceil, BoolLoop::Keep, false, None),
    unary_def("trunc", UnaryOp::Trunc, BoolLoop::Keep, false, None),
    simple(
        "invert",
        Family::Invert,
        Some(PyOperator::Unary(UnaryOperator::Invert)),
    ),
    simple("logical_not", Family::LogicalNot, None),
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
    simple("isnan", Family::Predicate(PredicateOp::IsNan), None),
    simple("isinf", Family::Predicate(PredicateOp::IsInf), None),
    simple("isfinite", Family::Predicate(PredicateOp::IsFinite), None),
    simple("signbit", Family::Predicate(PredicateOp::Signbit), None),
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

fn named(name: &str) -> usize {
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
        return Err(PyError::value_error(format!(
            "outer product only supported for binary functions"
        )))
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
#[derive(Default)]
pub(in crate::python) struct Options {
    pub out: Option<Array>,
    pub dtype: Option<DType>,
    pub casting: Option<Casting>,
    /// The call comes from a Python operator such as `a + b`. When every operand is a scalar,
    /// NumPy's scalar math also reports integer overflow and names the operation `scalar add`.
    pub operator: bool,
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
    let mut options = Options::default();
    let mut out = positional.get(nin).copied();
    for (name, value) in args.keywords() {
        match name.as_str() {
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
            "where" => {
                return Err(PyError::unsupported(format!(
                    "{}() with a where= mask is not supported",
                    ufunc.name
                )))
            }
            "subok" | "order" => {}
            _ => {
                return Err(PyError::type_error(format!(
                    "{}() got an unexpected keyword argument '{name}'",
                    ufunc.name
                )))
            }
        }
    }
    options.out = out_array(runtime, out)?;
    apply(runtime, index, &positional[..nin], &options)
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
enum Operand {
    Array(Array),
    Weak {
        value: PyValue,
        leaf: Leaf,
        weak: Weak,
    },
}

impl Operand {
    fn dtype(&self) -> Option<DType> {
        match self {
            Self::Array(array) => Some(array.dtype),
            Self::Weak { .. } => None,
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::Array(array) => array.shape(),
            Self::Weak { .. } => &[],
        }
    }
}

/// Prepare one input, reporting whether it is a scalar (a Python number or a NumPy scalar).
fn operand(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<(Operand, bool)> {
    if let Some((weak, leaf)) = convert::weak_scalar(runtime, &value)? {
        return Ok((Operand::Weak { value, leaf, weak }, true));
    }
    let scalar = super::scalar::unbox(runtime, &value).is_some();
    let array = convert::as_array(runtime, value)?;
    Ok((Operand::Array(array), scalar))
}

/// The dtype operands promote to under NEP 50.
fn common_dtype(operands: &[Operand]) -> PyResult<DType> {
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

/// A missing-loop error with NumPy's text.
fn no_loop(name: &str, inputs: &[DType]) -> PyError {
    let types = inputs
        .iter()
        .map(|dtype| format!("dtype('{}')", dtype.display()))
        .collect::<Vec<_>>()
        .join(", ");
    let rendered = if inputs.len() == 1 {
        format!("<class 'numpy.dtypes.{}DType'>", class_name(inputs[0]))
    } else {
        format!("({types})")
    };
    PyError::exception(
        "UFuncTypeError",
        format!(
        "ufunc '{name}' did not contain a loop with signature matching types {rendered} -> None"
    ),
    )
}

fn class_name(dtype: DType) -> String {
    match dtype.kind() {
        Kind::Bool => "Bool".to_string(),
        Kind::Str => "Str".to_string(),
        Kind::Object => "Object".to_string(),
        kind => {
            let name = kind.name();
            format!("{}{}", name[..1].to_uppercase(), &name[1..])
        }
    }
}

fn not_supported(name: &str) -> PyError {
    PyError::exception(
        "UFuncTypeError",
        format!(
            "ufunc '{name}' not supported for the input types, and the inputs could not be safely \
         coerced to any supported types according to the casting rule ''safe''"
        ),
    )
}

/// Loop and output dtypes chosen for one call.
struct Resolved {
    /// Dtype every input is cast to; `None` means each input keeps its own dtype.
    input: DType,
    output: DType,
}

fn resolve(
    ufunc: &UfuncDef,
    common: DType,
    inputs: &[DType],
    requested: Option<DType>,
) -> PyResult<Resolved> {
    let name = ufunc.name;
    let same = |dtype: DType| Resolved {
        input: dtype,
        output: dtype,
    };
    let object = common.kind() == Kind::Object;
    if object {
        if ufunc.object.is_none()
            && !matches!(ufunc.family, Family::Logical(_) | Family::LogicalNot)
        {
            return Err(PyError::type_error(format!(
                "loop of ufunc does not support argument 0 of type object which has no callable \
                 {name} method"
            )));
        }
        let output = match ufunc.family {
            Family::Compare(_) | Family::Logical(_) | Family::LogicalNot => DType::BOOL,
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
    }
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

/// Run ufunc `index` on `inputs`.
pub(in crate::python) fn apply(
    runtime: &mut dyn PyRuntime,
    index: usize,
    inputs: &[PyValue],
    options: &Options,
) -> PyResult {
    let ufunc = &UFUNCS[index];
    let mut operands = Vec::with_capacity(inputs.len());
    let mut all_scalars = true;
    for value in inputs {
        let (operand, scalar) = operand(runtime, *value)?;
        all_scalars &= scalar;
        operands.push(operand);
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
                return Err(PyError::exception(
                    "UFuncTypeError",
                    format!(
                    "Cannot cast ufunc '{}' input {position} from {} to {} with casting rule '{}'",
                    ufunc.name,
                    dtype.repr(),
                    resolved.input.repr(),
                    casting.name()
                ),
                ));
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
        if !dtype::can_cast(resolved.output, out.dtype, casting) {
            return Err(PyError::exception(
                "UFuncTypeError",
                format!(
                    "Cannot cast ufunc '{}' output from {} to {} with casting rule '{}'",
                    ufunc.name,
                    resolved.output.repr(),
                    out.dtype.repr(),
                    casting.name()
                ),
            ));
        }
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
    let (buffer, output_dtype, flags) = run(runtime, ufunc, &resolved, &prepared, &shape)?;
    if flags.any() {
        let scalar_math = options.operator && all_scalars;
        let mut flags = flags;
        if resolved.input.is_integer() && !scalar_math {
            // Integer loops wrap silently; only scalar operators report overflow.
            flags.overflow = false;
        }
        let name = if scalar_math {
            format!("scalar {}", ufunc.name)
        } else {
            ufunc.name.to_string()
        };
        super::errstate::report(runtime, &name, flags)?;
    }
    let result = new_array(runtime, buffer, output_dtype, shape)?;
    if let Some(out) = &options.out {
        super::array::assign(runtime, out, &result)?;
        return Ok(out.value());
    }
    if result.ndim() == 0 {
        return convert::element_to_scalar(runtime, &result, result.view.offset);
    }
    Ok(result.value())
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

fn prepare(runtime: &mut dyn PyRuntime, operand: &Operand, dtype: DType) -> PyResult<Array> {
    match operand {
        Operand::Array(array) if dtype.kind() == Kind::Str && array.dtype.kind() == Kind::Str => {
            Ok(array.clone())
        }
        Operand::Array(array) => convert::cast_array(runtime, array, dtype, false),
        Operand::Weak { value, leaf, .. } => convert::weak_array(runtime, *value, leaf, dtype),
    }
}

/// Relative cost of one element of a ufunc loop, in CPU units.
fn element_cost(family: Family) -> u64 {
    match family {
        Family::Float { .. } | Family::Float2(_) => 4,
        Family::Arith {
            op: ArithOp::Power, ..
        } => 4,
        _ => 1,
    }
}

/// Execute the loop into a fresh C-contiguous buffer.
fn run(
    runtime: &mut dyn PyRuntime,
    ufunc: &UfuncDef,
    resolved: &Resolved,
    inputs: &[Array],
    shape: &[usize],
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
        numeric_loop(family, kind, &data, offsets, &mut bytes, &mut flags)
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
) -> PyResult<()> {
    let unsupported = || {
        Err(PyError::runtime_error(
            "ufunc loop has no kernel for its dtype",
        ))
    };
    match family {
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
    }
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

fn compare_fn(op: CompareOp) -> fn(Option<Ordering>) -> bool {
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
                T::one().divide(a, flags)
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

fn float_flags_binary(a: f64, b: f64, result: f64, flags: &mut FpFlags) {
    if result.is_nan() && !a.is_nan() && !b.is_nan() {
        flags.invalid = true;
    } else if result.is_infinite() && a.is_finite() && b.is_finite() {
        flags.overflow = true;
    }
}

type Float2Pair = (fn(f64, f64) -> f64, fn(f32, f32) -> f32);

fn float2_fn(op: Float2Op) -> Float2Pair {
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
        FloatOp::Sqrt => |(re, im)| {
            if re == 0.0 && im == 0.0 {
                return (0.0, im);
            }
            let magnitude = re.hypot(im);
            let real = ((magnitude + re) / 2.0).sqrt();
            let imag = ((magnitude - re) / 2.0).sqrt().copysign(im);
            (real, imag)
        },
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

fn object_element(runtime: &mut dyn PyRuntime, ufunc: &UfuncDef, operands: &[PyValue]) -> PyResult {
    match ufunc.family {
        Family::Logical(op) => {
            let (a, b) = (runtime.truth(&operands[0])?, runtime.truth(&operands[1])?);
            Ok(Value::Bool(match op {
                LogicalOp::And => a && b,
                LogicalOp::Or => a || b,
                LogicalOp::Xor => a != b,
            }))
        }
        Family::LogicalNot => Ok(Value::Bool(!runtime.truth(&operands[0])?)),
        _ => {
            let operator = ufunc.object.expect("resolve checks object support");
            runtime.apply_operator(operator, operands)
        }
    }
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
                apply(runtime, named($name), &[left, right], &Options::operator()).map(Some)
            }

            pub(in crate::python) fn $reflected(
                runtime: &mut dyn PyRuntime,
                left: PyValue,
                right: PyValue,
            ) -> PyResult<Option<PyValue>> {
                apply(runtime, named($name), &[right, left], &Options::operator()).map(Some)
            }
        )*
    };
}

operator_slots! {
    slot_add, slot_reflected_add => "add";
    slot_subtract, slot_reflected_subtract => "subtract";
    slot_multiply, slot_reflected_multiply => "multiply";
    slot_divide, slot_reflected_divide => "divide";
    slot_floor_divide, slot_reflected_floor_divide => "floor_divide";
    slot_remainder, slot_reflected_remainder => "remainder";
    slot_power, slot_reflected_power => "power";
    slot_bitwise_and, slot_reflected_bitwise_and => "bitwise_and";
    slot_bitwise_or, slot_reflected_bitwise_or => "bitwise_or";
    slot_bitwise_xor, slot_reflected_bitwise_xor => "bitwise_xor";
    slot_left_shift, slot_reflected_left_shift => "left_shift";
    slot_right_shift, slot_reflected_right_shift => "right_shift";
}

macro_rules! comparison_slots {
    ($($slot:ident => $name:literal;)*) => {
        $(
            pub(in crate::python) fn $slot(
                runtime: &mut dyn PyRuntime,
                left: PyValue,
                right: PyValue,
            ) -> PyResult<Option<PyValue>> {
                apply(runtime, named($name), &[left, right], &Options::operator()).map(Some)
            }
        )*
    };
}

comparison_slots! {
    slot_equal => "equal";
    slot_not_equal => "not_equal";
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
