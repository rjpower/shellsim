//! Reductions: `ufunc.reduce`/`ufunc.accumulate`, the `numpy` reduction functions built on them
//! (`sum`, `prod`, `max`/`amax`, `min`/`amin`, `any`, `all`, `argmin`, `argmax`, `cumsum`,
//! `cumprod`), their `ndarray` methods, and the `python_method` trampoline other native areas use
//! to reach Python-level helpers by name (`mean`/`var`/`std` here through `numpy._stats`;
//! `squeeze`/`swapaxes`/`trace` in `shape.rs` through `numpy._shapes`; `clip` in `math.rs`
//! through `numpy._math`).
//!
//! # Walk order
//!
//! Every reduction splits an array's axes into the ones kept in the output and the ones being
//! combined, and reads the whole array once through [`layout::reading_order`] with the kept
//! axes outermost and the reduced axes innermost, in each axis's own logical order regardless of
//! strides. That lays out, for every output cell, a contiguous run of its inputs in logical
//! order. Each run is then combined with [`reduce_numeric`], [`reduce_integer`], [`reduce_bool`],
//! or [`reduce_object`], the same building block generic `ufunc.reduce` uses for every dtype.
//!
//! # Floating-point accuracy
//!
//! Floating-point and complex `add` reductions combine each run with a plain pairwise sum
//! ([`ops::pairwise_sum`]): recursive halving down to a small block, which keeps rounding error
//! growing like `log n` rather than `n`. The result does not depend on the array's strides, only
//! on the logical order of the values being summed. Every other reduction combines its run
//! sequentially in the order read, since only `add` is associative enough to reorder safely.
//!
//! # Supported ufunc families
//!
//! `ufunc.reduce`/`accumulate` support every binary ufunc NumPy itself supports them for:
//! `add`, `subtract`, `multiply`, `floor_divide`, `remainder`, `fmod`, `power`, `maximum`,
//! `minimum`, `fmax`, `fmin`, `divide`, the bitwise ufuncs, the logical ufuncs (which always
//! compute on a boolean cast of the input, as NumPy's own type resolution does), the comparison
//! ufuncs (only on boolean input, since that is the only dtype NumPy registers a comparison
//! reduce loop for), and the two-argument real functions `arctan2`, `hypot`, `copysign`,
//! `logaddexp`, `logaddexp2`, and `heaviside`. `power` rechecks each element used as an exponent
//! for a negative value on signed integer input, the same check ordinary `**` makes before its
//! loop runs, since a per-element reduction step cannot make that check once up front the way
//! whole-array application does. `scipy.special` ufuncs are rejected outright with their own
//! error text naming SciPy: NumPy allows some of these (`scipy.special.xlogy.reduce` runs a real
//! reduce loop upstream), but no caller in this codebase needs them, so this is an explicit
//! unsupported frontier rather than a guess at behavior nothing exercises. `nextafter` and
//! `ldexp` are not registered as ufuncs anywhere in shellsim's NumPy at all (`np.nextafter` and
//! `np.ldexp` do not exist), so there is no elementwise loop for `.reduce()` to call in the
//! first place; adding them is a math-kernel gap, not a reduction gap.

use std::cmp::Ordering;

use super::super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyArrayBuffer, PyError, PyResult,
    PyRuntime, PyValue,
};
use super::super::super::Value;
use super::args::{self, Axes, Bound, Signature};
use super::array::{self, Array};
use super::convert;
use super::dtype::{self, Category, DType, Kind};
use super::element::{dispatch_integer, dispatch_numeric, dispatch_real, Element, Number};
use super::layout;
use super::ops::{FpFlags, Integer, Numeric, Real};
use super::ufunc::{
    self, ArithOp, BitOp, CompareOp, Family, Float2Op, LogicalOp, UfuncDef, UFUNCS,
};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_reduce",
    functions: FUNCTIONS,
    values: &[],
};

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

static FUNCTIONS: &[FunctionDef] = &[
    function("sum", module_sum),
    function("prod", module_prod),
    function("amax", module_max),
    function("max", module_max),
    function("amin", module_min),
    function("min", module_min),
    function("any", module_any),
    function("all", module_all),
    function("argmin", module_argmin),
    function("argmax", module_argmax),
    function("cumsum", module_cumsum),
    function("cumprod", module_cumprod),
];

pub(in crate::python) static ARRAY_METHODS: NativeTypeDef = NativeTypeDef {
    name: "numpy.ndarray",
    methods: &[
        method("sum", method_sum),
        method("prod", method_prod),
        method("max", method_max),
        method("min", method_min),
        method("any", method_any),
        method("all", method_all),
        method("argmin", method_argmin),
        method("argmax", method_argmax),
        method("cumsum", method_cumsum),
        method("cumprod", method_cumprod),
        method("mean", method_mean),
        method("var", method_var),
        method("std", method_std),
    ],
    getters: &[],
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

/// Call a method implemented in Python at `module.<name>`, with `receiver` as the first
/// positional argument ahead of `args`. `mean`/`var`/`std` reach `numpy._stats._mean` etc. this
/// way; `shape.rs` and `math.rs` use the same trampoline to reach `numpy._shapes` and
/// `numpy._math` for `squeeze`/`swapaxes`/`trace` and `clip`.
pub(in crate::python) fn python_method(
    runtime: &mut dyn PyRuntime,
    module: &str,
    name: &str,
    receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    let module = runtime.import_module(module)?;
    let implementation = runtime
        .get_attribute(module, name)?
        .ok_or_else(|| PyError::runtime_error(format!("{name} is missing")))?;
    let (positional, keywords) = args.into_parts();
    let mut all_positional = Vec::with_capacity(positional.len() + 1);
    all_positional.push(receiver);
    all_positional.extend(positional);
    runtime.call_value(implementation, CallArgs::new(all_positional, keywords))
}

fn method_mean(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    python_method(runtime, "numpy._stats", "_mean", receiver, args)
}

fn method_var(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    python_method(runtime, "numpy._stats", "_var", receiver, args)
}

fn method_std(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    python_method(runtime, "numpy._stats", "_std", receiver, args)
}

/// The registered ufunc called `name`, for reductions that are fixed to one ufunc such as `sum`
/// and `trace`.
pub(in crate::python) fn ufunc_named(name: &str) -> &'static UfuncDef {
    &UFUNCS[ufunc::named(name)]
}

// ---------------------------------------------------------------------------------------------
// `ufunc.reduce` and `ufunc.accumulate`
// ---------------------------------------------------------------------------------------------

/// `ufunc.reduce(array, axis=0, dtype=None, out=None, keepdims=False, initial=<no value>,
/// where=True)`. `axis` defaults to `0` in NumPy's signature but every caller in this codebase
/// passes it explicitly or relies on `None` meaning every axis, which is also NumPy's behavior
/// for `.reduce()` specifically (unlike `.accumulate()`, whose default axis really is `0`).
pub(in crate::python) fn ufunc_reduce(
    runtime: &mut dyn PyRuntime,
    index: usize,
    args: CallArgs,
) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "reduce",
        &[
            "array", "axis", "dtype", "out", "keepdims", "initial", "where",
        ],
        1,
    );
    let ufunc = &UFUNCS[index];
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("array"))?;
    // Unlike `np.sum`, `ufunc.reduce` reduces only the first axis unless told otherwise; an
    // explicit `axis=None` reduces every axis.
    let axes = match bound.get("axis") {
        None if array.ndim() > 0 => Axes::Some(vec![0]),
        axis => args::axes(runtime, axis, array.ndim())?,
    };
    let dtype = args::optional_dtype(runtime, bound.get("dtype"))?;
    let out = out_array(runtime, bound.get("out"))?;
    let keepdims = args::flag(runtime, bound.get("keepdims"), false)?;
    let initial = bound.value("initial");
    let where_ = bound.value("where");
    reduce_call(
        runtime, ufunc, array, axes, dtype, out, keepdims, initial, where_,
    )
}

/// `ufunc.accumulate(array, axis=0, dtype=None, out=None)`: no `initial`, `where`, or
/// `keepdims`, and `axis` is a single axis (default `0`), never `None` or a tuple.
pub(in crate::python) fn ufunc_accumulate(
    runtime: &mut dyn PyRuntime,
    index: usize,
    args: CallArgs,
) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("accumulate", &["array", "axis", "dtype", "out"], 1);
    let ufunc = &UFUNCS[index];
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("array"))?;
    let axis = match bound.get("axis") {
        None => 0,
        Some(value) if value.is_none() => {
            return Err(PyError::value_error(
                "accumulate does not allow multiple axes",
            ))
        }
        Some(value) => array::normalize_axis(args::index_int(runtime, &value)?, array.ndim())?,
    };
    let dtype = args::optional_dtype(runtime, bound.get("dtype"))?;
    let out = out_array(runtime, bound.get("out"))?;
    accumulate_call(runtime, ufunc, array, axis, dtype, out)
}

/// An `out=` argument, parsed the way every reduction function accepts it: an array, or omitted
/// (`None` also counts as omitted, as it does throughout NumPy's reduction keywords).
fn out_array(runtime: &dyn PyRuntime, value: Option<PyValue>) -> PyResult<Option<Array>> {
    value
        .filter(|value| !value.is_none())
        .map(|value| Array::from_value(runtime, value))
        .transpose()
}

// ---------------------------------------------------------------------------------------------
// The reduction engine
// ---------------------------------------------------------------------------------------------

/// A binary operator computed through [`ops::Numeric`], the arithmetic family every element
/// type in this module already implements (see `ops.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumericOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    FloorDivide,
    Remainder,
    Fmod,
    Power,
    Maximum,
    Minimum,
    Fmax,
    Fmin,
}

/// A binary operator computed through [`ops::Integer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IntegerOp {
    And,
    Or,
    Xor,
    LeftShift,
    RightShift,
}

/// A binary operator on the boolean cast of the input, as the logical ufuncs always compute, or
/// one of the comparison ufuncs, which NumPy only registers a reduce/accumulate loop for on
/// boolean input (see [`combine_compare`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoolOp {
    And,
    Or,
    Xor,
    Compare(CompareOp),
}

/// The family a supported ufunc reduces or accumulates through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReduceOp {
    Numeric(NumericOp),
    Integer(IntegerOp),
    Bool(BoolOp),
    /// `arctan2`, `hypot`, `copysign`, `logaddexp`, `logaddexp2`, `heaviside`: real-only binary
    /// functions computed through [`ops::Real::zip`] with the same function pair
    /// ([`ufunc::float2_fn`]) the elementwise loop uses, so a two-element reduction matches
    /// elementwise application exactly.
    Float2(Float2Op),
}

/// Which [`ReduceOp`] a ufunc's [`Family`] reduces through, or the unsupported-frontier error
/// documented at the top of this module.
fn classify(ufunc: &UfuncDef) -> PyResult<ReduceOp> {
    match ufunc.family {
        Family::Arith { op, .. } => match op {
            ArithOp::Add => Ok(ReduceOp::Numeric(NumericOp::Add)),
            ArithOp::Subtract => Ok(ReduceOp::Numeric(NumericOp::Subtract)),
            ArithOp::Multiply => Ok(ReduceOp::Numeric(NumericOp::Multiply)),
            ArithOp::FloorDivide => Ok(ReduceOp::Numeric(NumericOp::FloorDivide)),
            ArithOp::Remainder => Ok(ReduceOp::Numeric(NumericOp::Remainder)),
            ArithOp::Fmod => Ok(ReduceOp::Numeric(NumericOp::Fmod)),
            ArithOp::Power => Ok(ReduceOp::Numeric(NumericOp::Power)),
            ArithOp::Maximum => Ok(ReduceOp::Numeric(NumericOp::Maximum)),
            ArithOp::Minimum => Ok(ReduceOp::Numeric(NumericOp::Minimum)),
            ArithOp::Fmax => Ok(ReduceOp::Numeric(NumericOp::Fmax)),
            ArithOp::Fmin => Ok(ReduceOp::Numeric(NumericOp::Fmin)),
        },
        Family::TrueDivide => Ok(ReduceOp::Numeric(NumericOp::Divide)),
        Family::Bitwise(op) => Ok(ReduceOp::Integer(match op {
            BitOp::And => IntegerOp::And,
            BitOp::Or => IntegerOp::Or,
            BitOp::Xor => IntegerOp::Xor,
            BitOp::LeftShift => IntegerOp::LeftShift,
            BitOp::RightShift => IntegerOp::RightShift,
        })),
        Family::Logical(op) => Ok(ReduceOp::Bool(match op {
            LogicalOp::And => BoolOp::And,
            LogicalOp::Or => BoolOp::Or,
            LogicalOp::Xor => BoolOp::Xor,
        })),
        Family::Compare(op) => Ok(ReduceOp::Bool(BoolOp::Compare(op))),
        Family::Float2(op) => Ok(ReduceOp::Float2(op)),
        _ if ufunc.nin() != 2 => Err(PyError::value_error(
            "reduce only supported for binary functions",
        )),
        // `scipy.special` ufuncs get their own message: they are a SciPy limitation, not a
        // NumPy one, even though a `scipy.special` ufunc reaches this classifier the same way
        // an unsupported NumPy ufunc does.
        Family::Special(_) => Err(unsupported_scipy_reduce(ufunc.name)),
        _ => Err(unsupported_reduce(ufunc.name)),
    }
}

fn unsupported_reduce(name: &str) -> PyError {
    PyError::not_implemented_error(format!(
        "reduce/accumulate for ufunc '{name}' is not supported by shellsim's NumPy"
    ))
}

fn unsupported_scipy_reduce(name: &str) -> PyError {
    PyError::not_implemented_error(format!(
        "reduce and accumulate of scipy.special.{name} are not supported by shellsim's SciPy"
    ))
}

fn no_identity_error(name: &str) -> PyError {
    PyError::value_error(format!(
        "zero-size array to reduction operation {name} which has no identity"
    ))
}

/// NumPy's message when a ufunc has no reduce/accumulate loop at all for a resolved dtype: a
/// comparison ufunc asked to reduce anything but a boolean array, or a real two-argument
/// function (`arctan2` and friends) asked to reduce anything but a float array.
fn no_matching_loop_error(name: &str) -> PyError {
    PyError::type_error(format!(
        "No loop matching the specified signature and casting was found for ufunc {name}"
    ))
}

/// NumPy's message when a ufunc's loop exists but categorically excludes a dtype (complex input
/// to `floor_divide`, `remainder`, or `fmod`), as opposed to [`no_matching_loop_error`], which
/// covers a dtype request the resolver never considers a candidate loop for.
fn no_complex_loop_error(name: &str) -> PyError {
    PyError::type_error(format!(
        "ufunc '{name}' not supported for the input types, and the inputs could not be safely \
         coerced to any supported types according to the casting rule ''safe''"
    ))
}

/// NumPy's message for `subtract`/`-` on boolean input.
fn boolean_subtract_error() -> PyError {
    PyError::type_error(
        "numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the \
         `^` operator, or the logical_xor function instead."
            .to_string(),
    )
}

fn where_needs_initial_error(name: &str) -> PyError {
    PyError::value_error(format!(
        "reduction operation '{name}' does not have an identity, so to use a where mask one has \
         to specify 'initial'"
    ))
}

/// The dtype every element is cast to before reducing, and whether the operator has an
/// identity: `Add`/`Multiply` promote bool and small integers to the platform integer of their
/// signedness the way NumPy's own `add`/`multiply` type resolution does for `.reduce()` and
/// `.accumulate()` (not for elementwise application); `divide` widens to float; the logical
/// ufuncs always compute on a boolean cast, ignoring any requested dtype, as NumPy's resolver
/// does; every other operator keeps the array's own dtype unless `dtype=` overrides it.
fn resolve_dtype(
    ufunc: &UfuncDef,
    op: ReduceOp,
    array_dtype: DType,
    requested: Option<DType>,
) -> PyResult<DType> {
    let dtype = match op {
        // The logical ufuncs always compute on a boolean cast of the input, whatever `dtype=`
        // asks for. A comparison ufunc instead only has a loop for boolean input at all: unlike
        // every other case here, an explicit `dtype=` that is not itself bool never widens or
        // narrows anything, it just changes which error fires (see `no_matching_loop_error`).
        ReduceOp::Bool(BoolOp::And | BoolOp::Or | BoolOp::Xor) => DType::BOOL,
        ReduceOp::Bool(BoolOp::Compare(_)) => {
            let resolved = requested.unwrap_or(array_dtype);
            if resolved.kind() != Kind::Bool {
                return Err(no_matching_loop_error(ufunc.name));
            }
            resolved
        }
        ReduceOp::Numeric(NumericOp::Add | NumericOp::Multiply) => {
            requested.unwrap_or_else(|| dtype::accumulator(array_dtype))
        }
        ReduceOp::Numeric(NumericOp::Divide) => {
            requested.unwrap_or_else(|| dtype::true_divide_dtype(array_dtype))
        }
        ReduceOp::Numeric(NumericOp::Subtract) => {
            let resolved = requested.unwrap_or(array_dtype);
            if resolved.kind() == Kind::Bool {
                return Err(boolean_subtract_error());
            }
            resolved
        }
        // `floor_divide`, `remainder`, and `fmod` promote a *default* boolean array (nothing
        // requested through `dtype=`) the way NumPy's own type resolution does for every
        // non-comparison arithmetic ufunc without a bool loop: to `int8`, the smallest signed
        // type that holds `0`/`1`. An explicit `dtype=bool` is not promoted the same way; NumPy
        // has no bool loop for these at all, so asking for one directly has no candidate loop,
        // the same error as any other dtype none of these ufuncs support. None of the three has
        // a complex loop either; `complex_divide`, which `floor_divide`/`remainder` would
        // otherwise reach, is still an unfinished clean-room stub in `ops.rs` (kernels' unit),
        // so this rejects complex up front with NumPy's own message instead of reaching that
        // panic.
        ReduceOp::Numeric(NumericOp::FloorDivide | NumericOp::Remainder | NumericOp::Fmod) => {
            let resolved = requested.unwrap_or(array_dtype);
            if resolved.category() == Category::Complex {
                return Err(no_complex_loop_error(ufunc.name));
            }
            match (requested, resolved.kind()) {
                (Some(_), Kind::Bool) => return Err(no_matching_loop_error(ufunc.name)),
                (None, Kind::Bool) => DType::INT8,
                _ => resolved,
            }
        }
        // `power` promotes a default boolean array the same way, but (unlike the three above)
        // does have a complex loop, so it is not rejected here.
        ReduceOp::Numeric(NumericOp::Power) => {
            let resolved = requested.unwrap_or(array_dtype);
            match (requested, resolved.kind()) {
                (Some(_), Kind::Bool) => return Err(no_matching_loop_error(ufunc.name)),
                (None, Kind::Bool) => DType::INT8,
                _ => resolved,
            }
        }
        ReduceOp::Numeric(_) => requested.unwrap_or(array_dtype),
        ReduceOp::Integer(int_op) => {
            let base = requested.unwrap_or(array_dtype);
            match (base.category(), int_op) {
                (Category::Bool, IntegerOp::LeftShift | IntegerOp::RightShift) => DType::INT8,
                (Category::Bool | Category::Signed | Category::Unsigned, _) => base,
                _ => {
                    return Err(PyError::type_error(format!(
                        "ufunc '{}' not supported for the input types",
                        ufunc.name
                    )))
                }
            }
        }
        // Int/bool promote to the smallest float that holds every value (`smallest_float_for`),
        // exactly as the elementwise loop resolves its working type before `dispatch_real!`
        // picks it; an explicit `dtype=` is used as given, with no further promotion. Complex
        // gets its own message the same way `floor_divide`/`remainder`/`fmod` do above (these
        // functions are real-only, categorically, not just missing a loop for one dtype); any
        // other non-float resolved dtype (bool, any integer width, str, object) has no
        // candidate loop at all.
        ReduceOp::Float2(_) => {
            let resolved = requested.unwrap_or_else(|| array_dtype.smallest_float_for());
            if resolved.category() == Category::Complex {
                return Err(no_complex_loop_error(ufunc.name));
            }
            if !matches!(
                resolved.kind(),
                Kind::Float16 | Kind::Float32 | Kind::Float64
            ) {
                return Err(no_matching_loop_error(ufunc.name));
            }
            resolved
        }
    };
    let numeric_object_ok = matches!(
        op,
        ReduceOp::Numeric(
            NumericOp::Add
                | NumericOp::Subtract
                | NumericOp::Multiply
                | NumericOp::Power
                | NumericOp::Fmod
                | NumericOp::Maximum
                | NumericOp::Minimum
                | NumericOp::Fmax
                | NumericOp::Fmin
        )
    );
    match dtype.kind() {
        Kind::Object if numeric_object_ok => Ok(dtype),
        Kind::Object | Kind::Str => Err(PyError::not_implemented_error(format!(
            "{} reduction of '{}' arrays is not supported by shellsim's NumPy",
            ufunc.name,
            dtype.kind().name()
        ))),
        _ => Ok(dtype),
    }
}

/// An array's axes split into the ones a reduction keeps and the ones it combines, both in
/// logical (ascending) axis order.
struct AxisSplit {
    kept_axes: Vec<usize>,
    reduce_axes: Vec<usize>,
    kept_shape: Vec<usize>,
    reduce_shape: Vec<usize>,
}

impl AxisSplit {
    fn new(shape: &[usize], axes: &Axes) -> Self {
        let ndim = shape.len();
        let kept_axes = (0..ndim)
            .filter(|&axis| !axes.contains(axis))
            .collect::<Vec<_>>();
        let reduce_axes = (0..ndim)
            .filter(|&axis| axes.contains(axis))
            .collect::<Vec<_>>();
        let kept_shape = kept_axes.iter().map(|&axis| shape[axis]).collect();
        let reduce_shape = reduce_axes.iter().map(|&axis| shape[axis]).collect();
        Self {
            kept_axes,
            reduce_axes,
            kept_shape,
            reduce_shape,
        }
    }

    /// Axis order with the kept axes outermost and the reduced axes innermost, so gathering
    /// `array` in this order lays out each output cell's inputs as one contiguous run.
    fn reading_order(&self) -> Vec<usize> {
        self.kept_axes
            .iter()
            .chain(&self.reduce_axes)
            .copied()
            .collect()
    }

    fn kept_count(&self) -> usize {
        self.kept_shape.iter().product()
    }

    fn reduce_count(&self) -> usize {
        self.reduce_shape.iter().product()
    }

    /// The output shape: the kept axes alone, or every axis with the reduced ones collapsed to
    /// length one, in their original positions.
    fn output_shape(&self, input_shape: &[usize], keepdims: bool) -> Vec<usize> {
        if !keepdims {
            return self.kept_shape.clone();
        }
        input_shape
            .iter()
            .enumerate()
            .map(|(axis, &dimension)| {
                if self.reduce_axes.contains(&axis) {
                    1
                } else {
                    dimension
                }
            })
            .collect()
    }
}

/// Read `array`'s elements through `order` (see [`AxisSplit::reading_order`]), charging CPU for
/// the elements visited before touching them, as every reduction here does before combining.
fn gather<T: Element>(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    order: &[usize],
) -> PyResult<Vec<T>> {
    array::read_elements(runtime, &layout::reading_order(array, order))
}

fn gather_objects(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    order: &[usize],
) -> PyResult<Vec<PyValue>> {
    array::read_objects(runtime, &layout::reading_order(array, order))
}

/// `where=`, converted to a boolean array, or `None` for the common case (omitted, or the
/// literal `True` NumPy's own signature defaults to) where every element participates.
fn where_array(runtime: &mut dyn PyRuntime, value: Option<PyValue>) -> PyResult<Option<Array>> {
    let Some(value) = value else { return Ok(None) };
    if value.bool_value() == Some(true) {
        return Ok(None);
    }
    Ok(Some(convert::array_from_python(
        runtime,
        value,
        Some(DType::BOOL),
        false,
    )?))
}

/// `mask` broadcast to `shape` and read through `order`, alongside the array it selects from.
fn gather_mask(
    runtime: &mut dyn PyRuntime,
    mask: &Array,
    shape: &[usize],
    order: &[usize],
) -> PyResult<Vec<bool>> {
    let broadcast = layout::broadcast_reading_order(mask, shape, order)?;
    array::read_elements(runtime, &broadcast)
}

/// The elements of one output cell's run: the whole window when `where=` selects everything
/// (the common case, at no extra cost), or the selected elements copied out otherwise.
fn window<'a, T: Copy>(
    values: &'a [T],
    mask: Option<&[bool]>,
    bucket: usize,
    count: usize,
    storage: &'a mut Vec<T>,
) -> &'a [T] {
    let values = &values[bucket * count..(bucket + 1) * count];
    let Some(mask) = mask else { return values };
    let mask = &mask[bucket * count..(bucket + 1) * count];
    storage.clear();
    storage.extend(
        values
            .iter()
            .zip(mask)
            .filter_map(|(value, selected)| selected.then_some(*value)),
    );
    storage
}

fn identity_numeric<T: Numeric>(op: NumericOp) -> Option<T> {
    match op {
        NumericOp::Add => Some(T::zero()),
        NumericOp::Multiply => Some(T::one()),
        _ => None,
    }
}

fn combine_numeric<T: Numeric>(op: NumericOp, a: T, b: T, flags: &mut FpFlags) -> T {
    match op {
        NumericOp::Add => a.add(b, flags),
        NumericOp::Subtract => a.subtract(b, flags),
        NumericOp::Multiply => a.multiply(b, flags),
        NumericOp::Divide => a.divide(b, flags),
        NumericOp::FloorDivide => a.floor_divide(b, flags),
        NumericOp::Remainder => a.remainder(b, flags),
        // `fmod` has no dedicated `Numeric` method (only `add`/`subtract`/.../`power`, which
        // every element type already needs for the arithmetic operators); reusing the
        // elementwise loop's own function table keeps this identical to `np.fmod(a, b)` instead
        // of a second, possibly-diverging implementation of C's `fmod` sign rule.
        NumericOp::Fmod => ufunc::arith_fn(ArithOp::Fmod)(a, b, flags),
        NumericOp::Power => a.power(b, flags),
        NumericOp::Maximum => a.maximum(b),
        NumericOp::Minimum => a.minimum(b),
        NumericOp::Fmax => a.fmax(b),
        NumericOp::Fmin => a.fmin(b),
    }
}

/// `power.reduce`/`accumulate` on a signed integer array rejects a negative element used as an
/// exponent, the same check `**` makes before its whole-array loop runs (see `ufunc.rs`'s
/// `reject_negative_exponent`, which cannot be called from here: it is private to that module,
/// and it inspects a whole operand up front rather than one element of a running fold). Every
/// element but the very first plays the exponent role at some point in a left-to-right fold, so
/// this runs once per element combined, not once per reduction. Unsigned integers can never
/// compare less than zero, and this only ever fires for `T::IS_INTEGER`, so float and complex
/// exponents (which real `power` allows to be negative) never reach it.
fn check_power_exponent<T: Numeric>(op: NumericOp, value: T) -> PyResult<()> {
    if op == NumericOp::Power && T::IS_INTEGER && value.compare(T::zero()) == Some(Ordering::Less) {
        return Err(PyError::value_error(
            "Integers to negative integer powers are not allowed.",
        ));
    }
    Ok(())
}

/// Combine every output cell's run of `T` (see [`AxisSplit`]) with `op`. `Add` sums pairwise
/// through [`ops::Numeric::pairwise_add`]; every other operator folds sequentially in the order
/// read. An operator without an identity (everything but `Add`/`Multiply`) needs `initial` or a
/// non-empty run.
fn reduce_numeric<T: Numeric + Element>(
    name: &str,
    op: NumericOp,
    split: &AxisSplit,
    values: &[T],
    mask: Option<&[bool]>,
    initial: Option<Number>,
) -> PyResult<(Vec<T>, FpFlags)> {
    let seed = initial
        .map(T::from_number)
        .or_else(|| identity_numeric::<T>(op));
    let (kept_count, reduce_count) = (split.kept_count(), split.reduce_count());
    let mut flags = FpFlags::default();
    let mut output = Vec::with_capacity(kept_count);
    let mut storage = Vec::new();
    for bucket in 0..kept_count {
        let run = window(values, mask, bucket, reduce_count, &mut storage);
        let result = if op == NumericOp::Add {
            seed.unwrap_or_else(T::zero).pairwise_add(run, &mut flags)
        } else {
            match seed {
                Some(seed) => run.iter().try_fold(seed, |acc, &value| {
                    check_power_exponent(op, value)?;
                    Ok(combine_numeric(op, acc, value, &mut flags))
                })?,
                None => {
                    let Some((&first, rest)) = run.split_first() else {
                        return Err(no_identity_error(name));
                    };
                    rest.iter().try_fold(first, |acc, &value| {
                        check_power_exponent(op, value)?;
                        Ok(combine_numeric(op, acc, value, &mut flags))
                    })?
                }
            }
        };
        output.push(result);
    }
    Ok((output, flags))
}

fn identity_integer<T: Integer>(op: IntegerOp) -> Option<T> {
    match op {
        IntegerOp::And => Some(T::from_number(Number::Int(-1))),
        IntegerOp::Or | IntegerOp::Xor => Some(T::zero()),
        IntegerOp::LeftShift | IntegerOp::RightShift => None,
    }
}

fn combine_integer<T: Integer>(op: IntegerOp, a: T, b: T) -> T {
    match op {
        IntegerOp::And => a.bit_and(b),
        IntegerOp::Or => a.bit_or(b),
        IntegerOp::Xor => a.bit_xor(b),
        IntegerOp::LeftShift => a.left_shift(b),
        IntegerOp::RightShift => a.right_shift(b),
    }
}

fn reduce_integer<T: Integer + Element>(
    name: &str,
    op: IntegerOp,
    split: &AxisSplit,
    values: &[T],
    mask: Option<&[bool]>,
    initial: Option<Number>,
) -> PyResult<Vec<T>> {
    let seed = initial
        .map(T::from_number)
        .or_else(|| identity_integer::<T>(op));
    let (kept_count, reduce_count) = (split.kept_count(), split.reduce_count());
    let mut output = Vec::with_capacity(kept_count);
    let mut storage = Vec::new();
    for bucket in 0..kept_count {
        let run = window(values, mask, bucket, reduce_count, &mut storage);
        let result = match seed {
            Some(seed) => run
                .iter()
                .fold(seed, |acc, &value| combine_integer(op, acc, value)),
            None => {
                let Some((&first, rest)) = run.split_first() else {
                    return Err(no_identity_error(name));
                };
                rest.iter()
                    .fold(first, |acc, &value| combine_integer(op, acc, value))
            }
        };
        output.push(result);
    }
    Ok(output)
}

/// `hypot`'s identity is `0` (`hypot.reduce([])` is `0.0`, no error). `logaddexp`/`logaddexp2`'s
/// is `-inf`, the identity for log-sum-exp (matching `log(0)`). `arctan2`, `copysign`, and
/// `heaviside` have none.
fn identity_float2<T: Real>(op: Float2Op) -> Option<T> {
    match op {
        Float2Op::Hypot => Some(T::from_f64(0.0)),
        Float2Op::Logaddexp | Float2Op::Logaddexp2 => Some(T::from_f64(f64::NEG_INFINITY)),
        Float2Op::Arctan2 | Float2Op::Copysign | Float2Op::Heaviside => None,
    }
}

/// `op`'s function pair ([`ufunc::float2_fn`]) applied through [`ops::Real::zip`], the same
/// mechanism the elementwise loop uses, so this matches `np.arctan2(a, b)` and friends exactly
/// rather than risking a second, possibly-diverging implementation of the same math.
fn combine_float2<T: Real>(op: Float2Op, a: T, b: T, flags: &mut FpFlags) -> T {
    let (double, single) = ufunc::float2_fn(op);
    let result = a.zip(b, double, single);
    ufunc::float_flags_binary(a.to_f64(), b.to_f64(), result.to_f64(), flags);
    result
}

/// Combine every output cell's run of `T` with `op`, always sequentially: none of `arctan2`,
/// `hypot`, `copysign`, `logaddexp`, `logaddexp2`, or `heaviside` is associative in a way
/// pairwise splitting would preserve, so unlike `add` this never sums in parallel halves.
fn reduce_float2<T: Real + Element>(
    name: &str,
    op: Float2Op,
    split: &AxisSplit,
    values: &[T],
    mask: Option<&[bool]>,
    initial: Option<Number>,
) -> PyResult<(Vec<T>, FpFlags)> {
    let seed = initial.map(T::from_number).or_else(|| identity_float2(op));
    let (kept_count, reduce_count) = (split.kept_count(), split.reduce_count());
    let mut flags = FpFlags::default();
    let mut output = Vec::with_capacity(kept_count);
    let mut storage = Vec::new();
    for bucket in 0..kept_count {
        let run = window(values, mask, bucket, reduce_count, &mut storage);
        let result = match seed {
            Some(seed) => run.iter().fold(seed, |acc, &value| {
                combine_float2(op, acc, value, &mut flags)
            }),
            None => {
                let Some((&first, rest)) = run.split_first() else {
                    return Err(no_identity_error(name));
                };
                rest.iter().fold(first, |acc, &value| {
                    combine_float2(op, acc, value, &mut flags)
                })
            }
        };
        output.push(result);
    }
    Ok((output, flags))
}

/// `And`/`Or`/`Xor` always have an identity; a comparison has none (`equal.reduce([])` raises
/// the same "which has no identity" error as `maximum.reduce([])`).
fn identity_bool(op: BoolOp) -> Option<bool> {
    match op {
        BoolOp::And => Some(true),
        BoolOp::Or | BoolOp::Xor => Some(false),
        BoolOp::Compare(_) => None,
    }
}

/// `equal`/`not_equal`/`less`/`less_equal`/`greater`/`greater_equal` on two already-boolean
/// operands: NumPy only registers a comparison reduce/accumulate loop for boolean input, so by
/// the time this runs `resolve_dtype` has already rejected anything else.
fn combine_compare(op: CompareOp, a: bool, b: bool) -> bool {
    match op {
        CompareOp::Equal => a == b,
        CompareOp::NotEqual => a != b,
        CompareOp::Less => !a & b,
        CompareOp::LessEqual => !a | b,
        CompareOp::Greater => a & !b,
        CompareOp::GreaterEqual => a | !b,
    }
}

fn combine_bool(op: BoolOp, a: bool, b: bool) -> bool {
    match op {
        BoolOp::And => a && b,
        BoolOp::Or => a || b,
        BoolOp::Xor => a ^ b,
        BoolOp::Compare(cmp) => combine_compare(cmp, a, b),
    }
}

/// `And`/`Or`/`Xor` always have an identity, so only a comparison reduction can fail on an
/// empty run without `initial`.
fn reduce_bool(
    name: &str,
    op: BoolOp,
    split: &AxisSplit,
    values: &[bool],
    mask: Option<&[bool]>,
    initial: Option<bool>,
) -> PyResult<Vec<bool>> {
    let seed = initial.or_else(|| identity_bool(op));
    let (kept_count, reduce_count) = (split.kept_count(), split.reduce_count());
    let mut storage = Vec::new();
    (0..kept_count)
        .map(|bucket| {
            let run = window(values, mask, bucket, reduce_count, &mut storage);
            match seed {
                Some(seed) => Ok(run
                    .iter()
                    .fold(seed, |acc, &value| combine_bool(op, acc, value))),
                None => {
                    let Some((&first, rest)) = run.split_first() else {
                        return Err(no_identity_error(name));
                    };
                    Ok(rest
                        .iter()
                        .fold(first, |acc, &value| combine_bool(op, acc, value)))
                }
            }
        })
        .collect()
}

fn identity_object(op: NumericOp) -> Option<PyValue> {
    match op {
        NumericOp::Add => Some(Value::Int(0)),
        NumericOp::Multiply => Some(Value::Int(1)),
        _ => None,
    }
}

/// Object arrays reduce with the ufunc's own Python operator ([`ufunc::object_element`]),
/// element by element, starting from `initial` or else the first element, so
/// `np.array(["a", "b"], dtype=object).sum()` is `"ab"`. Only an empty run falls back to the
/// ufunc's identity. There is no pairwise summation for `object` dtype.
fn reduce_object(
    runtime: &mut dyn PyRuntime,
    ufunc: &UfuncDef,
    op: NumericOp,
    split: &AxisSplit,
    values: &[PyValue],
    mask: Option<&[bool]>,
    initial: Option<PyValue>,
) -> PyResult<Vec<PyValue>> {
    let (kept_count, reduce_count) = (split.kept_count(), split.reduce_count());
    let mut output = Vec::with_capacity(kept_count);
    let mut storage = Vec::new();
    for bucket in 0..kept_count {
        let run = window(values, mask, bucket, reduce_count, &mut storage);
        let mut elements = run.iter().copied();
        let Some(mut accumulator) = initial
            .or_else(|| elements.next())
            .or_else(|| identity_object(op))
        else {
            return Err(no_identity_error(ufunc.name));
        };
        for value in elements {
            accumulator = ufunc::object_element(runtime, ufunc, &[accumulator, value])?;
        }
        output.push(accumulator);
    }
    Ok(output)
}

/// The shared body of `ufunc.reduce`, `sum`, `prod`, `max`/`amax`, `min`/`amin`, `any`, `all`,
/// and `trace`: resolve the dtype, split the axes, read `where=` and `initial=`, combine every
/// output cell's run, and assemble the result (see [`finish_reduction`]).
#[allow(clippy::too_many_arguments)]
pub(in crate::python) fn reduce_call(
    runtime: &mut dyn PyRuntime,
    ufunc: &UfuncDef,
    array: Array,
    axes: Axes,
    dtype: Option<DType>,
    out: Option<Array>,
    keepdims: bool,
    initial: Option<PyValue>,
    where_: Option<PyValue>,
) -> PyResult<PyValue> {
    let op = classify(ufunc)?;
    let dtype = resolve_dtype(ufunc, op, array.dtype, dtype)?;
    let split = AxisSplit::new(array.shape(), &axes);
    let array = if array.dtype == dtype {
        array
    } else {
        convert::cast_array(runtime, &array, dtype, false)?
    };
    let mask = where_array(runtime, where_)?;
    // An object reduction starts from its first element, so NumPy treats it as having no
    // identity for `where=`.
    let has_identity = has_identity(op) && dtype.kind() != Kind::Object;
    if mask.is_some() && initial.is_none() && !has_identity {
        return Err(where_needs_initial_error(ufunc.name));
    }
    let order = split.reading_order();
    array::reserve_elements(runtime, dtype, split.kept_count())?;
    runtime.charge_cpu(array.size() as u64 + 1)?;
    let mask_values = mask
        .as_ref()
        .map(|mask| gather_mask(runtime, mask, array.shape(), &order))
        .transpose()?;
    let (buffer, flags) = match (dtype.kind(), op) {
        (Kind::Object, ReduceOp::Numeric(numeric_op)) => {
            let values = gather_objects(runtime, &array, &order)?;
            let output = reduce_object(
                runtime,
                ufunc,
                numeric_op,
                &split,
                &values,
                mask_values.as_deref(),
                initial,
            )?;
            (PyArrayBuffer::Values(output), FpFlags::default())
        }
        (_, ReduceOp::Numeric(numeric_op)) => dispatch_numeric!(dtype.kind(), T => {
            let values = gather::<T>(runtime, &array, &order)?;
            let initial = initial.map(|value| numeric_initial(runtime, value, dtype)).transpose()?;
            let (output, flags) = reduce_numeric::<T>(ufunc.name, numeric_op, &split, &values, mask_values.as_deref(), initial)?;
            (PyArrayBuffer::Bytes(array::pack_elements(&output)), flags)
        }, _ => unreachable!("object and str dtypes are rejected before this dispatch")),
        (_, ReduceOp::Integer(integer_op)) => dispatch_integer!(dtype.kind(), T => {
            let values = gather::<T>(runtime, &array, &order)?;
            let initial = initial.map(|value| numeric_initial(runtime, value, dtype)).transpose()?;
            let output = reduce_integer::<T>(ufunc.name, integer_op, &split, &values, mask_values.as_deref(), initial)?;
            (PyArrayBuffer::Bytes(array::pack_elements(&output)), FpFlags::default())
        }, _ => unreachable!("resolve_dtype only allows bool/signed/unsigned for bitwise ops")),
        (Kind::Bool, ReduceOp::Bool(bool_op)) => {
            let values = gather::<bool>(runtime, &array, &order)?;
            let initial = initial.map(|value| runtime.truth(&value)).transpose()?;
            let output = reduce_bool(
                ufunc.name,
                bool_op,
                &split,
                &values,
                mask_values.as_deref(),
                initial,
            )?;
            (
                PyArrayBuffer::Bytes(array::pack_elements(&output)),
                FpFlags::default(),
            )
        }
        (_, ReduceOp::Bool(_)) => {
            unreachable!("resolve_dtype only resolves Bool ops to the bool dtype")
        }
        (_, ReduceOp::Float2(float2_op)) => dispatch_real!(dtype.kind(), T => {
            let values = gather::<T>(runtime, &array, &order)?;
            let initial = initial.map(|value| numeric_initial(runtime, value, dtype)).transpose()?;
            let (output, flags) = reduce_float2::<T>(ufunc.name, float2_op, &split, &values, mask_values.as_deref(), initial)?;
            (PyArrayBuffer::Bytes(array::pack_elements(&output)), flags)
        }, _ => unreachable!("resolve_dtype only resolves Float2 ops to a float dtype")),
    };
    super::errstate::report(runtime, "reduce", flags)?;
    finish_reduction(runtime, dtype, array.shape(), &split, keepdims, buffer, out)
}

fn has_identity(op: ReduceOp) -> bool {
    match op {
        ReduceOp::Numeric(numeric_op) => matches!(numeric_op, NumericOp::Add | NumericOp::Multiply),
        ReduceOp::Integer(_) => false,
        ReduceOp::Bool(bool_op) => identity_bool(bool_op).is_some(),
        ReduceOp::Float2(op) => identity_float2::<f64>(op).is_some(),
    }
}

fn numeric_initial(runtime: &mut dyn PyRuntime, value: PyValue, dtype: DType) -> PyResult<Number> {
    let leaf = convert::leaf(runtime, &value)?;
    convert::leaf_number(&leaf, dtype)
}

/// Wrap a reduction's flat output buffer (in [`AxisSplit::kept_shape`] order) in an array of the
/// right output shape, store it through `out=` if given, and box a 0-d result to a NumPy scalar.
fn finish_reduction(
    runtime: &mut dyn PyRuntime,
    dtype: DType,
    input_shape: &[usize],
    split: &AxisSplit,
    keepdims: bool,
    buffer: PyArrayBuffer,
    out: Option<Array>,
) -> PyResult<PyValue> {
    let shape = split.output_shape(input_shape, keepdims);
    let result = array::new_array(runtime, buffer, dtype, shape)?;
    if let Some(out) = out {
        array::assign(runtime, &out, &result)?;
        Ok(out.value())
    } else if result.ndim() == 0 {
        convert::element_to_scalar(runtime, &result, result.view.offset)
    } else {
        Ok(result.value())
    }
}

/// The shared body of `ufunc.accumulate`, `cumsum`, and `cumprod`: resolve the dtype, walk
/// `axis` with the other axes outermost, and write each run's running totals back over itself
/// (no pairwise summation: the whole point of `accumulate` is to expose every partial total, so
/// combining always folds sequentially in the order read).
fn accumulate_call(
    runtime: &mut dyn PyRuntime,
    ufunc: &UfuncDef,
    array: Array,
    axis: usize,
    dtype: Option<DType>,
    out: Option<Array>,
) -> PyResult<PyValue> {
    let op = classify(ufunc)?;
    let dtype = resolve_dtype(ufunc, op, array.dtype, dtype)?;
    let array = if array.dtype == dtype {
        array
    } else {
        convert::cast_array(runtime, &array, dtype, false)?
    };
    let kept_axes = (0..array.ndim())
        .filter(|&candidate| candidate != axis)
        .collect::<Vec<_>>();
    let order = kept_axes
        .iter()
        .chain(std::iter::once(&axis))
        .copied()
        .collect::<Vec<_>>();
    let axis_len = array.shape()[axis];
    let kept_count: usize = kept_axes
        .iter()
        .map(|&candidate| array.shape()[candidate])
        .product();
    array::reserve_elements(runtime, dtype, array.size())?;
    runtime.charge_cpu(array.size() as u64 + 1)?;
    let (buffer, flags) = match (dtype.kind(), op) {
        (Kind::Object, ReduceOp::Numeric(_)) => {
            let values = gather_objects(runtime, &array, &order)?;
            let output = accumulate_object(runtime, ufunc, &values, kept_count, axis_len)?;
            (PyArrayBuffer::Values(output), FpFlags::default())
        }
        (_, ReduceOp::Numeric(numeric_op)) => dispatch_numeric!(dtype.kind(), T => {
            let values = gather::<T>(runtime, &array, &order)?;
            let (output, flags) = accumulate_numeric::<T>(numeric_op, &values, kept_count, axis_len)?;
            (PyArrayBuffer::Bytes(array::pack_elements(&output)), flags)
        }, _ => unreachable!("object and str dtypes are rejected before this dispatch")),
        (_, ReduceOp::Integer(integer_op)) => dispatch_integer!(dtype.kind(), T => {
            let values = gather::<T>(runtime, &array, &order)?;
            let output = accumulate_integer::<T>(integer_op, &values, kept_count, axis_len);
            (PyArrayBuffer::Bytes(array::pack_elements(&output)), FpFlags::default())
        }, _ => unreachable!("resolve_dtype only allows bool/signed/unsigned for bitwise ops")),
        (Kind::Bool, ReduceOp::Bool(bool_op)) => {
            let values = gather::<bool>(runtime, &array, &order)?;
            let output = accumulate_bool(bool_op, &values, kept_count, axis_len);
            (
                PyArrayBuffer::Bytes(array::pack_elements(&output)),
                FpFlags::default(),
            )
        }
        (_, ReduceOp::Bool(_)) => {
            unreachable!("resolve_dtype only resolves Bool ops to the bool dtype")
        }
        (_, ReduceOp::Float2(float2_op)) => dispatch_real!(dtype.kind(), T => {
            let values = gather::<T>(runtime, &array, &order)?;
            let (output, flags) = accumulate_float2::<T>(float2_op, &values, kept_count, axis_len);
            (PyArrayBuffer::Bytes(array::pack_elements(&output)), flags)
        }, _ => unreachable!("resolve_dtype only resolves Float2 ops to a float dtype")),
    };
    super::errstate::report(runtime, "accumulate", flags)?;
    let result = layout::new_array(runtime, buffer, dtype, array.shape().to_vec(), &order)?;
    if let Some(out) = out {
        array::assign(runtime, &out, &result)?;
        Ok(out.value())
    } else {
        Ok(result.value())
    }
}

fn accumulate_numeric<T: Numeric + Element>(
    op: NumericOp,
    values: &[T],
    kept_count: usize,
    axis_len: usize,
) -> PyResult<(Vec<T>, FpFlags)> {
    let mut flags = FpFlags::default();
    let mut output = Vec::with_capacity(values.len());
    for bucket in 0..kept_count {
        let mut run = values[bucket * axis_len..(bucket + 1) * axis_len].iter();
        let Some(&first) = run.next() else { continue };
        let mut total = first;
        output.push(total);
        for &value in run {
            check_power_exponent(op, value)?;
            total = combine_numeric(op, total, value, &mut flags);
            output.push(total);
        }
    }
    Ok((output, flags))
}

fn accumulate_integer<T: Integer + Element>(
    op: IntegerOp,
    values: &[T],
    kept_count: usize,
    axis_len: usize,
) -> Vec<T> {
    let mut output = Vec::with_capacity(values.len());
    for bucket in 0..kept_count {
        let mut run = values[bucket * axis_len..(bucket + 1) * axis_len].iter();
        let Some(&first) = run.next() else { continue };
        let mut total = first;
        output.push(total);
        for &value in run {
            total = combine_integer(op, total, value);
            output.push(total);
        }
    }
    output
}

fn accumulate_bool(op: BoolOp, values: &[bool], kept_count: usize, axis_len: usize) -> Vec<bool> {
    let mut output = Vec::with_capacity(values.len());
    for bucket in 0..kept_count {
        let mut run = values[bucket * axis_len..(bucket + 1) * axis_len].iter();
        let Some(&first) = run.next() else { continue };
        let mut total = first;
        output.push(total);
        for &value in run {
            total = combine_bool(op, total, value);
            output.push(total);
        }
    }
    output
}

fn accumulate_float2<T: Real + Element>(
    op: Float2Op,
    values: &[T],
    kept_count: usize,
    axis_len: usize,
) -> (Vec<T>, FpFlags) {
    let mut flags = FpFlags::default();
    let mut output = Vec::with_capacity(values.len());
    for bucket in 0..kept_count {
        let mut run = values[bucket * axis_len..(bucket + 1) * axis_len].iter();
        let Some(&first) = run.next() else { continue };
        let mut total = first;
        output.push(total);
        for &value in run {
            total = combine_float2(op, total, value, &mut flags);
            output.push(total);
        }
    }
    (output, flags)
}

fn accumulate_object(
    runtime: &mut dyn PyRuntime,
    ufunc: &UfuncDef,
    values: &[PyValue],
    kept_count: usize,
    axis_len: usize,
) -> PyResult<Vec<PyValue>> {
    let mut output = Vec::with_capacity(values.len());
    for bucket in 0..kept_count {
        let mut run = values[bucket * axis_len..(bucket + 1) * axis_len]
            .iter()
            .copied();
        let Some(mut total) = run.next() else {
            continue;
        };
        output.push(total);
        for value in run {
            total = ufunc::object_element(runtime, ufunc, &[total, value])?;
            output.push(total);
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------------------------
// `sum`, `prod`, `max`/`amax`, `min`/`amin`, `any`, `all`
// ---------------------------------------------------------------------------------------------

static SUM_SIGNATURE: Signature = Signature::new(
    "sum",
    &["a", "axis", "dtype", "out", "keepdims", "initial", "where"],
    1,
);
static SUM_METHOD_SIGNATURE: Signature = Signature::new(
    "_sum",
    &["axis", "dtype", "out", "keepdims", "initial", "where"],
    0,
);
static PROD_SIGNATURE: Signature = Signature::new(
    "prod",
    &["a", "axis", "dtype", "out", "keepdims", "initial", "where"],
    1,
);
static PROD_METHOD_SIGNATURE: Signature = Signature::new(
    "_prod",
    &["axis", "dtype", "out", "keepdims", "initial", "where"],
    0,
);

fn sum_prod(
    runtime: &mut dyn PyRuntime,
    ufunc_name: &str,
    array: Array,
    bound: &Bound,
) -> PyResult {
    let ufunc = ufunc_named(ufunc_name);
    let axes = args::axes(runtime, bound.get("axis"), array.ndim())?;
    let dtype = args::optional_dtype(runtime, bound.get("dtype"))?;
    let out = out_array(runtime, bound.get("out"))?;
    let keepdims = args::flag(runtime, bound.get("keepdims"), false)?;
    let initial = bound.value("initial");
    let where_ = bound.value("where");
    reduce_call(
        runtime, ufunc, array, axes, dtype, out, keepdims, initial, where_,
    )
}

fn module_sum(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = SUM_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    sum_prod(runtime, "add", array, &bound)
}

fn method_sum(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = SUM_METHOD_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, receiver)?;
    sum_prod(runtime, "add", array, &bound)
}

fn module_prod(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = PROD_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    sum_prod(runtime, "multiply", array, &bound)
}

fn method_prod(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = PROD_METHOD_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, receiver)?;
    sum_prod(runtime, "multiply", array, &bound)
}

static MAX_SIGNATURE: Signature = Signature::new(
    "amax",
    &["a", "axis", "out", "keepdims", "initial", "where"],
    1,
);
static MAX_METHOD_SIGNATURE: Signature =
    Signature::new("_amax", &["axis", "out", "keepdims", "initial", "where"], 0);
static MIN_SIGNATURE: Signature = Signature::new(
    "amin",
    &["a", "axis", "out", "keepdims", "initial", "where"],
    1,
);
static MIN_METHOD_SIGNATURE: Signature =
    Signature::new("_amin", &["axis", "out", "keepdims", "initial", "where"], 0);

fn max_min(runtime: &mut dyn PyRuntime, ufunc_name: &str, array: Array, bound: &Bound) -> PyResult {
    let ufunc = ufunc_named(ufunc_name);
    let axes = args::axes(runtime, bound.get("axis"), array.ndim())?;
    let out = out_array(runtime, bound.get("out"))?;
    let keepdims = args::flag(runtime, bound.get("keepdims"), false)?;
    let initial = bound.value("initial");
    let where_ = bound.value("where");
    reduce_call(
        runtime, ufunc, array, axes, None, out, keepdims, initial, where_,
    )
}

fn module_max(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = MAX_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    max_min(runtime, "maximum", array, &bound)
}

fn method_max(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = MAX_METHOD_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, receiver)?;
    max_min(runtime, "maximum", array, &bound)
}

fn module_min(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = MIN_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    max_min(runtime, "minimum", array, &bound)
}

fn method_min(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = MIN_METHOD_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, receiver)?;
    max_min(runtime, "minimum", array, &bound)
}

static ANY_SIGNATURE: Signature =
    Signature::new("any", &["a", "axis", "out", "keepdims"], 1).keyword_only(&["where"]);
static ANY_METHOD_SIGNATURE: Signature =
    Signature::new("_any", &["axis", "out", "keepdims"], 0).keyword_only(&["where"]);
static ALL_SIGNATURE: Signature =
    Signature::new("all", &["a", "axis", "out", "keepdims"], 1).keyword_only(&["where"]);
static ALL_METHOD_SIGNATURE: Signature =
    Signature::new("_all", &["axis", "out", "keepdims"], 0).keyword_only(&["where"]);

fn any_all(runtime: &mut dyn PyRuntime, ufunc_name: &str, array: Array, bound: &Bound) -> PyResult {
    let ufunc = ufunc_named(ufunc_name);
    let axes = args::axes(runtime, bound.get("axis"), array.ndim())?;
    let out = out_array(runtime, bound.get("out"))?;
    let keepdims = args::flag(runtime, bound.get("keepdims"), false)?;
    let where_ = bound.value("where");
    reduce_call(
        runtime, ufunc, array, axes, None, out, keepdims, None, where_,
    )
}

fn module_any(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ANY_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    any_all(runtime, "logical_or", array, &bound)
}

fn method_any(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = ANY_METHOD_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, receiver)?;
    any_all(runtime, "logical_or", array, &bound)
}

fn module_all(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ALL_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    any_all(runtime, "logical_and", array, &bound)
}

fn method_all(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = ALL_METHOD_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, receiver)?;
    any_all(runtime, "logical_and", array, &bound)
}

// ---------------------------------------------------------------------------------------------
// `argmin`, `argmax`
// ---------------------------------------------------------------------------------------------

static ARGMIN_SIGNATURE: Signature =
    Signature::new("argmin", &["a", "axis", "out"], 1).keyword_only(&["keepdims"]);
static ARGMIN_METHOD_SIGNATURE: Signature =
    Signature::new("argmin", &["axis", "out"], 0).keyword_only(&["keepdims"]);
static ARGMAX_SIGNATURE: Signature =
    Signature::new("argmax", &["a", "axis", "out"], 1).keyword_only(&["keepdims"]);
static ARGMAX_METHOD_SIGNATURE: Signature =
    Signature::new("argmax", &["axis", "out"], 0).keyword_only(&["keepdims"]);

fn arg_extreme(
    runtime: &mut dyn PyRuntime,
    array: Array,
    axis: Option<usize>,
    out: Option<Array>,
    keepdims: bool,
    is_max: bool,
    name: &str,
) -> PyResult<PyValue> {
    let axes = match axis {
        Some(axis) => Axes::Some(vec![axis]),
        None => Axes::All,
    };
    let split = AxisSplit::new(array.shape(), &axes);
    if split.kept_count() > 0 && split.reduce_count() == 0 {
        return Err(PyError::value_error(format!(
            "attempt to get {name} of an empty sequence"
        )));
    }
    let order = split.reading_order();
    array::reserve_elements(runtime, DType::INT64, split.kept_count())?;
    runtime.charge_cpu(array.size() as u64 + 1)?;
    let bytes = match array.dtype.kind() {
        Kind::Object | Kind::Str => {
            return Err(PyError::not_implemented_error(format!(
                "{name} of '{}' arrays is not supported by shellsim's NumPy",
                array.dtype.kind().name()
            )))
        }
        kind => dispatch_numeric!(kind, T => {
            let values = gather::<T>(runtime, &array, &order)?;
            let output = arg_extreme_numeric::<T>(&values, split.kept_count(), split.reduce_count(), is_max);
            array::pack_elements(&output)
        }, _ => unreachable!("object and str are rejected above")),
    };
    finish_reduction(
        runtime,
        DType::INT64,
        array.shape(),
        &split,
        keepdims,
        PyArrayBuffer::Bytes(bytes),
        out,
    )
}

/// The index of the first extreme value in each output cell's run. NaN dominates both
/// directions, as [`ops::Numeric::maximum`]/`minimum` treat it, so the first NaN in a run
/// becomes that run's argmax/argmin and no later comparison can replace it.
fn arg_extreme_numeric<T: Numeric>(
    values: &[T],
    kept_count: usize,
    reduce_count: usize,
    is_max: bool,
) -> Vec<i64> {
    (0..kept_count)
        .map(|bucket| {
            let window = &values[bucket * reduce_count..(bucket + 1) * reduce_count];
            let mut best_index = 0;
            let mut best_value = window[0];
            for (index, &value) in window.iter().enumerate().skip(1) {
                if best_value.is_nan() {
                    break;
                }
                let replace = value.is_nan()
                    || matches!(
                        (is_max, value.compare(best_value)),
                        (true, Some(Ordering::Greater)) | (false, Some(Ordering::Less))
                    );
                if replace {
                    best_value = value;
                    best_index = index;
                }
            }
            best_index as i64
        })
        .collect()
}

fn module_argmin(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ARGMIN_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let axis = args::axis(runtime, bound.get("axis"), array.ndim())?;
    let out = out_array(runtime, bound.get("out"))?;
    let keepdims = args::flag(runtime, bound.get("keepdims"), false)?;
    arg_extreme(runtime, array, axis, out, keepdims, false, "argmin")
}

fn method_argmin(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = ARGMIN_METHOD_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, receiver)?;
    let axis = args::axis(runtime, bound.get("axis"), array.ndim())?;
    let out = out_array(runtime, bound.get("out"))?;
    let keepdims = args::flag(runtime, bound.get("keepdims"), false)?;
    arg_extreme(runtime, array, axis, out, keepdims, false, "argmin")
}

fn module_argmax(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ARGMAX_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let axis = args::axis(runtime, bound.get("axis"), array.ndim())?;
    let out = out_array(runtime, bound.get("out"))?;
    let keepdims = args::flag(runtime, bound.get("keepdims"), false)?;
    arg_extreme(runtime, array, axis, out, keepdims, true, "argmax")
}

fn method_argmax(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = ARGMAX_METHOD_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, receiver)?;
    let axis = args::axis(runtime, bound.get("axis"), array.ndim())?;
    let out = out_array(runtime, bound.get("out"))?;
    let keepdims = args::flag(runtime, bound.get("keepdims"), false)?;
    arg_extreme(runtime, array, axis, out, keepdims, true, "argmax")
}

// ---------------------------------------------------------------------------------------------
// `cumsum`, `cumprod`
// ---------------------------------------------------------------------------------------------

static CUMSUM_SIGNATURE: Signature = Signature::new("cumsum", &["a", "axis", "dtype", "out"], 1);
static CUMSUM_METHOD_SIGNATURE: Signature = Signature::new("cumsum", &["axis", "dtype", "out"], 0);
static CUMPROD_SIGNATURE: Signature = Signature::new("cumprod", &["a", "axis", "dtype", "out"], 1);
static CUMPROD_METHOD_SIGNATURE: Signature =
    Signature::new("cumprod", &["axis", "dtype", "out"], 0);

/// `axis=None` flattens the array first, as `cumsum`/`cumprod` do (unlike `ufunc.accumulate`,
/// which rejects `None`).
fn cumulative(
    runtime: &mut dyn PyRuntime,
    ufunc_name: &str,
    array: Array,
    bound: &Bound,
) -> PyResult {
    let ufunc = ufunc_named(ufunc_name);
    let (array, axis) = match args::axis(runtime, bound.get("axis"), array.ndim())? {
        Some(axis) => (array, axis),
        None => (array::ravel(runtime, &array)?, 0),
    };
    let dtype = args::optional_dtype(runtime, bound.get("dtype"))?;
    let out = out_array(runtime, bound.get("out"))?;
    accumulate_call(runtime, ufunc, array, axis, dtype, out)
}

fn module_cumsum(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = CUMSUM_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    cumulative(runtime, "add", array, &bound)
}

fn method_cumsum(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = CUMSUM_METHOD_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, receiver)?;
    cumulative(runtime, "add", array, &bound)
}

fn module_cumprod(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = CUMPROD_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    cumulative(runtime, "multiply", array, &bound)
}

fn method_cumprod(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = CUMPROD_METHOD_SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, receiver)?;
    cumulative(runtime, "multiply", array, &bound)
}
