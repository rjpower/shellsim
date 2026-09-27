//! Reductions and accumulations along axes: `ufunc.reduce`, `ufunc.accumulate`, `argmin` and
//! `argmax`, and the `sum`/`prod`/`max`/`min`/`any`/`all`/`cumsum`/`cumprod` functions and
//! methods built on them. The statistics NumPy writes in Python (`mean`, `var`, `std`,
//! `average`, the `nan*` functions) live in `numpy/_methods.py` and `numpy/_nanfunctions.py` on
//! top of these, as in NumPy.
//!
//! A reduction visits its input the way NumPy's iterator does, so floating-point results match
//! NumPy bit for bit in the common cases:
//!
//! - The result starts from the ufunc's identity (`0` for `add`) or from `initial=`. A ufunc
//!   without an identity, and every `object` reduction, starts from the first element instead.
//! - Axes are ordered by stride magnitude and adjacent axes that step evenly merge, as in
//!   NumPy's iterator; negative strides are kept, so a reversed view is visited in reverse.
//! - [`runs`] reproduces how NumPy's buffered iterator splits each result's elements into
//!   inner-loop calls. `add` sums each call's run pairwise ([`Numeric::add_lane`]) and adds that
//!   to the running result. When the innermost axis is kept, the fold goes element by element.
//!   An input that needs a cast, or that cannot be walked with one stride, is copied through
//!   NumPy's 8192-element buffer, and each buffer is one run.
//!
//! `add` and `multiply` accumulate booleans and small integers in the 64-bit integer of their
//! signedness, so `np.sum` of `int8` values does not wrap. Integer reductions otherwise wrap
//! silently; floating-point flags are reported under the names `reduce` and `accumulate`.

use std::cmp::Ordering;

use super::super::super::ast::ComparisonOperator;
use super::super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyArrayBuffer, PyArrayData,
    PyArrayDtype, PyArrayView, PyError, PyKind, PyOperator, PyResult, PyRuntime, PyValue,
    PyValueCast,
};
use super::super::super::Value;
use super::args::{self, Axes, Bound, Signature};
use super::array::{
    self, array_from_elements, contiguous_strides, element_count, new_array, reserve_elements,
    Array, Offsets,
};
use super::convert;
use super::dtype::{self, DType, Kind};
use super::element::{self, dispatch_integer, dispatch_numeric, dispatch_real, Number};
use super::ops::{FpFlags, Integer, Numeric, Real};
use super::ufunc::{self, ArithOp, BitOp, Family, LogicalOp, UfuncDef, UFUNCS};

/// NumPy's iterator buffer size, `NPY_BUFSIZE`, in elements.
const BUFFER_SIZE: usize = 8192;

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
    function("max", module_max),
    function("min", module_min),
    function("amax", module_max),
    function("amin", module_min),
    function("any", module_any),
    function("all", module_all),
    function("cumsum", module_cumsum),
    function("cumprod", module_cumprod),
    function("argmax", module_argmax),
    function("argmin", module_argmin),
];

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

/// Methods this area installs on `numpy.ndarray`.
pub(in crate::python) static ARRAY_METHODS: NativeTypeDef = NativeTypeDef {
    name: "numpy.ndarray",
    methods: &[
        method("sum", method_sum),
        method("prod", method_prod),
        method("max", method_max),
        method("min", method_min),
        method("any", method_any),
        method("all", method_all),
        method("cumsum", method_cumsum),
        method("cumprod", method_cumprod),
        method("argmax", method_argmax),
        method("argmin", method_argmin),
        method("mean", method_mean),
        method("var", method_var),
        method("std", method_std),
    ],
    getters: &[],
};

/// Options of one reduction.
pub(in crate::python) struct ReduceOptions {
    pub axes: Axes,
    pub dtype: Option<DType>,
    pub out: Option<Array>,
    pub keepdims: bool,
    pub initial: Option<PyValue>,
}

/// Reduce `array` with binary ufunc `index`.
pub(in crate::python) fn reduce(
    runtime: &mut dyn PyRuntime,
    index: usize,
    array: &Array,
    options: &ReduceOptions,
) -> PyResult {
    let ufunc = &UFUNCS[index];
    let requested = options.dtype.or(options.out.as_ref().map(|out| out.dtype));
    let dtype = loop_dtype(ufunc, array.dtype, requested)?;
    let shape = result_shape(array.shape(), &options.axes, options.keepdims);
    if let Some(out) = &options.out {
        check_out(ufunc.name, out, &shape)?;
    }
    let source = convert::cast_array(runtime, array, dtype, false)?;
    let layout = Layout::new(&source.view, &options.axes, array.dtype != dtype);
    let buffer = match dtype.kind() {
        Kind::Object => reduce_objects(runtime, ufunc, &source, &layout, options.initial)?,
        Kind::Str => {
            return Err(PyError::unsupported(format!(
                "{}.reduce on string arrays is not supported",
                ufunc.name
            )))
        }
        _ => reduce_numbers(runtime, ufunc, &source, &layout, options.initial)?,
    };
    let result = new_array(runtime, buffer, dtype, shape)?;
    finish(runtime, result, options.out.as_ref())
}

/// The dtype a reduction computes in: `dtype=` (or `out=`'s dtype) when given, the 64-bit
/// accumulator for `add` and `multiply` of booleans and integers, and otherwise the input's
/// dtype as the ufunc resolves it. The loop must map its dtype to itself. Object logical
/// reductions keep objects, returning an operand as Python's `and`/`or` do.
fn loop_dtype(ufunc: &UfuncDef, input: DType, requested: Option<DType>) -> PyResult<DType> {
    if ufunc.nin() != 2 {
        return Err(PyError::value_error(
            "reduce only supported for binary functions",
        ));
    }
    let accumulates = matches!(
        ufunc.family,
        Family::Arith {
            op: ArithOp::Add | ArithOp::Multiply,
            ..
        }
    );
    let common = match requested {
        Some(dtype) => dtype,
        None if accumulates && input.kind() != Kind::Object => dtype::accumulator(input),
        None => input,
    };
    if common.kind() == Kind::Object && matches!(ufunc.family, Family::Logical(_)) {
        return Ok(common);
    }
    let resolved = ufunc::resolve(ufunc, common, &[common, common], requested)?;
    if resolved.input != resolved.output {
        return Err(PyError::type_error(format!(
            "No loop matching the specified signature and casting was found for ufunc {}",
            ufunc.name
        )));
    }
    Ok(resolved.input)
}

/// The shape left after reducing `axes`, with length-1 axes in their place for `keepdims`.
fn result_shape(shape: &[usize], axes: &Axes, keepdims: bool) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .filter_map(|(axis, length)| match (axes.contains(axis), keepdims) {
            (false, _) => Some(*length),
            (true, true) => Some(1),
            (true, false) => None,
        })
        .collect()
}

fn check_out(name: &str, out: &Array, shape: &[usize]) -> PyResult<()> {
    if out.ndim() != shape.len() {
        return Err(PyError::value_error(format!(
            "output parameter for reduction operation {name} has the wrong number of dimensions: \
             Found {} but expected {}",
            out.ndim(),
            shape.len()
        )));
    }
    if out.shape() != shape {
        return Err(PyError::value_error(format!(
            "output parameter for reduction operation {name} has a non-reduction dimension not \
             equal to the output"
        )));
    }
    Ok(())
}

/// Return a result: stored into `out=`, boxed as a scalar when 0-d, or the array itself.
fn finish(runtime: &mut dyn PyRuntime, result: Array, out: Option<&Array>) -> PyResult {
    if let Some(out) = out {
        array::assign(runtime, out, &result)?;
        return Ok(out.value());
    }
    if result.ndim() == 0 {
        return convert::element_to_scalar(runtime, &result, result.view.offset);
    }
    Ok(result.value())
}

/// The ufunc's identity, the starting value of a numeric reduction without `initial=`.
fn identity(family: Family) -> Option<Number> {
    match family {
        Family::Arith {
            op: ArithOp::Add, ..
        }
        | Family::Logical(LogicalOp::Or | LogicalOp::Xor)
        | Family::Bitwise(BitOp::Or | BitOp::Xor) => Some(Number::Int(0)),
        Family::Arith {
            op: ArithOp::Multiply,
            ..
        } => Some(Number::Int(1)),
        Family::Logical(LogicalOp::And) => Some(Number::Bool(true)),
        Family::Bitwise(BitOp::And) => Some(Number::Int(-1)),
        _ => None,
    }
}

fn no_identity(name: &str) -> PyError {
    PyError::value_error(format!(
        "zero-size array to reduction operation {name} which has no identity"
    ))
}

/// How a reduction visits its input.
struct Layout {
    /// Shape and input strides of the kept axes, in order; results are stored in C order.
    kept_shape: Vec<usize>,
    kept_strides: Vec<isize>,
    /// Reduced axes in NumPy's visiting order, outer to inner.
    reduced_shape: Vec<usize>,
    reduced_strides: Vec<isize>,
    /// Byte offset of the first element.
    start: usize,
    /// NumPy's inner-loop calls: each result's elements, in visiting order, fall into blocks of
    /// `block` elements, and each block into runs of at most `run`. `add` sums a run pairwise.
    runs: Runs,
    /// Elements folded into each result.
    count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Runs {
    block: usize,
    run: usize,
}

/// One axis of NumPy's reduction iterator over (result, input).
#[derive(Clone, Copy, Debug)]
struct IterAxis {
    length: usize,
    /// Input stride in bytes.
    stride: isize,
    reduced: bool,
}

impl Layout {
    fn new(view: &PyArrayView, axes: &Axes, cast: bool) -> Self {
        let (mut kept_shape, mut kept_strides) = (Vec::new(), Vec::new());
        let mut iteration = Vec::new();
        let mut count = 1usize;
        for (axis, (&length, &stride)) in view.shape.iter().zip(&view.strides).enumerate() {
            let reduced = axes.contains(axis);
            if reduced {
                count = count.saturating_mul(length);
            } else {
                kept_shape.push(length);
                kept_strides.push(stride);
            }
            if length > 1 {
                iteration.push(IterAxis {
                    length,
                    stride,
                    reduced,
                });
            }
        }
        // NumPy stores iterator axes innermost first.
        iteration.reverse();
        let iteration = coalesce(order_axes(iteration));
        let (reduced_shape, reduced_strides) = if count == 0 {
            // The iterator skips empty axes; keep one so a result visits no elements.
            (vec![0], vec![0])
        } else {
            iteration
                .iter()
                .rev()
                .filter(|axis| axis.reduced)
                .map(|axis| (axis.length, axis.stride))
                .unzip()
        };
        Self {
            kept_shape,
            kept_strides,
            reduced_shape,
            reduced_strides,
            start: view.offset,
            runs: runs(&iteration, cast),
            count,
        }
    }

    fn results(&self) -> usize {
        self.kept_shape.iter().product()
    }

    /// Offsets of each result's first element, in result order.
    fn bases(&self) -> Offsets {
        Offsets::new(&self.kept_shape, &self.kept_strides, self.start)
    }

    /// Offsets of the elements of the result at `base`, in visiting order.
    fn elements(&self, base: usize) -> Offsets {
        Offsets::new(&self.reduced_shape, &self.reduced_strides, base)
    }
}

/// NumPy's iterator axis order (`npyiter_find_best_axis_ordering`), innermost first: a stable
/// insertion sort by stride magnitude. A zero stride is ambiguous, so the scan steps past it.
/// Reductions keep negative strides, so a reversed axis is still visited in reverse.
fn order_axes(mut axes: Vec<IterAxis>) -> Vec<IterAxis> {
    for next in 1..axes.len() {
        let stride = axes[next].stride;
        let mut position = next;
        for earlier in (0..next).rev() {
            let other = axes[earlier].stride;
            if stride == 0 || other == 0 {
                continue;
            }
            if other.unsigned_abs() <= stride.unsigned_abs() {
                break;
            }
            position = earlier;
        }
        axes[position..=next].rotate_right(1);
    }
    axes
}

/// Merge adjacent axes that both operands step through evenly. The result is allocated in
/// iteration order, so it merges two kept or two reduced axes; the input must also agree.
fn coalesce(axes: Vec<IterAxis>) -> Vec<IterAxis> {
    let mut merged: Vec<IterAxis> = Vec::with_capacity(axes.len());
    for axis in axes {
        match merged.last_mut() {
            Some(inner)
                if inner.reduced == axis.reduced
                    && inner.stride * inner.length as isize == axis.stride =>
            {
                inner.length *= axis.length;
            }
            _ => merged.push(axis),
        }
    }
    merged
}

/// The inner-loop runs of NumPy's buffered reduction iterator, following
/// `npyiter_find_buffering_setup` for its two operands, the result and the input. It picks
/// the outer dimension that minimizes estimated overhead. The input is buffered when it needs a
/// cast or when it cannot be walked with one stride up to that dimension, and a buffer holds at
/// most 8192 elements. In "reduce mode" the core below the first result/reduced boundary is
/// one inner loop; otherwise one buffer's worth of reduced elements is.
fn runs(axes: &[IterAxis], cast: bool) -> Runs {
    let sequential = Runs { block: 1, run: 1 };
    let Some(first) = axes.first() else {
        return sequential;
    };
    if !first.reduced {
        return sequential;
    }
    let mut cost = 1 + usize::from(cast);
    let (mut result_single, mut input_single) = (1, 1);
    let mut outer_reduce = 0;
    let mut size = first.length;
    let (mut best_dim, mut best_cost, mut best_size, mut best_core) = (0, cost, size, 1);
    for dim in 1..axes.len() {
        if outer_reduce != 0 || (size >= BUFFER_SIZE && cost > 1) {
            break;
        }
        let (previous, current) = (axes[dim - 1], axes[dim]);
        if result_single == dim {
            if previous.reduced == current.reduced {
                result_single += 1;
            } else {
                cost += 1;
                outer_reduce = dim;
            }
        }
        if input_single == dim {
            if previous.stride * previous.length as isize == current.stride {
                input_single += 1;
            } else if !cast {
                cost += 1;
            }
        }
        let core = size;
        size = size.saturating_mul(current.length);
        let buffered_size = if size > BUFFER_SIZE && cost > 1 {
            BUFFER_SIZE
        } else {
            size
        };
        if (cost as u128) * (best_size as u128) <= (best_cost as u128) * (buffered_size as u128) {
            (best_dim, best_cost, best_size, best_core) = (dim, cost, size, core);
        }
    }
    let reduce_mode = outer_reduce != 0 && best_dim == outer_reduce;
    if reduce_mode {
        return Runs {
            block: best_core,
            run: best_core,
        };
    }
    let buffered = cast || input_single <= best_dim;
    let run = if buffered && best_size > BUFFER_SIZE {
        best_core * (BUFFER_SIZE / best_core)
    } else {
        best_size
    };
    Runs {
        block: best_size,
        run,
    }
}

/// Run `$body` with `$T` bound to the element type of `$kind` and `$operation` to the ufunc's
/// binary kernel for it.
macro_rules! with_kernel {
    ($family:expr, $kind:expr, |$T:ident, $operation:ident| $body:expr) => {{
        let unsupported = || -> PyResult<()> {
            Err(PyError::runtime_error(
                "reduction has no kernel for its dtype",
            ))
        };
        match $family {
            Family::Arith { op, .. } => dispatch_numeric!($kind, $T => {
                let $operation = ufunc::arith_fn::<$T>(op);
                $body
            }, _ => unsupported()),
            Family::TrueDivide => dispatch_numeric!($kind, $T => {
                let $operation = |a: $T, b: $T, flags: &mut FpFlags| a.divide(b, flags);
                $body
            }, _ => unsupported()),
            Family::Logical(op) => {
                type $T = bool;
                let $operation = move |a: bool, b: bool, _: &mut FpFlags| match op {
                    LogicalOp::And => a && b,
                    LogicalOp::Or => a || b,
                    LogicalOp::Xor => a != b,
                };
                $body
            }
            Family::Compare(op) => {
                type $T = bool;
                let test = ufunc::compare_fn(op);
                let $operation = move |a: bool, b: bool, _: &mut FpFlags| test(a.compare(b));
                $body
            }
            Family::Bitwise(op) => dispatch_integer!($kind, $T => {
                let $operation = move |a: $T, b: $T, _: &mut FpFlags| match op {
                    BitOp::And => a.bit_and(b),
                    BitOp::Or => a.bit_or(b),
                    BitOp::Xor => a.bit_xor(b),
                    BitOp::LeftShift => a.left_shift(b),
                    BitOp::RightShift => a.right_shift(b),
                };
                $body
            }, _ => unsupported()),
            Family::Float2(op) => dispatch_real!($kind, $T => {
                let pair = ufunc::float2_fn(op);
                let $operation = move |a: $T, b: $T, flags: &mut FpFlags| {
                    let result = a.zip(b, pair.0, pair.1);
                    ufunc::float_flags_binary(a.to_f64(), b.to_f64(), result.to_f64(), flags);
                    result
                };
                $body
            }, _ => unsupported()),
            _ => unsupported(),
        }
    }};
}

fn reduce_numbers(
    runtime: &mut dyn PyRuntime,
    ufunc: &UfuncDef,
    source: &Array,
    layout: &Layout,
    initial: Option<PyValue>,
) -> PyResult<PyArrayBuffer> {
    let dtype = source.dtype;
    let mut start = [0u8; 16];
    let has_start = match initial {
        Some(value) => {
            let PyArrayBuffer::Bytes(bytes) = convert::value_to_buffer(runtime, value, dtype)?
            else {
                return Err(PyError::runtime_error(
                    "numeric initial value has object storage",
                ));
            };
            start[..bytes.len()].copy_from_slice(&bytes);
            true
        }
        None => match identity(ufunc.family) {
            Some(number) => {
                element::write_number(dtype.kind(), number, &mut start);
                true
            }
            None => false,
        },
    };
    if layout.count == 0 && !has_start {
        return Err(no_identity(ufunc.name));
    }
    let results = layout.results();
    runtime.charge_cpu((layout.count as u64).saturating_mul(results as u64) + 1)?;
    reserve_elements(runtime, dtype, results)?;
    // Scratch space for one run.
    runtime.reserve_memory(
        layout
            .runs
            .run
            .min(layout.count)
            .saturating_mul(dtype.itemsize()),
    )?;
    let mut output = vec![0u8; results * dtype.itemsize()];
    let mut flags = FpFlags::default();
    let fold_options = FoldOptions {
        start: has_start.then_some(&start[..]),
        pairwise: matches!(
            ufunc.family,
            Family::Arith {
                op: ArithOp::Add,
                ..
            }
        ),
    };
    let family = ufunc.family;
    runtime.read_arrays(&[source.handle], &mut |arrays| {
        let PyArrayData::Bytes(data) = arrays[0].data else {
            return Err(PyError::runtime_error("numeric array has object storage"));
        };
        with_kernel!(family, dtype.kind(), |T, operation| {
            fold::<T>(
                data,
                layout,
                &fold_options,
                &mut output,
                &mut flags,
                operation,
            )
        })
    })?;
    if dtype.is_integer() {
        flags.overflow = false;
    }
    super::errstate::report(runtime, "reduce", flags)?;
    Ok(PyArrayBuffer::Bytes(output))
}

struct FoldOptions<'a> {
    /// Bytes of the starting value, or `None` to start from the first element.
    start: Option<&'a [u8]>,
    /// Sum each run pairwise, as NumPy's `add` does.
    pairwise: bool,
}

/// Fold every result of a numeric reduction.
fn fold<T: Numeric>(
    data: &[u8],
    layout: &Layout,
    options: &FoldOptions<'_>,
    output: &mut [u8],
    flags: &mut FpFlags,
    operation: impl Fn(T, T, &mut FpFlags) -> T,
) -> PyResult<()> {
    let start = options.start.map(T::read);
    let read = |offset: usize| T::read(&data[offset..]);
    let Runs { block, run } = layout.runs;
    let mut lane = Vec::new();
    for (base, target) in layout.bases().zip(output.chunks_exact_mut(T::SIZE)) {
        let mut total = start;
        let mut elements = layout.elements(base);
        match total {
            Some(mut sum) if options.pairwise && run > 1 => {
                let mut left = layout.count;
                while left > 0 {
                    let mut block_left = block.min(left);
                    left -= block_left;
                    while block_left > 0 {
                        let length = run.min(block_left);
                        block_left -= length;
                        lane.clear();
                        lane.extend(elements.by_ref().take(length).map(read));
                        sum = sum.add_lane(&lane, flags);
                    }
                }
                total = Some(sum);
            }
            _ => {
                for offset in elements {
                    let value = read(offset);
                    total = Some(match total {
                        Some(sum) => operation(sum, value, flags),
                        None => value,
                    });
                }
            }
        }
        total
            .ok_or_else(|| PyError::runtime_error("empty reduction without a start value"))?
            .write(target);
    }
    Ok(())
}

/// The elements of each result of `layout`, in visiting order, from an object array.
fn object_groups(
    runtime: &mut dyn PyRuntime,
    source: &Array,
    layout: &Layout,
) -> PyResult<Vec<Vec<PyValue>>> {
    let results = layout.results();
    runtime.reserve_memory(source.size().saturating_mul(PyArrayDtype::VALUE_ITEMSIZE))?;
    let mut groups = Vec::with_capacity(results);
    runtime.read_arrays(&[source.handle], &mut |arrays| {
        let PyArrayData::Values(values) = arrays[0].data else {
            return Err(PyError::runtime_error("object array has byte storage"));
        };
        for base in layout.bases() {
            groups.push(
                layout
                    .elements(base)
                    .map(|offset| values[offset / PyArrayDtype::VALUE_ITEMSIZE])
                    .collect(),
            );
        }
        Ok(())
    })?;
    Ok(groups)
}

/// An `object` reduction applies the Python operator element by element, starting from the
/// first element. The identity is used only when there are no elements.
fn reduce_objects(
    runtime: &mut dyn PyRuntime,
    ufunc: &UfuncDef,
    source: &Array,
    layout: &Layout,
    initial: Option<PyValue>,
) -> PyResult<PyArrayBuffer> {
    let groups = object_groups(runtime, source, layout)?;
    let mut output = Vec::with_capacity(groups.len());
    for group in groups {
        let mut items = group.into_iter();
        let mut total = match initial.or_else(|| items.next()) {
            Some(value) => value,
            None => match identity(ufunc.family) {
                Some(Number::Bool(value)) => Value::Bool(value),
                Some(number) => Value::Int(number.wrapping_i64()),
                None => return Err(no_identity(ufunc.name)),
            },
        };
        for item in items {
            runtime.charge_cpu(1)?;
            total = object_operation(runtime, ufunc, total, item)?;
        }
        output.push(total);
    }
    Ok(PyArrayBuffer::Values(output))
}

/// One step of an `object` reduction or accumulation. Logical ufuncs return an operand, as
/// Python's `and` and `or` do.
fn object_operation(
    runtime: &mut dyn PyRuntime,
    ufunc: &UfuncDef,
    left: PyValue,
    right: PyValue,
) -> PyResult {
    match ufunc.family {
        Family::Logical(LogicalOp::And) => Ok(if runtime.truth(&left)? { right } else { left }),
        Family::Logical(LogicalOp::Or) => Ok(if runtime.truth(&left)? { left } else { right }),
        _ => ufunc::object_element(runtime, ufunc, &[left, right]),
    }
}

/// Accumulate `array` along `axis` with binary ufunc `index`: element `i` of the result
/// combines elements `0..=i`.
pub(in crate::python) fn accumulate(
    runtime: &mut dyn PyRuntime,
    index: usize,
    array: &Array,
    axis: usize,
    requested: Option<DType>,
    out: Option<&Array>,
) -> PyResult {
    let ufunc = &UFUNCS[index];
    let dtype = loop_dtype(ufunc, array.dtype, requested.or(out.map(|out| out.dtype)))?;
    if let Some(out) = out {
        if out.shape() != array.shape() {
            return Err(PyError::value_error(format!(
                "output operand with shape {} does not match the accumulation shape {}",
                array::format_shape(out.shape()),
                array::format_shape(array.shape())
            )));
        }
    }
    let source = convert::cast_array(runtime, array, dtype, false)?;
    let shape = source.shape().to_vec();
    let count = element_count(&shape)?;
    let (length, stride) = (shape[axis], source.strides()[axis]);
    let mut lane_shape = shape.clone();
    lane_shape.remove(axis);
    let mut lane_strides = source.strides().to_vec();
    lane_strides.remove(axis);
    let mut output_strides = contiguous_strides(&shape, dtype.itemsize().max(1));
    let output_stride = output_strides.remove(axis);
    let scan = Scan {
        inputs: Offsets::new(&lane_shape, &lane_strides, source.view.offset),
        outputs: Offsets::new(&lane_shape, &output_strides, 0),
        length,
        stride,
        output_stride,
    };
    runtime.charge_cpu(count as u64 + 1)?;
    let buffer = match dtype.kind() {
        Kind::Str => {
            return Err(PyError::unsupported(format!(
                "{}.accumulate on string arrays is not supported",
                ufunc.name
            )))
        }
        Kind::Object => accumulate_objects(runtime, ufunc, &source, scan, count)?,
        kind => {
            reserve_elements(runtime, dtype, count)?;
            let mut output = vec![0u8; count * dtype.itemsize()];
            let mut flags = FpFlags::default();
            let family = ufunc.family;
            let mut scan = Some(scan);
            runtime.read_arrays(&[source.handle], &mut |arrays| {
                let PyArrayData::Bytes(data) = arrays[0].data else {
                    return Err(PyError::runtime_error("numeric array has object storage"));
                };
                let scan = scan.take().expect("read callback runs once");
                with_kernel!(family, kind, |T, operation| {
                    scan_numbers::<T>(data, scan, &mut output, &mut flags, operation)
                })
            })?;
            if dtype.is_integer() {
                flags.overflow = false;
            }
            super::errstate::report(runtime, "accumulate", flags)?;
            PyArrayBuffer::Bytes(output)
        }
    };
    let result = new_array(runtime, buffer, dtype, shape)?;
    if let Some(out) = out {
        array::assign(runtime, out, &result)?;
        return Ok(out.value());
    }
    Ok(result.value())
}

/// Where an accumulation reads and writes: one lane per position of the other axes.
struct Scan {
    inputs: Offsets,
    outputs: Offsets,
    length: usize,
    stride: isize,
    output_stride: isize,
}

fn scan_numbers<T: Numeric>(
    data: &[u8],
    scan: Scan,
    output: &mut [u8],
    flags: &mut FpFlags,
    operation: impl Fn(T, T, &mut FpFlags) -> T,
) -> PyResult<()> {
    for (input, target) in scan.inputs.zip(scan.outputs) {
        let mut total: Option<T> = None;
        for step in 0..scan.length {
            let offset = (input as isize + step as isize * scan.stride) as usize;
            let value = T::read(&data[offset..]);
            let next = match total {
                Some(sum) => operation(sum, value, flags),
                None => value,
            };
            let position = (target as isize + step as isize * scan.output_stride) as usize;
            next.write(&mut output[position..position + T::SIZE]);
            total = Some(next);
        }
    }
    Ok(())
}

fn accumulate_objects(
    runtime: &mut dyn PyRuntime,
    ufunc: &UfuncDef,
    source: &Array,
    scan: Scan,
    count: usize,
) -> PyResult<PyArrayBuffer> {
    runtime.reserve_memory(count.saturating_mul(PyArrayDtype::VALUE_ITEMSIZE))?;
    // Snapshot every lane first: the Python operators below may run arbitrary code.
    let Scan {
        inputs,
        outputs,
        length,
        stride,
        output_stride,
    } = scan;
    let mut lanes = Some((inputs, outputs));
    let mut snapshot = Vec::new();
    runtime.read_arrays(&[source.handle], &mut |arrays| {
        let PyArrayData::Values(values) = arrays[0].data else {
            return Err(PyError::runtime_error("object array has byte storage"));
        };
        let (inputs, outputs) = lanes.take().expect("read callback runs once");
        for (input, target) in inputs.zip(outputs) {
            let items = (0..length)
                .map(|step| {
                    let offset = (input as isize + step as isize * stride) as usize;
                    values[offset / PyArrayDtype::VALUE_ITEMSIZE]
                })
                .collect::<Vec<_>>();
            snapshot.push((target, items));
        }
        Ok(())
    })?;
    let mut output = vec![Value::None; count];
    for (target, items) in snapshot {
        let mut total = None;
        for (step, item) in items.into_iter().enumerate() {
            runtime.charge_cpu(1)?;
            let next = match total {
                Some(sum) => object_operation(runtime, ufunc, sum, item)?,
                None => item,
            };
            let position = (target as isize + step as isize * output_stride) as usize;
            output[position / PyArrayDtype::VALUE_ITEMSIZE] = next;
            total = Some(next);
        }
    }
    Ok(PyArrayBuffer::Values(output))
}

/// Index of the first maximum (or minimum) along each lane. NaN counts as the extreme, so the
/// first NaN wins, as in NumPy.
fn arg_numbers<T: Numeric>(data: &[u8], scan: Scan, maximum: bool, output: &mut Vec<i64>) {
    let wanted = if maximum {
        Ordering::Greater
    } else {
        Ordering::Less
    };
    for input in scan.inputs {
        let read =
            |step: usize| T::read(&data[(input as isize + step as isize * scan.stride) as usize..]);
        let mut best = read(0);
        let mut index = 0;
        if !best.is_nan() {
            for step in 1..scan.length {
                let value = read(step);
                if value.is_nan() {
                    index = step;
                    break;
                }
                if value.compare(best) == Some(wanted) {
                    best = value;
                    index = step;
                }
            }
        }
        output.push(index as i64);
    }
}

/// `np.argmax`/`np.argmin`: positions in C order for `axis=None`, else along `axis`.
fn arg_extreme(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    bound: &Bound,
    maximum: bool,
) -> PyResult {
    let name = if maximum { "argmax" } else { "argmin" };
    let array = convert::as_array(runtime, value)?;
    let axis = args::axis(runtime, bound.get("axis"), array.ndim())?;
    let keepdims = args::flag(runtime, bound.get("keepdims"), false)?;
    let out = out_argument(runtime, bound.value("out"))?;
    let (source, lane_axis) = match axis {
        Some(axis) => (array.clone(), axis),
        None => (array::ravel(runtime, &array)?, 0),
    };
    let length = source.shape()[lane_axis];
    if length == 0 {
        return Err(PyError::value_error(format!(
            "attempt to get {name} of an empty sequence"
        )));
    }
    let mut lane_shape = source.shape().to_vec();
    lane_shape.remove(lane_axis);
    let mut lane_strides = source.strides().to_vec();
    lane_strides.remove(lane_axis);
    let results = element_count(&lane_shape)?;
    runtime.charge_cpu(source.size() as u64 + 1)?;
    runtime.reserve_memory(results.saturating_mul(8))?;
    let scan = Scan {
        inputs: Offsets::new(&lane_shape, &lane_strides, source.view.offset),
        outputs: Offsets::new(&[], &[], 0),
        length,
        stride: source.strides()[lane_axis],
        output_stride: 0,
    };
    let indices = match source.dtype.kind() {
        Kind::Object | Kind::Str => arg_values(runtime, &source, lane_axis, maximum)?,
        kind => {
            let mut indices = Vec::with_capacity(results);
            let mut scan = Some(scan);
            runtime.read_arrays(&[source.handle], &mut |arrays| {
                let PyArrayData::Bytes(data) = arrays[0].data else {
                    return Err(PyError::runtime_error("numeric array has object storage"));
                };
                let scan = scan.take().expect("read callback runs once");
                dispatch_numeric!(kind, T => arg_numbers::<T>(data, scan, maximum, &mut indices), _ => {});
                Ok(())
            })?;
            indices
        }
    };
    let shape = match (keepdims, axis) {
        (false, _) => lane_shape,
        (true, None) => vec![1; array.ndim()],
        (true, Some(axis)) => {
            let mut shape = array.shape().to_vec();
            shape[axis] = 1;
            shape
        }
    };
    let result = array_from_elements(runtime, DType::INT64, shape, &indices)?;
    finish(runtime, result, out.as_ref())
}

/// `argmax`/`argmin` over strings (code point order) and objects (Python `>` and `<`).
fn arg_values(
    runtime: &mut dyn PyRuntime,
    source: &Array,
    lane_axis: usize,
    maximum: bool,
) -> PyResult<Vec<i64>> {
    let objects = convert::cast_array(runtime, source, DType::OBJECT, false)?;
    let values = array::read_objects(runtime, &objects)?;
    // `values` is in C order, so walk it with C-order strides counted in elements.
    let mut lane_shape = source.shape().to_vec();
    let length = lane_shape.remove(lane_axis);
    let mut lane_strides = contiguous_strides(source.shape(), 1);
    let step = lane_strides.remove(lane_axis);
    let operator = PyOperator::Compare(if maximum {
        ComparisonOperator::Greater
    } else {
        ComparisonOperator::Less
    });
    let mut indices = Vec::new();
    for first in Offsets::new(&lane_shape, &lane_strides, 0) {
        let mut best = values[first];
        let mut index = 0;
        for position in 1..length {
            runtime.charge_cpu(1)?;
            let value = values[(first as isize + position as isize * step) as usize];
            let better = runtime.apply_operator(operator, &[value, best])?;
            if runtime.truth(&better)? {
                best = value;
                index = position;
            }
        }
        indices.push(index as i64);
    }
    Ok(indices)
}

/// Accept `out=array` or `out=None`.
fn out_argument(runtime: &mut dyn PyRuntime, value: Option<PyValue>) -> PyResult<Option<Array>> {
    value
        .map(|value| {
            Array::from_value(runtime, value)
                .map_err(|_| PyError::type_error("output must be an array"))
        })
        .transpose()
}

/// Reject `where=` masks other than `True`, which reductions do not implement.
fn reject_where(runtime: &mut dyn PyRuntime, bound: &Bound, name: &str) -> PyResult<()> {
    match bound.get("where") {
        Some(value) if !(runtime.kind(&value)? == PyKind::Bool && runtime.truth(&value)?) => Err(
            PyError::unsupported(format!("{name}() with a where= mask is not supported")),
        ),
        _ => Ok(()),
    }
}

/// Bind `args` for a method call, with the receiver as the first positional argument.
fn with_receiver(receiver: PyValue, args: &CallArgs) -> CallArgs {
    let mut positional = vec![receiver];
    positional.extend_from_slice(args.positional());
    CallArgs::new(positional, args.keywords().to_vec())
}

static SUM: Signature = Signature::new(
    "sum",
    &["a", "axis", "dtype", "out", "keepdims", "initial", "where"],
    1,
);
static PROD: Signature = Signature::new(
    "prod",
    &["a", "axis", "dtype", "out", "keepdims", "initial", "where"],
    1,
);
static MAX: Signature = Signature::new(
    "max",
    &["a", "axis", "out", "keepdims", "initial", "where"],
    1,
);
static MIN: Signature = Signature::new(
    "min",
    &["a", "axis", "out", "keepdims", "initial", "where"],
    1,
);
static ANY: Signature =
    Signature::new("any", &["a", "axis", "out", "keepdims"], 1).keyword_only(&["where"]);
static ALL: Signature =
    Signature::new("all", &["a", "axis", "out", "keepdims"], 1).keyword_only(&["where"]);
static CUMSUM: Signature = Signature::new("cumsum", &["a", "axis", "dtype", "out"], 1);
static CUMPROD: Signature = Signature::new("cumprod", &["a", "axis", "dtype", "out"], 1);
static ARGMAX: Signature =
    Signature::new("argmax", &["a", "axis", "out"], 1).keyword_only(&["keepdims"]);
static ARGMIN: Signature =
    Signature::new("argmin", &["a", "axis", "out"], 1).keyword_only(&["keepdims"]);

/// A whole-array reduction such as `np.sum`, bound to `signature`. `dtype` fixes the loop
/// dtype for `any` and `all`, which always reduce booleans.
fn reduction(
    runtime: &mut dyn PyRuntime,
    signature: &'static Signature,
    ufunc: &str,
    dtype: Option<DType>,
    args: &CallArgs,
) -> PyResult {
    let bound = signature.bind(args)?;
    reject_where(runtime, &bound, bound.function())?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let requested = match dtype {
        Some(dtype) => Some(dtype),
        None if signature.has("dtype") => args::optional_dtype(runtime, bound.get("dtype"))?,
        None => None,
    };
    let options = ReduceOptions {
        axes: args::axes(runtime, bound.get("axis"), array.ndim())?,
        dtype: requested,
        out: out_argument(runtime, bound.value("out"))?,
        keepdims: args::flag(runtime, bound.value("keepdims"), false)?,
        initial: if signature.has("initial") {
            bound.value("initial")
        } else {
            None
        },
    };
    reduce(runtime, ufunc::named(ufunc), &array, &options)
}

/// `np.cumsum`/`np.cumprod`: `axis=None` accumulates the flattened array.
fn cumulative(
    runtime: &mut dyn PyRuntime,
    signature: &'static Signature,
    ufunc: &str,
    args: &CallArgs,
) -> PyResult {
    let bound = signature.bind(args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let (array, axis) = match args::axis(runtime, bound.get("axis"), array.ndim())? {
        Some(axis) => (array, axis),
        None => (array::ravel(runtime, &array)?, 0),
    };
    let dtype = args::optional_dtype(runtime, bound.get("dtype"))?;
    let out = out_argument(runtime, bound.value("out"))?;
    accumulate(
        runtime,
        ufunc::named(ufunc),
        &array,
        axis,
        dtype,
        out.as_ref(),
    )
}

fn module_sum(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    reduction(runtime, &SUM, "add", None, &args)
}

fn module_prod(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    reduction(runtime, &PROD, "multiply", None, &args)
}

fn module_max(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    reduction(runtime, &MAX, "maximum", None, &args)
}

fn module_min(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    reduction(runtime, &MIN, "minimum", None, &args)
}

fn module_any(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    reduction(runtime, &ANY, "logical_or", Some(DType::BOOL), &args)
}

fn module_all(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    reduction(runtime, &ALL, "logical_and", Some(DType::BOOL), &args)
}

fn module_cumsum(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    cumulative(runtime, &CUMSUM, "add", &args)
}

fn module_cumprod(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    cumulative(runtime, &CUMPROD, "multiply", &args)
}

fn module_argmax(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ARGMAX.bind(&args)?;
    arg_extreme(runtime, bound.required("a"), &bound, true)
}

fn module_argmin(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ARGMIN.bind(&args)?;
    arg_extreme(runtime, bound.required("a"), &bound, false)
}

fn method_sum(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    module_sum(runtime, with_receiver(receiver, &args))
}

fn method_prod(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    module_prod(runtime, with_receiver(receiver, &args))
}

fn method_max(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    module_max(runtime, with_receiver(receiver, &args))
}

fn method_min(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    module_min(runtime, with_receiver(receiver, &args))
}

fn method_any(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    module_any(runtime, with_receiver(receiver, &args))
}

fn method_all(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    module_all(runtime, with_receiver(receiver, &args))
}

fn method_cumsum(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    module_cumsum(runtime, with_receiver(receiver, &args))
}

fn method_cumprod(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    module_cumprod(runtime, with_receiver(receiver, &args))
}

fn method_argmax(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    module_argmax(runtime, with_receiver(receiver, &args))
}

fn method_argmin(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    module_argmin(runtime, with_receiver(receiver, &args))
}

/// Call a function of the frozen `numpy._methods` module with the receiver first.
pub(in crate::python) fn python_method(
    runtime: &mut dyn PyRuntime,
    name: &str,
    receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    let module = runtime.import_module("numpy._methods")?;
    let function = runtime
        .get_attribute(module, name)?
        .ok_or_else(|| PyError::runtime_error(format!("numpy._methods.{name} is missing")))?;
    runtime.call_value(function, with_receiver(receiver, &args))
}

fn method_mean(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    python_method(runtime, "_mean", receiver, args)
}

fn method_var(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    python_method(runtime, "_var", receiver, args)
}

fn method_std(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    python_method(runtime, "_std", receiver, args)
}

static UFUNC_REDUCE: Signature = Signature::new(
    "reduce",
    &[
        "array", "axis", "dtype", "out", "keepdims", "initial", "where",
    ],
    1,
);
static UFUNC_ACCUMULATE: Signature =
    Signature::new("accumulate", &["array", "axis", "dtype", "out"], 1);

/// `ufunc.reduce(array, axis=0, dtype=None, out=None, keepdims=False, initial, where=True)`.
pub(in crate::python) fn ufunc_reduce(
    runtime: &mut dyn PyRuntime,
    index: usize,
    args: CallArgs,
) -> PyResult {
    let bound = UFUNC_REDUCE.bind(&args)?;
    reject_where(runtime, &bound, "reduce")?;
    let array = convert::as_array(runtime, bound.required("array"))?;
    let axes = match bound.get("axis") {
        None if array.ndim() == 0 => Axes::All,
        None => Axes::Some(vec![0]),
        axis => args::axes(runtime, axis, array.ndim())?,
    };
    let options = ReduceOptions {
        axes,
        dtype: args::optional_dtype(runtime, bound.get("dtype"))?,
        out: out_argument(runtime, bound.value("out"))?,
        keepdims: args::flag(runtime, bound.value("keepdims"), false)?,
        initial: bound.value("initial"),
    };
    reduce(runtime, index, &array, &options)
}

/// `ufunc.accumulate(array, axis=0, dtype=None, out=None)`.
pub(in crate::python) fn ufunc_accumulate(
    runtime: &mut dyn PyRuntime,
    index: usize,
    args: CallArgs,
) -> PyResult {
    let bound = UFUNC_ACCUMULATE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("array"))?;
    if array.ndim() == 0 {
        return Err(PyError::type_error("cannot accumulate on a scalar"));
    }
    let axis = match bound.get("axis") {
        None => 0,
        Some(value) if runtime.kind(&value)? == PyKind::Tuple => {
            let tuple = value.cast(runtime)?;
            match runtime.tuple_items(tuple)?.as_slice() {
                [single] => array::normalize_axis(args::index_int(runtime, single)?, array.ndim())?,
                _ => {
                    return Err(PyError::value_error(
                        "accumulate does not allow multiple axes",
                    ))
                }
            }
        }
        Some(value) if value.is_none() && array.ndim() == 1 => 0,
        Some(value) if value.is_none() => {
            return Err(PyError::value_error(
                "accumulate does not allow multiple axes",
            ))
        }
        Some(value) => array::normalize_axis(args::index_int(runtime, &value)?, array.ndim())?,
    };
    let dtype = args::optional_dtype(runtime, bound.get("dtype"))?;
    let out = out_argument(runtime, bound.value("out"))?;
    accumulate(runtime, index, &array, axis, dtype, out.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Iterator axes for a C-order `shape` of `f64` with the given element steps, innermost
    /// first, after NumPy's ordering and coalescing.
    fn axes(shape: &[usize], steps: &[isize], reduced: &[usize]) -> Vec<IterAxis> {
        let mut axes: Vec<_> = shape
            .iter()
            .zip(steps)
            .enumerate()
            .filter(|(_, (length, _))| **length > 1)
            .map(|(axis, (&length, &step))| IterAxis {
                length,
                stride: step * 8,
                reduced: reduced.contains(&axis),
            })
            .collect();
        axes.reverse();
        coalesce(order_axes(axes))
    }

    fn runs_of(shape: &[usize], steps: &[isize], reduced: &[usize], cast: bool) -> Runs {
        runs(&axes(shape, steps, reduced), cast)
    }

    // The expected runs reproduce the sums checked against NumPy 2.5.3 in
    // `test_float_sums_follow_numpy_iteration_order`.
    #[test]
    fn contiguous_reductions_use_one_run_per_result() {
        assert_eq!(
            runs_of(&[3000], &[1], &[0], false),
            Runs {
                block: 3000,
                run: 3000
            }
        );
        // Reducing the last axis: "reduce mode" makes each row one inner loop.
        assert_eq!(
            runs_of(&[30, 100], &[100, 1], &[1], false),
            Runs {
                block: 100,
                run: 100
            }
        );
        // Reducing the first axis adds one element per inner loop.
        assert_eq!(
            runs_of(&[30, 100], &[100, 1], &[0], false),
            Runs { block: 1, run: 1 }
        );
    }

    #[test]
    fn strided_blocks_are_buffered() {
        // `m[::-1].sum()`: the reversed rows do not coalesce, so the input is buffered whole.
        assert_eq!(
            runs_of(&[30, 100], &[-100, 1], &[0, 1], false),
            Runs {
                block: 3000,
                run: 3000
            }
        );
        // A 50-column slice of 1000 rows fills 8150-element buffers, a multiple of the row.
        assert_eq!(
            runs_of(&[1000, 50], &[100, 1], &[0, 1], false),
            Runs {
                block: 50_000,
                run: 8150
            }
        );
        // A cast input always goes through the buffer.
        assert_eq!(
            runs_of(&[20000], &[1], &[0], true),
            Runs {
                block: 20000,
                run: 8192
            }
        );
    }

    #[test]
    fn axes_order_by_stride_magnitude_and_keep_direction() {
        let ordered = axes(&[100, 30], &[1, -100], &[0, 1]);
        let strides: Vec<_> = ordered.iter().map(|axis| axis.stride).collect();
        assert_eq!(strides, [8, -800]);
        // Two reversed axes still step evenly and merge.
        let merged = axes(&[30, 100], &[-100, -1], &[0, 1]);
        assert_eq!(merged.len(), 1);
        assert_eq!((merged[0].length, merged[0].stride), (3000, -8));
    }
}
