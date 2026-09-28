//! Array products in the top-level namespace: `dot`, `vdot`, `inner`, `outer`, `matmul` (the
//! `@` operator), `tensordot`, and `ndarray.dot`.
//!
//! Every product except `outer` runs one kernel over a [`Plan`]: for each output position it
//! sums the products of paired elements. The plan holds byte offsets relative to each operand's
//! first element for the batch, free, and summed positions, so strided views and broadcast
//! batch dimensions need no copies. The kernel computes in the result dtype, so integer
//! products wrap at the dtype width as NumPy's do. `float16` accumulates in `float32` and rounds
//! once, like NumPy's half-precision loops. Object arrays call the Python `*` and `+` operators
//! element by element, starting from the first product.
//!
//! Work is charged before it starts: one CPU unit per multiply-add, plus memory for the offset
//! tables and the result. Floating-point flags raised by a product go to `numpy.errstate` under
//! the function's name, as NumPy reports `overflow encountered in dot`.

use super::super::super::ast::BinaryOperator;
use super::super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyArrayBuffer, PyArrayData,
    PyArrayDtype, PyError, PyKind, PyOperator, PyResult, PyRuntime, PyValue, PyValueCast,
};
use super::super::super::Value;
use super::args::{self, Signature};
use super::array::{self, Array};
use super::convert;
use super::dtype::{self, Casting, DType, Kind};
use super::element::{dispatch_numeric, Element, F16};
use super::ops::{FpFlags, Numeric};
use super::ufunc;

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_products",
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
    function("dot", module_dot),
    function("vdot", vdot),
    function("inner", inner),
    function("outer", outer),
    function("matmul", module_matmul),
    function("tensordot", tensordot),
];

/// Methods this area installs on `numpy.ndarray`.
pub(in crate::python) static ARRAY_METHODS: NativeTypeDef = NativeTypeDef {
    name: "numpy.ndarray",
    methods: &[MethodDef {
        type_name: "numpy.ndarray",
        name: "dot",
        call: method_dot,
    }],
    getters: &[],
};

/// The gufunc signature NumPy prints in `matmul` shape errors.
const MATMUL_SIGNATURE: &str = "(n?,k),(k,m?)->(n?,m?)";

/// Relative byte offsets describing one product.
///
/// The output enumerates, for every batch, each `a` free position against each `b` free
/// position, in C order. Each output element sums `a[batch + free + a_sum[i]] *
/// b[batch + free + b_sum[i]]` over `i`.
struct Plan {
    /// Offsets of each batch's first element in `a` and `b`.
    batches: Vec<(isize, isize)>,
    a_free: Vec<isize>,
    b_free: Vec<isize>,
    a_sum: Vec<isize>,
    b_sum: Vec<isize>,
}

impl Plan {
    /// One unit per multiply-add, and at least one per output for the store.
    fn cost(&self, outputs: usize) -> u64 {
        (outputs as u64).saturating_mul(self.a_sum.len().max(1) as u64)
    }
}

/// Offsets of a strided shape relative to its first element, after reserving their memory.
fn offset_table(
    runtime: &mut dyn PyRuntime,
    shape: &[usize],
    strides: &[isize],
) -> PyResult<Vec<isize>> {
    let count = array::element_count(shape)?;
    runtime.reserve_memory(count.saturating_mul(std::mem::size_of::<isize>()))?;
    runtime.charge_cpu(count as u64 / 8 + 1)?;
    Ok(super::index::relative_offsets(shape, strides))
}

/// The dtype a product kernel computes in: `float16` accumulates in `float32`.
fn work_dtype(dtype: DType) -> DType {
    if dtype.kind() == Kind::Float16 {
        DType::FLOAT32
    } else {
        dtype
    }
}

/// Cast an operand to the kernel's working dtype. Object operands are made C-contiguous so a
/// relative byte offset divided by the slot size indexes the C-order snapshot of the array.
fn prepare(runtime: &mut dyn PyRuntime, array: &Array, dtype: DType) -> PyResult<Array> {
    let cast = convert::cast_array(runtime, array, work_dtype(dtype), false)?;
    if cast.dtype.kind() == Kind::Object && !cast.is_c_contiguous() {
        return array::copy_array(runtime, &cast);
    }
    Ok(cast)
}

/// Promote two operand dtypes for a product; `str` operands have no product loop and raise
/// `no_loop` instead.
fn product_dtype(a: DType, b: DType, no_loop: impl Fn() -> PyError) -> PyResult<DType> {
    if a.kind() == Kind::Str || b.kind() == Kind::Str {
        return Err(no_loop());
    }
    dtype::promote(a, b)
}

/// What an object product with nothing to sum returns: `dot` and `matmul` give `0`, while
/// `vdot` leaves NumPy's output slot empty, which reads back as `None`.
#[derive(Clone, Copy)]
enum EmptySum {
    Zero,
    None,
}

/// Evaluate `plan` over `a` and `b`, already prepared for `dtype`, into a new array of `shape`.
///
/// `conjugate_left` conjugates each element of `a` first, for `vdot` on object arrays; numeric
/// callers conjugate complex operands before planning.
#[allow(clippy::too_many_arguments)]
fn run_plan(
    runtime: &mut dyn PyRuntime,
    name: &str,
    a: &Array,
    b: &Array,
    dtype: DType,
    plan: &Plan,
    shape: Vec<usize>,
    empty: EmptySum,
    conjugate_left: bool,
) -> PyResult<Array> {
    let count = array::element_count(&shape)?;
    runtime.charge_cpu(plan.cost(count) + 1)?;
    if dtype.kind() == Kind::Object {
        let values = object_products(runtime, a, b, plan, count, empty, conjugate_left)?;
        return array::new_array(runtime, PyArrayBuffer::Values(values), DType::OBJECT, shape);
    }
    let work = a.dtype;
    array::reserve_elements(runtime, work, count)?;
    let mut bytes = vec![0u8; count * work.itemsize()];
    let mut flags = FpFlags::default();
    let (a_start, b_start) = (a.view.offset as isize, b.view.offset as isize);
    runtime.read_arrays(&[a.handle, b.handle], &mut |arrays| {
        let (PyArrayData::Bytes(left), PyArrayData::Bytes(right)) =
            (&arrays[0].data, &arrays[1].data)
        else {
            return Err(PyError::runtime_error("numeric product saw object storage"));
        };
        dispatch_numeric!(work.kind(), T => {
            product_kernel::<T>(left, a_start, right, b_start, plan, &mut bytes, &mut flags);
        }, _ => return Err(PyError::runtime_error("product kernel needs a numeric dtype")));
        Ok(())
    })?;
    if !work.is_inexact() {
        // Integer products wrap silently, as NumPy's array loops do.
        flags = FpFlags::default();
    }
    let (bytes, result_dtype) = if dtype.kind() == Kind::Float16 {
        (narrow_to_half(&bytes, &mut flags), dtype)
    } else {
        (bytes, work)
    };
    super::errstate::report(runtime, name, flags)?;
    array::new_array(runtime, PyArrayBuffer::Bytes(bytes), result_dtype, shape)
}

/// Sum the paired products of every output position of `plan`, at `T`'s width.
fn product_kernel<T: Numeric>(
    left: &[u8],
    a_start: isize,
    right: &[u8],
    b_start: isize,
    plan: &Plan,
    output: &mut [u8],
    flags: &mut FpFlags,
) {
    let mut chunks = output.chunks_exact_mut(T::SIZE);
    for (a_batch, b_batch) in &plan.batches {
        for a_free in &plan.a_free {
            let row = a_start + a_batch + a_free;
            for b_free in &plan.b_free {
                let column = b_start + b_batch + b_free;
                let mut sum = T::zero();
                for (a_sum, b_sum) in plan.a_sum.iter().zip(&plan.b_sum) {
                    let x = T::read(&left[(row + a_sum) as usize..]);
                    let y = T::read(&right[(column + b_sum) as usize..]);
                    sum = sum.add(x.multiply(y, flags), flags);
                }
                if let Some(chunk) = chunks.next() {
                    sum.write(chunk);
                }
            }
        }
    }
}

/// Round `float32` sums to `float16`, flagging finite sums that overflow the narrower type.
fn narrow_to_half(bytes: &[u8], flags: &mut FpFlags) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::read(chunk);
        let half = F16::from_f32(value);
        if value.is_finite() && !half.to_f32().is_finite() {
            flags.overflow = true;
        }
        output.extend_from_slice(&half.0.to_le_bytes());
    }
    output
}

/// Object products through the Python operators, summing from the first product as NumPy's
/// object loops do.
fn object_products(
    runtime: &mut dyn PyRuntime,
    a: &Array,
    b: &Array,
    plan: &Plan,
    count: usize,
    empty: EmptySum,
    conjugate_left: bool,
) -> PyResult<Vec<PyValue>> {
    let left = array::read_objects(runtime, a)?;
    let right = array::read_objects(runtime, b)?;
    array::reserve_elements(runtime, DType::OBJECT, count)?;
    let slot = |offset: isize| offset as usize / PyArrayDtype::VALUE_ITEMSIZE;
    let multiply = PyOperator::Binary(BinaryOperator::Multiply);
    let add = PyOperator::Binary(BinaryOperator::Add);
    let mut values = Vec::with_capacity(count);
    for (a_batch, b_batch) in &plan.batches {
        for a_free in &plan.a_free {
            for b_free in &plan.b_free {
                let mut sum = None;
                for (a_sum, b_sum) in plan.a_sum.iter().zip(&plan.b_sum) {
                    let mut x = left[slot(a_batch + a_free + a_sum)];
                    let y = right[slot(b_batch + b_free + b_sum)];
                    if conjugate_left {
                        x = call_method(runtime, x, "conjugate")?;
                    }
                    let product = runtime.apply_operator(multiply, &[x, y])?;
                    sum = Some(match sum {
                        None => product,
                        Some(total) => runtime.apply_operator(add, &[total, product])?,
                    });
                }
                values.push(sum.unwrap_or(match empty {
                    EmptySum::Zero => Value::Int(0),
                    EmptySum::None => Value::None,
                }));
            }
        }
    }
    Ok(values)
}

fn call_method(runtime: &mut dyn PyRuntime, value: PyValue, name: &str) -> PyResult {
    let method = runtime.get_attribute(value, name)?.ok_or_else(|| {
        PyError::exception(
            "AttributeError",
            format!("object has no attribute '{name}'"),
        )
    })?;
    runtime.call_value(method, CallArgs::new(Vec::new(), Vec::new()))
}

/// Contract `a` and `b` over paired axes. The result's axes are `a`'s remaining axes followed
/// by `b`'s, each in their original order. Callers check that paired axes have equal lengths.
fn contract(
    runtime: &mut dyn PyRuntime,
    name: &str,
    a: &Array,
    b: &Array,
    dtype: DType,
    a_axes: &[usize],
    b_axes: &[usize],
) -> PyResult<Array> {
    let a = prepare(runtime, a, dtype)?;
    let b = prepare(runtime, b, dtype)?;
    let free = |array: &Array, summed: &[usize]| -> (Vec<usize>, Vec<isize>) {
        (0..array.ndim())
            .filter(|axis| !summed.contains(axis))
            .map(|axis| (array.shape()[axis], array.strides()[axis]))
            .unzip()
    };
    let (a_free_shape, a_free_strides) = free(&a, a_axes);
    let (b_free_shape, b_free_strides) = free(&b, b_axes);
    let sum_shape = a_axes
        .iter()
        .map(|axis| a.shape()[*axis])
        .collect::<Vec<_>>();
    let a_sum_strides = a_axes
        .iter()
        .map(|axis| a.strides()[*axis])
        .collect::<Vec<_>>();
    let b_sum_strides = b_axes
        .iter()
        .map(|axis| b.strides()[*axis])
        .collect::<Vec<_>>();
    let mut shape = a_free_shape.clone();
    shape.extend_from_slice(&b_free_shape);
    array::element_count(&shape)?;
    let plan = Plan {
        batches: vec![(0, 0)],
        a_free: offset_table(runtime, &a_free_shape, &a_free_strides)?,
        b_free: offset_table(runtime, &b_free_shape, &b_free_strides)?,
        a_sum: offset_table(runtime, &sum_shape, &a_sum_strides)?,
        b_sum: offset_table(runtime, &sum_shape, &b_sum_strides)?,
    };
    run_plan(
        runtime,
        name,
        &a,
        &b,
        dtype,
        &plan,
        shape,
        EmptySum::Zero,
        false,
    )
}

/// A 0-d result as the scalar NumPy returns; other arrays as themselves.
fn scalar_or_array(runtime: &mut dyn PyRuntime, result: &Array) -> PyResult {
    if result.ndim() == 0 {
        return convert::element_to_scalar(runtime, result, result.view.offset);
    }
    Ok(result.value())
}

/// `multiply(a, b)`, which `dot` and `inner` use when either operand is 0-d.
fn multiply(runtime: &mut dyn PyRuntime, a: &Array, b: &Array, out: Option<Array>) -> PyResult {
    let index = ufunc::find("multiply").expect("multiply is a ufunc");
    let options = ufunc::Options {
        out,
        ..ufunc::Options::default()
    };
    ufunc::apply(runtime, index, &[a.value(), b.value()], &options)
}

fn not_aligned(a: &Array, b: &Array, a_axis: usize, b_axis: usize) -> PyError {
    PyError::value_error(format!(
        "shapes {} and {} not aligned: {} (dim {a_axis}) != {} (dim {b_axis})",
        array::format_shape(a.shape()),
        array::format_shape(b.shape()),
        a.shape()[a_axis],
        b.shape()[b_axis]
    ))
}

fn dot_unavailable() -> PyError {
    PyError::value_error("dot not available for this type")
}

/// `np.dot(a, b, out=None)`: a sum product over the last axis of `a` and the second-to-last
/// axis of `b` (its only axis when `b` is 1-D). A 0-d operand multiplies elementwise.
fn dot(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    out: Option<PyValue>,
) -> PyResult {
    let a = convert::as_array(runtime, left)?;
    let b = convert::as_array(runtime, right)?;
    let out = out
        .filter(|value| !value.is_none())
        .map(|value| Array::from_value(runtime, value))
        .transpose()?;
    if a.ndim() == 0 || b.ndim() == 0 {
        return multiply(runtime, &a, &b, out);
    }
    let dtype = product_dtype(a.dtype, b.dtype, dot_unavailable)?;
    let a_axis = a.ndim() - 1;
    let b_axis = b.ndim().saturating_sub(2);
    if a.shape()[a_axis] != b.shape()[b_axis] {
        return Err(not_aligned(&a, &b, a_axis, b_axis));
    }
    let result = contract(runtime, "dot", &a, &b, dtype, &[a_axis], &[b_axis])?;
    let Some(out) = out else {
        return scalar_or_array(runtime, &result);
    };
    if out.ndim() != result.ndim() || out.dtype != result.dtype || !out.is_c_contiguous() {
        return Err(PyError::value_error(
            "output array is not acceptable (must have the right datatype, number of \
             dimensions, and be a C-Array)",
        ));
    }
    if out.shape() != result.shape() {
        return Err(PyError::value_error("output array has wrong dimensions"));
    }
    array::assign(runtime, &out, &result)?;
    Ok(out.value())
}

fn module_dot(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("dot", &["a", "b", "out"], 2);
    let bound = SIGNATURE.bind(&args)?;
    dot(
        runtime,
        bound.required("a"),
        bound.required("b"),
        bound.get("out"),
    )
}

fn method_dot(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("dot", &["b", "out"], 1);
    let bound = SIGNATURE.bind(&args)?;
    dot(runtime, receiver, bound.required("b"), bound.get("out"))
}

/// `np.vdot(a, b)`: the dot product of both operands flattened, conjugating `a`. The result is
/// always a scalar.
fn vdot(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("vdot", &["a", "b"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let b = convert::as_array(runtime, bound.required("b"))?;
    let dtype = product_dtype(a.dtype, b.dtype, || {
        PyError::value_error("function not available for this data type")
    })?;
    let a = array::ravel(runtime, &a)?;
    let b = array::ravel(runtime, &b)?;
    if a.size() != b.size() {
        return Err(PyError::value_error(format!(
            "cannot reshape array of size {} into shape ({},)",
            b.size(),
            a.size()
        )));
    }
    let mut a = prepare(runtime, &a, dtype)?;
    let b = prepare(runtime, &b, dtype)?;
    if a.dtype.category() == dtype::Category::Complex {
        let conjugate = ufunc::find("conjugate").expect("conjugate is a ufunc");
        let value = ufunc::apply(runtime, conjugate, &[a.value()], &ufunc::Options::default())?;
        a = Array::from_value(runtime, value)?;
    }
    let length = a.size();
    let plan = Plan {
        batches: vec![(0, 0)],
        a_free: vec![0],
        b_free: vec![0],
        a_sum: offset_table(runtime, &[length], a.strides())?,
        b_sum: offset_table(runtime, &[length], b.strides())?,
    };
    let conjugate_objects = dtype.kind() == Kind::Object;
    let result = run_plan(
        runtime,
        "vdot",
        &a,
        &b,
        dtype,
        &plan,
        Vec::new(),
        EmptySum::None,
        conjugate_objects,
    )?;
    scalar_or_array(runtime, &result)
}

/// `np.inner(a, b)`: a sum product over the last axes of both operands. A 0-d operand
/// multiplies elementwise.
fn inner(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("inner", &["a", "b"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let b = convert::as_array(runtime, bound.required("b"))?;
    if a.ndim() == 0 || b.ndim() == 0 {
        return multiply(runtime, &a, &b, None);
    }
    let dtype = product_dtype(a.dtype, b.dtype, dot_unavailable)?;
    let (a_axis, b_axis) = (a.ndim() - 1, b.ndim() - 1);
    if a.shape()[a_axis] != b.shape()[b_axis] {
        return Err(not_aligned(&a, &b, a_axis, b_axis));
    }
    let result = contract(runtime, "inner", &a, &b, dtype, &[a_axis], &[b_axis])?;
    scalar_or_array(runtime, &result)
}

/// `np.outer(a, b, out=None)`: `multiply(a.ravel()[:, None], b.ravel()[None, :], out)`.
fn outer(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("outer", &["a", "b", "out"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let b = convert::as_array(runtime, bound.required("b"))?;
    let out = bound
        .value("out")
        .map(|value| Array::from_value(runtime, value))
        .transpose()?;
    let a = array::ravel(runtime, &a)?;
    let b = array::ravel(runtime, &b)?;
    let column = array::new_view(
        runtime,
        &a,
        a.dtype,
        vec![a.size(), 1],
        vec![a.strides()[0], 0],
        a.view.offset,
    )?;
    let row = array::new_view(
        runtime,
        &b,
        b.dtype,
        vec![1, b.size()],
        vec![0, b.strides()[0]],
        b.view.offset,
    )?;
    multiply(runtime, &column, &row, out)
}

/// `np.tensordot(a, b, axes=2)`: contract the last `axes` axes of `a` with the first `axes`
/// axes of `b`, or explicit axis lists `(axes_a, axes_b)`. The result stays an array even when
/// it is 0-d, as NumPy's does.
fn tensordot(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("tensordot", &["a", "b", "axes"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let b = convert::as_array(runtime, bound.required("b"))?;
    let (a_axes, b_axes) = match bound.get("axes") {
        None => count_axes(2),
        Some(value) => match runtime.kind(&value)? {
            PyKind::Tuple | PyKind::List => {
                let pair = sequence_items(runtime, value)?;
                let [first, second] = pair.as_slice() else {
                    return Err(PyError::value_error(format!(
                        "too many values to unpack (expected 2, got {})",
                        pair.len()
                    )));
                };
                (axis_list(runtime, *first)?, axis_list(runtime, *second)?)
            }
            _ => count_axes(args::index_int(runtime, &value)?),
        },
    };
    for axes in [&a_axes, &b_axes] {
        let mut sorted = axes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != axes.len() {
            return Err(PyError::value_error(
                "duplicate axes are not allowed in tensordot",
            ));
        }
    }
    if a_axes.len() != b_axes.len() {
        return Err(PyError::value_error("shape-mismatch for sum"));
    }
    let out_of_range = || PyError::exception("IndexError", "tuple index out of range");
    let mut a_normalized = Vec::with_capacity(a_axes.len());
    let mut b_normalized = Vec::with_capacity(b_axes.len());
    for (a_axis, b_axis) in a_axes.iter().zip(&b_axes) {
        let a_axis = python_index(*a_axis, a.ndim()).ok_or_else(out_of_range)?;
        let b_axis = python_index(*b_axis, b.ndim()).ok_or_else(out_of_range)?;
        if a.shape()[a_axis] != b.shape()[b_axis] {
            return Err(PyError::value_error("shape-mismatch for sum"));
        }
        a_normalized.push(a_axis);
        b_normalized.push(b_axis);
    }
    let dtype = product_dtype(a.dtype, b.dtype, dot_unavailable)?;
    let result = contract(runtime, "dot", &a, &b, dtype, &a_normalized, &b_normalized)?;
    Ok(result.value())
}

/// `axes=n`: the last `n` axes of `a` against the first `n` of `b`. Negative counts select no
/// axes, as `range(-n, 0)` does in NumPy's implementation.
fn count_axes(count: i64) -> (Vec<i64>, Vec<i64>) {
    let count = count.max(0);
    ((-count..0).collect(), (0..count).collect())
}

/// Resolve a possibly negative index into a tuple of `length` items, as Python indexing does.
fn python_index(index: i64, length: usize) -> Option<usize> {
    let resolved = if index < 0 {
        index.checked_add(length as i64)?
    } else {
        index
    };
    usize::try_from(resolved)
        .ok()
        .filter(|index| *index < length)
}

fn sequence_items(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<PyValue>> {
    match runtime.kind(&value)? {
        PyKind::Tuple => {
            let tuple = value.cast(runtime)?;
            runtime.tuple_items(tuple)
        }
        PyKind::List => {
            let list = value.cast(runtime)?;
            runtime.list_items(list)
        }
        _ => Ok(vec![value]),
    }
}

/// One side of `axes=(axes_a, axes_b)`: an int or a sequence of ints.
fn axis_list(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<i64>> {
    sequence_items(runtime, value)?
        .iter()
        .map(|item| args::index_int(runtime, item))
        .collect()
}

/// `a @ b` and `np.matmul(a, b)`.
pub(in crate::python) fn matmul(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
) -> PyResult {
    matmul_values(runtime, left, right, None, None)
}

/// `np.matmul(x1, x2, /, out=None, *, casting='same_kind', order='K', dtype=None, subok=True)`.
fn module_matmul(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("matmul", &["x1", "x2", "out"], 2).keyword_only(&[
            "casting",
            "order",
            "dtype",
            "subok",
            "axes",
            "axis",
            "signature",
        ]);
    let bound = SIGNATURE.bind(&args)?;
    for name in ["axes", "axis", "signature"] {
        if bound.value(name).is_some() {
            return Err(PyError::not_implemented_error(format!(
                "matmul() with {name}= is not supported"
            )));
        }
    }
    let casting = match bound.value("casting") {
        Some(value) => {
            let text = runtime.string_value(&value)?.unwrap_or_default();
            Casting::parse(&text)?
        }
        None => Casting::SameKind,
    };
    let dtype = args::optional_dtype(runtime, bound.get("dtype"))?.map(|dtype| (dtype, casting));
    let out = bound
        .value("out")
        .map(|value| Array::from_value(runtime, value))
        .transpose()?
        .map(|out| (out, casting));
    matmul_values(
        runtime,
        bound.required("x1"),
        bound.required("x2"),
        dtype,
        out,
    )
}

fn matmul_values(
    runtime: &mut dyn PyRuntime,
    left: PyValue,
    right: PyValue,
    requested: Option<(DType, Casting)>,
    out: Option<(Array, Casting)>,
) -> PyResult {
    let a = convert::as_array(runtime, left)?;
    let b = convert::as_array(runtime, right)?;
    for (position, operand) in [&a, &b].into_iter().enumerate() {
        if operand.ndim() == 0 {
            return Err(PyError::value_error(format!(
                "matmul: Input operand {position} does not have enough dimensions (has 0, \
                 gufunc core with signature {MATMUL_SIGNATURE} requires 1)"
            )));
        }
    }
    let common = product_dtype(a.dtype, b.dtype, || {
        PyError::type_error(format!(
            "matmul not supported for dtypes ({}, {})",
            a.dtype.repr(),
            b.dtype.repr()
        ))
    })?;
    let dtype = match requested {
        None => common,
        Some((dtype, casting)) => {
            for (position, operand) in [&a, &b].into_iter().enumerate() {
                if !dtype::can_cast(operand.dtype, dtype, casting) {
                    return Err(PyError::type_error(format!(
                        "Cannot cast ufunc 'matmul' input {position} from {} to {} with \
                         casting rule '{}'",
                        operand.dtype.repr(),
                        dtype.repr(),
                        casting.name()
                    )));
                }
            }
            dtype
        }
    };
    let a_k = a.shape()[a.ndim() - 1];
    let b_k_axis = b.ndim().saturating_sub(2);
    let b_k = b.shape()[b_k_axis];
    if a_k != b_k {
        return Err(PyError::value_error(format!(
            "matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc \
             signature {MATMUL_SIGNATURE} (size {b_k} is different from {a_k})"
        )));
    }
    let a_batch_rank = a.ndim().saturating_sub(2);
    let b_batch_rank = b.ndim().saturating_sub(2);
    let batch = array::broadcast_shapes(&[&a.shape()[..a_batch_rank], &b.shape()[..b_batch_rank]])
        .map_err(|_| remapped_broadcast_error(&a, &b))?;
    let a = prepare(runtime, &a, dtype)?;
    let b = prepare(runtime, &b, dtype)?;
    let a_batch_strides = batch_strides(&a, a_batch_rank, &batch);
    let b_batch_strides = batch_strides(&b, b_batch_rank, &batch);
    let (rows, row_stride) = if a.ndim() >= 2 {
        (a.shape()[a.ndim() - 2], a.strides()[a.ndim() - 2])
    } else {
        (1, 0)
    };
    let (columns, column_stride) = if b.ndim() >= 2 {
        (b.shape()[b.ndim() - 1], b.strides()[b.ndim() - 1])
    } else {
        (1, 0)
    };
    let mut shape = batch.clone();
    if a.ndim() >= 2 {
        shape.push(rows);
    }
    if b.ndim() >= 2 {
        shape.push(columns);
    }
    array::element_count(&shape)?;
    let a_batches = offset_table(runtime, &batch, &a_batch_strides)?;
    let b_batches = offset_table(runtime, &batch, &b_batch_strides)?;
    let plan = Plan {
        batches: a_batches.into_iter().zip(b_batches).collect(),
        a_free: offset_table(runtime, &[rows], &[row_stride])?,
        b_free: offset_table(runtime, &[columns], &[column_stride])?,
        a_sum: offset_table(runtime, &[a_k], &[a.strides()[a.ndim() - 1]])?,
        b_sum: offset_table(runtime, &[a_k], &[b.strides()[b_k_axis]])?,
    };
    let result = run_plan(
        runtime,
        "matmul",
        &a,
        &b,
        dtype,
        &plan,
        shape,
        EmptySum::Zero,
        false,
    )?;
    let Some((out, casting)) = out else {
        return scalar_or_array(runtime, &result);
    };
    if out.shape() != result.shape() {
        return Err(PyError::value_error(format!(
            "matmul: output operand has shape {}, but the result has shape {}",
            array::format_shape(out.shape()),
            array::format_shape(result.shape())
        )));
    }
    if !dtype::can_cast(result.dtype, out.dtype, casting) {
        return Err(PyError::type_error(format!(
            "Cannot cast ufunc 'matmul' output from {} to {} with casting rule '{}'",
            result.dtype.repr(),
            out.dtype.repr(),
            casting.name()
        )));
    }
    array::assign(runtime, &out, &result)?;
    Ok(out.value())
}

/// Strides of an operand's leading `rank` batch axes, broadcast to `batch`.
fn batch_strides(array: &Array, rank: usize, batch: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; batch.len()];
    let extra = batch.len() - rank;
    for axis in 0..rank {
        if array.shape()[axis] != 1 {
            strides[extra + axis] = array.strides()[axis];
        }
    }
    strides
}

/// NumPy's gufunc broadcasting error, which lists each operand's batch axes followed by one
/// `newaxis` per core output axis.
fn remapped_broadcast_error(a: &Array, b: &Array) -> PyError {
    let core = usize::from(a.ndim() >= 2) + usize::from(b.ndim() >= 2);
    let remapped = |array: &Array| {
        let batch = &array.shape()[..array.ndim().saturating_sub(2)];
        let parts = batch
            .iter()
            .map(ToString::to_string)
            .chain(std::iter::repeat_n("newaxis".to_string(), core))
            .collect::<Vec<_>>();
        format!(
            "{}->({})",
            array::format_shape(array.shape()),
            parts.join(",")
        )
    };
    let mut requested = Vec::new();
    if a.ndim() >= 2 {
        requested.push(a.shape()[a.ndim() - 2].to_string());
    }
    if b.ndim() >= 2 {
        requested.push(b.shape()[b.ndim() - 1].to_string());
    }
    PyError::value_error(format!(
        "operands could not be broadcast together with remapped shapes [original->remapped]: \
         {} {}  and requested shape ({})",
        remapped(a),
        remapped(b),
        requested.join(",")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensordot_axis_counts_follow_python_ranges() {
        assert_eq!(count_axes(2), (vec![-2, -1], vec![0, 1]));
        assert_eq!(count_axes(0), (vec![], vec![]));
        assert_eq!(count_axes(-1), (vec![], vec![]));
        assert_eq!(python_index(-1, 3), Some(2));
        assert_eq!(python_index(-4, 3), None);
        assert_eq!(python_index(3, 3), None);
    }

    #[test]
    fn product_kernel_sums_strided_pairs_with_wrapping() {
        // a = [[1, 2], [3, 4]] as int8, b = [[100], [100]]: row sums wrap at 8 bits.
        let left = array::pack_elements(&[1i8, 2, 3, 4]);
        let right = array::pack_elements(&[100i8, 100]);
        let plan = Plan {
            batches: vec![(0, 0)],
            a_free: vec![0, 2],
            b_free: vec![0],
            a_sum: vec![0, 1],
            b_sum: vec![0, 1],
        };
        let mut output = vec![0u8; 2];
        let mut flags = FpFlags::default();
        product_kernel::<i8>(&left, 0, &right, 0, &plan, &mut output, &mut flags);
        assert_eq!(output, [300i32 as i8 as u8, 700i32 as i8 as u8]);
    }

    #[test]
    fn half_precision_sums_round_once_and_flag_overflow() {
        let mut flags = FpFlags::default();
        let bytes = array::pack_elements(&[1.5f32, 180_000.0]);
        let narrowed = narrow_to_half(&bytes, &mut flags);
        assert_eq!(F16::read(&narrowed[..2]).to_f32(), 1.5);
        assert!(F16::read(&narrowed[2..]).to_f32().is_infinite());
        assert!(flags.overflow);
    }
}
