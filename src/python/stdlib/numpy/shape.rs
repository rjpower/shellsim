//! Shape manipulation: reshaping, transposition, axis moves, joining, repetition, flipping,
//! rolling, padding, broadcasting, and triangle and diagonal extraction.
//!
//! Functions are exported through the native module `_numpy_shape`, which the frozen `numpy`
//! package re-exports.
//!
//! View-or-copy semantics follow NumPy: operations that only change shape and strides
//! (`reshape` when the layout allows, `transpose`, `swapaxes`, `moveaxis`, `squeeze`,
//! `expand_dims`, `flip`, `broadcast_to`, `diagonal`) return views, and the rest copy. Views that
//! NumPy makes read-only (`broadcast_to`, `diagonal`) are read-only here too.
//!
//! The submodules hold the kernels: [`join`] (concatenation), [`broadcast`], [`repeat`], and
//! [`diagonal`] (diagonals and trace). `tile`, `roll`, `pad`, `tril`/`triu`, `diff`, and the
//! stacking and splitting family are Python in `numpy._shape_base` and `numpy._arraypad`, as
//! they are in NumPy.

mod broadcast;
mod diagonal;
mod join;
mod repeat;

use super::super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyError, PyKind, PyNativeKind,
    PyResult, PyRuntime, PyValue, PyValueCast,
};
use super::super::super::Value;
use super::args::{self, Signature};
use super::array::{self, Array};
use super::convert;
use super::layout::{self, Order};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_shape",
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
    function("reshape", module_reshape),
    function("ravel", module_ravel),
    function("transpose", module_transpose),
    function("concatenate", join::module_concatenate),
    function("broadcast_to", broadcast::module_broadcast_to),
    function("repeat", repeat::module_repeat),
    function("diagonal", diagonal::module_diagonal),
    function("_normalize_axis_index", module_normalize_axis_index),
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

/// Methods this area installs on `numpy.ndarray`. `squeeze`, `swapaxes` and `trace` are frozen
/// Python in `numpy._shapes`, reached through [`super::reduce::python_method`] the same way
/// `mean`/`var`/`std` reach `numpy._stats`.
pub(in crate::python) static ARRAY_METHODS: NativeTypeDef = NativeTypeDef {
    name: "numpy.ndarray",
    methods: &[
        method("reshape", method_reshape),
        method("ravel", method_ravel),
        method("flatten", method_flatten),
        method("transpose", method_transpose),
        method("swapaxes", method_swapaxes),
        method("squeeze", method_squeeze),
        method("repeat", repeat::method_repeat),
        method("diagonal", diagonal::method_diagonal),
        method("trace", method_trace),
        method("resize", method_resize),
    ],
    getters: &[],
};

fn method_squeeze(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    super::reduce::python_method(runtime, "numpy._shapes", "squeeze", receiver, args)
}

fn method_swapaxes(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    super::reduce::python_method(runtime, "numpy._shapes", "swapaxes", receiver, args)
}

fn method_trace(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    super::reduce::python_method(runtime, "numpy._shapes", "trace", receiver, args)
}

/// Normalize `axis` against rank `ndim`, raising NumPy's `AxisError`. `prefix` names the
/// argument in the message, as in `axis1: axis 3 is out of bounds for array of dimension 2`.
pub(in crate::python) fn axis_index(
    axis: i64,
    ndim: usize,
    prefix: Option<&str>,
) -> PyResult<usize> {
    array::normalize_axis(axis, ndim).map_err(|error| match prefix {
        Some(prefix) => PyError::exception("AxisError", format!("{prefix}: {}", error.message())),
        None => error,
    })
}

/// Items of a tuple or list, or `None` for any other value.
pub(in crate::python) fn sequence_items(
    runtime: &mut dyn PyRuntime,
    value: &PyValue,
) -> PyResult<Option<Vec<PyValue>>> {
    Ok(match runtime.kind(value)? {
        PyKind::Tuple => {
            let tuple = value.cast(runtime)?;
            Some(runtime.tuple_items(tuple)?)
        }
        PyKind::List => {
            let list = value.cast(runtime)?;
            Some(runtime.list_items(list)?)
        }
        _ => None,
    })
}

/// Items of an iterable argument (tuple, list, ndarray, generator), or `None` for a value that
/// is not iterable. This is NumPy's `tuple(value)` with a fallback for scalars.
fn iterable_items(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<Option<Vec<PyValue>>> {
    if let Some(items) = sequence_items(runtime, value)? {
        return Ok(Some(items));
    }
    if runtime.native_kind(value)? == Some(PyNativeKind::Array) {
        let array = Array::from_value(runtime, *value)?;
        if array.ndim() == 0 {
            return Ok(None);
        }
        return super::ndarray::rows(runtime, &array).map(Some);
    }
    if matches!(
        runtime.kind(value)?,
        PyKind::None
            | PyKind::Bool
            | PyKind::Int
            | PyKind::Float
            | PyKind::Complex
            | PyKind::Function
            | PyKind::Class
            | PyKind::Module
    ) {
        return Ok(None);
    }
    let Ok(iterator) = runtime.iterator(*value) else {
        return Ok(None);
    };
    let mut items = Vec::new();
    while let Some(item) = runtime.iterator_next(iterator)? {
        runtime.charge_cpu(1)?;
        runtime.reserve_memory(16)?;
        items.push(item);
    }
    Ok(Some(items))
}

/// Integers from an int or an iterable of ints, as `tuple(value)` or `(value,)` in NumPy.
fn int_list(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<Vec<i64>> {
    match iterable_items(runtime, value)? {
        Some(items) => items
            .iter()
            .map(|item| args::index_int(runtime, item))
            .collect(),
        None => Ok(vec![args::index_int(runtime, value)?]),
    }
}

/// `_normalize_axis_index(axis, ndim, msg_prefix=None)`, for the frozen Python composites.
fn module_normalize_axis_index(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("normalize_axis_index", &["axis", "ndim", "msg_prefix"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let axis = args::index_int(runtime, &bound.required("axis"))?;
    let ndim = usize::try_from(args::index_int(runtime, &bound.required("ndim"))?)
        .map_err(|_| PyError::value_error("ndim must be non-negative"))?;
    let prefix = match bound.value("msg_prefix") {
        Some(prefix) => Some(runtime.string_value(&prefix)?.unwrap_or_default()),
        None => None,
    };
    Ok(Value::Int(axis_index(axis, ndim, prefix.as_deref())? as i64))
}

/// `A` order means Fortran order for arrays that are only Fortran-contiguous.
fn resolve_any_order(array: &Array, order: Order) -> Order {
    match order {
        Order::A if layout::is_fortran(array) => Order::F,
        Order::A => Order::C,
        order => order,
    }
}

/// Resolve a requested shape, which may contain one `-1`, against `size` elements.
pub(in crate::python) fn resolve_shape(requested: &[i64], size: usize) -> PyResult<Vec<usize>> {
    if requested
        .iter()
        .filter(|dimension| **dimension == -1)
        .count()
        > 1
    {
        return Err(PyError::value_error(
            "can only specify one unknown dimension",
        ));
    }
    if requested.iter().any(|dimension| *dimension < -1) {
        return Err(PyError::value_error("negative dimensions not allowed"));
    }
    let mismatch = || {
        PyError::value_error(format!(
            "cannot reshape array of size {size} into shape {}",
            shape_text(requested)
        ))
    };
    let known = requested
        .iter()
        .filter(|dimension| **dimension != -1)
        .try_fold(1usize, |total, dimension| {
            total.checked_mul(*dimension as usize)
        })
        .ok_or_else(mismatch)?;
    let shape = requested
        .iter()
        .map(|dimension| match *dimension {
            -1 if known == 0 || !size.is_multiple_of(known) => Err(mismatch()),
            -1 => Ok(size / known),
            dimension => Ok(dimension as usize),
        })
        .collect::<PyResult<Vec<_>>>()?;
    if shape.iter().product::<usize>() != size {
        return Err(mismatch());
    }
    Ok(shape)
}

/// NumPy's shape text in reshape errors: `(2,3)` or `(4,)`, with `-1` kept.
fn shape_text(shape: &[i64]) -> String {
    match shape {
        [single] => format!("({single},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

/// Strides that present an array of `old_shape`/`old_strides` as `shape` in C order without
/// copying, or `None` when the layout does not allow it. This is NumPy's
/// `_attempt_nocopy_reshape`: a run of old axes can be split or merged when each axis in the
/// run steps exactly over the next, and length-one axes take any stride. Both shapes must hold
/// the same, non-zero number of elements.
fn nocopy_strides(
    old_shape: &[usize],
    old_strides: &[isize],
    shape: &[usize],
    itemsize: usize,
) -> Option<Vec<isize>> {
    let (old_dims, old_steps): (Vec<usize>, Vec<isize>) = old_shape
        .iter()
        .zip(old_strides)
        .filter(|(dimension, _)| **dimension != 1)
        .map(|(dimension, stride)| (*dimension, *stride))
        .unzip();
    let mut strides = vec![0isize; shape.len()];
    let (mut old_start, mut old_end, mut new_start, mut new_end) = (0, 1, 0, 1);
    while new_start < shape.len() && old_start < old_dims.len() {
        let mut new_product = shape[new_start];
        let mut old_product = old_dims[old_start];
        while new_product != old_product {
            if new_product < old_product {
                new_product = new_product.checked_mul(*shape.get(new_end)?)?;
                new_end += 1;
            } else {
                old_product = old_product.checked_mul(*old_dims.get(old_end)?)?;
                old_end += 1;
            }
        }
        for axis in old_start..old_end - 1 {
            if old_steps[axis] != old_dims[axis + 1] as isize * old_steps[axis + 1] {
                return None;
            }
        }
        strides[new_end - 1] = old_steps[old_end - 1];
        for axis in (new_start + 1..new_end).rev() {
            strides[axis - 1] = strides[axis] * shape[axis] as isize;
        }
        new_start = new_end;
        new_end += 1;
        old_start = old_end;
        old_end += 1;
    }
    let last = if new_start >= 1 {
        strides[new_start - 1]
    } else {
        itemsize as isize
    };
    for stride in &mut strides[new_start..] {
        *stride = last;
    }
    Some(strides)
}

/// A C-order view of `array` with `shape`, when the layout allows one.
fn reshape_view(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    shape: &[usize],
) -> PyResult<Option<Array>> {
    let strides = if array.is_c_contiguous() {
        Some(array::contiguous_strides(shape, array.itemsize()))
    } else {
        nocopy_strides(array.shape(), array.strides(), shape, array.itemsize())
    };
    strides
        .map(|strides| {
            array::new_view(
                runtime,
                array,
                array.dtype,
                shape.to_vec(),
                strides,
                array.view.offset,
            )
        })
        .transpose()
}

/// `reshape` with NumPy's `order` and `copy` (`None`: copy only when needed).
fn reshape_with(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    requested: &[i64],
    order: Order,
    copy: Option<bool>,
) -> PyResult<Array> {
    let shape = resolve_shape(requested, array.size())?;
    array::element_count(&shape)?;
    match resolve_any_order(array, order) {
        Order::C => {}
        Order::F => {
            // Fortran order is C order on the reversed axes of both shapes.
            let reversed = transpose(runtime, array, None)?;
            let requested = shape
                .iter()
                .rev()
                .map(|dimension| *dimension as i64)
                .collect::<Vec<_>>();
            let result = reshape_with(runtime, &reversed, &requested, Order::C, copy)?;
            return transpose(runtime, &result, None);
        }
        Order::K | Order::A => {
            return Err(PyError::value_error(
                "order 'K' is not permitted for reshaping",
            ))
        }
    }
    if copy == Some(true) {
        let buffer = array::contiguous_buffer(runtime, array)?;
        return array::new_array(runtime, buffer, array.dtype, shape);
    }
    if let Some(view) = reshape_view(runtime, array, &shape)? {
        return Ok(view);
    }
    if copy == Some(false) {
        return Err(PyError::value_error(
            "Unable to avoid creating a copy while reshaping.",
        ));
    }
    let buffer = array::contiguous_buffer(runtime, array)?;
    array::new_array(runtime, buffer, array.dtype, shape)
}

/// Parse `reshape(2, 3)`, `reshape((2, 3))`, `reshape([2, 3])`, or an integer array, as
/// `PyArray_IntpConverter` does.
fn requested_shape(runtime: &mut dyn PyRuntime, values: &[PyValue]) -> PyResult<Vec<i64>> {
    let values = match values {
        [single] => iterable_items(runtime, single)?.unwrap_or_else(|| vec![*single]),
        values => values.to_vec(),
    };
    values
        .iter()
        .map(|value| args::index_int(runtime, value))
        .collect()
}

/// `copy=None | bool` as used by `reshape`.
fn copy_arg(runtime: &mut dyn PyRuntime, value: Option<PyValue>) -> PyResult<Option<bool>> {
    match value {
        None => Ok(None),
        Some(value) if value.is_none() => Ok(None),
        Some(value) => runtime.truth(&value).map(Some),
    }
}

fn method_reshape(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.reject_unknown_keywords("reshape", &["order", "copy", "shape"])?;
    let array = Array::from_value(runtime, receiver)?;
    let mut positional = args.positional().to_vec();
    if let Some(shape) = args.keyword("reshape", "shape")? {
        positional.push(*shape);
    }
    if positional.is_empty() {
        return Err(PyError::type_error(
            "reshape() missing required argument 'shape' (pos 0)",
        ));
    }
    let requested = requested_shape(runtime, &positional)?;
    let order = Order::parse(
        runtime,
        args.keyword("reshape", "order")?.copied(),
        Order::C,
    )?;
    let copy = copy_arg(runtime, args.keyword("reshape", "copy")?.copied())?;
    Ok(reshape_with(runtime, &array, &requested, order, copy)?.value())
}

fn module_reshape(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("reshape", &["a", "shape", "order"], 1).keyword_only(&["newshape", "copy"]);
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let shape = bound
        .value("shape")
        .or_else(|| bound.value("newshape"))
        .ok_or_else(|| {
            PyError::type_error("reshape() missing required argument 'shape' (pos 2)")
        })?;
    let requested = requested_shape(runtime, &[shape])?;
    let order = Order::parse(runtime, bound.value("order"), Order::C)?;
    let copy = copy_arg(runtime, bound.get("copy"))?;
    Ok(reshape_with(runtime, &array, &requested, order, copy)?.value())
}

/// `array` as one dimension in `order`: a view when the layout allows, otherwise a copy.
/// `K` reads elements in memory order: axes sorted by decreasing absolute stride.
fn ravel_order(runtime: &mut dyn PyRuntime, array: &Array, order: Order) -> PyResult<Array> {
    match resolve_any_order(array, order) {
        Order::C | Order::A => array::ravel(runtime, array),
        Order::F => {
            let reversed = transpose(runtime, array, None)?;
            array::ravel(runtime, &reversed)
        }
        Order::K => {
            let axes = layout::axes_like(array, Order::K, array.ndim());
            let permuted = permute(runtime, array, &axes)?;
            array::ravel(runtime, &permuted)
        }
    }
}

fn method_ravel(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("ravel", &["order"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    let order = Order::parse(runtime, bound.value("order"), Order::C)?;
    Ok(ravel_order(runtime, &array, order)?.value())
}

fn module_ravel(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("ravel", &["a", "order"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let order = Order::parse(runtime, bound.value("order"), Order::C)?;
    Ok(ravel_order(runtime, &array, order)?.value())
}

/// `a.flatten(order='C')`: always a copy.
fn method_flatten(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("flatten", &["order"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    let order = Order::parse(runtime, bound.value("order"), Order::C)?;
    let flat = ravel_order(runtime, &array, order)?;
    Ok(array::copy_array(runtime, &flat)?.value())
}

/// A view with the axes of `array` in `order`, which must be a permutation.
fn permute(runtime: &mut dyn PyRuntime, array: &Array, order: &[usize]) -> PyResult<Array> {
    let shape = order.iter().map(|axis| array.shape()[*axis]).collect();
    let strides = order.iter().map(|axis| array.strides()[*axis]).collect();
    array::new_view(
        runtime,
        array,
        array.dtype,
        shape,
        strides,
        array.view.offset,
    )
}

/// A view with axes permuted; `axes` defaults to reversing them.
pub(in crate::python) fn transpose(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    axes: Option<Vec<i64>>,
) -> PyResult<Array> {
    let ndim = array.ndim();
    let order = match axes {
        None => (0..ndim).rev().collect::<Vec<_>>(),
        Some(axes) => {
            if axes.len() != ndim {
                return Err(PyError::value_error("axes don't match array"));
            }
            let order = axes
                .iter()
                .map(|axis| array::normalize_axis(*axis, ndim))
                .collect::<PyResult<Vec<_>>>()?;
            let mut seen = vec![false; ndim];
            for axis in &order {
                if std::mem::replace(&mut seen[*axis], true) {
                    return Err(PyError::value_error("repeated axis in transpose"));
                }
            }
            order
        }
    };
    permute(runtime, array, &order)
}

fn transpose_axes(runtime: &mut dyn PyRuntime, values: &[PyValue]) -> PyResult<Option<Vec<i64>>> {
    match values {
        [] => Ok(None),
        [single] if single.is_none() => Ok(None),
        values => requested_shape(runtime, values).map(Some),
    }
}

fn method_transpose(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.reject_keywords("transpose")?;
    let array = Array::from_value(runtime, receiver)?;
    let axes = transpose_axes(runtime, args.positional())?;
    Ok(transpose(runtime, &array, axes)?.value())
}

fn module_transpose(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("transpose", &["a", "axes"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let axes = match bound.value("axes") {
        Some(axes) => transpose_axes(runtime, &[axes])?,
        None => None,
    };
    Ok(transpose(runtime, &array, axes)?.value())
}

/// `a.resize(new_shape)` changes an array's storage in place, which shellsim's array model
/// does not allow; `np.resize` returns a new array instead.
fn method_resize(_runtime: &mut dyn PyRuntime, _receiver: PyValue, _args: CallArgs) -> PyResult {
    Err(PyError::not_implemented_error(
        "ndarray.resize is not supported: arrays cannot change size in place; use np.resize",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nocopy_reshape_splits_and_merges_contiguous_runs() {
        // A transposed (2, 3) int64 array: shape (3, 2), strides (8, 24).
        assert_eq!(nocopy_strides(&[3, 2], &[8, 24], &[6], 8), None);
        assert_eq!(
            nocopy_strides(&[3, 2], &[8, 24], &[3, 1, 2], 8),
            Some(vec![8, 48, 24])
        );
        // Every other element of a (2, 3, 4) array: (2, 3, 2) with strides (96, 32, 16).
        assert_eq!(
            nocopy_strides(&[2, 3, 2], &[96, 32, 16], &[6, 2], 8),
            Some(vec![32, 16])
        );
        assert_eq!(
            nocopy_strides(&[2, 3, 2], &[96, 32, 16], &[12], 8),
            Some(vec![16])
        );
        assert_eq!(nocopy_strides(&[2, 2], &[32, 8], &[4], 8), None);
    }

    #[test]
    fn requested_shapes_resolve_one_unknown_dimension() {
        assert_eq!(resolve_shape(&[-1, 2], 6).unwrap(), [3, 2]);
        assert!(resolve_shape(&[4, -1], 6).is_err());
        assert!(resolve_shape(&[-1, -1], 6).is_err());
    }
}
