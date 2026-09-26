//! Shape manipulation: reshaping, transposition, joining, splitting, tiling, flipping, rolling,
//! padding, broadcasting helpers, and triangle and diagonal extraction.
//!
//! Functions are exported through the native module `_numpy_shape`, which the frozen `numpy`
//! package re-exports. Reshaping and transposition return views whenever the layout allows,
//! as NumPy does; otherwise they copy.

use super::super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyError, PyKind, PyResult,
    PyRuntime, PyValue, PyValueCast,
};
use super::args::{self, Signature};
use super::array::{self, Array};
use super::convert;

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
        method("reshape", method_reshape),
        method("ravel", method_ravel),
        method("flatten", method_flatten),
        method("transpose", method_transpose),
    ],
    getters: &[],
};

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
            -1 if known == 0 || size % known != 0 => Err(mismatch()),
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

/// `array.reshape(shape)`: a view of C-contiguous arrays, otherwise a copy.
pub(in crate::python) fn reshape(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    requested: &[i64],
) -> PyResult<Array> {
    let shape = resolve_shape(requested, array.size())?;
    array::element_count(&shape)?;
    if array.is_c_contiguous() {
        let strides = array::contiguous_strides(&shape, array.itemsize());
        let offset = array.view.offset;
        return array::new_view(runtime, array, array.dtype, shape, strides, offset);
    }
    let buffer = array::contiguous_buffer(runtime, array)?;
    array::new_array(runtime, buffer, array.dtype, shape)
}

/// Parse `reshape(2, 3)`, `reshape((2, 3))`, or `reshape([2, 3])`.
fn requested_shape(runtime: &mut dyn PyRuntime, values: &[PyValue]) -> PyResult<Vec<i64>> {
    let values = match values {
        [single] if runtime.kind(single)? == PyKind::Tuple => {
            let tuple = single.cast(runtime)?;
            runtime.tuple_items(tuple)?
        }
        [single] if runtime.kind(single)? == PyKind::List => {
            let list = single.cast(runtime)?;
            runtime.list_items(list)?
        }
        values => values.to_vec(),
    };
    values
        .iter()
        .map(|value| args::index_int(runtime, value))
        .collect()
}

fn method_reshape(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.reject_unknown_keywords("reshape", &["order", "copy", "shape"])?;
    let array = Array::from_value(runtime, receiver)?;
    let mut positional = args.positional().to_vec();
    if let Some(shape) = args.keyword("reshape", "shape")? {
        positional.push(*shape);
    }
    let requested = requested_shape(runtime, &positional)?;
    Ok(reshape(runtime, &array, &requested)?.value())
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
    Ok(reshape(runtime, &array, &requested)?.value())
}

fn method_ravel(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("ravel", &["order"], 0);
    SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    Ok(array::ravel(runtime, &array)?.value())
}

fn module_ravel(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("ravel", &["a", "order"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    Ok(array::ravel(runtime, &array)?.value())
}

/// `a.flatten()`: always a copy.
fn method_flatten(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("flatten", &["order"], 0);
    SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    let buffer = array::contiguous_buffer(runtime, &array)?;
    Ok(array::new_array(runtime, buffer, array.dtype, vec![array.size()])?.value())
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
