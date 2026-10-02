//! ndarray indexing: basic views, advanced selection, and assignment through both.
//!
//! An index is parsed into [`Item`]s, then resolved against the array in two passes.
//!
//! 1. **Basic pass.** Integers, slices, `None`, and `...` only move the view: they change
//!    shape, strides, and offset over the same storage, so the result is a view.
//! 2. **Advanced pass.** Integer and boolean arrays select elements by position. Boolean masks
//!    become one integer array per mask axis (their `nonzero`). A boolean scalar or 0-d mask
//!    adds a new axis of length 1 and selects all of it (`True`) or none of it (`False`), as
//!    NumPy's `HAS_0D_BOOL` does. When any array is present, plain integers join them as 0-d
//!    arrays, as in NumPy. A 0-d integer array alone acts as an integer, but a result that
//!    would be a view is copied instead (NumPy's `HAS_SCALAR_ARRAY`). The index arrays
//!    broadcast to one
//!    shape `B`, and the selected elements form a copy:
//!    - when the advanced items are adjacent, `B` replaces them in place;
//!    - otherwise `B` comes first, followed by the remaining axes.
//!
//! Both passes produce a [`Selection`]: a view, or the byte offsets of every selected element
//! in result order. Reading gathers those offsets; assignment scatters a broadcast value into
//! them. Because the value is fully materialized first, overlapping source and destination
//! behave as if the source were copied, and with repeated indices the last write wins.

use super::super::super::native::{
    PyArrayBuffer, PyArrayView, PyError, PyKind, PyNativeKind, PyResult, PyRuntime, PyValue,
    PyValueCast,
};
use super::super::super::slice::SlicePlan;
use super::array::{
    broadcast_shapes, broadcast_strides, format_shape, gather_into, new_array, new_view,
    read_elements, reserve_elements, scatter, Array, Offsets,
};
use super::convert;
use super::dtype::{Category, DType, Kind};

/// One parsed index component.
enum Item<'s> {
    Int(i64),
    /// A 0-d integer array, which indexes like an integer.
    ScalarArray(i64),
    Slice(Option<i64>, Option<i64>, Option<i64>),
    NewAxis,
    Ellipsis,
    /// An integer index array, as int64.
    Array(Array<'s>),
    /// A boolean mask.
    Mask(Array<'s>),
    /// A boolean scalar or 0-d mask.
    Bool(bool),
}

impl Item<'_> {
    /// Array axes the item consumes.
    fn consumes(&self) -> usize {
        match self {
            Self::Int(_) | Self::ScalarArray(_) | Self::Slice(..) | Self::Array(_) => 1,
            Self::Mask(mask) => mask.ndim(),
            Self::NewAxis | Self::Ellipsis | Self::Bool(_) => 0,
        }
    }
}

const INVALID_INDEX: &str = "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis \
                             (`None`) and integer or boolean arrays are valid indices";

fn index_error(message: impl Into<String>) -> PyError {
    PyError::exception("IndexError", message)
}

/// Parse a subscript into items. A tuple indexes several axes; anything else indexes one.
fn parse<'s>(runtime: &mut dyn PyRuntime<'s>, index: PyValue<'s>) -> PyResult<'s, Vec<Item<'s>>> {
    if runtime.kind(&index)? == PyKind::Tuple {
        let tuple = index.cast(runtime)?;
        let values = runtime.tuple_items(tuple)?;
        return values
            .into_iter()
            .map(|value| parse_item(runtime, value))
            .collect();
    }
    Ok(vec![parse_item(runtime, index)?])
}

fn parse_item<'s>(runtime: &mut dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Item<'s>> {
    if value.is_none() {
        return Ok(Item::NewAxis);
    }
    if runtime.is_ellipsis(&value) {
        return Ok(Item::Ellipsis);
    }
    if let Some((start, stop, step)) = runtime.slice_parts(&value)? {
        return Ok(Item::Slice(start, stop, step));
    }
    match runtime.kind(&value)? {
        PyKind::Bool => return Ok(Item::Bool(runtime.truth(&value)?)),
        PyKind::Int => {
            return runtime
                .int_value(&value)
                .map(Item::Int)
                .ok_or_else(|| index_error("cannot fit 'int' into an index-sized integer"))
        }
        PyKind::List | PyKind::Tuple => {}
        _ if runtime.native_kind(&value)? == Some(PyNativeKind::Array) => {}
        // NumPy indexes with any sequence as an array, so a range selects its integers.
        _ if runtime.type_name(&value)? == "range" => {}
        _ => {
            if let Some((dtype, number)) = super::scalar::unbox_number(runtime, &value) {
                if dtype.kind() == Kind::Bool {
                    return Ok(Item::Bool(number.wrapping_i64() != 0));
                }
                if super::scalar::is_index_dtype(dtype) {
                    return Ok(Item::Int(number.wrapping_i64()));
                }
            }
            return Err(index_error(INVALID_INDEX));
        }
    }
    let array = convert::array_from_python(runtime, value, None, false)?;
    // An empty list is an empty integer index, not a float array.
    let array = if array.size() == 0 && array.dtype == DType::FLOAT64 {
        convert::cast_array(runtime, &array, DType::INT64, false)?
    } else {
        array
    };
    match array.dtype.category() {
        Category::Bool if array.ndim() == 0 => Ok(Item::Bool(truth_values(runtime, &array)?[0])),
        Category::Bool => Ok(Item::Mask(array)),
        Category::Signed | Category::Unsigned if array.ndim() == 0 => {
            let array = convert::cast_array(runtime, &array, DType::INT64, false)?;
            Ok(Item::ScalarArray(read_elements::<i64>(runtime, &array)?[0]))
        }
        Category::Signed | Category::Unsigned => Ok(Item::Array(convert::cast_array(
            runtime,
            &array,
            DType::INT64,
            false,
        )?)),
        _ => Err(index_error(
            "arrays used as indices must be of integer (or boolean) type",
        )),
    }
}

/// Where an index points.
pub(in crate::python) enum Selection<'s> {
    /// A view of the indexed array. `scalar` means every axis took an integer, so reading
    /// returns an element instead of a 0-d array.
    View { view: Array<'s>, scalar: bool },
    /// Byte offsets of the selected elements in C order of `shape`.
    Gather {
        shape: Vec<usize>,
        offsets: Vec<usize>,
    },
}

/// Resolve `index` against `array`.
pub(in crate::python) fn select<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
    index: PyValue<'s>,
) -> PyResult<'s, Selection<'s>> {
    let items = parse(runtime, index)?;
    let ndim = array.ndim();
    let ellipses = items
        .iter()
        .filter(|item| matches!(item, Item::Ellipsis))
        .count();
    if ellipses > 1 {
        return Err(index_error(
            "an index can only have a single ellipsis ('...')",
        ));
    }
    let consumed: usize = items.iter().map(Item::consumes).sum();
    if consumed > ndim {
        return Err(index_error(format!(
            "too many indices for array: array is {ndim}-dimensional, but {consumed} were indexed"
        )));
    }
    let scalar = ellipses == 0
        && consumed == ndim
        && items
            .iter()
            .all(|item| matches!(item, Item::Int(_) | Item::ScalarArray(_)));
    let copy = items
        .iter()
        .any(|item| matches!(item, Item::ScalarArray(_)));
    let mut expanded = Vec::with_capacity(items.len() + ndim);
    let mut axis = 0;
    for item in items {
        match item {
            Item::Ellipsis => {
                for _ in 0..ndim - consumed {
                    expanded.push(Item::Slice(None, None, None));
                    axis += 1;
                }
            }
            Item::Mask(mask) => {
                for indices in mask_indices(runtime, array, &mask, axis)? {
                    expanded.push(Item::Array(indices));
                    axis += 1;
                }
            }
            item => {
                axis += item.consumes();
                expanded.push(item);
            }
        }
    }
    while axis < ndim {
        expanded.push(Item::Slice(None, None, None));
        axis += 1;
    }
    let advanced = expanded
        .iter()
        .any(|item| matches!(item, Item::Array(_) | Item::Bool(_)));
    if advanced {
        for item in &mut expanded {
            if let Item::Int(value) | Item::ScalarArray(value) = item {
                let zero_d = super::array::array_from_elements(
                    runtime,
                    DType::INT64,
                    Vec::new(),
                    &[*value],
                )?;
                *item = Item::Array(zero_d);
            }
        }
    }

    // Basic pass: build the view, keeping each advanced axis whole.
    let mut shape = Vec::with_capacity(expanded.len());
    let mut strides = Vec::with_capacity(expanded.len());
    let mut offset = array.view.offset as isize;
    let mut arrays = Vec::new();
    let mut source_axis = 0;
    for item in expanded {
        match item {
            Item::NewAxis => {
                shape.push(1);
                strides.push(0);
            }
            Item::Int(value) | Item::ScalarArray(value) => {
                let length = array.shape()[source_axis];
                let position = normalize_index(value, length, source_axis)?;
                offset += position as isize * array.strides()[source_axis];
                source_axis += 1;
            }
            Item::Slice(start, stop, step) => {
                let length = array.shape()[source_axis];
                let stride = array.strides()[source_axis];
                let plan =
                    SlicePlan::new(length, start, stop, step).map_err(PyError::value_error)?;
                if plan.len() > 0 {
                    offset += plan.first() * stride;
                }
                shape.push(plan.len());
                strides.push(stride * plan.step());
                source_axis += 1;
            }
            Item::Array(indices) => {
                arrays.push((shape.len(), source_axis, indices));
                shape.push(array.shape()[source_axis]);
                strides.push(array.strides()[source_axis]);
                source_axis += 1;
            }
            Item::Bool(value) => {
                // Index a new length-1 axis with `[0]` or `[]`.
                let length = usize::from(value);
                let indices = super::array::array_from_elements(
                    runtime,
                    DType::INT64,
                    vec![length],
                    &vec![0i64; length],
                )?;
                arrays.push((shape.len(), source_axis, indices));
                shape.push(1);
                strides.push(0);
            }
            Item::Mask(_) | Item::Ellipsis => unreachable!("expanded above"),
        }
    }
    if arrays.is_empty() {
        let view = new_view(runtime, array, array.dtype, shape, strides, offset as usize)?;
        if copy && !scalar {
            reserve_elements(runtime, DType::INT64, view.size())?;
            let offsets = view.offsets().collect();
            let shape = view.shape().to_vec();
            return Ok(Selection::Gather { shape, offsets });
        }
        return Ok(Selection::View { view, scalar });
    }
    gather_plan(runtime, &shape, &strides, offset, &arrays)
}

/// Normalize one integer index against an axis, with NumPy's error text.
fn normalize_index<'s>(value: i64, length: usize, axis: usize) -> PyResult<'s, usize> {
    let signed = length as i64;
    let position = if value < 0 { value + signed } else { value };
    if (0..signed).contains(&position) {
        Ok(position as usize)
    } else {
        Err(index_error(format!(
            "index {value} is out of bounds for axis {axis} with size {length}"
        )))
    }
}

/// Integer index arrays for a boolean mask applied at `axis`.
fn mask_indices<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
    mask: &Array<'s>,
    axis: usize,
) -> PyResult<'s, Vec<Array<'s>>> {
    for (offset, length) in mask.shape().iter().enumerate() {
        let size = array.shape()[axis + offset];
        if *length != size {
            return Err(index_error(format!(
                "boolean index did not match indexed array along axis {}; size of axis is {size} \
                 but size of corresponding boolean axis is {length}",
                axis + offset
            )));
        }
    }
    let coordinates = nonzero(runtime, mask)?;
    coordinates
        .into_iter()
        .map(|values| {
            let length = values.len();
            super::array::array_from_elements(runtime, DType::INT64, vec![length], &values)
        })
        .collect()
}

/// Coordinates of the true elements of `array`, one vector per axis, in C order. A 0-d array
/// yields one coordinate list per axis of a 1-d view, as `np.nonzero` does for `atleast_1d`.
pub(in crate::python) fn nonzero<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
) -> PyResult<'s, Vec<Vec<i64>>> {
    let truth = truth_values(runtime, array)?;
    let ndim = array.ndim().max(1);
    let shape = if array.ndim() == 0 {
        vec![1]
    } else {
        array.shape().to_vec()
    };
    let selected = truth.iter().filter(|value| **value).count();
    runtime.reserve_memory(selected.saturating_mul(ndim).saturating_mul(8))?;
    let mut coordinates = vec![Vec::with_capacity(selected); ndim];
    let mut index = vec![0usize; ndim];
    for value in truth {
        if value {
            for (axis, coordinate) in index.iter().enumerate() {
                coordinates[axis].push(*coordinate as i64);
            }
        }
        for axis in (0..ndim).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(coordinates)
}

/// The truth value of every element in C order.
pub(in crate::python) fn truth_values<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
) -> PyResult<'s, Vec<bool>> {
    match array.dtype.kind() {
        Kind::Object => {
            let values = super::array::read_objects(runtime, array)?;
            values.iter().map(|value| runtime.truth(value)).collect()
        }
        Kind::Str => {
            let cast = convert::cast_array(runtime, array, DType::BOOL, false)?;
            read_elements::<bool>(runtime, &cast)
        }
        Kind::Bool => read_elements::<bool>(runtime, array),
        _ => {
            let cast = convert::cast_array(runtime, array, DType::BOOL, false)?;
            read_elements::<bool>(runtime, &cast)
        }
    }
}

/// Byte offsets of an advanced selection.
///
/// `arrays` holds `(view axis, source axis, int64 index array)` for each advanced item; the
/// view built by the basic pass keeps those axes whole.
fn gather_plan<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    arrays: &[(usize, usize, Array<'s>)],
) -> PyResult<'s, Selection<'s>> {
    let shapes = arrays
        .iter()
        .map(|(_, _, indices)| indices.shape())
        .collect::<Vec<_>>();
    let broadcast = broadcast_shapes(&shapes).map_err(|_| {
        let rendered = shapes
            .iter()
            .map(|shape| format_shape(shape))
            .collect::<Vec<_>>()
            .join(" ");
        index_error(format!(
            "shape mismatch: indexing arrays could not be broadcast together with shapes \
             {rendered} "
        ))
    })?;
    let points = super::array::element_count(&broadcast)?;
    // Offset of each broadcast index position within the view.
    let mut base = vec![0isize; points];
    runtime.reserve_memory(points.saturating_mul(8))?;
    for (view_axis, source_axis, indices) in arrays {
        // `values` is a C-order copy, so element strides come from the index array's shape.
        let values = read_elements::<i64>(runtime, indices)?;
        let copy = PyArrayView {
            dtype: indices.view.dtype,
            shape: indices.shape().to_vec(),
            strides: super::array::contiguous_strides(indices.shape(), 1),
            offset: 0,
            writeable: false,
        };
        let element_strides = broadcast_strides(&copy, &broadcast)?;
        let length = shape[*view_axis];
        let stride = strides[*view_axis];
        for (point, element) in Offsets::new(&broadcast, &element_strides, 0).enumerate() {
            let position = normalize_index(values[element], length, *source_axis)?;
            base[point] += position as isize * stride;
        }
    }
    let advanced_axes = arrays.iter().map(|(axis, _, _)| *axis).collect::<Vec<_>>();
    let adjacent = advanced_axes.windows(2).all(|pair| pair[1] == pair[0] + 1);
    let rest = (0..shape.len())
        .filter(|axis| !advanced_axes.contains(axis))
        .collect::<Vec<_>>();
    let split = if adjacent {
        rest.partition_point(|axis| *axis < advanced_axes[0])
    } else {
        0
    };
    let (before, after) = rest.split_at(split);
    let axes_shape = |axes: &[usize]| axes.iter().map(|axis| shape[*axis]).collect::<Vec<_>>();
    let axes_strides = |axes: &[usize]| axes.iter().map(|axis| strides[*axis]).collect::<Vec<_>>();
    let before_offsets = relative_offsets(&axes_shape(before), &axes_strides(before));
    let after_offsets = relative_offsets(&axes_shape(after), &axes_strides(after));
    let mut result_shape = axes_shape(before);
    result_shape.extend_from_slice(&broadcast);
    result_shape.extend(axes_shape(after));
    let count = super::array::element_count(&result_shape)?;
    runtime.reserve_memory(count.saturating_mul(8))?;
    runtime.charge_cpu(count as u64 + 1)?;
    let mut offsets = Vec::with_capacity(count);
    for outer in &before_offsets {
        for point in &base {
            for inner in &after_offsets {
                offsets.push((offset + outer + point + inner) as usize);
            }
        }
    }
    Ok(Selection::Gather {
        shape: result_shape,
        offsets,
    })
}

/// Offsets, relative to the first element, of every element of a strided shape in C order.
pub(in crate::python) fn relative_offsets(shape: &[usize], strides: &[isize]) -> Vec<isize> {
    let count: usize = shape.iter().product();
    let mut offsets = Vec::with_capacity(count);
    let mut index = vec![0usize; shape.len()];
    let mut current = 0isize;
    for _ in 0..count {
        offsets.push(current);
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            current += strides[axis];
            if index[axis] < shape[axis] {
                break;
            }
            current -= strides[axis] * shape[axis] as isize;
            index[axis] = 0;
        }
    }
    offsets
}

/// `array[index]`.
pub(in crate::python) fn get_item<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
    index: PyValue<'s>,
) -> PyResult<'s, PyValue<'s>> {
    match select(runtime, array, index)? {
        Selection::View { view, scalar: true } => {
            convert::element_to_scalar(runtime, &view, view.view.offset)
        }
        Selection::View { view, .. } => Ok(view.value()),
        Selection::Gather { shape, offsets } => {
            Ok(gather(runtime, array, &offsets, shape)?.value())
        }
    }
}

/// A new array holding the elements of `array` at byte `offsets`.
pub(in crate::python) fn gather<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
    offsets: &[usize],
    shape: Vec<usize>,
) -> PyResult<'s, Array<'s>> {
    reserve_elements(runtime, array.dtype, offsets.len())?;
    runtime.charge_cpu(offsets.len() as u64 + 1)?;
    let mut buffer = super::array::buffer_with_capacity(array.dtype, offsets.len());
    runtime.read_arrays(&[array.handle], &mut |refs, arrays| {
        gather_into(refs, &arrays[0], offsets.iter().copied(), &mut buffer);
        Ok(())
    })?;
    new_array(runtime, buffer, array.dtype, shape)
}

/// `array[index] = value`.
pub(in crate::python) fn set_item<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
    index: PyValue<'s>,
    value: PyValue<'s>,
) -> PyResult<'s, ()> {
    if !array.view.writeable {
        return Err(PyError::value_error("assignment destination is read-only"));
    }
    match select(runtime, array, index)? {
        Selection::View { view, scalar: true } => {
            // One element: object arrays store the value itself, even a list.
            let buffer = if array.dtype.kind() == Kind::Object {
                PyArrayBuffer::Values(vec![value])
            } else if runtime.native_kind(&value)? == Some(PyNativeKind::Array) {
                let source = Array::from_value(runtime, value)?;
                if source.ndim() > 0 {
                    return Err(PyError::value_error(
                        "setting an array element with a sequence.",
                    ));
                }
                return super::array::assign(runtime, &view, &source);
            } else {
                convert::value_to_buffer(runtime, value, array.dtype)?
            };
            scatter(runtime, array, &[view.view.offset], &buffer)
        }
        Selection::View { view, .. } => {
            let source = assignment_source(runtime, value, array.dtype)?;
            super::array::assign(runtime, &view, &source)
        }
        Selection::Gather { shape, offsets } => {
            let source = assignment_source(runtime, value, array.dtype)?;
            if is_full_mask(runtime, array, index)? {
                check_mask_assignment(&source, offsets.len())?;
            } else if !super::array::assignable(source.shape(), &shape) {
                return Err(PyError::value_error(format!(
                    "shape mismatch: value array of shape {} could not be broadcast to indexing \
                     result of shape {}",
                    format_shape(source.shape()),
                    format_shape(&shape)
                )));
            }
            let buffer = super::array::broadcast_buffer(runtime, &source, array.dtype, &shape)?;
            scatter(runtime, array, &offsets, &buffer)
        }
    }
}

/// Whether `index`, alone or as a 1-tuple, is one boolean array of `array`'s shape, which NumPy
/// assigns through with its own rules (`array_assign_boolean_subscript`) rather than by
/// broadcasting.
fn is_full_mask<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
    index: PyValue<'s>,
) -> PyResult<'s, bool> {
    let index = if runtime.kind(&index)? == PyKind::Tuple {
        let tuple = index.cast(runtime)?;
        match runtime.tuple_items(tuple)?.as_slice() {
            [item] => *item,
            _ => return Ok(false),
        }
    } else {
        index
    };
    if runtime.native_kind(&index)? != Some(PyNativeKind::Array) {
        return Ok(false);
    }
    let mask = Array::from_value(runtime, index)?;
    Ok(mask.dtype.kind() == Kind::Bool && mask.shape() == array.shape())
}

/// A value assigned through a full boolean mask is a scalar, one value, or one value for each
/// `True` element.
fn check_mask_assignment<'s>(source: &Array<'s>, selected: usize) -> PyResult<'s, ()> {
    if source.ndim() > 1 {
        return Err(PyError::type_error(format!(
            "NumPy boolean array indexing assignment requires a 0 or 1-dimensional input, input \
             has {} dimensions",
            source.ndim()
        )));
    }
    if source.size() != 1 && source.size() != selected {
        return Err(PyError::value_error(format!(
            "NumPy boolean array indexing assignment cannot assign {} input values to the {} \
             output values where the mask is true",
            source.size(),
            selected
        )));
    }
    Ok(())
}

/// The array an assigned value stands for, converted with the destination's rules.
fn assignment_source<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    value: PyValue<'s>,
    dtype: DType,
) -> PyResult<'s, Array<'s>> {
    if runtime.native_kind(&value)? == Some(PyNativeKind::Array) {
        return Array::from_value(runtime, value);
    }
    convert::array_from_python(runtime, value, Some(dtype), false)
}

/// How `take` and `put` treat indices outside the axis, NumPy's `NPY_CLIPMODE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum ClipMode {
    Raise,
    Wrap,
    Clip,
}

impl ClipMode {
    /// Parse `mode=` as `PyArray_ClipmodeConverter` does; `None` means `raise`.
    pub(in crate::python) fn parse<'s>(
        runtime: &mut dyn PyRuntime<'s>,
        value: Option<PyValue<'s>>,
    ) -> PyResult<'s, Self> {
        let Some(value) = value.filter(|value| !value.is_none()) else {
            return Ok(Self::Raise);
        };
        let Some(text) = runtime.string_value(&value)? else {
            return Err(PyError::type_error("clipmode not understood"));
        };
        let (mode, exact) = match text.chars().next() {
            Some('c' | 'C') => (Self::Clip, text == "clip"),
            Some('w' | 'W') => (Self::Wrap, text == "wrap"),
            Some('r' | 'R') => (Self::Raise, text == "raise"),
            _ => {
                let repr = runtime.repr(&value)?;
                return Err(PyError::value_error(format!(
                    "clipmode must be one of 'clip', 'raise', or 'wrap' (got {repr})"
                )));
            }
        };
        if !exact {
            return Err(PyError::value_error(
                "Use one of 'clip', 'raise', or 'wrap' for clip mode",
            ));
        }
        Ok(mode)
    }
}

/// Resolve positions against `size` under `mode`, with NumPy's `take` error text for
/// `raise`. `wrap` reduces modulo `size`; `clip` clamps into `[0, size)`.
pub(in crate::python) fn flat_positions<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    indices: &Array<'s>,
    size: usize,
    axis: Option<usize>,
    mode: ClipMode,
) -> PyResult<'s, Vec<usize>> {
    let values = read_elements::<i64>(runtime, indices)?;
    let signed = size as i64;
    values
        .into_iter()
        .map(|value| match mode {
            ClipMode::Wrap => Ok(value.rem_euclid(signed.max(1)) as usize),
            ClipMode::Clip => Ok(value.clamp(0, (signed - 1).max(0)) as usize),
            ClipMode::Raise => {
                let position = if value < 0 { value + signed } else { value };
                if (0..signed).contains(&position) {
                    Ok(position as usize)
                } else {
                    Err(index_error(match axis {
                        Some(axis) => format!(
                            "index {value} is out of bounds for axis {axis} with size {size}"
                        ),
                        None => format!("index {value} is out of bounds for size {size}"),
                    }))
                }
            }
        })
        .collect()
}

/// An int64 index array from any integer array-like.
pub(in crate::python) fn index_array<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    value: PyValue<'s>,
) -> PyResult<'s, Array<'s>> {
    let array = convert::array_from_python(runtime, value, None, false)?;
    if array.size() == 0 {
        return convert::cast_array(runtime, &array, DType::INT64, false);
    }
    match array.dtype.category() {
        Category::Signed | Category::Unsigned | Category::Bool => {
            convert::cast_array(runtime, &array, DType::INT64, false)
        }
        _ => Err(PyError::type_error(
            "Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'",
        )),
    }
}
