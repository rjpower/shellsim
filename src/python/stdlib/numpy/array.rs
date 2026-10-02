//! Checked ndarray handles, stride iteration, broadcasting, and element movement.
//!
//! Arrays live in VM-owned storage ([`PyArrayBuffer`]); this module reads them only through
//! the runtime's closure borrows, which cannot re-enter Python. Offsets and strides are always
//! bytes. Object arrays store one heap reference per 8-byte slot, so `offset / 8` addresses
//! them; read callbacks turn a reference into a handle through the lent `PyRefs`, and write
//! callbacks store a handle through the lent `Builder`.
//!
//! Every function that allocates element storage reserves memory first and charges CPU in
//! proportion to the elements it touches. Costs are deliberately coarse.

use super::super::super::native::{
    PyArray, PyArrayBuffer, PyArrayData, PyArrayDataMut, PyArrayDtype, PyArrayRef, PyArrayView,
    PyError, PyRefs, PyResult, PyRuntime, PyValue, PyValueCast,
};
use super::super::super::Value;
use super::dtype::{DType, Kind};

/// Largest rank NumPy accepts.
pub(in crate::python) const MAX_DIMS: usize = 64;

/// Modeled bytes per object-array slot, matching the heap's per-value charge.
const VALUE_SLOT_BYTES: usize = 24;

/// A checked ndarray handle plus a snapshot of its view metadata.
#[derive(Clone, Debug)]
pub(in crate::python) struct Array<'s> {
    pub handle: PyArray<'s>,
    pub view: PyArrayView,
    pub dtype: DType,
}

impl<'s> Array<'s> {
    /// Check that `value` is an ndarray and snapshot its metadata.
    pub(in crate::python) fn from_value(
        runtime: &dyn PyRuntime<'s>,
        value: PyValue<'s>,
    ) -> PyResult<'s, Self> {
        let handle = value.cast::<PyArray<'s>>(runtime)?;
        Self::from_handle(runtime, handle)
    }

    pub(in crate::python) fn from_handle(
        runtime: &dyn PyRuntime<'s>,
        handle: PyArray<'s>,
    ) -> PyResult<'s, Self> {
        let view = runtime.array_view(handle)?;
        let dtype = DType::from_storage(view.dtype);
        Ok(Self {
            handle,
            view,
            dtype,
        })
    }

    /// The Python value for this array.
    pub(in crate::python) fn value(&self) -> PyValue<'s> {
        self.handle.value()
    }

    pub(in crate::python) fn shape(&self) -> &[usize] {
        &self.view.shape
    }

    pub(in crate::python) fn strides(&self) -> &[isize] {
        &self.view.strides
    }

    pub(in crate::python) fn ndim(&self) -> usize {
        self.view.shape.len()
    }

    /// Element count. Views are validated at creation, so this cannot overflow.
    pub(in crate::python) fn size(&self) -> usize {
        self.view.shape.iter().product()
    }

    pub(in crate::python) fn itemsize(&self) -> usize {
        self.dtype.itemsize()
    }

    pub(in crate::python) fn is_c_contiguous(&self) -> bool {
        is_c_contiguous(&self.view.shape, &self.view.strides, self.itemsize())
    }

    /// Byte offsets of every element in C order.
    pub(in crate::python) fn offsets(&self) -> Offsets {
        Offsets::new(&self.view.shape, &self.view.strides, self.view.offset)
    }

    /// Byte offset of the element at `index`, which must be in bounds.
    pub(in crate::python) fn offset_of(&self, index: &[usize]) -> usize {
        element_offset(&self.view, index)
    }
}

/// Whether `strides` describe a C-contiguous layout of `shape`.
pub(in crate::python) fn is_c_contiguous(
    shape: &[usize],
    strides: &[isize],
    itemsize: usize,
) -> bool {
    if shape.contains(&0) {
        return true;
    }
    let mut expected = itemsize as isize;
    for (dimension, stride) in shape.iter().zip(strides).rev() {
        if *dimension != 1 && *stride != expected {
            return false;
        }
        expected = expected.saturating_mul(*dimension as isize);
    }
    true
}

/// C-order byte strides for `shape`.
pub(in crate::python) fn contiguous_strides(shape: &[usize], itemsize: usize) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut stride = itemsize;
    for (axis, dimension) in shape.iter().enumerate().rev() {
        strides[axis] = stride as isize;
        stride = stride.saturating_mul((*dimension).max(1));
    }
    strides
}

/// Byte offset of one in-bounds element.
pub(in crate::python) fn element_offset(view: &PyArrayView, index: &[usize]) -> usize {
    let offset = index
        .iter()
        .zip(&view.strides)
        .fold(view.offset as isize, |offset, (index, stride)| {
            offset + *index as isize * stride
        });
    offset as usize
}

/// Element count of `shape`, checked against overflow and NumPy's rank limit.
pub(in crate::python) fn element_count<'s>(shape: &[usize]) -> PyResult<'s, usize> {
    if shape.len() > MAX_DIMS {
        return Err(PyError::value_error(format!(
            "maximum supported dimension for an ndarray is currently {MAX_DIMS}, found {}",
            shape.len()
        )));
    }
    shape
        .iter()
        .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
        .ok_or_else(|| PyError::value_error("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."))
}

/// Iterator over the byte offsets of a strided view in C order.
pub(in crate::python) struct Offsets {
    shape: Vec<usize>,
    strides: Vec<isize>,
    index: Vec<usize>,
    next: isize,
    remaining: usize,
}

impl Offsets {
    pub(in crate::python) fn new(shape: &[usize], strides: &[isize], offset: usize) -> Self {
        let remaining = shape.iter().product();
        Self {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            index: vec![0; shape.len()],
            next: offset as isize,
            remaining,
        }
    }
}

impl Iterator for Offsets {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let current = self.next as usize;
        for axis in (0..self.shape.len()).rev() {
            self.index[axis] += 1;
            self.next += self.strides[axis];
            if self.index[axis] < self.shape[axis] {
                break;
            }
            self.next -= self.strides[axis] * self.shape[axis] as isize;
            self.index[axis] = 0;
        }
        Some(current)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for Offsets {}

/// The broadcast shape of several operand shapes, or NumPy's error text.
pub(in crate::python) fn broadcast_shapes<'s>(shapes: &[&[usize]]) -> PyResult<'s, Vec<usize>> {
    let rank = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    let mut result = vec![1usize; rank];
    for shape in shapes {
        for (axis, dimension) in shape.iter().enumerate() {
            let target = &mut result[rank - shape.len() + axis];
            if *target == 1 {
                *target = *dimension;
            } else if *dimension != 1 && *dimension != *target {
                let rendered = shapes
                    .iter()
                    .map(|shape| format_shape(shape))
                    .collect::<Vec<_>>()
                    .join(" ");
                return Err(PyError::value_error(format!(
                    "operands could not be broadcast together with shapes {rendered} "
                )));
            }
        }
    }
    Ok(result)
}

/// Strides that read `view` as if it had `shape`, repeating broadcast axes with stride 0.
pub(in crate::python) fn broadcast_strides<'s>(
    view: &PyArrayView,
    shape: &[usize],
) -> PyResult<'s, Vec<isize>> {
    let extra = shape.len().checked_sub(view.shape.len()).ok_or_else(|| {
        PyError::value_error(format!(
            "could not broadcast input array from shape {} into shape {}",
            format_shape(&view.shape),
            format_shape(shape)
        ))
    })?;
    let mut strides = vec![0isize; shape.len()];
    for (axis, (dimension, stride)) in view.shape.iter().zip(&view.strides).enumerate() {
        let target = shape[extra + axis];
        if *dimension == target {
            strides[extra + axis] = if target == 1 { 0 } else { *stride };
        } else if *dimension != 1 {
            return Err(PyError::value_error(format!(
                "could not broadcast input array from shape {} into shape {}",
                format_shape(&view.shape),
                format_shape(shape)
            )));
        }
    }
    Ok(strides)
}

/// NumPy's shape text: `(3,)`, `(2,3)`, `()`.
pub(in crate::python) fn format_shape(shape: &[usize]) -> String {
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

/// Reserve memory for `count` elements of `dtype` before allocating them.
pub(in crate::python) fn reserve_elements<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    dtype: DType,
    count: usize,
) -> PyResult<'s, usize> {
    let per_element = if dtype.kind() == Kind::Object {
        VALUE_SLOT_BYTES
    } else {
        dtype.itemsize()
    };
    let bytes = count.checked_mul(per_element).ok_or_else(|| {
        PyError::exception(
            "MemoryError",
            format!("Unable to allocate array with {count} elements"),
        )
    })?;
    runtime.reserve_memory(bytes)?;
    Ok(bytes)
}

/// Allocate zero-filled storage for `count` elements, after reserving memory for it.
///
/// Object storage is filled with `None`, as `np.empty(n, dtype=object)` is.
pub(in crate::python) fn zeroed_buffer<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    dtype: DType,
    count: usize,
) -> PyResult<'s, PyArrayBuffer<'s>> {
    let bytes = reserve_elements(runtime, dtype, count)?;
    runtime.charge_cpu(count as u64 / 8 + 1)?;
    Ok(if dtype.kind() == Kind::Object {
        PyArrayBuffer::Values(vec![Value::None; count])
    } else {
        PyArrayBuffer::Bytes(vec![0; bytes])
    })
}

/// Wrap owned storage, which holds the elements in C order, in a new C-contiguous array.
pub(in crate::python) fn new_array<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    buffer: PyArrayBuffer<'s>,
    dtype: DType,
    shape: Vec<usize>,
) -> PyResult<'s, Array<'s>> {
    let axes = (0..shape.len()).collect::<Vec<_>>();
    super::layout::new_array(runtime, buffer, dtype, shape, &axes)
}

/// Wrap owned storage in a new array whose elements sit at `strides`; see
/// [`super::layout`] for building strides from a memory order.
pub(in crate::python) fn new_array_with_strides<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    buffer: PyArrayBuffer<'s>,
    dtype: DType,
    shape: Vec<usize>,
    strides: Vec<isize>,
) -> PyResult<'s, Array<'s>> {
    element_count(&shape)?;
    let value = runtime.new_array(buffer, dtype.storage(), shape, strides)?;
    Array::from_value(runtime, value)
}

/// Create a view sharing `base`'s storage.
pub(in crate::python) fn new_view<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    base: &Array<'s>,
    dtype: DType,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
) -> PyResult<'s, Array<'s>> {
    element_count(&shape)?;
    let view = PyArrayView {
        dtype: dtype.storage(),
        shape,
        strides,
        offset,
        writeable: base.view.writeable,
    };
    let value = runtime.new_array_view(base.handle, view)?;
    Array::from_value(runtime, value)
}

/// Copy the elements at `offsets` of one borrowed array into `output`, which must have the
/// same storage kind. Bytes are appended; object references become handles through `refs`.
pub(in crate::python) fn gather_into<'s>(
    refs: &dyn PyRefs<'s>,
    source: &PyArrayRef<'_>,
    offsets: impl Iterator<Item = usize>,
    output: &mut PyArrayBuffer<'s>,
) {
    let itemsize = source.view.dtype.itemsize();
    match (&source.data, output) {
        (PyArrayData::Bytes(bytes), PyArrayBuffer::Bytes(output)) => {
            for offset in offsets {
                output.extend_from_slice(&bytes[offset..offset + itemsize]);
            }
        }
        (PyArrayData::Values(values), PyArrayBuffer::Values(output)) => {
            for offset in offsets {
                output.push(refs.handle(&values[offset / PyArrayDtype::VALUE_ITEMSIZE]));
            }
        }
        _ => unreachable!("gather copies between storages of one element kind"),
    }
}

/// An empty buffer of the right storage kind with room for `count` elements.
pub(in crate::python) fn buffer_with_capacity<'s>(dtype: DType, count: usize) -> PyArrayBuffer<'s> {
    if dtype.kind() == Kind::Object {
        PyArrayBuffer::Values(Vec::with_capacity(count))
    } else {
        PyArrayBuffer::Bytes(Vec::with_capacity(count.saturating_mul(dtype.itemsize())))
    }
}

/// Copy `array`'s elements, in C order, into new contiguous storage of the same dtype.
pub(in crate::python) fn contiguous_buffer<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
) -> PyResult<'s, PyArrayBuffer<'s>> {
    let count = array.size();
    reserve_elements(runtime, array.dtype, count)?;
    runtime.charge_cpu(count as u64 + 1)?;
    let mut output = buffer_with_capacity(array.dtype, count);
    runtime.read_arrays(&[array.handle], &mut |refs, arrays| {
        gather_into(refs, &arrays[0], array.offsets(), &mut output);
        Ok(())
    })?;
    Ok(output)
}

/// A C-contiguous copy of `array`.
pub(in crate::python) fn copy_array<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
) -> PyResult<'s, Array<'s>> {
    let buffer = contiguous_buffer(runtime, array)?;
    new_array(runtime, buffer, array.dtype, array.shape().to_vec())
}

/// `array` as one dimension in C order: a view when the layout allows, otherwise a copy.
pub(in crate::python) fn ravel<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
) -> PyResult<'s, Array<'s>> {
    let size = array.size();
    if array.is_c_contiguous() {
        let stride = array.itemsize() as isize;
        return new_view(
            runtime,
            array,
            array.dtype,
            vec![size],
            vec![stride],
            array.view.offset,
        );
    }
    let buffer = contiguous_buffer(runtime, array)?;
    new_array(runtime, buffer, array.dtype, vec![size])
}

/// Store `source`, a contiguous buffer of `destination.dtype` elements in C order, into the
/// elements of `destination` at `offsets`. The source is fully materialized first, so aliasing
/// between source and destination storage is harmless.
pub(in crate::python) fn scatter<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    destination: &Array<'s>,
    offsets: &[usize],
    source: &PyArrayBuffer<'s>,
) -> PyResult<'s, ()> {
    runtime.charge_cpu(offsets.len() as u64 + 1)?;
    let itemsize = destination.itemsize();
    runtime.write_array(destination.handle, &mut |builder, target| {
        match (target.data, source) {
            (PyArrayDataMut::Bytes(bytes), PyArrayBuffer::Bytes(source)) => {
                for (index, offset) in offsets.iter().enumerate() {
                    bytes[*offset..*offset + itemsize]
                        .copy_from_slice(&source[index * itemsize..(index + 1) * itemsize]);
                }
            }
            (PyArrayDataMut::Values(values), PyArrayBuffer::Values(source)) => {
                for (index, offset) in offsets.iter().enumerate() {
                    values[*offset / PyArrayDtype::VALUE_ITEMSIZE] = builder.store(source[index]);
                }
            }
            _ => unreachable!("scatter copies between storages of one element kind"),
        }
        Ok(())
    })
}

/// Store `source` into every element of `destination`, broadcasting `source`'s shape.
pub(in crate::python) fn assign<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    destination: &Array<'s>,
    source: &Array<'s>,
) -> PyResult<'s, ()> {
    let buffer = broadcast_buffer(runtime, source, destination.dtype, destination.shape())?;
    let offsets = destination.offsets().collect::<Vec<_>>();
    scatter(runtime, destination, &offsets, &buffer)
}

/// The number of leading length-1 axes of `source` beyond `ndim`. NumPy drops them when it
/// assigns an array, so `a[...] = b[None]` stores `b`.
fn surplus_unit_axes(source: &[usize], ndim: usize) -> usize {
    source
        .iter()
        .take(source.len().saturating_sub(ndim))
        .take_while(|dimension| **dimension == 1)
        .count()
}

/// Whether assigning an array of shape `source` to `target` elements broadcasts, as NumPy's
/// `PyArray_AssignArray` decides after dropping surplus leading length-1 axes.
pub(in crate::python) fn assignable(source: &[usize], target: &[usize]) -> bool {
    let source = &source[surplus_unit_axes(source, target.len())..];
    source.len() <= target.len()
        && source
            .iter()
            .rev()
            .zip(target.iter().rev())
            .all(|(dimension, target)| dimension == target || *dimension == 1)
}

/// `source` cast to `dtype` and broadcast to `shape`, as a C-order buffer. Leading length-1
/// axes beyond `shape`'s rank are dropped first, as NumPy does when assigning.
pub(in crate::python) fn broadcast_buffer<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    source: &Array<'s>,
    dtype: DType,
    shape: &[usize],
) -> PyResult<'s, PyArrayBuffer<'s>> {
    let mut source = if source.dtype == dtype {
        source.clone()
    } else {
        super::convert::cast_array(runtime, source, dtype, false)?
    };
    let surplus = surplus_unit_axes(&source.view.shape, shape.len());
    source.view.shape.drain(..surplus);
    source.view.strides.drain(..surplus);
    let strides = broadcast_strides(&source.view, shape)?;
    let count = element_count(shape)?;
    reserve_elements(runtime, dtype, count)?;
    runtime.charge_cpu(count as u64 + 1)?;
    let mut buffer = buffer_with_capacity(dtype, count);
    runtime.read_arrays(&[source.handle], &mut |refs, arrays| {
        let offsets = Offsets::new(shape, &strides, source.view.offset);
        gather_into(refs, &arrays[0], offsets, &mut buffer);
        Ok(())
    })?;
    Ok(buffer)
}

/// Read every element of a numeric array, in C order, as `T`.
pub(in crate::python) fn read_elements<'s, T: super::element::Element>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
) -> PyResult<'s, Vec<T>> {
    debug_assert_eq!(T::SIZE, array.itemsize());
    let count = array.size();
    runtime.reserve_memory(count.saturating_mul(std::mem::size_of::<T>()))?;
    runtime.charge_cpu(count as u64 + 1)?;
    let mut output = Vec::with_capacity(count);
    runtime.read_arrays(&[array.handle], &mut |_, arrays| {
        let PyArrayData::Bytes(bytes) = arrays[0].data else {
            return Err(PyError::runtime_error("numeric array has object storage"));
        };
        output.extend(array.offsets().map(|offset| T::read(&bytes[offset..])));
        Ok(())
    })?;
    Ok(output)
}

/// Snapshot the references of an object array in C order. Callers then run Python code per
/// element without holding a storage borrow; later mutation of the array does not affect the
/// snapshot, as in NumPy's buffered iteration.
pub(in crate::python) fn read_objects<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
) -> PyResult<'s, Vec<PyValue<'s>>> {
    let count = array.size();
    runtime.reserve_memory(count.saturating_mul(VALUE_SLOT_BYTES))?;
    runtime.charge_cpu(count as u64 + 1)?;
    let mut output = Vec::with_capacity(count);
    runtime.read_arrays(&[array.handle], &mut |refs, arrays| {
        let PyArrayData::Values(values) = arrays[0].data else {
            return Err(PyError::runtime_error("object array has byte storage"));
        };
        output.extend(
            array
                .offsets()
                .map(|offset| refs.handle(&values[offset / PyArrayDtype::VALUE_ITEMSIZE])),
        );
        Ok(())
    })?;
    Ok(output)
}

/// Pack typed elements into a little-endian byte buffer.
pub(in crate::python) fn pack_elements<T: super::element::Element>(values: &[T]) -> Vec<u8> {
    let mut bytes = vec![0u8; values.len() * T::SIZE];
    for (value, chunk) in values.iter().zip(bytes.chunks_exact_mut(T::SIZE)) {
        value.write(chunk);
    }
    bytes
}

/// A new array from typed elements; memory must already be reserved by the caller's kernel
/// or is reserved here.
pub(in crate::python) fn array_from_elements<'s, T: super::element::Element>(
    runtime: &mut dyn PyRuntime<'s>,
    dtype: DType,
    shape: Vec<usize>,
    values: &[T],
) -> PyResult<'s, Array<'s>> {
    debug_assert_eq!(T::SIZE, dtype.itemsize());
    reserve_elements(runtime, dtype, values.len())?;
    new_array(
        runtime,
        PyArrayBuffer::Bytes(pack_elements(values)),
        dtype,
        shape,
    )
}

/// A new Fortran-ordered array of `shape` holding `values` in column-major order, the layout a
/// LAPACK-backed routine computes its result in.
pub(in crate::python) fn fortran_array_from_elements<'s, T: super::element::Element>(
    runtime: &mut dyn PyRuntime<'s>,
    dtype: DType,
    shape: Vec<usize>,
    values: &[T],
) -> PyResult<'s, Array<'s>> {
    debug_assert_eq!(T::SIZE, dtype.itemsize());
    reserve_elements(runtime, dtype, values.len())?;
    let axes = (0..shape.len()).rev().collect::<Vec<_>>();
    let buffer = PyArrayBuffer::Bytes(pack_elements(values));
    super::layout::new_array(runtime, buffer, dtype, shape, &axes)
}

/// Normalize a possibly negative axis for an array of rank `ndim`, with NumPy's `AxisError`.
pub(in crate::python) fn normalize_axis<'s>(axis: i64, ndim: usize) -> PyResult<'s, usize> {
    let rank = ndim as i64;
    let normalized = if axis < 0 { axis + rank } else { axis };
    if (0..rank).contains(&normalized) {
        Ok(normalized as usize)
    } else {
        Err(PyError::exception(
            "AxisError",
            format!("axis {axis} is out of bounds for array of dimension {ndim}"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_walk_c_order_with_negative_and_zero_strides() {
        let offsets = Offsets::new(&[2, 3], &[24, 8], 0).collect::<Vec<_>>();
        assert_eq!(offsets, [0, 8, 16, 24, 32, 40]);
        let reversed = Offsets::new(&[3], &[-8], 16).collect::<Vec<_>>();
        assert_eq!(reversed, [16, 8, 0]);
        let broadcast = Offsets::new(&[2, 2], &[0, 8], 0).collect::<Vec<_>>();
        assert_eq!(broadcast, [0, 8, 0, 8]);
        assert_eq!(Offsets::new(&[0, 3], &[24, 8], 0).count(), 0);
        assert_eq!(Offsets::new(&[], &[], 40).collect::<Vec<_>>(), [40]);
    }

    #[test]
    fn broadcasting_matches_numpy_rules() {
        assert_eq!(broadcast_shapes(&[&[3, 1], &[4]]).unwrap(), [3, 4]);
        assert_eq!(broadcast_shapes(&[&[], &[2]]).unwrap(), [2]);
        let error = broadcast_shapes(&[&[3], &[4]]).unwrap_err();
        assert_eq!(
            error.message,
            "operands could not be broadcast together with shapes (3,) (4,) "
        );
    }

    #[test]
    fn contiguity_ignores_unit_dimensions() {
        assert!(is_c_contiguous(&[2, 1, 3], &[24, 999, 8], 8));
        assert!(!is_c_contiguous(&[2, 3], &[8, 16], 8));
        assert_eq!(contiguous_strides(&[2, 3], 4), [12, 4]);
    }
}
