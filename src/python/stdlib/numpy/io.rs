//! Raw element bytes for `.npy`/`.npz` persistence: `ndarray.tobytes`, `ndarray.byteswap`, and
//! `np.frombuffer`.
//!
//! The frozen `numpy.lib.format` and `numpy.lib.npyio` modules build the file formats and text
//! I/O on these functions, so files go through the simulated `open` and the virtual filesystem.
//! Storage is little-endian, and a big-endian dtype's elements are byte-swapped as they cross
//! this boundary (see [`super::dtype`]). Object arrays have no byte representation.
//!
//! One divergence is deliberate: `frombuffer` copies the buffer, because simulated arrays cannot
//! share storage with a Python `bytearray`. The result is read-only, so writes that NumPy would
//! pass through to the `bytearray` fail loudly instead of silently diverging.

use super::super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyArrayBuffer, PyArrayDataMut,
    PyError, PyResult, PyRuntime, PyValue,
};
use super::args::{self, Signature};
use super::array::{self, Array};
use super::dtype::{Category, DType, Kind};
use super::layout::{self, Order};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_io",
    functions: FUNCTIONS,
    values: &[],
};

static FUNCTIONS: &[FunctionDef] = &[FunctionDef {
    module: "numpy",
    name: "frombuffer",
    call: frombuffer,
}];

/// Methods this area installs on `numpy.ndarray`.
pub(in crate::python) static ARRAY_METHODS: NativeTypeDef = NativeTypeDef {
    name: "numpy.ndarray",
    methods: &[
        MethodDef {
            type_name: "numpy.ndarray",
            name: "tobytes",
            call: method_tobytes,
        },
        MethodDef {
            type_name: "numpy.ndarray",
            name: "byteswap",
            call: method_byteswap,
        },
    ],
    getters: &[],
};

/// Bytes that reverse together when an element changes byte order: the whole element, or each
/// part of a complex number.
fn swap_unit(dtype: DType) -> usize {
    match dtype.category() {
        Category::Complex => dtype.itemsize() / 2,
        _ => dtype.itemsize(),
    }
}

/// Reverse the byte order of every element in packed `bytes`.
fn swap_elements(bytes: &mut [u8], dtype: DType) {
    let unit = swap_unit(dtype);
    if unit > 1 {
        for chunk in bytes.chunks_exact_mut(unit) {
            chunk.reverse();
        }
    }
}

/// `a.tobytes(order='C')`: the elements' bytes in C order, or Fortran order for `F` (and for
/// `A` when the array is Fortran-contiguous), in the dtype's byte order.
fn method_tobytes(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("tobytes", &["order"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    if array.dtype.kind() == Kind::Object {
        // NumPy returns the elements' addresses, which mean nothing outside the process.
        return Err(PyError::unsupported(
            "ndarray.tobytes of an object array is not supported",
        ));
    }
    let order = match Order::parse(runtime, bound.value("order"), Order::C)? {
        Order::K => Order::C,
        order => order,
    };
    let axes = layout::axes_like(&array, order, array.ndim());
    let PyArrayBuffer::Bytes(mut bytes) =
        array::contiguous_buffer(runtime, &layout::reading_order(&array, &axes))?
    else {
        unreachable!("non-object arrays have byte storage")
    };
    if !array.dtype.is_native() {
        swap_elements(&mut bytes, array.dtype);
    }
    runtime.new_bytes(bytes)
}

/// `a.byteswap(inplace=False)`: reverse the bytes of every element, keeping the dtype, so the
/// values change. Object arrays hold references, which NumPy leaves alone.
fn method_byteswap(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("byteswap", &["inplace"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    let inplace = args::flag(runtime, bound.value("inplace"), false)?;
    if array.dtype.kind() == Kind::Str {
        return Err(PyError::unsupported(
            "ndarray.byteswap of a str array is not supported",
        ));
    }
    if inplace {
        if array.dtype.kind() != Kind::Object {
            runtime.charge_cpu(array.size() as u64 + 1)?;
            let itemsize = array.itemsize();
            runtime.write_array(array.handle, &mut |target| {
                let PyArrayDataMut::Bytes(bytes) = target.data else {
                    unreachable!("numeric arrays have byte storage")
                };
                for offset in array.offsets() {
                    swap_elements(&mut bytes[offset..offset + itemsize], array.dtype);
                }
                Ok(())
            })?;
        }
        return Ok(receiver);
    }
    let axes = layout::axes_like(&array, Order::A, array.ndim());
    let mut buffer = array::contiguous_buffer(runtime, &layout::reading_order(&array, &axes))?;
    if let PyArrayBuffer::Bytes(bytes) = &mut buffer {
        swap_elements(bytes, array.dtype);
    }
    let shape = array.shape().to_vec();
    Ok(layout::new_array(runtime, buffer, array.dtype, shape, &axes)?.value())
}

/// `np.frombuffer(buffer, dtype=float, count=-1, offset=0)`: a read-only 1-d array of the
/// elements packed in a `bytes` or `bytearray`, read in the dtype's byte order.
fn frombuffer(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("frombuffer", &["buffer", "dtype", "count", "offset"], 1)
            .keyword_only(&["like"]);
    let bound = SIGNATURE.bind(&args)?;
    let buffer = bound.required("buffer");
    let Some(data) = runtime.bytes_value(&buffer)? else {
        return Err(PyError::type_error(format!(
            "a bytes-like object is required, not '{}'",
            runtime.type_name(&buffer)?
        )));
    };
    let dtype = args::optional_dtype(runtime, bound.value("dtype"))?.unwrap_or(DType::FLOAT64);
    if dtype.kind() == Kind::Object {
        return Err(PyError::value_error(
            "cannot create an OBJECT array from memory buffer",
        ));
    }
    let itemsize = dtype.itemsize();
    if itemsize == 0 {
        return Err(PyError::value_error("itemsize cannot be zero in type"));
    }
    let count = args::optional_int(runtime, bound.value("count"))?.unwrap_or(-1);
    let offset = args::optional_int(runtime, bound.value("offset"))?.unwrap_or(0);
    let offset = usize::try_from(offset)
        .ok()
        .filter(|offset| *offset <= data.len())
        .ok_or_else(|| {
            PyError::value_error(format!(
                "offset must be non-negative and no greater than buffer length ({})",
                data.len()
            ))
        })?;
    let available = data.len() - offset;
    let count = match usize::try_from(count) {
        Err(_) if available % itemsize != 0 => {
            return Err(PyError::value_error(
                "buffer size must be a multiple of element size",
            ))
        }
        Err(_) => available / itemsize,
        Ok(count) if count.saturating_mul(itemsize) > available => {
            return Err(PyError::value_error(
                "buffer is smaller than requested size",
            ))
        }
        Ok(count) => count,
    };
    array::reserve_elements(runtime, dtype, count)?;
    runtime.charge_cpu(count as u64 + 1)?;
    let mut bytes = data[offset..offset + count * itemsize].to_vec();
    if !dtype.is_native() {
        swap_elements(&mut bytes, dtype);
    }
    if dtype.kind() == Kind::Bool {
        // Bool storage holds 0 or 1; NumPy keeps other bytes, which read as True.
        for byte in &mut bytes {
            *byte = u8::from(*byte != 0);
        }
    }
    let array = array::new_array(runtime, PyArrayBuffer::Bytes(bytes), dtype, vec![count])?;
    runtime.set_array_writeable(array.handle, false)?;
    Ok(array.value())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_parts_swap_separately() {
        let mut bytes = [1, 2, 3, 4, 5, 6, 7, 8];
        swap_elements(&mut bytes, DType::COMPLEX64);
        assert_eq!(bytes, [4, 3, 2, 1, 8, 7, 6, 5]);
        swap_elements(&mut bytes, DType::FLOAT64);
        assert_eq!(bytes, [5, 6, 7, 8, 1, 2, 3, 4]);
    }
}
