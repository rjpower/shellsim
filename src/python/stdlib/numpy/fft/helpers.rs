//! Sample-frequency helpers: `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`.
//!
//! These do not touch [`super::kernel`] at all — they are plain array constructors and a
//! cyclic-permutation gather, listed here because NumPy groups them with the transforms.

use crate::python::native::{CallArgs, PyError, PyResult, PyRuntime, PyValue};
use crate::python::stdlib::numpy::args::Signature;
use crate::python::stdlib::numpy::array::Array;
use crate::python::stdlib::numpy::dtype::DType;
use crate::python::stdlib::numpy::{args, array, convert, scalar};

// `d=1.0` and an ignored `device=` keyword (added for array-API compatibility) are identical
// for both functions; only the name in error messages differs.
static FFTFREQ: Signature = Signature::new("fftfreq", &["n", "d"], 1).keyword_only(&["device"]);
static RFFTFREQ: Signature = Signature::new("rfftfreq", &["n", "d"], 1).keyword_only(&["device"]);

pub(super) fn fftfreq(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    build_freq(runtime, args, &FFTFREQ, false)
}

pub(super) fn rfftfreq(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    build_freq(runtime, args, &RFFTFREQ, true)
}

fn build_freq(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    signature: &'static Signature,
    one_sided: bool,
) -> PyResult {
    let bound = signature.bind(&args)?;
    let n = resolve_freq_n(runtime, bound.required("n"))?;
    let d = match bound.value("d") {
        Some(value) => args::float_arg(runtime, &value)?,
        None => 1.0,
    };
    let count = if one_sided { n / 2 + 1 } else { n };
    array::reserve_elements(runtime, DType::FLOAT64, count)?;
    runtime.charge_cpu(count as u64 + 1)?;
    let scale = 1.0 / (n as f64 * d);
    let values: Vec<f64> = if one_sided {
        (0..count).map(|k| k as f64 * scale).collect()
    } else {
        let half = n.div_ceil(2);
        (0..count)
            .map(|k| {
                let signed = if k < half {
                    k as i64
                } else {
                    k as i64 - n as i64
                };
                signed as f64 * scale
            })
            .collect()
    };
    Ok(array::array_from_elements(runtime, DType::FLOAT64, vec![count], &values)?.value())
}

/// `n` must be an integer type (NumPy's own message, distinct from the usual
/// "cannot be interpreted as an integer" `TypeError` other `n=`/`axis=` arguments raise).
/// Negative `n` reuses the message array construction gives a negative size. `n=0` reproduces
/// NumPy's own `ZeroDivisionError`: it computes the `1/(n*d)` scale before knowing there are no
/// elements to apply it to, so the division fails even though the result would otherwise be the
/// empty array.
fn resolve_freq_n(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<usize> {
    let n = if let Some(n) = runtime.int_value(&value) {
        n
    } else if let Some((dtype, number)) = scalar::unbox_number(runtime, &value) {
        if scalar::is_index_dtype(dtype) {
            number.wrapping_i64()
        } else {
            return Err(PyError::value_error("n should be an integer"));
        }
    } else {
        return Err(PyError::value_error("n should be an integer"));
    };
    if n < 0 {
        return Err(PyError::value_error("negative dimensions are not allowed"));
    }
    if n == 0 {
        return Err(PyError::zero_division_error("division by zero"));
    }
    Ok(n as usize)
}

static FFTSHIFT: Signature = Signature::new("fftshift", &["x", "axes"], 1);
static IFFTSHIFT: Signature = Signature::new("ifftshift", &["x", "axes"], 1);

pub(super) fn fftshift(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    shift_call(runtime, args, &FFTSHIFT, 1)
}

pub(super) fn ifftshift(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    shift_call(runtime, args, &IFFTSHIFT, -1)
}

fn shift_call(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    signature: &'static Signature,
    sign: i64,
) -> PyResult {
    let bound = signature.bind(&args)?;
    let source = convert::as_array(runtime, bound.required("x"))?;
    let axes = args::axes(runtime, bound.get("axes"), source.ndim())?;
    let axis_list: Vec<usize> = match axes {
        args::Axes::All => (0..source.ndim()).collect(),
        args::Axes::Some(list) => list,
    };
    Ok(shift(runtime, &source, &axis_list, sign)?.value())
}

/// `np.roll(x, sign * (shape[axis] // 2), axis)` for every axis in `axes` at once: a cyclic
/// gather, since the wraparound rules it out as a plain view. `fftshift` is `sign=1`;
/// `ifftshift` is `sign=-1`, which always exactly inverts it (`np.roll` composes by addition
/// modulo the axis length, so negating the shift is its own inverse regardless of parity).
fn shift(
    runtime: &mut dyn PyRuntime,
    source: &Array,
    axes: &[usize],
    sign: i64,
) -> PyResult<Array> {
    let shape = source.shape().to_vec();
    let count = array::element_count(&shape)?;
    array::reserve_elements(runtime, source.dtype, count)?;
    runtime.charge_cpu(count as u64 + 1)?;
    let shifts: Vec<i64> = axes
        .iter()
        .map(|&axis| sign * (shape[axis] / 2) as i64)
        .collect();
    let mut offsets = Vec::with_capacity(count);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..count {
        let mut source_index = index.clone();
        for (&axis, &axis_shift) in axes.iter().zip(&shifts) {
            let n = shape[axis] as i64;
            source_index[axis] = (((index[axis] as i64 - axis_shift) % n + n) % n) as usize;
        }
        offsets.push(source.offset_of(&source_index));
        for d in (0..shape.len()).rev() {
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    let mut buffer = array::buffer_with_capacity(source.dtype, count);
    runtime.read_arrays(&[source.handle], &mut |arrays| {
        array::gather_into(&arrays[0], offsets.iter().copied(), &mut buffer);
        Ok(())
    })?;
    array::new_array(runtime, buffer, source.dtype, shape)
}
