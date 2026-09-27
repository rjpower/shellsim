//! Multi-axis transforms (`fft2`, `fftn`, `rfft2`, `rfftn`, and their inverses), built by
//! repeating [`super::transform`] (or [`super::real_inverse_transform`]) once per axis.
//!
//! A separable multi-dimensional DFT is just a stack of one-dimensional DFTs, one per axis, and
//! the order in which independent axes run does not change the numeric result (each axis's
//! transform commutes with every other axis's). The one exception is the axis a real transform
//! designates as one-sided (`axes[-1]`, matching NumPy): `rfftn` must run that axis's transform
//! *first*, while the data is still real, so trimming it to `n/2+1` bins is valid; `irfftn`
//! must run it *last*, so the other axes' inverse transforms still see the full complex
//! spectrum they need.
//!
//! `s`/`axes` resolution (including `s`'s `-1` "keep this axis's length" entries, and `axes`
//! defaulting to the trailing `len(s)` axes when `s` is given but `axes` is not) matches NumPy's
//! observed behavior; it is checked against the reference interpreter in `test_fft.py` rather
//! than documented upstream.

use crate::python::native::{CallArgs, PyError, PyKind, PyResult, PyRuntime, PyValue, PyValueCast};
use crate::python::stdlib::numpy::args::{self, Signature};
use crate::python::stdlib::numpy::convert;

static MULTI_AXIS: Signature = Signature::new("fftn", &["a", "s", "axes", "norm", "out"], 1);

/// `s=` or `axes=` must be a tuple or list of ints; unlike `n=`/`axis=` on the one-axis
/// functions, a bare int is rejected the same way Python rejects iterating over one.
fn int_sequence(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<i64>> {
    match runtime.kind(&value)? {
        PyKind::Tuple => {
            let tuple = value.cast(runtime)?;
            runtime
                .tuple_items(tuple)?
                .iter()
                .map(|item| args::index_int(runtime, item))
                .collect()
        }
        PyKind::List => {
            let list = value.cast(runtime)?;
            runtime
                .list_items(list)?
                .iter()
                .map(|item| args::index_int(runtime, item))
                .collect()
        }
        _ => Err(PyError::type_error(format!(
            "'{}' object is not iterable",
            runtime.type_name(&value)?
        ))),
    }
}

/// Resolve `s=`/`axes=` to normalized axes and their target lengths. `default_axes` is the
/// fixed default for `fft2`-style functions (`[-2, -1]`); `None` means "derive it" the way
/// `fftn` does (all axes, or the trailing `len(s)` axes when only `s` is given).
///
/// `hermitian_default`, set only for `irfftn`/`irfft2`, gives the *last* axis (the one-sided
/// axis) `irfft`'s own `2*(m-1)` default length instead of the plain "keep this axis's length"
/// default every other axis gets, matching NumPy exactly (verified against the reference).
fn resolve_axes_and_lengths(
    runtime: &mut dyn PyRuntime,
    shape: &[usize],
    s: Option<PyValue>,
    axes: Option<PyValue>,
    default_axes: Option<&[i64]>,
    hermitian_default: bool,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let ndim = shape.len();
    let raw_s = s.map(|value| int_sequence(runtime, value)).transpose()?;
    let raw_axes = axes.map(|value| int_sequence(runtime, value)).transpose()?;
    let axes_list: Vec<i64> = match (&raw_axes, default_axes, &raw_s) {
        (Some(list), ..) => list.clone(),
        (None, Some(default), _) => default.to_vec(),
        (None, None, Some(s_list)) => (-(s_list.len() as i64)..0).collect(),
        (None, None, None) => (0..ndim as i64).collect(),
    };
    let normalized_axes = axes_list
        .iter()
        .map(|&axis| super::normalize_raw_axis(axis, ndim))
        .collect::<PyResult<Vec<_>>>()?;
    let lengths = match raw_s {
        Some(list) => {
            if list.len() != normalized_axes.len() {
                return Err(PyError::value_error(
                    "Shape and axes have different lengths.",
                ));
            }
            list.iter()
                .zip(&normalized_axes)
                .map(|(&value, &axis)| match value {
                    -1 => Ok(shape[axis]),
                    value if value <= 0 => Err(PyError::value_error(format!(
                        "Invalid number of FFT data points ({value}) specified."
                    ))),
                    value => Ok(value as usize),
                })
                .collect::<PyResult<Vec<_>>>()?
        }
        None => {
            let mut lengths: Vec<usize> = normalized_axes.iter().map(|&axis| shape[axis]).collect();
            if hermitian_default {
                if let (Some(last), Some(&real_axis)) = (lengths.last_mut(), normalized_axes.last())
                {
                    let default_n = 2 * (shape[real_axis] as i64 - 1);
                    if default_n <= 0 {
                        return Err(PyError::value_error(format!(
                            "Invalid number of FFT data points ({default_n}) specified."
                        )));
                    }
                    *last = default_n as usize;
                }
            }
            lengths
        }
    };
    Ok((normalized_axes, lengths))
}

fn complex_nd(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    default_axes: Option<&[i64]>,
    inverse: bool,
    name: &'static str,
) -> PyResult {
    let bound = MULTI_AXIS.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    super::reject_non_numeric(&array, name)?;
    let (axes, lengths) = resolve_axes_and_lengths(
        runtime,
        array.shape(),
        bound.get("s"),
        bound.get("axes"),
        default_axes,
        false,
    )?;
    let norm = super::Norm::parse(runtime, bound.get("norm"))?;
    let out = super::out_array(runtime, bound.get("out"))?;
    let mut current = array;
    for (&axis, &n) in axes.iter().zip(&lengths) {
        current = super::transform(runtime, &current, n, n, axis, inverse, norm)?;
    }
    super::finish(runtime, name, current, out)
}

fn real_forward_nd(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    default_axes: Option<&[i64]>,
) -> PyResult {
    let bound = MULTI_AXIS.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let (axes, lengths) = resolve_axes_and_lengths(
        runtime,
        array.shape(),
        bound.get("s"),
        bound.get("axes"),
        default_axes,
        false,
    )?;
    let Some((&real_axis, other_axes)) = axes.split_last() else {
        return Err(PyError::exception("IndexError", "tuple index out of range"));
    };
    let (&real_n, other_lengths) = lengths
        .split_last()
        .expect("axes and lengths have equal length");
    let name = super::real_ufunc_name(real_n);
    super::reject_complex(&array, name)?;
    let norm = super::Norm::parse(runtime, bound.get("norm"))?;
    let out = super::out_array(runtime, bound.get("out"))?;
    let mut current = super::transform(
        runtime,
        &array,
        real_n,
        real_n / 2 + 1,
        real_axis,
        false,
        norm,
    )?;
    for (&axis, &n) in other_axes.iter().zip(other_lengths) {
        current = super::transform(runtime, &current, n, n, axis, false, norm)?;
    }
    super::finish(runtime, name, current, out)
}

fn real_inverse_nd(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    default_axes: Option<&[i64]>,
) -> PyResult {
    let bound = MULTI_AXIS.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    super::reject_non_numeric(&array, "irfft")?;
    let (axes, lengths) = resolve_axes_and_lengths(
        runtime,
        array.shape(),
        bound.get("s"),
        bound.get("axes"),
        default_axes,
        true,
    )?;
    let Some((&real_axis, other_axes)) = axes.split_last() else {
        return Err(PyError::exception("IndexError", "tuple index out of range"));
    };
    let (&real_n, other_lengths) = lengths
        .split_last()
        .expect("axes and lengths have equal length");
    let norm = super::Norm::parse(runtime, bound.get("norm"))?;
    let out = super::out_array(runtime, bound.get("out"))?;
    let mut current = array;
    for (&axis, &n) in other_axes.iter().zip(other_lengths) {
        current = super::transform(runtime, &current, n, n, axis, true, norm)?;
    }
    let result = super::real_inverse_transform(runtime, &current, real_n, real_axis, true, norm)?;
    super::finish(runtime, "irfft", result, out)
}

const LAST_TWO: [i64; 2] = [-2, -1];

pub(super) fn fft2(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    complex_nd(runtime, args, Some(&LAST_TWO), false, "fft")
}

pub(super) fn ifft2(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    complex_nd(runtime, args, Some(&LAST_TWO), true, "ifft")
}

pub(super) fn fftn(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    complex_nd(runtime, args, None, false, "fft")
}

pub(super) fn ifftn(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    complex_nd(runtime, args, None, true, "ifft")
}

pub(super) fn rfft2(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    real_forward_nd(runtime, args, Some(&LAST_TWO))
}

pub(super) fn rfftn(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    real_forward_nd(runtime, args, None)
}

pub(super) fn irfft2(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    real_inverse_nd(runtime, args, Some(&LAST_TWO))
}

pub(super) fn irfftn(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    real_inverse_nd(runtime, args, None)
}
