//! Native kernels for `numpy._function_base` and a few standalone math functions: piecewise
//! linear interpolation (`_compiled_interp`/`_compiled_interp_complex`, backing `numpy.interp`),
//! cross-correlation (`_correlate`, backing `numpy.correlate`/`numpy.convolve`), `divmod`,
//! `round`/`around`, and `clip`.
//!
//! Interpolation and correlation compute in a fixed working precision (`float64`, or
//! `complex128` when the samples are complex) regardless of the input dtype, then the result is
//! cast back: interpolation always produces `float64`/`complex128` (matching NumPy), while
//! correlation and convolution cast back to the NEP 50 promotion of the two operand dtypes
//! (matching NumPy's own dtype-preserving behavior). The `float64` working precision means
//! integer correlation/convolution of values outside the 53-bit exactly-representable range may
//! round differently than NumPy's native-width accumulation; this module does not special-case
//! that.
//!
//! `round` and `clip` operate through existing machinery instead of a bespoke loop: `round`
//! reads and writes one generic [`element::Number`] per element (so it preserves any numeric
//! dtype without per-dtype dispatch), and `clip(a, lo, hi)` is exactly `minimum(maximum(a, lo),
//! hi)` through the `maximum`/`minimum` ufuncs, which already implement NumPy's broadcasting,
//! promotion, and `NaN` propagation.

use super::super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyArrayBuffer, PyArrayData,
    PyError, PyNativeKind, PyResult, PyRuntime, PyValue,
};
use super::super::super::number::NumberRef;
use super::args::{self, Signature};
use super::array::{self, Array};
use super::convert;
use super::dtype::{self, Category, DType};
use super::element::{self, Number, C128};
use super::ops;
use super::ufunc;

// ---------------------------------------------------------------------------------------------
// interp
// ---------------------------------------------------------------------------------------------

/// Where `x` falls among sorted sample points `xp`, found by bisection (`O(log xp.len())`, the
/// per-point cost [`charge_interp`] charges for).
enum Position {
    /// `x` is `NaN`; the result is `NaN` regardless of the samples.
    Nan,
    /// `x < xp[0]`.
    Below,
    /// `x > xp[last]`.
    Above,
    /// `x == xp[index]`.
    Exact(usize),
    /// `x` is strictly between `xp[index]` and `xp[index + 1]`.
    Interval(usize),
}

/// Locate `x` among sorted `xp`. On a tie the index is the *last* matching sample, as in
/// NumPy: `np.interp(2.0, [1, 2, 2, 3], [3, 2, 5, 0])` is `5.0`, `fp` at the second `2`.
fn locate(xp: &[f64], x: f64) -> Position {
    if x.is_nan() {
        return Position::Nan;
    }
    if x < xp[0] {
        return Position::Below;
    }
    if x > xp[xp.len() - 1] {
        return Position::Above;
    }
    let index = xp.partition_point(|&sample| sample <= x) - 1;
    if xp[index] == x {
        Position::Exact(index)
    } else {
        Position::Interval(index)
    }
}

/// The line with `slope` through `(x0, y0)` and `(x1, y1)` evaluated at `x`, computed as NumPy
/// does: from the left end, then from the right end if that is `NaN` (an infinite slope or
/// sample), and `y0` if both are `NaN` but the samples are equal, such as two equal infinities.
/// NumPy divides by the interval width for a real slope but multiplies by its reciprocal for
/// each part of a complex one, so the caller supplies the slope.
fn interpolate(x: f64, x0: f64, x1: f64, y0: f64, y1: f64, slope: f64) -> f64 {
    let value = slope * (x - x0) + y0;
    if !value.is_nan() {
        return value;
    }
    let value = slope * (x - x1) + y1;
    if value.is_nan() && y0 == y1 {
        y0
    } else {
        value
    }
}

/// Charge CPU for one `interp` call in proportion to `x.len() * log2(xp.len())`, before the
/// bisection loop runs.
fn charge_interp(runtime: &mut dyn PyRuntime, x_len: usize, xp_len: usize) -> PyResult<()> {
    let log_xp = xp_len.max(2).ilog2() as u64 + 1;
    runtime.charge_cpu((x_len as u64).saturating_mul(log_xp) + 1)
}

/// `xp`/`fp` as one-dimensional, equal-length, nonempty arrays, with NumPy's error text for
/// each way they can fail (confirmed black-box against the reference interpreter).
/// A one-dimensional array converted to `dtype`, with NumPy's "expected 1-D" error text: the
/// short "object of too small depth for desired array" for a 0-d value, and — for a value of
/// rank two or more — either "object too deep for desired array" when the value was already an
/// ndarray, or the longer "setting an array element with a sequence..." message when it was not
/// (a list of lists, for example). Both phrasings were confirmed black-box against the
/// reference interpreter. `sort::bincount`'s `weights=` needs exactly this validation, so it is
/// exposed from here rather than duplicated.
pub(in crate::python) fn one_dimensional(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    dtype: DType,
) -> PyResult<Array> {
    let is_array = runtime.native_kind(&value)? == Some(PyNativeKind::Array);
    let array = convert::as_array(runtime, value)?;
    match array.ndim() {
        0 => {
            return Err(PyError::value_error(
                "object of too small depth for desired array",
            ))
        }
        1 => {}
        _ if is_array => return Err(PyError::value_error("object too deep for desired array")),
        _ => {
            return Err(PyError::value_error(
                "setting an array element with a sequence. The requested array would exceed \
                 the maximum number of dimension of 1.",
            ))
        }
    }
    convert::cast_array(runtime, &array, dtype, false)
}

/// `xp`/`fp` as one-dimensional, equal-length, nonempty arrays. `xp` is cast to `float64`;
/// `fp`'s dtype is left to the caller, which knows whether it wants a real or complex working
/// precision.
fn validate_samples(
    runtime: &mut dyn PyRuntime,
    xp_value: PyValue,
    fp_value: PyValue,
) -> PyResult<(Array, Array)> {
    let fp = convert::as_array(runtime, fp_value)?;
    let xp = one_dimensional(runtime, xp_value, DType::FLOAT64)?;
    if xp.size() == 0 {
        return Err(PyError::value_error("array of sample points is empty"));
    }
    if xp.size() != fp.size() {
        return Err(PyError::value_error(
            "fp and xp are not of the same length.",
        ));
    }
    Ok((xp, fp))
}

/// A real or complex scalar argument (`left`/`right`), accepting a bare Python number or a
/// NumPy scalar of either kind.
fn complex_arg(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<(f64, f64)> {
    if let Some((_, number)) = super::scalar::unbox_number(runtime, value) {
        return Ok(number.as_complex());
    }
    if let Some(NumberRef::Complex(real, imag)) = runtime.number(value) {
        return Ok((real, imag));
    }
    Ok((args::float_arg(runtime, value)?, 0.0))
}

/// The scalar or array `result` boxed the way NumPy returns it: a NumPy scalar for a 0-d
/// result, the array itself otherwise.
fn finish_array(runtime: &mut dyn PyRuntime, result: Array) -> PyResult {
    if result.ndim() == 0 {
        convert::element_to_scalar(runtime, &result, result.view.offset)
    } else {
        Ok(result.value())
    }
}

/// `_compiled_interp(x, xp, fp, left, right)`: `numpy.interp` for real `fp`. Always computes
/// and returns `float64`, matching NumPy (confirmed black-box: a `float32` `x` or an integer
/// `fp` still interpolates to `float64`).
fn interp_real(
    runtime: &mut dyn PyRuntime,
    x_value: PyValue,
    xp_value: PyValue,
    fp_value: PyValue,
    left: Option<PyValue>,
    right: Option<PyValue>,
) -> PyResult {
    let (xp, fp) = validate_samples(runtime, xp_value, fp_value)?;
    let fp64 = convert::cast_array(runtime, &fp, DType::FLOAT64, false)?;
    let xp_values = array::read_elements::<f64>(runtime, &xp)?;
    let fp_values = array::read_elements::<f64>(runtime, &fp64)?;
    let left = left
        .map(|value| args::float_arg(runtime, &value))
        .transpose()?;
    let right = right
        .map(|value| args::float_arg(runtime, &value))
        .transpose()?;

    let x = convert::as_array(runtime, x_value)?;
    let x64 = convert::cast_array(runtime, &x, DType::FLOAT64, false)?;
    let x_values = array::read_elements::<f64>(runtime, &x64)?;

    charge_interp(runtime, x_values.len(), xp_values.len())?;
    array::reserve_elements(runtime, DType::FLOAT64, x_values.len())?;

    let last = xp_values.len() - 1;
    let output = x_values
        .iter()
        .map(|&value| match locate(&xp_values, value) {
            Position::Nan => f64::NAN,
            Position::Below => left.unwrap_or(fp_values[0]),
            Position::Above => right.unwrap_or(fp_values[last]),
            Position::Exact(index) => fp_values[index],
            Position::Interval(index) => {
                let (x0, x1) = (xp_values[index], xp_values[index + 1]);
                let (y0, y1) = (fp_values[index], fp_values[index + 1]);
                interpolate(value, x0, x1, y0, y1, (y1 - y0) / (x1 - x0))
            }
        })
        .collect::<Vec<_>>();

    let result =
        array::array_from_elements(runtime, DType::FLOAT64, x64.shape().to_vec(), &output)?;
    finish_array(runtime, result)
}

/// `_compiled_interp_complex(x, xp, fp, left, right)`: `numpy.interp` for complex `fp`. The
/// real and imaginary parts interpolate independently at the same `x` position; always computes
/// and returns `complex128`.
fn interp_complex(
    runtime: &mut dyn PyRuntime,
    x_value: PyValue,
    xp_value: PyValue,
    fp_value: PyValue,
    left: Option<PyValue>,
    right: Option<PyValue>,
) -> PyResult {
    let (xp, fp) = validate_samples(runtime, xp_value, fp_value)?;
    let fp128 = convert::cast_array(runtime, &fp, DType::COMPLEX128, false)?;
    let xp_values = array::read_elements::<f64>(runtime, &xp)?;
    let fp_values = array::read_elements::<C128>(runtime, &fp128)?
        .into_iter()
        .map(|value| (value.re, value.im))
        .collect::<Vec<_>>();
    let left = left.map(|value| complex_arg(runtime, &value)).transpose()?;
    let right = right
        .map(|value| complex_arg(runtime, &value))
        .transpose()?;

    let x = convert::as_array(runtime, x_value)?;
    let x64 = convert::cast_array(runtime, &x, DType::FLOAT64, false)?;
    let x_values = array::read_elements::<f64>(runtime, &x64)?;

    charge_interp(runtime, x_values.len(), xp_values.len())?;
    array::reserve_elements(runtime, DType::COMPLEX128, x_values.len())?;

    let last = xp_values.len() - 1;
    let output = x_values
        .iter()
        .map(|&value| {
            let (re, im) = match locate(&xp_values, value) {
                Position::Nan => (f64::NAN, f64::NAN),
                Position::Below => left.unwrap_or(fp_values[0]),
                Position::Above => right.unwrap_or(fp_values[last]),
                Position::Exact(index) => fp_values[index],
                Position::Interval(index) => {
                    let (x0, x1) = (xp_values[index], xp_values[index + 1]);
                    let ((re0, im0), (re1, im1)) = (fp_values[index], fp_values[index + 1]);
                    let inverse = 1.0 / (x1 - x0);
                    (
                        interpolate(value, x0, x1, re0, re1, (re1 - re0) * inverse),
                        interpolate(value, x0, x1, im0, im1, (im1 - im0) * inverse),
                    )
                }
            };
            C128 { re, im }
        })
        .collect::<Vec<_>>();

    let result =
        array::array_from_elements(runtime, DType::COMPLEX128, x64.shape().to_vec(), &output)?;
    finish_array(runtime, result)
}

fn module_compiled_interp(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("_compiled_interp", &["x", "xp", "fp", "left", "right"], 5);
    let bound = SIGNATURE.bind(&args)?;
    interp_real(
        runtime,
        bound.required("x"),
        bound.required("xp"),
        bound.required("fp"),
        bound.value("left"),
        bound.value("right"),
    )
}

fn module_compiled_interp_complex(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "_compiled_interp_complex",
        &["x", "xp", "fp", "left", "right"],
        5,
    );
    let bound = SIGNATURE.bind(&args)?;
    interp_complex(
        runtime,
        bound.required("x"),
        bound.required("xp"),
        bound.required("fp"),
        bound.value("left"),
        bound.value("right"),
    )
}

// ---------------------------------------------------------------------------------------------
// correlate / convolve
// ---------------------------------------------------------------------------------------------

/// `_correlate(a, v, mode, conjugate)`, backing both `numpy.correlate` (`conjugate=True`) and
/// `numpy.convolve` (which reverses `v` in Python first and calls this with `conjugate=False`).
///
/// `full[i] = sum_k a[k] * op(v[k - i + len(v) - 1])` over `k` where both indices are in range,
/// `op` the identity or conjugation; this is the standard sliding-window cross-correlation,
/// verified element-by-element against the reference interpreter's `np.correlate`. `'same'` and
/// `'valid'` are prefixes of that `'full'` result: `'same'` keeps `max(len(a), len(v))` elements
/// starting at `(len(v) - 1) / 2`, and `'valid'` keeps `|len(a) - len(v)| + 1` starting at
/// `min(len(a), len(v)) - 1` — both offsets confirmed black-box, including the asymmetric
/// `'same'` case where `v` is longer than `a`.
fn correlate_kernel(
    runtime: &mut dyn PyRuntime,
    a: &Array,
    v: &Array,
    mode: &str,
    conjugate: bool,
) -> PyResult<Array> {
    let (m, n) = (a.size(), v.size());
    if m == 0 {
        return Err(PyError::value_error("first array argument cannot be empty"));
    }
    if n == 0 {
        return Err(PyError::value_error(
            "second array argument cannot be empty",
        ));
    }
    let full_len = m + n - 1;
    let (start, length) = match mode {
        "full" => (0usize, full_len),
        "same" => ((n - 1) / 2, m.max(n)),
        "valid" => (m.min(n) - 1, m.max(n) - m.min(n) + 1),
        other => {
            return Err(PyError::value_error(format!(
                "mode must be one of 'valid', 'same', or 'full' (got '{other}')"
            )))
        }
    };
    // Every output element sums at most `n` products, so the total work is bounded by `m * n`
    // regardless of mode; charge that before running the loops below.
    runtime.charge_cpu((m as u64).saturating_mul(n as u64) + 1)?;

    let dtype = dtype::promote(a.dtype, v.dtype)?;
    let result = if dtype.category() == Category::Complex {
        let a128 = convert::cast_array(runtime, a, DType::COMPLEX128, false)?;
        let v128 = convert::cast_array(runtime, v, DType::COMPLEX128, false)?;
        let a_values = array::read_elements::<C128>(runtime, &a128)?;
        let v_values = array::read_elements::<C128>(runtime, &v128)?;
        array::reserve_elements(runtime, DType::COMPLEX128, length)?;
        let mut output = Vec::with_capacity(length);
        for i in start..start + length {
            let lag = i as isize - (n as isize - 1);
            let mut sum = (0.0, 0.0);
            for (k, &v_val) in v_values.iter().enumerate().take(n) {
                let source = lag + k as isize;
                if source < 0 || source as usize >= m {
                    continue;
                }
                let a_val = a_values[source as usize];
                let v_val = if conjugate {
                    (v_val.re, -v_val.im)
                } else {
                    (v_val.re, v_val.im)
                };
                let product = ops::complex_multiply((a_val.re, a_val.im), v_val);
                sum.0 += product.0;
                sum.1 += product.1;
            }
            output.push(C128 {
                re: sum.0,
                im: sum.1,
            });
        }
        array::array_from_elements(runtime, DType::COMPLEX128, vec![length], &output)?
    } else {
        let a64 = convert::cast_array(runtime, a, DType::FLOAT64, false)?;
        let v64 = convert::cast_array(runtime, v, DType::FLOAT64, false)?;
        let a_values = array::read_elements::<f64>(runtime, &a64)?;
        let v_values = array::read_elements::<f64>(runtime, &v64)?;
        array::reserve_elements(runtime, DType::FLOAT64, length)?;
        let mut output = Vec::with_capacity(length);
        for i in start..start + length {
            let lag = i as isize - (n as isize - 1);
            let mut sum = 0.0;
            for (k, &v_val) in v_values.iter().enumerate().take(n) {
                let source = lag + k as isize;
                if source < 0 || source as usize >= m {
                    continue;
                }
                sum += a_values[source as usize] * v_val;
            }
            output.push(sum);
        }
        array::array_from_elements(runtime, DType::FLOAT64, vec![length], &output)?
    };
    convert::cast_array(runtime, &result, dtype, false)
}

fn module_correlate(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("_correlate", &["a", "v", "mode", "conjugate"], 4);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let v = convert::as_array(runtime, bound.required("v"))?;
    let mode = runtime
        .string_value(&bound.required("mode"))?
        .unwrap_or_default();
    let conjugate = runtime.truth(&bound.required("conjugate"))?;
    Ok(correlate_kernel(runtime, &a, &v, &mode, conjugate)?.value())
}

// ---------------------------------------------------------------------------------------------
// round / around
// ---------------------------------------------------------------------------------------------

/// `10 ** decimals`, the factor [`round_scaled`] scales by before rounding and divides by
/// after.
fn decimal_scale(decimals: i64) -> f64 {
    10f64.powi(decimals.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
}

/// Round-half-to-even at `10 ** -decimals` resolution's inverse (`scale = 10 ** decimals`).
fn round_half_even(value: f64, scale: f64) -> f64 {
    (value * scale).round_ties_even() / scale
}

/// [`round_half_even`] lifted to a dtype-independent [`Number`]: both parts of a complex value
/// round independently, everything else rounds as a real number (through `as_f64`, which is
/// exact for every supported integer width and for `f32`/`f16`).
fn round_scaled(value: Number, scale: f64) -> Number {
    match value {
        Number::Complex(real, imag) => {
            Number::Complex(round_half_even(real, scale), round_half_even(imag, scale))
        }
        other => Number::Float(round_half_even(other.as_f64(), scale)),
    }
}

/// `round(scalar, decimals)`: NumPy scalars' `__round__` (via `scalar::method_round`), rounding
/// half-to-even at `decimals` places in the same [`Number`] representation callers already
/// hold. `dtype` is not needed for the arithmetic — `number`'s `f64` already holds the
/// receiver's exact value, widening a float16/32 losslessly — but is accepted because the call
/// site has already used it to reject a complex receiver (`__round__` is undefined there)
/// before reaching this function.
pub(in crate::python) fn round_number(dtype: DType, number: Number, decimals: i64) -> Number {
    let _ = dtype;
    round_scaled(number, decimal_scale(decimals))
}

/// `np.round`/`np.around`(a, decimals=0) and `a.round(decimals=0)`: round to `decimals` places
/// with round-half-to-even, preserving `a`'s dtype. Reads and writes one generic [`Number`] per
/// element (via [`element::read_number`]/[`element::write_number`]) rather than dispatching on
/// dtype, since rounding needs no dtype-specific arithmetic — every numeric kind's `Number` cast
/// back through `write_number` the same "unsafe" cast `astype` uses.
fn round_array(runtime: &mut dyn PyRuntime, array: &Array, decimals: i64) -> PyResult<Array> {
    let scale = decimal_scale(decimals);
    let count = array.size();
    array::reserve_elements(runtime, array.dtype, count)?;
    runtime.charge_cpu(count as u64 + 1)?;
    let kind = array.dtype.kind();
    let itemsize = array.itemsize();
    let mut buffer = array::buffer_with_capacity(array.dtype, count);
    runtime.read_arrays(&[array.handle], &mut |arrays| {
        let PyArrayData::Bytes(bytes) = arrays[0].data else {
            return Err(PyError::unsupported(
                "round on object arrays is not supported",
            ));
        };
        let PyArrayBuffer::Bytes(output) = &mut buffer else {
            unreachable!("numeric dtype has byte storage")
        };
        let mut slot = [0u8; 16];
        for offset in array.offsets() {
            let value = element::read_number(kind, &bytes[offset..offset + itemsize]);
            let rounded = round_scaled(value, scale);
            element::write_number(kind, rounded, &mut slot[..itemsize]);
            output.extend_from_slice(&slot[..itemsize]);
        }
        Ok(())
    })?;
    array::new_array(runtime, buffer, array.dtype, array.shape().to_vec())
}

/// Store `result` into `out=` if given, otherwise box it the way NumPy returns a result.
fn finish_with_out(runtime: &mut dyn PyRuntime, result: Array, out: Option<PyValue>) -> PyResult {
    match out {
        Some(out) => {
            let out = Array::from_value(runtime, out)?;
            array::assign(runtime, &out, &result)?;
            Ok(out.value())
        }
        None => finish_array(runtime, result),
    }
}

fn module_round(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("round", &["a", "decimals", "out"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let decimals = args::optional_int(runtime, bound.get("decimals"))?.unwrap_or(0);
    let result = round_array(runtime, &array, decimals)?;
    finish_with_out(runtime, result, bound.value("out"))
}

fn method_round(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("round", &["decimals", "out"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    let decimals = args::optional_int(runtime, bound.get("decimals"))?.unwrap_or(0);
    let result = round_array(runtime, &array, decimals)?;
    finish_with_out(runtime, result, bound.value("out"))
}

// ---------------------------------------------------------------------------------------------
// clip
// ---------------------------------------------------------------------------------------------

/// `np.clip(a, a_min, a_max)` and `a.clip(min, max)`: `minimum(maximum(a, a_min), a_max)`,
/// skipping either ufunc call when its bound is `None`. Reusing `maximum`/`minimum` (rather
/// than a bespoke comparison loop) gets their broadcasting, NEP 50 promotion, and `NaN`
/// propagation for free, and matches NumPy's own definition of `clip`.
///
/// The array method's parameters are named `min`/`max` (not `a_min`/`a_max`), matching real
/// NumPy's `ndarray.clip` signature: the frozen `numpy._numeric.clip` wrapper resolves the
/// module function's legacy `a_min`/`a_max` positionals and its newer keyword-only `min=`/`max=`
/// aliases, then always calls `a.clip(min=..., max=...)` on the array.
fn clip_array(
    runtime: &mut dyn PyRuntime,
    a: PyValue,
    a_min: Option<PyValue>,
    a_max: Option<PyValue>,
    out: Option<Array>,
) -> PyResult {
    match (a_min, a_max) {
        (None, None) => {
            let array = convert::as_array(runtime, a)?;
            let copy = array::copy_array(runtime, &array)?;
            finish_with_out(runtime, copy, out.map(|out| out.value()))
        }
        (Some(min), None) => ufunc::apply(
            runtime,
            ufunc::named("maximum"),
            &[a, min],
            &ufunc::Options {
                out,
                ..ufunc::Options::default()
            },
        ),
        (None, Some(max)) => ufunc::apply(
            runtime,
            ufunc::named("minimum"),
            &[a, max],
            &ufunc::Options {
                out,
                ..ufunc::Options::default()
            },
        ),
        (Some(min), Some(max)) => {
            let lower = ufunc::apply(
                runtime,
                ufunc::named("maximum"),
                &[a, min],
                &ufunc::Options::default(),
            )?;
            ufunc::apply(
                runtime,
                ufunc::named("minimum"),
                &[lower, max],
                &ufunc::Options {
                    out,
                    ..ufunc::Options::default()
                },
            )
        }
    }
}

fn out_argument(runtime: &mut dyn PyRuntime, value: Option<PyValue>) -> PyResult<Option<Array>> {
    value
        .map(|value| Array::from_value(runtime, value))
        .transpose()
}

fn module_clip(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("clip", &["a", "a_min", "a_max", "out"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let out = out_argument(runtime, bound.value("out"))?;
    clip_array(
        runtime,
        bound.required("a"),
        bound.value("a_min"),
        bound.value("a_max"),
        out,
    )
}

fn method_clip(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("clip", &["min", "max", "out"], 0);
    let bound = SIGNATURE.bind(&args)?;
    let out = out_argument(runtime, bound.value("out"))?;
    clip_array(
        runtime,
        receiver,
        bound.value("min"),
        bound.value("max"),
        out,
    )
}

// ---------------------------------------------------------------------------------------------
// module registration
// ---------------------------------------------------------------------------------------------

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

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_math",
    functions: &[
        function("_compiled_interp", module_compiled_interp),
        function("_compiled_interp_complex", module_compiled_interp_complex),
        function("_correlate", module_correlate),
        function("divmod", ufunc::call_divmod),
        function("round", module_round),
        function("around", module_round),
        function("clip", module_clip),
    ],
    values: &[],
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

/// Methods this module installs on `numpy.ndarray`.
pub(in crate::python) static ARRAY_METHODS: NativeTypeDef = NativeTypeDef {
    name: "numpy.ndarray",
    methods: &[method("round", method_round), method("clip", method_clip)],
    getters: &[],
};
