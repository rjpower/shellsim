//! `numpy.fft`: discrete Fourier transforms and their sample-frequency helpers.
//!
//! [`kernel`] holds the O(n log n) numeric core (radix-2 Cooley–Tukey plus Bluestein's
//! algorithm); this module resolves NumPy's argument conventions (`n`/`s`, `axis`/`axes`,
//! `norm`, `out`) down to calls into that core, and picks each result's dtype the way NumPy's
//! ufunc dispatch does: `float16`/`float32`/`complex64` inputs keep single precision, every
//! other numeric input promotes to double. [`multi`] builds the multi-axis transforms
//! (`fftn`, `rfftn`, ...) out of the one-axis functions defined here, and [`helpers`] holds
//! `fftfreq`, `rfftfreq`, `fftshift`, and `ifftshift`.
//!
//! Every transform reads its input generically through [`element::read_number`], works in
//! `f64` regardless of the array's storage precision, and rounds down to `f32` only when
//! writing a `complex64`/`float32` result — see `kernel`'s module doc for why that comfortably
//! beats the ~1e-15*log2(n) relative accuracy target even for single-precision results. CPU is
//! charged per one-dimensional transform (`kernel::transform_cost`), before that transform
//! runs, and memory is reserved for each buffer before it is filled, so a large batch of
//! transforms is metered before any of them run.

mod helpers;
mod kernel;
mod multi;

use super::super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyArrayData, PyError, PyResult, PyRuntime, PyValue,
};
use super::args::{self, Signature};
use super::array::{self, Array};
use super::convert;
use super::dtype::{self, DType, Kind};
use super::element::{self, C128, C64};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_fft",
    functions: FUNCTIONS,
    values: &[],
};

const fn function(
    name: &'static str,
    call: fn(&mut dyn PyRuntime, CallArgs) -> PyResult,
) -> FunctionDef {
    FunctionDef {
        module: "numpy.fft",
        name,
        call,
    }
}

static FUNCTIONS: &[FunctionDef] = &[
    function("fft", fft),
    function("ifft", ifft),
    function("rfft", rfft),
    function("irfft", irfft),
    function("hfft", hfft),
    function("ihfft", ihfft),
    function("fft2", multi::fft2),
    function("ifft2", multi::ifft2),
    function("fftn", multi::fftn),
    function("ifftn", multi::ifftn),
    function("rfft2", multi::rfft2),
    function("irfft2", multi::irfft2),
    function("rfftn", multi::rfftn),
    function("irfftn", multi::irfftn),
    function("fftfreq", helpers::fftfreq),
    function("rfftfreq", helpers::rfftfreq),
    function("fftshift", helpers::fftshift),
    function("ifftshift", helpers::ifftshift),
];

/// Which direction of the forward/backward pair `norm=` scales, and by how much.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Norm {
    Backward,
    Ortho,
    Forward,
}

impl Norm {
    fn parse(runtime: &mut dyn PyRuntime, value: Option<PyValue>) -> PyResult<Self> {
        let Some(value) = value.filter(|value| !value.is_none()) else {
            return Ok(Self::Backward);
        };
        let text = runtime
            .string_value(&value)?
            .ok_or_else(|| PyError::type_error("norm must be a str"))?;
        match text.as_str() {
            "backward" => Ok(Self::Backward),
            "ortho" => Ok(Self::Ortho),
            "forward" => Ok(Self::Forward),
            other => Err(PyError::value_error(format!(
                "Invalid norm value {other}; should be \"backward\", \"ortho\" or \"forward\"."
            ))),
        }
    }

    /// The multiplicative factor applied once, after the unnormalized transform.
    fn scale(self, n: usize, inverse: bool) -> f64 {
        match self {
            Self::Backward => {
                if inverse {
                    1.0 / n as f64
                } else {
                    1.0
                }
            }
            Self::Forward => {
                if inverse {
                    1.0
                } else {
                    1.0 / n as f64
                }
            }
            Self::Ortho => 1.0 / (n as f64).sqrt(),
        }
    }
}

/// `axis=` resolved the way `numpy.fft` resolves it: by indexing `a.shape` directly, so both a
/// too-large and a too-negative axis give Python's plain tuple-indexing `IndexError`, not
/// NumPy's usual `AxisError`.
fn normalize_raw_axis(axis: i64, ndim: usize) -> PyResult<usize> {
    let normalized = if axis < 0 { axis + ndim as i64 } else { axis };
    if normalized < 0 || normalized >= ndim as i64 {
        return Err(PyError::exception("IndexError", "tuple index out of range"));
    }
    Ok(normalized as usize)
}

fn resolve_axis(
    runtime: &mut dyn PyRuntime,
    value: Option<PyValue>,
    ndim: usize,
) -> PyResult<usize> {
    let axis = match value.filter(|value| !value.is_none()) {
        Some(value) => args::index_int(runtime, &value)?,
        None => -1,
    };
    normalize_raw_axis(axis, ndim)
}

/// `n=` resolved against `default` (the caller's un-overridden transform length), with NumPy's
/// error for a non-positive result either way.
fn resolve_n(runtime: &mut dyn PyRuntime, value: Option<PyValue>, default: i64) -> PyResult<usize> {
    let n = match args::optional_int(runtime, value)? {
        Some(n) => n,
        None => default,
    };
    if n <= 0 {
        return Err(PyError::value_error(format!(
            "Invalid number of FFT data points ({n}) specified."
        )));
    }
    Ok(n as usize)
}

fn out_array(runtime: &mut dyn PyRuntime, value: Option<PyValue>) -> PyResult<Option<Array>> {
    value
        .filter(|value| !value.is_none())
        .map(|value| {
            Array::from_value(runtime, value)
                .map_err(|_| PyError::type_error("output must be an array"))
        })
        .transpose()
}

fn unsupported_input_type(name: &str) -> PyError {
    PyError::type_error(format!(
        "ufunc '{name}' not supported for the input types, and the inputs could not be safely \
         coerced to any supported types according to the casting rule ''safe''"
    ))
}

fn reject_non_numeric(array: &Array, name: &str) -> PyResult<()> {
    if array.dtype.is_numeric() {
        Ok(())
    } else {
        Err(unsupported_input_type(name))
    }
}

/// `rfft`/`ihfft` require real input; pocketfft's real transform is one of two named loops
/// depending on whether the transform length is even or odd, and that name leaks into this
/// error (and into a mismatched `out=` dtype error) exactly as it does in NumPy.
fn real_ufunc_name(n: usize) -> &'static str {
    if n.is_multiple_of(2) {
        "rfft_n_even"
    } else {
        "rfft_n_odd"
    }
}

fn reject_complex(array: &Array, name: &str) -> PyResult<()> {
    match array.dtype.kind() {
        Kind::Complex64 | Kind::Complex128 => Err(unsupported_input_type(name)),
        _ => reject_non_numeric(array, name),
    }
}

/// Read every element of `array` (any numeric dtype) as a full-precision complex number, in C
/// order.
fn read_as_complex(runtime: &mut dyn PyRuntime, array: &Array) -> PyResult<Vec<C128>> {
    let count = array.size();
    runtime.reserve_memory(count.saturating_mul(16))?;
    runtime.charge_cpu(count as u64 + 1)?;
    let kind = array.dtype.kind();
    let mut output = Vec::with_capacity(count);
    runtime.read_arrays(&[array.handle], &mut |arrays| {
        let PyArrayData::Bytes(bytes) = arrays[0].data else {
            return Err(PyError::runtime_error("fft input has non-numeric storage"));
        };
        output.extend(array.offsets().map(|offset| {
            let (re, im) = element::read_number(kind, &bytes[offset..]).as_complex();
            C128 { re, im }
        }));
        Ok(())
    })?;
    Ok(output)
}

/// `values` (shape `shape`) cropped or zero-padded along `axis` to length `n`.
fn resize_axis(
    runtime: &mut dyn PyRuntime,
    values: &[C128],
    shape: &[usize],
    axis: usize,
    n: usize,
) -> PyResult<(Vec<C128>, Vec<usize>)> {
    let mut new_shape = shape.to_vec();
    let old_n = new_shape[axis];
    new_shape[axis] = n;
    let total = array::element_count(&new_shape)?;
    runtime.reserve_memory(total.saturating_mul(16))?;
    runtime.charge_cpu(total as u64 + 1)?;
    let step = shape[axis + 1..].iter().product::<usize>().max(1);
    let outer = shape[..axis].iter().product::<usize>().max(1);
    let copy_n = old_n.min(n);
    let mut output = vec![C128::default(); total];
    for o in 0..outer {
        let src_base = o * old_n * step;
        let dst_base = o * n * step;
        for inner in 0..step {
            for k in 0..copy_n {
                output[dst_base + k * step + inner] = values[src_base + k * step + inner];
            }
        }
    }
    Ok((output, new_shape))
}

/// `values` (shape `shape`, `m = shape[axis]` one-sided bins) extended along `axis` to the
/// length-`n` Hermitian-symmetric sequence `irfft`/`hfft` transform: the given bins keep their
/// place and `full[n-k] = conj(values[k])` mirrors the rest, leaving any bins that formula does
/// not reach (when `n` is larger than `2*(m-1)+1`) at zero.
fn hermitian_extend(
    runtime: &mut dyn PyRuntime,
    values: &[C128],
    shape: &[usize],
    axis: usize,
    n: usize,
) -> PyResult<(Vec<C128>, Vec<usize>)> {
    let m = shape[axis];
    let mut new_shape = shape.to_vec();
    new_shape[axis] = n;
    let total = array::element_count(&new_shape)?;
    runtime.reserve_memory(total.saturating_mul(16))?;
    runtime.charge_cpu(total as u64 + 1)?;
    let step = shape[axis + 1..].iter().product::<usize>().max(1);
    let outer = shape[..axis].iter().product::<usize>().max(1);
    let mirror = n.saturating_sub(m);
    let mut output = vec![C128::default(); total];
    for o in 0..outer {
        let src_base = o * m * step;
        let dst_base = o * n * step;
        for inner in 0..step {
            for k in 0..m {
                output[dst_base + k * step + inner] = values[src_base + k * step + inner];
            }
            for k in 1..=mirror {
                let value = values[src_base + k * step + inner];
                output[dst_base + (n - k) * step + inner] = C128 {
                    re: value.re,
                    im: -value.im,
                };
            }
        }
    }
    Ok((output, new_shape))
}

/// Run the unnormalized transform lane by lane along `axis`, then apply `norm`'s scale. Memory
/// for the working set is reserved once, up front (peak usage is one lane at a time, not the
/// sum across lanes); CPU is charged per lane, immediately before that lane runs, so a batch
/// that runs out of budget partway is caught before finishing the rest.
fn run_lanes(
    runtime: &mut dyn PyRuntime,
    values: &mut [C128],
    shape: &[usize],
    axis: usize,
    inverse: bool,
    norm: Norm,
) -> PyResult<()> {
    let n = shape[axis];
    let working = kernel::working_size(n);
    runtime.reserve_memory(working.saturating_mul(48))?;
    let step = shape[axis + 1..].iter().product::<usize>().max(1);
    let outer = shape[..axis].iter().product::<usize>().max(1);
    let scale = norm.scale(n, inverse);
    let mut lane = vec![C128::default(); n];
    for o in 0..outer {
        let base = o * n * step;
        for inner in 0..step {
            runtime.charge_cpu(kernel::transform_cost(n))?;
            for k in 0..n {
                lane[k] = values[base + k * step + inner];
            }
            kernel::dft(&mut lane, inverse);
            for (k, value) in lane.iter().enumerate() {
                values[base + k * step + inner] = C128 {
                    re: value.re * scale,
                    im: value.im * scale,
                };
            }
        }
    }
    Ok(())
}

fn complex_array(
    runtime: &mut dyn PyRuntime,
    precision: DType,
    shape: Vec<usize>,
    values: &[C128],
) -> PyResult<Array> {
    if precision.kind() == Kind::Complex64 {
        let narrow: Vec<C64> = values
            .iter()
            .map(|value| C64 {
                re: value.re as f32,
                im: value.im as f32,
            })
            .collect();
        array::array_from_elements(runtime, precision, shape, &narrow)
    } else {
        array::array_from_elements(runtime, precision, shape, values)
    }
}

fn real_array(
    runtime: &mut dyn PyRuntime,
    precision: DType,
    shape: Vec<usize>,
    values: &[C128],
) -> PyResult<Array> {
    if precision.kind() == Kind::Complex64 {
        let narrow: Vec<f32> = values.iter().map(|value| value.re as f32).collect();
        array::array_from_elements(runtime, DType::FLOAT32, shape, &narrow)
    } else {
        let wide: Vec<f64> = values.iter().map(|value| value.re).collect();
        array::array_from_elements(runtime, DType::FLOAT64, shape, &wide)
    }
}

/// The shared core of `fft`, `ifft`, `rfft`, and `ihfft`: promote `array` to the matching
/// complex precision, crop or zero-pad it to length `n` along `axis`, run the transform, and
/// (when `output_len < n`, for the one-sided real transforms) crop the result to `output_len`.
fn transform(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    n: usize,
    output_len: usize,
    axis: usize,
    inverse: bool,
    norm: Norm,
) -> PyResult<Array> {
    let precision = array.dtype.complex_for();
    let values = read_as_complex(runtime, array)?;
    let (mut values, mut shape) = resize_axis(runtime, &values, array.shape(), axis, n)?;
    run_lanes(runtime, &mut values, &shape, axis, inverse, norm)?;
    if output_len != n {
        let (trimmed, trimmed_shape) = resize_axis(runtime, &values, &shape, axis, output_len)?;
        values = trimmed;
        shape = trimmed_shape;
    }
    complex_array(runtime, precision, shape, &values)
}

/// The shared core of `irfft` and `hfft`: crop or zero-pad `array` to `n/2+1` bins along `axis`,
/// build the length-`n` Hermitian-symmetric extension, run the transform, and keep only the
/// (guaranteed, up to rounding) real part.
fn real_inverse_transform(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    n: usize,
    axis: usize,
    inverse: bool,
    norm: Norm,
) -> PyResult<Array> {
    let precision = array.dtype.complex_for();
    let values = read_as_complex(runtime, array)?;
    let m = n / 2 + 1;
    let (cropped, cropped_shape) = resize_axis(runtime, &values, array.shape(), axis, m)?;
    let (mut extended, shape) = hermitian_extend(runtime, &cropped, &cropped_shape, axis, n)?;
    run_lanes(runtime, &mut extended, &shape, axis, inverse, norm)?;
    real_array(runtime, precision, shape, &extended)
}

/// Return `result`, or write it into `out` (after checking NumPy's ufunc `out=` rules: an exact
/// shape match and a `same_kind`-castable dtype) and return `out`.
fn finish(
    runtime: &mut dyn PyRuntime,
    name: &str,
    result: Array,
    out: Option<Array>,
) -> PyResult<PyValue> {
    let Some(out) = out else {
        return Ok(result.value());
    };
    if out.shape() != result.shape() {
        return Err(PyError::value_error("output array has wrong shape."));
    }
    if !dtype::can_cast(result.dtype, out.dtype, dtype::Casting::SameKind) {
        return Err(PyError::type_error(format!(
            "Cannot cast ufunc '{name}' output from dtype('{}') to dtype('{}') with casting rule 'same_kind'",
            result.dtype.kind().name(),
            out.dtype.kind().name(),
        )));
    }
    array::assign(runtime, &out, &result)?;
    Ok(out.value())
}

static ONE_AXIS: Signature = Signature::new("fft", &["a", "n", "axis", "norm", "out"], 1);

fn fft(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ONE_AXIS.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    reject_non_numeric(&array, "fft")?;
    let axis = resolve_axis(runtime, bound.get("axis"), array.ndim())?;
    let n = resolve_n(runtime, bound.get("n"), array.shape()[axis] as i64)?;
    let norm = Norm::parse(runtime, bound.get("norm"))?;
    let out = out_array(runtime, bound.get("out"))?;
    let result = transform(runtime, &array, n, n, axis, false, norm)?;
    finish(runtime, "fft", result, out)
}

fn ifft(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ONE_AXIS.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    reject_non_numeric(&array, "ifft")?;
    let axis = resolve_axis(runtime, bound.get("axis"), array.ndim())?;
    let n = resolve_n(runtime, bound.get("n"), array.shape()[axis] as i64)?;
    let norm = Norm::parse(runtime, bound.get("norm"))?;
    let out = out_array(runtime, bound.get("out"))?;
    let result = transform(runtime, &array, n, n, axis, true, norm)?;
    finish(runtime, "ifft", result, out)
}

fn rfft(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ONE_AXIS.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let axis = resolve_axis(runtime, bound.get("axis"), array.ndim())?;
    let n = resolve_n(runtime, bound.get("n"), array.shape()[axis] as i64)?;
    let name = real_ufunc_name(n);
    reject_complex(&array, name)?;
    let norm = Norm::parse(runtime, bound.get("norm"))?;
    let out = out_array(runtime, bound.get("out"))?;
    let result = transform(runtime, &array, n, n / 2 + 1, axis, false, norm)?;
    finish(runtime, name, result, out)
}

fn ihfft(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ONE_AXIS.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let axis = resolve_axis(runtime, bound.get("axis"), array.ndim())?;
    let n = resolve_n(runtime, bound.get("n"), array.shape()[axis] as i64)?;
    let name = real_ufunc_name(n);
    reject_complex(&array, name)?;
    let norm = Norm::parse(runtime, bound.get("norm"))?;
    let out = out_array(runtime, bound.get("out"))?;
    let result = transform(runtime, &array, n, n / 2 + 1, axis, true, norm)?;
    finish(runtime, name, result, out)
}

fn irfft(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ONE_AXIS.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    reject_non_numeric(&array, "irfft")?;
    let axis = resolve_axis(runtime, bound.get("axis"), array.ndim())?;
    let default_n = 2 * (array.shape()[axis] as i64 - 1);
    let n = resolve_n(runtime, bound.get("n"), default_n)?;
    let norm = Norm::parse(runtime, bound.get("norm"))?;
    let out = out_array(runtime, bound.get("out"))?;
    let result = real_inverse_transform(runtime, &array, n, axis, true, norm)?;
    finish(runtime, "irfft", result, out)
}

fn hfft(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ONE_AXIS.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    reject_non_numeric(&array, "irfft")?;
    let axis = resolve_axis(runtime, bound.get("axis"), array.ndim())?;
    let default_n = 2 * (array.shape()[axis] as i64 - 1);
    let n = resolve_n(runtime, bound.get("n"), default_n)?;
    let norm = Norm::parse(runtime, bound.get("norm"))?;
    let out = out_array(runtime, bound.get("out"))?;
    let result = real_inverse_transform(runtime, &array, n, axis, false, norm)?;
    finish(runtime, "irfft", result, out)
}
