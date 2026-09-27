//! The native module `_numpy_fft`, which plays the part of NumPy's `_pocketfft_umath`: the
//! generalized ufuncs `fft`, `ifft`, `rfft_n_even`, `rfft_n_odd` and `irfft` that
//! `numpy/fft.py` calls as `ufunc(a, fct, axis, n, out)`.
//!
//! Each function transforms every line of `a` along `axis` with the [`pocketfft`] port, scales
//! it by `fct`, and stores the result in `out`, which the Python layer allocates with NumPy's
//! output dtype. Input longer than the transform is truncated and shorter input is padded with
//! zeros, as the gufunc loops do.
//!
//! The precision follows NumPy's dispatch, which runs the single-precision loop only when the
//! inputs match its dtypes exactly: complex64 input (float32 for `rfft`) with a float32 `fct`,
//! as `numpy.fft` passes whenever the normalization is not 1. Anything else, including float32
//! input to a complex transform or the Python int `fct` of an unnormalized one, goes through
//! legacy type resolution, which tries the double loop first; that loop accepts every input
//! that can be cast safely, and a complex64 `out` receives the rounded result.
//!
//! One plan serves every line. Plan memory and the output are reserved, and the plan's
//! set-up and each line's work are charged from pocketfft's cost model, before any of it runs.

mod pocketfft;

use super::super::super::native::{CallArgs, FunctionDef, ModuleDef, PyError, PyResult, PyRuntime};
use super::args::{float_arg, index_int, Signature};
use super::array::{self, Array};
use super::convert;
use super::dtype::{self, Casting, DType};
use super::element::C128;
use super::scalar::unbox_number;
use pocketfft::{Cmplx, ComplexPlan, Real, RealPlan};

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
        module: "numpy.fft._pocketfft_umath",
        name,
        call,
    }
}

static FUNCTIONS: &[FunctionDef] = &[
    function("fft", fft),
    function("ifft", ifft),
    function("rfft_n_even", rfft),
    function("rfft_n_odd", rfft),
    function("irfft", irfft),
];

/// Which transform a gufunc runs.
#[derive(Clone, Copy, PartialEq)]
enum Transform {
    Forward,
    Backward,
    RealForward,
    RealBackward,
}

impl Transform {
    fn name(self, n: usize) -> &'static str {
        match self {
            Self::Forward => "fft",
            Self::Backward => "ifft",
            Self::RealForward if n.is_multiple_of(2) => "rfft_n_even",
            Self::RealForward => "rfft_n_odd",
            Self::RealBackward => "irfft",
        }
    }

    fn is_real(self) -> bool {
        matches!(self, Self::RealForward | Self::RealBackward)
    }

    /// The input and output dtypes of the double or single loop.
    fn loop_dtypes(self, single: bool) -> (DType, DType) {
        let (complex, real) = if single {
            (DType::COMPLEX64, DType::FLOAT32)
        } else {
            (DType::COMPLEX128, DType::FLOAT64)
        };
        match self {
            Self::Forward | Self::Backward => (complex, complex),
            Self::RealForward => (real, complex),
            Self::RealBackward => (complex, real),
        }
    }

    /// The length of the output axis for a transform of `n` points.
    fn output_length(self, n: usize) -> usize {
        match self {
            Self::RealForward => n / 2 + 1,
            _ => n,
        }
    }
}

/// `fft(a, fct, axis, n, out)`: the forward complex transform.
fn fft(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    run(runtime, args, Transform::Forward)
}

/// `ifft(a, fct, axis, n, out)`: the backward complex transform, unnormalized unless `fct`
/// says otherwise.
fn ifft(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    run(runtime, args, Transform::Backward)
}

/// `rfft_n_even` and `rfft_n_odd`: the forward transform of real input, keeping the
/// `n // 2 + 1` non-negative frequencies.
fn rfft(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    run(runtime, args, Transform::RealForward)
}

/// `irfft(a, fct, axis, n, out)`: `n` real points from the non-negative frequencies of a
/// Hermitian spectrum.
fn irfft(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    run(runtime, args, Transform::RealBackward)
}

static SIGNATURE: Signature =
    Signature::new("_pocketfft_umath", &["a", "fct", "axis", "n", "out"], 5);

/// The lines of a C-order array along one axis: the product of the axes before it, its
/// length, and the product of the axes after it.
#[derive(Clone, Copy)]
struct Lines {
    outer: usize,
    length: usize,
    inner: usize,
}

impl Lines {
    fn count(&self) -> usize {
        self.outer * self.inner
    }

    /// The flat index of element `position` of line `line`.
    fn index(&self, line: usize, position: usize) -> usize {
        let (outer, inner) = (line / self.inner, line % self.inner);
        (outer * self.length + position) * self.inner + inner
    }
}

fn run(runtime: &mut dyn PyRuntime, args: CallArgs, transform: Transform) -> PyResult {
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let fct_value = bound.required("fct");
    let fct = float_arg(runtime, &fct_value)?;
    let axis = index_int(runtime, &bound.required("axis"))?;
    let axis = array::normalize_axis(axis, a.ndim())?;
    let n = index_int(runtime, &bound.required("n"))?;
    let n = usize::try_from(n).ok().filter(|n| *n >= 1).ok_or_else(|| {
        PyError::value_error(format!(
            "Invalid number of FFT data points ({n}) specified."
        ))
    })?;
    let out = Array::from_value(runtime, bound.required("out"))?;
    let name = transform.name(n);

    let float32_fct =
        matches!(unbox_number(runtime, &fct_value), Some((dtype, _)) if dtype == DType::FLOAT32);
    let single = float32_fct && a.dtype == transform.loop_dtypes(true).0;
    let (input_dtype, output_dtype) = transform.loop_dtypes(single);
    if !dtype::can_cast(a.dtype, input_dtype, Casting::Safe) {
        return Err(PyError::type_error(format!(
            "ufunc '{name}' not supported for the input types, and the inputs could not be \
             safely coerced to any supported types according to the casting rule ''safe''"
        )));
    }
    if !dtype::can_cast(output_dtype, out.dtype, Casting::SameKind) {
        return Err(PyError::exception(
            "UFuncTypeError",
            format!(
                "Cannot cast ufunc '{name}' output from {} to {} with casting rule 'same_kind'",
                output_dtype.repr(),
                out.dtype.repr()
            ),
        ));
    }
    let output_length = transform.output_length(n);
    let mut shape = a.shape().to_vec();
    shape[axis] = output_length;
    if out.shape() != shape.as_slice() {
        return Err(PyError::value_error("output array has wrong shape."));
    }

    let lines = Lines {
        outer: a.shape()[..axis].iter().product(),
        length: a.shape()[axis],
        inner: a.shape()[axis + 1..].iter().product(),
    };
    let output_lines = Lines {
        length: output_length,
        ..lines
    };
    let count = lines.count();
    array::reserve_elements(runtime, output_dtype, count.saturating_mul(output_length))?;
    if count > 0 {
        let padded = pocketfft::plan_length(n, transform.is_real());
        // Twiddles, scratch and Bluestein's buffers stay within a few complex values per
        // padded point.
        runtime.reserve_memory(padded.saturating_mul(6 * std::mem::size_of::<Cmplx<f64>>()))?;
        let per_line = pocketfft::transform_work(n, transform.is_real())
            .saturating_add((lines.length + output_length) as u64);
        runtime.charge_cpu(
            (4 * padded as u64).saturating_add((count as u64).saturating_mul(per_line)),
        )?;
    }
    // Values move in double precision either way; single-precision input and results convert
    // exactly.
    let (input_dtype, output_dtype) = transform.loop_dtypes(false);
    let a = if a.dtype == input_dtype {
        a
    } else {
        convert::cast_array(runtime, &a, input_dtype, false)?
    };
    let result = match transform {
        Transform::Forward | Transform::Backward => {
            let input = array::read_elements::<C128>(runtime, &a)?;
            let forward = transform == Transform::Forward;
            let values = if single {
                complex_lines::<f32>(&input, lines, output_lines, fct, forward)
            } else {
                complex_lines::<f64>(&input, lines, output_lines, fct, forward)
            };
            array::array_from_elements(runtime, output_dtype, shape, &values)?
        }
        Transform::RealForward => {
            let input = array::read_elements::<f64>(runtime, &a)?;
            let values = if single {
                real_forward_lines::<f32>(&input, lines, output_lines, n, fct)
            } else {
                real_forward_lines::<f64>(&input, lines, output_lines, n, fct)
            };
            array::array_from_elements(runtime, output_dtype, shape, &values)?
        }
        Transform::RealBackward => {
            let input = array::read_elements::<C128>(runtime, &a)?;
            let values = if single {
                real_backward_lines::<f32>(&input, lines, output_lines, fct)
            } else {
                real_backward_lines::<f64>(&input, lines, output_lines, fct)
            };
            array::array_from_elements(runtime, output_dtype, shape, &values)?
        }
    };
    array::assign(runtime, &out, &result)?;
    Ok(out.value())
}

fn to_cmplx<T: Real>(value: C128) -> Cmplx<T> {
    Cmplx {
        r: T::from_f64(value.re),
        i: T::from_f64(value.im),
    }
}

fn from_cmplx<T: Real>(value: Cmplx<T>) -> C128 {
    C128 {
        re: value.r.to_f64(),
        im: value.i.to_f64(),
    }
}

/// `fft_loop`: each line copied into an `n`-point buffer, transformed, and stored.
fn complex_lines<T: Real>(
    input: &[C128],
    lines: Lines,
    output_lines: Lines,
    fct: f64,
    forward: bool,
) -> Vec<C128> {
    let n = output_lines.length;
    let mut output = vec![C128::default(); output_lines.count() * n];
    if lines.count() == 0 {
        return output;
    }
    let plan = ComplexPlan::<T>::new(n);
    let fct = T::from_f64(fct);
    let mut buffer = vec![Cmplx::default(); n];
    let copied = lines.length.min(n);
    for line in 0..lines.count() {
        for (position, value) in buffer.iter_mut().enumerate() {
            *value = if position < copied {
                to_cmplx(input[lines.index(line, position)])
            } else {
                Cmplx::default()
            };
        }
        plan.exec(&mut buffer, fct, forward);
        for (position, value) in buffer.iter().enumerate() {
            output[output_lines.index(line, position)] = from_cmplx(*value);
        }
    }
    output
}

/// `rfft_impl`: each line's first `n` points (zero-padded) transformed to halfcomplex order,
/// then unpacked into `n / 2 + 1` complex values. The buffer holds one spare zero so that an
/// even `n` unpacks its Nyquist term with a zero imaginary part.
fn real_forward_lines<T: Real>(
    input: &[f64],
    lines: Lines,
    output_lines: Lines,
    n: usize,
    fct: f64,
) -> Vec<C128> {
    let output_length = output_lines.length;
    let mut output = vec![C128::default(); output_lines.count() * output_length];
    if lines.count() == 0 {
        return output;
    }
    let plan = RealPlan::<T>::new(n);
    let fct = T::from_f64(fct);
    let mut buffer = vec![T::default(); 2 * output_length - 1];
    let copied = lines.length.min(n);
    for line in 0..lines.count() {
        for (position, value) in buffer.iter_mut().enumerate() {
            *value = if position < copied {
                T::from_f64(input[lines.index(line, position)])
            } else {
                T::default()
            };
        }
        plan.exec(&mut buffer[..n], fct, true);
        output[output_lines.index(line, 0)] = C128 {
            re: buffer[0].to_f64(),
            im: 0.0,
        };
        for k in 1..output_length {
            output[output_lines.index(line, k)] = C128 {
                re: buffer[2 * k - 1].to_f64(),
                im: buffer[2 * k].to_f64(),
            };
        }
    }
    output
}

/// `irfft_loop`: each line's leading frequencies packed into halfcomplex order, dropping the
/// imaginary parts of the zero and (for even `n`) Nyquist terms, then transformed back to `n`
/// real points. Missing frequencies count as zero.
fn real_backward_lines<T: Real>(
    input: &[C128],
    lines: Lines,
    output_lines: Lines,
    fct: f64,
) -> Vec<f64> {
    let n = output_lines.length;
    let mut output = vec![0.0; output_lines.count() * n];
    if lines.count() == 0 {
        return output;
    }
    let plan = RealPlan::<T>::new(n);
    let fct = T::from_f64(fct);
    let mut buffer = vec![T::default(); n];
    for line in 0..lines.count() {
        let frequency = |k: usize| {
            to_cmplx::<T>(if k < lines.length {
                input[lines.index(line, k)]
            } else {
                C128::default()
            })
        };
        buffer[0] = frequency(0).r;
        for k in 1..=(n - 1) / 2 {
            let value = frequency(k);
            buffer[2 * k - 1] = value.r;
            buffer[2 * k] = value.i;
        }
        if n.is_multiple_of(2) {
            buffer[n - 1] = frequency(n / 2).r;
        }
        plan.exec(&mut buffer, fct, false);
        for (position, value) in buffer.iter().enumerate() {
            output[output_lines.index(line, position)] = value.to_f64();
        }
    }
    output
}
