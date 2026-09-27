//! Native kernels behind NumPy's Python-level math helpers: `interp` (NumPy's
//! `compiled_interp`), `correlate` and `convolve` (`correlate2` and `correlate`), and
//! `divmod`. The Python wrappers live in `numpy/_function_base.py`, as in NumPy.
//!
//! Functions are exported through the native module `_numpy_math`, which the frozen `numpy`
//! package re-exports.

use super::super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeFn, NativeTypeDef, PyArrayBuffer, PyError,
    PyResult, PyRuntime, PyValue,
};
use super::args::{self, Signature};
use super::array::{self, Array};
use super::convert;
use super::dtype::{self, Category, DType, Kind};
use super::element::{self, Complex, Element, Number, F16};
use super::ops::{FpFlags, Numeric};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_math",
    functions: FUNCTIONS,
    values: &[],
};

static FUNCTIONS: &[FunctionDef] = &[
    function("divmod", divmod),
    function("_compiled_interp", compiled_interp),
    function("_compiled_interp_complex", compiled_interp_complex),
    function("_correlate", correlate),
];

const fn function(name: &'static str, call: NativeFn) -> FunctionDef {
    FunctionDef {
        module: "numpy",
        name,
        call,
    }
}

fn divmod(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    super::ufunc::call_divmod(runtime, args)
}

/// Index `j` with `xp[j] <= key < xp[j + 1]`, or `-1` below the range and `len` above it,
/// using NumPy's `binary_search_with_guess`. The guess, the previous result, makes sorted
/// queries cheap; for unsorted `xp` the path it takes decides the answer, so it is ported
/// exactly.
fn search_with_guess(key: f64, xp: &[f64], guess: isize) -> isize {
    const LIKELY_IN_CACHE_SIZE: isize = 8;
    let len = xp.len() as isize;
    let at = |index: isize| xp[index as usize];
    if key > at(len - 1) {
        return len;
    }
    if key < at(0) {
        return -1;
    }
    if len <= 4 {
        let mut index = 1;
        while index < len && key >= at(index) {
            index += 1;
        }
        return index - 1;
    }
    let guess = guess.min(len - 3).max(1);
    let (mut low, mut high) = (0, len);
    if key < at(guess) {
        if key < at(guess - 1) {
            high = guess - 1;
            if guess > LIKELY_IN_CACHE_SIZE && key >= at(guess - LIKELY_IN_CACHE_SIZE) {
                low = guess - LIKELY_IN_CACHE_SIZE;
            }
        } else {
            return guess - 1;
        }
    } else if key < at(guess + 1) {
        return guess;
    } else if key < at(guess + 2) {
        return guess + 1;
    } else {
        low = guess + 2;
        if guess < len - LIKELY_IN_CACHE_SIZE - 1 && key < at(guess + LIKELY_IN_CACHE_SIZE) {
            high = guess + LIKELY_IN_CACHE_SIZE;
        }
    }
    while low < high {
        let middle = low + ((high - low) >> 1);
        if key >= at(middle) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low - 1
}

/// The values one interpolation reads: `x` as float64 in C order with its shape, and `xp` as a
/// contiguous float64 vector. `fp` is converted by the caller.
struct Samples {
    x: Vec<f64>,
    shape: Vec<usize>,
    xp: Vec<f64>,
}

/// Convert `x` and `xp` as `PyArray_ContiguousFromAny(..., NPY_DOUBLE)` does, checking that
/// `xp` and `fp` are 1-d and of equal length.
fn samples(runtime: &mut dyn PyRuntime, bound: &args::Bound, fp: &Array) -> PyResult<Samples> {
    let xp = one_dimensional(runtime, bound.required("xp"), DType::FLOAT64)?;
    let x = convert::array_from_python(runtime, bound.required("x"), Some(DType::FLOAT64), false)?;
    let shape = x.shape().to_vec();
    let x = array::read_elements::<f64>(runtime, &x)?;
    let xp = array::read_elements::<f64>(runtime, &xp)?;
    if xp.is_empty() && !x.is_empty() {
        return Err(PyError::value_error("array of sample points is empty"));
    }
    if fp.size() != xp.len() {
        return Err(PyError::value_error(
            "fp and xp are not of the same length.",
        ));
    }
    Ok(Samples { x, shape, xp })
}

/// A value converted to a 1-d array of `dtype`, with NumPy's depth errors.
fn one_dimensional(runtime: &mut dyn PyRuntime, value: PyValue, dtype: DType) -> PyResult<Array> {
    let array = convert::array_from_python(runtime, value, Some(dtype), false)?;
    match array.ndim() {
        0 => Err(PyError::value_error(
            "object of too small depth for desired array",
        )),
        1 => Ok(array),
        _ => Err(PyError::value_error("object too deep for desired array")),
    }
}

static INTERP: Signature = Signature::new("interp", &["x", "xp", "fp", "left", "right"], 3);

/// A float64 result with `x`'s shape; a 0-d result is a scalar, as `PyArray_Return` gives.
fn interp_result<T: Element>(
    runtime: &mut dyn PyRuntime,
    dtype: DType,
    shape: Vec<usize>,
    values: &[T],
) -> PyResult {
    let result = array::array_from_elements(runtime, dtype, shape, values)?;
    if result.ndim() == 0 {
        return convert::element_to_scalar(runtime, &result, result.view.offset);
    }
    Ok(result.value())
}

/// `numpy._core.multiarray.interp(x, xp, fp, left=None, right=None)`: NumPy's `arr_interp`.
fn compiled_interp(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = INTERP.bind(&args)?;
    let fp = one_dimensional(runtime, bound.required("fp"), DType::FLOAT64)?;
    let Samples { x, shape, xp } = samples(runtime, &bound, &fp)?;
    let fp = array::read_elements::<f64>(runtime, &fp)?;
    if x.is_empty() {
        return interp_result::<f64>(runtime, DType::FLOAT64, shape, &[]);
    }
    let fill = |runtime: &mut dyn PyRuntime, name: &str, default: f64| match bound.value(name) {
        Some(value) => args::float_arg(runtime, &value),
        None => Ok(default),
    };
    let left = fill(runtime, "left", fp[0])?;
    let right = fill(runtime, "right", fp[fp.len() - 1])?;
    runtime.charge_cpu((x.len() as u64).saturating_mul(4) + xp.len() as u64)?;
    runtime.reserve_memory(x.len().saturating_mul(8))?;
    let result = if xp.len() == 1 {
        x.iter()
            .map(|&value| {
                if value < xp[0] {
                    left
                } else if value > xp[0] {
                    right
                } else {
                    fp[0]
                }
            })
            .collect::<Vec<_>>()
    } else {
        let slope = |j: usize| (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
        let mut guess = 0isize;
        x.iter()
            .map(|&value| {
                if value.is_nan() {
                    return value;
                }
                guess = search_with_guess(value, &xp, guess);
                let last = xp.len() as isize - 1;
                if guess == -1 {
                    return left;
                }
                if guess > last {
                    return right;
                }
                let j = guess as usize;
                if guess == last || xp[j] == value {
                    return fp[j];
                }
                let slope = slope(j);
                // If one direction gives NaN, NumPy tries the other.
                let mut result = slope * (value - xp[j]) + fp[j];
                if result.is_nan() {
                    result = slope * (value - xp[j + 1]) + fp[j + 1];
                    if result.is_nan() && fp[j] == fp[j + 1] {
                        result = fp[j];
                    }
                }
                result
            })
            .collect()
    };
    interp_result(runtime, DType::FLOAT64, shape, &result)
}

/// A `left=` or `right=` fill for complex interpolation, as `PyComplex_RealAsDouble` and
/// `PyComplex_ImagAsDouble` read it.
fn complex_arg(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<(f64, f64)> {
    use super::super::super::number::NumberRef;
    if let Some((_, number)) = super::scalar::unbox_number(runtime, value) {
        return Ok(number.as_complex());
    }
    if let Some(NumberRef::Complex(real, imag)) = runtime.number(value) {
        return Ok((real, imag));
    }
    Ok((args::float_arg(runtime, value)?, 0.0))
}

/// `numpy._core.multiarray.interp_complex`: `arr_interp_complex`, which scales by the inverse
/// of each interval rather than dividing.
fn compiled_interp_complex(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = INTERP.bind(&args)?;
    let fp = one_dimensional(runtime, bound.required("fp"), DType::COMPLEX128)?;
    let Samples { x, shape, xp } = samples(runtime, &bound, &fp)?;
    let fp = array::read_elements::<element::C128>(runtime, &fp)?;
    if x.is_empty() {
        return interp_result::<element::C128>(runtime, DType::COMPLEX128, shape, &[]);
    }
    let (left, right) = (bound.value("left"), bound.value("right"));
    let left = match left {
        Some(value) => complex_arg(runtime, &value)?,
        None => (fp[0].re, fp[0].im),
    };
    let right = match right {
        Some(value) => complex_arg(runtime, &value)?,
        None => (fp[fp.len() - 1].re, fp[fp.len() - 1].im),
    };
    let (left, right) = (
        Complex {
            re: left.0,
            im: left.1,
        },
        Complex {
            re: right.0,
            im: right.1,
        },
    );
    runtime.charge_cpu((x.len() as u64).saturating_mul(8) + xp.len() as u64)?;
    runtime.reserve_memory(x.len().saturating_mul(16))?;
    // One part of the interpolated value, retried from the right end when it is NaN.
    let part = |slope: f64, value: f64, j: usize, low: f64, high: f64| {
        let mut result = slope * (value - xp[j]) + low;
        if result.is_nan() {
            result = slope * (value - xp[j + 1]) + high;
            if result.is_nan() && low == high {
                result = low;
            }
        }
        result
    };
    let result = if xp.len() == 1 {
        x.iter()
            .map(|&value| {
                if value < xp[0] {
                    left
                } else if value > xp[0] {
                    right
                } else {
                    fp[0]
                }
            })
            .collect::<Vec<_>>()
    } else {
        let mut guess = 0isize;
        x.iter()
            .map(|&value| {
                if value.is_nan() {
                    return Complex { re: value, im: 0.0 };
                }
                guess = search_with_guess(value, &xp, guess);
                let last = xp.len() as isize - 1;
                if guess == -1 {
                    return left;
                }
                if guess > last {
                    return right;
                }
                let j = guess as usize;
                if guess == last || xp[j] == value {
                    return fp[j];
                }
                let inverse = 1.0 / (xp[j + 1] - xp[j]);
                let slope_re = (fp[j + 1].re - fp[j].re) * inverse;
                let slope_im = (fp[j + 1].im - fp[j].im) * inverse;
                Complex {
                    re: part(slope_re, value, j, fp[j].re, fp[j + 1].re),
                    im: part(slope_im, value, j, fp[j].im, fp[j + 1].im),
                }
            })
            .collect()
    };
    interp_result(runtime, DType::COMPLEX128, shape, &result)
}

/// A correlation mode, as `PyArray_CorrelatemodeConverter` parses it.
#[derive(Clone, Copy)]
enum Mode {
    Valid,
    Same,
    Full,
}

fn correlate_mode(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<Mode> {
    if let Some(text) = runtime.string_value(value)? {
        let mode = match text.chars().next() {
            Some('v' | 'V') => (Mode::Valid, "valid"),
            Some('s' | 'S') => (Mode::Same, "same"),
            Some('f' | 'F') => (Mode::Full, "full"),
            _ => {
                return Err(PyError::value_error(format!(
                    "mode must be one of 'valid', 'same', or 'full' (got {})",
                    runtime.repr(value)?
                )))
            }
        };
        if text != mode.1 {
            return Err(PyError::value_error(
                "Use one of 'valid', 'same', or 'full' for convolve/correlate mode",
            ));
        }
        return Ok(mode.0);
    }
    let number = runtime
        .int_value(value)
        .ok_or_else(|| PyError::type_error("convolve/correlate mode not understood"))?;
    match number {
        0 => Ok(Mode::Valid),
        1 => Ok(Mode::Same),
        2 => Ok(Mode::Full),
        _ => Err(PyError::value_error(
            "integer convolve/correlate mode must be 0, 1, or 2",
        )),
    }
}

/// NumPy's `@name@_dot` loops: a sequential sum of products from zero, in the dtype's own
/// arithmetic. NumPy hands long float dot products to BLAS, whose summation order depends on
/// the host CPU, so results over many float terms can differ from NumPy in the last bits.
fn dot<T: Numeric>(a: &[T], b: &[T]) -> T {
    let mut flags = FpFlags::default();
    a.iter().zip(b).fold(T::zero(), |sum, (x, y)| {
        sum.add(x.multiply(*y, &mut flags), &mut flags)
    })
}

/// `HALF_dot` accumulates in single precision and rounds once at the end.
fn half_dot(a: &[F16], b: &[F16]) -> F16 {
    F16::from_f32(
        a.iter()
            .zip(b)
            .fold(0f32, |sum, (x, y)| sum + x.to_f32() * y.to_f32()),
    )
}

/// NumPy's `_pyarray_correlate` over typed values: `a` is at least as long as `v`, and both
/// are non-empty. Edge outputs use partial overlaps; the middle uses the whole of `v`.
fn correlate_values<T: Copy>(
    a: &[T],
    v: &[T],
    mode: Mode,
    dot: impl Fn(&[T], &[T]) -> T,
) -> Vec<T> {
    let (n1, n) = (a.len(), v.len());
    let (n_left, n_right) = match mode {
        Mode::Valid => (0, 0),
        Mode::Same => (n / 2, n - n / 2 - 1),
        Mode::Full => (n - 1, n - 1),
    };
    let mut output = Vec::with_capacity(n1 - n + 1 + n_left + n_right);
    for step in 0..n_left {
        let start = n_left - step;
        let count = n - start;
        output.push(dot(&a[..count], &v[start..start + count]));
    }
    for start in 0..=n1 - n {
        output.push(dot(&a[start..start + n], v));
    }
    for step in 0..n_right {
        let start = n1 - n + 1 + step;
        let count = n - 1 - step;
        output.push(dot(&a[start..start + count], &v[..count]));
    }
    output
}

/// `_correlate(a, v, mode, conjugate)`: `conjugate=True` is NumPy's `correlate2`, behind
/// `np.correlate`, which conjugates complex `v` and reverses the result when the inputs were
/// swapped; `conjugate=False` is the older `correlate` behind `np.convolve`.
fn correlate(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("correlate", &["a", "v", "mode", "conjugate"], 4);
    let bound = SIGNATURE.bind(&args)?;
    let mode = correlate_mode(runtime, &bound.required("mode"))?;
    let conjugate = runtime.truth(&bound.required("conjugate"))?;
    let first = convert::as_array(runtime, bound.required("a"))?;
    let second = convert::as_array(runtime, bound.required("v"))?;
    let dtype = dtype::promote(first.dtype, second.dtype)?;
    if !dtype.is_numeric() {
        return Err(PyError::unsupported(format!(
            "correlate and convolve support numeric arrays, not {}",
            dtype.repr()
        )));
    }
    let first = one_dimensional(runtime, first.value(), dtype)?;
    let second = one_dimensional(runtime, second.value(), dtype)?;
    if first.size() == 0 {
        return Err(PyError::value_error("first array argument cannot be empty"));
    }
    if second.size() == 0 {
        return Err(PyError::value_error(
            "second array argument cannot be empty",
        ));
    }
    let inverted = first.size() < second.size();
    let (long, short) = if inverted {
        (second, first)
    } else {
        (first, second)
    };
    let work = (long.size() as u64).saturating_mul(short.size() as u64);
    runtime.charge_cpu(work.saturating_add(1))?;
    let length = match mode {
        Mode::Valid => long.size() - short.size() + 1,
        Mode::Same => long.size(),
        Mode::Full => long.size() + short.size() - 1,
    };
    array::reserve_elements(runtime, dtype, length)?;
    // `correlate2` conjugates its second argument before any swap.
    let conjugate_short = conjugate && dtype.category() == Category::Complex && !inverted;
    let conjugate_long = conjugate && dtype.category() == Category::Complex && inverted;
    let result = if dtype.kind() == Kind::Float16 {
        let long = array::read_elements::<F16>(runtime, &long)?;
        let short = array::read_elements::<F16>(runtime, &short)?;
        let mut values = correlate_values(&long, &short, mode, half_dot);
        if inverted && conjugate {
            values.reverse();
        }
        array::pack_elements(&values)
    } else {
        element::dispatch_numeric!(dtype.kind(), T => {
            let mut long = array::read_elements::<T>(runtime, &long)?;
            let mut short = array::read_elements::<T>(runtime, &short)?;
            for values in [(&mut long, conjugate_long), (&mut short, conjugate_short)]
                .into_iter()
                .filter(|(_, conjugate)| *conjugate)
                .map(|(values, _)| values)
            {
                for value in values.iter_mut() {
                    if let Number::Complex(real, imag) = value.to_number() {
                        *value = T::from_number(Number::Complex(real, -imag));
                    }
                }
            }
            let mut values = correlate_values(&long, &short, mode, dot::<T>);
            if inverted && conjugate {
                values.reverse();
            }
            array::pack_elements(&values)
        }, _ => unreachable!("correlate checked for a numeric dtype"))
    };
    let result = array::new_array(runtime, PyArrayBuffer::Bytes(result), dtype, vec![length])?;
    Ok(result.value())
}

/// Methods this area installs on `numpy.ndarray`. They run NumPy's Python implementations in
/// `numpy._methods`.
pub(in crate::python) static ARRAY_METHODS: NativeTypeDef = NativeTypeDef {
    name: "numpy.ndarray",
    methods: &[
        MethodDef {
            type_name: "numpy.ndarray",
            name: "round",
            call: method_round,
        },
        MethodDef {
            type_name: "numpy.ndarray",
            name: "clip",
            call: method_clip,
        },
    ],
    getters: &[],
};

fn method_round(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    super::reduce::python_method(runtime, "_round", receiver, args)
}

fn method_clip(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    super::reduce::python_method(runtime, "_clip", receiver, args)
}

/// Round `value` of `dtype` to `decimals` places the way NumPy's `round` does:
/// `rint(x * 10**decimals) / 10**decimals`, ties to even, computed at the dtype's float precision.
/// Integers are unchanged for `decimals >= 0`; otherwise they round through float64 and wrap on
/// the cast back, so `round(np.uint8(255), -1)` is `4` as in NumPy.
pub(in crate::python) fn round_number(dtype: DType, value: Number, decimals: i64) -> Number {
    let factor = 10f64.powi(i32::try_from(decimals.unsigned_abs()).unwrap_or(i32::MAX));
    let scale = |value: f64| {
        if decimals >= 0 {
            (value * factor).round_ties_even() / factor
        } else {
            (value / factor).round_ties_even() * factor
        }
    };
    let scale_single = |value: f64| {
        let factor = factor as f32;
        let value = value as f32;
        f64::from(if decimals >= 0 {
            (value * factor).round_ties_even() / factor
        } else {
            (value / factor).round_ties_even() * factor
        })
    };
    let single =
        dtype.itemsize() <= 4 || dtype.category() == Category::Complex && dtype.itemsize() == 8;
    let round = |value: f64| {
        if single {
            scale_single(value)
        } else {
            scale(value)
        }
    };
    match value {
        Number::Bool(_) => value,
        Number::Int(_) | Number::UInt(_) if decimals >= 0 => value,
        Number::Int(integer) => Number::Int(scale(integer as f64) as i64),
        Number::UInt(integer) => Number::Int(scale(integer as f64) as i64),
        Number::Float(float) => Number::Float(round(float)),
        Number::Complex(real, imag) => Number::Complex(round(real), round(imag)),
    }
}
