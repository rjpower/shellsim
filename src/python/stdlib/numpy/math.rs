//! Native kernels for `round`/`around` and the `ndarray.clip` trampoline.
//!
//! `np.divmod`, `interp` (piecewise linear interpolation), `correlate`/`convolve`
//! (cross-correlation), and `clip` are frozen Python in `numpy._math`: `divmod` is
//! `floor_divide`/`remainder`, interpolation is `searchsorted` plus vectorized arithmetic, and
//! `clip` is the `maximum`/`minimum` ufuncs. This module keeps only the `ndarray.clip`
//! trampoline that reaches the Python `clip`, the same way `squeeze`/`swapaxes`/`trace` reach
//! `numpy._shapes`.
//!
//! `round` stays native: it reads and writes one generic [`element::Number`] per element (via
//! [`element::read_number`]/[`element::write_number`]), so it preserves any numeric dtype
//! without per-dtype dispatch. The `__divmod__` operator protocol on ndarrays is a different,
//! unrelated path from `np.divmod`: it lives in `ufunc::divmod`, called directly by the operator
//! dispatcher rather than registered as a module function here.

use super::super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyArrayBuffer, PyArrayData,
    PyError, PyResult, PyRuntime, PyValue,
};
use super::args::{self, Signature};
use super::array::{self, Array};
use super::convert;
use super::dtype::DType;
use super::element::{self, Number};

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
            return Err(PyError::not_implemented_error(
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

/// The scalar or array `result` boxed the way NumPy returns it: a NumPy scalar for a 0-d
/// result, the array itself otherwise.
fn finish_array(runtime: &mut dyn PyRuntime, result: Array) -> PyResult {
    if result.ndim() == 0 {
        convert::element_to_scalar(runtime, &result, result.view.offset)
    } else {
        Ok(result.value())
    }
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
// clip trampoline
// ---------------------------------------------------------------------------------------------

/// `ndarray.clip(min=None, max=None, out=None)`: reaches the frozen Python `numpy._math._clip`,
/// the same way `squeeze`/`swapaxes`/`trace` reach `numpy._shapes`.
fn method_clip(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    super::reduce::python_method(runtime, "numpy._math", "_clip", receiver, args)
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
        function("round", module_round),
        function("around", module_round),
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
