//! Elementwise helpers built on ufuncs: `isclose`, `allclose`, `array_equal`, `round`, `clip`,
//! `real`/`imag`/`angle`, `nan_to_num`, `diff`, `gradient`, `interp`, polynomials, `convolve`,
//! `cov`, and `corrcoef`.
//!
//! Functions are exported through the native module `_numpy_math`, which the frozen `numpy`
//! package re-exports.

use super::super::super::native::{
    CallArgs, FunctionDef, ModuleDef, NativeFn, NativeTypeDef, PyResult, PyRuntime,
};
use super::dtype::{Category, DType};
use super::element::Number;

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_math",
    functions: FUNCTIONS,
    values: &[],
};

static FUNCTIONS: &[FunctionDef] = &[function("divmod", divmod)];

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

/// Methods this area installs on `numpy.ndarray`.
pub(in crate::python) static ARRAY_METHODS: NativeTypeDef = NativeTypeDef {
    name: "numpy.ndarray",
    methods: &[],
    getters: &[],
};

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
