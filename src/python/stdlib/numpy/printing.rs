//! Array printing: the ndarray `repr`/`str` slots and the native `_numpy_print` module.
//!
//! The layout rules live in the frozen `numpy/_arrayprint.py`; the slots call its
//! `_array_repr_implementation` and `_array_str_implementation`, which read the current print
//! options. This module supplies the part that code needs natively: digit generation as
//! `format_positional` and `format_scientific`, where `-1` means an unset option.
//!
//! A NumPy `float16` or `float32` scalar is formatted at its own precision, so its shortest
//! digits identify it among values of that width. Any other argument is converted with
//! `float()` and formatted as a double.

use super::super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyKind, PyResult, PyRuntime, PyValue,
};
use super::args::{float_arg, index_int, Bound, Signature};
use super::element::Number;
use super::float_digits::{self, Options, Trim};
use super::format::Precision;
use super::scalar::{precision, unbox_number};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_print",
    functions: &[
        FunctionDef {
            module: "numpy",
            name: "format_positional",
            call: format_positional,
        },
        FunctionDef {
            module: "numpy",
            name: "format_scientific",
            call: format_scientific,
        },
    ],
    values: &[],
};

/// `repr(array)` or `str(array)` through `numpy._arrayprint.<function>`.
pub(in crate::python) fn array_text(
    runtime: &mut dyn PyRuntime,
    array: PyValue,
    function: &str,
) -> PyResult<PyValue> {
    let module = runtime.import_module("numpy._arrayprint")?;
    let implementation = runtime.get_attribute(module, function)?.ok_or_else(|| {
        PyError::runtime_error(format!("numpy._arrayprint.{function} is missing"))
    })?;
    runtime.call_value(implementation, CallArgs::new(vec![array], Vec::new()))
}

/// The value to format and the precision whose shortest digits identify it.
fn float_operand(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<(f64, Precision)> {
    match unbox_number(runtime, value) {
        Some((dtype, Number::Float(number))) => Ok((number, precision(dtype))),
        Some((_, Number::Complex(..))) => Err(PyError::type_error(format!(
            "must be real number, not {}",
            runtime.type_name(value)?
        ))),
        _ => Ok((float_arg(runtime, value)?, Precision::Double)),
    }
}

/// An `int` option; NumPy converts these to a C `int`.
fn int_option(
    runtime: &mut dyn PyRuntime,
    bound: &Bound,
    name: &str,
    default: i32,
) -> PyResult<i32> {
    let Some(value) = bound.get(name) else {
        return Ok(default);
    };
    let value = index_int(runtime, &value)?;
    i32::try_from(value)
        .map_err(|_| PyError::overflow_error("Python int too large to convert to C int"))
}

fn trim_option(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult<Trim> {
    let Some(value) = bound.get("trim") else {
        return Ok(Trim::Keep);
    };
    let code = match runtime.kind(&value)? {
        PyKind::String => runtime.string_value(&value)?,
        _ => None,
    };
    match code.as_deref().and_then(Trim::parse) {
        Some(trim) => Ok(trim),
        // NumPy formats the value with `%100S`, which right-aligns it in 100 columns.
        None => Err(PyError::type_error(format!(
            "if supplied, trim must be 'k', '.', '0' or '-' found `{:>100}`",
            runtime.display(&value)?
        ))),
    }
}

/// The options both entry points share.
fn common_options(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult<Options> {
    let options = Options {
        precision: int_option(runtime, bound, "precision", -1)?,
        unique: int_option(runtime, bound, "unique", 1)? != 0,
        sign: int_option(runtime, bound, "sign", 0)? != 0,
        trim: trim_option(runtime, bound)?,
        pad_left: int_option(runtime, bound, "pad_left", -1)?,
        min_digits: int_option(runtime, bound, "min_digits", -1)?,
        ..Options::new()
    };
    if !options.unique && options.precision < 0 {
        return Err(PyError::type_error(
            "in non-unique mode `precision` must be supplied",
        ));
    }
    Ok(options)
}

/// `format_positional(x, precision=-1, unique=1, fractional=0, sign=0, trim='k', pad_left=-1,
/// pad_right=-1, min_digits=-1)`.
fn format_positional(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "format_positional",
        &[
            "x",
            "precision",
            "unique",
            "fractional",
            "sign",
            "trim",
            "pad_left",
            "pad_right",
            "min_digits",
        ],
        1,
    );
    let bound = SIGNATURE.bind(&args)?;
    let (value, precision) = float_operand(runtime, &bound.required("x"))?;
    let options = Options {
        fractional: int_option(runtime, &bound, "fractional", 0)? != 0,
        pad_right: int_option(runtime, &bound, "pad_right", -1)?,
        ..common_options(runtime, &bound)?
    };
    let mut work = 0;
    let text = float_digits::positional(value, precision, &options, &mut work)?;
    runtime.charge_cpu(work)?;
    runtime.new_string(text)
}

/// `format_scientific(x, precision=-1, unique=1, sign=0, trim='k', pad_left=-1,
/// exp_digits=-1, min_digits=-1)`.
fn format_scientific(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "format_scientific",
        &[
            "x",
            "precision",
            "unique",
            "sign",
            "trim",
            "pad_left",
            "exp_digits",
            "min_digits",
        ],
        1,
    );
    let bound = SIGNATURE.bind(&args)?;
    let (value, precision) = float_operand(runtime, &bound.required("x"))?;
    let options = Options {
        exp_digits: int_option(runtime, &bound, "exp_digits", -1)?,
        ..common_options(runtime, &bound)?
    };
    let mut work = 0;
    let text = float_digits::scientific(value, precision, &options, &mut work)?;
    runtime.charge_cpu(work)?;
    runtime.new_string(text)
}
