//! Array `repr` and `str`.
//!
//! This is the minimal layout: elements joined with `, ` (repr) or a space (str), nested
//! brackets per axis, and a `dtype=` suffix for dtypes other than the defaults NumPy omits.
//! Column padding, float precision trimming, line wrapping, summarization, and print options
//! are not modeled yet.

use super::super::super::native::{PyResult, PyRuntime};
use super::array::Array;
use super::convert::{self, Element};
use super::dtype::{Category, DType, Kind};
use super::element::{read_number, Number};
use super::format::{complex_repr, float_repr};
use super::scalar::precision;

/// Text of one element inside an array.
fn element_text(runtime: &mut dyn PyRuntime, array: &Array, offset: usize) -> PyResult<String> {
    Ok(match convert::read_element(runtime, array, offset)? {
        Element::Number(bytes) => match read_number(array.dtype.kind(), &bytes) {
            Number::Bool(value) => if value { "True" } else { "False" }.to_string(),
            Number::Int(value) => value.to_string(),
            Number::UInt(value) => value.to_string(),
            Number::Float(value) => trim_float(float_repr(value, precision(array.dtype))),
            Number::Complex(real, imag) => complex_repr(real, imag, precision(array.dtype)),
        },
        Element::Str(text) => format!("'{text}'"),
        Element::Object(value) => runtime.repr(&value)?,
    })
}

/// NumPy prints integral floats as `1.` rather than `1.0`.
fn trim_float(text: String) -> String {
    match text.strip_suffix(".0") {
        Some(stripped) => format!("{stripped}."),
        None => text,
    }
}

fn nested(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    axis: usize,
    offset: usize,
    separator: &str,
    indent: usize,
) -> PyResult<String> {
    if axis == array.ndim() {
        return element_text(runtime, array, offset);
    }
    let length = array.shape()[axis];
    let stride = array.strides()[axis];
    let mut parts = Vec::with_capacity(length);
    for position in 0..length {
        let offset = (offset as isize + position as isize * stride) as usize;
        parts.push(nested(runtime, array, axis + 1, offset, separator, indent + 1)?);
    }
    let rows_separator = if axis + 1 < array.ndim() {
        let blank_lines = "\n".repeat(array.ndim() - axis - 1);
        format!("{}{blank_lines}{}", separator.trim_end(), " ".repeat(indent + 1))
    } else {
        separator.to_string()
    };
    Ok(format!("[{}]", parts.join(&rows_separator)))
}

/// Whether `repr` omits the dtype, as NumPy does for its default dtypes.
fn implicit_dtype(dtype: DType) -> bool {
    matches!(
        dtype.kind(),
        Kind::Bool | Kind::Int64 | Kind::Float64 | Kind::Complex128
    )
}

pub(in crate::python) fn array_repr(runtime: &mut dyn PyRuntime, array: &Array) -> PyResult<String> {
    runtime.charge_cpu(array.size() as u64 + 1)?;
    let prefix = "array(";
    let body = nested(runtime, array, 0, array.view.offset, ", ", prefix.len())?;
    let show_dtype = !implicit_dtype(array.dtype) || array.size() == 0;
    Ok(if show_dtype {
        let dtype = match array.dtype.category() {
            Category::Str => format!("'{}'", array.dtype.descr()),
            _ => array.dtype.name(),
        };
        format!("{prefix}{body}, dtype={dtype})")
    } else {
        format!("{prefix}{body})")
    })
}

pub(in crate::python) fn array_str(runtime: &mut dyn PyRuntime, array: &Array) -> PyResult<String> {
    runtime.charge_cpu(array.size() as u64 + 1)?;
    nested(runtime, array, 0, array.view.offset, " ", 0)
}
