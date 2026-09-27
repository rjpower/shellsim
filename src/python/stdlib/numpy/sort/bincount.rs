//! `np.bincount(x, /, weights=None, minlength=0)`, following `arr_bincount`.
//!
//! Counts are `int64`; with `weights` they are `float64` sums added in input order. The
//! result has one slot per value up to the largest, so its memory is reserved before
//! counting.

use super::super::super::super::native::{
    CallArgs, PyError, PyNativeKind, PyResult, PyRuntime, PyValue,
};
use super::super::args::{self, Signature};
use super::super::array;
use super::super::convert;
use super::super::dtype::{Category, DType};
use super::super::math::one_dimensional;

/// The values to count, as `arr_bincount` converts them: integers cast to `int64`, anything
/// else only when the cast is safe. An empty list is accepted whatever dtype it infers.
fn counted_values(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<i64>> {
    let is_array = runtime.native_kind(&value)? == Some(PyNativeKind::Array);
    let array = convert::as_array(runtime, value)?;
    match array.ndim() {
        0 => {
            return Err(PyError::value_error(
                "object of too small depth for desired array",
            ))
        }
        1 => {}
        _ => return Err(super::too_deep(is_array)),
    }
    if !is_array && array.size() == 0 {
        return Ok(Vec::new());
    }
    if !matches!(
        array.dtype.category(),
        Category::Bool | Category::Signed | Category::Unsigned
    ) {
        return Err(PyError::type_error(format!(
            "Cannot cast array data from {} to {} according to the rule 'safe'",
            array.dtype.repr(),
            DType::INT64.repr()
        )));
    }
    let array = convert::cast_array(runtime, &array, DType::INT64, false)?;
    array::read_elements::<i64>(runtime, &array)
}

pub(super) fn module_bincount(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("bincount", &["x", "weights", "minlength"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let values = counted_values(runtime, bound.required("x"))?;
    let minlength = match bound.get("minlength") {
        Some(value) if value.is_none() => {
            return Err(PyError::type_error("use 0 instead of None for minlength"))
        }
        Some(value) => args::index_int(runtime, &value)?,
        None => 0,
    };
    let minlength = usize::try_from(minlength)
        .map_err(|_| PyError::value_error("'minlength' must not be negative"))?;
    if values.is_empty() {
        runtime.reserve_memory(minlength.saturating_mul(8))?;
        let counts = vec![0i64; minlength];
        return Ok(
            array::array_from_elements(runtime, DType::INT64, vec![minlength], &counts)?.value(),
        );
    }
    let (smallest, largest) = values
        .iter()
        .fold((values[0], values[0]), |(low, high), value| {
            (low.min(*value), high.max(*value))
        });
    if smallest < 0 {
        return Err(PyError::value_error(
            "'list' argument must have no negative elements",
        ));
    }
    // `largest` is non-negative here, and reserving memory rejects an oversized result.
    let size = (largest as usize).saturating_add(1).max(minlength);
    runtime.reserve_memory(size.saturating_mul(8))?;
    runtime.charge_cpu(values.len() as u64 + size as u64)?;
    let Some(weights) = bound.value("weights") else {
        let mut counts = vec![0i64; size];
        for value in &values {
            counts[*value as usize] += 1;
        }
        return Ok(array::array_from_elements(runtime, DType::INT64, vec![size], &counts)?.value());
    };
    let weights = one_dimensional(runtime, weights, DType::FLOAT64)?;
    let weights = array::read_elements::<f64>(runtime, &weights)?;
    if weights.len() != values.len() {
        return Err(PyError::value_error(
            "The weights and list don't have the same length.",
        ));
    }
    let mut sums = vec![0.0f64; size];
    for (value, weight) in values.iter().zip(&weights) {
        sums[*value as usize] += weight;
    }
    Ok(array::array_from_elements(runtime, DType::FLOAT64, vec![size], &sums)?.value())
}
