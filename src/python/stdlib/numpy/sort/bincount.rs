//! `numpy.bincount`: counts, or weighted sums, of small non-negative integers.
//!
//! The output has one slot per integer from `0` to `max(x.max(), minlength - 1)`; slot `i`
//! holds the count of `i` in `x` (or, with `weights`, the sum of the matching weights). Cost is
//! charged once per input element before the accumulation loop runs, and the output length is
//! reserved before it is allocated, so a single huge value in `x` cannot force an unbounded
//! allocation for free.

use super::super::super::super::native::{CallArgs, PyError, PyResult, PyRuntime, PyValue};
use super::super::args::{self, Signature};
use super::super::array::{self, Array};
use super::super::convert;
use super::super::dtype::{Casting, DType};

static SIGNATURE: Signature =
    Signature::new("bincount", &["x"], 1).keyword_only(&["weights", "minlength"]);

pub(in crate::python) fn module_bincount<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let bound = SIGNATURE.bind(&args)?;
    bincount(
        runtime,
        bound.required("x"),
        bound.value("weights"),
        bound.value("minlength"),
    )
}

fn depth_error(ndim: usize) -> PyError {
    if ndim == 0 {
        PyError::value_error("object of too small depth for desired array")
    } else {
        PyError::value_error("object too deep for desired array")
    }
}

fn bincount<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    x: PyValue<'s>,
    weights: Option<PyValue<'s>>,
    minlength: Option<PyValue<'s>>,
) -> PyResult<'s> {
    let x = convert::as_array(runtime, x)?;
    if x.ndim() != 1 {
        return Err(depth_error(x.ndim()));
    }
    if !super::super::dtype::can_cast(x.dtype, DType::INT64, Casting::Safe) {
        return Err(PyError::type_error(format!(
            "Cannot cast array data from {} to {} according to the rule 'safe'",
            x.dtype.repr(),
            DType::INT64.repr()
        )));
    }
    let x64 = convert::cast_array(runtime, &x, DType::INT64, false)?;
    let values = array::read_elements::<i64>(runtime, &x64)?;
    let minlength = match minlength {
        Some(value) => {
            let requested = args::index_int(runtime, &value)?;
            if requested < 0 {
                return Err(PyError::value_error("'minlength' must not be negative"));
            }
            requested as usize
        }
        None => 0,
    };
    runtime.charge_cpu(values.len() as u64 + 1)?;
    let mut highest = -1i64;
    for &value in &values {
        if value < 0 {
            return Err(PyError::value_error(
                "'list' argument must have no negative elements",
            ));
        }
        highest = highest.max(value);
    }
    let length = usize::try_from(highest + 1).unwrap_or(0).max(minlength);

    match weights {
        None => {
            array::reserve_elements(runtime, DType::INT64, length)?;
            let mut counts = vec![0i64; length];
            for &value in &values {
                counts[value as usize] += 1;
            }
            Ok(array::array_from_elements(runtime, DType::INT64, vec![length], &counts)?.value())
        }
        Some(weights) => {
            let weights = weights_array(runtime, weights, values.len())?;
            array::reserve_elements(runtime, DType::FLOAT64, length)?;
            let mut sums = vec![0f64; length];
            for (&value, &weight) in values.iter().zip(&weights) {
                sums[value as usize] += weight;
            }
            Ok(array::array_from_elements(runtime, DType::FLOAT64, vec![length], &sums)?.value())
        }
    }
}

fn weights_array<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    weights: PyValue<'s>,
    n: usize,
) -> PyResult<'s, Vec<f64>> {
    let weights = convert::as_array(runtime, weights)?;
    if weights.size() != n {
        return Err(PyError::value_error(
            "The weights and list don't have the same length.",
        ));
    }
    let cast: Array = convert::cast_array(runtime, &weights, DType::FLOAT64, false)?;
    array::read_elements::<f64>(runtime, &cast)
}
