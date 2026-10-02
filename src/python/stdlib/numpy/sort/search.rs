//! `numpy.searchsorted`: binary search of a value into an assumed-sorted 1-D array.
//!
//! NumPy documents `searchsorted`'s result on unsorted input as unspecified, so this always
//! does a plain binary search — no attempt to detect or repair unsorted input — using the same
//! per-dtype order as [`super`]'s sort comparators, so a value inserts where `sort` would place
//! it. `sorter` (typically `argsort`'s own output) reindexes the search without moving `a`:
//! position `i` of the logical sorted sequence reads `a[sorter[i]]`, and the returned index is
//! a position in that logical sequence, matching NumPy.
//!
//! Cost is `m·log2(n)` for `m` values searched against a sequence of length `n`, charged before
//! the searches run.

use std::cmp::Ordering;

use super::super::super::super::native::{
    CallArgs, PyArrayBuffer, PyError, PyResult, PyRuntime, PyValue,
};
use super::super::args::Signature;
use super::super::array::{self, Array};
use super::super::convert;
use super::super::dtype::{self, DType, Kind};
use super::super::element::dispatch_numeric;
use super::{cost_log_n, less_than, SortKey};

static MODULE_SIGNATURE: Signature =
    Signature::new("searchsorted", &["a", "v", "side", "sorter"], 2);
static METHOD_SIGNATURE: Signature = Signature::new("searchsorted", &["v", "side", "sorter"], 1);

pub(in crate::python) fn module_searchsorted<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let bound = MODULE_SIGNATURE.bind(&args)?;
    searchsorted(
        runtime,
        bound.required("a"),
        bound.required("v"),
        bound.value("side"),
        bound.value("sorter"),
    )
}

pub(in crate::python) fn method_searchsorted<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver_value: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let bound = METHOD_SIGNATURE.bind(&args)?;
    searchsorted(
        runtime,
        receiver_value,
        bound.required("v"),
        bound.value("side"),
        bound.value("sorter"),
    )
}

fn depth_error(ndim: usize) -> PyError {
    if ndim == 0 {
        PyError::value_error("object of too small depth for desired array")
    } else {
        PyError::value_error("object too deep for desired array")
    }
}

fn parse_side<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    side: Option<PyValue<'s>>,
) -> PyResult<'s, bool> {
    let Some(value) = side else { return Ok(false) };
    let text = runtime.string_value(&value)?.unwrap_or_default();
    match text.as_str() {
        "left" => Ok(false),
        "right" => Ok(true),
        _ => Err(PyError::value_error(format!(
            "search side must be 'left' or 'right' (got '{text}')"
        ))),
    }
}

/// `sorter`'s int64 values, checked for an integer dtype and matching length.
fn parse_sorter<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    sorter: Option<PyValue<'s>>,
    n: usize,
) -> PyResult<'s, Option<Vec<i64>>> {
    let Some(value) = sorter else { return Ok(None) };
    let array = convert::as_array(runtime, value)?;
    if !array.dtype.is_integer() {
        return Err(PyError::type_error("sorter must only contain integers"));
    }
    if array.size() != n {
        return Err(PyError::value_error("sorter.size must equal a.size"));
    }
    let cast = convert::cast_array(runtime, &array, DType::INT64, false)?;
    array::read_elements::<i64>(runtime, &cast).map(Some)
}

/// `bisect_left`/`bisect_right` of `v` into the logical sequence `cmp(0..n)`, where `cmp(i)`
/// compares the logical sequence's `i`th element against `v`.
fn binary_search(n: usize, right: bool, mut cmp: impl FnMut(usize) -> Ordering) -> usize {
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let advance = match cmp(mid) {
            Ordering::Less => true,
            Ordering::Equal => right,
            Ordering::Greater => false,
        };
        if advance {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn searchsorted<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    a_value: PyValue<'s>,
    v_value: PyValue<'s>,
    side: Option<PyValue<'s>>,
    sorter: Option<PyValue<'s>>,
) -> PyResult<'s> {
    let a = convert::as_array(runtime, a_value)?;
    if a.ndim() != 1 {
        return Err(depth_error(a.ndim()));
    }
    let right = parse_side(runtime, side)?;
    let logical = parse_sorter(runtime, sorter, a.size())?;
    let v = convert::as_array(runtime, v_value)?;
    let common = dtype::promote(a.dtype, v.dtype)?;
    let a_cast = convert::cast_array(runtime, &a, common, false)?;
    let v_cast = convert::cast_array(runtime, &v, common, false)?;
    let n = a_cast.size();
    let count = v_cast.size();
    runtime.charge_cpu(
        (count as u64)
            .saturating_mul(cost_log_n(n))
            .saturating_add(1),
    )?;
    array::reserve_elements(runtime, DType::INT64, count)?;

    let positions = match common.kind() {
        Kind::Object => {
            let a_values = array::read_objects(runtime, &a_cast)?;
            let v_values = array::read_objects(runtime, &v_cast)?;
            let index = |i: usize| logical.as_ref().map_or(i, |order| order[i] as usize);
            let mut positions = Vec::with_capacity(count);
            for &v in &v_values {
                let mut lo = 0usize;
                let mut hi = n;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let element = a_values[index(mid)];
                    let advance = if right {
                        !less_than(runtime, v, element)?
                    } else {
                        less_than(runtime, element, v)?
                    };
                    if advance {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                positions.push(lo as i64);
            }
            positions
        }
        Kind::Str => {
            let width = common.itemsize();
            let a_bytes = str_bytes(runtime, &a_cast)?;
            let v_bytes = str_bytes(runtime, &v_cast)?;
            let index = |i: usize| logical.as_ref().map_or(i, |order| order[i] as usize);
            let read = |bytes: &[u8], i: usize| bytes[i * width..(i + 1) * width].to_vec();
            v_bytes
                .chunks_exact(width)
                .map(|value| {
                    binary_search(n, right, |mid| {
                        super::compare_code_points(&read(&a_bytes, index(mid)), value)
                    }) as i64
                })
                .collect()
        }
        kind => {
            let index = |i: usize| logical.as_ref().map_or(i, |order| order[i] as usize);
            dispatch_numeric!(kind, T => {
                let a_values = array::read_elements::<T>(runtime, &a_cast)?;
                let v_values = array::read_elements::<T>(runtime, &v_cast)?;
                v_values
                    .iter()
                    .map(|&target| binary_search(n, right, |mid| a_values[index(mid)].sort_cmp(target)) as i64)
                    .collect()
            }, _ => unreachable!("Str and Object are handled separately"))
        }
    };

    if v.ndim() == 0 {
        return super::super::scalar::box_number(
            runtime,
            DType::INT64,
            super::super::element::Number::Int(positions[0]),
        );
    }
    Ok(array::array_from_elements(runtime, DType::INT64, v.shape().to_vec(), &positions)?.value())
}

/// `array`'s elements as one contiguous UCS-4 buffer, in C order (1-D, so that is index order).
fn str_bytes<'s>(runtime: &mut dyn PyRuntime<'s>, array: &Array<'s>) -> PyResult<'s, Vec<u8>> {
    match array::contiguous_buffer(runtime, array)? {
        PyArrayBuffer::Bytes(bytes) => Ok(bytes),
        PyArrayBuffer::Values(_) => unreachable!("str arrays are byte-backed"),
    }
}
