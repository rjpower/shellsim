//! `np.searchsorted` and `ndarray.searchsorted`, following `PyArray_SearchSorted`.
//!
//! `a` and `v` convert to one dtype, found as array construction finds it
//! (`PyArray_DescrFromObject`): Python scalars take their default dtypes rather than NEP 50's
//! weak ones, and strings absorb numbers. Numeric kinds use NumPy 2.5's branchless search, and strings and
//! objects use the generic search that starts from the previous key's bounds. Both are ported
//! step for step, so results match NumPy's even when `a` is not sorted. A `sorter` index is
//! checked only when the search reads it, as in NumPy.

use std::cmp::Ordering;

use super::super::super::super::native::{CallArgs, PyError, PyResult, PyRuntime, PyValue};
use super::super::args::{Bound, Signature};
use super::super::array::{self, Array};
use super::super::convert;
use super::super::dtype::{Category, DType};
use super::Keys;

#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}

/// Parse `side=` as `PyArray_SearchsideConverter` does.
fn side(runtime: &mut dyn PyRuntime, value: Option<PyValue>) -> PyResult<Side> {
    let Some(value) = value else {
        return Ok(Side::Left);
    };
    let Some(text) = runtime.string_value(&value)? else {
        return Err(PyError::type_error(format!(
            "search side must be str, not {}",
            runtime.type_name(&value)?
        )));
    };
    let (side, exact) = match text.chars().next() {
        Some('l' | 'L') => (Side::Left, text == "left"),
        Some('r' | 'R') => (Side::Right, text == "right"),
        _ => {
            let repr = runtime.repr(&value)?;
            return Err(PyError::value_error(format!(
                "search side must be 'left' or 'right' (got {repr})"
            )));
        }
    };
    if !exact {
        return Err(PyError::value_error(
            "search side must be one of 'left' or 'right'",
        ));
    }
    Ok(side)
}

/// The haystack of one search: `a`'s keys and, optionally, the permutation that sorts it.
struct Haystack<'a> {
    keys: &'a Keys,
    length: usize,
    sorter: Option<&'a [i64]>,
    side: Side,
}

impl Haystack<'_> {
    /// The element of `a` at sorted position `position`.
    fn element(&self, position: usize) -> PyResult<usize> {
        let Some(sorter) = self.sorter else {
            return Ok(position);
        };
        usize::try_from(sorter[position])
            .ok()
            .filter(|index| *index < self.length)
            .ok_or_else(|| PyError::value_error("Sorter index out of range."))
    }

    /// Whether `a[element]` goes before key `key`: `<` for the left side, `<=` for the right.
    fn before(&self, runtime: &mut dyn PyRuntime, element: usize, key: usize) -> PyResult<bool> {
        let ordering = if self.keys.is_generic() {
            self.keys.compare(runtime, element, key)?
        } else {
            self.keys.order(element, key, false)
        };
        Ok(match self.side {
            Side::Left => ordering == Ordering::Less,
            Side::Right => ordering != Ordering::Greater,
        })
    }

    /// NumPy's numeric `binsearch`/`argbinsearch` for one key. Each step halves the candidate
    /// interval `[base, base + length]` by testing its midpoint.
    fn branchless(&self, runtime: &mut dyn PyRuntime, key: usize) -> PyResult<usize> {
        let mut length = self.length;
        let half = length >> 1;
        length -= half;
        let mut base = if self.before(runtime, self.element(half)?, key)? {
            half
        } else {
            0
        };
        while length > 1 {
            let half = length >> 1;
            length -= half;
            if self.before(runtime, self.element(base + half)?, key)? {
                base += half;
            }
        }
        Ok(base + usize::from(self.before(runtime, self.element(base)?, key)?))
    }

    /// NumPy's generic `npy_binsearch`/`npy_argbinsearch` over `keys`, which keeps the
    /// previous key's bounds when the keys ascend.
    fn generic(
        &self,
        runtime: &mut dyn PyRuntime,
        keys: impl Iterator<Item = usize>,
        output: &mut Vec<i64>,
    ) -> PyResult<()> {
        let (mut low, mut high) = (0, self.length);
        let mut last = None;
        for key in keys {
            let ascending = match last {
                Some(last) => {
                    let ordering = self.keys.compare(runtime, last, key)?;
                    match self.side {
                        Side::Left => ordering == Ordering::Less,
                        Side::Right => ordering != Ordering::Greater,
                    }
                }
                // The first key is compared with itself.
                None => matches!(self.side, Side::Right),
            };
            if ascending {
                high = self.length;
            } else {
                low = 0;
                high = (high + 1).min(self.length);
            }
            last = Some(key);
            while low < high {
                let middle = low + ((high - low) >> 1);
                if self.before(runtime, self.element(middle)?, key)? {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            output.push(low as i64);
        }
        Ok(())
    }
}

/// The permutation argument, checked and converted as `PyArray_SearchSorted` does.
fn sorter(
    runtime: &mut dyn PyRuntime,
    value: Option<PyValue>,
    length: usize,
) -> PyResult<Option<Vec<i64>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let sorter = convert::as_array(runtime, value)
        .ok()
        .filter(|sorter| sorter.ndim() == 1)
        .ok_or_else(|| PyError::type_error("could not parse sorter argument"))?;
    if !matches!(
        sorter.dtype.category(),
        Category::Signed | Category::Unsigned
    ) {
        return Err(PyError::type_error("sorter must only contain integers"));
    }
    if sorter.size() != length {
        return Err(PyError::value_error("sorter.size must equal a.size"));
    }
    let sorter = convert::cast_array(runtime, &sorter, DType::INT64, false)?;
    Ok(Some(array::read_elements::<i64>(runtime, &sorter)?))
}

fn searchsorted(runtime: &mut dyn PyRuntime, a: Array, bound: &Bound) -> PyResult {
    let side = side(runtime, bound.get("side"))?;
    let needles = convert::as_array(runtime, bound.required("v"))?;
    let dtype = convert::infer_promote(a.dtype, needles.dtype)?;
    let needles = convert::cast_array(runtime, &needles, dtype, false)?;
    match a.ndim() {
        0 => {
            return Err(PyError::value_error(
                "object of too small depth for desired array",
            ))
        }
        1 => {}
        _ => return Err(PyError::value_error("object too deep for desired array")),
    }
    let a = convert::cast_array(runtime, &a, dtype, false)?;
    let length = a.size();
    let sorter = sorter(runtime, bound.value("sorter"), length)?;
    let keys = Keys::read(runtime, &[&a, &needles])?;
    let count = needles.size();
    let depth = u64::from(usize::BITS - length.leading_zeros()) + 2;
    runtime.charge_cpu((count as u64).saturating_mul(depth))?;
    runtime.reserve_memory(count.saturating_mul(8))?;
    let haystack = Haystack {
        keys: &keys,
        length,
        sorter: sorter.as_deref(),
        side,
    };
    let mut positions = Vec::with_capacity(count);
    if length == 0 {
        positions.resize(count, 0);
    } else if keys.is_generic() {
        haystack.generic(runtime, length..length + count, &mut positions)?;
    } else {
        // NumPy reads the first pivot, and checks its sorter index, before any key.
        haystack.element(length >> 1)?;
        for key in length..length + count {
            positions.push(haystack.branchless(runtime, key)? as i64);
        }
    }
    let result =
        array::array_from_elements(runtime, DType::INT64, needles.shape().to_vec(), &positions)?;
    if result.ndim() == 0 {
        return convert::element_to_scalar(runtime, &result, result.view.offset);
    }
    Ok(result.value())
}

/// `np.searchsorted(a, v, side='left', sorter=None)`.
pub(super) fn module_searchsorted(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("searchsorted", &["a", "v", "side", "sorter"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    searchsorted(runtime, a, &bound)
}

/// `ndarray.searchsorted(v, side='left', sorter=None)`.
pub(super) fn method_searchsorted(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
) -> PyResult {
    static SIGNATURE: Signature = Signature::new("searchsorted", &["v", "side", "sorter"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = Array::from_value(runtime, receiver)?;
    searchsorted(runtime, a, &bound)
}
