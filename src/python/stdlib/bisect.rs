//! Capability-free binary-search helpers from Python's :mod:`bisect` module.
//!
//! These functions intentionally use Rust's `Ord` abstraction rather than the VM's dynamic
//! values.  The eventual adapter can implement Python's cross-type comparison rules and map its
//! `TypeError` to a failed comparison before calling these simple index algorithms.

use std::cmp::Ordering;

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyList, PyResult, PyRuntime, PySequence, PyValueCast,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "bisect",
    functions: &[
        FunctionDef {
            module: "bisect",
            name: "bisect_left",
            call: native_bisect_left,
        },
        FunctionDef {
            module: "bisect",
            name: "bisect_right",
            call: native_bisect_right,
        },
        FunctionDef {
            module: "bisect",
            name: "bisect",
            call: native_bisect_right,
        },
        FunctionDef {
            module: "bisect",
            name: "insort_left",
            call: native_insort_left,
        },
        FunctionDef {
            module: "bisect",
            name: "insort_right",
            call: native_insort_right,
        },
        FunctionDef {
            module: "bisect",
            name: "insort",
            call: native_insort_right,
        },
    ],
    values: &[],
};

fn native_bisect_left(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_bisect(runtime, args, false)
}

fn native_bisect_right(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_bisect(runtime, args, true)
}

fn native_bisect(runtime: &mut dyn PyRuntime, args: CallArgs, right: bool) -> PyResult {
    args.expect_positional("bisect", 2, 4)?;
    args.reject_keywords("bisect")?;
    let values = args.positional()[0]
        .cast::<PySequence>(runtime)?
        .items(runtime)?;
    let needle = &args.positional()[1];
    let (mut low, mut high) = bisect_bounds(runtime, &args, values.len())?;
    while low < high {
        runtime.charge_cpu(1)?;
        let middle = low + (high - low) / 2;
        let ordering = runtime.compare(&values[middle], needle)?;
        if ordering == Ordering::Less || (right && ordering == Ordering::Equal) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    i64::try_from(low)
        .map(Value::Int)
        .map_err(|_| PyError::overflow_error("bisect result exceeds bounded integer range"))
}

fn bisect_bounds(
    runtime: &dyn PyRuntime,
    args: &CallArgs,
    length: usize,
) -> PyResult<(usize, usize)> {
    let low = args
        .positional()
        .get(2)
        .map(|value| {
            runtime
                .int_value(value)
                .ok_or_else(|| PyError::type_error("lo must be an integer"))
                .and_then(|value| {
                    usize::try_from(value)
                        .map_err(|_| PyError::value_error("lo must be non-negative"))
                })
        })
        .transpose()?
        .unwrap_or(0);
    let high = args
        .positional()
        .get(3)
        .map(|value| {
            runtime
                .int_value(value)
                .ok_or_else(|| PyError::type_error("hi must be an integer"))
                .and_then(|value| {
                    usize::try_from(value)
                        .map_err(|_| PyError::value_error("hi must be non-negative"))
                })
        })
        .transpose()?
        .unwrap_or(length);
    if high > length {
        return Err(PyError::value_error("hi exceeds sequence length"));
    }
    Ok((low, high))
}

fn native_insort_left(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_insort(runtime, args, false)
}

fn native_insort_right(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    native_insort(runtime, args, true)
}

fn native_insort(runtime: &mut dyn PyRuntime, args: CallArgs, right: bool) -> PyResult {
    let list = args
        .positional()
        .first()
        .copied()
        .ok_or_else(|| PyError::type_error("insort expected a list and a value"))?
        .cast::<PyList>(runtime)?;
    let index_value = native_bisect(runtime, args.clone(), right)?;
    let index = runtime
        .int_value(&index_value)
        .and_then(|index| usize::try_from(index).ok())
        .ok_or_else(|| PyError::runtime_error("bisect returned an invalid index"))?;
    runtime.list_insert(list, index, args.positional()[1])?;
    Ok(Value::None)
}

/// Errors for an explicitly bounded search.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BisectError {
    InvalidBounds { lo: usize, hi: usize, length: usize },
}

/// Return the first index at which `item` may be inserted while preserving sorted order.
#[cfg(test)]
pub fn bisect_left<T: Ord>(items: &[T], item: &T) -> usize {
    bisect_left_range(items, item, 0, items.len()).expect("full slice bounds are valid")
}

/// Return the insertion point after existing equal values.
#[cfg(test)]
pub fn bisect_right<T: Ord>(items: &[T], item: &T) -> usize {
    bisect_right_range(items, item, 0, items.len()).expect("full slice bounds are valid")
}

/// Bounded equivalent of Python's `bisect_left(..., lo, hi)` for non-negative bounds.
#[cfg(test)]
pub fn bisect_left_range<T: Ord>(
    items: &[T],
    item: &T,
    lo: usize,
    hi: usize,
) -> Result<usize, BisectError> {
    validate_bounds(items.len(), lo, hi)?;
    let (mut low, mut high) = (lo, hi);
    while low < high {
        let middle = low + (high - low) / 2;
        if items[middle] < *item {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    Ok(low)
}

/// Bounded equivalent of Python's `bisect_right(..., lo, hi)` for non-negative bounds.
#[cfg(test)]
pub fn bisect_right_range<T: Ord>(
    items: &[T],
    item: &T,
    lo: usize,
    hi: usize,
) -> Result<usize, BisectError> {
    validate_bounds(items.len(), lo, hi)?;
    let (mut low, mut high) = (lo, hi);
    while low < high {
        let middle = low + (high - low) / 2;
        if *item < items[middle] {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    Ok(low)
}

/// Insert before equal values.
#[cfg(test)]
pub fn insort_left<T: Ord>(items: &mut Vec<T>, item: T) {
    let index = bisect_left(items, &item);
    items.insert(index, item);
}

/// Insert after equal values.
#[cfg(test)]
pub fn insort_right<T: Ord>(items: &mut Vec<T>, item: T) {
    let index = bisect_right(items, &item);
    items.insert(index, item);
}

#[cfg(test)]
fn validate_bounds(length: usize, lo: usize, hi: usize) -> Result<(), BisectError> {
    // CPython permits an empty interval when lo >= hi and simply returns lo. It does reject a
    // high bound that would cause an indexed read; negative bounds are represented and rejected
    // by the dynamic VM adapter before reaching this usize-only core.
    if hi > length {
        return Err(BisectError::InvalidBounds { lo, hi, length });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        bisect_left, bisect_left_range, bisect_right, bisect_right_range, insort_left,
        insort_right, BisectError,
    };

    #[test]
    fn duplicate_boundaries_match_bisect_contract() {
        let values = [1, 2, 2, 2, 5];
        assert_eq!(bisect_left(&values, &2), 1);
        assert_eq!(bisect_right(&values, &2), 4);
        assert_eq!(bisect_left(&values, &0), 0);
        assert_eq!(bisect_right(&values, &9), values.len());
    }

    #[test]
    fn bounded_search_and_invalid_bounds_are_explicit() {
        let values = [1, 2, 3, 4, 5];
        assert_eq!(bisect_left_range(&values, &1, 2, 5), Ok(2));
        assert_eq!(bisect_right_range(&values, &4, 0, 3), Ok(3));
        assert_eq!(bisect_left_range(&values, &2, 4, 2), Ok(4));
        assert_eq!(
            bisect_right_range(&values, &2, 0, 6),
            Err(BisectError::InvalidBounds {
                lo: 0,
                hi: 6,
                length: 5
            })
        );
    }

    #[test]
    fn insertion_supports_numbers_and_strings() {
        let mut numbers = vec![1, 2, 2, 5];
        insort_left(&mut numbers, 2);
        assert_eq!(numbers, [1, 2, 2, 2, 5]);
        let mut strings = vec!["a".to_string(), "c".to_string()];
        insort_right(&mut strings, "b".to_string());
        assert_eq!(strings, ["a", "b", "c"]);
    }
}
