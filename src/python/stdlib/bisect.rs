//! Capability-free binary-search helpers from Python's :mod:`bisect` module.
//!
//! These functions intentionally use Rust's `Ord` abstraction rather than the VM's dynamic
//! values.  The eventual adapter can implement Python's cross-type comparison rules and map its
//! `TypeError` to a failed comparison before calling these simple index algorithms.

use std::cmp::Ordering;

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyResult, PyRuntime, PySequence, PyValueCast,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "bisect",
    functions: &[FunctionDef {
        module: "bisect",
        name: "bisect_left",
        call: native_bisect_left,
    }],
    values: &[],
};

fn native_bisect_left(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("bisect_left", 2, 2)?;
    args.reject_keywords("bisect_left")?;
    let values = args.positional()[0]
        .cast::<PySequence>(runtime)?
        .items(runtime)?;
    let needle = &args.positional()[1];
    let mut low = 0usize;
    let mut high = values.len();
    while low < high {
        runtime.charge_cpu(1)?;
        let middle = low + (high - low) / 2;
        if runtime.compare(&values[middle], needle)? == Ordering::Less {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    i64::try_from(low)
        .map(Value::Int)
        .map_err(|_| PyError::overflow_error("bisect result exceeds bounded integer range"))
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
