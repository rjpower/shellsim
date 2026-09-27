//! Shuffling, permuting, and sampling without replacement.
//!
//! `shuffle` is the standard Fisher-Yates shuffle, walking the array from the last element to
//! the second and swapping each with a uniformly chosen earlier-or-equal element (Durstenfeld's
//! in-place variant of Fisher & Yates 1938). `permutation` is `shuffle` applied to a copy (or to
//! `arange(n)` for an integer argument). Both use masked rejection for the index draw on
//! `Generator` and legacy `RandomState` alike (see `integers::masked_bounded`'s doc) — unlike
//! `integers()`/`randint()`, which use Lemire on `Generator`. Sampling `size` distinct indices
//! out of `n` without replacement partially shuffles an identity array the same way and keeps
//! the last `size` slots (nearest the top) in the order they landed, which draws exactly `size`
//! bounded integers instead of a full permutation's `n`.

use super::bitgen::BitGen;
use super::integers::masked_bounded;

/// Fisher-Yates, in place.
pub(in crate::python) fn shuffle_indices(bitgen: &mut BitGen, values: &mut [i64]) {
    let n = values.len();
    for i in (1..n).rev() {
        let j = masked_bounded(bitgen, i as u128) as usize;
        values.swap(i, j);
    }
}

/// `size` distinct indices drawn from `[0, n)` without replacement, in the order NumPy's
/// partial shuffle produces: shuffle `arange(n)` from the top, stopping after `size` swaps, and
/// return the last `size` slots (nearest the top) in the order they landed.
pub(in crate::python) fn sample_without_replacement(
    bitgen: &mut BitGen,
    n: usize,
    size: usize,
) -> Vec<i64> {
    let mut pool: Vec<i64> = (0..n as i64).collect();
    let stop = n.saturating_sub(size);
    for i in (stop..n).rev() {
        let j = masked_bounded(bitgen, i as u128) as usize;
        pool.swap(i, j);
    }
    let mut result = pool[stop..].to_vec();
    result.reverse();
    result
}
