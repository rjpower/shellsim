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
//!
//! **`Generator.choice(..., replace=False, p=None)` is not implemented by this module.** NumPy's
//! own `Generator.choice` does not call `.permutation()` or this module's
//! `sample_without_replacement`: it draws a visibly different sequence even when the sample is
//! the whole population (`n == size`, so every element is "kept" either way and no slicing
//! ambiguity can explain a difference). For example, `Generator(3).permutation(5)` and
//! `Generator(3).choice(5, size=5, replace=False)` consume the *same* seed but diverge: the
//! former is `[4, 2, 1, 3, 0]` (this module's Fisher-Yates, verified bit-exact — see above), the
//! latter is `[4, 1, 2, 3, 0]` on real NumPy 2.5.3, and it consumes 4 raw 64-bit PCG64 words for
//! that call, not the 4 raw *32-bit* words (2 word64s) a range-adaptive masked or Lemire draw
//! over `[0, 5)` would use elsewhere in this crate — `choice`'s index draws are 64-bit width
//! regardless of how small the population is, unlike every other bounded draw this module or
//! `integers.rs` recovered. Both `test_generator_choice_without_replacement` and
//! `test_generator_choice_without_replacement_uses_floyd_and_tail_shuffle` (named after the
//! two algorithms NumPy's own public documentation says `Generator.choice` switches between
//! based on the sample-to-population ratio) were targeted with an extensive black-box search:
//! Floyd's algorithm (Bentley & Floyd 1987) over the classic sparse-array formulation, both as a
//! plain set-builder and as a full position-swap that also fixes an output order, ascending and
//! descending over the layer index, inclusive and exclusive range conventions, masked-rejection
//! and Lemire word draws at both 32- and 64-bit width, and reading the result in insertion order
//! or from the final sparse array in either direction — well over a hundred combinations in
//! total, checked against `Generator(3).choice(5, size=5, replace=False)`'s exact output using
//! the same raw-word replay technique validated elsewhere in this module. The closest candidate
//! (ascending Floyd with a full position swap, masked 64-bit draws, `range = j` inclusive, read
//! from the final sparse array ascending) reproduced `[1, 2, 3, 0, 4]`: the *same five values*
//! NumPy returns, in a fixed rotation of the correct order (right-rotate by one to get NumPy's
//! `[4, 1, 2, 3, 0]`), which recurred with the smaller `Generator(3).choice(10, size=5,
//! replace=False)` case too — evidence this module's understanding of *which* raw words feed
//! *which* comparisons is largely right, but not of the exact bookkeeping NumPy's C implementation
//! uses to place results in the final array. This matches a finding from an earlier pass on this
//! same problem (see `git log` on this file): a single swap step differing from NumPy's own with
//! no root cause found. Both `Generator.choice(..., replace=False)` test functions are left
//! failing rather than loosened or worked around; `RandomState.choice(..., replace=False)`
//! (legacy) is unaffected and already matches NumPy exactly (`test_legacy_choice_without_replacement`
//! passes) since it uses this module's `shuffle_indices`/`sample_without_replacement` directly, not
//! whatever `Generator.choice` does.

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
