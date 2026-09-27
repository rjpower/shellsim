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
//! `Generator.choice(..., replace=False, p=None)` uses a different pair of algorithms from the
//! rest of this module. The key that unlocked it was splitting `Generator.choice`'s own
//! `shuffle` parameter: `shuffle=False` isolates the selection step from the final reordering,
//! which a single combined black-box search (see `git log` on this file for that earlier,
//! unsuccessful pass) could not tell apart.
//!
//! - **Selection**, used for `shuffle=False` unconditionally and as the first stage for
//!   `shuffle=True` outside the large-population case below: Floyd's algorithm (Floyd 1979,
//!   popularized in Bentley & Floyd's "Programming Pearls"). Walk `j` ascending from `n - size`
//!   to `n - 1`, draw `t` uniformly from `[0, j]` with *Lemire's* method — not the masked
//!   rejection `shuffle`/`permutation` use above; replaying `Generator(3).choice(10, 3,
//!   replace=False, shuffle=False)`'s three raw words through both showed only Lemire's 32-bit
//!   form reproduces NumPy's `[6, 0, 1]` — and append `t` unless it was already appended, in
//!   which case append `j` instead. This is the textbook trick that makes an `O(size)` hash set
//!   stand in for an `O(n)` array with no retries ever needed on a collision. Verified bit for
//!   bit, including the *post-call* bit generator state and not just the returned values, against
//!   dozens of `(seed, n, size)` combinations, including the degenerate `size == n` case (whose
//!   result is always `arange(n)` regardless of the random draws, by induction, but which still
//!   *consumes* `size - 1` words rather than `size`: the final index's range is `[0, 0]`, and
//!   Lemire's method for a one-value range returns `0` without drawing at all, mirroring
//!   `integers.rs`'s `draw_bounded`'s `count == 1` shortcut — confirmed by replaying
//!   `Generator(3).choice(5, 5, replace=False, shuffle=False)` and finding the predicted and real
//!   post-call PCG64 state identical only once this shortcut was added).
//! - **Reordering**, `shuffle=True` only (`shuffle=False` returns the selection as-is): for a
//!   small population, or a sample that is a small fraction of a large one (the threshold below),
//!   a plain Fisher-Yates shuffle of the `size`-element selection, also via Lemire (`i`
//!   descending from `size - 1` to `1`, `j` from `[0, i]`). For a large population *and* a sample
//!   that isn't tiny relative to it, NumPy instead builds the whole answer in one pass: a partial
//!   Fisher-Yates over `arange(n)` itself (`i` descending from `n - 1` to `n - size`, same Lemire
//!   draw), keeping the last `size` slots in the order they land — not reversed, unlike
//!   `sample_without_replacement` above. That single pass is already shuffled, so no separate
//!   reorder step follows it. Black-box bisection against NumPy 2.5.3 pinned the switch to
//!   exactly `n > 10_000 && size > n / 50` (population over ten thousand *and* sample over two
//!   percent of it): every `(n, size)` tried on either side of both thresholds, across
//!   populations from 1,000 to 100,000, matched one algorithm or the other with no exceptions,
//!   and the size threshold's switch point was exact to the integer (`n / 50` still matches the
//!   small-sample algorithm, `n / 50 + 1` already needs the large-population one) for four
//!   different populations spanning a 5x range.
//!
//! `RandomState.choice(..., replace=False)` (legacy) does not go through any of this: it calls
//! `shuffle_indices`/`sample_without_replacement` directly (`test_legacy_choice_without_replacement`
//! passes) and has no `shuffle` parameter of its own.

use super::bitgen::BitGen;
use super::integers::{lemire_bounded, masked_bounded};
use std::collections::HashSet;

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

/// Population size above which `choice_without_replacement` can use the large-population
/// algorithm; see the module doc for how this and [`LARGE_POPULATION_SAMPLE_RATIO`] were
/// recovered.
const LARGE_POPULATION_THRESHOLD: usize = 10_000;

/// Sample-to-population ratio (as its reciprocal, so the comparison stays in integers) above
/// which a large population switches from Floyd-then-shuffle to the single-pass partial
/// Fisher-Yates. A `size` at most `n / LARGE_POPULATION_SAMPLE_RATIO` still uses Floyd.
const LARGE_POPULATION_SAMPLE_RATIO: usize = 50;

/// `Generator.choice(a, size, replace=False, shuffle=...)`'s index draws. See the module doc for
/// the two algorithms this switches between and the threshold that picks one.
pub(in crate::python) fn choice_without_replacement(
    bitgen: &mut BitGen,
    n: usize,
    size: usize,
    shuffle: bool,
) -> Vec<i64> {
    if shuffle && n > LARGE_POPULATION_THRESHOLD && size > n / LARGE_POPULATION_SAMPLE_RATIO {
        return partial_fisher_yates_forward(bitgen, n, size);
    }
    let selected = floyd_select(bitgen, n, size);
    if shuffle {
        fisher_yates_lemire(bitgen, selected)
    } else {
        selected
    }
}

/// Floyd's algorithm: `size` distinct indices from `[0, n)`, appended in the order they are
/// decided rather than sorted or shuffled. See the module doc for the derivation.
fn floyd_select(bitgen: &mut BitGen, n: usize, size: usize) -> Vec<i64> {
    let mut seen: HashSet<i64> = HashSet::with_capacity(size);
    let mut order = Vec::with_capacity(size);
    for j in (n - size)..n {
        let t = lemire_bounded(bitgen, (j + 1) as u128) as i64;
        let picked = if seen.contains(&t) { j as i64 } else { t };
        seen.insert(picked);
        order.push(picked);
    }
    order
}

/// Fisher-Yates over an already-selected `size`-element array, via Lemire draws (not the masked
/// rejection `shuffle_indices` uses — see the module doc).
fn fisher_yates_lemire(bitgen: &mut BitGen, mut values: Vec<i64>) -> Vec<i64> {
    for i in (1..values.len()).rev() {
        let j = lemire_bounded(bitgen, (i + 1) as u128) as usize;
        values.swap(i, j);
    }
    values
}

/// The large-population `shuffle=True` algorithm: a partial Fisher-Yates over `arange(n)` via
/// Lemire draws, keeping the last `size` slots in the order they land (not reversed). Already
/// shuffled by construction, so no separate reorder step follows it.
fn partial_fisher_yates_forward(bitgen: &mut BitGen, n: usize, size: usize) -> Vec<i64> {
    let mut pool: Vec<i64> = (0..n as i64).collect();
    let stop = n - size;
    for i in (stop..n).rev() {
        let j = lemire_bounded(bitgen, (i + 1) as u128) as usize;
        pool.swap(i, j);
    }
    pool[stop..].to_vec()
}
