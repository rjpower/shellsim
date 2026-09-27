//! Shuffling, permuting, and sampling without replacement.
//!
//! `shuffle` is Durstenfeld's in-place Fisher-Yates shuffle, walking from the last element down
//! and swapping each with a uniformly chosen element at or below it. `permutation` shuffles a
//! copy, or `arange(n)` for an integer. Sampling `size` distinct indices from `n` runs the same
//! walk on an identity array for only `size` steps and keeps the top `size` slots. These index
//! draws use masked rejection on both `Generator` and legacy `RandomState`.
//!
//! `Generator.choice(..., replace=False)` instead draws with Lemire's method and has two paths:
//!
//! - Floyd's algorithm (Bentley and Floyd, "Programming Pearls", 1987) selects the indices: for
//!   `j` from `n - size` to `n - 1`, draw `t` in `[0, j]` and take `t`, or `j` if `t` was already
//!   taken. With `shuffle=True` a Fisher-Yates pass over the selection follows.
//! - For a population above 10,000 and a sample above 2% of it, `shuffle=True` instead runs a
//!   partial Fisher-Yates over `arange(n)` and keeps the top `size` slots, already shuffled.
//!
//! The draw methods, the result order, and the switch between the two `choice` paths match
//! NumPy's, so seeded results agree exactly.

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
/// algorithm.
const LARGE_POPULATION_THRESHOLD: usize = 10_000;

/// Sample-to-population ratio (as its reciprocal, so the comparison stays in integers) above
/// which a large population switches from Floyd-then-shuffle to the single-pass partial
/// Fisher-Yates. A `size` at most `n / LARGE_POPULATION_SAMPLE_RATIO` still uses Floyd.
const LARGE_POPULATION_SAMPLE_RATIO: usize = 50;

/// `Generator.choice(a, size, replace=False, shuffle=...)`'s index draws, by the two paths the
/// module doc describes.
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

/// Floyd's algorithm: `size` distinct indices from `[0, n)`, in the order they are decided.
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

/// Fisher-Yates over an already selected array, with Lemire draws.
fn fisher_yates_lemire(bitgen: &mut BitGen, mut values: Vec<i64>) -> Vec<i64> {
    for i in (1..values.len()).rev() {
        let j = lemire_bounded(bitgen, (i + 1) as u128) as usize;
        values.swap(i, j);
    }
    values
}

/// The large-population `shuffle=True` path: a partial Fisher-Yates over `arange(n)` with Lemire
/// draws, keeping the top `size` slots in index order.
fn partial_fisher_yates_forward(bitgen: &mut BitGen, n: usize, size: usize) -> Vec<i64> {
    let mut pool: Vec<i64> = (0..n as i64).collect();
    let stop = n - size;
    for i in (stop..n).rev() {
        let j = lemire_bounded(bitgen, (i + 1) as u128) as usize;
        pool.swap(i, j);
    }
    pool[stop..].to_vec()
}
