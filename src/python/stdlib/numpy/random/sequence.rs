//! Shuffling and choosing without replacement.
//!
//! `shuffle_indices` is Durstenfeld's in-place Fisher-Yates shuffle. `choice_without_replacement`
//! is Floyd's algorithm (Bentley and Floyd, "Programming Pearls", 1987): for `j` from `n - size`
//! to `n - 1`, draw `t` in `[0, j]` and take it, or take `j` itself if `t` was already picked.
//! This selects `size` distinct indices from `[0, n)` in `O(size)` time and space regardless of
//! `n`, so shellsim does not need NumPy's separate large-population algorithm. A final
//! Fisher-Yates pass over the selection (`shuffle=True`, NumPy's default) mixes the order, since
//! Floyd's algorithm alone favors later indices toward the end.

use super::bitgen::Pcg64;
use super::integers::bounded_u64;
use std::collections::HashSet;

/// Fisher-Yates, in place.
pub(in crate::python) fn shuffle_indices(bitgen: &mut Pcg64, values: &mut [i64]) {
    for i in (1..values.len()).rev() {
        let j = bounded_u64(bitgen, i as u64) as usize;
        values.swap(i, j);
    }
}

/// `Generator.choice(a, size, replace=False, shuffle=...)`'s index draws.
pub(in crate::python) fn choice_without_replacement(
    bitgen: &mut Pcg64,
    n: usize,
    size: usize,
    shuffle: bool,
) -> Vec<i64> {
    let mut seen: HashSet<i64> = HashSet::with_capacity(size);
    let mut order = Vec::with_capacity(size);
    for j in (n - size)..n {
        let t = bounded_u64(bitgen, j as u64) as i64;
        let picked = if seen.contains(&t) { j as i64 } else { t };
        seen.insert(picked);
        order.push(picked);
    }
    if shuffle {
        for i in (1..order.len()).rev() {
            let j = bounded_u64(bitgen, i as u64) as usize;
            order.swap(i, j);
        }
    }
    order
}
