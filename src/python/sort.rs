//! Stable sorting with a fallible, guest-defined order.
//!
//! `sorted()` and `list.sort()` compare items through the object protocol, which can run guest
//! code and raise. A bottom-up merge sort needs O(n log n) comparisons, so sorting a large list
//! fits the CPU budget as it does in CPython, and stops at the first failing comparison.

/// Sort `items` stably, where `before(a, b)` reports whether `a` must precede `b`. Equal items
/// keep their order because an item from the right run moves ahead only when it is strictly
/// before the left one. On error `items` is left empty; callers discard it.
///
/// ```ignore
/// let mut items = vec![(2, 'a'), (1, 'b'), (2, 'c')];
/// merge_sort(&mut items, |a, b| Ok::<_, ()>(a.0 < b.0)).unwrap();
/// assert_eq!(items, [(1, 'b'), (2, 'a'), (2, 'c')]);
/// ```
pub(super) fn merge_sort<T: Copy, E>(
    items: &mut Vec<T>,
    mut before: impl FnMut(&T, &T) -> Result<bool, E>,
) -> Result<(), E> {
    let length = items.len();
    let mut source = std::mem::take(items);
    let mut target = Vec::with_capacity(length);
    let mut width = 1;
    while width < length {
        target.clear();
        for start in (0..length).step_by(width.saturating_mul(2)) {
            let middle = start.saturating_add(width).min(length);
            let end = middle.saturating_add(width).min(length);
            let (mut left, mut right) = (start, middle);
            while left < middle && right < end {
                if before(&source[right], &source[left])? {
                    target.push(source[right]);
                    right += 1;
                } else {
                    target.push(source[left]);
                    left += 1;
                }
            }
            target.extend_from_slice(&source[left..middle]);
            target.extend_from_slice(&source[right..end]);
        }
        std::mem::swap(&mut source, &mut target);
        width = width.saturating_mul(2);
    }
    *items = source;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_stably_with_n_log_n_comparisons() {
        let mut items = (0..1000)
            .map(|index| ((index * 7919) % 10, index))
            .collect::<Vec<_>>();
        let mut comparisons = 0;
        merge_sort(&mut items, |a, b| {
            comparisons += 1;
            Ok::<_, ()>(a.0 < b.0)
        })
        .unwrap();
        let mut expected = (0..1000)
            .map(|index| ((index * 7919) % 10, index))
            .collect::<Vec<_>>();
        expected.sort_by_key(|item| item.0);
        assert_eq!(items, expected);
        assert!(comparisons <= 1000 * 10, "{comparisons} comparisons");
    }

    #[test]
    fn a_failed_comparison_stops_the_sort() {
        let mut items = vec![3, 1, 2];
        let mut calls = 0;
        let result = merge_sort(&mut items, |_, _| {
            calls += 1;
            Err("boom")
        });
        assert_eq!(result, Err("boom"));
        assert_eq!(calls, 1);
    }
}
