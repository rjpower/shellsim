//! Capability-free arithmetic helpers for the VM's small :mod:`functools` slice.

/// Fold a finite sequence through a caller-supplied operation.
///
/// The operation itself stays in the VM so Python's dynamic call and comparison rules remain in
/// one place.  This helper only gives the adapter a clear fold shape.
pub fn reduce_steps<T, F>(values: impl IntoIterator<Item = T>, mut accumulator: T, mut step: F) -> T
where
    F: FnMut(T, T) -> T,
{
    for value in values {
        accumulator = step(accumulator, value);
    }
    accumulator
}

#[cfg(test)]
mod tests {
    use super::reduce_steps;

    #[test]
    fn reduce_steps_folds_left_to_right() {
        assert_eq!(reduce_steps([2, 3, 4], 1, |a, b| a * b), 24);
    }
}
