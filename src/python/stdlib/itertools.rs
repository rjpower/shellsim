//! Small, capability-free cores for the finite portions of :mod:`itertools`.
//!
//! The VM adapter is responsible for representing Python's lazy iterators and for charging
//! resource usage.  These helpers only describe the arithmetic and finite slicing contract.

/// Return the next value in a count sequence, checking the bounded integer domain.
pub fn count_next(current: i64, step: i64) -> Result<i64, &'static str> {
    current
        .checked_add(step)
        .ok_or("itertools.count exceeded the bounded integer range")
}

/// Produce a finite slice from an arithmetic count sequence.
pub fn count_slice(start: i64, step: i64, length: usize) -> Result<Vec<i64>, &'static str> {
    let mut values = Vec::with_capacity(length);
    let mut current = start;
    for _ in 0..length {
        values.push(current);
        current = count_next(current, step)?;
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{count_next, count_slice};

    #[test]
    fn count_is_bounded_and_slice_is_finite() {
        assert_eq!(count_next(4, 2), Ok(6));
        assert_eq!(count_slice(3, 1, 3), Ok(vec![3, 4, 5]));
        assert!(count_next(i64::MAX, 1).is_err());
    }
}
