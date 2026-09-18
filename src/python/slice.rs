//! Normalized plans for Python sequence slices.
//!
//! Parsing produces opaque slice values; sequence implementations turn those values into this
//! small plan. Keeping normalization here gives builtin containers and NumPy the same signed
//! bounds and step behavior without giving the VM a second slicing protocol.

use std::ops::Range;

/// A normalized slice over a sequence with a known length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SlicePlan {
    first: isize,
    step: isize,
    length: usize,
}

impl SlicePlan {
    /// Normalize Python's optional signed slice components against `length`.
    pub(super) fn new(
        length: usize,
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    ) -> Result<Self, String> {
        isize::try_from(length).map_err(|_| "sequence is too large to slice")?;
        let length = i64::try_from(length).map_err(|_| "sequence is too large to slice")?;
        let step = step.unwrap_or(1);
        if step == 0 {
            return Err("slice step cannot be zero".into());
        }
        let normalize = |value: i64, minimum: i64, maximum: i64| {
            let value = if value < 0 {
                value.saturating_add(length)
            } else {
                value
            };
            value.clamp(minimum, maximum)
        };
        let (first, stop) = if step > 0 {
            (
                start.map_or(0, |value| normalize(value, 0, length)),
                stop.map_or(length, |value| normalize(value, 0, length)),
            )
        } else {
            (
                start.map_or(length - 1, |value| normalize(value, -1, length - 1)),
                stop.map_or(-1, |value| normalize(value, -1, length - 1)),
            )
        };
        let selected = if (step > 0 && first < stop) || (step < 0 && first > stop) {
            usize::try_from((stop - first - step.signum()) / step + 1)
                .map_err(|_| "slice length overflow")?
        } else {
            0
        };
        Ok(Self {
            first: isize::try_from(first).map_err(|_| "slice offset overflow")?,
            step: isize::try_from(step).map_err(|_| "slice step overflow")?,
            length: selected,
        })
    }

    pub(super) fn first(self) -> isize {
        self.first
    }

    pub(super) fn step(self) -> isize {
        self.step
    }

    pub(super) fn len(self) -> usize {
        self.length
    }

    /// Return the replaceable range for an ordinary forward slice.
    pub(super) fn contiguous_range(self) -> Option<Range<usize>> {
        if self.step != 1 {
            return None;
        }
        let start = usize::try_from(self.first).ok()?;
        start.checked_add(self.length).map(|end| start..end)
    }

    /// Visit selected indices in Python order without allocating an index vector.
    pub(super) fn indices(self) -> impl Iterator<Item = usize> {
        (0..self.length).map(move |offset| {
            let offset = isize::try_from(offset).expect("slice length fits isize");
            let index = self
                .step
                .checked_mul(offset)
                .and_then(|offset| self.first.checked_add(offset))
                .expect("normalized slice index cannot overflow");
            usize::try_from(index).expect("normalized slice index is non-negative")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SlicePlan;

    #[test]
    fn normalizes_forward_reverse_and_empty_slices() {
        assert_eq!(
            SlicePlan::new(6, Some(1), Some(5), Some(2))
                .unwrap()
                .indices()
                .collect::<Vec<_>>(),
            [1, 3]
        );
        assert_eq!(
            SlicePlan::new(6, None, None, Some(-2))
                .unwrap()
                .indices()
                .collect::<Vec<_>>(),
            [5, 3, 1]
        );
        assert_eq!(
            SlicePlan::new(3, Some(20), Some(30), None)
                .unwrap()
                .contiguous_range(),
            Some(3..3)
        );
    }

    #[test]
    fn rejects_zero_step() {
        assert_eq!(
            SlicePlan::new(3, None, None, Some(0)).unwrap_err(),
            "slice step cannot be zero"
        );
    }
}
