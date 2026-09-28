//! Bounded integer draws by rejection sampling on the top bits.
//!
//! One method covers every range and dtype: mask a fresh 64-bit word down to the smallest
//! `2**k - 1` covering `[0, range_incl]`, and retry until the masked value falls inside. This is
//! exact (every value in range is equally likely) and simple; shellsim does not need NumPy's
//! split between Lemire's method and dtype-specific word widths, since streams need not match
//! NumPy's (see `random.py`'s module docstring).

use super::bitgen::Pcg64;

/// Smallest `2**k - 1 >= value`, i.e. the all-ones mask that covers `value`.
fn mask_covering(mut value: u64) -> u64 {
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    value
}

/// One draw uniform in `[0, range_incl]`.
pub(in crate::python) fn bounded_u64(bitgen: &mut Pcg64, range_incl: u64) -> u64 {
    if range_incl == 0 {
        return 0;
    }
    let mask = mask_covering(range_incl);
    loop {
        let value = bitgen.next_u64() & mask;
        if value <= range_incl {
            return value;
        }
    }
}

/// One draw uniform in `[low, high_incl]`. The caller has already checked `high_incl >= low`, so
/// the range width fits in a `u64` even when `low` and `high_incl` are both `i64`.
pub(in crate::python) fn draw_bounded(bitgen: &mut Pcg64, low: i64, high_incl: i64) -> i64 {
    let range_incl = high_incl.wrapping_sub(low) as u64;
    low.wrapping_add(bounded_u64(bitgen, range_incl) as i64)
}
