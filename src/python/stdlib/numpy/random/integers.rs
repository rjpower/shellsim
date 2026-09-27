//! Bounded integer draws.
//!
//! `Generator.integers` uses Lemire's rejection method (Lemire 2019, "Fast Random Integer
//! Generation in an Interval"): multiply the raw word by the range, and reject only the rare
//! low-order values that would bias the result. Legacy `RandomState.randint` uses the older
//! masked-rejection method: mask the raw word down to the smallest `2**k - 1` covering the
//! range and reject draws that fall outside it. Both pick the raw word width (32 or 64 bits)
//! from the requested range, not the output dtype: a small range draws 32-bit words even when
//! stored as `int64`, which is what lets a `float64`-sized `randint` call spend only as much
//! entropy as the range needs. This matches NumPy's own per-width specialization, confirmed
//! against its published output for narrow, wide, and dtype-buffered ranges.

use super::bitgen::BitGen;

/// Smallest `2**k - 1 >= value`, i.e. the all-ones mask that covers `value`.
fn mask_covering(mut value: u64, bits: u32) -> u64 {
    if bits >= 64 {
        value |= value >> 32;
    }
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    if bits >= 64 {
        value |= value >> 32;
    }
    value
}

/// Masked rejection in `[0, range_incl]` using 32-bit words.
pub(in crate::python) fn masked_u32(bitgen: &mut BitGen, range_incl: u32) -> u32 {
    if range_incl == 0 {
        return 0;
    }
    let mask = mask_covering(u64::from(range_incl), 32) as u32;
    loop {
        let value = bitgen.next_u32() & mask;
        if value <= range_incl {
            return value;
        }
    }
}

/// Masked rejection in `[0, range_incl]` using 64-bit words.
pub(in crate::python) fn masked_u64(bitgen: &mut BitGen, range_incl: u64) -> u64 {
    if range_incl == 0 {
        return 0;
    }
    let mask = mask_covering(range_incl, 64);
    loop {
        let value = bitgen.next_u64() & mask;
        if value <= range_incl {
            return value;
        }
    }
}

/// Lemire's method in `[0, range_excl)` using 32-bit words. `range_excl` may be `2**32`
/// (the full word range), which needs no rejection at all.
pub(in crate::python) fn lemire_u32(bitgen: &mut BitGen, range_excl: u64) -> u32 {
    debug_assert!((1..=1u64 << 32).contains(&range_excl));
    if range_excl == 1u64 << 32 {
        return bitgen.next_u32();
    }
    let range_excl = range_excl as u32;
    loop {
        let word = bitgen.next_u32();
        let product = u64::from(word) * u64::from(range_excl);
        let low = product as u32;
        if low < range_excl {
            let threshold = ((1u64 << 32) - u64::from(range_excl)) % u64::from(range_excl);
            if u64::from(low) < threshold {
                continue;
            }
        }
        return (product >> 32) as u32;
    }
}

/// Lemire's method in `[0, range_excl)` using 64-bit words. `range_excl` may be `2**64`.
pub(in crate::python) fn lemire_u64(bitgen: &mut BitGen, range_excl: u128) -> u64 {
    debug_assert!((1..=1u128 << 64).contains(&range_excl));
    if range_excl == 1u128 << 64 {
        return bitgen.next_u64();
    }
    let range_excl = range_excl as u64;
    loop {
        let word = bitgen.next_u64();
        let product = u128::from(word) * u128::from(range_excl);
        let low = product as u64;
        if low < range_excl {
            let threshold = ((1u128 << 64) - u128::from(range_excl)) % u128::from(range_excl);
            if u128::from(low) < threshold {
                continue;
            }
        }
        return (product >> 64) as u64;
    }
}

/// One bounded draw in `[low, high_incl]`, returned as the low 64 bits of the result (the
/// caller truncates to the output dtype's width, which is safe because the value is already
/// known to fit). `count` is `high_incl - low + 1` computed by the caller in `i128` so it never
/// overflows; it is `0` only when the range spans the full 64-bit ring (`u64::MAX + 1`).
pub(in crate::python) fn draw_bounded(
    bitgen: &mut BitGen,
    low: i64,
    count: u128,
    legacy: bool,
) -> i64 {
    if count == 1 {
        return low;
    }
    let range_incl = count - 1;
    let offset: u128 = if range_incl <= u128::from(u32::MAX) {
        u128::from(if legacy {
            masked_u32(bitgen, range_incl as u32)
        } else {
            lemire_u32(bitgen, count as u64)
        })
    } else {
        u128::from(if legacy {
            masked_u64(bitgen, range_incl as u64)
        } else {
            lemire_u64(bitgen, count)
        })
    };
    low.wrapping_add(offset as i64)
}
