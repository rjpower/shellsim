//! Bounded integer draws.
//!
//! `Generator.integers` uses Lemire's rejection method (Lemire 2019, "Fast Random Integer
//! Generation in an Interval"): multiply the raw word by the range, and reject only the rare
//! low-order values that would bias the result. Legacy `RandomState.randint` uses the older
//! masked-rejection method: mask the raw word down to the smallest `2**k - 1` covering the
//! range and reject draws that fall outside it. Both pick the raw word width (32 or 64 bits)
//! from the requested range, not the output dtype, so a small range draws 32-bit words even when
//! stored as `int64`. Dtypes narrower than 32 bits pack several draws into each 32-bit word.
//! The word widths and packing match NumPy's, so seeded draws agree exactly.

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

/// Masked rejection in `[0, range_incl]`, picking the 32- or 64-bit primitive from the range.
/// Shuffles and permutations use this on both `Generator` and legacy `RandomState`, although
/// `Generator.integers` uses Lemire's method.
pub(in crate::python) fn masked_bounded(bitgen: &mut BitGen, range_incl: u128) -> i64 {
    if range_incl <= u128::from(u32::MAX) {
        masked_u32(bitgen, range_incl as u32) as i64
    } else {
        masked_u64(bitgen, range_incl as u64) as i64
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

/// Lemire's method in `[0, range_excl)`, picking the 32- or 64-bit primitive from the range. A
/// single possible value returns `0` without drawing, like `draw_bounded`'s `count == 1` case.
/// `Generator.choice(..., replace=False)` draws its indices with this.
pub(in crate::python) fn lemire_bounded(bitgen: &mut BitGen, range_excl: u128) -> u64 {
    if range_excl <= 1 {
        return 0;
    }
    if range_excl <= 1u128 << 32 {
        u64::from(lemire_u32(bitgen, range_excl as u64))
    } else {
        lemire_u64(bitgen, range_excl)
    }
}

/// Unused bits of a `next_u32()` word, shared across one array fill so that a dtype narrower
/// than 32 bits takes several draws from each word, low bits first: two per word for 16-bit
/// dtypes, four for 8-bit dtypes, and 32 for `bool`. Each fill starts with an empty buffer.
pub(in crate::python) struct NarrowBuffer {
    word: u32,
    remaining: u32,
}

impl NarrowBuffer {
    pub(in crate::python) fn new() -> Self {
        NarrowBuffer {
            word: 0,
            remaining: 0,
        }
    }

    /// The next `chunk_bits`-wide chunk (low bits of the buffer first), refilling from a fresh
    /// `next_u32()` word whenever fewer than `chunk_bits` remain.
    fn next_chunk(&mut self, bitgen: &mut BitGen, chunk_bits: u32) -> u32 {
        if self.remaining < chunk_bits {
            self.word = bitgen.next_u32();
            self.remaining = 32;
        }
        let chunk = self.word & ((1u32 << chunk_bits) - 1);
        self.word >>= chunk_bits;
        self.remaining -= chunk_bits;
        chunk
    }
}

/// One bounded draw in `[0, range_incl]` from `chunk_bits`-wide pieces of `buffer`: `masked_u32`
/// or `lemire_u32` with a narrower word. The output dtype, and so `chunk_bits`, always covers
/// `range_incl`.
pub(in crate::python) fn draw_bounded_buffered(
    bitgen: &mut BitGen,
    buffer: &mut NarrowBuffer,
    range_incl: u32,
    legacy: bool,
    chunk_bits: u32,
) -> u32 {
    if range_incl == 0 {
        return 0;
    }
    if legacy {
        let mask = mask_covering(u64::from(range_incl), chunk_bits) as u32;
        loop {
            let value = buffer.next_chunk(bitgen, chunk_bits) & mask;
            if value <= range_incl {
                return value;
            }
        }
    } else {
        let range_excl = u64::from(range_incl) + 1;
        let modulus = 1u64 << chunk_bits;
        loop {
            let chunk = u64::from(buffer.next_chunk(bitgen, chunk_bits));
            let product = chunk * range_excl;
            let low = product & (modulus - 1);
            if low < range_excl {
                let threshold = (modulus - range_excl) % range_excl;
                if low < threshold {
                    continue;
                }
            }
            return (product >> chunk_bits) as u32;
        }
    }
}

/// The chunk width for each dtype narrower than 32 bits, or `None` for dtypes that draw whole
/// words.
pub(in crate::python) fn narrow_chunk_bits(dtype_name: &str) -> Option<u32> {
    match dtype_name {
        "bool" => Some(1),
        "int8" | "uint8" => Some(8),
        "int16" | "uint16" => Some(16),
        _ => None,
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
