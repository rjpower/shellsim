//! CPython's `hash()` algorithms for builtin values.
//!
//! Numbers use CPython's specified numeric hash, reduction modulo the Mersenne prime
//! `2**61 - 1`, so equal numbers of different types hash alike (`hash(2.0) == hash(2)`).
//! Strings and bytes use SipHash-1-3 with an all-zero key, which is what CPython 3.14 computes
//! under `PYTHONHASHSEED=0`. Real CPython randomizes the key per process; a fixed key keeps
//! simulated runs deterministic. Tuples and frozensets combine item hashes exactly as CPython
//! does. The VM supplies item hashes, because they may come from a user's `__hash__`.

use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;

/// The modulus of CPython's numeric hash, `2**61 - 1`.
const MODULUS: u64 = (1 << 61) - 1;
const MODULUS_BITS: i32 = 61;
/// `hash(float("inf"))`.
const INFINITY: i64 = 314_159;
/// Multiplier for the imaginary part of a complex hash.
const IMAGINARY: u64 = 1_000_003;

/// CPython never returns `-1` from a hash, because `-1` signals an error in its C API.
fn finish(hash: i64) -> i64 {
    if hash == -1 {
        -2
    } else {
        hash
    }
}

/// `hash(None)` in CPython 3.12 and later, a fixed constant.
pub(super) const NONE: i64 = 4_238_894_112;

/// `hash(n)` for a machine integer.
pub(super) fn integer(value: i64) -> i64 {
    let magnitude = value.unsigned_abs() % MODULUS;
    let magnitude = i64::try_from(magnitude).expect("reduced below 2**61");
    finish(if value < 0 { -magnitude } else { magnitude })
}

/// `hash(n)` for an arbitrary-precision integer.
pub(super) fn big_integer(value: &BigInt) -> i64 {
    let magnitude = (value.magnitude() % MODULUS)
        .to_u64()
        .expect("reduced below 2**61");
    let magnitude = i64::try_from(magnitude).expect("reduced below 2**61");
    finish(match value.sign() {
        Sign::Minus => -magnitude,
        Sign::NoSign | Sign::Plus => magnitude,
    })
}

/// `hash(x)` for a float: the hash of the exact rational value, so integral floats hash like
/// the equal integer. NaN hashes to `0`, a deterministic stand-in for CPython's identity-based
/// NaN hash.
pub(super) fn float(value: f64) -> i64 {
    if value.is_nan() {
        return 0;
    }
    if value.is_infinite() {
        return if value > 0.0 { INFINITY } else { -INFINITY };
    }
    if value == 0.0 {
        return 0;
    }
    // Follows `_Py_HashDouble`: consume the mantissa 28 bits at a time, then rotate by the
    // remaining exponent within the 61-bit modulus.
    let (mut mantissa, mut exponent) = frexp(value.abs());
    let mut hash: u64 = 0;
    while mantissa != 0.0 {
        hash = ((hash << 28) & MODULUS) | (hash >> (MODULUS_BITS - 28));
        mantissa *= 268_435_456.0;
        exponent -= 28;
        let digit = mantissa as u64;
        mantissa -= digit as f64;
        hash += digit;
        if hash >= MODULUS {
            hash -= MODULUS;
        }
    }
    let exponent = exponent.rem_euclid(MODULUS_BITS);
    hash = ((hash << exponent) & MODULUS) | (hash >> (MODULUS_BITS - exponent));
    let hash = i64::try_from(hash).expect("reduced below 2**61");
    finish(if value < 0.0 { -hash } else { hash })
}

/// Split a positive finite float into a mantissa in `[0.5, 1)` and a binary exponent.
fn frexp(value: f64) -> (f64, i32) {
    let bits = value.to_bits();
    let exponent = i32::try_from((bits >> 52) & 0x7ff).expect("eleven-bit exponent");
    if exponent == 0 {
        // Subnormal: scale into the normal range first.
        let (mantissa, exponent) = frexp(value * 2f64.powi(64));
        return (mantissa, exponent - 64);
    }
    let mantissa = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
    (mantissa, exponent - 1022)
}

/// `hash(z)` for a complex number.
pub(super) fn complex(real: f64, imag: f64) -> i64 {
    let combined = (float(real) as u64).wrapping_add(IMAGINARY.wrapping_mul(float(imag) as u64));
    finish(combined as i64)
}

/// `hash(s)` for a string, over CPython's compact representation: one, two, or four bytes per
/// code point depending on the largest code point.
pub(super) fn string(value: &str) -> i64 {
    let widest = value.chars().map(u32::from).max().unwrap_or(0);
    let mut bytes = Vec::with_capacity(value.len());
    for point in value.chars().map(u32::from) {
        match widest {
            0..=0xff => bytes.push(point as u8),
            0x100..=0xffff => bytes.extend_from_slice(&(point as u16).to_le_bytes()),
            _ => bytes.extend_from_slice(&point.to_le_bytes()),
        }
    }
    self::bytes(&bytes)
}

/// `hash(b)` for bytes.
pub(super) fn bytes(value: &[u8]) -> i64 {
    if value.is_empty() {
        return 0;
    }
    finish(siphash13(value) as i64)
}

/// SipHash-1-3 with an all-zero key, as CPython uses when hash randomization is disabled.
fn siphash13(data: &[u8]) -> u64 {
    let mut v0: u64 = 0x736f_6d65_7073_6575;
    let mut v1: u64 = 0x646f_7261_6e64_6f6d;
    let mut v2: u64 = 0x6c79_6765_6e65_7261;
    let mut v3: u64 = 0x7465_6462_7974_6573;
    let round = |v0: &mut u64, v1: &mut u64, v2: &mut u64, v3: &mut u64| {
        *v0 = v0.wrapping_add(*v1);
        *v1 = v1.rotate_left(13) ^ *v0;
        *v0 = v0.rotate_left(32);
        *v2 = v2.wrapping_add(*v3);
        *v3 = v3.rotate_left(16) ^ *v2;
        *v0 = v0.wrapping_add(*v3);
        *v3 = v3.rotate_left(21) ^ *v0;
        *v2 = v2.wrapping_add(*v1);
        *v1 = v1.rotate_left(17) ^ *v2;
        *v2 = v2.rotate_left(32);
    };
    let mut chunks = data.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("eight-byte chunk"));
        v3 ^= word;
        round(&mut v0, &mut v1, &mut v2, &mut v3);
        v0 ^= word;
    }
    let mut last = (data.len() as u64 & 0xff) << 56;
    for (index, byte) in chunks.remainder().iter().enumerate() {
        last |= u64::from(*byte) << (8 * index);
    }
    v3 ^= last;
    round(&mut v0, &mut v1, &mut v2, &mut v3);
    v0 ^= last;
    v2 ^= 0xff;
    for _ in 0..3 {
        round(&mut v0, &mut v1, &mut v2, &mut v3);
    }
    v0 ^ v1 ^ v2 ^ v3
}

const XXPRIME_1: u64 = 11_400_714_785_074_694_791;
const XXPRIME_2: u64 = 14_029_467_366_897_019_727;
const XXPRIME_5: u64 = 2_870_177_450_012_600_261;

/// `hash(t)` for a tuple, from its items' hashes (CPython's xxHash-based `tuplehash`).
pub(super) fn tuple(items: &[i64]) -> i64 {
    let accumulator =
        xxhash_lanes(items).wrapping_add((items.len() as u64) ^ (XXPRIME_5 ^ 3_527_539));
    finish_xxhash(accumulator)
}

/// `hash(s)` for a slice, from the hashes of its start, stop and step: the tuple hash without
/// its final length step, as CPython's `slice_hash` computes it.
pub(super) fn slice(items: &[i64]) -> i64 {
    finish_xxhash(xxhash_lanes(items))
}

fn xxhash_lanes(items: &[i64]) -> u64 {
    let mut accumulator = XXPRIME_5;
    for item in items {
        accumulator = accumulator.wrapping_add((*item as u64).wrapping_mul(XXPRIME_2));
        accumulator = accumulator.rotate_left(31);
        accumulator = accumulator.wrapping_mul(XXPRIME_1);
    }
    accumulator
}

fn finish_xxhash(accumulator: u64) -> i64 {
    if accumulator == u64::MAX {
        return 1_546_275_796;
    }
    accumulator as i64
}

/// `hash(s)` for a frozenset, from its items' hashes (CPython's `frozenset_hash`).
pub(super) fn frozenset(items: &[i64]) -> i64 {
    let shuffle = |hash: u64| ((hash ^ 89_869_747) ^ (hash << 16)).wrapping_mul(3_644_798_167);
    let mut hash = items
        .iter()
        .fold(0u64, |hash, item| hash ^ shuffle(*item as u64));
    hash ^= (items.len() as u64 + 1).wrapping_mul(1_927_868_237);
    hash ^= (hash >> 11) ^ (hash >> 25);
    hash = hash.wrapping_mul(69_069).wrapping_add(907_133_923);
    if hash == u64::MAX {
        return 590_923_713;
    }
    hash as i64
}

/// The default `object.__hash__`. CPython derives it from the object's address; the VM passes
/// a stable identity such as an arena index instead, so the hash stays deterministic.
pub(super) fn identity(identity: u64) -> i64 {
    finish(identity as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values come from CPython 3.14.4 run with PYTHONHASHSEED=0.
    #[test]
    fn numbers_match_cpython() {
        assert_eq!(integer(-1), -2);
        assert_eq!(integer(1 << 61), 1);
        assert_eq!(big_integer(&(BigInt::from(1) << 100usize)), 549_755_813_888);
        assert_eq!(
            big_integer(&-(BigInt::from(1) << 100usize)),
            -549_755_813_888
        );
        assert_eq!(float(1.5), 1_152_921_504_606_846_977);
        assert_eq!(float(-0.5), -1_152_921_504_606_846_976);
        assert_eq!(float(1e300), 1_224_995_262_755_759_164);
        assert_eq!(float(f64::INFINITY), 314_159);
        assert_eq!(float(-1.0), -2);
        assert_eq!(float(0.5), 1 << 60);
        assert_eq!(float(5e-324), 16_777_216);
        assert_eq!(float(2.0), integer(2));
        assert_eq!(complex(1.0, 2.0), 2_000_007);
        assert_eq!(complex(0.0, 3.0), 3_000_009);
    }

    #[test]
    fn strings_and_bytes_match_cpython_without_randomization() {
        assert_eq!(string("abc"), -4_594_863_902_769_663_758);
        assert_eq!(string(""), 0);
        assert_eq!(string("héllo"), 6_395_329_678_795_984_700);
        assert_eq!(string("日本"), 6_243_316_497_235_261_705);
        assert_eq!(string("😀x"), -8_926_728_262_118_538_918);
        assert_eq!(bytes(b"abc"), -4_594_863_902_769_663_758);
    }

    #[test]
    fn containers_match_cpython() {
        assert_eq!(tuple(&[integer(1), integer(2)]), -3_550_055_125_485_641_917);
        assert_eq!(tuple(&[]), 5_740_354_900_026_072_187);
        let inner = tuple(&[float(2.5)]);
        assert_eq!(
            tuple(&[integer(1), string("a"), inner]),
            5_350_091_667_078_189_583
        );
        assert_eq!(frozenset(&[]), 133_146_708_735_736);
        assert_eq!(
            frozenset(&[integer(1), integer(2), integer(3)]),
            -272_375_401_224_217_160
        );
    }
}
