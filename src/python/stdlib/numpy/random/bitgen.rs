//! MT19937 and PCG64 bit generators, plus `SeedSequence`'s NumPy-derived word arrays.
//!
//! MT19937 follows Matsumoto & Nishimura's 1998 "Mersenne Twister" reference algorithm:
//! `init_genrand` and `init_by_array` seed the 624-word state, and each draw either taps the
//! current word or re-twists the whole array (the standard recurrence with matrix `A =
//! 0x9908b0df`) before tempering it. PCG64 follows O'Neill's 2014 PCG family: 128-bit LCG state
//! advanced by the 128-bit multiplier `0x2360ed051fc65da44385df649fccf645`, output through the
//! "XSL RR" (xorshift-low, random-rotation) function that folds the state to 64 bits.
//!
//! Both generators reproduce NumPy's seeded streams bit for bit. The constants are the published
//! PCG and Mersenne Twister ones; the NumPy-specific choices (how `SeedSequence` output seeds
//! each generator, the initial twister position, PCG64's word order) match NumPy's observed
//! output.
//!
//! [`BitGen`] is the common draw interface both generators implement: `next_u32`/`next_u64` are
//! the "next word" primitives distributions are built on, `next_double` is each generator's own
//! fast 53-bit uniform (MT19937 combines two tempered words the classic `genrand_res53` way;
//! PCG64 takes the high 53 bits of one 64-bit draw), and `next_raw` is the generator's native
//! word zero-extended to 64 bits, which is what `random_raw()` exposes.

pub(in crate::python) const MT_N: usize = 624;
const MT_M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// `init_genrand`: the classic single-word MT19937 seed used by legacy `RandomState` for a
/// scalar integer seed.
pub(in crate::python) fn mt_init_genrand(seed: u32) -> Box<[u32; MT_N]> {
    let mut mt = Box::new([0u32; MT_N]);
    mt[0] = seed;
    for i in 1..MT_N {
        mt[i] = 1_812_433_253u32
            .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
            .wrapping_add(i as u32);
    }
    mt
}

/// `init_by_array`: MT19937 seeded from a key array, used by legacy `RandomState` for an
/// array-like seed. `MT19937(seed)` does not call this: it seeds directly from
/// `SeedSequence(seed).generate_state(624)`, only overwriting word 0 with the same
/// `0x80000000` this function's final assignment produces (see `random.rs`), which is a
/// property of the published algorithm rather than a NumPy-specific choice.
pub(in crate::python) fn mt_init_by_array(key: &[u32]) -> Box<[u32; MT_N]> {
    let mut mt = mt_init_genrand(19_650_218);
    let mut i = 1usize;
    let mut j = 0usize;
    let count = MT_N.max(key.len());
    for _ in 0..count {
        mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
        i += 1;
        j += 1;
        if i >= MT_N {
            mt[0] = mt[MT_N - 1];
            i = 1;
        }
        if j >= key.len() {
            j = 0;
        }
    }
    for _ in 0..MT_N - 1 {
        mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
        i += 1;
        if i >= MT_N {
            mt[0] = mt[MT_N - 1];
            i = 1;
        }
    }
    mt[0] = 0x8000_0000;
    mt
}

fn mt_twist(mt: &mut [u32; MT_N]) {
    for i in 0..MT_N {
        let y = (mt[i] & UPPER_MASK) | (mt[(i + 1) % MT_N] & LOWER_MASK);
        let mut next = mt[(i + MT_M) % MT_N] ^ (y >> 1);
        if y & 1 != 0 {
            next ^= MATRIX_A;
        }
        mt[i] = next;
    }
}

fn mt_temper(y: u32) -> u32 {
    let mut y = y;
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c_5680;
    y ^= (y << 15) & 0xefc6_0000;
    y ^= y >> 18;
    y
}

/// PCG64's 128-bit LCG multiplier (O'Neill 2014, the published `PCG_DEFAULT_MULTIPLIER_128`).
pub(in crate::python) const PCG_MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

fn pcg_step(state: u128, inc: u128) -> u128 {
    state.wrapping_mul(PCG_MULTIPLIER).wrapping_add(inc)
}

/// PCG XSL-RR 128/64 output: fold state to 64 bits with xorshift, then rotate by its top bits.
fn pcg_output(state: u128) -> u64 {
    let hi = (state >> 64) as u64;
    let lo = state as u64;
    let rotation = (state >> 122) as u32;
    (hi ^ lo).rotate_right(rotation)
}

/// The LCG jump-ahead behind `advance` and `jumped`: composing the step function
/// `delta` times is itself affine, and repeated squaring computes its coefficients in
/// `O(log2(delta))` steps (the standard PCG `pcg_advance_lcg_128` technique).
fn lcg_advance(state: u128, mut delta: u128, mut cur_mult: u128, mut cur_plus: u128) -> u128 {
    let mut acc_mult: u128 = 1;
    let mut acc_plus: u128 = 0;
    while delta > 0 {
        if delta & 1 == 1 {
            acc_mult = acc_mult.wrapping_mul(cur_mult);
            acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
        }
        cur_plus = (cur_mult.wrapping_add(1)).wrapping_mul(cur_plus);
        cur_mult = cur_mult.wrapping_mul(cur_mult);
        delta >>= 1;
    }
    acc_mult.wrapping_mul(state).wrapping_add(acc_plus)
}

pub(in crate::python) fn pcg_advance(state: u128, inc: u128, delta: u128) -> u128 {
    lcg_advance(state, delta, PCG_MULTIPLIER, inc)
}

/// PCG64's `pcg_setseq_128_srandom_r`: derive `(state, inc)` from a 128-bit initial state and a
/// 128-bit initial sequence constant, both taken from `SeedSequence`.
pub(in crate::python) fn pcg_seed(initstate: u128, initseq: u128) -> (u128, u128) {
    let inc = (initseq << 1) | 1;
    let mut state = 0u128;
    state = pcg_step(state, inc);
    state = state.wrapping_add(initstate);
    state = pcg_step(state, inc);
    (state, inc)
}

/// Mutable state of one of NumPy's two bit generators, plus the draw primitives every
/// distribution is built on.
pub(in crate::python) enum BitGen {
    Mt19937 {
        key: Box<[u32; MT_N]>,
        pos: usize,
    },
    Pcg64 {
        state: u128,
        inc: u128,
        has_uint32: bool,
        uinteger: u32,
    },
}

impl BitGen {
    /// The next 32-bit word: MT19937's own tempered output, or half of a buffered PCG64 draw
    /// (the low half returns first; the high half is cached in `uinteger` for the next call, as
    /// NumPy's `has_uint32`/`uinteger` bit generator fields track).
    pub(in crate::python) fn next_u32(&mut self) -> u32 {
        match self {
            Self::Mt19937 { key, pos } => {
                if *pos >= MT_N {
                    mt_twist(key);
                    *pos = 0;
                }
                let word = key[*pos];
                *pos += 1;
                mt_temper(word)
            }
            Self::Pcg64 {
                state,
                inc,
                has_uint32,
                uinteger,
            } => {
                if *has_uint32 {
                    *has_uint32 = false;
                    return *uinteger;
                }
                *state = pcg_step(*state, *inc);
                let word = pcg_output(*state);
                *uinteger = (word >> 32) as u32;
                *has_uint32 = true;
                word as u32
            }
        }
    }

    fn next_u64_pcg(&mut self) -> u64 {
        let Self::Pcg64 { state, inc, .. } = self else {
            unreachable!("next_u64_pcg is only called on PCG64 state")
        };
        *state = pcg_step(*state, *inc);
        pcg_output(*state)
    }

    /// The next 64-bit word: MT19937 combines two tempered draws with the first as the high
    /// half (`mt19937_next64`); PCG64 steps once and outputs natively.
    pub(in crate::python) fn next_u64(&mut self) -> u64 {
        match self {
            Self::Mt19937 { .. } => {
                let hi = u64::from(self.next_u32());
                let lo = u64::from(self.next_u32());
                (hi << 32) | lo
            }
            Self::Pcg64 { .. } => self.next_u64_pcg(),
        }
    }

    /// The generator's native word, zero-extended: what `random_raw()` returns.
    pub(in crate::python) fn next_raw(&mut self) -> u64 {
        match self {
            Self::Mt19937 { .. } => u64::from(self.next_u32()),
            Self::Pcg64 { .. } => self.next_u64(),
        }
    }

    /// A uniform double in `[0, 1)` at each generator's own best precision: MT19937's classic
    /// 27+26-bit `genrand_res53`, PCG64's top 53 bits of one 64-bit draw.
    pub(in crate::python) fn next_double(&mut self) -> f64 {
        match self {
            Self::Mt19937 { .. } => {
                let a = u64::from(self.next_u32() >> 5);
                let b = u64::from(self.next_u32() >> 6);
                (a as f64 * 67_108_864.0 + b as f64) * (1.0 / 9_007_199_254_740_992.0)
            }
            Self::Pcg64 { .. } => (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0),
        }
    }

    /// A uniform `f32` in `[0, 1)`: `next_u32() >> 8` scaled by `2**-24`, matching NumPy's
    /// single-precision draws (verified bit for bit against `Generator.random(dtype=float32)`).
    pub(in crate::python) fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0f32 / 16_777_216.0f32)
    }
}
