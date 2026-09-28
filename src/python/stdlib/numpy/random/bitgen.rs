//! PCG64 (O'Neill 2014, XSL-RR variant): the one bit generator behind every `numpy.random`
//! stream in shellsim.
//!
//! State is a 128-bit linear congruential generator advanced by PCG's published 128-bit
//! multiplier and a fixed odd 128-bit increment. Each step's output folds the 128 bits of state
//! to a 64-bit word by XORing its halves and rotating by the state's own top bits ("XSL RR", the
//! standard PCG64 output function).
//!
//! Seeding uses splitmix64 (Vigna's simple, well-distributed mixing step) to expand an arbitrary
//! seed into the 256 bits of state and increment. Streams are therefore reproducible within
//! shellsim but differ from NumPy's.

/// PCG's published 128-bit LCG multiplier (`PCG_DEFAULT_MULTIPLIER_128`).
const MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

fn step(state: u128, inc: u128) -> u128 {
    state.wrapping_mul(MULTIPLIER).wrapping_add(inc)
}

/// XSL-RR output: fold state to 64 bits with xorshift, then rotate by its own top 6 bits.
fn output(state: u128) -> u64 {
    let hi = (state >> 64) as u64;
    let lo = state as u64;
    let rotation = (state >> 122) as u32;
    (hi ^ lo).rotate_right(rotation)
}

/// One splitmix64 step (Vigna 2015): advance `state` by the golden-ratio increment and mix it
/// through two xorshift-multiply rounds. Used only to expand a seed's words into PCG64's initial
/// state and stream constant.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// PCG64's mutable state: the 128-bit LCG state and its fixed odd increment.
pub(in crate::python) struct Pcg64 {
    pub(in crate::python) state: u128,
    pub(in crate::python) inc: u128,
}

impl Pcg64 {
    /// Seed from any number of `u64` words (see `random.py`'s `_seed_words`, which turns an int
    /// or a sequence of ints into this list). The words are folded one at a time into a
    /// splitmix64 stream, then four more splitmix64 draws produce the 128-bit initial state and
    /// 128-bit stream constant PCG's own seeding procedure needs.
    pub(in crate::python) fn from_seed_words(words: &[u64]) -> Self {
        let mut mixer: u64 = words.len() as u64;
        for &word in words {
            mixer ^= word;
            mixer = splitmix64(&mut mixer);
        }
        let a = splitmix64(&mut mixer);
        let b = splitmix64(&mut mixer);
        let c = splitmix64(&mut mixer);
        let d = splitmix64(&mut mixer);
        let initstate = (u128::from(a) << 64) | u128::from(b);
        let initseq = (u128::from(c) << 64) | u128::from(d);
        Self::seeded(initstate, initseq)
    }

    /// PCG's own `pcg_setseq_128_srandom_r`: derive `(state, inc)` from a 128-bit initial state
    /// and a 128-bit initial sequence constant.
    fn seeded(initstate: u128, initseq: u128) -> Self {
        let inc = (initseq << 1) | 1;
        let mut state = step(0, inc);
        state = state.wrapping_add(initstate);
        state = step(state, inc);
        Pcg64 { state, inc }
    }

    /// The next 64-bit word.
    pub(in crate::python) fn next_u64(&mut self) -> u64 {
        self.state = step(self.state, self.inc);
        output(self.state)
    }

    /// A uniform double in `[0, 1)` from the top 53 bits of one 64-bit draw.
    pub(in crate::python) fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}
