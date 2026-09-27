//! Bit generators behind `numpy.random`: MT19937, PCG64, SeedSequence mixing, and the metered
//! stream every distribution draws from.
//!
//! The algorithms reproduce NumPy 2.5 bit for bit: MT19937 with NumPy's two legacy seedings
//! (`mt19937_seed` for integer seeds and `init_by_array` for array seeds; SeedSequence keys are
//! assembled in Python), PCG64 as the 128-bit LCG with the XSL-RR output and a buffered upper
//! half for 32-bit draws, and SeedSequence's hashmix entropy pool.
//!
//! State lives in a Python-visible `uint64` array owned by the bit-generator object, so the
//! frozen `numpy.random` classes stay plain Python and every native call reads the state, draws,
//! and writes it back. The layout is a small header followed by engine words; see
//! [`STATE_HEADER`]. Loading validates the array so a user-modified state cannot index out of
//! bounds.

use super::super::super::super::native::{PyError, PyResult, PyRuntime, PyValue};
use super::super::array::{self, Array};
use super::super::dtype::DType;

/// Words in the MT19937 key.
pub(super) const MT_N: usize = 624;
const MT_M: usize = 397;

/// Header slots shared by every engine: kind, legacy Gaussian cache, and the PCG64 32-bit
/// buffer. Engine words start at this index.
///
/// | slot | meaning |
/// |------|---------|
/// | 0 | engine kind ([`MT19937`] or [`PCG64`]) |
/// | 1 | `has_gauss` of `RandomState`'s polar-method cache |
/// | 2 | cached Gaussian as `f64` bits |
/// | 3 | `has_uint32` (PCG64 only) |
/// | 4 | buffered upper 32 bits (PCG64 only) |
/// | 5.. | MT19937: `pos`, then 624 key words; PCG64: state high/low, increment high/low |
pub(super) const STATE_HEADER: usize = 5;
pub(super) const MT19937: u64 = 1;
pub(super) const PCG64: u64 = 2;

const MT_STATE_LEN: usize = STATE_HEADER + 1 + MT_N;
const PCG_STATE_LEN: usize = STATE_HEADER + 4;

/// Chunk of CPU units charged whenever prepaid draws run out.
const CHARGE_CHUNK: u64 = 1024;

/// The Mersenne Twister state as NumPy's `mt19937_state` keeps it.
#[derive(Clone)]
pub(super) struct Mt19937 {
    key: [u32; MT_N],
    pos: usize,
}

impl Mt19937 {
    /// `mt19937_seed`: Knuth's linear recurrence from one 32-bit seed (`RandomState(int)`).
    pub(super) fn from_seed(seed: u32) -> Self {
        let mut key = [0u32; MT_N];
        let mut value = seed;
        for (index, word) in key.iter_mut().enumerate() {
            *word = value;
            value = 1_812_433_253u32
                .wrapping_mul(value ^ (value >> 30))
                .wrapping_add(index as u32 + 1);
        }
        Self { key, pos: MT_N }
    }

    /// `mt19937_init_by_array`: the reference array seeding used for sequence seeds.
    pub(super) fn from_key_array(init_key: &[u32]) -> Self {
        let mut state = Self::from_seed(19_650_218);
        let mt = &mut state.key;
        let length = init_key.len().max(1);
        let mut i = 1usize;
        let mut j = 0usize;
        for _ in 0..MT_N.max(length) {
            let previous = mt[i - 1] ^ (mt[i - 1] >> 30);
            let word = init_key.get(j).copied().unwrap_or(0);
            mt[i] = (mt[i] ^ previous.wrapping_mul(1_664_525))
                .wrapping_add(word)
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= MT_N {
                mt[0] = mt[MT_N - 1];
                i = 1;
            }
            if j >= length {
                j = 0;
            }
        }
        for _ in 0..MT_N - 1 {
            let previous = mt[i - 1] ^ (mt[i - 1] >> 30);
            mt[i] = (mt[i] ^ previous.wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= MT_N {
                mt[0] = mt[MT_N - 1];
                i = 1;
            }
        }
        mt[0] = 0x8000_0000;
        state.pos = MT_N;
        state
    }

    /// An explicit key and position, from `set_state` or a SeedSequence-derived key.
    pub(super) fn from_parts(key: &[u32], pos: usize) -> PyResult<Self> {
        if key.len() != MT_N {
            return Err(PyError::value_error("state must be 624 elements"));
        }
        if pos > MT_N {
            return Err(PyError::value_error("pos must be between 0 and 624"));
        }
        let mut words = [0u32; MT_N];
        words.copy_from_slice(key);
        Ok(Self { key: words, pos })
    }

    fn generate(&mut self) {
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        let key = &mut self.key;
        let twist = |upper: u32, lower: u32, far: u32| {
            let y = (upper & UPPER) | (lower & LOWER);
            far ^ (y >> 1) ^ (0u32.wrapping_sub(y & 1) & MATRIX_A)
        };
        for i in 0..MT_N - MT_M {
            key[i] = twist(key[i], key[i + 1], key[i + MT_M]);
        }
        for i in MT_N - MT_M..MT_N - 1 {
            key[i] = twist(key[i], key[i + 1], key[i + MT_M - MT_N]);
        }
        key[MT_N - 1] = twist(key[MT_N - 1], key[0], key[MT_M - 1]);
        self.pos = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.pos >= MT_N {
            self.generate();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

/// PCG64's multiplier, `PCG_DEFAULT_MULTIPLIER_128`.
const PCG_MULTIPLIER: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

/// PCG64 (`pcg_setseq_128_xsl_rr_64`) plus NumPy's buffered upper half for 32-bit draws.
#[derive(Clone)]
pub(super) struct Pcg64 {
    pub(super) state: u128,
    pub(super) inc: u128,
    pub(super) has_uint32: bool,
    pub(super) uinteger: u32,
}

impl Pcg64 {
    /// `pcg64_set_seed`: the state and stream words come from `SeedSequence.generate_state(4,
    /// uint64)`, most significant word first.
    pub(super) fn from_words(words: [u64; 4]) -> Self {
        let initial = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let sequence = (u128::from(words[2]) << 64) | u128::from(words[3]);
        let mut generator = Self {
            state: 0,
            inc: (sequence << 1) | 1,
            has_uint32: false,
            uinteger: 0,
        };
        generator.step();
        generator.state = generator.state.wrapping_add(initial);
        generator.step();
        generator
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.inc);
    }

    /// `pcg64_advance`: jump `delta` steps ahead in O(log delta) with
    /// `pcg_advance_lcg_128`, and drop the buffered 32-bit half.
    pub(super) fn advance(&mut self, mut delta: u128) {
        let (mut multiplier, mut increment) = (PCG_MULTIPLIER, self.inc);
        let (mut total_multiplier, mut total_increment) = (1u128, 0u128);
        while delta > 0 {
            if delta & 1 == 1 {
                total_multiplier = total_multiplier.wrapping_mul(multiplier);
                total_increment = total_increment
                    .wrapping_mul(multiplier)
                    .wrapping_add(increment);
            }
            increment = multiplier.wrapping_add(1).wrapping_mul(increment);
            multiplier = multiplier.wrapping_mul(multiplier);
            delta >>= 1;
        }
        self.state = total_multiplier
            .wrapping_mul(self.state)
            .wrapping_add(total_increment);
        self.has_uint32 = false;
        self.uinteger = 0;
    }

    fn next_u64(&mut self) -> u64 {
        self.step();
        let rotation = (self.state >> 122) as u32;
        (((self.state >> 64) as u64) ^ (self.state as u64)).rotate_right(rotation)
    }

    fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        next as u32
    }
}

#[derive(Clone)]
pub(super) enum Engine {
    Mt(Box<Mt19937>),
    Pcg(Pcg64),
}

/// A bit generator plus the legacy Gaussian cache that `RandomState` keeps beside it.
#[derive(Clone)]
pub(super) struct BitGen {
    pub(super) engine: Engine,
    pub(super) has_gauss: bool,
    pub(super) gauss: f64,
}

impl BitGen {
    pub(super) fn new(engine: Engine) -> Self {
        Self {
            engine,
            has_gauss: false,
            gauss: 0.0,
        }
    }

    fn next_u32(&mut self) -> u32 {
        match &mut self.engine {
            Engine::Mt(mt) => mt.next_u32(),
            Engine::Pcg(pcg) => pcg.next_u32(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        match &mut self.engine {
            Engine::Mt(mt) => (u64::from(mt.next_u32()) << 32) | u64::from(mt.next_u32()),
            Engine::Pcg(pcg) => pcg.next_u64(),
        }
    }

    /// `BitGenerator.random_raw`: MT19937's raw output is one 32-bit word.
    fn next_raw(&mut self) -> u64 {
        match &mut self.engine {
            Engine::Mt(mt) => u64::from(mt.next_u32()),
            Engine::Pcg(pcg) => pcg.next_u64(),
        }
    }

    /// A double in [0, 1) with 53 random bits; MT19937 builds it from two 32-bit words.
    fn next_double(&mut self) -> f64 {
        match &mut self.engine {
            Engine::Mt(mt) => {
                let high = f64::from(mt.next_u32() >> 5);
                let low = f64::from(mt.next_u32() >> 6);
                (high * 67_108_864.0 + low) / 9_007_199_254_740_992.0
            }
            Engine::Pcg(pcg) => (pcg.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0),
        }
    }

    /// Serialize into the state-array layout described at [`STATE_HEADER`].
    pub(super) fn to_words(&self) -> Vec<u64> {
        let mut words = vec![0u64; STATE_HEADER];
        words[1] = u64::from(self.has_gauss);
        words[2] = self.gauss.to_bits();
        match &self.engine {
            Engine::Mt(mt) => {
                words[0] = MT19937;
                words.push(mt.pos as u64);
                words.extend(mt.key.iter().map(|word| u64::from(*word)));
            }
            Engine::Pcg(pcg) => {
                words[0] = PCG64;
                words[3] = u64::from(pcg.has_uint32);
                words[4] = u64::from(pcg.uinteger);
                words.extend([
                    (pcg.state >> 64) as u64,
                    pcg.state as u64,
                    (pcg.inc >> 64) as u64,
                    pcg.inc as u64,
                ]);
            }
        }
        words
    }

    fn from_words(words: &[u64]) -> PyResult<Self> {
        let invalid = || PyError::value_error("invalid bit generator state");
        let kind = *words.first().ok_or_else(invalid)?;
        let engine = match (kind, words.len()) {
            (MT19937, MT_STATE_LEN) => {
                let key = words[STATE_HEADER + 1..]
                    .iter()
                    .map(|word| *word as u32)
                    .collect::<Vec<_>>();
                let pos = usize::try_from(words[STATE_HEADER]).map_err(|_| invalid())?;
                Engine::Mt(Box::new(Mt19937::from_parts(&key, pos)?))
            }
            (PCG64, PCG_STATE_LEN) => {
                let word = |index: usize| u128::from(words[STATE_HEADER + index]);
                Engine::Pcg(Pcg64 {
                    state: (word(0) << 64) | word(1),
                    inc: (word(2) << 64) | word(3),
                    has_uint32: words[3] != 0,
                    uinteger: words[4] as u32,
                })
            }
            _ => return Err(invalid()),
        };
        Ok(Self {
            engine,
            has_gauss: words[1] != 0,
            gauss: f64::from_bits(words[2]),
        })
    }
}

/// Allocate a new state array holding `bits`.
pub(super) fn new_state(runtime: &mut dyn PyRuntime, bits: &BitGen) -> PyResult<PyValue> {
    let words = bits.to_words();
    let shape = vec![words.len()];
    Ok(array::array_from_elements(runtime, DType::UINT64, shape, &words)?.value())
}

/// Read and validate a state array.
pub(super) fn load(runtime: &mut dyn PyRuntime, state: PyValue) -> PyResult<(Array, BitGen)> {
    let array = Array::from_value(runtime, state)?;
    if array.dtype != DType::UINT64 || array.ndim() != 1 {
        return Err(PyError::value_error("invalid bit generator state"));
    }
    let words = array::read_elements::<u64>(runtime, &array)?;
    let bits = BitGen::from_words(&words)?;
    Ok((array, bits))
}

/// Write `bits` back into `array`, which [`load`] validated.
pub(super) fn store(runtime: &mut dyn PyRuntime, array: &Array, bits: &BitGen) -> PyResult<()> {
    let words = bits.to_words();
    if words.len() != array.size() {
        return Err(PyError::value_error("invalid bit generator state"));
    }
    let offsets = array.offsets().collect::<Vec<_>>();
    runtime.write_array(array.handle, &mut |target| {
        let super::super::super::super::native::PyArrayDataMut::Bytes(bytes) = target.data else {
            return Err(PyError::runtime_error("state array has object storage"));
        };
        for (word, offset) in words.iter().zip(&offsets) {
            bytes[*offset..*offset + 8].copy_from_slice(&word.to_le_bytes());
        }
        Ok(())
    })
}

/// A bit generator borrowed from its state array for one native call.
///
/// Every raw draw spends one CPU unit of credit; when credit runs out the stream charges the
/// runtime for another chunk before drawing, so rejection loops stay metered however long they
/// run. [`Stream::prepay`] charges the expected cost of a whole fill up front. The state is
/// written back by [`Stream::finish`]; an error before that leaves the Python-visible state
/// unchanged.
pub(super) struct Stream<'r> {
    runtime: &'r mut dyn PyRuntime,
    array: Array,
    pub(super) bits: BitGen,
    credit: u64,
}

impl<'r> Stream<'r> {
    pub(super) fn open(runtime: &'r mut dyn PyRuntime, state: PyValue) -> PyResult<Self> {
        let (array, bits) = load(runtime, state)?;
        Ok(Self {
            runtime,
            array,
            bits,
            credit: 0,
        })
    }

    /// Charge `units` now and bank them for the draws that follow.
    pub(super) fn prepay(&mut self, units: u64) -> PyResult<()> {
        self.runtime.charge_cpu(units)?;
        self.credit = self.credit.saturating_add(units);
        Ok(())
    }

    /// Spend `units` of prepaid work, charging another chunk when the credit runs out.
    pub(super) fn spend(&mut self, units: u64) -> PyResult<()> {
        while self.credit < units {
            self.runtime.charge_cpu(CHARGE_CHUNK)?;
            self.credit += CHARGE_CHUNK;
        }
        self.credit -= units;
        Ok(())
    }

    pub(super) fn next_u32(&mut self) -> PyResult<u32> {
        self.spend(1)?;
        Ok(self.bits.next_u32())
    }

    pub(super) fn next_u64(&mut self) -> PyResult<u64> {
        self.spend(1)?;
        Ok(self.bits.next_u64())
    }

    pub(super) fn next_raw(&mut self) -> PyResult<u64> {
        self.spend(1)?;
        Ok(self.bits.next_raw())
    }

    pub(super) fn next_double(&mut self) -> PyResult<f64> {
        self.spend(1)?;
        Ok(self.bits.next_double())
    }

    /// NumPy's `next_float`: the top 24 bits of a 32-bit draw.
    pub(super) fn next_float(&mut self) -> PyResult<f32> {
        Ok((self.next_u32()? >> 8) as f32 * (1.0 / 16_777_216.0))
    }

    /// Write the advanced state back and release the runtime.
    pub(super) fn finish(self) -> PyResult<&'r mut dyn PyRuntime> {
        store(self.runtime, &self.array, &self.bits)?;
        Ok(self.runtime)
    }
}

/// SeedSequence's hash constants and mixing, from `numpy/random/bit_generator.pyx`.
mod seed_sequence {
    const INIT_A: u32 = 0x43b0_d7e5;
    const MULT_A: u32 = 0x931e_8875;
    const INIT_B: u32 = 0x8b51_f9dd;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_MULT_L: u32 = 0xca01_f9dd;
    const MIX_MULT_R: u32 = 0x4973_f715;
    const XSHIFT: u32 = 16;

    fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
        let mut value = value ^ *hash_const;
        *hash_const = hash_const.wrapping_mul(MULT_A);
        value = value.wrapping_mul(*hash_const);
        value ^ (value >> XSHIFT)
    }

    fn mix(x: u32, y: u32) -> u32 {
        let result = MIX_MULT_L
            .wrapping_mul(x)
            .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        result ^ (result >> XSHIFT)
    }

    /// `SeedSequence.mix_entropy`: hash the assembled entropy words into a pool.
    pub(in super::super) fn pool(entropy: &[u32], pool_size: usize) -> Vec<u32> {
        let mut hash_const = INIT_A;
        let mut pool = (0..pool_size)
            .map(|index| hashmix(entropy.get(index).copied().unwrap_or(0), &mut hash_const))
            .collect::<Vec<_>>();
        for source in 0..pool_size {
            for target in 0..pool_size {
                if source != target {
                    let hashed = hashmix(pool[source], &mut hash_const);
                    pool[target] = mix(pool[target], hashed);
                }
            }
        }
        for word in entropy.iter().skip(pool_size) {
            for slot in &mut pool {
                let hashed = hashmix(*word, &mut hash_const);
                *slot = mix(*slot, hashed);
            }
        }
        pool
    }

    /// `SeedSequence.generate_state` for `uint32` words.
    pub(in super::super) fn generate(pool: &[u32], words: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        (0..words)
            .map(|index| {
                let mut value = pool[index % pool.len()] ^ hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                value = value.wrapping_mul(hash_const);
                value ^ (value >> XSHIFT)
            })
            .collect()
    }
}

pub(super) use seed_sequence::{generate as seed_generate, pool as seed_pool};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_the_reference_first_outputs() {
        // `np.random.RandomState(0).randint(0, 2**32, dtype=np.uint32)` in NumPy 2.5.3.
        let mut mt = Mt19937::from_seed(0);
        let outputs = (0..3).map(|_| mt.next_u32()).collect::<Vec<_>>();
        assert_eq!(outputs, [2_357_136_044, 2_546_248_239, 3_071_714_933]);
    }

    #[test]
    fn seed_sequence_and_pcg64_match_numpy() {
        // `np.random.SeedSequence(0).generate_state(4)` and `PCG64(0).state` in NumPy 2.5.3.
        let pool = seed_pool(&[0], 4);
        assert_eq!(
            seed_generate(&pool, 4),
            [2_968_811_710, 3_677_149_159, 745_650_761, 2_884_920_346]
        );
        let words = seed_generate(&pool, 8);
        let pairs = [0, 1, 2, 3]
            .map(|index| u64::from(words[2 * index]) | (u64::from(words[2 * index + 1]) << 32));
        let pcg = Pcg64::from_words(pairs);
        assert_eq!(
            pcg.state,
            35_399_562_948_360_463_058_890_781_895_381_311_971
        );
        assert_eq!(pcg.inc, 87_136_372_517_582_989_555_478_159_403_783_844_777);
    }

    #[test]
    fn pcg64_buffers_the_upper_half_for_32_bit_draws() {
        let mut first = Pcg64::from_words([1, 2, 3, 4]);
        let mut second = first.clone();
        let wide = second.next_u64();
        assert_eq!(first.next_u32(), wide as u32);
        assert_eq!(first.next_u32(), (wide >> 32) as u32);
    }

    #[test]
    fn pcg64_advance_matches_repeated_steps() {
        let mut stepped = Pcg64::from_words([5, 6, 7, 8]);
        let mut advanced = stepped.clone();
        for _ in 0..1000 {
            stepped.step();
        }
        advanced.advance(1000);
        assert_eq!(advanced.state, stepped.state);
        assert_eq!(advanced.next_u64(), stepped.next_u64());
    }

    #[test]
    fn state_words_round_trip() {
        let bits = BitGen::new(Engine::Mt(Box::new(Mt19937::from_key_array(&[1, 2, 3]))));
        let restored = BitGen::from_words(&bits.to_words()).unwrap();
        assert_eq!(restored.to_words(), bits.to_words());
        let mut corrupt = bits.to_words();
        corrupt[STATE_HEADER] = 700;
        assert!(BitGen::from_words(&corrupt).is_err());
    }
}
