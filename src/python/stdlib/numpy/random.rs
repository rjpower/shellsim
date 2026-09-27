//! Seeding and sampling kernels behind `numpy.random`, exported as `_numpy_random`.
//!
//! The frozen `numpy.random` module keeps NumPy's classes (`SeedSequence`, `MT19937`, `PCG64`,
//! `Generator` and `RandomState`) in Python and calls these kernels to seed and to draw. A bit
//! generator's state is a `uint64` array owned by the Python object (layout in [`bitgen`]).
//! Every drawing kernel loads it, draws, and stores the advanced state, so a stream continues
//! across calls as NumPy's does.
//!
//! The kernels port NumPy 2.5's `distributions.c`, `legacy-distributions.c`, and the shuffles
//! and Floyd sampling of `_generator.pyx` and `mtrand.pyx` operation for operation, so seeded
//! streams match NumPy bit for bit. The ziggurat's `exp` and `log1p` come from the platform
//! libm, as NumPy's do. Each raw draw spends metered CPU credit through [`bitgen::Stream`], and
//! output arrays are reserved before they are filled.

mod bitgen;
mod distributions;
mod ziggurat;

use super::super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyResult, PyRuntime, PyValue,
};
use super::super::super::Value;
use super::args::index_int;
use super::array::{self, Array};
use super::dtype::{DType, Kind};
use super::element::Element;
use bitgen::{BitGen, Engine, Mt19937, Pcg64, Stream, MT_N};
use distributions::{Continuous, Family};
use ziggurat::{
    FI_DOUBLE, FI_FLOAT, KI_DOUBLE, KI_FLOAT, NOR_INV_R, NOR_INV_R_F, NOR_R, NOR_R_F, WI_DOUBLE,
    WI_FLOAT,
};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_random",
    functions: FUNCTIONS,
    values: &[],
};

const fn function(
    name: &'static str,
    call: fn(&mut dyn PyRuntime, CallArgs) -> PyResult,
) -> FunctionDef {
    FunctionDef {
        module: "_numpy_random",
        name,
        call,
    }
}

static FUNCTIONS: &[FunctionDef] = &[
    function("mt19937_seed", mt19937_seed),
    function("mt19937_init_by_array", mt19937_init_by_array),
    function("mt19937_from_key", mt19937_from_key),
    function("pcg64_from_seed", pcg64_from_seed),
    function("pcg64_from_parts", pcg64_from_parts),
    function("pcg64_advance", pcg64_advance),
    function("seed_sequence_pool", seed_sequence_pool),
    function("seed_sequence_generate", seed_sequence_generate),
    function("get_gauss", get_gauss),
    function("set_gauss", set_gauss),
    function("random_raw", random_raw),
    function("standard_uniform", standard_uniform),
    function("standard_normal", standard_normal),
    function("legacy_gauss", legacy_gauss),
    function("standard_exponential", standard_exponential),
    function("legacy_standard_exponential", legacy_standard_exponential),
    function("continuous", continuous),
    function("standard_gamma_f32", standard_gamma_f32),
    function("discrete", discrete),
    function("bounded_integers", bounded_integers),
    function("shuffle_indices", shuffle_indices),
    function("floyd_sample", floyd_sample),
];

/// The `N` positional arguments of an internal kernel; the Python layer validates user input
/// before calling, so these only guard the calling convention.
fn positional<const N: usize>(args: &CallArgs, function: &str) -> PyResult<[PyValue; N]> {
    args.reject_keywords(function)?;
    args.expect_positional(function, N, N)?;
    Ok(std::array::from_fn(|index| args.positional()[index]))
}

fn length(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<usize> {
    usize::try_from(index_int(runtime, value)?)
        .map_err(|_| PyError::value_error("negative dimensions are not allowed"))
}

fn flag(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<bool> {
    Ok(index_int(runtime, value)? != 0)
}

/// Every element of a 1-d array of exactly `dtype`.
fn elements<T: Element>(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    dtype: DType,
) -> PyResult<Vec<T>> {
    let array = Array::from_value(runtime, value)?;
    if array.dtype != dtype || array.ndim() != 1 {
        return Err(PyError::type_error(format!(
            "expected a 1-d {} array",
            dtype.repr()
        )));
    }
    array::read_elements(runtime, &array)
}

fn new_array<T: Element>(runtime: &mut dyn PyRuntime, dtype: DType, values: &[T]) -> PyResult {
    Ok(array::array_from_elements(runtime, dtype, vec![values.len()], values)?.value())
}

fn mt19937_state(runtime: &mut dyn PyRuntime, mt: Mt19937) -> PyResult {
    bitgen::new_state(runtime, &BitGen::new(Engine::Mt(Box::new(mt))))
}

/// `mt19937_seed(seed)`: the key `RandomState(seed)` starts from for an integer seed.
fn mt19937_seed(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [seed] = positional(&args, "mt19937_seed")?;
    let seed = u32::try_from(index_int(runtime, &seed)?)
        .map_err(|_| PyError::value_error("Seed must be between 0 and 2**32 - 1"))?;
    runtime.charge_cpu(MT_N as u64)?;
    mt19937_state(runtime, Mt19937::from_seed(seed))
}

/// `mt19937_init_by_array(key)`: the key for an array seed, from a `uint32` array.
fn mt19937_init_by_array(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [key] = positional(&args, "mt19937_init_by_array")?;
    let key = elements::<u32>(runtime, key, DType::UINT32)?;
    runtime.charge_cpu((2 * MT_N + key.len()) as u64)?;
    mt19937_state(runtime, Mt19937::from_key_array(&key))
}

/// `mt19937_from_key(key, pos)`: an explicit 624-word `uint32` key and position.
fn mt19937_from_key(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [key, pos] = positional(&args, "mt19937_from_key")?;
    let key = elements::<u32>(runtime, key, DType::UINT32)?;
    let pos = usize::try_from(index_int(runtime, &pos)?)
        .map_err(|_| PyError::value_error("pos must be between 0 and 624"))?;
    let mt = Mt19937::from_parts(&key, pos)?;
    mt19937_state(runtime, mt)
}

fn pcg64_words(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<[u64; 4]> {
    elements::<u64>(runtime, value, DType::UINT64)?
        .try_into()
        .map_err(|_| PyError::value_error("PCG64 state needs four uint64 words"))
}

/// `pcg64_from_seed(words)`: `pcg64_set_seed` from `SeedSequence.generate_state(4, uint64)`.
fn pcg64_from_seed(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [words] = positional(&args, "pcg64_from_seed")?;
    let words = pcg64_words(runtime, words)?;
    bitgen::new_state(runtime, &BitGen::new(Engine::Pcg(Pcg64::from_words(words))))
}

/// `pcg64_from_parts(words, has_uint32, uinteger)`: the `PCG64.state` setter. `words` holds
/// the state and increment, most significant word first.
fn pcg64_from_parts(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [words, has_uint32, uinteger] = positional(&args, "pcg64_from_parts")?;
    let [state_high, state_low, inc_high, inc_low] = pcg64_words(runtime, words)?;
    let has_uint32 = flag(runtime, &has_uint32)?;
    let uinteger = u32::try_from(index_int(runtime, &uinteger)?)
        .map_err(|_| PyError::exception("OverflowError", "uinteger is out of range for uint32"))?;
    let pcg = Pcg64 {
        state: (u128::from(state_high) << 64) | u128::from(state_low),
        inc: (u128::from(inc_high) << 64) | u128::from(inc_low),
        has_uint32,
        uinteger,
    };
    bitgen::new_state(runtime, &BitGen::new(Engine::Pcg(pcg)))
}

/// `pcg64_advance(state, delta)`: `PCG64.advance` with `delta` as two `uint64` words, most
/// significant first.
fn pcg64_advance(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, delta] = positional(&args, "pcg64_advance")?;
    let delta: [u64; 2] = elements::<u64>(runtime, delta, DType::UINT64)?
        .try_into()
        .map_err(|_| PyError::value_error("delta needs two uint64 words"))?;
    let mut stream = Stream::open(runtime, state)?;
    // The jump squares a 128-bit multiplier once per bit of `delta`.
    stream.spend(128)?;
    let Engine::Pcg(pcg) = &mut stream.bits.engine else {
        return Err(PyError::type_error("advance needs a PCG64 state"));
    };
    pcg.advance((u128::from(delta[0]) << 64) | u128::from(delta[1]));
    stream.finish()?;
    Ok(Value::None)
}

/// `seed_sequence_pool(entropy, pool_size)`: `SeedSequence.mix_entropy` over the assembled
/// `uint32` entropy.
fn seed_sequence_pool(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [entropy, pool_size] = positional(&args, "seed_sequence_pool")?;
    let entropy = elements::<u32>(runtime, entropy, DType::UINT32)?;
    let pool_size = length(runtime, &pool_size)?;
    let work = pool_size.saturating_mul(pool_size.saturating_add(entropy.len()));
    runtime.charge_cpu(work as u64)?;
    array::reserve_elements(runtime, DType::UINT32, pool_size)?;
    let pool = bitgen::seed_pool(&entropy, pool_size);
    new_array(runtime, DType::UINT32, &pool)
}

/// `seed_sequence_generate(pool, n_words, wide)`: `SeedSequence.generate_state`, as `uint64`
/// words built from little-endian pairs when `wide`.
fn seed_sequence_generate(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [pool, n_words, wide] = positional(&args, "seed_sequence_generate")?;
    let pool = elements::<u32>(runtime, pool, DType::UINT32)?;
    let n_words = length(runtime, &n_words)?;
    let wide = flag(runtime, &wide)?;
    if pool.is_empty() {
        return Err(PyError::value_error("the entropy pool is empty"));
    }
    let count = if wide {
        n_words
            .checked_mul(2)
            .ok_or_else(|| PyError::value_error("too many words"))?
    } else {
        n_words
    };
    runtime.charge_cpu(count as u64)?;
    array::reserve_elements(runtime, DType::UINT32, count)?;
    let words = bitgen::seed_generate(&pool, count);
    if !wide {
        return new_array(runtime, DType::UINT32, &words);
    }
    let pairs = words
        .chunks_exact(2)
        .map(|pair| u64::from(pair[0]) | (u64::from(pair[1]) << 32))
        .collect::<Vec<_>>();
    new_array(runtime, DType::UINT64, &pairs)
}

/// `get_gauss(state)`: `RandomState`'s cached polar-method Gaussian as `(has_gauss, gauss)`.
fn get_gauss(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state] = positional(&args, "get_gauss")?;
    let (_, bits) = bitgen::load(runtime, state)?;
    runtime.new_tuple(vec![
        Value::Int(i64::from(bits.has_gauss)),
        Value::Float(bits.gauss),
    ])
}

/// `set_gauss(state, has_gauss, gauss)`: replace the cached Gaussian.
fn set_gauss(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, has_gauss, gauss] = positional(&args, "set_gauss")?;
    let has_gauss = flag(runtime, &has_gauss)?;
    let gauss = super::args::float_arg(runtime, &gauss)?;
    let (array, mut bits) = bitgen::load(runtime, state)?;
    bits.has_gauss = has_gauss;
    bits.gauss = gauss;
    bitgen::store(runtime, &array, &bits)?;
    Ok(Value::None)
}

/// Draw `count` values into a new 1-d array of `dtype`, prepaying one CPU unit per value.
fn fill<T: Element>(
    runtime: &mut dyn PyRuntime,
    state: PyValue,
    dtype: DType,
    count: usize,
    mut draw: impl FnMut(&mut Stream<'_>) -> PyResult<T>,
) -> PyResult {
    array::reserve_elements(runtime, dtype, count)?;
    let mut stream = Stream::open(runtime, state)?;
    stream.prepay(count as u64)?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(draw(&mut stream)?);
    }
    let runtime = stream.finish()?;
    new_array(runtime, dtype, &values)
}

/// `random_raw(state, count)`: `BitGenerator.random_raw` as `uint64`.
fn random_raw(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, count] = positional(&args, "random_raw")?;
    let count = length(runtime, &count)?;
    fill(runtime, state, DType::UINT64, count, |stream| {
        stream.next_raw()
    })
}

/// `standard_uniform(state, count, single)`: `random_standard_uniform_fill`, or its `_f`
/// float32 variant when `single`.
fn standard_uniform(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, count, single] = positional(&args, "standard_uniform")?;
    let count = length(runtime, &count)?;
    if flag(runtime, &single)? {
        fill(runtime, state, DType::FLOAT32, count, |stream| {
            stream.next_float()
        })
    } else {
        fill(runtime, state, DType::FLOAT64, count, |stream| {
            stream.next_double()
        })
    }
}

/// `standard_normal(state, count, single)`: `Generator`'s ziggurat normals.
fn standard_normal(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, count, single] = positional(&args, "standard_normal")?;
    let count = length(runtime, &count)?;
    if flag(runtime, &single)? {
        fill(runtime, state, DType::FLOAT32, count, ziggurat_normal_f32)
    } else {
        fill(runtime, state, DType::FLOAT64, count, ziggurat_normal)
    }
}

/// `legacy_gauss(state, count)`: `RandomState`'s polar-method normals.
fn legacy_gauss(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, count] = positional(&args, "legacy_gauss")?;
    let count = length(runtime, &count)?;
    fill(runtime, state, DType::FLOAT64, count, polar_gauss)
}

/// `standard_exponential(state, count, single, inverse)`: `Generator`'s exponentials, by the
/// ziggurat or, with `inverse`, by inversion.
fn standard_exponential(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, count, single, inverse] = positional(&args, "standard_exponential")?;
    let count = length(runtime, &count)?;
    let single = flag(runtime, &single)?;
    match (single, flag(runtime, &inverse)?) {
        (false, false) => fill(runtime, state, DType::FLOAT64, count, |stream| {
            distributions::standard_exponential(stream, Family::Generator)
        }),
        (false, true) => fill(
            runtime,
            state,
            DType::FLOAT64,
            count,
            distributions::inverse_exponential,
        ),
        (true, false) => fill(
            runtime,
            state,
            DType::FLOAT32,
            count,
            distributions::ziggurat_exponential_f32,
        ),
        (true, true) => fill(
            runtime,
            state,
            DType::FLOAT32,
            count,
            distributions::inverse_exponential_f32,
        ),
    }
}

/// `legacy_standard_exponential(state, count)`: `RandomState`'s exponentials by inversion.
fn legacy_standard_exponential(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, count] = positional(&args, "legacy_standard_exponential")?;
    let count = length(runtime, &count)?;
    fill(runtime, state, DType::FLOAT64, count, |stream| {
        distributions::standard_exponential(stream, Family::Legacy)
    })
}

fn family(runtime: &mut dyn PyRuntime, legacy: &PyValue) -> PyResult<Family> {
    Ok(if flag(runtime, legacy)? {
        Family::Legacy
    } else {
        Family::Generator
    })
}

/// A parameter array for `count` draws: one value for every draw, or one value per draw.
fn parameter<T: Element>(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    dtype: DType,
    count: usize,
) -> PyResult<Vec<T>> {
    let values = elements(runtime, value, dtype)?;
    if values.len() != 1 && values.len() != count {
        return Err(PyError::value_error(
            "a distribution parameter does not match the draw count",
        ));
    }
    Ok(values)
}

fn at<T: Copy>(values: &[T], index: usize) -> T {
    values[if values.len() == 1 { 0 } else { index }]
}

/// `continuous(state, name, legacy, count, a, b)`: `count` draws of the distribution `name` (see
/// [`Continuous`]) with `float64` parameter arrays `a` and `b`, in C order.
fn continuous(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, name, legacy, count, a, b] = positional(&args, "continuous")?;
    let name = runtime.string_value(&name)?.unwrap_or_default();
    let distribution = Continuous::from_name(&name)
        .ok_or_else(|| PyError::value_error(format!("unknown distribution {name:?}")))?;
    let family = family(runtime, &legacy)?;
    let count = length(runtime, &count)?;
    let a = parameter::<f64>(runtime, a, DType::FLOAT64, count)?;
    let b = parameter::<f64>(runtime, b, DType::FLOAT64, count)?;
    let mut index = 0;
    fill(runtime, state, DType::FLOAT64, count, |stream| {
        let value = distribution.sample(stream, family, at(&a, index), at(&b, index));
        index += 1;
        value
    })
}

/// `standard_gamma_f32(state, count, shape)`: `Generator.standard_gamma` with `dtype=float32`.
fn standard_gamma_f32(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, count, shape] = positional(&args, "standard_gamma_f32")?;
    let count = length(runtime, &count)?;
    let shape = parameter::<f32>(runtime, shape, DType::FLOAT32, count)?;
    let mut index = 0;
    fill(runtime, state, DType::FLOAT32, count, |stream| {
        let value = distributions::standard_gamma_f32(stream, at(&shape, index));
        index += 1;
        value
    })
}

/// `discrete(state, name, legacy, count, a, b)`: `count` draws of `binomial` (`a` is `p`, and
/// `b` the `int64` trial counts `n`) or `poisson` (`a` is `lam`; `b` is ignored), as `int64`.
fn discrete(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, name, legacy, count, a, b] = positional(&args, "discrete")?;
    let name = runtime.string_value(&name)?.unwrap_or_default();
    let family = family(runtime, &legacy)?;
    let count = length(runtime, &count)?;
    let a = parameter::<f64>(runtime, a, DType::FLOAT64, count)?;
    let mut index = 0;
    match name.as_str() {
        "binomial" => {
            let n = parameter::<i64>(runtime, b, DType::INT64, count)?;
            fill(runtime, state, DType::INT64, count, |stream| {
                let value = distributions::binomial(stream, family, at(&a, index), at(&n, index));
                index += 1;
                value
            })
        }
        "poisson" => fill(runtime, state, DType::INT64, count, |stream| {
            let value = distributions::poisson(stream, at(&a, index));
            index += 1;
            value
        }),
        _ => Err(PyError::value_error(format!(
            "unknown distribution {name:?}"
        ))),
    }
}

/// `random_standard_normal`: 52 bits of magnitude, a sign bit and an 8-bit layer index from
/// one 64-bit draw.
fn ziggurat_normal(stream: &mut Stream<'_>) -> PyResult<f64> {
    loop {
        let r = stream.next_u64()?;
        let idx = (r & 0xff) as usize;
        let r = r >> 8;
        let sign = r & 0x1;
        let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
        let mut x = rabs as f64 * WI_DOUBLE[idx];
        if sign & 0x1 != 0 {
            x = -x;
        }
        if rabs < KI_DOUBLE[idx] {
            return Ok(x);
        }
        if idx == 0 {
            loop {
                let xx = -NOR_INV_R * (-stream.next_double()?).ln_1p();
                let yy = -(-stream.next_double()?).ln_1p();
                if yy + yy > xx * xx {
                    return Ok(if (rabs >> 8) & 0x1 != 0 {
                        -(NOR_R + xx)
                    } else {
                        NOR_R + xx
                    });
                }
            }
        }
        if (FI_DOUBLE[idx - 1] - FI_DOUBLE[idx]) * stream.next_double()? + FI_DOUBLE[idx]
            < (-0.5 * x * x).exp()
        {
            return Ok(x);
        }
    }
}

/// `random_standard_normal_f`: the float32 ziggurat on one 32-bit draw. The final comparison
/// runs in double, as C promotes `exp(-0.5 * x * x)`.
fn ziggurat_normal_f32(stream: &mut Stream<'_>) -> PyResult<f32> {
    loop {
        let r = stream.next_u32()?;
        let idx = (r & 0xff) as usize;
        let sign = (r >> 8) & 0x1;
        let rabs = (r >> 9) & 0x007f_ffff;
        let mut x = rabs as f32 * WI_FLOAT[idx];
        if sign & 0x1 != 0 {
            x = -x;
        }
        if rabs < KI_FLOAT[idx] {
            return Ok(x);
        }
        if idx == 0 {
            loop {
                let xx = -NOR_INV_R_F * (-stream.next_float()?).ln_1p();
                let yy = -(-stream.next_float()?).ln_1p();
                if yy + yy > xx * xx {
                    return Ok(if (rabs >> 8) & 0x1 != 0 {
                        -(NOR_R_F + xx)
                    } else {
                        NOR_R_F + xx
                    });
                }
            }
        }
        let edge = (FI_FLOAT[idx - 1] - FI_FLOAT[idx]) * stream.next_float()? + FI_FLOAT[idx];
        if f64::from(edge) < (-0.5 * f64::from(x) * f64::from(x)).exp() {
            return Ok(x);
        }
    }
}

/// `legacy_gauss`: the polar method, caching the second normal of each pair in the state.
fn polar_gauss(stream: &mut Stream<'_>) -> PyResult<f64> {
    if stream.bits.has_gauss {
        let cached = stream.bits.gauss;
        stream.bits.has_gauss = false;
        stream.bits.gauss = 0.0;
        return Ok(cached);
    }
    let (x1, x2, r2) = loop {
        let x1 = 2.0 * stream.next_double()? - 1.0;
        let x2 = 2.0 * stream.next_double()? - 1.0;
        let r2 = x1 * x1 + x2 * x2;
        if r2 < 1.0 && r2 != 0.0 {
            break (x1, x2, r2);
        }
    };
    let f = (-2.0 * r2.ln() / r2).sqrt();
    stream.bits.gauss = f * x1;
    stream.bits.has_gauss = true;
    Ok(f * x2)
}

/// `gen_mask`: the smallest all-ones mask covering `max`.
fn gen_mask(max: u64) -> u64 {
    if max == 0 {
        0
    } else {
        u64::MAX >> max.leading_zeros()
    }
}

/// NumPy's 32-bit buffer (`bcnt`, `buf`) that 8-bit, 16-bit and boolean draws consume a slice
/// at a time. It lives for one fill, so leftover bits are dropped between calls as in NumPy.
#[derive(Default)]
struct Buffer {
    remaining: u32,
    word: u32,
}

impl Buffer {
    /// Refill with a fresh word after `slices` further shifts of `shift` bits.
    fn next(&mut self, stream: &mut Stream<'_>, slices: u32, shift: u32) -> PyResult<u32> {
        if self.remaining == 0 {
            self.word = stream.next_u32()?;
            self.remaining = slices;
        } else {
            self.word >>= shift;
            self.remaining -= 1;
        }
        Ok(self.word)
    }

    fn next_u16(&mut self, stream: &mut Stream<'_>) -> PyResult<u16> {
        Ok(self.next(stream, 1, 16)? as u16)
    }

    fn next_u8(&mut self, stream: &mut Stream<'_>) -> PyResult<u8> {
        Ok(self.next(stream, 3, 8)? as u8)
    }
}

/// Offset-free draw in `[0, rng]` for 64-bit output (`random_bounded_uint64` less `off`).
fn bounded_u64(stream: &mut Stream<'_>, rng: u64, masked: bool) -> PyResult<u64> {
    if let Ok(narrow) = u32::try_from(rng) {
        return bounded_u32(stream, narrow, masked).map(u64::from);
    }
    if rng == u64::MAX {
        return stream.next_u64();
    }
    if masked {
        let mask = gen_mask(rng);
        loop {
            let value = stream.next_u64()? & mask;
            if value <= rng {
                return Ok(value);
            }
        }
    }
    // Lemire's multiply-shift with rejection of the biased low products.
    let rng_excl = u128::from(rng) + 1;
    let mut m = u128::from(stream.next_u64()?) * rng_excl;
    let mut leftover = m as u64;
    if u128::from(leftover) < rng_excl {
        let threshold = ((u64::MAX - rng) as u128 % rng_excl) as u64;
        while leftover < threshold {
            m = u128::from(stream.next_u64()?) * rng_excl;
            leftover = m as u64;
        }
    }
    Ok((m >> 64) as u64)
}

fn bounded_u32(stream: &mut Stream<'_>, rng: u32, masked: bool) -> PyResult<u32> {
    if rng == 0 {
        return Ok(0);
    }
    if rng == u32::MAX {
        return stream.next_u32();
    }
    if masked {
        let mask = gen_mask(u64::from(rng)) as u32;
        loop {
            let value = stream.next_u32()? & mask;
            if value <= rng {
                return Ok(value);
            }
        }
    }
    let rng_excl = rng + 1;
    let mut m = u64::from(stream.next_u32()?) * u64::from(rng_excl);
    let mut leftover = m as u32;
    if leftover < rng_excl {
        let threshold = (u32::MAX - rng) % rng_excl;
        while leftover < threshold {
            m = u64::from(stream.next_u32()?) * u64::from(rng_excl);
            leftover = m as u32;
        }
    }
    Ok((m >> 32) as u32)
}

fn bounded_u16(
    stream: &mut Stream<'_>,
    rng: u16,
    masked: bool,
    buffer: &mut Buffer,
) -> PyResult<u16> {
    if rng == 0 {
        return Ok(0);
    }
    if rng == u16::MAX {
        return buffer.next_u16(stream);
    }
    if masked {
        let mask = gen_mask(u64::from(rng)) as u16;
        loop {
            let value = buffer.next_u16(stream)? & mask;
            if value <= rng {
                return Ok(value);
            }
        }
    }
    let rng_excl = rng + 1;
    let mut m = u32::from(buffer.next_u16(stream)?) * u32::from(rng_excl);
    let mut leftover = m as u16;
    if leftover < rng_excl {
        let threshold = (u16::MAX - rng) % rng_excl;
        while leftover < threshold {
            m = u32::from(buffer.next_u16(stream)?) * u32::from(rng_excl);
            leftover = m as u16;
        }
    }
    Ok((m >> 16) as u16)
}

fn bounded_u8(stream: &mut Stream<'_>, rng: u8, masked: bool, buffer: &mut Buffer) -> PyResult<u8> {
    if rng == 0 {
        return Ok(0);
    }
    if rng == u8::MAX {
        return buffer.next_u8(stream);
    }
    if masked {
        let mask = gen_mask(u64::from(rng)) as u8;
        loop {
            let value = buffer.next_u8(stream)? & mask;
            if value <= rng {
                return Ok(value);
            }
        }
    }
    let rng_excl = rng + 1;
    let mut m = u16::from(buffer.next_u8(stream)?) * u16::from(rng_excl);
    let mut leftover = m as u8;
    if leftover < rng_excl {
        let threshold = (u8::MAX - rng) % rng_excl;
        while leftover < threshold {
            m = u16::from(buffer.next_u8(stream)?) * u16::from(rng_excl);
            leftover = m as u8;
        }
    }
    Ok((m >> 8) as u8)
}

/// `random_interval`: masked rejection on 32-bit draws when `max` fits, used by the shuffles.
fn random_interval(stream: &mut Stream<'_>, max: u64) -> PyResult<u64> {
    if max == 0 {
        return Ok(0);
    }
    let mask = gen_mask(max);
    loop {
        let value = if max <= u64::from(u32::MAX) {
            u64::from(stream.next_u32()?) & mask
        } else {
            stream.next_u64()? & mask
        };
        if value <= max {
            return Ok(value);
        }
    }
}

/// `bounded_integers(state, off, rng, count, dtype, masked)`: `count` integers in
/// `[off, off + rng]`, wrapping in the width of `dtype`, as `random_bounded_*` computes them.
/// `off` and `rng` are `uint64` arrays of one element or `count` elements (the broadcast
/// path); `masked` picks the legacy masked rejection over Lemire's method.
fn bounded_integers(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, off, rng, count, dtype, masked] = positional(&args, "bounded_integers")?;
    let offs = elements::<u64>(runtime, off, DType::UINT64)?;
    let rngs = elements::<u64>(runtime, rng, DType::UINT64)?;
    let count = length(runtime, &count)?;
    let dtype = super::args::dtype(runtime, dtype)?;
    let masked = flag(runtime, &masked)?;
    let parameter =
        |values: &[u64], index: usize| values[if values.len() == 1 { 0 } else { index }];
    for values in [&offs, &rngs] {
        if values.len() != 1 && values.len() != count {
            return Err(PyError::value_error("bounds do not match the output size"));
        }
    }
    if count > 0 && (offs.is_empty() || rngs.is_empty()) {
        return Err(PyError::value_error("bounds do not match the output size"));
    }
    array::reserve_elements(runtime, DType::UINT64, count)?;
    array::reserve_elements(runtime, dtype, count)?;
    let mut stream = Stream::open(runtime, state)?;
    stream.prepay(count as u64)?;
    let mut buffer = Buffer::default();
    let mut raw = Vec::with_capacity(count);
    for index in 0..count {
        let (off, rng) = (parameter(&offs, index), parameter(&rngs, index));
        let value = match dtype.kind() {
            Kind::Int64 | Kind::UInt64 => off.wrapping_add(bounded_u64(&mut stream, rng, masked)?),
            Kind::Int32 | Kind::UInt32 => {
                u64::from((off as u32).wrapping_add(bounded_u32(&mut stream, rng as u32, masked)?))
            }
            Kind::Int16 | Kind::UInt16 => u64::from((off as u16).wrapping_add(bounded_u16(
                &mut stream,
                rng as u16,
                masked,
                &mut buffer,
            )?)),
            Kind::Int8 | Kind::UInt8 => u64::from((off as u8).wrapping_add(bounded_u8(
                &mut stream,
                rng as u8,
                masked,
                &mut buffer,
            )?)),
            // `buffered_bounded_bool`: one bit per draw unless the range is a single value.
            Kind::Bool if rng == 0 => off,
            Kind::Bool => u64::from(buffer.next(&mut stream, 31, 1)? & 1),
            _ => {
                return Err(PyError::type_error(format!(
                    "Unsupported dtype {} for integers",
                    dtype.repr()
                )))
            }
        };
        raw.push(value);
    }
    let runtime = stream.finish()?;
    match dtype.kind() {
        Kind::Bool => new_array(
            runtime,
            dtype,
            &raw.iter().map(|value| *value != 0).collect::<Vec<_>>(),
        ),
        Kind::Int8 => new_array(
            runtime,
            dtype,
            &raw.iter().map(|value| *value as i8).collect::<Vec<_>>(),
        ),
        Kind::UInt8 => new_array(
            runtime,
            dtype,
            &raw.iter().map(|value| *value as u8).collect::<Vec<_>>(),
        ),
        Kind::Int16 => new_array(
            runtime,
            dtype,
            &raw.iter().map(|value| *value as i16).collect::<Vec<_>>(),
        ),
        Kind::UInt16 => new_array(
            runtime,
            dtype,
            &raw.iter().map(|value| *value as u16).collect::<Vec<_>>(),
        ),
        Kind::Int32 => new_array(
            runtime,
            dtype,
            &raw.iter().map(|value| *value as i32).collect::<Vec<_>>(),
        ),
        Kind::UInt32 => new_array(
            runtime,
            dtype,
            &raw.iter().map(|value| *value as u32).collect::<Vec<_>>(),
        ),
        Kind::Int64 => new_array(
            runtime,
            dtype,
            &raw.iter().map(|value| *value as i64).collect::<Vec<_>>(),
        ),
        _ => new_array(runtime, dtype, &raw),
    }
}

/// `shuffle_indices(state, n, first, lemire)`: the order a Fisher-Yates pass over positions
/// `n - 1` down to `first` leaves `arange(n)` in. The shuffles apply it as a gather, which
/// moves elements exactly as NumPy's in-place swaps do. `lemire` selects `_shuffle_int`'s
/// draws (`Generator.choice`) over `random_interval` (`shuffle` and `permutation`).
fn shuffle_indices(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, n, first, lemire] = positional(&args, "shuffle_indices")?;
    let n = length(runtime, &n)?;
    let first = length(runtime, &first)?;
    let lemire = flag(runtime, &lemire)?;
    array::reserve_elements(runtime, DType::INT64, n)?;
    let mut stream = Stream::open(runtime, state)?;
    stream.prepay(n as u64)?;
    let mut indices = (0..n as i64).collect::<Vec<_>>();
    for i in (first..n).rev() {
        let j = if lemire {
            bounded_u64(&mut stream, i as u64, false)?
        } else {
            random_interval(&mut stream, i as u64)?
        };
        indices.swap(i, j as usize);
    }
    let runtime = stream.finish()?;
    new_array(runtime, DType::INT64, &indices)
}

/// `floyd_sample(state, pop_size, size, shuffle)`: `Generator.choice`'s Floyd sampling of
/// `size` distinct indices below `pop_size` with an open-addressing hash set, then
/// `_shuffle_int` when `shuffle`.
fn floyd_sample(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let [state, pop_size, size, shuffle] = positional(&args, "floyd_sample")?;
    let pop_size = length(runtime, &pop_size)?;
    let size = length(runtime, &size)?;
    let shuffle = flag(runtime, &shuffle)?;
    if size > pop_size {
        return Err(PyError::value_error(
            "Cannot take a larger sample than population when replace is False",
        ));
    }
    // The table holds the smallest power of two above 1.2 * size, as NumPy sizes it.
    let mask = gen_mask((1.2 * size as f64) as u64);
    let slots = usize::try_from(mask)
        .ok()
        .and_then(|mask| mask.checked_add(1))
        .ok_or_else(|| PyError::exception("MemoryError", "sample is too large"))?;
    array::reserve_elements(runtime, DType::UINT64, slots)?;
    array::reserve_elements(runtime, DType::INT64, size)?;
    let mut stream = Stream::open(runtime, state)?;
    stream.prepay(size.saturating_mul(2) as u64)?;
    let empty = u64::MAX;
    let mut table = vec![empty; slots];
    let mut sample = vec![0i64; size];
    let slot = |value: u64| (value & mask) as usize;
    for j in pop_size - size..pop_size {
        let value = bounded_u64(&mut stream, j as u64, false)?;
        let mut location = slot(value);
        while table[location] != empty && table[location] != value {
            stream.spend(1)?;
            location = (location + 1) & mask as usize;
        }
        let chosen = if table[location] == empty {
            table[location] = value;
            value
        } else {
            location = slot(j as u64);
            while table[location] != empty {
                stream.spend(1)?;
                location = (location + 1) & mask as usize;
            }
            table[location] = j as u64;
            j as u64
        };
        sample[j + size - pop_size] = chosen as i64;
    }
    if shuffle {
        for i in (1..size).rev() {
            let j = bounded_u64(&mut stream, i as u64, false)?;
            sample.swap(i, j as usize);
        }
    }
    let runtime = stream.finish()?;
    new_array(runtime, DType::INT64, &sample)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_cover_the_maximum() {
        assert_eq!(gen_mask(0), 0);
        assert_eq!(gen_mask(1), 1);
        assert_eq!(gen_mask(5), 7);
        assert_eq!(gen_mask(8), 15);
        assert_eq!(gen_mask(u64::MAX), u64::MAX);
    }
}
