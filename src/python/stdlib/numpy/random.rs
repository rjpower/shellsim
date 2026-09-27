//! Native kernels behind `numpy.random`: the bulk draw loops for `Generator` and legacy
//! `RandomState`.
//!
//! `random::bitgen` holds the two bit generators (MT19937, PCG64); `random::integers` holds
//! bounded-integer sampling; `random::ziggurat` holds the normal/exponential ziggurat;
//! `random::gamma` holds `standard_gamma` and the distributions built on it; `random::discrete`
//! holds binomial and Poisson; `random::legacy` holds legacy `RandomState`'s own Gaussian;
//! `random::sequence` holds shuffling and sampling without replacement.
//!
//! The frozen `numpy/random.py` owns the object model (`SeedSequence`, `MT19937`, `PCG64`,
//! `Generator`, `RandomState`, the legacy module-level functions) and NumPy's own argument
//! handling (defaults, broadcasting parameter arrays to the output shape, `dtype=`/`size=`
//! parsing, error messages). This module is deliberately "dumb": every function here takes a
//! bit generator's raw state as plain Python values, an already-shaped and already-broadcast
//! `float64`/`int64` parameter array where a distribution needs one, and returns `(result,
//! new_state)`. Pushing shape resolution to Python lets it reuse NumPy's own broadcasting
//! (`np.broadcast_to`) instead of a second implementation here.
//!
//! Every function reserves the output's memory and charges CPU proportional to its size, in
//! that order, before the draw loop runs — reserving first so a too-large `size=` fails at the
//! memory limit without having spent any CPU or performed any (real, unaccounted) host
//! allocation, and charging CPU as one atomic prepayment so a too-tight CPU budget fails before
//! any values are drawn (`tests/python/resource_hardening.rs::
//! numpy_random_reserves_and_charges_before_drawing`).

mod bitgen;
mod discrete;
mod gamma;
mod integers;
mod legacy;
mod sequence;
mod ziggurat;

use super::super::super::native::{
    CallArgs, FunctionDef, ModuleDef, NativeFn, PyArrayBuffer, PyError, PyResult, PyRuntime,
    PyValue, PyValueCast,
};
use super::super::super::Value;
use super::array::{self, Array};
use super::dtype::DType;
use bitgen::{BitGen, MT_N};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_random",
    functions: FUNCTIONS,
    values: &[],
};

const fn function(name: &'static str, call: NativeFn) -> FunctionDef {
    FunctionDef {
        module: "numpy",
        name,
        call,
    }
}

static FUNCTIONS: &[FunctionDef] = &[
    function("_mt_seed_genrand", mt_seed_genrand),
    function("_mt_seed_array", mt_seed_array),
    function("_pcg_seed", pcg_seed),
    function("_pcg_advance", pcg_advance),
    function("_raw_fill", raw_fill),
    function("_uniform01_fill", uniform01_fill),
    function("_bounded_int_fill", bounded_int_fill),
    function("_standard_normal_fill", standard_normal_fill),
    function("_standard_exponential_fill", standard_exponential_fill),
    function("_standard_gamma_fill", standard_gamma_fill),
    function("_chisquare_fill", chisquare_fill),
    function("_f_fill", f_fill),
    function("_standard_t_fill", standard_t_fill),
    function("_binomial_fill", binomial_fill),
    function("_poisson_fill", poisson_fill),
    function("_shuffle_indices", shuffle_indices_fn),
    function("_sample_without_replacement", sample_without_replacement_fn),
    function("_choice_without_replacement", choice_without_replacement_fn),
];

// ---------------------------------------------------------------------------------------------
// State marshalling. A bit generator's raw state travels as a small Python list:
//   MT19937 (Generator):    ["MT19937", key: ndarray(624, uint32), pos: int]
//   MT19937 (RandomState):  ["MT19937", key, pos, has_gauss: bool, cached_gaussian: float]
//   PCG64 (Generator only): ["PCG64", state: int, inc: int, has_uint32: bool, uinteger: int]
// ---------------------------------------------------------------------------------------------

fn parse_u128(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<u128> {
    let text = runtime
        .integer_text(value)?
        .ok_or_else(|| PyError::type_error("expected an integer"))?;
    text.parse::<u128>()
        .map_err(|_| PyError::value_error("integer out of range for a bit generator state"))
}

fn u128_to_value(runtime: &mut dyn PyRuntime, value: u128) -> PyResult<PyValue> {
    runtime.new_integer(&value.to_string())
}

fn state_items(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<PyValue>> {
    let list = value.cast(runtime)?;
    runtime.list_items(list)
}

fn bitgen_from_items(
    runtime: &mut dyn PyRuntime,
    items: &[PyValue],
) -> PyResult<(BitGen, Option<(bool, f64)>)> {
    let kind = runtime
        .string_value(&items[0])?
        .ok_or_else(|| PyError::runtime_error("bit generator state is missing its kind"))?;
    match kind.as_str() {
        "MT19937" => {
            let array = Array::from_value(runtime, items[1])?;
            let key_vec = array::read_elements::<u32>(runtime, &array)?;
            if key_vec.len() != MT_N {
                return Err(PyError::runtime_error("MT19937 key must have 624 words"));
            }
            let mut key = Box::new([0u32; MT_N]);
            key.copy_from_slice(&key_vec);
            let pos = runtime
                .int_value(&items[2])
                .ok_or_else(|| PyError::type_error("expected an integer position"))?
                as usize;
            let gauss = if items.len() > 3 {
                let has_gauss = runtime.truth(&items[3])?;
                let cached = super::args::float_arg(runtime, &items[4])?;
                Some((has_gauss, cached))
            } else {
                None
            };
            Ok((BitGen::Mt19937 { key, pos }, gauss))
        }
        "PCG64" => {
            let state = parse_u128(runtime, &items[1])?;
            let inc = parse_u128(runtime, &items[2])?;
            let has_uint32 = runtime.truth(&items[3])?;
            let uinteger = runtime.int_value(&items[4]).unwrap_or(0) as u32;
            Ok((
                BitGen::Pcg64 {
                    state,
                    inc,
                    has_uint32,
                    uinteger,
                },
                None,
            ))
        }
        _ => Err(PyError::runtime_error("unknown bit generator kind")),
    }
}

fn bitgen_to_value(
    runtime: &mut dyn PyRuntime,
    bitgen: BitGen,
    gauss: Option<(bool, f64)>,
) -> PyResult<PyValue> {
    match bitgen {
        BitGen::Mt19937 { key, pos } => {
            array::reserve_elements(runtime, DType::UINT32, MT_N)?;
            runtime.charge_cpu(MT_N as u64 / 8 + 1)?;
            let mut bytes = Vec::with_capacity(MT_N * 4);
            for word in key.iter() {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            let array = array::new_array(
                runtime,
                PyArrayBuffer::Bytes(bytes),
                DType::UINT32,
                vec![MT_N],
            )?
            .value();
            let kind = runtime.new_string("MT19937".to_string())?;
            let mut items = vec![kind, array, Value::Int(pos as i64)];
            if let Some((has_gauss, cached)) = gauss {
                items.push(Value::Bool(has_gauss));
                items.push(Value::Float(cached));
            }
            runtime.new_list(items)
        }
        BitGen::Pcg64 {
            state,
            inc,
            has_uint32,
            uinteger,
        } => {
            let state_value = u128_to_value(runtime, state)?;
            let inc_value = u128_to_value(runtime, inc)?;
            let kind = runtime.new_string("PCG64".to_string())?;
            let items = vec![
                kind,
                state_value,
                inc_value,
                Value::Bool(has_uint32),
                Value::Int(i64::from(uinteger)),
            ];
            runtime.new_list(items)
        }
    }
}

fn result_and_state(
    runtime: &mut dyn PyRuntime,
    result: PyValue,
    bitgen: BitGen,
    gauss: Option<(bool, f64)>,
) -> PyResult {
    let state = bitgen_to_value(runtime, bitgen, gauss)?;
    runtime.new_tuple(vec![result, state])
}

// ---------------------------------------------------------------------------------------------
// Resource accounting and output construction. `reserve_and_charge` must run before any
// `Vec::with_capacity`-sized buffer for the output is built, not just before the draw loop:
// building that buffer is itself the "real, unbounded host work" the reservation guards.
// ---------------------------------------------------------------------------------------------

fn reserve_and_charge(
    runtime: &mut dyn PyRuntime,
    dtype: DType,
    count: usize,
    per_element: u64,
) -> PyResult<()> {
    array::reserve_elements(runtime, dtype, count)?;
    runtime.charge_cpu((count as u64).saturating_mul(per_element) + 1)
}

fn output_shape(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Vec<usize>> {
    super::args::shape(runtime, value)
}

fn wrap_f64(runtime: &mut dyn PyRuntime, shape: Vec<usize>, values: &[f64]) -> PyResult<PyValue> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(array::new_array(runtime, PyArrayBuffer::Bytes(bytes), DType::FLOAT64, shape)?.value())
}

fn wrap_f32(runtime: &mut dyn PyRuntime, shape: Vec<usize>, values: &[f32]) -> PyResult<PyValue> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(array::new_array(runtime, PyArrayBuffer::Bytes(bytes), DType::FLOAT32, shape)?.value())
}

fn wrap_i64(runtime: &mut dyn PyRuntime, shape: Vec<usize>, values: &[i64]) -> PyResult<PyValue> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(array::new_array(runtime, PyArrayBuffer::Bytes(bytes), DType::INT64, shape)?.value())
}

fn wrap_u64(runtime: &mut dyn PyRuntime, shape: Vec<usize>, values: &[u64]) -> PyResult<PyValue> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(array::new_array(runtime, PyArrayBuffer::Bytes(bytes), DType::UINT64, shape)?.value())
}

fn f64_params(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let array = Array::from_value(runtime, value)?;
    let values = array::read_elements::<f64>(runtime, &array)?;
    let shape = array.shape().to_vec();
    Ok((values, shape))
}

fn i64_params(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<(Vec<i64>, Vec<usize>)> {
    let array = Array::from_value(runtime, value)?;
    let values = array::read_elements::<i64>(runtime, &array)?;
    let shape = array.shape().to_vec();
    Ok((values, shape))
}

// ---------------------------------------------------------------------------------------------
// Seeding.
// ---------------------------------------------------------------------------------------------

/// `_mt_seed_genrand(seed)`: classic single-word MT19937 seeding for a scalar legacy seed.
fn mt_seed_genrand(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_mt_seed_genrand", 1, 1)?;
    let positional = args.positional();
    let seed = super::args::index_int(runtime, &positional[0])? as u32;
    let key = bitgen::mt_init_genrand(seed);
    bitgen_to_value(runtime, BitGen::Mt19937 { key, pos: MT_N }, None)
}

/// `_mt_seed_array(key)`: `init_by_array` for a legacy array-like seed.
fn mt_seed_array(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_mt_seed_array", 1, 1)?;
    let positional = args.positional();
    let (key, _) = i64_params(runtime, positional[0])?;
    let key: Vec<u32> = key.into_iter().map(|value| value as u32).collect();
    let state = bitgen::mt_init_by_array(&key);
    bitgen_to_value(
        runtime,
        BitGen::Mt19937 {
            key: state,
            pos: MT_N,
        },
        None,
    )
}

/// `_pcg_seed(w0, w1, w2, w3)`: PCG64's `pcg_setseq_128_srandom_r` from four `SeedSequence`
/// words (`w0, w1` form the 128-bit initial state, high word first; `w2, w3` the sequence).
fn pcg_seed(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_pcg_seed", 4, 4)?;
    let positional = args.positional().to_vec();
    let mut words = Vec::with_capacity(4);
    for value in &positional {
        words.push(parse_u128(runtime, value)? as u64);
    }
    let initstate = (u128::from(words[0]) << 64) | u128::from(words[1]);
    let initseq = (u128::from(words[2]) << 64) | u128::from(words[3]);
    let (state, inc) = bitgen::pcg_seed(initstate, initseq);
    bitgen_to_value(
        runtime,
        BitGen::Pcg64 {
            state,
            inc,
            has_uint32: false,
            uinteger: 0,
        },
        None,
    )
}

fn pcg_advance(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_pcg_advance", 2, 2)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (bitgen, _) = bitgen_from_items(runtime, &items)?;
    let BitGen::Pcg64 { state, inc, .. } = bitgen else {
        return Err(PyError::runtime_error("advance() needs a PCG64 state"));
    };
    let delta = parse_u128(runtime, &positional[1])?;
    let state = bitgen::pcg_advance(state, inc, delta);
    // Advancing discards the buffered half of a 64-bit word, as NumPy does.
    let bitgen = BitGen::Pcg64 {
        state,
        inc,
        has_uint32: false,
        uinteger: 0,
    };
    bitgen_to_value(runtime, bitgen, None)
}

// ---------------------------------------------------------------------------------------------
// Uniform draws.
// ---------------------------------------------------------------------------------------------

fn raw_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_raw_fill", 2, 2)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, gauss) = bitgen_from_items(runtime, &items)?;
    let n = super::args::index_int(runtime, &positional[1])? as usize;
    reserve_and_charge(runtime, DType::UINT64, n, 1)?;
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        values.push(bitgen.next_raw());
    }
    let out = wrap_u64(runtime, vec![n], &values)?;
    result_and_state(runtime, out, bitgen, gauss)
}

fn uniform01_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_uniform01_fill", 3, 3)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, gauss) = bitgen_from_items(runtime, &items)?;
    let shape = output_shape(runtime, positional[1])?;
    let single = runtime.string_value(&positional[2])?.unwrap_or_default() == "float32";
    let count = array::element_count(&shape)?;
    let out = if single {
        reserve_and_charge(runtime, DType::FLOAT32, count, 1)?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(bitgen.next_f32());
        }
        wrap_f32(runtime, shape, &values)?
    } else {
        reserve_and_charge(runtime, DType::FLOAT64, count, 1)?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(bitgen.next_double());
        }
        wrap_f64(runtime, shape, &values)?
    };
    result_and_state(runtime, out, bitgen, gauss)
}

/// `_bounded_int_fill(state, low, count, legacy, dtype_name)`: `low` and `count` are already
/// broadcast to the output shape by the Python caller (`count[i] = high_incl[i] - low[i] + 1`,
/// computed as a Python `float`, which is exact for every range this module supports).
/// `dtype_name` selects the output dtype's own raw-word buffering (see
/// `integers::NarrowBuffer`'s doc) for dtypes narrower than 32 bits; other dtypes draw a full raw
/// word per element, unbuffered, as before.
fn bounded_int_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_bounded_int_fill", 5, 5)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, gauss) = bitgen_from_items(runtime, &items)?;
    let (low, shape) = i64_params(runtime, positional[1])?;
    let (count, _) = f64_params(runtime, positional[2])?;
    let legacy = runtime.truth(&positional[3])?;
    let dtype_name = runtime.string_value(&positional[4])?.unwrap_or_default();
    reserve_and_charge(runtime, DType::INT64, low.len(), 2)?;
    let mut values = Vec::with_capacity(low.len());
    if let Some(chunk_bits) = integers::narrow_chunk_bits(&dtype_name) {
        let mut buffer = integers::NarrowBuffer::new();
        for (lo, count) in low.iter().zip(count.iter()) {
            let range_incl = (*count as u128 - 1) as u32;
            let offset = integers::draw_bounded_buffered(
                &mut bitgen,
                &mut buffer,
                range_incl,
                legacy,
                chunk_bits,
            );
            values.push(lo.wrapping_add(i64::from(offset)));
        }
    } else {
        for (lo, count) in low.iter().zip(count.iter()) {
            values.push(integers::draw_bounded(
                &mut bitgen,
                *lo,
                *count as u128,
                legacy,
            ));
        }
    }
    let out = wrap_i64(runtime, shape, &values)?;
    result_and_state(runtime, out, bitgen, gauss)
}

// ---------------------------------------------------------------------------------------------
// Normal and exponential.
// ---------------------------------------------------------------------------------------------

fn is_legacy_gauss(gauss: &Option<(bool, f64)>) -> bool {
    gauss.is_some()
}

fn normal_source(gauss: &mut Option<(bool, f64)>) -> Box<dyn FnMut(&mut BitGen) -> f64 + '_> {
    match gauss {
        Some((has_gauss, cached)) => {
            Box::new(move |bitgen: &mut BitGen| legacy::gauss(bitgen, has_gauss, cached))
        }
        None => Box::new(ziggurat::next_gauss),
    }
}

/// `dtype=np.float32`'s own source: `Generator`'s ziggurat has a narrower, 32-bit-word variant
/// (see `ziggurat::next_gauss_f32`'s doc); legacy `RandomState`'s polar method has no narrower
/// form (NumPy's own legacy Gaussian is always double precision), so it reuses `normal_source`
/// and only the *result* narrows to `f32`.
fn normal_source_f32(gauss: &mut Option<(bool, f64)>) -> Box<dyn FnMut(&mut BitGen) -> f64 + '_> {
    match gauss {
        Some((has_gauss, cached)) => {
            Box::new(move |bitgen: &mut BitGen| legacy::gauss(bitgen, has_gauss, cached))
        }
        None => Box::new(ziggurat::next_gauss_f32),
    }
}

/// `gamma::standard_gamma`'s exponential source for its `shape < 1` branch (see that module's
/// doc): `Generator` draws it via the ziggurat, matching `standard_exponential`'s default method;
/// legacy `RandomState` uses simple inversion, matching its own `standard_exponential`.
fn exponential_source(gauss: &Option<(bool, f64)>) -> Box<dyn FnMut(&mut BitGen) -> f64> {
    if is_legacy_gauss(gauss) {
        Box::new(legacy::exponential)
    } else {
        Box::new(ziggurat::next_exponential_zig)
    }
}

/// `dtype=np.float32`'s own exponential source for `standard_gamma`'s `shape < 1` branch: see
/// `ziggurat::next_exponential_zig_f32`'s doc for `Generator`'s narrower word; legacy
/// `RandomState` has no narrower form, so it reuses `exponential_source`.
fn exponential_source_f32(gauss: &Option<(bool, f64)>) -> Box<dyn FnMut(&mut BitGen) -> f64> {
    if is_legacy_gauss(gauss) {
        Box::new(legacy::exponential)
    } else {
        Box::new(ziggurat::next_exponential_zig_f32)
    }
}

/// `dtype=np.float32`'s own uniform source for `standard_gamma` (see `gamma.rs`'s module doc): a
/// single `next_u32()` word divided down, instead of the usual 53-bit `next_double()`.
fn uniform_source_f32() -> Box<dyn FnMut(&mut BitGen) -> f64> {
    Box::new(|bitgen: &mut BitGen| f64::from(bitgen.next_u32()) / 4_294_967_296.0)
}

fn standard_normal_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_standard_normal_fill", 3, 3)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, mut gauss) = bitgen_from_items(runtime, &items)?;
    let shape = output_shape(runtime, positional[1])?;
    let single = runtime.string_value(&positional[2])?.unwrap_or_default() == "float32";
    let count = array::element_count(&shape)?;
    let out = if single {
        reserve_and_charge(runtime, DType::FLOAT32, count, 4)?;
        let mut values = Vec::with_capacity(count);
        {
            let mut source = normal_source_f32(&mut gauss);
            for _ in 0..count {
                values.push(source(&mut bitgen) as f32);
            }
        }
        wrap_f32(runtime, shape, &values)?
    } else {
        reserve_and_charge(runtime, DType::FLOAT64, count, 4)?;
        let mut values = Vec::with_capacity(count);
        {
            let mut source = normal_source(&mut gauss);
            for _ in 0..count {
                values.push(source(&mut bitgen));
            }
        }
        wrap_f64(runtime, shape, &values)?
    };
    result_and_state(runtime, out, bitgen, gauss)
}

fn standard_exponential_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_standard_exponential_fill", 4, 4)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, gauss) = bitgen_from_items(runtime, &items)?;
    let shape = output_shape(runtime, positional[1])?;
    let single = runtime.string_value(&positional[2])?.unwrap_or_default() == "float32";
    let inversion = runtime.string_value(&positional[3])?.unwrap_or_default() == "inv";
    let legacy = is_legacy_gauss(&gauss);
    let count = array::element_count(&shape)?;
    let draw = |bitgen: &mut BitGen| -> f64 {
        if legacy {
            legacy::exponential(bitgen)
        } else if inversion {
            ziggurat::next_exponential_inv(bitgen)
        } else {
            ziggurat::next_exponential_zig(bitgen)
        }
    };
    let out = if single {
        reserve_and_charge(runtime, DType::FLOAT32, count, 2)?;
        let mut values = Vec::with_capacity(count);
        // `method="zig"` has its own narrower-word `f32` path (see
        // `ziggurat::next_exponential_zig_f32`'s doc); inversion and the legacy generator have
        // no narrower form (both are a single `next_double()`-driven formula), so they reuse
        // `draw` and only the *result* narrows to `f32`.
        if !legacy && !inversion {
            for _ in 0..count {
                values.push(ziggurat::next_exponential_zig_f32(&mut bitgen) as f32);
            }
        } else {
            for _ in 0..count {
                values.push(draw(&mut bitgen) as f32);
            }
        }
        wrap_f32(runtime, shape, &values)?
    } else {
        reserve_and_charge(runtime, DType::FLOAT64, count, 2)?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(draw(&mut bitgen));
        }
        wrap_f64(runtime, shape, &values)?
    };
    result_and_state(runtime, out, bitgen, gauss)
}

// ---------------------------------------------------------------------------------------------
// Gamma family.
// ---------------------------------------------------------------------------------------------

fn standard_gamma_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_standard_gamma_fill", 3, 3)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, mut gauss) = bitgen_from_items(runtime, &items)?;
    let (shapes, out_shape) = f64_params(runtime, positional[1])?;
    let single = runtime.string_value(&positional[2])?.unwrap_or_default() == "float32";
    let dtype = if single {
        DType::FLOAT32
    } else {
        DType::FLOAT64
    };
    reserve_and_charge(runtime, dtype, shapes.len(), 8)?;
    let mut values = Vec::with_capacity(shapes.len());
    {
        // `shape >= 1` draws normal deviates behind the scenes (Marsaglia-Tsang); matching
        // `standard_normal_fill`'s own dtype split keeps that source's word width consistent
        // with what a bare `standard_normal(dtype=...)` call would have consumed. `shape < 1`
        // and the squeeze test both narrow their own uniform draws at `f32` too (see `gamma.rs`).
        let mut exponential = if single {
            exponential_source_f32(&gauss)
        } else {
            exponential_source(&gauss)
        };
        let mut source = if single {
            normal_source_f32(&mut gauss)
        } else {
            normal_source(&mut gauss)
        };
        let mut uniform: Box<dyn FnMut(&mut BitGen) -> f64> = if single {
            uniform_source_f32()
        } else {
            Box::new(BitGen::next_double)
        };
        for shape in &shapes {
            values.push(gamma::standard_gamma(
                &mut bitgen,
                *shape,
                &mut *source,
                &mut *exponential,
                &mut *uniform,
            ));
        }
    }
    let out = if single {
        let values: Vec<f32> = values.iter().map(|value| *value as f32).collect();
        wrap_f32(runtime, out_shape, &values)?
    } else {
        wrap_f64(runtime, out_shape, &values)?
    };
    result_and_state(runtime, out, bitgen, gauss)
}

fn chisquare_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_chisquare_fill", 2, 2)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, mut gauss) = bitgen_from_items(runtime, &items)?;
    let (df, out_shape) = f64_params(runtime, positional[1])?;
    reserve_and_charge(runtime, DType::FLOAT64, df.len(), 8)?;
    let values: Vec<f64> = {
        let mut exponential = exponential_source(&gauss);
        let mut source = normal_source(&mut gauss);
        df.iter()
            .map(|df| gamma::chisquare(&mut bitgen, *df, &mut *source, &mut *exponential))
            .collect()
    };
    let out = wrap_f64(runtime, out_shape, &values)?;
    result_and_state(runtime, out, bitgen, gauss)
}

fn f_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_f_fill", 3, 3)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, mut gauss) = bitgen_from_items(runtime, &items)?;
    let (dfnum, out_shape) = f64_params(runtime, positional[1])?;
    let (dfden, _) = f64_params(runtime, positional[2])?;
    reserve_and_charge(runtime, DType::FLOAT64, dfnum.len(), 16)?;
    let values: Vec<f64> = {
        let mut exponential = exponential_source(&gauss);
        let mut source = normal_source(&mut gauss);
        dfnum
            .iter()
            .zip(dfden.iter())
            .map(|(num, den)| {
                gamma::f_distribution(&mut bitgen, *num, *den, &mut *source, &mut *exponential)
            })
            .collect()
    };
    let out = wrap_f64(runtime, out_shape, &values)?;
    result_and_state(runtime, out, bitgen, gauss)
}

fn standard_t_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_standard_t_fill", 2, 2)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, mut gauss) = bitgen_from_items(runtime, &items)?;
    let (df, out_shape) = f64_params(runtime, positional[1])?;
    reserve_and_charge(runtime, DType::FLOAT64, df.len(), 12)?;
    let values: Vec<f64> = {
        let mut exponential = exponential_source(&gauss);
        let mut source = normal_source(&mut gauss);
        df.iter()
            .map(|df| gamma::standard_t(&mut bitgen, *df, &mut *source, &mut *exponential))
            .collect()
    };
    let out = wrap_f64(runtime, out_shape, &values)?;
    result_and_state(runtime, out, bitgen, gauss)
}

// ---------------------------------------------------------------------------------------------
// Discrete distributions.
// ---------------------------------------------------------------------------------------------

fn binomial_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_binomial_fill", 3, 3)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, gauss) = bitgen_from_items(runtime, &items)?;
    let (n, out_shape) = i64_params(runtime, positional[1])?;
    let (p, _) = f64_params(runtime, positional[2])?;
    let legacy = is_legacy_gauss(&gauss);
    reserve_and_charge(runtime, DType::INT64, n.len(), 16)?;
    let values: Vec<i64> = n
        .iter()
        .zip(p.iter())
        .map(|(n, p)| discrete::binomial(&mut bitgen, *n, *p, legacy))
        .collect();
    let out = wrap_i64(runtime, out_shape, &values)?;
    result_and_state(runtime, out, bitgen, gauss)
}

fn poisson_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_poisson_fill", 2, 2)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, gauss) = bitgen_from_items(runtime, &items)?;
    let (lam, out_shape) = f64_params(runtime, positional[1])?;
    reserve_and_charge(runtime, DType::INT64, lam.len(), 16)?;
    let values: Vec<i64> = lam
        .iter()
        .map(|lam| discrete::poisson(&mut bitgen, *lam))
        .collect();
    let out = wrap_i64(runtime, out_shape, &values)?;
    result_and_state(runtime, out, bitgen, gauss)
}

// ---------------------------------------------------------------------------------------------
// Sequences.
// ---------------------------------------------------------------------------------------------

fn shuffle_indices_fn(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shuffle_indices", 2, 2)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, gauss) = bitgen_from_items(runtime, &items)?;
    let n = super::args::index_int(runtime, &positional[1])? as usize;
    reserve_and_charge(runtime, DType::INT64, n, 2)?;
    let mut values: Vec<i64> = (0..n as i64).collect();
    sequence::shuffle_indices(&mut bitgen, &mut values);
    let out = wrap_i64(runtime, vec![n], &values)?;
    result_and_state(runtime, out, bitgen, gauss)
}

/// `_sample_without_replacement(state, n, size)`: draws `size` distinct indices from `[0, n)`.
/// The underlying algorithm (a partial Fisher-Yates shuffle) builds an `n`-length working array
/// regardless of `size`, so memory is reserved for `n`, not just the `size`-length result.
fn sample_without_replacement_fn(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_sample_without_replacement", 3, 3)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, gauss) = bitgen_from_items(runtime, &items)?;
    let n = super::args::index_int(runtime, &positional[1])? as usize;
    let size = super::args::index_int(runtime, &positional[2])? as usize;
    reserve_and_charge(runtime, DType::INT64, n, 2)?;
    let values = sequence::sample_without_replacement(&mut bitgen, n, size);
    let out = wrap_i64(runtime, vec![size], &values)?;
    result_and_state(runtime, out, bitgen, gauss)
}

/// `_choice_without_replacement(state, n, size, shuffle)`: `Generator.choice(..., replace=False,
/// p=None)`'s own index draws (see `sequence::choice_without_replacement`'s doc for the two
/// algorithms NumPy switches between). The large-population algorithm builds an `n`-length
/// working array regardless of `size`, so memory is reserved for `n`, not just the
/// `size`-length result, matching `sample_without_replacement_fn` above.
fn choice_without_replacement_fn(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_choice_without_replacement", 4, 4)?;
    let positional = args.positional().to_vec();
    let items = state_items(runtime, positional[0])?;
    let (mut bitgen, gauss) = bitgen_from_items(runtime, &items)?;
    let n = super::args::index_int(runtime, &positional[1])? as usize;
    let size = super::args::index_int(runtime, &positional[2])? as usize;
    let shuffle = runtime.truth(&positional[3])?;
    reserve_and_charge(runtime, DType::INT64, n, 2)?;
    let values = sequence::choice_without_replacement(&mut bitgen, n, size, shuffle);
    let out = wrap_i64(runtime, vec![size], &values)?;
    result_and_state(runtime, out, bitgen, gauss)
}
