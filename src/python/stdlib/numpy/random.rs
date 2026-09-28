//! Native kernels behind `numpy.random`: the bulk draw loops for `Generator` (and legacy
//! `RandomState`, which is a thin Python wrapper around a `Generator`).
//!
//! `random::bitgen` holds the one bit generator (PCG64); `random::integers` holds bounded-integer
//! sampling; `random::normal` holds the normal/exponential samplers; `random::gamma` holds
//! `standard_gamma` and the distributions built on it; `random::discrete` holds binomial and
//! Poisson; `random::sequence` holds shuffling and sampling without replacement.
//!
//! The frozen `numpy/random.py` owns the object model (`PCG64`, `Generator`, `RandomState`, the
//! legacy module-level functions) and argument handling (defaults, broadcasting
//! parameter arrays to the output shape, `dtype=`/`size=` parsing, error messages). This module
//! is deliberately "dumb": every function here takes a bit generator's raw state as a plain
//! Python value, an already-shaped and already-broadcast `float64`/`int64` parameter array where
//! a distribution needs one, and returns `(result, new_state)`. Pushing shape resolution to
//! Python lets it reuse `np.broadcast_to` instead of a second
//! implementation here.
//!
//! Streams are deterministic and statistically sound but are not NumPy's streams (see
//! `numpy/random.py`'s module docstring), so every distribution uses one simple, documented
//! algorithm.
//!
//! Every function reserves the output's memory and charges CPU proportional to its size, in that
//! order, before the draw loop runs — reserving first so a too-large `size=` fails at the memory
//! limit without having spent any CPU or performed any (real, unaccounted) host allocation, and
//! charging CPU as one atomic prepayment so a too-tight CPU budget fails before any values are
//! drawn (`tests/python/resource_hardening.rs::numpy_random_reserves_and_charges_before_drawing`).

mod bitgen;
mod discrete;
mod gamma;
mod integers;
mod normal;
mod sequence;

use super::super::super::native::{
    CallArgs, FunctionDef, ModuleDef, NativeFn, PyArrayBuffer, PyError, PyResult, PyRuntime,
    PyValue, PyValueCast,
};
use super::array::{self, Array};
use super::dtype::DType;
use bitgen::Pcg64;

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
    function("_pcg_seed", pcg_seed),
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
    function("_choice_without_replacement", choice_without_replacement_fn),
];

// ---------------------------------------------------------------------------------------------
// State marshalling. A `Pcg64`'s raw state travels as a two-element Python list `[state, inc]`,
// both 128-bit integers threaded through as decimal strings (see `parse_u128`/`u128_to_value`).
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

fn bitgen_from_state(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Pcg64> {
    let list = value.cast(runtime)?;
    let items = runtime.list_items(list)?;
    if items.len() != 2 {
        return Err(PyError::runtime_error("PCG64 state must have 2 fields"));
    }
    let state = parse_u128(runtime, &items[0])?;
    let inc = parse_u128(runtime, &items[1])?;
    Ok(Pcg64 { state, inc })
}

fn state_to_value(runtime: &mut dyn PyRuntime, bitgen: Pcg64) -> PyResult<PyValue> {
    let state_value = u128_to_value(runtime, bitgen.state)?;
    let inc_value = u128_to_value(runtime, bitgen.inc)?;
    runtime.new_list(vec![state_value, inc_value])
}

fn result_and_state(runtime: &mut dyn PyRuntime, result: PyValue, bitgen: Pcg64) -> PyResult {
    let state = state_to_value(runtime, bitgen)?;
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

/// Wraps a `Vec<f64>` as either a `float64` or `float32` array, casting only at the end: NumPy's
/// own single-precision draws take narrower words per element, but shellsim does not need to
/// match its stream, and generating in `f64` and casting is simpler (see `random.py`'s module
/// docstring).
fn wrap_float(
    runtime: &mut dyn PyRuntime,
    shape: Vec<usize>,
    values: Vec<f64>,
    single: bool,
) -> PyResult<PyValue> {
    if single {
        let narrow: Vec<f32> = values.iter().map(|value| *value as f32).collect();
        wrap_f32(runtime, shape, &narrow)
    } else {
        wrap_f64(runtime, shape, &values)
    }
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

fn is_single(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<bool> {
    Ok(runtime.string_value(value)?.unwrap_or_default() == "float32")
}

fn float_dtype(single: bool) -> DType {
    if single {
        DType::FLOAT32
    } else {
        DType::FLOAT64
    }
}

// ---------------------------------------------------------------------------------------------
// Seeding.
// ---------------------------------------------------------------------------------------------

/// `_pcg_seed(words)`: seed a fresh `Pcg64` from the u64 words `random.py`'s `_seed_words`
/// expands an int, sequence of ints, or `None` into.
fn pcg_seed(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_pcg_seed", 1, 1)?;
    let positional = args.positional().to_vec();
    let list = positional[0].cast(runtime)?;
    let items = runtime.list_items(list)?;
    let mut words = Vec::with_capacity(items.len());
    for item in &items {
        words.push(parse_u128(runtime, item)? as u64);
    }
    let bitgen = Pcg64::from_seed_words(&words);
    state_to_value(runtime, bitgen)
}

// ---------------------------------------------------------------------------------------------
// Uniform and bounded-integer draws.
// ---------------------------------------------------------------------------------------------

fn uniform01_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_uniform01_fill", 3, 3)?;
    let positional = args.positional().to_vec();
    let mut bitgen = bitgen_from_state(runtime, positional[0])?;
    let shape = output_shape(runtime, positional[1])?;
    let single = is_single(runtime, &positional[2])?;
    let count = array::element_count(&shape)?;
    reserve_and_charge(runtime, float_dtype(single), count, 1)?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(bitgen.next_double());
    }
    let out = wrap_float(runtime, shape, values, single)?;
    result_and_state(runtime, out, bitgen)
}

/// `_bounded_int_fill(state, low, high_incl)`: `low` and `high_incl` are already broadcast to the
/// output shape by the Python caller. The result is always `int64`; the caller casts it down to
/// the requested dtype, since every value is already known to fit (`integers`/`randint` check
/// `low`/`high` against the dtype's bounds before calling).
fn bounded_int_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_bounded_int_fill", 3, 3)?;
    let positional = args.positional().to_vec();
    let mut bitgen = bitgen_from_state(runtime, positional[0])?;
    let (low, shape) = i64_params(runtime, positional[1])?;
    let (high_incl, _) = i64_params(runtime, positional[2])?;
    reserve_and_charge(runtime, DType::INT64, low.len(), 2)?;
    let values: Vec<i64> = low
        .iter()
        .zip(high_incl.iter())
        .map(|(lo, hi)| integers::draw_bounded(&mut bitgen, *lo, *hi))
        .collect();
    let out = wrap_i64(runtime, shape, &values)?;
    result_and_state(runtime, out, bitgen)
}

// ---------------------------------------------------------------------------------------------
// Normal and exponential.
// ---------------------------------------------------------------------------------------------

fn standard_normal_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_standard_normal_fill", 3, 3)?;
    let positional = args.positional().to_vec();
    let mut bitgen = bitgen_from_state(runtime, positional[0])?;
    let shape = output_shape(runtime, positional[1])?;
    let single = is_single(runtime, &positional[2])?;
    let count = array::element_count(&shape)?;
    reserve_and_charge(runtime, float_dtype(single), count, 4)?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(normal::next_gauss(&mut bitgen));
    }
    let out = wrap_float(runtime, shape, values, single)?;
    result_and_state(runtime, out, bitgen)
}

fn standard_exponential_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_standard_exponential_fill", 3, 3)?;
    let positional = args.positional().to_vec();
    let mut bitgen = bitgen_from_state(runtime, positional[0])?;
    let shape = output_shape(runtime, positional[1])?;
    let single = is_single(runtime, &positional[2])?;
    let count = array::element_count(&shape)?;
    reserve_and_charge(runtime, float_dtype(single), count, 2)?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(normal::next_exponential(&mut bitgen));
    }
    let out = wrap_float(runtime, shape, values, single)?;
    result_and_state(runtime, out, bitgen)
}

// ---------------------------------------------------------------------------------------------
// Gamma family.
// ---------------------------------------------------------------------------------------------

fn standard_gamma_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_standard_gamma_fill", 3, 3)?;
    let positional = args.positional().to_vec();
    let mut bitgen = bitgen_from_state(runtime, positional[0])?;
    let (shapes, out_shape) = f64_params(runtime, positional[1])?;
    let single = is_single(runtime, &positional[2])?;
    reserve_and_charge(runtime, float_dtype(single), shapes.len(), 8)?;
    let values: Vec<f64> = shapes
        .iter()
        .map(|shape| gamma::standard_gamma(&mut bitgen, *shape))
        .collect();
    let out = wrap_float(runtime, out_shape, values, single)?;
    result_and_state(runtime, out, bitgen)
}

fn chisquare_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_chisquare_fill", 2, 2)?;
    let positional = args.positional().to_vec();
    let mut bitgen = bitgen_from_state(runtime, positional[0])?;
    let (df, out_shape) = f64_params(runtime, positional[1])?;
    reserve_and_charge(runtime, DType::FLOAT64, df.len(), 8)?;
    let values: Vec<f64> = df
        .iter()
        .map(|df| gamma::chisquare(&mut bitgen, *df))
        .collect();
    let out = wrap_f64(runtime, out_shape, &values)?;
    result_and_state(runtime, out, bitgen)
}

fn f_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_f_fill", 3, 3)?;
    let positional = args.positional().to_vec();
    let mut bitgen = bitgen_from_state(runtime, positional[0])?;
    let (dfnum, out_shape) = f64_params(runtime, positional[1])?;
    let (dfden, _) = f64_params(runtime, positional[2])?;
    reserve_and_charge(runtime, DType::FLOAT64, dfnum.len(), 16)?;
    let values: Vec<f64> = dfnum
        .iter()
        .zip(dfden.iter())
        .map(|(num, den)| gamma::f_distribution(&mut bitgen, *num, *den))
        .collect();
    let out = wrap_f64(runtime, out_shape, &values)?;
    result_and_state(runtime, out, bitgen)
}

fn standard_t_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_standard_t_fill", 2, 2)?;
    let positional = args.positional().to_vec();
    let mut bitgen = bitgen_from_state(runtime, positional[0])?;
    let (df, out_shape) = f64_params(runtime, positional[1])?;
    reserve_and_charge(runtime, DType::FLOAT64, df.len(), 12)?;
    let values: Vec<f64> = df
        .iter()
        .map(|df| gamma::standard_t(&mut bitgen, *df))
        .collect();
    let out = wrap_f64(runtime, out_shape, &values)?;
    result_and_state(runtime, out, bitgen)
}

// ---------------------------------------------------------------------------------------------
// Discrete distributions.
// ---------------------------------------------------------------------------------------------

fn binomial_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_binomial_fill", 3, 3)?;
    let positional = args.positional().to_vec();
    let mut bitgen = bitgen_from_state(runtime, positional[0])?;
    let (n, out_shape) = i64_params(runtime, positional[1])?;
    let (p, _) = f64_params(runtime, positional[2])?;
    reserve_and_charge(runtime, DType::INT64, n.len(), 16)?;
    let values: Vec<i64> = n
        .iter()
        .zip(p.iter())
        .map(|(n, p)| discrete::binomial(&mut bitgen, *n, *p))
        .collect();
    let out = wrap_i64(runtime, out_shape, &values)?;
    result_and_state(runtime, out, bitgen)
}

fn poisson_fill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_poisson_fill", 2, 2)?;
    let positional = args.positional().to_vec();
    let mut bitgen = bitgen_from_state(runtime, positional[0])?;
    let (lam, out_shape) = f64_params(runtime, positional[1])?;
    reserve_and_charge(runtime, DType::INT64, lam.len(), 16)?;
    let values: Vec<i64> = lam
        .iter()
        .map(|lam| discrete::poisson(&mut bitgen, *lam))
        .collect();
    let out = wrap_i64(runtime, out_shape, &values)?;
    result_and_state(runtime, out, bitgen)
}

// ---------------------------------------------------------------------------------------------
// Sequences.
// ---------------------------------------------------------------------------------------------

fn shuffle_indices_fn(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shuffle_indices", 2, 2)?;
    let positional = args.positional().to_vec();
    let mut bitgen = bitgen_from_state(runtime, positional[0])?;
    let n = super::args::index_int(runtime, &positional[1])? as usize;
    reserve_and_charge(runtime, DType::INT64, n, 2)?;
    let mut values: Vec<i64> = (0..n as i64).collect();
    sequence::shuffle_indices(&mut bitgen, &mut values);
    let out = wrap_i64(runtime, vec![n], &values)?;
    result_and_state(runtime, out, bitgen)
}

/// `_choice_without_replacement(state, n, size, shuffle)`: `Generator.choice(..., replace=False,
/// p=None)`'s own index draws (see `sequence::choice_without_replacement`'s doc). Floyd's
/// algorithm runs in `O(size)`, so memory and CPU are reserved for `size`, not the population `n`.
fn choice_without_replacement_fn(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_choice_without_replacement", 4, 4)?;
    let positional = args.positional().to_vec();
    let mut bitgen = bitgen_from_state(runtime, positional[0])?;
    let n = super::args::index_int(runtime, &positional[1])? as usize;
    let size = super::args::index_int(runtime, &positional[2])? as usize;
    let shuffle = runtime.truth(&positional[3])?;
    reserve_and_charge(runtime, DType::INT64, size, 2)?;
    let values = sequence::choice_without_replacement(&mut bitgen, n, size, shuffle);
    let out = wrap_i64(runtime, vec![size], &values)?;
    result_and_state(runtime, out, bitgen)
}
