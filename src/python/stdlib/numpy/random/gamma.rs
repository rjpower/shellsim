//! Gamma-family draws: `standard_gamma`, and the `chisquare`, `f`, and `standard_t` variates
//! built from it.
//!
//! For `shape >= 1` this is Marsaglia & Tsang's 2000 squeeze-and-reject method ("A Simple
//! Method for Generating Gamma Variables"): propose `d*(1+c*z)**3` from a normal deviate `z`,
//! and accept it either through the cheap `0.0331*z**4` squeeze or the full log test.
//!
//! For `shape < 1` (`a` below) this is a rejection method with a Weibull-shaped envelope, split
//! on whether the driving uniform `u` is at most or beyond `1 - a` — *not* Ahrens & Dieter's 1974
//! "algorithm GS" boosted-uniform split this module previously assumed (that formula's `p < 1`
//! vs. `p >= 1` branches, using `b = (e + a) / e`, reproduced NumPy's value only for the lower
//! ~60% of the `p < 1` domain; see `git log` on this file for that dead end and the evidence that
//! ruled it out). Recovered entirely black-box against NumPy 2.5.3, using the same PCG64
//! raw-word-forcing technique as [`super::ziggurat`]: force the first raw word to a chosen `u`,
//! read the returned value, and replay the stream to count how many further words the call
//! consumed.
//!
//! - `u <= 1 - a`: `x = u.powf(1.0 / a)`, accept if `exponential(bitgen) >= x`.
//! - `u > 1 - a`: `z = (1 - a) - a * ((1 - u) / a).ln()`, `x = z.powf(1.0 / a)`, accept if
//!   `exponential(bitgen) >= x + ((1 - u) / a).ln()`.
//!
//! On rejection the whole draw restarts from a fresh `u`. The first branch's `x` is the value
//! formula this module already had; the second branch's `x` and its acceptance test were found
//! by change of variables (checked against `docs/numpy.md`'s provenance notes): plotting
//! `value.powf(a)` against `((1 - u) / a).ln()` for real forced-`u` draws with `u > 1 - a` is
//! exactly affine (`z` above, no fitted slop — 456/456 bit-exact against forced single-attempt
//! draws for `shape = 0.4`, and 0/720 mismatched across shapes from `0.05` to `0.99` once the
//! branch threshold is `1 - a` rather than `1 / boost`). The acceptance test follows from treating
//! `x` as a Weibull(`a`)-envelope proposal for the `Gamma(a)` target: the target-to-proposal
//! density ratio's log is `x.powf(a) / a - x - (1 - a) / a`, which is uniquely maximized at `x =
//! 1` with value `0` (so no separate envelope constant is needed), giving the accept condition
//! `exponential >= x - x.powf(a) / a + (1 - a) / a`, which simplifies to the `x + ((1 - u) /
//! a).ln()` form above since `x.powf(a) == z`. Verified 0/720 mismatches jointly on the
//! accept/reject *decision* (via word-count replay) and the returned value, across the same
//! shape sweep.
//!
//! The rejection test's second draw (`exponential` below) is an actual standard-exponential
//! variate, not a fresh `next_double()`: black-box word-count evidence for `Generator` (force the
//! first raw word to a known `u`, then replay the PCG64 stream to count how many further words a
//! `standard_gamma(shape < 1)` call consumes) shows the count matches exactly the number of words
//! [`super::ziggurat::next_exponential_zig`] itself would consume for the *following* word,
//! including its rarer wedge-test and tail cases (293/293 forced core-accept draws consumed
//! exactly 2 words total; the 7/300 remaining draws whose following word was a wedge or tail case
//! under the ziggurat's own tables consumed exactly as many extra words as `next_exponential_zig`
//! would) — `Generator`'s exponential source here is `next_exponential_zig`, mirroring legacy
//! `RandomState`'s `legacy::exponential` (confirmed by the same word-count method on MT19937:
//! every legacy draw consumes a multiple of 4 raw `uint32`s, i.e. a whole number of
//! `next_double()` pairs, with no ziggurat-shaped variation).
//!
//! **Accuracy** (checked against NumPy 2.5.3, see `docs/numpy.md`): both branches now match bit
//! for bit on `Generator` and legacy `RandomState`, for `shape >= 1` and `shape < 1` alike.
//!
//! **`dtype=np.float32`.** `standard_gamma`'s own `u` draws (the `uniform` parameter below — the
//! boosted uniform in the `shape < 1` branch, and the squeeze-test uniform in the `shape >= 1`
//! branch) narrow to a single 32-bit word (`next_u32() as f64 / 2**32`) instead of the usual
//! 53-bit `next_double()`, and the `shape < 1` branch's `exponential` narrows to
//! [`super::ziggurat::next_exponential_zig_f32`], confirmed by decoding the exact raw words a real
//! `Generator.standard_gamma([0.4, 3.3], dtype=np.float32)` call consumed (only 2 raw 64-bit
//! words for both elements combined, one of them reusing a `next_u32` half-word cached from an
//! earlier call — far too few for any `next_double()`-based path) and reproducing both output
//! values bit for bit with this narrower-word model. All arithmetic past the initial draws stays
//! `f64`; only the final array narrows to `f32`, same as the rest of this crate's `dtype=float32`
//! support.
//!
//! `chisquare(df) = 2 * standard_gamma(df / 2)`, `f(dfnum, dfden)` divides two independent
//! scaled chi-squares, and `standard_t(df)` divides a normal deviate by the root-mean-square of
//! an independent chi-square — the standard constructions for these distributions in terms of
//! gamma and normal deviates, applied per output element so array parameters consume the bit
//! generator in the same per-element order NumPy's own fused loops do.

use super::bitgen::BitGen;

/// One standard gamma draw with unit scale. `normal` supplies the source of normal deviates for
/// the `shape >= 1` branch (ziggurat for `Generator`, the polar method for `RandomState`);
/// `exponential` supplies the `shape < 1` branch's rejection-test variate (ziggurat for
/// `Generator`, simple inversion for `RandomState` — see the module doc for why these must
/// differ per stream); `uniform` supplies both branches' boosting/squeeze-test uniform (plain
/// `next_double()` at `f64`, a narrower `next_u32()`-based draw at `f32` — see the module doc).
pub(in crate::python) fn standard_gamma(
    bitgen: &mut BitGen,
    shape: f64,
    normal: &mut dyn FnMut(&mut BitGen) -> f64,
    exponential: &mut dyn FnMut(&mut BitGen) -> f64,
    uniform: &mut dyn FnMut(&mut BitGen) -> f64,
) -> f64 {
    if shape < 1.0 {
        return standard_gamma_small(bitgen, shape, exponential, uniform);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let (x, v) = loop {
            let x = normal(bitgen);
            let v = 1.0 + c * x;
            if v > 0.0 {
                break (x, v * v * v);
            }
        };
        let u = uniform(bitgen);
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 {
            return d * v;
        }
        if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// A Weibull-enveloped rejection method for `0 < shape < 1` (see the module doc for the
/// derivation and the black-box evidence that ruled out the textbook Ahrens-Dieter "algorithm
/// GS" formula this module previously used here).
fn standard_gamma_small(
    bitgen: &mut BitGen,
    shape: f64,
    exponential: &mut dyn FnMut(&mut BitGen) -> f64,
    uniform: &mut dyn FnMut(&mut BitGen) -> f64,
) -> f64 {
    loop {
        let u = uniform(bitgen);
        let (x, threshold) = if u <= 1.0 - shape {
            let x = u.powf(1.0 / shape);
            (x, x)
        } else {
            let log_term = ((1.0 - u) / shape).ln();
            let z = (1.0 - shape) - shape * log_term;
            let x = z.powf(1.0 / shape);
            (x, x + log_term)
        };
        if exponential(bitgen) >= threshold {
            return x;
        }
    }
}

/// `chisquare(df)`. Always `f64`, so `standard_gamma`'s `uniform` source is plain `next_double()`.
pub(in crate::python) fn chisquare(
    bitgen: &mut BitGen,
    df: f64,
    normal: &mut dyn FnMut(&mut BitGen) -> f64,
    exponential: &mut dyn FnMut(&mut BitGen) -> f64,
) -> f64 {
    2.0 * standard_gamma(
        bitgen,
        df / 2.0,
        normal,
        exponential,
        &mut BitGen::next_double,
    )
}

/// `f(dfnum, dfden)`.
pub(in crate::python) fn f_distribution(
    bitgen: &mut BitGen,
    dfnum: f64,
    dfden: f64,
    normal: &mut dyn FnMut(&mut BitGen) -> f64,
    exponential: &mut dyn FnMut(&mut BitGen) -> f64,
) -> f64 {
    let numerator = chisquare(bitgen, dfnum, normal, exponential) * dfden;
    let denominator = chisquare(bitgen, dfden, normal, exponential) * dfnum;
    numerator / denominator
}

/// `standard_t(df)`.
pub(in crate::python) fn standard_t(
    bitgen: &mut BitGen,
    df: f64,
    normal: &mut dyn FnMut(&mut BitGen) -> f64,
    exponential: &mut dyn FnMut(&mut BitGen) -> f64,
) -> f64 {
    let n = normal(bitgen);
    let g = standard_gamma(
        bitgen,
        df / 2.0,
        normal,
        exponential,
        &mut BitGen::next_double,
    );
    // NumPy computes this as `n * sqrt(df / 2) / sqrt(g)`, not the algebraically equivalent
    // `n * sqrt(df / (2 * g))`: the two forms differ by up to 1 ULP, and only the former
    // reproduces NumPy's stream bit for bit (confirmed against real `standard_t` output for
    // several `df` values).
    n * (df / 2.0).sqrt() / g.sqrt()
}
