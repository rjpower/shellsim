//! Gamma-family draws: `standard_gamma`, and the `chisquare`, `f`, and `standard_t` variates
//! built from it.
//!
//! For `shape >= 1` this is Marsaglia & Tsang's 2000 squeeze-and-reject method ("A Simple
//! Method for Generating Gamma Variables"): propose `d*(1+c*z)**3` from a normal deviate `z`,
//! and accept it either through the cheap `0.0331*z**4` squeeze or the full log test. For
//! `shape < 1` this is (an incompletely recovered variant of) Ahrens & Dieter's 1974 algorithm GS
//! ("Computer Methods for Sampling from Gamma, Beta, Poisson and Normal Distributions"): boost
//! through `u**(1/shape)` or the logarithmic branch depending on which side of `1` the scaled
//! uniform falls on.
//!
//! **Accuracy** (checked against NumPy 2.5.3, see `docs/numpy.md`): the `shape >= 1` branch
//! matches bit for bit for both legacy `RandomState` (whose Gaussian is the polar method in
//! [`super::legacy::gauss`]) and `Generator` (whose ziggurat normal, [`super::ziggurat`], now
//! reproduces NumPy's stream).
//!
//! The `shape < 1` branch's rejection test draws its second uniform (called `exponential` below)
//! from an actual standard-exponential variate, not a fresh `next_double()`: black-box word-count
//! evidence for `Generator` (force the first raw word to a known `u`, then replay the PCG64
//! stream to count how many further words a `standard_gamma(shape<1)` call consumes) shows the
//! count matches exactly the number of words [`super::ziggurat::next_exponential_zig`] itself
//! would consume for the *following* word, including its rarer wedge-test and tail cases (293/293
//! forced core-accept draws consumed exactly 2 words total; the 7/300 remaining draws whose
//! following word was a wedge or tail case under the ziggurat's own tables consumed exactly as
//! many extra words as `next_exponential_zig` would) — `Generator`'s exponential source here is
//! `next_exponential_zig`, mirroring legacy `RandomState`'s `legacy::exponential` (confirmed by
//! the same word-count method on MT19937: every legacy draw consumes a multiple of 4 raw
//! `uint32`s, i.e. a whole number of `next_double()` pairs, with no ziggurat-shaped variation).
//!
//! With that source wired in, the *common* case (`u * boost < 1` and `u` itself not too close to
//! the branch threshold `1 / boost`) matches bit for bit on both streams: `x = u.powf(1.0 /
//! shape)` reproduces NumPy's value exactly (checked against 1108 forced-`u` draws for
//! `shape = 0.4`, u up to roughly 0.6 of the way to the threshold). Beyond that point — the rest
//! of the `u * boost < 1` domain approaching the threshold, and all of the `u * boost >= 1`
//! branch — NumPy's actual returned value diverges smoothly from `u.powf(1.0 / shape)` in a way
//! this module could not identify from bit patterns alone: the *number* of raw words consumed
//! still matches the model above exactly (so the branch choice and the exponential source are
//! right), but the returned value does not equal `x` as computed here, with the discrepancy
//! growing continuously as `u` approaches the threshold (an "effective exponent"
//! `ln(value)/ln(u)` drifts smoothly away from `1/shape` rather than jumping), which rules out an
//! alternate `pow` rounding path (the exponent `1/shape` is small-integer-valued or exact for the
//! values checked, so any reasonable evaluation should agree to a few ULP) and points to NumPy
//! using a materially different closed form there that black-box probing did not recover.
//! `standard_gamma`, `chisquare`, `f`, and `standard_t` calls whose per-element draws land in
//! that region will not match NumPy's stream.
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
/// differ per stream).
pub(in crate::python) fn standard_gamma(
    bitgen: &mut BitGen,
    shape: f64,
    normal: &mut dyn FnMut(&mut BitGen) -> f64,
    exponential: &mut dyn FnMut(&mut BitGen) -> f64,
) -> f64 {
    if shape < 1.0 {
        return standard_gamma_small(bitgen, shape, exponential);
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
        let u = bitgen.next_double();
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 {
            return d * v;
        }
        if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Ahrens & Dieter's algorithm GS for `0 < shape < 1` (see the module doc for the known gap in
/// the returned value once `u` gets close to, at, or past the branch threshold `1 / boost`).
fn standard_gamma_small(
    bitgen: &mut BitGen,
    shape: f64,
    exponential: &mut dyn FnMut(&mut BitGen) -> f64,
) -> f64 {
    let boost = (std::f64::consts::E + shape) / std::f64::consts::E;
    loop {
        let u = bitgen.next_double();
        let scaled = boost * u;
        if scaled < 1.0 {
            let x = u.powf(1.0 / shape);
            if exponential(bitgen) >= x {
                return x;
            }
        } else {
            let x = -((boost - scaled) / shape).ln();
            if exponential(bitgen) >= (1.0 - shape) * x.ln() {
                return x;
            }
        }
    }
}

/// `chisquare(df)`.
pub(in crate::python) fn chisquare(
    bitgen: &mut BitGen,
    df: f64,
    normal: &mut dyn FnMut(&mut BitGen) -> f64,
    exponential: &mut dyn FnMut(&mut BitGen) -> f64,
) -> f64 {
    2.0 * standard_gamma(bitgen, df / 2.0, normal, exponential)
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
    let g = standard_gamma(bitgen, df / 2.0, normal, exponential);
    n * (df / (2.0 * g)).sqrt()
}
