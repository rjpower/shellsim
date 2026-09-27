//! Gamma-family draws: `standard_gamma`, and the `chisquare`, `f`, and `standard_t` variates
//! built from it.
//!
//! For `shape >= 1` this is Marsaglia & Tsang's 2000 squeeze-and-reject method ("A Simple
//! Method for Generating Gamma Variables"): propose `d*(1+c*z)**3` from a normal deviate `z`,
//! and accept it either through the cheap `0.0331*z**4` squeeze or the full log test. For
//! `shape < 1` this is Ahrens & Dieter's 1974 algorithm GS ("Computer Methods for Sampling from
//! Gamma, Beta, Poisson and Normal Distributions"): boost through `u**(1/shape)` or the
//! logarithmic branch depending on which side of `1` the scaled uniform falls on.
//!
//! **Accuracy** (checked against NumPy 2.5.3, see `docs/numpy.md`): the `shape >= 1` branch
//! matches bit for bit when the normal deviates behind it match, which holds for legacy
//! `RandomState` (whose Gaussian is the polar method in [`super::legacy::gauss`]) but not for
//! `Generator`, whose ziggurat normal does not reproduce NumPy's stream. The `shape < 1` branch's
//! common case (`u * boost < 1`, roughly `e / (e + shape)` of draws) matches bit for bit on
//! legacy `RandomState`; its rarer case (`u * boost >= 1`) does not reproduce NumPy's stream on
//! any bit generator, and `Generator` does not reproduce either case reliably even for a single
//! element — NumPy's `Generator` path apparently draws its second, rejection-test uniform through
//! the same undocumented mechanism as its ziggurat exponential, which this module could not
//! recover by black-box comparison in the time available.
//!
//! `chisquare(df) = 2 * standard_gamma(df / 2)`, `f(dfnum, dfden)` divides two independent
//! scaled chi-squares, and `standard_t(df)` divides a normal deviate by the root-mean-square of
//! an independent chi-square — the standard constructions for these distributions in terms of
//! gamma and normal deviates, applied per output element so array parameters consume the bit
//! generator in the same per-element order NumPy's own fused loops do.

use super::bitgen::BitGen;

/// One standard gamma draw with unit scale. `normal` supplies the source of normal deviates for
/// the `shape >= 1` branch (ziggurat for `Generator`, the polar method for `RandomState`).
pub(in crate::python) fn standard_gamma(
    bitgen: &mut BitGen,
    shape: f64,
    normal: &mut dyn FnMut(&mut BitGen) -> f64,
) -> f64 {
    if shape < 1.0 {
        return standard_gamma_small(bitgen, shape);
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

/// Ahrens & Dieter's algorithm GS for `0 < shape < 1`.
fn standard_gamma_small(bitgen: &mut BitGen, shape: f64) -> f64 {
    let boost = (std::f64::consts::E + shape) / std::f64::consts::E;
    loop {
        let u = bitgen.next_double();
        let scaled = boost * u;
        if scaled < 1.0 {
            let x = u.powf(1.0 / shape);
            let exponential = -(1.0 - bitgen.next_double()).ln();
            if exponential >= x {
                return x;
            }
        } else {
            let x = -((boost - scaled) / shape).ln();
            let exponential = -(1.0 - bitgen.next_double()).ln();
            if exponential >= (1.0 - shape) * x.ln() {
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
) -> f64 {
    2.0 * standard_gamma(bitgen, df / 2.0, normal)
}

/// `f(dfnum, dfden)`.
pub(in crate::python) fn f_distribution(
    bitgen: &mut BitGen,
    dfnum: f64,
    dfden: f64,
    normal: &mut dyn FnMut(&mut BitGen) -> f64,
) -> f64 {
    let numerator = chisquare(bitgen, dfnum, normal) * dfden;
    let denominator = chisquare(bitgen, dfden, normal) * dfnum;
    numerator / denominator
}

/// `standard_t(df)`.
pub(in crate::python) fn standard_t(
    bitgen: &mut BitGen,
    df: f64,
    normal: &mut dyn FnMut(&mut BitGen) -> f64,
) -> f64 {
    let n = normal(bitgen);
    let g = standard_gamma(bitgen, df / 2.0, normal);
    n * (df / (2.0 * g)).sqrt()
}
