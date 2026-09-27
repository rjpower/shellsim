//! Gamma-family draws: `standard_gamma`, and the `chisquare`, `f`, and `standard_t` variates
//! built from it.
//!
//! For `shape >= 1` this is Marsaglia and Tsang's squeeze-and-reject method ("A Simple Method
//! for Generating Gamma Variables", 2000): propose `d*(1+c*z)**3` from a normal deviate `z`, and
//! accept it through the cheap `0.0331*z**4` squeeze or the full log test.
//!
//! For `shape = a < 1` it is a rejection method split on the driving uniform `u`, each branch
//! accepting against an independent standard exponential `e`:
//!
//! - `u <= 1 - a`: `x = u**(1/a)`, accepted when `e >= x`;
//! - `u > 1 - a`: with `y = ln((1 - u)/a)`, `x = ((1 - a) - a*y)**(1/a)`, accepted when
//!   `e >= x + y`.
//!
//! A rejection restarts from a fresh `u`. The callers choose the deviate sources: `Generator`
//! uses the ziggurat normal and exponential, legacy `RandomState` the polar normal and inverted
//! exponential. For `dtype=np.float32`, `Generator` draws the uniforms from single 32-bit words
//! and the exponential from the `float32` ziggurat, as NumPy does; the arithmetic stays `f64`.
//!
//! `chisquare(df) = 2 * standard_gamma(df / 2)`, `f` is a ratio of scaled chi-squares, and
//! `standard_t` divides a normal deviate by the root-mean-square of a chi-square. Array
//! parameters draw element by element, in NumPy's order.

use super::bitgen::BitGen;

/// One standard gamma draw with unit scale. `normal` feeds the `shape >= 1` branch,
/// `exponential` the `shape < 1` acceptance test, and `uniform` both branches' uniforms.
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

/// The `0 < shape < 1` rejection method described in the module doc.
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
    // This grouping, rather than `n * sqrt(df / (2 * g))`, rounds as NumPy does.
    n * (df / 2.0).sqrt() / g.sqrt()
}
