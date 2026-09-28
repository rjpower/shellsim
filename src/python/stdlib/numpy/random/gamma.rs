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
//! A rejection restarts from a fresh `u`. `chisquare(df) = 2 * standard_gamma(df / 2)`, `f` is a
//! ratio of scaled chi-squares, and `standard_t` divides a normal deviate by the root-mean-square
//! of a chi-square. Array parameters draw element by element.

use super::bitgen::Pcg64;
use super::normal;

/// One standard gamma draw with unit scale.
pub(in crate::python) fn standard_gamma(bitgen: &mut Pcg64, shape: f64) -> f64 {
    if shape < 1.0 {
        return standard_gamma_small(bitgen, shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let (x, v) = loop {
            let x = normal::next_gauss(bitgen);
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

/// The `0 < shape < 1` rejection method described in the module doc.
fn standard_gamma_small(bitgen: &mut Pcg64, shape: f64) -> f64 {
    loop {
        let u = bitgen.next_double();
        let (x, threshold) = if u <= 1.0 - shape {
            let x = u.powf(1.0 / shape);
            (x, x)
        } else {
            let log_term = ((1.0 - u) / shape).ln();
            let z = (1.0 - shape) - shape * log_term;
            let x = z.powf(1.0 / shape);
            (x, x + log_term)
        };
        if normal::next_exponential(bitgen) >= threshold {
            return x;
        }
    }
}

/// `chisquare(df)`.
pub(in crate::python) fn chisquare(bitgen: &mut Pcg64, df: f64) -> f64 {
    2.0 * standard_gamma(bitgen, df / 2.0)
}

/// `f(dfnum, dfden)`.
pub(in crate::python) fn f_distribution(bitgen: &mut Pcg64, dfnum: f64, dfden: f64) -> f64 {
    let numerator = chisquare(bitgen, dfnum) * dfden;
    let denominator = chisquare(bitgen, dfden) * dfnum;
    numerator / denominator
}

/// `standard_t(df)`.
pub(in crate::python) fn standard_t(bitgen: &mut Pcg64, df: f64) -> f64 {
    let n = normal::next_gauss(bitgen);
    let g = standard_gamma(bitgen, df / 2.0);
    // This grouping, rather than `n * sqrt(df / (2 * g))`, keeps the two square roots small.
    n * (df / 2.0).sqrt() / g.sqrt()
}
