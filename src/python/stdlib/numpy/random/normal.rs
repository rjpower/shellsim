//! Standard normal and standard exponential draws.
//!
//! Normal deviates use Marsaglia's polar method: reject points outside the unit disc, then scale
//! by `sqrt(-2*ln(r2)/r2)`. Each accepted point actually yields two independent deviates; this
//! implementation keeps one and discards the other rather than threading a one-value cache
//! through every caller and through bit-generator state (NumPy's legacy generator keeps that
//! cache; shellsim does not need to reproduce its stream — see `random.py`'s module docstring —
//! so the simpler, cache-free form is enough).
//!
//! Exponential deviates use inversion, `-ln(1 - U)`. NumPy's `Generator` defaults to a ziggurat
//! method and offers inversion only as `method="inv"`; since shellsim's streams do not need to
//! match NumPy's, one simple, statistically sound method covers every caller.

use super::bitgen::Pcg64;

/// One standard normal draw.
pub(in crate::python) fn next_gauss(bitgen: &mut Pcg64) -> f64 {
    loop {
        let x1 = 2.0 * bitgen.next_double() - 1.0;
        let x2 = 2.0 * bitgen.next_double() - 1.0;
        let r2 = x1 * x1 + x2 * x2;
        if r2 < 1.0 && r2 != 0.0 {
            let f = (-2.0 * r2.ln() / r2).sqrt();
            return f * x1;
        }
    }
}

/// One standard exponential draw.
pub(in crate::python) fn next_exponential(bitgen: &mut Pcg64) -> f64 {
    -(1.0 - bitgen.next_double()).ln()
}
