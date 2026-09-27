//! Behavior unique to legacy `RandomState`: the polar-method Gaussian and its one-value cache.
//!
//! `RandomState`'s Gaussian predates the ziggurat method NumPy's `Generator` uses. It is the
//! Marsaglia polar method: reject points outside the unit disc, then scale by
//! `sqrt(-2*ln(r2)/r2)`. The method produces two independent deviates per accepted point, so
//! `RandomState` caches the second one (`has_gauss`/`gauss` in `get_state()`) for the following
//! draw. This matches NumPy's legacy stream bit for bit, including which of the two points is
//! returned first and which is cached.

use super::bitgen::BitGen;

/// One Gaussian draw from the legacy polar method, consuming or filling the one-value cache.
pub(in crate::python) fn gauss(bitgen: &mut BitGen, has_gauss: &mut bool, cached: &mut f64) -> f64 {
    if *has_gauss {
        *has_gauss = false;
        return *cached;
    }
    loop {
        let x1 = 2.0 * bitgen.next_double() - 1.0;
        let x2 = 2.0 * bitgen.next_double() - 1.0;
        let r2 = x1 * x1 + x2 * x2;
        if r2 < 1.0 && r2 != 0.0 {
            let f = (-2.0 * r2.ln() / r2).sqrt();
            *cached = f * x1;
            *has_gauss = true;
            return f * x2;
        }
    }
}

/// Legacy `standard_exponential`: simple inversion, `-log(1 - U)`. `Generator`'s default method
/// is the ziggurat instead (`super::ziggurat::next_exponential_zig`); its `method="inv"` uses
/// this same formula.
pub(in crate::python) fn exponential(bitgen: &mut BitGen) -> f64 {
    -(1.0 - bitgen.next_double()).ln()
}
