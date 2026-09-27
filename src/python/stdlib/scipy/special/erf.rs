//! The error-function family (`erf`, `erfc`, `erfinv`, `erfcinv`) and the normal-distribution
//! functions (`ndtr`, `log_ndtr`, `ndtri`) they share.
//!
//! `erf` and `erfc` reduce to the regularized incomplete gamma function at `a = 1/2`: the
//! standard identity `erf(x) = P(1/2, x^2)` for `x >= 0` (substitute `u = t^2` in the defining
//! integral of `erf`), reflected for negative `x`. This reuses [`super::igam`] instead of a
//! separate rational approximation.
//!
//! `ndtri` (the standard normal quantile) is Peter Acklam's rational approximation ("An
//! algorithm for computing the inverse normal cumulative distribution function", 2003, a widely
//! reproduced public-domain algorithm accurate to about `1.15e-9` relative error), refined to
//! full double precision by one step of Halley's rational method on `ndtr` (as Acklam's own
//! writeup recommends). `erfcinv` is then just `ndtri` rescaled: `erfcinv(x) = -ndtri(x/2) /
//! sqrt(2)`. `erfinv(x) = ndtri((x+1)/2) / sqrt(2)` the same way, plus one further Newton
//! step directly against `erf`: `ndtri`'s Halley step loses precision for `erfinv` of a small
//! `x`, where the probability it works in (`p = (x+1)/2`) sits next to `0.5` and so shares only
//! half of a `float64`'s bits with the deviation being refined, while `erf(y) - x` itself stays
//! full precision there.

use std::f64::consts::{PI, SQRT_2};

use super::igam::{gammainc, gammaincc};

/// `erf(x)`.
pub(in crate::python) fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        -gammainc(0.5, x * x)
    } else {
        gammainc(0.5, x * x)
    }
}

/// `erfc(x) = 1 - erf(x)`, computed directly from the upper incomplete gamma function so it
/// stays accurate for large `x`, where `1 - erf(x)` would lose precision by cancellation.
pub(in crate::python) fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        gammaincc(0.5, x * x)
    } else {
        2.0 - gammaincc(0.5, x * x)
    }
}

/// `ndtr(x) = 0.5 erfc(-x / sqrt(2))`, the standard normal CDF.
pub(in crate::python) fn ndtr(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    0.5 * erfc(-x / SQRT_2)
}

/// `log(ndtr(x))`. Below `x = -20`, `ndtr(x)` itself can underflow to `0.0` (its true value is
/// far smaller than the smallest positive `f64`), so this uses the standard asymptotic expansion
/// of the normal tail (DLMF 7.17: `Q(x) ~ phi(x)/x * (1 - 1/x^2 + 3/x^4 - 15/x^6 + 105/x^8)` for
/// the upper tail, mirrored here since `ndtr(x) = Q(-x)`) instead of `ndtr(x).ln()`.
pub(in crate::python) fn log_ndtr(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < -20.0 {
        let inverse_square = 1.0 / (x * x);
        let series = 1.0
            + inverse_square
                * (-1.0
                    + inverse_square * (3.0 + inverse_square * (-15.0 + inverse_square * 105.0)));
        return -0.5 * x * x - (-x).ln() - 0.5 * (2.0 * PI).ln() + series.ln();
    }
    // For `x > 0`, `ndtr(x)` is close to `1`, so `ln` of it directly loses precision the same way
    // taking `ln` of `1 - epsilon` always does; `ndtr(-x)` is the same small complement, and
    // `ln_1p` keeps full precision of `1 - small` instead of first rounding it to a `float64`
    // near `1`.
    let value = if x > 0.0 {
        (-ndtr(-x)).ln_1p()
    } else {
        ndtr(x).ln()
    };
    // `ndtr(x) < 1` strictly for finite `x`, so `log_ndtr(x)` is never truly `0.0`; a `0.0` here
    // is `ndtr(-x)` having underflowed to exactly `0.0`, and the sign bit is the only trace left
    // of which side of `1` the true value was on.
    if value == 0.0 && x.is_finite() {
        -0.0
    } else {
        value
    }
}

const ACKLAM_A: [f64; 6] = [
    -3.969_683_028_665_376e1,
    2.209_460_984_245_205e2,
    -2.759_285_104_469_687e2,
    1.383_577_518_672_69e2,
    -3.066_479_806_614_716e1,
    2.506_628_277_459_239,
];
const ACKLAM_B: [f64; 5] = [
    -5.447_609_879_822_406e1,
    1.615_858_368_580_409e2,
    -1.556_989_798_598_866e2,
    6.680_131_188_771_972e1,
    -1.328_068_155_288_572e1,
];
const ACKLAM_C: [f64; 6] = [
    -7.784_894_002_430_293e-3,
    -3.223_964_580_411_365e-1,
    -2.400_758_277_161_838,
    -2.549_732_539_343_734,
    4.374_664_141_464_968,
    2.938_163_982_698_783,
];
const ACKLAM_D: [f64; 4] = [
    7.784_695_709_041_462e-3,
    3.224_671_290_700_398e-1,
    2.445_134_137_142_996,
    3.754_408_661_907_416,
];

/// `P_LOW` from Acklam's algorithm: below this (or above `1 - P_LOW`), `ndtri` uses the tail
/// rational approximation instead of [`central_from_deviation`].
const P_LOW: f64 = 0.024_25;

/// The central branch of Acklam's rational approximation, taking `q = p - 0.5` directly rather
/// than recovering it from `p`. [`erfinv`] already has `q` exactly (`x / 2`, no addition
/// involved), and re-deriving it as `p - 0.5` after first rounding `p = (x + 1) / 2` for small
/// `x` would cancel away most of `x`'s significant digits before this even runs.
fn central_from_deviation(q: f64) -> f64 {
    let r = q * q;
    let num = ((((ACKLAM_A[0] * r + ACKLAM_A[1]) * r + ACKLAM_A[2]) * r + ACKLAM_A[3]) * r
        + ACKLAM_A[4])
        * r
        + ACKLAM_A[5];
    let den = ((((ACKLAM_B[0] * r + ACKLAM_B[1]) * r + ACKLAM_B[2]) * r + ACKLAM_B[3]) * r
        + ACKLAM_B[4])
        * r
        + 1.0;
    num * q / den
}

/// Acklam's rational approximation of `ndtri(p)`, before Halley refinement.
fn ndtri_rational(p: f64) -> f64 {
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        let num = ((((ACKLAM_C[0] * q + ACKLAM_C[1]) * q + ACKLAM_C[2]) * q + ACKLAM_C[3]) * q
            + ACKLAM_C[4])
            * q
            + ACKLAM_C[5];
        let den = (((ACKLAM_D[0] * q + ACKLAM_D[1]) * q + ACKLAM_D[2]) * q + ACKLAM_D[3]) * q + 1.0;
        return num / den;
    }
    if p <= 1.0 - P_LOW {
        return central_from_deviation(p - 0.5);
    }
    -ndtri_rational(1.0 - p)
}

/// Halley's rational method on `ndtr(x) - p = 0`, as Acklam's writeup recommends, to bring the
/// ~1e-9 rational starting point up to full double precision.
fn halley_refine(mut x: f64, p: f64) -> f64 {
    for _ in 0..2 {
        // `ndtr(x) - p`, computed so it never rounds `ndtr(x)` to "close to `1`" first: for
        // `x > 0`, `ndtr(x) = 1 - ndtr(-x)`, and `(1 - p) - ndtr(-x)` reaches the same value by
        // way of two quantities that are each already close to `0` (`ndtr(-x)` directly, `1 - p`
        // exactly by Sterbenz's lemma), rather than one close to `1` losing the low-order bits a
        // tail probability like `1e-10` needs.
        let error = if x <= 0.0 {
            ndtr(x) - p
        } else {
            (1.0 - p) - ndtr(-x)
        };
        let density = (-0.5 * x * x).exp() / (2.0 * PI).sqrt();
        if density == 0.0 {
            break;
        }
        let u = error / density;
        x -= u / (1.0 + x * u / 2.0);
    }
    x
}

/// `ndtri(p)`, the standard normal quantile function (inverse of [`ndtr`]).
pub(in crate::python) fn ndtri(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return if p == 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }
    if p >= 1.0 {
        return if p == 1.0 { f64::INFINITY } else { f64::NAN };
    }
    halley_refine(ndtri_rational(p), p)
}

/// `erfinv(x) = ndtri((x+1)/2) / sqrt(2)`, computed from `q = x / 2` (the deviation `p - 0.5`,
/// exact) rather than recovering `p - 0.5` from a first-rounded `p = (x + 1) / 2` (see
/// [`central_from_deviation`]).
pub(in crate::python) fn erfinv(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= -1.0 {
        return if x == -1.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }
    if x >= 1.0 {
        return if x == 1.0 { f64::INFINITY } else { f64::NAN };
    }
    let q = 0.5 * x;
    let p = 0.5 + q;
    let start = if q.abs() <= 0.5 - P_LOW {
        central_from_deviation(q)
    } else {
        ndtri_rational(p)
    };
    let y = halley_refine(start, p) / SQRT_2;
    // One more Newton step, directly on `erf(y) - x = 0` rather than `ndtr` (which is [`halley_
    // refine`]'s domain): for small `x`, `x` and `erf(y)` are both close to `0`, so their
    // difference keeps full precision, where `ndtr(y sqrt(2)) - p` loses precision to `p`'s own
    // proximity to `0.5` (see [`central_from_deviation`]'s doc comment for the same issue one
    // level up).
    let error = erf(y) - x;
    let derivative = std::f64::consts::FRAC_2_SQRT_PI * (-y * y).exp();
    if derivative == 0.0 {
        y
    } else {
        y - error / derivative
    }
}

/// `erfcinv(x) = -ndtri(x / 2) / sqrt(2)`: `erfc(y) = 2 ndtr(-y sqrt(2))` (from `erfc = 1 - erf`
/// and the `ndtr`/`erf` identity), so `x = erfc(y)` gives `ndtr(-y sqrt(2)) = x / 2`, i.e.
/// `-y sqrt(2) = ndtri(x / 2)`.
pub(in crate::python) fn erfcinv(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return if x == 0.0 { f64::INFINITY } else { f64::NAN };
    }
    if x >= 2.0 {
        return if x == 2.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }
    -ndtri(0.5 * x) / SQRT_2
}
