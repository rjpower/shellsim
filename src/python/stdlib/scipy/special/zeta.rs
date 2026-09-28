//! The Hurwitz zeta function `_scipy_special.zeta(x, q)`; the frozen `zeta(x, q=None)` wrapper in
//! `source/scipy/special.py` supplies `q = 1.0` for the one-argument Riemann zeta case and, only
//! then, applies SciPy's `x >= 1` domain restriction on the two-argument form in Python (the
//! computation below is mathematically valid for any real `x` away from the poles at non-positive
//! integer `q`; the narrower two-argument domain is an API-parity choice, not a numerical one, so
//! it does not belong in the kernel).
//!
//! Evaluation uses the standard Euler-Maclaurin summation (DLMF 25.11.2; the same method Cephes'
//! `zetac` and mpmath's `zeta` use): shift `q` up by the recurrence `zeta(s, q) = q^-s +
//! zeta(s, q+1)` until it is past a threshold where a short Bernoulli-number asymptotic tail
//! already reaches double precision, then add that tail. Shifting `q` up by exactly `1` each
//! step, rather than jumping to the threshold directly, is what makes the cost of a very negative
//! `q` (see `docs/scipy.md`, "Safety and accounting") proportional to `|q|`, matching SciPy's own
//! cost there.

use super::meter::tick;

/// `B_{2k} / (2k)!` for `k = 1..=6`, the coefficients of the Euler-Maclaurin tail.
const TAIL_COEFFICIENTS: [f64; 6] = [
    1.0 / 12.0,
    -1.0 / 720.0,
    1.0 / 30240.0,
    -1.0 / 1_209_600.0,
    1.0 / 47_900_160.0,
    -691.0 / 1_307_674_368_000.0,
];

/// `q` past which the Euler-Maclaurin tail alone reaches double precision.
const THRESHOLD: f64 = 15.0;

/// The asymptotic tail of the Euler-Maclaurin expansion at `q >= THRESHOLD`:
/// `q^(1-s)/(s-1) + q^-s/2 + sum_k B_2k/(2k)! (s)_(2k-1) q^(-s-2k+1)`.
fn tail(s: f64, q: f64) -> f64 {
    let q_neg_s = q.powf(-s);
    let mut result = q_neg_s * q / (s - 1.0) + 0.5 * q_neg_s;
    let mut rising = s;
    let mut power = q_neg_s / q;
    let q_squared = q * q;
    for (index, coefficient) in TAIL_COEFFICIENTS.iter().enumerate() {
        let k = (index + 1) as f64;
        result += coefficient * rising * power;
        rising *= (s + 2.0 * k - 1.0) * (s + 2.0 * k);
        power /= q_squared;
    }
    result
}

/// `zeta(s, q) = sum_{k>=0} (q+k)^-s`, for `q > 0` (the recurrence used to get there also
/// accepts non-positive `q`, away from its poles at the non-positive integers).
fn hurwitz_zeta(s: f64, q: f64) -> f64 {
    let mut q = q;
    let mut sum = 0.0;
    while q < THRESHOLD {
        sum += q.powf(-s);
        q += 1.0;
        if !tick() {
            break;
        }
    }
    sum + tail(s, q)
}

/// `_scipy_special.zeta(x, q)`. `q` may be any real away from the Hurwitz zeta's poles at the
/// non-positive integers, including large negative values (see the module doc comment on the
/// cost of that); `x` is unrestricted here (the pole at `x == 1` falls out of the formula as
/// `+/-inf` on its own).
pub(in crate::python) fn zeta(x: f64, q: f64) -> f64 {
    if x.is_nan() || q.is_nan() {
        return f64::NAN;
    }
    hurwitz_zeta(x, q)
}
