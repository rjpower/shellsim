//! The regularized incomplete beta function `betainc` (`I_x(a, b)`), its complement `betaincc`,
//! and their inverse `betaincinv`. Every distribution function built on the incomplete beta
//! (`stdtr`, `fdtr`, `bdtr`, the private binomial ufuncs, ...) goes through [`regularized`]
//! rather than its own copy.
//!
//! `I_x(a, b)` is evaluated by the classical continued fraction (W. H. Press et al., "Numerical
//! Recipes", chapter 6.4; the continued fraction itself is due to a result attributed to
//! Legendre): `I_x(a, b) = x^a (1-x)^b / (a B(a,b)) * K(a, b, x)`, with `K` evaluated by Lentz's
//! algorithm. The series converges quickly only for `x < (a+1)/(a+b+2)`; elsewhere this uses the
//! symmetry `I_x(a, b) = 1 - I_{1-x}(b, a)` to stay in the fast region. For modest positive
//! integer `a` and `b`, [`integer_sum`] instead sums the exact binomial identity relating `I_x`
//! to a binomial CDF; every caller still reaches this through [`regularized`], so it stays the
//! one incomplete beta implementation the module doc comment above promises, just with an exact
//! shortcut for that common case in place of the continued fraction's `exp(accumulated log)`
//! rounding.
//!
//! `betaincinv` solves `I_x(a, b) = y` for `x` by Newton's method safeguarded by bisection on
//! `[0, 1]` (no separate starting-point formula is needed: the domain is already a bounded
//! bracket, unlike the incomplete gamma's unbounded `x`).

use super::gamma::gammaln;
use super::meter::tick;

fn log_beta(a: f64, b: f64) -> f64 {
    gammaln(a) + gammaln(b) - gammaln(a + b)
}

/// Lentz's continued fraction for the incomplete beta function, valid (rapidly convergent) for
/// `x < (a+1)/(a+b+2)`.
fn continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    let mut m = 0.0;
    loop {
        m += 1.0;
        let m2 = 2.0 * m;
        let even = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + even * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + even / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let odd = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + odd * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + odd / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-17 || !tick() {
            return h;
        }
    }
}

/// `I_x(a, b) = sum_{j=a}^{a+b-1} C(a+b-1, j) x^j (1-x)^(a+b-1-j)` for a positive integer `a`
/// and `b` (the probability of at least `a` successes in `a + b - 1` Bernoulli(`x`) trials, a
/// standard identity relating the regularized incomplete beta function to the binomial CDF).
/// Bounded to a modest `a + b - 1` so this stays cheap; [`regularized`] falls back to the
/// continued fraction otherwise. Exact for a dyadic `x` like `0.5` where every term is a sum of
/// exact products, unlike the continued fraction's `exp(accumulated log)` reconstruction, which
/// is why this exists: some of `regularized`'s callers compare integer-parameter results exactly.
fn integer_sum(a: f64, b: f64, x: f64) -> Option<f64> {
    let n = a + b - 1.0;
    // Bounded well below where this would cost more than the continued fraction: each of the
    // `n` terms calls `binom(n, j)`, itself up to `O(n)` work, so this path is `O(n^2)` overall
    // where the continued fraction is not. A `bdtr`/`stdtr`/... caller with large integer
    // parameters (e.g. `_binom_ppf`'s search at `n = 1000`) needs to fall through to it.
    if n > 64.0 {
        return None;
    }
    let mut total = 0.0;
    let mut j = a;
    while j <= n {
        total += super::gamma::binom(n, j) * x.powf(j) * (1.0 - x).powf(n - j);
        j += 1.0;
        if !tick() {
            return None;
        }
    }
    Some(total)
}

/// `I_x(a, b)` for `a, b > 0` and `x` in `[0, 1]`; callers apply domain checks first.
pub(in crate::python) fn regularized(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if a >= 1.0 && b >= 1.0 && a == a.floor() && b == b.floor() {
        if let Some(exact) = integer_sum(a, b, x) {
            return exact;
        }
    }
    if x < (a + 1.0) / (a + b + 2.0) {
        let log_prefactor = a * x.ln() + b * (1.0 - x).ln() - log_beta(a, b) - a.ln();
        log_prefactor.exp() * continued_fraction(a, b, x)
    } else {
        let log_prefactor = b * (1.0 - x).ln() + a * x.ln() - log_beta(a, b) - b.ln();
        1.0 - log_prefactor.exp() * continued_fraction(b, a, 1.0 - x)
    }
}

fn density(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - log_beta(a, b)).exp()
}

/// Solve `I_x(a, b) = y` for `x`, by Newton's method safeguarded by bisection on `[0, 1]`.
fn invert(a: f64, b: f64, y: f64) -> f64 {
    if y <= 0.0 {
        return 0.0;
    }
    if y >= 1.0 {
        return 1.0;
    }
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut x = a / (a + b);
    if !(x > 0.0 && x < 1.0) {
        x = 0.5;
    }
    loop {
        let p = regularized(a, b, x);
        if p > y {
            hi = x;
        } else {
            lo = x;
        }
        let residual = p - y;
        if residual == 0.0 || !tick() {
            return x;
        }
        let slope = density(a, b, x);
        if slope > 0.0 {
            let newton = x - residual / slope;
            // A Newton correction too small to change `x` in `f64` precision means `x` is
            // already the closest representable root: stop here. Falling through to the bracket
            // check below would be wrong in this case, not just redundant -- if `x` just became
            // `hi` (or `lo`) on this iteration, `newton` rounding back to that exact boundary
            // reads as "outside the open bracket", triggering a bisection fallback all the way
            // back to the *other*, possibly still-untouched, bound instead of recognizing
            // convergence.
            if newton == x {
                return x;
            }
            if newton > lo && newton < hi {
                x = newton;
                if hi - lo <= 1e-16 {
                    return x;
                }
                continue;
            }
        }
        let midpoint = 0.5 * (lo + hi);
        if midpoint == x || hi - lo <= 1e-16 {
            return midpoint;
        }
        x = midpoint;
    }
}

fn domain_ok(a: f64, b: f64) -> bool {
    a > 0.0 && b > 0.0
}

/// `betainc(a, b, x)`.
pub(in crate::python) fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if !domain_ok(a, b) || !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    regularized(a, b, x)
}

/// `betaincc(a, b, x) = 1 - betainc(a, b, x)`, computed as `I_{1-x}(b, a)` directly so it stays
/// accurate where `betainc` is close to `1`.
pub(in crate::python) fn betaincc(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if !domain_ok(a, b) || !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    regularized(b, a, 1.0 - x)
}

/// `betaincinv(a, b, y)`: solve `betainc(a, b, x) = y` for `x`.
pub(in crate::python) fn betaincinv(a: f64, b: f64, y: f64) -> f64 {
    if a.is_nan() || b.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if !domain_ok(a, b) || !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    invert(a, b, y)
}

// Unit tests for `invert`'s Newton/bisection safeguard, which is easy to get subtly wrong (see
// the `newton == x` comment above): a bracket-membership check applied without that guard reads
// a Newton step that rounds to exactly `lo` or `hi` as "outside the bracket" and bisects against
// the *other*, possibly stale, bound instead of recognizing convergence. Undoing the guard
// reproduces an infinite loop, bounded only by `meter::evaluate`'s doubling allowance escalating
// forever until the real CPU budget is exhausted -- so these are worth pinning directly, not just
// relying on the portable suite's `betaincinv` values to happen to still exercise the branch.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::python::stdlib::scipy::special::meter::run_with_allowance_for_test;

    /// `betaincinv(5.0, 0.5, 0.1)` is the input that first exposed the bug: solving it walked
    /// `x` to within a sub-ULP Newton correction of the bisection bracket's own boundary on the
    /// same iteration `x` became that boundary.
    #[test]
    fn invert_terminates_when_newton_lands_on_the_bracket_boundary() {
        let x = run_with_allowance_for_test(1 << 20, || invert(5.0, 0.5, 0.1));
        assert!(x.is_finite());
        assert!((regularized(5.0, 0.5, x) - 0.1).abs() < 1e-12);
    }

    /// A battery of `(a, b, y)` away from the one known reproducer, including near the domain's
    /// own edges, all of which must converge (not merely terminate) to the root they claim.
    #[test]
    fn invert_converges_across_a_range_of_shapes_and_targets() {
        let cases = [
            (0.5, 0.5, 1e-8),
            (0.5, 0.5, 1.0 - 1e-8),
            (1.0, 1.0, 0.5),
            (100.0, 100.0, 0.5),
            (1000.0, 3.0, 0.999),
            (2.0, 1e6, 1e-9),
        ];
        for (a, b, y) in cases {
            let x = run_with_allowance_for_test(1 << 20, || invert(a, b, y));
            let p = regularized(a, b, x);
            assert!(
                (p - y).abs() < 1e-9,
                "invert({a}, {b}, {y}) = {x}, regularized(..) = {p}"
            );
        }
    }
}
