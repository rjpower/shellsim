//! The regularized incomplete gamma functions `gammainc` (`P`), `gammaincc` (`Q`), and their
//! inverses `gammaincinv`, `gammainccinv`. Every other function in this crate that needs an
//! incomplete gamma value (`erf`, `erfc`, `ndtr`, `chdtr`, `pdtr`, ...) goes through
//! [`regularized`] rather than its own copy.
//!
//! `P(a, x) = gamma(a, x) / Gamma(a)` is evaluated by whichever of two classical methods
//! converges fastest for the given `(a, x)` (W. H. Press et al., "Numerical Recipes", chapter on
//! incomplete gamma functions; the same split appears in Cephes' `igam`/`igamc`):
//!
//! - `x < a + 1`: the power series `P(a, x) = x^a e^{-x} / Gamma(a) * sum_{n>=0} x^n /
//!   ((a)(a+1)...(a+n))`.
//! - `x >= a + 1`: `Q(a, x) = x^a e^{-x} / Gamma(a) * K`, where `K` is Legendre's continued
//!   fraction `1 / (x + 1 - a - 1(1-a) / (x + 3 - a - 2(2-a) / (x + 5 - a - ...)))`, evaluated
//!   with Lentz's algorithm (W. J. Lentz, "Generating Bessel functions in Mie scattering
//!   calculations using continued fractions", Applied Optics 15(3), 1976).
//!
//! The inverses use Newton's method, safeguarded by bisection on a bracket found by doubling
//! search (the "rtsafe" strategy of Numerical Recipes), from a Wilson-Hilferty cube-root-normal
//! starting point (E. B. Wilson, M. M. Hilferty, "The distribution of chi-square", PNAS 17(12),
//! 1931) refined through [`super::erf::ndtri`].

use super::gamma::gammaln;
use super::meter::tick;

/// `(P(a, x), Q(a, x))`. `a` must be finite and positive and `x` finite and non-negative;
/// callers (the public ufunc kernels) turn other inputs into domain-error `NaN` before reaching
/// here.
pub(in crate::python) fn regularized(a: f64, x: f64) -> (f64, f64) {
    if x == 0.0 {
        return (0.0, 1.0);
    }
    let log_prefactor = a * x.ln() - x - gammaln(a);
    if x < a + 1.0 {
        let p = series(a, x, log_prefactor);
        (p, 1.0 - p)
    } else {
        let q = continued_fraction(a, x, log_prefactor);
        (1.0 - q, q)
    }
}

/// The power series for `P(a, x)`, valid (rapidly convergent) for `x < a + 1`.
fn series(a: f64, x: f64, log_prefactor: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut n = a;
    loop {
        n += 1.0;
        term *= x / n;
        sum += term;
        if term.abs() < sum.abs() * 1e-17 || !tick() {
            break;
        }
    }
    (log_prefactor + sum.ln()).exp()
}

/// Legendre's continued fraction for `Q(a, x)`, valid (rapidly convergent) for `x >= a + 1`,
/// evaluated with Lentz's algorithm.
fn continued_fraction(a: f64, x: f64, log_prefactor: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut result = d;
    let mut i = 0.0;
    loop {
        i += 1.0;
        let an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        result *= delta;
        if (delta - 1.0).abs() < 1e-17 || !tick() {
            break;
        }
    }
    (log_prefactor + result.ln()).exp()
}

/// `x^(a-1) e^{-x} / Gamma(a) = dP/dx = -dQ/dx`, the shared Newton derivative for both inverses.
fn density(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() - x - gammaln(a)).exp()
}

/// Solve `P(a, x) = target_p` (equivalently `Q(a, x) = target_q`, `target_p + target_q == 1`)
/// for `x >= 0`, by safeguarded Newton iteration. Working with whichever of `target_p`,
/// `target_q` is smaller keeps the residual accurate near either tail.
fn invert(a: f64, target_p: f64, target_q: f64) -> f64 {
    if target_p <= 0.0 {
        return 0.0;
    }
    if target_q <= 0.0 {
        return f64::INFINITY;
    }
    // Wilson-Hilferty starting point: a chi-square-like cube-root-normal approximation.
    let z = super::erf::ndtri(target_p);
    let ninth = 1.0 / (9.0 * a);
    let mut x = (a * (1.0 - ninth + z * ninth.sqrt()).powi(3)).max(1e-300);

    // Bracket the root by doubling search so Newton always has a safe fallback.
    let mut lo = 0.0;
    let mut hi = x.max(1.0);
    while regularized(a, hi).0 < target_p {
        lo = hi;
        hi *= 2.0;
        if !tick() {
            break;
        }
    }
    if x <= lo || x >= hi {
        x = 0.5 * (lo + hi);
    }

    loop {
        let (p, _) = regularized(a, x);
        let residual = if target_p <= 0.5 {
            p - target_p
        } else {
            target_q - (1.0 - p)
        };
        if residual > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        if residual == 0.0 || !tick() {
            return x;
        }
        let slope = density(a, x);
        if slope > 0.0 {
            let newton = x - residual / slope;
            // See `ibeta::invert`'s identical guard: a Newton correction too small to move `x`
            // means `x` is already the closest representable root, and falling through to the
            // bracket check below (when `x` just became one of the bounds this iteration) would
            // wrongly read that as "outside the bracket" and bisect all the way back to the
            // *other*, possibly stale, bound instead.
            if newton == x {
                return x;
            }
            if newton > lo && newton < hi {
                x = newton;
                if (hi - lo) <= (hi.abs().max(1.0)) * 1e-15 {
                    return x;
                }
                continue;
            }
        }
        let midpoint = 0.5 * (lo + hi);
        if midpoint == x || (hi - lo) <= (hi.abs().max(1.0)) * 1e-15 {
            return midpoint;
        }
        x = midpoint;
    }
}

/// `gammaincinv(a, y)`: solve `P(a, x) = y` for `x`.
pub(in crate::python) fn gammaincinv(a: f64, y: f64) -> f64 {
    if a <= 0.0 || !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    invert(a, y, 1.0 - y)
}

/// `gammainccinv(a, y)`: solve `Q(a, x) = y` for `x`.
pub(in crate::python) fn gammainccinv(a: f64, y: f64) -> f64 {
    if a <= 0.0 || !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    invert(a, 1.0 - y, y)
}

/// `gammainc(a, x)`, SciPy's regularized lower incomplete gamma `P(a, x)`.
pub(in crate::python) fn gammainc(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 1.0;
    }
    regularized(a, x).0
}

/// `gammaincc(a, x)`, SciPy's regularized upper incomplete gamma `Q(a, x)`.
pub(in crate::python) fn gammaincc(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }
    regularized(a, x).1
}

// See `ibeta::tests`' identical rationale: `invert`'s Newton/bisection safeguard is tricky
// enough, and its failure mode severe enough (an infinite loop bounded only by CPU exhaustion),
// to pin directly rather than trust to the portable suite's `gammaincinv`/`gammainccinv` values
// happening to still exercise the boundary-snapping branch.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::python::stdlib::scipy::special::meter::run_with_allowance_for_test;

    /// A battery of `(a, target_p)` including near the domain's edges (small `a`, `target_p`
    /// close to `0` or `1`, where the bracket doubling and Newton step are both least stable),
    /// all of which must converge to the root they claim.
    #[test]
    fn invert_converges_across_a_range_of_shapes_and_targets() {
        let cases = [
            (0.5, 1e-9),
            (0.5, 1.0 - 1e-9),
            (1.0, 0.5),
            (5.0, 0.1),
            (100.0, 0.5),
            (1000.0, 1e-6),
            (1e-3, 0.99),
        ];
        for (a, target_p) in cases {
            let x = run_with_allowance_for_test(1 << 20, || invert(a, target_p, 1.0 - target_p));
            let (p, _) = regularized(a, x);
            assert!(
                (p - target_p).abs() < 1e-9,
                "invert({a}, {target_p}) = {x}, regularized(..).0 = {p}"
            );
        }
    }
}
