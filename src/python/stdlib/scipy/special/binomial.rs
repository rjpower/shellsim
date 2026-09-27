//! The binomial distribution SciPy's private `_binom_pmf`, `_binom_cdf`, `_binom_sf`,
//! `_binom_ppf` and `_binom_isf` ufuncs compute for `scipy.stats.binom`.
//!
//! SciPy wraps Boost's `binomial_distribution` (`boost/math/distributions/binomial.hpp`) under
//! its stats policy: domain errors give NaN, and a discrete quantile rounds up to an integer.
//! The mass, distribution and survival functions here port Boost's code over this crate's
//! incomplete beta function.
//!
//! Boost finds a quantile by bracketing the root of the distribution function, continued to real
//! `k`, from a Cornish-Fisher guess, solving with TOMS 748 until the bracket's ends share a
//! ceiling, and rounding up (`inv_discrete_quantile.hpp`). The result is usually the smallest
//! integer `k` with `cdf(k) >= p`, but the search's shortcuts and floating-point tests decide
//! some results at the extremes, so this port follows Boost's steps rather than searching the
//! integers directly. Boost gives up after 200 evaluations, which SciPy reports as NaN.

use std::f64::consts::SQRT_2;

use super::erf_inv::erfc_inv;
use super::ibeta::{ibeta_derivative, ibeta_imp};
use super::{meter, roots};

/// SciPy's stats policy keeps Boost's default limit on root-finding iterations.
const MAX_ROOT_ITERATIONS: u64 = 200;

/// Boost's `check_dist`: `n` trials and success probability `p` form a distribution.
fn valid_distribution(n: f64, p: f64) -> bool {
    (0.0..=1.0).contains(&p) && n >= 0.0 && n.is_finite()
}

/// Boost's `check_dist_and_k`, which also requires `0 <= k <= n` with `k` finite.
fn valid_successes(k: f64, n: f64, p: f64) -> bool {
    valid_distribution(n, p) && k >= 0.0 && k.is_finite() && k <= n
}

/// The probability of exactly `k` successes in `n` trials with success probability `p`.
pub(super) fn binom_pmf(k: f64, n: f64, p: f64) -> f64 {
    if !valid_successes(k, n, p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return if k == 0.0 { 1.0 } else { 0.0 };
    }
    if p == 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }
    if n == 0.0 {
        return 1.0;
    }
    if k == n {
        return p.powf(k);
    }
    ibeta_derivative(k + 1.0, n - k + 1.0, p) / (n + 1.0)
}

/// Boost's `cdf`: the probability of at most `k` successes, for `k` in the support.
fn cdf(k: f64, n: f64, p: f64) -> f64 {
    if !valid_successes(k, n, p) {
        return f64::NAN;
    }
    if k == n || p == 0.0 {
        return 1.0;
    }
    if p == 1.0 {
        return 0.0;
    }
    ibeta_imp(k + 1.0, n - k, p, true)
}

/// Boost's complemented `cdf`: the probability of more than `k` successes.
fn sf(k: f64, n: f64, p: f64) -> f64 {
    if !valid_successes(k, n, p) {
        return f64::NAN;
    }
    if k == n || p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    ibeta_imp(k + 1.0, n - k, p, false)
}

/// SciPy's `binom_cdf_wrap`: infinite `k` gives 0 or 1 by its sign.
pub(super) fn binom_cdf(k: f64, n: f64, p: f64) -> f64 {
    if k.is_infinite() {
        return if k > 0.0 { 1.0 } else { 0.0 };
    }
    cdf(k, n, p)
}

pub(super) fn binom_sf(k: f64, n: f64, p: f64) -> f64 {
    sf(k, n, p)
}

/// Boost's `inverse_binomial_cornish_fisher`: a normal approximation with a skewness
/// correction to the quantile at lower-tail probability `lower` (`upper = 1 - lower`).
fn inverse_binomial_cornish_fisher(n: f64, p: f64, lower: f64, upper: f64) -> f64 {
    let mean = n * p;
    let sigma = (n * p * (1.0 - p)).sqrt();
    let skewness = (1.0 - 2.0 * p) / sigma;
    let mut x = erfc_inv(if lower > upper {
        2.0 * upper
    } else {
        2.0 * lower
    }) * SQRT_2;
    if lower < 0.5 {
        x = -x;
    }
    let w = mean + sigma * (x + skewness * (x * x - 1.0) / 6.0);
    if w < f64::MIN_POSITIVE {
        return f64::MIN_POSITIVE.sqrt();
    }
    if w > n {
        return n;
    }
    w
}

/// Evaluate `f` at `x` as one metered step.
fn metered(f: &mut impl FnMut(f64) -> f64, x: f64) -> Option<f64> {
    meter::step().then(|| f(x))
}

/// Boost's `do_inverse_discrete_quantile` with its `equal_ceil` tolerance: a bracket around the
/// real root of the increasing function `f` on `[0, n]`, found by unit steps near zero or from
/// `guess`, then by scaling with `multiplier`, and narrowed with TOMS 748 until its ends share a
/// ceiling. Returns the bracket's midpoint, or an end Boost returns directly; `None` where Boost
/// raises an error.
///
/// Boost's quirks are kept, since they decide results at the extremes: the unit steps test the
/// sign change with `fa * fb >= 0`, which an underflowing product passes, and a step down keeps
/// the value at the previous upper end.
fn do_inverse_discrete_quantile(
    f: &mut impl FnMut(f64) -> f64,
    n: f64,
    guess: f64,
    multiplier: f64,
) -> Option<f64> {
    let guess = guess.min(n).max(0.0);
    let mut fa = metered(f, guess)?;
    let mut count = MAX_ROOT_ITERATIONS - 1;
    let (mut a, mut b, mut fb) = (guess, 0.0, fa);
    if fa == 0.0 {
        return Some(guess);
    }
    if guess < 10.0 {
        b = a;
        while a < 10.0 && fa * fb >= 0.0 {
            if fb <= 0.0 {
                a = b;
                b = (a + 1.0).min(n);
                fb = metered(f, b)?;
                count -= 1;
                if fb == 0.0 || a == b {
                    return Some(b);
                }
            } else {
                b = a;
                a = (b - 1.0).max(0.0);
                fa = metered(f, a)?;
                count -= 1;
                if fa == 0.0 || a == b {
                    return Some(a);
                }
            }
        }
    } else if a + 1.0 != a {
        let step = |a: f64, fa: f64| {
            if fa < 0.0 {
                (a + 1.0).min(n)
            } else {
                (a - 1.0).max(0.0)
            }
        };
        b = step(a, fa);
        fb = metered(f, b)?;
        count -= 1;
        if fb == 0.0 {
            return Some(b);
        }
        if count != 0 && fa * fb >= 0.0 {
            a = b;
            fa = fb;
            b = step(a, fa);
            fb = metered(f, b)?;
            count -= 1;
        }
        if a > b {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }
    if roots::sign(fb) == roots::sign(fa) {
        if fa < 0.0 {
            while roots::sign(fb) == roots::sign(fa) && a != b {
                if count == 0 {
                    return None;
                }
                a = b;
                fa = fb;
                b = (b * multiplier).min(n);
                fb = metered(f, b)?;
                count -= 1;
            }
        } else {
            while roots::sign(fb) == roots::sign(fa) && a != b {
                if a.abs() < f64::MIN_POSITIVE {
                    return Some(0.0);
                }
                if count == 0 {
                    return None;
                }
                b = a;
                fb = fa;
                a /= multiplier;
                fa = metered(f, a)?;
                count -= 1;
            }
        }
    }
    let used = MAX_ROOT_ITERATIONS - count;
    if fa == 0.0 {
        return Some(a);
    }
    if fb == 0.0 || a == b {
        return Some(b);
    }
    // Boost's `adjust_bounds` for `equal_ceil`, which keeps `fa` from the unadjusted end.
    a += f64::EPSILON * a;
    a = a.max(f64::MIN_POSITIVE);
    let (low, high) = roots::toms748_solve(f, (a, b), (fa, fb), &roots::equal_ceil, &mut count)?;
    (used + count < MAX_ROOT_ITERATIONS).then_some((low + high) / 2.0)
}

/// Boost's `round_to_ceil`: the integer at or above the real root `result`, or the one below it
/// if its `probability` equals `target` exactly, then moved up while the next integer's
/// probability has not passed `target`. The lower tail's probability grows with `k`; the upper
/// tail's falls.
fn round_to_ceil(
    result: f64,
    n: f64,
    target: f64,
    complement: bool,
    probability: impl Fn(f64) -> f64,
) -> Option<f64> {
    let below = result.floor();
    let at_below = if below >= 0.0 {
        probability(below)
    } else {
        0.0
    };
    let mut result = if at_below == target {
        below
    } else {
        result.ceil()
    };
    loop {
        let next = result.next_up().ceil();
        if next > n {
            break;
        }
        if !meter::step() {
            return None;
        }
        let value = probability(next);
        if if complement {
            value < target
        } else {
            value > target
        } {
            break;
        }
        result = next;
    }
    Some(result)
}

/// Boost's `quantile_imp` under SciPy's stats policy: the lower-tail quantile for probability
/// `lower`, or with `complement`, the upper-tail quantile for probability `upper`, where `lower`
/// and `upper` sum to 1.
fn quantile(n: f64, p: f64, lower: f64, upper: f64, complement: bool) -> f64 {
    if !valid_distribution(n, p) || !(0.0..=1.0).contains(&lower) {
        return f64::NAN;
    }
    if lower == 0.0 {
        return 0.0;
    }
    if lower == 1.0 || p == 1.0 {
        return n;
    }
    if lower <= (1.0 - p).powf(n) {
        return 0.0;
    }
    // How far the search widens its bracket depends on how far Boost trusts the guess. Boost
    // writes the factors as `float` literals.
    let mut guess = inverse_binomial_cornish_fisher(n, p, lower, upper);
    let factor = if n > 100.0 {
        f64::from(1.01f32)
    } else if n > 10.0 && n - 1.0 > guess && guess > 3.0 {
        f64::from(1.15f32)
    } else if n < 10.0 {
        if guess > n / 64.0 {
            guess = n / 4.0;
            2.0
        } else {
            guess = n / 1024.0;
            8.0
        }
    } else {
        2.0
    };
    // `inverse_discrete_quantile` for `integer_round_up`.
    let target = if complement { upper } else { lower };
    if (if complement { 1.0 - target } else { target }) <= binom_pmf(0.0, n, p) {
        return 0.0;
    }
    let probability = |k: f64| {
        if complement {
            sf(k, n, p)
        } else {
            cdf(k, n, p)
        }
    };
    let mut objective = |k: f64| {
        if complement {
            target - probability(k)
        } else {
            probability(k) - target
        }
    };
    do_inverse_discrete_quantile(&mut objective, n, guess.ceil(), factor)
        .and_then(|root| round_to_ceil(root, n, target, complement, probability))
        .unwrap_or(f64::NAN)
}

/// SciPy's `binom_ppf`: the smallest number of successes whose cumulative probability reaches
/// `q`. Boost's result outside `[0, n]` becomes NaN.
pub(super) fn binom_ppf(q: f64, n: f64, p: f64) -> f64 {
    if q.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n < 0.0 || !(0.0..=1.0).contains(&p) || !(0.0..=1.0).contains(&q) {
        return f64::NAN;
    }
    let k = quantile(n, p, q, 1.0 - q, false);
    if k < 0.0 || k > n {
        return f64::NAN;
    }
    k
}

/// SciPy's `binom_isf`: the smallest number of successes whose survival probability falls to
/// `q`.
pub(super) fn binom_isf(q: f64, n: f64, p: f64) -> f64 {
    quantile(n, p, 1.0 - q, q, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_invert_the_distribution_function_at_integers() {
        let (n, p) = (20.0, 0.3);
        for k in 0..20 {
            let k = f64::from(k);
            assert_eq!(binom_ppf(cdf(k, n, p), n, p), k);
            assert_eq!(binom_isf(sf(k, n, p), n, p), k);
        }
    }

    #[test]
    fn quantiles_keep_boosts_search_at_the_extremes() {
        // cdf(0) < 1e-300 <= cdf(1), but f(0) * f(1) underflows to -0, which Boost's bracketing
        // takes for no sign change, so it steps down to 0.
        assert!(cdf(0.0, 1000.0, 0.5) < 1e-300 && cdf(1.0, 1000.0, 0.5) >= 1e-300);
        assert_eq!(binom_ppf(1e-300, 1000.0, 0.5), 0.0);
        // Away from underflow the result is the smallest k with cdf(k) >= q.
        assert_eq!(binom_ppf(1e-250, 1000.0, 0.5), 26.0);
        assert!(cdf(25.0, 1000.0, 0.5) < 1e-250 && cdf(26.0, 1000.0, 0.5) >= 1e-250);
    }

    #[test]
    fn degenerate_probabilities_follow_boost() {
        assert_eq!(binom_pmf(0.0, 5.0, 0.0), 1.0);
        assert_eq!(binom_pmf(5.0, 5.0, 1.0), 1.0);
        assert_eq!(binom_cdf(2.0, 5.0, 1.0), 0.0);
        assert_eq!(binom_sf(2.0, 5.0, 0.0), 0.0);
        assert_eq!(binom_ppf(0.5, 5.0, 1.0), 5.0);
        assert!(binom_pmf(6.0, 5.0, 0.5).is_nan());
        assert!(binom_sf(f64::INFINITY, 5.0, 0.5).is_nan());
        assert_eq!(binom_cdf(f64::NEG_INFINITY, 5.0, 0.5), 0.0);
    }
}
