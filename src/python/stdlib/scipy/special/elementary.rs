//! Closed-form special functions: the logistic family (`expit`, `logit`, `log_expit`),
//! `xlogy` and `xlog1py`, the information-theory functions (`entr`, `rel_entr`, `kl_div`), the
//! Box-Cox transform and its inverse, and the binomial coefficient.
//!
//! These follow SciPy's xsf templates and Cython sources. The templated functions compute in
//! the loop's precision, so their `float32` versions round every step to single precision as
//! SciPy's do; the rest compute in `f64`.

use std::f64::consts::PI;

use super::gamma::{beta, gamma, lbeta};
use super::unity::log1p;

macro_rules! logistic {
    ($float:ty, $expit:ident, $logit:ident, $log_expit:ident) => {
        /// The logistic sigmoid `1 / (1 + exp(-x))`.
        pub(super) fn $expit(x: $float) -> $float {
            1.0 / (1.0 + (-x).exp())
        }

        /// The inverse of the logistic sigmoid, `log(x / (1 - x))`. Near `x = 1/2` the
        /// difference of `log1p` terms avoids cancellation.
        pub(super) fn $logit(x: $float) -> $float {
            if !(0.3..=0.65).contains(&x) {
                (x / (1.0 - x)).ln()
            } else {
                let s = 2.0 * (x - 0.5);
                s.ln_1p() - (-s).ln_1p()
            }
        }

        /// `log(expit(x))`, written so that neither tail overflows or cancels.
        pub(super) fn $log_expit(x: $float) -> $float {
            if x < 0.0 {
                x - x.exp().ln_1p()
            } else {
                -(-x).exp().ln_1p()
            }
        }
    };
}

logistic!(f64, expit, logit, log_expit);
logistic!(f32, expit_f32, logit_f32, log_expit_f32);

/// `x * log(y)`, defined as 0 when `x` is 0 and `y` is not NaN.
pub(super) fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 && !y.is_nan() {
        return 0.0;
    }
    x * y.ln()
}

pub(super) fn xlogy_f32(x: f32, y: f32) -> f32 {
    if x == 0.0 && !y.is_nan() {
        return 0.0;
    }
    x * y.ln()
}

/// `x * log1p(y)`, defined as 0 when `x` is 0 and `y` is not NaN. SciPy takes `log1p` from
/// Cephes in both precisions.
pub(super) fn xlog1py(x: f64, y: f64) -> f64 {
    if x == 0.0 && !y.is_nan() {
        return 0.0;
    }
    x * log1p(y)
}

pub(super) fn xlog1py_f32(x: f32, y: f32) -> f32 {
    if x == 0.0 && !y.is_nan() {
        return 0.0;
    }
    #[allow(clippy::cast_possible_truncation)]
    let log = log1p(f64::from(y)) as f32;
    x * log
}

/// The elementwise entropy `-x log(x)`, 0 at 0 and `-inf` for negative `x`.
pub(super) fn entr(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else if x > 0.0 {
        -x * x.ln()
    } else if x == 0.0 {
        0.0
    } else {
        f64::NEG_INFINITY
    }
}

/// The Kullback-Leibler divergence term `x log(x / y) - x + y`.
pub(super) fn kl_div(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if x > 0.0 && y > 0.0 {
        x * (x / y).ln() - x + y
    } else if x == 0.0 && y >= 0.0 {
        y
    } else {
        f64::INFINITY
    }
}

/// The relative entropy term `x log(x / y)`.
pub(super) fn rel_entr(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 || y <= 0.0 {
        return if x == 0.0 && y >= 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
    }
    let ratio = x / y;
    if 0.5 < ratio && ratio < 2.0 {
        // log1p keeps precision when x and y are close.
        x * ((x - y) / y).ln_1p()
    } else if f64::MIN_POSITIVE < ratio && ratio < f64::INFINITY {
        x * ratio.ln()
    } else {
        // The ratio overflowed or lost precision; take logarithms first.
        x * (x.ln() - y.ln())
    }
}

/// The Box-Cox transform `(x^lmbda - 1) / lmbda`, which is `log(x)` at `lmbda = 0`.
pub(super) fn boxcox(x: f64, lmbda: f64) -> f64 {
    // Below about 3e-19, lmbda * log(x) is smaller than the precision expm1 can resolve.
    if lmbda.abs() < 1e-19 {
        x.ln()
    } else if lmbda * x.ln() < 709.78 {
        (lmbda * x.ln()).exp_m1() / lmbda
    } else {
        1f64.copysign(lmbda) * (lmbda * x.ln() - lmbda.abs().ln()).exp() - 1.0 / lmbda
    }
}

/// The inverse of [`boxcox`] in `x`.
pub(super) fn inv_boxcox(x: f64, lmbda: f64) -> f64 {
    if lmbda == 0.0 {
        x.exp()
    } else if lmbda * x < 1.79e308 {
        ((lmbda * x).ln_1p() / lmbda).exp()
    } else {
        (((1f64.copysign(lmbda) * (x + 1.0 / lmbda)).ln() + lmbda.abs().ln()) / lmbda).exp()
    }
}

/// The binomial coefficient `n choose k` for real `n` and `k`, xsf's `binom`.
pub(super) fn binom(n: f64, k: f64) -> f64 {
    if n < 0.0 && n == n.floor() {
        return f64::NAN;
    }
    let mut kx = k.floor();
    if k == kx && (n.abs() > 1e-8 || n == 0.0) {
        // Integer k: the product formula rounds less, and is exact when the result is an
        // integer. It loses precision for small nonzero n.
        let nx = n.floor();
        if nx == n && kx > nx / 2.0 && nx > 0.0 {
            kx = nx - kx;
        }
        if (0.0..20.0).contains(&kx) {
            let (mut num, mut den) = (1.0, 1.0);
            let mut i = 1.0;
            while i < 1.0 + kx {
                num *= i + n - kx;
                den *= i;
                if num.abs() > 1e50 {
                    num /= den;
                    den = 1.0;
                }
                i += 1.0;
            }
            return num / den;
        }
    }
    if n >= 1e10 * k && k > 0.0 {
        // Avoid overflowing or underflowing intermediate results.
        return (-lbeta(1.0 + n - k, 1.0 + k) - (n + 1.0).ln()).exp();
    }
    if k > 1e8 * n.abs() {
        // Asymptotic expansion in 1/k.
        let mut num = gamma(1.0 + n) / k.abs() + gamma(1.0 + n) * n / (2.0 * k * k);
        num /= PI * k.abs().powf(n);
        // xsf tests whether floor(k) survives a round trip through a C `int`.
        let kx = k.floor();
        if k > 0.0 {
            let (dk, sign) = if kx.abs() < 2_147_483_648.0 {
                #[allow(clippy::cast_possible_truncation)]
                let odd = (kx as i64) % 2 != 0;
                (k - kx, if odd { -1.0 } else { 1.0 })
            } else {
                (k, 1.0)
            };
            return num * ((dk - n) * PI).sin() * sign;
        }
        if kx.abs() < 2_147_483_648.0 {
            return 0.0;
        }
        return num * (k * PI).sin();
    }
    1.0 / (n + 1.0) / beta(1.0 + n - k, 1.0 + k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logit_inverts_expit_across_the_switch_points() {
        for x in [-30.0, -1.0, -0.2, 0.0, 0.5, 3.0] {
            let p = expit(x);
            assert!((logit(p) - x).abs() <= 1e-12 * x.abs().max(1.0), "{x}");
        }
        assert_eq!(logit(0.0), f64::NEG_INFINITY);
        assert_eq!(log_expit(-800.0), -800.0);
    }

    #[test]
    fn binom_uses_the_product_formula_for_integers() {
        assert_eq!(binom(10.0, 3.0), 120.0);
        assert_eq!(binom(10.0, 8.0), 45.0);
        assert!(binom(-3.0, 2.0).is_nan());
        assert!((binom(2.5, 1.5) - 2.5).abs() < 1e-14);
    }

    #[test]
    fn information_functions_have_their_boundary_values() {
        assert_eq!(entr(0.0), 0.0);
        assert_eq!(entr(-1.0), f64::NEG_INFINITY);
        assert_eq!(kl_div(0.0, 2.0), 2.0);
        assert_eq!(rel_entr(0.0, 2.0), 0.0);
        assert_eq!(rel_entr(1.0, 0.0), f64::INFINITY);
    }
}
