//! Binomial and Poisson draws.
//!
//! Binomial uses simple inversion (summing the binomial mass function against a uniform) when
//! `n*min(p, 1-p) <= 30`, and otherwise Kachitvichyanukul & Schmeiser's 1988 BTPE algorithm
//! ("Binomial Random Variate Generation", Communications of the ACM 31(2)): a rejection method
//! that proposes from a triangle-parallelogram-exponential-exponential majorizing shape fitted
//! to the binomial's mode, with a fast squeeze so the expensive exact term is rarely needed.
//!
//! Poisson uses direct multiplication of uniforms (Knuth's method: count uniforms until their
//! product drops below `exp(-lambda)`) when `lambda < 10`, and otherwise Hörmann's 1993 PTRS
//! algorithm ("The Transformed Rejection Method for Generating Poisson Random Variables",
//! Insurance: Mathematics and Economics 12(1)): transformed rejection from a fitted logistic-like
//! shape, with a squeeze and an exact log-probability test using [`log_gamma`] as the fallback.
//!
//! **Accuracy**: both BTPE and PTRS reproduce their papers' algorithms; see `docs/numpy.md` for
//! whether the large-parameter branches match NumPy 2.5.3's exact stream.

use super::bitgen::BitGen;

/// `ln(Gamma(x))` by the Stirling series, shifted up to `x >= 8` first via the recurrence
/// `Gamma(x) = Gamma(x+1)/x` so the asymptotic series converges to full `f64` precision.
pub(in crate::python) fn log_gamma(x: f64) -> f64 {
    let mut shift = 0.0;
    let mut x = x;
    while x < 8.0 {
        shift += x.ln();
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let series =
        inv * (1.0 / 12.0 + inv2 * (-1.0 / 360.0 + inv2 * (1.0 / 1260.0 + inv2 * (-1.0 / 1680.0))));
    (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln() + series - shift
}

/// `ln(k!)`.
fn log_factorial(k: i64) -> f64 {
    log_gamma(k as f64 + 1.0)
}

/// Binomial inversion (Kachitvichyanukul & Schmeiser's BINV): walk the cumulative mass function
/// from `x = 0`, matching a single uniform draw against the running probability.
fn binomial_inversion(bitgen: &mut BitGen, n: i64, p: f64) -> i64 {
    let q = 1.0 - p;
    let s = p / q;
    let a = (n as f64 + 1.0) * s;
    let mut r = q.powi(n as i32);
    let mut u = bitgen.next_double();
    let mut x = 0i64;
    loop {
        if u < r {
            return x;
        }
        u -= r;
        x += 1;
        r *= a / x as f64 - s;
        if x > n {
            return n;
        }
    }
}

/// Kachitvichyanukul & Schmeiser's BTPE, for `n * min(p, 1-p) > 30`.
#[allow(clippy::many_single_char_names)]
fn binomial_btpe(bitgen: &mut BitGen, n: i64, p: f64) -> i64 {
    let r = p.min(1.0 - p);
    let q = 1.0 - r;
    let flipped = p > 0.5;
    let nf = n as f64;
    let np = nf * r;
    let ffm = np + r;
    let m = ffm.floor();
    let fm = m;
    let npq = np * q;
    let p1 = (2.195 * npq.sqrt() - 4.6 * q).floor() + 0.5;
    let xm = fm + 0.5;
    let xl = xm - p1;
    let xr = xm + p1;
    let c = 0.134 + 20.5 / (15.3 + fm);
    let al = (ffm - xl) / (ffm - xl * r);
    let xll = al * (1.0 + 0.5 * al);
    let al = (xr - ffm) / (xr * q);
    let xlr = al * (1.0 + 0.5 * al);
    let p2 = p1 * (1.0 + c + c);
    let p3 = p2 + c / xll;
    let p4 = p3 + c / xlr;

    let result = loop {
        let u = bitgen.next_double() * p4;
        let mut v = bitgen.next_double();
        let ix: f64;
        if u <= p1 {
            break (xm - p1 * v + u).floor();
        } else if u <= p2 {
            let x = xl + (u - p1) / c;
            v = v * c + 1.0 - ((xm - x).abs() / p1);
            if !(0.0..=1.0).contains(&v) {
                continue;
            }
            ix = x.floor();
        } else if u <= p3 {
            ix = (xl + v.ln() / xll).floor();
            if ix < 0.0 {
                continue;
            }
            v *= (u - p2) * xll;
        } else {
            ix = (xr - v.ln() / xlr).floor();
            if ix > nf {
                continue;
            }
            v *= (u - p3) * xlr;
        }
        let k = (ix - m).abs();
        if k <= 20.0 || k >= npq / 2.0 - 1.0 {
            let mut f = 1.0;
            let ratio = r / q;
            let g = (nf + 1.0) * ratio;
            if m < ix {
                let mut i = m + 1.0;
                while i <= ix {
                    f *= g / i - ratio;
                    i += 1.0;
                }
            } else if m > ix {
                let mut i = ix + 1.0;
                while i <= m {
                    f /= g / i - ratio;
                    i += 1.0;
                }
            }
            if v <= f {
                break ix;
            }
            continue;
        }
        let amaxp = (k / npq) * ((k * (k / 3.0 + 0.625) + 0.1666666666666) / npq + 0.5);
        let ynorm = -k * k / (2.0 * npq);
        let alv = v.ln();
        if alv < ynorm - amaxp {
            break ix;
        }
        if alv > ynorm + amaxp {
            continue;
        }
        let x1 = ix + 1.0;
        let f1 = fm + 1.0;
        let z = nf + 1.0 - fm;
        let w = nf - ix + 1.0;
        let z2 = z * z;
        let x2 = x1 * x1;
        let f2 = f1 * f1;
        let w2 = w * w;
        let stirling = |t2: f64| -> f64 {
            (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / t2) / t2) / t2) / t2) / 166320.0
        };
        let t = xm * (f1 / x1).ln()
            + (nf - m + 0.5) * (z / w).ln()
            + (ix - m) * (w * r / (x1 * q)).ln()
            + stirling(f2) / f1
            + stirling(z2) / z
            + stirling(x2) / x1
            + stirling(w2) / w;
        if alv <= t {
            break ix;
        }
    };
    let result = result as i64;
    if flipped {
        n - result
    } else {
        result
    }
}

/// `binomial(n, p)`: `n` trials, each independently `True` with probability `p`.
///
/// `legacy` selects `RandomState`'s behavior at the degenerate `n == 0` or `p` in `{0, 1}`
/// edges: legacy still runs the general algorithm (which happens to draw exactly one uniform
/// and immediately accept, since the degenerate cumulative mass function is `1` everywhere),
/// while `Generator` short-circuits without drawing anything. Both are NumPy's actual observed
/// behavior, not a free choice: a `Generator` draw before and after a degenerate `binomial` call
/// reads the same bit-generator words either way, but `RandomState`'s does not.
pub(in crate::python) fn binomial(bitgen: &mut BitGen, n: i64, p: f64, legacy: bool) -> i64 {
    if !legacy {
        if n == 0 || p == 0.0 {
            return 0;
        }
        if p == 1.0 {
            return n;
        }
    }
    let mean_side = n as f64 * p.min(1.0 - p);
    if mean_side <= 30.0 {
        if p <= 0.5 {
            binomial_inversion(bitgen, n, p)
        } else {
            n - binomial_inversion(bitgen, n, 1.0 - p)
        }
    } else {
        binomial_btpe(bitgen, n, p)
    }
}

/// Knuth's multiplication method, for `lambda < 10`.
fn poisson_mult(bitgen: &mut BitGen, lam: f64) -> i64 {
    let enlam = (-lam).exp();
    let mut x = 0i64;
    let mut prod = 1.0;
    loop {
        prod *= bitgen.next_double();
        if prod <= enlam {
            return x;
        }
        x += 1;
    }
}

/// Hörmann's PTRS, for `lambda >= 10`.
fn poisson_ptrs(bitgen: &mut BitGen, lam: f64) -> i64 {
    let b = 0.931 + 2.53 * lam.sqrt();
    let a = -0.059 + 0.02483 * b;
    let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = bitgen.next_double() - 0.5;
        let v = bitgen.next_double();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + lam + 0.43).floor();
        if us >= 0.07 && v <= vr {
            return k as i64;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        let lhs = v.ln() + inv_alpha.ln() - (a / (us * us) + b).ln();
        let rhs = -lam + k * lam.ln() - log_factorial(k as i64);
        if lhs <= rhs {
            return k as i64;
        }
    }
}

/// `poisson(lam)`.
pub(in crate::python) fn poisson(bitgen: &mut BitGen, lam: f64) -> i64 {
    if lam < 10.0 {
        poisson_mult(bitgen, lam)
    } else {
        poisson_ptrs(bitgen, lam)
    }
}
