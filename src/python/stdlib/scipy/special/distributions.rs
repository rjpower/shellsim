//! Distribution CDFs, survival functions and quantiles built on the incomplete gamma and beta
//! functions: the chi-square (`chdtr*`), F (`fdtr*`), Student's t (`stdtr*`) and Poisson
//! (`pdtr*`) families from [`super::igam`] and [`super::ibeta`], and the binomial family
//! (`bdtr`, `bdtrc`, and the private `_binom_*` ufuncs `scipy.stats` calls) from
//! [`super::ibeta`] alone. Every one of these is a standard textbook identity relating a
//! distribution function to a regularized incomplete gamma or beta value; none has its own
//! series or continued fraction.
//!
//! `pdtrik` and the binomial quantiles (`_binom_ppf`, `_binom_isf`) invert a distribution
//! function with respect to a *shape* parameter (the Poisson rate, or the number of successes)
//! rather than the usual variable, so they bisect directly rather than reusing
//! `gammaincinv`/`betaincinv` (which invert with respect to the incomplete function's own `x`).

use super::ibeta;
use super::igam;
use super::meter::tick;

/// `chdtr(df, x) = P(df/2, x/2)`, the chi-square CDF.
pub(in crate::python) fn chdtr(df: f64, x: f64) -> f64 {
    if df.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if df <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    igam::gammainc(df / 2.0, x / 2.0)
}

/// `chdtrc(df, x) = Q(df/2, x/2)`, the chi-square survival function.
pub(in crate::python) fn chdtrc(df: f64, x: f64) -> f64 {
    if df.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if df <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    igam::gammaincc(df / 2.0, x / 2.0)
}

/// `chdtri(df, p)`: solve `chdtrc(df, x) = p` for `x`.
pub(in crate::python) fn chdtri(df: f64, p: f64) -> f64 {
    if df.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if df <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    2.0 * igam::gammainccinv(df / 2.0, p)
}

/// `fdtr(dfn, dfd, x) = I_z(dfn/2, dfd/2)`, `z = dfn x / (dfn x + dfd)`, the F CDF.
pub(in crate::python) fn fdtr(dfn: f64, dfd: f64, x: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let z = dfn * x / (dfn * x + dfd);
    ibeta::betainc(dfn / 2.0, dfd / 2.0, z)
}

/// `fdtrc(dfn, dfd, x) = 1 - fdtr(dfn, dfd, x)`, the F survival function.
pub(in crate::python) fn fdtrc(dfn: f64, dfd: f64, x: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    let z = dfn * x / (dfn * x + dfd);
    ibeta::betaincc(dfn / 2.0, dfd / 2.0, z)
}

/// `fdtri(dfn, dfd, p)`: solve `fdtr(dfn, dfd, x) = p` for `x`.
pub(in crate::python) fn fdtri(dfn: f64, dfd: f64, p: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let z = ibeta::betaincinv(dfn / 2.0, dfd / 2.0, p);
    dfd * z / (dfn * (1.0 - z))
}

/// `stdtr(df, t)`, the Student's t CDF, via `x = df / (df + t^2)` and the incomplete beta
/// function (Abramowitz & Stegun 26.7.1).
pub(in crate::python) fn stdtr(df: f64, t: f64) -> f64 {
    if df.is_nan() || t.is_nan() {
        return f64::NAN;
    }
    if df <= 0.0 {
        return f64::NAN;
    }
    if t.is_infinite() {
        return if t > 0.0 { 1.0 } else { 0.0 };
    }
    let x = df / (df + t * t);
    let half = ibeta::betainc(df / 2.0, 0.5, x);
    if t >= 0.0 {
        1.0 - 0.5 * half
    } else {
        0.5 * half
    }
}

/// `stdtrit(df, p)`: solve `stdtr(df, t) = p` for `t`.
pub(in crate::python) fn stdtrit(df: f64, p: f64) -> f64 {
    if df.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if df <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let (target, sign) = if p < 0.5 {
        (2.0 * p, -1.0)
    } else {
        (2.0 * (1.0 - p), 1.0)
    };
    let x = ibeta::betaincinv(df / 2.0, 0.5, target);
    sign * (df * (1.0 - x) / x).sqrt()
}

/// `pdtr(k, m) = Q(k+1, m)`, the Poisson CDF continued to real `k`.
pub(in crate::python) fn pdtr(k: f64, m: f64) -> f64 {
    if k.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if k <= -1.0 || m < 0.0 {
        return f64::NAN;
    }
    igam::gammaincc(k + 1.0, m)
}

/// `pdtrc(k, m) = P(k+1, m) = 1 - pdtr(k, m)`.
pub(in crate::python) fn pdtrc(k: f64, m: f64) -> f64 {
    if k.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if k <= -1.0 || m < 0.0 {
        return f64::NAN;
    }
    igam::gammainc(k + 1.0, m)
}

/// Solve `gammaincc(a, x) = target` for `a > 0`, with `x` fixed. `gammaincc` is increasing in
/// `a` for fixed `x` (raising the shape parameter of a gamma distribution shifts its mass past a
/// fixed point), so this is an ordinary bisection, bracketed by doubling search.
fn invert_shape(x: f64, target: f64) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    while igam::gammaincc(hi, x) < target {
        lo = hi;
        hi *= 2.0;
        if !tick() {
            break;
        }
    }
    let mut a = 0.5 * (lo + hi);
    loop {
        let value = igam::gammaincc(a, x);
        if value < target {
            lo = a;
        } else {
            hi = a;
        }
        if hi - lo <= hi.abs().max(1.0) * 1e-14 || !tick() {
            return 0.5 * (lo + hi);
        }
        a = 0.5 * (lo + hi);
    }
}

/// `pdtrik(p, m)`: solve `pdtr(k, m) = p` for `k`, continued to real `k`. Poisson rates below
/// the smallest representable root clamp `k` at `0`, as SciPy's does.
pub(in crate::python) fn pdtrik(p: f64, m: f64) -> f64 {
    if p.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if !(0.0..1.0).contains(&p) || m < 0.0 {
        return f64::NAN;
    }
    if m == 0.0 || p == 0.0 {
        return 0.0;
    }
    (invert_shape(m, p) - 1.0).max(0.0)
}

/// `bdtr(k, n, p) = I_{1-p}(n-k, k+1)`, the binomial CDF continued to real `k`
/// (Abramowitz & Stegun 26.5.24).
pub(in crate::python) fn bdtr(k: f64, n: f64, p: f64) -> f64 {
    if k.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if k == f64::NEG_INFINITY {
        return 0.0;
    }
    if k >= n {
        return 1.0;
    }
    ibeta::betainc(n - k, k + 1.0, 1.0 - p)
}

/// `bdtrc(k, n, p) = I_p(k+1, n-k) = 1 - bdtr(k, n, p)`.
pub(in crate::python) fn bdtrc(k: f64, n: f64, p: f64) -> f64 {
    if k.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if k == f64::NEG_INFINITY {
        return 1.0;
    }
    if k >= n {
        return 0.0;
    }
    ibeta::betainc(k + 1.0, n - k, p)
}

/// `binom_pmf(k, n, p) = C(n, k) p^k (1-p)^(n-k)`, computed directly (not as a difference of
/// `bdtr` values, which loses precision for small probabilities).
pub(in crate::python) fn binom_pmf(k: f64, n: f64, p: f64) -> f64 {
    if k.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n < 0.0 || !(0.0..=1.0).contains(&p) || k < 0.0 || k > n {
        return f64::NAN;
    }
    super::gamma::binom(n, k) * p.powf(k) * (1.0 - p).powf(n - k)
}

/// Smallest non-negative integer `k <= n` with `f(k) >= target`, by binary search. `f` is
/// non-decreasing in `k` (a CDF).
fn smallest_integer_at_least(n: f64, target: f64, f: impl Fn(f64) -> f64) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = n;
    while lo < hi {
        let mid = ((lo + hi) / 2.0).floor();
        if f(mid) < target {
            lo = mid + 1.0;
        } else {
            hi = mid;
        }
        if !tick() {
            break;
        }
    }
    lo
}

/// `_binom_ppf(q, n, p)`: the smallest integer `k` with `bdtr(k, n, p) >= q`.
pub(in crate::python) fn binom_ppf(q: f64, n: f64, p: f64) -> f64 {
    if q.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n < 0.0 || !(0.0..=1.0).contains(&p) || !(0.0..=1.0).contains(&q) {
        return f64::NAN;
    }
    if q == 0.0 {
        return 0.0;
    }
    smallest_integer_at_least(n, q, |k| bdtr(k, n, p))
}

/// `_binom_isf(q, n, p) = _binom_ppf(1 - q, n, p)`: the smallest integer `k` with
/// `bdtrc(k, n, p) <= q`.
pub(in crate::python) fn binom_isf(q: f64, n: f64, p: f64) -> f64 {
    if q.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    binom_ppf(1.0 - q, n, p)
}
