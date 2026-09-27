//! Distribution functions built on the incomplete beta and gamma functions: Student's t
//! (`stdtr`, `stdtrit`), F (`fdtr`, `fdtrc`, `fdtri`), chi-square (`chdtr`, `chdtrc`, `chdtri`),
//! Poisson (`pdtr`, `pdtrc`) and binomial (`bdtr`, `bdtrc`).
//!
//! SciPy takes the t and F functions from Boost's distributions and the rest from Cephes. The
//! formulas and domain checks here follow those sources, including SciPy's handling of Boost's
//! errors.

use super::ibeta::{betainc, betaincc};
use super::ibeta_inv::{ibeta_inv, students_t_quantile};
use super::igam::{igam, igamc, igamci};

/// Student's t cumulative distribution function with `df` degrees of freedom at `t`.
pub(super) fn stdtr(df: f64, t: f64) -> f64 {
    if df.is_nan() || t.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    if t.is_infinite() {
        return if t > 0.0 { 1.0 } else { 0.0 };
    }
    if t == 0.0 {
        return 0.5;
    }
    if df > 1.0 / f64::EPSILON {
        return super::erf::ndtr(t);
    }
    // Boost's choice between I_z(df/2, 1/2) and its complement keeps z away from 1.
    let t2 = t * t;
    let tail = if df > 2.0 * t2 {
        betaincc(0.5, df / 2.0, t2 / (df + t2)) / 2.0
    } else {
        betainc(df / 2.0, 0.5, df / (df + t2)) / 2.0
    };
    let result = if t > 0.0 { 1.0 - tail } else { tail };
    if (0.0..=1.0).contains(&result) {
        result
    } else {
        f64::NAN
    }
}

/// The inverse of [`stdtr`] in `t`: the `p` quantile of Student's t with `df` degrees of freedom.
/// Like SciPy, `p = 0` gives `+inf`, since SciPy maps Boost's overflow error to `+inf` whatever
/// its sign.
pub(super) fn stdtrit(df: f64, p: f64) -> f64 {
    if df.is_nan() || p.is_nan() || df <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    students_t_quantile(df, p).unwrap_or(f64::INFINITY)
}

/// Parameters outside the F distribution's domain, or a NaN argument.
fn f_domain_error(dfn: f64, dfd: f64, x: f64) -> bool {
    dfn.is_nan() || dfd.is_nan() || x.is_nan() || dfn <= 0.0 || dfd <= 0.0 || x < 0.0
}

/// The F distribution's cumulative distribution function at `x`.
pub(super) fn fdtr(dfn: f64, dfd: f64, x: f64) -> f64 {
    if f_domain_error(dfn, dfd, x) {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 1.0;
    }
    let v1x = dfn * x;
    if v1x > dfd {
        betaincc(dfd / 2.0, dfn / 2.0, dfd / (dfd + v1x))
    } else {
        betainc(dfn / 2.0, dfd / 2.0, v1x / (dfd + v1x))
    }
}

/// The F distribution's survival function at `x`.
pub(super) fn fdtrc(dfn: f64, dfd: f64, x: f64) -> f64 {
    if f_domain_error(dfn, dfd, x) {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let v1x = dfn * x;
    if v1x > dfd {
        betainc(dfd / 2.0, dfn / 2.0, dfd / (dfd + v1x))
    } else {
        betaincc(dfn / 2.0, dfd / 2.0, v1x / (dfd + v1x))
    }
}

/// The `p` quantile of the F distribution.
pub(super) fn fdtri(dfn: f64, dfd: f64, p: f64) -> f64 {
    if f_domain_error(dfn, dfd, p) || p > 1.0 || dfn.is_infinite() || dfd.is_infinite() {
        return f64::NAN;
    }
    // The quantile is dfd x / (dfn y) where I_x(dfn/2, dfd/2) = p and y = 1 - x, which the
    // inverse computes without cancellation.
    match ibeta_inv(dfn / 2.0, dfd / 2.0, p, true) {
        Some((x, y)) => {
            let quantile = dfd * x / (dfn * y);
            if quantile < 0.0 {
                f64::NAN
            } else {
                quantile
            }
        }
        None => f64::NAN,
    }
}

/// The chi-square cumulative distribution function with `df` degrees of freedom.
pub(super) fn chdtr(df: f64, x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    igam(df / 2.0, x / 2.0)
}

/// The chi-square survival function.
pub(super) fn chdtrc(df: f64, x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    igamc(df / 2.0, x / 2.0)
}

/// The inverse of [`chdtrc`] in `x`.
pub(super) fn chdtri(df: f64, y: f64) -> f64 {
    if !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    2.0 * igamci(0.5 * df, y)
}

/// The Poisson cumulative distribution function: the probability of at most `k` events at
/// rate `m`. Non-integer `k` rounds down.
pub(super) fn pdtr(k: f64, m: f64) -> f64 {
    if k < 0.0 || m < 0.0 {
        return f64::NAN;
    }
    if m == 0.0 {
        return 1.0;
    }
    igamc(k.floor() + 1.0, m)
}

/// The Poisson survival function, the complement of [`pdtr`].
pub(super) fn pdtrc(k: f64, m: f64) -> f64 {
    if k < 0.0 || m < 0.0 {
        return f64::NAN;
    }
    if m == 0.0 {
        return 0.0;
    }
    igam(k.floor() + 1.0, m)
}

/// The number of trials as Cephes receives it: a C `int`, so larger values wrap as the
/// conversion from `long` does.
fn trials(n: f64) -> Option<i32> {
    #[allow(clippy::cast_possible_truncation)]
    n.is_finite().then_some(n as i64 as i32)
}

/// The binomial cumulative distribution function: the probability of at most `k` successes in
/// `n` trials with success probability `p`. Non-integer `k` rounds down.
pub(super) fn bdtr(k: f64, n: f64, p: f64) -> f64 {
    let Some(n) = trials(n) else {
        return f64::NAN;
    };
    let fk = k.floor();
    if p.is_nan() || k.is_nan() || !(0.0..=1.0).contains(&p) || fk < 0.0 || f64::from(n) < fk {
        return f64::NAN;
    }
    if fk == f64::from(n) {
        return 1.0;
    }
    let dn = f64::from(n) - fk;
    if fk == 0.0 {
        (1.0 - p).powf(dn)
    } else {
        betainc(dn, fk + 1.0, 1.0 - p)
    }
}

/// The binomial survival function: the probability of more than `k` successes.
pub(super) fn bdtrc(k: f64, n: f64, p: f64) -> f64 {
    let Some(n) = trials(n) else {
        return f64::NAN;
    };
    let fk = k.floor();
    if p.is_nan() || k.is_nan() || !(0.0..=1.0).contains(&p) || f64::from(n) < fk {
        return f64::NAN;
    }
    if fk < 0.0 {
        return 1.0;
    }
    if fk == f64::from(n) {
        return 0.0;
    }
    let dn = f64::from(n) - fk;
    if k == 0.0 {
        if p < 0.01 {
            -(dn * (-p).ln_1p()).exp_m1()
        } else {
            1.0 - (1.0 - p).powf(dn)
        }
    } else {
        betainc(fk + 1.0, dn, p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        let error = ((actual - expected) / expected).abs();
        assert!(
            error < 1e-13,
            "{actual} != {expected} (relative error {error})"
        );
    }

    #[test]
    fn t_quantile_inverts_the_cdf_on_both_branches() {
        for (df, p) in [
            (3.0, 0.975),
            (3.0, 1e-10),
            (10.0, 0.6),
            (0.5, 0.3),
            (25.0, 0.5001),
        ] {
            close(stdtr(df, stdtrit(df, p)), p);
        }
    }

    #[test]
    fn f_quantile_inverts_the_cdf_on_both_branches() {
        for (dfn, dfd, p) in [
            (3.0, 7.0, 0.95),
            (3.0, 7.0, 0.05),
            (1.0, 1.0, 0.999),
            (40.0, 2.5, 1e-9),
        ] {
            close(fdtr(dfn, dfd, fdtri(dfn, dfd, p)), p);
        }
    }

    #[test]
    fn domain_errors_are_nan_and_limits_are_exact() {
        assert!(stdtr(0.0, 1.0).is_nan());
        assert_eq!(stdtrit(4.0, 0.0), f64::INFINITY);
        assert_eq!(fdtr(2.0, 3.0, f64::INFINITY), 1.0);
        assert!(pdtr(-1.0, 2.0).is_nan());
        assert_eq!(pdtr(3.0, 0.0), 1.0);
        assert!(bdtr(2.0, f64::INFINITY, 0.5).is_nan());
        assert!(bdtr(6.0, 5.0, 0.5).is_nan());
        assert_eq!(bdtrc(-1.0, 5.0, 0.5), 1.0);
    }
}
