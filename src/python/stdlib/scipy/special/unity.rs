//! Cephes' `log1p`, `expm1`, `log1p(x) - x`, and `log(gamma(1 + x))`.
//!
//! SciPy's Cephes routines call these rather than the C library's versions, so the ports use
//! them too to reproduce SciPy's rounding. Ported from `xsf/cephes/unity.h`.
#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

use super::gamma::{lgam, zeta, MACHEP};
use super::poly::{p1evl, polevl};

const UNITY_LP: [f64; 7] = [
    4.5270000862445199635215E-5,
    4.9854102823193375972212E-1,
    6.5787325942061044846969E0,
    2.9911919328553073277375E1,
    6.0949667980987787057556E1,
    5.7112963590585538103336E1,
    2.0039553499201281259648E1,
];
const UNITY_LQ: [f64; 6] = [
    1.5062909083469192043167E1,
    8.3047565967967209469434E1,
    2.2176239823732856465394E2,
    3.0909872225312059774938E2,
    2.1642788614495947685003E2,
    6.0118660497603843919306E1,
];

/// `log(1 + x)`.
pub(super) fn log1p(x: f64) -> f64 {
    let z = 1.0 + x;
    if !(FRAC_1_SQRT_2..=SQRT_2).contains(&z) {
        return z.ln();
    }
    let z = x * x;
    let z = -0.5 * z + x * (z * polevl(x, &UNITY_LP) / p1evl(x, &UNITY_LQ));
    x + z
}

/// `log(1 + x) - x`.
pub(super) fn log1pmx(x: f64) -> f64 {
    if x.abs() >= 0.5 {
        return log1p(x) - x;
    }
    let mut xfac = x;
    let mut result = 0.0;
    for n in 2..500 {
        xfac *= -x;
        let term = xfac / f64::from(n);
        result += term;
        if term.abs() < MACHEP * result.abs() {
            break;
        }
    }
    result
}

const UNITY_EP: [f64; 3] = [
    1.2617719307481059087798E-4,
    3.0299440770744196129956E-2,
    9.9999999999999999991025E-1,
];
const UNITY_EQ: [f64; 4] = [
    3.0019850513866445504159E-6,
    2.5244834034968410419224E-3,
    2.2726554820815502876593E-1,
    2.0000000000000000000897E0,
];

/// `exp(x) - 1`.
pub(super) fn expm1(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_nan() || x > 0.0 { x } else { -1.0 };
    }
    if !(-0.5..=0.5).contains(&x) {
        return x.exp() - 1.0;
    }
    let xx = x * x;
    let r = x * polevl(xx, &UNITY_EP);
    let r = r / (polevl(xx, &UNITY_EQ) - r);
    r + r
}

/// `log(gamma(x + 1))` near `x = 0` from its Taylor series.
fn lgam1p_taylor(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let mut result = -0.577215664901532860606512090082402431 * x;
    let mut xfac = -x;
    for n in 2..42 {
        xfac *= -x;
        let coefficient = zeta(f64::from(n), 1.0) * xfac / f64::from(n);
        result += coefficient;
        if coefficient.abs() < MACHEP * result.abs() {
            break;
        }
    }
    result
}

/// `log(gamma(x + 1))`.
pub(super) fn lgam1p(x: f64) -> f64 {
    if x.abs() <= 0.5 {
        lgam1p_taylor(x)
    } else if (x - 1.0).abs() < 0.5 {
        x.ln() + lgam1p_taylor(x - 1.0)
    } else {
        lgam(x + 1.0)
    }
}
