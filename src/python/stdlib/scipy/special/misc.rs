//! Elementary special functions with no series or iteration: the logistic family (`expit`,
//! `logit`, `log_expit`), the information-theory family (`xlogy`, `xlog1py`, `entr`, `rel_entr`,
//! `kl_div`), the Box-Cox transform, and `scipy.special`'s own `expm1`/`log1p` (registered
//! separately from NumPy's ufuncs of the same name; see `mod.rs`).
//!
//! Each is a direct closed-form expression, written in whichever algebraically equivalent form
//! avoids cancellation (`expit`/`log_expit` branch on the sign of `x` so `exp` never overflows;
//! `xlogy`/`rel_entr` special-case their `0 * ln(...)` boundary instead of computing `NaN`).

/// `expit(x) = 1 / (1 + exp(-x))`, branched on the sign of `x` so the exponential never
/// overflows.
pub(in crate::python) fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `logit(x) = ln(x / (1 - x))`.
pub(in crate::python) fn logit(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == 1.0 {
        return f64::INFINITY;
    }
    x.ln() - (1.0 - x).ln()
}

/// `log_expit(x) = ln(expit(x))`, branched on the sign of `x` so it stays accurate for large
/// `|x|` instead of taking the log of an `expit` result that has already saturated to `0` or
/// `1`.
pub(in crate::python) fn log_expit(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

/// `xlogy(x, y) = x ln(y)`, with `xlogy(0, y) = 0` even where `ln(y)` is `-inf`, except
/// `xlogy(0, nan) = nan`.
pub(in crate::python) fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        if y.is_nan() {
            f64::NAN
        } else {
            0.0
        }
    } else {
        x * y.ln()
    }
}

/// `xlog1py(x, y) = x ln(1 + y)`, with the same `x == 0` convention as [`xlogy`].
pub(in crate::python) fn xlog1py(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        if y.is_nan() {
            f64::NAN
        } else {
            0.0
        }
    } else {
        x * y.ln_1p()
    }
}

/// `entr(x) = -x ln(x)` for `x > 0`, continued to `entr(0) = 0` and `entr(x) = -inf` for `x <
/// 0`.
pub(in crate::python) fn entr(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == 0.0 {
        return 0.0;
    }
    -(x * x.ln())
}

/// `rel_entr(x, y) = x ln(x / y)` for `x, y > 0`, continued to `rel_entr(0, y) = 0` for `y >= 0`
/// and `rel_entr(x, y) = inf` for `x > 0`, `y <= 0`.
pub(in crate::python) fn rel_entr(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if y <= 0.0 {
        return if x > 0.0 { f64::INFINITY } else { f64::NAN };
    }
    x * (x / y).ln()
}

/// `kl_div(x, y) = rel_entr(x, y) - x + y`.
pub(in crate::python) fn kl_div(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    rel_entr(x, y) - x + y
}

/// `boxcox(x, lmbda) = (x^lmbda - 1) / lmbda`, continued to `ln(x)` at `lmbda == 0`.
pub(in crate::python) fn boxcox(x: f64, lmbda: f64) -> f64 {
    if lmbda == 0.0 {
        x.ln()
    } else {
        (lmbda * x.ln()).exp_m1() / lmbda
    }
}

/// `inv_boxcox(y, lmbda)`, the inverse of [`boxcox`]: `exp(y)` at `lmbda == 0`, else
/// `(lmbda y + 1)^(1/lmbda)`.
pub(in crate::python) fn inv_boxcox(y: f64, lmbda: f64) -> f64 {
    if lmbda == 0.0 {
        y.exp()
    } else {
        (lmbda * y + 1.0).powf(1.0 / lmbda)
    }
}

/// `scipy.special.expm1`: `exp(x) - 1`, accurate near `x = 0`.
pub(in crate::python) fn expm1(x: f64) -> f64 {
    x.exp_m1()
}

/// `scipy.special.log1p`: `ln(1 + x)`, accurate near `x = 0`.
pub(in crate::python) fn log1p(x: f64) -> f64 {
    x.ln_1p()
}
