//! The gamma family: `gamma`, `rgamma`, `gammaln`, `loggamma`, `psi`/`digamma`, `beta`,
//! `betaln`, `poch` and `binom`.
//!
//! Every function reduces to one core, [`lanczos_lgamma`], the Lanczos approximation of
//! `ln Gamma(x)` for `x >= 0.5` (g = 7, n = 9; coefficients from C. Lanczos, "A Precision
//! Approximation of the Gamma Function", SIAM J. Numer. Anal. 1(1), 1964, in the widely
//! reproduced form used by Numerical Recipes and Wikipedia's "Lanczos approximation"). Euler's
//! reflection formula `Gamma(x) Gamma(1-x) = pi / sin(pi x)` extends it to `x < 0.5`, and the
//! sign of `sin(pi x)` there also gives the sign of `Gamma(x)` without separate parity
//! bookkeeping. `digamma` uses the standard recurrence-plus-asymptotic-series method (shift the
//! argument up with `psi(x+1) = psi(x) + 1/x` until it is large, then apply the Bernoulli-number
//! asymptotic expansion), reflected the same way for `x < 0.5`.
//!
//! `beta`, `poch` and `binom` are all ratios of gammas, computed in the log domain and then
//! exponentiated so that large arguments (`binom(50, 25)`, `beta(1e5, 3)`) do not overflow an
//! intermediate `Gamma` value that the final ratio would bring back into range.

use std::f64::consts::PI;

/// Lanczos g parameter and coefficients (g = 7, n = 9).
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_312e-7,
];

/// `ln Gamma(x)` for `x >= 0.5`, via the Lanczos series. Always finite and positive-argument
/// safe: it never evaluates `Gamma` itself, so it has no overflow issue for large `x` (the
/// result grows like `x ln x`, not like `Gamma(x)`).
fn lanczos_lgamma(x: f64) -> f64 {
    let g = x - 1.0;
    let mut sum = LANCZOS_COEFFICIENTS[0];
    for (index, coefficient) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        sum += coefficient / (g + index as f64);
    }
    let base = g + LANCZOS_G + 0.5;
    0.5 * (2.0 * PI).ln() + (g + 0.5) * base.ln() - base + sum.ln()
}

/// True when `x` is zero or a negative integer, where `Gamma` has a pole.
fn is_nonpositive_integer(x: f64) -> bool {
    x <= 0.0 && x == x.floor()
}

/// `ln |Gamma(x)|`, for any real `x` that is not a pole. Poles report `+inf` (matching
/// `gammaln`'s convention; callers that need `NaN` at negative-integer poles check
/// [`is_nonpositive_integer`] themselves).
fn lgamma_abs(x: f64) -> f64 {
    if is_nonpositive_integer(x) {
        return f64::INFINITY;
    }
    // `Gamma(1) = Gamma(2) = 1` exactly, so `ln` of either is exactly `0.0`; the general Lanczos
    // series lands a machine epsilon or two away from that for `x == 1.0` (`x == 2.0` happens to
    // round exactly, but not reliably so), and callers compare at `atol=0`.
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    if x >= 0.5 {
        return lanczos_lgamma(x);
    }
    let sin = (PI * x).sin().abs();
    (PI / sin).ln() - lanczos_lgamma(1.0 - x)
}

/// `Gamma(x)`. Poles: `gamma(0.0) = inf`, `gamma(-0.0) = -inf` (the signed-zero limit of `1/x`),
/// and `gamma(-1), gamma(-2), ...  = nan` (SciPy's convention: the two-sided pole has no signed
/// limit once `x` is away from zero).
pub(in crate::python) fn gamma(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0 / x;
    }
    // `Gamma(n) = (n - 1)!` is exact for a small positive integer `n`, computed directly as a
    // product; going through `exp(lanczos_lgamma(n))` instead does not reliably round-trip back
    // to the exact integer (the Lanczos series is only accurate to within a couple of ULPs), and
    // callers compare small factorials like `gamma(5) == 24.0` exactly. `171` bounds this to
    // where `Gamma` is still finite in `f64`.
    if x > 0.0 && x <= 171.0 && x == x.floor() {
        let mut result = 1.0;
        let mut k = 2.0;
        while k < x {
            result *= k;
            k += 1.0;
            if !super::meter::tick() {
                break;
            }
        }
        return result;
    }
    if x >= 0.5 {
        return lanczos_lgamma(x).exp();
    }
    // `sin(pi * x)` is never exactly `0.0` in floating point for a negative integer `x` (`pi` is
    // itself only an approximation), so the pole there needs its own check rather than relying on
    // that product underflowing to zero.
    if is_nonpositive_integer(x) {
        return f64::NAN;
    }
    let sin = (PI * x).sin();
    // Reflection formula `Gamma(x) = pi / (sin(pi x) Gamma(1 - x))`, i.e. the *ratio* `pi /
    // sin(pi x)` divided by `Gamma(1 - x)` -- not `exp(pi / sin(pi x))`, which would be a
    // different (and wildly wrong, for small `sin`) quantity.
    let magnitude = (PI / sin.abs()) / lanczos_lgamma(1.0 - x).exp();
    magnitude.copysign(sin)
}

/// `1 / Gamma(x)`, which is entire (finite everywhere, zero at the poles of `Gamma`).
pub(in crate::python) fn rgamma(x: f64) -> f64 {
    if is_nonpositive_integer(x) {
        return 0.0;
    }
    1.0 / gamma(x)
}

/// `ln |Gamma(x)|`, SciPy's `gammaln`. Poles (including `x == 0`) give `+inf`.
pub(in crate::python) fn gammaln(x: f64) -> f64 {
    lgamma_abs(x)
}

/// `ln Gamma(x)`, SciPy's `loggamma` real loop. Equal to `gammaln` where `Gamma(x) > 0`; `NaN`
/// where `Gamma(x) < 0` (the true value is complex, and this is the real loop); `+inf` at the
/// `x == 0` pole, matching `gammaln`; `NaN` at the other poles.
pub(in crate::python) fn loggamma(x: f64) -> f64 {
    if x >= 0.5 {
        return lanczos_lgamma(x);
    }
    if is_nonpositive_integer(x) {
        return if x == 0.0 { f64::INFINITY } else { f64::NAN };
    }
    if (PI * x).sin() > 0.0 {
        lgamma_abs(x)
    } else {
        f64::NAN
    }
}

/// Asymptotic (Bernoulli-number) series for `psi(x)` once `x` is large: DLMF 5.11.2,
/// `psi(x) ~ ln(x) - 1/(2x) - sum B_{2k} / (2k x^{2k})`.
fn digamma_asymptotic(x: f64) -> f64 {
    let inverse_square = 1.0 / (x * x);
    // B_2/2=1/12, B_4/4=-1/120, B_6/6=1/252, B_8/8=-1/240, B_10/10=1/132, B_12/12=691/32760.
    let series = inverse_square
        * (1.0 / 12.0
            + inverse_square
                * (-1.0 / 120.0
                    + inverse_square
                        * (1.0 / 252.0
                            + inverse_square * (-1.0 / 240.0 + inverse_square / 132.0))));
    x.ln() - 0.5 / x - series
}

/// `psi(x)` (digamma), SciPy's `psi`/`digamma`. Shifts `x` up by the recurrence
/// `psi(x) = psi(x+1) - 1/x` until the asymptotic series applies, then reflects for `x < 0.5`
/// using `psi(1-x) - psi(x) = pi cot(pi x)`.
pub(in crate::python) fn digamma(x: f64) -> f64 {
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if is_nonpositive_integer(x) {
        return f64::NAN;
    }
    if x < 0.5 {
        return digamma(1.0 - x) - PI / (PI * x).tan();
    }
    let mut shifted = x;
    let mut correction = 0.0;
    while shifted < 10.0 {
        correction += 1.0 / shifted;
        shifted += 1.0;
        if !super::meter::tick() {
            break;
        }
    }
    digamma_asymptotic(shifted) - correction
}

/// `B(a, b) = Gamma(a) Gamma(b) / Gamma(a + b)`, computed in the log domain so large arguments
/// (where the individual gammas would overflow but the ratio is representable) still work.
pub(in crate::python) fn beta(a: f64, b: f64) -> f64 {
    gamma_ratio(a, b, a + b)
}

/// `ln |B(a, b)|`.
pub(in crate::python) fn betaln(a: f64, b: f64) -> f64 {
    lgamma_abs(a) + lgamma_abs(b) - lgamma_abs(a + b)
}

/// `Gamma(p) Gamma(q) / Gamma(r)`, the shared shape of `beta`, `poch` and `binom`. Any pole in
/// the numerator wins over a pole in the denominator (the ratio is infinite); a pole in the
/// denominator alone makes the ratio zero; poles in both make it indeterminate (`NaN`).
fn gamma_ratio(p: f64, q: f64, r: f64) -> f64 {
    let p_pole = is_nonpositive_integer(p);
    let q_pole = is_nonpositive_integer(q);
    let r_pole = is_nonpositive_integer(r);
    if (p_pole || q_pole) && r_pole {
        return f64::NAN;
    }
    if p_pole || q_pole {
        return f64::INFINITY * gamma_sign(p) * gamma_sign(q);
    }
    if r_pole {
        return 0.0;
    }
    // When every `Gamma` value involved is safely within range (so none of them overflows or
    // underflows to `0`), a direct ratio is more accurate than the general log-domain path below
    // -- exact, even, for the common case of small integer arguments (`poch`, `binom` on modest
    // inputs), where `exp(lgamma_abs(p) + ... )` accumulates enough rounding to land a couple of
    // ULPs off an exact integer ratio.
    if p.abs() <= 171.0 && q.abs() <= 171.0 && r.abs() <= 171.0 {
        let denominator = gamma(r);
        if denominator != 0.0 {
            return gamma(p) * gamma(q) / denominator;
        }
    }
    let magnitude = (lgamma_abs(p) + lgamma_abs(q) - lgamma_abs(r)).exp();
    magnitude * gamma_sign(p) * gamma_sign(q) * gamma_sign(r)
}

/// `+1` or `-1`, matching the sign convention `gamma` uses (the sign of `sin(pi x)` for
/// `x < 0.5`, and always positive for `x >= 0.5`). Callers only reach this away from poles.
fn gamma_sign(x: f64) -> f64 {
    if x >= 0.5 {
        1.0
    } else {
        (PI * x).sin().signum()
    }
}

/// `(x)_m = Gamma(x + m) / Gamma(x)`, the Pochhammer (rising factorial) symbol. `m == 0` is
/// always `1`, even where `x` is itself a pole (the empty product convention).
pub(in crate::python) fn poch(x: f64, m: f64) -> f64 {
    if m == 0.0 {
        return 1.0;
    }
    gamma_ratio(x + m, 1.0, x)
}

/// `C(n, k)` for a non-negative integer `k` and any `n` with `Gamma(n + 1)` finite, as the
/// falling-factorial product `prod_{i=0}^{k-1} (n - i) / (k - i)`. Equal to the Gamma-ratio
/// definition, but without its cancellation: for `n` around `1e10`, `lgamma_abs(n + 1)` and
/// `lgamma_abs(n - k + 1)` agree to about 20 significant digits, so their difference in
/// [`binom`]'s general path loses most of the precision this product keeps.
fn binom_falling_factorial(n: f64, k: f64) -> f64 {
    let mut result = 1.0;
    let mut i = 0.0;
    while i < k {
        result *= (n - i) / (k - i);
        i += 1.0;
        if !super::meter::tick() {
            break;
        }
    }
    result
}

/// The generalized binomial coefficient `C(n, k) = Gamma(n + 1) / (Gamma(k + 1) Gamma(n - k +
/// 1))`, defined for any real `n` and `k` (not just non-negative integers).
pub(in crate::python) fn binom(n: f64, k: f64) -> f64 {
    if k == 0.0 {
        return 1.0;
    }
    let p_pole = is_nonpositive_integer(n + 1.0);
    // Away from `Gamma(n + 1)`'s poles, a non-negative integer `k` can go through the exact
    // falling-factorial product instead. At a pole (`n` a negative integer), that product would
    // give a *different*, finite answer (the polynomial continuation SciPy's own `binom` does not
    // use), so this only applies where the two definitions agree.
    if !p_pole && k > 0.0 && k.is_finite() && k == k.floor() {
        return binom_falling_factorial(n, k);
    }
    let q1_pole = is_nonpositive_integer(k + 1.0);
    let q2_pole = is_nonpositive_integer(n - k + 1.0);
    if p_pole && (q1_pole || q2_pole) {
        return f64::NAN;
    }
    if q1_pole || q2_pole {
        return if p_pole { f64::NAN } else { 0.0 };
    }
    if p_pole {
        return f64::INFINITY * gamma_sign(k + 1.0) * gamma_sign(n - k + 1.0);
    }
    let magnitude = (lgamma_abs(n + 1.0) - lgamma_abs(k + 1.0) - lgamma_abs(n - k + 1.0)).exp();
    magnitude * gamma_sign(n + 1.0) * gamma_sign(k + 1.0) * gamma_sign(n - k + 1.0)
}
