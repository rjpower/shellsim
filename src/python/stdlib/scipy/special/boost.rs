//! Boost.Math building blocks as SciPy's GCC builds for x86-64 compile them: polynomial
//! evaluation, the Lanczos sums, `log1pmx`, and the logarithm limits Boost's code tests against.
//!
//! GCC builds select `BOOST_MATH_POLY_METHOD 3` and `BOOST_MATH_RATIONAL_METHOD 3`
//! (`boost/math/tools/config.hpp`): a second-order Horner scheme that evaluates the even and odd
//! coefficients as two polynomials in `x^2` and combines them at the end. It rounds differently
//! from the plain Horner scheme Cephes uses, so the Boost ports evaluate their fixed-size tables
//! here to reproduce SciPy's results to the last bit. Boost's `log1p` and `expm1` for `double`
//! call the C library's, as the ports do directly. The x86-64 baseline SciPy targets has no fused
//! multiply-add, so every product and sum rounds separately, as here.
//!
//! Coefficients are stored lowest degree first, as in Boost's tables.

/// Boost's `log_max_value<double>` and `log_min_value<double>`: the exact limits of `ln` over
/// the finite normal doubles, rounded toward zero to integers.
pub(super) const LOG_MAX: f64 = 709.0;
pub(super) const LOG_MIN: f64 = -708.0;

/// Horner's scheme in `z` over every `step`-th coefficient starting at `first`, highest degree
/// first.
fn horner_every(coefficients: &[f64], first: usize, step: usize, z: f64) -> f64 {
    let mut indices = (first..coefficients.len()).step_by(step).rev();
    let highest = indices
        .next()
        .expect("the table has a coefficient at `first`");
    indices.fold(coefficients[highest], |sum, index| {
        sum * z + coefficients[index]
    })
}

/// Boost's `evaluate_polynomial` for a fixed-size table: the even part plus `x` times the odd
/// part, each in `x^2`, for 5 to 20 terms (`polynomial_horner3_5.hpp` to `_20.hpp`), and plain
/// Horner otherwise.
pub(super) fn polynomial(coefficients: &[f64], x: f64) -> f64 {
    if !(5..=20).contains(&coefficients.len()) {
        return horner_every(coefficients, 0, 1, x);
    }
    let x2 = x * x;
    horner_every(coefficients, 0, 2, x2) + horner_every(coefficients, 1, 2, x2) * x
}

/// A 13-term Lanczos sum `num(x) / denom(x)` as Boost evaluates `lanczos13m53` for `double` on
/// x86-64 (`lanczos_sse2.hpp`), where SSE2 is always available: the numerator and denominator
/// run in the two lanes of one register, each as its even part plus `x` times its odd part in
/// `x^2`. Above `limit`, where that would overflow, both are plain Horner in `1 / x` with the
/// coefficients in reverse. Other platforms use the generic `rational_horner3_13.hpp`, which
/// inverts every argument above 1 and so rounds differently.
pub(super) fn lanczos_sum(num: &[f64; 13], denom: &[f64; 13], x: f64, limit: f64) -> f64 {
    if x > limit {
        let z = 1.0 / x;
        let reversed =
            |table: &[f64; 13]| table.iter().skip(1).fold(table[0], |sum, c| sum * z + c);
        return reversed(num) / reversed(denom);
    }
    let x2 = x * x;
    let split =
        |table: &[f64; 13]| horner_every(table, 0, 2, x2) + horner_every(table, 1, 2, x2) * x;
    split(num) / split(denom)
}

/// Boost's `log1pmx`: `ln(1 + x) - x`, NaN below -1. For `|x| <= 0.95` it sums the Taylor
/// series from its second term until a term falls to `EPSILON` times the sum, which takes at
/// most about 700 terms.
pub(super) fn log1pmx(x: f64) -> f64 {
    if x < -1.0 {
        return f64::NAN;
    }
    if x == -1.0 {
        return f64::NEG_INFINITY;
    }
    let a = x.abs();
    if a > 0.95 {
        return (1.0 + x).ln() - x;
    }
    if a < f64::EPSILON {
        return -x * x / 2.0;
    }
    let mut power = x;
    let mut result = 0.0;
    for k in 2..=1000 {
        power *= -x;
        let term = power / f64::from(k);
        result += term;
        if (f64::EPSILON * result).abs() >= term.abs() {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomials_evaluate_lowest_degree_first() {
        // 1 + 2x + 3x^2 and 1 - x + x^2 - x^3 + x^4 - x^5 at x = 2.
        assert_eq!(polynomial(&[1.0, 2.0, 3.0], 2.0), 17.0);
        assert_eq!(polynomial(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0], 2.0), -21.0);
    }

    #[test]
    fn lanczos_sums_agree_on_both_sides_of_the_limit() {
        let mut num = [0.0; 13];
        let mut denom = [0.0; 13];
        num[0] = 1.0;
        num[12] = 1.0;
        denom[12] = 2.0;
        denom[1] = 1.0;
        for x in [0.5f64, 3.0, 1e3] {
            let expected = (1.0 + x.powi(12)) / (x + 2.0 * x.powi(12));
            for limit in [0.0, f64::INFINITY] {
                let sum = lanczos_sum(&num, &denom, x, limit);
                assert!((sum - expected).abs() <= 1e-15 * expected.abs());
            }
        }
    }

    #[test]
    fn log1pmx_matches_its_definition() {
        // Away from 0 the definition loses little to cancellation; near 0 the series is exact.
        for x in [-0.99, -0.9, -0.3, 0.25, 0.94, 3.0] {
            let expected = f64::ln_1p(x) - x;
            assert!(
                (log1pmx(x) - expected).abs() <= 1e-14 * expected.abs(),
                "{x}"
            );
        }
        let x = -1e-10;
        assert_eq!(log1pmx(x), -x * x / 2.0 + x * x * x / 3.0);
        assert_eq!(log1pmx(1e-300), -0.0);
        assert!(log1pmx(-2.0).is_nan());
        assert_eq!(log1pmx(-1.0), f64::NEG_INFINITY);
    }
}
