//! Polynomial, rational, and Chebyshev series evaluation shared by the Cephes ports.
//!
//! Coefficients are stored highest degree first, as in Cephes, so each table can be copied
//! from the upstream source unchanged.

/// `coef[0] x^n + ... + coef[n]`.
pub(super) fn polevl(x: f64, coef: &[f64]) -> f64 {
    coef[1..]
        .iter()
        .fold(coef[0], |value, coefficient| value * x + coefficient)
}

/// `x^n + coef[0] x^(n-1) + ... + coef[n-1]`: a polynomial whose leading coefficient is 1 and
/// omitted from the table.
pub(super) fn p1evl(x: f64, coef: &[f64]) -> f64 {
    coef[1..]
        .iter()
        .fold(x + coef[0], |value, coefficient| value * x + coefficient)
}

/// The rational function `num(x) / denom(x)`, evaluated in `1/x` when `|x| > 1` to avoid
/// overflow. Both tables are stored highest degree first.
pub(super) fn ratevl(x: f64, num: &[f64], denom: &[f64]) -> f64 {
    let (m, n) = (num.len() - 1, denom.len() - 1);
    if x.abs() > 1.0 {
        let y = 1.0 / x;
        let num_value = num
            .iter()
            .rev()
            .skip(1)
            .fold(num[m], |value, c| value * y + c);
        let denom_value = denom
            .iter()
            .rev()
            .skip(1)
            .fold(denom[n], |value, c| value * y + c);
        x.powi(m as i32 - n as i32) * num_value / denom_value
    } else {
        let num_value = num.iter().skip(1).fold(num[0], |value, c| value * x + c);
        let denom_value = denom
            .iter()
            .skip(1)
            .fold(denom[0], |value, c| value * x + c);
        num_value / denom_value
    }
}

/// Sum of a Chebyshev series `array` at `x`, with Cephes' convention that the first
/// coefficient is doubled.
pub(super) fn chbevl(x: f64, array: &[f64]) -> f64 {
    let mut b0 = array[0];
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for coefficient in &array[1..] {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + coefficient;
    }
    0.5 * (b0 - b2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomials_evaluate_highest_degree_first() {
        assert_eq!(polevl(2.0, &[1.0, 0.0, -3.0]), 1.0);
        assert_eq!(p1evl(2.0, &[0.0, -3.0]), 1.0);
    }

    #[test]
    fn rational_functions_switch_to_reciprocal_argument() {
        let num = [1.0, 2.0];
        let denom = [3.0, 1.0];
        for x in [0.5, 4.0, -7.0] {
            let expected = (x + 2.0) / (3.0 * x + 1.0);
            assert!((ratevl(x, &num, &denom) - expected).abs() < 1e-15);
        }
    }
}
