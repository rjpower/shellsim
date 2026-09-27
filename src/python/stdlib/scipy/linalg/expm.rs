//! `scipy.linalg.expm`'s kernel: SciPy 1.18's `_matfuncs_expm.c`, the scaling and squaring
//! method of Al-Mohy and Higham (2009) with Padé approximants of order 3, 5, 7, 9 or 13.
//!
//! The port keeps SciPy's operation order, including the order of its `dgemm` operands, so
//! results round as SciPy's do up to OpenBLAS's kernel differences. It keeps one quirk as well:
//! the order-13 step computes `U` as `A (…) + A⁶`, because SciPy's last `dgemm` accumulates into
//! the buffer that holds `A⁶` (`beta = 1`). The extra term is scaled by `64^-s` and leaves a
//! relative error near `1e-12` in SciPy's results, which shellsim reproduces.
//!
//! Matrices are column-major here, as in SciPy's C code, and callers convert at the boundary.
//! The work is a fixed number of matrix products per Padé order plus one per squaring, so
//! callers charge `O(n³ (s + 10))` before calling [`expm`] (see [`cost`]).

use super::lapack::{self, Real};

/// The Padé order and scaling exponent [`pick_pade_structure`] selects.
struct Structure {
    order: u8,
    squarings: i32,
}

/// `(int)` of a C double, as x86's `cvttsd2si` computes it: NaN and out-of-range values give
/// `INT_MIN`.
fn c_int<T: Real>(value: T) -> i32 {
    let value = value.to_f64();
    if value.is_nan() || !(-2_147_483_648.0..2_147_483_648.0).contains(&value) {
        return i32::MIN;
    }
    value as i32
}

/// C's `fmax`, which ignores a NaN operand.
fn fmax<T: Real>(a: T, b: T) -> T {
    if a.is_nan() || (!b.is_nan() && b > a) {
        b
    } else {
        a
    }
}

/// C's `fmin`, which ignores a NaN operand.
fn fmin<T: Real>(a: T, b: T) -> T {
    if a.is_nan() || (!b.is_nan() && b < a) {
        b
    } else {
        a
    }
}

/// The largest element of `values`, starting from zero, as SciPy's loops compute it.
fn largest<T: Real>(values: &[T]) -> T {
    let mut result = T::ZERO;
    for value in values {
        if *value > result {
            result = *value;
        }
    }
    result
}

/// SciPy's `dnorm1`: the largest column sum of magnitudes of a column-major matrix.
fn norm1<T: Real>(a: &[T], n: usize) -> T {
    let mut norm = T::ZERO;
    for column in a.chunks_exact(n) {
        let mut sum = T::ZERO;
        for value in column {
            sum += value.abs();
        }
        if sum > norm {
            norm = sum;
        }
    }
    norm
}

/// SciPy's `dnorm1est`: `dlacn2`'s estimate of `‖Aᵖ‖₁`, used for matrices of order 400 or
/// more instead of forming another power.
fn norm1_estimate<T: Real>(a: &[T], n: usize, power: usize) -> T {
    let mut scratch = vec![T::ZERO; n];
    lapack::lacn2(n, |kase, x| {
        for _ in 0..power {
            lapack::gemv(kase != 1, n, a, x, &mut scratch);
            x.copy_from_slice(&scratch);
        }
        true
    })
    .unwrap_or(T::ZERO)
}

/// The matrices SciPy keeps in its `6 n² + 4 n` workspace: `powers[0]` is `A` and
/// `powers[1..4]` are `A²`, `A⁴` and `A⁶`, with `powers[4]` and `scratch` as temporaries.
struct Workspace<T> {
    n: usize,
    powers: [Vec<T>; 5],
    absolute: Vec<T>,
    /// The two vectors of the power iteration on `|A|`, alternating.
    spin: [Vec<T>; 2],
}

impl<T: Real> Workspace<T> {
    /// `dgemm` into `powers[target]`: `powers[left] powers[right] + beta powers[target]`.
    fn product(&mut self, left: usize, right: usize, target: usize, beta: T) {
        let n = self.n;
        let mut result = std::mem::take(&mut self.powers[target]);
        scale_for_gemm(&mut result, beta);
        lapack::gemm(
            n,
            &self.powers[left],
            &self.powers[right],
            &mut result,
            beta != T::ZERO,
        );
        self.powers[target] = result;
    }

    /// Two more steps of the power iteration `x ← |A|ᵀ x`, and the largest element after them.
    fn spin_twice(&mut self, rounds: usize) -> T {
        for _ in 0..rounds {
            let (first, second) = self.spin.split_at_mut(1);
            lapack::gemv(true, self.n, &self.absolute, &second[0], &mut first[0]);
            lapack::gemv(true, self.n, &self.absolute, &first[0], &mut second[0]);
        }
        largest(&self.spin[1])
    }
}

/// `dgemm`'s treatment of `beta` before it accumulates: zero, keep, or scale `c`.
fn scale_for_gemm<T: Real>(c: &mut [T], beta: T) {
    if beta == T::ZERO {
        c.iter_mut().for_each(|value| *value = T::ZERO);
    } else if beta != T::ONE {
        c.iter_mut().for_each(|value| *value = beta * *value);
    }
}

/// The `m` in `(int)ceil(log2(temp / normA / coeff) / divisor)`, clamped at zero.
fn excess<T: Real>(temp: T, norm: T, coefficient: f64, divisor: f64) -> i32 {
    let value = ((temp / norm / T::from_f64(coefficient)).log2() / T::from_f64(divisor)).ceil();
    c_int(value).max(0)
}

/// SciPy's `pick_pade_structure`: form `A²`, `A⁴` and `A⁶`, choose the Padé order, and for
/// order 13 the number of squarings, scaling the powers to match. `coefficients` are SciPy's
/// per-precision constants.
fn pick_pade_structure<T: Real>(work: &mut Workspace<T>, coefficients: &[f64; 5]) -> Structure {
    // SciPy's literals, kept verbatim; each parses to the double SciPy uses.
    #[allow(clippy::excessive_precision)]
    const THETA: [f64; 5] = [
        1.495585217958292e-002,
        2.539398330063230e-001,
        9.504178996162932e-001,
        2.097847961257068e+000,
        4.250000000000000e+000,
    ];
    let theta = THETA.map(T::from_f64);
    let n = work.n;
    work.spin[0].iter_mut().for_each(|value| *value = T::ONE);
    work.absolute = work.powers[0].iter().map(|value| value.abs()).collect();
    let (first, second) = work.spin.split_at_mut(1);
    lapack::gemv(true, n, &work.absolute, &first[0], &mut second[0]);
    let mut norm = largest(&work.spin[1]);

    work.product(0, 0, 1, T::ZERO);
    work.product(1, 1, 2, T::ZERO);
    work.product(2, 1, 3, T::ZERO);
    let d4 = norm1(&work.powers[2], n).powf(T::from_f64(0.25));
    let d6 = norm1(&work.powers[3], n).powf(T::from_f64(1.0 / 6.0));
    let eta0 = fmax(d4, d6);
    let eta1 = eta0;

    let temp = work.spin_twice(3);
    if eta0 < theta[0] && excess(temp, norm, coefficients[0], 6.0) == 0 {
        return Structure {
            order: 3,
            squarings: 0,
        };
    }
    let temp = work.spin_twice(2);
    if eta1 < theta[1] && excess(temp, norm, coefficients[1], 10.0) == 0 {
        return Structure {
            order: 5,
            squarings: 0,
        };
    }
    let d8 = if n < 400 {
        work.product(2, 2, 4, T::ZERO);
        norm1(&work.powers[4], n).powf(T::from_f64(0.125))
    } else {
        norm1_estimate(&work.powers[0], n, 8).powf(T::from_f64(0.125))
    };
    let eta2 = fmax(d6, d8);
    let temp = work.spin_twice(2);
    if eta2 < theta[2] && excess(temp, norm, coefficients[2], 14.0) == 0 {
        return Structure {
            order: 7,
            squarings: 0,
        };
    }
    let temp = work.spin_twice(2);
    if eta2 < theta[3] && excess(temp, norm, coefficients[3], 18.0) == 0 {
        if n >= 400 {
            work.product(2, 2, 4, T::ZERO);
        }
        return Structure {
            order: 9,
            squarings: 0,
        };
    }
    let d10 = if n < 400 {
        work.product(3, 2, 4, T::ZERO);
        norm1(&work.powers[4], n).powf(T::from_f64(0.1))
    } else {
        norm1_estimate(&work.powers[0], n, 10).powf(T::from_f64(0.1))
    };
    let eta3 = fmax(d8, d10);
    let eta4 = fmin(eta2, eta3);
    let mut squarings = c_int((eta4 / theta[4]).log2().ceil()).max(0);
    if squarings != 0 {
        let factor = T::from_f64(2.0).powf(T::from_f64(-f64::from(squarings)));
        work.absolute.iter_mut().for_each(|value| *value *= factor);
        // The power iteration has run 19 times already.
        let factor = T::from_f64(2.0).powf(T::from_f64(-19.0 * f64::from(squarings)));
        work.spin[1].iter_mut().for_each(|value| *value *= factor);
        norm *= T::from_f64(2.0).powf(T::from_f64(-f64::from(squarings)));
    }
    let temp = work.spin_twice(4);
    squarings = squarings.saturating_add(excess(temp, norm, coefficients[4], 26.0));
    if squarings != 0 {
        let exponent = T::from_f64(-f64::from(squarings));
        for (power, base) in [2.0, 4.0, 16.0, 64.0].into_iter().enumerate() {
            let factor = T::from_f64(base).powf(exponent);
            work.powers[power]
                .iter_mut()
                .for_each(|value| *value *= factor);
        }
    }
    Structure {
        order: 13,
        squarings,
    }
}

/// Add `value` to the diagonal of the column-major `n × n` matrix `a`.
fn add_diagonal<T: Real>(a: &mut [T], n: usize, value: T) {
    for i in 0..n {
        a[i + i * n] += value;
    }
}

/// SciPy's `pade_UV_calc`: form `U` and `V` of the Padé approximant of `order` and return
/// `I + 2 (V - U)⁻¹ U`, or LAPACK's `info` when `V - U` is exactly singular.
fn pade<T: Real>(work: &mut Workspace<T>, order: u8) -> Result<Vec<T>, usize> {
    let n = work.n;
    let b = |values: &[f64]| {
        values
            .iter()
            .map(|value| T::from_f64(*value))
            .collect::<Vec<_>>()
    };
    match order {
        3 => {
            let b = b(&[120.0, 60.0, 12.0]);
            work.powers[3] = work.powers[0].clone();
            work.product(1, 0, 3, b[1]);
            let factor = b[2];
            work.powers[1]
                .iter_mut()
                .for_each(|value| *value = factor * *value);
            add_diagonal(&mut work.powers[1], n, b[0]);
        }
        5 => {
            let b = b(&[30240.0, 15120.0, 3360.0, 420.0, 30.0]);
            let [a0, a1, a2, _, a4] = &mut work.powers;
            let _ = a0;
            for i in 0..n * n {
                a4[i] = a2[i] + b[3] * a1[i];
            }
            add_diagonal(a4, n, b[1]);
            work.product(4, 0, 3, T::ZERO);
            let [_, a1, a2, _, _] = &mut work.powers;
            for i in 0..n * n {
                a1[i] = b[4] * a2[i] + b[2] * a1[i];
            }
            add_diagonal(a1, n, b[0]);
        }
        7 => {
            let b = b(&[
                17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0,
            ]);
            let [_, a1, a2, a3, a4] = &mut work.powers;
            for i in 0..n * n {
                a4[i] = a3[i] + b[5] * a2[i] + b[3] * a1[i];
            }
            add_diagonal(a4, n, b[1]);
            for i in 0..n * n {
                a1[i] = b[6] * a3[i] + b[4] * a2[i] + b[2] * a1[i];
            }
            add_diagonal(a1, n, b[0]);
            work.product(4, 0, 3, T::ZERO);
        }
        9 => {
            let b = b(&[
                17643225600.0,
                8821612800.0,
                2075673600.0,
                302702400.0,
                30270240.0,
                2162160.0,
                110880.0,
                3960.0,
                90.0,
            ]);
            let [_, a1, a2, a3, a4] = &mut work.powers;
            for i in 0..n * n {
                let (temp1, temp2) = (a4[i], a1[i]);
                a4[i] = temp1 + b[7] * a3[i] + b[5] * a2[i] + b[3] * temp2;
                a1[i] = b[8] * temp1 + b[6] * a3[i] + b[4] * a2[i] + b[2] * temp2;
            }
            add_diagonal(a4, n, b[1]);
            add_diagonal(a1, n, b[0]);
            work.product(4, 0, 3, T::ZERO);
        }
        _ => {
            let b = b(&[
                64764752532480000.0,
                32382376266240000.0,
                7771770303897600.0,
                1187353796428800.0,
                129060195264000.0,
                10559470521600.0,
                670442572800.0,
                33522128640.0,
                1323241920.0,
                40840800.0,
                960960.0,
                16380.0,
                182.0,
            ]);
            let mut scratch = vec![T::ZERO; n * n];
            let [_, a1, a2, a3, a4] = &mut work.powers;
            for i in 0..n * n {
                let (temp1, temp2, temp3) = (a1[i], a2[i], a3[i]);
                a2[i] = b[7] * temp3 + b[5] * temp2 + b[3] * temp1;
                a4[i] = b[12] * temp3 + b[10] * temp2 + b[8] * temp1;
                a1[i] = b[6] * temp3 + b[4] * temp2 + b[2] * temp1;
                scratch[i] = temp3 + b[11] * temp2 + b[9] * temp1;
            }
            add_diagonal(a2, n, b[1]);
            add_diagonal(a1, n, b[0]);
            // V = A⁴-term A⁶ + V, then U = A (scratch A⁶ + U) + A⁶ (see the module notes).
            work.product(4, 3, 1, T::ONE);
            let [_, _, a2, a3, _] = &mut work.powers;
            lapack::gemm(n, &scratch, a3, a2, true);
            work.product(2, 0, 3, T::ONE);
        }
    }
    let [_, v, _, u, _] = &mut work.powers;
    for i in 0..n * n {
        v[i] -= u[i];
    }
    // SciPy solves the transposed system (V - U)ᵀ X = Uᵀ in column-major storage.
    let mut x = transpose(u, n);
    let (pivots, info) = lapack::getf2(n, n, v, n);
    if info > 0 {
        return Err(info);
    }
    lapack::getrs(true, n, n, v, n, &pivots, &mut x, n);
    let two = T::from_f64(2.0);
    x.iter_mut().for_each(|value| *value = two * *value);
    add_diagonal(&mut x, n, T::ONE);
    Ok(transpose(&x, n))
}

/// The transpose of a square matrix.
fn transpose<T: Real>(a: &[T], n: usize) -> Vec<T> {
    let mut result = vec![T::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            result[j + i * n] = a[i + j * n];
        }
    }
    result
}

/// SciPy's `bandwidth_d` on a C-order matrix: the lower and upper bandwidths.
fn bandwidths<T: Real>(a: &[T], n: usize) -> (usize, usize) {
    let (mut lower, mut upper) = (0, 0);
    for r in (1..n).rev() {
        let limit = r - lower;
        if let Some(c) = (0..limit).find(|&c| a[r * n + c] != T::ZERO) {
            lower = r - c;
        }
        if r <= lower {
            break;
        }
    }
    for r in 0..n.saturating_sub(1) {
        if let Some(c) = (r + upper + 1..n).rev().find(|&c| a[r * n + c] != T::ZERO) {
            upper = c - r;
        }
        if r + upper + 1 > n {
            break;
        }
    }
    (lower, upper)
}

/// CPU units to charge before [`expm`] on an `n × n` matrix: a Padé step of at most a dozen
/// matrix products, and one product per squaring. The squaring count grows with the log of the
/// matrix norm, so callers charge [`squaring_cost`] as it is known.
pub(super) fn cost(n: usize) -> u64 {
    (n as u64)
        .saturating_pow(3)
        .saturating_mul(12)
        .saturating_add(1)
}

/// CPU units for `squarings` matrix products of order `n`.
pub(super) fn squaring_cost(n: usize, squarings: i32) -> u64 {
    (n as u64)
        .saturating_pow(3)
        .saturating_mul(u64::try_from(squarings).unwrap_or(0))
}

/// The per-precision constants of `pick_pade_structure`.
pub(super) trait PadeCoefficients: Real {
    const COEFFICIENTS: [f64; 5];
}

impl PadeCoefficients for f64 {
    #[allow(clippy::excessive_precision)]
    const COEFFICIENTS: [f64; 5] = [
        1.1191048088221578e-11,
        1.1167770708198077e-06,
        4.9826124310493469e-01,
        6.5662862500000000e+05,
        1.2573361865339437e+19,
    ];
}

impl PadeCoefficients for f32 {
    #[allow(clippy::excessive_precision)]
    const COEFFICIENTS: [f64; 5] = [
        6.0081481933593750e-03,
        5.9956512451171875e+02,
        2.6750196800000000e+08,
        3.5252480874905600e+14,
        6.7502722515508048e+27,
    ];
}

/// `exp(A)` for the C-order `n × n` matrix `a`, returned in C order. `charge` is called with
/// the cost of the squarings once their number is known, before they run. An exactly singular
/// `V - U` returns LAPACK's `info`.
pub(super) fn expm<T: PadeCoefficients, E>(
    a: &[T],
    n: usize,
    charge: &mut dyn FnMut(u64) -> Result<(), E>,
) -> Result<Result<Vec<T>, usize>, E> {
    let (lower, upper) = bandwidths(a, n);
    if lower == 0 && upper == 0 {
        let mut result = vec![T::ZERO; n * n];
        for i in 0..n {
            result[i * n + i] = a[i * n + i].exp();
        }
        return Ok(Ok(result));
    }
    let matrix = transpose(a, n);
    let triangular = lower == 0 || upper == 0;
    let is_lower = upper == 0;
    let diagonal = (0..n).map(|i| matrix[i + i * n]).collect::<Vec<_>>();
    // The first off-diagonal: below the diagonal for a lower triangular matrix, else above.
    let off_diagonal = (0..n.saturating_sub(1))
        .map(|i| {
            if is_lower {
                matrix[i + 1 + i * n]
            } else {
                matrix[i + (i + 1) * n]
            }
        })
        .collect::<Vec<_>>();
    let zero = vec![T::ZERO; n * n];
    let mut work = Workspace {
        n,
        powers: [
            matrix,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
        ],
        absolute: zero,
        spin: [vec![T::ZERO; n], vec![T::ZERO; n]],
    };
    let structure = pick_pade_structure(&mut work, &T::COEFFICIENTS);
    let mut result = match pade(&mut work, structure.order) {
        Ok(result) => result,
        Err(info) => return Ok(Err(info)),
    };
    charge(squaring_cost(n, structure.squarings))?;
    let mut scratch = vec![T::ZERO; n * n];
    if structure.squarings > 0 && triangular {
        // Fragment 2.1 of Al-Mohy and Higham (2009): recompute the diagonal and first
        // off-diagonal exactly after each squaring.
        for iteration in (0..structure.squarings).rev() {
            lapack::gemm(n, &result, &result, &mut scratch, false);
            std::mem::swap(&mut result, &mut scratch);
            let scale = T::from_f64(-f64::from(iteration)).exp2();
            for i in 0..n.saturating_sub(1) {
                let d_i = diagonal[i] * scale;
                let d_next = diagonal[i + 1] * scale;
                let difference = d_next - d_i;
                let exp_sinch = if difference == T::ZERO {
                    d_i.exp()
                } else {
                    (d_next.exp() - d_i.exp()) / difference
                };
                let value = exp_sinch * off_diagonal[i] * scale;
                if is_lower {
                    result[i + i * n] = d_i.exp();
                    result[i + 1 + i * n] = value;
                } else {
                    result[i + (i + 1) * n] = value;
                    result[i + 1 + (i + 1) * n] = d_next.exp();
                }
            }
            if is_lower {
                result[(n - 1) * (n + 1)] = (diagonal[n - 1] * scale).exp();
            } else {
                result[0] = (diagonal[0] * scale).exp();
            }
        }
    } else {
        for _ in 0..structure.squarings {
            lapack::gemm(n, &result, &result, &mut scratch, false);
            std::mem::swap(&mut result, &mut scratch);
        }
    }
    Ok(Ok(transpose(&result, n)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unmetered(_: u64) -> Result<(), ()> {
        Ok(())
    }

    fn exponential(a: &[f64], n: usize) -> Vec<f64> {
        expm(a, n, &mut unmetered).unwrap().unwrap()
    }

    #[test]
    fn diagonal_matrices_take_elementwise_exponentials() {
        assert_eq!(
            exponential(&[1.0, 0.0, 0.0, 0.0], 2),
            vec![1.0f64.exp(), 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn nilpotent_matrices_have_polynomial_exponentials() {
        // exp([[0, 1], [0, 0]]) = [[1, 1], [0, 1]].
        assert_eq!(
            exponential(&[0.0, 1.0, 0.0, 0.0], 2),
            vec![1.0, 1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn dense_matrices_match_scipy() {
        // scipy.linalg.expm of [[0.1, 0.2], [0.3, 1.9]], which takes Padé order 7, and of
        // [[3, 6], [9, 12]], which takes order 13 and squarings.
        for (a, reference) in [
            (
                [0.1, 0.2, 0.3, 1.9],
                [
                    1.1720433769251966,
                    0.6259875479854236,
                    0.9389813219781356,
                    6.805931308794008,
                ],
            ),
            (
                [3.0, 6.0, 9.0, 12.0],
                [
                    2385847.2203679075,
                    3477197.936897128,
                    5215796.905345691,
                    7601644.125713603,
                ],
            ),
        ] {
            let result = exponential(&a, 2);
            for (value, expected) in result.iter().zip(reference) {
                // OpenBLAS's triangular solves differ from reference LAPACK's in the last bit.
                assert!((value / expected - 1.0).abs() < 5e-16, "{value} {expected}");
            }
        }
    }

    #[test]
    fn bandwidths_read_c_order() {
        let upper = [1.0, 2.0, 0.0, 3.0];
        assert_eq!(bandwidths(&upper, 2), (0, 1));
        assert_eq!(bandwidths(&[1.0, 0.0, 2.0, 3.0], 2), (1, 0));
    }
}
