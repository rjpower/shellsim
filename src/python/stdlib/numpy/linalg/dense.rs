//! Dense real matrix kernels behind `numpy.linalg`, on row-major `f64` matrices.
//!
//! NumPy calls LAPACK through OpenBLAS. These kernels follow the LAPACK routines where their
//! conventions are visible in results:
//!
//! - [`lu`] is `dgetf2`: partial pivoting on the first largest magnitude, and the column below
//!   each pivot scaled by the pivot's reciprocal. A zero or NaN pivot marks the matrix
//!   singular, as OpenBLAS reports it. [`Lu::solve`] is `dgetrs` with reference `dtrsm` order.
//!   OpenBLAS's blocked kernels fuse some multiply-adds, so a solve can differ from NumPy's in
//!   the last bit.
//! - [`cholesky`] is OpenBLAS's `potf2`, which scales each column by the diagonal's reciprocal.
//! - [`householder_qr`] and [`householder_q`] are `dgeqr2` and `dorg2r`, including `dlarfg`'s
//!   sign choice, so `R` has NumPy's diagonal signs.
//! - [`symmetric_eigen`] uses cyclic Jacobi rotations and [`svd`] one-sided Jacobi, in place of
//!   `dsyevd` and `dgesdd`. Values agree with LAPACK's to rounding. The signs of eigenvectors
//!   and singular vectors, which LAPACK does not specify either, may differ from NumPy's.
//!
//! Direct methods do cubic work that callers charge before calling. The Jacobi methods iterate
//! until convergence and charge each sweep through a `meter` callback before running it.

use super::super::super::super::native::{PyError, PyResult};

/// Jacobi sweeps allowed before a method reports that it did not converge. Both methods
/// converge quadratically, in well under 20 sweeps for any finite input.
const MAX_SWEEPS: usize = 60;

/// A row-major matrix.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize, data: Vec<f64>) -> Self {
        debug_assert_eq!(data.len(), rows * columns);
        Self {
            rows,
            columns,
            data,
        }
    }

    pub fn zeros(rows: usize, columns: usize) -> Self {
        Self::new(rows, columns, vec![0.0; rows * columns])
    }

    pub fn identity(size: usize) -> Self {
        let mut matrix = Self::zeros(size, size);
        for index in 0..size {
            matrix.set(index, index, 1.0);
        }
        matrix
    }

    pub fn get(&self, row: usize, column: usize) -> f64 {
        self.data[row * self.columns + column]
    }

    pub fn set(&mut self, row: usize, column: usize, value: f64) {
        self.data[row * self.columns + column] = value;
    }

    pub fn transpose(&self) -> Self {
        let mut transposed = Self::zeros(self.columns, self.rows);
        for row in 0..self.rows {
            for column in 0..self.columns {
                transposed.set(column, row, self.get(row, column));
            }
        }
        transposed
    }

    fn swap_rows(&mut self, first: usize, second: usize) {
        for column in 0..self.columns {
            self.data.swap(
                first * self.columns + column,
                second * self.columns + column,
            );
        }
    }

    fn column(&self, column: usize) -> Vec<f64> {
        (0..self.rows).map(|row| self.get(row, column)).collect()
    }

    fn set_column(&mut self, column: usize, values: &[f64]) {
        for (row, value) in values.iter().enumerate() {
            self.set(row, column, *value);
        }
    }

    /// Keep only the listed columns, in order.
    fn select_columns(&self, columns: &[usize]) -> Self {
        let mut selected = Self::zeros(self.rows, columns.len());
        for (target, source) in columns.iter().enumerate() {
            for row in 0..self.rows {
                selected.set(row, target, self.get(row, *source));
            }
        }
        selected
    }
}

/// An LU factorization `P A = L U` of a square matrix.
pub(super) struct Lu {
    /// `U` on and above the diagonal and `L` below it, with an implied unit diagonal.
    factors: Matrix,
    /// The row exchanged with row `k` at step `k`.
    pivots: Vec<usize>,
    odd_permutation: bool,
    /// A pivot was zero or NaN, so `U` is singular and the factors are incomplete.
    pub singular: bool,
}

/// Factor a square matrix as `dgetf2` does.
pub(super) fn lu(mut a: Matrix) -> Lu {
    let n = a.rows;
    let mut pivots = Vec::with_capacity(n);
    let mut odd_permutation = false;
    for k in 0..n {
        let mut pivot_row = k;
        let mut largest = a.get(k, k).abs();
        for row in k + 1..n {
            let magnitude = a.get(row, k).abs();
            if magnitude > largest {
                largest = magnitude;
                pivot_row = row;
            }
        }
        let pivot = a.get(pivot_row, k);
        if pivot == 0.0 || pivot.is_nan() {
            return Lu {
                factors: a,
                pivots,
                odd_permutation,
                singular: true,
            };
        }
        pivots.push(pivot_row);
        if pivot_row != k {
            a.swap_rows(k, pivot_row);
            odd_permutation = !odd_permutation;
        }
        if pivot.abs() >= f64::MIN_POSITIVE {
            let reciprocal = 1.0 / pivot;
            for row in k + 1..n {
                a.set(row, k, a.get(row, k) * reciprocal);
            }
        } else {
            for row in k + 1..n {
                a.set(row, k, a.get(row, k) / pivot);
            }
        }
        for column in k + 1..n {
            let upper = a.get(k, column);
            if upper == 0.0 {
                continue;
            }
            for row in k + 1..n {
                a.set(row, column, a.get(row, column) - a.get(row, k) * upper);
            }
        }
    }
    Lu {
        factors: a,
        pivots,
        odd_permutation,
        singular: false,
    }
}

impl Lu {
    /// `(sign, log|det|)` accumulated along `U`'s diagonal as NumPy's `slogdet` does; a
    /// singular matrix gives `(0, -inf)`.
    pub fn slogdet(&self) -> (f64, f64) {
        if self.singular {
            return (0.0, f64::NEG_INFINITY);
        }
        let mut sign = if self.odd_permutation { -1.0 } else { 1.0 };
        let mut logdet = 0.0;
        for index in 0..self.factors.rows {
            let mut element = self.factors.get(index, index);
            if element < 0.0 {
                sign = -sign;
                element = -element;
            }
            logdet += element.ln();
        }
        (sign, logdet)
    }

    /// Overwrite `b` with the solution of `A X = b`. The factorization must not be singular.
    pub fn solve(&self, b: &mut Matrix) {
        let n = self.factors.rows;
        for (row, pivot) in self.pivots.iter().enumerate() {
            if *pivot != row {
                b.swap_rows(row, *pivot);
            }
        }
        for column in 0..b.columns {
            for k in 0..n {
                let value = b.get(k, column);
                if value == 0.0 {
                    continue;
                }
                for row in k + 1..n {
                    let updated = b.get(row, column) - value * self.factors.get(row, k);
                    b.set(row, column, updated);
                }
            }
            for k in (0..n).rev() {
                if b.get(k, column) == 0.0 {
                    continue;
                }
                let value = b.get(k, column) / self.factors.get(k, k);
                b.set(k, column, value);
                for row in 0..k {
                    let updated = b.get(row, column) - value * self.factors.get(row, k);
                    b.set(row, column, updated);
                }
            }
        }
    }
}

/// The lower-triangular `L` with `A = L Lᵀ`, read from `a`'s lower triangle, or `None` when
/// a diagonal step is zero or negative. A NaN step propagates instead, as in OpenBLAS.
pub(super) fn cholesky(a: &Matrix) -> Option<Matrix> {
    let n = a.rows;
    let mut lower = Matrix::zeros(n, n);
    for row in 0..n {
        for column in 0..=row {
            lower.set(row, column, a.get(row, column));
        }
    }
    for j in 0..n {
        let mut dot = 0.0;
        for k in 0..j {
            dot += lower.get(j, k) * lower.get(j, k);
        }
        let diagonal = lower.get(j, j) - dot;
        if diagonal <= 0.0 {
            return None;
        }
        let diagonal = diagonal.sqrt();
        lower.set(j, j, diagonal);
        let reciprocal = 1.0 / diagonal;
        for row in j + 1..n {
            let mut value = lower.get(row, j);
            for k in 0..j {
                value -= lower.get(row, k) * lower.get(j, k);
            }
            lower.set(row, j, value * reciprocal);
        }
    }
    Some(lower)
}

/// The Euclidean norm, accumulated with an error-free sum of squares so it rounds like
/// OpenBLAS's extended-precision `dnrm2`. NaN propagates.
pub(in crate::python) fn norm2(values: &[f64]) -> f64 {
    let (mut sum, mut compensation) = (0.0f64, 0.0f64);
    for value in values {
        let square = value * value;
        let square_error = value.mul_add(*value, -square);
        let total = sum + square;
        let bits = total - sum;
        compensation += (sum - (total - bits)) + (square - bits) + square_error;
        sum = total;
    }
    let result = (sum + compensation).sqrt();
    if result.is_nan() || (result.is_finite() && result > f64::MIN_POSITIVE.sqrt()) {
        return result;
    }
    // Squares overflowed or underflowed: scale by the largest magnitude first.
    let scale = values
        .iter()
        .fold(0.0f64, |largest, value| largest.max(value.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return scale;
    }
    let scaled = values
        .iter()
        .map(|value| (value / scale).powi(2))
        .sum::<f64>();
    scale * scaled.sqrt()
}

/// `dlapy2`: `sqrt(x² + y²)` without undue overflow.
fn lapy2(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let (larger, smaller) = (x.abs().max(y.abs()), x.abs().min(y.abs()));
    if smaller == 0.0 || larger > f64::MAX {
        return larger;
    }
    larger * (1.0 + (smaller / larger).powi(2)).sqrt()
}

/// `dlarfg` on column `column` from row `column` down: replace the column with `beta` and the
/// reflector's vector, and return the reflector's `tau`.
fn generate_reflector(a: &mut Matrix, column: usize) -> f64 {
    let first = column;
    if a.rows - first <= 1 {
        return 0.0;
    }
    let below = |a: &Matrix| {
        (first + 1..a.rows)
            .map(|row| a.get(row, column))
            .collect::<Vec<_>>()
    };
    let mut alpha = a.get(first, column);
    let mut xnorm = norm2(&below(a));
    if xnorm == 0.0 {
        return 0.0;
    }
    let mut beta = -lapy2(alpha, xnorm).copysign(alpha);
    let safe_minimum = f64::MIN_POSITIVE / (f64::EPSILON * 0.5);
    let mut rescalings = 0;
    if beta.abs() < safe_minimum {
        let inverse = 1.0 / safe_minimum;
        loop {
            rescalings += 1;
            for row in first + 1..a.rows {
                a.set(row, column, a.get(row, column) * inverse);
            }
            beta *= inverse;
            alpha *= inverse;
            if !(beta.abs() < safe_minimum && rescalings < 20) {
                break;
            }
        }
        xnorm = norm2(&below(a));
        beta = -lapy2(alpha, xnorm).copysign(alpha);
    }
    let tau = (beta - alpha) / beta;
    let scale = 1.0 / (alpha - beta);
    for row in first + 1..a.rows {
        a.set(row, column, a.get(row, column) * scale);
    }
    for _ in 0..rescalings {
        beta *= safe_minimum;
    }
    a.set(first, column, beta);
    tau
}

/// `dlarf` from the left: apply `I - tau v vᵀ`, where `v` is column `vector` from row `first`
/// down, to rows `first..` and columns `columns..` of `a`.
fn apply_reflector(a: &mut Matrix, vector: usize, first: usize, columns: usize, tau: f64) {
    if tau == 0.0 {
        return;
    }
    let weights = (columns..a.columns)
        .map(|column| {
            let mut sum = 0.0;
            for row in first..a.rows {
                sum += a.get(row, column) * a.get(row, vector);
            }
            sum
        })
        .collect::<Vec<_>>();
    for (column, weight) in (columns..a.columns).zip(weights) {
        if weight == 0.0 {
            continue;
        }
        let scale = -tau * weight;
        for row in first..a.rows {
            let updated = a.get(row, column) + a.get(row, vector) * scale;
            a.set(row, column, updated);
        }
    }
}

/// Householder QR as `dgeqr2` leaves it: `R` on and above the diagonal, each reflector's
/// vector below it (with an implied leading 1), and the reflectors' `tau` factors.
pub(super) fn householder_qr(mut a: Matrix) -> (Matrix, Vec<f64>) {
    let steps = a.rows.min(a.columns);
    let mut tau = Vec::with_capacity(steps);
    for index in 0..steps {
        let factor = generate_reflector(&mut a, index);
        tau.push(factor);
        if index + 1 < a.columns {
            let saved = a.get(index, index);
            a.set(index, index, 1.0);
            apply_reflector(&mut a, index, index, index + 1, factor);
            a.set(index, index, saved);
        }
    }
    (a, tau)
}

/// The first `columns` columns of `Q` from [`householder_qr`]'s reflectors, as `dorg2r` forms
/// them. `columns` is at least the number of reflectors and at most the number of rows.
pub(super) fn householder_q(factors: &Matrix, tau: &[f64], columns: usize) -> Matrix {
    let rows = factors.rows;
    let steps = tau.len();
    let mut q = Matrix::zeros(rows, columns);
    for row in 0..rows {
        for column in 0..steps {
            q.set(row, column, factors.get(row, column));
        }
    }
    for column in steps..columns {
        q.set(column, column, 1.0);
    }
    for index in (0..steps).rev() {
        if index + 1 < columns {
            q.set(index, index, 1.0);
            apply_reflector(&mut q, index, index, index + 1, tau[index]);
        }
        for row in index + 1..rows {
            q.set(row, index, q.get(row, index) * -tau[index]);
        }
        q.set(index, index, 1.0 - tau[index]);
        for row in 0..index {
            q.set(row, index, 0.0);
        }
    }
    q
}

/// Eigenvalues in ascending order and, when `vectors` is set, the matching unit eigenvectors
/// as columns, of the symmetric matrix whose lower triangle is `a`'s. The cyclic Jacobi method
/// follows Numerical Recipes' `jacobi`, which accumulates diagonal updates separately to limit
/// rounding.
pub(super) fn symmetric_eigen(
    a: &Matrix,
    vectors: bool,
    meter: &mut dyn FnMut(u64) -> PyResult<()>,
) -> PyResult<(Vec<f64>, Option<Matrix>)> {
    let n = a.rows;
    // `work[p][q]` for `p < q` holds the current off-diagonal element (p, q).
    let mut work = Matrix::zeros(n, n);
    for row in 0..n {
        for column in 0..row {
            work.set(column, row, a.get(row, column));
        }
    }
    let mut v = Matrix::identity(n);
    let mut diagonal = (0..n).map(|index| a.get(index, index)).collect::<Vec<_>>();
    let mut base = diagonal.clone();
    let mut pending = vec![0.0; n];
    let cost = (n as u64).saturating_pow(3).saturating_add(1);
    let mut converged = n <= 1;
    for sweep in 1..=MAX_SWEEPS {
        if converged {
            break;
        }
        let off_diagonal: f64 = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| work.get(p, q).abs())
            .sum();
        if off_diagonal == 0.0 {
            converged = true;
            break;
        }
        meter(cost)?;
        let threshold = if sweep < 4 {
            0.2 * off_diagonal / (n * n) as f64
        } else {
            0.0
        };
        for p in 0..n.saturating_sub(1) {
            for q in p + 1..n {
                let element = work.get(p, q);
                let g = 100.0 * element.abs();
                if sweep > 4
                    && diagonal[p].abs() + g == diagonal[p].abs()
                    && diagonal[q].abs() + g == diagonal[q].abs()
                {
                    work.set(p, q, 0.0);
                    continue;
                }
                if element.abs() <= threshold {
                    continue;
                }
                let difference = diagonal[q] - diagonal[p];
                let t = if difference.abs() + g == difference.abs() {
                    element / difference
                } else {
                    let theta = 0.5 * difference / element;
                    let t = 1.0 / (theta.abs() + (1.0 + theta * theta).sqrt());
                    if theta < 0.0 {
                        -t
                    } else {
                        t
                    }
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);
                let h = t * element;
                pending[p] -= h;
                pending[q] += h;
                diagonal[p] -= h;
                diagonal[q] += h;
                work.set(p, q, 0.0);
                let rotate = |matrix: &mut Matrix, i: usize, j: usize, k: usize, l: usize| {
                    let (g, h) = (matrix.get(i, j), matrix.get(k, l));
                    matrix.set(i, j, g - s * (h + g * tau));
                    matrix.set(k, l, h + s * (g - h * tau));
                };
                for j in 0..p {
                    rotate(&mut work, j, p, j, q);
                }
                for j in p + 1..q {
                    rotate(&mut work, p, j, j, q);
                }
                for j in q + 1..n {
                    rotate(&mut work, p, j, q, j);
                }
                if vectors {
                    for j in 0..n {
                        rotate(&mut v, j, p, j, q);
                    }
                }
            }
        }
        for index in 0..n {
            base[index] += pending[index];
            diagonal[index] = base[index];
            pending[index] = 0.0;
        }
    }
    if !converged {
        return Err(PyError::exception(
            "LinAlgError",
            "Eigenvalues did not converge",
        ));
    }
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|left, right| diagonal[*left].total_cmp(&diagonal[*right]));
    let values = order.iter().map(|index| diagonal[*index]).collect();
    Ok((values, vectors.then(|| v.select_columns(&order))))
}

/// Which singular vectors [`svd`] computes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SvdVectors {
    None,
    /// `U` is `m × k` and `Vᵀ` is `k × n`, for `k = min(m, n)`.
    Reduced,
    /// `U` is `m × m` and `Vᵀ` is `n × n`.
    Full,
}

/// Singular values in descending order and, if requested, `(U, Vᵀ)`.
pub(super) struct Svd {
    pub values: Vec<f64>,
    pub vectors: Option<(Matrix, Matrix)>,
}

/// The singular value decomposition `A = U diag(s) Vᵀ`.
pub(super) fn svd(
    a: &Matrix,
    mode: SvdVectors,
    meter: &mut dyn FnMut(u64) -> PyResult<()>,
) -> PyResult<Svd> {
    if a.rows >= a.columns {
        return svd_tall(a, mode, meter);
    }
    // A = (Aᵀ)ᵀ = (U' S V'ᵀ)ᵀ = V' S U'ᵀ.
    let transposed = svd_tall(&a.transpose(), mode, meter)?;
    Ok(Svd {
        values: transposed.values,
        vectors: transposed
            .vectors
            .map(|(u, vt)| (vt.transpose(), u.transpose())),
    })
}

/// One-sided Jacobi SVD of a matrix with at least as many rows as columns: rotate pairs of
/// columns until all are orthogonal, then read the singular values as the column norms.
fn svd_tall(
    a: &Matrix,
    mode: SvdVectors,
    meter: &mut dyn FnMut(u64) -> PyResult<()>,
) -> PyResult<Svd> {
    let (m, n) = (a.rows, a.columns);
    let mut u = a.clone();
    let mut v = Matrix::identity(n);
    let cost = (m as u64)
        .saturating_mul((n as u64).saturating_pow(2))
        .saturating_add(1);
    let mut converged = n <= 1;
    for _ in 0..MAX_SWEEPS {
        if converged {
            break;
        }
        meter(cost)?;
        let mut rotated = false;
        for p in 0..n - 1 {
            for q in p + 1..n {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for row in 0..m {
                    let (up, uq) = (u.get(row, p), u.get(row, q));
                    alpha += up * up;
                    beta += uq * uq;
                    gamma += up * uq;
                }
                if gamma.is_nan() || alpha.is_nan() || beta.is_nan() {
                    return Err(PyError::exception("LinAlgError", "SVD did not converge"));
                }
                if gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + 1.0f64.hypot(zeta));
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for row in 0..m {
                    let (up, uq) = (u.get(row, p), u.get(row, q));
                    u.set(row, p, c * up - s * uq);
                    u.set(row, q, s * up + c * uq);
                }
                if mode != SvdVectors::None {
                    for row in 0..n {
                        let (vp, vq) = (v.get(row, p), v.get(row, q));
                        v.set(row, p, c * vp - s * vq);
                        v.set(row, q, s * vp + c * vq);
                    }
                }
            }
        }
        converged = !rotated;
    }
    if !converged {
        return Err(PyError::exception("LinAlgError", "SVD did not converge"));
    }
    let norms = (0..n)
        .map(|column| norm2(&u.column(column)))
        .collect::<Vec<_>>();
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|left, right| norms[*right].total_cmp(&norms[*left]));
    let values = order.iter().map(|index| norms[*index]).collect::<Vec<_>>();
    if mode == SvdVectors::None {
        return Ok(Svd {
            values,
            vectors: None,
        });
    }
    let columns = if mode == SvdVectors::Full { m } else { n };
    let mut left = Matrix::zeros(m, columns);
    let mut missing = (n..columns).collect::<Vec<_>>();
    for (target, source) in order.iter().enumerate() {
        let norm = norms[*source];
        if norm > 0.0 {
            let column = u.column(*source);
            let normalized = column.iter().map(|value| value / norm).collect::<Vec<_>>();
            left.set_column(target, &normalized);
        } else {
            missing.push(target);
        }
    }
    missing.sort_unstable();
    complete_orthonormal(&mut left, &missing);
    let vt = v.select_columns(&order).transpose();
    Ok(Svd {
        values,
        vectors: Some((left, vt)),
    })
}

/// Fill the listed columns of `q`, whose other columns are orthonormal, so that every column
/// is. Each new column is the standard basis vector with the largest component orthogonal to
/// the columns so far, orthogonalized twice by modified Gram-Schmidt.
fn complete_orthonormal(q: &mut Matrix, missing: &[usize]) {
    let rows = q.rows;
    let mut basis = (0..q.columns)
        .filter(|column| !missing.contains(column))
        .collect::<Vec<_>>();
    for &target in missing {
        let mut best: Option<(f64, Vec<f64>)> = None;
        for unit in 0..rows {
            let mut candidate = vec![0.0; rows];
            candidate[unit] = 1.0;
            for _ in 0..2 {
                for &column in &basis {
                    let projection: f64 = (0..rows)
                        .map(|row| q.get(row, column) * candidate[row])
                        .sum();
                    for (row, value) in candidate.iter_mut().enumerate() {
                        *value -= projection * q.get(row, column);
                    }
                }
            }
            let norm = norm2(&candidate);
            if best.as_ref().is_none_or(|(largest, _)| norm > *largest) {
                best = Some((norm, candidate));
            }
        }
        let Some((norm, candidate)) = best else {
            return;
        };
        let normalized = candidate
            .iter()
            .map(|value| value / norm)
            .collect::<Vec<_>>();
        q.set_column(target, &normalized);
        basis.push(target);
    }
}

/// The minimum-norm least-squares solution of `A X = B` through the SVD, with `dgelsd`'s rank
/// rule: singular values at most `rcond` times the largest count as zero.
pub(super) struct LeastSquares {
    pub solution: Matrix,
    /// Squared residual norms of each column of `B`, when `A` has more rows than columns and
    /// full column rank; zeros otherwise.
    pub residuals: Vec<f64>,
    pub rank: usize,
    pub singular_values: Vec<f64>,
}

pub(super) fn least_squares(
    a: &Matrix,
    b: &Matrix,
    rcond: f64,
    meter: &mut dyn FnMut(u64) -> PyResult<()>,
) -> PyResult<LeastSquares> {
    let (m, n) = (a.rows, a.columns);
    let decomposition = svd(a, SvdVectors::Reduced, meter)?;
    let values = decomposition.values;
    let (u, vt) = decomposition
        .vectors
        .expect("reduced vectors were requested");
    let rcond = if rcond < 0.0 { f64::EPSILON } else { rcond };
    let cutoff = values.first().map_or(0.0, |largest| rcond * largest);
    let rank = values.iter().filter(|value| **value > cutoff).count();
    let mut solution = Matrix::zeros(n, b.columns);
    for column in 0..b.columns {
        for (index, value) in values.iter().enumerate().take(rank) {
            let mut coefficient = 0.0;
            for row in 0..m {
                coefficient += u.get(row, index) * b.get(row, column);
            }
            coefficient /= value;
            for row in 0..n {
                let updated = solution.get(row, column) + coefficient * vt.get(index, row);
                solution.set(row, column, updated);
            }
        }
    }
    let mut residuals = vec![0.0; b.columns];
    if m > n && rank == n {
        for (column, residual) in residuals.iter_mut().enumerate() {
            *residual = (0..m)
                .map(|row| {
                    let fitted: f64 = (0..n)
                        .map(|k| a.get(row, k) * solution.get(k, column))
                        .sum();
                    (b.get(row, column) - fitted).powi(2)
                })
                .sum();
        }
    }
    Ok(LeastSquares {
        solution,
        residuals,
        rank,
        singular_values: values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unmetered(_: u64) -> PyResult<()> {
        Ok(())
    }

    fn product(left: &Matrix, right: &Matrix) -> Matrix {
        let mut result = Matrix::zeros(left.rows, right.columns);
        for row in 0..left.rows {
            for column in 0..right.columns {
                let value = (0..left.columns)
                    .map(|k| left.get(row, k) * right.get(k, column))
                    .sum();
                result.set(row, column, value);
            }
        }
        result
    }

    fn assert_close(left: &Matrix, right: &Matrix, tolerance: f64) {
        assert_eq!((left.rows, left.columns), (right.rows, right.columns));
        for (a, b) in left.data.iter().zip(&right.data) {
            assert!((a - b).abs() <= tolerance, "{left:?} != {right:?}");
        }
    }

    #[test]
    fn determinants_round_as_numpy_does() {
        // NumPy prints -2.0000000000000004 and 10.000000000000002: exp(log|det|) rounds.
        let lu_a = lu(Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]));
        let (sign, logdet) = lu_a.slogdet();
        assert_eq!(sign * logdet.exp(), -2.000_000_000_000_000_4);
        let lu_b = lu(Matrix::new(2, 2, vec![4.0, 7.0, 2.0, 6.0]));
        let (sign, logdet) = lu_b.slogdet();
        assert_eq!(sign * logdet.exp(), 10.000_000_000_000_002);
        assert!(lu(Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0])).singular);
        assert!(lu(Matrix::new(2, 2, vec![f64::NAN, 1.0, 1.0, 1.0])).singular);
    }

    #[test]
    fn lu_solve_pivots_and_inverts() {
        let a = Matrix::new(3, 3, vec![2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0]);
        let factors = lu(a);
        let mut inverse = Matrix::identity(3);
        factors.solve(&mut inverse);
        let expected = vec![0.0, 0.0, 1.0, -2.0, 1.0, 3.0, 3.0, -1.0, -5.0];
        assert_close(&inverse, &Matrix::new(3, 3, expected), 1e-15);
    }

    #[test]
    fn cholesky_matches_lapack_rounding_and_rejects_indefinite_matrices() {
        let a = Matrix::new(3, 3, vec![4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0]);
        let lower = cholesky(&a).expect("positive definite");
        assert_eq!(&lower.data[..8], &[2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.3, 0.35]);
        assert_close(&product(&lower, &lower.transpose()), &a, 1e-15);
        assert!(cholesky(&Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 1.0])).is_none());
    }

    #[test]
    fn householder_qr_uses_lapack_reflector_signs() {
        let a = Matrix::new(
            3,
            3,
            vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
        );
        let (factors, tau) = householder_qr(a.clone());
        let r = (0..3)
            .map(|index| factors.get(index, index))
            .collect::<Vec<_>>();
        assert_close(
            &Matrix::new(1, 3, r),
            &Matrix::new(1, 3, vec![-14.0, -175.0, -35.0]),
            1e-12,
        );
        let q = householder_q(&factors, &tau, 3);
        let mut upper = factors.clone();
        for row in 0..3 {
            for column in 0..row {
                upper.set(row, column, 0.0);
            }
        }
        assert_close(&product(&q, &upper), &a, 1e-12);
        assert_close(&product(&q.transpose(), &q), &Matrix::identity(3), 1e-15);
    }

    #[test]
    fn jacobi_eigenvalues_ascend_with_orthonormal_vectors() {
        let a = Matrix::new(3, 3, vec![4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0]);
        let (values, vectors) = symmetric_eigen(&a, true, &mut unmetered).unwrap();
        let expected = [2.372_468_98, 2.722_508_21, 6.905_022_81];
        for (value, expected) in values.iter().zip(expected) {
            assert!((value - expected).abs() < 1e-8);
        }
        let v = vectors.unwrap();
        assert_close(&product(&v.transpose(), &v), &Matrix::identity(3), 1e-14);
    }

    #[test]
    fn svd_descends_and_completes_full_bases() {
        let a = Matrix::new(2, 3, vec![3.0, 2.0, 2.0, 2.0, 3.0, -2.0]);
        let result = svd(&a, SvdVectors::Full, &mut unmetered).unwrap();
        assert_close(
            &Matrix::new(1, 2, result.values.clone()),
            &Matrix::new(1, 2, vec![5.0, 3.0]),
            1e-14,
        );
        let (u, vt) = result.vectors.unwrap();
        assert_eq!((u.rows, u.columns, vt.rows, vt.columns), (2, 2, 3, 3));
        assert_close(&product(&vt, &vt.transpose()), &Matrix::identity(3), 1e-14);
        let mut sigma = Matrix::zeros(2, 3);
        sigma.set(0, 0, 5.0);
        sigma.set(1, 1, 3.0);
        assert_close(&product(&product(&u, &sigma), &vt), &a, 1e-13);
    }

    #[test]
    fn least_squares_fits_a_line_and_reports_residuals() {
        let a = Matrix::new(4, 2, vec![0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0]);
        let b = Matrix::new(4, 1, vec![-1.0, 0.2, 0.9, 2.1]);
        let fit = least_squares(&a, &b, -1.0, &mut unmetered).unwrap();
        assert_close(&fit.solution, &Matrix::new(2, 1, vec![1.0, -0.95]), 1e-12);
        assert!((fit.residuals[0] - 0.05).abs() < 1e-12);
        assert_eq!(fit.rank, 2);
    }
}
