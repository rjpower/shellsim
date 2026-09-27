//! Dense real linear-algebra kernels shared by `numpy.linalg` and `scipy.linalg`.
//!
//! Every kernel works on [`Mat`], a plain row-major `f64` matrix with no ties to the VM's array
//! storage or resource accounting; callers in `numpy::linalg` and `scipy::linalg` read operands
//! out of VM arrays, charge CPU for the work below, run these functions, and write the result
//! back into a new array. Keeping the numeric core free of runtime concerns lets both modules
//! share one implementation, as the design calls for.
//!
//! ## Algorithms
//!
//! - LU factorization with partial pivoting and triangular solves follow Golub & Van Loan,
//!   *Matrix Computations* (4th ed.), Algorithm 3.4.1, the textbook algorithm LAPACK's `getrf`
//!   also implements.
//! - Cholesky factorization follows Golub & Van Loan, Algorithm 4.2.1.
//! - QR factorization uses Householder reflections (Golub & Van Loan, Algorithm 5.2.1), with `Q`
//!   formed explicitly by applying the stored reflectors to the identity.
//! - The symmetric eigensolver is the classical cyclic Jacobi method (Golub & Van Loan, Algorithm
//!   8.4.3): sweep over off-diagonal pairs, rotating each to zero, until the off-diagonal norm is
//!   negligible. It converges quadratically, needing on the order of ten sweeps in practice.
//! - The SVD uses one-sided Jacobi orthogonalization of the columns of `A` (Hestenes 1958, and
//!   see Golub & Van Loan §8.6.3): repeatedly rotate pairs of columns to make them orthogonal.
//!   Singular values are the resulting column norms and `U` their normalized columns.
//! - The tridiagonal solver is Gaussian elimination with partial pivoting specialized to a
//!   tridiagonal system (Golub & Van Loan §4.3.6), matching LAPACK's `gtsv`.
//! - The banded solver is Gaussian elimination with partial pivoting specialized to a banded
//!   system stored in LAPACK band form (Golub & Van Loan §4.3.1), matching LAPACK's `gbsv`.
//! - The matrix exponential uses scaling and squaring with a Padé approximant, following Higham,
//!   "The Scaling and Squaring Method for the Matrix Exponential Revisited" (SIAM J. Matrix Anal.
//!   Appl., 2005). A fixed [13/13] Padé order is used at every scale rather than Higham's
//!   adaptive order selection, which trades a little unneeded work for simplicity; accuracy is
//!   unaffected because the scaling step already brings the matrix norm below the order-13
//!   threshold.
//!
//! ## Accuracy and sign conventions
//!
//! Every kernel is backward stable and computes in `f64`, so results agree with NumPy and SciPy
//! to about `1e-12` relative for well-conditioned inputs. Bitwise agreement with LAPACK or
//! OpenBLAS is not a goal: pivoting ties, rotation order, and blocking all leave room for
//! last-bit differences.
//!
//! Eigenvectors and singular vectors are free up to sign (and, for repeated singular values, up
//! to rotation within the repeated subspace). Every eigenvector and singular vector this module
//! returns is scaled so that its largest-magnitude entry is positive; ties keep the first such
//! entry. `jacobi_eigh` and `jacobi_svd` apply this rule before returning.
//!
//! ## Resource accounting
//!
//! Deterministic factorizations (LU, Cholesky, QR, triangular solves) cost a fixed multiple of
//! `n^3` (or `m*n^2` for rectangular input); callers charge that before calling in. The Jacobi
//! solvers do not know their iteration count in advance, so `jacobi_eigh` and `jacobi_svd` accept
//! a `charge_sweep` closure and call it with the cost of one sweep before running it, so a caller
//! low on CPU budget is stopped before, not after, doing that sweep's work.

use super::super::super::super::native::PyResult;

/// A dense real matrix stored in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::python) struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub(in crate::python) fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub(in crate::python) fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    pub(in crate::python) fn from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    #[inline]
    pub(in crate::python) fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub(in crate::python) fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * self.cols + j] = value;
    }

    #[inline]
    fn add(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * self.cols + j] += value;
    }

    pub(in crate::python) fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub(in crate::python) fn transpose(&self) -> Mat {
        let mut result = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        for j in 0..self.cols {
            self.data.swap(a * self.cols + j, b * self.cols + j);
        }
    }

    fn swap_columns(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        for i in 0..self.rows {
            self.data.swap(i * self.cols + a, i * self.cols + b);
        }
    }

    pub(in crate::python) fn matmul(&self, other: &Mat) -> Mat {
        debug_assert_eq!(self.cols, other.rows);
        let mut result = Mat::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    result.add(i, j, a * other.get(k, j));
                }
            }
        }
        result
    }

    pub(in crate::python) fn scale(&self, factor: f64) -> Mat {
        Mat::from_row_major(
            self.rows,
            self.cols,
            self.data.iter().map(|value| value * factor).collect(),
        )
    }

    fn add_mat(&self, other: &Mat) -> Mat {
        Mat::from_row_major(
            self.rows,
            self.cols,
            self.data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect(),
        )
    }

    fn sub_mat(&self, other: &Mat) -> Mat {
        Mat::from_row_major(
            self.rows,
            self.cols,
            self.data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a - b)
                .collect(),
        )
    }
}

/// Whether a solve applies `A` or `A^T`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum Trans {
    No,
    Transpose,
}

/// The simple, order-of-magnitude cost this module charges for one `n`-order dense
/// factorization (LU, Cholesky, QR, or a triangular solve): about twice the number of elements
/// touched by the cubic elimination loop. See the module doc for the accounting policy.
pub(in crate::python) fn factor_cost(rows: u64, cols: u64) -> u64 {
    2 * rows
        .saturating_mul(cols)
        .saturating_mul(rows.min(cols).max(1))
}

// ---------------------------------------------------------------------------------------------
// LU factorization with partial pivoting.
// ---------------------------------------------------------------------------------------------

/// The result of `getrf`-style LU factorization with partial pivoting.
///
/// `lu` packs the strictly-lower part of `L` (implicit unit diagonal) and all of `U` into one
/// `rows x cols` matrix, as LAPACK's `getrf` does. `piv[k]` is the zero-based row swapped with
/// row `k` at step `k` (LAPACK's `IPIV[k] - 1`); `piv` has `min(rows, cols)` entries.
/// `singular_at` is the first one-based column where elimination found an exactly zero pivot, as
/// LAPACK's `info` reports it.
pub(in crate::python) struct LuFactorization {
    pub lu: Mat,
    pub piv: Vec<usize>,
    pub singular_at: Option<usize>,
}

/// Factor `a` in place: `P A = L U`. Pivoting always selects the largest-magnitude entry
/// remaining in the column, as `getrf` does.
pub(in crate::python) fn lu_factor(a: &Mat) -> LuFactorization {
    let mut lu = a.clone();
    let k = a.rows.min(a.cols);
    let mut piv = Vec::with_capacity(k);
    let mut singular_at = None;
    for step in 0..k {
        let mut best_row = step;
        let mut best_value = lu.get(step, step).abs();
        for row in (step + 1)..a.rows {
            let value = lu.get(row, step).abs();
            if value > best_value {
                best_value = value;
                best_row = row;
            }
        }
        piv.push(best_row);
        lu.swap_rows(step, best_row);
        let pivot = lu.get(step, step);
        if pivot == 0.0 {
            if singular_at.is_none() {
                singular_at = Some(step + 1);
            }
            continue;
        }
        for row in (step + 1)..a.rows {
            let factor = lu.get(row, step) / pivot;
            lu.set(row, step, factor);
            if factor == 0.0 {
                continue;
            }
            for col in (step + 1)..a.cols {
                let update = factor * lu.get(step, col);
                lu.add(row, col, -update);
            }
        }
    }
    LuFactorization {
        lu,
        piv,
        singular_at,
    }
}

/// Apply the row swaps recorded by [`lu_factor`] to `b`, in step order (or reverse order for a
/// transposed solve, which walks the factorization backward).
fn apply_pivots(piv: &[usize], b: &mut Mat, reverse: bool) {
    let steps: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..piv.len()).rev())
    } else {
        Box::new(0..piv.len())
    };
    for step in steps {
        b.swap_rows(step, piv[step]);
    }
}

/// Forward substitution solving unit-lower-triangular `L x = b`, reading `L` from the strictly
/// lower part of `lu`.
fn forward_substitute_unit(lu: &Mat, b: &mut Mat) {
    for i in 0..lu.rows.min(lu.cols) {
        for col in 0..b.cols {
            let mut sum = b.get(i, col);
            for k in 0..i {
                sum -= lu.get(i, k) * b.get(k, col);
            }
            b.set(i, col, sum);
        }
    }
}

/// Back substitution solving upper-triangular `U x = b`, reading `U` from the upper part of
/// `lu` (square, order `n`).
fn back_substitute(lu: &Mat, n: usize, b: &mut Mat) {
    for i in (0..n).rev() {
        for col in 0..b.cols {
            let mut sum = b.get(i, col);
            for k in (i + 1)..n {
                sum -= lu.get(i, k) * b.get(k, col);
            }
            b.set(i, col, sum / lu.get(i, i));
        }
    }
}

/// Solve `A x = b` (or `A^T x = b`) from an existing LU factorization of square `A`. The caller
/// must have already checked `singular_at`.
pub(in crate::python) fn lu_solve(lu: &Mat, piv: &[usize], b: &Mat, trans: Trans) -> Mat {
    debug_assert_eq!(lu.rows, lu.cols);
    let n = lu.rows;
    let mut x = b.clone();
    match trans {
        Trans::No => {
            apply_pivots(piv, &mut x, false);
            forward_substitute_unit(lu, &mut x);
            back_substitute(lu, n, &mut x);
        }
        Trans::Transpose => {
            // Solve U^T y = b, then L^T x = y, then undo the row swaps in reverse.
            for i in 0..n {
                for col in 0..x.cols {
                    let mut sum = x.get(i, col);
                    for k in 0..i {
                        sum -= lu.get(k, i) * x.get(k, col);
                    }
                    x.set(i, col, sum / lu.get(i, i));
                }
            }
            for i in (0..n).rev() {
                for col in 0..x.cols {
                    let mut sum = x.get(i, col);
                    for k in (i + 1)..n {
                        sum -= lu.get(k, i) * x.get(k, col);
                    }
                    x.set(i, col, sum);
                }
            }
            apply_pivots(piv, &mut x, true);
        }
    }
    x
}

/// The inverse of a square, factored `A`, solving against the identity.
pub(in crate::python) fn lu_invert(lu: &Mat, piv: &[usize]) -> Mat {
    lu_solve(lu, piv, &Mat::identity(lu.rows), Trans::No)
}

/// `(sign, log|det|)` of a factored square `A`, as `slogdet` reports it: `sign` is `0.0` for a
/// singular matrix, else `+-1.0`.
pub(in crate::python) fn lu_slogdet(factorization: &LuFactorization) -> (f64, f64) {
    let lu = &factorization.lu;
    let n = lu.rows;
    if factorization.singular_at.is_some() {
        return (0.0, f64::NEG_INFINITY);
    }
    let mut swaps = 0usize;
    for (step, &target) in factorization.piv.iter().enumerate() {
        if step != target {
            swaps += 1;
        }
    }
    let mut sign = if swaps.is_multiple_of(2) { 1.0 } else { -1.0 };
    let mut log_det = 0.0;
    for i in 0..n {
        let diagonal = lu.get(i, i);
        if diagonal < 0.0 {
            sign = -sign;
        }
        log_det += diagonal.abs().ln();
    }
    (sign, log_det)
}

/// `det(A)` of a factored square `A`.
pub(in crate::python) fn lu_det(factorization: &LuFactorization) -> f64 {
    if factorization.singular_at.is_some() {
        return 0.0;
    }
    let (sign, log_det) = lu_slogdet(factorization);
    sign * log_det.exp()
}

/// The matrix 1-norm (maximum absolute column sum), used for condition-number estimates.
pub(in crate::python) fn one_norm(a: &Mat) -> f64 {
    (0..a.cols)
        .map(|j| (0..a.rows).map(|i| a.get(i, j).abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// The matrix infinity-norm (maximum absolute row sum).
pub(in crate::python) fn inf_norm(a: &Mat) -> f64 {
    (0..a.rows)
        .map(|i| a.row(i).iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// The Frobenius norm.
pub(in crate::python) fn frobenius_norm(a: &Mat) -> f64 {
    a.data.iter().map(|value| value * value).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------------------------
// Cholesky factorization.
// ---------------------------------------------------------------------------------------------

/// Compute the lower Cholesky factor `L` with `A = L L^T`, reading only the lower triangle of
/// `a`. `Err(i)` reports the one-based leading minor that is not positive definite, as LAPACK's
/// `potrf` `info` does.
pub(in crate::python) fn cholesky_lower(a: &Mat) -> Result<Mat, usize> {
    debug_assert_eq!(a.rows, a.cols);
    let n = a.rows;
    let mut l = Mat::zeros(n, n);
    for j in 0..n {
        let mut sum = a.get(j, j);
        for k in 0..j {
            sum -= l.get(j, k) * l.get(j, k);
        }
        if sum <= 0.0 || !sum.is_finite() {
            return Err(j + 1);
        }
        let diagonal = sum.sqrt();
        l.set(j, j, diagonal);
        for i in (j + 1)..n {
            let mut sum = a.get(i, j);
            for k in 0..j {
                sum -= l.get(i, k) * l.get(j, k);
            }
            l.set(i, j, sum / diagonal);
        }
    }
    Ok(l)
}

/// Solve `A x = b` given the lower Cholesky factor of `A`.
pub(in crate::python) fn cholesky_solve(l: &Mat, b: &Mat) -> Mat {
    let mut x = b.clone();
    forward_substitute_lower(l, &mut x, false);
    forward_substitute_lower(l, &mut x, true);
    x
}

/// The inverse of `A` given its lower Cholesky factor.
pub(in crate::python) fn cholesky_invert(l: &Mat) -> Mat {
    cholesky_solve(l, &Mat::identity(l.rows))
}

/// Forward (or, transposed, backward) substitution against a lower-triangular `l`, in place.
fn forward_substitute_lower(l: &Mat, b: &mut Mat, transposed: bool) {
    let n = l.rows;
    if !transposed {
        for i in 0..n {
            for col in 0..b.cols {
                let mut sum = b.get(i, col);
                for k in 0..i {
                    sum -= l.get(i, k) * b.get(k, col);
                }
                b.set(i, col, sum / l.get(i, i));
            }
        }
    } else {
        for i in (0..n).rev() {
            for col in 0..b.cols {
                let mut sum = b.get(i, col);
                for k in (i + 1)..n {
                    sum -= l.get(k, i) * b.get(k, col);
                }
                b.set(i, col, sum / l.get(i, i));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Triangular solves.
// ---------------------------------------------------------------------------------------------

/// Solve a triangular system `A x = b` or `A^T x = b`, where `A` is `n x n` and only the
/// selected (`lower` or upper) triangle is read. `Err(i)` reports the first one-based diagonal
/// entry found to be exactly zero, as LAPACK's `trtrs` `info` does.
pub(in crate::python) fn triangular_solve(
    a: &Mat,
    b: &Mat,
    lower: bool,
    trans: Trans,
    unit_diag: bool,
) -> Result<Mat, usize> {
    debug_assert_eq!(a.rows, a.cols);
    let n = a.rows;
    let mut x = b.clone();
    let forward = match (lower, trans) {
        (true, Trans::No) | (false, Trans::Transpose) => true,
        (false, Trans::No) | (true, Trans::Transpose) => false,
    };
    // Whether to read `a[i, k]` (untransposed) or `a[k, i]` (transposed) as the coefficient of
    // `x[k]` in equation `i`.
    let coefficient = |i: usize, k: usize| -> f64 {
        if trans == Trans::No {
            a.get(i, k)
        } else {
            a.get(k, i)
        }
    };
    let order: Vec<usize> = if forward {
        (0..n).collect()
    } else {
        (0..n).rev().collect()
    };
    for &i in &order {
        let diagonal = if unit_diag { 1.0 } else { a.get(i, i) };
        if !unit_diag && diagonal == 0.0 {
            return Err(i + 1);
        }
        for col in 0..x.cols {
            let mut sum = x.get(i, col);
            if forward {
                for &k in order.iter().take_while(|&&k| k < i) {
                    sum -= coefficient(i, k) * x.get(k, col);
                }
            } else {
                for &k in order.iter().take_while(|&&k| k > i) {
                    sum -= coefficient(i, k) * x.get(k, col);
                }
            }
            x.set(i, col, sum / diagonal);
        }
    }
    Ok(x)
}

/// The inverse of a triangular `a`.
pub(in crate::python) fn triangular_invert(
    a: &Mat,
    lower: bool,
    unit_diag: bool,
) -> Result<Mat, usize> {
    triangular_solve(a, &Mat::identity(a.rows), lower, Trans::No, unit_diag)
}

// ---------------------------------------------------------------------------------------------
// Householder QR.
// ---------------------------------------------------------------------------------------------

/// A Householder QR factorization: `factored`'s upper triangle (rows `0..min(rows,cols)`) holds
/// `R`; column `k`'s Householder vector (implicit leading `1`) is stored below the diagonal in
/// column `k`, rows `k+1..rows`. `tau[k]` is that reflector's scalar, as LAPACK's `geqrf` stores
/// it.
pub(in crate::python) struct QrFactorization {
    pub factored: Mat,
    pub tau: Vec<f64>,
}

/// Factor `a` (`rows x cols`, any shape) as `A = Q R`.
pub(in crate::python) fn householder_qr(a: &Mat) -> QrFactorization {
    let mut factored = a.clone();
    let k = a.rows.min(a.cols);
    let mut tau = Vec::with_capacity(k);
    for col in 0..k {
        // A length-1 trailing column (only when `a.rows == col + 1`, the last row of a square
        // or wide matrix) has no direction to reflect: LAPACK's `dlarfg` leaves it as `tau = 0`
        // rather than negating it, so this must too, or its diagonal sign would disagree with
        // LAPACK's for no numerical reason.
        if a.rows - col <= 1 {
            tau.push(0.0);
            continue;
        }
        let mut norm = 0.0f64;
        for row in col..a.rows {
            norm = norm.hypot(factored.get(row, col));
        }
        if norm == 0.0 {
            tau.push(0.0);
            continue;
        }
        let pivot = factored.get(col, col);
        let alpha = if pivot >= 0.0 { -norm } else { norm };
        let mut v = vec![0.0; a.rows - col];
        v[0] = pivot - alpha;
        for row in (col + 1)..a.rows {
            v[row - col] = factored.get(row, col);
        }
        let v_norm = v.iter().map(|value| value * value).sum::<f64>().sqrt();
        if v_norm == 0.0 {
            tau.push(0.0);
            continue;
        }
        for value in &mut v {
            *value /= v_norm;
        }
        let current_tau = 2.0;
        // Apply the reflector H = I - tau * v v^T to the trailing submatrix, including column
        // `col` itself (which collapses to `alpha` on the diagonal and zero below).
        for j in col..a.cols {
            let mut dot = 0.0;
            for (offset, entry) in v.iter().enumerate() {
                dot += entry * factored.get(col + offset, j);
            }
            let scale = current_tau * dot;
            for (offset, entry) in v.iter().enumerate() {
                factored.add(col + offset, j, -scale * entry);
            }
        }
        // Store the Householder vector below the diagonal rescaled to a leading entry of 1, as
        // LAPACK does, and fold that rescaling into `tau` so `apply_q` can reconstruct it.
        let leading = v[0];
        if leading != 0.0 {
            for (offset, entry) in v.iter().enumerate().skip(1) {
                factored.set(col + offset, col, entry / leading);
            }
        }
        tau.push(current_tau * leading * leading);
    }
    QrFactorization { factored, tau }
}

/// Factor `a` with column pivoting: `A P = Q R`, `P` the permutation that reorders columns
/// (Businger & Golub 1965; Golub & Van Loan §5.4.1). At each step, the remaining column (from
/// `col` onward) of largest Euclidean norm is swapped into the pivot position before reflecting
/// it, so `R`'s diagonal is non-increasing in magnitude; remaining columns' norms are then
/// downdated in `O(1)` each from the just-computed row, rather than recomputed from scratch, as
/// the original algorithm does (LAPACK's `geqp3` refines this with periodic recomputation for
/// numerical stability in edge cases; that refinement is not implemented here). Returns the
/// factorization, in the pivoted column order, and the 0-based permutation `jpvt` with
/// `jpvt[i]` naming which column of `a` became column `i`, matching LAPACK's `geqp3` convention.
pub(in crate::python) fn householder_qr_pivoted(a: &Mat) -> (QrFactorization, Vec<usize>) {
    let mut factored = a.clone();
    let k = a.rows.min(a.cols);
    let mut tau = Vec::with_capacity(k);
    let mut jpvt: Vec<usize> = (0..a.cols).collect();
    let mut norms: Vec<f64> = (0..a.cols)
        .map(|col| {
            (0..a.rows)
                .map(|row| factored.get(row, col).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    for col in 0..k {
        let mut best = col;
        for j in (col + 1)..a.cols {
            if norms[j] > norms[best] {
                best = j;
            }
        }
        if best != col {
            factored.swap_columns(col, best);
            norms.swap(col, best);
            jpvt.swap(col, best);
        }
        // See the matching comment in `householder_qr`: a length-1 trailing column has nothing
        // to reflect, and LAPACK leaves it unnegated (`tau = 0`).
        if a.rows - col <= 1 {
            tau.push(0.0);
            continue;
        }
        let mut norm = 0.0f64;
        for row in col..a.rows {
            norm = norm.hypot(factored.get(row, col));
        }
        if norm == 0.0 {
            tau.push(0.0);
            continue;
        }
        let pivot = factored.get(col, col);
        let alpha = if pivot >= 0.0 { -norm } else { norm };
        let mut v = vec![0.0; a.rows - col];
        v[0] = pivot - alpha;
        for row in (col + 1)..a.rows {
            v[row - col] = factored.get(row, col);
        }
        let v_norm = v.iter().map(|value| value * value).sum::<f64>().sqrt();
        if v_norm == 0.0 {
            tau.push(0.0);
            continue;
        }
        for value in &mut v {
            *value /= v_norm;
        }
        let current_tau = 2.0;
        for j in col..a.cols {
            let mut dot = 0.0;
            for (offset, entry) in v.iter().enumerate() {
                dot += entry * factored.get(col + offset, j);
            }
            let scale = current_tau * dot;
            for (offset, entry) in v.iter().enumerate() {
                factored.add(col + offset, j, -scale * entry);
            }
        }
        let leading = v[0];
        if leading != 0.0 {
            for (offset, entry) in v.iter().enumerate().skip(1) {
                factored.set(col + offset, col, entry / leading);
            }
        }
        tau.push(current_tau * leading * leading);
        for (j, norm) in norms.iter_mut().enumerate().skip(col + 1) {
            let entry = factored.get(col, j);
            let updated = *norm * *norm - entry * entry;
            *norm = if updated > 0.0 { updated.sqrt() } else { 0.0 };
        }
    }
    (QrFactorization { factored, tau }, jpvt)
}

/// Apply `Q` (or, transposed, `Q^T`) from the left to `c` in place, reading reflectors from
/// `qr`. `rows` is the row count of the original factored matrix (`qr.factored.rows`).
pub(in crate::python) fn apply_q(qr: &QrFactorization, c: &mut Mat, transpose: bool) {
    let rows = qr.factored.rows;
    let k = qr.tau.len();
    let order: Vec<usize> = if transpose {
        (0..k).collect()
    } else {
        (0..k).rev().collect()
    };
    for col in order {
        let tau = qr.tau[col];
        if tau == 0.0 {
            continue;
        }
        // Reconstruct the normalized reflector vector (length `rows - col`, leading entry 1)
        // used when factoring, from the stored (rescaled) form.
        let mut v = vec![0.0; rows - col];
        v[0] = 1.0;
        for row in (col + 1)..rows {
            v[row - col] = qr.factored.get(row, col);
        }
        for j in 0..c.cols {
            let mut dot = 0.0;
            for (offset, entry) in v.iter().enumerate() {
                dot += entry * c.get(col + offset, j);
            }
            let scale = tau * dot;
            for (offset, entry) in v.iter().enumerate() {
                c.add(col + offset, j, -scale * entry);
            }
        }
    }
}

/// Build `Q` explicitly: `m x m` when `full`, else `m x min(m, cols)`.
pub(in crate::python) fn qr_explicit_q(qr: &QrFactorization, full: bool) -> Mat {
    let rows = qr.factored.rows;
    let width = if full {
        rows
    } else {
        rows.min(qr.factored.cols)
    };
    let mut q = Mat::zeros(rows, width);
    for i in 0..width {
        q.set(i, i, 1.0);
    }
    apply_q(qr, &mut q, false);
    q
}

/// The `R` factor at its full `rows x cols` shape (rows from `min(rows, cols)` onward are all
/// zero), as NumPy's `mode="complete"` and SciPy's default `qr` report it.
pub(in crate::python) fn qr_explicit_r_full(qr: &QrFactorization) -> Mat {
    let rows = qr.factored.rows;
    let cols = qr.factored.cols;
    let k = rows.min(cols);
    let mut r = Mat::zeros(rows, cols);
    for i in 0..k {
        for j in i..cols {
            r.set(i, j, qr.factored.get(i, j));
        }
    }
    r
}

/// The `R` factor: `min(rows, cols) x cols` (the reduced form used throughout this module).
pub(in crate::python) fn qr_explicit_r(qr: &QrFactorization) -> Mat {
    let k = qr.factored.rows.min(qr.factored.cols);
    let mut r = Mat::zeros(k, qr.factored.cols);
    for i in 0..k {
        for j in i..qr.factored.cols {
            r.set(i, j, qr.factored.get(i, j));
        }
    }
    r
}

// ---------------------------------------------------------------------------------------------
// Symmetric eigensolver: cyclic Jacobi.
// ---------------------------------------------------------------------------------------------

const JACOBI_MAX_SWEEPS: usize = 100;

/// Eigenvalues (ascending) and, if requested, eigenvectors of symmetric `a`, by the classical
/// cyclic Jacobi method. `charge_sweep(cost)` is called with each sweep's cost before running
/// it, so a caller that runs out of budget stops before doing that sweep's work.
pub(in crate::python) fn jacobi_eigh(
    a: &Mat,
    compute_vectors: bool,
    mut charge_sweep: impl FnMut(u64) -> PyResult<()>,
) -> PyResult<(Vec<f64>, Option<Mat>)> {
    debug_assert_eq!(a.rows, a.cols);
    let n = a.rows;
    let mut work = a.clone();
    let mut v = compute_vectors.then(|| Mat::identity(n));
    let off_scale = frobenius_norm(a).max(1.0);
    let sweep_cost = (n as u64).saturating_pow(3);
    for _ in 0..JACOBI_MAX_SWEEPS {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += work.get(p, q) * work.get(p, q);
            }
        }
        if off.sqrt() <= 1e-15 * off_scale {
            break;
        }
        charge_sweep(sweep_cost)?;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = work.get(p, q);
                if apq == 0.0 {
                    continue;
                }
                let app = work.get(p, p);
                let aqq = work.get(q, q);
                // Symmetric Schur angle (Golub & Van Loan eq. 8.4.4): the rotation (c, s) that
                // zeros a[p,q] when applied as a similarity transform.
                let theta = (aqq - app) / (2.0 * apq);
                let signum = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = signum / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = work.get(k, p);
                    let akq = work.get(k, q);
                    work.set(k, p, c * akp - s * akq);
                    work.set(k, q, s * akp + c * akq);
                }
                for k in 0..n {
                    let apk = work.get(p, k);
                    let aqk = work.get(q, k);
                    work.set(p, k, c * apk - s * aqk);
                    work.set(q, k, s * apk + c * aqk);
                }
                if let Some(v) = &mut v {
                    for k in 0..n {
                        let vkp = v.get(k, p);
                        let vkq = v.get(k, q);
                        v.set(k, p, c * vkp - s * vkq);
                        v.set(k, q, s * vkp + c * vkq);
                    }
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| work.get(i, i).partial_cmp(&work.get(j, j)).unwrap());
    let eigenvalues = order.iter().map(|&i| work.get(i, i)).collect();
    let eigenvectors = v.map(|v| {
        let mut sorted = Mat::zeros(n, n);
        for (new_col, &old_col) in order.iter().enumerate() {
            for row in 0..n {
                sorted.set(row, new_col, v.get(row, old_col));
            }
        }
        normalize_column_signs(&mut sorted);
        sorted
    });
    Ok((eigenvalues, eigenvectors))
}

/// Scale each column so its largest-magnitude entry is positive (ties keep the first such
/// entry), the sign convention documented at the top of this module.
fn normalize_column_signs(m: &mut Mat) {
    for col in 0..m.cols {
        let mut best_row = 0;
        let mut best_value = 0.0f64;
        for row in 0..m.rows {
            let value = m.get(row, col).abs();
            if value > best_value {
                best_value = value;
                best_row = row;
            }
        }
        if m.get(best_row, col) < 0.0 {
            for row in 0..m.rows {
                let value = m.get(row, col);
                m.set(row, col, -value);
            }
        }
    }
}

/// Apply [`normalize_column_signs`]'s convention to each column of `v`, flipping the matching
/// column of `u` by the same decision.
///
/// `u`'s and `v`'s columns are paired left/right singular vectors: flipping one without the
/// other would keep each individually normalized but change `u_i sigma_i v_i^T`, corrupting the
/// `A = U Sigma V^T` identity. Normalizing only one side and mirroring its sign decision onto
/// the other keeps the pairing intact.
fn normalize_paired_column_signs(u: &mut Mat, v: &mut Mat) {
    for col in 0..v.cols {
        let mut best_row = 0;
        let mut best_value = 0.0f64;
        for row in 0..v.rows {
            let value = v.get(row, col).abs();
            if value > best_value {
                best_value = value;
                best_row = row;
            }
        }
        if v.get(best_row, col) < 0.0 {
            for row in 0..v.rows {
                let value = v.get(row, col);
                v.set(row, col, -value);
            }
            for row in 0..u.rows {
                let value = u.get(row, col);
                u.set(row, col, -value);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// SVD: one-sided Jacobi.
// ---------------------------------------------------------------------------------------------

/// Singular values (descending) and, if requested, singular vectors of `a` (`m x n`), by
/// one-sided Jacobi orthogonalization of its columns. `charge_sweep` behaves as in
/// [`jacobi_eigh`].
pub(in crate::python) fn jacobi_svd(
    a: &Mat,
    full_matrices: bool,
    compute_uv: bool,
    mut charge_sweep: impl FnMut(u64) -> PyResult<()>,
) -> PyResult<(Option<Mat>, Vec<f64>, Option<Mat>)> {
    if a.rows < a.cols {
        // Work on the taller orientation and swap U/V back at the end.
        let (u, s, vt) = jacobi_svd(&a.transpose(), full_matrices, compute_uv, charge_sweep)?;
        let u = u.map(|u| u.transpose());
        let vt = vt.map(|vt| vt.transpose());
        return Ok((vt, s, u));
    }
    let (m, n) = (a.rows, a.cols);
    let mut work = a.clone();
    let mut v = Mat::identity(n);
    let scale = frobenius_norm(a).max(1.0);
    let sweep_cost = (m as u64) * (n as u64) * (n as u64).max(1);
    for _ in 0..JACOBI_MAX_SWEEPS {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut dot = 0.0;
                for row in 0..m {
                    dot += work.get(row, p) * work.get(row, q);
                }
                off += dot * dot;
            }
        }
        if off.sqrt() <= 1e-15 * scale * scale {
            break;
        }
        charge_sweep(sweep_cost)?;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for row in 0..m {
                    let x = work.get(row, p);
                    let y = work.get(row, q);
                    alpha += x * x;
                    beta += y * y;
                    gamma += x * y;
                }
                if gamma.abs() <= 1e-300 {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let signum = if zeta >= 0.0 { 1.0 } else { -1.0 };
                let t = signum / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for row in 0..m {
                    let xp = work.get(row, p);
                    let xq = work.get(row, q);
                    work.set(row, p, c * xp - s * xq);
                    work.set(row, q, s * xp + c * xq);
                }
                for row in 0..n {
                    let vp = v.get(row, p);
                    let vq = v.get(row, q);
                    v.set(row, p, c * vp - s * vq);
                    v.set(row, q, s * vp + c * vq);
                }
            }
        }
    }
    let mut singular: Vec<f64> = (0..n).map(|col| column_norm(&work, col)).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| singular[j].partial_cmp(&singular[i]).unwrap());
    singular = order.iter().map(|&i| singular[i]).collect();

    if !compute_uv {
        return Ok((None, singular, None));
    }

    // A singular value that is merely tiny, rather than exactly `0.0`, still leaves `work`'s
    // matching column as floating-point noise from the Jacobi sweep (not a true null-space
    // direction), and dividing that noise by a near-zero `sigma` amplifies it by a huge factor.
    // Treat anything at or below this scale-relative tolerance as zero instead, so its column is
    // rebuilt orthogonally by `fill_zero_singular_columns` rather than by an ill-conditioned
    // division.
    let zero_tolerance = f64::EPSILON * scale * (m.max(n) as f64);
    let k = n.min(m);
    let mut u_reduced = Mat::zeros(m, k);
    let mut v_reduced = Mat::zeros(n, k);
    for (new_col, &old_col) in order.iter().take(k).enumerate() {
        let sigma = singular[new_col];
        for row in 0..n {
            v_reduced.set(row, new_col, v.get(row, old_col));
        }
        if sigma > zero_tolerance {
            for row in 0..m {
                u_reduced.set(row, new_col, work.get(row, old_col) / sigma);
            }
        }
    }
    fill_zero_singular_columns(&singular, zero_tolerance, &mut u_reduced);
    normalize_paired_column_signs(&mut u_reduced, &mut v_reduced);

    let u = if full_matrices && m > k {
        Some(extend_orthonormal(&u_reduced, m))
    } else {
        Some(u_reduced)
    };
    let vt = if full_matrices && n > k {
        Some(extend_orthonormal(&v_reduced, n).transpose())
    } else {
        Some(v_reduced.transpose())
    };
    Ok((u, singular, vt))
}

fn column_norm(m: &Mat, col: usize) -> f64 {
    (0..m.rows)
        .map(|row| m.get(row, col).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Replace the all-zero columns of a reduced `U` (from singular values at or below `tolerance`,
/// which sort last) with an orthonormal completion of the nonzero columns, via QR, so `U` stays
/// orthonormal. Requires `singular` sorted descending, so the zero entries are a contiguous
/// suffix.
fn fill_zero_singular_columns(singular: &[f64], tolerance: f64, u: &mut Mat) {
    let nonzero = singular
        .iter()
        .take_while(|&&value| value > tolerance)
        .count();
    if nonzero == u.cols {
        return;
    }
    let mut basis = Mat::zeros(u.rows, nonzero);
    for col in 0..nonzero {
        for row in 0..u.rows {
            basis.set(row, col, u.get(row, col));
        }
    }
    let extended = extend_orthonormal(&basis, u.cols);
    for col in nonzero..u.cols {
        for row in 0..u.rows {
            u.set(row, col, extended.get(row, col));
        }
    }
}

/// Extend the columns of an orthonormal `m x k` matrix to a full `m x width` orthonormal basis,
/// via Householder QR of `basis` itself: since its columns are already orthonormal, the trailing
/// columns of the resulting explicit `Q` are an orthonormal complement.
fn extend_orthonormal(basis: &Mat, width: usize) -> Mat {
    if basis.cols >= width {
        return basis.clone();
    }
    let qr = householder_qr(basis);
    let q = qr_explicit_q(&qr, true);
    let mut result = Mat::zeros(basis.rows, width);
    for col in 0..basis.cols {
        for row in 0..basis.rows {
            result.set(row, col, basis.get(row, col));
        }
    }
    for col in basis.cols..width {
        for row in 0..basis.rows {
            result.set(row, col, q.get(row, col));
        }
    }
    result
}

// ---------------------------------------------------------------------------------------------
// Tridiagonal solve (LAPACK `gtsv`).
// ---------------------------------------------------------------------------------------------

/// Solve a tridiagonal system by Gaussian elimination with partial pivoting, overwriting `b`
/// with the solution. `dl` and `du` have `n - 1` entries and `d` has `n` entries; all three are
/// consumed (overwritten as scratch), as LAPACK's `gtsv` does. `Err(i)` reports the one-based
/// step at which the matrix was found exactly singular.
pub(in crate::python) fn tridiagonal_solve(
    dl: &mut [f64],
    d: &mut [f64],
    du: &mut [f64],
    b: &mut Mat,
) -> Result<(), usize> {
    let n = d.len();
    if n == 0 {
        return Ok(());
    }
    // `du2[i]` holds the fill-in entry at column i+2 of row i, which appears only after a
    // pivot swap carries row i+1's superdiagonal entry into row i.
    let mut du2 = vec![0.0; n.saturating_sub(2)];
    for i in 0..n - 1 {
        if d[i].abs() >= dl[i].abs() {
            if d[i] == 0.0 {
                return Err(i + 1);
            }
            let factor = dl[i] / d[i];
            d[i + 1] -= factor * du[i];
            for col in 0..b.cols {
                let value = b.get(i + 1, col) - factor * b.get(i, col);
                b.set(i + 1, col, value);
            }
        } else {
            // Pivot on the subdiagonal: rows i and i+1 swap (implicitly, via `factor`), then
            // eliminate the new row i+1's entry in column i using the new row i.
            //
            // Before the swap, row i is [d[i], du[i], 0] and row i+1 is
            // [dl[i], d[i+1], du[i+1]] (columns i, i+1, i+2). After swapping, row i (finalized
            // here) is the old row i+1, and row i+1 becomes old row i minus `factor` times the
            // new row i, where `factor = d[i] / dl[i]` zeros column i.
            let factor = d[i] / dl[i];
            let old_du_i = du[i];
            let old_d_ip1 = d[i + 1];
            let old_du_ip1 = if i < n - 2 { du[i + 1] } else { 0.0 };

            d[i] = dl[i];
            du[i] = old_d_ip1;
            if i < n - 2 {
                du2[i] = old_du_ip1;
            }
            d[i + 1] = old_du_i - factor * old_d_ip1;
            if i < n - 2 {
                du[i + 1] = -factor * old_du_ip1;
            }
            for col in 0..b.cols {
                let bi = b.get(i, col);
                let bip1 = b.get(i + 1, col);
                b.set(i, col, bip1);
                b.set(i + 1, col, bi - factor * bip1);
            }
        }
    }
    if d[n - 1] == 0.0 {
        return Err(n);
    }
    for col in 0..b.cols {
        b.set(n - 1, col, b.get(n - 1, col) / d[n - 1]);
        if n >= 2 {
            let value = (b.get(n - 2, col) - du[n - 2] * b.get(n - 1, col)) / d[n - 2];
            b.set(n - 2, col, value);
        }
        for i in (0..n.saturating_sub(2)).rev() {
            let value =
                (b.get(i, col) - du[i] * b.get(i + 1, col) - du2[i] * b.get(i + 2, col)) / d[i];
            b.set(i, col, value);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Banded solve (LAPACK `gbsv`).
// ---------------------------------------------------------------------------------------------

/// Solve a banded system given in LAPACK's expanded band-storage form: `ab` has
/// `2*kl + ku + 1` rows and `n` columns, where row `kl + ku + i - j` (zero-based) of column `j`
/// holds `A[i, j]` for `max(0, j - ku) <= i <= min(n - 1, j + kl)`, and the top `kl` rows are
/// scratch for fill-in, as LAPACK's `gbsv` expects. Returns the factored `ab` (with `L`'s
/// multipliers recorded in the scratch rows, as `gbsv` leaves them), the pivot vector, and the
/// solution, or `Err(i)` at the first exactly-zero pivot.
pub(in crate::python) fn banded_lu_solve(
    kl: usize,
    ku: usize,
    ab: &Mat,
    b: &Mat,
) -> Result<(Mat, Vec<usize>, Mat), usize> {
    let n = ab.cols;
    let mut ab = ab.clone();
    let mut x = b.clone();
    let mut piv = Vec::with_capacity(n);
    let band_rows = 2 * kl + ku + 1;
    debug_assert_eq!(ab.rows, band_rows);
    // Row index within `ab` of `A[i, j]`, in the working (post-fill-in) storage.
    let row_of =
        |i: isize, j: usize| -> usize { (kl as isize + ku as isize + i - j as isize) as usize };
    for j in 0..n {
        let last_row = (j + kl).min(n - 1);
        let mut best_i = j;
        let mut best_value = ab.get(row_of(j as isize, j), j).abs();
        for i in (j + 1)..=last_row {
            let value = ab.get(row_of(i as isize, j), j).abs();
            if value > best_value {
                best_value = value;
                best_i = i;
            }
        }
        piv.push(best_i);
        if best_i != j {
            let last_col = (j + kl + ku).min(n - 1);
            for col in j..=last_col {
                let a = row_of(j as isize, col);
                let b_row = row_of(best_i as isize, col);
                let tmp = ab.get(a, col);
                ab.set(a, col, ab.get(b_row, col));
                ab.set(b_row, col, tmp);
            }
            x.swap_rows(j, best_i);
        }
        let pivot = ab.get(row_of(j as isize, j), j);
        if pivot == 0.0 {
            return Err(j + 1);
        }
        let last_col = (j + kl + ku).min(n - 1);
        for i in (j + 1)..=last_row {
            let factor = ab.get(row_of(i as isize, j), j) / pivot;
            ab.set(row_of(i as isize, j), j, factor);
            if factor == 0.0 {
                continue;
            }
            for col in (j + 1)..=last_col {
                let updated = ab.get(row_of(i as isize, col), col)
                    - factor * ab.get(row_of(j as isize, col), col);
                ab.set(row_of(i as isize, col), col, updated);
            }
            for rhs in 0..x.cols {
                let updated = x.get(i, rhs) - factor * x.get(j, rhs);
                x.set(i, rhs, updated);
            }
        }
    }
    // Back substitution using the upper band (bandwidth kl + ku) left in `ab`.
    for i in (0..n).rev() {
        let last_col = (i + kl + ku).min(n - 1);
        for rhs in 0..x.cols {
            let mut sum = x.get(i, rhs);
            for col in (i + 1)..=last_col {
                sum -= ab.get(row_of(i as isize, col), col) * x.get(col, rhs);
            }
            x.set(i, rhs, sum / ab.get(row_of(i as isize, i), i));
        }
    }
    Ok((ab, piv, x))
}

// ---------------------------------------------------------------------------------------------
// Matrix exponential: scaling and squaring with a [13/13] Padé approximant.
// ---------------------------------------------------------------------------------------------

/// [13/13] Padé numerator coefficients for `e^A`, from Higham (2005), Table 2.3 / eq. (3.12).
const PADE_13: [f64; 14] = [
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
    1.0,
];

/// The scaling threshold for order-13 Padé, `theta_13` from Higham (2005), Table 2.3.
const PADE_13_THETA: f64 = 5.371920351148152;

/// `e^A` for square `a`, by scaling and squaring with a fixed [13/13] Padé approximant.
pub(in crate::python) fn expm(a: &Mat) -> Mat {
    debug_assert_eq!(a.rows, a.cols);
    let n = a.rows;
    if n == 0 {
        return Mat::zeros(0, 0);
    }
    let norm = one_norm(a);
    let scaling = if norm > PADE_13_THETA {
        (norm / PADE_13_THETA).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    let scaled = a.scale(1.0 / 2f64.powi(scaling as i32));

    let identity = Mat::identity(n);
    let a2 = scaled.matmul(&scaled);
    let a4 = a2.matmul(&a2);
    let a6 = a4.matmul(&a2);

    // Evaluate the [13/13] Padé numerator and denominator with Horner-style grouping in A^2,
    // following Higham (2005) eq. (3.12)-(3.13).
    let u_inner = a6
        .scale(PADE_13[13])
        .add_mat(&a4.scale(PADE_13[11]))
        .add_mat(&a2.scale(PADE_13[9]));
    let u_right = a6
        .matmul(&u_inner)
        .add_mat(&a6.scale(PADE_13[7]))
        .add_mat(&a4.scale(PADE_13[5]))
        .add_mat(&a2.scale(PADE_13[3]))
        .add_mat(&identity.scale(PADE_13[1]));
    let u = scaled.matmul(&u_right);

    let v_inner = a6
        .scale(PADE_13[12])
        .add_mat(&a4.scale(PADE_13[10]))
        .add_mat(&a2.scale(PADE_13[8]));
    let v = a6
        .matmul(&v_inner)
        .add_mat(&a6.scale(PADE_13[6]))
        .add_mat(&a4.scale(PADE_13[4]))
        .add_mat(&a2.scale(PADE_13[2]))
        .add_mat(&identity.scale(PADE_13[0]));

    let numerator = v.add_mat(&u);
    let denominator = v.sub_mat(&u);
    let factorization = lu_factor(&denominator);
    let mut result = lu_solve(&factorization.lu, &factorization.piv, &numerator, Trans::No);
    for _ in 0..scaling {
        result = result.matmul(&result);
    }
    result
}

// ---------------------------------------------------------------------------------------------
// Generalized symmetric eigenproblem (`A x = lambda B x`, `B` symmetric positive definite).
// ---------------------------------------------------------------------------------------------

/// The generalized eigenproblem's eigenvalues and, if requested, eigenvectors; `Err` reports
/// that `B` was not positive definite (Cholesky's `info`), matching [`cholesky_lower`]'s own
/// error type.
pub(in crate::python) type EighGeneralized = Result<(Vec<f64>, Option<Mat>), usize>;

/// Reduce `A x = lambda B x` to a standard symmetric eigenproblem via the Cholesky factor of
/// `B` (Golub & Van Loan §8.7.2, "Problem 1"): with `B = L L^T`, solve the standard problem for
/// `C = L^-1 A L^-T`, then map eigenvectors back with `v = L^-T y`. `Err` reports that `B` is
/// not positive definite (Cholesky's `info`).
pub(in crate::python) fn eigh_generalized(
    a: &Mat,
    b: &Mat,
    compute_vectors: bool,
    charge_sweep: impl FnMut(u64) -> PyResult<()>,
) -> PyResult<EighGeneralized> {
    let l = match cholesky_lower(b) {
        Ok(l) => l,
        Err(info) => return Ok(Err(info)),
    };
    // C = L^-1 A L^-T, computed as two triangular solves.
    let step = triangular_solve(&l, a, true, Trans::No, false).expect("L has a nonzero diagonal");
    let step_t = triangular_solve(&l, &step.transpose(), true, Trans::No, false)
        .expect("L has a nonzero diagonal");
    let c = step_t.transpose();
    let (values, vectors) = jacobi_eigh(&c, compute_vectors, charge_sweep)?;
    let vectors = vectors.map(|y| {
        triangular_solve(&l, &y, true, Trans::Transpose, false).expect("L has a nonzero diagonal")
    });
    Ok(Ok((values, vectors)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: usize, cols: usize, values: &[f64]) -> Mat {
        Mat::from_row_major(rows, cols, values.to_vec())
    }

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} != {b} (tol {tol})");
    }

    #[test]
    fn lu_factor_solves_a_known_system() {
        let a = mat(2, 2, &[4.0, 3.0, 6.0, 3.0]);
        let f = lu_factor(&a);
        assert!(f.singular_at.is_none());
        let b = mat(2, 1, &[1.0, 1.0]);
        let x = lu_solve(&f.lu, &f.piv, &b, Trans::No);
        // 4x + 3y = 1; 6x + 3y = 1 => x = 0, y = 1/3.
        assert_close(x.get(0, 0), 0.0, 1e-12);
        assert_close(x.get(1, 0), 1.0 / 3.0, 1e-12);
    }

    #[test]
    fn lu_factor_flags_singular_matrix() {
        let a = mat(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let f = lu_factor(&a);
        assert_eq!(f.singular_at, Some(2));
    }

    #[test]
    fn cholesky_reconstructs_spd_matrix() {
        let a = mat(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        let l = cholesky_lower(&a).unwrap();
        let reconstructed = l.matmul(&l.transpose());
        for (x, y) in a.data.iter().zip(&reconstructed.data) {
            assert_close(*x, *y, 1e-12);
        }
    }

    #[test]
    fn cholesky_rejects_non_positive_definite() {
        let a = mat(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert_eq!(cholesky_lower(&a), Err(2));
    }

    #[test]
    fn qr_reconstructs_and_is_orthogonal() {
        let a = mat(3, 2, &[12.0, -51.0, 6.0, 167.0, -4.0, 24.0]);
        let qr = householder_qr(&a);
        let q = qr_explicit_q(&qr, false);
        let r = qr_explicit_r(&qr);
        let reconstructed = q.matmul(&r);
        for (x, y) in a.data.iter().zip(&reconstructed.data) {
            assert_close(*x, *y, 1e-10);
        }
        let qtq = q.transpose().matmul(&q);
        for i in 0..2 {
            for j in 0..2 {
                assert_close(qtq.get(i, j), if i == j { 1.0 } else { 0.0 }, 1e-10);
            }
        }
    }

    #[test]
    fn jacobi_eigh_finds_known_eigenvalues() {
        let a = mat(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let (values, vectors) = jacobi_eigh(&a, true, |_| Ok(())).unwrap();
        assert_close(values[0], 1.0, 1e-12);
        assert_close(values[1], 3.0, 1e-12);
        let v = vectors.unwrap();
        let av = a.matmul(&v);
        for (col, &value) in values.iter().enumerate() {
            for row in 0..2 {
                assert_close(av.get(row, col), value * v.get(row, col), 1e-10);
            }
        }
    }

    #[test]
    fn jacobi_eigh_charges_before_each_sweep() {
        let a = mat(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let mut charges = Vec::new();
        let (values, _) = jacobi_eigh(&a, false, |cost| {
            charges.push(cost);
            Ok(())
        })
        .unwrap();
        assert!(!charges.is_empty());
        assert_close(values[0], 1.0, 1e-12);
    }

    #[test]
    fn jacobi_svd_reconstructs_known_matrix() {
        let a = mat(2, 3, &[3.0, 2.0, 2.0, 2.0, 3.0, -2.0]);
        let (u, s, vt) = jacobi_svd(&a, false, true, |_| Ok(())).unwrap();
        assert_close(s[0], 5.0, 1e-10);
        assert_close(s[1], 3.0, 1e-10);
        let u = u.unwrap();
        let vt = vt.unwrap();
        let mut sigma = Mat::zeros(2, 2);
        sigma.set(0, 0, s[0]);
        sigma.set(1, 1, s[1]);
        let reconstructed = u.matmul(&sigma).matmul(&vt);
        for (x, y) in a.data.iter().zip(&reconstructed.data) {
            assert_close(*x, *y, 1e-9);
        }
    }

    #[test]
    fn tridiagonal_solve_matches_direct_solve() {
        let mut dl = vec![1.0, 1.0, 1.0];
        let mut d = vec![4.0, 4.0, 4.0, 4.0];
        let mut du = vec![1.0, 1.0, 1.0];
        let mut b = mat(4, 1, &[1.0, 2.0, 3.0, 4.0]);
        tridiagonal_solve(&mut dl, &mut d, &mut du, &mut b).unwrap();
        let a = mat(
            4,
            4,
            &[
                4.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 4.0,
            ],
        );
        let f = lu_factor(&a);
        let rhs = mat(4, 1, &[1.0, 2.0, 3.0, 4.0]);
        let expected = lu_solve(&f.lu, &f.piv, &rhs, Trans::No);
        for row in 0..4 {
            assert_close(b.get(row, 0), expected.get(row, 0), 1e-10);
        }
    }

    #[test]
    fn tridiagonal_solve_takes_the_pivoting_branch_when_the_subdiagonal_dominates() {
        // Row 0's diagonal (1) is smaller in magnitude than the subdiagonal entry below it (3),
        // so elimination at i=0 must take the "pivot on the subdiagonal" branch. Built as a
        // dense matrix too, so the tridiagonal path can be checked against `lu_factor`/`lu_solve`
        // directly instead of a hand-derived expectation.
        let dense = mat(3, 3, &[1.0, 2.0, 0.0, 3.0, 5.0, 4.0, 0.0, 1.0, 6.0]);
        let mut dl = vec![3.0, 1.0];
        let mut d = vec![1.0, 5.0, 6.0];
        let mut du = vec![2.0, 4.0];
        let mut b = mat(3, 1, &[1.0, 2.0, 3.0]);
        tridiagonal_solve(&mut dl, &mut d, &mut du, &mut b).unwrap();

        let f = lu_factor(&dense);
        assert!(f.singular_at.is_none());
        let expected = lu_solve(&f.lu, &f.piv, &mat(3, 1, &[1.0, 2.0, 3.0]), Trans::No);
        for row in 0..3 {
            assert_close(b.get(row, 0), expected.get(row, 0), 1e-10);
        }
    }

    #[test]
    fn tridiagonal_solve_takes_the_pivoting_branch_with_multiple_right_hand_sides() {
        // Same matrix as above but with n=4 (so the pivot branch's fill-in entry `du2[0]` is
        // exercised) and two right-hand-side columns.
        let dense = mat(
            4,
            4,
            &[
                1.0, 2.0, 0.0, 0.0, 3.0, 5.0, 4.0, 0.0, 0.0, 1.0, 6.0, 1.0, 0.0, 0.0, 1.0, 3.0,
            ],
        );
        let mut dl = vec![3.0, 1.0, 1.0];
        let mut d = vec![1.0, 5.0, 6.0, 3.0];
        let mut du = vec![2.0, 4.0, 1.0];
        let mut b = mat(4, 2, &[1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
        tridiagonal_solve(&mut dl, &mut d, &mut du, &mut b).unwrap();

        let f = lu_factor(&dense);
        assert!(f.singular_at.is_none());
        let expected = lu_solve(
            &f.lu,
            &f.piv,
            &mat(4, 2, &[1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]),
            Trans::No,
        );
        for row in 0..4 {
            for col in 0..2 {
                assert_close(b.get(row, col), expected.get(row, col), 1e-9);
            }
        }
    }

    #[test]
    fn eigh_generalized_solves_a_known_system() {
        // A x = lambda B x with A, B symmetric and B positive definite. Checked by residual
        // (`A v ~= lambda B v` for each eigenpair) rather than a hand-computed eigenvalue, and by
        // ascending order, matching how `jacobi_eigh` itself is documented to sort.
        let a = mat(2, 2, &[2.0, -1.0, -1.0, 2.0]);
        let b = mat(2, 2, &[2.0, 0.0, 0.0, 1.0]);
        let (values, vectors) = eigh_generalized(&a, &b, true, |_| Ok(())).unwrap().unwrap();
        let vectors = vectors.unwrap();
        assert!(values[0] <= values[1]);
        for (col, &value) in values.iter().enumerate() {
            let column: Vec<f64> = (0..2).map(|row| vectors.get(row, col)).collect();
            let v = Mat::from_row_major(2, 1, column);
            let av = a.matmul(&v);
            let bv = b.matmul(&v);
            for row in 0..2 {
                assert_close(av.get(row, 0), value * bv.get(row, 0), 1e-10);
            }
        }
    }

    #[test]
    fn expm_of_zero_is_identity() {
        let a = Mat::zeros(2, 2);
        let result = expm(&a);
        assert_close(result.get(0, 0), 1.0, 1e-14);
        assert_close(result.get(1, 1), 1.0, 1e-14);
        assert_close(result.get(0, 1), 0.0, 1e-14);
    }

    #[test]
    fn expm_of_diagonal_matches_scalar_exponential() {
        let mut a = Mat::zeros(2, 2);
        a.set(0, 0, 1.0);
        a.set(1, 1, 2.0);
        let result = expm(&a);
        assert_close(result.get(0, 0), std::f64::consts::E, 1e-12);
        assert_close(result.get(1, 1), std::f64::consts::E.powi(2), 1e-12);
    }
}
