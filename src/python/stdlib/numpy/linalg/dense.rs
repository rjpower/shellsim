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
//! - The general (non-symmetric) eigenproblem reduces `A` to upper Hessenberg form (§7.4.2), then
//!   deflates it to real Schur form with the explicit double-shift QR step of §7.5; eigenvectors
//!   come from a few steps of shifted inverse iteration on `A` directly, in complex arithmetic.
//!   See [`eig_general`]'s own doc for the full citation and the (deliberate) choice of the
//!   explicit over the implicit ("bulge-chasing") form of the double shift.
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
//! entry. `jacobi_eigh` and `jacobi_svd` apply this rule before returning; `eig_general`'s
//! eigenvectors, which are generally complex even for a real matrix, generalize it to a phase
//! convention (see [`eig_general`]'s doc).
//!
//! ## Resource accounting
//!
//! Deterministic factorizations (LU, Cholesky, QR, triangular solves) cost a fixed multiple of
//! `n^3` (or `m*n^2` for rectangular input); callers charge that before calling in. The Jacobi
//! solvers and the general eigenproblem's QR iteration do not know their iteration count in
//! advance, so `jacobi_eigh`, `jacobi_svd`, and `eig_general` accept a `charge`/`charge_sweep`
//! closure and call it with the cost of one sweep (or, for `eig_general`, one QR step or one
//! eigenvector's inverse iteration) before running it, so a caller low on CPU budget is stopped
//! before, not after, doing that work.

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
// General (non-symmetric) eigenproblem: Hessenberg reduction, explicit double-shift QR, and
// inverse iteration.
// ---------------------------------------------------------------------------------------------

/// A complex number, used only internally: a real matrix's eigenvalues and eigenvectors are
/// generally complex, even though every arithmetic operation on the *input* stays real until the
/// Schur form's diagonal blocks are read off (see [`eig_general`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::python) struct Cplx {
    pub re: f64,
    pub im: f64,
}

impl Cplx {
    pub(in crate::python) fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn real(re: f64) -> Self {
        Self::new(re, 0.0)
    }

    pub(in crate::python) fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    fn scale(self, factor: f64) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }
}

impl std::ops::Add for Cplx {
    type Output = Cplx;
    fn add(self, other: Cplx) -> Cplx {
        Cplx::new(self.re + other.re, self.im + other.im)
    }
}

impl std::ops::Sub for Cplx {
    type Output = Cplx;
    fn sub(self, other: Cplx) -> Cplx {
        Cplx::new(self.re - other.re, self.im - other.im)
    }
}

impl std::ops::Mul for Cplx {
    type Output = Cplx;
    fn mul(self, other: Cplx) -> Cplx {
        Cplx::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

impl std::ops::Div for Cplx {
    type Output = Cplx;
    fn div(self, other: Cplx) -> Cplx {
        // Scale both parts by the smaller of the divisor's magnitudes first (Smith's algorithm)
        // to avoid overflow when `other` has a very large component; this module's matrices are
        // small, but the shift used in inverse iteration can still be numerically large.
        if other.re.abs() >= other.im.abs() {
            let r = other.im / other.re;
            let d = other.re + other.im * r;
            Cplx::new((self.re + self.im * r) / d, (self.im - self.re * r) / d)
        } else {
            let r = other.re / other.im;
            let d = other.re * r + other.im;
            Cplx::new((self.re * r + self.im) / d, (self.im * r - self.re) / d)
        }
    }
}

/// Eigenvalues and, if requested, eigenvectors of a general square real matrix.
pub(in crate::python) struct EigResult {
    pub values: Vec<Cplx>,
    /// `vectors[k]` is the (unit-norm) eigenvector for `values[k]`, stored as a length-`n` column.
    pub vectors: Option<Vec<Vec<Cplx>>>,
}

/// The number of Francis-style QR sweeps allowed per remaining eigenvalue before giving up (and
/// reporting non-convergence), a generous multiple of the O(n) sweeps typically needed.
const SCHUR_MAX_SWEEPS_PER_VALUE: usize = 40;

/// Eigenvalues and, if requested, eigenvectors of the general (non-symmetric) eigenproblem
/// `A x = lambda x`, or `None` if the QR iteration fails to converge within its sweep budget
/// (which the caller reports as `LinAlgError`, as NumPy's own non-convergence does).
///
/// Follows Golub & Van Loan (4th ed.) §7.4-7.5: reduce `a` to upper Hessenberg form by
/// Householder similarity transforms (§7.4.2), then repeatedly deflate it with the explicit
/// double-shift QR step of §7.5 (forming `M = H^2 - s*H + t*I` from the trailing active block's
/// trace `s` and determinant `t`, factoring `M = QR`, and updating `H := Q^T H Q`) until every
/// diagonal block is `1x1` (a real eigenvalue) or `2x2` (a complex-conjugate pair, read off by
/// the quadratic formula). This module uses the explicit form rather than LAPACK's implicit
/// bulge-chasing `dlahqr`: the two are mathematically equivalent, but the explicit form reuses
/// this module's existing `householder_qr` and `matmul` instead of a dedicated bulge-chase, at
/// the cost of doing asymptotically more arithmetic — an acceptable trade under this module's
/// resource-accounting policy (CPU is charged for, not minimized).
///
/// Eigenvectors, when requested, come from shifted inverse iteration on `a` directly (a few
/// steps of solving `(A - lambda*I) v_{k+1} = v_k` in complex arithmetic and renormalizing),
/// rather than back-substitution on the Schur form: simpler to implement correctly, and just as
/// accurate once `lambda` is known to near machine precision from the QR iteration. Every
/// eigenvector is normalized to unit 2-norm and rotated so its largest-magnitude entry is a
/// positive real number, generalizing this module's real sign convention to a phase convention.
pub(in crate::python) fn eig_general(
    a: &Mat,
    compute_vectors: bool,
    mut charge: impl FnMut(u64) -> PyResult<()>,
) -> PyResult<Option<EigResult>> {
    debug_assert_eq!(a.rows, a.cols);
    let n = a.rows;
    if n == 0 {
        return Ok(Some(EigResult {
            values: Vec::new(),
            vectors: compute_vectors.then(Vec::new),
        }));
    }
    let scale = frobenius_norm(a).max(1.0);
    // Hessenberg reduction is a fixed multiple of QR's own O(n^3) cost; the two-sided update
    // (applying each reflector from both sides) does roughly twice the work of one-sided QR.
    charge(factor_cost(n as u64, n as u64) * 2)?;
    let h = hessenberg(a);
    let Some(values) = schur_eigenvalues(h, scale, &mut charge)? else {
        return Ok(None);
    };
    if !compute_vectors {
        return Ok(Some(EigResult {
            values,
            vectors: None,
        }));
    }
    let mut vectors = Vec::with_capacity(n);
    for &lambda in &values {
        // A complex LU factorization plus a handful of solves, each cubic; charge before each
        // eigenvector's inverse iteration so a budget-limited caller stops between vectors.
        charge(factor_cost(n as u64, n as u64) * 6)?;
        vectors.push(inverse_iterate(a, lambda, scale));
    }
    Ok(Some(EigResult {
        values,
        vectors: Some(vectors),
    }))
}

/// Reduce `a` to upper Hessenberg form by Householder similarity transforms (Golub & Van Loan
/// §7.4.2): for each column, a reflector zeroing the entries below the subdiagonal is applied
/// from the left (to zero them) and from the right (to complete the similarity transform), so
/// the result is similar to `a` (shares its eigenvalues) without needing to track the
/// accumulated orthogonal transform itself (eigenvectors here come from inverse iteration on `a`
/// directly, not from back-substitution on the Hessenberg or Schur form).
fn hessenberg(a: &Mat) -> Mat {
    let n = a.rows;
    let mut h = a.clone();
    for k in 0..n.saturating_sub(2) {
        let mut norm = 0.0f64;
        for i in (k + 1)..n {
            norm = norm.hypot(h.get(i, k));
        }
        if norm == 0.0 {
            continue;
        }
        let pivot = h.get(k + 1, k);
        let alpha = if pivot >= 0.0 { -norm } else { norm };
        let mut v = vec![0.0; n - k - 1];
        v[0] = pivot - alpha;
        for i in (k + 2)..n {
            v[i - k - 1] = h.get(i, k);
        }
        let v_norm = v.iter().map(|value| value * value).sum::<f64>().sqrt();
        if v_norm == 0.0 {
            continue;
        }
        for value in &mut v {
            *value /= v_norm;
        }
        // Apply H = I - 2 v v^T from the left to rows k+1..n, all columns from k onward.
        for j in k..n {
            let mut dot = 0.0;
            for (offset, entry) in v.iter().enumerate() {
                dot += entry * h.get(k + 1 + offset, j);
            }
            let update = 2.0 * dot;
            for (offset, entry) in v.iter().enumerate() {
                h.add(k + 1 + offset, j, -update * entry);
            }
        }
        // Apply H from the right to all rows, columns k+1..n, completing the similarity.
        for i in 0..n {
            let mut dot = 0.0;
            for (offset, entry) in v.iter().enumerate() {
                dot += entry * h.get(i, k + 1 + offset);
            }
            let update = 2.0 * dot;
            for (offset, entry) in v.iter().enumerate() {
                h.add(i, k + 1 + offset, -update * entry);
            }
        }
    }
    // Entries below the subdiagonal are zero in exact arithmetic; clear the rounding residue so
    // the deflation checks below compare against exact zeros where the structure guarantees them.
    for i in 2..n {
        for j in 0..(i - 1) {
            h.set(i, j, 0.0);
        }
    }
    h
}

/// Whether the subdiagonal entry `h[i+1, i]` is negligible relative to its neighboring diagonal
/// entries (or, if both are zero, relative to `scale`), the standard deflation criterion for the
/// QR algorithm (Golub & Van Loan §7.5.4).
fn negligible_subdiagonal(h: &Mat, i: usize, scale: f64) -> bool {
    let local = h.get(i, i).abs() + h.get(i + 1, i + 1).abs();
    let tolerance = f64::EPSILON * local.max(scale);
    h.get(i + 1, i).abs() <= tolerance
}

/// The (possibly complex-conjugate) eigenvalues of the `2x2` block at `h[k..k+2, k..k+2]`, by
/// the quadratic formula applied to its trace and determinant.
fn eigenvalues_2x2(h: &Mat, k: usize) -> (Cplx, Cplx) {
    let a = h.get(k, k);
    let b = h.get(k, k + 1);
    let c = h.get(k + 1, k);
    let d = h.get(k + 1, k + 1);
    let trace = a + d;
    let det = a * d - b * c;
    let discriminant = trace * trace - 4.0 * det;
    if discriminant >= 0.0 {
        let root = discriminant.sqrt();
        (
            Cplx::real((trace + root) / 2.0),
            Cplx::real((trace - root) / 2.0),
        )
    } else {
        let root = (-discriminant).sqrt() / 2.0;
        let re = trace / 2.0;
        (Cplx::new(re, root), Cplx::new(re, -root))
    }
}

/// Deflate upper Hessenberg `h` down to its real Schur form by explicit double-shift QR steps
/// (see [`eig_general`]'s doc), returning the eigenvalues read off the resulting `1x1`/`2x2`
/// diagonal blocks, or `None` if a block fails to deflate within its sweep budget.
fn schur_eigenvalues(
    mut h: Mat,
    scale: f64,
    charge: &mut impl FnMut(u64) -> PyResult<()>,
) -> PyResult<Option<Vec<Cplx>>> {
    let n = h.rows;
    let mut values = vec![Cplx::real(0.0); n];
    let mut active_end = n;
    let max_sweeps = SCHUR_MAX_SWEEPS_PER_VALUE.saturating_mul(n).max(100);
    let mut sweeps = 0usize;
    let mut sweeps_without_progress = 0usize;
    while active_end > 2 {
        if negligible_subdiagonal(&h, active_end - 2, scale) {
            values[active_end - 1] = Cplx::real(h.get(active_end - 1, active_end - 1));
            active_end -= 1;
            sweeps_without_progress = 0;
            continue;
        }
        if active_end >= 3 && negligible_subdiagonal(&h, active_end - 3, scale) {
            let (first, second) = eigenvalues_2x2(&h, active_end - 2);
            values[active_end - 2] = first;
            values[active_end - 1] = second;
            active_end -= 2;
            sweeps_without_progress = 0;
            continue;
        }
        sweeps += 1;
        sweeps_without_progress += 1;
        if sweeps > max_sweeps {
            return Ok(None);
        }
        charge(factor_cost(active_end as u64, active_end as u64) * 8)?;
        let (mut s, mut t) = shift_from_trailing_block(&h, active_end);
        // An ad hoc "exceptional shift" (Golub & Van Loan §7.5.4, following Wilkinson): if many
        // sweeps have passed without a deflation, the ordinary shift has stagnated (a known
        // failure mode of the plain algorithm), so perturb it using the subdiagonal's own
        // magnitude to break the cycle.
        if sweeps_without_progress > 0 && sweeps_without_progress.is_multiple_of(11) {
            let kick = h.get(active_end - 1, active_end - 2).abs()
                + h.get(active_end - 2, active_end.saturating_sub(3)).abs();
            let d = h.get(active_end - 1, active_end - 1);
            s = 2.0 * d;
            t = d * d + kick * kick;
        }
        double_shift_step(&mut h, active_end, s, t);
    }
    if active_end == 2 {
        let (first, second) = eigenvalues_2x2(&h, 0);
        values[0] = first;
        values[1] = second;
    } else if active_end == 1 {
        values[0] = Cplx::real(h.get(0, 0));
    }
    Ok(Some(values))
}

/// The trace and determinant of the trailing `2x2` block of the active `m x m` submatrix,
/// `(s, t)`, used to form the double-shift polynomial `M = H^2 - s*H + t*I`. Real even when the
/// block's own eigenvalues are complex, which is exactly what lets the double-shift QR step stay
/// in real arithmetic while still converging toward complex-conjugate pairs.
fn shift_from_trailing_block(h: &Mat, m: usize) -> (f64, f64) {
    let a = h.get(m - 2, m - 2);
    let b = h.get(m - 2, m - 1);
    let c = h.get(m - 1, m - 2);
    let d = h.get(m - 1, m - 1);
    (a + d, a * d - b * c)
}

/// One explicit double-shift QR step (Golub & Van Loan §7.5), applied to `h`'s leading `m x m`
/// active block and propagated to the trailing columns `m..n` that carry the rest of the
/// (partially deflated) Schur form: form `M = H_active^2 - s*H_active + t*I`, factor `M = QR`,
/// and update `H_active := Q^T H_active Q` (a similarity transform, so eigenvalues are
/// preserved) and the trailing block `H[0:m, m:n] := Q^T H[0:m, m:n]` (so later deflation steps
/// still see a consistent, if not literally Hessenberg, matrix above the active block).
fn double_shift_step(h: &mut Mat, m: usize, s: f64, t: f64) {
    let n = h.rows;
    let active = extract_block(h, 0, 0, m, m);
    let squared = active.matmul(&active);
    let mut shifted = Mat::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            let mut value = squared.get(i, j) - s * active.get(i, j);
            if i == j {
                value += t;
            }
            shifted.set(i, j, value);
        }
    }
    let qr = householder_qr(&shifted);
    let q = qr_explicit_q(&qr, true);
    let updated = q.transpose().matmul(&active).matmul(&q);
    write_block(h, &updated, 0, 0);
    if m < n {
        let trailing = extract_block(h, 0, m, m, n - m);
        let updated_trailing = q.transpose().matmul(&trailing);
        write_block(h, &updated_trailing, 0, m);
    }
}

/// The `rows x cols` block of `m` starting at `(row0, col0)`.
fn extract_block(m: &Mat, row0: usize, col0: usize, rows: usize, cols: usize) -> Mat {
    let mut block = Mat::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            block.set(i, j, m.get(row0 + i, col0 + j));
        }
    }
    block
}

/// Write `block` into `m` starting at `(row0, col0)`.
fn write_block(m: &mut Mat, block: &Mat, row0: usize, col0: usize) {
    for i in 0..block.rows {
        for j in 0..block.cols {
            m.set(row0 + i, col0 + j, block.get(i, j));
        }
    }
}

/// A small complex LU factorization with partial pivoting, used only by [`inverse_iterate`]:
/// `a`'s rows are permuted (recorded in `piv`, applied eagerly rather than deferred) and reduced
/// to upper-triangular `u` with unit-diagonal multipliers packed below it, exactly as
/// [`lu_factor`] does for real matrices. A pivot that rounds to exactly zero (only possible here
/// if the shift in [`inverse_iterate`] lands exactly on a repeated eigenvalue's already-reduced
/// column) is nudged to a tiny nonzero value rather than dividing by zero, since the caller only
/// wants *a* solution vector dominated by the eigenspace, not a certified factorization.
struct ComplexLu {
    lu: Vec<Vec<Cplx>>,
    piv: Vec<usize>,
}

fn complex_lu_factor(mut a: Vec<Vec<Cplx>>) -> ComplexLu {
    let n = a.len();
    let mut piv = Vec::with_capacity(n);
    for k in 0..n {
        let mut best = k;
        let mut best_value = a[k][k].abs();
        for (i, row) in a.iter().enumerate().skip(k + 1) {
            let value = row[k].abs();
            if value > best_value {
                best_value = value;
                best = i;
            }
        }
        piv.push(best);
        a.swap(k, best);
        if a[k][k].abs() < 1e-300 {
            a[k][k] = Cplx::real(1e-300);
        }
        let pivot = a[k][k];
        // Elimination reads row `k` while writing row `i`, two different rows of the same `Vec`,
        // so this clones the (short) pivot row rather than fighting the borrow checker over it.
        let pivot_row = a[k].clone();
        for row in a.iter_mut().skip(k + 1) {
            let factor = row[k] / pivot;
            row[k] = factor;
            for (j, value) in row.iter_mut().enumerate().skip(k + 1) {
                *value = *value - factor * pivot_row[j];
            }
        }
    }
    ComplexLu { lu: a, piv }
}

fn complex_lu_solve(factorization: &ComplexLu, b: &[Cplx]) -> Vec<Cplx> {
    let n = b.len();
    let mut x = b.to_vec();
    for (k, &p) in factorization.piv.iter().enumerate() {
        x.swap(k, p);
    }
    for i in 0..n {
        let mut sum = x[i];
        for (k, &value) in x.iter().enumerate().take(i) {
            sum = sum - factorization.lu[i][k] * value;
        }
        x[i] = sum;
    }
    for i in (0..n).rev() {
        let mut sum = x[i];
        for (k, &value) in x.iter().enumerate().skip(i + 1) {
            sum = sum - factorization.lu[i][k] * value;
        }
        x[i] = sum / factorization.lu[i][i];
    }
    x
}

/// The unit-norm eigenvector of real `a` for the (possibly complex) eigenvalue `lambda`, by a
/// few steps of shifted inverse iteration in complex arithmetic: starting from an arbitrary
/// vector, repeatedly solve `(A - lambda'*I) v_{k+1} = v_k` and renormalize, where `lambda'` is
/// `lambda` nudged by a scale-relative epsilon so the shifted system is never exactly singular
/// (inverse iteration is famously insensitive to how accurately that system is solved, so this
/// nudge costs essentially nothing; see Golub & Van Loan §7.6.1). The result is rotated so its
/// largest-magnitude entry is a positive real number (this module's real sign convention,
/// generalized to a phase for a genuinely complex eigenvector).
fn inverse_iterate(a: &Mat, lambda: Cplx, scale: f64) -> Vec<Cplx> {
    let n = a.rows;
    let epsilon = 1e-10 * scale;
    let shifted = Cplx::new(lambda.re + epsilon, lambda.im + 0.5 * epsilon);
    let mut matrix = vec![vec![Cplx::real(0.0); n]; n];
    for (i, row) in matrix.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            *entry = Cplx::real(a.get(i, j));
        }
        row[i] = row[i] - shifted;
    }
    let factorization = complex_lu_factor(matrix);
    let mut v = vec![Cplx::real(1.0); n];
    for _ in 0..4 {
        v = complex_lu_solve(&factorization, &v);
        normalize_complex_vector(&mut v);
    }
    fix_complex_phase(&mut v);
    v
}

fn normalize_complex_vector(v: &mut [Cplx]) {
    let norm = v
        .iter()
        .map(|value| value.abs() * value.abs())
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for value in v.iter_mut() {
            *value = value.scale(1.0 / norm);
        }
    }
}

/// Rotate `v` (in place) so its largest-magnitude entry becomes a positive real number, the
/// phase convention documented at the top of [`inverse_iterate`].
fn fix_complex_phase(v: &mut [Cplx]) {
    let mut best = 0usize;
    let mut best_magnitude = 0.0f64;
    for (i, value) in v.iter().enumerate() {
        let magnitude = value.abs();
        if magnitude > best_magnitude {
            best_magnitude = magnitude;
            best = i;
        }
    }
    if best_magnitude == 0.0 {
        return;
    }
    let unit = v[best].scale(1.0 / best_magnitude);
    for value in v.iter_mut() {
        *value = *value / unit;
    }
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

    fn assert_close_cplx(a: Cplx, b: Cplx, tol: f64) {
        assert_close(a.re, b.re, tol);
        assert_close(a.im, b.im, tol);
    }

    /// Every eigenpair `(lambda, v)` `eig_general` returns must satisfy `A v = lambda v`, in
    /// complex arithmetic, regardless of the algorithm's internal deflation order.
    fn assert_eigenpairs_satisfy_av_eq_lambda_v(a: &Mat, result: &EigResult) {
        let vectors = result.vectors.as_ref().expect("vectors requested");
        for (&lambda, v) in result.values.iter().zip(vectors) {
            for row in 0..a.rows {
                let mut sum = Cplx::real(0.0);
                for (col, &value) in v.iter().enumerate() {
                    sum = sum + Cplx::real(a.get(row, col)) * value;
                }
                assert_close_cplx(sum, lambda * v[row], 1e-8);
            }
        }
    }

    #[test]
    fn eig_general_finds_real_eigenvalues_of_a_diagonal_matrix() {
        let a = mat(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let result = eig_general(&a, true, |_| Ok(())).unwrap().unwrap();
        let mut values: Vec<f64> = result.values.iter().map(|v| v.re).collect();
        values.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_close(values[0], 2.0, 1e-10);
        assert_close(values[1], 3.0, 1e-10);
        assert!(result.values.iter().all(|v| v.im.abs() < 1e-10));
        assert_eigenpairs_satisfy_av_eq_lambda_v(&a, &result);
    }

    #[test]
    fn eig_general_finds_complex_conjugate_eigenvalues_of_a_rotation() {
        let theta: f64 = 0.7;
        let a = mat(2, 2, &[theta.cos(), -theta.sin(), theta.sin(), theta.cos()]);
        let result = eig_general(&a, true, |_| Ok(())).unwrap().unwrap();
        let mut values = result.values.clone();
        values.sort_by(|x, y| x.im.partial_cmp(&y.im).unwrap());
        assert_close_cplx(values[0], Cplx::new(theta.cos(), -theta.sin()), 1e-10);
        assert_close_cplx(values[1], Cplx::new(theta.cos(), theta.sin()), 1e-10);
        assert_eigenpairs_satisfy_av_eq_lambda_v(&a, &result);
    }

    #[test]
    fn eig_general_handles_a_nontrivial_three_by_three_matrix() {
        let a = mat(3, 3, &[2.0, -1.0, 0.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0]);
        let result = eig_general(&a, true, |_| Ok(())).unwrap().unwrap();
        assert_eq!(result.values.len(), 3);
        assert_eigenpairs_satisfy_av_eq_lambda_v(&a, &result);
        for v in result.vectors.as_ref().unwrap() {
            let norm = v.iter().map(|c| c.abs() * c.abs()).sum::<f64>().sqrt();
            assert_close(norm, 1.0, 1e-10);
        }
    }

    #[test]
    fn eig_general_repeated_eigenvalue_of_a_defective_matrix() {
        // A Jordan block: eigenvalue 2 with algebraic multiplicity 2 but only one independent
        // eigenvector, so both computed eigenvectors must point the same direction (up to sign).
        let a = mat(2, 2, &[2.0, 1.0, 0.0, 2.0]);
        let result = eig_general(&a, true, |_| Ok(())).unwrap().unwrap();
        for value in &result.values {
            assert_close_cplx(*value, Cplx::real(2.0), 1e-9);
        }
        let vectors = result.vectors.unwrap();
        let dot = (vectors[0][0] * vectors[1][0] + vectors[0][1] * vectors[1][1]).abs();
        assert_close(dot, 1.0, 1e-6);
    }

    #[test]
    fn eig_general_charges_before_hessenberg_reduction_and_each_sweep() {
        let a = mat(2, 2, &[0.0, -1.0, 1.0, 0.0]);
        let mut charges = Vec::new();
        let result = eig_general(&a, false, |cost| {
            charges.push(cost);
            Ok(())
        })
        .unwrap()
        .unwrap();
        assert!(!charges.is_empty());
        assert_eq!(result.values.len(), 2);
    }
}
