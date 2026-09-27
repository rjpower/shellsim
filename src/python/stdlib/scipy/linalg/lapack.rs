//! The LAPACK and BLAS routines behind `scipy.linalg`, ported to Rust.
//!
//! SciPy calls LAPACK through OpenBLAS, which replaces some LAPACK routines with its own and runs
//! the rest from reference LAPACK on its BLAS kernels. The ports follow the routine SciPy
//! actually runs, so that results round as SciPy's do:
//!
//! - `getf2`, `trti2`, `potf2` and `lauu2` follow OpenBLAS's own implementations, and `getrs` and
//!   `trtrs` follow OpenBLAS in solving a single right-hand side with `trsv`. They call the
//!   OpenBLAS kernels in [`super::openblas`] and agree with SciPy's OpenBLAS to the last bit up
//!   to the orders where OpenBLAS switches to blocked drivers: 33 for `getrf`, 32 for `potrf`
//!   and 64 for `lauum`. OpenBLAS's threaded `potri` drivers, which it selects from order 17 on
//!   a multi-core machine, are not modeled.
//! - `getri` and the condition estimators follow reference LAPACK 3.12 on OpenBLAS's `gemv` and
//!   `trsv` kernels. The remaining routines follow reference LAPACK and the reference BLAS order.
//! - Several right-hand sides go through `trsm`, split by column as OpenBLAS's threaded `getrs`,
//!   `trtrs` and `dtrsm` split them on the 16 threads of the reference machine. `trsm` agrees
//!   with OpenBLAS up to order 256.
//!
//! Blocked drivers such as `dgetrf`, `dpotrf`, `dsytrf` and `dtrtri` are replaced by their
//! unblocked kernels at every order.
//!
//! Matrices are column-major slices with a leading dimension, as in LAPACK. Indices are 0-based;
//! an `info` value keeps LAPACK's 1-based meaning, so `info = k` names the `k`-th pivot. Every
//! routine is generic over [`Real`], because SciPy computes `float32` input in single precision.
//!
//! Callers charge the work: the direct methods do cubic work in the matrix order, and the
//! condition estimators run at most a dozen triangular solves.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::super::super::numpy::norm2;
use super::openblas::{self, trsv};

/// A floating-point type LAPACK computes in, with `dlamch`'s machine constants.
pub(super) trait Real:
    Copy
    + PartialOrd
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    const ZERO: Self;
    const ONE: Self;
    const HALF: Self;
    /// `dlamch('S')`: the smallest normal number, whose reciprocal does not overflow.
    const SAFE_MIN: Self;
    /// `dlamch('P')`: the machine epsilon, `eps * base`, which is also C's
    /// `numeric_limits<T>::epsilon()` that SciPy compares reciprocal condition numbers against.
    const PRECISION: Self;
    /// `dlamch('O')`: the largest finite number.
    const OVERFLOW: Self;

    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn log2(self) -> Self;
    fn ceil(self) -> Self;
    fn powf(self, exponent: Self) -> Self;
    fn copysign(self, sign: Self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn is_nan(self) -> bool;
    fn from_usize(value: usize) -> Self;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    /// `dnrm2` or `snrm2` as OpenBLAS computes them: without harmful overflow or underflow,
    /// and rounded once from extended precision.
    fn nrm2(values: &[Self]) -> Self;
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;
    const SAFE_MIN: Self = f64::MIN_POSITIVE;
    const PRECISION: Self = f64::EPSILON;
    const OVERFLOW: Self = f64::MAX;

    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn exp(self) -> Self {
        f64::exp(self)
    }
    fn exp2(self) -> Self {
        f64::exp2(self)
    }
    fn log2(self) -> Self {
        f64::log2(self)
    }
    fn ceil(self) -> Self {
        f64::ceil(self)
    }
    fn powf(self, exponent: Self) -> Self {
        f64::powf(self, exponent)
    }
    fn copysign(self, sign: Self) -> Self {
        f64::copysign(self, sign)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    fn from_usize(value: usize) -> Self {
        value as f64
    }
    fn from_f64(value: f64) -> Self {
        value
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn nrm2(values: &[Self]) -> Self {
        norm2(values)
    }
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;
    const SAFE_MIN: Self = f32::MIN_POSITIVE;
    const PRECISION: Self = f32::EPSILON;
    const OVERFLOW: Self = f32::MAX;

    fn abs(self) -> Self {
        f32::abs(self)
    }
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    fn exp(self) -> Self {
        f32::exp(self)
    }
    fn exp2(self) -> Self {
        f32::exp2(self)
    }
    fn log2(self) -> Self {
        f32::log2(self)
    }
    fn ceil(self) -> Self {
        f32::ceil(self)
    }
    fn powf(self, exponent: Self) -> Self {
        f32::powf(self, exponent)
    }
    fn copysign(self, sign: Self) -> Self {
        f32::copysign(self, sign)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32::mul_add(self, a, b)
    }
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    fn from_usize(value: usize) -> Self {
        value as f32
    }
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    /// OpenBLAS accumulates single-precision squares in double precision.
    fn nrm2(values: &[Self]) -> Self {
        let sum: f64 = values.iter().map(|value| f64::from(*value).powi(2)).sum();
        sum.sqrt() as f32
    }
}

/// Fortran's `MAX` for ordered operands.
fn max<T: Real>(a: T, b: T) -> T {
    if b > a {
        b
    } else {
        a
    }
}

/// Fortran's `MIN` for ordered operands.
fn min<T: Real>(a: T, b: T) -> T {
    if b < a {
        b
    } else {
        a
    }
}

// ---------------------------------------------------------------------------------------------
// BLAS
// ---------------------------------------------------------------------------------------------

/// `idamax`: the first index among `n >= 1` elements `x[start + k * inc]` of largest magnitude.
fn iamax<T: Real>(n: usize, x: &[T], start: usize, inc: usize) -> usize {
    let mut best = 0;
    let mut largest = x[start].abs();
    for k in 1..n {
        let magnitude = x[start + k * inc].abs();
        if magnitude > largest {
            best = k;
            largest = magnitude;
        }
    }
    best
}

/// `dasum`: the sum of magnitudes, accumulated in order.
fn asum<T: Real>(n: usize, x: &[T], start: usize, inc: usize) -> T {
    let mut sum = T::ZERO;
    for k in 0..n {
        sum += x[start + k * inc].abs();
    }
    sum
}

/// `dscal`: scale `n` elements by `factor`; scaling by one is skipped.
fn scal<T: Real>(n: usize, factor: T, x: &mut [T], start: usize, inc: usize) {
    if factor == T::ONE {
        return;
    }
    for k in 0..n {
        let index = start + k * inc;
        x[index] = factor * x[index];
    }
}

/// `ddot` of two strided vectors of one buffer, accumulated in order.
fn dot<T: Real>(
    n: usize,
    x: &[T],
    x_start: usize,
    x_inc: usize,
    y_start: usize,
    y_inc: usize,
) -> T {
    let mut sum = T::ZERO;
    for k in 0..n {
        sum += x[x_start + k * x_inc] * x[y_start + k * y_inc];
    }
    sum
}

/// `ddot` of vectors in two buffers.
fn dot2<T: Real>(n: usize, x: &[T], x_start: usize, x_inc: usize, y: &[T], y_start: usize) -> T {
    let mut sum = T::ZERO;
    for k in 0..n {
        sum += x[x_start + k * x_inc] * y[y_start + k];
    }
    sum
}

/// `dswap` of two strided vectors of one buffer.
fn swap<T: Real>(
    n: usize,
    x: &mut [T],
    x_start: usize,
    x_inc: usize,
    y_start: usize,
    y_inc: usize,
) {
    for k in 0..n {
        x.swap(x_start + k * x_inc, y_start + k * y_inc);
    }
}

/// Swap rows `first` and `second` of the `columns`-column matrix `b`.
fn swap_rows<T: Real>(b: &mut [T], ldb: usize, columns: usize, first: usize, second: usize) {
    if first != second {
        swap(columns, b, first, ldb, second, ldb);
    }
}

/// `dgemm` with no transposes on `n × n` column-major operands: `c = a b`, or `c = a b + c`
/// when `accumulate`. This follows OpenBLAS's x86-64 kernels rather than the reference loop:
/// each element of `a b` is a chain of fused multiply-adds from zero, added to `c` at the end.
/// `expm` squares its result repeatedly, which amplifies the difference between the two orders
/// to about `1e-14` after a few squarings.
pub(super) fn gemm<T: Real>(n: usize, a: &[T], b: &[T], c: &mut [T], accumulate: bool) {
    for j in 0..n {
        for i in 0..n {
            let mut sum = T::ZERO;
            for l in 0..n {
                sum = a[i + l * n].mul_add(b[l + j * n], sum);
            }
            c[i + j * n] = if accumulate { c[i + j * n] + sum } else { sum };
        }
    }
}

/// `dgemv` with `alpha = 1` and `beta = 0` on an `n × n` column-major matrix: `y = A x`, or
/// `y = Aᵀ x` when `transpose`.
pub(super) fn gemv<T: Real>(transpose: bool, n: usize, a: &[T], x: &[T], y: &mut [T]) {
    if transpose {
        for (j, value) in y[..n].iter_mut().enumerate() {
            *value = dot2(n, a, j * n, 1, x, 0);
        }
    } else {
        y[..n].iter_mut().for_each(|value| *value = T::ZERO);
        for j in 0..n {
            let temp = x[j];
            for i in 0..n {
                y[i] += temp * a[i + j * n];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// General matrices: dgetf2, dgetrs, dgecon, dgetri
// ---------------------------------------------------------------------------------------------

/// `dgetf2` as OpenBLAS implements it (`getf2_k.c`), which its `dgetrf` runs whenever
/// `min(m, n) <= 33`: a left-looking Crout factorization of the `m × n` matrix `a`, in place as
/// `P A = L U`. Each column receives the earlier row exchanges, then the updates from the columns
/// to its left through a strided `dot` against each row of `L` and one `gemv_n`, and is then
/// pivoted and scaled by the reciprocal of its pivot. Larger matrices take OpenBLAS's recursive
/// blocked driver, whose last bits this does not reproduce.
///
/// Returns the 0-based row exchanged at each step and LAPACK's `info`, the 1-based index of the
/// first zero pivot. As in OpenBLAS, a pivot smaller than the smallest normal number neither
/// exchanges rows nor scales its column, although its row is still recorded.
pub(super) fn getf2<T: Real>(m: usize, n: usize, a: &mut [T], lda: usize) -> (Vec<usize>, usize) {
    let mut pivots = Vec::with_capacity(m.min(n));
    let mut info = 0;
    for j in 0..n {
        let (left, right) = a.split_at_mut(j * lda);
        let column = &mut right[..m];
        let factored = j.min(m);
        for (row, pivot) in pivots.iter().enumerate() {
            column.swap(row, *pivot);
        }
        for i in 1..factored {
            let update = openblas::dot_strided(i, &left[i..], lda, column, 1);
            column[i] -= update;
        }
        if j >= m {
            continue;
        }
        let (above, below) = column.split_at_mut(j);
        openblas::gemv_n(m - j, j, -T::ONE, &left[j..], lda, above, 1, below);
        let jp = j + iamax(m - j, below, 0, 1);
        pivots.push(jp);
        let pivot = column[jp];
        if pivot == T::ZERO {
            if info == 0 {
                info = j + 1;
            }
        } else if pivot.abs() >= T::SAFE_MIN {
            let reciprocal = T::ONE / pivot;
            if jp != j {
                swap(j, left, j, lda, jp, lda);
                column.swap(j, jp);
            }
            for value in &mut column[j + 1..] {
                *value = reciprocal * *value;
            }
        }
    }
    (pivots, info)
}

/// `dlaswp` over `columns` columns of `b`: apply the row exchanges in `pivots` in order, or in
/// reverse when `forward` is false.
pub(super) fn laswp<T: Real>(
    b: &mut [T],
    ldb: usize,
    columns: usize,
    pivots: &[usize],
    forward: bool,
) {
    if forward {
        for (row, pivot) in pivots.iter().enumerate() {
            swap_rows(b, ldb, columns, row, *pivot);
        }
    } else {
        for (row, pivot) in pivots.iter().enumerate().rev() {
            swap_rows(b, ldb, columns, row, *pivot);
        }
    }
}

/// `dgetrs` as OpenBLAS implements it: overwrite the `n × nrhs` matrix `b` with `A⁻¹ b`, or
/// `A⁻ᵀ b` when `transpose`, from `getf2`'s factors.
#[allow(clippy::too_many_arguments)]
pub(super) fn getrs<T: Real>(
    transpose: bool,
    n: usize,
    nrhs: usize,
    lu: &[T],
    lda: usize,
    pivots: &[usize],
    b: &mut [T],
    ldb: usize,
) {
    if n == 0 || nrhs == 0 {
        return;
    }
    // OpenBLAS's `getrs` solves a single right-hand side with `trsv`, and splits several among
    // its threads. The columns are independent, so each triangular solve can cover all of them.
    if transpose {
        if nrhs == 1 {
            openblas::trsv(true, true, false, n, lu, lda, &mut b[..n]);
            openblas::trsv(false, true, true, n, lu, lda, &mut b[..n]);
        } else {
            openblas::trsm_threaded(true, true, false, n, nrhs, lu, lda, b, ldb);
            openblas::trsm_threaded(false, true, true, n, nrhs, lu, lda, b, ldb);
        }
        laswp(b, ldb, nrhs, pivots, false);
    } else {
        laswp(b, ldb, nrhs, pivots, true);
        if nrhs == 1 {
            openblas::trsv(false, false, true, n, lu, lda, &mut b[..n]);
            openblas::trsv(true, false, false, n, lu, lda, &mut b[..n]);
        } else {
            openblas::trsm_threaded(false, false, true, n, nrhs, lu, lda, b, ldb);
            openblas::trsm_threaded(true, false, false, n, nrhs, lu, lda, b, ldb);
        }
    }
}

/// The outcome of a LAPACK condition estimate: `rcond` and `info`, where a negative `info`
/// rejects an argument, as SciPy observes it.
pub(super) struct Condition<T> {
    pub rcond: T,
    pub info: i64,
}

/// `dlacn2`: estimate the 1-norm of a linear operator by Higham's modification of Hager's
/// method. `apply(kase, x)` overwrites `x` with `A x` for `kase == 1` and `Aᵀ x` for
/// `kase == 2`, and returns false to abandon the estimate, which then returns `None`.
pub(super) fn lacn2<T: Real>(n: usize, mut apply: impl FnMut(u8, &mut [T]) -> bool) -> Option<T> {
    const ITMAX: usize = 5;
    let sign = |value: T| if value >= T::ZERO { T::ONE } else { -T::ONE };
    let mut x = vec![T::ONE / T::from_usize(n); n];
    let mut v = vec![T::ZERO; n];
    let mut signs = vec![T::ZERO; n];
    if !apply(1, &mut x) {
        return None;
    }
    if n == 1 {
        return Some(x[0].abs());
    }
    let mut estimate = asum(n, &x, 0, 1);
    for i in 0..n {
        x[i] = sign(x[i]);
        signs[i] = x[i];
    }
    if !apply(2, &mut x) {
        return None;
    }
    let mut j = iamax(n, &x, 0, 1);
    let mut iteration = 2;
    loop {
        x.iter_mut().for_each(|value| *value = T::ZERO);
        x[j] = T::ONE;
        if !apply(1, &mut x) {
            return None;
        }
        v.copy_from_slice(&x);
        let previous = estimate;
        estimate = asum(n, &v, 0, 1);
        let converged = (0..n).all(|i| sign(x[i]) == signs[i]);
        if converged || estimate <= previous {
            break;
        }
        for i in 0..n {
            x[i] = sign(x[i]);
            signs[i] = x[i];
        }
        if !apply(2, &mut x) {
            return None;
        }
        let last = j;
        j = iamax(n, &x, 0, 1);
        if x[last] != x[j].abs() && iteration < ITMAX {
            iteration += 1;
            continue;
        }
        break;
    }
    let mut alternating = T::ONE;
    let denominator = T::from_usize(n - 1);
    for (i, value) in x.iter_mut().enumerate() {
        *value = alternating * (T::ONE + T::from_usize(i) / denominator);
        alternating = -alternating;
    }
    if !apply(1, &mut x) {
        return None;
    }
    let temp = T::from_f64(2.0) * (asum(n, &x, 0, 1) / T::from_usize(3 * n));
    if temp > estimate {
        estimate = temp;
    }
    Some(estimate)
}

/// `drscl`: scale `x` by `1 / factor` without overflow or harmful underflow.
fn rscl<T: Real>(factor: T, x: &mut [T]) {
    let smlnum = T::SAFE_MIN;
    let bignum = T::ONE / smlnum;
    let mut denominator = factor;
    let mut numerator = T::ONE;
    loop {
        let denominator1 = denominator * smlnum;
        let numerator1 = numerator / bignum;
        let (multiplier, done) = if denominator1.abs() > numerator.abs() && numerator != T::ZERO {
            denominator = denominator1;
            (smlnum, false)
        } else if numerator1.abs() > denominator.abs() {
            numerator = numerator1;
            (bignum, false)
        } else {
            (numerator / denominator, true)
        };
        let n = x.len();
        scal(n, multiplier, x, 0, 1);
        if done {
            return;
        }
    }
}

/// The largest magnitude among `values`, propagating NaN as `dlange('M')` does.
fn max_magnitude<T: Real>(values: impl Iterator<Item = T>) -> T {
    let mut value = T::ZERO;
    for element in values {
        let magnitude = element.abs();
        if value < magnitude || magnitude.is_nan() {
            value = magnitude;
        }
    }
    value
}

/// `dlatrs`: solve a triangular system `op(A) x = s b` with a scale factor `s <= 1` chosen so
/// that `x` does not overflow. `cnorm` holds the off-diagonal column norms; it is computed here
/// when `normin` is false and reused otherwise. Returns `s`.
#[allow(clippy::too_many_arguments)]
fn latrs<T: Real>(
    upper: bool,
    transpose: bool,
    unit: bool,
    normin: bool,
    n: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
    cnorm: &mut [T],
) -> T {
    let nounit = !unit;
    let mut scale = T::ONE;
    if n == 0 {
        return scale;
    }
    let smlnum = T::SAFE_MIN / T::PRECISION;
    let bignum = T::ONE / smlnum;
    if !normin {
        if upper {
            for (j, norm) in cnorm[..n].iter_mut().enumerate() {
                *norm = asum(j, a, j * lda, 1);
            }
        } else {
            for (j, norm) in cnorm[..n - 1].iter_mut().enumerate() {
                *norm = asum(n - j - 1, a, j + 1 + j * lda, 1);
            }
            cnorm[n - 1] = T::ZERO;
        }
    }
    let imax = iamax(n, cnorm, 0, 1);
    let mut tmax = cnorm[imax];
    let tscal;
    if tmax <= bignum {
        tscal = T::ONE;
    } else if tmax <= T::OVERFLOW {
        tscal = T::ONE / (smlnum * tmax);
        scal(n, tscal, cnorm, 0, 1);
    } else {
        tmax = T::ZERO;
        if upper {
            for j in 1..n {
                tmax = max(max_magnitude((0..j).map(|i| a[i + j * lda])), tmax);
            }
        } else {
            for j in 0..n - 1 {
                tmax = max(max_magnitude((j + 1..n).map(|i| a[i + j * lda])), tmax);
            }
        }
        if tmax <= T::OVERFLOW {
            tscal = T::ONE / (smlnum * tmax);
            for j in 0..n {
                if cnorm[j] <= T::OVERFLOW {
                    cnorm[j] *= tscal;
                } else {
                    cnorm[j] = T::ZERO;
                    let rows = if upper { 0..j } else { j + 1..n };
                    for i in rows {
                        cnorm[j] += tscal * a[i + j * lda].abs();
                    }
                }
            }
        } else {
            trsv(upper, transpose, unit, n, a, lda, x);
            return scale;
        }
    }

    let j = iamax(n, x, 0, 1);
    let mut xmax = x[j].abs();
    let mut xbnd = xmax;
    // Columns are visited in the order the substitution eliminates them.
    let forward = upper == transpose;
    let order = |n: usize| -> Vec<usize> {
        if forward {
            (0..n).collect()
        } else {
            (0..n).rev().collect()
        }
    };
    let mut grow;
    if tscal != T::ONE {
        grow = T::ZERO;
    } else if !transpose {
        if nounit {
            grow = T::ONE / max(xbnd, smlnum);
            xbnd = grow;
            let mut stopped = false;
            for j in order(n) {
                if grow <= smlnum {
                    stopped = true;
                    break;
                }
                let tjj = a[j + j * lda].abs();
                xbnd = min(xbnd, min(T::ONE, tjj) * grow);
                if tjj + cnorm[j] >= smlnum {
                    grow *= tjj / (tjj + cnorm[j]);
                } else {
                    grow = T::ZERO;
                }
            }
            if !stopped {
                grow = xbnd;
            }
        } else {
            grow = min(T::ONE, T::ONE / max(xbnd, smlnum));
            for j in order(n) {
                if grow <= smlnum {
                    break;
                }
                grow *= T::ONE / (T::ONE + cnorm[j]);
            }
        }
    } else if nounit {
        grow = T::ONE / max(xbnd, smlnum);
        xbnd = grow;
        let mut stopped = false;
        for j in order(n) {
            if grow <= smlnum {
                stopped = true;
                break;
            }
            let xj = T::ONE + cnorm[j];
            grow = min(grow, xbnd / xj);
            let tjj = a[j + j * lda].abs();
            if xj > tjj {
                xbnd *= tjj / xj;
            }
        }
        if !stopped {
            grow = min(grow, xbnd);
        }
    } else {
        grow = min(T::ONE, T::ONE / max(xbnd, smlnum));
        for j in order(n) {
            if grow <= smlnum {
                break;
            }
            let xj = T::ONE + cnorm[j];
            grow /= xj;
        }
    }

    if grow * tscal > smlnum {
        trsv(upper, transpose, unit, n, a, lda, x);
    } else {
        if xmax > bignum {
            scale = bignum / xmax;
            scal(n, scale, x, 0, 1);
            xmax = bignum;
        }
        if !transpose {
            for j in order(n) {
                let mut xj = x[j].abs();
                let tjjs = if nounit {
                    a[j + j * lda] * tscal
                } else {
                    tscal
                };
                if nounit || tscal != T::ONE {
                    let tjj = tjjs.abs();
                    if tjj > smlnum {
                        if tjj < T::ONE && xj > tjj * bignum {
                            let rec = T::ONE / xj;
                            scal(n, rec, x, 0, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        x[j] /= tjjs;
                        xj = x[j].abs();
                    } else if tjj > T::ZERO {
                        if xj > tjj * bignum {
                            let mut rec = (tjj * bignum) / xj;
                            if cnorm[j] > T::ONE {
                                rec /= cnorm[j];
                            }
                            scal(n, rec, x, 0, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        x[j] /= tjjs;
                        xj = x[j].abs();
                    } else {
                        x.iter_mut().for_each(|value| *value = T::ZERO);
                        x[j] = T::ONE;
                        xj = T::ONE;
                        scale = T::ZERO;
                        xmax = T::ZERO;
                    }
                }
                if xj > T::ONE {
                    let mut rec = T::ONE / xj;
                    if cnorm[j] > (bignum - xmax) * rec {
                        rec *= T::HALF;
                        scal(n, rec, x, 0, 1);
                        scale *= rec;
                    }
                } else if xj * cnorm[j] > bignum - xmax {
                    scal(n, T::HALF, x, 0, 1);
                    scale *= T::HALF;
                }
                if upper {
                    if j > 0 {
                        let factor = -x[j] * tscal;
                        if factor != T::ZERO {
                            for i in 0..j {
                                x[i] += factor * a[i + j * lda];
                            }
                        }
                        let i = iamax(j, x, 0, 1);
                        xmax = x[i].abs();
                    }
                } else if j + 1 < n {
                    let factor = -x[j] * tscal;
                    if factor != T::ZERO {
                        for i in j + 1..n {
                            x[i] += factor * a[i + j * lda];
                        }
                    }
                    let i = j + 1 + iamax(n - j - 1, x, j + 1, 1);
                    xmax = x[i].abs();
                }
            }
        } else {
            for j in order(n) {
                let mut xj = x[j].abs();
                let mut uscal = tscal;
                let mut rec = T::ONE / max(xmax, T::ONE);
                let mut tjjs = tscal;
                if cnorm[j] > (bignum - xj) * rec {
                    rec *= T::HALF;
                    tjjs = if nounit {
                        a[j + j * lda] * tscal
                    } else {
                        tscal
                    };
                    let tjj = tjjs.abs();
                    if tjj > T::ONE {
                        rec = min(T::ONE, rec * tjj);
                        uscal /= tjjs;
                    }
                    if rec < T::ONE {
                        scal(n, rec, x, 0, 1);
                        scale *= rec;
                        xmax *= rec;
                    }
                }
                let mut sumj = T::ZERO;
                let rows = if upper { 0..j } else { j + 1..n };
                if uscal == T::ONE {
                    for i in rows {
                        sumj += a[i + j * lda] * x[i];
                    }
                } else {
                    for i in rows {
                        sumj += (a[i + j * lda] * uscal) * x[i];
                    }
                }
                if uscal == tscal {
                    x[j] -= sumj;
                    xj = x[j].abs();
                    let tjjs = if nounit {
                        a[j + j * lda] * tscal
                    } else {
                        tscal
                    };
                    if nounit || tscal != T::ONE {
                        let tjj = tjjs.abs();
                        if tjj > smlnum {
                            if tjj < T::ONE && xj > tjj * bignum {
                                let rec = T::ONE / xj;
                                scal(n, rec, x, 0, 1);
                                scale *= rec;
                                xmax *= rec;
                            }
                            x[j] /= tjjs;
                        } else if tjj > T::ZERO {
                            if xj > tjj * bignum {
                                let rec = (tjj * bignum) / xj;
                                scal(n, rec, x, 0, 1);
                                scale *= rec;
                                xmax *= rec;
                            }
                            x[j] /= tjjs;
                        } else {
                            x.iter_mut().for_each(|value| *value = T::ZERO);
                            x[j] = T::ONE;
                            scale = T::ZERO;
                            xmax = T::ZERO;
                        }
                    }
                } else {
                    x[j] = x[j] / tjjs - sumj;
                }
                xmax = max(xmax, x[j].abs());
            }
        }
        scale /= tscal;
    }
    if tscal != T::ONE {
        scal(n, T::ONE / tscal, cnorm, 0, 1);
    }
    scale
}

/// Rescale the estimator's vector after a scaled triangular solve, as the `*con` routines do
/// between `dlacn2` steps. Returns false when the scale factor underflows, which ends the
/// estimate with `rcond = 0`.
fn rescale<T: Real>(x: &mut [T], scale: T, smlnum: T) -> bool {
    if scale != T::ONE {
        let ix = iamax(x.len(), x, 0, 1);
        if scale < x[ix].abs() * smlnum || scale == T::ZERO {
            return false;
        }
        rscl(scale, x);
    }
    true
}

/// `dgecon`: estimate the reciprocal condition number of the matrix whose `getf2` factors are
/// `lu`, given its 1-norm `anorm`, or its infinity norm when `one_norm` is false.
pub(super) fn gecon<T: Real>(
    one_norm: bool,
    n: usize,
    lu: &[T],
    lda: usize,
    anorm: T,
) -> Condition<T> {
    let mut condition = Condition {
        rcond: T::ZERO,
        info: 0,
    };
    if n == 0 {
        condition.rcond = T::ONE;
        return condition;
    }
    if anorm == T::ZERO {
        return condition;
    }
    if anorm.is_nan() {
        condition.rcond = anorm;
        condition.info = -5;
        return condition;
    }
    if anorm > T::OVERFLOW {
        condition.info = -5;
        return condition;
    }
    let smlnum = T::SAFE_MIN;
    let mut normin = false;
    let mut cnorm_l = vec![T::ZERO; n];
    let mut cnorm_u = vec![T::ZERO; n];
    let kase1 = if one_norm { 1 } else { 2 };
    let estimate = lacn2(n, |kase, x| {
        let (sl, su) = if kase == kase1 {
            let sl = latrs(false, false, true, normin, n, lu, lda, x, &mut cnorm_l);
            let su = latrs(true, false, false, normin, n, lu, lda, x, &mut cnorm_u);
            (sl, su)
        } else {
            let su = latrs(true, true, false, normin, n, lu, lda, x, &mut cnorm_u);
            let sl = latrs(false, true, true, normin, n, lu, lda, x, &mut cnorm_l);
            (sl, su)
        };
        normin = true;
        rescale(x, sl * su, smlnum)
    });
    let Some(ainvnm) = estimate else {
        return condition;
    };
    if ainvnm == T::ZERO {
        condition.info = 1;
        return condition;
    }
    condition.rcond = (T::ONE / ainvnm) / anorm;
    if condition.rcond.is_nan() || condition.rcond > T::OVERFLOW {
        condition.info = 1;
    }
    condition
}

/// `dgetri` (unblocked, as reference LAPACK runs it up to order 64): overwrite `getf2`'s factors
/// with `A⁻¹`. Returns `info`, the 1-based index of a zero diagonal element of `U`.
pub(super) fn getri<T: Real>(n: usize, a: &mut [T], lda: usize, pivots: &[usize]) -> usize {
    if n == 0 {
        return 0;
    }
    let info = trti2_checked(true, false, n, a, lda);
    if info > 0 {
        return info;
    }
    let mut work = vec![T::ZERO; n];
    for j in (0..n).rev() {
        for i in j + 1..n {
            work[i] = a[i + j * lda];
            a[i + j * lda] = T::ZERO;
        }
        if j + 1 < n {
            // dgemv('N', n, n-j-1, -1, A(:, j+1:), work(j+1:), 1, A(:, j))
            let (left, right) = a.split_at_mut((j + 1) * lda);
            let target = &mut left[j * lda..j * lda + n];
            openblas::gemv_n(n, n - j - 1, -T::ONE, right, lda, &work[j + 1..], 1, target);
        }
    }
    for j in (0..n.saturating_sub(1)).rev() {
        let jp = pivots[j];
        if jp != j {
            swap(n, a, j * lda, 1, jp * lda, 1);
        }
    }
    0
}

// ---------------------------------------------------------------------------------------------
// Triangular matrices: dtrtrs, dtrcon, dtrtri
// ---------------------------------------------------------------------------------------------

/// `dtrtrs`: overwrite `b` with `op(A)⁻¹ b` for triangular `a`. Returns `info`, the 1-based
/// index of a zero diagonal element when `a` is not unit triangular.
#[allow(clippy::too_many_arguments)]
pub(super) fn trtrs<T: Real>(
    upper: bool,
    transpose: bool,
    unit: bool,
    n: usize,
    nrhs: usize,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) -> usize {
    if n == 0 {
        return 0;
    }
    if !unit {
        if let Some(zero) = (0..n).find(|&i| a[i + i * lda] == T::ZERO) {
            return zero + 1;
        }
    }
    // OpenBLAS's `trtrs` solves a single right-hand side with `trsv`, and splits several among
    // its threads.
    if nrhs == 1 {
        openblas::trsv(upper, transpose, unit, n, a, lda, &mut b[..n]);
    } else {
        openblas::trsm_threaded(upper, transpose, unit, n, nrhs, a, lda, b, ldb);
    }
    0
}

/// `dlantr` with the 1-norm: the largest column sum of magnitudes of a triangular matrix.
fn lantr_one<T: Real>(upper: bool, unit: bool, n: usize, a: &[T], lda: usize) -> T {
    let mut value = T::ZERO;
    for j in 0..n {
        let mut sum;
        if unit {
            sum = T::ONE;
            let rows = if upper { 0..j } else { j + 1..n };
            for i in rows {
                sum += a[i + j * lda].abs();
            }
        } else {
            sum = T::ZERO;
            let rows = if upper { 0..j + 1 } else { j..n };
            for i in rows {
                sum += a[i + j * lda].abs();
            }
        }
        if value < sum || sum.is_nan() {
            value = sum;
        }
    }
    value
}

/// `dtrcon` with the 1-norm: estimate the reciprocal condition number of a triangular matrix.
pub(super) fn trcon<T: Real>(upper: bool, unit: bool, n: usize, a: &[T], lda: usize) -> T {
    if n == 0 {
        return T::ONE;
    }
    let smlnum = T::SAFE_MIN * T::from_usize(n.max(1));
    let anorm = lantr_one(upper, unit, n, a, lda);
    if anorm <= T::ZERO {
        return T::ZERO;
    }
    let mut normin = false;
    let mut cnorm = vec![T::ZERO; n];
    let estimate = lacn2(n, |kase, x| {
        let scale = latrs(upper, kase != 1, unit, normin, n, a, lda, x, &mut cnorm);
        normin = true;
        rescale(x, scale, smlnum)
    });
    match estimate {
        Some(ainvnm) if ainvnm != T::ZERO => (T::ONE / anorm) / ainvnm,
        _ => T::ZERO,
    }
}

/// `dtrti2` as OpenBLAS implements it (`trti2_U.c` and `trti2_L.c`), which its `dtrtri` runs up
/// to order 256, preceded by `dtrtri`'s check for a zero diagonal element: overwrite a
/// triangular matrix with its inverse. Each column is multiplied by the inverse found so far
/// with `trmv`, then scaled by minus its inverted diagonal. Returns `info`, the 1-based index of
/// a zero diagonal element.
pub(super) fn trti2_checked<T: Real>(
    upper: bool,
    unit: bool,
    n: usize,
    a: &mut [T],
    lda: usize,
) -> usize {
    if n == 0 {
        return 0;
    }
    if !unit {
        if let Some(zero) = (0..n).find(|&i| a[i + i * lda] == T::ZERO) {
            return zero + 1;
        }
    }
    let invert_diagonal = |a: &mut [T], j: usize| {
        if unit {
            return -T::ONE;
        }
        let inverse = T::ONE / a[j + j * lda];
        a[j + j * lda] = inverse;
        -inverse
    };
    if upper {
        for j in 0..n {
            let factor = invert_diagonal(a, j);
            let (inverted, right) = a.split_at_mut(j * lda);
            let column = &mut right[..j];
            openblas::trmv(true, unit, j, inverted, lda, column);
            column.iter_mut().for_each(|value| *value = factor * *value);
        }
    } else {
        for j in (0..n).rev() {
            let factor = invert_diagonal(a, j);
            if j + 1 == n {
                continue;
            }
            let (left, inverted) = a.split_at_mut((j + 1) * lda);
            let column = &mut left[j * lda + j + 1..j * lda + n];
            openblas::trmv(false, unit, n - j - 1, &inverted[j + 1..], lda, column);
            column.iter_mut().for_each(|value| *value = factor * *value);
        }
    }
    0
}

// ---------------------------------------------------------------------------------------------
// Symmetric positive definite matrices: dpotf2, dpotrs, dpocon, dpotri
// ---------------------------------------------------------------------------------------------

/// `dpotf2` as OpenBLAS implements it (`potf2_U.c` and `potf2_L.c`), which its `dpotrf` runs up
/// to order 32: factor a symmetric positive definite matrix in place as `Uᵀ U` (upper) or `L Lᵀ`
/// (lower), reading only that triangle. Returns `info`, the 1-based order of the first leading
/// minor that is not positive definite; as in OpenBLAS, a NaN minor is not detected.
///
pub(super) fn potf2<T: Real>(upper: bool, n: usize, a: &mut [T], lda: usize) -> usize {
    for j in 0..n {
        let ajj = if upper {
            let column = &a[j * lda..j * lda + j];
            a[j + j * lda] - openblas::dot(column, column)
        } else {
            a[j + j * lda] - openblas::dot_strided(j, &a[j..], lda, &a[j..], lda)
        };
        if ajj <= T::ZERO {
            a[j + j * lda] = ajj;
            return j + 1;
        }
        let ajj = ajj.sqrt();
        a[j + j * lda] = ajj;
        if j + 1 == n {
            continue;
        }
        if upper {
            // dgemv('T', j, n-j-1, -1, A(0:j, j+1:), A(0:j, j), 1, A(j, j+1:)), on a copy of
            // row j because it shares storage with the matrix.
            let mut row: Vec<T> = (j + 1..n).map(|column| a[j + column * lda]).collect();
            let column = &a[j * lda..j * lda + j];
            openblas::gemv_t(
                j,
                n - j - 1,
                -T::ONE,
                &a[(j + 1) * lda..],
                lda,
                column,
                &mut row,
            );
            for (offset, value) in row.into_iter().enumerate() {
                a[j + (j + 1 + offset) * lda] = value;
            }
            scal(n - j - 1, T::ONE / ajj, a, j + (j + 1) * lda, lda);
        } else {
            let (left, right) = a.split_at_mut(j * lda);
            let below = &mut right[j + 1..n];
            if j > 0 {
                // dgemv('N', n-j-1, j, -1, A(j+1:, 0:j), A(j, 0:j), lda, A(j+1:, j))
                openblas::gemv_n(
                    n - j - 1,
                    j,
                    -T::ONE,
                    &left[j + 1..],
                    lda,
                    &left[j..],
                    lda,
                    below,
                );
            }
            let reciprocal = T::ONE / ajj;
            below
                .iter_mut()
                .for_each(|value| *value = reciprocal * *value);
        }
    }
    0
}

/// `dpotrs`: overwrite `b` with `A⁻¹ b` from `potf2`'s factor.
#[allow(clippy::too_many_arguments)]
pub(super) fn potrs<T: Real>(
    upper: bool,
    n: usize,
    nrhs: usize,
    c: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) {
    if n == 0 || nrhs == 0 {
        return;
    }
    if upper {
        openblas::trsm_interface(true, true, false, n, nrhs, c, lda, b, ldb);
        openblas::trsm_interface(true, false, false, n, nrhs, c, lda, b, ldb);
    } else {
        openblas::trsm_interface(false, false, false, n, nrhs, c, lda, b, ldb);
        openblas::trsm_interface(false, true, false, n, nrhs, c, lda, b, ldb);
    }
}

/// `dpocon`: estimate the reciprocal condition number from `potf2`'s factor and `anorm`.
pub(super) fn pocon<T: Real>(upper: bool, n: usize, c: &[T], lda: usize, anorm: T) -> T {
    if n == 0 {
        return T::ONE;
    }
    if anorm == T::ZERO {
        return T::ZERO;
    }
    let smlnum = T::SAFE_MIN;
    let mut normin = false;
    let mut cnorm = vec![T::ZERO; n];
    let estimate = lacn2(n, |_, x| {
        let (scale_l, scale_u) = if upper {
            let scale_l = latrs(true, true, false, normin, n, c, lda, x, &mut cnorm);
            let scale_u = latrs(true, false, false, true, n, c, lda, x, &mut cnorm);
            (scale_l, scale_u)
        } else {
            let scale_l = latrs(false, false, false, normin, n, c, lda, x, &mut cnorm);
            let scale_u = latrs(false, true, false, true, n, c, lda, x, &mut cnorm);
            (scale_l, scale_u)
        };
        normin = true;
        rescale(x, scale_l * scale_u, smlnum)
    });
    match estimate {
        Some(ainvnm) if ainvnm != T::ZERO => (T::ONE / ainvnm) / anorm,
        _ => T::ZERO,
    }
}

/// `dpotri`: overwrite `potf2`'s factor with the matching triangle of `A⁻¹`. Returns `info`,
/// the 1-based index of a zero diagonal element of the factor.
pub(super) fn potri<T: Real>(upper: bool, n: usize, a: &mut [T], lda: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let info = trti2_checked(upper, false, n, a, lda);
    if info > 0 {
        return info;
    }
    lauu2(upper, n, a, lda);
    0
}

/// `dlauu2` as OpenBLAS implements it (`lauu2_U.c` and `lauu2_L.c`), which its `dlauum` runs up
/// to order 64: overwrite a triangular `U` with `U Uᵀ`, or `L` with `Lᵀ L`. Each step scales row
/// or column `i` by its diagonal element, adds the squared off-diagonal part to the diagonal with
/// `dot`, and accumulates the remaining products with `gemv`.
fn lauu2<T: Real>(upper: bool, n: usize, a: &mut [T], lda: usize) {
    for i in 0..n {
        let aii = a[i + i * lda];
        if upper {
            scal(i + 1, aii, a, i * lda, 1);
        } else {
            scal(i + 1, aii, a, i, lda);
        }
        if i + 1 == n {
            continue;
        }
        let rest = n - i - 1;
        if upper {
            let row = &a[i + (i + 1) * lda..];
            a[i + i * lda] += openblas::dot_strided(rest, row, lda, row, lda);
            if i > 0 {
                // dgemv('N', i, n-i-1, 1, A(0:i, i+1:), A(i, i+1:), lda, A(0:i, i))
                let (left, right) = a.split_at_mut((i + 1) * lda);
                let target = &mut left[i * lda..i * lda + i];
                openblas::gemv_n(i, rest, T::ONE, right, lda, &right[i..], lda, target);
            }
        } else {
            let column = &a[i + 1 + i * lda..n + i * lda];
            a[i + i * lda] += openblas::dot(column, column);
            // dgemv('T', n-i-1, i, 1, A(i+1:, 0:i), A(i+1:, i), 1, A(i, 0:i)), on a copy of row
            // i because it shares storage with the matrix.
            let mut row: Vec<T> = (0..i).map(|column| a[i + column * lda]).collect();
            let column = &a[i + 1 + i * lda..n + i * lda];
            openblas::gemv_t(rest, i, T::ONE, &a[i + 1..], lda, column, &mut row);
            for (offset, value) in row.into_iter().enumerate() {
                a[i + offset * lda] = value;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Symmetric indefinite matrices: dsytf2, dsytrs, dsycon, dsytri
// ---------------------------------------------------------------------------------------------

/// `dsytf2`: factor a symmetric matrix in place with Bunch-Kaufman diagonal pivoting, reading
/// only one triangle. Pivots keep LAPACK's 1-based signed convention: `k > 0` exchanged a 1×1
/// block with row `k`, and a pair of `-k` marks a 2×2 block. Returns the pivots and `info`, the
/// 1-based index of an exactly zero block.
pub(super) fn sytf2<T: Real>(upper: bool, n: usize, a: &mut [T], lda: usize) -> (Vec<i64>, usize) {
    let alpha = (T::ONE + T::from_f64(17.0).sqrt()) / T::from_f64(8.0);
    let mut pivots = vec![0_i64; n];
    let mut info = 0;
    let at = |i: usize, j: usize| i + j * lda;
    if upper {
        // k is 1-based, as in the reference loop.
        let mut k = n;
        while k >= 1 {
            let kc = k - 1;
            let mut kstep = 1;
            let absakk = a[at(kc, kc)].abs();
            let (imax, colmax) = if k > 1 {
                let imax = iamax(k - 1, a, at(0, kc), 1);
                (imax, a[at(imax, kc)].abs())
            } else {
                (0, T::ZERO)
            };
            let kp;
            if max(absakk, colmax) == T::ZERO || absakk.is_nan() {
                if info == 0 {
                    info = k;
                }
                kp = kc;
            } else {
                if absakk >= alpha * colmax {
                    kp = kc;
                } else {
                    let jmax = imax + 1 + iamax(kc - imax, a, at(imax, imax + 1), lda);
                    let mut rowmax = a[at(imax, jmax)].abs();
                    if imax > 0 {
                        let jmax = iamax(imax, a, at(0, imax), 1);
                        rowmax = max(rowmax, a[at(jmax, imax)].abs());
                    }
                    if absakk >= alpha * colmax * (colmax / rowmax) {
                        kp = kc;
                    } else if a[at(imax, imax)].abs() >= alpha * rowmax {
                        kp = imax;
                    } else {
                        kp = imax;
                        kstep = 2;
                    }
                }
                let kk = kc + 1 - kstep;
                if kp != kk {
                    swap(kp, a, at(0, kk), 1, at(0, kp), 1);
                    swap(kk - kp - 1, a, at(kp + 1, kk), 1, at(kp, kp + 1), lda);
                    a.swap(at(kk, kk), at(kp, kp));
                    if kstep == 2 {
                        a.swap(at(kc - 1, kc), at(kp, kc));
                    }
                }
                if kstep == 1 {
                    let r1 = T::ONE / a[at(kc, kc)];
                    // dsyr('U', k-1, -r1, A(0:k-1, k), A)
                    for j in 0..kc {
                        let xj = a[at(j, kc)];
                        if xj != T::ZERO {
                            let temp = -r1 * xj;
                            for i in 0..=j {
                                a[at(i, j)] = a[at(i, j)] + a[at(i, kc)] * temp;
                            }
                        }
                    }
                    scal(kc, r1, a, at(0, kc), 1);
                } else if k > 2 {
                    let mut d12 = a[at(kc - 1, kc)];
                    let d22 = a[at(kc - 1, kc - 1)] / d12;
                    let d11 = a[at(kc, kc)] / d12;
                    let t = T::ONE / (d11 * d22 - T::ONE);
                    d12 = t / d12;
                    for j in (0..kc - 1).rev() {
                        let wkm1 = d12 * (d11 * a[at(j, kc - 1)] - a[at(j, kc)]);
                        let wk = d12 * (d22 * a[at(j, kc)] - a[at(j, kc - 1)]);
                        for i in (0..=j).rev() {
                            a[at(i, j)] = a[at(i, j)] - a[at(i, kc)] * wk - a[at(i, kc - 1)] * wkm1;
                        }
                        a[at(j, kc)] = wk;
                        a[at(j, kc - 1)] = wkm1;
                    }
                }
            }
            let kp1 = kp as i64 + 1;
            if kstep == 1 {
                pivots[kc] = kp1;
            } else {
                pivots[kc] = -kp1;
                pivots[kc - 1] = -kp1;
            }
            k -= kstep;
        }
    } else {
        let mut kc = 0;
        while kc < n {
            let mut kstep = 1;
            let absakk = a[at(kc, kc)].abs();
            let (imax, colmax) = if kc + 1 < n {
                let imax = kc + 1 + iamax(n - kc - 1, a, at(kc + 1, kc), 1);
                (imax, a[at(imax, kc)].abs())
            } else {
                (0, T::ZERO)
            };
            let kp;
            if max(absakk, colmax) == T::ZERO || absakk.is_nan() {
                if info == 0 {
                    info = kc + 1;
                }
                kp = kc;
            } else {
                if absakk >= alpha * colmax {
                    kp = kc;
                } else {
                    let jmax = kc + iamax(imax - kc, a, at(imax, kc), lda);
                    let mut rowmax = a[at(imax, jmax)].abs();
                    if imax + 1 < n {
                        let jmax = imax + 1 + iamax(n - imax - 1, a, at(imax + 1, imax), 1);
                        rowmax = max(rowmax, a[at(jmax, imax)].abs());
                    }
                    if absakk >= alpha * colmax * (colmax / rowmax) {
                        kp = kc;
                    } else if a[at(imax, imax)].abs() >= alpha * rowmax {
                        kp = imax;
                    } else {
                        kp = imax;
                        kstep = 2;
                    }
                }
                let kk = kc + kstep - 1;
                if kp != kk {
                    if kp + 1 < n {
                        swap(n - kp - 1, a, at(kp + 1, kk), 1, at(kp + 1, kp), 1);
                    }
                    swap(kp - kk - 1, a, at(kk + 1, kk), 1, at(kp, kk + 1), lda);
                    a.swap(at(kk, kk), at(kp, kp));
                    if kstep == 2 {
                        a.swap(at(kc + 1, kc), at(kp, kc));
                    }
                }
                if kstep == 1 {
                    if kc + 1 < n {
                        let d11 = T::ONE / a[at(kc, kc)];
                        // dsyr('L', n-k-1, -d11, A(k+1:, k), A(k+1:, k+1:))
                        for j in kc + 1..n {
                            let xj = a[at(j, kc)];
                            if xj != T::ZERO {
                                let temp = -d11 * xj;
                                for i in j..n {
                                    a[at(i, j)] = a[at(i, j)] + a[at(i, kc)] * temp;
                                }
                            }
                        }
                        scal(n - kc - 1, d11, a, at(kc + 1, kc), 1);
                    }
                } else if kc + 2 < n {
                    let mut d21 = a[at(kc + 1, kc)];
                    let d11 = a[at(kc + 1, kc + 1)] / d21;
                    let d22 = a[at(kc, kc)] / d21;
                    let t = T::ONE / (d11 * d22 - T::ONE);
                    d21 = t / d21;
                    for j in kc + 2..n {
                        let wk = d21 * (d11 * a[at(j, kc)] - a[at(j, kc + 1)]);
                        let wkp1 = d21 * (d22 * a[at(j, kc + 1)] - a[at(j, kc)]);
                        for i in j..n {
                            a[at(i, j)] = a[at(i, j)] - a[at(i, kc)] * wk - a[at(i, kc + 1)] * wkp1;
                        }
                        a[at(j, kc)] = wk;
                        a[at(j, kc + 1)] = wkp1;
                    }
                }
            }
            let kp1 = kp as i64 + 1;
            if kstep == 1 {
                pivots[kc] = kp1;
            } else {
                pivots[kc] = -kp1;
                pivots[kc + 1] = -kp1;
            }
            kc += kstep;
        }
    }
    (pivots, info)
}

/// `dsytrs`: overwrite `b` with `A⁻¹ b` from `sytf2`'s factorization.
#[allow(clippy::too_many_arguments)]
pub(super) fn sytrs<T: Real>(
    upper: bool,
    n: usize,
    nrhs: usize,
    a: &[T],
    lda: usize,
    pivots: &[i64],
    b: &mut [T],
    ldb: usize,
) {
    if n == 0 || nrhs == 0 {
        return;
    }
    let at = |i: usize, j: usize| i + j * lda;
    let bt = |i: usize, j: usize| i + j * ldb;
    let pivot = |k: usize| pivots[k].unsigned_abs() as usize - 1;
    // dger(m, nrhs, -1, x = A(rows, column), y = B(row, :), B(first.., :))
    let ger = |b: &mut [T], rows: std::ops::Range<usize>, column: usize, row: usize| {
        for j in 0..nrhs {
            let y = b[bt(row, j)];
            if y != T::ZERO {
                let temp = -y;
                for i in rows.clone() {
                    b[bt(i, j)] = b[bt(i, j)] + a[at(i, column)] * temp;
                }
            }
        }
    };
    // dgemv('T', m, nrhs, -1, B(rows, :), A(rows, column), 1, B(row, :))
    let gemv = |b: &mut [T], rows: std::ops::Range<usize>, column: usize, row: usize| {
        for j in 0..nrhs {
            let mut temp = T::ZERO;
            for i in rows.clone() {
                temp += b[bt(i, j)] * a[at(i, column)];
            }
            b[bt(row, j)] = b[bt(row, j)] + -temp;
        }
    };
    let solve_block = |b: &mut [T], first: usize, second: usize, akm1k: T, akm1: T, ak: T| {
        let denominator = akm1 * ak - T::ONE;
        for j in 0..nrhs {
            let bkm1 = b[bt(first, j)] / akm1k;
            let bk = b[bt(second, j)] / akm1k;
            b[bt(first, j)] = (ak * bkm1 - bk) / denominator;
            b[bt(second, j)] = (akm1 * bk - bkm1) / denominator;
        }
    };
    if upper {
        let mut k = n;
        while k >= 1 {
            let kc = k - 1;
            if pivots[kc] > 0 {
                swap_rows(b, ldb, nrhs, kc, pivot(kc));
                ger(b, 0..kc, kc, kc);
                scal(nrhs, T::ONE / a[at(kc, kc)], b, kc, ldb);
                k -= 1;
            } else {
                swap_rows(b, ldb, nrhs, kc - 1, pivot(kc));
                ger(b, 0..kc - 1, kc, kc);
                ger(b, 0..kc - 1, kc - 1, kc - 1);
                let akm1k = a[at(kc - 1, kc)];
                let akm1 = a[at(kc - 1, kc - 1)] / akm1k;
                let ak = a[at(kc, kc)] / akm1k;
                solve_block(b, kc - 1, kc, akm1k, akm1, ak);
                k -= 2;
            }
        }
        let mut kc = 0;
        while kc < n {
            if pivots[kc] > 0 {
                gemv(b, 0..kc, kc, kc);
                swap_rows(b, ldb, nrhs, kc, pivot(kc));
                kc += 1;
            } else {
                gemv(b, 0..kc, kc, kc);
                gemv(b, 0..kc, kc + 1, kc + 1);
                swap_rows(b, ldb, nrhs, kc, pivot(kc));
                kc += 2;
            }
        }
    } else {
        let mut kc = 0;
        while kc < n {
            if pivots[kc] > 0 {
                swap_rows(b, ldb, nrhs, kc, pivot(kc));
                if kc + 1 < n {
                    ger(b, kc + 1..n, kc, kc);
                }
                scal(nrhs, T::ONE / a[at(kc, kc)], b, kc, ldb);
                kc += 1;
            } else {
                swap_rows(b, ldb, nrhs, kc + 1, pivot(kc));
                if kc + 2 < n {
                    ger(b, kc + 2..n, kc, kc);
                    ger(b, kc + 2..n, kc + 1, kc + 1);
                }
                let akm1k = a[at(kc + 1, kc)];
                let akm1 = a[at(kc, kc)] / akm1k;
                let ak = a[at(kc + 1, kc + 1)] / akm1k;
                solve_block(b, kc, kc + 1, akm1k, akm1, ak);
                kc += 2;
            }
        }
        let mut k = n;
        while k >= 1 {
            let kc = k - 1;
            if pivots[kc] > 0 {
                if kc + 1 < n {
                    gemv(b, kc + 1..n, kc, kc);
                }
                swap_rows(b, ldb, nrhs, kc, pivot(kc));
                k -= 1;
            } else {
                if kc + 1 < n {
                    gemv(b, kc + 1..n, kc, kc);
                    gemv(b, kc + 1..n, kc - 1, kc - 1);
                }
                swap_rows(b, ldb, nrhs, kc, pivot(kc));
                k -= 2;
            }
        }
    }
}

/// `dsycon`: estimate the reciprocal condition number from `sytf2`'s factorization and `anorm`.
pub(super) fn sycon<T: Real>(
    upper: bool,
    n: usize,
    a: &[T],
    lda: usize,
    pivots: &[i64],
    anorm: T,
) -> T {
    if n == 0 {
        return T::ONE;
    }
    if anorm <= T::ZERO {
        return T::ZERO;
    }
    let singular = |i: usize| pivots[i] > 0 && a[i + i * lda] == T::ZERO;
    let any_singular = if upper {
        (0..n).rev().any(singular)
    } else {
        (0..n).any(singular)
    };
    if any_singular {
        return T::ZERO;
    }
    let estimate = lacn2(n, |_, x| {
        sytrs(upper, n, 1, a, lda, pivots, x, n);
        true
    });
    match estimate {
        Some(ainvnm) if ainvnm != T::ZERO => (T::ONE / ainvnm) / anorm,
        _ => T::ZERO,
    }
}

/// `dsytri`: overwrite `sytf2`'s factorization with the matching triangle of `A⁻¹`. Returns
/// `info`, the 1-based index of an exactly zero diagonal block.
pub(super) fn sytri<T: Real>(
    upper: bool,
    n: usize,
    a: &mut [T],
    lda: usize,
    pivots: &[i64],
) -> usize {
    if n == 0 {
        return 0;
    }
    let at = |i: usize, j: usize| i + j * lda;
    let zero_block = |i: usize| pivots[i] > 0 && a[at(i, i)] == T::ZERO;
    let blocked = if upper {
        (0..n).rev().find(|&i| zero_block(i))
    } else {
        (0..n).find(|&i| zero_block(i))
    };
    if let Some(index) = blocked {
        return index + 1;
    }
    let mut work = vec![T::ZERO; n];
    // dsymv(uplo, m, -1, A(first.., first..), work, 0, A(first.., column)) for the m × m
    // trailing (lower) or leading (upper) block.
    let symv = |a: &mut [T], work: &[T], first: usize, m: usize, column: usize| {
        for i in 0..m {
            a[at(first + i, column)] = T::ZERO;
        }
        for j in 0..m {
            let temp1 = -work[j];
            let mut temp2 = T::ZERO;
            if upper {
                for i in 0..j {
                    let aij = a[at(first + i, first + j)];
                    a[at(first + i, column)] = a[at(first + i, column)] + temp1 * aij;
                    temp2 += aij * work[i];
                }
                let ajj = a[at(first + j, first + j)];
                a[at(first + j, column)] = a[at(first + j, column)] + temp1 * ajj + -temp2;
            } else {
                let ajj = a[at(first + j, first + j)];
                a[at(first + j, column)] = a[at(first + j, column)] + temp1 * ajj;
                for i in j + 1..m {
                    let aij = a[at(first + i, first + j)];
                    a[at(first + i, column)] = a[at(first + i, column)] + temp1 * aij;
                    temp2 += aij * work[i];
                }
                a[at(first + j, column)] = a[at(first + j, column)] + -temp2;
            }
        }
    };
    if upper {
        let mut k = 0;
        while k < n {
            let kstep;
            if pivots[k] > 0 {
                a[at(k, k)] = T::ONE / a[at(k, k)];
                if k > 0 {
                    for i in 0..k {
                        work[i] = a[at(i, k)];
                    }
                    symv(a, &work, 0, k, k);
                    a[at(k, k)] = a[at(k, k)] - dot2(k, a, at(0, k), 1, &work, 0);
                }
                kstep = 1;
            } else {
                let t = a[at(k, k + 1)].abs();
                let ak = a[at(k, k)] / t;
                let akp1 = a[at(k + 1, k + 1)] / t;
                let akkp1 = a[at(k, k + 1)] / t;
                let d = t * (ak * akp1 - T::ONE);
                a[at(k, k)] = akp1 / d;
                a[at(k + 1, k + 1)] = ak / d;
                a[at(k, k + 1)] = -akkp1 / d;
                if k > 0 {
                    for i in 0..k {
                        work[i] = a[at(i, k)];
                    }
                    symv(a, &work, 0, k, k);
                    a[at(k, k)] = a[at(k, k)] - dot2(k, a, at(0, k), 1, &work, 0);
                    a[at(k, k + 1)] = a[at(k, k + 1)] - dot(k, a, at(0, k), 1, at(0, k + 1), 1);
                    for i in 0..k {
                        work[i] = a[at(i, k + 1)];
                    }
                    symv(a, &work, 0, k, k + 1);
                    a[at(k + 1, k + 1)] =
                        a[at(k + 1, k + 1)] - dot2(k, a, at(0, k + 1), 1, &work, 0);
                }
                kstep = 2;
            }
            let kp = pivots[k].unsigned_abs() as usize - 1;
            if kp != k {
                swap(kp, a, at(0, k), 1, at(0, kp), 1);
                swap(k - kp - 1, a, at(kp + 1, k), 1, at(kp, kp + 1), lda);
                a.swap(at(k, k), at(kp, kp));
                if kstep == 2 {
                    a.swap(at(k, k + 1), at(kp, k + 1));
                }
            }
            k += kstep;
        }
    } else {
        let mut k = n;
        while k >= 1 {
            let kc = k - 1;
            let kstep;
            if pivots[kc] > 0 {
                a[at(kc, kc)] = T::ONE / a[at(kc, kc)];
                if kc + 1 < n {
                    let m = n - kc - 1;
                    for i in 0..m {
                        work[i] = a[at(kc + 1 + i, kc)];
                    }
                    symv(a, &work, kc + 1, m, kc);
                    a[at(kc, kc)] = a[at(kc, kc)] - dot2(m, a, at(kc + 1, kc), 1, &work, 0);
                }
                kstep = 1;
            } else {
                let t = a[at(kc, kc - 1)].abs();
                let ak = a[at(kc - 1, kc - 1)] / t;
                let akp1 = a[at(kc, kc)] / t;
                let akkp1 = a[at(kc, kc - 1)] / t;
                let d = t * (ak * akp1 - T::ONE);
                a[at(kc - 1, kc - 1)] = akp1 / d;
                a[at(kc, kc)] = ak / d;
                a[at(kc, kc - 1)] = -akkp1 / d;
                if kc + 1 < n {
                    let m = n - kc - 1;
                    for i in 0..m {
                        work[i] = a[at(kc + 1 + i, kc)];
                    }
                    symv(a, &work, kc + 1, m, kc);
                    a[at(kc, kc)] = a[at(kc, kc)] - dot2(m, a, at(kc + 1, kc), 1, &work, 0);
                    a[at(kc, kc - 1)] =
                        a[at(kc, kc - 1)] - dot(m, a, at(kc + 1, kc), 1, at(kc + 1, kc - 1), 1);
                    for i in 0..m {
                        work[i] = a[at(kc + 1 + i, kc - 1)];
                    }
                    symv(a, &work, kc + 1, m, kc - 1);
                    a[at(kc - 1, kc - 1)] =
                        a[at(kc - 1, kc - 1)] - dot2(m, a, at(kc + 1, kc - 1), 1, &work, 0);
                }
                kstep = 2;
            }
            let kp = pivots[kc].unsigned_abs() as usize - 1;
            if kp != kc {
                if kp + 1 < n {
                    swap(n - kp - 1, a, at(kp + 1, kc), 1, at(kp + 1, kp), 1);
                }
                swap(kp - kc - 1, a, at(kc + 1, kc), 1, at(kp, kc + 1), lda);
                a.swap(at(kc, kc), at(kp, kp));
                if kstep == 2 {
                    a.swap(at(kc, kc - 1), at(kp, kc - 1));
                }
            }
            k -= kstep;
        }
    }
    0
}

// ---------------------------------------------------------------------------------------------
// Tridiagonal matrices: dgttrf, dgttrs, dgtcon, dgtsv
// ---------------------------------------------------------------------------------------------

/// `dgttrf`'s factors of a tridiagonal matrix: the modified diagonals, the second superdiagonal
/// of `U`, and the 0-based row exchanged at each step.
pub(super) struct TridiagonalLu<T> {
    pub dl: Vec<T>,
    pub d: Vec<T>,
    pub du: Vec<T>,
    pub du2: Vec<T>,
    pub pivots: Vec<usize>,
}

/// `dgttrf`: factor a tridiagonal matrix with partial pivoting. Returns the factors and `info`,
/// the 1-based index of the first zero pivot.
pub(super) fn gttrf<T: Real>(
    mut dl: Vec<T>,
    mut d: Vec<T>,
    mut du: Vec<T>,
) -> (TridiagonalLu<T>, usize) {
    let n = d.len();
    let mut pivots: Vec<usize> = (0..n).collect();
    let mut du2 = vec![T::ZERO; n.saturating_sub(2)];
    let mut eliminate = |i: usize, last: bool, dl: &mut [T], d: &mut [T], du: &mut [T]| {
        if d[i].abs() >= dl[i].abs() {
            if d[i] != T::ZERO {
                let fact = dl[i] / d[i];
                dl[i] = fact;
                d[i + 1] -= fact * du[i];
            }
        } else {
            let fact = d[i] / dl[i];
            d[i] = dl[i];
            dl[i] = fact;
            let temp = du[i];
            du[i] = d[i + 1];
            d[i + 1] = temp - fact * d[i + 1];
            if !last {
                du2[i] = du[i + 1];
                du[i + 1] = -fact * du[i + 1];
            }
            pivots[i] = i + 1;
        }
    };
    for i in 0..n.saturating_sub(2) {
        eliminate(i, false, &mut dl, &mut d, &mut du);
    }
    if n > 1 {
        eliminate(n - 2, true, &mut dl, &mut d, &mut du);
    }
    let info = d
        .iter()
        .position(|value| *value == T::ZERO)
        .map_or(0, |i| i + 1);
    (
        TridiagonalLu {
            dl,
            d,
            du,
            du2,
            pivots,
        },
        info,
    )
}

/// `dgttrs` (through `dgtts2`): overwrite the `n × nrhs` matrix `b` with `A⁻¹ b`, or `A⁻ᵀ b`
/// when `transpose`, from `gttrf`'s factors.
pub(super) fn gttrs<T: Real>(
    transpose: bool,
    lu: &TridiagonalLu<T>,
    nrhs: usize,
    b: &mut [T],
    ldb: usize,
) {
    let n = lu.d.len();
    if n == 0 || nrhs == 0 {
        return;
    }
    let (dl, d, du, du2) = (&lu.dl, &lu.d, &lu.du, &lu.du2);
    for j in 0..nrhs {
        let column = &mut b[j * ldb..j * ldb + n];
        if !transpose {
            for i in 0..n - 1 {
                if lu.pivots[i] == i {
                    column[i + 1] -= dl[i] * column[i];
                } else {
                    let temp = column[i];
                    column[i] = column[i + 1];
                    column[i + 1] = temp - dl[i] * column[i];
                }
            }
            column[n - 1] /= d[n - 1];
            if n > 1 {
                column[n - 2] = (column[n - 2] - du[n - 2] * column[n - 1]) / d[n - 2];
            }
            for i in (0..n.saturating_sub(2)).rev() {
                column[i] = (column[i] - du[i] * column[i + 1] - du2[i] * column[i + 2]) / d[i];
            }
        } else {
            column[0] /= d[0];
            if n > 1 {
                column[1] = (column[1] - du[0] * column[0]) / d[1];
            }
            for i in 2..n {
                column[i] =
                    (column[i] - du[i - 1] * column[i - 1] - du2[i - 2] * column[i - 2]) / d[i];
            }
            for i in (0..n - 1).rev() {
                if lu.pivots[i] == i {
                    column[i] -= dl[i] * column[i + 1];
                } else {
                    let temp = column[i + 1];
                    column[i + 1] = column[i] - dl[i] * temp;
                    column[i] = temp;
                }
            }
        }
    }
}

/// `dgtcon` with the 1-norm: estimate the reciprocal condition number from `gttrf`'s factors.
pub(super) fn gtcon<T: Real>(lu: &TridiagonalLu<T>, anorm: T) -> T {
    let n = lu.d.len();
    if n == 0 {
        return T::ONE;
    }
    if anorm == T::ZERO || lu.d.contains(&T::ZERO) {
        return T::ZERO;
    }
    let estimate = lacn2(n, |kase, x| {
        gttrs(kase != 1, lu, 1, x, n);
        true
    });
    match estimate {
        Some(ainvnm) if ainvnm != T::ZERO => (T::ONE / ainvnm) / anorm,
        _ => T::ZERO,
    }
}

/// `dgtsv`: solve a tridiagonal system by Gaussian elimination with partial pivoting,
/// overwriting the `n × nrhs` matrix `b`. Returns `info`, the 1-based index of a zero pivot.
pub(super) fn gtsv<T: Real>(
    dl: &mut [T],
    d: &mut [T],
    du: &mut [T],
    nrhs: usize,
    b: &mut [T],
    ldb: usize,
) -> usize {
    let n = d.len();
    if n == 0 {
        return 0;
    }
    let bt = |i: usize, j: usize| i + j * ldb;
    let steps = n - 1;
    for i in 0..steps {
        let last = i + 1 == steps;
        if d[i].abs() >= dl[i].abs() {
            if d[i] == T::ZERO {
                return i + 1;
            }
            let fact = dl[i] / d[i];
            d[i + 1] -= fact * du[i];
            for j in 0..nrhs {
                b[bt(i + 1, j)] = b[bt(i + 1, j)] - fact * b[bt(i, j)];
            }
            if !last {
                dl[i] = T::ZERO;
            }
        } else {
            let fact = d[i] / dl[i];
            d[i] = dl[i];
            let temp = d[i + 1];
            d[i + 1] = du[i] - fact * temp;
            if !last {
                dl[i] = du[i + 1];
                du[i + 1] = -fact * dl[i];
            }
            du[i] = temp;
            for j in 0..nrhs {
                let temp = b[bt(i, j)];
                b[bt(i, j)] = b[bt(i + 1, j)];
                b[bt(i + 1, j)] = temp - fact * b[bt(i + 1, j)];
            }
        }
    }
    if d[n - 1] == T::ZERO {
        return n;
    }
    for j in 0..nrhs {
        b[bt(n - 1, j)] = b[bt(n - 1, j)] / d[n - 1];
        if n > 1 {
            b[bt(n - 2, j)] = (b[bt(n - 2, j)] - du[n - 2] * b[bt(n - 1, j)]) / d[n - 2];
        }
        for i in (0..n.saturating_sub(2)).rev() {
            b[bt(i, j)] = (b[bt(i, j)] - du[i] * b[bt(i + 1, j)] - dl[i] * b[bt(i + 2, j)]) / d[i];
        }
    }
    0
}

// ---------------------------------------------------------------------------------------------
// Band matrices: dgbtf2 and dgbtrs, as `dgbsv` runs them
// ---------------------------------------------------------------------------------------------

/// `dgbtf2`: factor an `n × n` band matrix with `kl` subdiagonals and `ku` superdiagonals held
/// in LAPACK band storage `ab` (`ldab = 2 kl + ku + 1` rows, the first `kl` of them workspace).
/// Returns the 0-based row exchanged at each step and `info`.
pub(super) fn gbtf2<T: Real>(
    n: usize,
    kl: usize,
    ku: usize,
    ab: &mut [T],
    ldab: usize,
) -> (Vec<usize>, usize) {
    let kv = ku + kl;
    let at = |i: usize, j: usize| i + j * ldab;
    let mut pivots = Vec::with_capacity(n);
    let mut info = 0;
    // Zero the fill-in elements in columns ku+1 .. kv-1 (0-based).
    for j in ku + 1..kv.min(n) {
        for i in kv - j..kl {
            ab[at(i, j)] = T::ZERO;
        }
    }
    let mut ju = 0;
    for j in 0..n {
        if j + kv < n {
            for i in 0..kl {
                ab[at(i, j + kv)] = T::ZERO;
            }
        }
        let km = kl.min(n - j - 1);
        let jp = iamax(km + 1, ab, at(kv, j), 1);
        pivots.push(jp + j);
        if ab[at(kv + jp, j)] != T::ZERO {
            ju = ju.max((j + ku + jp).min(n - 1));
            if jp != 0 {
                swap(
                    ju - j + 1,
                    ab,
                    at(kv + jp, j),
                    ldab - 1,
                    at(kv, j),
                    ldab - 1,
                );
            }
            if km > 0 {
                scal(km, T::ONE / ab[at(kv, j)], ab, at(kv + 1, j), 1);
                if ju > j {
                    // dger(km, ju-j, -1, AB(kv+1, j), AB(kv-1, j+1) stride ldab-1, AB(kv, j+1))
                    for c in 0..ju - j {
                        let y = ab[at(kv - 1, j + 1) + c * (ldab - 1)];
                        if y != T::ZERO {
                            let temp = -y;
                            let target = at(kv, j + 1) + c * (ldab - 1);
                            for i in 0..km {
                                ab[target + i] += ab[at(kv + 1 + i, j)] * temp;
                            }
                        }
                    }
                }
            }
        } else if info == 0 {
            info = j + 1;
        }
    }
    (pivots, info)
}

/// `dgbtrs` without transposition: overwrite `b` with `A⁻¹ b` from `gbtf2`'s factors.
#[allow(clippy::too_many_arguments)]
pub(super) fn gbtrs<T: Real>(
    n: usize,
    kl: usize,
    ku: usize,
    nrhs: usize,
    ab: &[T],
    ldab: usize,
    pivots: &[usize],
    b: &mut [T],
    ldb: usize,
) {
    if n == 0 || nrhs == 0 {
        return;
    }
    let kd = ku + kl;
    let at = |i: usize, j: usize| i + j * ldab;
    if kl > 0 {
        for j in 0..n - 1 {
            let lm = kl.min(n - j - 1);
            swap_rows(b, ldb, nrhs, pivots[j], j);
            for column in 0..nrhs {
                let y = b[j + column * ldb];
                if y != T::ZERO {
                    let temp = -y;
                    for i in 0..lm {
                        let row = j + 1 + i;
                        b[row + column * ldb] += ab[at(kd + 1 + i, j)] * temp;
                    }
                }
            }
        }
    }
    // dtbsv('U', 'N', 'N', n, kl+ku, ab, x) for each column.
    let k = kl + ku;
    for column in 0..nrhs {
        let x = &mut b[column * ldb..column * ldb + n];
        for j in (0..n).rev() {
            x[j] /= ab[at(k, j)];
            let temp = x[j];
            for i in (j.saturating_sub(k)..j).rev() {
                x[i] -= temp * ab[at(k + i - j, j)];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Householder QR: dgeqr2, dgeqp3 (through dlaqp2) and dorg2r
// ---------------------------------------------------------------------------------------------

/// `dlapy2`: `sqrt(x² + y²)` without undue overflow.
fn lapy2<T: Real>(x: T, y: T) -> T {
    if x.is_nan() {
        return x;
    }
    if y.is_nan() {
        return y;
    }
    let (w, z) = (max(x.abs(), y.abs()), min(x.abs(), y.abs()));
    if z == T::ZERO || w > T::OVERFLOW {
        return w;
    }
    let ratio = z / w;
    w * (T::ONE + ratio * ratio).sqrt()
}

/// `dlarfg` on the `m`-element column that starts at `a[start]`: overwrite it with `beta` and
/// the reflector's vector (whose leading 1 is implied), and return the reflector's `tau`.
fn larfg<T: Real>(m: usize, a: &mut [T], start: usize) -> T {
    if m <= 1 {
        return T::ZERO;
    }
    let tail = start + 1..start + m;
    let mut xnorm = T::nrm2(&a[tail.clone()]);
    if xnorm == T::ZERO {
        return T::ZERO;
    }
    let mut alpha = a[start];
    let mut beta = -lapy2(alpha, xnorm).copysign(alpha);
    // dlamch('S') / dlamch('E'), where 'E' is half the machine epsilon.
    let safmin = T::SAFE_MIN / (T::PRECISION * T::HALF);
    let mut rescalings = 0;
    if beta.abs() < safmin {
        let rsafmn = T::ONE / safmin;
        loop {
            rescalings += 1;
            scal(m - 1, rsafmn, a, start + 1, 1);
            beta *= rsafmn;
            alpha *= rsafmn;
            if !(beta.abs() < safmin && rescalings < 20) {
                break;
            }
        }
        xnorm = T::nrm2(&a[tail]);
        beta = -lapy2(alpha, xnorm).copysign(alpha);
    }
    let tau = (beta - alpha) / beta;
    scal(m - 1, T::ONE / (alpha - beta), a, start + 1, 1);
    for _ in 0..rescalings {
        beta *= safmin;
    }
    a[start] = beta;
    tau
}

/// `dlarf` from the left: apply `I - tau v vᵀ` to rows `row..m` of columns `columns` of the
/// `m`-row matrix `a`, where `v` is rows `row..m` of column `vector`, whose first element the
/// caller has set to 1.
fn larf_left<T: Real>(
    a: &mut [T],
    m: usize,
    row: usize,
    vector: usize,
    columns: std::ops::Range<usize>,
    tau: T,
) {
    if tau == T::ZERO {
        return;
    }
    let length = m - row;
    let weights = columns
        .clone()
        .map(|column| dot(length, a, row + column * m, 1, row + vector * m, 1))
        .collect::<Vec<_>>();
    for (column, weight) in columns.zip(weights) {
        if weight != T::ZERO {
            let temp = -tau * weight;
            for i in row..m {
                a[i + column * m] += a[i + vector * m] * temp;
            }
        }
    }
}

/// Apply reflector `step`, stored below the diagonal of column `step`, to the columns after it.
fn apply_reflector<T: Real>(a: &mut [T], m: usize, n: usize, step: usize, tau: T) {
    if step + 1 < n {
        let diagonal = step + step * m;
        let saved = a[diagonal];
        a[diagonal] = T::ONE;
        larf_left(a, m, step, step, step + 1..n, tau);
        a[diagonal] = saved;
    }
}

/// `dgeqr2`: factor the `m × n` column-major matrix `a` in place as `Q R`, leaving `R` on and
/// above the diagonal and the reflectors below it. Returns the reflectors' `tau`.
pub(super) fn geqr2<T: Real>(m: usize, n: usize, a: &mut [T]) -> Vec<T> {
    let steps = m.min(n);
    let mut tau = Vec::with_capacity(steps);
    for step in 0..steps {
        let factor = larfg(m - step, a, step + step * m);
        tau.push(factor);
        apply_reflector(a, m, n, step, factor);
    }
    tau
}

/// `dgeqp3` with every column free, which runs `dlaqp2` for matrices of up to 128 columns:
/// QR with column pivoting, `A P = Q R`. Returns `tau` and the 0-based original index of each
/// column of `A P`.
pub(super) fn geqp3<T: Real>(m: usize, n: usize, a: &mut [T]) -> (Vec<T>, Vec<usize>) {
    let steps = m.min(n);
    let mut pivots = (0..n).collect::<Vec<_>>();
    let mut tau = Vec::with_capacity(steps);
    if steps == 0 {
        return (tau, pivots);
    }
    let mut norms = (0..n)
        .map(|j| T::nrm2(&a[j * m..(j + 1) * m]))
        .collect::<Vec<_>>();
    let mut exact = norms.clone();
    // sqrt(dlamch('Epsilon')), with 'Epsilon' half the machine epsilon.
    let tol3z = (T::PRECISION * T::HALF).sqrt();
    for i in 0..steps {
        let pvt = i + iamax(n - i, &norms, i, 1);
        if pvt != i {
            swap(m, a, pvt * m, 1, i * m, 1);
            pivots.swap(pvt, i);
            norms[pvt] = norms[i];
            exact[pvt] = exact[i];
        }
        let factor = larfg(m - i, a, i + i * m);
        tau.push(factor);
        apply_reflector(a, m, n, i, factor);
        for j in i + 1..n {
            if norms[j] == T::ZERO {
                continue;
            }
            let ratio = a[i + j * m].abs() / norms[j];
            let temp = max(T::ONE - ratio * ratio, T::ZERO);
            let scaled = norms[j] / exact[j];
            if temp * (scaled * scaled) <= tol3z {
                if i + 1 < m {
                    norms[j] = T::nrm2(&a[i + 1 + j * m..(j + 1) * m]);
                    exact[j] = norms[j];
                } else {
                    norms[j] = T::ZERO;
                    exact[j] = T::ZERO;
                }
            } else {
                norms[j] *= temp.sqrt();
            }
        }
    }
    (tau, pivots)
}

/// `dorg2r`: the first `columns` columns of `Q` from the reflectors that [`geqr2`] or [`geqp3`]
/// left in the `m`-row matrix `factors`. `columns` is at least `tau.len()` and at most `m`.
pub(super) fn org2r<T: Real>(m: usize, columns: usize, factors: &[T], tau: &[T]) -> Vec<T> {
    let k = tau.len();
    let mut q = vec![T::ZERO; m * columns];
    q[..m * k].copy_from_slice(&factors[..m * k]);
    for j in k..columns {
        q[j + j * m] = T::ONE;
    }
    for i in (0..k).rev() {
        if i + 1 < columns {
            q[i + i * m] = T::ONE;
            larf_left(&mut q, m, i, i, i + 1..columns, tau[i]);
        }
        if i + 1 < m {
            scal(m - i - 1, -tau[i], &mut q, i + 1 + i * m, 1);
        }
        q[i + i * m] = T::ONE - tau[i];
        for l in 0..i {
            q[l + i * m] = T::ZERO;
        }
    }
    q
}

/// The norms `dlange` computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MatrixNorm {
    /// `'M'`: the largest magnitude.
    Max,
    /// `'1'` or `'O'`: the largest column sum of magnitudes.
    One,
    /// `'I'`: the largest row sum of magnitudes.
    Infinity,
}

/// `dlange` for an `m × n` column-major matrix. NaN propagates.
pub(super) fn lange<T: Real>(norm: MatrixNorm, m: usize, n: usize, a: &[T]) -> T {
    let largest = |values: &mut dyn Iterator<Item = T>| {
        let mut value = T::ZERO;
        for sum in values {
            if value < sum || sum.is_nan() {
                value = sum;
            }
        }
        value
    };
    match norm {
        MatrixNorm::Max => max_magnitude(a[..m * n].iter().copied()),
        MatrixNorm::One => largest(&mut (0..n).map(|j| asum(m, a, j * m, 1))),
        MatrixNorm::Infinity => {
            let mut work = vec![T::ZERO; m];
            for j in 0..n {
                for (i, sum) in work.iter_mut().enumerate() {
                    *sum += a[i + j * m].abs();
                }
            }
            largest(&mut work.into_iter())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column_major(rows: &[&[f64]]) -> Vec<f64> {
        let n = rows.len();
        let m = rows[0].len();
        let mut data = vec![0.0; n * m];
        for (i, row) in rows.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                data[i + j * n] = *value;
            }
        }
        data
    }

    #[test]
    fn lu_factors_solve_and_estimate_the_condition_number() {
        let a = column_major(&[&[2.0, 1.0, 1.0], &[4.0, -6.0, 0.0], &[-2.0, 7.0, 2.0]]);
        let mut lu = a.clone();
        let (pivots, info) = getf2(3, 3, &mut lu, 3);
        assert_eq!((pivots, info), (vec![1, 1, 2], 0));
        let mut b = vec![5.0, -2.0, 9.0];
        getrs(false, 3, 1, &lu, 3, &[1, 1, 2], &mut b, 3);
        assert_eq!(b, vec![1.0, 1.0, 2.0]);
        // ‖A‖₁ = 14 and ‖A⁻¹‖₁ = 2.25, which the estimator finds exactly here.
        let condition = gecon(true, 3, &lu, 3, 14.0);
        assert_eq!(condition.info, 0);
        assert!((condition.rcond - 1.0 / (14.0 * 2.25)).abs() < 1e-15);
    }

    #[test]
    fn lu_routines_round_as_scipy_does() {
        // scipy.linalg.lu_factor, lu_solve and inv of this matrix under SciPy 1.18.1's
        // OpenBLAS; the reference LAPACK order leaves U[2, 2] = -0.49999999999999967.
        let a = column_major(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);
        let mut lu = a.clone();
        let (pivots, info) = getf2(3, 3, &mut lu, 3);
        assert_eq!((pivots.clone(), info), (vec![2, 2, 2], 0));
        let expected = [
            7.0,
            0.14285714285714285,
            0.5714285714285714,
            8.0,
            0.8571428571428572,
            0.5000000000000002,
            10.0,
            1.5714285714285716,
            -0.5,
        ];
        assert_eq!(lu, expected);
        let mut b = vec![1.0, 2.0, 3.0];
        getrs(false, 3, 1, &lu, 3, &pivots, &mut b, 3);
        assert_eq!(b, [-0.3333333333333333, 0.6666666666666666, -0.0]);
        let mut b = vec![1.0, 2.0, 3.0];
        getrs(true, 3, 1, &lu, 3, &pivots, &mut b, 3);
        assert_eq!(b, [1.0, -0.0, 0.0]);
        assert_eq!(getri(3, &mut lu, 3, &pivots), 0);
        let expected = [
            -0.6666666666666662,
            -0.6666666666666676,
            1.0000000000000004,
            -1.333333333333333,
            3.6666666666666665,
            -2.0,
            0.9999999999999996,
            -1.9999999999999996,
            0.9999999999999999,
        ];
        assert_eq!(lu, expected);
    }

    #[test]
    fn cholesky_solves_round_as_scipy_does() {
        // lapack.dpotrf and dpotrs under SciPy 1.18.1's OpenBLAS; the reference order solves
        // the first unknown as exactly 0.0.
        let mut c = column_major(&[&[4.0, 1.0, 2.0], &[1.0, 5.0, 3.0], &[2.0, 3.0, 6.0]]);
        assert_eq!(potf2(true, 3, &mut c, 3), 0);
        let upper = [
            2.0,
            1.0,
            2.0,
            0.5,
            2.179449471770337,
            3.0,
            1.0,
            1.1470786693528088,
            1.9194297398747862,
        ];
        assert_eq!(c, upper);
        let mut b = vec![1.0, 2.0, 3.0, 0.0, 1.0, 0.0];
        potrs(true, 3, 2, &c, 3, &mut b, 3);
        let expected = [
            -2.0816681711721685e-17,
            0.14285714285714277,
            0.42857142857142866,
            0.0,
            0.2857142857142857,
            -0.14285714285714285,
        ];
        assert_eq!(b, expected);
    }

    #[test]
    fn singular_lu_reports_the_first_zero_pivot_and_continues() {
        let mut lu = column_major(&[&[1.0, 2.0], &[2.0, 4.0]]);
        let (pivots, info) = getf2(2, 2, &mut lu, 2);
        assert_eq!((pivots, info), (vec![1, 1], 2));
        assert_eq!(lu, vec![2.0, 0.5, 4.0, 0.0]);
    }

    #[test]
    fn bunch_kaufman_takes_a_two_by_two_pivot_for_an_indefinite_matrix() {
        let a = column_major(&[&[0.0, 1.0, 2.0], &[1.0, 0.0, 3.0], &[2.0, 3.0, 0.0]]);
        for upper in [true, false] {
            let mut factors = a.clone();
            let (pivots, info) = sytf2(upper, 3, &mut factors, 3);
            assert_eq!(info, 0);
            assert!(pivots.iter().any(|pivot| *pivot < 0), "{pivots:?}");
            let mut b = vec![3.0, 4.0, 5.0];
            sytrs(upper, 3, 1, &factors, 3, &pivots, &mut b, 3);
            // A x = b for x = (1, 1, 1) up to rounding.
            for value in b {
                assert!((value - 1.0).abs() < 1e-15, "{value}");
            }
        }
    }

    #[test]
    fn tridiagonal_and_band_solvers_agree_with_the_dense_solution() {
        // A = [[4, 1, 0], [1, 4, 1], [0, 1, 4]], b = A (1, 2, 3).
        let b = [6.0, 12.0, 14.0];
        let mut x = b.to_vec();
        let info = gtsv(
            &mut [1.0, 1.0],
            &mut [4.0, 4.0, 4.0],
            &mut [1.0, 1.0],
            1,
            &mut x,
            3,
        );
        assert_eq!(info, 0);
        assert!(x
            .iter()
            .zip([1.0, 2.0, 3.0])
            .all(|(a, b)| (a - b).abs() < 1e-15));
        let (lu, info) = gttrf(vec![1.0, 1.0], vec![4.0, 4.0, 4.0], vec![1.0, 1.0]);
        assert_eq!(info, 0);
        let mut y = b.to_vec();
        gttrs(false, &lu, 1, &mut y, 3);
        assert_eq!(x, y);
        // Band storage with kl = ku = 1 and ldab = 4.
        let mut ab = vec![
            0.0, 0.0, 4.0, 1.0, //
            0.0, 1.0, 4.0, 1.0, //
            0.0, 1.0, 4.0, 0.0,
        ];
        let (pivots, info) = gbtf2(3, 1, 1, &mut ab, 4);
        assert_eq!(info, 0);
        let mut z = b.to_vec();
        gbtrs(3, 1, 1, 1, &ab, 4, &pivots, &mut z, 3);
        assert!(z
            .iter()
            .zip([1.0, 2.0, 3.0])
            .all(|(a, b)| (a - b).abs() < 1e-15));
    }

    #[test]
    fn inverses_from_each_factorization_match() {
        let a = column_major(&[&[4.0, 2.0, 0.6], &[2.0, 5.0, 1.0], &[0.6, 1.0, 3.0]]);
        let mut general = a.clone();
        let (pivots, _) = getf2(3, 3, &mut general, 3);
        assert_eq!(getri(3, &mut general, 3, &pivots), 0);
        for upper in [true, false] {
            let mut cholesky = a.clone();
            assert_eq!(potf2(upper, 3, &mut cholesky, 3), 0);
            assert_eq!(potri(upper, 3, &mut cholesky, 3), 0);
            let mut symmetric = a.clone();
            let (pivots, _) = sytf2(upper, 3, &mut symmetric, 3);
            assert_eq!(sytri(upper, 3, &mut symmetric, 3, &pivots), 0);
            for j in 0..3 {
                for i in 0..3 {
                    let stored = if upper == (i <= j) {
                        i + j * 3
                    } else {
                        j + i * 3
                    };
                    assert!((cholesky[stored] - general[i + j * 3]).abs() < 1e-15);
                    assert!((symmetric[stored] - general[i + j * 3]).abs() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn triangular_condition_estimate_matches_the_exact_value_for_a_diagonal_matrix() {
        let a = column_major(&[&[2.0, 0.0], &[0.0, 0.5]]);
        // ‖A‖₁ = 2 and ‖A⁻¹‖₁ = 2.
        assert_eq!(trcon(true, false, 2, &a, 2), 0.25);
        let mut b = vec![1.0, 1.0];
        assert_eq!(trtrs(true, false, false, 2, 1, &a, 2, &mut b, 2), 0);
        assert_eq!(b, vec![0.5, 2.0]);
        let singular = column_major(&[&[1.0, 1.0], &[0.0, 0.0]]);
        assert_eq!(trtrs(true, false, false, 2, 1, &singular, 2, &mut b, 2), 2);
    }
}
