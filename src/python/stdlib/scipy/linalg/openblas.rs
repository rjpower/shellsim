//! OpenBLAS's BLAS kernels behind `scipy.linalg`, in the order SciPy's OpenBLAS evaluates them.
//!
//! SciPy's wheels bundle OpenBLAS, which selects kernels for the host CPU when it loads. These
//! ports follow the kernels that SciPy 1.18.1's OpenBLAS 0.3.31 selects on the reference machine
//! for shellsim's SciPy tests, an AMD Zen 2 CPU for which OpenBLAS picks its Haswell kernels:
//! AVX2 assembly whose fused multiply-adds are explicit, and C loops, some of which the compiler
//! fuses. Each routine's documentation gives the order and fusing, which were checked against
//! `scipy.linalg.blas` on that machine. Reproducing them makes small factorizations and solves
//! agree with SciPy's to the last bit; in the reference BLAS order,
//! `det([[1,2,3],[4,5,6],[7,8,10]])` is `-2.9999999999999982` where SciPy prints `-3.0`.
//!
//! OpenBLAS runs 16 threads on that machine and splits several right-hand sides among them, so a
//! lone column takes `trsm`'s single-column kernel; [`trsm_threaded`] splits them the same way.
//!
//! `trsv` and `trmv` agree with OpenBLAS at every order and `trsm` up to order 256. `float32`
//! input follows the double-precision kernels, although OpenBLAS's single-precision kernels block
//! by eight columns and 32 elements, so `float32` results can differ from SciPy's in the last bit
//! from about order 5.
//!
//! The level-1 and level-2 routines do work linear in their operand sizes and `trsm` quadratic
//! work per column; the LAPACK drivers that call them charge the totals up front.

use std::ops::Range;

use super::lapack::Real;

/// Rows and columns per block in OpenBLAS's triangular level-2 drivers (`DTB_ENTRIES`).
const TRIANGULAR_BLOCK: usize = 64;

/// Rows per block in the `dgemv_t` kernel (`NBMAX`).
const GEMV_ROW_BLOCK: usize = 2048;

/// `ddot` with unit strides (`ddot.c` with the Haswell microkernel): sixteen lanes of fused
/// multiply-adds over the first `n & -16` elements, reduced as the AVX2 registers are, then a
/// tail of separately rounded products added in order.
pub(super) fn dot<T: Real>(x: &[T], y: &[T]) -> T {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let blocked = n & !15;
    let mut sum = T::ZERO;
    if blocked > 0 {
        let mut lanes = [T::ZERO; 16];
        for (xs, ys) in x[..blocked]
            .chunks_exact(16)
            .zip(y[..blocked].chunks_exact(16))
        {
            for (lane, (x, y)) in lanes.iter_mut().zip(xs.iter().zip(ys)) {
                *lane = x.mul_add(*y, *lane);
            }
        }
        // Four registers of four lanes: fold each register's halves, add the registers
        // pairwise, then add the last two lanes.
        let fold = |r: usize| {
            [
                lanes[4 * r] + lanes[4 * r + 2],
                lanes[4 * r + 1] + lanes[4 * r + 3],
            ]
        };
        let (r0, r1, r2, r3) = (fold(0), fold(1), fold(2), fold(3));
        let low = [r0[0] + r1[0], r0[1] + r1[1]];
        let high = [r2[0] + r3[0], r2[1] + r3[1]];
        sum = (low[0] + high[0]) + (low[1] + high[1]);
    }
    for (x, y) in x[blocked..].iter().zip(&y[blocked..]) {
        sum += *y * *x;
    }
    sum
}

/// `ddot` of `n` elements where either stride is not one (`ddot.c`'s scalar loop): products in
/// groups of four, the first and third added into one sum and the second and fourth into
/// another, with the tail added into the first.
pub(super) fn dot_strided<T: Real>(n: usize, x: &[T], x_inc: usize, y: &[T], y_inc: usize) -> T {
    let (mut first, mut second) = (T::ZERO, T::ZERO);
    let product = |k: usize| y[k * y_inc] * x[k * x_inc];
    let grouped = n & !3;
    for k in (0..grouped).step_by(4) {
        let (m1, m2, m3, m4) = (product(k), product(k + 1), product(k + 2), product(k + 3));
        first += m1 + m3;
        second += m2 + m4;
    }
    for k in grouped..n {
        first += product(k);
    }
    first + second
}

/// `daxpy` with unit strides: `y += alpha x`, fused over the first `n & -16` elements by the
/// Haswell microkernel and rounded twice in the C tail.
pub(super) fn axpy<T: Real>(alpha: T, x: &[T], y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len());
    let blocked = x.len() & !15;
    for (y, x) in y[..blocked].iter_mut().zip(&x[..blocked]) {
        *y = alpha.mul_add(*x, *y);
    }
    for (y, x) in y[blocked..].iter_mut().zip(&x[blocked..]) {
        *y += alpha * *x;
    }
}

/// `dgemv_n` (`dgemv_n_4.c` with the Haswell microkernels): `y += alpha A x` for the `m × n`
/// column-major matrix `a` with leading dimension `lda`, `x` with stride `x_inc`, and a
/// contiguous `y` of length `m`.
///
/// Rows in multiples of four go through the vector kernels one block of columns at a time:
/// four columns as `fma(alpha, fma(a2, x2, a0 x0) + fma(a3, x3, a1 x1), y)`; with a contiguous
/// `x`, then two columns as `fma(alpha, a0 x0 + a1 x1, y)` and a last column as
/// `y + a (alpha x)`, while a strided `x` takes every remaining column singly. The last `m % 4`
/// rows sum `a x` in column order and add `alpha` times the sum. OpenBLAS unrolls those last rows
/// in pairs of columns when `lda == m % 4`; no caller here reaches that case with four or more
/// columns, where the two orders differ.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemv_n<T: Real>(
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    x_inc: usize,
    y: &mut [T],
) {
    if m == 0 || n == 0 {
        return;
    }
    let vector_rows = m & !3;
    let x_at = |k: usize| x[k * x_inc];
    let at = |i: usize, k: usize| a[i + k * lda];
    let blocks = n / 4;
    for k in (0..4 * blocks).step_by(4) {
        let (x0, x1, x2, x3) = (x_at(k), x_at(k + 1), x_at(k + 2), x_at(k + 3));
        for (i, y) in y[..vector_rows].iter_mut().enumerate() {
            let even = at(i, k + 2).mul_add(x2, at(i, k) * x0);
            let odd = at(i, k + 3).mul_add(x3, at(i, k + 1) * x1);
            *y = alpha.mul_add(even + odd, *y);
        }
    }
    let mut k = 4 * blocks;
    if x_inc == 1 && n & 2 != 0 {
        let (x0, x1) = (x_at(k), x_at(k + 1));
        for (i, y) in y[..vector_rows].iter_mut().enumerate() {
            *y = alpha.mul_add(at(i, k) * x0 + at(i, k + 1) * x1, *y);
        }
        k += 2;
    }
    for k in k..n {
        let scaled = alpha * x_at(k);
        for (i, y) in y[..vector_rows].iter_mut().enumerate() {
            *y += at(i, k) * scaled;
        }
    }
    for (i, y) in y[..m].iter_mut().enumerate().skip(vector_rows) {
        let mut sum = T::ZERO;
        for k in 0..n {
            sum += at(i, k) * x_at(k);
        }
        *y += alpha * sum;
    }
}

/// `dgemv_t` (`dgemv_t_4.c` with the Haswell microkernel): `y += alpha Aᵀ x` for the `m × n`
/// column-major matrix `a` with leading dimension `lda`, a contiguous `x` of length `m`, and a
/// contiguous `y` of length `n`.
///
/// Rows in multiples of four are summed in blocks of up to 2048, and each column's block sum
/// times `alpha` is added to `y`. Columns in groups of four sum through four fused lanes, each
/// taking every fourth row, folded as `(l0 + l2) + (l1 + l3)`; a remaining pair of columns sums
/// separately rounded products in two lanes of alternate rows; a last single column sums them in
/// four lanes folded like the first kernel. The last `m % 4` rows then add
/// `a0 (alpha x0) + a1 (alpha x1) + a2 (alpha x2)` to each element of `y`, fused as
/// `y + fma(a2, x2', fma(a0, x0', a1 x1'))`, or `fma(a0, x0', y)` for a single row.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemv_t<T: Real>(
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    y: &mut [T],
) {
    if m == 0 || n == 0 {
        return;
    }
    let vector_rows = m & !3;
    let at = |i: usize, j: usize| a[i + j * lda];
    let fold = |lanes: [T; 4]| (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
    let grouped = n & !3;
    for start in (0..vector_rows).step_by(GEMV_ROW_BLOCK) {
        let rows = start..vector_rows.min(start + GEMV_ROW_BLOCK);
        for (j, y) in y[..grouped].iter_mut().enumerate() {
            let mut lanes = [T::ZERO; 4];
            for i in rows.clone() {
                lanes[i % 4] = at(i, j).mul_add(x[i], lanes[i % 4]);
            }
            *y += fold(lanes) * alpha;
        }
        let mut j = grouped;
        if n & 2 != 0 {
            for (column, y) in y[j..j + 2].iter_mut().enumerate() {
                let mut lanes = [T::ZERO; 2];
                for i in rows.clone() {
                    lanes[i % 2] += at(i, j + column) * x[i];
                }
                *y += (lanes[0] + lanes[1]) * alpha;
            }
            j += 2;
        }
        if n & 1 != 0 {
            let mut lanes = [T::ZERO; 4];
            for i in rows.clone() {
                lanes[i % 4] += at(i, j) * x[i];
            }
            y[j] += fold(lanes) * alpha;
        }
    }
    // OpenBLAS compiles this C tail with contraction, so GCC fuses its multiply-adds.
    let scaled: Vec<T> = x[vector_rows..m].iter().map(|x| *x * alpha).collect();
    let r = vector_rows;
    for (j, y) in y[..n].iter_mut().enumerate() {
        *y = match scaled[..] {
            [] => *y,
            [x0] => at(r, j).mul_add(x0, *y),
            [x0, x1] => *y + at(r, j).mul_add(x0, at(r + 1, j) * x1),
            [x0, x1, x2] => *y + at(r + 2, j).mul_add(x2, at(r, j).mul_add(x0, at(r + 1, j) * x1)),
            _ => unreachable!("fewer than four tail rows"),
        };
    }
}

/// `dtrsv` with a unit stride (OpenBLAS's `trsv_L.c` and `trsv_U.c`): overwrite `x` with
/// `op(A)⁻¹ x` for the `n × n` triangular `a`, reading only its `upper` or lower triangle.
///
/// The solve runs in blocks of 64: within a block each solved element updates the rest of the
/// block with `axpy`, or the transposed forms subtract a `dot` before dividing, and a `gemv`
/// carries each block's result to the unsolved rows.
#[allow(clippy::too_many_arguments)]
pub(super) fn trsv<T: Real>(
    upper: bool,
    transpose: bool,
    unit: bool,
    n: usize,
    a: &[T],
    lda: usize,
    x: &mut [T],
) {
    let diagonal = |i: usize| a[i + i * lda];
    // Forward substitution: lower without transpose, or upper transposed, from the top.
    if upper == transpose {
        for start in (0..n).step_by(TRIANGULAR_BLOCK) {
            let end = n.min(start + TRIANGULAR_BLOCK);
            if transpose && start > 0 {
                let (solved, rest) = x.split_at_mut(start);
                gemv_t(
                    start,
                    end - start,
                    -T::ONE,
                    &a[start * lda..],
                    lda,
                    solved,
                    &mut rest[..end - start],
                );
            }
            for i in start..end {
                if transpose && i > start {
                    let column = &a[start + i * lda..i + i * lda];
                    x[i] -= dot(column, &x[start..i]);
                }
                if !unit {
                    x[i] /= diagonal(i);
                }
                if !transpose && i + 1 < end {
                    let (solved, rest) = x.split_at_mut(i + 1);
                    axpy(
                        -solved[i],
                        &a[i + 1 + i * lda..end + i * lda],
                        &mut rest[..end - i - 1],
                    );
                }
            }
            if !transpose && end < n {
                let (solved, rest) = x.split_at_mut(end);
                gemv_n(
                    n - end,
                    end - start,
                    -T::ONE,
                    &a[end + start * lda..],
                    lda,
                    &solved[start..],
                    1,
                    rest,
                );
            }
        }
        return;
    }
    // Back substitution: upper without transpose, or lower transposed, from the bottom.
    let mut end = n;
    while end > 0 {
        let start = end.saturating_sub(TRIANGULAR_BLOCK);
        if transpose && end < n {
            let (unsolved, solved) = x.split_at_mut(end);
            gemv_t(
                n - end,
                end - start,
                -T::ONE,
                &a[end + start * lda..],
                lda,
                solved,
                &mut unsolved[start..],
            );
        }
        for i in (start..end).rev() {
            if transpose && i + 1 < end {
                let column = &a[i + 1 + i * lda..end + i * lda];
                x[i] -= dot(column, &x[i + 1..end]);
            }
            if !unit {
                x[i] /= diagonal(i);
            }
            if !transpose && i > start {
                let (unsolved, solved) = x.split_at_mut(i);
                axpy(
                    -solved[0],
                    &a[start + i * lda..i + i * lda],
                    &mut unsolved[start..],
                );
            }
        }
        if !transpose && start > 0 {
            let (unsolved, solved) = x.split_at_mut(start);
            gemv_n(
                start,
                end - start,
                -T::ONE,
                &a[start * lda..],
                lda,
                &solved[..end - start],
                1,
                unsolved,
            );
        }
        end = start;
    }
}

/// `dtrmv` without transpose and with a unit stride (OpenBLAS's `trmv_U.c` and `trmv_L.c`):
/// overwrite `x` with `A x` for the `n × n` triangular `a`, reading only its `upper` or lower
/// triangle. Each block of 64 first receives the `gemv` contribution of the blocks already
/// finished; within a block each element adds its multiple of its column with `axpy`, then is
/// scaled by the diagonal.
pub(super) fn trmv<T: Real>(upper: bool, unit: bool, n: usize, a: &[T], lda: usize, x: &mut [T]) {
    if upper {
        for start in (0..n).step_by(TRIANGULAR_BLOCK) {
            let end = n.min(start + TRIANGULAR_BLOCK);
            if start > 0 {
                let (done, block) = x.split_at_mut(start);
                gemv_n(
                    start,
                    end - start,
                    T::ONE,
                    &a[start * lda..],
                    lda,
                    &block[..end - start],
                    1,
                    done,
                );
            }
            for i in start..end {
                if i > start {
                    let (above, rest) = x.split_at_mut(i);
                    axpy(
                        rest[0],
                        &a[start + i * lda..i + i * lda],
                        &mut above[start..],
                    );
                }
                if !unit {
                    x[i] *= a[i + i * lda];
                }
            }
        }
        return;
    }
    let mut end = n;
    while end > 0 {
        let start = end.saturating_sub(TRIANGULAR_BLOCK);
        if end < n {
            let (block, done) = x.split_at_mut(end);
            gemv_n(
                n - end,
                end - start,
                T::ONE,
                &a[end + start * lda..],
                lda,
                &block[start..],
                1,
                done,
            );
        }
        for i in (start..end).rev() {
            if i + 1 < end {
                let (head, below) = x.split_at_mut(i + 1);
                axpy(
                    head[i],
                    &a[i + 1 + i * lda..end + i * lda],
                    &mut below[..end - i - 1],
                );
            }
            if !unit {
                x[i] *= a[i + i * lda];
            }
        }
        end = start;
    }
}

/// `dtrsm` with `side = 'L'` and `alpha = 1` (OpenBLAS's `trsm_L.c` driver with the generic
/// `trsm_kernel_LT.c` and `trsm_kernel_LN.c` kernels and the Haswell `dgemm` micro-kernels):
/// overwrite the `m × n` column-major matrix `b` with `op(A)⁻¹ b` for the triangular `a`, reading
/// only its `upper` or lower triangle.
///
/// Rows are solved in blocks of four with a trailing block of two and of one: forward from the
/// top when `op(A)` is lower triangular, or from the bottom, remainder rows first, when it is
/// upper triangular. Each block first subtracts the product of its rows of `op(A)` with the rows
/// already solved, as a `dgemm` micro-kernel accumulates it, then solves within the block,
/// multiplying by the reciprocal of each diagonal element. The product is one chain of fused
/// multiply-adds per element, except in the four-row kernel for a lone last column, which splits
/// the chain into four lanes by `k mod 4` over blocks of eight `k`. OpenBLAS blocks columns in
/// eights, then a four, a two and a one, so a lone column is the last of an odd number.
///
/// This agrees with OpenBLAS up to order 256, beyond which OpenBLAS also blocks the rows already
/// solved (`GEMM_Q`).
#[allow(clippy::too_many_arguments)]
pub(super) fn trsm<T: Real>(
    upper: bool,
    transpose: bool,
    unit: bool,
    m: usize,
    n: usize,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) {
    let forward = upper == transpose;
    let op = |r: usize, k: usize| {
        if transpose {
            a[k + r * lda]
        } else {
            a[r + k * lda]
        }
    };
    let reciprocal: Vec<T> = (0..m)
        .map(|i| {
            if unit {
                T::ONE
            } else {
                T::ONE / a[i + i * lda]
            }
        })
        .collect();
    let mut blocks = Vec::new();
    if forward {
        let mut start = 0;
        while start + 4 <= m {
            blocks.push(start..start + 4);
            start += 4;
        }
        for size in [2, 1] {
            if m & size != 0 {
                blocks.push(start..start + size);
                start += size;
            }
        }
    } else {
        for size in [1, 2] {
            if m & size != 0 {
                let start = (m & !(size - 1)) - size;
                blocks.push(start..start + size);
            }
        }
        for start in (0..m & !3).step_by(4).rev() {
            blocks.push(start..start + 4);
        }
    }
    for j in 0..n {
        let lone = n % 2 == 1 && j + 1 == n;
        let column = &mut b[j * ldb..j * ldb + m];
        for block in &blocks {
            let solved = if forward {
                0..block.start
            } else {
                block.end..m
            };
            for r in block.clone().filter(|_| !solved.is_empty()) {
                let sum = if lone && block.len() == 4 {
                    let mut lanes = [T::ZERO; 4];
                    let unrolled = solved.len() / 8 * 8;
                    for (index, k) in solved.clone().enumerate() {
                        let lane = if index < unrolled { index % 4 } else { 0 };
                        lanes[lane] = op(r, k).mul_add(column[k], lanes[lane]);
                    }
                    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
                } else {
                    solved
                        .clone()
                        .fold(T::ZERO, |sum, k| op(r, k).mul_add(column[k], sum))
                };
                column[r] += -sum;
            }
            if forward {
                for i in block.clone() {
                    column[i] *= reciprocal[i];
                    for k in i + 1..block.end {
                        column[k] -= column[i] * op(k, i);
                    }
                }
            } else {
                for i in block.clone().rev() {
                    column[i] *= reciprocal[i];
                    for k in block.start..i {
                        column[k] -= column[i] * op(k, i);
                    }
                }
            }
        }
    }
}

/// Threads SciPy's OpenBLAS starts on the reference machine, one per logical CPU of its
/// eight-core Ryzen 7 3800X.
const THREADS: usize = 16;

/// `m × n` sizes from which OpenBLAS's `dtrsm` interface threads (`SMP_FACTOR` times
/// `GEMM_MULTITHREAD_THRESHOLD`).
const TRSM_THREAD_SIZE: usize = 256 * 4;

/// The columns of each thread when OpenBLAS's `gemm_thread_n` splits `n` columns: each thread
/// takes the remaining columns divided by the remaining threads, rounded up, so up to 16 columns
/// go one to a thread.
fn thread_columns(n: usize) -> impl Iterator<Item = Range<usize>> {
    let mut start = 0;
    (0..THREADS).map_while(move |thread| {
        let width = (n - start).div_ceil(THREADS - thread);
        let columns = start..start + width;
        start += width;
        (width > 0).then_some(columns)
    })
}

/// [`trsm`] as OpenBLAS's threaded drivers run it: each thread of [`thread_columns`] solves its
/// columns with its own `trsm`, which matters because a lone column takes a different kernel.
///
/// OpenBLAS's `getrs` and `trtrs` take this path for any number of right-hand sides above one;
/// its `dtrsm` interface takes it from [`TRSM_THREAD_SIZE`] elements (see [`trsm_interface`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn trsm_threaded<T: Real>(
    upper: bool,
    transpose: bool,
    unit: bool,
    m: usize,
    n: usize,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) {
    for columns in thread_columns(n) {
        let width = columns.len();
        let b = &mut b[columns.start * ldb..];
        trsm(upper, transpose, unit, m, width, a, lda, b, ldb);
    }
}

/// [`trsm`] as OpenBLAS's `dtrsm` interface runs it, which reference LAPACK routines such as
/// `dpotrs` call: single-threaded below [`TRSM_THREAD_SIZE`] elements of `b`, and split by
/// column as [`trsm_threaded`] describes from there.
#[allow(clippy::too_many_arguments)]
pub(super) fn trsm_interface<T: Real>(
    upper: bool,
    transpose: bool,
    unit: bool,
    m: usize,
    n: usize,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) {
    if m.saturating_mul(n) < TRSM_THREAD_SIZE {
        trsm(upper, transpose, unit, m, n, a, lda, b, ldb);
    } else {
        trsm_threaded(upper, transpose, unit, m, n, a, lda, b, ldb);
    }
}

#[cfg(test)]
mod tests {
    //! Expected values come from SciPy 1.18.1's OpenBLAS 0.3.31 on the reference machine, through
    //! `scipy.linalg.blas`; each input is chosen so that the reference BLAS order gives a
    //! different result.

    use super::*;

    /// `[((i * a) % b) / c - d for i in range(n)]`, which rounds identically in Python and Rust.
    fn generate(n: usize, a: usize, b: usize, c: f64, d: f64) -> Vec<f64> {
        (0..n).map(|i| ((i * a) % b) as f64 / c - d).collect()
    }

    fn in_order_dot(x: impl Iterator<Item = f64>, y: impl Iterator<Item = f64>) -> f64 {
        x.zip(y).fold(0.0, |sum, (x, y)| sum + x * y)
    }

    #[test]
    fn contiguous_dot_fuses_sixteen_lanes() {
        let mut x: Vec<f64> = generate(37, 7, 101, 7.0, 7.0)
            .iter()
            .map(|x| x * 1e3)
            .collect();
        x[0] = 1e17 / 3.0;
        let mut y = generate(37, 53, 97, 13.0, 3.0);
        y[0] = 1.0 / 3.0;
        // blas.ddot(x, y)
        assert_eq!(dot(&x, &y), 1.1111111111063726e16);
        assert_eq!(
            in_order_dot(x.into_iter(), y.into_iter()),
            1.1111111111063732e16
        );
    }

    #[test]
    fn strided_dot_pairs_alternate_products() {
        let mut x: Vec<f64> = generate(37, 3, 101, 7.0, 7.0)
            .iter()
            .map(|x| x * 1e3)
            .collect();
        x[0] = 1e17 / 3.0;
        let y = generate(37, 53, 97, 13.0, 3.0);
        // blas.ddot(x, y, n=18, incx=2, incy=2)
        assert_eq!(dot_strided(18, &x, 2, &y, 2), -9.999999999999867e16);
        let every_other = |v: &[f64]| v.iter().step_by(2).take(18).copied().collect::<Vec<_>>();
        let in_order = in_order_dot(every_other(&x).into_iter(), every_other(&y).into_iter());
        assert_eq!(in_order, -9.999999999999866e16);
    }

    #[test]
    fn axpy_fuses_blocks_of_sixteen() {
        let x = generate(37, 37, 101, 7.0, 7.0);
        let mut y = generate(37, 53, 97, 13.0, 3.0);
        axpy(-0.7, &x, &mut y);
        // blas.daxpy(x, y, a=-0.7); a separately rounded product gives 1.8999999999999995,
        // -4.807692307692307 and -1.0384615384615383 in the first three places.
        assert_eq!(y[0], 1.8999999999999997);
        assert_eq!(y[2], -4.8076923076923075);
        assert_eq!(y[5], -1.038461538461538);
        assert_eq!(y[36], 5.0);
    }

    /// The 7 × 7 column-major matrix `((i * 7 + j * 3) % 19) / 11 - 0.8`.
    fn gemv_matrix() -> Vec<f64> {
        let mut a = vec![0.0; 49];
        for j in 0..7 {
            for i in 0..7 {
                a[i + j * 7] = ((i * 7 + j * 3) % 19) as f64 / 11.0 - 0.8;
            }
        }
        a
    }

    #[test]
    fn gemv_n_follows_the_vector_and_tail_kernels() {
        let a = gemv_matrix();
        let x = generate(7, 29, 31, 9.0, 1.5);
        let mut y = vec![1.0 / 3.0; 7];
        gemv_n(7, 7, -1.0, &a, 7, &x, 1, &mut y);
        // blas.dgemv(-1.0, a, x, beta=1.0, y=y)
        let expected = [
            -0.8878787878787877,
            -0.06969696969696965,
            1.2282828282828282,
            -0.8323232323232317,
            -1.0696969696969694,
            3.2030303030303036,
            -0.3929292929292923,
        ];
        assert_eq!(y, expected);
    }

    #[test]
    fn gemv_n_with_a_strided_x_takes_leftover_columns_singly() {
        let a = gemv_matrix();
        let x = generate(14, 29, 31, 9.0, 1.5);
        let mut y = vec![1.0 / 3.0; 7];
        gemv_n(7, 7, -1.0, &a, 7, &x, 2, &mut y);
        // blas.dgemv(-1.0, a, x, beta=1.0, y=y, incx=2)
        let expected = [
            0.8939393939393938,
            -1.0757575757575757,
            1.2727272727272725,
            -0.505050505050505,
            -1.227272727272727,
            3.7121212121212124,
            -1.1363636363636365,
        ];
        assert_eq!(y, expected);
    }

    /// An order-70 matrix, which crosses the 64-element block boundary of every driver.
    fn diagonally_dominant(n: usize) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                a[i + j * n] = if i == j {
                    2.0
                } else {
                    ((i * 3 + j * 5) % 7) as f64 / 64.0
                };
            }
        }
        a
    }

    #[test]
    fn trsv_undoes_trmv_across_blocks() {
        let n = 70;
        let a = diagonally_dominant(n);
        let original: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
        for upper in [false, true] {
            for unit in [false, true] {
                let mut x = original.clone();
                trmv(upper, unit, n, &a, n, &mut x);
                trsv(upper, false, unit, n, &a, n, &mut x);
                for (x, expected) in x.iter().zip(&original) {
                    assert!((x - expected).abs() < 1e-12, "upper {upper}, unit {unit}");
                }
            }
        }
    }

    #[test]
    fn transposed_trsv_solves_the_transposed_system() {
        let n = 70;
        let a = diagonally_dominant(n);
        let b: Vec<f64> = (0..n).map(|i| (i % 3) as f64 + 1.0).collect();
        for upper in [false, true] {
            let mut x = b.clone();
            trsv(upper, true, false, n, &a, n, &mut x);
            for (j, b) in b.iter().enumerate() {
                let rows = if upper { 0..j + 1 } else { j..n };
                let value: f64 = rows.map(|i| a[i + j * n] * x[i]).sum();
                assert!((value - b).abs() < 1e-12, "upper {upper}");
            }
        }
    }

    #[test]
    fn threads_split_columns_as_gemm_thread_n_does() {
        let widths = |n| thread_columns(n).map(|c| c.len()).collect::<Vec<_>>();
        assert_eq!(widths(0), Vec::<usize>::new());
        assert_eq!(widths(3), [1, 1, 1]);
        assert_eq!(widths(16), [1; 16]);
        assert_eq!(widths(20), [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(thread_columns(20).last(), Some(19..20));
    }
}
