//! The per-slice loops of SciPy 1.18's C++ `_batched_linalg` module: structure detection,
//! the LAPACK sequence each structure runs, and the status SciPy reports for each slice.
//!
//! Every function takes a stack of `count` C-order matrices and returns C-order results. As in
//! SciPy, each slice is copied to column-major storage before LAPACK sees it, and some state
//! persists from slice to slice: the triangle `solve` and `inv` read (`uplo`), which a detected
//! triangular slice changes for every later slice.
//!
//! SciPy's quirks are kept because they are visible in warnings and results: the general
//! solver passes the infinity norm where `dgecon` expects the 1-norm, the tridiagonal norm
//! omits the last superdiagonal element, and a diagonal slice reports its condition number,
//! not its reciprocal, as `rcond`.

use super::lapack::{self, Real};

/// SciPy's matrix structure codes, shared with the Python layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Structure {
    Detect,
    General,
    Diagonal,
    UpperTriangular,
    LowerTriangular,
    Tridiagonal,
    Banded,
    PositiveDefinite,
    Symmetric,
    Hermitian,
}

impl Structure {
    pub(super) fn from_code(code: i64) -> Option<Self> {
        Some(match code {
            -1 => Self::Detect,
            0 => Self::General,
            11 => Self::Diagonal,
            21 => Self::UpperTriangular,
            22 => Self::LowerTriangular,
            31 => Self::Tridiagonal,
            41 => Self::Banded,
            101 => Self::PositiveDefinite,
            201 => Self::Symmetric,
            211 => Self::Hermitian,
            _ => return None,
        })
    }

    pub(super) fn code(self) -> i64 {
        match self {
            Self::Detect => -1,
            Self::General => 0,
            Self::Diagonal => 11,
            Self::UpperTriangular => 21,
            Self::LowerTriangular => 22,
            Self::Tridiagonal => 31,
            Self::Banded => 41,
            Self::PositiveDefinite => 101,
            Self::Symmetric => 201,
            Self::Hermitian => 211,
        }
    }
}

/// What SciPy records about a slice that is singular, ill-conditioned, or rejected by LAPACK.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct SliceStatus {
    pub num: usize,
    pub structure: Structure,
    pub is_singular: bool,
    pub is_ill_conditioned: bool,
    pub rcond: f64,
    pub lapack_info: i64,
}

impl SliceStatus {
    fn new(num: usize, structure: Structure) -> Self {
        Self {
            num,
            structure,
            is_singular: false,
            is_ill_conditioned: false,
            rcond: 0.0,
            lapack_info: 0,
        }
    }

    /// SciPy's test: NaN or below the machine epsilon of the working precision.
    fn record_rcond<T: Real>(&mut self, rcond: T) {
        self.rcond = rcond.to_f64();
        self.is_ill_conditioned = rcond.is_nan() || rcond < T::PRECISION;
    }
}

/// `_detect_problems`: record a problem slice. Returns true when the loop must stop.
fn detect_problems(status: &SliceStatus, statuses: &mut Vec<SliceStatus>) -> bool {
    if status.lapack_info < 0 || status.is_singular {
        statuses.push(status.clone());
        return true;
    }
    if status.is_ill_conditioned {
        statuses.push(status.clone());
    }
    false
}

/// The `rows × columns` C-order matrix `a` in column-major order.
pub(super) fn to_column_major<T: Real>(a: &[T], rows: usize, columns: usize) -> Vec<T> {
    let mut result = Vec::with_capacity(rows * columns);
    for j in 0..columns {
        for i in 0..rows {
            result.push(a[i * columns + j]);
        }
    }
    result
}

/// The `rows × columns` column-major matrix `a` in C order.
pub(super) fn to_row_major<T: Real>(a: &[T], rows: usize, columns: usize) -> Vec<T> {
    let mut result = Vec::with_capacity(rows * columns);
    for i in 0..rows {
        for j in 0..columns {
            result.push(a[i + j * rows]);
        }
    }
    result
}

/// SciPy's `bandwidth` for an `n × m` matrix whose element `(r, c)` is nonzero when
/// `nonzero(r, c)`: the lower and upper bandwidths.
pub(super) fn bandwidth(
    n: usize,
    m: usize,
    nonzero: impl Fn(usize, usize) -> bool,
) -> (usize, usize) {
    let (mut lower, mut upper) = (0, 0);
    for c in 0..m.saturating_sub(1) {
        if let Some(r) = (c + lower + 1..n).rev().find(|&r| nonzero(r, c)) {
            lower = r - c;
        }
        if c + lower + 1 > m {
            break;
        }
    }
    for c in (1..m).rev() {
        if c > upper {
            if let Some(r) = (0..c - upper).find(|&r| nonzero(r, c)) {
                upper = c - r;
            }
        }
        if c <= upper {
            break;
        }
    }
    (lower, upper)
}

/// `is_sym_or_herm` for real data: whether the column-major matrix equals its transpose.
fn is_symmetric<T: Real>(data: &[T], n: usize) -> bool {
    (0..n).all(|i| (0..n).all(|j| data[i * n + j] == data[i + j * n]))
}

/// SciPy's `norm1_` on column-major data, which sums rows: the infinity norm.
fn row_sum_norm<T: Real>(data: &[T], n: usize) -> T {
    let mut sums = data[..n]
        .iter()
        .map(|value| value.abs())
        .collect::<Vec<_>>();
    for i in 1..n {
        for (j, sum) in sums.iter_mut().enumerate() {
            *sum += data[i * n + j].abs();
        }
    }
    largest(&sums)
}

/// SciPy's `norm1_sym_herm`: the 1-norm of the symmetric matrix stored in one triangle.
fn symmetric_norm<T: Real>(lower: bool, data: &[T], n: usize) -> T {
    let mut sums;
    if lower {
        // `norm1_sym_herm_upper`, reading column-major data transposed.
        sums = data[..n]
            .iter()
            .map(|value| value.abs())
            .collect::<Vec<_>>();
        for i in 1..n {
            sums[i] += data[i * n + i].abs();
            for j in i + 1..n {
                let temp = data[i * n + j].abs();
                sums[j] += temp;
                sums[i] += temp;
            }
        }
    } else {
        sums = vec![T::ZERO; n];
        for i in 0..n {
            sums[i] += data[i * n + i].abs();
            for j in 0..i {
                let temp = data[i * n + j].abs();
                sums[j] += temp;
                sums[i] += temp;
            }
        }
    }
    largest(&sums)
}

/// The largest of `values`, starting from zero, compared as SciPy's loops compare.
fn largest<T: Real>(values: &[T]) -> T {
    let mut result = T::ZERO;
    for value in values {
        if *value > result {
            result = *value;
        }
    }
    result
}

/// Copy one triangle of the column-major matrix onto the other.
fn fill_other_triangle<T: Real>(lower: bool, data: &mut [T], n: usize) {
    for j in 0..n {
        for i in j + 1..n {
            if lower {
                data[j + i * n] = data[i + j * n];
            } else {
                data[i + j * n] = data[j + i * n];
            }
        }
    }
}

/// Zero the triangle of the column-major matrix opposite the one `lower` names.
fn zero_other_triangle<T: Real>(lower: bool, data: &mut [T], n: usize) {
    for j in 0..n {
        for i in j + 1..n {
            if lower {
                data[j + i * n] = T::ZERO;
            } else {
                data[i + j * n] = T::ZERO;
            }
        }
    }
}

/// The structure SciPy detects for a column-major slice, and the triangle it will read.
fn detect<T: Real>(data: &[T], n: usize, tridiagonal: bool, lower: &mut bool) -> Structure {
    let (lower_band, upper_band) = bandwidth(n, n, |r, c| data[c * n + r] != T::ZERO);
    if lower_band == 0 && upper_band == 0 {
        Structure::Diagonal
    } else if tridiagonal && lower_band == 1 && upper_band == 1 && n > 3 {
        Structure::Tridiagonal
    } else if lower_band == 0 {
        *lower = false;
        Structure::UpperTriangular
    } else if upper_band == 0 {
        *lower = true;
        Structure::LowerTriangular
    } else if is_symmetric(data, n) {
        // Real symmetric matrices try Cholesky first and fall back to Bunch-Kaufman.
        Structure::PositiveDefinite
    } else {
        Structure::General
    }
}

/// The per-slice loop state `solve` and `inv` share.
struct Loop {
    structure: Structure,
    lower: bool,
    posdef_fallback: bool,
}

impl Loop {
    fn new(structure: Structure, lower: bool) -> Self {
        let lower = match structure {
            Structure::LowerTriangular => true,
            Structure::UpperTriangular => false,
            _ => lower,
        };
        Self {
            structure,
            lower,
            posdef_fallback: structure != Structure::PositiveDefinite,
        }
    }
}

/// `x = A⁻¹ b`, or `A⁻ᵀ b` when `transposed`, for `count` slices of `n × n` matrices and
/// `n × nrhs` right-hand sides. Banded structure is not supported here; callers reject it.
#[allow(clippy::too_many_arguments)]
pub(super) fn solve<T: Real>(
    a: &[T],
    b: &[T],
    count: usize,
    n: usize,
    nrhs: usize,
    structure: Structure,
    lower: bool,
    transposed: bool,
) -> (Vec<T>, Vec<SliceStatus>) {
    let mut state = Loop::new(structure, lower);
    let mut statuses = Vec::new();
    let mut x = vec![T::ZERO; count * n * nrhs];
    for index in 0..count {
        let original = to_column_major(&a[index * n * n..(index + 1) * n * n], n, n);
        let mut data = original.clone();
        let mut rhs = to_column_major(&b[index * n * nrhs..(index + 1) * n * nrhs], n, nrhs);
        let mut slice_structure = state.structure;
        if slice_structure == Structure::Detect {
            slice_structure = detect(&data, n, true, &mut state.lower);
        }
        let mut status = SliceStatus::new(index, slice_structure);
        let upper = !state.lower;
        let stop = match slice_structure {
            Structure::Diagonal => {
                solve_diagonal(&data, n, nrhs, &mut rhs, &mut status);
                detect_problems(&status, &mut statuses)
            }
            Structure::Tridiagonal => {
                solve_tridiagonal(&data, n, nrhs, &mut rhs, transposed, &mut status);
                detect_problems(&status, &mut statuses)
            }
            Structure::UpperTriangular | Structure::LowerTriangular => {
                let info = lapack::trtrs(upper, transposed, false, n, nrhs, &data, n, &mut rhs, n);
                status.lapack_info = info as i64;
                status.is_singular = info > 0;
                status.record_rcond(lapack::trcon(upper, false, n, &data, n));
                detect_problems(&status, &mut statuses)
            }
            Structure::PositiveDefinite | Structure::Symmetric | Structure::Hermitian => {
                let mut use_symmetric = slice_structure != Structure::PositiveDefinite;
                if !use_symmetric {
                    solve_cholesky(upper, &mut data, n, nrhs, &mut rhs, &mut status);
                    if status.lapack_info == 0 || !status.is_singular {
                        if status.is_ill_conditioned {
                            statuses.push(status.clone());
                        }
                    } else if state.posdef_fallback {
                        data = original;
                        status = SliceStatus::new(index, slice_structure);
                        use_symmetric = true;
                    } else {
                        statuses.push(status.clone());
                    }
                }
                if use_symmetric {
                    solve_symmetric(upper, &mut data, n, nrhs, &mut rhs, &mut status);
                    detect_problems(&status, &mut statuses)
                } else {
                    false
                }
            }
            Structure::General | Structure::Detect | Structure::Banded => {
                solve_general(&mut data, n, nrhs, &mut rhs, transposed, &mut status);
                detect_problems(&status, &mut statuses)
            }
        };
        if stop {
            break;
        }
        x[index * n * nrhs..(index + 1) * n * nrhs].copy_from_slice(&to_row_major(&rhs, n, nrhs));
    }
    (x, statuses)
}

fn solve_diagonal<T: Real>(
    data: &[T],
    n: usize,
    nrhs: usize,
    b: &mut [T],
    status: &mut SliceStatus,
) {
    let (mut largest_value, mut largest_inverse) = (T::ZERO, T::ZERO);
    for j in 0..n {
        let ajj = data[j * n + j];
        status.is_singular = ajj == T::ZERO;
        if status.is_singular {
            status.lapack_info = j as i64;
            return;
        }
        let inverse = T::ONE / ajj;
        for i in 0..nrhs {
            b[j + i * n] *= inverse;
        }
        if ajj.abs() > largest_value {
            largest_value = ajj.abs();
        }
        if inverse.abs() > largest_inverse {
            largest_inverse = inverse.abs();
        }
    }
    let condition = largest_value * largest_inverse;
    status.is_ill_conditioned = condition > T::ONE / T::PRECISION;
    status.rcond = condition.to_f64();
}

fn solve_tridiagonal<T: Real>(
    data: &[T],
    n: usize,
    nrhs: usize,
    b: &mut [T],
    transposed: bool,
    status: &mut SliceStatus,
) {
    let d = (0..n).map(|i| data[i + i * n]).collect::<Vec<_>>();
    let dl = (0..n - 1).map(|i| data[i * n + i + 1]).collect::<Vec<_>>();
    let du = (0..n - 1)
        .map(|i| data[(i + 1) * n + i])
        .collect::<Vec<_>>();
    // SciPy's `norm1_tridiag` leaves out the last superdiagonal element.
    let mut sums = d.iter().map(|value| value.abs()).collect::<Vec<_>>();
    for i in 0..n - 1 {
        sums[i] += dl[i].abs();
    }
    for i in 1..n - 1 {
        sums[i] += du[i - 1].abs();
    }
    let anorm = largest(&sums);
    let (factors, info) = lapack::gttrf(dl, d, du);
    status.lapack_info = info as i64;
    if info > 0 {
        status.is_singular = true;
        return;
    }
    status.record_rcond(lapack::gtcon(&factors, anorm));
    lapack::gttrs(transposed, &factors, nrhs, b, n);
}

fn solve_cholesky<T: Real>(
    upper: bool,
    data: &mut [T],
    n: usize,
    nrhs: usize,
    b: &mut [T],
    status: &mut SliceStatus,
) {
    let anorm = symmetric_norm(!upper, data, n);
    let info = lapack::potf2(upper, n, data, n);
    status.lapack_info = info as i64;
    if info > 0 {
        status.is_singular = true;
        return;
    }
    status.record_rcond(lapack::pocon(upper, n, data, n, anorm));
    lapack::potrs(upper, n, nrhs, data, n, b, n);
}

fn solve_symmetric<T: Real>(
    upper: bool,
    data: &mut [T],
    n: usize,
    nrhs: usize,
    b: &mut [T],
    status: &mut SliceStatus,
) {
    let anorm = symmetric_norm(!upper, data, n);
    let (pivots, info) = lapack::sytf2(upper, n, data, n);
    status.lapack_info = info as i64;
    if info > 0 {
        status.is_singular = true;
        return;
    }
    status.record_rcond(lapack::sycon(upper, n, data, n, &pivots, anorm));
    lapack::sytrs(upper, n, nrhs, data, n, &pivots, b, n);
}

fn solve_general<T: Real>(
    data: &mut [T],
    n: usize,
    nrhs: usize,
    b: &mut [T],
    transposed: bool,
    status: &mut SliceStatus,
) {
    let anorm = row_sum_norm(data, n);
    let (pivots, info) = lapack::getf2(n, n, data, n);
    status.lapack_info = info as i64;
    if info > 0 {
        status.is_singular = true;
        return;
    }
    let condition = lapack::gecon(true, n, data, n, anorm);
    status.rcond = condition.rcond.to_f64();
    if condition.info >= 0 {
        status.record_rcond(condition.rcond);
        lapack::getrs(transposed, n, nrhs, data, n, &pivots, b, n);
    }
}

/// `A⁻¹` for `count` slices of `n × n` matrices, with the structures `solve` detects except
/// tridiagonal.
pub(super) fn inv<T: Real>(
    a: &[T],
    count: usize,
    n: usize,
    structure: Structure,
    lower: bool,
) -> (Vec<T>, Vec<SliceStatus>) {
    let mut state = Loop::new(structure, lower);
    let mut statuses = Vec::new();
    let mut result = vec![T::ZERO; count * n * n];
    for index in 0..count {
        let original = to_column_major(&a[index * n * n..(index + 1) * n * n], n, n);
        let mut data = original.clone();
        let mut slice_structure = state.structure;
        if slice_structure == Structure::Detect {
            slice_structure = detect(&data, n, false, &mut state.lower);
        }
        let mut status = SliceStatus::new(index, slice_structure);
        let upper = !state.lower;
        let stop = match slice_structure {
            Structure::Diagonal => {
                let (mut largest_value, mut largest_inverse) = (T::ZERO, T::ZERO);
                for j in 0..n {
                    let ajj = data[j * n + j];
                    status.is_singular = ajj == T::ZERO;
                    if status.is_singular {
                        status.lapack_info = j as i64;
                        break;
                    }
                    let inverse = T::ONE / ajj;
                    data[j * n + j] = inverse;
                    if ajj.abs() > largest_value {
                        largest_value = ajj.abs();
                    }
                    if inverse.abs() > largest_inverse {
                        largest_inverse = inverse.abs();
                    }
                }
                if !status.is_singular {
                    let condition = largest_value * largest_inverse;
                    status.is_ill_conditioned = condition > T::ONE / T::PRECISION;
                    status.rcond = condition.to_f64();
                }
                detect_problems(&status, &mut statuses)
            }
            Structure::UpperTriangular | Structure::LowerTriangular => {
                let info = lapack::trti2_checked(upper, false, n, &mut data, n);
                status.is_singular = info > 0;
                status.lapack_info = info as i64;
                status.record_rcond(lapack::trcon(upper, false, n, &data, n));
                let stop = detect_problems(&status, &mut statuses);
                zero_other_triangle(!upper, &mut data, n);
                stop
            }
            Structure::PositiveDefinite | Structure::Symmetric | Structure::Hermitian => {
                let mut use_symmetric = slice_structure != Structure::PositiveDefinite;
                if !use_symmetric {
                    invert_cholesky(upper, &mut data, n, &mut status);
                    if status.lapack_info == 0 || !status.is_singular {
                        if status.is_ill_conditioned {
                            statuses.push(status.clone());
                        }
                        fill_other_triangle(!upper, &mut data, n);
                    } else if state.posdef_fallback {
                        data = original;
                        status = SliceStatus::new(index, slice_structure);
                        use_symmetric = true;
                    } else {
                        statuses.push(status.clone());
                    }
                }
                if use_symmetric {
                    invert_symmetric(upper, &mut data, n, &mut status);
                    let stop = detect_problems(&status, &mut statuses);
                    fill_other_triangle(!upper, &mut data, n);
                    stop
                } else {
                    false
                }
            }
            Structure::General | Structure::Detect | Structure::Tridiagonal | Structure::Banded => {
                let anorm = row_sum_norm(&data, n);
                let (pivots, info) = lapack::getf2(n, n, &mut data, n);
                status.lapack_info = info as i64;
                if info > 0 {
                    status.is_singular = true;
                } else {
                    let condition = lapack::gecon(true, n, &data, n, anorm);
                    status.rcond = condition.rcond.to_f64();
                    if condition.info >= 0 {
                        status.record_rcond(condition.rcond);
                        status.is_singular = lapack::getri(n, &mut data, n, &pivots) > 0;
                    }
                }
                detect_problems(&status, &mut statuses)
            }
        };
        if stop {
            break;
        }
        result[index * n * n..(index + 1) * n * n].copy_from_slice(&to_row_major(&data, n, n));
    }
    (result, statuses)
}

fn invert_cholesky<T: Real>(upper: bool, data: &mut [T], n: usize, status: &mut SliceStatus) {
    let anorm = symmetric_norm(!upper, data, n);
    let info = lapack::potf2(upper, n, data, n);
    status.lapack_info = info as i64;
    if info > 0 {
        status.is_singular = true;
        return;
    }
    status.record_rcond(lapack::pocon(upper, n, data, n, anorm));
    status.is_singular = lapack::potri(upper, n, data, n) > 0;
}

fn invert_symmetric<T: Real>(upper: bool, data: &mut [T], n: usize, status: &mut SliceStatus) {
    let anorm = symmetric_norm(!upper, data, n);
    let (pivots, info) = lapack::sytf2(upper, n, data, n);
    status.lapack_info = info as i64;
    if info > 0 {
        status.is_singular = true;
        return;
    }
    status.record_rcond(lapack::sycon(upper, n, data, n, &pivots, anorm));
    status.is_singular = lapack::sytri(upper, n, data, n, &pivots) > 0;
}

/// One slice's LU factors in SciPy's `lu` layout: the permutation with `A[perm] = L U`, the
/// `m × k` unit lower triangular `L` (rows permuted back when `permute_l`), and the `k × n`
/// upper triangular `U`, all C-order.
pub(super) struct LuFactors<T> {
    pub perm: Vec<usize>,
    pub l: Vec<T>,
    pub u: Vec<T>,
}

/// `lu_decompose` for one C-order `m × n` slice.
pub(super) fn lu<T: Real>(a: &[T], m: usize, n: usize, permute_l: bool) -> LuFactors<T> {
    let k = m.min(n);
    let mut data = to_column_major(a, m, n);
    let (pivots, _) = lapack::getf2(m, n, &mut data, m);
    let mut u = vec![T::ZERO; k * n];
    for i in 0..k {
        for j in i..n {
            u[i * n + j] = data[j * m + i];
        }
    }
    let mut l = vec![T::ZERO; m * k];
    for j in 0..k {
        for i in j + 1..m {
            l[i * k + j] = data[j * m + i];
        }
        l[j * k + j] = T::ONE;
    }
    let mut perm = (0..m).collect::<Vec<_>>();
    for (i, pivot) in pivots.iter().enumerate().rev() {
        perm.swap(i, *pivot);
    }
    if permute_l {
        let original = l.clone();
        for (row, source) in perm.iter().enumerate() {
            l[row * k..(row + 1) * k].copy_from_slice(&original[source * k..(source + 1) * k]);
        }
    }
    LuFactors { perm, l, u }
}

/// `det_from_lu`: the determinant of a C-order `n × n` slice, accumulated in double
/// precision and rounded to `T` as SciPy does for `float32`. An exactly singular
/// factorization gives zero.
pub(super) fn det<T: Real>(a: &[T], n: usize) -> T {
    let mut data = to_column_major(a, n, n);
    let (pivots, info) = lapack::getf2(n, n, &mut data, n);
    if info > 0 {
        return T::ZERO;
    }
    let mut det = 1.0f64;
    let mut swaps = 0usize;
    for (k, pivot) in pivots.iter().enumerate() {
        det *= data[k * (n + 1)].to_f64();
        if *pivot != k {
            swaps += 1;
        }
    }
    T::from_f64(if swaps % 2 == 1 { -det } else { det })
}

/// `_cholesky` for one C-order slice: the factor in C order with zeros in the other
/// triangle, or `potrf`'s `info`.
pub(super) fn cholesky<T: Real>(a: &[T], n: usize, lower: bool) -> Result<Vec<T>, usize> {
    let mut data = vec![T::ZERO; n * n];
    for i in 0..n {
        let columns = if lower { 0..i + 1 } else { i..n };
        for j in columns {
            data[i * n + j] = a[i * n + j];
        }
    }
    // The C-order lower triangle is the column-major upper triangle, and vice versa.
    match lapack::potf2(lower, n, &mut data, n) {
        0 => Ok(data),
        info => Err(info),
    }
}

/// SciPy's QR modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QrMode {
    Full,
    R,
    Raw,
    Economic,
}

/// One slice's QR factorization in SciPy's layout: `Q` (`m × m` for full mode, `m × k` for
/// economic, or the raw `m × n` factors in raw mode), the `R` SciPy returns (`m × n` for full
/// and `r` modes, `k × n` otherwise), `tau` in raw mode, and the column permutation when
/// pivoting. All matrices are C order.
pub(super) struct QrFactors<T> {
    pub q: Option<Vec<T>>,
    pub r: Vec<T>,
    pub tau: Option<Vec<T>>,
    pub pivots: Option<Vec<usize>>,
}

/// `_qr` for one C-order `m × n` slice.
pub(super) fn qr<T: Real>(
    a: &[T],
    m: usize,
    n: usize,
    mode: QrMode,
    pivoting: bool,
) -> QrFactors<T> {
    let k = m.min(n);
    let mut data = to_column_major(a, m, n);
    let (tau, pivots) = if pivoting {
        let (tau, pivots) = lapack::geqp3(m, n, &mut data);
        (tau, Some(pivots))
    } else {
        (lapack::geqr2(m, n, &mut data), None)
    };
    let r_rows = match mode {
        QrMode::Full | QrMode::R => m,
        QrMode::Raw | QrMode::Economic => k,
    };
    let mut r = vec![T::ZERO; r_rows * n];
    for i in 0..k {
        for j in i..n {
            r[i * n + j] = data[j * m + i];
        }
    }
    let q = match mode {
        QrMode::R => None,
        QrMode::Raw => Some(to_row_major(&data, m, n)),
        QrMode::Full => Some(to_row_major(&lapack::org2r(m, m, &data, &tau), m, m)),
        QrMode::Economic => Some(to_row_major(&lapack::org2r(m, k, &data, &tau), m, k)),
    };
    QrFactors {
        q,
        r,
        tau: (mode == QrMode::Raw).then_some(tau),
        pivots,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_follows_scipy_including_the_tridiagonal_size_limit() {
        let mut lower = false;
        let upper_triangular = to_column_major(&[1.0, 2.0, 0.0, 3.0], 2, 2);
        assert_eq!(
            detect(&upper_triangular, 2, true, &mut lower),
            Structure::UpperTriangular
        );
        let symmetric = to_column_major(&[2.0, 1.0, 1.0, 2.0], 2, 2);
        assert_eq!(
            detect(&symmetric, 2, true, &mut lower),
            Structure::PositiveDefinite
        );
        // A 3 × 3 tridiagonal matrix is symmetric here, so it takes the Cholesky path; a 4 × 4
        // one is solved as tridiagonal.
        let tridiagonal = |n: usize| {
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                a[i * n + i] = 4.0;
                if i + 1 < n {
                    a[i * n + i + 1] = 1.0;
                    a[(i + 1) * n + i] = 1.0;
                }
            }
            a
        };
        assert_eq!(
            detect(&tridiagonal(3), 3, true, &mut lower),
            Structure::PositiveDefinite
        );
        assert_eq!(
            detect(&tridiagonal(4), 4, true, &mut lower),
            Structure::Tridiagonal
        );
        assert_eq!(
            detect(&tridiagonal(4), 4, false, &mut lower),
            Structure::PositiveDefinite
        );
    }

    #[test]
    fn an_indefinite_symmetric_matrix_falls_back_from_cholesky() {
        let a = [1.0, 2.0, 2.0, 1.0];
        let (x, statuses) = solve(&a, &[3.0, 3.0], 1, 2, 1, Structure::Detect, false, false);
        assert!(statuses.is_empty());
        assert_eq!(x, vec![1.0, 1.0]);
        // With an explicit 'pos' there is no fallback, and the slice is reported singular.
        let (_, statuses) = solve(
            &a,
            &[3.0, 3.0],
            1,
            2,
            1,
            Structure::PositiveDefinite,
            false,
            false,
        );
        assert_eq!(statuses.len(), 1);
        assert!(statuses[0].is_singular);
        assert_eq!(statuses[0].lapack_info, 2);
    }

    #[test]
    fn singular_and_ill_conditioned_slices_are_reported() {
        let singular = [1.0, 2.0, 2.0, 4.0, 1.0, 0.0, 0.0, 1.0];
        let (_, statuses) = solve(
            &singular,
            &[1.0, 1.0, 1.0, 1.0],
            2,
            2,
            1,
            Structure::General,
            false,
            false,
        );
        assert_eq!(statuses.len(), 1);
        assert!(statuses[0].is_singular);
        assert_eq!(statuses[0].lapack_info, 2);
        // A diagonal slice reports its condition number as `rcond`.
        let (x, statuses) = solve(
            &[1.0, 0.0, 0.0, 1e-20],
            &[1.0, 1.0],
            1,
            2,
            1,
            Structure::Detect,
            false,
            false,
        );
        assert_eq!(x, vec![1.0, 1e20]);
        assert!(statuses[0].is_ill_conditioned);
        assert_eq!(statuses[0].rcond, 1e20);
    }

    #[test]
    fn inverses_restore_the_unused_triangle() {
        let spd = [4.0, 2.0, 2.0, 3.0];
        let (inverse, statuses) = inv(&spd, 1, 2, Structure::Detect, false);
        assert!(statuses.is_empty());
        // SciPy 1.18.1 returns these values; potri's rounding leaves them a bit off the exact ones.
        assert_eq!(
            inverse,
            vec![
                0.375,
                -0.24999999999999994,
                -0.24999999999999994,
                0.4999999999999999
            ]
        );
        let (inverse, _) = inv(&[2.0, 1.0, 0.0, 4.0], 1, 2, Structure::Detect, false);
        assert_eq!(inverse, vec![0.5, -0.125, 0.0, 0.25]);
    }

    #[test]
    fn lu_permutes_rows_as_scipy_reports_them() {
        let factors = lu(&[1.0, 2.0, 3.0, 4.0], 2, 2, false);
        assert_eq!(factors.perm, vec![1, 0]);
        assert_eq!(factors.l, vec![1.0, 0.0, 1.0 / 3.0, 1.0]);
        assert_eq!(factors.u, vec![3.0, 4.0, 0.0, 2.0 - 4.0 / 3.0]);
        assert_eq!(det(&[1.0, 2.0, 3.0, 4.0], 2), -(3.0 * (2.0 - 4.0 / 3.0)));
    }

    #[test]
    fn bandwidth_handles_rectangular_matrices() {
        let a = [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 0.0],
            [3.0, 0.0, 1.0, 0.0],
        ];
        assert_eq!(bandwidth(3, 4, |r, c| a[r][c] != 0.0), (2, 3));
    }
}
